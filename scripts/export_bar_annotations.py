"""
Export per-bar pipeline annotations for the interactive review chart.

Replays the Phase 2-4 pipeline over stored history and emits, for EVERY trigger
trigger bar, the raw OHLCV plus what the strategy saw at that bar: the 4H
regime label, how many pattern/fade candidates were live, and every breakout
trigger that fired — whether it was taken, rejected by the R:R floor, or
skipped because a trade was already open.

Trades come from run_backtest itself, so the annotations cannot disagree with
the backtest the reports are built on. Rejection reasons are derived by
re-calling the production builders with the R:R floor relaxed rather than
reimplementing the screen, so this script owns no screening logic of its own.

Unlike the engine's lazy cache, candidates are computed at every 1H bar, so
breakouts that the engine never evaluated (because a trade was open) still
show up — those are the "missing breakouts" a reviewer is looking for.

Usage:
    python scripts/export_bar_annotations.py --out annotations.json
"""

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from trading_bot import config
from trading_bot.backtest.engine import run_backtest
from trading_bot.data import storage
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.regime.classifier import classify_series
from trading_bot.risk.atr_stop import net_rr
from trading_bot.signals.breakout import check_breakout
from trading_bot.signals.meanrev import (
    _to_trigger_candidate,
    build_fade_signal,
    detect_fade_setups,
)
from trading_bot.signals.patterns import detect_patterns
from trading_bot.signals.setup import build_signal

logger = logging.getLogger("trading_bot")

# Regime label -> compact code carried in the per-bar array.
REGIME_CODES = {"uncertain": 0, "trending": 1, "ranging": 2, "extreme-volatility": 3}

# An RR floor loose enough that the R:R screen cannot be the binding
# constraint, used only to attribute a rejection to the ratio floor vs geometry.
# Relaxing the floor to -inf, not 0.0: the fade builder gates on the
# COST-ADJUSTED ratio, which goes negative when reward cannot cover cost. A
# floor of 0.0 would still reject those, misattributing them to geometry.
RELAXED_RR_FLOOR = float("-inf")


def _df(conn, symbol: str, timeframe: str) -> pd.DataFrame:
    rows = storage.load_candles(conn, symbol, timeframe)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if len(df):
        df["ts"] = df["ts"].astype(int)
    return df.set_index("ts")


def _build(method: str, symbol: str, cand, event, atr_value: float, **kw):
    """Call the production signal builder for the candidate's method."""
    if method == "breakout":
        return build_signal(symbol, cand, event, atr_value, **kw)
    return build_fade_signal(symbol, cand, event, **kw)


def _rejection_reason(method: str, symbol: str, cand, event, atr_value: float) -> str:
    """
    Attribute a build_* rejection to the R:R floor or to setup geometry.

    There is exactly one tunable filter (reward:risk >= rr_floor), so re-running
    the same production builder with the floor fully relaxed isolates it: if
    that makes the setup pass, the floor was the binding constraint at the real
    threshold; if it still fails, the rejection came from geometry (unsizeable
    or non-positive stop/reward, or entry already at/through the target).

    Fades get a third bucket. Their gate is the COST-ADJUSTED ratio, so a
    non-positive net ratio means reward does not even cover round-trip cost —
    a guaranteed loser rather than a merely thin one. That distinction is the
    whole point of the net ratio, so the chart should not bury it under the
    generic "below floor" label.
    """
    relaxed = _build(method, symbol, cand, event, atr_value, rr_floor=RELAXED_RR_FLOOR)
    if relaxed is None:
        return "geometry"
    if method == "fade":
        nr = net_rr(
            relaxed.reward_pct, relaxed.risk_pct, config.FEE_PCT, config.SLIPPAGE_PCT
        )
        if nr <= 0:
            return "reward-below-cost"
    return "rr-below-floor"


def annotate_symbol(conn, symbol: str) -> dict:
    """
    Replay the pipeline over one symbol and return its chart payload.

    Returns:
        Dict with parallel per-bar arrays (ts/o/h/l/c/v/regime/ncand), a sparse
        event list, and the trade list, all in trigger-bar terms.
    """
    df_4h = _df(conn, symbol, config.REGIME_TIMEFRAME)
    df_1h = _df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME)
    df_15m = _df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME)
    if df_4h.empty or df_1h.empty or df_15m.empty:
        return {}

    logger.info("%s: replaying backtest", symbol)
    trades = run_backtest(conn, symbol)

    labels = classify_series(df_4h)
    close_4h = df_4h.index.to_numpy() + storage.TIMEFRAME_MS[config.REGIME_TIMEFRAME]
    close_1h = df_1h.index.to_numpy() + storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME]
    atr_close_1h = wilder_atr(df_1h, period=config.ATR_STOP_PERIOD).to_numpy()
    ts_15m = df_15m.index.to_numpy()
    close_15m = ts_15m + storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]

    # Bar index of each trade's entry and exit, for marker placement and for
    # telling a "would have fired but we were busy" trigger from a taken one.
    idx_of_ts = {int(t): i for i, t in enumerate(ts_15m)}
    entry_idx = {}
    busy_until = np.zeros(len(ts_15m), dtype=bool)
    trade_rows = []
    for tr in trades:
        i = idx_of_ts.get(tr.entry_ts)
        j = idx_of_ts.get(tr.exit_ts)
        if i is None:
            continue
        entry_idx[i] = len(trade_rows)
        if j is not None:
            busy_until[i + 1 : j + 1] = True
        trade_rows.append(
            {
                "i": i,
                "exit_i": j,
                "regime": tr.regime,
                "pattern": tr.pattern,
                "direction": tr.direction,
                "entry": tr.entry,
                "stop": tr.stop,
                "target": tr.target,
                "exit_price": tr.exit_price,
                "outcome": tr.outcome,
                "pnl_pct": tr.pnl_pct,
                "volume_high": bool(tr.volume_high),
            }
        )

    def regime_at(t: int) -> str:
        k = int(np.searchsorted(close_4h, t, side="right")) - 1
        return str(labels.iloc[k]) if k >= 0 else "uncertain"

    cand_cache: dict[int, tuple[str, list]] = {}

    def candidates_for(h_idx: int) -> tuple[str, list]:
        """Candidates live at a 1H bar, same windowing the engine uses."""
        if h_idx not in cand_cache:
            reg = regime_at(int(close_1h[h_idx]))
            window = df_1h.iloc[max(0, h_idx + 1 - config.PATTERN_LOOKBACK_BARS) : h_idx + 1]
            cands: list = []
            if reg == "trending":
                cands = [("breakout", c) for c in detect_patterns(window)]
            elif reg == "ranging":
                cands = [("fade", c) for c in detect_fade_setups(window)]
            cand_cache[h_idx] = (reg, cands)
        return cand_cache[h_idx]

    regimes = np.zeros(len(ts_15m), dtype=np.uint8)
    ncand = np.zeros(len(ts_15m), dtype=np.uint8)
    events: list[dict] = []
    claimed: set[int] = set()  # entry bars already attributed to their trade

    logger.info(
        "%s: annotating %d %s bars",
        symbol,
        len(ts_15m),
        config.SIGNAL_TRIGGER_TIMEFRAME,
    )
    for j in range(len(ts_15m)):
        bc = int(close_15m[j])
        h_idx = int(np.searchsorted(close_1h, bc, side="right")) - 1
        if h_idx < 0:
            continue
        reg, cands = candidates_for(h_idx)
        regimes[j] = REGIME_CODES.get(reg, 0)
        ncand[j] = min(len(cands), 255)
        if not cands or j < 1:
            continue

        window_15m = df_15m.iloc[max(0, j - (config.VOLUME_LOOKBACK + 1)) : j + 1]
        atr_value = float(atr_close_1h[h_idx])
        for method, cand in cands:
            trig = cand if method == "breakout" else _to_trigger_candidate(cand)
            event = check_breakout(window_15m, trig)
            if event is None:
                continue
            sig = _build(method, symbol, cand, event, atr_value)
            row = {
                "i": j,
                "method": method,
                "pattern": getattr(cand, "kind", "bollinger-fade"),
                "direction": cand.direction,
                "level": event.level,
                "entry": event.price,
                "volume_ratio": None if np.isnan(event.volume_ratio) else event.volume_ratio,
                "volume_high": bool(event.volume_high),
            }
            if sig is None:
                row["type"] = "rejected"
                row["reason"] = _rejection_reason(method, symbol, cand, event, atr_value)
            else:
                row.update(
                    {
                        "stop": sig.stop,
                        "target": sig.target,
                        "risk_pct": sig.risk_pct,
                        "reward_pct": sig.reward_pct,
                        "rr": sig.rr,
                    }
                )
                if j in entry_idx and j not in claimed:
                    # First passing candidate at this bar is the one the engine
                    # filled; the engine breaks out of the candidate loop here.
                    claimed.add(j)
                    row["type"] = "taken"
                    row["trade"] = entry_idx[j]
                elif busy_until[j]:
                    row["type"] = "busy"
                    row["reason"] = "trade-already-open"
                else:
                    # Passed the screen, no trade open, yet the engine did not
                    # enter: it had already entered on an earlier candidate at
                    # this same bar (engine breaks on the first signal).
                    row["type"] = "superseded"
                    row["reason"] = "another-candidate-filled-this-bar"
            events.append(row)

    logger.info(
        "%s: %d triggers (%d taken, %d rejected, %d busy), %d trades",
        symbol,
        len(events),
        sum(1 for e in events if e["type"] == "taken"),
        sum(1 for e in events if e["type"] == "rejected"),
        sum(1 for e in events if e["type"] == "busy"),
        len(trade_rows),
    )

    return {
        "ts": [int(t) // 1000 for t in ts_15m],
        "o": [round(float(x), 4) for x in df_15m["open"]],
        "h": [round(float(x), 4) for x in df_15m["high"]],
        "l": [round(float(x), 4) for x in df_15m["low"]],
        "c": [round(float(x), 4) for x in df_15m["close"]],
        "v": [round(float(x), 3) for x in df_15m["volume"]],
        "regime": regimes.tolist(),
        "ncand": ncand.tolist(),
        "events": events,
        "trades": trade_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(config.SYMBOLS))
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--db", default=config.DB_PATH)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = storage.connect(args.db)
    payload = {
        "generated_ms": int(time.time() * 1000),
        "trigger_timeframe": config.SIGNAL_TRIGGER_TIMEFRAME,
        "pattern_timeframe": config.SIGNAL_PATTERN_TIMEFRAME,
        "regime_timeframe": config.REGIME_TIMEFRAME,
        "params": {
            "MAX_RISK_PCT": config.MAX_RISK_PCT,
            "ATR_STOP_PERIOD": config.ATR_STOP_PERIOD,
            "ATR_STOP_MULTIPLE": config.ATR_STOP_MULTIPLE,
            "RR_FLOOR": config.RR_FLOOR,
            "ADX_TREND_THRESHOLD": config.ADX_TREND_THRESHOLD,
            "VOLUME_HIGH_RATIO": config.VOLUME_HIGH_RATIO,
            "FEE_PCT": config.FEE_PCT,
            "SLIPPAGE_PCT": config.SLIPPAGE_PCT,
            "FUNDING_PCT_PER_DAY": config.FUNDING_PCT_PER_DAY,
            "COST_RATIO_CEILING": config.COST_RATIO_CEILING,
            "MAX_HOLD_BARS_TRIGGER": config.MAX_HOLD_BARS_TRIGGER,
        },
        "symbols": {},
    }
    for symbol in args.symbols.split(","):
        symbol = symbol.strip()
        if not symbol:
            continue
        data = annotate_symbol(conn, symbol)
        if data:
            payload["symbols"][symbol] = data

    with open(args.out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
