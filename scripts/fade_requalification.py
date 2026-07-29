"""
Measure the ranging-regime Bollinger fade sleeve under the Phase 2 risk model
and the Phase 4 tiers, for the Phase 6 keep/drop decision.

Runs the production run_backtest per symbol over the TUNING span only
(--start default 2023-07-27 -> end_of_data - WF_OOS_DAYS), so this measurement
cannot spend the one-shot OOS holdout. Nothing is swept: every parameter is
the frozen config default, which is why running on the tuning span consumes
no degrees of freedom.

Owns no screening logic — fade trades are whatever run_backtest produced with
pattern == meanrev.FADE_KIND, and the candidate funnel (block 4) re-calls
detect_fade_setups / check_breakout / build_fade_signal exactly as
engine.candidates_for does, so these numbers cannot disagree with the
backtest the decision is written from. A candidate's rr, even when REJECTED
by the RR floor, is read off a second build_fade_signal call with the floor
relaxed to -inf (the same technique export_bar_annotations.py uses for
rejection attribution) rather than re-derived here.

--start defaults to 2023-07-27, not BACKFILL_START: the 1D regime tier's
REGIME_MIN_BARS warmup is 207 CALENDAR DAYS, so no non-"uncertain" regime
label exists before ~that date and an earlier span reports a wall of
zero-trade months.

Usage:
    python scripts/fade_requalification.py
    python scripts/fade_requalification.py --start 2023-07-27 --symbol BTCUSDT
    python scripts/fade_requalification.py --symbol BTCUSDT --end 2024-01-01
"""

import argparse
import statistics
import sys
import time

import numpy as np

from trading_bot import config
from trading_bot.backtest.engine import _df, run_backtest
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.cli import _fmt, _print_metrics
from trading_bot.data import storage
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.regime.classifier import classify_series
from trading_bot.signals import meanrev
from trading_bot.signals.breakout import check_breakout

# 207 calendar days of 1D regime warmup from BACKFILL_START (config.py's
# REGIME_MIN_BARS comment) — the earliest date any non-"uncertain" label
# exists. Copied as a literal, not derived, per the plan's GOTCHA #1.
DEFAULT_START = "2023-07-27"

# Copied verbatim from backtest/walkforward.py's tune_end arithmetic — this
# script must never invent its own tuning/OOS split.
DAY_MS = 86_400_000


def _iso(ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


def _date_arg(value: str) -> int:
    try:
        return config.date_to_ms(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _median(xs) -> float | None:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return statistics.median(xs) if xs else None


def _quantiles(xs):
    """min / p25 / median / p75 / max, nearest-rank (no interpolation)."""
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x)))
    if not xs:
        return (None, None, None, None, None)

    def q(p: float) -> float:
        idx = min(len(xs) - 1, max(0, round(p * (len(xs) - 1))))
        return xs[idx]

    return (xs[0], q(0.25), q(0.5), q(0.75), xs[-1])


def _risk_conformance(fade_trades, df_setup) -> dict | None:
    """
    Median risk_pct, its ATR multiple, and the measured cost ratio c for one
    symbol's fade trades.

    ATR is indexed at each trade's entry_ts against the setup-tier ATR series,
    the same alignment run_backtest uses to size the (breakout) stop: entry is
    matched to the last setup bar closed at/before it via searchsorted.
    """
    if not fade_trades:
        return None

    close_setup = df_setup.index.to_numpy() + storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME]
    atr_vals = wilder_atr(df_setup, period=config.ATR_STOP_PERIOD).to_numpy()

    risk_pcts: list[float] = []
    atr_multiples: list[float] = []
    hold_days: list[float] = []
    excluded_nan_atr = 0

    for t in fade_trades:
        risk_pct = abs(t.entry - t.stop) / t.entry
        risk_pcts.append(risk_pct)
        hold_days.append((t.exit_ts - t.entry_ts) / DAY_MS)

        h_idx = int(np.searchsorted(close_setup, t.entry_ts, side="right")) - 1
        atr_value = float(atr_vals[h_idx]) if 0 <= h_idx < len(atr_vals) else float("nan")
        if np.isnan(atr_value) or atr_value <= 0:
            excluded_nan_atr += 1
            continue
        atr_multiples.append(abs(t.entry - t.stop) / atr_value)

    median_risk_pct = _median(risk_pcts)
    median_atr_multiple = _median(atr_multiples)
    median_hold_days = _median(hold_days) or 0.0
    round_trip_cost = 2 * (config.FEE_PCT + config.SLIPPAGE_PCT) + (
        config.FUNDING_PCT_PER_DAY * median_hold_days
    )
    c = round_trip_cost / median_risk_pct if median_risk_pct else None

    return {
        "median_risk_pct": median_risk_pct,
        "median_atr_multiple": median_atr_multiple,
        "excluded_nan_atr": excluded_nan_atr,
        "c": c,
    }


def _candidate_funnel(conn, symbol: str, start_ms: int, end_ms: int) -> dict:
    """
    Re-walk the setup/trigger tiers exactly as engine.candidates_for does,
    re-calling classify_series / detect_fade_setups / check_breakout /
    build_fade_signal, to produce the fade candidate funnel plus the rr and
    stretch_depth distributions of triggered candidates.

    Stage counts:
        setups    -- sum of len(detect_fade_setups(...)) over each distinct
                     ranging-regime SETUP bar evaluated (0-2 per bar; a
                     candidate persisting across several TRIGGER bars within
                     the same setup bar is counted once).
        triggered -- check_breakout fired for a candidate.
        passed_rr -- build_fade_signal (real config.RR_FLOOR) accepted it.
    "Actually traded" (stage 4) is NOT computed here — it comes from the
    Trade list run_backtest already produced (rank_signals ties and
    already-open trades are the engine's business, not this script's).
    """
    df_regime = _df(conn, symbol, config.REGIME_TIMEFRAME)
    df_setup = _df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME)
    df_trig = _df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME)
    empty = {
        "setups": 0,
        "triggered": 0,
        "passed_rr": 0,
        "rr_accepted": [],
        "rr_rejected": [],
        "stretch_accepted": [],
        "stretch_rejected": [],
    }
    if df_regime.empty or df_setup.empty or df_trig.empty:
        return empty

    regime_ms = storage.TIMEFRAME_MS[config.REGIME_TIMEFRAME]
    setup_ms = storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME]
    trigger_ms = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]

    labels = classify_series(df_regime)
    close_regime = df_regime.index.to_numpy() + regime_ms
    close_setup = df_setup.index.to_numpy() + setup_ms
    ts_trig = df_trig.index.to_numpy()
    close_trig = ts_trig + trigger_ms

    def regime_at(t: int) -> str:
        k = int(np.searchsorted(close_regime, t, side="right")) - 1
        return str(labels.iloc[k]) if k >= 0 else "uncertain"

    cand_cache: dict[int, list] = {}

    def candidates_for(h_idx: int) -> list:
        if h_idx not in cand_cache:
            t = int(close_setup[h_idx])
            reg = regime_at(t)
            window = df_setup.iloc[max(0, h_idx + 1 - config.PATTERN_LOOKBACK_BARS) : h_idx + 1]
            cand_cache[h_idx] = meanrev.detect_fade_setups(window) if reg == "ranging" else []
        return cand_cache[h_idx]

    setups_seen = 0
    counted_setup_bars: set[int] = set()
    triggered = 0
    passed_rr = 0
    rr_accepted: list[float] = []
    rr_rejected: list[float] = []
    stretch_accepted: list[float] = []
    stretch_rejected: list[float] = []

    for j in range(len(ts_trig)):
        bc = int(close_trig[j])
        if bc < start_ms or bc > end_ms or j < 1:
            continue
        h_idx = int(np.searchsorted(close_setup, bc, side="right")) - 1
        if h_idx < 0:
            continue
        cands = candidates_for(h_idx)
        if h_idx not in counted_setup_bars:
            counted_setup_bars.add(h_idx)
            setups_seen += len(cands)
        if not cands:
            continue

        window_trig = df_trig.iloc[max(0, j - (config.VOLUME_LOOKBACK + 1)) : j + 1]
        for cand in cands:
            trig_cand = meanrev._to_trigger_candidate(cand)
            event = check_breakout(
                window_trig, trig_cand, lookback_bars=1, interval_ms=trigger_ms
            )
            if event is None:
                continue
            triggered += 1
            stretch_depth = abs(cand.stop_level - cand.trigger_level) / cand.trigger_level

            # Relaxed floor isolates geometry from the RR screen (same
            # technique as export_bar_annotations.py's _rejection_reason):
            # a Signal built here still carries the production-computed rr
            # even for a candidate the real floor would reject.
            relaxed = meanrev.build_fade_signal(symbol, cand, event, rr_floor=float("-inf"))
            if relaxed is None:
                continue  # geometry-invalid (non-positive risk/reward); no rr to report
            real = meanrev.build_fade_signal(symbol, cand, event)
            if real is not None:
                passed_rr += 1
                rr_accepted.append(relaxed.rr)
                stretch_accepted.append(stretch_depth)
            else:
                rr_rejected.append(relaxed.rr)
                stretch_rejected.append(stretch_depth)

    return {
        "setups": setups_seen,
        "triggered": triggered,
        "passed_rr": passed_rr,
        "rr_accepted": rr_accepted,
        "rr_rejected": rr_rejected,
        "stretch_accepted": stretch_accepted,
        "stretch_rejected": stretch_rejected,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--symbol",
        action="append",
        help="Symbol to measure (repeatable, e.g. --symbol BTCUSDT); default is all "
        "configured symbols",
    )
    ap.add_argument(
        "--start",
        type=_date_arg,
        default=config.date_to_ms(DEFAULT_START),
        help=f"UTC start date YYYY-MM-DD (default: {DEFAULT_START} -- the earliest date "
        "the 1D regime tier's 207-calendar-day warmup produces a non-'uncertain' label; "
        "anything earlier reports a wall of zero-trade months)",
    )
    ap.add_argument(
        "--end",
        type=_date_arg,
        default=None,
        help="UTC end date YYYY-MM-DD (default: the tuning-span end, "
        "end_of_data - WF_OOS_DAYS). Exits 2 without printing metrics if this "
        "would read into the OOS holdout.",
    )
    ap.add_argument("--db", default=config.DB_PATH, help="Path to SQLite database")
    args = ap.parse_args()

    symbols = args.symbol if args.symbol else list(config.SYMBOLS)

    full_end_ms = int(time.time() * 1000)
    oos_ms = config.WF_OOS_DAYS * DAY_MS
    tune_end = full_end_ms - oos_ms
    start_ms = args.start
    end_ms = args.end if args.end is not None else tune_end

    print("=" * 78)
    print("Fade re-qualification span (Block 1)")
    print(f"  start:        {_iso(start_ms)}")
    print(f"  tune_end:     {_iso(tune_end)}  (WF_OOS_DAYS={config.WF_OOS_DAYS})")
    print(f"  measured end: {_iso(end_ms)}")
    print(f"  OOS holdout NOT touched: {_iso(tune_end)} -> {_iso(full_end_ms)}")
    print("=" * 78)

    if end_ms > tune_end:
        print(
            f"ERROR: --end {_iso(end_ms)} exceeds tune_end {_iso(tune_end)}; that would "
            "read the one-shot OOS holdout this phase must not touch. Refusing to run.",
            file=sys.stderr,
        )
        return 2

    conn = storage.connect(args.db)

    per_symbol_trades: dict[str, list] = {}
    all_fade_trades: list = []
    for symbol in symbols:
        trades = run_backtest(conn, symbol, start_ms=start_ms, end_ms=end_ms)
        fade_trades = [t for t in trades if t.pattern == meanrev.FADE_KIND]
        per_symbol_trades[symbol] = fade_trades
        all_fade_trades.extend(fade_trades)

    print()
    print("Block 2: per-symbol fade metrics + pooled")
    for symbol in symbols:
        print(f"{symbol}:")
        _print_metrics(compute_metrics(per_symbol_trades[symbol]), indent="  ")
    print("POOLED:")
    _print_metrics(compute_metrics(all_fade_trades), indent="  ")

    print()
    print("Block 3: risk-model conformance")
    for symbol in symbols:
        df_setup = _df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME)
        conf = _risk_conformance(per_symbol_trades[symbol], df_setup)
        if conf is None:
            print(f"{symbol}: no fade trades -- median_risk_pct=-- atr_multiple=-- c=--")
            continue
        c = conf["c"]
        verdict = "PASS" if c is not None and c <= config.COST_RATIO_CEILING else "FAIL"
        print(
            f"{symbol}: median_risk_pct={_fmt(conf['median_risk_pct'], '.4%')}  "
            f"median_atr_multiple={_fmt(conf['median_atr_multiple'], '.2f')}x  "
            f"c={_fmt(c, '.4f')}  ceiling={config.COST_RATIO_CEILING:.4f}  [{verdict}]  "
            f"(excluded_nan_atr={conf['excluded_nan_atr']})"
        )

    print()
    print("Block 4: candidate funnel")
    funnels: dict[str, dict] = {}
    for symbol in symbols:
        f = _candidate_funnel(conn, symbol, start_ms, end_ms)
        funnels[symbol] = f
        traded = len(per_symbol_trades[symbol])
        lo, p25, med, p75, hi = _quantiles(f["rr_accepted"] + f["rr_rejected"])
        print(
            f"{symbol}: setups={f['setups']}  triggered={f['triggered']}  "
            f"passed_rr={f['passed_rr']}  traded={traded}"
        )
        print(
            f"    rr distribution (triggered, accepted+rejected): "
            f"min={_fmt(lo, '.2f')} p25={_fmt(p25, '.2f')} median={_fmt(med, '.2f')} "
            f"p75={_fmt(p75, '.2f')} max={_fmt(hi, '.2f')}"
        )

    print()
    print("Block 5: adverse-selection check (median stretch_depth, RR-accepted vs RR-rejected)")
    for symbol in symbols:
        f = funnels[symbol]
        med_acc = _median(f["stretch_accepted"])
        med_rej = _median(f["stretch_rejected"])
        print(
            f"{symbol}: accepted={_fmt(med_acc, '.4%')} (n={len(f['stretch_accepted'])})  "
            f"rejected={_fmt(med_rej, '.4%')} (n={len(f['stretch_rejected'])})"
        )

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
