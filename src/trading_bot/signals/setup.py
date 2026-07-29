"""
Signal assembly: regime gate, SL/TP computation, and R:R band screening.

Pipeline (read-only consumer of the Phase 1 store, like the regime classifier):

  1. Regime gate — this method is active ONLY in the "trending" regime; any
     other label yields no signals (Phase 4's mean-reversion method owns
     "ranging"; extreme-volatility and uncertain suppress everything).
  2. Pattern scan — pivots + geometric detectors on the last
     PATTERN_LOOKBACK_BARS closed SETUP bars.
  3. Breakout trigger — latest closed TRIGGER bar must freshly cross a candidate's
     level (see breakout.py).
  4. SL/TP — stop is ATR_STOP_MULTIPLE * ATR(setup timeframe) away from entry
     (a volatility-derived, market-structure distance); target at the
     classical measured move (level +/- pattern height).
  5. R:R ratio screen (Phase 2): reward_pct / risk_pct must be >= RR_FLOOR.

Volume ratio rides along on the Signal for Phase 6 confidence calibration;
low volume never blocks a signal here.
"""

import logging
import time
from dataclasses import dataclass

import pandas as pd

from trading_bot import config
from trading_bot.data import storage
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.regime.classifier import current_regime
from trading_bot.risk.atr_stop import compute_atr_stop
from trading_bot.signals.breakout import BreakoutEvent, check_breakout
from trading_bot.signals.patterns import PatternCandidate, detect_patterns
from trading_bot.signals.pivots import find_pivots

logger = logging.getLogger("trading_bot")


@dataclass(frozen=True)
class Signal:
    """A fully-screened trade setup ready for alerting (Phase 7).

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT").
        ts: Epoch-ms of the trigger bar.
        direction: "long" or "short".
        pattern: Pattern kind that produced the setup.
        entry: Trigger bar close.
        stop: Stop-loss price (the broken breakout level).
        target: Take-profit price (measured-move projection).
        risk_pct: |entry - stop| / entry.
        reward_pct: |target - entry| / entry.
        rr: reward_pct / risk_pct.
        volume_ratio: Breakout volume vs rolling average (NaN if undefined).
        volume_high: Whether breakout volume was notably high.
    """

    symbol: str
    ts: int
    direction: str
    pattern: str
    entry: float
    stop: float
    target: float
    risk_pct: float
    reward_pct: float
    rr: float
    volume_ratio: float
    volume_high: bool


def build_signal(
    symbol: str,
    candidate: PatternCandidate,
    event: BreakoutEvent,
    atr_value: float,
    *,
    atr_multiple: float | None = None,
    rr_floor: float | None = None,
) -> Signal | None:
    """
    Compute an ATR-scaled SL/TP for a triggered candidate and apply the R:R
    ratio floor.

    Stop sits atr_multiple * ATR away from entry (a volatility-derived,
    market-structure distance — see risk/atr_stop.py), replacing the old
    level-anchored buffer stop. Target is unchanged: the measured move
    (level +/- candidate.target_height). Setups are rejected when ATR is
    undefined (NaN, e.g. warmup) or non-positive, risk is zero or negative,
    reward is non-positive, or the reward:risk ratio falls below rr_floor.
    There is no longer an absolute risk-percentage cap nor an entry-extension
    cap — those were the adverse-selection mechanism identified in the
    market-research benchmark (rejects the trades whose stops are freakishly
    tight, which are exactly the trades noise destroys).

    Args:
        symbol: Trading pair symbol.
        candidate: The pattern that broke out.
        event: The trigger-bar breakout event (entry reference).
        atr_value: Current ATR on the setup timeframe (caller computes). NaN
            or non-positive values are rejected here directly — never
            propagated into a stop — so a missing/warmup ATR is safe to pass
            through without a caller-side check.
        atr_multiple: Stop distance = atr_multiple * atr_value
            (default config.ATR_STOP_MULTIPLE).
        rr_floor: Minimum acceptable reward_pct/risk_pct
            (default config.RR_FLOOR).

    Returns:
        Signal if the setup passes, else None.
    """
    if atr_multiple is None:
        atr_multiple = config.ATR_STOP_MULTIPLE
    if rr_floor is None:
        rr_floor = config.RR_FLOOR

    def reject(reason: str, *args) -> None:
        logger.debug(
            "%s %s %s rejected: " + reason,
            symbol,
            candidate.kind,
            candidate.direction,
            *args,
        )

    entry = event.price
    if event.level <= 0 or entry <= 0:
        return None

    # NaN comparisons are always False, so `atr_value <= 0` alone would let a
    # NaN ATR (Wilder warmup) silently produce a stop = nan Signal instead of
    # being rejected. `not (atr_value > 0)` catches NaN, zero, and negative.
    if not (atr_value > 0):
        reject("atr_value %s is undefined or non-positive", atr_value)
        return None

    stop = compute_atr_stop(entry, candidate.direction, atr_value, atr_multiple)
    if candidate.direction == "long":
        target = event.level + candidate.target_height
        reward = target - entry
    else:
        target = event.level - candidate.target_height
        reward = entry - target

    risk = abs(entry - stop)
    if risk <= 0 or reward <= 0:
        return None

    risk_pct = risk / entry
    reward_pct = reward / entry
    rr = reward_pct / risk_pct

    if rr < rr_floor:
        reject("rr %.2f below floor %.2f", rr, rr_floor)
        return None

    return Signal(
        symbol=symbol,
        ts=event.ts,
        direction=candidate.direction,
        pattern=candidate.kind,
        entry=entry,
        stop=stop,
        target=target,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr=rr,
        volume_ratio=event.volume_ratio,
        volume_high=event.volume_high,
    )


def _contiguous_tail(df: pd.DataFrame, interval_ms: int, symbol: str, timeframe: str) -> pd.DataFrame:
    """Trim to the longest run of evenly-spaced bars ending at the latest bar.

    Pattern geometry is positional: a missing bar silently compresses pivot
    spacing and pattern width, and makes the bar before a crossing a stale
    reference. Rather than compute on a distorted series, drop everything before
    the most recent gap and log it — a shorter clean window is honest, a long
    dirty one is not.
    """
    if len(df) < 2:
        return df
    ts = df.index.to_numpy()
    gaps = (ts[1:] - ts[:-1]) != interval_ms
    if not gaps.any():
        return df
    # Start just after the last gap.
    start = int(gaps.nonzero()[0][-1]) + 1
    logger.warning(
        "%s %s has gaps; using the %d contiguous bars after ts=%d (dropped %d)",
        symbol,
        timeframe,
        len(df) - start,
        int(ts[start - 1]),
        start,
    )
    return df.iloc[start:]


def _load_df(conn, symbol: str, timeframe: str, now_ms: int) -> pd.DataFrame:
    """Load closed bars for symbol/timeframe as an OHLCV DataFrame.

    Mirrors current_regime's closed-bar rule: a bar is closed when
    ts + interval <= now_ms, enforced via load_candles' inclusive end_ms.
    Trimmed to a contiguous tail so downstream positional geometry is valid.
    """
    interval = storage.TIMEFRAME_MS[timeframe]
    rows = storage.load_candles(conn, symbol, timeframe, end_ms=now_ms - interval)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if len(df):
        df["ts"] = df["ts"].astype(int)
    df = df.set_index("ts")
    return _contiguous_tail(df, interval, symbol, timeframe)


def rank_signals(signals: list[Signal]) -> list[Signal]:
    """Order signals best-first: highest R:R, then pattern kind for determinism.

    The single shared ranking rule for live scanning and backtesting. The
    backtest can only act on one signal per bar, so if it picked by any other
    order (the candidate list's alphabetical sort, say) its measured performance
    would describe a name-biased subset rather than the method as it behaves
    live.
    """
    return sorted(signals, key=lambda s: (-s.rr, s.pattern, s.direction))


def scan_breakout_signals(conn, symbol: str, now_ms: int) -> list[Signal]:
    """
    RETIRED (PRD Phase 5): chart-pattern geometry has left the active dispatch
    path — signals.scan now routes "trending" to signals.donchian. This function
    and signals/patterns.py + signals/pivots.py are kept, unreferenced by the
    dispatcher, as the reference implementation behind the retirement decision;
    they are NOT deleted and their tests stay green. Do not re-register them.

    Breakout-scan one symbol WITHOUT a regime check (dispatcher applies the gate).

    Args:
        conn: Database connection.
        symbol: Trading pair symbol.
        now_ms: Evaluation time in epoch milliseconds.

    Returns:
        Screened breakout Signals ordered best-R:R-first (possibly empty). At
        most one per (pattern, direction), since detect_patterns deduplicates
        overlapping geometry of the same type.
    """
    df_setup = _load_df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME, now_ms)
    df_trig = _load_df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME, now_ms)
    if df_setup.empty or df_trig.empty:
        return []

    df_setup = df_setup.tail(config.PATTERN_LOOKBACK_BARS)
    # Trigger needs the volume window, the crossing pair, and any extra bars the
    # configured trigger lookback may reach back over. All counts are in
    # TRIGGER-timeframe bars and are therefore tier-independent.
    trigger_bars = config.VOLUME_LOOKBACK + 1 + max(1, config.BREAKOUT_TRIGGER_LOOKBACK_BARS)
    df_trig = df_trig.tail(trigger_bars)

    trigger_interval = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
    latest_close = int(df_trig.index[-1]) + trigger_interval
    if now_ms - latest_close >= trigger_interval:
        logger.warning(
            "%s %s data is %d ms behind now_ms; a breakout may already be older than "
            "the %d-bar trigger window",
            symbol,
            config.SIGNAL_TRIGGER_TIMEFRAME,
            now_ms - latest_close,
            config.BREAKOUT_TRIGGER_LOOKBACK_BARS,
        )

    atr_series = wilder_atr(df_setup, period=config.ATR_STOP_PERIOD)
    atr_value = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")

    pivots = find_pivots(df_setup)
    candidates = detect_patterns(df_setup, pivots)

    signals: list[Signal] = []
    if not (atr_value > 0):  # covers NaN (warmup) and zero/negative
        logger.warning(
            "%s: ATR undefined or non-positive on %s bars; no signals",
            symbol,
            config.SIGNAL_PATTERN_TIMEFRAME,
        )
        return []
    for candidate in candidates:
        event = check_breakout(df_trig, candidate, interval_ms=trigger_interval)
        if event is None:
            continue
        signal = build_signal(symbol, candidate, event, atr_value)
        if signal is not None:
            signals.append(signal)

    return rank_signals(signals)


def current_signals(
    conn,
    symbol: str,
    *,
    now_ms: int | None = None,
) -> tuple[str, list[Signal]]:
    """
    RETIRED (PRD Phase 5): chart-pattern geometry has left the active dispatch
    path — signals.scan now routes "trending" to signals.donchian. This function
    and signals/patterns.py + signals/pivots.py are kept, unreferenced by the
    dispatcher, as the reference implementation behind the retirement decision;
    they are NOT deleted and their tests stay green. Do not re-register them.

    Scan one symbol for chart-pattern breakout signals as of now_ms.

    Args:
        conn: Database connection.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        now_ms: Evaluation time in epoch milliseconds (default: current time).

    Returns:
        Tuple of (regime_label, signals). regime_label is always the symbol's
        current regime-timeframe regime; signals is empty unless it is "trending"
        and at least one triggered candidate passed the R:R band.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    regime_label, _, _ = current_regime(conn, symbol, now_ms=now_ms)
    if regime_label != "trending":
        return (regime_label, [])

    return (regime_label, scan_breakout_signals(conn, symbol, now_ms))
