"""
Donchian channel breakout signal method (Phase 5, trending regime only).

Replaces the retired Phase 3 chart-pattern geometry. Structural mirror of
meanrev.py: the setup is detected on the SETUP tier and wrapped in a
PatternCandidate so breakout.check_breakout supplies the trigger-tier
crossing/freshness/volume logic unchanged.

  1. Regime gate — "trending" only (the dispatcher applies it).
  2. Setup on config.SIGNAL_PATTERN_TIMEFRAME (4H after the Phase 4 tier
     shift): the trailing DONCHIAN_ENTRY_PERIOD (20) channel extreme is the
     level a trigger bar must close beyond, with two canonical confirmations
     on the same tier — ADX(ADX_PERIOD) >= ADX_TREND_THRESHOLD (25), the same
     Wilder ADX the 1D regime classifier uses applied to the finer tier; and
     the latest close on the correct side of the DONCHIAN_TREND_PERIOD (55)
     MID-LINE (long above, short below).
  3. Trigger — check_breakout on the trigger tier.
  4. SL/TP — delegated wholly to setup.build_signal: stop =
     ATR_STOP_MULTIPLE * ATR(setup TF) (Phase 2); target = level +/-
     target_height, where target_height is the 20-bar CHANNEL WIDTH
     (upper - lower) — a derived quantity, the range the market has just
     resolved, adding no parameter beyond the canonical 20.
  5. R:R ratio floor — build_signal's rr >= RR_FLOOR (Phase 2).

Stated Assumptions (from the plan, reproduced here so the decision survives
without it):

A1 — Channels compute on the 4H SETUP tier (config.SIGNAL_PATTERN_TIMEFRAME),
not 1D: the stop is 1.5 * ATR(setup TF), so computing the channel on a
different tier than ATR would put stop and target on different volatility
scales; 1D bars do not exist in the DB (as of this phase).

A2 — "20/55 entry/exit channels" is read as: 20 = entry AND exit channel, 55
= trend filter (not the Turtle-System-2 reading of 55 entry / 20 exit, which
would make the 55-mid filter tautological — for a long, the 55-mid is always
below a fresh 55-bar high). The alternative reading is pre-registered as a
legitimate Phase 7 grid axis, not explored here.

Exits (opposite-channel touch, ATR ratchet trail) are not expressible in the
frozen Signal dataclass and are enforced by the backtest engine; see
channel_exit_levels() and backtest/engine.py.

No geometry parameters exist here. 20 and 55 are canonical and frozen; ADX 25
and the ATR multiple are reused from existing config, never redeclared.
"""

import logging

import pandas as pd

from trading_bot import config
from trading_bot.data import storage
from trading_bot.indicators.donchian import donchian
from trading_bot.indicators.wilder import adx as wilder_adx
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.signals.breakout import check_breakout
from trading_bot.signals.patterns import PatternCandidate
from trading_bot.signals.setup import Signal, _load_df, build_signal, rank_signals

logger = logging.getLogger("trading_bot")

DONCHIAN_KIND = "donchian-breakout"


def detect_donchian_setups(
    df: pd.DataFrame,
    *,
    entry_period: int | None = None,
    trend_period: int | None = None,
    adx_period: int | None = None,
    adx_min: float | None = None,
) -> list[PatternCandidate]:
    """
    Emit at most one Donchian breakout candidate for the latest closed bar.

    Args:
        df: Setup-tier OHLCV DataFrame, CLOSED bars only, ascending epoch-ms
            index. Callers pass the last config.PATTERN_LOOKBACK_BARS bars.
        entry_period / trend_period / adx_period / adx_min: defaults
            config.DONCHIAN_ENTRY_PERIOD / DONCHIAN_TREND_PERIOD /
            ADX_PERIOD / ADX_TREND_THRESHOLD.

    Returns:
        A list of 0 or 1 PatternCandidate. Long and short are mutually
        exclusive here (a close cannot sit both above and below the 55-mid),
        unlike the triangle detector which emitted both sides.
    """
    if entry_period is None:
        entry_period = config.DONCHIAN_ENTRY_PERIOD
    if trend_period is None:
        trend_period = config.DONCHIAN_TREND_PERIOD
    if adx_period is None:
        adx_period = config.ADX_PERIOD
    if adx_min is None:
        adx_min = config.ADX_TREND_THRESHOLD

    n = len(df)
    if n < max(trend_period + 1, 2 * adx_period - 1):
        return []

    entry_ch = donchian(df, period=entry_period)
    trend_ch = donchian(df, period=trend_period)
    adx_vals = wilder_adx(df, period=adx_period)

    i = n - 1  # latest CLOSED setup bar
    upper = float(entry_ch["upper"].iloc[i])
    lower = float(entry_ch["lower"].iloc[i])
    mid = float(trend_ch["mid"].iloc[i])
    adx_now = float(adx_vals.iloc[i])
    close = float(df["close"].to_numpy()[i])

    if pd.isna(upper) or pd.isna(lower) or pd.isna(mid) or pd.isna(adx_now):
        return []  # warmup
    if adx_now < adx_min:
        return []  # trend not confirmed on the setup tier
    width = upper - lower
    if width <= 0:
        return []  # degenerate channel

    if close > mid:
        direction, level = "long", upper
    elif close < mid:
        direction, level = "short", lower
    else:
        return []  # exactly on the mid-line: no side

    interval = int(df.index.to_numpy()[i]) - int(df.index.to_numpy()[i - 1])
    return [
        PatternCandidate(
            kind=DONCHIAN_KIND,
            direction=direction,
            breakout_level=level,
            target_height=width,
            start_ts=int(df.index.to_numpy()[max(0, i - entry_period)]),
            # end_ts is the setup bar's CLOSE time, so check_breakout's
            # `ts < candidate.end_ts` skip guarantees the trigger bar opens
            # at or after the setup bar closed — no intra-bar lookahead.
            end_ts=int(df.index.to_numpy()[i]) + interval,
        )
    ]


def channel_exit_levels(
    df: pd.DataFrame, *, period: int | None = None
) -> pd.DataFrame:
    """Trailing opposite-channel exit levels for the setup tier.

    A long exits when price touches the trailing `period`-bar LOW; a short
    when it touches the trailing `period`-bar HIGH. Identical to the entry
    channel by construction (same period, other side) — exposed as its own
    function so the backtest engine does not import the indicator directly
    and the "same 20 bars, other side" contract is stated in one place.

    Returns:
        DataFrame indexed like df with columns upper, lower, mid (NaN during
        the first `period` bars).
    """
    return donchian(df, period=period or config.DONCHIAN_ENTRY_PERIOD)


def scan_donchian_signals(conn, symbol: str, now_ms: int) -> list[Signal]:
    """
    Donchian-scan one symbol WITHOUT a regime check (dispatcher applies it).

    Args:
        conn: Database connection.
        symbol: Trading pair symbol.
        now_ms: Evaluation time in epoch milliseconds.

    Returns:
        Screened Signals, best-R:R first (0 or 1 in practice, since
        detect_donchian_setups emits at most one candidate).
    """
    df_setup = _load_df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME, now_ms)
    df_trig = _load_df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME, now_ms)
    if df_setup.empty or df_trig.empty:
        return []

    df_setup = df_setup.tail(config.PATTERN_LOOKBACK_BARS)
    trigger_bars = config.VOLUME_LOOKBACK + 1 + max(1, config.BREAKOUT_TRIGGER_LOOKBACK_BARS)
    df_trig = df_trig.tail(trigger_bars)

    interval_trig = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
    latest_close = int(df_trig.index[-1]) + interval_trig
    if now_ms - latest_close >= interval_trig:
        # Copy setup.py:272-279's staleness warning verbatim, with the
        # timeframe name substituted for the hardcoded "15m".
        logger.warning(
            "%s %s data is %d ms behind now_ms; a breakout may already be older "
            "than the %d-bar trigger window",
            symbol, config.SIGNAL_TRIGGER_TIMEFRAME, now_ms - latest_close,
            config.BREAKOUT_TRIGGER_LOOKBACK_BARS,
        )

    atr_series = wilder_atr(df_setup, period=config.ATR_STOP_PERIOD)
    atr_value = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
    if not (atr_value > 0):  # covers NaN (warmup) and zero/negative
        logger.warning("%s: ATR undefined on %s bars; no signals", symbol,
                       config.SIGNAL_PATTERN_TIMEFRAME)
        return []

    signals: list[Signal] = []
    for candidate in detect_donchian_setups(df_setup):
        event = check_breakout(df_trig, candidate, interval_ms=interval_trig)
        if event is None:
            continue
        signal = build_signal(symbol, candidate, event, atr_value)
        if signal is not None:
            signals.append(signal)

    return rank_signals(signals)
