"""
Market regime classifier using ADX and ATR percentile thresholds.

This module classifies each bar into one of four market regimes based on trend strength
(ADX) and volatility (ATR percentile rank). The classifier is purely computational,
reading from stored candles without modifying any state.

Regimes:
  - trending: ADX >= threshold (default 25.0), indicating strong directional movement
  - ranging: Low ADX and low volatility, no clear trend
  - extreme-volatility: RELATIVE ATR (ATR/close) percentile >= threshold
    (default 0.90), overrides other labels
  - uncertain: Insufficient data (warmup period), NaN indicators, or empty DB

Precedence (checked in order):
  1. If bar is in warmup, has NaN indicators, or index < REGIME_MIN_BARS → uncertain
  2. If ATR percentile >= extreme threshold → extreme-volatility
  3. If ADX >= trend threshold → trending
  4. Else → ranging
"""

import logging
import time

import pandas as pd

from trading_bot import config
from trading_bot.data import storage
from trading_bot.indicators import wilder

logger = logging.getLogger("trading_bot")

REGIMES = ("trending", "ranging", "extreme-volatility", "uncertain")


def _relative_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR expressed as a FRACTION OF CLOSE, for percentile ranking.

    The extreme-volatility gate asks "is volatility unusually high right now?".
    Ranking raw ATR — which is denominated in quote currency — answers a
    different question: over a trailing window in which price trends, absolute
    ATR trends with the price LEVEL even when relative volatility is flat, so a
    symbol grinding higher sits near its own window maximum more or less
    permanently and is labelled extreme-volatility indefinitely. On stored
    2023-2026 history that suppressed 23.1% / 20.3% / 20.5% of BTC / ETH / SOL
    days, and 34.8% / 24.4% / 30.7% of all ADX >= threshold days — the very
    days a trend method exists to trade.

    Dividing by close makes the ranked quantity dimensionless and comparable
    across the window, which is what the gate's 0.90 threshold always assumed.
    See .claude/PRPs/reports/code review/phase4-7-code-review.md (HIGH-1).
    """
    return wilder.atr(df, period=period) / df["close"]


def _atr_percentile_rank(atr_vals: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank of each ATR value within its trailing window.

    Rank is computed over the window ending at (and including) each bar — no
    lookahead. A value of 1.0 means the current ATR is the highest in the
    window; 0.90 means it exceeds 90% of the window. Bars with fewer than
    `window` defined ATR values are NaN (percentile undefined during warmup).

    Args:
        atr_vals: ATR series (leading NaNs preserved from the indicator warmup).
        window: Trailing window length in bars.

    Returns:
        pd.Series of percentile ranks in (0, 1], NaN during warmup.
    """
    return atr_vals.rolling(window, min_periods=window).rank(pct=True)


def classify_series(
    df: pd.DataFrame,
    *,
    adx_period: int | None = None,
    adx_trend_threshold: float | None = None,
    atr_percentile_window: int | None = None,
    atr_extreme_percentile: float | None = None,
) -> pd.Series:
    """
    Classify each bar in a DataFrame into a market regime.

    The classification is based on ADX (trend strength) and ATR percentile rank
    (volatility). All threshold parameters default to config values if not provided,
    allowing Phase 5 to sweep thresholds without monkeypatching.

    Args:
        df: DataFrame with columns open, high, low, close, volume indexed by epoch-ms ts.
        adx_period: ADX lookback period (default config.ADX_PERIOD).
        adx_trend_threshold: ADX value >= this means trending (default config.ADX_TREND_THRESHOLD).
        atr_percentile_window: Rolling window for ATR percentile (default config.ATR_PERCENTILE_WINDOW).
        atr_extreme_percentile: ATR percentile >= this means extreme-volatility
                                (default config.ATR_EXTREME_PERCENTILE).

    Returns:
        pd.Series of regime labels indexed by ts, same length as input. Labels are strings
        from REGIMES. Bars in the warmup period (< REGIME_MIN_BARS) are marked "uncertain".
    """
    if adx_period is None:
        adx_period = config.ADX_PERIOD
    if adx_trend_threshold is None:
        adx_trend_threshold = config.ADX_TREND_THRESHOLD
    if atr_percentile_window is None:
        atr_percentile_window = config.ATR_PERCENTILE_WINDOW
    if atr_extreme_percentile is None:
        atr_extreme_percentile = config.ATR_EXTREME_PERCENTILE

    # Initialize result series with "uncertain"
    labels = pd.Series("uncertain", index=df.index, dtype=object)

    # Early exit: not enough data
    if len(df) < config.REGIME_MIN_BARS:
        return labels

    # Compute indicators
    adx_vals = wilder.adx(df, period=adx_period)

    # ATR percentile rank within the trailing window (inclusive, no lookahead);
    # NaN until a full window of defined ATR values exists. Ranked on ATR/close,
    # never on raw ATR — see _relative_atr.
    atr_percentile = _atr_percentile_rank(
        _relative_atr(df, adx_period), atr_percentile_window
    )

    # Apply classification rules, in precedence order
    # Rule 1: bars in warmup (index < REGIME_MIN_BARS) or with NaN indicators → uncertain (already default)

    # Rule 2: ATR percentile >= extreme threshold → extreme-volatility (overrides others)
    extreme_mask = atr_percentile >= atr_extreme_percentile
    labels[extreme_mask] = "extreme-volatility"

    # Rule 3: ADX >= threshold → trending
    trending_mask = (adx_vals >= adx_trend_threshold) & ~extreme_mask & (adx_vals.notna())
    labels[trending_mask] = "trending"

    # Rule 4: else (not extreme, not trending) → ranging (only if past warmup and not NaN)
    ranging_mask = (
        ~extreme_mask & ~trending_mask & (adx_vals.notna()) & (atr_percentile.notna())
    )
    labels[ranging_mask] = "ranging"

    # Ensure bars before warmup stay "uncertain"
    labels.iloc[: config.REGIME_MIN_BARS] = "uncertain"

    return labels


def current_regime(
    conn,
    symbol: str,
    *,
    now_ms: int | None = None,
) -> tuple[str, float, float]:
    """
    Classify the latest closed regime-timeframe candle for a symbol.

    Loads recent candles via load_candles (REGIME_TIMEFRAME), computes indicators,
    and returns the regime label plus ADX and ATR percentile values for the most
    recent CLOSED bar. A closed bar is one whose ts + TIMEFRAME_MS[timeframe] <= now_ms.

    If insufficient data or DB is empty, returns ("uncertain", nan, nan).

    Args:
        conn: Database connection.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        now_ms: Current time in epoch milliseconds. If None, uses time.time() * 1000.

    Returns:
        Tuple of (regime_label, adx_value, atr_percentile_value).
        Labels are strings from REGIMES. Values are floats (may be nan if insufficient data).
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    timeframe = config.REGIME_TIMEFRAME
    interval = storage.TIMEFRAME_MS[timeframe]

    # A closed bar satisfies ts + interval <= now_ms; load_candles' end_ms is
    # inclusive, so end_ms = now_ms - interval excludes the forming bar without
    # assuming candle timestamps sit on an epoch-aligned grid
    rows = storage.load_candles(conn, symbol, timeframe, end_ms=now_ms - interval)

    if not rows:
        return ("uncertain", float("nan"), float("nan"))

    # Convert rows to DataFrame
    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    df["ts"] = df["ts"].astype(int)
    df = df.set_index("ts")

    # Classify entire series
    labels = classify_series(df)

    # Get the label for the latest closed bar (which is the last row)
    latest_label = labels.iloc[-1]
    latest_ts = df.index[-1]

    # Indicator values for the latest closed bar, using the same computations
    # classify_series used for its label
    adx_vals = wilder.adx(df, period=config.ADX_PERIOD)
    atr_percentile_val = float(
        _atr_percentile_rank(
            _relative_atr(df, config.ADX_PERIOD), config.ATR_PERCENTILE_WINDOW
        ).iloc[-1]
    )
    adx_val = float(adx_vals.iloc[-1])

    return (latest_label, adx_val, atr_percentile_val)
