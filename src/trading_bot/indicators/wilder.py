"""
Wilder's indicators: ATR, DI, ADX — pure pandas, no external dependencies.

This module implements the directional movement system from Welles Wilder Jr.'s 1978
"New Concepts in Technical Trading Systems". All indicators use Wilder's smoothing
(RMA: Running Moving Average), which differs from simple or exponential moving averages.

Wilder smoothing with period N uses the recursive formula:
    smoothed_value = (prev_smoothed * (N - 1) + current_value) / N

The first smoothed value is the simple average of the first N values; subsequent
values follow the recursive formula. This produces values that stabilize over time,
making ADX reliable after a warmup period of 2*N - 1 bars.

All inputs are DataFrames with columns: open, high, low, close, volume (indexed by
epoch-ms int timestamp). All outputs are pd.Series indexed identically, with leading
NaNs preserved (never filled).
"""

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the True Range (TR) for each bar.

    True Range is the maximum of:
    - high - low (current bar range)
    - |high - previous close| (gap up)
    - |low - previous close| (gap down)

    Args:
        df: DataFrame with columns open, high, low, close, volume indexed by ts.

    Returns:
        pd.Series of True Range values. Row 0 has no previous close, so its two
        gap terms are NaN; .max(axis=1) skips them and returns high - low, so
        row 0 is DEFINED (not NaN) and ATR seeds one bar earlier than a
        textbook implementation that drops it.
    """
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()

    return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Apply Wilder's smoothing (Running Moving Average / RMA) to a series.

    The first smoothed value is the simple average of the first `period` values
    (ignoring any leading NaNs). Once seeded, subsequent values use the recursive:
        smoothed[i] = (smoothed[i-1] * (period - 1) + series[i]) / period

    If the series has leading NaNs (e.g., DX computed from DI which starts at
    index period-1), the first smoothed value is delayed until enough valid
    data is accumulated.

    Args:
        series: Input series (typically TR or DM).
        period: Lookback period (e.g., 14 for ADX/ATR).

    Returns:
        pd.Series with NaN until a first smoothed value can be computed.
    """
    # Operates on a NumPy buffer rather than per-element .iloc access. The
    # recursion is inherently sequential (each value depends on the previous),
    # but .iloc scalar indexing dominated the cost: adx() invokes this five
    # times, and the walk-forward calls run_backtest hundreds of times, which
    # made a full pooled run take many hours. Semantics are unchanged — same
    # seed index, same seed value, same NaN propagation.
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return pd.Series(out, index=series.index, dtype=float)

    # First index starting a run of `period` consecutive non-NaN values.
    nan_prefix = np.concatenate(([0], np.cumsum(np.isnan(values))))
    first_valid_idx = None
    for i in range(n - period + 1):
        if nan_prefix[i + period] - nan_prefix[i] == 0:
            first_valid_idx = i
            break

    if first_valid_idx is None:
        # Not enough consecutive non-NaN values to seed smoothing
        return pd.Series(out, index=series.index, dtype=float)

    # Seed: simple average of the first `period` consecutive non-NaN values.
    seed_idx = first_valid_idx + period - 1
    prev = float(values[first_valid_idx : first_valid_idx + period].mean())
    out[seed_idx] = prev

    # Recursive formula for subsequent values. Once a NaN input appears, prev
    # becomes NaN and stays NaN, matching the original's carry-forward of an
    # undefined previous value.
    for i in range(seed_idx + 1, n):
        v = values[i]
        prev = np.nan if (np.isnan(v) or np.isnan(prev)) else (prev * (period - 1) + v) / period
        out[i] = prev

    return pd.Series(out, index=series.index, dtype=float)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) using Wilder's smoothing.

    ATR is the Wilder-smoothed True Range over `period` bars.

    Args:
        df: DataFrame with columns open, high, low, close, volume indexed by ts.
        period: Lookback period (default 14, Wilder's standard).

    Returns:
        pd.Series of ATR values. First defined value at index period-1.
    """
    tr = true_range(df)
    return wilder_smooth(tr, period)


def plus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the positive Directional Indicator (+DI) using Wilder's method.

    +DI measures upward movement and is expressed as a percentage of ATR.

    Directional Movement rules:
    - up_move = high - previous high
    - down_move = previous low - low
    - If up_move > 0 and up_move > down_move: +DM = up_move, -DM = 0
    - If down_move > 0 and down_move > up_move: +DM = 0, -DM = down_move
    - Otherwise: +DM = 0, -DM = 0

    Then: +DI = 100 * Wilder_smooth(+DM, period) / ATR(period)

    Args:
        df: DataFrame with columns open, high, low, close, volume indexed by ts.
        period: Lookback period (default 14).

    Returns:
        pd.Series of +DI values. First defined value at index 2*period-2.
    """
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]

    # Determine which directional movement is valid
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    # Where up_move is positive and greater than down_move
    up_cond = (up_move > 0) & (up_move > down_move)
    plus_dm[up_cond] = up_move[up_cond]

    # Where down_move is positive and greater than up_move
    down_cond = (down_move > 0) & (down_move > up_move)
    minus_dm[down_cond] = down_move[down_cond]

    # First bar (index 0) has no previous bar; it is set to 0 (no movement)

    plus_dm_smooth = wilder_smooth(plus_dm, period)
    atr_val = atr(df, period)

    # Avoid division by zero
    result = pd.Series(float("nan"), index=df.index)
    valid = (atr_val > 0) & (atr_val.notna())
    result[valid] = 100.0 * plus_dm_smooth[valid] / atr_val[valid]

    return result


def minus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the negative Directional Indicator (-DI) using Wilder's method.

    -DI measures downward movement and is expressed as a percentage of ATR.

    See plus_di() for directional movement rules.

    Then: -DI = 100 * Wilder_smooth(-DM, period) / ATR(period)

    Args:
        df: DataFrame with columns open, high, low, close, volume indexed by ts.
        period: Lookback period (default 14).

    Returns:
        pd.Series of -DI values. First defined value at index 2*period-2.
    """
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]

    # Determine which directional movement is valid
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    # Where up_move is positive and greater than down_move
    up_cond = (up_move > 0) & (up_move > down_move)
    plus_dm[up_cond] = up_move[up_cond]

    # Where down_move is positive and greater than up_move
    down_cond = (down_move > 0) & (down_move > up_move)
    minus_dm[down_cond] = down_move[down_cond]

    # First bar (index 0) has no previous bar; it is set to 0 (no movement)

    minus_dm_smooth = wilder_smooth(minus_dm, period)
    atr_val = atr(df, period)

    # Avoid division by zero
    result = pd.Series(float("nan"), index=df.index)
    valid = (atr_val > 0) & (atr_val.notna())
    result[valid] = 100.0 * minus_dm_smooth[valid] / atr_val[valid]

    return result


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX) using Wilder's method.

    ADX measures trend strength (0-100 scale) independent of direction.

    DX is calculated as:
        DX = 100 * |+DI - -DI| / (+DI + -DI)

    When +DI + -DI = 0 (no directional movement), DX is set to 0.

    Then ADX is DX smoothed with Wilder's method over `period` bars.

    ADX requires 2*period - 1 bars before the first defined value.
    Earlier indices are NaN.

    Args:
        df: DataFrame with columns open, high, low, close, volume indexed by ts.
        period: Lookback period (default 14). ADX(14) needs 27 bars minimum.

    Returns:
        pd.Series of ADX values. First defined value at index 2*period-2.
    """
    pdi = plus_di(df, period)
    mdi = minus_di(df, period)

    # Calculate DX
    di_sum = pdi + mdi
    di_diff = (pdi - mdi).abs()

    # DX with proper handling of edge cases
    dx = pd.Series(float("nan"), index=df.index)
    valid = pdi.notna() & mdi.notna()  # Both DI values must be defined

    # Where +DI + -DI > 0, compute DX
    nonzero_di = (di_sum > 0) & valid
    dx[nonzero_di] = 100.0 * di_diff[nonzero_di] / di_sum[nonzero_di]

    # Where +DI + -DI = 0, set DX to 0 (no directional movement)
    zero_di = (di_sum == 0) & valid
    dx[zero_di] = 0.0

    # ADX is Wilder-smoothed DX
    return wilder_smooth(dx, period)
