"""
Donchian channels, pure pandas.

Upper band is the highest high and lower band the lowest low over the
`period` bars PRECEDING each bar; mid is their average. The current bar is
deliberately EXCLUDED from its own channel: a channel including the current
bar's high can never be closed above (max >= close by construction), so a
self-inclusive channel makes breakout detection impossible while still
reading as a working indicator. NaN until `period` prior bars exist.

Trailing-only, no lookahead: the value at bar i depends on bars
[i-period, i-1] and nothing later, so a series computed over full history is
identical bar-for-bar to one computed incrementally as bars close.
"""

import logging

import pandas as pd

from trading_bot import config

logger = logging.getLogger("trading_bot")


def donchian(df: pd.DataFrame, *, period: int | None = None) -> pd.DataFrame:
    """
    Compute trailing Donchian channels over an OHLCV DataFrame.

    Args:
        df: DataFrame with high/low columns, indexed by epoch-ms ts, ascending.
        period: Channel lookback in bars, EXCLUDING the current bar
            (default config.DONCHIAN_ENTRY_PERIOD).

    Returns:
        DataFrame indexed like df with columns upper, lower, mid. All three
        are NaN for the first `period` bars (period prior bars are required,
        so the first defined value is at positional index `period`).
    """
    if period is None:
        period = config.DONCHIAN_ENTRY_PERIOD

    upper = df["high"].rolling(period, min_periods=period).max().shift(1)
    lower = df["low"].rolling(period, min_periods=period).min().shift(1)
    return pd.DataFrame(
        {"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0},
        index=df.index,
    )
