"""
Bollinger bands, pure pandas.

Middle band is a simple moving average of closes; upper/lower bands sit
`num_std` rolling standard deviations (ddof=0, population std — the classical
Bollinger definition) above/below it. Values are NaN until a full period of
closes exists.
"""

import logging

import pandas as pd

from trading_bot import config

logger = logging.getLogger("trading_bot")


def bollinger(
    df: pd.DataFrame,
    *,
    period: int | None = None,
    num_std: float | None = None,
) -> pd.DataFrame:
    """
    Compute Bollinger bands over an OHLCV DataFrame's closes.

    Args:
        df: DataFrame with a close column, indexed by epoch-ms ts, ascending.
        period: SMA/std lookback in bars (default config.BB_PERIOD).
        num_std: Band width in standard deviations (default config.BB_STD).

    Returns:
        DataFrame indexed like df with columns middle, upper, lower.
        All three are NaN for the first period-1 bars.
    """
    if period is None:
        period = config.BB_PERIOD
    if num_std is None:
        num_std = config.BB_STD

    close = df["close"]
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + num_std * std,
            "lower": middle - num_std * std,
        },
        index=df.index,
    )
