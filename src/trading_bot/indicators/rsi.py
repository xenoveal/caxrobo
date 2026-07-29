"""
Wilder's Relative Strength Index — pure pandas, no external dependency.

ATTRIBUTION. Ported from `scripts/bruteforce/indicators.py:130` per the v0.3.0
shared architecture contract §0b ("do not re-derive an indicator that exists in
scripts/bruteforce/ — port it, with attribution"). That module is research-grade
and outside the tested package; §0b's condition for entering
`src/trading_bot/` is production-standard tests, which
`tests/test_detectors_oscillator.py::TestRsiIndicator` supplies, including
hand-computed expected values in the style of
`tests/test_wilder.py::TestHandComputedValues`.

WHY WILDER SMOOTHING AND NOT AN EMA. This uses the production
`indicators.wilder.wilder_smooth`, i.e. the same Running Moving Average the
ATR and ADX in this repo already use, rather than the simple-EMA approximation
most libraries ship. One smoothing convention across every indicator means an
RSI reading and an ATR reading warm up on the same recursion, and a reader of
one number does not have to remember which of two conventions produced it.

THE ONE NON-OBVIOUS CLAUSE. `avg_loss == 0` with a positive `avg_gain` is a
pure, unbroken uptrend, where the textbook RS is +inf and RSI is exactly 100.
Without that clause the division yields NaN and RSI is UNDEFINED through every
sustained uptrend — which would silently kill every bearish-divergence
candidate, since divergence pairing needs a defined, elevated first RSI high.
Kept from the donor deliberately (donor line 144-145).

NaN RULE, as everywhere in `indicators/`: NaN means *not knowable yet* and is
never back-filled. `wilder_smooth` seeds on the mean of the first `period`
values, and `diff()` costs one bar, so the first defined RSI sits at positional
index `period` — one later than ATR's `period - 1`. Assert that index; do not
assume it.

SCALE. RSI is already scale-free (it is a ratio of averaged absolute moves), so
unlike `scripts/bruteforce/indicators.py`'s MACD it is NOT normalised by price.
Normalising it again would break the [0, 100] contract every threshold reads.
"""

import numpy as np
import pandas as pd

from trading_bot import config
from trading_bot.indicators.wilder import wilder_smooth


def rsi(close: pd.Series, *, period: int | None = None) -> pd.Series:
    """
    Wilder's RSI in [0, 100].

    Args:
        close: Close price series, indexed by epoch-ms ts, ascending.
        period: Bars in the Wilder-smoothed gain/loss average
            (default config.RSI_PERIOD).

    Returns:
        pd.Series indexed identically to ``close``, values in [0, 100]. The
        first ``period`` entries are NaN (warmup); a flat series with neither
        gains nor losses is NaN, because a market that has not moved has no
        relative strength. A pure uptrend is exactly 100.0 and a pure downtrend
        exactly 0.0.
    """
    if period is None:
        period = config.RSI_PERIOD

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    # A zero average loss makes RS infinite; NaN it here and restore the exact
    # value below, so the division never emits a warning or an inf.
    rs = avg_gain / avg_loss.where(avg_loss != 0.0)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 with a positive avg_gain is a pure uptrend: RSI is 100.
    return out.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)


def rsi_frame(close: pd.Series, *, period: int | None = None) -> pd.DataFrame:
    """
    RSI reshaped as an OHLCV-shaped frame, so `signals.pivots.find_pivots` can
    find swings in it.

    Divergence needs pivots in the OSCILLATOR as well as in price. Rather than
    write a second fractal implementation that can drift from
    `find_pivots`, the oscillator is presented as a degenerate OHLCV frame
    (open = high = low = close = the RSI value, volume = 0) and the SAME
    fractal algorithm — with the same strict-domination rule and the same
    confirmation lag — is run over it.

    The NaN warmup rows are DROPPED rather than filled: NaN makes every
    comparison False, so leaving them in would silently suppress pivots near
    the start of the series while looking like a geometry problem. Dropping
    them keeps the epoch-ms index aligned with ``close`` for the rows that
    remain, which is what the divergence detector matches pivots on.

    Args:
        close: Close price series.
        period: RSI period (default config.RSI_PERIOD).

    Returns:
        DataFrame with columns open/high/low/close/volume indexed by the subset
        of ``close``'s index where RSI is defined. Empty when RSI never warms up.
    """
    series = rsi(close, period=period)
    series = series[series.notna()]
    return pd.DataFrame(
        {
            "open": series,
            "high": series,
            "low": series,
            "close": series,
            "volume": np.zeros(len(series), dtype=float),
        },
        index=series.index,
    )
