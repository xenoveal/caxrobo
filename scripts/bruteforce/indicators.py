"""Indicator library for the brute-force strategy search.

Two rules govern everything here, because ``core.assert_causal`` enforces them
mechanically and a violation kills the strategy that uses it:

1. **Trailing windows only.** No ``center=True``, no ``shift(-n)``, no
   full-sample statistic (``series.mean()``, ``series.quantile()``). Rolling
   statistics use ``min_periods=window`` so the warmup region stays ``NaN``
   rather than being computed from a short, and therefore different, sample.
2. **Never back-fill.** ``NaN`` means "not knowable yet". Filling it forward is
   fine; filling it backward imports the future.

Wilder ATR/ADX/DI and Donchian channels are RE-EXPORTED from
``trading_bot.indicators`` rather than rewritten -- those are the measured-healthy
components the PRD keeps, and a second implementation would be free to drift.

Every function takes a frame with ``open/high/low/close/volume`` columns indexed
by bar open time in epoch ms, and returns a Series/DataFrame on the same index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trading_bot.indicators.donchian import donchian  # noqa: F401,E402
from trading_bot.indicators.wilder import (  # noqa: F401,E402
    adx,
    atr,
    minus_di,
    plus_di,
    true_range,
    wilder_smooth,
)

__all__ = [
    # re-exported from production
    "atr", "adx", "plus_di", "minus_di", "true_range", "wilder_smooth", "donchian",
    # moving averages / trend
    "sma", "ema", "wma", "hma", "slope", "linreg_slope",
    # oscillators
    "rsi", "stochastic", "macd", "cci", "williams_r", "roc", "momentum",
    # volatility / bands
    "bollinger", "bbwidth", "keltner", "squeeze_on", "atr_pct", "realized_vol",
    "natr", "chandelier",
    # volume
    "vol_ratio", "obv", "vwap_session", "mfi",
    # normalisation
    "zscore", "percentile_rank",
    # structure
    "pivot_high", "pivot_low", "swing_levels", "rolling_high", "rolling_low",
    "range_position",
    # candlesticks
    "bullish_engulfing", "bearish_engulfing", "hammer", "shooting_star",
    "doji", "inside_bar", "outside_bar", "three_bar_reversal", "marubozu",
    # chart patterns
    "double_bottom", "double_top", "triangle_squeeze", "bull_flag", "bear_flag",
    "head_and_shoulders", "inverse_head_and_shoulders", "breakout_of_range",
]


# ---------------------------------------------------------------------------
# Moving averages and trend
# ---------------------------------------------------------------------------


def sma(s: pd.Series, period: int) -> pd.Series:
    """Simple moving average. NaN until ``period`` observations exist."""
    return s.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    """Exponential MA. ``adjust=False`` so the value at bar i depends only on
    bars <= i (the adjusted form renormalises using the whole series)."""
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(s: pd.Series, period: int) -> pd.Series:
    """Linearly weighted MA (most recent bar weighted highest)."""
    w = np.arange(1, period + 1, dtype=float)
    w /= w.sum()
    return s.rolling(period, min_periods=period).apply(
        lambda x: float(np.dot(x, w)), raw=True
    )


def hma(s: pd.Series, period: int) -> pd.Series:
    """Hull MA: WMA(2*WMA(n/2) - WMA(n), sqrt(n)). Faster, lower lag than EMA."""
    half, root = max(1, period // 2), max(1, int(np.sqrt(period)))
    return wma(2 * wma(s, half) - wma(s, period), root)


def slope(s: pd.Series, period: int) -> pd.Series:
    """Per-bar change over ``period`` bars, normalised by the level.

    Scale-free, so it is comparable across symbols priced from $0.10 to $100k.
    """
    return (s - s.shift(period)) / (period * s.shift(period).abs())


def linreg_slope(s: pd.Series, period: int) -> pd.Series:
    """OLS slope of the last ``period`` values, normalised by the level.

    Steadier than a two-point ``slope`` because a single outlier bar cannot
    define the whole reading.
    """
    x = np.arange(period, dtype=float)
    x -= x.mean()
    denom = float((x * x).sum())

    def _fit(win: np.ndarray) -> float:
        return float(np.dot(x, win - win.mean()) / denom)

    raw = s.rolling(period, min_periods=period).apply(_fit, raw=True)
    return raw / s.abs()


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100].

    Uses ``wilder_smooth`` (the production implementation) on gains and losses
    so it matches the ATR/ADX smoothing convention already in the codebase
    rather than the simple-EMA approximation most libraries ship.
    """
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 with a positive avg_gain is a pure uptrend: RSI is 100.
    return out.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)


def stochastic(
    df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator. Returns columns ``k`` and ``d`` in [0, 100]."""
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    span = (hh - ll).replace(0.0, np.nan)
    raw_k = 100.0 * (df["close"] - ll) / span
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    return pd.DataFrame({"k": k, "d": k.rolling(smooth_d, min_periods=smooth_d).mean()})


def macd(
    s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD. Returns ``macd``, ``signal``, ``hist``, each normalised by price.

    Normalising makes thresholds portable across symbols; a raw MACD threshold
    that works on BTC is meaningless on DOGE.
    """
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame(
        {"macd": line / s, "signal": sig / s, "hist": (line - sig) / s}
    )


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index (mean absolute deviation form)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = sma(tp, period)
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: float(np.abs(x - x.mean()).mean()), raw=True
    )
    return (tp - ma) / (0.015 * mad.replace(0.0, np.nan))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R in [-100, 0]."""
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    return -100.0 * (hh - df["close"]) / (hh - ll).replace(0.0, np.nan)


def roc(s: pd.Series, period: int) -> pd.Series:
    """Rate of change over ``period`` bars, as a fraction."""
    return s.pct_change(period)


def momentum(s: pd.Series, period: int) -> pd.Series:
    """Log return over ``period`` bars. Additive across horizons, unlike ROC."""
    return np.log(s / s.shift(period))


# ---------------------------------------------------------------------------
# Volatility and bands
# ---------------------------------------------------------------------------


def bollinger(s: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands. Returns ``mid``, ``upper``, ``lower``.

    Population stdev (``ddof=0``), matching the production
    ``trading_bot.indicators.bollinger`` convention.
    """
    mid = sma(s, period)
    sd = s.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + num_std * sd, "lower": mid - num_std * sd}
    )


def bbwidth(s: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger bandwidth as a fraction of the mid-band -- a volatility proxy."""
    b = bollinger(s, period, num_std)
    return (b["upper"] - b["lower"]) / b["mid"].replace(0.0, np.nan)


def keltner(
    df: pd.DataFrame, period: int = 20, atr_period: int = 10, mult: float = 2.0
) -> pd.DataFrame:
    """Keltner channels around an EMA, width in ATR. Returns mid/upper/lower."""
    mid = ema(df["close"], period)
    a = atr(df, period=atr_period)
    return pd.DataFrame({"mid": mid, "upper": mid + mult * a, "lower": mid - mult * a})


def squeeze_on(
    df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
    kc_period: int = 20, kc_mult: float = 1.5,
) -> pd.Series:
    """TTM-squeeze state: True when Bollinger bands sit INSIDE Keltner channels.

    A squeeze marks compressed volatility; the tradeable event is its RELEASE
    (``squeeze_on`` flipping True -> False), not the squeeze itself.
    """
    b = bollinger(df["close"], bb_period, bb_std)
    k = keltner(df, kc_period, kc_period, kc_mult)
    return (b["upper"] < k["upper"]) & (b["lower"] > k["lower"])


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a fraction of close -- the scale-free volatility measure."""
    return atr(df, period=period) / df["close"]


def natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Normalised ATR in percent (``atr_pct`` * 100), for readability."""
    return 100.0 * atr_pct(df, period)


def realized_vol(s: pd.Series, period: int = 20, bars_per_year: int = 8760) -> pd.Series:
    """Annualised close-to-close realised volatility.

    ``bars_per_year`` defaults to 1H bars (24*365); pass 2190 for 4H, 365 for 1D.
    """
    return np.log(s / s.shift(1)).rolling(period, min_periods=period).std(
        ddof=1
    ) * np.sqrt(bars_per_year)


def chandelier(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """Chandelier exit levels: rolling extreme -/+ mult*ATR. Returns long/short."""
    a = atr(df, period=period)
    return pd.DataFrame(
        {
            "long": df["high"].rolling(period, min_periods=period).max() - mult * a,
            "short": df["low"].rolling(period, min_periods=period).min() + mult * a,
        }
    )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def vol_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume relative to its trailing average. >1.5 is the production
    ``VOLUME_HIGH_RATIO`` notion of 'notably high'."""
    return df["volume"] / sma(df["volume"], period).replace(0.0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume (cumulative signed volume)."""
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return (sign * df["volume"]).cumsum()


def vwap_session(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """Rolling VWAP over ``period`` bars.

    A ROLLING window, not a calendar session: a session VWAP would need the
    session's own start, and a rolling one is both simpler and causal.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).rolling(period, min_periods=period).sum()
    v = df["volume"].rolling(period, min_periods=period).sum()
    return pv / v.replace(0.0, np.nan)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index in [0, 100] -- a volume-weighted RSI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    flow = tp * df["volume"]
    up = flow.where(tp.diff() > 0, 0.0).rolling(period, min_periods=period).sum()
    down = flow.where(tp.diff() < 0, 0.0).rolling(period, min_periods=period).sum()
    return 100.0 - 100.0 / (1.0 + up / down.replace(0.0, np.nan))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def zscore(s: pd.Series, period: int) -> pd.Series:
    """Rolling z-score. TRAILING window -- a full-sample z-score is lookahead."""
    mu = s.rolling(period, min_periods=period).mean()
    sd = s.rolling(period, min_periods=period).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def percentile_rank(s: pd.Series, period: int) -> pd.Series:
    """Fraction of the trailing ``period`` values at or below the current one.

    In [0, 1]. This is the causal form of "is today extreme?" -- the same
    device the production regime classifier uses for its ATR percentile gate.
    """
    return s.rolling(period, min_periods=period).apply(
        lambda x: float((x <= x[-1]).mean()), raw=True
    )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def rolling_high(df: pd.DataFrame, period: int, *, exclude_current: bool = True) -> pd.Series:
    """Highest high of the trailing ``period`` bars.

    ``exclude_current=True`` (the default, and what a breakout rule needs) shifts
    the window back one bar so the level does not include the bar being tested
    against it -- otherwise "close > rolling high" can never be true.
    """
    h = df["high"].shift(1) if exclude_current else df["high"]
    return h.rolling(period, min_periods=period).max()


def rolling_low(df: pd.DataFrame, period: int, *, exclude_current: bool = True) -> pd.Series:
    """Lowest low of the trailing ``period`` bars. See ``rolling_high``."""
    low = df["low"].shift(1) if exclude_current else df["low"]
    return low.rolling(period, min_periods=period).min()


def range_position(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Where close sits in the trailing range: 0 = at the low, 1 = at the high."""
    hh = rolling_high(df, period, exclude_current=False)
    ll = rolling_low(df, period, exclude_current=False)
    return (df["close"] - ll) / (hh - ll).replace(0.0, np.nan)


def pivot_high(df: pd.DataFrame, span: int = 3) -> pd.Series:
    """Fractal pivot highs, CONFIRMED ``span`` bars late.

    A pivot at bar t is only knowable at bar t+span, so the flag is placed at
    t+span and the level carried is the high at t. Marking it at t would be
    textbook lookahead -- and is exactly the trap ``assert_causal`` catches.

    Returns:
        Series of the pivot's price, NaN where no pivot confirmed on that bar.
    """
    h = df["high"]
    win = 2 * span + 1
    # rolling(win) ends at the current bar, so its centre is `span` bars back.
    centre = h.shift(span)
    is_max = centre == h.rolling(win, min_periods=win).max()
    return centre.where(is_max)


def pivot_low(df: pd.DataFrame, span: int = 3) -> pd.Series:
    """Fractal pivot lows, confirmed ``span`` bars late. See ``pivot_high``."""
    low = df["low"]
    win = 2 * span + 1
    centre = low.shift(span)
    is_min = centre == low.rolling(win, min_periods=win).min()
    return centre.where(is_min)


def swing_levels(df: pd.DataFrame, span: int = 3) -> pd.DataFrame:
    """Most recent CONFIRMED pivot high/low, forward-filled.

    Returns ``res`` (resistance) and ``sup`` (support): the price of the latest
    confirmed pivot of each kind as of each bar. Forward-fill is causal; the
    NaN warmup before the first pivot is left as NaN.
    """
    return pd.DataFrame(
        {"res": pivot_high(df, span).ffill(), "sup": pivot_low(df, span).ffill()}
    )


# ---------------------------------------------------------------------------
# Candlestick patterns
#
# All are single- or few-bar and complete at the CURRENT bar's close, so they are
# causal by construction. Bodies and ranges are compared as fractions so the
# same thresholds work on any symbol.
# ---------------------------------------------------------------------------


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0.0, np.nan)


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Down bar followed by an up bar whose body engulfs it."""
    prev_down = df["close"].shift(1) < df["open"].shift(1)
    up = df["close"] > df["open"]
    engulf = (df["close"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1))
    return prev_down & up & engulf


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Up bar followed by a down bar whose body engulfs it."""
    prev_up = df["close"].shift(1) > df["open"].shift(1)
    down = df["close"] < df["open"]
    engulf = (df["close"] <= df["open"].shift(1)) & (df["open"] >= df["close"].shift(1))
    return prev_up & down & engulf


def hammer(df: pd.DataFrame, *, wick_mult: float = 2.0, body_max: float = 0.35) -> pd.Series:
    """Long lower wick, small body in the upper part of the range."""
    lower = (df[["open", "close"]].min(axis=1) - df["low"])
    return (lower >= wick_mult * _body(df)) & (_body(df) / _range(df) <= body_max)


def shooting_star(
    df: pd.DataFrame, *, wick_mult: float = 2.0, body_max: float = 0.35
) -> pd.Series:
    """Long upper wick, small body in the lower part of the range."""
    upper = (df["high"] - df[["open", "close"]].max(axis=1))
    return (upper >= wick_mult * _body(df)) & (_body(df) / _range(df) <= body_max)


def doji(df: pd.DataFrame, *, body_max: float = 0.1) -> pd.Series:
    """Body is a negligible fraction of the range -- indecision."""
    return _body(df) / _range(df) <= body_max


def marubozu(df: pd.DataFrame, *, body_min: float = 0.9) -> pd.Series:
    """Body fills almost the whole range -- one-sided conviction."""
    return _body(df) / _range(df) >= body_min


def inside_bar(df: pd.DataFrame) -> pd.Series:
    """Range contained by the previous bar's -- compression before expansion."""
    return (df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))


def outside_bar(df: pd.DataFrame) -> pd.Series:
    """Range engulfs the previous bar's -- volatility expansion."""
    return (df["high"] >= df["high"].shift(1)) & (df["low"] <= df["low"].shift(1))


def three_bar_reversal(df: pd.DataFrame) -> pd.DataFrame:
    """Down-down-up (``bull``) and up-up-down (``bear``) three-bar turns."""
    down = df["close"] < df["open"]
    up = ~down
    return pd.DataFrame(
        {
            "bull": down.shift(2).fillna(False) & down.shift(1).fillna(False) & up
            & (df["close"] > df["high"].shift(1)),
            "bear": up.shift(2).fillna(False) & up.shift(1).fillna(False) & down
            & (df["close"] < df["low"].shift(1)),
        }
    )


# ---------------------------------------------------------------------------
# Chart patterns
#
# The PRD RETIRED triangle/flag/H&S geometry after measuring them as losers
# under the OLD absolute-percentage risk model. They are re-implemented here for
# a fair re-test under ATR stops, per the user's explicit request. Each detector
# fires on the bar the geometry COMPLETES, using only bars up to that close.
# ---------------------------------------------------------------------------


def double_bottom(
    df: pd.DataFrame, *, span: int = 3, tol: float = 0.02, max_gap: int = 60
) -> pd.Series:
    """W-shape: two confirmed pivot lows within ``tol`` of each other.

    Fires on the bar the SECOND low is confirmed. ``max_gap`` bounds how far
    apart the two lows may be, so an unrelated low from months ago cannot pair
    with today's.
    """
    lows = pivot_low(df, span)
    out = pd.Series(False, index=df.index)
    idx = np.flatnonzero(lows.notna().to_numpy())
    vals = lows.to_numpy()
    for a, b in zip(idx, idx[1:]):
        if b - a <= max_gap and abs(vals[b] - vals[a]) <= tol * vals[a]:
            out.iloc[b] = True
    return out


def double_top(
    df: pd.DataFrame, *, span: int = 3, tol: float = 0.02, max_gap: int = 60
) -> pd.Series:
    """M-shape: two confirmed pivot highs within ``tol``. See ``double_bottom``."""
    highs = pivot_high(df, span)
    out = pd.Series(False, index=df.index)
    idx = np.flatnonzero(highs.notna().to_numpy())
    vals = highs.to_numpy()
    for a, b in zip(idx, idx[1:]):
        if b - a <= max_gap and abs(vals[b] - vals[a]) <= tol * vals[a]:
            out.iloc[b] = True
    return out


def triangle_squeeze(
    df: pd.DataFrame, *, period: int = 40, min_contraction: float = 0.4
) -> pd.Series:
    """Converging range: current range width is a small fraction of its own past.

    A measurement-based stand-in for fitted trendlines. The production
    ``signals/patterns.py`` fits actual lines through pivots; that version was
    measured to lose money and, more importantly, it is slow. This captures the
    tradeable content -- contraction before expansion -- with no fitted
    parameters to overfit.
    """
    half = max(2, period // 2)
    recent = rolling_high(df, half, exclude_current=False) - rolling_low(
        df, half, exclude_current=False
    )
    older = recent.shift(half)
    return recent <= min_contraction * older


def bull_flag(
    df: pd.DataFrame, *, pole: int = 12, consol: int = 12,
    pole_min: float = 0.03, max_retrace: float = 0.5,
) -> pd.Series:
    """Impulse up (``pole``), then a shallow drift that holds most of the gain."""
    c = df["close"]
    pole_start = c.shift(pole + consol)
    pole_end = c.shift(consol)
    gain = (pole_end - pole_start) / pole_start
    lowest = rolling_low(df, consol, exclude_current=False)
    retrace = (pole_end - lowest) / (pole_end - pole_start).replace(0.0, np.nan)
    return (gain >= pole_min) & (retrace <= max_retrace) & (retrace >= 0)


def bear_flag(
    df: pd.DataFrame, *, pole: int = 12, consol: int = 12,
    pole_min: float = 0.03, max_retrace: float = 0.5,
) -> pd.Series:
    """Mirror of ``bull_flag``: impulse down, then a shallow bounce."""
    c = df["close"]
    pole_start = c.shift(pole + consol)
    pole_end = c.shift(consol)
    drop = (pole_start - pole_end) / pole_start
    highest = rolling_high(df, consol, exclude_current=False)
    retrace = (highest - pole_end) / (pole_start - pole_end).replace(0.0, np.nan)
    return (drop >= pole_min) & (retrace <= max_retrace) & (retrace >= 0)


def _hs(df: pd.DataFrame, span: int, tol: float, prominence: float, inverse: bool) -> pd.Series:
    piv = pivot_low(df, span) if inverse else pivot_high(df, span)
    out = pd.Series(False, index=df.index)
    idx = np.flatnonzero(piv.notna().to_numpy())
    vals = piv.to_numpy()
    for a, b, c in zip(idx, idx[1:], idx[2:]):
        left, head, right = vals[a], vals[b], vals[c]
        if abs(right - left) > tol * left:
            continue  # shoulders must be comparable
        if inverse:
            if head < min(left, right) * (1.0 - prominence):
                out.iloc[c] = True
        else:
            if head > max(left, right) * (1.0 + prominence):
                out.iloc[c] = True
    return out


def head_and_shoulders(
    df: pd.DataFrame, *, span: int = 3, tol: float = 0.03, prominence: float = 0.01
) -> pd.Series:
    """Three pivot highs, middle highest, outer two within ``tol``. Bearish.

    Fires when the RIGHT shoulder confirms -- not at the neckline break, which
    a strategy can add itself as a trigger condition.
    """
    return _hs(df, span, tol, prominence, inverse=False)


def inverse_head_and_shoulders(
    df: pd.DataFrame, *, span: int = 3, tol: float = 0.03, prominence: float = 0.01
) -> pd.Series:
    """Three pivot lows, middle lowest. Bullish. See ``head_and_shoulders``."""
    return _hs(df, span, tol, prominence, inverse=True)


def breakout_of_range(
    df: pd.DataFrame, *, period: int = 40, max_width: float = 0.06
) -> pd.DataFrame:
    """Break out of a TIGHT trailing range. Returns ``up`` and ``down``.

    The width filter is what distinguishes this from a plain Donchian break:
    only ranges narrow enough to be a real consolidation qualify, so the
    breakout has a defined, small risk.
    """
    hi = rolling_high(df, period)
    lo = rolling_low(df, period)
    width = (hi - lo) / df["close"]
    tight = width <= max_width
    return pd.DataFrame(
        {"up": tight & (df["close"] > hi), "down": tight & (df["close"] < lo)}
    )
