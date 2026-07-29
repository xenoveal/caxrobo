"""
MACD (Appel 1979) and the EMA it is built on -- pure pandas, no dependencies.

PROVENANCE. Ported from `scripts/bruteforce/indicators.py` -- `macd` at :160-172
and `ema` at :79-82 -- the research donor described in the v0.3.0 shared
architecture contract §0b, whose rule is explicit: "do not re-derive an
indicator or detector that exists in scripts/bruteforce/". The arithmetic here is
deliberately IDENTICAL to the donor's; only the docstrings, the config wiring and
the tests are new. `pandas-ta` is gone from PyPI and TA-Lib needs a C library, so
the hand-rolled set in this package is extended rather than a dependency added
(contract §1). Note the dependency direction: `scripts/bruteforce/` imports from
`src/trading_bot`, never the reverse.

THE EMA SEEDING CONVENTION, spelled out -- this is the classic source of a silent
MACD mismatch, so it is stated rather than left to be inferred:

    ema(s, p) == s.ewm(span=p, adjust=False, min_periods=p).mean()

With `adjust=False`, pandas runs the recursion from the FIRST OBSERVATION:

    y[0] = x[0];  y[i] = (1 - a) * y[i-1] + a * x[i],  a = 2 / (p + 1)

and then MASKS the first `p - 1` outputs as NaN. It does NOT seed with a simple
moving average at index `p - 1`.

TWO CONVENTIONS THIS DIFFERS FROM, both on purpose:

  (a) `indicators/wilder.py`'s `wilder_smooth` seeds with the SIMPLE AVERAGE of
      the first `period` values (wilder.py:92-94) and decays at 1/N, not
      2/(N+1). MACD is defined on EMAs, so it must NOT use Wilder smoothing --
      that would be a different indicator. (The donor's `rsi` deliberately DOES
      use Wilder smoothing; its `macd` deliberately does not.) On
      [10, 20, 30, 40] with period 3 the two disagree at index 2: `ema` gives
      22.5, `wilder_smooth` gives 20.0. Pinned by tests/test_macd.py.
  (b) TA-Lib and TradingView seed the first EMA with an SMA, so values near the
      START of a series differ from those platforms and converge asymptotically.
      DO NOT "fix" this to match a chart; fix the comparison.

CONSEQUENCE -- MACD IS NOT WINDOW-INVARIANT. The recursion is seeded at the first
bar of whatever series it is given, so in general

    macd(s.iloc[-N:]) != macd(s).iloc[-N:]

near the start of the window, converging later. The same is already true of
Wilder ADX, which signals/donchian.py computes per lookback window. Because a
windowed value is therefore a function of an arbitrary window length, the Phase 4
plug-ins (`plugins/detectors/macd_cross.py`, `plugins/confirmations/macd.py`)
compute MACD over FULL HISTORY through `EvalContext.series(...)`, which memoizes
one causal array per (frame content, parameters) and hands back a read-only slice
truncated at the current bar. That is the CONVENTION for MACD in this package:

    MACD is computed on full history and truncated, never on a fixed-size window.

It is available because `ema` is trailing-only, so the full-history value at bar
i depends only on bars <= i and truncation loses nothing (proved per-plug-in with
`framework.context.assert_trailing_only`). tests/test_macd.py pins the window
sensitivity itself, so a future change that quietly re-windows MACD -- or that
"optimizes" a windowed computation into a hoisted full-history one -- fails
loudly instead of silently changing every signal.

CAUSALITY. `adjust=False` plus `min_periods=period` satisfy the donor's two rules
(scripts/bruteforce/indicators.py:1-12): trailing windows only, and NaN means
"not knowable yet" and is never back-filled.

NORMALISATION. `macd`, `signal` and `hist` are each divided by the close. Dividing
by price makes a threshold portable across symbols -- "a raw MACD threshold that
works on BTC is meaningless on DOGE" (the donor's stated rationale at
indicators.py:164-166). The divisor is positive, so a cross or sign test is
identical normalised or not.

WARMUP. At 12/26/9 the `macd` column is first defined at positional index 25
(= slow - 1) and `signal`/`hist` at index 33 (= slow - 1 + signal - 1);
`config.MACD_MIN_BARS` is the corresponding bar COUNT, 34.

I/O CONTRACT, mirroring wilder.py:15-18. Input is a pd.Series of closes on an
epoch-ms int index, ascending. Output is indexed identically with leading NaNs
preserved (never filled).
"""

import pandas as pd

from trading_bot import config

MACD_COLUMNS = ("macd", "signal", "hist")


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, seeded at the first observation.

    `adjust=False` so the value at bar i depends only on bars <= i (the adjusted
    form renormalises using the whole series). `min_periods=period` is
    LOAD-BEARING: without it `ewm` returns a value at index 0 and the warmup
    region silently becomes a short-sample estimate.

    Args:
        series: Input series (typically closes).
        period: Span of the EMA. `alpha = 2 / (period + 1)`.

    Returns:
        pd.Series indexed like `series`, NaN for the first `period - 1` values.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def macd(
    series: pd.Series,
    *,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
) -> pd.DataFrame:
    """MACD line, signal line and histogram, each normalised by price.

    Body identical to the donor at scripts/bruteforce/indicators.py:160-172.

    Args:
        series: Closes, indexed by epoch-ms ts, ascending.
        fast: Fast EMA span (default config.MACD_FAST_PERIOD).
        slow: Slow EMA span (default config.MACD_SLOW_PERIOD).
        signal: Signal-line EMA span (default config.MACD_SIGNAL_PERIOD).

    Returns:
        DataFrame indexed like `series` with columns macd, signal, hist:
            macd   = (ema(s, fast) - ema(s, slow)) / s
            signal = ema(macd_line, signal) / s
            hist   = (macd_line - signal_line) / s
        An empty input returns an empty frame carrying the same three columns,
        so a caller never has to special-case the shape.
    """
    if fast is None:
        fast = config.MACD_FAST_PERIOD
    if slow is None:
        slow = config.MACD_SLOW_PERIOD
    if signal is None:
        signal = config.MACD_SIGNAL_PERIOD

    if len(series) == 0:
        return pd.DataFrame(
            {name: pd.Series(dtype=float) for name in MACD_COLUMNS}, index=series.index
        )

    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    # `line` carries slow-1 leading NaNs. ewm SKIPS them (they are not
    # observations), so the signal line is defined once `signal` non-NaN line
    # values exist -> index slow-1 + signal-1. Never dropna()/fillna() `line`
    # first: that shifts the whole indicator.
    #
    # No guard on the division. Closes are positive by construction in this
    # store, and a zero close would already have broken risk_pct everywhere; a
    # `replace(0, nan)` the donor does not have would be behavioral drift from
    # the port, which is the one thing a port must not introduce.
    return pd.DataFrame(
        {"macd": line / series, "signal": sig / series, "hist": (line - sig) / series}
    )


def macd_hist(df: pd.DataFrame, *, fast: int, slow: int, signal: int):
    """The normalised histogram as a plain array, for `EvalContext.series`.

    A frame-in/array-out factory, which is the shape
    `framework.context.EvalContext.series` and `assert_trailing_only` require.
    Reading `df["close"]` here rather than in each caller keeps the "MACD is
    computed on full history" convention (see the module docstring) in one place.

    Args:
        df: OHLCV frame indexed by epoch-ms ts, ascending.
        fast / slow / signal: MACD spans. Keyword-only and REQUIRED, because
            these values are part of the caller's cache key and a silent config
            default there would serve one graph another graph's numbers.

    Returns:
        numpy array of the normalised histogram, leading NaNs preserved.
    """
    return macd(df["close"], fast=fast, slow=slow, signal=signal)["hist"].to_numpy()
