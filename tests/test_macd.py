"""Tests for indicators/macd.py (v0.3.0 Phase 4).

Mirrors tests/test_wilder.py::TestHandComputedValues: every expected numeric
value is HAND-COMPUTED with the arithmetic written beside the assertion, so a
reader can re-derive it. A numeric test whose expected value cannot be
re-derived is a snapshot, and a snapshot of the wrong smoothing convention is
exactly the failure these tests exist to prevent.
"""

import math

import numpy as np
import pandas as pd

from trading_bot import config
from trading_bot.indicators import macd as macd_mod
from trading_bot.indicators.wilder import wilder_smooth

START = 1_700_000_000_000
INTERVAL = 3_600_000


def series(values):
    """A closes Series on an epoch-ms int index, the module's stated I/O shape."""
    return pd.Series(
        [float(v) for v in values],
        index=pd.Index([START + i * INTERVAL for i in range(len(values))], name="ts"),
        dtype=float,
    )


def ramp(n, base=100.0):
    """A monotone close ramp; MACD on it is smooth and its sign is stable."""
    return series([base + i for i in range(n)])


def frame(values):
    """An OHLCV frame whose close is `values`, for the macd_hist factory."""
    s = series(values)
    return pd.DataFrame(
        {"open": s, "high": s + 1.0, "low": s - 1.0, "close": s, "volume": 10.0},
        index=s.index,
    )


class TestEmaSeedingConvention:
    """The load-bearing tests: which seed, which decay, which mask.

    All three assertions below are on [10, 20, 30, 40] with period=3, where
    alpha = 2 / (3 + 1) = 0.5, so the recursion is exactly a running midpoint
    and every value is an exact binary fraction (no float tolerance needed).
    """

    def test_ema_hand_computed_first_observation_seed(self):
        # adjust=False runs the recursion from the FIRST OBSERVATION:
        #   y0 = x0                     = 10
        #   y1 = 0.5*10 + 0.5*20        = 15
        #   y2 = 0.5*15 + 0.5*30        = 22.5
        #   y3 = 0.5*22.5 + 0.5*40      = 31.25
        # min_periods=3 then MASKS indices 0 and 1.
        out = macd_mod.ema(series([10, 20, 30, 40]), 3)
        assert math.isnan(out.iloc[0])
        assert math.isnan(out.iloc[1])
        assert out.iloc[2] == 22.5
        assert out.iloc[3] == 31.25

    def test_ema_is_not_sma_seeded(self):
        """An SMA-seeded EMA (TA-Lib, TradingView) would give 20.0 at index 2.

        That convention seeds y[period-1] with mean(x[0:period]) =
        (10 + 20 + 30) / 3 = 20.0. Ours does not. This assertion is the reason
        the module docstring names all three conventions: a future reader
        comparing against a chart must fix the COMPARISON, not this code.
        """
        out = macd_mod.ema(series([10, 20, 30, 40]), 3)
        assert out.iloc[2] != 20.0
        assert out.iloc[2] == 22.5

    def test_ema_is_not_wilder_smooth(self):
        """Two smoothing conventions coexist in indicators/ and must not be confused.

        wilder_smooth seeds with the SIMPLE AVERAGE of the first `period` values
        (wilder.py:92-94) and decays at 1/N:
            seed = (10 + 20 + 30) / 3            = 20.0
            next = (20.0 * 2 + 40) / 3           = 26.666...
        ema seeds at the first observation and decays at 2/(N+1):
            22.5, then 31.25 (see the test above).
        MACD is defined on EMAs, so using Wilder smoothing here would be a
        different indicator.
        """
        s = series([10, 20, 30, 40])
        e = macd_mod.ema(s, 3)
        w = wilder_smooth(s, 3)
        assert w.iloc[2] == 20.0
        assert math.isclose(w.iloc[3], 80.0 / 3.0)
        assert e.iloc[2] != w.iloc[2]
        assert e.iloc[3] != w.iloc[3]

    def test_min_periods_masks_the_warmup(self):
        """min_periods is load-bearing: without it ewm returns a value at index 0
        and the warmup region silently becomes a short-sample estimate."""
        out = macd_mod.ema(ramp(10), 5)
        assert out.iloc[:4].isna().all()
        assert not math.isnan(out.iloc[4])


class TestWarmupIndices:
    """Warmup derived from the config constants, never from literals.

    Deriving the expectation from config (the tests/test_backtest.py:24-33
    pattern) means a period change cannot leave this test green and wrong.
    """

    def test_first_valid_indices(self):
        fast = config.MACD_FAST_PERIOD
        slow = config.MACD_SLOW_PERIOD
        sig = config.MACD_SIGNAL_PERIOD
        s = ramp(60)
        out = macd_mod.macd(s)
        idx = list(out.index)

        # macd line = ema(s, fast) - ema(s, slow); the slower EMA binds, and its
        # first defined positional index is slow - 1 (= 25 at 12/26/9).
        assert idx.index(out["macd"].first_valid_index()) == slow - 1
        assert fast < slow  # the premise of the line above

        # signal = ema(line, sig). `line` has slow-1 leading NaNs, which ewm
        # SKIPS, so `sig` non-NaN line values are needed on top:
        #   slow - 1 + sig - 1 = 25 + 8 = 33.
        assert idx.index(out["signal"].first_valid_index()) == slow - 1 + sig - 1
        assert idx.index(out["hist"].first_valid_index()) == slow - 1 + sig - 1

    def test_macd_min_bars_is_the_bar_count_for_that_index(self):
        """MACD_MIN_BARS is a COUNT, one more than the first defined INDEX."""
        s = ramp(60)
        out = macd_mod.macd(s)
        first_idx = list(out.index).index(out["hist"].first_valid_index())
        assert config.MACD_MIN_BARS == first_idx + 1
        assert config.MACD_MIN_BARS == (
            config.MACD_SLOW_PERIOD + config.MACD_SIGNAL_PERIOD - 1
        )

    def test_exactly_min_bars_yields_one_defined_hist(self):
        out = macd_mod.macd(ramp(config.MACD_MIN_BARS))
        assert out["hist"].notna().sum() == 1
        out_one_short = macd_mod.macd(ramp(config.MACD_MIN_BARS - 1))
        assert out_one_short["hist"].isna().all()


class TestNoLookahead:
    """Trailing-only: truncating the series cannot change an earlier value.

    This is what adjust=False buys, and it is the property that makes computing
    MACD over full history and slicing at the current bar safe.
    """

    def test_truncation_leaves_earlier_values_unchanged(self):
        s = ramp(80)
        full = macd_mod.macd(s)
        for cut in (40, 55, 79):
            partial = macd_mod.macd(s.iloc[: cut + 1])
            assert len(partial) == cut + 1
            for col in macd_mod.MACD_COLUMNS:
                a, b = float(full[col].iloc[cut]), float(partial[col].iloc[cut])
                assert (math.isnan(a) and math.isnan(b)) or a == b, (
                    f"{col} at index {cut}: {a!r} over full history, {b!r} truncated"
                )

    def test_appending_a_bar_does_not_change_history(self):
        s = ramp(60)
        before = macd_mod.macd(s)["hist"].to_numpy()
        after = macd_mod.macd(series(list(s) + [999.0]))["hist"].to_numpy()
        np.testing.assert_array_equal(before, after[:-1])


class TestWindowDependence:
    """MACD IS NOT WINDOW-INVARIANT, and that is pinned deliberately.

    The recursion is seeded at the first bar of whatever series it is given, so
    macd(s[-N:]) differs from macd(s)[-N:] near the START of the window and
    converges later. The same is already true of Wilder ADX, which
    signals/donchian.py:103 computes per lookback window.

    Because a windowed value is a function of an arbitrary window length, the
    Phase 4 plug-ins compute MACD over FULL history via EvalContext.series and
    truncate (see indicators/macd.py's CONSEQUENCE paragraph). Without this test
    a future change that re-windows MACD -- or that hoists a windowed
    computation out to full history -- would silently change every signal.
    """

    def test_window_differs_early_and_converges_late(self):
        # A series with real curvature; on a straight ramp the two agree too
        # closely for the divergence to be visible.
        s = series([100.0 + 20.0 * math.sin(i / 7.0) + 0.3 * i for i in range(200)])
        n = 60
        windowed = macd_mod.macd(s.iloc[-n:])["hist"].to_numpy()
        sliced = macd_mod.macd(s)["hist"].to_numpy()[-n:]

        # Early in the window the windowed value is still NaN (its own warmup)
        # while the full-history value is already defined -- the starkest form of
        # the divergence.
        assert math.isnan(windowed[0])
        assert not math.isnan(sliced[0])

        # At the first bar BOTH are defined they disagree, and the disagreement
        # DECAYS as the recursion forgets its seed. MEASURED on this fixture
        # (pandas 3.0.3): at n=60 the gap is 5.97e-03 at the first commonly
        # defined bar (index 33) and 6.55e-04 at the last, a factor of 9.1. The
        # assertion is on the decay factor rather than on an absolute tolerance,
        # because the residual at 60 bars is still ~7e-4 -- which is the whole
        # point: a 60-bar window does NOT reproduce full-history MACD.
        first_both = next(
            i for i in range(n) if not math.isnan(windowed[i]) and not math.isnan(sliced[i])
        )
        gap_first = abs(windowed[first_both] - sliced[first_both])
        gap_last = abs(windowed[-1] - sliced[-1])
        assert windowed[first_both] != sliced[first_both]
        assert gap_first > 1e-4, "the fixture must show a material early divergence"
        assert gap_last > 0.0, "60 bars is NOT enough to reproduce full history"
        assert gap_last < gap_first / 5.0, "the divergence must decay"

    def test_a_long_window_does_converge(self):
        """The counterpart: given enough bars the two agree to 1e-5.

        MEASURED on the same fixture at n=140: gap 6.52e-07. Together with the
        test above this states the convention precisely -- the divergence is a
        warmup artifact of the seed, not a permanent disagreement -- while still
        forbidding a re-window, because "enough bars" is not something a caller
        with a fixed lookback can guarantee.
        """
        s = series([100.0 + 20.0 * math.sin(i / 7.0) + 0.3 * i for i in range(200)])
        n = 140
        windowed = macd_mod.macd(s.iloc[-n:])["hist"].to_numpy()
        sliced = macd_mod.macd(s)["hist"].to_numpy()[-n:]
        assert math.isclose(windowed[-1], sliced[-1], rel_tol=0.0, abs_tol=1e-5)

    def test_hist_factory_reads_close_and_matches_macd(self):
        """macd_hist is the EvalContext.series factory; it must agree exactly."""
        df = frame([100.0 + 5.0 * math.sin(i / 5.0) for i in range(80)])
        direct = macd_mod.macd(
            df["close"],
            fast=config.MACD_FAST_PERIOD,
            slow=config.MACD_SLOW_PERIOD,
            signal=config.MACD_SIGNAL_PERIOD,
        )["hist"].to_numpy()
        via_factory = macd_mod.macd_hist(
            df,
            fast=config.MACD_FAST_PERIOD,
            slow=config.MACD_SLOW_PERIOD,
            signal=config.MACD_SIGNAL_PERIOD,
        )
        np.testing.assert_array_equal(direct, via_factory)


class TestNormalisation:
    """hist = (line - signal) / close, and its SIGN is normalisation-invariant.

    The sign invariance is the premise D4 relies on: confirmation.macd is a sign
    test on the normalised histogram, and dividing by a positive close cannot
    change a sign.
    """

    def test_hist_equals_unnormalised_difference_over_close(self):
        s = series([100.0 + 4.0 * math.sin(i / 6.0) for i in range(80)])
        out = macd_mod.macd(s)
        line_raw = macd_mod.ema(s, config.MACD_FAST_PERIOD) - macd_mod.ema(
            s, config.MACD_SLOW_PERIOD
        )
        sig_raw = macd_mod.ema(line_raw, config.MACD_SIGNAL_PERIOD)
        for k in (-3, -2, -1):
            expected = float((line_raw.iloc[k] - sig_raw.iloc[k]) / s.iloc[k])
            assert math.isclose(
                float(out["hist"].iloc[k]), expected, rel_tol=0.0, abs_tol=1e-15
            )

    def test_sign_is_invariant_to_normalisation(self):
        s = series([100.0 + 4.0 * math.sin(i / 6.0) for i in range(120)])
        out = macd_mod.macd(s)
        line_raw = macd_mod.ema(s, config.MACD_FAST_PERIOD) - macd_mod.ema(
            s, config.MACD_SLOW_PERIOD
        )
        sig_raw = macd_mod.ema(line_raw, config.MACD_SIGNAL_PERIOD)
        raw_hist = (line_raw - sig_raw).to_numpy()
        norm_hist = out["hist"].to_numpy()
        defined = ~np.isnan(raw_hist)
        assert defined.sum() > 50, "fixture must produce plenty of defined bars"
        assert np.all(np.sign(raw_hist[defined]) == np.sign(norm_hist[defined]))
        # and the fixture genuinely crosses zero, so the assertion has teeth
        assert (norm_hist[defined] > 0).any() and (norm_hist[defined] < 0).any()


class TestDegenerate:
    def test_empty_series_returns_empty_frame_with_columns(self):
        out = macd_mod.macd(series([]))
        assert list(out.columns) == list(macd_mod.MACD_COLUMNS)
        assert len(out) == 0

    def test_short_series_is_all_nan_without_raising(self):
        out = macd_mod.macd(ramp(10))
        assert len(out) == 10
        assert out["hist"].isna().all()
        assert out["macd"].isna().all()

    def test_constant_series_gives_zero_hist(self):
        """A flat series has both EMAs equal, so line, signal and hist are 0."""
        out = macd_mod.macd(series([100.0] * 60))
        assert float(out["hist"].iloc[-1]) == 0.0
        assert float(out["macd"].iloc[-1]) == 0.0
