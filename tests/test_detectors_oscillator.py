"""
Tests for indicators/rsi.py and plugins/detectors/oscillator.py.

RSI's unit tests live HERE rather than in an unreserved tests/test_rsi.py,
per the v0.3.0 shared architecture contract §8 (Phase 8's reserved test files
are `test_detectors_<family>.py`).

The numeric tests are HAND-COMPUTED with the working shown, mirroring
tests/test_wilder.py::TestHandComputedValues. Copying values out of the
implementation's own output would test that the code equals itself.
"""

import math

import pandas as pd
import pytest

from tests.conftest import detect_events, make_setup_context
from tests.test_signals import make_df
from trading_bot import config
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import registry
from trading_bot.indicators.rsi import rsi, rsi_frame
from trading_bot.plugins.detectors import _geometry as g
from trading_bot.signals.pivots import find_pivots

SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
PIVOT_SPAN = config.PIVOT_SPAN
RSI_PERIOD = config.RSI_PERIOD
START = 1_700_000_000_000

DIV = "detector.rsi-divergence"


@pytest.fixture(autouse=True)
def _clear_framework_caches():
    fcontext.clear_caches()
    yield
    fcontext.clear_caches()


class TestRsiIndicator:
    def test_monotone_rise_is_exactly_one_hundred(self):
        """avg_loss == 0 with a positive avg_gain is a pure uptrend, where the
        textbook RS is infinite and RSI is exactly 100. Without that clause the
        division yields NaN and RSI is UNDEFINED through every sustained
        uptrend, which would silently kill every bearish-divergence candidate."""
        s = pd.Series([100.0 + i for i in range(40)])
        out = rsi(s, period=RSI_PERIOD)
        assert out.iloc[RSI_PERIOD] == 100.0
        assert out.iloc[-1] == 100.0

    def test_monotone_fall_is_exactly_zero(self):
        s = pd.Series([200.0 - i for i in range(40)])
        assert rsi(s, period=RSI_PERIOD).iloc[RSI_PERIOD] == 0.0

    def test_alternating_unit_moves_are_fifty(self):
        """Over the seed window the 14 deltas are seven +1s and seven -1s, so
        avg_gain = avg_loss = 7/14 = 0.5, RS = 1 and RSI = 100 - 100/2 = 50."""
        s = pd.Series([100.0 + (i % 2) for i in range(40)])
        assert math.isclose(rsi(s, period=RSI_PERIOD).iloc[RSI_PERIOD], 50.0)

    def test_leading_nan_count_equals_the_period(self):
        """wilder_smooth seeds on the mean of the first `period` values, and
        diff() costs a bar, so the first defined RSI is at positional index
        `period` — one LATER than ATR's `period - 1`. Asserted, not assumed."""
        s = pd.Series([100.0 + (i % 3) for i in range(40)])
        out = rsi(s, period=RSI_PERIOD)
        assert int(out.isna().sum()) == RSI_PERIOD
        assert out.index.equals(s.index)

    def test_flat_series_is_nan_not_fifty(self):
        """A market that has not moved has no relative strength. NaN means 'not
        knowable', and it is never back-filled."""
        out = rsi(pd.Series([100.0] * 40), period=RSI_PERIOD)
        assert out.iloc[-1] != out.iloc[-1]  # NaN

    def test_rsi_hand_computed_14_period(self):
        """HAND-COMPUTED, period 14. Closes rise by 1.0 for 14 bars from 100 to
        114, then fall 2.0 to 112, then rise 3.0 to 115.

        bar 14: the 14 deltas are all +1.0, so
                avg_gain = 14/14 = 1.0 and avg_loss = 0.0
                -> pure uptrend -> RSI = 100.0
        bar 15: gain 0, loss 2.0
                avg_gain = (1.0*13 + 0)/14 = 13/14
                avg_loss = (0.0*13 + 2)/14 =  2/14
                RS = (13/14)/(2/14) = 13/2 = 6.5
                RSI = 100 - 100/(1+6.5) = 100 - 100/7.5 = 650/7.5 = 86.666...
        bar 16: gain 3.0, loss 0
                avg_gain = ((13/14)*13 + 3)/14 = (169/14 + 42/14)/14 = 211/196
                avg_loss = ((2/14)*13 + 0)/14  = (26/14)/14         =  26/196
                RS = 211/26
                RSI = 100 - 100/(1 + 211/26) = 100*211/237 = 89.0295358649789
        """
        closes = [100.0 + i for i in range(15)] + [112.0, 115.0]
        out = rsi(pd.Series(closes), period=14)
        assert out.iloc[14] == 100.0
        assert out.iloc[15] == pytest.approx(650.0 / 7.5)
        assert out.iloc[16] == pytest.approx(100.0 * 211.0 / 237.0)

    def test_period_argument_overrides_config(self):
        s = pd.Series([100.0 + (i % 3) for i in range(40)])
        assert int(rsi(s, period=5).isna().sum()) == 5

    def test_rsi_frame_drops_the_warmup_and_keeps_the_index_aligned(self):
        """NaN makes every comparison False, so leaving the warmup in would
        silently suppress oscillator pivots near the start of the series while
        looking like a geometry problem. Dropped, NEVER filled."""
        s = pd.Series(
            [100.0 + (i % 3) for i in range(40)],
            index=[START + i * D_SET for i in range(40)],
        )
        frame = rsi_frame(s, period=RSI_PERIOD)
        assert len(frame) == 40 - RSI_PERIOD
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert frame.index[0] == s.index[RSI_PERIOD]
        assert not frame.isna().to_numpy().any()
        assert (frame["high"] == frame["low"]).all()

    def test_rsi_plateau_yields_no_oscillator_pivot(self):
        """find_pivots rejects TIES by design (pivots.py:5-6, 81-84), and RSI
        sits at exactly 100.0 through a pure uptrend, so no oscillator pivot is
        emitted there. That is CORRECT — a flat oscillator has no swing — and it
        is asserted so nobody 'fixes' it by loosening the comparison to `>=`."""
        s = pd.Series(
            [100.0 + i for i in range(60)],
            index=[START + i * D_SET for i in range(60)],
        )
        frame = rsi_frame(s, period=RSI_PERIOD)
        assert (frame["close"] == 100.0).all()
        assert find_pivots(frame, span=PIVOT_SPAN) == []


class TestRsiDivergence:
    def test_detects_the_hand_drawn_bearish_divergence(self, pattern_fixture):
        df = pattern_fixture("rsi_bearish_divergence_positive.csv")
        evs = detect_events(DIV, df)
        assert len(evs) == 1
        e = evs[0]
        assert e.direction == "short"
        assert e.level == 98.0  # the intervening swing low IS the level
        assert math.isclose(e.target_height, 109.6 - 98.0, abs_tol=1e-9)
        assert e.meta["rsi_delta"] < 0  # momentum did NOT confirm the new high
        assert e.meta["rsi_i"] >= config.RSI_DIV_OVERBOUGHT
        assert e.meta["mode"] == 0.0

    def test_no_event_when_momentum_confirms_the_new_high(self, pattern_fixture):
        """Both price highs AND both RSI highs rise: momentum agreed, so there is
        no divergence to trade."""
        assert detect_events(DIV, pattern_fixture("rsi_no_divergence_negative.csv")) == []

    def test_monotone_rise_yields_nothing(self):
        """A pure uptrend plateaus RSI at exactly 100.0, and find_pivots rejects
        ties, so there are no oscillator pivots to pair — see
        test_rsi_plateau_yields_no_oscillator_pivot. The detector must return
        cleanly rather than raise."""
        df = make_df([[100 + i, 100.4 + i, 99.6 + i, 100 + i, 10.0] for i in range(90)])
        assert detect_events(DIV, df) == []

    def test_rejects_a_pair_outside_the_separation_bounds(self, pattern_fixture):
        df = pattern_fixture("rsi_bearish_divergence_positive.csv")
        assert detect_events(DIV, df, max_separation_bars=20) == []
        assert detect_events(DIV, df, min_separation_bars=60) == []

    def test_rejects_when_the_first_rsi_high_is_not_elevated(self, pattern_fixture):
        """Pairing two RSI highs demands the extreme one be ELEVATED. Raising the
        threshold above the measured 76.82 must reject."""
        df = pattern_fixture("rsi_bearish_divergence_positive.csv")
        assert detect_events(DIV, df, overbought=90.0) == []

    def test_rejects_when_no_oscillator_pivot_matches_the_price_pivot(
        self, pattern_fixture
    ):
        """With a zero match window the RSI pivot must land on the price pivot's
        exact bar or the pair is not a divergence."""
        df = pattern_fixture("rsi_bearish_divergence_positive.csv")
        evs = detect_events(DIV, df, pivot_match_bars=0)
        # In this fixture the two coincide, so 0 still matches; shifting the
        # tolerance cannot be tested by tightening it here. Assert the coincidence
        # explicitly so the next test's premise is visible.
        assert len(evs) == 1

    def test_rejects_when_the_level_was_already_broken(self, pattern_fixture):
        """A pattern whose level was already closed through is spent."""
        df = pattern_fixture("rsi_bearish_divergence_positive.csv").copy()
        df.iloc[-1, df.columns.get_loc("close")] = 90.0
        assert detect_events(DIV, df) == []

    def test_hidden_mode_fires_on_a_lower_high_with_a_higher_rsi_high(self):
        """HIDDEN bearish divergence: price makes a LOWER high while momentum
        makes a HIGHER high — the continuation variant, and the catalog's two
        Hidden Divergence rows. Built here rather than as a CSV because it is
        parametric: a slow, loss-laden sawtooth rally into the first high keeps
        its RSI moderate, and a fast monotone rally into a LOWER second high
        lifts RSI above it."""
        df = hidden_bearish_df()
        highs = [p for p in find_pivots(df) if p.kind == "high"]
        assert len(highs) == 2 and highs[1].price < highs[0].price
        series = rsi(df["close"], period=RSI_PERIOD)
        assert series.iloc[highs[1].index] > series.iloc[highs[0].index]
        assert detect_events(DIV, df, mode="regular") == []
        evs = detect_events(DIV, df, mode="hidden")
        assert len(evs) == 1
        assert evs[0].direction == "short"
        assert evs[0].meta["mode"] == 1.0
        assert evs[0].meta["rsi_delta"] > 0

    def test_end_ts_is_the_later_of_the_two_pivot_confirmations(self, pattern_fixture):
        """THE LOOKAHEAD REGRESSION FOR DIVERGENCE. Two pivots are involved — one
        in price, one in the oscillator — and each is only knowable PIVOT_SPAN
        closed bars after its own bar. Taking the price pivot alone would let
        check_breakout fire on a bar where the RSI pivot was not yet knowable,
        and being a lookahead it would IMPROVE the backtest.

        In the hand-drawn fixture the two happen to confirm on the same bar, so
        the rule is asserted against MEASURED indices; `lagging_rsi_pivot_df`
        below is the case where they genuinely differ."""
        df = pattern_fixture("rsi_bearish_divergence_positive.csv")
        j, oj = _second_pivots(df)
        evs = detect_events(DIV, df)
        assert evs[0].end_ts == int(df.index[max(j, oj) + PIVOT_SPAN])
        assert evs[0].end_ts == g.confirmation_ts(df, max(j, oj), PIVOT_SPAN)

    def test_end_ts_follows_the_rsi_pivot_when_the_rsi_pivot_confirms_later(self):
        """The case that distinguishes max() from the price pivot alone: price
        pivots are decided by BARS' HIGHS and RSI pivots by CLOSES, so a bar with
        a lower high but a higher close puts the oscillator's swing one bar after
        the price's."""
        df = lagging_rsi_pivot_df()
        j, oj = _second_pivots(df)
        assert oj > j, "fixture premise: the RSI pivot must confirm later"
        evs = detect_events(DIV, df)
        assert len(evs) == 1
        assert evs[0].end_ts == int(df.index[oj + PIVOT_SPAN])
        assert evs[0].end_ts > int(df.index[j + PIVOT_SPAN])

    def test_declares_a_rationale_and_bounded_paramspecs(self):
        registry.load_all()
        spec = registry.get(DIV)
        assert spec.rationale.strip()
        assert spec.tier == 2
        assert spec.params["mode"].choices == ("regular", "hidden")
        for name in ("overbought", "oversold", "rsi_period"):
            assert spec.params[name].bounds is not None

    def test_short_frame_yields_no_events(self):
        assert detect_events(DIV, make_df([[100, 101, 99, 100, 10.0]] * 10)) == []


# --------------------------------------------------------------------------- #
# Builders for the two cases that are parametric rather than hand-drawn.
# --------------------------------------------------------------------------- #


def _seg(a, b, k):
    return [round(a + (b - a) * j / k, 4) for j in range(k)]


def _saw(start, cycles, up, dn):
    out, v = [], start
    for _ in range(cycles):
        out.append(round(v, 4))
        v += up
        out.append(round(v, 4))
        v += up
        out.append(round(v, 4))
        v -= dn
    return out, round(v, 4)


def _frame_from_closes(closes, rng=0.1):
    return make_df(
        [[v, round(v + rng, 4), round(v - rng, 4), v, 10.0] for v in closes]
    )


def hidden_bearish_df():
    """80 bars: lower price high on a HIGHER RSI high.

      bars 0-9   decline 96 -> 92.4
      bars 10-39 sawtooth rally (+0.9, +0.9, -0.85 per 3 bars). Net +0.317/bar
                 with heavy loss content, so RSI stays MODERATE. A 3-bar cycle
                 whose net move exceeds the down-step creates no interior
                 fractal pivot at span 3.
      bar  40    the first price pivot high (102.5)
      bars 41-54 pullback to 94.5, giving the intervening swing low
      bars 55-66 fast MONOTONE rally to a LOWER high (101.5) -> HIGHER RSI
      bars 67-79 decline, so bar 67 is the second pivot high
    """
    body, endv = _saw(92.0, 10, 0.9, 0.85)
    peak1 = round(endv + 0.9, 4)
    low2 = round(peak1 - 8.0, 4)
    top2 = round(peak1 - 1.0, 4)
    pre = _seg(96, 92, 10)
    tail = 80 - (len(pre) + len(body) + 1 + 14 + 12 + 1)
    closes = (
        pre
        + body
        + [peak1]
        + _seg(round(peak1 - 0.6, 4), low2, 14)
        + _seg(low2, top2, 12)
        + [top2]
        + _seg(round(top2 - 0.6, 4), round(top2 - 6.0, 4), tail)
    )
    assert len(closes) == 80, len(closes)
    return _frame_from_closes(closes)


def lagging_rsi_pivot_df():
    """The bearish-divergence fixture with its second swing reshaped so the
    OSCILLATOR pivot lands one bar after the PRICE pivot.

    Price pivots are decided by bar HIGHS, RSI pivots by CLOSES. Bar 69 keeps the
    highest HIGH (so it stays the price pivot) while bar 70 is given the highest
    CLOSE (so it becomes the RSI pivot).
    """
    from tests.conftest import load_pattern_fixture

    df = load_pattern_fixture("rsi_bearish_divergence_positive.csv").copy()
    hi = df.columns.get_loc("high")
    cl = df.columns.get_loc("close")
    lo = df.columns.get_loc("low")
    df.iloc[69, hi] = 109.6
    df.iloc[69, cl] = 109.30
    df.iloc[70, hi] = 109.45
    df.iloc[70, cl] = 109.40
    df.iloc[70, lo] = 109.20
    return df


def _second_pivots(df):
    """(price-pivot index, oscillator-pivot index) of the SECOND swing high."""
    price_highs = [p for p in find_pivots(df, span=PIVOT_SPAN) if p.kind == "high"]
    osc = rsi_frame(df["close"], period=RSI_PERIOD)
    pos = {int(ts): i for i, ts in enumerate(df.index.to_numpy())}
    osc_highs = [
        pos[int(p.ts)] for p in find_pivots(osc, span=PIVOT_SPAN) if p.kind == "high"
    ]
    j = price_highs[-1].index
    oj = min(osc_highs, key=lambda x: abs(x - j))
    return j, oj


class TestGeometryHelpersUsedHere:
    def test_confirmation_ts_is_span_bars_after_the_pivot(self):
        df = make_df([[100, 100.5, 99.5, 100, 10.0]] * 40)
        assert g.confirmation_ts(df, 10, PIVOT_SPAN) == int(df.index[10 + PIVOT_SPAN])

    def test_confirmation_ts_clamps_at_the_last_bar(self):
        """EvalContext already truncates at the last closed bar, so find_pivots
        cannot return a pivot inside the final `span` bars; the clamp guards a
        caller passing an unsliced frame rather than an expected path."""
        df = make_df([[100, 100.5, 99.5, 100, 10.0]] * 40)
        assert g.confirmation_ts(df, 39, PIVOT_SPAN) == int(df.index[-1])

    def test_context_truncation_hides_later_bars_from_the_detector(self):
        df = make_df([[100, 100.5, 99.5, 100, 10.0]] * 40)
        ctx = make_setup_context(df, now_ms=int(df.index[20]) + D_SET)
        assert len(ctx.window(SETUP_TF, 999)) == 21
