"""
Geometry fixture tests for plugins/detectors/wyckoff_events.py.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH, restated because it is the whole
point of stated assumptions A1/A2: they validate THE RULE — a probe beyond a
tight trailing range that closes back inside on elevated volume — by
construction. They do NOT, and cannot, validate THE CONCEPT: nothing here shows
that such a bar marks Wyckoff accumulation or distribution, because there is no
agreed numeric definition of those and no labelled data to score against.
`test_docstring_states_the_non_claim` pins that honesty requirement itself.
"""

import math

import pytest

from tests.conftest import detect_events
from tests.test_signals import make_df
from trading_bot import config
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import registry
from trading_bot.plugins.detectors import wyckoff_events

SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
RANGE_BARS = config.WYCKOFF_RANGE_BARS
VOLUME_LOOKBACK = config.VOLUME_LOOKBACK

SPRING = "detector.wyckoff-spring"
UPTHRUST = "detector.wyckoff-upthrust"


@pytest.fixture(autouse=True)
def _clear_framework_caches():
    fcontext.clear_caches()
    yield
    fcontext.clear_caches()


def range_then(probe, *, n=RANGE_BARS, high=102.0, low=98.0, close=100.0, volume=10.0):
    """`n` flat bars inside [low, high] at volume `volume`, then one probe bar.

    The range bar's close sits strictly inside the range, so the range floor has
    never been ACCEPTED by a close — which is what condition 2 tests.
    """
    rows = [[close, high, low, close, volume]] * n
    rows.append(list(probe))
    return make_df(rows)


class TestSpring:
    def test_detects_a_probe_that_closes_back_inside_on_high_volume(self):
        """Range [98, 102] over 30 bars at volume 10; probe low 97.0 (2.04% below
        the 98.0 floor, against WYCKOFF_PROBE_MIN_PCT = 0.3%), close 99.5 back
        inside, volume 30 = 3.0x the trailing mean of 10."""
        df = range_then([99.0, 99.6, 97.0, 99.5, 30.0])
        evs = detect_events(SPRING, df)
        assert len(evs) == 1
        e = evs[0]
        assert e.direction == "long"
        assert e.level == 98.0  # the range floor
        assert math.isclose(e.target_height, 4.0)  # the range width, a STATED convention
        assert math.isclose(e.meta["volume_ratio"], 3.0)
        assert e.start_ts == int(df.index[0])
        assert e.end_ts == int(df.index[-1])  # the probe IS the latest bar

    def test_rejects_a_probe_that_does_not_reclaim(self):
        """Closing at 97.5, BELOW the floor, is a breakdown rather than a spring:
        the level was not defended. `close > lo` is STRICT — a close exactly at
        support is ambiguous, and pivots.py:5-6 sets the house convention that
        ties reject."""
        assert detect_events(SPRING, range_then([99.0, 99.6, 97.0, 97.5, 30.0])) == []

    def test_rejects_a_close_exactly_at_the_floor(self):
        assert detect_events(SPRING, range_then([99.0, 99.6, 97.0, 98.0, 30.0])) == []

    def test_rejects_a_probe_without_a_volume_expansion(self):
        """A probe WITHOUT volume is precisely the ordinary failed breakout this
        detector cannot otherwise exclude, which is why volume is a HARD
        condition here rather than the graded input signals/breakout.py:20-24
        uses."""
        assert detect_events(SPRING, range_then([99.0, 99.6, 97.0, 99.5, 10.0])) == []

    def test_rejects_a_probe_shallower_than_the_minimum(self):
        """97.95 is 0.05% below the floor, inside WYCKOFF_PROBE_MIN_PCT: noise,
        not a probe."""
        assert detect_events(SPRING, range_then([99.0, 99.6, 97.95, 99.5, 30.0])) == []

    def test_rejects_a_range_wider_than_the_ceiling(self):
        """The tight-range filter (scripts/bruteforce/indicators.py:626-628's
        idea) is what gives the probe a defined, SMALL risk. A 22%-wide 'range'
        is a trend, and its width would become the target."""
        df = range_then([99.0, 99.6, 88.0, 99.5, 30.0], high=112.0, low=90.0)
        assert detect_events(SPRING, df) == []

    def test_rejects_when_an_earlier_bar_already_accepted_the_floor(self):
        """A bar that CLOSED on the range floor accepted that price, so the floor
        is not a level being defended and a later probe of it is a continuation
        rather than a spring."""
        df = range_then([99.0, 99.6, 97.0, 99.5, 30.0])
        df = df.copy()
        df.iloc[10, df.columns.get_loc("close")] = 98.0  # closes AT the floor
        assert detect_events(SPRING, df) == []

    def test_rejects_when_an_earlier_probe_already_moved_the_floor(self):
        """A prior identical probe lowers the measured range floor to ITS low, so
        today's shallower probe no longer pierces it. The level was already
        left."""
        df = range_then([99.0, 99.6, 97.0, 99.5, 30.0])
        df = df.copy()
        df.iloc[10, df.columns.get_loc("low")] = 97.0
        assert detect_events(SPRING, df) == []

    def test_short_volume_history_emits_nothing_rather_than_raising(self):
        """A rolling mean over too few bars is NaN, and `NaN >= ratio` is False,
        so the detector correctly emits nothing. Asserted rather than guarded
        against, mirroring tests/test_signals.py:366-371."""
        df = range_then([99.0, 99.6, 97.0, 99.5, 30.0], n=RANGE_BARS, volume=0.0)
        assert detect_events(SPRING, df) == []

    def test_frame_shorter_than_the_range_yields_nothing(self):
        rows = [[100.0, 102.0, 98.0, 100.0, 10.0]] * (RANGE_BARS - 5)
        assert detect_events(SPRING, make_df(rows)) == []


class TestUpthrust:
    def test_detects_a_probe_above_the_range_that_closes_back_inside(self):
        df = range_then([101.0, 103.0, 100.4, 100.5, 30.0])
        evs = detect_events(UPTHRUST, df)
        assert len(evs) == 1
        e = evs[0]
        assert e.direction == "short"
        assert e.level == 102.0  # the range ceiling
        assert math.isclose(e.target_height, 4.0)

    def test_rejects_a_probe_that_holds_above_the_ceiling(self):
        assert detect_events(UPTHRUST, range_then([101.0, 103.0, 100.4, 102.5, 30.0])) == []

    def test_rejects_a_close_exactly_at_the_ceiling(self):
        assert detect_events(UPTHRUST, range_then([101.0, 103.0, 100.4, 102.0, 30.0])) == []

    def test_spring_and_upthrust_do_not_both_fire_on_one_bar(self):
        """They are opposite probes; a bar cannot be both."""
        low_probe = range_then([99.0, 99.6, 97.0, 99.5, 30.0])
        assert detect_events(UPTHRUST, low_probe) == []
        high_probe = range_then([101.0, 103.0, 100.4, 100.5, 30.0])
        assert detect_events(SPRING, high_probe) == []


class TestTheNonClaim:
    def test_docstring_states_the_non_claim(self):
        """PINS THE HONESTY REQUIREMENT ITSELF (stated assumption A2). If a
        refactor drops these words, a reader — or Phase 9 — could take a
        wyckoff-spring event as evidence of accumulation, which is a claim this
        detector never made and cannot make."""
        # Whitespace-normalised: the requirement is the WORDS, not where the
        # docstring happens to wrap them.
        doc = " ".join(wyckoff_events.__doc__.split())
        assert "do **NOT** detect accumulation or distribution" in doc
        assert "cannot distinguish a spring from an ordinary failed breakout" in doc
        assert "no agreed numeric definition" in doc
        assert "no labelled dataset" in doc
        assert "DEFERRED" in doc
        assert "validate THE RULE" in doc

    def test_registry_rationale_states_the_non_claim(self):
        registry.load_all()
        for key in (SPRING, UPTHRUST):
            r = registry.get(key).rationale
            assert "does NOT detect accumulation" in r
            assert "DEFERRED" in r
            assert "no reference implementation" in r

    def test_meta_carries_no_phase_labels(self):
        """Labelling Wyckoff phases A/B/C/D is exactly the claim A2 forbids."""
        evs = detect_events(SPRING, range_then([99.0, 99.6, 97.0, 99.5, 30.0]))
        assert set(evs[0].meta) == {
            "range_high",
            "range_low",
            "range_width_pct",
            "probe_depth_pct",
            "volume_ratio",
        }

    def test_the_structure_is_deferred_in_the_ledger(self):
        """The buildable EVENTS are covered; the STRUCTURE is not."""
        from trading_bot.plugins.detectors.catalog import CATALOG

        wyckoff = {e.pattern: e for e in CATALOG if e.family_no == 9}
        assert wyckoff["Accumulation"].status == "deferred"
        assert wyckoff["Distribution"].status == "deferred"
        assert wyckoff["Spring"].status == "covered"
        assert wyckoff["Upthrust"].status == "covered"


class TestSharedDiscipline:
    def test_reuses_the_existing_volume_lookback_constant(self):
        """No second lookback constant was invented: the trailing volume mean
        uses config.VOLUME_LOOKBACK, which has existed since v0.2.0."""
        df = range_then([99.0, 99.6, 97.0, 99.5, 30.0], n=RANGE_BARS)
        evs = detect_events(SPRING, df)
        # The mean is over the VOLUME_LOOKBACK bars before the probe, all at 10.
        assert VOLUME_LOOKBACK <= RANGE_BARS
        assert math.isclose(evs[0].meta["volume_ratio"], 30.0 / 10.0)

    def test_declares_bounded_paramspecs_and_a_tier(self):
        registry.load_all()
        for key in (SPRING, UPTHRUST):
            spec = registry.get(key)
            assert spec.tier == 1
            assert spec.timeframes == (SETUP_TF,)
            for name in ("range_bars", "range_max_width_pct", "probe_min_pct",
                         "probe_vol_ratio"):
                assert spec.params[name].bounds is not None
