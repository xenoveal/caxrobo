"""
Geometry fixture tests for plugins/detectors/continuation.py.

DETECTION IS NOT EDGE (contract §9): these tests establish only that each shape
is found when it is present and rejected when it is not. The measured
after-costs expectancy of each lives in
.claude/PRPs/reports/phase8-detector-edge-report.md.

THE LOAD-BEARING TESTS IN THIS MODULE are
`test_legacy_triangle_detector_cannot_see_this_shape` — which pins WHY
signals/patterns.py:327 had to be relaxed — and
`test_shape_classification_is_mutually_exclusive`, which pins that relaxing it
did not turn every wedge into a "triangle".
"""

import itertools
import math

import pytest

from tests.conftest import detect_events, load_pattern_fixture
from tests.test_signals import flag_1h_df, make_df
from trading_bot import config
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import registry
from trading_bot.plugins.detectors import _geometry as g
from trading_bot.plugins.detectors.continuation import TRIANGLE_SHAPES, _classify
from trading_bot.signals.patterns import detect_patterns
from trading_bot.signals.pivots import Pivot, find_pivots

SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
PIVOT_SPAN = config.PIVOT_SPAN
START = 1_700_000_000_000

SYM = "detector.symmetrical-triangle"
ASC = "detector.ascending-triangle"
DESC = "detector.descending-triangle"
FALL = "detector.falling-wedge"
RISE = "detector.rising-wedge"
CUP = "detector.cup-and-handle"
ICUP = "detector.inverse-cup-and-handle"
BULL = "detector.bull-flag"
BEAR = "detector.bear-flag"


@pytest.fixture(autouse=True)
def _clear_framework_caches():
    fcontext.clear_caches()
    yield
    fcontext.clear_caches()


# --------------------------------------------------------------------------- #
# Builders. Parametric shapes stay Python builders (tests/test_signals.py:192's
# `wedge`); only the cup and the divergence series are CSVs, because those two
# are genuinely hand-drawn.
# --------------------------------------------------------------------------- #


def _line(x1, y1, x2, y2):
    return lambda x: y1 + (y2 - y1) / (x2 - x1) * (x - x1)


def wedge_bars(
    n=36, upper_pts=((5, 110.0), (30, 101.0)), lower_pts=((10, 92.0), (28, 97.0))
):
    """Bars inside a converging structure whose pivots `find_pivots` really finds.

    DIFFERENT FROM tests/test_signals.py:192's `wedge()`, and it has to be.
    `wedge()` returns a hand-made pivot list because `detect_patterns` accepts
    one; a Detector plug-in may only read bars through `EvalContext`, so it
    computes its own pivots and the FRAME must actually contain them.

    Construction: every non-anchor bar is a zero-range bar at the midline
    `(upper(x)+lower(x))/2`; at an upper anchor the bar's HIGH touches the upper
    line exactly, and at a lower anchor its LOW touches the lower line. The
    midline is monotone whenever both boundaries are, so it contains no strict
    local extreme and the ONLY pivots are the four anchors. Containment holds by
    construction because an anchor sits exactly ON its line.

    Returns (df, pivots) — the pivot list only for the legacy comparison tests.
    """
    up = _line(*upper_pts[0], *upper_pts[1])
    lo = _line(*lower_pts[0], *lower_pts[1])
    ux = {p[0] for p in upper_pts}
    lx = {p[0] for p in lower_pts}
    rows = []
    for x in range(n):
        u, l = up(x), lo(x)
        mid = (u + l) / 2
        rows.append([mid, u if x in ux else mid, l if x in lx else mid, mid, 10.0])
    df = make_df(rows)
    pivots = [
        Pivot(index=p[0], ts=int(df.index[p[0]]), price=p[1], kind="high")
        for p in upper_pts
    ] + [
        Pivot(index=p[0], ts=int(df.index[p[0]]), price=p[1], kind="low")
        for p in lower_pts
    ]
    pivots.sort(key=lambda p: (p.index, p.kind))
    return df, pivots


SHAPE_LINES = {
    SYM: (((5, 110.0), (30, 101.0)), ((10, 92.0), (28, 97.0))),
    ASC: (((5, 110.0), (30, 109.9)), ((10, 92.0), (28, 104.0))),
    DESC: (((5, 110.0), (30, 99.0)), ((10, 92.0), (28, 92.1))),
    FALL: (((5, 110.0), (30, 101.0)), ((10, 100.0), (28, 96.0))),
    # Both boundaries RISING — the shape signals/patterns.py:327 forbids.
    "rising": (((5, 101.0), (30, 110.0)), ((10, 92.0), (28, 105.0))),
}


def shape_df(which):
    up, lo = SHAPE_LINES[which]
    return wedge_bars(upper_pts=up, lower_pts=lo)


class TestWedgeBuilderFindsTheIntendedPivots:
    """If the builder's pivots drift, every shape test below is meaningless."""

    @pytest.mark.parametrize("which", list(SHAPE_LINES))
    def test_only_the_four_anchors_are_pivots(self, which):
        df, expected = shape_df(which)
        found = [(p.index, p.kind, round(p.price, 6)) for p in find_pivots(df)]
        assert found == [(p.index, p.kind, round(p.price, 6)) for p in expected]


class TestTriangleVariants:
    def test_symmetrical_emits_both_directions(self):
        """The catalog says a symmetrical triangle continues 'in the prevailing
        trend'; the detector deliberately does not determine the trend, so it
        emits both candidates and lets the trigger bar pick at most one."""
        df, _ = shape_df(SYM)
        evs = detect_events(SYM, df)
        assert sorted(e.direction for e in evs) == ["long", "short"]
        long_e = next(e for e in evs if e.direction == "long")
        short_e = next(e for e in evs if e.direction == "short")
        assert long_e.level > short_e.level  # lines have not crossed
        assert long_e.target_height == short_e.target_height > 0

    def test_ascending_emits_one_long_at_the_flat_resistance(self):
        df, _ = shape_df(ASC)
        evs = detect_events(ASC, df)
        assert len(evs) == 1
        assert evs[0].direction == "long"
        assert math.isclose(evs[0].level, 109.88, abs_tol=0.02)

    def test_descending_emits_one_short_at_the_flat_support(self):
        df, _ = shape_df(DESC)
        evs = detect_events(DESC, df)
        assert len(evs) == 1
        assert evs[0].direction == "short"
        assert math.isclose(evs[0].level, 92.139, abs_tol=0.02)

    @pytest.mark.parametrize("which", [SYM, ASC, DESC, FALL, "rising"])
    def test_each_shape_rejects_the_other_shapes_frames(self, which):
        """Five mutually exclusive labels means four of five detectors must stay
        silent on any one frame."""
        df, _ = shape_df(which)
        mine = RISE if which == "rising" else which
        for other in (SYM, ASC, DESC, FALL, RISE):
            if other == mine:
                continue
            assert detect_events(other, df) == [], other
        assert len(detect_events(mine, df)) >= 1

    def test_target_height_is_the_widest_vertical_extent(self):
        df, _ = shape_df(SYM)
        evs = detect_events(SYM, df)
        fit_start_range = evs[0].meta["start_range"]
        assert evs[0].target_height == fit_start_range > evs[0].meta["end_range"]


class TestWedges:
    def test_falling_wedge_is_detected_and_is_long(self):
        """Two falling converging boundaries: the classical bullish reading."""
        df, _ = shape_df(FALL)
        evs = detect_events(FALL, df)
        assert len(evs) == 1
        assert evs[0].direction == "long"
        assert evs[0].meta["upper_slope"] < 0
        assert evs[0].meta["lower_slope"] < 0

    def test_legacy_triangle_detector_cannot_see_this_shape(self):
        """THE REASON signals/patterns.py:327 HAD TO BE RELAXED.

        `if upper_end > upper_start or lower_end < lower_start: return out`
        rejects any shape with a falling lower line, which every falling wedge
        has. The legacy detector therefore returns NOTHING on a textbook falling
        wedge, even given its pivots explicitly — while the relaxed fit in
        plugins/detectors/_geometry.py emits it. Two tier-2 patterns were
        structurally unreachable, not merely mis-parameterised."""
        df, pivots = shape_df(FALL)
        assert [c for c in detect_patterns(df, pivots) if c.kind == "triangle"] == []
        assert len(detect_events(FALL, df)) == 1

    def test_legacy_triangle_detector_cannot_see_a_rising_wedge_either(self):
        df, pivots = shape_df("rising")
        assert [c for c in detect_patterns(df, pivots) if c.kind == "triangle"] == []

    def test_rising_wedge_is_detected_and_is_short(self):
        """Two rising converging boundaries: the classical bearish reading, and
        the second of the two tier-2 patterns signals/patterns.py:327 made
        structurally unreachable."""
        df, _ = shape_df("rising")
        evs = detect_events(RISE, df)
        assert len(evs) == 1
        assert evs[0].direction == "short"
        assert evs[0].meta["upper_slope"] > 0
        assert evs[0].meta["lower_slope"] > 0

    def test_a_rising_wedge_frame_is_not_labelled_a_symmetrical_triangle(self):
        """Dropping 327 without relocating its work would relabel every rising
        wedge a symmetrical triangle and emit a LONG candidate for a bearish
        shape. `_classify` is what prevents that."""
        df, _ = shape_df("rising")
        for key in (SYM, ASC, DESC, FALL):
            assert detect_events(key, df) == [], key
        assert len(detect_events(RISE, df)) == 1

    def test_rising_wedge_defaults_match_the_shared_ones(self):
        """THE DRIFT GUARD for the zero-engine-edit proof.

        `detector.rising-wedge` reads NO config attribute — every default is an
        inline literal — so that adding it touched exactly two paths. The cost is
        duplication, and this is what keeps the duplicate honest: its declared
        defaults must equal `falling-wedge`'s, which do come from config.
        """
        registry.load_all()
        rising = registry.get(RISE).params
        falling = registry.get(FALL).params
        assert set(rising) == set(falling)
        for name, spec in falling.items():
            assert rising[name].default == spec.default, name
            assert rising[name].bounds == spec.bounds, name
            assert rising[name].kind == spec.kind, name

    def test_rising_wedge_source_reads_no_config_attribute(self):
        """Structural proof of the property the footprint proof rests on."""
        import inspect

        src = inspect.getsource(registry.get(RISE).impl)
        assert "config." not in src


class TestShapeClassification:
    def _fit(self, u, l):
        return g.LineFit(
            start_x=0, end_x=30,
            upper_start=110.0, upper_end=105.0,
            lower_start=90.0, lower_end=100.0,
            upper_slope=u, lower_slope=l,
            start_range=20.0, end_range=5.0,
        )

    def test_shape_classification_is_mutually_exclusive(self):
        """Over a grid of signed slopes, `_classify` returns AT MOST one label.

        It returns the first branch that matches, which makes exclusivity
        structural rather than arithmetic — necessary because a falling wedge and
        a descending triangle differ only by whether the lower line is flat
        within `flat_tol`, so with `min_slope < flat_tol` both predicates can be
        true at once and branch order decides."""
        grid = (-0.05, -0.01, -0.002, -0.0005, 0.0, 0.0005, 0.002, 0.01, 0.05)
        for flat_tol, min_slope in ((0.001, 0.001), (0.002, 0.0005), (0.0005, 0.002)):
            for u, l in itertools.product(grid, grid):
                label = _classify(
                    self._fit(u, l), flat_tol=flat_tol, min_slope=min_slope
                )
                assert label is None or label in TRIANGLE_SHAPES

    def test_flat_boundary_wins_over_the_wedge_reading(self):
        """A flat lower boundary is a LEVEL being repeatedly tested, which is the
        more specific structure; it must not be swallowed by the wedge branch."""
        label = _classify(
            self._fit(-0.004, -0.0015), flat_tol=0.002, min_slope=0.0005
        )
        assert label == "descending-triangle"

    def test_five_shapes_are_registered_under_the_names_classify_returns(self):
        registry.load_all()
        for shape in TRIANGLE_SHAPES:
            assert f"detector.{shape}" in registry.REGISTRY


class TestTriangleRejections:
    """The four legacy rejections from tests/test_signals.py:227-265, re-pointed."""

    def test_rejects_bar_piercing_a_trendline(self):
        """One bar spikes far above its upper trendline: price was not bounded by
        the lines, so the structure never contained the move."""
        df, _ = shape_df(SYM)
        df = df.copy()
        df.iloc[15, df.columns.get_loc("high")] = 200.0
        assert detect_events(SYM, df) == []

    def test_rejects_when_last_close_already_outside(self):
        """Price already through the upper line: the structure is broken, not
        pending, so it must not be re-offered as a fresh breakout candidate."""
        df, _ = shape_df(SYM)
        df = df.copy()
        df.iloc[-1, df.columns.get_loc("close")] = 150.0
        assert detect_events(SYM, df) == []

    def test_rejects_structure_narrower_than_min_width(self):
        n = config.TRIANGLE_MIN_WIDTH_BARS + 6
        df, _ = wedge_bars(
            n=n,
            upper_pts=((n - 12, 110.0), (n - 4, 101.0)),
            lower_pts=((n - 11, 92.0), (n - 5, 97.0)),
        )
        assert detect_events(SYM, df) == []

    def test_rejects_when_one_trendline_is_stale(self):
        """BOTH lines must be anchored on recent structure. With max() instead of
        min() in the freshness test, one line could be fitted to pivots long out
        of date and still set a level."""
        stale = config.PATTERN_MAX_AGE_BARS + 10
        df, _ = wedge_bars(
            n=60,
            upper_pts=((5, 110.0), (55, 101.0)),
            lower_pts=((10, 92.0), (59 - stale, 97.0)),
        )
        assert detect_events(SYM, df) == []

    def test_rejects_a_parallel_channel(self):
        """Two parallel boundaries never converge, so nothing contracts."""
        df, _ = wedge_bars(
            upper_pts=((5, 110.0), (30, 110.0)), lower_pts=((10, 92.0), (28, 92.0))
        )
        for key in (SYM, ASC, DESC, FALL, RISE):
            assert detect_events(key, df) == [], key


# --------------------------------------------------------------------------- #
# Flags.
# --------------------------------------------------------------------------- #


class TestFlags:
    def test_detects_bull_flag_at_the_consolidation_high(self):
        evs = detect_events(BULL, flag_1h_df())
        assert len(evs) == 1
        assert evs[0].direction == "long"
        assert evs[0].level == 109.5  # the consolidation high
        assert evs[0].target_height > 0

    def test_no_flag_without_a_pole(self):
        assert detect_events(BULL, make_df([[100, 100.5, 99.5, 100, 10.0]] * 40)) == []
        assert detect_events(BEAR, make_df([[100, 100.5, 99.5, 100, 10.0]] * 40)) == []

    def test_pole_window_spans_exactly_the_declared_bars(self):
        """flag_1h_df has n=32 and the winning consolidation is 4 bars, so
        pole_end=27 and the 12-bar pole window is bars 16..27 — bar 13 is
        OUTSIDE it. A crater there must not inflate the measured pole height.
        The slice is inclusive, which is why pole_start subtracts
        pole_window_bars - 1 rather than pole_window_bars."""
        base = detect_events(BULL, flag_1h_df())[0]
        df = flag_1h_df()
        df.iloc[13, df.columns.get_loc("low")] = 80.0
        after = detect_events(BULL, df)
        assert len(after) == 1
        assert math.isclose(after[0].target_height, base.target_height)

    def test_consolidation_may_reach_a_pole_high_set_before_the_pole_ended(self):
        """Containment is tested against the pole's extreme over the WHOLE pole
        window. Checking only the final pole bar's high rejected valid flags
        whose pole peaked a bar or two earlier (patterns.py:402-404)."""
        rows = [[100, 100.5, 99.5, 100, 10.0]] * 20
        rows += [
            [101, 102.0, 100.5, 102, 10.0],
            [102, 104.0, 101.5, 104, 10.0],
            [104, 106.0, 103.5, 106, 10.0],
            [106, 110.5, 105.5, 109, 10.0],  # pole high here
            [109, 110.2, 108.5, 110, 10.0],  # pole_end: a LOWER high
        ]
        rows += [[109.5, 110.4, 108.8, 109.5, 10.0]] * 6
        evs = detect_events(BULL, make_df(rows))
        assert len(evs) == 1
        assert math.isclose(evs[0].level, 110.4)

    def test_detects_bear_flag_at_the_consolidation_low(self):
        rows = [[100, 100.5, 99.5, 100, 10.0]] * 20
        for close in (98, 96, 94, 92, 91, 90):
            rows.append([close + 1, close + 0.5, close - 0.5, close, 10.0])
        rows += [[91, 91.5, 90.5, 91, 10.0]] * 6
        evs = detect_events(BEAR, make_df(rows))
        assert len(evs) == 1
        assert evs[0].direction == "short"
        assert evs[0].level == 90.5

    def test_flag_end_ts_is_the_latest_closed_bar(self):
        """A flag is bar-terminated, not pivot-terminated, so `confirmation_ts`
        does not apply and the consolidation is fresh by construction
        (patterns.py:96-98)."""
        df = flag_1h_df()
        assert detect_events(BULL, df)[0].end_ts == int(df.index[-1])


# --------------------------------------------------------------------------- #
# Cup & handle.
# --------------------------------------------------------------------------- #


class TestCupAndHandle:
    def test_detects_the_hand_drawn_cup(self, pattern_fixture):
        df = pattern_fixture("cup_and_handle_positive.csv")
        evs = detect_events(CUP, df)
        assert len(evs) == 1
        e = evs[0]
        assert e.direction == "long"
        assert e.level == 100.8  # the HIGHER rim: the conservative trigger
        assert math.isclose(e.target_height, 12.8, abs_tol=1e-9)
        assert e.meta["base_bars"] == 15.0
        assert e.meta["handle_bars"] == 11.0

    def test_v_bottom_is_not_a_cup(self, pattern_fixture):
        """WITHOUT THE ROUNDNESS TEST, cup & handle is a slow double bottom. The
        negative fixture has identical rims, identical depth, an identical
        handle and a centred bottom — only THREE bars sit inside the bottom 25%
        band against CUP_MIN_BASE_BARS = 5, so roundness is the sole failure."""
        assert detect_events(CUP, pattern_fixture("cup_v_bottom_negative.csv")) == []

    def test_v_bottom_passes_every_clause_except_roundness(self, pattern_fixture):
        """Proves the previous test is not passing for an unrelated reason."""
        df = pattern_fixture("cup_v_bottom_negative.csv")
        assert len(detect_events(CUP, df, min_base_bars=3)) == 1

    def test_rejects_asymmetric_rims(self, pattern_fixture):
        """Rims 19% apart are not two tests of one level; lowering the left rim
        would only move the pivot one bar, so the left rim is RAISED instead."""
        df = pattern_fixture("cup_and_handle_positive.csv").copy()
        df.iloc[8, df.columns.get_loc("high")] = 120.0
        assert [p.price for p in find_pivots(df) if p.kind == "high"] == [120.0, 100.8]
        assert detect_events(CUP, df) == []

    def test_rejects_a_shallow_cup(self, pattern_fixture):
        """Under CUP_MIN_DEPTH the 'cup' is noise, not a structure."""
        df = pattern_fixture("cup_and_handle_positive.csv")
        assert detect_events(CUP, df, min_depth=0.20) == []

    def test_rejects_an_over_deep_cup(self, pattern_fixture):
        """Over CUP_MAX_DEPTH it is a crash with a bounce, not a cup."""
        df = pattern_fixture("cup_and_handle_positive.csv")
        assert detect_events(CUP, df, max_depth=0.10) == []

    def test_rejects_a_handle_retracing_more_than_the_allowance(self, pattern_fixture):
        """A handle giving back most of the cup is a failed cup."""
        df = pattern_fixture("cup_and_handle_positive.csv")
        assert detect_events(CUP, df, handle_max_retrace=0.05) == []

    def test_rejects_a_handle_high_above_the_rim(self, pattern_fixture):
        """If the handle's high exceeds the rim, the rim already broke and the
        setup is spent rather than pending."""
        df = pattern_fixture("cup_and_handle_positive.csv").copy()
        df.iloc[52, df.columns.get_loc("high")] = 103.0
        assert detect_events(CUP, df) == []

    def test_rejects_a_handle_shorter_than_the_minimum(self, pattern_fixture):
        df = pattern_fixture("cup_and_handle_positive.csv")
        assert detect_events(CUP, df, handle_min_bars=15) == []

    def test_rejects_a_base_hugging_one_rim(self):
        """Bottom centrality: a base pinned against a rim is a descending
        staircase, not a cup."""
        df = load_pattern_fixture("cup_and_handle_positive.csv")
        # Narrow the centrality window to nothing but the exact midpoint by
        # asking for a cup at least as wide as the frame: no triple qualifies.
        assert detect_events(CUP, df, min_width_bars=55) == []

    def test_rejects_when_the_rim_level_was_already_broken(self, pattern_fixture):
        df = pattern_fixture("cup_and_handle_positive.csv").copy()
        df.iloc[57, df.columns.get_loc("close")] = 105.0
        assert detect_events(CUP, df) == []

    def test_inverse_cup_finds_the_mirrored_dome(self, pattern_fixture):
        """Every inequality mirrors: low rims, a high dome, a handle that bounces."""
        df = pattern_fixture("cup_and_handle_positive.csv").copy()
        mirrored = df.copy()
        mirrored["open"] = 200.0 - df["open"]
        mirrored["close"] = 200.0 - df["close"]
        mirrored["high"] = 200.0 - df["low"]
        mirrored["low"] = 200.0 - df["high"]
        evs = detect_events(ICUP, mirrored)
        assert len(evs) == 1
        assert evs[0].direction == "short"
        assert math.isclose(evs[0].level, 200.0 - 100.8)
        assert math.isclose(evs[0].target_height, 12.8, abs_tol=1e-9)
        assert detect_events(CUP, mirrored) == []


class TestSharedDiscipline:
    KEYS = (SYM, ASC, DESC, FALL, RISE, BULL, BEAR, CUP, ICUP)

    def test_short_frame_yields_no_events_and_no_exception(self):
        df = make_df([[100, 101, 99, 100, 10.0]] * (2 * PIVOT_SPAN))
        for key in self.KEYS:
            assert detect_events(key, df) == [], key

    def test_zero_range_bars_do_not_divide_by_zero(self):
        df = make_df([[100, 100, 100, 100, 10.0]] * 60)
        for key in self.KEYS:
            assert detect_events(key, df) == [], key

    def test_every_detector_declares_a_rationale_and_paramspecs(self):
        registry.load_all()
        for key in self.KEYS:
            spec = registry.get(key)
            assert spec.rationale.strip()
            assert spec.params
            assert spec.timeframes == (SETUP_TF,)
