"""
Geometry fixture tests for plugins/detectors/reversal.py.

DETECTION IS NOT EDGE (contract §9). These tests establish only that the four
reversal detectors find the geometry their names claim, on bars built so the
answer is known BY CONSTRUCTION. Whether any of them makes money is a separate
question, answered by `cli.py detector-report` and recorded in
.claude/PRPs/reports/phase8-detector-edge-report.md.

Every rejection test's docstring names the real failure mode it prevents, in the
style of tests/test_signals.py:227-234.
"""

import math

import pytest

from tests.conftest import detect_events, make_setup_context
from tests.test_signals import make_df, path_df
from trading_bot import config
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import registry
from trading_bot.signals.patterns import detect_patterns
from trading_bot.signals.pivots import Pivot, find_pivots

# Tier constants come from config, never hardcoded, so a future tier shift cannot
# leave these tests green on the old timeframes (contract §8).
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
PIVOT_SPAN = config.PIVOT_SPAN
START = 1_700_000_000_000

HS = "detector.head-and-shoulders"
IHS = "detector.inverse-head-and-shoulders"
DT = "detector.double-top"
DB = "detector.double-bottom"

# The v0.3.2 prior-trend gate (D1) fails CLOSED on insufficient history, and
# its bounds are deliberately floored so no ParamSpec override can neutralise
# it (a correctness gate's bound IS its safety envelope — param_jitter.py
# samples every declared ParamSpec inside its bounds, so a reachable "off"
# value would let the evolution engine re-discover the exact false positives
# this gate exists to remove; see graph.py:141-148 on RegimeGate for the same
# argument). So tests below that exercise UNRELATED mechanics (neckline
# placement, end_ts, noise-pivot pairing) on short `path_df` fixtures that
# predate this gate cannot disable it — they are given real runway instead: a
# genuine, monotonic prior move prepended before the original anchors, long
# and steep enough (`RUNWAY_UP`/`RUNWAY_DOWN`, `pivot_span`-worth of segments
# so the fixture clears the default 20-bar lookback exactly) to satisfy D1 at
# its DEFAULT strength. A monotonic run introduces no new pivots (verified:
# `find_pivots` over `RUNWAY_UP + anchors` adds none), so it changes nothing
# about the geometry the test is actually checking — only how much history
# precedes it.
RUNWAY_UP = [40, 55, 70]  # advance into the first high, 3 segments = 15 bars
RUNWAY_DOWN = [190, 160, 130]  # decline into the first low, mirrored


@pytest.fixture(autouse=True)
def _clear_framework_caches():
    """Drop the framework's memos around EVERY test.

    Mirrors tests/test_backtest.py's engine.clear_caches() discipline: test
    isolation must not depend on a content fingerprint being right (contract §8).
    """
    fcontext.clear_caches()
    yield
    fcontext.clear_caches()


# --------------------------------------------------------------------------- #
# The seven H&S fixtures from tests/test_signals.py, re-pointed at the plug-in.
# --------------------------------------------------------------------------- #

# (name, df-builder, explicit pivots or None). Each mirrors one case in
# tests/test_signals.py::TestHeadAndShoulders so parity is checked against the
# SAME shapes the legacy geometry is pinned on.
def _hs_cases():
    flat40 = make_df([[100, 100, 100, 100, 10.0]] * 40)
    flat60 = make_df([[100, 100, 100, 100, 10.0]] * 60)
    run_pivots = [
        Pivot(index=5, ts=int(flat40.index[5]), price=100.0, kind="high"),
        Pivot(index=8, ts=int(flat40.index[8]), price=99.0, kind="high"),  # noise
        Pivot(index=12, ts=int(flat40.index[12]), price=95.0, kind="low"),
        Pivot(index=18, ts=int(flat40.index[18]), price=106.0, kind="high"),
        Pivot(index=24, ts=int(flat40.index[24]), price=95.5, kind="low"),
        Pivot(index=30, ts=int(flat40.index[30]), price=100.4, kind="high"),
    ]
    same_bar_pivots = [
        Pivot(index=5, ts=int(flat40.index[5]), price=100.0, kind="high"),
        Pivot(index=5, ts=int(flat40.index[5]), price=95.0, kind="low"),
        Pivot(index=18, ts=int(flat40.index[18]), price=106.0, kind="high"),
        Pivot(index=24, ts=int(flat40.index[24]), price=95.5, kind="low"),
        Pivot(index=30, ts=int(flat40.index[30]), price=100.4, kind="high"),
    ]
    overlap = [
        (5, 100.0, "high"), (10, 95.0, "low"), (16, 106.0, "high"),
        (22, 95.5, "low"), (28, 100.4, "high"), (34, 95.2, "low"),
        (40, 107.0, "high"), (46, 95.6, "low"), (50, 100.2, "high"),
    ]
    overlap_pivots = [
        Pivot(index=i, ts=int(flat60.index[i]), price=p, kind=k) for i, p, k in overlap
    ]
    return [
        ("hs_short", path_df([90, 100, 95, 106, 95.5, 100.4, 97]), None),
        ("inverse_hs_long", path_df([110, 100, 105, 94, 104.5, 99.6, 103]), None),
        ("asymmetric_shoulders", path_df([90, 100, 95, 106, 95.5, 104.5, 96, 94]), None),
        ("no_head_prominence", path_df([90, 100, 95, 100.8, 95.5, 100.4, 96, 94]), None),
        ("stale", path_df([90, 100, 95, 106, 95.5, 100.4, 96, 95, 94.5, 94.2, 94]), None),
        ("broken_neckline", path_df([90, 100, 95, 106, 95.5, 100.4, 96, 94]), None),
        ("broken_inverse_neckline", path_df([110, 100, 105, 94, 104.5, 99.6, 104, 106]), None),
        ("through_same_kind_run", flat40, run_pivots),
        ("shoulder_and_trough_same_bar", flat40, same_bar_pivots),
        ("overlapping_windows", flat60, overlap_pivots),
    ]


def _legacy_hs(df, pivots, kind):
    return [c for c in detect_patterns(df, pivots) if c.kind == kind]


def _plugin_hs(df, pivots, key):
    """Run the plug-in, forcing the pivot list when a case supplies one.

    A detector may only read bars through `EvalContext`, so it computes its own
    pivots. Three of the ten legacy cases inject a hand-made pivot list that
    `find_pivots` would never return from a flat frame, so for those the pivot
    source is monkey-patched at the module the detector imports it from. That is
    a test-harness detail, not a hole in the contract: the detector still
    receives only an EvalContext.
    """
    from trading_bot.plugins.detectors import reversal

    if pivots is None:
        return detect_events(key, df)
    original = reversal.find_pivots
    try:
        reversal.find_pivots = lambda frame, span=None: list(pivots)
        return detect_events(key, df)
    finally:
        reversal.find_pivots = original


class TestHeadAndShouldersParity:
    """At ParamSpec defaults the refined detector must reproduce
    signals/patterns.py EXACTLY.

    Phase 3's plugins/detectors/legacy_patterns.py wraps the same geometry and
    Phase 3's parity gate freezes it, so a silent change here would mean two
    plug-ins disagreeing about one shape. Parity is what makes the three
    refinements — and not the base shape — the thing under test.
    """

    @pytest.mark.parametrize("name,df,pivots", _hs_cases(), ids=[c[0] for c in _hs_cases()])
    def test_bearish_matches_legacy(self, name, df, pivots):
        legacy = _legacy_hs(df, pivots, "head-and-shoulders")
        new = _plugin_hs(df, pivots, HS)
        assert [
            (c.kind, c.direction, c.breakout_level, c.target_height, c.start_ts, c.end_ts)
            for c in legacy
        ] == [
            (e.kind, e.direction, e.level, e.target_height, e.start_ts, e.end_ts)
            for e in new
        ]

    @pytest.mark.parametrize("name,df,pivots", _hs_cases(), ids=[c[0] for c in _hs_cases()])
    def test_bullish_matches_legacy(self, name, df, pivots):
        legacy = _legacy_hs(df, pivots, "inverse-head-and-shoulders")
        new = _plugin_hs(df, pivots, IHS)
        assert [
            (c.kind, c.direction, c.breakout_level, c.target_height, c.start_ts, c.end_ts)
            for c in legacy
        ] == [
            (e.kind, e.direction, e.level, e.target_height, e.start_ts, e.end_ts)
            for e in new
        ]

    def test_positive_case_is_actually_positive(self):
        """A parity test over ten empty lists would pass vacuously."""
        evs = detect_events(HS, path_df([90, 100, 95, 106, 95.5, 100.4, 97]))
        assert len(evs) == 1
        assert evs[0].level == 95.0  # the stricter (lower) trough
        assert math.isclose(evs[0].target_height, 11.0)
        assert evs[0].meta["head"] == 106.0
        # Pivot timestamps for the replay chart (v0.3.2): additive to meta,
        # paired with the price keys above by the `_ts` suffix.
        assert evs[0].meta["head_ts"] > evs[0].meta["left_shoulder_ts"]
        assert evs[0].meta["right_shoulder_ts"] > evs[0].meta["head_ts"]

    def test_inverse_positive_case_is_actually_positive(self):
        evs = detect_events(IHS, path_df([110, 100, 105, 94, 104.5, 99.6, 103]))
        assert len(evs) == 1
        assert evs[0].level == 105.0  # the stricter (higher) peak
        assert math.isclose(evs[0].target_height, 11.0)
        assert evs[0].meta["head_ts"] > evs[0].meta["left_shoulder_ts"]
        assert evs[0].meta["right_shoulder_ts"] > evs[0].meta["head_ts"]


class TestHeadAndShouldersRefinements:
    """Each refinement is a no-op at its default and changes the outcome when set."""

    DF = path_df([90, 100, 95, 106, 95.5, 100.4, 97])

    def test_volume_taper_default_is_a_no_op(self):
        assert len(detect_events(HS, self.DF, volume_taper=False)) == 1

    def test_volume_taper_rejects_a_rising_right_shoulder_volume(self):
        """The classical distribution tell: the right shoulder should be made on
        LESS volume than the left. path_df gives every bar volume 10, so the
        ratio is exactly 1.0 and a strict `<` must reject."""
        assert detect_events(HS, self.DF, volume_taper=True) == []

    def test_volume_taper_ratio_is_recorded_even_when_it_does_not_gate(self):
        """A measurement that only exists when it is switched on cannot be used
        to decide whether switching it on is worthwhile."""
        evs = detect_events(HS, self.DF, volume_taper=False)
        assert evs[0].meta["volume_taper_ratio"] == 1.0

    def test_time_symmetry_default_is_a_no_op(self):
        assert len(detect_events(HS, self.DF, time_symmetry_tol=1.0)) == 1

    def test_time_symmetry_rejects_a_lopsided_shape(self):
        """A 'head & shoulders' whose right leg is many times the left is not the
        classical shape; path_df's legs are equal, so a tolerance of 0.0 must
        still accept, and a lopsided purpose-built shape must not."""
        assert len(detect_events(HS, self.DF, time_symmetry_tol=0.0)) == 1
        # Left leg 12 bars, right leg 24 bars over a 36-bar span: asymmetry 1/3.
        # The last pivot is at index 56 of 60 so the shape is still FRESH — a
        # staleness rejection here would pass for the wrong reason.
        flat = make_df([[100, 100, 100, 100, 10.0]] * 60)
        pivots = [
            Pivot(index=20, ts=int(flat.index[20]), price=100.0, kind="high"),
            Pivot(index=26, ts=int(flat.index[26]), price=95.0, kind="low"),
            Pivot(index=32, ts=int(flat.index[32]), price=106.0, kind="high"),
            Pivot(index=44, ts=int(flat.index[44]), price=95.5, kind="low"),
            Pivot(index=56, ts=int(flat.index[56]), price=100.4, kind="high"),
        ]
        lopsided = _plugin_hs(flat, pivots, HS)
        assert len(lopsided) == 1
        assert math.isclose(lopsided[0].meta["time_asymmetry"], 12 / 36)
        from trading_bot.plugins.detectors import reversal

        original = reversal.find_pivots
        try:
            reversal.find_pivots = lambda frame, span=None: list(pivots)
            assert detect_events(HS, flat, time_symmetry_tol=0.2) == []
        finally:
            reversal.find_pivots = original

    def test_sloped_neckline_default_is_a_no_op(self):
        assert len(detect_events(HS, self.DF, neckline_sloped=False)) == 1

    def test_sloped_neckline_moves_the_level_off_the_stricter_trough(self):
        """With troughs at 95 and 95.5 the flat neckline is min() = 95.0; the
        sloped line through both, extrapolated to the last pivot's bar, is HIGHER
        than either trough — a different, less conservative trigger. Whether that
        is better is the edge report's question, not this test's."""
        sloped = detect_events(HS, self.DF, neckline_sloped=True)
        assert len(sloped) == 1
        assert sloped[0].meta["neckline_sloped"] == 1.0
        assert sloped[0].level > 95.0
        assert sloped[0].target_height < 11.0  # head - a higher neckline


# --------------------------------------------------------------------------- #
# Double top / double bottom.
# --------------------------------------------------------------------------- #


def _mult_path_df(flat_bars, steps, start=100.0, volume=10.0):
    """Build an O=H=L=C series: `flat_bars` bars at `start`, then each
    `(mult, count)` in `steps` multiplies the running price `count` times.

    Mirrors the plan's evidence-fixture recipes verbatim (e.g. "15 flat bars,
    then p *= 1.02 x 20"), which `path_df`'s anchor-interpolation cannot
    express — these fixtures are defined by a compounding step, not a straight
    line between two prices.
    """
    p = start
    vals = [start] * flat_bars
    for mult, count in steps:
        for _ in range(count):
            p *= mult
            vals.append(p)
    return make_df([[v, v, v, v, volume] for v in vals])


class TestDoubleBottomCorrectnessGates:
    """v0.3.2 (code review 2026-07-29): D1/D2/D3, reproduced from the plan's
    evidence fixtures. Each fired a `double-bottom` on master; each must be
    silent here."""

    def test_rejects_straight_line_rally_with_no_prior_decline(self):
        """D1. A +49% straight-line rally followed by an ordinary
        consolidation is not a reversal of anything: the 'first low' sits
        inside an still-uptrending market, not at the bottom of a decline.
        Fired `double-bottom` at level 149.32 on master with no prior-trend
        gate."""
        df = _mult_path_df(
            15,
            [(1.02, 20), (0.985, 5), (1.0135, 6), (0.9875, 6), (1.004, 4)],
        )
        assert detect_events(DB, df) == []

    def test_rejects_a_lower_low_between_the_two_lows(self):
        """D2. Lows at 88.58 and 89.33 pass tolerance, but a 76.35 low
        BETWEEN them (13.8% below both) means neither is the pattern's real
        extreme — a third, lower low exists. Fired `double-bottom` at level
        101.12, separation_bars 28, on master because the dominance check only
        looked at the pivot list, which never confirmed the spike as a pivot."""
        df = _mult_path_df(
            12,
            [(0.98, 6), (1.02, 4), (0.975, 9), (1.0285, 10), (0.9755, 5), (1.002, 4)],
        )
        assert detect_events(DB, df) == []

    def test_rejects_a_trough_bump_that_is_really_a_trend_leg(self):
        """D3. Lows at 81.67 and 81.09 with a +39% / 32-bar rally between them
        is a range with a big swing in it, not a W's modest middle bounce.
        Fired `double-bottom` at level 113.88, target_height 32.50, on master
        because `min_trough_depth` has no ceiling."""
        df = _mult_path_df(
            12,
            [(0.975, 8), (1.021, 16), (0.979, 16), (1.003, 4)],
        )
        assert detect_events(DB, df) == []

    def test_genuine_double_bottom_still_fires(self):
        """A regression that silently detects nothing is worse than the bug:
        prior downtrend into the first low, two equal lows, a modest bump
        between them, and nothing lower in between must still fire, with sane
        level / target_height / meta."""
        df = _mult_path_df(
            5,
            [(0.985, 20), (1.03, 4), (0.9705, 4), (1.01, 4)],
        )
        evs = detect_events(DB, df)
        assert len(evs) == 1
        e = evs[0]
        assert e.direction == "long"
        assert e.kind == "double-bottom"
        assert e.level > 0
        assert e.target_height > 0
        assert e.meta["peak_a"] > 0 and e.meta["peak_b"] > 0
        assert math.isclose(e.meta["peak_a"], e.meta["peak_b"], rel_tol=0.02)
        # The pivot timestamps the replay chart needs to place the W on the
        # x-axis: additive, not a substitute for the price keys above.
        assert e.meta["peak_a_ts"] == float(e.start_ts)
        assert e.meta["peak_b_ts"] > e.meta["trough_ts"] > e.meta["peak_a_ts"]

    def test_rejects_when_there_is_insufficient_history_for_the_lookback(self):
        """FAIL CLOSED, not clamped-open. This is otherwise a valid double
        bottom (nothing lower in between, modest bump) but its first low sits
        only 5 bars into the frame — short of the default 20-bar
        prior_trend_lookback_bars. 'Trend unknown' must reject, not silently
        measure over a shortened window that could read as confirmed. The
        same shape fires once given a lookback the frame can actually support."""
        df = path_df([110, 95, 104, 94.5, 101])  # first low pivot at index 5
        assert detect_events(DB, df) == []
        # A lookback the frame can support (5 bars, at the ParamSpec's own
        # floor) plus the gate's minimum non-zero strength (0.01, its floor —
        # not 0.0, which the bounds no longer permit) is still a real decline
        # here (110 -> 95 is 13.6%), so this proves the earlier rejection was
        # specifically about insufficient history, not some other gate.
        assert (
            detect_events(
                DB, df, prior_trend_lookback_bars=5, prior_trend_min_move=0.01
            )
            != []
        )


class TestDoubleTopCorrectnessGates:
    """Symmetric coverage: the same three gates, mirrored for double-top."""

    def test_rejects_straight_line_decline_with_no_prior_advance(self):
        """D1 mirrored. A straight-line decline followed by a consolidation
        is not a reversal of anything: the 'first high' sits inside a still
        down-trending market."""
        df = _mult_path_df(
            15,
            [(0.98, 20), (1.015, 5), (0.9865, 6), (1.0125, 6), (0.996, 4)],
        )
        assert detect_events(DT, df) == []

    def test_rejects_a_higher_high_between_the_two_highs(self):
        """D2 mirrored. A spike ABOVE both highs, between them, that never
        confirmed as a fractal pivot must still disqualify the pair — checked
        against `df['high']` over the bar range, not the pivot list."""
        df = _mult_path_df(
            12,
            [(1.02, 6), (0.98, 4), (1.025, 9), (0.9715, 10), (1.0245, 5), (0.998, 4)],
        )
        assert detect_events(DT, df) == []

    def test_rejects_a_peak_bump_that_is_really_a_trend_leg(self):
        """D3 mirrored. A deep, wide sell-off between the two highs is a
        trend leg, not an M's modest middle dip."""
        df = _mult_path_df(
            12,
            [(1.025, 8), (0.979, 16), (1.021, 16), (0.997, 4)],
        )
        assert detect_events(DT, df) == []

    def test_genuine_double_top_still_fires(self):
        """Prior uptrend into the first high, two equal highs, a modest dip
        between them, nothing higher in between: must still fire."""
        df = _mult_path_df(
            5,
            [(1.015, 20), (0.97, 4), (1.0305, 4), (0.99, 4)],
        )
        evs = detect_events(DT, df)
        assert len(evs) == 1
        e = evs[0]
        assert e.direction == "short"
        assert e.kind == "double-top"
        assert e.level > 0
        assert e.target_height > 0
        assert e.meta["peak_a"] > 0 and e.meta["peak_b"] > 0
        assert math.isclose(e.meta["peak_a"], e.meta["peak_b"], rel_tol=0.02)
        assert e.meta["peak_a_ts"] == float(e.start_ts)
        assert e.meta["peak_b_ts"] > e.meta["trough_ts"] > e.meta["peak_a_ts"]


class TestDoubleTop:
    def test_detects_m_shape_with_neckline_and_measured_move(self):
        """RUNWAY_UP prepended so the D1 prior-trend gate — floored so it
        cannot be tuned off — is satisfied for real, at its default strength,
        rather than bypassed; it adds no pivots, so the M's own geometry
        (level, target, separation) is exactly what a bare `path_df` would
        have produced."""
        evs = detect_events(DT, path_df(RUNWAY_UP + [90, 105, 96, 105.5, 99]))
        assert len(evs) == 1
        e = evs[0]
        assert e.direction == "short"
        assert e.level == 96.0  # the intervening trough IS the neckline
        assert math.isclose(e.target_height, (105.0 + 105.5) / 2 - 96.0)
        assert e.meta["separation_bars"] == 10.0

    def test_rejects_negligible_intervening_dip(self):
        """Two highs within tolerance and a 0.6% dip between them is a FLAT
        RANGE, not an M. The bruteforce donor has no such test, which is why it
        fires on ranges."""
        assert detect_events(DT, path_df([90, 105, 104.6, 105.5, 99])) == []

    def test_rejects_peaks_outside_the_height_tolerance(self):
        """105 and 112 differ by 6.3%, far outside DOUBLE_TOLERANCE = 2%: those
        are two unrelated peaks."""
        assert detect_events(DT, path_df([90, 105, 96, 112, 99])) == []

    def test_rejects_a_pair_beyond_max_gap(self):
        """An unrelated high from months ago must not pair with today's."""
        df = path_df([90, 105, 96, 105.5, 99])
        assert detect_events(DT, df, max_gap_bars=5) == []

    def test_rejects_a_pair_closer_than_min_separation(self):
        """Two highs three bars apart are one peak seen twice, not a double top."""
        df = path_df([90, 105, 96, 105.5, 99])
        assert detect_events(DT, df, min_separation_bars=20) == []

    def test_noise_pivot_high_between_the_tops_does_not_hide_the_shape(self):
        """A minor high between the two tops makes the pivot list
        high/low/HIGH/low/high. Pairing only CONSECUTIVE pivot highs (the
        donor's `zip(idx, idx[1:])` at bruteforce/indicators.py:527) misses this;
        iterating every ORDERED pair finds it."""
        df = path_df(RUNWAY_UP + [90, 105, 96, 100, 97, 105.5, 99])
        kinds = [(p.index, p.kind) for p in find_pivots(df)]
        # Same five-pivot sequence as the bare fixture — the runway prefix
        # is monotonic and confirmed (by construction) to add no pivots.
        assert [k for _, k in kinds] == ["high", "low", "high", "low", "high"]
        evs = detect_events(DT, df)
        assert len(evs) == 1
        assert evs[0].level == 96.0  # the MOST extreme intervening trough
        assert evs[0].meta["separation_bars"] == 20.0

    def test_end_ts_is_the_confirmation_bar_not_the_pivot_bar(self):
        """THE LOOKAHEAD REGRESSION. find_pivots returns index = t for a pivot at
        bar t, but the pivot is only knowable at t + PIVOT_SPAN
        (pivots.py:9-12). Reading the pivot's own bar as the confirmation bar
        gives check_breakout a PIVOT_SPAN-bar head start — a lookahead that
        IMPROVES every backtest and so does not look like a bug."""
        df = path_df(RUNWAY_UP + [90, 105, 96, 105.5, 99])
        second = [p for p in find_pivots(df) if p.kind == "high"][-1]
        evs = detect_events(DT, df)
        assert evs[0].end_ts == int(df.index[second.index + PIVOT_SPAN])
        assert evs[0].end_ts > int(df.index[second.index])

    def test_rejects_when_the_neckline_was_already_broken(self):
        """A pattern whose level was already closed through is spent: a later
        crossing is a retest re-break, not the breakout."""
        df = path_df([90, 105, 96, 105.5, 99, 94])
        assert detect_events(DT, df) == []


class TestDoubleBottom:
    def test_detects_w_shape(self):
        """RUNWAY_DOWN mirrors RUNWAY_UP: a genuine decline into the first
        low, satisfying D1 at its default strength rather than bypassing it."""
        evs = detect_events(DB, path_df(RUNWAY_DOWN + [110, 95, 104, 94.5, 101]))
        assert len(evs) == 1
        e = evs[0]
        assert e.direction == "long"
        assert e.level == 104.0  # the intervening peak
        assert math.isclose(e.target_height, 104.0 - (95.0 + 94.5) / 2)

    def test_rejects_negligible_intervening_bump(self):
        assert detect_events(DB, path_df([110, 95, 95.4, 94.5, 101])) == []

    def test_rejects_when_the_level_was_already_broken(self):
        assert detect_events(DB, path_df([110, 95, 104, 94.5, 101, 106])) == []


class TestSharedDiscipline:
    def test_short_frame_yields_no_events_and_no_exception(self):
        df = make_df([[100, 101, 99, 100, 10.0]] * (2 * PIVOT_SPAN))
        for key in (HS, IHS, DT, DB):
            assert detect_events(key, df) == []

    def test_flat_series_has_no_pivots_and_therefore_no_events(self):
        df = make_df([[100, 100, 100, 100, 10.0]] * 60)
        for key in (HS, IHS, DT, DB):
            assert detect_events(key, df) == []

    def test_dedupe_collapses_two_events_of_one_kind_and_direction(self):
        """Overlapping windows of one pattern type describe ONE setup."""
        df = path_df([90, 105, 96, 105.5, 97, 105.2, 99])
        evs = detect_events(DT, df)
        assert len({(e.kind, e.direction) for e in evs}) == len(evs) <= 1 or len(evs) == 1

    def test_every_detector_declares_a_rationale_and_paramspecs(self):
        registry.load_all()
        for key in (HS, IHS, DT, DB):
            spec = registry.get(key)
            assert spec.rationale.strip()
            assert spec.params
            assert spec.tier in (1, 2)

    def test_detector_reads_bars_only_through_the_context(self):
        """A detector must never receive `conn` or an untruncated frame. Bars
        after now_ms are invisible, so a shape completed later cannot be seen."""
        df = path_df([90, 105, 96, 105.5, 99])
        # now_ms at the second-to-last bar's close hides the final bar entirely.
        ctx = make_setup_context(df, now_ms=int(df.index[-2]) + D_SET)
        assert len(ctx.frame(SETUP_TF)) == len(df) - 1
        assert not hasattr(ctx, "conn")
