"""Tests for the Phase 3 chart-pattern breakout signal method."""

import math

import pandas as pd

from trading_bot import config
from trading_bot.cli import _signal_command
from trading_bot.data import storage
from trading_bot.signals import setup as setup_mod
from trading_bot.signals.breakout import check_breakout
from trading_bot.signals.patterns import PatternCandidate, detect_patterns
from trading_bot.signals.pivots import Pivot, find_pivots
from trading_bot.signals.setup import Signal, build_signal, current_signals, rank_signals

SYMBOL = "BTCUSDT"
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000  # arbitrary epoch-ms base


def make_df(rows, start=START, interval=D_SET):
    """Build an OHLCV DataFrame from [open, high, low, close, volume] rows."""
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype(int)
    return df.set_index("ts")


def path_df(anchors, step=5, volume=10.0):
    """Linear price path through anchor values, `step` bars between anchors.

    Every bar has open=high=low=close=value so fractal pivots land exactly on
    the anchor extremes (strict comparisons hold on monotone segments).
    """
    values = []
    for a, b in zip(anchors, anchors[1:]):
        for j in range(step):
            values.append(a + (b - a) * j / step)
    values.append(anchors[-1])
    return make_df([[v, v, v, v, volume] for v in values])


class TestFindPivots:
    def test_too_short_series_returns_empty(self):
        df = make_df([[100, 101, 99, 100, 10]] * (2 * config.PIVOT_SPAN))
        assert find_pivots(df) == []

    def test_single_peak_and_trough(self):
        df = path_df([90, 100, 90])
        pivots = find_pivots(df)
        highs = [p for p in pivots if p.kind == "high"]
        assert len(highs) == 1
        assert highs[0].price == 100.0
        assert highs[0].index == 5  # the anchor bar

    def test_flat_ties_are_not_pivots(self):
        df = make_df([[100, 100, 100, 100, 10]] * 20)
        assert find_pivots(df) == []

    def test_no_pivots_within_span_of_edges(self):
        df = path_df([90, 100, 90])
        for p in find_pivots(df):
            assert config.PIVOT_SPAN <= p.index < len(df) - config.PIVOT_SPAN

    def test_pivot_low_detected(self):
        df = path_df([110, 95, 110])
        lows = [p for p in find_pivots(df) if p.kind == "low"]
        assert len(lows) == 1
        assert lows[0].price == 95.0


class TestHeadAndShoulders:
    def test_detects_hs_short(self):
        # P1=100, T1=95, head=106, T2=95.5, P3=100.4, then a confirming tail that
        # stops short of the 95 neckline — a tail through it means the breakout
        # already happened (see test_rejects_already_broken_neckline).
        df = path_df([90, 100, 95, 106, 95.5, 100.4, 97])
        hs = [c for c in detect_patterns(df) if c.kind == "head-and-shoulders"]
        assert len(hs) == 1
        c = hs[0]
        assert c.direction == "short"
        assert c.breakout_level == 95.0  # stricter (lower) trough
        assert math.isclose(c.target_height, 11.0)

    def test_detects_inverse_hs_long(self):
        # Mirror of the above: the tail stops short of the 105 neckline.
        df = path_df([110, 100, 105, 94, 104.5, 99.6, 103])
        inv = [c for c in detect_patterns(df) if c.kind == "inverse-head-and-shoulders"]
        assert len(inv) == 1
        c = inv[0]
        assert c.direction == "long"
        assert c.breakout_level == 105.0  # stricter (higher) peak
        assert math.isclose(c.target_height, 11.0)

    def test_rejects_asymmetric_shoulders(self):
        # Shoulders 100 vs 104.5 differ by ~4.3% > HS_SHOULDER_TOLERANCE (3%).
        df = path_df([90, 100, 95, 106, 95.5, 104.5, 96, 94])
        assert [c for c in detect_patterns(df) if c.kind == "head-and-shoulders"] == []

    def test_rejects_insufficient_head_prominence(self):
        # Head 100.8 exceeds shoulders 100/100.4 by < 1% prominence.
        df = path_df([90, 100, 95, 100.8, 95.5, 100.4, 96, 94])
        assert [c for c in detect_patterns(df) if c.kind == "head-and-shoulders"] == []

    def test_rejects_stale_pattern(self):
        # Long tail after P3 pushes the last pivot beyond PATTERN_MAX_AGE_BARS.
        df = path_df([90, 100, 95, 106, 95.5, 100.4, 96, 95, 94.5, 94.2, 94])
        assert [c for c in detect_patterns(df) if c.kind == "head-and-shoulders"] == []

    def test_rejects_already_broken_neckline(self):
        # Tail closes through the 95 neckline: the breakout is in the past, so a
        # later trigger crossing would be a retest re-break, not the breakout.
        df = path_df([90, 100, 95, 106, 95.5, 100.4, 96, 94])
        assert [c for c in detect_patterns(df) if c.kind == "head-and-shoulders"] == []

    def test_rejects_already_broken_inverse_neckline(self):
        df = path_df([110, 100, 105, 94, 104.5, 99.6, 104, 106])
        assert [
            c for c in detect_patterns(df) if c.kind == "inverse-head-and-shoulders"
        ] == []

    def test_detects_through_same_kind_pivot_run(self):
        # A noise pivot high between the left shoulder and its trough makes the
        # raw pivot list non-alternating (high, high, low, ...). Collapsing runs
        # to their extreme keeps the geometry visible.
        df = make_df([[100, 100, 100, 100, 10.0]] * 40)
        pivots = [
            Pivot(index=5, ts=int(df.index[5]), price=100.0, kind="high"),
            Pivot(index=8, ts=int(df.index[8]), price=99.0, kind="high"),  # noise
            Pivot(index=12, ts=int(df.index[12]), price=95.0, kind="low"),
            Pivot(index=18, ts=int(df.index[18]), price=106.0, kind="high"),
            Pivot(index=24, ts=int(df.index[24]), price=95.5, kind="low"),
            Pivot(index=30, ts=int(df.index[30]), price=100.4, kind="high"),
        ]
        assert [p.kind for p in pivots][:2] == ["high", "high"]  # no 5-window alternates
        hs = [c for c in detect_patterns(df, pivots) if c.kind == "head-and-shoulders"]
        assert len(hs) == 1
        assert hs[0].breakout_level == 95.0
        assert math.isclose(hs[0].target_height, 11.0)  # head 106 - neckline 95

    def test_rejects_shoulder_and_trough_on_same_bar(self):
        # An outside bar can be both a pivot high and a pivot low; a shoulder and
        # its adjacent trough sharing one bar is time-degenerate, not a pattern.
        df = make_df([[100, 100, 100, 100, 10.0]] * 40)
        pivots = [
            Pivot(index=5, ts=int(df.index[5]), price=100.0, kind="high"),
            Pivot(index=5, ts=int(df.index[5]), price=95.0, kind="low"),  # same bar
            Pivot(index=18, ts=int(df.index[18]), price=106.0, kind="high"),
            Pivot(index=24, ts=int(df.index[24]), price=95.5, kind="low"),
            Pivot(index=30, ts=int(df.index[30]), price=100.4, kind="high"),
        ]
        assert [c for c in detect_patterns(df, pivots) if c.kind == "head-and-shoulders"] == []

    def test_dedupes_overlapping_candidates_of_one_kind(self):
        # Two overlapping H&S windows describe one setup, not two signals.
        df = make_df([[100, 100, 100, 100, 10.0]] * 60)
        prices = [
            (5, 100.0, "high"), (10, 95.0, "low"), (16, 106.0, "high"),
            (22, 95.5, "low"), (28, 100.4, "high"), (34, 95.2, "low"),
            (40, 107.0, "high"), (46, 95.6, "low"), (50, 100.2, "high"),
        ]
        pivots = [
            Pivot(index=i, ts=int(df.index[i]), price=p, kind=k) for i, p, k in prices
        ]
        hs = [c for c in detect_patterns(df, pivots) if c.kind == "head-and-shoulders"]
        assert len(hs) == 1


class TestTriangles:
    def test_detects_converging_triangle_both_directions(self):
        df = path_df([100, 110, 90, 106, 93, 103.5, 96, 99])
        tri = [c for c in detect_patterns(df) if c.kind == "triangle"]
        assert sorted(c.direction for c in tri) == ["long", "short"]
        long_c = next(c for c in tri if c.direction == "long")
        short_c = next(c for c in tri if c.direction == "short")
        assert long_c.breakout_level > short_c.breakout_level  # lines not crossed
        assert long_c.target_height == short_c.target_height > 0

    def test_rejects_parallel_channel(self):
        df = path_df([100, 110, 90, 110, 90, 110, 90, 95])
        assert [c for c in detect_patterns(df) if c.kind == "triangle"] == []


def _line(x1, y1, x2, y2):
    """Callable for the line through two points (test-side mirror of the fit)."""
    return lambda x: y1 + (y2 - y1) / (x2 - x1) * (x - x1)


def wedge(n=36, upper_pts=((5, 110.0), (30, 101.0)), lower_pts=((10, 92.0), (28, 97.0))):
    """Bars sitting inside a converging wedge, plus the pivots defining it.

    Returns (df, pivots). Every bar's range is inset from the trendlines so
    containment holds; callers mutate specific bars to test the rejections.
    """
    up = _line(*upper_pts[0], *upper_pts[1])
    lo = _line(*lower_pts[0], *lower_pts[1])
    rows = []
    for x in range(n):
        u, l = up(x), lo(x)
        inset = (u - l) * 0.02
        mid = (u + l) / 2
        rows.append([mid, u - inset, l + inset, mid, 10.0])
    df = make_df(rows)
    pivots = [
        Pivot(index=upper_pts[0][0], ts=int(df.index[upper_pts[0][0]]),
              price=upper_pts[0][1], kind="high"),
        Pivot(index=lower_pts[0][0], ts=int(df.index[lower_pts[0][0]]),
              price=lower_pts[0][1], kind="low"),
        Pivot(index=lower_pts[1][0], ts=int(df.index[lower_pts[1][0]]),
              price=lower_pts[1][1], kind="low"),
        Pivot(index=upper_pts[1][0], ts=int(df.index[upper_pts[1][0]]),
              price=upper_pts[1][1], kind="high"),
    ]
    pivots.sort(key=lambda p: (p.index, p.kind))
    return df, pivots


class TestTriangleGeometry:
    def test_detects_contained_wedge(self):
        df, pivots = wedge()
        tri = [c for c in detect_patterns(df, pivots) if c.kind == "triangle"]
        assert sorted(c.direction for c in tri) == ["long", "short"]

    def test_rejects_bar_piercing_a_trendline(self):
        df, pivots = wedge()
        df = df.copy()
        # One bar spikes far above its upper trendline: price was not bounded by
        # the lines, so the "triangle" never contained the move.
        df.iloc[15, df.columns.get_loc("high")] = 200.0
        assert [c for c in detect_patterns(df, pivots) if c.kind == "triangle"] == []

    def test_rejects_when_last_close_already_outside(self):
        df, pivots = wedge()
        df = df.copy()
        # Price already through the upper line: the structure is broken, not
        # pending, so it must not be re-offered as a fresh breakout candidate.
        df.iloc[-1, df.columns.get_loc("close")] = 150.0
        assert [c for c in detect_patterns(df, pivots) if c.kind == "triangle"] == []

    def test_rejects_expanding_upper_side(self):
        # Upper line rises while the lower line rises faster: the gap closes, but
        # a rising upper boundary is not a triangle.
        df, pivots = wedge(upper_pts=((5, 101.0), (30, 110.0)), lower_pts=((10, 92.0), (28, 105.0)))
        assert [c for c in detect_patterns(df, pivots) if c.kind == "triangle"] == []

    def test_rejects_structure_narrower_than_min_width(self):
        n = config.TRIANGLE_MIN_WIDTH_BARS + 6
        df, pivots = wedge(
            n=n,
            upper_pts=((n - 12, 110.0), (n - 4, 101.0)),
            lower_pts=((n - 11, 92.0), (n - 5, 97.0)),
        )
        assert [c for c in detect_patterns(df, pivots) if c.kind == "triangle"] == []

    def test_rejects_when_one_trendline_is_stale(self):
        # Lower line's last pivot is far older than PATTERN_MAX_AGE_BARS even
        # though the upper line is fresh: half the geometry is out of date.
        stale = config.PATTERN_MAX_AGE_BARS + 10
        df, pivots = wedge(
            n=60, upper_pts=((5, 110.0), (55, 101.0)), lower_pts=((10, 92.0), (59 - stale, 97.0))
        )
        assert [c for c in detect_patterns(df, pivots) if c.kind == "triangle"] == []


def flag_1h_df():
    """20 flat bars at 100, 6-bar pole rising to 110, 6-bar consolidation."""
    rows = [[100, 100.5, 99.5, 100, 10.0]] * 20
    for close in (102, 104, 106, 108, 109, 110):
        rows.append([close - 1, close + 0.5, close - 0.5, close, 10.0])
    rows += [[109, 109.5, 108.5, 109, 10.0]] * 6
    return make_df(rows)


class TestFlags:
    def test_detects_bull_flag(self):
        flags = [c for c in detect_patterns(flag_1h_df()) if c.kind == "flag"]
        longs = [c for c in flags if c.direction == "long"]
        assert len(longs) == 1
        c = longs[0]
        assert c.breakout_level == 109.5  # consolidation high
        assert c.target_height > 0

    def test_no_flag_without_pole(self):
        df = make_df([[100, 100.5, 99.5, 100, 10.0]] * 40)
        assert [c for c in detect_patterns(df) if c.kind == "flag"] == []

    def test_pole_window_spans_exactly_config_bars(self):
        # flag_1h_df: n=32, c=6 => pole_end=25, so the 12-bar pole window is
        # bars 14..25 and bar 13 is outside it. A crater at bar 13 must not
        # inflate the measured pole height.
        base = [c for c in detect_patterns(flag_1h_df()) if c.direction == "long"][0]
        df = flag_1h_df()
        df.iloc[13, df.columns.get_loc("low")] = 80.0
        after = [c for c in detect_patterns(df) if c.kind == "flag" and c.direction == "long"]
        assert len(after) == 1
        assert math.isclose(after[0].target_height, base.target_height)

    def test_consolidation_may_reach_pole_high_set_earlier_than_pole_end(self):
        # Pole peaks at 110.5 two bars before it ends; the consolidation tops at
        # 110.4 — under the pole's high but above the final pole bar's high.
        rows = [[100, 100.5, 99.5, 100, 10.0]] * 20
        rows += [
            [101, 102.0, 100.5, 102, 10.0],
            [102, 104.0, 101.5, 104, 10.0],
            [104, 106.0, 103.5, 106, 10.0],
            [106, 110.5, 105.5, 109, 10.0],  # pole high here
            [109, 110.2, 108.5, 110, 10.0],  # pole_end: lower high
        ]
        rows += [[109.5, 110.4, 108.8, 109.5, 10.0]] * 6
        flags = [c for c in detect_patterns(make_df(rows)) if c.direction == "long"]
        assert len(flags) == 1
        assert math.isclose(flags[0].breakout_level, 110.4)


def make_candidate(direction="long", level=100.0, height=5.0, end_ts=START):
    return PatternCandidate(
        kind="flag",
        direction=direction,
        breakout_level=level,
        target_height=height,
        start_ts=START - 10 * D_SET,
        end_ts=end_ts,
    )


def breakout_df(prev_close, last_close, last_volume=30.0, n_prior=21):
    """Trigger df: n_prior flat bars (vol 10), then prev/last closes."""
    rows = [[100, 100.5, 99.5, 100, 10.0]] * (n_prior - 1)
    rows.append([prev_close, prev_close + 0.2, prev_close - 0.2, prev_close, 10.0])
    rows.append([last_close, last_close + 0.2, last_close - 0.2, last_close, last_volume])
    return make_df(rows, interval=D_TRIG)


class TestCheckBreakout:
    def test_fresh_long_cross_fires_with_volume(self):
        df = breakout_df(prev_close=99.8, last_close=100.4)
        event = check_breakout(df, make_candidate())
        assert event is not None
        assert event.direction == "long"
        assert event.price == 100.4
        assert math.isclose(event.volume_ratio, 3.0)
        assert event.volume_high is True

    def test_stale_cross_does_not_refire(self):
        df = breakout_df(prev_close=100.3, last_close=100.4)  # already above
        assert check_breakout(df, make_candidate()) is None

    def test_no_cross_returns_none(self):
        df = breakout_df(prev_close=99.5, last_close=99.8)
        assert check_breakout(df, make_candidate()) is None

    def test_short_cross_fires(self):
        df = breakout_df(prev_close=100.2, last_close=99.6)
        event = check_breakout(df, make_candidate(direction="short"))
        assert event is not None
        assert event.direction == "short"

    def test_trigger_before_pattern_end_rejected(self):
        df = breakout_df(prev_close=99.8, last_close=100.4)
        late_end = int(df.index[-1]) + D_TRIG  # pattern completes after trigger bar
        assert check_breakout(df, make_candidate(end_ts=late_end)) is None

    def test_insufficient_volume_history_gives_nan_ratio(self):
        df = breakout_df(prev_close=99.8, last_close=100.4, n_prior=5)
        event = check_breakout(df, make_candidate())
        assert event is not None
        assert math.isnan(event.volume_ratio)
        assert event.volume_high is False

    def _crossed_two_bars_ago(self):
        """Trigger df whose fresh cross of 100.0 is three bars from the end."""
        rows = [[99.5, 99.7, 99.3, 99.5, 10.0]] * 20
        rows.append([100.0, 100.5, 99.9, 100.4, 30.0])  # the crossing bar
        rows += [[100.4, 100.7, 100.3, 100.5, 10.0], [100.5, 100.8, 100.4, 100.6, 10.0]]
        return make_df(rows, interval=D_TRIG)

    def test_default_lookback_sees_only_the_latest_bar(self):
        assert check_breakout(self._crossed_two_bars_ago(), make_candidate()) is None

    def test_widened_lookback_finds_the_most_recent_crossing(self):
        event = check_breakout(self._crossed_two_bars_ago(), make_candidate(), lookback_bars=3)
        assert event is not None
        assert event.price == 100.4  # the crossing bar's close, not the latest
        assert math.isclose(event.volume_ratio, 3.0)  # volume from that bar too

    def test_gap_in_the_crossing_pair_is_skipped(self):
        df = breakout_df(prev_close=99.8, last_close=100.4)
        gapped = df.drop(index=df.index[-2])  # last pair now two intervals apart
        assert check_breakout(gapped, make_candidate(), interval_ms=D_TRIG) is None
        # Without the contiguity check the same slice reads as a fresh crossing.
        assert check_breakout(gapped, make_candidate()) is not None


class TestBuildSignal:
    def test_long_setup_passes_rr_floor(self):
        df = breakout_df(prev_close=99.8, last_close=100.3)
        candidate = make_candidate(level=100.0, height=5.0)  # generous reward
        event = check_breakout(df, candidate)
        signal = build_signal(SYMBOL, candidate, event, atr_value=1.0)
        assert signal is not None
        assert math.isclose(signal.stop, 100.3 - config.ATR_STOP_MULTIPLE * 1.0)
        assert signal.rr >= config.RR_FLOOR
        assert math.isclose(signal.rr, signal.reward_pct / signal.risk_pct)

    def test_rr_below_floor_rejected(self):
        df = breakout_df(prev_close=99.8, last_close=100.3)
        candidate = make_candidate(level=100.0, height=0.5)  # thin reward
        event = check_breakout(df, candidate)
        # A wide ATR stop dominates a thin measured-move reward: rr < floor.
        assert build_signal(SYMBOL, candidate, event, atr_value=1.0) is None

    def test_rr_floor_is_overridable(self):
        df = breakout_df(prev_close=99.8, last_close=100.3)
        candidate = make_candidate(level=100.0, height=0.5)
        event = check_breakout(df, candidate)
        assert build_signal(SYMBOL, candidate, event, atr_value=1.0, rr_floor=0.1) is not None

    def test_zero_atr_yields_zero_risk_rejected(self):
        df = breakout_df(prev_close=99.8, last_close=100.3)
        candidate = make_candidate(level=100.0, height=5.0)
        event = check_breakout(df, candidate)
        assert build_signal(SYMBOL, candidate, event, atr_value=0.0) is None

    def test_nan_atr_rejected(self):
        # Wilder ATR is NaN for the first ATR_STOP_PERIOD bars (warmup); a
        # live symbol with short history can hand this straight through, so
        # it must never reach compute_atr_stop and produce stop=nan.
        df = breakout_df(prev_close=99.8, last_close=100.3)
        candidate = make_candidate(level=100.0, height=5.0)
        event = check_breakout(df, candidate)
        assert build_signal(SYMBOL, candidate, event, atr_value=float("nan")) is None

    def test_rr_exactly_at_floor_is_accepted(self):
        # Derivation (entry=100.3, level=100.0, atr_value=1.0, k=ATR_STOP_MULTIPLE=1.5):
        #   stop = entry - k*atr = 100.3 - 1.5 = 98.8
        #   risk = entry - stop = 1.5
        # Solve height so rr == RR_FLOOR (1.5) exactly. rr = reward/risk since
        # both reward_pct and risk_pct divide by the same entry (it cancels):
        #   reward = RR_FLOOR * risk = 1.5 * 1.5 = 2.25
        #   height = reward + (entry - level) = 2.25 + 0.3 = 2.55
        # 2.25 / 1.5 == 1.5 exactly in IEEE-754 double (both terms are exact
        # binary fractions), so this lands on the boundary without float
        # slop — verified via math.isclose below rather than assumed.
        df = breakout_df(prev_close=99.8, last_close=100.3)
        candidate = make_candidate(level=100.0, height=2.55)
        event = check_breakout(df, candidate)
        signal = build_signal(SYMBOL, candidate, event, atr_value=1.0)
        assert signal is not None
        assert math.isclose(signal.rr, config.RR_FLOOR)

    def test_entry_beyond_target_rejected(self):
        # Force an event whose entry overshot the measured-move target.
        from trading_bot.signals.breakout import BreakoutEvent

        candidate = make_candidate(level=100.0, height=0.2)
        event = BreakoutEvent(
            ts=START, price=100.4, level=100.0, direction="long",
            volume_ratio=1.0, volume_high=False,
        )
        assert build_signal(SYMBOL, candidate, event, atr_value=1.0) is None


class TestRankSignals:
    def _sig(self, pattern, rr):
        return Signal(
            symbol=SYMBOL, ts=START, direction="long", pattern=pattern,
            entry=100.0, stop=99.5, target=105.0,
            risk_pct=0.005, reward_pct=0.005 * rr, rr=rr,
            volume_ratio=1.0, volume_high=False,
        )

    def test_orders_by_rr_not_pattern_name(self):
        # "triangle" sorts last alphabetically but wins on R:R.
        ranked = rank_signals([self._sig("flag", 2.0), self._sig("triangle", 9.0)])
        assert [s.pattern for s in ranked] == ["triangle", "flag"]

    def test_ties_break_deterministically_on_pattern(self):
        ranked = rank_signals([self._sig("triangle", 3.0), self._sig("flag", 3.0)])
        assert [s.pattern for s in ranked] == ["flag", "triangle"]


class TestContiguousTail:
    def test_untouched_when_evenly_spaced(self):
        df = make_df([[100, 100.5, 99.5, 100, 10.0]] * 10)
        assert len(setup_mod._contiguous_tail(df, D_SET, SYMBOL, SETUP_TF)) == 10

    def test_trims_to_the_run_after_the_last_gap(self):
        df = make_df([[100, 100.5, 99.5, 100, 10.0]] * 10)
        gapped = df.drop(index=df.index[4])  # gap between original bars 3 and 5
        trimmed = setup_mod._contiguous_tail(gapped, D_SET, SYMBOL, SETUP_TF)
        assert len(trimmed) == 5  # original bars 5..9
        assert int(trimmed.index[0]) == int(df.index[5])

    def test_warns_when_trimming(self, caplog):
        df = make_df([[100, 100.5, 99.5, 100, 10.0]] * 10)
        gapped = df.drop(index=df.index[4])
        with caplog.at_level("WARNING", logger="trading_bot"):
            setup_mod._contiguous_tail(gapped, D_SET, SYMBOL, SETUP_TF)
        assert "has gaps" in caplog.text


class TestLineValue:
    def test_raises_on_coincident_x(self):
        import pytest

        from trading_bot.signals.patterns import _line_value

        with pytest.raises(ValueError):
            _line_value(7, 100.0, 7, 105.0, 9)


def seed_candles(conn, symbol, timeframe, df):
    rows = [
        [int(ts), r["open"], r["high"], r["low"], r["close"], r["volume"]]
        for ts, r in df.iterrows()
    ]
    storage.upsert_candles(conn, symbol, timeframe, rows)


class TestCurrentSignals:
    def test_empty_db_is_uncertain_with_no_signals(self, tmp_path):
        conn = storage.connect(str(tmp_path / "test.db"))
        regime, signals = current_signals(conn, SYMBOL, now_ms=START)
        assert regime == "uncertain"
        assert signals == []

    def test_non_trending_regime_gates_signals(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "test.db"))
        monkeypatch.setattr(
            setup_mod, "current_regime", lambda *a, **k: ("ranging", 20.0, 0.5)
        )
        regime, signals = current_signals(conn, SYMBOL, now_ms=START)
        assert regime == "ranging"
        assert signals == []

    def test_trending_flag_breakout_produces_signal(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "test.db"))
        monkeypatch.setattr(
            setup_mod, "current_regime", lambda *a, **k: ("trending", 30.0, 0.5)
        )
        # Fixed ATR for determinism — this test asserts pipeline wiring, not
        # ATR's own math (that's wilder.py's test suite).
        fixed_atr = 1.0
        monkeypatch.setattr(
            setup_mod,
            "wilder_atr",
            lambda df, period=None: pd.Series(fixed_atr, index=df.index),
        )

        df_setup = flag_1h_df()  # bull flag, breakout level 109.5
        seed_candles(conn, SYMBOL, SETUP_TF, df_setup)

        # Trigger series ending shortly after the last setup bar: fresh cross of 109.5.
        last_setup_ts = int(df_setup.index[-1])
        rows_trig = [[109.2, 109.4, 109.0, 109.2, 10.0]] * 21
        rows_trig.append([109.2, 110.0, 109.1, 109.9, 30.0])
        df_trig = make_df(rows_trig, start=last_setup_ts - 19 * D_TRIG, interval=D_TRIG)
        seed_candles(conn, SYMBOL, TRIGGER_TF, df_trig)

        now_ms = int(df_trig.index[-1]) + D_TRIG
        regime, signals = current_signals(conn, SYMBOL, now_ms=now_ms)
        assert regime == "trending"
        assert len(signals) == 1
        s = signals[0]
        assert s.direction == "long"
        assert s.pattern == "flag"
        assert s.entry == 109.9
        assert math.isclose(s.stop, 109.9 - config.ATR_STOP_MULTIPLE * fixed_atr)
        assert s.volume_high is True
        assert s.rr >= config.RR_FLOOR


class TestSignalCommand:
    def _fake_signal(self):
        return Signal(
            symbol=SYMBOL, ts=START, direction="long", pattern="flag",
            entry=109.9, stop=109.5, target=119.0,
            risk_pct=0.0036, reward_pct=0.083, rr=22.8,
            volume_ratio=3.0, volume_high=True,
        )

    def test_exit_zero_and_row_per_signal(self, monkeypatch, capsys):
        import trading_bot.cli as cli

        results = {
            "BTCUSDT": ("trending", [self._fake_signal()]),
            "ETHUSDT": ("extreme-volatility", []),
        }
        monkeypatch.setattr(
            cli, "scan_symbol", lambda conn, sym, now_ms=None: results[sym]
        )
        exit_code = _signal_command(None, ["BTCUSDT", "ETHUSDT"], now_ms=START)
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "flag*" in out  # volume_high marker
        assert "(inactive)" in out  # no-method regime placeholder row

    def test_exit_one_on_uncertain(self, monkeypatch, capsys):
        import trading_bot.cli as cli

        monkeypatch.setattr(
            cli, "scan_symbol", lambda conn, sym, now_ms=None: ("uncertain", [])
        )
        exit_code = _signal_command(None, ["BTCUSDT"], now_ms=START)
        assert exit_code == 1
        assert "uncertain" in capsys.readouterr().out
