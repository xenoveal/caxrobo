"""Tests for the three Phase 4 Confirmation plug-ins.

Contexts are real EvalContexts built from a synthetic 3-tier SQLite DB via
EvalSession + OhlcvSource, exactly as tests/test_framework_contracts.py does it.
Two context stubs in one repo is how live/backtest parity dies, so no second stub
shape is invented here.
"""

import dataclasses
import math

import pandas as pd
import pytest

from trading_bot import config
from trading_bot.backtest import engine
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import contracts
from trading_bot.framework.contracts import ConfirmationVerdict, DetectedEvent
from trading_bot.framework.graph import RegimeGate
from trading_bot.plugins.confirmations.macd import macd_confirmation
from trading_bot.plugins.confirmations.trend_context import (
    CONTINUATION_KINDS,
    KNOWN_KINDS,
    NO_TREND_CLAIM_KINDS,
    REVERSAL_KINDS,
    _required_trend_direction,
    _trend_fraction,
    trend_context,
)
from trading_bot.plugins.confirmations.volume_breakout import volume_breakout
from trading_bot.plugins.data.ohlcv import OhlcvSource
from trading_bot.signals.breakout import BreakoutEvent

SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded (contract §8).
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_REG = storage.TIMEFRAME_MS[REGIME_TF]
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000

N_SETUP = 200  # >> config.MACD_MIN_BARS, so MACD is defined at the last bar


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Clear BOTH memos around every test.

    Both are keyed by content fingerprints, so cross-fixture collisions should be
    impossible — but test isolation must not DEPEND on that being right.
    """
    engine.clear_caches()
    fcontext.clear_caches()
    yield
    engine.clear_caches()
    fcontext.clear_caches()


def seed(conn, timeframe, rows, start=START, interval=D_SET):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    storage.upsert_candles(conn, SYMBOL, timeframe, data)
    return data


def ramp(n, base=100.0, step=1.0):
    """A monotone close ramp. step > 0 drives the MACD histogram positive;
    step < 0 drives it negative. That is the only property the MACD tests need."""
    return [
        [base + step * i, base + step * i + 2.0, base + step * i - 2.0, base + step * i, 10.0]
        for i in range(n)
    ]


def make_session(conn, *, regime_gate=None):
    return fcontext.EvalSession(
        OhlcvSource(conn),
        SYMBOL,
        tiers=(REGIME_TF, SETUP_TF, TRIGGER_TF),
        regime_gate=regime_gate or RegimeGate(),
    )


def three_tier(tmp_path, *, setup_step=1.0, name="t.db"):
    conn = storage.connect(str(tmp_path / name))
    seed(conn, REGIME_TF, ramp(300), interval=D_REG)
    seed(conn, SETUP_TF, ramp(N_SETUP, step=setup_step), interval=D_SET)
    seed(conn, TRIGGER_TF, ramp(400), interval=D_TRIG)
    return conn


def trigger_ctx(conn, *, regime_gate=None):
    """A context bound to a trigger-bar OPEN late enough that every setup bar of
    the fixture has closed — the production role/now_ms pairing."""
    session = make_session(conn, regime_gate=regime_gate)
    return session.context("trigger", START + (N_SETUP + 5) * D_SET)


# trend_context's OHLC measure reads high/low, which `ramp()` offsets by a
# FIXED +/-2.0 around whatever the close is that bar. Over `three_tier`'s
# usual setup_step magnitudes (0.4-1.0, chosen for MACD's sign test, which
# only cares whether the histogram is non-zero) a lookback-20 window's real
# directional signal is the same order of magnitude as that fixed +/-2 wick
# noise, so a genuinely declining series can still show a spuriously positive
# "up" fraction from wick overlap alone. TREND_STEP is chosen so the real
# signal over one lookback window (`TREND_N` bars, `TREND_STEP` each) is an
# order of magnitude larger than the +/-2 wick noise, and TREND_BASE keeps
# every price in the window positive for either sign of TREND_STEP.
TREND_N = 30
TREND_STEP = 3.0
TREND_BASE = 300.0


def trending_ctx(tmp_path, *, step, name="trend.db"):
    """A three-tier fixture whose SETUP_TF ramp is steep enough for
    trend_context's OHLC-range measure to read cleanly (see TREND_STEP's
    comment). Regime and trigger tiers are the same flat-shape ramps
    `three_tier` uses — trend_context never reads them."""
    conn = storage.connect(str(tmp_path / name))
    seed(conn, REGIME_TF, ramp(300), interval=D_REG)
    seed(conn, SETUP_TF, ramp(TREND_N, base=TREND_BASE, step=step), interval=D_SET)
    seed(conn, TRIGGER_TF, ramp(400), interval=D_TRIG)
    session = make_session(conn)
    return session.context("trigger", START + (TREND_N + 5) * D_SET)


def d1_rally_then_consolidation_prices():
    """The plan's D1 reproduction fixture verbatim (Evidence section): 15 flat
    bars, then a +49%/20-bar rally, then an ordinary consolidation. On
    `master` this reproduces `_detect_double` firing a `double-bottom` with no
    decline anywhere to reverse — the systemic defect trend-context exists to
    close at the pipeline level."""
    p = 100.0
    prices = []
    for _ in range(15):
        prices.append(p)
    for _ in range(20):
        p *= 1.02
        prices.append(p)
    for _ in range(5):
        p *= 0.985
        prices.append(p)
    for _ in range(6):
        p *= 1.0135
        prices.append(p)
    for _ in range(6):
        p *= 0.9875
        prices.append(p)
    for _ in range(4):
        p *= 1.004
        prices.append(p)
    return prices


def d1_ctx(tmp_path, *, name="d1.db"):
    """A three-tier fixture whose SETUP_TF is the D1 price path, as
    open=high=low=close flat bars — the plan's own fixture shape ("Bars are
    open=high=low=close synthetic paths built with make_df")."""
    prices = d1_rally_then_consolidation_prices()
    n = len(prices)
    conn = storage.connect(str(tmp_path / name))
    seed(conn, REGIME_TF, ramp(300), interval=D_REG)
    flat_rows = [[v, v, v, v, 10.0] for v in prices]
    data = [[START + i * D_SET] + row for i, row in enumerate(flat_rows)]
    storage.upsert_candles(conn, SYMBOL, SETUP_TF, data)
    seed(conn, TRIGGER_TF, ramp(400), interval=D_TRIG)
    session = make_session(conn)
    return session.context("trigger", START + (n + 5) * D_SET)


def make_event(
    direction="long", *, volume_ratio=2.0, level=150.0, height=10.0, kind="donchian-breakout"
):
    """A DetectedEvent already stamped with the trigger facts, which is the only
    state in which a Confirmation ever sees one (execute.py stamps before
    calling).

    `kind` defaults to "donchian-breakout" (unchanged from every pre-existing
    caller) but is overridable so TestTrendContext can build events for the
    thirteen catalog chart-pattern kinds without a second event-construction
    helper (two stub shapes for one thing is how live/backtest parity dies,
    per this module's own docstring)."""
    event = DetectedEvent(
        kind=kind,
        direction=direction,
        level=level,
        target_height=height,
        start_ts=START,
        end_ts=START + 10 * D_SET,
    )
    trig = BreakoutEvent(
        ts=START + 11 * D_SET,
        price=level + 1.0 if direction == "long" else level - 1.0,
        level=level,
        direction=direction,
        volume_ratio=volume_ratio,
        volume_high=(not math.isnan(volume_ratio))
        and volume_ratio >= config.VOLUME_HIGH_RATIO,
    )
    return contracts.with_trigger(event, trig)


DEFAULT_VOL = dict(
    min_ratio=config.VOLUME_CONFIRM_MIN_RATIO,
    require_defined=config.VOLUME_CONFIRM_REQUIRE_DEFINED,
)
DEFAULT_MACD = dict(
    fast=config.MACD_FAST_PERIOD,
    slow=config.MACD_SLOW_PERIOD,
    signal=config.MACD_SIGNAL_PERIOD,
    min_hist=config.MACD_CONFIRM_MIN_HIST,
)
# trend_context reads NO config attribute for its defaults (module docstring's
# "WHY EVERY DEFAULT BELOW IS AN INLINE LITERAL" section) so these mirror the
# ParamSpec literals directly rather than a config.* constant.
DEFAULT_TREND = dict(lookback=20, min_trend_frac=0.03)


class TestVolumeBreakout:
    """The gate that closes KNOWN-LIMITATIONS §0c.

    §0c: "volume is computed on every signal but gates nothing." These tests are
    the proof that it now gates something, at the plug-in level; the end-to-end
    proof is tests/test_pipeline_thin_slice.py::TestVolumeGateActuallyGates.
    """

    def test_boundary_ratio_exactly_at_threshold_passes(self, tmp_path):
        """Boundary is `>=`, matching signals/breakout.py:138's
        `volume_ratio >= volume_high_ratio`."""
        ctx = trigger_ctx(three_tier(tmp_path))
        v = volume_breakout(
            ctx, make_event(volume_ratio=config.VOLUME_CONFIRM_MIN_RATIO), **DEFAULT_VOL
        )
        assert v.passed is True
        assert v.score == config.VOLUME_CONFIRM_MIN_RATIO

    def test_just_below_threshold_fails(self, tmp_path):
        ctx = trigger_ctx(three_tier(tmp_path))
        ratio = config.VOLUME_CONFIRM_MIN_RATIO - 0.01
        v = volume_breakout(ctx, make_event(volume_ratio=ratio), **DEFAULT_VOL)
        assert v.passed is False
        assert v.score == ratio
        # The reason names BOTH numbers, so a rejection is diagnosable from logs.
        assert f"{ratio:.4f}" in v.reason
        assert f"{config.VOLUME_CONFIRM_MIN_RATIO:.4f}" in v.reason

    def test_nan_ratio_fails_closed(self, tmp_path):
        """'Unknown' must never read as 'confirmed'."""
        ctx = trigger_ctx(three_tier(tmp_path))
        v = volume_breakout(ctx, make_event(volume_ratio=float("nan")), **DEFAULT_VOL)
        assert v.passed is False
        assert "undefined" in v.reason
        assert math.isnan(v.score)

    def test_nan_ratio_passes_when_require_defined_is_off(self, tmp_path):
        """The escape hatch exists for Phase 6 to jitter, and is OFF by default —
        config.VOLUME_CONFIRM_REQUIRE_DEFINED is True."""
        assert config.VOLUME_CONFIRM_REQUIRE_DEFINED is True
        ctx = trigger_ctx(three_tier(tmp_path))
        v = volume_breakout(
            ctx,
            make_event(volume_ratio=float("nan")),
            min_ratio=config.VOLUME_CONFIRM_MIN_RATIO,
            require_defined=False,
        )
        assert v.passed is True

    def test_score_is_the_measured_ratio_in_every_branch(self, tmp_path):
        """D3's promise: the graded value is REPORTED even though the gate is hard,
        so Phases 5-6 can use it without Phase 4 pretending to."""
        ctx = trigger_ctx(three_tier(tmp_path))
        for ratio in (0.1, 1.0, config.VOLUME_CONFIRM_MIN_RATIO, 5.0):
            v = volume_breakout(ctx, make_event(volume_ratio=ratio), **DEFAULT_VOL)
            assert v.score == ratio

    def test_min_ratio_override_changes_the_verdict(self, tmp_path):
        """Proves the ParamSpec default is RESOLVED by the caller, not hardcoded."""
        ctx = trigger_ctx(three_tier(tmp_path))
        event = make_event(volume_ratio=1.2)
        assert volume_breakout(ctx, event, **DEFAULT_VOL).passed is False
        assert volume_breakout(
            ctx, event, min_ratio=1.0, require_defined=True
        ).passed is True

    def test_unstamped_event_raises_rather_than_rejecting(self, tmp_path):
        """An event that never saw a trigger is an EXECUTOR BUG, not a rejection.

        Silently returning passed=False would hide the bug behind "this strategy
        takes no trades".
        """
        from trading_bot.framework.errors import ContractError

        ctx = trigger_ctx(three_tier(tmp_path))
        bare = DetectedEvent(
            kind="donchian-breakout", direction="long", level=150.0,
            target_height=10.0, start_ts=START, end_ts=START + D_SET,
        )
        with pytest.raises(ContractError):
            volume_breakout(ctx, bare, **DEFAULT_VOL)


class TestMacdConfirmation:
    def test_long_passes_on_a_rising_series(self, tmp_path):
        ctx = trigger_ctx(three_tier(tmp_path, setup_step=1.0))
        v = macd_confirmation(ctx, make_event("long"), **DEFAULT_MACD)
        assert v.passed is True
        assert v.score > 0.0

    def test_long_fails_on_a_falling_series(self, tmp_path):
        ctx = trigger_ctx(three_tier(tmp_path, setup_step=-0.4))
        v = macd_confirmation(ctx, make_event("long"), **DEFAULT_MACD)
        assert v.passed is False
        assert v.score < 0.0

    def test_short_is_the_exact_mirror(self, tmp_path):
        """Sign-flipped data must flip both verdicts, or the gate is asymmetric."""
        up = trigger_ctx(three_tier(tmp_path, setup_step=1.0, name="up.db"))
        down = trigger_ctx(three_tier(tmp_path, setup_step=-0.4, name="down.db"))
        assert macd_confirmation(up, make_event("short"), **DEFAULT_MACD).passed is False
        assert macd_confirmation(down, make_event("short"), **DEFAULT_MACD).passed is True

    def test_warmup_fails_closed(self, tmp_path):
        """Fewer than MACD_MIN_BARS setup bars => histogram undefined => reject."""
        conn = storage.connect(str(tmp_path / "short.db"))
        n_short = config.MACD_MIN_BARS - 5
        seed(conn, REGIME_TF, ramp(300), interval=D_REG)
        seed(conn, SETUP_TF, ramp(n_short), interval=D_SET)
        seed(conn, TRIGGER_TF, ramp(400), interval=D_TRIG)
        ctx = make_session(conn).context("trigger", START + (n_short + 2) * D_SET)
        v = macd_confirmation(ctx, make_event("long"), **DEFAULT_MACD)
        assert v.passed is False
        assert "undefined" in v.reason
        assert math.isnan(v.score)

    def test_min_hist_rejects_small_magnitude_on_BOTH_sides(self, tmp_path):
        """The sign-error pin for `-min_hist` on the short branch.

        A positive min_hist must make the gate symmetric AROUND ZERO. Using
        `min_hist` rather than `-min_hist` for shorts would leave the short gate
        PERMISSIVE while looking symmetric, which no single-direction test catches.
        """
        up = trigger_ctx(three_tier(tmp_path, setup_step=1.0, name="up.db"))
        down = trigger_ctx(three_tier(tmp_path, setup_step=-0.4, name="down.db"))
        long_hist = macd_confirmation(up, make_event("long"), **DEFAULT_MACD).score
        short_hist = macd_confirmation(down, make_event("short"), **DEFAULT_MACD).score
        assert long_hist > 0 > short_hist
        # A threshold above both magnitudes must reject both.
        huge = max(abs(long_hist), abs(short_hist)) * 2.0
        params = dict(DEFAULT_MACD, min_hist=huge)
        assert macd_confirmation(up, make_event("long"), **params).passed is False
        assert macd_confirmation(down, make_event("short"), **params).passed is False

    def test_default_min_hist_is_a_pure_sign_test(self, tmp_path):
        assert config.MACD_CONFIRM_MIN_HIST == 0.0
        ctx = trigger_ctx(three_tier(tmp_path, setup_step=1.0))
        v = macd_confirmation(ctx, make_event("long"), **DEFAULT_MACD)
        assert v.passed is (v.score > 0.0)


class TestTrendContext:
    """confirmation.trend-context — the Confirmation three detector docstrings
    deferred to "Phase 4" (continuation.py:27, :759; reversal.py) and nobody
    built until now. See the plug-in module docstring for the full argument;
    these tests pin the pieces a report can't take on faith.
    """

    def test_lookup_covers_every_registered_chart_pattern_kind_exactly_once(self):
        """The lookup is an explicit, enumerable table (plan's own requirement),
        not a string heuristic. Built from the LIVE registry rather than a
        second hardcoded kind list, so a future detector addition that forgets
        to update trend_context.py's sets fails this test instead of silently
        going unclassified."""
        from trading_bot.framework.registry import load_all

        registry = load_all()
        pattern_kinds = {
            spec.name
            for spec in registry.values()
            if spec.kind == "detector"
            and spec.module.endswith(
                ("plugins.detectors.reversal", "plugins.detectors.continuation")
            )
        }
        # Sanity: this is the thirteen family-1/2 rows, not zero and not "all
        # detectors" (which would also include e.g. rsi-divergence).
        assert len(pattern_kinds) == 13
        assert pattern_kinds == KNOWN_KINDS

        # Disjoint: every kind classified into EXACTLY one of the three sets.
        assert REVERSAL_KINDS & CONTINUATION_KINDS == set()
        assert REVERSAL_KINDS & NO_TREND_CLAIM_KINDS == set()
        assert CONTINUATION_KINDS & NO_TREND_CLAIM_KINDS == set()

    @pytest.mark.parametrize(
        "kind",
        sorted(REVERSAL_KINDS),
    )
    def test_reversal_kind_wants_the_opposite_of_its_own_bias(self, kind):
        assert _required_trend_direction(kind, "long") == "down"
        assert _required_trend_direction(kind, "short") == "up"

    @pytest.mark.parametrize(
        "kind",
        sorted(CONTINUATION_KINDS),
    )
    def test_continuation_kind_wants_the_same_as_its_own_bias(self, kind):
        assert _required_trend_direction(kind, "long") == "up"
        assert _required_trend_direction(kind, "short") == "down"

    def test_symmetrical_triangle_has_no_claim(self):
        assert _required_trend_direction("symmetrical-triangle", "long") is None
        assert _required_trend_direction("symmetrical-triangle", "short") is None

    def test_unknown_kind_has_no_claim(self):
        assert _required_trend_direction("some-future-detector", "long") is None

    def test_double_bottom_passes_after_a_real_decline(self, tmp_path):
        """A double-bottom's own direction is "long"; it reverses a DECLINE, so
        a falling setup-tier ramp must satisfy it."""
        ctx = trending_ctx(tmp_path, step=-TREND_STEP)
        v = trend_context(
            ctx, make_event("long", kind="double-bottom"), **DEFAULT_TREND
        )
        assert v.passed is True
        assert v.score > DEFAULT_TREND["min_trend_frac"]

    def test_double_bottom_rejects_a_rally_with_no_decline_to_reverse(self, tmp_path):
        """D1's fixture in miniature: this is the systemic gate that reproduces
        the +49% rally / ordinary-consolidation defect at the plug-in level."""
        ctx = trending_ctx(tmp_path, step=TREND_STEP)
        v = trend_context(
            ctx, make_event("long", kind="double-bottom"), **DEFAULT_TREND
        )
        assert v.passed is False
        assert v.score < DEFAULT_TREND["min_trend_frac"]
        assert "double-bottom" in v.reason and "down-trend" in v.reason

    def test_double_top_is_the_exact_mirror_of_double_bottom(self, tmp_path):
        """Sign-flipped data must flip both verdicts, or the gate is asymmetric
        — the same pin macd_confirmation's mirror test makes."""
        up = trending_ctx(tmp_path, step=TREND_STEP, name="up.db")
        down = trending_ctx(tmp_path, step=-TREND_STEP, name="down.db")
        assert trend_context(
            up, make_event("short", kind="double-top"), **DEFAULT_TREND
        ).passed is True
        assert trend_context(
            down, make_event("short", kind="double-top"), **DEFAULT_TREND
        ).passed is False

    def test_bull_flag_passes_after_a_real_advance(self, tmp_path):
        """A bull-flag continues an ADVANCE; direction is "long"."""
        ctx = trending_ctx(tmp_path, step=TREND_STEP)
        v = trend_context(ctx, make_event("long", kind="bull-flag"), **DEFAULT_TREND)
        assert v.passed is True

    def test_bull_flag_rejects_with_no_prior_advance(self, tmp_path):
        ctx = trending_ctx(tmp_path, step=-TREND_STEP)
        v = trend_context(ctx, make_event("long", kind="bull-flag"), **DEFAULT_TREND)
        assert v.passed is False
        assert "bull-flag" in v.reason and "up-trend" in v.reason

    def test_symmetrical_triangle_passes_unconditionally_either_direction(
        self, tmp_path
    ):
        """No prevailing-trend claim exists for this kind (module docstring's
        SYMMETRICAL TRIANGLE section) — it must pass regardless of the
        measured trend, and say so rather than silently reading as "measured
        and satisfied"."""
        ctx = trending_ctx(tmp_path, step=TREND_STEP)
        for direction in ("long", "short"):
            v = trend_context(
                ctx,
                make_event(direction, kind="symmetrical-triangle"),
                **DEFAULT_TREND,
            )
            assert v.passed is True
            assert math.isnan(v.score)
            assert "no prior-trend claim" in v.reason

    def test_unknown_kind_passes_through_rather_than_rejecting(self, tmp_path):
        """trend-context only judges the thirteen catalog chart-pattern kinds;
        an event from an unrelated detector (e.g. donchian-breakout) is out of
        its scope, and rejecting it would silently break any graph that mixes
        this Confirmation with a non-pattern detector."""
        ctx = trending_ctx(tmp_path, step=-TREND_STEP)
        v = trend_context(ctx, make_event("long"), **DEFAULT_TREND)
        assert v.passed is True
        assert math.isnan(v.score)
        assert "donchian-breakout" in v.reason

    def test_warmup_insufficient_history_fails_closed(self, tmp_path):
        """Fewer than `lookback` setup bars have closed => trend unknown =>
        reject. 'Unknown' must never read as 'confirmed', the same idiom
        macd_confirmation's warmup test and volume_breakout's NaN test use."""
        conn = storage.connect(str(tmp_path / "short.db"))
        n_short = DEFAULT_TREND["lookback"] - 3
        seed(conn, REGIME_TF, ramp(300), interval=D_REG)
        seed(conn, SETUP_TF, ramp(n_short, step=-1.0), interval=D_SET)
        seed(conn, TRIGGER_TF, ramp(400), interval=D_TRIG)
        ctx = make_session(conn).context("trigger", START + (n_short + 2) * D_SET)
        v = trend_context(
            ctx, make_event("long", kind="double-bottom"), **DEFAULT_TREND
        )
        assert v.passed is False
        assert "insufficient" in v.reason
        assert math.isnan(v.score)

    def test_failure_reason_names_kind_class_direction_and_both_numbers(
        self, tmp_path
    ):
        """A rejection must be diagnosable from the verdict alone, without a
        debugger — the contract's own framing for `reason`."""
        ctx = trending_ctx(tmp_path, step=TREND_STEP)
        v = trend_context(
            ctx,
            make_event("long", kind="double-bottom"),
            lookback=20,
            min_trend_frac=0.03,
        )
        assert v.passed is False
        assert "double-bottom" in v.reason
        assert "reversal" in v.reason
        assert "long" in v.reason
        assert "0.0300" in v.reason
        assert f"{v.score:.4f}" in v.reason

    def test_lookback_and_min_trend_frac_are_ParamSpec_resolved_not_hardcoded(
        self, tmp_path
    ):
        """Proves the ParamSpec defaults are the CALLER's, not baked into the
        function body — the same proof volume_breakout's override test makes."""
        ctx = trending_ctx(tmp_path, step=-TREND_STEP)
        event = make_event("long", kind="double-bottom")
        # A tiny min_trend_frac passes the measured decline; a huge one rejects it.
        assert trend_context(ctx, event, lookback=20, min_trend_frac=0.0001).passed is True
        assert trend_context(ctx, event, lookback=20, min_trend_frac=0.9).passed is False

    def test_up_and_down_can_never_both_pass_the_same_window(self):
        """The mutual-exclusivity pin. An EARLIER formulation measured "up" and
        "down" as two independent one-sided reaches
        (`trailing.high.max() - leading.low.min()` and its mirror), which a
        choppy window could satisfy in BOTH directions at once — the exact
        counterexample below, which cleared this plug-in's own default
        `min_trend_frac=0.03` on both sides under that formulation. The
        current net-displacement formula (module docstring's HOW THE TREND IS
        MEASURED) makes `_trend_fraction(w, "down") == -_trend_fraction(w, "up")`
        by construction, so for any `min_trend_frac > 0` at most one direction
        can clear the gate on the same bars.
        """

        def bars(pairs):
            """pairs: list of (low, high); open/close pinned to the midpoint,
            irrelevant here since _trend_fraction never reads them."""
            rows = []
            for low, high in pairs:
                mid = (low + high) / 2.0
                rows.append(
                    {"open": mid, "high": high, "low": low, "close": mid, "volume": 10.0}
                )
            return pd.DataFrame(rows)

        adversarial_windows = [
            # The coordinator's exact counterexample: leading [90,100],
            # trailing [95,110]. Under the rejected formulation this passed
            # BOTH up (0.222) and down (0.050) at min_trend_frac=0.03.
            bars([(90, 100)] * 10 + [(95, 110)] * 10),
            # A symmetric widening, no net level shift at all.
            bars([(95, 105)] * 10 + [(90, 110)] * 10),
            # A narrowing range shifted slightly down.
            bars([(85, 115)] * 10 + [(95, 105)] * 10),
            # Pure noise, alternating wide/narrow, no directional drift.
            bars(([(90, 110), (95, 105)] * 10)),
        ]
        min_trend_frac = DEFAULT_TREND["min_trend_frac"]
        for w in adversarial_windows:
            up = _trend_fraction(w, "up")
            down = _trend_fraction(w, "down")
            assert down == pytest.approx(-up), (w["low"].tolist(), up, down)
            assert not (up >= min_trend_frac and down >= min_trend_frac), (
                up,
                down,
            )

    def test_double_bottom_rejects_the_d1_plan_fixture(self, tmp_path):
        """The plan's own reproduction case (Evidence section, D1): a +49%
        rally followed by an ordinary consolidation, which `_detect_double`
        fires as `double-bottom` on `master` with no decline anywhere to
        reverse. trend_context's lookback window at default settings sees the
        tail consolidation, not the earlier rally, and its measured net
        decline there (~0.9%) is below the default `min_trend_frac` (3%), so
        the systemic gate rejects it — this is the pipeline-level half of the
        fix; WS-A's detector-local gate is the other half (plan's WS-A
        section: 'the two are complementary, not duplicates')."""
        ctx = d1_ctx(tmp_path)
        event = make_event(
            "long", kind="double-bottom", level=149.32, height=10.0
        )
        v = trend_context(ctx, event, **DEFAULT_TREND)
        assert v.passed is False
        assert v.score < DEFAULT_TREND["min_trend_frac"]

    def test_reads_no_config_attribute_for_its_defaults(self):
        """Hard constraint: every default in the trend-context ParamSpec block
        is an inline literal, mirroring the rising-wedge precedent
        (continuation.py:346-361) rather than a config.py constant — this
        cycle config.py is WS-A's file."""
        from trading_bot.framework.registry import load_all

        spec = load_all()["confirmation.trend-context"]
        assert spec.params["lookback"].default == 20
        assert spec.params["min_trend_frac"].default == 0.03


CONFIRMATIONS = (
    (volume_breakout, DEFAULT_VOL),
    (macd_confirmation, DEFAULT_MACD),
    (trend_context, DEFAULT_TREND),
)


class TestConfirmationsNeverMutate:
    """Contract §3: a Confirmation "NEVER mutates the event".

    Why it matters beyond tidiness: the audit trail is what makes "why did it
    trade" answerable, which the PRD names as evolution's stated advantage over
    deep RL. A Confirmation that edited the event would make the recorded reason
    for a trade a function of gate ORDER, and the trail would be fiction.
    """

    @pytest.mark.parametrize("fn,params", CONFIRMATIONS, ids=["volume", "macd", "trend-context"])
    def test_event_is_unchanged_and_still_the_same_object(self, tmp_path, fn, params):
        ctx = trigger_ctx(three_tier(tmp_path))
        event = make_event("long")
        before = dataclasses.asdict(event)
        identity = id(event)
        verdict = fn(ctx, event, **params)
        assert dataclasses.asdict(event) == before
        assert id(event) is identity or id(event) == identity
        assert isinstance(verdict, ConfirmationVerdict)
        assert verdict.name and verdict.reason

    @pytest.mark.parametrize("fn,params", CONFIRMATIONS, ids=["volume", "macd", "trend-context"])
    def test_name_is_the_full_registry_key(self, tmp_path, fn, params):
        """Trade.confirmations records registry KEYS (D8), so the verdict must
        carry the '<kind>.<name>' form, not the bare name."""
        ctx = trigger_ctx(three_tier(tmp_path))
        verdict = fn(ctx, make_event("long"), **params)
        assert verdict.name.startswith("confirmation.")


class TestVerdictAuditability:
    """Every failure path names the measured value AND the threshold, so a
    rejection is diagnosable from the verdict alone without a debugger."""

    def test_volume_failure_reason_carries_both_numbers(self, tmp_path):
        ctx = trigger_ctx(three_tier(tmp_path))
        v = volume_breakout(ctx, make_event(volume_ratio=0.5), **DEFAULT_VOL)
        assert v.passed is False
        assert "0.5000" in v.reason and "1.5000" in v.reason

    def test_macd_failure_reason_carries_the_value_and_direction(self, tmp_path):
        ctx = trigger_ctx(three_tier(tmp_path, setup_step=-0.4))
        v = macd_confirmation(ctx, make_event("long"), **DEFAULT_MACD)
        assert v.passed is False
        assert "MACD hist" in v.reason and "long" in v.reason
