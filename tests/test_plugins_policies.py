"""Tests for policy.measured-move (v0.3.0 Phase 4).

The load-bearing test here is TestEquivalenceWithBuildSignal: policy.measured-move
is signals.setup.build_signal's arithmetic re-expressed against the contracts, so
the cheapest possible proof that the re-expression did not drift is to run both on
the same numbers and compare. It is the local echo of contract §5's parity
requirement.
"""

import math

import pytest

from trading_bot import config
from trading_bot.backtest import engine
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import contracts
from trading_bot.framework.contracts import DetectedEvent, PositionPlan
from trading_bot.framework.graph import RegimeGate
from trading_bot.plugins.data.ohlcv import OhlcvSource
from trading_bot.plugins.policies.measured_move import measured_move
from trading_bot.signals.breakout import BreakoutEvent
from trading_bot.signals.patterns import PatternCandidate
from trading_bot.signals.setup import build_signal

SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded (contract §8).
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_REG = storage.TIMEFRAME_MS[REGIME_TF]
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000

N_SETUP = 80
# A constant-true-range setup series gives an EXACT, hand-checkable Wilder ATR.
# TR is 2.0 on every bar (high - low = 2.0 dominates both gap terms on a flat
# series), so ATR(14) seeds at 2.0 and the recursion keeps it at 2.0 forever.
FLAT_ATR = 2.0

DEFAULTS = dict(atr_multiple=config.ATR_STOP_MULTIPLE, atr_period=config.ATR_STOP_PERIOD)


@pytest.fixture(autouse=True)
def _isolate_caches():
    engine.clear_caches()
    fcontext.clear_caches()
    yield
    engine.clear_caches()
    fcontext.clear_caches()


def seed(conn, timeframe, rows, start=START, interval=D_SET):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    storage.upsert_candles(conn, SYMBOL, timeframe, data)
    return data


def flat_rows(n, close=100.0):
    """Constant true range 2.0 => ATR(14) is exactly 2.0 after warmup."""
    return [[close, close + 1.0, close - 1.0, close, 10.0] for _ in range(n)]


def ctx_with_atr(tmp_path, *, name="t.db", setup_rows=None):
    conn = storage.connect(str(tmp_path / name))
    seed(conn, REGIME_TF, flat_rows(300), interval=D_REG)
    seed(conn, SETUP_TF, setup_rows if setup_rows is not None else flat_rows(N_SETUP),
         interval=D_SET)
    seed(conn, TRIGGER_TF, flat_rows(400), interval=D_TRIG)
    session = fcontext.EvalSession(
        OhlcvSource(conn), SYMBOL,
        tiers=(REGIME_TF, SETUP_TF, TRIGGER_TF), regime_gate=RegimeGate(),
    )
    return session.context("trigger", START + (N_SETUP + 5) * D_SET)


def stamped(direction="long", *, level=100.0, height=5.0, entry=100.3):
    """A DetectedEvent stamped with the trigger facts, as execute.py hands it over."""
    event = DetectedEvent(
        kind="donchian-breakout", direction=direction, level=level,
        target_height=height, start_ts=START, end_ts=START + 10 * D_SET,
    )
    trig = BreakoutEvent(
        ts=START + 11 * D_SET, price=entry, level=level, direction=direction,
        volume_ratio=2.0, volume_high=True,
    )
    return contracts.with_trigger(event, trig)


class TestAtrFixture:
    def test_fixture_atr_is_exactly_two(self, tmp_path):
        """Pins the fixture's premise, so a later ATR assertion that fails points
        at the policy rather than at the fixture."""
        ctx = ctx_with_atr(tmp_path)
        atr = ctx.atr(SETUP_TF, config.ATR_STOP_PERIOD)
        assert math.isclose(float(atr[-1]), FLAT_ATR, rel_tol=0.0, abs_tol=1e-12)


class TestMeasuredMoveLong:
    def test_hand_computed_stop_and_target(self, tmp_path):
        # entry = 100.3, atr = 2.0, k = config.ATR_STOP_MULTIPLE (1.5)
        #   stop   = 100.3 - 1.5 * 2.0        = 97.3
        #   target = level + height = 100.0 + 5.0 = 105.0
        #   risk   = |100.3 - 97.3|           = 3.0
        #   reward = 105.0 - 100.3            = 4.7
        ctx = ctx_with_atr(tmp_path)
        plan = measured_move(ctx, stamped("long"), **DEFAULTS)
        assert plan is not None
        assert math.isclose(plan.stop, 100.3 - config.ATR_STOP_MULTIPLE * FLAT_ATR)
        assert math.isclose(plan.target, 105.0)
        assert math.isclose(plan.entry, 100.3)
        assert plan.direction == "long"

    def test_percentages_are_fractions_of_entry_not_of_level(self, tmp_path):
        """Getting this wrong is invisible until the Filter's arithmetic is off by
        ~1%, so it is asserted numerically rather than trusted."""
        ctx = ctx_with_atr(tmp_path)
        plan = measured_move(ctx, stamped("long"), **DEFAULTS)
        assert math.isclose(plan.risk_pct, 3.0 / 100.3)
        assert math.isclose(plan.reward_pct, 4.7 / 100.3)
        # and NOT fractions of level (100.0), which would give 0.03 / 0.047
        assert not math.isclose(plan.risk_pct, 3.0 / 100.0)

    def test_rr_is_gross(self, tmp_path):
        """PositionPlan.rr is reward_pct / risk_pct, cost-blind. The NET ratio is
        filter.rr-after-costs' output and lands on Trade.planned_rr."""
        ctx = ctx_with_atr(tmp_path)
        plan = measured_move(ctx, stamped("long"), **DEFAULTS)
        assert math.isclose(plan.rr, plan.reward_pct / plan.risk_pct)
        assert math.isclose(plan.rr, 4.7 / 3.0)

    def test_source_is_the_event_kind(self, tmp_path):
        """PositionPlan.source becomes Trade.pattern, so metrics.by_bucket's
        'regime/pattern' keys stay meaningful."""
        ctx = ctx_with_atr(tmp_path)
        plan = measured_move(ctx, stamped("long"), **DEFAULTS)
        assert plan.source == "donchian-breakout"
        assert plan.ts == START + 11 * D_SET
        assert plan.symbol == SYMBOL


class TestMeasuredMoveShort:
    def test_exact_mirror(self, tmp_path):
        # entry = 99.7, atr = 2.0, k = 1.5, level = 100.0, height = 5.0
        #   stop   = 99.7 + 3.0              = 102.7  (ABOVE entry)
        #   target = 100.0 - 5.0             = 95.0
        #   risk   = 3.0 ; reward = 99.7 - 95.0 = 4.7
        ctx = ctx_with_atr(tmp_path)
        plan = measured_move(ctx, stamped("short", entry=99.7), **DEFAULTS)
        assert plan is not None
        assert math.isclose(plan.stop, 99.7 + config.ATR_STOP_MULTIPLE * FLAT_ATR)
        assert plan.stop > plan.entry
        assert math.isclose(plan.target, 95.0)
        assert math.isclose(plan.risk_pct, 3.0 / 99.7)
        assert math.isclose(plan.reward_pct, 4.7 / 99.7)


class TestMeasuredMoveRejections:
    """A rejection returns None and must never raise."""

    def test_nan_atr_returns_none(self, tmp_path):
        """The regression pin for setup.py:131-136's reasoning: a NaN comparison
        is always False, so `atr_value <= 0` alone would let a warmup NaN through
        and produce a stop = nan plan."""
        conn = storage.connect(str(tmp_path / "warmup.db"))
        n_short = config.ATR_STOP_PERIOD - 2  # ATR still NaN at the last bar
        seed(conn, REGIME_TF, flat_rows(300), interval=D_REG)
        seed(conn, SETUP_TF, flat_rows(n_short), interval=D_SET)
        seed(conn, TRIGGER_TF, flat_rows(400), interval=D_TRIG)
        session = fcontext.EvalSession(
            OhlcvSource(conn), SYMBOL,
            tiers=(REGIME_TF, SETUP_TF, TRIGGER_TF), regime_gate=RegimeGate(),
        )
        ctx = session.context("trigger", START + (n_short + 2) * D_SET)
        assert math.isnan(float(ctx.atr(SETUP_TF, config.ATR_STOP_PERIOD)[-1]))
        assert measured_move(ctx, stamped("long"), **DEFAULTS) is None

    def test_zero_atr_multiple_gives_zero_risk_and_returns_none(self, tmp_path):
        """k = 0 puts the stop AT entry, so risk is 0 — rejected, not divided by."""
        ctx = ctx_with_atr(tmp_path)
        plan = measured_move(ctx, stamped("long"), atr_multiple=0.0,
                             atr_period=config.ATR_STOP_PERIOD)
        assert plan is None

    def test_zero_atr_series_returns_none(self, tmp_path):
        """A perfectly flat bar (high == low == close) gives TR 0 => ATR 0."""
        rows = [[100.0, 100.0, 100.0, 100.0, 10.0] for _ in range(N_SETUP)]
        ctx = ctx_with_atr(tmp_path, name="zero.db", setup_rows=rows)
        assert float(ctx.atr(SETUP_TF, config.ATR_STOP_PERIOD)[-1]) == 0.0
        assert measured_move(ctx, stamped("long"), **DEFAULTS) is None

    def test_reward_non_positive_returns_none(self, tmp_path):
        """Entry already at or beyond the target."""
        ctx = ctx_with_atr(tmp_path)
        # level 100, height 5 => target 105; entry 105.5 is already past it.
        assert measured_move(ctx, stamped("long", entry=105.5), **DEFAULTS) is None

    def test_non_positive_entry_or_level_returns_none(self, tmp_path):
        ctx = ctx_with_atr(tmp_path)
        assert measured_move(ctx, stamped("long", entry=0.0), **DEFAULTS) is None
        assert measured_move(ctx, stamped("long", level=0.0), **DEFAULTS) is None

    def test_atr_multiple_override_changes_the_stop(self, tmp_path):
        """Proves the ParamSpec default is resolved by the caller, not hardcoded."""
        ctx = ctx_with_atr(tmp_path)
        a = measured_move(ctx, stamped("long"), **DEFAULTS)
        b = measured_move(ctx, stamped("long"), atr_multiple=1.0,
                          atr_period=config.ATR_STOP_PERIOD)
        assert math.isclose(a.stop, 100.3 - 1.5 * FLAT_ATR)
        assert math.isclose(b.stop, 100.3 - 1.0 * FLAT_ATR)


class TestNoRrScreen:
    """policy.measured-move applies NO reward:risk rejection — Task 8's GOTCHA 2.

    build_signal conflates SL/TP computation with the `rr < rr_floor` screen; the
    contracts split them. If the screen leaked in here it would apply RR_FLOOR =
    1.5 GROSS before the net-2.0 Filter, silently pre-filtering the very
    distribution `graph-backtest --rr-report` measures and making that report a
    report on a truncated sample.
    """

    def test_a_plan_below_rr_floor_is_still_returned(self, tmp_path):
        ctx = ctx_with_atr(tmp_path)
        # height 3.2 => target 103.2, reward 2.9, risk 3.0 => gross rr 0.967,
        # far below RR_FLOOR = 1.5. build_signal would reject; this must not.
        plan = measured_move(ctx, stamped("long", height=3.2), **DEFAULTS)
        assert plan is not None
        assert plan.rr < config.RR_FLOOR
        # and the plug-in declares no rr_floor parameter at all
        from trading_bot.framework import registry
        registry.load_all()
        assert "rr_floor" not in registry.get("policy.measured-move").params


class TestEquivalenceWithBuildSignal:
    """The drift guard: identical numbers in, identical geometry out.

    The BreakoutEvent is constructed DIRECTLY rather than by driving
    check_breakout, so the comparison isolates the arithmetic from the trigger
    logic (the same approach tests/test_signals.py takes).
    """

    @pytest.mark.parametrize("direction,entry", [("long", 100.3), ("short", 99.7)])
    def test_matches_build_signal_geometry(self, tmp_path, direction, entry):
        ctx = ctx_with_atr(tmp_path)
        event = stamped(direction, entry=entry)
        plan = measured_move(ctx, event, **DEFAULTS)
        assert plan is not None

        candidate = PatternCandidate(
            kind=event.kind, direction=direction, breakout_level=event.level,
            target_height=event.target_height, start_ts=event.start_ts,
            end_ts=event.end_ts,
        )
        trig = BreakoutEvent(
            ts=START + 11 * D_SET, price=entry, level=event.level,
            direction=direction, volume_ratio=2.0, volume_high=True,
        )
        sig = build_signal(
            SYMBOL, candidate, trig, FLAT_ATR,
            atr_multiple=config.ATR_STOP_MULTIPLE,
            # rr_floor low enough that build_signal's screen cannot fire, so the
            # comparison is of GEOMETRY, not of the screen the policy omits.
            rr_floor=0.0,
        )
        assert sig is not None
        for field in ("stop", "target", "risk_pct", "reward_pct", "rr", "entry"):
            a, b = getattr(plan, field), getattr(sig, field)
            assert math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15), (
                f"{field}: policy {a!r} != build_signal {b!r}"
            )
        assert plan.ts == sig.ts
        assert plan.direction == sig.direction
        assert plan.source == sig.pattern

    def test_plan_to_signal_round_trips(self, tmp_path):
        """The adapter the executor uses must carry every field across."""
        ctx = ctx_with_atr(tmp_path)
        event = stamped("long")
        plan = measured_move(ctx, event, **DEFAULTS)
        sig = contracts.plan_to_signal(plan, contracts.trigger_from_meta(event))
        back = contracts.plan_from_signal(sig, source=plan.source)
        assert back == plan
        assert isinstance(back, PositionPlan)
