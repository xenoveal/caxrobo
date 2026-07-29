"""Tests for framework/contracts.py and framework/context.py (v0.3.0 Phase 3).

The load-bearing class here is TestNoLookahead: EvalContext's whole premise is
that a plug-in cannot see a bar that has not closed, and one off-by-one in
bar_index would be a lookahead nobody would ever see in the numbers.
"""

import math

import numpy as np

import pytest

from trading_bot import config
from trading_bot.backtest import engine
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import contracts
from trading_bot.framework.contracts import (
    ConfirmationVerdict,
    DataSource,
    DetectedEvent,
    FilterVerdict,
    ParamSpec,
    PositionPlan,
    check_callable_shape,
)
from trading_bot.framework.errors import ContractError
from trading_bot.framework.graph import RegimeGate
from trading_bot.plugins.data.ohlcv import OhlcvSource
from trading_bot.regime.classifier import classify_series
from trading_bot.signals.breakout import BreakoutEvent
from trading_bot.signals.patterns import PatternCandidate
from trading_bot.signals.pivots import find_pivots
from trading_bot.signals.setup import Signal

SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded: these fixtures follow config forever, so a
# future tier shift cannot leave the tests on the old timeframes while
# production moves (contract §8).
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_REG = storage.TIMEFRAME_MS[REGIME_TF]
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Clear BOTH memos around every test.

    engine._CACHE and framework.context._CACHE are keyed by content
    fingerprints, so cross-fixture collisions should be impossible — but test
    isolation must not DEPEND on that argument being right.
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


def ramp(n, base=100.0):
    """A monotone ramp whose CLOSE equals base + i, so an off-by-one in an index
    is visible by inspection rather than by arithmetic."""
    return [[base + i, base + i + 2.0, base + i - 1.0, base + i, 10.0] for i in range(n)]


def make_session(conn, *, regime_gate=None):
    return fcontext.EvalSession(
        OhlcvSource(conn),
        SYMBOL,
        tiers=(REGIME_TF, SETUP_TF, TRIGGER_TF),
        regime_gate=regime_gate or RegimeGate(),
    )


@pytest.fixture
def three_tier(tmp_path):
    """A synthetic 3-tier DB with enough bars for real indicator values."""
    conn = storage.connect(str(tmp_path / "t.db"))
    seed(conn, REGIME_TF, ramp(300), interval=D_REG)
    seed(conn, SETUP_TF, ramp(200), interval=D_SET)
    seed(conn, TRIGGER_TF, ramp(400), interval=D_TRIG)
    return conn


class TestParamSpec:
    def test_int_legal_and_illegal(self):
        s = ParamSpec(kind="int", default=20, bounds=(5, 200), doc="period")
        assert s.step == 1
        assert s.is_legal(5) and s.is_legal(200) and s.is_legal(20)
        assert not s.is_legal(4)
        assert not s.is_legal(201)
        assert not s.is_legal(20.5)

    def test_bool_is_not_an_int(self):
        """bool subclasses int in Python, so entry_period=True would otherwise
        validate as the integer 1 and a graph could carry a boolean where a
        period belongs."""
        s = ParamSpec(kind="int", default=20, bounds=(0, 200), doc="period")
        assert s.is_legal(True) is False
        assert s.is_legal(False) is False

    def test_float_rejects_nan_and_bool(self):
        s = ParamSpec(kind="float", default=1.5, bounds=(0.0, 5.0), doc="k")
        assert s.is_legal(1.5) and s.is_legal(3)
        assert not s.is_legal(float("nan"))
        assert not s.is_legal(float("inf"))
        assert not s.is_legal(True)

    def test_bool_kind(self):
        s = ParamSpec(kind="bool", default=False, doc="on")
        assert s.is_legal(True) and s.is_legal(False)
        assert not s.is_legal(1)
        assert not s.is_legal("yes")

    def test_choice_kind(self):
        s = ParamSpec(kind="choice", default="a", choices=("a", "b"), doc="which")
        assert s.is_legal("a") and not s.is_legal("c")

    def test_illegal_default_raises_at_construction(self):
        with pytest.raises(ContractError, match="default"):
            ParamSpec(kind="int", default=999, bounds=(1, 10), doc="x")

    def test_numeric_without_bounds_raises(self):
        with pytest.raises(ContractError, match="requires bounds"):
            ParamSpec(kind="int", default=1, doc="x")
        with pytest.raises(ContractError, match="requires bounds"):
            ParamSpec(kind="float", default=1.0, doc="x")

    def test_choice_without_choices_raises(self):
        with pytest.raises(ContractError, match="non-empty choices"):
            ParamSpec(kind="choice", default="a", doc="x")

    def test_bool_with_bounds_raises(self):
        with pytest.raises(ContractError, match="must not carry bounds"):
            ParamSpec(kind="bool", default=False, bounds=(0, 1), doc="x")

    def test_numeric_with_choices_raises(self):
        with pytest.raises(ContractError, match="must not carry choices"):
            ParamSpec(kind="int", default=1, bounds=(1, 3), choices=(1, 2), doc="x")

    def test_empty_doc_raises(self):
        with pytest.raises(ContractError, match="non-empty doc"):
            ParamSpec(kind="int", default=1, bounds=(1, 3), doc="   ")

    def test_unknown_kind_raises(self):
        with pytest.raises(ContractError, match="not one of"):
            ParamSpec(kind="str", default="a", doc="x")

    def test_list_bounds_or_choices_raise(self):
        with pytest.raises(ContractError, match="must be a TUPLE"):
            ParamSpec(kind="int", default=1, bounds=[1, 3], doc="x")
        with pytest.raises(ContractError, match="must be a TUPLE"):
            ParamSpec(kind="choice", default="a", choices=["a"], doc="x")

    def test_inverted_bounds_raise(self):
        with pytest.raises(ContractError, match="inverted"):
            ParamSpec(kind="int", default=5, bounds=(10, 1), doc="x")

    def test_nonpositive_step_raises(self):
        with pytest.raises(ContractError, match="step must be > 0"):
            ParamSpec(kind="float", default=1.0, bounds=(0.0, 2.0), step=0.0, doc="x")

    def test_clamp_below_above_inside(self):
        s = ParamSpec(kind="int", default=20, bounds=(5, 200), doc="x")
        assert s.clamp(-3) == 5
        assert s.clamp(9999) == 200
        assert s.clamp(30) == 30

    def test_clamp_nan_falls_back_to_default(self):
        s = ParamSpec(kind="float", default=1.5, bounds=(1.0, 5.0), doc="x")
        assert s.clamp(float("nan")) == 1.5
        assert s.clamp("nonsense") == 1.5

    def test_clamp_choice_miss_falls_back_to_default(self):
        s = ParamSpec(kind="choice", default="a", choices=("a", "b"), doc="x")
        assert s.clamp("zzz") == "a"

    def test_clamp_bool_coerces(self):
        s = ParamSpec(kind="bool", default=False, doc="x")
        assert s.clamp(1) is True
        assert s.clamp(0) is False

    def test_clamp_snaps_to_step(self):
        s = ParamSpec(kind="float", default=1.0, bounds=(0.0, 2.0), step=0.5, doc="x")
        assert s.clamp(1.3) == 1.5
        assert s.clamp(1.1) == 1.0

    def test_check_message_names_where(self):
        s = ParamSpec(kind="int", default=20, bounds=(5, 200), doc="x")
        with pytest.raises(ContractError, match=r"detector\.foo\.entry_period"):
            s.check(4, where="detector.foo.entry_period")

    def test_check_coerces_int_to_float_for_float_kind(self):
        """20 and 20.0 must not be able to produce two graph hashes."""
        s = ParamSpec(kind="float", default=2.0, bounds=(0.0, 5.0), doc="x")
        got = s.check(3, where="w")
        assert isinstance(got, float) and got == 3.0


class TestAdapters:
    def _candidate(self):
        return PatternCandidate(
            kind="donchian-breakout",
            direction="long",
            breakout_level=160.0,
            target_height=22.0,
            start_ts=START,
            end_ts=START + D_SET,
        )

    def _trigger(self):
        return BreakoutEvent(
            ts=START + 5 * D_TRIG,
            price=161.0,
            level=160.0,
            direction="long",
            volume_ratio=2.25,
            volume_high=True,
        )

    def test_event_candidate_round_trip_all_six_fields(self):
        c = self._candidate()
        e = contracts.event_from_candidate(c)
        assert (e.kind, e.direction, e.level, e.target_height, e.start_ts, e.end_ts) == (
            c.kind, c.direction, c.breakout_level, c.target_height, c.start_ts, c.end_ts,
        )
        back = contracts.candidate_from_event(e)
        assert back == c

    def test_meta_is_dropped_in_the_reverse_direction(self):
        """PatternCandidate has nowhere to put meta, and check_breakout never
        reads it — so the drop is documented, not accidental."""
        e = contracts.event_from_candidate(self._candidate(), meta={"adx": 30.0})
        assert e.meta == {"adx": 30.0}
        back = contracts.candidate_from_event(e)
        assert not hasattr(back, "meta")

    def test_plan_signal_round_trip_is_lossless(self):
        sig = Signal(
            symbol=SYMBOL, ts=START, direction="long", pattern="donchian-breakout",
            entry=161.0, stop=156.5, target=182.0, risk_pct=0.02795,
            reward_pct=0.13043, rr=4.666, volume_ratio=2.25, volume_high=True,
        )
        plan = contracts.plan_from_signal(sig, source=sig.pattern)
        again = contracts.plan_to_signal(plan, self._trigger())
        assert again == sig

    def test_plan_source_lands_on_signal_pattern(self):
        plan = PositionPlan(
            symbol=SYMBOL, ts=START, direction="short", entry=100.0, stop=102.0,
            target=95.0, risk_pct=0.02, reward_pct=0.05, rr=2.5, source="bollinger-fade",
        )
        sig = contracts.plan_to_signal(
            plan,
            BreakoutEvent(
                ts=START, price=100.0, level=100.0, direction="short",
                volume_ratio=float("nan"), volume_high=False,
            ),
        )
        assert sig.pattern == "bollinger-fade"

    def test_plan_to_signal_requires_the_event(self):
        """volume_ratio/volume_high live ONLY on the trigger event; defaulting
        them would silently zero Trade.volume_high and break parity in a way no
        arithmetic check would catch."""
        plan = PositionPlan(
            symbol=SYMBOL, ts=START, direction="long", entry=1.0, stop=0.5,
            target=2.0, risk_pct=0.5, reward_pct=1.0, rr=2.0, source="k",
        )
        with pytest.raises(TypeError):
            contracts.plan_to_signal(plan)  # type: ignore[call-arg]

    def test_with_trigger_does_not_mutate_and_round_trips(self):
        e = contracts.event_from_candidate(self._candidate(), meta={"adx": 30.0})
        trig = self._trigger()
        stamped = contracts.with_trigger(e, trig)
        assert e.meta == {"adx": 30.0}, "with_trigger must not mutate the event"
        assert stamped.meta["adx"] == 30.0
        assert contracts.trigger_from_meta(stamped) == trig

    def test_trigger_from_meta_raises_when_unstamped(self):
        e = contracts.event_from_candidate(self._candidate())
        with pytest.raises(ContractError, match="no trigger facts"):
            contracts.trigger_from_meta(e)


class TestCallableShape:
    def test_correct_detector_passes(self):
        def d(ctx, *, period=1):
            return []

        check_callable_shape("detector", d, key="detector.x")

    def test_wrong_argument_order_raises_naming_the_key(self):
        def d(df, ctx):
            return []

        with pytest.raises(ContractError, match="detector.x"):
            check_callable_shape("detector", d, key="detector.x")

    def test_policy_missing_event_raises(self):
        def p(ctx, *, k=1):
            return None

        with pytest.raises(ContractError, match=r"\('ctx', 'event'\)"):
            check_callable_shape("policy", p, key="policy.x")

    def test_extra_positional_param_must_be_keyword_only(self):
        def d(ctx, period):
            return []

        with pytest.raises(ContractError, match="KEYWORD-ONLY"):
            check_callable_shape("detector", d, key="detector.x")

    def test_positional_only_extra_param_raises(self):
        def d(ctx, *, period=1):
            return []

        # A genuinely positional-only extra parameter.
        src = "def d2(ctx, period=1, /):\n    return []\n"
        ns: dict = {}
        exec(src, ns)
        with pytest.raises(ContractError):
            check_callable_shape("detector", ns["d2"], key="detector.x")

    def test_unknown_kind_raises(self):
        with pytest.raises(ContractError, match="unknown plug-in kind"):
            check_callable_shape("nope", lambda ctx: [], key="nope.x")


class TestEvalContextSurface:
    """The surface must not grow a lookahead hole by accident."""

    EXPECTED = {
        "symbol", "role", "now_ms", "tiers", "interval_ms", "bar_index",
        "frame", "window", "latest", "regime", "series", "atr",
    }

    def test_public_surface_is_exactly_documented(self, three_tier):
        ctx = make_session(three_tier).context("setup", START + 100 * D_SET)
        public = {n for n in dir(ctx) if not n.startswith("_")}
        assert public == self.EXPECTED

    def test_no_conn_no_frames_no_clock_no_rng(self, three_tier):
        ctx = make_session(three_tier).context("setup", START + 100 * D_SET)
        for forbidden in ("conn", "_conn", "frames", "time", "rng", "random"):
            assert not hasattr(ctx, forbidden), forbidden

    def test_no_accessor_returns_an_untruncated_frame(self, three_tier):
        session = make_session(three_tier)
        full = len(session.frame_of(SETUP_TF))
        ctx = session.context("setup", START + 100 * D_SET)
        assert len(ctx.frame(SETUP_TF)) < full


class TestNoLookahead:
    def test_frame_excludes_the_forming_bar(self, three_tier):
        session = make_session(three_tier)
        for k in range(20, 200, 7):
            now = START + k * D_SET + D_SET // 3  # mid-bar: bar k is FORMING
            ctx = session.context(f"probe{k}", now)
            df = ctx.frame(SETUP_TF)
            assert int(df.index[-1]) + D_SET <= now

    def test_frame_includes_a_bar_closing_exactly_at_now(self, three_tier):
        """Matches setup.py:209-210 / classifier.py:181-184: a bar is closed
        when ts + interval <= now_ms, inclusive."""
        session = make_session(three_tier)
        k = 50
        now = START + k * D_SET + D_SET  # exactly bar k's close
        ctx = session.context("setup", now)
        assert ctx.bar_index(SETUP_TF) == k
        assert int(ctx.frame(SETUP_TF).index[-1]) == START + k * D_SET

    def test_bar_index_matches_searchsorted_rule(self, three_tier):
        session = make_session(three_tier)
        closes = session.closes(SETUP_TF)
        for k in range(50):
            now = START + k * 3 * D_SET + 11
            ctx = session.context(f"p{k}", now)
            expected = int(np.searchsorted(closes, now, side="right")) - 1
            assert ctx.bar_index(SETUP_TF) == expected

    def test_bar_index_is_minus_one_before_any_close(self, three_tier):
        session = make_session(three_tier)
        ctx = session.context("setup", START)  # bar 0 has not closed yet
        assert ctx.bar_index(SETUP_TF) == -1
        assert len(ctx.frame(SETUP_TF)) == 0
        assert len(ctx.window(SETUP_TF, 10)) == 0
        assert ctx.latest(SETUP_TF) is None

    def test_window_equals_engine_slice(self, three_tier):
        session = make_session(three_tier)
        df = session.frame_of(SETUP_TF)
        for k in (30, 90, 150):
            ctx = session.context(f"w{k}", START + k * D_SET + D_SET)
            got = ctx.window(SETUP_TF, config.PATTERN_LOOKBACK_BARS)
            want = df.iloc[max(0, k + 1 - config.PATTERN_LOOKBACK_BARS) : k + 1]
            assert got.equals(want)

    def test_regime_matches_engine_regime_at(self, three_tier):
        session = make_session(three_tier)
        labels = classify_series(
            session.frame_of(REGIME_TF),
            adx_trend_threshold=config.ADX_TREND_THRESHOLD,
            atr_extreme_percentile=config.ATR_EXTREME_PERCENTILE,
        )
        close_regime = session.closes(REGIME_TF)

        def engine_regime_at(t):
            k = int(np.searchsorted(close_regime, t, side="right")) - 1
            return str(labels.iloc[k]) if k >= 0 else "uncertain"

        for k in range(0, 200, 9):
            t = START + k * D_SET + D_SET
            ctx = session.context(f"r{k}", t)
            assert ctx.regime() == engine_regime_at(t)

    def test_regime_is_uncertain_before_any_regime_bar_closes(self, three_tier):
        session = make_session(three_tier)
        ctx = session.context("early", START)  # no regime bar has closed yet
        assert ctx.bar_index(REGIME_TF) == -1
        assert ctx.regime() == "uncertain"
        assert ctx.latest(REGIME_TF) is None

    def test_returned_array_is_read_only(self, three_tier):
        session = make_session(three_tier)
        ctx = session.context("setup", START + 100 * D_SET + D_SET)
        arr = ctx.atr(SETUP_TF, config.ATR_STOP_PERIOD)
        with pytest.raises(ValueError):
            arr[0] = 123.0

    def test_series_is_truncated_at_the_current_bar(self, three_tier):
        session = make_session(three_tier)
        k = 60
        ctx = session.context("setup", START + k * D_SET + D_SET)
        arr = ctx.series(SETUP_TF, "close", lambda df: df["close"])
        assert len(arr) == k + 1
        assert arr[-1] == pytest.approx(100.0 + k)

    def test_now_ms_must_not_go_backwards_per_role(self, three_tier):
        session = make_session(three_tier)
        session.context("setup", START + 10 * D_SET)
        with pytest.raises(ContractError, match="role 'setup'"):
            session.context("setup", START + 9 * D_SET)

    def test_roles_advance_independently(self, three_tier):
        """A setup-bar context legitimately trails the previous trigger-bar one."""
        session = make_session(three_tier)
        session.context("trigger", START + 100 * D_TRIG)
        session.context("setup", START + 2 * D_SET)  # earlier, different role: legal

    def test_pivots_confirmed_only_with_span_closed_bars(self, tmp_path):
        """Inherited from truncation (pivots.py:7-13): find_pivots on a frame
        ending at the last CLOSED bar cannot emit a pivot inside PIVOT_SPAN of
        that bar."""
        conn = storage.connect(str(tmp_path / "p.db"))
        rows = []
        for i in range(80):
            base = 100.0 + (i % 7)
            rows.append([base, base + 1.0, base - 1.0, base, 10.0])
        seed(conn, REGIME_TF, ramp(300), interval=D_REG)
        seed(conn, SETUP_TF, rows, interval=D_SET)
        seed(conn, TRIGGER_TF, ramp(100), interval=D_TRIG)
        session = make_session(conn)
        k = 60
        ctx = session.context("setup", START + k * D_SET + D_SET)
        window = ctx.window(SETUP_TF, config.PATTERN_LOOKBACK_BARS)
        n = len(window)
        for p in find_pivots(window):
            assert p.index <= n - 1 - config.PIVOT_SPAN

    def test_assert_interval_runs_on_load(self, tmp_path):
        """A 15m series seeded under the setup-tier key raises ValueError with
        engine's exact message."""
        conn = storage.connect(str(tmp_path / "bad.db"))
        seed(conn, REGIME_TF, ramp(300), interval=D_REG)
        seed(conn, SETUP_TF, ramp(60), interval=storage.TIMEFRAME_MS["15m"])
        seed(conn, TRIGGER_TF, ramp(100), interval=D_TRIG)
        with pytest.raises(ValueError, match="series spacing is"):
            make_session(conn)


class TestSeriesCache:
    def test_identical_calls_hit(self, three_tier):
        session = make_session(three_tier)
        ctx = session.context("setup", START + 100 * D_SET + D_SET)
        fcontext.clear_caches()
        ctx.atr(SETUP_TF, 14)
        assert fcontext.cache_stats()["hits"] == 0
        ctx.atr(SETUP_TF, 14)
        assert fcontext.cache_stats()["hits"] == 1

    def test_period_is_in_the_atr_key(self, tmp_path):
        """engine.py:293's key OMITS period because ATR_STOP_PERIOD is
        config-fixed there. The graph exposes atr_period as a ParamSpec, so
        omitting it would serve ATR(14) to a graph asking for ATR(21).

        Needs a VARYING true range: the plain ramp has a constant range of 3, so
        ATR(14) and ATR(21) coincide there and the test would pass vacuously.
        """
        conn = storage.connect(str(tmp_path / "v.db"))
        rows = [
            [100.0 + i, 100.0 + i + 1.0 + (i % 9), 100.0 + i - 1.0 - (i % 5), 100.0 + i, 10.0]
            for i in range(200)
        ]
        seed(conn, REGIME_TF, ramp(300), interval=D_REG)
        seed(conn, SETUP_TF, rows, interval=D_SET)
        seed(conn, TRIGGER_TF, ramp(400), interval=D_TRIG)
        ctx = make_session(conn).context("setup", START + 100 * D_SET + D_SET)
        fcontext.clear_caches()
        a = ctx.atr(SETUP_TF, 14)
        b = ctx.atr(SETUP_TF, 21)
        assert fcontext.cache_stats()["hits"] == 0, "different period must MISS"
        assert not math.isclose(float(a[-1]), float(b[-1]))

    def test_params_are_in_the_generic_series_key(self, three_tier):
        session = make_session(three_tier)
        ctx = session.context("setup", START + 100 * D_SET + D_SET)

        def sma(df, *, period):
            return df["close"].rolling(period, min_periods=period).mean()

        fcontext.clear_caches()
        a = ctx.series(SETUP_TF, "sma", sma, period=5)
        b = ctx.series(SETUP_TF, "sma", sma, period=20)
        assert fcontext.cache_stats()["hits"] == 0
        assert not math.isclose(float(a[-1]), float(b[-1]))
        ctx.series(SETUP_TF, "sma", sma, period=5)
        assert fcontext.cache_stats()["hits"] == 1

    def test_equal_shapes_different_prices_do_not_collide(self, tmp_path):
        """Mirrors test_backtest.py::TestIndicatorMemo: equal bar counts and
        timestamps with different prices must not share a cache entry."""
        c1 = storage.connect(str(tmp_path / "a.db"))
        seed(c1, REGIME_TF, ramp(300), interval=D_REG)
        seed(c1, SETUP_TF, ramp(200, base=100.0), interval=D_SET)
        seed(c1, TRIGGER_TF, ramp(400), interval=D_TRIG)
        c2 = storage.connect(str(tmp_path / "b.db"))
        seed(c2, REGIME_TF, ramp(300), interval=D_REG)
        seed(c2, SETUP_TF, [[o * 2, h * 2, low * 2, c * 2, v] for o, h, low, c, v in ramp(200)], interval=D_SET)
        seed(c2, TRIGGER_TF, ramp(400), interval=D_TRIG)

        now = START + 150 * D_SET + D_SET
        a = make_session(c1).context("setup", now).atr(SETUP_TF, 14)
        b = make_session(c2).context("setup", now).atr(SETUP_TF, 14)
        assert not math.isclose(float(a[-1]), float(b[-1]))

    def test_cache_disabled_gives_equal_values_with_zero_hits(self, three_tier, monkeypatch):
        session = make_session(three_tier)
        ctx = session.context("setup", START + 100 * D_SET + D_SET)
        want = float(ctx.atr(SETUP_TF, 14)[-1])
        fcontext.clear_caches()
        monkeypatch.setattr(config, "FRAMEWORK_CACHE_ENABLED", False)
        got1 = float(ctx.atr(SETUP_TF, 14)[-1])
        got2 = float(ctx.atr(SETUP_TF, 14)[-1])
        assert fcontext.cache_stats()["hits"] == 0
        assert got1 == got2 == want


class TestAssertTrailingOnly:
    def test_trailing_factory_passes(self, three_tier):
        df = make_session(three_tier).frame_of(SETUP_TF)
        fcontext.assert_trailing_only(
            lambda d, *, period: d["close"].rolling(period, min_periods=period).mean(),
            df,
            period=5,
        )

    def test_rejects_shift_minus_one(self, three_tier):
        df = make_session(three_tier).frame_of(SETUP_TF)
        with pytest.raises(ContractError, match="not trailing-only"):
            fcontext.assert_trailing_only(lambda d: d["close"].shift(-1), df)

    def test_rejects_centred_rolling_mean(self, three_tier):
        df = make_session(three_tier).frame_of(SETUP_TF)
        with pytest.raises(ContractError, match="not trailing-only"):
            fcontext.assert_trailing_only(
                lambda d: d["close"].rolling(5, center=True, min_periods=1).mean(), df
            )


class TestDataSourceProtocol:
    def test_ohlcv_source_satisfies_the_protocol(self, three_tier):
        assert isinstance(OhlcvSource(three_tier), DataSource)

    def test_an_object_without_frame_does_not(self):
        class Nope:
            def timeframes(self):
                return ()

        assert not isinstance(Nope(), DataSource)

    def test_timeframes_tiers_and_all(self, three_tier):
        src = OhlcvSource(three_tier)
        assert src.timeframes() == (REGIME_TF, SETUP_TF, TRIGGER_TF)
        every = OhlcvSource(three_tier, expose="all").timeframes()
        assert set(every) == {REGIME_TF, SETUP_TF, TRIGGER_TF}

    def test_frame_bounds_are_inclusive(self, three_tier):
        """storage.load_candles' contract (storage.py:216-217)."""
        src = OhlcvSource(three_tier)
        lo = START + 5 * D_SET
        hi = START + 9 * D_SET
        df = src.frame(SYMBOL, SETUP_TF, start_ms=lo, end_ms=hi)
        assert int(df.index[0]) == lo and int(df.index[-1]) == hi and len(df) == 5


class TestPayloadDataclasses:
    def test_frozen(self):
        e = DetectedEvent("k", "long", 1.0, 1.0, 1, 2)
        with pytest.raises(Exception):
            e.level = 2.0  # type: ignore[misc]

    def test_defaults(self):
        assert DetectedEvent("k", "long", 1.0, 1.0, 1, 2).meta == {}
        assert ConfirmationVerdict(True, "n").score == 0.0
        assert FilterVerdict(True, "n").measured == {}

    def test_detected_event_equality_is_used_by_candidate_parity(self):
        a = DetectedEvent("k", "long", 1.0, 2.0, 1, 2, {"x": 1.0})
        b = DetectedEvent("k", "long", 1.0, 2.0, 1, 2, {"x": 1.0})
        assert a == b
