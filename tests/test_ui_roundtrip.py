"""
The success-signal test (v0.3.0 Phase 7): a strategy composed entirely in the
UI round-trips through serialization, backtests, and displays its gate
verdict.

The strategy is composed the way the BROWSER composes one: read
/api/plugins, take each ParamSpec's default, build `stages`, POST it. Nothing
is hand-written as graph JSON, so a Phase 3 format change fails this test
loudly instead of being silently duplicated in the UI.

Fixture-building follows tests/test_framework_parity.py's `seed_scenario` /
`donchian_rows` (imported, not re-derived — the plan's own instruction to
"reuse its helper if importable rather than writing a second one" — `tests`
is a real package here, so this is a plain intra-package import).
"""

import dataclasses

import pytest

from trading_bot import config
from trading_bot.backtest import engine
from trading_bot.backtest.engine import BacktestParams
from trading_bot.backtest.walkforward import GATE_CONDITIONS, walk_forward_pooled
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.execute import run_graph_backtest
from trading_bot.ui import api

from tests.test_framework_parity import (
    D_SET,
    D_TRIG,
    START,
    SYMBOL,
    donchian_rows,
    seed,
)

DETECTOR_KEY = "detector.donchian-breakout"
DETECTOR_KEY2 = "detector.inverse-head-and-shoulders"
POLICY_KEY = "policy.atr-stop-measured-move"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STRATEGY_DIR", str(tmp_path / "strategies"))
    monkeypatch.setattr(config, "STATE_DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.setattr(api, "RUN_ROOT", tmp_path / "ui_runs")
    registry.load_all()
    engine.clear_caches()
    fcontext.clear_caches()
    yield
    engine.clear_caches()
    fcontext.clear_caches()
    api.reset_jobs()


@pytest.fixture
def conn(tmp_path):
    """A synthetic OHLCV store spanning ~19 days: 60 setup-tier (4h) ramp
    bars (GOTCHA #1: small explicit train/test/oos, not the 180/60/90
    production defaults, which need years of bars), then a Donchian breakout
    at the trigger tier followed by a long flat tail so the trade survives
    long enough to close on the time-stop rather than needing exact price
    choreography. 25 regime-tier bars extend coverage past the trigger span."""
    c = storage.connect(str(tmp_path / "ohlcv.db"))
    seed(c, config.REGIME_TIMEFRAME, [[100, 111, 99, 105, 10.0]] * 25,
        start=START, interval=storage.TIMEFRAME_MS[config.REGIME_TIMEFRAME])
    seed(c, config.SIGNAL_PATTERN_TIMEFRAME, donchian_rows(), start=START, interval=D_SET)

    last_setup_ts = START + 59 * D_SET
    level = 160.0
    below = level - 0.8
    entry = level + 1.0
    rows_trig = [[below, below + 0.2, below - 0.2, below, 10.0]] * 21
    rows_trig.append([below, entry + 0.1, below - 0.2, entry, 30.0])  # breakout bar
    rows_trig += [[180.0, 190.0, 181.0, 189.0, 10.0]] * 200  # long flat tail
    seed(
        c, config.SIGNAL_TRIGGER_TIMEFRAME, rows_trig,
        start=last_setup_ts + D_SET, interval=D_TRIG,
    )
    yield c
    c.close()


def _stage_defaults() -> list:
    """Build `stages` the way the BROWSER would: read /api/plugins, take
    each ParamSpec's default."""
    plugins = api.get_plugins().payload["plugins"]
    detector = next(p for p in plugins["detector"] if p["key"] == DETECTOR_KEY)
    policy = next(p for p in plugins["policy"] if p["key"] == POLICY_KEY)
    return [
        {"kind": "detector", "key": DETECTOR_KEY,
         "params": {n: s["default"] for n, s in detector["params"].items()}},
        {"kind": "policy", "key": POLICY_KEY,
         "params": {n: s["default"] for n, s in policy["params"].items()}},
    ]


def _two_detector_stage_defaults() -> list:
    """Build a 2-DETECTOR `stages` list the way the BROWSER would compose a
    multi-detector strategy: read /api/plugins, take each ParamSpec's
    default for BOTH detectors and the shared policy. Proves the multi-
    branch mapping end to end from the composer's own data source, not from
    hand-written graph JSON (same rationale as `_stage_defaults` above)."""
    plugins = api.get_plugins().payload["plugins"]
    detector1 = next(p for p in plugins["detector"] if p["key"] == DETECTOR_KEY)
    detector2 = next(p for p in plugins["detector"] if p["key"] == DETECTOR_KEY2)
    policy = next(p for p in plugins["policy"] if p["key"] == POLICY_KEY)
    return [
        {"kind": "detector", "key": DETECTOR_KEY,
         "params": {n: s["default"] for n, s in detector1["params"].items()}},
        {"kind": "detector", "key": DETECTOR_KEY2,
         "params": {n: s["default"] for n, s in detector2["params"].items()}},
        {"kind": "policy", "key": POLICY_KEY,
         "params": {n: s["default"] for n, s in policy["params"].items()}},
    ]


class TestMultiDetectorComposeRoundTrip:
    """The success signal for the multi-detector feature, mirroring
    TestComposeRoundTrip above one level up: a strategy with TWO detector
    stages, composed exactly the way the browser composes one, must
    round-trip AND actually execute -- proving the graph is real, not merely
    serializable."""

    def test_compose_two_detectors_from_paramspec_defaults_saves_and_reloads_editable(self):
        resp = api.post_strategy("roundtrip-2det", {"stages": _two_detector_stage_defaults()})
        assert resp.status == 200
        assert resp.payload["schema_version"] == fgraph.SCHEMA_VERSION

        g = fgraph.load(fgraph.path_for("roundtrip-2det"))
        assert [b.detector.key for b in g.branches] == [DETECTOR_KEY, DETECTOR_KEY2]

        got = api.get_strategy("roundtrip-2det")
        assert got.status == 200
        assert got.payload["editable"] is True
        detector_stages = [s for s in got.payload["stages"] if s["kind"] == "detector"]
        assert [s["key"] for s in detector_stages] == [DETECTOR_KEY, DETECTOR_KEY2]

    def test_backtest_through_run_graph_backtest_runs_both_branches_and_produces_a_trade(self, conn):
        """The whole point of this file: run_graph_backtest must actually
        WALK a 2-branch graph, not just deserialize it. The donchian branch
        trades on this fixture's breakout (same data as
        TestComposeRoundTrip's single-detector test); the second branch
        legitimately finds nothing in this data and contributes zero trades
        -- both are exercised, and the run must not raise for either."""
        api.post_strategy("roundtrip-2det", {"stages": _two_detector_stage_defaults()})
        g = fgraph.load(fgraph.path_for("roundtrip-2det"))
        trades = run_graph_backtest(conn, g, SYMBOL)
        assert isinstance(trades, list)
        assert len(trades) == 1
        assert trades[0].stop is not None
        assert trades[0].target is not None


class TestComposeRoundTrip:
    def test_compose_from_paramspec_defaults_and_save(self):
        resp = api.post_strategy("roundtrip", {"stages": _stage_defaults()})
        assert resp.status == 200
        assert resp.payload["schema_version"] == fgraph.SCHEMA_VERSION

    def test_saved_file_deserializes_through_from_dict(self):
        save_resp = api.post_strategy("roundtrip", {"stages": _stage_defaults()})
        g = fgraph.load(fgraph.path_for("roundtrip"))
        assert fgraph.graph_hash(g) == save_resp.payload["graph_hash"]

    def test_backtest_through_run_graph_backtest_returns_trades(self, conn):
        api.post_strategy("roundtrip", {"stages": _stage_defaults()})
        g = fgraph.load(fgraph.path_for("roundtrip"))
        trades = run_graph_backtest(conn, g, SYMBOL)
        assert isinstance(trades, list)

    def test_curves_include_the_buy_and_hold_basket(self, conn):
        api.post_strategy("roundtrip", {"stages": _stage_defaults()})
        g = fgraph.load(fgraph.path_for("roundtrip"))
        start_ms = START + 60 * D_SET
        end_ms = start_ms + 8 * 24 * 3600 * 1000
        curves = api._oos_curves(conn, g, [SYMBOL], start_ms, end_ms)
        assert "strategy" in curves
        assert "basket" in curves
        assert curves["strategy"][0] == 1.0

    def test_engine_core_untouched(self):
        """§12.4: zero engine-core edits to add a plug-in. Pinned here: the
        UI added no BacktestParams field and no registry.KINDS entry."""
        field_names = {f.name for f in dataclasses.fields(BacktestParams)}
        assert field_names == {
            "adx_trend_threshold", "atr_extreme_percentile", "bb_num_std",
            "rr_floor", "trail_enabled", "trail_atr_multiple", "target_enabled",
        }
        assert registry.KINDS == (
            "data", "detector", "confirmation", "policy", "filter", "reviewer", "mutator",
        )

    def test_gate_verdict_has_all_seven_conditions_rendered(self, conn):
        """walk_forward_pooled(strategy=graph) -> _serialize_wf_result ->
        exactly len(GATE_CONDITIONS) rows, each with name/ok/measured/
        threshold, plus a benchmark block.

        Asserts SHAPE AND REACHABILITY, never PASS. On this synthetic
        fixture the gate will fail (too few trades for a 30-trade floor), and
        a test demanding a pass would be a test applying pressure to the
        gate — precisely what contract §4 forbids.
        """
        api.post_strategy("roundtrip", {"stages": _stage_defaults()})
        g = fgraph.load(fgraph.path_for("roundtrip"))
        start_ms = START + 60 * D_SET
        end_ms = start_ms + 8 * 24 * 3600 * 1000
        result = walk_forward_pooled(
            conn, [SYMBOL], start_ms=start_ms, end_ms=end_ms, strategy=g,
            grid={"max_hold_bars": (config.MAX_HOLD_BARS_TRIGGER,)},
            train_days=3, test_days=2, oos_days=2, min_trades=1,
        )
        curves = api._oos_curves(conn, g, [SYMBOL], result.oos_start, result.oos_end)
        payload = api._serialize_wf_result(result, curves=curves)
        assert len(payload["gate"]["conditions"]) == len(GATE_CONDITIONS)
        assert payload["gate"]["passed"] == all(result.gate.values())
        assert "strategy" in payload["curves"]
        assert "basket" in payload["curves"]
        for cond in payload["gate"]["conditions"]:
            assert set(cond) == {"name", "ok", "measured", "threshold"}
