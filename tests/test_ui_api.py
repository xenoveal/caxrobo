"""
Tests for trading_bot.ui.api.

Calls api.py's handlers DIRECTLY. No socket, no server, no browser — the
api/server split (contract §8) exists so this file can exist.
"""

import dataclasses
import json
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from trading_bot import config
from trading_bot.data import storage
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.plugins import build_v020_graph
from trading_bot.ui import api

from tests.test_framework_parity import D_SET, D_TRIG, START, SYMBOL, donchian_rows, seed

DETECTOR_KEY = "detector.donchian-breakout"
DETECTOR_KEY2 = "detector.inverse-head-and-shoulders"
CONFIRMATION_KEY = "confirmation.volume-breakout"
POLICY_KEY = "policy.atr-stop-measured-move"

# A window wide enough to act as "no bound at all" (mirrors run_graph_backtest's
# own start_ms=None/end_ms=None "start/end of data" default) so a member's
# frozen evaluation window never clips the one synthetic trade `ohlcv_conn`
# produces.
_REPLAY_WINDOW_START_MS = 0
_REPLAY_WINDOW_END_MS = 9_999_999_999_999


@pytest.fixture(autouse=True)
def _isolate_ui(tmp_path, monkeypatch):
    """Point the UI's every filesystem/DB seam at tmp_path, and drain the job
    table before AND after: a leaked job thread poisons every test after it,
    and a test writing into the real data/strategies/ will one day overwrite
    a real strategy."""
    monkeypatch.setattr(config, "STRATEGY_DIR", str(tmp_path / "strategies"))
    monkeypatch.setattr(config, "STATE_DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.setattr(api, "RUN_ROOT", tmp_path / "ui_runs")
    registry.load_all()
    api.reset_jobs()
    yield
    api.reset_jobs()
    assert api._THREADS == {}, "a job thread leaked past its test"


@pytest.fixture
def ohlcv_conn(tmp_path, monkeypatch):
    """The evolution-replay endpoint reads candles from `config.DB_PATH`
    directly (`storage.connect(config.DB_PATH)` inside `get_evolution_replay`)
    -- the one filesystem seam `_isolate_ui` above does NOT patch, because no
    other endpoint touches the OHLCV store. Without this fixture a replay
    test would silently read the developer's real data/ohlcv.db, which would
    make the test machine-dependent and non-hermetic.

    Mirrors tests/test_ui_roundtrip.py's `conn` fixture: the same Donchian
    ramp + breakout + long flat tail, built from
    tests/test_framework_parity.py's own synthetic-OHLCV builders (imported,
    not re-derived), so a graph built from DETECTOR_KEY/POLICY_KEY produces
    exactly one real, closed trade to assert on.
    """
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "replay_ohlcv.db"))
    c = storage.connect(config.DB_PATH)
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


def _stages():
    return [
        {"kind": "detector", "key": DETECTOR_KEY, "params": {}},
        {"kind": "policy", "key": POLICY_KEY, "params": {}},
    ]


class TestPlugins:
    def test_every_kind_present(self):
        resp = api.get_plugins()
        assert resp.status == 200
        for k in registry.KINDS:
            assert k in resp.payload["plugins"]

    def test_every_rationale_non_empty(self):
        resp = api.get_plugins()
        empty = [
            p["key"]
            for lst in resp.payload["plugins"].values()
            for p in lst
            if not p["rationale"].strip()
        ]
        assert empty == []

    def test_paramspec_to_control_bounds(self):
        resp = api.get_plugins()
        for lst in resp.payload["plugins"].values():
            for p in lst:
                for spec in p["params"].values():
                    if spec["kind"] == "bool":
                        assert "min" not in spec and "max" not in spec and "step" not in spec
                    elif spec["kind"] == "choice":
                        assert "choices" in spec and "min" not in spec
                    else:
                        assert "min" in spec and "max" in spec and "step" in spec


class TestStrategyIO:
    def test_traversal_name_rejected(self):
        resp = api.post_strategy("../../etc/passwd", {"stages": _stages()})
        assert resp.status == 400
        assert not Path(config.STRATEGY_DIR).exists() or not any(
            Path(config.STRATEGY_DIR).glob("*passwd*")
        )

    def test_uppercase_name_rejected(self):
        resp = api.post_strategy("MyStrategy", {"stages": _stages()})
        assert resp.status == 400

    def test_unknown_plugin_key_rejected_and_no_file_written(self):
        resp = api.post_strategy(
            "t1",
            {
                "stages": [
                    {"kind": "detector", "key": "detector.nope", "params": {}},
                    {"kind": "policy", "key": POLICY_KEY, "params": {}},
                ]
            },
        )
        assert resp.status == 400
        assert not (Path(config.STRATEGY_DIR) / "t1.strategy.json").exists()

    def test_save_round_trip_sets_schema_version_and_matches_on_reload(self):
        resp = api.post_strategy("t2", {"stages": _stages()})
        assert resp.status == 200
        assert resp.payload["schema_version"] == fgraph.SCHEMA_VERSION
        got = api.get_strategy("t2")
        assert got.status == 200
        assert got.payload["graph_hash"] == resp.payload["graph_hash"]
        assert got.payload["editable"] is True
        # _graph_to_stages prepends the implicit "data" stage the linear
        # composer defaulted (it was never in the POSTed body).
        assert got.payload["stages"] == [
            {"kind": "data", "key": "data.ohlcv", "params": {}}
        ] + _stages()

    def test_non_linear_graph_loads_read_only(self):
        g = build_v020_graph(name="two-branch")
        fgraph.save(g)
        got = api.get_strategy("two-branch")
        assert got.status == 200
        assert got.payload["editable"] is False

    def test_overwriting_a_non_linear_graph_is_refused_and_file_is_intact(self):
        """DATA LOSS REGRESSION. `editable=False` was advisory to the page only:
        the write path accepted a flattened graph and silently discarded the
        other branches. Observed for real -- thin-slice went 2 branches -> 1
        from a single POST -- so the refusal is pinned here."""
        fgraph.save(build_v020_graph(name="two-branch"))
        n_before = len(fgraph.load(str(fgraph.path_for("two-branch"))).branches)
        assert n_before > 1

        resp = api.post_strategy("two-branch", {"stages": _stages()})

        assert resp.status == 409
        assert "refusing to overwrite" in resp.payload["error"]
        after = fgraph.load(str(fgraph.path_for("two-branch")))
        assert len(after.branches) == n_before  # nothing was discarded

    def test_allow_flatten_permits_the_overwrite_explicitly(self):
        """The guard is a safety catch, not a wall: an explicit opt-in still works."""
        fgraph.save(build_v020_graph(name="two-branch"))
        resp = api.post_strategy(
            "two-branch", {"stages": _stages(), "allow_flatten": True}
        )
        assert resp.status == 200
        assert len(fgraph.load(str(fgraph.path_for("two-branch"))).branches) == 1

    def test_overwriting_a_linear_graph_is_still_allowed(self):
        """The guard must not block the ordinary edit-and-save loop."""
        assert api.post_strategy("t3", {"stages": _stages()}).status == 200
        assert api.post_strategy("t3", {"stages": _stages()}).status == 200

    def test_missing_strategy_404(self):
        resp = api.get_strategy("does-not-exist")
        assert resp.status == 404

    def test_list_strategies(self):
        api.post_strategy("t3", {"stages": _stages()})
        resp = api.list_strategies()
        assert "t3" in resp.payload["strategies"]

    def test_validate_without_writing(self):
        resp = api.post_validate({"stages": _stages()})
        assert resp.status == 200
        assert resp.payload["valid"] is True
        assert not Path(config.STRATEGY_DIR).exists() or list(
            Path(config.STRATEGY_DIR).glob("*.strategy.json")
        ) == []


class TestMultiDetectorComposition:
    """N detector stages become N branches (one detector per branch), all
    sharing the composed confirmations/policy -- Task 4's core mapping."""

    def _two_detector_stages(self):
        # Order matches a verified fixture: [detector, detector, confirmation,
        # policy] with no explicit data stage, so node ids come out as
        # detector-0/detector-1/confirmation-2/policy-3 (+ their
        # branch-1 clones) exactly, and this test can pin them precisely
        # rather than merely asserting "some ids, uniquely".
        return [
            {"kind": "detector", "key": DETECTOR_KEY, "params": {}},
            {"kind": "detector", "key": DETECTOR_KEY2, "params": {}},
            {"kind": "confirmation", "key": CONFIRMATION_KEY, "params": {}},
            {"kind": "policy", "key": POLICY_KEY, "params": {}},
        ]

    def test_two_detector_stages_produce_two_branches_with_unique_node_ids(self):
        stages = self._two_detector_stages()
        resp = api.post_strategy("two-detectors", {"stages": stages})
        assert resp.status == 200

        g = fgraph.load(fgraph.path_for("two-detectors"))
        assert [b.id for b in g.branches] == ["main", "inverse-head-and-shoulders"]

        node_ids = [g.data.id]
        for b in g.branches:
            node_ids.append(b.detector.id)
            node_ids.append(b.policy.id)
            node_ids.extend(c.id for c in b.confirmations)
        assert len(node_ids) == len(set(node_ids)), "node ids must be graph-unique"
        assert set(node_ids) == {
            "data", "detector-0", "detector-1", "confirmation-2", "policy-3",
            "inverse-head-and-shoulders-confirmation-2",
            "inverse-head-and-shoulders-policy-3",
        }

        validated = api.post_validate({"stages": stages})
        assert validated.status == 200
        assert validated.payload["valid"] is True

    def test_get_of_saved_two_detector_strategy_is_editable_and_meta_round_trips(self):
        stages = self._two_detector_stages()
        meta = {"evo": {"eligible_detectors": [DETECTOR_KEY, DETECTOR_KEY2]}}
        resp = api.post_strategy("two-detectors-meta", {"stages": stages, "meta": meta})
        assert resp.status == 200

        got = api.get_strategy("two-detectors-meta")
        assert got.status == 200
        assert got.payload["editable"] is True
        detector_stages = [s for s in got.payload["stages"] if s["kind"] == "detector"]
        assert [s["key"] for s in detector_stages] == [DETECTOR_KEY, DETECTOR_KEY2]
        assert got.payload["meta"] == meta


class TestMetaHandling:
    """meta.evo.eligible_detectors is validated at save time (a stale or
    typo'd detector key would otherwise be a constraint evolution can never
    satisfy), and every OTHER strategy must keep saving with no meta key at
    all, exactly as before this feature existed."""

    def test_unregistered_eligible_detector_key_rejected_and_nothing_written(self):
        resp = api.post_strategy(
            "elig-bad",
            {"stages": _stages(), "meta": {"evo": {"eligible_detectors": ["detector.nope"]}}},
        )
        assert resp.status == 400
        assert not (Path(config.STRATEGY_DIR) / "elig-bad.strategy.json").exists()

    def test_valid_eligible_detectors_list_is_accepted_and_persists_to_disk(self):
        meta = {"evo": {"eligible_detectors": [DETECTOR_KEY, DETECTOR_KEY2]}}
        resp = api.post_strategy("elig-good", {"stages": _stages(), "meta": meta})
        assert resp.status == 200
        g = fgraph.load(fgraph.path_for("elig-good"))
        assert g.to_dict()["meta"] == meta

    def test_single_detector_strategy_saves_with_empty_meta_when_none_sent(self):
        """Regression: a strategy composed with the ORIGINAL single-detector
        Composer, which never sends a `meta` key at all, must keep saving
        byte-identically to before this feature -- i.e. `meta == {}`, not a
        default eligibility list or anything else invented on its behalf."""
        resp = api.post_strategy("single-meta", {"stages": _stages()})
        assert resp.status == 200
        g = fgraph.load(fgraph.path_for("single-meta"))
        assert g.to_dict()["meta"] == {}


class TestDivergentConfirmationsGuard:
    """`_graph_to_stages`'s uniformity check compares policy, confirmations,
    regimes AND exits -- this pins the confirmations axis in isolation: two
    branches sharing the identical policy, regimes and exits but DIFFERENT
    per-branch confirmations must still come back non-uniform, or a
    per-branch confirmation constraint (e.g. an evolved champion) would be
    silently flattened on the next save."""

    def _build_divergent_confirmations_graph(self, name="divergent-conf"):
        data = fgraph.NodeSpec(id="data", key="data.ohlcv", params={})
        branch_a = fgraph.Branch(
            id="main",
            detector=fgraph.NodeSpec(id="detector-a", key=DETECTOR_KEY, params={}),
            policy=fgraph.NodeSpec(id="policy-a", key=POLICY_KEY, params={}),
            confirmations=(fgraph.NodeSpec(id="confirmation-a", key="confirmation.macd", params={}),),
        )
        branch_b = fgraph.Branch(
            id="second",
            detector=fgraph.NodeSpec(id="detector-b", key=DETECTOR_KEY, params={}),
            policy=fgraph.NodeSpec(id="policy-b", key=POLICY_KEY, params={}),
            confirmations=(fgraph.NodeSpec(id="confirmation-b", key=CONFIRMATION_KEY, params={}),),
        )
        return fgraph.StrategyGraph(name=name, data=data, branches=(branch_a, branch_b))

    def test_divergent_confirmations_graph_loads_read_only(self):
        fgraph.save(self._build_divergent_confirmations_graph())
        got = api.get_strategy("divergent-conf")
        assert got.status == 200
        assert got.payload["editable"] is False

    def test_overwriting_divergent_confirmations_graph_is_refused_and_file_intact(self):
        fgraph.save(self._build_divergent_confirmations_graph())
        before = fgraph.load(str(fgraph.path_for("divergent-conf"))).to_dict()
        assert len(before["branches"]) == 2

        resp = api.post_strategy("divergent-conf", {"stages": _stages()})

        assert resp.status == 409
        assert "refusing to overwrite" in resp.payload["error"]
        after = fgraph.load(str(fgraph.path_for("divergent-conf"))).to_dict()
        assert after == before  # nothing was discarded or altered


class TestUniformityAxisPinning:
    """`_graph_to_stages`'s `sig()` compares FOUR axes -- policy,
    confirmations, regimes, exits -- plus a separate disabled-branch guard.
    `TestDivergentConfirmationsGuard` above pins the confirmations axis; this
    class pins the other three axes AND the disabled-branch guard, each in
    ISOLATION (two branches identical on every OTHER axis), so that removing
    any ONE comparison from `sig()` is caught here. `build_v020_graph`
    cannot do this job: its two branches differ on several axes at once, so
    it only catches removing the check entirely, not a single-axis
    regression.

    The positive control (`test_uniform_graph_on_all_four_axes_is_editable`)
    matters just as much: a hypersensitive `sig()` that flags a harmless
    difference would silently make every ordinary multi-detector strategy
    read-only, which is the opposite failure and equally bad.
    """

    def _base_branch(self, branch_id, detector_id, policy_id, *,
                     policy_key=POLICY_KEY, policy_params=None,
                     regimes=("any",), confirmations=(), exits=None, enabled=True):
        return fgraph.Branch(
            id=branch_id,
            detector=fgraph.NodeSpec(id=detector_id, key=DETECTOR_KEY, params={}),
            policy=fgraph.NodeSpec(id=policy_id, key=policy_key, params=dict(policy_params or {})),
            regimes=regimes,
            confirmations=confirmations,
            exits=exits if exits is not None else fgraph.ExitPolicySpec(),
            enabled=enabled,
        )

    def _graph(self, name, branch_a, branch_b):
        return fgraph.StrategyGraph(
            name=name,
            data=fgraph.NodeSpec(id="data", key="data.ohlcv", params={}),
            branches=(branch_a, branch_b),
        )

    def test_uniform_graph_on_all_four_axes_is_editable(self):
        """Positive control: two branches identical in policy, confirmations,
        regimes and exits must round-trip editable=True."""
        branch_a = self._base_branch("main", "detector-a", "policy-a")
        branch_b = self._base_branch("second", "detector-b", "policy-b")
        fgraph.save(self._graph("uniform-control", branch_a, branch_b))

        got = api.get_strategy("uniform-control")
        assert got.status == 200
        assert got.payload["editable"] is True

    def test_policy_key_divergence_alone_is_not_editable(self):
        branch_a = self._base_branch("main", "detector-a", "policy-a", policy_key=POLICY_KEY)
        branch_b = self._base_branch(
            "second", "detector-b", "policy-b", policy_key="policy.fade-structural-stop"
        )
        fgraph.save(self._graph("divergent-policy-key", branch_a, branch_b))

        got = api.get_strategy("divergent-policy-key")
        assert got.status == 200
        assert got.payload["editable"] is False

    def test_policy_params_divergence_alone_is_not_editable_and_409s(self):
        """The subtler, more likely real-world case: the SAME policy
        plug-in, but different tuned params on each branch (e.g. one
        branch's rr_floor moved by evolution). `sig()` compares
        (key, params), not key alone, so this must be caught exactly like a
        different policy KEY is."""
        branch_a = self._base_branch("main", "detector-a", "policy-a", policy_params={})
        branch_b = self._base_branch(
            "second", "detector-b", "policy-b", policy_params={"rr_floor": 2.0}
        )
        fgraph.save(self._graph("divergent-policy-params", branch_a, branch_b))

        got = api.get_strategy("divergent-policy-params")
        assert got.status == 200
        assert got.payload["editable"] is False

        before = fgraph.load(str(fgraph.path_for("divergent-policy-params"))).to_dict()
        resp = api.post_strategy("divergent-policy-params", {"stages": _stages()})
        assert resp.status == 409
        assert "refusing to overwrite" in resp.payload["error"]
        after = fgraph.load(str(fgraph.path_for("divergent-policy-params"))).to_dict()
        assert after == before

    def test_regimes_divergence_alone_is_not_editable(self):
        branch_a = self._base_branch("main", "detector-a", "policy-a", regimes=("trending",))
        branch_b = self._base_branch("second", "detector-b", "policy-b", regimes=("ranging",))
        fgraph.save(self._graph("divergent-regimes", branch_a, branch_b))

        got = api.get_strategy("divergent-regimes")
        assert got.status == 200
        assert got.payload["editable"] is False

    def test_exits_divergence_alone_is_not_editable_and_409s(self):
        default_exits = fgraph.ExitPolicySpec()
        branch_a = self._base_branch("main", "detector-a", "policy-a", exits=default_exits)
        branch_b = self._base_branch(
            "second", "detector-b", "policy-b",
            exits=dataclasses.replace(default_exits, max_hold_bars=50),
        )
        fgraph.save(self._graph("divergent-exits", branch_a, branch_b))

        got = api.get_strategy("divergent-exits")
        assert got.status == 200
        assert got.payload["editable"] is False

        before = fgraph.load(str(fgraph.path_for("divergent-exits"))).to_dict()
        resp = api.post_strategy("divergent-exits", {"stages": _stages()})
        assert resp.status == 409
        assert "refusing to overwrite" in resp.payload["error"]
        after = fgraph.load(str(fgraph.path_for("divergent-exits"))).to_dict()
        assert after == before

    def test_disabled_second_branch_is_not_editable(self):
        """The disabled-branch guard is separate from `sig()`: a graph could
        be uniform on all four axes and still be unsafe to flatten if one
        branch is disabled -- flattening would silently drop the disabled
        branch rather than preserve it (off)."""
        branch_a = self._base_branch("main", "detector-a", "policy-a")
        branch_b = self._base_branch("second", "detector-b", "policy-b", enabled=False)
        fgraph.save(self._graph("disabled-branch", branch_a, branch_b))

        got = api.get_strategy("disabled-branch")
        assert got.status == 200
        assert got.payload["editable"] is False


class TestEvolutionMembers:
    """`/api/evolution/members` -- the sidebar's per-generation member list.
    `graph_json` is the largest column by far and must never reach the
    sidebar; `detectors`/`n_branches` are parsed out of it server-side
    instead."""

    CAMPAIGN_ID = "20260101-membersx"

    def _seed_campaign(self, conn, campaign_id):
        from trading_bot.evolution import population

        population.ensure_schema(conn)
        population.insert_campaign(conn, population.Campaign(
            campaign_id=campaign_id, seed=1, seed_graph_hash="h",
            seed_graph_json="{}", config_json="{}", symbols=("BTCUSDT",),
            train_start_ms=1, train_end_ms=2, audit_start_ms=3, audit_end_ms=4,
            population=4, generations=1, started_ts=100, status="done",
            label="members-camp", strategy_name="members-strat",
        ))

    def _seed_two_members(self, conn, campaign_id):
        from trading_bot.evolution import population

        two_branch = build_v020_graph(name="members-two-branch")
        one_branch = build_v020_graph(name="members-one-branch", include_fade=False)
        population.insert_members(conn, [
            population.Member(
                member_id="mem-2b", campaign_id=campaign_id, gen_index=0,
                member_index=0, graph_hash="h2b",
                graph_json=json.dumps(two_branch.to_dict()), rng_seed=1, role="offspring",
            ),
            population.Member(
                member_id="mem-1b", campaign_id=campaign_id, gen_index=0,
                member_index=1, graph_hash="h1b",
                graph_json=json.dumps(one_branch.to_dict()), rng_seed=2, role="offspring",
            ),
        ])

    def test_members_endpoint_strips_graph_json_and_reports_detectors(self):
        from trading_bot.data.statestore import connect as _connect

        conn = _connect(config.STATE_DB_PATH)
        try:
            self._seed_campaign(conn, self.CAMPAIGN_ID)
            self._seed_two_members(conn, self.CAMPAIGN_ID)
        finally:
            conn.close()

        resp = api.get_evolution_members(self.CAMPAIGN_ID, 0)
        assert resp.status == 200
        assert resp.payload["campaign"] == self.CAMPAIGN_ID
        assert resp.payload["gen"] == 0

        members = {m["member_id"]: m for m in resp.payload["members"]}
        assert "graph_json" not in members["mem-2b"]
        assert "graph_json" not in members["mem-1b"]
        assert members["mem-2b"]["n_branches"] == 2
        assert set(members["mem-2b"]["detectors"]) == {
            "detector.donchian-breakout", "detector.bollinger-fade",
        }
        assert members["mem-1b"]["n_branches"] == 1
        assert members["mem-1b"]["detectors"] == ["detector.donchian-breakout"]

    def test_members_missing_campaign_400(self):
        resp = api.get_evolution_members(None, 0)
        assert resp.status == 400

    def test_members_non_integer_gen_400(self):
        resp = api.get_evolution_members("some-campaign", "not-an-int")
        assert resp.status == 400


class TestEvolutionReplay:
    """`/api/evolution/replay` -- a deterministic RE-DERIVATION of one
    member's already-scored evaluation window, never a second evaluation:
    it must never write to state.db or the trial ledger (an explicit
    acceptance criterion), so every happy-path test below also proves that.
    """

    CAMPAIGN_ID = "20260101-replayxx"

    def _seed_campaign(self, conn, campaign_id, symbols=(SYMBOL,)):
        from trading_bot.evolution import population

        population.ensure_schema(conn)
        population.insert_campaign(conn, population.Campaign(
            campaign_id=campaign_id, seed=1, seed_graph_hash="h",
            seed_graph_json="{}", config_json="{}", symbols=symbols,
            train_start_ms=1, train_end_ms=2, audit_start_ms=3, audit_end_ms=4,
            population=4, generations=1, started_ts=100, status="done",
            label="replay-camp", strategy_name="replay-strat",
        ))

    def _member_graph_json(self):
        graph = api._stages_to_graph("replay-member", _stages())
        return json.dumps(graph.to_dict())

    def _seed_member(self, conn, member_id, campaign_id, *, scored=True, error=""):
        from trading_bot.evolution import population

        population.insert_members(conn, [population.Member(
            member_id=member_id, campaign_id=campaign_id, gen_index=0,
            member_index=0, graph_hash=f"h-{member_id}",
            graph_json=self._member_graph_json(), rng_seed=1, role="offspring",
        )])
        if scored:
            population.update_member_result(conn, member_id, {
                "window_start_ms": _REPLAY_WINDOW_START_MS,
                "window_end_ms": _REPLAY_WINDOW_END_MS,
                "error": error,
            })

    def _state_conn(self):
        from trading_bot.data.statestore import connect as _connect

        return _connect(config.STATE_DB_PATH)

    def test_replay_happy_path_returns_bars_and_trades_with_stop_and_target(self, ohlcv_conn):
        conn = self._state_conn()
        try:
            self._seed_campaign(conn, self.CAMPAIGN_ID)
            self._seed_member(conn, "m-ok", self.CAMPAIGN_ID)
        finally:
            conn.close()

        resp = api.get_evolution_replay("m-ok")
        assert resp.status == 200
        assert resp.payload["symbol"] == SYMBOL
        assert resp.payload["timeframe"] == "1d"

        bars = resp.payload["bars"]
        assert bars["ts"], "the seeded 1d bars must come back"
        assert len({len(bars[k]) for k in ("ts", "o", "h", "l", "c", "v")}) == 1

        trades = resp.payload["trades"]
        assert len(trades) == 1
        assert trades[0]["stop"] == 156.5
        assert trades[0]["target"] == 182.0

    def test_replay_route_is_registered_and_dispatches_through_handle(self, ohlcv_conn):
        """At least one assertion must go through api.handle(...) rather than
        calling get_evolution_replay directly, so the ROUTES entry itself --
        not just the function behind it -- is under test."""
        conn = self._state_conn()
        try:
            self._seed_campaign(conn, self.CAMPAIGN_ID)
            self._seed_member(conn, "m-route", self.CAMPAIGN_ID)
        finally:
            conn.close()

        resp = api.handle("GET", "/api/evolution/replay", query={"member": "m-route"})
        assert resp.status == 200
        assert resp.payload["trades"][0]["stop"] == 156.5
        assert resp.payload["trades"][0]["target"] == 182.0

    def test_replay_never_writes_to_state_db(self, ohlcv_conn):
        conn = self._state_conn()
        try:
            self._seed_campaign(conn, self.CAMPAIGN_ID)
            self._seed_member(conn, "m-ro", self.CAMPAIGN_ID)
        finally:
            conn.close()

        db_path = Path(config.STATE_DB_PATH)
        before = db_path.read_bytes()

        resp = api.get_evolution_replay("m-ro")
        assert resp.status == 200

        after = db_path.read_bytes()
        assert after == before, "replay must never write to state.db or the trial ledger"

    def test_replay_missing_member_query_400(self):
        resp = api.get_evolution_replay(None)
        assert resp.status == 400

    def test_replay_unknown_member_404(self):
        from trading_bot.evolution import population

        conn = self._state_conn()
        try:
            population.ensure_schema(conn)  # table must exist before the 404 lookup
        finally:
            conn.close()

        resp = api.get_evolution_replay("no-such-member")
        assert resp.status == 404

    def test_replay_unscored_member_409(self):
        conn = self._state_conn()
        try:
            self._seed_campaign(conn, self.CAMPAIGN_ID)
            self._seed_member(conn, "m-unscored", self.CAMPAIGN_ID, scored=False)
        finally:
            conn.close()

        resp = api.get_evolution_replay("m-unscored")
        assert resp.status == 409
        assert "never scored" in resp.payload["error"]

    def test_replay_stored_error_is_409_and_echoed(self, ohlcv_conn):
        conn = self._state_conn()
        try:
            self._seed_campaign(conn, self.CAMPAIGN_ID)
            self._seed_member(conn, "m-err", self.CAMPAIGN_ID, error="oracle blew up")
        finally:
            conn.close()

        resp = api.get_evolution_replay("m-err")
        assert resp.status == 409
        assert "oracle blew up" in resp.payload["error"]

    def test_replay_missing_campaign_404(self, ohlcv_conn):
        """A member whose campaign row is absent (should not normally happen,
        but the endpoint must not 500 over it)."""
        from trading_bot.evolution import population

        conn = self._state_conn()
        try:
            population.ensure_schema(conn)  # no campaign row is ever inserted here
            self._seed_member(conn, "m-ghost", "20260101-ghostcamp")
        finally:
            conn.close()

        resp = api.get_evolution_replay("m-ghost")
        assert resp.status == 404

    def test_replay_bad_symbol_400(self, ohlcv_conn):
        conn = self._state_conn()
        try:
            self._seed_campaign(conn, self.CAMPAIGN_ID)
            self._seed_member(conn, "m-sym", self.CAMPAIGN_ID)
        finally:
            conn.close()

        resp = api.get_evolution_replay("m-sym", symbol="NOPEUSDT")
        assert resp.status == 400
        assert "symbol" in resp.payload["error"]

    def test_replay_bad_timeframe_400(self, ohlcv_conn):
        conn = self._state_conn()
        try:
            self._seed_campaign(conn, self.CAMPAIGN_ID)
            self._seed_member(conn, "m-tf", self.CAMPAIGN_ID)
        finally:
            conn.close()

        resp = api.get_evolution_replay("m-tf", timeframe="7m")
        assert resp.status == 400
        assert "timeframe" in resp.payload["error"]


class TestSerializers:
    def _fake_result(self):
        from trading_bot.backtest.benchmark import BenchmarkResult
        from trading_bot.backtest.engine import BacktestParams
        from trading_bot.backtest.walkforward import GATE_CONDITIONS, WalkForwardResult

        gate = {name: (name != "sample_adequacy") for name in GATE_CONDITIONS}
        bundle = {
            "total_return": 1.1, "ann_return_pct": 0.10, "sharpe": 0.5,
            "sortino": 0.6, "max_drawdown_pct": 0.10, "n_days": 90,
        }
        bench = BenchmarkResult(
            per_symbol={"BTCUSDT": bundle}, basket=bundle, start_ms=0, end_ms=1
        )
        return WalkForwardResult(
            folds=[],
            final_params=BacktestParams(),
            final_max_hold_bars=None,
            oos_start=0,
            oos_end=1,
            oos_metrics={
                "n_trades": 5, "win_rate": 0.5, "expectancy_pct": 0.01,
                "avg_win_pct": 0.02, "avg_loss_pct": -0.01, "profit_factor": 1.2,
                "max_drawdown_pct": 0.1, "by_bucket": {},
            },
            oos_equity={
                "sharpe": float("nan"), "sortino": 1.0, "dsr": 0.5,
                "max_drawdown_pct": 0.1, "ann_return_pct": 0.2, "n_days": 90,
            },
            per_symbol_expectancy={"BTCUSDT": 0.01},
            gate=gate,
            benchmark=bench,
            n_trials_used=3,
            passed=False,
        )

    def test_seven_rows_and_passed_equals_all(self):
        from trading_bot.backtest.walkforward import GATE_CONDITIONS

        result = self._fake_result()
        payload = api._serialize_wf_result(result)
        assert len(payload["gate"]["conditions"]) == len(GATE_CONDITIONS)
        assert payload["gate"]["passed"] == all(result.gate.values())

    def test_benchmark_row_shows_both_sides(self):
        result = self._fake_result()
        payload = api._serialize_wf_result(result)
        row = next(
            c for c in payload["gate"]["conditions"] if c["name"] == "beats_benchmark_return"
        )
        assert "vs" in row["measured"]
        row2 = next(
            c for c in payload["gate"]["conditions"] if c["name"] == "beats_benchmark_sharpe"
        )
        assert "vs" in row2["measured"]

    def test_nan_serializes_to_null(self):
        result = self._fake_result()
        payload = api._serialize_wf_result(result)
        assert payload["oos_equity"]["sharpe"] is None

    def test_trials_before_after_and_campaign_carried(self):
        result = self._fake_result()
        payload = api._serialize_wf_result(
            result, trials_before=1, trials_after=2, campaign="ui-smoke"
        )
        assert payload["trials_before"] == 1
        assert payload["trials_after"] == 2
        assert payload["campaign"] == "ui-smoke"


class TestRuns:
    def _seed_strategy(self, name="s1"):
        resp = api.post_strategy(name, {"stages": _stages()})
        assert resp.status == 200
        return name

    def test_unknown_kind_400(self):
        resp = api.post_run({"kind": "nope", "strategy": "x"})
        assert resp.status == 400

    def test_unstored_symbol_400(self):
        name = self._seed_strategy()
        resp = api.post_run({"kind": "backtest", "strategy": name, "symbols": ["NOPEUSDT"]})
        assert resp.status == 400

    def test_bad_date_400(self):
        name = self._seed_strategy()
        resp = api.post_run(
            {
                "kind": "backtest", "strategy": name,
                "symbols": [config.SYMBOLS[0]], "start": "not-a-date",
            }
        )
        assert resp.status == 400

    def test_missing_campaign_for_gate_400(self):
        name = self._seed_strategy()
        resp = api.post_run(
            {"kind": "gate", "strategy": name, "symbols": [config.SYMBOLS[0]]}
        )
        assert resp.status == 400

    def test_missing_strategy_400(self):
        resp = api.post_run(
            {"kind": "backtest", "strategy": "no-such-strategy", "symbols": [config.SYMBOLS[0]]}
        )
        assert resp.status == 400

    def test_tier_a_concurrency_429(self, monkeypatch):
        name = self._seed_strategy()
        gate = threading.Event()

        def slow_job(cmd):
            gate.wait(5)
            return {"kind": "backtest"}

        monkeypatch.setitem(api._TIER_A_JOBS, "backtest", slow_job)
        r1 = api.post_run({"kind": "backtest", "strategy": name, "symbols": [config.SYMBOLS[0]]})
        assert r1.status == 202
        r2 = api.post_run({"kind": "backtest", "strategy": name, "symbols": [config.SYMBOLS[0]]})
        assert r2.status == 429
        gate.set()
        for _ in range(50):
            if api._STATE["active_tier_a"] == 0:
                break
            time.sleep(0.05)

    def test_trial_cost_echoed_before_and_after_for_gate(self, monkeypatch):
        name = self._seed_strategy()
        monkeypatch.setitem(api._TIER_A_JOBS, "gate", lambda cmd: {"kind": "gate"})
        resp = api.post_run(
            {
                "kind": "gate", "strategy": name, "symbols": [config.SYMBOLS[0]],
                "campaign": "ui-smoke-test",
            }
        )
        assert resp.status == 202
        assert resp.payload["trials_before"] == 0
        assert resp.payload["trials_estimated"] == 1
        for _ in range(50):
            if api._STATE["active_tier_a"] == 0:
                break
            time.sleep(0.05)

    def test_recover_runs_marks_tier_a_orphaned(self):
        run_id = api.new_run_id()
        run_dir = api._run_dir(run_id)
        run_dir.mkdir(parents=True)
        api._write_json_atomic(run_dir / "status.json", {"state": "running", "tier": "A"})
        api.recover_runs()
        status = api._read_json(run_dir / "status.json")
        assert status["state"] == "orphaned"

    def test_recover_runs_marks_dead_tier_b_failed(self):
        run_id = api.new_run_id()
        run_dir = api._run_dir(run_id)
        run_dir.mkdir(parents=True)
        api._write_json_atomic(
            run_dir / "status.json", {"state": "running", "tier": "B", "pid": 999_999_999}
        )
        api.recover_runs()
        status = api._read_json(run_dir / "status.json")
        assert status["state"] == "failed"

    def _seed_done_campaign(self, campaign_id):
        from trading_bot.data.statestore import connect as connect_state
        from trading_bot.evolution import population

        conn = connect_state(config.STATE_DB_PATH)
        try:
            population.ensure_schema(conn)
            population.insert_campaign(conn, population.Campaign(
                campaign_id=campaign_id, seed=1, seed_graph_hash="h",
                seed_graph_json="{}", config_json="{}", symbols=(SYMBOL,),
                train_start_ms=1, train_end_ms=2, audit_start_ms=3, audit_end_ms=4,
                population=4, generations=1, started_ts=100, status="done",
                finished_ts=12345, label="stale-camp", strategy_name="stale-strat",
            ))
        finally:
            conn.close()

    def test_recover_runs_finalizes_dead_tier_b_as_done_when_campaign_finished(self):
        """The bug this guards: a UI server restart orphans the evolve
        subprocess (Popen children outlive their dead parent) but takes the
        in-memory watcher that would finalize status.json with it. If the
        subprocess then finishes on its own, nothing is left alive to record
        that — the run must instead look at the campaign row it actually
        wrote, not assume a dead pid means it failed."""
        campaign_id = "20260101-donecamp"
        self._seed_done_campaign(campaign_id)
        run_id = api.new_run_id()
        run_dir = api._run_dir(run_id)
        run_dir.mkdir(parents=True)
        api._write_json_atomic(
            run_dir / "status.json",
            {
                "state": "running", "tier": "B", "pid": 999_999_999,
                "campaign_id": campaign_id, "exit_code": None, "finished_ms": None,
            },
        )
        api.recover_runs()
        status = api._read_json(run_dir / "status.json")
        assert status["state"] == "done"
        assert status["exit_code"] == 0
        assert status["finished_ms"] == 12345

    def test_get_run_self_heals_a_stale_tier_b_status_without_a_restart(self):
        """recover_runs only fires once at server boot. A subprocess orphaned
        by a restart that finishes *after* boot must still be caught the next
        time anyone reads its status — via get_run/get_runs/the SSE poll —
        not just at the next restart."""
        campaign_id = "20260101-donecamp2"
        self._seed_done_campaign(campaign_id)
        run_id = api.new_run_id()
        run_dir = api._run_dir(run_id)
        run_dir.mkdir(parents=True)
        api._write_json_atomic(
            run_dir / "status.json",
            {
                "state": "running", "tier": "B", "pid": 999_999_999,
                "campaign_id": campaign_id, "exit_code": None, "finished_ms": None,
            },
        )
        resp = api.get_run(run_id)
        assert resp.payload["status"]["state"] == "done"
        assert resp.payload["status"]["exit_code"] == 0
        # and it persisted the correction, not just the response payload
        assert api._read_json(run_dir / "status.json")["state"] == "done"

    def test_evolve_argv_is_a_list_of_strings(self):
        name = self._seed_strategy()
        cmd = {
            "strategy_path": str(fgraph.path_for(name)),
            "symbols": [config.SYMBOLS[0]],
            "campaign": "some-campaign", "strategy": name,
        }
        argv = api._evolve_argv(cmd, seed=1)
        assert isinstance(argv, list)
        assert all(isinstance(a, str) for a in argv)
        assert "--seed-graph" in argv
        # A new campaign MUST carry its label and strategy: without them the
        # campaign is unfindable tomorrow and unbound to any strategy.
        assert argv[argv.index("--label") + 1] == "some-campaign"
        assert argv[argv.index("--strategy-name") + 1] == name

    def test_evolve_argv_continues_instead_of_reseeding(self):
        """A continuation resumes the campaign; it must not pass a fresh seed
        graph, which would restart the search and discard paid-for generations."""
        name = self._seed_strategy()
        cmd = {
            "strategy_path": str(fgraph.path_for(name)),
            "symbols": [config.SYMBOLS[0]],
            "campaign": "some-campaign", "strategy": name,
        }
        argv = api._evolve_argv(cmd, seed=1, resume="20260101-abcdef", extend=3)
        assert argv[argv.index("--resume") + 1] == "20260101-abcdef"
        assert argv[argv.index("--extend") + 1] == "3"
        assert "--seed-graph" not in argv
        assert "--seed" not in argv


class TestCampaigns:
    """A campaign is one strategy's evolution history (one strategy, many
    campaigns), so the dropdown is strategy-scoped and a mismatched pair is
    refused rather than quietly mixing two searches into one trial ledger."""

    def _seed_campaign(self, label, strategy_name):
        from trading_bot.data.statestore import connect as _connect
        from trading_bot.evolution import population

        conn = _connect(config.STATE_DB_PATH)
        try:
            population.ensure_schema(conn)
            population.insert_campaign(conn, population.Campaign(
                campaign_id=f"20260101-{label[:6]:0<6}", seed=1, seed_graph_hash="h",
                seed_graph_json="{}", config_json="{}", symbols=("BTCUSDT",),
                train_start_ms=1, train_end_ms=2, audit_start_ms=3, audit_end_ms=4,
                population=4, generations=2, started_ts=100, status="done",
                label=label, strategy_name=strategy_name,
            ))
        finally:
            conn.close()

    def test_lists_campaigns_and_filters_by_strategy(self):
        self._seed_campaign("alpha", "strat-a")
        self._seed_campaign("beta", "strat-b")
        every = api.handle("GET", "/api/campaigns")
        assert every.status == 200
        assert {c["label"] for c in every.payload["campaigns"]} == {"alpha", "beta"}

        scoped = api.handle("GET", "/api/campaigns", query={"strategy": "strat-a"})
        assert [c["label"] for c in scoped.payload["campaigns"]] == ["alpha"]
        assert scoped.payload["campaigns"][0]["strategy_name"] == "strat-a"

    def test_running_a_campaign_against_the_wrong_strategy_is_refused(self):
        resp = api.post_strategy("other-strategy", {"stages": _stages()})
        assert resp.status == 200
        self._seed_campaign("alpha", "strat-a")
        resp = api.post_run({
            "kind": "gate", "strategy": "other-strategy",
            "symbols": [config.SYMBOLS[0]], "campaign": "alpha",
        })
        assert resp.status == 400
        assert "strat-a" in resp.payload["error"]

    def test_an_unknown_campaign_name_is_treated_as_new(self):
        name = api.post_strategy("s-new", {"stages": _stages()})
        assert name.status == 200
        resp = api.handle("GET", "/api/campaigns", query={"strategy": "s-new"})
        assert resp.payload["campaigns"] == []


class TestChampionEvaluation:
    """Evolution's whole point is that the campaign gets better, so a backtest
    or gate of an evolved campaign must score the CHAMPION. Reloading the seed
    file instead made a 16-generation campaign report metrics identical to
    generation zero — the search ran, was charged for, and was then discarded.
    """

    def _campaign_with_champion(self, label, strategy_name, champion_graph):
        from trading_bot.data.statestore import connect as _connect
        from trading_bot.evolution import population

        campaign_id = f"20260101-{label[:6]:0<6}"
        conn = _connect(config.STATE_DB_PATH)
        try:
            population.ensure_schema(conn)
            population.insert_campaign(conn, population.Campaign(
                campaign_id=campaign_id, seed=1, seed_graph_hash="seedhash",
                seed_graph_json="{}", config_json="{}", symbols=("BTCUSDT",),
                train_start_ms=1, train_end_ms=2, audit_start_ms=3, audit_end_ms=4,
                population=4, generations=2, started_ts=100, status="done",
                label=label, strategy_name=strategy_name,
            ))
            population.insert_members(conn, [population.Member(
                member_id="m-champ", campaign_id=campaign_id, gen_index=7,
                member_index=0, graph_hash="champhash",
                graph_json=json.dumps(champion_graph), rng_seed=5, role="offspring",
            )])
            population.update_member_result(conn, "m-champ", {"fitness": 9.5, "tier": "A"})
        finally:
            conn.close()
        return campaign_id

    def test_tier_a_run_of_an_evolved_campaign_scores_the_champion(self):
        """The defect in one assertion: the graph the job receives must be the
        evolved one, not the file the campaign was seeded from."""
        assert api.post_strategy("champ-strat", {"stages": _stages()}).status == 200
        champion = fgraph.load(fgraph.path_for("champ-strat")).to_dict()
        campaign_id = self._campaign_with_champion("evolved", "champ-strat", champion)

        # The real backtest job is replaced: this asserts on WHICH graph is
        # dispatched, and running a full backtest would say nothing more.
        with mock.patch.dict(api._TIER_A_JOBS,
                             {"backtest": lambda cmd: {"kind": "backtest"}}):
            resp = api.post_run({
                "kind": "backtest", "strategy": "champ-strat",
                "symbols": [config.SYMBOLS[0]], "campaign": "evolved",
            })
            assert resp.status == 202
            run_id = resp.payload["run_id"]
            api.reset_jobs()

        cmd = json.loads((api._run_dir(run_id) / "cmd.json").read_text(encoding="utf-8"))
        assert cmd["graph_json"] == json.dumps(champion)
        assert cmd["graph_source"] == {
            "kind": "champion", "campaign": "evolved", "campaign_id": campaign_id,
            "member_id": "m-champ", "gen_index": 7, "graph_hash": "champhash",
            "fitness": 9.5,
        }
        graph, source = api._graph_for_run(cmd)
        assert graph.to_dict() == champion
        assert source["kind"] == "champion"

    def test_a_campaign_with_no_scored_member_still_uses_the_seed_file(self):
        """A brand-new campaign has evolved nothing yet; falling back to the
        seed file is correct there, and must stay labelled as the seed."""
        assert api.post_strategy("fresh-strat", {"stages": _stages()}).status == 200
        path = str(fgraph.path_for("fresh-strat"))
        graph, source = api._graph_for_run({"strategy_path": path, "strategy": "fresh-strat"})
        assert source == {"kind": "seed", "strategy": "fresh-strat"}
        assert graph.to_dict() == fgraph.load(path).to_dict()


class TestDispatch:
    def test_404_unknown_path(self):
        resp = api.handle("GET", "/api/nope")
        assert resp.status == 404

    def test_405_with_allow_header(self):
        resp = api.handle("DELETE", "/api/config")
        assert resp.status == 405
        assert "GET" in resp.headers["Allow"]

    def test_403_foreign_origin(self):
        resp = api.handle(
            "POST", "/api/validate", body={},
            headers={"Content-Type": "application/json", "Origin": "https://evil.example"},
        )
        assert resp.status == 403

    def test_415_wrong_content_type(self):
        resp = api.handle("POST", "/api/validate", body={}, headers={"Content-Type": "text/plain"})
        assert resp.status == 415

    def test_matching_origin_allowed(self):
        api.set_origin_port(8770)
        resp = api.handle(
            "POST", "/api/validate", body={"stages": []},
            headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:8770"},
        )
        assert resp.status == 200

    def test_stop_requires_json_content_type_matching_frontend_contract(self):
        """The bug this pins: app.js's `api()` helper only sets Content-Type
        when a body is passed, and stopRun() used to call it with none — so
        every browser POST to /stop arrived without the header and got 415'd
        here, every single time the operator tried to stop an evolve run.
        Fixed by having stopRun() pass an explicit {} body; this test is the
        server side of that contract."""
        run_id = api.new_run_id()
        run_dir = api._run_dir(run_id)
        run_dir.mkdir(parents=True)
        api._write_json_atomic(
            run_dir / "status.json", {"state": "running", "tier": "B", "pid": None}
        )

        resp = api.handle("POST", f"/api/runs/{run_id}/stop")
        assert resp.status == 415

        resp = api.handle(
            "POST", f"/api/runs/{run_id}/stop", body={},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200


class TestSseFraming:
    def test_multiline_log_one_data_line_each(self):
        frames = api.sse_frames(0, "line1\nline2\n", {"state": "running"})
        joined = "".join(frames)
        assert "data: line1" in joined
        assert "data: line2" in joined

    def test_id_is_byte_offset(self):
        frames = api.sse_frames(10, "abcd\n", {"state": "running"})
        assert any("id: 15" in f for f in frames)

    def test_done_carries_exit_code(self):
        frames = api.sse_frames(0, "", {"state": "done", "exit_code": 0, "has_result": True})
        assert any("event: done" in f and '"exit_code": 0' in f for f in frames)

    def test_no_activity_is_a_heartbeat(self):
        frames = api.sse_frames(0, "", {"state": "running"})
        assert frames == [": heartbeat\n"]


class TestCli:
    def test_ui_command_refuses_non_loopback_host(self):
        from trading_bot.cli import _ui_command

        assert _ui_command(host="10.0.0.5") == 2
