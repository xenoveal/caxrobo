"""Tests for framework/graph.py (v0.3.0 Phase 3).

Follows tests/test_tiers.py's "configuration invariant" genre: every assertion
here is about the graph model and the hash, never about strategy outcome.

TestCanonicalHash is the class Phase 5's strategy_versions and Phase 6's
trial_ledger depend on: two graphs that hash equal must BEHAVE identically, and
two graphs that behave identically must hash equal (A5).
"""

import dataclasses
import json

import pytest

from trading_bot import config
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.errors import GraphError
from trading_bot.framework.graph import (
    SCHEMA_VERSION,
    Branch,
    ExitPolicySpec,
    NodeSpec,
    RegimeGate,
    StrategyGraph,
    TriggerSpec,
    canonical_dict,
    graph_hash,
    short_hash,
    validate,
)
from trading_bot.plugins import build_v020_graph

# The pinned digest of build_v020_graph() at config's current defaults.
#
# REGENERATE DELIBERATELY, NEVER CASUALLY: this digest changes if ANY plug-in's
# ParamSpec default changes, and Phase 5's strategy_versions plus Phase 6's
# trial_ledger key on it. A change here means every persisted row referring to
# this graph now points at a different strategy. Regenerate with:
#   .venv/bin/python -c "from trading_bot.framework import registry, graph; \
#     from trading_bot.plugins import build_v020_graph; registry.load_all(); \
#     print(graph.graph_hash(build_v020_graph()))"
V020_DIGEST = "b7cb327cba10d88fdfefcfa0a0c6752b258b180f4d040caf79e17af2ff7a4c8b"


@pytest.fixture(autouse=True)
def _plugins_loaded():
    registry.load_all()


def maximal_graph():
    """Three branches, confirmations, filters, non-default exits, populated meta.

    Confirmations and filters are stubbed with throwaway registrations by the
    caller when needed; here they are empty, because Phase 3 ships no production
    Confirmation or Filter (A8) and a round-trip test must not invent one that
    validate() would then reject.
    """
    return StrategyGraph(
        name="maximal",
        data=NodeSpec(id="src", key="data.ohlcv", params={"timeframes": "all"}),
        branches=(
            Branch(
                id="alpha",
                detector=NodeSpec(id="a-det", key="detector.donchian-breakout",
                                  params={"entry_period": 30}),
                policy=NodeSpec(id="a-pol", key="policy.atr-stop-measured-move"),
                regimes=("trending",),
                exits=ExitPolicySpec(channel_exit=True, channel_period=30,
                                     target_enabled=True, max_hold_bars=48),
            ),
            Branch(
                id="beta",
                detector=NodeSpec(id="b-det", key="detector.bollinger-fade"),
                policy=NodeSpec(id="b-pol", key="policy.fade-structural-stop"),
                regimes=("ranging", "extreme-volatility"),
            ),
            Branch(
                id="gamma",
                detector=NodeSpec(id="g-det", key="detector.legacy-patterns"),
                policy=NodeSpec(id="g-pol", key="policy.atr-stop-measured-move"),
                regimes=("any",),
                enabled=False,
            ),
        ),
        regime=RegimeGate(adx_trend_threshold=30.0),
        trigger=TriggerSpec(volume_lookback=10),
        filters=(),
        meta={"note": "everything at once"},
    )


class TestRoundTrip:
    def test_v020_graph_round_trips(self):
        g = build_v020_graph()
        assert StrategyGraph.from_dict(g.to_dict()) == g

    def test_maximal_graph_round_trips_through_json(self):
        g = maximal_graph()
        again = StrategyGraph.from_dict(json.loads(json.dumps(g.to_dict())))
        assert again == g

    def test_unknown_top_level_key_raises(self):
        payload = build_v020_graph().to_dict()
        payload["exits"] = {}
        with pytest.raises(GraphError, match="unknown key"):
            StrategyGraph.from_dict(payload)

    def test_unknown_exits_key_trail_vs_trail_enabled_raises(self):
        """The named trap (Task 6 GOTCHA b): a UI or mutator writing "trail"
        would otherwise silently get trail_enabled=False and the operator would
        believe a trail was tested."""
        payload = build_v020_graph().to_dict()
        payload["branches"][0]["exits"]["trail"] = True
        with pytest.raises(GraphError, match=r"unknown key\(s\) \['trail'\]"):
            StrategyGraph.from_dict(payload)

    def test_unknown_regime_key_raises(self):
        payload = build_v020_graph().to_dict()
        payload["regime"]["adx_period"] = 21
        with pytest.raises(GraphError, match="graph.regime"):
            StrategyGraph.from_dict(payload)

    def test_unknown_trigger_key_raises(self):
        payload = build_v020_graph().to_dict()
        payload["trigger"]["lookback"] = 2
        with pytest.raises(GraphError, match="graph.trigger"):
            StrategyGraph.from_dict(payload)

    def test_unknown_node_key_raises(self):
        payload = build_v020_graph().to_dict()
        payload["data"]["kind"] = "data"
        with pytest.raises(GraphError, match="graph.data"):
            StrategyGraph.from_dict(payload)

    def test_missing_required_key_raises(self):
        payload = build_v020_graph().to_dict()
        del payload["branches"]
        with pytest.raises(GraphError, match="missing required key"):
            StrategyGraph.from_dict(payload)

    def test_list_where_a_tuple_is_required_raises(self):
        g = build_v020_graph()
        with pytest.raises(GraphError, match="must be a TUPLE"):
            dataclasses.replace(g, branches=list(g.branches))
        with pytest.raises(GraphError, match="must be a TUPLE"):
            dataclasses.replace(g.branches[0], regimes=["trending"])


class TestCanonicalHash:
    def test_hash_is_stable_across_calls(self):
        g = build_v020_graph()
        assert graph_hash(g) == graph_hash(g)

    def test_pinned_digest(self):
        """Proves the hash is stable ACROSS PROCESSES, which is the only thing
        that makes a persisted ledger meaningful."""
        assert graph_hash(build_v020_graph()) == V020_DIGEST
        assert short_hash(build_v020_graph()) == V020_DIGEST[:12]

    def test_branch_order_does_not_change_hash(self):
        g = build_v020_graph()
        assert graph_hash(dataclasses.replace(g, branches=tuple(reversed(g.branches)))) == graph_hash(g)

    def test_confirmation_and_filter_order_do_not_change_hash(self):
        g = maximal_graph()
        assert graph_hash(dataclasses.replace(g, filters=tuple(reversed(g.filters)))) == graph_hash(g)

    def test_regime_list_order_does_not_change_hash(self):
        g = maximal_graph()
        b = g.branches[1]
        flipped = dataclasses.replace(b, regimes=tuple(reversed(b.regimes)))
        other = dataclasses.replace(g, branches=(g.branches[0], flipped, g.branches[2]))
        assert graph_hash(other) == graph_hash(g)

    def test_name_does_not_change_hash(self):
        g = build_v020_graph()
        assert graph_hash(dataclasses.replace(g, name="renamed")) == graph_hash(g)

    def test_meta_does_not_change_hash(self):
        g = build_v020_graph()
        assert graph_hash(dataclasses.replace(g, meta={"note": "x"})) == graph_hash(g)

    def test_explicit_default_equals_omitted(self):
        g = build_v020_graph()
        det = g.branches[0].detector
        explicit = dataclasses.replace(
            det, params={**det.params, "entry_period": config.DONCHIAN_ENTRY_PERIOD}
        )
        b = dataclasses.replace(g.branches[0], detector=explicit)
        assert graph_hash(dataclasses.replace(g, branches=(b, g.branches[1]))) == graph_hash(g)

    def test_int_and_float_forms_agree(self):
        g = build_v020_graph()
        pol = g.branches[0].policy
        as_int = dataclasses.replace(pol, params={"rr_floor": config.RR_FLOOR, "atr_multiple": 2})
        as_float = dataclasses.replace(pol, params={"rr_floor": config.RR_FLOOR, "atr_multiple": 2.0})
        h1 = graph_hash(dataclasses.replace(
            g, branches=(dataclasses.replace(g.branches[0], policy=as_int), g.branches[1])))
        h2 = graph_hash(dataclasses.replace(
            g, branches=(dataclasses.replace(g.branches[0], policy=as_float), g.branches[1])))
        assert h1 == h2

    def test_param_change_changes_hash(self):
        g = build_v020_graph()
        det = dataclasses.replace(g.branches[0].detector, params={"trend_period": 60})
        b = dataclasses.replace(g.branches[0], detector=det)
        assert graph_hash(dataclasses.replace(g, branches=(b, g.branches[1]))) != graph_hash(g)

    def test_exit_flag_change_changes_hash(self):
        a = build_v020_graph(trail_enabled=False)
        b = build_v020_graph(trail_enabled=True)
        assert graph_hash(a) != graph_hash(b)

    def test_regime_threshold_change_changes_hash(self):
        a = build_v020_graph()
        b = build_v020_graph(adx_trend_threshold=30.0)
        assert graph_hash(a) != graph_hash(b)

    def test_schema_version_change_changes_hash(self):
        g = build_v020_graph()
        assert graph_hash(dataclasses.replace(g, schema_version=99)) != graph_hash(g)

    def test_resolved_hashes_equal(self):
        g = build_v020_graph()
        assert graph_hash(g) == graph_hash(g.resolved())

    def test_ordered_branches_matches_canonical_order(self):
        """THE A5 INVARIANT that ties hash equality to behavioural equality:
        canonical order IS runtime order, so two hash-equal graphs cannot break a
        rank_signals tie differently."""
        g = maximal_graph()
        runtime = [b.id for b in g.ordered_branches()]
        canonical = [b["id"] for b in canonical_dict(g)["branches"]]
        assert runtime == canonical
        reversed_g = dataclasses.replace(g, branches=tuple(reversed(g.branches)))
        assert [b.id for b in reversed_g.ordered_branches()] == runtime

    def test_nan_param_is_unhashable(self):
        g = build_v020_graph()
        pol = dataclasses.replace(g.branches[0].policy, params={"rr_floor": float("nan")})
        b = dataclasses.replace(g.branches[0], policy=pol)
        bad = dataclasses.replace(g, branches=(b, g.branches[1]))
        with pytest.raises(ValueError):
            graph_hash(bad)

    def test_canonical_dict_excludes_name_and_meta(self):
        d = canonical_dict(build_v020_graph())
        assert "name" not in d and "meta" not in d
        assert d["schema_version"] == SCHEMA_VERSION


class TestValidation:
    def _bad(self, g):
        with pytest.raises(GraphError) as exc:
            validate(g)
        return str(exc.value)

    def test_accepts_the_v020_graph(self):
        validate(build_v020_graph())

    def test_rejects_wrong_schema_version(self):
        msg = self._bad(dataclasses.replace(build_v020_graph(), schema_version=99))
        assert "99" in msg and "regenerate or migrate" in msg

    def test_rejects_bad_name(self):
        msg = self._bad(dataclasses.replace(build_v020_graph(), name="Bad Name!"))
        assert "Bad Name!" in msg

    def test_rejects_slot_kind_mismatch(self):
        g = build_v020_graph()
        b = dataclasses.replace(
            g.branches[0],
            detector=NodeSpec(id="oops", key="policy.atr-stop-measured-move"),
        )
        msg = self._bad(dataclasses.replace(g, branches=(b, g.branches[1])))
        assert "must name a 'detector' plug-in" in msg

    def test_rejects_unknown_key(self):
        g = build_v020_graph()
        b = dataclasses.replace(
            g.branches[0], detector=NodeSpec(id="x", key="detector.no-such")
        )
        msg = self._bad(dataclasses.replace(g, branches=(b, g.branches[1])))
        assert "unknown plug-in" in msg

    def test_unknown_key_suggests_close_matches(self):
        g = build_v020_graph()
        b = dataclasses.replace(
            g.branches[0], detector=NodeSpec(id="x", key="detector.donchian-breakou")
        )
        msg = self._bad(dataclasses.replace(g, branches=(b, g.branches[1])))
        assert "donchian-breakout" in msg

    def test_rejects_duplicate_node_id(self):
        g = build_v020_graph()
        b = dataclasses.replace(
            g.branches[0],
            detector=dataclasses.replace(g.branches[0].detector, id="ohlcv"),
        )
        msg = self._bad(dataclasses.replace(g, branches=(b, g.branches[1])))
        assert "duplicate node id 'ohlcv'" in msg and "graph.data" in msg

    def test_rejects_duplicate_branch_id(self):
        g = build_v020_graph()
        dup = dataclasses.replace(g.branches[1], id=g.branches[0].id)
        msg = self._bad(dataclasses.replace(g, branches=(g.branches[0], dup)))
        assert "duplicate branch id" in msg

    def test_rejects_no_branches(self):
        msg = self._bad(dataclasses.replace(build_v020_graph(), branches=()))
        assert "no branches" in msg

    def test_rejects_no_enabled_branch(self):
        g = build_v020_graph()
        off = tuple(dataclasses.replace(b, enabled=False) for b in g.branches)
        msg = self._bad(dataclasses.replace(g, branches=off))
        assert "no ENABLED branch" in msg

    def test_rejects_illegal_regime_label(self):
        g = build_v020_graph()
        b = dataclasses.replace(g.branches[0], regimes=("bull-market",))
        msg = self._bad(dataclasses.replace(g, branches=(b, g.branches[1])))
        assert "bull-market" in msg

    def test_rejects_illegal_param_value(self):
        g = build_v020_graph()
        det = dataclasses.replace(g.branches[0].detector, params={"entry_period": 9999})
        b = dataclasses.replace(g.branches[0], detector=det)
        msg = self._bad(dataclasses.replace(g, branches=(b, g.branches[1])))
        assert "9999" in msg

    def test_rejects_stopless_strategy(self):
        g = build_v020_graph()
        b = dataclasses.replace(
            g.branches[0], exits=dataclasses.replace(g.branches[0].exits, stop=False)
        )
        msg = self._bad(dataclasses.replace(g, branches=(b, g.branches[1])))
        assert "stopless strategy is not expressible" in msg

    def test_rejects_nonpositive_trail_multiple(self):
        g = build_v020_graph(trail_enabled=True)
        trend = next(b for b in g.branches if b.id == "trend")
        bad = dataclasses.replace(
            trend, exits=dataclasses.replace(trend.exits, trail_atr_multiple=0.0)
        )
        others = tuple(b for b in g.branches if b.id != "trend")
        msg = self._bad(dataclasses.replace(g, branches=(bad,) + others))
        assert "trail_atr_multiple" in msg

    def test_rejects_channel_period_mismatch(self):
        """Prevents a SILENT DIVERGENCE: the exit channel and the entry channel
        are the same bars, other side (donchian.py:146-152). Letting them differ
        changes the exit without changing the entry, so a backtest would measure
        an exit rule nobody chose."""
        g = build_v020_graph()
        trend = next(b for b in g.branches if b.id == "trend")
        bad = dataclasses.replace(
            trend, exits=dataclasses.replace(trend.exits, channel_period=25)
        )
        others = tuple(b for b in g.branches if b.id != "trend")
        msg = self._bad(dataclasses.replace(g, branches=(bad,) + others))
        assert "channel_period=25" in msg and "entry_period=20" in msg

    def test_rejects_trail_atr_period_mismatch(self):
        """Prevents a SILENT DIVERGENCE: engine.py:537 uses ONE atr_value for both
        the entry stop and the trail, so two ATR periods would mean the trail
        ratchets on a series the stop was never derived from."""
        g = build_v020_graph(trail_enabled=True)
        trend = next(b for b in g.branches if b.id == "trend")
        bad = dataclasses.replace(
            trend, exits=dataclasses.replace(trend.exits, trail_atr_period=21)
        )
        others = tuple(b for b in g.branches if b.id != "trend")
        msg = self._bad(dataclasses.replace(g, branches=(bad,) + others))
        assert "trail_atr_period=21" in msg and "atr_period=14" in msg

    def test_rejects_bad_trigger_spec(self):
        g = build_v020_graph()
        msg = self._bad(dataclasses.replace(g, trigger=TriggerSpec(lookback_bars=0)))
        assert "lookback_bars" in msg
        msg = self._bad(dataclasses.replace(g, trigger=TriggerSpec(volume_lookback=0)))
        assert "volume_lookback" in msg
        msg = self._bad(dataclasses.replace(g, trigger=TriggerSpec(volume_high_ratio=0.0)))
        assert "volume_high_ratio" in msg

    def test_rejects_unknown_regime_timeframe(self):
        g = build_v020_graph()
        msg = self._bad(dataclasses.replace(g, regime=RegimeGate(timeframe="7h")))
        assert "'7h'" in msg

    def test_reports_every_problem_at_once(self):
        """A builder UI showing one error at a time is a bad UI, and a mutator
        that produced three illegal params should learn all three."""
        g = dataclasses.replace(build_v020_graph(), name="Bad!", schema_version=99)
        msg = self._bad(dataclasses.replace(g, regime=RegimeGate(timeframe="7h")))
        assert "3 problem(s)" in msg


class TestSaveLoad:
    def test_bare_name_resolves_under_strategy_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "STRATEGY_DIR", str(tmp_path / "strats"))
        p = fgraph.save(build_v020_graph())
        assert p == tmp_path / "strats" / "donchian-v020.strategy.json"
        assert p.exists()

    def test_file_ends_with_newline_and_is_byte_stable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "STRATEGY_DIR", str(tmp_path / "s"))
        g = build_v020_graph()
        first = fgraph.save(g).read_bytes()
        second = fgraph.save(g).read_bytes()
        assert first == second
        assert first.endswith(b"\n")

    def test_save_validates_first_and_writes_nothing_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "STRATEGY_DIR", str(tmp_path / "s"))
        bad = dataclasses.replace(build_v020_graph(), branches=())
        with pytest.raises(GraphError):
            fgraph.save(bad)
        assert not fgraph.path_for("donchian-v020").exists()

    def test_load_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "STRATEGY_DIR", str(tmp_path / "s"))
        g = build_v020_graph()
        assert fgraph.load(fgraph.save(g)) == g

    def test_load_corrupt_json_raises_grapherror(self, tmp_path):
        p = tmp_path / "broken.strategy.json"
        p.write_text("{not json")
        with pytest.raises(GraphError, match="not valid JSON"):
            fgraph.load(p)

    def test_load_missing_file_raises_grapherror(self, tmp_path):
        with pytest.raises(GraphError, match="no strategy graph at"):
            fgraph.load(tmp_path / "absent.strategy.json")

    def test_committed_parity_graph_validates(self):
        """The committed fixture is the record of what was tested."""
        g = fgraph.load("data/strategies/donchian-v020.strategy.json")
        assert graph_hash(g) == V020_DIGEST
        assert g == build_v020_graph()


class TestBranchHash:
    def test_same_detector_config_shares_a_branch_hash(self):
        """Why the candidate memo works across a population: graphs sharing a
        detector configuration share its candidates."""
        a, b = build_v020_graph(), build_v020_graph(target_enabled=True)
        ta = next(x for x in a.branches if x.id == "trend")
        tb = next(x for x in b.branches if x.id == "trend")
        assert fgraph.branch_hash(a, ta) == fgraph.branch_hash(b, tb)

    def test_detector_param_change_changes_the_branch_hash(self):
        a = build_v020_graph()
        ta = next(x for x in a.branches if x.id == "trend")
        changed = dataclasses.replace(
            ta, detector=dataclasses.replace(ta.detector, params={"entry_period": 30})
        )
        assert fgraph.branch_hash(a, ta) != fgraph.branch_hash(a, changed)
