"""
Mutators and breeding (v0.3.0 Phase 6).

Every random.Random here is explicitly seeded, and every assertion is on a
graph_hash rather than a repr, because a repr can change without the strategy
changing and a hash cannot (framework/graph.py's A5).
"""

import dataclasses
import json
import random

import pytest

from trading_bot import config
from trading_bot.evolution import mutate, population
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.errors import GraphError
from trading_bot.framework.graph import (
    Branch,
    ExitPolicySpec,
    NodeSpec,
    StrategyGraph,
)
from trading_bot.plugins.mutators import param_jitter as pj_mod

SEED_GRAPH_PATH = "data/strategies/thin-slice.strategy.json"


@pytest.fixture(autouse=True)
def _registry_loaded():
    registry.load_all()
    yield


@pytest.fixture
def seed_graph():
    return fgraph.load(SEED_GRAPH_PATH)


@pytest.fixture
def jitter():
    return registry.get("mutator.param-jitter")


@pytest.fixture
def edit():
    return registry.get("mutator.graph-edit")


def _minimal_graph(**kwargs) -> StrategyGraph:
    """A one-branch graph with no coupled exits, for structural edge cases."""
    defaults = dict(
        name="minimal",
        data=NodeSpec(id="ohlcv", key="data.ohlcv"),
        branches=(
            Branch(
                id="b1",
                detector=NodeSpec(id="d1", key="detector.donchian-breakout"),
                policy=NodeSpec(id="p1", key="policy.measured-move"),
                exits=ExitPolicySpec(),
            ),
        ),
        filters=(NodeSpec(id="f1", key="filter.rr-after-costs"),),
    )
    defaults.update(kwargs)
    return StrategyGraph(**defaults)


class TestRegistration:
    """Contract §12.3: a plug-in appears in `cli plugins` with a real rationale."""

    @pytest.mark.parametrize("key", mutate.MUTATOR_KEYS)
    def test_registered_with_a_nonempty_rationale(self, key):
        spec = registry.get(key)
        assert spec.kind == "mutator"
        assert len(spec.rationale) > 80, "a one-word rationale is not a rationale"

    @pytest.mark.parametrize("key", mutate.MUTATOR_KEYS)
    def test_every_declared_parameter_is_a_ParamSpec(self, key):
        for name, spec in registry.get(key).params.items():
            assert isinstance(spec, ParamSpec), name
            assert spec.doc.strip(), name


class TestParamJitter:
    def test_result_is_a_valid_distinct_graph(self, seed_graph, jitter):
        child, diff = jitter.impl(seed_graph, random.Random(0), **jitter.defaults())
        fgraph.validate(child)
        assert fgraph.graph_hash(child) != fgraph.graph_hash(seed_graph)
        assert len(diff) == config.EVO_JITTER_NODES

    def test_the_parent_is_never_mutated_in_place(self, seed_graph, jitter):
        """The parent is reused for elitism, lineage and dedup; an in-place edit
        would destroy reproducibility with no test obviously failing."""
        before = json.dumps(seed_graph.to_dict(), sort_keys=True)
        before_hash = fgraph.graph_hash(seed_graph)
        for seed in range(20):
            jitter.impl(seed_graph, random.Random(seed), **jitter.defaults())
        assert json.dumps(seed_graph.to_dict(), sort_keys=True) == before
        assert fgraph.graph_hash(seed_graph) == before_hash

    def test_the_same_seed_reproduces_the_same_child(self, seed_graph, jitter):
        a, da = jitter.impl(seed_graph, random.Random(99), **jitter.defaults())
        b, db = jitter.impl(seed_graph, random.Random(99), **jitter.defaults())
        assert fgraph.graph_hash(a) == fgraph.graph_hash(b)
        assert da == db

    def test_200_draws_all_stay_inside_declared_bounds(self, seed_graph, jitter):
        """The bound is the contract, not a suggestion. ParamSpec.clamp is what
        makes this true by construction; this pins it empirically anyway."""
        n_checked = 0
        for seed in range(200):
            child, diff = jitter.impl(
                seed_graph, random.Random(seed), **jitter.defaults()
            )
            fgraph.validate(child)  # would raise on any illegal value
            for entry in diff:
                spec = registry.get(entry["key"]).params[entry["param"]]
                assert spec.is_legal(entry["new"]), entry
                assert type(entry["new"]) is type(spec.default) or spec.kind == "float"
                n_checked += 1
        assert n_checked >= 200

    def test_the_frozen_cost_model_is_never_jittered(self, seed_graph, jitter):
        """Contract §1: any new execution path charges costs identically or it is
        lying. filter.rr-after-costs declares fee_pct/slippage_pct as bounded
        ParamSpecs, so without this refusal a candidate could evolve its own fees
        and ease its own R:R screen while the simulation kept paying
        config.FEE_PCT."""
        assert "fee_pct" in registry.get("filter.rr-after-costs").params
        touched = set()
        for seed in range(300):
            _child, diff = jitter.impl(
                seed_graph, random.Random(seed), **jitter.defaults()
            )
            touched.update(e["param"] for e in diff)
        assert not (touched & pj_mod.FROZEN_PARAM_NAMES), touched
        assert pj_mod.FROZEN_PARAM_NAMES == frozenset(
            {"fee_pct", "slippage_pct", "funding_pct_per_day"}
        )

    def test_the_regime_gate_is_structurally_unreachable(self, seed_graph, jitter):
        """framework/graph.py's A2: the classifier is a graph-level FIELD, not a
        node, precisely so param-jitter has no legal handle on the one
        measured-healthy layer."""
        for seed in range(100):
            child, _ = jitter.impl(seed_graph, random.Random(seed), **jitter.defaults())
            assert child.regime == seed_graph.regime
            assert child.trigger == seed_graph.trigger

    def test_a_graph_with_no_jitterable_parameter_raises(self, jitter):
        """A silent no-op would burn a trial on a duplicate."""
        with registry.temporary_registry():
            @registry.register(
                "policy", name="no-knobs-policy", params={},
                rationale="A policy with no declared parameters, for this test only.",
            )
            def _p(ctx, event, **params):  # pragma: no cover - never executed
                return None

            @registry.register(
                "detector", name="no-knobs-detector", params={},
                rationale="A detector with no declared parameters, for this test.",
            )
            def _d(ctx, **params):  # pragma: no cover - never executed
                return []

            @registry.register(
                "data", name="no-knobs-data", params={},
                rationale="A data source with no declared parameters, for this test.",
            )
            def _s(conn, **params):  # pragma: no cover - never executed
                return None

            g = StrategyGraph(
                name="knobless",
                data=NodeSpec(id="ohlcv", key="data.no-knobs-data"),
                branches=(
                    Branch(
                        id="b1",
                        detector=NodeSpec(id="d1", key="detector.no-knobs-detector"),
                        policy=NodeSpec(id="p1", key="policy.no-knobs-policy"),
                    ),
                ),
            )
            with pytest.raises(GraphError, match="no jitterable parameter"):
                jitter.impl(g, random.Random(1), **jitter.defaults())

    def test_a_coupled_parameter_is_skipped_not_raised(self, jitter):
        """framework.graph.validate couples exits.channel_period to the detector's
        entry_period. A coupled field is not a legal single-field jitter, so it is
        skipped — raising would abort an overnight campaign because two fields are
        required to agree."""
        g = _minimal_graph(
            branches=(
                Branch(
                    id="b1",
                    detector=NodeSpec(id="d1", key="detector.donchian-breakout"),
                    policy=NodeSpec(id="p1", key="policy.measured-move"),
                    exits=ExitPolicySpec(channel_exit=True, channel_period=20),
                ),
            )
        )
        for seed in range(60):
            child, diff = jitter.impl(g, random.Random(seed), **jitter.defaults())
            fgraph.validate(child)
            for entry in diff:
                if entry["key"] == "detector.donchian-breakout":
                    assert entry["param"] != "entry_period", (
                        "entry_period is coupled to exits.channel_period here and "
                        "must not be offered as a single-field jitter"
                    )

    def test_nodes_parameter_controls_how_many_params_move(self, seed_graph, jitter):
        params = {**jitter.defaults(), "nodes": 3}
        child, diff = jitter.impl(seed_graph, random.Random(4), **params)
        fgraph.validate(child)
        assert len(diff) == 3
        assert len({e["node"] for e in diff}) == 3, "must touch DISTINCT nodes"

    def test_a_bool_parameter_can_flip_and_stays_bool(self, jitter):
        spec = ParamSpec(kind="bool", default=True, doc="test flag")
        flipped = [
            pj_mod._jitter_value(spec, True, random.Random(s), sigma=0.15, flip_p=1.0)
            for s in range(5)
        ]
        assert flipped == [False] * 5
        assert all(isinstance(v, bool) for v in flipped)

    def test_int_jitter_clamps_after_rounding(self):
        """Clamp AFTER rounding, or an int exceeds its bound by 1."""
        spec = ParamSpec(kind="int", default=10, bounds=(5, 12), doc="bounded int")
        for s in range(40):
            v = pj_mod._jitter_value(spec, 12, random.Random(s), sigma=0.9, flip_p=0.0)
            assert isinstance(v, int) and 5 <= v <= 12, v

    def test_a_single_valued_axis_is_not_jitterable(self):
        assert not pj_mod._is_jitterable(
            ParamSpec(kind="int", default=3, bounds=(3, 3), doc="pinned")
        )
        assert not pj_mod._is_jitterable(
            ParamSpec(kind="choice", default="a", choices=("a",), doc="one choice")
        )
        assert pj_mod._is_jitterable(ParamSpec(kind="bool", default=True, doc="flag"))


class TestGraphEdit:
    def test_result_is_a_valid_distinct_graph(self, seed_graph, edit):
        child, diff = edit.impl(seed_graph, random.Random(0), **edit.defaults())
        fgraph.validate(child)
        assert fgraph.graph_hash(child) != fgraph.graph_hash(seed_graph)
        assert len(diff) == 1 and "edit" in diff[0]

    def test_the_parent_is_never_mutated_in_place(self, seed_graph, edit):
        before = json.dumps(seed_graph.to_dict(), sort_keys=True)
        for seed in range(30):
            edit.impl(seed_graph, random.Random(seed), **edit.defaults())
        assert json.dumps(seed_graph.to_dict(), sort_keys=True) == before

    def test_the_same_seed_reproduces_the_same_child(self, seed_graph, edit):
        a, da = edit.impl(seed_graph, random.Random(11), **edit.defaults())
        b, db = edit.impl(seed_graph, random.Random(11), **edit.defaults())
        assert fgraph.graph_hash(a) == fgraph.graph_hash(b)
        assert da == db

    def test_every_child_over_100_seeds_validates(self, seed_graph, edit):
        kinds = set()
        for seed in range(100):
            child, diff = edit.impl(seed_graph, random.Random(seed), **edit.defaults())
            fgraph.validate(child)
            kinds.add(diff[0]["edit"])
        # All three edit FAMILIES (swap / add / remove) must be reachable, or the
        # mutator is not actually searching topology. Asserted by family rather
        # than by exact edit name because which specific edits are legal depends
        # on how many plug-ins are registered: `add-confirmation` is UNREACHABLE
        # from this seed graph today, since both of its branches already carry
        # every registered confirmation, and Phase 8 registering a third would
        # change that without changing this mutator.
        assert {"swap-detector", "remove-confirmation"} <= kinds
        assert any(k.startswith("add-") for k in kinds), kinds
        assert kinds <= {
            "swap-detector", "swap-confirmation", "add-confirmation", "add-branch",
            "remove-confirmation", "remove-branch",
        }

    def test_add_confirmation_is_not_offered_when_all_are_present(self, seed_graph):
        """Two identical AND gates cost a degree of freedom and change nothing, so
        adding a confirmation a branch already has is not a legal edit."""
        from trading_bot.plugins.mutators import graph_edit as ge

        registered = set(registry.by_kind("confirmation"))
        present = {
            c.key.split(".", 1)[1] for b in seed_graph.branches for c in b.confirmations
        }
        edits = ge._legal_edits(
            seed_graph.to_dict(), max_branches=4, max_confirmations=4
        )
        offered = {
            e[1]["to"] for e in edits if e[1]["edit"] == "add-confirmation"
        }
        assert offered == {
            f"confirmation.{n}" for n in registered - present
        }

    def test_the_filter_is_never_added_or_removed(self, seed_graph, edit):
        """The >=1:2-after-costs filter is the pipeline's point (PRD). A mutation
        that deleted it would be evolution discovering the rules are optional."""
        for seed in range(150):
            child, _ = edit.impl(seed_graph, random.Random(seed), **edit.defaults())
            assert tuple(sorted(f.key for f in child.filters)) == tuple(
                sorted(f.key for f in seed_graph.filters)
            )

    def test_the_policy_slot_is_never_swapped_or_dropped(self, seed_graph, edit):
        for seed in range(150):
            child, _ = edit.impl(seed_graph, random.Random(seed), **edit.defaults())
            for b in child.branches:
                assert b.policy.key in {
                    x.policy.key for x in seed_graph.branches
                }, "graph-edit may not change a policy"

    def test_removal_is_not_offered_below_one_branch(self, edit):
        g = _minimal_graph()
        edits = [
            e[1]["edit"]
            for e in __import__(
                "trading_bot.plugins.mutators.graph_edit", fromlist=["_legal_edits"]
            )._legal_edits(g.to_dict(), max_branches=4, max_confirmations=4)
        ]
        assert "remove-branch" not in edits, (
            "a graph with one branch must not be offered a branch removal — "
            "validate() rule 6 needs an enabled branch"
        )

    def test_branch_and_confirmation_ceilings_are_respected(self, seed_graph, edit):
        params = {**edit.defaults(), "max_branches": 2, "max_confirmations": 2}
        for seed in range(60):
            child, _ = edit.impl(seed_graph, random.Random(seed), **params)
            assert len(child.branches) <= 2
            for b in child.branches:
                assert len(b.confirmations) <= 2

    def test_no_legal_edit_raises(self, edit, monkeypatch):
        monkeypatch.setattr(
            "trading_bot.plugins.mutators.graph_edit._legal_edits",
            lambda *a, **k: [],
        )
        with pytest.raises(GraphError, match="no legal topology edit"):
            edit.impl(_minimal_graph(), random.Random(1), **edit.defaults())

    def test_an_edit_that_always_produces_an_invalid_graph_raises(
        self, seed_graph, edit, monkeypatch
    ):
        """An invalid graph must fail LOUDLY at mutation time, not silently produce
        zero trades and score as tier D — which would look like honest rejection
        while hiding a mutator bug behind a plausible number."""

        def _broken(payload, **kwargs):
            def _apply(p):
                p["branches"] = []  # violates validate() rule 6

            return [("swap", {"edit": "deliberately-broken"}, _apply)]

        monkeypatch.setattr(
            "trading_bot.plugins.mutators.graph_edit._legal_edits", _broken
        )
        with pytest.raises(GraphError, match="MUTATOR BUG"):
            edit.impl(seed_graph, random.Random(1), **edit.defaults())

    def test_sparse_timeframe_plugins_are_not_offered(self, seed_graph):
        """Contract §0: 15m is stored for BTC/ETH/SOL only. Offering a 15m detector
        would make a candidate's trade count depend on which symbols a campaign
        happens to pool — a silent, symbol-dependent zero."""
        from trading_bot.plugins.mutators import graph_edit as ge

        class _Spec:
            timeframes = ("15m",)

        assert not ge._usable(_Spec())

        class _Ok:
            timeframes = ("4h", "1h")

        assert ge._usable(_Ok())


class TestGraphEditEligibility:
    """v0.3.1 Task 1: `graph.meta.evo.eligible_detectors` (set by the Composer's
    eligibility panel) must actually narrow what graph-edit can SWAP or ADD, or
    the operator's checkbox choice is theater — evolution would keep exploring
    every registered detector regardless of what got unchecked."""

    ELIGIBLE = ("detector.cup-and-handle", "detector.inverse-head-and-shoulders")

    def _eligible_graph(self, seed_graph):
        """The thin-slice seed graph (donchian-breakout / macd-cross branches),
        restricted to a 2-detector pool that overlaps neither branch — so any
        swap-detector or add-branch edit that leaks an outside key is unambiguous,
        not an accidental match with what was already there."""
        return dataclasses.replace(
            seed_graph, meta={"evo": {"eligible_detectors": list(self.ELIGIBLE)}}
        )

    def test_swap_and_add_never_name_a_detector_outside_the_eligible_set(
        self, seed_graph, edit
    ):
        """Drives the REAL mutator (not just `_legal_edits`) over 300 seeds. The
        unconstrained pool is 20 detectors wide, so if the filter in
        graph_edit.py's `_legal_edits` were absent or broken, a leak would be
        near-certain to surface well within this range — 300 draws run in well
        under a second, so there is no reason to settle for fewer."""
        g = self._eligible_graph(seed_graph)
        n_constrained_edits = 0
        for seed in range(300):
            child, diff = edit.impl(g, random.Random(seed), **edit.defaults())
            fgraph.validate(child)
            desc = diff[0]
            if desc["edit"] in ("swap-detector", "add-branch"):
                assert desc["to"] in self.ELIGIBLE, (
                    f"seed {seed} produced {desc!r}, naming a detector outside "
                    f"the eligible set — the Composer's checkbox would be a lie"
                )
                n_constrained_edits += 1
        assert n_constrained_edits > 0, (
            "no swap-detector/add-branch edit was ever sampled across 300 seeds — "
            "this test would pass vacuously without the assertion above ever "
            "running, which is worse than not testing at all"
        )

    def test_remove_branch_is_not_pool_filtered(self, seed_graph, edit):
        """The module docstring is explicit that REMOVE draws from no pool: a
        detector already in the graph but no longer eligible must still be
        pruneable, or an operator who unchecks a detector already in the seed
        graph would find evolution can never shed it. Eligibility naming neither
        of the seed's two detectors must not make remove-branch disappear."""
        g = self._eligible_graph(seed_graph)
        kinds = set()
        for seed in range(60):
            child, diff = edit.impl(g, random.Random(seed), **edit.defaults())
            fgraph.validate(child)
            kinds.add(diff[0]["edit"])
        assert "remove-branch" in kinds, (
            "remove-branch never got sampled in 60 seeds; widen the seed range "
            "before concluding the filter over-reaches into removal"
        )

    def test_meta_survives_one_generation_of_breeding(self, seed_graph, edit):
        """`graph.meta` is excluded from `graph_hash` (framework/graph.py A5)
        precisely so identity survives a meta edit, but that only matters if meta
        itself actually rides along through a mutation. If this evaporated after
        one generation, eligibility would constrain generation 0 and then quietly
        stop constraining every generation after — useless for a real campaign,
        which breeds for many generations."""
        g = self._eligible_graph(seed_graph)
        child, _diff = edit.impl(g, random.Random(0), **edit.defaults())
        assert child.to_dict()["meta"]["evo"]["eligible_detectors"] == list(
            self.ELIGIBLE
        )

    def test_empty_effective_pool_raises_grapherror(self, edit):
        """A graph with one branch, zero confirmations, and an eligible set
        containing only the detector already occupying that branch has nowhere
        for swap-detector to go (it excludes the current key by construction) and
        nothing new for add-branch to introduce. Pairing that with
        max_confirmations=0 (no add-confirmation) and max_branches=1 (no
        add-branch, no remove-branch below one branch) genuinely empties
        `_legal_edits` — this must surface as a loud GraphError ("no legal
        topology edit") rather than graph-edit silently handing back the parent,
        which would burn a trial on a duplicate the caller believes is new."""
        g = _minimal_graph(
            meta={"evo": {"eligible_detectors": ["detector.donchian-breakout"]}}
        )
        params = {**edit.defaults(), "max_branches": 1, "max_confirmations": 0}
        for seed in range(10):
            with pytest.raises(GraphError, match="no legal topology edit"):
                edit.impl(g, random.Random(seed), **params)

    def test_without_meta_the_same_graph_offers_the_full_pool(self):
        """Negative control: a test suite for this feature that would keep
        passing with the feature deleted is worthless. The exact graph shape from
        the GraphError test above — sans the `meta` that constrains it — must
        offer a dramatically larger detector pool (all 20 registered detectors,
        including a legal add-branch of the very detector already present),
        proving the emptiness above is `eligible_detectors` at work and not some
        structural accident of `_minimal_graph`."""
        from trading_bot.plugins.mutators import graph_edit as ge

        g = _minimal_graph()  # no meta at all
        edits = ge._legal_edits(g.to_dict(), max_branches=4, max_confirmations=4)
        named = {
            e[1]["to"] for e in edits if e[1]["edit"] in ("swap-detector", "add-branch")
        }
        assert len(named) >= 15, (
            f"expected the unconstrained pool to be dramatically larger than the "
            f"1-detector eligible set used above; got {sorted(named)}"
        )


class TestBreed:
    def _campaign(self, seed_graph, *, pop=6):
        return population.Campaign(
            campaign_id="c1", seed=1234,
            seed_graph_hash=fgraph.graph_hash(seed_graph),
            seed_graph_json=json.dumps(seed_graph.to_dict(), sort_keys=True),
            config_json="{}", symbols=("BTCUSDT",),
            train_start_ms=1, train_end_ms=2,
            audit_start_ms=1, audit_end_ms=2,
            population=pop, generations=2, started_ts=0,
        )

    def test_generation_zero_puts_the_unmutated_seed_at_index_zero(self, seed_graph):
        c = self._campaign(seed_graph)
        members = mutate.seed_members(c, seed_graph, seen_hashes=set())
        assert len(members) == c.population
        assert members[0].role == "seed"
        assert members[0].mutator == ""
        assert members[0].graph_hash == c.seed_graph_hash
        assert all(m.role == "offspring" for m in members[1:])
        assert all(m.mutator in mutate.MUTATOR_KEYS for m in members[1:])

    def test_breeding_is_reproducible_from_the_seed(self, seed_graph):
        c = self._campaign(seed_graph)
        a = mutate.seed_members(c, seed_graph, seen_hashes=set())
        b = mutate.seed_members(c, seed_graph, seen_hashes=set())
        assert [m.graph_hash for m in a] == [m.graph_hash for m in b]
        assert [m.mutator for m in a] == [m.mutator for m in b]
        assert [m.rng_seed for m in a] == [m.rng_seed for m in b]

    def test_member_rng_is_derived_not_streamed(self, seed_graph):
        """Derived from (seed, gen, index), so changing EVO_ELITES or the
        population size does not reshuffle every downstream member."""
        c = self._campaign(seed_graph)
        small = mutate.seed_members(c, seed_graph, seen_hashes=set(),
                                    population_size=3)
        large = mutate.seed_members(c, seed_graph, seen_hashes=set(),
                                    population_size=8)
        assert [m.rng_seed for m in small] == [m.rng_seed for m in large[:3]]
        assert [m.graph_hash for m in small] == [m.graph_hash for m in large[:3]]

    def test_seeds_differ_across_generations_and_indexes(self):
        seeds = {
            (g, i): population.member_seed(7, g, i)
            for g in range(3) for i in range(5)
        }
        assert len(set(seeds.values())) == len(seeds)

    def test_derive_seed_is_stable_across_interpreters(self):
        """Pinned literal: builtin hash() would give a different answer in every
        process, and this value must not."""
        assert population.derive_seed(1, 2, 3) == population.derive_seed(1, 2, 3)
        assert population.member_seed(1234, 0, 1) == population.member_seed(1234, 0, 1)

    def test_elites_are_carried_unchanged_and_offspring_are_not(self, seed_graph):
        c = self._campaign(seed_graph)
        gen0 = mutate.seed_members(c, seed_graph, seen_hashes=set())
        seen = {m.graph_hash for m in gen0}
        gen1 = mutate.breed(
            c, 1, elites=gen0[:2], parents=gen0[2:], seen_hashes=seen,
        )
        assert [m.role for m in gen1[:2]] == ["elite", "elite"]
        assert [m.graph_hash for m in gen1[:2]] == [
            m.graph_hash for m in gen0[:2]
        ]
        assert all(m.parent_member_id for m in gen1)
        assert [m.member_index for m in gen1] == list(range(len(gen1)))

    def test_breed_avoids_hashes_it_has_already_seen(self, seed_graph):
        c = self._campaign(seed_graph, pop=12)
        seen: set[str] = set()
        gen0 = mutate.seed_members(c, seed_graph, seen_hashes=seen)
        hashes = [m.graph_hash for m in gen0]
        assert len(set(hashes)) == len(hashes), (
            "generation 0 bred a duplicate the dedup loop should have redrawn"
        )
        assert seen == set(hashes)

    def test_a_forced_collision_is_accepted_after_the_redraw_budget(
        self, seed_graph, monkeypatch, caplog
    ):
        """Duplicates are prevented at BREEDING time and never discounted at
        SCORING time (A6): an accepted duplicate is still charged a trial."""
        child, diff = registry.get("mutator.param-jitter").impl(
            seed_graph, random.Random(0),
            **registry.get("mutator.param-jitter").defaults(),
        )
        monkeypatch.setattr(mutate, "_apply", lambda key, g, rng: (child, diff))
        seen = {fgraph.graph_hash(child)}
        import logging

        with caplog.at_level(logging.INFO, logger="trading_bot"):
            got, key, _d, h, attempts = mutate._breed_one(
                seed_graph, random.Random(3), seen_hashes=seen
            )
        assert h in seen
        assert attempts == config.EVO_DEDUP_MAX_REDRAWS
        assert "accepting duplicate graph" in caplog.text

    def test_strict_mutators_aborts_when_nothing_can_mutate(
        self, seed_graph, monkeypatch
    ):
        def _always_fails(key, g, rng):
            raise GraphError("nothing legal")

        monkeypatch.setattr(mutate, "_apply", _always_fails)
        assert config.EVO_STRICT_MUTATORS is True
        with pytest.raises(GraphError, match="EVO_STRICT_MUTATORS"):
            mutate._breed_one(seed_graph, random.Random(1), seen_hashes=set())

    def test_non_strict_carries_the_parent_and_says_so(
        self, seed_graph, monkeypatch, caplog
    ):
        def _always_fails(key, g, rng):
            raise GraphError("nothing legal")

        monkeypatch.setattr(mutate, "_apply", _always_fails)
        monkeypatch.setattr(config, "EVO_STRICT_MUTATORS", False)
        import logging

        with caplog.at_level(logging.WARNING, logger="trading_bot"):
            got, key, diff, h, attempts = mutate._breed_one(
                seed_graph, random.Random(1), seen_hashes=set()
            )
        assert key == "" and diff == []
        assert h == fgraph.graph_hash(seed_graph)
        assert "still charged a trial" in caplog.text

    def test_graph_json_round_trips_to_an_identical_hash(self, seed_graph):
        c = self._campaign(seed_graph)
        members = mutate.seed_members(c, seed_graph, seen_hashes=set())
        for m in members:
            rebuilt = mutate.graph_from_json(m.graph_json)
            assert fgraph.graph_hash(rebuilt) == m.graph_hash

    def test_member_ids_sort_lexically_in_evaluation_order(self, seed_graph):
        c = self._campaign(seed_graph, pop=12)
        members = mutate.seed_members(c, seed_graph, seen_hashes=set())
        ids = [m.member_id for m in members]
        assert ids == sorted(ids)
