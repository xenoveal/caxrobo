"""
Parent-side breeding: a scored generation becomes the next generation's rows
(v0.3.0 Phase 6).

ALL RANDOMNESS LIVES HERE, IN THE PARENT, BEFORE DISPATCH. That is the single
decision that makes parallelism and reproducibility compatible: workers become
pure functions of (graph, window), so worker count, completion order and machine
speed cannot change the population. A mutation performed inside a worker would
destroy reproducibility with no test obviously failing.

Each member's RNG is derived from (campaign_seed, gen_index, member_index) — NOT
drawn from one shared stream advanced in loop order — so changing EVO_ELITES or
the population size does not reshuffle every downstream member.

Elites are carried UNCHANGED and are still submitted for scoring, which charges a
trial (A6): this generation's window differs from the last one's, so an elite's
previous score is not a score on this window, and reusing it would be a cached
result without a ledger row — the exact bypass this phase exists to prevent.
"""

import copy
import json
import logging

from trading_bot import config
from trading_bot.evolution import population
from trading_bot.framework import registry
from trading_bot.framework.errors import GraphError

logger = logging.getLogger("trading_bot")

__all__ = ("MUTATOR_KEYS", "breed", "member_id_for", "seed_members")

# The two Mutator plug-ins Phase 6 ships. Looked up through the registry, so a
# Phase 8 mutator appears here by registering, not by editing this tuple.
MUTATOR_KEYS = ("mutator.param-jitter", "mutator.graph-edit")


def member_id_for(campaign_id: str, gen_index: int, member_index: int) -> str:
    """"<campaign_id>:<gen>:<idx>", zero-padded so ids sort lexically."""
    return f"{campaign_id}:{gen_index:03d}:{member_index:04d}"


def _choose_mutator(rng) -> str:
    """graph-edit with probability EVO_GRAPH_EDIT_SHARE, else param-jitter."""
    return (
        "mutator.graph-edit"
        if rng.random() < config.EVO_GRAPH_EDIT_SHARE
        else "mutator.param-jitter"
    )


def _apply(key: str, parent_graph, rng):
    """Run one registered mutator. Returns (child_graph, diff_list).

    The registry is the authority on the mutator's parameters: `spec.defaults()`
    resolves every declared ParamSpec, so a plug-in that gains a parameter needs
    no change here.
    """
    spec = registry.get(key)
    return spec.impl(parent_graph, rng, **spec.defaults())


def _breed_one(parent_graph, rng, *, seen_hashes: set[str]):
    """One offspring genome, with duplicate avoidance.

    A duplicate graph_hash is REDRAWN up to EVO_DEDUP_MAX_REDRAWS times from the
    SAME rng, so the redraw sequence is itself reproducible; after that the
    duplicate is accepted and logged at INFO. Duplicates are prevented at
    BREEDING time and never discounted at scoring time (A6) — a duplicate that
    reaches the oracle is charged a trial like anything else.

    Raises:
        GraphError: When a mutator cannot produce a valid distinct child and
            config.EVO_STRICT_MUTATORS is True.
    """
    last_exc: GraphError | None = None
    for attempt in range(config.EVO_DEDUP_MAX_REDRAWS + 1):
        key = _choose_mutator(rng)
        try:
            child, diff = _apply(key, parent_graph, rng)
        except GraphError as exc:
            last_exc = exc
            logger.debug("breed: %s failed (%s)", key, exc)
            continue
        h = population.graph_hash(child)
        if h not in seen_hashes:
            return child, key, diff, h, attempt
        if attempt == config.EVO_DEDUP_MAX_REDRAWS:
            logger.info(
                "breed: accepting duplicate graph %s after %d redraws — the "
                "reachable mutation space is small until Phase 8 widens it",
                h[:12], attempt,
            )
            return child, key, diff, h, attempt
    if config.EVO_STRICT_MUTATORS:
        raise GraphError(
            f"no mutator produced a valid child in "
            f"{config.EVO_DEDUP_MAX_REDRAWS + 1} attempts; last error: {last_exc}. "
            f"EVO_STRICT_MUTATORS is True, so this aborts the campaign rather than "
            f"carrying an unmutated parent that would score as a fresh candidate."
        )
    logger.warning(
        "breed: no valid mutation available (%s); carrying the parent unchanged. "
        "It is still scored and still charged a trial.", last_exc,
    )
    return parent_graph, "", [], population.graph_hash(parent_graph), -1


def _member(campaign, gen_index: int, member_index: int, graph, *, role: str,
            mutator: str, diff, parent_member_id: str | None, graph_hash: str):
    return population.Member(
        member_id=member_id_for(campaign.campaign_id, gen_index, member_index),
        campaign_id=campaign.campaign_id,
        gen_index=gen_index,
        member_index=member_index,
        graph_hash=graph_hash,
        graph_json=json.dumps(graph.to_dict(), sort_keys=True, separators=(",", ":")),
        rng_seed=population.member_seed(campaign.seed, gen_index, member_index),
        role=role,
        parent_member_id=parent_member_id,
        mutator=mutator,
        mutation_json=json.dumps(diff, sort_keys=True, default=str),
        graph=graph,
    )


def seed_members(campaign, seed_graph, *, seen_hashes: set[str], population_size=None):
    """Generation 0: the unmutated seed at index 0, then mutated descendants.

    The seed is member 0 and role='seed' so the audit round can find it, and so
    "the best beat the seed" is a comparison against a row that was scored by the
    same oracle on the same window — not against a remembered number.
    """
    if population_size is None:
        population_size = campaign.population
    seen = set(seen_hashes)
    members = [
        _member(
            campaign, 0, 0, seed_graph, role="seed", mutator="", diff=[],
            parent_member_id=None, graph_hash=campaign.seed_graph_hash,
        )
    ]
    seen.add(campaign.seed_graph_hash)
    for idx in range(1, int(population_size)):
        rng = population.member_rng(campaign.seed, 0, idx)
        child, key, diff, h, _ = _breed_one(seed_graph, rng, seen_hashes=seen)
        seen.add(h)
        members.append(
            _member(
                campaign, 0, idx, child, role="offspring", mutator=key, diff=diff,
                parent_member_id=members[0].member_id, graph_hash=h,
            )
        )
    seen_hashes.update(seen)
    return members


def breed(campaign, gen_index: int, *, elites, parents, seen_hashes: set[str]):
    """Turn selected parents into the next generation's Member rows.

    Args:
        campaign: The Campaign (supplies seed, id and population).
        gen_index: The generation being bred (>= 1).
        elites: Members carried UNCHANGED, in rank order. Re-scored on this
            generation's window, so each still charges a trial.
        parents: One Member per offspring slot, chosen by
            tournament.select_parents. len(elites) + len(parents) is the new
            population size.
        seen_hashes: Every graph_hash the campaign has bred. Updated in place, and
            seeded from population_members on --resume so a resumed campaign does
            not re-breed duplicates it already paid for.

    Returns:
        list[Member], in member_index order.

    Raises:
        GraphError: Propagated from a mutator under EVO_STRICT_MUTATORS. This
            happens in the PARENT, before dispatch, and aborts the campaign —
            deliberately a different severity from a per-task evaluation error,
            which is recorded and counted.
    """
    members: list[population.Member] = []
    idx = 0
    for elite in elites:
        members.append(
            _member(
                campaign, gen_index, idx, elite.graph, role="elite", mutator="",
                diff=[], parent_member_id=elite.member_id,
                graph_hash=elite.graph_hash,
            )
        )
        idx += 1
    for parent in parents:
        rng = population.member_rng(campaign.seed, gen_index, idx)
        child, key, diff, h, _ = _breed_one(
            parent.graph, rng, seen_hashes=seen_hashes
        )
        seen_hashes.add(h)
        members.append(
            _member(
                campaign, gen_index, idx, child, role="offspring", mutator=key,
                diff=diff, parent_member_id=parent.member_id, graph_hash=h,
            )
        )
        idx += 1
    return members


def graph_from_json(graph_json: str):
    """Rebuild a StrategyGraph from a persisted population_members.graph_json.

    Deliberately a copy of the parsed payload: StrategyGraph.from_dict keeps
    references to the mapping it was handed, and a caller reusing the payload
    would otherwise be able to reach into a graph another member owns.
    """
    from trading_bot.framework.graph import StrategyGraph

    return StrategyGraph.from_dict(copy.deepcopy(json.loads(graph_json)))
