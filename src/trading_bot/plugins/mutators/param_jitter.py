"""
mutator.param-jitter (v0.3.0 Phase 6): local search inside declared ParamSpec
bounds.

The cheapest useful move, and the only one that cannot change topology. Every
value is CLAMPED through ParamSpec.clamp rather than rejection-sampled, so a
jitter can never produce an out-of-bounds parameter — the bound is the contract,
not a suggestion.

TWO REFUSALS THAT ARE THE POINT OF THE FILE:

 1. A parameter with no legal RANGE is not jitterable and is SKIPPED, never
    invented around: an int/float axis whose bounds collapse to a point, or a
    choice with one option. ParamSpec exists so a mutator never has to guess a
    range (contract §3); guessing one here would defeat it.

 2. THE COST MODEL IS NOT SEARCHABLE. filter.rr-after-costs declares `fee_pct`
    and `slippage_pct` as bounded ParamSpecs so the filter can be TESTED under a
    different cost assumption — not so a population can evolve its own fees. The
    cost model is frozen (contract §1: "Any new execution path charges costs
    identically or it is lying"), and a candidate that lowered fee_pct would ease
    its own R:R screen while the simulated P&L kept paying config.FEE_PCT. Those
    names are refused by FROZEN_PARAM_NAMES below and a test pins it.

A jitter that would produce an INVALID graph (the coupling rules in
framework.graph.validate — exits.channel_period must equal the detector's
entry_period, exits.trail_atr_period the policy's atr_period) is a candidate that
is not legal, so it is skipped and the next candidate is tried. Only when NO
legal jitter exists does this raise GraphError. Deviation from the plan's literal
"raise on validate() failure", taken deliberately: a coherence coupling is not a
bug, and under EVO_STRICT_MUTATORS a raise here would abort an overnight campaign
because two fields are required to agree. The safety property the plan actually
wants — an invalid graph never escapes a mutator — is preserved exactly.
"""

import copy
import logging

from trading_bot import config
from trading_bot.framework import registry
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.errors import FrameworkError, GraphError
from trading_bot.framework.graph import StrategyGraph, validate

logger = logging.getLogger("trading_bot")

__all__ = ("FROZEN_PARAM_NAMES", "param_jitter")

# Parameter names no mutator may touch, whatever plug-in declares them, because
# they are the FROZEN COST MODEL (config.FEE_PCT / SLIPPAGE_PCT /
# FUNDING_PCT_PER_DAY, contract §1). Matched by name rather than by plug-in so a
# future filter that re-declares one is covered without an edit here.
FROZEN_PARAM_NAMES = frozenset({"fee_pct", "slippage_pct", "funding_pct_per_day"})


def _iter_node_paths(payload: dict):
    """(path, node_dict) for every node in a to_dict() payload.

    A path is the sequence of keys/indexes to the node, so a jitter can be
    written back into a deep copy without a second traversal.
    """
    yield ("data",), payload["data"]
    for i, branch in enumerate(payload["branches"]):
        yield ("branches", i, "detector"), branch["detector"]
        yield ("branches", i, "policy"), branch["policy"]
        for j, conf in enumerate(branch.get("confirmations", [])):
            yield ("branches", i, "confirmations", j), conf
    for i, filt in enumerate(payload.get("filters", [])):
        yield ("filters", i), filt


def _at(payload: dict, path: tuple) -> dict:
    """The node dict a path points at, inside `payload`."""
    node = payload
    for step in path:
        node = node[step]
    return node


def _is_jitterable(spec: ParamSpec) -> bool:
    """Whether a ParamSpec declares more than one legal value."""
    if spec.kind == "bool":
        return True
    if spec.kind == "choice":
        return len(spec.choices or ()) > 1
    low, high = spec.bounds  # type: ignore[misc]
    return high > low


def _jitter_value(spec: ParamSpec, old, rng, *, sigma: float, flip_p: float):
    """One clamped jitter of `old` under `spec`.

    int: a whole step of max(1, round(sigma * bound_width)) up or down, then
    clamped — clamping AFTER rounding, or an int can exceed its bound by 1.
    float: gaussian around the current value with sigma as a fraction of the
    bound width, clamped.
    bool: flips with probability flip_p.
    choice: uniform over the OTHER choices, so a choice jitter always moves.
    """
    if spec.kind == "bool":
        return (not bool(old)) if rng.random() < flip_p else bool(old)
    if spec.kind == "choice":
        alternatives = [c for c in (spec.choices or ()) if c != old]
        return rng.choice(alternatives) if alternatives else old
    low, high = spec.bounds  # type: ignore[misc]
    width = float(high) - float(low)
    if spec.kind == "int":
        step = max(1, int(round(sigma * width)))
        return spec.clamp(int(old) + rng.choice((-step, step)))
    return spec.clamp(rng.gauss(float(old), sigma * width))


@registry.register(
    "mutator",
    name="param-jitter",
    params={
        "nodes": ParamSpec(
            kind="int", default=config.EVO_JITTER_NODES, bounds=(1, 8),
            doc="How many distinct nodes one mutation touches (one parameter each)",
        ),
        "sigma": ParamSpec(
            kind="float", default=config.EVO_JITTER_SIGMA, bounds=(0.01, 1.0),
            doc="Jitter scale as a fraction of a numeric parameter's bound width",
        ),
        "flip_p": ParamSpec(
            kind="float", default=config.EVO_BOOL_FLIP_P, bounds=(0.0, 1.0),
            doc="Probability a boolean parameter flips",
        ),
    },
    rationale=(
        "Local search inside declared ParamSpec bounds: the cheapest useful move, "
        "and the only one that cannot change topology, so a jittered child is "
        "always structurally valid. Every value is clamped through ParamSpec.clamp "
        "rather than rejection-sampled, so a bound can never be exceeded. Two "
        "refusals are deliberate and pinned by tests: a parameter with no declared "
        "range is skipped rather than given an invented one, and the frozen cost "
        "model (fee_pct, slippage_pct, funding_pct_per_day) is never jittered — "
        "evolving your own fees while the simulation charges config.FEE_PCT is the "
        "one mutation that would make every downstream number a lie."
    ),
)
def param_jitter(graph, rng, *, nodes: int, sigma: float, flip_p: float):
    """Return a NEW graph with up to `nodes` parameters jittered.

    Args:
        graph: The parent StrategyGraph. NEVER mutated — the parent is reused for
            elitism, lineage and dedup.
        rng: A seeded random.Random. Same seed, same graph, same child (A7).
        nodes / sigma / flip_p: Resolved from this plug-in's ParamSpecs.

    Returns:
        (StrategyGraph, list[dict]) — the child and the applied diff, one entry
        per jitter: {"node", "key", "param", "old", "new"}. The diff is returned
        beside the graph because the graph alone does not say which move produced
        it, and "which mutation helped" is the only question a population search
        can answer about itself.

    Raises:
        GraphError: If the graph declares no jitterable parameter at all, or if
            every legal candidate would produce an invalid graph. A silent no-op
            would burn a trial on a duplicate.
    """
    payload = copy.deepcopy(graph.to_dict())

    candidates: list[tuple[tuple, str, ParamSpec]] = []
    for path, node in _iter_node_paths(payload):
        spec = registry.get(node["key"])
        for pname, pspec in spec.params.items():
            if pname in FROZEN_PARAM_NAMES:
                continue
            if not _is_jitterable(pspec):
                logger.debug(
                    "param-jitter: %s.%s declares one legal value; not jitterable",
                    node["key"], pname,
                )
                continue
            candidates.append((path, pname, pspec))

    if not candidates:
        raise GraphError(
            f"graph {graph.name!r} declares no jitterable parameter (every "
            f"declared parameter is frozen or has a single legal value), so "
            f"param-jitter cannot produce a distinct child; a silent no-op would "
            f"burn a trial on a duplicate"
        )

    rng.shuffle(candidates)
    diff: list[dict] = []
    touched: set[tuple] = set()
    for path, pname, pspec in candidates:
        if len(diff) >= int(nodes):
            break
        if path in touched:
            continue
        node = _at(payload, path)
        resolved = registry.get(node["key"]).resolve(node["params"])
        old = resolved[pname]
        new = _jitter_value(pspec, old, rng, sigma=sigma, flip_p=flip_p)
        if new == old and type(new) is type(old):
            continue  # a no-op draw: try another candidate rather than duplicate
        trial = copy.deepcopy(payload)
        _at(trial, path)["params"][pname] = new
        try:
            child = StrategyGraph.from_dict(trial)
            validate(child)
        except FrameworkError as exc:
            # Not a bug: framework.graph.validate couples some parameters
            # (exits.channel_period == detector entry_period). A coupled param is
            # simply not a legal single-field jitter.
            logger.debug(
                "param-jitter: %s.%s=%r rejected as illegal (%s)",
                node["key"], pname, new, exc,
            )
            continue
        payload = trial
        touched.add(path)
        diff.append(
            {"node": node["id"], "key": node["key"], "param": pname,
             "old": old, "new": new}
        )

    if not diff:
        raise GraphError(
            f"graph {graph.name!r}: no legal parameter jitter exists among "
            f"{len(candidates)} candidate(s) — every draw either was a no-op or "
            f"produced an invalid graph. Raising rather than returning the parent "
            f"unchanged, which would charge a trial for a duplicate."
        )
    return StrategyGraph.from_dict(payload), diff
