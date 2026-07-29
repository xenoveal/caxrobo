"""
mutator.graph-edit (v0.3.0 Phase 6): detector / confirmation swap, add, remove.

Topology search is the only move that can reach a strategy the seed's SHAPE
cannot express — the gap KNOWN-LIMITATIONS §0c records as never searched ("the
only search ever run moved exit management; the entry/feature space was never
explored").

LEGALITY IS ENUMERATED FIRST, THEN SAMPLED. `_legal_edits` builds every legal
move before any randomness is spent, so an illegal edit is never "attempted and
repaired". What counts as legal:

  SWAP   a detector or confirmation node's plug-in for another REGISTERED key of
         the same kind, new parameters at each ParamSpec default.
  ADD    a confirmation to a branch (never one the branch already has — two
         identical AND gates cost a degree of freedom and change nothing), or a
         second detector as a NEW BRANCH cloned from an existing branch's policy,
         regimes and exits. A Branch holds exactly one detector, so "add a
         detector" IS "add a branch".
  REMOVE a confirmation, or a whole branch while >= 1 branch remains.

NEVER: a second policy or filter, removing the policy, or removing the filter.
The >= 1:2-reward:risk-after-costs filter is the pipeline's point (PRD); a
mutation that deletes it would be evolution discovering that the rules are
optional.

A candidate whose result fails framework.graph.validate is DROPPED and the next
candidate tried — the coupling rules (exits.channel_period == the detector's
entry_period) make some swaps genuinely illegal rather than buggy. Only when NO
candidate survives does this raise GraphError. That distinction is the one the
plan asks to keep visible in the audit log: an INVALID GRAPH never escapes this
function (bug-proof), while a VALID graph that takes zero trades is a legitimate
tier-D result, recorded, charged and selected against.

ELIGIBILITY IS A POOL FILTER, NOT A TOPOLOGY RULE. `graph.meta.evo.eligible_detectors`,
when the operator sets it in the Composer, is read here and used to narrow the
`detectors` pool before SWAP and ADD-BRANCH enumerate it — those are the only two
move kinds that can introduce a detector key the graph does not already contain.
REMOVE-branch draws from no pool at all: a branch already in the graph can always
be pruned, eligible or not, because "evolution may not explore this detector" is
not the same claim as "evolution may not shed it once present". Left unenforced,
evolution would defeat the very restriction the operator asked for: legality would
still say "swap to ANY registered detector", silently reintroducing a key the
operator explicitly unchecked. The constraint is deliberately loose the other way
too — it never forces removal of an ineligible detector the seed graph already
carries; it only closes the door SWAP and ADD would otherwise open back into the
excluded pool. Malformed meta (a string where a dict or list belongs, a missing
key at any level) is treated as "no constraint" rather than an error: `graph.meta`
is free-form JSON that survived `StrategyGraph.to_dict`/`from_dict` round-trips
across possibly many generations, and a defensive read here must not turn a
shape mismatch into a campaign-aborting exception. Finally, note that `meta` is
EXCLUDED from `graph_hash` (framework.graph — `name` and `meta` are excluded from
the content hash), so filtering the pool by `eligible_detectors` never mints a new
strategy identity: the same graph content with a different eligibility list is
still, by design, the same strategy.
"""

import copy
import logging

from trading_bot import config
from trading_bot.framework import registry
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.errors import FrameworkError, GraphError
from trading_bot.framework.graph import StrategyGraph, validate

logger = logging.getLogger("trading_bot")

__all__ = ("EDIT_TYPES", "graph_edit")

EDIT_TYPES = ("swap", "add", "remove")

# Timeframes that exist for only SOME symbols (contract §0: 15m is stored for
# BTC/ETH/SOL only, 1d/4h/1h for all 20). Offering a plug-in that reads one would
# make a candidate's trade count depend on which symbols a campaign happens to
# pool — a silent, symbol-dependent zero rather than an honest result.
SPARSE_TIMEFRAMES = frozenset({"15m"})


def _usable(spec) -> bool:
    """Whether a plug-in reads only timeframes stored for every symbol."""
    return not (set(spec.timeframes) & SPARSE_TIMEFRAMES)


def _default_node(node_id: str, key: str) -> dict:
    """A node dict at every ParamSpec default (params={} resolves to defaults)."""
    return {"id": node_id, "key": key, "params": {}}


def _unique_id(base: str, taken: set[str]) -> str:
    """A graph-unique id matching framework.graph._ID_RE, derived from `base`."""
    candidate = base
    n = 2
    while candidate in taken:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _legal_edits(payload: dict, *, max_branches: int, max_confirmations: int):
    """Every legal edit as (type, description, apply_fn) over a to_dict payload.

    apply_fn takes a deep copy of the payload and mutates it in place; the caller
    owns the copying, so an aborted candidate cannot leak a half-edit.
    """
    detectors = {
        f"detector.{n}": s for n, s in registry.by_kind("detector").items() if _usable(s)
    }
    confirmations = {
        f"confirmation.{n}": s
        for n, s in registry.by_kind("confirmation").items()
        if _usable(s)
    }

    # Eligibility constrains the SWAP/ADD-BRANCH pool only (see module docstring
    # "ELIGIBILITY IS A POOL FILTER, NOT A TOPOLOGY RULE"). Confirmations are
    # untouched, and a detector already on the graph but no longer eligible can
    # still be pruned by REMOVE-branch, which draws from no pool at all.
    meta = payload.get("meta")
    evo = meta.get("evo") if isinstance(meta, dict) else None
    elig = evo.get("eligible_detectors") if isinstance(evo, dict) else None
    if isinstance(elig, list) and elig:
        elig_set = set(elig)
        detectors = {k: s for k, s in detectors.items() if k in elig_set}

    node_ids = {payload["data"]["id"]}
    for b in payload["branches"]:
        node_ids.add(b["detector"]["id"])
        node_ids.add(b["policy"]["id"])
        node_ids.update(c["id"] for c in b.get("confirmations", []))
    for f in payload.get("filters", []):
        node_ids.add(f["id"])
    branch_ids = {b["id"] for b in payload["branches"]}

    edits: list[tuple[str, dict, object]] = []

    for i, branch in enumerate(payload["branches"]):
        bid = branch["id"]
        current_det = branch["detector"]["key"]
        confs = branch.get("confirmations", [])
        conf_keys = {c["key"] for c in confs}

        # --- SWAP detector -------------------------------------------------
        for key in detectors:
            if key == current_det:
                continue

            def _swap_det(p, i=i, key=key):
                p["branches"][i]["detector"] = _default_node(
                    p["branches"][i]["detector"]["id"], key
                )

            edits.append(
                ("swap", {"edit": "swap-detector", "branch": bid,
                          "from": current_det, "to": key}, _swap_det)
            )

        # --- SWAP confirmation ---------------------------------------------
        for j, conf in enumerate(confs):
            for key in confirmations:
                if key in conf_keys:
                    continue

                def _swap_conf(p, i=i, j=j, key=key):
                    p["branches"][i]["confirmations"][j] = _default_node(
                        p["branches"][i]["confirmations"][j]["id"], key
                    )

                edits.append(
                    ("swap", {"edit": "swap-confirmation", "branch": bid,
                              "from": conf["key"], "to": key}, _swap_conf)
                )

        # --- ADD confirmation ----------------------------------------------
        if len(confs) < max_confirmations:
            for key in confirmations:
                if key in conf_keys:
                    continue
                new_id = _unique_id(f"{bid}-{key.split('.', 1)[1]}", node_ids)

                def _add_conf(p, i=i, key=key, new_id=new_id):
                    p["branches"][i].setdefault("confirmations", []).append(
                        _default_node(new_id, key)
                    )

                edits.append(
                    ("add", {"edit": "add-confirmation", "branch": bid, "to": key},
                     _add_conf)
                )

        # --- REMOVE confirmation -------------------------------------------
        for j, conf in enumerate(confs):

            def _rm_conf(p, i=i, j=j):
                del p["branches"][i]["confirmations"][j]

            edits.append(
                ("remove", {"edit": "remove-confirmation", "branch": bid,
                            "from": conf["key"]}, _rm_conf)
            )

        # --- REMOVE branch (i.e. remove a detector) ------------------------
        # Only while >= 1 branch remains: framework.graph.validate rule 6 needs an
        # enabled branch, and a graph with none "would read as a strategy with no
        # edge".
        if len(payload["branches"]) > 1:

            def _rm_branch(p, i=i):
                del p["branches"][i]

            edits.append(
                ("remove", {"edit": "remove-branch", "branch": bid,
                            "from": current_det}, _rm_branch)
            )

    # --- ADD detector as a NEW BRANCH --------------------------------------
    if payload["branches"] and len(payload["branches"]) < max_branches:
        template = payload["branches"][0]
        for key in detectors:
            short = key.split(".", 1)[1]
            new_bid = _unique_id(short, branch_ids)
            det_id = _unique_id(f"{new_bid}-det", node_ids)
            pol_id = _unique_id(f"{new_bid}-pol", node_ids)

            def _add_branch(p, key=key, new_bid=new_bid, det_id=det_id, pol_id=pol_id,
                            template=template):
                p["branches"].append(
                    {
                        "id": new_bid,
                        "detector": _default_node(det_id, key),
                        "policy": _default_node(pol_id, template["policy"]["key"]),
                        "regimes": list(template.get("regimes", ["any"])),
                        "confirmations": [],
                        "exits": copy.deepcopy(template["exits"]),
                        "enabled": True,
                    }
                )

            edits.append(
                ("add", {"edit": "add-branch", "branch": new_bid, "to": key},
                 _add_branch)
            )

    return edits


@registry.register(
    "mutator",
    name="graph-edit",
    params={
        "p_swap": ParamSpec(
            kind="float", default=0.5, bounds=(0.0, 1.0),
            doc="Relative weight of swapping a detector or confirmation",
        ),
        "p_add": ParamSpec(
            kind="float", default=0.3, bounds=(0.0, 1.0),
            doc="Relative weight of adding a confirmation or a branch",
        ),
        "p_remove": ParamSpec(
            kind="float", default=0.2, bounds=(0.0, 1.0),
            doc="Relative weight of removing a confirmation or a branch",
        ),
        "max_branches": ParamSpec(
            kind="int", default=4, bounds=(1, 8),
            doc="Ceiling on branches; each one costs a full detection pass",
        ),
        "max_confirmations": ParamSpec(
            kind="int", default=4, bounds=(0, 8),
            doc="Ceiling on confirmations per branch (all must pass, AND)",
        ),
    },
    rationale=(
        "Topology search is the only move that can reach a strategy the seed's "
        "shape cannot express — the gap KNOWN-LIMITATIONS §0c records as never "
        "searched: v0.2.0's only sweep moved exit management, and the "
        "entry/feature space was never explored at all. Legal moves are ENUMERATED "
        "before any randomness is spent, and the set deliberately excludes adding "
        "or removing a policy or the >=1:2-after-costs filter: a mutation that "
        "deleted the filter would be evolution discovering that the PRD's rules "
        "are optional."
    ),
)
def graph_edit(graph, rng, *, p_swap: float, p_add: float, p_remove: float,
               max_branches: int, max_confirmations: int):
    """Return a NEW graph one topology edit away from `graph`.

    Args:
        graph: The parent StrategyGraph. Never mutated.
        rng: A seeded random.Random.
        p_swap / p_add / p_remove: Relative weights over EDIT_TYPES. All-zero
            falls back to uniform rather than raising — a weight vector is a
            preference, not a legality statement.
        max_branches / max_confirmations: Structural ceilings.

    Returns:
        (StrategyGraph, list[dict]) — the child and a one-entry diff describing
        the edit that was applied.

    Raises:
        GraphError: If the graph admits no legal edit at all, or if every legal
            candidate produces an invalid graph. Never a silent no-op: that would
            charge a trial for a duplicate.
    """
    payload = graph.to_dict()
    edits = _legal_edits(
        payload, max_branches=int(max_branches),
        max_confirmations=int(max_confirmations),
    )
    if not edits:
        raise GraphError(
            f"graph {graph.name!r} admits no legal topology edit "
            f"(branches={len(payload['branches'])}, "
            f"registered detectors={len(registry.by_kind('detector'))}); "
            f"graph-edit refuses to return the parent unchanged"
        )

    weights = {"swap": float(p_swap), "add": float(p_add), "remove": float(p_remove)}
    available = [t for t in EDIT_TYPES if any(e[0] == t for e in edits)]
    w = [weights[t] for t in available]
    if sum(w) <= 0:
        w = [1.0] * len(available)
    chosen = rng.choices(available, weights=w, k=1)[0]

    # Ordered candidate list: the chosen type first, then the others in EDIT_TYPES
    # order. Each group shuffled with the member's own rng, so the whole sequence
    # is reproducible from the seed and a rejected candidate does not change what
    # a later member draws.
    ordered: list[tuple[str, dict, object]] = []
    for t in [chosen] + [x for x in available if x != chosen]:
        group = [e for e in edits if e[0] == t]
        rng.shuffle(group)
        ordered.extend(group)

    for _type, description, apply_fn in ordered:
        trial = copy.deepcopy(payload)
        apply_fn(trial)
        try:
            child = StrategyGraph.from_dict(trial)
            validate(child)
        except FrameworkError as exc:
            logger.debug("graph-edit: %s rejected as illegal (%s)", description, exc)
            continue
        return child, [description]

    raise GraphError(
        f"graph {graph.name!r}: all {len(ordered)} enumerated topology edits "
        f"produce an invalid graph. This is a MUTATOR BUG, not a candidate that "
        f"happens to take no trades — raising so it aborts the campaign instead of "
        f"scoring as a plausible tier D."
    )
