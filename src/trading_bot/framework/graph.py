"""
StrategyGraph: the serializable, content-hashed description of a strategy.

The graph is a TYPED-SLOT PIPELINE, not a free-form DAG:

    DataSource -> RegimeGate -> Branch(Detector -> Confirmations -> Policy
                                       -> ExitPolicySpec) -> Filters

A general node/edge DAG would make validate() a graph-theory exercise and every
no-lookahead guarantee negotiable. Three of this phase's stated assumptions are
what make the typed slots sufficient, and they are reproduced here so they
survive without the plan:

A2 — THE REGIME CLASSIFIER IS A GRAPH-LEVEL FIELD (RegimeGate), NOT A NODE, and
branch activation is a `Branch.regimes` tuple. Three reasons in order of weight:
(i) the classifier is the one measured-healthy layer and its thresholds are
never swept (contract §1, walkforward.py:26-32) — a plug-in node with ParamSpec
bounds is an open invitation for Phase 6's param-jitter mutator to sweep them,
whereas a graph-level dataclass with a NOT-SWEEPABLE banner is a structural
refusal; (ii) engine.run_backtest computes classify_series once per run over the
whole regime frame and memoizes it (engine.py:279-289) — a per-branch node would
either recompute or need its own cache seam for no benefit; (iii) WHICH regime
labels enable a branch is routing, not computation, and Branch.regimes reads
exactly like scan.py:55-61's dispatch table, which is the drift surface we most
want legible.

A4 — EXITS ARE A DECLARATIVE PER-BRANCH ExitPolicySpec, NOT A PLUG-IN KIND. This
is what makes engine.py's MANDATORY DEVIATION (engine.py:392-411, 523-530)
expressible: Donchian trades take trail + opposite-channel exits, fade trades
keep frozen stop/target/time/end behavior. Making exits a plug-in would put a
mutable stop behind a Protocol boundary — the single highest-risk lookahead
surface in v0.2.0, whose guard is the ratchet ordering at engine.py:435-452 —
and there is exactly one exit implementation. A frozen dataclass of flags keeps
the ratchet ordering inside the executor where its comment lives, and ONE
generic exit block reproduces BOTH legacy branches from these flags with no
special-casing. If a special case ever seems necessary, the graph model is
wrong; fix the model, not the executor.

A5 — THE CONTENT HASH COVERS *RESOLVED* PARAMETERS, EXCLUDES `name` AND `meta`,
CANONICALIZES COLLECTION ORDER — AND THE EXECUTOR ITERATES IN THAT SAME
CANONICAL ORDER. Phase 5's strategy_versions and Phase 6's trial_ledger both key
on graph_hash (contract §6), so two graphs that hash equal must BEHAVE
identically and two graphs that behave identically must hash equal.
ordered_branches() — sorted by id — is the only iteration order the executor
uses, which closes that loop: without it, two hash-equal graphs could break a
rank_signals tie differently.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from trading_bot import config
from trading_bot.data import storage
from trading_bot.framework import registry
from trading_bot.framework.errors import GraphError
from trading_bot.regime.classifier import REGIMES

# Bump ONLY on a breaking layout change, and add a migration in from_dict when
# you do: every persisted graph, every strategy_versions row and every
# trial_ledger row keys on a hash that INCLUDES this number, so a bump
# repartitions all of them.
SCHEMA_VERSION = 1

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ANY_REGIME = "any"
LEGAL_REGIMES = tuple(REGIMES) + (ANY_REGIME,)

# Which registry kind each graph slot must hold.
_SLOT_KINDS = {
    "data": "data",
    "detector": "detector",
    "confirmations": "confirmation",
    "policy": "policy",
    "filters": "filter",
}


def _reject_list(value, *, where: str) -> None:
    """Collections in a graph must be tuples, never lists.

    Phase 6 mutates graphs with dataclasses.replace over comprehensions; a list
    makes the graph order-unstable and unhashable-by-convention, and would let
    two behaviourally distinct graphs compare equal or vice versa.
    """
    if isinstance(value, list):
        raise GraphError(
            f"{where} must be a TUPLE, not a list — graph collections are frozen "
            f"so dataclasses.replace() in Phase 6 always produces a valid, "
            f"order-stable graph"
        )


@dataclass(frozen=True)
class NodeSpec:
    """One plug-in reference inside a graph.

    Attributes:
        id: Unique within the whole graph; [a-z0-9][a-z0-9_-]*.
        key: Registry key, e.g. "detector.donchian-breakout".
        params: OVERRIDES ONLY. Anything omitted resolves to the plug-in's
            ParamSpec default, and omitting a parameter hashes identically to
            setting it to that default (A5).
    """

    id: str
    key: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _reject_list(self.params, where=f"node {self.id!r} params")


@dataclass(frozen=True)
class RegimeGate:
    """A2: the regime classifier is a graph-level FIELD, not a node.

    NOT SWEEPABLE. These are the one measured-healthy layer's thresholds
    (walkforward.py:26-32 keeps them out of every grid); they live here as a
    plain dataclass rather than a plug-in with ParamSpec bounds precisely so
    Phase 6's param-jitter mutator has no legal handle on them.
    `enabled=False` labels every bar "any" and makes every branch eligible —
    for a regime-free strategy, not for escaping the gate.

    adx_period and atr_percentile_window are deliberately ABSENT: they come from
    config, exactly as engine.run_backtest leaves them (engine.py:284-288).
    """

    timeframe: str = config.REGIME_TIMEFRAME
    adx_trend_threshold: float = config.ADX_TREND_THRESHOLD
    atr_extreme_percentile: float = config.ATR_EXTREME_PERCENTILE
    enabled: bool = True


@dataclass(frozen=True)
class TriggerSpec:
    """A3: check_breakout is executor machinery, not a plug-in kind.

    Only these three numbers are graph-visible; the crossing test, the freshness
    pair (breakout.py:124-129), the `ts < end_ts` skip (breakout.py:113-114) and
    the interval_ms contiguity check (breakout.py:115-122) are NOT expressible in
    a graph. Making the trigger swappable per branch would let a mutated graph
    disable the freshness test or the gap check and produce a flattering backtest
    that still validates.
    """

    lookback_bars: int = 1  # 1 = the bar under evaluation only
    volume_lookback: int = config.VOLUME_LOOKBACK
    volume_high_ratio: float = config.VOLUME_HIGH_RATIO


@dataclass(frozen=True)
class ExitPolicySpec:
    """A4: exits are declarative flags, per branch.

    Attributes:
        stop: Must be True; see validate() rule 9.
        target_enabled: Whether the measured-move target is an EXIT (the R:R
            screen inside the policy uses it either way).
        trail_enabled: Whether the ATR ratchet trail is active.
        trail_atr_multiple: Trail distance in ATRs. Only read when trailing.
        trail_atr_period: ATR period for the trail. validate() requires it to
            equal the policy's atr_period, because engine.py:537 uses ONE
            atr_value for both the entry stop and the trail.
        channel_exit: Whether a touch of the trailing opposite channel exits.
        channel_period: Bars in that channel. validate() requires it to equal
            the detector's entry_period — "the same 20 bars, other side"
            (donchian.py:146-152).
        max_hold_bars: None => run_graph_backtest's value.
    """

    stop: bool = True
    target_enabled: bool = False
    trail_enabled: bool = False
    trail_atr_multiple: float = config.TRAIL_ATR_MULTIPLE
    trail_atr_period: int = config.ATR_STOP_PERIOD
    channel_exit: bool = False
    channel_period: int = config.DONCHIAN_ENTRY_PERIOD
    max_hold_bars: int | None = None


@dataclass(frozen=True)
class Branch:
    """One detector and everything downstream of it.

    `regimes` is the routing rule, and it reads like scan.py:55-61's dispatch
    table on purpose (A2/A6) — that is the surface most at risk of drifting from
    the live path. ("any",) means every label activates the branch.
    """

    id: str
    detector: NodeSpec
    policy: NodeSpec
    regimes: tuple[str, ...] = (ANY_REGIME,)
    confirmations: tuple[NodeSpec, ...] = ()  # ALL must pass (AND)
    exits: ExitPolicySpec = field(default_factory=ExitPolicySpec)
    enabled: bool = True

    def __post_init__(self) -> None:
        _reject_list(self.regimes, where=f"branch {self.id!r} regimes")
        _reject_list(self.confirmations, where=f"branch {self.id!r} confirmations")


@dataclass(frozen=True)
class StrategyGraph:
    """A complete, serializable strategy."""

    name: str
    data: NodeSpec
    branches: tuple[Branch, ...]
    regime: RegimeGate = field(default_factory=RegimeGate)
    trigger: TriggerSpec = field(default_factory=TriggerSpec)
    filters: tuple[NodeSpec, ...] = ()  # ALL must accept
    meta: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _reject_list(self.branches, where="graph branches")
        _reject_list(self.filters, where="graph filters")
        _reject_list(self.meta, where="graph meta")

    def ordered_branches(self) -> tuple[Branch, ...]:
        """Branches sorted by id — the ONLY iteration order the executor uses.

        A5: canonical order is also runtime order. Without this, two graphs that
        hash identically could break a rank_signals tie differently and produce
        different trades under the same hash, which would corrupt Phase 5's
        version registry and Phase 6's ledger at the root.
        """
        return tuple(sorted(self.branches, key=lambda b: b.id))

    def resolved(self) -> "StrategyGraph":
        """Every node's params materialized from registry defaults.

        graph_hash(g) == graph_hash(g.resolved()) by construction.
        """

        def rn(node: NodeSpec) -> NodeSpec:
            return replace(node, params=registry.get(node.key).resolve(node.params))

        return replace(
            self,
            data=rn(self.data),
            branches=tuple(
                replace(
                    b,
                    detector=rn(b.detector),
                    policy=rn(b.policy),
                    confirmations=tuple(rn(c) for c in b.confirmations),
                )
                for b in self.branches
            ),
            filters=tuple(rn(f) for f in self.filters),
        )

    def to_dict(self) -> dict:
        """Full, lossless JSON-ready payload (includes name, meta, schema_version)."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "data": _node_to_dict(self.data),
            "regime": {
                "timeframe": self.regime.timeframe,
                "adx_trend_threshold": self.regime.adx_trend_threshold,
                "atr_extreme_percentile": self.regime.atr_extreme_percentile,
                "enabled": self.regime.enabled,
            },
            "trigger": {
                "lookback_bars": self.trigger.lookback_bars,
                "volume_lookback": self.trigger.volume_lookback,
                "volume_high_ratio": self.trigger.volume_high_ratio,
            },
            "branches": [
                {
                    "id": b.id,
                    "detector": _node_to_dict(b.detector),
                    "policy": _node_to_dict(b.policy),
                    "regimes": list(b.regimes),
                    "confirmations": [_node_to_dict(c) for c in b.confirmations],
                    "exits": {
                        "stop": b.exits.stop,
                        "target_enabled": b.exits.target_enabled,
                        "trail_enabled": b.exits.trail_enabled,
                        "trail_atr_multiple": b.exits.trail_atr_multiple,
                        "trail_atr_period": b.exits.trail_atr_period,
                        "channel_exit": b.exits.channel_exit,
                        "channel_period": b.exits.channel_period,
                        "max_hold_bars": b.exits.max_hold_bars,
                    },
                    "enabled": b.enabled,
                }
                for b in self.branches
            ],
            "filters": [_node_to_dict(f) for f in self.filters],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "StrategyGraph":
        """Parse a to_dict() payload.

        Unknown keys are REJECTED at every level, never ignored: a UI or mutator
        writing {"exits": {"trail": true}} (wrong name) would otherwise silently
        get trail_enabled=False and the operator would believe a trail was tested.

        Raises:
            GraphError: On an unknown or missing key, or a wrongly-typed section.
        """
        _check_keys(
            payload,
            {
                "schema_version",
                "name",
                "data",
                "regime",
                "trigger",
                "branches",
                "filters",
                "meta",
            },
            required={"name", "data", "branches"},
            where="graph",
        )
        regime_raw = payload.get("regime") or {}
        _check_keys(
            regime_raw,
            {"timeframe", "adx_trend_threshold", "atr_extreme_percentile", "enabled"},
            required=set(),
            where="graph.regime",
        )
        trigger_raw = payload.get("trigger") or {}
        _check_keys(
            trigger_raw,
            {"lookback_bars", "volume_lookback", "volume_high_ratio"},
            required=set(),
            where="graph.trigger",
        )
        branches_raw = payload["branches"]
        if not isinstance(branches_raw, (list, tuple)):
            raise GraphError("graph.branches must be a list")

        branches = []
        for i, b in enumerate(branches_raw):
            _check_keys(
                b,
                {
                    "id",
                    "detector",
                    "policy",
                    "regimes",
                    "confirmations",
                    "exits",
                    "enabled",
                },
                required={"id", "detector", "policy"},
                where=f"graph.branches[{i}]",
            )
            exits_raw = b.get("exits") or {}
            _check_keys(
                exits_raw,
                {
                    "stop",
                    "target_enabled",
                    "trail_enabled",
                    "trail_atr_multiple",
                    "trail_atr_period",
                    "channel_exit",
                    "channel_period",
                    "max_hold_bars",
                },
                required=set(),
                where=f"graph.branches[{i}].exits",
            )
            branches.append(
                Branch(
                    id=b["id"],
                    detector=_node_from_dict(b["detector"], where=f"graph.branches[{i}].detector"),
                    policy=_node_from_dict(b["policy"], where=f"graph.branches[{i}].policy"),
                    regimes=tuple(b.get("regimes", (ANY_REGIME,))),
                    confirmations=tuple(
                        _node_from_dict(c, where=f"graph.branches[{i}].confirmations[{k}]")
                        for k, c in enumerate(b.get("confirmations", ()))
                    ),
                    exits=ExitPolicySpec(**exits_raw),
                    enabled=bool(b.get("enabled", True)),
                )
            )

        return cls(
            name=payload["name"],
            data=_node_from_dict(payload["data"], where="graph.data"),
            branches=tuple(branches),
            regime=RegimeGate(**regime_raw),
            trigger=TriggerSpec(**trigger_raw),
            filters=tuple(
                _node_from_dict(f, where=f"graph.filters[{i}]")
                for i, f in enumerate(payload.get("filters", ()))
            ),
            meta=dict(payload.get("meta", {})),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )


def _node_to_dict(node: NodeSpec) -> dict:
    return {"id": node.id, "key": node.key, "params": dict(node.params)}


def _node_from_dict(payload: Mapping, *, where: str) -> NodeSpec:
    _check_keys(payload, {"id", "key", "params"}, required={"id", "key"}, where=where)
    params = payload.get("params", {})
    if not isinstance(params, Mapping):
        raise GraphError(f"{where}.params must be an object, got {type(params).__name__}")
    return NodeSpec(id=payload["id"], key=payload["key"], params=dict(params))


def _check_keys(payload, allowed: set, *, required: set, where: str) -> None:
    if not isinstance(payload, Mapping):
        raise GraphError(f"{where} must be an object, got {type(payload).__name__}")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise GraphError(
            f"{where}: unknown key(s) {unknown}; legal keys are {sorted(allowed)}. "
            f"Unknown keys are rejected rather than ignored — a misspelled flag "
            f"('trail' for 'trail_enabled') would otherwise read as tested when it "
            f"silently kept the default."
        )
    missing = sorted(required - set(payload))
    if missing:
        raise GraphError(f"{where}: missing required key(s) {missing}")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(graph: StrategyGraph) -> None:
    """Check a graph against every structural rule, reporting ALL problems.

    Collects every problem and raises ONE GraphError listing them: a builder UI
    showing one error at a time is a bad UI, and a mutator that produced three
    illegal params should learn all three.

    Raises:
        GraphError: listing every violated rule.
    """
    problems: list[str] = []

    # 1 — schema version
    if graph.schema_version != SCHEMA_VERSION:
        problems.append(
            f"schema_version is {graph.schema_version} but this build speaks "
            f"{SCHEMA_VERSION}; regenerate or migrate the graph"
        )

    # 2 — name
    if not isinstance(graph.name, str) or not _NAME_RE.match(graph.name or ""):
        problems.append(
            f"name {graph.name!r} must match {_NAME_RE.pattern} — it becomes "
            f"{config.STRATEGY_DIR}/<name>.strategy.json"
        )

    # 3/4 — slot/kind agreement and registry membership
    def check_node(node: NodeSpec, slot: str) -> registry.PluginSpec | None:
        kind = _SLOT_KINDS[slot]
        if not node.key.startswith(kind + "."):
            problems.append(
                f"slot {slot!r} (node {node.id!r}) holds key {node.key!r}; it must "
                f"name a {kind!r} plug-in, i.e. start with {kind + '.'!r}"
            )
            return None
        try:
            return registry.get(node.key)
        except Exception as exc:  # noqa: BLE001 — collected, not raised
            problems.append(f"node {node.id!r}: {exc}")
            return None

    specs: dict[str, registry.PluginSpec | None] = {}
    specs[graph.data.id] = check_node(graph.data, "data")
    for b in graph.branches:
        specs[b.detector.id] = check_node(b.detector, "detector")
        specs[b.policy.id] = check_node(b.policy, "policy")
        for c in b.confirmations:
            specs[c.id] = check_node(c, "confirmations")
    for f in graph.filters:
        specs[f.id] = check_node(f, "filters")

    # 5 — unique ids
    owners: dict[str, str] = {}
    for owner, node in _iter_nodes(graph):
        if not _ID_RE.match(node.id or ""):
            problems.append(f"node id {node.id!r} must match {_ID_RE.pattern}")
        if node.id in owners:
            problems.append(
                f"duplicate node id {node.id!r}: claimed by both {owners[node.id]} "
                f"and {owner}"
            )
        else:
            owners[node.id] = owner
    seen_branches: dict[str, int] = {}
    for i, b in enumerate(graph.branches):
        if not _ID_RE.match(b.id or ""):
            problems.append(f"branch id {b.id!r} must match {_ID_RE.pattern}")
        if b.id in seen_branches:
            problems.append(
                f"duplicate branch id {b.id!r}: branches[{seen_branches[b.id]}] and "
                f"branches[{i}]"
            )
        else:
            seen_branches[b.id] = i

    # 6 — at least one enabled branch
    if not graph.branches:
        problems.append(
            "graph has no branches; a graph with no enabled branch produces no "
            "trades and would read as a strategy with no edge"
        )
    elif not any(b.enabled for b in graph.branches):
        problems.append(
            "graph has no ENABLED branch; a graph with no enabled branch produces "
            "no trades and would read as a strategy with no edge"
        )

    # 7 — legal regime labels
    for b in graph.branches:
        for label in b.regimes:
            if label not in LEGAL_REGIMES:
                problems.append(
                    f"branch {b.id!r}: regime label {label!r} is not one of "
                    f"{LEGAL_REGIMES}"
                )

    # 8 — parameters resolve
    resolved: dict[str, dict] = {}
    for _owner, node in _iter_nodes(graph):
        spec = specs.get(node.id)
        if spec is None:
            continue
        try:
            resolved[node.id] = spec.resolve(node.params)
        except Exception as exc:  # noqa: BLE001 — collected, not raised
            problems.append(f"node {node.id!r}: {exc}")

    # 9-12 — exit policy coherence
    for b in graph.branches:
        ex = b.exits
        if ex.stop is not True:
            problems.append(
                f"branch {b.id!r}: exits.stop must be True — a stopless strategy is "
                f"not expressible in v0.3.0; the PositionPlan's stop is mandatory"
            )
        if ex.trail_enabled and not (ex.trail_atr_multiple > 0):
            problems.append(
                f"branch {b.id!r}: exits.trail_atr_multiple is "
                f"{ex.trail_atr_multiple!r}; it must be > 0 when trail_enabled"
            )
        det = resolved.get(b.detector.id, {})
        pol = resolved.get(b.policy.id, {})
        if ex.channel_exit and "entry_period" in det:
            if ex.channel_period != det["entry_period"]:
                problems.append(
                    f"branch {b.id!r}: exits.channel_period={ex.channel_period} but "
                    f"detector entry_period={det['entry_period']}. The exit channel "
                    f"and the entry channel are the same bars, other side "
                    f"(donchian.py:146-152); letting them diverge silently changes "
                    f"the exit without changing the entry"
                )
        if ex.trail_enabled and "atr_period" in pol:
            if ex.trail_atr_period != pol["atr_period"]:
                problems.append(
                    f"branch {b.id!r}: exits.trail_atr_period={ex.trail_atr_period} "
                    f"but policy atr_period={pol['atr_period']}. engine.py uses ONE "
                    f"atr_value for both the entry stop and the trail "
                    f"(engine.py:537)"
                )

    # 13 — trigger spec
    t = graph.trigger
    if not (isinstance(t.lookback_bars, int) and t.lookback_bars >= 1):
        problems.append(
            f"trigger.lookback_bars is {t.lookback_bars!r}; it must be an int >= 1"
        )
    if not (isinstance(t.volume_lookback, int) and t.volume_lookback >= 1):
        problems.append(
            f"trigger.volume_lookback is {t.volume_lookback!r}; it must be an int >= 1"
        )
    if not (t.volume_high_ratio > 0):
        problems.append(
            f"trigger.volume_high_ratio is {t.volume_high_ratio!r}; it must be > 0"
        )

    # 14 — regime timeframe
    if graph.regime.timeframe not in storage.TIMEFRAME_MS:
        problems.append(
            f"regime.timeframe {graph.regime.timeframe!r} is not a stored timeframe; "
            f"legal keys are {tuple(storage.TIMEFRAME_MS)}"
        )

    if problems:
        raise GraphError(
            f"strategy graph {graph.name!r} has {len(problems)} problem(s):\n  - "
            + "\n  - ".join(problems)
        )


def _iter_nodes(graph: StrategyGraph):
    """(owner-description, NodeSpec) for every node in the graph."""
    yield ("graph.data", graph.data)
    for b in graph.branches:
        yield (f"branch {b.id!r} detector", b.detector)
        yield (f"branch {b.id!r} policy", b.policy)
        for c in b.confirmations:
            yield (f"branch {b.id!r} confirmation", c)
    for f in graph.filters:
        yield ("graph.filters", f)


# --------------------------------------------------------------------------- #
# Content hashing
# --------------------------------------------------------------------------- #


def _norm(value):
    """Normalize a JSON scalar for hashing.

    floats become float(v) with -0.0 folded to 0.0, so 20 and 20.0 cannot
    produce two hashes for one strategy. bool is checked BEFORE int because
    bool is a subclass of int in Python.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        return 0.0 if value == 0.0 else float(value)
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return {k: _norm(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return value


def canonical_dict(graph: StrategyGraph) -> dict:
    """Order-independent, default-resolved dict for hashing.

    A5, restated as rules:
      - params are RESOLVED against registry defaults first, so omitting a
        parameter and setting it to its default hash identically;
      - `name` and `meta` are EXCLUDED — a rename or a note is not a new
        strategy, and Phase 5's version registry must not fork on one;
      - `branches`, each branch's `confirmations`, and `filters` are sorted by
        id; `regimes` is sorted; every dict is emitted with sort_keys;
      - `schema_version` IS included: a schema change changes meaning;
      - floats are normalized (float(v), and -0.0 -> 0.0);
      - NaN/Inf never reach here: ParamSpec.check rejects non-finite values, and
        graph_hash dumps with allow_nan=False as a second line of defence.
    """
    g = graph.resolved()
    payload = g.to_dict()
    payload.pop("name", None)
    payload.pop("meta", None)
    for b in payload["branches"]:
        b["regimes"] = sorted(b["regimes"])
        b["confirmations"] = sorted(b["confirmations"], key=lambda n: n["id"])
    payload["branches"] = sorted(payload["branches"], key=lambda b: b["id"])
    payload["filters"] = sorted(payload["filters"], key=lambda n: n["id"])
    return _norm(payload)


def graph_hash(graph: StrategyGraph) -> str:
    """sha256 hex digest (64 chars) of canonical_dict's JSON.

    Phase 5's strategy_versions and Phase 6's trial_ledger both key on this
    (contract §6). Requires the referenced plug-ins to be importable, because
    resolving defaults needs the registry — call registry.load_all() first.
    """
    payload = json.dumps(
        canonical_dict(graph),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short_hash(graph: StrategyGraph) -> str:
    """First 12 hex chars of graph_hash — for CLI output and log lines only,
    never a persisted key."""
    return graph_hash(graph)[:12]


def branch_hash(graph: StrategyGraph, branch: Branch) -> str:
    """Content hash of ONE branch's resolved detector node.

    The candidate memo's key (framework.context.EvalSession.candidates) so that
    graphs sharing a detector configuration share its candidates — the whole
    performance argument for Phase 6's population.
    """
    spec = registry.get(branch.detector.key)
    payload = json.dumps(
        _norm(
            {
                "schema_version": graph.schema_version,
                "key": branch.detector.key,
                "params": spec.resolve(branch.detector.params),
                "regimes": sorted(branch.regimes),
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def path_for(name: str) -> Path:
    """config.STRATEGY_DIR / "<name>.strategy.json"."""
    return Path(config.STRATEGY_DIR) / f"{name}.strategy.json"


def save(graph: StrategyGraph, path: str | Path | None = None) -> Path:
    """Validate, then write the graph as JSON.

    Validation happens FIRST, so an invalid graph writes no file at all.
    JSON is emitted with indent=2, sort_keys=True and a trailing newline so
    committed graphs diff cleanly.

    Args:
        graph: The graph to persist.
        path: Destination. A bare name resolves to path_for(graph.name).

    Returns:
        The Path written.
    """
    validate(graph)
    target = path_for(graph.name) if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(graph.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(text, encoding="utf-8")
    return target


def load(path: str | Path) -> StrategyGraph:
    """Parse and validate a serialized graph.

    Raises:
        GraphError: On unreadable/corrupt JSON as well as on any validation
            failure, so a caller needs one except clause rather than two.
    """
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GraphError(f"no strategy graph at {p}") from exc
    except json.JSONDecodeError as exc:
        raise GraphError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GraphError(f"{p} must contain a JSON object, got {type(payload).__name__}")
    graph = StrategyGraph.from_dict(payload)
    validate(graph)
    return graph
