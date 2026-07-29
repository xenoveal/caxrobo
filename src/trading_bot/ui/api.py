"""
Pure request -> JSON surface for the builder UI (v0.3.0 Phase 7).

NO `http` or `socket` IMPORT ANYWHERE IN THIS MODULE. `server.py` is the only
place a socket is opened; this module is tested by calling `handle()` (or the
individual functions) directly, with no port bound (contract §8).

THE LOAD-BEARING IDEA: the UI is a view and a launcher, never a store.
Strategies live in Phase 3's serialization (data/strategies/*.strategy.json).
Verdicts live in Phase 1's gate (backtest/walkforward.py). Progress lives in
Phase 6's state.db tables. Trials live in Phase 1's ledger
(backtest/trials.py). Reviews live in Phase 5's records. The UI's only
private state is the run-bookkeeping directory under RUN_ROOT — a pid, an
argv, a status and a log — and deleting it loses nothing of value.

THE GATE IS THE ONLY FITNESS ORACLE (contract §4). The one evaluation this
module can trigger that produces a gate verdict — a "gate" run — routes
through `walkforward.walk_forward_pooled(..., strategy=graph, ledger=...)`,
the exact function `cli.py walkforward --graph` calls, with a
`backtest.trials.TrialLedger` passed in so every evaluation increments the
persistent ledger. A "backtest" run calls `framework.execute.run_graph_backtest`
directly (the same function `cli.py graph-backtest` uses) and touches no
ledger, because it is explicitly the same in-sample diagnostic that command
is — never presented as a gate verdict.
"""

import dataclasses
import json
import logging
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

from trading_bot import config
from trading_bot.backtest import trials
from trading_bot.backtest.benchmark import buy_and_hold  # noqa: F401 (re-export site for tests)
from trading_bot.backtest.equity import compute_equity_metrics, daily_returns
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.backtest.walkforward import (
    DEFAULT_GRID,
    GATE_CONDITIONS,
    GATE_MAX_DRAWDOWN,
    GATE_MIN_DSR,
    GATE_MIN_SHARPE,
    walk_forward_pooled,
)
from trading_bot.data import storage
from trading_bot.data.statestore import connect as connect_state
from trading_bot.evolution import population as evo_population
from trading_bot.evolution import runner as evo_runner
from trading_bot.feedback import records as feedback_records
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.errors import FrameworkError, GraphError, RegistryError
from trading_bot.framework.execute import run_graph_backtest
from trading_bot.framework.graph import Branch, NodeSpec, StrategyGraph

logger = logging.getLogger("trading_bot")

# --------------------------------------------------------------------------- #
# Module constants (Task 2). Only UI_HOST/UI_PORT are reserved in config.py
# (contract §7) — everything the UI itself tunes lives here or in server.py.
# --------------------------------------------------------------------------- #

RUN_ROOT = Path("data/ui_runs")  # monkeypatched in tests
MAX_TIER_A_JOBS = 1  # a 2nd concurrent gate/backtest run -> 429
MAX_BODY_BYTES = 1 << 20
RUN_ID_RE = re.compile(r"^[0-9]{10,20}-[0-9a-f]{6}$")  # <ms>-<rand hex>, sortable
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")  # strategy + campaign ids

_STRATEGY_NAME_GROUP = r"[a-z0-9][a-z0-9._-]{0,63}"
_RUN_ID_GROUP = r"[0-9]{10,20}-[0-9a-f]{6}"

_STATE_LOCK = threading.Lock()
_STATE = {"active_tier_a": 0}
_THREADS: dict[str, threading.Thread] = {}
_ORIGIN_PORT = {"port": config.UI_PORT}


@dataclass(frozen=True)
class ApiResponse:
    """One HTTP-agnostic response: a status code, a JSON-able payload, and
    optional extra headers (only `Allow` on a 405 today)."""

    status: int
    payload: dict
    headers: dict = field(default_factory=dict)


def _ok(payload: dict, status: int = 200) -> ApiResponse:
    return ApiResponse(status, payload, {})


def _err(status: int, message: str, headers: dict | None = None) -> ApiResponse:
    return ApiResponse(status, {"error": message}, headers or {})


def _jsonsafe(obj):
    """Recursively make ``obj`` valid JSON: dataclasses -> dict, tuples ->
    lists, NaN/Inf -> None.

    json.dumps emits bare NaN by default, which is not valid JSON — JSON.parse
    throws in the browser and the panel goes blank with no error. Handled here,
    at the one seam every serializer passes through, rather than hoped away.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonsafe(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonsafe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def set_origin_port(port: int) -> None:
    """Called once by server.serve(); a test that never calls this still gets
    the config.UI_PORT default, so the 403 Origin check is always assertable."""
    _ORIGIN_PORT["port"] = int(port)


def reset_jobs() -> None:
    """Test isolation hook: join every tracked job thread and zero the Tier A
    counter. A leaked job thread poisons every test after it."""
    with _STATE_LOCK:
        threads = list(_THREADS.values())
        _THREADS.clear()
        _STATE["active_tier_a"] = 0
    for t in threads:
        t.join(timeout=5)


# --------------------------------------------------------------------------- #
# Task 3: read-only surface — /api/plugins, /api/data/coverage, /api/config
# --------------------------------------------------------------------------- #


def _paramspec_to_dict(ps) -> dict:
    """ParamSpec -> an HTML-control description.

    The SAME declaration Phase 6 jitters within (contract §3), so a control
    can never offer a value the mutator considers illegal. `step` IS a
    ParamSpec field (framework/contracts.py) — verified before writing this,
    per Task 3's own instruction to check field names first — so it is read
    from the spec, never re-derived.
    """
    d = {"kind": ps.kind, "default": ps.default, "doc": ps.doc}
    if ps.choices is not None:
        d["choices"] = list(ps.choices)
    if ps.bounds is not None:
        lo, hi = ps.bounds
        d["min"] = lo
        d["max"] = hi
        d["step"] = ps.step if ps.step is not None else max((hi - lo) / 100.0, 1e-6)
    return d


def get_plugins() -> ApiResponse:
    """Every registered plug-in, grouped by kind, with its rationale and
    ParamSpec-derived controls. Import errors during load_all() are FATAL by
    contract §3 -> 500, never a silently short list."""
    try:
        registry.load_all()
    except FrameworkError as exc:
        return _err(500, str(exc))

    by_kind: dict[str, list[dict]] = {k: [] for k in registry.KINDS}
    for spec in sorted(registry.REGISTRY.values(), key=lambda s: (s.kind, s.name)):
        by_kind[spec.kind].append(
            {
                "key": spec.key,
                "name": spec.name,
                "kind": spec.kind,
                "tier": spec.tier,
                "timeframes": list(spec.timeframes),
                "rationale": spec.rationale,
                "dof": spec.combo_count(),
                "params": {n: _paramspec_to_dict(p) for n, p in spec.params.items()},
            }
        )
    return _ok({"kinds": list(registry.KINDS), "plugins": by_kind, "n": len(registry.REGISTRY)})


def _stored_symbols() -> list[str]:
    conn = storage.connect(config.DB_PATH)
    try:
        rows = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def get_data_coverage() -> ApiResponse:
    """Per (symbol, timeframe): bar count, first/last ts, gap count.

    Read-only: lane 1 REPORTS, it does not fetch. Backfill stays a CLI job.
    """
    conn = storage.connect(config.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT symbol, timeframe, COUNT(*), MIN(ts), MAX(ts) "
            "FROM ohlcv GROUP BY symbol, timeframe ORDER BY symbol, timeframe"
        ).fetchall()
        out = []
        for symbol, tf, n, first_ts, last_ts in rows:
            try:
                gaps = storage.find_gaps(conn, symbol, tf)
                n_gaps = len(gaps)
            except Exception as exc:  # noqa: BLE001 — a report must not 500 on one bad row
                logger.warning("coverage: gap check failed for %s/%s: %s", symbol, tf, exc)
                n_gaps = None
            out.append(
                {
                    "symbol": symbol,
                    "timeframe": tf,
                    "bars": n,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                    "gaps": n_gaps,
                }
            )
    finally:
        conn.close()
    return _ok({"coverage": out})


def _gate_thresholds() -> dict:
    """Every gate condition's threshold, so app.js never hardcodes a number
    that could go stale silently (Task 3 GOTCHA #1)."""
    return {
        "sample_adequacy": {"op": ">=", "value": config.WF_MIN_TRADES, "unit": "trades"},
        "sharpe": {"op": ">=", "value": GATE_MIN_SHARPE, "unit": ""},
        "dsr": {"op": ">", "value": GATE_MIN_DSR, "unit": ""},
        "max_drawdown": {"op": "<=", "value": GATE_MAX_DRAWDOWN, "unit": "fraction"},
        "per_symbol_expectancy": {"op": "all > 0", "value": None, "unit": "AND across symbols"},
        "beats_benchmark_return": {"op": "> basket", "value": None, "unit": "ann_return_pct"},
        "beats_benchmark_sharpe": {"op": "> basket", "value": None, "unit": "sharpe"},
    }


def get_ui_config() -> ApiResponse:
    """Everything app.js needs to render lane 1 without hardcoding a symbol,
    timeframe, gate threshold, or path."""
    try:
        stored = _stored_symbols()
    except Exception as exc:  # noqa: BLE001 — an unreadable ohlcv.db must not 500 the whole page
        logger.warning("get_ui_config: could not read stored symbols: %s", exc)
        stored = []
    return _ok(
        {
            "symbols": list(config.SYMBOLS),
            "research_symbols": list(config.RESEARCH_SYMBOLS),
            "stored_symbols": stored,
            "timeframes": list(config.TIMEFRAMES),
            "backfill_start": config.BACKFILL_START,
            "strategy_dir": config.STRATEGY_DIR,
            "state_db_path": config.STATE_DB_PATH,
            "gate_conditions": list(GATE_CONDITIONS),
            "gate_thresholds": _gate_thresholds(),
            "schema_version": fgraph.SCHEMA_VERSION,
            "max_tier_a_jobs": MAX_TIER_A_JOBS,
            "ui_host": config.UI_HOST,
            "ui_port": config.UI_PORT,
        }
    )


# --------------------------------------------------------------------------- #
# Task 4: strategy read/write over the Phase 3 serialization
# --------------------------------------------------------------------------- #


def _unique_branch_id(base: str, taken: set[str]) -> str:
    """A graph-unique branch id derived from `base`, matching
    framework.graph._ID_RE ([a-z0-9][a-z0-9_-]*).

    This is api.py's own tiny copy of plugins/mutators/graph_edit.py's
    `_unique_id` idea, not an import of it — a UI module has no business
    importing a plugin module, and the dedup logic is three lines.
    """
    candidate = base
    n = 2
    while candidate in taken:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _validate_eligible_detectors(meta) -> None:
    """Reject `meta.evo.eligible_detectors` naming anything that is not a
    registered detector key, so the Composer can never save a constraint
    evolution can never satisfy (`plugins/mutators/graph_edit.py`'s pool
    filter reads this same key). Anything else about `meta` is intentionally
    left alone — it is free-form JSON that survives `to_dict`/`from_dict`
    round-trips and is excluded from `graph_hash` (framework/graph.py), so
    this function's only job is to catch a typo'd or stale detector key
    before it is written to disk, not to police the shape of `meta` at large.
    """
    if not isinstance(meta, Mapping):
        return
    evo = meta.get("evo")
    if evo is None:
        return
    if not isinstance(evo, Mapping):
        raise GraphError("meta.evo must be an object")
    elig = evo.get("eligible_detectors")
    if elig is None:
        return
    if not isinstance(elig, list) or not all(isinstance(k, str) for k in elig):
        raise GraphError("meta.evo.eligible_detectors must be a list of strings")
    known = {f"detector.{n}" for n in registry.by_kind("detector")}
    bad = sorted(k for k in elig if not k.startswith("detector.") or k not in known)
    if bad:
        raise GraphError(
            f"meta.evo.eligible_detectors names unregistered detector(s): {bad}"
        )


def _stages_to_graph(name: str, stages: list, meta: Mapping | None = None) -> StrategyGraph:
    """The ONE mapping site between the HTTP body's linear `stages` list and
    Phase 3's StrategyGraph. `stages` exists only in HTTP bodies, never on
    disk (Task 4) — the file is always `graph.to_dict()`.

    The linear composer allows one OR MORE detector stages and exactly one
    policy stage, any number of confirmation and filter stages, and an
    optional single data stage (default data.ohlcv). Each detector stage
    becomes its own Branch — a Branch holds exactly one detector, so "N
    detector stages" IS "N branches" — every branch sharing the SAME
    composed confirmations and policy (mirrors
    plugins/mutators/graph_edit.py's add-branch move, which clones an
    existing branch's policy/regimes/exits for a new detector). The first
    branch is "main" (so a single-detector strategy keeps its historical
    id and hashes/round-trips byte-identically to before this change);
    branch k>0 gets an id derived from that detector's short registry name,
    deduped with `_unique_branch_id`. Node ids must be unique across the
    WHOLE graph (framework.graph.validate rule 5), so a branch's confirmation
    and policy NodeSpecs are cloned with id `f"{branch_id}-{orig_id}"` for
    every branch after the first — the detector node itself needs no
    cloning, since each detector stage already produced its own node.

    `meta` is the optional top-level `StrategyGraph.meta` payload (currently
    only `meta.evo.eligible_detectors`, validated here so a bad detector key
    is caught before anything is written).
    """
    data_node: NodeSpec | None = None
    detectors: list[NodeSpec] = []
    confirmations: list[NodeSpec] = []
    policy: NodeSpec | None = None
    filters: list[NodeSpec] = []

    for i, st in enumerate(stages):
        if not isinstance(st, Mapping):
            raise GraphError(f"stage {i} must be an object")
        kind = st.get("kind")
        key = st.get("key")
        if not isinstance(key, str) or not key:
            raise GraphError(f"stage {i}: missing or empty 'key'")
        params = st.get("params", {})
        if not isinstance(params, Mapping):
            raise GraphError(f"stage {i}: 'params' must be an object")
        node_id = st.get("id") or f"{kind}-{i}"
        node = NodeSpec(id=str(node_id), key=key, params=dict(params))

        if kind == "data":
            if data_node is not None:
                raise GraphError("only one data stage is supported by the linear composer")
            data_node = node
        elif kind == "detector":
            detectors.append(node)
        elif kind == "confirmation":
            confirmations.append(node)
        elif kind == "policy":
            if policy is not None:
                raise GraphError("only one policy stage is supported by the linear composer")
            policy = node
        elif kind == "filter":
            filters.append(node)
        else:
            raise GraphError(
                f"stage {i}: unknown kind {kind!r}; expected one of "
                f"data, detector, confirmation, policy, filter"
            )

    if not detectors:
        raise GraphError("a strategy needs at least one detector stage")
    if policy is None:
        raise GraphError("a strategy needs exactly one policy stage")
    if data_node is None:
        data_node = NodeSpec(id="data", key="data.ohlcv", params={})

    _validate_eligible_detectors(meta)

    branches: list[Branch] = []
    taken_branch_ids: set[str] = set()
    for k, det in enumerate(detectors):
        if k == 0:
            branch_id = "main"
        else:
            short = det.key.split(".", 1)[1] if "." in det.key else det.key
            branch_id = _unique_branch_id(short, taken_branch_ids)
        taken_branch_ids.add(branch_id)

        if k == 0:
            branch_confirmations = tuple(confirmations)
            branch_policy = policy
        else:
            branch_confirmations = tuple(
                dataclasses.replace(c, id=f"{branch_id}-{c.id}") for c in confirmations
            )
            branch_policy = dataclasses.replace(policy, id=f"{branch_id}-{policy.id}")

        branches.append(
            Branch(id=branch_id, detector=det, policy=branch_policy, confirmations=branch_confirmations)
        )

    return StrategyGraph(
        name=name,
        data=data_node,
        branches=tuple(branches),
        filters=tuple(filters),
        meta=dict(meta or {}),
    )


def _graph_to_stages(g: StrategyGraph) -> tuple[list, bool]:
    """StrategyGraph -> (stages, editable).

    editable=False whenever the graph's shape is not representable in the
    linear composer — the page must render it READ-ONLY rather than silently
    flatten and re-save it (NOT Building list). That used to mean "more than
    one branch"; it now means "branches that disagree with each other",
    because `_stages_to_graph` itself produces multi-branch graphs (one
    branch per detector stage) that all share the SAME policy, confirmations
    (same order), regimes and exits. A graph shaped exactly that way is this
    function's own round-trip and must come back editable, or every
    multi-detector strategy the Composer saves would immediately render
    read-only on reload. Anything less uniform — per-branch confirmations, a
    disabled branch, mismatched regimes/exits (an evolved champion, or
    plugins/mutators/graph_edit.py's add-branch move before this feature
    existed) — still returns ([], False); the data-loss guard in
    `post_strategy` relies on that to keep protecting graphs the linear
    composer cannot faithfully represent.

    Comparison is done on `StrategyGraph.to_dict()`'s branch dicts rather
    than the dataclass fields directly: to_dict() always emits plain
    JSON-shaped lists/dicts (never a tuple where another branch's copy has a
    list), so equality here can never trip over a list-vs-tuple artefact of
    which code path built the graph. Only each node's `id` is excluded from
    the comparison — branch k>0's confirmations/policy ids are legitimately
    different (they were cloned as f"{branch_id}-{orig_id}" precisely so
    every node id is graph-unique), and an id difference is not a content
    difference.
    """
    payload = g.to_dict()
    branch_dicts = payload["branches"]
    if not branch_dicts or not all(b["enabled"] for b in branch_dicts):
        return [], False

    def sig(b: dict) -> tuple:
        return (
            (b["policy"]["key"], b["policy"]["params"]),
            tuple((c["key"], c["params"]) for c in b["confirmations"]),
            b["regimes"],
            b["exits"],
        )

    base_sig = sig(branch_dicts[0])
    if any(sig(b) != base_sig for b in branch_dicts[1:]):
        return [], False

    first = g.branches[0]
    stages = [{"kind": "data", "key": g.data.key, "params": dict(g.data.params)}]
    for b in g.branches:
        stages.append({"kind": "detector", "key": b.detector.key, "params": dict(b.detector.params)})
    for c in first.confirmations:
        stages.append({"kind": "confirmation", "key": c.key, "params": dict(c.params)})
    stages.append({"kind": "policy", "key": first.policy.key, "params": dict(first.policy.params)})
    for f in g.filters:
        stages.append({"kind": "filter", "key": f.key, "params": dict(f.params)})
    return stages, True


def list_strategies() -> ApiResponse:
    d = Path(config.STRATEGY_DIR)
    names = []
    if d.exists():
        for p in sorted(d.glob("*.strategy.json")):
            names.append(p.name.removesuffix(".strategy.json"))
    return _ok({"strategies": names})


def get_strategy(name: str) -> ApiResponse:
    if not NAME_RE.match(name or ""):
        return _err(400, f"invalid strategy name {name!r}; expected {NAME_RE.pattern}")
    path = fgraph.path_for(name)
    if not path.exists():
        return _err(404, f"no strategy named {name!r} at {path}")
    try:
        registry.load_all()
        g = fgraph.load(path)
    except FrameworkError as exc:
        # A saved graph naming a plug-in that no longer registers: render red
        # with the reason, Save disabled — never silently drop the stage.
        return _ok(
            {"name": name, "graph": None, "stages": [], "editable": False, "error": str(exc)}
        )
    stages, editable = _graph_to_stages(g)
    return _ok(
        {
            "name": g.name,
            "graph": g.to_dict(),
            "stages": stages,
            "editable": editable,
            "graph_hash": fgraph.graph_hash(g),
            "meta": g.to_dict().get("meta", {}),
        }
    )


def post_strategy(name: str, body) -> ApiResponse:
    if not isinstance(name, str) or not NAME_RE.match(name):
        return _err(400, f"invalid strategy name {name!r}; expected {NAME_RE.pattern}")
    if not isinstance(body, Mapping) or not isinstance(body.get("stages"), list):
        return _err(400, "body must be an object with a 'stages' list")

    try:
        registry.load_all()
    except FrameworkError as exc:
        return _err(500, str(exc))

    try:
        graph = _stages_to_graph(name, body["stages"], meta=body.get("meta"))
        payload = graph.to_dict()
        # Round-trip BEFORE writing: a file the executor cannot read back is
        # worse than a rejection.
        if StrategyGraph.from_dict(payload).to_dict() != payload:
            return _err(500, "graph failed to_dict/from_dict round-trip")
        fgraph.validate(graph)
    except (KeyError, TypeError, GraphError, RegistryError) as exc:
        return _err(400, str(exc))

    target = fgraph.path_for(name).resolve()
    strategy_dir = Path(config.STRATEGY_DIR).resolve()
    if strategy_dir not in target.parents:
        return _err(400, "resolved path escapes STRATEGY_DIR")  # belt and braces

    # DATA-LOSS GUARD. `_graph_to_stages` marks a graph editable=False when the
    # linear composer cannot represent it (branches that disagree with each
    # other on policy/confirmations/regimes/exits, or a disabled branch), and
    # its docstring requires the page render it READ-ONLY "rather than
    # silently flatten and re-save it". Nothing enforced that on the WRITE
    # path, so any client could replace a committed multi-branch strategy with a
    # flattened single-branch one and lose the other branches irrecoverably --
    # observed: thin-slice went 2 branches -> 1 via one POST. The read-only flag
    # is advisory to the UI; this is the check that actually protects the file.
    # A UNIFORM multi-branch graph (this feature's own N-detector output) is
    # editable=True and saves normally; only a DIVERGENT multi-branch graph
    # (an evolved champion, e.g.) still trips this guard.
    if target.exists():
        try:
            existing = fgraph.load(str(target))
        except Exception:
            existing = None  # unreadable/corrupt: nothing coherent to protect
        if existing is not None:
            _, editable = _graph_to_stages(existing)
            if not editable and not bool(body.get("allow_flatten")):
                return _err(
                    409,
                    f"refusing to overwrite {name!r}: the stored graph has "
                    f"{len(existing.branches)} branches and is not representable in the "
                    "linear composer, so saving would silently discard the others. "
                    "Save under a new name, or resend with allow_flatten=true if "
                    "flattening is genuinely intended.",
                )

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return _ok(
        {
            "name": name,
            "schema_version": payload.get("schema_version"),
            "graph_hash": fgraph.graph_hash(graph),
        }
    )


def post_validate(body) -> ApiResponse:
    """Validate a composed strategy WITHOUT writing it — the [Validate] button."""
    if not isinstance(body, Mapping) or not isinstance(body.get("stages"), list):
        return _err(400, "body must be an object with a 'stages' list")
    name = body.get("name") or "validate-scratch"
    try:
        registry.load_all()
    except FrameworkError as exc:
        return _err(500, str(exc))
    try:
        graph = _stages_to_graph(name, body["stages"], meta=body.get("meta"))
        fgraph.validate(graph)
    except (KeyError, TypeError, GraphError, RegistryError) as exc:
        return _ok({"valid": False, "error": str(exc)})
    return _ok({"valid": True, "graph_hash": fgraph.graph_hash(graph)})


# --------------------------------------------------------------------------- #
# Task 5: result serialisers — gate rows, equity-vs-benchmark, reviews,
# generations. No metric is recomputed here; every number is read off the
# dataclass the oracle (or Phase 5/6) already produced.
# --------------------------------------------------------------------------- #


def _fmt(v, spec: str = ".4f") -> str:
    return format(v, spec) if v is not None else "n/a"


def _gate_threshold_and_measured(name: str, result) -> tuple[str, str]:
    """The (threshold text, measured text) pair for one gate condition.

    A benchmark row shows BOTH sides ("+X% vs +Y%") — a benchmark row without
    the benchmark number is the exact hiding KNOWN-LIMITATIONS §0 is about.
    """
    eq = result.oos_equity
    m = result.oos_metrics
    b = result.benchmark.basket
    if name == "sample_adequacy":
        return f">= {config.WF_MIN_TRADES}", f"{m['n_trades']} trades"
    if name == "sharpe":
        return f">= {GATE_MIN_SHARPE}", _fmt(eq["sharpe"], ".2f")
    if name == "dsr":
        return f"> {GATE_MIN_DSR}", _fmt(eq["dsr"], ".4f")
    if name == "max_drawdown":
        return f"<= {GATE_MAX_DRAWDOWN:.0%}", (
            "n/a" if eq["max_drawdown_pct"] is None else f"{eq['max_drawdown_pct']:.2%}"
        )
    if name == "per_symbol_expectancy":
        vals = result.per_symbol_expectancy
        n_pos = sum(1 for v in vals.values() if v is not None and v > 0)
        return "all symbols > 0", f"{n_pos} of {len(vals)} positive"
    if name == "beats_benchmark_return":
        strat = "n/a" if eq["ann_return_pct"] is None else f"{eq['ann_return_pct']:+.2%}"
        bench = "n/a" if b["ann_return_pct"] is None else f"{b['ann_return_pct']:+.2%}"
        return "> basket ann_return_pct", f"{strat} vs {bench}"
    if name == "beats_benchmark_sharpe":
        strat = _fmt(eq["sharpe"], "+.3f")
        bench = _fmt(b["sharpe"], "+.3f")
        return "> basket sharpe", f"{strat} vs {bench}"
    return "", ""  # pragma: no cover - defensive; GATE_CONDITIONS is closed


def _serialize_wf_result(
    result,
    *,
    curves: dict | None = None,
    trials_before: int | None = None,
    trials_after: int | None = None,
    campaign: str | None = None,
) -> dict:
    """WalkForwardResult -> page JSON. Reads `result.gate`, never recomputes
    it — a UI that re-derived PASS/FAIL could disagree with the engine, the
    one failure mode a verdict view must not have (contract §4)."""
    conditions = []
    for name in GATE_CONDITIONS:
        threshold, measured = _gate_threshold_and_measured(name, result)
        conditions.append(
            {
                "name": name,
                "ok": bool(result.gate.get(name, False)),
                "measured": measured,
                "threshold": threshold,
            }
        )
    payload = {
        "gate": {"passed": bool(result.passed), "conditions": conditions},
        "n_trials_used": result.n_trials_used,
        "oos_start_ms": result.oos_start,
        "oos_end_ms": result.oos_end,
        "oos_metrics": result.oos_metrics,
        "oos_equity": result.oos_equity,
        "per_symbol_expectancy": result.per_symbol_expectancy,
        "final_params": dataclasses.asdict(result.final_params),
        "final_max_hold_bars": result.final_max_hold_bars,
        "benchmark": dataclasses.asdict(result.benchmark),
        "folds": [dataclasses.asdict(f) for f in result.folds],
        "curves": curves or {},
    }
    if trials_before is not None:
        payload["trials_before"] = trials_before
    if trials_after is not None:
        payload["trials_after"] = trials_after
    if campaign is not None:
        payload["campaign"] = campaign
    return _jsonsafe(payload)


def _oos_curves(conn, graph: StrategyGraph, symbols, oos_start: int, oos_end: int) -> dict:
    """Strategy equity vs the buy-and-hold basket over the OOS span, as number
    arrays (the page is static and must redraw for a different run).

    Re-runs run_graph_backtest over the SAME already-scored OOS window to get
    the trade list for charting — a deterministic re-derivation of the exact
    trades walk_forward_pooled already produced and scored, NOT a second
    evaluation: nothing here calls the oracle a second time or touches the
    trial ledger. The basket leg mirrors backtest.benchmark.buy_and_hold's
    daily-rebalance arithmetic (COUPLING: calls its private `_close_returns`
    rather than re-deriving the fee/rebalance formula a second time — see
    test_ui_api.py::TestSerializers::test_oos_curves_basket_matches_buy_and_hold
    for the drift guard).
    """
    from trading_bot.backtest import benchmark as bt_benchmark

    trades = []
    for sym in symbols:
        trades.extend(run_graph_backtest(conn, graph, sym, start_ms=oos_start, end_ms=oos_end))
    strat_rets = daily_returns(trades, oos_start, oos_end, attribution=config.PNL_ATTRIBUTION_MODE)

    series: dict[str, list[float]] = {}
    for sym in symbols:
        rets = bt_benchmark._close_returns(
            conn, sym, start_ms=oos_start, end_ms=oos_end,
            timeframe=config.BENCHMARK_TIMEFRAME, charge_fees=config.BENCHMARK_CHARGE_FEES,
        )
        if rets:
            series[sym] = rets
    basket_rets: list[float] = []
    if series:
        lengths = {len(v) for v in series.values()}
        if len(lengths) == 1:
            n = lengths.pop()
            basket_rets = [
                sum(series[s][i] for s in series) / len(series) for i in range(n)
            ]

    return {
        "strategy": _compound_equity(strat_rets),
        "basket": _compound_equity(basket_rets),
        "oos_start_ms": oos_start,
        "oos_end_ms": oos_end,
        "trades": _serialize_trades(trades),
    }


def _compound_equity(returns) -> list[float]:
    """Daily returns -> an equity path starting at 1.0. ONE implementation, used
    by both the gate curves and the backtest curve, so the two charts cannot
    drift apart."""
    eq, out = 1.0, [1.0]
    for r in returns:
        eq *= 1.0 + r
        out.append(eq)
    return out


def _serialize_trades(trades) -> list[dict]:
    """Compact per-trade rows for the results view's trade table and P&L
    histogram. ADDITIVE and presentational only: these are the trades the run
    already produced, not a re-evaluation — nothing here touches the ledger or
    recomputes a verdict. Sorted by entry so the table is deterministic.

    `pnl_pct` is the engine's own AFTER-COST figure. The page must never show a
    gross number beside a net one without saying which (contract §1: any new
    path charges costs identically or it is lying).

    pattern_start_ts/pattern_end_ts/pattern_level/pattern_meta (v0.3.2 WS-B,
    D4) are the same additive, presentational pass-through as everything else
    here: `Trade.pattern_meta` is a sorted tuple of (key, value) pairs (Trade
    is frozen and a dict is unhashable — see engine.py's Trade docstring), and
    JSON has no tuple, so it is re-expanded into a plain dict here. A run_backtest
    (legacy-engine) trade or a pre-v0.3.2 persisted run has these at their
    "no geometry" defaults (0 / 0.0 / ()), which this deliberately does NOT
    paper over with a fallback — evoDrawTrade (app.js) is expected to render no
    zone at all for pattern_start_ts == 0, not the old hardcoded box.
    """
    rows = [
        {
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts,
            "entry": t.entry,
            "stop": t.stop,
            "target": t.target,
            "exit_price": t.exit_price,
            "pnl_pct": t.pnl_pct,
            "outcome": t.outcome,
            "regime": t.regime,
            "pattern": t.pattern,
            "planned_rr": t.planned_rr,
            "pattern_start_ts": t.pattern_start_ts,
            "pattern_end_ts": t.pattern_end_ts,
            "pattern_level": t.pattern_level,
            "pattern_meta": dict(t.pattern_meta),
        }
        for t in trades
    ]
    rows.sort(key=lambda r: (r["entry_ts"], r["symbol"]))
    return rows


def get_reviews(*, strategy_version: str | None = None, limit: int = 50) -> ApiResponse:
    """Recent review_records, via Phase 5's own accessor
    (feedback.records.load_records) — never raw SQL over its table."""
    conn = connect_state(config.STATE_DB_PATH)
    try:
        records = feedback_records.load_records(conn, strategy_version=strategy_version)
    finally:
        conn.close()
    records = records[-limit:] if limit else records
    return _ok({"reviews": [_jsonsafe(r) for r in records], "n": len(records)})


def get_generations(campaign: str | None) -> ApiResponse:
    """A campaign's generation rows plus a short leaderboard, via Phase 6's
    own accessors (evolution.population.generation_rows / top_members) —
    never raw SQL over its tables (Task 5 GOTCHA #1)."""
    if not campaign:
        return _err(400, "campaign query parameter is required")
    conn = connect_state(config.STATE_DB_PATH)
    try:
        gens = evo_population.generation_rows(conn, campaign)
        top = evo_population.top_members(conn, campaign, limit=10)
    finally:
        conn.close()
    return _ok({"campaign": campaign, "generations": _jsonsafe(gens), "top_members": _jsonsafe(top)})


def get_evolution_members(campaign: str | None, gen) -> ApiResponse:
    """One generation's SCORED members, via evo_population.generation_members
    — never raw SQL over population_members (Task 5 GOTCHA #1).

    `graph_json` is the largest column by far and the sidebar's member list
    never needs the whole graph, only "what does this member run" — so it is
    STRIPPED from every row here, and its branch detector keys are parsed out
    server-side into `detectors` / `n_branches` instead. A NULL or
    unparseable `graph_json` (should not happen, but this endpoint must not
    500 the whole sidebar over one bad row) reports empty detectors rather
    than raising.
    """
    if not campaign:
        return _err(400, "campaign query parameter is required")
    try:
        gen_index = int(gen)
    except (TypeError, ValueError):
        return _err(400, f"gen query parameter must be an integer, got {gen!r}")

    conn = connect_state(config.STATE_DB_PATH)
    try:
        rows = evo_population.generation_members(conn, campaign, gen_index)
    finally:
        conn.close()

    members = []
    for row in rows:
        r = dict(row)
        graph_json = r.pop("graph_json", None)
        detectors: list[str] = []
        try:
            if graph_json:
                branches = json.loads(graph_json).get("branches") or []
                detectors = [b["detector"]["key"] for b in branches]
        except (TypeError, ValueError, KeyError):  # noqa: BLE001 — see docstring
            detectors = []
        r["detectors"] = detectors
        r["n_branches"] = len(detectors)
        members.append(r)
    return _ok({"campaign": campaign, "gen": gen_index, "members": _jsonsafe(members)})


def get_evolution_replay(member: str | None, symbol: str | None = None, timeframe: str | None = None) -> ApiResponse:
    """Bar-by-bar replay of one population member's evaluation window.

    A DETERMINISTIC RE-DERIVATION of the exact trades the campaign already
    scored, over the exact window it scored them in (the member's own
    `window_start_ms` / `window_end_ms`, frozen at evaluation time) — NOT a
    second evaluation, exactly like `_oos_curves` above: this never calls
    `walk_forward_pooled`, never opens a `backtest.trials.TrialLedger`, and
    never writes a row to state.db. Read access to `population_members` and
    `campaigns` goes only through `evolution.population`'s own accessors
    (Task 5 GOTCHA #1) — no raw SQL here either. Repeat calls (e.g. scrubbing
    the same member back and forth) are cheap because
    `framework.execute`'s indicator cache memoizes the per-(symbol,
    timeframe, params) indicator work across calls.

    Order of checks: 404 if the member id does not exist; 409 if it was never
    scored (`window_start_ms` NULL — a member that errored before evaluation
    or is still queued has no window to replay) or if it recorded an error
    during evaluation (the stored error text is echoed back rather than
    silently replaying an empty run); 400 if `symbol` is given but is not one
    of the owning campaign's symbols, or if `timeframe` is not a stored
    timeframe. `symbol` defaults to the campaign's first symbol; `timeframe`
    defaults to "1d".
    """
    if not member:
        return _err(400, "member query parameter is required")

    state_conn = connect_state(config.STATE_DB_PATH)
    try:
        row = evo_population.load_member(state_conn, member)
        if row is None:
            return _err(404, f"no member {member!r}")
        if row.get("window_start_ms") is None or row.get("window_end_ms") is None:
            return _err(
                409, f"member {member!r} was never scored (no evaluation window)"
            )
        if row.get("error"):
            return _err(
                409, f"member {member!r} errored during evaluation: {row['error']}"
            )
        campaign_row = evo_population.load_campaign(state_conn, row["campaign_id"])
    finally:
        state_conn.close()

    if campaign_row is None:
        return _err(404, f"no campaign {row['campaign_id']!r} for member {member!r}")

    symbols = list(campaign_row.symbols)
    if not symbols:
        return _err(500, f"campaign {row['campaign_id']!r} has no symbols")
    if symbol is None:
        symbol = symbols[0]
    elif symbol not in symbols:
        return _err(
            400, f"symbol {symbol!r} is not one of the campaign's symbols {symbols}"
        )

    timeframe = timeframe or "1d"
    if timeframe not in config.TIMEFRAMES:
        return _err(
            400, f"timeframe {timeframe!r} must be one of {list(config.TIMEFRAMES)}"
        )

    try:
        registry.load_all()
    except FrameworkError as exc:
        return _err(500, str(exc))

    from trading_bot.evolution.mutate import graph_from_json  # local import: see _oos_curves

    try:
        graph = graph_from_json(row["graph_json"])
    except (GraphError, ValueError) as exc:
        return _err(500, f"member {member!r} has an unreadable graph: {exc}")

    conn = storage.connect(config.DB_PATH)
    try:
        trades = run_graph_backtest(
            conn, graph, symbol,
            start_ms=row["window_start_ms"], end_ms=row["window_end_ms"],
        )
        bars = storage.load_candles(
            conn, symbol, timeframe,
            start_ms=row["window_start_ms"], end_ms=row["window_end_ms"],
        )
    finally:
        conn.close()

    # Parallel arrays, not a list of {ts,o,h,l,c,v} objects: a 90-180 day
    # window is thousands of bars, and the replay chart re-reads this whole
    # payload on every symbol/timeframe change.
    bars_out: dict[str, list] = {"ts": [], "o": [], "h": [], "l": [], "c": [], "v": []}
    for ts, o, h, l, c, v in bars:  # noqa: E741 — mirrors load_candles' own (ts,o,h,l,c,v)
        bars_out["ts"].append(ts)
        bars_out["o"].append(o)
        bars_out["h"].append(h)
        bars_out["l"].append(l)
        bars_out["c"].append(c)
        bars_out["v"].append(v)

    member_out = {k: v for k, v in row.items() if k != "graph_json"}

    return _ok(
        {
            "member": _jsonsafe(member_out),
            "symbol": symbol,
            "timeframe": timeframe,
            "window_start_ms": row["window_start_ms"],
            "window_end_ms": row["window_end_ms"],
            "bars": _jsonsafe(bars_out),
            "trades": _serialize_trades(trades),
        }
    )


# --------------------------------------------------------------------------- #
# Task 6: starting and observing runs — Tier A thread, Tier B subprocess
# --------------------------------------------------------------------------- #


def new_run_id() -> str:
    return f"{int(time.time() * 1000)}-{secrets.token_hex(3)}"


def _run_dir(run_id: str) -> Path:
    return RUN_ROOT / run_id


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: Path, payload) -> None:
    """tmp + replace: an SSE poll or a reloaded page must never read a
    truncated status.json on a healthy run."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonsafe(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _parse_date(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return config.date_to_ms(value)
    raise ValueError(f"bad date {value!r}; expected YYYY-MM-DD or an epoch-ms int")


def champion_of(state_conn, campaign_id: str) -> dict | None:
    """The best scored member of a campaign, or None if it has none yet.

    top_members' SQL order is only a shortlist filter (see its docstring), but
    with limit=1 over a single campaign the shortlist head IS the champion under
    the same (tier, fitness) key the leaderboard shows, so the Evolution page
    and the graph evaluated here can never name different winners.
    """
    rows = evo_population.top_members(state_conn, campaign_id, limit=1)
    return rows[0] if rows else None


def _graph_for_run(cmd: dict):
    """The graph a Tier A run must evaluate, plus its provenance.

    A campaign that has evolved is evaluated AS ITS CHAMPION, not as the seed
    file on disk. Re-reading the seed was the defect that made a 16-generation
    campaign score identically to generation zero: evolution wrote its winners
    to population_members and nothing ever read them back. A campaign with no
    scored member yet still starts from the seed file.
    """
    if cmd.get("graph_json"):
        graph = fgraph.StrategyGraph.from_dict(json.loads(cmd["graph_json"]))
        return graph, cmd.get("graph_source")
    return fgraph.load(cmd["strategy_path"]), {"kind": "seed", "strategy": cmd.get("strategy")}


def _run_backtest_job(cmd: dict) -> dict:
    """Tier A: in-sample diagnostic. Touches NO trial ledger — the same
    honesty framing as `cli.py graph-backtest`."""
    conn = storage.connect(config.DB_PATH)
    try:
        registry.load_all()
        graph, graph_source = _graph_for_run(cmd)
        per_symbol: dict[str, list] = {}
        pooled: list = []
        for sym in cmd["symbols"]:
            trades = run_graph_backtest(
                conn, graph, sym, start_ms=cmd["start_ms"], end_ms=cmd["end_ms"]
            )
            per_symbol[sym] = trades
            pooled.extend(trades)
        payload = {
            "kind": "backtest",
            "graph_source": graph_source,
            "honesty": (
                "IN-SAMPLE DIAGNOSTIC — not evidence. Fitness comes only from "
                "THE GATE (contract §4); no trial was charged for this run."
            ),
            "per_symbol_metrics": {s: compute_metrics(t) for s, t in per_symbol.items()},
            "pooled_metrics": compute_metrics(pooled),
            "pooled_equity": compute_equity_metrics(pooled, cmd["start_ms"], cmd["end_ms"]),
            "trades": _serialize_trades(pooled),
            "start_ms": cmd["start_ms"],
            "end_ms": cmd["end_ms"],
            # Compounded from equity.daily_returns — the SAME arithmetic
            # compute_equity_metrics scores, under the configured attribution
            # mode — so the chart and the metrics cannot disagree. The page must
            # never derive an equity path of its own.
            "curves": {
                "strategy": _compound_equity(
                    daily_returns(
                        pooled, cmd["start_ms"], cmd["end_ms"],
                        attribution=config.PNL_ATTRIBUTION_MODE,
                    )
                ),
                "oos_start_ms": cmd["start_ms"],
                "oos_end_ms": cmd["end_ms"],
            },
        }
        return _jsonsafe(payload)
    finally:
        conn.close()


def _run_gate_job(cmd: dict) -> dict:
    """Tier A: THE GATE. Charges exactly what walk_forward_pooled charges —
    one ledger row per (pooled combo, fold), through the SAME oracle path
    `cli.py walkforward --graph` uses."""
    conn = storage.connect(config.DB_PATH)
    state_conn = connect_state(config.STATE_DB_PATH)
    try:
        registry.load_all()
        graph, graph_source = _graph_for_run(cmd)
        ledger = trials.TrialLedger(state_conn, cmd.get("ledger_key") or cmd["campaign"])
        trials_before = ledger.count()
        result = walk_forward_pooled(
            conn, cmd["symbols"], start_ms=cmd["start_ms"], end_ms=cmd["end_ms"],
            strategy=graph, grid={"max_hold_bars": DEFAULT_GRID["max_hold_bars"]},
            ledger=ledger,
        )
        trials_after = ledger.count()
        curves = _oos_curves(conn, graph, cmd["symbols"], result.oos_start, result.oos_end)
        payload = _serialize_wf_result(
            result, curves=curves, trials_before=trials_before,
            trials_after=trials_after, campaign=cmd["campaign"],
        )
        payload["kind"] = "gate"
        payload["graph_source"] = _jsonsafe(graph_source)
        return payload
    finally:
        conn.close()
        state_conn.close()


_TIER_A_JOBS = {"backtest": _run_backtest_job, "gate": _run_gate_job}


def _tier_a_worker(run_id: str, cmd: dict) -> None:
    run_dir = _run_dir(run_id)
    log_path = run_dir / "stdout.log"
    try:
        fn = _TIER_A_JOBS[cmd["kind"]]
        result = fn(cmd)
        _write_json_atomic(run_dir / "result.json", result)
        status = _read_json(run_dir / "status.json") or {}
        status.update(
            {"state": "done", "finished_ms": int(time.time() * 1000), "exit_code": 0}
        )
        _write_json_atomic(run_dir / "status.json", status)
    except BaseException as exc:  # noqa: BLE001 — an unhandled exception here must
        # not vanish silently in a daemon thread, leaving the page showing
        # "running" forever.
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"FAILED: {type(exc).__name__}: {exc}\n")
                fh.write(traceback.format_exc())
        except OSError:  # pragma: no cover - defensive
            pass
        status = _read_json(run_dir / "status.json") or {}
        status.update(
            {"state": "failed", "finished_ms": int(time.time() * 1000), "exit_code": 1}
        )
        _write_json_atomic(run_dir / "status.json", status)
    finally:
        with _STATE_LOCK:
            _STATE["active_tier_a"] = max(0, _STATE["active_tier_a"] - 1)


def _tier_b_watcher(run_id: str, proc: subprocess.Popen) -> None:
    """Reap a Tier B (evolve) subprocess and finalize its status.json.

    Without this, a finished evolve run leaves a zombie process and a
    status.json stuck at state='running' forever — nothing else ever calls
    wait() on it or writes a terminal state, so the dashboard (and the Stop
    button, which just sends SIGTERM to an already-exited pid) can't tell
    the run is over."""
    run_dir = _run_dir(run_id)
    try:
        returncode = proc.wait()
    except BaseException as exc:  # noqa: BLE001 — an unhandled exception in this
        # daemon thread vanishes silently and leaves the page showing "running"
        # forever, which is precisely the stuck-run bug this watcher exists to
        # prevent (Phase 7 Task 6 GOTCHA #2).
        logger.warning("tier B watcher for %s failed: %s", run_id, exc)
        try:
            with open(run_dir / "stdout.log", "a", encoding="utf-8") as fh:
                fh.write(f"WATCHER FAILED: {type(exc).__name__}: {exc}\n")
        except OSError:  # pragma: no cover - defensive
            pass
        returncode = -1
    status = _read_json(run_dir / "status.json") or {}
    status.update(
        {
            "state": "done" if returncode == 0 else "failed",
            "finished_ms": int(time.time() * 1000),
            "exit_code": returncode,
        }
    )
    _write_json_atomic(run_dir / "status.json", status)


def _evolve_argv(cmd: dict, seed: int, *, resume: str | None = None,
                 extend: int | None = None, population: int | None = None,
                 generations: int | None = None) -> list[str]:
    """Popen argv for a Tier B evolve run. A LIST, never a shell string —
    Popen takes a list and shell=False, so no request value is ever
    interpolated into a shell command (injection guard).

    Two shapes: a NEW campaign carries --label/--strategy-name so it can be
    found again tomorrow, and a CONTINUATION carries --resume/--extend so it
    improves the population it already paid for rather than restarting.

    `population` only applies to the NEW-campaign shape: run_campaign silently
    ignores population_size on a resume (population is locked to what the
    campaign was created with), so a --population flag on a --resume would be
    a value the engine quietly drops — never emit it there. `generations`
    doubles as the operator's --extend count on a resume: "how many more
    generations" is the same question on either shape, just answered against
    a different baseline (from zero vs. from the last completed generation).
    """
    argv = [sys.executable, "-m", "trading_bot.cli", "evolve"]
    if resume:
        n = extend if extend is not None else generations
        argv += ["--resume", resume, "--extend", str(int(n or config.EVO_GENERATIONS))]
    else:
        argv += [
            "--seed-graph", cmd["strategy_path"], "--seed", str(seed),
            "--label", cmd["campaign"], "--strategy-name", cmd["strategy"],
        ]
        for s in cmd["symbols"]:
            argv += ["--symbol", s]
        if population is not None:
            argv += ["--population", str(int(population))]
        if generations is not None:
            argv += ["--generations", str(int(generations))]
    return argv


def get_campaigns(strategy: str | None = None) -> ApiResponse:
    """Existing campaigns, newest first, for the campaign dropdowns.

    A campaign is an evolution history OF one strategy (one strategy, many
    campaigns), so `strategy` filters to the one currently loaded in the
    composer — offering a campaign bred from a different graph would invite
    exactly the mismatch post_run refuses.
    """
    if strategy is not None and not NAME_RE.match(strategy):
        return _err(400, f"invalid strategy name {strategy!r}")
    conn = connect_state(config.STATE_DB_PATH)
    try:
        evo_population.ensure_schema(conn)
        rows = evo_population.list_campaigns(conn, strategy_name=strategy)
    finally:
        conn.close()
    return _ok({"strategy": strategy, "campaigns": _jsonsafe(rows)})


def post_run(body) -> ApiResponse:
    if not isinstance(body, Mapping):
        return _err(400, "body must be an object")
    kind = body.get("kind")
    if kind not in _RUN_KINDS:
        return _err(400, f"unknown run kind {kind!r}; expected one of {sorted(_RUN_KINDS)}")

    strategy = body.get("strategy")
    if not isinstance(strategy, str) or not NAME_RE.match(strategy):
        return _err(400, f"invalid strategy name {strategy!r}; expected {NAME_RE.pattern}")
    strategy_path = fgraph.path_for(strategy)
    if not strategy_path.exists():
        return _err(400, f"no strategy named {strategy!r} at {strategy_path}")

    symbols = body.get("symbols") or list(config.SYMBOLS)
    if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) for s in symbols):
        return _err(400, "symbols must be a non-empty list of strings")
    try:
        stored = set(_stored_symbols())
    except Exception as exc:  # noqa: BLE001
        return _err(500, f"could not read stored symbols: {exc}")
    bad = [s for s in symbols if s not in stored]
    if bad:
        return _err(400, f"symbol(s) not in the stored set: {bad}")

    try:
        start_ms = (
            _parse_date(body["start"]) if body.get("start")
            else config.date_to_ms(config.BACKFILL_START)
        )
        end_ms = _parse_date(body["end"]) if body.get("end") else int(time.time() * 1000)
    except ValueError as exc:
        return _err(400, f"bad date: {exc}")

    campaign = body.get("campaign")
    tier = _RUN_KINDS[kind]
    if kind in ("gate", "evolve"):
        if not isinstance(campaign, str) or not NAME_RE.match(campaign):
            return _err(
                400,
                f"campaign is required for {kind!r} runs and must match {NAME_RE.pattern} "
                f"— every ledger row must be attributable to a named campaign",
            )

    # Both optional, evolve-only. `generations` means "generations to run" for
    # a brand-new campaign and "how many MORE generations" (--extend) once one
    # is being resumed — _evolve_argv picks the right meaning; population only
    # ever applies to a new campaign (resume locks it, see _evolve_argv's note).
    population = body.get("population")
    generations = body.get("generations")
    for name, value in (("population", population), ("generations", generations)):
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return _err(400, f"{name} must be a positive integer, got {value!r}")

    # Resolve the operator's campaign label to a campaign, and enforce the
    # binding: a campaign is an evolution history OF one strategy, so gating or
    # evolving it against a different graph would silently mix two searches into
    # one trial ledger.
    existing = None
    champion = None
    if campaign:
        state_conn = connect_state(config.STATE_DB_PATH)
        try:
            evo_population.ensure_schema(state_conn)
            existing = evo_population.campaign_by_label(state_conn, campaign)
            # Tier A scores what the campaign actually evolved into. Tier B is
            # excluded on purpose: run_campaign reloads the population itself,
            # and handing it a champion here would collapse the search to one
            # lineage.
            if existing is not None and tier == "A":
                champion = champion_of(state_conn, existing.campaign_id)
        finally:
            state_conn.close()
        if existing is not None and existing.strategy_name and (
            existing.strategy_name != strategy
        ):
            return _err(
                400,
                f"campaign {campaign!r} belongs to strategy "
                f"{existing.strategy_name!r}, not {strategy!r}. A campaign is one "
                f"strategy's evolution history — pick a campaign for "
                f"{strategy!r} or name a new one.",
            )

    # The ledger keys on the campaign_id once one exists, so a gate run and an
    # evolve run of the same campaign share ONE cumulative count (contract §4).
    # Before the first evolve there is no campaign_id, so the label stands in and
    # run_campaign adopts those rows when it creates the campaign.
    ledger_key = existing.campaign_id if existing is not None else campaign

    if tier == "A":
        with _STATE_LOCK:
            if _STATE["active_tier_a"] >= MAX_TIER_A_JOBS:
                return _err(429, "busy — 1 gate/backtest run at a time")

    run_id = new_run_id()
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    trials_before = None
    if kind == "gate":
        state_conn = connect_state(config.STATE_DB_PATH)
        try:
            trials_before = trials.TrialLedger(state_conn, ledger_key).count()
        finally:
            state_conn.close()

    cmd = {
        "kind": kind, "strategy": strategy, "strategy_path": str(strategy_path),
        "symbols": symbols, "start_ms": start_ms, "end_ms": end_ms, "campaign": campaign,
        # `campaign` stays the operator's label (what the page shows); ledger_key
        # is what the trial ledger is actually charged under.
        "ledger_key": ledger_key,
        # None until this campaign's first evolve creates it. Carried so every
        # panel can name the campaign the same way the engine does.
        "campaign_id": existing.campaign_id if existing is not None else None,
    }
    if champion is not None:
        # Frozen into cmd.json rather than looked up at job time so a run stays
        # reproducible: a concurrent evolve that crowns a new champion must not
        # change what an already-launched gate is scoring.
        cmd["graph_json"] = champion["graph_json"]
        cmd["graph_source"] = {
            "kind": "champion",
            "campaign": campaign,
            "campaign_id": existing.campaign_id,
            "member_id": champion["member_id"],
            "gen_index": champion["gen_index"],
            "graph_hash": champion["graph_hash"],
            "fitness": champion["fitness"],
        }
    _write_json_atomic(run_dir / "cmd.json", cmd)
    status = {
        "state": "running", "tier": tier, "pid": None, "argv": None,
        "started_ms": int(time.time() * 1000), "finished_ms": None, "exit_code": None,
    }

    if tier == "A":
        with _STATE_LOCK:
            _STATE["active_tier_a"] += 1
        _write_json_atomic(run_dir / "status.json", status)
        th = threading.Thread(target=_tier_a_worker, args=(run_id, cmd), daemon=True)
        with _STATE_LOCK:
            _THREADS[run_id] = th
        th.start()
    else:
        seed = evo_population.derive_seed("ui-campaign", campaign)
        if existing is not None:
            # CONTINUE, never restart. Re-deriving a seed here would begin a
            # fresh search and throw away every generation the operator already
            # paid trials for; --resume/--extend breeds on from the evolved
            # population instead, which is the whole point of evolving again.
            argv = _evolve_argv(cmd, seed, resume=existing.campaign_id, generations=generations)
            resolved_campaign_id = existing.campaign_id
        else:
            try:
                registry.load_all()
                seed_graph = fgraph.load(cmd["strategy_path"])
                resolved_campaign_id = evo_runner._campaign_id(
                    seed, evo_population.graph_hash(seed_graph)
                )
            except Exception as exc:  # noqa: BLE001 — do not fail the launch over a display hint
                logger.warning("could not predict campaign_id: %s", exc)
                resolved_campaign_id = None
            argv = _evolve_argv(cmd, seed, population=population, generations=generations)
        status["argv"] = argv
        status["campaign_id"] = resolved_campaign_id
        status["continued"] = existing is not None
        log_path = run_dir / "stdout.log"
        with open(log_path, "ab") as log_fh:
            proc = subprocess.Popen(
                argv, stdout=log_fh, stderr=subprocess.STDOUT,
                start_new_session=True, shell=False,
            )
        status["pid"] = proc.pid
        _write_json_atomic(run_dir / "status.json", status)
        th = threading.Thread(target=_tier_b_watcher, args=(run_id, proc), daemon=True)
        with _STATE_LOCK:
            _THREADS[run_id] = th
        th.start()

    trials_estimated = 1 if kind == "gate" else (None if kind == "backtest" else "unbounded")
    return ApiResponse(
        202,
        {
            "run_id": run_id, "tier": tier,
            "trials_before": trials_before, "trials_estimated": trials_estimated,
        },
    )


_RUN_KINDS = {"backtest": "A", "gate": "A", "evolve": "B"}


def get_runs() -> ApiResponse:
    if not RUN_ROOT.exists():
        return _ok({"runs": []})
    out = []
    for run_dir in sorted(RUN_ROOT.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        status = _reconcile_stale_tier_b(run_dir, _read_json(run_dir / "status.json") or {})
        out.append(
            {
                "run_id": run_dir.name,
                "cmd": _read_json(run_dir / "cmd.json") or {},
                "status": status,
            }
        )
    return _ok({"runs": out})


def get_run(run_id: str) -> ApiResponse:
    if not RUN_ID_RE.match(run_id or ""):
        return _err(400, f"invalid run id {run_id!r}")
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        return _err(404, f"no run {run_id!r}")
    status = _reconcile_stale_tier_b(run_dir, _read_json(run_dir / "status.json") or {})
    return _ok(
        {
            "run_id": run_id,
            "cmd": _read_json(run_dir / "cmd.json") or {},
            "status": status,
            "result": _read_json(run_dir / "result.json"),
        }
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def post_run_stop(run_id: str) -> ApiResponse:
    if not RUN_ID_RE.match(run_id or ""):
        return _err(400, f"invalid run id {run_id!r}")
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        return _err(404, f"no run {run_id!r}")
    status = _read_json(run_dir / "status.json") or {}
    if status.get("tier") != "B":
        return _err(
            400,
            "only Tier B (evolve) runs can be stopped — Tier A (gate/backtest) has "
            "no cooperative cancel point (NOT Building list)",
        )
    pid = status.get("pid")
    stopping = bool(pid and _pid_alive(pid))
    if stopping:
        os.kill(pid, signal.SIGTERM)
    return _ok({"run_id": run_id, "stopping": stopping})


def _reconcile_stale_tier_b(run_dir: Path, status: dict) -> dict:
    """Self-heal a Tier B status.json stuck at state='running'.

    A Tier B (evolve) subprocess is reaped by an in-memory watcher thread
    (`_tier_b_watcher`) owned by the server process that spawned it. If that
    server restarts, the subprocess is orphaned — it keeps running (Popen
    children survive their parent's exit) but nothing will ever be alive to
    write its terminal state, even after it exits cleanly. `recover_runs`
    only catches this once at boot, and only if the pid is *already* dead by
    then; a subprocess that outlives the restart and finishes afterward slips
    through forever. So every read of a Tier B status also lands here.

    Ground truth is the campaign row in state.db, not the exit code we never
    got to observe: the evolve subprocess writes that row itself as it runs,
    so a dead pid whose campaign reached status='done' succeeded, regardless
    of who was or wasn't around to reap it.
    """
    if status.get("tier") != "B" or status.get("state") != "running":
        return status
    pid = status.get("pid")
    if pid and _pid_alive(pid):
        return status
    campaign = None
    campaign_id = status.get("campaign_id")
    if campaign_id:
        conn = connect_state(config.STATE_DB_PATH)
        try:
            campaign = evo_population.load_campaign(conn, campaign_id)
        finally:
            conn.close()
    if campaign is not None and campaign.status == "done":
        status["state"] = "done"
        status["exit_code"] = 0
        status["finished_ms"] = status.get("finished_ms") or campaign.finished_ts or int(
            time.time() * 1000
        )
    else:
        status["state"] = "failed"
        status["exit_code"] = status.get("exit_code", 1)
        status["finished_ms"] = status.get("finished_ms") or int(time.time() * 1000)
    _write_json_atomic(run_dir / "status.json", status)
    return status


def recover_runs() -> None:
    """Run once at server start: reclassify any `status.json` still 'running'
    from a previous process. Without this a restarted server shows phantom
    runs forever."""
    if not RUN_ROOT.exists():
        return
    for run_dir in RUN_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        status_path = run_dir / "status.json"
        status = _read_json(status_path)
        if not status or status.get("state") != "running":
            continue
        if status.get("tier") == "B" and status.get("pid"):
            _reconcile_stale_tier_b(run_dir, status)
        else:
            status["state"] = "orphaned"
            status["finished_ms"] = status.get("finished_ms") or int(time.time() * 1000)
            _write_json_atomic(status_path, status)


# --------------------------------------------------------------------------- #
# Task 9 (framing only): pure SSE frame formatting. server.py writes bytes to
# the socket; this module only decides what the frames say.
# --------------------------------------------------------------------------- #


def sse_frames(offset: int, log_text: str, status: dict, progress: dict | None = None) -> list[str]:
    """Format one poll's worth of SSE frames from a log-tail read.

    Args:
        offset: Byte offset BEFORE this read (used to compute the new `id:`).
        log_text: Newly-read bytes from stdout.log, decoded as text. May be
            empty (nothing new since the last poll).
        status: The run's current status.json dict.
        progress: Optional Tier B progress payload (generation/population
            counts); emitted only when given.

    Returns:
        A list of complete SSE frame strings (each already `\n\n`-terminated
        except heartbeats' single `\n`). Multi-line log text becomes one
        `data:` line PER LINE — the spec's `\n`-join rule, and a raw newline
        inside one `data:` line silently truncates the event.
    """
    frames: list[str] = []
    if log_text:
        lines = log_text.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        new_offset = offset + len(log_text.encode("utf-8"))
        for line in lines:
            clean = line.replace("\r", "")
            frames.append(f"id: {new_offset}\nevent: log\ndata: {clean}\n\n")
    if progress is not None:
        frames.append(f"event: progress\ndata: {json.dumps(_jsonsafe(progress))}\n\n")
    if status.get("state") in ("done", "failed", "orphaned", "aborted"):
        done_payload = {
            "state": status.get("state"),
            "exit_code": status.get("exit_code"),
            "has_result": bool(status.get("has_result")),
        }
        frames.append(f"event: done\ndata: {json.dumps(done_payload)}\n\n")
    if not frames:
        frames.append(": heartbeat\n")
    return frames


# --------------------------------------------------------------------------- #
# Task 7: the dispatcher
# --------------------------------------------------------------------------- #


def _review_query(query: Mapping) -> dict:
    def first(key):
        v = query.get(key)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    limit_raw = first("limit")
    try:
        limit = int(limit_raw) if limit_raw else 50
    except (TypeError, ValueError):
        limit = 50
    return {"strategy_version": first("strategy_version"), "limit": limit}


def _generations_query(query: Mapping) -> str:
    v = query.get("campaign")
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def _members_query(query: Mapping) -> tuple[str, str]:
    def first(key):
        v = query.get(key)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    return first("campaign"), first("gen")


def _replay_query(query: Mapping) -> dict:
    def first(key):
        v = query.get(key)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    return {
        "member": first("member"),
        "symbol": first("symbol"),
        "timeframe": first("timeframe"),
    }


def _campaigns_query(query: Mapping) -> str | None:
    """?strategy=<name>, or None for "every campaign".

    None and "" are DIFFERENT here: None lists all campaigns, while ""  would
    filter to campaigns whose strategy_name is empty (the pre-binding rows).
    """
    v = query.get("strategy")
    if isinstance(v, list):
        v = v[0] if v else None
    return v or None


ROUTES = (
    ("GET", re.compile(r"^/api/config$"), lambda **kw: get_ui_config()),
    ("GET", re.compile(r"^/api/plugins$"), lambda **kw: get_plugins()),
    ("GET", re.compile(r"^/api/data/coverage$"), lambda **kw: get_data_coverage()),
    ("GET", re.compile(r"^/api/strategies$"), lambda **kw: list_strategies()),
    (
        "GET",
        re.compile(rf"^/api/strategies/(?P<name>{_STRATEGY_NAME_GROUP})$"),
        lambda name, **kw: get_strategy(name),
    ),
    (
        "POST",
        re.compile(rf"^/api/strategies/(?P<name>{_STRATEGY_NAME_GROUP})$"),
        lambda name, body, **kw: post_strategy(name, body),
    ),
    ("POST", re.compile(r"^/api/validate$"), lambda body, **kw: post_validate(body)),
    (
        "GET",
        re.compile(r"^/api/campaigns$"),
        lambda query=None, **kw: get_campaigns(_campaigns_query(query or {})),
    ),
    ("GET", re.compile(r"^/api/runs$"), lambda **kw: get_runs()),
    ("POST", re.compile(r"^/api/runs$"), lambda body, **kw: post_run(body)),
    (
        "GET",
        re.compile(rf"^/api/runs/(?P<rid>{_RUN_ID_GROUP})$"),
        lambda rid, **kw: get_run(rid),
    ),
    (
        "POST",
        re.compile(rf"^/api/runs/(?P<rid>{_RUN_ID_GROUP})/stop$"),
        lambda rid, **kw: post_run_stop(rid),
    ),
    ("GET", re.compile(r"^/api/reviews$"), lambda query, **kw: get_reviews(**_review_query(query))),
    (
        "GET",
        re.compile(r"^/api/generations$"),
        lambda query, **kw: get_generations(_generations_query(query)),
    ),
    (
        "GET",
        re.compile(r"^/api/evolution/members$"),
        lambda query, **kw: get_evolution_members(*_members_query(query)),
    ),
    (
        "GET",
        re.compile(r"^/api/evolution/replay$"),
        lambda query, **kw: get_evolution_replay(**_replay_query(query)),
    ),
)


def handle(method: str, path: str, query=None, body=None, headers=None) -> ApiResponse:
    """The single entry point. No socket, no http import — tests call this.

    1. path matches a route     -> else 404
    2. method matches that path -> else 405 with Allow:
    3. POST: Origin, if present, must be http://127.0.0.1:<port> or
       http://localhost:<port> -> else 403; Content-Type must be
       application/json -> else 415.
       THIS IS NOT A SECURITY BOUNDARY. It stops a random web page the
       operator has open from POSTing here (HTML forms cannot send
       application/json; a cross-origin fetch carries a rejected Origin). It
       stops NOTHING local: any process or user on this Mac can drive every
       endpoint. The boundary is the loopback bind plus a single-user machine.
    4. dispatch; ValueError/KeyError from a handler -> 400 with its message
       (they are argument errors); anything else -> 500, logged, generic body.
    """
    query = query or {}
    headers = headers or {}

    path_matches = [r for r in ROUTES if r[1].match(path)]
    if not path_matches:
        return _err(404, f"no route for {path}")

    allowed = sorted({r[0] for r in path_matches})
    method_matches = [r for r in path_matches if r[0] == method]
    if not method_matches:
        return _err(405, f"method {method} not allowed", headers={"Allow": ", ".join(allowed)})

    if method == "POST":
        origin = headers.get("Origin", headers.get("origin"))
        if origin is not None:
            port = _ORIGIN_PORT["port"]
            allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            if origin not in allowed_origins:
                return _err(403, f"origin {origin!r} is not allowed")
        ctype = headers.get("Content-Type", headers.get("content-type", ""))
        if ctype.split(";")[0].strip() != "application/json":
            return _err(415, "Content-Type must be application/json")
        if body is None:
            body = {}

    _method, path_re, fn = method_matches[0]
    m = path_re.match(path)
    kwargs = dict(m.groupdict())

    try:
        if method == "POST":
            return fn(body=body, query=query, **kwargs)
        return fn(query=query, **kwargs)
    except (ValueError, KeyError) as exc:
        return _err(400, str(exc))
    except Exception:  # noqa: BLE001 — logged, never a traceback to the browser
        logger.exception("ui: unhandled error dispatching %s %s", method, path)
        return _err(500, "internal error")
