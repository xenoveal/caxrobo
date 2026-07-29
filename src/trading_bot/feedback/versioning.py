"""
StrategyVersion registry and lineage — v0.3.0 Phase 5 (contract §3, §6).

THE ORACLE BOUNDARY applies here too (see feedback/records.py's docstring in
full): this module may not import backtest.walkforward, backtest.engine,
backtest.trials, framework.execute, or anything under evolution/. It reads and
writes strategy_versions rows and calls framework.graph's PURE, side-effect-
free hashing/serialization functions — none of which can cause a trade to
exist.

CONTENT ADDRESSING (A5). version_id = framework.graph.graph_hash(graph)[:16].
Phase 3 already publishes a canonical, order-independent, default-resolved
hash (graph.py's canonical_dict/graph_hash), so this module does NOT invent a
second canonicalization — the plan's fallback path ("if graph.py publishes
only to_dict, hash it yourself") does not apply, because graph_hash() exists.
graph.py's OWN short_hash() (12 hex) is documented as "never a persisted key"
(graph.py comment), so version_id deliberately uses a DIFFERENT truncation
length (16 hex) of the SAME underlying 64-hex digest, never short_hash().

REPRODUCIBILITY IS RECORDED, NOT PINNED (A6). config.py is mutable
module-level global state; see config_snapshot()'s docstring for the
consequence and the mitigation.

Trade objects are accepted DUCK-TYPED (attribute access only) in
stamp_trades(), exactly as records.py accepts them — dataclasses.replace()
operates on the instance's own class without this module importing
backtest.engine.Trade, which is what keeps the import ban enforceable.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace

from trading_bot import config
from trading_bot.data import statestore
from trading_bot.framework import graph as fgraph

logger = logging.getLogger("trading_bot")

TABLE = "strategy_versions"

# Simulation-affecting constants: a version's config_snapshot records these AT
# CREATION TIME because each one changes the trades a graph produces (directly,
# or by changing what a plug-in resolves its defaults to). Every registered
# plug-in's ParamSpec.default reads one of these, or the executor itself reads
# one (regime/trigger/cost). NOT snapshotted are paths, symbol universes, UI
# config, and REVIEW_* diagnostic thresholds — see VERSION_CONFIG_IGNORED.
VERSION_CONFIG_KEYS = (
    "FEE_PCT", "SLIPPAGE_PCT", "FUNDING_PCT_PER_DAY",
    "REGIME_TIMEFRAME", "SIGNAL_PATTERN_TIMEFRAME", "SIGNAL_TRIGGER_TIMEFRAME",
    "ADX_PERIOD", "ADX_TREND_THRESHOLD", "ATR_PERCENTILE_WINDOW",
    "ATR_EXTREME_PERCENTILE", "REGIME_MIN_BARS", "ATR_STOP_PERIOD",
    "ATR_STOP_MULTIPLE", "RR_FLOOR", "MAX_HOLD_BARS_TRIGGER", "PIVOT_SPAN",
    "PATTERN_LOOKBACK_BARS", "PATTERN_MAX_AGE_BARS", "DONCHIAN_ENTRY_PERIOD",
    "DONCHIAN_TREND_PERIOD", "DONCHIAN_MIN_BARS", "DONCHIAN_TARGET_ENABLED",
    "TRAIL_ENABLED", "TRAIL_ATR_MULTIPLE", "BB_PERIOD", "BB_STD",
    "FADE_ENABLED", "FADE_STRETCH_MAX_AGE_BARS", "VOLUME_LOOKBACK",
    "VOLUME_HIGH_RATIO", "BREAKOUT_TRIGGER_LOOKBACK_BARS",
    "HS_SHOULDER_TOLERANCE", "HS_HEAD_MIN_PROMINENCE",
    "TRIANGLE_MIN_PIVOTS_PER_SIDE", "TRIANGLE_MIN_CONVERGENCE",
    "TRIANGLE_MAX_WIDTH_BARS", "TRIANGLE_MIN_WIDTH_BARS",
    "TRIANGLE_CONTAINMENT_TOL", "FLAG_POLE_WINDOW_BARS", "FLAG_POLE_MIN_PCT",
    "FLAG_CONSOL_MIN_BARS", "FLAG_CONSOL_MAX_BARS", "FLAG_MAX_RETRACE",
    "WF_TRAIN_DAYS", "WF_TEST_DAYS", "WF_OOS_DAYS", "WF_MIN_TRADES",
    "TARGET_ANN_RETURN", "RR_TARGET_MIN", "PNL_ATTRIBUTION_MODE",
    "MACD_FAST_PERIOD", "MACD_SLOW_PERIOD", "MACD_SIGNAL_PERIOD",
    "MACD_MIN_BARS", "MACD_CONFIRM_MIN_HIST",
    "VOLUME_CONFIRM_MIN_RATIO", "VOLUME_CONFIRM_REQUIRE_DEFINED",
    # Phase 8 (landed concurrently with this phase): detector geometry/
    # indicator defaults, each the default of a declared ParamSpec on a
    # detector plug-in (contract §9) -- simulation-affecting exactly like the
    # Phase 3/4 pattern constants above.
    "RSI_PERIOD",
    "CUP_MIN_WIDTH_BARS", "CUP_MAX_WIDTH_BARS", "CUP_MIN_DEPTH", "CUP_MAX_DEPTH",
    "CUP_RIM_TOLERANCE", "CUP_ROUND_BAND", "CUP_MIN_BASE_BARS",
    "CUP_HANDLE_MIN_BARS", "CUP_HANDLE_MAX_BARS", "CUP_HANDLE_MAX_RETRACE",
    "DOUBLE_TOLERANCE", "DOUBLE_MIN_SEPARATION_BARS", "DOUBLE_MAX_GAP_BARS",
    "DOUBLE_MIN_TROUGH_DEPTH",
    # v0.3.2 WS-A correctness gates (code review 2026-07-29): ParamSpec
    # defaults on the same detector, simulation-affecting exactly like the
    # four DOUBLE_* entries above.
    "DOUBLE_MAX_TROUGH_DEPTH", "DOUBLE_DOMINANCE_TOL",
    "DOUBLE_PRIOR_TREND_LOOKBACK_BARS", "DOUBLE_PRIOR_TREND_MIN_MOVE",
    "HS_NECKLINE_SLOPED", "HS_TIME_SYMMETRY_TOL", "HS_VOLUME_TAPER_REQUIRED",
    "TRIANGLE_FLAT_SLOPE_TOL", "WEDGE_MIN_SLOPE",
    "RSI_DIV_LOOKBACK_BARS", "RSI_DIV_PIVOT_MATCH_BARS",
    "RSI_DIV_MIN_SEPARATION_BARS", "RSI_DIV_MAX_SEPARATION_BARS",
    "RSI_DIV_OVERBOUGHT", "RSI_DIV_OVERSOLD",
    "WYCKOFF_RANGE_BARS", "WYCKOFF_RANGE_MAX_WIDTH_PCT",
    "WYCKOFF_PROBE_MIN_PCT", "WYCKOFF_PROBE_VOL_RATIO",
)

# NOT snapshotted: paths, symbol lists, UI host/port, benchmark-null config,
# correlation research constants, and the REVIEW_* diagnostic thresholds —
# none changes the trades a graph produces. Classified explicitly so the
# completeness test (test_every_config_constant_is_classified) can tell
# "irrelevant" from "forgotten".
VERSION_CONFIG_IGNORED = (
    "SYMBOLS", "TIMEFRAMES", "BACKFILL_START", "DB_PATH", "STATE_DB_PATH",
    "STALENESS_INTERVALS", "MAX_RISK_PCT", "COST_RATIO_CEILING",
    "CORRELATION_TIMEFRAME", "CORRELATION_START", "CORRELATION_MIN_OVERLAP_BARS",
    "CORRELATION_ANCHOR_SYMBOL", "CORRELATION_SELECT_N",
    "CORRELATION_MIN_QUOTE_VOLUME_USD", "CORRELATION_MAX_BTC_BETA",
    "CORRELATION_EFFECTIVE_N_MIN_RATIO", "RESEARCH_SYMBOLS",
    "BENCHMARK_TIMEFRAME", "BENCHMARK_REBALANCE", "BENCHMARK_CHARGE_FEES",
    "STRATEGY_DIR", "FRAMEWORK_PLUGIN_PACKAGE", "FRAMEWORK_CACHE_ENABLED",
    "REVIEW_TIMEFRAME", "REVIEW_MIN_BAR_COVERAGE", "REVIEW_TP_CAPTURE_GOOD",
    "REVIEW_TP_UNREACHABLE_FRAC", "REVIEW_SL_SLACK_MAX", "REVIEW_SL_NEAR_MISS",
    "REVIEW_PACE_MIN_TRADES", "REVIEW_PACE_MIN_DAYS", "REVIEW_STOP_DOMINANCE",
    "REVIEW_TIME_DOMINANCE", "REVIEW_DEAD_WEIGHT_COVERAGE",
    "REVIEW_REFINE_STEP_FRAC", "REVIEW_FORWARD_MIN_DAYS",
    "REVIEW_LEGACY_VERSION_ID",
    # Phase 8 (landed concurrently): report-only diagnostics, same treatment
    # as REVIEW_* above -- they label a report, never a strategy's trades.
    "DETECTOR_MIN_EVENTS_FOR_REPORT", "DETECTOR_REPORT_HOLDOUT_GUARD_DAYS",
    "DETECTOR_REPORT_CAMPAIGN",
    # Phase 6 (landed concurrently): evolution/*.py's SEARCH PROCESS config —
    # population size, generation count, worker count, DB retry knobs. These
    # govern how many candidates are tried and how the search is run, never
    # what trades a GIVEN graph produces, so they do not belong in a
    # per-strategy-version reproducibility snapshot.
    "EVO_POPULATION", "EVO_GENERATIONS", "EVO_ELITES", "EVO_FINALISTS",
    "EVO_TOURNAMENT_K", "EVO_BOOL_FLIP_P", "EVO_JITTER_NODES",
    "EVO_JITTER_SIGMA", "EVO_GRAPH_EDIT_SHARE", "EVO_MIN_UNIQUE_FRACTION",
    "EVO_DEDUP_MAX_REDRAWS", "EVO_STRICT_MUTATORS", "EVO_WORKERS",
    "EVO_BUDGET_HOURS", "EVO_WINDOW_DAYS", "EVO_WINDOW_JITTER",
    "EVO_TRAIN_START", "EVO_TRAIN_END", "EVO_DB_RETRIES",
    "EVO_DB_RETRY_SLEEP_S", "EVO_DB_BUSY_TIMEOUT_MS",
    # Phase 7 (landed concurrently): the builder UI's bind host/port. Neither
    # changes what trades a graph produces — they configure a local HTTP
    # server, not the strategy — so they are classified IGNORED rather than
    # simulation-affecting, the same treatment as STRATEGY_DIR above.
    "UI_HOST", "UI_PORT",
    # Phase 9 (landed last): the campaign PROTOCOL — which span is the holdout,
    # how large the search is, which symbols the gate pools, the champion trade
    # floor, the northstar target, where the report is written. Every one of
    # these governs how a strategy is MEASURED, never what trades a given graph
    # produces, so none belongs in a per-strategy-version reproducibility
    # snapshot (same argument as EVO_* above). Note in particular that
    # HOLDOUT_* is protocol, not simulation: moving the holdout changes which
    # bars the verdict is computed on, not the bars a graph replays.
    "HOLDOUT_START_MS", "HOLDOUT_END_MS", "HOLDOUT_DAYS", "HOLDOUT_LOCKED",
    "CAMPAIGN_EVOLVE_START_MS", "CAMPAIGN_SYMBOLS", "CAMPAIGN_POPULATION_SIZE",
    "CAMPAIGN_GENERATIONS", "CAMPAIGN_WALL_CLOCK_BUDGET_HOURS",
    "CAMPAIGN_PATIENCE_GENERATIONS", "CAMPAIGN_SEED",
    "CAMPAIGN_CHECKPOINT_EVERY", "CAMPAIGN_MIN_CHAMPION_TRADES",
    "CAMPAIGN_NORTHSTAR_ANN_RETURN", "CAMPAIGN_REPORT_DIR",
)


@dataclass(frozen=True)
class StrategyVersion:
    version_id: str
    parent_id: str | None
    label: str
    graph_json: str
    graph_hash: str
    schema_version: str
    config_hash: str
    config_snapshot: dict
    provenance: dict
    created_ts: int


_COLUMNS = (
    "version_id", "parent_id", "label", "graph_json", "graph_hash",
    "schema_version", "config_hash", "config_snapshot", "provenance",
    "created_ts",
)


def ensure_schema(conn) -> None:
    """Create strategy_versions and its indexes if absent (idempotent)."""
    with statestore._db_lock:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                version_id TEXT PRIMARY KEY, parent_id TEXT, label TEXT NOT NULL,
                graph_json TEXT NOT NULL, graph_hash TEXT NOT NULL, schema_version TEXT NOT NULL,
                config_hash TEXT NOT NULL, config_snapshot TEXT NOT NULL,
                provenance TEXT NOT NULL, created_ts INTEGER NOT NULL
            )
            """
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_version_parent ON {TABLE} (parent_id)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_version_created ON {TABLE} (created_ts)")
        conn.commit()


def version_id_for(graph) -> str:
    """16-hex content id — see module docstring for why this is NOT short_hash()."""
    return fgraph.graph_hash(graph)[:16]


def _row_to_version(row: tuple) -> StrategyVersion:
    d = dict(zip(_COLUMNS, row))
    d["config_snapshot"] = json.loads(d["config_snapshot"])
    d["provenance"] = json.loads(d["provenance"])
    return StrategyVersion(**d)


def _select_row(conn, version_id: str) -> tuple | None:
    with statestore._db_lock:
        row = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM {TABLE} WHERE version_id = ?",
            (version_id,),
        ).fetchone()
    return row


def register_version(
    conn, graph, *, parent_id: str | None = None, label: str = "",
    provenance: dict | None = None, created_ts: int | None = None,
) -> StrategyVersion:
    """Register a graph, or return the existing row unchanged if its content
    hash already exists (idempotent re-registration is information, not an
    error — a mutator re-deriving an existing variant is observable this way).

    Raises:
        ValueError: If parent_id equals this graph's own version_id.
    """
    ensure_schema(conn)
    vid = version_id_for(graph)
    if parent_id == vid:
        raise ValueError(
            f"parent_id equals version_id ({vid}); a version cannot be its own parent"
        )
    existing_row = _select_row(conn, vid)
    if existing_row is not None:
        logger.info("re-registered identical graph version_id=%s", vid)
        return _row_to_version(existing_row)

    if created_ts is None:
        created_ts = int(time.time() * 1000)
    if provenance is None:
        provenance = {"source": "manual"}

    graph_json = json.dumps(graph.to_dict(), sort_keys=True, separators=(",", ":"))
    snapshot = config_snapshot()
    version = StrategyVersion(
        version_id=vid, parent_id=parent_id, label=label, graph_json=graph_json,
        graph_hash=fgraph.graph_hash(graph), schema_version=str(graph.schema_version),
        config_hash=config_hash(snapshot), config_snapshot=snapshot,
        provenance=provenance, created_ts=created_ts,
    )
    with statestore._db_lock:
        conn.execute(
            f"INSERT INTO {TABLE} ({', '.join(_COLUMNS)}) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                version.version_id, version.parent_id, version.label,
                version.graph_json, version.graph_hash, version.schema_version,
                version.config_hash, json.dumps(version.config_snapshot, sort_keys=True),
                json.dumps(version.provenance, sort_keys=True), version.created_ts,
            ),
        )
        conn.commit()
    return version


def get_version(conn, version_id: str) -> StrategyVersion:
    """Raises KeyError on an unknown id — a silent None would let a forward
    test run the wrong strategy."""
    ensure_schema(conn)
    row = _select_row(conn, version_id)
    if row is None:
        raise KeyError(f"unknown strategy_version {version_id!r}")
    return _row_to_version(row)


def list_versions(
    conn, *, parent_id: str | None = None, limit: int | None = None
) -> list:
    ensure_schema(conn)
    query = f"SELECT {', '.join(_COLUMNS)} FROM {TABLE}"
    params: list = []
    if parent_id is not None:
        query += " WHERE parent_id = ?"
        params.append(parent_id)
    query += " ORDER BY created_ts ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with statestore._db_lock:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_version(r) for r in rows]


def load_graph(conn, version_id: str):
    """Reload a registered version as a StrategyGraph.

    graph_hash(load_graph(conn, register_version(conn, g).version_id)) ==
    graph_hash(g) — this round-trip is what makes a version RE-RUNNABLE.

    Raises:
        KeyError: On an unknown version_id.
    """
    version = get_version(conn, version_id)
    payload = json.loads(version.graph_json)
    graph = fgraph.StrategyGraph.from_dict(payload)
    fgraph.validate(graph)
    return graph


def lineage(conn, version_id: str) -> list:
    """Root-first ancestry chain, [root, ..., version_id]'s version.

    Raises:
        KeyError: On an unknown version_id (from get_version).
        ValueError: If a cycle is detected — a corrupt parent_id chain must
            not hang the caller.
    """
    chain: list = []
    seen: set = set()
    current = version_id
    cap = 10_000  # hard iteration cap; a legitimate lineage is never this deep
    for _ in range(cap):
        if current in seen:
            raise ValueError(f"cycle detected in lineage at version_id={current!r}")
        seen.add(current)
        v = get_version(conn, current)
        chain.append(v)
        if v.parent_id is None:
            break
        current = v.parent_id
    else:  # pragma: no cover - defensive; cap reached without terminating
        raise ValueError(f"lineage of {version_id!r} exceeded {cap} hops; likely a cycle")
    return list(reversed(chain))


def config_snapshot() -> dict:
    """Every VERSION_CONFIG_KEYS constant, read at CALL time via getattr.

    config.py is MUTABLE MODULE-LEVEL GLOBAL STATE. A strategy graph does not
    carry the cost model, the tier timeframes, the ATR stop multiple or the
    regime thresholds — yet each changes the trades the graph produces. This
    snapshot records their values AT VERSION-CREATION TIME so a later re-run
    is CHECKABLE. It does NOT pin them: re-running version X under a different
    FEE_PCT silently produces different trades, and verify_reproducible() is
    the only thing that will tell you. Named as a residual hole rather than
    papered over (A6).

    Keys absent from the running build (e.g. a constant a later phase removed)
    are skipped rather than raising — a version created before that phase
    existed should record what existed then, not crash on load.
    """
    snap = {}
    for key in VERSION_CONFIG_KEYS:
        if hasattr(config, key):
            snap[key] = _json_primitive(getattr(config, key))
    return snap


def _json_primitive(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def config_hash(snapshot: dict) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def verify_reproducible(conn, version_id: str) -> dict:
    """Compare a version's recorded config_snapshot against the LIVE config.

    NEVER RAISES on a mismatch — it returns a report; the CLI prints a loud
    WARNING and proceeds, because the operator must be able to re-test an old
    version under a corrected cost model KNOWING that is what they are doing.

    Raises:
        KeyError: On an unknown version_id (from get_version) — that is an
            identity error, not a drift measurement, and stays loud.
    """
    version = get_version(conn, version_id)
    recorded = version.config_snapshot
    live = config_snapshot()
    diff = {}
    for key in sorted(set(recorded) | set(live)):
        if recorded.get(key) != live.get(key):
            diff[key] = {"recorded": recorded.get(key), "live": live.get(key)}
    missing = sorted(set(recorded) - set(live))
    added = sorted(set(live) - set(recorded))
    return {
        "version_id": version_id,
        "config_hash_matches": config_hash(live) == version.config_hash,
        "diff": diff,
        "missing": missing,
        "added": added,
        "schema_version_matches": str(fgraph.SCHEMA_VERSION) == version.schema_version,
    }


def stamp_trades(trades, version_id: str) -> tuple:
    """Return NEW Trade objects with strategy_version set — dataclasses.replace
    operates on each trade's own class, so this module never imports
    backtest.engine.Trade (see module docstring)."""
    return tuple(replace(t, strategy_version=version_id) for t in trades)
