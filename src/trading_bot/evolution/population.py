"""
Campaign / generation / member state, hashing, and the three state.db tables
(v0.3.0 Phase 6; contract §6 assigns `campaigns`, `generations` and
`population_members` to this module).

WHY THE DDL LIVES HERE. Contract §6: each table's schema is owned by the module
that uses it, via CREATE TABLE IF NOT EXISTS, so there is no central migration
file to drift out of sync. Mirrors backtest/trials.py's `ensure_schema`.

WHY NOT builtin hash(). PYTHONHASHSEED salts str hashing per process, and a
spawned worker gets a DIFFERENT salt, so a campaign seeded from hash() would be
unreproducible in a way that looks exactly like nondeterministic code. Every
derivation here is blake2b or sha256.

WHO WRITES WHAT. The PARENT process is the only writer of these three tables.
Workers write only `trial_ledger` rows, on their own connection (trap 1 in the
plan's Architecture section). No function in this module ever accepts a
connection created in another process.

TIMESTAMPS. Epoch milliseconds, UTC, like every other column in state.db.
"""

import dataclasses
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trading_bot import config
from trading_bot.data import storage
from trading_bot.data.statestore import _db_lock
from trading_bot.framework import graph as fgraph

logger = logging.getLogger("trading_bot")

__all__ = (
    "Campaign",
    "Generation",
    "Member",
    "TIER_UNKNOWN",
    "derive_seed",
    "ensure_schema",
    "finish_campaign",
    "finish_generation",
    "graph_hash",
    "insert_campaign",
    "insert_generation",
    "insert_members",
    "last_completed_generation",
    "load_campaign",
    "member_rng",
    "seen_graph_hashes",
    "top_members",
    "update_member_result",
    "write_with_retry",
)

TIER_UNKNOWN = ""  # a member row before its result lands

# Roles a member row can carry. 'seed' is generation 0's unmutated ancestor;
# 'finalist' rows are the audit round's re-evaluations (A3), which are separate
# rows rather than an overwrite so the audit is auditable next to the training
# score it is meant to check.
ROLES = ("seed", "offspring", "elite", "finalist")


# --------------------------------------------------------------------------- #
# Hashing and seed derivation
# --------------------------------------------------------------------------- #


def graph_hash(graph) -> str:
    """Content hash of a StrategyGraph — Phase 3's, never a second definition.

    Delegated rather than reimplemented because `trial_ledger`, Phase 5's
    `strategy_versions` and `population_members` must all key on the SAME
    identity: two graphs that hash equal have to behave identically
    (framework/graph.py's A5).
    """
    return fgraph.graph_hash(graph)


def derive_seed(*parts: Any) -> int:
    """Deterministic 63-bit integer from any tuple of JSON-able parts.

    blake2b over canonical JSON, NOT builtin hash(): str hashing is salted per
    process by PYTHONHASHSEED, and spawned workers get a different salt, so a
    hash()-derived campaign would replay differently on every run while looking
    like a race condition.
    """
    payload = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def member_seed(campaign_seed: int, gen_index: int, member_index: int) -> int:
    """The RNG seed for one member.

    Derived from (campaign_seed, gen_index, member_index) rather than drawn from
    one shared stream advanced in loop order, so changing EVO_ELITES or the
    population size does not reshuffle every downstream member — a campaign
    stays comparable to its own variants.
    """
    return derive_seed("member", campaign_seed, gen_index, member_index)


def member_rng(campaign_seed: int, gen_index: int, member_index: int):
    """A freshly seeded random.Random for one member (A7: CPython 3.11.6)."""
    import random

    return random.Random(member_seed(campaign_seed, gen_index, member_index))


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Campaign:
    """One evolution run, and everything a replay or a resume needs.

    Attributes:
        seed: THE one number a replay needs. Every member RNG derives from it.
        seed_graph_json: The seed graph inline, so `--resume` never has to find
            the original file again (a moved or edited seed file would otherwise
            silently change what a resumed campaign is evolving).
        train_start_ms / train_end_ms: The oracle's hard span ceiling. No
            generation may score a bar at or beyond train_end_ms (A5) — that is
            Phase 9's holdout.
        audit_start_ms / audit_end_ms: The FIXED audit window, declared here at
            campaign start precisely so it cannot later be chosen to flatter a
            winner (A3). Fitness on jittered per-generation windows is not
            comparable across generations; the audit round is.
        config_json: Every EVO_* value ACTUALLY used, not the module defaults, so
            a report can never quote a constant the run did not use.
    """

    campaign_id: str
    seed: int
    seed_graph_hash: str
    seed_graph_json: str
    config_json: str
    symbols: tuple[str, ...]
    train_start_ms: int
    train_end_ms: int
    audit_start_ms: int
    audit_end_ms: int
    population: int
    generations: int
    started_ts: int
    status: str = "running"
    finished_ts: int | None = None
    # label / strategy_name are the OPERATOR's identifiers; campaign_id stays
    # the engine's. campaign_id embeds today's date (_campaign_id), so it is not
    # stable across days for the same seed+graph and cannot be the handle a
    # human uses to continue a campaign tomorrow. Default '' so campaigns
    # written before these columns existed still load.
    label: str = ""
    strategy_name: str = ""


@dataclass(frozen=True)
class Generation:
    """One generation's window, counts and outcome.

    `trials_cumulative` is the LEDGER count after this generation, not the
    generation's own evaluations: contract §4 charges the DSR the campaign's
    cumulative count, and recording the running total per generation is what
    makes that auditable after the fact.

    `db_retries` exists so "ledger contention is negligible" stays a
    measurement rather than a claim.
    """

    campaign_id: str
    gen_index: int
    window_start_ms: int
    window_end_ms: int
    population_size: int
    started_ts: int
    n_evaluated: int = 0
    n_errors: int = 0
    n_unique_graphs: int = 0
    trials_cumulative: int = 0
    best_member_id: str | None = None
    best_fitness: float | None = None
    db_retries: int = 0
    wall_seconds: float | None = None
    finished_ts: int | None = None


@dataclass(frozen=True)
class Member:
    """One candidate: its lineage, its genome, and (once scored) its result.

    `mutation_json` carries the (node, param, old, new) diff beside the graph
    because the graph alone does not say WHICH move produced it — and "which
    mutation helped" is the only question a population search can answer about
    itself.

    `rng_seed` is stored so a single member can be re-bred in isolation without
    replaying the whole campaign.
    """

    member_id: str
    campaign_id: str
    gen_index: int
    member_index: int
    graph_hash: str
    graph_json: str
    rng_seed: int
    role: str
    parent_member_id: str | None = None
    mutator: str = ""
    mutation_json: str = "{}"
    # Result fields, all None until the oracle answers.
    fitness: float | None = None
    tier: str = TIER_UNKNOWN
    excess_sharpe: float | None = None
    excess_ann_return: float | None = None
    sharpe: float | None = None
    dsr: float | None = None
    ann_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    n_trades: int | None = None
    bench_sharpe: float | None = None
    bench_ann_return_pct: float | None = None
    n_trials_used: int | None = None
    gate_json: str | None = None
    window_start_ms: int | None = None
    window_end_ms: int | None = None
    eval_seconds: float | None = None
    error: str = ""
    # Not persisted: the live graph object, kept beside the row while the parent
    # breeds from it. Excluded from every INSERT by name.
    graph: Any = field(default=None, compare=False, repr=False)


_MEMBER_COLUMNS = tuple(
    f.name for f in dataclasses.fields(Member) if f.name != "graph"
)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS campaigns (
      campaign_id TEXT PRIMARY KEY,
      label TEXT NOT NULL DEFAULT '',
      strategy_name TEXT NOT NULL DEFAULT '',
      seed INTEGER NOT NULL,
      seed_graph_hash TEXT NOT NULL,
      seed_graph_json TEXT NOT NULL,
      config_json TEXT NOT NULL,
      symbols_json TEXT NOT NULL,
      train_start_ms INTEGER NOT NULL,
      train_end_ms INTEGER NOT NULL,
      audit_start_ms INTEGER NOT NULL,
      audit_end_ms INTEGER NOT NULL,
      population INTEGER NOT NULL,
      generations INTEGER NOT NULL,
      started_ts INTEGER NOT NULL,
      finished_ts INTEGER,
      status TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS generations (
      campaign_id TEXT NOT NULL,
      gen_index INTEGER NOT NULL,
      window_start_ms INTEGER NOT NULL,
      window_end_ms INTEGER NOT NULL,
      population_size INTEGER NOT NULL,
      n_evaluated INTEGER NOT NULL,
      n_errors INTEGER NOT NULL DEFAULT 0,
      n_unique_graphs INTEGER NOT NULL,
      trials_cumulative INTEGER NOT NULL,
      best_member_id TEXT,
      best_fitness REAL,
      db_retries INTEGER NOT NULL DEFAULT 0,
      wall_seconds REAL,
      started_ts INTEGER NOT NULL,
      finished_ts INTEGER,
      PRIMARY KEY (campaign_id, gen_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS population_members (
      member_id TEXT PRIMARY KEY,
      campaign_id TEXT NOT NULL,
      gen_index INTEGER NOT NULL,
      member_index INTEGER NOT NULL,
      graph_hash TEXT NOT NULL,
      graph_json TEXT NOT NULL,
      rng_seed INTEGER NOT NULL,
      role TEXT NOT NULL,
      parent_member_id TEXT,
      mutator TEXT NOT NULL DEFAULT '',
      mutation_json TEXT NOT NULL DEFAULT '{}',
      fitness REAL,
      tier TEXT,
      excess_sharpe REAL,
      excess_ann_return REAL,
      sharpe REAL,
      dsr REAL,
      ann_return_pct REAL,
      max_drawdown_pct REAL,
      n_trades INTEGER,
      bench_sharpe REAL,
      bench_ann_return_pct REAL,
      n_trials_used INTEGER,
      gate_json TEXT,
      window_start_ms INTEGER,
      window_end_ms INTEGER,
      eval_seconds REAL,
      error TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pm_campaign_gen "
    "ON population_members(campaign_id, gen_index)",
    "CREATE INDEX IF NOT EXISTS idx_pm_graph_hash ON population_members(graph_hash)",
    # A label is the operator's handle on a campaign and must resolve to exactly
    # one campaign_id, or "continue my campaign" has no single answer. Partial
    # index: pre-label rows carry '' and must not collide with each other.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_campaigns_label "
    "ON campaigns(label) WHERE label != ''",
    "CREATE INDEX IF NOT EXISTS idx_campaigns_strategy ON campaigns(strategy_name)",
)

# Columns added after the original Phase 6 schema shipped. ensure_schema's
# CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a state.db
# written before these columns existed needs an explicit ALTER.
_CAMPAIGN_MIGRATIONS = (
    ("label", "ALTER TABLE campaigns ADD COLUMN label TEXT NOT NULL DEFAULT ''"),
    ("strategy_name",
     "ALTER TABLE campaigns ADD COLUMN strategy_name TEXT NOT NULL DEFAULT ''"),
)


def ensure_schema(conn) -> None:
    """Create the three Phase 6 tables and their indexes if absent (idempotent).

    Also applies _CAMPAIGN_MIGRATIONS. The ALTERs run BEFORE the index
    statements: idx_campaigns_label names a column that an old state.db does
    not have yet, so creating the indexes first would fail on exactly the
    databases the migration exists to repair.
    """
    with _db_lock:
        conn.execute(_DDL[0])
        existing = {row[1] for row in conn.execute("PRAGMA table_info(campaigns)")}
        for column, stmt in _CAMPAIGN_MIGRATIONS:
            if column not in existing:
                conn.execute(stmt)
        for stmt in _DDL[1:]:
            conn.execute(stmt)
        conn.commit()


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


def write_with_retry(fn: Callable[[], Any], *, retries: int | None = None,
                     sleep_s: float | None = None, sleep=time.sleep) -> tuple[Any, int]:
    """Run a write, retrying only CONTENTION errors. Returns (result, n_retries).

    The classification is storage.is_transient_db_error's and nothing else's: a
    bare `except sqlite3.OperationalError` would swallow "no such table" and
    "database disk image is malformed", turning a corrupt state.db into a slow
    silent no-op. Locked/busy retries; everything else propagates immediately.

    Args:
        fn: The zero-argument write to attempt.
        retries: Attempts after the first (default config.EVO_DB_RETRIES).
        sleep_s: Linear backoff base (default config.EVO_DB_RETRY_SLEEP_S).
        sleep: Injectable so tests do not actually wait.
    """
    if retries is None:
        retries = config.EVO_DB_RETRIES
    if sleep_s is None:
        sleep_s = config.EVO_DB_RETRY_SLEEP_S
    attempt = 0
    while True:
        try:
            return fn(), attempt
        except sqlite3.OperationalError as exc:
            if not storage.is_transient_db_error(exc):
                raise
            attempt += 1
            if attempt > retries:
                raise
            logger.warning(
                "state.db contention (%s); retry %d/%d", exc, attempt, retries
            )
            sleep(sleep_s * attempt)


# --------------------------------------------------------------------------- #
# CRUD — parent process only
# --------------------------------------------------------------------------- #


def insert_campaign(conn, campaign: Campaign) -> None:
    """Write the campaigns row. Idempotent per campaign_id via INSERT OR IGNORE."""

    def _write():
        with _db_lock:
            conn.execute(
                "INSERT OR IGNORE INTO campaigns (campaign_id, label, strategy_name, "
                "seed, seed_graph_hash, "
                "seed_graph_json, config_json, symbols_json, train_start_ms, "
                "train_end_ms, audit_start_ms, audit_end_ms, population, generations, "
                "started_ts, finished_ts, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    campaign.campaign_id, campaign.label, campaign.strategy_name,
                    campaign.seed, campaign.seed_graph_hash,
                    campaign.seed_graph_json, campaign.config_json,
                    json.dumps(list(campaign.symbols)),
                    campaign.train_start_ms, campaign.train_end_ms,
                    campaign.audit_start_ms, campaign.audit_end_ms,
                    campaign.population, campaign.generations,
                    campaign.started_ts, campaign.finished_ts, campaign.status,
                ),
            )
            conn.commit()

    write_with_retry(_write)


_CAMPAIGN_SELECT = (
    "SELECT campaign_id, seed, seed_graph_hash, seed_graph_json, config_json, "
    "symbols_json, train_start_ms, train_end_ms, audit_start_ms, audit_end_ms, "
    "population, generations, started_ts, finished_ts, status, label, strategy_name "
    "FROM campaigns"
)


def _row_to_campaign(row) -> Campaign:
    return Campaign(
        campaign_id=row[0], seed=row[1], seed_graph_hash=row[2], seed_graph_json=row[3],
        config_json=row[4], symbols=tuple(json.loads(row[5])),
        train_start_ms=row[6], train_end_ms=row[7],
        audit_start_ms=row[8], audit_end_ms=row[9],
        population=row[10], generations=row[11],
        started_ts=row[12], finished_ts=row[13], status=row[14],
        label=row[15] or "", strategy_name=row[16] or "",
    )


def load_campaign(conn, campaign_id: str) -> Campaign | None:
    """Read one campaigns row back as a Campaign, or None."""
    with _db_lock:
        row = conn.execute(
            f"{_CAMPAIGN_SELECT} WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
    return None if row is None else _row_to_campaign(row)


def campaign_by_label(conn, label: str) -> Campaign | None:
    """Resolve an operator's campaign label to its Campaign, or None.

    The label is the handle the UI and CLI carry; campaign_id embeds the
    creation date and is not something an operator can retype tomorrow.
    """
    if not label:
        return None
    with _db_lock:
        row = conn.execute(
            f"{_CAMPAIGN_SELECT} WHERE label = ?", (label,)
        ).fetchone()
    return None if row is None else _row_to_campaign(row)


def list_campaigns(conn, *, strategy_name: str | None = None) -> list[dict]:
    """Campaigns with their progress, newest first, for the UI's dropdowns.

    `generations_done` counts FINISHED generations (finished_ts IS NOT NULL),
    the same predicate last_completed_generation resumes from — so the number
    an operator reads is the number a continuation will build on, and a
    generation interrupted mid-flight is never counted as work completed.

    Args:
        strategy_name: Restrict to one strategy. A campaign is an evolution
            history OF a strategy, so the Run card filters by the loaded one.
    """
    sql = f"{_CAMPAIGN_SELECT}"
    params: list = []
    if strategy_name is not None:
        sql += " WHERE strategy_name = ?"
        params.append(strategy_name)
    sql += " ORDER BY started_ts DESC"
    with _db_lock:
        rows = conn.execute(sql, params).fetchall()
        # trial_ledger belongs to Phase 1 (backtest/trials.py), not to this
        # module's ensure_schema — on a state.db that has never run a backtest
        # it is simply absent, and a listing must not 500 over a count.
        has_ledger = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trial_ledger'"
        ).fetchone() is not None
        out = []
        for row in rows:
            c = _row_to_campaign(row)
            done = conn.execute(
                "SELECT COUNT(*) FROM generations "
                "WHERE campaign_id = ? AND finished_ts IS NOT NULL",
                (c.campaign_id,),
            ).fetchone()[0]
            trials = conn.execute(
                "SELECT COUNT(*) FROM trial_ledger WHERE campaign = ?",
                (c.campaign_id,),
            ).fetchone()[0] if has_ledger else 0
            out.append(
                {
                    "campaign_id": c.campaign_id, "label": c.label,
                    "strategy_name": c.strategy_name, "status": c.status,
                    "seed": c.seed, "seed_graph_hash": c.seed_graph_hash,
                    "symbols": list(c.symbols), "population": c.population,
                    "generations": c.generations, "generations_done": done,
                    "trials": trials, "started_ts": c.started_ts,
                    "finished_ts": c.finished_ts,
                }
            )
    return out


def adopt_ledger_rows(conn, *, from_key: str, to_key: str) -> int:
    """Re-key trial_ledger rows from an operator label onto a campaign_id.

    A gate run started before this campaign's first evolve charged the ledger
    under the label the operator typed, because no campaign_id existed yet.
    Leaving those rows behind would understate how many evaluations the
    strategy has actually consumed, and contract §4 deflates the DSR by the
    campaign's CUMULATIVE count — so they are adopted, not abandoned. Returns
    the number of rows moved.

    COUPLING: trial_ledger is Phase 1's table (backtest/trials.py). Touched here
    because campaign identity is what changed; there is one site to delete if an
    accessor is published.
    """
    if not from_key or not to_key or from_key == to_key:
        return 0
    with _db_lock:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trial_ledger'"
        ).fetchone() is None:
            return 0
        n = conn.execute(
            "SELECT COUNT(*) FROM trial_ledger WHERE campaign = ?", (from_key,)
        ).fetchone()[0]
        if n:
            conn.execute(
                "UPDATE trial_ledger SET campaign = ? WHERE campaign = ?",
                (to_key, from_key),
            )
            conn.commit()
    return n


def set_generations(conn, campaign_id: str, generations: int) -> None:
    """Raise a campaign's generation ceiling so it can keep improving.

    The runner's loop is `range(start_gen, campaign.generations)`, so a campaign
    that finished every declared generation resumes into an EMPTY range and does
    nothing. Extending the ceiling is what turns "run it again" into "continue
    evolving the population I already paid for" rather than a fresh search under
    a new seed. Never lowered: shrinking the ceiling would strand generations
    that already hold members and ledger rows.
    """

    def _write():
        with _db_lock:
            conn.execute(
                "UPDATE campaigns SET generations = ?, status = 'running', "
                "finished_ts = NULL WHERE campaign_id = ? AND generations < ?",
                (int(generations), campaign_id, int(generations)),
            )
            conn.commit()

    write_with_retry(_write)


def finish_campaign(conn, campaign_id: str, *, status: str) -> None:
    """Stamp a terminal status and finished_ts.

    A half-written campaign with no finished_ts is how a resumed run
    double-counts, so SIGINT writes status='aborted' WITH a timestamp rather
    than leaving the row open.
    """

    def _write():
        with _db_lock:
            conn.execute(
                "UPDATE campaigns SET status = ?, finished_ts = ? WHERE campaign_id = ?",
                (status, int(time.time() * 1000), campaign_id),
            )
            conn.commit()

    write_with_retry(_write)


def insert_generation(conn, gen: Generation) -> None:
    """Write (or replace) a generations row at its start."""

    def _write():
        with _db_lock:
            conn.execute(
                "INSERT OR REPLACE INTO generations (campaign_id, gen_index, "
                "window_start_ms, window_end_ms, population_size, n_evaluated, "
                "n_errors, n_unique_graphs, trials_cumulative, best_member_id, "
                "best_fitness, db_retries, wall_seconds, started_ts, finished_ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    gen.campaign_id, gen.gen_index, gen.window_start_ms,
                    gen.window_end_ms, gen.population_size, gen.n_evaluated,
                    gen.n_errors, gen.n_unique_graphs, gen.trials_cumulative,
                    gen.best_member_id, gen.best_fitness, gen.db_retries,
                    gen.wall_seconds, gen.started_ts, gen.finished_ts,
                ),
            )
            conn.commit()

    write_with_retry(_write)


def finish_generation(conn, campaign_id: str, gen_index: int, *, n_evaluated: int,
                      n_errors: int, n_unique_graphs: int, trials_cumulative: int,
                      best_member_id: str | None, best_fitness: float | None,
                      db_retries: int, wall_seconds: float) -> None:
    """Close a generations row. finished_ts is what makes it 'completed'."""

    def _write():
        with _db_lock:
            conn.execute(
                "UPDATE generations SET n_evaluated=?, n_errors=?, n_unique_graphs=?, "
                "trials_cumulative=?, best_member_id=?, best_fitness=?, db_retries=?, "
                "wall_seconds=?, finished_ts=? WHERE campaign_id=? AND gen_index=?",
                (
                    n_evaluated, n_errors, n_unique_graphs, trials_cumulative,
                    best_member_id, best_fitness, db_retries, wall_seconds,
                    int(time.time() * 1000), campaign_id, gen_index,
                ),
            )
            conn.commit()

    write_with_retry(_write)


def insert_members(conn, members) -> None:
    """Write a generation's member rows before any of them is evaluated.

    Written FIRST so an interrupted generation still records what it intended to
    try. `graph` is excluded by name: it is a live object, not a column.
    """
    cols = ", ".join(_MEMBER_COLUMNS)
    marks = ", ".join("?" for _ in _MEMBER_COLUMNS)
    rows = [
        tuple(getattr(m, c) for c in _MEMBER_COLUMNS) for m in members
    ]

    def _write():
        with _db_lock:
            conn.executemany(
                f"INSERT OR REPLACE INTO population_members ({cols}) VALUES ({marks})",
                rows,
            )
            conn.commit()

    write_with_retry(_write)


_RESULT_COLUMNS = (
    "fitness", "tier", "excess_sharpe", "excess_ann_return", "sharpe", "dsr",
    "ann_return_pct", "max_drawdown_pct", "n_trades", "bench_sharpe",
    "bench_ann_return_pct", "n_trials_used", "gate_json", "window_start_ms",
    "window_end_ms", "eval_seconds", "error",
)


def update_member_result(conn, member_id: str, values: Mapping[str, Any]) -> int:
    """Store one member's oracle result. Returns the retry count it cost.

    Called only by the parent (single-writer discipline): a worker returns plain
    data and never writes these tables.
    """
    unknown = sorted(set(values) - set(_RESULT_COLUMNS))
    if unknown:
        raise ValueError(
            f"update_member_result: unknown column(s) {unknown}; legal columns are "
            f"{list(_RESULT_COLUMNS)}"
        )
    cols = [c for c in _RESULT_COLUMNS if c in values]
    assign = ", ".join(f"{c} = ?" for c in cols)
    args = [values[c] for c in cols] + [member_id]

    def _write():
        with _db_lock:
            conn.execute(
                f"UPDATE population_members SET {assign} WHERE member_id = ?", args
            )
            conn.commit()

    _, retries = write_with_retry(_write)
    return retries


def top_members(conn, campaign_id: str, *, limit: int = 5,
                distinct_graphs: bool = True, roles=("offspring", "elite", "seed")):
    """Best scored members of a campaign, best first.

    Ordering here is SQL's, and it is deliberately only a shortlist filter — the
    authoritative total order is tournament.rank_key, applied by the caller.
    tier ASC works because TIER_ORDER is ("A","B","C","D") and 'A' < 'B' as
    text; NULL/'' tiers sort first in SQLite, so unscored rows are excluded by
    the fitness IS NOT NULL predicate rather than by relying on tier text.
    """
    placeholders = ", ".join("?" for _ in roles)
    with _db_lock:
        rows = conn.execute(
            "SELECT member_id, gen_index, member_index, graph_hash, graph_json, tier, "
            "fitness, excess_sharpe, excess_ann_return, sharpe, dsr, ann_return_pct, "
            "max_drawdown_pct, n_trades, bench_sharpe, bench_ann_return_pct, "
            "n_trials_used, gate_json, mutator "
            "FROM population_members WHERE campaign_id = ? AND fitness IS NOT NULL "
            f"AND role IN ({placeholders}) "
            "ORDER BY tier ASC, fitness DESC, member_id ASC",
            (campaign_id, *roles),
        ).fetchall()
    out = []
    seen: set[str] = set()
    for row in rows:
        if distinct_graphs and row[3] in seen:
            continue
        seen.add(row[3])
        out.append(
            {
                "member_id": row[0], "gen_index": row[1], "member_index": row[2],
                "graph_hash": row[3], "graph_json": row[4], "tier": row[5],
                "fitness": row[6], "excess_sharpe": row[7],
                "excess_ann_return": row[8], "sharpe": row[9], "dsr": row[10],
                "ann_return_pct": row[11], "max_drawdown_pct": row[12],
                "n_trades": row[13], "bench_sharpe": row[14],
                "bench_ann_return_pct": row[15], "n_trials_used": row[16],
                "gate_json": row[17], "mutator": row[18],
            }
        )
        if len(out) >= limit:
            break
    return out


def last_completed_generation(conn, campaign_id: str) -> int | None:
    """Highest gen_index with a finished_ts, or None.

    finished_ts, not MAX(gen_index): a generation interrupted mid-flight has a
    row but no finished_ts, and resuming from it would double-count the members
    it already paid trials for.
    """
    with _db_lock:
        row = conn.execute(
            "SELECT MAX(gen_index) FROM generations "
            "WHERE campaign_id = ? AND finished_ts IS NOT NULL",
            (campaign_id,),
        ).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def seen_graph_hashes(conn, campaign_id: str) -> set[str]:
    """Every graph_hash this campaign has already bred.

    Rebuilt on --resume so a resumed campaign does not re-breed (and re-pay for)
    duplicates it already evaluated.
    """
    with _db_lock:
        rows = conn.execute(
            "SELECT DISTINCT graph_hash FROM population_members WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchall()
    return {r[0] for r in rows}


def generation_rows(conn, campaign_id: str) -> list[dict]:
    """Every generations row for a campaign, ascending — for `evolve --report`."""
    with _db_lock:
        rows = conn.execute(
            "SELECT gen_index, window_start_ms, window_end_ms, population_size, "
            "n_evaluated, n_errors, n_unique_graphs, trials_cumulative, "
            "best_member_id, best_fitness, db_retries, wall_seconds "
            "FROM generations WHERE campaign_id = ? ORDER BY gen_index ASC",
            (campaign_id,),
        ).fetchall()
    keys = (
        "gen_index", "window_start_ms", "window_end_ms", "population_size",
        "n_evaluated", "n_errors", "n_unique_graphs", "trials_cumulative",
        "best_member_id", "best_fitness", "db_retries", "wall_seconds",
    )
    return [dict(zip(keys, r)) for r in rows]


_MEMBER_SELECT_COLS = (
    "member_id", "campaign_id", "gen_index", "member_index", "graph_hash",
    "graph_json", "rng_seed", "role", "parent_member_id", "mutator",
    "mutation_json", "fitness", "tier", "excess_sharpe", "excess_ann_return",
    "sharpe", "dsr", "ann_return_pct", "max_drawdown_pct", "n_trades",
    "bench_sharpe", "bench_ann_return_pct", "n_trials_used", "gate_json",
    "window_start_ms", "window_end_ms", "eval_seconds", "error",
)


def generation_members(conn, campaign_id: str, gen_index: int) -> list[dict]:
    """Every SCORED member row of one generation, in member_index order.

    What `--resume` needs to actually resume: without it a resumed campaign
    re-seeds from the original graph and throws away the evolutionary progress it
    already paid trials for, which is bookkeeping that continues correctly while
    the SEARCH silently restarts.
    """
    with _db_lock:
        rows = conn.execute(
            f"SELECT {', '.join(_MEMBER_SELECT_COLS)} FROM population_members "
            "WHERE campaign_id = ? AND gen_index = ? AND role != 'finalist' "
            "ORDER BY member_index ASC",
            (campaign_id, gen_index),
        ).fetchall()
    return [dict(zip(_MEMBER_SELECT_COLS, r)) for r in rows]


def load_member(conn, member_id: str) -> dict | None:
    """One member row by id, regardless of role, or None if it doesn't exist.

    Unlike `generation_members`, this deliberately does NOT filter out
    `role == 'finalist'`. That filter exists there to keep the audit round's
    rows out of a generation's regular listing; it has no business here. A
    finalist is a fully scored member with its own `graph_json` and evaluation
    window like any other — a legitimate replay target — so excluding it would
    make "inspect this member" silently fail for exactly the members the audit
    round singled out as most worth inspecting.
    """
    with _db_lock:
        row = conn.execute(
            f"SELECT {', '.join(_MEMBER_SELECT_COLS)} FROM population_members "
            "WHERE member_id = ?",
            (member_id,),
        ).fetchone()
    return None if row is None else dict(zip(_MEMBER_SELECT_COLS, row))


def finalist_rows(conn, campaign_id: str) -> list[dict]:
    """The audit round's rows (role='finalist'), in member_id order."""
    return top_members(
        conn, campaign_id, limit=10_000, distinct_graphs=False, roles=("finalist",)
    )
