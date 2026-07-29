"""
Persistent trial ledger for THE GATE's DSR correction (v0.3.0 Phase 1).

WHAT IT IS. One row per evaluation the fitness oracle ever performs: which
campaign, which strategy graph, which parameters, over which span, at what
time. The rows live in state.db (NOT ohlcv.db), so the count survives process
restarts and overnight runs — an evolution run that crashes and resumes must
not reset its own degrees-of-freedom count.

WHAT IT IS FOR. `n_trials` passed to equity.deflated_sharpe is the CUMULATIVE
count for the campaign, across generations (v0.3.0 contract §4's resolution of
the PRD's open question on honest trial counting under a population). This
will produce brutal DSR values. That is the correct, honest answer and must not
be softened; if it makes the gate unpassable, that is the finding (PRD honesty
clause).

`graph_hash` is an OPAQUE string here. This module imports nothing from
trading_bot.framework — Phase 1 gates Phase 3, not the reverse — and
LEGACY_GRAPH_HASH is the sentinel for pre-framework engine.run_backtest runs.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, fields, is_dataclass

from trading_bot.data.statestore import _db_lock

logger = logging.getLogger("trading_bot")

LEDGER_TABLE = "trial_ledger"

# Sentinel graph identity for evaluations that predate the Phase 3 plug-in
# framework: the hardcoded v0.2.0 Donchian pipeline in engine.run_backtest.
LEGACY_GRAPH_HASH = "legacy-donchian"


@dataclass(frozen=True)
class TrialRecord:
    """One recorded oracle evaluation. ts is epoch ms, UTC (contract §6)."""

    campaign: str
    graph_hash: str
    params_hash: str
    start_ms: int
    end_ms: int
    ts: int


def ensure_schema(conn) -> None:
    """Create the trial_ledger table and its index if absent (idempotent).

    The DDL lives here, not in statestore.py: contract §6 gives each table's
    schema to the module that uses it, so there is no central migration file to
    drift out of sync.
    """
    with _db_lock:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign TEXT NOT NULL,
                graph_hash TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                ts INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{LEDGER_TABLE}_campaign "
            f"ON {LEDGER_TABLE}(campaign)"
        )
        conn.commit()


def stable_hash(obj) -> str:
    """Short, process-stable digest of a JSON-able object.

    Never Python's builtin hash(): PYTHONHASHSEED randomizes str hashing per
    process, so a ledger keyed on it would fail to recognize the same
    configuration across restarts — precisely the failure this module exists to
    prevent.

    default=str makes this total rather than raising mid-run, at the cost that
    two objects sharing a repr collide. Acceptable here (every params value is
    bool/int/float/str) and pinned by a test rather than engineered around.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def params_hash(params) -> str:
    """Digest of a parameter set: a (frozen) dataclass, a Mapping, or None."""
    if params is None:
        return stable_hash({})
    if is_dataclass(params) and not isinstance(params, type):
        return stable_hash({f.name: getattr(params, f.name) for f in fields(params)})
    return stable_hash(dict(params))


class TrialLedger:
    """Append-only evaluation ledger for one campaign.

    There is deliberately NO UNIQUE constraint: re-evaluating a configuration
    is a legitimate separate row. The ledger measures evaluations PERFORMED,
    and silently deduping would understate n_trials in the flattering
    direction. distinct_count() serves whoever wants the other number.
    """

    def __init__(self, conn, campaign: str):
        if not campaign:
            raise ValueError("campaign must be a non-empty string")
        self.conn = conn
        self.campaign = campaign
        ensure_schema(conn)

    def record(
        self,
        *,
        graph_hash: str,
        params_hash: str,
        start_ms: int,
        end_ms: int,
        ts: int | None = None,
    ) -> int:
        """Append one evaluation; return the campaign's POST-insert count."""
        if ts is None:
            ts = int(time.time() * 1000)
        with _db_lock:
            self.conn.execute(
                f"INSERT INTO {LEDGER_TABLE} "
                "(campaign, graph_hash, params_hash, start_ms, end_ms, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self.campaign, graph_hash, params_hash, start_ms, end_ms, ts),
            )
            self.conn.commit()
        return self.count()

    def count(self) -> int:
        """Rows recorded for this campaign — what the DSR is charged."""
        with _db_lock:
            cur = self.conn.execute(
                f"SELECT COUNT(*) FROM {LEDGER_TABLE} WHERE campaign = ?",
                (self.campaign,),
            )
            return cur.fetchone()[0]

    def distinct_count(self) -> int:
        """Distinct (graph, params, span) tuples. REPORTED ONLY, never charged
        to the DSR: contract §4 says the ledger counts every evaluation the
        oracle performs, and over-counting errs pessimistically."""
        with _db_lock:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT graph_hash, params_hash, "
                f"start_ms, end_ms FROM {LEDGER_TABLE} WHERE campaign = ?)",
                (self.campaign,),
            )
            return cur.fetchone()[0]

    def records(self, limit: int | None = None) -> list[TrialRecord]:
        """Recorded evaluations for this campaign, oldest first."""
        query = (
            "SELECT campaign, graph_hash, params_hash, start_ms, end_ms, ts "
            f"FROM {LEDGER_TABLE} WHERE campaign = ? ORDER BY id ASC"
        )
        params: list = [self.campaign]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with _db_lock:
            rows = self.conn.execute(query, params).fetchall()
        return [TrialRecord(*row) for row in rows]
