"""
SQLite connection layer for framework state (v0.3.0 Phase 1).

WHY A SECOND DATABASE. ohlcv.db holds ~117 MB of irreplaceable backfilled
price history; state.db holds derived, reproducible bookkeeping (the trial
ledger, and later review records / strategy versions / populations). Keeping
them apart means a corrupt experiment log is a `rm` and a re-run, never a
data-loss event.

TABLE OWNERSHIP. This module creates NO tables. Each table's DDL is owned by
the module that uses it, via `CREATE TABLE IF NOT EXISTS` (see
backtest/trials.py for `trial_ledger`), so there is no central migration file
to drift out of sync with the code that reads the rows.

TIMESTAMPS. Every ts column in state.db is epoch MILLISECONDS, UTC — the same
convention as ohlcv.db (e.g. 1672531200000 == 2023-01-01T00:00:00Z). Never
formatted date strings.
"""

import logging
import sqlite3
import threading
from pathlib import Path

from trading_bot import config

logger = logging.getLogger("trading_bot")

# Serializes all state.db access, mirroring storage._db_lock's placement
# (beside the connection's accessor function rather than in any one caller).
# Deliberately a SEPARATE lock object from storage._db_lock: state.db writes
# must not queue behind a long ohlcv.db backfill, and sharing one lock across
# two databases couples them for no benefit. Every table owner in state.db
# shares THIS lock.
_db_lock = threading.Lock()


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """
    Connect to the framework-state SQLite database.

    Args:
        db_path: Path to the database file. If None, uses
            trading_bot.config.STATE_DB_PATH. ":memory:" is legal (its parent
            resolves to ".", so the mkdir below is a no-op) and is what tests
            use when persistence is not the property under test.

    Returns:
        sqlite3.Connection with WAL mode enabled and NO tables created —
        DDL belongs to the module that owns the table.
    """
    if db_path is None:
        db_path = config.STATE_DB_PATH

    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False for the same reason storage.connect() does it:
    # one connection may be shared across threads, and _db_lock serializes
    # every write.
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn
