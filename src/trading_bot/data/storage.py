"""
SQLite storage layer for OHLCV candles.

Stores candles in a single table with idempotent upsert semantics.
Timestamps are epoch milliseconds (ccxt/Binance format).
"""

import logging
import sqlite3
import threading
import time
from pathlib import Path

from trading_bot import config

logger = logging.getLogger("trading_bot")

# Daily bars are UTC-midnight-aligned: every Binance 1d open_time satisfies
# ts % 86_400_000 == 0, which keeps find_gaps' grid-alignment invariant (#2)
# true for 1d without special-casing. config.BACKFILL_START ("2023-01-01" ->
# 1672531200000) is likewise 1d-aligned.
TIMEFRAME_MS = {
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Serializes all DB access across callers (poller, backfill, cli) that may
# share a single connection across threads. Lives here, beside the
# connection's accessor functions, rather than in any one caller module.
_db_lock = threading.Lock()


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """
    Connect to the SQLite database.

    Args:
        db_path: Path to the database file. If None, uses trading_bot.config.DB_PATH.

    Returns:
        sqlite3.Connection with WAL mode enabled and the ohlcv table created.
    """
    if db_path is None:
        db_path = config.DB_PATH

    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: the live poller shares one connection across
    # APScheduler worker threads; this module's _db_lock serializes all access
    # so this is safe (WAL mode handles the concurrent-reader case for later
    # phases).
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            ts INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timeframe, ts)
        )
        """
    )
    conn.commit()
    return conn


def upsert_candles(
    conn: sqlite3.Connection, symbol: str, timeframe: str, rows: list[list]
) -> int:
    """
    Upsert OHLCV candles (idempotent).

    Args:
        conn: Database connection.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        timeframe: Candle timeframe (e.g., "15m").
        rows: List of ccxt-style rows [ts_ms, open, high, low, close, volume].

    Returns:
        Number of rows passed in.
    """
    with _db_lock:
        conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv
            (symbol, timeframe, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(symbol, timeframe, r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        )
        conn.commit()
    return len(rows)


def last_ts(conn: sqlite3.Connection, symbol: str, timeframe: str) -> int | None:
    """
    Return the maximum stored ts for a symbol/timeframe, or None if empty.
    """
    with _db_lock:
        cursor = conn.execute(
            "SELECT MAX(ts) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        result = cursor.fetchone()
    return result[0] if result[0] is not None else None


def find_gaps(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    *,
    now_ms: int | None = None,
    start_ms: int | None = None,
) -> list[tuple[int, int]]:
    """
    Find gaps in the stored candle series.

    For each consecutive pair of stored timestamps where the difference exceeds
    the timeframe interval, append (prev_ts + interval, next_ts - interval) as
    the missing inclusive range.

    Also detects staleness: if the series is stale by more than
    config.STALENESS_INTERVALS * interval, appends a trailing gap from
    (anchor + interval) to expected_end, where anchor is the last stored
    timestamp (or start_ms - interval for an empty series).

    When the series is empty (never populated), reports the entire expected
    backfill window as a single gap: (start_ms, expected_end).

    INVARIANTS (hold for every returned tuple, for any inputs):
      1. start <= end — never inverted.
      2. start % interval == 0 and end % interval == 0 — grid-aligned, uniform
         tuple shape for interior AND trailing gaps.
      3. If expected_end < start_ms (e.g. now_ms predates the backfill
         window), returns [] — nothing was expected yet.

    Args:
        conn: Database connection.
        symbol: Trading pair symbol.
        timeframe: Candle timeframe.
        now_ms: Current time in epoch milliseconds. If None, uses current time.
        start_ms: Expected start of the series. If None, defaults to
            config.date_to_ms(config.BACKFILL_START).

    Returns:
        List of (gap_start_ts, gap_end_ts) tuples. Empty list means no gaps.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = config.date_to_ms(config.BACKFILL_START)

    interval = TIMEFRAME_MS[timeframe]
    # Open-time of the most recent FULLY CLOSED candle. The candle opening at
    # floor(now_ms, interval) is still forming and is never "missing".
    expected_end = (now_ms // interval) * interval - interval

    if expected_end < start_ms:
        return []

    with _db_lock:
        cursor = conn.execute(
            "SELECT ts FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY ts ASC",
            (symbol, timeframe),
        )
        timestamps = [row[0] for row in cursor.fetchall()]

    gaps: list[tuple[int, int]] = []
    for prev_ts, next_ts in zip(timestamps, timestamps[1:]):
        if next_ts - prev_ts > interval:
            gaps.append((prev_ts + interval, next_ts - interval))

    anchor = timestamps[-1] if timestamps else start_ms - interval

    if not timestamps:
        # An entirely-missing series is never "just poller lag".
        gaps.append((start_ms, expected_end))
    elif (
        now_ms - anchor > config.STALENESS_INTERVALS * interval
        and anchor + interval <= expected_end
    ):
        gaps.append((anchor + interval, expected_end))

    return gaps


def is_transient_db_error(exc: sqlite3.OperationalError) -> bool:
    """True for contention errors worth retrying/skipping ("database is locked",
    "database is busy"); False for permanent failures (corrupt file, disk I/O,
    missing table) which must propagate."""
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def load_candles(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[tuple]:
    """
    Load OHLCV candles for a symbol/timeframe, optionally bounded by time.

    Rows are returned in ascending order by timestamp. Both start_ms and end_ms
    are inclusive bounds (if specified).

    Args:
        conn: Database connection.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        timeframe: Candle timeframe (e.g., "4h").
        start_ms: Optional lower bound for timestamp (inclusive), in epoch milliseconds.
        end_ms: Optional upper bound for timestamp (inclusive), in epoch milliseconds.

    Returns:
        List of tuples (ts, open, high, low, close, volume) ordered by ts ASC.
        Returns empty list if no rows match the criteria.
    """
    query = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = ?"
    params: list = [symbol, timeframe]

    if start_ms is not None:
        query += " AND ts >= ?"
        params.append(start_ms)

    if end_ms is not None:
        query += " AND ts <= ?"
        params.append(end_ms)

    query += " ORDER BY ts ASC"

    cursor = conn.execute(query, params)
    return cursor.fetchall()
