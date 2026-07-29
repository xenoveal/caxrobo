"""
Live polling loop for OHLCV candles.

Periodically fetches latest candles for each symbol/timeframe and stores them.
Jobs are boundary-aligned (15m at minute 0/15/30/45, 1h at minute 0, 4h at hour
boundaries, 1d at UTC midnight).
Errors on a single symbol do not block fetches for others.
"""

import logging
import sqlite3
import time

import apscheduler.schedulers.background
import ccxt

from trading_bot import config
from trading_bot.data import storage
from trading_bot.exchange import binance_client

logger = logging.getLogger("trading_bot")


def poll_once(
    conn: sqlite3.Connection, timeframe: str, *, exchange=None, lookback: int = 3
) -> int:
    """
    Fetch and upsert the latest candles for all symbols in a given timeframe.

    For each symbol, computes since_ms based on the last stored timestamp (or now - lookback intervals),
    fetches a page of candles, and upserts them. Errors on any one symbol are logged and skipped;
    the function does not raise.

    Args:
        conn: Database connection.
        timeframe: Candle timeframe (e.g., "15m", "1h", "4h").
        exchange: Optional exchange instance for testing. If None, uses get_exchange().
        lookback: Number of intervals to look back if no stored data exists.

    Returns:
        Total number of rows upserted across all symbols.
    """
    interval_ms = storage.TIMEFRAME_MS[timeframe]
    now_ms = int(time.time() * 1000)
    total_rows = 0

    for symbol in config.SYMBOLS:
        try:
            # Compute since_ms: use last_ts if available, otherwise now - lookback intervals
            last = storage.last_ts(conn, symbol, timeframe)
            if last is not None:
                # Start one interval before the last timestamp to catch any gaps/updates
                since_ms = last - interval_ms
            else:
                # No data yet: look back `lookback` intervals
                since_ms = now_ms - (lookback * interval_ms)

            # Fetch a page of candles
            rows = binance_client.fetch_ohlcv_page(
                symbol, timeframe, since_ms, limit=1000, exchange=exchange
            )

            # Upsert into database
            count = storage.upsert_candles(conn, symbol, timeframe, rows)
            total_rows += count

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(f"Error polling {symbol}/{timeframe}: {e}")
            # Continue to next symbol without raising
        except sqlite3.OperationalError as e:
            if storage.is_transient_db_error(e):
                logger.error(f"Error polling {symbol}/{timeframe}: {e}")
                # Transient (lock/busy) contention: continue to next symbol
            else:
                raise

    return total_rows


# Mapping of timeframe to cron job schedule (Finding 7: data-driven scheduling)
_CRON_BY_TIMEFRAME = {
    "15m": {"minute": "0,15,30,45", "second": 10},
    "1h": {"minute": "0", "second": 10},
    # Any hour-anchored job MUST pin timezone="UTC": BackgroundScheduler()
    # defaults to the host's LOCAL timezone, so on a UTC+7 host an unpinned
    # {"hour": "0,4,8,12,16,20"} fires 7h off the UTC candle-close grid and
    # fetches still-forming bars as if they were closed. 15m and 1h are
    # minute-anchored only, so they are timezone-invariant.
    "4h": {"hour": "0,4,8,12,16,20", "minute": "0", "second": 10, "timezone": "UTC"},
    "1d": {"hour": "0", "minute": "0", "second": 10, "timezone": "UTC"},
}


def build_scheduler(
    conn: sqlite3.Connection, *, exchange=None
) -> apscheduler.schedulers.background.BackgroundScheduler:
    """
    Build a background scheduler with one job per timeframe.

    Jobs are aligned to market boundaries with a 10-second grace period:
    - 15m: runs at minute 0/15/30/45 second 10
    - 1h: runs at minute 0 second 10
    - 4h: runs at hour 0/4/8/12/16/20 minute 0 second 10
    - 1d: runs at 00:00:10 UTC (explicitly timezone-pinned)

    Args:
        conn: Database connection.
        exchange: Optional exchange instance for testing.

    Returns:
        APScheduler BackgroundScheduler with jobs added (not started).
    """
    scheduler = apscheduler.schedulers.background.BackgroundScheduler()

    # Add one job per timeframe, driven by _CRON_BY_TIMEFRAME mapping
    for timeframe, cron_kwargs in _CRON_BY_TIMEFRAME.items():
        scheduler.add_job(
            poll_once,
            "cron",
            args=(conn, timeframe),
            kwargs={"exchange": exchange},
            **cron_kwargs,
        )

    return scheduler


def run_polling_loop(conn=None, exchange=None):
    """
    Run the polling loop indefinitely.

    Connects to the database if conn is None, builds and starts the scheduler,
    then blocks on a sleep loop until KeyboardInterrupt.

    Args:
        conn: Optional database connection. If None, calls storage.connect().
        exchange: Optional exchange instance for testing.
    """
    if conn is None:
        conn = storage.connect()

    scheduler = build_scheduler(conn, exchange=exchange)
    scheduler.start()

    logger.info("Polling loop started")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down")
        scheduler.shutdown()
