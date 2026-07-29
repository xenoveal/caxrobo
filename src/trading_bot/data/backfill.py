"""
Historical OHLCV backfill with resumability support.

Fetches candle data from the exchange in pages and stores them in the database.
Resumes from the last stored timestamp to avoid duplicate work.
"""

import dataclasses
import logging
import sqlite3
import time

import ccxt

from trading_bot import config
from trading_bot.data import storage
from trading_bot.data.storage import TIMEFRAME_MS, last_ts, upsert_candles
from trading_bot.exchange.binance_client import fetch_ohlcv_page

logger = logging.getLogger("trading_bot")


@dataclasses.dataclass
class BackfillResult:
    """Outcome of a single backfill_series run.

    Attributes:
        rows: Total number of rows upserted during the run.
        complete: True if the series reached now_ms or the exchange ran out
            of data (empty page); False if the run ended early due to a
            stalled cursor or a fetch/DB error.
        reason: None when complete; otherwise one of "cursor-stalled",
            "network-error", "exchange-error", "transient-db-error".
    """

    rows: int
    complete: bool
    reason: str | None = None


def backfill_series(
    conn,
    symbol: str,
    timeframe: str,
    *,
    exchange=None,
    now_ms: int | None = None,
    start_ms: int | None = None,
) -> BackfillResult:
    """Fetch and store OHLCV data for a symbol/timeframe.

    Determines the start time from the database (if data exists) or from
    the provided start_ms / config.BACKFILL_START. Fetches pages until
    reaching now_ms or an empty page is returned.

    Args:
        conn: sqlite3.Connection to the database.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        timeframe: OHLCV timeframe (e.g., "15m").
        exchange: Optional exchange instance for testing. If None, uses get_exchange().
        now_ms: End time in epoch milliseconds. If None, uses current time.
        start_ms: Start time override for empty series (epoch milliseconds).
                  If None, uses config.BACKFILL_START.

    Returns:
        A BackfillResult with the total rows upserted and whether the run
        completed the series or ended early (see BackfillResult docstring).
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    # Determine cursor (start time for fetching)
    last_stored_ts = last_ts(conn, symbol, timeframe)
    if last_stored_ts is not None:
        cursor = last_stored_ts + TIMEFRAME_MS[timeframe]
    else:
        # Empty series: use provided start_ms or config.BACKFILL_START
        if start_ms is not None:
            cursor = start_ms
        else:
            cursor = config.date_to_ms(config.BACKFILL_START)

    total_rows = 0
    interval = TIMEFRAME_MS[timeframe]
    prev_cursor = None

    while cursor <= now_ms:
        # Fetch a page of OHLCV data
        try:
            page = fetch_ohlcv_page(symbol, timeframe, since_ms=cursor, exchange=exchange)
        except ccxt.NetworkError as e:
            logger.error(f"Backfill {symbol} {timeframe}: network error: {e}")
            return BackfillResult(total_rows, complete=False, reason="network-error")
        except ccxt.ExchangeError as e:
            logger.error(f"Backfill {symbol} {timeframe}: exchange error: {e}")
            return BackfillResult(total_rows, complete=False, reason="exchange-error")

        if not page:
            # Empty page signals end of available data (normal end of series)
            break

        # Upsert the page (idempotent, so safe to call multiple times)
        try:
            count = upsert_candles(conn, symbol, timeframe, page)
        except sqlite3.OperationalError as e:
            if storage.is_transient_db_error(e):
                logger.error(f"Backfill {symbol} {timeframe}: transient DB error: {e}")
                return BackfillResult(total_rows, complete=False, reason="transient-db-error")
            raise
        total_rows += count

        # Log progress
        last_row_ts = page[-1][0]
        logger.info(
            f"Backfill {symbol} {timeframe}: {count} rows, last_ts={last_row_ts}"
        )

        # Advance cursor to next interval after the last row
        prev_cursor = cursor
        cursor = last_row_ts + interval

        # Guard against infinite loop / crawling advances: require the cursor
        # to advance by at least one full interval each page.
        if cursor < prev_cursor + interval:
            logger.error(
                f"Backfill {symbol} {timeframe}: cursor stalled (prev={prev_cursor}, "
                f"new={cursor}, last_row_ts={last_row_ts}). Aborting to avoid infinite loop."
            )
            return BackfillResult(total_rows, complete=False, reason="cursor-stalled")

    return BackfillResult(total_rows, complete=True, reason=None)


def backfill_all(
    conn, *, exchange=None, symbols=None, timeframes=None, start_ms: int | None = None
) -> dict[tuple[str, str], BackfillResult]:
    """Backfill all symbol/timeframe combinations sequentially.

    Args:
        conn: sqlite3.Connection to the database.
        exchange: Optional exchange instance for testing.
        symbols: List of symbols to backfill. If None, uses config.SYMBOLS.
        timeframes: List of timeframes to backfill. If None, uses config.TIMEFRAMES.
        start_ms: Start time override for empty series (epoch milliseconds).

    Returns:
        Dictionary mapping (symbol, timeframe) to its BackfillResult.
    """
    if symbols is None:
        symbols = config.SYMBOLS
    if timeframes is None:
        timeframes = config.TIMEFRAMES

    results = {}
    for symbol in symbols:
        for timeframe in timeframes:
            result = backfill_series(
                conn,
                symbol,
                timeframe,
                exchange=exchange,
                start_ms=start_ms,
            )
            results[(symbol, timeframe)] = result

    return results
