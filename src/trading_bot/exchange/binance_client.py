"""
Binance USDT-M futures client wrapper using CCXT.

Provides paginated OHLCV data fetching with memoized exchange instance.
"""

import logging
import os
from functools import lru_cache

import ccxt

logger = logging.getLogger("trading_bot")


def to_ccxt_symbol(symbol: str) -> str:
    """
    Convert symbol to CCXT format for Binance USDT-M futures.

    Normalizes various input forms to the canonical perpetual format "{BASE}/USDT:USDT":
    - "BTCUSDT"       -> "BTC/USDT:USDT"   (raw concatenated)
    - "BTC/USDT"      -> "BTC/USDT:USDT"   (spot-style)
    - "BTC/USDT:USDT" -> "BTC/USDT:USDT"   (already canonical, idempotent)

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT", "BTC/USDT", or "BTC/USDT:USDT").

    Returns:
        CCXT format symbol in perpetual format "{BASE}/USDT:USDT".

    Raises:
        ValueError: If the symbol format is not USDT-quoted or unsupported.
    """
    # Already in canonical perpetual format
    if symbol.endswith("/USDT:USDT"):
        base = symbol[:-10]  # Remove "/USDT:USDT"
        if base:  # Ensure non-empty base
            return symbol

    # Spot-style format "BASE/USDT"
    if "/" in symbol and symbol.endswith("/USDT"):
        base = symbol[:-5]  # Remove "/USDT"
        if base:  # Ensure non-empty base
            return f"{base}/USDT:USDT"

    # Raw format "BASEUSDT"
    if symbol.endswith("USDT") and "/" not in symbol:
        base = symbol[:-4]  # Remove "USDT"
        if base:  # Ensure non-empty base
            return f"{base}/USDT:USDT"

    # Unsupported format
    raise ValueError(
        f"Unsupported symbol format: {symbol!r}; "
        f"expected a USDT-quoted pair like 'BTCUSDT' or 'BTC/USDT'"
    )


@lru_cache(maxsize=1)
def get_exchange():
    """
    Get a memoized CCXT Binance USDT-M exchange instance.

    Returns:
        A ccxt.binanceusdm instance with rate limiting enabled and
        API credentials from environment variables.
    """
    return ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "apiKey": os.getenv("BINANCE_API_KEY", ""),
            "secret": os.getenv("BINANCE_API_SECRET", ""),
        }
    )


def fetch_ohlcv_page(
    symbol: str, timeframe: str, since_ms: int, limit: int = 1000, exchange=None
) -> list[list]:
    """
    Fetch a single page of OHLCV data for a symbol and timeframe.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT" or "BTC/USDT:USDT").
        timeframe: OHLCV timeframe (e.g., "15m", "1h", "4h").
        since_ms: Start time in epoch milliseconds.
        limit: Number of candles to fetch (default 1000, max per exchange).
        exchange: Optional exchange instance for testing. If None, uses get_exchange().

    Returns:
        List of OHLCV rows [timestamp_ms, open, high, low, close, volume],
        sorted ascending by timestamp. Rows are unmodified from the exchange.

    Raises:
        Any exceptions from the exchange (ccxt.* exceptions, network errors, etc.)
        are propagated to the caller.
    """
    if exchange is None:
        exchange = get_exchange()

    # Convert symbol to CCXT format (idempotent)
    ccxt_symbol = to_ccxt_symbol(symbol)
    rows = exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since_ms, limit=limit)
    # Sort by timestamp (first element) in ascending order
    rows.sort(key=lambda row: row[0])
    return rows
