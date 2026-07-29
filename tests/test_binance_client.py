"""
Unit and integration tests for Binance CCXT client wrapper.
"""

import os
import pytest

from trading_bot.exchange.binance_client import fetch_ohlcv_page, get_exchange, to_ccxt_symbol


class FakeExchange:
    """Mock exchange for testing."""

    def __init__(self, rows=None, exception=None):
        """
        Initialize fake exchange.

        Args:
            rows: List of OHLCV rows to return from fetch_ohlcv.
            exception: Exception to raise from fetch_ohlcv.
        """
        self.rows = rows or []
        self.exception = exception
        self.last_symbol = None  # Track the symbol called

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        """Return rows or raise exception, recording the symbol."""
        self.last_symbol = symbol
        if self.exception:
            raise self.exception
        return self.rows


def test_fetch_ohlcv_page_sorts_ascending():
    """Test that fetch_ohlcv_page sorts rows by timestamp ascending."""
    # Create rows out of order by timestamp (first element)
    unsorted_rows = [
        [1500000000000, 100, 110, 90, 105, 1000],  # ts: 1500000000000
        [1500000001000, 105, 115, 95, 110, 1000],  # ts: 1500000001000
        [1499999999000, 95, 105, 85, 100, 1000],   # ts: 1499999999000 (earliest)
    ]

    fake_exchange = FakeExchange(rows=unsorted_rows)

    result = fetch_ohlcv_page(
        "BTC/USDT:USDT", "15m", since_ms=1499999999000, exchange=fake_exchange
    )

    # Assert rows are sorted by timestamp ascending
    assert result[0][0] == 1499999999000
    assert result[1][0] == 1500000000000
    assert result[2][0] == 1500000001000

    # Assert rows are otherwise unmodified
    assert result == [
        [1499999999000, 95, 105, 85, 100, 1000],
        [1500000000000, 100, 110, 90, 105, 1000],
        [1500000001000, 105, 115, 95, 110, 1000],
    ]


def test_fetch_ohlcv_page_propagates_exception():
    """Test that fetch_ohlcv_page propagates exchange exceptions."""
    test_exception = ValueError("Exchange error")
    fake_exchange = FakeExchange(exception=test_exception)

    with pytest.raises(ValueError, match="Exchange error"):
        fetch_ohlcv_page(
            "BTC/USDT:USDT", "15m", since_ms=1500000000000, exchange=fake_exchange
        )


def test_to_ccxt_symbol_idempotent():
    """Test that to_ccxt_symbol is idempotent for CCXT format symbols."""
    assert to_ccxt_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"
    assert to_ccxt_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"
    assert to_ccxt_symbol("SOL/USDT:USDT") == "SOL/USDT:USDT"


def test_to_ccxt_symbol_converts_config_format():
    """Test that to_ccxt_symbol converts config symbol format to CCXT format."""
    assert to_ccxt_symbol("BTCUSDT") == "BTC/USDT:USDT"
    assert to_ccxt_symbol("ETHUSDT") == "ETH/USDT:USDT"
    assert to_ccxt_symbol("SOLUSDT") == "SOL/USDT:USDT"


def test_to_ccxt_symbol_converts_spot_style_format():
    """Test that to_ccxt_symbol converts spot-style format to perpetual format (regression test for idempotency bug)."""
    assert to_ccxt_symbol("BTC/USDT") == "BTC/USDT:USDT"
    assert to_ccxt_symbol("ETH/USDT") == "ETH/USDT:USDT"
    assert to_ccxt_symbol("SOL/USDT") == "SOL/USDT:USDT"


def test_fetch_ohlcv_page_converts_symbol():
    """Test that fetch_ohlcv_page converts config symbol format to CCXT before calling exchange."""
    fake_exchange = FakeExchange(rows=[[1500000000000, 100, 110, 90, 105, 1000]])

    result = fetch_ohlcv_page(
        "BTCUSDT", "15m", since_ms=1500000000000, exchange=fake_exchange
    )

    # Assert the exchange was called with CCXT format
    assert fake_exchange.last_symbol == "BTC/USDT:USDT"

    # Assert result is unchanged
    assert result == [[1500000000000, 100, 110, 90, 105, 1000]]


def test_fetch_ohlcv_page_idempotent_with_ccxt_format():
    """Test that fetch_ohlcv_page is idempotent if already given CCXT format."""
    fake_exchange = FakeExchange(rows=[[1500000000000, 100, 110, 90, 105, 1000]])

    result = fetch_ohlcv_page(
        "BTC/USDT:USDT", "15m", since_ms=1500000000000, exchange=fake_exchange
    )

    # Assert the exchange was called with the same symbol
    assert fake_exchange.last_symbol == "BTC/USDT:USDT"

    # Assert result is unchanged
    assert result == [[1500000000000, 100, 110, 90, 105, 1000]]


def test_to_ccxt_symbol_rejects_unsupported_formats():
    """Test that to_ccxt_symbol raises ValueError for unsupported symbol formats."""
    # Non-USDT quotes
    with pytest.raises(ValueError, match="Unsupported symbol format"):
        to_ccxt_symbol("BTCUSD")

    with pytest.raises(ValueError, match="Unsupported symbol format"):
        to_ccxt_symbol("BTC/BUSD")

    # Empty base
    with pytest.raises(ValueError, match="Unsupported symbol format"):
        to_ccxt_symbol("USDT")

    # Random strings
    with pytest.raises(ValueError, match="Unsupported symbol format"):
        to_ccxt_symbol("FOO")

    # Empty string
    with pytest.raises(ValueError, match="Unsupported symbol format"):
        to_ccxt_symbol("")


@pytest.mark.network
@pytest.mark.skipif(
    not os.getenv("RUN_NETWORK_TESTS"),
    reason="Network tests disabled by default; set RUN_NETWORK_TESTS=1 to enable",
)
def test_fetch_ohlcv_page_network_smoke():
    """
    Network smoke test: fetch BTC/USDT:USDT 15m candles from live Binance.

    This test is skipped by default unless RUN_NETWORK_TESTS=1 is set.
    """
    # Use the real exchange from get_exchange()
    result = fetch_ohlcv_page(
        symbol="BTC/USDT:USDT",
        timeframe="15m",
        since_ms=1609459200000,  # 2021-01-01 00:00:00 UTC
        limit=10,
    )

    # Assert non-empty result
    assert len(result) > 0

    # Assert all timestamps are in ascending order
    for i in range(len(result) - 1):
        assert result[i][0] < result[i + 1][0]

    # Assert each row has the expected 6 elements (ts, o, h, l, c, v)
    for row in result:
        assert len(row) == 6
        assert isinstance(row[0], (int, float))  # timestamp
