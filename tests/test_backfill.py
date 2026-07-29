"""
Tests for the OHLCV backfill module.

Uses a fake exchange to verify:
1. Full range coverage: all expected candles are fetched and stored.
2. Resumability: pre-inserted data is not re-fetched, and no duplicates occur.
3. No duplicate rows: COUNT(*) == COUNT(DISTINCT ts) for each series.
"""

import sqlite3

import ccxt
import pytest
from datetime import datetime, timezone

from trading_bot import config
from trading_bot.data.backfill import BackfillResult, backfill_series, backfill_all
from trading_bot.data.storage import TIMEFRAME_MS, connect, upsert_candles


class FakeExchange:
    """Fake exchange that returns deterministic synthetic OHLCV pages."""

    def __init__(self, symbol, timeframe, window_start_ms, window_end_ms):
        """
        Args:
            symbol: CCXT-format symbol (e.g., "BTC/USDT:USDT").
            timeframe: Timeframe (e.g., "15m").
            window_start_ms: Start of synthetic data window (epoch ms).
            window_end_ms: End of synthetic data window (epoch ms).
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_start_ms = window_start_ms
        self.window_end_ms = window_end_ms
        self.interval_ms = TIMEFRAME_MS[timeframe]
        self.calls = []  # Track all fetch_ohlcv calls for verification

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        """Fetch synthetic OHLCV data.

        Returns a list of [ts, open, high, low, close, volume] rows,
        or an empty list if since >= window_end_ms.
        """
        if since is None:
            since = self.window_start_ms

        # Track the call for verification in tests
        self.calls.append({"since": since, "limit": limit})

        # If requested start is at or past window end, return empty
        if since >= self.window_end_ms:
            return []

        candles = []
        ts = since
        while len(candles) < limit and ts < self.window_end_ms:
            # Synthetic price based on timestamp
            open_price = 100.0 + (ts % 1_000_000) / 10_000.0
            high = open_price + 1.0
            low = open_price - 0.5
            close = open_price + 0.5
            volume = 1000.0
            candles.append([ts, open_price, high, low, close, volume])
            ts += self.interval_ms

        return candles


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a temporary database connection."""
    db_file = tmp_path / "test.db"
    conn = connect(str(db_file))
    yield conn
    conn.close()


class TestBackfillFullRange:
    """Test that backfill_series fetches the entire data range."""

    def test_full_range_covered(self, temp_db):
        """After backfill_series, stored candle count equals expected count."""
        symbol = "BTCUSDT"
        ccxt_symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]

        # Create a synthetic window: 10 days of 15m candles (UTC)
        window_start = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        window_end = window_start + 10 * 24 * 60 * 60 * 1000  # 10 days

        exchange = FakeExchange(ccxt_symbol, timeframe, window_start, window_end)

        # Backfill
        result = backfill_series(
            temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
        )

        # Expected count: 10 days * 24 hours * 60 minutes / 15 minutes = 960 candles
        expected_count = (window_end - window_start) // interval_ms
        assert result.rows == expected_count
        assert result.complete is True
        assert result.reason is None

        # Verify stored count matches
        cursor = temp_db.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        stored_count = cursor.fetchone()[0]
        assert stored_count == expected_count

    def test_backfill_1d_full_range_no_duplicates(self, temp_db):
        """1d backfill paginates and stores a contiguous, duplicate-free series."""
        interval = TIMEFRAME_MS["1d"]
        window_start = config.date_to_ms("2023-01-01")
        window_end = window_start + 1300 * interval   # ~ the real backfill size
        fake = FakeExchange("BTC/USDT:USDT", "1d", window_start, window_end)

        result = backfill_series(temp_db, "BTCUSDT", "1d", exchange=fake,
                                 now_ms=window_end, start_ms=window_start)

        assert result.complete is True and result.reason is None
        total, distinct = temp_db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ts) FROM ohlcv WHERE symbol=? AND timeframe=?",
            ("BTCUSDT", "1d"),
        ).fetchone()
        assert total == distinct == 1300
        # >1 page: FakeExchange caps at limit=1000, so pagination is exercised
        assert len(fake.calls) >= 2
        assert all(ts % interval == 0 for (ts,) in temp_db.execute(
            "SELECT ts FROM ohlcv WHERE symbol=? AND timeframe=?", ("BTCUSDT", "1d")))


class TestBackfillResumability:
    """Test that backfill resumes correctly from pre-inserted data."""

    def test_resumability_no_duplicates(self, temp_db):
        """Pre-insert first half, run backfill, assert resumability and no duplicates."""
        symbol = "BTCUSDT"
        ccxt_symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]

        # Create a synthetic window: 10 days of 15m candles (UTC)
        window_start = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        window_end = window_start + 10 * 24 * 60 * 60 * 1000  # 10 days
        first_half_end = window_start + 5 * 24 * 60 * 60 * 1000  # 5 days

        # Generate and pre-insert first half
        first_half_candles = []
        ts = window_start
        while ts < first_half_end:
            open_price = 100.0 + (ts % 1_000_000) / 10_000.0
            first_half_candles.append(
                [ts, open_price, open_price + 1.0, open_price - 0.5, open_price + 0.5, 1000.0]
            )
            ts += interval_ms

        upsert_candles(temp_db, symbol, timeframe, first_half_candles)
        first_half_count = len(first_half_candles)

        # Create exchange covering full window
        exchange = FakeExchange(ccxt_symbol, timeframe, window_start, window_end)

        # Run backfill (should resume from first_half_end + interval_ms)
        result = backfill_series(
            temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
        )
        second_half_count = result.rows
        assert result.complete is True
        assert result.reason is None

        # Verify resumability: first call should start from last stored ts + interval
        assert len(exchange.calls) > 0
        first_call_since = exchange.calls[0]["since"]
        expected_resume_ts = first_half_candles[-1][0] + interval_ms
        assert first_call_since == expected_resume_ts

        # Verify total and no duplicates
        expected_total = (window_end - window_start) // interval_ms
        total_count = first_half_count + second_half_count
        assert total_count == expected_total

        # Verify no duplicate timestamps
        cursor = temp_db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ts) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        total_ts, distinct_ts = cursor.fetchone()
        assert total_ts == distinct_ts == expected_total


class TestNoDuplicateRows:
    """Test that duplicate upserts don't create duplicate rows."""

    def test_multiple_backfills_no_duplicates(self, temp_db):
        """Running backfill multiple times should not create duplicates."""
        symbol = "BTCUSDT"
        ccxt_symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]

        # Create a small synthetic window (UTC)
        window_start = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        window_end = window_start + 2 * 24 * 60 * 60 * 1000  # 2 days

        exchange = FakeExchange(ccxt_symbol, timeframe, window_start, window_end)

        # Backfill multiple times
        backfill_series(
            temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
        )
        backfill_series(
            temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
        )

        # Verify no duplicates
        cursor = temp_db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ts) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        total_ts, distinct_ts = cursor.fetchone()
        assert total_ts == distinct_ts

    def test_backfill_all_no_duplicates(self, temp_db):
        """backfill_all with multiple symbols/timeframes should not create duplicates."""
        ccxt_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        symbols = ["BTCUSDT", "ETHUSDT"]
        timeframes = ["15m", "1h"]

        window_start = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        window_end = window_start + 2 * 24 * 60 * 60 * 1000

        # Create fake exchanges for each symbol/timeframe
        exchanges_map = {
            (ccxt_symbol, timeframe): FakeExchange(ccxt_symbol, timeframe, window_start, window_end)
            for ccxt_symbol in ccxt_symbols
            for timeframe in timeframes
        }

        # Mock exchange selection: use a wrapper that routes to the right fake
        class MultiExchange:
            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
                return exchanges_map[(symbol, timeframe)].fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )

        multi_exchange = MultiExchange()

        # Backfill all
        backfill_all(
            temp_db,
            exchange=multi_exchange,
            symbols=symbols,
            timeframes=timeframes,
        )

        # Verify no duplicates for each symbol/timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                cursor = temp_db.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT ts) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                    (symbol, timeframe),
                )
                total_ts, distinct_ts = cursor.fetchone()
                assert total_ts == distinct_ts


class TestStartParameter:
    """Test the start_ms parameter for empty series."""

    def test_start_ms_overrides_backfill_start(self, temp_db):
        """start_ms should override config.BACKFILL_START for empty series."""
        symbol = "BTCUSDT"
        ccxt_symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]

        # Custom start time (not the config default, UTC)
        custom_start = int(datetime(2023, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
        window_end = custom_start + 5 * 24 * 60 * 60 * 1000

        exchange = FakeExchange(ccxt_symbol, timeframe, custom_start, window_end)

        # Backfill with custom start_ms
        backfill_series(
            temp_db,
            symbol,
            timeframe,
            exchange=exchange,
            now_ms=window_end,
            start_ms=custom_start,
        )

        # Verify first call started from custom_start
        assert len(exchange.calls) > 0
        assert exchange.calls[0]["since"] == custom_start

        # Verify data was stored
        cursor = temp_db.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        count = cursor.fetchone()[0]
        assert count > 0


class TestForwardProgressGuard:
    """Test the forward-progress guard against infinite loops."""

    def test_backfill_breaks_on_non_advancing_cursor(self, temp_db):
        """backfill_series should break and log error if fetch_ohlcv_page doesn't advance cursor."""
        symbol = "BTCUSDT"
        ccxt_symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]

        # Fixed timestamp for the candle (UTC)
        fixed_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        # Create a fake exchange that returns the same page twice
        # (simulating a stuck/frozen feed)
        class StuckExchange:
            def __init__(self):
                self.call_count = 0

            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
                self.call_count += 1
                if self.call_count == 1:
                    # First call: return one candle
                    return [[fixed_ts, 100.0, 101.0, 99.0, 100.5, 1000.0]]
                elif self.call_count == 2:
                    # Second call: return the SAME candle (cursor won't advance)
                    return [[fixed_ts, 100.0, 101.0, 99.0, 100.5, 1000.0]]
                else:
                    # Never reached due to break on non-advancing cursor
                    return []

        exchange = StuckExchange()
        window_end = fixed_ts + 10 * interval_ms

        # This should not hang; the forward-progress guard should break the loop
        result = backfill_series(
            temp_db,
            symbol,
            timeframe,
            exchange=exchange,
            now_ms=window_end,
        )

        # Should have processed at least the first page
        assert result.rows >= 1
        assert result.complete is False
        assert result.reason == "cursor-stalled"
        # Should not have looped indefinitely; call_count should be small
        assert exchange.call_count == 2  # First successful fetch, then detected non-advance


class TestBackfillResultContract:
    """Test the BackfillResult contract (C5): truncation reasons and completion."""

    def test_backfill_series_slow_advance_reports_incomplete(self, temp_db):
        """A cursor that crawls forward by 1ms per page (instead of a full
        interval) must be treated as stalled, not as legitimate progress.
        Capped at ~5 pages so a regression (old `cursor <= prev_cursor` guard,
        which lets 1ms crawls through) fails fast instead of hanging.
        """
        symbol = "BTCUSDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]
        start_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        class SlowAdvanceExchange:
            """Each page contains one candle whose ts only crawls by 1ms."""

            def __init__(self, start_ts, max_calls=5):
                self.start_ts = start_ts
                self.max_calls = max_calls
                self.call_count = 0

            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
                self.call_count += 1
                if self.call_count > self.max_calls:
                    return []
                ts = self.start_ts + (self.call_count - 1)  # advances by 1ms
                return [[ts, 100.0, 101.0, 99.0, 100.5, 1000.0]]

        exchange = SlowAdvanceExchange(start_ts)
        window_end = start_ts + 1000 * interval_ms

        result = backfill_series(
            temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
        )

        assert result.complete is False
        assert result.reason == "cursor-stalled"
        # Must have stopped well before exhausting the capped fake exchange.
        assert exchange.call_count <= 5

    def test_backfill_series_network_error_reports_incomplete(self, temp_db):
        """A ccxt.NetworkError mid-run ends the series with reason
        "network-error" and rows == rows upserted before the error."""
        symbol = "BTCUSDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]
        start_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        class NetworkErrorExchange:
            def __init__(self, start_ts):
                self.start_ts = start_ts
                self.call_count = 0

            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
                self.call_count += 1
                if self.call_count == 1:
                    return [[self.start_ts, 100.0, 101.0, 99.0, 100.5, 1000.0]]
                raise ccxt.NetworkError("connection reset")

        exchange = NetworkErrorExchange(start_ts)
        window_end = start_ts + 10 * interval_ms

        result = backfill_series(
            temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
        )

        assert result.rows == 1
        assert result.complete is False
        assert result.reason == "network-error"

    def test_backfill_series_transient_db_error_reports_incomplete(
        self, temp_db, monkeypatch
    ):
        """A transient sqlite3.OperationalError ("database is locked") from
        upsert_candles ends the series with reason "transient-db-error"."""
        symbol = "BTCUSDT"
        ccxt_symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]
        start_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        window_end = start_ts + 5 * interval_ms

        exchange = FakeExchange(ccxt_symbol, timeframe, start_ts, window_end)

        def raising_upsert(conn, symbol, timeframe, rows):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(
            "trading_bot.data.backfill.upsert_candles", raising_upsert
        )

        result = backfill_series(
            temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
        )

        assert result.rows == 0
        assert result.complete is False
        assert result.reason == "transient-db-error"

    def test_backfill_series_permanent_db_error_propagates(self, temp_db, monkeypatch):
        """A permanent sqlite3.OperationalError (e.g. missing table) must
        propagate, not be swallowed into a truncated BackfillResult."""
        symbol = "BTCUSDT"
        ccxt_symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]
        start_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        window_end = start_ts + 5 * interval_ms

        exchange = FakeExchange(ccxt_symbol, timeframe, start_ts, window_end)

        def raising_upsert(conn, symbol, timeframe, rows):
            raise sqlite3.OperationalError("no such table: ohlcv")

        monkeypatch.setattr(
            "trading_bot.data.backfill.upsert_candles", raising_upsert
        )

        with pytest.raises(sqlite3.OperationalError):
            backfill_series(
                temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
            )

    def test_backfill_series_complete_run(self, temp_db):
        """A normal run that exhausts exchange data reports complete=True,
        reason=None, with the correct row count."""
        symbol = "BTCUSDT"
        ccxt_symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        interval_ms = TIMEFRAME_MS[timeframe]
        window_start = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        window_end = window_start + 2 * 24 * 60 * 60 * 1000  # 2 days

        exchange = FakeExchange(ccxt_symbol, timeframe, window_start, window_end)

        result = backfill_series(
            temp_db, symbol, timeframe, exchange=exchange, now_ms=window_end
        )

        expected_count = (window_end - window_start) // interval_ms
        assert result.rows == expected_count
        assert result.complete is True
        assert result.reason is None
