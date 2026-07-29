"""
Tests for the live polling loop.
"""

import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import ccxt
import pytest

from trading_bot import config
from trading_bot.data import poller, storage


class TestPollOnce:
    """Tests for poll_once function."""

    def test_poll_once_with_fixed_candles(self):
        """
        Test that poll_once fetches and upserts candles from a fake exchange.

        Verifies:
        - Candles are upserted into the database (count > 0)
        - Calling poll_once again is idempotent (same data)
        """
        # Create a temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = storage.connect(str(db_path))

            # Create a fake exchange that returns fixed candles
            fake_exchange = MagicMock()
            fake_exchange.fetch_ohlcv.return_value = [
                [1000, 100.0, 101.0, 99.0, 100.5, 1000.0],
                [1900, 100.5, 102.0, 100.0, 101.0, 1100.0],
            ]

            # First call should upsert 2 rows
            count1 = poller.poll_once(conn, "15m", exchange=fake_exchange)
            assert count1 > 0, "First poll should upsert rows"

            # Verify data is in the database
            cursor = conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE timeframe = ?", ("15m",)
            )
            db_count = cursor.fetchone()[0]
            assert db_count > 0, "Database should have rows after poll"

            # Second call should be idempotent (same rows upserted again)
            count2 = poller.poll_once(conn, "15m", exchange=fake_exchange)
            assert count2 > 0, "Second poll should also return rows"

            cursor = conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE timeframe = ?", ("15m",)
            )
            db_count_after = cursor.fetchone()[0]
            # Idempotent: same data inserted twice means no duplicates
            assert db_count_after == db_count, "Idempotent upsert should not increase count"

            conn.close()

    def test_poll_once_error_on_one_symbol(self, caplog):
        """
        Test that poll_once handles errors gracefully.

        When one symbol's fetch fails with a ccxt.NetworkError, the error is logged
        and other symbols are still fetched. The function does not raise.

        Verifies:
        - No exception is raised for ccxt errors
        - An error is logged for the failing symbol
        - Healthy symbols are still upserted
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = storage.connect(str(db_path))

            # Create a fake exchange that fails for the first symbol
            fake_exchange = MagicMock()
            call_count = [0]

            def fetch_side_effect(symbol, timeframe, since, limit):
                call_count[0] += 1
                if call_count[0] == 1:  # First call fails with ccxt error
                    raise ccxt.NetworkError("Network unavailable")
                # Subsequent calls return data
                return [
                    [2000, 100.0, 101.0, 99.0, 100.5, 1000.0],
                    [2900, 100.5, 102.0, 100.0, 101.0, 1100.0],
                ]

            fake_exchange.fetch_ohlcv.side_effect = fetch_side_effect

            # Call poll_once; it should not raise
            with caplog.at_level("ERROR"):
                total = poller.poll_once(conn, "15m", exchange=fake_exchange)

            # Should have upserted rows from the healthy symbols
            assert total > 0, "Healthy symbols should be upserted despite one failure"

            # Check that an error was logged
            error_records = [r for r in caplog.records if r.levelname == "ERROR"]
            assert len(error_records) > 0, "Error should be logged for failing symbol"
            assert "Network" in error_records[0].message or "unavailable" in error_records[0].message.lower()

            conn.close()

    def test_poll_once_utc_correct_now_ms(self):
        """
        Test that poll_once computes now_ms using UTC time (time.time()), not local time.

        Verifies:
        - now_ms is computed as int(time.time() * 1000), not datetime.utcnow().timestamp()
        - When a symbol has no historical data, since_ms is correctly computed as now_ms - (lookback * interval_ms)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = storage.connect(str(db_path))

            fake_exchange = MagicMock()
            fake_exchange.fetch_ohlcv.return_value = []

            # Monkeypatch time.time to return a known value
            fixed_time = 1609459200.5  # 2021-01-01 00:00:00.5 UTC
            with patch("trading_bot.data.poller.time.time", return_value=fixed_time):
                poller.poll_once(conn, "15m", exchange=fake_exchange)

            # Verify that the fetch_ohlcv was called with the correct since_ms
            # For 15m with lookback=3: since_ms = now_ms - (3 * 900_000)
            # now_ms = int(1609459200.5 * 1000) = 1609459200500
            # interval_ms for 15m = 900_000
            # since_ms = 1609459200500 - (3 * 900_000) = 1609459200500 - 2700000 = 1609456500500
            expected_now_ms = int(fixed_time * 1000)
            expected_since_ms = expected_now_ms - (3 * 900_000)

            # Check that fetch_ohlcv was called with correct since parameter
            assert fake_exchange.fetch_ohlcv.called, "fetch_ohlcv should have been called"
            # fetch_ohlcv is called with (symbol, timeframe, since=since_ms, limit=limit)
            call_kwargs = fake_exchange.fetch_ohlcv.call_args[1]
            actual_since_ms = call_kwargs.get("since")
            assert actual_since_ms == expected_since_ms, (
                f"Expected since_ms {expected_since_ms}, got {actual_since_ms}"
            )

            conn.close()

    def test_poll_once_catches_ccxt_network_error(self):
        """
        Test that poll_once catches ccxt.NetworkError and continues processing other symbols.

        Verifies:
        - ccxt.NetworkError raised by exchange is caught (logged, not raised)
        - poll_once continues to the next symbol
        - Healthy symbols are still processed and upserted
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = storage.connect(str(db_path))

            fake_exchange = MagicMock()
            call_count = [0]

            def fetch_side_effect(symbol, timeframe, since, limit):
                call_count[0] += 1
                if call_count[0] == 1:  # First call raises NetworkError
                    raise ccxt.NetworkError("Network unavailable")
                # Subsequent calls return data
                return [
                    [3000, 100.0, 101.0, 99.0, 100.5, 1000.0],
                ]

            fake_exchange.fetch_ohlcv.side_effect = fetch_side_effect

            # Should not raise despite the NetworkError
            total = poller.poll_once(conn, "15m", exchange=fake_exchange)

            # Healthy symbols should be upserted (total > 0)
            assert total > 0, "Healthy symbols should be upserted despite NetworkError"

            conn.close()

    def test_poll_once_propagates_non_ccxt_errors(self):
        """
        Test that poll_once propagates non-ccxt errors (e.g., ValueError, TypeError).

        Verifies:
        - Non-ccxt exceptions (ValueError, TypeError, etc.) are NOT caught
        - poll_once raises the error immediately
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = storage.connect(str(db_path))

            fake_exchange = MagicMock()
            fake_exchange.fetch_ohlcv.side_effect = ValueError("Invalid argument")

            # Should raise because ValueError is not a ccxt error
            with pytest.raises(ValueError, match="Invalid argument"):
                poller.poll_once(conn, "15m", exchange=fake_exchange)

            conn.close()

    def test_poll_once_concurrent_access(self, tmp_path):
        """
        Test that poll_once is thread-safe with concurrent access.

        storage.py owns the DB lock internally (upsert_candles/last_ts/find_gaps),
        so poll_once itself needs no locking; this is an integration-level check
        that concurrent callers still behave. The deterministic proof that the
        lock is held during DB access lives in test_storage.py.

        Verifies:
        - Multiple threads can safely call poll_once on the same connection
        - No exceptions escape from any thread
        - Data is correctly upserted despite concurrent access
        """
        # Create a shared temp-file database (in-memory won't work for multiple threads)
        db_path = tmp_path / "concurrent.db"
        conn = storage.connect(str(db_path))

        # Create a fake exchange that returns fixed candles
        fake_exchange = MagicMock()
        fake_exchange.fetch_ohlcv.return_value = [
            [5000, 100.0, 101.0, 99.0, 100.5, 1000.0],
            [5900, 100.5, 102.0, 100.0, 101.0, 1100.0],
        ]

        # List to collect exceptions from threads
        thread_exceptions: list[tuple[threading.Thread, Exception]] = []

        def thread_target(thread_id: int):
            """Target function for each thread to call poll_once."""
            try:
                poller.poll_once(conn, "15m", exchange=fake_exchange)
            except Exception as e:
                thread_exceptions.append((threading.current_thread(), e))

        # Launch 2 threads to poll the same timeframe concurrently
        threads = [
            threading.Thread(target=thread_target, args=(i,), name=f"poll-{i}")
            for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert no exception escaped any thread
        assert (
            len(thread_exceptions) == 0
        ), f"Threads raised exceptions: {thread_exceptions}"

        # Verify data is in the database
        cursor = conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE timeframe = ?", ("15m",)
        )
        db_count = cursor.fetchone()[0]
        assert db_count > 0, "Database should have rows from concurrent poll_once calls"

        conn.close()

    def test_poll_once_catches_sqlite_operational_error(self, tmp_path):
        """
        Test that poll_once catches a transient sqlite3.OperationalError and continues.

        Verifies:
        - A transient sqlite3.OperationalError (e.g., "database is locked") is caught
        - poll_once returns without raising
        - Healthy symbols are still processed
        """
        db_path = tmp_path / "test_sql_op.db"
        conn = storage.connect(str(db_path))

        fake_exchange = MagicMock()
        fake_exchange.fetch_ohlcv.return_value = [
            [6000, 100.0, 101.0, 99.0, 100.5, 1000.0]
        ]

        with patch(
            "trading_bot.data.poller.storage.upsert_candles",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            # Should not raise despite the transient OperationalError
            total = poller.poll_once(conn, "15m", exchange=fake_exchange)

        assert total == 0, "All symbols failed transiently, so nothing was upserted"

        conn.close()

    def test_poll_once_permanent_db_error_propagates(self, tmp_path):
        """
        Test that poll_once propagates permanent sqlite3.OperationalError.

        Verifies:
        - A non-transient sqlite3.OperationalError (e.g., "no such table") raised
          from upsert_candles is NOT swallowed
        - poll_once raises the error immediately (R3#4)
        """
        db_path = tmp_path / "test_sql_op_permanent.db"
        conn = storage.connect(str(db_path))

        fake_exchange = MagicMock()
        fake_exchange.fetch_ohlcv.return_value = [
            [6000, 100.0, 101.0, 99.0, 100.5, 1000.0]
        ]

        with patch(
            "trading_bot.data.poller.storage.upsert_candles",
            side_effect=sqlite3.OperationalError("no such table: ohlcv"),
        ):
            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                poller.poll_once(conn, "15m", exchange=fake_exchange)

        conn.close()

    def test_poll_once_propagates_non_caught_errors(self, tmp_path):
        """
        Test that poll_once propagates non-caught errors (TypeError, ValueError, etc.).

        Verifies:
        - TypeError raised inside the loop is NOT caught
        - poll_once raises the error immediately
        """
        db_path = tmp_path / "test_type_error.db"
        conn = storage.connect(str(db_path))

        fake_exchange = MagicMock()
        fake_exchange.fetch_ohlcv.side_effect = TypeError("Invalid type")

        # Should raise because TypeError is not a caught exception
        with pytest.raises(TypeError, match="Invalid type"):
            poller.poll_once(conn, "15m", exchange=fake_exchange)

        conn.close()


class TestBuildScheduler:
    """Tests for build_scheduler function."""

    def test_build_scheduler_covers_every_configured_timeframe(self):
        """One job per config.TIMEFRAMES entry, including the Phase 1 1d tier.

        Asserted against config rather than a literal set: _CRON_BY_TIMEFRAME is
        hand-maintained, so an unpolled timeframe is the failure mode this guards.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = storage.connect(str(db_path))

            scheduler = poller.build_scheduler(conn)
            jobs = scheduler.get_jobs()

            assert len(jobs) == len(config.TIMEFRAMES)
            assert {job.args[1] for job in jobs} == set(config.TIMEFRAMES)
            assert "1d" in {job.args[1] for job in jobs}
            assert not scheduler.running
            conn.close()
