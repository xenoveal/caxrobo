"""
Tests for CLI commands.
"""

import sys
import tempfile
from pathlib import Path

import ccxt
import pytest

from trading_bot import cli, config
from trading_bot.cli import _gap_report_command
from trading_bot.data.storage import TIMEFRAME_MS, connect, upsert_candles


class TestDateToMs:
	"""Tests for config.date_to_ms helper."""

	def test_date_to_ms_returns_correct_utc_epoch(self):
		"""Test that date_to_ms("2023-01-01") returns correct UTC epoch-ms."""
		# 2023-01-01T00:00:00Z is 1672531200 seconds = 1672531200000 ms
		expected_ms = 1672531200000
		result = config.date_to_ms("2023-01-01")
		assert result == expected_ms

	def test_date_to_ms_midnight_utc(self):
		"""Test that date_to_ms handles UTC midnight correctly."""
		# 2023-06-15T00:00:00Z
		result = config.date_to_ms("2023-06-15")
		# Verify by parsing back
		from datetime import datetime, timezone
		dt = datetime.fromtimestamp(result / 1000, tz=timezone.utc)
		assert dt.year == 2023
		assert dt.month == 6
		assert dt.day == 15
		assert dt.hour == 0
		assert dt.minute == 0
		assert dt.second == 0


def make_synthetic_candles(start_ts, count, interval_ms):
    """
    Generate synthetic OHLCV candles.

    Args:
        start_ts: Starting timestamp in milliseconds.
        count: Number of candles to generate.
        interval_ms: Interval between candles in milliseconds.

    Returns:
        List of [ts, open, high, low, close, volume] rows.
    """
    candles = []
    for i in range(count):
        ts = start_ts + (i * interval_ms)
        candles.append([ts, 100 + i, 110 + i, 90 + i, 105 + i, 1000 + i])
    return candles


class TestGapReportCommand:
    """Tests for the gap-report CLI command."""

    def test_gap_report_contiguous_returns_zero(self):
        """Test that gap-report returns 0 when all series are contiguous."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = connect(str(db_path))

            # Insert contiguous candles for all symbols and timeframes.
            # To ensure all data appears fresh (not stale), insert each timeframe
            # with enough duration that it ends near the same absolute time.
            start_ts = 1609459200000  # 2021-01-01 00:00:00 UTC

            # Use a common end time for all timeframes
            common_end = start_ts + 10 * TIMEFRAME_MS["4h"]

            for symbol in config.SYMBOLS:
                for timeframe in config.TIMEFRAMES:
                    interval_ms = TIMEFRAME_MS[timeframe]
                    # Calculate how many candles are needed to cover up to common_end
                    num_candles = (common_end - start_ts) // interval_ms
                    candles = make_synthetic_candles(start_ts, num_candles, interval_ms)
                    upsert_candles(conn, symbol, timeframe, candles)

            # Run gap-report with now_ms just after the common end time
            now_ms = common_end + TIMEFRAME_MS["15m"]
            exit_code = _gap_report_command(conn, now_ms=now_ms, start_ms=start_ts)

            # Assert no gaps found
            assert exit_code == 0

            conn.close()

    def test_gap_report_detects_single_gap(self, capsys):
        """Test that gap-report returns 1 and reports the gap when a gap exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = connect(str(db_path))

            # Insert contiguous candles for all symbols and timeframes
            start_ts = 1609459200000
            for symbol in config.SYMBOLS:
                for timeframe in config.TIMEFRAMES:
                    interval_ms = TIMEFRAME_MS[timeframe]
                    candles = make_synthetic_candles(start_ts, 10, interval_ms)
                    upsert_candles(conn, symbol, timeframe, candles)

            # Now remove one candle from BTCUSDT 15m (middle one)
            interval_ms = TIMEFRAME_MS["15m"]
            btc_start = start_ts
            gap_ts = btc_start + (5 * interval_ms)  # Middle of the 10 candles

            # Fetch and filter out the gap candle
            cursor = conn.execute(
                "SELECT ts FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY ts",
                ("BTCUSDT", "15m"),
            )
            all_ts = [row[0] for row in cursor.fetchall()]

            # Delete the middle candle
            conn.execute(
                "DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ? AND ts = ?",
                ("BTCUSDT", "15m", gap_ts),
            )
            conn.commit()

            # Run gap-report
            exit_code = _gap_report_command(conn)

            # Assert gap detected
            assert exit_code == 1

            # Check that output mentions the gap for BTCUSDT 15m
            captured = capsys.readouterr()
            assert "BTCUSDT 15m:" in captured.out
            assert "gap(s)" in captured.out

            conn.close()

    def test_gap_report_empty_database_has_gaps(self):
        """Test that gap-report returns 1 on an empty database (full backfill window is a gap)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = connect(str(db_path))

            # Empty database: all series are empty, so all report the full backfill window as a gap
            exit_code = _gap_report_command(conn, now_ms=1700000000000)

            # Assert gaps detected (empty series report the full backfill window)
            assert exit_code == 1

            conn.close()

    def test_gap_report_multiple_gaps_in_one_series(self, capsys):
        """Test that gap-report detects multiple gaps in a single series."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = connect(str(db_path))

            start_ts = 1609459200000
            interval_ms_1h = TIMEFRAME_MS["1h"]

            # Create a series with two consecutive gaps: [0-2], gap, [4-6], gap, [8-9]
            candles = []
            for i in [0, 1, 2, 4, 5, 6, 8, 9]:
                ts = start_ts + (i * interval_ms_1h)
                candles.append([ts, 100, 110, 90, 105, 1000])

            upsert_candles(conn, "ETHUSDT", "1h", candles)

            # Use the same end time as ETHUSDT 1h data for other series
            # The last ETHUSDT 1h candle is at start_ts + 9 * interval_ms_1h
            common_end = candles[-1][0]

            # Insert contiguous data for other symbol/timeframe combinations,
            # covering up to the same end time
            for symbol in config.SYMBOLS:
                for timeframe in config.TIMEFRAMES:
                    if symbol == "ETHUSDT" and timeframe == "1h":
                        continue  # Skip the one with gaps
                    interval_ms = TIMEFRAME_MS[timeframe]
                    num_candles = (common_end - start_ts) // interval_ms + 1
                    contiguous = make_synthetic_candles(start_ts, num_candles, interval_ms)
                    upsert_candles(conn, symbol, timeframe, contiguous)

            # Run gap-report with now_ms just after the common end. Use the
            # smallest timeframe's interval as the "just after" margin so
            # the other (contiguous, fresher-grained) series don't get
            # spuriously flagged stale relative to a coarser 1h margin
            # (R2#3: cross-timeframe staleness leak).
            now_ms = common_end + TIMEFRAME_MS["15m"]

            exit_code = _gap_report_command(conn, now_ms=now_ms, start_ms=start_ts)

            # Assert gap detected
            assert exit_code == 1

            # Check output
            captured = capsys.readouterr()
            assert "ETHUSDT 1h:" in captured.out
            assert "2 gap(s)" in captured.out  # Two consecutive gaps

            # Explicit assertion: every series OTHER than the seeded-gap one
            # must print OK. This guards against future cross-timeframe
            # staleness leaks (R2#3) instead of only checking the gapped
            # series in isolation.
            for symbol in config.SYMBOLS:
                for timeframe in config.TIMEFRAMES:
                    if symbol == "ETHUSDT" and timeframe == "1h":
                        continue
                    assert f"{symbol} {timeframe}: OK" in captured.out, (
                        f"{symbol} {timeframe} unexpectedly reported a gap"
                    )

            conn.close()

    def test_gap_report_as_of_recent_no_staleness(self, capsys):
        """Test that gap-report with --as-of set to recent time (no staleness) returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = connect(str(db_path))

            # Insert contiguous data for all symbols/timeframes
            start_ts = 1609459200000
            num_candles = 50

            for symbol in config.SYMBOLS:
                for timeframe in config.TIMEFRAMES:
                    interval_ms = TIMEFRAME_MS[timeframe]
                    candles = make_synthetic_candles(start_ts, num_candles, interval_ms)
                    upsert_candles(conn, symbol, timeframe, candles)

            # Calculate the end time of the last candle (same for all since same num_candles)
            interval_15m = TIMEFRAME_MS["15m"]
            last_ts = start_ts + (num_candles - 1) * interval_15m

            # Use as_of just after the last candle (within 2*interval, so no staleness)
            as_of_ms = last_ts + interval_15m

            # Run gap-report with the as_of time
            exit_code = _gap_report_command(conn, now_ms=as_of_ms, start_ms=start_ts)

            # Should return 0 (no gaps, no staleness)
            assert exit_code == 0

            conn.close()

    def test_gap_report_as_of_stale_detects_staleness(self, capsys):
        """Test that gap-report with --as-of set to old time detects staleness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = connect(str(db_path))

            # Insert contiguous data for all symbols/timeframes
            start_ts = 1609459200000
            num_candles = 10

            for symbol in config.SYMBOLS:
                for timeframe in config.TIMEFRAMES:
                    interval_ms = TIMEFRAME_MS[timeframe]
                    candles = make_synthetic_candles(start_ts, num_candles, interval_ms)
                    upsert_candles(conn, symbol, timeframe, candles)

            # Calculate the end time of the last candle
            interval_15m = TIMEFRAME_MS["15m"]
            last_ts = start_ts + (num_candles - 1) * interval_15m

            # Use as_of far in the future (more than 2*interval, so staleness detected)
            as_of_ms = last_ts + 5 * interval_15m

            # Run gap-report with the as_of time
            exit_code = _gap_report_command(conn, now_ms=as_of_ms, start_ms=start_ts)

            # Should return 1 (staleness gaps detected)
            assert exit_code == 1

            conn.close()


class TestMainArgv:
    """
    End-to-end tests that drive cli.main() through sys.argv (R3#5, R3#6, R3#7,
    R2#2's caller half). These exercise the argparse wiring itself, not just
    the inner _gap_report_command helper.
    """

    def test_main_gap_report_as_of_argv(self, tmp_path, monkeypatch, capsys):
        """gap-report --as-of wired through argv on an empty DB reports the
        full-window gap and exits 1 (R3#6)."""
        db_path = tmp_path / "test.db"
        monkeypatch.setattr(
            sys,
            "argv",
            ["trading-bot", "--db", str(db_path), "gap-report", "--as-of", "2026-07-05"],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "gap(s)" in captured.out

    def test_main_gap_report_bad_as_of_clean_error(self, tmp_path, monkeypatch, capsys):
        """gap-report --as-of with a malformed date exits 2 with a clean
        argparse usage error, not an uncaught traceback (R3#5)."""
        db_path = tmp_path / "test.db"
        monkeypatch.setattr(
            sys,
            "argv",
            ["trading-bot", "--db", str(db_path), "gap-report", "--as-of", "not-a-date"],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid date" in captured.err
        assert "Traceback" not in captured.err

    def test_main_backfill_bad_start_clean_error(self, tmp_path, monkeypatch, capsys):
        """backfill --start with a malformed date exits 2 with a clean
        argparse usage error, not an uncaught traceback (R3#5)."""
        db_path = tmp_path / "test.db"
        monkeypatch.setattr(
            sys,
            "argv",
            ["trading-bot", "--db", str(db_path), "backfill", "--start", "not-a-date"],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid date" in captured.err
        assert "Traceback" not in captured.err

    def test_main_backfill_incomplete_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        """A backfill run that hits a network error mid-run must be
        distinguishable from a clean run: prints INCOMPLETE(<reason>) and
        exits 1 (R2#2 caller half)."""
        db_path = tmp_path / "test.db"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "trading-bot",
                "--db",
                str(db_path),
                "backfill",
                "--symbol",
                "BTCUSDT",
                "--timeframe",
                "15m",
            ],
        )

        def _boom(*args, **kwargs):
            raise ccxt.NetworkError("simulated network outage")

        monkeypatch.setattr("trading_bot.data.backfill.fetch_ohlcv_page", _boom)

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "INCOMPLETE(network-error)" in captured.out

    def test_main_gap_report_start_override(self, tmp_path, monkeypatch):
        """A series backfilled from a later date than config.BACKFILL_START
        must not report the (BACKFILL_START, expected_end) window as a false
        gap once --start is supplied to match the real backfill date (R3#7)."""
        db_path = tmp_path / "test.db"
        conn = connect(str(db_path))

        late_start = config.date_to_ms("2024-06-01")
        seconds_per_day = 24 * 60 * 60 * 1000
        for symbol in config.SYMBOLS:
            for timeframe in config.TIMEFRAMES:
                interval_ms = TIMEFRAME_MS[timeframe]
                num_candles = seconds_per_day // interval_ms
                candles = make_synthetic_candles(late_start, num_candles, interval_ms)
                upsert_candles(conn, symbol, timeframe, candles)
        conn.close()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "trading-bot",
                "--db",
                str(db_path),
                "gap-report",
                "--start",
                "2024-06-01",
                "--as-of",
                "2024-06-02",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
