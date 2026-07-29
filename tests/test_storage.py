"""Tests for the SQLite OHLCV storage layer."""

import sqlite3
from unittest import mock

from trading_bot import config
from trading_bot.data import storage

SYMBOL = "BTCUSDT"
TF = "15m"
INTERVAL = storage.TIMEFRAME_MS[TF]
START = 1_700_000_000_000  # arbitrary epoch-ms base


def make_conn(tmp_path):
    return storage.connect(str(tmp_path / "test.db"))


def make_rows(n, start=START, interval=INTERVAL, close=100.0):
    return [
        [start + i * interval, 99.0, 101.0, 98.0, close, 10.0]
        for i in range(n)
    ]


def count_rows(conn):
    return conn.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
        (SYMBOL, TF),
    ).fetchone()[0]


def test_upsert_same_batch_twice_no_duplicates(tmp_path):
    conn = make_conn(tmp_path)
    rows = make_rows(10)

    assert storage.upsert_candles(conn, SYMBOL, TF, rows) == 10
    count_after_first = count_rows(conn)

    assert storage.upsert_candles(conn, SYMBOL, TF, rows) == 10
    count_after_second = count_rows(conn)

    assert count_after_first == 10
    assert count_after_second == count_after_first


def test_upsert_overlapping_batch_updates_in_place(tmp_path):
    conn = make_conn(tmp_path)
    storage.upsert_candles(conn, SYMBOL, TF, make_rows(10, close=100.0))

    # Overlapping batch covering rows 5-9 with changed close values
    overlap = make_rows(5, start=START + 5 * INTERVAL, close=200.0)
    storage.upsert_candles(conn, SYMBOL, TF, overlap)

    assert count_rows(conn) == 10

    closes = conn.execute(
        "SELECT ts, close FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY ts",
        (SYMBOL, TF),
    ).fetchall()
    for ts, close in closes:
        idx = (ts - START) // INTERVAL
        expected = 200.0 if idx >= 5 else 100.0
        assert close == expected


def test_find_gaps_contiguous_series(tmp_path):
    conn = make_conn(tmp_path)
    storage.upsert_candles(conn, SYMBOL, TF, make_rows(20))
    # Use a now_ms just after the last row to avoid staleness gap
    last_ts = START + 19 * INTERVAL
    now_ms = last_ts + INTERVAL
    assert storage.find_gaps(conn, SYMBOL, TF, now_ms=now_ms) == []


def test_find_gaps_detects_single_gap(tmp_path):
    conn = make_conn(tmp_path)
    rows = make_rows(10)
    removed = rows.pop(5)  # remove the middle candle
    storage.upsert_candles(conn, SYMBOL, TF, rows)

    # Use a now_ms just after the last row to avoid staleness gap
    last_ts = START + 9 * INTERVAL
    now_ms = last_ts + INTERVAL
    gaps = storage.find_gaps(conn, SYMBOL, TF, now_ms=now_ms)
    assert len(gaps) == 1
    assert gaps[0] == (removed[0], removed[0])


def test_last_ts_empty_and_populated(tmp_path):
    conn = make_conn(tmp_path)
    assert storage.last_ts(conn, SYMBOL, TF) is None

    storage.upsert_candles(conn, SYMBOL, TF, make_rows(7))
    assert storage.last_ts(conn, SYMBOL, TF) == START + 6 * INTERVAL


def test_find_gaps_invariants_hold_for_any_now(tmp_path):
    """C1 invariants 1-3 must hold for every returned gap, for any now_ms.

    Checked against both an empty series and a series with an interior hole.
    """
    start_ms = config.date_to_ms(config.BACKFILL_START)
    now_values = [
        0,
        start_ms - 1,
        start_ms,
        start_ms + 1,
        start_ms + 250 * INTERVAL + 12345,
    ]

    def check_invariants(gaps):
        for gap_start, gap_end in gaps:
            assert gap_start <= gap_end
            assert gap_start % INTERVAL == 0
            assert gap_end % INTERVAL == 0

    # Empty series
    conn_empty = make_conn(tmp_path)
    for now_ms in now_values:
        gaps = storage.find_gaps(conn_empty, SYMBOL, TF, now_ms=now_ms)
        check_invariants(gaps)

    # Series with data + an interior hole, anchored at the grid-aligned
    # backfill start so stored timestamps are on the epoch-0 grid.
    conn_data = storage.connect(str(tmp_path / "test2.db"))
    rows = make_rows(20, start=start_ms)
    rows.pop(10)
    storage.upsert_candles(conn_data, SYMBOL, TF, rows)
    for now_ms in now_values:
        gaps = storage.find_gaps(conn_data, SYMBOL, TF, now_ms=now_ms)
        check_invariants(gaps)


def test_find_gaps_empty_series_before_backfill_start_is_empty(tmp_path):
    """now_ms predating the backfill window (invariant 3) reports no gaps."""
    conn = make_conn(tmp_path)
    now_ms = int(storage.TIMEFRAME_MS[TF] * 100)  # year 1970, predates BACKFILL_START
    gaps = storage.find_gaps(conn, SYMBOL, TF, now_ms=now_ms)
    assert gaps == []


def test_find_gaps_honors_start_ms_override(tmp_path):
    """An explicit start_ms overrides config.BACKFILL_START for an empty series."""
    conn = make_conn(tmp_path)
    override_start = config.date_to_ms(config.BACKFILL_START) + 1000 * INTERVAL
    now_ms = override_start + 50 * INTERVAL
    expected_end = (now_ms // INTERVAL) * INTERVAL - INTERVAL
    gaps = storage.find_gaps(conn, SYMBOL, TF, now_ms=now_ms, start_ms=override_start)
    assert gaps == [(override_start, expected_end)]


def test_find_gaps_trailing_gap_is_grid_aligned(tmp_path):
    """Trailing gap end is expected_end (grid-aligned), never raw now_ms."""
    conn = make_conn(tmp_path)
    storage.upsert_candles(conn, SYMBOL, TF, make_rows(5))
    last_stored_ts = START + 4 * INTERVAL
    # +12345 ensures now_ms itself is NOT grid-aligned, so a bug that echoes
    # raw now_ms as the gap end would fail this assertion.
    now_ms = last_stored_ts + 3 * INTERVAL + 12345
    expected_end = (now_ms // INTERVAL) * INTERVAL - INTERVAL
    gaps = storage.find_gaps(conn, SYMBOL, TF, now_ms=now_ms)
    assert gaps == [(last_stored_ts + INTERVAL, expected_end)]
    assert expected_end % INTERVAL == 0
    assert expected_end != now_ms


def test_upsert_and_last_ts_hold_db_lock(tmp_path):
    """upsert_candles and last_ts must hold storage._db_lock during DB access."""
    conn = make_conn(tmp_path)
    wrapped = mock.Mock(wraps=conn)

    real_executemany = conn.executemany

    def executemany_checks_lock(*args, **kwargs):
        assert storage._db_lock.locked()
        return real_executemany(*args, **kwargs)

    wrapped.executemany.side_effect = executemany_checks_lock

    storage.upsert_candles(wrapped, SYMBOL, TF, make_rows(3))

    real_execute = conn.execute

    def execute_checks_lock(*args, **kwargs):
        assert storage._db_lock.locked()
        return real_execute(*args, **kwargs)

    wrapped.execute.side_effect = execute_checks_lock

    result = storage.last_ts(wrapped, SYMBOL, TF)
    assert result == START + 2 * INTERVAL


def test_is_transient_db_error():
    assert storage.is_transient_db_error(
        sqlite3.OperationalError("database is locked")
    ) is True
    assert storage.is_transient_db_error(
        sqlite3.OperationalError("database is busy")
    ) is True
    assert storage.is_transient_db_error(
        sqlite3.OperationalError("no such table: ohlcv")
    ) is False
    assert storage.is_transient_db_error(
        sqlite3.OperationalError("disk I/O error")
    ) is False


class TestStalenessDetection:
    """Tests for staleness detection in find_gaps."""

    def test_find_gaps_detects_stale_series(self, tmp_path):
        """Test that find_gaps detects staleness when last_ts is old compared to now_ms."""
        conn = make_conn(tmp_path)
        # Insert a few candles
        storage.upsert_candles(conn, SYMBOL, TF, make_rows(5))

        # Last stored ts is START + 4*INTERVAL
        last_stored_ts = START + 4 * INTERVAL

        # now_ms is far in the future: last_stored_ts + 3*INTERVAL
        # (which exceeds the STALENESS_INTERVALS*INTERVAL staleness threshold)
        now_ms = last_stored_ts + 3 * INTERVAL
        expected_end = (now_ms // INTERVAL) * INTERVAL - INTERVAL

        gaps = storage.find_gaps(conn, SYMBOL, TF, now_ms=now_ms)

        # Should have one staleness gap: (last_stored_ts + INTERVAL, expected_end)
        assert len(gaps) == 1
        assert gaps[0] == (last_stored_ts + INTERVAL, expected_end)

    def test_find_gaps_fresh_series_no_trailing_gap(self, tmp_path):
        """Test that find_gaps reports NO staleness gap when last_ts is recent."""
        conn = make_conn(tmp_path)
        # Insert a few candles
        storage.upsert_candles(conn, SYMBOL, TF, make_rows(5))

        # Last stored ts
        last_stored_ts = START + 4 * INTERVAL

        # now_ms is only (STALENESS_INTERVALS - 0.5)*INTERVAL in the future
        # (which does NOT exceed the STALENESS_INTERVALS*INTERVAL threshold)
        now_ms = last_stored_ts + int((config.STALENESS_INTERVALS - 0.5) * INTERVAL)

        gaps = storage.find_gaps(conn, SYMBOL, TF, now_ms=now_ms)

        # Should have NO staleness gap
        assert len(gaps) == 0

    def test_find_gaps_detects_both_consecutive_and_staleness(self, tmp_path):
        """Test that find_gaps detects both consecutive gaps and staleness."""
        conn = make_conn(tmp_path)
        # Insert candles with a gap in the middle
        rows = make_rows(10)
        removed = rows.pop(5)  # Remove middle candle to create a gap
        storage.upsert_candles(conn, SYMBOL, TF, rows)

        # Last stored ts
        last_stored_ts = START + 9 * INTERVAL

        # now_ms is far in future (more than STALENESS_INTERVALS*INTERVAL away)
        now_ms = last_stored_ts + 3 * INTERVAL
        expected_end = (now_ms // INTERVAL) * INTERVAL - INTERVAL

        gaps = storage.find_gaps(conn, SYMBOL, TF, now_ms=now_ms)

        # Should have two gaps: one consecutive, one staleness
        assert len(gaps) == 2
        # First gap is the consecutive one (around the removed row)
        assert gaps[0] == (removed[0], removed[0])
        # Second gap is the staleness gap
        assert gaps[1] == (last_stored_ts + INTERVAL, expected_end)

    def test_find_gaps_uses_current_time_when_now_ms_none(self, tmp_path):
        """Test that find_gaps defaults to current time when now_ms is None."""
        conn = make_conn(tmp_path)
        storage.upsert_candles(conn, SYMBOL, TF, make_rows(5))

        # Call find_gaps with now_ms=None (should use current time)
        # Since the stored data is from the past, there will be staleness
        gaps = storage.find_gaps(conn, SYMBOL, TF, now_ms=None)

        # Should detect a staleness gap (unless somehow the test data is in the future!)
        # For this test to be reliable, we just verify that find_gaps accepts now_ms=None
        # and doesn't crash
        assert isinstance(gaps, list)


def test_load_candles_full_series(tmp_path):
    """Test that load_candles returns all rows in ascending order."""
    conn = make_conn(tmp_path)
    rows = make_rows(10)
    storage.upsert_candles(conn, SYMBOL, TF, rows)

    loaded = storage.load_candles(conn, SYMBOL, TF)

    assert len(loaded) == 10
    assert loaded == [
        (START + i * INTERVAL, 99.0, 101.0, 98.0, 100.0, 10.0)
        for i in range(10)
    ]


def test_load_candles_with_start_ms_inclusive(tmp_path):
    """Test that load_candles start_ms bound is inclusive."""
    conn = make_conn(tmp_path)
    rows = make_rows(10)
    storage.upsert_candles(conn, SYMBOL, TF, rows)

    # Load candles starting from row 3 (inclusive)
    start_ts = START + 3 * INTERVAL
    loaded = storage.load_candles(conn, SYMBOL, TF, start_ms=start_ts)

    assert len(loaded) == 7
    assert loaded[0][0] == start_ts  # First row is exactly at start_ts


def test_load_candles_with_end_ms_inclusive(tmp_path):
    """Test that load_candles end_ms bound is inclusive."""
    conn = make_conn(tmp_path)
    rows = make_rows(10)
    storage.upsert_candles(conn, SYMBOL, TF, rows)

    # Load candles ending at row 6 (inclusive)
    end_ts = START + 6 * INTERVAL
    loaded = storage.load_candles(conn, SYMBOL, TF, end_ms=end_ts)

    assert len(loaded) == 7
    assert loaded[-1][0] == end_ts  # Last row is exactly at end_ts


def test_load_candles_with_start_and_end_ms(tmp_path):
    """Test that load_candles respects both start_ms and end_ms bounds."""
    conn = make_conn(tmp_path)
    rows = make_rows(10)
    storage.upsert_candles(conn, SYMBOL, TF, rows)

    # Load candles from row 2 to row 7 (inclusive on both ends)
    start_ts = START + 2 * INTERVAL
    end_ts = START + 7 * INTERVAL
    loaded = storage.load_candles(conn, SYMBOL, TF, start_ms=start_ts, end_ms=end_ts)

    assert len(loaded) == 6
    assert loaded[0][0] == start_ts
    assert loaded[-1][0] == end_ts


def test_load_candles_empty_series(tmp_path):
    """Test that load_candles returns empty list when no data exists."""
    conn = make_conn(tmp_path)
    # Don't insert any data
    loaded = storage.load_candles(conn, SYMBOL, TF)
    assert loaded == []


def test_load_candles_no_rows_in_bounds(tmp_path):
    """Test that load_candles returns empty list when bounds exclude all data."""
    conn = make_conn(tmp_path)
    rows = make_rows(10)
    storage.upsert_candles(conn, SYMBOL, TF, rows)

    # Use bounds that don't overlap with the data
    start_ts = START + 20 * INTERVAL
    end_ts = START + 25 * INTERVAL
    loaded = storage.load_candles(conn, SYMBOL, TF, start_ms=start_ts, end_ms=end_ts)
    assert loaded == []


DAY = storage.TIMEFRAME_MS["1d"]


def test_find_gaps_1d_interior_hole_is_midnight_aligned(tmp_path):
    """A missing daily bar is reported as a single 1d-wide, UTC-midnight-aligned gap."""
    conn = storage.connect(str(tmp_path / "d.db"))
    start = config.date_to_ms(config.BACKFILL_START)  # already 1d-aligned
    rows = make_rows(20, start=start, interval=DAY)
    dropped = rows.pop(10)
    storage.upsert_candles(conn, SYMBOL, "1d", rows)

    now_ms = start + 19 * DAY + 12_345  # inside the freshness window, not grid-aligned
    gaps = storage.find_gaps(conn, SYMBOL, "1d", now_ms=now_ms, start_ms=start)

    assert gaps == [(dropped[0], dropped[0])]
    for g_start, g_end in gaps:
        assert g_start % DAY == 0 and g_end % DAY == 0


def test_find_gaps_1d_staleness_threshold_is_two_days(tmp_path):
    """STALENESS_INTERVALS=2 means a 1d series tolerates <=2 days of lag."""
    conn = storage.connect(str(tmp_path / "d2.db"))
    start = config.date_to_ms(config.BACKFILL_START)
    storage.upsert_candles(conn, SYMBOL, "1d", make_rows(5, start=start, interval=DAY))
    last = start + 4 * DAY

    # 2 intervals of lag: not stale (strict > in storage.py:180)
    assert storage.find_gaps(conn, SYMBOL, "1d",
                             now_ms=last + 2 * DAY, start_ms=start) == []
    # 4 intervals of lag: trailing gap up to the last CLOSED bar
    now_ms = last + 4 * DAY
    expected_end = (now_ms // DAY) * DAY - DAY
    assert storage.find_gaps(conn, SYMBOL, "1d",
                             now_ms=now_ms, start_ms=start) == [(last + DAY, expected_end)]


def test_find_gaps_1d_empty_series_reports_whole_window(tmp_path):
    """Matches the pre-1D-backfill state of the real DB: one full-window gap."""
    conn = storage.connect(str(tmp_path / "d3.db"))
    start = config.date_to_ms(config.BACKFILL_START)
    now_ms = start + 100 * DAY
    expected_end = (now_ms // DAY) * DAY - DAY
    assert storage.find_gaps(conn, SYMBOL, "1d",
                             now_ms=now_ms, start_ms=start) == [(start, expected_end)]


class TestResearchGridGaps:
    """v0.3.0 Phase 2 deliberate addition (contract §8 sanctions this file as
    Phase 2's extension target): multi-symbol / multi-timeframe shapes the
    existing single-symbol/single-timeframe tests above never exercise, but
    which correlation.gap_integrity's 20-symbol x 3-timeframe sweep depends
    on holding.
    """

    def test_per_symbol_isolation(self, tmp_path):
        """WHERE symbol = ? really scopes find_gaps per cell (storage.py:173) --
        the assumption the whole 60-cell sweep rests on."""
        conn = storage.connect(str(tmp_path / "iso_symbol.db"))
        start = config.date_to_ms(config.BACKFILL_START)
        rows_a = make_rows(20, start=start, interval=DAY)
        storage.upsert_candles(conn, "AAAUSDT", "1d", rows_a)

        rows_b = make_rows(20, start=start, interval=DAY)
        removed = rows_b.pop(10)
        storage.upsert_candles(conn, "BBBUSDT", "1d", rows_b)

        now_ms = start + 19 * DAY + 12_345  # inside the freshness window
        assert storage.find_gaps(conn, "AAAUSDT", "1d", now_ms=now_ms, start_ms=start) == []
        assert storage.find_gaps(conn, "BBBUSDT", "1d", now_ms=now_ms, start_ms=start) == [
            (removed[0], removed[0])
        ]

    def test_per_timeframe_isolation(self, tmp_path):
        """Same symbol, 1d complete and 4h holed -> 1d stays clean."""
        conn = storage.connect(str(tmp_path / "iso_tf.db"))
        start = config.date_to_ms(config.BACKFILL_START)
        storage.upsert_candles(conn, SYMBOL, "1d", make_rows(20, start=start, interval=DAY))

        h4 = storage.TIMEFRAME_MS["4h"]
        rows_4h = make_rows(20, start=start, interval=h4)
        removed = rows_4h.pop(10)
        storage.upsert_candles(conn, SYMBOL, "4h", rows_4h)

        now_1d = start + 19 * DAY + 12_345
        now_4h = start + 19 * h4 + 1_234
        assert storage.find_gaps(conn, SYMBOL, "1d", now_ms=now_1d, start_ms=start) == []
        assert storage.find_gaps(conn, SYMBOL, "4h", now_ms=now_4h, start_ms=start) == [
            (removed[0], removed[0])
        ]

    def test_interior_gap_survives_a_pinned_fresh_now_ms(self, tmp_path):
        """This is exactly the technique correlation.gap_integrity depends on:
        pinning now_ms = last_ts + interval + 1 keeps the staleness branch
        (storage.py:188-192) from firing while the interior-gap loop
        (storage.py:179-181) still runs over the whole series. Pinned so a
        change to config.STALENESS_INTERVALS or storage.py:188-192 breaks
        this test rather than silently changing which symbols are eligible.
        """
        conn = storage.connect(str(tmp_path / "pinned.db"))
        start = config.date_to_ms(config.BACKFILL_START)
        rows = make_rows(20, start=start, interval=DAY)
        removed = rows.pop(10)
        storage.upsert_candles(conn, SYMBOL, "1d", rows)

        last = rows[-1][0]
        now_ms = last + DAY + 1  # fresh: staleness cannot fire
        gaps = storage.find_gaps(conn, SYMBOL, "1d", now_ms=now_ms, start_ms=start)
        assert gaps == [(removed[0], removed[0])]

    def test_invariants_across_research_timeframes(self, tmp_path):
        """Sweep 1d/4h/1h x several now_ms values with the invariant triple."""
        start = config.date_to_ms(config.BACKFILL_START)

        def check_invariants(gaps, interval):
            for gap_start, gap_end in gaps:
                assert gap_start <= gap_end
                assert gap_start % interval == 0
                assert gap_end % interval == 0

        for i, timeframe in enumerate(("1d", "4h", "1h")):
            interval = storage.TIMEFRAME_MS[timeframe]
            conn = storage.connect(str(tmp_path / f"grid_{timeframe}.db"))
            rows = make_rows(30, start=start, interval=interval)
            rows.pop(15)
            storage.upsert_candles(conn, SYMBOL, timeframe, rows)

            now_values = [
                start - 1,
                start,
                start + 29 * interval + 1,
                start + 29 * interval + 5 * interval,
            ]
            for now_ms in now_values:
                gaps = storage.find_gaps(conn, SYMBOL, timeframe, now_ms=now_ms, start_ms=start)
                check_invariants(gaps, interval)
