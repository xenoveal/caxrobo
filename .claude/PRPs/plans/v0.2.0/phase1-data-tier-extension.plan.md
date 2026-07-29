# Plan: Data Tier Extension (PRD Phase 1)

## Summary
Add the `1d` timeframe to the ingestion layer (`storage.TIMEFRAME_MS`, `config.TIMEFRAMES`, `poller._CRON_BY_TIMEFRAME`) and backfill 1D bars for BTCUSDT/ETHUSDT/SOLUSDT from `BACKFILL_START` (2023-01-01), so Phase 4 can move `REGIME_TIMEFRAME` from `4h` to `1d`. `gap-report` and `backfill` already iterate `config.TIMEFRAMES`, so both extend to the new grid for free — the poller does **not**, and that is the one non-obvious code change. 1-minute bars are explicitly **out of v1 scope** (PRD Open Question #4); the extension point is documented, not built.

**Connectivity status — the PRD's known blocker is RESOLVED.** Re-verified 2026-07-26 from this machine:
- `GET https://fapi.binance.com/fapi/v1/ping` → **HTTP 200** in 0.29 s (was curl exit 60 / TLS failure on 2026-07-05).
- `ccxt.binanceusdm().fetch_ohlcv('BTC/USDT:USDT','1d',since=1672531200000,limit=5)` → 5 rows, first `[1672531200000, 16537.5, …]`.
- `'1d' in ex.timeframes` → `True`.
- All 3 symbols return 1000 1D rows from 2023-01-01 (BTC/ETH/SOL identical coverage).

Tasks 1–5 are pure code + offline tests (do them regardless). Task 6 is the live data acquisition (now unblocked). Task 7 is a **CONTINGENCY-ONLY** offline import path from `data.binance.vision` should connectivity regress — the endpoints there were also verified live today (see Task 7).

## User Story
As the bot's sole user, I want daily bars stored alongside 15m/1h/4h so that the regime classifier can be re-pointed at the 1D tier in Phase 4 without any change to the ingestion layer, and so `gap-report` proves the 1D series is complete and contiguous before any strategy work depends on it.

## Problem → Solution
**Current**: `TIMEFRAME_MS = {"15m", "1h", "4h"}` (`storage.py:18`) is the single chokepoint — `find_gaps`, `backfill_series`, `poll_once`, and `engine.run_backtest` all index it, so `1d` raises `KeyError` everywhere. `config.TIMEFRAMES` (`config.py:11`) drives what `backfill_all` and `_gap_report_command` enumerate. `data/ohlcv.db` holds zero 1D rows.
**Solution**: Add `"1d": 86_400_000` to `TIMEFRAME_MS`, add `"1d"` to `config.TIMEFRAMES`, add a UTC-anchored daily job to `poller._CRON_BY_TIMEFRAME`, run `backfill`, then `gap-report` green across all 12 symbol×timeframe cells.

## Metadata
- **Complexity**: **Low** (3 source files, ~8 changed lines of production code; the work is in verification, data acquisition, and the contingency path)
- **Source PRD**: `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`
- **PRD Phase**: Phase 1 — Data tier extension (parallel with 2 and 3, no dependencies)
- **Estimated Files**: 7 (1 new/contingency, 6 modified)

---

## UX Design

N/A — data-layer change. The only observable surface is the existing CLI: `backfill` and `gap-report` each gain three `1d` rows in their output tables (one per symbol). No new subcommands, no new flags.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/trading_bot/data/storage.py` | 18 | `TIMEFRAME_MS` — the single map every other module indexes. One-line change; a missing entry is a `KeyError`, not a soft failure |
| P0 | `src/trading_bot/data/storage.py` | 108-185 | `find_gaps` — grid alignment (`expected_end = (now_ms // interval) * interval - interval`), the three documented INVARIANTS, and the `STALENESS_INTERVALS` trailing-gap rule. Read the invariants block at 131-136 before touching anything |
| P0 | `src/trading_bot/config.py` | 10-16 | `SYMBOLS`, `TIMEFRAMES`, `BACKFILL_START = "2023-01-01"`, `STALENESS_INTERVALS = 2` |
| P0 | `src/trading_bot/data/poller.py` | 79-117 | `_CRON_BY_TIMEFRAME` is a **hand-maintained dict, not derived from `config.TIMEFRAMES`** — adding `1d` to config alone silently leaves the 1D series unpolled and permanently stale |
| P1 | `src/trading_bot/data/backfill.py` | 41-166 | `backfill_series` pagination (cursor = `last_row_ts + interval`, forward-progress guard at 124-129) and `backfill_all`, which iterates `config.TIMEFRAMES` — so 1D backfill needs zero code change |
| P1 | `src/trading_bot/cli.py` | 236-263 | `_gap_report_command` iterates `config.SYMBOLS × config.TIMEFRAMES` — "extend gap-report to the new grid" requires **no** CLI edit |
| P1 | `src/trading_bot/exchange/binance_client.py` | 77-106 | `fetch_ohlcv_page(symbol, timeframe, since_ms, limit=1000)`; passes `timeframe` straight through to ccxt, sorts ascending. `limit=1000` means ~1302 daily bars = 2 pages per symbol |
| P1 | `tests/test_poller.py` | 340-374 | `test_build_scheduler_has_three_jobs` hard-asserts `len(jobs) == 3` and `{"15m","1h","4h"}` — the one test that **breaks** on this change |
| P2 | `tests/test_storage.py` | 1-12, 95-165 | `TF`/`INTERVAL` module constants and the `find_gaps` invariant tests — the pattern new 1D gap tests must mirror |
| P2 | `tests/test_backfill.py` | 20-77 | `FakeExchange` — generates synthetic pages for any timeframe present in `TIMEFRAME_MS`; reuse it for the 1D backfill test, do not write a new fake |
| P2 | `src/trading_bot/regime/classifier.py` | 152-161 | `current_regime` already excludes the forming bar via `end_ms = now_ms - interval`. Confirms the 1D forming-bar hazard is handled on the live path (but see Task 6 GOTCHA for the backtest path) |

## External Documentation

- **Binance USDT-M klines REST** — `GET https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1d&limit=3`. **VERIFIED live 2026-07-26**: HTTP 200, returns 12-element arrays with `open_time` in epoch-ms, UTC-midnight aligned (`1784851200000 % 86_400_000 == 0`). Reached indirectly via ccxt; the plan does not call it directly.
- **Binance public data dumps (`data.binance.vision`)** — contingency source only. **VERIFIED live 2026-07-26**, see Task 7 for the exact URLs, ZIP layout, CSV header, and checksum format actually observed.
- No new Python dependency is needed: `ccxt` (already a dependency) supports `1d` on `binanceusdm` (`'1d' in ex.timeframes` → `True`, verified).

---

## Patterns to Mirror

### TIMEFRAME_MAP_SHAPE
```python
# SOURCE: src/trading_bot/data/storage.py:18 — flat dict, literal ms values with
# underscore separators, ordered coarsest-last. No derivation, no parsing of the
# timeframe string anywhere in the codebase.
TIMEFRAME_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
```

### CONFIG_TUPLE_SHAPE
```python
# SOURCE: src/trading_bot/config.py:10-12 — annotated tuples, ordered fine->coarse
SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h")
BACKFILL_START = "2023-01-01"
```

### POLLER_CRON_TABLE
```python
# SOURCE: src/trading_bot/data/poller.py:79-84 — "Finding 7: data-driven
# scheduling". Kwargs are splatted straight into scheduler.add_job(..., "cron", **cron_kwargs)
# at poller.py:109-115, so any CronTrigger kwarg (including `timezone`) is legal here.
_CRON_BY_TIMEFRAME = {
    "15m": {"minute": "0,15,30,45", "second": 10},
    "1h": {"minute": "0", "second": 10},
    "4h": {"hour": "0,4,8,12,16,20", "minute": "0", "second": 10},
}
```

### GAP_INVARIANT_TEST
```python
# SOURCE: tests/test_storage.py:100-130 — invariant sweep over several now_ms
# values, asserting grid alignment and non-inversion for every returned tuple.
def check_invariants(gaps):
    for gap_start, gap_end in gaps:
        assert gap_start <= gap_end
        assert gap_start % INTERVAL == 0
        assert gap_end % INTERVAL == 0
```

### BACKFILL_TEST_SHAPE
```python
# SOURCE: tests/test_backfill.py:81-112 — construct FakeExchange over a synthetic
# window, run backfill_series, assert COUNT(*) == COUNT(DISTINCT ts) and full coverage.
fake = FakeExchange("BTC/USDT:USDT", timeframe, window_start_ms, window_end_ms)
result = backfill_series(temp_db, "BTCUSDT", timeframe, exchange=fake,
                         now_ms=window_end_ms, start_ms=window_start_ms)
assert result.complete is True
```

---

## Current State of `data/ohlcv.db` (measured 2026-07-26, read-only `sqlite3`)

| symbol | timeframe | rows | first bar (UTC) | last bar (UTC) |
|---|---|---|---|---|
| BTCUSDT | 15m | 124,792 | 2022-12-31 17:00 | 2026-07-23 14:45 |
| BTCUSDT | 1h | 31,198 | 2022-12-31 17:00 | 2026-07-23 14:00 |
| BTCUSDT | 4h | 7,799 | 2022-12-31 20:00 | 2026-07-23 12:00 |
| ETHUSDT | 15m / 1h / 4h | 124,792 / 31,198 / 7,799 | same | same |
| SOLUSDT | 15m / 1h / 4h | 124,792 / 31,198 / 7,799 | same | same |
| **all** | **1d** | **0** | — | — |

DB file size 48.6 MB. **1D backfill adds ~1,302 rows per symbol (~3,906 total)** — `(expected_end − date_to_ms("2023-01-01")) / 86_400_000 + 1` as of 2026-07-26. That is ~0.3% of current row count; storage impact is negligible.

**Two facts that bite on the success criterion:**
1. `date_to_ms("2023-01-01") = 1672531200000`, and `1672531200000 % 86_400_000 == 0` — `BACKFILL_START` is already 1D-grid-aligned, so `find_gaps`' invariant #2 holds for 1D with no special-casing.
2. Existing series are **stale by 15 four-hour intervals / 229 fifteen-minute intervals** as of 2026-07-26 (last bar 2026-07-23). `STALENESS_INTERVALS = 2`, so a plain `gap-report` today reports trailing gaps on **all nine existing cells**, not just 1D. Task 8 therefore refreshes the whole grid, not only `1d`.

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/data/storage.py` | UPDATE | Add `"1d": 86_400_000` to `TIMEFRAME_MS` (line 18). Unblocks `find_gaps`/`backfill_series`/`poll_once` for the new grid |
| `src/trading_bot/config.py` | UPDATE | Add `"1d"` to `TIMEFRAMES` (line 11); add a comment recording that `1m` is deliberately excluded and where it would go |
| `src/trading_bot/data/poller.py` | UPDATE | Add a `"1d"` entry to `_CRON_BY_TIMEFRAME` with `timezone="UTC"`; update the docstring's boundary list (lines 5, 93-96) |
| `tests/test_poller.py` | UPDATE | `test_build_scheduler_has_three_jobs` → four jobs incl. `1d`; rename accordingly |
| `tests/test_storage.py` | UPDATE | Add 1D gap-detection tests (grid alignment at UTC midnight, interior hole, staleness at `2 × 1d`) |
| `tests/test_backfill.py` | UPDATE | Add a `1d` case to the full-range / no-duplicate coverage using the existing `FakeExchange` |
| `scripts/import_vision_klines.py` | CREATE (**CONTINGENCY ONLY**) | Offline 1D import from `data.binance.vision` monthly ZIPs. Write this **only if** Task 6's connectivity check fails |

## NOT Building

- **1-minute bars.** PRD Open Question #4 is unresolved and the PRD's MoSCoW lists 1m as `Could`. Backfilling 1m for 3 symbols over 3.5 years is ~1.84M bars/symbol (~5.5M rows, roughly 70× the current DB row count) — out of v1 scope. Extension point: add `"1m": 60_000` to `TIMEFRAME_MS` and `"1m"` to `config.TIMEFRAMES`, plus a `{"minute": "*", "second": 10}` cron entry. Nothing else in this plan changes.
- **Re-pointing `REGIME_TIMEFRAME` to `1d`** (`config.py:19`) — that is Phase 4, and Phase 4 also depends on Phase 2. Phase 1 only makes the bars *exist*. Do not touch `REGIME_TIMEFRAME`, `SIGNAL_PATTERN_TIMEFRAME`, `SIGNAL_TRIGGER_TIMEFRAME`, `REGIME_MIN_BARS`, or `MAX_HOLD_BARS_15M`.
- **Auditing hardcoded `15m` literals.** Explicitly Phase 4's scope ("audit all hardcoded 15m assumptions"). The grep was run for *this* plan's benefit (results in Notes) but no literal is changed here.
- **Deriving `TIMEFRAME_MS` from a timeframe-string parser.** Tempting, but the flat literal dict is the established pattern and 22 call sites across `src/` and `tests/` index it directly. Introducing parsing is unjustified scope.
- **Deriving `_CRON_BY_TIMEFRAME` from `config.TIMEFRAMES`.** Also tempting (it would have prevented this class of bug) but each timeframe needs a genuinely different cron expression; auto-derivation is a refactor, not this phase.
- **Fixing the pre-existing 4H poller timezone misalignment.** Discovered while writing this plan (see Risks) — report it, do not fix it here.
- **Any change to** `exchange/binance_client.py`, `data/backfill.py`, `cli.py`, `backtest/*`, `signals/*`, `regime/*`, `indicators/*`. All of them handle `1d` correctly the moment `TIMEFRAME_MS` knows about it.

---

## Step-by-Step Tasks

### Task 1: Add `1d` to `TIMEFRAME_MS`
- **ACTION**: Edit `src/trading_bot/data/storage.py:18`.
- **IMPLEMENT**:
  ```python
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
  ```
- **MIRROR**: TIMEFRAME_MAP_SHAPE — literal ms values, underscore separators, coarsest last.
- **IMPORTS**: None.
- **GOTCHA**: Do **not** write `24 * 3_600_000` or import a constant — `walkforward.py:33` already defines its own `DAY_MS = 86_400_000` and the two are deliberately independent (one is a bar interval, the other a fold-window unit). Duplicating the literal is correct here.
- **VALIDATE**:
  ```bash
  .venv/bin/python -c "from trading_bot.data.storage import TIMEFRAME_MS; \
  print(TIMEFRAME_MS['1d'], 1672531200000 % TIMEFRAME_MS['1d'])"
  ```
  EXPECT: `86400000 0`

### Task 2: Add `1d` to `config.TIMEFRAMES`
- **ACTION**: Edit `src/trading_bot/config.py:11`.
- **IMPLEMENT**:
  ```python
  # Phase 1 (v1.0 PRD): "1d" added so the regime tier can move up to daily bars
  # in Phase 4. "1m" is deliberately NOT here — PRD Open Question #4 (1-minute
  # exit resolution) is unresolved, and 1m would be ~1.84M bars/symbol/3.5yr.
  # If it is ever resolved to "yes": add "1m" here and "1m": 60_000 to
  # storage.TIMEFRAME_MS, plus a per-minute entry in poller._CRON_BY_TIMEFRAME.
  TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h", "1d")
  ```
- **MIRROR**: CONFIG_TUPLE_SHAPE.
- **IMPORTS**: None.
- **GOTCHA**: This single edit widens the grid for `backfill_all` (`backfill.py:151-152`) **and** `_gap_report_command` (`cli.py:253-254`) **and** four loops in `tests/test_cli.py` (lines 79, 104, 179, 209, 229, 260, 383) that iterate `config.TIMEFRAMES` to seed synthetic data. Those CLI tests were traced by hand and **all still pass** — each uses a `now_ms` close enough to `start_ts` that the 1D series' `expected_end` falls at or before `start_ms`, so `find_gaps` returns `[]` (invariant #3) or a single-bar series with no trailing gap. Do not pre-emptively "fix" `test_cli.py`; run it and confirm.
- **VALIDATE**:
  ```bash
  .venv/bin/python -c "from trading_bot import config; print(config.TIMEFRAMES)"
  .venv/bin/python -m pytest tests/test_cli.py -q
  ```
  EXPECT: `('15m', '1h', '4h', '1d')`; all `test_cli.py` tests pass with **no** edits to that file.

### Task 3: Add the 1D poller job (UTC-anchored)
- **ACTION**: Edit `src/trading_bot/data/poller.py:79-84`, and the two docstrings at line 5 and lines 93-96.
- **IMPLEMENT**:
  ```python
  # Mapping of timeframe to cron job schedule (Finding 7: data-driven scheduling)
  _CRON_BY_TIMEFRAME = {
      "15m": {"minute": "0,15,30,45", "second": 10},
      "1h": {"minute": "0", "second": 10},
      "4h": {"hour": "0,4,8,12,16,20", "minute": "0", "second": 10},
      # Daily bars close at 00:00 UTC. BackgroundScheduler() defaults to the
      # host's LOCAL timezone, so this job must pin timezone="UTC" explicitly —
      # on a UTC+7 host an unpinned {"hour": "0"} would fire at 17:00 UTC and
      # fetch the still-forming daily bar as if it were closed.
      "1d": {"hour": "0", "minute": "0", "second": 10, "timezone": "UTC"},
  }
  ```
  Update the module docstring (line 5) and `build_scheduler`'s docstring (93-96) to list the 1D job.
- **MIRROR**: POLLER_CRON_TABLE. `timezone` is a legal `CronTrigger` kwarg (verified: `apscheduler 3.11.3`, `CronTrigger.__init__` accepts `timezone`), and `poller.py:109-115` splats `**cron_kwargs` into `add_job`, so no signature change is needed.
- **IMPORTS**: None.
- **GOTCHA**: This is the **only** place in the codebase where the timeframe set is duplicated outside `config.TIMEFRAMES`/`TIMEFRAME_MS`. Skipping it does not raise — `poll_once` is simply never called for `1d`, the series silently drifts, and `gap-report` starts failing on a trailing staleness gap days later. Verified on this host: `date` → `WIB` (UTC+7), so the timezone pin is load-bearing, not defensive boilerplate.
- **VALIDATE**:
  ```bash
  .venv/bin/python -c "
  from trading_bot.data.poller import _CRON_BY_TIMEFRAME
  from trading_bot import config
  assert set(_CRON_BY_TIMEFRAME) == set(config.TIMEFRAMES), set(config.TIMEFRAMES) ^ set(_CRON_BY_TIMEFRAME)
  print('cron table covers every configured timeframe')"
  ```

### Task 4: Update the scheduler-job-count test
- **ACTION**: Edit `tests/test_poller.py:344-374`.
- **IMPLEMENT**: Rename `test_build_scheduler_has_three_jobs` → `test_build_scheduler_covers_every_configured_timeframe` and make it derive from config instead of hardcoding, so the next tier addition cannot silently skip the cron table:
  ```python
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
  ```
- **MIRROR**: The existing test body verbatim (tempdir + `storage.connect` + `job.args[1]` + not-running assertion) — only the expected set changes.
- **IMPORTS**: `from trading_bot import config` — check the top of `tests/test_poller.py`; it already imports `config` (used by `poll_once` tests). Do not add a duplicate import.
- **GOTCHA**: `job.args[1]` is the timeframe because `add_job(..., args=(conn, timeframe))` at `poller.py:112`. Keep that indexing; do not switch to `job.id` (ids are auto-generated UUIDs).
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_poller.py -q` → all pass.

### Task 5: Offline tests for 1D gap detection and 1D backfill
- **ACTION**: Add tests to `tests/test_storage.py` and `tests/test_backfill.py`. Everything in this task runs with **no network**.
- **IMPLEMENT**: In `tests/test_storage.py` (module constants `TF = "15m"` / `INTERVAL` stay as-is; use locals for the 1D cases):
  ```python
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
  ```
  In `tests/test_backfill.py`, extend the existing coverage by parametrizing or adding a `1d` case that reuses `FakeExchange`:
  ```python
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
  ```
- **MIRROR**: GAP_INVARIANT_TEST for the storage tests; BACKFILL_TEST_SHAPE and the existing `FakeExchange` (`tests/test_backfill.py:20-66`) for the backfill test — it derives its interval from `TIMEFRAME_MS[timeframe]` at line 35, so it needs **zero** changes to support `1d`.
- **IMPORTS**: `tests/test_storage.py` already imports `config` and `storage`; `tests/test_backfill.py` already imports `TIMEFRAME_MS`, `backfill_series`, and `FakeExchange` locally — add `from trading_bot import config` there if it is not already present (check line 16-17).
- **GOTCHA**: The staleness comparison at `storage.py:180` is **strict** `>` (`now_ms - anchor > STALENESS_INTERVALS * interval`), so lag of *exactly* 2 intervals is NOT stale. The `last + 2 * DAY` assertion above depends on that; if it fails, the production comparison changed, not the test.
- **VALIDATE**:
  ```bash
  .venv/bin/python -m pytest tests/test_storage.py tests/test_backfill.py -q
  ```

### Task 6: Live 1D backfill for all 3 symbols
- **ACTION**: Run the existing CLI. No code change. **Re-verify connectivity first** — the PRD lists this as blocked and that state can change back.
- **IMPLEMENT**:
  ```bash
  # 6a. Connectivity gate. Confirmed HTTP 200 on 2026-07-26; if this fails, STOP and go to Task 7.
  curl -s -o /dev/null -w "fapi:%{http_code}\n" --max-time 15 https://fapi.binance.com/fapi/v1/ping
  RUN_NETWORK_TESTS=1 .venv/bin/python -m pytest tests/test_binance_client.py -m network -q

  # 6b. 1D only, all three symbols, from BACKFILL_START.
  .venv/bin/python -m trading_bot.cli backfill --timeframe 1d
  ```
  EXPECT (per `cli.py:156-165`): three rows, `Rows` ≈ 1300–1310 each, `Status` = `OK`. Any `INCOMPLETE(<reason>)` exits 1; `reason` is one of `cursor-stalled` / `network-error` / `exchange-error` / `transient-db-error` (`backfill.py:29-34`).
  ```bash
  # 6c. Verify what landed.
  sqlite3 data/ohlcv.db "SELECT symbol, COUNT(*), COUNT(DISTINCT ts), \
    datetime(MIN(ts)/1000,'unixepoch'), datetime(MAX(ts)/1000,'unixepoch'), \
    SUM(ts % 86400000 <> 0) AS misaligned \
    FROM ohlcv WHERE timeframe='1d' GROUP BY symbol;"
  ```
  EXPECT: 3 rows; `COUNT(*) == COUNT(DISTINCT ts)`; min = `2023-01-01 00:00:00`; max = yesterday or today; `misaligned = 0`.
- **MIRROR**: `backfill --timeframe 1d` uses the same `backfill_all(timeframes=["1d"])` path as every prior backfill (`cli.py:151-153`); `--timeframe` is `action="append"` so repeating it adds timeframes.
- **IMPORTS**: N/A.
- **GOTCHA (forming-bar contamination)**: `fetch_ohlcv_page` returns the **currently-forming** daily bar, and `upsert_candles` stores it (`INSERT OR REPLACE`, `storage.py:85`). The live path is safe — `classifier.current_regime` sets `end_ms = now_ms - interval` to exclude it (`classifier.py:158-161`) — but `backtest.engine._df` (`engine.py:83-89`) loads **all** rows unfiltered, so a partial 1D bar can sit in the DB for up to 24 h and taint the tail of a backtest. At 4h that window was 4 h; at 1D it is 24 h. Mitigation for this phase: after backfill, confirm the max stored 1D `ts` is a *closed* bar with
  `sqlite3 data/ohlcv.db "SELECT symbol, MAX(ts) FROM ohlcv WHERE timeframe='1d' GROUP BY symbol;"`
  and if it equals `floor(now_ms/86400000)*86400000`, re-run the backfill after 00:00 UTC or bound the eventual backtest with `--end`. Record this as a Phase 4 input; do **not** add row-filtering logic to `engine.py` here (out of scope).
- **VALIDATE**: 6c returns clean; then Task 8.

### Task 7: CONTINGENCY — offline 1D import from `data.binance.vision`
- **ACTION**: **Only if Task 6a fails.** Create `scripts/import_vision_klines.py`. Skip this task entirely when the REST path works — do not build it speculatively.
- **IMPLEMENT**: A standalone stdlib-only script (`urllib.request`, `zipfile`, `csv`, `hashlib`, `io`) that downloads Binance's published monthly UM-futures kline dumps, converts them to the 6-column ccxt row shape, and hands them to the **existing** `storage.upsert_candles` — so the idempotent-upsert and schema guarantees are unchanged and only the *transport* differs.

  **Endpoints and formats verified live 2026-07-26** (do not substitute unverified paths):
  | What | Value | Verified |
  |---|---|---|
  | Monthly ZIP | `https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/1d/{SYMBOL}-1d-{YYYY}-{MM}.zip` | HTTP 200 for `SOLUSDT-1d-2026-06.zip` |
  | Daily ZIP (current partial month) | `https://data.binance.vision/data/futures/um/daily/klines/{SYMBOL}/1d/{SYMBOL}-1d-{YYYY}-{MM}-{DD}.zip` | HTTP 200 for `BTCUSDT-1d-2026-07-20.zip` |
  | Checksum | same URL + `.CHECKSUM`; body is `"<sha256>  <filename>"` | HTTP 200, e.g. `47d6fcfde…8a61  SOLUSDT-1d-2026-06.zip` |
  | ZIP contents | exactly one `{SYMBOL}-1d-{period}.csv` | confirmed |
  | CSV **has a header row** | `open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore` | confirmed on both files |
  | Row shape | `1780272000000,82.4000,83.0800,79.0600,81.2500,22141295.39,…` — `open_time` epoch-**ms**, UTC-midnight aligned | confirmed |

  Core of the script:
  ```python
  ROW_COLS = ("open_time", "open", "high", "low", "close", "volume")

  def rows_from_zip(blob: bytes) -> list[list]:
      """Parse one vision kline ZIP into ccxt-shaped [ts, o, h, l, c, v] rows.

      Binance added a header row to these dumps; the first field of a data row
      always parses as an int, so sniff rather than assume either way.
      """
      z = zipfile.ZipFile(io.BytesIO(blob))
      (name,) = z.namelist()
      out: list[list] = []
      for rec in csv.reader(io.StringIO(z.read(name).decode())):
          try:
              ts = int(rec[0])
          except ValueError:
              continue  # header row
          out.append([ts, float(rec[1]), float(rec[2]),
                      float(rec[3]), float(rec[4]), float(rec[5])])
      out.sort(key=lambda r: r[0])
      return out
  ```
  Drive it over `config.SYMBOLS`, months from `BACKFILL_START` to the last complete month, plus daily files for the current month; verify each ZIP's sha256 against its `.CHECKSUM` before parsing; call `storage.upsert_candles(conn, symbol, "1d", rows)` per file.
- **MIRROR**: `backfill_series`'s division of labour — fetch/parse in the caller, persistence exclusively through `storage.upsert_candles`. Log via `logging.getLogger("trading_bot")` like every other module.
- **IMPORTS**: stdlib only (`csv`, `hashlib`, `io`, `logging`, `urllib.request`, `zipfile`) + `from trading_bot import config` + `from trading_bot.data import storage`. **Do not add a dependency** to `pyproject.toml`.
- **GOTCHA**: The vision dumps are `futures/um/` (USDT-M perpetuals) — the same instrument `ccxt.binanceusdm` serves. `spot/` or `futures/cm/` (COIN-M) would silently produce a different price series. Also: `data.binance.vision` is CloudFront-fronted and resolves independently of `fapi.binance.com`, which is exactly why it is a viable fallback for a `fapi`-specific block — but it is **not** a general fix; the live poller (Task 3) still needs `fapi`, so a persistent block means 1D data is backfillable but not maintainable, and that must be escalated to the user rather than papered over.
- **VALIDATE**: after import, run Task 6c's SQL. Row counts should match the REST path to within the current partial month.

### Task 8: Refresh the full grid and drive `gap-report` to zero
- **ACTION**: Run the CLI. No code change. This is the phase's stated success signal.
- **IMPLEMENT**:
  ```bash
  # 8a. Existing series are stale (last bar 2026-07-23 vs today) — 15 x 4h and
  # 229 x 15m intervals of lag, far past STALENESS_INTERVALS=2. Refresh all.
  .venv/bin/python -m trading_bot.cli backfill

  # 8b. The success criterion: zero gaps across every symbol x timeframe.
  .venv/bin/python -m trading_bot.cli gap-report; echo "exit=$?"
  ```
  EXPECT: **12 lines**, all `"<SYMBOL> <TF>: OK"` (3 symbols × 4 timeframes), `exit=0`.
- **MIRROR**: `_gap_report_command` (`cli.py:236-263`) prints one line per cell and returns `1` if any cell has gaps — no edit needed for the wider grid.
- **IMPORTS**: N/A.
- **GOTCHA**: If a cell still reports a gap, distinguish the two causes before touching code:
  - a **trailing** gap ending at `expected_end` = staleness → the backfill did not reach now; re-run 8a, or pin the report with `gap-report --as-of YYYY-MM-DD` to the real data end.
  - an **interior** gap = a genuine exchange hole. Binance UM does have historical maintenance halts; a *real* missing daily bar cannot be conjured. Document it and either re-run `backfill` (which resumes from `last_ts`, so it will **not** re-fetch behind an interior hole — `backfill.py:73-75`) or delete rows after the hole so the cursor rewinds. Do not "fix" `find_gaps` to hide it.
  - `gap-report --start` exists for series legitimately backfilled from a later date than `BACKFILL_START` (`cli.py:72-76`). Note the current DB's 15m/1h series actually begin **2022-12-31 17:00**, *before* `BACKFILL_START` — extra leading data creates no gap, so no override is needed.
- **VALIDATE**: `exit=0` with 12 `OK` lines. Record the final per-cell row counts in the trial log.

### Task 9: Full-suite regression + record the outcome
- **ACTION**: Run the whole suite and note the state for Phase 4.
- **IMPLEMENT**:
  ```bash
  .venv/bin/python -m pytest tests/ -q
  ```
  EXPECT: **baseline measured before this phase was `188 passed, 1 skipped` in 2.6 s** (the skip is `test_fetch_ohlcv_page_network_smoke`, gated on `RUN_NETWORK_TESTS`). After this phase: `≥ 192 passed, 1 skipped`, zero failures. Then append to `.claude/PRPs/plans/` notes or the PRD's phase table: 1D row counts per symbol, the exact `gap-report` timestamp, and whether Task 6 or Task 7 supplied the data.
- **MIRROR**: N/A.
- **IMPORTS**: N/A.
- **GOTCHA**: Use `.venv/bin/python -m pytest`, not a bare `pytest`. Per the repo's known quirk, the venv must be built with Homebrew `python@3.11` (`/opt/homebrew/opt/python@3.11/bin/python3`); the existing `.venv` is 3.11.6 and works. `python3 -m venv` with the default Homebrew 3.12/3.14 on this host fails with a broken `pyexpat`/`ensurepip`. Do not recreate the venv.
- **VALIDATE**: Suite green; PRD phase-1 status flippable to `complete`.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| `TIMEFRAME_MS['1d']` | — | `86_400_000` | — |
| `BACKFILL_START` 1d-alignment | `1672531200000 % 86_400_000` | `0` | Boundary: guarantees `find_gaps` invariant #2 |
| 1D interior hole | 20 daily bars, index 10 dropped | one gap `(ts, ts)`, both ends `% 86_400_000 == 0` | — |
| 1D staleness at exactly 2 intervals | `now = last + 2*DAY` | `[]` (comparison is strict `>`) | **Edge: off-by-one on the threshold** |
| 1D staleness at 4 intervals | `now = last + 4*DAY` | `[(last+DAY, expected_end)]` | — |
| 1D empty series | no rows, `now = start + 100*DAY` | `[(start, expected_end)]` — full window | Matches the real pre-backfill DB state |
| 1D `now_ms` predating window | `now < start` | `[]` (invariant #3) | Edge: covered by the existing invariant sweep |
| 1D backfill pagination | `FakeExchange`, 1300-bar window, `limit=1000` | `complete=True`, 1300 rows, `COUNT == COUNT(DISTINCT)`, ≥2 fetch calls | Edge: multi-page cursor advance |
| 1D backfill idempotence | run `backfill_series` twice | row count unchanged | — |
| Scheduler job coverage | `build_scheduler(conn)` | `len(jobs) == len(config.TIMEFRAMES)`, set includes `1d` | Guards the hand-maintained cron table |
| `_CRON_BY_TIMEFRAME` ↔ `config.TIMEFRAMES` | set symmetric difference | empty | The Task 3 failure mode, asserted |
| `test_cli.py` gap-report suite | unchanged tests, widened `config.TIMEFRAMES` | all still pass, **no test edits** | Regression: confirms the widening is transparent |

### Edge Cases Checklist
- [x] UTC-midnight alignment of every 1D `ts` (asserted in SQL and in tests)
- [x] `BACKFILL_START` lands on the 1D grid (so no special-case in `find_gaps`)
- [x] Staleness threshold is strict `>`, so exactly-2-intervals is fresh
- [x] Multi-page 1D pagination (~1302 bars vs `limit=1000`)
- [x] Idempotent re-backfill produces no duplicates
- [x] Forming (unclosed) 1D bar stored by backfill/poller — live path filtered by `classifier`, backtest path flagged for Phase 4
- [x] Local-timezone scheduler default vs UTC-aligned daily bar (`timezone="UTC"` pin)
- [x] Empty 1D series before backfill reports one full-window gap, not a crash
- [x] Network unavailable → contingency source with verified URLs + checksum
- [ ] Concurrent access — N/A beyond the existing `storage._db_lock`, unchanged
- [ ] Permission denied — N/A

---

## Validation Commands

### Static Analysis
```bash
.venv/bin/python -m py_compile src/trading_bot/data/storage.py \
  src/trading_bot/config.py src/trading_bot/data/poller.py
```
EXPECT: zero syntax errors. No linter or type checker is configured — `pyproject.toml` declares only `ccxt`, `pandas`, `apscheduler`, `python-dotenv` (+ `pytest` as the sole dev extra) and a `[tool.pytest.ini_options]` block with a `network` marker. There is no `Makefile`, `noxfile.py`, `tox.ini`, or `setup.cfg`. Do not invent a lint/typecheck step.

### Unit Tests
```bash
.venv/bin/python -m pytest tests/test_storage.py tests/test_backfill.py \
  tests/test_poller.py tests/test_cli.py -q
```

### Full Test Suite
```bash
.venv/bin/python -m pytest tests/ -q
```
EXPECT: `≥192 passed, 1 skipped`. Pre-change baseline: `188 passed, 1 skipped`.

### Consistency Greps
```bash
grep -rn "TIMEFRAME_MS = " src/
grep -rn "TIMEFRAMES: tuple" src/
grep -rn "_CRON_BY_TIMEFRAME = " src/
```
EXPECT: exactly one definition each, all three containing `1d`.

### Network Smoke (opt-in)
```bash
RUN_NETWORK_TESTS=1 .venv/bin/python -m pytest tests/test_binance_client.py -m network -q
```
EXPECT: 1 passed. Skipped without the env var (`tests/test_binance_client.py:146-150`).

### Manual Validation (the phase's success signal)
```bash
.venv/bin/python -m trading_bot.cli backfill
.venv/bin/python -m trading_bot.cli gap-report; echo "exit=$?"
sqlite3 data/ohlcv.db "SELECT symbol, timeframe, COUNT(*), \
  datetime(MIN(ts)/1000,'unixepoch'), datetime(MAX(ts)/1000,'unixepoch') \
  FROM ohlcv GROUP BY symbol, timeframe ORDER BY symbol, timeframe;"
```
- [ ] `gap-report` prints 12 `OK` lines and exits 0
- [ ] 1D row count is ~1,300 per symbol, min bar `2023-01-01 00:00:00`
- [ ] `SUM(ts % 86400000 <> 0) = 0` for every 1D series

---

## Acceptance Criteria
- [ ] All applicable tasks completed (Task 7 skipped when Task 6a succeeds — record which path ran)
- [ ] All validation commands pass
- [ ] Tests written and passing (3 new storage tests, 1 new backfill test, 1 rewritten poller test)
- [ ] No type errors — no type checker configured, N/A
- [ ] No lint errors — no linter configured, N/A
- [ ] Matches UX design — N/A, data-layer change
- [ ] `gap-report` exits 0 across all 12 symbol × timeframe cells, including 1D

## Completion Checklist
- [ ] Code follows discovered patterns (literal-ms `TIMEFRAME_MS`, annotated config tuples, data-driven cron table)
- [ ] Error handling matches codebase style — unchanged; `BackfillResult(complete=False, reason=...)` still carries every failure mode
- [ ] Logging follows codebase conventions (`logging.getLogger("trading_bot")`; contingency script included)
- [ ] Tests follow test patterns (`FakeExchange` reused not reinvented; `check_invariants` shape mirrored)
- [ ] No hardcoded values outside the two canonical maps — timeframe strings appear only in `TIMEFRAME_MS`, `config.TIMEFRAMES`, `_CRON_BY_TIMEFRAME`
- [ ] Documentation updated — `poller.py` docstrings list the 1D job; `config.py` records why `1m` is excluded and where it would go
- [ ] No unnecessary scope additions — `REGIME_TIMEFRAME` untouched, no 15m-literal audit, no cron-table refactor
- [ ] Self-contained — no questions needed during implementation

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `fapi.binance.com` regresses to the 2026-07-05 TLS-blocked state mid-backfill | Low-Medium (it was blocked 3 weeks ago; verified 200 today) | High — blocks Phase 4 | Task 6a gates on an explicit `ping` before any run; Task 7 is a pre-researched, URL-verified offline path via `data.binance.vision` (independent host). Note the fallback covers *backfill* only, not the live poller |
| `_CRON_BY_TIMEFRAME` not updated (Task 3 skipped) → 1D never polled, series silently drifts stale, `gap-report` starts failing days later with no code change to blame | Medium (nothing raises; it is a dict lookup that simply never happens) | Medium | Task 4's test asserts the cron table equals `config.TIMEFRAMES` as a set, so the omission fails CI rather than production |
| **Pre-existing defect discovered, not fixed here**: `BackgroundScheduler()` (`poller.py:105`) uses the host's local timezone, so the `4h` job's `hour="0,4,8,12,16,20"` fires at 17:00/21:00/… UTC on this UTC+7 host — off the UTC 4h grid, storing a forming 4H bar for up to 4 h | High (host tz confirmed `WIB` = UTC+7) | Medium | Out of scope for Phase 1. The new `1d` entry pins `timezone="UTC"` so the *daily* tier is correct; report the 4H misalignment to the user as a Phase 4 input (the tier shift makes 4H the setup tier, raising the stakes) |
| A forming 1D bar sits in the DB for up to 24 h and taints the tail of a `run_backtest` (`engine._df` loads all rows unfiltered) | Medium | Medium | Task 6 GOTCHA gives the SQL check and the two mitigations (re-run after 00:00 UTC, or bound with `--end`). Root fix belongs to Phase 4's no-lookahead audit, not here |
| Genuine interior holes in Binance's 1D history (maintenance halts) make "zero gaps" unachievable | Low (daily bars are the most robust tier; 4H/1H/15m are already contiguous 2022-12-31→2026-07-23 in this DB) | Medium | Task 8 GOTCHA separates interior from trailing gaps and forbids masking a real hole by loosening `find_gaps`. `backfill` resumes from `last_ts` and will not fill behind a hole — rows after it must be deleted to rewind the cursor |
| Widening `config.TIMEFRAMES` breaks a `test_cli.py` gap-report test that seeds data per-timeframe | Low (all six loops traced by hand; every case passes because the 1D `expected_end` falls at or before `start_ms`) | Low | Task 2 VALIDATE runs `test_cli.py` explicitly before moving on, and instructs *not* to pre-emptively edit that file |
| Scope creep into the Phase 4 tier shift (re-pointing `REGIME_TIMEFRAME`, renaming `MAX_HOLD_BARS_15M`) because the 1D bars now exist | Medium | Medium | NOT Building names each constant that must stay untouched. Phase 4 also depends on Phase 2, which is still `in-progress` — shifting tiers now would confound the risk-model change with the tier change, the exact confounding the PRD's build order exists to prevent |

## Notes
- **This phase has no source-code dependency on Phases 2 or 3.** It touches `data/storage.py`, `config.py`, `data/poller.py`; Phase 2 touches `signals/*`, `backtest/engine.py`, `risk/*`, and `config.py`. The only shared file is `config.py`, and the two edits are in different, non-adjacent blocks (line 11 here, lines 67-100 there) — a trivial merge. Start Phase 1 first, as the PRD advises, because it is the only externally-blockable one.
- **The `15m` literal audit is Phase 4's, but the grep was run for this plan** and is recorded here so Phase 4 does not repeat it. Timeframe strings and intervals appear outside the two canonical maps at: `config.py:19,28,29` (tier names — indirected, correct), `config.py:96` (`MAX_HOLD_BARS_15M = 96`), `engine.py:126-148` (`df_4h`/`df_1h`/`df_15m`, `m15`, `close_4h`/`close_1h`/`close_15m` locals), `engine.py:201-248` (`ts_15m`, `window_15m`), `setup.py:259-286` (`df_1h`/`df_15m`, `interval_15m`, staleness check), `meanrev.py:256-266`, `walkforward.py:33` (`DAY_MS`, independent), `poller.py:5` (docstring). All read their timeframe from config; only the **variable names** and `MAX_HOLD_BARS_15M` are misleading. None blocks Phase 1.
- **`gap-report` needs no code change for the new grid** — it already iterates `config.SYMBOLS × config.TIMEFRAMES` (`cli.py:253-254`). The PRD's "extend `gap-report` to the new grid" is satisfied by Task 2 alone. Same for `backfill_all` (`backfill.py:151-152`).
- **Memory files to update after this phase**: `~/.claude/projects/…/memory/binance-fapi-unreachable.md` records the 2026-07-05 TLS block as current. It is stale — `fapi.binance.com/fapi/v1/ping` returned 200 on 2026-07-26 and a real ccxt 1D fetch succeeded. Amend it (or supersede it) so a future session does not skip the live backfill on outdated information.
