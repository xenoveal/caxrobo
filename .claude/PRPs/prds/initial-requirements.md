# Plan: L1 Crypto Signal-Only Trading Bot MVP — Phase 1: Data Infrastructure

**Source PRD**: `.claude/PRPs/prds/l1-crypto-signal-bot-mvp.prd.md`
**Selected Milestone**: Phase 1 — Data Infrastructure (no dependencies; unblocks Phase 2 Regime Classifier)
**Complexity**: Medium

## Summary
Build the exchange-connectivity and OHLCV storage foundation the rest of the bot depends on: a ccxt-based Binance USDT-M Futures client, a historical backfill job (≥2-3 years), a live polling loop, and local storage — all for BTCUSDT/ETHUSDT/SOLUSDT at 15m/1H/4H. Success is defined by the PRD as "backfilled + live data available and queryable for all three symbols at 15m/1H/4H with no gaps," so gap detection is a first-class deliverable, not an afterthought.

## Patterns to Mirror
No existing code in this repository (fresh project — the working tree contains only `.claude/` config, no source files). The sibling `investment-app` project is a Next.js/TypeScript app for a different (equities) system and does not share a language or domain with this Python/ccxt trading bot, so there is no in-repo or sibling-repo pattern to mirror. This phase establishes the conventions later phases will follow (module layout, storage schema, error-handling shape).

| Category | Source | Pattern |
|---|---|---|
| Naming | — | None found; establishing fresh (see Files to Change) |
| Error handling | — | None found; PRD mandates fail-silent-on-outage + deduplicated error logging (Phase 7 delivers the Discord sink — this phase just needs a clean, swappable log interface) |
| Logging | — | None found; use stdlib `logging` with a single named logger (`trading_bot`) so Phase 7 can attach a Discord handler without refactoring |
| Data access | — | None found; SQLite chosen for this phase (see rationale below) |
| Tests | — | None found; `pytest` assumed as the ecosystem-standard choice, no existing test config to mirror |

## Files to Change
| File | Action | Why |
|---|---|---|
| `pyproject.toml` | CREATE | Package metadata + dependencies (ccxt, pandas, pandas-ta, apscheduler, pytest) |
| `.gitignore` | CREATE | Exclude `data/*.db`, `.env`, `__pycache__` |
| `.env.example` | CREATE | Document expected env vars (Binance API key/secret if needed for higher rate limits; OHLCV itself is public) |
| `src/trading_bot/__init__.py` | CREATE | Package marker |
| `src/trading_bot/config.py` | CREATE | Locked constants: symbols (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`), timeframes (`15m`, `1h`, `4h`), backfill start date, DB path |
| `src/trading_bot/exchange/__init__.py` | CREATE | Package marker |
| `src/trading_bot/exchange/binance_client.py` | CREATE | Thin ccxt wrapper: `fetch_ohlcv_page(symbol, timeframe, since, limit)` with rate-limit-aware pagination |
| `src/trading_bot/data/__init__.py` | CREATE | Package marker |
| `src/trading_bot/data/storage.py` | CREATE | SQLite schema + idempotent upsert + gap-detection query |
| `src/trading_bot/data/backfill.py` | CREATE | Resumable historical backfill (walks forward from last stored candle, or from config start date if empty) |
| `src/trading_bot/data/poller.py` | CREATE | Live polling loop (APScheduler), fetches latest candles per symbol/timeframe, upserts, logs failures without raising |
| `src/trading_bot/cli.py` | CREATE | Entry points: `backfill`, `poll`, `gap-report` |
| `tests/test_storage.py` | CREATE | Upsert idempotency, gap-detection correctness |
| `tests/test_backfill.py` | CREATE | Backfill resumability against a mocked ccxt client (no live network calls in tests) |

## Tasks

### Task 1: Project scaffolding
- **Action**: Create `pyproject.toml` (deps: `ccxt`, `pandas`, `pandas-ta`, `apscheduler`, `pytest`, `python-dotenv`), `.gitignore`, `.env.example`, and the `src/trading_bot` package skeleton with `config.py` holding the locked symbol/timeframe/backfill-window constants from the PRD (Decisions Log: 4H/1H/15m tiers, BTCUSDT/ETHUSDT/SOLUSDT, USDT-M only).
- **Mirror**: N/A — first code in the repo.
- **Validate**: `pip install -e .` (or `uv sync`) succeeds; `python -c "from trading_bot import config"` imports cleanly.

### Task 2: Binance ccxt client wrapper
- **Action**: Implement `binance_client.py` using `ccxt.binanceusdm()`, exposing a single `fetch_ohlcv_page(symbol, timeframe, since_ms, limit=1000)` function that returns raw OHLCV rows and respects ccxt's built-in rate limiter (`enableRateLimit=True`). No auth required for public OHLCV endpoints, but read API key/secret from env if present (future-proofs for Phase 8 account/position read-only needs without redoing this module).
- **Mirror**: N/A.
- **Validate**: A manual smoke script (or a `@pytest.mark.network`-marked test, skipped by default) fetches one page for BTCUSDT/15m and asserts non-empty, ascending timestamps.

### Task 3: Storage layer (SQLite)
- **Action**: Implement `storage.py` with one table `ohlcv(symbol TEXT, timeframe TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL, PRIMARY KEY(symbol, timeframe, ts))`. Provide `upsert_candles(rows)` (idempotent — safe to re-run backfill or re-poll overlapping ranges) and `find_gaps(symbol, timeframe) -> list[(gap_start, gap_end)]` that walks stored timestamps looking for missing expected intervals. SQLite chosen over Parquet because upserts (needed for both backfill resumption and live polling overlap) are a native `INSERT OR REPLACE` here, versus manual dedup logic against append-only Parquet files.
- **Mirror**: N/A.
- **Validate**: `pytest tests/test_storage.py` — covers idempotent re-insert (no duplicate rows), and gap detection against a synthetic series with a deliberately missing candle.

### Task 4: Historical backfill
- **Action**: Implement `backfill.py`: for each of the 3 symbols × 3 timeframes, page backward/forward via `fetch_ohlcv_page` from the configured start date (≥2-3 years back per PRD) to now, upserting each page as it arrives so an interrupted run resumes from last-stored timestamp rather than restarting. Expose via `cli.py backfill [--symbol ...] [--timeframe ...]`.
- **Mirror**: N/A.
- **Validate**: `pytest tests/test_backfill.py` using a mocked client (fixed set of fake pages) — assert full range covered, resumption skips already-stored candles, and no duplicate rows result.

### Task 5: Live polling loop
- **Action**: Implement `poller.py`: an APScheduler job per timeframe (aligned to timeframe boundaries, e.g. 15m job at :00/:15/:30/:45) that fetches the latest candles for all 3 symbols and upserts. On any exchange/network error, log via the `trading_bot` logger and continue (fail-silent per PRD's error-handling risk mitigation) — do not raise or crash the loop. `cli.py poll` starts the scheduler and blocks.
- **Mirror**: N/A.
- **Validate**: Manual run (`python -m trading_bot.cli poll`) observed for a few minutes against a mocked/testnet client, confirming candles land in SQLite and a simulated fetch failure is logged once without crashing the loop.

### Task 6: Gap-report CLI + phase acceptance check
- **Action**: Implement `cli.py gap-report` that runs `find_gaps` across all 3×3 symbol/timeframe combinations and prints a clean table (empty = pass). This is the concrete artifact that proves the PRD's Phase 1 success signal ("no gaps").
- **Mirror**: N/A.
- **Validate**: After a full backfill run against real Binance data, `python -m trading_bot.cli gap-report` returns zero gaps for all 9 combinations.

## Validation
```bash
# from repo root
pip install -e .
pytest tests/ -v
python -m trading_bot.cli backfill --start 2023-01-01
python -m trading_bot.cli gap-report
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Binance rate limits during 2-3yr backfill across 9 symbol/timeframe series | Medium | Rely on ccxt's built-in `enableRateLimit`; page sequentially per series rather than fully parallel; resumable backfill means a throttling failure doesn't lose progress |
| Silent gaps from exchange downtime or missed poll cycles going unnoticed | Medium | `gap-report` CLI is the explicit PRD-mandated acceptance check; run it after every backfill and periodically once polling is live |
| SQLite write contention between the live poller and concurrent backtest reads (later phases) | Low-Medium | SQLite's WAL mode (`PRAGMA journal_mode=WAL`) allows concurrent readers during writes; revisit if Phase 5's backtester needs heavier concurrent access |
| Candle timestamp/boundary misalignment across 15m/1H/4H (exchange-reported open time vs. close time conventions) | Low | Store ccxt's raw timestamp convention as-is and document it in `config.py`; do not synthesize/resample timeframes from finer data in v1 — fetch each timeframe natively from the exchange |

## Acceptance
- [ ] All tasks complete
- [ ] Validation passes (`pytest`, backfill, gap-report all green with zero gaps across 9 symbol/timeframe combinations)
- [ ] Patterns mirrored, not reinvented (N/A this phase — first code; conventions set here should carry into Phase 2+)
