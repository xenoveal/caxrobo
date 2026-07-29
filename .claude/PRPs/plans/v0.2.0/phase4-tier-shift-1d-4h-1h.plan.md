# Plan: Tier Shift to 1D / 4H / 1H (PRD Phase 4)

## Summary
Move the whole signal stack up one timeframe tier — **1D regime / 4H setup / 1H trigger** (from 4H / 1H / 15m) — and eliminate every hardcoded 15-minute assumption that the current tier names leave behind. This is the change that puts the strategy on the right side of the cost frontier (mean Sharpe +0.791 at 60m vs −12.71 at 1m across 81 walk-forward configs) and simultaneously fixes the founding user problem: a human cannot reliably act on a 15m alert, but can act on an hourly one. Mechanically it is three config strings plus a rigorous, enumerated audit of every `15m` / `96` / `900_000` literal, plus a runtime invariant that makes a future mis-tiering loud instead of silent.

## User Story
As the bot's sole user, I want alerts triggered on 1H bar closes against 4H setups inside a 1D regime, so that (a) the round-trip cost is a small fraction of my stop distance instead of 58–68% of it, and (b) an alert that fires while I am asleep is still actionable when I wake up.

## Problem → Solution
**Current**: `REGIME_TIMEFRAME = "4h"`, `SIGNAL_PATTERN_TIMEFRAME = "1h"`, `SIGNAL_TRIGGER_TIMEFRAME = "15m"` (`config.py:19,28-29`). The tier names are indirected through config and read that way by the engine (`engine.py:126-128`), but the *identifiers, docstrings, bar-count constants and test fixtures* are all written as if 15m were permanent: `MAX_HOLD_BARS_15M`, `m15`, `df_15m`, `window_15m`, `interval_15m`, and `M15 = storage.TIMEFRAME_MS["15m"]` in three test files. Flipping the three config strings alone would "work" while leaving a codebase whose names lie, and — worse — leaving no mechanism to catch a future edit that reintroduces a 15m assumption. The PRD names this as a Medium-likelihood technical risk: *"Tier shift misses a hardcoded 15m assumption and produces a subtly lookahead-biased backtest."*

**Solution**: (1) flip the three tier strings; (2) rename `MAX_HOLD_BARS_15M` → `MAX_HOLD_BARS_TRIGGER` and keep the bar count at 96, which rescales the time-stop from 24h to 4 days *without introducing a new free parameter*; (3) rename every `*_15m` identifier to role-based names (`df_trig`, `window_trig`, `trigger_ms`, `df_setup`, `df_regime`); (4) add `_assert_interval()` in the engine that raises if a loaded series' observed bar spacing disagrees with its configured timeframe; (5) make the three signal/backtest test files derive their intervals from `config` + `storage.TIMEFRAME_MS` instead of hardcoding `"15m"`, so the fixtures follow the tier config forever; (6) audit and explicitly dispose of every remaining lookback constant whose *duration* semantics change under coarser bars.

## Metadata
- **Complexity**: **Medium-High** — the code diff is small and mostly mechanical, but the audit surface is wide (12 files) and a missed hit produces a silently wrong backtest rather than a crash. The risk is in completeness, not in difficulty.
- **Source PRD**: `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`
- **PRD Phase**: Phase 4 — Tier shift to 1D/4H/1H
- **Depends on**: Phase 1 (1D bars exist in `storage.TIMEFRAME_MS` + backfilled), Phase 2 (`.claude/PRPs/plans/v0.2.0/phase2-honest-cost-and-risk-model.plan.md`, ATR risk model landed)
- **Estimated Files**: 12 (1 new, 11 modified)

---

## UX Design

N/A — internal change. Observable surfaces are the existing `signal` / `backtest` / `walkforward` CLI commands, which will report fewer, larger, longer-held trades. No new output columns.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/trading_bot/config.py` | 10-32, 57-65, 87-100 | Every timeframe string, bar-count, lookback and hold-limit constant. Lines 19/28-29 are the tier strings; line 96 is `MAX_HOLD_BARS_15M`; lines 22/24/31/32/58/89/91 are bar-count constants whose *duration* semantics change |
| P0 | `src/trading_bot/backtest/engine.py` | 1-28, 91-148, 176-260 | Whole file. Docstring (8-28) states the no-lookahead guarantees in 15m/1H/4H language; 126-148 loads the three series and derives `m15`/`ts_15m`/`close_15m`; 236 slices `window_15m`; 221 applies the hold limit in trigger bars |
| P0 | `src/trading_bot/signals/setup.py` | 245-293 | `scan_breakout_signals` — `trigger_bars` arithmetic (267-268), `interval_15m` (270-272), the "15m data is … behind now_ms" warning (274) |
| P0 | `src/trading_bot/regime/classifier.py` | 35-51, 91-125, 128-161 | `REGIME_MIN_BARS` gate (94, 123) and `ATR_PERCENTILE_WINDOW` (103, 185). Determines whether 1D warmup is satisfiable against real history. Do NOT change classifier logic (PRD: the one measured-healthy component) — only its docstring's "4H" wording |
| P1 | `src/trading_bot/signals/meanrev.py` | 244-273 | `scan_fade_signals` — `df_15m` (257-262), `df_15m.tail(VOLUME_LOOKBACK + 2)`, `check_breakout` called *without* `interval_ms` (266) |
| P1 | `src/trading_bot/data/storage.py` | 18 | `TIMEFRAME_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}` — **`"1d"` is missing**; Phase 1 owns adding it. Phase 4 must hard-depend on it |
| P1 | `src/trading_bot/backtest/walkforward.py` | 33, 93-135, 178-207 | Fold windows are in **days** (`DAY_MS = 86_400_000`, `WF_*_DAYS * DAY_MS`), so they are tier-invariant. Coarser bars mean fewer trades per fold — a Phase 7 concern, not this phase's |
| P1 | `tests/test_backtest.py` | 16-20, 56-88, 90-147 | `H4`/`H1`/`M15` constants and `seed_scenario` seed literal `"4h"`/`"1h"`/`"15m"`. `TestRunBacktest` is the de-facto no-lookahead suite: entry-at-trigger-close, conservative same-bar stop, next-bar-only exits |
| P1 | `tests/test_signals.py` | 17-28, 316-393, 499-551 | `H1`/`M15` module constants, `make_df`, `breakout_df`, `seed_candles`, and `test_trending_flag_breakout_produces_signal` which seeds literal `"1h"`/`"15m"` |
| P2 | `tests/test_meanrev.py` | 21-28, 200-232 | Same shape as `test_signals.py`; `test_ranging_stretch_recross_produces_signal` seeds `"1h"`/`"15m"` |
| P2 | `tests/test_classifier.py` | 16-17 | `TF = "4h"` / `INTERVAL = storage.TIMEFRAME_MS[TF]`. Everything else is expressed in `config.REGIME_MIN_BARS`, so this one line is the whole change |
| P2 | `.claude/PRPs/plans/v0.2.0/phase2-honest-cost-and-risk-model.plan.md` | 155-163, 168, 525-575 | Phase 2's post-state: new `risk/` package, `build_signal(…, atr_value, …)`, `atr_1h`/`atr_close_1h` in the engine, funding term in `close_out`. Note line 168 explicitly defers the `k` re-derivation to this phase |

## External Documentation

None needed. Every value used here is either already in the repo or a derived quantity recorded in `.claude/PRPs/reports/market-research-capability-benchmark.md`. The cost-frontier evidence (`c` = 7.1 / 5.2 / 3.7% all-taker at 4H setup bars vs 14.9% at 1H) is PRD Open Question #2.

---

## Ground Truth: What Is Actually in the Database

Measured read-only via `sqlite3 data/ohlcv.db` on 2026-07-26:

| Symbol | 15m | 1h | 4h | **1d** | Span |
|---|---|---|---|---|---|
| BTCUSDT | 124,792 | 31,198 | 7,799 | **0 — absent** | 2022-12-31 17:00 → 2026-07-23 14:45 UTC |
| ETHUSDT | 124,792 | 31,198 | 7,799 | **0 — absent** | same |
| SOLUSDT | 124,792 | 31,198 | 7,799 | **0 — absent** | same |

**Consequences that must be honored:**
- **1D bars do not exist yet.** `storage.TIMEFRAME_MS` has no `"1d"` key, so `_df(conn, symbol, "1d")` raises `KeyError` today. This phase is *hard-blocked* on Phase 1 (which is itself blocked on the `fapi.binance.com` TLS issue, see `~/.claude/.../memory/binance-fapi-unreachable.md`).
- **Expected 1D bar count ≈ 1,300** (2023-01-01 → 2026-07-22 inclusive). New setup tier (4H) keeps its 7,799 bars; new trigger tier (1H) keeps 31,198.
- **1D regime warmup is satisfiable but expensive.** `REGIME_MIN_BARS = 2*14 - 1 + 180 = 207` (`config.py:24`). At 1D that is **207 calendar days**: bar index 206 ≈ **2023-07-26**, so the first non-`uncertain` 1D label lands ≈ **2023-07-27**, leaving ≈ **1,093 usable 1D bars** (84% of history). Satisfiable — but any backtest or walk-forward fold whose window starts before 2023-07-27 will see `uncertain` for its whole span and produce **zero trades**. With `WF_TRAIN_DAYS = 180`, fold 0 starting at `BACKFILL_START = "2023-01-01"` is entirely inside warmup. Validation must pass `--start 2023-07-27` (or later). This is not a bug to fix here; it is a fact to record and to state in the Phase 7 handoff.
- Warmup is consumed at the *front of DB history*, not per fold: `engine._df` (`engine.py:83-88`) loads the entire stored series with no time filter and `classify_series` runs over all of it; `start_ms` only gates *entries*. So the 207-bar cost is paid once.
- Trigger-loop iteration count drops 124,792 → 31,198 and the `cand_cache` recomputes candidates once per 4H setup bar instead of once per 1H bar — the backtest gets ~4× faster, not slower.

---

## The Grep Audit (exhaustive, enumerated, with dispositions)

Run and reproduced verbatim from:
```bash
grep -rn '15m' src tests --include='*.py'
grep -rn '15M\|_15m\|m15' src tests --include='*.py'
grep -rn '900_000\|900000' src tests --include='*.py'
grep -rn '\b96\b' src tests --include='*.py'
grep -rn '"4h"\|4H\|_4h\|"1h"\|1H\b' src tests --include='*.py'
```

### A. `900_000` (the 15m interval in milliseconds) — 5 hits, **zero to change**

| file:line | Hit | Disposition |
|---|---|---|
| `data/storage.py:18` | `TIMEFRAME_MS = {"15m": 900_000, …}` | **LEAVE.** 15m stays a *stored* timeframe (poller keeps ingesting it); it just stops being a *tier*. Phase 1 adds `"1d": 86_400_000` here |
| `tests/test_poller.py:132,134,135,137` | comments + `expected_now_ms - (3 * 900_000)` | **LEAVE.** Poller-layer test of `since_ms` arithmetic for the 15m series; tier-agnostic |

No `900_000` literal exists anywhere in the signal, regime, or backtest path. Confirmed.

### B. `96` — 6 hits, **1 to change**

| file:line | Hit | Disposition |
|---|---|---|
| `config.py:96` | `MAX_HOLD_BARS_15M = 96` | **RENAME + RE-COMMENT** → `MAX_HOLD_BARS_TRIGGER = 96`. Value stays 96; at 1H bars that is 4 days instead of 24h. See Task 1 GOTCHA on why 96 and not 24 or 168 |
| `tests/test_signals.py:98,103,108,114,172` | `path_df([… 96, 94])` price anchors | **LEAVE.** Coincidental price values in pivot/pattern fixtures. Not bar counts |

### C. `15m` / `15M` / `_15m` / `m15` identifiers and docstrings — 62 hits, grouped

**C1. Tier configuration — CHANGE (3 lines)**

| file:line | Hit | Disposition |
|---|---|---|
| `config.py:29` | `SIGNAL_TRIGGER_TIMEFRAME = "15m"` | **CHANGE** → `"1h"` |
| `config.py:28` | `SIGNAL_PATTERN_TIMEFRAME = "1h"` | **CHANGE** → `"4h"` |
| `config.py:19` | `REGIME_TIMEFRAME = "4h"` | **CHANGE** → `"1d"` |

**C2. Config comments that name 15m — REWORD (5 lines)**

`config.py:58` (`VOLUME_LOOKBACK = 20  # 15m bars…`), `config.py:61` (trigger-lookback comment), `config.py:74` (stop-buffer noise comment — *deleted entirely by Phase 2*), `config.py:96` (hold-limit comment), `config.py:11` (`TIMEFRAMES` tuple — Phase 1 appends `"1d"`; do not remove `"15m"`).
→ **REWORD** to say "trigger-timeframe bars" rather than "15m bars". `config.py:74` requires no action if Phase 2 has landed (it deletes `BREAKOUT_STOP_BUFFER_PCT` and its comment block, `config.py:72-85`); if Phase 2 has not landed, do not touch it — Phase 2 owns that block.

**C3. `engine.py` identifiers — RENAME (24 hits)**

| file:line | Hit | Disposition |
|---|---|---|
| `engine.py:8-28` | module docstring: "15m bar", "1H bars", "4H bar", "MAX_HOLD_BARS_15M" | **REWRITE** in tier-role language (regime / setup / trigger) |
| `engine.py:108,110,114` | `run_backtest` docstring: "15m bars", "Time-stop in 15m bars (default config.MAX_HOLD_BARS_15M)" | **REWRITE** |
| `engine.py:123` | `max_hold = config.MAX_HOLD_BARS_15M if …` | **RENAME** const → `MAX_HOLD_BARS_TRIGGER` |
| `engine.py:126,129,133,137,151` | `df_4h`, `close_4h` | **RENAME** → `df_regime`, `close_regime` |
| `engine.py:127,129,138,161` | `df_1h`, `close_1h` | **RENAME** → `df_setup`, `close_setup` |
| `engine.py:128,129,140,146,147,148,236` | `df_15m` | **RENAME** → `df_trig` |
| `engine.py:139,242,246` | `m15` | **RENAME** → `trigger_ms` |
| `engine.py:140,141,191,201` | `ts_15m` | **RENAME** → `ts_trig` |
| `engine.py:141,143,144,202` | `close_15m` | **RENAME** → `close_trig` |
| `engine.py:236` | `window_15m = df_15m.iloc[max(0, j - (VOLUME_LOOKBACK + 1)) : j + 1]` | **RENAME** → `window_trig`; the arithmetic is in *trigger bars* and is already tier-correct — do **not** rescale it |
| `engine.py:154` | comment "per 1H bar" | **REWORD** → "per setup bar" |
| *(post-Phase-2)* `atr_1h`, `atr_close_1h` | Phase 2 Task 7 introduces these | **RENAME** → `atr_setup`, `atr_setup_vals`. Phase 2's GOTCHA about keeping them index-aligned with `close_setup` carries forward unchanged |
| *(post-Phase-2)* `hold_days = (… - s.ts) / 86_400_000.0` | funding term in `close_out` | **LEAVE.** Wall-clock days; tier-invariant and now materially larger, which is the point |

**C4. `setup.py` identifiers + docstrings — RENAME/REWORD (12 hits)**

| file:line | Hit | Disposition |
|---|---|---|
| `setup.py:11,45,104` | docstrings: "latest closed 15m bar", "Epoch-ms of the 15m trigger bar", "The 15m breakout event" | **REWORD** → "trigger bar" |
| `setup.py:260,261,268` | `df_15m` | **RENAME** → `df_trig` |
| `setup.py:270,271,272,286` | `interval_15m` | **RENAME** → `trigger_interval` |
| `setup.py:274` | `"%s 15m data is %d ms behind now_ms; …"` | **PARAMETERIZE** → `"%s %s data is %d ms behind now_ms; …"` with `config.SIGNAL_TRIGGER_TIMEFRAME` as the second arg |
| `setup.py:264` | `df_1h = df_1h.tail(config.PATTERN_LOOKBACK_BARS)` | **RENAME** local → `df_setup`; keep `PATTERN_LOOKBACK_BARS = 180` (now 180 × 4H = 30 days; see §D) |
| `setup.py:267-268` | `trigger_bars = VOLUME_LOOKBACK + 1 + max(1, BREAKOUT_TRIGGER_LOOKBACK_BARS)` | **LEAVE the arithmetic.** It is already expressed purely in trigger bars. This is the "trigger-bar arithmetic in setup.py:270-279" the PRD flags — the finding is that it is *already* tier-parameterized and needs only renaming, not rescaling |
| `setup.py:312` | docstring "current 4H regime" | **REWORD** → "current regime-timeframe regime" |

**C5. `meanrev.py` identifiers + docstrings — RENAME/REWORD (8 hits)**

| file:line | Hit | Disposition |
|---|---|---|
| `meanrev.py:12,13,41,45,191` | docstrings: "Trigger on 15m", "closed 15m bar", "15m re-cross trigger" | **REWORD** |
| `meanrev.py:257,258,262` | `df_15m` | **RENAME** → `df_trig` |
| `meanrev.py:256,261` | `df_1h` | **RENAME** → `df_setup` |
| `meanrev.py:266` | `check_breakout(df_15m, _to_trigger_candidate(candidate))` — **no `interval_ms=`** | **FIX (real defect, in scope).** The breakout path passes `interval_ms` (`setup.py:286`) so a gapped crossing pair is skipped; the fade path does not, so a stale prior close can read as a fresh re-cross. Add `interval_ms=storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]`. This gets *more* dangerous at 1H bars (a gap is now hours, not minutes) and it is a no-lookahead invariant, squarely in this phase's remit. Requires adding `from trading_bot.data import storage` to `meanrev.py` imports |
| `meanrev.py:262` | `df_15m.tail(config.VOLUME_LOOKBACK + 2)` | **LEAVE the arithmetic** (trigger bars), rename the local |

**C6. `breakout.py` / `scan.py` / `classifier.py` — docstrings only (10 hits)**

`breakout.py:2,4,13,43,70,73` ("15m breakout trigger", "closed 15m bar", "the 15m boundary"), `scan.py:5` ("current 4H regime"), `classifier.py:135` ("latest closed 4H candle"), `patterns.py:28,65,349` ("the 15m trigger's fresh-crossing rule"), `patterns.py:2,89` ("1H pivot structure").
→ **REWORD** to tier-role language. **Zero logic changes.** `check_breakout` is already fully generic over `interval_ms` / `lookback_bars`.

**C7. Data-layer and exchange hits — LEAVE (16 hits)**

`data/poller.py:5,35,81,94` (cron map + docstrings — Phase 1 adds a `"1d"` entry: `{"hour": "0", "minute": "0", "second": 10}`), `data/backfill.py:59`, `data/storage.py:76,213`, `exchange/binance_client.py:85`.
→ **LEAVE.** 15m remains an ingested timeframe. Phase 1 owns the `"1d"` additions here.

**C8. Test hits — see §E for dispositions (37 hits across 8 test files)**

### D. Bar-count / lookback constants whose *duration* semantics change

Every one of these is already indirected through `config`; the question is whether the number should change. Enumerated with an explicit decision so no reader has to guess:

| Constant (`config.py`) | Value | Old duration | New duration | Disposition |
|---|---|---|---|---|
| `ADX_PERIOD = 14` | 14 | 56h @4H | 14 days @1D | **LEAVE.** ADX(14) on daily bars is the canonical Wilder parameterization. Warmup `2*14-1 = 27` bars, trivially satisfied |
| `ATR_PERCENTILE_WINDOW = 180` | 180 | 30 days @4H | **180 days @1D** | **LEAVE (deliberate).** Rescaling to 30 would preserve the calendar window but (a) re-tunes the one component the PRD says to keep unchanged, (b) consumes a DoF, (c) a 30-sample percentile has 1/30 resolution, making the 0.90 gate effectively "top 3 bars". Document the semantic change in the config comment; do not sweep it |
| `REGIME_MIN_BARS = 2*ADX_PERIOD-1 + ATR_PERCENTILE_WINDOW` = 207 | 207 | 34.5 days @4H | **207 days @1D** | **LEAVE the formula.** Consequence (usable history starts ≈2023-07-27) recorded in Ground Truth above and surfaced in Task 8's validation |
| `PATTERN_LOOKBACK_BARS = 180` | 180 | 7.5 days @1H | 30 days @4H | **LEAVE.** 30 days of setup history is generous, and Phase 5's Donchian-55 needs only 55 setup bars — comfortably inside 180 |
| `PATTERN_MAX_AGE_BARS = 12` | 12 | 12h @1H | 2 days @4H | **LEAVE.** Pattern geometry (`patterns.py`, `pivots.py`) leaves the active path in Phase 5; retuning it now would be wasted work and a consumed DoF |
| `PIVOT_SPAN = 3` | 3 | — | — | **LEAVE.** Positional, unit-free |
| `FLAG_*`, `TRIANGLE_*`, `HS_*` (`config.py:34-55`) | — | 1H-bar counts | 4H-bar counts | **LEAVE, all of them.** Retired in Phase 5. Explicitly out of scope |
| `VOLUME_LOOKBACK = 20` | 20 | 5h @15m | 20h @1H | **LEAVE.** A ~1-day rolling volume mean on the trigger timeframe is if anything more meaningful than a 5-hour one. Volume is a graded confidence input, never a block (`breakout.py:20-23`) |
| `VOLUME_HIGH_RATIO = 1.5` | 1.5 | — | — | **LEAVE.** Dimensionless |
| `BREAKOUT_TRIGGER_LOOKBACK_BARS = 1` | 1 | — | — | **LEAVE.** Tier-invariant; the backtest hardcodes `lookback_bars=1` anyway (`engine.py:242,246`) |
| `BB_PERIOD = 20` | 20 | 20h @1H | 80h @4H | **LEAVE.** BB(20) is canonical on any timeframe |
| `FADE_STRETCH_MAX_AGE_BARS = 6` | 6 | 6h @1H | 24h @4H | **LEAVE.** Phase 6 re-qualifies the fade sleeve and owns any retune |
| `MAX_HOLD_BARS_15M = 96` | 96 | 24h @15m | **4 days @1H** | **RENAME → `MAX_HOLD_BARS_TRIGGER`, value unchanged.** See Task 1 |
| `STALENESS_INTERVALS = 2` | 2 | — | — | **LEAVE.** Expressed in intervals |
| `WF_TRAIN_DAYS / WF_TEST_DAYS / WF_OOS_DAYS` | 180/60/90 | days | days | **LEAVE.** `walkforward.py:126-128` multiplies by `DAY_MS`; tier-invariant. Trades-per-fold will drop sharply — Phase 7's problem, flagged in the Notes |
| `FEE_PCT / SLIPPAGE_PCT / FUNDING_PCT_PER_DAY` | — | — | — | **LEAVE.** Frozen by PRD decree |

### E. Test-file dispositions

| file:line | Hit | Disposition |
|---|---|---|
| `tests/test_backtest.py:17-19` | `H4 = TIMEFRAME_MS["4h"]`, `H1 = …["1h"]`, `M15 = …["15m"]` | **REPLACE** with tier-derived: `REGIME_TF`/`SETUP_TF`/`TRIGGER_TF` from config, `D_REG`/`D_SET`/`D_TRIG` intervals |
| `tests/test_backtest.py:27` | `exit_ts=START + M15` in `make_trade` | **RENAME** → `D_TRIG` |
| `tests/test_backtest.py:56` | `def seed(conn, timeframe, rows, start=START, interval=H1)` | **CHANGE default** → `interval=D_SET` |
| `tests/test_backtest.py:62-68` | `flag_rows()` docstring "1H bull flag" | **REWORD** → "setup-timeframe bull flag" |
| `tests/test_backtest.py:71-80` | `seed_scenario` seeds literal `"4h"`, `"1h"`, `"15m"` | **PARAMETERIZE** off config constants |
| `tests/test_backtest.py:73` | `seed(conn, "4h", [[100,111,99,105,10.0]]*12, interval=H4)` — only 12 regime bars | **KEEP 12 bars.** `patch_trending` monkeypatches `classify_series` (`test_backtest.py:83-87`), so `REGIME_MIN_BARS = 207` never bites in these fixtures. Just retarget the timeframe/interval |
| `tests/test_backtest.py:90-147` | `TestRunBacktest` — the de-facto no-lookahead suite | **MUST STAY GREEN with only fixture-interval edits.** If any assertion needs a *logic*-shaped change, stop: that is a real regression, not a rescale |
| `tests/test_backtest.py:101,116` | `109.5 * (1 - config.BREAKOUT_STOP_BUFFER_PCT)` | **LEAVE — Phase 2 owns these.** Phase 2 rewrites them for the ATR stop. Do not touch; if Phase 2 has landed they already read differently |
| `tests/test_signals.py:17-18` | `H1`/`M15` module constants | **REPLACE** with `D_SET`/`D_TRIG` derived from config |
| `tests/test_signals.py:22,327-333,372` | `make_df(…, interval=H1)`, `breakout_df`, docstrings "15m df" | **RETARGET** interval defaults + reword |
| `tests/test_signals.py:472,477,485` | `_contiguous_tail(df, H1, SYMBOL, "1h")` | **RETARGET** to `D_SET` / `SETUP_TF`. Logic unchanged |
| `tests/test_signals.py:530,532-539` | `seed_candles(conn, SYMBOL, "1h"/"15m", …)`, `start=last_1h_ts - 19 * M15` | **PARAMETERIZE** timeframe strings and the `-19 * M15` offset → `-19 * D_TRIG` |
| `tests/test_meanrev.py:21-22,26,207,210-218` | mirror of the above | **PARAMETERIZE** identically |
| `tests/test_meanrev.py:230-231` | `risk_pct <= config.MAX_RISK_PCT`, `reward_pct >= MIN_REWARD_PCT` | **LEAVE — Phase 2 owns these** (its Task 6 replaces them with `rr >= RR_FLOOR`) |
| `tests/test_classifier.py:16-17` | `TF = "4h"`; `INTERVAL = storage.TIMEFRAME_MS[TF]` | **CHANGE** → `TF = config.REGIME_TIMEFRAME`. Every other reference in the file is already symbolic (`config.REGIME_MIN_BARS` at lines 48,58,64,66,83,90,105,115,126,144,159,164,185,195,223,265,291,309) so the file needs exactly this one edit |
| `tests/test_wilder.py:10` | `TF = "4h"` | **LEAVE.** Pure indicator math; tier-agnostic |
| `tests/test_storage.py:10` | `TF = "15m"` | **LEAVE.** Storage-layer |
| `tests/test_backfill.py` (14 hits), `tests/test_binance_client.py` (6 hits) | `"15m"` fixtures | **LEAVE.** Data/exchange layer |
| `tests/test_poller.py:348,358,364` | asserts exactly 3 scheduler jobs, `{"15m","1h","4h"}` | **LEAVE — Phase 1 owns this.** Adding `"1d"` to `_CRON_BY_TIMEFRAME` breaks it. Phase 4 must merely *confirm* the suite is green after Phase 1 lands, and escalate to Phase 1 if not |
| `tests/test_cli.py:76,87,110,117,124,136,192,235-239,266-270,357` | `TIMEFRAME_MS["15m"]`/`["4h"]` in gap-report fixtures that loop `config.TIMEFRAMES` | **LEAVE — Phase 1 owns this.** Once `"1d"` is in `config.TIMEFRAMES`, these fixtures must seed 1d too or gap-report flags it. Confirm green; escalate to Phase 1 if not |

---

## Patterns to Mirror

### CONFIG_CONSTANT_STYLE
```python
# SOURCE: config.py:18-24 — SCREAMING_SNAKE constants under a comment banner
# naming the phase, with units stated inline; derived constants computed from
# their inputs rather than hardcoded.
REGIME_TIMEFRAME = "4h"
ADX_PERIOD = 14
ATR_PERCENTILE_WINDOW = 180
REGIME_MIN_BARS = 2 * ADX_PERIOD - 1 + ATR_PERCENTILE_WINDOW
```

### TIMEFRAME_INDIRECTION (the pattern the whole phase leans on)
```python
# SOURCE: engine.py:126-139 — every series is loaded by CONFIG NAME and its
# interval read out of storage.TIMEFRAME_MS. Never a literal string, never a
# literal millisecond count.
df_4h = _df(conn, symbol, config.REGIME_TIMEFRAME)
df_1h = _df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME)
df_15m = _df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME)
...
close_4h = df_4h.index.to_numpy() + storage.TIMEFRAME_MS[config.REGIME_TIMEFRAME]
m15 = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
```

### CLOSED_BAR_RULE (the no-lookahead primitive — must survive untouched)
```python
# SOURCE: classifier.py:158-161 (and setup.py:224-225, identical)
# A bar is closed when ts + interval <= now_ms; load_candles' end_ms is
# inclusive, so end_ms = now_ms - interval excludes the forming bar.
interval = storage.TIMEFRAME_MS[timeframe]
rows = storage.load_candles(conn, symbol, timeframe, end_ms=now_ms - interval)
```

### SEARCHSORTED_ALIGNMENT (the other no-lookahead primitive)
```python
# SOURCE: engine.py:150-152, 228 — side="right" then -1 gives the last bar
# CLOSED AT OR BEFORE t. Correct at every tier: 1D and 4H closes fall exactly
# on 1H boundaries, so a same-instant close is legitimately included.
def regime_at(t: int) -> str:
    k = int(np.searchsorted(close_4h, t, side="right")) - 1
    return str(labels.iloc[k]) if k >= 0 else "uncertain"
...
h_idx = int(np.searchsorted(close_1h, bc, side="right")) - 1
```

### GUARD_STYLE (raise on data/config contradiction, return None on filter failure)
```python
# SOURCE: setup.py:206-213 — a data problem is logged loudly with full context;
# SOURCE: walkforward.py:133-136 — a config/data contradiction RAISES ValueError
raise ValueError(
    "span too short: need at least one train+test fold before the OOS holdout"
)
```

### TEST_FIXTURE_STYLE
```python
# SOURCE: tests/test_backtest.py:56-59 — `seed()` builds rows from an interval
# and upserts; every fixture derives timestamps from a module-level interval
# constant rather than inlining milliseconds.
def seed(conn, timeframe, rows, start=START, interval=H1):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    storage.upsert_candles(conn, SYMBOL, timeframe, data)
    return data
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/config.py` | UPDATE | Three tier strings; `MAX_HOLD_BARS_15M` → `MAX_HOLD_BARS_TRIGGER`; reword every comment naming 15m; document `ATR_PERCENTILE_WINDOW`'s new 180-day semantics |
| `src/trading_bot/backtest/engine.py` | UPDATE | Rename all `*_4h`/`*_1h`/`*_15m`/`m15` locals to tier roles; rewrite the no-lookahead docstring; add `_assert_interval()` and call it for all three series |
| `src/trading_bot/signals/setup.py` | UPDATE | Rename `df_15m`/`interval_15m`/`df_1h`; parameterize the staleness warning; reword docstrings |
| `src/trading_bot/signals/meanrev.py` | UPDATE | Same renames **plus** the real fix: pass `interval_ms` to `check_breakout` (C5) |
| `src/trading_bot/signals/breakout.py` | UPDATE | Docstrings only (module header + `BreakoutEvent.ts` + `check_breakout` args) |
| `src/trading_bot/signals/scan.py` | UPDATE | Docstring: "current 4H regime" → tier-role wording |
| `src/trading_bot/signals/patterns.py` | UPDATE | Docstrings only (lines 2, 28, 65, 89, 349) |
| `src/trading_bot/regime/classifier.py` | UPDATE | Docstring only (line 135, "latest closed 4H candle"). **No logic change** |
| `tests/test_backtest.py` | UPDATE | Tier-derive `H4`/`H1`/`M15`; parameterize `seed_scenario`'s timeframe literals |
| `tests/test_signals.py` | UPDATE | Tier-derive interval constants; parameterize `seed_candles` timeframes |
| `tests/test_meanrev.py` | UPDATE | Mirror of `test_signals.py` |
| `tests/test_classifier.py` | UPDATE | One line: `TF = config.REGIME_TIMEFRAME` |
| `tests/test_tiers.py` | CREATE | The phase's own regression net: tier ordering, `TIMEFRAME_MS` coverage, `_assert_interval` behavior, no-15m-literals grep-as-a-test |

## NOT Building

- **1D backfill execution** (Phase 1) — this plan *depends on* `storage.TIMEFRAME_MS["1d"]` existing and the data being present; it does not add either. Also Phase 1's territory: `config.TIMEFRAMES` gaining `"1d"`, `poller._CRON_BY_TIMEFRAME` gaining `"1d"`, and the resulting `tests/test_poller.py` / `tests/test_cli.py` fixture updates.
- **Donchian channel engine** (Phase 5) — `indicators/donchian.py`, `signals/donchian.py`, and retiring `patterns.py`/`pivots.py` from dispatch. This phase keeps the *existing* breakout + fade methods running, on new tiers, so the tier change is measurable in isolation.
- **Retuning any pattern-geometry constant** (`FLAG_*`, `TRIANGLE_*`, `HS_*`, `PATTERN_MAX_AGE_BARS`) — those detectors leave the active path in Phase 5.
- **Fade re-qualification** (Phase 6) — no keep/drop decision, no retuning of `BB_PERIOD` / `FADE_STRETCH_MAX_AGE_BARS`. The one fade change here (`interval_ms`) is a no-lookahead correctness fix, not a re-qualification.
- **Walk-forward repair and THE GATE** (Phase 7) — no pooling, no `WF_MIN_TRADES` change, no DSR, no grid change, no `plateau_ratio` replacement, and **no running of the one-shot OOS holdout**. Coarser bars will cut trades per fold; recording that number is this phase's only obligation.
- **Sharpe / Sortino / equity-curve metrics** (Phase 3) — `metrics.py` untouched.
- **Volatility-target sizing** (Phase 8), **Discord alerting + trial log** (Phase 9), **Option B / C sleeves** (Phase 10).
- **Re-planning Phase 2's work** — the ATR stop, `RR_FLOOR`, funding term, `risk/` package, and retirement of `MAX_RISK_PCT`/`MIN_REWARD_PCT`/`BREAKOUT_STOP_BUFFER_PCT`/`BREAKOUT_MAX_ENTRY_EXTENSION_PCT` all belong to `.claude/PRPs/plans/v0.2.0/phase2-honest-cost-and-risk-model.plan.md`. Where this plan touches the same lines it says so and defers.
- **1-minute bars for exit resolution** (PRD Open Question #4, `Could` priority) — not in v1 scope.
- **Sweeping `ATR_STOP_MULTIPLE`** — see Task 5: `k` is *re-derived once* against 4H ATR and re-frozen. Deriving is not sweeping.

---

## Step-by-Step Tasks

### Task 1: Flip the tier strings and rename the hold limit
- **ACTION**: Edit `src/trading_bot/config.py`.
- **IMPLEMENT**:
  ```python
  # Regime classifier thresholds. Phase 4: the regime tier is 1D.
  # NOTE on ATR_PERCENTILE_WINDOW: 180 bars was ~30 days at the old 4H tier and
  # is ~180 days at 1D. The window is deliberately NOT rescaled — the PRD keeps
  # the regime classifier unchanged (it is the one measured-healthy component),
  # a 30-sample percentile has too coarse a resolution for a 0.90 gate, and
  # retuning it here would consume a degree of freedom for no measured benefit.
  REGIME_TIMEFRAME = "1d"
  ADX_PERIOD = 14
  ADX_TREND_THRESHOLD = 25.0
  ATR_PERCENTILE_WINDOW = 180
  ATR_EXTREME_PERCENTILE = 0.90
  # 207 bars. At the 1D tier this is 207 CALENDAR DAYS of warmup: with history
  # from 2023-01-01, the first non-"uncertain" label lands ~2023-07-27. Any
  # backtest or walk-forward window starting before then produces zero trades.
  REGIME_MIN_BARS = 2 * ADX_PERIOD - 1 + ATR_PERCENTILE_WINDOW

  # Phase 4 tiers: 1D regime / 4H setup / 1H trigger.
  SIGNAL_PATTERN_TIMEFRAME = "4h"  # setups detected on setup-timeframe bars
  SIGNAL_TRIGGER_TIMEFRAME = "1h"  # trigger confirmed on trigger-timeframe bars
  ```
  and, in the Phase 5 block:
  ```python
  # Time-stop, in SIGNAL_TRIGGER_TIMEFRAME bars: 96 * 1h = 4 days.
  # Renamed from MAX_HOLD_BARS_15M in Phase 4. The BAR COUNT is unchanged, so
  # the tier shift rescales the holding limit 24h -> 4 days for free without
  # introducing a new free parameter. Changing this number is a consumed degree
  # of freedom and must be logged as one (PRD: trial-log discipline).
  MAX_HOLD_BARS_TRIGGER = 96
  ```
  Also reword the comments at lines 58 and 61 from "15m bars" to "trigger-timeframe bars".
- **MIRROR**: `CONFIG_CONSTANT_STYLE` above — phase-named comment banners, units inline, derived constants computed.
- **IMPORTS**: none.
- **GOTCHA**: Resist both the "24 bars keeps it at 24h" and the "168 bars gives trend-following room" temptations. **Keeping 96** is the *only* option that introduces zero new information: the number is inherited, and the duration change is a mechanical consequence of the tier shift the PRD already decided on. Picking 24 or 168 is a parameter choice fitted on nothing. Also: do NOT touch `config.py:67-85` (the `MAX_RISK_PCT` / `MIN_REWARD_PCT` / `BREAKOUT_*` block) — Phase 2 owns it, and `config.py:74`'s "15m noise" comment disappears with it.
- **VALIDATE**:
  ```bash
  .venv/bin/python -c "from trading_bot import config as c; print(c.REGIME_TIMEFRAME, c.SIGNAL_PATTERN_TIMEFRAME, c.SIGNAL_TRIGGER_TIMEFRAME, c.MAX_HOLD_BARS_TRIGGER)"
  # EXPECT: 1d 4h 1h 96
  grep -rn "MAX_HOLD_BARS_15M" src tests   # EXPECT: no matches after Task 2
  ```

### Task 2: Rename engine locals to tier roles and rewrite its no-lookahead docstring
- **ACTION**: Edit `src/trading_bot/backtest/engine.py` — docstring 1-28, `run_backtest` 91-148, `close_out` 176-198, main loop 200-260.
- **IMPLEMENT**: Apply the C3 rename table mechanically. The head of `run_backtest` becomes:
  ```python
  max_hold = config.MAX_HOLD_BARS_TRIGGER if max_hold_bars is None else max_hold_bars
  cost = 2 * (fee + slip)

  df_regime = _df(conn, symbol, config.REGIME_TIMEFRAME)
  df_setup = _df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME)
  df_trig = _df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME)
  if df_regime.empty or df_setup.empty or df_trig.empty:
      return []

  regime_ms = _assert_interval(df_regime, config.REGIME_TIMEFRAME, symbol, "regime")
  setup_ms = _assert_interval(df_setup, config.SIGNAL_PATTERN_TIMEFRAME, symbol, "setup")
  trigger_ms = _assert_interval(df_trig, config.SIGNAL_TRIGGER_TIMEFRAME, symbol, "trigger")

  labels = classify_series(
      df_regime,
      adx_trend_threshold=params.adx_trend_threshold,
      atr_extreme_percentile=params.atr_extreme_percentile,
  )
  close_regime = df_regime.index.to_numpy() + regime_ms
  close_setup = df_setup.index.to_numpy() + setup_ms
  ts_trig = df_trig.index.to_numpy()
  close_trig = ts_trig + trigger_ms
  ```
  Replace the module docstring's no-lookahead block with tier-role wording:
  ```
  No-lookahead guarantees (stated in tier roles, not in fixed timeframes —
  the tiers are config.REGIME_TIMEFRAME / SIGNAL_PATTERN_TIMEFRAME /
  SIGNAL_TRIGGER_TIMEFRAME, currently 1d / 4h / 1h):
    - Regime labels come from classify_series, whose indicators use only
      trailing windows; the label applied at time t is the last REGIME bar
      CLOSED by t.
    - Candidates at a TRIGGER bar use only SETUP bars closed by that bar's
      close, and pivots are only confirmed with PIVOT_SPAN closed bars after
      them (same as live). Coarser tiers do not change this: regime and setup
      bar closes fall exactly on trigger-bar boundaries, so searchsorted's
      side="right" includes only bars genuinely closed at or before the
      trigger close.
    - The TRIGGER slice ends at the bar under evaluation.
    - Time-stop after MAX_HOLD_BARS_TRIGGER trigger bars.
  ```
  Everything else — `regime_at`, `candidates_for`, `cand_cache`, the exit branches, `close_out`, the `j <= entry_j` next-bar rule, the conservative same-bar stop rule — is **renamed only, never restructured**.
- **MIRROR**: `TIMEFRAME_INDIRECTION` and `SEARCHSORTED_ALIGNMENT` above.
- **IMPORTS**: none new (`numpy as np`, `pandas as pd`, `storage` already imported).
- **GOTCHA**: `engine.py:236`'s `max(0, j - (config.VOLUME_LOOKBACK + 1))` and `engine.py:225`'s `j < 1` guard are in **trigger bars** and are already correct at any tier. Renaming the variable is the whole change; rescaling the slice would silently break `check_breakout`'s volume window. Also, if Phase 2 has landed, `atr_1h` / `atr_close_1h` exist here — rename to `atr_setup` / `atr_setup_vals` and preserve Phase 2's index-alignment invariant (both derive from `df_setup`; never slice one without the other).
- **VALIDATE**: `grep -n "_15m\|m15\|_4h\|_1h" src/trading_bot/backtest/engine.py` → no matches. `.venv/bin/pytest tests/test_backtest.py -v` after Task 6.

### Task 3: Add the runtime tier assertion (`_assert_interval`)
- **ACTION**: Add a module-level helper to `src/trading_bot/backtest/engine.py`, just below `_df` (after line 88).
- **IMPLEMENT**:
  ```python
  def _assert_interval(df: pd.DataFrame, timeframe: str, symbol: str, role: str) -> int:
      """Verify a loaded series' bar spacing matches its configured timeframe.

      The PRD's named mitigation for this phase's risk. Every no-lookahead
      guarantee in this module assumes df_trig really is
      SIGNAL_TRIGGER_TIMEFRAME bars, close_setup lands on trigger-bar
      boundaries, and max_hold counts bars of the configured duration. If
      config and stored data disagree — a partial tier migration, a series
      backfilled under the wrong key, a fixture seeded at the old interval —
      all of them fail SILENTLY, in the flattering direction. Raise instead.

      Uses the MEDIAN inter-bar spacing so a handful of missing candles cannot
      trip the check; a wholesale interval mismatch always shifts the median.

      Args:
          df: Loaded OHLCV frame, epoch-ms index, ascending.
          timeframe: The config timeframe key this series was loaded under.
          symbol: For the error message.
          role: "regime" | "setup" | "trigger", for the error message.

      Returns:
          The expected interval in milliseconds (storage.TIMEFRAME_MS[timeframe]).

      Raises:
          ValueError: If the observed median spacing differs from the expected
              interval.
      """
      expected = storage.TIMEFRAME_MS[timeframe]
      ts = df.index.to_numpy()
      if len(ts) >= 2:
          observed = int(np.median(np.diff(ts)))
          if observed != expected:
              raise ValueError(
                  f"{symbol} {role} series spacing is {observed} ms but config "
                  f"names timeframe {timeframe!r} ({expected} ms). The tier "
                  f"configuration and the stored data disagree; every "
                  f"no-lookahead guarantee in this engine would be void."
              )
      return expected
  ```
- **MIRROR**: `GUARD_STYLE` — `walkforward.py:133-136` raises `ValueError` for a config/data contradiction. Use `raise`, **not** `assert`: `assert` is stripped under `python -O`, and this is exactly the invariant that must never be optional.
- **IMPORTS**: none new.
- **GOTCHA**: `storage.TIMEFRAME_MS[timeframe]` raises `KeyError` for `"1d"` until Phase 1 lands — that is the desired hard failure, not something to defend against with a `.get()`. Do **not** add a fallback. Also, `len(ts) < 2` returns silently: a 1-bar series has no observable spacing, and the existing `df.empty` guard already handles the degenerate case.
- **VALIDATE**: covered by `tests/test_tiers.py` (Task 7).

### Task 4: Rename and reword the signal modules; fix the fade path's missing `interval_ms`
- **ACTION**: Edit `src/trading_bot/signals/setup.py` (docstrings 1-21, 45, 104, 312; `scan_breakout_signals` 259-293), `src/trading_bot/signals/meanrev.py` (docstrings 1-19, 41-52, 191; `scan_fade_signals` 256-273), `src/trading_bot/signals/breakout.py` (docstrings only), `src/trading_bot/signals/scan.py:5`, `src/trading_bot/signals/patterns.py` (docstrings only), `src/trading_bot/regime/classifier.py:135` (docstring only).
- **IMPLEMENT**: `scan_breakout_signals` becomes:
  ```python
  df_setup = _load_df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME, now_ms)
  df_trig = _load_df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME, now_ms)
  if df_setup.empty or df_trig.empty:
      return []

  df_setup = df_setup.tail(config.PATTERN_LOOKBACK_BARS)
  # Trigger needs the volume window, the crossing pair, and any extra bars the
  # configured trigger lookback may reach back over. All counts are in
  # TRIGGER-timeframe bars and are therefore tier-independent.
  trigger_bars = config.VOLUME_LOOKBACK + 1 + max(1, config.BREAKOUT_TRIGGER_LOOKBACK_BARS)
  df_trig = df_trig.tail(trigger_bars)

  trigger_interval = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
  latest_close = int(df_trig.index[-1]) + trigger_interval
  if now_ms - latest_close >= trigger_interval:
      logger.warning(
          "%s %s data is %d ms behind now_ms; a breakout may already be older "
          "than the %d-bar trigger window",
          symbol,
          config.SIGNAL_TRIGGER_TIMEFRAME,
          now_ms - latest_close,
          config.BREAKOUT_TRIGGER_LOOKBACK_BARS,
      )
  ```
  In `scan_fade_signals`, the same renames **plus** the correctness fix:
  ```python
  from trading_bot.data import storage  # add to meanrev.py imports
  ...
  trigger_interval = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
  df_trig = df_trig.tail(config.VOLUME_LOOKBACK + 2)
  ...
  for candidate in detect_fade_setups(df_setup):
      event = check_breakout(
          df_trig, _to_trigger_candidate(candidate), interval_ms=trigger_interval
      )
  ```
- **MIRROR**: the breakout path at `setup.py:286` already passes `interval_ms=`; mirror it exactly.
- **IMPORTS**: `from trading_bot.data import storage` in `meanrev.py` (it currently imports `config`, `bollinger`, `current_regime`, `check_breakout`, `PatternCandidate`, and `Signal`/`_load_df` from `setup`). No circularity: `storage` imports only `config`.
- **GOTCHA**: The `interval_ms` fix changes fade behavior slightly (gapped crossing pairs are now skipped), so a fade test could flip. That is a *correct* flip: without it, `check_breakout`'s fresh-crossing test compares against a "preceding close" that may be hours stale, which is precisely the class of silent lookahead-flavored bug this phase exists to eliminate — and it gets worse, not better, at 1H bars. If a `test_meanrev.py` assertion breaks, verify the fixture is contiguous rather than reverting the fix. Second gotcha: `patterns.py` and `pivots.py` get **docstring edits only**; their geometry is positional and tier-agnostic.
- **VALIDATE**: `grep -rn "_15m\|interval_15m\|m15" src/trading_bot/signals/` → no matches. `grep -rn "15m" src/trading_bot/signals/` → no matches (all docstring mentions reworded).

### Task 5: Re-derive and re-freeze `k` (`ATR_STOP_MULTIPLE`) against 4H ATR — *Phase-2 artifact*
- **ACTION**: Measure, then confirm-or-adjust `config.ATR_STOP_MULTIPLE` once. **Depends on Phase 2 having landed** (`ATR_STOP_PERIOD`, `ATR_STOP_MULTIPLE`, `risk/atr_stop.py::cost_ratio`, `COST_RATIO_CEILING`).
- **IMPLEMENT**: Phase 2's plan states explicitly (its NOT Building, line 168): *"`k = 1.5` is frozen now against 1H-tier ATR; if Phase 4 changes the setup timeframe, `k` is re-derived then, not swept."* Phase 4 changes the setup timeframe, so do the derivation exactly once, from the cost constraint and nothing else:
  ```python
  # Read-only measurement script; do NOT commit as a module.
  from trading_bot import config
  from trading_bot.data import storage
  from trading_bot.indicators.wilder import atr
  from trading_bot.risk.atr_stop import cost_ratio
  import pandas as pd

  conn = storage.connect()
  for sym in config.SYMBOLS:
      rows = storage.load_candles(conn, sym, config.SIGNAL_PATTERN_TIMEFRAME)
      df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).set_index("ts")
      a = atr(df, period=config.ATR_STOP_PERIOD)
      # median stop distance as a fraction of price, at k = ATR_STOP_MULTIPLE
      med = float((config.ATR_STOP_MULTIPLE * a / df["close"]).dropna().median())
      print(sym, f"risk_pct={med:.5f}",
            f"c={cost_ratio(med, config.FEE_PCT, config.SLIPPAGE_PCT):.4f}")
  ```
  Expected (PRD Open Question #2, measured at 4H setup bars, all-taker): `c` ≈ **7.1% / 5.2% / 3.7%** for BTC / ETH / SOL — all inside `COST_RATIO_CEILING = 0.10`. If the measurement confirms this, **leave `ATR_STOP_MULTIPLE = 1.5` exactly as Phase 2 set it** and record the measured `c` per symbol in the plan-completion note. Only if some symbol exceeds 0.10 do you raise `k` to the *smallest* value that clears 0.10 on all three, and record the derivation.
- **MIRROR**: Phase 2's `risk/atr_stop.py::cost_ratio` — do not reimplement the cost formula.
- **IMPORTS**: N/A (throwaway script; if you prefer, put it under `scripts/`, which already holds `build_review_chart.py` and `export_bar_annotations.py`).
- **GOTCHA**: **Deriving `k` from the cost constraint is not sweeping `k`.** Sweeping means choosing `k` to maximize returns, which the PRD forbids by name ("Sweeping the stop multiple `k`" is in NOT Building, and the Phase 5 buffer sweep already burned a DoF on BTCUSDT 2023–2026). Derive from `c ≤ 0.10`, freeze, never look at PnL. If `k` changes, note it as a consumed DoF.
- **VALIDATE**: script output shows `c ≤ 0.10` on all three symbols; the value in `config.ATR_STOP_MULTIPLE` is justified by that output alone.

### Task 6: Retarget the test fixtures to the configured tiers
- **ACTION**: Edit `tests/test_backtest.py`, `tests/test_signals.py`, `tests/test_meanrev.py`, `tests/test_classifier.py` per the §E table.
- **IMPLEMENT**: Replace hardcoded interval constants with tier-derived ones. In `tests/test_backtest.py`:
  ```python
  REGIME_TF = config.REGIME_TIMEFRAME
  SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
  TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
  D_REG = storage.TIMEFRAME_MS[REGIME_TF]
  D_SET = storage.TIMEFRAME_MS[SETUP_TF]
  D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
  ```
  and `seed_scenario` becomes:
  ```python
  def seed_scenario(conn, outcome_rows_trig):
      """Seed regime/setup/trigger data: flag + trigger breakout at 109.5, then outcome bars."""
      seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 12, interval=D_REG)
      rows_setup = flag_rows()
      seed(conn, SETUP_TF, rows_setup, interval=D_SET)
      last_setup_ts = START + (len(rows_setup) - 1) * D_SET
      rows_trig = [[109.2, 109.4, 109.0, 109.2, 10.0]] * 21
      rows_trig.append([109.3, 110.0, 109.2, 109.9, 30.0])  # breakout bar (entry)
      rows_trig += outcome_rows_trig
      seed(conn, TRIGGER_TF, rows_trig, start=last_setup_ts + D_SET, interval=D_TRIG)
  ```
  Note `start=last_setup_ts + D_SET` (was `last_1h_ts + H1`) — the trigger series must begin *after* the last setup bar closes, so the trigger bar is genuinely post-pattern. Do the analogous substitutions in `test_signals.py:536` (`start=last_setup_ts - 19 * D_TRIG`) and `test_meanrev.py:214` (`start=last_setup_ts - 17 * D_TRIG`). `tests/test_classifier.py` needs exactly one edit: `TF = config.REGIME_TIMEFRAME`.
- **MIRROR**: `TEST_FIXTURE_STYLE` above — `seed()`'s `(start, interval, rows)` contract stays; only the interval source changes.
- **IMPORTS**: `from trading_bot import config` is already imported in all four files.
- **GOTCHA**: `test_backtest.py:73` seeds only **12** regime bars while `REGIME_MIN_BARS = 207`. That is fine and must stay fine: `patch_trending` (`test_backtest.py:83-87`) monkeypatches `engine.classify_series` wholesale, so the warmup gate is never reached. Do **not** inflate the fixture to 207 bars — you would be testing the classifier, not the engine, and slow the suite down for nothing. Second gotcha: `test_backtest.py:101,116` and `test_meanrev.py:230-231` assert on Phase-2-retired constants (`BREAKOUT_STOP_BUFFER_PCT`, `MAX_RISK_PCT`, `MIN_REWARD_PCT`) — **leave them entirely alone**; Phase 2's Task 6/8 rewrites them. If Phase 2 has already landed they will already read differently, and this task must not re-touch them.
- **VALIDATE**: `.venv/bin/pytest tests/test_backtest.py tests/test_signals.py tests/test_meanrev.py tests/test_classifier.py -v` — all pass **with fixture-interval edits only**. Any assertion that needs a semantic change is a red flag: stop and investigate a real regression.

### Task 7: New `tests/test_tiers.py` — the phase's own regression net
- **ACTION**: Create `tests/test_tiers.py`.
- **IMPLEMENT**:
  ```python
  """Tier-configuration invariants (Phase 4: tier shift to 1D/4H/1H).

  These tests exist so that a future edit cannot silently reintroduce a
  hardcoded timeframe assumption. They assert properties of the CONFIGURATION
  and of the engine's tier guard, not of any strategy outcome.
  """

  import pathlib
  import re

  import pandas as pd
  import pytest

  from trading_bot import config
  from trading_bot.backtest.engine import _assert_interval
  from trading_bot.data import storage

  SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "trading_bot"


  def _frame(interval_ms, n=10):
      ts = [1_700_000_000_000 + i * interval_ms for i in range(n)]
      return pd.DataFrame(
          {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
          index=pd.Index(ts, name="ts"),
      )


  class TestTierConfiguration:
      def test_all_three_tiers_are_known_timeframes(self):
          for tf in (
              config.REGIME_TIMEFRAME,
              config.SIGNAL_PATTERN_TIMEFRAME,
              config.SIGNAL_TRIGGER_TIMEFRAME,
          ):
              assert tf in storage.TIMEFRAME_MS, f"{tf} missing from TIMEFRAME_MS"

      def test_tiers_are_strictly_coarse_to_fine(self):
          """Regime must be coarser than setup, setup coarser than trigger."""
          reg = storage.TIMEFRAME_MS[config.REGIME_TIMEFRAME]
          setup = storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME]
          trig = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
          assert reg > setup > trig

      def test_coarser_closes_land_on_trigger_boundaries(self):
          """The no-lookahead searchsorted alignment depends on this."""
          trig = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
          assert storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME] % trig == 0
          assert storage.TIMEFRAME_MS[config.REGIME_TIMEFRAME] % trig == 0

      def test_tiers_are_the_prd_phase_4_values(self):
          assert config.REGIME_TIMEFRAME == "1d"
          assert config.SIGNAL_PATTERN_TIMEFRAME == "4h"
          assert config.SIGNAL_TRIGGER_TIMEFRAME == "1h"

      def test_hold_limit_is_named_for_the_trigger_tier(self):
          assert hasattr(config, "MAX_HOLD_BARS_TRIGGER")
          assert not hasattr(config, "MAX_HOLD_BARS_15M")


  class TestAssertInterval:
      def test_matching_interval_returns_expected_ms(self):
          tf = config.SIGNAL_TRIGGER_TIMEFRAME
          ms = storage.TIMEFRAME_MS[tf]
          assert _assert_interval(_frame(ms), tf, "BTCUSDT", "trigger") == ms

      def test_mismatched_interval_raises(self):
          tf = config.SIGNAL_TRIGGER_TIMEFRAME
          wrong = storage.TIMEFRAME_MS[tf] // 4
          with pytest.raises(ValueError, match="spacing"):
              _assert_interval(_frame(wrong), tf, "BTCUSDT", "trigger")

      def test_single_bar_series_is_not_checked(self):
          tf = config.SIGNAL_TRIGGER_TIMEFRAME
          ms = storage.TIMEFRAME_MS[tf]
          assert _assert_interval(_frame(ms, n=1), tf, "BTCUSDT", "trigger") == ms

      def test_a_few_missing_bars_do_not_trip_the_median(self):
          tf = config.SIGNAL_TRIGGER_TIMEFRAME
          ms = storage.TIMEFRAME_MS[tf]
          df = _frame(ms, n=12).drop(index=[1_700_000_000_000 + 5 * ms])
          assert _assert_interval(df, tf, "BTCUSDT", "trigger") == ms


  class TestNoHardcodedTierLiterals:
      """Grep-audit-as-a-test: the PRD's named mitigation, made permanent."""

      SIGNAL_PATH = (
          "signals/setup.py", "signals/meanrev.py", "signals/breakout.py",
          "signals/scan.py", "signals/patterns.py", "signals/pivots.py",
          "backtest/engine.py", "regime/classifier.py",
      )

      def test_no_15m_mentions_on_the_signal_path(self):
          offenders = []
          for rel in self.SIGNAL_PATH:
              text = (SRC / rel).read_text()
              for i, line in enumerate(text.splitlines(), 1):
                  if re.search(r"15m|15M|900_000|900000", line):
                      offenders.append(f"{rel}:{i}: {line.strip()}")
          assert not offenders, "hardcoded 15m assumption reintroduced:\n" + "\n".join(offenders)

      def test_signal_path_never_hardcodes_a_timeframe_string(self):
          """Timeframes must be read from config, never inlined."""
          offenders = []
          for rel in self.SIGNAL_PATH:
              for i, line in enumerate((SRC / rel).read_text().splitlines(), 1):
                  if re.search(r'''["'](?:1m|5m|15m|30m|1h|2h|4h|6h|12h|1d|1w)["']''', line):
                      offenders.append(f"{rel}:{i}: {line.strip()}")
          assert not offenders, "inline timeframe literal:\n" + "\n".join(offenders)
  ```
- **MIRROR**: class-per-concern structure and direct assertions, per `tests/test_signals.py`.
- **IMPORTS**: as shown. `_assert_interval` is imported by its underscore name deliberately — `tests/test_signals.py:472` already reaches into `setup_mod._contiguous_tail`, so testing a private helper directly is established practice here.
- **GOTCHA**: `TestNoHardcodedTierLiterals` will fail loudly if Task 4's docstring rewording is incomplete — that is its job. Do not weaken the regexes to make it pass; reword the docstring. Note `data/`, `exchange/`, and `config.py` are deliberately **excluded** from `SIGNAL_PATH`: those legitimately name timeframes (`TIMEFRAME_MS`, the poller cron map, the tier definitions themselves).
- **VALIDATE**: `.venv/bin/pytest tests/test_tiers.py -v` — all pass.

### Task 8: End-to-end validation on real data, and record the numbers
- **ACTION**: Run the CLI against `data/ohlcv.db` (requires Phase 1's 1D backfill to be present).
- **IMPLEMENT**:
  ```bash
  # 1) 1D bars must exist for all three symbols before anything else.
  sqlite3 data/ohlcv.db "SELECT symbol, COUNT(*), datetime(MIN(ts)/1000,'unixepoch'), datetime(MAX(ts)/1000,'unixepoch') FROM ohlcv WHERE timeframe='1d' GROUP BY symbol;"
  # EXPECT: 3 rows, ~1300 bars each, 2023-01-01 -> ~2026-07-22.

  # 2) Regime tier resolves on 1D bars (no longer "uncertain" from warmup).
  .venv/bin/python -m trading_bot.cli regime

  # 3) Backtest end-to-end, starting AFTER the 207-bar 1D warmup.
  for s in BTCUSDT ETHUSDT SOLUSDT; do
    .venv/bin/python -m trading_bot.cli backtest --symbol $s --start 2023-07-27
  done
  ```
- **MIRROR**: N/A.
- **GOTCHA**: Running `backtest` without `--start` uses `BACKFILL_START = "2023-01-01"`, which is inside the 1D warmup — the first ~7 months will produce no trades and the run will look broken. That is expected, not a bug; pass `--start 2023-07-27`. Also: **do not run `walkforward`** as part of this phase's validation beyond a smoke check. Its OOS holdout is a one-shot, depleting resource reserved for Phase 7; a casual run spends it.
- **VALIDATE** — record all of these in the completion note:
  - [ ] Backtest completes without exception on all 3 symbols (phase success signal #1).
  - [ ] Trade count per symbol, and trades-per-60-day-fold implied by it, versus the old 15m-tier counts. Expect a large drop — this is the number Phase 7 needs for its `WF_MIN_TRADES ≥ 30` decision.
  - [ ] Median realized `risk_pct` and the resulting `c = cost_ratio(...)` per symbol; confirm `c ≤ 0.10` on all three (this is what Phase 2 could not close alone — PRD Open Question #2).
  - [ ] Median stop distance ≥ 1.0 × ATR(4H), carried over from Phase 2's success signal but now measured on the correct tier.
  - [ ] Mean and max holding duration in days, to sanity-check `MAX_HOLD_BARS_TRIGGER = 96` (4 days) against the observed exit-reason mix; if "time" exits dominate, note it for Phase 7's grid (the hold limit *is* on Phase 7's list of levers that actually bind) — **do not retune it here**.
  - [ ] `.venv/bin/pytest tests/ -q -m "not network"` → **189+ passed** (baseline is 188 passed, 1 deselected, measured 2026-07-26).

---

## Testing Strategy

### Unit Tests

New tests are specified in full in Task 7. The **existing** tests that carry this phase's real safety guarantee (retargeted fixtures only, assertions unchanged):

| Test (file:line) | Guarantee it encodes |
|---|---|
| `test_backtest.py:91` `test_target_exit` | Entry price is the trigger bar's **own** close — the core no-lookahead property |
| `test_backtest.py:107` `test_stop_exit_conservative_same_bar` | Intrabar order is never assumed favorably; stop fills first |
| `test_backtest.py:119` `test_time_exit` | Hold limit is counted in **trigger bars** (`max_hold_bars=2`) |
| `test_backtest.py:130` `test_unresolved_trade_closed_at_data_end` | Open trade at data end closes at last close, outcome `"end"` |
| `test_backtest.py:139` `test_inactive_regime_produces_no_trades` | Regime gate still suppresses on the coarser regime tier |
| `test_signals.py:387` `test_gap_in_the_crossing_pair_is_skipped` | The freshness invariant Task 4 extends to the fade path |
| `test_signals.py:523` `test_trending_flag_breakout_produces_signal` | Full live path end-to-end on the new tiers |
| `test_meanrev.py:200` `test_ranging_stretch_recross_produces_signal` | Task 4's `interval_ms` fix does not break the fade happy path |
| `test_classifier.py:47,57` warmup tests | `REGIME_MIN_BARS` gate still holds at the 1D tier |

### Edge Cases Checklist
- [x] Tier config names a timeframe absent from `TIMEFRAME_MS` (→ `KeyError`, deliberately unhandled: Phase 1 dependency)
- [x] Stored series spacing disagrees with config (→ `ValueError` from `_assert_interval`)
- [x] Series with real gaps (median-based check tolerates them)
- [x] 1-bar and empty series (existing `df.empty` guard + `len(ts) < 2` early return)
- [x] Backtest window entirely inside the 207-bar 1D regime warmup (→ zero trades; documented, validated via `--start`)
- [x] Coarser bar closes must align with trigger boundaries (asserted)
- [x] Fade trigger with a non-contiguous crossing pair (now skipped, matching breakout)
- [ ] Concurrent access — N/A (single-threaded backtest; `storage._db_lock` unchanged)
- [ ] Network failure — N/A (no network in this phase; the 1D fetch is Phase 1)
- [ ] Permission denied — N/A

---

## Validation Commands

Environment note: the venv **must** be the Homebrew python@3.11 one already present at `.venv` (`.venv/bin/python --version` → `Python 3.11.6`, pandas 3.0.3). `pandas-ta` is *not* a dependency — `pyproject.toml` lists only `ccxt`, `pandas`, `apscheduler`, `python-dotenv`, plus `pytest` under `[project.optional-dependencies] dev`. Always invoke through `.venv/bin/`.

### Static Analysis
```bash
.venv/bin/python -m py_compile \
  src/trading_bot/config.py \
  src/trading_bot/backtest/engine.py \
  src/trading_bot/signals/setup.py \
  src/trading_bot/signals/meanrev.py \
  src/trading_bot/signals/breakout.py \
  src/trading_bot/signals/scan.py \
  src/trading_bot/signals/patterns.py \
  src/trading_bot/regime/classifier.py
```
EXPECT: zero syntax errors. **There is no linter and no type checker configured** in this repo (no `[tool.ruff]`, `[tool.mypy]`, `[tool.black]`, no `Makefile`, no pre-commit config). Do not invent a lint/typecheck step; keep style consistent with surrounding code by inspection.

### Grep Verification (the audit, re-run as a gate)
```bash
grep -rn '15m\|15M\|900_000\|900000' src/trading_bot/signals src/trading_bot/backtest src/trading_bot/regime
# EXPECT: no matches

grep -rn 'MAX_HOLD_BARS_15M' src tests
# EXPECT: no matches

grep -rn '_15m\|m15\|_4h\b\|_1h\b' src/trading_bot
# EXPECT: no matches (data/ and exchange/ never used these names)

grep -rn '"1d"' src/trading_bot/data/storage.py src/trading_bot/config.py src/trading_bot/data/poller.py
# EXPECT: matches in all three — this is the Phase 1 dependency check
```

### Unit Tests
```bash
.venv/bin/pytest tests/test_tiers.py tests/test_backtest.py tests/test_signals.py \
                 tests/test_meanrev.py tests/test_classifier.py -v
```
EXPECT: all pass.

### Full Test Suite
```bash
.venv/bin/pytest tests/ -q -m "not network"
```
EXPECT: **189+ passed, 1 deselected** (baseline measured 2026-07-26: `188 passed, 1 deselected in 2.35s`). Zero regressions in `test_wilder.py`, `test_storage.py`, `test_backfill.py`, `test_binance_client.py`. If `test_poller.py` or `test_cli.py` fail, the cause is Phase 1's `"1d"` additions to `_CRON_BY_TIMEFRAME` / `config.TIMEFRAMES`, not this phase — escalate to Phase 1 rather than patching those fixtures here.

### Manual Validation
See Task 8. Additionally:
```bash
.venv/bin/python -m trading_bot.cli gap-report --start 2023-01-01   # 1d grid included, zero gaps
.venv/bin/python -m trading_bot.cli signal                          # scans 4H setups on 1H triggers
```

---

## Acceptance Criteria
- [ ] All 8 tasks completed
- [ ] All validation commands pass, including every grep gate
- [ ] `tests/test_tiers.py` created and green; existing no-lookahead assertions in `TestRunBacktest` / `TestCheckBreakout` green with **fixture-interval edits only**
- [ ] Runtime assertion in place: `_assert_interval` raises when the trigger series' spacing disagrees with `config.SIGNAL_TRIGGER_TIMEFRAME` (the PRD's named mitigation, and this phase's stated success signal)
- [ ] Backtest runs clean end-to-end on all 3 symbols on the new tiers
- [ ] `c ≤ 0.10` measured and recorded per symbol; `ATR_STOP_MULTIPLE` confirmed or re-derived once and re-frozen
- [ ] Every entry in the grep-audit tables above has a disposition that was actually applied (or an explicit "owned by Phase 1/2/5/6/7" note)
- [ ] No type errors — N/A, no type checker configured
- [ ] No lint errors — N/A, no linter configured
- [ ] Matches UX design — N/A, internal change

## Completion Checklist
- [ ] Code follows discovered patterns (`TIMEFRAME_INDIRECTION`, `SEARCHSORTED_ALIGNMENT`, `CLOSED_BAR_RULE`, `GUARD_STYLE`)
- [ ] Error handling matches codebase style (`raise ValueError` for a config/data contradiction; `return None` / `return []` for ordinary rejection; never `assert` for a production invariant)
- [ ] Logging follows conventions (`logging.getLogger("trading_bot")`, WARNING for staleness with the timeframe name interpolated, DEBUG for skipped crossings)
- [ ] Tests follow test patterns (class-per-concern, existing `seed`/`make_df`/`breakout_df`/`seed_candles` helpers reused and retargeted, not reinvented)
- [ ] No hardcoded values — every timeframe is read from `config`, every interval from `storage.TIMEFRAME_MS`
- [ ] Documentation updated — `engine.py`'s no-lookahead docstring rewritten in tier-role language; `config.py` comments state the new durations and flag the deliberate `ATR_PERCENTILE_WINDOW` semantic change
- [ ] No unnecessary scope additions — Donchian, fade re-qualification, walk-forward repair, sizing, alerting, and the 1D backfill are untouched
- [ ] Consumed degrees of freedom recorded: the `MAX_HOLD_BARS_TRIGGER` duration rescale (bar count unchanged, so arguably zero) and any `k` adjustment from Task 5
- [ ] Self-contained — no questions needed during implementation

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 1 has not landed: `storage.TIMEFRAME_MS` lacks `"1d"` and no 1D bars exist, so nothing downstream of Task 1 can run | **High** (PRD flags the `fapi.binance.com` TLS block as a High risk; still unresolved as of the memory note dated 2026-07-05) | **Blocking** | Verify with the `sqlite3` command in Task 8 step 1 **before** starting. Tasks 1–4, 6, 7 are all writable and unit-testable against synthetic fixtures without 1D data; only Task 5 and Task 8 truly need it. Sequence accordingly and escalate the connectivity issue early |
| An unaudited 15m assumption survives and produces a silently mis-tiered (flatteringly biased) backtest — the PRD's named risk for this phase | Medium | **High** | Three independent nets: the enumerated disposition tables above (every hit, every file:line); `_assert_interval` raising at runtime on all three series; and `TestNoHardcodedTierLiterals` failing CI if any timeframe literal reappears on the signal path |
| Phase 2 is in flight and rewrites the same `build_signal` / `build_fade_signal` / `engine.py` lines → merge conflict or a double-edit that reverts one of them | **High** (Phase 2 status is `in-progress`) | Medium | Every task above names its Phase-2 overlaps and defers explicitly (`config.py:67-85`, `test_backtest.py:101,116`, `test_meanrev.py:230-231`, `atr_1h`/`atr_close_1h`, `close_out`'s funding term). Land Phase 2 first if at all possible; if not, apply Phase 4 as renames only on those lines |
| 1D regime warmup (207 bars ≈ 207 days) silently zeroes out early backtest windows and the first walk-forward fold | **High** (certain, given `WF_TRAIN_DAYS = 180` from `BACKFILL_START`) | Medium | Documented in Ground Truth with the exact cutover date (≈2023-07-27); Task 8 mandates `--start 2023-07-27`; handed to Phase 7 as a fold-window constraint rather than silently absorbed |
| Trades per fold collapse under coarser bars, so Phase 7's `WF_MIN_TRADES ≥ 30` becomes unreachable on 3 symbols | Medium-High | Medium | Task 8 requires measuring and recording trades-per-fold now, so Phase 7 designs its pooling against a real number. Pooling all 3 symbols (the PRD's "only free lunch") is Phase 7's stated remedy and is out of scope here |
| 1H exit resolution is coarser than 15m, so the conservative same-bar stop-first rule fires more often and understates results | Medium | Low-Medium | Deliberately conservative in the *pessimistic* direction, so it cannot flatter the strategy. PRD Open Question #4 notes wide ATR stops (1.3–2.5%) invoke the rule far less than 0.2% stops did. 1m ingestion is a `Could`, out of scope |
| `ATR_PERCENTILE_WINDOW = 180` on 1D bars silently changes the extreme-vol gate's meaning (30-day → 180-day volatility context), altering regime occupancy from the measured-healthy baseline | Medium | Medium | Decision and reasoning documented in §D and in the config comment. Task 8's `cli regime` check plus the backtest's regime-bucket breakdown surface any occupancy collapse. **If occupancy degenerates, report it — do not retune the window here**, since that is a re-tune of the one validated component and a consumed DoF |
| Fixing the fade path's missing `interval_ms` changes fade behavior mid-phase, confounding attribution | Low-Medium | Low | It is a strict no-lookahead correctness fix, not a tuning change, and its effect is only to *reject* candidates whose freshness test was invalid. Called out explicitly in Task 4's GOTCHA and in the completion note |
| The `k = 1.5` re-derivation (Task 5) drifts into a return-maximizing sweep | Low | **High** (spends a degree of freedom and reintroduces exactly the Phase-5 mistake) | Task 5 prescribes the derivation from `c ≤ 0.10` only, with an explicit prohibition on looking at PnL, and requires recording the measured `c` per symbol |

## Notes
- **This phase changes exactly one variable.** Signal methods (breakout geometry, Bollinger fade) and the risk model (Phase 2's ATR stop, `RR_FLOOR`, frozen costs) are held fixed; only the tiers move. That isolation is the PRD's sequencing rationale ("Change one thing at a time"). Resist bundling any Phase 5/6/7 work in.
- **The audit's central finding**: the codebase is already almost fully tier-parameterized — every timeframe loaded by config name, every interval from `storage.TIMEFRAME_MS`, `check_breakout` generic over `interval_ms`/`lookback_bars`. The three genuine hazards are (a) `MAX_HOLD_BARS_15M`'s name encoding a tier into a bar count, (b) `meanrev.py:266` omitting `interval_ms`, (c) 37 test-fixture literals that would keep the *tests* on the old tiers while production moved — the classic way a tier shift passes CI while being wrong.
- **Handoff to Phase 7**: measured trades-per-fold on the new tiers, the ≈2023-07-27 usable-history start date, measured `c` per symbol, and the exit-reason mix (does `MAX_HOLD_BARS_TRIGGER` bind?). Phase 7 owns pooling, `WF_MIN_TRADES`, DSR, grid contents, and the one-shot gate.
- **Handoff to Phase 5**: a 4H setup tier with 7,799 bars/symbol — ample for Donchian-55 inside `PATTERN_LOOKBACK_BARS = 180` — and a `check_breakout` interface verified tier-agnostic.
- Before starting, check `git status` (currently: PRD modified, `.claude/PRPs/plans/` untracked, branch `master`). The PRD warns that branch `fix/phase3-breakout-detection` touches `engine.py` / `config.py` / `signals/*` / `tests/test_signals.py`; its premise is superseded — resolve or abandon it first, as it collides with Tasks 2, 4, and 6.
