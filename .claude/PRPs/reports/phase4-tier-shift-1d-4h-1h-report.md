# Implementation Report: Tier Shift to 1D / 4H / 1H (PRD Phase 4)

**Date:** 2026-07-27
**Branch:** `feat/phase4-tier-shift` (not pushed)
**Plan:** `.claude/PRPs/plans/v0.2.0/phase4-tier-shift-1d-4h-1h.plan.md`

## Summary

Moved the signal stack up one tier — **1D regime / 4H setup / 1H trigger** (from 4H / 1H / 15m) —
and removed every hardcoded 15-minute assumption behind it. Three config strings, a mechanical
rename of tier-named locals to role names, a runtime tier assertion, and retargeted test fixtures
that now derive their intervals from `config` so they can never again lag production.

**This is the phase that worked.** Sharpe improved on all three symbols, and the cost-frontier
constraint Phase 2 could not close is now closed for the sleeve it governs.

## Assessment vs Reality

| Metric | Predicted (Plan) | Actual |
|---|---|---|
| Complexity | Medium-High (wide audit surface, small diff) | Accurate |
| Files changed | 12 (1 new, 11 modified) | **14** (1 new, 13 modified) |
| Risk that mattered | "an unaudited 15m assumption survives" | None survived; 3 nets in place |

Two files beyond the plan's list: `scripts/export_bar_annotations.py` (referenced the renamed
`MAX_HOLD_BARS_15M` and would have crashed) and `tests/test_meanrev.py` (needed a real test for the
`interval_ms` fix — see Deviations).

## Tasks Completed

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | Flip tier strings, rename hold limit | Complete | `MAX_HOLD_BARS_TRIGGER = 96`, value unchanged → 24h becomes 4 days for free |
| 2 | Engine renames + no-lookahead docstring | Complete | Renames only; no control flow restructured |
| 3 | `_assert_interval` runtime guard | Complete | `raise ValueError`, not `assert` (survives `python -O`) |
| 4 | Signal renames, reword, fade `interval_ms` | Complete | Premise of the "defect" was wrong — see Findings |
| 5 | Re-derive and re-freeze `k` | Complete | **`k = 1.5` CONFIRMED, unchanged → zero DoF consumed** |
| 6 | Retarget test fixtures | Complete | Fixture-interval edits only; **zero assertions changed** |
| 7 | `tests/test_tiers.py` | Complete | 12 tests (plan specified 11; added one) |
| 8 | End-to-end validation | Complete | Numbers below |

## Validation Results

| Level | Status | Notes |
|---|---|---|
| Static analysis | Pass | `py_compile` clean; no linter/type checker configured (N/A per plan) |
| Unit tests | Pass | **242 passed, 1 skipped** (baseline 229 → +13) |
| Grep gates | Pass | All four gates clean, plus the inline-timeframe-literal gate |
| Build | N/A | No build step |
| Manual validation | Pass | `gap-report` zero gaps on all 12 pairs; `cli regime` resolves on 1D; backtest clean on 3/3 |
| Mutation testing | Pass | 3 injected, 2 caught immediately, 1 exposed a missing test (now caught) |

### Mutation results

| Mutation | Caught by |
|---|---|
| `_assert_interval`'s raise defanged (`if False`) | `test_mismatched_interval_raises` |
| `SIGNAL_TRIGGER_TIMEFRAME` reverted to `"15m"` | `test_tiers_are_the_prd_phase_4_values` |
| Fade `interval_ms=` removed | **Initially NOTHING** → new `test_fade_path_passes_the_trigger_interval_to_check_breakout` |

Note on mutation 2: because the fixtures are now tier-derived, they *follow* a tier mutation rather
than failing on it. `test_tiers_are_the_prd_phase_4_values` is the anchor that makes derived
fixtures safe — without that one literal assertion, a tier revert would pass the entire suite.

## Findings

### 1. The tier shift delivered what the PRD predicted — Sharpe improved on all three

Measured with `--start 2023-07-27` (after the 207-bar 1D warmup):

| Symbol | Sharpe (1H tier) | Sharpe (new tiers) | Trades | Equity max DD |
|---|---|---|---|---|
| BTCUSDT | −2.12 | **−0.09** | 150 | 27.45% (was 68.25%) |
| ETHUSDT | −1.15 | **−0.89** | 152 | 55.99% (was 70.32%) |
| SOLUSDT | −1.45 | **−0.56** | 205 | 60.78% (was 87.20%) |

BTC is now within noise of break-even, and its `trending/flag` bucket is genuinely positive
(+0.145%/trade, PF 1.11, n=47). Still negative overall on all three, so this does not clear the
gate — but it is the first change in this PRD that moved the number the right way, and it moved it a
lot. The cost-frontier thesis is supported.

### 2. `k = 1.5` confirmed against the 4H tier — no degree of freedom consumed

Derived from `c ≤ COST_RATIO_CEILING` only, never from PnL, per Task 5's prohibition. Measured
ex-ante on stored 4H history at `k = 1.5`:

| Symbol | median risk_pct | c | Verdict |
|---|---|---|---|
| BTCUSDT | 1.968% | 0.0711 | PASS |
| ETHUSDT | 2.679% | 0.0523 | PASS |
| SOLUSDT | 3.752% | 0.0373 | PASS |

This reproduces the PRD's predicted 7.1 / 5.2 / 3.7% almost exactly. `k` was **not** adjusted.
The stale `ATR_STOP_MULTIPLE` comment (written earlier the same day against the 1H tier, where the
ceiling was *not* satisfiable) was corrected in `config.py`.

### 3. Realized `c` passes on the breakout sleeve and fails on the fade sleeve — on all 3 symbols

Blended realized numbers initially looked like a failure (BTC c = 0.132). Splitting by method
explains it completely:

| Symbol | breakout: risk / c / stop÷ATR | fade: risk / c / stop÷ATR |
|---|---|---|
| BTCUSDT | 1.902% / 0.0736 PASS / 1.45× PASS | 0.799% / 0.1751 FAIL / 0.61× FAIL |
| ETHUSDT | 2.602% / 0.0538 PASS / 1.46× PASS | 1.216% / 0.1151 FAIL / 0.68× FAIL |
| SOLUSDT | 3.508% / 0.0399 PASS / 1.40× PASS | 1.382% / 0.1013 FAIL / 0.55× FAIL |

The breakout path clears both of Phase 4's success signals everywhere, at the expected ≈1.45×
(k = 1.5 minus warmup effects). The fade path fails both everywhere because its stop is the
excursion extreme, not `k × ATR`, and lands at 0.55–0.68 × ATR — tighter than the breakout's 1.5 ×.

**This is not a Phase 4 regression.** The fade stop is structural by explicit PRD decision, so no
tier shift can fix it; only re-qualifying or re-specifying the sleeve can. That is Phase 6's remit.
Recorded here rather than patched.

### 4. Trades per fold is still short of Phase 7's threshold, even pooled

| Scope | Trades | Trades / 60-day fold |
|---|---|---|
| BTCUSDT | 150 | 8.3 |
| ETHUSDT | 152 | 8.3 |
| SOLUSDT | 205 | 11.2 |
| **Pooled (3 symbols)** | **507** | **27.8** |

Pooling all three — the PRD's stated remedy and "only free lunch" — yields 27.8, still under
`WF_MIN_TRADES ≥ 30`. Phase 7 cannot assume pooling alone closes this gap. Usable span is
1,084–1,096 days from 2023-07-27.

### 5. The hold limit does not bind; stops dominate

| Symbol | mean hold | median | max | exits |
|---|---|---|---|---|
| BTCUSDT | 0.94d | 0.54d | 4.00d | stop 103, target 41, time 6 |
| ETHUSDT | 1.21d | 0.77d | 4.00d | stop 106, target 32, time 14 |
| SOLUSDT | 0.95d | 0.50d | 4.00d | stop 143, target 48, time 14 |

Time exits are 4–9% of trades, so `MAX_HOLD_BARS_TRIGGER = 96` (4 days) is not a binding lever —
it should be low priority in Phase 7's grid. Stops are ~68% of exits against a ~29% win rate:
the methods are being stopped out, not timed out. **Not retuned here** per Task 8.

### 6. The fade `interval_ms` "defect" was not reachable in production

The plan called `meanrev.py`'s missing `interval_ms` a "real defect, in scope" allowing a stale
prior close to read as a fresh re-cross. It is not reachable through `scan_fade_signals`:
`setup._load_df` already routes every series through `_contiguous_tail` (`setup.py:174-215`), which
trims to the longest evenly-spaced run ending at the latest bar. A gapped pair therefore cannot
reach `check_breakout` by that path at all.

Verified empirically: seeding a gapped trigger series produced no signal *with or without* the fix —
my first attempt at a regression test passed for the wrong reason (the trim left a single bar, so
there was no crossing pair to evaluate). The argument is still correct to pass — defense-in-depth for
any caller that bypasses the trim, and consistency with the breakout path — but it is **not** a live
bug fix, and the backtest numbers above are unaffected by it. The replacement test asserts the
pass-through directly and does fail when the argument is removed.

## Files Changed

| File | Action | Notes |
|---|---|---|
| `src/trading_bot/config.py` | UPDATED | 3 tier strings, hold-limit rename, `k` derivation recorded, `ATR_PERCENTILE_WINDOW` semantics documented |
| `src/trading_bot/backtest/engine.py` | UPDATED | `_assert_interval` added; all tier-named locals → role names; docstring rewritten |
| `src/trading_bot/signals/setup.py` | UPDATED | Renames + parameterized staleness warning |
| `src/trading_bot/signals/meanrev.py` | UPDATED | Renames + `interval_ms` + `storage` import |
| `src/trading_bot/signals/breakout.py` | UPDATED | Docstrings only |
| `src/trading_bot/signals/scan.py` | UPDATED | Docstring only |
| `src/trading_bot/signals/patterns.py` | UPDATED | Docstrings only |
| `src/trading_bot/regime/classifier.py` | UPDATED | Docstring only — no logic touched |
| `scripts/export_bar_annotations.py` | UPDATED | **Beyond plan** — stale `MAX_HOLD_BARS_15M` reference |
| `tests/test_tiers.py` | CREATED | 12 tests |
| `tests/test_backtest.py` | UPDATED | Tier-derived intervals, parameterized `seed_scenario` |
| `tests/test_signals.py` | UPDATED | Tier-derived intervals |
| `tests/test_meanrev.py` | UPDATED | Tier-derived intervals + new `interval_ms` test |
| `tests/test_classifier.py` | UPDATED | One line |

## Deviations from Plan

1. **`scripts/export_bar_annotations.py` fixed (not in the plan's file list).** It referenced
   `config.MAX_HOLD_BARS_15M`, so the rename would have broken it. `scripts/` is outside pytest
   `testpaths`, so no test would have caught it — the same failure mode that broke this script in
   Phase 2.
2. **Added `test_every_tier_is_an_ingested_timeframe`** to `test_tiers.py` (12 tests, plan specified
   11). A tier that the poller and backfill never fetch would silently stay empty forever; the plan's
   `TIMEFRAME_MS` check does not cover `config.TIMEFRAMES`.
3. **Replaced the planned fade gap-regression test** with a pass-through assertion, because the
   gap-based version passed for the wrong reason. See Finding 6.
4. **Corrected the `ATR_STOP_MULTIPLE` comment** written earlier the same session, which asserted the
   cost ceiling was unsatisfiable. True at the 1H tier, false at 4H.
5. **`--start 2023-07-27` used for all validation**, per the plan's Ground Truth warmup analysis.

## Issues Encountered

- One 15m tail bar drifted stale between backfill and validation (no poller running). Topped up;
  `gap-report` green on all 12 pairs. Unrelated to the tier shift — 15m is no longer a tier.
- `test_poller.py` / `test_cli.py` did **not** need the Phase-1 escalation the plan anticipated;
  Phase 1's `"1d"` additions had already landed with their fixture updates.

## Tests Written

| Test File | Tests | Coverage |
|---|---|---|
| `tests/test_tiers.py` | 12 | Tier ordering/divisibility, `TIMEFRAME_MS` + `TIMEFRAMES` coverage, PRD tier values, hold-limit rename, `_assert_interval` (match / mismatch / 1-bar / gaps), grep-audit-as-a-test ×2 |
| `tests/test_meanrev.py` | 1 | Fade path passes `interval_ms` to `check_breakout` |

### 7. Two residual items in `scripts/export_bar_annotations.py` (deliberately not fixed)

The plan scopes `scripts/` out, so only what was *broken* was fixed (the renamed constant) plus the
user-visible log line, which claimed "annotating N 15m bars" while annotating 1H bars. Left alone,
recorded here:

- **Internal locals are still tier-named** (`df_15m`, `df_4h`, `df_1h`, `close_15m`, `window_15m`,
  `ts_15m`). Cosmetic — they are all loaded via `config.SIGNAL_*_TIMEFRAME`, so they are correct,
  just misleadingly named. Not covered by `test_tiers.py`, whose `SIGNAL_PATH` excludes `scripts/`.
- **`check_breakout` is called without `interval_ms`** (`export_bar_annotations.py:198`), whereas the
  engine passes it. So the review chart can display a trigger the engine would have rejected for
  non-contiguity. A genuine fidelity gap in the review tool — worth fixing, but it changes chart
  output, so it should not be slipped in mid-phase.

## Next Steps

- [ ] `/code-review` on this branch
- [ ] **Phase 6 owns the fade sleeve's cost ratio** (Finding 3) — it is the only thing failing Phase 4's success signals
- [ ] **Phase 7 must not assume pooling closes `WF_MIN_TRADES`** (Finding 4): pooled is 27.8 < 30
- [ ] Phase 7 grid: deprioritize `MAX_HOLD_BARS_TRIGGER` (Finding 5); walk-forward not run here by design (OOS is a one-shot resource)
