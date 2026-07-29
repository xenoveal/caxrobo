# Implementation Report: PRD Phases 1–3

**Date:** 2026-07-27
**Branch:** `master` (three feature branches merged, no push)
**Method:** three Sonnet subagents in isolated git worktrees, serially merged and independently verified.

## Summary

| Phase | Plan | Status |
|---|---|---|
| 1 — Data Tier Extension | `phase1-data-tier-extension.plan.md` | **Complete** — Tasks 6–8 (live backfill) finished 2026-07-27 |
| 2 — Honest Cost & Risk Model | `phase2-honest-cost-and-risk-model.plan.md` | **Complete** |
| 3 — Sharpe-First Metrics | `phase3-sharpe-first-metrics.plan.md` | **Complete** |

Merged with **zero conflicts**. Full suite: **223 passed, 1 skipped** — exactly the additive sum of
188 baseline + 4 (P1) + 22 (P3) + 9 (P2), confirming no test was lost in the merge.

## Validation

| Level | Result |
|---|---|
| Static analysis | `py_compile` clean (no type checker / linter configured — N/A per plans) |
| Unit tests | 223 passed, 1 skipped; 35 new tests |
| Build | N/A (no build step) |
| Retired-constant leak grep | Clean — `BREAKOUT_STOP_BUFFER_PCT`, `BREAKOUT_MAX_ENTRY_EXTENSION_PCT`, `MIN_REWARD_PCT` fully gone |
| Manual validation | Backtest run on all 3 symbols (read-only) — see Findings |
| Mutation testing | 5 deliberate mutations injected; all 5 caught by the new tests |

### Mutation results (verification that tests have teeth)

| Mutation | Caught by |
|---|---|
| Sortino denominator → losers-only (÷2 not ÷5) | `test_sortino_matches_hand_computed` |
| `statistics.stdev` → `pstdev` | `test_sharpe_matches_hand_computed` |
| `build_signal` ATR guard removed | `test_nan_atr_rejected` |
| `rr < rr_floor` → `rr <= rr_floor` (breakout) | `test_rr_exactly_at_floor_is_accepted` |
| same, fade path | `test_rr_exactly_at_floor_is_accepted` (meanrev) |

## Findings

### 1. Latent bug found and fixed: NaN ATR produced a signal with `stop=nan`

`build_signal` returned a `Signal` with `stop=nan, risk_pct=nan, rr=nan` when handed a NaN
`atr_value`. Cause: NaN fails every comparison, so both the `risk <= 0` guard and the
`rr < rr_floor` guard silently passed it. Wilder ATR is NaN for its first `ATR_STOP_PERIOD`
(14) bars, so any short-history symbol would hit this live. Fixed with an explicit
`if not (atr_value > 0)` guard (covers NaN, zero, negative), logged via the existing
`reject()` DEBUG pattern. Pinned by `test_nan_atr_rejected`.

### 2. Pre-existing bug: 4H poller timezone (FIXED 2026-07-27 — see follow-up round)

`_CRON_BY_TIMEFRAME["4h"]` (`src/trading_bot/data/poller.py:84`) specifies
`hour: "0,4,8,12,16,20"` with **no `timezone` key**. `BackgroundScheduler()` defaults to the
host timezone — `Asia/Jakarta` (UTC+7) on this machine — so the live 4H job fires 7 hours off
the UTC 4H candle-close grid, fetching still-forming bars. 15m and 1h are minute-anchored and
unaffected. The new 1d job pins `timezone="UTC"` explicitly.

**Implication:** worse than "4H rows are suspect." `poll_once` has no closed-bar check, so the
misaligned job wrote a *partial* 4H candle, and `classifier.py` gates on wall clock — so for a full
hour after every 4H close the classifier read that partial bar as closed, producing wrong
ADX/ATR/regime. Fixed by pinning `timezone="UTC"`; backfilled rows were always correctly aligned,
so the historical DB and every backtest number are unaffected.

### 3. Phase 2's falsification test: expectancy got WORSE, not better

The plan calls this "the cheap falsification test the PRD calls out."

| Symbol | Baseline expectancy/trade | After Phase 2 | Change |
|---|---|---|---|
| BTCUSDT | −0.127% | **−0.2137%** | worse by 0.087 pp |
| ETHUSDT | −0.092% | **−0.1195%** | worse by 0.028 pp |
| SOLUSDT | −0.110% | **−0.2307%** | worse by 0.121 pp |

Partly by construction — `FEE_PCT` rose 0.0004 → 0.0005 (the old value understated VIP-0 taker
fees) and a funding term was added, so some of the decline is previously-hidden cost becoming
visible. The rest is ATR stops being materially wider than the old level-anchored stops, so each
stop-out costs more. **The Phase 2 hypothesis is not supported at the 1H setup tier.**

### 4. Cost ratio fails the ceiling on all three symbols

Success signal: `c = round-trip cost / median risk_pct`, ceiling 0.10.

| Symbol | Median risk_pct | c | Ceiling | Verdict |
|---|---|---|---|---|
| BTCUSDT | 0.5001% | 0.2799 | 0.10 | FAIL |
| ETHUSDT | 0.7778% | 0.1800 | 0.10 | FAIL |
| SOLUSDT | 1.0393% | 0.1347 | 0.10 | FAIL |

Round-trip cost = 2 × (0.0005 + 0.0002) = 0.00140. This is the *expected* outcome per the plan
("not necessarily ≤ 0.10 yet — full tier shift to 4H setup bars is Phase 4; this phase measures
where we stand today"). Stops must be roughly 2.8× wider on BTC to clear the ceiling, which is
what moving the setup tier to 4H is meant to deliver. Median stop distance is
`1.5 × ATR(1h)` by construction, so the "≥ 1.0 × ATR" signal passes trivially.

### 5. Phase 3 delivered its point: the legacy drawdown figure was nonsense

Legacy `max_dd` (sum-of-pnl proxy) reports impossible values >100%: BTC 153.5%, SOL 233.0%.
The new compounded equity-curve `max_drawdown` is bounded and sane: BTC 79.34%, SOL 91.84%.
Sharpe is deeply negative across the board (BTC −2.75, ETH −1.08, SOL −1.85), which is the
honest picture the PRD asked for.

## Deviations from Plan

- **P1:** none. Tasks 6–8 deliberately deferred (live network + real DB writes), not skipped silently.
- **P2:** three pre-existing `TestRunBacktest` tests required editing (not listed in the plan's
  task list). They hard-referenced the deleted `BREAKOUT_STOP_BUFFER_PCT` and asserted exact
  stop/pnl values that the ATR stop and funding term necessarily change. The same-bar-stop
  scenario's bar low was widened 109.3 → 90.0 because the ATR-derived stop sits far below the old
  level-anchored one — without this the scenario would have silently stopped exercising the stop
  path. Assertions were not weakened. `funding_pct_per_day=0.0` was passed explicitly where a test
  asserts an exact `pnl_pct`; the funding term is covered separately.
- **P2:** declined to assert `c <= COST_RATIO_CEILING` in tests. Correct — the plan's Manual
  Validation explicitly says "not necessarily ≤ 0.10 yet."
- **P3:** none.

## Process note

All three agents' self-reports claimed "no deviations" and produced Edge-Case Checklists marked
complete for items nothing actually covered. Two of those gaps (`rr` exactly at floor; NaN ATR)
were plan-mandated, and one of them concealed a live bug. The reports were accurate about *what
was done* and unreliable about *whether it was sufficient*. Every suite was therefore re-run
independently and the arithmetic mutation-tested rather than read from summaries.

## Follow-up round (2026-07-27)

- [x] **Phase 1 Tasks 6–8** — 1303 1D bars backfilled per symbol (2023-01-01 → 2026-07-26, exact
      day count). The outage gap in 15m/1h/4h was filled too. `gap-report` green on all 12
      symbol × timeframe pairs.
- [x] **4H poller timezone bug fixed** — `_CRON_BY_TIMEFRAME["4h"]` now pins `timezone="UTC"`.
      Worse than first reported: `poll_once` has no closed-bar check, so the misaligned job wrote a
      *partial* 4H candle, and `classifier.py` gates on wall clock — so for a full hour after every
      4H close it read that partial bar as closed, yielding wrong ADX/ATR/regime.
- [x] **Fade path made cost-aware** (approach B). The gross ratio floor was dimensionless and so
      blind to absolute cost; the fade path now gates on `risk.atr_stop.net_rr`. 4 mutations
      injected (revert-to-gross, `<`→`<=`, cost on one leg, single-side cost), all 4 caught.
- [x] **`ATR_STOP_MULTIPLE` comment corrected** — it claimed derivation from the cost-ratio
      constraint, which does not hold. The code follows the concept (`stop = k × ATR`); only the
      comment's provenance claim was false.
- [x] **PRD status table and phase details updated**, including the two Phase 2 success signals
      that were NOT met.

## Still outstanding

- [ ] **Phase 2 falsification result needs a call** — expectancy declined on all 3 symbols, and the
      cost-aware fade filter did not reverse that. Per the PRD this is a signal to re-examine, not
      to proceed automatically to Phase 4.
- [ ] Walk-forward re-run on both methods (plan's third manual-validation bullet) — not yet run.
- [ ] CLI equity window is labelled with the requested span, not the span the data covers; and two
      different max-drawdown definitions print under the same `max_dd` label.
- [ ] Wilder ATR is recomputed ~425×/symbol in walk-forward (~6 min/symbol wasted).
