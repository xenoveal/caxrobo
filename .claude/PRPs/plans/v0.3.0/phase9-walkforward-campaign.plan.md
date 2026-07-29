# Plan: Walk-Forward Campaign — the verdict on the key hypothesis (v0.3.0 Phase 9)

## Summary

This phase runs the **one campaign that answers the PRD's question**, and its deliverable is
a *locked protocol plus a measured report*, not a number anyone hopes for. It adds one module
(`campaign.py`), one CLI subcommand (`campaign`), one config block (`CAMPAIGN_*` / `HOLDOUT_*`),
one test file (`test_campaign.py`) and one report generator — then drives Phase 6's evolution
engine over a training span that **structurally cannot reach** the final holdout, selects a
champion by a tie-broken rule declared in advance, and evaluates it against Phase 1's extended
7-condition gate **exactly once**, with the buy-and-hold basket alongside and the cumulative
trial count the DSR was actually charged.

The single most important property is ordering: **the holdout span, the gate thresholds, the
symbol set, the population size, the generation count, the champion rule and the stopping rule
are all committed to `config.py` and to this plan before the campaign runs**, so no result can
be rationalised afterwards. The second most important property is that the plan says, in
advance and in measured numbers, what the likely verdict is — see
[Pre-registered verdict distribution](#pre-registered-verdict-distribution). It is not a pass.

## User Story

As the bot's sole operator, I want one pre-registered campaign whose protocol was fixed before
it ran and whose holdout is enforced unreachable by code rather than by good intentions, so
that whatever verdict comes out — northstar met, or a statistically honest "no" — I can trust
it and act on it, instead of having produced another number I have to discount.

## Problem → Solution

**Current**: v0.2.0's verdict was produced by a harness that pooled 3 correlated symbols
(mean r ≈ 0.76, effective N ≈ 1.2 — KNOWN-LIMITATIONS §0b), reported **+3.45% annualised
against a +29.4% buy-and-hold basket it never computed** (§0), failed 2 of 5 gate conditions,
and reported a single pass/fail bit that hid the per-condition detail (§1). Its headline "+68%
annualised" was a 90-day extrapolation from 23 trades (§2). There is no mechanism preventing a
second look at a holdout, no persistent trial accounting, and no record of what the gate does
not see.

**Solution**: a `campaign` command with three stages — `evolve`, `holdout`, `report` — where:
the evolution stage is handed `end_ms = config.HOLDOUT_START_MS` and additionally guarded by
`assert_no_holdout_overlap()` on every span it constructs; the holdout stage writes a
`holdout_consumption` row in `state.db` **before** it reads a single holdout bar, so a killed
process still burns the peek; the champion is chosen by a deterministic SQL query with a
declared tie-break; the gate is Phase 1's 7-condition `dict[str, bool]` reported **per
condition** with the benchmark basket beside it; and every figure in the report carries the
command and span that produced it.

## Metadata

- **Complexity**: **Large** (5 files: 3 created, 2 updated; ~700 net new lines incl. tests and
  the report generator; the risk is in protocol design and long-run mechanics, not in volume)
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md`
- **PRD Phase**: Phase 9 — Walk-forward campaign
- **Binding contract**: `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md`
- **Estimated Files**: 5 (3 new, 2 modified) + 2 generated report artifacts
- **Depends on**: Phases 7 and 8 — i.e. everything (contract §11). **Nothing runs parallel.**
- **Test baseline**: **286 tests collected** (`.venv/bin/python -m pytest --collect-only -q`,
  verified 2026-07-27). All 286 must stay green (contract §8, §12.1).

---

## UX Design

### Before

```
┌──────────────────────────────────────────────────────────────┐
│ $ python -m trading_bot.cli walkforward                      │
│   fold 0: …  fold 12: …                                      │
│   one-shot OOS (pooled): trades=23 …                         │
│   GATE: FAIL                    ← one bit. Which condition?  │
│                                 ← beaten by buy-and-hold?    │
│                                 ← how many trials charged?   │
│                                 ← may be re-run any time     │
└──────────────────────────────────────────────────────────────┘
```

### After

```
┌────────────────────────────────────────────────────────────────────────────┐
│ $ python -m trading_bot.cli campaign --stage evolve                        │
│   campaign v0.3.0-2026-07-28  seed=20260727  symbols=9                     │
│   evolution span 2023-07-27 → 2026-01-01   HOLDOUT 2026-01-01 → 2026-07-27 │
│   T_eval measured 11.4s → budget 10.0h → 3000 evals planned (120 x 25)     │
│   gen  1/25  best_fitness=…  trials=120   elapsed 0:23   [checkpointed]    │
│   …  (resumable: re-running --stage evolve continues at gen 14)            │
│                                                                            │
│ $ python -m trading_bot.cli campaign --stage holdout                       │
│   champion strategy_version=v0.3.0-g19-m044  graph_hash=8f2c…             │
│   sample projection: 8 symbols x 0.082 tr/day x 207 d = 136 raw            │
│                      independent-equiv ~= 31.1 (effN 1.834, measured P2)   │
│   *** HOLDOUT CONSUMED — recorded, run_index=1 ***                         │
│   GATE (7 conditions):                                                     │
│     sample_adequacy        136 trades      >= 30        PASS               │
│     sharpe                 1.34            >= 1.0       PASS               │
│     dsr                    0.0004          > 0.95       FAIL               │
│     max_drawdown           21.7%            <= 25%      PASS               │
│     per_symbol_expectancy  7 of 9 positive  all > 0     FAIL               │
│     beats_benchmark_return  +18.2% vs +9.4% basket       PASS              │
│     beats_benchmark_sharpe  1.34 vs 0.41 basket          PASS              │
│   n_trials charged: 3011   northstar: +18.2% vs >50% target: NOT MET       │
│   VERDICT: B2 — honest "no" (unprovable at this sample; see report)        │
│                                                                            │
│ $ python -m trading_bot.cli campaign --stage holdout   # second attempt    │
│   ERROR: holdout 2026-01-01→2026-07-27 already consumed 2026-07-28T04:11Z  │
│          (run_index=1). Re-running requires --force-holdout-rerun REASON,  │
│          which is recorded and stamps the report NOT A CLEAN HOLDOUT.      │
└────────────────────────────────────────────────────────────────────────────┘
```

### Interaction Changes

| Touchpoint | Before | After | Notes |
|---|---|---|---|
| Gate verdict | one `passed` bit | 7 named conditions, each with measured value + threshold | contract §4 |
| Null hypothesis | implicit zero | buy-and-hold basket printed beside the strategy | KNOWN-LIMITATIONS §0 |
| Trial accounting | `len(combos)*len(folds)` computed in-process | cumulative campaign ledger from `state.db`, printed | contract §4 |
| Holdout re-runs | unrestricted | refused unless `--force-holdout-rerun REASON`, logged, report stamped | this phase |
| Long runs | restart from zero | resumes at the last checkpointed generation | this phase |
| Numbers in reports | some derived | every figure carries its command + span | contract §12.5 |
| Northstar | conflated with gate | reported separately (gate has no return condition) | see discrepancy note |

---

## Pre-registered verdict distribution

**This section is written before the campaign runs and must not be edited afterwards.** A plan
that reads as though passing is the default has failed at its job. Three measured facts bound
expectations:

1. **v0.2.0 returned +3.45% annualised against a +29.4% buy-and-hold basket** and lost on
   return, Sharpe and (vs BTC) drawdown — "there is no axis on which it wins"
   (KNOWN-LIMITATIONS §0). **Those annualised figures are a genuine 3-year CAGR** — Phase 1
   confirmed the span is 1095 days = exactly 3 years — **not an extrapolation.** The
   extrapolation warning belongs *only* to §2's "+68% annualised" OOS headline, which is a
   90-day window with 23 trades. Do not conflate the two: §0's column is measured history,
   §2's is a projection. This plan's report must keep them typographically distinct.
2. **The gate failed on sample adequacy (23 trades vs a floor of 30) and on DSR (0.0210), and
   DSR still measured 0.742 with the multiple-testing correction switched off entirely**
   (`n_trials=1`). That is a *data* problem, not an `n_trials` convention problem
   (KNOWN-LIMITATIONS §1). Do not relitigate the convention.
3. **Contract §4's cumulative ledger pushes `n_trials` into the thousands**, which makes DSR
   *harder*, not easier — and §4 says explicitly that if that makes the gate unpassable, "that
   is the finding".

### The arithmetic that decides this in advance

Measured with the repo's own `equity.deflated_sharpe`, no price data involved (so this is not a
holdout peek). Command, reproducible verbatim:

```bash
.venv/bin/python - <<'EOF'
import sys, math; sys.path.insert(0, "src")
from trading_bot.backtest.equity import deflated_sharpe
def required_daily_sr(n_trials, n_obs, skew=0.0, kurt=3.0, target=0.95):
    lo, hi = 0.0, 5.0
    for _ in range(300):
        mid = (lo + hi) / 2
        d = deflated_sharpe(mid, n_trials, n_obs, skew, kurt)
        lo, hi = (mid, hi) if (d is None or d < target) else (lo, mid)
    return hi
for nt in (1, 12, 132, 1000, 3000, 10000):
    r = required_daily_sr(nt, 207)
    print(nt, round(r, 4), round(r * math.sqrt(365), 2))
EOF
```

**Annualised Sharpe required for `dsr > 0.95` on a 207-day holdout** (skew 0, kurtosis 3 — the
post-MEDIUM-5-fix shape Phase 1 delivers):

| `n_trials` charged | daily SR needed | annualised SR needed | what charges this |
|---|---|---|---|
| 1 | 0.1150 | **2.20** | correction off entirely |
| 12 | 0.2337 | **4.47** | v0.2.0's single grid, one fold |
| 132 | 0.3044 | **5.82** | v0.2.0's 12 combos × 11 folds |
| 1 000 | 0.3518 | **6.72** | a small campaign |
| **3 011** | **0.3748** | **7.16** | **this campaign (120 × 25 + 11)** |
| 10 000 | 0.3985 | 7.61 | a larger campaign |

And the holdout-length lever (the *only* honest one — more data, never a weaker test):

| holdout days | holdout start | evolution days | folds | ann. SR needed @3011 | @1 |
|---|---|---|---|---|---|
| 90 | 2026-04-28 | 1006 | 13 | 11.44 | 3.36 |
| 180 | 2026-01-28 | 916 | 12 | 7.72 | 2.36 |
| **207** | **2026-01-01** | **889** | **11** | **7.16** | **2.20** |
| 273 | 2025-10-27 | 823 | 10 | 6.18 | 1.91 |
| 365 | 2025-07-27 | 731 | 9 | 5.31 | 1.65 |

**Re-checked against Phase 1's measured MEDIUM-5 fix** (which lands *before* this phase, so its
effect is already in the numbers the campaign will produce). Phase 1 measured, on v0.2.0's OOS
series: **kurtosis 31.2449 → 15.5429, skew 3.7883 → 1.4034, OOS Sharpe 1.175 → 1.512.** Feeding
the post-fix moments into the same solver:

| moments used | daily SR needed @3011 | ann. SR needed @3011 |
|---|---|---|
| ideal (skew 0, kurt 3) | 0.3749 | 7.16 |
| **post-MEDIUM-5 measured (skew 1.4034, kurt 15.5429)** | **0.3551** | **6.78** |
| pre-fix (skew 3.7883, kurt 31.2449) | 0.2630 | 5.02 |

So the requirement is **6.78–7.16 annualised**, not exactly 7.16 — a ~5% softening, immaterial to
the conclusion. And evaluated *at* the improved observed Sharpe:

| `n_trials` | DSR at ann. Sharpe **1.512** (post-fix moments) |
|---|---|
| 1 | **0.8829** — better than v0.2.0's 0.742, still under 0.95 |
| 132 | 0.0753 |
| **3 011** | **0.0090** |

**Conclusion, unchanged and stated up front: the `dsr` condition is expected to FAIL with
near-certainty.** No crypto system in the surveyed literature (PRD Research Summary) demonstrates
an annualised Sharpe near 7 out of sample; v0.2.0 measured 1.175, and 1.512 after Phase 1's
attribution fix. Under contract §4's cumulative ledger the gate is, for a population-based search
on ~200 days of holdout, **structurally unpassable on that one condition**. Note what the fix
*did* buy: with the correction off entirely, DSR moves 0.742 → 0.883 — genuinely closer to
significance, and evidence that MEDIUM-5 was the right repair — while the honest cumulative figure
stays ≈0.009. That contrast is the cleanest available statement of the problem: **the obstacle is
the size of the search relative to the sample, not the shape of the return distribution.** It must
be reported, not softened, and always *with* the `n_trials=1` companion figure so a reader can
distinguish "no edge" from "unprovable edge".

### The buy-and-hold null can be toothless — and this holdout may be such a window

**Measured by Phase 1**: over v0.2.0's 90-day OOS holdout, the equal-weight basket's Sharpe was
**−1.303**. Buy-and-hold *lost money* on that window. Both of contract §4's new conditions —
`beats_benchmark_return` and `beats_benchmark_sharpe` — therefore **PASS trivially** there, while
the overall verdict still fails on sample adequacy and DSR.

This is a real hole in the protection KNOWN-LIMITATIONS §0 asked for. §0's complaint was that the
gate could bless a strategy **worse than inaction**; in a bear window almost anything beats
inaction, so a `beats_benchmark_*` PASS earned against a losing basket is **a materially weaker
claim** than one earned against a basket that made money. Two of this campaign's seven conditions
could be satisfied for free if 2026-01-01 → 2026-07-27 happens to be a drawdown period for the
majors — which is unknown at planning time and **deliberately not measured here** (computing the
holdout's benchmark now would leak the bar to beat).

Three rules follow, binding on Tasks 9, 10 and 11:

1. **Never print the benchmark bits alone.** Every report of `beats_benchmark_return` /
   `beats_benchmark_sharpe` is accompanied by the **basket's own absolute performance over the
   holdout** — total return, annualised, Sharpe, max drawdown — and by each symbol's. A reader
   must be able to see *what* was beaten without doing arithmetic.
2. **Tag the claim's strength.** The verdict carries a `benchmark_context` label:
   `BENCHMARK_POSITIVE` (basket `ann_return_pct > 0` **and** `sharpe > 0`) or
   `BENCHMARK_NEGATIVE` (either ≤ 0). A `beats_benchmark_*` PASS under `BENCHMARK_NEGATIVE`
   **must not be reported as northstar evidence** and must not appear in any summary sentence
   without the label attached.
3. **Do not add a gate condition requiring a positive benchmark.** That would be moving the gate
   after the fact, which this whole phase exists to prevent — and it would also be wrong on the
   merits: beating a falling market is a legitimate, if weaker, result. **Report the context; let
   the reader judge.** `GATE_CONDITIONS` stays exactly the contract's seven.

The label is descriptive, never dispositive: it changes how a PASS is *described*, never whether
it is a PASS.

### Verdict taxonomy — declared in advance, with the action for each

`campaign --stage report` classifies the run into exactly one of these. The classifier is code
(Task 11), not judgement, and its thresholds are the ones in this table.

**Every verdict below is reported as `<id> / <benchmark_context>`** — e.g. `B1 / BENCHMARK_NEGATIVE`
— per the preceding section. The id is decided by the gate dict alone; the context qualifies how
strong the "beats inaction" part of the claim is. The two are never collapsed into one label.

| ID | Definition (on the 7-condition gate dict + northstar) | Meaning | Recommended next action | Prior probability |
|---|---|---|---|---|
| **A** | all 7 conditions PASS **and** `ann_return_pct > CAMPAIGN_NORTHSTAR_ANN_RETURN` | Northstar met. Hypothesis survived. | Freeze the champion graph and version; do **not** tune further; move to a forward paper period before any capital. The DoF ledger stays attached to the claim forever. **Under `BENCHMARK_NEGATIVE`, the northstar claim is reported as "met in a window where buy-and-hold lost money" — a weaker result that still needs a positive-benchmark window before it is trusted.** | **very low** (needs ann. SR ≈ 6.8–7.2) |
| **A′** | all 7 PASS, `ann_return_pct ≤` northstar | Gate passed, northstar missed. | Completed outcome. Report the gap honestly; the PRD's >50% was always at the optimistic edge of the field. Decide separately whether a ~20–30% honest edge is worth sizing — and note that under `BENCHMARK_NEGATIVE` "beat the basket" contributed little to that decision. | very low |
| **B1** | `dsr` is the **only** FAIL; `beats_benchmark_return` **and** `beats_benchmark_sharpe` PASS; `sample_adequacy` PASS | **The economically interesting "no"** — *if* `BENCHMARK_POSITIVE`: the family beat a basket that made money, out of sample, but cannot be proven significant under honest multiple-testing correction at this sample size. **Under `BENCHMARK_NEGATIVE` this is a much weaker "no": it beat a losing basket, which in a bear window is close to free** (Phase 1 measured basket Sharpe −1.303 on v0.2.0's OOS window). | Report both DSR figures (cumulative and `n_trials=1`) **and the basket's absolute numbers**. Recommended action: **extend the holdout in calendar time** by waiting for real forward data — not by re-running with a lower `n_trials`. Paper-forward the champion; re-gate after ≥1 more year of bars exists. Under `BENCHMARK_NEGATIVE`, the re-gate must specifically cover a window where the majors rose. | **moderate** |
| **B2** | `dsr` FAILs **plus** ≥1 other condition, but `beats_benchmark_*` PASS | Honest "no" with a diagnosed second weakness (usually `per_symbol_expectancy`). | Report per-condition. Action follows the *other* failure: per-symbol failure ⇒ the edge is concentrated, revisit Phase 8 detector selection; drawdown failure ⇒ the equity path is unacceptable regardless of mean. | **most likely** |
| **B3** | `beats_benchmark_return` or `beats_benchmark_sharpe` FAILs | Same verdict as v0.2.0: **loses to inaction**. Note this is the *one* id where `BENCHMARK_NEGATIVE` makes the result **worse**, not weaker: failing to beat a basket that itself lost money is a strong negative signal and must be reported as such. | Abandon this strategy family. The framework and workflow keep their value (PRD honesty clause). Do not tune; §0 says the task at that point is "deciding whether to keep it". | moderate |
| **B4** | `sample_adequacy` FAILs | Data problem, unchanged from v0.2.0 §4. | Record the measured trade rate. **Never** lower `WF_MIN_TRADES` and **never** shrink the holdout (KNOWN-LIMITATIONS §4 forbids both explicitly). The honest responses are: more symbols with genuinely lower correlation, a higher per-symbol trade rate from more detectors, or more calendar time. | **low** — the *raw* floor of 30 clears at 3–5× on a 207-day holdout with Phase 2's set (next section). The *independent-equivalent* floor is the live question and is a reported caveat, never a gate condition |
| **C** | **the only failure mode**: the run did not produce an unambiguous, reproducible verdict | Ambiguous. | Enumerated and individually prevented — see below. | must be ~0 |

**Verdict C is the only outcome that counts as this phase failing.** Its concrete forms, each
with the mechanism that prevents it:

| Ambiguity | Prevention |
|---|---|
| Holdout was peeked at, then "properly" run | `holdout_consumption` row written *before* any holdout bar is read (Task 3); `run_index > 1` stamps the report `NOT A CLEAN HOLDOUT` |
| Champion identity depends on dict/iteration order | deterministic SQL + declared tie-break (Task 5) |
| Numbers in the report cannot be reproduced | every figure carries command + span (Task 10); the campaign persists its own inputs (seed, symbols, spans, `n_trials`) in `state.db` |
| Evolution silently trained on holdout bars | `end_ms = HOLDOUT_START_MS` **and** `assert_no_holdout_overlap()` on every span (Task 2), verified by test |
| Run died at hour 6 and was restarted differently | checkpoint/resume from the `generations` table with the same seed (Task 6) |
| `n_trials` unknown or reset | cumulative ledger in `state.db`, queried, printed, recorded in the report (Task 4) |
| Gate reported as one bit | per-condition dict printed and persisted (Task 7) |

### Sample adequacy is arithmetic, not hope

Measured inputs: v0.2.0 produced **23 OOS trades over 90 days on 3 symbols** (§1), i.e.
**0.085 trades/symbol/day**; KNOWN-LIMITATIONS §4's more pessimistic framing is
**~0.15/day pooled** = 0.050 trades/symbol/day.

**Effective N is now MEASURED, not assumed** (Phase 2, 2026-07-27, on 1095 daily returns over
2023-07-27 → 2026-07-26, zero missing bars — Kish effective N from the full pairwise matrix,
not back-derived from a single `r̄`):

| Symbol set | mean pairwise `r` | Kish `effN` |
|---|---|---|
| 3 core (BTC/ETH/SOL) | **0.7574** | **1.193** — reproduces KNOWN-LIMITATIONS §0b's "≈1.2" |
| best 8-symbol low-correlation subset | — | **1.868** |
| **Phase 2's pre-registered selection** (BTC pinned + lowest-8 by mean `r`) | — | **1.834** ← *this campaign* |

The analytic `effN = n / (1 + (n-1)·r̄)` remains the right intuition (it reproduces the 3-core
figure), but **the campaign must use Phase 2's measured 1.834**, because the broadened set's
pairwise structure is not summarised faithfully by one mean.

The clean identity that follows: with `n` symbols each producing `k` trades,
raw pooled trades `= n·k` but **independent-equivalent trades `≈ k · effN`**. Adding a correlated
symbol multiplies the rows and divides the factor, so rows grow far faster than information —
measured: **3 → 20 symbols buys 6.7× the rows and only ~1.5× the independent information**
(`1.834 / 1.193 = 1.54×`). §0b's trap is therefore **mitigated, not solved**.

| Holdout | symbols | rate/symbol/day | `k` per symbol | raw pooled | `effN` (measured) | indep-equiv | vs floor 30 |
|---|---|---|---|---|---|---|---|
| 90 d | 3 | 0.085 | 7.7 | 23 (measured, v0.2.0) | 1.193 | **9.1** | raw FAIL, indep FAIL |
| 207 d | 3 | 0.085 | 17.6 | 52.9 | 1.193 | 21.0 | raw PASS, indep FAIL |
| **207 d** | **8** | **0.050** | **10.4** | **82.8** | **1.834** | **19.0** | raw PASS, indep FAIL |
| **207 d** | **8** | **0.085** | **17.6** | **140.8** | **1.834** | **32.3** | raw PASS, **indep PASS** |
| 207 d | 9 | 0.085 | 17.6 | 158.4 | 1.834 | 32.3 | raw PASS, indep PASS |

Note the last two rows: **independent-equivalent is `k · effN` and does not move with symbol
count** — only the raw column does. The break-even is therefore a *rate*, and it is arithmetic:

> 30 independent-equivalent trades needs `k ≥ 30 / 1.834 = ` **16.4 trades per symbol**, i.e. a
> measured rate of **≥ 0.0790 trades/symbol/day** over the 207-day holdout.

v0.2.0's realized rate was **0.085** — just above that line. §4's pessimistic framing (0.050) is
well below it.

**Findings to state in the plan, before the run:**

1. **The raw floor of 30 clears comfortably.** The lever that makes it clear is the *longer
   holdout* (207 days instead of 90) — more data, not a weaker test. At 207 days even the legacy
   3 symbols clear it; the broadened set clears it by 3–5×.
2. **The independent-equivalent floor of 30 is now *plausibly reachable*, and the decision rests
   on one measured number: the trade rate.** This is a change from the pre-Phase-2 expectation:
   with `effN = 1.834` rather than 1.193, the requirement falls from ~25.2 to **16.4 trades per
   symbol**. At v0.2.0's realized 0.085/symbol/day it clears (32.3); at §4's pessimistic
   0.050 it does not (19.0). Task 8 measures which world we are in *on the evolution span,
   before the holdout opens*, so this is decided by measurement rather than by hope — and the
   answer is genuinely uncertain rather than foreclosed.
3. **Mitigated is not solved.** Even in the clearing case, `effN = 1.834` on 8 symbols means the
   pooled sample carries roughly the information of **fewer than two independent instruments**;
   all 20 stored symbols are liquid majors and all are BTC beta (contract §0a). DSR's near-iid
   assumption is still violated, just less severely. The report and the limitations document
   must say so in those words, and must **never** present the raw count as evidence of
   independence.
4. The honest response to either outcome is to **report both numbers** — raw (what the gate
   checks, unchanged) and independent-equivalent (the caveat). It is **never** lowering
   `WF_MIN_TRADES` and **never** shrinking the window (KNOWN-LIMITATIONS §4).
5. The projection must be computed from a trade rate **measured on the evolution span only**
   (Task 8), before the holdout is opened. If the projection says the raw floor is unreachable,
   the campaign still runs and reports `sample_adequacy: FAIL` (verdict B4) — it does not
   retune to manufacture a pass.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md` | §4, §11, §12, §7, §8, §6 | BINDING. The gate, the 7 conditions, cumulative trial counting, "the final holdout is never seen by evolution", reserved names, `state.db` conventions, definition of done |
| P0 | `src/trading_bot/backtest/walkforward.py` | 1-412 (all) | The harness this phase **drives and never edits**. Especially 257-269 (signature), 296-308 (span guard / `tune_end`), 310 + 318-355 (fold loop), 357-374 (`median_low` + the on-grid invariant), 376-383 (`n_trials` resolution), 385-400 (the one-shot OOS + gate) |
| P0 | `src/trading_bot/backtest/equity.py` | 95-156, 173-220 | `probabilistic_sharpe`, `expected_max_sharpe`, `deflated_sharpe`, `compute_equity_metrics` — the machinery whose behaviour under `n_trials≈3000` produces this phase's headline finding |
| P0 | `.claude/PRPs/reports/KNOWN-LIMITATIONS.md` | all; esp. §0, §1, §2, §4, §9 | The honest accounting this phase's v0.3.0 equivalent must match in candour. §4 forbids lowering `WF_MIN_TRADES` or shrinking the OOS window |
| P0 | `.claude/PRPs/prds/self-learning-pattern-framework.prd.md` | 18-23, 33-43, 194-197 | Key Hypothesis **and its honesty clause**; Success Metrics table; Phase 9 row and detail |
| P1 | `src/trading_bot/cli.py` | 25-36, 129-144, 199-209, 215-234, 356-368, 396-439 | Subparser registration, the `backtest`/`walkforward` dispatch arm, `_date_arg`, `_fmt`/`_print_metrics`, and `_walkforward_command` — the exact shape `_campaign_command` mirrors |
| P1 | `src/trading_bot/config.py` | 10, 32, 40, 103, 141-151 | `SYMBOLS`, `REGIME_TIMEFRAME`, `REGIME_MIN_BARS` (207-bar warmup ⇒ usable start 2023-07-27), the `FADE_ENABLED` comment style to copy, and the `WF_*` protocol block the new block sits after |
| P1 | `src/trading_bot/data/storage.py` | 22-27, 32, 35-73 | `TIMEFRAME_MS`, the module-level `_db_lock`, and `connect()` — the pattern `data/statestore.py` (Phase 1) mirrors and that this phase's DDL must respect |
| P1 | `scripts/build_performance_chart.py` | 14-16, 18-33, 51-59, 140-171, 174-214, 687-725 | The **live** report generator (`build_review_chart.py` is stale — KNOWN-LIMITATIONS §8). Mirror its `collect()` → `_scorecard()` → `build()` shape, its `sys.path.insert(0, "src")` bootstrap, and its `--out`-required `main()` |
| P1 | `.claude/PRPs/plans/v0.2.0/phase7-validation-protocol-repair-gate.plan.md` | 565-572, 593-601 | The prior one-shot gate plan. Its "Manual Validation — THE GATE, run exactly once" section and Risks table are the discipline bar this plan must clear |
| P2 | `tests/test_backtest.py` | 24-33, 36-46, 49-57, 469-500, 591-637 | Tier-derived constants, the autouse cache-clearing fixture, `make_trade`, `per_symbol_fake_run_factory`, and the CLI exit-code test idiom |
| P2 | `.claude/technical-pattern.md` | reliability table | Only to interpret which detectors Phase 8 shipped; this phase adds none |

## External Documentation

No external research needed — DSR/PSR are already implemented in `backtest/equity.py` and this
phase only calls them. The effective-N formula `n / (1 + (n-1)·r̄)` is the standard design-effect
correction and is already used numerically in KNOWN-LIMITATIONS §0b (r̄ = 0.76 → 1.2), so it is
reproduced, not researched. The PRD's Research Summary already records the relevant literature
point: single-validation walk-forward is insufficient, which is exactly why a never-touched
final holdout exists (`arxiv.org/pdf/2209.05559`, cited in the PRD).

---

## Patterns to Mirror

Every snippet below is copied from the working tree (verified 2026-07-27). Follow these exactly.

### DRIVE_THE_GATE_NEVER_EDIT_IT

```python
# SOURCE: src/trading_bot/backtest/walkforward.py:257-269
def walk_forward_pooled(
    conn,
    symbols: list[str],
    *,
    start_ms: int,
    end_ms: int,
    grid: dict[str, tuple] | None = None,
    train_days: int | None = None,
    test_days: int | None = None,
    oos_days: int | None = None,
    min_trades: int | None = None,
    n_trials: int | None = None,
) -> WalkForwardResult:
```
Phase 3 appended `strategy: StrategyGraph | None = None` (contract §5). Phase 9 calls this
function and **never edits the file**. Every campaign knob maps onto an argument here.

### THE_HOLDOUT_BOUNDARY_IS_ARITHMETIC

```python
# SOURCE: src/trading_bot/backtest/walkforward.py:296-308
    train_ms = (config.WF_TRAIN_DAYS if train_days is None else train_days) * DAY_MS
    test_ms = (config.WF_TEST_DAYS if test_days is None else test_days) * DAY_MS
    oos_ms = (config.WF_OOS_DAYS if oos_days is None else oos_days) * DAY_MS
    ...
    tune_end = end_ms - oos_ms
    if start_ms + train_ms + test_ms > tune_end:
        raise ValueError(
            "span too short: need at least one train+test fold before the OOS holdout"
        )
```
`tune_end = end_ms - oos_days*DAY_MS` is the *only* thing separating the fold sweep from the
holdout. Phase 9 therefore passes `end_ms=HOLDOUT_END_MS` and `oos_days=HOLDOUT_DAYS` and
**asserts `tune_end == HOLDOUT_START_MS`** rather than trusting the arithmetic (Task 7).
`DAY_MS = 86_400_000` at `walkforward.py:44`.

### ON_GRID_INVARIANT_AND_median_low

```python
# SOURCE: src/trading_bot/backtest/walkforward.py:357-374
    # median_low, NOT median: plain median INTERPOLATES on an even number of
    # folds, producing values no fold ever evaluated — [1.25, 2.0] -> 1.625,
    # [48, 96] -> 72 — so the one-shot OOS, the whole point of the protocol,
    # could be run at an unvalidated configuration that is not even on the grid.
    final_combo = {
        axis: statistics.median_low(c[axis] for c in best_combos) for axis in grid
    }
    for axis, value in final_combo.items():
        if value not in grid[axis]:  # pragma: no cover - defensive invariant
            raise AssertionError(
                f"final parameter {axis}={value!r} is not on the grid "
                f"{grid[axis]!r}; the one-shot OOS would be unvalidated"
            )
```
This is why Task 7 passes a **degenerate grid** (one value per axis, the champion's): with a
single value per axis, `median_low` provably returns the champion's value, the on-grid assertion
passes trivially, and the holdout is run at *exactly* the configuration evolution produced — no
re-selection, no interpolation, no off-grid parameters.

### n_trials_IS_AN_ARGUMENT_SO_THE_LEDGER_CAN_OWN_IT

```python
# SOURCE: src/trading_bot/backtest/walkforward.py:376-383
    # n_trials for DSR: every combo evaluated per fold. Resolved HERE, after
    # the fold loop, because it needs len(folds). Deliberately does NOT
    # include the neighbour probes from _positive_neighbour_stats ...
    n_trials_used = n_trials if n_trials is not None else len(combos) * max(1, len(folds))
```
The in-process default is the *fallback*. Phase 9 always passes `n_trials=` explicitly, sourced
from the cumulative ledger in `state.db` (contract §4), so the DSR is charged for every
evaluation the campaign ever performed — thousands, not eleven.

### GATE_FAILS_SAFE_NEVER_RAISES

```python
# SOURCE: src/trading_bot/backtest/walkforward.py:243-254
    if oos_metrics["n_trades"] < min_trades:
        return False
    if oos_equity["sharpe"] is None or oos_equity["sharpe"] < GATE_MIN_SHARPE:
        return False
    if oos_equity["dsr"] is None or oos_equity["dsr"] <= GATE_MIN_DSR:
        return False
    ...
    for exp in per_symbol_expectancy.values():
        if exp is None or exp <= 0:
            return False
    return True
```
Phase 1 changes this to `-> dict[str, bool]` keyed by `GATE_CONDITIONS` (contract §4) while
keeping `passed == all(gate.values())`. Phase 9 consumes the dict and **never reimplements a
threshold**: every printed threshold is read from `walkforward.GATE_MIN_SHARPE` /
`GATE_MIN_DSR` / `GATE_MAX_DRAWDOWN` / `config.WF_MIN_TRADES`, exactly as the live chart already
does (next pattern).

### SCORECARD_ROWS_READ_THRESHOLDS_FROM_SOURCE

```python
# SOURCE: scripts/build_performance_chart.py:174-199
def _scorecard(result):
    """The five gate conditions as (label, value, threshold, ok) rows.

    Every verdict ships with a mark and a word, never colour alone.
    """
    m, eq = result.oos_metrics, result.oos_equity
    per_sym = result.per_symbol_expectancy
    n_pos = sum(1 for v in per_sym.values() if v is not None and v > 0)
    return [
        (
            "Sample adequacy",
            f"{m['n_trades']} trades",
            f"≥ {config.WF_MIN_TRADES}",
            m["n_trades"] >= config.WF_MIN_TRADES,
        ),
        (
            "Sharpe (annualised)",
            _fmt_num(eq["sharpe"]),
            f"≥ {walkforward.GATE_MIN_SHARPE}",
            eq["sharpe"] is not None and eq["sharpe"] >= walkforward.GATE_MIN_SHARPE,
        ),
        ...
```
Task 10's report generator mirrors this **row shape** (`label, value, threshold, ok`) for all
**seven** conditions, and takes `ok` from Phase 1's `result.gate[condition]` rather than
recomputing it — two implementations of a threshold is how a report ends up disagreeing with
the gate.

### REPORT_GENERATOR_SHAPE

```python
# SOURCE: scripts/build_performance_chart.py:18-33, 687-698, 714-725
import argparse
...
sys.path.insert(0, "src")

from trading_bot import config  # noqa: E402
from trading_bot.backtest import walkforward  # noqa: E402
...
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output HTML path")
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--symbols", default=",".join(config.SYMBOLS))
    ...
    data = collect(conn, symbols, start_ms, end_ms)
    conn.close()
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(build(data))
    ...
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### CLI_SUBCOMMAND_AND_DISPATCH

```python
# SOURCE: src/trading_bot/cli.py:129-144
    wf_parser = subparsers.add_parser(
        "walkforward",
        help="Walk-forward validation with one-shot OOS gate (Phase 5 MVP gate)",
    )
    wf_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to validate (repeatable); default is all symbols",
    )
    wf_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD (default: BACKFILL_START)",
    )

# SOURCE: src/trading_bot/cli.py:199-209
    elif args.command in ("backtest", "walkforward"):
        conn = connect(args.db)
        symbols = args.symbol if args.symbol else config.SYMBOLS
        start_ms = args.start if args.start else config.date_to_ms(config.BACKFILL_START)
        end_ms = args.end if args.end else int(time.time() * 1000)
        ...
        conn.close()
        sys.exit(exit_code)
```

### CLI_HANDLER_PRINT_AND_EXIT_CODE

```python
# SOURCE: src/trading_bot/cli.py:396-439 (abridged)
def _walkforward_command(conn, symbols, *, start_ms: int, end_ms: int) -> int:
    """
    Returns:
        0 if the pooled gate passes, 1 otherwise (including a span too
        short to form a single fold).
    """
    try:
        result = walk_forward_pooled(conn, symbols, start_ms=start_ms, end_ms=end_ms)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    ...
    print("per-symbol OOS expectancy:")
    for symbol, exp in result.per_symbol_expectancy.items():
        flag = "OK" if (exp is not None and exp > 0) else "FAIL"
        print(f"  {symbol}: {_fmt(exp, '.4%')}  [{flag}]")
    print(f"GATE: {'PASS' if result.passed else 'FAIL'}")

    return 0 if result.passed else 1
```
`_fmt` (`cli.py:356-358`) returns `"--"` for `None`; `_print_metrics` (`cli.py:361-368`) prints a
`compute_metrics` dict. Reuse both; do not write new formatters in `cli.py`.

### SQLITE_CONNECT_WITH_IDEMPOTENT_DDL_AND_LOCK

```python
# SOURCE: src/trading_bot/data/storage.py:32, 35-73 (abridged)
_db_lock = threading.Lock()


def connect(db_path: str | None = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = config.DB_PATH
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL,
            ...
            PRIMARY KEY (symbol, timeframe, ts)
        )
        """
    )
    conn.commit()
    return conn
```
Contract §6: `data/statestore.py` (Phase 1) mirrors this for `state.db`, and **DDL is owned by
the module that uses the table**. So `campaign.py` owns the `CREATE TABLE IF NOT EXISTS
holdout_consumption` statement and calls it from its own `_ensure_schema(state_conn)`.

### CONFIG_BLOCK_COMMENT_STYLE

```python
# SOURCE: src/trading_bot/config.py:90-103 (abridged) — the tone to copy: name the
# phase, say why the value exists, say whether it is frozen, and record the
# measurement or decision that set it.
# Phase 6: fade re-qualification. The ranging sleeve is kept behind an
# explicit switch so a DROP verdict is a recorded decision rather than a code
# deletion — the method stays tested and a future re-test costs nothing.
# ...
# DROPPED per .claude/PRPs/reports/fade-requalification.md (2026-07-27): measured
# on the tuning span ... pooled expectancy -0.3883% (n=297) ...
FADE_ENABLED = False
```

### TEST_TIER_CONSTANTS_AND_CACHE_ISOLATION

```python
# SOURCE: tests/test_backtest.py:24-33
SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded: these fixtures follow config forever, so a
# future tier shift cannot leave the tests on the old timeframes while
# production moves (the classic way a tier change passes CI while being wrong).
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000

# SOURCE: tests/test_backtest.py:36-46
@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    """Clear the engine's indicator memo around every test. ..."""
    engine.clear_caches()
    yield
    engine.clear_caches()
```

### TEST_STUB_AND_CLI_EXIT_CODE_IDIOM

```python
# SOURCE: tests/test_backtest.py:469-495 (abridged)
def per_symbol_fake_run_factory(peak_exp=0.01, n_trades=8, bad_symbol=None):
    """... Trade exit_ts are spread evenly across [start_ms, end_ms) ... so
    compute_equity_metrics' calendar-day bucketing ... actually sees these
    trades inside the queried span — required for Sharpe/DSR to be
    non-trivially computed on the OOS window instead of silently None."""
    def fake_run(conn, symbol, *, start_ms=None, end_ms=None, params=None, **kw):
        ...
    return fake_run

# SOURCE: tests/test_backtest.py:619-626
        monkeypatch.setattr(cli, "walk_forward_pooled", lambda conn, syms, **kw: make_result(True))
        assert _walkforward_command(None, [SYMBOL], start_ms=0, end_ms=DAY_MS) == 0
        assert "GATE: PASS" in capsys.readouterr().out
```
`test_campaign.py` monkeypatches `campaign.walk_forward_pooled` and
`campaign.run_campaign` (Phase 6's runner, imported into `campaign.py`'s namespace) the same
way — at the name the module under test looks it up on, never inside `walkforward.py` or
`evolution/`.

---

## Stated Assumptions (interfaces this phase consumes)

This phase depends on **everything**. Where the contract fixes an interface, it is quoted; where
it does not, the assumption is stated with a **guaranteed fallback** so implementation never
blocks. Assumptions are numbered so the implementer can check them off in one pass.

| # | Assumption | Contract basis | Fallback if the name differs |
|---|---|---|---|
| **A1** | `data/statestore.connect(db_path=None) -> sqlite3.Connection`, default `config.STATE_DB_PATH`, WAL, module `_db_lock` | §6 — verbatim | none needed; §6 fixes it |
| **A2** | `WalkForwardResult` has `gate: dict[str, bool]`, `benchmark: BenchmarkResult`, `n_trials_used: int`, `passed: bool` (appended, never reordered); `walkforward.GATE_CONDITIONS` is the 7-tuple | §4 — verbatim | if `gate` is absent, Phase 1 is not done: **stop**, do not synthesise a gate dict |
| **A3** | Trial ledger: table `trial_ledger(campaign, graph_hash, params_hash, span, ts)` in `state.db`, owned by `backtest/trials.py` | §4.2, §6 table list | **Use SQL** — `SELECT COUNT(*) FROM trial_ledger WHERE campaign = ?` is guaranteed by §6's column list. Prefer a helper if `trials.py` exposes one; **never create a second ledger** |
| **A4** | Phase 6 exposes a resumable campaign runner in `evolution/runner.py` that accepts a campaign id, symbols, `start_ms`/`end_ms`, population size, generations, seed, and a state connection | §2 (`evolution/runner.py`), §6 (`campaigns`, `generations`, `population_members`) | Read its actual signature and adapt **at the single call site** in `campaign.py`. Do not edit `evolution/` |
| **A5** | Champion identification is possible from `population_members(member_id, generation, graph_hash, fitness, gate_verdicts)` joined to `strategy_versions(version_id, parent, graph_json, created_ts, provenance)` | §6 table list | This SQL path is the guaranteed one **by design** — Task 5 uses it rather than a Phase 6 helper, so champion selection is deterministic and inspectable |
| **A6** | `framework/graph.py` provides `StrategyGraph`, `from_dict`, `to_dict`, `SCHEMA_VERSION`; `walk_forward_pooled(..., strategy=graph)` routes to `framework/execute.run_graph_backtest` | §2, §5 | none needed; §5 fixes it |
| **A7** | `backtest/benchmark.buy_and_hold(conn, symbols, *, start_ms, end_ms) -> BenchmarkResult` with `.basket` and `.per_symbol` dicts carrying `ann_return_pct`, `sharpe`, `max_drawdown_pct`, `total_return`, `n_days` | §4 — verbatim | none needed. Phase 9 reads `result.benchmark` from the gate run; it calls `buy_and_hold` directly **only** for the full-span (evolution+holdout) diagnostic |
| **A8** | Phase 2 published `config.RESEARCH_SYMBOLS` (the gap-verified, correlation-selected subset) and `data/correlation.py` with a pairwise matrix + effective-N | §2, §7 prefix table | If `RESEARCH_SYMBOLS` is absent, Phase 2 is not done: **stop**. Do not fall back to `config.SYMBOLS` silently — the broadened universe is the phase's premise |
| **A9** | Phase 1's `PNL_ATTRIBUTION_*` default spreads trade P&L across holding days (MEDIUM-5 fixed), with the old exit-day behaviour reachable by keyword | §4 "MEDIUM-5" | Report the measured kurtosis of the holdout daily-return series either way; if attribution is still exit-day, say so in the report — it changes what Sharpe/DSR mean (§3) |
| **A10** | Phase 5's `config.TARGET_ANN_RETURN` exists (its reserved name) and equals the 0.50 northstar | §7 prefix table | `CAMPAIGN_NORTHSTAR_ANN_RETURN` **references** it (`= TARGET_ANN_RETURN`) rather than redefining 0.50, so there is one source of truth. If Phase 5 omitted it, define the literal with a comment naming the omission |

### Contract / PRD discrepancies, resolved in the contract's favour

Per the brief: where the PRD and the contract disagree, follow the contract and note it.

1. **"All 5 conditions" vs seven.** The PRD's Success Metrics row says "All 5 conditions pass"
   (`prd.md:39`). Contract §4 defines **seven** `GATE_CONDITIONS`, adding
   `beats_benchmark_return` and `beats_benchmark_sharpe`. **Follow the contract: seven.** The
   PRD's own adjacent row ("Beats buy-and-hold null") asks for exactly the two additions, so
   this is a counting slip, not a design conflict.
2. **The gate contains no return threshold.** The PRD's headline metric is ">50% OOS annualised"
   (`prd.md:37`), but no member of `GATE_CONDITIONS` tests return magnitude. **Follow the
   contract**: the gate is the 7 conditions; the northstar is reported *separately* as
   `ann_return_pct` vs `CAMPAIGN_NORTHSTAR_ANN_RETURN`. This is why the verdict taxonomy needs
   both **A** (gate + northstar) and **A′** (gate only) — a passing gate does not imply the
   northstar, and the report must never conflate them.
3. **Phase 2 is selection, not backfill.** The PRD describes Phase 2 as backfilling; contract
   §0a corrects this (20 symbols already stored). Phase 9 consumes `RESEARCH_SYMBOLS` and does
   **no** data acquisition. If `fapi.binance.com` matters at all here it is only to top up the
   last few bars before the run, which is `backfill`'s job, not this phase's.
4. **§2's layout table has no Phase 9 module.** The canonical layout (contract §2) assigns no
   `src/` file to Phase 9, yet the phase must hold real logic (holdout enforcement, champion
   selection, checkpoint/resume) that does not belong in a CLI handler. This plan therefore
   claims **exactly one** new module, `src/trading_bot/campaign.py`, at top level beside
   `cli.py`/`config.py`, and claims no other path, no new package, and no name reserved by
   another phase. Recorded here as a documented micro-extension of §2 rather than a silent one.

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/campaign.py` | **CREATE** | The locked protocol as code: spans, holdout enforcement + consumption ledger, cumulative trial count, champion selection, degenerate-grid construction, evolution-stage driver with checkpoint/resume, one-shot holdout stage, diagnostics, verdict classifier. See discrepancy note 4 |
| `src/trading_bot/config.py` | UPDATE | Append the Phase 9 block only: `HOLDOUT_*` and `CAMPAIGN_*` (contract §7 reserves these prefixes to Phase 9). No existing constant is modified |
| `src/trading_bot/cli.py` | UPDATE | Register the `campaign` subparser in phase order and add `_campaign_command` (contract §7 reserves both names to Phase 9) |
| `tests/test_campaign.py` | **CREATE** | The reserved test file (contract §8) |
| `scripts/build_campaign_report.py` | **CREATE** | The report generator. `scripts/` is outside §2's layout table; `build_performance_chart.py` is the live prior art it mirrors |
| `.claude/PRPs/reports/phase9-campaign-verdict.md` | GENERATED | Written by the generator above, once, from persisted campaign state. Every figure carries its command + span (contract §12.5) |
| `.claude/PRPs/reports/KNOWN-LIMITATIONS-v0.3.0.md` | GENERATED (hand-written) | The v0.3.0 equivalent of v0.2.0's honest accounting, written **knowing** the verdict. A real deliverable of this phase, not an afterthought |

## NOT Building

- **`src/trading_bot/backtest/walkforward.py`** — Phase 1 owns the gate extension, Phase 3 added
  `strategy=`. Phase 9 **drives** it and does not touch it. No new gate condition, no threshold
  change, no `min_trades` override, no new `oos_days` semantics.
- **`src/trading_bot/evolution/**`** — Phase 6 owns population, mutation, tournament, oracle and
  runner. Phase 9 calls the runner and reads its tables. If the runner cannot resume, that is a
  Phase 6 defect to report, not a Phase 9 reimplementation.
- **Lowering `WF_MIN_TRADES` or shrinking the holdout.** Explicitly forbidden by
  KNOWN-LIMITATIONS §4. Neither appears as a config value, a CLI flag, or a fallback anywhere in
  this plan.
- **Any new fitness path.** Contract §4.1: fitness comes only from `walk_forward_pooled`.
  `scripts/bruteforce/core.score` is never imported.
- **New detectors, indicators, plug-ins, or mutators** — Phases 4, 6 and 8. The campaign runs
  what is registered; it adds nothing to the registry.
- **`scripts/build_performance_chart.py`** — left untouched. It runs its own
  `walk_forward_pooled` without a graph (`build_performance_chart.py:140-156`), so it cannot
  depict a graph champion; teaching it `--graph` would be an edit to a shared artifact for a
  cosmetic gain. `build_review_chart.py` stays stale (KNOWN-LIMITATIONS §8).
- **Backfill / network access.** No `ccxt` call, no `fapi.binance.com` ping. The campaign runs on
  stored bars; a stale tail is reported, never fetched mid-campaign.
- **Position sizing, multi-position margin accounting, live execution.** Contract §10 items 7 and
  the inherited v0.2.0 boundaries. The reported drawdown is an **unsized** figure and must be
  labelled as such (KNOWN-LIMITATIONS §7).
- **A second campaign to "confirm" the first.** A re-run is a new campaign with a new id, a new
  cumulative trial count, and a report stamped `NOT A CLEAN HOLDOUT`. There is no "best of two".
- **Cloud / distributed execution.** PRD risk table: Mac-only, overnight, cloud deferred.
- **HTML charting of the campaign.** Markdown report only (see NOT Building above re
  `build_performance_chart.py`). The verdict is a table of measured numbers with provenance; it
  does not need SVG to be honest.

---

## Step-by-Step Tasks

### Task 1: Pre-register the protocol in `config.py`

- **ACTION**: Append one clearly-headed Phase 9 block at the **end** of `config.py` (contract §7:
  blocks appended in phase order, existing constants untouched).
- **IMPLEMENT**:
  ```python
  # ---------------------------------------------------------------------------
  # Phase 9 (v0.3.0): the walk-forward CAMPAIGN protocol. Every value in this
  # block is PRE-REGISTERED: it is committed before the campaign runs so the
  # result cannot be rationalised afterwards. Changing any of them after a
  # holdout has been consumed does not produce a better verdict, it produces a
  # different (and no longer clean) experiment — see
  # campaign.holdout_is_consumed() and the --force-holdout-rerun audit path.
  # ---------------------------------------------------------------------------

  # THE HOLDOUT. Never seen by any evolution generation (contract §4.4).
  # Epoch ms, UTC, candle OPEN times, both from date_to_ms():
  #   HOLDOUT_START_MS = date_to_ms("2026-01-01")  ->  1767225600000
  #   HOLDOUT_END_MS   = date_to_ms("2026-07-27")  ->  1785110400000
  #
  # HOLDOUT_END_MS IS AN EXCLUSIVE UPPER BOUND. Read it any other way and the
  # verdict picks up a partial bar. Measured 2026-07-27:
  #   sqlite3 data/ohlcv.db "select max(ts) from ohlcv where timeframe='1d'"
  #   -> 1785110400000 == 2026-07-27T00:00:00Z
  # That bar is TODAY'S and is STILL FORMING; storage.find_gaps excludes
  # forming candles by construction (storage.py:164-166), so it is not a gap.
  #   * last CLOSED 1d bar    : opens 2026-07-26T00:00:00Z (1785024000000)
  #   * HOLDOUT_END_MS        : 1785110400000, EXCLUSIVE -> the forming bar
  #                             contributes nothing
  # (This supersedes shared-contract §0's "2026-07-25", which was wrong;
  # re-measured and corrected in the contract after Phase 9's report.)
  #
  # Consequence for every consumer: pass HOLDOUT_END_MS as the exclusive
  # end_ms of a half-open span [start, end). storage.load_candles' bounds are
  # INCLUSIVE (contract §1), so any direct load_candles call for the holdout
  # must use HOLDOUT_END_MS - 1, never HOLDOUT_END_MS.
  HOLDOUT_START_MS = 1767225600000  # 2026-01-01T00:00:00Z, INCLUSIVE
  HOLDOUT_END_MS = 1785110400000    # 2026-07-27T00:00:00Z, EXCLUSIVE
  # 207 days. Asserted, not assumed:
  #   HOLDOUT_END_MS - HOLDOUT_DAYS * 86_400_000 == HOLDOUT_START_MS
  # Length chosen on a MEASURED basis, not by taste: lengthening the holdout is
  # the only lever that lowers the Sharpe required to clear DSR at a given
  # trial count without weakening the test (see the plan's required-Sharpe
  # table: 90d needs ann. SR 11.44 at n_trials=3011, 207d needs 7.16). It is
  # bounded below by leaving >= 889 days (11 folds at WF_TRAIN_DAYS=180 /
  # WF_TEST_DAYS=60) for evolution, because v0.2.0's train windows were
  # themselves undersampled (KNOWN-LIMITATIONS §2: 4 of 13 folds fell back to
  # defaults). 207 days is where those two pressures meet.
  HOLDOUT_DAYS = 207
  # A committed tripwire: campaign.py refuses to run the holdout stage when
  # this is False. Flipping it is a recorded decision, not a convenience.
  HOLDOUT_LOCKED = True

  # Evolution span start: 2023-07-27 == date_to_ms("2023-07-27") == 1690416000000.
  # NOT BACKFILL_START: REGIME_MIN_BARS = 207 daily bars of warmup means the
  # first non-"uncertain" regime label lands ~2023-07-27 (see the comment at
  # REGIME_MIN_BARS above), so earlier windows produce zero trades and would
  # silently dilute every fold.
  CAMPAIGN_EVOLVE_START_MS = 1690416000000

  # The symbol set is FROZEN before the run and recorded with the campaign.
  # Sourced from Phase 2's measured, gap-verified, low-correlation selection —
  # never from SYMBOLS (3 correlated majors, effective N ~= 1.2, §0b).
  CAMPAIGN_SYMBOLS: tuple[str, ...] = RESEARCH_SYMBOLS

  # Population and generations. Product = 3000 oracle evaluations, which is
  # what the DSR will be charged (plus the final gate's own 11). Sized by
  # BUDGET, not by ambition: campaign --stage probe measures T_eval on stored
  # bars and the run aborts if population * generations * T_eval exceeds
  # CAMPAIGN_WALL_CLOCK_BUDGET_HOURS. The declared response to an over-budget
  # probe is to reduce GENERATIONS (never to shorten the holdout, shrink the
  # fold windows, or lower WF_MIN_TRADES).
  CAMPAIGN_POPULATION_SIZE = 120
  CAMPAIGN_GENERATIONS = 25
  CAMPAIGN_WALL_CLOCK_BUDGET_HOURS = 10.0
  # Stopping rule, declared in advance so "stop when it looks good" is
  # impossible: stop at the FIRST of (a) CAMPAIGN_GENERATIONS reached,
  # (b) budget exhausted, (c) no improvement in best fitness for
  # CAMPAIGN_PATIENCE_GENERATIONS consecutive generations.
  CAMPAIGN_PATIENCE_GENERATIONS = 5
  # Reproducibility: one seed for the whole campaign, recorded in state.db.
  CAMPAIGN_SEED = 20260727
  CAMPAIGN_CHECKPOINT_EVERY = 1  # generations between checkpoints; 1 = every one

  # Reported beside the gate, NOT a gate condition: contract §4's seven
  # GATE_CONDITIONS contain no return threshold, while the PRD's northstar is
  # >50% annualised OOS. Referencing Phase 5's constant keeps one source of
  # truth rather than a second literal 0.50.
  CAMPAIGN_NORTHSTAR_ANN_RETURN = TARGET_ANN_RETURN

  # Where the verdict report and the v0.3.0 limitations document are written.
  CAMPAIGN_REPORT_DIR = ".claude/PRPs/reports"
  ```
- **MIRROR**: `CONFIG_BLOCK_COMMENT_STYLE` — `config.py:90-103`'s tone (name the phase, say why,
  record the measurement). The `WF_*` block at `config.py:141-151` is the structural precedent.
- **IMPORTS**: none. `RESEARCH_SYMBOLS` (Phase 2) and `TARGET_ANN_RETURN` (Phase 5) are
  module-level names defined earlier in the same file — phase-ordered blocks make this legal.
- **GOTCHA**: do **not** add `HOLDOUT_START`/`HOLDOUT_END` date *strings*. Contract §6: "Never
  store formatted date strings." The dates live in comments; the constants are epoch ms.
- **VALIDATE**:
  ```bash
  .venv/bin/python -c "from trading_bot import config as c; \
    assert c.HOLDOUT_END_MS - c.HOLDOUT_DAYS*86_400_000 == c.HOLDOUT_START_MS; \
    assert c.CAMPAIGN_EVOLVE_START_MS < c.HOLDOUT_START_MS; \
    print(len(c.CAMPAIGN_SYMBOLS), 'symbols', c.HOLDOUT_DAYS, 'holdout days')"
  ```

### Task 2: `campaign.py` — spans and the structural holdout barrier

- **ACTION**: Create `src/trading_bot/campaign.py` with the module docstring that states the
  locked protocol, the error type, and the span helpers.
- **IMPLEMENT**:
  ```python
  """
  The v0.3.0 walk-forward CAMPAIGN — Phase 9, the verdict on the key hypothesis.

  Protocol (LOCKED before the run; see config.py's Phase 9 block):
    1. EVOLUTION runs on [CAMPAIGN_EVOLVE_START_MS, HOLDOUT_START_MS). Phase 6's
       runner carves its own internal validation OOS inside that span, which it
       sees thousands of times — that window is NOT evidence and is labelled so
       in the report.
    2. THE HOLDOUT is [HOLDOUT_START_MS, HOLDOUT_END_MS) and is never seen by
       any generation. This is enforced twice: by construction (the runner is
       handed end_ms = HOLDOUT_START_MS) and by assertion
       (assert_no_holdout_overlap on every span this module builds or forwards).
    3. THE CHAMPION is chosen by a deterministic query with a declared
       tie-break (select_champion) and evaluated on the holdout EXACTLY ONCE,
       through walkforward.walk_forward_pooled with a DEGENERATE grid, so no
       parameter is re-selected on holdout data.
    4. DSR is charged the CUMULATIVE trial count for the campaign, read from
       state.db's trial_ledger (contract §4). This is expected to make the gate
       unpassable on the dsr condition; that is the finding, not a bug.
    5. The holdout is CONSUMED-ONCE: a holdout_consumption row is written
       BEFORE any holdout bar is read, so killing the process still burns the
       peek. A second run requires an explicit override that is itself logged.

  This module DRIVES walkforward.py and evolution/ and edits neither.
  """

  import json
  import logging
  import sqlite3
  import statistics
  import time
  from dataclasses import dataclass

  from trading_bot import config
  from trading_bot.backtest import walkforward
  from trading_bot.backtest.metrics import compute_metrics
  from trading_bot.backtest.walkforward import DAY_MS, walk_forward_pooled

  logger = logging.getLogger("trading_bot")


  class CampaignError(RuntimeError):
      """Any protocol violation that must abort the campaign."""


  class HoldoutViolation(CampaignError):
      """A span that would let evolution touch the final holdout."""


  def holdout_span() -> tuple[int, int]:
      """(start_ms, end_ms) of the final holdout. end_ms is EXCLUSIVE."""
      start, end = config.HOLDOUT_START_MS, config.HOLDOUT_END_MS
      if end - config.HOLDOUT_DAYS * DAY_MS != start:
          raise CampaignError(
              f"HOLDOUT_DAYS={config.HOLDOUT_DAYS} disagrees with "
              f"[{start}, {end}); the declared span and its length must match "
              f"exactly or walk_forward_pooled's tune_end lands off-boundary"
          )
      return start, end


  def evolution_span() -> tuple[int, int]:
      """(start_ms, end_ms) evolution may see. end_ms IS the holdout start."""
      start = config.CAMPAIGN_EVOLVE_START_MS
      end, _ = holdout_span()
      if start >= end:
          raise CampaignError("evolution span is empty; check CAMPAIGN_EVOLVE_START_MS")
      return start, end


  def assert_no_holdout_overlap(start_ms: int, end_ms: int, *, what: str) -> None:
      """Raise unless [start_ms, end_ms) is entirely before the holdout.

      This is the mechanism, not the convention. Called on every span this
      module builds and on every span it forwards to Phase 6's runner, so a
      config edit, a mutator that widens a window, or a hand-typed --start
      cannot quietly reach holdout bars. Any half-open interval that
      intersects [HOLDOUT_START_MS, HOLDOUT_END_MS) is a violation, including
      one that merely straddles the boundary.
      """
      h_start, h_end = holdout_span()
      if start_ms < h_end and end_ms > h_start:
          raise HoldoutViolation(
              f"{what} span [{start_ms}, {end_ms}) intersects the holdout "
              f"[{h_start}, {h_end}); evolution must end at or before "
              f"{h_start}. Refusing to run."
          )
  ```
- **MIRROR**: `walkforward.py:42` (`logger = logging.getLogger("trading_bot")`),
  `walkforward.py:44` (`DAY_MS`), and the frozen-dataclass style at `walkforward.py:83-95`.
- **GOTCHA**: the overlap test must be the half-open-interval test above, not
  `end_ms <= h_start`. A candidate span of `[h_start, h_start)` is empty and harmless, but
  `[h_start - 1, h_start + 1)` reaches one holdout bar and must raise. Test both.
- **VALIDATE**: `python -m py_compile src/trading_bot/campaign.py` and
  `pytest tests/test_campaign.py::TestSpans -v`.

### Task 3: The consume-once holdout ledger

- **ACTION**: Add the `holdout_consumption` table (DDL owned by this module, contract §6) plus
  its read/write helpers.
- **IMPLEMENT**:
  ```python
  _HOLDOUT_DDL = """
  CREATE TABLE IF NOT EXISTS holdout_consumption (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      campaign_id TEXT NOT NULL,
      kind TEXT NOT NULL,              -- 'consume' | 'violation'
      run_index INTEGER NOT NULL,      -- 1 = the clean run; >1 = an override
      holdout_start_ms INTEGER NOT NULL,
      holdout_end_ms INTEGER NOT NULL,
      graph_hash TEXT,
      strategy_version TEXT,
      n_trials_charged INTEGER,
      opened_ts INTEGER NOT NULL,      -- epoch ms, written BEFORE the run
      completed_ts INTEGER,            -- NULL until the run finishes
      gate_json TEXT,                  -- the 7-condition dict, once known
      override_reason TEXT             -- required when run_index > 1
  )
  """


  def _ensure_schema(state_conn) -> None:
      """Idempotent DDL, owned here (contract §6: the module that uses the
      table owns its CREATE TABLE IF NOT EXISTS, not a central migration)."""
      state_conn.execute(_HOLDOUT_DDL)
      state_conn.commit()


  def holdout_consumption_rows(state_conn, *, kind="consume") -> list[sqlite3.Row]:
      """Every recorded touch of the CURRENTLY DECLARED holdout span, oldest first.

      Keyed on the span, not the campaign: a new campaign id does not grant a
      fresh look at the same bars.
      """


  def holdout_is_consumed(state_conn) -> bool:
      """True when the declared holdout span already has a 'consume' row."""


  def open_holdout(state_conn, *, campaign_id, graph_hash, strategy_version,
                   n_trials, override_reason=None) -> int:
      """Record the consumption BEFORE the holdout is read; return run_index.

      Write-then-run is deliberate: if the process dies mid-evaluation the
      holdout is still burned. Otherwise `kill -9` after a glimpse of the
      numbers would be a free peek, and the one-shot guarantee would be
      advisory. Raises CampaignError when the span is already consumed and no
      override_reason is supplied, and when config.HOLDOUT_LOCKED is False.
      """


  def close_holdout(state_conn, run_id: int, *, gate: dict[str, bool]) -> None:
      """Attach the gate verdict and completed_ts to the open row."""


  def record_violation(state_conn, *, campaign_id, start_ms, end_ms, detail) -> None:
      """Audit a HoldoutViolation attempt (kind='violation', run_index=0).

      Violations do NOT consume the holdout — no bar was read — but they are
      recorded, because 'the code tried to train on the holdout once' is
      exactly the kind of thing a report must not be able to omit.
      """
  ```
- **MIRROR**: `SQLITE_CONNECT_WITH_IDEMPOTENT_DDL_AND_LOCK` (`storage.py:32, 35-73`). All
  timestamps epoch ms UTC (contract §6). Use `statestore.connect()` (A1) — never open
  `state.db` directly, and never write to `ohlcv.db`.
- **GOTCHA #1**: `holdout_is_consumed` filters `kind='consume'`; a `violation` row must not
  block a legitimate first run.
- **GOTCHA #2**: `open_holdout` computes `run_index = 1 + count(kind='consume' for this span)`
  and **requires** a non-empty `override_reason` whenever that is > 1. The reason is stored, and
  Task 10 stamps any report with `run_index > 1` as `NOT A CLEAN HOLDOUT` in its header — the
  override is possible, never invisible.
- **GOTCHA #3**: do not add `data/state.db*` handling here; `.gitignore` coverage for the WAL
  sidecars is Phase 1's job (contract §6). Verify it during Task 12 and report it if missing —
  today's `.gitignore` has `data/*.db`, which does **not** match `state.db-wal`.
- **VALIDATE**: `pytest tests/test_campaign.py::TestHoldoutLedger -v`.

### Task 4: Cumulative trial accounting

- **ACTION**: Read the campaign's cumulative trial count from `state.db` and pass it to the gate.
- **IMPLEMENT**:
  ```python
  def cumulative_trials(state_conn, campaign_id: str) -> int:
      """Every (graph, params) evaluation the oracle ever performed for this
      campaign, across generations and across process restarts (contract §4).

      Guaranteed path is SQL over Phase 1's trial_ledger, whose columns §6
      fixes: SELECT COUNT(*) FROM trial_ledger WHERE campaign = ?. Prefer a
      helper from backtest/trials.py if one exists; NEVER count in-process and
      NEVER create a second ledger — a ledger that resets on restart is worse
      than none, because it looks authoritative.
      """


  def charge_gate_trials(state_conn, *, campaign_id, graph_hash, grid, n_folds) -> int:
      """Record the final holdout run's OWN evaluations in the ledger too.

      The one-shot gate run is itself len(_combos(grid)) * n_folds evaluations
      on the training span plus one holdout evaluation. With the degenerate
      grid from Task 5 that is 1 * n_folds + 1 = 12 — small, but omitting it
      would make the ledger a lie about its own last step.
      """
  ```
- **MIRROR**: `n_trials_IS_AN_ARGUMENT_SO_THE_LEDGER_CAN_OWN_IT` (`walkforward.py:376-383`) —
  always pass `n_trials=` explicitly so the in-process default never applies.
- **GOTCHA**: `walkforward.py` counts `len(combos) * max(1, len(folds))` when `n_trials is None`.
  If Phase 9 ever forgot to pass `n_trials`, the DSR would silently be charged **12** instead of
  **~3011**, flattering the result by ~5 Sharpe points of required threshold (see the
  required-Sharpe table). Task 11's test asserts `result.n_trials_used == cumulative_trials(...)`
  so this cannot regress.
- **VALIDATE**: `pytest tests/test_campaign.py::TestTrialAccounting -v`.

### Task 5: Deterministic champion selection and the degenerate grid

- **ACTION**: Select the champion by SQL with a declared tie-break; build the one-value-per-axis
  grid that pins the holdout run to it.
- **IMPLEMENT**:
  ```python
  @dataclass(frozen=True)
  class ChampionRef:
      campaign_id: str
      strategy_version: str
      graph_hash: str
      generation: int
      member_id: str
      fitness: float
      graph_json: str


  def select_champion(state_conn, campaign_id: str) -> ChampionRef:
      """The campaign's champion, chosen deterministically.

      Declared tie-break, in this order (committed BEFORE the run so a tie
      cannot be resolved by whoever is looking at the numbers):
        1. highest fitness (the gate oracle's score — contract §4.1: there is
           no other scoring function);
        2. then EARLIEST generation — an equally-fit earlier candidate has
           survived more selection rounds and consumed fewer degrees of
           freedom;
        3. then lexicographically smallest graph_hash — arbitrary but total,
           so the result never depends on SQLite row order or dict iteration.

      Reads Phase 6's population_members joined to Phase 5's
      strategy_versions for the graph JSON (contract §6 fixes both schemas).
      Raises CampaignError when the campaign has no members — a campaign with
      no champion is verdict C (ambiguous), never a silent fallback to the
      seed strategy.
      """


  def frozen_grid(graph, grid_axes: dict[str, tuple]) -> dict[str, tuple]:
      """A DEGENERATE grid: exactly one value per axis, the champion's.

      walk_forward_pooled takes final parameters as the per-axis median_low of
      fold winners (walkforward.py:357-373). With one value per axis,
      median_low provably returns that value, the on-grid assertion at
      walkforward.py:368-373 passes trivially, and the holdout is evaluated at
      EXACTLY the configuration evolution produced.

      Rejected alternative: pass DEFAULT_GRID and let the folds re-tune on the
      training span. That would run the holdout at parameters evolution never
      selected, consume additional degrees of freedom for the verdict itself,
      and make "what was tested?" unanswerable.
      """
  ```
- **MIRROR**: `ON_GRID_INVARIANT_AND_median_low` (`walkforward.py:357-374`) and
  `_combos`/`_neighbors` (`walkforward.py:117-130`), which are generic over any grid dict and
  need no change.
- **GOTCHA #1**: with a single value per axis, `_neighbors` (`walkforward.py:122-130`) returns
  `[]`, so `_positive_neighbour_stats` returns `(None, None)` (`walkforward.py:218-219`) and every
  fold reports `pos_neighbours=--`. That is **expected**, not a defect: robustness-around-the-
  winner was measured during evolution, and re-probing neighbours here would evaluate
  configurations the champion is not, adding trials for no verdict value. Say so in the report so
  a reader does not read `--` as a missing measurement.
- **GOTCHA #2**: if the champion graph carries parameters that are *not* `walkforward` grid axes
  (the normal case — a `StrategyGraph` carries its own node params, contract §5), then
  `grid_axes` may legitimately be a single dummy axis pinned to its config default, because the
  graph itself is passed via `strategy=`. Keep exactly one axis so the fold loop and `median_low`
  have something well-defined to chew on, and document which axis it is in the report.
- **VALIDATE**: `pytest tests/test_campaign.py::TestChampionSelection -v`.

### Task 6: The evolution stage — budget, checkpointing, resume

- **ACTION**: Drive Phase 6's runner over the evolution span with a measured wall-clock budget
  and crash-resumable progress.
- **IMPLEMENT**:
  ```python
  @dataclass(frozen=True)
  class BudgetProbe:
      seconds_per_eval: float
      n_probe_evals: int
      projected_hours: float
      within_budget: bool


  def probe_budget(conn, state_conn, *, campaign_id, symbols, n_evals=3) -> BudgetProbe:
      """MEASURE the cost of one oracle evaluation; never estimate it.

      Runs n_evals gate evaluations of the seed strategy on the TRAINING span
      only (assert_no_holdout_overlap first), times them, and projects
      CAMPAIGN_POPULATION_SIZE * CAMPAIGN_GENERATIONS * seconds_per_eval.
      These probe evaluations are real evaluations and ARE charged to the
      ledger — a measurement that dodges the ledger is exactly the
      side-channel contract §4.2 forbids.

      Declared response to within_budget=False: reduce CAMPAIGN_GENERATIONS
      (and re-commit config) — never shorten the holdout, never shrink the
      fold windows, never lower WF_MIN_TRADES.
      """


  def run_evolution_stage(conn, state_conn, *, campaign_id, resume=True) -> int:
      """Run (or resume) the evolution stage. Returns the last completed generation.

      Wall-clock: CAMPAIGN_WALL_CLOCK_BUDGET_HOURS, checked between
      generations, so an overnight run stops cleanly at a checkpoint rather
      than being killed mid-generation.

      Stopping rule, evaluated in this fixed order (declared in config, so
      "stop when the numbers look good" is not reachable):
        1. generation == CAMPAIGN_GENERATIONS  -> COMPLETE
        2. elapsed >= budget                   -> BUDGET_EXHAUSTED
        3. best fitness unimproved for CAMPAIGN_PATIENCE_GENERATIONS
                                               -> CONVERGED
      All three are completed evolution stages; the holdout stage runs after
      any of them, and the report records WHICH one fired.

      Resume: state lives in Phase 6's `generations` / `population_members`
      tables in state.db (contract §6), so a campaign that dies at hour 6
      continues at the last checkpointed generation with the SAME
      CAMPAIGN_SEED, rather than restarting. resume=False starts a NEW
      campaign id; it never overwrites an existing campaign's rows.
      """
  ```
  Every span forwarded to Phase 6's runner passes through
  `assert_no_holdout_overlap(start, end, what="evolution")` first, and `end` is
  `evolution_span()[1]`, i.e. `HOLDOUT_START_MS` exactly.
- **MIRROR**: the "slow part" logging idiom at `build_performance_chart.py:142`
  (`logger.info("running pooled walk-forward (this is the slow part)")`) and
  `walkforward.py:333`'s `logger.info("fold %d: ...")` — INFO for progress, never `print` inside
  library code (printing belongs to `cli.py`).
- **GOTCHA #1**: resume correctness depends on the seed. Re-running with a different
  `CAMPAIGN_SEED` is a **new campaign**, not a resume; `run_evolution_stage` must compare the
  seed recorded in Phase 6's `campaigns` row with `config.CAMPAIGN_SEED` and raise
  `CampaignError` on mismatch instead of silently continuing a different experiment.
- **GOTCHA #2**: `time.monotonic()` for the budget (not `time.time()`), so a clock adjustment
  during a 10-hour overnight run cannot end it early or extend it.
- **GOTCHA #3**: contract §10 item 7 — one open trade per symbol, equal notional. The campaign
  changes nothing about that; do not let a "wider population" turn into a portfolio change.
- **VALIDATE**: `pytest tests/test_campaign.py::TestEvolutionStage -v`.

### Task 7: The one-shot holdout stage

- **ACTION**: Evaluate the champion on the holdout exactly once, through
  `walk_forward_pooled`, and capture everything the verdict and the report need.
- **IMPLEMENT**:
  ```python
  @dataclass(frozen=True)
  class HoldoutOutcome:
      campaign_id: str
      champion: ChampionRef
      run_index: int
      n_trials_charged: int
      result: object                 # walkforward.WalkForwardResult
      gate: dict[str, bool]          # result.gate — the 7 conditions
      benchmark: object              # result.benchmark — BenchmarkResult
      diagnostics: dict              # Task 9
      stop_reason: str               # from Task 6
      verdict: str                   # Task 11


  def run_holdout_gate(conn, state_conn, *, campaign_id, force_reason=None) -> HoldoutOutcome:
      """THE ONE-SHOT RUN. Everything before this point is preparation.

      Order of operations is the protocol and must not be rearranged:
        1. refuse unless config.HOLDOUT_LOCKED;
        2. select_champion (deterministic; no champion => CampaignError);
        3. cumulative_trials + charge_gate_trials  -> n_trials;
        4. sample_adequacy_projection on the TRAINING span (Task 8) — logged,
           never a gate;
        5. open_holdout(...)  <-- the holdout is BURNED HERE, before any
           holdout bar is read;
        6. walk_forward_pooled(
               conn, list(config.CAMPAIGN_SYMBOLS),
               start_ms=config.CAMPAIGN_EVOLVE_START_MS,
               end_ms=config.HOLDOUT_END_MS,
               oos_days=config.HOLDOUT_DAYS,
               grid=frozen_grid(champion_graph, ...),
               strategy=champion_graph,
               n_trials=n_trials,
           );
        7. assert the harness's own boundary:
               result.oos_start == config.HOLDOUT_START_MS
               result.oos_end   == config.HOLDOUT_END_MS
           (walkforward.py:304 computes tune_end = end_ms - oos_ms; asserting
           the result rather than trusting the arithmetic is what makes the
           holdout boundary a check instead of a convention);
        8. collect_diagnostics (Task 9);
        9. classify_verdict (Task 11);
       10. close_holdout(gate=result.gate).

      min_trades is NOT passed: the floor stays config.WF_MIN_TRADES = 30.
      There is no code path in this module that lowers it.
      """
  ```
- **MIRROR**: `DRIVE_THE_GATE_NEVER_EDIT_IT` (`walkforward.py:257-269`) and
  `THE_HOLDOUT_BOUNDARY_IS_ARITHMETIC` (`walkforward.py:296-308`).
- **GOTCHA #1**: the fold loop inside this call sweeps windows **inside the evolution span** —
  data evolution has already seen thousands of times. Those fold `test_metrics` are **not**
  independent evidence and the report must label them so (KNOWN-LIMITATIONS §2 reported "10 of
  13 folds negative" precisely because fold-level detail is informative *as diagnosis*, not as
  proof). Only `result.oos_*` is evidence.
- **GOTCHA #2**: `walk_forward_pooled` raises `ValueError` when the span is too short
  (`walkforward.py:305-308`). With `CAMPAIGN_EVOLVE_START_MS` → `HOLDOUT_END_MS` and
  `oos_days=207`, `tune_end - start = 889` days ≥ `WF_TRAIN_DAYS + WF_TEST_DAYS = 240`, giving
  **11 folds** — verified arithmetically before the run, not discovered during it.
- **GOTCHA #3**: `Trade` gained appended fields in Phase 4 (`planned_rr`, `confirmations`,
  `strategy_version`, contract §5). Read them for the report if present; never construct a
  `Trade` positionally in this module.
- **VALIDATE**: `pytest tests/test_campaign.py::TestHoldoutStage -v`.

### Task 8: Sample-adequacy projection (measured on training data only)

- **ACTION**: Project the holdout's raw and independent-equivalent trade counts *before* the
  holdout is opened, from a trade rate measured on the evolution span.
- **IMPLEMENT**:
  ```python
  @dataclass(frozen=True)
  class SampleProjection:
      n_symbols: int
      measured_trades_per_symbol_day: float   # on the EVOLUTION span only
      holdout_days: int
      projected_raw_trades: float
      mean_pairwise_correlation: float        # from Phase 2's data/correlation.py
      effective_n: float                      # Kish effN, MEASURED by Phase 2
      projected_independent_trades: float     # per-symbol trades * effective_n
      required_rate_for_independent_floor: float  # WF_MIN_TRADES/effN/holdout_days
      raw_floor_reachable: bool               # vs config.WF_MIN_TRADES
      independent_floor_reachable: bool


  def sample_adequacy_projection(conn, state_conn, *, campaign_id, champion) -> SampleProjection:
      """Arithmetic, not hope (KNOWN-LIMITATIONS §4).

      Rate is MEASURED by running the champion over the evolution span (a span
      it has already seen — so this costs no holdout information) and dividing
      trades by symbols by days.

      effective_n is READ from Phase 2's data/correlation.py (Kish effective N
      over the full pairwise matrix), never re-derived here from a single mean.
      Phase 2 measured, 2026-07-27, 1095 daily returns 2023-07-27→2026-07-26:
        3 core (BTC/ETH/SOL)                r_bar 0.7574 -> effN 1.193
        best 8-symbol low-corr subset                    -> effN 1.868
        pre-registered selection (BTC + lowest-8)        -> effN 1.834  <- ours
      The analytic n / (1 + (n-1)*r_bar) reproduces the 3-core figure (1.19,
      matching §0b) and is kept only as the intuition; the BROADENED set's
      structure is not faithfully summarised by one mean, so the measured
      1.834 is authoritative.

      The identity worth internalising: with n symbols each producing k
      trades, raw pooled trades = n*k but independent-equivalent
      trades ~= k * effective_n — so independent-equivalent does NOT grow with
      symbol count, only the raw count does. Adding a correlated symbol
      multiplies the rows and divides the factor. §0b's 'rows, not
      information', literally: 3 -> 20 symbols is 6.7x the rows and ~1.5x the
      information (1.834/1.193). The trap is MITIGATED, NOT SOLVED.

      Hence the break-even is a RATE, and it is pre-computed:
        30 / 1.834 = 16.4 trades per symbol over 207 days
                   = 0.0790 trades/symbol/day
      v0.2.0's realized rate was 0.085 (just above); §4's pessimistic framing
      is 0.050 (below). Which world we are in is what this function measures.

      This projection is REPORTED and LOGGED. It is never a gate, and its
      failure NEVER triggers lowering WF_MIN_TRADES or shrinking the holdout
      (KNOWN-LIMITATIONS §4 forbids both by name). If the raw floor is
      unreachable the campaign still runs and reports sample_adequacy: FAIL.
      """
  ```
- **MIRROR**: `compute_metrics` for the trade count (`walkforward.py:176`'s use of it) and Phase
  2's `data/correlation.py` for the Kish `effN` and `r̄` (A8) — **do not recompute a correlation
  matrix here**, and do not substitute the analytic formula for Phase 2's measured value.
- **GOTCHA #1**: measure the rate with the **champion**, not the seed. The broadened detector set
  from Phase 8 is the whole reason the rate might have moved, and a seed-based rate would
  understate it. Record the measured rate, the break-even rate (0.0790), and the two reference
  rates from KNOWN-LIMITATIONS (0.085 and 0.050 trades/symbol/day) so a reader can see the
  direction of travel.
- **GOTCHA #2**: **`pandas` is 3.0.3 in this venv and `Series.pct_change()` is unsafe there.**
  Anywhere this phase computes returns from a price series, write the ratio explicitly:
  `r = s / s.shift(1) - 1.0`. This applies to Task 9's per-symbol end-state as well. Prefer
  reusing Phase 1's `backtest/benchmark.py` and `backtest/equity.py`, which already own the
  return-series construction, over writing a second one here.
- **VALIDATE**: `pytest tests/test_campaign.py::TestSampleProjection -v` — in particular the
  analytic `effN` sanity case `n=3, r̄=0.7574 → 1.193` (matches §0b and Phase 2's measurement),
  the regression pin that the projection consumes Phase 2's **measured** `effN = 1.834` rather
  than an analytic re-derivation, and the break-even identity
  `required_rate == WF_MIN_TRADES / effN / HOLDOUT_DAYS == 0.0790`.

### Task 9: What the gate does **not** see

- **ACTION**: Collect the diagnostics KNOWN-LIMITATIONS §2 had to add by hand, so the v0.3.0
  report carries them by construction.
- **IMPLEMENT**:
  ```python
  def collect_diagnostics(conn, result, champion, projection) -> dict:
      """Everything a single PASS/FAIL bit hides. Keys, each MEASURED:

        full_span_max_drawdown_pct   champion over [evolve_start, holdout_end)
        holdout_max_drawdown_pct     result.oos_equity['max_drawdown_pct']
                                     (§2: v0.2.0's gate saw 19.13% while the
                                     full span was -56%)
        per_symbol_end_state         symbol -> equity multiple over the full
                                     span (§2: ETHUSDT ended at 0.85x despite
                                     positive OOS expectancy)
        fold_test_expectancy         list of per-fold test expectancy, plus
                                     n_negative / n_folds (§2: 10 of 13)
        folds_fell_back_to_defaults  count of folds where no combo reached
                                     WF_MIN_TRADES (train windows themselves
                                     undersampled — §2: 4 of 13)
        benchmark_basket_absolute    the BASKET'S OWN performance over the
                                     holdout: total_return, ann_return_pct,
                                     sharpe, max_drawdown_pct, n_days. REQUIRED
                                     beside the two beats_benchmark_* bits —
                                     Phase 1 measured basket Sharpe -1.303 on
                                     v0.2.0's OOS window, i.e. both conditions
                                     can PASS against a basket that LOST MONEY
        benchmark_per_symbol_absolute same, per symbol, from result.benchmark
        benchmark_context            'BENCHMARK_POSITIVE' when basket
                                     ann_return_pct > 0 AND sharpe > 0, else
                                     'BENCHMARK_NEGATIVE'. Descriptive only —
                                     NOT a gate condition (see the toothless-
                                     null section: adding one would move the
                                     gate)
        holdout_daily_skew           moments of the holdout daily return
        holdout_daily_kurtosis       series. Phase 1 MEASURED its fix on
                                     v0.2.0's series: kurtosis 31.2449 ->
                                     15.5429, skew 3.7883 -> 1.4034, OOS
                                     Sharpe 1.175 -> 1.512. Report this
                                     campaign's measured values and compare to
                                     those, so the fix's effect on THIS series
                                     is visible rather than assumed
        pnl_attribution_mode         'holding-days' or 'exit-day' (A9)
        dsr_at_n_trials_1            the correction OFF, for contrast (§1:
                                     0.742 in v0.2.0 — proves the failure is
                                     data, not convention)
        dsr_at_grid_x_folds          what walkforward would have charged by
                                     default (walkforward.py:383)
        n_trials_charged             the honest cumulative count
        realized_vs_projected_rate   measured holdout trade rate vs Task 8
        mean_cost_pct_per_trade      §0's cost framing, recomputed
        northstar_ann_return_gap     ann_return_pct - CAMPAIGN_NORTHSTAR_ANN_RETURN
        unsized_drawdown_note        constant: every drawdown here is UNSIZED
                                     (KNOWN-LIMITATIONS §7)
      """
  ```
- **MIRROR**: `equity.compute_equity_metrics` (`equity.py:173-220`) for the drawdown/moment
  numbers, `equity._skew_kurt` (`equity.py:159-170`) for the moments, and
  `equity.deflated_sharpe` (`equity.py:129-156`) for the two contrast DSRs.
- **GOTCHA #1**: the full-span drawdown requires running the champion over
  `[CAMPAIGN_EVOLVE_START_MS, HOLDOUT_END_MS)`, which **includes** the holdout — legal only
  *after* `open_holdout` has already burned it (step 5 of Task 7). Never call this before the
  holdout is opened; `collect_diagnostics` must therefore never be reachable from the evolve
  stage. Assert `holdout_is_consumed(state_conn)` on entry.
- **GOTCHA #2**: `HOLDOUT_END_MS` is **exclusive** and the last *closed* 1d bar opens
  2026-07-26T00:00:00Z. `storage.load_candles`' bounds are **inclusive** (contract §1), so any
  direct load for a diagnostic must pass `HOLDOUT_END_MS - 1`; passing `HOLDOUT_END_MS` pulls in
  the still-forming bar that `storage.find_gaps` deliberately ignores
  (`storage.py:164-166`) and would put a partial day in the verdict.
- **GOTCHA #3**: `pandas` 3.0.3 — **never `Series.pct_change()`**; write `s / s.shift(1) - 1.0`.
  `per_symbol_end_state` is the one diagnostic here that touches a price series directly.
- **VALIDATE**: `pytest tests/test_campaign.py::TestDiagnostics -v`.

### Task 10: `scripts/build_campaign_report.py` — every figure with its command

- **ACTION**: Create the report generator that renders
  `.claude/PRPs/reports/phase9-campaign-verdict.md` from persisted campaign state.
- **IMPLEMENT**: mirror `build_performance_chart.py`'s shape exactly — `sys.path.insert(0, "src")`
  bootstrap (`:28`), `collect()` (`:140-171`), a scorecard builder (`:174-214`), `build()`
  (`:217`), `main()` with `--out` required (`:687-698`), `raise SystemExit(main())` (`:728-729`).
  Output sections, in order:
  1. **Header** — campaign id, seed, symbol set, spans in **both** epoch ms and ISO dates,
     `run_index`, and, when `run_index > 1`, a `NOT A CLEAN HOLDOUT` banner with the override
     reason verbatim.
  2. **The gate, per condition** — seven rows shaped `(label, measured value, threshold, verdict)`
     as in `_scorecard` (`build_performance_chart.py:174-214`), with `ok` taken from
     `result.gate[condition]`, never recomputed.
  3. **Buy-and-hold beside it, in absolute terms** — the basket and every symbol from
     `result.benchmark` (contract §4's `BenchmarkResult`) with **total return, annualised, Sharpe
     and max drawdown**, so §0's "no axis on which it wins" is checkable at a glance rather than
     reconstructable. **The two `beats_benchmark_*` bits may never appear without these numbers
     next to them**, and the section carries the `benchmark_context` label. Rationale, measured:
     Phase 1 found the basket's Sharpe over v0.2.0's OOS window was **−1.303**, so both bits can
     PASS against a basket that lost money — which is not the protection §0 asked for. Also label
     annualised figures by kind: a full-span figure is a **CAGR** (v0.2.0's span was 1095 days =
     exactly 3 years), whereas a short-window figure is an **extrapolation** (§2's "+68%" from 90
     days and 23 trades). Never print the two in the same column without that distinction.
  4. **Trial accounting** — `n_trials_charged`, the two contrast DSRs, and the required-Sharpe
     row for that trial count, with the command that computes it.
  5. **Northstar** — `ann_return_pct` vs `CAMPAIGN_NORTHSTAR_ANN_RETURN`, explicitly labelled
     "reported, not gated" (discrepancy note 2).
  6. **Sample adequacy** — raw and independent-equivalent counts, `r̄`, `effective_n`, and the
     sentence that the floor was never lowered.
  7. **What the gate does not see** — every Task 9 key.
  8. **Verdict** — rendered as `<id> / <benchmark_context>`, with its meaning and recommended
     action copied from the taxonomy table in this plan (the classifier in Task 11 owns both
     labels; the report only renders them). A `beats_benchmark_*` PASS under
     `BENCHMARK_NEGATIVE` must be described as "beat a basket that lost money" and must **not**
     be presented as northstar evidence.
  9. **Provenance table** — one row per figure: `figure | value | command | span`. Contract §12.5
     and this repo's own committed discipline (`git log`: "Use measured rather than derived
     figures in the benchmark table").
- **MIRROR**: `REPORT_GENERATOR_SHAPE` and `SCORECARD_ROWS_READ_THRESHOLDS_FROM_SOURCE`.
- **GOTCHA #1**: the generator must **not** re-run the campaign. It reads `state.db` (and, for
  charts it does not draw, nothing else). A report generator that re-runs the gate is a second
  peek at the holdout wearing a reporting hat. If the campaign state is absent, exit non-zero
  with "no completed campaign; run `campaign --stage holdout` first".
- **GOTCHA #2**: no figure may be typed by hand into the template, and none may be derived from
  another. If a number cannot be traced to a command, it does not go in the report.
- **VALIDATE**:
  ```bash
  python -m py_compile scripts/build_campaign_report.py
  .venv/bin/python scripts/build_campaign_report.py --help
  ```

### Task 11: Verdict classifier + `cli.py` wiring

- **ACTION**: Add `classify_verdict` to `campaign.py`; register the `campaign` subcommand and
  `_campaign_command` in `cli.py`.
- **IMPLEMENT**:
  ```python
  # campaign.py
  VERDICTS = ("A", "A_PRIME", "B1", "B2", "B3", "B4", "C")
  BENCHMARK_CONTEXTS = ("BENCHMARK_POSITIVE", "BENCHMARK_NEGATIVE")


  def benchmark_context(benchmark) -> str:
      """How much a 'beats the benchmark' PASS is actually worth.

      'BENCHMARK_POSITIVE' iff the basket's ann_return_pct > 0 AND sharpe > 0;
      otherwise 'BENCHMARK_NEGATIVE'.

      Measured motivation: Phase 1 found the equal-weight basket's Sharpe over
      v0.2.0's 90-day OOS window was -1.303 — buy-and-hold LOST MONEY there, so
      both beats_benchmark_* conditions PASS trivially. In a bear window almost
      anything beats inaction, and KNOWN-LIMITATIONS §0's complaint was about
      blessing something WORSE than inaction. This label is how the report keeps
      that distinction visible.

      DELIBERATELY NOT A GATE CONDITION. GATE_CONDITIONS stays the contract's
      seven; requiring a positive benchmark would be moving the gate, and
      beating a falling market is a real if weaker result. Report the context;
      let the reader judge.
      """


  def classify_verdict(gate: dict[str, bool], oos_equity: dict, projection) -> str:
      """Map the 7-condition gate onto the taxonomy declared in the plan.

      Returns the id only. Callers pair it with benchmark_context() and render
      '<id> / <context>'; the two are never collapsed into one label.

      Code, not judgement — the thresholds are the plan's, and the order of
      tests is fixed so exactly one id can apply:
        all pass and ann_return > northstar          -> "A"
        all pass                                     -> "A_PRIME"
        not gate['beats_benchmark_return'] or
            not gate['beats_benchmark_sharpe']       -> "B3"   (loses to inaction)
        not gate['sample_adequacy']                  -> "B4"   (data problem)
        only 'dsr' fails                             -> "B1"   (unprovable edge)
        'dsr' fails plus others                      -> "B2"
        (anything unclassifiable / run incomplete)   -> "C"
      B3 is tested before B4 and B1 because losing to buy-and-hold makes the
      significance question moot: §0's lesson is that a gate which cannot see
      inaction can bless something worse than it.
      """
  ```
  ```python
  # cli.py — registered in phase order, after the existing subparsers (cli.py:129-144)
  campaign_parser = subparsers.add_parser(
      "campaign",
      help="Run the pre-registered v0.3.0 walk-forward campaign (Phase 9): "
           "evolution on the training span, then the one-shot holdout gate",
  )
  campaign_parser.add_argument(
      "--stage", choices=("probe", "evolve", "holdout", "report", "all"),
      default="all", help="Campaign stage to run (default: all)",
  )
  campaign_parser.add_argument(
      "--campaign-id", default=None,
      help="Resume an existing campaign; default derives a new id from the UTC date",
  )
  campaign_parser.add_argument(
      "--no-resume", action="store_true",
      help="Start a fresh campaign instead of continuing the latest one",
  )
  campaign_parser.add_argument(
      "--force-holdout-rerun", metavar="REASON", default=None,
      help="Re-run an already-consumed holdout. The reason is RECORDED and the "
           "report is stamped NOT A CLEAN HOLDOUT. Not a routine flag.",
  )
  ```
  Dispatch as its own `elif args.command == "campaign":` arm following the
  `("backtest", "walkforward")` arm's shape (`cli.py:199-209`): `connect(args.db)` for
  `ohlcv.db`, `statestore.connect()` for `state.db`, call `_campaign_command`, close both,
  `sys.exit(exit_code)`.

  `_campaign_command(conn, state_conn, *, stage, campaign_id, resume, force_reason) -> int`
  prints with `_fmt`/`_print_metrics` (`cli.py:356-368`) and returns:

  | Exit | Meaning |
  |---|---|
  | 0 | gate PASS (verdict A or A′) |
  | 1 | gate FAIL — **a completed, valid outcome** (B1–B4). The code reports the *gate*, not whether the phase succeeded |
  | 2 | ambiguous / aborted (verdict C): `HoldoutViolation`, holdout already consumed without override, missing dependency, no champion, `HOLDOUT_LOCKED` false |

- **MIRROR**: `CLI_SUBCOMMAND_AND_DISPATCH` and `CLI_HANDLER_PRINT_AND_EXIT_CODE`
  (`cli.py:129-144, 199-209, 396-439`). Print `GATE: PASS|FAIL` in the same words the existing
  handler uses (`cli.py:437`) so the two commands read alike, then the per-condition block below
  it. The two `beats_benchmark_*` lines print the basket's absolute
  `ann_return_pct`/`sharpe`/`max_drawdown_pct` on the same line as the comparison, and the
  verdict line prints `VERDICT: <id> / <benchmark_context>` — never the id alone.
- **GOTCHA #1**: exit code 1 is not a bug report. Document in the handler docstring that a
  statistically honest "no" exits 1 by design, and that only exit 2 means the phase failed to
  produce a verdict.
- **GOTCHA #2**: `campaign` needs **two** connections. Keep them separate and never pass
  `state_conn` where a bar-loading function expects `conn` — contract §6's whole point is that a
  corrupt experiment log cannot endanger 117 MB of price history.
- **GOTCHA #3**: `--stage all` runs `probe → evolve → holdout → report` in one process. It must
  still honour every guard: an already-consumed holdout stops the pipeline at exit 2 *after* the
  evolve stage has been resumed/completed, so a long run is never wasted by the guard firing at
  the end.
- **VALIDATE**: `pytest tests/test_campaign.py::TestCampaignCli -v`.

### Task 12: `tests/test_campaign.py`

- **ACTION**: Create the reserved test file (contract §8). Class-per-area, tier constants derived
  from config, autouse cache clearing, synthetic in-memory SQLite, hand-computed expectations for
  the arithmetic.
- **IMPLEMENT**: classes and their load-bearing cases —
  ```python
  class TestSpans:
      # holdout_span() rejects a HOLDOUT_DAYS that disagrees with the endpoints
      # evolution_span() ends exactly at HOLDOUT_START_MS
      # assert_no_holdout_overlap: passes for [evolve_start, holdout_start);
      #   raises for [holdout_start-1, holdout_start+1)  <- the straddle case
      #   raises for a span strictly inside the holdout
      #   passes for the empty span [holdout_start, holdout_start)
      # config invariant: HOLDOUT_END_MS - HOLDOUT_DAYS*DAY_MS == HOLDOUT_START_MS

  class TestHoldoutLedger:
      # first open_holdout -> run_index 1; holdout_is_consumed() True afterwards
      # second open_holdout without a reason -> CampaignError
      # second WITH a reason -> run_index 2, reason persisted
      # a 'violation' row does NOT make holdout_is_consumed() True
      # the row exists BEFORE the gate runs: simulate a raising walk_forward_pooled
      #   and assert the consumption row survives with completed_ts IS NULL
      # HOLDOUT_LOCKED=False -> CampaignError

  class TestTrialAccounting:
      # cumulative_trials counts rows for the campaign only, not other campaigns
      # the value passed to walk_forward_pooled is the cumulative count:
      #   capture kwargs via monkeypatch and assert n_trials == cumulative
      # REGRESSION: n_trials is never left to walkforward's default
      #   (len(combos)*len(folds) == 11 or 12) — the flattering path

  class TestChampionSelection:
      # highest fitness wins
      # tie -> earliest generation
      # tie -> lexicographically smallest graph_hash (total, order-independent)
      # empty population -> CampaignError (never falls back to a seed strategy)
      # frozen_grid: one value per axis; statistics.median_low over it returns
      #   that value; walkforward's on-grid assertion cannot fire

  class TestSampleProjection:
      # analytic effN sanity: n=3, r=0.7574 -> 1.193 (matches §0b AND Phase 2)
      #                       n=1 -> 1.0 ; r=0 -> n
      # REGRESSION: the projection READS Phase 2's measured Kish effN (1.834 for
      #   the pre-registered selection) and does NOT re-derive it analytically
      #   from a mean r — stub data/correlation.py and assert the value flows
      #   through unchanged
      # independent-equivalent == per-symbol trades * effective_n, and is
      #   INVARIANT to symbol count (8 and 9 symbols at the same rate give the
      #   same independent-equivalent; only raw differs)
      # break-even identity: required_rate == 30 / 1.834 / 207 == 0.0790
      # at 0.085/symbol/day -> independent_floor_reachable is True  (32.3)
      # at 0.050/symbol/day -> independent_floor_reachable is False (19.0)
      # raw floor compares against config.WF_MIN_TRADES, unmodified
      # REGRESSION: nothing in campaign.py writes WF_MIN_TRADES or oos_days
      #   -> assert config.WF_MIN_TRADES == 30 and grep-style source assertion

  class TestEvolutionStage:
      # stops at CAMPAIGN_GENERATIONS -> stop_reason 'COMPLETE'
      # stops on budget -> 'BUDGET_EXHAUSTED' (fake monotonic clock)
      # stops on patience -> 'CONVERGED'
      # resume continues at the last checkpointed generation, same seed
      # seed mismatch on resume -> CampaignError
      # every span forwarded to the runner ends at HOLDOUT_START_MS

  class TestHoldoutStage:
      # asserts result.oos_start/oos_end == the declared holdout bounds
      # a stub result with a failing dsr yields verdict B1/B2, exit code 1
      # HoldoutViolation anywhere -> exit 2 and a 'violation' row
      # collect_diagnostics refuses to run before the holdout is opened

  class TestDiagnostics:
      # full-span drawdown differs from holdout-window drawdown on a fixture
      #   where it must (pins KNOWN-LIMITATIONS §2's 19.13% vs -56% asymmetry)
      # dsr_at_n_trials_1 > dsr_at_cumulative on the same series
      # folds_fell_back_to_defaults counts folds with train_expectancy is None

  class TestCampaignCli:
      # PASS stub -> exit 0 and 'GATE: PASS' in stdout
      # FAIL stub -> exit 1 and each of the 7 condition names in stdout
      # already-consumed holdout without --force-holdout-rerun -> exit 2
      # --force-holdout-rerun prints the NOT A CLEAN HOLDOUT stamp

  class TestVerdictClassifier:
      # one case per taxonomy id, including B3 taking precedence over B1/B4
      # exactly one id ever applies (exhaustive over a small gate-dict space)
      # benchmark_context: basket ann>0 and sharpe>0 -> BENCHMARK_POSITIVE
      #   basket sharpe -1.303 (Phase 1's measured v0.2.0 OOS basket)
      #     -> BENCHMARK_NEGATIVE even though both beats_benchmark_* PASS
      #   ann>0 but sharpe<=0, and sharpe>0 but ann<=0 -> both NEGATIVE
      # REGRESSION: benchmark_context is NOT in GATE_CONDITIONS and does not
      #   change `passed` — assert a BENCHMARK_NEGATIVE all-pass run still
      #   yields verdict A/A_PRIME and exit 0 (the gate was not moved)
  ```
- **MIRROR**: `TEST_TIER_CONSTANTS_AND_CACHE_ISOLATION` (`tests/test_backtest.py:24-46`),
  `make_trade` (`:49-57`), `per_symbol_fake_run_factory` (`:469-495`) — including its comment
  about spreading `exit_ts` across the span so `daily_returns` (`equity.py:42-51`) actually
  buckets them — and `TestBacktestCli`'s stub-and-assert idiom (`:591-637`).
- **IMPORTS**: `from trading_bot import campaign, config`,
  `from trading_bot.cli import _campaign_command`, `from trading_bot.data import statestore`,
  `import sqlite3` for `:memory:` state fixtures.
- **GOTCHA #1**: monkeypatch `campaign.walk_forward_pooled` (the name `campaign.py` looked up),
  never `walkforward.walk_forward_pooled`, and never Phase 6 internals — contract §2's ownership
  is enforced socially, but this is where it leaks in practice.
- **GOTCHA #2**: state-db tests use `sqlite3.connect(":memory:")` plus `campaign._ensure_schema`
  and Phase 6's / Phase 1's DDL as needed. Never touch the real `data/state.db` from a test; a
  test that consumes the real holdout row is a catastrophic false positive.
- **GOTCHA #3**: no test may be marked `network` — this phase makes no network call.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_campaign.py -v`.

### Task 13: Run the campaign, write the two report artifacts

- **ACTION**: Execute the protocol and write the deliverables. This is the task that produces the
  phase's actual output; the twelve before it exist to make this one honest.
- **IMPLEMENT**:
  1. `campaign --stage probe` → record measured `T_eval`; if over budget, reduce
     `CAMPAIGN_GENERATIONS` **in config**, re-commit, and note the change as a consumed degree of
     freedom (KNOWN-LIMITATIONS §9 discipline).
  2. `campaign --stage evolve` (overnight; resumable).
  3. `campaign --stage holdout` — **exactly once**. See Manual Validation below.
  4. `campaign --stage report` → `.claude/PRPs/reports/phase9-campaign-verdict.md`.
  5. Hand-write `.claude/PRPs/reports/KNOWN-LIMITATIONS-v0.3.0.md`, modelled section-for-section
     on v0.2.0's file, **knowing** the verdict. Required sections, at minimum:
     - §0 the verdict against buy-and-hold, per condition, with the basket
     - §1 the gate, all seven conditions, and `n_trials` charged
     - §2 what the gate does not see (Task 9's keys, prose)
     - §3 the sample-size arithmetic: raw vs independent-equivalent, and the explicit statement
       that `WF_MIN_TRADES` was not lowered and the window was not shrunk
     - §4 degrees of freedom consumed by the whole of v0.3.0 (Phases 1–9), cumulative
     - §5 what IS established (the framework's own success metrics: zero engine-core edits per
       plug-in, parity test green, feedback-loop liveness) — separate from strategy performance
     - §6 scope not built, carried forward from v0.2.0 §7 with anything new
     - §7 the recommended next action from the verdict taxonomy
- **GOTCHA**: the limitations document is written **after** the verdict and must not be drafted
  optimistically in advance. v0.2.0's file works because it was written knowing the gate rejected
  the build. Match that candour: if the answer is "the framework works and the strategy family
  does not", say exactly that in the first paragraph.
- **VALIDATE**: both files exist, every number in them appears in the provenance table of the
  generated report, and the full suite is green (below).

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| Holdout arithmetic | config constants | `END - DAYS*DAY_MS == START` | config regression guard |
| Overlap straddle | `[h_start-1, h_start+1)` | `HoldoutViolation` | **the** case a convention would miss |
| Overlap empty span | `[h_start, h_start)` | no raise | off-by-one guard |
| Consume once | two `open_holdout` calls | 2nd raises without a reason | core protocol |
| Consume-before-run | gate raises mid-run | consumption row persists, `completed_ts` NULL | prevents the `kill -9` free peek |
| Violation row | `record_violation` | `holdout_is_consumed()` still False | audit without false blocking |
| Cumulative trials | ledger with 3011 rows | `n_trials=3011` reaches `walk_forward_pooled` | prevents the flattering default (12) |
| Champion tie-break | equal fitness, gens 4 and 7 | gen 4 wins | determinism |
| Champion tie-break 2 | equal fitness and gen | smaller `graph_hash` wins | total order |
| No champion | empty population | `CampaignError`, exit 2 | never falls back to seed |
| Degenerate grid | 1 value/axis | `median_low` returns it; no on-grid assertion | pins `walkforward.py:357-373` |
| Effective N (analytic sanity) | n=3, r̄=0.7574 | 1.193 | hand-computed; matches §0b **and** Phase 2's measurement |
| Effective N (measured) | Phase 2's selection | 1.834 flows through unchanged | regression: never re-derived from a mean |
| Independent-equiv | k=17.6, effN=1.834 | 32.3 | the identity in Task 8; clears the floor |
| Independent-equiv | k=10.4, effN=1.834 | 19.0 | the pessimistic-rate world; does not clear |
| Independent-equiv invariance | 8 vs 9 symbols, same rate | same indep-equiv, different raw | "rows, not information" |
| Break-even rate | 30 / 1.834 / 207 | 0.0790 trades/symbol/day | the number the verdict turns on |
| Budget stop | fake clock past budget | `stop_reason='BUDGET_EXHAUSTED'` | overnight mechanics |
| Patience stop | flat fitness × 5 gens | `stop_reason='CONVERGED'` | declared stopping rule |
| Resume | checkpoint at gen 13 | continues at 14, same seed | crash at hour 6 |
| Seed mismatch | different `CAMPAIGN_SEED` | `CampaignError` | a resume is not a new experiment |
| Boundary assertion | stub result with wrong `oos_start` | raises | holdout boundary is a check |
| Verdict B3 first | benchmark FAIL + dsr FAIL | `"B3"` | precedence declared in advance |
| Verdict B1 | only dsr FAIL | `"B1"` | the interesting "no" |
| Verdict exhaustive | all gate-dict combinations | exactly one id each | no unclassifiable state |
| CLI exit codes | PASS / FAIL / violation stubs | 0 / 1 / 2 | exit 1 is a valid outcome |
| CLI per-condition | FAIL stub | all 7 names in stdout | §0's "one bit hides it" |
| Benchmark context | basket sharpe −1.303, ann < 0 | `BENCHMARK_NEGATIVE` while both `beats_benchmark_*` PASS | Phase 1's measured toothless-null case |
| Benchmark context | basket ann > 0, sharpe > 0 | `BENCHMARK_POSITIVE` | the strong-claim case |
| Gate not moved | all 7 PASS + `BENCHMARK_NEGATIVE` | verdict A/A′, exit 0 | context is descriptive, never a condition |
| Benchmark absolutes printed | any run | basket total/ann/Sharpe/DD on the `beats_benchmark_*` lines | never the bit alone |
| `WF_MIN_TRADES` guard | — | `== 30`, unmodified by this module | KNOWN-LIMITATIONS §4 |

### Edge Cases Checklist

- [x] Span straddling the holdout boundary by one millisecond → `HoldoutViolation`
- [x] Process killed between `open_holdout` and the gate → holdout stays burned
- [x] Second holdout run → refused; with override → allowed, logged, report stamped
- [x] Empty population / no champion → exit 2, verdict C, no holdout consumed
- [x] Campaign resumed with a different seed → refused
- [x] Trial ledger empty (Phase 1 wired wrong) → `n_trials` would be 0; treat as verdict C
      rather than charging DSR nothing
- [x] `RESEARCH_SYMBOLS` missing → stop; never silently fall back to the 3 correlated majors
- [x] Gate metric `None` (degenerate series) → Phase 1's gate fails safe
      (`walkforward.py:243-254`); the report prints `--` via `_fmt` (`cli.py:356-358`)
- [x] Fewer than 11 folds because history moved → the fold count is asserted, not assumed
- [ ] Concurrent campaigns — N/A by design: one operator, one machine; `state.db` serialises via
      the statestore `_db_lock`, and a second concurrent campaign is not a supported mode
- [ ] Network failure — N/A: this phase makes no network call

---

## Validation Commands

### Static Analysis

```bash
python -m py_compile src/trading_bot/campaign.py src/trading_bot/cli.py \
    src/trading_bot/config.py scripts/build_campaign_report.py
```
EXPECT: clean. **No linter and no type checker are configured** in this repo
(KNOWN-LIMITATIONS §8, contract §12.2) — `py_compile` and `pytest` are the whole of static and
dynamic validation. Do not invent a `mypy`/`ruff` invocation.

### Unit Tests

```bash
.venv/bin/python -m pytest tests/test_campaign.py -v
```
EXPECT: all new classes pass.

### Full Test Suite

```bash
.venv/bin/python -m pytest -q
```
EXPECT: **286 pre-existing tests still green** (baseline verified 2026-07-27 via
`.venv/bin/python -m pytest --collect-only -q` → "286 tests collected"), plus this phase's new
tests. Contract §8: the 286 must stay green at every phase boundary.

### Protocol Guard Checks

```bash
# The floor and the window are never touched by this phase.
grep -rn "WF_MIN_TRADES\s*=" src/trading_bot/ | grep -v "config.py:151"
grep -rn "min_trades=" src/trading_bot/campaign.py
grep -rn "oos_days=" src/trading_bot/campaign.py
```
EXPECT: first command prints nothing (only `config.py:151`'s `WF_MIN_TRADES = 30` assigns it);
second prints nothing (`campaign.py` never overrides the floor); third prints exactly one line,
`oos_days=config.HOLDOUT_DAYS`.

```bash
# Files this phase must not have touched.
git diff --name-only | grep -E "backtest/walkforward\.py|evolution/"
```
EXPECT: no output. Phase 1 owns the gate, Phase 6 owns evolution (contract §7, §2).

```bash
# No second fitness oracle.
grep -rn "bruteforce" src/trading_bot/campaign.py
```
EXPECT: no output (contract §4.1).

### Config Pre-registration Check

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "src")
from trading_bot import config as c
assert c.HOLDOUT_END_MS - c.HOLDOUT_DAYS * 86_400_000 == c.HOLDOUT_START_MS
assert c.CAMPAIGN_EVOLVE_START_MS < c.HOLDOUT_START_MS
assert c.HOLDOUT_LOCKED is True
assert c.WF_MIN_TRADES == 30
print("holdout", c.HOLDOUT_START_MS, "->", c.HOLDOUT_END_MS, f"({c.HOLDOUT_DAYS}d)")
print("evolve days", (c.HOLDOUT_START_MS - c.CAMPAIGN_EVOLVE_START_MS) // 86_400_000)
print("symbols", len(c.CAMPAIGN_SYMBOLS), "planned evals",
      c.CAMPAIGN_POPULATION_SIZE * c.CAMPAIGN_GENERATIONS)
EOF
```
EXPECT: `holdout 1767225600000 -> 1785110400000 (207d)`, `evolve days 889`, and a symbol count
matching Phase 2's selection.

### Budget Probe (safe, repeatable — training span only)

```bash
python -m trading_bot.cli campaign --stage probe
```
EXPECT: a measured `seconds_per_eval` and a projected wall clock. Touches no holdout bar. May be
run any number of times; each probe evaluation **is** charged to the trial ledger.

### Manual Validation — THE CAMPAIGN, run exactly once

The holdout stage is the one irreversible command in the project. Treat this checklist as part of
the protocol, not as advice.

**Before running `campaign --stage holdout`:**

- [ ] `git status` is clean and `config.py`'s Phase 9 block is **committed**. A holdout consumed
      against uncommitted thresholds is verdict C — the protocol must be recoverable from the
      repository alone.
- [ ] The Config Pre-registration Check above passes, and its printed spans match this plan's
      declared numbers (`1767225600000` → `1785110400000`, 207 days, 889 evolution days).
- [ ] The evolve stage reported a `stop_reason` and a champion exists
      (`campaign --stage evolve` printed one, and `select_champion` returns without raising).
- [ ] `campaign --stage holdout` has **never** been run against this span:
      `sqlite3 data/state.db "select id, run_index, opened_ts, override_reason from
      holdout_consumption where kind='consume'"` returns **no rows**.
- [ ] The sample projection from Task 8 is recorded, whatever it says. If it projects fewer than
      30 raw trades, **run anyway** and report `sample_adequacy: FAIL`. Do not adjust anything.
- [ ] You have read [Pre-registered verdict distribution](#pre-registered-verdict-distribution)
      **today**, and specifically the row saying `dsr` is expected to fail. If the outcome
      surprises you into wanting to change a threshold, the plan has already told you the answer:
      you may not.

**Running it:**

```bash
python -m trading_bot.cli campaign --stage holdout
python -m trading_bot.cli campaign --stage report
```

**After running it:**

- [ ] Record the verdict verbatim — all seven conditions, the benchmark basket, `n_trials`
      charged, `run_index` — in `.claude/PRPs/reports/phase9-campaign-verdict.md` (the generator
      does this; verify it did).
- [ ] Every number in both report artifacts appears in the provenance table with its command and
      span (contract §12.5).
- [ ] Write `KNOWN-LIMITATIONS-v0.3.0.md` **knowing** the verdict (Task 13.5).
- [ ] On FAIL (verdicts B1–B4): the pre-committed response is the taxonomy's recommended action
      for that id. It is **never** re-running with a lower `n_trials`, a lower `WF_MIN_TRADES`, a
      shorter holdout, a different champion rule, or "one more generation". A second run is a new
      campaign, stamped `NOT A CLEAN HOLDOUT`, and its DSR is charged the *combined* trial count.
- [ ] Record every degree of freedom this phase consumed (contract §12.6): the probe evaluations,
      the population × generations, the final gate's own folds, and any config value changed
      after the probe.

---

## Acceptance Criteria

- [ ] All 13 tasks completed; every validation command above passes
- [ ] `HOLDOUT_START_MS` / `HOLDOUT_END_MS` / `HOLDOUT_DAYS` committed, mutually consistent, and
      asserted by a test
- [ ] Evolution provably never sees the holdout: `end_ms = HOLDOUT_START_MS` **and**
      `assert_no_holdout_overlap` on every span, both covered by tests including the straddle case
- [ ] The holdout is consumed exactly once, recorded **before** the run, and a second attempt is
      refused without a logged override
- [ ] `n_trials` passed to the gate equals the cumulative ledger count for the campaign; the
      in-process default (`len(combos)*len(folds)`) is never used
- [ ] The champion is selected deterministically by the declared tie-break, and the holdout is
      evaluated at exactly its configuration (degenerate grid; no re-selection)
- [ ] The gate verdict is reported **per condition** (all 7), with the buy-and-hold basket beside
      it and the trial count printed
- [ ] The basket's **absolute** performance (total return, annualised, Sharpe, max DD) appears
      beside the two `beats_benchmark_*` bits, and every verdict is rendered
      `<id> / <benchmark_context>`. `GATE_CONDITIONS` is still exactly the contract's seven — no
      benchmark-positivity condition was added
- [ ] Annualised figures are labelled by kind: full-span = **CAGR**, short-window =
      **extrapolation** (never mixed in one column)
- [ ] "What the gate does not see" is reported: full-span vs OOS drawdown, per-symbol end state,
      fold test-expectancy distribution, folds that fell back to defaults, holdout return moments,
      and both contrast DSRs
- [ ] Raw **and** independent-equivalent sample sizes are reported, with `r̄` and Phase 2's
      **measured** Kish `effN` (1.834 for the pre-registered selection), the 0.0790 break-even
      rate, the measured rate beside it, and the sentence that the correlation trap is
      *mitigated, not solved*
- [ ] `WF_MIN_TRADES` is still 30 and the holdout window is unshrunk — verified by grep
- [ ] `walkforward.py` and `evolution/` are unmodified — verified by `git diff --name-only`
- [ ] 286 pre-existing tests green; `python -m py_compile` clean; no linter/type-checker invented
- [ ] Both report artifacts exist, and every figure in them has a command and a span
- [ ] The verdict is one of A/A′/B1–B4 — i.e. **not** C

## Completion Checklist

- [ ] Frozen dataclasses, keyword-only `| None = None` config fallbacks, and
      `logging.getLogger("trading_bot")` — the codebase's shapes, not new ones
- [ ] `CampaignError`/`HoldoutViolation` subclass `RuntimeError`; `ValueError` from
      `walk_forward_pooled` is caught at the CLI boundary and turned into an exit code, as
      `_walkforward_command` already does (`cli.py:405-409`)
- [ ] Library code logs; only `cli.py` prints
- [ ] Tests derive tier constants from config and clear engine caches (autouse), per
      `tests/test_backtest.py:24-46`
- [ ] No hardcoded gate threshold anywhere in this phase — all read from `walkforward.GATE_MIN_*`
      and `config.WF_MIN_TRADES`
- [ ] Only the reserved names used: `CAMPAIGN_*` / `HOLDOUT_*`, `campaign` / `_campaign_command`,
      `tests/test_campaign.py`
- [ ] No file owned by another phase created or edited (contract §2), and the one documented
      extension (`src/trading_bot/campaign.py`) is called out in discrepancy note 4
- [ ] Self-contained — the only genuine unknowns are A3/A4's exact helper names, each with a
      guaranteed SQL fallback

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **DSR fails, and the failure gets rationalised into an `n_trials` argument** | **High** — it is the expected outcome | **High**: it would repeat v0.2.0's mistake with more machinery | The required-Sharpe table is committed **before** the run; KNOWN-LIMITATIONS §1 already measured that DSR fails at `n_trials=1` too; contract §4 forbids softening. Verdict B1's recommended action is *more calendar time*, the only honest lever |
| Someone re-runs the holdout after a bad verdict | Medium (human) | **Critical** — invalidates the whole phase | Consumption row written before the run; override requires a recorded reason; report stamped `NOT A CLEAN HOLDOUT`; combined trial count charged on any re-run |
| A crash mid-holdout is treated as "didn't happen" | Medium | **Critical** | Write-then-run ordering; a test asserts the row survives a raising gate |
| Phase 6's runner is not actually resumable | Medium | High (a 10-hour run restarts) | Resume is a Task 6 acceptance test against Phase 6's tables; if it fails, it is reported as a Phase 6 defect — Phase 9 does not reimplement `evolution/` |
| Wall-clock blows past overnight | Medium | Medium | `--stage probe` **measures** `T_eval`; the declared over-budget response is fewer generations, not a shorter holdout |
| Phase 2's broadened set is still BTC beta | **Confirmed, measured**: all 20 stored symbols are BTC beta (contract §0a); the selection's Kish `effN` is **1.834** vs 1.193 for the 3 core — 6.7× rows for ~1.5× information | Medium: raw sample clears easily, independent sample is marginal | Trap is **mitigated, not solved** — say exactly that. Report both counts; quote §0b; never present rows as independence. No attempt to pass by adding more majors |
| Trade rate lands below the 0.0790 break-even (e.g. Phase 4's R:R-after-costs filter rejects more) | **Medium** — v0.2.0's realized 0.085 is only just above it | Medium — independent-equivalent falls under 30 while raw still clears | Measured, not assumed, by Task 8 on the training span before the holdout opens. The gate still uses the raw count (unchanged); the shortfall is a **reported caveat**, never a reason to retune, lower the floor, or shrink the window |
| The measured `effN = 1.834` gets used as licence to call the sample independent | Medium (interpretation) | High — it would repeat §0b's error in the opposite direction | 1.834 means "fewer than two independent instruments". The report states the number **and** its meaning together; Task 12 pins the invariance test showing symbol count does not buy information |
| `campaign.py` accretes logic that belongs to Phase 6 | Medium | Medium (contract §2 erosion) | NOT Building is explicit; the guard greps in Validation Commands check `git diff` |
| The report quietly contains a derived figure | Medium | High (the repo has already committed once to fixing exactly this) | Provenance table is a required section; Task 10 GOTCHA #2 forbids hand-typed numbers |
| A test consumes the real `data/state.db` holdout row | Low | **Critical** | In-memory SQLite only; Task 12 GOTCHA #2; `.gitignore` check in Task 3 GOTCHA #3 |
| Verdict C through some unanticipated ambiguity | Low | High (the phase's only failure mode) | The ambiguity table enumerates every known form with its prevention; the classifier is exhaustive over the gate-dict space by test |

## Notes

- **The deliverable is the protocol, not the number.** Tasks 1–12 exist so that Task 13's output
  is believable whichever way it goes. If implementation time runs short, the thing that must not
  be cut is the holdout enforcement (Tasks 2, 3) or the trial accounting (Task 4) — cutting the
  report generator would merely make the write-up manual.
- **Why 207 days.** It is the point where two measured pressures meet: the required-Sharpe table
  (longer holdout ⇒ lower required Sharpe at a given trial count) and KNOWN-LIMITATIONS §2's
  observation that v0.2.0's *train* windows were themselves undersampled (4 of 13 folds fell back
  to defaults), which argues for leaving evolution ≥ 889 days and 11 folds. Both figures are in
  this plan; neither was chosen for how the answer would look.
- **Fold-level numbers in this phase are diagnosis, not evidence.** The fold sweep inside the
  final `walk_forward_pooled` call runs on data evolution has seen thousands of times. Only
  `result.oos_*` — the 207-day holdout — is evidence. The report must label this explicitly, or a
  future reader will do what §2 warns about and read a favourable in-sample window as an edge.
- **Evolution's own OOS is not a holdout.** Phase 6's runner carves an internal validation window
  inside the evolution span and scores against it thousands of times. It is a *selection* signal.
  The report gives it that name and no other.
- **Contract §4's cumulative ledger is the most consequential decision in v0.3.0**, and this
  phase is where its bill arrives. Charging DSR for 3011 trials over 207 days demands an
  annualised Sharpe of ~7.16. That is the honest price of a large search on a small sample, and
  the finding is worth writing down even though — especially though — it makes the northstar
  unreachable through this route. The alternative (charging 12) is not a better result; it is the
  same result reported dishonestly.
- **Amended 2026-07-27 after Phase 2's plan landed** (it completed after this one and *measured*
  the constant the sample arithmetic depends on). Three changes, all narrowing uncertainty rather
  than relaxing anything:
  1. **`effN` is now measured, not assumed**: 1.193 for the 3 core (r̄ 0.7574, reproducing §0b)
     but **1.834** for the pre-registered broadened selection, over 1095 daily returns with zero
     missing bars. The identity `indep ≈ k · effN` was right; the constant fed into it was not.
     Consequence: the independent-30 floor needs **16.4 trades/symbol (0.0790/symbol/day)**, not
     ~25.2 — so the conclusion moved from *"almost certainly does not clear"* to **"plausibly
     clears, decided by one measured rate"** (0.085 clears at 32.3; 0.050 fails at 19.0).
     Verdict B4's prior drops accordingly. **Nothing about the gate, the floor, or the window
     changed** — only the honesty of the projection.
  2. The correlation trap is **mitigated, not solved**: 3 → 20 symbols is 6.7× the rows and
     ~1.5× the information, and every stored symbol is BTC beta. `effN = 1.834` still means
     fewer than two independent instruments, so DSR's near-iid assumption stays violated — less
     severely, not acceptably.
  3. `pandas` 3.0.3 makes `Series.pct_change()` unsafe; every return computation in this phase
     uses `s / s.shift(1) - 1.0`, and prefers Phase 1's existing benchmark/equity paths over a
     second implementation.
- **Amended again 2026-07-27 after Phase 1's plan landed.** Three measured inputs, none of which
  relaxes anything:
  1. **The buy-and-hold null can be toothless.** Phase 1 measured the equal-weight basket's Sharpe
     over v0.2.0's 90-day OOS window at **−1.303** — buy-and-hold *lost money*, so both
     `beats_benchmark_*` conditions PASS for free there. Since this campaign's 207-day holdout may
     also be a majors drawdown, the plan now requires the basket's **absolute** numbers beside the
     two bits, tags every verdict with `BENCHMARK_POSITIVE`/`BENCHMARK_NEGATIVE`, and forbids
     reporting a `BENCHMARK_NEGATIVE` pass as northstar evidence. It **deliberately does not** add
     a condition requiring a positive benchmark — that would be moving the gate, which is the one
     thing this phase exists to prevent.
  2. **CAGR ≠ extrapolation.** §0's annualised column is a genuine 3-year CAGR (1095 days);
     only §2's "+68%" is a 90-day extrapolation from 23 trades. The report must label figures by
     kind and never mix them in a column.
  3. **MEDIUM-5's measured effect, checked rather than assumed**: kurtosis 31.2449 → 15.5429,
     skew 3.7883 → 1.4034, OOS Sharpe 1.175 → 1.512. Re-solving the DSR requirement with the
     post-fix moments moves it **7.16 → 6.78 annualised** (~5%, immaterial), and DSR *at* the
     improved Sharpe is **0.883 at `n_trials=1`** (up from 0.742 — the fix genuinely helped) but
     **0.0090 at `n_trials=3011`**. **The `dsr` pre-registration therefore does not move.** The
     useful reading: the obstacle is the size of the search relative to the sample, not the shape
     of the return distribution.
- **`HOLDOUT_END_MS` is exclusive, and that is load-bearing.** The re-measured end of data
  (2026-07-27, correcting shared-contract §0's "2026-07-25") is *today's still-forming* 1d bar,
  which `storage.find_gaps` excludes by construction (`storage.py:164-166`). The last **closed**
  1d bar opens **2026-07-26T00:00:00Z**. Because `storage.load_candles`' bounds are inclusive
  (contract §1), any direct load for the holdout must use `HOLDOUT_END_MS - 1`. Read as inclusive,
  the verdict would silently include a partial day.
- **What success looks like here.** Not a PASS. Success is: a verdict that survives being read
  six months from now by someone who did not run it, because the protocol was committed first,
  the holdout was touched once, every figure has a command, and the limitations document says
  plainly what is still unknown.
