# Phase 4 — Strategy Pipeline Thin Slice (v0.3.0)

Measured 2026-07-27 on `data/ohlcv.db`, branch `feat/v0.3.0`, Python 3.11.6,
pandas 3.0.3, pytest 9.1.1.

**Every number below comes from one of the five commands in §0.** No spreadsheet
arithmetic, no "approximately". Contract §12.5.

**Nothing here queries the gate.** Contract §4: fitness comes only from
`walkforward.walk_forward_pooled`. These are IN-SAMPLE diagnostics on stored
history and none may be reported as a validated result (KNOWN-LIMITATIONS §9).
Phase 1's trial ledger is not incremented, because none of these runs queried the
oracle.

---

## 0. The commands

Span `2023-07-27 → 2026-07-26` throughout — the span KNOWN-LIMITATIONS §0 uses, so
the figures are comparable to the v0.2.0 record. The end bound is a **closed** bar:
today is 2026-07-27, so the 1d bar opening 2026-07-27 00:00 is still forming and is
excluded.

```bash
SPAN="--start 2023-07-27 --end 2026-07-26"
SYMS="--symbol BTCUSDT --symbol ETHUSDT --symbol SOLUSDT"

# (1) R:R survival — run FIRST, before any conclusion about the filter
python -m trading_bot.cli graph-backtest --graph data/strategies/thin-slice.strategy.json $SYMS $SPAN --rr-report
# (2) full pipeline with the audit trail
python -m trading_bot.cli graph-backtest --graph data/strategies/thin-slice.strategy.json $SYMS $SPAN --audit
# (3) ablation: no Confirmations
python -m trading_bot.cli graph-backtest --graph data/strategies/thin-slice-noconfirm.strategy.json $SYMS $SPAN
# (4) the v0.2.0 reference point, unchanged code path, same span
python -m trading_bot.cli backtest $SYMS $SPAN
# (5) plug-in registration
python -m trading_bot.cli plugins
```

All five exit 0.

---

## A. R:R survival — **the prediction is REFUTED**

The plan predicted, in writing and before measurement, that "a 2.0 net floor may
reject nearly every plan the thin slice produces." **It does not.**

| | `n_plans` | `n_pass(2.0)` | `survival_rate` | net min / median / max |
|---|---|---|---|---|
| BTCUSDT | 125 | 87 | **69.60%** | 0.3690 / 2.4134 / 4.9428 |
| ETHUSDT | 132 | 91 | **68.94%** | 0.5842 / 2.4174 / 5.0594 |
| SOLUSDT | 126 | 101 | **80.16%** | 0.4258 / 2.7315 / 4.9105 |
| **POOLED** | **383** | **279** | **72.85%** | 0.3690 / 2.5224 / 5.0594 |

Pooled `n_pass` by NET threshold (**diagnosis only** — printing 1.5 is not
permission to use 1.5): `2.00:279  1.75:315  1.50:345  1.25:367  1.00:378`.

Pooled deciles (10%…90%):

- net &nbsp;&nbsp;`1.508 1.834 2.057 2.271 2.522 2.820 3.136 3.460 3.968`
- gross `1.637 2.036 2.265 2.457 2.718 3.032 3.379 3.687 4.155`

### Decision-rule branch: **branch 1 — proceed**

`survival_rate = 72.85% ≥ 0.20` **and** `N2 = 279 ≥ 30`. Per the pre-registered
rule, `RR_TARGET_MIN` stays 2.0 and Phase 4 succeeds as specified. No remedy is
needed and none was applied.

### The gross-floor prediction is CONFIRMED

The plan predicted a required gross floor of ≈2.11–2.21. Measured
`gross_rr_required` medians:

| Symbol | predicted (from config's median `risk_pct`) | measured median | measured min / max |
|---|---|---|---|
| BTCUSDT | 2.2134 | **2.2178** | 2.0737 / 2.5231 |
| ETHUSDT | 2.1568 | **2.1523** | 2.0601 / 2.4093 |
| SOLUSDT | 2.1119 | **2.1187** | 2.0436 / 2.2200 |
| POOLED | — | 2.1524 | 2.0436 / 2.5231 |

So the **algebra** was right to three decimals and the **inference from it** was
wrong.

### Why the prediction failed — a finding about the reasoning, not the code

The plan reasoned from `walkforward.py`'s recorded fact that "the minimum planned
gross R:R across every trade the engine ever took was 1.56", and treated 1.56 as
representative. **It is a floor artifact, not a central tendency.** v0.2.0's
`build_signal` screens `rr < RR_FLOOR = 1.5`, so 1.56 is simply the smallest value
that survived that screen; the measured gross distribution has a pooled **median of
2.7177**, comfortably above the ~2.15 the net-2.0 floor demands. Confusing a
measured minimum with a measured distribution is the error, and it is recorded here
so the next phase does not repeat it.

Two secondary contributors, both stated for completeness: the thin slice sets
`target_enabled=True` (v0.2.0 leaves it False), and the `macd-cross` detector
supplies additional plans whose channel-width targets are drawn from the same
distribution.

---

## B. Volume / MACD gate cost — the gates **help**, substantially

Runs (2) vs (3). Identical graphs; the only difference is the two Confirmation
nodes.

| | `n_trades` | `expectancy_pct` | pooled `sharpe` | pooled `ann_return_pct` | pooled `max_drawdown_pct` (equity) |
|---|---|---|---|---|---|
| no Confirmations | 477 | −0.2523% | −0.86 | −40.02% | 81.06% |
| + volume + MACD | 276 | **+0.2028%** | **+0.53** | **+13.24%** | **36.16%** |
| Δ | −201 | +0.4551 pp | +1.39 | +53.26 pp | −44.90 pp |

Per symbol:

| Symbol | | `n_trades` | `expectancy_pct` | `sharpe` | `ann_return_pct` |
|---|---|---|---|---|---|
| BTCUSDT | no conf. | 140 | −0.1926% | −0.57 | −9.72% |
| BTCUSDT | + conf. | 87 | +0.0455% | +0.11 | +0.66% |
| ETHUSDT | no conf. | 152 | +0.0697% | +0.15 | +0.97% |
| ETHUSDT | + conf. | 90 | +0.6602% | +1.11 | +20.00% |
| SOLUSDT | no conf. | 185 | −0.5619% | −1.12 | −32.62% |
| SOLUSDT | + conf. | 99 | −0.0749% | −0.10 | −5.14% |

The gates improve expectancy on **all three** symbols and flip the pooled sign.
This is the first entry-side change in the project's history
(KNOWN-LIMITATIONS §0c: no sweep ever touched the entry or the feature set), and
it moved the needle. **It is in-sample and it is not a validated result** — the
gate has not been run. It is a reason for Phase 6 to put these on a grid axis, not
a claim of edge.

### The graded value D3 promised to report

`ConfirmationVerdict.score` carries the volume ratio in every branch. Measured on
BTCUSDT over the same span via `run_graph_backtest`:

| Graph | trades | `volume_high` | share |
|---|---|---|---|
| thin-slice (gate on) | 87 | 87 | **100.0%** |
| thin-slice-noconfirm | 140 | 75 | 53.6% |

So the volume gate is exactly the `VOLUME_HIGH_RATIO` threshold, and it removed the
46.4% of would-be BTC positions that had sub-average participation.

**Not measured, stated as a gap:** the full ratio distribution among *rejected*
events. Only `filter.rr-after-costs` has a verdict recorder; the Confirmations do
not. `ConfirmationVerdict.score` is the channel Phase 5's Reviewer will read.

---

## C. Detector edge report (contract §9: detection ≠ edge)

Pooled across 3 symbols and all regimes, from run (2)'s
`by detector (pooled across symbols and regimes)` block. **Expectancy is after
costs** — `Trade.pnl_pct` is net of `2*(fee+slippage)` plus
`FUNDING_PCT_PER_DAY * hold_days`.

| Detector | tier | `n_trades` | hit rate | expectancy after costs | `profit_factor` | `max_dd` |
|---|---|---|---|---|---|---|
| `donchian-breakout` | 2 | 111 | 33.33% | **+0.2159%** | 1.10 | 36.89% |
| `macd-cross` | **3** | 165 | 38.18% | **+0.1939%** | 1.10 | 70.07% |

### Does `macd-cross` have an edge? **Stated plainly: no — not one this measurement can establish.**

- Its expectancy is **+0.1939% per trade after costs**, positive but tiny: about
  **1.4× the 0.14% round-trip cost**, and squarely in the same "noise" territory
  KNOWN-LIMITATIONS §0 used to dismiss v0.2.0's 15-basis-point entry edge.
- `profit_factor` 1.10 on 165 trades is not separable from zero at this sample
  size, and contract §0a measures the effective independent sample at
  **effN ≈ 1.83**, so 165 pooled trades are worth roughly 165 × 1.83 / 3 ≈ 101
  independent-equivalent observations — not enough to distinguish 1.10 from 1.00.
- Its **maximum drawdown is 70.07%, nearly double** the Donchian detector's
  36.89%, for the same profit factor. On a risk-adjusted basis it is the worse of
  the two.
- **Without** the Confirmations its expectancy is **−0.5005%** (330 trades,
  PF 0.77). So essentially all of its apparent edge is contributed by the two gates,
  not by the cross itself.

This is the expected outcome and it costs the phase nothing: contract §9 makes
tiers 3–4 "a direction, not a gate", and **no Phase 4 success criterion depends on
`macd-cross` having edge**. It is in the graph to prove the framework composes more
than one detector, which it does. Its registry `rationale` states the Tier-3 prior
honestly before any measurement.

For comparison, run (4) — the v0.2.0 reference on the identical span, unchanged
code path: BTC 43 trades / +1.1651% / sharpe 1.69; ETH 55 / −0.1003% / −0.14;
SOL 60 / −0.0481% / −0.05.

---

## D. Cost-frontier check

Median `risk_pct` per symbol computed from run (2)'s `--audit` lines (276 taken
positions), and `c = risk.atr_stop.cost_ratio(median_risk_pct, FEE_PCT, SLIPPAGE_PCT)`.

| Symbol | n | median `risk_pct` | `c` | `COST_RATIO_CEILING` | verdict |
|---|---|---|---|---|---|
| BTCUSDT | 87 | 2.0132% | 0.0695 | 0.10 | **PASS** |
| ETHUSDT | 90 | 2.7976% | 0.0500 | 0.10 | **PASS** |
| SOLUSDT | 99 | 3.4865% | 0.0402 | 0.10 | **PASS** |
| POOLED | 276 | 2.8256% | 0.0495 | 0.10 | **PASS** |

All three clear the ceiling, and they reproduce `config.py:166-168`'s recorded
medians (1.968% / 2.679% / 3.752%) closely. `k` was **not** touched
(`config.py:169-173`: raising ATR by coarsening the tier delivered the ceiling,
"not widening k, which would fit the stop to the fee schedule").

Caveat, repeated because it is load-bearing: `cost_ratio` counts fee + slippage
only and **understates** realized cost, which also includes funding.

---

## E. PRD Success Metric row 4

> "100% of taken positions pass ≥1:2 R:R after costs at entry."

**276 / 276 taken positions have `planned_rr ≥ 2.0`**, printed by run (2) as
`planned_rr >= RR_TARGET_MIN (2.0) on 276/276 taken positions`. Verified from the
command's output, not from the code, and asserted independently by
`tests/test_pipeline_thin_slice.py::TestThinSliceProducesTrades::test_every_taken_position_clears_RR_TARGET_MIN`.

---

## F. Degrees of freedom consumed

| Item | Kind | Count | Selected on returns? |
|---|---|---|---|
| volume gate on/off | binary ablation | 2 runs | **No** — both reported |
| MACD gate on/off | binary ablation | (same 2 runs) | **No** |
| `VOLUME_CONFIRM_MIN_RATIO` | reused `VOLUME_HIGH_RATIO` by reference | 0 new values | No |
| `MACD_CONFIRM_MIN_HIST` | 0.0, pure sign test | 0 new values | No |
| MACD 12/26/9 | canonical, inherited from the donor | 0 | No |
| `RR_TARGET_MIN` | derived from the PRD requirement + cost algebra | 1 constant, not swept | **No** |
| `macd-cross` `target_height` | derived (canonical 20-bar channel width) | 0 new params | No |
| `macd-cross` `level` | derived (crossing bar's high/low) | 0 new params | No |
| `target_enabled = True` in both thin-slice branches | **1 structural exit choice** | 1 | **No** — see below |
| `macd-cross` branch `regimes = ("any",)` | **1 structural routing choice** | 1 | **No** — see below |
| Total graph evaluations | 4 | — | none selected |

**No parameter in this phase was chosen by search. The four runs are diagnostic
ablations and all four are reported, so none constitutes a hidden selection.
Phase 1's trial ledger is not incremented, because none of these runs queried the
gate oracle.**

Two structural choices are recorded above as consumed degrees of freedom because
they are decisions the plan did not pre-specify, and both were fixed **before** any
measurement:

1. `target_enabled = True` on both thin-slice branches (v0.2.0 defaults it False).
   Reason: an R:R filter requiring reward ≥ 2× risk after costs is vacuous if the
   reward leg is never realizable as an exit. Not tried both ways.
2. `macd-cross` routed on `("any",)` while `donchian-breakout` keeps
   `("trending",)`. Reason: the Donchian branch is the parity anchor and must keep
   v0.2.0's routing; the MACD branch exists to exercise multi-branch composition,
   and restricting it to one regime would have made the second branch nearly
   inert. Not tried both ways.

---

## G. Frozen-surface audit

```
$ git diff --stat src/trading_bot/
 src/trading_bot/backtest/engine.py   |  27 ++++
 src/trading_bot/cli.py               | 249 +++++++++++++++++++++++++++++++++++
 src/trading_bot/config.py            |  72 ++++++++++
 src/trading_bot/framework/execute.py |  40 ++++--
 4 files changed, 378 insertions(+), 10 deletions(-)
```

Exactly the four files the plan authorizes. **Untouched:** every module under
`signals/`, `risk/`, `regime/`, `data/`, plus `backtest/metrics.py`,
`backtest/equity.py`, `backtest/walkforward.py`, `indicators/wilder.py`,
`indicators/bollinger.py`, `indicators/donchian.py`.

- `engine.py`: three appended `Trade` fields with defaults + docstring. Nothing
  else. First 13 fields unchanged and in order (asserted by test).
- `framework/execute.py`: one import, three keyword arguments at the single
  `Trade(...)` site, and the A11 docstring updated to record what was done. The
  10 deletions are the replaced placeholder comment.
- `config.py`: additions only, 72 lines, appended after `RESEARCH_SYMBOLS` and
  before `date_to_ms`. `RR_FLOOR = 1.5` unmodified.
- `cli.py`: one subparser, one dispatch branch, one handler + three print helpers.

---

## H. Honest gaps and deviations

1. **`signals/scan.py` dispatch — NOT migrated. Deliberate, see §I.**
2. **The volume-ratio distribution among rejected events is not measured** (§B).
3. **`filter.rr-after-costs` ACCEPTS a `risk_pct == 0` plan.** `net_rr`'s
   `risk + cost` denominator floors risk at the cost term, so a zero-risk plan
   scores 34.71 instead of being rejected, while `cost_ratio` correctly reports
   `+inf`. **Unreachable through the pipeline** — `policy.measured-move` rejects
   `risk <= 0` first — and pinned as the actual behavior by
   `tests/test_plugins_filters.py::TestDegenerate`. Not "fixed", because Task 10's
   GOTCHA 1 forbids adding floors to this filter.
4. **The exact `net_rr == RR_TARGET_MIN` boundary is not reliably reachable in
   IEEE-754.** Constructing `reward = (X + (X+1)c) * risk` lands a few ulps either
   side of X depending on `c`, so `>=` at the constructed boundary is False for
   some inputs (measured at `risk_pct` 0.005 with X=2.0, and at 0.10 for both X).
   The algebra is asserted to 1e-12 and the *decision* is asserted just above and
   just below. A property of floating point, not of the filter.
5. **Funding is not charged at signal time**, by design and necessity — hold
   duration is unknowable at entry. The filter is therefore slightly **permissive**,
   never strict. The engine still charges funding on the realized trade.

---

## I. Decision on the `signals/scan.py` dispatch gap

**Decision: NOT migrated in Phase 4. `TestScanDriftGuard` stays in place.**

Phase 3 recommended Phase 4 take it, since Phase 4 owns `graph-backtest`. Declined,
for four reasons:

1. **It is not the same shape of problem.** `scan_symbol` returns `list[Signal]`
   for alerting. It has no `EvalSession`, no `start_ms/end_ms`, no `Trade`, and no
   replay loop. Migrating it is not "reuse `run_graph_backtest`" — it is authoring
   a *second* graph entry point (a live, single-instant evaluator) and giving it
   its own no-lookahead argument. That is a phase-sized piece of work, not a task.
2. **It would put an unvalidated path in front of the only thing the operator
   acts on.** The live scan produces the alerts. Phase 4's validation surface is
   backtest parity plus unit tests; neither covers live alerting. Rewiring the
   alert path on the strength of a backtest gate is precisely the trade Phase 3's
   A6 declined to make, and nothing changed in Phase 4 to make it safer.
3. **Phase 4 has a strictly larger blast radius than Phase 3 did.** This phase
   already edits the frozen `Trade` dataclass and Phase 3's `execute.py`. Adding a
   live-path rewrite on top of the one edit the parity test is most sensitive to
   trades a real deliverable for a cleanup.
4. **The risk is currently guarded and cheap to keep guarding.**
   `TestScanDriftGuard` asserts that for every label in `classifier.REGIMES` the
   v0.2.0 graph's active branch matches what `scan.scan_symbol` dispatches, and
   `FADE_ENABLED` is read at call time inside the detector so the kill switch
   cannot desync. Full-history parity is itself a drift detector. The guard is
   green.

**Consequence, stated rather than hidden:** contract §1's "one code path" property
remains *partial*. Live and backtest share detectors, policies and the trigger rule,
but not dispatch. There are still two dispatch tables — `scan.py`'s regime→scanner
map and `Branch.regimes` — and they can disagree in ways only the drift guard
catches. **Recommended owner: Phase 7**, which builds the operator UI and therefore
has to answer "what will this graph alert on?" anyway, making the live evaluator a
deliverable rather than a chore.

**Cost arithmetic duplication (the second gap):** I touched
`framework/execute._close_out` but **not** its cost arithmetic — `gross`, `cost`
and `funding_cost` are byte-identical to before. The added `planned_rr` term calls
`risk.atr_stop.net_rr`, the single frozen cost definition, and adds no local cost
computation. `engine.close_out` is untouched, so the two remain in agreement and
`TestCostArithmetic` is green.

---

## J. Validation

- Full suite: **727 passed, 2 skipped** in 41.58s (baseline 597 passed / 2 skipped
  → **+130 new tests**; 729 collected).
- `python -m py_compile` on all 18 new/changed files: **clean, 0 syntax errors**.
- **No linter and no type checker are configured in this repo**
  (KNOWN-LIMITATIONS §8). `py_compile` + `pytest` is the whole validation surface;
  no `mypy`/`ruff` step was invented.
- Phase 3's parity test: **43 passed**, and re-verified directly after the `Trade`
  change — BTC 43, ETH 55, SOL 60 = **158 pooled trades, zero non-identical
  simulation field values**.
- `graph-validate` on both committed graphs: exit 0.
  `thin-slice` hash `1addaebb5427`, `thin-slice-noconfirm` hash `af4b5bdb551d`.
- One pre-existing test widened: `tests/test_framework_registry.py::TestLookup::
  test_by_kind_is_sorted_by_name` asserted an *exhaustive* detector list, which any
  phase adding a detector breaks. Narrowed to sortedness + containment, matching
  the forward-compatible shape `TestLoadAll` already used.
