# Known limitations — v0.3.0

**The framework works and the strategy family does not.**

That is the one-sentence result of nine phases. v0.3.0 set out to build a self-learning
pattern framework and to test whether >50% annualized out-of-sample return is achievable.
It built the framework, kept the measuring stick honest, and answered the question: **no.**

Every figure below was produced by a committed command over a stated span. Where a number is
ambiguous or a claim is weaker than it looks, that is said here rather than left for a reader
to discover. This continues v0.2.0's discipline (`git log`: "Use measured rather than derived
figures in the benchmark table") and its companion document,
[KNOWN-LIMITATIONS.md](KNOWN-LIMITATIONS.md), which this file does **not** supersede — the
v0.2.0 findings still stand.

---

## 0. THE ANSWER — >50% annualized OOS is not achievable by this strategy family

Measured on **182 days of genuinely unseen data**, holdout `[1769385600000, 1785110400000)`
= 2026-01-26 → 2026-07-27, end exclusive, 9 symbols:

| | Champion | Buy-and-hold basket |
|---|---|---|
| total return | — | **0.6386×** |
| annualized (182-day extrapolation) | **−45.47%** | **−59.32%** |
| Sharpe | **−0.4720** | **−1.4189** |
| max drawdown | **50.82%** | 45.73% |
| n_days | 182 | 182 |

The champion **lost money outright**. Target was >+50%; it returned −45.47%. Verdict
`B2 / BENCHMARK_NEGATIVE`, CLI exit code 1.

This is a completed, valid outcome, not a failed run. The PRD's honesty clause anticipated it.

## 0a. The two gate conditions it passed are TOOTHLESS — do not read them as success

The gate is 7 conditions. The champion passed 2 of them, and **both are the benchmark
conditions, passed against a basket that lost 36.1% of its value in 182 days.**

```
beats_benchmark_return   -45.47% vs basket -59.32%   > basket   PASS
beats_benchmark_sharpe   -0.4720 vs basket -1.4189   > basket   PASS
```

Independently reproduced: 8 of 9 symbols lost money over the holdout — BCHUSDT 0.3753×,
OPUSDT 0.3180×, XRPUSDT 0.5839×, FILUSDT 0.5862×, AAVEUSDT 0.6525×, BNBUSDT 0.6537×,
BTCUSDT 0.7419×, UNIUSDT 0.8242×. Only TRXUSDT was positive at 1.1181×.

So these passes mean **"lost less money than holding"** and nothing more. This is exactly the
trap the shared architecture contract §4 point 3 named in advance, and it fired. Worse: the
champion's own drawdown (50.82%) was **larger** than the basket's (45.73%), so there is no
axis on which it wins — the same sentence v0.2.0's §0 had to write about BTC buy-and-hold.

**Any future report quoting a `beats_benchmark_*` pass must print the basket's absolute
performance beside it.** The gate now does this by construction, and so does the UI.

## 1. THE GATE FAILS — 5 of 7 conditions, and no bookkeeping choice rescues it

```
sample_adequacy          171 trades                    >= 30       PASS
sharpe                   -0.4720                       >= 1.0      FAIL
dsr                      0.000387                      >  0.95     FAIL
max_drawdown             50.8190%                      <= 25%      FAIL
per_symbol_expectancy    6 of 9 positive               all > 0     FAIL
beats_benchmark_return   -45.47% vs basket -59.32%     >  basket   PASS  (toothless, §0a)
beats_benchmark_sharpe   -0.4720 vs basket -1.4189     >  basket   PASS  (toothless, §0a)
```

Per-symbol OOS expectancy: BTCUSDT −0.7804%, BNBUSDT −0.8092%, UNIUSDT −1.5788% failing;
TRXUSDT +0.1028%, BCHUSDT +0.0611%, XRPUSDT +0.6815%, OPUSDT +0.1806%, AAVEUSDT +0.6698%,
FILUSDT +0.4120% positive.

**The multiple-testing argument is moot here, and that is worth stating plainly.** DSR was
charged 462 trials, but `dsr` fails at *every* candidate trial count including 1, because the
observed Sharpe is **negative** — there is nothing to deflate:

| n_trials | DSR | required ann. Sharpe |
|---|---|---|
| 1 (correction off) | 0.369572 | 2.39 |
| 12 (walkforward default) | 0.022869 | 5.04 |
| 24 (this gate run only) | 0.010368 | 5.59 |
| 198 (Phase 6 alone) | 0.000983 | 7.05 |
| **462 (charged)** | **0.000387** | **7.58** |
| 465 (honest cumulative, §7d) | 0.000385 | 7.59 |

Contrast with v0.2.0, where DSR was the binding constraint and the search size did the
killing. Here the strategy simply loses.

## 2. Sample size is NO LONGER the excuse — and that makes the result stronger, not weaker

For the first time in this project the **independent-equivalent** trade floor cleared:

| Figure | v0.2.0 | v0.3.0 |
|---|---|---|
| raw holdout trades | 23 | **171** |
| mean pairwise r | 0.7574 | **0.4769** |
| Kish effective N | 1.193 | **1.8692** |
| independent-equivalent trades | 9.1 | **35.5** (≥ 30 floor) |

`WF_MIN_TRADES` stayed 30 and the holdout window was never shrunk (verified by grep). Trade
rate on the evolution span was 0.1127/symbol/day, projecting 184.6 holdout trades; 171 were
realized (0.1044) — a projection that held.

**Consequence, stated bluntly: the edge is absent, not merely unproven.** v0.2.0 could
honestly say "too few observations to tell." v0.3.0 cannot. With 171 trades and 35.5
independent-equivalent observations, this is a measurement.

**But effN 1.8692 across 9 symbols still means fewer than two independent instruments.** The
correlation trap is **mitigated, not solved** — 3× the symbols bought ~1.54× the information.
This lever is nearly exhausted; adding more liquid majors adds rows, not information.

## 3. What the gate does not see

| Diagnostic | Measured |
|---|---|
| **full-span max drawdown** (2023-07-27 → 2026-07-27) | **91.97%** |
| holdout-window max drawdown (what the gate saw) | 50.82% |
| full-span trades | 1098 |
| negative fold test expectancy | 7 of 12 |
| folds that fell back to defaults | **0 of 12** (v0.2.0: 4 of 13) |
| holdout daily skew / kurtosis | −0.1868 / 6.8790 |
| per-symbol full-span equity below 1.0× | **6 of 9** (BTC 0.5642×, FIL 0.4078×) |

Two things to take from this. First, v0.2.0's §2 asymmetry **reproduces at larger scale**: the
gate saw 50.82% while the full span was −91.97%. A windowed drawdown is not a risk estimate.
Second, the return distribution is now **well-behaved** (kurtosis 6.88, near-zero skew, versus
v0.2.0's post-fix 15.5429 / 1.4034), so this failure is **not** a moments artifact — the
MEDIUM-5 repair worked and the strategy still loses.

## 4. A correctness fix that FLATTERED the result, logged because it flatters

Phase 1's MEDIUM-5 fix spreads each trade's P&L across its holding days instead of booking it
all on the exit day. Measured effect:

| | exit_day (old) | spread (new) |
|---|---|---|
| daily skew / kurtosis | 3.7883 / 31.2449 | **1.4034 / 15.5429** |
| annualized Sharpe | 1.1750 | **1.5120** |
| DSR at n_trials=1 | 0.742 | **0.8829** |

The fix moves every number in the **favourable** direction, which is exactly the kind of
change that must have its direction recorded rather than buried. Compare v0.2.0 §6, where a
correctness fix *hurt* in-sample performance and was kept anyway. Both were kept for the same
reason: correctness is not negotiated against results. The old behavior stays reachable behind
`attribution="exit_day"` so the delta remains measurable rather than asserted.

## 5. NO DETECTOR HAS AN ESTABLISHED EDGE

19 of 144 catalog rows delivered (13.2%), 6 deferred, 119 out of scope — pinned by a test that
**parses** `.claude/technical-pattern.md` rather than hardcoding the count. Of 16 detectors
measured over 2023-07-27 → 2025-07-26 on BTC/ETH/SOL, **6 had insufficient samples and of the
10 measured, 5 were positive and 5 negative. None has an established edge.**

| detector | tier | trades | hit rate | expectancy after costs | verdict |
|---|---|---|---|---|---|
| cup-and-handle | 1 | 31 | 41.9% | **+2.9618%** | best row, still not an edge (n=31) |
| double-bottom | 2 | 131 | 35.1% | +1.0901% | not established |
| double-top | 2 | 100 | 36.0% | +0.6500% | not established |
| bull-flag | 2 | 276 | 37.0% | +0.5323% | not established (largest sample) |
| rsi-divergence | 2 | 29 | 24.1% | −0.1897% | no edge |
| bear-flag | 2 | 242 | 30.6% | −0.6860% | **no edge**, 182% drawdown |
| head-and-shoulders | 1 | 29 | 24.1% | −1.0696% | **no edge**, loses |
| inverse-cup-and-handle | — | 42 | 19.0% | −1.1095% | **no edge**, loses |

All cost ratios landed in 0.050–0.068 against a 0.10 ceiling, so **the failures are
directional, not cost-driven.** No default was retuned after seeing this table.

Calibration anchor: Phase 4's `detector.macd-cross` measured +0.1939%/trade against a 0.14%
round-trip cost — about 1.4× costs, the same noise band v0.2.0's §0 used to dismiss its entry
edge — and was called no edge. Strip its confirmations and it is −0.5005%, so its apparent
edge came from the gates, not the cross.

## 5a. "Tier 2 = 6 of 6" means IMPLEMENTED, not PRODUCING SIGNALS

This is the most misreadable claim in v0.3.0. Verified independently over the span:

- `falling-wedge`: **0 events**
- `rising-wedge`: **0 events**
- `ascending-triangle`: 1 · `descending-triangle`: 1 · `symmetrical-triangle`: 1

The v0.2.0 tolerances (`TRIANGLE_MIN_CONVERGENCE=0.25`, `CONTAINMENT_TOL=0.005`,
`PATTERN_MAX_AGE_BARS=12` at 4H) starve the family. This is **inherited, not introduced** — the
frozen legacy detector produces 1 triangle trade over the same span. **The tolerances were not
loosened to manufacture events**, which would have spent degrees of freedom for cosmetics.

Related: the wedge geometry was structurally unreachable in v0.2.0. `signals/patterns.py:327`
reads `if upper_end > upper_start or lower_end < lower_start: return out`, which rejects a
rising wedge (both trendlines rise) by construction. That file is **frozen by the parity gate
and was not edited**; the relaxed fit lives in `plugins/detectors/_geometry.py`. **Two geometry
implementations therefore coexist by design and are allowed to disagree** — the legacy path's
job is reproducing v0.2.0 exactly, `_geometry.py`'s job is being correct. Converging them is
only safe once the parity test is retired.

## 6. Evolution works, is honest, and produced nothing that passes

Phase 6's campaign: population 24 × 8 generations = 192 evaluations + 6 audit = 198 trials.
**Zero of 192 candidates reached tier A** (all seven conditions). Phase 9's top-up added 18.

The honesty properties hold and were each verified:

1. **One code path.** Fitness comes only from `walkforward.walk_forward_pooled`. AST-verified:
   `bruteforce`/`run_backtest`/`compute_metrics` reachable from nowhere in evolution;
   `scripts/bruteforce/core.score` is never called.
2. **Scoring without a ledger is structurally impossible.** `ledger` is `GateOracle.__init__`'s
   first positional parameter with no default, every other parameter keyword-only;
   `GateOracle()` raises `TypeError` and `ledger=None` raises `ValueError` citing §4.2.
3. **The holdout was never seen.** Across all 7 campaigns in `trial_ledger`, **zero rows** have
   `end_ms` past the 2026-01-26 barrier. Consumed exactly once.
4. **Fitness is `excess_sharpe`, never DSR** — DSR drifts with trial count, so a generation-1
   candidate would not be comparable to a generation-40 one.
5. **`beats_benchmark_*` demotes rather than eliminates** — 100 of 192 candidates were tier C
   and stayed selectable. Hard elimination would make generation 0 extinct, since the v0.2.0
   seed itself fails it (+3.45% vs +29.4%).
6. **Nothing was softened.** No population shrink, no ledger sampling; `distinct_count` is
   printed labelled "REPORTED ONLY".

## 6a. `excess_sharpe` as fitness REWARDS NOT TRADING — an unresolved design flaw

Measured: tier B averaged **14.9 trades**, tier C **20.1**. Beating a falling benchmark is
easier if you barely trade. Generation 7's winner had **ONE trade** at Sharpe 4.608, DSR
0.9394, and still scored tier B.

`sample_adequacy` catches this at *verdict* time, but **selection does not**, so the population
drifts toward degenerate low-trade graphs.

Phase 6 deliberately did **not** fix this after observing it: tuning search hyper-parameters
against an observed outcome is an unpriced degree of freedom. Phase 9 instead pre-registered a
**champion trade floor** before running and before querying any member's trade count:

```python
CAMPAIGN_MIN_CHAMPION_TRADES = WF_MIN_TRADES   # = 30, the gate's existing floor — no new number
```

It did the intended work: the unfiltered winner had fitness 7.6728 on **12 trades** and was
rejected; only **10 of 200** scored members cleared 30. The champion became a 34-trade member at
fitness 3.4296.

**Still unresolved:** the floor patches *selection*, not *fitness*. The search itself remains
biased toward not trading. Fixing that means changing `excess_sharpe`, which changes the
dynamics of a search already performed and unrepeatable. Whoever runs the next campaign owns
this decision, and it must be pre-registered.

## 7. Known gaps carried forward, each with an owner

### 7a. "One code path" is PARTIAL — two dispatch tables persist
`signals/scan.py` still has its own dispatch table. Live and backtest share **detectors** but
not **dispatch**, which is a real dent in the contract §1 property. Phase 4 declined the
migration as scope creep (`scan_symbol` returns `list[Signal]` with no `EvalSession` or span, so
migrating means authoring a second live-evaluator entry point) and recommended Phase 7 own it.
**Phase 7's agent stalled before deciding, so this is UNOWNED, not resolved.** `TestScanDriftGuard`
holds the line: it asserts branch↔scanner correspondence for all four regimes, and `FADE_ENABLED`
is read at call time so both paths suppress together.

### 7b. Cost arithmetic is duplicated
`framework/execute._close_out` duplicates `engine.close_out`. Pinned by `TestCostArithmetic`
rather than by construction, so **when one changes, both must.** Consumed technical debt.

### 7c. The UI composer can edit only single-branch strategies — 3 of the 4 committed are read-only
`_graph_to_stages` returns `editable=False` for any graph with more than one branch or a
disabled branch (`ui/api.py:356`), so the page renders read-only rather than silently
flattening and re-saving.

**AMENDED 2026-07-28.** As first written this section said "cannot edit *any* strategy
committed today", naming three strategies. That was true at `703cc91` and was invalidated by
the *later* UI rebuild (`019dbb2`), which committed a fourth, single-branch strategy.
Re-measured against the current tree:

| strategy | branches | editable |
|---|---|---|
| `donchian-v020` | 2 | no |
| `thin-slice` | 2 | no |
| `thin-slice-noconfirm` | 2 | no |
| `tech-pattern` | **1** | **yes** (5 stages) |

`tech-pattern` also **round-trips to a bit-identical `graph_hash`** (`6940db10f4632cd9` both
before and after a composer load/save), so the one editable strategy is edited losslessly
rather than approximately — the property the read-only guard exists to protect.

The limitation is therefore narrower than originally stated but not gone: the composer is
still a *linear* editor, and the three multi-branch (regime-dispatching) strategies remain
uneditable. Silent flattening would have been worse; this is a usability gap, not a
correctness one.

### 7d. The trial ledger under-records its own final step
Phase 9's gate run received `n_trials=` but not `ledger=`, so its 24 fold evaluations were
**charged** to the DSR without being **persisted** as rows. Charge 462 exceeds recorded rows
441. The direction is **pessimistic** — nothing was flattered — and the defect is fixed and
pinned by a regression test, but the holdout was not re-run to repair the bookkeeping, because
re-running it would spend the holdout a second time.

### 7e. Confirmation usefulness is UNMEASURABLE from closed trades
Confirmations are hard gates (`execute.py` returns `None` on the first failure), so a trade
exists only if every confirmation passed. Coverage is therefore necessarily 1.00 **by
construction** and there is no counterfactual to compare against — the gate destroyed it.

This produced a live bug worth recording: the review layer originally labelled full coverage
"DEAD WEIGHT" and suggested dropping the confirmations, and `--loop` acted on it. That
suggestion is **measurably harmful** — Phase 4 measured 276 trades/+0.2028%/Sharpe +0.53 with
the confirmations versus 477/−0.2523%/−0.86 without. Now fixed: full coverage with no
counterfactual is labelled `unmeasurable` and never yields a drop suggestion, and a decided
`dead-weight` verdict is reachable only where a real counterfactual exists.

**The missing capability is recording rejected events.** Until something logs them, hard gates
cannot be assessed honestly. Phase 4 independently hit the same wall.

### 7f. Multi-position margin and liquidation accounting is OUT OF SCOPE
v0.3.0 keeps **one open trade per symbol**, equal notional, no position sizing. This is the
known gap between the pivot guide's "trust the signal" principle and the engine's one-open-trade
rule. Also inherited unchanged and not reopened: alert-only (no order placement),
`FADE_ENABLED = False`, funding as a frozen pessimistic placeholder.

## 8. Tooling debt

No linter and no type checker are configured. Validation is `.venv/bin/python -m pytest -q`
plus `python -m py_compile`. This is stated rather than papered over with an invented command.

`pandas 3.0.3` changed `Series.pct_change()` behavior; it is **unsafe here** and returns are
computed with explicit arithmetic (`s / s.shift(1) - 1.0`). `pandas-ta` is gone from PyPI and
TA-Lib needs a C library, so the indicator set is hand-rolled and **no new indicator dependency
may be added.**

**A cross-agent hazard, recorded:** `data/state.db` was truncated to 0 bytes mid-session by a
concurrent process, destroying one campaign's evidence (recovered additively from a salvage
copy; pre-merge state at `data/state-pre-phase9-backup.db`). The 117 MB of irreplaceable price
history in `ohlcv.db` was never at risk — precisely why contract §6 put framework state in a
**separate** database. The design decision paid for itself.

**Five separate phases found their own plans instructing them to use raw `MAX(ts)`** for a span
boundary, which would have treated a forming or partial candle as closed — a lookahead bug that
fails in the flattering direction. All five deviated correctly. Phase 9's was the most
consequential: its plan declared the holdout start as 2026-01-01, which would have handed
evolution **25 days it had already scored.** Corrected to Phase 6's measured ceiling of
2026-01-26.

Related and subtle: the stored 2026-07-27 daily bar is a **partial** bar, not merely a forming
one — polling stopped at 06:00 UTC, so it holds 7 hours of trade (volume 25535.7 vs the prior
day's 41197.3, its high/low exactly the extremes of the seven 1h bars that exist). The last
*complete* daily bar opens 2026-07-26.

## 9. Degrees of freedom consumed

| Phase | DoF |
|---|---|
| 1 Validation integrity | **zero** — no threshold, grid axis or window changed; the condition *set* grew 5 → 7 |
| 2 Data breadth | **zero** — pre-registered selection rule, no backfill |
| 3 Plug-in framework | **zero strategy DoF** — `ParamSpec` bounds *declared* on 17 parameters (summed 383 combinations), **none searched** |
| 4 Thin slice | 2 structural, both fixed before measurement (`target_enabled=True`; `macd-cross` on `("any",)`) |
| 5 Feedback loop | **zero** — every `REVIEW_*` constant labels a trade; none entered a grid |
| 6 Evolution | 198 evaluations (192 + 6 audit) |
| 8 Pattern coverage | no default retuned after seeing the edge table |
| 9 Campaign | 18 top-up + 24 gate folds + 6 probes; trade floor **reused** the gate's existing 30 |
| **Total charged to DSR** | **462** (honest cumulative 465, §7d) |

Holdout start, end and length cost **zero** — both endpoints were forced by measurement rather
than chosen. Symbol set cost zero — Phase 2's pre-registered selection. No config value was
changed after the probe.

## What IS established

Stated separately from the strategy result, because these are real and were verified
independently rather than accepted from a report:

1. **The framework wraps the honest core rather than replacing it.** The v0.2.0 Donchian
   strategy expressed as a serialized graph reproduces `engine.run_backtest` **bit-identically**:
   43/55/60 = 158 trades pooled, **zero** non-identical values across all 13 simulation fields.
   Not "within 1e-12" — exactly zero.
2. **Adding a plug-in requires zero engine-core edits** (the PRD's success metric). Demonstrated
   by registering a throwaway detector that appeared in the CLI with the tree byte-for-byte
   unchanged. `engine.py`, `signals/`, `regime/`, `indicators/`, `risk/`, `metrics.py` and
   `equity.py` show an empty diff across Phase 3.
3. **The measuring stick is honest.** The buy-and-hold null reproduces v0.2.0's published §0
   table from **one command** (BTC 2.2104×/+30.26%/0.799/52.98%; basket 2.1628×/+29.32%/0.728/
   64.32%), every cell within a rounding unit.
4. **The trial ledger survives process restarts** — verified incrementing 1 → 2 across two
   separate CLI invocations, and 417 rows persisted across a truncation accident.
5. **The holdout barrier held.** Zero of 462 charged evaluations touched data past 2026-01-26.
6. **The independent-equivalent sample floor cleared for the first time** (35.5 ≥ 30), which is
   what makes §0's failure a measurement rather than a shrug.
7. **1252 tests pass**, including all 286 pre-existing tests unchanged. (1249 when this
   document was written at `703cc91`; the two subsequent UI commits added 3. Re-measured
   2026-07-28: `.venv/bin/python -m pytest -q -m "not network"` → 1252 passed, 2 deselected
   in 78.76 s. The 2 deselected are the network tests, which need a reachable
   `fapi.binance.com`.)

## The next four things, in order

Pre-committed for verdict `B2`, so the next run cannot be chosen after seeing these results:

1. **Follow the non-DSR failures.** `per_symbol_expectancy` failed 3 of 9 (BTC −0.78%, BNB
   −0.81%, UNI −1.58%) — the edge is concentrated, so revisit Phase 8 detector selection.
2. **Fix the equity path.** 50.82% holdout drawdown against a 25% ceiling, and 91.97% full-span,
   is unacceptable regardless of mean return.
3. **Decide `excess_sharpe` vs the trade floor (§6a)** and pre-register it before searching.
4. **Record rejected events (§7e)** so confirmation usefulness becomes decidable at all.

**Explicitly forbidden as a response to this result:** re-running with a lower `n_trials`, a
lower trade floor, a shorter holdout, a different champion rule, or "one more generation." A
second run is a **new campaign**, must be stamped `NOT A CLEAN HOLDOUT`, and must be charged the
combined trial count. The holdout has been consumed once and that is recorded in
`holdout_consumption`.
