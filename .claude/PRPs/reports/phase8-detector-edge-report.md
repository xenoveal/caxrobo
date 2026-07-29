# Phase 8 — Pattern Coverage Expansion: detector edge report

**Status: COMPLETE.** 16 detectors registered, contract §9 tier 2 closed at 6/6, tier 1 at 2/3
with Wyckoff Accumulation/Distribution `deferred`. 144/144 catalog rows enumerated and
machine-checked against `.claude/technical-pattern.md`.

**Every number in this document was produced by a command printed beside it.** Nothing is
derived, extrapolated or estimated — the discipline `git log` already records this repo
committing to once ("Use measured rather than derived figures in the benchmark table").

Measured 2026-07-27 on `data/ohlcv.db`, Python 3.11.6, pandas 3.0.3.

---

## 0. The honest summary, before the tables

1. **Detection works.** All 16 geometries are found on bars built so the answer is known by
   construction, and rejected on bars built to fail one clause at a time. 176 new tests.
2. **Edge is mostly absent, and where present it is not established.** Of 16 detectors, 10
   produced enough trades to print a rate at all; of those 10, **5 have a positive after-costs
   expectancy and 5 are negative**. None of those numbers is a gate verdict, none survives a
   multiple-testing correction, and the largest is drawn from 31 trades.
3. **Two tier-2 detectors that were structurally impossible are now expressible and produce
   ZERO trades on this data.** `falling-wedge` and `rising-wedge` were unreachable through
   `signals/patterns.py:327`; relaxing the guard makes them emittable, and they still fire
   nothing over two years of three symbols. That is a real finding, not a bug — see §4.
4. **The single biggest caveat on this whole table**: `detector-report` measures each detector
   with the regime gate OFF, no confirmations and no filters, on the tuning span. It is a
   diagnostic of the ENTRY, which KNOWN-LIMITATIONS §0c records as never having been explored.
   It is not a strategy evaluation, and contract §4 reserves that verdict for the gate.

---

## 1. The measured edge table

Generating command, run verbatim:

```
.venv/bin/python -m trading_bot.cli detector-report \
  --start 2023-07-27 --end 2025-07-26 \
  --state-db /tmp/p8final2.db \
  --out .claude/PRPs/reports/phase8-detector-report.out.md
```

Output (verbatim; also written to `.claude/PRPs/reports/phase8-detector-report.out.md`):

```
=== DIAGNOSTIC -- NOT A GATE VERDICT =====================================
 span 2023-07-27..2025-07-26 (TUNING span; holdout guard: 150d reserved)
 symbols BTCUSDT,ETHUSDT,SOLUSDT
 fixed exit policy, NOTHING SWEPT.  every detector at its ParamSpec defaults.
 ledger campaign="detector-report" rows=+16 (campaign total 16)
 SELECTION COST IF USED: if any detector is dropped from a campaign because of
 this table, add n_trials += 16 to that campaign. Reading it is free; letting it
 inform selection is not.
 rows sorted by TIER then NAME -- never by expectancy, because sorting by
 expectancy IS selection.
==========================================================================

detector                     tier  trades    win%      exp%     pf   maxdd%       c  verdict
--------------------------------------------------------------------------------------------------------
cup-and-handle                  1      31   41.9%  +2.9618%   2.88   20.39%   0.068  MEASURED
head-and-shoulders              1      29   24.1%  -1.0696%   0.60   46.76%   0.052  MEASURED
inverse-head-and-shoulders      1      28   17.9%  +0.1343%   1.06   34.38%   0.067  MEASURED
wyckoff-spring                  1       5      --        --     --       --      --  INSUFFICIENT(<20)
wyckoff-upthrust                1       5      --        --     --       --      --  INSUFFICIENT(<20)
ascending-triangle              2       1      --        --     --       --      --  INSUFFICIENT(<20)
bear-flag                       2     242   30.6%  -0.6860%   0.72  181.98%   0.051  MEASURED
bull-flag                       2     276   37.0%  +0.5323%   1.25   51.55%   0.056  MEASURED
double-bottom                   2     131   35.1%  +1.0901%   1.62   31.69%   0.065  MEASURED
double-top                      2     100   36.0%  +0.6500%   1.31   36.20%   0.053  MEASURED
falling-wedge                   2       0      --        --     --       --      --  INSUFFICIENT(<20)
rising-wedge                    2       0      --        --     --       --      --  INSUFFICIENT(<20)
rsi-divergence                  2      29   24.1%  -0.1897%   0.93   34.79%   0.050  MEASURED
descending-triangle            --       1      --        --     --       --      --  INSUFFICIENT(<20)
inverse-cup-and-handle         --      42   19.0%  -1.1095%   0.56   46.60%   0.050  MEASURED
symmetrical-triangle           --       1      --        --     --       --      --  INSUFFICIENT(<20)
--------------------------------------------------------------------------------------------------------
 c = mean round-trip cost / median risk_pct, against config.COST_RATIO_CEILING = 0.1. A detector whose c exceeds
 the ceiling is STRUCTURALLY unable to pay for itself regardless of win rate.
 exp% is NET of costs: run_graph_backtest charged 2*(FEE_PCT+SLIPPAGE_PCT) + FUNDING_PCT_PER_DAY*days,
 exactly as production charges it. No second P&L path exists in this command.
```

### 1a. Per-detector verdict, in plain words

| detector | trades | hit rate | expectancy after costs | c | plain verdict |
|---|---|---|---|---|---|
| `cup-and-handle` | 31 | 41.9% | **+2.9618%** | 0.068 | Best row in the table, and **not** an edge. 31 trades is above the report's own floor of 20 but below `WF_MIN_TRADES = 30` by one; a +2.96% mean on 31 observations with a 2.88 profit factor is the kind of number that vanishes on a holdout. Worth Phase 9's attention; worth nothing else. |
| `head-and-shoulders` | 29 | 24.1% | −1.0696% | 0.052 | **No edge.** Loses. Consistent with KNOWN-LIMITATIONS §8's record of the v0.2.0 geometry losing; the ATR risk model did not rescue it, and all three refinements were OFF, so the base shape is what was measured. |
| `inverse-head-and-shoulders` | 28 | 17.9% | +0.1343% | 0.067 | **No edge.** +0.13% against a 0.14% round-trip cost is inside the noise, and this is exactly the calibration Phase 4 used to call `detector.macd-cross` (+0.1939%) no edge. Same verdict, weaker number. |
| `wyckoff-spring` | 5 | — | — | — | **INSUFFICIENT.** 5 trades. A sample finding, and NOT a licence to loosen `WYCKOFF_PROBE_VOL_RATIO`. |
| `wyckoff-upthrust` | 5 | — | — | — | **INSUFFICIENT.** As above. |
| `ascending-triangle` | 1 | — | — | — | **INSUFFICIENT.** See §4 — the converging-line family is sample-starved at the 4H setup tier, and the frozen legacy detector is too. |
| `bear-flag` | 242 | 30.6% | −0.6860% | 0.051 | **No edge, and clearly negative.** 181.98% max drawdown on the trade series. The largest sample in the table and the worst outcome; the v0.2.0 accounting that flags dominated trade volume and lost is reproduced. |
| `bull-flag` | 276 | 37.0% | +0.5323% | 0.056 | **Not established.** The largest sample (276) with a positive mean, profit factor 1.25. This is the only row where the sample is large enough for the number to mean much, and 0.53% per trade against a 0.14% cost is a thin but real-looking margin. It is a candidate for Phase 9, not a result. |
| `double-bottom` | 131 | 35.1% | **+1.0901%** | 0.065 | **Not established, but the most interesting row.** 131 trades, profit factor 1.62, drawdown 31.69%. Newly expressible (the donor had no level and no target). Phase 9 should test it. |
| `double-top` | 100 | 36.0% | +0.6500% | 0.053 | **Not established.** Positive on 100 trades, weaker than its mirror. The long/short asymmetry across the pairs (`double-bottom` > `double-top`, `bull-flag` > `bear-flag`, `cup` > `inverse-cup`) is consistent with a bull-biased sample and is a reason to distrust all four positive numbers. |
| `falling-wedge` | 0 | — | — | — | **INSUFFICIENT — zero events.** Newly expressible; fires nothing. See §4. |
| `rising-wedge` | 0 | — | — | — | **INSUFFICIENT — zero events.** As above. |
| `rsi-divergence` | 29 | 24.1% | −0.1897% | 0.050 | **No edge on this sample.** The single largest hole in v0.2.0's search (KNOWN-LIMITATIONS §0c: RSI and divergence absent entirely) is now measurable, and it measures slightly negative on 29 trades. Honest answer to a question that previously could not be asked. |
| `descending-triangle` | 1 | — | — | — | **INSUFFICIENT.** Free-rider mirror, `tier=None`. |
| `inverse-cup-and-handle` | 42 | 19.0% | −1.1095% | 0.050 | **No edge.** Loses on the largest tier-1-family sample. Note the contrast with its upright mirror (+2.96% on 31): two mirrored geometries with opposite signs on comparable samples is the signature of a directional bias in the data, not of a shape that works. |
| `symmetrical-triangle` | 1 | — | — | — | **INSUFFICIENT.** Free-rider mirror, `tier=None`. |

**Scoreboard, stated the way the contract asks:** 6 of 16 detectors are `INSUFFICIENT`; of the
10 measured, **5 positive and 5 negative after costs**; **0 have an established edge**, because
establishing one is the gate's job and the gate has not been run on any of them. Every `c` is
comfortably under `COST_RATIO_CEILING = 0.10` (range 0.050–0.068), so no detector is
*structurally* unable to pay for itself — the failures here are directional, not cost-driven.

---

## 2. Coverage ledger

Generating command:

```
.venv/bin/python -m trading_bot.cli detector-report --coverage
```

Output (verbatim):

```
catalog: 144 rows / 18 families   covered 19  deferred 6  out-of-scope 119
 contract §9 concepts -- tier 1: 2/3  (Wyckoff Accumulation / Distribution DEFERRED: unmeasurable, see catalog.py)   tier 2: 6/6
 ledger rows carrying a tier   -- tier 1: 3/5   tier 2: 11/11

  # family                            rows  cov  def  oos
--------------------------------------------------------
  1 Reversal Patterns                   16    4    2   10
  2 Continuation Patterns               15    9    2    4
  3 Bullish Candlestick Patterns        14    0    0   14
  4 Bearish Candlestick Patterns        14    0    0   14
  5 Indecision Candlestick Patterns      5    0    0    5
  6 Gap Patterns                         4    0    0    4
  7 Harmonic Patterns                    8    0    0    8
  8 Elliott Wave                         2    0    0    2
  9 Wyckoff Structures                   4    2    2    0
 10 Volume-Based Patterns                4    0    0    4
 11 Support & Resistance Patterns        6    0    0    6
 12 Trendline Patterns                   4    0    0    4
 13 Moving Average Patterns              6    0    0    6
 14 Oscillator Signals                  14    4    0   10
 15 Fibonacci Patterns                   7    0    0    7
 16 Market Structure                     9    0    0    9
 17 Smart Money Concepts (SMC)           8    0    0    8
 18 Volatility Patterns                  4    0    0    4
--------------------------------------------------------
16 detector key(s) referenced by covered rows
```

**19 delivered / 6 deferred / 119 out of scope = 144.** 13.2% of the catalog. Stated plainly,
because the PRD's own framing is that the full catalog is "a direction, not a v1 gate", and
this phase closes tier 2 completely.

The six deferred rows and their reasons:

| row | family | reason (abridged; verbatim in `catalog.py`) |
|---|---|---|
| Accumulation | 9 Wyckoff | A1 — a phase SEQUENCE, no agreed numeric definition, no reference implementation, no labelled dataset. Precision is **unmeasurable**, not merely low. |
| Distribution | 9 Wyckoff | A1, as above. |
| Triple Top | 1 Reversal | A double-top generalisation — one extra pivot in `_detect_double`'s loop. Held back so this phase's degrees of freedom stay countable. |
| Triple Bottom | 1 Reversal | As above. |
| Bull Pennant | 2 Continuation | A flag whose consolidation converges; needs `fit_converging_lines` inside the flag scan. A genuine small build, deferred to keep tier 2 closed at 6/6 rather than opened at 7/8. |
| Bear Pennant | 2 Continuation | As above. |

The 119 out-of-scope rows each carry one of five distinct categories (all five are used, and the
most-repeated single reason string appears 27 times against the test's ceiling of 40):
`tier 3-4 by contract §9` · `belongs to another plug-in kind` ·
`no reference implementation and no labelled data` · `owned by another phase` ·
`already covered by another registered plug-in`.

**The test that pins it**, and the only thing making coverage tracked rather than claimed:

```
.venv/bin/python -m pytest -q tests/test_detectors_catalog.py
# -> 14 passed
```

`tests/test_detectors_catalog.py::TestCatalogLedger::test_catalog_matches_the_source_document`
parses `.claude/technical-pattern.md`, collects every table row's left-column text from the 18
numbered family sections, skips the 15-row reliability table, and asserts
`sorted(doc_rows) == sorted((family_no, subsection, pattern) for e in CATALOG)` plus
`len(CATALOG) == 144`. Corrupting one `pattern` string fails it loudly.

---

## 3. Contract §9 scorecard, and the one item not delivered

| tier | committed | delivered | note |
|---|---|---|---|
| 1 | Cup & Handle, Head & Shoulders (refined), Wyckoff Accumulation/Distribution | **2 of 3** | Wyckoff acc/dist **DEFERRED** per A1. |
| 2 | Double Top/Bottom, Bull/Bear Flag, Ascending Triangle, Falling Wedge, Rising Wedge, RSI Divergence | **6 of 6** | Closed. |
| 3–4 | — | not built, by design | Contract §9: a direction, not a gate. No success criterion here depends on them. |

**Tier 1 is 2 of 3, not 3 of 3.** The deferral is the decision most likely to be overruled and it
is deliberately cheap to overrule: flip two `CATALOG` entries to `covered`, add
`plugins/detectors/wyckoff_structure.py`, and `test_tier_scorecard` will tell you to update A1.
What the phase will not do is pretend: a Wyckoff detector built to a made-up numeric definition
and validated against bars drawn to satisfy it would report a precision figure that means
nothing, and would enter Phase 9's campaign wearing a five-star label it had not earned.

In its place, the two mechanically decidable events ship: `wyckoff-spring` and
`wyckoff-upthrust`, whose module docstring and registry rationales state the non-claim in words
("do **NOT** detect accumulation or distribution", "cannot distinguish a spring from an ordinary
failed breakout"), pinned by
`tests/test_detectors_wyckoff_events.py::TestTheNonClaim::test_docstring_states_the_non_claim`.

---

## 4. The wedges fire nothing, and why that is a finding rather than a bug

`falling-wedge` and `rising-wedge` are the two patterns
`signals/patterns.py:327` made **structurally unreachable** — its
`if upper_end > upper_start or lower_end < lower_start: return out` rejects any shape with a
rising upper line or a falling lower line, which is every wedge. Relaxing that guard into
`plugins/detectors/continuation.py::_classify` is what makes them expressible at all, and it is
proved by
`tests/test_detectors_continuation.py::TestWedges::test_legacy_triangle_detector_cannot_see_this_shape`,
which builds a textbook falling wedge, asserts `detect_patterns(df, pivots) == []` and asserts the
new detector emits exactly one long event on the same frame.

They then produce **zero trades on two years of three symbols**. Two measurements establish that
this is inherited sample starvation in the converging-line fit, not a Phase 8 defect:

**(a) The frozen legacy detector is equally starved.** Same span, same symbols, same
single-detector diagnostic graph:

```
legacy-patterns trades by kind: Counter({'flag': 439, 'inverse-head-and-shoulders': 9,
                                         'head-and-shoulders': 9, 'triangle': 1})
```

One `triangle` trade. Phase 8's five-way split of that one `triangle` kind produces
1 + 1 + 1 + 0 + 0 = **3** trades — i.e. relaxing the guard *increased* the family's trade count
threefold, from a base of one. The tolerances that starve it (`TRIANGLE_MIN_CONVERGENCE = 0.25`,
`TRIANGLE_CONTAINMENT_TOL = 0.005`, `PATTERN_MAX_AGE_BARS = 12` at a 4H setup tier = 2 days of
freshness) are v0.2.0 constants that this phase did **not** touch.

**(b) It is not a trigger problem — the detectors emit no EVENTS at all.** Raw `DetectedEvent`
counts, walking every setup bar of the span for BTCUSDT through `EvalContext` (no P&L, no ledger
row — this is a measurement, not an evaluation):

```
BTCUSDT setup bars evaluated: 4381  (span 2023-07-27..2025-07-26, 4h tier)
  detector.symmetrical-triangle          raw DetectedEvents: 0
  detector.ascending-triangle            raw DetectedEvents: 1
  detector.descending-triangle           raw DetectedEvents: 4
  detector.falling-wedge                 raw DetectedEvents: 0
  detector.rising-wedge                  raw DetectedEvents: 0
```

Five raw events across the whole family over 4381 bars. The wedges find nothing because the
structure never occurs under these tolerances, not because a trigger never fired.

**(c) The geometry is correct by construction.** `TestWedges` and `TestTriangleVariants` build all
five shapes from explicit trendlines, assert `find_pivots` finds exactly the four intended anchors,
and assert each detector fires on its own shape and stays silent on the other four. The mutual
exclusivity of the five labels is proved over a 9×9 grid of signed slopes at three
`(flat_tol, min_slope)` settings by `test_shape_classification_is_mutually_exclusive`.

**No tolerance was loosened in response to this measurement.** The manual-validation checklist
warns that zero events for a tier-2 detector "is a geometry bug, not an absent edge — check the
tolerances against the fixture that passes"; the fixtures pass, the legacy comparison shows the
starvation is inherited, and retuning after seeing the table would convert the diagnostic into an
oracle. **Recommendation for Phase 9**, recorded and not acted on here: the converging-line family
needs either a coarser freshness bound or a longer `max_width_bars` to be measurable at the 4H
tier, and that change is a consumed degree of freedom that must be pre-registered.

---

## 5. `end_ts` conventions — what Phase 9 needs to know

`end_ts` is the bar a trigger must not predate, so getting it early is a lookahead that
*improves* every backtest. Three conventions coexist, deliberately:

| detectors | `end_ts` | why |
|---|---|---|
| `double-top`, `double-bottom`, `rsi-divergence` | `_geometry.confirmation_ts(df, pivot_index, span)` — the pivot's bar **plus `pivot_span`** | A fractal pivot is only knowable `span` closed bars after its own bar (`pivots.py:9-12`). This is the correct convention and is what every NEW pivot-terminated detector uses. `rsi-divergence` takes the **later** of the price pivot's and the RSI pivot's confirmation, `max(j, oj)`, because the two can confirm on different bars. |
| `head-and-shoulders`, `inverse-head-and-shoulders` | `window[-1].ts` — the last pivot's **own** bar | **A KNOWING EXCEPTION.** `plugins/detectors/legacy_patterns.py` wraps the same geometry and Phase 3's parity gate freezes it, so bit-exact parity with `signals/patterns.py:191-257` is load-bearing. The newer convention is stricter. **FOLLOW-UP: reconcile `legacy_patterns.py`'s `end_ts` once the parity test is retired.** Until then the H&S pair is `pivot_span` bars more optimistic than the rest of the roster, and Phase 9 must not compare it to them as if the convention were shared. |
| the five triangle/wedge shapes, `bull-flag`, `bear-flag`, `cup-and-handle`, `inverse-cup-and-handle`, `wyckoff-spring`, `wyckoff-upthrust` | the latest CLOSED bar | These are bar-terminated, not pivot-terminated: their levels are read at the current bar, which is already knowable. `confirmation_ts` does not apply. |

Two named regression tests pin the pivot-terminated cases:

- `tests/test_detectors_reversal.py::TestDoubleTop::test_end_ts_is_the_confirmation_bar_not_the_pivot_bar`
- `tests/test_detectors_oscillator.py::TestRsiDivergence::test_end_ts_is_the_later_of_the_two_pivot_confirmations`
  and `::test_end_ts_follows_the_rsi_pivot_when_the_rsi_pivot_confirms_later`, the second of which
  builds a frame where the RSI pivot genuinely confirms one bar after the price pivot (price pivots
  are decided by bar HIGHS, RSI pivots by CLOSES).

---

## 6. Degrees of freedom consumed (contract §12.6)

**Oracle evaluations charged to the trial ledger: 16.**

```
campaign="detector-report"  rows=+16  (campaign total 16)
```

One row per (detector, params-hash, span) under `config.DETECTOR_REPORT_CAMPAIGN`, written by
`backtest/trials.py`. Pinned by
`tests/test_detectors_report.py::TestReportBody::test_one_ledger_row_per_detector`.

**The selection cost, printed in the banner and restated here:** if any detector is dropped from a
campaign because of the table in §1, that campaign owes `n_trials += 16`. Reading the table is
free; letting it inform selection is not.

**Geometry choices made from classical definitions BEFORE any measurement**, and not retuned
after seeing the table:

| constant | value | chosen because |
|---|---|---|
| `CUP_ROUND_BAND` | 0.25 | The bottom quartile of the cup's depth — the band a rounded base sits in and a V does not. |
| `CUP_MIN_BASE_BARS` | 5 | A V bottom has one or two bars in that band; a rounded base has many. |
| `RSI_DIV_OVERBOUGHT` | 60.0 | Pairing two RSI highs needs the extreme one *elevated*, not extreme; a 70 floor eliminates most real pairs. |
| `RSI_DIV_OVERSOLD` | 40.0 | Mirror of the above. |
| `WYCKOFF_PROBE_VOL_RATIO` | 1.5 | Defined by reference to the existing `VOLUME_HIGH_RATIO = 1.5`, so activating a volume test invented no new number. |

`CUP_ROUND_BAND` and `CUP_MIN_BASE_BARS` interact (widen the band and any V passes given enough
bars). Both are recorded as **degrees of freedom consumed by a geometry choice, not by fitting**,
and that remains true: `test_v_bottom_passes_every_clause_except_roundness` shows the negative
fixture flips to a positive at `min_base_bars=3`, which is exactly how sensitive the choice is.

**No detector's defaults were changed after reading the edge table.** The two rows that most
invite it — `falling-wedge` and `rising-wedge` at zero trades — are analysed in §4 and left alone.

Every other threshold is a declared `ParamSpec` with bounds, defaulting to the `config.py` value
of the same name, so Phase 6's mutator can jitter it inside a legal range and the UI can render a
control from the same declaration. Summed declared degrees of freedom across all 20 registered
detectors: **3 536 001** (`cli.py plugins --kind detector`). That is what a graph *could* expose;
the DSR is charged for evaluations actually performed, which is 16.

---

## 7. The zero-engine-edit proof, measured

The PRD's headline success metric is "a new detector or rule added with **zero engine-core
edits**". `detector.rising-wedge` was implemented **last**, after every shared helper existed, and
reads **no `config` attribute at all** — every default in its decorator is an inline literal, which
is why adding it could not touch `config.py`.

**DEVIATION FROM THE PLAN, stated up front:** Task 15 specifies measuring this with
`git diff --name-only HEAD~1 HEAD` over a dedicated `rising-wedge` commit. This phase was
instructed not to commit (the orchestrator reviews and commits), so the equivalent measurement was
taken over **content hashes of the whole working tree** immediately before and immediately after
the `rising-wedge` step:

```
# before the rising-wedge step
find . -type f -not -path './.git/*' -not -path './.venv/*' -not -name '*.pyc' \
  -not -path '*/__pycache__/*' -not -path './data/*' -not -path './.claude/PRPs/reports/*' \
  -exec shasum -a 256 {} \; | sort -k2 > /tmp/p8_before_rising_wedge.manifest   # 189 files

# ... register detector.rising-wedge + its tests ...

# after
find . ... > /tmp/p8_after_rising_wedge.manifest
comm -13 <(sort /tmp/p8_before_rising_wedge.manifest) <(sort /tmp/p8_after_rising_wedge.manifest) \
  | awk '{print $2}' | grep -v '^\./\.pytest_cache/' | sort
```

Output, verbatim:

```
./src/trading_bot/plugins/detectors/continuation.py
./tests/test_detectors_continuation.py
```

**Exactly two paths.** No `config.py`, no `cli.py`, no `framework/`, no `signals/`, no
`backtest/`. (`.pytest_cache/v/cache/{lastfailed,nodeids}` also changed and is excluded above;
`git check-ignore` confirms it is gitignored build residue, not repo content.)

**No `FRAMEWORK-GAP` was recorded.** Every one of the 16 detectors was expressible inside
`plugins/detectors/` against Phase 3's `Detector` contract with no edit anywhere else. Two
structural tests keep the property honest rather than anecdotal:
`test_rising_wedge_source_reads_no_config_attribute` and
`test_rising_wedge_defaults_match_the_shared_ones` (the drift guard for the duplicated literals
the no-config rule forces).

---

## 8. `signals/patterns.py` is unedited

```
git diff --stat -- src/trading_bot/signals/patterns.py     # (no output)
git diff -- src/trading_bot/signals/patterns.py | wc -l    # 0
git status --porcelain -- src/trading_bot/signals src/trading_bot/framework \
    src/trading_bot/backtest src/trading_bot/data src/trading_bot/regime src/trading_bot/risk
                                                            # (no output)
```

**Two geometry implementations now coexist, and that is recorded so nobody "cleans it up":**

- `signals/patterns.py`, reached through `plugins/detectors/legacy_patterns.py`, whose job is
  reproducing v0.2.0 **exactly**. Frozen by Phase 3's parity gate (43 passed).
- `plugins/detectors/_geometry.py`, whose job is being **correct** — same helpers, ported verbatim
  with their docstrings, minus the expanding-side guard.

They are allowed to disagree. Converging them is only safe once the parity test is retired. Both
`_geometry.py`'s module docstring and this section say so.

---

## 9. Validation

| check | command | result |
|---|---|---|
| New Phase 8 tests | `pytest -q tests/test_detectors_{reversal,continuation,oscillator,wyckoff_events,catalog,report}.py` | **176 passed** |
| Pre-existing baseline | `pytest -q tests/test_{backfill,backtest,binance_client,classifier,cli,donchian,equity,meanrev,poller,risk_atr_stop,signals,storage,tiers,wilder}.py` | **297 passed, 1 skipped** |
| Phase 3 parity gate | `pytest -q tests/test_framework_parity.py` | **43 passed** in 212.92s |
| Whole suite (all phases, run concurrently with Phases 5-6) | `pytest -q` | **1136 passed, 2 skipped** in 156.74s |
| Byte-compilation | `python -m py_compile` over all 10 new/changed files | clean, exit 0 |
| Registered detectors | `cli plugins \| grep -c '^detector\.'` | **20** (16 new + 4 from Phases 3–4) |
| Rationales | `cli plugins \| grep -A1 '^detector\.' \| grep -c 'rationale:'` | **20** — every one non-empty |
| Holdout guard | `cli detector-report --start 2023-07-27 --end 2026-07-26; echo $?` | **exit 2**, refusal names the boundary date `2026-02-27` |
| Coverage ledger | `cli detector-report --coverage` | 144 rows / 18 families, 19/6/119, tier 1 2/3, tier 2 6/6 |

**No linter and no type checker are configured in this repo** (KNOWN-LIMITATIONS §8, contract
§12.2), so "static analysis" means byte-compilation only. No command was invented.

**No new dependency.** `pandas-ta` is gone from PyPI and TA-Lib needs a C library; RSI is
hand-rolled over the production `indicators/wilder.wilder_smooth`, and every geometry is pandas +
numpy, exactly as `patterns.py` and `wilder.py` already are.

### The tests that would still matter if every detector were rewritten

| test | what it protects |
|---|---|
| `TestHeadAndShouldersParity` (20 parametrized cases) | The refined port reproduces `signals/patterns.py` bit-for-bit at defaults, over all ten legacy H&S fixtures, for both directions. |
| `test_end_ts_is_the_confirmation_bar_not_the_pivot_bar` | The `pivot_high`-vs-`find_pivots` convention clash: a `span`-bar lookahead that *improves* backtests. |
| `test_end_ts_is_the_later_of_the_two_pivot_confirmations` (+ the lagging-pivot variant) | Two pivots, two confirmation lags; taking one is a real lookahead. |
| `test_legacy_triangle_detector_cannot_see_this_shape` | Pins *why* `patterns.py:327` had to be relaxed. |
| `test_shape_classification_is_mutually_exclusive` | Deleting 327 without exclusivity relabels every wedge a "triangle". |
| `test_v_bottom_is_not_a_cup` | Without roundness, cup & handle is a slow double bottom. |
| `test_rsi_plateau_yields_no_oscillator_pivot` | `find_pivots` rejects ties by design; asserting it stops a future "fix" to `>=`. |
| `test_docstring_states_the_non_claim` | Pins the honesty requirement itself. |
| `test_rsi_hand_computed_14_period` | Hand-computed RSI at three indices, working shown in the docstring. |
| `test_catalog_matches_the_source_document` | The only thing making coverage *tracked* rather than claimed. |
| `test_tier_scorecard` | Pins scope decision A1 against silent drift. |
| `test_holdout_guard_rejects_recent_end` | Keeps the diagnostic out of Phase 9's holdout mechanically. |
| `test_insufficient_suppresses_expectancy` | KNOWN-LIMITATIONS §1's lesson one level down. |
| `test_no_second_pnl_path` | Contract §1: a new execution path charges costs identically or it is lying. |
| `test_report_is_sorted_by_tier_then_name_not_by_expectancy` | Sorting by expectancy *is* selection. |

---

## 10. Files

| file | action | lines |
|---|---|---|
| `src/trading_bot/indicators/rsi.py` | CREATE | 114 |
| `src/trading_bot/plugins/detectors/_geometry.py` | CREATE | 380 |
| `src/trading_bot/plugins/detectors/reversal.py` | CREATE | 464 |
| `src/trading_bot/plugins/detectors/continuation.py` | CREATE | 883 |
| `src/trading_bot/plugins/detectors/oscillator.py` | CREATE | 317 |
| `src/trading_bot/plugins/detectors/wyckoff_events.py` | CREATE | 252 |
| `src/trading_bot/plugins/detectors/catalog.py` | CREATE | 2104 |
| `src/trading_bot/config.py` | UPDATE (append-only) | +88 |
| `src/trading_bot/cli.py` | UPDATE (append/insert only) | +396 |
| `tests/conftest.py` | CREATE | 160 |
| `tests/fixtures/patterns/*.csv` | CREATE (4 files) | 322 |
| `tests/test_detectors_reversal.py` | CREATE | 356 |
| `tests/test_detectors_continuation.py` | CREATE | 514 |
| `tests/test_detectors_oscillator.py` | CREATE | 362 |
| `tests/test_detectors_wyckoff_events.py` | CREATE | 215 |
| `tests/test_detectors_catalog.py` | CREATE | 242 |
| `tests/test_detectors_report.py` | CREATE | 392 |
| `.claude/PRPs/reports/phase8-detector-edge-report.md` | CREATE | this file |
| `.claude/PRPs/reports/phase8-detector-report.out.md` | CREATE | raw `--out` artifact |

Nothing deleted. Nothing outside `plugins/detectors/`, `indicators/rsi.py`, `config.py`, `cli.py`,
`tests/` was touched.

---

## 11. Follow-ups handed forward

1. **Reconcile `legacy_patterns.py`'s `end_ts`** with `_geometry.confirmation_ts` once Phase 3's
   parity test is retired. Until then the H&S pair is `pivot_span` bars more optimistic than the
   rest of the roster (§5).
2. **Repoint `DETECTOR_REPORT_HOLDOUT_GUARD_DAYS`** at Phase 9's `HOLDOUT_*` when it lands. It
   currently derives from `WF_OOS_DAYS + WF_TEST_DAYS = 150` because Phase 9's constants do not
   exist yet — a recorded coupling, not a hidden one.
3. **The converging-line family is unmeasurable at the 4H setup tier** under the inherited
   v0.2.0 tolerances (§4). Any change to them is a pre-registered degree of freedom.
4. **Converge the two geometry implementations** only after the parity gate is retired (§8).
5. **Two ledger rows are one line of code from `covered`** — Triple Top / Triple Bottom — and two
   more are a small build — Bull / Bear Pennant. Both deferrals exist to keep this phase's degrees
   of freedom countable, not because the work is hard.

---

## 12. Deviations from the plan — what and why

Every one of these is a deviation from `phase8-pattern-coverage-expansion.plan.md`, recorded so a
reviewer does not have to find them. None changes what the phase delivers; three of them fix
places where the plan contradicted itself or specified something unimplementable.

| # | plan said | what was done | why |
|---|---|---|---|
| 1 | Task 15: prove the footprint with `git diff --name-only HEAD~1 HEAD` over a dedicated `rising-wedge` commit | Content-hash manifests of the whole tree, immediately before and after the `rising-wedge` step (§7) | This phase was instructed not to commit. The measurement is equivalent and its two-path output is quoted verbatim. |
| 2 | `detector-report --out .claude/PRPs/reports/phase8-detector-edge-report.md` | `--out` writes `phase8-detector-report.out.md`; the hand-written phase report is `phase8-detector-edge-report.md` | Task 15 also requires the hand-written report at that path. One would overwrite the other. Both files exist and the report quotes the artifact verbatim. |
| 3 | Table has `events` and `trig` columns, "from the detector-level counters recorded in `meta`" | Table has `trades`; raw event counts measured separately (§4) | **No such counter mechanism exists.** A Detector returns events for one bar; nothing aggregates across bars, and adding an aggregator means editing `framework/execute.py` — which would be a `FRAMEWORK-GAP` for a reporting nicety. Measuring events with a separate read-only walk (§4b) gets the same information without an engine edit. |
| 4 | `tests/conftest.py` gets "exactly one fixture, `pattern_fixture`" | Two fixtures (`pattern_fixture`, `setup_context`) plus three importable module-level helpers (`load_pattern_fixture`, `make_setup_context`, `detect_events`) | Four Phase 8 test modules need an `EvalContext` harness; putting it in each would be four copies of a no-lookahead harness. Still purely additive: no autouse fixture, no `pytest_*` hook, nothing that changes any pre-existing test. |
| 5 | `mode="hidden"` "reverses the **price** inequality only", then defines hidden bearish as "lower price high with a **higher** RSI high" | The parenthetical (classical) definition: hidden bearish = lower price high **and** higher RSI high | The two halves of the sentence contradict each other; the classical definition is what the catalog's Hidden Divergence rows mean. Stated in `oscillator.py`'s docstring. |
| 6 | Wyckoff condition 2: "no *close* below `lo`" | `min(close) > lo` (strict), i.e. no bar may have CLOSED **at or through** the floor | The plan's literal form is **vacuous**: `lo` is the window's minimum LOW, and `close >= low >= lo` always. The strict form is the non-vacuous reading — a bar that closed on the floor *accepted* that price, so the floor is not being defended. Documented in `wyckoff_events.py` and covered by two negative tests. |
| 7 | Roster: `inverse-cup-and-handle` `tier=1` | `tier=None` in both registry and ledger | The plan's own "Free riders" note says it is not in §9's tier list and the ledger should record `tier=None`. The roster's Tier column contradicts that note; the note wins, so the registry does not advertise a five-star tier the shape has not earned. |
| 8 | Roster †: `wyckoff-spring` / `-upthrust` `tier=1` | Registry `tier=1` (as the roster says); **ledger `tier=None`** | The reliability table does not name Spring or Upthrust. Ledger tier is read strictly off that table, so these two carry None there while the registry keeps the roster's 1. Contract §9's "tier 1 lands 2 of 3" is carried by the CONCEPT scorecard (`section9_scorecard`), not by row counts — both are printed so neither can be misread. |
| 9 | `CatalogEntry` fields: `family_no, family, pattern, tier, status, detector_key, reason` | Added `subsection: str = ""` | `(family_no, pattern)` is **not unique**: the MACD and Stochastic subsections of family 14 each contain a row literally named "Bullish Cross" and one named "Bearish Cross". The uniqueness key is `(family_no, subsection, pattern)`. Without the field the ledger silently collapses four rows into two, and the two Stochastic crosses inherit MACD's reason. |
| 10 | H&S gotcha (d): with a sloped neckline, "reject when the sloped level at `end_x` is not strictly between the head and the nearer shoulder" | Rejects on the existing shape inequalities evaluated at **each shoulder's own bar** (`p1 > neck_at(left.index)`, `p3 > neck_at(right.index)`), plus `neckline > 0` | As written the clause does not parse: a bearish neckline sits BELOW both shoulders, so "between the head and the nearer shoulder" describes an empty interval. The implemented form is what the clause is protecting against — an extrapolated level that is not a sane support for this shape — and it clamps nothing, exactly as instructed. |
| 11 | "the seven H&S fixtures at `tests/test_signals.py:75-169`" | **Ten** fixtures, parity-checked in both directions (20 parametrized cases) | `tests/test_signals.py::TestHeadAndShoulders` contains ten cases, not seven. All ten are checked, plus two non-vacuity tests so a parity test over ten empty lists cannot pass by accident. |
| 12 | (unstated) | Three of the ten parity cases monkeypatch `reversal.find_pivots` | Those legacy cases inject a hand-made pivot list into `detect_patterns`, which a Detector cannot receive — it computes its own pivots from `EvalContext`. The patch is a harness detail; the detector still receives only an `EvalContext`. |
| 13 | `--detector` "filters to registry keys" | Also **refuses** a registered non-Phase-8 detector (exit 1) | Phases 3–4's detectors were not designed for this fixed exit policy, and silently measuring them here would produce a comparison nobody asked for. |
| 14 | Task 8: cup declares `max_age_bars` | Declared, and it bounds nothing | Freshness is skipped for the cup (the handle ends at the latest bar, `patterns.py:96-98`). The parameter is kept only so `_scan_params()` stays one thing; stated in the detector's docstring so it is not mistaken for an active clause. |

**Nothing was left incomplete.** All 15 plan tasks are done. The one thing the plan itself
declared out of reach — Wyckoff Accumulation/Distribution — is `deferred` with its reason in code
and pinned by a test.

---

## 13. What this phase can and cannot prove

It **can** prove sixteen geometries are detected correctly: fixtures make that a matter of
construction, not opinion. It **can** measure each one's after-costs expectancy on the tuning
span. It **cannot** prove any of them has edge — contract §4 reserves that verdict for the gate,
and KNOWN-LIMITATIONS §1's arithmetic (23 trades could not clear a floor of 30) means most
per-detector samples here are too small to support one. Ten of sixteen cleared a floor of 20;
three cleared 100.

The right reading of a finished Phase 8 is: *the engine can now express these sixteen things, and
here is what they did on two years of three symbols.* Not: *here are the profitable patterns.*

The value that is real and immediate is diagnostic. KNOWN-LIMITATIONS §0c is the sharpest sentence
in this project's own accounting — the v0.2.0 search "never explored the entry or the feature
set." RSI and divergence were **absent entirely** from every sweep. They are now measurable, and
`rsi-divergence` measures −0.1897% on 29 trades. That is a worse answer than hoped for and a
better situation than before, because for the first time it is an answer.
