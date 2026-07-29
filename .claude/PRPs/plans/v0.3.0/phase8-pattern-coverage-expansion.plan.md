# Plan: Pattern Coverage Expansion (PRD Phase 8)

## Summary

Widen the engine from four hardcoded geometries to **sixteen registered `Detector` plug-ins** covering
contract §9's reliability tiers 1–2. Each lands with **geometry fixture tests** (bars built so the
answer is known by construction) and a **per-detector edge report** (hit rate, expectancy after costs).
Detection and edge are validated separately on purpose: a detector can be perfectly correct and
worthless, and only the report tells them apart. Coverage of the catalog's 144 rows goes in a
machine-checked ledger, so what is *not* built stays visible.

## User Story

As the operator whose v0.2.0 entry "carried almost no predictive content" (KNOWN-LIMITATIONS §0), I
want registered detectors I can compose and individually measure, so the search finally explores the
**entry and feature space** that §0c records as never having been touched — and so I can rule detectors
out on evidence, not taste.

## Problem → Solution

`signals/patterns.py` implements four geometries and is *retired from dispatch* (§8).
`scripts/bruteforce/` implements eight more plus RSI, but as boolean `pd.Series` outside the tested
package — no level, no target, no coverage from the 286 production tests. MACD, RSI, stochastic,
divergence and market structure were **absent entirely** from every sweep (§0c). The catalog has 144
rows; the engine expresses 4. → Sixteen detectors as plug-ins, each with fixtures, a `ParamSpec` the UI
can render and the Mutator can jitter, and an after-costs edge report on the tuning span only; plus a
ledger enumerating all 144 rows, tested against the catalog document so it cannot drift.

## Metadata

- **Complexity**: Large — 16 detectors, 1 indicator, 1 CLI command, 6 test modules (~2 400 LOC source,
  ~2 000 LOC tests)
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md`, Phase 8
- **Binding contract**: `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md` (§ refs below are
  to it unless prefixed KNOWN-LIMITATIONS)
- **Depends on**: Phase 4 (→ 3). **Parallel with**: Phase 7. **Gates**: Phase 9.
- **Files**: 8 created, 2 updated (`config.py`, `cli.py`), 6 test modules, 0 deleted
- **Test baseline**: **286 tests collected** (`.venv/bin/python -m pytest --collect-only -q`, verified
  2026-07-27). All 286 stay green at the phase boundary (§8, §12.1).

---

## Stated Assumptions and Scope Decisions

**A1 — Wyckoff Accumulation/Distribution is DEFERRED, not built.** §9 lists it tier 1 *and* requires
any plan proposing it to "say what precision is achievable and how correctness is checked, or defer it
explicitly." Deferred. The reason is not effort: an accumulation is a *sequence of phases*
(PS → SC → AR → ST → Spring → LPS → SOS) whose distinguishing evidence is effort-versus-result judged
across weeks. There is no agreed numeric definition, no reference implementation (PRD Research
Summary), and **no labelled dataset** — achievable precision is not merely low, it is *unmeasurable*.
Building it yields a detector whose only validation is "it fires where I drew it firing," which is
circular. In its place: the two mechanically decidable *events* Wyckoff schematics place inside Phase C
— **Spring** and **Upthrust** — with an explicit non-claim (A2). Ledger status `deferred` with this
reason. This is the one §9 tier-1 item the phase does not deliver; flagged to the user.

**A2 — the Spring/Upthrust non-claim.** They detect *a probe below/above an established range that
closes back inside on elevated volume*. They do **not** detect accumulation or distribution, and cannot
distinguish a spring from an ordinary failed breakout — that distinction lives in the surrounding phase
sequence. The registry `rationale` says so in those words. Correctness is validated only
by-construction: we validate the *rule*, never the *concept*.

**A3 — no success criterion depends on tier 3 or 4** (§9). MACD Cross is Phase 4's; Golden Cross,
harmonics, Elliott, SMC, candlestick-alone are ledger rows with reasons. `indicators/macd.py` is
**Phase 4's file** — consume, never create.

**A4 — the edge report is a diagnostic, never an oracle** (§4). It runs through
`framework.execute.run_graph_backtest` so costs are charged by the frozen production path; sweeps
nothing; never calls `walk_forward_pooled`; never prints a gate verdict; refuses to run inside the
reserved holdout; writes its own trial-ledger rows. Mechanism: Task 11.

**A5 — cross-detector dedup is deliberately absent.** Two detectors firing long on one bar is
*confluence*, which the catalog's own checklist (`.claude/technical-pattern.md:309-321`) treats as the
strongest condition available. Collapsing them is a Phase 4 policy decision. Do not "fix" it here.

**A6 — where PRD and contract disagree, the contract wins.** The PRD lists Phase 8's patterns as one
undifferentiated set; §9 splits them across tiers and fixes the ordering — followed, not re-litigated.
The PRD makes per-detector edge reports a "Could"; §9 makes them mandatory per detector. Mandatory.

---

## Dependencies on Other Phases

| From | Artifact | What breaks without it |
|---|---|---|
| **P3** | `contracts.Detector`; `DetectedEvent(kind, direction, level, target_height, start_ts, end_ts, meta)` | Nothing to register against. Field names differ from `PatternCandidate` (`level`, not `breakout_level`) — emit the **contract's** names |
| **P3** | `registry.register(kind, *, name, params, rationale, timeframes=(), tier=None)`, `ParamSpec`, `REGISTRY`, `load_all()` | No decorator. `tier` already exists in the signature — every Phase 8 detector passes 1 or 2 |
| **P3** | `context.EvalContext` | Detectors must **never** load bars themselves. `EvalContext` is what inherits `engine._assert_interval` (`engine.py:185`) and the closed-bars-only slicing that makes every pivot causal |
| **P3** | `execute.run_graph_backtest(conn, graph, symbol, *, start_ms, end_ms, fee_pct, slippage_pct, funding_pct_per_day, max_hold_bars)`; `plugins/data/ohlcv.py`; `plugins.load_all()` | The edge report would need a second P&L path — what §1 forbids. Without `load_all`, decorators never run and a family "reads as tested and found wanting" (§3) |
| **P4** | `plugins/policies/measured_move.py` (entry/TP/SL from `level` ± `target_height`) | No way to turn an event into a trade → no edge report. **Do not stub a local policy**: that is the forbidden second path |
| **P4** | `Trade.planned_rr / confirmations / strategy_version`, appended with defaults (§5) | Report loses the R:R audit column. Not fatal |
| **P1** | `backtest/trials.py` + `data/statestore.py` | The report cannot record its own degrees of freedom, violating §4.2 |

`backtest/benchmark.py` is deliberately **not** consumed: one detector is not a strategy, so comparing
it to buy-and-hold is a category error. Phase 9 makes that comparison.

**If any detector needs an edit outside `plugins/detectors/`, that is a Phase 3 defect.** Record a
`FRAMEWORK-GAP-<n>` naming the exact expression that could not be written, and hand it to Phase 3.
Working around it by editing the engine silently retires the PRD's headline metric.

---

## UX Design

**Before**: four geometries exist in `patterns.py`; none is dispatched. "Does the double top make
money?" is not an expressible question.

**After**:

```
$ python -m trading_bot.cli detector-report --coverage
catalog: 144 rows / 18 families   covered 19  deferred 6  out-of-scope 119
 tier 1: 2/3  (wyckoff-accumulation, wyckoff-distribution DEFERRED — see A1)   tier 2: 6/6

$ python -m trading_bot.cli detector-report --start 2023-07-27 --end 2025-07-26
=== DIAGNOSTIC — NOT A GATE VERDICT ======================================
 span 2023-07-27..2025-07-26 (tuning span; holdout guard: 150d reserved)
 fixed exit policy, nothing swept.  ledger campaign="detector-report" rows=+16
 SELECTION COST IF USED: if any detector is dropped from a campaign because of
 this table, add n_trials += 16 to that campaign.
=========================================================================
detector             tier events trig trades  win%   exp%    pf   c     verdict
double-top              2    143   61     58 44.8% -0.0912 0.81 0.061  MEASURED
falling-wedge           2     96   47     44 52.3% +0.1104 1.19 0.055  MEASURED
cup-and-handle          1     22    9      9   --      --    --   --   INSUFFICIENT(<20)
```

| Touchpoint | Before | After |
|---|---|---|
| Adding a pattern | Edit `patterns.py` + `engine.py` dispatch + `scan.py` | One function + `@register` in `plugins/detectors/<family>.py`; proved by diff footprint (Task 15) |
| "Does pattern X pay?" | Not expressible | `detector-report` — per-detector, after costs, tuning span only |
| What is *not* covered | Nothing | `--coverage` + a test parsing the catalog doc |
| Tuning a tolerance | Edit config, restart | `ParamSpec` bounds — jitterable by the Mutator, renderable by the UI (§3) |

---

## Mandatory Reading

| P | File | Lines | Why |
|---|---|---|---|
| P0 | `_shared-architecture-contract.md` | §0b, §3, §7, §8, §9 | Binding. §0b port-don't-rewrite; §3 contracts + registry; §7 reserves `DETECTOR_*` and `detector-report`; §9 fixes tiers |
| P0 | `signals/patterns.py` | 1-38, 58-77, 103-141 | Docstring states the two cross-detector rules (freshness, unbroken level) inherited verbatim; `PatternCandidate`; `detect_patterns` orchestration; `_dedupe` (125) |
| P0 | same | 144-188 | Helpers to port: `_collapse_runs` (144), `_level_unbroken` (167), `_strictly_ordered` (181). Each docstring names a real bug it prevents — carry the docstrings across |
| P0 | same | 191-257 | `_detect_head_and_shoulders` — reproduce **bit-identically at defaults**, incl. `neckline = min(t1,t2)` (209) and the both-shoulders-outside tests (215-217, 240-242) |
| P0 | same | 272-385 | `_detect_triangles` + `_triangle_contains`. **Line 327 is this phase's load-bearing line** — it makes rising/falling wedges structurally impossible. Task 3 drops it; Task 6 relocates its work |
| P0 | same | 388-464 | `_detect_flags`. Takes no `fresh` closure: flags are fresh by construction (96-98) |
| P0 | `signals/pivots.py` | 42-87 | `find_pivots(df, *, span=None) -> list[Pivot]`; `Pivot(index, ts, price, kind)`. The bound `range(span, n-span)` (75) is why a pivot is only *returned* once `span` bars closed after it — every pivot detector's causality rests on it. Outside-bar caveat (57-61) is why `_strictly_ordered` exists |
| P0 | `scripts/bruteforce/indicators.py` | 130-145 | `rsi` — donor. Uses production `wilder_smooth`, not an EMA approximation. Drop the `avg_loss==0 & avg_gain>0 → 100.0` clause (144-145) and RSI is NaN through every uptrend |
| P0 | same | 500-531 | `double_bottom` (500) / `double_top` (519). **Boolean `pd.Series`, no level, no target** — the adapter is real work (Task 5) |
| P0 | same | 580-613 | `_hs` (580), `head_and_shoulders` (598), `inverse_head_and_shoulders` (609). Weaker than production's: 3 pivots not 5, no neckline, no freshness, no unbroken-level test, pairs only *consecutive* pivots (585) so a noise pivot hides the shape. Production is the better base |
| P0 | `tests/test_signals.py` | 24-43 | `make_df(rows, start, interval)`, `path_df(anchors, step, volume)`. `path_df`'s `open=high=low=close` makes pivots land exactly on anchors |
| P0 | same | 187-265 | `_line` (187), `wedge(n, upper_pts, lower_pts)` (192), `TestTriangleGeometry` (221-265) — the style to match: build the shape, mutate one bar, test one rejection |
| P1 | `scripts/bruteforce/indicators.py` | 346-406, 616-631 | `rolling_high`/`rolling_low` + `exclude_current` (346/357); `pivot_high`/`pivot_low` confirmed `span` late (370/388) and the "textbook lookahead" comment (375-377); `breakout_of_range`'s tight-range filter (626-628), reused by Wyckoff. `triangle_squeeze` (533) is a *measurement* stand-in — **do not** use it for the triangles |
| P1 | `signals/breakout.py` | 1-58 | `check_breakout` reads only `level`, `direction`, `end_ts` — why §3 made `DetectedEvent` field-compatible, and why the report gets triggering for free |
| P1 | `config.py` | 42-82, 141-183 | Per-pattern block style (`HS_*` 51, `TRIANGLE_*` 55, `FLAG_*` 67, `VOLUME_*` 74); `MAX_HOLD_BARS_TRIGGER` (147), `WF_TEST_DAYS`/`WF_OOS_DAYS` (149-150), `ATR_STOP_*` (162-174), `FEE_PCT`/`SLIPPAGE_PCT`/`FUNDING_PCT_PER_DAY`/`COST_RATIO_CEILING` (180-183) — every value the report's fixed exit policy reads and **sweeps none of** |
| P1 | `cli.py` | 129-146, 199-213, 356-368, 396-439 | Subcommand pattern: `add_parser` + `type=_date_arg`, `args.command ==` dispatch, `_<name>_command(...) -> int`; `_fmt`/`_print_metrics` to reuse |
| P1 | `backtest/engine.py` | 322-345 | `candidates_for(h_idx)` — the per-setup-bar cached detection call the report walks the equivalent of, through `EvalContext` |
| P1 | `backtest/metrics.py` | 15-34 | `compute_metrics(trades) -> dict`; `by_bucket` keys on `f"{t.regime}/{t.pattern}"` (33), so `DetectedEvent.kind` becomes a bucket name for free |
| P2 | `scripts/bruteforce/registry.py` | 1-80 | `@register(family, grid, rationale)` prior art; mandatory-`rationale` rule (71-73: "an unmotivated strategy in a 10,000-combo sweep is just noise with a name"); trial-counting doctrine (28-35) |
| P2 | `indicators/wilder.py` | 49-105 | `wilder_smooth(series, period)` — RSI's base. Leading NaNs preserved, never filled |
| P2 | `.claude/technical-pattern.md` | all | **Measured 2026-07-27: 144 pattern rows / 18 families** (159 table rows − the 15-row reliability table) |
| P2 | `.claude/PRPs/reports/KNOWN-LIMITATIONS.md` | §0, §0c, §1 | §0 the entry carries no predictive content — this phase is the direct attack; §0c RSI/divergence absent entirely; §1 23 trades could not clear a floor of 30, the arithmetic that bounds every per-detector report and is why `INSUFFICIENT` exists |

**External documentation**: none needed, none permitted to add a dependency. `pandas-ta` is gone from
PyPI, TA-Lib needs a C library (§1); every geometry is hand-rolled over pandas + numpy as
`patterns.py` and `wilder.py` already are. The PRD's Research Summary is consumed as a *finding* — no
mature library covers this catalog — which is the justification for A1.

---

## Patterns to Mirror

### DEDUP_RULE — one event per (kind, direction), freshest then largest target

```python
# SOURCE: signals/patterns.py:125-141
def _dedupe(candidates):
    """Keep one candidate per (kind, direction): freshest, then largest target.
    Overlapping windows of the same pattern type (several 5-pivot H&S windows
    sharing a head, say) describe one setup, not several. Emitting them all
    raises duplicate signals for the same symbol on the same bar."""
    best: dict[tuple[str, str], PatternCandidate] = {}
    for c in candidates:
        key = (c.kind, c.direction)
        inc = best.get(key)
        if inc is None or (c.end_ts, c.target_height) > (inc.end_ts, inc.target_height):
            best[key] = c
    return list(best.values())
```

### THE_LINE_THAT_FORBIDS_WEDGES

```python
# SOURCE: signals/patterns.py:327-328
    if upper_end > upper_start or lower_end < lower_start:
        return out  # a side is expanding: not a triangle
# A rising wedge has BOTH lines rising; a falling wedge has BOTH falling, so this
# guard makes both unreachable. Task 3 removes it from the fit; Task 6 RELOCATES
# its work into per-detector slope classification. Removing without relocating
# relabels every wedge a "triangle".
```

### WILDER_RSI_DONOR

```python
# SOURCE: scripts/bruteforce/indicators.py:130-145
def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain, loss = delta.clip(lower=0.0), (-delta).clip(lower=0.0)
    avg_gain, avg_loss = wilder_smooth(gain, period), wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 with a positive avg_gain is a pure uptrend: RSI is 100.
    return out.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
```

### TEST_STRUCTURE — build the shape, mutate one bar, assert the rejection

```python
# SOURCE: tests/test_signals.py:221-241
class TestTriangleGeometry:
    def test_detects_contained_wedge(self):
        df, pivots = wedge()
        tri = [c for c in detect_patterns(df, pivots) if c.kind == "triangle"]
        assert sorted(c.direction for c in tri) == ["long", "short"]

    def test_rejects_bar_piercing_a_trendline(self):
        df, pivots = wedge(); df = df.copy()
        # One bar spikes far above its upper trendline: price was not bounded by
        # the lines, so the "triangle" never contained the move.
        df.iloc[15, df.columns.get_loc("high")] = 200.0
        assert [c for c in detect_patterns(df, pivots) if c.kind == "triangle"] == []
```

### Cited, not re-quoted — read them at the line given

- **FRESHNESS_CLOSURE** — `patterns.py:113-114`: `fresh = lambda i: (n-1) - i <= max_age_bars`, threaded
  into H&S and triangles at 116-117 but **not** into flags (118), because "Flags are inherently fresh
  (their consolidation ends at the latest bar) and skip this check" (96-98).
- **SPENT_LEVEL_RULE** — `patterns.py:167-178` `_level_unbroken(df, level, direction, after_index)`:
  "A pattern whose level was already breached is spent … a later crossing is a retest re-break, not the
  breakout."
- **NOISE_PIVOT_COLLAPSE** — `patterns.py:144-164`: "Fractal pivots do not strictly alternate … each run
  of consecutive same-kind pivots collapses to its highest high / lowest low."
- **CONSERVATIVE_LEVEL_CHOICE** — `patterns.py:209` `neckline = min(t1, t2)  # stricter (lower) trough:
  conservative trigger`; `:234` `neckline = max(p1, p2)`.
- **CONTAINMENT_TEST** — `patterns.py:373-385`: every bar in `[start_x, end_x]` inside the lines within
  `TRIANGLE_CONTAINMENT_TOL`. "Without this, two lines that merely converge qualify as a triangle even
  when price spent the window well outside them."
- **PIVOT_CONFIRMATION_LAG** — `pivots.py:75` (the loop bound *is* the guarantee) and `pivots.py:9-12`;
  the same rule stated as a trap at `bruteforce/indicators.py:374-377`: "Marking it at t would be
  textbook lookahead — and is exactly the trap `assert_causal` catches."
- **TIGHT_RANGE_DEFINITION** — `bruteforce/indicators.py:626-628`: `width = (hi-lo)/close`,
  `tight = width <= max_width`; reason at 619-624 — "only ranges narrow enough to be a real
  consolidation qualify, so the breakout has a defined, small risk."
- **TIER_CONSTANTS_FROM_CONFIG** — `tests/test_signals.py:17-21`:
  `SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME`, `D_SET = storage.TIMEFRAME_MS[SETUP_TF]`,
  `START = 1_700_000_000_000`. Never hardcode a timeframe in a test (§8).
- **SINGLE_INDICATOR_MODULE** — `indicators/bollinger.py`: one public function, keyword-only
  `| None = None` with config fallback, output indexed like the input. `indicators/wilder.py:14-18` for
  the leading-NaN contract wording.
- **CONFIG_BLOCK_STYLE** — `config.py:51-53`: purpose comment, then constants each with a trailing
  comment giving units and meaning.

---

## Detector Roster

Registry names are lowercase-hyphen (§3). `kind` equals the registry name, so it becomes a
`metrics.by_bucket` key for free (`metrics.py:33`).

| # | Registry key | Tier | Module | Dir | Origin | level / target_height |
|---|---|---|---|---|---|---|
| 1 | `detector.cup-and-handle` | 1 | `continuation.py` | long | **NEW** — no donor exists | `max(rim_L, rim_R)` / cup depth |
| 2 | `detector.inverse-cup-and-handle` | 1 | `continuation.py` | short | NEW, mirror of 1 | `min(rim_L, rim_R)` / dome height |
| 3 | `detector.head-and-shoulders` | 1 | `reversal.py` | short | **PORT+REFINE** `patterns.py:191` | neckline / head − neckline |
| 4 | `detector.inverse-head-and-shoulders` | 1 | `reversal.py` | long | PORT+REFINE `patterns.py:232` | neckline / neckline − head |
| 5 | `detector.wyckoff-spring` | 1† | `wyckoff_events.py` | long | NEW, narrowed (A1/A2) | range support / range width |
| 6 | `detector.wyckoff-upthrust` | 1† | `wyckoff_events.py` | short | NEW, mirror of 5 | range resistance / range width |
| 7 | `detector.double-top` | 2 | `reversal.py` | short | **PORT** `bruteforce:519` + adapter | intervening trough / peak_mean − trough |
| 8 | `detector.double-bottom` | 2 | `reversal.py` | long | PORT `bruteforce:500` + adapter | intervening peak / peak − trough_mean |
| 9 | `detector.bull-flag` | 2 | `continuation.py` | long | PORT `patterns.py:388` | consolidation high / pole height |
| 10 | `detector.bear-flag` | 2 | `continuation.py` | short | PORT `patterns.py:388` | consolidation low / pole height |
| 11 | `detector.ascending-triangle` | 2 | `continuation.py` | long | REFINE `patterns.py:272` | flat upper line at `end_x` / start range |
| 12 | `detector.descending-triangle` | –‡ | `continuation.py` | short | REFINE, mirror of 11 | flat lower line at `end_x` / start range |
| 13 | `detector.symmetrical-triangle` | –‡ | `continuation.py` | both | PORT `patterns.py:272` as-is | upper & lower at `end_x` / start range |
| 14 | `detector.falling-wedge` | 2 | `continuation.py` | long | REFINE — needs the §327 relaxation | upper line at `end_x` / start range |
| 15 | `detector.rising-wedge` | 2 | `continuation.py` | short | REFINE — needs the §327 relaxation | lower line at `end_x` / start range |
| 16 | `detector.rsi-divergence` | 2 | `oscillator.py` | both | **NEW** on `indicators/rsi.py` | intervening swing / swing height |

† `tier=1` because they sit inside a tier-1 structure; the ledger records the *structure*
(`wyckoff-accumulation` / `-distribution`) as `deferred`, not covered.
‡ Not in §9's tiers. Near-zero-marginal-cost mirrors sharing the same line fit; ledger `tier=None` with
that reason. **No success criterion depends on them.**

**§9 scorecard**: tier 1 → **2 of 3** (Cup & Handle ✓, H&S refined ✓, Wyckoff deferred per A1).
Tier 2 → **6 of 6** (Double Top/Bottom, Bull/Bear Flag, Ascending Triangle, Falling Wedge, Rising
Wedge, RSI Divergence). **Catalog coverage delivered**: 19 of 144 rows (13.2%) — Reversal 4/16,
Continuation 9/15, Wyckoff 2/4, Oscillators-RSI 4/6. Stated plainly, because the PRD says the full
catalog is "a direction, not a v1 gate."

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `indicators/rsi.py` | CREATE | §2 assigns it to Phase 8. Wilder RSI ported from `bruteforce:130`; no new dependency |
| `plugins/detectors/_geometry.py` | CREATE | Shared private helpers ported from `patterns.py` + the relaxed line fit. Underscore-prefixed, registers nothing, safe under `load_all()`'s pkgutil walk |
| `plugins/detectors/reversal.py` | CREATE | Catalog family 1 — detectors 3, 4, 7, 8 |
| `plugins/detectors/continuation.py` | CREATE | Catalog family 2 — detectors 1, 2, 9–15 |
| `plugins/detectors/oscillator.py` | CREATE | Catalog family 14 — detector 16 |
| `plugins/detectors/wyckoff_events.py` | CREATE | Catalog family 9, narrowed per A1/A2 — detectors 5, 6 |
| `plugins/detectors/catalog.py` | CREATE | The ledger: `CatalogEntry`, `CATALOG`, `coverage_summary()` |
| `cli.py` | UPDATE | Append `detector-report` subparser + `_detector_report_command` (§7). Existing subcommands untouched |
| `config.py` | UPDATE | Append the Phase 8 block: `DETECTOR_*` plus per-pattern prefixes (`CUP_`, `DOUBLE_`, `WEDGE_`, `RSI_`, `WYCKOFF_`) and **new** `HS_*`/`TRIANGLE_*` names. Modifies no existing constant |
| `tests/conftest.py` | CREATE | One fixture, `pattern_fixture`. See deconfliction note |
| `tests/fixtures/patterns/*.csv` | CREATE | 4 hand-drawn series for the two shapes too tedious to build parametrically (cup, RSI divergence) |
| `tests/test_detectors_{reversal,continuation,oscillator,wyckoff_events,catalog,report}.py` | CREATE | §8's `test_detectors_<family>.py` reservation |
| `.claude/PRPs/reports/phase8-detector-edge-report.md` | CREATE | Measured edge table, ledger snapshot, diff-footprint proof, degrees of freedom |

**Deconfliction note on `tests/conftest.py`**: it does not exist today and is in no phase's reserved
list. Phase 8 creates it with exactly one fixture. Any later phase **appends**; never rewrites. Flagged
so a parallel Phase 7 does not collide.

**Reserved-name compliance**: config prefixes `DETECTOR_*` + per-pattern names (§7 row 8); CLI
`detector-report` → `_detector_report_command` (§7 row 8); tests `test_detectors_<family>.py` +
`tests/fixtures/patterns/` (§8 row 8). RSI's unit tests live in
`test_detectors_oscillator.py::TestRsiIndicator`, not an unreserved `test_rsi.py`.

## NOT Building

- **Wyckoff Accumulation/Distribution as a phase-labelled structure** (A1) — ledger entry, not code.
- **Harmonics (8 rows), Elliott (2), SMC (8), Fibonacci (7), Gap patterns (4)** — tier 3–4 or worse.
- **The 33 candlestick rows** (families 3–5) — §9 tier 4 "candlestick-alone ⭐⭐☆☆☆".
  `bruteforce:426-487` implements 9; the ledger records donor line numbers so a later port is cheap.
- **Golden Cross / Death Cross / MACD Cross** — tier 3; MACD Cross is Phase 4's module.
- **`indicators/macd.py`, `indicators/stochastic.py`** — Phase 4's, and unassigned.
- **Any `Confirmation`, `PositionPolicy` or `Filter`** — Phase 4 owns those directories. `RSI > 70` /
  `< 30` are Confirmation material, not Detectors; the ledger says so.
- **Vision/ML matching** — `patterns.py:1-5` commits to "explicit, parameterized geometric rules over
  confirmed pivots (no ML, no visual matching)".
- **Any change to `walkforward.py`, `engine.py`, `metrics.py`, `equity.py`, `signals/*`,
  `framework/*`** — if one is needed it is a `FRAMEWORK-GAP` for Phase 3.
- **Cross-detector dedup** (A5). **Tuning any default against the edge report** — that converts a
  diagnostic into an oracle.

---

## Step-by-Step Tasks

Shared conventions, stated once so each task can be terse. Every detector: is a function decorated with
`@register("detector", name=…, params={…ParamSpec…}, rationale=…, tier=…, timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,))`;
takes `(ctx: EvalContext, **params)`; reads bars **only** through `ctx`; emits `DetectedEvent`; ends with
`return g.dedupe_events(out)`; and imports
`from trading_bot import config`, `from trading_bot.framework.contracts import DetectedEvent, EvalContext`,
`from trading_bot.framework.registry import ParamSpec, register`,
`from trading_bot.signals.pivots import find_pivots`,
`from trading_bot.plugins.detectors import _geometry as g`. Every threshold named in a task is a
`ParamSpec` with `bounds`, defaulting to the `config.py` constant of the same name (Task 1) — a detector
never reads `config` for a declared param. `rationale` is mandatory and non-empty (§3).

### Task 1: Config — the Phase 8 block

- **ACTION**: Append one clearly-headed block at the end of `config.py`.
- **IMPLEMENT**: `# --- Phase 8 (v0.3.0): pattern coverage expansion ---`, then
  `DETECTOR_MIN_EVENTS_FOR_REPORT=20`;
  `DETECTOR_REPORT_HOLDOUT_GUARD_DAYS = WF_OOS_DAYS + WF_TEST_DAYS` (=150);
  `DETECTOR_REPORT_CAMPAIGN="detector-report"`; `RSI_PERIOD=14`;
  `CUP_MIN_WIDTH_BARS=20`, `CUP_MAX_WIDTH_BARS=90`, `CUP_MIN_DEPTH=0.08`, `CUP_MAX_DEPTH=0.50`,
  `CUP_RIM_TOLERANCE=0.05`, `CUP_ROUND_BAND=0.25`, `CUP_MIN_BASE_BARS=5`, `CUP_HANDLE_MIN_BARS=3`,
  `CUP_HANDLE_MAX_BARS=20`, `CUP_HANDLE_MAX_RETRACE=0.40`; `DOUBLE_TOLERANCE=0.02`,
  `DOUBLE_MAX_GAP_BARS=60`, `DOUBLE_MIN_SEPARATION_BARS=8`, `DOUBLE_MIN_TROUGH_DEPTH=0.03`;
  `HS_NECKLINE_SLOPED=False`, `HS_TIME_SYMMETRY_TOL=1.0`, `HS_VOLUME_TAPER_REQUIRED=False`;
  `TRIANGLE_FLAT_SLOPE_TOL=0.001`; `WEDGE_MIN_SLOPE=0.001`; `RSI_DIV_LOOKBACK_BARS=120`,
  `RSI_DIV_MIN_SEPARATION_BARS=6`, `RSI_DIV_MAX_SEPARATION_BARS=60`, `RSI_DIV_PIVOT_MATCH_BARS=3`,
  `RSI_DIV_OVERBOUGHT=60.0`, `RSI_DIV_OVERSOLD=40.0`; `WYCKOFF_RANGE_BARS=30`,
  `WYCKOFF_RANGE_MAX_WIDTH_PCT=0.08`, `WYCKOFF_PROBE_MIN_PCT=0.003`, `WYCKOFF_PROBE_VOL_RATIO=1.5`.
- **MIRROR**: `CONFIG_BLOCK_STYLE` — trailing comment giving units on every constant.
- **GOTCHA**: (a) §7 forbids modifying existing constants — the new `HS_*`/`TRIANGLE_*` names go in the
  **Phase 8 block**, so the diff is append-only. (b) The three `HS_*` flags default to the *old*
  behaviour (`False`/`1.0` = permit everything); that is what makes Task 4's parity test possible.
  (c) `DETECTOR_REPORT_HOLDOUT_GUARD_DAYS` derives from Phase 5's `WF_*` because Phase 9's `HOLDOUT_*`
  does not exist yet — repoint it when Phase 9 lands (recorded follow-up, not a hidden coupling).
- **VALIDATE**: `.venv/bin/python -c "from trading_bot import config; print(config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS, config.RSI_PERIOD)"`
  → `150 14`; `git diff --stat src/trading_bot/config.py` shows insertions only.

### Task 2: `indicators/rsi.py`

- **ACTION**: Create the module; port with attribution.
- **IMPLEMENT**: `rsi(close: pd.Series, *, period: int | None = None) -> pd.Series` in [0,100], indexed
  identically, leading NaNs preserved; body is `WILDER_RSI_DONOR` with
  `period = config.RSI_PERIOD if period is None else period`. Docstring states: the attribution ("Ported
  from `scripts/bruteforce/indicators.py:130` per §0b; research-grade code enters `src/trading_bot/`
  only with production-standard tests"); that it uses production `wilder_smooth`, matching the ATR/ADX
  convention; and the NaN rule ("NaN means not knowable yet; never back-fill").
- **MIRROR**: `SINGLE_INDICATOR_MODULE`.
- **IMPORTS**: `numpy as np`, `pandas as pd`, `config`,
  `from trading_bot.indicators.wilder import wilder_smooth`.
- **GOTCHA**: (a) Without the `avg_loss==0 & avg_gain>0 → 100.0` clause RSI is NaN through every
  sustained uptrend, silently killing every bearish-divergence candidate. (b) Do **not** normalise by
  price the way `bruteforce.macd` does — RSI is already scale-free. (c) `wilder_smooth` seeds on a mean
  of the first `period` values, so the first defined value is at index `period`, one later than ATR's
  `period-1` because `diff()` costs a bar. Assert that index; don't assume it.
- **VALIDATE**: `TestRsiIndicator` — monotone rise → exactly `100.0`; monotone fall → `0.0`; alternating
  ±1 → 50.0 ± tol; leading NaN count == `period`; one hand-computed 14-period value with the working in
  the docstring (mirroring `tests/test_wilder.py::TestHandComputedValues`).

### Task 3: `plugins/detectors/_geometry.py` — the shared spine

- **ACTION**: Create the private helper module every family imports.
- **IMPLEMENT**:
  - `dedupe_events(events)` — `DEDUP_RULE` on the contract's field names (`level`, `target_height`).
  - `collapse_runs`, `level_unbroken`, `strictly_ordered`, `line_value` — ported **verbatim including
    docstrings** from `patterns.py:144 / 167 / 181 / 260`; `line_value` keeps its `ValueError` on
    `x1 == x2`.
  - `fresh_factory(n, max_age_bars)` — `FRESHNESS_CLOSURE`.
  - `confirmation_ts(df, pivot_index, span)` = `int(df.index[min(len(df)-1, pivot_index+span)])`: the bar
    on which a pivot first becomes knowable. **The `end_ts` every new pivot-terminated detector uses.**
  - `fit_converging_lines(df, pivots, *, min_pivots_per_side, min_width_bars, max_width_bars,
    min_convergence, containment_tol, fresh) -> LineFit | None`. `LineFit` frozen with `start_x, end_x,
    upper_start, upper_end, lower_start, lower_end, upper_slope, lower_slope, start_range, end_range`.
    Slopes **normalised per bar by the level** (`(y2-y1)/((x2-x1)*y1)`) so tolerances are comparable
    across symbols priced from $0.10 to $100k — the scale-free device of `bruteforce.slope`
    (`indicators.py:100-105`).
- **MIRROR**: `_detect_triangles` (`patterns.py:272-370`). Keep in order: the `max(2, …)` per-side floor
  (295), the `earliest` window (301), the distinct-x guard (306), the **both**-lines-fresh test (311 —
  its comment explains why `min` not `max`), min-width (315), the degenerate/crossed-range guard (332),
  convergence (334), `CONTAINMENT_TEST` (337), price-inside-at-`end_x` (340-342). **Drop exactly one
  thing**: the expanding-side guard 327-328.
- **IMPORTS**: `dataclass`, `pandas as pd`, `DetectedEvent`,
  `from trading_bot.signals.pivots import Pivot` (§1 lists `Pivot` as reused unchanged — do not
  redefine).
- **GOTCHA**: (a) `THE_LINE_THAT_FORBIDS_WEDGES` — dropping 327 without per-detector slope classification
  emits a "symmetrical triangle" for every rising wedge. Relocate the guard's work (Task 6); don't delete
  it. (b) `fit_converging_lines` returns pure geometry — it must **not** decide direction or emit events.
  (c) Underscore-prefixed with no `@register`: verify `plugins.load_all()`'s pkgutil walk imports it
  cleanly and `REGISTRY` gains nothing.
- **VALIDATE**: `py_compile`; `TestFitConvergingLines` — on `tests/test_signals.py:192`'s `wedge()`,
  `upper_slope < 0 < lower_slope` and `end_range < start_range*(1-config.TRIANGLE_MIN_CONVERGENCE)`; on a
  both-lines-rising variant a `LineFit` **is** returned where `patterns.py` returned `[]`.

### Task 4: `reversal.py` — H&S refined (detectors 3, 4)

- **ACTION**: Create `plugins/detectors/reversal.py`; refined H&S pair first.
- **IMPLEMENT**: One `_detect_hs(ctx, *, inverse, **params)` driving two registered functions. Geometry
  is `patterns.py:191-257` on `DetectedEvent`, plus four refinements each defaulting to legacy behaviour:
  (1) `neckline_sloped` — neckline becomes `line_value(t1.index, t1.price, t2.index, t2.price, end_x)`
  instead of `min(t1,t2)`; classical necklines slope, the horizontal `min()` is the conservative special
  case. (2) `time_symmetry_tol` — reject when `abs((head-left)-(right-head))/(right-left) > tol`; `1.0`
  is a no-op. (3) `volume_taper` — require the right shoulder's bar volume below the left's (the
  classical distribution tell); always recorded in `meta["volume_taper_ratio"]` whether or not it gates.
  (4) `shoulder_tol`, `head_prominence`, `pivot_span`, `max_age_bars` promoted from silent config reads
  to declared `ParamSpec`s — §3's reason for replacing `grid` with `params`.
  `rationale`: "§9 tier 1 by reliability. The v0.2.0 geometry measured as a loser under the old
  absolute-percentage risk model and was retired (KNOWN-LIMITATIONS §8); re-offered under the ATR risk
  model with sloping-neckline and volume-taper options, so the *refinements* — not the base shape — are
  what is being tested."
- **MIRROR**: `patterns.py:191-257` line-for-line — `CONSERVATIVE_LEVEL_CHOICE`, the
  both-shoulders-outside-the-neckline tests (215-217 / 240-242), `NOISE_PIVOT_COLLAPSE` before the scan
  (198), `strictly_ordered` on every window (204), then `FRESHNESS_CLOSURE` and `SPENT_LEVEL_RULE` last.
- **GOTCHA**: (a) `end_ts` keeps `window[-1].ts`, **not** `confirmation_ts`, because parity with
  `patterns.py` is load-bearing — Phase 3's `legacy_patterns.py` wraps the same geometry and the two must
  agree. Record in the report that *new* pivot-terminated detectors use the stricter convention and that
  reconciling `legacy_patterns.py` is a follow-up; do not unify here. (b) The bearish window is
  `(high, low, high, low, high)` — wrong order and the detector silently finds nothing. (c) With
  `neckline_sloped=True` the both-shoulders-outside test must evaluate the sloped line at **each
  shoulder's own index**, not at `end_x`. (d) A sloped neckline can land outside the price range at
  `end_x` when the troughs are far apart and steeply offset; clamp nothing — reject when the sloped level
  at `end_x` is not strictly between the head and the nearer shoulder, and comment why.
- **VALIDATE**: `TestHeadAndShouldersParity` — for each of the seven H&S fixtures at
  `tests/test_signals.py:75-169`, the new detector at defaults returns
  `(kind, direction, level, target_height, start_ts, end_ts)` equal to the legacy candidate's
  `(kind, direction, breakout_level, target_height, start_ts, end_ts)` **exactly**. Then
  `TestHeadAndShouldersRefinements` — one test per refinement asserting the default is a no-op and the
  flag changes the outcome on a purpose-built shape.

### Task 5: `reversal.py` — double top / bottom (detectors 7, 8)

- **ACTION**: Port `bruteforce:519`/`:500`; build the level/target adapter the donor lacks.
- **IMPLEMENT**: `_detect_double(ctx, *, inverse, **params)`:
  1. `pivots = g.collapse_runs(find_pivots(df, span=pivot_span))` — not the donor's raw consecutive
     pairing.
  2. Every **ordered pair** of same-kind pivots `(a, b)` with
     `min_separation <= b.index-a.index <= max_gap` and
     `abs(b.price-a.price)/max(a.price,b.price) <= tol`.
  3. **The adapter.** Find the intervening opposite-kind pivot `m` (`a.index < m.index < b.index`), most
     extreme if several. No `m` → skip: two peaks with no trough between them are one peak.
     `level = m.price`; `peak_mean = (a.price+b.price)/2`; `target_height = abs(peak_mean-m.price)`.
  4. Reject when `abs(peak_mean-m.price)/peak_mean < min_trough_depth` — two highs within `tol` with a
     negligible dip is a flat range, not an M. The donor has no such test.
  5. `direction = "short"`/`"long"`; `fresh(b.index)`; `g.level_unbroken(df, level, direction, b.index)`;
     `start_ts = a.ts`; `end_ts = g.confirmation_ts(df, b.index, pivot_span)`;
     `meta = {"peak_a","peak_b","trough","separation_bars"}`.
  `rationale`: "§9 tier 2 (⭐⭐⭐⭐☆). Ported from `bruteforce/indicators.py:519` per §0b; the donor
  returned a boolean Series with no level and no target, so the neckline and measured move are added
  here, with the minimum-trough-depth test the donor lacked."
- **MIRROR**: `bruteforce:519-530` for the pairing/tolerance arithmetic; `CONSERVATIVE_LEVEL_CHOICE`;
  `NOISE_PIVOT_COLLAPSE` for why `zip(idx, idx[1:])` (`bruteforce:527`) is replaced.
- **GOTCHA**: (a) **The lookahead trap.** The donor uses `pivot_high` (`bruteforce:370`), which flags at
  `t+span` carrying the price from `t`; production `find_pivots` returns `index = t` and relies on the
  caller passing a frame sliced to the evaluation bar (`pivots.py:9-12`, `engine.py:335`). Reading
  `find_pivots`' `index` as a confirmation bar creates a `span`-bar lookahead that *improves* every
  backtest. Use `find_pivots` only, via `EvalContext`, with `end_ts` from `confirmation_ts`. (b) Iterate
  **all** ordered pairs: a noise pivot high between the tops is exactly what `collapse_runs` absorbs, and
  the donor's consecutive-only pairing is why it misses most real double tops. (c) `max_gap` is in
  *setup-tier bars* — at 4h the donor's 60 is 10 days; say so in the `ParamSpec` comment. (d) `tol` is
  relative to `max(a,b)` per `patterns.py:213`; the donor divides by `vals[a]` (`bruteforce:528`), which
  is asymmetric in pair order. Use production's convention.
- **VALIDATE**: `TestDoubleTop` — `path_df([90,105,96,105.5,99])` → one short event, `level == 96.0`,
  `target_height == 105.25-96.0`; `path_df([90,105,104.6,105.5,99])` (dip <3%) → none;
  `path_df([90,105,96,112,99])` (peaks 6.3% apart) → none; a pair beyond `max_gap` → none; a noise pivot
  high inserted between the tops → still one event; and
  `test_end_ts_is_the_confirmation_bar_not_the_pivot_bar` asserting `end_ts == df.index[b+PIVOT_SPAN]`.

### Task 6: `continuation.py` — triangles and wedges (detectors 11–15)

- **ACTION**: Create the module; build the classification layer on `g.fit_converging_lines`; register
  five shapes. **`rising-wedge` is implemented LAST and is the zero-engine-edit proof detector
  (Task 15) — it must add no config constant.**
- **IMPLEMENT**: `_classify(fit, *, flat_tol, min_slope) -> str | None`:

  | Shape | upper slope | lower slope | Emits |
  |---|---|---|---|
  | symmetrical triangle | `< -flat_tol` | `> +flat_tol` | **two**: long at `upper_end`, short at `lower_end` |
  | ascending triangle | `abs(u) <= flat_tol` | `> +flat_tol` | one long at `upper_end` |
  | descending triangle | `< -flat_tol` | `abs(l) <= flat_tol` | one short at `lower_end` |
  | falling wedge | `< -min_slope` | `< -min_slope` | one long at `upper_end` ("usually bullish") |
  | rising wedge | `> +min_slope` | `> +min_slope` | one short at `lower_end` ("usually bearish") |

  Five detectors, each calling `fit_converging_lines` once and checking `_classify` equals its own shape.
  All five: `target_height = fit.start_range` (the widest vertical extent — the classical measured move),
  `start_ts = df.index[fit.start_x]`, `end_ts = df.index[fit.end_x]` (the bar the levels are evaluated
  at, `patterns.py:345`), `meta` carrying both slopes and both ranges. The symmetrical pair follows
  `patterns.py:347-349` — "direction is decided by which side actually breaks; emit both candidates and
  let the trigger bar pick at most one."
  `rationale` (ascending): "§9 tier 2 (⭐⭐⭐⭐☆). Horizontal resistance repeatedly tested while lows rise
  is the classical accumulation-into-breakout shape; refined from `signals/patterns.py:272`, whose single
  `triangle` kind conflated five distinct shapes with different directional priors."
  `rationale` (falling wedge): "§9 tier 2 (⭐⭐⭐⭐☆). Two falling, converging boundaries mean sellers are
  losing ground faster than buyers. Not emittable by `signals/patterns.py` at all: its expanding-side
  guard (line 327) rejects any shape whose lower line falls."
  `rising-wedge` declares `min_slope`, `flat_tol`, `min_convergence`, `containment_tol`,
  `min_width_bars`, `max_width_bars`, `max_age_bars`, `pivot_span` **inline in the decorator**, reading
  no `config` attribute at all.
- **MIRROR**: `patterns.py:272-370`, split between `fit_converging_lines` (the tests) and `_classify`
  (the slope signs); `THE_LINE_THAT_FORBIDS_WEDGES` for why the wedges need Task 3.
- **GOTCHA**: (a) `TRIANGLE_FLAT_SLOPE_TOL` is on the **normalised per-bar** slope — `0.001` means 0.1%
  of price per bar; raw slopes make the tolerance meaningless across symbols. (b) The five shapes must be
  **mutually exclusive**; a falling wedge and a descending triangle differ **only** by whether the lower
  line is flat within `flat_tol`, so branch order is load-bearing. (c) Two events with the same `kind`
  and different `direction` is fine under `dedupe_events` — precisely why the key is a pair. (d) The
  catalog says the symmetrical triangle continues "in the prevailing trend"; the detector deliberately
  does **not** determine the prevailing trend — a `Confirmation`'s job (Phase 4). Emitting both sides is
  the honest handling and what the legacy code did. (e) Do not substitute `max(start_range, end_range)`
  for `fit.start_range`; if they ever invert that is a bug to surface, not to paper over. (f)
  `rising-wedge`'s no-config rule is not stylistic — it makes the diff-footprint proof a clean two-path
  assertion. Say so above the decorator.
- **VALIDATE**: `TestTriangleVariants`/`TestWedges` via `wedge()`: symmetrical
  `((5,110),(30,101))/((10,92),(28,97))` → 2 events; ascending
  `((5,110),(30,109.9))/((10,92),(28,104))` → 1 long, `level ≈ 109.9`; descending
  `((5,110),(30,99))/((10,92),(28,92.1))` → 1 short; falling wedge
  `((5,110),(30,101))/((10,100),(28,96))` → one long `falling-wedge` and **zero** triangle events;
  mirrored both-rising → one short `rising-wedge`. Plus the four legacy rejections re-pointed (pierced
  trendline, last close already outside, narrower than `TRIANGLE_MIN_WIDTH_BARS`, one trendline stale —
  `tests/test_signals.py:227-265`); `test_shape_classification_is_mutually_exclusive` over a grid of
  synthetic `LineFit`s; and `test_legacy_triangle_detector_cannot_see_this_shape` asserting
  `detect_patterns(df, pivots) == []` on the wedge frame.

### Task 7: `continuation.py` — bull / bear flag (detectors 9, 10)

- **ACTION**: Port `patterns.py:388-464` — the production geometry, **not** `bruteforce:552`.
- **IMPLEMENT**: `_detect_flag(ctx, *, direction, **params)` reproducing the
  pole-window/consolidation-length scan: for each `c` in `[consol_min, consol_max]` smallest-first,
  `pole_end = n-1-c`, `pole_start = max(0, pole_end-pole_window+1)`, then the three tests
  (`pole_height/closes[pole_end] >= pole_min_pct`; consolidation extreme holds within
  `max_retrace*pole_height`; consolidation extreme does not exceed the **pole window's** extreme).
  `level` = consolidation high (long) / low (short); `target_height` = pole height;
  `start_ts = ts[pole_start]`, `end_ts = ts[n-1]`. One event per direction, tightest consolidation wins
  (`patterns.py:411-415`).
  `rationale`: "§9 tier 2 (⭐⭐⭐⭐☆). Ported from `signals/patterns.py:388`, strictly stronger than the
  `bruteforce/indicators.py:552` donor, which measures the pole from two `shift()`ed closes and has no
  containment test — so it accepts a pole that already gave back its gain."
- **MIRROR**: `patterns.py:388-464` line-for-line, including the docstring paragraph that containment is
  tested against the pole's extreme over the **whole** pole window (402-404) — "checking only the final
  pole bar's high rejected valid flags whose pole peaked a bar or two earlier", pinned by
  `tests/test_signals.py:301-315`.
- **GOTCHA**: (a) Flags **skip freshness** by construction — the consolidation ends at the latest bar
  (`patterns.py:96-98`). Threading `fresh` in rejects every flag whose pole is older than
  `PATTERN_MAX_AGE_BARS`, i.e. most of them. (b) The pole slice is inclusive (`pole_start:pole_end+1`),
  so `pole_start` subtracts `pole_window-1`, not `pole_window` (`patterns.py:419-420`; pinned by
  `tests/test_signals.py:290-299`'s crater bar just outside the window). (c) `level_unbroken` is also
  **not** applied: the consolidation extreme by definition has not been closed through, or the
  consolidation would have ended.
- **VALIDATE**: `TestFlags` — `flag_1h_df()` (`tests/test_signals.py:268`) → one long event,
  `level == 109.5`, `target_height > 0`; 40 flat bars → none; the crater-at-bar-13 test asserting
  `target_height` unchanged; the pole-peaks-early test asserting `level == 110.4`.

### Task 8: `continuation.py` — cup & handle and its inverse (detectors 1, 2)

- **ACTION**: Build new geometry. **No donor exists anywhere in the repo** — the one genuinely new tier-1
  build.
- **IMPLEMENT**: `_detect_cup(ctx, *, inverse, **params)`:
  1. `pivots = g.collapse_runs(find_pivots(df, span=pivot_span))`.
  2. Ordered triples (high `L`, low `B`, high `R`), `strictly_ordered`,
     `min_width <= R.index-L.index <= max_width`.
  3. **Rim symmetry**: `abs(R.price-L.price)/max(R.price,L.price) <= rim_tolerance`.
  4. **Depth**: `depth = max(L.price,R.price) - B.price`; require
     `min_depth <= depth/max(L.price,R.price) <= max_depth`. Under 8% is noise; over 50% is a crash with
     a bounce.
  5. **Roundness** — the only novel geometry here and the test separating a cup from a V. Count bars in
     `[L.index, R.index]` whose low sits within the bottom `round_band` (25%) of the depth
     (`low <= B.price + round_band*depth`); require at least `min_base_bars` (5). A V bottom has one or
     two such bars; a rounded base has many. Decidable, cheap, testable by construction — the
     alternative (fitting a parabola and bounding residuals) adds two fitted parameters for no measured
     benefit.
  6. **Bottom centrality**: `B.index` in the middle half of `[L.index, R.index]`, so a base hugging one
     rim is rejected.
  7. **Handle**: `handle_min <= n-1-R.index <= handle_max`; handle's lowest low
     `>= R.price - handle_max_retrace*depth` **and** `> B.price + depth/2` (a handle giving back more
     than half the cup is a failed cup); handle's highest high must not exceed `R.price`, or the rim
     already broke.
  8. `level = max(L.price, R.price)` (`CONSERVATIVE_LEVEL_CHOICE` applied to rims);
     `target_height = depth`; `direction = "long"`; `start_ts = L.ts`; `end_ts = ts[n-1]`;
     `meta = {"depth_pct","base_bars","handle_bars","rim_asymmetry"}`;
     `g.level_unbroken(df, level, "long", R.index)`. Freshness **skipped**: the handle ends at the latest
     bar by construction, exactly like a flag (`patterns.py:96-98`).
  9. `inverse=True` mirrors every inequality (low rims, high dome, handle bounce).
  `rationale`: "§9 tier 1 (⭐⭐⭐⭐⭐) — the highest-reliability entry in the catalog's table and the only
  tier-1 shape with no implementation anywhere in this repo. The roundness test (a minimum bar count
  inside the bottom quartile of depth) is what distinguishes it from a V reversal; without it this
  detector is a slow double bottom."
- **MIRROR**: `patterns.py:191-257` for the multi-pivot-window scan and tolerance style;
  `patterns.py:388-464` for "structure ends at the latest bar, so skip freshness".
- **GOTCHA**: (a) With `PIVOT_SPAN = 3` a 20–90 bar cup usually holds several noise pivot lows near the
  base; `collapse_runs` merges only *adjacent same-kind* pivots, so an alternating micro-high/micro-low
  base survives and offers many (L,B,R) triples. Bound the scan: iterate `R` over the last
  `pivot_span*4` pivot highs only, and break the `L` loop once `R.index-L.index > max_width`. Unbounded
  this is O(p³) on a 180-bar lookback. (b) The catalog lists cup & handle as a *continuation* pattern,
  but the detector deliberately does **not** test for a prior uptrend — a `Confirmation`'s job
  (Phase 4); baking it in would hide the decision. State this in the docstring so nobody adds it
  silently. (c) `round_band` and `min_base_bars` interact — widen the band and any V passes with enough
  bars. Both are recorded in the report as **degrees of freedom consumed by geometry choice, not by
  fitting**: chosen from the classical definition *before* any measurement, and that must stay true.
- **VALIDATE**: `TestCupAndHandle` against `cup_and_handle_positive.csv` (60 bars: rims 100 / 100.8,
  base 88, 8-bar handle to 96) → one long event, `level == 100.8`, `target_height ≈ 12.8`. Negatives,
  each a file or a one-bar mutation: `cup_v_bottom_negative.csv` (2 base bars → roundness fails); rims
  12% apart; depth 4%; handle retracing 70% of depth; handle high above the rim; base centred at 15% of
  the span.

### Task 9: `oscillator.py` — RSI divergence (detector 16)

- **ACTION**: Create the module. Build divergence on price pivots **and** oscillator pivots.
- **IMPLEMENT**:
  1. `series = rsi(df["close"], period=rsi_period)`.
  2. **Oscillator pivots via the same fractal implementation.** Build
     `osc = pd.DataFrame({"open": series, "high": series, "low": series, "close": series,
     "volume": 0.0}, index=df.index)` with the NaN warmup rows dropped, then
     `find_pivots(osc, span=pivot_span)`. One fractal algorithm, one confirmation-lag rule, no second
     implementation to drift.
  3. `price_pivots = g.collapse_runs(find_pivots(df, span=pivot_span))`;
     `osc_pivots = g.collapse_runs(find_pivots(osc, span=pivot_span))`.
  4. **Pairing.** Bearish, `mode="regular"`: consecutive price pivot **highs** `(i, j)` with
     `min_sep <= j.index-i.index <= max_sep` and `j.price > i.price` (higher high); osc pivot highs
     `oi`, `oj` within `pivot_match_bars` of `i` and `j`; require `oj.price < oi.price` (lower RSI high)
     and `oi.price >= overbought`. Bullish mirrors on lows with `oi.price <= oversold`. `mode="hidden"`
     reverses the **price** inequality only (lower price high with a higher RSI high → hidden bearish),
     covering the catalog's two Hidden Divergence rows with one detector and a `choice` param.
  5. **Level and target — a stated convention, not a classical measured move.** `level` = the price of
     the intervening opposite-kind price pivot between `i` and `j` (for bearish, the swing **low**): the
     structure that must break for the reversal to be more than a wiggle, the same logical role the H&S
     neckline plays. `target_height = abs(j.price - level)`. Rejected alternative: an ATR multiple,
     which puts a fitted parameter inside a detector.
  6. `end_ts = g.confirmation_ts(df, max(j.index, oj.index), pivot_span)`;
     `g.level_unbroken(df, level, direction, j.index)`; `fresh(j.index)`;
     `meta = {"rsi_i","rsi_j","rsi_delta","price_delta_pct","separation_bars","mode"}`.
  `rationale`: "§9 tier 2 (⭐⭐⭐⭐☆) and the single largest hole in v0.2.0's search:
  KNOWN-LIMITATIONS §0c records RSI and divergence as **absent entirely** from every sweep. Price making
  a new extreme while momentum does not is the classical exhaustion read; the intervening swing is the
  confirmation level, so the event is a structure break rather than an opinion."
- **MIRROR**: `WILDER_RSI_DONOR`; `PIVOT_CONFIRMATION_LAG` (both citations); `NOISE_PIVOT_COLLAPSE`;
  `SPENT_LEVEL_RULE`; `CONSERVATIVE_LEVEL_CHOICE`. Extra import:
  `from trading_bot.indicators.rsi import rsi`.
- **GOTCHA**: (a) **Divergence is where lookahead bugs hide.** A pivot is confirmed only `PIVOT_SPAN`
  closed bars later (`pivots.py:9-12`), and *two* pivots are involved — price and RSI — which can confirm
  on different bars. `end_ts` must be the **later** of the two, hence `max(j.index, oj.index)`. Taking
  the price pivot alone lets `check_breakout` fire on a bar where the RSI pivot was not yet knowable;
  pinned by a named regression test. (b) `find_pivots` rejects ties (`pivots.py:81-84`); RSI plateaus at
  exactly `100.0` through a pure uptrend, so **no RSI pivot is emitted there** — correct behaviour (a
  flat oscillator has no swing), not a bug. A fixture must assert it so nobody "fixes" it by loosening
  to `>=`. (c) The RSI warmup is NaN for the first `RSI_PERIOD` bars; NaNs make every comparison `False`
  and silently yield no pivots near the start. Slice the warmup off explicitly, keep index alignment,
  **never `fillna`**. (d) `overbought = 60.0`/`oversold = 40.0` are deliberately not 70/30: pairing two
  RSI highs demands the first be *elevated*, not extreme, and 70 eliminates most real pairs. Both are
  `ParamSpec`s with bounds so the choice is testable rather than assumed. (e) Do not consume
  `indicators/macd.py`: MACD divergence is tier 3 (§9) and `macd.py` is Phase 4's file. A
  `macd-divergence` detector is a ledger row, not a task.
- **VALIDATE**: `TestRsiDivergence` against `rsi_bearish_divergence_positive.csv` (80 bars: warmup,
  swing high 105 with RSI ≈ 72, pullback to 98, higher high 108 with RSI ≈ 64) → one short event,
  `level == 98.0`; `rsi_no_divergence_negative.csv` (both highs and both RSI highs rising) → none; a
  monotone-rise fixture → none, docstring naming gotcha (b); `mode="hidden"` on a
  lower-price-high/higher-RSI-high series → one event; and
  `test_end_ts_is_the_later_of_the_two_pivot_confirmations` asserting
  `end_ts == df.index[max(j,oj)+PIVOT_SPAN]`.

### Task 10: `wyckoff_events.py` — spring / upthrust (detectors 5, 6)

- **ACTION**: Create the module; lead with the precision statement; build only the narrowed events.
- **IMPLEMENT**: **The module docstring is a deliverable**, stating A1 and A2 verbatim: what is detected
  (a probe below/above an established range closing back inside on elevated volume), what is **not**
  (accumulation/distribution as a phase sequence), why (no agreed numeric definition, no reference
  implementation, no labelled dataset — PRD Research Summary), and what correctness validation is
  therefore limited to (geometry-by-construction fixtures; the rule is validated, the concept is not).
  Then `_detect_probe(ctx, *, inverse, **params)`:
  1. **Range** over the trailing `range_bars` bars ending at `n-2` (**excluding the probe bar**):
     `hi = highs.max()`, `lo = lows.min()`; require `(hi-lo)/closes[-1] <= range_max_width_pct` — the
     `TIGHT_RANGE_DEFINITION` idea, so the probe has a defined, small risk.
  2. **No prior breakdown**: no *close* below `lo` (spring) / above `hi` (upthrust) within those bars. A
     probe of a level price already left is a continuation, not a spring.
  3. **The probe bar** is the latest: `low < lo*(1-probe_min_pct)` **and** `close > lo` (the reclaim).
     Upthrust mirrors.
  4. **Volume**: `volume[-1] / mean(volume[-1-VOLUME_LOOKBACK:-1]) >= probe_vol_ratio`. Reuse
     `config.VOLUME_LOOKBACK`; add no second lookback constant.
  5. `level = lo`/`hi`; `target_height = hi-lo` (the range the market is expected to leave — a stated
     convention); `start_ts = ts[n-1-range_bars]`; `end_ts = ts[n-1]`;
     `meta = {"range_high","range_low","range_width_pct","probe_depth_pct","volume_ratio"}`. Freshness
     skipped (the probe **is** the latest bar); `level_unbroken` subsumed by step 2.
  `rationale`: "One event Wyckoff schematics place inside Phase C of an accumulation. It does NOT detect
  accumulation, and cannot distinguish a spring from an ordinary failed breakout, because that
  distinction lives in a multi-week phase sequence with no agreed numeric definition and no reference
  implementation to check against (§9, PRD Research Summary). Registered so the buildable subset is
  measurable; the structure itself is DEFERRED in `catalog.py`."
- **MIRROR**: `TIGHT_RANGE_DEFINITION`; `signals/breakout.py:20-24`'s treatment of volume as a graded
  input — except here volume is a **hard** condition, and the docstring must say this is a deliberate
  departure, because a probe *without* volume is precisely the ordinary failed breakout the detector
  cannot otherwise exclude.
- **GOTCHA**: (a) The range must exclude the probe bar, or the probe's own low defines the support it is
  probing and the detector never fires. (b) `close > lo` strictly, not `>=`: a close exactly at support
  is ambiguous, and `pivots.py:5-6` sets the house convention that ties reject. (c) Short volume history
  makes the rolling mean NaN; `NaN >= ratio` is `False`, so the detector correctly emits nothing —
  assert that rather than guarding it, mirroring `tests/test_signals.py:366-371`. (d) Resist adding
  "Phase A/B/C/D" labels to `meta`; labelling phases is the claim A2 forbids.
- **VALIDATE**: `TestSpring` — 30 flat bars in [98,102] (vol 10) then one bar
  `low=97.0, close=99.5, volume=30` → one long event, `level == 98.0`, `target_height == 4.0`.
  Negatives: probe closing at 97.5 (breakdown, not spring); probe volume 10 (ratio 1.0); a 20%-wide
  range; a prior bar already closing at 96 inside the window; probe low 97.95 (shallower than
  `probe_min_pct`). Plus `test_docstring_states_the_non_claim` asserting the docstring contains "does NOT
  detect accumulation" — a test pinning the honesty requirement.

### Task 11: `catalog.py` and `cli.py detector-report`

- **ACTION**: Create the coverage ledger, then the reserved subcommand where "detection ≠ edge" becomes
  executable.
- **IMPLEMENT (ledger)**: `CatalogEntry` frozen dataclass — `family_no: int` (1..18, matching the
  catalog's headings), `family: str`, `pattern: str` (the **exact** left-column text), `tier: int|None`
  (from the reliability table; `None` if unlisted), `status: str`
  (`"covered"|"deferred"|"out-of-scope"`), `detector_key: str|None`, `reason: str` (required unless
  covered). `CATALOG: tuple[CatalogEntry, ...]` with **exactly 144 entries**;
  `coverage_summary() -> dict` by tier, family and status.
  - **`covered`** — the 19 rows the 16 detectors serve.
  - **`deferred`** — six: `Wyckoff Accumulation`/`Distribution` (A1 verbatim); `Triple Top`/`Triple
    Bottom` ("a double-top generalisation, one line of `_detect_double` away, left for a later tier so
    this phase's degrees of freedom stay countable"); `Bull Pennant`/`Bear Pennant` ("a flag with a
    converging rather than parallel consolidation; needs `fit_converging_lines` inside the flag scan — a
    genuine small build, deferred to keep tier 2 closed").
  - **`out-of-scope`** — the remaining 119, each with one of five distinct reasons: *tier 3–4 by §9*;
    *belongs to another plug-in kind* (`RSI > 70` is Confirmation material; `Higher High`/`Lower Low`
    are market-structure inputs, not tradeable events alone); *no reference implementation and no
    labelled data* (harmonics, Elliott, SMC); *owned by another phase* (`MACD Cross` → Phase 4);
    *already covered by a non-Phase-8 plug-in* (`Bollinger Band Squeeze` → `bollinger_fade.py`;
    `Channel Breakout`/`Resistance Breakout` → `donchian.py`). Where a donor exists, name it with its
    line number so a later port is cheap — e.g. `Hammer` → "`bruteforce/indicators.py:442`, tier 4".
- **IMPLEMENT (CLI)**: Parser `detector-report` with `--symbol` (append), `--start`/`--end`
  (`type=_date_arg`), `--detector` (append, filters to registry keys), `--coverage` (print the ledger
  summary and exit 0 without running anything), `--out`. Dispatch appended to `main()` in phase order.
  `_detector_report_command(conn, symbols, *, start_ms, end_ms, detectors=None, coverage=False,
  out_path=None) -> int`. Per (detector, symbol):
  1. **Holdout guard, first, before any work.** Return `2` if
     `end_ms > now_ms - DETECTOR_REPORT_HOLDOUT_GUARD_DAYS*86_400_000`, printing the computed boundary
     date and the reason: Phase 9 owns a span no diagnostic may see. Mechanism, not promise.
  2. Build a **canonical single-detector diagnostic graph**: `ohlcv` DataSource → the one detector at its
     `ParamSpec` **defaults** → Phase 4's `policies/measured_move.py` → **no** Confirmations, **no**
     Filters. Serialize it so it is reproducible and inspectable.
  3. `trades = run_graph_backtest(conn, graph, symbol, start_ms=…, end_ms=…)` — §5's single graph→`Trade`
     seam, with **all cost arguments left at their `None` defaults** so `config.FEE_PCT` /
     `SLIPPAGE_PCT` / `FUNDING_PCT_PER_DAY` apply exactly as production applies them. No second P&L path
     exists in this phase; enforced by a test.
  4. `compute_metrics(trades)` for `n_trades`, `win_rate`, `expectancy_pct`, `avg_win_pct`,
     `avg_loss_pct`, `profit_factor`, `max_drawdown_pct`. Event/trigger counts come from the
     detector-level counters recorded in `meta`.
  5. Cost-honesty columns: `mean_cost_pct` and `c = mean_cost_pct / median_risk_pct` against
     `config.COST_RATIO_CEILING = 0.10` (`config.py:183`). A detector whose `c` exceeds the ceiling is
     *structurally* unable to pay for itself regardless of win rate — the most useful column in the
     table, and measured rather than asserted.
  6. **`INSUFFICIENT` gating**: if `n_trades < DETECTOR_MIN_EVENTS_FOR_REPORT` (20), print
     `INSUFFICIENT(<20)` and **suppress every rate and expectancy figure**. KNOWN-LIMITATIONS §1 records
     23 trades failing a floor of 30; printing an expectancy off 9 trades repeats that error one level
     down. Suppression is what stops the table being mined.
  7. **Trial ledger**: one `backtest/trials.py` row per (detector, params-hash, span) under
     `campaign=config.DETECTOR_REPORT_CAMPAIGN`. §4.2 — nothing scores a candidate without a ledger
     handle, and this report is not exempt.
  8. **The banner**, printed and written into `--out`: `DIAGNOSTIC — NOT A GATE VERDICT`; the span;
     `fixed exit policy, nothing swept`; the ledger row count; and `SELECTION COST IF USED: if any
     detector is dropped from a campaign because of this table, add n_trials += <N> to that campaign`,
     `N` = the number of detectors evaluated. Free to *read*; costed the moment it *informs* selection.
  9. Exit `0` when a report is produced (a detector with no edge is a result, not an error — mirroring
     `_backtest_command`'s "a backtest with zero trades is a result"), `1` on a data error, `2` on the
     holdout guard.
- **MIRROR**: `cli.py:129-146` (parser), `199-213` (dispatch), `356-368` (`_fmt`/`_print_metrics`, to
  reuse), `283-284` (aligned header); `patterns.py:50-55`'s `PATTERN_KINDS` for the ledger's
  module-constant style; `bruteforce/registry.py:45-57`'s `FAMILIES` for the "keep families to a known
  set so the report doesn't fragment" rationale.
- **IMPORTS**: `compute_metrics` (already at `cli.py:16`);
  `from trading_bot.framework.execute import run_graph_backtest`;
  `from trading_bot.framework.registry import by_kind, load_all`;
  `from trading_bot.framework.graph import NodeSpec, StrategyGraph`;
  `from trading_bot.backtest import trials`;
  `from trading_bot.plugins.detectors.catalog import coverage_summary`.
- **GOTCHA**: (a) The dominant failure mode is quietly becoming a fitness oracle. Steps 1, 2, 3, 6, 7, 8
  all exist to prevent it; removing any reopens the hole. Specifically: **never** add a `--param`
  override, **never** loop over parameter values, **never** sort the output by expectancy — sorting by
  expectancy *is* selection. Sort by tier then name. (b) Passing explicit fees "for the report" would let
  report and production disagree; leave them unset. (c) `load_all()` import errors are fatal by §3 — do
  not catch them; a family missing from the table would read as "tested and found wanting" (§3's own
  words). (d) `--coverage` must touch neither DB nor ledger. (e) `pattern` must be the catalog's
  **exact** left-column text — Task 14 parses the markdown and compares sets, so a paraphrase fails,
  which is the point. (f) `detector_key` is not unique across entries (`rsi-divergence` serves four
  rows); the uniqueness invariant runs the other way. (g) `out-of-scope` must not become a dumping ground
  with one copy-pasted reason — the five categories are distinct and load-bearing.
- **VALIDATE**: Tasks 14 and 15.

### Task 12: `tests/conftest.py` and the fixture files

- **ACTION**: Create the shared loader and four CSVs.
- **IMPLEMENT**: One fixture, `pattern_fixture`, resolving
  `Path(__file__).parent / "fixtures" / "patterns"`, reading `ts,open,high,low,close,volume` with
  `comment="#"`, casting `ts` to `int`, setting the index, and **asserting the index is strictly
  increasing and evenly spaced by `storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME]`** — so a
  hand-edited CSV cannot silently introduce a gap that makes bar arithmetic meaningless. Files:
  `cup_and_handle_positive.csv`, `cup_v_bottom_negative.csv`, `rsi_bearish_divergence_positive.csv`,
  `rsi_no_divergence_negative.csv`.
- **MIRROR**: `TIER_CONSTANTS_FROM_CONFIG` — the CSV `ts` step is derived from config, never hardcoded,
  so a future tier shift cannot leave these fixtures green on the old timeframe (§8).
- **GOTCHA**: (a) Everything parametric stays a **Python builder** in its test module, matching `wedge()`
  (`tests/test_signals.py:192`) and `path_df()` (`:32`). CSVs are only for the cup and the divergence
  series, whose shapes are genuinely hand-drawn; do not migrate the existing builders into files.
  (b) Generating a fixture by *running the detector* is circular: every CSV's expected outcome must be
  derivable by reading the numbers, and each file's first line is a `#` comment stating the shape and the
  expected verdict. (c) Append to `conftest.py`, never rewrite.
- **VALIDATE**: `.venv/bin/python -m pytest tests/ -q --collect-only` still collects — a broken conftest
  breaks the whole suite, so validate collection before writing any test that uses it.

### Task 13: the four detector test modules

- **ACTION**: Create `tests/test_detectors_{reversal,continuation,oscillator,wyckoff_events}.py` with the
  cases enumerated in the VALIDATE lines of Tasks 2, 4–10.
- **IMPLEMENT**: Uniform module shape — header constants from `TIER_CONSTANTS_FROM_CONFIG` plus
  `PIVOT_SPAN = config.PIVOT_SPAN`; `from tests.test_signals import make_df, path_df, wedge`
  (`tests/__init__.py` exists, so `tests` is a package — one builder, one behaviour);
  `@pytest.fixture(autouse=True)` clearing framework caches around every test, mirroring
  `tests/test_backtest.py`'s `engine.clear_caches()` discipline (§8: isolation must not *depend* on a
  fingerprint argument being right); a `_detect(key, df, pivots=None, **overrides)` helper resolving the
  detector out of `REGISTRY` **by key** and calling it through a minimal in-memory `EvalContext`, so
  tests exercise the **registered** plug-in and a missing `@register` fails loudly; one class per
  detector; every rejection test's docstring stating **which real failure mode** it prevents, in the
  style of `tests/test_signals.py:227-234`. Lookahead regressions carry the finding in the name.
- **MIRROR**: `TEST_STRUCTURE`; the whole of `tests/test_signals.py:46-315` for tone and density.
- **GOTCHA**: (a) `path_df` bars have `open=high=low=close`, i.e. **zero range** — any detector reading
  `high-low` (Wyckoff width, containment tolerance) behaves degenerately. Use `make_df` with explicit
  ranges there, as `flag_1h_df()` (`:268`) and `wedge()` do. (b) `PATTERN_MAX_AGE_BARS = 12` is 2 days
  at the 4h tier; a fixture with a long tail after the last pivot fails freshness for reasons that look
  like a geometry bug — when a test intends staleness, say so in the name
  (`tests/test_signals.py:108-111` is the precedent). (c) **Never** assert an absolute `len(REGISTRY)`:
  Phases 3, 4, 6, 7 also register plug-ins, and a hardcoded total turns every parallel phase into a
  broken build. Assert membership of specific keys.
- **VALIDATE**: the four modules pass individually, then the full suite.

### Task 14: `tests/test_detectors_catalog.py` — the ledger invariants

- **ACTION**: Make the ledger machine-checked. Without this test, `catalog.py` is a comment.
- **IMPLEMENT**: `class TestCatalogLedger:` with exactly:
  1. `test_every_covered_entry_has_a_registered_detector` — after `load_all()`, each
     `status == "covered"` entry's `detector_key` is in `REGISTRY`.
  2. `test_every_phase8_detector_appears_in_the_catalog` — every `detector.*` key registered from a
     `trading_bot.plugins.detectors.*` module, excluding Phase 3/4-owned modules by name
     (`legacy_patterns`, `donchian`, `bollinger_fade`, `macd_cross`), appears in at least one entry. No
     orphan detectors.
  3. `test_non_covered_entries_state_a_reason` — every `deferred`/`out-of-scope` entry has a non-empty
     `reason` of at least 20 characters, and no reason appears more than 40 times (a blanket copy-paste
     defeats the ledger).
  4. `test_catalog_matches_the_source_document` — parse `.claude/technical-pattern.md`, collect every
     table row's left-column text from the 18 **numbered** family sections (excluding the
     `# Highest Historical Reliability` table's 15 rows), assert set equality with
     `{e.pattern for e in CATALOG}` and count **144**. Resolve the path via
     `Path(__file__).resolve().parents[1]`; `pytest.skip` with an explicit message if absent (a source
     distribution has no `.claude/`).
  5. `test_tier_scorecard` — every tier-2 pattern named in §9 is `covered`; of the three tier-1
     patterns, `Cup and Handle` and `Head & Shoulders` are `covered` and `Wyckoff Accumulation` /
     `Wyckoff Distribution` are `deferred`. If someone later covers Wyckoff, this tells them to update
     A1 rather than letting scope drift silently.
- **MIRROR**: §8's "regression tests **pin** each repaired finding, with the finding id in the test name
  or docstring" — here the "finding" is scope decision A1.
- **GOTCHA**: (a) The parse must skip the reliability table or the count is 159 (measured: 159 total
  table rows − 15 reliability rows = 144). Anchor on the `# <n>. ` headings. (b) Rows with parentheses
  (`Rounded Bottom (Saucer)`, `Runaway (Measuring) Gap`) are compared as exact literals; strip only
  surrounding pipes and whitespace. (c) This test reads a file outside the package deliberately: the
  catalog document *is* the specification, and a ledger that cannot drift from its spec is the only kind
  worth having.
- **VALIDATE**: 5 passed. Then deliberately corrupt one entry's `pattern` text and confirm invariant 4
  fails loudly.

### Task 15: `tests/test_detectors_report.py`, the zero-edit proof, and the phase report

- **ACTION**: Test the report's methodology (not its numbers), measure the diff footprint of the last
  detector, write `.claude/PRPs/reports/phase8-detector-edge-report.md`.
- **IMPLEMENT (tests)**: on a synthetic in-memory SQLite DB seeded bar-by-bar (§8;
  `START = 1_700_000_000_000`): `test_holdout_guard_rejects_recent_end` (exit 2, zero ledger rows);
  `test_coverage_flag_touches_neither_db_nor_ledger`; `test_insufficient_suppresses_expectancy` (a
  9-trade detector prints `INSUFFICIENT(<20)` and no `exp%`); `test_one_ledger_row_per_detector`;
  `test_no_second_pnl_path` — asserts by source inspection that `_detector_report_command` contains no
  arithmetic on `FEE_PCT`/`SLIPPAGE_PCT`/`FUNDING_PCT_PER_DAY` and calls `run_graph_backtest` (a crude
  but effective structural guard, in the spirit of §1's "any new execution path charges costs
  identically or it is lying"); `test_banner_states_selection_cost`; and
  `test_report_is_sorted_by_tier_then_name_not_by_expectancy`, pinning the anti-mining rule so a later
  "helpful" sort cannot slip in. Mirror `tests/test_signals.py:515-521`'s `seed_candles` and `:576-609`'s
  `TestSignalCommand` shape (monkeypatch the collaborator, assert on `capsys` and the exit code). The
  holdout-guard test must compute `end_ms` from `time.time()` at test time, not a literal, or it starts
  passing for the wrong reason once the clock moves past it.
- **IMPLEMENT (proof)**: implement `detector.rising-wedge` **last**, in its own commit, after every
  shared helper exists. Run `git diff --name-only HEAD~1 HEAD`; record the output **verbatim**. Assert
  the changed-path set is exactly `{src/trading_bot/plugins/detectors/continuation.py,
  tests/test_detectors_continuation.py}` — two paths; no `config.py`, `cli.py`, `framework/`, `signals/`,
  `backtest/`. That is the PRD's "new detector or rule added with zero engine-core edits", **measured**
  rather than claimed — which is why Task 6 forbids `rising-wedge` from adding a config constant. **If
  the set is larger, do not adjust the assertion.** Record a `FRAMEWORK-GAP-<n>` naming the exact
  expression that could not be written inside `plugins/detectors/`, and hand it to Phase 3. A framework
  needing an engine edit per detector has failed its only purpose, and hiding that here would let
  Phase 9 draw conclusions from a broken premise.
- **IMPLEMENT (report)**: (i) the measured edge table for all 16 detectors, banner included; (ii) the
  ledger snapshot (19/144 covered, 6 deferred, 119 out-of-scope; tier 1 2/3, tier 2 6/6); (iii) the
  diff-footprint proof; (iv) which detectors use `confirmation_ts` and which use the legacy pivot-bar
  `end_ts`, so Phase 9 knows; (v) **degrees of freedom consumed** (§12.6): 16 detector evaluations at
  fixed defaults = 16 ledger rows under `campaign="detector-report"`, plus the stated geometry choices
  (`CUP_ROUND_BAND`, `CUP_MIN_BASE_BARS`, `RSI_DIV_OVERBOUGHT`, `RSI_DIV_OVERSOLD`,
  `WYCKOFF_PROBE_VOL_RATIO`) picked from classical definitions *before any measurement* and not to be
  retuned after seeing the table; (vi) two follow-ups — reconciling `legacy_patterns.py`'s `end_ts`, and
  repointing `DETECTOR_REPORT_HOLDOUT_GUARD_DAYS` at Phase 9's `HOLDOUT_*`. **Every number is produced
  by a committed command**, printed beside it (§12.5; the repo already committed once to "use measured
  rather than derived figures").
- **MIRROR**: `KNOWN-LIMITATIONS.md`'s tone — measured tables, explicit verdicts, and a section for what
  is *not* established.
- **GOTCHA**: do not let the table's worst rows become a reason to retune a detector inside this phase. A
  bad measurement is the phase's *output*; retuning against it converts the diagnostic into an oracle
  and spends degrees of freedom Phase 9 will be charged for without a record.
- **VALIDATE**: 7 tests pass; the report exists; every table has its generating command; the
  diff-footprint section contains literal `git diff --name-only` output.

---

## The Detector Template (for every tier-3/4 detector added after this phase)

1. **Ledger first.** Flip the `CATALOG` entry to `covered` with a `detector_key`. Task 14's invariant 1
   now fails — that failing test is the spec.
2. **Choose the module** by catalog family; create `plugins/detectors/<family>.py` if none exists. Never
   a new sibling directory (§2).
3. **Check for a donor** in `scripts/bruteforce/indicators.py`; cite its line number in the module
   docstring. Port, do not rewrite (§0b). Nine candlestick donors sit at `indicators.py:426-487`.
4. **Write the adapter honestly.** A boolean-`pd.Series` donor owes you a `level` and a
   `target_height`. If the pattern has no classical measured move, say so, state the convention you
   chose, and state the alternative you rejected and why.
5. **Register** with `kind`, `tier` from the reliability table, `timeframes`, a **non-empty `rationale`
   stating the prior**, and a `ParamSpec` per threshold with bounds — no silent config reads for
   anything declarable.
6. **Plug into the shared discipline**: `dedupe_events` as the last statement; `fresh_factory` unless
   the structure terminates at the latest bar (then say why, citing `patterns.py:96-98`);
   `level_unbroken` unless subsumed; `confirmation_ts` for `end_ts` on anything pivot-terminated.
7. **Geometry fixtures**: one positive known by construction, one negative per rejection clause, each
   naming the failure mode it prevents.
8. **Edge report**: `detector-report --detector <key>`. If `n_trades < 20`, the honest output is
   `INSUFFICIENT`, not a smaller threshold.
9. **Footprint check**: `git diff --name-only` shows only the family module and its test file. Anything
   else is a `FRAMEWORK-GAP` for Phase 3.

---

## Testing Strategy

Per-detector fixture cases live in each task's VALIDATE line (not duplicated here). The table below is
the **cross-cutting** set — the tests that would still matter if every detector were rewritten.

| Test | Why it matters |
|---|---|
| `TestHeadAndShouldersParity` | The refined port must not silently change a shape Phase 3's `legacy_patterns.py` also wraps |
| `test_end_ts_is_the_confirmation_bar_not_the_pivot_bar` (double top) | The `pivot_high` vs `find_pivots` convention clash — a `span`-bar lookahead that *improves* backtests |
| `test_end_ts_is_the_later_of_the_two_pivot_confirmations` (divergence) | Two pivots, two confirmation lags; taking one is a real lookahead |
| `test_legacy_triangle_detector_cannot_see_this_shape` | Pins *why* `patterns.py:327` had to be relaxed |
| `test_shape_classification_is_mutually_exclusive` | Deleting 327 without exclusivity relabels every wedge a "triangle" |
| `test_v_bottom_is_not_a_cup` | Without roundness, cup & handle is a slow double bottom |
| `test_rsi_plateau_yields_no_oscillator_pivot` | `find_pivots` rejects ties by design; asserting it stops a future "fix" to `>=` |
| `test_docstring_states_the_non_claim` (Wyckoff) | Pins the honesty requirement itself |
| `test_rsi_hand_computed_14_period` | Mirrors `tests/test_wilder.py::TestHandComputedValues` |
| `test_catalog_matches_the_source_document` | The only thing making coverage *tracked* rather than claimed |
| `test_tier_scorecard` | Pins scope decision A1 against silent drift |
| `test_holdout_guard_rejects_recent_end` | Keeps the diagnostic out of Phase 9's holdout mechanically |
| `test_insufficient_suppresses_expectancy` | KNOWN-LIMITATIONS §1's lesson one level down |
| `test_no_second_pnl_path` | §1: a new execution path charges costs identically or it is lying |
| `test_report_is_sorted_by_tier_then_name` | Sorting by expectancy *is* selection |

### Edge Cases Checklist

- [ ] Frame shorter than `2*PIVOT_SPAN+1` → `[]` from every detector, no exception
- [ ] All-NaN RSI warmup → no oscillator pivots, no events, no `fillna`
- [ ] Zero-range bars → no division by zero in any width test
- [ ] Flat series → no pivots at all (`tests/test_signals.py:59` precedent) → no pivot-based events
- [ ] Outside bar giving both a pivot high and low at one index → `strictly_ordered` rejects the
      degenerate window (`patterns.py:181-188`)
- [ ] Level exactly equal to a close (tie) → rejected, per `pivots.py:5-6`
- [ ] Volume history shorter than `VOLUME_LOOKBACK` → NaN ratio → detector emits nothing
- [ ] Two events with identical `(kind, direction)` → `dedupe_events` collapses
- [ ] `--detector` naming an unregistered key → clean error, exit 1, not a `KeyError` traceback
- [ ] Span shorter than `config.REGIME_MIN_BARS` (207 bars, `config.py:37-39`) → zero events, reported
      as zero, not as an error
- [ ] Concurrent access: none — detectors are pure; the only I/O is the DataSource's (serialized by
      `storage._db_lock`) and the ledger's (`statestore._db_lock`). No network; no
      `@pytest.mark.network` test added

---

## Validation Commands

```bash
# Static analysis. No linter and no type checker are configured (KNOWN-LIMITATIONS §8, §12.2),
# so this means byte-compilation only.  EXPECT: no output, exit 0.
.venv/bin/python -m py_compile \
  src/trading_bot/indicators/rsi.py \
  src/trading_bot/plugins/detectors/{_geometry,reversal,continuation,oscillator,wyckoff_events,catalog}.py \
  src/trading_bot/cli.py src/trading_bot/config.py

# EXPECT: all pass.
.venv/bin/python -m pytest \
  tests/test_detectors_{reversal,continuation,oscillator,wyckoff_events,catalog,report}.py -q

# EXPECT: the 286 pre-existing tests still pass (baseline verified 2026-07-27 via
# --collect-only -q -> "286 tests collected"), plus this phase's new tests. A phase that
# breaks any of the 286 is not done (§8, §12.1). Report the new total.
.venv/bin/python -m pytest -q

# EXPECT: >=16 Phase 8 keys (plus Phase 3's and 4's), every one with a non-empty rationale (§12.3).
.venv/bin/python -m trading_bot.cli plugins | grep -c '^detector\.'
.venv/bin/python -m trading_bot.cli plugins | grep '^detector\.' | grep -c 'rationale'

# EXPECT: "catalog: 144 rows / 18 families  covered 19  deferred 6  out-of-scope 119";
#         tier 1 2/3; tier 2 6/6.
.venv/bin/python -m trading_bot.cli detector-report --coverage

# EXPECT: exit 0; banner present; one row per detector; each either MEASURED with a cost ratio c,
#         or INSUFFICIENT(<20) with rates suppressed; "n_trials += 16".
.venv/bin/python -m trading_bot.cli detector-report --start 2023-07-27 --end 2025-07-26 \
  --out .claude/PRPs/reports/phase8-detector-edge-report.md

# EXPECT: refusal naming the boundary date, exit 2. The guard, proved rather than promised.
.venv/bin/python -m trading_bot.cli detector-report --start 2023-07-27 --end 2026-07-26; echo $?

# EXPECT exactly two lines: src/trading_bot/plugins/detectors/continuation.py and
# tests/test_detectors_continuation.py. Anything more is a FRAMEWORK-GAP for Phase 3,
# recorded -- never worked around.
git log --oneline -1 && git diff --name-only HEAD~1 HEAD   # the rising-wedge commit
```

### Manual Validation

- [ ] Event counts are *plausible*: tens-to-low-hundreds per detector per symbol over two years of 4h
      bars. **Zero** events for a tier-2 detector is a geometry bug, not an absent edge — check the
      tolerances against the fixture that passes.
- [ ] Cup & handle and the Wyckoff events are the likeliest `INSUFFICIENT` rows. That is a sample
      finding, reported as such, **not** a licence to loosen tolerances.
- [ ] No detector's defaults were changed after reading the table. If any were, that is a consumed
      degree of freedom and must appear in the report's DoF section.
- [ ] `git status` shows no modification to `framework/`, `backtest/`, `signals/`, `indicators/` (other
      than the new `rsi.py`), or `data/`.
- [ ] Eyeball one detector: `detector-report --detector detector.falling-wedge` on BTCUSDT, take one
      event's `start_ts`/`end_ts`, and confirm on a chart that the shape is what the name says. Geometry
      that passes fixtures can still be nonsense on real bars.

---

## Acceptance Criteria

- [ ] All 15 tasks complete; 16 detectors registered, each with a non-empty `rationale`, a `tier`, and a
      `ParamSpec` per threshold
- [ ] §9 tier 2 fully covered (6/6); tier 1 covered 2/3 with Wyckoff Acc/Dist recorded as `deferred` per
      A1, its reason in `catalog.py`
- [ ] Every detector has geometry fixture tests: one positive known by construction, one negative per
      rejection clause
- [ ] Every detector has an edge-report row: `MEASURED` (hit rate, after-costs expectancy, profit factor,
      cost ratio) or `INSUFFICIENT(<20)` with rates suppressed
- [ ] The report is labelled a diagnostic, refuses to run in the holdout, sweeps nothing, writes its own
      ledger rows, prints its selection cost
- [ ] `catalog.py` enumerates all **144** rows; the test proves it against `.claude/technical-pattern.md`
- [ ] `indicators/rsi.py` exists, is Wilder-smoothed via production `wilder_smooth`, has hand-computed
      tests
- [ ] The `rising-wedge` commit's diff touches exactly two paths
- [ ] All 286 pre-existing tests pass; `py_compile` clean; no new dependency; no linter/type checker
      invented
- [ ] `.claude/PRPs/reports/phase8-detector-edge-report.md` written, every number beside its command

## Completion Checklist

- [ ] Every ported module docstring attributes its donor with a line number (§0b)
- [ ] `dedupe_events` is the last statement of every detector
- [ ] Freshness applied, or skipped with a reason citing `patterns.py:96-98`; `level_unbroken` applied,
      or subsumed with a stated reason
- [ ] `confirmation_ts` used for `end_ts` on every new pivot-terminated detector; the H&S parity
      exception documented in the report
- [ ] No detector reads bars except through `EvalContext`; none reads `config` for a declared param;
      `rising-wedge` reads no `config` attribute at all
- [ ] No hardcoded timeframe in any test; no test asserts an absolute `len(REGISTRY)`
- [ ] Degrees of freedom recorded (§12.6); self-contained — no codebase search needed to implement

## Risks

| Risk | L | I | Mitigation |
|---|---|---|---|
| **The edge report becomes a fitness oracle** — someone sorts by expectancy, drops the bottom half, never charges the trials. The gate becomes theater, exactly as the PRD's top risk row predicts | **H** | **Critical** | Six independent mechanisms in Task 11: fixed defaults with no `--param`, no expectancy sort, `INSUFFICIENT` suppression, the holdout guard (exit 2), per-run ledger rows, the printed `SELECTION COST IF USED`. Four pinned by tests in Task 15 |
| Relaxing `patterns.py:327` without exclusivity classification makes every wedge a "triangle" and doubles the event count with mislabelled directions | M | **High** (silently wrong signals) | Task 3 gotcha (a): the guard's work must be *relocated*. `test_shape_classification_is_mutually_exclusive` and `test_legacy_triangle_detector_cannot_see_this_shape` both mandatory |
| **Pivot-confirmation lookahead**, especially in divergence where two pivots confirm on different bars. A `span`-bar lookahead *improves* every backtest, so it will not look like a bug | M | **Critical** | `confirmation_ts` in `_geometry.py`; the `max(j, oj)` rule in Task 9; two named regression tests; the `pivot_high` vs `find_pivots` clash documented as Task 5 gotcha (a) |
| Wyckoff Spring gets read as "detects accumulation", and a strategy is built on a claim the detector never made | M | High | A2; the module docstring is a deliverable; `test_docstring_states_the_non_claim` pins the words; the ledger records the structure as `deferred` |
| Per-detector samples too small to say anything — 16 detectors × 3 symbols over 2 years of 4h bars, most producing tens of trades | **H** | M | Expected, not preventable: KNOWN-LIMITATIONS §1's arithmetic one level down. `INSUFFICIENT(<20)` makes it visible instead of printing noise. Phase 2's broadened symbol set is the real remedy and lands independently |
| Cup & handle's roundness parameters get retuned after seeing the table, converting a geometry choice into a fitted one | M | High | Task 8 gotcha (c) and Task 15 gotcha; both defaults recorded as pre-measurement choices; an explicit "no defaults changed" manual-validation item |
| Phase 3's `Detector` contract cannot express something a detector needs (multi-timeframe access; variable-length `meta`) | M | High | The `FRAMEWORK-GAP-<n>` procedure, in Dependencies and Task 15: report to Phase 3, never work around. Working around retires the PRD's headline metric silently |
| Phase 4's `policies/measured_move.py` absent or differently shaped, so the diagnostic graph cannot be built | M | M | Listed by name in Dependencies. Failure mode is an `ImportError` at first test run, not a silent wrong number. Do **not** stub a local policy — the forbidden second execution path |
| `tests/conftest.py` collides with a parallel Phase 7 | L | M | The deconfliction note: created with exactly one fixture; later phases append, never rewrite |
| The ledger drifts from `.claude/technical-pattern.md` when someone edits the doc | L | M | `test_catalog_matches_the_source_document` fails loudly on the next run. That is the design |
| 13.2% catalog coverage reads as failure to someone who skipped the PRD | L | L | Stated up front in the roster with the PRD's own framing: the full catalog is "a direction, not a v1 gate," and this phase closes tier 2 completely |

## Notes

**What this phase can and cannot prove.** It can prove sixteen geometries are detected correctly —
fixtures make that a matter of construction, not opinion. It can measure each one's after-costs
expectancy on the tuning span. It **cannot** prove any of them has edge: §4 reserves that verdict for the
gate, and KNOWN-LIMITATIONS §1's arithmetic means most per-detector samples will be too small to support
one. The right reading of a finished Phase 8 is "the engine can now express these sixteen things, and
here is what they did on two years of three symbols" — not "here are the profitable patterns."

**Why the edge report is worth building anyway.** KNOWN-LIMITATIONS §0c is the sharpest sentence in the
project's own accounting: the search "never explored the entry or the feature set." The report is the
instrument that makes the entry side inspectable for the first time. Its value is diagnostic — a detector
with cost ratio `c > 0.10` is structurally unable to pay for itself and can be reasoned about before
Phase 9 spends a generation on it. That reasoning is legitimate; doing it without charging the trials is
not, which is why the banner prints the number.

**On the deferral.** A1 is the decision most likely to be overruled, and it should be easy to overrule:
delete two entries' `deferred` status, add `plugins/detectors/wyckoff_structure.py`, and
`test_tier_scorecard` will tell you to update A1. What the plan will not do is *pretend*. A Wyckoff
detector built to a made-up numeric definition, validated against bars drawn to satisfy it, would report
a precision number that means nothing — and would enter Phase 9's campaign wearing a ⭐⭐⭐⭐⭐ label it
had not earned. The deferral costs one tier-1 row on the scorecard; the alternative costs the credibility
of the scorecard.

**Free riders.** `descending-triangle`, `symmetrical-triangle` and `inverse-cup-and-handle` are not in
§9's tier lists. They exist because the shared line fit and the cup geometry make them nearly free; the
ledger records them with `tier=None` and that reason. No success criterion depends on them, and if the
report shows they add only noise, dropping them costs three `@register` blocks.

**One number worth restating.** The catalog has **144 pattern rows across 18 families** — measured by
parsing the document, not estimated from the PRD's "~150". After this phase, 19 are covered, 6 are
deferred with reasons, and 119 are out of scope with reasons. Every one of the 144 is enumerated in code
and checked by a test against the document that defines it. That is the entire content of "tracked, not
assumed."
