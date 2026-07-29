# Plan: Plug-in Framework Core (v0.3.0 PRD Phase 3)

## Summary
Author the seven plug-in contracts, the registry, the strategy-graph serialization format, the
no-lookahead `EvalContext`, and `run_graph_backtest` — the one seam that turns a serialized
strategy graph into `engine.Trade` objects. Migrate `donchian`, `bollinger`/`meanrev`, and
`patterns`/`pivots` behind those contracts as the first four plug-ins, and prove the whole thing
honest by reproducing `engine.run_backtest`'s trade list **exactly** from a serialized graph
expressing the v0.2.0 Donchian strategy. Nothing in `signals/`, `regime/`, `indicators/`,
`data/`, or `backtest/engine.py` is edited; `walkforward.py` gains exactly one keyword.

## User Story
As the bot's sole operator, I want a new detector, confirmation, or rule to be a new registered
module rather than an engine edit, so that the next strategy iteration costs a plug-in instead of
a rebuild — and so that the framework wrapping v0.2.0's honest core provably does not change what
that core measures.

## Problem → Solution
**Current**: adding a signal method means editing `signals/scan.py`'s dispatch table (`scan.py:55-61`)
*and* `backtest/engine.py`'s `candidates_for` (`engine.py:335-349`) *and* its entry site
(`engine.py:496-541`) *and* its exit loop (`engine.py:387-453`) — four coupled edits in two files,
each of which carries a documented no-lookahead guarantee. v0.2.0 and v0.1.0 both paid full rebuild
cost, and the brute-force explorer that *could* express RSI/MACD lived outside the tested package
with its own cost model and its own scoring function.
**Solution**: a `StrategyGraph` — a serializable, content-hashed description of
`DataSource → RegimeGate → Branch(Detector → Confirmations → PositionPolicy → ExitPolicy) → Filters` —
executed by one function, `framework.execute.run_graph_backtest`, which reproduces
`engine.run_backtest` bar for bar and returns the same `Trade` dataclass, so metrics, equity,
walk-forward, the benchmark, reviewers, and the UI all keep working unchanged.

## Metadata
- **Complexity**: **XL** — 24 files (20 new, 4 modified), ~2,600 net new lines including ~900 lines of
  tests. Two new packages. One new subsystem (the graph executor) that must be bit-identical to an
  existing 546-line one. No new dependencies.
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md`
- **PRD Phase**: Phase 3 — Plug-in framework core. Depends on Phase 1 (validation integrity).
  **Gates Phase 4**, and transitively 5/6/7/8/9.
- **Binding contract**: `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md` §2 (layout),
  §3 (the seven Protocols + registry + naming), §5 (the graph→Trade seam + parity), §7 (shared
  files), §8 (test files), §12 (definition of done). **Where the PRD and the contract disagree,
  the contract wins** — the two divergences are recorded in Stated Assumptions A1 and A8.
- **Test baseline**: **286 tests collected** (`.venv/bin/python -m pytest --collect-only -q`,
  measured 2026-07-27). All 286 must stay green. This is a design constraint, not a formality:
  it is why `run_backtest` is not refactored, why `Trade` gains no fields here, and why
  `walkforward.walk_forward_pooled` gains a keyword with a `None` default rather than a new
  positional or a changed return type.

---

## UX Design

### Before
```
┌──────────────────────────────────────────────────────────────────┐
│ Add a detector to v0.2.0:                                        │
│                                                                  │
│   edit signals/scan.py       (dispatch table, line 55)           │
│   edit backtest/engine.py    (candidates_for, line 342)          │
│   edit backtest/engine.py    (entry site,     line 497)          │
│   edit backtest/engine.py    (exit loop,      line 398)          │
│   edit backtest/engine.py    (cache key,      line 328)          │
│                                                                  │
│ Every one of those lines carries a no-lookahead guarantee         │
│ documented at engine.py:16-52. Four of the five must agree.       │
└──────────────────────────────────────────────────────────────────┘
```

### After
```
┌──────────────────────────────────────────────────────────────────┐
│ Add a detector in v0.3.0:                                        │
│                                                                  │
│   create src/trading_bot/plugins/detectors/<name>.py             │
│     @register("detector", name="...", params={...},              │
│               rationale="...")                                   │
│     def detect(ctx, **params) -> list[DetectedEvent]: ...        │
│                                                                  │
│   add a Branch naming it to a .strategy.json                     │
│                                                                  │
│ Zero edits to framework/, engine.py, scan.py, or walkforward.py. │
│ The plug-in cannot see the future because EvalContext will not    │
│ hand it a bar that has not closed.                               │
└──────────────────────────────────────────────────────────────────┘
```

### Interaction Changes
| Touchpoint | Before | After | Notes |
|---|---|---|---|
| `cli.py plugins` | does not exist | table of every registered plug-in: key, tier, timeframes, params, rationale | The PRD success metric "new plug-in with zero engine-core edits" is verified here |
| `cli.py graph-validate <path>` | does not exist | validates a `.strategy.json`, prints its content hash and node count, exit 1 on any error | Phase 7's UI writes these files; this is the pre-flight check |
| `cli.py walkforward` | engine path only | `--graph <path>` routes the whole walk-forward through `run_graph_backtest` | The end-to-end manual proof of the seam |
| `cli.py backtest` / `signal` / everything else | unchanged | unchanged | The live path is deliberately NOT migrated — Stated Assumption A6 |
| Trade output | unchanged | unchanged | `Trade` gains no fields in this phase (Phase 4 appends three, contract §5) |

---

## Stated Assumptions

The shared contract fixes the names and the seam; it leaves the internal shape of the graph, the
context, and the caching to this phase. Ten decisions are made here, once, with their reasoning,
so they survive without the plan. Each is reproduced in the module docstring named beside it.

**A1 — There is no `Indicator` contract; indicators are reached only through detector plug-ins.**
*(→ `framework/contracts.py` docstring.)* The PRD's Phase 3 scope says "migrate patterns.py,
pivots.py, donchian, bollinger, wilder behind the contracts". The contract's §3 defines exactly
seven Protocols and none of them is an indicator. **Contract wins** (per the task's tie-break rule):
`indicators/{wilder,bollinger,donchian}.py` stay byte-unchanged as pure pandas functions, and
"behind the contracts" is satisfied by the fact that after this phase every one of them is
*reachable from a strategy graph only through a registered plug-in that declares its parameters as
`ParamSpec`s* — `detector.donchian-breakout` (donchian + wilder ADX/ATR),
`detector.bollinger-fade` (bollinger), `detector.legacy-patterns` (pivots + patterns),
`policy.atr-stop-measured-move` (wilder ATR). An eighth Protocol would be a new abstraction with
one implementation shape and no consumer; it is not built.

**A2 — The regime classifier is a graph-level field (`RegimeGate`), not a node; branch activation
is a `Branch.regimes` tuple.** *(→ `framework/graph.py` docstring.)* Three reasons, in order of
weight: (i) the classifier is *the one measured-healthy layer and its thresholds are never swept*
(contract §1, `walkforward.py:26-29`) — a plug-in node with `ParamSpec` bounds is an open
invitation for Phase 6's `mutator.param-jitter` to sweep them, whereas a graph-level dataclass
with a `# NOT sweepable` banner is a structural refusal; (ii) `engine.run_backtest` computes
`classify_series` once per run over the whole regime frame and memoizes it (`engine.py:279-289`) —
a per-branch node would either recompute or need its own cache seam for no benefit; (iii) *which*
regime labels enable a branch is routing, not computation, and `Branch.regimes` reads exactly like
`scan.py:55-61`'s dispatch table, which is the drift surface we most want legible. `RegimeGate`
carries only the two thresholds `BacktestParams` carries (`adx_trend_threshold`,
`atr_extreme_percentile`) plus the tier name and an `enabled` flag — `adx_period` and
`atr_percentile_window` come from config, exactly as `engine.py` leaves them.

**A3 — The breakout trigger (`check_breakout`) is executor machinery, parameterized by a
graph-level `TriggerSpec`, not a plug-in kind.** *(→ `framework/execute.py` docstring.)*
`signals/breakout.check_breakout` is the *shared* live/backtest crossing rule and it is where two
no-lookahead guarantees live: the `ts < candidate.end_ts` skip (`breakout.py:113-114`) and the
`interval_ms` contiguity check (`breakout.py:115-122`). Making it swappable per branch would let a
mutated graph disable the freshness test or the gap check and produce a flattering backtest that
still validates. `TriggerSpec` exposes only the three numbers that are already config-tunable
(`lookback_bars`, `volume_lookback`, `volume_high_ratio`); the crossing and freshness logic itself
is not expressible in the graph at all.

**A4 — Exits are a declarative per-branch `ExitPolicySpec`, not a plug-in kind.**
*(→ `framework/graph.py` docstring.)* This is what makes the MANDATORY DEVIATION at
`engine.py:392-411` / `523-530` expressible: Donchian trades get trail + opposite-channel exits,
fade trades keep frozen stop/target/time/end behavior. Making exits a plug-in would put a mutable
stop behind a Protocol boundary — the single highest-risk lookahead surface in v0.2.0 (its ratchet
ordering at `engine.py:435-452` is the guard) — and there is exactly one exit implementation. A
frozen dataclass of flags keeps the ratchet ordering inside the executor where its comment lives,
and §"Parity by construction" below shows one generic exit block reproducing **both** legacy
branches from those flags with no special-casing.

**A5 — The content hash covers *resolved* parameters, excludes `name` and `meta`, canonicalizes
collection order — and the executor iterates in that same canonical order.**
*(→ `framework/graph.py::graph_hash` docstring.)* Phase 5's `strategy_versions` table and Phase
6's `trial_ledger` both key on this hash (contract §6), so two graphs that hash equal must
*behave* identically and two graphs that behave identically must hash equal. Hence: params are
merged with registry defaults before hashing (so "omit `entry_period`" and "set `entry_period=20`"
hash the same); `name`/`meta` are excluded (a rename is not a new strategy); `branches`,
`confirmations`, and `filters` are sorted by node/branch `id` and `regimes` is sorted; **and
`StrategyGraph.ordered_branches()` — sorted by id — is the only iteration order the executor
uses**, which closes the loop. Without that last clause two hash-equal graphs could break a
`rank_signals` tie differently. `schema_version` **is** hashed (a schema change changes meaning).

**A6 — `signals/scan.py` is NOT migrated to graph dispatch in this phase.** *(→ recorded in the
phase report and in `framework/execute.py`'s docstring.)* Reasons: the live path returns
`list[Signal]` for alerting and has no `EvalSession`, no `start_ms`/`end_ms`, and no `Trade`; this
phase's acceptance gate is *backtest* parity, and rewiring live dispatch before a single graph has
passed the gate would put an unvalidated code path in front of the only thing the operator acts
on. **The drift risk is real and is named**: there are now two dispatch tables — `scan.py:55-61`
(`regime → scanner`) and `Branch.regimes` — and they can silently disagree. Three mitigations ship
in this phase: (i) `config.FADE_ENABLED` is read **at call time inside the fade detector plug-in**
(A7), so the kill switch cannot desync; (ii) `test_framework_parity.py::TestScanDriftGuard` asserts
that for every label in `classifier.REGIMES` the v0.2.0 graph's active branch corresponds to what
`scan.scan_symbol` dispatches (monkeypatched sentinels); (iii) full-history parity is itself a
drift detector, since both tables feed the same detectors. Migration is **recommended for Phase 4**,
which already owns `graph-backtest` and the thin slice, and must be listed as an open item in this
phase's report.

**A7 — `config.FADE_ENABLED` is read at call time inside `detector.bollinger-fade`, and appears in
the candidate cache key.** *(→ `plugins/detectors/bollinger_fade.py` docstring.)* This is the
faithful port of `engine.py:343` (`elif reg == "ranging" and config.FADE_ENABLED`) and
`scan.py:58`, and of the cache-key comment at `engine.py:322-327` ("FADE_ENABLED — which is read at
call time, so a live flip must not be served a cached pre-flip candidate list"). Putting the check
in the detector rather than in the executor means the flag keeps working identically no matter
which executor runs, and the graph can carry the fade branch unconditionally.

**A8 — The R:R floor stays inside the migrated policies for parity; `filter.*` plug-ins arrive in
Phase 4.** *(→ `plugins/policies/legacy_signal.py` docstring.)* In v0.2.0 the floor is applied
*inside* `setup.build_signal` (`setup.py:154-156`, gross ratio) and inside
`meanrev.build_fade_signal` (`meanrev.py:233-234`, `net_rr`), and the two use *different*
definitions on purpose. Extracting them into a `Filter` node would change which setups survive and
break parity. So the v0.2.0 parity graph has **zero `Filter` nodes and zero `Confirmation` nodes**
— which is the honest description of v0.2.0 (KNOWN-LIMITATIONS §0c: "volume is computed on every
signal but gates nothing"). The `Filter` and `Confirmation` contracts are authored and tested here
with in-test stub plug-ins; their first production instances are Phase 4's
`filters/rr_after_costs.py` (`RR_TARGET_MIN = 2.0`) and `confirmations/{volume_breakout,macd}.py`.
This is the second PRD/contract divergence: the PRD's Phase 3 row implies the pipeline is composed
here, contract §2 assigns those files to Phase 4. **Contract wins.**

**A9 — The executor's internal currency is `signals.setup.Signal`, and the multi-candidate
tie-break calls `signals.setup.rank_signals` unchanged.** *(→ `framework/execute.py` docstring.)*
A `PositionPolicy` returns a `PositionPlan` per contract §3; the executor adapts it to a `Signal`
via `contracts.plan_to_signal` *before* ranking, so ties break by the identical
`(-rr, pattern, direction)` key the engine uses (`setup.py:227`) and `Trade` construction is
field-for-field the same code shape as `engine.close_out`. `PositionPlan.source` carries the
detector event's `kind`, so `Trade.pattern` — and therefore `metrics.by_bucket`'s `"regime/pattern"`
keys (`metrics.py:31`) — are unchanged.

**A10 — The cost arithmetic in `execute._close_out` is a deliberate duplication of
`engine.run_backtest.close_out` (`engine.py:354-378`), pinned by a test, not a refactor.**
*(→ `framework/execute.py` docstring, verbatim.)* Contract §1 says do not rewrite, wrap, or fork
`engine.py`; extracting a shared helper *is* an edit to it, and the 286-test baseline plus
`_CACHE` semantics make even a pure extraction a nonzero risk for zero benefit at this stage.
The duplication is stated in the docstring ("when one changes, both must"), pinned by
`test_framework_parity.py::TestCostArithmetic`, and covered end-to-end by the full-history parity
test. Recorded as consumed technical debt in the phase report.

**A11 — `_close_out` is the designated extension point for Phase 4's three appended `Trade`
fields, and it is shaped for that up front.** *(→ `framework/execute.py::_close_out` docstring.)*
Contract §5 has Phase 4 append `planned_rr: float = 0.0`, `confirmations: tuple[str, ...] = ()`,
and `strategy_version: str = ""` to `Trade` and populate them **on the graph path only** —
`engine.run_backtest` never will. Two consequences this phase must design for, not discover:

1. **Phase 4 edits `framework/execute.py`, which Phase 3 owns.** That edit is authorized by
   contract §5 and must not require restructuring. So `_close_out` takes the whole open-trade
   record (which already carries the winning `Branch`, the `PositionPlan`, and the
   `BreakoutEvent`) rather than loose scalars, and constructs `Trade(...)` with **keyword
   arguments only**, in field order, with the three future fields named in a trailing comment.
   Phase 4's diff is then literally three added keywords — `planned_rr=plan.rr`,
   `confirmations=tuple(names)`, `strategy_version=graph.meta.get("version", "")` — sourced from
   values `_close_out` already holds. Task 7 spells out the record's fields for exactly this
   reason.
2. **The parity assertion must therefore never be `Trade == Trade`.** Once Phase 4 populates
   those fields, dataclass equality between a graph trade and an engine trade is false *by
   design*, and a parity test written as `==` would go green in Phase 3 and red in Phase 4 —
   reading as "the graph executor regressed" when nothing regressed at all. The parity assertion
   is defined over an explicit field list; see "The parity assertion, defined once" below. Do not
   "simplify" it back to equality.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md` | §2, §3, §5, §7, §8, §12 | BINDING. §3 is the literal specification of `contracts.py` and `registry.py`; §5 is `execute.py`'s signature and the parity requirement; §7 is the config/cli/walkforward deconfliction; §8 the test filenames |
| P0 | `src/trading_bot/backtest/engine.py` | **all 546** | The module `run_graph_backtest` must reproduce. Read the no-lookahead block (16-52) and the Simulation-rules block (30-61) as specification text, not commentary |
| P0 | `src/trading_bot/backtest/engine.py` | 135-175 | `_CACHE` and `_fingerprint`. The memo exists because ~870 `run_backtest` calls per pooled walk-forward made recomputation dominate runtime. `_fingerprint` is **imported**, never re-derived |
| P0 | `src/trading_bot/backtest/engine.py` | 185-222 | `_assert_interval`. Every new data path must keep this, or a partial tier migration fails silently in the flattering direction. **Imported, not reimplemented** — same error text |
| P0 | `src/trading_bot/backtest/engine.py` | 458-481 | The MEDIUM-2 repair: setup-bar selection keys on the trigger bar's **OPEN** (`ts_trig[j]`, line 481), not its close. The comment records that fixing it *reduced* in-sample Sharpe 0.431→0.255 and was kept anyway. **Do not silently re-break it** |
| P0 | `src/trading_bot/backtest/engine.py` | 387-453 | The exit loop with the MANDATORY DEVIATION (392-411): fade trades keep frozen behavior; only Donchian trades get trail + channel. The ratchet-after-exits ordering (435-452) is the intra-bar lookahead guard |
| P0 | `src/trading_bot/backtest/engine.py` | 496-541 | Entry site: `check_breakout(..., lookback_bars=1, interval_ms=…)`, `atr_value` from `atr_setup_vals[h_idx]`, `rank_signals(bar_signals)[0]`, and the second MANDATORY DEVIATION note (523-530) |
| P0 | `scripts/bruteforce/registry.py` | all 179 | The registry's direct ancestor. Carry forward: mandatory non-empty `rationale` (140-141, and the docstring at 70-72), duplicate-name rejection naming the claiming module (129-133), fatal import errors in `load_all()` (162-167), trial counting as a first-class concept (28-34, 156-159) |
| P0 | `src/trading_bot/signals/setup.py` | 40-171, 218-227 | `Signal` (the executor's internal currency), `build_signal` (the policy being wrapped — note the `not (atr_value > 0)` NaN guard at 134 and the gross-`rr` floor at 154), `rank_signals` (the tie-break) |
| P0 | `src/trading_bot/signals/breakout.py` | 38-149 | `BreakoutEvent` and `check_breakout`. The trigger contract: reads only `breakout_level`, `direction`, `end_ts`; the freshness pair (124-129); the `end_ts` skip (113-114); the contiguity check (115-122); volume grading (131-138) |
| P0 | `src/trading_bot/signals/donchian.py` | 65-211 | `detect_donchian_setups` (the first detector to migrate — note `end_ts = index[i] + interval`, i.e. the setup bar's **close**), `channel_exit_levels`, and `scan_donchian_signals` as the live-path shape |
| P0 | `src/trading_bot/signals/meanrev.py` | 42-173, 176-249 | `FadeCandidate`, `detect_fade_setups`, `_to_trigger_candidate` (159-173 — the adapter idiom `DetectedEvent` generalizes), `build_fade_signal` (note `net_rr`, not gross) |
| P1 | `src/trading_bot/signals/patterns.py` | 50-120 | `PATTERN_KINDS`, `PatternCandidate` (58-77 — `DetectedEvent` is deliberately field-compatible with it), `detect_patterns`'s signature and dedupe contract |
| P1 | `src/trading_bot/signals/pivots.py` | 1-60 | `Pivot`, `find_pivots`, and the docstring's no-lookahead argument (7-13): a pivot at bar i is knowable only once `span` bars have closed after it, and `find_pivots` on a truncated frame already guarantees it. `EvalContext` inherits this by truncation |
| P1 | `src/trading_bot/regime/classifier.py` | 33, 75-148 | `REGIMES`, `classify_series` and its two threshold kwargs — the exact call `RegimeGate` must reproduce |
| P1 | `src/trading_bot/data/storage.py` | 22-27, 35-73, 205-244 | `TIMEFRAME_MS`, `connect` (WAL + `check_same_thread=False` + `_db_lock`), `load_candles` (**both bounds inclusive**). The `data.ohlcv` plug-in is a thin wrapper over 205-244 |
| P1 | `src/trading_bot/backtest/walkforward.py` | 165-177, 257-269, 386-394 | `_pooled_expectancy`, the `walk_forward_pooled` signature, the OOS loop. The three places `strategy=` threads through — and the only three lines of this file Phase 3 touches |
| P1 | `src/trading_bot/indicators/donchian.py` | 26-47 | `donchian(df, *, period)`. The `.shift(1)` is the whole correctness argument; note the first defined value is at positional index `period`, one later than Bollinger's convention |
| P1 | `src/trading_bot/indicators/bollinger.py` | 19-52 | The single-indicator-module exemplar: keyword-only `| None = None` params with config fallback, `pd.DataFrame` return indexed like the input |
| P1 | `tests/test_backtest.py` | 23-47, 83-134, 185-190, 854-880 | Tier constants derived from config (23-33), the autouse `engine.clear_caches()` fixture (36-47), `seed`/`donchian_rows`/`donchian_atr_at_entry`/`seed_scenario` (83-134), `patch_trending` (185-190), and `TestIndicatorMemo` (854+) — the fingerprint-collision tests your cache seam must satisfy too |
| P1 | `tests/test_tiers.py` | 1-50 | The "configuration invariant" test genre: assertions about config and about the engine's tier guard, never about strategy outcome. `test_framework_graph.py` follows it |
| P2 | `src/trading_bot/cli.py` | 25-36, 129-146, 199-213, 237-266 | `subparsers.add_parser` + `_<name>_command(...) -> int` pattern, the `walkforward` parser to extend, the dispatch block, and `_gap_report_command`'s table-printing style |
| P2 | `src/trading_bot/config.py` | 105-152, 153-183 | The phase-banner convention: a comment naming the phase, why the value exists, and whether it is frozen or sweepable |
| P2 | `.claude/PRPs/reports/KNOWN-LIMITATIONS.md` | "What IS established", §6 | The four things that hold up (cost model, enforced no-lookahead, one code path, grid self-selection) are exactly what parity must preserve. §6 is MEDIUM-2 |
| P2 | `scripts/bruteforce/core.py` | `assert_causal`, `Ctx` | Prior art for the causality assertion and multi-timeframe alignment. Port the *idea* (`assert_trailing_only`), not the code — `Ctx` has no `_assert_interval` and its own cost path |

## External Documentation

No external research needed — this phase uses only established internal patterns plus two stdlib
facilities.

```
KEY_INSIGHT: typing.Protocol with @runtime_checkable supports isinstance() but checks only
             member PRESENCE, never signatures or types.
APPLIES_TO:  framework/contracts.py — the seven Protocols.
GOTCHA:      isinstance(obj, Detector) passes for anything with a callable `detect`. The six
             function-shaped contracts are therefore checked at REGISTER time by
             contracts.check_callable_shape() (inspect.signature arity + first-parameter name),
             not by isinstance. Only DataSource, which is an object, is isinstance-checked.
```

```
KEY_INSIGHT: `ParamSpec` is also a name in typing (PEP 612, Python 3.10+).
APPLIES_TO:  framework/contracts.py, and every module importing it.
GOTCHA:      The name is fixed by contract §3 and must not be renamed. Never write
             `from typing import ParamSpec` or `typing.ParamSpec` anywhere in trading_bot.
             contracts.py imports named symbols from typing only (Protocol, runtime_checkable,
             TYPE_CHECKING, TypeAlias, Any) — never `import typing` with attribute access.
```

```
KEY_INSIGHT: `pkgutil.walk_packages` (not iter_modules) is required to reach plugins/<sub>/<mod>.py.
APPLIES_TO:  framework/registry.py::load_all().
GOTCHA:      walk_packages IMPORTS each package to read its __path__, so a broken subpackage
             __init__.py raises there rather than at the leaf. That is the intended fatal
             behavior (registry.py:162-167) but the error message must name the module — wrap
             in RegistryError ... from exc.
```

```
KEY_INSIGHT: the venv installs trading_bot via a plain path .pth, not a setuptools import hook.
APPLIES_TO:  the two new subpackages trading_bot.framework and trading_bot.plugins.
MEASURED:    .venv/lib/python3.11/site-packages/__editable__.trading_bot-0.1.0.pth contains
             exactly `/Users/.../trading/src`. New subpackages are importable with NO reinstall.
             pyproject's [tool.setuptools.packages.find] where=["src"] picks them up on any
             future build. Do not `pip install -e .` as part of this phase.
```

---

## Dependencies on Other Phases

| Dependency | Artifact | What breaks without it |
|---|---|---|
| **Phase 1** (must be complete) | `_evaluate_gate` returns `dict[str, bool]`; `WalkForwardResult` gains `gate`/`benchmark`/`n_trials_used`; `passed` kept as `all(gate.values())` | Nothing in Phase 3 *reads* the gate. Phase 3 touches `walkforward.py` at three disjoint sites (the signature, `_pooled_expectancy`, the OOS loop); Phase 1 touches `_evaluate_gate`, `WalkForwardResult`, and the result construction. **Merge boundary: no shared line.** If Phase 1 has not landed, everything here still works — but do not start, because contract §11's ordering exists so the measuring stick is honest before anything optimizes against it |
| **Phase 1** | `data/statestore.py`, `backtest/trials.py` | Not consumed here. `run_graph_backtest` takes no ledger handle — the ledger is charged by the *oracle* (Phase 6's `evolution/oracle.py`), one level up, per contract §4.2. Do **not** add ledger writes to the executor; a backtest is not an evaluation |
| Phase 4 (downstream) | `plugins/{confirmations,filters,policies/measured_move}.py`, `indicators/macd.py`, `Trade`'s three appended fields | Phase 4 must create `plugins/confirmations/__init__.py` and `plugins/filters/__init__.py` itself. **Phase 3 creates `plugins/policies/__init__.py`** (for `legacy_signal.py`); Phase 4 must not re-create it |
| Phase 5 (downstream) | `graph_hash()` keys `strategy_versions`; `ReviewRecord`/`ReviewContext` are `Any` aliases in `contracts.py` | Typed as `Any` deliberately so Phase 5 never needs to edit `framework/`, which is Phase 3's exclusively (contract §2) |
| Phase 6 (downstream) | `Mutator` protocol; `ParamSpec.bounds/choices/clamp`; `ordered_branches()`; `StrategyGraph` immutability; `graph_hash` |  Phase 6 mutates graphs with `dataclasses.replace`, which requires every graph type to be a frozen dataclass holding **tuples, not lists**. Enforced in `__post_init__` |
| Phase 7 (downstream) | `ParamSpec.kind/bounds/choices/step/doc` render a UI control; `registry.by_kind()` populates the palette; `to_dict`/`from_dict` is the wire format | Get `ParamSpec` right once, here |

---

## Patterns to Mirror

Every snippet below is copied verbatim from the working tree at the cited `file:line`.

### CONTENT_FINGERPRINT_MEMO  — the caching seam to extend, not reinvent
```python
# SOURCE: backtest/engine.py:135-146 (comment) and 158-174 (the function)
_CACHE: dict = {}

def _fingerprint(df: pd.DataFrame) -> tuple:
    """Content fingerprint of a loaded OHLCV frame.

    Bar count and end timestamps alone are not enough: distinct fixtures
    routinely share them while holding different prices, which would collide.
    The column sums make the key sensitive to the actual values at negligible
    cost next to the indicators being cached.
    """
    ts = df.index.to_numpy()
    return (
        len(ts),
        int(ts[0]),
        int(ts[-1]),
        round(float(df["high"].sum()), 6),
        round(float(df["low"].sum()), 6),
        round(float(df["close"].sum()), 6),
    )
```

### TRAILING_LOOKUP  — the one no-lookahead index rule, used three times
```python
# SOURCE: backtest/engine.py:315-317 — regime label at time t is the last
# regime bar CLOSED by t. close_regime = df_regime.index.to_numpy() + regime_ms.
def regime_at(t: int) -> str:
    k = int(np.searchsorted(close_regime, t, side="right")) - 1
    return str(labels.iloc[k]) if k >= 0 else "uncertain"

# SOURCE: backtest/engine.py:416 — same rule for the setup-tier exit channel.
s_idx = int(np.searchsorted(close_setup, int(close_trig[j]), side="right")) - 1

# SOURCE: backtest/engine.py:481 — and for setup-bar selection at entry, keyed
# on the trigger bar's OPEN (ts_trig[j]), NOT its close. This is MEDIUM-2.
h_idx = int(np.searchsorted(close_setup, int(ts_trig[j]), side="right")) - 1
```

### TIER_GUARD  — imported, never reimplemented
```python
# SOURCE: backtest/engine.py:185-222 (abridged; read all 38 lines)
def _assert_interval(df: pd.DataFrame, timeframe: str, symbol: str, role: str) -> int:
    """... If config and stored data disagree — a partial tier migration, a
    series backfilled under the wrong key, a fixture seeded at the old
    interval — all of them fail SILENTLY, in the flattering direction. Raise
    instead. Uses the MEDIAN inter-bar spacing ..."""
    expected = storage.TIMEFRAME_MS[timeframe]
    ts = df.index.to_numpy()
    if len(ts) >= 2:
        observed = int(np.median(np.diff(ts)))
        if observed != expected:
            raise ValueError(
                f"{symbol} {role} series spacing is {observed} ms but config "
                f"names timeframe {timeframe!r} ({expected} ms). ..."
            )
    return expected
```

### PER_SETUP_BAR_CANDIDATE_CACHE  — the structure `execute.py` generalizes per branch
```python
# SOURCE: backtest/engine.py:328-349. Note WHAT is in the key: the frame
# fingerprint, every param that can change a candidate, and FADE_ENABLED.
cand_key = (
    "cands", symbol, config.SIGNAL_PATTERN_TIMEFRAME, fp_setup,
    params.adx_trend_threshold, params.atr_extreme_percentile,
    params.bb_num_std, bool(config.FADE_ENABLED),
)
cand_cache: dict[int, tuple[str, list]] = _CACHE.setdefault(cand_key, {})

def candidates_for(h_idx: int) -> tuple[str, list]:
    if h_idx not in cand_cache:
        t = int(close_setup[h_idx])
        reg = regime_at(t)
        window = df_setup.iloc[max(0, h_idx + 1 - config.PATTERN_LOOKBACK_BARS) : h_idx + 1]
        cands: list = []
        if reg == "trending":
            cands = [("donchian", c) for c in detect_donchian_setups(window)]
        elif reg == "ranging" and config.FADE_ENABLED:
            cands = [("fade", c) for c in detect_fade_setups(window, num_std=params.bb_num_std)]
        cand_cache[h_idx] = (reg, cands)
    return cand_cache[h_idx]
```

### MANDATORY_DEVIATION_EXIT_ASYMMETRY  — what `ExitPolicySpec` must express
```python
# SOURCE: backtest/engine.py:392-411 — fade trades keep EXACTLY today's
# stop/target/time/end behavior; only Donchian trades get trail + channel.
if not open_trade["is_donchian"]:
    if s.direction == "long":
        if lows[j] <= s.stop:
            close_out(j, s.stop, "stop")  # conservative: stop first
        elif highs[j] >= s.target:
            close_out(j, s.target, "target")
    else:
        ...
    if open_trade is not None and j - open_trade["entry_j"] >= max_hold:
        close_out(j, float(closes[j]), "time")
    continue

# SOURCE: backtest/engine.py:417-424 — the Donchian branch. Priority is
# stop/trail, then channel, then target. target only when params.target_enabled.
if s.direction == "long":
    chan = float(exit_lower[s_idx]) if s_idx >= 0 else float("nan")
    if lows[j] <= stop:
        close_out(j, stop, "trail" if open_trade["trailed"] else "stop")
    elif not np.isnan(chan) and lows[j] <= chan:
        close_out(j, chan, "channel")
    elif params.target_enabled and highs[j] >= s.target:
        close_out(j, s.target, "target")
```

### RATCHET_AFTER_EXITS  — the intra-bar lookahead guard, copied ordering and all
```python
# SOURCE: backtest/engine.py:435-452
# Ratchet AFTER this bar's exits are resolved: a stop derived from
# bar j's own extreme, tested against bar j's own low, is intra-bar
# lookahead. The trail only ever binds from bar j+1 onward.
if open_trade is not None and params.trail_enabled and open_trade["atr"] > 0:
    trail_dist = params.trail_atr_multiple * open_trade["atr"]
    if s.direction == "long":
        open_trade["extreme"] = max(open_trade["extreme"], float(highs[j]))
        new_stop = open_trade["extreme"] - trail_dist
        if new_stop > open_trade["stop"]:
            open_trade["stop"] = new_stop
            open_trade["trailed"] = True
    else:
        ...
```

### TRADE_CONSTRUCTION_AND_COSTS  — duplicated verbatim in `execute._close_out` (A10)
```python
# SOURCE: backtest/engine.py:354-378
def close_out(j: int, price: float, outcome: str) -> None:
    nonlocal open_trade
    s = open_trade["signal"]
    sign = 1.0 if s.direction == "long" else -1.0
    gross = sign * (price - s.entry) / s.entry
    hold_days = (int(ts_trig[j]) - s.ts) / 86_400_000.0
    funding_cost = funding * hold_days
    trades.append(
        Trade(
            symbol=symbol, regime=open_trade["regime"], pattern=s.pattern,
            direction=s.direction, entry_ts=s.ts, entry=s.entry, stop=s.stop,
            target=s.target, exit_ts=int(ts_trig[j]), exit_price=price,
            outcome=outcome, pnl_pct=gross - cost - funding_cost,
            volume_high=s.volume_high,
        )
    )
    open_trade = None
# and, at engine.py:263 — the round-trip cost:
cost = 2 * (fee + slip)
```

### TIE_BREAK  — called unchanged, so ties resolve identically
```python
# SOURCE: signals/setup.py:218-227
def rank_signals(signals: list[Signal]) -> list[Signal]:
    """Order signals best-first: highest R:R, then pattern kind for determinism.

    The single shared ranking rule for live scanning and backtesting. ...
    """
    return sorted(signals, key=lambda s: (-s.rr, s.pattern, s.direction))
```

### CANDIDATE_ADAPTER  — what `contracts.candidate_from_event` generalizes
```python
# SOURCE: signals/meanrev.py:159-173 — a non-geometry setup is wrapped in a
# PatternCandidate purely so check_breakout's crossing/freshness/volume logic
# can be reused unchanged.
def _to_trigger_candidate(candidate: FadeCandidate) -> PatternCandidate:
    return PatternCandidate(
        kind=FADE_KIND,
        direction=candidate.direction,
        breakout_level=candidate.trigger_level,
        target_height=abs(candidate.target - candidate.trigger_level),
        start_ts=candidate.start_ts,
        end_ts=candidate.end_ts,
    )
```

### REGISTRY_DECORATOR  — the shape and every validation to carry forward
```python
# SOURCE: scripts/bruteforce/registry.py:127-153
def deco(fn: Callable) -> Callable:
    key = name or fn.__name__
    if key in ALL:
        raise ValueError(
            f"strategy {key!r} is already registered (by "
            f"{ALL[key].build.__module__}); pick a distinct name"
        )
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if not rationale.strip():
        raise ValueError(f"{key}: a rationale is required")
    for axis, values in (grid or {}).items():
        if not values:
            raise ValueError(f"{key}: grid axis {axis!r} has no values")
    ALL[key] = Strategy(...)
    return fn

# SOURCE: scripts/bruteforce/registry.py:162-179
def load_all() -> dict[str, Strategy]:
    """Import every module in ``strategies/`` so their decorators run.

    Import errors are FATAL rather than skipped: a family silently missing from
    the leaderboard would read as "tested and found wanting".
    """
```

### CONFIG_FALLBACK_KEYWORD_ONLY  — the house style for every plug-in parameter
```python
# SOURCE: indicators/bollinger.py:19-52
def bollinger(
    df: pd.DataFrame, *, period: int | None = None, num_std: float | None = None
) -> pd.DataFrame:
    if period is None:
        period = config.BB_PERIOD
    ...
    return pd.DataFrame({...}, index=df.index)
```

### NAN_REJECTION  — the idiom every numeric guard in the framework uses
```python
# SOURCE: signals/setup.py:131-136
# NaN comparisons are always False, so `atr_value <= 0` alone would let a
# NaN ATR (Wilder warmup) silently produce a stop = nan Signal instead of
# being rejected. `not (atr_value > 0)` catches NaN, zero, and negative.
if not (atr_value > 0):
    reject("atr_value %s is undefined or non-positive", atr_value)
    return None
```

### TEST_TIER_CONSTANTS_AND_CACHE_ISOLATION
```python
# SOURCE: tests/test_backtest.py:23-33
SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded: these fixtures follow config forever, so a
# future tier shift cannot leave the tests on the old timeframes while
# production moves (the classic way a tier change passes CI while being wrong).
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_REG = storage.TIMEFRAME_MS[REGIME_TF]
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000

# SOURCE: tests/test_backtest.py:36-47
@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    """Clear the engine's indicator memo around every test. ... test isolation
    must not DEPEND on that argument being right."""
    engine.clear_caches()
    yield
    engine.clear_caches()
```

### TEST_FIXTURE_BUILDERS  — reused by `test_framework_parity.py`, copied not imported
```python
# SOURCE: tests/test_backtest.py:83-97 and 112-133
def seed(conn, timeframe, rows, start=START, interval=D_SET):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    storage.upsert_candles(conn, SYMBOL, timeframe, data)
    return data

def donchian_rows():
    """Setup-timeframe ramp: 60 bars, ADX(14) >> 25 after warmup, last bar
    closes exactly at the trailing 20-bar upper channel (160.0) and above the
    55-bar mid (131.5) ... Constant true range (3) gives an exact ATR(14) = 3.0.
    """
    return [[100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 10.0] for i in range(60)]

def seed_scenario(conn, outcome_rows_trig):
    """... Entry = 161.0, stop = compute_atr_stop(161.0, "long", 3.0,
    ATR_STOP_MULTIPLE) = 156.5, target = 160.0 + 22.0 = 182.0."""
    seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 12, interval=D_REG)
    ...
```

### CLI_SUBCOMMAND
```python
# SOURCE: cli.py:64-77 (parser) and 176-181 (dispatch) and 237-266 (handler)
gap_report_parser = subparsers.add_parser(
    "gap-report", help="Report gaps in stored OHLCV data"
)
gap_report_parser.add_argument("--as-of", type=_date_arg, help="...")
...
elif args.command == "gap-report":
    conn = connect(args.db)
    exit_code = _gap_report_command(conn, now_ms=args.as_of, start_ms=args.start)
    conn.close()
    sys.exit(exit_code)
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/config.py` | UPDATE | Append the Phase 3 block: `STRATEGY_DIR`, `FRAMEWORK_PLUGIN_PACKAGE`, `FRAMEWORK_CACHE_ENABLED`. Reserved prefixes per contract §7 |
| `src/trading_bot/framework/__init__.py` | CREATE | Re-export the public surface: contracts, registry, graph API, `EvalContext`, `run_graph_backtest`, errors |
| `src/trading_bot/framework/errors.py` | CREATE | `FrameworkError`, `ContractError`, `RegistryError`, `GraphError` |
| `src/trading_bot/framework/contracts.py` | CREATE | The seven `Protocol`s, `ParamSpec`, `DetectedEvent`, `ConfirmationVerdict`, `PositionPlan`, `FilterVerdict`, the four adapters, `check_callable_shape` |
| `src/trading_bot/framework/registry.py` | CREATE | `PluginSpec`, `KINDS`, `register`, `get`, `by_kind`, `load_all`, `temporary_registry` |
| `src/trading_bot/framework/graph.py` | CREATE | `SCHEMA_VERSION`, `NodeSpec`, `RegimeGate`, `TriggerSpec`, `ExitPolicySpec`, `Branch`, `StrategyGraph`, `validate`, `to_dict`/`from_dict`, `graph_hash`, `save`/`load` |
| `src/trading_bot/framework/context.py` | CREATE | `EvalSession`, `EvalContext`, the series/candidate cache seam, `clear_caches`, `cache_stats`, `assert_trailing_only` |
| `src/trading_bot/framework/execute.py` | CREATE | `run_graph_backtest` — the ONE graph→Trade seam (contract §5) |
| `src/trading_bot/plugins/__init__.py` | CREATE | Package docstring; `build_v020_graph()` — the parity fixture and Phase 6's seed |
| `src/trading_bot/plugins/data/__init__.py` | CREATE | Subpackage marker |
| `src/trading_bot/plugins/data/ohlcv.py` | CREATE | `data.ohlcv` — `DataSource` over `storage.load_candles` + `engine._assert_interval` |
| `src/trading_bot/plugins/detectors/__init__.py` | CREATE | Subpackage marker |
| `src/trading_bot/plugins/detectors/donchian.py` | CREATE | `detector.donchian-breakout` wrapping `signals.donchian.detect_donchian_setups` |
| `src/trading_bot/plugins/detectors/bollinger_fade.py` | CREATE | `detector.bollinger-fade` wrapping `signals.meanrev.detect_fade_setups`; reads `FADE_ENABLED` at call time (A7) |
| `src/trading_bot/plugins/detectors/legacy_patterns.py` | CREATE | `detector.legacy-patterns` wrapping `signals.patterns.detect_patterns` + `signals.pivots.find_pivots` |
| `src/trading_bot/plugins/policies/__init__.py` | CREATE | Subpackage marker. **Phase 4 must not re-create this** |
| `src/trading_bot/plugins/policies/legacy_signal.py` | CREATE | `policy.atr-stop-measured-move` (wraps `setup.build_signal`) and `policy.fade-structural-stop` (wraps `meanrev.build_fade_signal`) |
| `src/trading_bot/backtest/walkforward.py` | UPDATE | **One** addition: `strategy: StrategyGraph | None = None` (contract §5), threaded to `_pooled_expectancy` and the OOS loop, plus the unsupported-axis guard |
| `src/trading_bot/cli.py` | UPDATE | `plugins` and `graph-validate` subcommands + handlers; `walkforward --graph` |
| `data/strategies/donchian-v020.strategy.json` | CREATE | The v0.2.0 strategy serialized — the parity fixture, committed |
| `tests/test_framework_contracts.py` | CREATE | ParamSpec, adapters, `EvalContext` no-lookahead, `assert_trailing_only` |
| `tests/test_framework_registry.py` | CREATE | Duplicate/rationale/kind/name/param validation, `load_all` fatality + idempotence, the all-plug-ins-well-formed meta-test |
| `tests/test_framework_graph.py` | CREATE | Round-trip, hash canonicality + pinned digest, every validation error |
| `tests/test_framework_parity.py` | CREATE | **The acceptance gate**: full trade-list equality vs `run_backtest`, plus nine localizing sub-parity tests |

## NOT Building

- **No `Filter` or `Confirmation` production plug-ins.** Phase 4 owns
  `plugins/filters/rr_after_costs.py` and `plugins/confirmations/{volume_breakout,macd}.py`.
  Their contracts are authored and tested here with in-test stubs (A8).
- **`indicators/macd.py`, `indicators/rsi.py`** — Phase 4 and Phase 8. Do not port anything from
  `scripts/bruteforce/indicators.py` in this phase.
- **No edits to `backtest/engine.py`.** Not one line. Two private names are *imported* from it
  (`_fingerprint`, `_assert_interval`) — the same private-import convention `meanrev.py:35` and
  `engine.py:81` already use.
- **No edits to `signals/*`, `regime/*`, `indicators/*`, `risk/*`, `data/*`, `backtest/{engine,metrics,equity}.py`.**
- **No `Trade` field additions.** Contract §5 assigns `planned_rr`, `confirmations`,
  `strategy_version` to Phase 4/5. `PositionPlan.rr` carries the R:R inside the executor; it is
  simply not persisted yet, and saying so is more honest than half-adding the audit trail.
- **No gate changes.** `_evaluate_gate`, `DEFAULT_GRID`, `GATE_*` are Phase 1's. Do not run
  `walkforward` against the real holdout as a "check" — that spends the one-shot OOS.
- **No `signals/scan.py` migration** (A6) and **no live-path behavior change**. `cli.py signal`
  produces byte-identical output after this phase.
- **No `data/state.db` writes, no trial-ledger integration.** The executor takes no ledger handle.
- **No process pool / parallelism.** Phase 6 owns `evolution/runner.py`.
- **No performance tuning beyond the two documented cache seams.** Measure and report; do not
  optimize speculatively.
- **No new dependencies.** `pandas-ta` is gone from PyPI; TA-Lib needs a C library. stdlib +
  pandas + numpy only.
- **No general DAG.** The graph is a typed-slot pipeline (A2/A3/A4). A free-form node/edge DAG
  would make `validate()` a graph-theory exercise and every no-lookahead guarantee negotiable.
- **No `Indicator` protocol** (A1).
- **No UI, no evolution, no reviewers.** The `Reviewer` and `Mutator` Protocols are authored and
  shape-tested; no implementation ships.

---

## Parity by construction — the design argument, before the tasks

Parity is not a test you hope passes; it is a property the executor is *built* to have. Read this
table before writing `execute.py`. Left column is `engine.run_backtest`; right column is the graph
expression that reproduces it. If any row cannot be satisfied, stop and fix the graph model — do
not "adjust" the executor.

| `engine.run_backtest` step | line(s) | Graph expression |
|---|---|---|
| Load 3 tiers | 265-267 | `data.ohlcv` plug-in, `frame(symbol, tf)` per tier from `RegimeGate.timeframe` / `config.SIGNAL_PATTERN_TIMEFRAME` / `config.SIGNAL_TRIGGER_TIMEFRAME` |
| `_assert_interval` × 3 | 271-273 | `plugins/data/ohlcv.py` calls the **imported** `engine._assert_interval` with the same `role` strings |
| Empty-frame early return | 268-269 | identical |
| `classify_series(...)` memoized | 279-289 | `EvalSession` labels cache, key `("labels", symbol, tf, fp, adx_thr, atr_pct)` — byte-identical key shape; thresholds from `RegimeGate` |
| `close_regime/close_setup/close_trig` | 291-292, 306 | identical arithmetic in `EvalSession` |
| `atr_setup_vals` memoized | 293-296 | `EvalSession.atr(setup_tf, period)`, key gains `period` (see Task 5 GOTCHA) |
| `exit_lower/exit_upper` memoized | 300-304 | `EvalSession.channels(setup_tf, period)` from `ExitPolicySpec.channel_period` |
| `start`/`end` defaults | 308-309 | identical |
| `regime_at(t)` | 315-317 | `EvalContext.regime()`, same `searchsorted(..., "right") - 1`, same `"uncertain"` for `k < 0` |
| `candidates_for(h_idx)` per-bar memo | 328-349 | per-**branch** memo; regime gate becomes `label in branch.regimes`; window is `ctx.window(setup_tf, bars=lookback_bars)` == `df_setup.iloc[max(0,h_idx+1-180):h_idx+1]` |
| `FADE_ENABLED` at call time | 343 | read inside `detector.bollinger-fade`; `bool(config.FADE_ENABLED)` in the cache key (A7) |
| `for j in range(len(ts_trig))`, `break` on `bc > end` | 381-385 | identical |
| open-trade branch first, `j <= entry_j: continue` | 387-389 | identical |
| Exit asymmetry (fade frozen / Donchian trail+channel) | 392-432 | **one** generic block driven by `ExitPolicySpec` — see the proof below |
| time stop after level checks | 409-410, 433-434 | identical, `max_hold = exits.max_hold_bars or run-level max_hold` |
| ratchet AFTER exits | 435-452 | identical, gated on `exits.trail_enabled` |
| `bc < start or j < 1: continue` | 455-456 | identical |
| `h_idx` on trigger **OPEN** (MEDIUM-2) | 481 | identical, with the 24-line comment carried across verbatim |
| `window_trig` slice | 489 | identical: `df_trig.iloc[max(0, j - (VOLUME_LOOKBACK+1)) : j+1]`, `VOLUME_LOOKBACK` from `TriggerSpec` |
| `check_breakout(..., lookback_bars=1, interval_ms=trigger_ms)` | 498, 505-510 | identical, on `candidate_from_event(event)` |
| `atr_value = atr_setup_vals[h_idx]` | 492-494 | identical |
| `build_signal` / `build_fade_signal` | 500-515 | `policy.atr-stop-measured-move` / `policy.fade-structural-stop` → `PositionPlan` → `plan_to_signal` |
| `rank_signals(bar_signals)[0]` | 522 | identical call, on the adapted `Signal`s |
| `open_trade` dict incl. `stop`/`atr`/`extreme`/`trailed` | 531-541 | identical, `is_donchian` replaced by `exits` (the branch's `ExitPolicySpec`) |
| `close_out` + `cost = 2*(fee+slip)` + funding | 263, 354-378 | `_close_out`, duplicated verbatim (A10) |
| unresolved trade at data end → `"end"` | 543-544 | identical |

**The exit-asymmetry proof.** One generic block reproduces both legacy branches, with no
`if is_donchian`:

```python
if s.direction == "long":
    chan = float(exit_lower[s_idx]) if (ex.channel_exit and s_idx >= 0) else float("nan")
    if lows[j] <= stop:
        close_out(j, stop, "trail" if open_trade["trailed"] else "stop")
    elif not np.isnan(chan) and lows[j] <= chan:
        close_out(j, chan, "channel")
    elif ex.target_enabled and highs[j] >= s.target:
        close_out(j, s.target, "target")
else:
    ...  # mirrored
```

- Fade branch (`channel_exit=False, target_enabled=True, trail_enabled=False`): `chan` is NaN so
  the channel arm is skipped; `target_enabled` is True so the target arm runs unconditionally →
  identical to `engine.py:398-408`. `trailed` stays False forever (nothing ratchets it), so the
  outcome label is `"stop"` and never `"trail"` → identical.
- Donchian branch (`channel_exit=True, target_enabled=params.target_enabled,
  trail_enabled=params.trail_enabled`) → identical to `engine.py:417-432`.
- Time stop and ratchet sit outside the direction split at the same positions → identical.

That is why exits are declarative flags (A4) rather than a plug-in: the asymmetry is *data*, and
the code that must not be touched stays in one place.

**Mapping `BacktestParams` onto the v0.2.0 graph** (the table `build_v020_graph` implements):

| `BacktestParams` field | v0.2.0 default | Graph location |
|---|---|---|
| `adx_trend_threshold` | 25.0 | `graph.regime.adx_trend_threshold` |
| `atr_extreme_percentile` | 0.90 | `graph.regime.atr_extreme_percentile` |
| `bb_num_std` | 2.0 | branch `range` → `detector.bollinger-fade` param `num_std` |
| `rr_floor` | 1.5 | both policies' `rr_floor` param |
| `trail_enabled` | False | branch `trend` → `exits.trail_enabled` |
| `trail_atr_multiple` | 3.0 | branch `trend` → `exits.trail_atr_multiple` |
| `target_enabled` | False | branch `trend` → `exits.target_enabled` |
| (`max_hold_bars`, a run kwarg) | 96 | `run_graph_backtest(max_hold_bars=...)`; branch `exits.max_hold_bars=None` |
| (`config.ATR_STOP_MULTIPLE`) | 1.5 | `policy.atr-stop-measured-move` param `atr_multiple` |
| (`config.ATR_STOP_PERIOD`) | 14 | policy param `atr_period` **and** `exits.trail_atr_period` — validation requires them equal |
| (`config.DONCHIAN_ENTRY_PERIOD`) | 20 | detector param `entry_period` **and** `exits.channel_period` — validation requires them equal |
| (`config.FADE_ENABLED`) | False | not in the graph — read at call time (A7) |

### The parity assertion, defined once

**Never assert `graph_trades == engine_trades`.** `Trade` is a frozen dataclass, so `==` compares
*every* field — including the three Phase 4 appends per contract §5 (`planned_rr`,
`confirmations`, `strategy_version`), which Phase 4 populates on the graph path and which
`engine.run_backtest` will never populate. Dataclass equality therefore passes in Phase 3 and
fails in Phase 4, at the moment it is most misleading: it would read as "the graph executor
regressed" when in fact the graph path merely gained an audit trail the legacy path does not have.

The single shared helper, defined once at the top of `tests/test_framework_parity.py` and used by
every parity test in this phase and every later phase that needs one:

```python
# The fields that constitute a simulated round-trip. Deliberately EXPLICIT and
# deliberately NOT `Trade == Trade`:
#
# contract §5 has Phase 4 append planned_rr / confirmations / strategy_version
# to Trade and populate them on the GRAPH path only — engine.run_backtest never
# will. Dataclass equality would therefore start failing the moment Phase 4
# lands, and would read as a graph-executor regression rather than as the
# intended divergence it is. Parity is about the SIMULATION, not about the
# audit trail attached to it.
#
# Do not replace this with `==`. If a new field belongs in parity, add its name
# here explicitly, with a reason.
PARITY_EXACT = ("symbol", "regime", "pattern", "direction",
                "entry_ts", "exit_ts", "outcome", "volume_high")
PARITY_CLOSE = ("entry", "stop", "target", "exit_price", "pnl_pct")

# Bit-identity is the EXPECTED outcome: the executor performs the same
# arithmetic in the same order on the same floats. abs_tol exists so a genuine
# reassociation reports as a tiny delta instead of an opaque False, and it is
# tight enough that a real behavioural difference (a different exit level, a
# different cost term) can never hide under it. If a diff shows up at 1e-13,
# something reordered — investigate, do not widen the tolerance.
PARITY_ABS_TOL = 1e-12

def assert_trades_match(got, want, *, abs_tol=PARITY_ABS_TOL):
    assert len(got) == len(want), (
        f"trade COUNT differs: graph {len(got)} vs engine {len(want)}; "
        f"graph entries {[t.entry_ts for t in got][:5]} ... "
        f"engine entries {[t.entry_ts for t in want][:5]}"
    )
    for i, (g, e) in enumerate(zip(got, want)):
        for f in PARITY_EXACT:
            assert getattr(g, f) == getattr(e, f), f"trade {i} field {f}: {getattr(g, f)!r} != {getattr(e, f)!r}"
        for f in PARITY_CLOSE:
            gv, ev = getattr(g, f), getattr(e, f)
            assert math.isclose(gv, ev, rel_tol=0.0, abs_tol=abs_tol), \
                f"trade {i} field {f}: {gv!r} vs {ev!r} (delta {gv - ev:.3e})"
```

Two properties worth stating because they are what makes a failure debuggable: the count
assertion fires *first* and prints the first five entry timestamps from each side (a count
mismatch is almost always a single missing or extra entry, and the timestamps localize it
immediately), and the per-field messages name the trade index and the field, so an exit-logic bug
and an entry-selection bug never look the same.

The nine sub-parity tests in Task 20 exist for the same reason: a bare
`assert_trades_match(graph, engine)` failure on 150 trades is nearly uninformative, whereas
"regime labels match, candidate lists match, but the exit outcome mix differs" points at one
block.

---

## Step-by-Step Tasks

Build order is dependency order: errors → contracts → registry → context → graph → execute →
plug-ins → integration → tests. Each task's VALIDATE runs before the next begins.

### Task 1: Config block
- **ACTION**: Append a Phase 3 banner block to the end of `src/trading_bot/config.py` (after the
  cost constants ending line 183, before `date_to_ms`).
- **IMPLEMENT**:
  ```python
  # ---------------------------------------------------------------------------
  # Phase 3 (v0.3.0): plug-in framework core. Reserved prefixes STRATEGY_DIR /
  # FRAMEWORK_* per the v0.3.0 shared architecture contract §7.
  #
  # None of these is a strategy parameter and none may ever appear in a
  # walk-forward grid or a ParamSpec: they configure where graphs live and
  # whether memoization is on, not what a strategy does.
  # ---------------------------------------------------------------------------
  STRATEGY_DIR = "data/strategies"  # serialized StrategyGraphs: <name>.strategy.json
  # The package framework.registry.load_all() walks. A string (not the module
  # object) so tests can point load_all at a throwaway package.
  FRAMEWORK_PLUGIN_PACKAGE = "trading_bot.plugins"
  # Kill switch for the framework's indicator/candidate memo. Exists as a
  # MEASUREMENT tool, not a tuning knob: a graph backtest run with caching off
  # must be bit-identical to one run with it on, and that is asserted in
  # tests/test_framework_parity.py. If the two ever differ, the cache key is
  # missing a parameter and every cached result since is suspect.
  FRAMEWORK_CACHE_ENABLED = True
  ```
- **MIRROR**: `config.py:153-159` — the `# ---` banner naming the phase, why the value exists, and
  whether it is frozen.
- **IMPORTS**: none.
- **GOTCHA**: Do **not** add a `FRAMEWORK_SCHEMA_VERSION`. `graph.SCHEMA_VERSION` is the single
  source of truth; a config mirror is a guaranteed drift. Do not modify any existing constant —
  `RR_FLOOR = 1.5` stays the legacy floor, and Phase 4 adds `RR_TARGET_MIN = 2.0` separately
  (contract §7).
- **VALIDATE**: `.venv/bin/python -c "from trading_bot import config; print(config.STRATEGY_DIR, config.FRAMEWORK_PLUGIN_PACKAGE, config.FRAMEWORK_CACHE_ENABLED)"`
  → `data/strategies trading_bot.plugins True`.

### Task 2: `framework/errors.py`
- **ACTION**: Create `src/trading_bot/framework/__init__.py` (empty placeholder for now, filled in
  Task 8) and `src/trading_bot/framework/errors.py`.
- **IMPLEMENT**:
  ```python
  """
  Framework exception hierarchy.

  Every error class inherits BOTH FrameworkError and ValueError. The double
  inheritance is deliberate: the repo's established convention is that a
  programmer/config error raises ValueError (storage.py, engine._assert_interval
  at engine.py:216, scripts/bruteforce/registry.py:130), and existing callers
  and tests catch ValueError. Inheriting it means `except ValueError` keeps
  working while `except FrameworkError` becomes possible, and no caller has to
  learn a new base class to keep behaving correctly.
  """


  class FrameworkError(Exception):
      """Base for every framework error. Never raised directly."""


  class ContractError(FrameworkError, ValueError):
      """A plug-in or value violates a contract in framework/contracts.py.

      Raised at REGISTER time for a malformed callable or ParamSpec, and at
      RUN time when a plug-in returns the wrong type.
      """


  class RegistryError(FrameworkError, ValueError):
      """A registration or lookup failed: duplicate key, unknown kind, bad
      name casing, empty rationale, unknown plug-in key, or a fatal plug-in
      import during load_all()."""


  class GraphError(FrameworkError, ValueError):
      """A StrategyGraph is malformed, references an unknown plug-in, carries an
      illegal parameter value, or was serialized under a different
      SCHEMA_VERSION."""
  ```
- **MIRROR**: `engine.py:216-221` for the "raise instead of failing silently" error-message register
  — every message says what disagreed with what, and why it matters.
- **IMPORTS**: none.
- **GOTCHA**: Resist adding an error code enum or a `.details` dict. The repo raises with formatted
  f-string messages; matching that keeps `pytest.raises(..., match=...)` readable.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot.framework.errors import GraphError; assert issubclass(GraphError, ValueError); print('ok')"`.

### Task 3: `framework/contracts.py`
- **ACTION**: Create `src/trading_bot/framework/contracts.py`. This is the largest single file
  (~380 lines with docstrings) and every other module depends on it.
- **IMPLEMENT**, in this order:

  1. **Module docstring** stating the universal conventions and reproducing A1 verbatim:
     ```python
     """
     The seven plug-in contracts (v0.3.0 shared architecture contract §3).

     UNIVERSAL CONVENTIONS, true of every type and every plug-in here:
       - All timestamps are epoch MILLISECONDS, UTC, candle OPEN time — the same
         convention as data/ohlcv.db and config.py's header. Never a formatted
         date string, never seconds.
       - All prices are float. All frames are pd.DataFrame indexed by ts
         (ascending) with columns open/high/low/close/volume.
       - Every contract is PURE and side-effect free except DataSource (reads
         SQLite) and Reviewer (writes review records).
       - Plug-ins are plain FUNCTIONS registered by decorator, not subclasses.
         The Protocols below document the call shape and support isinstance()
         for object-shaped contracts; @runtime_checkable checks member PRESENCE
         only, never signatures, so the six function-shaped kinds are validated
         at register time by check_callable_shape().

     A1 — THERE IS NO INDICATOR CONTRACT. indicators/{wilder,bollinger,donchian}.py
     stay pure pandas functions with no Protocol of their own; "migrated behind
     the contracts" means every one of them is reachable from a strategy graph
     only through a registered plug-in that declares its parameters as
     ParamSpecs. The PRD's Phase 3 row implies an eighth contract; the shared
     architecture contract defines seven, and it wins. An eighth Protocol would
     have one implementation shape and no consumer.
     """
     ```
  2. **Imports** — note the two that must be `TYPE_CHECKING` only:
     ```python
     import inspect
     import math
     import random
     from dataclasses import dataclass, field
     from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, TypeAlias, runtime_checkable

     import pandas as pd

     from trading_bot.framework.errors import ContractError
     from trading_bot.signals.breakout import BreakoutEvent
     from trading_bot.signals.patterns import PatternCandidate
     from trading_bot.signals.setup import Signal

     if TYPE_CHECKING:  # runtime import would be circular / needlessly heavy
         from trading_bot.backtest.engine import Trade
         from trading_bot.framework.context import EvalContext
         from trading_bot.framework.graph import StrategyGraph

     # Phase 5 owns the concrete ReviewRecord/ReviewContext (feedback/records.py).
     # Aliased to Any here so the Reviewer protocol is expressible now WITHOUT
     # Phase 3 pre-empting Phase 5's schema and WITHOUT Phase 5 needing to edit
     # framework/, which is Phase 3's exclusively (contract §2).
     ReviewContext: TypeAlias = Any
     ReviewRecord: TypeAlias = Any
     ```
  3. **`ParamSpec`** — the type that must simultaneously render a UI control (Phase 7) and bound a
     mutation (Phase 6):
     ```python
     PARAM_KINDS = ("int", "float", "bool", "choice")


     @dataclass(frozen=True)
     class ParamSpec:
         """One declared, legal-bounded plug-in parameter.

         This replaces scripts/bruteforce/registry.py's `grid: dict[str, list]`
         and is the single reason the grid becomes a spec: a list of values can
         be swept, but it cannot tell a UI what control to draw nor a mutator
         what the legal range is BETWEEN the listed values.

         Attributes:
             kind: One of PARAM_KINDS.
             default: The value used when a graph omits this parameter. MUST be
                 legal under this spec — checked here, at import time, so a bad
                 default is a startup failure rather than a silent mid-sweep one.
             bounds: (low, high) INCLUSIVE. Required for int/float, forbidden
                 otherwise. This is what Phase 6's mutator jitters within.
             choices: Allowed values. Required for choice, forbidden otherwise.
             step: Control granularity for the UI and the mutation quantum for
                 the mutator. Defaults to 1 for int, None for float.
             doc: One-line control label / tooltip. Required non-empty — an
                 unlabelled knob in a builder UI is an invitation to sweep
                 something nobody understands.
         """

         kind: str
         default: Any
         bounds: tuple[float, float] | None = None
         choices: tuple[Any, ...] | None = None
         step: float | None = None
         doc: str = ""

         def __post_init__(self) -> None: ...   # see validation rules below
         def is_legal(self, value: Any) -> bool: ...
         def check(self, value: Any, *, where: str) -> Any:
             """Return value coerced to this spec's type, or raise ContractError
             naming `where` (e.g. "detector.donchian-breakout.entry_period")."""
         def clamp(self, value: Any) -> Any:
             """Nearest legal value. int/float clamp to bounds and snap to step;
             choice falls back to default; bool coerces via bool(). Phase 6's
             mutator calls this so a jitter can never produce an illegal graph."""
     ```
     `__post_init__` validation rules, all raising `ContractError`:
     `kind in PARAM_KINDS`; `doc.strip()` non-empty; `int`/`float` require `bounds` with
     `low <= high` and forbid `choices`; `choice` requires a non-empty `choices` tuple and forbids
     `bounds`; `bool` forbids both; `is_legal(default)` must hold; `bounds`/`choices` must be
     tuples, not lists (frozen-dataclass hashability — Phase 6 hashes graphs);
     `step` must be `> 0` when given. Default `step = 1` for `kind == "int"`.
     `is_legal`: `int` → `isinstance(v, int) and not isinstance(v, bool) and low <= v <= high`;
     `float` → `isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and low <= v <= high`;
     `bool` → `isinstance(v, bool)`; `choice` → `v in choices`.
  4. **The four payload dataclasses**, exactly the fields in contract §3's table, all
     `@dataclass(frozen=True)`:
     ```python
     @dataclass(frozen=True)
     class DetectedEvent:
         """A detector's structural finding, awaiting a trigger.

         DELIBERATELY field-compatible with signals.patterns.PatternCandidate
         (kind/direction/breakout_level->level/target_height/start_ts/end_ts) so
         migrating a v0.2.0 detector is an adapter, not a rewrite, and
         signals.breakout.check_breakout can be reused against it unchanged.

         `meta` carries whatever the matching PositionPolicy needs that the six
         fixed fields cannot express — the fade policy's stop_level and target,
         a detector's ADX reading — plus the trigger facts the executor stamps
         in before calling confirmations and the policy (see execute.py:
         trigger_ts / trigger_price / trigger_level / volume_ratio /
         volume_high). Values are floats: booleans are 0.0/1.0.
         """
         kind: str
         direction: str          # "long" | "short"
         level: float
         target_height: float
         start_ts: int
         end_ts: int
         meta: Mapping[str, float] = field(default_factory=dict)

     @dataclass(frozen=True)
     class ConfirmationVerdict:
         passed: bool
         name: str
         score: float = 0.0
         reason: str = ""

     @dataclass(frozen=True)
     class PositionPlan:
         symbol: str
         ts: int
         direction: str
         entry: float
         stop: float
         target: float
         risk_pct: float
         reward_pct: float
         rr: float
         source: str      # the DetectedEvent.kind — becomes Trade.pattern

     @dataclass(frozen=True)
     class FilterVerdict:
         accepted: bool
         name: str
         reason: str = ""
         measured: Mapping[str, float] = field(default_factory=dict)
     ```
  5. **The seven Protocols**, verbatim from contract §3, with `@runtime_checkable`, string
     annotations for `EvalContext`/`Trade`/`StrategyGraph`.
  6. **The four adapters** — the migration seam:
     ```python
     def event_from_candidate(c: PatternCandidate, *, meta=None) -> DetectedEvent:
         """PatternCandidate -> DetectedEvent. breakout_level -> level; every
         other field carries across by name."""

     def candidate_from_event(e: DetectedEvent) -> PatternCandidate:
         """DetectedEvent -> PatternCandidate, so signals.breakout.check_breakout
         works against it unchanged. check_breakout reads only breakout_level,
         direction and end_ts (breakout.py:104, 113, 124-127), so this round-trip
         is lossless for everything the trigger consumes. `meta` is dropped —
         PatternCandidate has nowhere to put it and check_breakout never reads it.
         """

     def plan_to_signal(plan: PositionPlan, event: BreakoutEvent) -> Signal:
         """PositionPlan + trigger event -> signals.setup.Signal, the executor's
         internal currency (A9).

         The trigger event is a REQUIRED argument rather than optional because
         volume_ratio/volume_high live only on it: PositionPlan has no volume
         fields, and defaulting them would silently zero Trade.volume_high and
         break parity in a way no arithmetic check would catch.
         `Signal.pattern = plan.source`, so metrics' "regime/pattern" buckets
         (metrics.py:31) are unchanged.
         """

     def plan_from_signal(sig: Signal, *, source: str) -> PositionPlan:
         """Signal -> PositionPlan, for policies that wrap a v0.2.0 builder which
         already returns a Signal (setup.build_signal, meanrev.build_fade_signal).
         plan_to_signal(plan_from_signal(s, source=s.pattern), event) == s for
         every field, given the event that produced s — pinned by
         test_framework_contracts.py::TestAdapters."""
     ```
  7. **`check_callable_shape`** — the register-time structural check:
     ```python
     _EXPECTED_FIRST_PARAMS = {
         "data": ("conn",),
         "detector": ("ctx",),
         "confirmation": ("ctx", "event"),
         "policy": ("ctx", "event"),
         "filter": ("ctx", "plan"),
         "reviewer": ("trade", "context"),
         "mutator": ("graph", "rng"),
     }

     def check_callable_shape(kind: str, fn: Callable, *, key: str) -> None:
         """Verify a plug-in's leading positional parameters are named as its
         contract requires, and that every remaining parameter is keyword-capable.

         @runtime_checkable Protocols check member PRESENCE only, never
         signatures, so isinstance() would accept a detector whose first argument
         is a DataFrame. This is the cheap structural substitute, run once at
         import time. It checks NAMES, not types (no annotations are required
         anywhere in this repo), which is enough to catch the mistakes that
         actually happen: wrong argument order and a forgotten ctx.

         Raises:
             ContractError: naming `key`, the expected leading parameters, and
                 what was found.
         """
     ```
- **MIRROR**: `patterns.py:58-77` for the frozen-dataclass-with-Attributes-docstring style;
  `bollinger.py:19-52` for keyword-only-with-config-fallback; `setup.py:131-136` for the NaN guard
  idiom (`ParamSpec.is_legal` uses `math.isfinite`, never `v == v`).
- **IMPORTS**: as listed above.
- **GOTCHA**: (a) **`ParamSpec` shadows `typing.ParamSpec`** — never `import typing` with attribute
  access anywhere in `trading_bot`; import named symbols only. (b) `Trade`, `EvalContext` and
  `StrategyGraph` must be `TYPE_CHECKING`-only with **quoted** annotations: `graph.py` imports
  `contracts.ParamSpec`, so a runtime `contracts → graph` import is a cycle, and `context.py`
  likewise. Parameter annotations are evaluated at `def` time in this repo (no
  `from __future__ import annotations`), so an unquoted `EvalContext` is a `NameError` at import.
  (c) `bool` is a subclass of `int` in Python — `is_legal` for `kind="int"` must explicitly reject
  `isinstance(v, bool)`, or `entry_period=True` validates as the integer 1. (d) `meta` defaults must
  use `field(default_factory=dict)`, and a frozen dataclass holding a plain `dict` is **unhashable**
  — that is fine because graphs are hashed via canonical JSON (Task 6), never via `hash()`. Do not
  "fix" it by adding `eq=False` or freezing to a `tuple`; `DetectedEvent` equality is used by the
  candidate-parity test.
- **VALIDATE**: `.venv/bin/python -m py_compile src/trading_bot/framework/contracts.py` then
  `.venv/bin/python -c "from trading_bot.framework import contracts as c; s=c.ParamSpec(kind='int', default=20, bounds=(5,200), doc='x'); print(s.step, s.is_legal(20), s.is_legal(True), s.clamp(999))"`
  → `1 True False 200`.

### Task 4: `framework/registry.py`
- **ACTION**: Create `src/trading_bot/framework/registry.py`.
- **IMPLEMENT**:
  ```python
  """
  Plug-in registry: the @register decorator, REGISTRY, and load_all().

  Direct descendant of scripts/bruteforce/registry.py, and its four hard-won
  lessons are carried forward deliberately rather than rediscovered:

    1. `rationale` is MANDATORY and non-empty. Quoting the ancestor: "an
       unmotivated strategy in a 10,000-combo sweep is just noise with a name,
       and the report needs to state the prior." Under Phase 6's population this
       matters more, not less.
    2. Duplicate names RAISE, naming the module that already claimed the key.
    3. Import errors during load_all() are FATAL, never skipped — "a family
       silently missing from the leaderboard would read as 'tested and found
       wanting'."
    4. Trial counting is first-class. Here it is `PluginSpec.combo_count()` over
       ParamSpec bounds/choices, reported by `cli.py plugins`, so the degrees of
       freedom a graph exposes are visible BEFORE Phase 6 spends them.
  """
  ```
  Then:
  ```python
  KINDS = ("data", "detector", "confirmation", "policy", "filter", "reviewer", "mutator")
  _NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")   # lowercase-hyphen, contract §3
  TIERS = (1, 2, 3, 4)                                    # contract §9 reliability tiers

  @dataclass(frozen=True)
  class PluginSpec:
      kind: str
      name: str
      key: str                                  # f"{kind}.{name}"
      impl: Callable
      params: Mapping[str, ParamSpec]
      rationale: str
      timeframes: tuple[str, ...] = ()
      tier: int | None = None
      module: str = ""                          # impl.__module__, for error messages

      def defaults(self) -> dict: ...
      def resolve(self, overrides: Mapping | None) -> dict:
          """defaults() merged with overrides, every value ParamSpec-checked.
          Unknown keys raise RegistryError listing the legal ones — a typo'd
          parameter must never be silently ignored, which is how a sweep ends up
          measuring the default 200 times."""
      def combo_count(self, *, points: int = 3) -> int:
          """Rough degrees of freedom: len(choices) per choice/bool axis, `points`
          per bounded numeric axis. Reported by `cli.py plugins`, never used as a
          DSR trial count — the ledger counts ACTUAL evaluations (contract §4)."""

  REGISTRY: dict[str, PluginSpec] = {}

  def register(kind, *, name, params, rationale, timeframes=(), tier=None) -> Callable: ...
  def get(key: str) -> PluginSpec: ...
  def by_kind(kind: str) -> dict[str, PluginSpec]: ...
  def load_all(package: str | None = None) -> dict[str, PluginSpec]: ...

  @contextlib.contextmanager
  def temporary_registry():
      """Snapshot REGISTRY, yield, restore. For tests that register throwaway
      plug-ins.

      Deliberately a snapshot/restore rather than a clear(): emptying the
      registry cannot be undone by re-importing, because importlib caches
      modules and the decorators would never run again — a clear() would leave
      every subsequent test in the session looking at an empty registry and
      failing for a reason unrelated to itself.
      """
  ```
  `register` validations, all raising `RegistryError` and all naming the offending key:
  `kind in KINDS`; `_NAME_RE.match(name)` (message must say "lowercase-hyphen, e.g.
  'donchian-breakout'" and explicitly reject `camelCase` / `snake_case`); `rationale.strip()`
  non-empty; every `params` value `isinstance(..., ParamSpec)`; every `timeframes` entry in
  `storage.TIMEFRAME_MS`; `tier in TIERS or tier is None`; duplicate `key` →
  `f"{key!r} is already registered (by {REGISTRY[key].module}); pick a distinct name"`. Then
  `contracts.check_callable_shape(kind, fn, key=key)`. Returns `fn` **unchanged** (prior art:
  `registry.py:151`) so the plain function stays directly callable and unit-testable.
  `load_all` uses `pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + ".")`, skips names
  whose last segment starts with `_`, and wraps any exception as
  `RegistryError(f"plug-in module {mod!r} failed to import; import errors are fatal, never skipped, because a silently missing family reads as 'tested and found wanting'") from exc`.
- **MIRROR**: `scripts/bruteforce/registry.py:107-153` (decorator + every validation) and
  `162-179` (`load_all`) — read both before writing.
- **IMPORTS**: `contextlib`, `pkgutil`, `importlib`, `re`, `dataclasses`, `typing`;
  `from trading_bot import config`; `from trading_bot.data import storage`;
  `from trading_bot.framework.contracts import ParamSpec, check_callable_shape`;
  `from trading_bot.framework.errors import RegistryError`.
- **GOTCHA**: (a) `load_all()` must be **idempotent** — `importlib.import_module` is cached, so the
  decorators run once and a second call is a no-op. Pin this with a test; if it ever raises
  "already registered", something is importing a module under two names (e.g. mixing
  `trading_bot.plugins.x` with a `sys.path`-relative `plugins.x`). (b) `walk_packages` **imports
  each package** to read `__path__`, so a broken `__init__.py` raises there, not at the leaf —
  the wrapped message must include the module name or the traceback is unreadable. (c) Do not
  `REGISTRY.clear()` anywhere outside `temporary_registry`. (d) `by_kind` must return a dict sorted
  by name so `cli.py plugins` output is stable and diffable.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot.framework import registry as r; print(r.KINDS); print(len(r.load_all()))"` → the 7 kinds and (after Tasks 9-14) `6`.

### Task 5: `framework/context.py`
- **ACTION**: Create `src/trading_bot/framework/context.py`: `EvalSession`, `EvalContext`, the two
  cache seams, `clear_caches`, `cache_stats`, `assert_trailing_only`.
- **IMPLEMENT**:
  1. **Module docstring — the ten structural guarantees.** This text is the phase's central
     argument; write it before the code:
     ```python
     """
     EvalContext: a read-only, time-bounded view of market data for ONE
     evaluation instant, and EvalSession, which owns the frames and the memos.

     Every no-lookahead guarantee engine.py documents at engine.py:16-52 is
     preserved here STRUCTURALLY rather than by convention. A plug-in author
     cannot see the future by accident, and cannot see it on purpose without
     obviously reaching around this class:

       1. A plug-in never receives `conn` or a full frame. It receives an
          EvalContext and nothing else.
       2. The master frames live on the EvalSession in a name-mangled attribute;
          the EvalContext's public API has no accessor that returns one. Every
          frame/series accessor slices at now_ms INSIDE the context.
       3. Returned numpy arrays are zero-copy slices with flags.writeable
          = False; returned DataFrames are pandas copy-on-write slices, so a
          plug-in cannot mutate what the next plug-in sees.
       4. now_ms is set by the executor from BAR CLOSE TIMES only. There is no
          setter and no wall clock anywhere in this module — nothing calls
          time.time().
       5. now_ms is non-decreasing PER ROLE ("regime"/"setup"/"trigger"), which
          catches an executor bug that walks backwards. Per-role rather than
          global because a setup-bar context legitimately trails the previous
          trigger-bar context.
       6. Every loaded series passes engine._assert_interval — the SAME function,
          imported, with the same error text — so a partial tier migration
          cannot fail silently in the flattering direction.
       7. series() factories must be TRAILING-ONLY (value at i depends on bars
          [0..i]). assert_trailing_only() proves it for a factory; truncation is
          the second line of defence, not the first.
       8. regime() is the single implementation of the "last regime bar CLOSED by
          t" rule (engine.py:315-317). A plug-in cannot roll its own from a
          regime frame, because frame(regime_tf) is truncated too.
       9. Pivot confirmation is INHERITED from truncation: find_pivots on a frame
          ending at the last closed bar can only emit pivots with a full
          PIVOT_SPAN window on both sides (pivots.py:7-13). Nothing extra is
          needed, and nothing may bypass the truncation to "get one more pivot".
      10. Warmup NaNs are never filled. A plug-in that gets NaN during warmup
          must return no event, exactly as detect_donchian_setups does
          (donchian.py:112-113).
     """
     ```
  2. **`EvalSession`** — one per `run_graph_backtest` call, per symbol:
     ```python
     class EvalSession:
         """Owns the loaded frames, the derived close-time arrays, the regime
         labels, and the memos. One per (symbol, graph, run)."""

         def __init__(self, source, symbol, *, tiers, regime_gate): ...
             # loads each tier via source.frame(...), calls engine._assert_interval
             # with role="regime"/"setup"/"trigger", stores fingerprints, computes
             # close_<tier> = index + interval, and memoizes the regime labels.

         def context(self, role: str, now_ms: int) -> "EvalContext":
             """A context bound to now_ms. Asserts now_ms is non-decreasing for
             this role (ContractError otherwise)."""
     ```
  3. **`EvalContext`** — the entire surface a plug-in may ask for, and nothing else:
     ```python
     class EvalContext:
         @property
         def symbol(self) -> str: ...
         @property
         def now_ms(self) -> int:
             """Close time of the bar under evaluation. Data-derived, never a clock."""
         @property
         def tiers(self) -> tuple[str, str, str]:
             """(regime_tf, setup_tf, trigger_tf) — names, so a plug-in can ask
             for 'the setup tier' without hardcoding '4h'."""

         def interval_ms(self, timeframe: str) -> int: ...
         def bar_index(self, timeframe: str) -> int:
             """Positional index of the last bar CLOSED by now_ms; -1 if none."""
         def frame(self, timeframe: str) -> pd.DataFrame:
             """All CLOSED bars: df.iloc[: bar_index+1]."""
         def window(self, timeframe: str, bars: int) -> pd.DataFrame:
             """The trailing `bars` closed bars. Equals engine.py:339's
             df_setup.iloc[max(0, h_idx+1-N) : h_idx+1] exactly."""
         def latest(self, timeframe: str) -> pd.Series | None: ...
         def regime(self) -> str:
             """Label of the last regime bar CLOSED by now_ms; 'uncertain' if none."""
         def series(self, timeframe: str, name: str, factory, **params) -> np.ndarray:
             """Memoized indicator array, TRUNCATED at the current bar."""
     ```
     **What a plug-in may NOT ask for, and why** — put this list in the class docstring: no `conn`
     (a plug-in must not be able to query arbitrary history or write); no untruncated frame; no
     wall-clock time; no `random` (Mutators receive their own seeded `rng`, contract §3); no
     network; no other symbol (cross-sectional strategies are a deliberate future extension that
     needs its own no-lookahead argument, not an accident of API surface).
  4. **`series()` — the caching seam**, with the key spelled out:
     ```python
     def series(self, timeframe, name, factory, **params):
         """Memoized indicator array, TRUNCATED at the last bar closed by now_ms.

         factory(full_frame, **params) -> np.ndarray | pd.Series is computed ONCE
         over full history per key and cached process-wide; the array handed back
         is a zero-copy, READ-ONLY slice arr[: bar_index + 1]. That is what makes
         the seam both fast and safe: the computation sees all the bars (so it is
         O(n) per run, not O(n^2)), and the caller cannot index into the future.

         The factory MUST be trailing-only. Truncation is not a substitute:
         a centred rolling window would poison arr[i] itself. Use
         assert_trailing_only() in the plug-in's tests.

         CACHE KEY — everything that can change a value, and nothing that cannot:
             ("series", name, symbol, timeframe,
              engine._fingerprint(master_frame),      # CONTENT, not (symbol, tf)
              tuple(sorted(params.items())))
         In the key because it changes the numbers: the frame content, every
         factory parameter (period, num_std, ...).
         NOT in the key because it cannot: start_ms/end_ms (they bound the loop,
         not the series), fee/slippage/funding, max_hold_bars, and the graph's
         name or meta.
         """
     ```
     Also on the session, mirroring `engine.py:293-304` one-for-one:
     `atr(timeframe, period)` → key `("atr", symbol, tf, fp, period)`;
     `channels(timeframe, period)` → key `("chan", symbol, tf, fp, period)` returning
     `(lower, upper)`; `labels()` → key
     `("labels", symbol, regime_tf, fp, adx_trend_threshold, atr_extreme_percentile)`.
     And the **candidate cache**, the seam that actually makes Phase 6 viable:
     ```python
     def candidates(self, branch_key: str, setup_idx: int, produce) -> list:
         """Per-(branch, setup-bar) memo of detector output.

         This — not the indicator memo — is what engine.py's cand_cache
         (engine.py:328-349) actually relies on, and why: detect_donchian_setups
         computes a Wilder ADX over a PATTERN_LOOKBACK_BARS window for EVERY
         setup bar, and a Wilder recursion over a 180-bar window is NOT the tail
         of one over full history, so that cost cannot be cached away by the
         indicator seam. Memoizing the CANDIDATE instead is exact and it is
         shared across every graph and every parameter combo that leaves the
         branch untouched — which is the whole performance argument for a
         population: 200 graphs that share a donchian branch share its candidates.

         CACHE KEY:
             ("cands", branch_content_hash, symbol, setup_tf,
              fingerprint(df_setup), regime_thresholds, bool(config.FADE_ENABLED))
         branch_content_hash is the canonical hash of the branch's RESOLVED
         detector node, so any parameter change invalidates. FADE_ENABLED is in
         the key for the reason engine.py:322-327 states: it is read at call
         time, so a live flip must not be served a cached pre-flip list.
         """
     ```
  5. **`clear_caches()`**, **`cache_stats() -> dict`** (entries, hits, misses, approximate bytes —
     so the performance claims in the phase report are measured, not asserted), and
     **`assert_trailing_only(factory, df, *, sample=8, **params)`** raising `ContractError` with
     the first offending index — the port of `scripts/bruteforce/core.assert_causal`'s idea.
- **MIRROR**: `engine.py:158-174` (`_fingerprint`, imported); `engine.py:315-317` (the
  `searchsorted` rule, reproduced exactly in `bar_index`/`regime`); `engine.py:328-349` (the
  candidate memo's structure and its key comment); `classifier.py:56-72` for the
  `rolling(window, min_periods=window)` trailing-window idiom the docstring cites.
- **IMPORTS**: `numpy as np`, `pandas as pd`, `from trading_bot import config`,
  `from trading_bot.data import storage`,
  `from trading_bot.backtest.engine import _assert_interval, _fingerprint`,
  `from trading_bot.regime.classifier import classify_series`,
  `from trading_bot.framework.errors import ContractError`.
- **GOTCHA**: (a) **The `atr` key must include `period`; `engine.py:293` omits it** because
  `ATR_STOP_PERIOD` is config-fixed there. The graph exposes `atr_period` as a `ParamSpec`, so
  omitting it would serve a 14-period ATR to a graph asking for 21 — a silent wrong-number bug that
  parity would *not* catch (the parity graph uses 14). Same for `channel_period`. This is a
  deliberate improvement on the engine's key, not a divergence. (b) `flags.writeable = False` must
  be set on the **slice**, not the cached array — setting it on the cached array is also fine and
  stricter, so do both: cache immutable, hand out immutable views. (c) Importing two private names
  from `engine` is intentional and precedented (`meanrev.py:35` imports `setup._load_df`;
  `engine.py:81` imports `meanrev._to_trigger_candidate`). Do **not** copy their bodies. (d) The
  candidate memo grows without bound across a Phase 6 generation — `cache_stats()` exists to make
  that measurable and `clear_caches()` to bound it; recommend in the docstring that Phase 6 clear
  between generations. (e) `bar_index` uses `searchsorted(close_times, now_ms, side="right") - 1`
  where `close_times = index + interval`; a bar is closed when `ts + interval <= now_ms`, matching
  `classifier.current_regime`'s rule (`classifier.py:181-184`) and `setup._load_df`
  (`setup.py:209-210`). Off-by-one here is the single most likely silent-lookahead bug in the
  phase — `test_framework_contracts.py::TestNoLookahead` pins it against a hand-built series.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_framework_contracts.py -v` (after Task 17).
  Interim: `.venv/bin/python -m py_compile src/trading_bot/framework/context.py`.

### Task 6: `framework/graph.py`
- **ACTION**: Create `src/trading_bot/framework/graph.py`. Reproduce A2, A4 and A5 in the module
  docstring.
- **IMPLEMENT**:
  1. `SCHEMA_VERSION = 1` with a banner: "bump ONLY on a breaking layout change, and add a
     migration in `from_dict` when you do; every persisted graph, every `strategy_versions` row and
     every `trial_ledger` row keys on a hash that includes this number."
  2. **The five frozen dataclasses.** Every collection field is a `tuple`, never a `list`, so
     `dataclasses.replace` in Phase 6 produces valid graphs and `__post_init__` can enforce it:
     ```python
     @dataclass(frozen=True)
     class NodeSpec:
         id: str                                    # unique in the graph, [a-z0-9][a-z0-9_-]*
         key: str                                   # registry key, e.g. "detector.donchian-breakout"
         params: Mapping[str, Any] = field(default_factory=dict)   # OVERRIDES only

     @dataclass(frozen=True)
     class RegimeGate:
         """A2: the regime classifier is a graph-level FIELD, not a node.

         NOT SWEEPABLE. These are the one measured-healthy layer's thresholds
         (walkforward.py:26-29 keeps them out of every grid); they live here as a
         plain dataclass rather than a plug-in with ParamSpec bounds precisely so
         Phase 6's param-jitter mutator has no legal handle on them.
         `enabled=False` labels every bar "any" and makes every branch eligible —
         for a regime-free strategy, not for escaping the gate.
         """
         timeframe: str = config.REGIME_TIMEFRAME
         adx_trend_threshold: float = config.ADX_TREND_THRESHOLD
         atr_extreme_percentile: float = config.ATR_EXTREME_PERCENTILE
         enabled: bool = True

     @dataclass(frozen=True)
     class TriggerSpec:
         """A3: check_breakout is executor machinery. Only these three numbers are
         graph-visible; the crossing test, the freshness pair, the end_ts skip and
         the interval_ms contiguity check are NOT expressible in a graph."""
         lookback_bars: int = 1          # 1 = the bar under evaluation only
         volume_lookback: int = config.VOLUME_LOOKBACK
         volume_high_ratio: float = config.VOLUME_HIGH_RATIO

     @dataclass(frozen=True)
     class ExitPolicySpec:
         """A4: exits are declarative flags, per branch.

         This is what makes engine.py's MANDATORY DEVIATION (engine.py:392-411,
         523-530) expressible: Donchian trades take trail + opposite-channel
         exits, fade trades keep frozen stop/target/time/end behavior. See the
         plan's "exit-asymmetry proof" — one generic block reproduces both.
         """
         stop: bool = True                  # must be True; see validate()
         target_enabled: bool = False
         trail_enabled: bool = False
         trail_atr_multiple: float = config.TRAIL_ATR_MULTIPLE
         trail_atr_period: int = config.ATR_STOP_PERIOD
         channel_exit: bool = False
         channel_period: int = config.DONCHIAN_ENTRY_PERIOD
         max_hold_bars: int | None = None   # None => run_graph_backtest's value

     @dataclass(frozen=True)
     class Branch:
         """One detector and everything downstream of it.

         `regimes` is the routing rule, and it reads like scan.py:55-61's dispatch
         table on purpose (A2/A6) — that is the surface most at risk of drifting
         from the live path.
         """
         id: str
         detector: NodeSpec
         policy: NodeSpec
         regimes: tuple[str, ...] = ("any",)
         confirmations: tuple[NodeSpec, ...] = ()     # ALL must pass (AND)
         exits: ExitPolicySpec = field(default_factory=ExitPolicySpec)
         enabled: bool = True

     @dataclass(frozen=True)
     class StrategyGraph:
         name: str
         data: NodeSpec
         branches: tuple[Branch, ...]
         regime: RegimeGate = field(default_factory=RegimeGate)
         trigger: TriggerSpec = field(default_factory=TriggerSpec)
         filters: tuple[NodeSpec, ...] = ()           # ALL must accept
         meta: Mapping[str, Any] = field(default_factory=dict)
         schema_version: int = SCHEMA_VERSION

         def ordered_branches(self) -> tuple[Branch, ...]:
             """Branches sorted by id — the ONLY iteration order the executor uses.

             A5: canonical order is also runtime order. Without this, two graphs
             that hash identically could break a rank_signals tie differently and
             produce different trades under the same hash, which would corrupt
             Phase 5's version registry and Phase 6's ledger at the root.
             """
         def resolved(self) -> "StrategyGraph":
             """Every node's params materialized from registry defaults.
             graph_hash(g) == graph_hash(g.resolved())."""
         def to_dict(self) -> dict: ...
         @classmethod
         def from_dict(cls, payload: Mapping) -> "StrategyGraph": ...
     ```
  3. **`validate(graph) -> None`** — collects **all** problems then raises one `GraphError` listing
     them (a builder UI showing one error at a time is a bad UI, and a mutator that produced three
     illegal params should learn all three). Rules, each with its own message:
     | # | Rule | Message must say |
     |---|---|---|
     | 1 | `schema_version == SCHEMA_VERSION` | found vs expected, and "regenerate or migrate" |
     | 2 | `name` matches `^[a-z0-9][a-z0-9._-]*$` and is non-empty | it becomes `data/strategies/<name>.strategy.json` |
     | 3 | slot/kind agreement: `data.key` starts `"data."`, detectors `"detector."`, confirmations `"confirmation."`, policies `"policy."`, filters `"filter."` | which slot, which key |
     | 4 | every key in `REGISTRY` | the key, plus `difflib.get_close_matches` suggestions from `by_kind` |
     | 5 | node `id`s unique across the whole graph; branch `id`s unique | the duplicate id and both owners |
     | 6 | `branches` non-empty and at least one `enabled` | "a graph with no enabled branch produces no trades and would read as a strategy with no edge" |
     | 7 | every `regimes` entry in `classifier.REGIMES + ("any",)` | the bad label and the legal set |
     | 8 | params: `spec.resolve(node.params)` succeeds for every node | delegated to `PluginSpec.resolve` (unknown key / illegal value) |
     | 9 | `exits.stop is True` | "a stopless strategy is not expressible in v0.3.0; the PositionPlan's stop is mandatory" |
     | 10 | `exits.trail_atr_multiple > 0` when `trail_enabled` | the value |
     | 11 | `exits.channel_period` == the branch detector's resolved `entry_period`, when `channel_exit` and the detector declares one | both values, and "the exit channel and the entry channel are the same 20 bars, other side (donchian.py:146-152); letting them diverge silently changes the exit without changing the entry" |
     | 12 | `exits.trail_atr_period` == the branch policy's resolved `atr_period`, when `trail_enabled` and the policy declares one | both values, and "engine.py uses ONE atr_value for both the entry stop and the trail (engine.py:537)" |
     | 13 | `trigger.lookback_bars >= 1`; `volume_lookback >= 1`; `volume_high_ratio > 0` | the value |
     | 14 | `regime.timeframe` in `storage.TIMEFRAME_MS` | the value and the legal keys |
  4. **`graph_hash(graph) -> str`** and **`canonical_dict(graph) -> dict`**:
     ```python
     def canonical_dict(graph: "StrategyGraph") -> dict:
         """Order-independent, default-resolved dict for hashing.

         A5, restated as rules:
           - params are RESOLVED against registry defaults first, so omitting a
             parameter and setting it to its default hash identically;
           - `name` and `meta` are EXCLUDED — a rename or a note is not a new
             strategy, and Phase 5's version registry must not fork on one;
           - `branches`, each branch's `confirmations`, and `filters` are sorted
             by id; `regimes` is sorted; every dict is emitted with sort_keys;
           - `schema_version` IS included: a schema change changes meaning;
           - floats are normalized (`float(v)`, and -0.0 -> 0.0) so 20 and 20.0
             cannot produce two hashes for one strategy;
           - NaN/Inf are rejected (allow_nan=False) — a non-finite parameter is
             never legal and must not be hashable.
         """

     def graph_hash(graph: "StrategyGraph") -> str:
         """sha256 hex digest (64 chars) of canonical_dict's JSON.

         Phase 5's strategy_versions and Phase 6's trial_ledger both key on this
         (contract §6). Requires the referenced plug-ins to be importable, because
         resolving defaults needs the registry — call registry.load_all() first.
         """
         payload = json.dumps(canonical_dict(graph), sort_keys=True,
                              separators=(",", ":"), ensure_ascii=True, allow_nan=False)
         return hashlib.sha256(payload.encode("utf-8")).hexdigest()

     def short_hash(graph) -> str:
         """First 12 hex chars — for CLI output and log lines only, never a key."""
     ```
  5. **`save(graph, path=None) -> Path`** / **`load(path) -> StrategyGraph`** — JSON with
     `indent=2, sort_keys=True` and a trailing newline (so committed graphs diff cleanly); `save`
     validates first, creates `config.STRATEGY_DIR` with `parents=True, exist_ok=True` (mirroring
     `storage.connect`'s `db_file.parent.mkdir` at `storage.py:49`), and resolves a bare name to
     `Path(config.STRATEGY_DIR) / f"{graph.name}.strategy.json"`; `load` validates after parsing.
- **MIRROR**: `patterns.py:58-77` for frozen-dataclass docstrings; `walkforward.py:98-115` for the
  "appended, never reordered" dataclass discipline; `storage.py:140-146` for the numbered-INVARIANTS
  docstring style that `canonical_dict` copies.
- **IMPORTS**: `difflib`, `hashlib`, `json`, `re`, `dataclasses`, `pathlib.Path`, `typing`;
  `from trading_bot import config`; `from trading_bot.data import storage`;
  `from trading_bot.regime.classifier import REGIMES`;
  `from trading_bot.framework import registry`;
  `from trading_bot.framework.errors import GraphError`.
- **GOTCHA**: (a) `graph.py` imports `registry`, and `registry` imports `contracts` — so
  `contracts` must **not** import `graph` at runtime (Task 3 GOTCHA b). (b) `from_dict` must
  reject unknown top-level and nested keys, not ignore them: a UI or mutator that writes
  `"exits": {"trail": true}` (wrong name) would otherwise silently get `trail_enabled=False` and
  the operator would believe a trail was tested. (c) `field(default_factory=...)` is required for
  the mutable-ish defaults (`RegimeGate()`, `ExitPolicySpec()`, `dict`); a bare
  `regime: RegimeGate = RegimeGate()` shares one instance — harmless while frozen, but it breaks the
  moment anyone unfreezes, so use the factory. (d) `__post_init__` on `Branch`/`StrategyGraph` must
  raise `GraphError` if a collection arrived as a `list` (Phase 6 does `replace(...)` with
  comprehensions, and a list makes the graph unhashable-by-convention and order-unstable). (e) The
  pinned digest in `test_framework_graph.py` will change if **any** plug-in's `ParamSpec` default
  changes — that is correct and intended; regenerate it deliberately, with a note in the commit
  message, never casually.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_framework_graph.py -v` (after Task 19).
  Interim: `py_compile`.

### Task 7: `framework/execute.py` — `run_graph_backtest`
- **ACTION**: Create `src/trading_bot/framework/execute.py`. **Before writing a line, re-read
  `engine.py:225-546` and the "Parity by construction" table above.** Work down that table in order;
  do not improvise structure.
- **IMPLEMENT**:
  1. **Module docstring** reproducing A3, A6, A9, A10 and A11 verbatim, plus a pointer:
     "This module is a faithful re-expression of `backtest/engine.py:225-546`. Where the two
     disagree, `engine.py` is right and this file is broken. `tests/test_framework_parity.py` is the
     proof, and it is the acceptance gate for PRD Phase 3."
  2. **Signature exactly as contract §5**, argument for argument with `engine.run_backtest` minus
     `params`:
     ```python
     def run_graph_backtest(
         conn, graph: StrategyGraph, symbol: str, *,
         start_ms: int | None = None, end_ms: int | None = None,
         fee_pct: float | None = None, slippage_pct: float | None = None,
         funding_pct_per_day: float | None = None, max_hold_bars: int | None = None,
     ) -> list[Trade]: ...
     ```
     Config fallbacks copied from `engine.py:257-263`, including `cost = 2 * (fee + slip)`.
  3. **Setup**: `validate(graph)`; instantiate the DataSource from `graph.data` via
     `registry.get(graph.data.key).impl(conn, **resolved_params)`;
     `isinstance(source, contracts.DataSource)` or `ContractError`; build the `EvalSession`; return
     `[]` on any empty tier (`engine.py:268-269`).
  4. **The replay loop**, in `engine.py`'s exact order. Two internal helpers:
     ```python
     def _detect(branch, h_idx) -> list[DetectedEvent]:
         """One branch's events at a setup bar, memoized via session.candidates.
         Applies the regime gate (label in branch.regimes or "any" in
         branch.regimes) BEFORE calling the detector, exactly as engine.py:341-347
         gates on `reg == "trending"`."""

     def _plan(branch, event, trig_event, ctx) -> tuple[PositionPlan, Signal] | None:
         """Stamp the trigger facts into event.meta; run every confirmation
         (AND, short-circuit on the first failure); call the policy; run every
         graph filter; adapt to a Signal. Returns None at the first rejection."""
     ```
     The `open_trade` record — **shaped for Phase 4's extension (A11)**, so name every field:
     ```python
     open_trade = {
         "signal": sig,          # the adapted Signal — Trade is built from this
         "plan": plan,           # PositionPlan: carries .rr for Phase 4's planned_rr
         "confirmed": names,     # tuple[str, ...] of Confirmations that passed —
                                 #   for Phase 4's Trade.confirmations
         "branch": branch,       # the winning Branch: .exits drives the exit loop
         "exits": branch.exits,
         "entry_j": j,
         "regime": reg,
         "stop": sig.stop,       # MUTABLE when exits.trail_enabled
         "atr": atr_value,       # setup-tier ATR at entry, frozen (engine.py:537)
         "extreme": sig.entry,   # best price seen since entry
         "trailed": False,
         "h_idx": h_idx,
     }
     ```
  5. **`_close_out(rec, j, price, outcome)`** — the duplication of `engine.py:354-378` (A10) and the
     designated Phase 4 extension point (A11):
     ```python
     def _close_out(rec, j, price, outcome) -> None:
         """Append one Trade. Deliberate duplication of engine.run_backtest's
         close_out (engine.py:354-378) — see A10. When one changes, both must, and
         tests/test_framework_parity.py::TestCostArithmetic will say so.

         PHASE 4 EXTENSION POINT (contract §5). Phase 4 appends planned_rr,
         confirmations and strategy_version to Trade and populates them HERE, on
         the graph path only. The three values are already in `rec`
         (rec["plan"].rr, rec["confirmed"], and the graph's meta), so the diff is
         three added keyword arguments below and nothing else. Trade is
         constructed with KEYWORD ARGUMENTS ONLY, in field order, for exactly that
         reason — do not collapse it to positional.
         """
         s = rec["signal"]
         sign = 1.0 if s.direction == "long" else -1.0
         gross = sign * (price - s.entry) / s.entry
         hold_days = (int(ts_trig[j]) - s.ts) / 86_400_000.0
         trades.append(Trade(
             symbol=symbol, regime=rec["regime"], pattern=s.pattern,
             direction=s.direction, entry_ts=s.ts, entry=s.entry, stop=s.stop,
             target=s.target, exit_ts=int(ts_trig[j]), exit_price=price,
             outcome=outcome, pnl_pct=gross - cost - funding * hold_days,
             volume_high=s.volume_high,
             # Phase 4 appends here: planned_rr=rec["plan"].rr,
             #   confirmations=rec["confirmed"],
             #   strategy_version=graph.meta.get("version", ""),
         ))
     ```
     Note `s.stop` is the **initial** stop even when the trail moved it — `Trade.stop` is "a frozen
     record of the setup" (`engine.py:114-117`), and `rec["stop"]` is the live one. Getting this
     backwards changes `Trade.stop` on every trailed trade.
  6. **The exit block** — the single generic version from the exit-asymmetry proof, with the
     ratchet-after-exits ordering and its comment carried across verbatim from
     `engine.py:435-438`.
  7. **The entry block** — including the MEDIUM-2 `h_idx` line and a condensed version of its
     24-line comment (`engine.py:458-480`), ending with: "MEASURED CONSEQUENCE: recovering those
     bars ADDS trades that were unprofitable in-sample (142 → 158 trades, pooled Sharpe 0.431 →
     0.255, annualised 11.0% → 0.25%). The fix was kept because an entry rule must be an explicit,
     pre-registered decision. **Do not silently re-break it** — `test_framework_parity.py`
     pins it."
     Multi-candidate handling, reproducing `engine.py:518-522`:
     ```python
     # Only one trade at a time, so pick by the SAME rule live scanning ranks by
     # (setup.rank_signals) — taking the first candidate instead would make
     # measured performance a function of branch iteration order.
     if bar_signals:
         ranked = rank_signals([s for _, _, s in bar_signals])
         winner = ranked[0]
         branch, plan, _ = next(t for t in bar_signals if t[2] is winner)
     ```
     `is` identity, not `==`: two branches can legitimately produce equal-valued `Signal`s, and
     `==` would pick the wrong branch's `ExitPolicySpec`.
  8. **`clear_caches()`** delegating to `context.clear_caches()`, so a test needs one call.
- **MIRROR**: the whole of `engine.py:225-546`; specifically `engine.py:354-378` (`_close_out`),
  `engine.py:387-453` (exits), `engine.py:455-541` (entry), `engine.py:543-544` (the `"end"` close).
- **IMPORTS**: `numpy as np`; `from trading_bot import config`;
  `from trading_bot.backtest.engine import Trade`;
  `from trading_bot.signals.breakout import check_breakout`;
  `from trading_bot.signals.setup import rank_signals`;
  `from trading_bot.framework import contracts, registry`;
  `from trading_bot.framework.context import EvalSession`;
  `from trading_bot.framework.graph import StrategyGraph, validate`;
  `from trading_bot.framework.errors import ContractError`.
- **GOTCHA**: (a) **Do not import `run_backtest`.** If `run_graph_backtest` can ever delegate to it,
  the parity test is vacuous. (b) `check_breakout` must be called with
  `lookback_bars=graph.trigger.lookback_bars` **and** `interval_ms=trigger_ms` — dropping
  `interval_ms` disables the gap check (`breakout.py:115-122`) and a gapped pair reads as a fresh
  crossing. (c) Iterate `graph.ordered_branches()`, never `graph.branches` (A5). (d) The regime
  label is read at the **setup** bar's close (`engine.py:337`: `t = int(close_setup[h_idx])`), not
  at the trigger bar's — mixing them changes which candidates exist. (e) `atr_value` comes from
  `session.atr(setup_tf, period)[h_idx]` with the **branch policy's** `atr_period`, and the same
  array feeds the trail (validation rule 12 keeps them equal). (f) The `"end"` close uses `last_j`,
  which is the last bar with `bc <= end` — not `len(ts_trig)-1`. (g) `graph.filters` run on the
  `PositionPlan`, before `plan_to_signal`; confirmations run on the `DetectedEvent`, after the
  trigger fires and after the trigger facts are stamped into `meta`.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_framework_parity.py -v` (after Task 20).
  Interim: `py_compile`.

### Task 8: `framework/__init__.py`
- **ACTION**: Fill in `src/trading_bot/framework/__init__.py`.
- **IMPLEMENT**: A package docstring naming the five modules and the one seam, then explicit
  re-exports with `__all__`: the four errors; `ParamSpec`, `DetectedEvent`, `ConfirmationVerdict`,
  `PositionPlan`, `FilterVerdict`, the seven Protocols, the four adapters; `REGISTRY`, `KINDS`,
  `PluginSpec`, `register`, `get`, `by_kind`, `load_all`; `SCHEMA_VERSION`, `NodeSpec`, `Branch`,
  `RegimeGate`, `TriggerSpec`, `ExitPolicySpec`, `StrategyGraph`, `validate`, `graph_hash`,
  `short_hash`, `save`, `load`; `EvalContext`, `EvalSession`, `assert_trailing_only`;
  `run_graph_backtest`, `clear_caches`.
- **MIRROR**: `src/trading_bot/indicators/__init__.py` and `signals/__init__.py` for house style
  (read them; if they are bare, keep this one to a docstring plus the imports and skip `__all__`
  ornamentation).
- **GOTCHA**: Import order matters — `errors`, `contracts`, `registry`, `graph`, `context`,
  `execute`. Importing `trading_bot.framework` therefore imports `backtest.engine` (via `context`
  and `execute`), which imports every signal module. That is acceptable (~40 ms) but it means
  **`framework` must never be imported from `engine.py`, `signals/*`, or `indicators/*`** or the
  cycle closes. `walkforward.py` importing it is fine: nothing in `framework` imports
  `walkforward`.
- **VALIDATE**: `.venv/bin/python -c "import trading_bot.framework as f; print(f.SCHEMA_VERSION, len(f.KINDS))"` → `1 7`.

### Task 9: `plugins/` packages and `build_v020_graph`
- **ACTION**: Create `src/trading_bot/plugins/__init__.py` plus the three subpackage markers
  `plugins/{data,detectors,policies}/__init__.py`.
- **IMPLEMENT**: `plugins/__init__.py` gets the package docstring ("one module per plug-in family;
  `framework.registry.load_all()` walks this package and import errors are fatal") **and**:
  ```python
  def build_v020_graph(
      *,
      name: str = "donchian-v020",
      trail_enabled: bool = config.TRAIL_ENABLED,
      trail_atr_multiple: float = config.TRAIL_ATR_MULTIPLE,
      target_enabled: bool = config.DONCHIAN_TARGET_ENABLED,
      rr_floor: float = config.RR_FLOOR,
      adx_trend_threshold: float = config.ADX_TREND_THRESHOLD,
      atr_extreme_percentile: float = config.ATR_EXTREME_PERCENTILE,
      bb_num_std: float = config.BB_STD,
      include_fade: bool = True,
  ) -> StrategyGraph:
      """The v0.2.0 strategy expressed as a graph — the parity fixture, the
      `cli.py graph-validate` example, and Phase 6's seed genome.

      Every default is read from config, so this function tracks v0.2.0 rather
      than freezing a copy of it: if TRAIL_ENABLED ever flips, the parity fixture
      follows and the parity test keeps testing the CURRENT engine defaults
      instead of a historical snapshot. The keyword arguments exist so the parity
      test can sweep the same axes walkforward.DEFAULT_GRID sweeps
      (walkforward.py:69-73) and prove parity for all of them, not just defaults.

      Branch ids are "range" and "trend". Order is irrelevant to results (the two
      regime sets are disjoint, so at most one branch is active per setup bar and
      no rank_signals tie is possible) but it IS fixed by ordered_branches(), so
      the JSON fully determines behavior.

      The fade branch is included even though config.FADE_ENABLED is False,
      because A7 puts that check inside the detector at CALL TIME: including the
      branch is what proves the kill switch still suppresses identically through
      the graph path.
      """
  ```
  The graph it returns: `data = NodeSpec("ohlcv", "data.ohlcv")`;
  `regime = RegimeGate(adx_trend_threshold=..., atr_extreme_percentile=...)`;
  `trigger = TriggerSpec()`; `filters = ()`; two branches per the mapping table above; and
  `meta = {"provenance": "v0.2.0 Phases 1-7, see .claude/PRPs/reports/KNOWN-LIMITATIONS.md",
  "gate": "FAILING — 2 of 5 conditions (KNOWN-LIMITATIONS §1)"}`. That last note matters: this
  graph is the *parity* reference, not a recommendation, and anyone who finds the JSON should learn
  that from the file.
- **MIRROR**: `signals/donchian.py:1-45` for reproducing stated assumptions in a module docstring.
- **GOTCHA**: (a) `plugins/__init__.py` importing `framework.graph` is fine, but it must **not**
  import the sibling plug-in modules — `load_all()` does that, and an eager import here would make
  a broken plug-in break `build_v020_graph` too. `build_v020_graph` only names keys as strings.
  (b) Phase 3 creates `plugins/policies/__init__.py`; **Phase 4 must not re-create it**. Phase 4
  creates `plugins/{confirmations,filters,reviewers,mutators}/__init__.py` as it needs them.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot.framework import registry, graph; from trading_bot.plugins import build_v020_graph; registry.load_all(); g=build_v020_graph(); graph.validate(g); print(graph.short_hash(g), len(g.branches))"`
  → a 12-char hash and `2`.

### Task 10: `plugins/data/ohlcv.py` — the `DataSource`
- **ACTION**: Create `src/trading_bot/plugins/data/ohlcv.py`.
- **IMPLEMENT**:
  ```python
  @register("data", name="ohlcv",
            params={"timeframes": ParamSpec(kind="choice", default="tiers",
                                            choices=("tiers", "all"),
                                            doc="Which stored timeframes to expose")},
            rationale="The stored OHLCV series is the only data v0.3.0 has. "
                      "Wrapping storage.load_candles behind DataSource is what "
                      "lets a later phase add a second source (funding, "
                      "sentiment) without touching the executor.",
            timeframes=())
  def ohlcv(conn, *, timeframes="tiers"):
      """Factory returning an OhlcvSource bound to `conn`."""
      return OhlcvSource(conn, expose=timeframes)


  class OhlcvSource:
      """DataSource over data/ohlcv.db.

      frame() returns EVERY stored bar for (symbol, timeframe) within
      [start_ms, end_ms] — both bounds INCLUSIVE, matching
      storage.load_candles' contract (storage.py:216-217). Truncation to "closed
      by now_ms" is EvalContext's job, not this class's: a DataSource that also
      truncated would give two places to get the closed-bar rule wrong.

      engine._assert_interval runs on every loaded frame — the same function with
      the same error text — so a series backfilled under the wrong timeframe key
      raises here rather than silently voiding every no-lookahead guarantee
      downstream (engine.py:185-222).
      """
  ```
  `frame(symbol, timeframe, *, start_ms=None, end_ms=None)` builds the DataFrame exactly as
  `engine._df` does (`engine.py:177-182`: columns `ts,open,high,low,close,volume`, `ts` cast to
  `int`, `set_index("ts")`), calls `_assert_interval(df, timeframe, symbol, role)` where `role` is
  derived from which config tier the timeframe matches (`"regime"`/`"setup"`/`"trigger"`, else the
  timeframe name), and memoizes per `(symbol, timeframe, start_ms, end_ms)` for the session's
  lifetime. `timeframes()` returns the three config tiers for `"tiers"`, or every key of
  `storage.TIMEFRAME_MS` present in the DB for `"all"`.
- **MIRROR**: `engine.py:177-182` (`_df`), `storage.py:205-244` (`load_candles`).
- **GOTCHA**: (a) `run_graph_backtest` passes `start_ms`/`end_ms` to bound the **loop**, not the
  load: `engine.run_backtest` loads the *whole* series and then bounds `start`/`end` inside the loop
  (`engine.py:265-267` vs `308-309`), which matters because indicator warmup needs bars *before*
  `start_ms`. Loading `[start_ms, end_ms]` instead would silently truncate warmup and change every
  early candidate. **The session must load unbounded and bound the loop.** This is the single
  easiest way to fail parity by 5-10 trades. (b) `conn` is captured, never opened here — the CLI
  owns connection lifecycle (`cli.py:149`/`168`) and `storage._db_lock` serializes access.
- **VALIDATE**: covered by `test_framework_parity.py::test_assert_interval_enforced`.

### Task 11: `plugins/detectors/donchian.py`
- **ACTION**: Create `src/trading_bot/plugins/detectors/donchian.py`.
- **IMPLEMENT**:
  ```python
  @register(
      "detector", name="donchian-breakout",
      params={
          "entry_period": ParamSpec(kind="int", default=config.DONCHIAN_ENTRY_PERIOD,
                                    bounds=(5, 200), doc="Entry/exit channel lookback (bars)"),
          "trend_period": ParamSpec(kind="int", default=config.DONCHIAN_TREND_PERIOD,
                                    bounds=(10, 400), doc="Mid-line trend-filter lookback (bars)"),
          "adx_period": ParamSpec(kind="int", default=config.ADX_PERIOD,
                                  bounds=(5, 50), doc="Wilder ADX period"),
          "adx_min": ParamSpec(kind="float", default=config.ADX_TREND_THRESHOLD,
                               bounds=(0.0, 60.0), doc="Minimum setup-tier ADX"),
          "lookback_bars": ParamSpec(kind="int", default=config.PATTERN_LOOKBACK_BARS,
                                     bounds=(60, 500),
                                     doc="Setup bars the detector may see"),
      },
      rationale="Canonical Donchian channel breakout (Donchian 1960s, Turtle "
                "lineage), ADX-confirmed on the setup tier and filtered by the "
                "55-bar mid-line. The most-replicated directional edge in "
                "systematic trading, and v0.2.0's only live signal method.",
      timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,), tier=2,
  )
  def donchian_breakout(ctx, *, entry_period, trend_period, adx_period, adx_min, lookback_bars):
      """Wrap signals.donchian.detect_donchian_setups behind the Detector contract.

      Zero logic of its own: the window comes from ctx, the detection from the
      v0.2.0 function, and the adaptation from contracts.event_from_candidate.
      That is deliberate — a reimplementation here would be a second Donchian
      detector to keep in sync, and parity would be testing this file rather than
      the migration.

      20 and 55 remain CANONICAL. They are exposed as ParamSpecs because Phase 6
      needs legal bounds and Phase 7 needs a control, NOT because this phase
      sweeps them; the v0.2.0 note that they "must never appear in a walk-forward
      grid" (config.py:105-111) is a decision about THIS phase's grid, and
      spending them later is a logged degree of freedom.
      """
      window = ctx.window(config.SIGNAL_PATTERN_TIMEFRAME, lookback_bars)
      return [
          contracts.event_from_candidate(c, meta={"channel_width": c.target_height})
          for c in detect_donchian_setups(
              window, entry_period=entry_period, trend_period=trend_period,
              adx_period=adx_period, adx_min=adx_min,
          )
      ]
  ```
- **MIRROR**: `signals/donchian.py:65-140` (the function being wrapped) and
  `bollinger.py:19-52` (keyword-only params).
- **GOTCHA**: (a) `ctx.window(setup_tf, lookback_bars)` must equal `engine.py:339`'s
  `df_setup.iloc[max(0, h_idx + 1 - config.PATTERN_LOOKBACK_BARS) : h_idx + 1]` **exactly** — same
  bar count, same last bar. Wilder ADX is recursive from the window start, so a window one bar
  longer changes `adx_now` and can flip the `adx_min` gate: an off-by-one here shows up as a handful
  of extra or missing trades over full history and nothing else. (b) `detect_donchian_setups` sets
  `end_ts = index[i] + interval` (the setup bar's **close**, `donchian.py:127-138`) — do not
  "normalize" `end_ts` in the adapter; `check_breakout`'s `ts < end_ts` skip depends on it. (c) Do
  not read the timeframe from `ctx.tiers[1]` — read `config.SIGNAL_PATTERN_TIMEFRAME`, because the
  wrapped function does, and the two must not be able to disagree.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot.framework import registry; registry.load_all(); s=registry.get('detector.donchian-breakout'); print(s.tier, s.defaults())"`.

### Task 12: `plugins/detectors/bollinger_fade.py`
- **ACTION**: Create `src/trading_bot/plugins/detectors/bollinger_fade.py`.
- **IMPLEMENT**: `@register("detector", name="bollinger-fade", ...)` with params `period`
  (`config.BB_PERIOD`, 5-100), `num_std` (`config.BB_STD`, 0.5-4.0), `max_age_bars`
  (`config.FADE_STRETCH_MAX_AGE_BARS`, 1-48), `lookback_bars`; `tier=2`; rationale naming the
  Phase 6 DROP verdict so nobody reads its presence as an endorsement. The function:
  ```python
  def bollinger_fade(ctx, *, period, num_std, max_age_bars, lookback_bars):
      """Wrap signals.meanrev.detect_fade_setups behind the Detector contract.

      A7 — config.FADE_ENABLED IS READ HERE, AT CALL TIME, and returns [] when
      false. This is the faithful port of engine.py:343 and scan.py:58: the kill
      switch must suppress live and backtest identically, and putting the check
      in the detector rather than in the executor means it keeps working through
      ANY executor. framework.context's candidate memo carries
      bool(config.FADE_ENABLED) in its key for the reason engine.py:322-327
      states — a live flip must not be served a cached pre-flip candidate list.

      The sleeve is DROPPED (config.FADE_ENABLED = False, Phase 6 verdict: pooled
      expectancy -0.3883% on n=297, 0 of 3 symbols positive, cost ratio above the
      0.10 ceiling on all three). It is migrated anyway, and kept tested, so the
      decision stays reversible and a future re-test costs nothing.
      """
      if not config.FADE_ENABLED:
          return []
      window = ctx.window(config.SIGNAL_PATTERN_TIMEFRAME, lookback_bars)
      out = []
      for c in detect_fade_setups(window, period=period, num_std=num_std,
                                  max_age_bars=max_age_bars):
          out.append(DetectedEvent(
              kind=FADE_KIND,
              direction=c.direction,
              level=c.trigger_level,                              # meanrev.py:169
              target_height=abs(c.target - c.trigger_level),      # meanrev.py:170
              start_ts=c.start_ts,
              end_ts=c.end_ts,      # the stretch bar's ts, NOT +interval — see GOTCHA
              meta={"stop_level": c.stop_level, "target": c.target,
                    "trigger_level": c.trigger_level},
          ))
      return out
  ```
- **MIRROR**: `meanrev.py:159-173` (`_to_trigger_candidate` — the adapter this reproduces field for
  field) and `meanrev.py:65-156`.
- **GOTCHA**: (a) **`end_ts` asymmetry.** The Donchian detector's `end_ts` is the setup bar's
  *close* (`ts + interval`); the fade's is the most recent stretch bar's *ts* (`meanrev.py:152`,
  `_to_trigger_candidate` passing `candidate.end_ts` through). Do **not** "make them consistent" —
  `check_breakout`'s `ts < end_ts` skip means changing either one changes which trigger bars are
  eligible, and the fade's looser rule is v0.2.0's measured behavior. (b) `meta` must carry
  `stop_level` and `target`, because `build_fade_signal` needs the `FadeCandidate`'s structural
  stop and mean target and the six fixed `DetectedEvent` fields cannot express them. (c) The
  detector is called *before* the trigger fires, so `meta` here has none of the trigger facts —
  those are stamped in by the executor (Task 7 step 4).
- **VALIDATE**: `test_framework_parity.py::TestFadeEnabled`.

### Task 13: `plugins/detectors/legacy_patterns.py`
- **ACTION**: Create `src/trading_bot/plugins/detectors/legacy_patterns.py`.
- **IMPLEMENT**: `@register("detector", name="legacy-patterns", ...)`, `tier=2`, wrapping
  `find_pivots` + `detect_patterns`. Params: `pivot_span` (`config.PIVOT_SPAN`, 2-8),
  `max_age_bars` (`config.PATTERN_MAX_AGE_BARS`, 2-60), `lookback_bars`. Rationale must state the
  measured verdict, not a hope: "H&S / triangle / flag geometry, retired from v0.2.0's dispatch in
  Phase 5 because triangle and flag were ~98% of trade volume and lost on all three symbols.
  Migrated to keep the reference implementation reachable and to give Phase 8 a baseline to
  improve on; it is NOT in the v0.2.0 parity graph."
  ```python
  def legacy_patterns(ctx, *, pivot_span, max_age_bars, lookback_bars):
      """Wrap signals.pivots.find_pivots + signals.patterns.detect_patterns.

      This is where pivots.py lands behind the contracts: `pivot_span` is its one
      parameter, and pivot confirmation needs no extra guard here because
      ctx.window() already ends at the last CLOSED setup bar, so find_pivots can
      only emit pivots with a full span of closed bars on both sides
      (pivots.py:7-13).

      SCOPE NOTE: the ~14 geometry tolerances (HS_SHOULDER_TOLERANCE,
      TRIANGLE_MIN_CONVERGENCE, FLAG_POLE_MIN_PCT, ...) are NOT exposed as
      ParamSpecs. patterns.py's private detectors read them from config directly,
      so exposing them means editing a v0.2.0 module this phase must not fork.
      Phase 8 owns re-authoring these detectors as first-class plug-ins with real
      ParamSpecs; until then this plug-in honestly declares three parameters
      rather than pretending to declare seventeen.
      """
  ```
- **MIRROR**: `patterns.py:79-120`, `pivots.py:43-60`, `setup.py:277-278` (the retired call site).
- **GOTCHA**: `detect_patterns` returns at most one candidate per `(kind, direction)` and orders
  them by `(kind, direction)` (`patterns.py:96-101`); it can return **several** events at once,
  unlike the Donchian detector's 0-or-1. The executor must handle a multi-event branch — that is
  what `rank_signals` is for, and `test_framework_parity.py::TestTieBreak` exercises it with this
  detector rather than inventing a stub.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot.framework import registry; registry.load_all(); print(sorted(registry.by_kind('detector')))"`
  → `['bollinger-fade', 'donchian-breakout', 'legacy-patterns']`.

### Task 14: `plugins/policies/legacy_signal.py`
- **ACTION**: Create `src/trading_bot/plugins/policies/legacy_signal.py` registering **two**
  policies.
- **IMPLEMENT**: Module docstring reproducing A8. Then:
  ```python
  @register(
      "policy", name="atr-stop-measured-move",
      params={
          "atr_multiple": ParamSpec(kind="float", default=config.ATR_STOP_MULTIPLE,
                                    bounds=(0.5, 6.0), doc="Stop distance in ATRs"),
          "atr_period": ParamSpec(kind="int", default=config.ATR_STOP_PERIOD,
                                  bounds=(5, 50), doc="Wilder ATR period (setup tier)"),
          "rr_floor": ParamSpec(kind="float", default=config.RR_FLOOR,
                                bounds=(1.0, 5.0), doc="Minimum gross reward:risk"),
      },
      rationale="v0.2.0's entry/stop/target rule: stop k*ATR(setup tier) from "
                "entry (volatility-derived, never a fixed percentage), target at "
                "the measured move (level +/- target_height), screened on gross "
                "reward:risk. Wraps setup.build_signal unchanged so the migrated "
                "path cannot disagree with the live one.",
      timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
  )
  def atr_stop_measured_move(ctx, event, *, atr_multiple, atr_period, rr_floor):
      """Delegate wholly to signals.setup.build_signal, then adapt.

      A8 — the R:R floor stays INSIDE build_signal (setup.py:154-156, on the
      GROSS ratio) rather than moving to a Filter node, because moving it would
      change which setups survive and break parity. Phase 4's
      filters/rr_after_costs.py adds the >=1:2-after-costs screen as a NEW filter
      against RR_TARGET_MIN, leaving this legacy floor measured and unchanged.

      Returns None on rejection — warmup NaN ATR, non-positive risk or reward, or
      rr below the floor — exactly as build_signal does. A rejection is not an
      error and must not raise.
      """
      atr = float(ctx.series(config.SIGNAL_PATTERN_TIMEFRAME, "atr",
                             _atr_factory, period=atr_period)[-1])
      sig = build_signal(ctx.symbol, contracts.candidate_from_event(event),
                         _breakout_event_from_meta(event), atr,
                         atr_multiple=atr_multiple, rr_floor=rr_floor)
      return None if sig is None else contracts.plan_from_signal(sig, source=event.kind)
  ```
  And `policy.fade-structural-stop` with a single `rr_floor` param, wrapping
  `meanrev.build_fade_signal` — reconstructing a `FadeCandidate` from
  `event.meta["stop_level"]`, `event.meta["target"]`, `event.level`, `event.start_ts`,
  `event.end_ts`, with a docstring noting that this policy's floor is applied to
  `risk.atr_stop.net_rr`, the **cost-adjusted** ratio, not the gross one
  (`meanrev.py:186-198` explains why: its structural stop has no ATR floor) — and that the two
  policies using two different R:R definitions is v0.2.0's measured behavior, deliberately
  preserved.
  `_breakout_event_from_meta(event) -> BreakoutEvent` rebuilds the trigger event the executor
  stamped into `meta`.
- **MIRROR**: `setup.py:73-171`, `meanrev.py:176-249`, `setup.py:131-136` (the NaN guard).
- **GOTCHA**: (a) `build_signal` takes `atr_value` **positionally** (`setup.py:76`) and rejects
  NaN/non-positive itself, so pass it through without a caller-side check — mirroring
  `engine.py:500-503`'s `if event and atr_value > 0` is *also* required, since the engine skips the
  call entirely when `atr_value <= 0`; reproduce **the engine's** guard in the executor, and keep
  the policy's own guard as defence in depth. (b) The `atr` the policy computes must be the same
  number `engine.py:492-494` uses: `atr_setup_vals[h_idx]`, i.e. the ATR at the **setup** bar, not
  at the trigger bar. Because the policy runs with a context bound to the trigger close, `[-1]` of
  a setup-tier series is the last setup bar closed by that trigger bar — which is `h_idx` **only
  when** `h_idx` was selected on the trigger bar's *open*. It is (MEDIUM-2, `engine.py:481`), and
  since setup closes land on trigger boundaries (`test_tiers.py:44-49`) the two agree. Verify with
  `test_framework_parity.py::TestAtrAtEntry`, which asserts the policy's ATR equals
  `atr_setup_vals[h_idx]` on every entry — do not assume it. (c) Two registrations in one module is
  fine and keeps the two halves of v0.2.0's R:R story in one file where the asymmetry is visible.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot.framework import registry; registry.load_all(); print(sorted(registry.by_kind('policy')))"`
  → `['atr-stop-measured-move', 'fade-structural-stop']`.

### Task 15: `walkforward.py` — the one keyword
- **ACTION**: Edit `src/trading_bot/backtest/walkforward.py` at **three disjoint sites**. Nothing
  else in this file is Phase 3's.
- **IMPLEMENT**:
  1. Imports and a module-level constant:
     ```python
     from trading_bot.framework.execute import run_graph_backtest
     from trading_bot.framework.graph import StrategyGraph

     # Grid axes that are run_backtest KWARGS rather than BacktestParams fields,
     # and therefore still meaningful when a graph carries the strategy.
     _RUN_LEVEL_AXES = frozenset({"max_hold_bars"})
     ```
  2. A dispatch helper beside `_pooled_expectancy`, and one changed line inside it:
     ```python
     def _run_one(conn, symbol, *, start, end, params, extra_kwargs, strategy):
         """One symbol's trades for a fold — engine path or graph path.

         strategy=None keeps the legacy engine.run_backtest path bit-for-bit, so
         all 286 pre-existing tests are unaffected (contract §5).
         """
         if strategy is None:
             return run_backtest(conn, symbol, start_ms=start, end_ms=end,
                                 params=params, **extra_kwargs)
         return run_graph_backtest(conn, strategy, symbol, start_ms=start,
                                   end_ms=end, **extra_kwargs)
     ```
     `_pooled_expectancy` gains a `strategy=None` keyword and calls `_run_one`; the OOS loop
     (`walkforward.py:386-394`) does the same.
  3. `walk_forward_pooled` gains `strategy: StrategyGraph | None = None` as the **last**
     keyword-only parameter, documented in the Args block, plus this guard immediately after
     `grid` is resolved:
     ```python
     if strategy is not None:
         unsupported = sorted(set(grid) - _RUN_LEVEL_AXES)
         if unsupported:
             raise ValueError(
                 f"a graph strategy carries its own parameters, so grid axes "
                 f"{unsupported} would be silently ignored — producing "
                 f"{len(_combos(grid))} identical trade lists per fold and "
                 f"deflating the DSR for configurations that were never "
                 f"distinct (the exact pathology repaired as HIGH-3, see "
                 f"DEFAULT_GRID's note). Pass grid={{'max_hold_bars': (...)}} or "
                 f"sweep graphs by mutation (Phase 6), not by grid."
             )
     ```
- **MIRROR**: `walkforward.py:165-177` (`_pooled_expectancy`'s shape),
  `walkforward.py:53-68` (the HIGH-3 note the guard's message cites).
- **IMPORTS**: the two above. No cycle: nothing in `framework` imports `walkforward`.
- **GOTCHA**: (a) **Merge boundary with Phase 1.** Phase 1 owns `_evaluate_gate`,
  `WalkForwardResult`, `GATE_CONDITIONS`, and the result construction; Phase 3 owns the signature's
  last parameter, `_run_one`, and the two call sites. No shared line. If Phase 1's diff already
  landed, apply on top; do not reconcile by rewriting either. (b) `strategy` must be the **last**
  keyword-only parameter so any caller using keywords is unaffected. (c) Do **not** touch
  `DEFAULT_GRID`, `_CONFIG_DEFAULT_ATTR`, `_default_combo`, or `_combo_to_kwargs`: a graph run still
  builds a `BacktestParams` from the combo and simply ignores it, which is harmless and keeps the
  fold bookkeeping (`FoldResult.best_params`) unchanged. (d) The refusal is deliberate: silently
  ignoring `trail_enabled` in a graph run is exactly how a gate becomes theatre.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_backtest.py -q` → unchanged pass count
  (`TestWalkForwardPooled` at `test_backtest.py:495` must be untouched).

### Task 16: `cli.py` — `plugins`, `graph-validate`, `walkforward --graph`
- **ACTION**: Edit `src/trading_bot/cli.py`: two new parsers after the `walkforward` parser
  (`cli.py:129-144`), one new argument on it, two dispatch branches, two handlers.
- **IMPLEMENT**:
  1. Parsers:
     ```python
     plugins_parser = subparsers.add_parser(
         "plugins", help="List registered framework plug-ins")
     plugins_parser.add_argument("--kind", choices=registry.KINDS,
                                 help="Only this kind; default all")
     gv_parser = subparsers.add_parser(
         "graph-validate", help="Validate serialized strategy graphs")
     gv_parser.add_argument("path", nargs="+",
                            help="Path(s) to <name>.strategy.json")
     wf_parser.add_argument(
         "--graph", help="Run the walk-forward through a serialized strategy "
                         "graph instead of the legacy engine path")
     ```
  2. `_plugins_command(*, kind=None) -> int` — `load_all()`, then a table in
     `_gap_report_command`'s style (`cli.py:237-266`): header, dashes, one row per key sorted by
     `(kind, name)` with columns `key | tier | timeframes | params | dof | rationale`, where `dof`
     is `combo_count()`. Print a footer: total plug-ins, total by kind, and summed `dof` with the
     line *"degrees of freedom a graph could expose; the DSR is charged for evaluations actually
     performed, not for this number (contract §4)."* Returns 1 if `load_all()` raised (message to
     stderr), else 0.
  3. `_graph_validate_command(paths) -> int` — for each path: `graph.load()`, then print
     `OK  <name>  <short_hash>  branches=N  nodes=M  <path>`; on `GraphError` print
     `FAIL <path>` followed by the multi-problem message indented. Returns 1 if any failed.
  4. Dispatch: `plugins` and `graph-validate` need no DB connection — do **not** call `connect()`
     for them (a read-only listing that creates `data/ohlcv.db` as a side effect is a trap).
     `walkforward` gains: if `args.graph`, `strategy = graph.load(args.graph)` and pass
     `strategy=strategy` plus `grid={"max_hold_bars": config.MAX_HOLD_BARS_TRIGGER,}`-shaped
     single-axis grid through `_walkforward_command`.
  5. Fix the stale help string at `cli.py:97-98` — `signal`'s help still says "trending: pattern
     breakout", retired in v0.2.0 Phase 5. One line, and the CLI help stops lying.
- **MIRROR**: `cli.py:64-77` (parser), `cli.py:176-181` (dispatch), `cli.py:237-266` (table style
  and `-> int` exit codes).
- **IMPORTS**: `from trading_bot.framework import graph, registry`. Note this makes
  `python -m trading_bot.cli --help` import the framework; measure the delta and report it (expect
  <100 ms).
- **GOTCHA**: (a) `_plugins_command` must not truncate `rationale` to fit a column — the whole point
  of the mandatory rationale is that it is readable. Print it on a continuation line indented under
  the key. (b) `--graph` on `walkforward` will hit the Task 15 guard unless the grid is reduced;
  `_walkforward_command` must pass a single-axis grid when a graph is given, and say so in its
  output header so the operator knows the sweep changed shape.
- **VALIDATE**:
  ```bash
  .venv/bin/python -m trading_bot.cli plugins
  .venv/bin/python -m trading_bot.cli plugins --kind detector
  .venv/bin/python -m trading_bot.cli graph-validate data/strategies/donchian-v020.strategy.json
  echo "exit=$?"
  ```
  → six plug-ins listed with non-empty rationales; `graph-validate` prints `OK` and exits 0.

### Task 17: `tests/test_framework_contracts.py`
- **ACTION**: Create the file. Classes: `TestParamSpec`, `TestAdapters`, `TestCallableShape`,
  `TestEvalContextSurface`, `TestNoLookahead`, `TestSeriesCache`, `TestAssertTrailingOnly`.
- **IMPLEMENT**: Header copies the tier constants and the autouse cache fixture from
  `tests/test_backtest.py:23-47`, extended to clear the framework caches too:
  ```python
  @pytest.fixture(autouse=True)
  def _isolate_caches():
      engine.clear_caches(); framework_context.clear_caches()
      yield
      engine.clear_caches(); framework_context.clear_caches()
  ```
  - `TestParamSpec`: one test per `PARAM_KINDS` entry for legal/illegal values; `bool` rejected as an
    `int` (`is_legal(True) is False` for `kind="int"`); illegal `default` raises at construction;
    `int`/`float` without `bounds` raise; `choice` without `choices` raises; `bool` with `bounds`
    raises; empty `doc` raises; a `list` for `bounds`/`choices` raises; `clamp` on
    below/above/inside/NaN/`choice`-miss; `step` snapping; `check()`'s message contains the `where`
    string.
  - `TestAdapters`: `event_from_candidate`/`candidate_from_event` round-trip on all six fields;
    `meta` dropped in the reverse direction and documented as such;
    `plan_to_signal(plan_from_signal(sig, source=sig.pattern), event) == sig` field-for-field for
    both a `build_signal` output and a `build_fade_signal` output — the lossless-round-trip pin;
    `plan_to_signal` requires the event (calling it without raises `TypeError`);
    `PositionPlan.source` lands on `Signal.pattern`.
  - `TestCallableShape`: a correct detector passes; `def d(df, ctx)` raises naming the key;
    a policy missing `event` raises; a plug-in with a positional-only extra param raises.
  - `TestEvalContextSurface`: the context exposes exactly the documented members —
    `set(public attrs) == EXPECTED` — and **has no** `conn`, `_frames`, `time`, `rng`, or any
    attribute returning an untruncated frame. This is the test that keeps the surface from growing a
    lookahead hole by accident.
  - `TestNoLookahead` (the load-bearing class), on a hand-built 3-tier synthetic where every bar's
    close is its index so an off-by-one is visible by inspection:
    | Test | Assertion |
    |---|---|
    | `test_frame_excludes_the_forming_bar` | for every `now_ms`, `frame(tf).index[-1] + interval <= now_ms` |
    | `test_frame_includes_a_bar_closing_exactly_at_now` | a bar with `ts + interval == now_ms` **is** included (matches `setup.py:209-210`) |
    | `test_bar_index_matches_searchsorted_rule` | equals `searchsorted(close, now, "right") - 1` for 50 sampled `now_ms` |
    | `test_window_equals_engine_slice` | `window(setup_tf, 180)` equals `df.iloc[max(0,k+1-180):k+1]` |
    | `test_regime_matches_engine_regime_at` | equals `engine`'s `regime_at` on the same labels/closes, including `"uncertain"` for `k < 0` |
    | `test_returned_array_is_read_only` | `series(...)` result raises on assignment |
    | `test_now_ms_must_not_go_backwards_per_role` | `ContractError` naming the role |
    | `test_roles_advance_independently` | a setup context trailing the previous trigger context is legal |
    | `test_pivots_confirmed_only_with_span_closed_bars` | `find_pivots(ctx.window(...))` emits no pivot within `PIVOT_SPAN` of the last closed bar |
    | `test_assert_interval_runs_on_load` | a 15m series seeded under `"1h"` raises `ValueError` with engine's exact message |
  - `TestSeriesCache`: identical calls hit (`cache_stats()["hits"] == 1`); a different `period`
    misses (**the engine's key omits `period`; ours must not**); two fixtures with equal bar counts
    and timestamps but different prices do **not** collide (mirroring
    `test_backtest.py::TestIndicatorMemo`); `FRAMEWORK_CACHE_ENABLED = False` returns equal values
    with zero hits.
  - `TestAssertTrailingOnly`: a trailing factory passes; `lambda df: df["close"].shift(-1)` raises
    `ContractError` naming the first offending index; a centred rolling mean raises.
- **MIRROR**: `tests/test_backtest.py:23-47`, `tests/test_wilder.py::TestLeadingNaNs` (warmup
  assertions), `tests/test_tiers.py` (configuration-invariant genre).
- **GOTCHA**: Derive every timeframe from config (`SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME`),
  never `"4h"` — contract §8. Use `caplog.at_level("WARNING", logger="trading_bot")` for any log
  assertion, matching `test_signals.py`.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_framework_contracts.py -v`.

### Task 18: `tests/test_framework_registry.py`
- **ACTION**: Create the file. Classes: `TestRegisterValidation`, `TestLookup`, `TestLoadAll`,
  `TestPluginSpec`, `TestRegisteredPluginsAreWellFormed`.
- **IMPLEMENT**: Every registration test runs inside `with registry.temporary_registry():`.
  - `TestRegisterValidation`: duplicate key raises **and the message names the claiming module**
    (`match=` on the module name); unknown kind; `camelCase`, `snake_case`, `Leading-Caps` and
    `trailing-` names all raise with a message showing the legal form; empty/whitespace rationale
    raises; a non-`ParamSpec` value in `params` raises; an unknown timeframe raises; `tier=5`
    raises; a valid registration returns the function **unchanged** (`fn is decorated`) and it stays
    directly callable.
  - `TestLookup`: `get` on a missing key raises `RegistryError` with close-match suggestions;
    `by_kind` is sorted by name; `by_kind` on an unknown kind raises.
  - `TestLoadAll`: returns all six production keys; **idempotent** (two calls, no raise, same dict);
    a monkeypatched `importlib.import_module` raising for one plug-in module makes `load_all` raise
    `RegistryError` naming that module — pinning "import errors are fatal, never skipped"
    (`scripts/bruteforce/registry.py:164-167`); `load_all` on a package with a `_private.py` skips
    it.
  - `TestPluginSpec`: `defaults()`; `resolve(None)` == `defaults()`; `resolve` with an unknown key
    raises listing the legal keys; `resolve` with an out-of-bounds value raises naming the
    parameter; `resolve` coerces `int`-valued floats where legal; `combo_count()` on a known spec.
  - `TestRegisteredPluginsAreWellFormed` — the meta-test every later phase inherits for free:
    after `load_all()`, for **every** spec in `REGISTRY`: `rationale.strip()` is non-empty and
    ≥ 40 characters; `key == f"{kind}.{name}"`; `_NAME_RE` matches; every param value is a
    `ParamSpec` whose `default` is legal and whose `doc` is non-empty; `tier is None or in TIERS`;
    `check_callable_shape` passes. Phase 4's and Phase 8's plug-ins are validated by this test the
    moment they land, with no new test code.
- **MIRROR**: `scripts/bruteforce/registry.py:120-125` (the docstring enumerating exactly what
  raises — the test list is that list).
- **GOTCHA**: Never `REGISTRY.clear()` — `importlib` caching means the decorators will not re-run
  and every later test in the session sees an empty registry (Task 4 GOTCHA c). Use
  `temporary_registry()`.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_framework_registry.py -v`.

### Task 19: `tests/test_framework_graph.py`
- **ACTION**: Create the file. Classes: `TestRoundTrip`, `TestCanonicalHash`, `TestValidation`,
  `TestSaveLoad`.
- **IMPLEMENT**:
  - `TestRoundTrip`: `from_dict(to_dict(g)) == g` for the v0.2.0 graph and for a
    maximal graph (three branches, confirmations, filters, non-default exits, populated `meta`);
    JSON `dumps`/`loads` in between; an unknown top-level key raises; an unknown key inside
    `exits`/`regime`/`trigger`/a node raises (Task 6 GOTCHA b — the `"trail"` vs `"trail_enabled"`
    trap gets its own named test); a `list` where a `tuple` is required raises.
  - `TestCanonicalHash`, the class Phase 5 and Phase 6 depend on:
    | Test | Assertion |
    |---|---|
    | `test_hash_is_stable_across_calls` | equal on repeated calls |
    | `test_pinned_digest` | `graph_hash(build_v020_graph())` equals a **literal 64-char digest** pinned in the test |
    | `test_branch_order_does_not_change_hash` | reversed `branches` → same hash |
    | `test_confirmation_and_filter_order_do_not_change_hash` | same |
    | `test_regime_list_order_does_not_change_hash` | `("ranging","trending")` vs reversed → same |
    | `test_name_does_not_change_hash` | rename → same |
    | `test_meta_does_not_change_hash` | added note → same |
    | `test_explicit_default_equals_omitted` | `params={"entry_period": 20}` vs `{}` → same |
    | `test_int_and_float_forms_agree` | `20` vs `20.0` → same |
    | `test_param_change_changes_hash` | `entry_period=21` → different |
    | `test_exit_flag_change_changes_hash` | `trail_enabled=True` → different |
    | `test_regime_threshold_change_changes_hash` | different |
    | `test_schema_version_change_changes_hash` | different |
    | `test_resolved_hashes_equal` | `graph_hash(g) == graph_hash(g.resolved())` |
    | `test_ordered_branches_matches_canonical_order` | `[b.id for b in g.ordered_branches()]` equals the branch order in `canonical_dict` — **the A5 invariant that ties hash equality to behavioural equality** |
    | `test_nan_param_is_unhashable` | `GraphError`/`ValueError`, not a digest |
  - `TestValidation`: one test per numbered rule in Task 6 step 3, each asserting the message
    mentions the offending value — fourteen tests, named
    `test_rejects_<rule>`. `test_rejects_channel_period_mismatch` and
    `test_rejects_trail_atr_period_mismatch` carry docstrings explaining the silent-divergence they
    prevent. `test_unknown_key_suggests_close_matches` asserts
    `"donchian-breakout"` appears in the message for `"detector.donchian-breakou"`.
  - `TestSaveLoad`: `save` into `tmp_path` via monkeypatched `config.STRATEGY_DIR`; a bare name
    resolves to `<STRATEGY_DIR>/<name>.strategy.json`; the file ends with a newline and is
    `sort_keys`-stable (`save` twice → identical bytes); `save` validates first (an invalid graph
    writes **no** file — assert the path does not exist); `load` of a corrupt JSON raises
    `GraphError`, not `json.JSONDecodeError`.
- **MIRROR**: `tests/test_tiers.py` (configuration-invariant genre); `tests/test_backtest.py`'s
  `dataclasses` import for `dataclasses.replace` in the mutation tests.
- **GOTCHA**: The pinned digest is computed once by the implementer
  (`.venv/bin/python -c "from trading_bot.framework import registry, graph; from trading_bot.plugins import build_v020_graph; registry.load_all(); print(graph.graph_hash(build_v020_graph()))"`)
  and pasted in with a comment: *"Regenerate deliberately, never casually: this digest changes if any
  plug-in's ParamSpec default changes, and Phase 5's strategy_versions plus Phase 6's trial_ledger
  key on it. A change here means every persisted row referring to this graph now points at a
  different strategy."*
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_framework_graph.py -v`.

### Task 20: `tests/test_framework_parity.py` — the acceptance gate
- **ACTION**: Create the file. This is the phase's gate; write it last and expect it to find bugs in
  Tasks 5-7.
- **IMPLEMENT**:
  1. **Header**: tier constants and the autouse dual-cache fixture as in Task 17; the fixture
     builders `seed`, `donchian_rows`, `donchian_atr_at_entry`, `seed_scenario`, `fade_setup_rows`,
     `seed_fade_scenario`, `patch_trending` **copied** from `tests/test_backtest.py:83-190` (the
     suite has no `conftest.py` and duplicates fixtures across modules by convention — follow it);
     and the `PARITY_EXACT` / `PARITY_CLOSE` / `assert_trades_match` helper defined verbatim in
     "The parity assertion, defined once" above, **including its comment about the Phase 4
     fields**.
  2. `class TestSyntheticParity` — on in-memory SQLite fixtures, fast, always runs:
     - `test_donchian_scenario_parity`, `test_fade_scenario_parity` (with
       `config.FADE_ENABLED = True` monkeypatched), `test_no_trades_scenario_parity`,
       `test_empty_db_parity` (both return `[]`).
     - `test_parity_across_every_default_grid_combo` — parametrized over the full cartesian product
       of `walkforward.DEFAULT_GRID` (`trail_enabled` × `target_enabled` × `max_hold_bars` = 12
       combos), asserting parity for each. This is what makes the graph safe for
       `walkforward --graph`, and it is the test most likely to catch an `ExitPolicySpec`
       mis-mapping.
     - `test_parity_with_cache_disabled` — `FRAMEWORK_CACHE_ENABLED = False`, bit-identical.
  3. `class TestStoredHistoryParity` — `@pytest.mark.skipif(not Path(config.DB_PATH).exists(), reason="stored OHLCV DB not present")`:
     - `test_parity_one_year_all_symbols` over `config.SYMBOLS` on a one-year slice (bounded
       runtime; full span is a manual validation step below). Report the trade count in the
       assertion message so the phase report can quote a measured number.
  4. The nine **localizing** sub-parity tests, each isolating one row of the parity table so a
     failure names its own cause:
     | Class::test | Isolates |
     |---|---|
     | `TestRegimeLabels::test_labels_match_engine` | `classify_series` call + the `searchsorted` rule |
     | `TestCandidateLists::test_events_match_per_setup_bar` | detector window + regime gate; compares `candidate_from_event(e)` to `engine`'s candidates for every setup bar |
     | `TestMedium2::test_last_trigger_bar_of_a_setup_window_can_trigger` | the MEDIUM-2 repair — constructs a breakout on the final trigger bar of a setup window and asserts **both** paths take it. Docstring names MEDIUM-2 and KNOWN-LIMITATIONS §6 |
     | `TestAtrAtEntry::test_policy_atr_equals_engine_atr_setup_vals` | Task 14 GOTCHA b |
     | `TestExitMix::test_outcome_counts_match` | the generic exit block vs both legacy branches |
     | `TestExitAsymmetry::test_fade_trades_never_trail_or_channel` | the MANDATORY DEVIATION; asserts no fade trade has outcome `"trail"`/`"channel"` even with `trail_enabled=True` |
     | `TestRatchetOrdering::test_trail_cannot_fire_on_the_bar_that_set_its_extreme` | the intra-bar lookahead guard (ported from `test_backtest.py::TestDonchianExits`) |
     | `TestTieBreak::test_multi_event_branch_ranks_by_rank_signals` | uses `detector.legacy-patterns`, which emits several events, and asserts the taken trade is `rank_signals(...)[0]` |
     | `TestCostArithmetic::test_pnl_matches_hand_computed` | A10's duplication: hand-computed `gross - 2*(fee+slip) - funding*hold_days` for a known trade, plus equality against `engine`'s value |
  5. `class TestFadeEnabled` — `test_flag_read_at_call_time`: flip `config.FADE_ENABLED` between two
     `run_graph_backtest` calls **on the same graph object** and assert the candidate set changes
     (A7 + the cache key), then assert parity with `run_backtest` under both settings.
  6. `class TestAssertIntervalEnforced` — seed 15m bars under the `"1h"` key; both paths raise
     `ValueError` with the same message.
  7. `class TestScanDriftGuard` — A6's mitigation: for each label in `classifier.REGIMES`,
     monkeypatch `scan.scan_donchian_signals`/`scan_fade_signals` to sentinels, call
     `scan.scan_symbol`, and assert the sentinel returned corresponds to the detector kind of the
     v0.2.0 graph's branch whose `regimes` contains that label — with `("extreme-volatility",
     "uncertain")` mapping to no branch on both sides. Docstring states plainly that `scan.py` is
     not yet graph-dispatched and that this test is the drift alarm until it is.
  8. `class TestWalkForwardGraphPath` — `walk_forward_pooled(..., strategy=g, grid={"max_hold_bars": (48, 96)})`
     returns a `WalkForwardResult` whose OOS trades equal the engine path's for the same final
     params; and the unsupported-axis guard raises `ValueError` with `DEFAULT_GRID`.
- **MIRROR**: `tests/test_backtest.py:192-468` (`TestRunBacktest`, `TestDonchianExits`,
  `TestFadeEnabledSwitch`) — the scenarios already exist; parity tests re-run them through both
  paths rather than inventing new ones.
- **GOTCHA**: (a) **`assert_trades_match`, never `==`** (A11). (b) Every synthetic test must pass
  `fee_pct=0.0, slippage_pct=0.0, funding_pct_per_day=0.0` **to both paths identically** where exact
  arithmetic is asserted — `test_backtest.py:110,130` already does this. (c) `patch_trending`
  monkeypatches `engine.classify_series`; the graph path calls `classify_series` imported into
  `framework.context`, so the fixture must patch **both** names or the two paths see different
  regimes and every parity test fails for the wrong reason. Patch
  `framework.context.classify_series` alongside `engine.classify_series` — this is the single most
  likely cause of a confusing initial red. (d) Clear **both** caches in the autouse fixture.
  (e) Do not run `walkforward` against the real final holdout anywhere in this file; the
  `TestWalkForwardGraphPath` case uses synthetic fixtures.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_framework_parity.py -v`.

### Task 21: Serialize the parity graph, measure, and record
- **ACTION**: Generate `data/strategies/donchian-v020.strategy.json`, run the full validation suite,
  and record every measured number.
- **IMPLEMENT**:
  ```bash
  cd /Users/ttaa/Documents/Project.nosync/InvestmentBot/trading
  .venv/bin/python - <<'PY'
  from trading_bot.framework import graph, registry
  from trading_bot.plugins import build_v020_graph
  registry.load_all()
  g = build_v020_graph()
  p = graph.save(g)
  print(p, graph.graph_hash(g))
  PY
  git add data/strategies/donchian-v020.strategy.json
  ```
  Then paste the digest into `test_framework_graph.py::test_pinned_digest` and record, in the phase
  report, each of these **measured by the command beside it** (contract §12.5 — measured, never
  derived; this repo has already committed once to that discipline):
  | Number | Command |
  |---|---|
  | Test count before / after | `.venv/bin/python -m pytest --collect-only -q \| tail -1` |
  | Full-suite pass/fail | `.venv/bin/python -m pytest -q` |
  | Parity trade count, BTC full span | the manual validation command below |
  | Graph vs engine wall time, same span | `time` on the two CLI commands |
  | Cache hit rate over a full-span graph run | `framework.context.cache_stats()` |
  | Plug-in count and summed declared DoF | `.venv/bin/python -m trading_bot.cli plugins` footer |
  | The pinned graph hash | the snippet above |
- **GOTCHA**: `data/strategies/` is **committed** (unlike `data/state.db*`, which Phase 1
  `.gitignore`s): the serialized graph is the record of what was tested, it is 2 KB, and Phase 5's
  version registry references its hash. Do not add it to `.gitignore`.
- **VALIDATE**: the full validation block below, all green.

---

## Testing Strategy

### Unit Tests — the eleven that carry the phase's correctness

| Test | Input | Expected | Why it matters |
|---|---|---|---|
| `TestSyntheticParity::test_parity_across_every_default_grid_combo` | 12 `DEFAULT_GRID` combos on the Donchian fixture | `assert_trades_match` for all 12 | The `ExitPolicySpec` mapping is right for every configuration the gate can select, not just defaults |
| `TestStoredHistoryParity::test_parity_one_year_all_symbols` | real OHLCV, 1 year, 3 symbols | identical trade lists | The acceptance gate. Synthetic fixtures cannot produce warmup, gaps, or regime transitions |
| `TestMedium2::test_last_trigger_bar_of_a_setup_window_can_trigger` | breakout on the final trigger bar of a setup window | both paths take it | MEDIUM-2 re-broken = 25% of opportunities silently vanish, and in-sample results *improve* (KNOWN-LIMITATIONS §6). The most dangerous possible regression |
| `TestExitAsymmetry::test_fade_trades_never_trail_or_channel` | fade trade, `trail_enabled=True` | no `"trail"`/`"channel"` outcome | The MANDATORY DEVIATION. Applying trend exits to the fade sleeve silently changes what Phase 6's DROP verdict measured |
| `TestRatchetOrdering::test_trail_cannot_fire_on_the_bar_that_set_its_extreme` | one bar: high far up, low below `high − k·ATR` | no exit on that bar | Intra-bar lookahead; inflates every result |
| `TestNoLookahead::test_frame_excludes_the_forming_bar` | 50 sampled `now_ms` | `index[-1] + interval <= now_ms` ∀ | The whole `EvalContext` premise |
| `TestNoLookahead::test_bar_index_matches_searchsorted_rule` | hand-built closes | equals `searchsorted(close, now, "right") - 1` | One off-by-one here is a lookahead nobody would see in the numbers |
| `TestAssertTrailingOnly::test_rejects_shift_minus_one` | `lambda df: df["close"].shift(-1)` | `ContractError` at the first bad index | Truncation cannot save a non-causal factory; this is the first line of defence |
| `TestCanonicalHash::test_ordered_branches_matches_canonical_order` | reordered branches | canonical order == runtime order | Ties hash equality to behavioural equality; Phase 5/6 keys are built on it |
| `TestCanonicalHash::test_pinned_digest` | `build_v020_graph()` | a literal digest | Proves the hash is stable across processes, which is the only thing that makes a persisted ledger meaningful |
| `TestSeriesCache::test_period_is_in_the_key` | same frame, `period=14` then `21` | a miss, different values | `engine.py:293`'s key omits `period`; ours must not, or a graph asking for ATR(21) is served ATR(14) |

### Edge Cases Checklist
- [x] Empty DB / empty tier → `[]` on both paths
- [x] Fewer bars than warmup → no events, no crash (`donchian.py:97-99`)
- [x] NaN indicator during warmup → no event (`not (x > 0)` idiom, never `x != x`)
- [x] Bar closing exactly at `now_ms` is included; the forming bar never is
- [x] `bar_index == -1` (no closed bar yet) → `regime()` returns `"uncertain"`
- [x] Gapped trigger pair → `check_breakout` skips it (`interval_ms` passed through)
- [x] Interval mismatch between config and stored data → `ValueError`, same message as engine
- [x] Multi-event branch → `rank_signals` tie-break, `is`-identity branch recovery
- [x] Two branches active on one bar → one open trade per symbol
- [x] `FADE_ENABLED` flipped mid-process → candidate set changes, cache not stale
- [x] Trail moved the stop → `Trade.stop` still the **initial** stop
- [x] Unresolved trade at data end → outcome `"end"` at `last_j`, not `len-1`
- [x] Duplicate registry key / empty rationale / bad name casing → raise
- [x] Unknown or out-of-bounds graph parameter → `GraphError` naming it
- [x] Unknown JSON key (`"trail"` for `"trail_enabled"`) → raise, never ignore
- [x] `-0.0`, `20` vs `20.0`, NaN in a hashed parameter
- [x] `load_all()` called twice → idempotent; one broken module → fatal
- [ ] Concurrent access — N/A (single-threaded replay; `storage._db_lock` unchanged)
- [ ] Network failure — N/A (no network in this phase; `fapi.binance.com` is Phase 2's concern)
- [ ] Permission denied — N/A beyond `save`'s `mkdir(parents=True, exist_ok=True)`

---

## Validation Commands

The project has **no linter and no type checker** (`pyproject.toml` declares only `ccxt`, `pandas`,
`apscheduler`, `python-dotenv`, and a `dev` extra of `pytest`; there is no `Makefile`).
"Validation" means `pytest` and `py_compile`. Do not invent a lint or typecheck step. The
interpreter is the repo venv, Python 3.11.6 (Homebrew `python@3.11`) — always invoke it explicitly.

### Static Analysis
```bash
cd /Users/ttaa/Documents/Project.nosync/InvestmentBot/trading
.venv/bin/python -m py_compile \
  src/trading_bot/config.py \
  src/trading_bot/framework/__init__.py \
  src/trading_bot/framework/errors.py \
  src/trading_bot/framework/contracts.py \
  src/trading_bot/framework/registry.py \
  src/trading_bot/framework/context.py \
  src/trading_bot/framework/graph.py \
  src/trading_bot/framework/execute.py \
  src/trading_bot/plugins/__init__.py \
  src/trading_bot/plugins/data/ohlcv.py \
  src/trading_bot/plugins/detectors/donchian.py \
  src/trading_bot/plugins/detectors/bollinger_fade.py \
  src/trading_bot/plugins/detectors/legacy_patterns.py \
  src/trading_bot/plugins/policies/legacy_signal.py \
  src/trading_bot/backtest/walkforward.py \
  src/trading_bot/cli.py
```
EXPECT: zero output.

### Import health (catches the cycle this phase can create)
```bash
.venv/bin/python -c "import trading_bot.framework, trading_bot.plugins; print('ok')"
.venv/bin/python -c "
import trading_bot.backtest.engine, trading_bot.signals.scan, trading_bot.indicators.wilder, sys
assert not [m for m in sys.modules if m.startswith('trading_bot.framework')], \
    'engine/signals/indicators must NOT import framework — that closes the cycle'
print('no reverse dependency')"
```
EXPECT: `ok`, `no reverse dependency`.

### New unit tests
```bash
.venv/bin/python -m pytest tests/test_framework_contracts.py tests/test_framework_registry.py \
  tests/test_framework_graph.py tests/test_framework_parity.py -v
```
EXPECT: all pass. `TestStoredHistoryParity` skips only if `data/ohlcv.db` is absent — on this
machine it is present (117 MB), so it must **run**, not skip. A skip here is a failed phase.

### Full suite — the 286-test baseline
```bash
.venv/bin/python -m pytest -q
```
EXPECT: **286 pre-existing tests still pass**, plus the new ones. Record both numbers.
`test_backtest.py`, `test_signals.py`, `test_meanrev.py`, `test_donchian.py`, `test_tiers.py`,
`test_equity.py`, `test_cli.py` are all untouched by this phase and any change in their results is
a regression, not an update.

### Zero-engine-edit verification (contract §12.4, the PRD's success metric)
```bash
git diff --stat -- src/trading_bot/backtest/engine.py src/trading_bot/signals src/trading_bot/regime \
                   src/trading_bot/indicators src/trading_bot/risk src/trading_bot/data \
                   src/trading_bot/backtest/metrics.py src/trading_bot/backtest/equity.py
```
EXPECT: **no output.** Not one line changed in the honest core.
```bash
git diff --stat -- src/trading_bot/backtest/walkforward.py
```
EXPECT: a small diff touching only the signature, `_run_one`, `_pooled_expectancy`, the OOS loop,
and the guard — and **not** `_evaluate_gate`, `DEFAULT_GRID`, `GATE_*`, or `WalkForwardResult`.

### Plug-in and graph CLI
```bash
.venv/bin/python -m trading_bot.cli plugins
.venv/bin/python -m trading_bot.cli plugins --kind detector
.venv/bin/python -m trading_bot.cli graph-validate data/strategies/donchian-v020.strategy.json; echo "exit=$?"
```
EXPECT: six plug-ins (1 data, 3 detector, 2 policy), each with a non-empty rationale; `OK` and
`exit=0`.

### Manual Validation — the phase's own success signal
```bash
# 1. The parity claim, on real history, at full span, all three symbols.
.venv/bin/python - <<'PY'
import time
from trading_bot import config
from trading_bot.backtest.engine import run_backtest, clear_caches
from trading_bot.data.storage import connect
from trading_bot.framework import registry, context
from trading_bot.framework.execute import run_graph_backtest
from trading_bot.plugins import build_v020_graph

registry.load_all()
conn, g = connect(), build_v020_graph()
FIELDS = ("symbol","regime","pattern","direction","entry_ts","entry","stop",
          "target","exit_ts","exit_price","outcome","pnl_pct","volume_high")
for sym in config.SYMBOLS:
    clear_caches(); context.clear_caches()
    t0 = time.time(); a = run_backtest(conn, sym); t1 = time.time()
    b = run_graph_backtest(conn, g, sym);          t2 = time.time()
    ok = len(a) == len(b) and all(
        getattr(x, f) == getattr(y, f) or abs(getattr(x, f) - getattr(y, f)) < 1e-12
        for x, y in zip(a, b) for f in FIELDS
        if isinstance(getattr(x, f), float) or getattr(x, f) == getattr(y, f)
    )
    print(f"{sym}: engine {len(a)} trades {t1-t0:.1f}s | graph {len(b)} trades "
          f"{t2-t1:.1f}s | PARITY {'OK' if ok else 'FAIL'}")
print(context.cache_stats())
PY
```
- [ ] All three symbols report `PARITY OK` with a **non-zero** trade count. Zero trades on any
      symbol means the graph produced nothing — usually a wrong `ctx.window` length or a regime
      gate that never matches, not a parity bug.
- [ ] Trade counts match the v0.2.0 record. KNOWN-LIMITATIONS §6 measures **158 trades** pooled on
      stored history post-MEDIUM-2 at default params; a materially different pooled count means the
      engine path itself has drifted since, and that must be investigated *before* the graph is
      blamed.
- [ ] Graph wall time is within ~2× of the engine's per symbol. Slower than ~5× means the candidate
      memo is not being hit — check `cache_stats()["hits"]` and the branch hash in the key. This
      matters because Phase 6 runs hundreds of graphs per generation.
- [ ] `cache_stats()` shows a hit rate above ~90% on the second and third symbols' runs? No — the
      key includes the symbol, so expect ~0 cross-symbol hits and high **within**-run hits. Report
      the actual number rather than a target.
- [ ] `walkforward --graph` completes and its OOS trade list matches the engine path's for the same
      final parameters:
      ```bash
      .venv/bin/python -m trading_bot.cli walkforward --start 2023-01-01 --end 2026-07-26
      .venv/bin/python -m trading_bot.cli walkforward --start 2023-01-01 --end 2026-07-26 \
        --graph data/strategies/donchian-v020.strategy.json
      ```
      **Both runs are on the already-spent v0.2.0 span and report the already-known FAILING verdict**
      (KNOWN-LIMITATIONS §1: 2 of 5 conditions fail). They are run here as an equality check between
      two code paths, not to obtain a verdict. Do **not** report the Sharpe or the DSR as a Phase 3
      result, and do not run against any span reserved as a fresh holdout.
- [ ] `cli.py signal` output is byte-identical before and after this phase (A6 — the live path did
      not move):
      ```bash
      .venv/bin/python -m trading_bot.cli signal --as-of 2026-07-20 > /tmp/sig_after.txt
      git stash && .venv/bin/python -m trading_bot.cli signal --as-of 2026-07-20 > /tmp/sig_before.txt; git stash pop
      diff /tmp/sig_before.txt /tmp/sig_after.txt && echo "live path unchanged"
      ```
- [ ] Adding a seventh plug-in requires **zero** framework edits. Verify by writing a throwaway
      `plugins/detectors/_scratch_noop.py` (leading underscore, so `load_all` skips it — rename it
      without the underscore to test, then delete): it must appear in `cli.py plugins` with no other
      file touched. This is the PRD's iteration-cost metric, and it is the one claim the whole phase
      exists to make.

---

## Acceptance Criteria
- [ ] All 21 tasks completed
- [ ] `tests/test_framework_parity.py` **green, with `TestStoredHistoryParity` running (not skipped)** — the contract §5 gate that Phase 4 may only build on
- [ ] Full-span manual parity: 3 of 3 symbols `PARITY OK`, non-zero trade counts, counts consistent with the v0.2.0 record
- [ ] **286 pre-existing tests still pass**, plus the four new test modules
- [ ] `git diff --stat` shows **zero** changes to `engine.py`, `signals/*`, `regime/*`, `indicators/*`, `risk/*`, `data/*`, `metrics.py`, `equity.py`
- [ ] `walkforward.py`'s diff touches only the five Phase 3 sites; `_evaluate_gate`, `DEFAULT_GRID`, `GATE_*`, `WalkForwardResult` untouched
- [ ] Six plug-ins registered, each with a non-empty rationale, visible in `cli.py plugins`
- [ ] `graph-validate` exits 0 on the committed `donchian-v020.strategy.json`
- [ ] `cli.py signal` output byte-identical to pre-phase (the live path did not move)
- [ ] No new dependency in `pyproject.toml`
- [ ] No type errors — N/A (no type checker configured)
- [ ] No lint errors — N/A (no linter configured; match surrounding style by inspection)
- [ ] Matches UX design — the two new subcommands and `--graph`; nothing else user-facing changed

## Completion Checklist
- [ ] Code follows discovered patterns: keyword-only optional overrides with config fallback; `logging.getLogger("trading_bot")`; frozen dataclasses with `Attributes:` docstrings; `return None`/`[]` on a failed filter rather than raising; `not (x > 0)` for NaN-safe numeric guards
- [ ] Error handling matches codebase style: `ValueError`-compatible exceptions with messages naming what disagreed with what and why it matters (`engine.py:216-221` is the register)
- [ ] Logging follows conventions: DEBUG for per-candidate rejections, WARNING for stale data / undefined ATR, nothing at INFO inside the replay loop (it runs ~31k times per symbol)
- [ ] Tests follow conventions: tier constants derived from config, autouse cache-clearing fixture, synthetic in-memory SQLite seeded bar-by-bar, `START = 1_700_000_000_000`, hand-computed expected values, fixtures duplicated per-module rather than shared via a `conftest.py` the suite does not have
- [ ] Every regression pin names its finding (MEDIUM-2, HIGH-2, HIGH-3) in the test name or docstring
- [ ] No hardcoded values: every number resolves through `config`, a `ParamSpec` default, or a graph field. No `"4h"`, no `"1h"`, no `20`, no `1.5` literals in `framework/` or `plugins/`
- [ ] Documentation: A1-A11 reproduced in the module docstrings named beside each, so the decisions survive without this plan
- [ ] `framework/execute.py`'s docstring states the A10 duplication and the A11 Phase 4 extension point, and `_close_out` constructs `Trade` with keyword arguments only
- [ ] `test_framework_parity.py` asserts over the explicit field list, never `Trade == Trade`, with the reason in a comment
- [ ] Every number in the phase report is produced by a committed command (contract §12.5)
- [ ] Degrees of freedom consumed by this phase recorded as **zero strategy DoF**: no parameter was tuned, no grid was swept, no threshold was chosen. `ParamSpec` *bounds* are declared but nothing inside them was searched — say so explicitly, because declaring bounds looks like spending them and it is not
- [ ] Self-contained — no questions needed during implementation

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Parity fails by a handful of trades** and the cause is hard to localize | **High** | **High** | Nine sub-parity tests, each isolating one row of the parity table. Build them *before* debugging the aggregate. The three most likely causes, in order: the DataSource loading `[start_ms, end_ms]` instead of unbounded (Task 10 GOTCHA a), `ctx.window` off by one bar so Wilder ADX differs (Task 11 GOTCHA a), and `patch_trending` patching only `engine.classify_series` (Task 20 GOTCHA c) |
| **MEDIUM-2 silently re-broken** — the executor keys `h_idx` on the trigger bar's close | Medium | **Critical** | Costs 25% of opportunities *and improves in-sample results* (Sharpe 0.431→0.255 when fixed), so it hides as good news. `TestMedium2` pins it; `engine.py:458-480`'s comment is carried into `execute.py` |
| **A non-causal `series()` factory** in a future plug-in, saved only by truncation | Medium | **Critical** | `assert_trailing_only` is the first line of defence and is a required test for every plug-in; truncation is the second. Both documented in `context.py`'s docstring |
| **The candidate memo is not hit**, making the executor unusably slow for Phase 6's population | Medium | High | The key is per-branch content hash rather than per-graph, so graphs sharing a branch share candidates. `cache_stats()` makes it measurable and the manual validation step gates on wall time within ~2× |
| **The graph model cannot express the exit asymmetry**, and someone "fixes" parity by special-casing Donchian in the executor | Medium | **Critical** | The exit-asymmetry proof shows one generic block reproducing both branches from `ExitPolicySpec` flags. If a special case seems necessary, the graph model is wrong — fix the model. A4 says so in `graph.py`'s docstring |
| **Cache key missing a parameter** — e.g. ATR `period`, inherited from `engine.py:293`'s key | Medium | High | `TestSeriesCache::test_period_is_in_the_key`; `FRAMEWORK_CACHE_ENABLED=False` must produce bit-identical results, asserted in `test_parity_with_cache_disabled`. If those two ever disagree, every cached result since is suspect |
| **`contracts` ↔ `graph`/`context` import cycle** | Medium | Medium | `TYPE_CHECKING` + quoted annotations (Task 3 GOTCHA b), and the import-health command asserts no reverse dependency from `engine`/`signals`/`indicators` into `framework` |
| **Phase 4's `Trade` fields break the parity test** | Medium (certain, once Phase 4 lands) | High | The whole point of A11: parity asserts over `PARITY_EXACT`/`PARITY_CLOSE`, never `==`, with the reason in a comment so nobody "simplifies" it back. `_close_out` is pre-shaped for the three-keyword diff |
| **The pinned graph digest churns**, invalidating Phase 5/6 keys | Medium | Medium | A5 excludes `name`/`meta` and resolves defaults, which removes the churn sources that matter. The remaining trigger — a changed `ParamSpec` default — is legitimate and the test's comment says to regenerate deliberately |
| **`scan.py` drift**: live and backtest dispatch diverge (A6) | Medium | Medium | `TestScanDriftGuard`; `FADE_ENABLED` read at call time inside the detector; migration recommended for Phase 4 and listed as an open item in the phase report. **Accepted, not solved** |
| **Scope creep into Phase 4** — building the volume/MACD confirmation "while we're here" | Medium | Medium | A8 and the NOT Building list. The parity graph has zero confirmations and zero filters *because v0.2.0 has none*; adding one makes parity impossible and would be the third build in a row that grew past its phase |
| A `ParamSpec` design that serves the mutator but not the UI (or vice versa) — a change Phase 7 cannot make without editing `framework/` | Low-Medium | High | `kind`/`bounds`/`choices`/`step`/`doc` chosen for both consumers, with `clamp()` for the mutator and `doc` mandatory for the UI. Reviewed against Phase 6's and Phase 7's needs in the Dependencies table |
| The generic exit block's `elif` chain reorders exit priority | Low | **Critical** | Priority is stop/trail → channel → target, the conservative same-bar rule (`engine.py:36-37`). `TestExitMix` compares outcome counts, which is exactly what a reordering changes |
| `Trade.stop` reports the ratcheted stop instead of the initial one | Low | Medium | `engine.py:114-117` documents `stop` as the initial value; Task 7 step 5 states it; the parity field list includes `stop` |
| Two branches produce equal-valued `Signal`s and the wrong branch's exits are applied | Low | High | Branch recovery uses `is`-identity, not `==` (Task 7 step 7) |
| Registry `clear()` used in a test, breaking every later test in the session | Low | Medium | No `clear()` exists; `temporary_registry()` snapshots and restores, with the reason in its docstring |

## Notes

- **The parity test is the deliverable.** Everything else in this phase is scaffolding that only
  earns its keep if `run_graph_backtest` reproduces `engine.run_backtest` on real history. Build in
  the order given, and when parity is red, fix the *executor* to match the engine — never the
  engine to match the executor, and never the test to match both. The engine is the honest core;
  this phase's entire claim is that the framework wraps it rather than replacing it.
- **Zero strategy degrees of freedom are consumed here, and that must be stated positively.** No
  parameter is tuned, no grid is swept, no threshold is chosen. `ParamSpec` bounds are *declared* —
  which looks like spending DoF and is not — so the phase report should say "declared bounds on N
  parameters across 6 plug-ins; searched none" alongside the summed `combo_count()` from
  `cli.py plugins`. Phase 6 spends them; Phase 3 only inventories them.
- **The two honest gaps in this phase, both to be reported rather than quietly carried**: (1)
  `signals/scan.py` still has its own dispatch table (A6), so live and backtest now share detectors
  but not dispatch — a real regression in the "one code path" property KNOWN-LIMITATIONS lists under
  "What IS established", mitigated by three guards and recommended for Phase 4; (2) the cost
  arithmetic exists in two places (A10), pinned by a test rather than by construction.
- **What this phase deliberately does not prove.** That the framework makes money, that any plug-in
  has edge, that the graph model is expressive enough for the Phase 8 catalog, or that the executor
  is fast enough for a 500-member population. Those are Phases 4, 8, 6 and 9. Phase 3 proves exactly
  one thing: **the new dispatch measures the same thing the old one did.** If parity is green, every
  later phase can trust its numbers; if parity is red, nothing downstream means anything.
- **`FADE_ENABLED = False` stays false.** Migrating the fade detector is not re-enabling the sleeve.
  The graph carries the branch so the kill switch can be *proved* to still suppress; flipping it is
  a strategy decision with a logged verdict behind it (`config.py:96-102`).
- The 12-combo `DEFAULT_GRID` parity test is the cheapest insurance in the plan: it costs one
  `pytest.mark.parametrize` and it covers every configuration the gate could actually select, which
  is the set of behaviours that matter rather than the single default.
- Read `tests/test_backtest.py:701-880` (`TestPhase47ReviewRepairs`, `TestIndicatorMemo`) before
  writing the parity tests. Those classes already pin the repairs and the memo semantics this phase
  must not disturb, and several of them can be re-run through the graph path almost verbatim.
