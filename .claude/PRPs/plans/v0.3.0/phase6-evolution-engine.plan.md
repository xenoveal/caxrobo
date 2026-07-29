# Plan: Evolution Engine (v0.3.0 Phase 6)

## Summary
Phase 6 turns the Phase 4 pipeline into a **population-based search**: a campaign breeds mutated strategy graphs, scores every one through **exactly one** fitness oracle — `walkforward.walk_forward_pooled`, wrapped so no candidate can be scored without first charging Phase 1's persistent `trial_ledger` — selects parents by tournament on risk-adjusted return **versus the buy-and-hold null**, and records every generation in `data/state.db` so the search is auditable and replayable from a seed. It runs process-parallel on the 8-core M1 with worker-local caches, trains only inside a declared training span so Phase 9's holdout stays untouched, and reports a DSR deflated by *every* variant the campaign evaluated.

This plan **wraps** `walkforward.py`. It does not edit it.

## User Story
As the bot's sole operator, I want to start an overnight evolution campaign and wake up to a ranked, versioned population scored out-of-sample against buy-and-hold — with a trial count I did not have to remember to increment — so "the strategy improved" is a measured claim, and a brutal DSR tells me honestly when it is search noise.

## Problem → Solution
**Current** (KNOWN-LIMITATIONS §0c): the only search ever run moved exit management. `10 variants + 12 combos × 13 folds` were evaluated and counted by hand (§9); the entry/feature space was never explored. Nothing makes a search self-accounting — a human decides `n_trials` afterwards, from memory.

**Solution**: a `Campaign` with a seed, training span and window policy; `evolution/oracle.py` as the single scoring seam whose `evaluate()` takes a ledger handle as its **first positional argument** and charges it *before* scoring; `evolution/mutate.py` plus two `Mutator` plug-ins that jitter parameters inside `ParamSpec` bounds and edit topology inside the registry's legal moves; `evolution/tournament.py` ranking lexicographically by gate tier then Sharpe-in-excess-of-basket; `evolution/runner.py` driving a spawn-safe `ProcessPoolExecutor` where all randomness lives in the parent and workers are pure.

## Metadata
- **Complexity**: **Large** (9 new source files, 4 new test files, 2 append-only shared files; ~1150 source + ~900 test lines)
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md` — Phase 6, Evolution engine
- **Binding contract**: `v0.3.0/_shared-architecture-contract.md` §3–§8. Where the PRD and contract disagree the contract wins — see Notes.
- **Depends on**: P1 (`backtest/trials.py`, `benchmark.py`, `data/statestore.py`, `gate: dict[str,bool]`), P3 (`framework/*`, `strategy=` on `walk_forward_pooled`), P4 (a composed graph to seed from). **Parallel with** P5. **Gates** P7 and P9.
- **Estimated Files**: 15 (13 new; `config.py` and `cli.py` appended)
- **Test baseline**: **286 tests collected** (`.venv/bin/python -m pytest --collect-only -q`, verified 2026-07-27, CPython 3.11.6). All 286 stay green.

---

## Stated Assumptions

Numbered so code comments can cite them (`config.py:107` already cites another plan's "Stated Assumptions A1/A2" — repo convention).

- **A1 — The graph is the genome; the walk-forward grid is not.** Each candidate is evaluated with a **degenerate one-combo grid**. The parameters being searched live in the graph, so also sweeping `DEFAULT_GRID` would double-search and double the degrees of freedom for one candidate. One combo ⇒ no neighbour probes ⇒ ~27 graph backtests per candidate instead of ~870 (`engine.py:140-141`).
- **A2 — Fitness is `excess_sharpe`, never `dsr`.** `dsr` depends on `n_trials`, which grows through a campaign, so a DSR-based fitness would make an unchanged candidate look worse in generation 30 than generation 1 for reasons unrelated to the candidate. Selection ranks a candidate property; DSR decides the *verdict*, never the *selection*.
- **A3 — Fitness compares only inside one generation.** All members of a generation share one window (fairness); windows differ *between* generations (partial data, pivot method §2). So "best beat the seed" is only answerable by a final **audit round** re-scoring finalists *and the seed* on one fixed, pre-declared window (`runner._finalize`).
- **A4 — Failing "beats buy-and-hold" demotes; it does not eliminate.** The seed itself fails it: +3.45% ann. vs +29.4% basket (KNOWN-LIMITATIONS §0). Hard elimination makes generation 0 extinct. It is a rank tier instead.
- **A5 — Two holdouts, two levels.** `walk_forward_pooled` reserves `WF_OOS_DAYS` inside each window — the **window-OOS**, which fitness is measured on. Phase 9's **campaign holdout** is a span no generation sees; the oracle refuses spans crossing into it. Only Phase 9's is a clean holdout across a whole campaign.
- **A6 — Over-charging the ledger is the only safe error.** The ledger row is written *before* scoring, so a crash costs a trial. Duplicates are prevented at breeding time, never discounted at scoring time.
- **A7 — Reproducibility is pinned to `.venv` CPython 3.11.6.** Mersenne Twister is stable within CPython 3.x; a campaign replays bit-identically on this interpreter. That is the guarantee claimed.
- **A8 — `WF_MIN_TRADES = 30` is not lowered** (KNOWN-LIMITATIONS §4 forbids it). Most candidates therefore fail `sample_adequacy` on a 90-day window-OOS and land in tier B/C. That is a data problem (Phase 2), not a threshold problem.

---

## UX Design

**Before**: `cli walkforward` — 12 combos × 13 folds of exit parameters, once, by hand; `n_trials` guessed afterwards; no entry/feature axis searched (§0c).

**After**:
```
$ cli evolve --calibrate --repeats 3
  warm 18.4 s/candidate  cold 31.7 s  workers 6  budget 8.0 h → capacity 9391
  suggested: population 96 × generations 96  |  288 × 32
$ cli evolve --population 96 --generations 96 --seed 20260727
  campaign 20260727-a1b2c3  train 2023-07-27 → 2026-01-26
  gen 0  window 2024-03-02→2025-08-24  evals 96/96  trials 96
         best m0007  tier B  exSharpe +0.41  gate 4/7  dsr 0.0031
  AUDIT ROUND (fixed window 2024-08-04→2026-01-26): 5 finalists + seed
    seed tier C exSharpe -0.45     m4471 tier B exSharpe +0.71 ← best
  trials charged 9222   ledger total 9222
  BEST BEATS SEED: YES     GATE: FAIL (dsr, sample_adequacy)
```

| Touchpoint | Before → After |
|---|---|
| Search breadth | exit params → graph topology + every declared `ParamSpec` (closes §0c) |
| Trial accounting | human memory → one `trial_ledger` row per evaluation, charged pre-scoring (§4.2) |
| Sizing | asserted → `evolve --calibrate` measures first (§12.5) |
| Null hypothesis | zero → buy-and-hold basket, per candidate (§4.3) |
| Reproducibility | none → `--resume <campaign_id>` replays the population |
| Verdict | one bool → 7-condition gate dict + campaign-deflated DSR |

---

## Mandatory Reading

| P | File | Lines | Why |
|---|---|---|---|
| P0 | `_shared-architecture-contract.md` | all, esp. §3–§8 | Binding. §4 is this phase's central constraint. |
| P0 | `backtest/walkforward.py` | **1–412 (all)** | Wrapped, never edited. `_pooled_expectancy` (165), the `_default_combo`/on-grid trap (133-150, 365-373), `n_trials_used` (383), `_evaluate_gate` (225-254). |
| P0 | `backtest/equity.py` | 114-156, 173-220 | `expected_max_sharpe`, `deflated_sharpe`, `compute_equity_metrics(..., n_trials)`. |
| P0 | `backtest/engine.py` | 90-174 | `Trade`, `_CACHE` (146), `clear_caches` (149); 135-145 is this phase's sizing fact. |
| P0 | `backtest/trials.py` (P1) | all | The ledger you charge — verify real names before writing the adapter. |
| P0 | `framework/{contracts,registry,graph}.py` (P3) | all | `Mutator`, `ParamSpec`, `StrategyGraph`, `register`, `by_kind`, `validate()`. |
| P1 | `scripts/bruteforce/runner.py` | 57-102, 137, 161-178, 199-215 | Pool pattern, spawn belt-and-braces (72-73), workers default, **HOLDOUT refusal guard**, `initializer=`. |
| P1 | `scripts/bruteforce/registry.py` | 28-35, 107-159 | Trial counting as first-class; mandatory `rationale`; duplicate rejection. |
| P1 | `scripts/bruteforce/core.py` | 86-96, 103-129 | `SPLITS`, per-process `_FRAME_CACHE` + read-only URI connect (116). **Technique donor only — `core.score` is never an oracle.** |
| P1 | `data/storage.py` | 32, 35-73, 197-202 | `_db_lock`, WAL (56), `is_transient_db_error`. |
| P1 | `cli.py` | 128-144, 199-209, 356-368, 396-439 | Subparser, dispatch, `_fmt`, exit-code idiom. |
| P1 | `config.py` | 37-40, 105-113, 141-151, 164-173 | `REGIME_MIN_BARS = 207` (window math), frozen-constant comment style, `WF_*`. |
| P1 | `tests/test_backtest.py` | 1-46, 469-501 | Config-derived constants, autouse `clear_caches`, stub factory, tiny knobs. |
| P1 | `KNOWN-LIMITATIONS.md` | §1, §4, §9 | Why DSR is the wall and why softening it is forbidden. |
| P2 | `.claude/pivot-guide.md` | Method 1–4 | Reward/punishment, partial data, stupid→smart, ~1000 models, battling. |
| P2 | `v0.2.0/phase7-validation-protocol-repair-gate.plan.md` | all | Format bar and prior gate plan. |

**External documentation**: none needed. DSR/PSR already exist in `equity.py`; tournament selection needs no library; parallelism is stdlib. **No new dependencies** — `pandas-ta`'s disappearance from PyPI is the standing lesson (§1).

---

## The honest-accounting position (read before writing code)

Cumulative counting pushes `n_trials` into the thousands and `expected_max_sharpe` grows with it. State this in `oracle.py`'s docstring and the phase report. **Never soften the count.**

```bash
.venv/bin/python -c "
import math
from trading_bot.backtest.equity import expected_max_sharpe
v = 0.981 / 89        # SR-estimator variance at 90 daily obs, skew~0.3 kurt~4
for n in (1, 12, 64, 1920, 9216):
    print(f'{n:>6}  E[max ann. Sharpe] = {expected_max_sharpe(n, v)*math.sqrt(365):.2f}')
"
```

| `n_trials` | 1 | 12 (v0.2.0's grid) | 64 (one small gen) | 1 920 (64×30) | 9 216 (96×96) |
|---|---|---|---|---|---|
| E[max] ann. Sharpe to beat | 0.00 | ≈3.3 | ≈4.8 | ≈6.9 | ≈7.7 |

1. **Honest cumulative counting is affordable.** The bar grows roughly like √log N — 12 → 9 216 trials only doubles it. Population size is *not* what kills the gate.
2. **90 daily observations is what kills the gate.** KNOWN-LIMITATIONS §1 records DSR = **0.0210** measured, and **0.742 with the correction off entirely (`n_trials=1`)** — still under 0.95. *No choice of `n_trials` rescues that result.* The binding constraint is `n_obs`; the fixes are P1's MEDIUM-5 repair, P2's breadth, and P9's longer holdout on a pre-committed shortlist (mirroring `bruteforce/runner.py:161-178`).

**So honest accounting may make the gate unpassable inside a campaign. That is the correct answer, not a bug** — per the PRD's honesty clause ("the framework and workflow retain their value… even if this particular strategy family fails the gate"), an honest "no" is a completed outcome. Forbidden repairs: softening the count, `n_trials=1`, returning a cached score without a ledger row, shrinking the window-OOS, lowering `WF_MIN_TRADES`, scoring through anything but the oracle. *(The table's skew/kurt are illustrative — re-derive with P1's measured post-MEDIUM-5 moments for the report.)*

---

## Architecture: one seam, two holdouts, three traps

```
┌── parent: sole writer of campaigns/generations/population_members ───────────┐
│ seed graph ─► mutate.breed(rng) ─► [Member…] ─► tournament.select(rng) ─┐    │
│                ▲ ALL randomness lives here                             │    │
│                └───────────────── next generation ◄────────────────────┘    │
└─────────────────────────── picklable dicts only ────────────────────────────┘
 ProcessPoolExecutor(mp_context=spawn, initializer=_worker_init, max_workers=EVO_WORKERS)
┌── worker: pure; no RNG, no selection ───────────────────────────────────────┐
│ _W["ohlcv"]=file:ohlcv.db?mode=ro   _W["state"]=statestore+busy_timeout      │
│ _W["oracle"]=GateOracle(ledger,…)   ← the ONLY scoring path                 │
│   evaluate(): refuse span > train_end (A5) → ledger.charge (A6) →           │
│   walk_forward_pooled(strategy=graph, grid=evo_grid(), n_trials=…) (A1) →    │
│   OracleResult (plain, picklable).  Caches stay warm: the pool persists.     │
└─────────────────────────────────────────────────────────────────────────────┘
2023-01-01    2023-07-27                            2026-01-26    2026-07-25
│ regime warmup │◄────── EVO training span ────────►│◄ P9 holdout ►│
│ 207 × 1d bars │ windows jitter inside here ONLY   │ NEVER scored │
               │ window = [train | test ×N | window-OOS] → fitness │
```

**Window arithmetic (derived, not guessed)**: earliest usable start **2023-07-27** — `REGIME_MIN_BARS = 2*14-1+180 = 207` **daily** warmup bars from 2023-01-01, the date `config.py:37-40` states; earlier windows give zero trades. Minimum window `180+60+90 = **330 d**` or `walk_forward_pooled` raises `ValueError` (`walkforward.py:305-308`). Training span with P9 reserving 180 days: 2023-07-27 → 2026-01-26 = **913 d**. `EVO_WINDOW_DAYS = 540` ⇒ tune span 450 d ⇒ folds at t0 = 0/60/120/180 ⇒ **4 folds** + a 90-day window-OOS, and **373 d of start jitter** (913−540). A test pins these rather than trusting them.

### The three macOS process-parallelism traps (Task 8 implements; stated once, here)
1. **`sqlite3.Connection` is not fork-safe or shareable.** No connection is ever a task argument or `Campaign` field; each worker connects itself in `_worker_init`. `ohlcv.db` is opened **read-only** (`file:…?mode=ro`, `uri=True`) so a worker physically cannot write price history; the identical call is proven on this DB at `bruteforce/core.py:116`, and WAL (`storage.py:56`) makes concurrent readers safe. On `sqlite3.OperationalError` from that open, fall back to `storage.connect()` and log `WARNING`. **`state.db` write strategy**: the parent is the *only* writer of `campaigns`/`generations`/`population_members`; workers write **only** `trial_ledger` rows on their own connection, via `write_with_retry` + `PRAGMA busy_timeout`, classified by `storage.is_transient_db_error` so locked/busy retries and a corrupt file propagates. One small row per multi-second evaluation makes contention negligible — and `generations.db_retries` records it, so "negligible" stays a measurement.
2. **`engine._CACHE` is per-process** (`engine.py:146`), so the memo that makes ~870 backtests tractable (`engine.py:140-141`) is worker-local. Therefore **use a persistent pool and never set `max_tasks_per_child`**: workers are reused, the first task per worker pays the cold cost (frame decode, regime labels, ATR, channels) and later tasks are warm. One member per task is right *because* workers persist — batching would only save IPC, microseconds against a multi-second evaluation. Never call `engine.clear_caches()` in a worker. `calibrate()` reports cold and warm separately so this is measured.
3. **macOS defaults to `spawn`, not `fork`.** Pass `mp_context=multiprocessing.get_context("spawn")` **explicitly** so behaviour matches on Linux too, and accept the consequences: workers re-import everything, `__main__` never runs, module state starts empty, every payload must be picklable. Hence `initializer=_worker_init` (`bruteforce/runner.py:213`), belt-and-braces re-init inside the task (`bruteforce/runner.py:72-73`), and **plain dicts across the boundary** — no `StrategyGraph`, `pd.DataFrame`, `Connection`, `random.Random` or registry objects.

---

## Patterns to Mirror

Every snippet is real code at the cited `file:line`.

### GRID_DEGENERACY_TRAP
```python
# SOURCE: walkforward.py:141 + 368-373 — _default_combo() reads CONFIG defaults,
# and the final per-axis median is asserted to be ON the grid.
    return {axis: getattr(config, _CONFIG_DEFAULT_ATTR[axis]) for axis in grid}
        if value not in grid[axis]:  # pragma: no cover - defensive invariant
            raise AssertionError(f"final parameter {axis}={value!r} is not on the "
                f"grid {grid[axis]!r}; the one-shot OOS would be unvalidated")
```
**Consequence for A1**: a one-combo grid whose value is not the config default raises `AssertionError` the first time a fold misses `min_trades` and falls back. `evo_grid()` must pin exactly the config defaults, forever pinned by a test.

### TRIAL_COUNTING_AS_FIRST_CLASS
```python
# SOURCE: scripts/bruteforce/runner.py:199-201
    # Charged against the FULL registry, not the selected subset: the search
    # that produced any winner is the whole thing, and DSR must know that.
    n_trials = registry.total_trials() * len(symbols)
```

### HOLDOUT_REFUSAL_GUARD
```python
# SOURCE: scripts/bruteforce/runner.py:165-171 — refusing the unguarded case IS
# the feature. Phase 6's oracle raises instead of printing (it is a library).
        if not args.i_am_spending_the_holdout:
            print("REFUSING: --split HOLDOUT needs --i-am-spending-the-holdout.\n"
                  "The holdout may be evaluated ONCE, on the final shortlist only.",
                  file=sys.stderr)
            return 3
```

### SPAWN_SAFE_POOL
```python
# SOURCE: scripts/bruteforce/runner.py:212-215, 72-73 — initializer runs ONCE per
# worker; the in-task re-check covers spawn re-imports where __main__ never runs.
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=registry.load_all
    ) as pool:
        futures = {pool.submit(_run_one_symbol, j): j for j in jobs}
    if strat_name not in registry.ALL:
        registry.load_all()
```

### PER_PROCESS_CACHE_AND_READONLY_CONNECT
```python
# SOURCE: scripts/bruteforce/core.py:103-116 — per-process frame cache and the
# read-only URI connection, proven over thousands of sweeps on this same DB.
_FRAME_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
# SOURCE: engine.py:140-146 — why cache warmth IS the sizing question:
# "With ~870 run_backtest calls per pooled walk-forward that dominated runtime
#  completely."  _CACHE is a module-level dict: per-process, never shared.
_CACHE: dict = {}
```

### TRANSIENT_DB_ERROR_CLASSIFIER
```python
# SOURCE: storage.py:197-202 — the only sanctioned way to decide a write is worth
# retrying. Never bare-except a sqlite3.OperationalError.
def is_transient_db_error(exc: sqlite3.OperationalError) -> bool:
    """True for contention errors worth retrying/skipping ("database is locked",
    "database is busy"); False for permanent failures (corrupt file, disk I/O,
    missing table) which must propagate."""
```

### MANDATORY_RATIONALE_REGISTRATION
```python
# SOURCE: scripts/bruteforce/registry.py:129-141 — duplicates raise and name the
# claiming module; an empty rationale raises. P3's registry carries this forward.
        if key in ALL:
            raise ValueError(f"strategy {key!r} is already registered (by "
                             f"{ALL[key].build.__module__}); pick a distinct name")
        if not rationale.strip():
            raise ValueError(f"{key}: a rationale is required")
```

### DETERMINISTIC_TIE_BREAK (cautionary precedent, not a pattern to copy)
```python
# SOURCE: walkforward.py:60-63 — what happens when ties break on iteration order.
# Ten of twelve combos therefore produced IDENTICAL trade lists, fold winners
# were decided by itertools.product ordering over tied expectancies, and the
# DSR was deflated for 12 configurations per fold that were never distinct.
```

### TEST_ISOLATION_AND_STUBBING
```python
# SOURCE: tests/test_backtest.py:28-46 — constants from config; caches cleared
# around EVERY test. 481-490: stub at the name the caller resolves, and spread
# exit_ts across the span so daily_returns actually sees the trades.
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
START = 1_700_000_000_000

@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    """... test isolation must not DEPEND on that argument being right."""
    engine.clear_caches(); yield; engine.clear_caches()
```
Phase 6 tests stub **`oracle.walk_forward_pooled`** (the name bound inside `evolution/oracle.py`), never `engine.run_backtest`, so ledger-parity assertions exercise the real wrapper. Frozen result dataclasses follow `walkforward.py:98-115` — docstring says *why* a non-obvious field exists, fields appended never reordered.

---

## Consumed interfaces (verify against the real files; a rename changes only the adapter)

```python
# P1 backtest/trials.py (table `trial_ledger`, §6) — assumed:
#   record_trial(conn, *, campaign_id, graph_hash, params_hash, start_ms, end_ms) -> int
#   count_trials(conn, *, campaign_id) -> int
# P1 data/statestore.py: connect(db_path=None) -> Connection
# P1 backtest/benchmark.py: BenchmarkResult.basket -> {total_return, ann_return_pct,
#     sharpe, sortino, max_drawdown_pct, n_days}
# P1 walkforward: WalkForwardResult gains gate: dict[str,bool], benchmark,
#     n_trials_used, passed; GATE_CONDITIONS is the 7-name tuple
# P3 framework: StrategyGraph(to_dict/from_dict/validate), register, by_kind,
#     PluginSpec, ParamSpec(default, bounds|choices, kind), GraphError
# P3 walk_forward_pooled(..., strategy: StrategyGraph | None = None)
# P4 a serialized thin-slice graph under config.STRATEGY_DIR — the campaign seed

# Adapter in evolution/oracle.py:
class LedgerHandle(Protocol):
    def charge(self, *, graph_hash: str, params_hash: str,
               start_ms: int, end_ms: int) -> int: ...
    def count(self) -> int: ...
class TrialLedger:                 # the only concrete implementation
    def __init__(self, state_conn, campaign_id: str): ...
```

## Files to Change

| File | Action | Justification |
|---|---|---|
| `evolution/__init__.py` | CREATE | Package marker; re-exports `Campaign`, `GateOracle`, `run_campaign` (§2 assigns `evolution/` to P6). |
| `evolution/population.py` | CREATE | Dataclasses, `graph_hash`, seed derivation, the three `state.db` tables + DDL, single-writer CRUD with retry. |
| `evolution/oracle.py` | CREATE | **The one scoring seam**: `GateOracle`, `OracleResult`, `TrialLedger`, `HoldoutViolation`, `evo_grid()`. |
| `evolution/mutate.py` | CREATE | Parent-side breeding: RNG derivation, mutator choice, dedup redraw, `Member` construction. |
| `evolution/tournament.py` | CREATE | `Fitness`, gate tiering, total-order ranking, k-way tournament, elitism, diversity guard. |
| `evolution/runner.py` | CREATE | Spawn-safe pool, worker init, generation loop, calibration, audit round, resume, SIGINT, progress. |
| `plugins/mutators/__init__.py` | CREATE | Sub-package marker so P3's `load_all()` walk reaches the mutators. |
| `plugins/mutators/param_jitter.py` | CREATE | `mutator.param-jitter` — jitter inside `ParamSpec` bounds/choices. |
| `plugins/mutators/graph_edit.py` | CREATE | `mutator.graph-edit` — detector swap/add/remove from an enumerated legal-edit set. |
| `config.py` | UPDATE (append) | `EVO_*` block, phase-headed (§7). No existing constant modified. |
| `cli.py` | UPDATE (append) | `evolve` subparser + `_evolve_command`, in phase order. |
| `tests/test_evolution_{mutate,tournament,oracle,runner}.py` | CREATE | The four reserved test files (§8). |

## Config constants — the whole `EVO_*` block

Appended to `config.py` (§7 reserves `EVO_*` for P6). Two values are **provisional until Task 9 measures them**.

```python
# ---------------------------------------------------------------------------
# Phase 6 (v0.3.0): evolution engine. The gate is the only fitness oracle
# (contract §4); nothing here may soften trial counting.
# ---------------------------------------------------------------------------
EVO_TRAIN_START = "2023-07-27"  # FROZEN: first date with a non-"uncertain" regime
                                # label (207 daily warmup bars, see REGIME_MIN_BARS)
EVO_TRAIN_END = "2026-01-26"    # FROZEN CEILING. Never None, never "now": an implicit
                                # end would swallow Phase 9's holdout. When Phase 9
                                # lands HOLDOUT_START_MS the ceiling becomes min(both).
EVO_WINDOW_DAYS = 540           # >= WF_TRAIN+TEST+OOS (330). At 540: 4 folds + a
                                # 90-day window-OOS, and 373 days of start jitter.
EVO_WINDOW_JITTER = True        # partial-data training, pivot-guide method §2
EVO_POPULATION = 32             # PROVISIONAL — replace from `evolve --calibrate`
EVO_GENERATIONS = 8             # PROVISIONAL — replace from `evolve --calibrate`
EVO_BUDGET_HOURS = 8.0          # overnight wall-clock budget for the sizing math
EVO_WORKERS = max(1, (os.cpu_count() or 4) - 2)  # mirrors bruteforce/runner.py:137
EVO_ELITES = 2                  # carried unchanged but RE-SCORED, so still charged a
                                # trial (A6): carrying an elite is not free
EVO_TOURNAMENT_K = 3            # k-way tournament, sampled WITH replacement
EVO_GRAPH_EDIT_SHARE = 0.35     # share of offspring bred by graph-edit vs jitter
EVO_JITTER_NODES = 1            # nodes touched per param-jitter mutation
EVO_JITTER_SIGMA = 0.15         # gaussian sigma as a fraction of a float bound width
EVO_BOOL_FLIP_P = 0.25          # probability a bool param flips
EVO_DEDUP_MAX_REDRAWS = 8       # duplicate graph_hash -> redraw, then accept + log
EVO_MIN_UNIQUE_FRACTION = 0.5   # below this the diversity guard fires
EVO_FINALISTS = 5               # audit-round size, plus the seed (A3)
EVO_STRICT_MUTATORS = True      # a Mutator returning an invalid graph is a BUG:
                                # abort the campaign. Never silently skip.
EVO_DB_RETRIES = 5              # ledger write retries on transient errors
EVO_DB_RETRY_SLEEP_S = 0.25     # linear backoff base
EVO_DB_BUSY_TIMEOUT_MS = 5000   # per-connection PRAGMA in each worker
```

**Deliberately absent**: any drawdown-penalty coefficient. Sharpe is already risk-adjusted and drawdown enters through the gate tier; a weight would be another fitted degree of freedom for no measured benefit (same reasoning as `config.py:169-171`, "k was NOT adjusted, so no degree of freedom was consumed").

## `state.db` schema (§6 — owned by `evolution/population.py`)

Idempotent `CREATE TABLE IF NOT EXISTS` in the owning module, mirroring `storage.connect()`. **All timestamps epoch ms, UTC.**

```sql
CREATE TABLE IF NOT EXISTS campaigns (
  campaign_id TEXT PRIMARY KEY,      -- "<YYYYMMDD>-<6 hex of seed graph hash>"
  seed INTEGER NOT NULL,             -- the ONE number a replay needs
  seed_graph_hash TEXT NOT NULL,
  seed_graph_json TEXT NOT NULL,     -- self-contained: resume needs no seed file
  config_json TEXT NOT NULL,         -- every EVO_* value ACTUALLY used
  symbols_json TEXT NOT NULL,
  train_start_ms INTEGER NOT NULL, train_end_ms INTEGER NOT NULL,  -- oracle ceiling
  audit_start_ms INTEGER NOT NULL,   -- fixed audit window declared UP FRONT so it
  audit_end_ms INTEGER NOT NULL,     -- cannot later be chosen to flatter a winner
  population INTEGER NOT NULL, generations INTEGER NOT NULL,
  started_ts INTEGER NOT NULL, finished_ts INTEGER,
  status TEXT NOT NULL               -- running | done | aborted
);
CREATE TABLE IF NOT EXISTS generations (
  campaign_id TEXT NOT NULL, gen_index INTEGER NOT NULL,
  window_start_ms INTEGER NOT NULL, window_end_ms INTEGER NOT NULL,
  population_size INTEGER NOT NULL, n_evaluated INTEGER NOT NULL,
  n_errors INTEGER NOT NULL DEFAULT 0, n_unique_graphs INTEGER NOT NULL,
  trials_cumulative INTEGER NOT NULL,   -- ledger.count() AFTER this generation
  best_member_id TEXT, best_fitness REAL,
  db_retries INTEGER NOT NULL DEFAULT 0, wall_seconds REAL,
  started_ts INTEGER NOT NULL, finished_ts INTEGER,
  PRIMARY KEY (campaign_id, gen_index)
);
CREATE TABLE IF NOT EXISTS population_members (
  member_id TEXT PRIMARY KEY,        -- "<campaign_id>:<gen>:<idx>"
  campaign_id TEXT NOT NULL, gen_index INTEGER NOT NULL, member_index INTEGER NOT NULL,
  parent_member_id TEXT,             -- lineage: every candidate is auditable
  graph_hash TEXT NOT NULL, graph_json TEXT NOT NULL,
  mutator TEXT NOT NULL DEFAULT '',  -- registry key; '' = seed/elite
  mutation_json TEXT NOT NULL DEFAULT '{}',   -- the (node, param, old, new) diff
  rng_seed INTEGER NOT NULL,
  role TEXT NOT NULL,                -- seed | offspring | elite | finalist
  fitness REAL, tier TEXT, excess_sharpe REAL, excess_ann_return REAL,
  sharpe REAL, dsr REAL, ann_return_pct REAL, max_drawdown_pct REAL, n_trades INTEGER,
  bench_sharpe REAL, bench_ann_return_pct REAL, n_trials_used INTEGER,
  gate_json TEXT,                    -- dict[str,bool] keyed by GATE_CONDITIONS
  window_start_ms INTEGER, window_end_ms INTEGER,
  eval_seconds REAL, error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_pm_campaign_gen ON population_members(campaign_id, gen_index);
CREATE INDEX IF NOT EXISTS idx_pm_graph_hash ON population_members(graph_hash);
```

## NOT Building

- **Any edit to `backtest/walkforward.py`.** P1 owns the gate extension, P3 added `strategy=`; P6 wraps it, passing `strategy=`, `grid=`, `n_trials=`, `start_ms=`, `end_ms=`. §7: "Phase 6 does **not** edit this file; it wraps it." Enforced by `git diff --stat`.
- **A second fitness path.** `scripts/bruteforce/core.score` is **not** an oracle and must not be imported under `src/trading_bot/`. `evolution/` never calls `run_backtest`, `run_graph_backtest`, `compute_metrics` or `compute_equity_metrics` directly.
- **Deep RL / neural policies.** Deferred by the PRD Decisions Log (Mac-only compute; auditable variants; deep RL's overfitting record) and by "What We're NOT Building". Every variant stays a readable config so "why did it trade" remains answerable.
- **The final holdout campaign** — P9. P6 provides the ceiling hook and refuses to cross it; no `HOLDOUT_*`/`CAMPAIGN_*` constant is defined here (§7).
- **`feedback/` anything** (P5 owns it; P6 writes neither `strategy_versions` nor `review_records`) and **`ui/` anything** (P7; progress is stdout + `state.db`, which P7 reads).
- **New detectors or indicators** — P8. `graph_edit` swaps only what is already registered.
- **New dependencies** (stdlib only); **cloud/distributed execution** (deferred "until the loop shows life"); **multi-position margin accounting** (§10 Q7: out of scope for all nine phases); **CPCV or any fold-mechanics change** (spans are passed to the existing splitter).
- **A cross-process result cache** — duplicates are prevented at breeding time (A6); a cache returning a score without a ledger row is exactly the bypass this phase prevents.
- **Crossover of two parents.** Mutation-only for v1: crossover on a typed graph needs a compatibility theory that does not exist yet. Listed so its absence is a decision.
- **Lowering `WF_MIN_TRADES`, shrinking `WF_OOS_DAYS`, touching the cost model** — forbidden by KNOWN-LIMITATIONS §4 and §1.

---

## Step-by-Step Tasks

### Task 1: `config.py` — append the `EVO_*` block
- **ACTION / IMPLEMENT**: append the block above verbatim; add `import os` (currently only `datetime` is imported). The two provisional constants keep the literal `PROVISIONAL`.
- **MIRROR**: `config.py:105-113` — phase-headed comment saying why the value exists and whether it is frozen.
- **GOTCHA**: modify **no** existing constant (`RR_FLOOR`, `WF_MIN_TRADES`, `FEE_PCT`, `MAX_HOLD_BARS_TRIGGER` are frozen); define nothing named `HOLDOUT_*`, `CAMPAIGN_*`, `REVIEW_*`, `UI_*`, `STRATEGY_DIR`.
- **VALIDATE**: `python -c "from trading_bot import config; print(config.EVO_WORKERS, config.EVO_WINDOW_DAYS)"` → `6 540` here (`cpu_count == 8`); `grep -c "^EVO_" config.py` → 20.

### Task 2: `evolution/population.py` — state, hashing, persistence
- **ACTION**: create `evolution/__init__.py` and `population.py`.
- **IMPLEMENT**: `Campaign`/`Generation`/`Member` as `@dataclass(frozen=True)`; `graph_hash(graph)` = P3's hash if exposed else `sha256` of canonical JSON (`sort_keys=True, separators=(",", ":")`); `derive_seed(*parts)` and `member_rng(campaign_seed, gen, idx)` via `hashlib.blake2b(key, digest_size=8)`; `ensure_schema`, `insert_campaign`, `load_campaign`, `insert_generation`, `finish_generation`, `insert_members`, `update_member_result`, `top_members`, `last_completed_generation`; `write_with_retry(conn, fn)` retrying `EVO_DB_RETRIES` times with `EVO_DB_RETRY_SLEEP_S * attempt` **iff** `storage.is_transient_db_error(exc)`, re-raising otherwise, returning `(result, n_retries)`.
- **MIRROR**: TRANSIENT_DB_ERROR_CLASSIFIER; `storage.py:35-73` for DDL-in-the-owning-module; `walkforward.py:98-115` for dataclass style.
- **IMPORTS**: `hashlib`, `json`, `random`, `sqlite3`, `time`, `logging`, `dataclasses`; `config`; `data.{statestore,storage}`.
- **GOTCHA**: **never use builtin `hash()`** for seed derivation — it is `PYTHONHASHSEED`-salted on `str` and spawned workers get a *different* salt, so a campaign would be unreproducible in a way that looks like nondeterministic code. No function here accepts a connection from another process (trap 1).
- **VALIDATE**: `derive_seed(1,2,3)` prints the same integer across two separate interpreter invocations.

### Task 3: `evolution/oracle.py` — the only fitness oracle
- **ACTION**: create. The highest-stakes file in the phase.
- **IMPLEMENT**: module docstring stating the honest-accounting position verbatim (one code path; `core.score` is not an oracle; charge before scoring; DSR 0.0210 and 0.742 at `n_trials=1`; never soften). Then, with `__all__ = ("GateOracle","OracleResult","TrialLedger","LedgerHandle","HoldoutViolation","evo_grid")`:
  ```python
  class HoldoutViolation(RuntimeError): ...   # span would cross train_end
  def evo_grid() -> dict[str, tuple]:         # ONE combo == config defaults (A1)
  @dataclass(frozen=True)
  class OracleResult: ...   # the population_members metric fields + graph_hash
  class GateOracle:
      def __init__(self, ledger: LedgerHandle, *, ohlcv_conn, symbols,
                   train_start_ms, train_end_ms, min_trades=None) -> None: ...
      def evaluate(self, graph, *, window_start_ms, window_end_ms) -> OracleResult:
          # 1. refuse an out-of-bounds span FIRST, before charging       (A5)
          # 2. n_trials = self._ledger.charge(...)                       (A6)
          # 3. walk_forward_pooled(self._conn, list(self._symbols), start_ms=…,
          #      end_ms=…, grid=evo_grid(), strategy=graph,
          #      n_trials=n_trials, min_trades=self._min_trades)          (A1)
          # 4. flatten to OracleResult; ValueError/GraphError -> tier-D + error=
  ```
  `ledger` is a **required positional argument**; everything else keyword-only. No default, no `None` fallback, no `set_ledger()`, no module-level function that scores a graph.
- **MIRROR**: TRIAL_COUNTING_AS_FIRST_CLASS; HOLDOUT_REFUSAL_GUARD (as an exception — this is a library); GRID_DEGENERACY_TRAP.
- **IMPORTS**: `from trading_bot.backtest.walkforward import walk_forward_pooled` — **the only such import in the package**; `backtest.trials`; `config`; `data.storage`.
- **GOTCHA**: `params_hash` = hash of the degenerate combo JSON, constant across candidates — exactly right: it records that **one** configuration was evaluated per candidate, not twelve. On `ValueError` from a short span return tier D with `error=` and **keep the ledger row**. `sharpe`/`dsr`/`ann_return_pct` can each be `None` (`equity.py:211-220`) — never coerce to `0.0`; a missing metric must not read as a mediocre one. Read `benchmark.basket[...]` for the excess but take the `beats_benchmark_*` booleans from P1's `gate` dict — two implementations of one condition is how they drift.
- **Catching a future violation in review**: the four checks in Validation Commands — a grep proving `oracle.py` is the sole importer, a grep proving zero bruteforce/`run_backtest` leakage, `TestNoSecondOracle` asserting both from inside pytest (no linter needed), and `TestLedgerParity` counting oracle calls against `ledger.count()`.
- **VALIDATE**: `pytest tests/test_evolution_oracle.py -v` plus those greps.

### Task 4: `evolution/tournament.py` — fitness, tiers, selection
- **ACTION**: create. Pure functions over `OracleResult`; no I/O, no RNG creation (`rng` is always passed in).
- **IMPLEMENT**: `TIER_ORDER = ("A","B","C","D")` (A best) — **A** every gate condition passes; **B** both `beats_benchmark_*` True but another condition fails; **C** valid metrics but a `beats_benchmark_*` fails; **D** undefined (`sharpe` None / `n_trades` 0 / oracle error). `Fitness(tier, score, excess_sharpe, excess_ann_return, n_trades)` frozen, `score = excess_sharpe` (`-inf` for tier D). `score_member(result)`; `rank_key(f, member_index) -> (-tier_rank, score, excess_ann_return, n_trades, -member_index)` — a **total** order, the last term guaranteeing no tie breaks on dict or completion order; `select_parents(scored, rng, *, k, n)`; `elites(scored, *, n)`; `unique_fraction`; `diversity_guard_needed`. Tiering reads only `gate: dict[str,bool]` and the benchmark fields, never re-deriving a condition.
- **MIRROR**: DETERMINISTIC_TIE_BREAK; `walkforward.py:225-254` for "fails safely on `None`, never raises".
- **GOTCHA**: A4 — tier C is **selectable** (the seed is tier C); eliminating it empties generation 0. Never put `dsr` in `score` (A2). `rng.choices` samples **with** replacement (intended); `rng.sample` would silently change selection pressure. If every member is tier D, log `WARNING` and let the runner re-breed from the previous generation's elites with the diversity guard on.
- **VALIDATE**: `pytest tests/test_evolution_tournament.py -v`.

### Task 5: `plugins/mutators/param_jitter.py`
- **ACTION**: create `plugins/mutators/__init__.py` (docstring only) and `param_jitter.py`.
- **IMPLEMENT**: `@register("mutator", name="param-jitter", params={nodes, sigma, flip_p as ParamSpecs defaulting to EVO_JITTER_NODES/EVO_JITTER_SIGMA/EVO_BOOL_FLIP_P}, rationale="Local search inside declared ParamSpec bounds: the cheapest useful move, and the only one that cannot change topology, so a jittered child is always structurally valid.")`. Per-kind, always **clamped**, never rejection-looped: `int` → `clamp(old ± max(1, round(sigma*(hi-lo))), lo, hi)`; `float` → `clamp(rng.gauss(old, sigma*(hi-lo)), lo, hi)`; `bool` → flip with `flip_p`; `choice` → `rng.choice([c for c in choices if c != old])`. Return a **new** graph via `from_dict(deepcopy(to_dict(...)))`.
- **MIRROR**: MANDATORY_RATIONALE_REGISTRATION; §3's `Mutator.mutate(self, graph, rng, **params)` (a registered plain function is fine per §3).
- **IMPORTS**: `random`, `copy`; `config`; `framework.registry.register`; `framework.contracts.ParamSpec`; `framework.errors.GraphError`.
- **GOTCHA**: **never mutate the input graph in place** — the parent is reused for elitism, lineage and dedup, and an in-place edit destroys reproducibility. A param with no declared `bounds`/`choices` is **not** jitterable: skip and log `DEBUG` — inventing a range is exactly the failure `ParamSpec` exists to prevent (§3). If no node has a jitterable param, raise `GraphError` rather than return a silent no-op that burns a trial on a duplicate. Clamp **after** rounding for ints or bounds get exceeded by 1.
- **VALIDATE**: `cli plugins | grep "mutator.param-jitter"` shows a non-empty rationale (§12.3).

### Task 6: `plugins/mutators/graph_edit.py`
- **ACTION**: create — detector swap/add/remove.
- **IMPLEMENT**: register with `params={"p_swap","p_add","p_remove"}` and a rationale naming the gap it closes ("topology search is the only move that can reach a strategy the seed's shape cannot express — the gap KNOWN-LIMITATIONS §0c records as never searched"). `_legal_edits(graph)` **enumerates** every legal move first; `rng` then picks one weighted by `p_*`. Legality: SWAP a `detector`/`confirmation` node's plug-in for another **registered** key of the same kind (`registry.by_kind`), new params from each `ParamSpec.default`; ADD a `confirmation` or second `detector` where arity allows, never a second `policy`/`filter`; REMOVE a `confirmation`, or a `detector` **only while ≥1 remains**, never the policy and never the filter (the ≥1:2-R:R-after-costs filter is the pipeline's point). Then `new_graph.validate()`; on failure raise `GraphError` naming the attempted edit. Empty `_legal_edits` ⇒ raise `GraphError`, **no no-op fallback**.
- **MIRROR**: MANDATORY_RATIONALE_REGISTRATION; `registry.by_kind` (§3).
- **GOTCHA**: **an invalid graph must fail loudly at mutation time, not silently produce zero trades and score as "safe"** — a broken graph that takes no trades yields `sharpe=None` → tier D → `-inf`, which *looks* like honest rejection while hiding a mutator bug behind a plausible number. Validate at construction; `EVO_STRICT_MUTATORS=True` makes `GraphError` abort the campaign, mirroring registry's "import errors during `load_all()` are fatal, never skipped" (§3). Keep the modes distinct in the audit log: **invalid graph** (bug → abort) vs **valid graph taking zero trades** (legitimate tier D → recorded, charged, selected against). Do not offer a detector whose declared `timeframes` are absent from the store — 15m exists for BTC/ETH/SOL only (§0).
- **VALIDATE**: `cli plugins | grep "mutator.graph-edit"`; `pytest tests/test_evolution_mutate.py -k graph_edit -v`.

### Task 7: `evolution/mutate.py` — parent-side breeding
- **ACTION**: create. Turns a scored generation into the next generation's `Member` rows. **Parent-only.**
- **IMPLEMENT**: `breed(campaign, gen_index, parents, *, rng, seen_hashes) -> list[Member]` — elites first (`role='elite'`, `mutator=''`), then offspring until full. Per offspring: `member_rng = population.member_rng(campaign.seed, gen_index, idx)`; choose `graph-edit` with probability `EVO_GRAPH_EDIT_SHARE` else `param-jitter`; call `spec.fn(parent_graph, member_rng, **params)`; hash; if the hash is in `seen_hashes`, redraw up to `EVO_DEDUP_MAX_REDRAWS` times **from the same `member_rng`** (so the redraw sequence is itself reproducible), then accept the duplicate and log `INFO`.
- **MIRROR**: `walkforward.py:313-316`'s idiom of keeping a list parallel to results because a dataclass cannot carry everything — keep the mutation diff beside the `Member` and persist it.
- **GOTCHA**: **all randomness lives here, in the parent, before dispatch** — the single decision that makes parallelism and reproducibility compatible: workers become pure functions of `(graph, window)`, so worker count, completion order and machine speed cannot change the population, and a mutation inside a worker would destroy reproducibility with no test obviously failing. Derive each member's RNG from `(seed, gen_index, member_index)`, **not** a shared stream advanced in loop order, so changing elite count or population size does not reshuffle every downstream member. Seed `seen_hashes` from `population_members` on `--resume` or a resumed campaign re-breeds duplicates it already paid for.
- **VALIDATE**: `pytest tests/test_evolution_mutate.py -v`; two `breed()` calls with identical inputs give identical `graph_hash` lists.

### Task 8: `evolution/runner.py` — the spawn-safe campaign loop
- **ACTION**: create. Implements the three traps in Architecture — re-read them before writing.
- **IMPLEMENT**: module-level `_W: dict` holding `"ohlcv"`, `"state"`, `"oracle"`. `_worker_init(campaign_json)` runs once per worker: `plugins.load_all()` (import errors FATAL) → read-only `ohlcv` connect → `statestore.connect()` + `PRAGMA busy_timeout` → `GateOracle(TrialLedger(st, campaign_id), ohlcv_conn=…, symbols=…, train_start_ms=…, train_end_ms=…)`. `_evaluate_task(payload) -> dict` is pure: re-init if `_W` is empty (spawn), `StrategyGraph.from_dict(payload["graph"])`, `evaluate(...)`, return `dataclasses.asdict(result)` plus `member_id`. Plus `calibrate(*, repeats=3, workers=None)`, `run_campaign(*, seed_graph, seed, symbols, population, generations, workers=None, window_days=None, train_start_ms=None, train_end_ms=None, resume_campaign_id=None, progress=print)`, `_finalize(...)`.
  Generation loop: draw the window from the **campaign** rng (one window per generation, shared by all members, A3) → `insert_generation` → `mutate.breed` → `insert_members` → submit one task per member → collect → `update_member_result` (parent, single writer) → rank → `finish_generation` with `trials_cumulative = ledger.count()`. `_finalize`: the `EVO_FINALISTS` best distinct `graph_hash`es across all generations **plus the seed**, re-evaluated on `campaign.audit_start_ms/audit_end_ms` (fixed at campaign start), each charging a trial; `BEST BEATS SEED` decided from that comparison alone.
- **MIRROR**: SPAWN_SAFE_POOL; PER_PROCESS_CACHE_AND_READONLY_CONNECT; `bruteforce/runner.py:216-228` for `as_completed` + `done/total … eta` progress.
- **IMPORTS**: `dataclasses`, `json`, `logging`, `math`, `os`, `signal`, `sqlite3`, `time`; `from concurrent.futures import ProcessPoolExecutor, as_completed`; `multiprocessing`; `config`, `plugins`; `data.{statestore,storage}`; `framework.graph.StrategyGraph`; `evolution.{mutate,oracle,population,tournament}`.
- **GOTCHA**: merge results in **member-index order**, never `as_completed` order, before ranking (`as_completed` yields by completion time; ranking on it makes selection machine-speed-dependent — `rank_key`'s `-member_index` is the second line of defence). Catch `Exception` per task, record `population_members.error`, count `generations.n_errors`, continue (`bruteforce/runner.py:101-102`) — but a mutator `GraphError` happens in the *parent* before dispatch and, under `EVO_STRICT_MUTATORS`, aborts: different failures, different severities, on purpose. On SIGINT set a stop flag, let the generation drain, write `status='aborted'` with `finished_ts` (a half-written generation with no `finished_ts` is how a resumed campaign double-counts). `--resume` reloads `Campaign` from `campaigns` (the row carries `seed_graph_json`, so resume never needs the seed file), finds `last_completed_generation`, rebuilds `seen_hashes`, continues — the ledger is persistent, so `n_trials` continues from the true total across restarts, exactly as §4 requires.
- **VALIDATE**: `pytest tests/test_evolution_runner.py -v`; then `cli evolve --population 4 --generations 2 --workers 2 --seed 1` writes 2 `generations` rows and 8 `population_members` rows.

### Task 9: Calibration — measure per-candidate cost, then derive the size
- **ACTION**: implement `calibrate()` and `--calibrate`. **Nothing in this plan asserts a population size; this task produces it.**
- **IMPLEMENT**: evaluate the seed graph `repeats` times on a mid-training window, single-process, timing each: `cold_seconds` (first evaluation in a fresh process — SQLite decode plus regime/ATR/channel warm-up), `warm_seconds` (median of the rest), `backtests_per_eval = (n_folds*2 + 1) * len(symbols)` printed so the cost is explainable. Then `capacity = floor(workers * budget_hours * 3600 / warm_seconds)`; print `(population, generations)` pairs with product ≤ capacity, biased to `generations ≈ population` and `population ≥ 8 * EVO_TOURNAMENT_K`. Print the pivot-guide reconciliation: *"~1000 simultaneous models" is reachable per generation only if `warm_seconds ≤ workers * budget_hours * 3600 / (1000 * generations)` — at 6 workers / 8 h / 30 generations, 5.8 s per candidate. Above that the honest answer is a smaller population, which the PRD's risk table already predicts ("Mac-only compute caps population size/generations").*
- **MIRROR**: `bruteforce/runner.py:222-228` (rate + ETA); `config.py:164-173` for recording a measurement *in the source*, beside the constant it justifies.
- **GOTCHA**: calibrate with one worker in-process so cache effects are attributable; `workers` in the formula is a **multiplier assumption**, so also report a 2-worker spot check and note that the M1's 8 cores are 4 performance + 4 efficiency — throughput does **not** scale linearly past ~4–6.
- **VALIDATE**: `cli evolve --calibrate --repeats 3`, **then edit `config.py`** replacing the two `PROVISIONAL` values with the derived ones *and the measurement* — e.g. `EVO_POPULATION = 96  # from cli evolve --calibrate 2026-07-2X on M1/8-core: warm 18.4 s, cold 31.7 s, 6 workers, 8.0 h -> capacity 9391. NOT guessed.` Afterwards `grep -n PROVISIONAL config.py` returns nothing.

### Task 10: `cli.py` — the `evolve` subcommand
- **ACTION**: append the subparser (phase order, after P5's `review`) and `_evolve_command`.
- **IMPLEMENT**: args `--seed-graph`, `--seed`, `--symbol` (repeatable; default `config.SYMBOLS`), `--population`, `--generations`, `--workers`, `--window-days`, `--train-start`/`--train-end` (`type=_date_arg`), `--budget-hours`, `--calibrate`, `--repeats`, `--resume`, `--report`, `--dry-run`. Exit codes: **0** completed and the audit round shows the best beating the seed; **1** completed, no improvement; **2** aborted or fatal (`GraphError`, `HoldoutViolation`). Printout: per-generation line, audit-round table, `trials charged` vs `ledger total`, the **full 7-condition gate dict** in `GATE_CONDITIONS` order for the best member, `BEST BEATS SEED: YES|NO`. `--report` reprints a finished campaign from `state.db` without evaluating.
- **MIRROR**: `cli.py:356-368` + `425-439` (print/exit idiom); `cli.py:128-144` (subparser); `cli.py:199-209` (dispatch).
- **GOTCHA**: `--symbol` defaults to `config.SYMBOLS` (3); when P2 publishes `RESEARCH_SYMBOLS`, **pass it explicitly** rather than changing the default — the symbol set is part of a campaign's identity and is stored in `campaigns.symbols_json`. `--train-end` must be validated against `EVO_TRAIN_END` and refused if later, in the spirit of `bruteforce/runner.py:165-171`: a CLI flag must not be able to spend P9's holdout. `--dry-run` breeds and prints generation 0 with **zero** oracle calls and zero trials — say so in the help text, because a dry run that charged trials would be a trap. Do **not** fold `evolve` into the existing `elif args.command in ("backtest","walkforward")` block: that defaults `end_ms` to *now*, and this phase needs the training ceiling.
- **VALIDATE**: `cli evolve --help`; the Task 8 smoke campaign; `cli evolve --report <id>`.

### Tasks 11–14: the four test modules
Each row of the Testing Strategy table below names its module, its input and its expected result — that table is the specification, not a summary. For all four: class-based grouping, `tmp_path` SQLite via `statestore.connect(str(tmp_path / "s.db"))` (never the real `data/state.db`), constants derived from config, `START = 1_700_000_000_000`, autouse cache clearing, every `random.Random` explicitly seeded, assertions on `graph_hash` never `repr`, and the finding or assumption each regression pins named in the test name or docstring (§8).

- **Task 11 — `tests/test_evolution_mutate.py`**: `TestParamJitter`, `TestGraphEdit`, `TestBreed`. **MIRROR** TEST_ISOLATION_AND_STUBBING and `tests/test_wilder.py::TestHandComputedValues` for hand-computed clamp arithmetic.
- **Task 12 — `tests/test_evolution_tournament.py`**: `TestTiers`, `TestRanking`, `TestSelection`. **GOTCHA**: build `gate` dicts from `walkforward.GATE_CONDITIONS`, not literal strings, so a P1 rename fails loudly instead of silently mis-tiering everything.
- **Task 13 — `tests/test_evolution_oracle.py`** (the architectural-invariant file): `TestLedgerParity`, `TestLedgerIsRequired` (`inspect.signature(GateOracle.__init__).parameters["ledger"]` positional with **no default**), `TestNoSecondOracle` (walk `evolution/*.py` with `pathlib`), `TestHoldoutCeiling` (raises before any ledger row *and* before the gate is called), `TestEvoGrid`, `TestNTrialsPassthrough`. **GOTCHA**: monkeypatch the name **bound inside `oracle.py`**; patching `walkforward.walk_forward_pooled` would not intercept an already-imported reference.
- **Task 14 — `tests/test_evolution_runner.py`**: `TestDeterminism`, `TestPickleSafety`, `TestWorkerInit`, `TestPersistence`, `TestOrdering`, `TestAuditRound`, `TestCalibration`. **MIRROR** `tests/test_backtest.py:495-501`'s tiny knobs (`train_days=10, test_days=5, oos_days=5`) so a synthetic campaign runs in milliseconds. **GOTCHA**: do not spawn real processes in the suite — under `spawn`, pytest re-imports the test module in each child, which is slow and can deadlock under capture. Test `_worker_init`/`_evaluate_task` directly in-process, and test pool wiring by asserting arguments passed to a monkeypatched `ProcessPoolExecutor`.

### Task 15: Run the campaign; write the phase report
- **ACTION**: with the suite green: calibrate, set the measured constants, run one overnight campaign on the largest available symbol set, then write `.claude/PRPs/reports/phase6-evolution-campaign.md`.
- **IMPLEMENT** — every number produced by a committed command (§12.5): pytest count before/after (**286** → 286 + new); the calibration measurement; campaign id, seed, symbols, training span, window policy, audit window; per generation best fitness, tier, `n_errors`, `n_unique_graphs`, `db_retries`, `wall_seconds`, `trials_cumulative`; the audit round and `BEST BEATS SEED`; the best member's full 7-condition gate dict, `dsr` and `n_trials_used`; degrees of freedom consumed (see below); the measured `expected_max_sharpe` table using P1's post-MEDIUM-5 moments; and which gate conditions failed with why softening them is not the response.
- **MIRROR**: `KNOWN-LIMITATIONS.md`'s tone — measured tables, named findings, no rounding in the flattering direction. The repo already committed once to "Use measured rather than derived figures in the benchmark table" (`git log`); keep it.
- **GOTCHA**: if the best candidate beats the seed but the gate still fails on `dsr`/`sample_adequacy`, that is the **expected** outcome; say so plainly, citing §1's `n_trials=1 → 0.742`. Do not re-run hunting a pass — a re-run is a new campaign with its own trials, and the ledger remembers.
- **VALIDATE**: every number traceable to `state.db` or a pasted command; `cli evolve --report <id>` reproduces the tables.

---

## Testing Strategy

| Module | Test | Input | Expected |
|---|---|---|---|
| oracle | ledger parity | N evaluations incl. failures | `ledger.count() == oracle calls` (**§4**) |
| oracle | ledger required | `GateOracle()` | `TypeError`; no free scoring function in `__all__` |
| oracle | no second oracle | source scan of `evolution/*.py` | `walk_forward_pooled` only in `oracle.py`; zero bruteforce/`run_backtest` hits |
| oracle | holdout ceiling | `window_end > train_end` | `HoldoutViolation`, ledger untouched (**A5**) |
| oracle | `evo_grid()` pinning | — | one combo == `_default_combo(DEFAULT_GRID)` — pins `walkforward.py:368-373` |
| oracle | `n_trials` passthrough | 3 sequential evaluations | strictly increasing, == ledger count |
| mutate | jitter bounds / purity | 200 seeded draws; any mutation | inside bounds, types preserved; parent `to_dict()` unchanged |
| mutate | invalid graph | monkeypatched broken edit | `GraphError`, **not** a zero-trade graph |
| mutate | unjitterable / removal floor | no bounded params; 1 detector | `GraphError`; REMOVE not offered |
| mutate | breed determinism / dedup | same seed twice; forced collision | identical `graph_hash` list; redraw then accept + log |
| tournament | tier order, tier-C selectable | hand-built gate dicts | A>B>C>D regardless of score; tier C eligible (**A4**) |
| tournament | total order | 1000 shuffles of tied fitnesses | identical ranking — pins `walkforward.py:60-63` |
| tournament | all-tier-D population | every member zero-trade | warns, re-breeds from prior elites, no crash |
| runner | campaign determinism, resume | same seed; interrupt at gen 1 | identical hashes and `best_member_id`; resume matches (**A7**) |
| runner | pickle safety / worker init | payloads; `_worker_init` twice | round-trips, no Connection/DataFrame/Random; idempotent (**trap 3**) |
| runner | completion-order independence | shuffled `as_completed` | identical ranking |
| runner | retry classification | `"locked"` / `"no such table"` | retry / immediate re-raise (**trap 1**) |
| runner | schema, `trials_cumulative` | `ensure_schema` twice; 3 gens | no error; non-decreasing |
| runner | audit round | finalists + seed | one shared window, one trial each (**A3**) |
| runner | capacity / window arithmetic | synthetic timings; `WF_*` | `pop*gens<=capacity`, `pop>=8k`; ≥330 d, 4 folds, 373 d jitter |

### Edge Cases Checklist
- [x] Zero-trade candidate → tier D, `-inf`, recorded, **charged** — never a silent 0.0
- [x] `sharpe`/`dsr`/`ann_return_pct` all `None` → never coerced to 0.0
- [x] Window too short → `ValueError` caught → tier D with `error=`, ledger row kept
- [x] Population < `EVO_TOURNAMENT_K` → refused at CLI parse time
- [x] Concurrent access → parent-only writer for campaign tables; worker-only ledger rows with busy-timeout + classified retry
- [x] SIGINT mid-generation → `status='aborted'`, resumable, charged trials stay charged
- [x] Every mutation invalid → `EVO_STRICT_MUTATORS` aborts loudly; `data/state.db` absent → created by `ensure_schema`
- [ ] Network failure / permission denied — N/A (no network; SQLite errors propagate unclassified)

---

## Validation Commands

```bash
# Static analysis — no linter and no type checker are configured (KNOWN-LIMITATIONS §8).
.venv/bin/python -m py_compile \
  src/trading_bot/evolution/{__init__,population,oracle,mutate,tournament,runner}.py \
  src/trading_bot/plugins/mutators/{__init__,param_jitter,graph_edit}.py \
  src/trading_bot/config.py src/trading_bot/cli.py

# Tests — new modules, then the 286 baseline
.venv/bin/python -m pytest tests/test_evolution_mutate.py tests/test_evolution_tournament.py \
  tests/test_evolution_oracle.py tests/test_evolution_runner.py -v
.venv/bin/python -m pytest -q          # EXPECT: 286 pre-existing green + the new ones

# Architectural invariants
git diff --stat HEAD -- src/trading_bot/backtest/walkforward.py   # EXPECT: empty
grep -rln "walk_forward_pooled" src/trading_bot/evolution/        # EXPECT: oracle.py only
grep -rn "bruteforce\|core\.score\|run_backtest\|run_graph_backtest\|compute_equity_metrics" \
  src/trading_bot/evolution/                                      # EXPECT: no output
grep -rn "hash(" src/trading_bot/evolution/ | grep -v "graph_hash\|hashlib\|_hash ="  # none
grep -rn "HOLDOUT_\|CAMPAIGN_\|REVIEW_\|UI_" src/trading_bot/evolution/ \
  src/trading_bot/plugins/mutators/                               # EXPECT: no output
grep -c "^EVO_" src/trading_bot/config.py                         # EXPECT: 20
grep -n "PROVISIONAL" src/trading_bot/config.py                   # EXPECT: none after Task 9
git status --porcelain | grep -E "feedback/|ui/|backtest/(trials|benchmark)\.py|framework/"  # none
.venv/bin/python -m trading_bot.cli plugins | grep -E "mutator\.(param-jitter|graph-edit)"

# Database — the queries that matter
sqlite3 data/state.db "SELECT gen_index, trials_cumulative FROM generations \
  WHERE campaign_id='<id>' ORDER BY gen_index;"
sqlite3 data/state.db "SELECT COUNT(*) FROM trial_ledger WHERE campaign_id='<id>';"
sqlite3 data/state.db "SELECT COUNT(*) FROM trial_ledger WHERE campaign_id='<id>' \
  AND end_ms > <train_end_ms>;"    # PROOF the holdout was never touched (A5) — EXPECT 0
```
EXPECT: the last generation's `trials_cumulative` equals the campaign's ledger row count; the holdout query returns **0**.

### Manual Validation
- [ ] `evolve --dry-run --population 8` prints generation 0 and charges **zero** trials (ledger count unchanged)
- [ ] `evolve --calibrate --repeats 3` prints cold/warm seconds and a capacity table; derived values land in `config.py` with the measurement beside them
- [ ] Smoke campaign completes; re-running with the same seed gives the same `best_member_id`
- [ ] Ctrl-C: `campaigns.status == 'aborted'`; `--resume` continues and the ledger total keeps climbing
- [ ] `.gitignore` ignores `data/state.db*` **including `-wal`/`-shm`** (§6). Current `.gitignore` has `data/*.db`, which does **not** match `data/state.db-wal` — that line is P1's; if missing, report it to P1 rather than editing here
- [ ] Overnight run: record `wall_seconds` per generation; a rising trend is thermal throttling, not a code regression, and belongs in the report

---

## Acceptance Criteria / Completion Checklist
- [ ] All 15 tasks complete; every validation command passes; **286 pre-existing tests green** plus the four new modules
- [ ] `walkforward.py` byte-identical; `walk_forward_pooled` imported in exactly one file under `evolution/`; `GateOracle.evaluate` unreachable without a ledger handle
- [ ] Every oracle evaluation has a `trial_ledger` row; `n_trials` passed to the gate is the campaign-cumulative count; no row's `end_ms` exceeds `train_end_ms`
- [ ] `evo_grid()` is one combo equal to `_default_combo(DEFAULT_GRID)` on every axis
- [ ] Both mutators registered with non-empty rationales; jitter provably inside `ParamSpec` bounds; graph edits `validate()` or raise
- [ ] Same seed reproduces the population; `--resume` matches an uninterrupted run
- [ ] `EVO_POPULATION`/`EVO_GENERATIONS` carry the calibration measurement; no `PROVISIONAL` markers remain
- [ ] The three tables created idempotently by `evolution/population.py`; timestamps epoch-ms UTC; `cli evolve` exit codes 0/1/2 as specified; `--report` reprints without evaluating
- [ ] Frozen dataclasses whose docstrings explain *why* non-obvious fields exist; `logging.getLogger("trading_bot")` (INFO for progress/dedup, WARNING for tier-D populations, connection fallbacks and db retries); no `print` outside `cli.py` and the runner's injected `progress`
- [ ] `HoldoutViolation`/`GraphError` propagate; per-task exceptions recorded and counted; `sqlite3.OperationalError` classified via `is_transient_db_error`, never bare-excepted
- [ ] No hardcoded values outside the `EVO_*` block; no threshold duplicated from `walkforward.GATE_*`; **zero engine-core edits** (§12.4) — nothing under `backtest/`, `framework/`, `signals/`, `indicators/`, `regime/`
- [ ] stdlib only; phase report written with measured numbers and degrees of freedom consumed
- [ ] Self-contained — the only flagged ambiguity is P1's exact `trials.py` names, isolated behind `TrialLedger`

## Risks

| Risk | L | I | Mitigation |
|---|---|---|---|
| **Honest DSR makes the gate unpassable in a campaign** | **H** | H if misread as failure | The correct answer, not a bug. Stated in `oracle.py`, the honest-accounting section and the report. The bar is ≈6.9 ann. Sharpe at ~1900 trials, but v0.2.0 also failed at `n_trials=1` (0.742), so `n_obs` binds. Never soften; escalate to P2 breadth and P9's holdout. |
| Evolution finds a window-specific artifact | H | H | Per-generation window jitter; fixed pre-declared audit window; gate as sole oracle; buy-and-hold null; cumulative counting; P9's untouched holdout. |
| Measured cost makes ~1000 models/generation impossible | M-H | M | `--calibrate` measures before sizing and prints the threshold (5.8 s/candidate at 6 workers / 8 h / 30 gens). A smaller population is the honest answer; the PRD predicts it. |
| `state.db` contention; cold caches destroy throughput | L / M | M | Parent-only writer, one small ledger row per multi-second evaluation, `busy_timeout`, classified retry, `db_retries` recorded; persistent pool with `max_tasks_per_child` never set and `--calibrate` reporting cold vs warm. |
| Parallelism silently breaks reproducibility | M | H (campaign unauditable) | All RNG in the parent; pure workers; `blake2b` per-member seeds; member-index merge order; total-order `rank_key`; determinism and resume tested. |
| A contributor adds a second scoring path | M | **Fatal to the phase's purpose** | Required positional ledger; restricted `__all__`; four mechanical review checks including a pytest source scan. |
| P1/P3 APIs differ from the assumed shapes | M | L-M | Isolated behind `TrialLedger` and a small flatten step; mutators consume only `bounds`/`choices`/`kind`/`default` and `validate()`. |
| `graph_edit` collapses the population to zero-trade graphs | M | M | Tier D + `-inf`; diversity guard; re-breed from prior elites; `n_errors`/`n_unique_graphs` make it visible. |
| Mutators reach a small space until P8; overnight run dies at hour 6 | H / M | L / M | Expected — P8's detectors widen the search with zero changes here (§12.4); every generation is committed as it finishes and `--resume` rebuilds `seen_hashes`. |

## Degrees of freedom consumed (§12.6, KNOWN-LIMITATIONS §9)

1. **`population × generations + EVO_FINALISTS + 1` oracle evaluations** — each a `trial_ledger` row, each charged to the campaign's DSR. This is the honest number the PRD's open question asked for.
2. **The `EVO_*` search hyper-parameters** (`EVO_WINDOW_DAYS`, `EVO_ELITES`, `EVO_TOURNAMENT_K`, `EVO_GRAPH_EDIT_SHARE`, `EVO_JITTER_SIGMA`, `EVO_BOOL_FLIP_P`). The ledger does **not** price these. **Fix them once before the first real campaign and never tune them against a result** — that is an unpriced degree of freedom and the most plausible way this phase quietly becomes theater. Any change gets a dated decision comment in `config.py:164-173` style.
3. **Zero** for the cost model, gate thresholds, regime classifier and `WF_*` — untouched by design.
4. **Calibration evaluations** — run under a throwaway campaign id, but state their count and be conservative about whether they belong in the campaign's DSR; including them is the safe direction (A6).

## Notes

- **The most load-bearing idea**: exactly one function here can turn a strategy into a number, and it charges the ledger before it does. The required positional argument, restricted `__all__`, pytest source scan, parent-only writer and pure workers all exist to keep that true under future edits. A change that lets a candidate be scored without a ledger row is wrong regardless of how much faster it is.
- **A1 is the performance decision and the honesty decision at once**: one combo cuts ~870 backtests per candidate to ~27 *and* stops the campaign charging 12 configurations the graph never varied. `_default_combo` is the one place this fails silently — hence the `evo_grid()` pinning test. **A3 is the other easy mistake**: fitness on jittered windows is not comparable across generations, so declaring the audit window in the `campaigns` row *at campaign start* is what stops it being chosen later to flatter a winner.
- **On "1000 simultaneous models"**: honored as a target that measurement arbitrates — if the answer is 96, the plan says 96 and says why. **On "models battling each other"**: k-way tournament selection with elitism, not head-to-head simulated trading; two graphs never trade against each other, they are compared on the same window through the same oracle, which is the only reading the gate can score.
- **`scripts/bruteforce/` is a technique donor, never a second oracle** (§0b, §4.1). This phase borrows its pool pattern, `initializer`, spawn belt-and-braces, read-only connect, per-process cache, mandatory-rationale discipline, trial-counting framing and holdout refusal guard. It borrows **none** of its scoring.
- **Where the PRD and the contract disagree, the contract wins.** Two instances: the PRD's Phase 6 scope says "tournament selection scored only through the gate oracle" without resolving trial counting — §4 resolves it as a cumulative persistent ledger, which this plan implements; and the PRD's monitoring language implies P6 surfaces progress interactively — §2 gives `ui/` to P7, so P6 emits only stdout and `state.db` rows. Separately, §10 Q7 keeps multi-position margin accounting out of scope, so evolved strategies inherit **one open trade per symbol, equal notional** — the known gap between the pivot guide's "trust the signal / keep 2 positions" principle and this build, recorded so results are not read as if overlapping positions had been modelled.
