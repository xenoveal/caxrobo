# Plan: Feedback Loop MVP (v0.3.0 Phase 5)

## Summary

Every past iteration rebuilt its engine from scratch because there was **no standardised workflow
for improving a strategy between versions** — the pivot guide names this as gap #2, ahead of "the
performance itself was still bad". This phase builds that workflow as code: a `Reviewer` plug-in
measuring each closed trade against its own prediction (was the TP the best exit available? was
the SL over-conservative? did this trade's pace clear >50% annual?), a persistent
`review_records` table, a content-addressed `strategy_versions` registry that makes a version
re-runnable, and a forward-test protocol evaluating a candidate on unseen data **through THE GATE
and nothing else**.

The phase's entire engineering risk is one thing: a review system that scores trades is one
refactor away from being a second fitness oracle — the side-channel evaluation that dodges DSR
accounting and turns the gate into theatre. The design below makes that structurally hard rather
than merely discouraged.

## User Story

As the bot's sole operator, I want a closed trade to automatically produce a durable,
machine-readable verdict on its TP, SL and pace against the northstar, and a candidate refinement
to be versioned and forward-tested through the same gate everything else is scored by, so that the
*next* iteration starts from recorded evidence instead of a blank engine.

## Problem → Solution

**Current**: a backtest emits `list[Trade]` and nothing survives the process. No record of whether
a target was reachable, whether a stop was ever approached, or which configuration produced the
trades — `Trade` carries no version, `state.db` carries no review, "improvement" is a human
reading a chart. KNOWN-LIMITATIONS §0c records the consequence: *the search never explored the
entry or the feature set*, because nothing accumulated that would have said where to look.

**Solution**: three modules under `src/trading_bot/feedback/`, one reviewer plug-in, one CLI
subcommand — the pivot guide's Feedback Steps 1–6 as a repeatable procedure. Reviews **inform**;
the walk-forward gate **decides**.

## Metadata

- **Complexity**: **Large** — 5 new source files, 3 new test files, 2 shared files appended;
  ~1 100 new source lines, ~750 test lines; **no new dependencies**.
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md` (Phase 5)
- **Binding contract**: `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md` §3, §4, §6, §7, §8
- **Depends on**: Phase 4 → 3 → 1 (`state.db`, `statestore.connect`, `trials.py`, extended gate).
  **Parallel with Phase 6. Gates Phase 7.**
- **Estimated files**: 10 (7 CREATE, 2 UPDATE, 1 conditional)
- **Test baseline**: **286 tests collected** (`.venv/bin/python -m pytest --collect-only -q`,
  verified 2026-07-27). All 286 stay green; expected new total ≈ 286 + 35.

---

## THE ORACLE BOUNDARY — read before anything else

Contract §4 is absolute and this is the phase most likely to breach it. Put this in the module
docstrings, in these words:

> **A review never runs a backtest. A review never returns a number that ranks a strategy.
> Fitness comes only from `walkforward.walk_forward_pooled`, charged to Phase 1's trial ledger.**

Four structural locks, strongest first:

1. **`ReviewRecord` has no aggregate field.** No `fitness`, `score`, `reward`, `objective`,
   `rank`. Every quantitative field carries units and an axis (`tp_capture_ratio`,
   `sl_headroom_ratio`, `pace_ratio`, `mfe_pct`, `mae_pct`) and the three axes are **never
   combined** anywhere in the package. Test-enforced.
2. **Import ban, test-enforced.** `feedback/records.py`, `feedback/versioning.py` and
   `plugins/reviewers/trade_quality.py` may not import `backtest.walkforward`, `backtest.engine`,
   `backtest.trials`, `framework.execute`, or anything under `evolution/`. They consume
   *already-closed* trades and *stored bars*: they cannot cause a trade to exist, therefore they
   cannot evaluate a strategy. (`backtest.metrics`/`backtest.equity` are pure measurement over
   trades that already happened, consume no trial, and are **allowed**.)
3. **`feedback/protocol.py` is the only module that causes trades to exist**, and only via
   `walk_forward_pooled(..., strategy=graph)` with a ledger-derived `n_trials`. A spy test asserts
   the routing.
4. **Every record is self-labelling.** `span_class ∈ {"in-sample", "forward"}`. In-sample output
   prints `IN-SAMPLE DIAGNOSTIC — NOT EVIDENCE` and the label persists in the row forever, so a
   diagnostic can never later be quoted as a result.

**Divergence noted**: the PRD says "Reviewer plug-in *scoring* each closed trade". Where PRD and
contract disagree the contract wins; "scoring" is per-axis measurement plus a categorical
verdict, never a scalar.

---

## UX Design

**Before**: `cli backtest` prints aggregate metrics and nothing survives the process — no
per-trade verdict, no version, no lineage.

**After**:
```
$ cli review --strategy data/strategies/thin-slice.strategy.json --register
registered strategy_version=8f2c1a9d0b4e6577  parent=--  provenance=manual

$ cli review --version 8f2c1a9d0b4e6577 --start 2023-07-27 --end 2026-04-27
IN-SAMPLE DIAGNOSTIC — NOT EVIDENCE (span was available to tuning)
Symbol   Dir   Outcome  pnl%     MFE%  MAE%  TPcap TPverdict      SLhead SLverdict  Pace
BTCUSDT  long  stop     -1.8500  0.61  2.10  0.00  never-favoured 1.05   hit        loss
BTCUSDT  long  channel   2.4100  4.02  0.44  0.60  good           0.22   over-wide  on-pace
... 47 records written to data/state.db
diagnosis (n=47, 2023-07-27 -> 2026-04-27, in-sample):
  outcome mix  : stop 21 (-1.71%)  channel 14 (+1.98%)  time 11 (+0.12%)  end 1 (-0.40%)
  TP           : median capture 0.41  good 12  left-money 19  target-too-far 16
  SL           : median headroom 0.38  hit 21  over-wide 22  tight 3  ok 1
  pace         : ann_return=+3.1% target=+50.0% n=47 over 1005d sample_adequate=True
  confirmations: volume-breakout coverage=1.00 (DEAD WEIGHT)  macd coverage=0.62
  suggestions  : tighten-stop, narrow-target, drop-confirmation:volume-breakout
  digest       : d41f8a7c9e2b1055

$ cli review --version 8f2c1a9d0b4e6577 --loop --forward-start 2026-04-27 --forward-end 2026-07-26
refine: applied tighten-stop, narrow-target -> child 3ac7fe1188b90d42 (parent 8f2c1a9d…)
forward test 2026-04-27 -> 2026-07-26 via THE GATE (n_trials=1243, cumulative)
  sample_adequacy=False sharpe=True dsr=False max_drawdown=True per_symbol_expectancy=False
  beats_benchmark_return=False beats_benchmark_sharpe=False
  GATE: FAIL          19 forward review records written
```

| Touchpoint | Before | After |
|---|---|---|
| Closed trade | vanishes with the process | a `ReviewRecord` row in `state.db` (PRD "feedback-loop liveness") |
| Strategy identity | none — `Trade` has no version | content-addressed id + lineage |
| "Is this good?" | eyeball an HTML chart | per-axis verdicts + closed-vocabulary diagnosis |
| Refinement | rebuild the engine | `--loop` refines, versions, forward-tests in one command |
| Fitness claim | ad-hoc numbers in reports | only `--forward` prints a gate verdict; all else banner-labelled |

---

## Mandatory Reading

| P | File | Lines | Why |
|---|---|---|---|
| P0 | `_shared-architecture-contract.md` | §3,§4,§6,§7,§8 | Binding: `Reviewer` Protocol, oracle rule, `state.db` conventions, reserved names |
| P0 | `backtest/engine.py` | 108-132 | `Trade` — the whole review input. `entry_ts`/`exit_ts` are bar **OPEN** epoch-ms; `outcome ∈ {"stop","trail","channel","target","time","end"}` |
| P0 | `backtest/engine.py` | 380-453 | Exit loop. **Line 388 `if j <= open_trade["entry_j"]: continue` defines the MFE/MAE window** (Task 3). 435-438 = the repo's no-intra-bar-lookahead reasoning; mirror its tone |
| P0 | `backtest/engine.py` | 354-378 | `close_out`; line 359 `hold_days = (int(ts_trig[j]) - s.ts) / 86_400_000.0` — reuse **exactly** |
| P0 | `data/storage.py` | 22-32, 35-73 | `TIMEFRAME_MS`, module-level `_db_lock`, `connect()` — the pattern this DDL follows |
| P0 | `data/storage.py` | 205-244 | `load_candles` — **both bounds inclusive** (217-218), ascending |
| P0 | `data/statestore.py` | all | Phase 1's. `connect()`; the one-lock-per-database question (Task 0) |
| P0 | `backtest/trials.py` | all | Phase 1's ledger: does `evaluate_forward` read, or also increment? |
| P0 | `framework/graph.py` | all | Phase 3's `StrategyGraph`, `to_dict`/`from_dict`, `SCHEMA_VERSION`, **canonical content hash** |
| P0 | `framework/registry.py` | all | `register(kind, *, name, params, rationale, …)`; `ParamSpec` bounds (`apply_suggestion` clamps to them) |
| P1 | `backtest/walkforward.py` | 225-269, 383 | The gate + `walk_forward_pooled`. Phase 1 made `_evaluate_gate -> dict[str,bool]`, added `benchmark`/`n_trials_used`; Phase 3 added `strategy=`. Read the **current** file |
| P1 | `backtest/equity.py` | 28-51, 173-220 | `daily_returns`/`compute_equity_metrics` — the only honest annualised return |
| P1 | `backtest/metrics.py` | 38-74 | None-for-undefined convention (42-50) |
| P1 | `cli.py` | 128-144, 199-209, 215-234, 356-439 | Subparser, dispatch, `_date_arg`, `_fmt`, `_walkforward_command` |
| P1 | `config.py` | 90-103, 153-183 | Block style: phase name, why the value exists, frozen or sweepable |
| P1 | `tests/test_backtest.py` | 1-58, 83-112, 193-194 | Config-derived tiers (24-33), autouse cache clear (36-48), `make_trade` (51-58), `seed` (83-85), `storage.connect(str(tmp_path/"t.db"))` |
| P2 | `reports/KNOWN-LIMITATIONS.md` | §2, §3, §9 | Why pace is designed as it is. §2: "+68% annualised" was a 90-day extrapolation from 23 trades |
| P2 | `.claude/pivot-guide.md` | 36-45 | Feedback Steps 1-6 verbatim — this phase's spec |
| P2 | `plans/v0.2.0/phase3-sharpe-first-metrics.plan.md` | all | Format bar |

**External documentation: none needed.** Every mechanism is internal (SQLite via the established
`connect()` pattern, stdlib `hashlib`/`json`, existing `equity`/`metrics`). No dependency may be
added — `pandas-ta` disappearing from PyPI is the standing lesson (§1).

---

## Stated Assumptions

- **A1 — Excursions measured on the TRIGGER tier** (`config.SIGNAL_TRIGGER_TIMEFRAME`, 1h today):
  the tier the exit loop actually evaluates. A finer tier judges the engine against exits it never
  had; a coarser one hides excursions it saw. Recorded per record as `review_tf` so a tier shift
  cannot silently change what old records mean. (15m exists for BTC/ETH/SOL only, §0.)
- **A2 — Window is `(entry_ts, exit_ts]`** in bar-open terms:
  `load_candles(..., start_ms=entry_ts + interval_ms, end_ms=exit_ts)`. Both bounds inclusive, so
  this is exactly the bars `j > entry_j … j_exit` of engine.py:388. **The entry bar is excluded** —
  entry fills at its *close*, so its range largely precedes entry and including it inflates
  MFE/MAE.
- **A3 — Post-hoc, therefore not lookahead.** Legitimate only because nothing in the review path
  can reach an entry decision: the package cannot produce trades (lock #2), the reviewer runs on
  closed trades, and no Detector/Confirmation/Policy/Filter imports `feedback`. Goes verbatim into
  `trade_quality.py`'s docstring.
- **A4 — A single trade's return is never annualised.** `pace_ratio` compares realized `pnl_pct` to
  the target's compounded pro-rata over the same holding days. The only annualised figure the
  phase produces is `diagnose()`'s, via `compute_equity_metrics`, with a sample-adequacy flag.
  KNOWN-LIMITATIONS §2 is the reason.
- **A5 — `version_id` = Phase 3's canonical graph content hash, 16 hex.** Content addressing makes
  registration idempotent and makes "the mutator re-invented an existing variant" observable. A
  free-text `label` carries human names.
- **A6 — Reproducibility is recorded and checkable, not pinned.** `config.py` is mutable global
  state and this phase does not change that; a version stores a `config_snapshot` + hash and
  `verify_reproducible()` diffs live config against it. A real residual hole, named not hidden.
- **A7 — The reviewer plug-in is pure; the caller persists.** §3 *permits* `Reviewer` to write
  records, it does not require the plug-in to. Narrowing keeps it DB-free and unit-testable.
- **A8 — Forward-test aggregates are not persisted.** §6 fixes the table list and allots none;
  inventing one risks colliding with Phase 6's `campaigns`/`population_members`. The forward run's
  per-trade records persist with `span_class="forward"`; the verdict is reproducible from
  (version_id, span).
- **A9 — One trade per symbol, equal notional, alert-only** (§10.7). No review field assumes
  overlapping positions.
- **A10 — `Trade.strategy_version` is populated at the review/forward seam**, not in the engine
  (`framework/execute.py` is Phase 3's). If Phase 3/4 already sets it, prefer that. Unstamped
  trades record under `config.REVIEW_LEGACY_VERSION_ID = "legacy-engine"`, never `""`.

---

## Patterns to Mirror

All snippets from the working tree, verified 2026-07-27.

### DB_MODULE_PATTERN
```python
# SOURCE: src/trading_bot/data/storage.py:29-73
_db_lock = threading.Lock()   # serializes all DB access across callers sharing one connection

def connect(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS ohlcv (... PRIMARY KEY (symbol, timeframe, ts))""")
    conn.commit()
    return conn
```
**Adaptation**: the lock belongs to the *database*, not the table-owning module. `state.db` is
opened by `statestore.connect()`, so a second `threading.Lock()` in `records.py` would serialize
nothing. Use the lock Phase 1 publishes; if it publishes only `connect()`, use
`statestore._db_lock` with a comment saying why a local lock would be wrong. Do **not** edit
`statestore.py`.

### IDEMPOTENT_WRITE
```python
# SOURCE: src/trading_bot/data/storage.py:91-101
    with _db_lock:
        conn.executemany("INSERT OR REPLACE INTO ohlcv (...) VALUES (?,?,?,?,?,?,?,?)", [...])
        conn.commit()
```
Both new tables use `INSERT OR REPLACE` on a content-derived primary key, so re-running `review`
over the same span is a no-op, not a duplicate.

### CONTENT_FINGERPRINT — why ids are content-derived
```python
# SOURCE: src/trading_bot/backtest/engine.py:158-174
def _fingerprint(df: pd.DataFrame) -> tuple:
    """Bar count and end timestamps alone are not enough: distinct fixtures
    routinely share them while holding different prices, which would collide."""
```
Same reasoning for `record_id`/`version_id`: two runs meaning the same thing collapse; two that
differ cannot collide.

### INTERVAL_ASSERTION — the guarantee every new data path must keep (§1)
```python
# SOURCE: src/trading_bot/backtest/engine.py:211-222
    expected = storage.TIMEFRAME_MS[timeframe]
    observed = int(np.median(np.diff(ts)))
    if observed != expected:
        raise ValueError(f"{symbol} {role} series spacing is {observed} ms but config names "
                         f"timeframe {timeframe!r} ({expected} ms). ...")
```
The excursion loader repeats this on its window; no numpy needed for a short window — compare
consecutive diffs and raise the same shape of message.

### NONE_FOR_UNDEFINED
```python
# SOURCE: src/trading_bot/backtest/metrics.py:41-50
    if n == 0:
        return {"n_trades": 0, "win_rate": None, ...}
```
`mfe_pct`, `mae_pct`, `tp_capture_ratio`, `sl_headroom_ratio`, `pace_ratio` are `None` when
undefined — never NaN, never 0.0-as-missing, never an exception. Verdicts are `"n/a"` likewise.
`logger = logging.getLogger("trading_bot")` (metrics.py:10-12); WARNING for a skipped review, INFO
for counts written, never `print` outside `cli.py`.

### HOLDING_TIME — reuse, do not re-derive
```python
# SOURCE: src/trading_bot/backtest/engine.py:359
        hold_days = (int(ts_trig[j]) - s.ts) / 86_400_000.0
```
`holding_days = (trade.exit_ts - trade.entry_ts) / 86_400_000.0` — the same formula the funding
charge uses, so review and cost accounting cannot disagree.

### CLI_SUBCOMMAND_PATTERN
```python
# SOURCE: src/trading_bot/cli.py:128-144, 199-209, 356-358, 437-439
    wf_parser = subparsers.add_parser("walkforward", help="...")
    wf_parser.add_argument("--symbol", action="append", help="... (repeatable); default all symbols")
    wf_parser.add_argument("--start", type=_date_arg, help="UTC start YYYY-MM-DD (default: BACKFILL_START)")
...
    elif args.command in ("backtest", "walkforward"):
        conn = connect(args.db)
        symbols = args.symbol if args.symbol else config.SYMBOLS
        start_ms = args.start if args.start else config.date_to_ms(config.BACKFILL_START)
        end_ms = args.end if args.end else int(time.time() * 1000)
        ...
        conn.close(); sys.exit(exit_code)
...
def _fmt(v, spec=".4f") -> str: return format(v, spec) if v is not None else "--"
...
    print(f"GATE: {'PASS' if result.passed else 'FAIL'}"); return 0 if result.passed else 1
```

### CONFIG_BLOCK_STYLE
```python
# SOURCE: src/trading_bot/config.py:90-103
# Phase 6: fade re-qualification. The ranging sleeve is kept behind an explicit
# switch so a DROP verdict is a recorded decision rather than a code deletion.
FADE_ENABLED = False
```

### TEST_STRUCTURE
```python
# SOURCE: tests/test_backtest.py:24-58, 83-85
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME       # tier-derived, NEVER hardcoded
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]; START = 1_700_000_000_000

@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    engine.clear_caches(); yield; engine.clear_caches()

def make_trade(pnl, regime="trending", pattern="flag", exit_ts=None):
    exit_ts = START + D_TRIG if exit_ts is None else exit_ts
    return Trade(symbol=SYMBOL, regime=regime, pattern=pattern, direction="long",
                 entry_ts=START, entry=100.0, stop=99.0, target=101.0, exit_ts=exit_ts,
                 exit_price=100.0, outcome="target", pnl_pct=pnl, volume_high=False)

def seed(conn, timeframe, rows, start=START, interval=D_SET):
    storage.upsert_candles(conn, SYMBOL, timeframe,
                           [[start + i * interval] + list(r) for i, r in enumerate(rows)])
```
Phase 5 fixtures need **two** connections: `storage.connect(str(tmp_path/"ohlcv.db"))` and
`statestore.connect(str(tmp_path/"state.db"))`. Never touch the real `data/*.db` from a test.

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `feedback/__init__.py` | CREATE | Thin re-exports (`ReviewRecord`, `Diagnosis`, `StrategyVersion`, `run_loop_iteration`); no logic |
| `feedback/records.py` | CREATE | `ReviewRecord`, `ReviewContext`, `review_records` DDL + persistence, `diagnose()` + `Diagnosis` + suggestion vocabulary |
| `feedback/versioning.py` | CREATE | `StrategyVersion`, `strategy_versions` DDL, `register_version`, `load_graph`, `lineage`, config snapshot, `verify_reproducible`, `stamp_trades` |
| `feedback/protocol.py` | CREATE | `ForwardSpan` + holdout guard, `evaluate_forward`, `apply_suggestion`, `run_loop_iteration` |
| `plugins/reviewers/trade_quality.py` | CREATE | The `Reviewer` plug-in, registered `reviewer.trade-quality` |
| `plugins/reviewers/__init__.py` | CREATE (cond.) | Only if Phase 3 did not; `load_all()` uses `pkgutil` and a missing `__init__.py` can hide the subpackage |
| `config.py` | UPDATE | Append the Phase 5 block (`TARGET_ANN_RETURN`, `REVIEW_*`); modify nothing existing |
| `cli.py` | UPDATE | `review` subparser + `_review_command`, in phase order after Phase 4's `graph-backtest` |
| `tests/test_feedback_records.py` | CREATE | Excursions, verdicts, persistence, diagnosis, oracle boundary |
| `tests/test_feedback_versioning.py` | CREATE | Content addressing, lineage, config-snapshot completeness, drift |
| `tests/test_feedback_protocol.py` | CREATE | Span discipline, gate routing, suggestion bounds, full loop, CLI |
| `.gitignore` | UPDATE (cond.) | §6 requires `data/state.db*`; today's `data/*.db` does **not** match `state.db-wal`. Phase 1 owns it — if still missing, add the line and say so in the report |

## NOT Building

Scope creep lives almost entirely in Step 4. The MVP is a *thin* loop. Explicitly out:

- **No LLM in the loop.** Step 4 is deterministic aggregation producing a **closed-vocabulary**
  `Diagnosis` — not a model writing code, not free text, not a prompt. Vocabulary test-enforced.
- **No new fitness function, no reward shaping, no RL.** Population fitness, tournaments and
  mutation are Phase 6 (`evolution/`); this phase creates nothing there, nothing like `oracle.py`.
- **No general graph mutation.** `apply_suggestion` is one deterministic bounded parameter step
  driven by a diagnosis, no RNG. Phase 6's mutators are stochastic and population-driven; if
  Phase 6 lands first with an equivalent bounded edit, delegate.
- **No post-exit counterfactuals** ("would a wider stop have won?"). Needs bars after `exit_ts` and
  an arbitrary horizon, and it is the easiest place for a review metric to become a hidden fitness
  function. Deferred with that reason in the docstring.
- **No trades table** — §6 fixes the table list; this phase owns exactly `review_records` and
  `strategy_versions`. Trades are re-derivable from (version, span). **No forward-test result
  table** (A8). **No UI** (Phase 7 owns `ui/`; `diagnose()` returns a dataclass precisely so the UI
  needs nothing from here).
- **No live/paper trade ingestion** — Steps 1–2 are satisfied by the simulated engine; alert-only
  (§10), nothing places orders.
- **No changes to** `walkforward.py`, `engine.py`, `equity.py`, `metrics.py`, `statestore.py`,
  `trials.py`, or anything under `framework/`. If a change there seems needed, the phase boundary
  is wrong — consume the published interface.
- **No SQLite retry/backoff layer.** WAL + one lock + idempotent upsert is the whole concurrency
  story (`storage.is_transient_db_error` exists if a later phase needs it).
- **No lowering of any gate threshold or sample floor** to make a forward test pass
  (KNOWN-LIMITATIONS §4: "explicitly not done, and should not be").

---

## Step-by-Step Tasks

### Task 0: Reconnaissance of the interfaces this phase consumes
- **ACTION**: before writing anything, read the *actual* published interfaces of Phases 1/3/4 and
  record the answers in the phase report. This plan is written against the contract; the code is
  the truth.
- **ANSWER**: (1) `statestore.connect()` signature; is the lock public or `_db_lock`? does
  `config.STATE_DB_PATH` exist? (2) `trials.py`: how is an evaluation recorded and a cumulative
  count read — **and does `walk_forward_pooled` increment the ledger itself?** Exactly one place
  may increment per evaluation. (3) `walkforward`: `strategy=` present; `_evaluate_gate ->
  dict[str,bool]`; `WalkForwardResult` has `gate`/`benchmark`/`n_trials_used`/`passed`; does it
  expose the OOS trade list? (4) `graph.py`: canonical hash name, `SCHEMA_VERSION`, graph-level
  `max_hold_bars`? (5) `registry.register` signature + `ParamSpec` fields
  (`default`/`bounds`/`choices`/`kind`). (6) `ReviewContext` is **assigned to Phase 5** by
  contract §3 and defined in `feedback/records.py` (Task 2) — confirm Phase 3 did not also define
  one in `contracts.py`; if it did, Phase 5's is canonical and the duplicate goes in the report. (7) does `Trade` have `planned_rr`, `confirmations`,
  `strategy_version` with defaults, and does `execute.py` populate the last? (8) does
  `plugins/reviewers/__init__.py` exist? (9) does `.gitignore` cover `data/state.db*`?
- **GOTCHA**: on any contradiction the **contract** is authoritative on names, the **code** on
  signatures. Record the divergence; never silently adapt.
- **VALIDATE**: `.venv/bin/python -m pytest -q` on the clean tree; record the collected count as
  this phase's regression baseline (≥ 286).

### Task 1: `config.py` — the Phase 5 block
- **ACTION**: append one clearly-headed block at the end. Touch nothing existing.
- **IMPLEMENT**:
  ```python
  # ---------------------------------------------------------------------------
  # Phase 5 (v0.3.0): feedback loop. REVIEW_* are DIAGNOSTIC thresholds — they
  # label a closed trade so a human or Phase 6's mutators can read the pattern.
  # They never select a strategy: THE GATE is the only fitness oracle (§4). None
  # is a walk-forward grid axis; none may become one.
  # ---------------------------------------------------------------------------
  TARGET_ANN_RETURN = 0.50   # the northstar, stated once; everything reads it from here

  # None => resolve SIGNAL_TRIGGER_TIMEFRAME at CALL time (never import time), so a
  # tier shift moves the review with the engine — same discipline as FADE_ENABLED.
  REVIEW_TIMEFRAME = None
  # Minimum fraction of expected bars present before MFE/MAE are trusted. A gap
  # understates both, and an understated MAE reads as "the stop was over-wide" —
  # exactly backwards.
  REVIEW_MIN_BAR_COVERAGE = 0.90

  REVIEW_TP_CAPTURE_GOOD = 0.50       # realized favourable / MFE at or above this = good exit
  REVIEW_TP_UNREACHABLE_FRAC = 0.50   # MFE below this fraction of target distance = never approached
  REVIEW_SL_SLACK_MAX = 0.50          # MAE/stop distance below this = less than half the risk used
  REVIEW_SL_NEAR_MISS = 0.90          # at or above, without stopping out = the stop barely held

  # Pace (KNOWN-LIMITATIONS §2: a +68% headline was a 90-day extrapolation from 23
  # trades). A pace CLAIM needs BOTH floors; below them diagnose() reports
  # "insufficient-sample" and emits NO refinement suggestions.
  REVIEW_PACE_MIN_TRADES = 30   # deliberately equal to WF_MIN_TRADES today
  REVIEW_PACE_MIN_DAYS = 180

  REVIEW_STOP_DOMINANCE = 0.50        # fraction of trades exiting on the stop
  REVIEW_TIME_DOMINANCE = 0.33        # fraction exiting on the time stop
  REVIEW_DEAD_WEIGHT_COVERAGE = 0.98  # a Confirmation passing on ~every trade carries no
                                      # information (§0c: volume is computed but gates nothing)

  REVIEW_REFINE_STEP_FRAC = 0.25      # one bounded step, as a fraction of a ParamSpec's range
  REVIEW_FORWARD_MIN_DAYS = 30        # a shorter forward window is refused, not reported
  REVIEW_LEGACY_VERSION_ID = "legacy-engine"
  ```
- **MIRROR**: CONFIG_BLOCK_STYLE. **GOTCHA**: `TARGET_ANN_RETURN` and `REVIEW_*` are the only names
  this phase may add (§7); use a literal 30 rather than aliasing `WF_MIN_TRADES` so the two stay
  independently auditable.
- **VALIDATE**: `python -m py_compile src/trading_bot/config.py`.

### Task 2: `feedback/records.py` — `ReviewRecord`, `ReviewContext`, schema, persistence
- **ACTION**: create the package and the record type with its table.
- **IMPLEMENT**:
  - Docstring: the oracle boundary verbatim; this module never runs a backtest; all timestamps
    epoch ms UTC; `record_id` is content-derived so re-review is idempotent.
  - `ReviewContext` — **Phase 5 owns it** (contract §3, settled after this plan was drafted):
    define it here. Frozen dataclass `conn` (OHLCV connection, **read only**), `review_tf`,
    `strategy_version`, `span_class`, `target_ann_return`, `created_ts` (one value per batch).
  - `SPAN_CLASSES = ("in-sample", "forward")`, `TP_VERDICTS`, `SL_VERDICTS`, `PACE_VERDICTS` as
    module tuples so tests and Phase 7 can enumerate them.
  - `ReviewRecord` frozen dataclass, fields in DDL order. **No aggregate field.**
    ```sql
    CREATE TABLE IF NOT EXISTS review_records (
        record_id TEXT PRIMARY KEY,
        strategy_version TEXT NOT NULL, reviewer TEXT NOT NULL, span_class TEXT NOT NULL,
        symbol TEXT NOT NULL, regime TEXT NOT NULL, pattern TEXT NOT NULL,
        direction TEXT NOT NULL, outcome TEXT NOT NULL,
        entry_ts INTEGER NOT NULL, exit_ts INTEGER NOT NULL, created_ts INTEGER NOT NULL,
        entry REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL,
        exit_price REAL NOT NULL, pnl_pct REAL NOT NULL, holding_days REAL NOT NULL,
        review_tf TEXT NOT NULL, bars_reviewed INTEGER NOT NULL, bars_expected INTEGER NOT NULL,
        mfe_pct REAL, mae_pct REAL,
        target_distance_pct REAL NOT NULL, stop_distance_pct REAL NOT NULL,
        tp_capture_ratio REAL, tp_verdict TEXT NOT NULL,
        sl_headroom_ratio REAL, sl_verdict TEXT NOT NULL,
        pace_ratio REAL, pace_verdict TEXT NOT NULL,
        confirmations TEXT NOT NULL,   -- JSON array of names (Phase 4's Trade field)
        planned_rr REAL,               -- Phase 4's Trade field; NULL for legacy trades
        notes TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_review_version ON review_records (strategy_version, exit_ts);
    CREATE INDEX IF NOT EXISTS idx_review_exit ON review_records (exit_ts);
    ```
  - `ensure_schema(conn)` — DDL under the state-db lock + commit; called at the top of every public
    read/write so ordering never matters.
  - `_record_id(strategy_version, reviewer, span_class, symbol, entry_ts, exit_ts, review_tf)` =
    `sha256("|".join(...))[:24]`. `span_class` is part of the identity **on purpose**: an in-sample
    diagnostic and a forward test of the same trade are different claims; both must survive.
  - `insert_records(conn, records) -> int` (`INSERT OR REPLACE`, `executemany`, one commit);
    `load_records(conn, *, strategy_version=None, symbol=None, span_class=None, start_ms=None,
    end_ms=None)` building WHERE the way `storage.load_candles` does (params list, `ORDER BY
    exit_ts ASC`); `_to_row`/`_from_row`.
- **MIRROR**: DB_MODULE_PATTERN, IDEMPOTENT_WRITE, CONTENT_FINGERPRINT, NONE_FOR_UNDEFINED.
- **IMPORTS**: `dataclasses`, `hashlib`, `json`, `logging`, `sqlite3`; `config`; `data.statestore`.
  **Nothing** from `backtest.engine`/`walkforward`/`trials`, `framework.execute`, `evolution`.
  `Trade` is accepted **duck-typed** (attribute access only) — comment this; it is what makes the
  import ban possible.
- **GOTCHA**: all timestamps epoch-ms integers; never `datetime`, never ISO strings in the DB.
  `holding_days` is a derived float measurement, not a timestamp.
- **VALIDATE**: `pytest tests/test_feedback_records.py::TestSchema -v`.

### Task 3: `plugins/reviewers/trade_quality.py` — excursions (MFE/MAE)
- **ACTION**: create the plug-in and implement the window. The subtle part of the phase.
- **IMPLEMENT**:
  - Docstring must contain in full: (1) the no-lookahead argument (A3) and what would break it
    (any Detector/Confirmation/Policy/Filter importing `feedback`); (2) the window definition and
    citation — *"the window is the bar set the engine's exit loop actually evaluated:
    `engine.py:388` skips `j <= entry_j`, so the entry bar — whose close is the entry price — is
    excluded; including it would count pre-entry intra-bar range as excursion and bias every TP
    verdict toward 'left money'"*; (3) the tier choice (A1).
  - `_load_window(ctx, trade)`:
    ```python
    interval = storage.TIMEFRAME_MS[ctx.review_tf]
    start = trade.entry_ts + interval        # exclude the entry bar (A2)
    end = trade.exit_ts                      # inclusive: the exit bar IS evaluated
    rows = storage.load_candles(ctx.conn, trade.symbol, ctx.review_tf, start_ms=start, end_ms=end)
    bars_expected = max(0, (trade.exit_ts - trade.entry_ts) // interval)
    ```
    Then INTERVAL_ASSERTION over consecutive `ts` diffs: **median** spacing ≠
    `TIMEFRAME_MS[review_tf]` → `ValueError` in `engine._assert_interval`'s message shape. A single
    missing candle must not raise (coverage's job); a wholesale mismatch must.
  - `measure_excursions(ctx, trade) -> (mfe_pct, mae_pct, bars_reviewed, bars_expected)`:
    ```python
    sign = 1.0 if trade.direction == "long" else -1.0
    # Favourable extreme: highest high for a long, lowest low for a short; adverse is
    # the opposite. Both POSITIVE fractions of entry, clipped at 0.0 — "never went
    # favourable" is 0% excursion, not a negative one.
    mfe_pct = max(0.0, max(sign * (h_or_l - trade.entry) / trade.entry for ...))
    mae_pct = max(0.0, max(-sign * (l_or_h - trade.entry) / trade.entry for ...))
    ```
    If `bars_expected == 0` or `bars_reviewed / bars_expected < REVIEW_MIN_BAR_COVERAGE`: return
    `(None, None, …)` + `logger.warning` with symbol, entry_ts, coverage.
- **IMPORTS**: `config`, `data.storage`. No pandas — plain tuples keep the arithmetic
  hand-checkable, the same reason `equity.py` avoids it. This also keeps the phase clear of
  pandas 3.0.3's unsafe `Series.pct_change()`: every return in this phase — excursions, capture,
  headroom, pace — is written explicitly as `a / b - 1.0` (or `(x - entry) / entry`), never via a
  pandas percentage-change helper.
- **GOTCHA #1**: `entry_ts`/`exit_ts` are bar **OPEN** times (`Signal.ts` = "epoch-ms of the
  trigger bar"; `exit_ts=int(ts_trig[j])`, engine.py:371), so `entry_ts + interval` is the first
  bar the engine could have exited on. A one-bar error here is the most likely defect in the phase;
  Task 12 pins it with a favourable-wick fixture.
- **GOTCHA #2**: `outcome == "end"` trades still satisfy `exit_ts > entry_ts` (engine.py:543
  requires `last_j > entry_j`), but guard `bars_expected == 0` anyway for synthetic input.
- **GOTCHA #3**: resolve `review_tf` from `config.REVIEW_TIMEFRAME or
  config.SIGNAL_TRIGGER_TIMEFRAME` **at call time**.
- **VALIDATE**: `pytest tests/test_feedback_records.py::TestExcursions -v`.

### Task 4: `trade_quality.py` — the three axes, the record, registration
- **ACTION**: same file. Turn excursions into TP/SL/pace and register.
- **IMPLEMENT**:
  ```python
  sign = 1.0 if trade.direction == "long" else -1.0
  realized_pct        = sign * (trade.exit_price - trade.entry) / trade.entry   # gross
  target_distance_pct = abs(trade.target - trade.entry) / trade.entry
  stop_distance_pct   = abs(trade.entry - trade.stop) / trade.entry
  tp_capture_ratio    = None if (mfe_pct is None or mfe_pct <= 0) else max(0.0, realized_pct)/mfe_pct
  sl_headroom_ratio   = None if (mae_pct is None or stop_distance_pct <= 0) else mae_pct/stop_distance_pct
  ```
  **TP verdicts** (first match wins): `"n/a"` if `mfe_pct is None`; `"never-favoured"` if
  `mfe_pct <= 0` (an entry problem, not a TP problem); `"target-too-far"` if
  `mfe_pct < target_distance_pct * REVIEW_TP_UNREACHABLE_FRAC`; `"left-money"` if
  `tp_capture_ratio < REVIEW_TP_CAPTURE_GOOD`; else `"good"`.

  **SL verdicts**: `"n/a"` if ratio is None; `"hit"` if `outcome in ("stop","trail")` (whether it
  *should* have bound is a post-exit question this phase refuses); `"over-wide"` if
  `ratio < REVIEW_SL_SLACK_MAX` — less than half the budgeted risk was ever used, so R:R was
  understated at entry; `"tight"` if `ratio >= REVIEW_SL_NEAR_MISS`; else `"ok"`.

  **Pace** — the honest per-trade version and nothing more:
  ```python
  holding_days = (trade.exit_ts - trade.entry_ts) / 86_400_000.0   # engine.py:359
  # Required return over the SAME holding days at the northstar rate, COMPOUNDED —
  # consistent with equity.compute_equity_metrics' equity ** (365 / n) - 1.
  # Compounded is ~18% LESS demanding than linear pro-rata at 30 days; chosen for
  # consistency with the aggregate path, not for flattery, and recorded so it is
  # mistaken for neither.
  required   = (1.0 + ctx.target_ann_return) ** (holding_days / 365.0) - 1.0
  pace_ratio = None if (holding_days <= 0 or required <= 0) else trade.pnl_pct / required
  ```
  Verdicts: `"loss"` if `pnl_pct <= 0`; `"on-pace"` if `>= 1`; `"behind"` if `0 < r < 1`; else
  `"n/a"`. **The docstring must say, in these words:** *`pace_ratio` is a per-trade diagnostic with
  no statistical significance and it ignores idle capital — at most one position per symbol is
  held, so a portfolio of `pace_ratio > 1` trades does **not** imply >50% annual. The only honest
  pace statement in this codebase is `diagnose()`'s, computed by `equity.compute_equity_metrics`
  over a calendar-complete series with a sample-adequacy flag. Annualising a single trade over its
  holding days is the error KNOWN-LIMITATIONS §2 records.*
  - `review(trade, context, **params) -> ReviewRecord` assembles the record; `notes` is compact
    `key=value` (e.g. `"mfe_bars=37 coverage=1.00"`), never prose. Pure — no DB writes (A7).
  - Registration (exact signature from Task 0):
    ```python
    @register("reviewer", name="trade-quality",
              params={"tp_capture_good": ParamSpec(default=config.REVIEW_TP_CAPTURE_GOOD,
                                                   bounds=(0.1, 0.9), kind="float"),
                      "sl_slack_max":    ParamSpec(default=config.REVIEW_SL_SLACK_MAX,
                                                   bounds=(0.1, 0.9), kind="float")},
              rationale="v0.2.0 produced trades and no record of whether their exits were any "
                        "good; KNOWN-LIMITATIONS §0c shows the search never learned from them. "
                        "Measures TP capture against max favourable excursion, SL headroom "
                        "against max adverse excursion, and pace against TARGET_ANN_RETURN — "
                        "diagnostics only; THE GATE decides.")
    ```
    `rationale` must be non-empty (§3: "an unmotivated strategy in a 10 000-combo sweep is just
    noise with a name").
- **GOTCHA**: exposing `params` lets a Mutator jitter the *review thresholds* — that changes
  labels, never fitness. Note it in the docstring; only two are exposed for that reason.
- **VALIDATE**: `pytest tests/test_feedback_records.py::TestVerdicts tests/test_feedback_records.py::TestPace -v`;
  `cli plugins | grep trade-quality`.

### Task 5: `records.py` — `diagnose()`, `Diagnosis`, closed suggestion vocabulary
- **ACTION**: same file as Task 2. Pivot-guide **Step 4**, scoped deliberately small.
- **IMPLEMENT**:
  ```python
  SUGGESTIONS = ("insufficient-sample", "widen-target", "narrow-target",
                 "widen-stop", "tighten-stop", "raise-max-hold", "lower-max-hold",
                 "drop-confirmation")   # emitted parametrised: "drop-confirmation:<name>"
  ```
  Closed by design: Phase 6's mutators and Phase 7's UI consume it mechanically and it can never
  become free text. Validation is `s.split(":", 1)[0] in SUGGESTIONS`.
  - `Diagnosis` frozen dataclass: `n_records`, `span_start_ms`, `span_end_ms`, `span_class`,
    `strategy_version`, `by_outcome`, `outcome_mean_pnl`, `dominant_outcome`,
    `tp_verdicts`/`sl_verdicts`/`pace_verdicts`, `median_tp_capture`, `median_sl_headroom`,
    `confirmation_coverage`, `confirmation_pnl_delta`, `pace`, `suggestions`, `digest`.
    Docstring: **"A structured description of what these trades did. It contains no fitness value
    and cannot be ordered against another Diagnosis. `confirmation_pnl_delta` is in-sample and
    undeflated — a hint about where to look, never evidence."**
  - `diagnose(records, *, start_ms, end_ms) -> Diagnosis`:
    - outcome counts + per-outcome mean `pnl_pct`; `dominant_outcome` = max by count; verdict
      histograms; medians over non-None ratios (`statistics.median`).
    - `confirmation_coverage[name]` = fraction of records containing it;
      `confirmation_pnl_delta[name]` = mean pnl with − without (`None` if either side empty).
      Coverage ≥ `REVIEW_DEAD_WEIGHT_COVERAGE` is dead weight: it passed on essentially everything,
      so it discriminated nothing — the same failure §0c records for volume.
    - **pace, the one aggregate**: build pseudo-trades from records (`exit_ts`, `pnl_pct` are all
      `daily_returns` needs) and call `compute_equity_metrics(pseudo, start_ms, end_ms)`. Report
      `{"ann_return_pct", "target_ann_return", "n_trades", "n_days", "sample_adequate"}` with
      `sample_adequate = n_records >= REVIEW_PACE_MIN_TRADES and n_days >= REVIEW_PACE_MIN_DAYS`.
      **Deliberately omit Sharpe/DSR** — they belong to the gate, and printing them here would make
      `review` look like an evaluation. Select keys explicitly; never merge the dict.
    - `suggestions`, deterministic, fixed order:
      1. `n_records < REVIEW_PACE_MIN_TRADES` → `("insufficient-sample",)` **and stop.** No
         refinement from a noise sample — this single rule is what stops Step 4 becoming an
         overfitting engine.
      2. `time / n >= REVIEW_TIME_DOMINANCE` → `"raise-max-hold"`.
      3. `stop / n >= REVIEW_STOP_DOMINANCE` and `median_sl_headroom >= 1.0` → `"widen-stop"`; elif
         `sl_verdicts["over-wide"] / n > 0.5` → `"tighten-stop"`. Mutually exclusive by
         construction; a test asserts they never co-occur.
      4. `tp_verdicts["target-too-far"] / n > 0.5` → `"narrow-target"`; elif `median_tp_capture <
         REVIEW_TP_CAPTURE_GOOD` → `"widen-target"`.
      5. each name at coverage ≥ `REVIEW_DEAD_WEIGHT_COVERAGE`, when more than one confirmation
         exists → `"drop-confirmation:<name>"`.
    - `digest` = sha256 of canonical JSON of all fields except `digest`, 16 hex — a child version's
      provenance stores it, so "which diagnosis motivated this version" stays answerable.
  - `diagnosis_to_dict(d)` for Phase 7 and provenance.
- **IMPORTS**: adds `statistics` and `from trading_bot.backtest.equity import
  compute_equity_metrics` — **allowed** (measures trades that already happened, consumes no trial).
  Comment that, or a future reader will "tidy up" the import-ban test.
- **VALIDATE**: `pytest tests/test_feedback_records.py::TestDiagnosis -v`.

### Task 6: `feedback/versioning.py` — registry and lineage
- **ACTION**: create the version registry over `state.db`.
- **IMPLEMENT**:
  - `StrategyVersion` frozen: `version_id`, `parent_id: str | None`, `label`, `graph_json`,
    `graph_hash`, `schema_version`, `config_hash`, `config_snapshot`, `provenance`, `created_ts`.
    ```sql
    CREATE TABLE IF NOT EXISTS strategy_versions (
        version_id TEXT PRIMARY KEY, parent_id TEXT, label TEXT NOT NULL,
        graph_json TEXT NOT NULL, graph_hash TEXT NOT NULL, schema_version TEXT NOT NULL,
        config_hash TEXT NOT NULL, config_snapshot TEXT NOT NULL,  -- JSON, sorted keys
        provenance TEXT NOT NULL, created_ts INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_version_parent ON strategy_versions (parent_id);
    CREATE INDEX IF NOT EXISTS idx_version_created ON strategy_versions (created_ts);
    ```
  - `version_id_for(graph)` = **Phase 3's canonical content hash**, 16 hex (A5). If `graph.py`
    publishes a hash function, call it; if only `to_dict`, use
    `sha256(json.dumps(g.to_dict(), sort_keys=True, separators=(",", ":")))` **and** record in the
    docstring that a second canonicalisation now exists, to be consolidated into `graph.py`
    (Risks).
  - `register_version(conn, graph, *, parent_id=None, label="", provenance=None,
    created_ts=None)`: if the id exists, **return the existing row unchanged** and log INFO
    ("re-registered identical graph") — re-deriving a variant is information, not a duplicate.
    Refuse `parent_id == version_id` (`ValueError`). `provenance` defaults to
    `{"source": "manual"}`; callers pass `{"source": "diagnosis", "diagnosis_digest": …,
    "suggestions": [...]}` or `{"source": "mutator:<name>", "campaign": …}`. `created_ts` defaults
    to `int(time.time() * 1000)`, injectable for deterministic tests.
  - `get_version`, `list_versions(conn, *, parent_id=None, limit=None)` (`ORDER BY created_ts`),
    `load_graph(conn, version_id)` via Phase 3's `from_dict` — raise `KeyError` on unknown ids; a
    silent `None` would let a forward test run the wrong strategy.
  - `lineage(conn, version_id)` root-first with a `seen` set and a hard iteration cap so a corrupt
    row cannot hang the loop.
  - `stamp_trades(trades, version_id)` via `dataclasses.replace` (`Trade` is frozen, the field is
    appended with a default, §5). Skip entirely if Phase 3/4 already populates it (A10).
- **GOTCHA**: store the **canonical** JSON you hashed, so text and hash cannot disagree.
- **VALIDATE**: `pytest tests/test_feedback_versioning.py::TestRegister tests/test_feedback_versioning.py::TestLineage -v`.

### Task 7: `versioning.py` — config snapshot and `verify_reproducible`
- **ACTION**: same file. Confront the mutable-global reproducibility hole explicitly.
- **IMPLEMENT**:
  ```python
  # config.py is MUTABLE MODULE-LEVEL GLOBAL STATE. A strategy graph does not carry
  # the cost model, the tier timeframes, the ATR stop multiple or the regime
  # thresholds — yet each changes the trades the graph produces. This snapshot
  # records their values AT VERSION-CREATION TIME so a later re-run is CHECKABLE. It
  # does NOT pin them: re-running version X under a different FEE_PCT silently
  # produces different trades and verify_reproducible() is the only thing that will
  # tell you. Named as a residual hole rather than papered over.
  VERSION_CONFIG_KEYS = (
      "FEE_PCT", "SLIPPAGE_PCT", "FUNDING_PCT_PER_DAY",
      "REGIME_TIMEFRAME", "SIGNAL_PATTERN_TIMEFRAME", "SIGNAL_TRIGGER_TIMEFRAME",
      "ADX_PERIOD", "ADX_TREND_THRESHOLD", "ATR_PERCENTILE_WINDOW",
      "ATR_EXTREME_PERCENTILE", "REGIME_MIN_BARS", "ATR_STOP_PERIOD",
      "ATR_STOP_MULTIPLE", "RR_FLOOR", "MAX_HOLD_BARS_TRIGGER", "PIVOT_SPAN",
      "PATTERN_LOOKBACK_BARS", "PATTERN_MAX_AGE_BARS", "DONCHIAN_ENTRY_PERIOD",
      "DONCHIAN_TREND_PERIOD", "DONCHIAN_TARGET_ENABLED", "TRAIL_ENABLED",
      "TRAIL_ATR_MULTIPLE", "BB_PERIOD", "BB_STD", "FADE_ENABLED", "VOLUME_LOOKBACK",
      "VOLUME_HIGH_RATIO", "WF_TRAIN_DAYS", "WF_TEST_DAYS", "WF_OOS_DAYS",
      "WF_MIN_TRADES", "TARGET_ANN_RETURN",
      # ... plus every simulation-affecting constant Phases 1-4 added (RR_TARGET_MIN,
      # MACD_*, VOLUME_CONFIRM_*, ...) — Task 0 enumerates them.
  )
  # NOT snapshotted: paths, symbol lists, UI host/port, and the REVIEW_* diagnostic
  # thresholds — none changes the trades a graph produces. Classified explicitly so
  # the completeness test can tell "irrelevant" from "forgotten".
  VERSION_CONFIG_IGNORED = ("SYMBOLS", "TIMEFRAMES", "BACKFILL_START", "DB_PATH",
                            "STATE_DB_PATH", "STALENESS_INTERVALS", "MAX_RISK_PCT",
                            "COST_RATIO_CEILING", "REVIEW_TIMEFRAME", ...)
  ```
  - `config_snapshot()` reads each key at call time via `getattr`, skipping names absent in the
    running build (a version created before Phase 8 exists should record what existed, not crash);
    values JSON-primitive (tuple → list). `config_hash(snapshot)` = sha256 of canonical JSON, 16
    hex.
  - `verify_reproducible(conn, version_id) -> dict` → `{"version_id", "config_hash_matches",
    "diff": {key: {"recorded", "live"}}, "missing", "added", "schema_version_matches"}`. **Never
    raises on mismatch** — it returns a report; `--forward` prints a loud
    `WARNING: config drift since version creation: FEE_PCT 0.0005 -> 0.001 (N keys differ)` and
    proceeds, because the operator must be able to re-test an old version under a corrected cost
    model *knowing* that is what they are doing.
- **GOTCHA**: the whitelist is the weak point — a later phase's new constant would fall outside it
  silently. Task 13's completeness test is the actual mitigation; the list is just data.
- **VALIDATE**: `pytest tests/test_feedback_versioning.py::TestConfigSnapshot -v`.

### Task 8: `feedback/protocol.py` — spans and the holdout guard
- **ACTION**: create the module, starting with span discipline. Getting this wrong contaminates
  Phase 9's verdict, the project's only deliverable.
- **IMPLEMENT**:
  - Docstring: this is the **only** module in `feedback/` permitted to cause trades to exist, and
    only through `walk_forward_pooled` (lock #3).
  - `class ForwardSpanError(ValueError)` — a dedicated type so the CLI prints a clean message
    (mirrors `_walkforward_command`'s `except ValueError`, cli.py:407-409).
  - `ForwardSpan` frozen: `history_start_ms` (earliest bar the walk-forward may train on),
    `forward_start_ms` (first bar of the UNSEEN window), `forward_end_ms`, `n_forward_days`.
  - `forward_span(*, history_start_ms, forward_start_ms, forward_end_ms)` with three
    `ForwardSpanError` checks:
    1. `forward_end_ms > forward_start_ms > history_start_ms`;
    2. `n_forward_days >= config.REVIEW_FORWARD_MIN_DAYS` — a window too short to say anything is
       refused, not reported (§2 is what 90 days bought; 5 buys less);
    3. **Phase 9's locked holdout is untouchable.** Read `getattr(config, "HOLDOUT_START", None)` /
       `HOLDOUT_END` (Phase 9's reserved prefix, §7) **at call time**. If both are set and the span
       — including `history_start_ms` — intersects `[HOLDOUT_START, HOLDOUT_END]` at all, raise:
       `"forward span A..B intersects the LOCKED final holdout C..D (config.HOLDOUT_START/END).
       Phase 9 owns that span; no feedback iteration, forward test or evolution generation may see
       it."` If the constants do not exist yet, log WARNING and record `holdout_guard="absent"` in
       the result notes, so the phase report can state plainly that the guard was inert when
       measured. Writing it now means nothing changes when Phase 9 lands.
  - `DAY_MS = 86_400_000`, matching `walkforward.py:44`.
- **GOTCHA**: intersection is `not (forward_end < holdout_start or forward_start > holdout_end)` —
  inclusive both sides, because `load_candles` bounds are inclusive and an off-by-one here means
  the holdout was seen.
- **VALIDATE**: `pytest tests/test_feedback_protocol.py::TestForwardSpan -v`.

### Task 9: `protocol.py` — `evaluate_forward`, through the gate and only the gate
- **ACTION**: same file. Pivot-guide **Step 6**.
- **IMPLEMENT**:
  - `ForwardTestResult` frozen: `version_id`, `symbols`, `span`, `gate: Mapping[str,bool]`,
    `passed`, `n_trials_used`, `oos_metrics`, `oos_equity`, `benchmark`, `trades`, `config_drift`,
    `notes`.
  - `evaluate_forward(conn, state_conn, *, version_id, symbols, span, n_trials=None)`:
    1. `graph = versioning.load_graph(...)`; `config_drift = versioning.verify_reproducible(...)`.
    2. **Frozen grid**: a forward test evaluates a *chosen* candidate, so it must not tune. Build
       `{axis: (value,)}` from the version's own parameters — one combo, and
       `walkforward._neighbors` returns `[]` for 1-tuples, so no neighbour probes fire either. "No
       tuning happens" is thereby **structural**, not promised.
    3. **`n_trials` is the campaign's cumulative ledger count** (§4's resolution of the PRD's open
       question), never `len(combos) * len(folds)`. Read it from Phase 1's `trials.py`. If, per
       Task 0, the gate does not itself record the evaluation, record exactly one entry here keyed
       on (graph hash, params hash, span). **Never both.** State the choice in the phase report.
    4. Call the oracle, once:
       ```python
       wf = walk_forward_pooled(conn, list(symbols),
                                start_ms=span.history_start_ms, end_ms=span.forward_end_ms,
                                grid=frozen_grid, oos_days=span.n_forward_days,
                                n_trials=n_trials, strategy=graph)
       ```
       The forward window **is** the gate's one-shot OOS holdout, so `gate`, `benchmark` and
       `n_trials_used` describe exactly the unseen span and the buy-and-hold null (§4.3) comes
       free. *This is the design's central move: the forward test is not a new evaluation path, it
       is the existing gate pointed at a later window.*
    5. Collect the OOS trades for records. If `WalkForwardResult` does not expose them (Task 0),
       re-derive with one `run_graph_backtest` per symbol over the forward window using
       `wf.final_params`, commented as *record-generation, not scoring — the verdict already came
       from the gate above and this call adds no trial.* Prefer the exposed list.
    6. Stamp trades with `version_id` (A10) and return.
  - Wrap `walk_forward_pooled`'s `ValueError` ("span too short: need at least one train+test fold
    before the OOS holdout") into a `ForwardSpanError` naming the fix: history must precede the
    forward window by ≥ `WF_TRAIN_DAYS + WF_TEST_DAYS`.
- **GOTCHA**: `oos_days` is in **days** and the gate computes `tune_end = end_ms - oos_ms`, so
  `forward_start_ms` must equal `forward_end_ms - n_forward_days * DAY_MS` exactly, or the gate's
  holdout and the declared window differ by hours. Assert the round-trip in `forward_span()`.
- **VALIDATE**: `pytest tests/test_feedback_protocol.py::TestEvaluateForward -v`.

### Task 10: `protocol.py` — `apply_suggestion` and `run_loop_iteration`
- **ACTION**: same file. Pivot-guide **Step 5** and the phase's success signal.
- **IMPLEMENT**:
  ```python
  # Deterministic, bounded, diagnosis-driven single-step edits. NOT a mutator: no RNG,
  # no population, one step per suggestion. Phase 6's plugins/mutators/* are the
  # stochastic, population-driven counterpart; if Phase 6 publishes an equivalent
  # bounded edit, delegate rather than keep two implementations.
  _SUGGESTION_EDIT = {"widen-target": ("target_multiple", +1), "narrow-target": ("target_multiple", -1),
                      "widen-stop": ("atr_multiple", +1),      "tighten-stop": ("atr_multiple", -1),
                      "raise-max-hold": ("max_hold_bars", +1), "lower-max-hold": ("max_hold_bars", -1)}
  ```
  `apply_suggestion(graph, suggestion) -> StrategyGraph | None`:
  - `"drop-confirmation:<name>"` → remove that node via Phase 3's graph API, only if more than one
    confirmation remains; else `None`.
  - a parameter edit → find the node declaring the param, read its `ParamSpec`, step by
    `REVIEW_REFINE_STEP_FRAC * (hi - lo)`, clamp to `bounds`, round when `kind == "int"`. If the
    result equals the input (already at a bound) → `None`.
  - **Fail closed**: unknown suggestion, absent param, missing `ParamSpec`/`bounds` → `None` +
    WARNING. Never invent a parameter name — exact names come from Phase 4's registered policy
    (Task 0); if they differ, update `_SUGGESTION_EDIT`, do not guess. Returns a *new* graph.
  - `LoopIterationResult` frozen: `parent_version_id`, `records_written`, `diagnosis`,
    `suggestions_applied`, `child_version_id: str | None`, `forward: ForwardTestResult | None`,
    `notes`.
  - `run_loop_iteration(conn, state_conn, *, version_id, symbols, review_start_ms, review_end_ms,
    span_class="in-sample", forward=None, reviewer="reviewer.trade-quality", persist=True)` — the
    single entry point the CLI wraps, executing Steps 3→6:
    1. **Step 3** — obtain closed trades for the review span and review each. In `"in-sample"` mode
       they come from one `run_graph_backtest` per symbol over an **already-seen** span: that makes
       no new claim (the span was available to tuning), output is banner-labelled, and every record
       carries `span_class="in-sample"` forever. In `"forward"` mode they come from
       `evaluate_forward`, i.e. from the gate.
    2. Persist via `records.insert_records`. 3. **Step 4** — `records.diagnose(...)`.
    4. **Step 5** — apply each suggestion in order, then `versioning.register_version(child,
       parent_id=version_id, provenance={"source": "diagnosis", "diagnosis_digest": d.digest,
       "suggestions": [...]})`. If nothing applied, or the child hash equals the parent's,
       `child_version_id is None` and `notes` says why — an iteration that changes nothing is a
       valid outcome and must not be disguised as progress.
    5. **Step 6** — if `forward` is given and a child exists, forward-test the **child** and persist
       its records with `span_class="forward"`.
- **GOTCHA**: never forward-test the *parent* by default — its span was seen, and a "forward test"
  of it on the same window is exactly the side channel §4 forbids.
- **VALIDATE**: `pytest tests/test_feedback_protocol.py::TestFullLoop -v`.

### Task 11: `cli.py` — the `review` subcommand
- **ACTION**: register `review` → `_review_command` (§7) after Phase 4's `graph-backtest`.
- **IMPLEMENT**:
  - Args: `--strategy PATH | --version ID` (exactly one); `--register` (register and exit);
    `--symbol` (repeatable, default `config.SYMBOLS`); `--start`/`--end` (`type=_date_arg`);
    `--diagnose-only` (aggregate STORED records, runs nothing); `--loop`;
    `--forward-start`/`--forward-end`; `--reviewer` (default `trade-quality`); `--no-persist`;
    `--state-db` (default `config.STATE_DB_PATH`) **on the subcommand**, not as a new global flag,
    so Phases 1/6/7 cannot collide — unless Phase 1 added a global one (Task 0).
  - Dispatch: extend the existing `elif args.command in ("backtest", "walkforward")` family with a
    `review` branch opening **both** connections, resolving the span exactly as cli.py:202-203,
    calling `_review_command`, closing both in `finally`, `sys.exit(exit_code)`.
  - `_review_command(conn, state_conn, symbols, *, …) -> int`:
    - `--register`: print `registered strategy_version=<id> parent=<..> provenance=<source>`; 0.
    - `--diagnose-only`: `load_records` → `diagnose` → print; 0, or 1 when there are no records (an
      empty answer the operator should notice).
    - default/`--loop`: `protocol.run_loop_iteration`. Print `IN-SAMPLE DIAGNOSTIC — NOT EVIDENCE`
      whenever `span_class == "in-sample"`, then the per-trade table, record count, diagnosis, and
      the refinement/forward blocks when present.
    - forward mode prints the per-condition gate dict, the benchmark line, and
      `GATE: {'PASS' if passed else 'FAIL'}` in `_walkforward_command`'s exact shape
      (cli.py:437-439); returns `0 if passed else 1`.
    - non-forward modes return 0 when they ran (zero trades is a result, cli.py:374-376) and 1 when
      they could not run (unknown version, unreadable strategy file).
    - `ForwardSpanError`/`KeyError` → `print(f"ERROR: {exc}")`, return 1. Never a traceback.
  - Reuse `_fmt` for every possibly-None value; add no second formatter.
- **GOTCHA**: call `registry.load_all()` before resolving `--reviewer`. §3: import errors during
  `load_all()` are **fatal, never skipped** — do not wrap it in `try/except`.
- **VALIDATE**: `python -m trading_bot.cli review --help`; `pytest tests/test_feedback_protocol.py::TestReviewCli -v`.

### Task 12: `tests/test_feedback_records.py`
Classes `TestSchema`, `TestExcursions`, `TestVerdicts`, `TestPace`, `TestDiagnosis`,
`TestOracleBoundary`. Two-DB `tmp_path` fixture; tiers from config; MIRROR TEST_STRUCTURE.
- **`TestExcursions::test_entry_bar_is_excluded`** — the phase's key test: entry at bar 0's close,
  exit at bar 3, **bar 0 carrying a huge favourable wick**; assert `mfe_pct` reflects bars 1-3
  only. Docstring names the invariant and cites `engine.py:388`.
- `test_exit_bar_is_included`; `test_mfe_mae_hand_computed_long`/`_short` (bars chosen so the
  arithmetic is checkable by eye — mirror `test_wilder.py::TestHandComputedValues`);
  `test_never_favourable_gives_zero_mfe_and_none_capture`;
  `test_coverage_below_threshold_returns_none` (delete bars → `None`, `"n/a"`, WARNING via
  `caplog`); `test_interval_mismatch_raises` (window at the wrong interval → `ValueError` naming
  both; pins §1 for this new data path).
- `TestVerdicts` — table-driven over `(mfe, mae, outcome)` triples covering every verdict string,
  plus `test_verdict_strings_are_declared` (each in `records.TP_VERDICTS` etc.).
- `TestPace` — `test_pace_ratio_hand_computed`: `pnl_pct=0.05`, `holding_days=30` ⇒
  `required = 1.5**(30/365) - 1 ≈ 0.03392`, `pace_ratio ≈ 1.474`;
  `test_pace_ratio_none_when_holding_days_zero`;
  **`test_single_trade_never_produces_an_annualised_figure`** (regression pinning §2): no
  `ReviewRecord` field name contains `"ann"`, and a one-record `diagnose()` gives
  `pace["sample_adequate"] is False` with `suggestions == ("insufficient-sample",)`.
- `TestDiagnosis` — outcome mix/dominance; dead weight at coverage 1.0;
  `test_widen_and_tighten_stop_are_mutually_exclusive`;
  `test_suggestions_are_from_the_closed_vocabulary`;
  `test_below_min_trades_emits_only_insufficient_sample`;
  `test_digest_is_stable_and_content_sensitive`.
- `TestSchema` — DDL twice is a no-op; round-trip preserves every field; double insert → 1 row;
  `PRAGMA table_info` shows every `*_ts` column `INTEGER` (pins §6); `test_no_aggregate_field` (no
  field matches `fitness|score|reward|objective|rank`).
- **`TestOracleBoundary::test_review_modules_do_not_import_the_oracle`** — `ast`-parse
  `records.py`, `versioning.py`, `trade_quality.py`; assert no import names `backtest.walkforward`,
  `backtest.engine`, `backtest.trials`, `framework.execute`, `evolution`. Docstring: *"A review
  that can run a backtest is a second fitness oracle. Contract §4 forbids one; this test is the
  lock."*

### Task 13: `tests/test_feedback_versioning.py`
Classes `TestRegister`, `TestLineage`, `TestConfigSnapshot`, `TestLoadGraph`.
- `test_same_graph_gives_same_version_id`; `test_key_order_does_not_change_the_id`;
  `test_different_params_give_different_ids`; `test_reregistering_is_idempotent` (1 row, original
  `created_ts`); `test_self_parent_is_refused`; `test_graph_json_matches_the_stored_hash`;
  `test_unknown_version_raises_keyerror`.
- `TestLineage` — `test_lineage_is_root_first` over a 3-deep chain; `test_lineage_stops_on_a_cycle`
  (hand-insert a cyclic row → raises, does not hang).
- `TestLoadGraph::test_roundtrip_through_phase3_from_dict` — the reloaded graph's canonical hash
  equals the stored `graph_hash`. This is what makes a version *re-runnable*.
- **`TestConfigSnapshot::test_every_config_constant_is_classified`** — for every `UPPER_CASE`,
  non-callable, non-dunder attribute of `trading_bot.config`, assert membership in
  `VERSION_CONFIG_KEYS` or `VERSION_CONFIG_IGNORED`. Docstring: *"Fails when a later phase adds a
  constant without deciding whether it changes what a strategy version does. That decision is the
  whole mitigation for config.py being mutable global state."*
- `test_verify_reproducible_detects_drift` (`monkeypatch.setattr(config, "FEE_PCT", 0.001)` after
  registering → `config_hash_matches is False`, `diff["FEE_PCT"] == {"recorded": 0.0005, "live":
  0.001}`); `test_verify_reproducible_clean_when_nothing_changed`;
  `test_stamp_trades_sets_the_version_without_mutating_the_original`.

### Task 14: `tests/test_feedback_protocol.py`
Classes `TestForwardSpan`, `TestEvaluateForward`, `TestApplySuggestion`, `TestFullLoop`,
`TestReviewCli`.
- `TestForwardSpan` — inverted/zero spans raise; below `REVIEW_FORWARD_MIN_DAYS` raises;
  `n_forward_days` round-trips exactly against `forward_end - forward_start`.
  **`test_locked_holdout_is_refused`** — `monkeypatch.setattr(config, "HOLDOUT_START", …,
  raising=False)` / `HOLDOUT_END`, then `ForwardSpanError` for a contained span, a left overlap, a
  right overlap, and a `history_start_ms` reaching into the holdout; a span strictly before it is
  accepted. Cites §4.4. Plus `test_absent_holdout_constants_warn_but_do_not_block`.
- `TestEvaluateForward` — monkeypatch `walk_forward_pooled` with a spy: called **exactly once**;
  `strategy=` is the loaded graph; the grid has exactly one combo (`all(len(v) == 1 …)`);
  `oos_days == span.n_forward_days`; `n_trials` is the ledger-derived value, not
  `len(combos) * len(folds)`. Plus `test_short_history_raises_forward_span_error` (message names
  `WF_TRAIN_DAYS + WF_TEST_DAYS`) and `test_config_drift_is_reported_not_fatal`.
- `TestApplySuggestion` — a step stays inside `ParamSpec.bounds`; a param at its bound → `None`;
  unknown suggestion → `None` + WARNING; `int`-kind params stay integral; `drop-confirmation` on a
  single-confirmation graph → `None`; the input graph is unchanged (compare canonical hashes).
- **`TestFullLoop::test_one_iteration_executes_without_manual_glue`** — the phase success signal on
  a synthetic two-DB fixture: (1) seed regime/setup/trigger bars (reuse `test_backtest.py`'s `seed`
  shape) over a span long enough for train+test plus a forward window — pass **reduced**
  `train_days`/`test_days` through the protocol rather than seeding three years; (2) register the
  seed graph; (3) `run_loop_iteration(..., span_class="in-sample", forward=ForwardSpan(...))`;
  (4) assert parent records with `span_class="in-sample"`; a `Diagnosis` whose suggestions are all
  in the closed vocabulary; a child whose `parent_id` is the seed and whose
  `provenance["diagnosis_digest"] == diagnosis.digest`; `lineage()` depth 2; a `ForwardTestResult`
  whose `gate` is a dict over the contract's `GATE_CONDITIONS`; forward records with
  `span_class="forward"`; the trial ledger incremented exactly once per evaluation.
  Docstring: *"PRD Phase 5 success signal — trade, close, review, refine, version, forward-test
  with no manual glue. If this test needs a helper defined in the test file, that helper belongs in
  protocol.py instead."* Plus `test_no_suggestion_yields_no_child_and_says_so`.
- `TestReviewCli` — `_review_command` in `--register`, default, `--diagnose-only`, `--loop` modes
  via `capsys`: the in-sample banner appears in diagnostic mode and **not** in forward mode;
  forward mode prints `GATE:` and returns 0/1 matching `passed`; unknown `--version` prints
  `ERROR:` and returns 1. Mirrors `tests/test_cli.py`'s `capsys` idiom.

---

## Testing Strategy

### Unit Tests

Summary of what Tasks 12–14 specify (that is where each case's fixture and docstring live).

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| `test_entry_bar_is_excluded` | entry at bar 0's close with a huge favourable wick on bar 0; exit at bar 3 | `mfe_pct` from bars 1-3 only | **the key invariant** |
| `test_exit_bar_is_included` | favourable extreme on the exit bar | counted in `mfe_pct` | — |
| `test_mfe_mae_hand_computed_long` / `_short` | 4 hand-built bars | exact fractions of entry | — |
| `test_never_favourable_gives_zero_mfe_and_none_capture` | monotonically adverse bars | `mfe_pct=0.0`, capture `None`, `"never-favoured"` | Edge |
| `test_coverage_below_threshold_returns_none` | window with bars deleted | `(None, None)`, verdicts `"n/a"`, WARNING | Edge |
| `test_interval_mismatch_raises` | window seeded at 4h while `review_tf="1h"` | `ValueError` naming both intervals | Edge |
| `TestVerdicts` (table-driven) | `(mfe, mae, outcome)` triples | every TP/SL verdict string reached; each in its declared tuple | — |
| `test_pace_ratio_hand_computed` | `pnl_pct=0.05`, `holding_days=30` | `required ≈ 0.03392`, `pace_ratio ≈ 1.474` | — |
| `test_pace_ratio_none_when_holding_days_zero` | `exit_ts == entry_ts` | `None`, `"n/a"` | Edge |
| `test_single_trade_never_produces_an_annualised_figure` | 1 record | no field name contains `"ann"`; `sample_adequate=False`; `("insufficient-sample",)` | **pins §2** |
| `test_below_min_trades_emits_only_insufficient_sample` | n < `REVIEW_PACE_MIN_TRADES` | exactly one suggestion | — |
| `test_widen_and_tighten_stop_are_mutually_exclusive` | adversarial verdict mix | never both | — |
| `test_suggestions_are_from_the_closed_vocabulary` | any diagnosis | every `s.split(":",1)[0] in SUGGESTIONS` | — |
| `test_digest_is_stable_and_content_sensitive` | same / altered records | equal / different digest | — |
| `TestSchema` round-trip + double insert | one record inserted twice | every field preserved; 1 row | — |
| `PRAGMA table_info` check | `review_records` schema | every `*_ts` column `INTEGER` | contract §6 |
| `test_no_aggregate_field` | `ReviewRecord.__dataclass_fields__` | no `fitness\|score\|reward\|objective\|rank` | **contract §4** |
| `test_review_modules_do_not_import_the_oracle` | `ast` scan of 3 modules | no gate/engine/trials/`framework.execute`/evolution import | **contract §4** |
| `test_same_graph_gives_same_version_id` / `test_key_order_does_not_change_the_id` | same graph, shuffled key order | identical id | — |
| `test_reregistering_is_idempotent` | same graph twice | 1 row, original `created_ts` | — |
| `test_self_parent_is_refused` | `parent_id == version_id` | `ValueError` | Edge |
| `test_lineage_stops_on_a_cycle` | hand-inserted cyclic row | raises, does not hang | Edge |
| `test_roundtrip_through_phase3_from_dict` | stored `graph_json` | reloaded graph's canonical hash == `graph_hash` | — |
| `test_every_config_constant_is_classified` | every UPPER config attr | in `VERSION_CONFIG_KEYS` or `_IGNORED` | **the A6 mitigation** |
| `test_verify_reproducible_detects_drift` | monkeypatched `FEE_PCT=0.001` | `config_hash_matches=False`, exact diff | — |
| `test_locked_holdout_is_refused` | 4 overlap shapes vs `HOLDOUT_*` | `ForwardSpanError` each; earlier span accepted | **protects Phase 9** |
| `TestEvaluateForward` spy | one `evaluate_forward` call | `walk_forward_pooled` called once, `strategy=graph`, 1-combo grid, ledger `n_trials` | **contract §4** |
| `test_short_history_raises_forward_span_error` | history < train+test before the window | `ForwardSpanError` naming `WF_TRAIN_DAYS + WF_TEST_DAYS` | Edge |
| `TestApplySuggestion` bounds | param already at its bound | `None`, input graph unchanged | Edge |
| `test_one_iteration_executes_without_manual_glue` | synthetic 2-DB fixture | records → diagnosis → child (lineage depth 2) → forward gate dict; ledger +1 per evaluation | **success signal** |
| `TestReviewCli` modes | `--register` / default / `--diagnose-only` / `--loop` | banner in-sample only; `GATE:` + 0/1 in forward mode; `ERROR:` + 1 on unknown version | — |

### Edge Cases Checklist
- [x] Empty trade list → 0 records, `Diagnosis(n_records=0)`, `("insufficient-sample",)`
- [x] Zero-length / inverted span → refused with a named error
- [x] Missing bars → `None` excursions, never a silent 0
- [x] Bar interval ≠ config → raises
- [x] `bars_expected == 0` → `None`, no ZeroDivisionError
- [x] `strategy_version == ""` → `REVIEW_LEGACY_VERSION_ID`
- [x] Trade missing `planned_rr`/`confirmations` → `getattr` defaults, `NULL` / `[]`
- [x] Unknown version id → `KeyError` / `ERROR:` + exit 1, never the wrong strategy
- [x] Duplicate review run → idempotent replace
- [x] Concurrent access → WAL + the state-db lock + content-addressed upsert; no retry layer
- [x] Network → none; every test synthetic, nothing `@pytest.mark.network`

## Validation Commands

```bash
# Static analysis — no linter and no type checker are configured (KNOWN-LIMITATIONS §8,
# contract §12): py_compile + pytest IS validation here. Do not invent mypy/ruff.
python -m py_compile src/trading_bot/feedback/__init__.py \
  src/trading_bot/feedback/records.py src/trading_bot/feedback/versioning.py \
  src/trading_bot/feedback/protocol.py \
  src/trading_bot/plugins/reviewers/trade_quality.py \
  src/trading_bot/config.py src/trading_bot/cli.py

# Phase tests — EXPECT all pass, incl. entry-bar exclusion, oracle boundary,
# config completeness, holdout guard, full loop.
.venv/bin/python -m pytest tests/test_feedback_records.py \
    tests/test_feedback_versioning.py tests/test_feedback_protocol.py -v

# Full suite — EXPECT the 286 pre-existing tests still pass (plus Phases 1-4's own),
# plus ~35 new. Report the exact collected count; §8 requires it at every boundary.
.venv/bin/python -m pytest -q

# Registration — EXPECT one reviewer row with a non-empty rationale (§12.3).
python -m trading_bot.cli plugins | grep trade-quality

# Database
sqlite3 data/state.db ".schema review_records" ".schema strategy_versions"
sqlite3 data/ohlcv.db ".schema"    # must be byte-identical to before this phase
```

### Manual validation — the success signal, end to end
```bash
cli review --strategy data/strategies/thin-slice.strategy.json --register
cli review --version <id> --start 2023-07-27 --end 2026-04-27
cli review --version <id> --diagnose-only
cli review --version <id> --loop --forward-start 2026-04-27 --forward-end 2026-07-26
```
- [ ] Every closed trade in the span produced exactly one record (count matches the backtest's)
- [ ] Diagnostic output carries the in-sample banner; the forward run does not
- [ ] `--loop` printed a child id whose parent is the seed, and a verdict listing every
      `GATE_CONDITIONS` entry
- [ ] Re-running the identical command changes no row counts (idempotency)
- [ ] The trial ledger grew by **exactly** the number of oracle evaluations performed (measure
      before/after). If more, Task 0 question 2 was not resolved — stop and fix it
- [ ] `git status` shows no `data/state.db*` files staged

## Acceptance Criteria
- [ ] All 15 tasks complete; all validation commands pass; 286 pre-existing tests green with the
      exact new collected count in the phase report
- [ ] `ReviewRecord` has no aggregate/fitness field; the `ast` import-ban test passes
- [ ] `evaluate_forward` provably routes through `walk_forward_pooled` with a one-combo grid and a
      ledger-derived `n_trials`; exactly one ledger entry per evaluation
- [ ] The excursion window excludes the entry bar and includes the exit bar, pinned by a test
      citing `engine.py:388`
- [ ] No per-trade field annualises a single trade; `diagnose()`'s pace carries `sample_adequate`
      and refuses suggestions below the floors
- [ ] `TARGET_ANN_RETURN = 0.50` and every `REVIEW_*` under the reserved prefix; no existing
      constant modified
- [ ] `review` in `cli.py --help`, exits 0/1 per the documented semantics
- [ ] `strategy_versions` rows carry graph JSON + schema version + config snapshot;
      `verify_reproducible` detects a monkeypatched `FEE_PCT`
- [ ] `forward_span()` refuses any overlap with `config.HOLDOUT_START/END`
- [ ] One `review --loop` performs trade → close → review → refine → version → forward-test with
      no manual glue (PRD Phase 5 success signal)
- [ ] Zero edits to `engine.py`, `walkforward.py`, `equity.py`, `metrics.py`, `statestore.py`,
      `trials.py`, or anything under `framework/` (§12.4)

## Completion Checklist
- [ ] Code follows the discovered patterns — DB_MODULE_PATTERN, IDEMPOTENT_WRITE,
      CONTENT_FINGERPRINT, INTERVAL_ASSERTION, HOLDING_TIME, CLI_SUBCOMMAND_PATTERN
- [ ] Error handling matches codebase style: None-for-undefined (never NaN, never 0.0-as-missing);
      `ValueError` on an interval mismatch; `ForwardSpanError`/`KeyError` surfaced by the CLI as
      `ERROR: …` + exit 1, never a traceback
- [ ] Logging follows conventions: `logging.getLogger("trading_bot")`, WARNING for a skipped or
      undefined review, INFO for counts written, no `print` outside `cli.py`
- [ ] Tests follow the test patterns: config-derived tiers (never hardcoded), autouse cache clear,
      `make_trade`/`seed` builders extended rather than forked, two-DB `tmp_path` fixtures, real
      `data/*.db` never touched
- [ ] No hardcoded values — every threshold reads from `config` under the `REVIEW_*` /
      `TARGET_ANN_RETURN` prefix; `review_tf` resolved at call time, not import time
- [ ] Conventions: all timestamps epoch ms UTC in both tables; one lock per database (Phase 1's);
      `CREATE TABLE IF NOT EXISTS` owned by the using module; lowercase-hyphen registry name,
      snake_case modules, no camelCase; non-empty `rationale`; no new dependency
- [ ] Module docstrings state the oracle boundary and the no-lookahead argument
- [ ] No unnecessary scope additions — nothing from the NOT Building list crept in; no file owned
      by another phase created or edited
- [ ] Degrees of freedom consumed by the phase recorded (KNOWN-LIMITATIONS §9 discipline)
- [ ] Self-contained — no codebase searching needed during implementation beyond Task 0's
      recorded answers

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **The reviewer becomes a second fitness oracle** (someone adds `ReviewRecord.fitness`; a mutator ranks on `pace_ratio`) | Medium | **Critical** — all DSR accounting becomes theatre | Four structural locks, two test-enforced: no aggregate field, `ast` import ban, single gate seam, self-labelling `span_class` |
| MFE/MAE window off by one bar | **High** — it is the natural way to write it | High — every verdict biased, Step 4 suggests the wrong edits | Favourable-wick fixture; the invariant + `engine.py:388` citation in the docstring |
| A pace figure quoted as evidence (the §2 error again) | Medium | High — it already happened once | No per-trade annualisation; `sample_adequate`; below-floor diagnoses emit only `insufficient-sample`; in-sample banner; the `"ann"` field-name test |
| Double-counted or uncounted trials | Medium | High — DSR over-penalised for the wrong reason, or dishonest | Task 0 Q2 resolves who increments; before/after ledger measurement; the one-call spy test |
| `config.py` mutable globals make an old version unreproducible | **Certain** (design limit) | Medium | `config_snapshot` + hash + `verify_reproducible`; loud drift warning; the completeness test. **Named, not solved** (A6) |
| Two graph canonicalisations (Phase 3's and the fallback) drift | Medium | High — lineage and idempotency break | Prefer Phase 3's published hash; if the fallback is used, say so and pin the round-trip test; flag consolidation to Phase 3 |
| Phase 3/4 names differ from this plan (`target_multiple`, `ReviewContext`, ledger API) | **High** | Low if handled | Task 0 is a hard prerequisite; `apply_suggestion` **fails closed** rather than guessing |
| Phases 5 and 6 both implement bounded parameter edits (parallel phases) | Medium | Low | Documented boundary: deterministic/diagnosis-driven here, stochastic/population-driven there; delegate if Phase 6 publishes an equivalent |
| Forward window overlaps Phase 9's holdout before Phase 9 defines it | Medium | **Critical** — contaminates the project's only verdict | Guard written now, reads `HOLDOUT_*` at call time; when absent it WARNs and records `holdout_guard="absent"` so the report says the guard was inert |
| `state.db` WAL sidecars committed | Low | Low | Task 0 `.gitignore` check — today's `data/*.db` does not match `state.db-wal` |
| `review` over 3.5 years is slow | Medium | Low | Reuses `engine._CACHE`'s content-keyed memo; `--start/--end` scope it; `--diagnose-only` runs nothing |

## Notes

**Why this phase decides iteration cost.** The pivot guide lists three gaps; Phases 3–4 fix #1
(expressiveness), Phase 6 attacks #3 (performance). Gap #2 — "there's no standardisation workflow
to improve the model" — is this phase alone, and it is what makes the *next* version cheaper.
`review_records` and `strategy_versions` are the first artefacts in this project's history that
outlive a process.

**What Step 4 deliberately is**: a `Diagnosis` — which outcome class dominates, which confirmation
is dead weight, whether stops were approached, whether targets were reachable, and whether the
sample is even large enough to say — with a closed suggestion vocabulary. **Not** an LLM proposing
code, not a learned model, not a scalar. That restraint is what stops a 10 000-variant search
being steered by an uncounted, undeflated in-sample signal.

**Honest limitations to carry into the phase report**, stated up front so they are findings rather
than discoveries: (1) `confirmation_pnl_delta` is in-sample and undeflated — it says where to look,
nothing more; (2) the "would a wider stop have won?" counterfactual is not built, so the SL axis
can only say "never approached" or "it bound", never "it was wrong"; (3) forward-test aggregates
are not persisted (A8), only their per-trade records; (4) reproducibility is recorded, not pinned
(A6); (5) the pace denominator is compounded and therefore ~18% less demanding at 30 days than a
linear pro-rata one — chosen for consistency with `compute_equity_metrics`, recorded so it is
mistaken for neither rigour nor flattery.

**Degrees of freedom consumed: zero.** Every `REVIEW_*` constant labels a trade; none selects a
strategy, none enters a walk-forward grid, and none may. The only additions to the project's
search budget are the trials the forward tests charge to the ledger — which is the point.

**Handoff.** *Phase 6* consumes `versioning.register_version` (per-generation candidates) and
`diagnose(...).suggestions` (a mutation prior), and must route every evaluation through its own
`evolution/oracle.py`, never through `feedback/`. *Phase 7* consumes `records.load_records`,
`diagnosis_to_dict`, `versioning.list_versions` and `lineage` — all plain dataclasses/dicts, so the
UI needs no new serialisation layer. *Phase 9* must set `config.HOLDOUT_START/END` **before** its
campaign starts; until then this phase's guard is inert and the report says so.
