# Plan: Validation Integrity (v0.3.0 Phase 1)

## Summary
THE GATE compares the strategy against **zero** and computes Sharpe/DSR on a return series that books each trade's whole P&L on its exit day. Both are documented measurement defects: KNOWN-LIMITATIONS §0 ("THE GATE could have blessed a strategy worse than inaction") and §3 (daily skew 3.79 / kurtosis 31.24, finding MEDIUM-5, unfixed). This phase makes the measuring stick honest **before** Phase 3 builds a framework that optimizes against it: a buy-and-hold null computed in every walk-forward run, a per-condition gate dict with two new benchmark conditions, holding-day P&L attribution, and a persistent trial ledger in a second SQLite DB so DSR's `n_trials` survives restarts.

Nothing here changes the strategy. Every number it produces is a **measurement of the existing build**, and the acceptance test is that it reproduces a table already published in §0 from one command.

## User Story
As the bot's sole operator, I want one command that scores the existing strategy against buy-and-hold on the same span with an undistorted daily return series, so that when the evolution engine starts searching in Phase 6 it optimizes against a null that is real rather than against zero.

## Problem → Solution
**Current**: `_evaluate_gate` returns a bare `bool` over 5 conditions, none referencing an alternative to trading. `WalkForwardResult` carries no benchmark and discards the `n_trials` it charged the DSR (`walkforward.py:383`). `daily_returns` (`equity.py:44-51`) books `t.pnl_pct` entirely on `t.exit_ts // DAY_MS`, so 23 trades over 90 days give 73 exact-zero days and kurtosis 31.24. Nothing counts degrees of freedom across runs.

**Solution**: `backtest/benchmark.py::buy_and_hold` computes per-symbol and equal-weight-basket buy-and-hold from stored 1d bars; `walk_forward_pooled` calls it on the **same OOS span it scores the strategy on** and adds `beats_benchmark_return` / `beats_benchmark_sharpe` to a 7-element `GATE_CONDITIONS`; `_evaluate_gate` returns `dict[str, bool]` and `passed` becomes `all(gate.values())`; `daily_returns` gains `attribution="spread"` spreading P&L over `entry_ts`→`exit_ts` inclusive with `"exit_day"` reachable so the delta is measured; `data/statestore.py` + `backtest/trials.py` provide `data/state.db` and a `trial_ledger` table every evaluation can increment.

## Metadata
- **Complexity**: **Large** — 4 new source files, 4 modified, 3 new test files, 2 deliberately-edited test files; ~900 net new/changed lines including tests.
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md` — Phase 1, Validation integrity
- **Binding contract**: `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md` §2, §4, §6, §7, §8, §12
- **Estimated Files**: 12 (7 new, 5 modified) + `.gitignore`
- **Depends on**: nothing. Parallel with Phase 2. **Gates Phase 3.**
- **Test baseline**: **286 tests collected** (`.venv/bin/python -m pytest --collect-only -q`, verified 2026-07-27). All 286 must stay green.

---

## UX Design

Internal change plus one new read-only subcommand. No GUI. Today `walkforward` prints one bit — `GATE: FAIL` — with no answer to *which condition* or *compared to what*. After:

```
$ python -m trading_bot.cli benchmark --start 2023-07-27 --end 2026-07-26
symbol         total       ann   sharpe  sortino   max_dd  n_days
BTCUSDT      2.2104x   +30.26%    0.799    1.208   52.98%    1095
ETHUSDT      1.0262x    +0.87%    0.338    0.508   67.56%    1095
SOLUSDT      2.9974x   +44.18%    0.851    1.326   76.27%    1095
BASKET       2.1628x   +29.32%    0.728    1.086   64.32%    1095

$ python -m trading_bot.cli walkforward
benchmark (equal-weight basket, same OOS span):
  BASKET       0.8527x   -47.61%   -1.303   -1.732   31.83%      90
n_trials charged to DSR: 36
gate conditions:
  sample_adequacy          FAIL
  sharpe                   PASS
  dsr                      FAIL
  max_drawdown             PASS
  per_symbol_expectancy    PASS
  beats_benchmark_return   PASS
  beats_benchmark_sharpe   PASS
GATE: FAIL
```

| Touchpoint | Change | Notes |
|---|---|---|
| `walkforward` printout | + 7 named conditions, benchmark row, `n_trials` | The `GATE: PASS`/`GATE: FAIL` line stays byte-identical — `tests/test_backtest.py:622,626` assert on it |
| `benchmark` subcommand | new | Exit 0 when every requested symbol produced a full metric set, else 1 |
| `WalkForwardResult` | 9 → 12 fields (`gate`, `benchmark`, `n_trials_used` before `passed`) | All construction sites use keyword args |
| `data/` | + `state.db` | Separate DB so a corrupt experiment log cannot endanger 117 MB of price history |

---

## Mandatory Reading

| P | File | Lines | Why |
|---|---|---|---|
| P0 | `_shared-architecture-contract.md` | §2, §4, §6, §7, §8 | BINDING. `BenchmarkResult` shape, `buy_and_hold` signature, `GATE_CONDITIONS`, `state.db` table ownership, reserved config prefixes / CLI names / test filenames |
| P0 | `backtest/equity.py` | 1-13, 28-51, 159-220 | Docstring states the exit-day convention being changed; `daily_returns`; `_skew_kurt`; the wipe-out guard (188-210) is the defensiveness to mirror |
| P0 | `backtest/walkforward.py` | 78-114, 225-254, 383-412 | Gate thresholds, `WalkForwardResult`, `_evaluate_gate` (return type changes), the OOS block where `n_trials_used` is computed and discarded |
| P0 | `data/storage.py` | 22-32, 35-73, 205-244 | `TIMEFRAME_MS`, module-level `_db_lock`, `connect()` (the shape `statestore.connect()` mirrors), `load_candles()` — both bounds **inclusive**, rows are `(ts, open, high, low, close, volume)` tuples |
| P0 | `.claude/PRPs/reports/KNOWN-LIMITATIONS.md` | §0, §0b, §3, §9 | The table to reproduce, the correlation caveat, the MEDIUM-5 statement, the degrees-of-freedom discipline |
| P1 | `cli.py` | 14-22, 111-126, 199-209, 356-368, 396-439 | Imports, the `backtest` parser to copy, the dispatch `elif` to extend, `_fmt`/`_print_metrics`, `_walkforward_command` |
| P1 | `config.py` | 10, 148-151, 180-183, 186-196 | `SYMBOLS`, `WF_*`, the cost constants the benchmark charges, `date_to_ms` |
| P1 | `tests/test_backtest.py` | 23-56, 469-501, 524-567, 603-626 | Tier-derived constants, `make_trade`, `per_symbol_fake_run_factory`, the walk-forward class, the `WalkForwardResult` construction needing 3 new fields |
| P1 | `tests/test_equity.py` | 21-62 | `make_trade` sets `entry_ts=exit_ts` — why most of this file survives the change |
| P2 | `backtest/engine.py` | 109-132, 185-200, 225-236 | `Trade` fields (`entry_ts`/`exit_ts` already exist — **no schema change needed**), `_assert_interval`'s "fail loudly rather than silently in the flattering direction", `run_backtest` signature |
| P2 | `tests/test_cli.py` | 281-322 | `TestMainArgv` — argv-driven `cli.main()` pattern for the subcommand wiring test |
| P2 | `backtest/metrics.py` | 1-28 | "Pure computation … Ratios are None when undefined" — the convention `benchmark.py` follows |

**External documentation: none needed.** Every formula used (`sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `probabilistic_sharpe`, `expected_max_sharpe`, `deflated_sharpe`) is already at `equity.py:54-156` and reused unchanged. Buy-and-hold is arithmetic on stored closes.

---

## Measured Ground Truth (verified 2026-07-27 against `data/ohlcv.db`; do not re-derive)

**A. Span.** `date_to_ms("2023-07-27") = 1690416000000`, `date_to_ms("2026-07-26") = 1785024000000`; span 1095 days. `load_candles(..., "1d", …)` returns **1096 rows** (bounds inclusive) → **1095 daily returns**. All three production symbols have full coverage.

**B. Full-span buy-and-hold reproduces §0.** Fee = one round-trip, `2*(FEE_PCT+SLIPPAGE_PCT) = 0.0014`. Format: total / ann / Sharpe / max DD.

| | measured, no fee | measured, with fee | §0 published |
|---|---|---|---|
| BTC | 2.2135× / +30.324% / 0.8002 / 52.976% | **2.2104× / +30.263% / 0.7992 / 52.976%** | 2.21× / +30.3% / 0.80 / 53.0% |
| ETH | 1.0277× / +0.914% / 0.3389 / 67.560% | 1.0262× / +0.867% / 0.3382 / 67.560% | 1.03× / +0.9% / 0.34 / 67.6% |
| SOL | 3.0016× / +44.251% / 0.8516 / 76.266% | 2.9974× / +44.183% / 0.8510 / 76.266% | 3.00× / +44.3% / 0.85 / 76.3% |
| **Basket, daily-rebalanced** | 2.1658× / +29.383% / 0.7292 / 64.317% | **2.1628× / +29.322% / 0.7284 / 64.317%** | **2.17× / +29.4% / 0.73 / 64.3%** |
| Basket, buy-once-hold | 2.0809× / +27.67% / 0.6998 / 66.91% | — | ✗ matches nothing |

**→ Basket is EQUAL-WEIGHT, DAILY-REBALANCED — measured, not assumed.** Its daily return is the arithmetic mean of the three symbols' daily returns; buy-once-hold misses all four published figures. Contract §4 already says "daily-rebalanced"; this confirms it. **Fee-booking is immaterial at published precision**, and first-day vs last-day booking is identical to 4 decimals on every metric — which is what makes the tolerances defensible.

**C. Annualization honesty.** `2.2135^(365/1095) − 1 = +30.32%`, and the span is exactly three years, so §0's annualized column is a **true 3-year CAGR**.

⚠ **Correction to this phase's brief**: it says "the §0 numbers came from a 90-day OOS window; `ann_return_pct` there is an extrapolation." Wrong — §0's header states the span and the arithmetic confirms it. The 90-day extrapolation is **§2**'s "+68% annualised" headline and §1's OOS drawdown. So: the `benchmark` CLI over a long span gives an honest, quotable CAGR; the null computed *inside* `walk_forward_pooled` over the 90-day holdout is a 90-day extrapolation on **both** sides — the comparison is valid (identical span, identical annualization), the level is not quotable. The CLI must print a NOTE below 365 days.

**D. The 90-day OOS null was NEGATIVE — a finding.** Over `end_ms − 90 days` → `end_ms`: BTC 0.8349× / −51.90% / Sharpe −1.958; ETH 0.8292× / −53.21% / −1.340; SOL 0.8867× / −38.59% / −0.667; **basket 0.8527× / −47.61% / Sharpe −1.303 / 31.83% DD**.

**Expect both new gate conditions to PASS on v0.2.0**: strategy OOS Sharpe 1.175 (exit-day) / 1.512 (spread) vs basket −1.303, positive OOS return vs −47.6%. The verdict is **unchanged: FAIL on `sample_adequacy` and `dsr`, 5 of 7 pass** — exactly §1's two failures. Do not "fix" the benchmark because it fails to reject. For the phase report: the holdout was a **bear window**, and a mostly-flat strategy beats a long-only null in a bear window by not being long (§2: "the OOS upturn is a small tail on a long decline"). The null is **necessary but not sufficient** — trivially passable on a short adverse window. Phase 9 must also score the full-span null; Phase 1 records that handoff rather than fixing it.

**E. MEDIUM-5 before/after, measured on the real 23 OOS trades.** Gate-selected params (`trail_enabled=False`, `target_enabled=False`, `max_hold_bars=48`), pooled over the 3 production symbols, 90-day OOS window:

| | exit-day (current) | spread (this phase) |
|---|---|---|
| n_trades / n_days / exact-zero days | 23 / 90 / **73** | 23 / 90 / **60** |
| **skew / kurtosis** | **3.7883 / 31.2449** | **1.4034 / 15.5429** |
| daily Sharpe / **annualized Sharpe** | 0.06150 / **1.1750** | 0.07914 / **1.5120** |
| Sortino / max drawdown | 2.4875 / 19.13% | 2.7172 / 18.92% |
| DSR (n_trials=36) / (n_trials=1) | 0.0672 / 0.7423 | 0.0860 / 0.7829 |
| Σ daily returns | 0.16509935688760607 | 0.16509935688760607 |

The exit-day column reproduces §3's "skew 3.79, kurtosis 31.24" to 4 significant figures — proof the harness is wired to the same trades §3 measured. Four things to internalize:
1. **31.24 → 15.54 satisfies the success signal** ("no longer 30+ kurtosis"). It does **not** reach normality (3.0) and must not be claimed to: holds are 1–3 calendar days (median 3, `max_hold_bars=48` at the 1h trigger tier), so there is only so much smearing available.
2. **The fix RAISES annualized Sharpe, 1.175 → 1.512** — flattering movement on a *gate condition*. Correct (the exit-day zero/spike alternation inflates its own variance) but it makes that condition easier to pass. Record it; do not bury it.
3. **DSR still fails** (0.0860 ≤ 0.95; 0.7829 even at `n_trials=1`). §1 already established no `n_trials` rescues it.
4. **Σ daily returns is invariant** to attribution mode — the strongest available unit test for the new path.

---

## Patterns to Mirror

### SQLITE_CONNECT + LOCKED_WRITE
```python
# SOURCE: data/storage.py:32-73, 91-101 — module-level lock beside the accessors,
# WAL, check_same_thread=False, parent mkdir, DDL inside connect(); every write
# holds the lock and commits INSIDE it. statestore.connect() mirrors this exactly
# except it creates NO tables (contract §6: DDL belongs to the module that uses
# the table); trials.TrialLedger.record mirrors the write shape.
_db_lock = threading.Lock()

def connect(db_path: str | None = None) -> sqlite3.Connection:
    if db_path is None: db_path = config.DB_PATH
    db_file = Path(db_path); db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS ohlcv (...)""")
    conn.commit(); return conn

    with _db_lock:
        conn.executemany("""INSERT OR REPLACE INTO ohlcv ... VALUES (?,?,?,?,?,?,?,?)""", rows)
        conn.commit()
```

### DEFENSIVE_NUMERIC + NONE_ON_UNDEFINED
```python
# SOURCE: equity.py:188-210 and 54-62 — refuse to lie rather than propagate
# nonsense, and return None (never 0.0, never a raise) for an uncomputable metric.
# benchmark._metrics_from_returns copies BOTH verbatim.
    equity = 1.0; wiped_out = False
    for r in rets:
        factor = 1.0 + r
        # A non-positive factor means a single day's loss consumed the whole
        # account ... an even number of such factors would otherwise multiply back
        # to a spuriously large positive equity ...
        if factor <= 0.0:
            equity = 0.0; wiped_out = True; break
        equity *= factor
    if n == 0:       ann_return = None
    elif wiped_out:  ann_return = -1.0
    elif equity > 0: ann_return = equity ** (PERIODS_PER_YEAR / n) - 1.0
    else:            ann_return = None

def sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR) -> float | None:
    """Annualized Sharpe (risk-free rate 0). None if < 2 obs or zero variance."""
    if len(returns) < 2: return None
    sd = statistics.stdev(returns)          # sample stdev (n-1)
    if sd == 0: return None
    return (statistics.fmean(returns) / sd) * math.sqrt(periods_per_year)
```

### FROZEN_RESULT_DATACLASS + GATE_FAILS_SAFE
```python
# SOURCE: walkforward.py:98-114 (frozen dataclass, heavily-commented fields, dicts
# for metric bundles) and 225-254 ("Fails safely (returns False) whenever a
# required metric is None; never raises"). BenchmarkResult mirrors the former;
# _evaluate_gate keeps the latter while changing its return bool -> dict.
@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]; final_params: BacktestParams
    final_max_hold_bars: int | None; oos_start: int; oos_end: int
    oos_metrics: dict   # from compute_metrics
    oos_equity: dict    # from compute_equity_metrics
    per_symbol_expectancy: dict[str, float | None]
    passed: bool
```

### CLI_SUBCOMMAND_AND_TABLE
```python
# SOURCE: cli.py:111-126 + 199-209 (parser block; dispatch branch opens conn,
# resolves span defaults, calls _<name>_command(...) -> int, closes, sys.exit) and
# cli.py:283-294 + 356-358 (fixed-width header, dashed rule, "--" for None).
    elif args.command in ("backtest", "walkforward"):
        conn = connect(args.db)
        symbols = args.symbol if args.symbol else config.SYMBOLS
        start_ms = args.start if args.start else config.date_to_ms(config.BACKFILL_START)
        end_ms = args.end if args.end else int(time.time() * 1000)

def _fmt(v, spec=".4f") -> str:
    """Format a possibly-None metric value."""
    return format(v, spec) if v is not None else "--"
```

### TEST_STUB + TEST_CONVENTIONS
```python
# SOURCE: tests/test_backtest.py:469-492 — stub the name AT THE MODULE THAT
# IMPORTED IT. The new walkforward.buy_and_hold seam is stubbed the same way.
def per_symbol_fake_run_factory(peak_exp=0.01, n_trades=8, bad_symbol=None):
    def fake_run(conn, symbol, *, start_ms=None, end_ms=None, params=None, **kw):
        params = params or BacktestParams()
        base = -0.01 if symbol == bad_symbol else peak_exp
        exp = base * (1 - abs(params.rr_floor - 1.5) / 1.5) if base > 0 else base
        step = max(1, ((end_ms or 0) - (start_ms or 0)) // (n_trades + 1))
        return [make_trade(exp, exit_ts=(start_ms or 0) + step * (i + 1)) for i in range(n_trades)]
    return fake_run
monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())

# SOURCE: tests/test_backtest.py:23-46 — tier constants DERIVED from config (never
# hardcoded, so a tier shift can't leave tests green on old timeframes), autouse
# fixture clearing module caches around every test.
D_TRIG = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
START = 1_700_000_000_000

@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    engine.clear_caches(); yield; engine.clear_caches()

# SOURCE: tests/test_equity.py:66-81 — hand-computed closed-form literal with the
# derivation in the comment: mu = 0.003 exactly; sd = sqrt(1.45e-4);
# sharpe = (0.003/sd)*sqrt(365). The test never re-implements the code.
assert math.isclose(sharpe_ratio([0.01, -0.005, 0.02, 0.0, -0.01]), 4.7597449946, rel_tol=1e-9)

# SOURCE: tests/test_cli.py:288-304 — drive cli.main() through sys.argv to exercise
# the argparse wiring, not just the inner helper.
monkeypatch.setattr(sys, "argv", ["trading-bot", "--db", str(db_path), "gap-report", "--as-of", "2026-07-05"])
with pytest.raises(SystemExit) as exc_info: cli.main()
assert exc_info.value.code == 1
```

### CONFIG_BLOCK_PATTERN
```python
# SOURCE: config.py:115-136 — a dated, phase-labelled block naming WHY the value
# exists, what was measured, and whether it is frozen or sweepable.
# Phase 5 exit management, REPAIRED per .../phase4-7-code-review.md (measured 2026-07-27).
# TRAIL_ATR_MULTIPLE is deliberately a SEPARATE constant from ATR_STOP_MULTIPLE.
# Reusing k for both made the ratchet trail exactly as tight as the entry stop,
# which closed 91.3% of all trades at a median 10-hour hold ...
TRAIL_ENABLED = False
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `config.py` | UPDATE | Append the Phase 1 block: `STATE_DB_PATH`, `BENCHMARK_*`, `PNL_ATTRIBUTION_*` (contract §7 reserved prefixes) |
| `.gitignore` | UPDATE | `data/*.db` covers `state.db`; the `-wal`/`-shm` sidecars are **not** covered and contract §6 requires them ignored (they currently leak for `ohlcv.db` too) |
| `data/statestore.py` | CREATE | `state.db` connection layer (contract §2, §6) |
| `backtest/trials.py` | CREATE | Persistent trial ledger; owns the `trial_ledger` table |
| `backtest/benchmark.py` | CREATE | `buy_and_hold` + `BenchmarkResult` (contract §4) |
| `backtest/equity.py` | UPDATE | MEDIUM-5: `attribution=` on `daily_returns` / `compute_equity_metrics`; docstring correction |
| `backtest/walkforward.py` | UPDATE | `GATE_CONDITIONS`, `_evaluate_gate -> dict`, 3 new result fields, benchmark call, optional `ledger=` |
| `cli.py` | UPDATE | `benchmark` subcommand + `_benchmark_command`; richer `_walkforward_command` printout |
| `tests/test_benchmark.py` | CREATE | Synthetic units + the §0 acceptance anchor + CLI wiring |
| `tests/test_trials.py` | CREATE | Ledger schema, counting, persistence-across-reconnect, hashing |
| `tests/test_pnl_attribution.py` | CREATE | Spreading, conservation, clipping, mode equivalence, the measured kurtosis delta |
| `tests/test_equity.py` | UPDATE (deliberate) | `make_trade` gains `entry_ts`; the test named for the old convention gets renamed |
| `tests/test_backtest.py` | UPDATE (deliberate) | `make_trade` gains `entry_ts`; two fakes must place `entry_ts` in-span; `WalkForwardResult` gains 3 fields; new gate tests; `buy_and_hold` stub fixture |

## NOT Building
- **`framework/`, `plugins/`, `StrategyGraph`, `run_graph_backtest`** — Phase 3. `trials.py` takes `graph_hash` as an **opaque string** precisely so it does not depend on Phase 3.
- **`evolution/oracle.py`, campaigns, populations** — Phase 6. Phase 1 ships the ledger mechanism and the `ledger=` seam; Phase 6 makes bypassing it impossible.
- **`data/correlation.py` / effective-N** — Phase 2. §0b's correlations are quoted as context, never recomputed.
- **New symbols or backfill** — Phase 2 (17 extra symbols are already stored; contract §0a).
- **Changing any gate threshold.** `GATE_MIN_SHARPE=1.0`, `GATE_MIN_DSR=0.95`, `GATE_MAX_DRAWDOWN=0.25` untouched; only the condition *set* grows.
- **Changing `WF_MIN_TRADES`, `WF_OOS_DAYS`, or `DEFAULT_GRID`** — manufacturing a pass is forbidden (KNOWN-LIMITATIONS §4).
- **Position sizing, funding ingestion, vol-targeting** (§7); **a null that shorts, levers, or rebalances on another cadence**; **a non-zero risk-free rate** (`equity.py` uses 0, so the benchmark must too or the Sharpes are not comparable).
- **Fixing `scripts/build_review_chart.py`** (§8) or adding a pytest marker to `pyproject.toml`.

---

## Step-by-Step Tasks

### Task 1: Config block + `.gitignore` sidecars
- **ACTION**: Append a Phase 1 block at the **end** of `config.py`; add two `.gitignore` lines.
- **IMPLEMENT**:
  ```python
  # ---------------------------------------------------------------------------
  # v0.3.0 Phase 1: validation integrity. Nothing here changes the strategy —
  # these constants govern how it is MEASURED.
  # ---------------------------------------------------------------------------
  # Second SQLite DB for framework state (trial ledger; later review records /
  # strategy versions / populations). Deliberately NOT ohlcv.db: a corrupt
  # experiment log must never endanger 117 MB of irreplaceable price history.
  STATE_DB_PATH = "data/state.db"

  # Buy-and-hold null (KNOWN-LIMITATIONS §0: the old gate compared against ZERO,
  # so it "could have blessed a strategy worse than inaction").
  # BENCHMARK_REBALANCE is "daily" because that is what MEASURED 2026-07-27
  # reproduces §0's published basket (2.1628x / +29.32% / 0.7284 / 64.32% DD).
  # "none" (buy-once-hold) measures 2.0809x / 0.6998 / 66.91% and matches none of
  # the four published figures; it stays reachable so the choice remains a
  # measurement rather than a belief.
  BENCHMARK_TIMEFRAME = "1d"      # returns tier; must exist in storage.TIMEFRAME_MS
  BENCHMARK_REBALANCE = "daily"   # "daily" | "none"
  BENCHMARK_CHARGE_FEES = True    # one round-trip; NO funding — see benchmark.py docstring

  # MEDIUM-5 (KNOWN-LIMITATIONS §3). "spread" books each trade's pnl_pct evenly
  # across the UTC days it was open, entry_ts -> exit_ts inclusive. "exit_day" is
  # v0.2.0's behavior, kept reachable so the moment delta is MEASURED, not
  # asserted. Measured on the 23 OOS trades: kurtosis 31.2449 -> 15.5429, skew
  # 3.7883 -> 1.4034, annualized Sharpe 1.1750 -> 1.5120 (the Sharpe RISES: the
  # exit-day series' zero/spike alternation inflates its own variance).
  PNL_ATTRIBUTION_MODE = "spread"  # "spread" | "exit_day"
  ```
  `.gitignore`, beside the existing `data/*.db`: `data/*.db-wal` and `data/*.db-shm`.
- **MIRROR**: `CONFIG_BLOCK_PATTERN`.
- **IMPORTS**: none.
- **GOTCHA**: Touch **no** existing constant — `RR_FLOOR = 1.5` in particular stays (contract §7). `config.py` is shared: append only, at the end, in phase order.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot import config; print(config.STATE_DB_PATH, config.BENCHMARK_REBALANCE, config.PNL_ATTRIBUTION_MODE)"` → `data/state.db daily spread`; `git check-ignore -v data/state.db data/state.db-wal data/state.db-shm` → all ignored.

### Task 2: `data/statestore.py` — the `state.db` connection layer
- **ACTION**: Create `src/trading_bot/data/statestore.py`.
- **IMPLEMENT**: `connect(db_path: str | None = None) -> sqlite3.Connection`, defaulting to `config.STATE_DB_PATH`; `mkdir(parents=True, exist_ok=True)`; `sqlite3.connect(str(db_file), check_same_thread=False)`; `PRAGMA journal_mode=WAL`; `commit()`; return. Module-level `_db_lock = threading.Lock()` and `logger = logging.getLogger("trading_bot")`. **Creates no tables.**
  Docstring states: (a) why a second database — "ohlcv.db holds 117 MB of irreplaceable backfilled history; state.db holds derived, reproducible bookkeeping, so a corrupt experiment log is a `rm` and a re-run, never a data-loss event"; (b) each table's DDL is owned by the module that uses it, via `CREATE TABLE IF NOT EXISTS`, with **no central migration file to drift out of sync**; (c) timestamps are epoch **milliseconds**, UTC, never formatted strings (contract §6).
- **MIRROR**: `SQLITE_CONNECT` — identical shape minus the DDL.
- **IMPORTS**: `logging`, `sqlite3`, `threading`, `pathlib.Path`, `trading_bot.config`.
- **GOTCHA #1**: A **separate** lock object from `storage._db_lock` — state.db writes must not queue behind a long ohlcv.db backfill, and sharing one lock across two databases couples them for no benefit.
- **GOTCHA #2**: `:memory:` is a legal `db_path`; `Path(":memory:").parent` is `.`, which exists, so `mkdir` is harmless and tests can use in-memory DBs. Do not special-case it. No singleton, no `close_all()` — callers own their connection, as for `storage.connect()`.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot.data import statestore; c=statestore.connect(':memory:'); print(c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall())"` → `[]`.

### Task 3: `backtest/trials.py` — the persistent trial ledger
- **ACTION**: Create `src/trading_bot/backtest/trials.py`.
- **IMPLEMENT**:
  ```python
  LEDGER_TABLE = "trial_ledger"
  LEGACY_GRAPH_HASH = "legacy-donchian"   # sentinel for pre-framework engine runs

  @dataclass(frozen=True)
  class TrialRecord:
      campaign: str; graph_hash: str; params_hash: str
      start_ms: int; end_ms: int; ts: int          # epoch ms, UTC (contract §6)

  def ensure_schema(conn) -> None
  def stable_hash(obj) -> str                      # sha256(canonical json)[:16]
  def params_hash(params) -> str                   # dataclass | Mapping | None

  class TrialLedger:
      def __init__(self, conn, campaign: str)      # ValueError on empty campaign; calls ensure_schema
      def record(self, *, graph_hash, params_hash, start_ms, end_ms, ts=None) -> int  # POST-insert count()
      def count(self) -> int                       # rows for this campaign — what DSR is charged
      def distinct_count(self) -> int              # DISTINCT (graph_hash, params_hash, start_ms, end_ms) — reported only
      def records(self, limit=None) -> list[TrialRecord]
  ```
  Schema, in `ensure_schema`, inside `with _db_lock:`:
  ```sql
  CREATE TABLE IF NOT EXISTS trial_ledger (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      campaign TEXT NOT NULL, graph_hash TEXT NOT NULL, params_hash TEXT NOT NULL,
      start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL, ts INTEGER NOT NULL);
  CREATE INDEX IF NOT EXISTS idx_trial_ledger_campaign ON trial_ledger(campaign);
  ```
  `stable_hash` = `hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]`. `params_hash` accepts a frozen dataclass (`{f.name: getattr(p, f.name) for f in fields(p)}` when `is_dataclass(p) and not isinstance(p, type)`), a mapping, or `None` (→ `{}`). `record` defaults `ts` to `int(time.time() * 1000)`.
  Docstring states contract §4's resolution: one row per oracle evaluation, in `state.db`, so the count survives restarts and overnight runs; `n_trials` is the **cumulative** count for the campaign, across generations; "this will produce brutal DSR values — that is the correct answer and must not be softened (PRD honesty clause)"; `graph_hash` is opaque here so this module imports nothing from `trading_bot.framework` (Phase 3), which Phase 1 gates.
- **MIRROR**: `LOCKED_WRITE`, `FROZEN_RESULT_DATACLASS`.
- **IMPORTS**: `hashlib`, `json`, `logging`, `time`, `dataclasses.{dataclass, fields, is_dataclass}`, `from trading_bot.data.statestore import _db_lock` — importing the private lock is deliberate: state.db has one lock and every table owner shares it, mirroring how `storage.py` keeps its lock "beside the connection's accessor functions."
- **GOTCHA #1**: **No `UNIQUE` constraint.** Re-evaluating a configuration is a legitimate separate row — the ledger measures evaluations *performed*, and silently deduping would understate `n_trials` in the flattering direction. `distinct_count()` serves whoever wants the other number.
- **GOTCHA #2**: Never use Python's `hash()`. `PYTHONHASHSEED` randomizes str hashing per process, so a ledger keyed on it would double-count the same configuration across restarts — precisely the failure this module prevents.
- **GOTCHA #3**: The f-string interpolates a **module constant** table name only; all values bind with `?`. Do not extend it to caller-supplied identifiers.
- **GOTCHA #4**: `default=str` makes `stable_hash` total rather than raising mid-run, at the cost that two objects with the same `repr` collide. Acceptable — every params value here is bool/int/float/str — but document it in a test docstring rather than engineering around it.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_trials.py -v`.

### Task 4: MEDIUM-5 — holding-day P&L attribution in `equity.py`
- **ACTION**: Modify `src/trading_bot/backtest/equity.py`.
- **IMPLEMENT**: module constants beside `DAY_MS`/`PERIODS_PER_YEAR` (`equity.py:22-23`) — `ATTRIBUTION_SPREAD = "spread"`, `ATTRIBUTION_EXIT_DAY = "exit_day"`, `ATTRIBUTION_MODES = (…)`; then `daily_returns(trades, start_ms, end_ms, *, attribution: str = ATTRIBUTION_SPREAD)`. Validate the mode (`raise ValueError` otherwise — a silent fallback would make the gate's definition of Sharpe depend on a typo); keep the existing `end_ms <= start_ms → []`, `first_day`, `n_days`, `rets` scaffolding verbatim (`equity.py:40-44`). Then per trade:
  ```python
      if attribution == ATTRIBUTION_EXIT_DAY:
          d = t.exit_ts // DAY_MS - first_day
          if 0 <= d < n_days: rets[d] += t.pnl_pct
          else: logger.warning("trade exit_ts %d outside metrics span; dropped", t.exit_ts)
          continue
      # ATTRIBUTION_SPREAD. entry_ts/exit_ts already exist on Trade
      # (engine.py:124,128) — NO schema change is needed.
      d_first = t.entry_ts // DAY_MS - first_day
      d_last  = t.exit_ts  // DAY_MS - first_day
      if d_last < d_first:            # pragma: no cover - defensive invariant
          logger.warning("trade exit_ts %d precedes entry_ts %d; booked on the exit day",
                         t.exit_ts, t.entry_ts)
          d_first = d_last
      lo, hi = max(d_first, 0), min(d_last, n_days - 1)
      if hi < lo:
          logger.warning("trade %d->%d does not overlap the metrics span; dropped",
                         t.entry_ts, t.exit_ts)
          continue
      # Divide by the CLIPPED day count, not the full holding length, so a trade
      # straddling a span boundary still contributes its WHOLE pnl_pct: sum() is
      # then invariant to the attribution mode for any overlapping trade
      # (measured: 0.16509935688760607 both ways on the 23 OOS trades). Dividing
      # by full length and dropping the out-of-span share would make total
      # reported P&L depend on where the window is cut.
      share = t.pnl_pct / (hi - lo + 1)
      for d in range(lo, hi + 1): rets[d] += share
  ```
  `compute_equity_metrics(trades, start_ms, end_ms, n_trials=1, *, attribution=ATTRIBUTION_SPREAD)` passes it through and **adds** three keys: `"skew"`, `"kurtosis"` (the moments the DSR was computed on, so a report can quote them without recomputing) and `"attribution"`. Every existing key keeps its meaning.
  Module docstring (`equity.py:1-13`): replace "Each trade contributes its full net pnl_pct on its EXIT day" with the spread description, naming the change — "v0.2.0 booked the whole pnl_pct on the EXIT day, producing daily skew 3.79 / kurtosis 31.24 on 23 trades and blocking every significance test (KNOWN-LIMITATIONS §3, finding MEDIUM-5); pass `attribution=ATTRIBUTION_EXIT_DAY` to reproduce it" — keeping the existing "Days with no open trade are zero-return days and are INCLUDED: idle time is real time" sentence.
- **MIRROR**: `NONE_ON_UNDEFINED`; the warning string at `equity.py:50` is kept verbatim on the exit-day path.
- **IMPORTS**: none new.
- **GOTCHA #1**: **`equity.py` must stay config-free.** Its docstring promises "Pure computation, no I/O, no config coupling" (`equity.py:12`). The default is the module constant, **not** `config.PNL_ATTRIBUTION_MODE`; the config constant is read by the already-config-coupled callers (`walkforward.py`, `cli.py`), which is what makes flipping it flip the gate. A grep in Validation enforces this.
- **GOTCHA #2**: Adding return-dict keys is safe — `tests/test_equity.py:179-181` asserts `key in em` for a fixed list, not dict equality. All 6 `compute_equity_metrics` call sites (`cli.py:386`, `walkforward.py:397`, 4 in tests) were checked; none compares key sets.
- **GOTCHA #3**: **Behavior change** when `exit_ts` is after `end_ms` while `entry_ts` is inside: exit-day mode **dropped** it, spread mode books its in-span days. Deliberate (a trade open during the window did affect equity during the window) but a span-clipped run can now show non-zero returns where v0.2.0 showed zeros. Note in the phase report.
- **GOTCHA #4**: `n_days` comes from the span, never from the trades. Do not derive it from min/max trade timestamps — calendar completeness is what keeps idle time in the Sharpe denominator (`equity.py:8-9`).
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_pnl_attribution.py tests/test_equity.py -v`.

### Task 5: `backtest/benchmark.py` — the buy-and-hold null
- **ACTION**: Create `src/trading_bot/backtest/benchmark.py`.
- **IMPLEMENT**:
  ```python
  REBALANCE_DAILY = "daily"; REBALANCE_NONE = "none"
  REBALANCE_MODES = (REBALANCE_DAILY, REBALANCE_NONE)
  METRIC_KEYS = ("total_return", "ann_return_pct", "sharpe", "sortino",
                 "max_drawdown_pct", "n_days")          # contract §4 fixes this set

  @dataclass(frozen=True)
  class BenchmarkResult:
      per_symbol: dict[str, dict]   # symbol -> METRIC_KEYS bundle; None where undefined
      basket: dict                  # equal-weight, daily-rebalanced, same keys
      start_ms: int
      end_ms: int

  def _metrics_from_returns(rets: list[float]) -> dict
  def _close_returns(conn, symbol, *, start_ms, end_ms, timeframe, charge_fees) -> list[float]
  def buy_and_hold(conn, symbols, *, start_ms: int, end_ms: int) -> BenchmarkResult
  ```
  `_metrics_from_returns`: empty → `dict.fromkeys(METRIC_KEYS, None) | {"n_days": 0}`; else the wipe-out-guarded compounding loop copied from `equity.py:188-210`, then `total_return = equity` (an equity **multiple**: 2.21 means 2.21×), `ann_return_pct = equity ** (PERIODS_PER_YEAR / n) - 1.0`, and `sharpe_ratio` / `sortino_ratio` / `max_drawdown` **called from `equity.py`, not reimplemented** — the benchmark's Sharpe must come from the identical function the strategy's does or the two are not comparable. `ann_return_pct` and `max_drawdown_pct` are **fractions** (0.303 = 30.3%), matching `compute_equity_metrics`' misleading-but-established naming (`equity.py:208`) so `cli._fmt`'s `'.2%'` specs work unchanged.
  `_close_returns`: `load_candles(...)`, `closes = [r[4] for r in rows]`, `< 2` bars → warn + `[]`, else simple close-to-close returns; if `charge_fees`, book one round-trip on the first day: `rets[0] = (1 + rets[0]) * (1 - 2*(config.FEE_PCT + config.SLIPPAGE_PCT)) - 1`.
  `buy_and_hold`: read `config.BENCHMARK_TIMEFRAME` / `BENCHMARK_CHARGE_FEES` / `BENCHMARK_REBALANCE` **at call time**; unknown mode → `ValueError`. Per symbol fill `per_symbol[symbol]` and collect non-empty series. Basket: no series → warn, undefined; **differing** lengths → warn (`"symbols disagree on bar count %s; basket undefined (run gap-report)"`) and leave undefined, refusing rather than zip-truncating to the shortest (which would silently shorten the span the gate compares on); `REBALANCE_DAILY` → `[statistics.fmean([series[s][i] for s in series]) for i in range(n)]`; `REBALANCE_NONE` → compound each symbol into an equity path from 1.0, take the equal-weighted sum path, difference it back into returns.
  Docstring carries three stated assumptions:
  1. **Costs** — one round-trip of fees+slippage, **no funding**. The null is a spot-equivalent hold, not a perpetual position. Charging the frozen pessimistic `FUNDING_PCT_PER_DAY = 0.0001` over 1095 days would cost the null ~11.6% and hand the strategy an ~11-point head start a real operator could dodge by buying spot. That placeholder exists to make the **strategy's** costs pessimistic; applying it to the benchmark makes the comparison flattering — the opposite of its purpose. The fee *is* charged so the null is not costless, and it is measurably immaterial (BTC 2.2135× → 2.2104×, Sharpe 0.8002 → 0.7992).
  2. **Basket** — equal-weight, **daily-rebalanced**, because measured 2026-07-27 that reproduces all four of §0's basket figures while buy-once-hold (2.0809× / 0.6998 / 66.91%) matches none.
  3. **Annualization** — a true CAGR on a multi-year span (§0's is exactly 1095 days = 3 years); a **90-day extrapolation** on a walk-forward holdout, exactly as the strategy's is. The comparison stays valid because both sides are annualized identically on the identical span; the level is not quotable, and callers printing a short-span figure must say so.
- **MIRROR**: `DEFENSIVE_NUMERIC`, `NONE_ON_UNDEFINED`, `FROZEN_RESULT_DATACLASS`.
- **IMPORTS**: `logging`, `statistics`, `dataclasses.dataclass`, `trading_bot.config`, `from trading_bot.backtest.equity import PERIODS_PER_YEAR, max_drawdown, sharpe_ratio, sortino_ratio`, `from trading_bot.data.storage import load_candles`.
- **GOTCHA #1**: `load_candles` returns **tuples** and close is index **4** (`storage.py:230`). An off-by-one here silently benchmarks the low.
- **GOTCHA #2**: Both bounds are **inclusive** (`storage.py:216-217`): the §0 span yields **1096 bars → 1095 returns**, and `365/1095 = 1/3` exactly. Do not subtract a day "to make it exclusive."
- **GOTCHA #3**: A symbol with `< 2` bars gets all-`None` metrics and is **excluded from the basket** — a 2-symbol honest basket beats a 3-symbol one padded with zeros.
- **GOTCHA #4**: Import nothing from `walkforward.py`; `walkforward` imports `benchmark`, and the reverse is circular.
- **GOTCHA #5**: No `if conn is None` branch — see Task 6 GOTCHA #2 for why the test seam is a monkeypatch instead.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_benchmark.py -v`; `.venv/bin/python -m trading_bot.cli benchmark --start 2023-07-27 --end 2026-07-26`.

### Task 6: Extend THE GATE — `GATE_CONDITIONS`, dict verdict, benchmark, ledger
- **ACTION**: Modify `src/trading_bot/backtest/walkforward.py`.
- **IMPLEMENT**: beside the untouched `GATE_MIN_*` constants (`walkforward.py:78-80`):
  ```python
  # The gate's condition set, in report order. Contract §4 fixes these names;
  # _evaluate_gate returns a dict keyed by exactly this tuple, and
  # passed == all(gate.values()). Adding a name here without adding it in
  # _evaluate_gate is caught by a test.
  GATE_CONDITIONS: tuple[str, ...] = (
      "sample_adequacy",          # n_trades >= min_trades
      "sharpe",                   # >= GATE_MIN_SHARPE
      "dsr",                      # >  GATE_MIN_DSR
      "max_drawdown",             # <= GATE_MAX_DRAWDOWN
      "per_symbol_expectancy",    # EVERY symbol > 0 (AND, not average)
      "beats_benchmark_return",   # NEW: ann_return_pct > basket ann_return_pct
      "beats_benchmark_sharpe",   # NEW: sharpe        > basket sharpe
  )

  def _evaluate_gate(oos_metrics, oos_equity, per_symbol_expectancy,
                     min_trades, benchmark) -> dict[str, bool]:
      b = benchmark.basket
      sharpe, ann = oos_equity["sharpe"], oos_equity["ann_return_pct"]
      dd, dsr = oos_equity["max_drawdown_pct"], oos_equity["dsr"]
      return {
          "sample_adequacy": oos_metrics["n_trades"] >= min_trades,
          "sharpe": sharpe is not None and sharpe >= GATE_MIN_SHARPE,
          "dsr": dsr is not None and dsr > GATE_MIN_DSR,
          "max_drawdown": dd is not None and dd <= GATE_MAX_DRAWDOWN,
          "per_symbol_expectancy": bool(per_symbol_expectancy) and all(
              e is not None and e > 0 for e in per_symbol_expectancy.values()),
          "beats_benchmark_return": (ann is not None
              and b["ann_return_pct"] is not None and ann > b["ann_return_pct"]),
          "beats_benchmark_sharpe": (sharpe is not None
              and b["sharpe"] is not None and sharpe > b["sharpe"]),
      }
  ```
  Its docstring keeps the existing "Fails safely … never raises" promise and adds: "a missing benchmark is a FAIL, not a pass — if the null could not be computed, the comparison was not made."
  `WalkForwardResult` — three fields inserted **before** `passed` (nothing reordered; `passed` stays last, as contract §4's block shows):
  ```python
      gate: dict[str, bool]        # per-condition verdicts, keyed by GATE_CONDITIONS
      benchmark: BenchmarkResult   # the null over the SAME OOS span
      n_trials_used: int           # what the DSR was actually charged
      passed: bool                 # == all(gate.values()); every existing caller still works
  ```
  `walk_forward_pooled` gains keyword-only `ledger=None` (a `trials.TrialLedger`). `_pooled_expectancy` gains `ledger=None` and, when supplied, records **one row per pooled evaluation** — one configuration on one span, *not* one per symbol, since pooling is a single evaluation:
  ```python
      if ledger is not None:
          ledger.record(graph_hash=trials.LEGACY_GRAPH_HASH,
                        params_hash=trials.stable_hash(combo),
                        start_ms=start, end_ms=end)
  ```
  Thread `ledger=ledger` through every call site (`walkforward.py:325`, `335-337`, `339`) **including `_positive_neighbour_stats`**. Replace line 383 with:
  ```python
      # With a ledger, the DSR charge is the campaign's CUMULATIVE evaluation
      # count, persisted in state.db so it survives restarts and overnight runs
      # (contract §4) — and it INCLUDES the neighbour probes the ledgerless
      # default deliberately omits (see the comment above). The ledgered number
      # is therefore strictly larger and the DSR strictly worse. That is the point.
      if n_trials is not None:   n_trials_used = n_trials
      elif ledger is not None:   n_trials_used = ledger.count()
      else:                      n_trials_used = len(combos) * max(1, len(folds))
  ```
  In the OOS block pass `attribution=config.PNL_ATTRIBUTION_MODE` to `compute_equity_metrics`, then:
  ```python
      # The null, on the SAME span the strategy is scored on. Over a 90-day
      # holdout its ann_return_pct is a 90-day extrapolation exactly as the
      # strategy's is — identical annualization on an identical span keeps the
      # COMPARISON valid even though neither LEVEL is quotable.
      benchmark = buy_and_hold(conn, symbols, start_ms=oos_start, end_ms=oos_end)
      gate = _evaluate_gate(oos_metrics, oos_equity, per_symbol_expectancy,
                            min_trades, benchmark)
      passed = all(gate.values())
  ```
- **MIRROR**: `GATE_FAILS_SAFE`, `FROZEN_RESULT_DATACLASS`, and the existing `| None = None` config-fallback kwarg idiom (`walkforward.py:263-268`).
- **IMPORTS**: `from trading_bot.backtest.benchmark import BenchmarkResult, buy_and_hold`; `from trading_bot.backtest import trials`.
- **GOTCHA #1**: `passed` **must stay a plain `bool`**. `all(...)` already returns one. Do not store the dict in `passed` and do not make it a property — `tests/test_backtest.py:522,551,581` assert `result.passed is False`, which is identity-sensitive.
- **GOTCHA #2**: `buy_and_hold` must be a **module-level name in `walkforward`** (imported, then called bare) so tests can `monkeypatch.setattr(walkforward, "buy_and_hold", …)` — the seam `run_backtest` already provides (`tests/test_backtest.py:504`). Calling `benchmark.buy_and_hold(...)` through the module object defeats it. This matters because `conn is None` in every existing walk-forward test.
- **GOTCHA #3**: Every `WalkForwardResult(...)` construction uses keyword args (`walkforward.py:402`, `tests/test_backtest.py:607`), so inserting fields before `passed` is safe for production but **breaks the test site**, which supplies none of them. That is a deliberate recorded test edit (Task 11), **not** a reason to give the new fields defaults: a result constructible without a benchmark is a result that can silently claim a verdict without a null.
- **GOTCHA #4**: `_evaluate_gate` gains a 5th positional parameter. One caller exists (`walkforward.py:400`); no test calls it directly (verified by grep).
- **GOTCHA #5**: `per_symbol_expectancy` previously passed **vacuously** on an empty dict (`for … in {}.values()` never fails). `bool(...) and all(...)` fixes that — zero symbols is not "every symbol positive." A change in the strict direction; no existing test passes an empty symbol list.
- **GOTCHA #6**: `_positive_neighbour_stats` must forward `ledger`, or the neighbour probes go uncounted and the ledger understates the search — the exact dishonesty it exists to prevent.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_backtest.py -v`; then `.venv/bin/python -c "import dataclasses; from trading_bot.backtest import walkforward as w; print(w.GATE_CONDITIONS); print([f.name for f in dataclasses.fields(w.WalkForwardResult)])"` → 7 names; field list ends `… per_symbol_expectancy, gate, benchmark, n_trials_used, passed`.

### Task 7: CLI — `benchmark` subcommand and a gate printout that names its failures
- **ACTION**: Modify `src/trading_bot/cli.py`.
- **IMPLEMENT**:
  - Imports: `from trading_bot.backtest.benchmark import METRIC_KEYS, buy_and_hold`; add `GATE_CONDITIONS` to the existing `walkforward` import.
  - Parser registered **after** `wf_parser` (`cli.py:144`) per contract §7's phase order: `subparsers.add_parser("benchmark", help="Buy-and-hold null hypothesis: per-symbol and equal-weight basket return / Sharpe / Sortino / max drawdown over a span")` with `--symbol` (`action="append"`), `--start`, `--end` (`type=_date_arg`) — copied from the `backtest` parser (`cli.py:112-126`).
  - Dispatch: **extend the existing tuple** to `elif args.command in ("backtest", "walkforward", "benchmark"):` with an inner `elif args.command == "benchmark": exit_code = _benchmark_command(...)`. Do not add a separate branch — the span-default resolution is shared and duplicating it invites drift.
  - `_print_benchmark(b, indent="") -> None`: fixed-width header `symbol / total / ann / sharpe / sortino / max_dd / n_days`, dashed rule, then `list(b.per_symbol.items()) + [("BASKET", b.basket)]`; `total` as `f"{v:.4f}x"` or `"--"`, everything else through `_fmt` (`'+.2%'`, `'.3f'`, `'.3f'`, `'.2%'`).
  - `_benchmark_command(conn, symbols, *, start_ms, end_ms) -> int`: call `buy_and_hold`, print the table; if `0 < basket["n_days"] < 365` print `NOTE: ann is a {n}-day extrapolation, not a compound annual rate.`; collect names whose bundle has any `None` in `METRIC_KEYS`, print `INCOMPLETE: … (insufficient data in span)` and return 1, else 0 — mirroring `_regime_command`'s "1 when a symbol had insufficient data" contract (`cli.py:296-299`).
  - `_walkforward_command`: insert before the existing `print(f"GATE: …")` at `cli.py:437`, keeping that line byte-identical:
    ```python
        print("benchmark (equal-weight basket, same OOS span):")
        _print_benchmark(result.benchmark, indent="  ")
        print(f"n_trials charged to DSR: {result.n_trials_used}")
        print("gate conditions:")
        for name in GATE_CONDITIONS:
            print(f"  {name:<24} {'PASS' if result.gate[name] else 'FAIL'}")
    ```
- **MIRROR**: `CLI_SUBCOMMAND_AND_TABLE`; reuse `_fmt`/`_print_metrics` (`cli.py:356-368`) as-is.
- **GOTCHA #1**: Iterate `GATE_CONDITIONS`, not `result.gate.items()` — the tuple is the canonical report order, and iterating it makes a missing key an immediate `KeyError` rather than a quietly short report.
- **GOTCHA #2**: `tests/test_backtest.py:622,626` assert `"GATE: PASS"` / `"GATE: FAIL"` in stdout. Keep that final line exactly.
- **GOTCHA #3**: Never `format(None, '+.2%')` — route every possibly-`None` value through `_fmt`.
- **VALIDATE**: `.venv/bin/python -m trading_bot.cli benchmark --start 2023-07-27 --end 2026-07-26; echo "exit=$?"` (§0 table, exit 0); `… --symbol BTCUSDT --start 2026-07-20 --end 2026-07-26` (6-day table with the extrapolation NOTE).

### Task 8: `tests/test_pnl_attribution.py`
- **ACTION**: Create the file (reserved for Phase 1, contract §8). Local helper `trade(pnl, *, entry_ts, exit_ts)`.
- **IMPLEMENT** — `class TestSpreadAttribution`: `test_three_day_hold_splits_evenly` (`trade(0.03, entry_ts=10, exit_ts=2*DAY_MS+10)` over 4 days → `[0.01, 0.01, 0.01, 0.0]`); `test_same_day_hold_is_identical_to_exit_day`; `test_sum_is_conserved` (**the invariant** — two overlapping multi-day trades, `Σspread == Σexit_day == 0.02`); `test_entry_before_span_conserves_whole_pnl` (entry −2d, exit day 1 → `Σ == 0.02`, day 2 still `0.0`); `test_exit_after_span_now_contributes` (**behavior change**: spread conserves `0.02`, exit-day gives `[0.0]*3`); `test_no_overlap_is_dropped`; `test_unknown_mode_raises` (`ValueError`); `test_spread_lowers_kurtosis_on_a_synthetic_spiky_series` (5 trades × 4-day holds in a 40-day span → `kurt_spread < kurt_exit_day`). `class TestConfigDefault`: `config.PNL_ATTRIBUTION_MODE == ATTRIBUTION_SPREAD`; `"attribution"`, `"skew"`, `"kurtosis"` present in the metrics dict.

  Then `class TestMedium5MeasuredDelta`, guarded by `@pytest.mark.skipif(not Path(config.DB_PATH).exists(), reason="requires the real OHLCV store (data/ohlcv.db)")`, with `_oos_trades()` calling `engine.clear_caches()` then, for each `config.SYMBOLS`, `run_backtest(conn, sym, start_ms=end_ms - config.WF_OOS_DAYS*DAY_MS, end_ms=config.date_to_ms("2026-07-26"), params=BacktestParams(trail_enabled=False, target_enabled=False), max_hold_bars=48)`:
  - `test_exit_day_reproduces_the_published_moments` — `len(trades) == 23`; skew `≈ 3.7883`, kurtosis `≈ 31.2449` (`rel=1e-3`).
  - `test_spread_drops_kurtosis_below_thirty` — skew `≈ 1.4034`, kurtosis `≈ 15.5429`, **and** `kurt < 30.0` (the PRD success signal, literally).
  - `test_total_pnl_is_identical_across_modes`.
  - `test_spread_raises_sharpe_and_dsr_still_fails` — `old["sharpe"] ≈ 1.1750`, `new["sharpe"] ≈ 1.5120`, `new > old`, **both** DSR `< 0.95`. Docstring: "the fix moves Sharpe in the FLATTERING direction and does NOT rescue DSR; both recorded so neither is discovered later as a surprise."
- **MIRROR**: `TEST_CONVENTIONS`; contract §8's "regression tests pin each repaired finding, with the finding id in the test name or docstring."
- **IMPORTS**: `math`, `pathlib.Path`, `pytest`, `trading_bot.config`, `trading_bot.backtest.engine` (+ `BacktestParams`, `Trade`, `run_backtest`), `from trading_bot.backtest.equity import ATTRIBUTION_EXIT_DAY, ATTRIBUTION_SPREAD, DAY_MS, _skew_kurt, compute_equity_metrics, daily_returns`, `from trading_bot.data.storage import connect`. Importing the private `_skew_kurt` is consistent with `tests/test_equity.py`, which imports module internals freely.
- **GOTCHA #1**: A `skipif` is **not** a pass. Validation requires `0 skipped`; a skip means the anchor never ran.
- **GOTCHA #2**: `trail_enabled=False`, `target_enabled=False`, `max_hold_bars=48` are §1's **gate-selected** parameters. Change any and the trade count leaves 23 and the pinned moments become meaningless.
- **GOTCHA #3**: Pin the **measured** 3.7883 / 31.2449 at `rel=1e-3`, not §3's rounded 3.79 / 31.24 — the rounded values are a looser, less useful test.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_pnl_attribution.py -v` — all pass, **0 skipped**.

### Task 9: `tests/test_trials.py`
- **ACTION**: Create the file. Fixture `ledger(tmp_path)` → `trials.TrialLedger(statestore.connect(str(tmp_path / "state.db")), "test-campaign")` — **file-backed, not `:memory:`**, so the reconnect test is meaningful.
- **IMPLEMENT**:
  - `TestSchema`: `connect()` creates no tables; `ensure_schema` twice raises nothing; `config.STATE_DB_PATH == "data/state.db"`.
  - `TestCounting`: `record()` returns 1 then 2 (post-insert cumulative); `test_repeat_evaluations_are_separate_rows` — 3 identical records → `count()==3`, `distinct_count()==1`, docstring "no UNIQUE constraint: the ledger counts evaluations PERFORMED, and deduping would understate n_trials in the flattering direction"; campaigns isolated; empty campaign → `ValueError`.
  - `TestPersistence`: `test_count_survives_reconnect` — record, reopen the same path, `count() == 1`; docstring "the whole reason the ledger is on disk: an overnight run that restarts must not reset its own degrees-of-freedom count". `ts` is an `int` > `1_600_000_000_000` (epoch **ms**, contract §6).
  - `TestHashing`: order-independence (`{a:1,b:2}` == `{b:2,a:1}`); a **pinned literal digest** for `stable_hash("x")`, docstring "must be reproducible ACROSS PROCESSES; PYTHONHASHSEED would break a ledger keyed on `hash()`"; `params_hash(BacktestParams())` equals the hash of its field dict and `params_hash(None) == stable_hash({})`; one test documenting the `default=str` collision limitation.
  - `TestWalkForwardIntegration`: with a ledger passed to `walk_forward_pooled` (`TEST_STUB` fakes), `result.n_trials_used == ledger.count()` and it **exceeds** the ledgerless `len(combos) * len(folds)` because neighbour probes are counted; without a ledger the legacy count is reproduced exactly.
- **MIRROR**: `tests/test_storage.py`'s `tmp_path` SQLite fixtures; class-per-area grouping.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_trials.py -v`.

### Task 10: `tests/test_benchmark.py`
- **ACTION**: Create the file, including **the acceptance anchor**. Tier constants derived: `BENCH_TF = config.BENCHMARK_TIMEFRAME`, `D_BENCH = TIMEFRAME_MS[BENCH_TF]`. Helper `seed_closes(conn, symbol, closes)` seeding one bar per `D_BENCH` with `o=h=l=c`.
- **IMPLEMENT**:
  - `TestSyntheticSingleSymbol`: doubling over two bars (`100 → 200` → `total_return = 2.0 * (1 - 0.0014) = 1.9972`); flat series → Sharpe `None` (not 0.0); single bar → all-`None`, `n_days == 0`; missing symbol → all-`None`; `n_days == bars − 1` (the inclusive-bounds trap); wipe-out → `ann_return_pct == -1.0`, mirroring `tests/test_equity.py:201-220`.
  - `TestBasketConstruction`: daily rebalance **is** the mean of daily returns; `BENCHMARK_REBALANCE="none"` differs; unknown mode raises; a symbol without data is **excluded, not zero-padded**; ragged bar counts → basket all-`None` (docstring: "refuse rather than zip-truncate — truncating would silently shorten the span the gate compares on").
  - `TestCostAssumption`: the round-trip fee is charged **once** (not twice, not per day); raising `FUNDING_PCT_PER_DAY` 100× must **not** move the benchmark (spot-equivalent hold).
  - `TestBenchmarkCli`: `_benchmark_command` with a stubbed `cli.buy_and_hold` prints `BASKET`, exits 0; an all-`None` bundle prints `INCOMPLETE`, exits 1; `test_main_benchmark_argv` drives `cli.main()` via `sys.argv` against an empty tmp DB → `SystemExit(1)`.
  - `TestKnownLimitationsSection0Anchor` — `@pytest.mark.skipif(not Path(config.DB_PATH).exists(), …)`, `SPAN = dict(start_ms=config.date_to_ms("2023-07-27"), end_ms=config.date_to_ms("2026-07-26"))`, class-scoped `result` fixture calling `buy_and_hold(connect(), list(config.SYMBOLS), **SPAN)`:
    - `test_span_is_1095_daily_returns` — `basket["n_days"] == 1095`; docstring "1096 inclusive bars → 1095 returns → 365/1095 = 1/3 exactly, so the annualized column is a true 3-year CAGR, NOT an extrapolation."
    - `test_per_symbol_reproduces_section_0`, parametrized `("BTCUSDT", 2.21, 0.303, 0.80, 0.530)`, `("ETHUSDT", 1.03, 0.009, 0.34, 0.676)`, `("SOLUSDT", 3.00, 0.443, 0.85, 0.763)`.
    - `test_basket_reproduces_section_0` — `2.17 / 0.294 / 0.73 / 0.643`.
    - `test_buy_once_hold_does_NOT_reproduce_section_0` — with `BENCHMARK_REBALANCE="none"`, basket `≈ 2.081×` and **not** `≈ 2.17×`. Docstring: "the measurement that decided the construction; pinned so a future 'simplification' to buy-once cannot pass the anchor above."
    - `test_ninety_day_oos_benchmark_is_negative` — over `end_ms − WF_OOS_DAYS`, basket Sharpe `≈ -1.303`, `ann_return_pct < 0`. Docstring: "§0's other half: over the gate's holdout the null LOSES, which is why v0.2.0 passes both new gate conditions while still failing on sample size and DSR. Pinned so that fact is a test, not a memory."

    **Tolerances, in the class docstring**: `total_return` `abs=0.01`, `ann_return_pct` `abs=0.002`, `sharpe` `abs=0.01`, `max_drawdown_pct` `abs=0.001`. Each is **one rounding unit of the published figure** ("2.21x" is 2 d.p.; "+30.3%" is 0.1 pp; "0.80" is 2 d.p.; "53.0%" is 0.1 pp) — the tightest bound that cannot fail on the fee-booking choice, since measured no-fee vs with-fee differ by at most 0.0031 on `total_return` and 0.0010 on Sharpe. Tighter would pin an assumption the published table never made; looser would stop being evidence.
- **MIRROR**: `TEST_CONVENTIONS`; `tests/test_equity.py:201-220`'s wipe-out guard.
- **GOTCHA #1**: The class-scoped `result` fixture opens the **real** DB once, read-only. No `tmp_path`. `engine.clear_caches()` is irrelevant — `benchmark.py` has no cache.
- **GOTCHA #2**: `monkeypatch.setattr(config, "BENCHMARK_REBALANCE", …)` works only because `buy_and_hold` reads the attribute at **call** time — the discipline `FADE_ENABLED` relies on (`config.py:93-94`). Do not hoist it to a module constant.
- **GOTCHA #3**: Seed synthetic bars with `o=h=l=c`. With distinct highs/lows, an off-by-one on the close index (`r[4]`) would still produce plausible numbers and the bug would ship.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_benchmark.py -v` — all pass, **0 skipped**.

### Task 11: Deliberate, recorded edits to `tests/test_equity.py` and `tests/test_backtest.py`
**Recorded behavior changes, not collateral damage.** Every item is named.

**The finding that reshapes this task.** `tests/test_equity.py:25-31`'s `make_trade` sets `entry_ts=exit_ts`, so under spread attribution a zero-length hold books on its single day — **identical** to exit-day booking. Verified case by case:

| Test | `entry_ts` vs `exit_ts` | Under spread | Verdict |
|---|---|---|---|
| `TestDailyReturns::test_single_trade_placed_on_exit_day` | equal | `[0.0, 0.02, 0.0]` unchanged | **passes**; the NAME is now false |
| `TestDailyReturns::test_two_trades_same_day_are_summed` | equal | `0.03` on day 1 unchanged | **passes** |
| `TestDailyReturns::test_trade_outside_span_dropped` | equal, day 10 vs 3-day span | no overlap → dropped | **passes** |
| `TestDailyReturns::test_empty_span_returns_empty` | n/a | unchanged | **passes** |
| `TestComputeEquityMetrics::*` (4 tests) | equal | unchanged | **pass** |

So **no assertion in `test_equity.py` breaks** — what breaks is the file's *honesty*. Required edits:
1. `make_trade` (line 25) — add `entry_ts=None` defaulting to `exit_ts`, preserving every existing call.
2. Rename `test_single_trade_placed_on_exit_day` → `test_zero_length_hold_books_on_its_single_day`, docstring: "A trade whose entry and exit land on the same UTC day books there under BOTH attribution modes — which is why this class survived the MEDIUM-5 fix unchanged. Multi-day attribution is covered in tests/test_pnl_attribution.py."
3. Add one test asserting the module default is `ATTRIBUTION_SPREAD` and that a 3-day hold splits — a signpost so the next reader does not conclude from this file that exit-day booking is still the convention.

**`tests/test_backtest.py`** — here assertions genuinely break and two more silently *rot*:

4. `make_trade` (line 49) — add `entry_ts=None` defaulting to `START`. Required by the next two items.
5. `per_symbol_fake_run_factory` (line 469) — **BUG THIS EXPOSES**: it calls `make_trade(exp, exit_ts=(start_ms or 0) + step*(i+1))`, leaving `entry_ts = START = 1_700_000_000_000` while the test span is `0 … 25*DAY_MS`. Under spread attribution `entry_ts // DAY_MS = 19675` is far **after** the exit day, so `hi < lo` and **every trade is dropped** — folds would score an all-zero series and the class would keep passing for a degenerate reason. Fix: pass `entry_ts=exit_ts` (or `exit_ts − D_TRIG` for a genuinely multi-day fake) and extend the factory docstring, which already explains why `exit_ts` is spread across the span, to say the same of `entry_ts`.
6. `volatile_fake_run` inside `TestWalkForwardPooled::test_gate_requires_sharpe_and_dsr` (line 531) — same defect, same fix.
7. `TestBacktestCli::test_walkforward_command_exit_codes` (line 603) — the `WalkForwardResult(...)` call must supply `gate={c: passed for c in walkforward.GATE_CONDITIONS}`, a minimal `benchmark=BenchmarkResult(per_symbol={SYMBOL: …}, basket=…, start_ms=0, end_ms=1)` with all six `METRIC_KEYS` present, and `n_trials_used=1`. Both `assert "GATE: PASS"/"GATE: FAIL"` assertions stay as they are.
8. `TestWalkForwardPooled` gains an autouse fixture stubbing the benchmark, because `conn=None` in this class:
   ```python
       @pytest.fixture(autouse=True)
       def _stub_benchmark(self, monkeypatch):
           """conn is None here, so buy_and_hold cannot read SQLite. Stub a
           BEATABLE null (Sharpe 0.0, ann 0.0) so the existing tests keep failing
           for the reasons their names claim — an unbeatable null would make
           test_gate_requires_sharpe_and_dsr pass for the wrong reason, and its
           docstring says a False verdict 'can only come from Sharpe/DSR/max_dd'."""
           monkeypatch.setattr(walkforward, "buy_and_hold",
                               lambda conn, syms, **kw: _beatable_benchmark())
   ```
9. New tests in `TestWalkForwardPooled` (Phase 1's "extends" slot, contract §8): `test_gate_is_a_dict_keyed_by_gate_conditions` (`set(result.gate) == set(GATE_CONDITIONS)`, `len == 7`); `test_passed_is_still_a_bool_and_equals_all_of_gate` (`isinstance(result.passed, bool)` **and** `result.passed is all(result.gate.values())`); `test_beats_benchmark_return_fails_against_an_unbeatable_null` (stub `ann_return_pct=99.0` → condition `False`, `passed is False`); `test_beats_benchmark_sharpe_fails_against_an_unbeatable_null` (same for `sharpe=99.0`); `test_missing_benchmark_fails_the_gate` (all-`None` null → both benchmark conditions `False`); `test_n_trials_used_is_reported` (`== len(combos) * len(folds)` ledgerless); `test_empty_symbol_dict_fails_per_symbol_condition` (pins Task 6 GOTCHA #5).

- **MIRROR**: `TEST_STUB`, `TEST_CONVENTIONS`.
- **IMPORTS** (`tests/test_backtest.py`): add `from trading_bot.backtest.benchmark import BenchmarkResult`.
- **GOTCHA**: Do **not** "fix" items 5–6 by making `daily_returns` tolerant of `entry_ts > exit_ts`. Production code warns and falls back for that impossible case (Task 4), but the *tests* were wrong — a fake producing trades the metrics layer discards is a fake that tests nothing.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_backtest.py tests/test_equity.py -v`, then `.venv/bin/python -m pytest -q`.

### Task 12: Run the reproduction and record the measured numbers
- **ACTION**: Execute both commands, capture output verbatim, write the phase report. **No number may be derived or estimated** (contract §12.5).
- **IMPLEMENT** — the report must contain, each as literal command output: (1) `benchmark --start 2023-07-27 --end 2026-07-26` beside §0's published figures with the per-cell delta; (2) `walkforward` — the 7-condition table, the OOS-span benchmark row, `n_trials charged to DSR`, the `GATE:` line; (3) the MEDIUM-5 before/after moment table (from Task 8's `-v` output). Plus: **(4) Degrees of freedom consumed: ZERO** — no threshold, grid axis, window, or strategy parameter changed; `WF_MIN_TRADES` still 30, `WF_OOS_DAYS` still 90, `DEFAULT_GRID` untouched. State it explicitly (KNOWN-LIMITATIONS §9); it is the whole reason this phase can be believed. **(5)** Three sentences for Phase 9's handoff: the v0.2.0 verdict is unchanged (FAIL on `sample_adequacy` + `dsr`, 5 of 7 pass); both benchmark conditions PASS **because the holdout was a bear window**, so the null is necessary but not sufficient and Phase 9 must also score the full-span null; the attribution fix **raised** OOS Sharpe 1.175 → 1.512 and did not rescue DSR.
- **MIRROR**: commit 86777ef, "Use measured rather than derived figures in the benchmark table" — that commit exists because a published figure had been inferred rather than measured. Do not repeat it.
- **GOTCHA**: `walkforward` re-runs the one-shot OOS holdout. That holdout is already spent and its verdict published, so re-running it to **reproduce** a published result consumes nothing new. Running it after changing a parameter would be different — which this phase does not do.
- **VALIDATE**: every number in the report traceable to a pasted command.

---

## Testing Strategy

Per-test expectations are enumerated in Tasks 8–11. The load-bearing ones: **Σ daily returns invariant across attribution modes**; **exit-day mode reproduces §3's skew 3.7883 / kurtosis 31.2449 on the real 23 trades**; **spread mode measures 1.4034 / 15.5429 with `kurt < 30` asserted**; **the §0 per-symbol and basket tables reproduce within one rounding unit**; **buy-once-hold does NOT reproduce them**; **`passed` stays a `bool` equal to `all(gate.values())`**; **each new benchmark condition fails against an unbeatable null and against an uncomputable one**.

### Edge Cases Checklist
- [x] Empty input — no trades, no symbols, no bars; single observation → `None`, never a fabricated `0.0`
- [x] Trade straddling either span boundary, and wholly outside
- [x] `exit_ts < entry_ts` (impossible-but-defended: warn, book on exit day)
- [x] Wipe-out compounding (mirrors `equity.py:188-210`)
- [x] Zero variance → Sharpe `None`; zero downside → Sortino `None`
- [x] Ragged / missing symbol series in the basket
- [x] Unknown `attribution` / `BENCHMARK_REBALANCE` → `ValueError`
- [x] Benchmark uncomputable → gate fails; does not pass and does not raise
- [x] Ledger persistence across process restart
- [x] `conn=None` in existing walk-forward tests (stubbed seam, not a `None` branch)
- [ ] Concurrent access — `_db_lock` is in place for later phases; not exercised here (sequential replays)
- [ ] Network failure / permission denied — N/A; this phase makes no network call

---

## Validation Commands

**The repo has no linter and no type checker** (KNOWN-LIMITATIONS §8, contract §0). Validation is `pytest` and `py_compile`. Do not invent mypy/ruff/flake8 commands.

```bash
# 1. Static analysis — EXPECT clean, no output
.venv/bin/python -m py_compile \
  src/trading_bot/config.py src/trading_bot/cli.py src/trading_bot/data/statestore.py \
  src/trading_bot/backtest/benchmark.py src/trading_bot/backtest/trials.py \
  src/trading_bot/backtest/equity.py src/trading_bot/backtest/walkforward.py

# 2. Baseline, BEFORE touching anything — EXPECT "286 tests collected"
.venv/bin/python -m pytest --collect-only -q | tail -1

# 3. New test files — EXPECT all pass, 0 SKIPPED. A skipif firing on
#    data/ohlcv.db means the acceptance anchor did not run and this phase is NOT done.
.venv/bin/python -m pytest tests/test_benchmark.py tests/test_trials.py tests/test_pnl_attribution.py -v

# 4. Deliberately-edited files — EXPECT all pass;
#    test_zero_length_hold_books_on_its_single_day present, the old name gone
.venv/bin/python -m pytest tests/test_equity.py tests/test_backtest.py -v

# 5. Full suite — EXPECT the 286 pre-existing tests still green plus the new ones;
#    the only permitted skip is the pre-existing @pytest.mark.network one
#    (KNOWN-LIMITATIONS §8: "285 passed, 1 skipped")
.venv/bin/python -m pytest -q

# 6. Grep verification
grep -rn "config\." src/trading_bot/backtest/equity.py     # EXPECT: zero matches
grep -rn "hash(" src/trading_bot/backtest/trials.py        # EXPECT: no bare hash(
grep -rn "GATE_CONDITIONS" src/trading_bot/                # EXPECT: walkforward.py + cli.py
grep -rn "attribution" src/trading_bot/backtest/equity.py  # EXPECT: present

# 7. Database — EXPECT trial_ledger + its index; data/state.db (+ sidecars) on disk
#    and ABSENT from git status; data/ohlcv.db size unchanged (117 MB)
.venv/bin/python -c "
from trading_bot.data import statestore
from trading_bot.backtest import trials
c = statestore.connect(); trials.ensure_schema(c)
print(c.execute('SELECT name FROM sqlite_master').fetchall())"
ls -la data/ && git status --short data/
```

### Manual validation — the success signal, one command each
```bash
.venv/bin/python -m trading_bot.cli benchmark --start 2023-07-27 --end 2026-07-26
```
- [ ] BTC row ≈ `2.2104x  +30.26%  0.799  1.208  52.98%  1095`; BASKET ≈ `2.1628x  +29.32%  0.728  1.086  64.32%  1095`
- [ ] Every cell within one rounding unit of KNOWN-LIMITATIONS §0; exit 0; no extrapolation NOTE (span ≥ 365 days)

```bash
.venv/bin/python -m trading_bot.cli walkforward
```
- [ ] 7 named conditions, in `GATE_CONDITIONS` order
- [ ] `sample_adequacy FAIL` (23 < 30) and `dsr FAIL` — **§1's verdict, from one command**
- [ ] `beats_benchmark_return PASS` and `beats_benchmark_sharpe PASS` — expected; the holdout was a bear window (basket Sharpe ≈ −1.303). A measured finding, not a bug
- [ ] `sharpe PASS` at ≈ 1.512 (up from 1.175 — the attribution fix; record it)
- [ ] `n_trials charged to DSR: 36` on the ledgerless path
- [ ] Final line exactly `GATE: FAIL`; exit code 1

---

## Acceptance Criteria
- [ ] All 12 tasks complete; every validation command passes; **286 pre-existing tests green**
- [ ] `buy_and_hold(conn, symbols, *, start_ms, end_ms) -> BenchmarkResult` matches contract §4 field-for-field
- [ ] Basket is **equal-weight, daily-rebalanced**, with a committed test pinning that buy-once-hold does **not** reproduce §0
- [ ] The §0 table reproduces from ONE command, network-free, from stored 1d bars, within the stated per-metric tolerances
- [ ] `GATE_CONDITIONS` has exactly 7 entries; `_evaluate_gate -> dict[str, bool]` keyed by it
- [ ] `WalkForwardResult.passed` is still a `bool` and `== all(gate.values())`; the result gains `gate`, `benchmark`, `n_trials_used` with no field reordered or renamed
- [ ] `daily_returns(trades, start_ms, end_ms)` still works positionally; `attribution="exit_day"` reproduces v0.2.0
- [ ] Measured kurtosis delta recorded (31.2449 → 15.5429, `< 30` asserted); `Σ daily_returns` invariance pinned by a test
- [ ] `data/state.db` created; `trial_ledger` DDL owned by `trials.py`; git-ignored including sidecars
- [ ] `cli.py benchmark` → `_benchmark_command`, exit 0/1
- [ ] Phase report contains only measured numbers and states DoF consumed = **zero**

## Completion Checklist
- [ ] Frozen dataclasses; `| None = None` config-fallback kwargs; `_db_lock`-guarded writes
- [ ] Undefined metrics return `None`, never `0.0`; unknown modes raise `ValueError`
- [ ] `logging.getLogger("trading_bot")`; WARNING for dropped/undefined data; no new print in library code
- [ ] Tests: class-grouped, tier-derived constants, hand-computed literals, finding ids in names
- [ ] No hardcoded threshold outside `GATE_MIN_*` / `GATE_CONDITIONS` / the new config block; `equity.py` still has **zero** `config.` references
- [ ] No file owned by another phase created or edited (contract §2); only Phase 1's reserved prefixes, CLI name and test filenames used
- [ ] Self-contained — no codebase search needed during implementation

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Existing walk-forward tests pass `conn=None`; a real `buy_and_hold` call raises | **Certain** | High (7 tests) | Module-level `buy_and_hold` seam + autouse stub fixture (Task 11 item 8). Deliberately no `conn is None` branch in production |
| `per_symbol_fake_run_factory`'s stale `entry_ts` silently drops every fake trade | **Certain** | **High — a green suite testing nothing** | Diagnosed and fixed in Task 11 item 5, with the arithmetic (`19675` vs a 0–25-day span) written out |
| The attribution fix raises Sharpe 1.175 → 1.512, easing a gate condition | Certain | Medium | Measured, pinned by a test, required in the phase report. Not buried |
| Both new benchmark conditions PASS on v0.2.0, so the null looks toothless | Certain | Medium | Measured cause (bear holdout, basket Sharpe −1.303), pinned by a test, handed to Phase 9 as "also score the full-span null" |
| §0's figures were measured ad hoc (commit 86777ef); the new module might not reproduce them | Low — **already reproduced** in Ground Truth | High if it happened | Every cell verified before this plan was written; the basket construction was chosen *by* that measurement; tolerances derived from the fee sensitivity |
| The `skipif` on `data/ohlcv.db` silently skips the acceptance anchor | Medium | **High — the phase's whole point** | `-v` + "0 skipped" required in Validation; a skip is explicitly not a pass |
| Phase 3 later needs a different `ledger=` shape on `walk_forward_pooled` | Medium | Low | `ledger=None` is additive and independent of Phase 3's `strategy=`; both keyword-only, order-free |
| DSR becomes unpassable once the ledger counts every generation's variants | High (by design) | Medium | Contract §4: "if it makes the gate unpassable, that is the finding." Ledger is opt-in in Phase 1, so v0.2.0's reproduction is unaffected |

## Notes

1. ⚠ **The brief's "§0 is a 90-day window" premise is wrong; the contract is right.** §0's span is 1095 days = exactly 3 years, and `2.2135^(1/3) − 1 = +30.32%` confirms its annualized column is a true CAGR. The 90-day extrapolation is §2's "+68% annualised" headline and §1's OOS drawdown. The plan reports **both** views: the `benchmark` CLI over a long span (honest, quotable CAGR) and the in-gate null over the holdout (identically annualized on both sides, so the comparison is valid but neither level is quotable), with an explicit NOTE below 365 days.

2. **`ledger=` on `walk_forward_pooled` belongs to Phase 1, by elimination.** Contract §7 says Phase 6 "does **not** edit this file; it wraps it," and §4 says the oracle increments the ledger. The increment must happen where evaluations happen — inside `_pooled_expectancy` — and only Phases 1 and 3 may edit `walkforward.py` (§7). So the seam lands here. `ledger=None` preserves today's behavior exactly and composes with Phase 3's `strategy=` without ordering constraints.

3. **`n_trials` is charged `ledger.count()`, not `distinct_count()`.** Contract §4 says "the ledger counts every (graph, params) evaluation the oracle ever performs" and "n_trials … is that cumulative count." The argument for `distinct_count()` — re-evaluating an identical configuration on an identical span is not a new degree of freedom — is real, and it is implemented and reported so the choice stays auditable. It is **not** the default, because that would contradict a binding document and because over-counting errs pessimistically, the correct direction here. Relatedly, the ledgered count **includes neighbour probes** that `walkforward.py:376-382` deliberately omits from the ledgerless default, so `ledger.count() > len(combos) * len(folds)` and the ledgered DSR is strictly worse. Both paths stay available and the printout says which was charged.

4. **`equity.py` stays config-free.** Its docstring promises "no config coupling" (`equity.py:12`), so the default is the module constant `ATTRIBUTION_SPREAD` and `config.PNL_ATTRIBUTION_MODE` is read by the already-config-coupled callers. Flipping the config still flips the gate; the pure module stays pure. A grep in Validation enforces it.

5. **Clipped-share attribution, and why.** An overlapping trade contributes its **whole** `pnl_pct` divided by the number of **in-span** days. The alternative (divide by full holding length, drop the out-of-span share) makes total reported P&L depend on where the window is cut. Conservation is the property worth having, and it yields the strongest available test — Σ measured identical to 17 significant figures on the real trades.

6. **Why no `conn is None` branch in `buy_and_hold`.** `None`-tolerance would produce an all-`None` benchmark that fails the gate — a *plausible-looking* fail for the wrong reason. Stubbing at the `walkforward` module level is the codebase's established seam (`tests/test_backtest.py:504` already stubs `run_backtest` there), keeps production code honest, and makes each test's intent explicit.

7. **§0b is context, not scope.** BTC/ETH/SOL daily-return correlations (+0.809 / +0.744 / +0.719; effective N ≈ 1.2) mean the basket null is close to one bet, so "beats the basket" is a weaker statement than three symbols suggest. Phase 2 owns `data/correlation.py`; this phase recomputes no correlation.

8. **What this phase does not fix, stated so it is not assumed.** Kurtosis lands at 15.54, not 3.0 — holds are 1–3 days and there is only so much smearing available. DSR still fails at 0.086 (0.783 even at `n_trials=1`); §1 established no `n_trials` rescues it. Sample adequacy is still 23 < 30, which is Phase 2's problem and not a measurement defect. The null is necessary, not sufficient: a short adverse holdout can hand a flat strategy a free pass on both new conditions.

9. **Degrees of freedom consumed: zero.** No threshold, window, grid axis, or strategy parameter changes here. That is what makes the reproduction of §0 and §1 evidence rather than a new result, and it is the property Phase 3 inherits when it starts optimizing against this gate.
