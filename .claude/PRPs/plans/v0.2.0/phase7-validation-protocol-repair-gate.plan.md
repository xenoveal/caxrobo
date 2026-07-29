# Plan: Validation Protocol Repair + THE GATE (PRD Phase 7)

## Summary
`walk_forward()` today sweeps the **healthiest** layer (the three regime-classifier thresholds), validates at `WF_MIN_TRADES=5`, runs **per-symbol only** (no pooling), and its `plateau_ratio` divides by a near-zero `best_exp` — numerically unstable and reporting fake catastrophic fragility. This phase rebuilds the harness to pool all 3 symbols into every fold, raise `min_trades` to 30, sweep the levers that actually bind (R:R floor / hold limit / Donchian lookback, **not** the frozen cost/`k` constants), replace `plateau_ratio` with a stable positive-neighbour-fraction metric, wire in Phase 3's Sharpe/DSR machinery, and then run the **one-shot OOS holdout exactly once** — the actual verdict the whole PRD pivots on.

## User Story
As the bot's sole user, I want a walk-forward harness whose PASS/FAIL verdict is trustworthy — pooled across symbols, validated at a real sample size, scored on Sharpe/DSR instead of raw expectancy — so that a single one-shot run tells me honestly whether the strategy has edge, with no second chances to quietly re-tune a failing result.

## Problem → Solution
**Current**: `walk_forward(conn, symbol, ...)` is single-symbol; grid = `{adx_trend_threshold, atr_extreme_percentile, bb_num_std}` (regime layer, measured healthy); `min_trades=5` (noise-fits from 18 combos); `plateau_ratio = mean(neighbours)/best` (blows up near zero); gate = `n_trades >= min_trades and expectancy_pct > 0` (no Sharpe, no cross-symbol requirement, no DSR).
**Solution**: `walk_forward_pooled(conn, symbols, ...)` runs every combo/fold across all symbols and concatenates trade lists before scoring — 3× the sample at zero extra degrees of freedom, which also gives the cross-symbol consistency gate "for free" (it's the same pooled trade list, sliced per symbol only for the consistency check). Grid becomes `{rr_floor, max_hold_bars}` with `k`/costs/Donchian-canonical params frozen out entirely (added as fixed kwargs, never swept — this is Phase 5's engine surface, this plan documents the fields needed but does not implement Donchian). `plateau_ratio` → `positive_neighbour_fraction` (fraction of neighbours with positive expectancy) + `neighbour_spread` (max−min expectancy among neighbours, an absolute-scale companion that never divides by a fitted value). Gate now checks, on the pooled one-shot OOS run: `n_trades >= 30`, `sharpe >= 1.0`, `dsr > 0.95`, **and** per-symbol positive expectancy (the mandatory 3-of-3 gate) computed from the same OOS trades sliced by symbol.

## Metadata
- **Complexity**: Large (4 files touched, ~450 net new/changed lines incl. tests; depends on Phase 3's `equity.py` and Phase 5's Donchian engine existing, though this plan's own tasks are engine-agnostic and testable with a stub)
- **Source PRD**: `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`
- **PRD Phase**: Phase 7 — Validation protocol repair + THE GATE
- **Estimated Files**: 4 (1 new, 3 modified)
- **Depends on**: Phase 3 (Sharpe/DSR primitives in `backtest/equity.py` — see `.claude/PRPs/plans/v0.2.0/phase3-sharpe-first-metrics.plan.md`), Phase 5 (Donchian engine — signal method is a parameter to this harness, not built here), Phase 6 (fade re-qualification result feeds the ranging-regime path this harness also backtests)
- **This is a hard barrier**: nothing in Phases 8-9 should start before this phase's gate returns a verdict.

---

## UX Design

N/A — internal change. Observable output: `python -m trading_bot.cli walkforward` prints pooled fold results, the new robustness metrics, Sharpe/DSR on the OOS run, and a GATE line evaluated against the full Success Metrics table instead of `n_trades/expectancy_pct` alone.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/trading_bot/backtest/walkforward.py` | 1-222 (whole file) | The module being rebuilt. Read the docstring (1-20) first — it states the *current* (soon-superseded) protocol; every function here (`_combos`, `_neighbors`, `_expectancy`, `walk_forward`) is either replaced or adapted |
| P0 | `src/trading_bot/backtest/equity.py` | all (created by Phase 3 — see that plan) | `compute_equity_metrics(trades, start_ms, end_ms, n_trials)` is the function this phase calls to get Sharpe/DSR on pooled trade lists; `deflated_sharpe(sr, n_trials, n_obs, skew, kurt, sr_var=None)` signature must match exactly |
| P0 | `src/trading_bot/backtest/metrics.py` | all | `compute_metrics(trades)` — still used for the per-symbol positive-expectancy consistency check and the `by_bucket` breakdown; untouched by this phase |
| P0 | `src/trading_bot/backtest/engine.py` | 91-124, 51-58 | `run_backtest(conn, symbol, *, start_ms, end_ms, params: BacktestParams, fee_pct, slippage_pct, max_hold_bars)` — signature is **per-symbol**; pooling happens by calling it once per symbol per combo/fold and concatenating, NOT by changing its signature. `BacktestParams` — the dataclass whose fields become the (eventually Donchian-aware) sweep grid |
| P1 | `src/trading_bot/config.py` | 93-100 | `WF_TRAIN_DAYS/WF_TEST_DAYS/WF_OOS_DAYS/WF_MIN_TRADES` — `WF_MIN_TRADES` changes 5→30 here; also where `RR_FLOOR`/`MAX_HOLD_BARS_15M` (or their Phase 4 renames) already live as the frozen defaults this grid sweeps around |
| P1 | `src/trading_bot/cli.py` | 388-422 | `_walkforward_command` — the printing/exit-code pattern; must be adapted to pooled multi-symbol output and the richer gate |
| P1 | `tests/test_backtest.py` | 150-235 | `fake_run_factory`, `TestWalkForward`, `TestBacktestCli` — the monkeypatch idiom (`monkeypatch.setattr(walkforward, "run_backtest", ...)`) this phase's tests must keep using, now stubbing a **per-symbol-aware** fake |
| P2 | `.claude/PRPs/plans/v0.2.0/phase3-sharpe-first-metrics.plan.md` | all | Sibling plan — read first if implementing both phases in sequence; this plan assumes `equity.py` already exists exactly as specified there |

## External Documentation

No external research needed — DSR/PSR formulas are already fully specified in Phase 3's plan and implemented in `equity.py`; this phase only calls them. Positive-neighbour-fraction is a straightforward robustness statistic, not a published-method lookup.

---

## Patterns to Mirror

### DATACLASS_RESULT_SHAPE
```python
# SOURCE: walkforward.py:43-66 — frozen dataclasses for fold/aggregate results;
# extend, don't replace, the field set.
@dataclass(frozen=True)
class FoldResult:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: BacktestParams
    train_expectancy: float | None
    plateau_ratio: float | None      # RENAMED/REPLACED — see Task 3
    test_metrics: dict
```

### GRID_SWEEP_PATTERN
```python
# SOURCE: walkforward.py:69-90 — _combos() (cartesian product of a dict of
# tuples) and _neighbors() (one-step-per-axis) are pure and reusable AS-IS;
# only DEFAULT_GRID's keys change (Task 1), not this machinery.
def _combos(grid: dict[str, tuple]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
```

### OPTIONAL_OVERRIDE_KWARGS
```python
# SOURCE: walkforward.py:99-104, config.py:93-100 — every protocol knob is a
# keyword-only `| None = None` falling back to a config default.
def walk_forward(conn, symbol, *, start_ms, end_ms, grid=None,
                  train_days=None, test_days=None, oos_days=None,
                  min_trades=None) -> WalkForwardResult:
    if min_trades is None:
        min_trades = config.WF_MIN_TRADES
```

### CLI_EXIT_CODE_PATTERN
```python
# SOURCE: cli.py:388-422 — 0 iff every symbol/fold-set passes; ValueError
# (span too short) caught per-symbol and treated as a fail, not a crash.
def _walkforward_command(conn, symbols, *, start_ms, end_ms) -> int:
    all_passed = True
    for symbol in symbols:
        try:
            result = walk_forward(conn, symbol, start_ms=start_ms, end_ms=end_ms)
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            all_passed = False
            continue
        ...
```
This phase's pooled command takes `symbols` as one call (not a per-symbol loop) — see Task 5.

### TEST_STUB_PATTERN
```python
# SOURCE: tests/test_backtest.py:150-158 — stub run_backtest at the module
# level walkforward imports it from, so walk_forward's internal calls are
# intercepted without touching engine.py.
def fake_run_factory(peak_exp=0.01, n_trades=6):
    def fake_run(conn, symbol, *, start_ms=None, end_ms=None, params=None, **kw):
        params = params or BacktestParams()
        exp = peak_exp * (1 - abs(params.adx_trend_threshold - 25.0) / 25.0)
        return [make_trade(exp) for _ in range(n_trades)]
    return fake_run
monkeypatch.setattr(walkforward, "run_backtest", fake_run_factory())
```
Pooled tests need a **per-symbol-varying** fake (Task 6) so the cross-symbol consistency gate has something to actually fail on.

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/config.py` | UPDATE | `WF_MIN_TRADES` 5→30; add `WF_RR_FLOOR_GRID`/`WF_HOLD_GRID` tuples (or inline in walkforward.py's `DEFAULT_GRID` — see Task 1 for the tradeoff) |
| `src/trading_bot/backtest/walkforward.py` | REWRITE | New `DEFAULT_GRID`; `walk_forward_pooled()` replacing per-symbol `walk_forward()`; `_positive_neighbour_stats()` replacing the `plateau_ratio` computation; pooled OOS run wired through `equity.compute_equity_metrics`; richer gate logic |
| `src/trading_bot/cli.py` | UPDATE | `_walkforward_command` calls `walk_forward_pooled(conn, symbols, ...)` once instead of looping `walk_forward(conn, symbol, ...)` per symbol; prints per-symbol consistency + pooled Sharpe/DSR |
| `tests/test_backtest.py` | UPDATE | `TestWalkForward` rewritten for pooling; new per-symbol-varying fake; CLI tests updated for the new single-call shape |

## NOT Building

- **Donchian engine itself** (Phase 5) — this harness's grid references `BacktestParams` fields; if Phase 5 adds `donchian_entry_lookback`/`donchian_exit_lookback` as genuinely uncertain (not canonical-frozen) fields, they join the grid then. This plan's grid is `{rr_floor, max_hold_bars}` using fields that already exist or are named in the PRD (Phase 2's `RR_FLOOR`, existing `MAX_HOLD_BARS_15M`).
- **Sweeping `ATR_STOP_MULTIPLE`, `k`, or any cost constant** — explicitly frozen per the PRD Decisions Log; passed as fixed kwargs to every `run_backtest` call, never in `grid`.
- **CPCV (Combinatorial Purged Cross-Validation)** — noted in the PRD as a stronger future option; out of scope for v1.
- **Changing `run_backtest`'s signature** — pooling is achieved entirely in `walkforward.py` by calling it N times (once per symbol) and concatenating; `engine.py` is untouched.
- **A written trial log** — that's Phase 9 (per PRD phase table); this phase's job is to make ONE trustworthy one-shot run possible, not to log every historical sweep.
- **Re-running the gate on failure** — a FAIL is a terminal verdict for v1 per the PRD's pre-committed escalation (Option B), not a signal to loosen the grid and retry.

---

## Step-by-Step Tasks

### Task 1: Fix the grid — sweep what binds, freeze what's canonical
- **ACTION**: Edit `config.py` (add `WF_MIN_TRADES = 30`) and `walkforward.py`'s `DEFAULT_GRID`.
- **IMPLEMENT**:
  ```python
  # config.py — change in place:
  WF_MIN_TRADES = 30  # was 5; selecting from 18 combos at n=5 is noise-fitting

  # walkforward.py — replace DEFAULT_GRID:
  # Grid contents changed per PRD Phase 7: sweep the levers that actually bind
  # (R:R floor, hold-time limit), NOT the regime-classifier thresholds (the
  # one measured-healthy layer, kept fixed at config defaults) and NOT the
  # cost model or ATR stop multiple k (frozen per the PRD Decisions Log — see
  # config.ATR_STOP_MULTIPLE / config.RR_FLOOR docstrings from Phase 2).
  DEFAULT_GRID: dict[str, tuple] = {
      "rr_floor": (1.25, 1.5, 1.75, 2.0),
      "max_hold_bars": (48, 96, 144),  # in trigger-timeframe bars; Phase 4 rescales
  }
  ```
  `BacktestParams` (in `engine.py`) must expose `rr_floor` and `max_hold_bars` as sweepable fields once Phase 2/4 land — if `engine.py`'s `run_backtest` still takes `max_hold_bars` as a top-level kwarg (not inside `BacktestParams`) at the time this phase is implemented, keep `_combos()`/`_expectancy()` unchanged in shape but split the combo dict at the call site: `BacktestParams` fields go to `params=`, `max_hold_bars` goes to its own kwarg. Check `engine.py`'s actual signature before writing this — it may have moved between phases.
- **MIRROR**: The grid-as-dict-of-tuples shape (`walkforward.py:36-40`), "small, coarse grid by design" comment.
- **IMPORTS**: None new.
- **GOTCHA**: `_neighbors()` (walkforward.py:74-82) is generic over any grid dict — it needs NO code change, only benefits from the new keys. Don't touch it.
- **VALIDATE**: `python -c "from trading_bot.backtest.walkforward import DEFAULT_GRID; print(DEFAULT_GRID)"` shows the new keys; `grep -n "adx_trend_threshold" src/trading_bot/backtest/walkforward.py` returns nothing (fully removed from the grid — regime params are no longer swept here).

### Task 2: Pool symbols per fold
- **ACTION**: Rewrite `walk_forward()` → `walk_forward_pooled(conn, symbols: list[str], *, start_ms, end_ms, grid=None, train_days=None, test_days=None, oos_days=None, min_trades=None, n_trials=None)`.
- **IMPLEMENT**:
  ```python
  def _pooled_expectancy(conn, symbols: list[str], combo: dict, start: int, end: int) -> tuple[float | None, int, list]:
      """Concatenate trades across all symbols for one combo/fold; return
      (pooled expectancy, pooled n_trades, pooled trade list)."""
      all_trades = []
      for symbol in symbols:
          all_trades.extend(
              run_backtest(conn, symbol, start_ms=start, end_ms=end, params=BacktestParams(**combo))
          )
      m = compute_metrics(all_trades)
      return m["expectancy_pct"], m["n_trades"], all_trades

  def walk_forward_pooled(
      conn, symbols: list[str], *, start_ms: int, end_ms: int,
      grid: dict[str, tuple] | None = None,
      train_days: int | None = None, test_days: int | None = None,
      oos_days: int | None = None, min_trades: int | None = None,
      n_trials: int | None = None,
  ) -> WalkForwardResult:
      """
      Pooled 3-symbol walk-forward: identical fold/train/test/OOS mechanics to
      the original walk_forward, except every combo is scored on the
      CONCATENATED trade list across all `symbols` rather than one symbol at
      a time. This triples the sample per fold at zero additional degrees of
      freedom (Decisions Log: "the only free lunch on the list") and makes
      cross-symbol consistency checkable on the exact trades the gate scores.
      """
      if grid is None:
          grid = DEFAULT_GRID
      train_ms = (config.WF_TRAIN_DAYS if train_days is None else train_days) * DAY_MS
      test_ms = (config.WF_TEST_DAYS if test_days is None else test_days) * DAY_MS
      oos_ms = (config.WF_OOS_DAYS if oos_days is None else oos_days) * DAY_MS
      if min_trades is None:
          min_trades = config.WF_MIN_TRADES

      tune_end = end_ms - oos_ms
      if start_ms + train_ms + test_ms > tune_end:
          raise ValueError("span too short: need at least one train+test fold before the OOS holdout")

      combos = _combos(grid)

      folds: list[FoldResult] = []
      t0 = start_ms
      while t0 + train_ms + test_ms <= tune_end:
          train_start, train_end = t0, t0 + train_ms
          test_start, test_end = train_end, train_end + test_ms

          scored = []
          for combo in combos:
              exp, n, _ = _pooled_expectancy(conn, symbols, combo, train_start, train_end)
              if exp is not None and n >= min_trades:
                  scored.append((exp, combo))

          if scored:
              best_exp, best_combo = max(scored, key=lambda x: x[0])
          else:
              best_exp, best_combo = None, _default_combo(grid)
              logger.info("fold %d: no combo reached min_trades; using defaults", len(folds))

          pos_frac, spread = _positive_neighbour_stats(conn, symbols, grid, best_combo, best_exp, train_start, train_end)

          _, _, test_trades = _pooled_expectancy(conn, symbols, best_combo, test_start, test_end)
          folds.append(
              FoldResult(
                  train_start=train_start, train_end=train_end,
                  test_start=test_start, test_end=test_end,
                  best_params=BacktestParams(**best_combo),
                  train_expectancy=best_exp,
                  positive_neighbour_fraction=pos_frac,
                  neighbour_spread=spread,
                  test_metrics=compute_metrics(test_trades),
              )
          )
          t0 += test_ms

      final_combo = {
          axis: statistics.median(getattr(f.best_params, axis) for f in folds)
          for axis in grid
      }
      final_params = BacktestParams(**final_combo)

      # n_trials for DSR: every combo evaluated per fold, plus neighbour probes
      # (4 or fewer neighbours per axis-adjacent step, per fold). Resolved HERE,
      # after the fold loop, because it needs len(folds).
      n_trials_used = (
          n_trials if n_trials is not None
          else len(combos) * max(1, len(folds))
      )

      oos_start, oos_end = tune_end, end_ms
      oos_trades = []
      per_symbol_oos: dict[str, list] = {}
      for symbol in symbols:
          t = run_backtest(conn, symbol, start_ms=oos_start, end_ms=oos_end, params=final_params)
          per_symbol_oos[symbol] = t
          oos_trades.extend(t)

      oos_metrics = compute_metrics(oos_trades)
      oos_equity = compute_equity_metrics(oos_trades, oos_start, oos_end, n_trials=n_trials_used)
      per_symbol_expectancy = {s: compute_metrics(t)["expectancy_pct"] for s, t in per_symbol_oos.items()}

      passed = _evaluate_gate(oos_metrics, oos_equity, per_symbol_expectancy, min_trades)

      return WalkForwardResult(
          folds=folds, final_params=final_params,
          oos_start=oos_start, oos_end=oos_end,
          oos_metrics=oos_metrics, oos_equity=oos_equity,
          per_symbol_expectancy=per_symbol_expectancy,
          passed=passed,
      )
  ```
  Add `from trading_bot.backtest.equity import compute_equity_metrics` to imports.
- **MIRROR**: The existing fold-loop control flow (`walkforward.py:145-194`) — same `while t0 + train_ms + test_ms <= tune_end` structure, same median-of-fold-winners final-params rule.
- **IMPORTS**: `from trading_bot.backtest.equity import compute_equity_metrics`.
- **GOTCHA #1**: `_default_combo(grid)` must be written fresh (the old code's `default_combo` referenced `config.ADX_TREND_THRESHOLD` etc., which are no longer grid axes) — build it from `config.RR_FLOOR` and `config.MAX_HOLD_BARS_15M` (or Phase 4's renamed equivalent) instead: `{"rr_floor": config.RR_FLOOR, "max_hold_bars": config.MAX_HOLD_BARS_15M}`.
- **GOTCHA #2**: `_pooled_expectancy` calls `run_backtest` once per symbol per combo per fold — with a 4×3=12-combo grid, ~5 folds, 3 symbols, that's ~180 backtest runs for the tuning sweep alone (before neighbour probing). This is deliberately more expensive than the old per-symbol version; note it in Testing Strategy but do not optimize prematurely — correctness first.
- **VALIDATE**: `pytest tests/test_backtest.py::TestWalkForwardPooled -v` (Task 6).

### Task 3: Replace `plateau_ratio` with a stable robustness pair
- **ACTION**: New helper `_positive_neighbour_stats`; update `FoldResult` dataclass.
- **IMPLEMENT**:
  ```python
  @dataclass(frozen=True)
  class FoldResult:
      train_start: int
      train_end: int
      test_start: int
      test_end: int
      best_params: BacktestParams
      train_expectancy: float | None
      positive_neighbour_fraction: float | None  # REPLACES plateau_ratio
      neighbour_spread: float | None              # NEW: max-min neighbour expectancy
      test_metrics: dict


  def _positive_neighbour_stats(
      conn, symbols: list[str], grid: dict, best_combo: dict,
      best_exp: float | None, start: int, end: int,
  ) -> tuple[float | None, float | None]:
      """
      Robustness check around the winning combo, computed on TRAIN data.

      Replaces the old `mean(neighbours) / best` plateau_ratio, which divides
      by a value that can be arbitrarily close to zero and reports numerically
      meaningless "fragility" (e.g. -11.3) for ordinary noisy neighbourhoods.

      Returns:
          (positive_neighbour_fraction, neighbour_spread) — the fraction of
          one-step neighbours with positive expectancy (near 1.0 = broad
          plateau of profitability; near 0 = an isolated spike), and the
          absolute spread (max - min) among neighbour expectancies (an
          absolute-scale companion that never divides by a fitted value).
          (None, None) if best_exp is None (no combo reached min_trades) or
          there are no neighbours to probe.
      """
      if best_exp is None:
          return None, None
      neigh_exps = []
      for nb in _neighbors(grid, best_combo):
          exp, _, _ = _pooled_expectancy(conn, symbols, nb, start, end)
          if exp is not None:
              neigh_exps.append(exp)
      if not neigh_exps:
          return None, None
      pos_frac = sum(1 for e in neigh_exps if e > 0) / len(neigh_exps)
      spread = max(neigh_exps) - min(neigh_exps)
      return pos_frac, spread
  ```
- **MIRROR**: `_neighbors()` is reused unchanged (walkforward.py:74-82).
- **GOTCHA**: The old code only computed `plateau` when `best_exp is not None and best_exp > 0` (walkforward.py:166) — i.e. it never flagged a *negative*-expectancy winner's neighbourhood as fragile-or-not. The new function computes stats whenever `best_exp is not None`, positive or negative, since "is this a fluke or a broad negative region" is equally informative. Document this behavior change in the docstring (done above) — a reviewer diffing against the old logic should not read it as a bug.
- **VALIDATE**: `pytest tests/test_backtest.py::TestWalkForwardPooled::test_positive_neighbour_fraction_replaces_plateau_ratio -v` (Task 6).

### Task 4: Wire the richer gate (Sharpe, DSR, cross-symbol, sample size)
- **ACTION**: `WalkForwardResult` gains `oos_equity` and `per_symbol_expectancy`; new `_evaluate_gate`.
- **IMPLEMENT**:
  ```python
  @dataclass(frozen=True)
  class WalkForwardResult:
      folds: list[FoldResult]
      final_params: BacktestParams
      oos_start: int
      oos_end: int
      oos_metrics: dict          # from compute_metrics — n_trades, expectancy, etc.
      oos_equity: dict           # from compute_equity_metrics — sharpe, sortino, dsr, max_drawdown_pct
      per_symbol_expectancy: dict[str, float | None]  # symbol -> OOS expectancy_pct
      passed: bool


  # Gate thresholds — mirrors the PRD Success Metrics table exactly. Not
  # sweepable; changing these is a decision about the north star, not a
  # tuning knob, and must not live inside the grid.
  GATE_MIN_SHARPE = 1.0
  GATE_MIN_DSR = 0.95
  GATE_MAX_DRAWDOWN = 0.25


  def _evaluate_gate(
      oos_metrics: dict, oos_equity: dict,
      per_symbol_expectancy: dict[str, float | None], min_trades: int,
  ) -> bool:
      """
      THE GATE. All conditions are mandatory (AND, not average):
        1. n_trades >= min_trades (sample adequacy)
        2. sharpe is not None and sharpe >= GATE_MIN_SHARPE
        3. dsr is not None and dsr > GATE_MIN_DSR (significant at p < 0.05)
        4. max_drawdown_pct is not None and max_drawdown_pct <= GATE_MAX_DRAWDOWN
        5. EVERY symbol's OOS expectancy_pct is not None and > 0
           (mandatory 3-of-3 cross-symbol gate, not an average)
      """
      if oos_metrics["n_trades"] < min_trades:
          return False
      if oos_equity["sharpe"] is None or oos_equity["sharpe"] < GATE_MIN_SHARPE:
          return False
      if oos_equity["dsr"] is None or oos_equity["dsr"] <= GATE_MIN_DSR:
          return False
      if oos_equity["max_drawdown_pct"] is None or oos_equity["max_drawdown_pct"] > GATE_MAX_DRAWDOWN:
          return False
      for exp in per_symbol_expectancy.values():
          if exp is None or exp <= 0:
              return False
      return True
  ```
- **MIRROR**: The old inline boolean `passed = (...)` (walkforward.py:208-212) — same "all mandatory conditions" shape, now factored into a named function since there are 5 conditions instead of 2.
- **IMPORTS**: None new beyond Task 2's `compute_equity_metrics` import.
- **GOTCHA**: `n_trials_used` (computed in Task 2, at the end of `walk_forward_pooled`, right before the OOS block) must count the **total configurations evaluated in the tuning search**, not just `len(combos)` — the PRD's DSR correction is meant to cover "the full grid × fold search count" (Success Metrics table). `len(combos) * len(folds)` is the default; it deliberately does NOT include neighbour probes from Task 3 (those examine robustness, not selection, and adding them would double-count in a way that's hard to justify precisely) — document this as a conservative-but-approximate choice, and leave `n_trials` as an explicit override for a future, more careful count.
- **VALIDATE**: `pytest tests/test_backtest.py::TestWalkForwardPooled::test_cross_symbol_gate_fails_on_one_bad_symbol -v`.

### Task 5: CLI — pooled invocation and richer printout
- **ACTION**: Rewrite `_walkforward_command` in `cli.py`.
- **IMPLEMENT**:
  ```python
  def _walkforward_command(conn, symbols, *, start_ms: int, end_ms: int) -> int:
      """
      Run the POOLED walk-forward protocol across all symbols and print fold,
      per-symbol OOS, and pooled Sharpe/DSR results.

      Returns:
          0 if the pooled gate passes, 1 otherwise (including a span too
          short to form a single fold).
      """
      try:
          result = walk_forward_pooled(conn, symbols, start_ms=start_ms, end_ms=end_ms)
      except ValueError as exc:
          print(f"ERROR: {exc}")
          return 1

      for i, fold in enumerate(result.folds):
          tm = fold.test_metrics
          print(
              f"fold {i}: params={fold.best_params}  "
              f"train_exp={_fmt(fold.train_expectancy, '.4%')}  "
              f"pos_neighbours={_fmt(fold.positive_neighbour_fraction, '.2f')}  "
              f"neighbour_spread={_fmt(fold.neighbour_spread, '.4%')}  "
              f"test_exp={_fmt(tm['expectancy_pct'], '.4%')}  "
              f"test_trades={tm['n_trades']}"
          )
      print(f"final params: {result.final_params}")
      print("one-shot OOS (pooled):")
      _print_metrics(result.oos_metrics, indent="  ")
      em = result.oos_equity
      print(
          f"  sharpe={_fmt(em['sharpe'], '.2f')}  sortino={_fmt(em['sortino'], '.2f')}  "
          f"dsr={_fmt(em['dsr'], '.4f')}  max_dd={_fmt(em['max_drawdown_pct'], '.2%')}  "
          f"ann_return={_fmt(em['ann_return_pct'], '.2%')}"
      )
      print("per-symbol OOS expectancy:")
      for symbol, exp in result.per_symbol_expectancy.items():
          flag = "OK" if (exp is not None and exp > 0) else "FAIL"
          print(f"  {symbol}: {_fmt(exp, '.4%')}  [{flag}]")
      print(f"GATE: {'PASS' if result.passed else 'FAIL'}")

      return 0 if result.passed else 1
  ```
  Update the import line: `from trading_bot.backtest.walkforward import walk_forward_pooled` (replacing `walk_forward`). Check `main()`'s existing call site to `_walkforward_command` — it should already pass `symbols` as the full list (e.g. `config.SYMBOLS`); only the function body's internal looping structure changes, not necessarily the call site's arguments.
- **MIRROR**: `_fmt`/`_print_metrics` (cli.py:355-367) reused as-is.
- **GOTCHA**: The old per-symbol loop printed `{symbol}:` as a section header per symbol (cli.py:398); the pooled version has ONE fold sequence (not one per symbol), so that header structure is gone — don't try to preserve it, the whole point is folds are pooled now.
- **VALIDATE**: `pytest tests/test_backtest.py::TestBacktestCli::test_walkforward_command_exit_codes -v` (rewritten in Task 6) + manual run in Validation Commands.

### Task 6: Rewrite walk-forward tests for pooling
- **ACTION**: Rewrite `TestWalkForward` (renamed `TestWalkForwardPooled`) and the walk-forward-related `TestBacktestCli` tests in `tests/test_backtest.py`.
- **IMPLEMENT**:
  ```python
  def per_symbol_fake_run_factory(peak_exp=0.01, n_trades=8, bad_symbol=None):
      """run_backtest stub whose expectancy peaks at rr_floor=1.5, and where
      `bad_symbol` (if given) always returns negative expectancy — lets tests
      exercise the mandatory per-symbol gate."""
      def fake_run(conn, symbol, *, start_ms=None, end_ms=None, params=None, **kw):
          params = params or BacktestParams()
          base = -0.01 if symbol == bad_symbol else peak_exp
          exp = base * (1 - abs(params.rr_floor - 1.5) / 1.5) if base > 0 else base
          return [make_trade(exp) for _ in range(n_trades)]
      return fake_run


  class TestWalkForwardPooled:
      SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
      SPAN = dict(start_ms=0, end_ms=25 * DAY_MS)
      KNOBS = dict(train_days=10, test_days=5, oos_days=5, min_trades=6)  # 6 = 2 trades/symbol x 3

      def test_pools_across_symbols(self, monkeypatch):
          monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
          result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
          # 8 trades/symbol x 3 symbols = 24 per fold's test window.
          assert result.folds[0].test_metrics["n_trades"] == 24

      def test_cross_symbol_gate_fails_on_one_bad_symbol(self, monkeypatch):
          monkeypatch.setattr(
              walkforward, "run_backtest",
              per_symbol_fake_run_factory(bad_symbol="SOLUSDT"),
          )
          result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
          assert result.per_symbol_expectancy["SOLUSDT"] < 0
          assert result.passed is False  # even though BTC/ETH are positive

      def test_gate_requires_sharpe_and_dsr(self, monkeypatch):
          # A fake with positive-but-wildly-varying pnl per trade should push
          # sharpe/dsr below threshold even with positive expectancy on every
          # symbol -- assert result.passed is False in that construction.
          ...

      def test_positive_neighbour_fraction_replaces_plateau_ratio(self, monkeypatch):
          monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
          result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
          for fold in result.folds:
              assert fold.positive_neighbour_fraction is None or 0.0 <= fold.positive_neighbour_fraction <= 1.0
              assert not hasattr(fold, "plateau_ratio")

      def test_min_trades_30_default(self):
          assert config.WF_MIN_TRADES == 30

      def test_span_too_short_raises(self, monkeypatch):
          monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
          with pytest.raises(ValueError):
              walk_forward_pooled(None, self.SYMBOLS, start_ms=0, end_ms=10 * DAY_MS, **self.KNOBS)
  ```
  Update `TestBacktestCli`'s walk-forward tests: they currently construct a bare `WalkForwardResult(folds=[], final_params=..., oos_start=0, oos_end=1, oos_metrics=..., passed=passed)` (test_backtest.py:212-216) — add the two new required fields (`oos_equity`, `per_symbol_expectancy`) to that constructor call, and update `monkeypatch.setattr(cli, "walk_forward", ...)` to `monkeypatch.setattr(cli, "walk_forward_pooled", ...)`.
- **MIRROR**: `fake_run_factory` (test_backtest.py:150-158) — same closure-stub shape, extended with a `bad_symbol` parameter.
- **IMPORTS**: `from trading_bot.backtest.walkforward import walk_forward_pooled` in place of `walk_forward` throughout the test file.
- **GOTCHA**: `KNOBS["min_trades"]` in the new tests must be set so the fake's `n_trades` (per symbol) × 3 symbols clears it, or every fold falls back to defaults and the tests silently exercise the wrong branch — this bit the original per-symbol tests too (`test_too_few_trades_uses_defaults_and_fails_gate` relies on exactly this). Compute it deliberately, don't copy the old per-symbol `min_trades=5`.
- **VALIDATE**: `pytest tests/test_backtest.py -v` — full file, since Task 6 touches most of its walk-forward-adjacent classes.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| Pooling | 3 symbols, 8 trades each | fold test_metrics n_trades == 24 | — |
| Cross-symbol gate | one symbol negative, two positive | `passed is False` | Core Phase 7 behavior |
| Sharpe/DSR gate | positive expectancy, high variance | `passed is False` (Sharpe or DSR too low) | Core Phase 7 behavior |
| positive_neighbour_fraction range | any fold | `0.0 <= x <= 1.0` or `None` | Replaces unstable plateau_ratio |
| `WF_MIN_TRADES` | — | `== 30` | Config regression guard |
| Too-few-trades fallback | fake returns < min_trades | folds use `_default_combo`, gate fails on sample size | — |
| Span too short | 10-day span, 10-day train | `ValueError` | Unchanged from original |
| CLI exit codes | pass / fail stub results | `0` / `1` | — |

### Edge Cases Checklist
- [x] One symbol fails cross-symbol gate while others pass → whole gate fails
- [x] No combo reaches `min_trades` in a fold → falls back to `_default_combo`, `train_expectancy=None`, neighbour stats `(None, None)`
- [x] `best_exp` negative → neighbour stats still computed (documented behavior change from old code)
- [x] Span too short for one fold + OOS → `ValueError`, caught by CLI, exit code 1
- [x] DSR pathological moments (denominator ≤ 0) → `None`, gate fails safely (never crashes)
- [ ] Concurrent access — N/A (sequential backtest replays)
- [ ] Network failure — N/A

---

## Validation Commands

### Static Analysis
```bash
python -m py_compile src/trading_bot/backtest/walkforward.py src/trading_bot/cli.py src/trading_bot/config.py
```
EXPECT: clean. (No mypy/linter configured — skip.)

### Unit Tests
```bash
pytest tests/test_backtest.py -v
```
EXPECT: all pass, including the new `TestWalkForwardPooled` class; zero references to `plateau_ratio` remain (grep check below).

### Full Test Suite
```bash
pytest tests/ -v
```
EXPECT: no regressions in `test_classifier.py`, `test_wilder.py`, `test_storage.py`, `test_signals.py`, `test_meanrev.py`, `test_equity.py` (Phase 3) — none of these touch the walk-forward harness directly.

### Grep Verification
```bash
grep -rn "plateau_ratio\|def walk_forward\b" src/ tests/
```
EXPECT: no matches for `plateau_ratio`; `walk_forward_pooled` is the only walk-forward entry point (the bare `def walk_forward\b` pattern, without `_pooled`, should not appear).

### Manual Validation — THE GATE, run exactly once
```bash
python -m trading_bot.cli walkforward
```
- [ ] Confirm this is the untouched OOS holdout — check `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`'s Open Question on "how much of 2026 is reserved" before running; do not run this command speculatively or more than once against the same holdout span once Phases 1-6 are actually complete.
- [ ] Record the printed verdict (PASS/FAIL), Sharpe, DSR, max_dd, and per-symbol expectancy verbatim wherever the project's trial log lives (formal mechanism is Phase 9, but the PRD's own risk table says an untracked sweep silently spends the holdout — write it down even informally now).
- [ ] On FAIL: the pre-committed response is escalation to Option B (universe expansion), NOT re-tuning the grid and re-running. Do not loop back into Task 1 with a wider grid.

---

## Acceptance Criteria
- [ ] All 6 tasks completed; all validation commands pass
- [ ] `walk_forward_pooled` pools 3 symbols per fold and per OOS run
- [ ] `WF_MIN_TRADES == 30`
- [ ] Grid sweeps `rr_floor`/`max_hold_bars`, never regime thresholds or frozen cost/`k` constants
- [ ] `plateau_ratio` fully replaced by `positive_neighbour_fraction` + `neighbour_spread`
- [ ] Gate requires: sample size, Sharpe ≥ 1.0, DSR > 0.95, max DD ≤ 25%, AND all 3 symbols' OOS expectancy > 0
- [ ] The one-shot OOS run executes exactly once per invocation (no internal retry/re-tune loop)

## Completion Checklist
- [ ] Code follows discovered patterns (frozen dataclasses, optional-override kwargs, module-level `run_backtest` monkeypatch point preserved for tests)
- [ ] Error handling matches codebase style (`ValueError` on too-short span, propagated not swallowed)
- [ ] Logging follows conventions (`logging.getLogger("trading_bot")`, INFO for fallback-to-defaults)
- [ ] Tests follow test patterns (class-per-function, closure-based `run_backtest` stubs)
- [ ] No hardcoded gate thresholds outside the named `GATE_MIN_*` constants
- [ ] No unnecessary scope additions — no Donchian implementation, no trial log, no CPCV
- [ ] Self-contained — no questions needed during implementation (flagged ambiguity: `engine.py`'s exact `BacktestParams`/kwarg split for `max_hold_bars` at implementation time — see Task 1 GOTCHA)

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pooled sweep is ~9x more `run_backtest` calls than the old per-symbol version (3 symbols × combos × folds, plus neighbour probes) → the tuning phase becomes slow on 3+ years of 1H/4H data | High | Medium (dev friction, not correctness) | Accept for v1 — correctness over speed; if intolerable, cache `run_backtest` results per (symbol, combo, span) tuple, but that is an optimization out of this plan's scope |
| `n_trials` for DSR undercounts the true search (doesn't include Phase 5/6's earlier tuning, if any leaked into the frozen params) | Medium | Medium (DSR overstates significance) | Document explicitly in the manual OOS run notes; `n_trials` is a parameter, not hardcoded, so it can be revised upward if a trial log (Phase 9) later reveals a larger true count |
| Gate depends on Phase 3's `compute_equity_metrics` existing with the exact dict keys assumed here (`sharpe`, `sortino`, `dsr`, `max_drawdown_pct`, `ann_return_pct`) | Low (both plans are self-consistent) | High if the two phases implemented independently drift | This plan's Mandatory Reading explicitly points at the sibling plan; implement Phase 3 first or verify the dict shape before writing `_evaluate_gate` |
| `BacktestParams`/`run_backtest` kwarg shape for `max_hold_bars` may have changed by the time Phase 4/5 land (tier shift, Donchian params) | Medium | Medium (Task 1 code may need a small adaptation) | Task 1 explicitly calls out checking `engine.py`'s actual signature before writing the grid-to-kwargs split |
| A FAIL gate gets quietly re-run with a loosened grid, defeating the one-shot holdout's purpose | Medium (human process risk, not code) | High (invalidates the whole validation) | Documented prominently in Manual Validation and the PRD's own Decisions Log ("Gate-failure response: Pre-committed escalation to Option B") |

## Notes
- This phase does not implement Donchian (Phase 5) or the tier shift (Phase 4) — it assumes `BacktestParams` and `run_backtest` already have whatever shape those phases left them in, and its own grid axes (`rr_floor`, `max_hold_bars`) are chosen to exist regardless of implementation order, since both are already named PRD constants (`RR_FLOOR` from Phase 2, `MAX_HOLD_BARS_15M` renamed in Phase 4).
- The "3× the sample at zero additional degrees of freedom" framing from the PRD is the single most load-bearing idea in this plan — every design choice (pooling before sweeping, not after) exists to preserve that property. Do not pool only the OOS run while leaving per-symbol tuning; that would reintroduce per-symbol overfitting risk while only fixing the final verdict's sample size.
- Per the PRD's own phase table, this is the hard barrier: Phase 8 (sizing) and Phase 9 (alerting) should not start on a FAIL, and should not start speculatively before this phase's gate returns any verdict at all.
