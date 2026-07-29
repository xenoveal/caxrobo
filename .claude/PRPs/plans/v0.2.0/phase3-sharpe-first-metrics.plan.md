# Plan: Sharpe-First Metrics (PRD Phase 3)

## Summary
The PRD's north star is "Sharpe ≥ 1.0 net, max DD ≤ 25%", but `metrics.py` cannot measure any of it: it computes expectancy/PF over an **unordered** `pnl_pct` list and its "max drawdown" is a sum-of-percentages proxy on equal-sized trades. This phase adds a **time-indexed** daily return series built from the trade list, real Sharpe/Sortino with annualization, compounded-equity max drawdown, and a Deflated Sharpe Ratio (DSR) primitive — as a **new code path** alongside (never replacing) the existing equal-size bucket stats, which Phase 7's gate and Phase 8's sizing both consume.

## User Story
As the bot's sole user, I want backtest output stated in Sharpe / Sortino / equity-curve drawdown terms, so that the Phase 7 gate can deliver a verdict against the actual success metrics instead of a per-trade expectancy proxy.

## Problem → Solution
**Current**: `compute_metrics` treats trades as an orderless bag; no notion of time, compounding, or idle periods. A strategy that makes 1% per trade once a year and one that does it daily are indistinguishable.
**Solution**: New pure module `backtest/equity.py`: `daily_returns(trades, start_ms, end_ms)` → calendar-complete daily return series (trade pnl booked on exit day, zero-return days included); `sharpe_ratio` / `sortino_ratio` (annualized √365); `max_drawdown` on the compounded equity curve; `deflated_sharpe(...)` implementing Bailey & López de Prado with stdlib `statistics.NormalDist` (no scipy). `compute_metrics` and its callers stay untouched except the CLI printing the new numbers.

## Metadata
- **Complexity**: Medium (3 files touched, ~350 new lines incl. tests; no new dependencies)
- **Source PRD**: `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`
- **PRD Phase**: Phase 3 — Sharpe-first metrics
- **Estimated Files**: 4 (2 new, 2 modified)
- **Parallelism**: Independent of Phases 1 and 2 (touches only the metrics layer). Phase 7 hard-depends on this — it imports `compute_equity_metrics` and `deflated_sharpe` from the new module.

---

## UX Design

N/A — internal change. Only observable output: `python -m trading_bot.cli backtest` prints an extra line of equity metrics per symbol.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/trading_bot/backtest/metrics.py` | all (73 lines) | The module being *extended, not modified*. Its docstring's "each trade is treated as equal-sized" assumption is exactly what the new path relaxes over time (still equal notional per trade, but time-placed) |
| P0 | `src/trading_bot/backtest/engine.py` | 60-80 | `Trade` dataclass — fields available: `entry_ts`, `exit_ts` (epoch **ms**), `pnl_pct` (fraction, net of costs). This is the entire input contract |
| P1 | `src/trading_bot/cli.py` | 355-385 | `_fmt` / `_print_metrics` / `_backtest_command` — the printing pattern to mirror for the new equity line |
| P1 | `tests/test_backtest.py` | 16-54 | `make_trade()` helper + `TestComputeMetrics` — the test idiom to mirror (class-per-function, `math.isclose`) |
| P2 | `src/trading_bot/indicators/wilder.py` | all | Style reference for pure, dependency-light numeric functions with Args/Returns docstrings |

## External Documentation

| Topic | Source | Key Takeaway |
|---|---|---|
| Deflated Sharpe Ratio | Bailey & López de Prado (2014), "The Deflated Sharpe Ratio" | Full formulas reproduced in Task 3 below — **no further research needed during implementation** |
| Annualization | Crypto convention | Perps trade 365 d/yr → √365, not √252 |

---

## Patterns to Mirror

### MODULE_DOCSTRING + PURE FUNCTIONS
```python
# SOURCE: metrics.py:1-8 — module docstring states the computation model and
# assumptions up front; functions are pure (no I/O, no config import).
"""
Trade-list performance metrics.

Pure computation over the engine's Trade list. Expectancy and drawdown are in
percent-of-entry terms (each trade is treated as equal-sized, ...
"""
```

### LOGGING_PATTERN
```python
# SOURCE: metrics.py:10-12
import logging
logger = logging.getLogger("trading_bot")
```

### NONE_FOR_UNDEFINED_RATIOS
```python
# SOURCE: metrics.py:41-50 — undefined statistics are None, never NaN/exception
if n == 0:
    return {"n_trades": 0, "win_rate": None, ...}
```
Sharpe with zero return-variance, Sortino with no negative days, DSR with n_obs < 2 → all return `None`, same convention.

### CLI_PRINT_PATTERN
```python
# SOURCE: cli.py:360-367
def _print_metrics(m: dict, indent: str = "") -> None:
    print(
        f"{indent}trades={m['n_trades']}  win_rate={_fmt(m['win_rate'], '.2%')}  ..."
    )
```

### TEST_STRUCTURE
```python
# SOURCE: tests/test_backtest.py:23-46 — make_trade() builder, class-per-function
def make_trade(pnl, regime="trending", pattern="flag"):
    return Trade(symbol=SYMBOL, ..., entry_ts=START, exit_ts=START + M15,
                 pnl_pct=pnl, volume_high=False)

class TestComputeMetrics:
    def test_basic_stats(self):
        ...
        assert math.isclose(m["max_drawdown_pct"], 0.005)
```
Note: for equity tests, `make_trade` needs an `exit_ts` parameter added (it currently hardcodes `START + M15`) — extend the helper with `exit_ts=None` defaulting to `entry_ts + M15`, don't fork a new builder.

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/backtest/equity.py` | CREATE | All new math lives here: `daily_returns`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `deflated_sharpe`, `compute_equity_metrics` |
| `src/trading_bot/cli.py` | UPDATE | `_backtest_command` prints the equity metrics line after the existing ones |
| `tests/test_equity.py` | CREATE | Unit tests incl. hand-computed Sharpe fixture (the phase's stated success signal) |
| `tests/test_backtest.py` | UPDATE | Extend `make_trade()` with `exit_ts` param; add one CLI assertion for the new output line |

## NOT Building

- **No change to `compute_metrics` or its return dict** — Phase 7's fold logic and every existing test consume it as-is; equity metrics are an additive, separate function.
- **No vol-targeted equity curve** — that is Phase 8; this phase's curve assumes equal notional per trade (1 unit of equity per trade, pnl compounds daily).
- **No walk-forward integration** — Phase 7 calls `compute_equity_metrics`; this phase only makes it exist.
- **No intraday (sub-daily) return resolution** — daily bucketing is sufficient for Sharpe on multi-day-hold strategies and keeps the series length honest (~1,300 obs over the data span).
- **No scipy/numpy-stats dependency** — `statistics.NormalDist` (stdlib, Python ≥3.8) covers Φ and Φ⁻¹.
- **No benchmark-relative metrics** (alpha/beta vs BTC) — not in the success-metric table.

---

## Step-by-Step Tasks

### Task 1: `daily_returns` — the time-indexed series
- **ACTION**: Create `src/trading_bot/backtest/equity.py` with module docstring (mirror `metrics.py`'s) and the series builder.
- **IMPLEMENT**:
  ```python
  """
  Time-indexed equity metrics (Phase 3: Sharpe-first).

  Builds a calendar-complete daily return series from the engine's Trade list
  and computes Sharpe / Sortino / equity-curve max drawdown / Deflated Sharpe
  Ratio on it. Each trade contributes its full net pnl_pct on its EXIT day
  (equal notional per trade — the vol-targeted curve arrives in Phase 8).
  Days with no exits are zero-return days and are INCLUDED: idle time is real
  time, and excluding it inflates Sharpe.

  Pure computation, no I/O, no config coupling. Pools naturally across
  symbols: pass trades from several symbols and their same-day pnls sum.
  """
  import logging
  import math
  import statistics
  from statistics import NormalDist

  logger = logging.getLogger("trading_bot")

  DAY_MS = 86_400_000
  PERIODS_PER_YEAR = 365  # crypto perps trade every calendar day


  def daily_returns(trades, start_ms: int, end_ms: int) -> list[float]:
      """Calendar-complete daily return series over [start_ms, end_ms).

      Args:
          trades: Iterable of engine.Trade (needs exit_ts, pnl_pct).
          start_ms / end_ms: Span in epoch ms; every UTC day in the span gets
              an entry (0.0 if no trade exited that day).

      Returns:
          List of daily returns (fractions), one per UTC day, oldest first.
          Empty list if the span is empty or inverted.
      """
      if end_ms <= start_ms:
          return []
      first_day = start_ms // DAY_MS
      n_days = (end_ms - 1) // DAY_MS - first_day + 1
      rets = [0.0] * n_days
      for t in trades:
          d = t.exit_ts // DAY_MS - first_day
          if 0 <= d < n_days:
              rets[d] += t.pnl_pct
          else:
              logger.warning("trade exit_ts %d outside metrics span; dropped", t.exit_ts)
      return rets
  ```
- **MIRROR**: `metrics.py` module shape (docstring → logger → functions).
- **IMPORTS**: stdlib only. Deliberately **no pandas** — plain lists keep the math transparent and the tests hand-checkable.
- **GOTCHA**: `exit_ts` is epoch **ms** (bar open time of the exit bar). Integer floor-division by `DAY_MS` gives the UTC day — do not use local-time datetime conversion. Summing same-day pnls (multiple exits, multiple symbols) is correct under the equal-notional model.
- **VALIDATE**: `pytest tests/test_equity.py::TestDailyReturns -v` (Task 5).

### Task 2: Sharpe, Sortino, max drawdown
- **ACTION**: Same file, add the three ratio functions.
- **IMPLEMENT**:
  ```python
  def sharpe_ratio(returns: list[float], periods_per_year: int = PERIODS_PER_YEAR) -> float | None:
      """Annualized Sharpe (risk-free rate 0). None if < 2 obs or zero variance."""
      if len(returns) < 2:
          return None
      mu = statistics.fmean(returns)
      sd = statistics.stdev(returns)  # sample stdev (n-1)
      if sd == 0:
          return None
      return (mu / sd) * math.sqrt(periods_per_year)


  def sortino_ratio(returns: list[float], periods_per_year: int = PERIODS_PER_YEAR) -> float | None:
      """Annualized Sortino: mean / downside deviation (all obs in denominator).

      Downside deviation = sqrt(mean(min(r, 0)^2)) over ALL returns — the
      full-series convention, not stdev of the losing subset.
      None if < 2 obs or no downside at all.
      """
      if len(returns) < 2:
          return None
      mu = statistics.fmean(returns)
      dd = math.sqrt(statistics.fmean([min(r, 0.0) ** 2 for r in returns]))
      if dd == 0:
          return None
      return (mu / dd) * math.sqrt(periods_per_year)


  def max_drawdown(returns: list[float]) -> float | None:
      """Peak-to-trough drawdown (positive fraction) on the COMPOUNDED equity
      curve — not the sum-of-percentages proxy metrics.py uses. None if empty."""
      if not returns:
          return None
      equity = peak = 1.0
      max_dd = 0.0
      for r in returns:
          equity *= 1.0 + r
          peak = max(peak, equity)
          max_dd = max(max_dd, 1.0 - equity / peak)
      return max_dd
  ```
- **MIRROR**: None-for-undefined convention from `metrics.py`.
- **GOTCHA**: Sharpe here is on the **daily** series; annualize by √periods, never by annualizing the mean and stdev separately. `statistics.stdev` is the n−1 sample estimator — the test fixture in Task 5 must hand-compute with n−1 too.
- **VALIDATE**: `pytest tests/test_equity.py::TestRatios -v`.

### Task 3: Deflated Sharpe Ratio primitive
- **ACTION**: Same file. This is the piece Phase 7 imports; get the formulas exactly right — they are reproduced here in full so no external lookup is needed.
- **IMPLEMENT**:
  ```python
  _EULER_GAMMA = 0.5772156649015329


  def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int,
                           skew: float, kurt: float) -> float | None:
      """PSR: probability the true (per-period) Sharpe exceeds sr_benchmark.

      PSR = Phi( (sr - sr*) * sqrt(n_obs - 1)
                 / sqrt(1 - skew*sr + ((kurt - 1) / 4) * sr^2) )

      All Sharpe values PER-PERIOD (daily), NOT annualized. kurt is the raw
      (non-excess) kurtosis: 3.0 for a normal distribution.
      """
      if n_obs < 2:
          return None
      denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
      if denom_sq <= 0:
          return None  # pathological higher moments; refuse rather than lie
      z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
      return NormalDist().cdf(z)


  def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
      """E[max SR] under n_trials independent zero-true-Sharpe trials
      (Bailey & Lopez de Prado eq. for the expected maximum):

      SR0 = sqrt(sr_var) * ((1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N*e)))
      """
      if n_trials <= 1 or sr_var <= 0:
          return 0.0
      z = NormalDist().inv_cdf
      return math.sqrt(sr_var) * (
          (1.0 - _EULER_GAMMA) * z(1.0 - 1.0 / n_trials)
          + _EULER_GAMMA * z(1.0 - 1.0 / (n_trials * math.e))
      )


  def deflated_sharpe(sr: float, n_trials: int, n_obs: int,
                      skew: float, kurt: float,
                      sr_var: float | None = None) -> float | None:
      """DSR: PSR evaluated against the expected-max Sharpe of the search.

      Args:
          sr: Observed PER-PERIOD (daily) Sharpe of the selected strategy.
          n_trials: TOTAL configurations evaluated during the search
              (grid combos x folds + neighbor probes — the caller counts).
          n_obs: Length of the daily return series.
          skew / kurt: Sample skewness and raw kurtosis of the daily returns.
          sr_var: Variance of Sharpe estimates across trials. Default: the
              SR estimator variance (1 - skew*sr + (kurt-1)/4*sr^2)/(n_obs-1)
              — a conservative stand-in when per-trial SRs weren't retained.

      Returns:
          Probability in [0, 1]; DSR > 0.95 == significant at p < 0.05.
          None when undefined (n_obs < 2 or pathological moments).
      """
      if n_obs < 2:
          return None
      if sr_var is None:
          v = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
          if v <= 0:
              return None
          sr_var = v / (n_obs - 1)
      sr0 = expected_max_sharpe(n_trials, sr_var)
      return probabilistic_sharpe(sr, sr0, n_obs, skew, kurt)


  def _skew_kurt(returns: list[float]) -> tuple[float, float]:
      """Sample skewness and RAW kurtosis (normal => 3.0). (0.0, 3.0) if degenerate."""
      n = len(returns)
      if n < 2:
          return 0.0, 3.0
      mu = statistics.fmean(returns)
      m2 = statistics.fmean([(r - mu) ** 2 for r in returns])
      if m2 == 0:
          return 0.0, 3.0
      m3 = statistics.fmean([(r - mu) ** 3 for r in returns])
      m4 = statistics.fmean([(r - mu) ** 4 for r in returns])
      return m3 / m2 ** 1.5, m4 / m2 ** 2
  ```
- **MIRROR**: `wilder.py`'s style of formula-bearing docstrings.
- **GOTCHA #1**: **Everything in DSR-land is per-period (daily) Sharpe** — `sharpe_ratio()` returns the *annualized* number for reporting; `compute_equity_metrics` (Task 4) must divide by √365 (or compute `mu/sd` directly) before feeding PSR/DSR. Mixing these up inflates significance by ~19×.
- **GOTCHA #2**: `kurt` is raw kurtosis (normal = 3), matching the `(kurt − 1)/4` term in the published formula. If a future reader swaps in excess kurtosis the denominator silently shrinks.
- **VALIDATE**: `pytest tests/test_equity.py::TestDSR -v` — includes the sanity anchors in Task 5.

### Task 4: `compute_equity_metrics` aggregator + CLI wiring
- **ACTION**: Same file, one dict-returning entry point (the shape `_print_metrics` expects); then edit `cli.py`.
- **IMPLEMENT**:
  ```python
  def compute_equity_metrics(trades, start_ms: int, end_ms: int,
                             n_trials: int = 1) -> dict:
      """One-stop equity metrics for a trade list over a span.

      Returns dict with: n_days, sharpe (annualized), sortino (annualized),
      max_drawdown_pct, ann_return_pct (compounded, reported-not-gated per
      PRD), dsr (probability), daily_sharpe. None values where undefined.
      """
      rets = daily_returns(trades, start_ms, end_ms)
      n = len(rets)
      sr_daily = None
      if n >= 2:
          sd = statistics.stdev(rets)
          sr_daily = (statistics.fmean(rets) / sd) if sd > 0 else None
      skew, kurt = _skew_kurt(rets)
      equity = 1.0
      for r in rets:
          equity *= 1.0 + r
      ann_return = equity ** (PERIODS_PER_YEAR / n) - 1.0 if n > 0 and equity > 0 else None
      return {
          "n_days": n,
          "sharpe": sharpe_ratio(rets),
          "sortino": sortino_ratio(rets),
          "max_drawdown_pct": max_drawdown(rets),
          "ann_return_pct": ann_return,
          "daily_sharpe": sr_daily,
          "dsr": deflated_sharpe(sr_daily, n_trials, n, skew, kurt)
                 if sr_daily is not None else None,
      }
  ```
  In `cli.py`: import `compute_equity_metrics`; in `_backtest_command` after `_print_metrics(m, indent="  ")` add:
  ```python
  em = compute_equity_metrics(trades, start_ms, end_ms)
  print(
      f"  equity: sharpe={_fmt(em['sharpe'], '.2f')}  "
      f"sortino={_fmt(em['sortino'], '.2f')}  "
      f"max_dd={_fmt(em['max_drawdown_pct'], '.2%')}  "
      f"ann_return={_fmt(em['ann_return_pct'], '.2%')}"
  )
  ```
- **MIRROR**: `_fmt`/`_print_metrics` formatting (cli.py:355-367).
- **GOTCHA**: `_backtest_command` receives `start_ms`/`end_ms` already resolved by `main()` — pass those, not the trade list's own bounds (idle head/tail time must count). Do not print `dsr` in the plain backtest command — `n_trials=1` makes it a plain PSR and printing it invites misreading; DSR reporting belongs to Phase 7's walk-forward output.
- **VALIDATE**: `pytest tests/test_backtest.py::TestBacktestCli -v` (extended in Task 5).

### Task 5: Tests
- **ACTION**: Create `tests/test_equity.py`; extend `tests/test_backtest.py`.
- **IMPLEMENT** — key cases (mirror `make_trade` from test_backtest, adding `exit_ts`):
  ```python
  from trading_bot.backtest.equity import (
      DAY_MS, daily_returns, sharpe_ratio, sortino_ratio, max_drawdown,
      deflated_sharpe, expected_max_sharpe, probabilistic_sharpe,
      compute_equity_metrics,
  )

  class TestDailyReturns:
      # span of 3 days, one trade exiting day 1 -> [0, pnl, 0]
      # two trades same day -> summed
      # trade outside span -> dropped (and series unchanged)
      # empty span -> []

  class TestRatios:
      def test_sharpe_matches_hand_computed(self):
          # PRD success signal: hand-computed fixture.
          rets = [0.01, -0.005, 0.02, 0.0, -0.01]
          mu = sum(rets) / 5                      # 0.003
          sd = statistics.stdev(rets)             # n-1 estimator
          assert math.isclose(sharpe_ratio(rets), mu / sd * math.sqrt(365))
      # zero-variance -> None; single obs -> None
      # sortino: all-positive returns -> None (no downside)
      # max_drawdown compounds: [0.10, -0.10] -> dd = 0.10 (not 0.0):
      #   equity 1.10 -> 0.99, peak 1.10, dd = 1 - 0.99/1.10
      # constant negative returns -> dd ≈ 1 - (1+r)^n / 1

  class TestDSR:
      # anchors that catch formula transcription errors:
      # probabilistic_sharpe(sr=0, benchmark=0, ...) == 0.5 exactly
      # expected_max_sharpe(1, v) == 0.0; monotone increasing in n_trials
      # deflated_sharpe with n_trials=1 > with n_trials=100 (same sr):
      #   more search -> lower probability
      # normal-ish fixture: skew=0, kurt=3 -> denominator sqrt(1 + sr^2/2)

  class TestComputeEquityMetrics:
      # end-to-end on 3 synthetic trades over a 10-day span;
      # assert every key present, ann_return sign matches pnl sign
  ```
  In `test_backtest.py`: change `make_trade` signature to `make_trade(pnl, regime="trending", pattern="flag", exit_ts=None)` with `exit_ts = START + M15 if exit_ts is None else exit_ts` (all existing call sites keep working); in `test_backtest_command_prints_metrics`, add `assert "sharpe=" in out`.
- **MIRROR**: `TestComputeMetrics` (test_backtest.py:32-53) — class-per-function, `math.isclose`.
- **GOTCHA**: The hand-computed Sharpe must use `statistics.stdev` (n−1); recomputing with population stdev makes the fixture "mysteriously" fail by a few percent.
- **VALIDATE**: `pytest tests/test_equity.py tests/test_backtest.py -v`.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| daily_returns placement | 1 trade, 3-day span | `[0, pnl, 0]` | — |
| daily_returns pooling | 2 trades same UTC day | summed into one entry | — |
| daily_returns out-of-span | exit_ts past end | dropped + warning | Edge |
| sharpe hand-computed | 5 fixed returns | `mu/sd*sqrt(365)` exact | PRD success signal |
| sharpe zero variance | constant returns | None | Edge |
| sortino no downside | all positive | None | Edge |
| max_drawdown compounding | `[0.10, -0.10]` | `1 - 0.99/1.10` | Proves not sum-proxy |
| PSR at benchmark | sr == sr* | 0.5 | Formula anchor |
| E[maxSR] monotone | n=1 vs n=100 | 0.0 vs > 0 | Formula anchor |
| DSR search penalty | n_trials 1 vs 100 | strictly decreasing | Core DSR property |

### Edge Cases Checklist
- [x] Empty trade list / empty span → empty series, all-None metrics
- [x] Single observation → None ratios (never ZeroDivisionError)
- [x] Zero variance / no downside / degenerate moments → None
- [x] Trade exiting outside the span → dropped, warned, not crashed
- [x] Pathological skew/kurt making PSR denominator ≤ 0 → None
- [ ] Concurrent access — N/A (pure functions)
- [ ] Network — N/A

---

## Validation Commands

### Static Analysis
```bash
python -m py_compile src/trading_bot/backtest/equity.py src/trading_bot/cli.py
```
EXPECT: clean. (No mypy/linter configured in this project — skip.)

### Unit Tests
```bash
pytest tests/test_equity.py tests/test_backtest.py -v
```
EXPECT: all pass, including the hand-computed Sharpe fixture.

### Full Test Suite
```bash
pytest tests/ -v
```
EXPECT: no regressions — `compute_metrics` and every existing consumer untouched.

### Manual Validation
```bash
python -m trading_bot.cli backtest --symbol BTCUSDT
```
- [ ] Output shows an `equity:` line with sharpe/sortino/max_dd/ann_return.
- [ ] `max_dd` (compounded) differs from the legacy `max_dd` (sum proxy) — if identical to 4 decimals on a real run, suspect the wrong series was wired in.

---

## Acceptance Criteria
- [ ] All 5 tasks completed; all validation commands pass
- [ ] Existing `compute_metrics` behavior and return shape byte-identical
- [ ] Sharpe matches the hand-computed fixture (PRD phase success signal)
- [ ] DSR primitive exported with the exact signature Phase 7 needs: `deflated_sharpe(sr, n_trials, n_obs, skew, kurt, sr_var=None)`

## Completion Checklist
- [ ] Pure functions, no config import, stdlib-only (matches `wilder.py` ethos)
- [ ] None-for-undefined convention throughout
- [ ] `logging.getLogger("trading_bot")`, WARNING only for dropped trades
- [ ] No unnecessary scope: no vol-targeting, no walk-forward changes, no pandas

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Annualized-vs-daily Sharpe confusion in DSR | Medium | High (fake significance) | `daily_sharpe` is a separate dict key; DSR docstring screams per-period; test asserts DSR uses the daily value |
| Excess-vs-raw kurtosis mixup | Medium | Medium | `_skew_kurt` returns raw (normal=3), documented at both producer and consumer |
| Booking whole pnl on exit day understates intra-trade drawdown | Certain (by design) | Low-Medium | Documented in module docstring; equity max-DD is a lower bound; acceptable for v1, tighten only if 1m exit resolution lands |
| `sr_var` default is a stand-in, not the true cross-trial variance | Medium | Medium | Signature accepts explicit `sr_var`; Phase 7 may pass the variance of fold-level Sharpes if it retains them |

## Notes
- The equal-size `compute_metrics` path is deliberately preserved forever: it is the signal-quality attribution tool (per-bucket stats) and Phase 8's vol-targeted curve will need the *contrast* against it.
- Everything here pools across symbols for free (returns sum by day) — that is exactly what Phase 7's pooled gate needs; no per-symbol assumption anywhere in this module.
