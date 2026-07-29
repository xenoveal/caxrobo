# Plan: Honest Cost & Risk Model (PRD Phase 2)

## Summary
Replace the infeasible fixed-percentage R:R band (`MAX_RISK_PCT ≤0.5%` / `MIN_REWARD_PCT ≥0.75%`) with an ATR-scaled stop and an R:R-**ratio** floor, and make transaction costs honest (0.05% taker, frozen) and complete (add a funding term). This is the single highest-impact change identified by the market-research benchmark and doubles as a cheap falsification test: re-running the existing walk-forward gate on the *existing* breakout/fade methods under the new risk model tells us whether the risk model was the whole problem, before any new signal engine (Donchian, Phase 5) is built.

## User Story
As the bot's sole user (a discretionary crypto trader), I want the stop-loss and target computation to reflect actual market volatility and honest trading costs instead of an arbitrary percentage band, so that the backtest's verdict on whether the strategy has edge is trustworthy rather than an artifact of noise-sized stops.

## Problem → Solution
**Current**: `build_signal`/`build_fade_signal` reject candidates whose stop-to-entry distance exceeds a fixed 0.5% or whose reward is below 0.75% — a band that sits entirely inside the measured 15m noise floor (median bar range 0.25–0.52%), producing gross expectancy ≈ 0 on all 3 symbols. Costs are modeled at 0.04% taker (Binance's actual VIP-0 taker is 0.05%) with no funding term.
**Solution**: Stop = `ATR_STOP_MULTIPLE × ATR(setup timeframe)` (a market-structure quantity), filtered by `rr = reward/risk ≥ RR_FLOOR` (a ratio, not an absolute band). `MAX_RISK_PCT`/`MIN_REWARD_PCT` are deleted from stop/filter logic entirely — documented as an account-risk budget the human discharges via position sizing, out of this bot's scope. Costs: `FEE_PCT = 0.0005`, plus a new `FUNDING_PCT_PER_DAY` term scaled by holding days.

## Metadata
- **Complexity**: Medium (7 files touched, ~250 net new/changed lines including tests; no new external dependencies)
- **Source PRD**: `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`
- **PRD Phase**: Phase 2 — Honest cost & risk model
- **Estimated Files**: 7 (2 new, 5 modified)

---

## UX Design

N/A — internal change. No user-facing surface exists yet (alerting is Phase 9). The only observable output is the CLI's existing `signal`/`backtest` commands returning different (more conservative, fewer) signals and different backtest metrics.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/trading_bot/signals/setup.py` | 1-186 | `build_signal` is the function being rewritten; docstring at 96-99 already flags this as deferred "Phase 6" work — it's now this phase |
| P0 | `src/trading_bot/signals/meanrev.py` | 173-241 | `build_fade_signal` is structurally parallel to `build_signal` and needs the identical stop/filter change |
| P0 | `src/trading_bot/indicators/wilder.py` | 95-109 | `atr(df, period)` — the exact function to call; note it returns NaN for the first `period-1` rows (Wilder-smoothed, trailing-only, no lookahead) |
| P0 | `src/trading_bot/config.py` | 67-100 | Constants being added/removed: `MAX_RISK_PCT`/`MIN_REWARD_PCT`/`BREAKOUT_STOP_BUFFER_PCT`/`BREAKOUT_MAX_ENTRY_EXTENSION_PCT` retired from stop logic; `FEE_PCT`, `SLIPPAGE_PCT` frozen; new constants added here |
| P1 | `src/trading_bot/backtest/engine.py` | 91-125, 176-198 | `run_backtest`'s cost line (`cost = 2 * (fee + slip)`) needs the funding term; `close_out` computes `pnl_pct` — the funding term must scale with holding duration, which this closure has via `entry_ts`/`exit_ts` |
| P1 | `src/trading_bot/signals/breakout.py` | 38-58 | `BreakoutEvent` — unchanged, but confirms `build_signal`/`build_fade_signal`'s only inputs are `candidate` (level/direction/target_height) and `event` (entry price); ATR must be threaded in separately since neither carries it today |
| P1 | `tests/test_signals.py` | 395-447, 507-550 | `TestBuildSignal` and the `test_trending_flag_breakout_produces_signal` integration test hard-assert the retired constants — must be rewritten, not deleted |
| P2 | `tests/test_meanrev.py` | ~140-230 | Same retired-constant assertions on the fade side — mirror the rewrite |
| P2 | `src/trading_bot/signals/patterns.py` | (PatternCandidate dataclass) | Confirms `candidate.target_height` is still used for `target` computation — untouched by this phase |

## External Documentation

No external research needed — ATR (Wilder 1978) is already implemented in this codebase (`indicators/wilder.py`), and the `k = 1.5`, `RR_FLOOR = 1.5`, `c ≤ 0.10` values are derived quantities already computed in `.claude/PRPs/reports/market-research-capability-benchmark.md` §2.4 and §R3, not new research.

---

## Patterns to Mirror

### NAMING_CONVENTION
```python
# SOURCE: config.py:69-70 (existing) — SCREAMING_SNAKE module constants, grouped
# under a comment banner naming the phase that introduced them
MAX_RISK_PCT = 0.005  # entry-to-SL distance must be <= 0.5% of entry
MIN_REWARD_PCT = 0.0075  # entry-to-TP distance must be >= 0.75% of entry
```
New constants follow the same style: `ATR_STOP_MULTIPLE`, `ATR_STOP_PERIOD`, `RR_FLOOR`, `FUNDING_PCT_PER_DAY`, `COST_RATIO_CEILING`.

### ERROR_HANDLING / REJECTION LOGGING
```python
# SOURCE: setup.py:145-171 — local reject() closure logs at DEBUG with symbol/
# pattern/direction context, then early-returns None. No exceptions raised for
# a candidate that fails the filter — this is expected, high-frequency control
# flow, not an error.
def reject(reason: str, *args) -> None:
    logger.debug(
        "%s %s %s rejected: " + reason,
        symbol, candidate.kind, candidate.direction, *args,
    )
...
if extension > max_entry_extension_pct:
    reject("entry %.4f is %.3f%% past level %.4f (cap %.3f%%)", ...)
    return None
```
Keep this shape for the new `rr < RR_FLOOR` rejection. Do NOT raise/assert inside `build_signal` for a failed filter — that is normal per-candidate rejection, not a data or config error.

### LOGGING_PATTERN
```python
# SOURCE: setup.py:36 — one module logger, named "trading_bot" (matches every
# other module in the package; Phase 7's Discord sink attaches here later)
logger = logging.getLogger("trading_bot")
```

### DATACLASS_SIGNAL_SHAPE
```python
# SOURCE: setup.py:39-69 — frozen dataclass, exhaustive field docstring
@dataclass(frozen=True)
class Signal:
    symbol: str
    ts: int
    direction: str
    pattern: str
    entry: float
    stop: float
    target: float
    risk_pct: float
    reward_pct: float
    rr: float
    volume_ratio: float
    volume_high: bool
```
This dataclass is UNCHANGED by this phase (no new fields needed — `rr` already exists and becomes the filter's basis instead of a derived-only value).

### OPTIONAL_OVERRIDE_PARAMS
```python
# SOURCE: setup.py:76-123 — every tunable has a keyword-only `| None = None`
# parameter that falls back to a config default; this is how walk-forward
# sweeps parameters without mutating global config.
def build_signal(
    symbol: str,
    candidate: PatternCandidate,
    event: BreakoutEvent,
    *,
    max_risk_pct: float | None = None,
    min_reward_pct: float | None = None,
    ...
) -> Signal | None:
    if max_risk_pct is None:
        max_risk_pct = config.MAX_RISK_PCT
```
New signature keeps this shape but swaps the parameter set (see Task 2).

### COST_MODEL_PATTERN
```python
# SOURCE: engine.py:121-124 — costs resolved once per backtest run, config
# defaults overridable via keyword args (mirrors the sweepable-param pattern)
fee = config.FEE_PCT if fee_pct is None else fee_pct
slip = config.SLIPPAGE_PCT if slippage_pct is None else slippage_pct
...
cost = 2 * (fee + slip)
```

### TEST_STRUCTURE
```python
# SOURCE: tests/test_signals.py:395-406 — class-per-function-under-test,
# `make_candidate()`/`breakout_df()` helper builders (defined earlier in the
# file), direct dataclass field assertions, math.isclose for float comparisons
class TestBuildSignal:
    def test_long_setup_passes_band(self):
        df = breakout_df(prev_close=99.8, last_close=100.3)
        candidate = make_candidate(level=100.0, height=1.2)
        event = check_breakout(df, candidate)
        signal = build_signal(SYMBOL, candidate, event)
        assert signal is not None
        assert math.isclose(signal.stop, 100.0 * (1 - config.BREAKOUT_STOP_BUFFER_PCT))
        ...
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/config.py` | UPDATE | Retire `MAX_RISK_PCT`, `MIN_REWARD_PCT`, `BREAKOUT_STOP_BUFFER_PCT`, `BREAKOUT_MAX_ENTRY_EXTENSION_PCT` from stop/filter role (keep `MAX_RISK_PCT` as a documented-only account-risk-budget constant, unused by signal code); add `ATR_STOP_MULTIPLE = 1.5`, `ATR_STOP_PERIOD = 14`, `RR_FLOOR = 1.5`, `FEE_PCT = 0.0005`, `FUNDING_PCT_PER_DAY`, `COST_RATIO_CEILING = 0.10` |
| `src/trading_bot/risk/__init__.py` | CREATE | New package for the shared stop/cost-ratio helper, since both `setup.py` and `meanrev.py` need identical logic (avoid duplicating ATR-stop math in two modules) |
| `src/trading_bot/risk/atr_stop.py` | CREATE | `compute_atr_stop(entry, direction, atr_value, k) -> float`, `cost_ratio(risk_pct, fee_pct, slip_pct) -> float` — pure functions, no I/O |
| `src/trading_bot/signals/setup.py` | UPDATE | `build_signal` takes an `atr_value: float` param instead of `stop_buffer_pct`/`max_entry_extension_pct`; stop = ATR-derived; filter on `rr >= rr_floor` instead of absolute band; `scan_breakout_signals` computes ATR from the 1H df and passes it through |
| `src/trading_bot/signals/meanrev.py` | UPDATE | `build_fade_signal` takes `atr_value`; filter on `rr >= rr_floor`; NOTE — fade's stop stays at the excursion extreme (structurally already correct per PRD Decisions Log), so ATR here informs the *filter*, not a new stop placement. See Task 3 for the precise semantics. |
| `src/trading_bot/backtest/engine.py` | UPDATE | Compute ATR series once per symbol from the 1H df; thread `atr_value` into both `build_signal`/`build_fade_signal` calls; add funding term to `close_out`'s cost computation, scaled by `(exit_ts - entry_ts)` in days |
| `tests/test_signals.py` | UPDATE | Rewrite `TestBuildSignal` and the retired-constant assertions in `TestCurrentSignals` for the new stop/filter signature |
| `tests/test_meanrev.py` | UPDATE | Mirror rewrite for `build_fade_signal`'s tests |
| `tests/test_risk_atr_stop.py` | CREATE | Unit tests for the new pure helper functions |
| `tests/test_backtest.py` | UPDATE | Add a funding-term assertion; verify median stop distance / ATR ratio and cost ratio on a synthetic fixture |

## NOT Building

- Donchian channel breakout signal engine (Phase 5) — this phase only fixes risk/cost for the *existing* breakout and fade methods.
- Tier shift to 1D/4H/1H (Phase 4) — ATR is computed on the **current** setup timeframe (`config.SIGNAL_PATTERN_TIMEFRAME`, i.e. `1h`), not the eventual 4H. `k=1.5` is frozen now against 1H-tier ATR; if Phase 4 changes the setup timeframe, `k` is re-derived then, not swept.
- Sharpe/Sortino/DSR metrics (Phase 3) — `metrics.py` is untouched; this phase only changes what feeds into `pnl_pct`.
- A runtime hard-fail if `c > COST_RATIO_CEILING` — the ceiling is asserted in tests against measured backtest output, not enforced as a live rejection (a legitimately high-cost regime should still produce a Signal; the ceiling is a validation gate on the strategy as a whole, not a per-candidate filter).
- Sweeping `ATR_STOP_MULTIPLE` or `RR_FLOOR` in walk-forward — both are frozen constants per the PRD's explicit decision (fitting `k` on returns is the mistake Phase 5's buffer sweep already made).
- Any change to `regime/classifier.py`, `data/storage.py`, `data/backfill.py`, `data/poller.py`, `exchange/binance_client.py`, `indicators/wilder.py`, `indicators/bollinger.py`, `signals/breakout.py`, `signals/patterns.py`, `signals/pivots.py` — none of these need to change for this phase.

---

## Step-by-Step Tasks

### Task 1: Add and retire config constants
- **ACTION**: Edit `config.py`.
- **IMPLEMENT**:
  ```python
  # Phase 2: honest cost & risk model. Stop distance is now derived from
  # volatility (ATR), never from a fixed percentage of entry. MAX_RISK_PCT
  # below is retained ONLY as documentation of the account-risk budget the
  # human discharges via position sizing (Phase 8) — it is not read by any
  # signal or filter code from this phase forward.
  MAX_RISK_PCT = 0.005  # ACCOUNT-RISK BUDGET (human sizing), NOT a stop-distance rule

  ATR_STOP_PERIOD = 14  # Wilder ATR period for the stop distance
  ATR_STOP_MULTIPLE = 1.5  # k in stop = k * ATR; derived from cost-ratio constraint, frozen
  RR_FLOOR = 1.5  # reward_pct / risk_pct must be >= this; replaces the absolute band

  FEE_PCT = 0.0005  # Binance USDT-M VIP-0 taker fee per side (was 0.0004 — understated)
  SLIPPAGE_PCT = 0.0002  # assumed slippage per side (unchanged)
  FUNDING_PCT_PER_DAY = 0.0001  # frozen pessimistic placeholder; real ingestion is Option C
  COST_RATIO_CEILING = 0.10  # c = cost / risk_pct; asserted in tests/reports, not enforced at runtime
  ```
  Delete `MIN_REWARD_PCT`, `BREAKOUT_STOP_BUFFER_PCT`, `BREAKOUT_MAX_ENTRY_EXTENSION_PCT` entirely (grep confirms their only consumers are `setup.py`/`meanrev.py`/tests, all updated in this phase).
- **MIRROR**: Existing constant-grouping-by-phase-comment style (`config.py:26,87,93`).
- **IMPORTS**: None (same file).
- **GOTCHA**: `FUNDING_PCT_PER_DAY = 0.0001` is a placeholder frozen constant per PRD Open Question #3 ("frozen pessimistic constant in v1"); do not attempt to derive it precisely — document the assumption in the docstring/comment as shown.
- **VALIDATE**: `python -c "from trading_bot import config; print(config.ATR_STOP_MULTIPLE, config.RR_FLOOR, config.FEE_PCT)"` prints `1.5 1.5 0.0005`; `grep -rn "MIN_REWARD_PCT\|BREAKOUT_STOP_BUFFER_PCT\|BREAKOUT_MAX_ENTRY_EXTENSION_PCT" src/` returns nothing.

### Task 2: New `risk` package with pure ATR-stop / cost-ratio helpers
- **ACTION**: Create `src/trading_bot/risk/__init__.py` (empty package marker, mirrors every other subpackage) and `src/trading_bot/risk/atr_stop.py`.
- **IMPLEMENT**:
  ```python
  """
  ATR-scaled stop distance and cost-ratio helpers (Phase 2: honest risk model).

  Stop distance is a market-structure quantity (how far price plausibly moves
  against the setup before invalidating the thesis), governed by volatility —
  NOT an account-risk sizing rule. These are pure functions with no I/O so both
  setup.build_signal and meanrev.build_fade_signal can share one implementation
  instead of drifting apart.
  """

  def compute_atr_stop(entry: float, direction: str, atr_value: float, k: float) -> float:
      """Stop price = k * ATR away from entry, in the direction that invalidates the trade.

      Args:
          entry: Entry price.
          direction: "long" or "short".
          atr_value: Current ATR (already computed by the caller; NaN is the
              caller's responsibility to check before calling).
          k: ATR multiple (config.ATR_STOP_MULTIPLE).

      Returns:
          Stop price. Always on the losing side of entry.
      """
      distance = k * atr_value
      return entry - distance if direction == "long" else entry + distance


  def cost_ratio(risk_pct: float, fee_pct: float, slippage_pct: float) -> float:
      """c = round-trip cost / risk unit. Used to assert the strategy sits on
      the right side of the cost frontier (config.COST_RATIO_CEILING), never to
      gate an individual candidate.

      Args:
          risk_pct: |entry - stop| / entry.
          fee_pct: Per-side taker fee.
          slippage_pct: Per-side assumed slippage.

      Returns:
          cost / risk_pct. Undefined (returns float('inf')) if risk_pct <= 0.
      """
      if risk_pct <= 0:
          return float("inf")
      round_trip_cost = 2 * (fee_pct + slippage_pct)
      return round_trip_cost / risk_pct
  ```
- **MIRROR**: `indicators/wilder.py`'s style of pure, dependency-free functions with full docstrings (Args/Returns).
- **IMPORTS**: None needed beyond stdlib.
- **GOTCHA**: Do NOT import `config` inside `atr_stop.py` — keep it a pure-function module with no config coupling, so callers (setup.py, meanrev.py, engine.py, tests) explicitly pass `k`/`fee_pct`/`slippage_pct`, matching the existing optional-override pattern.
- **VALIDATE**: New file `tests/test_risk_atr_stop.py`:
  ```python
  from trading_bot.risk.atr_stop import compute_atr_stop, cost_ratio

  def test_long_stop_is_below_entry():
      assert compute_atr_stop(100.0, "long", atr_value=2.0, k=1.5) == 97.0

  def test_short_stop_is_above_entry():
      assert compute_atr_stop(100.0, "short", atr_value=2.0, k=1.5) == 103.0

  def test_cost_ratio_basic():
      # risk_pct=0.02, round-trip cost = 2*(0.0005+0.0002) = 0.0014 -> c = 0.07
      import math
      assert math.isclose(cost_ratio(0.02, 0.0005, 0.0002), 0.07)

  def test_cost_ratio_zero_risk_is_infinite():
      assert cost_ratio(0.0, 0.0005, 0.0002) == float("inf")
  ```
  `pytest tests/test_risk_atr_stop.py -v` — all pass.

### Task 3: Rewrite `build_signal` (breakout) to use ATR stop + R:R ratio floor
- **ACTION**: Edit `setup.py:72-186`.
- **IMPLEMENT**: Replace the signature and body:
  ```python
  def build_signal(
      symbol: str,
      candidate: PatternCandidate,
      event: BreakoutEvent,
      atr_value: float,
      *,
      atr_multiple: float | None = None,
      rr_floor: float | None = None,
  ) -> Signal | None:
      """
      Compute an ATR-scaled SL/TP for a triggered candidate and apply the R:R
      ratio floor.

      Stop sits atr_multiple * ATR away from entry (a volatility-derived,
      market-structure distance — see risk/atr_stop.py), replacing the old
      level-anchored buffer stop. Target is unchanged: the measured move
      (level +/- candidate.target_height). Setups are rejected when risk is
      zero or negative, reward is non-positive, or the reward:risk ratio falls
      below rr_floor. There is no longer an absolute risk-percentage cap nor
      an entry-extension cap — those were the adverse-selection mechanism
      identified in the market-research benchmark (rejects the trades whose
      stops are freakishly tight, which are exactly the trades noise destroys).

      Args:
          symbol: Trading pair symbol.
          candidate: The pattern that broke out.
          event: The 15m breakout event (entry reference).
          atr_value: Current ATR on the setup timeframe (caller computes; NaN
              bars must be filtered out by the caller before calling this).
          atr_multiple: Stop distance = atr_multiple * atr_value
              (default config.ATR_STOP_MULTIPLE).
          rr_floor: Minimum acceptable reward_pct/risk_pct
              (default config.RR_FLOOR).

      Returns:
          Signal if the setup passes, else None.
      """
      if atr_multiple is None:
          atr_multiple = config.ATR_STOP_MULTIPLE
      if rr_floor is None:
          rr_floor = config.RR_FLOOR

      entry = event.price
      if event.level <= 0 or entry <= 0:
          return None

      stop = compute_atr_stop(entry, candidate.direction, atr_value, atr_multiple)
      if candidate.direction == "long":
          target = event.level + candidate.target_height
          reward = target - entry
      else:
          target = event.level - candidate.target_height
          reward = entry - target

      risk = abs(entry - stop)
      if risk <= 0 or reward <= 0:
          return None

      risk_pct = risk / entry
      reward_pct = reward / entry
      rr = reward_pct / risk_pct

      def reject(reason: str, *args) -> None:
          logger.debug(
              "%s %s %s rejected: " + reason,
              symbol, candidate.kind, candidate.direction, *args,
          )

      if rr < rr_floor:
          reject("rr %.2f below floor %.2f", rr, rr_floor)
          return None

      return Signal(
          symbol=symbol, ts=event.ts, direction=candidate.direction,
          pattern=candidate.kind, entry=entry, stop=stop, target=target,
          risk_pct=risk_pct, reward_pct=reward_pct, rr=rr,
          volume_ratio=event.volume_ratio, volume_high=event.volume_high,
      )
  ```
  Add `from trading_bot.risk.atr_stop import compute_atr_stop` to imports.
- **MIRROR**: The `reject()` closure and docstring shape from the original (see Patterns to Mirror).
- **IMPORTS**: `from trading_bot.risk.atr_stop import compute_atr_stop` in `setup.py`.
- **GOTCHA**: `event.level` is still used for `target` computation (measured-move projection is unchanged — only the stop and filter change). Do not remove `event.level <= 0` guard. Also: `atr_value` is now a **required positional** parameter, not optional — every caller (`scan_breakout_signals`, the backtest engine, all tests) must be updated to pass it; there is no silent fallback because a missing ATR value must never produce a phantom stop.
- **VALIDATE**: `pytest tests/test_signals.py::TestBuildSignal -v` after Task 6 rewrites it.

### Task 4: Wire ATR into `scan_breakout_signals`
- **ACTION**: Edit `setup.py:245-293` (`scan_breakout_signals`).
- **IMPLEMENT**: Compute the ATR series on `df_1h` once, take the last valid value, pass it to every `build_signal` call:
  ```python
  from trading_bot.indicators.wilder import atr as wilder_atr
  ...
  def scan_breakout_signals(conn, symbol: str, now_ms: int) -> list[Signal]:
      df_1h = _load_df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME, now_ms)
      df_15m = _load_df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME, now_ms)
      if df_1h.empty or df_15m.empty:
          return []

      df_1h = df_1h.tail(config.PATTERN_LOOKBACK_BARS)
      trigger_bars = config.VOLUME_LOOKBACK + 1 + max(1, config.BREAKOUT_TRIGGER_LOOKBACK_BARS)
      df_15m = df_15m.tail(trigger_bars)
      ... # unchanged latest_close / staleness check

      atr_series = wilder_atr(df_1h, period=config.ATR_STOP_PERIOD)
      atr_value = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")

      pivots = find_pivots(df_1h)
      candidates = detect_patterns(df_1h, pivots)

      signals: list[Signal] = []
      if not (atr_value > 0):  # covers NaN (warmup) and zero/negative
          logger.warning("%s: ATR undefined or non-positive on %s bars; no signals", symbol, config.SIGNAL_PATTERN_TIMEFRAME)
          return []
      for candidate in candidates:
          event = check_breakout(df_15m, candidate, interval_ms=interval_15m)
          if event is None:
              continue
          signal = build_signal(symbol, candidate, event, atr_value)
          if signal is not None:
              signals.append(signal)

      return rank_signals(signals)
  ```
- **MIRROR**: The existing empty-guard-then-continue style already used for `df_1h.empty or df_15m.empty`.
- **IMPORTS**: `from trading_bot.indicators.wilder import atr as wilder_atr` (aliased to avoid shadowing any local `atr` variable name).
- **GOTCHA**: `PATTERN_LOOKBACK_BARS = 180` and `ATR_STOP_PERIOD = 14` — ATR needs `2*period-1`-ish bars to stabilize per `wilder_atr`'s own contract, well within 180, so no lookback change needed here. Do NOT slice `df_1h` to fewer bars before computing ATR — the tail-180 window is already generous.
- **VALIDATE**: Existing `TestCurrentSignals` integration test (rewritten in Task 6) exercises this path end-to-end.

### Task 5: Rewrite `build_fade_signal` (meanrev) with the same ATR/ratio semantics
- **ACTION**: Edit `meanrev.py:173-241` and `scan_fade_signals` (`meanrev.py:244-273`).
- **IMPLEMENT**: Fade's stop is **already** the excursion extreme (`candidate.stop_level`) — per the PRD Decisions Log this placement is "structurally correct" and is **kept as-is**, not replaced by an ATR-derived distance. This phase changes only the *filter*: swap the absolute band for the ratio floor.
  ```python
  def build_fade_signal(
      symbol: str,
      candidate: FadeCandidate,
      event: BreakoutEvent,
      *,
      rr_floor: float | None = None,
  ) -> Signal | None:
      """
      Compute SL/TP for a triggered fade and apply the R:R ratio floor.

      Stop stays at the excursion extreme (candidate.stop_level) — this
      placement is already volatility-structural (the extreme the fade thesis
      is invalidated by), unlike the breakout method's old level-anchored
      buffer stop. Only the filter changes: the fixed MAX_RISK_PCT/
      MIN_REWARD_PCT band is replaced by rr_floor, matching build_signal.

      Args:
          symbol: Trading pair symbol.
          candidate: The fade setup that re-crossed.
          event: The 15m re-cross event (entry reference).
          rr_floor: Minimum acceptable reward_pct/risk_pct
              (default config.RR_FLOOR).

      Returns:
          Signal (pattern="bollinger-fade") if the setup passes, else None.
      """
      if rr_floor is None:
          rr_floor = config.RR_FLOOR

      entry = event.price
      stop = candidate.stop_level
      target = candidate.target
      if entry <= 0:
          return None

      if candidate.direction == "long":
          risk = entry - stop
          reward = target - entry
      else:
          risk = stop - entry
          reward = entry - target

      if risk <= 0 or reward <= 0:
          return None

      risk_pct = risk / entry
      reward_pct = reward / entry
      rr = reward_pct / risk_pct
      if rr < rr_floor:
          return None

      return Signal(
          symbol=symbol, ts=event.ts, direction=candidate.direction,
          pattern=FADE_KIND, entry=entry, stop=stop, target=target,
          risk_pct=risk_pct, reward_pct=reward_pct, rr=rr,
          volume_ratio=event.volume_ratio, volume_high=event.volume_high,
      )
  ```
  `scan_fade_signals` needs NO ATR wiring (fade doesn't use ATR for its stop) — leave its body as-is except that `build_fade_signal`'s call site drops the removed kwargs (it already doesn't pass them, so this call site needs no edit).
- **MIRROR**: `build_signal`'s new shape (Task 3) — same `rr < rr_floor` semantics, same docstring structure, deliberately no `reject()` logger call here since the original `build_fade_signal` never had one either (keep behavior parity, don't add new logging as a drive-by).
- **IMPORTS**: None new (no ATR import needed in `meanrev.py`).
- **GOTCHA**: Do NOT copy the ATR-stop call from Task 3 into this function — that would silently move the fade stop off the excursion extreme, which the PRD explicitly says is already correct and should be re-tested (Phase 6), not changed (Phase 2). This is the one place where breakout and fade diverge.
- **VALIDATE**: `pytest tests/test_meanrev.py -v` after Task 6 rewrites its `MAX_RISK_PCT`/`MIN_REWARD_PCT` assertions.

### Task 6: Rewrite the retired-constant test assertions
- **ACTION**: Edit `tests/test_signals.py` (`TestBuildSignal` at 395-447, `test_trending_flag_breakout_produces_signal` at 523-550) and `tests/test_meanrev.py` (the two blocks at ~148-149, ~228-229).
- **IMPLEMENT**: For `test_signals.py`, replace `TestBuildSignal` with ATR-aware versions. Preserve test *intent* (long-setup-passes, entry-extension-no-longer-applicable, risk-band-rejection becomes rr-floor-rejection, reward-band-rejection becomes rr-floor-rejection from the other side, entry-beyond-target-rejected is unchanged):
  ```python
  class TestBuildSignal:
      def test_long_setup_passes_rr_floor(self):
          df = breakout_df(prev_close=99.8, last_close=100.3)
          candidate = make_candidate(level=100.0, height=5.0)  # generous reward
          event = check_breakout(df, candidate)
          signal = build_signal(SYMBOL, candidate, event, atr_value=1.0)
          assert signal is not None
          assert math.isclose(signal.stop, 100.3 - config.ATR_STOP_MULTIPLE * 1.0)
          assert signal.rr >= config.RR_FLOOR
          assert math.isclose(signal.rr, signal.reward_pct / signal.risk_pct)

      def test_rr_below_floor_rejected(self):
          df = breakout_df(prev_close=99.8, last_close=100.3)
          candidate = make_candidate(level=100.0, height=0.5)  # thin reward
          event = check_breakout(df, candidate)
          # A wide ATR stop dominates a thin measured-move reward: rr < floor.
          assert build_signal(SYMBOL, candidate, event, atr_value=1.0) is None

      def test_rr_floor_is_overridable(self):
          df = breakout_df(prev_close=99.8, last_close=100.3)
          candidate = make_candidate(level=100.0, height=0.5)
          event = check_breakout(df, candidate)
          assert build_signal(SYMBOL, candidate, event, atr_value=1.0, rr_floor=0.1) is not None

      def test_zero_atr_yields_zero_risk_rejected(self):
          df = breakout_df(prev_close=99.8, last_close=100.3)
          candidate = make_candidate(level=100.0, height=5.0)
          event = check_breakout(df, candidate)
          assert build_signal(SYMBOL, candidate, event, atr_value=0.0) is None

      def test_entry_beyond_target_rejected(self):
          from trading_bot.signals.breakout import BreakoutEvent
          candidate = make_candidate(level=100.0, height=0.2)
          event = BreakoutEvent(
              ts=START, price=100.4, level=100.0, direction="long",
              volume_ratio=1.0, volume_high=False,
          )
          assert build_signal(SYMBOL, candidate, event, atr_value=1.0) is None
  ```
  Update `test_trending_flag_breakout_produces_signal` (523-550): it currently asserts `s.stop == 109.5 * (1 - BREAKOUT_STOP_BUFFER_PCT)` and the retired band. Since this test seeds real candle data and doesn't control ATR directly, either (a) monkeypatch `setup_mod.wilder_atr` to return a fixed series, matching the existing `monkeypatch.setattr(setup_mod, "current_regime", ...)` pattern already used two tests above it, or (b) assert only `s.rr >= config.RR_FLOOR` and drop the exact-stop assertion (ATR is now data-derived, not a fixed offset from level). Prefer (a) for determinism — it matches the file's existing monkeypatch idiom exactly.
  For `test_meanrev.py`, replace the two `risk_pct <= MAX_RISK_PCT` / `reward_pct >= MIN_REWARD_PCT` assertions with `rr >= config.RR_FLOOR`.
- **MIRROR**: Existing `make_candidate()`/`breakout_df()` helpers already defined earlier in `test_signals.py` — reuse them, don't redefine.
- **IMPORTS**: No new imports (config already imported).
- **GOTCHA**: `test_entry_extended_past_level_rejected` (408-422) and `test_risk_above_band_rejected` (424-430) test *retired* mechanisms (`max_entry_extension_pct`, `stop_buffer_pct` no longer exist as parameters) — delete these two tests outright rather than adapting them; their intent (reject-on-adverse-selection) is now covered by `test_rr_below_floor_rejected`.
- **VALIDATE**: `pytest tests/test_signals.py tests/test_meanrev.py -v` — all pass, zero references to removed config constants remain (`grep -rn "MAX_RISK_PCT\|MIN_REWARD_PCT\|BREAKOUT_STOP_BUFFER_PCT\|BREAKOUT_MAX_ENTRY_EXTENSION_PCT" tests/` returns nothing beyond the retained account-risk-budget docstring reference, if any).

### Task 7: Thread ATR and funding cost through the backtest engine
- **ACTION**: Edit `engine.py:91-260` (`run_backtest`, `candidates_for`, `close_out`).
- **IMPLEMENT**:
  1. Compute the 1H ATR series once, alongside `df_1h`:
     ```python
     from trading_bot.indicators.wilder import atr as wilder_atr
     ...
     atr_1h = wilder_atr(df_1h, period=config.ATR_STOP_PERIOD)
     atr_close_1h = atr_1h.to_numpy()  # aligned index-for-index with close_1h
     ```
  2. In `candidates_for(h_idx)`, no change needed to candidate *generation* — ATR is looked up at signal-build time instead, using `h_idx` (already available in the closure) to index `atr_close_1h[h_idx]`.
  3. Where `build_signal`/`build_fade_signal` are called inside the main loop (around line 240-248), pass the ATR value:
     ```python
     atr_value = float(atr_close_1h[h_idx]) if h_idx < len(atr_close_1h) else float("nan")
     for method, cand in cands:
         if method == "breakout":
             event = check_breakout(window_15m, cand, lookback_bars=1, interval_ms=m15)
             sig = build_signal(symbol, cand, event, atr_value) if event and atr_value > 0 else None
         else:
             event = check_breakout(
                 window_15m, _to_trigger_candidate(cand), lookback_bars=1, interval_ms=m15
             )
             sig = build_fade_signal(symbol, cand, event) if event else None
         if sig is not None:
             bar_signals.append(sig)
     ```
  4. Add the funding term. `cost` is currently a fixed scalar computed once (`cost = 2 * (fee + slip)`); funding depends on holding duration, so it must move into `close_out` where `entry_ts`/`exit_ts` are both known:
     ```python
     def close_out(j: int, price: float, outcome: str) -> None:
         nonlocal open_trade
         s = open_trade["signal"]
         sign = 1.0 if s.direction == "long" else -1.0
         gross = sign * (price - s.entry) / s.entry
         hold_days = (int(ts_15m[j]) - s.ts) / 86_400_000.0
         funding_cost = funding_pct_per_day * hold_days
         trades.append(
             Trade(
                 symbol=symbol, regime=open_trade["regime"], pattern=s.pattern,
                 direction=s.direction, entry_ts=s.ts, entry=s.entry, stop=s.stop,
                 target=s.target, exit_ts=int(ts_15m[j]), exit_price=price,
                 outcome=outcome, pnl_pct=gross - cost - funding_cost,
                 volume_high=s.volume_high,
             )
         )
         open_trade = None
     ```
     where `funding_pct_per_day = config.FUNDING_PCT_PER_DAY if funding_pct_per_day is None else funding_pct_per_day` is resolved at the top of `run_backtest` alongside `fee`/`slip`, and `run_backtest`'s signature gains `funding_pct_per_day: float | None = None`.
- **MIRROR**: The existing `fee_pct`/`slippage_pct` optional-override resolution pattern at `engine.py:121-123`.
- **IMPORTS**: `from trading_bot.indicators.wilder import atr as wilder_atr` in `engine.py`.
- **GOTCHA**: `atr_close_1h` and `close_1h` must stay index-aligned — both are derived from the same `df_1h`, so no re-indexing is needed, but do NOT slice one without the other. Also: the `candidates_for` cache (`cand_cache`) keys on `h_idx` and stores `(regime, candidates)` — ATR lookup happens *outside* that cache (at call time in the main loop), so it does not need to be memoized there; keep the cache untouched.
- **VALIDATE**: `pytest tests/test_backtest.py -v`.

### Task 8: Backtest tests — funding term and cost-ratio assertions
- **ACTION**: Edit `tests/test_backtest.py`.
- **IMPLEMENT**: Add tests verifying (a) a trade held longer accrues more funding cost than an identical trade held for one bar, and (b) the median realized stop distance across a synthetic multi-trade run is `>= 1.0 * ATR` when `ATR_STOP_MULTIPLE >= 1.0` (proves R1 landed, per the PRD's own success signal for this phase). Use the file's existing fixture-building helpers (`grep -n "^def \|^class " tests/test_backtest.py` to find them before writing — do not invent new fixture shapes if one already exists for seeding a symbol's OHLCV).
- **MIRROR**: Whatever synthetic-data helper `test_backtest.py` already uses (mirrors `seed_candles`/`make_df` conventions from `test_signals.py` if no local equivalent exists).
- **IMPORTS**: `from trading_bot.risk.atr_stop import cost_ratio` if asserting `c <= config.COST_RATIO_CEILING` directly in a test.
- **GOTCHA**: This is an assertion *about the risk model's mechanics*, not a claim that the strategy is profitable — do not add an assertion like "expectancy > 0" here; that verdict belongs to Phase 7's gate, not this phase's unit tests.
- **VALIDATE**: `pytest tests/test_backtest.py -v`.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| `compute_atr_stop` long | entry=100, atr=2, k=1.5 | 97.0 | — |
| `compute_atr_stop` short | entry=100, atr=2, k=1.5 | 103.0 | — |
| `cost_ratio` basic | risk_pct=0.02, fee=0.0005, slip=0.0002 | 0.07 | — |
| `cost_ratio` zero risk | risk_pct=0.0 | inf | Edge: division by zero |
| `build_signal` rr floor pass | wide reward, small ATR | Signal with rr >= RR_FLOOR | — |
| `build_signal` rr floor fail | thin reward, wide ATR | None | Edge: exactly at floor (rr == floor) should PASS (>=, not >) |
| `build_signal` zero ATR | atr_value=0.0 | None (risk<=0) | Edge case |
| `build_fade_signal` rr floor | same pattern as breakout | rr-based accept/reject | — |
| Funding cost accrual | 1-bar hold vs 96-bar hold | 96-bar trade's pnl_pct is more negative by the extra holding-days funding term | — |
| Backtest median stop/ATR ratio | synthetic multi-trade fixture | median stop distance >= 1.0 * ATR | Proves R1 landed (phase success signal) |

### Edge Cases Checklist
- [x] rr exactly at floor (boundary: `>=` not `>`)
- [x] ATR = 0 or NaN at signal-build time (never produce a stop)
- [x] Zero/negative risk or reward (existing guard, preserved)
- [x] Long vs short direction symmetry for `compute_atr_stop`
- [x] Funding cost scaling with holding duration (short hold vs full 96-bar hold)
- [ ] Concurrent access — N/A (pure functions / single-threaded backtest)
- [ ] Network failure — N/A (no network calls in this phase)
- [ ] Permission denied — N/A

---

## Validation Commands

### Static Analysis
```bash
python -m py_compile src/trading_bot/risk/atr_stop.py src/trading_bot/signals/setup.py src/trading_bot/signals/meanrev.py src/trading_bot/backtest/engine.py src/trading_bot/config.py
```
EXPECT: Zero syntax errors. (No type-checker such as mypy is configured in this project — `pyproject.toml` lists only ccxt/pandas/pandas-ta/apscheduler/pytest/python-dotenv as deps; skip a type-check step that doesn't exist rather than inventing one.)

### Unit Tests
```bash
pytest tests/test_risk_atr_stop.py tests/test_signals.py tests/test_meanrev.py tests/test_backtest.py -v
```
EXPECT: All tests pass; zero references to `MAX_RISK_PCT`/`MIN_REWARD_PCT`/`BREAKOUT_STOP_BUFFER_PCT`/`BREAKOUT_MAX_ENTRY_EXTENSION_PCT` remain as stop/filter logic (grep check below).

### Full Test Suite
```bash
pytest tests/ -v
```
EXPECT: No regressions in `test_classifier.py`, `test_wilder.py`, `test_storage.py`, `test_backfill.py`, `test_poller.py`, `test_binance_client.py`, `test_cli.py` (none of these touch signal/risk/cost logic).

### Grep Verification (retired-constant leak check)
```bash
grep -rn "MIN_REWARD_PCT\|BREAKOUT_STOP_BUFFER_PCT\|BREAKOUT_MAX_ENTRY_EXTENSION_PCT" src/ tests/
```
EXPECT: No matches.

### Manual Validation (the phase's own success signal, per the PRD)
```bash
python -m trading_bot.cli backtest --symbol BTCUSDT
python -m trading_bot.cli backtest --symbol ETHUSDT
python -m trading_bot.cli backtest --symbol SOLUSDT
```
- [ ] For each symbol, compute median realized `stop_pct` from the trade list and confirm it is `>= 1.0 * ATR_STOP_PERIOD-bar ATR` on the setup timeframe (this is the phase's stated success signal — "Median realized stop distance ≥ 1.0 × ATR(setup TF) on all 3 symbols").
- [ ] Compute `c = cost_ratio(median_risk_pct, config.FEE_PCT, config.SLIPPAGE_PCT)` per symbol and confirm it is reported (not necessarily `<= 0.10` yet — full tier shift to 4H setup bars is Phase 4; this phase measures where we stand today at the 1H setup tier per PRD Open Question #2).
- [ ] Re-run the *existing* (unrepaired) walk-forward/backtest gate on both methods and note whether expectancy improves versus the benchmark's recorded baseline (BTC −0.127%, ETH −0.092%, SOL −0.110% per trade) — this is the cheap falsification test the PRD calls out; record the result in a trial-log note even though the formal trial-log mechanism is Phase 9.

---

## Acceptance Criteria
- [ ] All 8 tasks completed
- [ ] All validation commands pass
- [ ] Tests written and passing (new `test_risk_atr_stop.py`; rewritten `TestBuildSignal` and fade equivalents)
- [ ] No type errors (no type checker configured — N/A)
- [ ] No lint errors (no linter configured in `pyproject.toml` — N/A; keep style consistent with surrounding code by inspection)
- [ ] Matches UX design — N/A, internal change

## Completion Checklist
- [ ] Code follows discovered patterns (optional-override kwargs, `reject()` logging, frozen dataclasses)
- [ ] Error handling matches codebase style (return `None` on filter failure, never raise)
- [ ] Logging follows codebase conventions (`logging.getLogger("trading_bot")`, DEBUG for rejections, WARNING for undefined-ATR/staleness)
- [ ] Tests follow test patterns (class-per-function, `make_df`/`make_candidate`/`breakout_df` helpers reused, not reinvented)
- [ ] No hardcoded values — `k`, `RR_FLOOR`, fee/slip/funding all live in `config.py`
- [ ] Documentation updated — docstrings on `build_signal`/`build_fade_signal` rewritten to describe the new mechanics (done inline in Tasks 3/5)
- [ ] No unnecessary scope additions — Donchian, tier shift, Sharpe metrics, sizing are explicitly NOT touched
- [ ] Self-contained — no questions needed during implementation

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ATR computed on 1H bars (current setup TF) may not satisfy `c ≤ 0.10` until Phase 4's tier shift to 4H — this phase cannot fully close the cost-ratio gate alone | High (flagged in PRD Open Question #2) | Medium | Explicitly scope this phase's success signal as "median stop ≥ 1.0×ATR and c is *measured and reported*" per symbol, not "c ≤ 0.10 achieved" — that full closure is Phase 4's job. Documented in the manual validation section above |
| `FUNDING_PCT_PER_DAY` is a guessed placeholder, not measured | Medium (flagged in PRD Open Question #3) | Low-Medium | Frozen, documented as an assumption in the config comment (Task 1); real ingestion is out of scope until Option C |
| Rewriting `test_signals.py`'s `TestBuildSignal` might silently drop a real regression-catching assertion (e.g., the entry-extension check was catching a genuine adverse-selection bug class) | Low-Medium (flagged in PRD's own technical risk table) | Medium | Task 6 explicitly ports test *intent* (adverse-selection rejection is preserved via `test_rr_below_floor_rejected`) rather than deleting coverage silently |
| `atr_close_1h` misalignment with `close_1h` in `engine.py` if a future edit slices one without the other | Low | High (silent lookahead or index bug) | Both arrays are derived from the same `df_1h` in the same function scope with no independent slicing — flagged explicitly in Task 7's GOTCHA |
| Fade's stop-placement is left untouched, which could look like an oversight to a future reader | Low | Low | Task 5's docstring and GOTCHA explicitly state this is deliberate (PRD Decisions Log: fade's excursion-extreme stop is already structurally correct; re-qualification is Phase 6, not this phase) |

## Notes
- This phase is deliberately narrow: it changes the risk/cost *specification*, not the signal *methods* (breakout/fade geometry is untouched) or the *timeframe* (still 1H setup / 15m trigger until Phase 4). That isolation is the point — per the PRD's own sequencing rationale, running the existing methods under the new risk model in isolation is what tells us whether the risk model was the whole problem, before Phase 5 builds a new engine on top of an unproven risk fix.
- `uncommitted work in flight` warning from the PRD (branch `fix/phase3-breakout-detection` touching `engine.py`, `config.py`, `signals/*`, `tests/test_signals.py`) — check `git status` / `git branch` before starting Task 1; if that branch's changes are still present in the working tree, reconcile or abandon them first (per the PRD: "That branch's premise is superseded by this PRD — resolve or abandon it before Phase 2 lands").
