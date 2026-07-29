# Plan: Strategy Pipeline — Thin Slice (v0.3.0 Phase 4)

## Summary

Run the pivot guide's **Strategy Steps 1–5** end-to-end as composed plug-ins on the Phase 3
framework: a MACD indicator ported from `scripts/bruteforce/indicators.py`, a MACD-cross
Detector, two Confirmations (volume-on-breakout and MACD), a measured-move PositionPolicy,
and the **≥1:2 R:R-after-costs** Filter — wired into one serialized strategy graph that
backtests on 3+ symbols through the single `run_graph_backtest` seam, with every taken
position carrying its R:R justification on the `Trade`.

Two things in this phase are behavior changes with a measurable cost, not additions:
turning `Signal.volume_ratio` from a computed-but-unused number into a **hard gate**
(KNOWN-LIMITATIONS §0c: "volume is computed on every signal but gates nothing"), and
raising the reward:risk requirement from `RR_FLOOR = 1.5` **gross** to
`RR_TARGET_MIN = 2.0` **net of costs**. Both are measured before and after, and the
measurement is expected to be uncomfortable — see [The R:R prediction](#the-rr-prediction-read-this-before-task-10).

## User Story

As the bot's sole operator, I want to compose a pattern → confirmation → position →
R:R-filter pipeline out of registered plug-ins and backtest it with one command, so that
I can change the *entry and feature side* of the strategy — the axis KNOWN-LIMITATIONS §0c
records was never explored — without editing the engine, and so that every trade the system
takes carries a machine-readable answer to "why did it trade".

## Problem → Solution

**Current**: the entry is a hardcoded four-condition Donchian/ADX switch dispatched by
regime (`engine.py:335-349`), whose measured gross edge is 15 basis points per trade —
"noise; the *entry* carries almost no predictive content" (KNOWN-LIMITATIONS §0). Volume is
computed on every event and gates nothing (`breakout.py:19-23` states this as a design
decision). Reward:risk is screened on the **dimensionless gross** ratio for breakouts
(`setup.py:150-156`), which the codebase already knows is blind to absolute cost
(`risk/atr_stop.py:49-58`). No MACD, RSI, or any oscillator exists in `src/trading_bot`.

**Desired**: a composed graph whose entry is `Detector → Confirmation* → PositionPolicy →
Filter`, where each stage is a registered plug-in swappable without engine edits; volume and
MACD are real gates whose cost in trades and expectancy is measured; and the R:R screen is
`risk.atr_stop.net_rr(...) >= config.RR_TARGET_MIN` — cost-adjusted, with a pre-registered
decision rule for what to do when it rejects almost everything.

## Metadata

- **Complexity**: **Large** — 8 new source modules, 5 new test modules, 3 shared files
  extended (`config.py`, `cli.py`, `engine.py` `Trade`), 1 authorized cross-ownership edit
  (`framework/execute.py`), 2 committed graph artifacts. ~900–1100 net new lines including
  tests. **No new external dependency** (hand-rolled MACD, per contract §1).
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md`
- **PRD Phase**: Phase 4 — Strategy pipeline (thin slice)
- **Binding contract**: `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md`
  (§3 contracts, §5 seam + `Trade` append rule, §7 reserved names, §8 test files, §12 DoD)
- **Depends on**: Phase 3 (plug-in framework core) — **and specifically on Phase 3's parity
  test (`tests/test_framework_parity.py`) being GREEN**. Contract §5: "Phase 4 onward may
  only extend from a green parity test."
- **Gates**: Phases 5, 6 and 8.
- **Estimated Files**: 20 (13 create, 7 update)
- **Test baseline**: **286 tests collected** (`.venv/bin/python -m pytest --collect-only -q`,
  verified 2026-07-27, pytest 9.1.1). Phase 3 will have added its own; this plan's floor is
  "286 pre-existing + every test Phase 3 added, all still green".

---

## UX Design

Internal change, but it has an operator-visible surface: a new CLI subcommand.

### Before
```
$ python -m trading_bot.cli backtest --symbol BTCUSDT
BTCUSDT:
  trades=61  win_rate=32.79%  expectancy=0.0412%  ...
# Entry logic: hardcoded regime switch inside engine.py (engine.py:335-349).
# Volume: computed, printed as "Vol×" (cli.py:345-346), gates nothing.
# R:R: gross ratio >= 1.5, and no record of it survives on the Trade.
# To change the entry: edit engine.py.
```

### After
```
$ python -m trading_bot.cli graph-backtest \
      --graph data/strategies/thin-slice.strategy.json --audit

graph: thin-slice  schema=<P3 SCHEMA_VERSION>  nodes=7
  data.ohlcv | detector.donchian-breakout, detector.macd-cross
  confirmation.volume-breakout, confirmation.macd
  policy.measured-move | filter.rr-after-costs (RR_TARGET_MIN=2.0)

BTCUSDT:
  trades=N  win_rate=..  expectancy=..  profit_factor=..  max_dd=..
  equity: sharpe=..  sortino=..  max_dd=..  ann_return=..
  audit (every taken position, with its R:R justification):
    ts=1690848000000 long donchian-breakout entry=29812.5 stop=.. target=..
      risk=1.97% reward=4.51% net_rr=2.14 planned_rr=2.14
      confirmations=confirmation.macd,confirmation.volume-breakout
  rejections: confirmation.volume-breakout (ratio 1.02 < 1.50);
              filter.rr-after-costs (net_rr 1.39 < 2.00)

# To change the entry: register a plug-in. Engine untouched.
```

### Interaction Changes

| Touchpoint | Before | After | Notes |
|---|---|---|---|
| Entry composition | edit `engine.py` | edit a `.strategy.json` node list | PRD success metric: "zero engine-core edits" |
| Volume | printed only (`cli.py:345-346`) | **hard gate**, reason logged on rejection | KNOWN-LIMITATIONS §0c closed |
| MACD | does not exist | Detector **and** Confirmation | KNOWN-LIMITATIONS §0c closed |
| R:R screen | gross ≥ 1.5, invisible after the fact | **net ≥ 2.0**, recorded as `Trade.planned_rr` | PRD Success Metric row 4 |
| "Why did it trade?" | unanswerable per trade | `Trade.confirmations` + `--audit` | PRD's stated advantage of evolution over deep RL |
| CLI | `backtest`, `walkforward` | + `graph-backtest` | contract §7 reserved name |

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md` | §3, §5, §7, §8, §12 | BINDING. Contract shapes, the one seam, reserved names, test files, DoD |
| P0 | `src/trading_bot/risk/atr_stop.py` | 1-101 (all) | `net_rr` (46-76) is the Filter's whole implementation; `round_trip_cost_pct` (37-43); `cost_ratio` (79-101) and its funding caveat (82-88) |
| P0 | `src/trading_bot/signals/setup.py` | 40-171 | `Signal` dataclass (40-70) = field-shape template for `PositionPlan` consumption; `build_signal` (73-171) is the reference for the whole policy+filter body: NaN guard (131-136), measured-move target (139-144), rr screen (150-156) |
| P0 | `scripts/bruteforce/indicators.py` | 1-19, 79-82, 160-172 | The donor. Causality rules (1-19), `ema` (79-82) — **the seeding convention**, `macd` (160-172) |
| P0 | `src/trading_bot/indicators/wilder.py` | 1-18, 49-105, 108-122 | Module-docstring convention (1-18), `wilder_smooth`'s **SMA seed** (92-94) which MACD deliberately does NOT use, `atr` (108-122) |
| P0 | `src/trading_bot/config.py` | 32-49, 74-82, 105-151, 153-183 | Tier constants (32,45,46), the volume block being activated (74-76), warmup-derivation style (40, 139), frozen cost/risk block + the k derivation record (162-174) |
| P0 | `src/trading_bot/backtest/engine.py` | 108-133, 225-263, 354-378 | `Trade` (108-133) — the append site; cost resolution (259-263); `close_out` (354-378) — where `pnl_pct` and the new fields meet |
| P0 | `src/trading_bot/backtest/walkforward.py` | 53-68 | **The measured fact that drives this phase's biggest risk**: min planned R:R ever taken = 1.56; 100% of trades cleared 1.25 and 1.5 |
| P1 | `src/trading_bot/signals/breakout.py` | 19-23, 38-58, 131-138 | The volume machinery being activated: the "never a hard block" docstring being reversed (19-23), `BreakoutEvent` (38-58), the ratio computation and its NaN case (131-138) |
| P1 | `src/trading_bot/signals/donchian.py` | 1-45, 65-140 | The reference detector. Module docstring's Stated Assumptions A1/A2 (25-37) is the tier-choice argument this plan reuses; `target_height = channel width` (116, 133); `end_ts` no-lookahead convention (135-138) |
| P1 | `src/trading_bot/signals/patterns.py` | 50-76 | `PATTERN_KINDS` naming (50-55) and `PatternCandidate` (58-76) — the shape `DetectedEvent` is field-compatible with (contract §3) |
| P1 | `src/trading_bot/indicators/donchian.py` | 25-45 | `donchian(df, period=...)` — reused for the macd-cross detector's `target_height` |
| P1 | `src/trading_bot/cli.py` | 112-126, 199-209, 356-393 | Subparser shape, dispatch block, `_fmt` / `_print_metrics` / `_backtest_command` (incl. its "0 always" exit rule at 374-377) |
| P1 | `tests/test_backtest.py` | 1-57 | Tier-derived constants (24-33), autouse cache fixture (36-47), `make_trade` keyword construction (50-57) |
| P1 | `tests/test_wilder.py` | 325-358 | `TestHandComputedValues` — the hand-computed-numeric-test pattern MACD must mirror |
| P2 | `.claude/PRPs/reports/KNOWN-LIMITATIONS.md` | §0, §0c, §3, §9 | Why the entry is the target (§0), what was never searched (§0c), why numbers must be measured (§3), trial-log discipline (§9) |
| P2 | `.claude/pivot-guide.md` | 31-35 | Strategy Steps 1–5 verbatim — the scope of this phase |
| P2 | `scripts/bruteforce/registry.py` | 60-120 | The registry lessons carried into Phase 3: mandatory `rationale` (70-73), `combo_count` |
| P2 | `tests/test_signals.py` | 318-335 | `make_candidate()` / `breakout_df()` helpers to mirror (do not reinvent) |
| P2 | `.claude/PRPs/plans/v0.2.0/phase2-honest-cost-and-risk-model.plan.md` | all | The R:R / cost mechanics this phase extends, and the FORMAT bar |

## External Documentation

**No external research needed.** MACD (Gerald Appel, 1979) is already implemented, working
and causality-checked, at `scripts/bruteforce/indicators.py:160-172` on top of
`ema` at `:79-82`. Porting it with attribution is contract §0b's explicit rule: "do not
re-derive an indicator or detector that exists in `scripts/bruteforce/`." The R:R algebra in
this plan is derived from `risk/atr_stop.net_rr`'s own definition, not from literature.

---

## The R:R prediction (read this before Task 10)

This is the sharpest change in the phase and it is **stricter than it looks**. It is stated
here, up front, with its algebra and its pre-registered decision rule, so that no one
"discovers" the problem mid-implementation and softens the filter to make progress.

### The algebra

`risk/atr_stop.net_rr(reward_pct, risk_pct, fee, slip) = (reward - cost) / (risk + cost)`
where `cost = 2*(fee + slip) = 0.0014` at config defaults (`config.py:180-181`).

Requiring `net_rr >= X` is equivalent to requiring, in **gross** terms:

```
reward_pct >= X*risk_pct + (X+1)*cost
  divide by risk_pct, and let c = cost / risk_pct  (= risk.atr_stop.cost_ratio)
gross_rr >= X + (X + 1) * c
```

So a **net** floor of 2.0 is a **gross** floor of `2 + 3c`. Using the median `risk_pct`
values already measured and recorded in `config.py:166-168` (BTC 1.968%, ETH 2.679%,
SOL 3.752%, with their `c` of 0.0711 / 0.0523 / 0.0373):

| Symbol | median `risk_pct` | `c` | gross R:R needed for **net 2.0** | gross R:R needed for net 1.5 |
|---|---|---|---|---|
| BTCUSDT | 1.968% | 0.0711 | **2.213** | 1.678 |
| ETHUSDT | 2.679% | 0.0523 | **2.157** | 1.631 |
| SOLUSDT | 3.752% | 0.0373 | **2.112** | 1.593 |

And the fact that matters, from `walkforward.py:53-68`: **the minimum planned R:R across
every trade the engine ever took was 1.56**, and 100% of trades cleared both 1.25 and 1.5.
A trade sitting at gross 1.56 has `net_rr` of **1.39 / 1.43 / 1.47** (BTC/ETH/SOL) — it
clears neither 2.0 nor even 1.5 *net*. The 1.5-gross floor was measured non-binding; a
2.0-**net** floor is a genuinely new, much higher bar.

### Prediction

**A 2.0 net floor may reject nearly every plan the thin slice produces.** This is predicted
here, before measurement, so that a near-zero survivor count reads as a *confirmed
prediction* rather than a surprise to be engineered away.

### The measurement, before any tuning (Task 10 / Task 16)

`filter.rr-after-costs` supports a **measure-only** run: `graph-backtest --rr-report`
evaluates the Filter on every `PositionPlan` that reaches it and prints, per symbol and
pooled over the full stored span:

- `n_plans` — plans reaching the filter
- `n_pass` at each of `{2.0 (RR_TARGET_MIN), 1.75, 1.5, 1.25, 1.0}` net thresholds
- the deciles of the net-R:R distribution, and its min / median / max
- the same for **gross** R:R, so the gross-vs-net gap is visible, not asserted
- `survival_rate = n_pass(2.0) / n_plans`

Every number in the phase report comes from this one command (contract §12.5). The
additional thresholds are printed **for diagnosis only** — printing 1.5 is not permission
to use 1.5.

### Pre-registered decision rule

Let `N2 = n_pass(2.0)` pooled across the 3 production symbols over the full stored span.

| Branch | Action |
|---|---|
| `survival_rate >= 0.20` **and** `N2 >= 30` | Proceed. `RR_TARGET_MIN` stays 2.0. Phase 4 succeeds as specified. |
| `0 < N2 < 30` | **`RR_TARGET_MIN` stays 2.0.** Report: "the ≥1:2-after-costs requirement is satisfiable but starves the sample." The remedy is *more candidates* — Phase 2's symbol breadth (20 stored, 3 in production, contract §0a) and Phase 8's detector breadth — not a lower floor. Phase 4 is **still done**: the pipeline works; the sample does not clear `WF_MIN_TRADES = 30`. Say so in the report. |
| `N2 == 0` | **`RR_TARGET_MIN` stays 2.0.** Report as a **finding**: "the PRD's ≥1:2-after-costs requirement is unsatisfiable by the thin slice's TP/SL geometry (target = channel width, stop = 1.5·ATR)." Escalate to the user with exactly three non-weakening remedies: (a) detectors with a structurally larger `target_height` (Phase 8), (b) a tighter stop with a *structural* basis, not a fitted one, (c) accept the requirement falsifies this strategy family — which the PRD's honesty clause explicitly permits. |

**Forbidden under every branch, without an explicit user decision logged as a consumed
degree of freedom:** lowering `RR_TARGET_MIN`; switching the Filter from `net_rr` to gross
R:R; dropping funding or slippage from the cost term; modifying `RR_FLOOR` (contract §7:
"Do not modify existing constants"); or adding an `or` escape clause to the Filter.

This mirrors the discipline `config.py:162-174` already records for `k`: **the filter is
derived from costs, not fitted to returns.** A run that produces zero trades is a result
(`cli.py:374-377` already codifies this: "a backtest with zero trades is a result, not an
error").

---

## Patterns to Mirror

Every snippet below is copied from the live tree. Follow them exactly.

### NAMING_CONVENTION — config constants, grouped under a phase banner
```python
# SOURCE: config.py:74-76
# Breakout volume confirmation (graded confidence input, never a hard block).
VOLUME_LOOKBACK = 20  # trigger-timeframe bars in the rolling volume average
VOLUME_HIGH_RATIO = 1.5  # trigger volume >= ratio * average => "notably high"
```
```python
# SOURCE: config.py:139 — warmup counts are DERIVED, never magic numbers
# Warmup: the channel needs PERIOD prior bars (trailing, current bar excluded)
# and ADX(14) needs 2*14-1 = 27 bars. 55 dominates. On 4H bars that is ~9.2 days.
DONCHIAN_MIN_BARS = max(DONCHIAN_TREND_PERIOD + 1, 2 * ADX_PERIOD - 1)
```
```python
# SOURCE: config.py:40 — same rule, additive rather than max()
REGIME_MIN_BARS = 2 * ADX_PERIOD - 1 + ATR_PERCENTILE_WINDOW
```

### NAMING_CONVENTION — plug-in registry names are lowercase-hyphen
```python
# SOURCE: signals/patterns.py:50-55
PATTERN_KINDS = ("head-and-shoulders", "inverse-head-and-shoulders", "triangle", "flag")
# SOURCE: signals/donchian.py:62
DONCHIAN_KIND = "donchian-breakout"
```
This phase's names: `macd-cross`, `volume-breakout`, `macd`, `measured-move`,
`rr-after-costs`. Module names stay snake_case (`macd_cross.py`, `volume_breakout.py`,
`measured_move.py`, `rr_after_costs.py`). Never camelCase (contract §3).

### INDICATOR_MODULE_SHAPE — hand-rolled, dependency-free, convention stated in the docstring
```python
# SOURCE: indicators/wilder.py:1-18 (abridged) — the docstring contract to mirror
"""
Wilder's indicators: ATR, DI, ADX — pure pandas, no external dependencies.
...
Wilder smoothing with period N uses the recursive formula:
    smoothed_value = (prev_smoothed * (N - 1) + current_value) / N
The first smoothed value is the simple average of the first N values; subsequent
values follow the recursive formula.
...
All inputs are DataFrames with columns: open, high, low, close, volume (indexed by
epoch-ms int timestamp). All outputs are pd.Series indexed identically, with leading
NaNs preserved (never filled).
"""
```
`indicators/macd.py` must state its own seeding convention just as explicitly — because it
is a **different** convention from `wilder_smooth`'s (`wilder.py:92-94` seeds with the simple
average). See Task 2's GOTCHA.

### THE DONOR — the exact code being ported, with its rationale
```python
# SOURCE: scripts/bruteforce/indicators.py:79-82
def ema(s: pd.Series, period: int) -> pd.Series:
    """Exponential MA. ``adjust=False`` so the value at bar i depends only on
    bars <= i (the adjusted form renormalises using the whole series)."""
    return s.ewm(span=period, adjust=False, min_periods=period).mean()
```
```python
# SOURCE: scripts/bruteforce/indicators.py:160-172
def macd(
    s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD. Returns ``macd``, ``signal``, ``hist``, each normalised by price.

    Normalising makes thresholds portable across symbols; a raw MACD threshold
    that works on BTC is meaningless on DOGE.
    """
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame(
        {"macd": line / s, "signal": sig / s, "hist": (line - sig) / s}
    )
```
```python
# SOURCE: scripts/bruteforce/indicators.py:1-12 (the causality rules the port inherits)
# 1. **Trailing windows only.** No ``center=True``, no ``shift(-n)``, no
#    full-sample statistic... Rolling statistics use ``min_periods=window`` so
#    the warmup region stays ``NaN``...
# 2. **Never back-fill.** ``NaN`` means "not knowable yet".
```

### DETECTOR_SHAPE — level + target_height, derived not fitted, no-lookahead `end_ts`
```python
# SOURCE: signals/donchian.py:112-139
    if pd.isna(upper) or pd.isna(lower) or pd.isna(mid) or pd.isna(adx_now):
        return []  # warmup
    if adx_now < adx_min:
        return []  # trend not confirmed on the setup tier
    width = upper - lower
    if width <= 0:
        return []  # degenerate channel
    ...
    interval = int(df.index.to_numpy()[i]) - int(df.index.to_numpy()[i - 1])
    return [
        PatternCandidate(
            kind=DONCHIAN_KIND,
            direction=direction,
            breakout_level=level,
            target_height=width,
            start_ts=int(df.index.to_numpy()[max(0, i - entry_period)]),
            # end_ts is the setup bar's CLOSE time, so check_breakout's
            # `ts < candidate.end_ts` skip guarantees the trigger bar opens
            # at or after the setup bar closed — no intra-bar lookahead.
            end_ts=int(df.index.to_numpy()[i]) + interval,
        )
    ]
```

### NAN_GUARD — the only correct way to reject an undefined indicator value
```python
# SOURCE: signals/setup.py:131-136
    # NaN comparisons are always False, so `atr_value <= 0` alone would let a
    # NaN ATR (Wilder warmup) silently produce a stop = nan Signal instead of
    # being rejected. `not (atr_value > 0)` catches NaN, zero, and negative.
    if not (atr_value > 0):
        reject("atr_value %s is undefined or non-positive", atr_value)
        return None
```
Every plug-in in this phase that reads an indicator uses `not (x > 0)` / explicit
`math.isnan` — never `x <= 0` — and **fails closed** on NaN.

### POLICY_BODY — measured move, then the ratio, exactly as build_signal does it
```python
# SOURCE: signals/setup.py:138-156
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

    if rr < rr_floor:
        reject("rr %.2f below floor %.2f", rr, rr_floor)
        return None
```

### FILTER_BODY — the cost-adjusted ratio, already written and tested
```python
# SOURCE: risk/atr_stop.py:72-76
    cost = round_trip_cost_pct(fee_pct, slippage_pct)
    net_risk = risk_pct + cost
    if net_risk <= 0:
        return float("-inf")
    return (reward_pct - cost) / net_risk
```
```python
# SOURCE: signals/meanrev.py:233 — the existing net-R:R gate; the Filter mirrors this call
    if net_rr(reward_pct, risk_pct, config.FEE_PCT, config.SLIPPAGE_PCT) < rr_floor:
```

### OPTIONAL_OVERRIDE_PARAMS — keyword-only `| None = None` falling back to config
```python
# SOURCE: signals/setup.py:73-116 (abridged)
def build_signal(
    symbol: str,
    candidate: PatternCandidate,
    event: BreakoutEvent,
    atr_value: float,
    *,
    atr_multiple: float | None = None,
    rr_floor: float | None = None,
) -> Signal | None:
    if atr_multiple is None:
        atr_multiple = config.ATR_STOP_MULTIPLE
    if rr_floor is None:
        rr_floor = config.RR_FLOOR
```
Every plug-in's `**params` follows this: a declared `ParamSpec` default equal to the config
constant, resolved once at the top of the function.

### ERROR_HANDLING / REJECTION_LOGGING — DEBUG-level closure, return None, never raise
```python
# SOURCE: signals/setup.py:118-125
    def reject(reason: str, *args) -> None:
        logger.debug(
            "%s %s %s rejected: " + reason,
            symbol,
            candidate.kind,
            candidate.direction,
            *args,
        )
```
A candidate failing a gate is normal, high-frequency control flow — **not** an error. But
note the difference from v0.2.0: a Confirmation/Filter now returns a *verdict object* with a
`reason` string (contract §3), so the reason is machine-readable as well as logged.

### LOGGING_PATTERN
```python
# SOURCE: signals/setup.py:37 (identical in breakout.py:35, donchian.py:60, engine.py:87)
logger = logging.getLogger("trading_bot")
```

### FROZEN_DATACLASS_SHAPE — frozen, with an exhaustive `Attributes:` docstring
```python
# SOURCE: signals/breakout.py:38-58
@dataclass(frozen=True)
class BreakoutEvent:
    """A confirmed breakout of a pattern's level on the trigger timeframe.

    Attributes:
        ts: Epoch-ms of the trigger bar (the latest closed one).
        price: Trigger bar close — the signal's entry reference.
        ...
        volume_ratio: Trigger volume / rolling mean of the prior
            VOLUME_LOOKBACK bars; NaN if the average is undefined.
        volume_high: True when volume_ratio >= VOLUME_HIGH_RATIO.
    """
```
Same shape at `signals/setup.py:40-70` (`Signal`) and `backtest/engine.py:108-133` (`Trade`).

### TRADE_APPEND_SITE — the 13 existing fields, in order, that must not move
```python
# SOURCE: backtest/engine.py:108-133 (docstring elided; it is part of the pattern)
@dataclass(frozen=True)
class Trade:
    symbol: str; regime: str; pattern: str; direction: str
    entry_ts: int; entry: float; stop: float; target: float
    exit_ts: int; exit_price: float; outcome: str
    pnl_pct: float; volume_high: bool
```
(Written one-per-line in the real file — collapsed here only to show the order.)

### COST_RESOLUTION — config defaults resolved once, at the top
```python
# SOURCE: backtest/engine.py:259-263
    fee = config.FEE_PCT if fee_pct is None else fee_pct
    slip = config.SLIPPAGE_PCT if slippage_pct is None else slippage_pct
    funding = config.FUNDING_PCT_PER_DAY if funding_pct_per_day is None else funding_pct_per_day
    max_hold = config.MAX_HOLD_BARS_TRIGGER if max_hold_bars is None else max_hold_bars
    cost = 2 * (fee + slip)
```

### CLI_SUBCOMMAND
```python
# SOURCE: cli.py:112-126
    backtest_parser = subparsers.add_parser(
        "backtest", help="Replay the signal pipeline over stored history"
    )
    backtest_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to backtest (repeatable); default is all symbols",
    )
    backtest_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD (default: BACKFILL_START)",
    )
```
```python
# SOURCE: cli.py:371-393 — note the RETURN-CODE RULE, which graph-backtest inherits
def _backtest_command(conn, symbols, *, start_ms: int, end_ms: int) -> int:
    """
    Run a single backtest per symbol and print metrics with bucket breakdown.

    Returns:
        0 always (a backtest with zero trades is a result, not an error).
    """
    for symbol in symbols:
        trades = run_backtest(conn, symbol, start_ms=start_ms, end_ms=end_ms)
        m = compute_metrics(trades)
        print(f"{symbol}:")
        _print_metrics(m, indent="  ")
        ...
        em = compute_equity_metrics(trades, start_ms, end_ms)
```

### TEST_STRUCTURE — tier-derived constants, autouse cache clear, hand-computed values
```python
# SOURCE: tests/test_backtest.py:24-33
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
```
```python
# SOURCE: tests/test_backtest.py:36-47
@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    """Clear the engine's indicator memo around every test.
    ...test isolation must not DEPEND on that argument being right..."""
    engine.clear_caches()
    yield
    engine.clear_caches()
```
```python
# SOURCE: tests/test_wilder.py:325-358 (abridged) — the bar for numeric tests
class TestHandComputedValues:
    """Test against hand-computed reference values for correctness."""

    def test_atr_hand_computed_simple(self):
        rows = [[START + i * INTERVAL, 100.0, 102.0, 98.0, 100.0, 10.0] for i in range(7)]
        df = make_dataframe(rows)
        atr = wilder.atr(df, period=3)
        # All TR values are 4.0
        # atr[2] = (4 + 4 + 4) / 3 = 4.0
        assert abs(atr.iloc[2] - 4.0) < 0.01
```

---

## Design decisions (made here, so implementation asks nothing)

### D1 — MACD tier: the **SETUP** tier (`config.SIGNAL_PATTERN_TIMEFRAME`, currently 4h)

Chosen, with the same reasoning `signals/donchian.py:28-31` already recorded as
Stated Assumption A1:

1. **Risk and reward are computed on the setup tier.** The stop is
   `ATR_STOP_MULTIPLE * ATR(setup TF)`; the target is `level ± target_height` where
   `target_height` is a setup-tier channel width. Putting the momentum confirmation on a
   different tier would confirm on one volatility scale and size risk on another — exactly
   the mistake A1 was written to prevent.
2. **Every existing detector already lives there.** `detect_donchian_setups`,
   `detect_fade_setups` and `detect_patterns` all take a setup-tier window
   (`engine.py:339-347`). A setup-tier MACD is one code path, not a fourth tier.
3. **The 1d regime tier is wrong**: it holds only 1304 bars/symbol, and a daily MACD changes
   state once per day — nearly constant across a whole setup window. A gate that is on or
   off for days at a time is a regime proxy, and the regime layer already exists and is the
   *one measured-healthy component* (contract §1). Duplicating it would be KNOWN-LIMITATIONS
   §0's error repeated: "three of its four conditions … are the same trend-strength idea
   measured three ways."
4. **The 1h trigger tier is wrong**: MACD state could flip *inside* a setup window,
   decoupling the confirmation from the geometry it confirms, and would make MACD the
   fastest-moving input in a system whose regime is daily.

**Rejected**: multi-tier MACD agreement (1d *and* 4h). It doubles the gate's degrees of
freedom for an unmeasured benefit and belongs on a Phase 6 grid axis if anywhere.

### D2 — MACD warmup, derived (never a magic number)

`ema(s, p)` uses `ewm(span=p, adjust=False, min_periods=p)`. **Verified numerically**
(`.venv/bin/python`, pandas 3.0.3) on a 60-bar series:

- `line = ema(s,12) - ema(s,26)` first defined at positional index **25** (`= slow - 1`)
- `sig  = ema(line, 9)` first defined at positional index **33** (`= slow - 1 + signal - 1`)

Therefore:

```python
# Warmup: the MACD line needs MACD_SLOW_PERIOD bars before it is defined, and the
# signal line needs MACD_SIGNAL_PERIOD defined line values on top of that. First
# defined positional index is MACD_SLOW_PERIOD - 1 + MACD_SIGNAL_PERIOD - 1, so the
# bar COUNT required is one more than that. On 4H setup bars that is ~5.7 days —
# dominated by REGIME_MIN_BARS (207 daily bars), so it costs no usable history.
MACD_MIN_BARS = MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD - 1  # 34
```

34 ≪ `PATTERN_LOOKBACK_BARS = 180`, so **no lookback constant changes** — mirroring the
same finding recorded in the v0.2.0 Phase 2 plan's Task 4 GOTCHA.

### D3 — Volume: **hard gate**, with the graded value reported alongside

The PRD's Must row says "Breakout confirmation: volume-on-breakout + MACD" — a
confirmation, i.e. a gate. `breakout.py:19-23` and `config.py:74` both currently document
the opposite ("graded confidence input, never a hard block"), and KNOWN-LIMITATIONS §0c
records that the graded use "was never built". This phase picks the gate. Reasons:

1. **A graded confidence has nothing to grade.** Contract §10 keeps v0.2.0's equal-notional,
   one-open-trade-per-symbol rule and explicitly rules position sizing out of all nine
   phases. With no sizing and no ranking contest (`rank_signals` already orders by R:R), a
   confidence score can influence behavior only through a threshold — which *is* a hard
   gate. "Graded" without sizing would be decorative, and decoration that looks like a
   feature is how §0c happened.
2. **A gate is measurable; a weight is not.** A gate's cost is a trade-count and expectancy
   delta from one ablation run (Task 16). A weight's effect is entangled with everything
   downstream.
3. **The information is not thrown away.** `ConfirmationVerdict` carries `score: float`
   (contract §3). `volume_breakout` returns `score = volume_ratio`, so the graded value is
   in the audit trail for Phase 5's Reviewer and Phase 6's Mutator to use later without
   Phase 4 pretending to use it now.

**Rejected: graded confidence as a soft weight** — no sizing to apply it to (see 1); would
need a second mechanism (confidence → something) that no phase owns; and it re-creates the
"computed but gates nothing" failure mode under a new name.

Recorded as a consumed degree of freedom (KNOWN-LIMITATIONS §9) with the ablation
measurement attached.

**Threshold: no new number.** `VOLUME_CONFIRM_MIN_RATIO = VOLUME_HIGH_RATIO` (1.5) — defined
*by reference* to the existing constant, so activating the gate invents no value and consumes
no additional freedom. `VOLUME_HIGH_RATIO` itself is not modified (contract §7).

**NaN → reject (fail closed).** `breakout.py:131-137` yields `volume_ratio = NaN` when fewer
than `VOLUME_LOOKBACK` prior bars exist or their mean is ≤ 0. "Unknown" must never read as
"confirmed". Cost: the first ~21 trigger bars of each symbol's history cannot trade —
negligible against `REGIME_MIN_BARS`' 207-day warmup, and stated in the module docstring.

### D4 — MACD Confirmation semantics: sign test on the normalized histogram

`confirmation.macd` passes when the MACD histogram at the setup bar agrees with the event
direction: `hist > 0` for long, `hist < 0` for short.

- Magnitude threshold `MACD_CONFIRM_MIN_HIST = 0.0` — a pure sign test by default, so **no
  fitted number enters**. The constant exists so Phase 6 can jitter it within declared
  bounds, not so Phase 4 can tune it.
- Because the donor normalizes by price (`hist = (line - sig) / s`), the threshold is in
  fraction-of-price units and is portable across symbols — that is the donor's stated
  rationale (`indicators.py:164-166`), and sign is unaffected by a positive divisor.
- `score = hist` (the graded value, reported not used — same discipline as D3).
- NaN `hist` (warmup) → **reject**, fail closed, distinct `reason`.

### D5 — `macd_cross` Detector: level = the crossing bar's extreme, `target_height` = the 20-bar channel width

A MACD cross has no natural breakout level, and `DetectedEvent` requires `level` and
`target_height` (contract §3). Chosen, both parameter-free:

- **`level`** = the crossing setup bar's `high` (long) / `low` (short). The trigger bar must
  close beyond the extreme of the bar on which the cross completed. This makes the whole
  existing trigger machinery — `check_breakout`'s fresh-crossing rule, its contiguity check,
  its volume computation — reusable **unchanged**, which is contract §3's stated reason for
  making `DetectedEvent` field-compatible with `PatternCandidate`.
- **`target_height`** = `donchian(window, period=config.DONCHIAN_ENTRY_PERIOD)` width
  (`upper - lower`) at the crossing bar. Rationale is quoted verbatim from
  `signals/donchian.py:19-22`: "the 20-bar CHANNEL WIDTH — a derived quantity, the range the
  market has just resolved, adding no parameter beyond the canonical 20."

Consequence that makes this phase's central measurement valid: both detectors' plans have
the **same reward and risk construction** (channel-width target, 1.5·ATR stop), so their
net-R:R distributions are directly comparable and the survival measurement is not confounded
by two different TP conventions.

**Rejected: `target_height = MACD_TARGET_ATR_MULTIPLE * ATR`.** It injects a fitted
parameter into the exact quantity the R:R Filter measures — fitting the target to clear the
filter is precisely the dishonesty `config.py:162-174` guards against.

MACD Cross is a **Tier-3 (⭐⭐⭐☆☆)** pattern in `.claude/technical-pattern.md:302`. Contract
§9 makes tiers 3–4 "a direction, not a gate": no success criterion in this phase depends on
`macd-cross` having edge. It is here because the pivot guide's Step 2 names MACD and because
the thin slice needs to prove the framework composes **more than one** detector.

### D6 — Parity is protected by comparing named fields, not `Trade` equality

Contract §5 defines parity as "same count, same entry/exit timestamps, same `pnl_pct` to
floating-point tolerance" — deliberately field-wise. Once `run_graph_backtest` populates
`planned_rr` and `confirmations`, a graph-produced `Trade` will **not** be `==` to an
`engine.run_backtest` `Trade` even for an identical strategy. If Phase 3's
`tests/test_framework_parity.py` compares whole dataclasses, Phase 4 must narrow it to the
three named fields, cite contract §5 in the test docstring, and add a regression test in
`tests/test_pipeline_thin_slice.py` pinning that the new fields are excluded from parity by
design. **Check this before writing any code** (Task 0).

### D7 — `Trade.strategy_version` is added but left empty

Phase 4 appends the field with its `""` default and does **not** populate it. Contract §5
assigns it to "Phase 5's version registry" and §6 gives Phase 5 the `strategy_versions`
table. A test asserts `strategy_version == ""` for Phase 4 trades, so Phase 5 has exactly
one writer and no ambiguity about who owns the value.

### D8 — `Trade.confirmations` is not redundant

With hard gates, a trade exists only if every Confirmation passed, so the tuple looks like a
restatement of the graph. It is not: it survives serialization, it distinguishes graphs after
Phase 6 mutates node sets, and it stays correct when a future Confirmation is advisory
(`passed=True` always, `score` varying). Populate it with the **registry keys of the
Confirmations that returned `passed=True`, sorted**, so the value is deterministic across
runs — determinism matters because Phase 6 hashes graph results.

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/config.py` | UPDATE | Append the Phase 4 block: `MACD_*`, `VOLUME_CONFIRM_*`, `RR_TARGET_MIN` (contract §7 reserved prefixes). No existing constant modified. |
| `src/trading_bot/indicators/macd.py` | CREATE | MACD beside `wilder.py`, ported from `scripts/bruteforce/indicators.py:160-172` + `:79-82` with attribution. Hand-rolled, no new dependency (contract §1). |
| `src/trading_bot/plugins/detectors/macd_cross.py` | CREATE | MACD cross as a `Detector` (contract §2 assigns this path to P4). |
| `src/trading_bot/plugins/confirmations/volume_breakout.py` | CREATE | Volume-on-breakout `Confirmation` — activates machinery that today gates nothing. |
| `src/trading_bot/plugins/confirmations/macd.py` | CREATE | MACD `Confirmation`. |
| `src/trading_bot/plugins/policies/measured_move.py` | CREATE | `PositionPolicy`: direction/entry/TP/SL from the pattern outcome (pivot guide Step 3). |
| `src/trading_bot/plugins/filters/rr_after_costs.py` | CREATE | The ≥1:2-after-costs `Filter` (pivot guide Step 4) + its measure-only report. |
| `src/trading_bot/backtest/engine.py` | UPDATE | **`Trade` gains 3 fields, appended, with defaults** (contract §5). Nothing else in this file changes. |
| `src/trading_bot/framework/execute.py` | UPDATE | **Authorized cross-ownership edit** — see Task 12. Additive kwargs at the single `Trade(...)` construction site so `planned_rr` / `confirmations` are populated. Contract §5 assigns the fields to Phase 4; the seam is the only place their values exist. |
| `src/trading_bot/cli.py` | UPDATE | `graph-backtest` subparser + `_graph_backtest_command` (contract §7 reserved name/handler). |
| `data/strategies/thin-slice.strategy.json` | CREATE | The composed thin-slice graph, serialized by Phase 3's `StrategyGraph.to_dict()`. |
| `data/strategies/thin-slice-noconfirm.strategy.json` | CREATE | Same graph minus the two Confirmation nodes — the ablation baseline for Task 16's measurement. |
| `tests/test_macd.py` | CREATE | Hand-computed MACD values, the seeding convention, and the window-dependence pin. |
| `tests/test_plugins_confirmations.py` | CREATE | Both Confirmations: pass, fail, NaN-fail-closed, never-mutates-event. |
| `tests/test_plugins_policies.py` | CREATE | `measured-move`: long/short symmetry, NaN ATR, degenerate risk/reward. |
| `tests/test_plugins_filters.py` | CREATE | `rr-after-costs`: boundary, the `2 + 3c` identity, funding-blindness note, report shape. |
| `tests/test_pipeline_thin_slice.py` | CREATE | End-to-end composed graph on a synthetic fixture; `Trade` field population; parity-exclusion regression; graph-file round-trip. |
| `tests/test_framework_parity.py` | UPDATE (only if needed) | Narrow to contract §5's named fields — see D6/Task 0. |
| `.claude/PRPs/reports/phase4-thin-slice.md` | CREATE | The phase report: R:R survival, volume/MACD ablation, DOF ledger (contract §12.5/§12.6). |
| `.gitignore` | UPDATE (verify only) | Confirm `data/strategies/*.strategy.json` is **not** ignored — these are committed artifacts, unlike `data/state.db*`. |

## NOT Building

- **`indicators/rsi.py` and any additional detector** — Phase 8 owns them (contract §2).
  `macd-cross` is the only new detector here.
- **Any change to `walkforward.py`** — Phase 1 owns the gate extension, Phase 3 the
  `strategy=` keyword. Phase 4 does not touch it, does not add a grid axis, and does not run
  the gate. `graph-backtest` is a *backtest*, not a validation run.
- **Any sweep, grid, or tuning.** No parameter in this phase is selected on returns. The four
  runs in Task 16 are diagnostic ablations and **all four are reported**, so none is a hidden
  selection.
- **Lowering `RR_TARGET_MIN`, or touching `RR_FLOOR`.** See the decision rule above.
- **Modifying `RR_FLOOR`, `ATR_STOP_MULTIPLE`, `VOLUME_HIGH_RATIO`, `VOLUME_LOOKBACK`, the
  cost constants, or the regime classifier.** Frozen (contract §1, §7).
- **Rewriting the legacy path.** `signals/setup.py`, `signals/donchian.py`,
  `signals/breakout.py`, `signals/meanrev.py`, `regime/classifier.py`, `data/storage.py`,
  `backtest/metrics.py`, `backtest/equity.py` are **read-only** this phase. The legacy
  `build_signal` keeps gating on gross `RR_FLOOR = 1.5`; its measured behavior is not
  silently altered (contract §7).
- **Multi-position / overlapping long+short.** Contract §10 Q7: out of scope for all nine
  phases; one open trade per symbol, equal notional. The known gap against the pivot guide's
  "trust the signal" principle is already flagged to the user there.
- **Position sizing, live execution, news data.** PRD "What We're NOT Building".
- **Reviewer, Mutator, evolution, UI.** Phases 5, 6, 7.
- **A runtime hard-fail on `cost_ratio > COST_RATIO_CEILING`.** It is asserted in tests and
  reports, never enforced per-candidate — the rule the v0.2.0 Phase 2 plan set and
  `config.py:183` records ("asserted in tests/reports, not enforced at runtime").
- **Funding inside the Filter.** `net_rr` charges fee + slippage only;
  `risk/atr_stop.py:82-88` states the funding blindness explicitly. The Filter inherits that
  limitation and **must repeat the caveat in its docstring** rather than invent a
  hold-duration estimate at signal time (unknowable then — `round_trip_cost_pct`'s docstring
  at `:41-42` says so).

---

## Step-by-Step Tasks

### Task 0: Verify the Phase 3 baseline before writing anything
- **ACTION**: Confirm the ground this phase stands on.
- **IMPLEMENT**: Run and record:
  ```bash
  .venv/bin/python -m pytest --collect-only -q 2>&1 | tail -2
  .venv/bin/python -m pytest -q
  .venv/bin/python -m pytest tests/test_framework_parity.py -q
  ```
  Then read `src/trading_bot/framework/contracts.py`, `registry.py`, `graph.py`,
  `context.py`, `execute.py` as Phase 3 actually shipped them, and write down: the exact
  `register(...)` signature and `ParamSpec` shape; how `EvalContext` exposes a setup-tier
  frame and the current bar (this is what every plug-in below calls); the `NodeSpec` /
  `to_dict` envelope; and **whether `test_framework_parity.py` compares whole `Trade`
  objects** (D6).
- **MIRROR**: `scripts/bruteforce/registry.py:105-120` for what the decorator will look like.
- **IMPORTS**: n/a.
- **GOTCHA**: If parity is red, **stop**. Contract §5: Phase 4 may only extend from a green
  parity test. If `EvalContext` cannot hand a plug-in a setup-tier window ending at the
  current bar, that is a Phase 3 gap — report it rather than reaching into `storage` from a
  plug-in, which would bypass `_assert_interval` (`engine.py:185-222`) and void every
  no-lookahead guarantee.
- **VALIDATE**: Baseline test count recorded; parity green; the five framework signatures
  transcribed into the implementation notes.

### Task 1: Append the Phase 4 config block
- **ACTION**: Append to the **end** of `src/trading_bot/config.py`. Modify nothing above.
- **IMPLEMENT**:
  ```python
  # ---------------------------------------------------------------------------
  # Phase 4 (v0.3.0): strategy pipeline thin slice — MACD, volume gating, and the
  # >=1:2-after-costs reward:risk target. Reserved prefixes per the v0.3.0 shared
  # architecture contract §7: MACD_*, VOLUME_CONFIRM_*, RR_TARGET_MIN.
  # ---------------------------------------------------------------------------

  # MACD (Appel). Canonical 12/26/9 — inherited, NOT fitted here, and never swept
  # in this phase. Computed on the SETUP tier (SIGNAL_PATTERN_TIMEFRAME): risk
  # (1.5*ATR) and reward (channel width) are both setup-tier quantities, so
  # confirming momentum on a different tier would put signal and risk on
  # different volatility scales — the same argument as signals/donchian.py's
  # Stated Assumption A1.
  MACD_FAST_PERIOD = 12
  MACD_SLOW_PERIOD = 26
  MACD_SIGNAL_PERIOD = 9
  # Warmup, DERIVED (never a magic number), mirroring DONCHIAN_MIN_BARS above:
  # the MACD line is first defined at positional index MACD_SLOW_PERIOD - 1, and
  # the signal line needs MACD_SIGNAL_PERIOD defined line values on top of that,
  # so the first defined index is MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD - 2 and
  # the required BAR COUNT is one more. Verified against pandas' ewm(adjust=False,
  # min_periods=period): line idx 25, signal idx 33 for 12/26/9.
  # 34 setup bars is ~5.7 days at the 4H tier — dominated by REGIME_MIN_BARS
  # (207 daily bars), so MACD costs no usable history.
  MACD_MIN_BARS = MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD - 1  # 34

  # MACD confirmation threshold on the PRICE-NORMALISED histogram
  # ((line - signal) / close, so the value is portable across symbols). 0.0 makes
  # it a pure SIGN test, which introduces no fitted number. The constant exists so
  # the Phase 6 Mutator can jitter it within declared bounds — not so this phase
  # can tune it.
  MACD_CONFIRM_MIN_HIST = 0.0

  # Volume-on-breakout confirmation, Phase 4. THIS IS A BEHAVIOR CHANGE:
  # VOLUME_LOOKBACK / VOLUME_HIGH_RATIO have existed since Phase 3 and gated
  # NOTHING (KNOWN-LIMITATIONS §0c: "volume is computed on every signal but gates
  # nothing"). The Confirmation plug-in makes it a HARD GATE.
  #
  # The threshold is defined BY REFERENCE to the existing constant, so activating
  # the gate invents no new number and consumes no additional degree of freedom.
  # VOLUME_HIGH_RATIO itself is unchanged (contract §7: do not modify existing
  # constants).
  VOLUME_CONFIRM_MIN_RATIO = VOLUME_HIGH_RATIO
  # Undefined volume (NaN ratio: fewer than VOLUME_LOOKBACK prior bars, or a
  # non-positive rolling mean — see signals/breakout.py:131-137) REJECTS. "Unknown"
  # must never read as "confirmed". Cost: the first ~VOLUME_LOOKBACK+1 trigger bars
  # of each series cannot trade, negligible against REGIME_MIN_BARS' 207 days.
  VOLUME_CONFIRM_REQUIRE_DEFINED = True

  # The PRD's ">=1:2 risk:reward AFTER COSTS" requirement, as a NET floor consumed
  # by plugins/filters/rr_after_costs.py via risk.atr_stop.net_rr.
  #
  # This is a NEW constant, deliberately NOT a change to RR_FLOOR = 1.5, which
  # stays the legacy breakout path's GROSS floor so its measured behavior is not
  # silently altered.
  #
  # Requiring net_rr >= X is equivalent to requiring gross_rr >= X + (X+1)*c,
  # where c = cost/risk_pct (risk.atr_stop.cost_ratio). At X = 2.0 and the median
  # risk_pct measured at the 4H setup tier and recorded above (BTC 1.968% -> c
  # 0.0711, ETH 2.679% -> 0.0523, SOL 3.752% -> 0.0373), that is a GROSS floor of
  # 2.213 / 2.157 / 2.112. walkforward.py:53-68 records that the MINIMUM planned
  # gross R:R across every trade the engine ever took was 1.56 — whose net_rr is
  # 1.39/1.43/1.47. This floor is therefore expected to reject the large majority
  # of plans. That is measured (cli graph-backtest --rr-report), reported, and NOT
  # a licence to weaken it: derived from costs, never fitted to returns, exactly as
  # ATR_STOP_MULTIPLE was.
  RR_TARGET_MIN = 2.0
  ```
- **MIRROR**: `config.py:74-76` (volume block style), `config.py:139` and `config.py:40`
  (derived warmup), `config.py:162-174` (a constant's derivation recorded in the comment).
- **IMPORTS**: none (same module).
- **GOTCHA**: `VOLUME_CONFIRM_MIN_RATIO = VOLUME_HIGH_RATIO` must be a **reference**, not a
  re-typed `1.5`. A future change to `VOLUME_HIGH_RATIO` must move both together or the two
  numbers drift silently — the identical trap `TRAIL_ATR_MULTIPLE` was split out of
  (`config.py:118-125`). Also: append at the **end of the file, after `COST_RATIO_CEILING`
  and before `date_to_ms`** — appending inside an earlier block breaks the phase-ordering
  convention contract §7 relies on.
- **VALIDATE**:
  ```bash
  .venv/bin/python -c "from trading_bot import config as c; \
    print(c.MACD_FAST_PERIOD, c.MACD_SLOW_PERIOD, c.MACD_SIGNAL_PERIOD, c.MACD_MIN_BARS, \
          c.MACD_CONFIRM_MIN_HIST, c.VOLUME_CONFIRM_MIN_RATIO, c.RR_TARGET_MIN, c.RR_FLOOR)"
  # EXPECT: 12 26 9 34 0.0 1.5 2.0 1.5
  git diff --stat src/trading_bot/config.py   # additions only, zero deletions
  ```

### Task 2: `indicators/macd.py` — port the donor with attribution
- **ACTION**: Create `src/trading_bot/indicators/macd.py`.
- **IMPLEMENT**: Two functions —
  `ema(series, period) -> pd.Series` and
  `macd(series, *, fast=None, slow=None, signal=None) -> pd.DataFrame` (columns
  `macd`, `signal`, `hist`) — bodies **identical to the donor**, periods defaulting to
  `config.MACD_FAST_PERIOD / _SLOW_PERIOD / _SIGNAL_PERIOD`, plus a guard returning an
  all-NaN frame of the right shape for an empty input.

  The module docstring is load-bearing and must contain, in this order:
  1. **Provenance**: ported from `scripts/bruteforce/indicators.py:160-172` (`macd`) and
     `:79-82` (`ema`), the donor described in contract §0b; behavior deliberately identical,
     only docstrings/config-wiring/tests are new. `pandas-ta` is gone from PyPI and TA-Lib
     needs a C library, so the hand-rolled set is extended rather than a dependency added
     (contract §1).
  2. **The EMA seeding convention, spelled out** — the classic source of silent MACD
     mismatch. `ema(s, p) == s.ewm(span=p, adjust=False, min_periods=p).mean()`. With
     `adjust=False`, pandas runs the recursion from the **first observation**
     (`y[0] = x[0]`; `y[i] = (1-a)·y[i-1] + a·x[i]`, `a = 2/(p+1)`) and then **masks** the
     first `p-1` outputs as NaN. It does **not** seed with an SMA at index `p-1`.
  3. **Both conventions it differs from**: (a) `indicators/wilder.py`'s `wilder_smooth` seeds
     with the *simple average* of the first `period` values (`wilder.py:92-94`) and decays at
     1/N, not 2/(N+1) — MACD is defined on EMAs, so it must **not** use Wilder smoothing;
     that would be a different indicator. (The donor's `rsi` deliberately *does* use
     `wilder_smooth`; its `macd` deliberately does not.) (b) TA-Lib and TradingView seed the
     first EMA with an SMA, so values near the start of a series differ from those platforms
     and converge asymptotically. **Do not "fix" this to match a chart; fix the comparison.**
  4. **CONSEQUENCE — MACD is not window-invariant.** The recursion is seeded at the first bar
     of whatever series it is given, so `MACD(window) != MACD(full)[window]` in general: the
     same bar can show a cross on a 180-bar window and not on the full history. The same is
     already true of Wilder ADX, which `signals/donchian.py:103` computes per lookback
     window. **Convention: MACD is always computed on the window the caller supplies**
     (`EvalContext` / the `PATTERN_LOOKBACK_BARS` slice), never on a differently sized slice,
     so live and backtest see identical values. `tests/test_macd.py` pins this.
  5. **Causality**: `adjust=False` + `min_periods=period` satisfy the donor's two rules
     (`indicators.py:1-12`) — trailing windows only; NaN means "not knowable yet", never
     back-filled.
  6. **Normalisation**: `hist = (line - signal) / close`. Dividing by price makes a threshold
     portable across symbols — "a raw MACD threshold that works on BTC is meaningless on
     DOGE" (`indicators.py:164-166`). Sign is unaffected, so a cross/sign test is identical
     normalised or not.
  7. **Warmup**: `hist` first defined at positional index `slow - 1 + signal - 1` (33 at
     12/26/9); `config.MACD_MIN_BARS` is the corresponding bar **count**.
  8. I/O contract, mirroring `wilder.py:15-18`: input a `pd.Series` of closes on an epoch-ms
     int index; output indexed identically with leading NaNs preserved.
- **MIRROR**: `indicators/wilder.py:1-18` (docstring conventions block),
  `indicators/donchian.py:25-45` (config-defaulted keyword period), the donor at
  `scripts/bruteforce/indicators.py:79-82, 160-172`.
- **IMPORTS**: `import pandas as pd`; `from trading_bot import config`.
- **GOTCHA**:
  1. **Do not** implement `ema` via `wilder_smooth`. Different seed, different decay,
     different indicator.
  2. `min_periods=period` is load-bearing. Without it, `ewm` returns a value at index 0 and
     the warmup region silently becomes a short-sample estimate — violating the donor's
     causality rule 1.
  3. `ema(line, signal)`: `line` has 25 leading NaNs. `ewm` **skips** them (they are not
     observations), so the signal line is defined once 9 non-NaN `line` values exist →
     index 33. Verified numerically. Do not `dropna()` or `fillna()` `line` first — that
     shifts the whole indicator.
  4. Division by `series`: guard nothing. Closes are positive by construction in this store;
     a zero close would already have broken `risk_pct` everywhere. Do **not** add a
     `replace(0, nan)` that the donor does not have — behavioral drift from the donor is the
     one thing a port must not introduce.
  5. `scripts/bruteforce/indicators.py` **imports from `src`** (`:29-41`). Never import the
     other direction: `src/trading_bot` must not depend on `scripts/`.
- **VALIDATE**: `python -m py_compile src/trading_bot/indicators/macd.py`; then Task 3.

### Task 3: `tests/test_macd.py` — hand-computed values, the seeding convention, the window pin
- **ACTION**: Create `tests/test_macd.py`.
- **IMPLEMENT**: Classes mirroring `tests/test_wilder.py`'s organization:
  - `class TestEmaSeedingConvention` — the load-bearing tests. On
    `[10.0, 20.0, 30.0, 40.0]` with `period=3` (α = 0.5), **hand-computed**:
    `y0 = 10`, `y1 = 15`, `y2 = 22.5`, `y3 = 31.25`; `min_periods=3` masks indices 0–1, so
    the returned series is `[nan, nan, 22.5, 31.25]`. **Verified numerically against pandas
    3.0.3.** Assert index 2 is `22.5`, and explicitly assert it is **not** `20.0` — the value
    an SMA-seeded (TA-Lib/TradingView) implementation would give — with the comment naming
    that convention. Also assert `ema` is not `wilder_smooth`: `wilder_smooth([10,20,30,40], 3)`
    seeds at `20.0` (the simple mean of the first three) — assert the two differ at index 2.
  - `class TestWarmupIndices` — on a ≥60-bar ramp with 12/26/9: `macd` column first valid at
    positional index **25**; `signal` and `hist` at **33**; and
    `config.MACD_MIN_BARS == 33 + 1`. Derives the expectation from the config constants, not
    from literals (`tests/test_backtest.py:24-33` pattern), so a period change cannot leave
    the test green and wrong.
  - `class TestNoLookahead` — truncating the series after bar `i` leaves every value at
    indices ≤ `i` unchanged (this is what `adjust=False` buys).
  - `class TestWindowDependence` — the pin for the module docstring's CONSEQUENCE
    paragraph: on a synthetic series, `macd(s.iloc[-60:])` differs from
    `macd(s).iloc[-60:]` at the earliest bars of the window, and **converges** later.
    Docstring states this is deliberate and names the convention (always compute on the
    caller's window). Without this test, a future "optimization" that hoists MACD out of the
    per-window computation would silently change every signal.
  - `class TestNormalisation` — `hist == (macd_line_unnormalised - signal_unnormalised)/close`
    in sign and magnitude for a hand-checkable 3-bar tail; and that sign is invariant to
    normalisation (the property D4 relies on).
  - `class TestDegenerate` — empty frame → empty result with the three columns; a series
    shorter than `MACD_MIN_BARS` → all-NaN `hist`, no exception.
- **MIRROR**: `tests/test_wilder.py:325-358` (`TestHandComputedValues`, comment-the-arithmetic
  style, `abs(x - expected) < 0.01` tolerance), `tests/test_backtest.py:24-33` (derive
  constants from config).
- **IMPORTS**: `import math`, `import pandas as pd`, `import pytest`,
  `from trading_bot import config`, `from trading_bot.indicators import macd as macd_mod`,
  `from trading_bot.indicators.wilder import wilder_smooth`.
- **GOTCHA**: Write the arithmetic **in comments beside each assertion**, as
  `test_wilder.py:353-356` does. A numeric test whose expected value cannot be re-derived by
  a reader is a snapshot, not a test — and a snapshot of a wrong convention is exactly the
  failure this task exists to prevent. No `pytest.approx` on the seeding tests: the values
  are exact.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_macd.py -v` — all pass.

### Task 4: `plugins/detectors/macd_cross.py`
- **ACTION**: Create `src/trading_bot/plugins/detectors/macd_cross.py`.
- **IMPLEMENT**: Registered under `detector.macd-cross` with a **non-empty `rationale`**
  (registry requirement, `scripts/bruteforce/registry.py:70-73`) naming its
  `.claude/technical-pattern.md:302` Tier-3 reliability honestly. Declared `params`
  (`ParamSpec`s): `fast`, `slow`, `signal` (int, defaults from config), `min_hist` (float,
  default `config.MACD_CONFIRM_MIN_HIST`). Declared `timeframes` = the setup tier;
  `tier` per Phase 3's convention.

  Logic, on the setup-tier window `ctx` supplies:
  1. Return `[]` if `len(window) < config.MACD_MIN_BARS` (warmup).
  2. `h = macd(window["close"], ...)["hist"]`; take the last two values `h[-2]`, `h[-1]`.
  3. Return `[]` if either is NaN (`not math.isfinite(...)`) — fail closed.
  4. Bullish cross: `h[-2] <= 0 < h[-1]`. Bearish: `h[-2] >= 0 > h[-1]`. Otherwise `[]`.
     Strict inequality on the current bar and non-strict on the previous is the same
     fresh-crossing shape `breakout.py:124-127` uses — one bar, one event, no re-trigger
     while the histogram stays on one side.
  5. `level` = last bar's `high` (long) / `low` (short) — D5.
  6. `target_height` = `donchian(window, period=config.DONCHIAN_ENTRY_PERIOD)` width at the
     last bar; return `[]` if NaN or ≤ 0 (mirrors `donchian.py:116-118`'s degenerate-channel
     guard).
  7. `start_ts` = the bar `DONCHIAN_ENTRY_PERIOD` back (clamped);
     `end_ts` = last bar's ts **+ the observed setup interval**, computed from the last two
     index values exactly as `donchian.py:127` does — so the trigger bar is guaranteed to
     open at or after the setup bar closed.
  8. `meta` carries `{"hist": float(h[-1]), "hist_prev": float(h[-2])}` — the audit trail.
- **MIRROR**: `signals/donchian.py:65-140` end to end (warmup guard → NaN guard → threshold
  → degenerate guard → direction → candidate construction with derived `target_height` and
  interval-derived `end_ts`).
- **IMPORTS**: `import math`; `from trading_bot import config`;
  `from trading_bot.indicators.donchian import donchian`;
  `from trading_bot.indicators.macd import macd`;
  `from trading_bot.framework.registry import register`;
  `from trading_bot.framework.contracts import DetectedEvent`.
- **GOTCHA**:
  1. `donchian()` **excludes the current bar** (`indicators/donchian.py:31-37,42-43`: it
     `.shift(1)`s), so its first defined value is at positional index `period`. The window
     must therefore hold at least `max(config.MACD_MIN_BARS, DONCHIAN_ENTRY_PERIOD + 1)`
     bars. Derive the guard from both constants; do not assume 34 dominates.
  2. Do **not** compute MACD on the full series and slice — D5/Task 3's window pin.
  3. `DetectedEvent` is frozen (contract §3). Build it once; never mutate.
  4. Emit **at most one** event, like `detect_donchian_setups` (`donchian.py:84-86`), so the
     graph's `rank`-by-R:R stage has a well-defined single candidate per detector per bar.
- **VALIDATE**: `python -m py_compile`; covered end-to-end by Task 15's fixture, and by
  `cli plugins` listing `detector.macd-cross` with its rationale (contract §12.3).

### Task 5: `plugins/confirmations/volume_breakout.py`
- **ACTION**: Create `src/trading_bot/plugins/confirmations/volume_breakout.py`.
- **IMPLEMENT**: Registered `confirmation.volume-breakout`. Params: `min_ratio` (float,
  default `config.VOLUME_CONFIRM_MIN_RATIO`), `lookback` (int, default
  `config.VOLUME_LOOKBACK`), `require_defined` (bool, default
  `config.VOLUME_CONFIRM_REQUIRE_DEFINED`).

  Module docstring must open by naming the change: **this Confirmation converts a
  computed-but-unused number into a hard gate.** `config.VOLUME_LOOKBACK` /
  `VOLUME_HIGH_RATIO`, `Signal.volume_ratio` / `volume_high` and `Trade.volume_high` have
  existed since Phase 3 and gated nothing (KNOWN-LIMITATIONS §0c); `signals/breakout.py:19-23`
  documents the old policy ("graded confidence input, never a hard block"), which this plug-in
  deliberately reverses **for graph-composed strategies only** — the legacy `signals/*` path
  is unchanged. Then the rejected alternative from D3: graded confidence as a soft weight has
  nothing to scale under equal notional / one-open-trade-per-symbol (contract §10), so it
  could only act through a threshold, which is this gate; the graded value is still reported
  as `ConfirmationVerdict.score` for Phases 5–6.

  Logic: compute the trigger-bar volume ratio the **same way** `breakout.py:131-137` does —
  trigger volume ÷ mean of the `lookback` bars **strictly preceding** it — from the
  trigger-tier frame `ctx` supplies. Then:
  - ratio NaN and `require_defined` → `ConfirmationVerdict(passed=False, name=…, score=nan,
    reason="volume ratio undefined (fewer than N prior bars or non-positive mean)")`
  - `ratio >= min_ratio` → `passed=True`, `score=ratio`, reason states both numbers
  - else `passed=False`, `score=ratio`, `reason=f"volume ratio {ratio:.2f} < {min_ratio:.2f}"`
- **MIRROR**: `signals/breakout.py:131-138` for the ratio arithmetic **verbatim** (including
  `len(prior_vol) < volume_lookback or float(prior_vol.mean()) <= 0 → NaN`);
  `signals/setup.py:118-125` for the DEBUG reject log alongside the verdict.
- **IMPORTS**: `import math`; `from trading_bot import config`;
  `from trading_bot.framework.registry import register`;
  `from trading_bot.framework.contracts import ConfirmationVerdict`.
- **GOTCHA**:
  1. **Never mutate the event** (contract §3). No `dataclasses.replace`, no attribute
     assignment, no stashing into `event.meta`. Return a verdict. A test asserts the event
     is unchanged and is still the same object.
  2. Two off-by-one traps in one line: the window is the bars **before** the trigger bar
     (`df["volume"].iloc[:i].tail(lookback)`), and it must have **exactly** `lookback`
     entries or the ratio is NaN. Copy `breakout.py:132-136`; do not re-derive.
  3. `math.isnan` — not `ratio != ratio`, not `pd.isna` on a float — matching
     `breakout.py:138`'s `not math.isnan(volume_ratio)`.
  4. Reuse `VOLUME_LOOKBACK`; do **not** introduce `VOLUME_CONFIRM_LOOKBACK` with a
   different value. One volume window in the system.
- **VALIDATE**: Task 7.

### Task 6: `plugins/confirmations/macd.py`
- **ACTION**: Create `src/trading_bot/plugins/confirmations/macd.py`.
- **IMPLEMENT**: Registered `confirmation.macd`. Params: `fast`/`slow`/`signal` (config
  defaults), `min_hist` (default `config.MACD_CONFIRM_MIN_HIST`).

  Logic on the **setup-tier** window (D1 — the docstring must carry D1's four-point
  argument, and cite `signals/donchian.py:28-31`'s A1 as its precedent):
  - `hist = macd(window["close"], …)["hist"].iloc[-1]`
  - not finite → `passed=False`, reason "MACD histogram undefined (warmup: needs
    MACD_MIN_BARS bars)"
  - `event.direction == "long"` → `passed = hist > min_hist`
  - `event.direction == "short"` → `passed = hist < -min_hist`
  - `score = float(hist)`; reason states the value, the threshold and the direction.

  Symmetry note in the docstring: with `min_hist = 0.0` the two branches are the plain sign
  test; a positive `min_hist` makes the gate symmetric around zero rather than biased long.
- **MIRROR**: `signals/setup.py:131-136` NaN-guard idiom; Task 5's verdict shape.
- **IMPORTS**: `import math`; `from trading_bot import config`;
  `from trading_bot.indicators.macd import macd`;
  `from trading_bot.framework.registry import register`;
  `from trading_bot.framework.contracts import ConfirmationVerdict`.
- **GOTCHA**:
  1. `-min_hist` for shorts, not `min_hist` — a sign error here makes the short gate
     *permissive* while looking symmetric. A test asserts long/short symmetry on mirrored
     data.
  2. Same window rule as Task 4: setup-tier window as supplied, never a re-slice.
  3. Never mutate the event.
  4. Do not re-implement MACD here. One implementation, `indicators/macd.py`.
- **VALIDATE**: Task 7.

### Task 7: `tests/test_plugins_confirmations.py`
- **ACTION**: Create `tests/test_plugins_confirmations.py`.
- **IMPLEMENT**:
  - `class TestVolumeBreakout`: passes at `ratio == min_ratio` exactly (boundary is `>=`,
    matching `breakout.py:138`); fails just below; NaN ratio on a short window fails closed
    with the undefined-reason string; `score` equals the measured ratio in every branch; a
    `min_ratio` override via params changes the verdict (proves the `ParamSpec` default is
    resolved, not hardcoded).
  - `class TestMacdConfirmation`: long passes on positive `hist`, fails on negative; short is
    the exact mirror on sign-flipped data; warmup (`< MACD_MIN_BARS` bars) fails closed;
    `min_hist > 0` rejects a small-magnitude positive `hist` for a long **and** a
    small-magnitude negative one for a short (the symmetry pin for Task 6's gotcha 1).
  - `class TestConfirmationsNeverMutate` — parametrized over **both** Confirmations:
    snapshot `dataclasses.asdict(event)` before, call `confirm`, assert equal after, and
    assert the returned verdict is a `ConfirmationVerdict` with a non-empty `name` and
    `reason`. Docstring cites contract §3: "NEVER mutates the event", and states *why* it
    matters — the audit trail is what makes "why did it trade" answerable, the PRD's stated
    advantage of evolution over deep RL.
  - `class TestVerdictAuditability` — every failure path returns a `reason` containing both
    the measured value and the threshold, so a rejection is diagnosable from logs alone.
- **MIRROR**: `tests/test_signals.py:318-335` (`make_candidate` / `breakout_df` builders —
  write the analogous `make_event()` / `vol_frame()` helpers, one per file, reusing the same
  shapes); `tests/test_backtest.py:24-33` tier constants; `tests/test_backtest.py:36-47`
  autouse cache clear if Phase 3's `EvalContext` memoizes anything.
- **IMPORTS**: `dataclasses`, `math`, `pandas as pd`, `pytest`, `from trading_bot import
  config`, the two plug-in modules, `from trading_bot.framework.contracts import
  ConfirmationVerdict, DetectedEvent`.
- **GOTCHA**: Confirmations receive an `EvalContext`, so tests need a minimal context. Use
  whatever Phase 3 provides for its own tests (found in Task 0) rather than inventing a
  second stub shape — two context stubs in one repo is how live/backtest parity dies.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_plugins_confirmations.py -v`.

### Task 8: `plugins/policies/measured_move.py`
- **ACTION**: Create `src/trading_bot/plugins/policies/measured_move.py`.
- **IMPLEMENT**: Registered `policy.measured-move`. Params: `atr_multiple` (default
  `config.ATR_STOP_MULTIPLE`), `atr_period` (default `config.ATR_STOP_PERIOD`).

  This is the pivot guide's Step 3 — "decide the position based on the chart pattern outcome
  (long/short/no-entry) and its entry, and TP/SL point" — and it is `build_signal`'s body
  re-expressed against the contracts. Logic:
  1. `entry` = the trigger bar's close (the entry reference, as `BreakoutEvent.price` is at
     `breakout.py:44`).
  2. `atr_value` = Wilder `atr(setup_window, period=atr_period)` at the last setup bar.
  3. `if not (atr_value > 0): return None` — the NaN/zero/negative guard, verbatim from
     `setup.py:131-136`, comment included.
  4. `stop = compute_atr_stop(entry, event.direction, atr_value, atr_multiple)`.
  5. `target = event.level ± event.target_height` per direction; `reward` accordingly
     (`setup.py:139-144`).
  6. `risk = abs(entry - stop)`; `if risk <= 0 or reward <= 0: return None`.
  7. `risk_pct`, `reward_pct`, `rr = reward_pct / risk_pct` (**gross** — the Filter owns net).
  8. Return `PositionPlan(symbol, ts=event.end_ts-derived trigger ts, direction, entry,
     stop, target, risk_pct, reward_pct, rr, source=event.kind)`.
- **MIRROR**: `signals/setup.py:113-171` line for line — this is a re-expression, not a
  redesign. Reuse `risk.atr_stop.compute_atr_stop` (do not inline `entry - k*atr`).
- **IMPORTS**: `from trading_bot import config`;
  `from trading_bot.indicators.wilder import atr as wilder_atr`;
  `from trading_bot.risk.atr_stop import compute_atr_stop`;
  `from trading_bot.framework.registry import register`;
  `from trading_bot.framework.contracts import PositionPlan`.
- **GOTCHA**:
  1. **`PositionPlan.rr` is GROSS** (`reward_pct / risk_pct`), matching `Signal.rr`
     (`setup.py:54`). The **net** ratio is the Filter's output and lands on
     `Trade.planned_rr`. Two ratios with the same name in different places is the single most
     likely confusion in this phase — say which is which in both docstrings.
  2. The policy performs **no R:R rejection at all**. `build_signal` conflates SL/TP
     computation with the `rr < rr_floor` screen; the contracts split them
     (`PositionPolicy` decides, `Filter` accepts). Do not port the screen here — that would
     apply `RR_FLOOR = 1.5` gross *before* the net-2.0 Filter, silently pre-filtering the
     very distribution Task 10's report is measuring, and making `--rr-report` a report on a
     truncated sample.
  3. ATR comes from the **setup** tier (`config.SIGNAL_PATTERN_TIMEFRAME`) — the tier
     `engine.py:293-296` computes it on. A trigger-tier ATR would shrink every stop by
     roughly √4 and silently blow through `COST_RATIO_CEILING`; `config.py:169-173` records
     that the ceiling was **unsatisfiable at the 1H setup tier**.
  4. Alignment: the ATR value must be the last setup bar **closed by** the trigger bar,
     mirroring `engine.py:481` / `engine.py:492-494`'s `h_idx` rule. If Phase 3's
     `EvalContext` already resolves this, use it; if not, that is a Phase 3 gap (Task 0).
- **VALIDATE**: Task 9.

### Task 9: `tests/test_plugins_policies.py`
- **ACTION**: Create `tests/test_plugins_policies.py`.
- **IMPLEMENT**:
  - Long: `entry=100.3`, `atr=1.0`, `k=config.ATR_STOP_MULTIPLE` → `stop == 100.3 - 1.5`;
    `level=100.0`, `target_height=5.0` → `target == 105.0`; `rr == reward_pct/risk_pct`
    hand-checked. (Same fixture numbers the v0.2.0 Phase 2 plan used for `TestBuildSignal`,
    so the two are trivially comparable.)
  - Short: exact mirror; `stop` above entry, `target = level - height`.
  - `atr_value` NaN → `None`; `0.0` → `None`; negative → `None` (three separate tests; the
    NaN one is the regression pin for `setup.py:131-136`'s reasoning).
  - `reward <= 0` (entry already beyond target) → `None`.
  - `risk_pct`/`reward_pct` are fractions of **entry**, not of level — asserted numerically,
    because getting this wrong is invisible until the Filter's arithmetic is off by ~1%.
  - **Equivalence test**: for a `DetectedEvent` built from the same numbers as a
    `PatternCandidate` + `BreakoutEvent`, `measured-move`'s `(stop, target, risk_pct,
    reward_pct, rr)` equal `signals.setup.build_signal(...)`'s to floating-point tolerance.
    This is the local echo of contract §5's parity requirement and the cheapest possible
    proof that the re-expression did not drift.
  - `atr_multiple` override changes the stop (ParamSpec default is resolved, not hardcoded).
- **MIRROR**: `tests/test_signals.py:397-465` (`TestBuildSignal`) — port its intent one test
  at a time; `math.isclose` for floats.
- **IMPORTS**: `math`, `pytest`, `from trading_bot import config`,
  `from trading_bot.signals.setup import build_signal`,
  `from trading_bot.signals.patterns import PatternCandidate`,
  `from trading_bot.signals.breakout import BreakoutEvent`, the policy module.
- **GOTCHA**: The equivalence test must construct a `BreakoutEvent` **directly** (as
  `tests/test_signals.py` does at its `test_entry_beyond_target_rejected`) rather than
  driving `check_breakout`, so the comparison isolates the arithmetic from the trigger logic.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_plugins_policies.py -v`.

### Task 10: `plugins/filters/rr_after_costs.py` — the ≥1:2-after-costs Filter
- **ACTION**: Create `src/trading_bot/plugins/filters/rr_after_costs.py`.
- **IMPLEMENT**: Registered `filter.rr-after-costs`. Params: `rr_target_min` (float, default
  `config.RR_TARGET_MIN`), `fee_pct` (default `config.FEE_PCT`), `slippage_pct` (default
  `config.SLIPPAGE_PCT`).

  The module docstring is the most important prose in the phase. It must contain:
  - The PRD requirement it implements (Success Metrics row 4: "100% of taken positions pass
    ≥1:2 R:R after costs at entry").
  - Why **net**, not gross, quoting `risk/atr_stop.py:49-58`: the gross ratio is
    dimensionless and "a setup risking 0.05% to make 0.10% scores a healthy 2.0 while
    round-trip cost … exceeds the entire reward."
  - The full `gross_rr >= X + (X+1)*c` derivation, the per-symbol table (2.213 / 2.157 /
    2.112), and `walkforward.py:53-68`'s measured minimum of 1.56 → net 1.39/1.43/1.47.
  - The **explicit statement that this floor is expected to reject most plans**, that the
    survival rate is measured by `cli graph-backtest --rr-report`, and that a near-zero
    survivor count is a finding to report — with the pre-registered decision rule reproduced
    verbatim from this plan.
  - Its own limitation, inherited from `net_rr`: **funding is not charged.**
    `risk/atr_stop.py:82-88` states it — the returned ratio *understates* the engine's
    realized cost, because holding duration is unknowable at signal time
    (`round_trip_cost_pct` docstring, `:41-42`). So the Filter is, if anything, **too
    permissive**, never too strict. Say so; do not paper over it with an assumed hold.
  - That `RR_FLOOR = 1.5` is untouched and still governs the legacy path (contract §7).

  Logic:
  ```
  net = net_rr(plan.reward_pct, plan.risk_pct, fee_pct, slippage_pct)
  c   = cost_ratio(plan.risk_pct, fee_pct, slippage_pct)
  accepted = net >= rr_target_min          # >=, boundary passes, matching setup.py:154
  return FilterVerdict(
      accepted=accepted,
      name="filter.rr-after-costs",
      reason=(f"net_rr {net:.4f} {'>=' if accepted else '<'} {rr_target_min:.2f} "
              f"(gross {plan.rr:.4f}, c {c:.4f}, gross floor needed "
              f"{rr_target_min + (rr_target_min + 1) * c:.4f})"),
      measured={"net_rr": net, "gross_rr": plan.rr, "cost_ratio": c,
                "risk_pct": plan.risk_pct, "reward_pct": plan.reward_pct,
                "gross_rr_required": rr_target_min + (rr_target_min + 1) * c},
  )
  ```
  Plus a module-level **measure-only** helper the CLI calls:
  ```python
  RR_REPORT_THRESHOLDS = (2.0, 1.75, 1.5, 1.25, 1.0)

  def rr_distribution_report(verdicts: list[FilterVerdict]) -> dict:
      """Survival counts and deciles over every plan that reached the filter.

      Returns n_plans, n_pass per threshold in RR_REPORT_THRESHOLDS,
      survival_rate at config.RR_TARGET_MIN, and min/deciles/median/max for BOTH
      net and gross R:R so the gross-vs-net gap is measured, not asserted.

      The sub-target thresholds are printed FOR DIAGNOSIS ONLY. Printing 1.5 is
      not permission to use 1.5 — see the decision rule above.
      """
  ```
- **MIRROR**: `risk/atr_stop.py:46-76` (the function being called),
  `signals/meanrev.py:233` (the existing net-R:R gate call shape),
  `signals/setup.py:154` (`>=` boundary semantics — `rr < floor` rejects, so equality passes).
- **IMPORTS**: `from trading_bot import config`;
  `from trading_bot.risk.atr_stop import cost_ratio, net_rr, round_trip_cost_pct`;
  `from trading_bot.framework.registry import register`;
  `from trading_bot.framework.contracts import FilterVerdict`.
- **GOTCHA**:
  1. Boundary is `>=`. `net_rr` returns ≤ 0 when reward does not cover cost
     (`risk/atr_stop.py:59-61`), so no separate absolute-reward floor is needed — that is
     `net_rr`'s documented design property. Do not add one.
  2. `net_rr` can return `-inf` (`:74-75`). `-inf >= 2.0` is `False`; correct, but the
     `reason` f-string must not crash formatting it. A test covers `-inf`.
  3. **Do not** compute costs yourself. `round_trip_cost_pct` is the one definition
     (contract §1: the cost model is frozen; "any new execution path charges costs
     identically or it is lying").
  4. **Do not** add an `or plan.rr >= config.RR_FLOOR` escape, a hold-duration funding
     estimate, or a "relax if too few trades" branch. Any of those makes the PRD's success
     metric unfalsifiable.
  5. `measured` is a `Mapping[str, float]` (contract §3) — floats only, no strings, no None.
- **VALIDATE**: Task 11.

### Task 11: `tests/test_plugins_filters.py`
- **ACTION**: Create `tests/test_plugins_filters.py`.
- **IMPLEMENT**:
  - **The identity test** (the phase's central arithmetic): for `c` in a spread of values,
    construct a plan with `risk_pct = r` and `reward_pct = (X + (X+1)*c) * r` and assert
    `net_rr` comes out at exactly `X` (within `1e-12`) and the verdict `accepted` is `True`
    at `rr_target_min = X`; then shave the reward by `1e-9 * r` and assert `False`.
    Parametrize over `X ∈ {1.5, 2.0}`. This pins the derivation the whole plan rests on.
  - **The recorded regression**: `risk_pct = 0.01968`, `gross_rr = 1.56` (the minimum planned
    R:R `walkforward.py:53-68` measured), `fee/slip` at config defaults → `net_rr ≈ 1.3900`
    (verified), so `accepted is False` at 2.0 **and** at 1.5. Test name/docstring cites
    `walkforward.py:53-68`, per contract §8's "regression tests pin each repaired finding".
  - Per-symbol required-gross table: assert `measured["gross_rr_required"]` equals
    2.2134 / 2.1568 / 2.1119 (±1e-3) at the three recorded median `risk_pct` values from
    `config.py:166-168`.
  - Boundary: `net_rr == rr_target_min` exactly → accepted.
  - `risk_pct` so small that `net_rr` ≤ 0 → rejected, and `measured["cost_ratio"]` is large —
    the scenario `risk/atr_stop.py:52-56` describes (0.05% risk / 0.10% reward "scores a
    healthy 2.0" gross) → assert gross ≥ 2.0 **and** rejected. This single test is the
    clearest possible demonstration of why the filter is net.
  - `risk_pct = 0` → `cost_ratio` is `inf` (`:98-99`), verdict rejected, no exception.
  - `rr_distribution_report`: empty list → `n_plans = 0` and no ZeroDivisionError;
    a hand-built list of 10 verdicts → exact `n_pass` per threshold and
    `survival_rate == n_pass(RR_TARGET_MIN)/10`.
  - **Guard test**: `config.RR_FLOOR == 1.5` and `config.RR_TARGET_MIN == 2.0` — a literal
    pin so a future edit that "harmonises" them fails loudly. Docstring: contract §7.
  - **Funding-blindness note test**: assert the module docstring mentions funding (a cheap
    `assert "funding" in module.__doc__.lower()`), so the caveat cannot be deleted silently.
- **MIRROR**: `tests/test_risk_atr_stop.py`'s style for pure-function tests (`math.isclose`,
  one assertion per behavior).
- **IMPORTS**: `math`, `pytest`, `from trading_bot import config`,
  `from trading_bot.risk.atr_stop import cost_ratio, net_rr`, the filter module,
  `from trading_bot.framework.contracts import PositionPlan`.
- **GOTCHA**: Use `1e-12` tolerance on the identity test, not `pytest.approx`'s default
  relative tolerance — the point is that the algebra is exact, and a loose tolerance would
  hide a `(X+1)` vs `X` error at small `c`.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_plugins_filters.py -v`.

### Task 12: `Trade` gains three fields; the seam populates two of them
- **ACTION**: Edit `src/trading_bot/backtest/engine.py` (`Trade`, lines 108-133) and
  `src/trading_bot/framework/execute.py` (the single `Trade(...)` construction site).
- **IMPLEMENT**: In `engine.py`, **append** to `Trade` — never reorder, never rename:
  ```python
      # Phase 4 (v0.3.0), APPENDED WITH DEFAULTS per the shared architecture
      # contract §5. Existing fields keep their positions and meanings: both
      # keyword construction sites (tests/test_backtest.py:51,
      # tests/test_equity.py:26) omit these, and engine.py's own close_out
      # leaves them at their defaults, so the legacy path is bit-identical.
      planned_rr: float = 0.0            # R:R AFTER COSTS at entry (filter.rr-after-costs'
                                         # net_rr). 0.0 on legacy-engine trades, which
                                         # never computed it. NOT the gross Signal.rr.
      confirmations: tuple[str, ...] = ()  # registry keys of Confirmations that PASSED,
                                         # sorted for determinism. Empty on the legacy path.
      strategy_version: str = ""         # set by Phase 5's version registry; Phase 4
                                         # deliberately leaves it empty.
  ```
  Extend the class docstring with a paragraph explaining `planned_rr` vs the gross ratio and
  why `confirmations` is not redundant (D8).

  In `framework/execute.py`, at the one place a `Trade` is constructed, add three kwargs:
  `planned_rr=<FilterVerdict.measured["net_rr"]>`,
  `confirmations=tuple(sorted(k for k, v in verdicts if v.passed))`, and leave
  `strategy_version` unset. Nothing else in that file changes.
- **MIRROR**: `engine.py:108-133` for the dataclass and its docstring style; `engine.py:362`
  for the keyword-construction call shape.
- **GOTCHA**:
  1. **This is the phase's only cross-ownership edit.** `framework/execute.py` is Phase 3's
     (contract §2). It is authorized because contract §5 assigns these three fields to
     Phase 4 *and* the seam is the only place their values exist — Phase 3 cannot populate
     fields that do not yet exist. Keep the edit to the three kwargs, note it in the phase
     report, and treat `test_framework_parity.py` as the acceptance check.
  2. **Parity (D6).** Once populated, a graph `Trade` is no longer `==` an
     `engine.run_backtest` `Trade` for the same strategy. Contract §5 defines parity as
     count + entry/exit timestamps + `pnl_pct`. If Phase 3's parity test compares whole
     dataclasses, narrow it to those fields and cite §5 in its docstring. Add the
     complementary regression in Task 15.
  3. `Trade` is `frozen=True`. Defaults on appended fields are what keep `make_trade` in
     `tests/test_backtest.py:51` and `tests/test_equity.py:26` green — both use keywords and
     omit the new fields. Reordering would still be forbidden: `engine.py:362` and any future
     positional or `astuple` use depend on the order.
  4. `confirmations` must be a **tuple**, not a list — `Trade` is frozen and hashability
     matters to Phase 6's result hashing.
  5. `compute_metrics` buckets on `regime`/`pattern` only, and `equity` reads named fields,
     so neither changes. Verify by running their tests, not by reasoning.
- **VALIDATE**:
  ```bash
  .venv/bin/python -m pytest tests/test_backtest.py tests/test_equity.py \
      tests/test_framework_parity.py -q      # EXPECT: all green
  .venv/bin/python -c "from trading_bot.backtest.engine import Trade; \
    import dataclasses; f=[x.name for x in dataclasses.fields(Trade)]; \
    print(f[:13]); print(f[13:])"
  # EXPECT: first 13 unchanged and in the original order; last 3 the new ones.
  ```

### Task 13: Compose and commit the thin-slice graph (and its ablation twin)
- **ACTION**: Create `data/strategies/thin-slice.strategy.json` and
  `data/strategies/thin-slice-noconfirm.strategy.json`.
- **IMPLEMENT**: Build both graphs **in Python** using Phase 3's `StrategyGraph` API and dump
  with `to_dict()` + `json.dump(..., indent=2, sort_keys=True)`. Never hand-author the JSON:
  the envelope (`SCHEMA_VERSION`, `NodeSpec` field names) is Phase 3's, and a hand-written
  file will drift from `from_dict`'s expectations.

  `thin-slice.strategy.json` nodes:

  | Kind | Registry key | Params | Why it is in the thin slice |
  |---|---|---|---|
  | data | `data.ohlcv` | tiers = config's 1d/4h/1h | Pivot guide Step 0 / PRD "OHLCV only" |
  | detector | `detector.donchian-breakout` | defaults (canonical 20/55, frozen) | **The parity anchor** — the one detector whose behavior is already measured |
  | detector | `detector.macd-cross` | defaults | Proves the graph composes >1 detector; Tier-3 reliability, no success criterion depends on it |
  | confirmation | `confirmation.volume-breakout` | defaults | Pivot guide Step 2; closes KNOWN-LIMITATIONS §0c |
  | confirmation | `confirmation.macd` | defaults | Pivot guide Step 2 |
  | policy | `policy.measured-move` | defaults | Pivot guide Step 3 |
  | filter | `filter.rr-after-costs` | defaults (`RR_TARGET_MIN = 2.0`) | Pivot guide Step 4 |

  `thin-slice-noconfirm.strategy.json`: byte-for-byte the same minus the two Confirmation
  nodes. It exists solely as Task 16's ablation baseline.
- **MIRROR**: `signals/donchian.py:25-44`'s Stated-Assumptions discipline — record, in the
  phase report, that all detector/indicator parameters are at their canonical config
  defaults and that **nothing in either graph was chosen by search**.
- **IMPORTS**: n/a (artifacts).
- **GOTCHA**:
  1. These are **committed** artifacts. Confirm `.gitignore` ignores `data/state.db*`
     (contract §6) but **not** `data/strategies/*.strategy.json`, and that `data/ohlcv.db`'s
     ignore rule does not glob them out.
  2. `sort_keys=True` + `indent=2` so a diff is reviewable; an unordered dump makes every
     regeneration look like a rewrite.
  3. Do not add a third graph "to try something". Every committed graph is a degree of
     freedom in the ledger.
- **VALIDATE**:
  ```bash
  .venv/bin/python -m trading_bot.cli graph-validate \
      --graph data/strategies/thin-slice.strategy.json      # Phase 3's validator, exit 0
  .venv/bin/python -m trading_bot.cli graph-validate \
      --graph data/strategies/thin-slice-noconfirm.strategy.json
  .venv/bin/python -m trading_bot.cli plugins | grep -E "macd-cross|volume-breakout|measured-move|rr-after-costs"
  # EXPECT: all four listed, each with a non-empty rationale (contract §12.3)
  ```

### Task 14: `cli.py` — `graph-backtest` / `_graph_backtest_command`
- **ACTION**: Edit `src/trading_bot/cli.py`.
- **IMPLEMENT**: A subparser registered **after** Phase 3's `plugins` / `graph-validate` and
  before Phase 5's, per contract §7's phase-ordering rule:
  ```
  graph-backtest --graph PATH [--symbol S]... [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                 [--audit] [--rr-report]
  ```
  and `def _graph_backtest_command(conn, symbols, *, graph_path: str, start_ms: int,
  end_ms: int, audit: bool = False, rr_report: bool = False) -> int`.

  Behavior:
  1. Load and validate the graph via Phase 3's `from_dict`; print a one-line header (name,
     schema version, node count, and the node keys grouped by kind).
  2. Per symbol: `run_graph_backtest(conn, graph, symbol, start_ms=…, end_ms=…)` →
     `compute_metrics` + `compute_equity_metrics`, printed with the **existing** `_fmt` /
     `_print_metrics` helpers and `_backtest_command`'s exact layout.
  3. `--audit`: one line per trade with `entry_ts`, direction, `pattern`, entry/stop/target,
     `risk_pct`, `reward_pct`, `planned_rr`, and `confirmations` joined by commas — the PRD's
     "every position logged against its R:R justification", satisfied by a committed command.
  4. `--rr-report`: print `rr_distribution_report`'s output per symbol and pooled.
  5. Pooled totals across symbols at the end, so the phase report needs one invocation.
  6. Return **0 always** for a successful run, zero trades included — quoting
     `_backtest_command`'s rule (`cli.py:374-377`): "a backtest with zero trades is a result,
     not an error." Return **1** only for an unreadable/invalid graph, catching Phase 3's
     `GraphError`/`ContractError` and printing `f"ERROR: {exc}"` like
     `_walkforward_command` does at `cli.py:407-409`.
- **MIRROR**: `cli.py:112-126` (subparser + repeatable `--symbol` + `_date_arg` dates),
  `cli.py:199-209` (the shared `backtest`/`walkforward` dispatch block — extend it or add a
  sibling branch in the same style), `cli.py:356-393` (`_fmt`, `_print_metrics`,
  `_backtest_command`).
- **IMPORTS**: `import json`; `from trading_bot.framework.graph import StrategyGraph` (or
  Phase 3's actual loader); `from trading_bot.framework.execute import run_graph_backtest`;
  `from trading_bot.plugins import load_all` (or `framework.registry.load_all`) — **the
  registry must be populated before a graph resolves node keys**.
- **GOTCHA**:
  1. `load_all()` must run **before** graph resolution, and its import errors are **fatal,
     never skipped** (contract §3) — "a family silently missing from a report reads as
     'tested and found wanting.'"
  2. `--graph` is **required**; there is no default graph. A default would make the audit
     trail ambiguous about which strategy produced a number.
  3. Do not touch the existing subcommands' behavior (contract §7).
  4. `conn` open/close must follow the existing pattern exactly (`connect(args.db)` →
     command → `conn.close()` → `sys.exit(exit_code)`, `cli.py:199-209`).
- **VALIDATE**:
  ```bash
  .venv/bin/python -m trading_bot.cli graph-backtest --help          # exit 0
  .venv/bin/python -m trading_bot.cli graph-backtest \
      --graph data/strategies/thin-slice.strategy.json --symbol BTCUSDT --audit
  echo "exit=$?"                                                     # EXPECT 0
  .venv/bin/python -m trading_bot.cli graph-backtest --graph /nope.json; echo "exit=$?"  # 1
  ```

### Task 15: `tests/test_pipeline_thin_slice.py` — the end-to-end proof
- **ACTION**: Create `tests/test_pipeline_thin_slice.py`.
- **IMPLEMENT**:
  - **Synthetic in-memory fixture**, seeded bar-by-bar at all three tiers with
    `START = 1_700_000_000_000` and tier constants derived from config
    (`tests/test_backtest.py:24-33`). Reuse `test_backtest.py`'s `donchian_rows()`-style
    ramp so a Donchian setup + trigger crossing is guaranteed, and give the trigger bar a
    volume ≥ `1.5 ×` the prior 20-bar mean so the volume gate can pass. Autouse fixture
    clearing `engine.clear_caches()` and any Phase 3 cache.
  - `class TestThinSliceProducesTrades`: the full graph on the fixture produces ≥ 1 trade;
    every trade has `planned_rr >= config.RR_TARGET_MIN` (**the PRD's 100%-compliance
    metric, as an assertion**), a non-empty `confirmations` tuple that is **sorted**, and
    `strategy_version == ""` (D7).
  - `class TestVolumeGateActuallyGates`: the identical fixture with the trigger bar's volume
    lowered below the threshold produces **zero** trades, and the same fixture through
    `thin-slice-noconfirm` produces ≥ 1. This is the test that proves KNOWN-LIMITATIONS §0c
    is closed — cite §0c in the docstring.
  - `class TestMacdGateActuallyGates`: same shape, sign-flipping the histogram.
  - `class TestRrFilterActuallyGates`: a fixture whose channel width makes `net_rr` land
    between 1.5 and 2.0 produces zero trades — proving the filter binds where `RR_FLOOR`
    would not. Docstring records the gross/net values.
  - `class TestParityFieldsExcluded` (D6/Task 12): the legacy-Donchian graph's trades match
    `engine.run_backtest`'s on **count, `entry_ts`, `exit_ts`, `pnl_pct`** while
    `planned_rr` / `confirmations` differ by design. Docstring cites contract §5's exact
    parity definition. This is the guard against a future "tighten parity to `==`" change
    silently making Phase 4 unmergeable.
  - `class TestGraphFilesRoundTrip`: both committed `.strategy.json` files load via
    `from_dict`, re-serialize to an identical dict (`to_dict(from_dict(x)) == x`), and
    `noconfirm` differs from the full graph **only** by the two Confirmation nodes. This
    catches hand-editing drift (Task 13's gotcha 1).
  - `class TestGraphBacktestCommand`: `_graph_backtest_command` on the fixture returns 0,
    with `--audit` and `--rr-report` both exercised; missing graph → 1. Mirrors
    `tests/test_backtest.py`'s existing CLI-command tests, which import
    `_backtest_command` / `_walkforward_command` directly (`test_backtest.py:16`).
  - `class TestZeroEngineCoreEdits`: assert `git diff --stat` (or a recorded list) shows the
    only `src/trading_bot/backtest/` change is the three appended `Trade` fields. If a
    shell-out is unpalatable in a test, encode it as a docstring checklist item in the phase
    report instead — but say which you chose and why. (PRD success metric: "New detector or
    rule added with zero engine-core edits", contract §12.4.)
- **MIRROR**: `tests/test_backtest.py` wholesale — it is the reference module (contract §8):
  tier-derived constants, autouse cache clearing, in-memory SQLite seeded bar-by-bar,
  class-per-behavior, direct CLI-handler invocation.
- **IMPORTS**: `pytest`, `pandas as pd`, `from trading_bot import config`,
  `from trading_bot.backtest import engine`, `from trading_bot.backtest.engine import
  run_backtest`, `from trading_bot.backtest.metrics import compute_metrics`,
  `from trading_bot.data import storage`, `from trading_bot.cli import
  _graph_backtest_command`, Phase 3's graph/execute/registry entry points.
- **GOTCHA**:
  1. `_assert_interval` (`engine.py:185-222`) **raises** if seeded bar spacing disagrees with
     the configured tier. Seed with `storage.TIMEFRAME_MS[...]`, never a literal — this is
     the exact trap the tier constants at `test_backtest.py:24-33` were written to avoid.
  2. The regime tier gates everything: a trending label needs `REGIME_MIN_BARS = 207` daily
     bars (`config.py:40`). `test_backtest.py`'s existing fixtures already solve this —
     reuse their approach rather than discovering it again.
  3. The volume window needs `VOLUME_LOOKBACK + 1` trigger bars before the trigger bar, and
     MACD needs `MACD_MIN_BARS = 34` setup bars. Derive both from config in the fixture.
  4. No test in this file asserts profitability. Expectancy, Sharpe and the gate are
     Phase 1/9's business; asserting a synthetic fixture is profitable is how a test becomes
     a lie.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_pipeline_thin_slice.py -v`.

### Task 16: The two measurements, on real stored history
- **ACTION**: Run the four committed commands, on the full stored span, and record every
  number. No code changes in this task.
- **IMPLEMENT**:
  ```bash
  SPAN="--start 2023-07-27 --end 2026-07-26"   # the span KNOWN-LIMITATIONS §0 uses, so the
                                               # numbers are comparable to the v0.2.0 record
  SYMS="--symbol BTCUSDT --symbol ETHUSDT --symbol SOLUSDT"

  # (1) R:R survival — run FIRST, before drawing any conclusion about the filter.
  .venv/bin/python -m trading_bot.cli graph-backtest \
      --graph data/strategies/thin-slice.strategy.json $SYMS $SPAN --rr-report

  # (2) full pipeline, with the audit trail
  .venv/bin/python -m trading_bot.cli graph-backtest \
      --graph data/strategies/thin-slice.strategy.json $SYMS $SPAN --audit

  # (3) ablation: no Confirmations (the "volume/MACD gate nothing" baseline)
  .venv/bin/python -m trading_bot.cli graph-backtest \
      --graph data/strategies/thin-slice-noconfirm.strategy.json $SYMS $SPAN

  # (4) the v0.2.0 reference point, unchanged code path, same span
  .venv/bin/python -m trading_bot.cli backtest $SYMS $SPAN
  ```
  Record in `.claude/PRPs/reports/phase4-thin-slice.md`:

  **A. R:R survival** — `n_plans`, `n_pass` at each threshold, `survival_rate` at 2.0, net
  and gross deciles, per symbol and pooled. Then state which branch of the
  [pre-registered decision rule](#pre-registered-decision-rule) the measurement lands in,
  and follow it. Also confirm-or-refute the prediction that the gross floor needed is
  ≈ 2.11–2.21 by comparing `measured["gross_rr_required"]` against the table.

  **B. Volume/MACD gate cost** — a before/after table over runs (2) vs (3):

  | | `n_trades` | `expectancy_pct` | pooled `sharpe` | pooled `ann_return_pct` | `max_drawdown_pct` |
  |---|---|---|---|---|---|
  | no Confirmations | | | | | |
  | + volume + MACD | | | | | |
  | Δ | | | | | |

  Plus, from the audit output, the distribution of `volume_ratio` (score) among accepted and
  rejected events — the graded value D3 promised to report.

  **C. Cost-frontier check** — median `risk_pct` per symbol from the audit output, and
  `cost_ratio` derived from it, asserted against `config.COST_RATIO_CEILING = 0.10`. If a
  symbol exceeds the ceiling, report it; **do not** widen `k` in response
  (`config.py:169-173`: raising ATR by coarsening the tier delivered the ceiling, "not
  widening k, which would fit the stop to the fee schedule").

  **D. Degrees of freedom consumed** (KNOWN-LIMITATIONS §9 discipline):

  | Item | Kind | Count | Selected on returns? |
  |---|---|---|---|
  | volume gate on/off | binary ablation | 2 runs | **No** — both reported |
  | MACD gate on/off | binary ablation | (same 2 runs) | **No** |
  | `VOLUME_CONFIRM_MIN_RATIO` | reused `VOLUME_HIGH_RATIO` | 0 new values | No |
  | `MACD_CONFIRM_MIN_HIST` | 0.0, pure sign test | 0 new values | No |
  | MACD 12/26/9 | canonical, inherited | 0 | No |
  | `RR_TARGET_MIN` | derived from the PRD requirement + cost algebra | 1 constant, not swept | **No** |
  | `macd-cross` `target_height` | derived (canonical 20-bar width) | 0 new params | No |
  | Total graph evaluations | 4 | — | none selected |

  Statement to include verbatim: *"No parameter in this phase was chosen by search. The four
  runs are diagnostic ablations and all four are reported, so none constitutes a hidden
  selection. Phase 1's trial ledger is not incremented, because none of these runs queried
  the gate oracle."*
- **MIRROR**: KNOWN-LIMITATIONS' own table style; contract §12.5 ("every number quoted in a
  phase report is measured by a committed command") and the repo's committed precedent
  (`git log`: "Use measured rather than derived figures in the benchmark table").
- **GOTCHA**:
  1. Run (1) **before** forming any opinion about the filter. The whole point of a
     pre-registered rule is that the measurement is not read through the lens of a desired
     conclusion.
  2. Do not report a Sharpe or DSR as a *result*. Contract §4: fitness comes only from
     `walk_forward_pooled`. These are diagnostics on stored history — label them as such,
     exactly as KNOWN-LIMITATIONS §9 requires ("none of the in-sample numbers in the review
     may be reported as results").
  3. Every figure comes from a command in this task. No spreadsheet arithmetic, no
     "approximately".
- **VALIDATE**: The report exists, every cell traces to one of the four commands, the
  decision-rule branch is named, and the DOF table is complete.

### Task 17: Full validation and phase-boundary check
- **ACTION**: Run the full suite and the compile check.
- **IMPLEMENT**:
  ```bash
  python -m py_compile \
    src/trading_bot/indicators/macd.py \
    src/trading_bot/plugins/detectors/macd_cross.py \
    src/trading_bot/plugins/confirmations/volume_breakout.py \
    src/trading_bot/plugins/confirmations/macd.py \
    src/trading_bot/plugins/policies/measured_move.py \
    src/trading_bot/plugins/filters/rr_after_costs.py \
    src/trading_bot/backtest/engine.py \
    src/trading_bot/framework/execute.py \
    src/trading_bot/cli.py \
    src/trading_bot/config.py

  .venv/bin/python -m pytest -q
  .venv/bin/python -m pytest --collect-only -q 2>&1 | tail -2
  ```
- **GOTCHA**: The **286 pre-existing tests plus every test Phase 3 added must all still be
  green** (contract §8, §12.1). A phase that breaks them is not done. Report the collected
  count. There is **no linter and no type checker** in this repo — `python -m py_compile` and
  `pytest` are the whole of "validation" (KNOWN-LIMITATIONS §8); do not invent a `mypy` or
  `ruff` step.
- **VALIDATE**: Zero failures; collected count = baseline + this phase's new tests.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| `ema` seeding | `[10,20,30,40]`, p=3 | `[nan, nan, 22.5, 31.25]` | **The** convention pin |
| `ema` ≠ SMA-seeded | same | index 2 is `22.5`, **not** `20.0` | TA-Lib/TradingView divergence |
| `ema` ≠ `wilder_smooth` | same | differ at index 2 (`22.5` vs `20.0`) | two conventions coexist |
| MACD warmup indices | 60-bar ramp, 12/26/9 | `macd` first valid idx 25; `hist` idx 33; `MACD_MIN_BARS == 34` | derived from config |
| MACD no-lookahead | truncate after bar i | values at ≤ i unchanged | causality |
| MACD window dependence | `macd(s[-60:])` vs `macd(s)[-60:]` | differ early, converge late | **silent-mismatch pin** |
| MACD normalisation | 3-bar tail | `hist == (line-sig)/close`; sign invariant | D4's premise |
| MACD empty / short | 0 bars / 10 bars | empty or all-NaN `hist`, no raise | Edge |
| `volume-breakout` boundary | ratio == 1.5 | `passed=True` | `>=` |
| `volume-breakout` below | ratio 1.49 | `passed=False`, reason names both numbers | — |
| `volume-breakout` NaN | 5 prior bars | `passed=False`, "undefined" reason | **fail closed** |
| `volume-breakout` score | any | `score == measured ratio` | graded value reported |
| `macd` confirm long/short | ±hist | mirror-image verdicts | symmetry pin |
| `macd` confirm warmup | < 34 bars | `passed=False` | fail closed |
| `macd` confirm `min_hist>0` | small ±hist | rejected on **both** sides | sign-error pin |
| Confirmations never mutate | any event | `asdict(event)` identical before/after | contract §3 |
| `measured-move` long | entry 100.3, atr 1.0, level 100, h 5 | stop 98.8, target 105.0 | — |
| `measured-move` short | mirrored | stop above entry, target = level − h | symmetry |
| `measured-move` NaN/0/neg ATR | three cases | `None` each | Edge, 3 tests |
| `measured-move` reward ≤ 0 | entry past target | `None` | Edge |
| `measured-move` ≡ `build_signal` | same numbers | identical stop/target/rr | **drift guard** |
| Filter identity | `reward = (X+(X+1)c)·risk` | `net_rr == X` to 1e-12; accept at X | **the algebra** |
| Filter recorded regression | risk 1.968%, gross 1.56 | `net_rr ≈ 1.3900`; reject at 2.0 **and** 1.5 | pins `walkforward.py:53-68` |
| Filter required-gross table | the 3 recorded `risk_pct` | 2.2134 / 2.1568 / 2.1119 | prediction pin |
| Filter dimensionless trap | risk 0.05%, reward 0.10% | gross ≥ 2.0 **and** rejected | why net, in one test |
| Filter `risk_pct = 0` | 0 | `cost_ratio = inf`, rejected, no raise | Edge |
| Filter `-inf` net_rr | negative costs | rejected, reason formats | Edge |
| `RR_FLOOR` / `RR_TARGET_MIN` guard | config | 1.5 / 2.0 | contract §7 |
| Funding caveat present | module docstring | mentions "funding" | can't be deleted silently |
| `rr_distribution_report` empty | `[]` | `n_plans=0`, no ZeroDivisionError | Edge |
| `Trade` field order | `dataclasses.fields` | first 13 unchanged, 3 appended | contract §5 |
| Thin slice produces trades | synthetic fixture | ≥1 trade, all `planned_rr >= 2.0`, sorted `confirmations`, `strategy_version == ""` | PRD 100% metric |
| Volume gate binds | low-volume fixture | 0 trades; `noconfirm` gives ≥1 | closes §0c |
| MACD gate binds | flipped-hist fixture | 0 trades | — |
| R:R filter binds | net between 1.5 and 2.0 | 0 trades | binds where `RR_FLOOR` wouldn't |
| Parity fields excluded | legacy graph vs `run_backtest` | count/ts/`pnl_pct` equal; new fields differ | contract §5 / D6 |
| Graph files round-trip | both `.strategy.json` | `to_dict(from_dict(x)) == x`; noconfirm differs by 2 nodes | hand-edit drift |
| `graph-backtest` handler | fixture; bad path | 0 (incl. zero trades); 1 | `cli.py:374-377` rule |

### Edge Cases Checklist
- [x] Empty input (empty frame → empty MACD; empty verdict list → report with `n_plans=0`)
- [x] Warmup / insufficient bars (MACD < 34; volume window < 20+1; Donchian < 21)
- [x] NaN indicator values — **fail closed** everywhere (`not (x > 0)` / `math.isnan`)
- [x] Boundary equality (`net_rr == 2.0` accepts; `ratio == 1.5` accepts)
- [x] Long/short symmetry (MACD confirmation, policy stop/target)
- [x] Degenerate geometry (zero/negative channel width, `risk_pct = 0`, reward ≤ 0)
- [x] `-inf` from `net_rr`
- [x] Zero trades is a valid result, not an error
- [x] Window-vs-full-series indicator divergence (EMA is not window-invariant)
- [x] Tier/interval mismatch (`_assert_interval` raises — fixtures derive from config)
- [ ] Concurrent access — N/A (pure functions; `storage._db_lock` unchanged; single-threaded)
- [ ] Network failure — N/A (no network path in this phase)
- [ ] Permission denied — N/A

---

## Validation Commands

### Static Analysis
```bash
python -m py_compile \
  src/trading_bot/indicators/macd.py \
  src/trading_bot/plugins/detectors/macd_cross.py \
  src/trading_bot/plugins/confirmations/volume_breakout.py \
  src/trading_bot/plugins/confirmations/macd.py \
  src/trading_bot/plugins/policies/measured_move.py \
  src/trading_bot/plugins/filters/rr_after_costs.py \
  src/trading_bot/backtest/engine.py \
  src/trading_bot/framework/execute.py \
  src/trading_bot/cli.py src/trading_bot/config.py
```
EXPECT: zero syntax errors. **No type checker and no linter are configured in this repo**
(KNOWN-LIMITATIONS §8, contract §0). `py_compile` + `pytest` is the whole validation surface;
do not invent a `mypy`/`ruff` step.

### Unit Tests (this phase)
```bash
.venv/bin/python -m pytest tests/test_macd.py tests/test_plugins_confirmations.py \
  tests/test_plugins_policies.py tests/test_plugins_filters.py \
  tests/test_pipeline_thin_slice.py -v
```
EXPECT: all pass.

### Regression — the phase-boundary gate
```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest --collect-only -q 2>&1 | tail -2
```
EXPECT: zero failures. **286 pre-existing tests** (verified 2026-07-27) plus every test
Phase 3 added, all still green, plus this phase's new ones. Report the collected count
(contract §8, §12.1).

### Frozen-surface check
```bash
git diff --stat src/trading_bot/
# EXPECT changed: config.py (additions only), backtest/engine.py (3 appended Trade
# fields + docstring), framework/execute.py (3 kwargs at one call site), cli.py
# (one subparser + one handler). Anything else in signals/, risk/, regime/, data/,
# backtest/metrics.py, backtest/equity.py, backtest/walkforward.py, indicators/wilder.py
# is OUT OF SCOPE for this phase — revert it.

grep -n "RR_FLOOR" src/trading_bot/config.py
# EXPECT: exactly the original line 178 (RR_FLOOR = 1.5), unmodified.
```

### Plug-in registration (contract §12.3)
```bash
.venv/bin/python -m trading_bot.cli plugins
```
EXPECT: `detector.macd-cross`, `confirmation.volume-breakout`, `confirmation.macd`,
`policy.measured-move`, `filter.rr-after-costs` all listed, each with a non-empty rationale.

### Graph validation
```bash
.venv/bin/python -m trading_bot.cli graph-validate --graph data/strategies/thin-slice.strategy.json
.venv/bin/python -m trading_bot.cli graph-validate --graph data/strategies/thin-slice-noconfirm.strategy.json
```
EXPECT: exit 0 both.

### Manual Validation — the phase's own success signal
- [ ] `graph-backtest --graph …/thin-slice.strategy.json --symbol BTCUSDT --symbol ETHUSDT
      --symbol SOLUSDT --start 2023-07-27 --end 2026-07-26 --audit` runs on **3 symbols** and
      prints, for every taken position, its entry/stop/target and its `planned_rr` — the PRD's
      "every position logged against its R:R justification".
- [ ] **100% of taken positions have `planned_rr >= config.RR_TARGET_MIN`.** Verify from the
      audit output, not from the code (PRD Success Metrics row 4).
- [ ] `--rr-report` produces the survival table, and the report names which decision-rule
      branch the measurement landed in.
- [ ] The volume/MACD ablation table (Task 16 B) is filled in with measured numbers.
- [ ] Median `risk_pct` and `cost_ratio` per symbol are reported against
      `COST_RATIO_CEILING = 0.10`.
- [ ] Adding these five plug-ins required **zero** edits to `backtest/engine.py` beyond the
      three appended `Trade` fields, and **zero** edits to any `signals/*` module — verified
      by `git diff --stat` (contract §12.4, PRD "Iteration cost" metric).
- [ ] `.claude/PRPs/reports/phase4-thin-slice.md` exists with the DOF ledger.

---

## Acceptance Criteria
- [ ] All 18 tasks (0–17) completed
- [ ] All validation commands pass
- [ ] 286 pre-existing tests + Phase 3's + this phase's new tests all green; count reported
- [ ] `python -m py_compile` clean
- [ ] No type errors — **N/A, no type checker configured** (say so; do not invent one)
- [ ] No lint errors — **N/A, no linter configured**
- [ ] Matches the UX design above (`graph-backtest` output shape, `--audit` lines)
- [ ] `RR_FLOOR` unmodified; `RR_TARGET_MIN` at 2.0; no parameter selected on returns
- [ ] The R:R survival measurement was run **before** any conclusion about the filter, and
      the pre-registered decision rule was followed and named in the report

## Completion Checklist
- [ ] Code follows discovered patterns (config-defaulted keyword params, `reject()` DEBUG
      logging, frozen dataclasses, `not (x > 0)` NaN guards, derived warmup constants)
- [ ] Error handling matches codebase style — verdicts and `None`, never exceptions for a
      failed gate; `ValueError`-style messages only for config/data disagreement
- [ ] Logging follows conventions (`logging.getLogger("trading_bot")`, DEBUG for rejections)
- [ ] Tests follow `tests/test_backtest.py` conventions (tier constants from config, autouse
      cache clearing, in-memory SQLite, hand-computed numerics with the arithmetic in
      comments, findings pinned by id in the test name/docstring)
- [ ] No hardcoded values — every threshold in `config.py` under a reserved prefix
- [ ] `indicators/macd.py` credits `scripts/bruteforce/indicators.py:160-172` and `:79-82`
      and states its EMA seeding convention explicitly
- [ ] No unnecessary scope additions (see NOT Building; `rsi.py` and extra detectors are
      Phase 8's)
- [ ] Contract §12.5 satisfied: every number in the phase report is produced by a committed
      command
- [ ] Contract §12.6 satisfied: the DOF ledger is written
- [ ] Self-contained — no questions needed during implementation

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **`RR_TARGET_MIN = 2.0` net rejects nearly everything.** Min gross R:R ever taken was 1.56 (`walkforward.py:53-68`) → net 1.39–1.47; net 2.0 needs gross ≈ 2.11–2.21 | **High — predicted, not feared** | High (thin slice may produce ~0 trades) | Predicted in the plan; measured by `--rr-report` **before** any tuning; pre-registered decision rule with three non-weakening remedies; weakening the filter is forbidden without a logged user decision |
| Volume hard gate removes trades and worsens expectancy | Medium-High | Medium | Measured as a before/after ablation on committed graphs (Task 16 B); recorded as a consumed DOF; both runs reported, so an unfavourable result cannot be quietly dropped |
| MACD EMA seeding silently differs from TradingView/TA-Lib and a future reader "fixes" it | **High** (this is the classic MACD bug) | High (silent signal change) | Convention stated in the module docstring naming all three conventions; three dedicated tests including an explicit "not `20.0`" assertion; `wilder_smooth`-differs test |
| MACD is not window-invariant; a future perf optimization hoists it out of the per-window computation and changes every signal | Medium | High | `TestWindowDependence` pins the divergence and the docstring states the convention (always compute on the caller's window), exactly as Wilder ADX is already computed per window at `donchian.py:103` |
| `framework/execute.py` edit collides with Phase 3's ownership or its parity test | Medium | High (Phase 4 unmergeable) | Task 0 checks parity's comparison basis first; D6 narrows it to contract §5's named fields; the edit is 3 kwargs at 1 site; `TestParityFieldsExcluded` guards it permanently |
| Gross vs net `rr` confusion — `PositionPlan.rr` (gross) vs `Trade.planned_rr` (net) | Medium | Medium-High | Named and explained in both docstrings, in Task 8 GOTCHA 1, and in the `Trade` field comment; the Filter's `reason` prints both |
| Policy silently re-applies `RR_FLOOR` gross before the net Filter, truncating the measured distribution | Medium | High (makes `--rr-report` a report on a pre-filtered sample) | Task 8 GOTCHA 2 forbids it; `TestRrFilterActuallyGates` uses a fixture between 1.5 and 2.0, which would produce a trade if the policy pre-filtered and zero if it does not |
| Appending `Trade` fields breaks something that compares whole `Trade`s | Medium | High (286 tests) | Both construction sites verified keyword-only and field-omitting; defaults keep them green; field-order assertion; run `test_backtest.py`/`test_equity.py` immediately after the edit |
| `EvalContext` cannot supply a setup-tier window aligned to the trigger bar the way `engine.py:481` does | Medium | High | Task 0 verifies it before any plug-in is written; if absent it is a Phase 3 gap to report, **not** to work around by reaching into `storage` (which would bypass `_assert_interval`) |
| The composed pipeline produces zero trades on real data and the phase looks failed | Medium | Medium | Contract §12 and `cli.py:374-377` both treat zero trades as a result; the synthetic fixture proves mechanics independently of real-data outcomes; the report distinguishes "pipeline broken" from "pipeline works, filter binds" |
| `macd-cross` has no edge (Tier-3, ⭐⭐⭐☆☆) | High | Low | Contract §9: tiers 3–4 are "a direction, not a gate"; **no success criterion in this phase depends on `macd-cross` having edge**. It is here to prove multi-detector composition |
| Scope creep into Phase 8 detectors or Phase 5/6 machinery | Medium | Medium | NOT Building is explicit; contract §2 file ownership is the arbiter |
| `cost_ratio` exceeds `COST_RATIO_CEILING` on some symbol under the new pipeline | Low-Medium | Medium | Measured and reported (Task 16 C); the response is a report, never a wider `k` (`config.py:169-173`) |

## Notes

- **Where the PRD and the contract disagree, the contract wins** (per instruction). Two
  disagreements are load-bearing here:
  1. The PRD's Phase 4 row says "backtest parity with v0.2.0 harness." **Contract §5 assigns
     parity to Phase 3** (`tests/test_framework_parity.py`) and says Phase 4 "may only extend
     from a green parity test." This plan therefore *verifies* parity (Task 0) and *protects*
     it (D6, Task 15) rather than establishing it.
  2. The v0.2.0 PRD intended volume as "graded confidence" and `breakout.py:19-23` /
     `config.py:74` document it that way. The v0.3.0 PRD's Must row asks for volume as a
     *confirmation*. This plan picks the **hard gate** (D3), states the rejected alternative
     and why (no sizing for a weight to act through, contract §10), and leaves the legacy
     `signals/*` path's documented behavior untouched.
- **This phase is where the entry finally changes.** KNOWN-LIMITATIONS §0 is blunt: "a 15
  basis-point gross edge is noise — the *entry* carries almost no predictive content," and
  §0c records that no sweep ever touched the entry or the feature set. Volume gating and MACD
  are the first entry-side changes in the project's history. They may well not help. The
  point of this phase is that they are now **expressible and measurable**, which is the
  framework's whole thesis — and per the PRD's honesty clause, the framework's value survives
  a negative result.
- **The R:R filter is the phase's honest core.** It is derived from cost algebra
  (`gross ≥ X + (X+1)c`), not fitted to returns — the same discipline `config.py:162-174`
  records for `k = 1.5`, which was "confirmed against the cost constraint alone, never from
  PnL." If it turns out that ≥1:2-after-costs is unreachable with a channel-width target and
  a 1.5·ATR stop, that is a **finding about this strategy family**, cheaply obtained, and
  exactly what the PRD says a good outcome can look like.
- **Two smoothing conventions now coexist** in `indicators/`: `wilder_smooth` (SMA seed, 1/N
  decay) and `ema` (first-observation seed, 2/(N+1) decay, `min_periods` mask). That is
  correct — Wilder's indicators are defined on RMA and MACD on EMA — but it is a trap for
  anyone adding an indicator later. Both module docstrings must cross-reference each other.
- **Funding remains uncharged at signal time**, by design and by necessity: hold duration is
  unknowable at entry (`risk/atr_stop.py:41-42`), and `cost_ratio`'s docstring already states
  the returned `c` "UNDERSTATES the engine's realized cost." The engine still charges funding
  on the realized trade (`engine.py:359-360, 374`). So the Filter is slightly permissive;
  saying so is the honest posture, and inventing an assumed hold to close the gap would be a
  new fitted parameter.
- **Nothing here queries the gate.** Contract §4: `walk_forward_pooled` is the only fitness
  oracle, Phase 1 owns its extension, and Phase 6's oracle wrapper cannot bypass the trial
  ledger. Phase 4 produces `Trade` lists and diagnostics. No number from this phase may be
  reported as a validated result.
