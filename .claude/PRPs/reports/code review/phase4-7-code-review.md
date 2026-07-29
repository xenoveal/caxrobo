# Code Review: Phases 4-7 — hunting the missing performance

**Reviewed:** 2026-07-27
**Branch:** `feat/phase4-tier-shift` (HEAD `b0f8b57`, plus uncommitted Phase 5/6/7 work)
**Scope:** Phase 4 (tier shift), Phase 5 (Donchian trend engine), Phase 6 (fade
re-qualification), Phase 7 (walk-forward repair gate)
**Question asked:** is there a calculation bug or strategy miscalculation that,
once fixed, improves overall performance?
**Decision:** REQUEST CHANGES — two HIGH defects are individually measurable and
additive; a third makes THE GATE unable to detect either.

## Summary

The strategy is not badly broken; it is sitting on exactly zero. Measured over
full stored history, 3 symbols pooled, current code:

| Metric | Value |
|---|---|
| Trades | 183 over 1056 days |
| Win rate | 41.0% |
| Profit factor | **1.003** |
| Sharpe (annualised) | **+0.007** |
| Annualised return | **−4.9%** |
| Equity max drawdown | 37.8% |
| Mean **gross** P&L / trade | **+0.150%** |
| Mean all-in cost / trade | **+0.146%** (0.140% fee+slippage, 0.006% funding) |

Costs consume **97% of the gross edge**. The gross edge is thin because two
design defects truncate it, and a third prevents the validation harness from
seeing either. Fixing the two defects moves the same 3-symbol history from
−4.9% to **+11.0%** annualised and Sharpe +0.007 → **+0.431**.

Everything below is measured on the tuning span with the engine's own cost
model. **None of it is validated.** It is diagnostic evidence that two specific
lines of code are wrong, not a claim that the strategy now works. See
"Degrees of freedom consumed".

## Findings

### HIGH-1 — The regime classifier ranks ATR in absolute price units

`src/trading_bot/regime/classifier.py:99-103`

```python
atr_vals = wilder.atr(df, period=adx_period)
atr_percentile = _atr_percentile_rank(atr_vals, atr_percentile_window)
```

`_atr_percentile_rank` ranks each bar's ATR against the trailing 180 **raw ATR
values, in quote-currency units**. Over a 180-day window in which price trends,
absolute ATR trends with the price level regardless of whether *relative*
volatility changed at all. A symbol grinding steadily higher therefore sits
near its own 180-day ATR maximum more or less permanently, gets
`atr_percentile >= 0.90`, and is labelled `extreme-volatility` — which
suppresses all trading.

The percentile is meant to answer "is volatility unusually high right now?".
Ranking price units answers "has price risen since 180 days ago?".

**Measured.** Share of days labelled `extreme-volatility`, raw vs `ATR/close`:

| Symbol | raw ATR | ATR/close |
|---|---|---|
| BTCUSDT | 23.1% | 17.4% |
| ETHUSDT | 20.3% | 18.8% |
| SOLUSDT | 20.5% | 13.3% |

**Effect on the trend engine** (variant G, normalise only, nothing else
changed): trades 183 → 206, Sharpe +0.007 → **+0.344**, annualised −4.9% →
**+6.4%**, max DD 37.8% → 34.9%. More trades *and* better trades — the
suppressed days were disproportionately good ones.

**Fix:** rank `wilder.atr(df, period) / df["close"]`. This is a bug fix, not a
tuning change: it consumes no degree of freedom, because 0.90 and 180 are
unchanged and the quantity being ranked becomes the dimensionless one the
docstring already describes.

### HIGH-2 — The ATR ratchet trail truncates the trend edge it exists to capture

`src/trading_bot/backtest/engine.py:350-363`

The Phase 5 trail ratchets the stop to `extreme − 1.5 × ATR(4H)` and is tested
against every 1H bar. `ATR_STOP_MULTIPLE` is reused for both the initial stop
and the trail, so the trail is as tight as the entry stop.

**Measured.** 167 of 183 exits (**91.3%**) are the trail, mean **−0.43%**,
median hold **10 hours** — on a system whose regime tier is daily and setup tier
4H. The 12 trades that survive to target average **+7.15%**. A daily-regime
trend follower is being closed out inside half a day.

I verified this is not a labelling artifact: **0 of the 167** `trail` exits left
at the untouched initial stop — all 167 were genuinely ratcheted exits. The
trail is doing this, not the entry stop.

**Corollary — Phase 5's opposite-channel exit is dead code.** Variant E
(channel exit removed entirely) is *bit-identical* to baseline: 183 trades,
Sharpe +0.007, same outcome histogram. The channel exit records **0 exits**
because the trail always fires first. The headline exit mechanism Phase 5 was
built to add has never once bound. It only comes alive when the trail is
removed (9 channel exits in variants B and I).

**The fixed target compounds it.** `target = level + channel_width` caps the
right tail of a trend system at roughly one 20-bar channel width. Removing the
target alone (variant D) lifts Sharpe +0.007 → +0.182.

**Measured, exit logic only** (regime untouched):

| Variant | n | PF | Sharpe | Ann | Med hold |
|---|---|---|---|---|---|
| A baseline | 183 | 1.003 | +0.007 | −4.9% | 10h |
| B no trail | 143 | 1.043 | +0.107 | −3.7% | 26h |
| C trail 3.0× | 148 | 0.973 | −0.070 | −9.5% | 24h |
| D no target | 182 | 1.077 | +0.182 | +0.4% | 10h |
| E no channel exit | 183 | 1.003 | +0.007 | −4.9% | 10h |
| F trail 3.0 + no target | 139 | 0.974 | −0.063 | −9.1% | 30h |
| I no trail + no target | 127 | 1.105 | +0.207 | +0.1% | 61h |

Note C and F: *widening* the trail to 3.0× is worse than either 1.5× or none.
The trail is not mis-parameterised, it is structurally wrong for this system —
do not try to tune it. Removing it (and the target) reproduces the canonical
Turtle shape: fixed initial stop, opposite-channel exit, time stop.

**Combined with HIGH-1** (variant J — no trail, no target, normalised regime
ATR): n=142, PF **1.223**, Sharpe **+0.431**, annualised **+11.0%**, median hold
52h. The two fixes are additive.

**Caveat, stated plainly:** J's max drawdown is **51.2%**, worse than
baseline's 37.8% and far outside `GATE_MAX_DRAWDOWN = 0.25`. Removing the trail
buys expectancy with drawdown. That trade is exactly what Phase 8's
vol-targeted sizing is supposed to arbitrate, and it is a real cost, not a
rounding error.

### HIGH-3 — THE GATE sweeps two levers that provably do not bind

`src/trading_bot/backtest/walkforward.py:51-54`

```python
DEFAULT_GRID = {
    "rr_floor": (1.25, 1.5, 1.75, 2.0),
    "max_hold_bars": (48, 96, 144),
}
```

Measured against baseline trades:

| Axis value | Effect |
|---|---|
| `rr_floor=1.25` | 183/183 trades clear it (100%) |
| `rr_floor=1.5` | 183/183 clear it (100%) |
| `rr_floor=1.75` | 177/183 (96.7%) |
| `rr_floor=2.0` | 171/183 (93.4%) |
| `max_hold_bars=48` | 13/183 trades reach it (7.1%) |
| `max_hold_bars=96` | **0/183** (0.0%) |
| `max_hold_bars=144` | **0/183** (0.0%) |

Planned R:R distribution: min **1.56**, p25 2.61, median 3.37, max 5.29. The
minimum planned R:R across every trade the engine takes is 1.56 — above the
`RR_FLOOR = 1.5` default. The floor is not merely weak, it is *unreachable* at
its configured value, because the Donchian target is a full 20-bar channel
width against a 1.5×ATR stop.

Consequence: of 12 grid combos, 10 produce identical trade lists. Fold winners
are chosen by `max()` over tied expectancies — i.e. by `itertools.product`
ordering. Meanwhile `n_trials = len(combos) * len(folds)` deflates the DSR for
12 configurations per fold that were never meaningfully distinct. THE GATE
pays the full multiple-testing penalty for a search it did not perform, and
cannot reward a fix to HIGH-1 or HIGH-2 because neither is on an axis.

**Fix:** the grid must sweep the levers that bind. On this evidence those are
the exit mode (trail on/off, target on/off), the trail multiple as an axis
*separate* from the entry stop multiple, and the regime `atr_extreme_percentile`
gate. Drop `max_hold_bars` (0% binding at default) and either drop `rr_floor`
or move its range above 1.56 where it starts to bite.

### MEDIUM-1 — `extreme-volatility` unconditionally overrides `trending`

`src/trading_bot/regime/classifier.py:109-114`. Precedence puts the volatility
veto above the trend label, so a strongly trending, high-volatility day trades
nothing. For a mean-reversion system that is prudent; for a Donchian breakout
engine it removes the days the method exists to trade.

**Measured** share of all `ADX >= 25` days masked as `extreme-volatility`:

| Symbol | ADX≥25 days | masked (raw ATR) | masked (normalised) |
|---|---|---|---|
| BTCUSDT | 543 | 189 (34.8%) | 140 (25.8%) |
| ETHUSDT | 668 | 163 (24.4%) | 156 (23.4%) |
| SOLUSDT | 550 | 169 (30.7%) | 128 (23.3%) |

Fixing HIGH-1 reduces this but does not remove it: roughly a quarter of trending
days stay suppressed. Whether the veto should outrank `trending` at all is a
strategy decision worth an explicit trial, not an accident of rule ordering.

### MEDIUM-2 — 25% of trigger bars can never fire (`h_idx`/`end_ts` off-by-one)

`engine.py:369` selects `h_idx` as the last setup bar closed by the trigger
bar's *close*, while `breakout.py:113` rejects any trigger bar whose *open*
predates that setup bar's close. On the last trigger bar of each setup window
those two rules select different setup bars, and the bar is silently dropped.
Reproduced exactly with the engine's own arithmetic (4H setup / 1H trigger):

```
trigger ts=11:00 -> h_idx=0 (setup closing 12:00)  eligible=False   <- correct
trigger ts=12:00 -> h_idx=0 (setup closing 12:00)  eligible=True
trigger ts=13:00 -> h_idx=0 (setup closing 12:00)  eligible=True
trigger ts=14:00 -> h_idx=0 (setup closing 12:00)  eligible=True
trigger ts=15:00 -> h_idx=1 (setup closing 16:00)  eligible=False   <- dropped
```

Only 3 of the 4 hourly bars following each 4H close can ever trigger. A fresh
crossing in the final hour of every 4H window is discarded, and by the next bar
the candidate's level has moved. This is lost opportunity, not lookahead — the
no-lookahead guarantees hold — but it costs ~25% of trigger chances in a system
already starved of samples (183 trades / 1056 days).

### MEDIUM-3 — The freshness rule rejects the cleanest breakouts

`breakout.py:125-127` requires `closes[i-1] <= level`. When the setup bar's own
close is already beyond its (current-bar-excluded) channel level, the 1H bar
ending that setup bar also closed beyond it, so no eligible trigger bar can
present a fresh crossing unless price first retreats inside the channel.

**Measured:** 9.7% / 10.4% / 9.9% of long setups (BTC/ETH/SOL) and 7.2% / 6.3% /
8.0% of shorts are already beyond the level at the setup close. Those are the
decisive breakouts; the rule structurally prefers hesitant ones. Smaller than I
first suspected, but it is a systematic adverse selection, and it interacts with
MEDIUM-2.

### MEDIUM-4 — Off-grid final parameters in THE GATE

`walkforward.py:336-338` takes `statistics.median` per axis over fold winners.
With an even number of folds this interpolates to values no fold ever
evaluated:

```
[1.25, 2.0] -> 1.625  (not in grid)
[48, 96]    -> 72.0   (not in grid)
```

The one-shot OOS — the entire point of the protocol — can therefore be run at a
configuration that was never validated on any train window. Use
`statistics.median_low` (or the mode of fold winners) to stay on-grid, and add a
test asserting `final_combo[axis] in grid[axis]`.

### MEDIUM-5 — Daily returns concentrate each trade's whole P&L on its exit day

`backtest/equity.py:44-48`. A trade held 52 hours contributes its entire
`pnl_pct` to one UTC day. This leaves the mean daily return correct but inflates
daily standard deviation, so **Sharpe is biased downward** — and Sharpe is the
gate. The behaviour is documented in the module docstring as a deliberate
choice, so this is a calibration warning rather than a defect: the gate
threshold `GATE_MIN_SHARPE = 1.0` is being applied to a systematically
understated statistic. Either spread each trade's P&L across the days it was
open, or lower the threshold to match the estimator actually in use.

### LOW findings

1. **Selection metric contradicts the north star** (`walkforward.py:308`): fold
   winners are chosen by pooled per-trade `expectancy_pct`, while the gate scores
   Sharpe. Per-trade expectancy is maximised by taking fewer, larger-variance
   trades — the opposite of what improves Sharpe. Select on the metric you gate on.
2. **`pnl == 0` counted as a loss** (`metrics.py:53`): `losses = [p for p in pnls if p <= 0]`
   puts exact zeros in the loss bucket, mildly depressing win rate and profit factor.
3. **Walk-forward runtime is impractical** (`indicators/wilder.py:86-90`):
   `wilder_smooth` is a per-bar Python loop, and `adx` invokes it 5×. One
   3-symbol full-history backtest takes ~4 minutes; a 17-fold pooled
   walk-forward needs ~660 `run_backtest` calls, i.e. well over 12 hours. This
   is why the gate is hard to iterate on. Vectorising the recursion (or caching
   indicators per symbol/timeframe across combos, since only `rr_floor` and
   `max_hold_bars` vary and neither touches the indicators) would cut this by
   orders of magnitude.
4. **Docstring rot** (`wilder.py:36`): `true_range` claims "First row is NaN".
   `pd.concat(...).max(axis=1)` skips NaN, so row 0 returns `high - low`. ATR is
   therefore seeded one bar earlier than documented.

## Recommended order of work

1. Fix HIGH-1 (`ATR/close` in the percentile rank). One line, it is a genuine
   bug, and it is the single largest measured improvement (+0.34 Sharpe).
2. Fix HIGH-3 (repair the grid) **before** touching HIGH-2, so the exit-mode
   change is decided by the walk-forward rather than by my in-sample sweep.
3. Address HIGH-2 as a pre-registered grid axis: split the trail multiple from
   the entry-stop multiple, and put trail-on/off and target-on/off on the grid.
   Do not simply hard-code variant J — its 51.2% drawdown needs the gate's
   verdict, not mine.
4. MEDIUM-4 and the LOW-3 performance work, which together make the gate
   runnable and trustworthy.
5. MEDIUM-2 and MEDIUM-3 as sample-recovery work; both add trades without
   adding parameters.
6. Revisit MEDIUM-1 (regime precedence) as an explicit logged trial.

## Degrees of freedom consumed by this review

Per the PRD's trial-log discipline, this review evaluated **10 engine variants**
on the tuning span. That is 10 configurations of search that must be counted in
any future `n_trials` for the DSR, and it is why none of the numbers above may
be reported as a result. The honest next step is a walk-forward run with the
repaired grid, scored on the untouched `WF_OOS_DAYS` holdout.

## Validation

| Check | Result |
|---|---|
| Tests (`pytest`) | **Pass** — 270 passed, 1 skipped |
| Type check | Skipped — no type checker configured |
| Lint | Skipped — no linter configured |
| Build | N/A |

Two coverage gaps worth closing: no test asserts the walk-forward's final
parameters land on the grid (MEDIUM-4), and no test would have caught that the
Phase 5 opposite-channel exit never fires in a realistic run (HIGH-2) —
`test_opposite_channel_touch_exits` passes on a fixture built so the trail
cannot preempt it.

## Files reviewed

| File | Phase | Change |
|---|---|---|
| `src/trading_bot/config.py` | 4/5/6/7 | Modified |
| `src/trading_bot/backtest/engine.py` | 4/5 | Modified |
| `src/trading_bot/backtest/walkforward.py` | 7 | Modified |
| `src/trading_bot/backtest/equity.py` | 3 | Read (gate dependency) |
| `src/trading_bot/backtest/metrics.py` | 3 | Read (gate dependency) |
| `src/trading_bot/signals/donchian.py` | 5 | Added |
| `src/trading_bot/indicators/donchian.py` | 5 | Added |
| `src/trading_bot/signals/scan.py` | 5/6 | Modified |
| `src/trading_bot/signals/setup.py` | 4/5 | Modified |
| `src/trading_bot/signals/breakout.py` | 4 | Read |
| `src/trading_bot/signals/meanrev.py` | 6 | Read |
| `src/trading_bot/regime/classifier.py` | 4 | Read |
| `src/trading_bot/indicators/wilder.py` | — | Read |
| `src/trading_bot/risk/atr_stop.py` | 2 | Read |
| `src/trading_bot/cli.py` | 4/7 | Modified |

---

# Addendum: first walk-forward run (2026-07-27)

Run after the repairs above, `--start 2023-07-27` (post-1D-warmup), 3 symbols
pooled. **Runtime 1m47s**, down from an estimated ~3.6 hours, because the
engine now memoizes per-(symbol, tier) indicators across grid combos — the grid
varies only exit parameters, which touch no indicator (LOW-3, now fixed).

## Verdict: GATE FAIL

| Gate condition | Value | Threshold | Result |
|---|---|---|---|
| Sample adequacy | 23 trades | >= 30 | **FAIL** |
| Sharpe (annualised) | **1.17** | >= 1.0 | PASS |
| DSR | **0.0210** | > 0.95 | **FAIL** |
| Equity max drawdown | 19.13% | <= 25% | PASS |
| Per-symbol expectancy | +0.23% / +0.80% / +1.53% | all > 0 | PASS (3 of 3) |

Selected parameters: `trail_enabled=False`, `target_enabled=False`,
`max_hold_bars=48`. The grid chose BOTH exit-mode repairs on its own, on train
data, without being told which way the in-sample sweep had pointed.

One-shot OOS (90-day holdout, untouched by tuning): 23 trades, win rate 43.5%,
expectancy +0.718%/trade, profit factor 1.51, Sortino 2.47, annualised +67.1%.

## The two failures are one failure: the sample is too small to prove anything

The point estimates are the best this project has produced. They also cannot be
distinguished from luck, and the DSR is not being unfair about it:

| `n_trials` convention | Expected max SR | DSR |
|---|---|---|
| 156 (combos x folds — what the code uses) | 0.2537 | 0.0210 |
| 12 (distinct configurations only) | 0.1574 | 0.1552 |
| **1 (no multiple-testing penalty at all)** | 0.0000 | **0.7423** |

Even with the multiple-testing correction switched off entirely, DSR is 0.74
against a 0.95 bar. **No choice of `n_trials` rescues this**, so there is no
point relitigating that convention — the constraint is the data, not the
penalty.

The mechanism is visible in the daily return moments: **skew 3.79, kurtosis
31.24** over **90 observations**. PSR's denominator carries
`((kurt - 1) / 4) * sr^2`, so a kurtosis of 31 inflates the standard error of
the Sharpe estimate enormously. The holdout's +67% annualised rests on a
handful of days.

## This promotes MEDIUM-5 from a footnote to the binding constraint

That kurtosis is substantially manufactured by the metric, not by the market.
`equity.daily_returns` books each trade's entire `pnl_pct` on its exit day, so
23 trades across 90 days produce ~67 exact-zero days and a few very large ones
— a distribution engineered to look fat-tailed. Spreading each trade's P&L
across the days it was actually open would cut skew and kurtosis sharply, raise
the daily Sharpe estimate's precision, and is the methodologically correct
attribution regardless. It is now the highest-leverage single change available,
ahead of any further strategy work.

## Recommended next steps, in order

1. **Fix MEDIUM-5** (spread P&L across holding days). Cheap, correct, and it
   directly attacks the kurtosis that is blocking every significance test.
2. **Raise the sample.** 23 OOS trades cannot clear `WF_MIN_TRADES = 30` at any
   Sharpe. Only 3 symbols exist in the DB; pooling more is the PRD's stated
   "only free lunch" and costs zero degrees of freedom. This means backfilling
   more pairs — re-ping `fapi.binance.com` first (it was blocked 2026-07-05,
   reachable again 2026-07-26).
3. **Do NOT** lower `WF_MIN_TRADES` or shrink the OOS window to manufacture a
   pass. Both spend the credibility the gate exists to protect.
4. Only then revisit MEDIUM-1 / MEDIUM-3 as logged strategy trials.

## Fold-level health, recorded honestly

13 folds. **4 of 13** (folds 7, 9, 11, 12) had no combo reach `WF_MIN_TRADES`
on their 180-day train window and fell back to config defaults — the train
windows are themselves undersampled. Test-window expectancy was negative in
**10 of 13** folds. So the fold machinery is noisy and the favourable holdout
should not be read as a stable edge; it is one 90-day window that went well.
