# Known limitations — v0.2.0

**Status: NOT VALIDATED. Do not commit capital on the basis of this build.**

This file is the honest accounting for the version merged to `master` on
2026-07-27 (Phases 1–7). It exists because the build was finalised *knowing* the
validation gate rejects it — a deliberate decision to checkpoint working code,
not a claim that the strategy works.

**This document is still current. It is NOT superseded by
[KNOWN-LIMITATIONS-v0.3.0.md](KNOWN-LIMITATIONS-v0.3.0.md)** — that file says so itself, and
the v0.2.0 findings below still stand.

Evidence for everything below:
`.claude/PRPs/reports/code review/phase4-7-code-review.md` (findings + measurements) and
`.claude/PRPs/reports/phase7-walkforward-result.chart.html` (the walk-forward
run, rendered).

> **The chart is LOCAL-ONLY as of 2026-07-28** — untracked from git as a generated
> artifact (153 KB). It is still on disk here, and it is both recoverable and
> regenerable; see [DEPRECATED-ARTIFACTS.md](DEPRECATED-ARTIFACTS.md). Regenerate with
> `.venv/bin/python scripts/build_performance_chart.py --out .claude/PRPs/reports/phase7-walkforward-result.chart.html`.

---

## 0. IT LOSES TO BUY-AND-HOLD — no benchmark was ever computed

Measured 2026-07-27, same span (2023-07-27 → 2026-07-26), gate-selected parameters:

| | Total | Annualised | Sharpe | Max DD |
|---|---|---|---|---|
| **Strategy (pooled)** | **1.11×** | **+3.45%** | **+0.282** | **56.15%** |
| BTC buy-and-hold | 2.21× | +30.3% | +0.80 | **53.0%** |
| ETH buy-and-hold | 1.03× | +0.9% | +0.34 | 67.6% |
| SOL buy-and-hold | 3.00× | +44.3% | +0.85 | 76.3% |
| Equal-weight basket | 2.17× | +29.4% | +0.73 | 64.3% |

The strategy loses to doing nothing on return **and** on risk-adjusted return, and
BTC buy-and-hold carries a *lower* drawdown. There is no axis on which it wins.

`GATE_MIN_SHARPE = 1.0` compares against **zero**, not against the obvious
alternative, so THE GATE could have blessed a strategy worse than inaction. Any
future candidate must be scored against buy-and-hold, not against zero.

Related: the reported "costs consume 97% of the gross edge" (mean gross
0.150%/trade vs 0.146% cost) was framed as a cost problem. That framing is
wrong. A 15-basis-point gross edge is noise — the *entry* carries almost no
predictive content, and three of its four conditions (1D ADX≥25, 4H ADX≥25,
close above the 55-bar mid) are the same trend-strength idea measured three
ways. No exit policy repairs that.

## 0b. Pooling 3 symbols is not the "free lunch" — they are ~0.76 correlated

Daily-return correlations over the same span: BTC/ETH **+0.809**, BTC/SOL
**+0.744**, ETH/SOL **+0.719**. At mean r ≈ 0.76 the effective independent
sample from three symbols is about **1.2**, not 3.

Consequences: "3 of 3 symbols positive" is closer to one observation than three;
23 OOS trades are worth perhaps ~9 independent ones; and DSR's near-iid
assumption is violated. Pooling more *correlated* majors will not fix the sample
problem — §4's backfill plan must prioritise assets that are not BTC beta.

## 0c. The search never explored the entry or the feature set

Every sweep — the review's 10 variants and both walk-forward grids — moved only
exit management (trail, target, channel) plus `max_hold_bars` and one classifier
bug fix. No entry-side or feature-side axis was ever tested. Absent entirely:
MACD, RSI, stochastic, ROC, divergence, market structure, cross-sectional
relative strength, session/funding-time effects, BTC-beta neutralisation, and
alternative entry mechanics (stop-order at the level, retest). Volume is computed
on every signal but gates nothing, and the PRD's "graded confidence" use of it
was never built.

This is a gap in the *research*, not a defect in the code — recorded so it is not
mistaken for ground already covered.

## 1. THE GATE FAILS — 2 of 5 mandatory conditions

Phase 7 walk-forward, 13 folds, 3 symbols pooled, 2023-07-27 → 2026-07-26:

| Condition | Measured | Threshold | Verdict |
|---|---|---|---|
| Sample adequacy | 23 OOS trades | ≥ 30 | **FAIL** |
| Sharpe (annualised) | 1.17 | ≥ 1.0 | PASS |
| Deflated Sharpe (DSR) | 0.0210 | > 0.95 | **FAIL** |
| Equity max drawdown | 19.13% | ≤ 25% | PASS |
| Per-symbol expectancy | 3 of 3 positive | all > 0 | PASS |

Selected parameters: `trail_enabled=False`, `target_enabled=False`,
`max_hold_bars=48`.

**The two failures are one failure: there is not enough data to prove anything.**
DSR fails even with the multiple-testing correction switched off entirely
(`n_trials=1` → 0.742, still under 0.95). No choice of `n_trials` rescues it, so
there is nothing to gain by relitigating that convention.

## 2. The favourable out-of-sample window is not an edge

The headline **+68% annualised** on the holdout is a 90-day extrapolation from
**23 trades**. Read the equity curve, not the number: pooled equity peaked near
**2.1×** in early 2024 and spent two years giving it back, ending at **1.11×**.
The OOS upturn is a small tail on a long decline.

Numbers the gate does not see:

- **Full-span worst drawdown: −56%.** The gate's 19.13% is the OOS window only.
- **ETHUSDT ends below water at 0.85×** over the full span, despite positive OOS
  expectancy.
- **10 of 13 folds** had negative test-window expectancy.
- **4 of 13 folds** had no parameter combination reach `WF_MIN_TRADES` on their
  180-day train window and fell back to config defaults. The train windows are
  themselves undersampled.

## 3. Sharpe and DSR are computed on a distorted return series

`backtest/equity.daily_returns` books each trade's entire `pnl_pct` on its **exit
day**. With 23 trades across 90 days that yields ~67 exact-zero days and a few
very large ones — daily **skew 3.79, kurtosis 31.24**. PSR's denominator carries
`((kurt − 1) / 4) · sr²`, so that kurtosis inflates the Sharpe estimate's
standard error enormously and blocks every significance test.

This is review finding **MEDIUM-5 and it is NOT FIXED**. It is the
highest-leverage remaining change: spreading each trade's P&L across the days it
was open is the correct attribution regardless, and it directly attacks the
kurtosis. Until then, treat both Sharpe and DSR as measured on a proxy.

## 4. Sample size cannot be fixed by tuning

Only **3 symbols** exist in the store (BTC/ETH/SOL). At the current trade rate
(~0.15/day pooled) a 90-day holdout yields ~13–23 trades against a floor of 30.
Raising the sample means **backfilling more pairs** — the PRD's stated "only free
lunch", zero degrees of freedom. `fapi.binance.com` was blocked 2026-07-05 and
reachable again 2026-07-26; re-ping before planning.

Explicitly **not** done, and should not be: lowering `WF_MIN_TRADES` or shrinking
the OOS window to manufacture a pass.

## 5. Unfixed review findings

| ID | Issue | Why left |
|---|---|---|
| MEDIUM-1 | `extreme-volatility` outranks `trending`, suppressing ~a quarter of all ADX≥25 days from a trend engine | Strategy decision needing a logged trial, not a defect |
| MEDIUM-3 | The breakout freshness rule rejects setups already beyond the channel at the setup close (~8–10%) — the cleanest breakouts | Fixing it is a strategy redesign, not a bug fix |
| MEDIUM-5 | Exit-day P&L attribution (see §3) | Changes what the Sharpe gate means; needs a deliberate decision |
| LOW-1 | Fold winners are selected on per-trade expectancy while the gate scores Sharpe — the two disagree by construction | Protocol change; wanted the gate's baseline first |

## 6. A correctness fix that cost in-sample performance

Review finding MEDIUM-2 (`engine.py`): `h_idx` keyed on the trigger bar's close,
making the last trigger bar of every setup window unreachable — 25% of
opportunities. Fixing it **reduced** in-sample results (Sharpe 0.431 → 0.255,
annualised 11.0% → 0.25%), because the old off-by-one acted as an accidental
trigger-recency filter and that filter was helping.

The fix was kept: an entry rule must be an explicit, pre-registered decision, not
an artifact of two code paths disagreeing about which setup bar is current. If
trigger recency is worth filtering on, it belongs on the grid, priced as a degree
of freedom. Recorded here so it is not silently re-broken.

## 7. Scope not yet built

- **The ranging sleeve is disabled.** `FADE_ENABLED = False` (Phase 6 DROP
  verdict). Roughly 45% of classified days are `ranging` and now produce no
  signals at all. `meanrev.py` is intact and tested so the decision is
  reversible.
- **No position sizing.** Every backtest assumes equal notional per trade.
  `MAX_RISK_PCT` documents a budget a human discharges manually; the vol-targeted
  equity curve is Phase 8 and does not exist. The −56% full-span drawdown is
  therefore an *unsized* figure.
- **Alert-only.** Nothing places orders; a human executes, serially, one open
  trade per symbol.
- **Funding is a frozen placeholder** (`FUNDING_PCT_PER_DAY = 0.0001`), not
  ingested from the exchange. Fees and slippage are real assumptions
  (0.05% + 0.02% per side); funding is a guess.

## 8. Tooling debt

- **`scripts/build_review_chart.py` is stale and will not run.** It dispatches
  trending setups to `signals.patterns.detect_patterns` (retired in Phase 5, so
  it plots geometry the engine no longer generates) and hardcodes `STEP_S = 900`,
  raising `ValueError` on the 1h trigger tier adopted in Phase 4. Use
  `scripts/build_performance_chart.py` until it is ported.
- **No remote is configured.** The repository exists only on this machine; a lost
  disk is a lost project.
- No linter and no type checker are configured, so "validation" means `pytest`
  (285 passed, 1 skipped) and `py_compile`.

## 9. Degrees of freedom consumed

Per the PRD's trial-log discipline, the Phase 4–7 review evaluated **10 engine
variants** on the tuning span, and the walk-forward evaluated **12 combos × 13
folds**. Any future DSR must count these. This is why none of the in-sample
numbers in the review may be reported as results.

---

## What IS established

Not everything here is doubt. These hold up:

- **The cost model is honest.** Fees, slippage and a funding term are charged on
  every simulated trade; mean all-in cost is ~0.146%/trade, and the review
  measured gross edge against it directly.
- **No-lookahead is enforced, not asserted.** Regime labels, setup candidates and
  pivots use only bars closed by the evaluation time, and `_assert_interval`
  raises when stored bar spacing disagrees with the configured tier — so a
  partial tier migration cannot fail silently in the flattering direction.
- **Backtest and live share one code path.** `signals/scan.py` and
  `backtest/engine.py` dispatch the same detectors, and `FADE_ENABLED` is read at
  call time so both suppress together.
- **The grid chose the exit repairs on its own.** `trail_enabled=False` and
  `target_enabled=False` were selected on train data, independently corroborating
  the review's HIGH-2 finding.
- **285 tests pass**, including regressions pinning every repair in the review.

## The next four things, in order

1. Make buy-and-hold the null hypothesis (§0). Until a candidate beats +29%
   annualised at Sharpe 0.73 risk-adjusted, tuning this system is not the task —
   deciding whether to keep it is.
2. Fix MEDIUM-5 (spread P&L across holding days). Cheap, correct, and it attacks
   the kurtosis blocking every significance test.
3. Backfill more symbols — uncorrelated ones (§0b). 23 OOS trades cannot clear a
   floor of 30 at any Sharpe, and more BTC beta adds rows, not information.
4. Re-run the walk-forward and only then reconsider whether this is a strategy
   worth sizing.
