# Brute-Force Strategy Search

**Started:** 2026-07-27 · **Commit:** `master` @ `33a8c37` · **Status:** complete
(see `FINDINGS.md`)

The trunk in this repo is `master`; there is no `main`. `33a8c37` fully contains
`feat/phase4-tier-shift` (merge `8778003`) and the phase1/2/3 branches, and
`git diff HEAD -- src/trading_bot/` was empty throughout the search — so the
production components the harness imports were read at their committed state.

Motivation: two PRD iterations (`hybrid-trend-voltarget.prd.md`, superseding
`initial-requirements.md`) failed the acceptance gate. Rather than tune the single
selected strategy a third time, this session searches broadly — many strategy
families, many parameter settings, 20 symbols — to find out *what the data
actually supports*, and to establish whether the best strategy differs per coin.

---

## The gate being targeted

From the PRD's Success Metrics. A candidate passes only if **all** hold:

| Gate | Threshold |
|---|---|
| Sharpe (net, out-of-sample) | ≥ 1.0 |
| Max drawdown at the 25% vol target | ≤ 25% |
| Deflated Sharpe Ratio | > 0.95 (significant at p < 0.05) |
| Cost ratio `c = round-trip cost / median risk` | ≤ 0.10 |
| Cross-symbol consistency | positive expectancy on **all three** of BTC/ETH/SOL |
| Sample adequacy | ≥ 30 trades |

Annualized return is **reported, never optimized toward** — the PRD is explicit
that return is an outcome of Sharpe plus the volatility target, and treating it
as a target is how the previous iterations got into trouble.

## Method

**One shared harness, many authors.** `scripts/bruteforce/` holds a vectorized
multi-timeframe backtester (`core.py`), an indicator library (`indicators.py`), a
strategy registry (`registry.py`), and a parallel sweep driver (`runner.py`).
Analysts declare strategies as `@register`-ed functions returning a `Plan` of
per-bar decisions; everything else — costs, fills, metrics, split enforcement,
trial counting — is handled identically for every strategy, so the leaderboard
compares like with like.

Production code under `src/trading_bot/` is **not modified**. The harness imports
from it (Wilder ATR/ADX, Donchian, the cost constants, and every equity metric in
`backtest/equity.py`) so the two cannot drift, but nothing flows the other way.

### Data

20 USDT-M perpetuals × 1d/4h/1h, 2023-01-01 → 2026-07-24, in `data/ohlcv.db`.
Selected as the highest-liquidity symbols with full history, spread across
sectors so cross-sectional strategies see real dispersion rather than 20 proxies
for BTC beta. See `scripts/bruteforce/universe.py` for the list and the rule.

The gate is still evaluated on **BTC/ETH/SOL**; the other 17 symbols provide
breadth for cross-sectional work and an out-of-sample robustness check.

### The three-way split

| Split | Span | Use |
|---|---|---|
| TRAIN | 2023-08-01 → 2025-07-01 | sweep freely (starts after the 207-bar 1D warmup) |
| SELECT | 2025-07-01 → 2026-01-01 | rank, shortlist |
| HOLDOUT | 2026-01-01 → 2026-07-24 | **one shot**, final shortlist only |

`runner.py` refuses to run HOLDOUT without both an explicit `--only` shortlist
and `--i-am-spending-the-holdout`. The holdout is a depleting resource: once a
strategy has been selected against it, it is no longer out-of-sample, and every
later claim of out-of-sample performance would be false.

### Anti-overfitting controls

These are the substance of the exercise, not paperwork. The previous iterations
did not fail for lack of ideas; they failed because the measurement was
generous.

1. **Causality is proven mechanically, not asserted.** `core.assert_causal`
   rebuilds each strategy on truncated history and requires every decision array
   to be bit-identical on the prefix. `shift(-1)`, centered windows, full-sample
   normalisation, and `bfill` all fail it loudly. A strategy that fails is
   disqualified before any of its numbers reach the leaderboard.

   The check has a canary: `strategies/baseline.py` registers
   `lookahead_canary`, which peeks at the next bar deliberately and **must** be
   disqualified on every run. Without it, "all strategies passed" could just
   mean the check does nothing — and in fact the first version of the check had
   a slack window that let the canary through. That bug is documented in
   `assert_causal`'s docstring so it cannot be reintroduced as a "fix".

2. **The cost model is frozen and never swept.** Taker 0.05% + 0.02% slippage
   per side, charged twice, plus funding per day held. The PRD identifies
   optimistic cost assumptions as a silent overfitting channel worth ~5%/yr of
   phantom return at current turnover.

3. **Risk cannot buy Sharpe.** Sizing is fixed-fractional-risk, so position size
   is a constant multiple of `1 / risk_pct`. Sharpe is therefore invariant to the
   risk level — a unit test pins this. "Finetuning the risk" changes return and
   drawdown, which are reported at the PRD's 25% vol target, and cannot move the
   headline metric.

4. **Grid discipline.** Each strategy is capped at roughly 72 parameter combos,
   with grids centred on canonical values (Donchian 20/55, ATR 14, ADX 25) rather
   than searched arbitrarily. Every combo is a degree of freedom charged against
   the Deflated Sharpe Ratio.

5. **DSR is charged for the whole search.** The trial count passed to
   `deflated_sharpe` is the total across every registered strategy × symbol, not
   the count for the winner alone. A winner is priced for the fact that it was
   selected out of thousands of attempts.

6. **TRAIN/SELECT agreement is a first-class check.** A strategy strong on TRAIN
   and dead on SELECT is reported as overfit rather than promising.

### Harness validation

`scripts/bruteforce/test_harness.py` — 28 tests, all passing. They pin fill
timing (entry at the signal bar's close; the entry bar's own low cannot stop the
trade), the conservative same-bar rule (a bar touching both stop and target books
the loss), cost arithmetic, the trail's inability to fire on the bar that set its
own extreme, multi-timeframe alignment (a 4H value is invisible before that bar
closes), and the fact that the causality audit can actually fail.

Independent corroboration: the harness measures BTC `cost_ratio = 0.0698` at
k=1.5×ATR(4H), against the 0.0711 the PRD measured separately — and ETH matches
to four decimals (0.0523). The cost and risk model reproduces production.

---

## Files

| Path | What it is |
|---|---|
| `scripts/bruteforce/core.py` | harness: data, alignment, simulator, metrics, causality check |
| `scripts/bruteforce/indicators.py` | indicator library (re-exports production Wilder/Donchian) |
| `scripts/bruteforce/registry.py` | `@register` contract and grid expansion |
| `scripts/bruteforce/runner.py` | parallel sweep driver, split enforcement |
| `scripts/bruteforce/universe.py` | the 20-symbol research universe |
| `scripts/bruteforce/test_harness.py` | 28 harness tests |
| `scripts/bruteforce/strategies/` | one module per family |
| `family-*.md` (this directory) | per-family findings, written by each analyst |
| `results/<SPLIT>.csv` | raw leaderboard rows |

### Leaderboard CSV columns

`strategy, family, symbol, split, trigger_tf, params_json, trades, win_rate,
expectancy_R, profit_factor, sharpe, sortino, max_dd_pct, ann_return_pct,
ann_return_at_target_pct, max_dd_at_target_pct, ann_vol_pct, vol_scale,
cost_ratio, median_risk_pct, dsr, avg_bars_held, long_share, stop_rate,
target_rate, time_rate, n_days`

---

## Families under test

| Family | Prior going in |
|---|---|
| trend | The incumbent. Production Donchian scores TRAIN Sharpe BTC +0.26 / ETH −0.45 / SOL +0.62 — the bar to beat. |
| momentum | Highest hope. The PRD names universe expansion as its designated escalation path (published Sharpe >1.5), driven by cross-sectional breadth. |
| meanrev | The Bollinger fade sleeve was dropped for cost-ratio failure (0.18/0.11/0.10 vs a 0.10 ceiling). Re-tested with ATR-floored stops. |
| volatility | Volatility is what killed the previous system, which makes it worth trading deliberately. Also tests whether extreme-vol suppression discards edge. |
| structure | Support/resistance and liquidity sweeps. Structural stops are the natural fix for noise-floor stops — if ATR-floored. |
| candlestick | Weakest prior. Tested mainly as *confirmation filters*, with the filtered-vs-unfiltered delta reported. |
| chartpattern | Re-test of geometry the PRD **retired** (triangle/flag/H&S lost on all three symbols), now under ATR stops, at the user's explicit request. |
| ensemble | Regime-switched composites, authored after the single-family results are in. |

## Progress

- [x] Backfill 20-symbol universe (1d/4h/1h) — all series complete
- [x] Build harness, registry, runner
- [x] Validate harness (28 tests) + prove the causality audit can fail
- [ ] Family strategy authoring and TRAIN/SELECT sweeps (7 analysts in parallel)
- [ ] Ensemble/composite family
- [ ] Consolidated leaderboard + per-coin recommendations
- [ ] One-shot HOLDOUT evaluation of the final shortlist
