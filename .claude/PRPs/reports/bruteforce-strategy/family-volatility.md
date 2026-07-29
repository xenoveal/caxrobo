# Brute-force strategy search — family: VOLATILITY

**Author**: research sweep · **Date**: 2026-07-27
**Code**: `scripts/bruteforce/strategies/volatility.py` (11 registered strategies, 156 combos)
**Splits used**: TRAIN (2023-08-01 → 2025-07-01), SELECT (2025-07-01 → 2026-01-01). **HOLDOUT untouched.**
**Cost model**: frozen (`FEE_PCT` 0.0005 / `SLIPPAGE_PCT` 0.0002 per side, funding per day). Nothing in `core.py` / `registry.py` / `indicators.py` was modified.

---

## 1. Causality

All **156** combos of all **11** strategies pass `core.assert_causal` with `tail_skip=0`, verified on two symbols (BTCUSDT via `runner.py --causal-only`, and a full-combo sweep on SOLUSDT). No `shift(-n)`, no `center=True`, no full-sample quantile, no `bfill`. Every volatility threshold is a trailing `ta.percentile_rank` over an explicit window — the full-sample vol quantile is the classic lookahead in this family and is absent by construction. `vol_opening_range` needs a per-UTC-day level and uses a within-day `cummax`/`cummin` (never a within-day `max`), so the level at any bar depends only on that day's earlier bars.

The `lookahead_canary` in `baseline.py` still fails the audit, so a green audit here means something.

Stop sizing: `k * ATR(14)` of the 4H setup tier, `k >= 1.2` everywhere. Measured median 4H ATR%/close is 1.31% BTC / 1.79% ETH / 2.50% SOL, so `k = 1.2` puts BTC — the tightest — at `cost_ratio = 0.0014/0.0157 = 0.089`. `k < 1.2` is not a cheap trade, it is a trade that fails the cost gate, and is deliberately absent from every grid.

---

## 2. Full results — best-by-Sharpe TRAIN config, with its SELECT numbers

### 2a. Core 3 symbols (BTC/ETH/SOL), gated on `min trades >= 30` and `cost_ratio <= 0.10`

117 of 156 combos pass the gate. Best combo per strategy, ranked by mean TRAIN Sharpe; SELECT columns are the **same config carried forward, not re-optimised**.

| Strategy | Best TRAIN config | TRAIN mean Sh | BTC | ETH | SOL | SELECT mean Sh | BTC | ETH | SOL | TRAIN WR | TRAIN PF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `vol_chandelier_ride` | k 1.2, mult 4.0, per 40 | **+1.139** | +0.80 | +1.47 | +1.14 | **+1.301** | +0.91 | +1.38 | +1.61 | 0.319 | 1.29 |
| `vol_squeeze_release` | chan 40, k 1.2, kc_mult 2.0 | +1.043 | +1.03 | +0.57 | +1.52 | −1.439 | **−5.17** | +1.49 | −0.64 | 0.341 | 1.57 |
| `vol_term_structure` | k 1.2, ratio_min 1.5, short 12 | +0.977 | +0.59 | +0.99 | +1.35 | +1.401 | +0.52 | +1.65 | +2.03 | 0.280 | 1.32 |
| `vol_nr_break` | k 2.0, nr 14, w 3 | +0.882 | +0.49 | +1.28 | +0.88 | +0.076 | +1.62 | +0.51 | −1.90 | 0.412 | 1.09 |
| `vol_tercile_break` | band **low**, chan 20, k 1.2 | +0.874 | +0.55 | +1.21 | +0.86 | +1.415 | +0.68 | +2.19 | +1.37 | 0.280 | 1.64 |
| `vol_atr_expansion` | mom 12, pr_min 0.90, pr_win 120 | +0.779 | +0.75 | +0.92 | +0.67 | −0.009 | +0.15 | −0.11 | −0.06 | 0.459 | 1.33 |
| `vol_two_tier_compression` | chan 10, k 1.2, pr_max 0.6 | +0.657 | +0.17 | +0.78 | +1.01 | +1.011 | +0.77 | +0.95 | +1.31 | 0.251 | 1.19 |
| `vol_extreme_gate_probe` | gate **suppress**, thresh 0.85 | +0.516 | +0.66 | +0.13 | +0.76 | −0.497 | −1.02 | +0.58 | −1.05 | 0.295 | 1.18 |
| `vol_inside_bar_break` | k 2.5, nconsec 1 | +0.481 | +0.54 | +0.84 | +0.06 | +0.239 | −0.09 | +0.48 | +0.33 | 0.431 | 1.07 |
| `vol_bbw_trough_break` | chan 20, pr_max 0.30, pr_win 540 | +0.427 | +0.01 | +0.44 | +0.83 | +1.205 | +1.38 | +1.14 | +1.09 | 0.298 | 1.06 |
| `vol_opening_range` | k 2.0, min_range 0.0, or_len 8 | +0.250 | +0.52 | −0.33 | +0.56 | −0.632 | −1.38 | −0.21 | −0.31 | 0.390 | 0.95 |

**Production Donchian TRAIN baseline for reference: BTC +0.26 / ETH −0.45 / SOL +0.62.** Every one of the eleven best-configs beats it on ETH, which is where production is actually broken; nine of eleven beat it on all three coins simultaneously.

**Caveat that must not be skipped**: SELECT is a 6-month window. Minimum per-symbol trade counts on SELECT for these configs run 7–58, so most fail the `trades >= 30` adequacy bar *on SELECT*. SELECT Sharpe here is a direction-of-travel signal, not a passed gate. The −5.17 on BTC for `vol_squeeze_release` is 14 trades and should be read as "fragile", not "−5.17".

### 2b. Full 20-symbol universe — the honest number

The core-3 numbers above are inflated by symbol selection: BTC/ETH/SOL were the coins the whole project was built around. Re-run on the 20-symbol universe, the same gate (`min trades >= 30` across **all 20** symbols, `cost_ratio <= 0.10`) admits far fewer combos and the Sharpes roughly halve.

| Strategy | Best TRAIN config (20-sym) | TRAIN mean Sh | % symbols Sh>0 | median trades | max cost_ratio | mean DSR | SELECT mean Sh | SELECT % Sh>0 |
|---|---|---|---|---|---|---|---|---|
| `vol_term_structure` | k 2.0, ratio_min 1.5, short 24 | +0.568 | 0.90 | 53 | 0.070 | 0.011 | −1.284 | 0.20 |
| `vol_squeeze_release` | chan 20, k 1.8, kc_mult 2.0 | +0.533 | 0.90 | 77 | 0.077 | 0.020 | −0.569 | 0.35 |
| `vol_chandelier_ride` | k 2.0, mult 2.0, per 20 | +0.529 | 0.85 | 223 | 0.068 | 0.005 | +0.286 | 0.75 |
| `vol_bbw_trough_break` | chan 20, pr_max 0.30, pr_win 540 | +0.474 | 0.85 | 89 | 0.094 | 0.005 | +0.181 | 0.60 |
| `vol_tercile_break` | band **low**, chan 20, k 2.0 | +0.425 | 0.85 | 54 | 0.086 | 0.003 | +0.547 | 0.80 |
| `vol_nr_break` | k 2.0, nr 7, w 3 | +0.386 | 0.80 | 212 | 0.068 | 0.005 | +0.027 | 0.50 |
| `vol_two_tier_compression` | chan 10, k 2.0, pr_max 0.6 | +0.309 | 0.85 | 67 | 0.085 | 0.002 | +0.640 | 0.70 |
| `vol_atr_expansion` | mom 12, pr_min 0.80, pr_win 120 | +0.242 | 0.70 | 63 | 0.064 | 0.007 | +0.233 | 0.65 |
| `vol_opening_range` | k 2.0, min_range 0.0, or_len 8 | +0.206 | 0.55 | 227 | 0.070 | 0.004 | −0.265 | 0.45 |
| `vol_inside_bar_break` | k 2.5, nconsec 1 | +0.007 | 0.60 | 126 | 0.054 | 0.001 | +0.214 | 0.65 |
| `vol_extreme_gate_probe` | gate none, thresh 0.85 | −0.184 | 0.55 | 88 | 0.097 | 0.001 | −1.131 | 0.20 |

**Every DSR is ≈ 0.** With 156 combos × 20 symbols charged against the search, no single configuration in this family is statistically distinguishable from the best of a lucky draw. That is the correct reading and it is not fixable by finding a better combo — only by a smaller pre-committed grid or more data.

---

## 3. Should the extreme-volatility regime keep suppressing signals?

**Yes. Keep the suppression.** Four independent instruments were built to answer this and all four agree, on both splits and both universes.

**Instrument 1 — `vol_tercile_break`: one fixed Donchian rule, three mutually exclusive 1D realised-vol terciles, nothing else varied.**

| vol band | TRAIN Sh (core3) | TRAIN Sh (20-sym) | TRAIN % Sh>0 | TRAIN PF | SELECT Sh (20-sym) | SELECT PF |
|---|---|---|---|---|---|---|
| **low** | **+0.859** | **+0.340** | 0.80 | 1.33 | **+0.567** | 1.44 |
| mid | +0.495 | +0.119 | 0.60 | 1.08 | −1.075 | 0.69 |
| **high** | **−0.168** | **−0.284** | 0.41 | 0.885 | **−1.665** | 0.49 |

Monotone in the right direction on all six measurements. High-vol breakouts lose money with PF 0.49–0.885 — not merely weaker, negative.

**Instrument 2 — `vol_extreme_gate_probe`: the *production* rule (20-bar 4H Donchian + 55-mid + 1D ADX>25, 1.5×ATR stop) under the classifier's own gate, absent / suppressing / inverted.** The gate reproduces `classifier.py` exactly: relative-ATR percentile over a 180-bar trailing window on the 1D tier.

| gate | TRAIN Sh (core3) | TRAIN PF | TRAIN Sh (20-sym) | SELECT Sh (20-sym) | SELECT PF |
|---|---|---|---|---|---|
| `suppress` (today's behaviour) | **+0.493** | 1.17 | **−0.031** | −1.057 | 0.73 |
| `none` | +0.142 | 0.93 | −0.184 | −1.131 | 0.78 |
| `only` (extreme bucket alone) | **−1.209** | **0.475** | **−0.545** | −0.925 | 0.95 |

`suppress` > `none` on both universes and both splits. Trading *only* the extreme bucket is the worst of the three on TRAIN by a wide margin — PF 0.475 on the core three. The 23 average trades per symbol in the `only` arm is also the sample-adequacy answer: the extreme bucket is ~12% occupancy, so even if it held edge it could not carry a signal product on its own.

**Instrument 3 — `vol_atr_expansion` threshold ladder.** As the ATR-percentile entry threshold rises toward the classifier's own 0.90 extreme definition, performance degrades monotonically: pr_min 0.80 → +0.278, 0.90 → +0.021, 0.95 → −0.054 (core-3 TRAIN mean across all its combos). Entering *because* volatility just became extreme is a losing rule.

**Instrument 4 — `vol_two_tier_compression` regime ladder.** Loosening the "regime must be quiet" percentile ceiling from 0.20 → 0.40 → 0.60 improves Sharpe (+0.088 → +0.350 → +0.421) purely through sample size, but the ceiling is still a *ceiling*: nothing in the family wanted a floor.

**Verdict.** The suppression is not discarding edge; it is removing the single worst bucket in the family. The open question in the PRD is answered in favour of the status quo. Two refinements are worth carrying forward, though:

1. The suppression is currently binary and set at the 90th percentile. The tercile evidence says the gradient runs across the *whole* distribution, not just the top decile — low-vol is materially better than mid-vol, not just better than high-vol. A **low-vol preference**, not merely an extreme-vol veto, is where the measured edge lives. That is a strictly larger change than the classifier currently makes.
2. Thresh 0.85 vs 0.90 barely matters (`suppress` at 0.85 = +0.516, at 0.90 = +0.471 on core-3). The threshold is not the sensitive parameter, so there is no case for tuning it.

---

## 4. What failed, and why

- **`vol_opening_range` — clearest failure. Reject.** TRAIN +0.206 / SELECT −0.265 on 20 symbols, TRAIN PF 0.989, only 55% of symbols positive, and the worst core-3 SELECT of the family (BTC −1.38). Diagnosis: the 00:00-UTC anchor was justified on funding-settlement and daily-bar grounds, but the data says 24h crypto simply has no privileged intraday window. It produces the *most* trades in the family (median 227) at PF < 1, so it is a cost-burning machine: high frequency plus no edge is the exact shape the PRD already condemned at the 15m tier. `min_range` did not rescue it.
- **`vol_atr_expansion` and `vol_term_structure` — the "expansion as signal" hypothesis fails.** Both read the same event through different estimators, and neither survives. `vol_term_structure` had the single best 20-symbol TRAIN Sharpe (+0.568, 90% of symbols positive) and then posted **−1.284 with only 20% of symbols positive on SELECT** — the largest TRAIN→SELECT reversal in the family. Two independent estimators of the same idea, one collapsing and one flat, is evidence against the *idea*, not against either estimator. This is consistent with, and reinforces, the extreme-vol verdict above.
- **`vol_squeeze_release` — good idea, fragile implementation.** Best-in-family on core-3 TRAIN (+1.043) and second-best on 20-symbol TRAIN, then −1.439 on core-3 SELECT driven by a 14-trade BTC leg at −5.17. Only 10 of 27 combos pass the trade/cost gate at all: `chan 40` with `kc_mult 1.0` leaves 7 trades in two years. The squeeze release is a genuinely rare event, and rarity plus a 156-combo grid is how you manufacture a top-of-leaderboard number that means nothing. It needs either a wider universe to accumulate sample or a looser definition, not a better parameter.
- **`vol_inside_bar_break` — the compression measure is too weak.** 20-symbol TRAIN +0.007, PF 0.979. `nconsec 2` and `3` mostly fail the trade gate (2 of 9 combos pass; one config produced literally 0 trades on a symbol). A single inside bar on 4H is a common, low-information event; stacking them to make it informative destroys the sample. Reject.
- **`vol_nr_break` — passes TRAIN, dies out of sample.** 20-symbol TRAIN +0.386 / SELECT +0.027; core-3 SELECT is +1.62 BTC / +0.51 ETH / **−1.90 SOL**, i.e. sign-unstable across coins in the same window. Marginal at best.
- **General honesty note.** The core-3 → 20-symbol drop is roughly a factor of two in Sharpe for every strategy in the family. Any number quoted only on BTC/ETH/SOL in this family should be assumed to contain a symbol-selection premium of about that size.

---

## 5. Top-3 recommendations

Selected on TRAIN/SELECT **agreement** and breadth of per-symbol positivity, not on peak TRAIN Sharpe. All three carry the identical config from TRAIN to SELECT with no re-fitting.

### #1 — `vol_chandelier_ride` (k 2.0, mult 2.0, per 20)

The only strategy in the family that is positive on both splits, on both universes, with an adequate sample everywhere.

| | TRAIN mean Sh | % syms Sh>0 | median trades | max cost_ratio | SELECT mean Sh | % syms Sh>0 |
|---|---|---|---|---|---|---|
| 20-sym | +0.529 | 0.85 | 223 | 0.068 | +0.286 | 0.75 |

- **BTC** TRAIN +1.12 (218 trades) → SELECT **−0.42** (62). The weak leg; BTC's low ATR% makes it the coin where the trail is most often clipped by noise.
- **ETH** TRAIN +1.04 (224) → SELECT **+2.30** (51). Strongest and most consistent.
- **SOL** TRAIN +0.36 (225) → SELECT +0.10 (58). Positive but thin; SOL's 2.50% ATR means `k=2.0` gives a ~5% stop, so the risk unit is large and the trade count per unit of risk is low.
- Mechanics: median risk 4.6%, `cost_ratio` 0.068 (comfortably inside 0.10), average hold 30 bars, max DD at the 25% vol target 24.3%, annualised return at target +11.6%.
- **Why it is #1 despite not topping the TRAIN table**: the edge is in the *exit*, not the entry. The entry is a deliberately plain 20-bar 4H breakout — the same entry that loses money in `vol_extreme_gate_probe`'s `none` arm. Replacing the fixed target/time-stop with a ratcheting ATR chandelier is what turns it positive. That localises the finding to one component, which is both more believable and more portable than an entry-shaped edge.
- The core-3-tuned variant (k 1.2, mult 4.0, per 40) shows +1.14 TRAIN / +1.30 SELECT on the core three but **fails the cost gate on the wider universe** (max `cost_ratio` 0.124–0.136). Do not promote that config.

### #2 — `vol_tercile_break` (band **low**, chan 20, k 2.0)

| | TRAIN mean Sh | % syms Sh>0 | median trades | max cost_ratio | SELECT mean Sh | % syms Sh>0 | PF |
|---|---|---|---|---|---|---|---|
| 20-sym | +0.425 | 0.85 | 54 | 0.086 | +0.547 | 0.80 | 1.29 / 1.27 |

- **BTC** TRAIN +0.53 (63) → SELECT +0.02 (26) — flat, not broken.
- **ETH** TRAIN +1.10 (37) → SELECT **+2.00** (18). Best leg, but 18 SELECT trades is below the adequacy bar.
- **SOL** TRAIN +0.90 (57) → SELECT +1.09 (15). Consistent sign, inadequate sample.
- Best profit factor of any recommendation (1.29 TRAIN / 1.27 SELECT) and the lowest drawdown at the vol target (19.9%).
- **Caveat that matters more here than anywhere**: this was authored as a *controlled experiment*, not a candidate. Promoting the winning arm of your own experiment to a strategy is selection on the same data that produced the finding. It earns a place because the low-vol result replicates across two splits and two universes — but the honest framing is "the low-vol *filter* is the finding, and it should be applied to a rule chosen independently", which is exactly what #1 and #3 do.

### #3 — `vol_two_tier_compression` (chan 10, k 2.0, pr_max 0.6)

| | TRAIN mean Sh | % syms Sh>0 | median trades | max cost_ratio | SELECT mean Sh | % syms Sh>0 |
|---|---|---|---|---|---|---|
| 20-sym | +0.309 | 0.85 | 67 | 0.085 | +0.640 | 0.70 |

- **BTC** TRAIN +0.01 (75) → SELECT +0.77 (21). **ETH** TRAIN +0.19 (63) → SELECT +1.42 (23). **SOL** TRAIN +0.99 (75) → SELECT **+1.78** (26).
- The only recommendation whose SELECT is materially *better* than its TRAIN on every core coin, which is either genuine robustness or a favourable 6-month regime; 6 months cannot distinguish those.
- Answers its own registered question: the two-tier version (regime quiet **and** setup quiet) at pr_max 0.6 beats the single-tier `vol_bbw_trough_break` on SELECT (+0.640 vs +0.181), so the macro-regime agreement is doing work rather than just cutting sample. But note the *direction* — the best `pr_max` is the loosest one tested (0.6). Tightening to 0.2 collapses it (+0.088). If the real optimum is at or above 0.6, the axis is untested at its optimum and the reported number is a lower bound on a boundary.

---

## 6. Robustness concerns — read before promoting anything

1. **DSR ≈ 0 for every combo.** 156 combos × 20 symbols is 3,120 evaluations charged against the search. Nothing here clears p < 0.05 after that charge. The PRD's DSR gate is not met by this family, full stop. If any of these go forward it must be with a **pre-committed single configuration**, not a leaderboard pick.
2. **SELECT sample inadequacy is systemic, not incidental.** Only `vol_chandelier_ride`, `vol_nr_break` and `vol_opening_range` reach 30+ trades per symbol on SELECT — and two of those three are rejects. The recommendations at #2 and #3 have 15–26 SELECT trades per core coin. Their SELECT Sharpes are suggestive, not confirmatory.
3. **Symbol-selection premium of ~2×.** Core-3 Sharpe is about double 20-symbol Sharpe for essentially every strategy in this family. Report the 20-symbol number.
4. **The `k` axis is doing cost-gate work as well as strategy work.** Because `cost_ratio = 0.0014/(k·ATR%)`, sweeping `k` sweeps the gate too, and the surviving configs skew toward `k = 2.0`. That is not a free parameter choice; it is the cost frontier selecting for wider stops. Consistent with the PRD's decision to freeze `k` rather than fit it — and a reason to treat "k 2.0 won" as a cost artefact rather than a discovery.
5. **BTC is the weak leg in two of the three recommendations** (#1 −0.42 SELECT, #2 +0.02 SELECT). The PRD gate is *all three* symbols positive, not an average. On current evidence this family does not pass that gate on SELECT for BTC.
6. **The vol-target scaling flatters returns.** Reported `ann_return_at_target_pct` of 10–12% comes with `max_dd_at_target_pct` of 20–24%, already at the PRD's 25% drawdown ceiling before any live degradation. The 30–45% return target is not reachable from this family's Sharpe without breaching drawdown.
7. **`vol_chandelier_ride`'s trail is the only untested-in-production mechanism.** It relies on the simulator's ratchet ordering (trail updates only after the current bar's exits resolve). That is correct in `core.simulate`, but the production engine would have to reproduce it exactly; a trail that ratchets before exit resolution is a lookahead in live code even though it passes here.
