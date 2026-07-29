# Family report: TREND

**Author**: trend-family analyst · **Date**: 2026-07-27
**Module**: `scripts/bruteforce/strategies/trend.py` (10 strategies, 116 combos)
**Splits used**: TRAIN (2023-08-01 → 2025-07-01), SELECT (2025-07-01 → 2026-01-01).
**HOLDOUT: not touched.**
**CSVs**: `/tmp/trend_train.csv`, `/tmp/trend_select.csv` (core 3) ·
`/tmp/trend_train_u20.csv`, `/tmp/trend_select_u20.csv` (20-symbol universe).

---

## Headline

**One idea in this family survives both splits and the universe expansion: a
volatility-normalised channel break with a ratcheting ATR trail
(`trend_keltner_trail`).** It is the only strategy of the eleven tested
(ten mine + the production baseline) whose mean Sharpe is positive on TRAIN *and*
SELECT when averaged over every combo and all 20 symbols — +0.269 / +0.259, with
68% / 76% of symbol-combos positive. All 12 of its combos are positive on both
splits.

**The production Donchian baseline does not generalise.** On the 20-symbol
universe it is negative on TRAIN (mean Sharpe −0.184) and badly negative on
SELECT (−1.131, only 20% of symbols positive). Every Donchian-*entry* variant I
built reproduces that same shape. This is the most important negative result in
the report and it is discussed below, because the PRD's A-core is built on that
entry rule.

**No strategy in this family clears the Deflated Sharpe Ratio.** Best DSR
observed anywhere is 0.205; median is ~0.000. Reported for transparency, not used
for ranking, per the coordinator's instruction — see the DSR caveat at the end.

---

## Methodology notes that bear on the overfitting argument

- **The risk unit is frozen, not fitted.** Every strategy stops at
  `1.5 × ATR(14)` of the 4H setup tier. `k` is never a grid axis anywhere in the
  module. This matters: it means `cost_ratio` is a property of the market, not a
  quantity I searched over. Measured `c` across all 20 symbols is 0.030–0.098 —
  under the 0.10 ceiling everywhere — at median `risk_pct` ≈ 2.7%, comfortably
  above the ~1.4% floor the brief specified. Only exit geometry (ATR-trail
  multiple, exit-channel length) is swept, and neither moves `median_risk_pct`.
- **Grids are small by design.** ≤18 combos per strategy, 2–3 values per axis,
  centred on textbook values (Donchian 20/55, ATR 14, ADX 25, EMA 12/21/50/100,
  Keltner 20). 116 combos total across ten strategies.
- **Causality.** All 10 pass `core.assert_causal`; `lookahead_canary` still fails
  it in the same run, so the audit is demonstrably live. Coarse data enters only
  via `ctx.align`; the three helpers (`_prev`, `_cross_up`/`_cross_dn`,
  `_recent`) read the current and strictly earlier trigger bars only.
- **Ranking rule.** Candidates are ranked on Sharpe plus TRAIN/SELECT agreement,
  and on the *whole combo neighbourhood* rather than the single best cell. A
  strategy whose best cell is high but whose neighbours are negative is treated
  as a fluke, not a finding.

---

## Full results table

Best combo per strategy by **TRAIN mean Sharpe across BTC/ETH/SOL**, restricted
to combos meeting `trades ≥ 30` and `cost_ratio ≤ 0.10` on all three coins; the
**same** combo's SELECT numbers alongside. `npos` = coins with Sharpe > 0 (of 3).

| Strategy | Best TRAIN combo | TRAIN mean Sh | TRAIN min Sh | npos T | SELECT mean Sh | SELECT min Sh | npos S | Verdict |
|---|---|---|---|---|---|---|---|---|
| `trend_keltner_trail` | mult 2.5, period 20, trail 3.0 | **+1.241** | +1.115 | 3/3 | **+1.198** | −0.382 | 2/3 | **KEEP** — best in family, whole grid robust |
| `trend_ema_cross` | fast 12, slow 100, no 1D confirm | +1.049 | +0.558 | 3/3 | +0.349 | +0.165 | 3/3 | Conditional keep — decays, thin SELECT sample |
| `trend_turtle` | chan 20, exit 10, no trail | +0.976 | +0.408 | 3/3 | +0.198 | −0.415 | 1/3 | Weak keep — long-only, decays |
| `trend_tsmom` | lookback 30, slope 20, trail 2.0 | +0.516 | −0.083 | 2/3 | +0.383 | +0.142 | 3/3 | REJECT — sign-unstable across grid |
| `trend_donchian_volume` | ADX 25, chan 55, vol 2.0 | +0.515 | +0.363 | 3/3 | −0.458 | −0.909 | 0/3 | REJECT — overfit |
| `trend_donchian_len` | ADX 20, chan 34, mid on | +0.467 | +0.050 | 3/3 | −0.309 | −0.599 | 0/3 | REJECT — overfit |
| `trend_ma_distance` | ADX 25, dist 1.5, MA 100 | +0.452 | +0.393 | 3/3 | **−1.448** | −1.800 | 0/3 | REJECT — worst TRAIN→SELECT reversal |
| `trend_dual_tf` | ADX 25, chan 55, d_chan 55 | +0.390 | +0.137 | 3/3 | −1.106 | −3.579 | 1/3 | REJECT — overfit |
| `trend_donchian_retest` | ADX 25, chan 55, wait 24 | +0.364 | +0.143 | 3/3 | −0.758 | −2.114 | 0/3 | REJECT — overfit |
| `donchian_production` *(baseline)* | — (frozen) | +0.142 | −0.450 | 2/3 | −1.110 | −2.116 | 1/3 | Reference — reproduces the documented +0.26/−0.45/+0.62 |
| `trend_ribbon` | ADX off, short 21, long 100 | −0.060 | −0.339 | 1/3 | +0.173 | −1.422 | 2/3 | REJECT — no signal either way |

Baseline calibration check: the harness reproduces the documented production
figures exactly — BTC **+0.26**, ETH **−0.45**, SOL **+0.62** on TRAIN, at
`cost_ratio` 0.070 and `median_risk_pct` 2.01%. The harness is sound.

**Eight of ten strategies beat the baseline's TRAIN mean of +0.142.** But TRAIN
outperformance is nearly worthless here: six of those eight go negative on
SELECT. Only Keltner, EMA-cross and Turtle beat the baseline on both splits.

### 20-symbol universe, averaged over all combos and all symbols

This is the harshest and most informative view — no cell-picking at all.

| Strategy | TRAIN mean Sh | TRAIN frac pos | SELECT mean Sh | SELECT frac pos |
|---|---|---|---|---|
| `trend_keltner_trail` | **+0.269** | 68% | **+0.259** | 76% |
| `trend_ema_cross` | +0.187 | 65% | −0.273 | 47% |
| `trend_turtle` | +0.047 | 54% | −0.069 | 56% |
| `trend_tsmom` | −0.078 | 43% | +0.249 | 63% |
| `trend_dual_tf` | −0.096 | 53% | −1.176 | 21% |
| `trend_ribbon` | −0.107 | 52% | +0.055 | 62% |
| `trend_ma_distance` | −0.137 | 45% | −0.776 | 29% |
| `trend_donchian_volume` | −0.174 | 48% | −1.142 | 23% |
| `trend_donchian_retest` | −0.182 | 50% | −1.141 | 30% |
| `donchian_production` | −0.184 | 55% | −1.131 | 20% |
| `trend_donchian_len` | −0.221 | 43% | −1.013 | 20% |

Keltner is the only row positive in both columns. Everything whose entry is an
N-bar price extreme sits at the bottom of both.

---

## Rationale per strategy, and why the failures failed

**`trend_keltner_trail` — volatility-normalised channel + ratcheting ATR trail.**
Rationale: "price is more than m ATR above its own EMA" means the same
statistical surprise in a quiet and a violent market, whereas "N-bar high" means
very different things in each. Paired with a ratcheting trail, which is the only
exit that can capture a move whose size you did not have to predict.
**Result: the family's one genuine finding.** All 12 combos positive on both
splits on the core 3; 12/12 positive on both splits on the 20-symbol universe.
Trade counts are healthy (median 126–340 on TRAIN, 27–88 on SELECT), so this is
also the only candidate that reliably satisfies the ≥30-trades bar on the short
SELECT window.

**`trend_ema_cross` — 4H EMA crossover, optional 1D agreement.**
Positive on 8/8 combos on TRAIN but only 4/8 on SELECT, and negative on the
20-symbol universe on SELECT (−0.273). *Interesting sub-result: requiring 1D
confirmation made it strictly worse in every pairing* (e.g. fast 12/slow 50:
+0.985 without confirm vs +0.860 with; SELECT +0.227 vs +0.025). The
dual-timeframe filter cuts trade count roughly in half (87 → 41 on TRAIN) without
improving per-trade quality — it is removing trades at random with respect to
outcome, which is the signature of a filter that carries no information. That is
a useful finding against the "dual-timeframe agreement" idea generally.

**`trend_turtle` — wide entry channel, shorter opposite exit channel.**
Long-only **by necessity, not preference**: `Plan.exit_signal` is
direction-agnostic by contract, so "exit longs on the N-bar low" and "exit shorts
on the N-bar high" cannot coexist in one Plan — an entry-side breakout always
satisfies the opposite side's exit immediately. My first version had exactly that
bug and produced 503–766 trades at mean Sharpe −1.36; I caught it from the trade
count, made the strategy long-only, and re-ran. **The corrected version scores
TRAIN +0.976** — a ~2.3 Sharpe swing that was entirely my implementation, not the
idea. Worth recording as a harness limitation: the short leg of any
asymmetric-exit system needs an exit array that carries direction.
Post-fix it is robust on TRAIN (12/12 combos positive, median +0.87) but decays
on SELECT and is flat on the universe. Notably its *trailed* combos hold up on
SELECT far better than the untrailed ones (trail 2.0/3.0 → SELECT +0.45 to +0.89;
trail off → +0.198 and −0.318), which feeds the hypothesis below.

**`trend_tsmom` — 1D time-series momentum + linreg-slope agreement.**
REJECT, and the reason matters: it is not that it lost, it is that its sign is
unstable across the grid. 6/12 combos positive on TRAIN, and the combos that win
on TRAIN are largely *not* the combos that win on SELECT (lookback 60/slope 40
goes −0.451 TRAIN → +0.827 SELECT; lookback 90/slope 20 goes −0.005 → −0.005).
DSR ≈ 0.000 throughout. A strategy whose parameter surface has no stable sign is
measuring noise, whatever its best cell says.

**The whole Donchian-entry branch — `trend_donchian_len`, `_retest`, `_volume`,
`trend_dual_tf`, and the production baseline.**
All four of my variants and the baseline share one shape: mildly positive on
TRAIN (core 3), decisively negative on SELECT (−0.31 to −1.11) and on the
20-symbol universe (−0.17 to −0.22 TRAIN, −0.87 to −1.14 SELECT). Five
structurally different implementations of the same entry failing identically
indicts **the shared component — the N-bar-extreme entry — not any one
variant's parameters.** That is the same inference the PRD itself used to indict
the old risk model.

Two candidate explanations, and I can partly discriminate between them:
1. *Regime.* SELECT (2025-07 → 2026-01) may simply be hostile to breakouts. The
   baseline's own collapse from +0.14 to −1.11 over the identical, frozen rule is
   consistent with this, and it means the SELECT window is doing real work as an
   out-of-sample check rather than merely being noisy.
2. *Exit, not entry.* The strategies that survive SELECT all have a ratcheting
   ATR trail; none of the Donchian variants has one. Keltner (always trailed)
   survives; Turtle survives better *with* the trail than without it; every
   untrailed Donchian variant dies. This suggests the trail — not the entry
   condition — is carrying the edge, and that the PRD's A-core may be attributing
   its hopes to the wrong half of the rule.

I did not build the clean discriminating test (a Donchian entry with an ATR trail
and nothing else changed), and I am flagging it as **the single highest-value
follow-up in this family** rather than asserting the conclusion. It is a one-
strategy, ~6-combo experiment.

**`trend_ma_distance` — price ≥ d ATR from its 4H EMA.**
The sharpest TRAIN→SELECT reversal in the family: +0.452 (3/3 coins positive) →
−1.448 (0/3, min −1.800), and −0.776 on the universe SELECT. Buying extension
after price has already travelled 1.0–1.5 ATR from its mean is, in a
mean-reverting tape, systematically buying the top. This is the clearest overfit
in the set and I would not revisit it.

**`trend_ribbon` — 1D+4H ribbon stacked, entry on reclaim of the 4H fast MA.**
No edge in either direction on any view (TRAIN −0.107, SELECT +0.055 on the
universe; ~52%/62% positive, i.e. coin-flip). Requiring ribbon agreement at two
horizons is so restrictive that what remains is timing noise. Consistent with the
`trend_ema_cross` finding that the 1D-confirmation filter carries no information.

---

## Top 3 recommendations

### 1. `trend_keltner_trail` — the only strategy I would advance

Recommended cell is **not** the TRAIN-best cell. Best-by-TRAIN is
mult 2.5 / period 20 / trail 3.0, but it fails BTC on SELECT (−0.38) and drops to
25–34 trades per coin. Its immediate neighbour is better on every robustness axis:

**Recommended: `mult 2.0, period 20, trail_k 2.0`**

| Coin | TRAIN Sharpe | TRAIN trades | `c` | SELECT Sharpe | SELECT trades |
|---|---|---|---|---|---|
| BTCUSDT | +0.50 | 202 | 0.068 | **+1.94** | 43 |
| ETHUSDT | **+1.46** | 181 | 0.054 | **+2.07** | 44 |
| SOLUSDT | +1.07 | 177 | 0.037 | +0.91 | 42 |

Positive on all 3 coins on **both** splits, ≥30 trades per coin on both splits,
`c` ≤ 0.068 everywhere. Vol-target restatement: ann. return 10–40% at max DD
15–20%, i.e. inside the PRD's ≤25% DD band. On the 20-symbol universe this cell
is positive on 15/20 symbols on TRAIN and 16/20 on SELECT.

Per-coin notes: **ETH is the strongest and most consistent** (+1.46/+2.07). BTC
is weak on TRAIN (+0.50) but strong on SELECT — do not read BTC's TRAIN number as
the expected case in either direction. **SOL is the most stable across the whole
grid.** Outside the core 3, AVAX/TRX/DOGE/ADA/XRP are all positive on both
splits; **BCHUSDT is negative on both (−0.68 / −2.00) and should be excluded**;
LTC and BNB flip hard negative on SELECT.

### 2. `trend_ema_cross`, no 1D confirmation — conditional, with a caveat

`fast 12, slow 100, confirm_1d False`: TRAIN BTC +1.20 / ETH +1.39 / SOL +0.56;
SELECT BTC +0.55 / ETH +0.33 / SOL +0.17. Positive on all six coin-split cells,
which only Keltner also manages. **But** SELECT trade counts are 15–16 per coin,
half the ≥30 bar, so the SELECT numbers are not statistically load-bearing; and
it is negative on the 20-symbol SELECT (−0.273). Advance it only as a
diversifier against Keltner, and only if pooling the 3 coins gets the sample to
30+. Always without the 1D filter.

### 3. `trend_turtle` (long-only), trailed — weakest of the three

`chan 20, exit 10, trail 3.0`: TRAIN mean +0.967, SELECT mean +0.452, 12/12
combos positive on TRAIN. Robust *within* TRAIN but decays out of sample and is
flat on the universe. I include it third mainly because it is the one surviving
member of the channel-breakout branch, and its survival correlates with having a
trail — so it is evidence for the follow-up experiment above. **Long-only, which
halves its usefulness as a standalone sleeve.** Do not advance it ahead of the
Donchian-plus-trail test; that test may well supersede it.

---

## Concerns and caveats

- **DSR does not clear anywhere.** Best observed 0.205, median ~0.000, computed
  against a registry of **70 strategies / 1,180 combos × symbols** at the time of
  my run (7 families loaded concurrently). Per the coordinator I have not tried
  to correct for it and did not rank on it; the consolidated pass owns the DSR of
  record. But no result here should be described as statistically significant
  after multiple-testing correction.
- **SELECT is six months and thin.** Only 36 of 115 combos reach ≥30 trades on
  all three coins on SELECT. Keltner is one of the few that does, which is part
  of why I trust it more than EMA-cross. Treat single-coin SELECT Sharpes built
  on 15–25 trades as directional only.
- **TRAIN→SELECT rank agreement is moderate, not strong**: Spearman 0.506 /
  Pearson 0.583 across the 113 gate-passing combos. Selecting on TRAIN alone
  would have picked wrong for six of eleven strategies.
- **The recommended Keltner cell is a neighbour of the TRAIN-best cell, chosen
  partly on SELECT behaviour.** That is a mild use of SELECT for selection, which
  is what SELECT is for, but it should be counted as a consumed degree of
  freedom. The honest defence is that all 12 combos are positive on both splits,
  so the choice among them moves the result much less than the choice of strategy.
- **`mult` sits at a grid edge** (2.5 is the top value and the TRAIN-best). I did
  not extend the grid to 3.0+ to avoid spending more DoF; if Keltner advances,
  that edge is worth one bounded check.
- **Turtle's short leg is untested** because of the direction-agnostic
  `exit_signal` contract. If asymmetric-exit systems matter, the harness needs an
  exit array that carries direction — a harness change, not a strategy change.
- **Regime dependence is unresolved.** I cannot distinguish "breakout entries are
  broken" from "2025-H2 was hostile to breakout entries" with two splits. The
  Donchian-plus-trail experiment would help; a third opinion from the HOLDOUT
  would settle it, and is explicitly not mine to spend.
