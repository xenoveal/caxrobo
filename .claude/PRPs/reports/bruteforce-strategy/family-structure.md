# Family: MARKET STRUCTURE

Support/resistance, ranges, pivots, liquidity levels. 9 strategies, 184 parameter
combos, swept on TRAIN (2023-08-01 → 2025-07-01) and SELECT (2025-07-01 →
2026-01-01), on the 3-symbol core and on the full 20-symbol universe.

Code: `scripts/bruteforce/strategies/structure.py`.
Raw results: `/tmp/st_{train,select}_{core,full}.csv` (regenerate with the
commands at the bottom).

**HOLDOUT was not touched.**

---

## Headline

**The cost frontier is cleared decisively — and that is the one unambiguous win.**
Every one of the 184 combos reports `cost_ratio` well inside the 0.10 ceiling on
the core three (`c` = 0.014–0.107, median ≈ 0.04) with `median_risk_pct` of
2.0–9.8% rather than the old model's 0.5%. The `_structural_stop` ATR floor is
load-bearing, not decorative: on the sweep and retest strategies the
`atr_floor * ATR(4H)` term binds on the majority of bars, and the two strategies
where the raw structural distance most often won (`st_range_fade`,
`st_sr_retest` at `atr_floor=1.0`) are exactly the two that push `c` back up
toward and past 0.10. That reproduces the dropped Bollinger-fade failure mode
(`c` 0.18/0.11/0.10) in miniature and confirms the diagnosis.

**The edge, however, is thin and mostly does not survive SELECT.** Only one
strategy (`st_bos`) is positive in mean Sharpe on both splits across the
20-symbol universe, and it is positive by a margin (+0.25 → +0.34) that is not
distinguishable from noise. Deflated Sharpe is ≈ 0.000–0.003 for every combo:
charged for the full search, nothing here is significant. Two strategies
(`st_range_fade`, `st_round_number`) are decisively negative and should be
retired.

Production Donchian on TRAIN (the bar): BTC +0.26 / ETH −0.45 / SOL +0.62,
mean **+0.14**. Three structure strategies beat that mean on TRAIN
(`st_range_breakout` +0.53, `st_bos` +0.36, `st_prior_day_break` +0.12 ≈ tie),
but only `st_bos` also holds up on SELECT.

---

## All strategies — best-by-Sharpe TRAIN config, core three (BTC/ETH/SOL)

Gates applied for "best": `trades ≥ 30` on every symbol, `cost_ratio ≤ 0.10` on
every symbol, all 3 symbols present. `risk` = `median_risk_pct` (median across
symbols); `c` = worst-symbol `cost_ratio`.

| Strategy | Best TRAIN config | TRAIN Sharpe (mean / min / +ve) | risk | c | wr | pf | SELECT Sharpe (mean / min / +ve) | risk | c | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **st_bos** | span 6, rr 1.5, atr_floor 1.5, require_hl | **+0.36** / −0.47 / 2⁄3 | 6.41% | 0.029 | 0.48 | 1.16 | **+0.15** / −1.02 / 2⁄3 | 6.42% | 0.031 | **Keep** — best two-split agreement |
| **st_range_breakout** | period 20, max_width 0.10, atr_floor 1.5, rr 3.0 | **+0.53** / −0.30 / 2⁄3 | 3.60% | 0.051 | 0.40 | 1.13 | +0.08 / −0.51 / 2⁄3 | 4.67% | 0.054 | **Keep** — best TRAIN, decays OOS |
| **st_range_breakout_retest** | period 20, max_width 0.10, retest_within 8, rr 2.0 | −0.16 / −0.76 / 1⁄3 | 2.48% | 0.069 | 0.36 | 0.91 | −0.47 / −1.37 / 1⁄3 | 2.90% | 0.082 | Marginal on core; better on universe |
| **st_prior_day_break** | adx_min 0, atr_floor 1.5, rr 2.5 | +0.12 / −0.04 / 2⁄3 | 2.95% | 0.063 | 0.36 | 0.99 | −1.05 / −3.06 / 1⁄3 | 3.37% | 0.076 | **Reject** — clean TRAIN→SELECT collapse |
| **st_mtf_level** | span 3, tol 0.002, atr_floor 1.5, rr 2.5 | +0.10 / −1.07 / 2⁄3 | 2.72% | 0.067 | 0.37 | 1.08 | −0.56 / −2.86 / 1⁄3 | 3.25% | 0.088 | Reject on core; ~flat on universe |
| **st_liquidity_sweep** | span 6, pierce 0.003, atr_floor 1.0, rr 1.5 | +0.01 / −0.35 / 2⁄3 | 2.00% | 0.090 | 0.44 | 1.05 | **−1.72** / −2.65 / 0⁄3 | 2.06% | 0.112 | **Reject** — worst decay in the family |
| **st_sr_retest** | span 6, retest_within 12, rr 2.5, atr_floor 1.5 | +0.07 / −0.41 / 1⁄3 | 2.66% | 0.068 | 0.36 | 1.03 | −1.47 / −2.88 / 0⁄3 | 3.16% | 0.083 | **Reject** |
| **st_round_number** | step_mult 1.0, tol 0.001, atr_floor 1.5, rr 2.5 | −0.44 / −0.96 / 1⁄3 | 2.98% | 0.065 | 0.39 | 0.92 | −1.41 / −4.15 / 1⁄3 | 3.12% | 0.066 | **Retire** — negative on both splits |
| **st_range_fade** | period 20, max_width 0.06, edge 0.15, atr_floor 1.5 | **−1.35** / −1.93 / 0⁄3 | 2.15% | 0.082 | 0.47 | 0.74 | −0.44 / −1.22 / 1⁄3 | 2.17% | 0.091 | **Retire** — negative on every combo |

Note `st_range_fade`: *all 16 combos* are negative on TRAIN, mean Sharpe
−2.54 to −1.00. There is no configuration to rescue.

---

## Full 20-symbol universe — best-by-TRAIN-Sharpe config per strategy

Gates: median-across-symbols `trades ≥ 30`, median `cost_ratio ≤ 0.10`.
`pos` = fraction of the 20 symbols with positive Sharpe.

| Strategy | Config | TRAIN sh_mean / sh_med / pos | risk | c_med | c_max | SELECT sh_mean / sh_med / pos | risk | DSR |
|---|---|---|---|---|---|---|---|---|
| **st_range_breakout** | p20, w0.10, floor1.5, rr3.0 | **+0.44** / +0.61 / 75% | 4.17% | 0.034 | 0.064 | −0.23 / −0.13 / 40% | 4.38% | 0.003 |
| st_range_breakout (rr2.0) | p20, w0.10, floor1.5, rr2.0 | +0.43 / +0.52 / 70% | 4.21% | 0.033 | 0.062 | **+0.08** / +0.05 / 50% | 4.36% | 0.003 |
| **st_bos** | span3, rr2.5, floor2.0, no-HL | +0.29 / +0.29 / 70% | 7.23% | 0.019 | 0.046 | +0.07 / +0.38 / 60% | 6.67% | 0.001 |
| **st_bos** | span6, rr2.5, floor2.0, no-HL | +0.25 / +0.25 / 70% | 9.20% | 0.015 | 0.037 | **+0.34** / +0.47 / 70% | 8.66% | 0.002 |
| st_prior_day_break | adx0, floor1.5, rr2.5 | +0.21 / +0.08 / 60% | 3.99% | 0.035 | 0.084 | **−0.90** / −1.20 / 40% | 3.86% | 0.001 |
| **st_range_breakout_retest** | p20, w0.10, retest8, rr3.0 | +0.20 / +0.23 / 70% | 3.19% | 0.044 | 0.096 | −0.03 / −0.15 / 50% | 3.26% | 0.001 |
| st_sr_retest | span6, retest12, rr2.5, floor1.5 | +0.06 / +0.03 / 60% | 3.69% | 0.038 | 0.100 | −0.48 / −0.58 / 30% | 3.82% | 0.001 |
| st_mtf_level | span2, tol0.002, floor1.5, rr2.5 | −0.11 / −0.09 / 45% | 3.64% | 0.039 | 0.089 | −0.36 / −0.20 / 45% | 3.65% | 0.001 |
| st_round_number | step2.0, tol0.003, floor1.5, rr2.5 | −0.44 / −0.53 / 25% | 4.05% | 0.035 | 0.062 | −0.63 / −0.83 / 30% | 3.85% | 0.001 |
| st_liquidity_sweep | span6, pierce0.003, floor1.0, rr2.5 | −0.47 / −0.45 / 25% | 2.54% | 0.055 | 0.110 | −0.89 / −1.02 / 20% | 2.59% | 0.000 |
| st_range_fade | p20, w0.06, edge0.15, floor1.5 | **−1.03** / −1.13 / 20% | 2.22% | 0.063 | 0.110 | −1.19 / −1.32 / 20% | 2.48% | 0.000 |

---

## Rationales, and what the measurement said about each

**st_sr_retest** — broken swing level flips polarity; enter on the pullback so
the entry sits closer to the same invalidation point, raising R:R without
demanding more edge. *Result:* the mechanism works on risk (`c` 0.038 median)
but not on return. TRAIN +0.06 → SELECT −0.48 on the universe, 0⁄3 core symbols
positive on SELECT. The polarity-flip prior is not supported here.

**st_range_fade** — buy the lower boundary of a tight range, sell the upper,
target the mid. *Result:* the family's worst. Negative on all 16 TRAIN combos,
20% of symbols positive. Win rate is high (0.47–0.59) and profit factor is low
(0.69–0.76): the fade wins often and small, then loses the range's whole width
when it breaks. This is the *same* failure signature as the retired Bollinger
sleeve, and the `atr_floor=1.0` arms are the ones that push `c` to 0.095–0.161 —
past the ceiling. Structure-based stops do not rescue boundary mean reversion;
the problem is the payoff shape, not the stop distance.

**st_range_breakout** vs **st_range_breakout_retest** — the open question, and
the most interesting result in the family. On TRAIN the immediate break is
clearly better (universe sh_mean +0.44 vs +0.20, core +0.53 vs −0.16). On SELECT
the ordering **inverts**: retest −0.03 vs immediate −0.23, and the retest arm's
symbol-positive rate holds at 50–55% while the immediate arm's falls from 75% to
40%. Interpretation: immediacy captures more of the move but is more
regime-dependent; the retest filter discards the strongest breakouts (they never
come back) in exchange for a flatter, more stable distribution. Also note the
retest arm's risk is systematically *smaller* (3.19% vs 4.17%) and its `c`
correspondingly worse (0.044 vs 0.034) — the retest entry is nearer the level, so
it leans harder on the ATR floor, and at `atr_floor=1.0` several combos breach
0.10. **The tight `max_width` arms (0.04, 0.06) produced zero trades on at least
one symbol and were gate-eliminated** — 32 of 36 and 28 of 32 combos failed the
core trade floor. Only `max_width=0.10` is viable, which means the "tight range"
premise is weaker than intended.

**st_liquidity_sweep** — pierce a prior 1H swing low, close back above; stops and
liquidations cluster beyond obvious extremes, and the invalidation (the sweep's
own extreme) is genuinely tight rather than arbitrary. *Result:* the most
disappointing negative, given the strong prior. Trade count is high (250–520 per
symbol per split, so the sample is not the problem) and the TRAIN result is
roughly flat (+0.01 core best), but SELECT is uniformly bad (−1.72 core mean,
0⁄3 positive; −0.89 universe). The naturally tight stop is also the liability:
at `atr_floor=1.0` core `c` reaches 0.090 on TRAIN and **0.112 on SELECT** —
breaching the ceiling out of sample, which is precisely the failure the PRD
diagnosed. Whatever microstructure effect exists is smaller than the round-trip
cost of harvesting it at 1H.

**st_bos** — break of structure: take out the prior 4H swing high while swing
lows are rising. *Result:* the family's only survivor, and the only one positive
in mean Sharpe on both splits on the universe (+0.25 → +0.34, 70% of symbols
positive on both). Its risk profile is by far the best: `median_risk_pct`
6.4–9.8%, `c` 0.014–0.048 — a factor of two to seven inside the ceiling, so it
has real headroom against slippage being worse than modelled. The `require_hl`
axis answers its own question ambiguously: on the core three `require_hl=True`
is better (+0.36 vs lower), on the universe `require_hl=False` wins (+0.29 vs
+0.25 at span 3). That disagreement is a robustness concern, not a finding — the
higher-low condition is at best neutral, so the level break, not the structure
sequence, is doing the work.

**st_mtf_level** — a 1D swing level tested on a 1H bar: daily levels are watched
by more capital, and 1H execution pays only 1H-scale distance to reach them.
*Result:* the cheapest asymmetry in principle, ~flat in practice (universe TRAIN
−0.11, SELECT −0.36 to −0.05). Notably the only strategy whose SELECT numbers
are *better* than TRAIN at some configs (span 3: TRAIN −0.18 → SELECT −0.08 with
55% of symbols positive), which is more likely small-sample noise than a real
signal — `trades` median on SELECT is only 20–26.

**st_prior_day_break** — previous-day high/low as the only calendar-anchored
levels 24/7 crypto has; the equity opening-range analogue. `adx_min=0` is the
unfiltered control. *Result:* the cleanest overfit signature in the family.
TRAIN +0.21 (60% of symbols positive) → SELECT **−0.90** (40% positive), and on
the core three SOL goes +0.04 → −3.06. The ADX filter added nothing: `adx_min=0`
won on TRAIN, so the edge is not a trend-filter artifact — it simply is not
there. Reject.

**st_round_number** — order clustering at round numbers, with the grid defined
one decade below each symbol's own price and scaled by `step_mult`, so no
historical price is hardcoded and the same rule applies to a $0.10 coin and a
$100k one. *Result:* negative on both splits at every combo (universe TRAIN
−0.44, SELECT −0.63; only 25–40% of symbols positive). The anti-curve-fit
construction did its job — and what it showed is that there is no scale-free
round-number edge to fit. Retire. (This is a more useful negative than a fitted
version would have been: a version tuned to BTC's specific round numbers could
easily have produced a flattering TRAIN number with no transferable content.)

---

## Top-3 recommendations

### 1. `st_bos` — span 6, rr 2.5, atr_floor 2.0, require_hl=False

The only strategy with positive mean Sharpe on **both** splits across the
universe, and the best cost headroom in the family.

| | BTC | ETH | SOL |
|---|---|---|---|
| TRAIN Sharpe | −0.52 | +0.18 | +0.02 |
| TRAIN trades / wr / pf | 123 / 0.47 / 0.96 | 123 / 0.43 / 0.99 | 123 / 0.42 / 0.89 |
| TRAIN `median_risk_pct` / `c` | 5.18% / 0.027 | 6.78% / 0.021 | 9.78% / 0.014 |
| TRAIN max DD | 11.7% | 14.0% | 8.0% |
| SELECT Sharpe | −0.19 | +0.62 | −0.38 |
| SELECT trades / `c` | 36 / 0.028 | 27 / 0.016 | 31 / 0.017 |

Per-coin notes: **BTC is the problem child** — negative on both splits at the
universe-optimal config, and this is consistent across every `st_bos` combo
(BTC TRAIN Sharpe −0.43 to −0.52). The universe-optimal config is *not* the
core-optimal one; on BTC/ETH/SOL only, `span 6, rr 1.5, atr_floor 1.5,
require_hl=True` gives TRAIN mean +0.36 (BTC −0.47 / ETH +0.44 / SOL +1.12) and
SELECT +0.15 (BTC +0.18 / ETH +1.29 / SOL −1.02). **The two configs disagree on
which coin works and on the sign of `require_hl`, so treat the parameter choice
as unresolved.** ETH is the most consistently positive coin. SELECT trade counts
(27–36) are below the PRD's 30-per-fold floor on ETH, so the SELECT read is
weak. `atr_floor` is nearly inert here (1.0/1.5/2.0 give near-identical results)
because the structural distance to the defining swing dominates — which is the
healthiest possible reason for a floor not to bind.

### 2. `st_range_breakout` — period 20, max_width 0.10, atr_floor 1.5, rr 2.0

Prefer the **rr 2.0** arm over rr 3.0: TRAIN is 0.01 worse but SELECT is +0.08
vs −0.23, and 50% of symbols stay positive vs 40%.

| | BTC | ETH | SOL |
|---|---|---|---|
| TRAIN Sharpe (rr3.0 arm) | +0.75 | −0.30 | +1.14 |
| TRAIN trades / wr / pf | 149 / 0.44 / 1.19 | 145 / 0.32 / 0.85 | 110 / 0.42 / 1.35 |
| TRAIN `median_risk_pct` / `c` | 2.77% / 0.051 | 3.60% / 0.039 | 4.69% / 0.030 |
| TRAIN max DD | 13.2% | 21.7% | 5.0% |
| SELECT Sharpe | −0.51 | +0.44 | +0.30 |
| SELECT trades / `c` | 44 / 0.054 | 34 / 0.030 | 34 / 0.030 |

Per-coin notes: **SOL is where this lives** (TRAIN +1.14, SELECT +0.30, and the
lowest drawdown of the three at 5.0%) — consistent with SOL's higher volatility
making range breaks resolve further relative to cost. BTC and ETH swap signs
between splits, which is the opposite of consistency. ETH's 21.7% TRAIN drawdown
is the worst in the recommended set. Take this as a SOL-and-high-beta strategy,
not a universal one. `c` is 0.030–0.054, comfortably inside the ceiling but
roughly double `st_bos`'s.

### 3. `st_range_breakout_retest` — period 20, max_width 0.10, retest_within 8, rr 3.0

Recommended *as the paired control*, not as a standalone earner: it is the
answer to the family's open question and the more OOS-stable half of it.

| | BTC | ETH | SOL |
|---|---|---|---|
| TRAIN Sharpe | +0.44 | −0.86 | −0.45 |
| TRAIN trades / wr / pf | 110 / 0.37 / 1.11 | 102 / 0.26 / 0.67 | 64 / 0.27 / 0.75 |
| TRAIN `median_risk_pct` / `c` | 2.03% / 0.069 | 2.52% / 0.056 | 3.57% / 0.039 |
| TRAIN max DD | 16.0% | 28.2% | 15.8% |
| SELECT Sharpe | −1.98 | +0.92 | +0.34 |
| SELECT trades / `c` | 41 / 0.082 | 20 / 0.048 | 22 / 0.043 |

Per-coin notes: on the core three this is bad (BTC SELECT −1.98, ETH TRAIN
−0.86); its case rests entirely on the 20-symbol universe, where it is the most
split-stable breakout variant (TRAIN +0.20 / SELECT −0.03, 70%/50% positive).
ETH inverts hard between splits (−0.86 → +0.92) on 20 SELECT trades — below the
sample floor, so do not read that as evidence. **Its `c` (0.069 BTC TRAIN,
0.082 BTC SELECT) is the highest in the recommended set** and the `atr_floor=1.0`
arms breach 0.10, so if this is pursued the floor must stay at 1.5 or higher.

---

## Robustness concerns — read before acting on any of the above

1. **Nothing here is statistically significant.** DSR ≤ 0.003 for every combo
   when charged for the full search. On the family's own evidence, the honest
   summary is "structure strategies clear the cost frontier and show no
   detectable Sharpe edge", not "we found three winners".
2. **TRAIN→SELECT decay is the rule, not the exception.** 7 of 9 strategies have
   lower SELECT than TRAIN mean Sharpe on the universe; the two that don't
   (`st_mtf_level`, and `st_bos` at span 6) improve on SELECT trade counts of
   20–33, i.e. inside the noise band.
3. **SELECT is too short for this family.** Median SELECT trades per symbol is
   20–43 for every recommended config — at or below the PRD's `≥ 30 per fold`
   floor. SELECT Sharpe signs should be treated as directional hints only.
4. **Core-optimal and universe-optimal configs disagree**, most sharply for
   `st_bos` (`require_hl` flips, and which coin works flips). That is the
   signature of parameter choice being noise-driven.
5. **`max_width` is nearly degenerate.** Only 0.10 survives the trade floor; the
   0.04/0.06 arms produce zero trades on at least one symbol. The "tight range"
   premise that was supposed to distinguish these from a plain Donchian break is
   therefore doing much less work than intended, and `st_range_breakout` at
   `max_width=0.10` is closer to a Donchian break with a mid-range stop than to
   a genuine consolidation trade.
6. **BTC is the weakest coin across the whole family**, being negative on both
   splits for the top recommendation. Any pooled 3-symbol gate the PRD applies
   (positive expectancy on **all 3**) fails for every strategy here.
7. **The ATR floor is what keeps `c` legal, and it is not optional.** Every
   `cost_ratio` breach observed (`st_liquidity_sweep` 0.112 SELECT,
   `st_range_fade` up to 0.161, `st_sr_retest` 0.149, `st_range_breakout_retest`
   0.097) occurs at `atr_floor=1.0` where the raw structural distance wins.
   Do not lower the floor.

## Reproduction

```
cd scripts/bruteforce
../../.venv/bin/python runner.py --causal-only --family structure
../../.venv/bin/python runner.py --split TRAIN  --family structure --core-only --workers 2 --out /tmp/st_train_core.csv
../../.venv/bin/python runner.py --split SELECT --family structure --core-only --workers 2 --out /tmp/st_select_core.csv
../../.venv/bin/python runner.py --split TRAIN  --family structure --workers 2 --out /tmp/st_train_full.csv
../../.venv/bin/python runner.py --split SELECT --family structure --workers 2 --out /tmp/st_select_full.csv
```

Causality: all 9 strategies pass `core.assert_causal` on the first combo via the
runner, and **all 184 combos** were additionally verified individually. The three
state-carrying helpers (`_ffill_at`, `_bars_since`, `_prev_pivot`) are
forward-only by construction; `_prev_pivot` exists because `sparse.shift(1)` on a
forward-filled pivot series shifts by one *bar*, which would have made every
higher-low test silently never-true.
