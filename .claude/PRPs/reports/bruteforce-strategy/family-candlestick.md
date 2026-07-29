# Family report: CANDLESTICK

**Module**: `scripts/bruteforce/strategies/candlestick.py` · 9 registered strategies · 106 grid combos
**Splits used**: TRAIN (2023-08-01 → 2025-07-01), SELECT (2025-07-01 → 2026-01-01). **HOLDOUT never touched.**
**Universe**: core-3 (BTC/ETH/SOL) and the full 20-symbol research universe.
**Trigger tier**: 4H for every strategy (1D for regime/levels). Causality audit: **9/9 passed**.

---

## The prior, and whether it survived

Candlesticks are the weakest-evidenced family in this search. The module was written to
test the two defensible versions only — patterns *location-conditioned* (at a trailing
extreme / 1D swing level) and patterns *as confirmation on a trend signal* — and the
results largely confirm the weak prior. **No standalone candlestick strategy in this
family is recommendable.** One matched-pair result is genuinely interesting but does not
survive out-of-sample generalisation testing across symbols.

Every stop is `max(wick_distance * 1.1, 1.5 * ATR(14, 4H))`. This worked as intended:
`median_risk_pct` is 2.7–4.3% everywhere and `cost_ratio` is **0.050–0.071** on every
reported config, comfortably inside the 0.10 ceiling. The 4H tier is what buys this; the
same patterns on 1H would reproduce the old failure.

---

## Every strategy: best-by-Sharpe TRAIN (core-3) config → SELECT

Selection rule: highest mean core-3 TRAIN Sharpe among configs passing `min trades ≥ 30`
per symbol and `cost_ratio ≤ 0.10`. Sharpe columns are the core-3 mean; `min` is the worst
of the three symbols.

| Strategy | Best TRAIN config | TRAIN Sharpe (min) | min trades | cost_ratio | median risk | SELECT Sharpe (min) | Verdict |
|---|---|---|---|---|---|---|---|
| `cs_donchian_candle_confirm` | chan 40, ema 50, rr 2.0 | **+0.99** (+0.50) | 50 | 0.065 | 3.1% | **+0.83** (+0.23) | Best in family; see caveat |
| `cs_marubozu_ignition` | body 0.8, range 1.5×ATR, rr 2.0 | +0.92 (+0.57) | 56 | 0.050 | 4.3% | **−0.71** (−2.91) | **Overfit** — sign flip |
| `cs_inside_break` | ema 50, n_inside 1, rr 3.0 | +0.86 (−0.02) | 123 | 0.064 | 2.9% | +0.37 (+0.24) | Marginal, degrades hard |
| `cs_pinbar_trend` | ema 50, rr 3.5, vol_min 1.5 | +0.63 (−0.25) | 69 | 0.065 | 2.8% | +0.58 (+0.03) | Weak but stable; ~0 universe-wide |
| `cs_wick_reject_level` | look 60, rr 3.5, wick 0.5 | +0.43 (+0.05) | 71 | 0.064 | 2.9% | **−1.60** (−2.93) | **Fails** — worst reversal |
| `cs_donchian_plain` (control) | chan 40, ema 50, rr 2.0 | +0.41 (+0.26) | 84 | 0.065 | 2.9% | +0.45 (+0.11) | Honest baseline |
| `cs_three_bar_level` | rr 3.0, span 3, tol 0.04 | +0.10 (−0.19) | 32 | 0.071 | 2.7% | −0.34 (−2.43) | **Fails** |
| `cs_engulf_extreme` | look 20, rr 3.5, vol_min 1.0 | −0.26 (−1.03) | 32 | 0.064 | 3.5% | −0.08 (−0.41) | **Fails** |
| `cs_streak_fade` (control) | n 4, rr 1.0, rsi 75 | **−1.20** (−2.11) | 34 | 0.065 | 3.1% | **−1.39** (−2.03) | **Fails, informatively** |

Production Donchian on TRAIN for reference: BTC +0.26 / ETH −0.45 / SOL +0.62.

### The same configs on the full 20-symbol universe

| Strategy | TRAIN mean Sharpe | % symbols > 0 | SELECT mean Sharpe | % symbols > 0 |
|---|---|---|---|---|
| `cs_inside_break` | +0.47 | 70% | −0.34 | 50% |
| `cs_donchian_candle_confirm` | +0.32 | 75% | **+0.16** | 65% |
| `cs_donchian_plain` | +0.30 | 75% | **+0.21** | 65% |
| `cs_marubozu_ignition` | +0.27 | 70% | −0.93 | 25% |
| `cs_engulf_extreme` | +0.08 | 50% | +0.11 | 50% |
| `cs_three_bar_level` | −0.05 | 45% | −0.36 | 45% |
| `cs_wick_reject_level` | −0.18 | 55% | −1.12 | 20% |
| `cs_pinbar_trend` | −0.23 | 40% | −0.00 | 60% |
| `cs_streak_fade` | −0.44 | 25% | −0.03 | 40% |

Read that table before the core-3 one. Universe-wide, the *only* two strategies that stay
positive on both splits are the Donchian pair — and the unfiltered control is the better
of the two.

---

## THE HEADLINE QUESTION: do candlestick patterns add value as a confirmation filter?

**Answer: on the core-3 symbols, apparently and repeatably yes; universe-wide, no —
statistically indistinguishable from zero. The honest verdict is NO VALUE ESTABLISHED.**

`cs_donchian_plain` and `cs_donchian_candle_confirm` are byte-identical except that the
treatment arm additionally requires the breakout bar to be *decisive* — a body filling
≥70% of its range, or an engulfing of the prior bar. Same channel, same 1D EMA trend
filter, same frozen 1.5×ATR stop, same R:R, same 8-combo grid. The Sharpe difference is
therefore a clean estimate of what the candle is worth.

### Core-3, mean across the three symbols, all 8 configs

| chan / ema / rr | TRAIN plain | TRAIN filtered | Δ | SELECT plain | SELECT filtered | Δ |
|---|---|---|---|---|---|---|
| 20 / 100 / 2.0 | +0.15 | +0.75 | +0.61 | −0.07 | +0.96 | +1.02 |
| 20 / 100 / 3.0 | +0.26 | +0.81 | +0.55 | −0.35 | +0.54 | +0.89 |
| 20 / 50 / 2.0 | +0.34 | +0.92 | +0.58 | +0.14 | +0.56 | +0.42 |
| 20 / 50 / 3.0 | +0.41 | +0.97 | +0.56 | −0.41 | +0.20 | +0.61 |
| 40 / 100 / 2.0 | +0.35 | +0.85 | +0.49 | +0.26 | +1.08 | +0.82 |
| 40 / 100 / 3.0 | +0.30 | +0.83 | +0.53 | −0.22 | +0.74 | +0.96 |
| 40 / 50 / 2.0 | +0.41 | +0.99 | +0.57 | +0.45 | +0.83 | +0.38 |
| 40 / 50 / 3.0 | +0.35 | +0.84 | +0.49 | −0.11 | +0.66 | +0.77 |

Core-3 paired delta: **TRAIN +0.548 (t = +7.1, n=24)**, **SELECT +0.735 (t = +4.4, n=24)**.
The filter is positive in **16 of 16** cells. It also improves the mechanics in the way the
premise predicts, not just the headline: win rate +4 to +8 pp (TRAIN 0.29→0.34 at rr 3.0,
0.38→0.44 at rr 2.0), profit factor 1.03→1.32, at the cost of ~30–35% of trades. That is
the signature of a filter removing bad trades rather than removing trades at random.

### The same comparison across all 20 symbols — where it falls apart

| | paired n | mean Δ | sd | t | % of pairs Δ>0 |
|---|---|---|---|---|---|
| TRAIN, 20 symbols | 160 | **+0.058** | 0.520 | +1.41 | 56% |
| SELECT, 20 symbols | 160 | **+0.071** | 1.079 | +0.83 | 53% |
| TRAIN, core-3 only | 24 | +0.548 | — | +7.09 | 100% |
| SELECT, core-3 only | 24 | +0.735 | — | +4.44 | 100% |

Per-symbol Δ (mean over the grid) on SELECT ranges from **+1.48 (SOL)** and **+1.31 (AVAX)**
to **−1.53 (FIL)** and **−1.14 (LTC)**. The filter helps 55% of symbols — a coin flip. The
core-3 effect is ~10× the universe effect, on the three symbols that are also the most
searched-over assets in this repo's entire history.

**Interpretation.** Two readings are available and I will not pretend the data chooses
between them:
1. *Real but asset-specific.* The effect replicates out-of-sample on the same three
   symbols with a plausible mechanism (a mid-range close on a channel break is a probe;
   a close on the bar's extreme is a commitment), and majors are where 4H order flow is
   most informative.
2. *A three-symbol coincidence.* n=3 assets is n=3, the universe test is flat, and BTC/ETH/SOL
   are exactly the symbols against which every prior sweep in this project was run.

Given the PRD's stated prior on this family and the flat 20-symbol result, reading (2) is
the responsible default. The filter should be treated as **unproven**, and the *unfiltered*
Donchian break is the better universe-wide performer (SELECT +0.21 vs +0.16).

---

## What failed, and why

- **`cs_streak_fade` (−1.20 TRAIN / −1.39 SELECT, 25% of symbols positive).** This is the
  most informative failure in the module. It was registered as a *control on the exhaustion
  premise*: 4 consecutive same-direction 4H closes plus an RSI extreme is the cleanest
  measurable form of "the move is stretched, fade it". It loses money consistently, on both
  splits, on most symbols. Since every reversal candle (hammer, engulfing, three-bar) rests
  on that same premise, this result undercuts the whole reversal sub-family independently of
  shape. Crypto 4H momentum continues; it does not exhaust on a schedule.
- **`cs_wick_reject_level` (+0.43 → −1.60).** The largest TRAIN→SELECT collapse. A failed
  breakout at a 60-bar extreme is a textbook setup and it is simply not one here: in a
  trending 2025 H2, "poked above the trailing high and closed back below" is a *pause in an
  uptrend*, and shorting it is shorting the trend. Textbook status was the only thing
  supporting it, and that is not evidence.
- **`cs_marubozu_ignition` (+0.92 → −0.71, BTC −2.91).** The clearest overfit in the family.
  It looked like the best standalone idea on TRAIN and inverted completely; universe-wide it
  drops from +0.27 to −0.93 with only 25% of symbols positive. The mechanism (one-sided
  order flow) is the most defensible in the module and it still did not hold, which is a
  useful reminder that a good story is not a result.
- **`cs_engulf_extreme` (−0.26 / −0.08).** Failed twice. First, the *initial* implementation
  produced **literally zero trades** across all 18 combos, because location was tested on
  the bar's close: a bullish engulfing bar closes well off its low by definition, so
  "closes within 0.5% of the 60-bar low" is unsatisfiable. Fixed to test the bar's *low*
  against the trailing low (documented in the code). Once judgeable, it is flat-to-negative
  and only 3 of 18 combos even clear the ≥30-trades gate. Engulfing bars at extremes are
  rare *and* uninformative.
- **`cs_three_bar_level` (+0.10 / −0.34).** Sample-starved at the original tolerance grid
  (n≈7 per symbol, unjudgeable). The proximity band was widened once, to 0.02/0.04/0.06, for
  sample adequacy — not for Sharpe — after which it is judgeable and negative. Noted as a
  consumed degree of freedom.
- **`cs_pinbar_trend` (+0.63 / +0.58).** Does not fail outright and is the most *stable*
  standalone strategy in the family, but it is only stable around zero: −0.23 TRAIN /
  −0.00 SELECT universe-wide, 40%/60% of symbols positive. A hammer-timed trend entry
  neither helps nor hurts.

---

## Top 3 recommendations

### 1. `cs_donchian_plain` — chan 40, ema 50, rr 2.0 (the control wins)
TRAIN core-3 **+0.41** (BTC +0.26 / ETH +0.54 / SOL +0.44, n = 91/84/108) ·
SELECT **+0.45** (BTC +0.11 / ETH +0.50 / SOL +0.74, n = 27/26/25) ·
universe TRAIN +0.30 / SELECT +0.21, 65–75% of symbols positive.

The most robust thing this family produced is the arm with no candlesticks in it. It is
positive on all three core symbols on both splits, positive on two-thirds of a 20-symbol
universe on both splits, `cost_ratio` 0.065, and it beats production Donchian on ETH
(+0.54 vs −0.45) — the symbol production fails on — while giving up a little on SOL. It is
a 4H-triggered Donchian break with a 1D EMA(50) trend filter and nothing else, which is
about as few degrees of freedom as a strategy can have. **Per-coin note:** ETH is the
gain, SOL is the strongest but also the least stable, BTC is thin (+0.11 SELECT).

### 2. `cs_donchian_candle_confirm` — chan 40, ema 50, rr 2.0 (promising, unproven)
TRAIN core-3 **+0.99** (ETH +0.50 / BTC +0.90 / SOL +1.56, n = 50/59/62) ·
SELECT **+0.83** (ETH +0.23 / BTC +0.28 / SOL +1.99, n = 17/18/17) ·
universe TRAIN +0.32 / SELECT +0.16.

Best core-3 numbers in the family and the only strategy here that clearly beats the
production baseline on all three symbols on TRAIN. Carry it forward **only as a paired
candidate with #1**, never alone, and only if the shortlist can afford to spend holdout on
a hypothesis whose 20-symbol test is flat. **Per-coin note:** the entire core-3 edge is
SOL-weighted (+1.56 / +1.99); BTC and ETH SELECT Sharpe are +0.28 and +0.23 on n = 18 and
19 trades, which is below the PRD's 30-trade adequacy bar per symbol per split. That
sample thinness is the filter's real problem: it cuts 30–35% of trades from an already
low-frequency 4H rule.

### 3. `cs_inside_break` — ema 50, n_inside 1, rr 3.0 (watch only)
TRAIN core-3 +0.86 (BTC +1.67 / ETH +0.92 / SOL −0.02, n = 123/131/151) ·
SELECT +0.37 (BTC +0.24 / ETH +0.37 / SOL +0.49) · universe TRAIN +0.47 / SELECT −0.34.

Included over `cs_pinbar_trend` for one reason: it is the only strategy here with a
comfortable sample (n > 120 per symbol on TRAIN, ≥30 on SELECT) and it stays positive on
all three core symbols on SELECT. But it is a volatility-compression idea wearing a
candlestick costume, it goes negative universe-wide on SELECT, and its TRAIN edge is
BTC-concentrated. **Recommendation: do not shortlist; hand the compression premise to the
volatility family, which can test it with a proper band statistic.**

---

## Robustness concerns

- **Sample adequacy is this family's binding constraint, not Sharpe.** At a 4H trigger over
  a 6-month SELECT window, most strategies land 15–30 trades per symbol. The PRD's ≥30
  trades/symbol bar is met on SELECT by `cs_inside_break` alone. Every SELECT Sharpe above
  is computed on a sample too small to separate skill from luck, and `cs_donchian_candle_confirm`
  is the worst affected because filtering removes a third of an already sparse signal.
- **Core-3 vs universe divergence is the dominant robustness signal.** Five of nine
  strategies (`marubozu`, `inside_break`, `wick_reject`, `three_bar`, `streak_fade`) look
  materially better on BTC/ETH/SOL than on 20 symbols. The core-3 leaderboard in this
  family should be treated as contaminated by symbol selection.
- **One consumed degree of freedom, logged:** `cs_three_bar_level`'s `tol` grid was widened
  once (0.01/0.02/0.03 → 0.02/0.04/0.06) for sample adequacy. `cs_engulf_extreme`'s
  location test was corrected from close-based to extreme-based after it produced zero
  trades. Neither change was made in response to a Sharpe reading.
- **K_STOP was never swept** (frozen 1.5), `WICK_BUFFER` fixed at 1.1, cost model untouched.
  `cost_ratio` ≤ 0.071 everywhere, so no result here is a cost-model artefact.
- **Direction asymmetry untested.** All strategies are two-sided. The 2023–2025 TRAIN span is
  net bullish, and several failures (`wick_reject`, `streak_fade`) are plausibly short-side
  losses being averaged with tolerable long-side results. A long-only re-test of the two
  Donchian arms would be a cheap and informative follow-up.
