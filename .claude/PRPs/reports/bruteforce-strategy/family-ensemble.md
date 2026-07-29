# Family report: ENSEMBLE (+ the entry-vs-exit factorial)

**Author**: ensemble analyst · **Date**: 2026-07-27
**Module**: `scripts/bruteforce/strategies/ensemble.py` (8 strategies, 82 combos)
**Splits used**: TRAIN (2023-08-01 → 2025-07-01), SELECT (2025-07-01 → 2026-01-01).
**HOLDOUT: not touched.**
**CSVs**: `/tmp/ens_train.csv`, `/tmp/ens_select.csv` (20 symbols × 82 combos = 1,640 rows each);
per-combo summary `/tmp/ens_combo_summary.csv`.
**Causality**: 8/8 pass `core.assert_causal` at `tail_skip=0`; `lookahead_canary`
still fails in the same run, so the audit is live. `core.py`, `registry.py`,
`indicators.py` and the cost model were not modified.

All Sharpe figures below are **mean or median per-symbol** Sharpe over the
20-symbol universe with the fraction of symbols positive. **No pooled or
portfolio Sharpe is quoted anywhere**: `equity.daily_returns` books each trade's
whole PnL on its exit day with no daily mark-to-market, so pooling harvests fake
√N diversification (measured cross-symbol ρ 0.02–0.09 against a true ~0.27–0.33).
Core-3 is reported for the PRD gate only, never used for selection.

---

## PART 1 — THE VERDICT: it is the EXIT. The entry does not matter.

**The ratcheting ATR trail is the active ingredient. The Keltner-vs-Donchian
entry choice is indistinguishable from zero and its sign is not even stable
across splits. There is no meaningful interaction — if anything the trail helps
the Donchian entry *more* than the Keltner one.**

`fac_entry_exit` crosses entry ∈ {N-bar 4H extreme (Donchian), ATR channel
(Keltner, multiple frozen at 2.0)} against exit ∈ {ratcheting ATR trail, fixed
ATR target, return-to-mean}, with everything else held constant: stop frozen at
`1.5 × ATR(14)` of the 4H setup tier, 1H trigger, harness-default 96-bar time
stop, both directions, shared lookback axis {20, 55} and shared ATR-multiple axis
{2.0, 3.0}. 24 combos × 20 symbols.

### Marginal effect of the EXIT (pooled over entry, lookback, geometry)

| exit | TRAIN mean Sh | TRAIN med | TRAIN % pos | TRAIN PF | SELECT mean Sh | SELECT med | SELECT % pos | SELECT PF | med DD@target |
|---|---|---|---|---|---|---|---|---|---|
| **trail** | **+0.262** | +0.377 | **67.5%** | 0.996 | **+0.316** | +0.570 | **76.2%** | **1.138** | 24.2% / 10.0% |
| target | +0.149 | +0.186 | 60.0% | 0.953 | **−0.415** | −0.420 | 37.5% | 0.871 | 27.9% / 17.1% |
| return-to-mean | −0.004 | −0.073 | 44.4% | 0.924 | −0.228 | −0.003 | 50.0% | 0.923 | 27.6% / 14.9% |

Trail minus pooled no-trail: **+0.19 Sharpe on TRAIN, +0.64 on SELECT**, and it
is the only exit whose profit factor exceeds 1.0 on either split. It also cuts
drawdown at the vol target by roughly a third on SELECT.

### Marginal effect of the ENTRY (pooled over exit, lookback, geometry)

| entry | TRAIN mean Sh | TRAIN % pos | SELECT mean Sh | SELECT % pos |
|---|---|---|---|---|
| Donchian (N-bar extreme) | +0.101 | 57.5% | −0.080 | 55.0% |
| Keltner (ATR channel) | +0.171 | 57.1% | −0.138 | 54.2% |

Keltner minus Donchian: **+0.07 on TRAIN, −0.06 on SELECT**, with the fraction of
symbols positive identical to within half a percentage point on both splits. The
entry factor is a null result. The trend family's belief that
"volatility-normalised beats N-bar extreme" does not replicate once the exit is
controlled — `trend_keltner_trail` looked like an entry finding because it was
the family's *only* trailed channel strategy.

### The 2×2 (trail vs no-trail × entry) — no interaction, and the trail rescues Donchian

| | TRAIN mean Sh | TRAIN % pos | SELECT mean Sh | SELECT % pos |
|---|---|---|---|---|
| Donchian, no trail | +0.022 | 51.9% | **−0.308** | 43.8% |
| **Donchian, trail** | **+0.259** | 68.8% | **+0.374** | **77.5%** |
| Keltner, no trail | +0.123 | 52.5% | −0.336 | 43.8% |
| **Keltner, trail** | **+0.265** | 66.2% | **+0.258** | 75.0% |

Trail effect: **+0.237 (TRAIN) / +0.682 (SELECT)** on the Donchian entry;
**+0.142 / +0.594** on the Keltner entry. The two trailed cells are
statistically indistinguishable from each other on both splits (+0.259 vs +0.265
TRAIN; +0.374 vs +0.258 SELECT), while both untrailed cells are negative on
SELECT. **The whole spread in this factorial lies on the exit axis.**

### Per-symbol trail effect (pooled over entry, lookback, geometry)

| symbol | TRAIN no-trail → trail (Δ) | SELECT no-trail → trail (Δ) |
|---|---|---|
| AAVE | −0.35 → −0.21 (+0.14) | −1.38 → +0.72 (**+2.10**) |
| ADA | +0.35 → +0.70 (+0.35) | +0.83 → +1.41 (+0.58) |
| APT | −0.59 → −0.75 (−0.16) | −0.40 → +0.40 (+0.80) |
| ATOM | −0.16 → −0.06 (+0.10) | −1.07 → +0.32 (+1.39) |
| AVAX | +0.76 → +1.02 (+0.26) | +0.16 → +1.16 (+1.00) |
| BCH | −0.90 → −0.58 (+0.32) | −3.05 → −3.21 (−0.16) |
| BNB | −0.32 → −0.01 (+0.31) | +0.15 → +0.19 (+0.04) |
| **BTC** | +0.47 → +0.73 (+0.26) | −0.12 → +0.47 (+0.59) |
| DOGE | +0.97 → +1.02 (+0.05) | −0.14 → +0.49 (+0.63) |
| DOT | +0.03 → +0.08 (+0.05) | −0.51 → +0.66 (+1.17) |
| **ETH** | +1.07 → +1.18 (+0.11) | +1.55 → +1.87 (+0.32) |
| FIL | −0.14 → −0.04 (+0.10) | −0.54 → −0.13 (+0.41) |
| LINK | +0.26 → +0.40 (+0.14) | +0.33 → +0.70 (+0.37) |
| LTC | −1.10 → −0.91 (+0.19) | −1.58 → −1.97 (−0.39) |
| NEAR | +0.21 → +0.09 (−0.12) | −0.84 → −0.76 (+0.08) |
| OP | −0.28 → +0.03 (+0.31) | −0.61 → +0.20 (+0.81) |
| **SOL** | +0.77 → +0.73 (−0.04) | +0.52 → +0.60 (+0.08) |
| TRX | +0.41 → +1.10 (+0.69) | +0.19 → +1.24 (+1.05) |
| UNI | +0.15 → +0.42 (+0.27) | −0.21 → +1.14 (+1.35) |
| XRP | −0.16 → +0.32 (+0.48) | +0.27 → +0.82 (+0.55) |

**17 of 20 symbols improve on TRAIN (exceptions APT −0.16, NEAR −0.12, SOL
−0.04) and 18 of 20 on SELECT (exceptions LTC −0.39, BCH −0.16).** A one-factor
change that is directionally positive on 35 of 40 symbol-splits is about as close
to a robust component finding as this search has produced. Note that the two
symbols the trail does not help on SELECT (BCH, LTC) are negative under every
configuration in every family report — they are bad vehicles, not
counter-evidence about the trail.

Core-3 view for the gate (exit marginal, pooled): TRAIN trail **+0.878** vs
target +0.763 vs return-to-mean +0.783; SELECT trail **+0.982** vs target +0.496
vs return-to-mean +0.810. The effect is present on the core three but *smaller*
than universe-wide — the reverse of the usual core-3 inflation, and further
evidence it is real rather than a symbol-selection artifact.

### What this means for the production A-core

The frozen production baseline `donchian_production` scores **−0.184 TRAIN /
−1.131 SELECT (20% of symbols positive)** universe-wide. The factorial's
Donchian-entry-plus-trail cells score **+0.259 / +0.374 (77.5% positive)**, and
the single best cell (`donchian / trail / look 55 / geom 2.0`) scores TRAIN
+0.148 (60% positive) / **SELECT +0.673 (80% positive)**, with 30–40 SELECT
trades per symbol — above the adequacy floor — and `cost_ratio` 0.030–0.108.
The `look 20 / geom 2.0` cell is the more balanced one: TRAIN **+0.461** (80%
positive) / SELECT **+0.327** (75% positive).

So: **the PRD spent two iterations tuning the entry of a rule whose failure was
in the exit, and the fix is one array.** `Plan.trail_atr`-equivalent behaviour in
production is a ratcheting stop at `k × ATR(4H)` behind the favourable extreme,
updated only *after* the current bar's exits resolve. That ordering is
load-bearing: a trail that ratchets before exit resolution is a lookahead in live
code even though `core.simulate` gets it right (`core.py:448-459`, and pinned by
`test_harness.py`).

Two caveats on the verdict, stated plainly:

1. **The trail is not free.** It reduces trade count materially (TRAIN median 188
   vs 254 for the target arm) and it lowers win rate to ~0.32–0.34 against ~0.42
   for the target arm. The mechanism is exactly the intended one — it converts a
   capped-winner, high-hit-rate payoff into an uncapped, low-hit-rate one, and
   profit factor rises from 0.87–0.95 to 1.00–1.14. **Anyone reading "improve the
   win rate" literally should note that the change that improves profit
   *lowers* the win rate.**
2. **A cheaper explanation I cannot fully exclude**: the trail systematically
   shortens holding time, and SELECT (2025-H2) was a mean-reverting window in
   which shorter holds paid. The `return-to-mean` arm partially controls for this
   (it also exits early, and it does not work), which weakens but does not kill
   the objection. A third window would settle it; HOLDOUT is not mine to spend.

---

## PART 2 — Ensembles

Seven composites, each answering a registered question. Per-strategy, all combos
pooled over the 20-symbol universe:

| Strategy | TRAIN mean Sh | TRAIN med | TRAIN % pos | SELECT mean Sh | SELECT med | SELECT % pos | med trades T/S | max `c` | max DSR |
|---|---|---|---|---|---|---|---|---|---|
| `ens_lowvol_keltner_trail` | **+0.515** | +0.506 | 79.6% | **+0.760** | +1.085 | 80.0% | 88 / 23 | 0.132 | **0.573** |
| `ens_vote3_majority` | +0.472 | +0.445 | 76.9% | +0.403 | +0.626 | 72.5% | 126 / 33 | 0.108 | 0.240 |
| `ens_vol_gate_ladder` | +0.450 | +0.440 | 78.3% | +0.672 | +0.929 | 78.3% | 143 / 38 | 0.128 | 0.327 |
| `ens_regime_dispatch` | +0.414 | +0.399 | 78.1% | +0.486 | +0.981 | 73.8% | 243 / 62 | 0.096 | 0.151 |
| `ens_vote2_kelt_mom` | +0.397 | +0.454 | 75.6% | +0.010 | +0.347 | 56.2% | 68 / 18 | 0.103 | 0.228 |
| `ens_mom_accel_trail` | +0.378 | +0.382 | 73.1% | +0.052 | +0.406 | 61.2% | 61 / 14 | 0.037 | 0.336 |
| `ens_bos_trail` | +0.323 | +0.365 | 73.1% | +0.357 | +0.531 | 66.9% | 262 / 67 | 0.054 | 0.367 |

Every one of the seven is positive in mean Sharpe on **both** splits with ≥66% of
symbols positive — which no prior family managed for more than one strategy. That
is not seven discoveries; it is one discovery (the trail) applied seven times,
plus the volatility gate.

### The low-volatility preference replicates, and it beats the production veto

`ens_vol_gate_ladder` is the controlled three-arm test: one fixed rule (Keltner
20 × 2.0 + ATR trail), three gate shapes, nothing else varying.

| gate | TRAIN mean Sh | TRAIN % pos | TRAIN PF | SELECT mean Sh | SELECT % pos | SELECT PF | med trades T/S | max `c` |
|---|---|---|---|---|---|---|---|---|
| `none` (no gate) | +0.353 | 72.5% | 0.949 | +0.406 | 75.0% | 1.205 | 175 / 43 | 0.106 |
| `veto` (production: suppress > 90th pct) | +0.438 | 72.5% | 1.042 | +0.621 | 77.5% | 1.342 | 155 / 39 | 0.106 |
| **`lowpref` (keep quietest 50%)** | **+0.559** | **90.0%** | **1.217** | **+0.989** | 82.5% | **1.746** | 92 / 27 | **0.128** |

Monotone `lowpref > veto > none` on both splits, on every column except cost.
This is an independent replication of the volatility family's finding on a
*different* base rule — theirs was Donchian-based, this one is the rule that
actually survived. `ens_lowvol_keltner_trail`'s `pr_max` ladder says the same
thing continuously: TRAIN +0.520 / +0.561 / +0.465 and SELECT +0.898 / +0.783 /
+0.600 at `pr_max` 0.4 / 0.6 / 0.8.

**But the low-vol preference has a cost problem the volatility family did not
flag.** Filtering to quiet tape shrinks median `risk_pct` (the stop is
ATR-derived), so `c = round-trip cost / median risk` rises. At `pr_max` 0.4–0.6
the worst-symbol `c` reaches **0.101–0.132** — through the 0.10 ceiling on TRX
(1.4% median risk) and BTC on SELECT (0.103). The tighter the vol filter, the
better the Sharpe and the worse the cost ratio, and the two cross right around
`pr_max` 0.6. This is the same cost-frontier trap that killed the Bollinger
sleeve, arriving through a new door. Any low-vol gate that ships must either sit
at `pr_max` ≥ 0.8, or exclude the low-ATR% symbols (TRX, BTC, BNB), or widen `k`
— and widening `k` is a frozen parameter, so it is not available.

### What the other composites showed

- **`ens_regime_dispatch`** (momentum leg in ADX>25 tape, Keltner leg in quiet
  tape, flat otherwise) works and is the most sample-adequate composite:
  159–229 TRAIN and 43–55 SELECT trades per symbol at every gated combo,
  `c ≤ 0.096`. `adx_min=25` beats 20 on both splits (+0.485/+0.539 vs
  +0.344/+0.432), so the regime threshold is carrying information rather than
  just cutting sample. It is, however, essentially the trail finding plus a vol
  filter with extra machinery: it does not beat `ens_lowvol_keltner_trail`.
- **`ens_vote3_majority`** — majority (2-of-3) clearly beats unanimity (3-of-3):
  TRAIN +0.541 vs +0.402, SELECT +0.440 vs +0.366, and unanimity drops median
  SELECT trades from 48 to 21. Combining three families' survivors on direction
  does add something: 2-of-3 at span 6 is the only recommendation positive on all
  three core coins on **both** splits.
- **`ens_vote2_kelt_mom`** — the pairwise version **fails**, and instructively:
  TRAIN +0.397 → SELECT +0.010, 56% of symbols positive, median 18 SELECT trades.
  Requiring a fast channel break to coincide with a slow 1D momentum state is a
  conjunction of two sparse conditions; it starves the sample without improving
  per-trade quality (PF 0.914 on SELECT). This reproduces the trend family's
  finding about 1D confirmation filters *even when the confirmer has its own
  measured edge* — so the problem is conjunction, not the confirmer's quality.
- **`ens_mom_accel_trail`** — the trail overlay on `mom_accel` improves TRAIN
  (+0.378 pooled vs the momentum family's +0.712 for the best cell, so not
  directly comparable; at the matched `lookback 30 / gap 20` cell it is +0.477
  TRAIN / +0.585 SELECT vs the family's reported +0.712 / +0.430). It has the
  **best cost profile in this module** (`c ≤ 0.037` at 9–20% median risk) but
  only **8–17 SELECT trades per symbol** at the 4H/180-bar tier. The trail
  generalises to a momentum entry, which is the corroboration Part 1 wanted, but
  the sample is too thin to rank on.
- **`ens_bos_trail`** — replacing `st_bos`'s fixed R:R target with a ratchet
  turns the structure family's +0.25/+0.34 into **+0.395/+0.370 at 70%/65% of
  symbols positive with 271 TRAIN and 66 SELECT trades per symbol and `c` of
  0.011–0.034** — a factor of three to nine inside the ceiling, the best cost
  headroom in the entire search. Its Sharpe is the lowest of the seven, but it is
  the only candidate whose result would survive slippage being three times worse
  than modelled.

---

## Top-3 recommendations

Selected on TRAIN/SELECT agreement, breadth of per-symbol positivity, and sample
adequacy — **not** on peak Sharpe, and never on core-3.

### #1 `ens_lowvol_keltner_trail` — mult 2.0, pr_max 0.8, trail_k 2.0

TRAIN **+0.549** mean / +0.575 median, 80% of symbols positive, min 137 trades;
SELECT **+0.666** / +0.980, 80% positive, min 27 trades. Max `c` 0.098 TRAIN /
0.109 SELECT. **DSR 0.387 — the highest observed anywhere in the ~1,280-combo
search** (previous best 0.205), still far under the 0.95 threshold.

| Coin | TRAIN Sh | TRAIN trades | `c` | PF | DD@target | SELECT Sh | SELECT trades | `c` | PF |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | +0.76 | 167 | 0.072 | 1.12 | 14.7% | **+2.59** | 27 | **0.099** | 3.36 |
| ETHUSDT | **+1.76** | 142 | 0.055 | 1.78 | 11.0% | **+2.07** | 44 | 0.047 | 1.51 |
| SOLUSDT | +1.31 | 150 | 0.039 | 1.37 | 18.2% | +1.28 | 38 | 0.039 | 1.29 |

Positive on all three core coins on both splits with ≥27 trades per coin per
split — it passes the PRD's cross-symbol consistency gate, which no prior
family's top pick did on SELECT. Outside the core: AVAX, ADA, DOGE, UNI, TRX,
LINK, AAVE positive on both. **BCH (−0.29 / −2.32) and FIL (−0.13 / −0.87) are
negative on both and must be excluded**; BNB inverts catastrophically
(+0.63 → −2.22) and LTC/DOT/NEAR/OP are noise. BTC's `c` at 0.099 sits on the
ceiling and TRX at 0.109 is through it — the two lowest-ATR% names are the ones
the vol filter pushes over the cost frontier.

**Do not promote the higher-Sharpe `pr_max` 0.4/0.6 arms** (+0.635/+1.172 and
+0.618/+1.022) despite their better numbers: 15–22 minimum SELECT trades and
`c` up to 0.132. They are the same finding with the sample and the cost gate
spent.

### #2 `ens_vote3_majority` — span 6, vote_min 2, trail_k 2.0

The most sample-adequate and most cost-comfortable of the three: 188–227 TRAIN
and 44–65 SELECT trades on **every** symbol, `c ≤ 0.098`. TRAIN **+0.553** /
+0.441, 75% positive; SELECT **+0.499** / +0.772, 75% positive. DSR 0.151.

| Coin | TRAIN Sh | TRAIN trades | `c` | SELECT Sh | SELECT trades | `c` |
|---|---|---|---|---|---|---|
| BTCUSDT | +0.31 | 219 | 0.067 | +0.77 | 45 | 0.083 |
| ETHUSDT | +1.14 | 195 | 0.051 | +1.32 | 50 | 0.046 |
| SOLUSDT | +1.43 | 189 | 0.036 | +0.77 | 48 | 0.039 |

**The only recommendation positive on all six core coin-split cells with ≥45
trades in every one** — the strongest sample-backed pass of the PRD's consistency
gate in the whole search. Best extended names: DOGE +1.47/+1.57, AVAX
+1.87/+0.81, ADA +0.57/+1.82, TRX +1.09/+1.19, UNI +0.72/+1.25. Persistent
losers to exclude: **BCH (−0.04/−2.31), LTC (−0.24/−0.37), BNB (−0.07/−0.23),
APT (−0.17/−0.09)**; FIL and ATOM are flat-to-negative. Prefer this over #1 if
sample adequacy and cost headroom matter more than headline Sharpe — which,
given that nothing here is statistically significant, is a defensible position.

### #3 `ens_regime_dispatch` — adx_min 25, pr_max 0.5, trail_k 3.0

TRAIN **+0.494** / +0.457 with **85% of symbols positive** (the best breadth in
the module) and min 159 trades; SELECT **+0.556** / +1.140, 70% positive, min 43
trades. `c ≤ 0.092`, comfortably legal on every symbol on both splits.

| Coin | TRAIN Sh | TRAIN trades | `c` | SELECT Sh | SELECT trades | `c` |
|---|---|---|---|---|---|---|
| BTCUSDT | **+1.15** | 192 | 0.068 | **+1.44** | 48 | 0.074 |
| ETHUSDT | +1.04 | 198 | 0.050 | **+2.18** | 43 | 0.045 |
| SOLUSDT | +0.58 | 184 | 0.034 | **−0.06** | 51 | 0.036 |

**This is the only candidate in the search that is strong on BTC**, which every
prior family named as its problem child (structure: negative on both splits;
volatility: −0.42 SELECT; production Donchian: the whole reason we are here).
SOL going flat on SELECT (−0.06) fails the strict all-three gate. Best extended:
ADA +0.45/+2.53, LINK +0.87/+1.94, XRP +1.26/+1.62, DOGE +1.05/+1.90, AVAX
+1.01/+1.61. **Worst: FIL +0.40 → −2.81, BCH −0.08 → −2.25, NEAR +0.06 → −1.80,
LTC −0.44 → −1.79 — a heavier left tail than #1 or #2**, and the reason it is
third despite the best TRAIN breadth. The dispatch adds real machinery (two
engines, two regime thresholds) for no advantage over #1; include it mainly as a
BTC sleeve and as a diversifier whose leg composition differs from both others.

### Universal per-coin guidance

**BCHUSDT is negative on both splits under every strategy in this module and in
every prior family report — drop it from the universe outright.** LTCUSDT is
negative on both splits in six of seven composites. FILUSDT, NEARUSDT and
APTUSDT are TRAIN-positive/SELECT-negative in most, i.e. the classic overfit
signature. **ETHUSDT is the single most reliable vehicle** (positive on both
splits in all seven composites, +1.3 to +2.8 on SELECT) — notable because it is
production Donchian's worst coin at −0.45. AVAX, ADA, DOGE, UNI, TRX, LINK and
XRP form a stable positive extended cohort.

---

## Does anything clear mean per-symbol Sharpe ≥ 1.0 universe-wide?

**No.** Best gated TRAIN mean per-symbol Sharpe across all 82 combos is **+0.589**
(`ens_vote3_majority` span 3 / vote_min 2 / trail_k 2.0); best ungated is
**+0.635**. Three combos exceed 1.0 on SELECT (`ens_vol_gate_ladder` lowpref
+1.200, `ens_lowvol_keltner_trail` pr_max 0.4 +1.172 and pr_max 0.6 +1.022) but
all three fail the ≥30-trades-per-symbol adequacy bar on SELECT (15–22 minimum)
and two breach `c ≤ 0.10`. **On the universe-wide measure the PRD's Sharpe ≥ 1.0
gate is not met by anything in this module.** It *is* met on the core three by
#1 and #3 on both splits — which is precisely the ~2× core-3 inflation three
analysts independently documented, and should be read as gate compliance, not as
expected performance.

---

## Robustness concerns — read before acting

1. **DSR does not clear.** Max 0.573 (`ens_lowvol_keltner_trail`), which is
   nearly 3× the previous best in the search and still nowhere near 0.95. Charged
   against ~1,280 combos × 20 symbols, nothing here is significant after
   multiple-testing correction. This is the expected finding, not a defect — but
   no result in this report may be described as statistically significant.
2. **SELECT is six months.** 15–65 trades per symbol depending on strategy. Every
   SELECT Sharpe here carries a standard error on the order of ±0.5–0.9. The
   Part 1 verdict is the exception worth trusting more, because it is a
   *difference* between arms measured on the same bars, on 20 symbols, in the
   same direction 35 times out of 40 — a paired design is far less exposed to
   window noise than a level comparison.
3. **The low-vol preference trades Sharpe against the cost gate.** Tightening
   `pr_max` monotonically improves Sharpe and monotonically worsens `c`, crossing
   0.10 around `pr_max` 0.6 on the low-ATR% names. Do not read the `pr_max` 0.4
   numbers as achievable; `k` is frozen, so there is no headroom to buy back.
4. **`ens_vol_gate_ladder`'s winning arm is my own experiment's winner.**
   Promoting it would be selection on the data that produced the finding. It is
   reported as an instrument and #1 is a *different* strategy that happens to
   embody the same filter — the same discipline the volatility family applied and
   flagged.
5. **`pr_max`, `vote_min` and `trail_k` all won at grid edges or near them**
   (`pr_max` best at the tightest tested; `trail_k=2.0` beat 3.0 almost
   everywhere, and 2.0 is the low end). The `trail_k` edge is the one worth one
   bounded extension downward, since it is the parameter of the component this
   report says matters — but note that a tighter trail is also a shorter hold,
   which feeds concern (6).
6. **The trail-shortens-holds confound is not fully excluded.** See Part 1,
   caveat 2. SELECT was a mean-reverting window; the `return-to-mean` arm is only
   a partial control.
7. **`Plan.exit_signal` is direction-agnostic**, so the factorial's no-trail arm
   had to use a symmetric return-to-mean exit rather than the textbook
   opposite-channel exit (which forced `trend_turtle` long-only). The
   opposite-channel exit is therefore still untested for shorts. That is a
   harness limitation, not a strategy choice, and it is the second analyst to
   report it — worth fixing before the next round.
8. **The `_bos_state` helper drops `require_hl`** and re-enters on every bar the
   level is exceeded rather than only the first (`structure.py` uses
   `_first_of_run`). Because `core.simulate` holds one position at a time this
   changes nothing about which trades occur, but it means `ens_bos_trail` is not
   a bit-identical re-registration of `st_bos` and its numbers should not be
   diffed against the structure report to more than one decimal.
9. **HOLDOUT untouched.** Nothing here is out-of-sample in the PRD's one-shot
   sense.

## Reproduction

```
cd scripts/bruteforce
../../.venv/bin/python runner.py --causal-only --family ensemble
../../.venv/bin/python runner.py --split TRAIN  --family ensemble --workers 3 --out /tmp/ens_train.csv
../../.venv/bin/python runner.py --split SELECT --family ensemble --workers 3 --out /tmp/ens_select.csv
```
