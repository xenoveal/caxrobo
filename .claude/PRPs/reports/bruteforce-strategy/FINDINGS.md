# Brute-Force Strategy Search — Findings

**Date:** 2026-07-27 · **Commit:** `master` @ `33a8c37` · **Holdout: SPENT**

Provenance verified: `33a8c37` is the trunk tip (this repo's trunk is `master`;
there is no `main`) and fully contains `feat/phase4-tier-shift` via merge
`8778003`, along with the phase1/phase2/phase3 branches. `git diff HEAD --
src/trading_bot/` was empty at measurement time, so every production component
the harness imports -- the frozen cost constants in `config.py`, Wilder ATR/ADX,
Donchian, and all of `backtest/equity.py` -- was read at its committed state, not
from an in-flight edit.

77 strategies across 8 families, 1,262 parameter combinations, 20 symbols,
~25,000 scored evaluations. Eight analysts working in parallel through one
shared, unit-tested harness.

---

## 1. Headline: the gate is not passed, and the search says why

**No strategy passes the PRD gate.** More importantly, the holdout did not merely
fail to confirm the shortlist — it **inverted** it.

Pre-committed shortlist, one frozen config each, evaluated once on
2026-01-01 → 2026-07-24. Mean per-symbol Sharpe across 20 symbols:

| Strategy | TRAIN | SELECT | **HOLDOUT** | % symbols +ve (HOLDOUT) | max `c` |
|---|---|---|---|---|---|
| `ens_lowvol_keltner_trail` | +0.549 | +0.666 | **−0.446** | 35% | 0.149 |
| `ens_vote3_majority` | +0.553 | +0.499 | **−0.769** | 35% | 0.144 |
| `ens_regime_dispatch` | +0.494 | +0.556 | **−0.309** | 45% | 0.145 |
| `trend_keltner_trail` | +0.402 | +0.576 | **−0.552** | 25% | 0.147 |
| `fac_entry_exit` (Donchian+trail) | +0.148 | +0.673 | **−0.418** | 50% | 0.154 |
| `donchian_production` (incumbent) | −0.184 | −1.131 | **+0.135** | 55% | 0.148 |

The incumbent — worst on both prior splits — is the only positive one on the
holdout. Five candidates that agreed across two independent 2-year and 6-month
spans all failed, in the same direction, at the same time.

**Read this as a rank inversion, not five separate disappointments.** If the
candidates had degraded toward zero, the honest conclusion would be "real but
weaker edge." Simultaneous sign flips with the control moving the other way is
what fitting noise looks like. Deflated Sharpe predicted it: every candidate
scored DSR ≈ 0.0001 or lower against 25,240 charged trials, versus the 0.95
threshold. **DSR was the only metric in the whole exercise that was right in
advance.**

Secondary observation, and a caution against over-reading the table: `cost_ratio`
rose to 0.144–0.154 for **every** strategy on the holdout, including the control.
2026 was a lower-volatility tape, so ATR-derived stops tightened and every
candidate breached the 0.10 ceiling for reasons that have nothing to do with
strategy choice. Part of the holdout's severity is regime, not selection — which
is itself the point of §1's lesson.

### The methodological lesson

TRAIN/SELECT agreement was treated — by me, in the session design — as the
primary defence against overfitting. It is not sufficient. Two adjacent spans of
the same asset class share regime; a strategy fitted to 2023–2025 crypto can
agree with itself on H2-2025 and still be describing a regime rather than an
edge. The three-way split caught nothing that DSR had not already flagged, and it
cost the holdout to find that out.

What would have been better: **purged combinatorial cross-validation** over
non-adjacent blocks, so "out-of-sample" means a different regime rather than a
later date. That is the recommended shape for any future search here.

---

## 2. What survived as knowledge

The strategy search failed. The *measurements* did not, and several are durable
because they are paired differences on identical bars rather than level
comparisons.

### 2.1 The edge is in the exit, not the entry (strongest result)

A factorial (`fac_entry_exit`, 24 combos): entry {Donchian N-bar break, Keltner
ATR channel} × exit {ATR ratchet trail, fixed ATR target, return-to-mean} with
stop frozen at 1.5×ATR(4H).

| Factor | TRAIN | SELECT |
|---|---|---|
| **Exit = trail** | **+0.262** (67.5% +ve) | **+0.316** (76.2% +ve) |
| Exit = fixed target | +0.149 | **−0.415** |
| Exit = return-to-mean | −0.004 | −0.228 |
| Entry = Donchian | +0.101 | −0.080 |
| Entry = Keltner | +0.171 | −0.138 |

The exit factor moves Sharpe by +0.19 (TRAIN) and +0.64 (SELECT). The entry
factor moves it by +0.07 then −0.06 — **sign-unstable, and % symbols positive is
identical to within 0.5pp on both splits.** No interaction: the trail helps
Donchian *more* than Keltner. Directionally consistent on 35 of 40
symbol-splits.

This holds up better than anything else here because it is a paired difference
measured on the same bars, and it explains a prior puzzle: Phase 4's
exit-management repair moved the numbers more than any signal change across
three phases. It also means `trend_keltner_trail` was never an entry discovery —
it was the trend family's only trailed channel strategy.

Caveat, disclosed: the trail shortens holds, and SELECT was a mean-reverting
tape. The return-to-mean arm only partly controls for that confound.

### 2.2 Cost was never the whole problem

The PRD's central diagnosis — noise-floor stops, `c` = 0.58–0.68 — was correct
about the mechanism and **incomplete about the consequence.** The ATR floor
fixed the cost ratio completely and did not make the strategies profitable:

| Family | Cost ratio, before → after | Outcome |
|---|---|---|
| chartpattern | 0.58–0.68 → 0.013–0.081 | still no edge (TRAIN→SELECT rho **+0.047**) |
| meanrev | 0.10–0.18 → 0.007–0.094 (**0 of 7,590 rows breach**) | **0 of 330 combos** positive at 20 symbols |
| structure | → 0.014–0.107 | one strategy positive on both splits |

**The missing ingredient is prediction, not payoff geometry.** Corroborated
independently: `mr_wr_daily` lengthened the horizon specifically to raise R per
winner, and got *worse* (−1.25, PF 0.62).

### 2.3 The fade sleeve question is closed: NO

`mr_bb_fade_atr` is the dropped Bollinger sleeve with exactly the ATR-floored stop
its own post-mortem prescribed. Cost ratio now passes comfortably (0.037–0.072
vs the failing 0.1812/0.1147/0.1001). Profit factors: **0.91 / 0.81 / 0.70 —
essentially unchanged from the 0.91 / 0.58 / 0.63 that got it dropped.**

Fixing the stop fixed the cost ratio and not the P&L, so the Phase 6 causal story
was wrong. Keep `FADE_ENABLED = False`. Reached twice independently: the structure
analyst found boundary mean-reversion fails on **payoff shape** (win rate
0.47–0.59 but PF 0.69–0.76 — wins small, loses the whole range width), from a
completely different construction. `mr_keltner_lowadx` tested the PRD's "unserved
ranging regime" head-on and is the worst strategy in that module.

### 2.4 Chart-pattern retirement stands

Across 1,763 judgeable cells: **TRAIN→SELECT rank correlation +0.047, sign
agreement 51.5%.** A coin flip. Removing the cost drag moved these from losing to
random, not to profitable.

The structural trap: **Sharpe is inversely monotonic with sample size.**
`cp_sr_retest` at 115 trades/symbol pools to −0.29; `cp_range_break` at 4 trades
to +0.38. Every judgeable member is ≈0 or negative; every positive member is
unjudgeable. Head-and-shoulders reproduced the PRD's history exactly — median 6
SELECT trades/symbol, 100% of cells under 30, even loosened across 20 symbols.
The wider universe bought **breadth but not depth**.

One fragment worth keeping: `cp_range_break` is Donchian plus a range-width
filter and looked better than plain Donchian where it traded at all. Hand the
filter to the trend engine; do not keep it as a standalone strategy.

### 2.5 Option B (universe expansion) is not supported

The PRD's designated escalation path does not survive testing:

- The two purely cross-sectional strategies ranked **9th and 10th of 10**.
  Plain time-series momentum beat relative strength ~7× on identical universe,
  stops and costs.
- Widening core-3 → 20 symbols **lowered** per-symbol Sharpe for 9 of 10
  momentum strategies (`mom_accel` 1.162 → 0.712).

What the backfill did buy: **20 independent replications instead of 3.** That is
Option A's mechanism on more data, and it is the reason this report can state
negative results with confidence.

### 2.6 Volatility: keep suppression, and prefer low volatility

Four independent instruments agree on both splits and both universes. One
identical rule across three volatility terciles:

| Tercile | TRAIN | SELECT |
|---|---|---|
| Low | **+0.340** | **+0.567** |
| Mid | +0.119 | −1.075 |
| High | **−0.284** | **−1.665** |

The production 90th-percentile veto is directionally right but crude — the
gradient runs across the *whole* distribution, so a **low-volatility preference**
dominates a binary veto (`lowpref` +0.559/+0.989 > `veto` +0.438/+0.621 >
`none` +0.353/+0.406, monotone on both splits). Cost: it raises `c` to 0.10–0.13
by shrinking median risk, trading Sharpe against the cost gate.

The converse hypothesis is dead: `vol_term_structure` had the best 20-symbol
TRAIN in its family (+0.568, 90% positive) then **−1.284 at 20% positive**.

### 2.7 Candlestick confirmation adds nothing

Matched pair, identical Donchian breakout ± a body/engulfing requirement on the
breakout bar. Core-3: TRAIN Δ +0.548 (t=+7.1), SELECT Δ +0.735 (t=+4.4), positive
in **16 of 16** cells. Universe-20: TRAIN Δ +0.058 (t=+1.41), per-symbol Δ from
+1.48 (SOL) to −1.53 (FIL).

A ~10× effect confined to the three most-searched symbols in this repo's history
is symbol selection, not edge. **Core-3 Sharpe runs roughly 2× the universe-20
figure family-wide** — found independently by three analysts, and the reason all
headline numbers in this report are universe-wide.

---

## 3. Two measurement defects found in production code

### 3.1 Pooled Sharpe is not measurable (affects the PRD's primary gate)

`src/trading_bot/backtest/equity.py:daily_returns` books each trade's entire PnL
on its **exit day**, with no daily mark-to-market of open positions. At multi-day
holds this makes concurrent positions look independent. Measured directly:

| | mean per-symbol Sharpe | pooled (exit-day) | mean pairwise rho |
|---|---|---|---|
| TRAIN core-3 | +1.06 | **+1.84** | −0.001 |
| TRAIN universe-20 | +0.75 | **+2.35** | +0.041 |

Spreading each trade's PnL across its holding days puts pairwise rho at
**+0.27 to +0.33**. So pooling under exit-day booking harvests fake √N
diversification, which is why pooled (+2.35) sits far above mean per-symbol
(+0.75).

Spreading is *not* the fix — it smooths daily variance and yields an absurd +6
to +8. Both schemes are wrong in opposite directions. **Pooled Sharpe is not
measurable without daily mark-to-market.**

This matters because the PRD's primary gate is *"pooled 3-symbol ... time-indexed
equity curve"*, computed by this function. **That gate does not currently measure
what it intends to.** All figures in this report are therefore mean/median
per-symbol Sharpe plus % symbols positive. Fixing it requires revaluing open
positions each day from bar closes.

### 3.2 `Plan.exit_signal` is direction-agnostic

Reported independently by two analysts. The short leg of any asymmetric-exit
system is untestable, which forced a symmetric return-to-mean exit in the
factorial instead of the textbook opposite-channel exit. Harness limitation, not
a bug; recorded because it bounds §2.1.

---

## 4. Recommendations

### 4.1 Do not promote anything to live

Nothing cleared the gate; five candidates failed a one-shot holdout in the same
direction. The correct action is no action.

### 4.2 The one cheap production change worth making

Add a **ratcheting ATR trail** to the existing Donchian engine (§2.1). It is the
best-evidenced result here, it is a small diff against `TRAIL_ENABLED` /
`TRAIL_ATR_MULTIPLE` which already exist, and it is worth doing *even though the
holdout failed* — because the finding is a paired within-sample difference, not a
level claim about future returns.

Two caveats. It **lowers win rate** (0.32 vs 0.42) while raising profit factor —
if the goal is stated as win rate, this moves the wrong metric for the right
reason. And Phase 4 measured the trail as *harmful* when its multiple was pinned
to the entry stop's `k`; the constants must stay independent
(`TRAIL_ATR_MULTIPLE` ≈ 2.0–3.0 against a 1.5 entry stop).

### 4.3 Fix the metric before the next search

§3.1 is a prerequisite. Any future portfolio or pooled claim is currently
unfalsifiable, and the PRD's headline gate is among them.

### 4.4 If searching again, change the validation, not the strategies

Purged combinatorial cross-validation over **non-adjacent** blocks, DSR as a
ranking input rather than a footnote, and a pre-registered trial budget. This
session's failure was not a shortage of ideas — 77 strategies is not too few. It
was that adjacent-span validation cannot distinguish regime from edge.

### 4.5 Per-coin strategy selection: not supported

The session was asked whether the best strategy differs per coin. On this
evidence the honest answer is that **per-coin selection is not measurable here**:
per-symbol SELECT Sharpes carry ±0.6–0.9 standard errors on 15–45 trades, and the
coin that "wins" flips between splits (`ens_regime_dispatch`: SOL +0.58 → −0.06;
`st_bos`: which coin works flips with a single parameter). Choosing a different
strategy per coin on these samples would be fitting noise per coin.

One durable per-symbol result: **BCHUSDT is negative on both splits under every
strategy in every family** and should be dropped from the research universe.

### 4.6 Honest ceiling

The best gated mean per-symbol Sharpe found anywhere, before the holdout, was
**+0.589**. After the holdout, no candidate is distinguishable from zero. Against
the PRD's target of ≥1.0 out-of-sample with all three symbols positive, the gap
is not a tuning gap. A 30–45% annualized return goal is not supported by anything
measured in this session.

---

## 5. Provenance and honesty log

- **Holdout spent once**, on 6 pre-committed configs (5 candidates + control)
  named in writing before evaluation. No grid was swept on the holdout.
- **Causality:** all 77 strategies pass `core.assert_causal`; the deliberate
  `lookahead_canary` fails on every run. My first version of that check had a
  56-bar slack window that let the canary through — documented in the docstring
  so it cannot be reintroduced as a "fix".
- **Harness validated:** 28 unit tests. Cost model independently corroborated
  (BTC `c` = 0.0698 vs the PRD's separately measured 0.0711; ETH matches to four
  decimals). `donchian_production` reproduces the published Phase 4 baseline
  exactly (BTC +0.26 / ETH −0.45 / SOL +0.62) — verified by four analysts.
- **Costs frozen throughout**; never a grid axis. `k` frozen at 1.5×ATR in the
  trend family so `c` stayed a market property.
- **Bug found by analysts, fixed by me:** `runner.py` workers never loaded the
  registry under macOS `spawn`; every scoring job died with `KeyError`. Three
  analysts diagnosed it independently and none worked around it by editing the
  harness.
- **Degrees of freedom disclosed by analysts, not hidden:** one post-hoc strategy
  specified after seeing TRAIN (`mr_pullback_trend`, failed 0/12, further
  undercutting its own family's top pick); one grid widened for sample adequacy;
  one strategy corrected from producing zero trades. None in response to a Sharpe
  reading.
- **Survivorship bias, unresolved:** `universe.py` selected 20 symbols known to
  have survived to 2026. Cross-sectional results are optimistic relative to a
  real-time-formed universe.
- `src/trading_bot/**` was **not modified**. The harness imports from it so the
  two cannot drift.

## 6. Where things are

| Path | Contents |
|---|---|
| `README.md` | method, gate definition, anti-overfitting controls |
| `family-{trend,momentum,meanrev,volatility,structure,candlestick,chartpattern,ensemble}.md` | per-family detail, every strategy, every rationale |
| `results/HOLDOUT_shortlist.csv` | the one-shot holdout evaluation |
| `scripts/bruteforce/` | harness, indicators, registry, runner, 28 tests, 8 strategy modules |
