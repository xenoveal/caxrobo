# Family Report — MEAN REVERSION (`mr_*`)

**Date**: 2026-07-27
**Module**: `scripts/bruteforce/strategies/meanrev.py`
**Strategies registered**: 10 · **Parameter combos**: 330 (all ≤ 54 per strategy, ≤ 72 cap respected)
**Causality**: all 10 pass `core.assert_causal` (`runner.py --causal-only --family meanrev`)
**Splits used**: TRAIN (2023-08-01 → 2025-07-01), SELECT (2025-07-01 → 2026-01-01). **HOLDOUT untouched.**
**Production bar being measured against**: Donchian on TRAIN — BTC **+0.26**, ETH **−0.45**, SOL **+0.62**.

---

## 0. Headline verdict

**The family has no edge, and the result is not marginal.**

| Scope | Combos with positive mean Sharpe |
|---|---|
| TRAIN, 20-symbol universe | **0 of 330** |
| TRAIN, core 3 (BTC/ETH/SOL) | **6 of 330** — all six are one strategy at one grid corner |
| Those 6 carried to SELECT | collapse (mean **−0.39**, ETH **−2.39**) |

**Nothing here beats the production Donchian baseline on any split.** No candidate is recommended
for promotion. Recommendations in §5 are therefore about what to *stop* doing.

The one genuinely valuable finding is a negative one that resolves a live PRD question — see §4.

## 1. The cost-ratio problem IS fixed. That was not the binding constraint.

Every strategy places its stop as `max(k · ATR, floor_pct · close)`, the remedy the Phase 6
post-mortem explicitly named ("an ATR floor on the fade stop"). It worked, completely:

| | Phase 6 dropped sleeve | This module |
|---|---|---|
| median `risk_pct` | 0.79% / 1.26% / 1.43% | **4.5%** (core) / **6.0%** (20-sym), min 1.50% |
| median stop in ATR units | 0.63–0.75 × ATR | ≥ 1.5 × ATR by construction |
| measured `c` | 0.1812 / 0.1147 / 0.1001 — **FAIL on all 3** | range **0.0069 – 0.0936** |
| rows breaching `c ≤ 0.10` | 3 of 3 symbols | **0 of 7,590 rows** |

This is the load-bearing result of the whole exercise. The Phase 6 report attributed the fade's
failure to its stop being too tight, and that diagnosis was **testable and is now tested**: the
stop was widened until the cost objection provably disappeared, and the strategies still lose.
Cost was a real defect but it was not the reason there was no money in the trade.

Widening the stop is not free — mean holding time runs 25–124 trigger bars and expectancy per
trade is negative in R terms — but the losses are far too large to be explained by the cost
model. Gross of costs these signals do not predict.

## 2. Every strategy, best-by-Sharpe TRAIN config with that same config on SELECT

Core 3 symbols. `c` and `trades` are the worst value across the three symbols (the gate is
per-symbol). All rows shown satisfy `trades ≥ 30` and `c ≤ 0.10` unless flagged.

| Strategy | Best TRAIN config | TRAIN mean Sharpe | TRAIN per-coin (BTC/ETH/SOL) | SELECT mean | SELECT per-coin | max `c` | med risk | exp (R) | PF | grid frac >0 |
|---|---|---|---|---|---|---|---|---|---|---|
| `mr_rsi2_trend` | rsi_p 2, lo 5, k 1.5, floor 2.4% | **+0.212** | +0.12 / +0.31 / +0.20 | **−0.392** | +0.05 / **−2.39** / +1.16 | 0.058 | 2.85% | +0.032 | 1.07 | 6/36 |
| `mr_pullback_trend` | bb_std 2.0, k 1.5, floor 1.8% | −0.262 | +0.15 / −0.11 / −0.83 | +0.028 | −0.29 / −0.96 / +1.34 | 0.063 | 2.93% | −0.043 | 0.96 | 0/12 |
| `mr_bb_walkback` | bb_p 20, std 2.5, k 1.5, floor 1.4% | −0.319 | +0.61 / +0.50 / **−2.07** | −0.329 | −0.27 / −0.63 / −0.09 | 0.069 | 2.65% | −0.039 | 1.04 | 0/36 |
| `mr_vwap_stretch` | vwap_p 48, stretch 3.0, k 3.0, floor 1.4% | −0.479 | −0.58 / −0.32 / −0.54 | −0.694 | −1.22 / +0.11 / −0.97 | 0.032 | 5.58% | −0.057 | 0.96 | 0/36 |
| `mr_zscore` | z_p 48, z_min 3.0, k 2.0, floor 2.4% | −0.602 | +0.17 / −0.63 / −1.34 | −0.443 | −0.59 / +0.76 / −1.50 | 0.052 | 3.83% | −0.123 | 0.92 | 0/54 |
| `mr_rsi_raw` | rsi_p 2, lo 5, k 2.5, floor 2.4% | −0.652 | −0.82 / −0.58 / −0.56 | **−2.343** | −2.50 / −3.71 / −0.82 | 0.042 | 4.56% | −0.047 | 0.90 | 0/36 |
| `mr_stoch_confirm` | st_p 14, lo 10, k 3.0, floor 1.4% | −0.765 | −0.88 / −0.31 / −1.11 | +0.472 | −0.20 / −0.21 / +1.82 | 0.031 | 6.19% | −0.073 | 0.84 | 0/24 |
| `mr_bb_fade_atr` | bb_p 20, std 2.5, k 2.5, floor 1.4% | −0.807 | −0.09 / −0.13 / **−2.21** | −0.557 | −1.07 / −1.34 / +0.74 | 0.043 | 4.37% | −0.094 | 0.95 | 0/36 |
| `mr_wr_daily` | wr_p 20, lo −90, k 1.5, floor 1.4% | −1.254 | −1.18 / −1.22 / −1.36 | −0.065 | +0.98 / −1.70 / +0.53 | 0.027 | 7.09% | −0.242 | 0.62 | 0/24 |
| `mr_keltner_lowadx` | kc 2.5, adx_max 22, k 3.0, floor 1.4% | −1.373 | −0.07 / −1.89 / −2.16 | −1.487 | −1.63 / −1.10 / −1.74 | 0.034 | 5.31% | −0.225 | 0.64 | 0/36 |

On the **20-symbol universe** the best config of every single strategy is negative on TRAIN
(best in family: `mr_vwap_stretch` at −0.223, `mr_wr_daily` at −0.256). The `frac >0` column is
**0.000 for all ten strategies** at 20 symbols. Breadth does not rescue the family; it removes
the last positive reading.

## 3. Rationales, and what actually failed

- **`mr_rsi2_trend`** (Connors RSI-2 + 1D 200-MA filter) — *counterparty: liquidated leveraged
  longs; reverting force: makers who absorbed the cascade re-hedging to flat.* The only strategy
  with a positive TRAIN reading, and **it is overfit**: 6 of 36 combos positive, all at the single
  corner `rsi_p=2, lo=5` (the most extreme threshold on both axes — a ridge on the grid edge, which
  is the classic signature of fitting), it dies on SELECT (ETH −2.39), and it is negative at 20
  symbols. n=120–151 per coin on TRAIN, so this is not small-sample noise, it is genuine
  in-sample fitting.
- **`mr_rsi_raw`** (same trigger, no trend filter) — registered as the controlled comparison, and
  it earned its keep. TRAIN −0.652 vs +0.212 filtered, SELECT **−2.343**, the worst reading in the
  module on ~250–360 trades per coin. Reading the pair together: whatever `mr_rsi2_trend` had was
  contributed by the **200-MA trend filter**, not by the RSI extreme. That is trend exposure, and
  the trend sleeve already harvests it more cheaply.
- **`mr_bb_fade_atr`** (the dropped sleeve, ATR-floored) — see §4. Fails everywhere.
- **`mr_bb_walkback`** (fade only after a close back INSIDE the band) — *the absorption already
  happened, so we are not fading live flow.* The most interesting failure: **BTC +0.61 and
  ETH +0.50** on TRAIN, but **SOL −2.07**, and both BTC/ETH readings vanish on SELECT (−0.27/−0.63).
  It is consistently better than the unconfirmed fade on BTC/ETH across the whole grid, so the
  confirmation idea is directionally right — it just does not survive out of sample or reach SOL.
- **`mr_zscore`** (|z| vs trailing mean) — self-calibrating band fade. Negative at every threshold;
  raising `z_min` to 3.0 cut trades to 54–64 without turning expectancy positive.
- **`mr_keltner_lowadx`** (fade only when 1D ADX < 18/22) — *the complement to the trend sleeve:
  fade only the tape Donchian refuses.* **The worst strategy in the family** (TRAIN −1.373, SELECT
  −1.487, PF 0.64, 0/36 positive). This is the sharpest answer available to the PRD's "ranging
  regime is unserved" hypothesis: the unserved bucket was tested directly and it is not unserved
  because nobody looked — it is unserved because there is nothing in it. Low ADX does not mark a
  mean-reverting tape, it marks a tape with no exploitable movement at all, and the wide ATR stop
  needed to clear the cost gate is then never repaid.
- **`mr_stoch_confirm`** (stochastic extreme + turn-back bar) — TRAIN −0.765 on a 59.5% win rate.
  A textbook illustration of the brief's warning: the highest win rate in the module and firmly
  negative expectancy, because the mid-band target is much nearer than the ATR-floored stop.
  SELECT is +0.47, driven entirely by SOL +1.82 on 35 trades; with TRAIN at −0.765 this is a
  sign-flip on a small sample, not a discovery.
- **`mr_vwap_stretch`** (distance from rolling VWAP in ATR units) — *VWAP-benchmarked execution
  algos lean against their own slippage.* Best-in-family at 20 symbols (−0.223) and the closest
  thing to a graceful failure, but negative on every one of 36 combos on both scopes.
- **`mr_wr_daily`** (Williams %R on the 1D tier, 240-bar hold) — tested whether reversion edge is a
  matter of *horizon*. It is not: TRAIN −1.254 with PF 0.62, the second-worst in the module,
  despite `c` = 0.027 (the most comfortable cost ratio here, median risk 7.1%). Lengthening the
  horizon raised R per winner exactly as intended and the strategy got *worse*, which is strong
  evidence the missing ingredient is prediction, not payoff geometry.
- **`mr_pullback_trend`** — **specified after seeing TRAIN, and disclosed as such.** Since the only
  two signals with any positive reading (`mr_rsi2_trend`, `mr_bb_walkback`) shared exactly two
  features — a long-side trend filter and a demonstrated-absorption entry — I registered their
  minimal conjunction to test whether that shared structure was the driver. **It is not: 0 of 12
  combos positive on TRAIN on either scope.** The hypothesis is refuted, which further undercuts
  the `mr_rsi2_trend` reading. Recorded because a post-hoc idea that *fails* is still a consumed
  degree of freedom.

## 4. Does the dropped fade sleeve deserve reinstatement? **No.**

This was the live PRD question, so it gets an explicit answer with the decision-relevant numbers.

`mr_bb_fade_atr` is the dropped sleeve with one change — the ATR-floored stop that Phase 6
identified as the fix. On TRAIN, core 3:

| Symbol | median Sharpe (36 combos) | best Sharpe | median exp (R) | median PF | max `c` | med risk |
|---|---|---|---|---|---|---|
| BTCUSDT | −0.654 | −0.085 | −0.069 | 0.91 | 0.072 | 3.36% |
| ETHUSDT | −1.416 | −0.131 | −0.125 | 0.81 | 0.054 | 4.57% |
| SOLUSDT | −1.858 | −1.551 | −0.217 | 0.70 | 0.037 | 6.53% |

- **0 of 36 combos have positive pooled expectancy.** 0 of 36 have positive mean Sharpe.
- **No combo is positive on any symbol** — the best single reading anywhere is BTC at −0.085.
- **`c` now passes comfortably on all three symbols** (0.037–0.072 vs the 0.10 ceiling), where the
  original sleeve failed at 0.1812/0.1147/0.1001.

That last line is the point. **The cost objection has been removed and the sleeve still loses on
all three symbols, with profit factors (0.91/0.81/0.70) essentially unchanged from the 0.91/0.58/0.63
that got it dropped.** The Phase 6 report's diagnosis — "the stop was too tight, costs ate it" —
was therefore *incomplete*: fixing the stop fixes the cost ratio and does not fix the P&L, so the
sleeve's problem was never primarily the risk model. There is no gross edge in fading a 2σ band
excursion on 4H crypto perps.

**Recommendation: keep `config.FADE_ENABLED = False`. Close the reinstatement question.** The
Phase 6 verdict stands, and it now stands on a stronger footing than the evidence that produced it
— the remedy its own post-mortem proposed has been implemented and measured, and it does not work.
A future re-test should not be run without a *new* mechanism, not a new stop.

## 5. Top-3 and recommendations

There is no candidate worth promoting. Ranked by "least bad, most informative":

1. **`mr_vwap_stretch`** — best at 20 symbols (TRAIN −0.223) and the only strategy whose losses
   are uniform rather than symbol-driven. *Per coin:* BTC −0.58, ETH −0.32, SOL −0.54 on TRAIN;
   BTC −1.22, ETH +0.11, SOL −0.97 on SELECT. **Do not promote.** Its interest is that a
   volume-anchored reference degrades more gently than time-anchored ones; if anyone revisits
   reversion, VWAP is the anchor to start from and funding-rate data is the missing input.
2. **`mr_bb_walkback`** — the only idea with a *reproducible cross-sectional pattern*: it beats the
   unconfirmed fade on BTC and ETH across the entire grid (TRAIN BTC +0.61, ETH +0.50) and fails
   catastrophically on SOL (−2.07). *Per coin on SELECT:* BTC −0.27, ETH −0.63, SOL −0.09 — the
   BTC/ETH edge does **not** replicate. **Do not promote.** The transferable lesson is that
   requiring absorption before entry is strictly better than fading live flow; any future fade
   must be confirmed, and must be tested per-coin because SOL behaves like a different asset class.
3. **`mr_rsi2_trend`** — listed only because it is the family's sole positive TRAIN reading
   (+0.212; BTC +0.12, ETH +0.31, SOL +0.20, all three positive, n ≥ 120). **Explicitly flagged as
   overfit, not as a candidate**: grid-edge ridge, 6/36 combos, SELECT −0.392 with ETH at −2.39,
   negative at 20 symbols, and its own control (`mr_rsi_raw`) plus its own follow-up
   (`mr_pullback_trend`) both refute the mechanism. Its TRAIN numbers superficially rival
   production Donchian (+0.26/−0.45/+0.62) and that resemblance is exactly the trap; DSR is
   8.3e−04, i.e. not remotely significant against 330 combos × 20 symbols of search.

### Per-coin notes
- **BTC** is the most reversion-friendly of the three: it produces the least-negative reading for
  7 of 10 strategies and the only two positive TRAIN Sharpes above +0.5 (`mr_bb_walkback` +0.61).
- **ETH** is the SELECT-split destroyer: −2.39 (`mr_rsi2_trend`), −3.71 (`mr_rsi_raw`), −3.26 at 20
  symbols. Any reversion signal that looks good on ETH in TRAIN should be assumed fitted.
- **SOL** kills band fades (−2.07 walkback, −2.21 fade, −2.16 keltner). It trends through bands
  rather than reverting off them, consistent with it having the widest median bar range in the PRD's
  noise study. Its occasional large positive SELECT readings (+1.82, +1.34, +1.16) sit on 29–39
  trades against firmly negative TRAIN values and are sign-flips, not signal.

### Robustness concerns (stated plainly)
- **Sign instability dominates.** Six strategies flip sign between TRAIN and SELECT. With
  0/330 positive on the 20-symbol TRAIN, the honest reading is that per-symbol Sharpe here is noise
  around a negative mean, and any positive cell is a draw from that distribution.
- **DSR is negligible everywhere** (max 0.086, most ≤ 1e−3) against 330 combos × 20 symbols.
- **One post-hoc degree of freedom was consumed** (`mr_pullback_trend`, §3) and it failed.
- The `floor_pct` axis was swept only over {1.4%, 1.8%, 2.4%} — values at or above the `c = 0.10`
  boundary derived from the frozen cost model, not fitted on returns. Results are near-insensitive
  to it, which corroborates §1: cost is no longer the binding constraint.

## 6. Harness bug found (not worked around silently)

`runner.py` cannot run a sweep on this machine: `_run_one_symbol` reads `registry.ALL` inside
`ProcessPoolExecutor` workers, but macOS uses the **spawn** start method, so workers re-import
`registry` with an **empty** `ALL` and every job dies with `KeyError: '<strategy name>'`. It
reproduces on the shipped baseline (`--only donchian_production`), so it is **pre-existing and
unrelated to this family**; `--causal-only` is unaffected because the audit runs in-process.

Per the brief I did not edit the harness. All numbers above were produced by a scratchpad driver
that calls `runner._run_one_symbol` in-process, reusing the harness's own scoring path verbatim —
no cost, simulation, or metric logic is reimplemented. **Suggested fix (for the harness owner):**
call `registry.load_all()` in a `ProcessPoolExecutor(initializer=...)`.

## 7. Reproduce

```
cd scripts/bruteforce
../../.venv/bin/python runner.py --causal-only --family meanrev        # 10 passed, 0 disqualified
```
The sweeps require the spawn fix in §6 (or the in-process driver); once fixed:
```
../../.venv/bin/python runner.py --split TRAIN  --family meanrev --core-only --workers 3
../../.venv/bin/python runner.py --split SELECT --family meanrev --core-only --workers 3
../../.venv/bin/python runner.py --split TRAIN  --family meanrev --workers 3   # 20 symbols
../../.venv/bin/python runner.py --split SELECT --family meanrev --workers 3
```
**HOLDOUT was never run.**
