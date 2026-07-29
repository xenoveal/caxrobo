# Family report: CHART PATTERN (`cp_*`) — a re-test of retired geometry

**Date:** 2026-07-27
**Module:** `scripts/bruteforce/strategies/chartpattern.py` (9 registered strategies, 180 combos)
**Splits used:** TRAIN (2023-08-01 → 2025-07-01), SELECT (2025-07-01 → 2026-01-01). HOLDOUT untouched.
**Universes:** full 20-symbol `UNIVERSE` and the 3-symbol `CORE` gate.
**Causality:** all 9 pass `core.assert_causal` (`--causal-only --family chartpattern`: 9 passed, 0 disqualified).

---

## Headline answer

**The PRD's retirement verdict stands.** Chart-pattern geometry does not survive re-testing under
ATR-derived stops. The ATR risk model fixed exactly the thing it was supposed to fix — `cost_ratio`
collapsed from the old 0.58–0.68 to **0.013–0.081**, comfortably inside the 0.10 ceiling, with median
risk 3–10% of price — and the patterns *still* have no edge. The old cost objection is now removed,
and with it removed the geometry is revealed as directionless rather than merely over-taxed.

The single most damning number is not any Sharpe. It is the TRAIN→SELECT relationship across all
1,763 (strategy, combo, symbol) cells with a judgeable TRAIN sample:

| Robustness statistic | Value | Interpretation |
|---|---|---|
| Spearman rho, TRAIN Sharpe vs SELECT Sharpe | **+0.047** | no rank persistence |
| Sign agreement TRAIN vs SELECT | **51.5%** | a coin flip |
| Per-strategy rho range | −0.068 … +0.143 | none distinguishable from zero |

A family whose configuration ranking does not survive a 6-month walk forward has nothing to select
from. Any "winner" below is a draw from that coin flip.

The second damning structure is the **inverse relationship between sample size and measured Sharpe**,
which is the PRD's original "98% of trade volume" complaint reappearing in a new form:

| Strategy | median trades/symbol (TRAIN, all combos) | pooled TRAIN Sharpe (20 sym, all combos) |
|---|---|---|
| `cp_sr_retest` | **115** | **−0.287** |
| `cp_double_neckline` | 68 | +0.033 |
| `cp_squeeze_expand` | 44 | −0.222 |
| `cp_double_ranging` | 36 | +0.123 |
| `cp_flag_trend` | 30 | −0.190 |
| `cp_hs_neckline` | 22 | −0.192 |
| `cp_range_break_vol` | 8 | **+0.327** |
| `cp_failed_break` | 7 | −0.401 |
| `cp_range_break` | **4** | **+0.376** |

Every strategy that generates a judgeable sample scores ≈0 or negative. Every strategy that scores
positively generates 4–8 trades per symbol — i.e. the family's apparent winners are precisely the
members that cannot be judged. That is the signature of noise, not of an edge concentrated in rare
setups.

**Reference bar** (`donchian_production`, reproduced in this harness): TRAIN core BTC **+0.26** /
ETH **−0.45** / SOL **+0.62**, matching the stated Phase-4 figures exactly — the harness is calibrated.
For context on how hostile SELECT is, the same production rule scores BTC −1.61 / ETH +0.39 / SOL −2.12
on SELECT. SELECT is a regime in which trend-following broke; positive SELECT numbers below should be
read with that in mind, not as validation.

---

## Every strategy: best-by-Sharpe TRAIN config (CORE, gated at n ≥ 30/symbol) and its SELECT numbers

Selection rule: highest mean TRAIN Sharpe across BTC/ETH/SOL among combos with ≥30 trades on **every**
core symbol and `cost_ratio ≤ 0.10`. Where no combo clears the sample gate the strategy is reported as
**insufficient sample** and its unconstrained best is shown only for completeness.

| Strategy | Best TRAIN config | TRAIN Sharpe BTC / ETH / SOL (mean) | TRAIN n BTC/ETH/SOL | max cost_ratio | median risk | SELECT Sharpe BTC / ETH / SOL (mean) | SELECT n | Verdict |
|---|---|---|---|---|---|---|---|---|
| `cp_double_ranging` | `adx_max=30, k=1.5, span=5, tol=0.03` | +0.60 / +1.06 / +0.79 (**+0.816**) | 53/36/46 | 0.026 | 6.4% | +1.29 / +1.54 / +2.20 (+1.678) | 17/7/9 | best TRAIN of the family, but SELECT n = 7–17 (< 30) and only 6/24 combos clear the sample gate |
| `cp_double_neckline` | `expiry=24, k=1.5, span=3, tol=0.03` | +0.51 / +0.30 / +0.49 (**+0.435**) | 128/134/108 | 0.033 | 5.3% | −0.25 / +1.75 / +0.43 (+0.644) | 38/32/27 | only member with a judgeable sample on both splits; TRAIN edge is thin and BTC flips negative |
| `cp_hs_neckline` | `expiry=24, k=1.5, prominence=0.01, span=3` | −0.67 / +0.12 / +1.26 (+0.236) | 30/41/46 | 0.020 | 8.8% | −1.92 / −0.85 / +0.29 (−0.828) | 6/6/9 | fails; SELECT negative on 2/3, n = 6–9 |
| `cp_squeeze_expand` | `contraction=0.7, k=1.5, period=55` | −0.26 / −0.54 / +1.38 (+0.192) | 57/55/53 | 0.027 | 7.0% | +0.65 / +2.42 / +0.21 (+1.094) | 16/15/15 | TRAIN positive on 1/3 symbols; sign flips wholesale into SELECT |
| `cp_flag_trend` | `consol=8, k=2.0, pole=12, pole_min=0.03` | +0.80 / −1.12 / +0.76 (+0.144) | 30/42/52 | 0.026 | 6.0% | +1.16 / +1.39 / +0.46 (+1.003) | 7/13/16 | ETH −1.12 on TRAIN; 2/16 combos clear the sample gate; pooled 20-symbol TRAIN Sharpe −0.19 |
| `cp_sr_retest` | `k=2.0, near_atr=0.3, span=3, trend=50` | +0.06 / −0.58 / +0.33 (**−0.064**) | 130/136/137 | 0.052 | 3.7% | +0.19 / +1.10 / +1.32 (+0.871) | 37/38/37 | the one strategy with n ≥ 30 everywhere on **both** splits, and its TRAIN Sharpe is negative. The cleanest measurement in the family, and it says no edge |
| `cp_range_break` | *insufficient sample* — 0/18 combos reach n ≥ 30 on all core | (best unconstrained `k=1.5, max_width=0.09, period=30`: +0.35 / +0.65 / +1.87) | 87/69/**28** | 0.021 | 7.4% | +0.46 / +1.15 / +1.61 | 26/10/8 | most promising numbers in the family and the only one with rho > 0.1 and 81% sign agreement — but SOL cannot reach 30 trades and SELECT n = 8–26 |
| `cp_range_break_vol` | *insufficient sample* — 0/24 combos reach n ≥ 30 on all core | (best `k=1.5, max_width=0.06, period=55, vol_min=1.3`: +1.23 / +0.60 / n/a) | 14/4/**0** | 0.024 | 5.8% | +1.88 / n/a / n/a | 8/0/0 | unjudgeable. PF 3.4–7.7 on 4–14 trades is a small-sample artefact, not a volume-filter edge |
| `cp_failed_break` | *insufficient sample* — 0/24 combos reach n ≥ 30 on all core | (best `k=2.0, max_width=0.09, period=40, within=6`: +0.48 / +0.86 / +0.80) | 63/37/**6** | 0.055 | 3.0% | −0.63 / −1.41 / −1.09 | 25/1/5 | the clearest single failure: positive on all three TRAIN symbols, **negative on all three SELECT symbols**, pooled 20-symbol TRAIN Sharpe −0.40 |

### Insufficient-sample summary

Per-symbol `trades ≥ 30`, the PRD's own adequacy bar:

- **Never judgeable on TRAIN core, at any config:** `cp_range_break` (0/18 combos), `cp_range_break_vol`
  (0/24), `cp_failed_break` (0/24). The 20-symbol universe does **not** rescue them — it raises the
  *pooled* count (e.g. `cp_range_break` 483 trades across 20 symbols) but the gate is per-symbol, and
  per-symbol medians stay at 4–8. Widening the universe bought breadth, not depth.
- **Marginal:** `cp_hs_neckline` (2/16 combos), `cp_flag_trend` (2/16), `cp_double_ranging` (6/24 on
  the 20-symbol view, 10/24 on core).
- **Judgeable on TRAIN:** `cp_sr_retest` (24/24), `cp_double_neckline` (14/16), `cp_squeeze_expand`
  (12/18) — and all three are ≈0 or negative.
- **On SELECT nothing is judgeable except `cp_sr_retest`** (51% of its cells reach n ≥ 30). For every
  other strategy, ≥ 82% of SELECT cells have n < 30 and the median is 1–19 trades. **H&S in particular
  reproduces the PRD's finding exactly**: median 6 SELECT trades per symbol, 100% of cells below 30,
  even at the coarsest defensible definition (span 3, prominence 1%) on the widest universe. The PRD's
  "likely never will" was correct.

---

## What the ATR stop actually changed

This is the part of the re-test worth keeping, because it isolates the variable:

| | Old model (PRD evidence) | This re-test |
|---|---|---|
| Stop | fixed 0.5% of price | `max(k·ATR(4H), structural distance)`, k ∈ {1.5, 2.0} |
| Median risk per trade | ~0.5% | **3.0% – 10.3%** |
| `cost_ratio` | 0.58 – 0.68 | **0.013 – 0.081** (every strategy, every reported config) |
| Verdict | lose on all three symbols | ≈0, and non-persistent |

So the diagnosis in the PRD ("costs consumed 58–68% of the risk unit") was right about the *mechanism*
and the fix works: the cost frontier is no longer the binding constraint anywhere in this family. It
simply was not the only thing wrong. Removing the cost drag moved these strategies from *losing* to
*random*, not to *profitable*.

---

## Top-3 recommendations, with per-coin notes

These are ranked as "least indefensible", not as candidates. **My recommendation is that none of them
proceed to the shortlist and none consume holdout.**

**1. `cp_range_break` (`k=1.5, max_width=0.09, period=30–40`) — the only one worth a follow-up, and not
as a chart pattern.**
TRAIN +0.35 / +0.65 / +1.87 (BTC/ETH/SOL), SELECT +0.46 / +1.15 / +1.61; 16/20 and 17/20 symbols
positive; the family's only non-trivial persistence (rho +0.14, sign agreement **81%**). But it is
Donchian with a consolidation-width filter — its evidence base is the trend family's, not the pattern
family's — and it is **blocked on sample size**: SOL reaches only 14–28 trades on TRAIN and 4–8 on
SELECT. *Per coin:* SOL shows the highest Sharpe on the fewest trades, which is the least trustworthy
combination in the table; BTC has the sample (64–87) and the weakest Sharpe (+0.35–0.39); ETH sits
between. **Actionable next step is to hand the width filter to the trend family as a modifier on
production Donchian** and measure whether it improves an already-sampled strategy, rather than to
promote a pattern strategy that cannot be judged.

**2. `cp_double_neckline` (`expiry=24, k=1.5, span=3, tol=0.03`) — the honest null result.**
The only pattern with a genuine sample on both splits (TRAIN 108–134, SELECT 27–38 per core symbol).
TRAIN +0.435 mean, which nominally beats the production bar's +0.14 core mean — but on the 20-symbol
universe the same construction pools to +0.03, only 14/20 symbols are positive, PF is 1.05–1.08, and
BTC flips to −0.25 on SELECT. *Per coin:* ETH is the whole apparent edge on SELECT (+1.75 on 32 trades)
and was the weakest on TRAIN (+0.30) — the ordering inverts, which is the non-persistence in miniature.
Value here is evidential: this is the measurement that shows double tops/bottoms are flat, not the
measurement that shows they work.

**3. `cp_double_ranging` (`adx_max=30, k=1.5, span=5, tol=0.03`) — highest TRAIN Sharpe, thinnest
justification.**
+0.816 TRAIN core mean, positive on all three symbols, +1.678 on SELECT. The regime gate (only take
reversal geometry when 1D ADX is low) is theoretically the right fix and it *is* the biggest single
improvement in the family (+0.816 vs +0.435 ungated). But it buys that by cutting the sample roughly
in half: 6/24 combos clear the 20-symbol gate, SELECT n = 7–17, and the pooled 20-symbol TRAIN Sharpe
is only +0.123. *Per coin:* SOL is +0.79 TRAIN / +2.20 SELECT on 46 and 9 trades — a two-standard-error
band wide enough to contain zero comfortably; ETH +1.06 on 36 trades is the most solid cell in the
family and still short of a 60-day-fold adequacy claim. Retest only if the regime-gate idea is being
evaluated on its own merits, in which case it should be applied to a sample-rich base strategy instead.

---

## Robustness concerns (read before quoting any number above)

1. **No rank persistence.** rho +0.047 and 51.5% sign agreement across 1,763 judgeable cells. Config
   selection inside this family is indistinguishable from picking at random. Every TRAIN "best" above
   should be treated as one draw.
2. **SELECT is six months and regime-idiosyncratic.** Production Donchian scores −1.61/−2.12 on BTC/SOL
   there. Several `cp_*` strategies look *better* on SELECT than TRAIN (`cp_squeeze_expand` −0.54 → +2.42
   on ETH); that is a regime effect, not out-of-sample confirmation.
3. **Sample inversion.** The positive strategies are the unjudgeable ones and vice versa, monotonically.
   With 180 combos × 20 symbols the family had 3,600 chances to produce a high Sharpe on 4 trades, and
   it took them.
4. **Wide structural stops shift the exit mix.** `max(k·ATR, structural)` produces median risk up to
   10.3% of price, so with the 96-bar time stop many trades exit on time rather than at the stop. This
   is what makes `cost_ratio` so comfortable, but it also means these are 3–4-day swing trades whose
   outcome is largely decided by the time stop, not by the pattern's target — worth knowing before
   reading `profit_factor`.
5. **Tolerance sensitivity is real but not exploitable.** Geometry tolerances were centred on
   conventional values (span 3/5/8, tol 1.5%/3%, prominence 1%/2%) and kept to 2–3 values per axis;
   grids are 16–24 combos each, well under the 72 cap. Within those grids `gate_pass_combos` varies
   from 0/24 to 24/24 — the *sample* is highly sensitive to tolerance, the *Sharpe* is not
   systematically so. Tightening tolerances to find a winner would be pure sample mining.
6. **Deflated Sharpe was charged against the whole registry** (`n_trials = total_trials × symbols`) as
   the harness intends; no `cp_*` config survives it in any meaningful sense given the above.

---

## Harness note (pre-existing bug, NOT worked around by editing the harness)

`runner.py` scores through `ProcessPoolExecutor`, whose workers call `registry.ALL[strat_name]` but
never call `registry.load_all()`. Under macOS's default `spawn` start method the workers therefore have
an empty registry and **every scoring job dies with `KeyError: <strategy name>`** — reproduced with
`--only donchian_production`, so it affects the whole harness, not this family. `--causal-only` is
unaffected (it runs in-process).

Nothing in `core.py`, `registry.py`, `indicators.py`, or the cost model was touched. The sweeps in this
report were driven by a scratchpad script that calls `runner._run_one_symbol` **verbatim** (identical
scoring path, identical `FIELDS`) from a `fork`-context pool created *after* `registry.load_all()`.
Calibration evidence that this changes nothing: `donchian_production` reproduces the published Phase-4
TRAIN numbers to two decimals (BTC +0.26 / ETH −0.45 / SOL +0.62).

**Suggested one-line fix for whoever owns `runner.py`:** call `registry.load_all()` at the top of
`_run_one_symbol` (idempotent), or create the pool with `mp_context=multiprocessing.get_context("fork")`.

## Reproduction

```
cd scripts/bruteforce
../../.venv/bin/python runner.py --causal-only --family chartpattern     # 9 passed, 0 disqualified
# scoring: see harness note above; sweeps were TRAIN and SELECT, full 20-symbol universe
```
