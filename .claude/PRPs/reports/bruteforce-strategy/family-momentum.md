# Family report: MOMENTUM

**Module**: `scripts/bruteforce/strategies/momentum.py`
**Strategies registered**: 10 · **Total combos**: 108 (max 12 per strategy)
**Tier**: 1D signal / 4H trigger, `max_hold_bars=180` (30 days)
**Splits used**: TRAIN (2023-08 → 2025-07), SELECT (2025-07 → 2026-01). **HOLDOUT untouched.**
**Causality audit**: 10/10 passed `core.assert_causal`; the `lookahead_canary` still fails, so the audit is live.

Raw sweeps: `/tmp/mom_train_full.csv`, `/tmp/mom_select_full.csv` (20 symbols × 108 combos = 2160 rows each).

---

## 1. Harness calibration check

`donchian_production` on TRAIN/CORE reproduces the stated bar exactly — BTC **+0.26**, ETH **−0.45**, SOL **+0.62**, mean per-symbol Sharpe **+0.142**. Every number below is comparable to that bar.

---

## 2. All 10 strategies — best TRAIN config and its SELECT result

Selection rule: best combo by **mean per-symbol Sharpe across the 20-symbol universe on TRAIN**, subject to `min trades ≥ 30 on every symbol` and `max cost_ratio ≤ 0.10`. `pos` = fraction of the 20 symbols with positive Sharpe.

| Strategy | Best TRAIN config | TRAIN mean Sharpe | TRAIN pos | SELECT mean Sharpe | SELECT pos | cost_ratio (max) | med risk_pct | med trades/sym (TR/SE) | Gate |
|---|---|---|---|---|---|---|---|---|---|
| `mom_accel` | lookback 30, gap 20, k 1.5 | **+0.712** | 19/20 | **+0.430** | 13/20 | 0.032 | 9.6% | 37 / 8 | pass |
| `mom_tsmom` | lookback 30, min_abs 0.05, k 1.5 | **+0.641** | 19/20 | −0.226 | 11/20 | 0.032 | 9.6% | 66 / 19 | pass |
| `mom_crash_guard` | long 60, short 20, k 2.5 | +0.540 | 19/20 | +0.007 | 9/20 | 0.019 | 15.9% | 57 / 17 | pass |
| `mom_macd_hist` | zwin 120, thr 0.5, k 1.5 | +0.503 | 17/20 | **−1.425** | 6/20 | 0.033 | 9.8% | 38 / 13 | pass |
| `mom_tsmom_trail` | lookback 60, trail 3.0, k 3.0 | +0.248 | 13/20 | **+0.534** | 15/20 | 0.016 | 19.7% | 45 / 11 | pass |
| `mom_xsec_rs` | lookback 30, top_q 0.3, k 1.5 | +0.094 | 12/20 | −0.299 | 13/20 | 0.033 | 9.9% | 41 / 10 | pass |
| `mom_xsec_sharpe` | lookback 30, top_q 0.3, k 1.5 | −0.007 | 11/20 | −0.409 | 11/20 | 0.031 | 9.9% | 41 / 10 | pass |
| `mom_dual` | lookback 30, top_q 0.3, k 2.5 | +0.397 | 17/20 | −0.164 | 11/20 | 0.019 | 15.6% | 25 / 7 | **FAIL — trades** |
| `mom_dual_long_only` | lookback 30, top_q 0.3, k 2.5 | +0.505 | 18/20 | −0.364 | 10/20 | 0.019 | 15.6% | 14 / 4 | **FAIL — trades** |
| `mom_vs_btc` | lookback 60, edge 0.0, k 2.5 | +0.277 | 16/20 | +0.960 | 18/20 | 0.017 | 16.5% | 21 / 7 | **FAIL — trades** |

Last three rows are the best *ungated* combo, since no combo of theirs reached 30 trades on all 20 symbols (best-case minimum: `mom_vs_btc` 26, `mom_dual` 19, `mom_dual_long_only` 8).

### Cost frontier: comfortably clear
Across all 2154 scored TRAIN rows: median `risk_pct` **11.8%** (IQR 9.6–16.7%), median `cost_ratio` **0.0118**, **max 0.036** — an order of magnitude inside the 0.10 ceiling and far above the ~1.4% risk floor. Moving the signal tier to 1D with `k·ATR(14,1D)` stops removed the cost problem entirely; the 58–68% cost consumption of the old fixed-0.5% stop model is gone (costs now consume ~1% of the risk unit). **No momentum strategy in this family is cost-constrained.** Average hold 102 trigger bars ≈ 17 days; long share 55%.

---

## 3. Does universe breadth (Option B) deliver the Sharpe lift the PRD hoped for?

**No — not through the cross-sectional mechanism, and the mechanism it does appear to deliver through is largely a metric artifact.** Three separate readings, all pointing the same way.

### 3a. The cross-sectional strategies are the family's worst performers
The two purely cross-sectional entries rank **9th and 10th of 10** on TRAIN mean Sharpe (`mom_xsec_rs` +0.094, `mom_xsec_sharpe` −0.007) and both go negative on SELECT (−0.299, −0.409). Plain single-asset time-series momentum (`mom_tsmom`, +0.641) beats relative strength by **7×** on the identical universe, stops, horizon and cost model. Adding the cross-section as a *filter* on top of absolute momentum (`mom_dual`, +0.397) makes it **worse** than absolute momentum alone, and additionally starves the strategy of trades. The ranking signal is not adding information — it is subtracting trades.

Vol-scaling the rank (`mom_xsec_sharpe`) did not rescue it either: it is marginally worse than raw-return ranking on the universe. So the failure is not the well-known "rank-on-return is a covert vol bet" confound.

The most plausible reason is the one the PRD itself already names: crypto perp returns are dominated by a single common factor. Cross-sectional dispersion over a 20-name, one-sector-of-one-asset-class universe is mostly idiosyncratic noise, not a persistent relative-strength ordering. `mom_vs_btc` — the one-benchmark control designed exactly to test this — is the only relative construct with a decent SELECT result (+0.960, 18/20 positive), and *it* is essentially "beat the market factor while the market factor is up", i.e. an absolute-momentum bet wearing a relative label.

### 3b. Widening the universe *lowered* per-symbol Sharpe for every strategy
Same config, TRAIN mean per-symbol Sharpe on CORE(3) vs UNIVERSE(20):

| Strategy | CORE(3) | UNIVERSE(20) |
|---|---|---|
| `mom_accel` | +1.162 | +0.712 |
| `mom_tsmom` | +0.883 | +0.641 |
| `mom_macd_hist` | +0.742 | +0.503 |
| `mom_crash_guard` | +0.725 | +0.540 |
| `mom_xsec_sharpe` | +0.515 | −0.007 |
| `mom_xsec_rs` | +0.156 | +0.094 |
| `mom_tsmom_trail` | +0.094 | +0.248 |

Nine of ten strategies are better on BTC/ETH/SOL than on the full 20. The 17 extended symbols are, per unit of risk taken, **worse** vehicles for momentum than the three majors. Breadth buys diversification, not signal quality.

### 3c. The portfolio-level "breadth lift" that does appear is mostly an accounting artifact
Equal-weighting the 20 per-symbol daily-return streams does produce a large Sharpe jump (`mom_accel`: mean per-symbol 0.712 → portfolio **2.355**), which naively looks exactly like the published Option B result. It should not be believed at face value. `trading_bot.backtest.equity.daily_returns` books a trade's entire PnL on its **exit day** and carries no mark-to-market between entry and exit. With ~17-day average holds, that turns genuinely overlapping, highly correlated positions into sparse point events, and the measured average pairwise correlation of the 20 streams collapses to **0.02–0.10**. A portfolio of 20 near-uncorrelated spikes mechanically shows √N diversification that the underlying correlated exposure does not have.

This is a property of the frozen metric, not a harness bug (the same accounting produces the baseline's numbers, so within-harness comparisons stay valid) — but it means **any portfolio-Sharpe claim from this harness, including the >1.5 Option B target, is not measurable here.** Confirming Option B honestly needs mark-to-market daily equity, which the current metric layer does not compute. I did not change `equity.py`, `core.py`, `indicators.py`, or the cost model.

### Verdict
Option B's *cross-sectional* premise is **not supported** on this universe and sample. Option B's *portfolio/inverse-vol sizing* premise is **untested**, because the exit-day return accounting cannot measure it. What the 20-symbol backfill *did* buy is a much stronger robustness test of single-asset momentum — 20 independent replications instead of 3 — and on that test, time-series momentum and its acceleration variant clear the production bar by a wide margin. That is a real result, but it is Option A's mechanism operating on more data, not Option B's mechanism working.

---

## 4. Peer-data causality — exactly how it was guaranteed

`core.assert_causal` truncates only the current symbol's frames; a peer frame from `core.load_frame` is full history in both runs, so **the audit provably cannot catch a peer leak**. Peer safety here is structural, and each of the four cross-symbol strategies (`mom_xsec_rs`, `mom_xsec_sharpe`, `mom_dual`, `mom_dual_long_only`, `mom_vs_btc`) routes 100% of its peer access through two functions:

1. **`_peer_signal(symbol, kind, period)`** computes the peer statistic on the peer's own 1D closes using **trailing windows only**: `ta.momentum` (a `shift(period)` log ratio) and `rolling(period, min_periods=period).std()`. No `shift(-n)`, no `center=True`, no `bfill`, no full-sample mean/std/quantile is applied to any peer series. Therefore peer value at peer-bar *j* is a function of peer bars ≤ *j* only. It returns the peer's **bar CLOSE times** (`index + TIMEFRAME_MS["1d"]`), never open times.

2. **`_align_peer(peer, my_close_times)`** is the only path from peer time to my time, and it is literally
   `idx = np.searchsorted(peer_close_times, my_close_times, side="right") - 1`, with `idx < 0` left as NaN — bit-identical to `Ctx._index_map` and to the production engine's regime alignment (`engine.py:316`). Two consequences: a peer bar that has not closed by my bar's close can never be selected, and the index chosen for my bar *i* depends **only on bar i's own close timestamp**, so it is invariant to how much history exists in either frame. Truncating my series does not change any earlier bar's peer index or value.

Three further deliberate choices:

- **The own-symbol leg never uses the peer cache.** `_own_signal` reads `ctx.frame("1d")` and projects with `ctx.align`, so the symbol's own contribution to the rank *is* truncated by the audit and the audit does bite on it. Only the other 19 names bypass truncation, and only via (1)+(2).
- **The rank is cross-sectional at a single bar, never across time.** `_xs_rank` compares the aligned peer values at bar *i* against my value at bar *i* and reports the fraction strictly below. No time-series statistic is taken of the rank, so ranking introduces no additional time dependence beyond what (1)+(2) already bound.
- **NaN is honoured as "not knowable yet."** Bars with fewer than `_MIN_PEERS = 12` valid peers, or with our own value NaN, produce `rank = NaN`, which yields no entry. Peers with insufficient history are dropped rather than imputed.

I reasoned through peer alignment by hand as required and state it explicitly: **no cross-sectional strategy in this module reads a peer bar that had not closed at or before the decision bar's close.**

---

## 5. What failed, and why

- **`mom_xsec_rs` / `mom_xsec_sharpe` (cross-sectional relative strength)** — the headline negative. Near-zero on TRAIN, negative on SELECT, worst two in the family. Cause: insufficient genuine cross-sectional dispersion in a single-factor-dominated 20-perp universe (§3a). Also the worst drawdowns in the family at the vol target (mean 26–27% vs 12–15% for the time-series strategies), i.e. it paid *more* risk for less return.
- **`mom_dual` / `mom_dual_long_only` (dual momentum)** — **disqualified on trade count**, not on Sharpe. Requiring absolute AND relative agreement is a conjunction of two sparse conditions; the best combo still leaves symbols with 19 and 8 trades over two years. `mom_dual_long_only` beat `mom_dual` on TRAIN (+0.505 vs +0.397) but lost on SELECT (−0.364 vs −0.164), so the intended attribution test is inconclusive: the short leg is not clearly a cost, and both are too thin to rank.
- **`mom_vs_btc`** — also **disqualified on trade count** (best minimum 26/20 symbols), despite the best SELECT number in the family (+0.960, 18/20 positive). Genuinely interesting and worth re-registering with a looser gate, but as measured it does not meet the ≥30 criterion and its TRAIN result (+0.277) is unremarkable, so the strong SELECT reading is likely small-sample luck over ~7 trades/symbol.
- **`mom_macd_hist`** — the sharpest TRAIN→SELECT collapse in the whole family: **+0.503 → −1.425**, positive symbols 17/20 → 6/20. A z-scored MACD histogram is a momentum *level* proxy with two extra smoothing constants and a normalisation window; it looked competitive in-sample and did not survive. Treat as overfit and drop.
- **`mom_tsmom` (plain time-series momentum)** — strong and remarkably consistent on TRAIN (+0.641, 19/20 positive) but **−0.226 on SELECT**. The regime cost it: SELECT is a 6-month window in which 30-day trend-following on 20 alts did not pay. This is the family's honest disappointment, because it is the effect with the deepest published prior.
- **`mom_crash_guard`** — the divergence exit did what it was designed to do on TRAIN (+0.540, 19/20) and then went flat on SELECT (+0.007, 9/20). It reliably *reduced drawdown* (lowest mean DD at target in the family, 12%) but did not add Sharpe out of sample. Keep as a risk overlay candidate, not a standalone signal.
- **Not a failure but a caveat that dominates all SELECT numbers**: the SELECT window is 6 months, which at momentum horizons yields only **4–19 trades per symbol**. Per-symbol SELECT Sharpes are extremely noisy and individual coin readings there (`mom_accel` FILUSDT −5.01 on 14 trades) should not be over-read.

---

## 6. Top-3 recommendations

### #1 `mom_accel` — acceleration (lookback 30, gap 20, k_atr 1.5)
The only strategy positive on both splits with a large TRAIN margin: **TRAIN +0.712 mean (19/20 symbols positive), SELECT +0.430 (13/20)**, vs the production bar of +0.142. Highest TRAIN DSR in the family (0.023). `cost_ratio ≤ 0.032`, median risk 9.6%, 37 trades/symbol on TRAIN. Trading the *change* in momentum rather than its level beat the level version on both splits, which is the family's cleanest positive finding.

Per-coin (TRAIN → SELECT Sharpe): BTC **1.42 → 0.45**, ETH **0.91 → 1.82**, SOL **1.16 → 2.80**. All three majors positive on both splits — it beats production Donchian on every core coin (BTC +0.26, ETH −0.45, SOL +0.62). Best extended names: UNI 1.13→1.42, TRX 0.99→2.37, LINK 0.67→2.37, XRP 0.71→1.81. Failures: **BCH −0.99→−1.63** (the only coin negative on both — exclude it), FIL 0.51→−5.01, NEAR 0.37→−1.68, DOT 0.98→−1.40.

### #2 `mom_tsmom_trail` — time-series momentum, ATR trail (lookback 60, trail_k 3.0, k_atr 3.0)
The **only** strategy whose SELECT beats its TRAIN (**+0.248 → +0.534**, 13/20 → 15/20 positive), which is the profile you want from an out-of-sample check rather than the reverse. Lowest cost of the family (`cost_ratio ≤ 0.016` at 19.7% median risk) and the mildest drawdown at target on SELECT (5.5%). Modest TRAIN Sharpe and DSR 0.004 mean it is not a strong in-sample winner — it is recommended for *stability*, not for headline return.

Per-coin: BTC **0.27 → 0.22**, ETH **0.30 → 1.41**, SOL **−0.29 → 1.22**. BTC/ETH positive on both; SOL recovers strongly. Extended standouts: XRP −0.38→1.81, BNB 0.61→1.65, TRX 0.81→1.50, ATOM 0.63→1.29, UNI −0.60→1.17. Persistent failures: **DOGE −0.43→−1.46** and **OPUSDT −0.19→−0.67** (negative on both, exclude); BCH 0.40→−0.60, NEAR 0.74→−1.28.

### #3 `mom_crash_guard` — as a risk overlay, not a standalone (long 60, short 20, k_atr 2.5)
TRAIN +0.540 with **19/20 symbols positive** — the joint-best breadth consistency in the family — and it produced the lowest drawdowns at the vol target (12% TRAIN / 7.6% SELECT). SELECT Sharpe is flat (+0.007), so it is not recommended as a signal. It is recommended as the **divergence-exit overlay** to bolt onto #1 or #2: the mechanism (exit when short-horizon momentum contradicts long-horizon) demonstrably cut tail risk on both splits without costing much, and it is the cheapest available answer to momentum's documented crash risk.

Per-coin TRAIN → SELECT: BTC 0.86 → −0.61, ETH 0.51 → 1.18, SOL 0.81 → −0.85. All three majors strongly positive on TRAIN; only ETH holds on SELECT. Best extended: TRX 0.73→1.34, ATOM 0.77→1.35, BNB 0.36→1.45, UNI 0.26→1.27. Worst: NEAR 1.27→−1.94, BCH 0.16→−1.74, FIL 0.75→−1.16.

**Coin-level guidance common to all three**: BCHUSDT is negative on SELECT for all three candidates and negative on both splits for #1 — drop it from any momentum sleeve. NEARUSDT is negative on SELECT for all three (and badly: −1.68 / −1.28 / −1.94). FILUSDT and DOGEUSDT are negative on SELECT for all three as well; OPUSDT is negative on both splits for #2 only. ETHUSDT is the single most reliable momentum vehicle (positive on both splits in all three), which is notable because it is production Donchian's *worst* symbol (−0.45) — the two families are complementary per-coin.

---

## 7. Robustness concerns (read before shortlisting anything)

1. **SELECT is too short for this horizon.** 6 months × ~17-day holds = 4–19 trades/symbol. Every SELECT Sharpe here has a standard error on the order of ±0.6–0.9. TRAIN/SELECT "agreement" claims in this report are therefore weak evidence, and the sign flips (`mom_tsmom` +0.641 → −0.226) may be regime, noise, or overfit — the data cannot separate them.
2. **Portfolio Sharpe is not measurable in this harness** (§3c). Exit-day PnL booking with no mark-to-market makes overlapping correlated positions look independent (measured ρ ≈ 0.02–0.10). Do not quote any portfolio number from this family against the PRD's >1.5 target.
3. **Only one strategy survived both splits with a positive sign** (`mom_accel`; `mom_tsmom_trail` improved but from a weak base; `mom_vs_btc` improved but is trade-starved). Out of 10 strategies × 108 combos, that is a thin survivor set and consistent with selection noise. DSR is low throughout (best 0.023).
4. **Grid discipline held** — 12 combos max per strategy, 108 total, 3 values per axis where possible — but the axes were not independent: `lookback ∈ {30,60,90}` won at **30 for six of the ten** best configs, i.e. the shortest value tested. That is a boundary optimum and a warning sign; the true optimum may be shorter than 30 days, which would move the family toward the cost frontier and out of the comfortable `cost_ratio` zone. Worth one deliberate extension of the lookback axis downward before trusting these configs.
5. **`min_abs = 0.05` dead band won on `mom_tsmom`**, i.e. the churn filter mattered. Any live implementation must reproduce the dead band, not just the sign rule.
6. **Extended-universe symbols carry survivorship-ish selection**: `universe.py` chose the 20 highest-liquidity perps *with full history to 2022-01*, which is a filter applied with knowledge of which names survived to 2026. Cross-sectional results on this set are optimistic relative to a universe formed in real time in 2023.
7. **HOLDOUT is untouched.** Nothing here has been validated out of sample in the PRD's one-shot sense.
