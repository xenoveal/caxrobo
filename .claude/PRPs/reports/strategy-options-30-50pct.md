# Strategy Options: Reaching 30–50%/yr at Moderate Risk

**Generated**: 2026-07-26 · **Builds on**: `market-research-capability-benchmark.md` (internal measurements) + fresh external research (sources per strategy)
**Status**: Research recommendation — nothing here is a profit projection. All figures are published backtests; budget 20–40% live degradation.

---

## Ground rules derived from the benchmark (non-negotiable, whatever strategy is chosen)

1. **Kill the `≤0.5% risk / ≥0.75% reward` band.** Measured noise alone stops out 78–95% of trades inside it. Replace with ATR-scaled stops (`k = 1.5 × ATR` on the setup timeframe) and a *ratio* filter (R:R ≥ 1.5–2.0), never absolute percentages.
2. **Move up one tier: 1D regime / 4H setup / 1H trigger.** Intraday cost-to-edge collapses below hourly bars (mean Sharpe −12.71 at 1m vs +0.79 at 60m, [arXiv 2602.10785](https://arxiv.org/html/2602.10785)).
3. **Honest costs, frozen**: 0.05% taker + 0.02% slip per side; prefer maker entries; keep cost/risk ≤ 0.10.
4. **The 30–50% target is a *sizing* outcome, not a signal outcome.** Every credible path below is: build a Sharpe ≥ 1.0 base strategy at 12–20% unlevered, then apply 2–3× leverage / volatility targeting. No published unlevered signal strategy on 3 majors delivers 30–50% at ≤25% drawdown directly.
5. **Retire "~1% good day / ≤1% bad day".** Replace with: Sharpe ≥ 1.0, max DD ≤ 25%, net of costs.

---

## Strategy Options

### Option A — Regime-Gated Donchian Trend + ATR Stops *(recommended core; smallest change from current code)*

**Recipe**: Keep the existing regime classifier (benchmark §2.6 — it's healthy). In *trending* regime, replace pivot-geometry patterns with a **Donchian channel breakout** (20/55-style entry/exit) confirmed by ADX > 25 and the 55-period mid-line as trend filter. Stop = 1.5×ATR(4H) (satisfies cost ratio ≤0.10 on all three symbols even all-taker, per benchmark R3). Exit on opposite-channel touch or ATR trail. In *ranging* regime, retain the existing Bollinger fade (its stop placement is already structurally correct) re-tested under the new risk model. Suppress in extreme-vol.

| Metric | Published evidence |
|---|---|
| Potential return | 12–20%/yr unlevered → **30–50% at 2.5–3× leverage** |
| Sharpe | 0.5–1.5 for MA/Donchian trend on BTC over a decade ([arXiv 2009.12155](https://arxiv.org/pdf/2009.12155)); 20/55 Donchian profitable through 2017–2023 bull/bear cycles with 30–40% win rate and 3–5× winner/loser ratio ([Algomatic](https://algomatictrading.substack.com/p/strategy-8-the-easiest-trend-system), [FinancialWisdom](https://www.financialwisdomtv.com/post/breakout-trading-using-donchain-channels)) |
| Drawdown | ~20–25% unlevered expected; levered scales proportionally — cap leverage so worst-case DD ≤ 25–30% |
| R:R | Effective ~2.5–4:1 (channel exits let winners run); filter floor 1.5:1 |

**Why it's resilient**: trend-following is the single most-replicated edge in the literature across 100+ years and every asset class; parameters (20/55, 1.5×ATR) are canonical, not fitted to this dataset. **Feasibility**: high — swaps `signals/breakout` internals, reuses regime layer, backtester, DB unchanged.

### Option B — Volatility-Targeted Rotational Trend (Zarattini-style) *(highest published Sharpe among directional options)*

**Recipe**: Ensemble of Donchian lookbacks (e.g. 20/55/100) on **daily bars**, applied across a wider universe (top 10–20 liquid perps, not just 3), with **inverse-volatility position sizing** and a portfolio-level volatility target (~20–30% annualized). Long-only or long/flat per coin; rotate into whatever is trending.

| Metric | Published evidence |
|---|---|
| Potential return | Net-of-fees Sharpe **> 1.5**, annualized alpha 10.8% vs BTC ([Zarattini, Pagani & Barbon, SSRN 5209907](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907), Swiss Finance Institute RP 25-80) |
| Sizing effect | Inverse-vol weighting lifted Sharpe 0.99 → 1.54 and cut max DD −30.8% → −13.8% in comparable portfolios ([Concretum](https://concretumgroup.com/position-sizing-in-trend-following-comparing-volatility-targeting-volatility-parity-and-pyramiding/), [QuantPedia](https://quantpedia.com/an-introduction-to-volatility-targeting/)) |
| Drawdown | ~15–25% at a 20–30% vol target |
| R:R | Trend-style (many small losses, few large winners) |

At Sharpe 1.5, a 25–30% vol target puts expected return squarely in the 30–45% range *without* discretionary leverage — the most honest route to the north star. **Feasibility**: medium — needs universe expansion (backfill more symbols; DB/schema already supports it) and a daily-bar portfolio loop; conflicts with the PRD's "3 symbols, single-method" constraint, which the user has explicitly allowed to change.

### Option C — Delta-Neutral Funding-Rate Carry Sleeve *(diversifier, not a core)*

**Recipe**: Hold spot (or coin-margined long) + short the perp when funding is persistently positive; collect funding. Market-neutral, no directional signal needed.

| Metric | Published evidence |
|---|---|
| Potential return | 12–25%/yr, Sharpe 3–6 in 2020–2023 backtests; DD typically <5% ([CoinCryptoRank](https://coincryptorank.com/blog/funding-rate-arbitrage), [ScienceDirect risk/return study](https://www.sciencedirect.com/science/article/pii/S2096720925000818)) |
| **Decay warning** | Full-sample Sharpe 6.45 fell to 4.06 from 2024 and **turned negative in 2025** ([arXiv 2510.14435](https://arxiv.org/pdf/2510.14435)) — the edge is crowded and regime-dependent |

Uncorrelated with trend, so even a modest carry sleeve lifts portfolio Sharpe. But it cannot be the plan: recent decay, and it conflicts with the alert-and-execute product shape (continuous two-leg positions). **Feasibility**: low-medium — needs funding-rate history ingestion (not in DB) and a different execution model. Treat as Phase 8+ optionality.

### Option D — Hybrid: A-core + vol-targeted sizing + optional C sleeve *(the recommended overall shape)*

Option A as the signal engine, Option B's **volatility targeting** as the sizing layer (target ~25% annualized vol, sized by the human per alert: `position = target_vol / realized_vol × equity`), Option C added later if constraints relax. Combining orthogonal sleeves is what pushes portfolio Sharpe above 1.0 (benchmark §2.7: combined crypto factor portfolios >1.0 vs 0.4–0.7 standalone; vol targeting adds Sortino +30–40% at 20–50% targets — [ForTraders](https://www.fortraders.com/blog/volatility-based-position-sizing-explained), [Van Hemert, *The Impact of Volatility Targeting*](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf)).

**Expected profile**: Sharpe 1.0–1.5 · 30–45%/yr at 25% vol target · max DD 20–25% · effective R:R ~2.5:1+. This is the only option that hits all three constraints (moderate risk, 30–50%, non-overfit) simultaneously.

---

## Recommended parameter changes to `initial-requirements` / PRD

| Old | New | Deduction |
|---|---|---|
| Risk ≤0.5%, reward ≥0.75% (absolute) | Stop = 1.5×ATR(setup TF); R:R ratio ≥ 1.5 | Old band sits entirely inside the measured noise floor; 0.5% is an account-risk budget, not a stop distance |
| 15m trigger / 1H setup / 4H regime | 1H trigger / 4H setup / 1D regime | Cost-to-edge frontier ([arXiv 2602.10785](https://arxiv.org/html/2602.10785)); human-executable alerts |
| ~1%/day payoff framing | Sharpe ≥ 1.0, DD ≤ 25% net of costs | No market pays a fixed daily rate; compounded it exceeds all published results |
| 30–50% from signal win rate | 12–20% unlevered × 2–3× vol-targeted leverage | Matches published capability benchmarks (5–25% net for good retail bots) |
| Fee 0.04% | 0.05% taker (frozen), prefer maker 0.02% | Binance VIP-0 published schedule |
| Chart-pattern taxonomy | Donchian/ATR channel breakout | Deeper evidence base, simpler to audit, no geometry parameters to overfit |

## Validation gates before believing any number

Walk-forward with pooled 3-symbol folds, `WF_MIN_TRADES ≥ 30`, Deflated Sharpe Ratio ([Bailey & López de Prado](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)), `k` derived from the cost constraint and frozen (never swept), exits resolved on 1m bars, cross-symbol consistency required. Any option that fails these gates is rejected regardless of its headline backtest.

## Sequencing

1. Fix cost model + risk model (benchmark R1/R3) — re-gate existing methods (cheap falsification test).
2. Implement Option A (Donchian swap + timeframe shift) — gate via repaired walk-forward.
3. Add vol-target sizing guidance to alerts (Option D sizing layer).
4. If Sharpe < 1.0 after step 2: expand universe toward Option B.
5. Carry sleeve (Option C) only after funding-history ingestion and explicit product-shape decision.

## Sources

- [Zarattini, Pagani & Barbon — Catching Crypto Trends (SSRN 5209907)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907)
- [A Decade of Evidence of Trend Following in Cryptocurrencies (arXiv 2009.12155)](https://arxiv.org/pdf/2009.12155)
- [Double out-of-sample walk-forward optimization, intraday crypto (arXiv 2602.10785)](https://arxiv.org/html/2602.10785)
- [Crypto as an Investable Asset Class — carry decay (arXiv 2510.14435)](https://arxiv.org/pdf/2510.14435)
- [Funding-rate arbitrage risk/return on CEX & DEX (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- [Van Hemert et al. — The Impact of Volatility Targeting](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf)
- [Concretum — Position Sizing in Trend-Following](https://concretumgroup.com/position-sizing-in-trend-following-comparing-volatility-targeting-volatility-parity-and-pyramiding/)
- [QuantPedia — Introduction to Volatility Targeting](https://quantpedia.com/an-introduction-to-volatility-targeting/)
- [Bailey & López de Prado — Deflated Sharpe Ratio (SSRN 2460551)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Algomatic — Donchian 20/55 BTC backtest](https://algomatictrading.substack.com/p/strategy-8-the-easiest-trend-system) · [FinancialWisdom — Donchian breakout drawdowns](https://www.financialwisdomtv.com/post/breakout-trading-using-donchain-channels) *(practitioner — indicative)*
- [CoinCryptoRank — Funding-rate arbitrage guide](https://coincryptorank.com/blog/funding-rate-arbitrage) *(vendor — indicative)*
- Internal: `.claude/PRPs/reports/market-research-capability-benchmark.md` (noise floor, cost ratio, cross-symbol failure measurements)
