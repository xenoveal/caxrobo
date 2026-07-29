# Market Research: Capability Benchmark & Path to the Profit Target

**Scope**: Benchmark the L1 crypto signal bot against industry capability and published best practice; recommend what must change for the PRD's north-star (~30–50% annualized) to be reachable.
**Inputs**: `phase5-breakout-strategy-investigation.report.html`, `l1-crypto-signal-bot-mvp.plan.md` / `.prd.md`, full source review of `src/trading_bot/`, plus new measurements run against `data/ohlcv.db` (3 symbols × 15m/1h/4h, 2022-12-31 → 2026-07-23).
**Generated**: 2026-07-26

---

## 1. Executive Summary

The engineering is genuinely good. The **strategy specification is mathematically infeasible**, and the reason is one parameter pair that nobody has questioned because it came from the user's own manual process and was frozen into the PRD as a hard requirement.

The Phase 5 report concluded the *breakout method* lacks edge. That conclusion is too narrow. I re-ran the pipeline across all three symbols (the report only tested BTCUSDT) and measured the raw noise characteristics of the data. The finding:

> **Every signal method, on every symbol, in every regime bucket with a meaningful sample, loses money — and the loss rate is fully explained by random price noise, not by bad signals.** The `≤0.5% risk / ≥0.75% reward` band forces stops so tight that ordinary 15-minute price movement closes 91–95% of trades before the thesis can play out. The band is not selecting good trades; it is selecting trades that cannot survive.

This is why the report's stop-buffer sweep showed profit factor flat and below 1.0 across the *entire* feasible range: no parameter inside that band can work. The band itself is the defect.

**The single most consequential recommendation**: the PRD conflated two different things under the word "risk" — *how far the stop sits from entry* (a market-structure question, governed by volatility) and *how much of the account is at risk* (a position-sizing question). It applied a sizing constraint (0.5%) as a stop-distance rule. Since the bot deliberately doesn't size positions, the 0.5% number has no business constraining stop distance at all. Replace it with volatility-scaled (ATR-multiple) stops and filter on the R:R *ratio* instead of on absolute percentages.

**On the profit target**: 30–50% annualized is achievable in principle but not via this architecture, and not without leverage. The honest path is a lower-return, higher-Sharpe base strategy with sizing applied on top — which is a decision the PRD assigns to the human, not the bot. The "~1% net gain on good days, ≤1% loss on bad days" framing should be retired; it implies a fixed daily payoff that does not exist in any market and compounds to an implausible number.

---

## 2. Key Findings

### 2.1 The measured noise floor sits above the entire permitted risk band

Median 15-minute bar range, measured across ~124,800 bars per symbol:

| Symbol | median bar range | p75 | p90 | median close-to-close |
|---|---|---|---|---|
| BTCUSDT | 0.251% | 0.399% | 0.620% | 0.105% |
| ETHUSDT | 0.339% | 0.532% | 0.822% | 0.136% |
| SOLUSDT | **0.520%** | 0.785% | 1.176% | 0.216% |

On SOLUSDT, **one median 15-minute bar equals the entire maximum permitted stop distance.** The `MAX_RISK_PCT = 0.005` cap was set once, globally, with no reference to per-symbol volatility.

### 2.2 Noise alone explains the observed stop-out rate

Probability that price moves adversely by X% within N 15-minute bars, measured unconditionally over the full history (no strategy, no edge assumed — pure random-entry adverse excursion):

| Symbol | stop dist | N=1 | N=4 | N=12 | N=96 (the 24h hold limit) |
|---|---|---|---|---|---|
| BTCUSDT | 0.10% | 51.6% | 73.2% | 84.8% | 95.5% |
| BTCUSDT | 0.20% | 26.4% | 52.4% | 70.9% | **91.2%** |
| BTCUSDT | 0.50% | 5.3% | 20.0% | 40.3% | 78.0% |
| ETHUSDT | 0.20% | 36.8% | 62.3% | 77.9% | **93.2%** |
| SOLUSDT | 0.20% | 54.0% | 75.3% | 85.9% | **95.4%** |

The Phase 5 report observed a 90.6% stop-out rate pre-fix, at a typical unbuffered stop distance under 0.10%. **Random noise at that distance produces a 95.5% adverse-hit rate.** The strategy's failure rate is not merely *consistent with* noise — it is indistinguishable from it. There is no residual signal left to measure.

Critically, even at the **widest stop the band permits (0.5%)**, noise alone hits it 78% of the time on BTC and 89% on SOL within the hold window. The whole feasible interval is inside the noise floor. This is the mechanical reason the buffer sweep was flat.

### 2.3 The R:R band is an adverse-selection machine

Measured over the 861 BTCUSDT trades the engine actually generated:

| Realized stop distance | value |
|---|---|
| p10 | 0.116% |
| p25 | 0.140% |
| **median** | **0.207%** |
| p75 | 0.324% |
| p90 | 0.426% |
| median target distance | **2.28%** |

Median nominal R:R is roughly **11:1**. That looks like a spectacular screen and is in fact the tell. The filter demands `risk ≤ 0.5%` *and* `reward ≥ 0.75%` simultaneously. Because the target is fixed by pattern geometry, the only candidates that satisfy both are those where the entry happens to sit a hair away from the level. **The screen systematically selects the trades with the least survivable stops** — 48% of accepted trades have stops under 0.20%, and 70% under 0.30%.

An "11:1 reward-to-risk setup" on a 15-minute chart is not an opportunity. It is a lottery ticket priced by noise.

### 2.4 Transaction costs consume most of the risk unit

- Modeled round trip: `2 × (0.04% fee + 0.02% slippage) = 0.12%`
- Against the median realized stop distance of 0.207%: **costs are 58% of the risk unit.**
- Worse, the modeled fee is optimistic. Binance's published USDT-M VIP-0 **taker** fee is **0.05%**, not 0.04% ([Binance fee schedule](https://binancemakertakerfee.org/), [TradersUnion](https://tradersunion.com/brokers/crypto/view/binance/futures-fees/)). At a realistic `2 × (0.05% + 0.02%) = 0.14%`, costs are **68% of the median risk unit**.
Break-even win rate is `p* = (1 + c) / (1 + m)`, where `c = cost/risk` and `m` is the R:R ratio. **The cost ratio inflates the required win rate by exactly the factor `(1 + c)`.** At today's `c = 0.58`, friction alone raises the bar by 58% — at a sane `m = 2`, that moves break-even from 33.3% to 52.7%, a 19.4pp penalty paid before any signal quality enters the picture.

**The decisive consequence — gross expectancy is approximately zero.** Adding the modeled 0.12% cost back to each symbol's net expectancy:

| Symbol | net expectancy/trade | **gross of costs** |
|---|---|---|
| BTCUSDT | −0.127% | **−0.007%** |
| ETHUSDT | −0.092% | **+0.028%** |
| SOLUSDT | −0.110% | **+0.010%** |

Two conclusions follow, and both matter:

1. **Gross ≈ 0 means no edge was detected.** This is exactly what a strategy with no predictive content produces. So costs are not "the problem to fix" in the sense that halving them yields profit — halving them moves BTC from −0.127% to −0.067%, still negative.
2. **But the cost ratio sets the bar the signal must clear.** Required gross edge per trade must exceed `C`; expressed in risk-units that is exactly `c`. Today `c = 0.58`, so the signal must generate **0.58 R of gross edge per trade**. Published systematic trend strategies operate around **0.05–0.15 R per trade**. The current specification demands roughly **4–10× more edge than real strategies deliver.** That is the quantitative statement of why the spec is infeasible, and it is independent of which signal method is used.

For calibration, published research puts the profitability threshold for intraday crypto moving-average strategies at roughly **0.4% total transaction cost** ([arXiv 2602.10785](https://arxiv.org/html/2602.10785)) — and that is for strategies holding for hours with wide stops. A system where friction eats two-thirds of every risk unit is far outside any regime where an edge can express itself.

### 2.5 Cross-symbol, cross-method backtest: uniform failure

The report tested BTCUSDT only. Running the existing `cli.py backtest` across all three symbols at default parameters — the cheapest robustness test available, and the one that was skipped:

| Symbol | trades | win rate | expectancy/trade | profit factor | max DD |
|---|---|---|---|---|---|
| BTCUSDT | 861 | 12.66% | −0.127% | 0.59 | 112.0% |
| ETHUSDT | 827 | 11.61% | −0.092% | 0.71 | 79.7% |
| SOLUSDT | 1,050 | 9.14% | −0.110% | 0.68 | 122.9% |

Per-bucket, every bucket with n ≥ 100 is negative:

| Bucket | BTC | ETH | SOL |
|---|---|---|---|
| trending/triangle | PF 0.65 (n=509) | PF 0.71 (n=452) | PF 0.60 (n=523) |
| trending/flag | PF 0.50 (n=137) | PF 0.67 (n=201) | PF 0.81 (n=330) |
| ranging/bollinger-fade | PF 0.49 (n=203) | PF 0.76 (n=163) | PF 0.86 (n=155) |

Two things follow. First, the failure is **not symbol-specific** — it reproduces across three assets and ~2,740 trades. Second, and more important: **the Phase 4 mean-reversion method fails too.** The report indicted only the breakout method, but fade fails on all three symbols as well. Two structurally unrelated signal methods failing identically points to the component they *share* — the R:R band and the cost model — not to either method's logic.

There is one supporting hint in the same table: fade win rates (20–24%) are roughly double breakout win rates (7–11%), and fade has the best profit factors. Fade places its stop at the excursion extreme, which is structurally *further away* than a hairline buffer past a breakout level. Wider stop → better outcome, exactly as the noise math predicts.

### 2.6 The regime classifier is sound and should be kept

Time-in-regime at default thresholds:

| Symbol | trending | ranging | extreme-vol | uncertain |
|---|---|---|---|---|
| BTCUSDT | 39.8% | 45.1% | 12.4% | 2.7% |
| ETHUSDT | 36.4% | 48.5% | 12.5% | 2.7% |
| SOLUSDT | 42.5% | 43.3% | 11.6% | 2.7% |

This is a healthy, non-degenerate distribution — no bucket is starved, extreme-vol suppression fires ~12% of the time as intended. The Wilder ADX/ATR implementation is correct, uses trailing windows only, and the classifier is used *defensively* (to suppress) rather than predictively, which is precisely the discipline the PRD's own source research recommends. **This layer is a keeper.** The problem is downstream of it.

### 2.7 What the literature says actually survives out-of-sample

The evidence base for **geometric chart patterns** is weak. Vendor-side claims put crypto pattern-breakout hit rates at 60–65% on *daily* bars with ~35% false-breakout rates absent volume confirmation ([chartscout](https://chartscout.io/chart-patterns-and-volume-analysis), [QuantStrategy on Bulkowski](https://quantstrategy.io/blog/how-to-backtest-chart-patterns-using-bulkowskis-statistical/)) — but these are promotional sources, on daily timeframes, with no cost modeling and no out-of-sample protocol. There is no peer-reviewed body of work establishing that pivot-geometry pattern detection produces net-of-cost edge on 15-minute crypto bars. The PRD itself flagged this as a HIGH-likelihood risk. It was right.

What *does* have published support:

| Edge source | Evidence | Standalone Sharpe |
|---|---|---|
| Perp **funding-rate carry** | One of four orthogonal crypto premia | ~0.4–0.7 |
| **Cross-sectional momentum** | Strongest at ~1-week holding in crypto (vs 12-month in equities); weak once illiquid coins excluded (Grobys & Huhta-Halkola 2019) | ~0.4–0.7 |
| **Time-series trend** | Works, but crypto favors cross-sectional over time-series | ~0.4–0.7 |
| **Combined 4-factor** | Simple averaging of orthogonal signals | **>1.0** |

Sources: [Quantt 2026 crypto quant review](https://www.quantt.co.uk/resources/crypto-quant-strategies-2026), [unravel.finance on cross-sectional alpha](https://blog.unravel.finance/p/cross-sectional-alpha-factors-in), [QuantPedia time-series vs cross-sectional](https://quantpedia.com/time-series-vs-cross-sectional-implementation-of-momentum-value-and-carry-strat/).

Note the honest tension: **the highest-evidence edges do not fit this PRD's shape.** Funding carry is a multi-day hold with no discrete entry alert. Cross-sectional momentum needs a wider universe than three assets to rank. Factor combination is explicitly forbidden ("no ensemble/blended voting"). This is a real constraint conflict for the user to resolve, not something to paper over.

### 2.8 Timeframe is on the wrong side of the cost frontier

The most directly transferable study tested EMA-crossover on intraday BTC/ETH/BNB with a **double out-of-sample** walk-forward protocol, sweeping 81 train/test window combinations across six timeframes ([arXiv 2602.10785](https://arxiv.org/html/2602.10785)):

| Timeframe | mean Sharpe across all 81 configs |
|---|---|
| 1-minute | **−12.71** |
| 60-minute | **+0.791** (all 81 configs positive) |

Best 60-minute config: Sharpe 1.252, 94.8% annualized, 35.2% max drawdown — though on the truly unseen period it performed "similarly to Buy-and-Hold but with lower drawdown and a higher Information Ratio." Statistical significance was marginal (bootstrap 3.5–4.4%).

The lesson is unambiguous: **at short timeframes transaction costs dominate and swamp any signal; the same logic on hourly bars is viable.** A 15-minute trigger with a 0.2% stop is much closer to the 1-minute end of that spectrum than the 60-minute end, in cost-to-edge terms.

### 2.9 Realistic performance benchmarks

| Benchmark | Figure | Source |
|---|---|---|
| Well-configured retail bot, net annual, above buy-and-hold | **5–25%** | [Altrady 2026](https://www.altrady.com/blog/crypto-bots/are-ai-crypto-trading-bots-profitable-2026) |
| Sharpe: acceptable / good / excellent | 1.0 / 1.5 / 2.0+ | [Altrady](https://www.altrady.com/blog/risk-management/sharpe-ratio-sortino-ratio-crypto), [Cryptorobot](https://cryptorobot.ai/performance/sharpe) |
| Max drawdown, well-managed bot | 15–20%; industry max 25% | Altrady |
| **Live-vs-backtest degradation** | **20–40% worse live** | [Bitget/Freqtrade review](https://www.bitget.com/academy/freqtrade-cryptocurrency-trading-bot-america-2026-comprehensive-guide-to-automated-trading) |
| Dominant failure mode | Curve fitting | Altrady, [Blockchain Council](https://www.blockchain-council.org/cryptocurrency/backtesting-ai-crypto-trading-strategies-avoiding-overfitting-lookahead-bias-data-leakage/) |

Against this, the PRD's 30–50% annualized north-star is at the **aggressive-but-not-absurd** end — *if* Sharpe ≈ 1.0+ is achieved first and leverage is applied deliberately. It is not reachable as a raw unlevered signal win rate. The "~1% good day / ≤1% bad day" formulation is the part to discard: it encodes a fixed daily payoff, and if compounded it implies returns an order of magnitude beyond anything in the literature.

---

## 3. Implications

1. **Phase 5's verdict was directionally right but misattributed.** It read the failure as "the breakout method has no edge." The cross-symbol run shows the mean-reversion method fails too. The correct reading is: *the shared risk model makes any signal method unprofitable*, so no signal method can clear the gate until it is replaced. Continuing to redesign breakout logic (the report's suggested next levers — wider R:R bands, different entry confirmation, longer holds) would have partially stumbled onto the fix without naming the cause.
2. **The walk-forward harness validated the wrong parameters.** Its grid is `(adx_trend_threshold, atr_extreme_percentile, bb_num_std)` — the regime layer, which my measurements show is the *healthiest* part of the system. The actual determinants of outcome (`MAX_RISK_PCT`, `MIN_REWARD_PCT`, `BREAKOUT_STOP_BUFFER_PCT`, `MAX_HOLD_BARS_15M`) are frozen at unvalidated defaults and never swept. The "FAIL" verdict is therefore a verdict on one arbitrary risk configuration, not on the strategy family.
3. **`WF_MIN_TRADES = 5` is itself an overfitting mechanism.** Selecting the max-expectancy combo from 18 candidates where each needs only 5 trades to qualify is close to pure noise-fitting. The observed plateau ratios as low as −11.3 are partly an artifact of the metric: `mean(neighbors)/best` is numerically unstable when `best` is a small positive number, so it reports catastrophic fragility even for mildly noisy neighborhoods.
4. **The alert-only, manual-execution design argues *for* a longer timeframe, not against it.** The user's original problem was missing entries because lower-timeframe confirmation prints while they aren't watching. A 15-minute trigger *maximizes* that failure mode. An hourly trigger gives a human time to actually place the order — so moving up a tier serves the founding user need and the cost-to-edge problem simultaneously.
5. **Backtest exit resolution is doing too much work.** With stops at 0.2% and median bar range at 0.25%, the conservative "stop fills first on ambiguous bars" rule is invoked constantly. It is the honest choice, but at these distances the 15-minute bar simply cannot resolve the sequence. Exits need finer bars to be measurable at all — another symptom of stops being below the data's resolution.

---

## 4. Risks & Caveats

- **My noise-hit probabilities are unconditional.** They assume no directional edge at entry. A genuine edge would lower them. The argument does not depend on the absence of an edge — it shows that at 0.2% stops the noise term is so dominant (91–95%) that any plausible edge is buried. But it is not a proof that no edge exists.
- **Median-based reasoning hides tail structure.** Volatility clusters; the strategy may be entering preferentially in higher- or lower-volatility conditions than the unconditional median. I did not condition the noise measurement on regime.
- **The cross-symbol backtest used default parameters only.** It is not a walk-forward result and shouldn't be read as one. Its purpose is narrow: to test whether failure is BTC-specific. It isn't.
- **Several benchmark figures come from vendor/marketing sources** (Altrady, Bitget, chartscout, Cryptorobot) and should be treated as indicative, not authoritative. The two load-bearing external citations — arXiv 2602.10785 on timeframe/cost, and Bailey & López de Prado on validation — are peer-reviewed or widely-cited academic work. The crypto-factor Sharpe figures (0.4–0.7 / >1.0) come from practitioner research and I could not verify them against a primary academic source.
- **The ~0.4% break-even cost threshold from arXiv 2602.10785 is for EMA-crossover strategies specifically** and does not transfer mechanically to a pattern-breakout system with different turnover.
- **Recommending ATR stops widens per-trade risk.** At 1.5× ATR on BTC 15m this is roughly 0.4–0.5% per trade versus 0.2% today — meaning the human must size positions *smaller* to hold account risk constant. If that isn't communicated clearly, the change increases real risk rather than fixing anything.
- **Regulatory framing**: the PRD's own note stands — SEC/CFTC/NASAA flag "consistently profitable AI trading bot" claims as the dominant 2025 crypto-fraud pattern. Nothing here should be read as a projection of profit.
- **Edge decay is real and the PRD already flags it.** Even a validated configuration requires ongoing re-validation; crypto carry Sharpe went negative in 2025.

---

## 5. Recommendation

**Do not build Phase 6 or 7.** The report was right to pause. But do not treat the breakout method as the thing to redesign either — insert a **Phase 5.5: Risk-Model Re-specification** ahead of any further signal work.

### R1 — Replace the fixed-percentage R:R band with volatility-scaled stops *(highest impact, smallest diff)*

This is the one change most likely to move the system from structurally-impossible to merely-hard.

- Stop distance = `k × ATR(period)` on the setup timeframe, `k ∈ [1.0, 2.0]`. Published intraday guidance is 1.0–2.0× ATR for short holds; 2× ATR has been shown to cut max drawdown by roughly a third versus static stops ([Volatility Box](https://volatilitybox.com/research/volatility-adjusted-stop-losses/), [Traders Second Brain](https://traderssecondbrain.com/guides/stop-loss-placement-methods)).
- **Filter on the R:R *ratio* (e.g. ≥ 1.5), not on absolute risk and reward percentages.** This single change removes the adverse selection: candidates no longer qualify by having freakishly tight stops.
- **Delete `MAX_RISK_PCT` as a stop-distance constraint.** State plainly in the PRD that 0.5% is an *account-risk* budget, discharged by the human's position sizing, and has no bearing on where the stop belongs. This is the conceptual correction the whole project turns on.
- Make `k` per-symbol or ATR-derived — a global constant cannot serve BTC (0.251% median bar) and SOL (0.520%) at once.
- Expect far fewer signals with much larger individual risk. That is the point.

### R2 — Shift every tier up one step: 1D regime / 4H setup / 1H trigger

- Preserves the PRD's justified 3-tier structure and 4–6× spacing; only the absolute scale moves.
- Directly addresses the cost-to-edge collapse documented at short timeframes (1-min mean Sharpe −12.71 vs 60-min +0.791).
- Improves the *founding* user problem: an hourly trigger is executable by a human who isn't glued to a screen; a 15-minute trigger is not.
- Extend `MAX_HOLD_BARS` proportionally, and **model funding cost** — at multi-day holds it stops being negligible, and the PRD already promises "net of realistic fees/funding" while the engine models only fee + slippage.

### R3 — Fix the cost model, then attack costs

- Correct `FEE_PCT` to **0.05%** (Binance USDT-M VIP-0 taker), not 0.04%. The current figure understates taker friction by 25%.
- Then reduce it: prefer **limit/maker entries** (0.02% maker) where the signal tolerates non-immediate fills, and apply the BNB discount (~10%). Maker-side entry roughly halves round-trip friction — which, at these margins, is a larger effect than most parameter tuning.
**Adopt a derived cost-ratio target: `c = cost / risk ≤ 0.10`.** Justification, not a round number: since `p* = (1+c)/(1+m)`, a cost ratio of 0.10 costs ~3–4pp of required win rate at sensible R:R — friction stays a second-order term. At `c ≥ 0.25` friction exceeds what most real edges are worth. Today `c` is 0.58–0.68.

Achievable round-trip cost, by execution style:

| Execution | Round trip | Notes |
|---|---|---|
| All-taker, current model | 0.12% | understates fees |
| All-taker, realistic | **0.14%** | VIP-0 taker 0.05% + 0.02% slip, per side |
| All-taker + BNB discount | ~0.13% | ~10% off fees |
| **Maker entry + limit target** | **~0.08%** | entry 0.02% no-slip; stop exits still taker |

`c ≤ 0.10` therefore requires **risk ≥ 1.40%** (all-taker) or **risk ≥ 0.80%** (maker entry). Translating via measured median ATR(14):

| Median ATR(14) | BTCUSDT | ETHUSDT | SOLUSDT |
|---|---|---|---|
| on 1H bars | 0.628% | 0.842% | 1.206% |
| on 4H bars | 1.314% | 1.788% | 2.503% |

- **`k = 1.5 × ATR(1H)` satisfies `c ≤ 0.10` on all three symbols with maker-side entry** (risk 0.94% / 1.26% / 1.81% → `c` = 8.5% / 6.3% / 4.4%). It also sits mid-range in the published 1.0–2.0× intraday guidance — so this is a constraint-derived value, not a fitted one.
- All-taker at `k = 1.5×ATR(1H)`, BTC fails narrowly (`c` = 14.9%). Either use maker entries or `k = 2.0` on BTC.
- **Shifting to 4H setup bars (R2) fixes the cost ratio for free**: at `k = 1.5 × ATR(4H)`, `c` = 7.1% / 5.2% / 3.7% even all-taker. R2 and maker execution are partial substitutes — do the cheaper one first.

**Never sweep the cost parameters.** Set `FEE_PCT`/`SLIPPAGE_PCT` pessimistically once and freeze them. Optimistic cost assumptions are a silent overfitting channel: the current 0.04-vs-0.05% understatement alone inflates every backtest by ~0.02%/trade, which at ~250 trades/year is ~5%/year of phantom return.

### R4 — Replace chart-pattern geometry with a better-evidenced trending method

Keep the regime-gated, single-method-at-a-time architecture (§2.6 shows it works). Swap the *trending* method's internals:

- Prefer **volatility/channel breakout** (Donchian or ATR-channel with trend confirmation) over pivot-geometry pattern detection. Trend-following has a far deeper published evidence base than chart-pattern recognition, and it is dramatically simpler to specify, test, and audit — no symmetry tolerances, no pattern taxonomy.
- This also retires the PRD's open question about pattern taxonomy, which the data has now effectively answered: triangle and flag account for ~98% of volume and both lose on all three symbols; head-and-shoulders variants never accumulated enough samples (n=5–27) to judge and likely never will.
- **Retain the fade method for ranging regimes** — it has the least-bad numbers and its stop placement (excursion extreme) is already structurally correct. Re-test it under R1/R2 before judging it.
- If the user is open to relaxing PRD constraints, **funding-rate carry** is the highest-evidence addition available (Sharpe ~0.4–0.7) and is nearly uncorrelated with trend. It requires ingesting funding-rate history (not currently in the DB) and conflicts with the alert-and-execute shape. Flagging as a strategic option, not a drop-in.

### R5 — Repair the validation protocol before re-running the gate

The harness is well-architected; its configuration is what fails. Concretely:

- **Sweep the parameters that matter.** Add stop multiple `k`, R:R floor, and hold limit to the grid. Regime thresholds barely bind and can be held fixed.
- **Raise `WF_MIN_TRADES` from 5 to ≥30** for parameter *selection*. Selecting a max from 18 combos at n=5 is noise-fitting by construction.
- **Pool the three symbols into each fold's statistics.** This is the highest-leverage overfitting-risk reduction available, and it resolves a genuine tension: wider (ATR-scaled) stops mean fewer qualifying trades, and smaller samples raise overfitting risk — partly cancelling the benefit of R1. At ≥30 trades per 60-day test fold you need ~180 trades/year; BTC alone currently yields ~246/year and R1 will cut that. Pooling gives **3× the sample at zero additional degrees of freedom**, and it simultaneously enforces the cross-symbol consistency gate below. Buying statistical power without buying parameters is the only free lunch in this list.
- **Derive the stop multiple `k` from the cost-ratio constraint (R3) and then freeze it — do not sweep it.** Converting a free parameter into a derived constraint removes an axis from the grid, which lowers probability-of-backtest-overfitting directly. Fitting `k` on returns is precisely what the Phase 5 buffer sweep did, and it has already consumed a degree of freedom on BTCUSDT 2023–2026.
- **Replace `plateau_ratio`.** Use the fraction of grid neighbours that are also positive, plus the absolute spread of neighbour expectancies. Avoid dividing by a near-zero best value.
- **Add the Deflated Sharpe Ratio** (Bailey & López de Prado 2014) to correct for selection bias across the 18-combo × 17-fold search — this is exactly the multiple-testing situation DSR exists for ([SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).
- **Consider Combinatorial Purged Cross-Validation** in place of, or alongside, rolling walk-forward. Published comparisons find CPCV yields lower probability-of-backtest-overfitting and higher DSR than walk-forward CV ([CPCV insights](https://www.scribd.com/document/725401650/SSRN-id4778909)).
- **Make all three symbols mandatory, and require cross-symbol consistency as a gate.** §2.5 shows how much signal this cheap test carries; running it earlier would have caught the shared-component diagnosis before the buffer investigation.
- **Resolve exits on 1-minute bars.** With stops near the 15-minute bar range, the conservative same-bar rule is guessing on a large fraction of trades.
- **Keep a written trial log.** The PRD mandates it; the buffer sweep in the Phase 5 report is now a consumed degree of freedom on BTCUSDT 2023–2026, and the honest OOS holdout is shrinking.

### R6 — Reset the north-star to something the evidence can support

- Retire "~1% net gain on good days, ≤1% loss on bad days." No market offers a fixed daily payoff, and compounded it implies returns far beyond anything documented.
- Replace it with a **Sharpe-first** objective: target **Sharpe ≥ 1.0 with max drawdown ≤ 25%**, measured net of realistic costs, before any discussion of return magnitude.
- Reach 30–50% annualized the honest way: a Sharpe ≈ 1.0 strategy yielding perhaps 12–20% unlevered, with **2–3× leverage applied deliberately by the human**, and drawdown scaling proportionally. This keeps return magnitude where the PRD already put it — in the human's sizing decision — instead of demanding the signal layer produce it.
- Budget for **20–40% live degradation versus backtest** as the expected case, not the pessimistic one.

### Suggested sequencing

| Step | Action | Gate |
|---|---|---|
| 1 | R3 — correct fee to 0.05%, add funding cost | Cost model honest |
| 2 | R1 — ATR stops + R:R-ratio filter; remove `MAX_RISK_PCT` from stop logic | Median stop ≥ 1.0× ATR; cost < 15% of stop distance |
| 3 | R5 — fix WF config (min_trades, grid contents, plateau metric, 3-symbol requirement) | Harness sweeps the levers that matter |
| 4 | Re-run gate on the *existing* two methods, all 3 symbols | Any bucket positive OOS? |
| 5 | R2 — shift timeframes up one tier if step 4 is still marginal | Re-gate |
| 6 | R4 — swap breakout internals for channel/volatility breakout | Re-gate |
| 7 | Only then: Phase 6 confidence scoring, Phase 7 alerting | — |

Steps 1–3 are configuration and small refactors, not redesigns. They are cheap enough to do before deciding anything larger, and step 4 will tell you whether the signal methods were ever the problem.

---

## 6. Sources

**Academic / peer-reviewed**
- [A novel approach to trading strategy parameter optimization, using double out-of-sample data and walk-forward techniques (arXiv 2602.10785)](https://arxiv.org/html/2602.10785) — intraday BTC/ETH/BNB, timeframe vs cost, 81-window sweep
- [Bailey & López de Prado, The Deflated Sharpe Ratio (SSRN 2460551)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Bailey et al., Statistical Overfitting and Backtest Performance (LBL)](https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf)
- [Combinatorial Purged Cross-Validation insights (SSRN 4778909)](https://www.scribd.com/document/725401650/SSRN-id4778909)
- [The probability of backtest overfitting](https://www.researchgate.net/publication/318600389_The_probability_of_backtest_overfitting)
- [Momentum Trading in Cryptocurrencies: Time-Series vs Cross-Sectional](https://www.researchgate.net/publication/406476873_Momentum_Trading_in_Cryptocurrencies_A_Comparative_Study_of_Time-Series_and_Cross-Sectional_Strategies)
- [Cryptocurrency factor momentum (Quantitative Finance 23:12)](https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2269999)
- [Trading Games: Beating Passive Strategies in the Bullish Crypto Market (J. Futures Markets, 2025)](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70018)

**Practitioner research**
- [Quantt — Crypto Quant Strategies 2026: What Actually Works](https://www.quantt.co.uk/resources/crypto-quant-strategies-2026)
- [unravel.finance — Cross-Sectional Alpha Factors in Crypto](https://blog.unravel.finance/p/cross-sectional-alpha-factors-in)
- [QuantPedia — Time-Series vs Cross-Sectional Momentum/Value/Carry](https://quantpedia.com/time-series-vs-cross-sectional-implementation-of-momentum-value-and-carry-strat/)
- [Volatility Box — Volatility-Adjusted Stop Losses (ATR/Chandelier/Keltner)](https://volatilitybox.com/research/volatility-adjusted-stop-losses/)
- [Traders Second Brain — Stop Loss Placement: ATR vs Structure vs Percentage](https://traderssecondbrain.com/guides/stop-loss-placement-methods)
- [QuantStrategy — Backtesting Chart Patterns with Bulkowski's Methods](https://quantstrategy.io/blog/how-to-backtest-chart-patterns-using-bulkowskis-statistical/)

**Benchmarks / market context** *(vendor sources — indicative only)*
- [Altrady — Are AI Crypto Trading Bots Profitable in 2026?](https://www.altrady.com/blog/crypto-bots/are-ai-crypto-trading-bots-profitable-2026)
- [Altrady — Sharpe and Sortino for Crypto](https://www.altrady.com/blog/risk-management/sharpe-ratio-sortino-ratio-crypto)
- [Cryptorobot — Sharpe Ratio Explained](https://cryptorobot.ai/performance/sharpe)
- [Bitget — Freqtrade guide 2026 (live-vs-backtest degradation)](https://www.bitget.com/academy/freqtrade-cryptocurrency-trading-bot-america-2026-comprehensive-guide-to-automated-trading)
- [Blockchain Council — Backtesting AI Crypto Strategies: overfitting, lookahead, leakage](https://www.blockchain-council.org/cryptocurrency/backtesting-ai-crypto-trading-strategies-avoiding-overfitting-lookahead-bias-data-leakage/)
- [chartscout — Chart patterns and volume analysis in crypto (2026)](https://chartscout.io/chart-patterns-and-volume-analysis)
- [Binance maker/taker fee reference](https://binancemakertakerfee.org/) · [TradersUnion — Binance Futures fees](https://tradersunion.com/brokers/crypto/view/binance/futures-fees/)

**Internal / measured this session**
- `.claude/PRPs/reports/phase5-breakout-strategy-investigation.report.html`
- `.claude/PRPs/prds/l1-crypto-signal-bot-mvp.prd.md`, `.claude/plans/l1-crypto-signal-bot-mvp.plan.md`
- Source review: `src/trading_bot/{config,regime/classifier,signals/*,backtest/*}.py`
- New measurements against `data/ohlcv.db` (3 symbols × 3 timeframes, 2022-12-31 → 2026-07-23): bar-range distributions, unconditional adverse-excursion probabilities, realized stop/target distributions, regime occupancy, and a 3-symbol default-parameter backtest via `cli.py backtest`
