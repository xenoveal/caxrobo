# A Method That Can Actually Clear a Gate

**Date:** 2026-07-27 · **Status:** proposal, nothing built · **Companion:** `FINDINGS.md`

The brute-force search is documented in `FINDINGS.md`. This document answers the
next question: *what method could work, given what we now know does not?*

Research used built-in `WebSearch`/`WebFetch` (firecrawl/exa are not configured
here), four parallel researchers, and — more importantly — four new measurements
taken directly from Binance and from this repo's own data. Where a claim is
measured rather than cited, it says so.

---

## 1. The diagnosis, in one line

**We were searching for alpha in the one place we could measure, and the target
was already sitting in beta, which we never measured.**

Three numbers establish it.

| Measurement | Value | Source |
|---|---|---|
| Mean Sharpe across 555 gated universe-wide trials | **−0.0055** | measured, our own results |
| Best trial vs. what pure noise predicts from 902 trials | **+0.712 vs +1.27** | measured, False Strategy Theorem |
| **BTC buy-and-hold, same window (2023-01→2026-07)** | **Sharpe 1.01** | measured, our own 1d bars |

The middle row is the one that should end the search programme: **our best
strategy was worse than chance.** Not a small edge — below the expected maximum
of a null distribution. The bottom row is the one that should redirect it: the
PRD's primary gate, Sharpe ≥ 1.0, was satisfied over the identical period by
holding one asset and writing no code.

### The search space was 14 questions, not 25,240

Inverting the False Strategy Theorem on our own trial distribution
(σ = 0.4122 across 555 gated trials), the observed maximum of +0.712 corresponds
to **N_eff ≈ 14 independent trials.**

| N_eff | E[max Sharpe] under zero edge |
|---|---|
| 5 | +0.49 |
| **14** | **+0.71** ← observed |
| 50 | +0.93 |
| 902 | +1.32 |
| 25,240 | +1.68 |

1,262 parameter combinations across 20 near-one-factor crypto symbols were
~99.9% redundant. **This is why "brute-force harder" cannot work:** more combos
add trials to the multiple-testing bill without adding questions, and the
required significance bar grows non-linearly in trials
([Harvey & Liu 2015](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489)).
Scaling the search is provably self-defeating.

### We were ~3 orders of magnitude over our statistical budget

Minimum Backtest Length: MinBTL < 2·ln(N)/(SR*)² years
([Bailey, Borwein, López de Prado & Zhu, *AMS Notices* 2014](https://www.ams.org/notices/201405/rnoti-p458.pdf)).

| Data | Target Sharpe | Affordable independent trials |
|---|---|---|
| 3.5 years | 1.0 | **≈ 6** |
| 3.5 years | 0.7 | ≈ 2 |

We spent 25,240. The authors' own worked example: with five years of data, no
more than ~45 configurations should be tried, or you are near-guaranteed an
in-sample Sharpe of 1 with an expected out-of-sample Sharpe of 0. **That is
precisely what happened, and it was predetermined before any strategy was
written.**

### Why TRAIN/SELECT agreement was not a safeguard

TRAIN (2023-08→2025-07) and SELECT (2025-07→2026-01) are **contiguous**: same
macro regime, same volatility level, same cross-sectional correlation structure.
Agreement between them is largely evidence of *regime persistence*, not
generalisation. Time, not row count, is the unit of information — 3.5 years of
crypto is a few regimes, not 30,000 hourly observations.

Worse, SELECT was a *second fitting stage*, not a test: candidates were chosen
using it. A filter that sees 5 survivors cannot repay a selection bias generated
by 25,240 trials. And the ranking **inversion** on holdout is the textbook
signature of overfitting rather than bad luck — overfitted strategies
*systematically* underperform out of sample
([PBO, Bailey et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)).
A controlled synthetic study finds walk-forward the **worst** of four methods at
false-discovery prevention ([Arian, Norouzi M. & Seco, *Knowledge-Based Systems* 305, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)).

DSR was ≈0.0001 throughout and was the only metric that took 25,240 as an input.
It was right in advance. It was reported and ignored.

---

## 2. The target, restated honestly

**The two constraints conflict.** 45% return with max drawdown ≤ 25% implies
Sharpe ≈ 1.8, not 1.0. At Sharpe 1.0 and 25% volatility you get ~25% return, and
a 25% drawdown is roughly the *expected* worst case rather than a safe ceiling.
The PRD asks for ~1.8 while writing 1.0.

For scale, from research: the SG CTA Index has managed **Sharpe 0.61 since 2000**;
BTC buy-and-hold is 0.70 since 2013; Medallion — the best documented record in
history — is >2.0 and closed to outside capital
([CFM](https://www.cfm.com/wp-content/uploads/2022/12/39-A-Good-Time-for-Trend-Following-FINAL.pdf),
[Cornell Capital](https://www.cornell-capital.com/blog/2020/02/medallion-fund-the-ultimate-counterexample.html)).
**Sustained Sharpe 1.0 net is a top-tier professional outcome, not a floor.**

### On "3x per year is easy"

Both halves are true, and the measured data separates them. From our own 1d bars:

| | 2023 | 2024 | 2025 | 2026 YTD | Full-sample |
|---|---|---|---|---|---|
| BTC return | +154.7% | +111.5% | −7.4% | −26.1% | CAGR 46.9%, vol 46.5%, **Sharpe 1.01**, maxDD 53.0% |
| ETH return | +90.3% | +41.7% | −11.6% | −34.4% | CAGR 14.9%, vol 63.0%, Sharpe 0.24, maxDD 67.6% |
| SOL return | **+920.1%** | +71.9% | −35.9% | −39.4% | CAGR 77.2%, vol 85.2%, Sharpe 0.91, maxDD 76.3% |

**Right:** SOL nearly 10x'd in 2023 and BTC more than doubled twice. Beta
delivered far more than 45% with no edge at all. Research confirms BTC beat +45%
in **six of nine** complete calendar years 2017–2025
([ChartRow](https://chartrow.com/bitcoin/returns)).

**Where the inference breaks:** 2025 and 2026 were negative for all three coins.
"3x *every* year" through this window requires a genuine short or market-neutral
capability, which is a far rarer claim than directional luck. And no documented
case of anyone sustaining 200%+ for three consecutive years was found; 3x
compounded for a decade is 59,049× — capacity-constrained by construction.

**The base rates matter because the reference is a sample of one, selected on
success.** Of Brazilian individuals who day-traded equity futures for >300 days,
**97% lost money; 1.1% earned more than minimum wage**, with no evidence of
learning from experience
([Chague, De-Losso & Giovannetti](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101)).
In Taiwan, **<1% of ~450,000 annual day traders were predictably profitable net
of fees** ([Barber, Lee, Liu & Odean](https://faculty.haas.berkeley.edu/odean/papers/Day%20Traders/Day%20Trade%20040330.pdf)).
Regulators converge: **68–89% of retail leveraged-derivatives accounts lose
money** ([ESMA](https://www.esma.europa.eu/press-news/esma-news/esma-agrees-prohibit-binary-options-restrict-cfds-protect-retail-investors),
[ASIC REP 828](https://download.asic.gov.au/media/tq0he35c/rep828-published-20-january-2026.pdf)).

And the professionals: ~130 crypto hedge funds returned **+40% in 2024 while BTC
returned +122%** — the industry underperformed the asset by ~80 points in a bull
year; market-neutral funds made 18.5%
([Opalesque](https://www.opalesque.com/AMW/846/Opalesque_Roundup_VisionTrack_Composite_Crypto_Index_logs846.html)).
That is the peer group for the thing this bot was trying to be.

---

## 3. What is measurably there

Two structural return sources, both with a nameable counterparty — the property
the 77 searched strategies all lacked.

### 3.1 Beta, sized to a drawdown budget

Measured, our own bars, 2023-01→2026-07:

| | Full-sample maxDD | Max notional for 25% DD | Resulting CAGR |
|---|---|---|---|
| BTC | 53.0% | **47%** of equity | **22.1%** |
| SOL | 76.3% | 33% | 25.3% |
| ETH | 67.6% | 37% | 5.5% |

Sizing BTC to the drawdown ceiling yields ~22% with **zero trials consumed** —
there is nothing fitted, so there is nothing to overfit.

Vol targeting improves this, modestly and genuinely: a liquidity-shock
risk-managed BTC strategy measured **Sharpe 1.21 vs 0.72 buy-and-hold**, with
skewness improving from −1.0 to **+0.41**
([*J. Multinational Financial Management*](https://www.sciencedirect.com/science/article/abs/pii/S1042444X22000019)).
Vol-targeted BTC with stablecoin-flow signals showed Sortino >30–40% above
benchmark ([arXiv 2603.23480](https://arxiv.org/pdf/2603.23480)). Be careful with
magnitudes: documented drawdown improvements are **3–9 percentage points**, not
40. Vol targeting takes −77% to roughly −68%. Only de-levering below 1x reaches
a 25% ceiling.

Leverage discipline, derived from our own measured numbers rather than asserted:
Kelly-optimal f* = μ/σ² = 0.469/0.465² ≈ **2.2x** on our sample, but our sample
is two bull years plus two down years — using the research's longer-run figures
(μ 50.5%, σ 67.0%) gives **f* ≈ 1.1x**. Practitioners use ½ to ¼ Kelly.
**Survivable leverage on long crypto is ≤1x, and ~0.5x for a 25% DD ceiling.**
Against exchanges advertising 125x, and the 10 Oct 2025 event that liquidated
**$19bn across 1.6 million accounts**
([Galaxy/CoinDesk](https://www.coindesk.com/research/market-spotlight-the-19-billion-liquidation-that-shook-crypto)).

### 3.2 Funding carry — real, structural, and currently too thin

Measured directly, 3,910 funding events per symbol, gross annualized carry a
delta-neutral short-perp position would have received:

| Year | BTC | ETH | SOL | % of 8h periods positive (BTC) |
|---|---|---|---|---|
| 2023 | +7.87% | +8.26% | +1.30% | 89.9% |
| 2024 | +11.92% | +12.96% | +13.62% | 91.6% |
| 2025 | **+5.13%** | **+4.93%** | +0.35% | 87.1% |
| 2026 YTD | +1.81% | +0.92% | **−2.70%** | 66.6% |

And the funding stream's volatility, which is the decisive number:

| | Ann. return | **Ann. vol** | Sharpe (upper bound) |
|---|---|---|---|
| BTC 2025 | +5.13% | **0.20%** | 26.3 |
| BTC 2026 YTD | +1.80% | 0.24% | 7.7 |

**Corrections to the PRD, both directions.** Its claim that funding "decayed to
negative in 2025" is **factually wrong** for BTC/ETH — 2025 was ~+5%, positive in
87% of periods. But research found the carry *total return* (funding **plus basis
mark-to-market**) did turn negative in 2025
([arXiv 2510.14435](https://arxiv.org/html/2510.14435v2), single source →
UNVERIFIED), consistent with CME basis compressing from ~25% (Feb 2024) to <10%
(Apr 2025) and briefly negative in Mar 2025
([CF Benchmarks](https://www.cfbenchmarks.com/blog/revisiting-the-bitcoin-basis-how-momentum-sentiment-impact-the-structural-drivers-of-basis-activity)).
**So the PRD reached a defensible conclusion through a wrong measurement** — and
the funding-only Sharpe above is an upper bound precisely because it omits basis.

Why this class is different from everything we searched: **the counterparty is
nameable.** Leveraged longs pay to be long, ~86% of the time across three years.
It is a fee on retail leverage demand — corroborated by the finding that when
Binance cut max leverage 125x→50x in Jul 2021, carry returns *and* Sharpe both
fell ([CMU](https://www.andrew.cmu.edu/user/azj/files/CarryTrade.v1.0.pdf)).
Structural, explicable, persistent. That is the standard of evidence none of the
77 strategies met.

The honest ceiling: the **only** figure measured net of retail costs is
**Sharpe 1.8 on a 2020–2022 sample, with deviations decaying ~11%/yr**
([arXiv 2212.06888](https://arxiv.org/html/2212.06888v5)). Extrapolated to 2026
that is ~1.15 — at the target and inside noise of missing it. Published gross
Sharpes of 4–11 are gross of everything and should not be planned against. The
BIS documents the fat tail the low volatility hides: crypto carry ranged from
**below −50% (FTX, Nov 2022) to above +45% (Jan 2024)**, and elevated carry
*predicts subsequent crashes* — it is a crash-risk premium, not free money
([BIS WP 1087](https://www.bis.org/publ/work1087.htm)).

Cost arithmetic makes it a hold-forever trade: a delta-neutral round trip is
**0.30% all-taker / 0.24% all-maker** of notional, so against 2026's +1.81% gross
one round trip consumes **10–17% of a full year**. Weekly symbol rotation would
cost ~12.5%/yr — more than the entire gross carry in every year except 2024.

**Verdict: switch it on when funding exceeds the fee hurdle, flat otherwise. At
2026 YTD levels it is OFF.**

---

## 4. What is measurably not there

**Cross-sectional factors are inaccessible on this instrument set.** The
canonical result ([Liu, Tsyvinski & Wu, *Journal of Finance* 77(2), 2022](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119);
[NBER w25882](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf))
uses **1,707 coins** screened at market cap >$1m, 2014–2018, ~**356 live at any
time**. The size factor works by *"longs the smallest coins and shorts the
largest coins"*, generating **>3% excess weekly returns**, with the authors' own
caveat that this depends on *"trading costs and the feasibility of short
selling."*

So it is a **micro-cap illiquidity premium**, and our implementation was
structurally incapable of capturing it three times over: we used 20 coins where
the factor needs ~356; we used the 20 *most liquid* perps, which are the factor's
**short leg**; and the long leg consists of coins with no perpetual market at
all. **This retires "expand the universe" as a fix** — 20→50 liquid perps would
not help, because the return source lives where USDT-M perps structurally do not
go. PRD Option B was never viable, and now we know why rather than merely that.

**Positioning data cannot be backtested at all.** Binance retains only **30 days**
on `openInterestHist`, `topLongShortAccountRatio`, `topLongShortPositionRatio`,
`globalLongShortAccountRatio`, `takerlongshortRatio`
([API docs](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)),
and there is no historical liquidation endpoint. Adopting a signal class we
cannot validate would repeat the mistake a third time. Paid history exists
(Coinglass $29–699/mo, Kaiko ~$9.5k–55k/yr) but only for classes rated
infeasible on other grounds. Research found **no** peer-reviewed evidence that
ΔOI, liquidations, or long/short ratio predict returns at 1h–1d.

**Order-flow signals die in 1–10 seconds.** Measured information half-life is
sub-second to seconds; a Binance-perp taker strategy on 3-second returns nets
**0.13–7.00% annualized** ([arXiv 2602.00776](https://arxiv.org/html/2602.00776v1));
10s mid returns (<10bps) are smaller than two-leg fees (~20bps). Binance
publishes no L3/queue data. **There is no version of this reaching a 1h bar.**

**One cheap thing worth testing, not deploying:** VPIN on a volume clock with 24h
holds is buildable from free `aggTrades` and reported OOS Sharpe **0.88** — below
target, BTC-only, and its own author reports net alpha **−15.6bps in 2026 YTD**.
Single unreplicated blog source. One clean replication, one trial.

---

## 5. The proposed method

**Name:** *Structural Carry & Risk-Managed Beta (SCRB)*

Not a signal bot. A **two-sleeve allocator** whose return sources are both
structural, plus the one exit finding our search actually earned. Designed
around a trial budget of **6**, not a search.

### Sleeve A — Risk-managed beta (core, ~80% of risk)

Long BTC (optionally ETH), notional sized so that **forecast drawdown ≤ the
budget**, using inverse-volatility targeting with a hard cap at **0.5x** notional.
Rules, all pre-committed, none fitted:

- Target volatility 20%; position = min(0.5, target_vol / realized_vol(30d)).
- **Trend gate for crash protection only:** flat when price is below its 200-day
  moving average. This is not a signal — it is a documented tail-reducer, and it
  is the single most-replicated result in systematic investing.
- **Trailing ATR exit**, `TRAIL_ATR_MULTIPLE` 2.0–3.0, independent of entry stop.
  This is the *one* durable finding from our own search: exit factor moves Sharpe
  +0.19/+0.64 across splits, directionally consistent on **35 of 40
  symbol-splits**, while the entry factor was sign-unstable (see `FINDINGS.md` §2.1).

Expected: **~20–25% CAGR at ~25% max DD**, Sharpe ~1.0. Trials consumed: **0–1**.

### Sleeve B — Conditional funding carry (satellite, ~20% of risk)

Delta-neutral long spot / short perp, **gated on the carry exceeding the cost
hurdle**:

- Enter when trailing 30-day annualized funding > **4%** (≈ 20x the 0.24% maker
  round-trip, i.e. amortised over a ≥2-week hold).
- Exit when it falls below **1.5%**, or on a basis-stress trigger.
- **Requires Binance Portfolio Margin** — with separate spot and futures wallets,
  an adverse spot rally margin-calls the perp leg while the offsetting profit
  sits in a wallet that cannot rescue it. Liquidation below uniMMR 1.05. Verify
  eligibility in-account first; research could not confirm the threshold.
- **Unlevered, or ≤2x.** The low funding volatility does *not* price the basis
  tail (BIS: −50% to +45%).

Expected: **+3–5% on deployed capital in a normal year, 0% when gated off** —
and it is gated off at 2026 YTD levels. Trials consumed: **1–2** (the two
thresholds).

### Sleeve C — deliberately empty

No pattern recognition, no oscillators, no chart geometry, no ML on price. We
measured that space at mean Sharpe −0.0055 across 555 trials with a maximum
*below* the null. **Leaving it empty is a result, not an omission.**

### What this does and does not achieve

| | Expected | Gate |
|---|---|---|
| Sharpe | ~1.0–1.2 | ≥1.0 ✓ (marginal) |
| Max DD | ~25% | ≤25% ✓ (at the limit) |
| Annual return | **~22–28%** | 30–45% ✗ |

**It does not reach 30–45% at a 25% drawdown ceiling, and neither does anything
else.** Those constraints jointly imply Sharpe ~1.8, which exceeds the SG CTA
index's 25-year record and the entire crypto fund industry's 2024 showing. The
choice is explicit:

- **Keep DD ≤25%** → plan for ~22–28%. Achievable, mostly without edge.
- **Want 30–45%** → accept 40–55% drawdowns, i.e. ~1x unlevered beta. Achievable,
  historically, in bull years, with a −53% to −76% bill in bad ones.
- **Want both** → requires Sharpe ~1.8 sustained. Not supported by any evidence
  gathered.

---

## 6. Validation protocol — the actual fix

The strategy change matters less than this. Our failure was methodological.

1. **Cap the trial budget in writing, before code.** 3.5 years at a Sharpe-1.0
   target affords **≈6 independent trials**. Every grid, universe variation, and
   "just one tweak" re-run debits it.
2. **Pre-register each hypothesis.** One paragraph naming the *counterparty and
   why they must trade*, the falsifiable prediction, the metric, the pass
   threshold. If you cannot name who is losing the money and why they keep doing
   it, do not spend a trial. Both SCRB sleeves pass this test; none of the 77
   searched strategies did.
3. **Make DSR and PBO the acceptance gate, failing closed.** Ours was 0.0001 and
   candidates advanced anyway. Reject at PBO > 0.5.
4. **Report every candidate as distance above the null's E[max]**, never as a raw
   Sharpe. One line of arithmetic would have flagged all five candidates.
5. **Stratify splits by regime, not by date.** Require a candidate to hold within
   each volatility tercile and each funding-sign regime *separately*. This repo
   already has a regime classifier. Adjacent-window agreement is worth little.
6. **Then** add CPCV with purge/embargo sized to the max trade horizon — but note
   its documented limits: it cannot extrapolate to an unseen regime, it is
   described as impractical for short-history assets **naming crypto explicitly**,
   and a 2026 preprint argues purged CV and triple-barrier under-control
   *selection* bias specifically ([arXiv 2604.15531](https://arxiv.org/abs/2604.15531)).
   CPCV would have fixed our false-discovery accounting; it would **not** have
   predicted the regime break.
7. **Always include naive baselines** — buy-and-hold, persistence, and the
   incumbent. Our incumbent *won* the holdout. Take that seriously.
8. **Measure total-return carry on our own data** before deploying Sleeve B:
   funding + basis mark-to-market, 2023–2026, from free spot and perp klines.
   This settles the 2025 question with our own numbers rather than a
   single-source citation, and it is about a day's work.
9. **Audit execution conventions separately.** Our prefix-invariance canary tests
   *prefix* invariance, not execution-convention leakage; same-bar-open fills
   using post-open information are a documented inflation source a canary will
   not catch ([arXiv 2605.23959](https://arxiv.org/abs/2605.23959)). We fill at
   bar close, which I believe is clean, but it is unproven.
10. **Start persisting the 30-day positioning endpoints now.** Zero cost; in 6–12
    months they become testable. This is the only action that *buys* future
    statistical budget.

Calibration anchors, so expectations stay honest: a hypothesis-first study — 5
pre-specified economic hypotheses, 10 years, 34 test periods, honest costs —
delivered **Sharpe 0.33** on US equities, itself not distinguishable from zero
([arXiv 2512.12924](https://arxiv.org/html/2512.12924v1)). Cost-aware,
properly-validated crypto ML lands at **Sharpe 0.5–1.0**, and several 2025–2026
papers report ML failing to beat naive persistence, one concluding crypto series
resemble **Brownian noise**
([*Algorithms* 19(2):101](https://www.mdpi.com/1999-4893/19/2/101),
[PMC12571449](https://pmc.ncbi.nlm.nih.gov/articles/PMC12571449/)).

---

## 7. Confidence and gaps

**High confidence (measured by us, reproducible):** the null-distribution
analysis and N_eff ≈ 14; funding levels and funding-stream volatility; per-year
beta, volatility and drawdown; the 25%-DD sizing arithmetic; the 1,707-coin
factor universe (extracted from the paper).

**High confidence (cross-verified in literature):** MinBTL and multiple-testing
arithmetic (two independent literatures agree); order-flow horizons; positioning
data unavailability (official API docs); retail loss base rates (full-population
administrative data).

**Medium confidence:** that carry *total return* was negative in 2025 — single
source, load-bearing for the PRD correction, and reproducible on our own data
(step 8). Vol targeting's benefit — direction consistent across four studies,
magnitudes vary widely. Net-of-cost carry Sharpe 1.8 — one paper, 2020–2022,
decaying.

**Explicit gaps:**
- **Max drawdown of a delta-neutral carry book is not reported anywhere.** The
  single most important number for sizing Sleeve B. Must be simulated on our own
  basis series.
- **No source stress-tests a specific delta-neutral leverage** (3x/5x/10x)
  against a historical basis blowout. Simulate, do not cite.
- **Portfolio Margin eligibility threshold unverified** — a hard gate on Sleeve B.
- **Spot borrow cost unquantified** (Binance margin rates are hourly and dynamic).
  Note the structural point: *unlevered* cash-and-carry has no borrow cost.
- **No Binance quarterly-vs-perp calendar-basis backtest exists** in the
  literature — a genuine gap we could fill.
- **Survivorship bias in our own universe:** `universe.py` picked 20 symbols
  known to have survived to 2026.
- Several 2026 arXiv preprints cited here are unrefereed and flagged inline; the
  load-bearing methodology papers (2014–2015) are peer-reviewed.

---

## 8. Recommendation

1. **Do not promote anything from the search.** Nothing cleared the gate; the
   best result was below chance.
2. **Fix the pooled-Sharpe metric** (`FINDINGS.md` §3.1) — the PRD's primary gate
   currently does not measure what it intends to.
3. **Reframe the product.** The honest deliverable is not a signal bot that beats
   the market; it is a **risk-managed allocator** that captures beta at a
   survivable drawdown and harvests funding when it is rich. That is a real
   product with a real expected return of ~22–28%.
4. **Decide the target explicitly.** 30–45% and ≤25% DD cannot both hold. This is
   the owner's call, and it should be made in writing before any more building.
5. **Spend the next trial budget on the two SCRB thresholds and the total-return
   carry measurement — six trials, not 25,000.**
