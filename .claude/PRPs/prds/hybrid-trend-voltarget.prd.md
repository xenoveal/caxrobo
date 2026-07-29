# Hybrid Trend + Volatility-Targeted Signal Bot — v1.0

**Version**: 1.0 · **Generated**: 2026-07-26 · **Status**: DRAFT — needs validation
**Supersedes**: `.claude/PRPs/prds/initial-requirements.md` (and the deleted `l1-crypto-signal-bot-mvp.prd.md` it referenced)
**Sources of truth**: `.claude/PRPs/reports/strategy-options-30-50pct.md` (strategy selection) · `.claude/PRPs/reports/market-research-capability-benchmark.md` (internal measurements)
**Selected strategy**: **Option D — Hybrid: A-core (regime-gated Donchian trend) + volatility-targeted sizing + optional C sleeve (funding carry) later**

> **Not a profit projection.** Every figure here is a published backtest or an internally measured statistic. Budget 20–40% live degradation versus backtest. SEC/CFTC/NASAA flag "consistently profitable AI trading bot" claims as the dominant crypto-fraud pattern; nothing in this document is a promise of returns.

---

## Problem Statement

A single retail trader (the product's only user) built a signal bot to stop missing crypto entries that print while they are not watching the screen. The bot works — 7,500 lines, tested, walk-forward harness and all — but the **product definition it was built to is mathematically infeasible**: measured across 3 symbols and ~2,740 trades, every signal method in every regime bucket loses money, and the loss rate is statistically indistinguishable from pure price noise. The cost of not solving it is total: the system cannot be turned on, and every further phase built on the current spec (Phase 6 confidence scoring, Phase 7 alerting) compounds sunk cost onto a foundation that cannot produce positive expectancy.

## Evidence

All measured internally against `data/ohlcv.db` (3 symbols × 15m/1h/4h, 2022-12-31 → 2026-07-23) — see `market-research-capability-benchmark.md`:

- **The permitted stop band sits entirely inside the noise floor.** Median 15m bar range is 0.251% (BTC) / 0.339% (ETH) / **0.520% (SOL)**. `MAX_RISK_PCT = 0.005` means one median SOL bar equals the entire maximum permitted stop distance.
- **Noise alone explains the failure.** Unconditional adverse-excursion probability within the 96-bar hold window: 91.2% (BTC @0.20%), 93.2% (ETH), 95.4% (SOL). Even at the widest stop the band permits (0.5%), noise hits it 78% of the time on BTC. The Phase 5 report's observed 90.6% stop-out rate is *inside* the noise band — no residual signal is left to measure.
- **Gross expectancy is ≈ 0, so no edge was ever detected**: BTC −0.007%, ETH +0.028%, SOL +0.010% per trade gross of costs.
- **Costs consume 58–68% of the risk unit** (`c = cost/risk`). Since break-even win rate is `p* = (1+c)/(1+m)`, the current spec demands ~0.58 R of gross edge per trade; published systematic trend strategies deliver **0.05–0.15 R**. The spec asks for 4–10× more edge than real strategies produce.
- **Failure is not symbol- or method-specific**: BTC PF 0.59 / ETH 0.71 / SOL 0.68; every bucket with n ≥ 100 negative, *including* the structurally unrelated mean-reversion fade method. Two unrelated methods failing identically indicts the component they share — the risk model — not either method.
- **Timeframe is on the wrong side of the cost frontier**: mean Sharpe across 81 walk-forward configs is **−12.71 at 1-minute** vs **+0.791 at 60-minute** ([arXiv 2602.10785](https://arxiv.org/html/2602.10785)).
- **One component is validated and kept**: the regime classifier's occupancy is healthy and non-degenerate (trending 36–43% / ranging 43–49% / extreme-vol ~12% / uncertain 2.7%), Wilder ADX/ATR is correct and trailing-window-only.

## Proposed Solution

Rebuild the strategy definition around **Option D**: a regime-gated **Donchian channel breakout** (20/55, ADX>25 confirmation, 55-mid trend filter) as the trending-regime signal engine, with **ATR-scaled stops** (`k = 1.5 × ATR(4H)`, derived from the cost constraint and then frozen) and an **R:R *ratio* floor (≥1.5)** replacing the absolute risk/reward band; the whole stack shifted up one tier to **1D regime / 4H setup / 1H trigger**; and a **volatility-targeting sizing layer** that converts a Sharpe-1.0-ish, 12–20% unlevered base strategy into the 30–45% annualized target by publishing `position = (target_vol / realized_vol) × equity` guidance on each alert rather than by demanding more from the signal.

Why this shape over the alternatives: Option A alone is the smallest diff and reuses the healthy regime layer but tops out around 12–20% unlevered. Option B (universe expansion, highest published Sharpe >1.5) requires backfilling 10–20 symbols and a daily portfolio loop — real work, deferred until we know A-core's Sharpe. Option C (funding carry) has decayed to negative in 2025 and needs funding-rate ingestion the DB lacks. Option D takes A's signal engine and B's *sizing* mathematics — inverse-vol weighting lifted Sharpe 0.99 → 1.54 and cut max DD −30.8% → −13.8% in comparable portfolios ([Concretum](https://concretumgroup.com/position-sizing-in-trend-following-comparing-volatility-targeting-volatility-parity-and-pyramiding/)) — which is the only combination that satisfies moderate risk, 30–50% return, and non-overfit simultaneously.

**The conceptual correction the whole project turns on**: the old spec conflated *stop distance* (a market-structure question governed by volatility) with *account risk* (a position-sizing question). It applied a 0.5% sizing budget as a stop-distance rule. Stop distance is now volatility-derived; account risk is discharged by the human's position size, informed by the vol-target guidance the bot publishes.

## Key Hypothesis

We believe a **regime-gated Donchian trend engine with ATR-scaled stops on 1D/4H/1H tiers, plus published volatility-target sizing guidance**, will **produce a positive-expectancy, Sharpe ≥ 1.0 signal stream whose returns the user can scale to 30–45% annualized at ≤25% drawdown** for **one retail discretionary crypto trader executing alerts manually**.

We'll know we're right when **pooled 3-symbol walk-forward, on the untouched out-of-sample holdout, nets Sharpe ≥ 1.0 with positive expectancy on all three symbols, ≥30 trades per test fold, a Deflated Sharpe Ratio significant at p < 0.05, and a measured cost ratio `c ≤ 0.10`.**

We'll know we're **wrong** when the OOS holdout fails any of those gates — at which point the escalation path is Option B (universe expansion), not further parameter tuning.

## What We're NOT Building

- **Automated order execution.** Alert-and-execute stays manual. The founding user problem is *missing* entries, not placing them; and manual execution keeps the sizing decision (hence leverage and account risk) with the human, which is exactly where a 30–50% target belongs.
- **Chart-pattern geometry (triangle / flag / head-and-shoulders).** Retired, not paused. Triangle + flag were ~98% of trade volume and both lose on all three symbols; H&S never accumulated a judgeable sample (n=5–27) and likely never will. `signals/patterns.py` and `signals/pivots.py` leave the active path.
- **The 15-minute trigger tier.** It maximizes the founding user's failure mode (a human cannot act on a 15m alert reliably) *and* sits on the wrong side of the cost frontier.
- **`MAX_RISK_PCT` / `MIN_REWARD_PCT` as absolute stop and target constraints.** Deleted from the stop logic. They are the adverse-selection machine.
- **Ensemble / blended signal voting.** One method per regime, dispatched by the classifier. Keeps attribution legible and the degrees of freedom countable.
- **Option B universe expansion and Option C funding carry, in v1.** Both are explicitly sequenced *after* the A-core gate result, and C additionally needs funding-rate ingestion and a product-shape decision (continuous two-leg positions do not fit alert-and-execute).
- **Any parameter sweep of the cost model.** `FEE_PCT`/`SLIPPAGE_PCT`/funding are set pessimistically once and frozen; optimistic cost assumptions are a silent overfitting channel worth ~5%/yr of phantom return at current turnover.
- **Sweeping the stop multiple `k`.** Derived from the `c ≤ 0.10` constraint, then frozen. Fitting `k` on returns is what the Phase 5 buffer sweep already did, and it consumed a degree of freedom on BTCUSDT 2023–2026.

## Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| **Primary — Sharpe (net, OOS)** | **≥ 1.0** | Pooled 3-symbol walk-forward, one-shot OOS holdout, net of frozen fee+slip+funding; time-indexed equity curve (new `metrics.sharpe`) |
| Max drawdown | ≤ 25% at the 25% vol target | Equity-curve peak-to-trough on the vol-targeted equity series |
| Deflated Sharpe Ratio | Significant at p < 0.05 | Bailey & López de Prado, correcting for the full grid × fold search count |
| Cost ratio `c = cost / risk` | ≤ 0.10 | Median realized stop distance vs frozen round-trip cost, asserted per symbol in the gate |
| Cross-symbol consistency | Positive OOS expectancy on **all 3** symbols | Mandatory 3-symbol gate (not an average) |
| Sample adequacy | ≥ 30 trades per 60-day test fold | Pooled folds; `WF_MIN_TRADES` raised 5 → 30 |
| Derived return (outcome, **not a gate**) | 12–20% unlevered → 30–45% at 25% vol target | Annualized from the OOS equity curve; reported, never optimized toward |
| Median stop distance | ≥ 1.0 × ATR(setup TF) | Trade-list distribution check — proves R1 actually landed |
| Time-to-actionability | Alert lands within one 1H bar close of the trigger | Poller/scan timing log |

## Open Questions

- [ ] **Vol-target parameterization**: 25% annualized is taken from the strategy doc's midpoint. Which realized-vol estimator (EWMA vs simple N-day close-to-close), what lookback, and what hard leverage cap does the user want? A vol target without a leverage cap can silently demand 5×+ in quiet regimes.
- [ ] **`c ≤ 0.10` depends on execution style.** At `k = 1.5×ATR(1H)` all-taker, BTC fails narrowly (`c` = 14.9%). Shifting to 4H setup bars fixes it for free (`c` = 7.1/5.2/3.7% even all-taker). Does the user commit to maker-side entries, or do we lean entirely on the 4H tier? This changes the alert's order-type instruction.
- [ ] **Funding cost model.** At multi-day holds funding stops being negligible and the old PRD already promised "net of realistic fees/funding" while the engine models only fee+slip. Do we ingest real funding history (needed for Option C anyway) or apply a frozen pessimistic constant in v1?
- [ ] **1-minute exit resolution.** The benchmark recommends resolving exits on 1m bars. With 1.5×ATR(4H) stops (~1.3–2.5%) the conservative same-bar rule is invoked far less often than at 0.2% stops — is 1m ingestion (a large backfill: ~1.9M bars/symbol/year) still worth it in v1, or does the wider stop retire the concern?
- [ ] **Donchian parameter provenance.** 20/55 is canonical (Turtle lineage), not fitted here — but on which timeframe do the channels compute, 4H setup bars or 1D? The strategy doc says "20/55-style entry/exit ... 55-period mid-line as trend filter" without pinning the tier.
- [ ] **Does the fade method survive re-testing?** Its numbers were least-bad (win rate 20–24%, best PFs) and its stop placement is already structurally correct, but it has never been tested under the new risk model. If it fails, the ranging regime becomes flat/no-signal — which is an acceptable outcome, not a failure.
- [ ] **Backfill dependency**: `fapi.binance.com` is TLS-blocked from this machine as of 2026-07-05. 1D bars (and any Option B universe expansion) need that connectivity fixed or an alternate data source.
- [ ] **Trial-log discipline**: the Phase 5 buffer sweep is a consumed degree of freedom on BTCUSDT 2023–2026 and the honest OOS holdout is shrinking. How much of 2026 do we reserve as never-touched?

---

## Users & Context

**Primary User**
- **Who**: One retail discretionary crypto trader (the project owner), trading BTC/ETH/SOL USDT-M perpetuals on Binance. Low-to-moderate risk tolerance, but an explicit 30–50% annualized return goal. Technically capable (reads the code, runs the CLI), not a quant.
- **Current behavior**: Watches charts manually, decides entries by eye, and misses setups that confirm while they are asleep, working, or away. Has a functioning bot they cannot switch on because it has no measured edge.
- **Trigger**: A trending-regime channel breakout confirms on a 1H bar close on one of three symbols — a moment that arrives a few times a week and, unattended, passes unnoticed.
- **Success state**: An alert arrives within a bar of the trigger, carrying entry, ATR-derived stop, target, R:R, and a **suggested position size derived from the volatility target** — enough to place a single order without re-deriving anything, and enough to trust that the setup came from a validated, cost-honest process.

**Job to Be Done**
When **a validated trend setup confirms on a symbol I'm not watching**, I want to **be told immediately, with the stop, target, and size already computed from current volatility**, so I can **place one order in minutes and let a positive-expectancy process compound at a risk level I chose deliberately.**

**Non-Users**
- Anyone wanting a hands-off auto-trading bot — execution stays manual by design.
- Multi-user / SaaS operation — single-tenant, local SQLite, no auth, no multi-account risk isolation.
- Scalpers and sub-hourly traders — the evidence says that end of the spectrum is where costs eat the edge, and v1 deliberately moves away from it.
- Spot-only or non-Binance traders — USDT-M perps on Binance only, since the funding and fee model is venue-specific.
- Anyone needing tax lots, portfolio accounting, or PnL reconciliation — out of scope entirely.

---

## Solution Detail

### Core Capabilities (MoSCoW)

| Priority | Capability | Rationale |
|----------|------------|-----------|
| **Must** | ATR-scaled stops (`k = 1.5 × ATR(setup TF)`, frozen) + R:R **ratio** floor ≥1.5; `MAX_RISK_PCT` removed from stop logic | The single highest-impact change; without it no signal method can clear any gate (evidence §2.1–2.3) |
| **Must** | Honest frozen cost model: taker 0.05%, slip 0.02%/side, funding modeled; `c ≤ 0.10` asserted | Sets the bar the signal must clear; the old 0.04% understated friction by 25% and inflated every backtest |
| **Must** | Tier shift to 1D regime / 4H setup / 1H trigger | Cost-to-edge frontier *and* the founding user problem, solved by one change |
| **Must** | Donchian 20/55 channel breakout as the trending-regime engine (ADX>25 confirm, 55-mid trend filter, ATR-trail / opposite-channel exit) | Deepest published evidence base of any option; canonical parameters, nothing to overfit |
| **Must** | Sharpe-first metrics: time-indexed equity curve, Sharpe, Sortino, equity-based max DD | The north star is now Sharpe ≥ 1.0 / DD ≤ 25%; current `metrics.py` computes neither, and its DD is a sum-of-percentages proxy |
| **Must** | Repaired validation protocol: pooled 3-symbol folds, `WF_MIN_TRADES ≥ 30`, DSR, sweep the levers that matter, replace `plateau_ratio` | Today's harness sweeps the *healthiest* layer (regime thresholds) and validates at n=5 — the FAIL verdict is a verdict on one arbitrary risk config |
| **Must** | Volatility-target sizing layer: `position = (target_vol / realized_vol) × equity`, published per alert with a hard leverage cap | This is the Option D differentiator and the only honest route to 30–45%; also lifts Sortino +30–40% at 20–50% targets |
| **Must** | Keep the regime classifier as-is (defensive suppression, extreme-vol gate) | Measured healthy; the one validated component (§2.6) |
| **Should** | Fade method re-qualified under the new risk model for ranging regimes | Least-bad numbers, structurally correct stop placement — deserves a fair re-test, but the ranging sleeve is droppable |
| **Should** | Alert delivery (Discord) carrying entry/stop/target/R:R/size/regime/confidence | The user-facing payoff, but worthless before the gate passes |
| **Should** | Written trial log of every sweep and its consumed degrees of freedom | The OOS holdout is a depleting resource; untracked sweeps silently spend it |
| **Could** | 1-minute bars for exit resolution | Wider ATR stops may retire the need; large backfill cost |
| **Could** | Option B universe expansion (10–20 perps, inverse-vol portfolio, daily loop) | The escalation path if A-core Sharpe < 1.0 — published Sharpe >1.5, but needs schema-scale backfill |
| **Won't** | Option C funding-carry sleeve | Sharpe decayed 6.45 → 4.06 → negative in 2025; needs funding-rate ingestion and conflicts with alert-and-execute. Revisit post-v1 |
| **Won't** | Automated execution, multi-user, sub-hourly triggers, ensemble voting, cost-parameter sweeps | See "What We're NOT Building" |

### MVP Scope

The minimum that validates the hypothesis is **everything through the gate, and nothing past it**: honest cost model + ATR risk model + tier shift + Donchian engine + Sharpe metrics + repaired pooled walk-forward, run once on the untouched holdout. That produces a single verdict — Sharpe ≥ 1.0 net with cross-symbol consistency, or not.

Sizing (Phase 8) and alerting (Phase 9) are deliberately *after* the gate: they are the product, but they cannot be validated by anything and add no information about whether the strategy works. Building them first is what the previous PRD did.

**Falsification-first ordering**: Phases 1–3 are cheap configuration and small refactors that can be applied to the *existing* two methods. Running the gate on those first tells us whether the signal methods were ever the problem — before spending Phase 5 on a new engine.

### User Flow

Critical path, shortest journey to value:

1. Poller ingests closed 1D/4H/1H bars into SQLite (unchanged infra, extended timeframes).
2. On each **1H** bar close: classify the symbol's **1D** regime → suppress on extreme-vol/uncertain.
3. Trending → evaluate Donchian channel state on **4H** setup bars; ranging → evaluate fade setup.
4. A **1H** bar closes beyond the channel with ADX>25 and price on the correct side of the 55-mid → candidate.
5. Compute stop = `1.5 × ATR(4H)` from entry, target from channel/ATR projection, reject if R:R < 1.5 or `c > 0.10`.
6. Compute realized vol → `size = (0.25 / realized_vol) × equity`, clamped to the leverage cap.
7. Alert fires with entry / stop / target / R:R / suggested size / regime / confidence.
8. Human places one order. Done in minutes, hours after they stopped watching the chart.

---

## Technical Approach

**Feasibility**: **HIGH** for Phases 1–5 (configuration changes and one new signal module against an already-correct spine); **MEDIUM** for Phases 7–8 (new metrics mathematics and a sizing layer with no existing analogue); **MEDIUM-LOW** for the deferred Option B/C sleeves (data ingestion at a new scale).

The codebase's architecture is the asset. What must change is mostly *specification*, which is why this PRD is a pivot and not a rewrite.

**Architecture Notes — what exists and what it costs to change**

- **Keep untouched**: `data/storage.py`, `data/backfill.py`, `data/poller.py`, `exchange/binance_client.py`, `indicators/wilder.py`, `regime/classifier.py`. The Wilder ATR/DI/ADX implementation is measured-correct and trailing-window-only; the classifier's occupancy is healthy. Reuse, don't touch.
- **Timeframe map is the tier-shift chokepoint**: `storage.py:18` hardcodes `TIMEFRAME_MS = {"15m", "1h", "4h"}` — 1D (and any 1m) must be added there, plus `config.TIMEFRAMES` (`config.py:11`) and a backfill run. Every timeframe is fetched natively, never resampled (`config.py:4`), so 1D is a real backfill, gated on the Binance connectivity issue.
- **Tier names are already indirected through config** — `REGIME_TIMEFRAME` (`config.py:19`), `SIGNAL_PATTERN_TIMEFRAME` / `SIGNAL_TRIGGER_TIMEFRAME` (`config.py:28-29`) — and the engine reads them (`engine.py:126-128`). The shift is therefore mostly a config edit **plus** auditing every hardcoded `15m` assumption: `MAX_HOLD_BARS_15M` (`config.py:96`), the `m15` local and `window_15m` slicing (`engine.py:139,236`), and the trigger-bar arithmetic in `setup.py:270-279`.
- **The risk model lives in one function**: `setup.build_signal` (`setup.py:72-186`) computes stop = level ± buffer and applies `max_risk_pct`/`min_reward_pct`. Its own docstring (`setup.py:96-99`) already flags "replacing the level-anchored stop with a structural or ATR stop is deliberately left to Phase 6 tuning." That is now Phase 2. `config.py:69-85` holds the four constants to retire.
- **Donchian slots in cleanly**: `signals/breakout.py:check_breakout` is generic over a `PatternCandidate`'s `breakout_level`/`direction`/`end_ts` — a Donchian channel can produce that same shape, so the trigger mechanics (fresh-crossing test, gap detection, volume grading) are reusable. New `indicators/donchian.py` + `signals/donchian.py`; `signals/patterns.py` and `pivots.py` drop out of the dispatch path in `setup.py:281-293` and `engine.py:164`.
- **Metrics need a new axis, not a patch**: `metrics.py` computes expectancy/PF/max-DD over an unordered `pnl_pct` list, treating every trade as equal-sized (`metrics.py:5-7,57-62`). Sharpe needs a *time-indexed* return series, and vol-targeted sizing breaks the equal-size assumption outright. This is the largest genuinely new piece of mathematics in v1.
- **Walk-forward is well-architected but mis-configured**: `walkforward.py:36-40` sweeps exactly the three regime parameters the benchmark identified as healthiest, at `WF_MIN_TRADES = 5` (`config.py:100`), selecting max-expectancy from 18 combos — noise-fitting by construction. `walk_forward()` is also **per-symbol** (`walkforward.py:93`), so pooling three symbols into each fold is a real signature change, not a flag. `plateau_ratio` (`walkforward.py:176`) divides by a near-zero best value and must be replaced with a positive-neighbour fraction.
- **Costs are one frozen constant to correct**: `config.FEE_PCT = 0.0004` → `0.0005` (`config.py:94`), and `engine.py:124` models `2*(fee+slip)` with no funding term — a `funding_pct_per_day × hold_days` term is a small, local addition.
- **Test suite is the safety net**: 3,300 lines across 10 files. `tests/test_signals.py` (586 lines) will need substantial rewriting since it encodes the retired R:R band; `test_backtest.py`, `test_classifier.py`, `test_wilder.py`, `test_storage.py` should survive largely intact.
- **Uncommitted work in flight**: `engine.py`, `config.py`, `signals/*`, `tests/test_signals.py` are all modified on `fix/phase3-breakout-detection`. That branch's premise (fix breakout detection) is superseded by this PRD — resolve or abandon it before Phase 2 lands.

**Technical Risks**

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Wider ATR stops → far fewer trades → smaller samples → *more* overfitting risk, partly cancelling the benefit | **High** | Pool all 3 symbols into every fold: 3× the sample at **zero** additional degrees of freedom. This is the only free lunch available. Gate on ≥30 trades/fold and refuse to select parameters below it |
| Donchian 20/55 is canonical but still gets *re-fitted* here through repeated gate re-runs | High | Freeze 20/55 and `k=1.5` as derived/canonical, never swept. Keep a written trial log of every sweep. DSR corrects the headline Sharpe for total search count |
| Binance connectivity (`fapi.binance.com` TLS-blocked since 2026-07-05) blocks the 1D backfill the tier shift needs | High | Fix connectivity or source 1D bars elsewhere **before** Phase 4; Phases 1–3 and 7 need no new data and can proceed in parallel |
| Vol-targeted sizing silently demands high leverage in quiet regimes, breaching "moderate risk" | Medium | Hard leverage cap as a first-class config constant; publish *both* the vol-target size and the cap-clamped size in the alert; stress-test the sizing series against the 2024–2025 low-vol stretches |
| Equal-size assumption is baked through `metrics.py` and every existing test; vol-targeting invalidates it | Medium | Build the time-indexed equity curve as a *new* code path (Phase 3) before sizing lands (Phase 8); keep the equal-size path for signal-quality attribution |
| Tier shift misses a hardcoded 15m assumption and produces a subtly lookahead-biased backtest | Medium | Grep-audit every `15m`/`96`/`900_000` literal; assert in the engine that the trigger interval matches `config.SIGNAL_TRIGGER_TIMEFRAME`; keep the existing no-lookahead tests green |
| A-core still fails the gate after all six phases | Medium | That is a *success* of the protocol, not of the project. Pre-committed escalation: Option B universe expansion — **not** more parameter tuning on 3 symbols |
| Funding cost at multi-day holds is guessed rather than measured | Medium | Frozen pessimistic constant in v1, flagged as an assumption in every report; real ingestion arrives with Option C |
| Live degradation of 20–40% versus backtest turns a Sharpe-1.0 backtest into Sharpe 0.6–0.8 live | Medium-High | Budget it explicitly as the *expected* case: require backtest Sharpe ≥ 1.0 knowing live lands lower, and re-validate on a schedule. Never present backtest numbers as expected returns |
| Rewriting `tests/test_signals.py` loses coverage that was catching real bugs | Low-Medium | Port test *intent* case by case (fresh-crossing, gap handling, volume grading are all still valid); only the R:R-band assertions are genuinely retired |

---

## Implementation Phases

<!--
  STATUS: pending | in-progress | complete
  PARALLEL: phases that can run concurrently
  DEPENDS: phases that must complete first
  PRP: link to generated plan file once created
-->

Prior phases carried forward: **old Phase 1 (data infrastructure)** and **old Phase 2 (regime classifier)** are **complete and kept as-is** — they are the validated foundation, not re-work. Old Phase 3 (chart-pattern breakout) is **retired**. Old Phase 4 (fade) is **re-qualified, not rebuilt**. Old Phase 5 (backtest + walk-forward harness) is **kept and repaired**. Old Phases 6–7 (confidence scoring, alerting) are **deferred behind the gate**.

| # | Phase | Description | Status | Parallel | Depends | PRP Plan |
|---|-------|-------------|--------|----------|---------|----------|
| 1 | Data tier extension | Add 1D (and optionally 1m) to `TIMEFRAME_MS` + `config.TIMEFRAMES`; backfill; gap-report green | **complete** | with 2, 3 | - | `.claude/PRPs/plans/v0.2.0/phase1-data-tier-extension.plan.md` |
| 2 | Honest cost & risk model | Fee → 0.05% frozen, funding term, ATR-scaled stop, R:R ratio floor, `MAX_RISK_PCT` removed from stop logic, `c ≤ 0.10` assertion | **complete** (`c ≤ 0.10` NOT met — see Phase 2 detail) | with 1, 3 | - | `.claude/PRPs/plans/v0.2.0/phase2-honest-cost-and-risk-model.plan.md` |
| 3 | Sharpe-first metrics | Time-indexed equity curve, Sharpe, Sortino, equity-based max DD, DSR primitive | **complete** | with 1, 2 | - | `.claude/PRPs/plans/v0.2.0/phase3-sharpe-first-metrics.plan.md` |
| 4 | Tier shift to 1D/4H/1H | Move every tier up one step; audit all hardcoded 15m assumptions; extend hold limit proportionally | **complete** | - | 1, 2 | `.claude/PRPs/plans/v0.2.0/phase4-tier-shift-1d-4h-1h.plan.md` |
| 5 | Donchian trend engine (A-core) | `indicators/donchian.py` + `signals/donchian.py`; 20/55 channel, ADX>25 confirm, 55-mid filter, ATR-trail exit; retire pattern geometry from dispatch | pending | with 6 | 4 | `.claude/PRPs/plans/v0.2.0/phase5-donchian-trend-engine.plan.md` |
| 6 | Fade re-qualification | Re-test the existing ranging-regime fade under the new risk model and tiers; keep or drop on evidence | pending | with 5 | 4 | `.claude/PRPs/plans/v0.2.0/phase6-fade-requalification.plan.md` |
| 7 | Validation protocol repair + **THE GATE** | Pooled 3-symbol folds, `WF_MIN_TRADES ≥ 30`, DSR, positive-neighbour robustness metric, sweep the levers that matter, freeze `k`; run the one-shot OOS gate | pending | - | 3, 5, 6 | `.claude/PRPs/plans/v0.2.0/phase7-validation-protocol-repair-gate.plan.md` |
| 8 | Volatility-target sizing layer | Realized-vol estimator, `position = (target_vol/realized_vol) × equity`, hard leverage cap, per-alert sizing payload, vol-targeted equity curve | pending | - | 7 | - |
| 9 | Alert delivery + trial log | Discord sink carrying entry/stop/target/R:R/size/regime/confidence; written degree-of-freedom log | pending | - | 8 | - |
| 10 | Optional sleeves (deferred) | Option B universe expansion if Sharpe < 1.0; Option C funding carry only after ingestion + product-shape decision | pending | - | 9 | - |

### Phase Details

**Phase 1: Data tier extension**
- **Goal**: Make 1D bars available so the regime tier can move up, without disturbing the working ingestion layer.
- **Scope**: `TIMEFRAME_MS` (`storage.py:18`) and `config.TIMEFRAMES` gain `1d`; backfill 1D for all 3 symbols from `BACKFILL_START`; `gap-report` extended to the new grid. 1m bars only if the Open Question on exit resolution resolves toward "yes."
- **Success signal**: `gap-report` returns zero gaps across every symbol × timeframe combination, including 1D.
- **Blocked by**: Binance TLS connectivity. Flag early — this is the one phase that cannot be worked around locally.
- **STATUS: complete (2026-07-27).** `1d` added to `TIMEFRAMES` and `TIMEFRAME_MS`; a UTC-pinned `1d` poller cron added. Backfilled 1303 1D bars per symbol (2023-01-01 → 2026-07-26, exact day count, no holes). The outage gap in 15m/1h/4h was also filled once connectivity returned. `gap-report` is green on all 12 symbol × timeframe pairs. `1m` deliberately NOT added — Open Question #4 is still unresolved.

**Phase 2: Honest cost & risk model**
- **Goal**: Replace the infeasible risk specification with a volatility-scaled one, and make costs pessimistic and frozen.
- **Scope**: `FEE_PCT` → 0.0005; add a per-day funding term to the engine's cost computation (`engine.py:124`); rewrite `build_signal`'s stop computation (`setup.py:125-171`) to `k × ATR(setup TF)` with `k` **derived** from `c ≤ 0.10` and frozen; replace the absolute band with `rr ≥ RR_FLOOR` (1.5); retire `MAX_RISK_PCT`, `MIN_REWARD_PCT`, `BREAKOUT_STOP_BUFFER_PCT`, `BREAKOUT_MAX_ENTRY_EXTENSION_PCT` from stop logic and document 0.5% as an *account-risk budget* the human discharges via sizing. Port `tests/test_signals.py` intent.
- **Success signal**: Median realized stop distance ≥ 1.0 × ATR(setup TF) on all 3 symbols, and measured `c ≤ 0.10` per symbol — asserted, not eyeballed. Note this phase is *also* the cheap falsification test: re-run the gate on the existing methods and see whether the risk model was the whole problem.
- **STATUS: complete (2026-07-27), but TWO success signals were NOT met — read before starting Phase 4.**
  - `k` is **not** "derived from `c ≤ 0.10` and frozen" as scoped above, and cannot be: clearing that ceiling at current costs requires `risk_pct ≥ 1.4%`, versus observed medians of ~1.18% (breakout) and ~0.40% (fade). `ATR_STOP_MULTIPLE = 1.5` is therefore a conventional unvalidated default, documented as such in `config.py`. Deriving `k` from the fee schedule would fit the stop to costs rather than to volatility; the intended route to the ceiling is this PRD's own Phase 4 tier shift, which raises ATR itself.
  - **`c ≤ 0.10` FAILS on all 3 symbols** (BTC 0.28, ETH 0.18, SOL 0.13). Expected per the plan at the 1H tier, but it is an open item, not a pass.
  - **The falsification test came back negative.** Expectancy got *worse* on all 3 symbols (BTC −0.127% → −0.214%, ETH −0.092% → −0.120%, SOL −0.110% → −0.231%), partly by construction (fee correction + funding term making hidden cost visible) and partly because ATR stops are wider than the old level-anchored ones. **The risk model was not the whole problem.** Per this PRD's own falsification-first ordering, that is a signal to re-examine before spending Phase 5.
  - **Follow-up fix (2026-07-27):** the ratio floor replaced two *absolute* filters with one *dimensionless* one, which made the fade path cost-blind — a setup risking 0.05% to make 0.10% scored `rr` 2.0 while round-trip cost (0.14%) exceeded the entire reward, so no outcome won. The fade path now gates on `risk.atr_stop.net_rr`, the cost-adjusted ratio. The breakout path keeps the gross ratio because `k × ATR` gives its risk a volatility floor. Effect on identical data: fade trades −40%/−35%/−22% (BTC/ETH/SOL); Sharpe −2.77 → −2.12 (BTC) and −1.86 → −1.45 (SOL), but −1.09 → −1.15 (ETH), and expectancy stays negative everywhere. Correct filter, not a rescue.

**Phase 3: Sharpe-first metrics**
- **Goal**: Be able to measure the thing the north star is stated in. Today we cannot.
- **Scope**: Time-indexed return series from the trade list; `sharpe`, `sortino`, annualization; max DD on a real equity curve rather than a sum-of-percentages proxy; a DSR primitive taking (Sharpe, n_trials, skew, kurtosis). Keep the existing equal-size bucket stats for signal-quality attribution.
- **Success signal**: Reproduces the existing metrics on a known trade list, and produces a Sharpe that matches a hand-computed value on a synthetic fixture.
- **STATUS: complete (2026-07-27).** `backtest/equity.py` adds `daily_returns`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `probabilistic_sharpe`, `expected_max_sharpe`, `deflated_sharpe`. Sharpe and Sortino are pinned to literal hand-derived values computed by exact rational arithmetic (the first attempt re-ran the implementation's own estimator and was tautological). It delivered its point: the legacy sum-of-pnl `max_dd` reported impossible values >100% (BTC 154%, SOL 233%) where the real compounded equity drawdown is 80%/92%. Sharpe is deeply negative on all 3 symbols — the honest picture this phase existed to expose.
- **Known gaps (not blocking):** the CLI equity window is labelled with the *requested* span rather than the span the data covers, and two different max-drawdown definitions print under the same `max_dd` label in the same output.

**Phase 4: Tier shift to 1D/4H/1H**
- **Goal**: Move the whole stack to the right side of the cost frontier and make alerts human-executable.
- **Scope**: `REGIME_TIMEFRAME` → `1d`, setup → `4h`, trigger → `1h`; rename and rescale `MAX_HOLD_BARS_15M`; audit every hardcoded 15m literal in `engine.py` and `setup.py`; adjust `REGIME_MIN_BARS` and lookback windows for the coarser bars; re-verify the no-lookahead invariants.
- **Success signal**: Backtest runs clean end-to-end on the new tiers, existing no-lookahead tests stay green, and an assertion confirms the trigger interval matches config.
- **STATUS: complete (2026-07-27).** Report: `.claude/PRPs/reports/phase4-tier-shift-1d-4h-1h-report.md`. All three success signals met: backtest clean on 3/3 symbols, all no-lookahead assertions green with fixture-interval edits only, and `engine._assert_interval` raises on any config/data spacing disagreement.
- **This is the phase that worked.** Sharpe improved on every symbol: BTC **−2.12 → −0.09**, ETH −1.15 → −0.89, SOL −1.45 → −0.56; equity max DD fell 68%→27% (BTC) and 87%→61% (SOL). BTC's `trending/flag` bucket is genuinely positive (+0.145%/trade, PF 1.11, n=47). Still negative overall, so the gate is not cleared — but the cost-frontier thesis is now supported by measurement.
- **`c ≤ 0.10` closed for the breakout sleeve** (what Phase 2 could not do): 0.074 / 0.054 / 0.040 BTC/ETH/SOL, at median stop 1.40–1.46 × ATR(4H). `k = 1.5` was re-derived from the cost constraint alone and **confirmed unchanged**, so zero degrees of freedom were consumed.
- **The fade sleeve is now the sole failure**: its structural stop sits at 0.55–0.68 × ATR, giving c = 0.175 / 0.115 / 0.101 — failing on all three symbols. No tier shift can fix this because the stop is structural by decision. **Phase 6 owns it.**
- **Handoff to Phase 7 — two hard constraints:** (1) trades per 60-day fold are 8.3 / 8.3 / 11.2 per symbol and **27.8 pooled across all three, still below `WF_MIN_TRADES ≥ 30`** — pooling alone does not close the gap; (2) usable history starts 2023-07-27 (207-bar 1D warmup), giving ~1,090 days. Also: `MAX_HOLD_BARS_TRIGGER` does **not** bind (time exits are 4–9%; stops are ~68%), so deprioritize it in the grid.

**Phase 5: Donchian trend engine (A-core)**
- **Goal**: Replace pattern geometry with the deepest-evidenced trending method available.
- **Scope**: `indicators/donchian.py` (upper/lower/mid over N bars, trailing-only); `signals/donchian.py` producing the same candidate shape `check_breakout` already consumes; 20/55 entry/exit with ADX>25 confirmation and the 55-mid as trend filter; exit on opposite-channel touch or ATR trail; dispatch swap in `setup.py` and `engine.py`; `patterns.py`/`pivots.py` leave the active path.
- **Success signal**: Signals generate on all 3 symbols in trending regimes at a plausible rate (a few/week, not a few/day), with no geometry parameters introduced beyond the canonical 20/55.

**Phase 6: Fade re-qualification**
- **Goal**: Give the ranging sleeve a fair test under the new risk model — and drop it without ceremony if it fails.
- **Scope**: Re-point `build_fade_signal` at the ATR stop and R:R-ratio filter; re-test on 4H/1H tiers; decide keep-or-drop on measured OOS evidence.
- **Success signal**: A documented keep/drop decision with numbers behind it. "Ranging regime produces no signals" is an acceptable result.

**Phase 7: Validation protocol repair + THE GATE**
- **Goal**: Make the harness capable of delivering a trustworthy verdict, then take the verdict once.
- **Scope**: Change `walk_forward()` to pool 3 symbols per fold; `WF_MIN_TRADES` 5 → 30; grid contents swapped from regime thresholds to the levers that actually bind (R:R floor, hold limit, Donchian lookback pair if genuinely uncertain) with `k` and the cost model frozen out of the grid; replace `plateau_ratio` with positive-neighbour fraction + absolute neighbour spread; add DSR; make cross-symbol consistency a hard gate. Then run the one-shot OOS holdout **exactly once** and log it.
- **Success signal**: The gate returns a verdict against the Success Metrics table. Pass → Phase 8. Fail → Option B, per the pre-committed escalation. **No re-tuning on a failed gate** — that is how the holdout gets spent.

**Phase 8: Volatility-target sizing layer**
- **Goal**: Convert a Sharpe-1.0 base strategy into the 30–45% target through sizing, not through demanding more of the signal.
- **Scope**: Realized-vol estimator (parameterization per Open Question); `position = (target_vol / realized_vol) × equity` at a 25% target; hard leverage cap as config; vol-targeted equity curve alongside the equal-size one; sizing fields on the signal payload.
- **Success signal**: The vol-targeted equity curve shows the expected profile — Sharpe roughly preserved, annualized return scaled to 30–45%, max DD ≤ 25% — and the realized leverage series never exceeds the cap.

**Phase 9: Alert delivery + trial log**
- **Goal**: Deliver the actual product to the actual user.
- **Scope**: Discord sink attached to the existing `trading_bot` logger interface (the old Phase 1 plan deliberately left this swappable); alert payload with entry/stop/target/R:R/suggested size/regime/confidence; a written trial log recording every sweep and the degrees of freedom it consumed.
- **Success signal**: A live trending setup produces an alert the user can act on within one 1H bar close, containing everything needed to place one order.

**Phase 10: Optional sleeves (deferred)**
- **Goal**: Have a pre-committed path if A-core underdelivers, and an orthogonal diversifier if constraints relax.
- **Scope**: Option B — backfill 10–20 liquid perps, daily-bar portfolio loop, inverse-vol weighting, Donchian lookback ensemble (20/55/100). Option C — funding-rate history ingestion, delta-neutral carry sleeve, and an explicit product-shape decision about continuous two-leg positions.
- **Success signal**: Not scoped in v1. Entered only on a Phase 7 miss (B) or an explicit constraint relaxation (C).

### Parallelism Notes

- **Phases 1, 2, 3 are fully independent** and should run concurrently: Phase 1 touches only the data layer, Phase 2 only the signal/risk layer, Phase 3 only the metrics module. They share no files. Phase 1 is also the one that can be *blocked externally* (Binance connectivity), so starting it first while 2 and 3 proceed protects the critical path.
- **Phase 4 needs both 1 and 2**: 1D bars must exist, and shifting tiers before fixing the risk model would confound the two variables in every subsequent measurement. Change one thing at a time — the previous PRD's failure was diagnosed only because someone finally isolated a shared component.
- **Phases 5 and 6 are parallel** — different signal modules, different regimes, one dispatch registration each. Merge order matters only in `setup.py`'s dispatch table.
- **Phase 7 is a hard barrier.** It needs Phase 3's Sharpe machinery and both signal sleeves settled, and everything downstream is gated on its verdict. Nothing in Phases 8–9 should be started speculatively before it; that is precisely the mistake this PRD is correcting.
- **Phases 8, 9, 10 are strictly serial** — sizing changes the equity mathematics that alerting reports, and the deferred sleeves are decision-gated on everything before them.

---

## Decisions Log

| Decision | Choice | Alternatives | Rationale |
|----------|--------|--------------|-----------|
| Overall strategy shape | **Option D** — A-core + vol-targeted sizing + optional C later | A alone; B (rotational universe); C (carry only) | Only option satisfying moderate risk + 30–50% + non-overfit simultaneously. A alone caps at 12–20% unlevered; B needs universe-scale backfill; C decayed negative in 2025 |
| Stop-distance rule | `k = 1.5 × ATR(setup TF)`, derived from `c ≤ 0.10` then **frozen** | Fixed % band (current); swept `k`; structural stops | The fixed band is inside the measured noise floor. Sweeping `k` is what the Phase 5 buffer sweep did — it consumed a DoF and found nothing, because nothing in the band works |
| Candidate filter | R:R **ratio** ≥ 1.5 | Absolute risk ≤0.5% AND reward ≥0.75% | The absolute band is an adverse-selection machine: it admits only trades whose stops are freakishly tight (median nominal R:R 11:1 — a lottery ticket priced by noise) |
| Meaning of 0.5% | An **account-risk budget** discharged by human position sizing | A stop-distance constraint (current) | The conceptual correction the project turns on: stop distance is market structure, account risk is sizing. Since the bot doesn't size, 0.5% has no business constraining stop distance |
| Timeframe tiers | 1D regime / 4H setup / 1H trigger | 4H/1H/15m (current); 1D/1D/4H | Mean Sharpe +0.791 at 60m vs −12.71 at 1m. Also fixes `c` for free at 4H setup bars (7.1/5.2/3.7% all-taker), *and* serves the founding user need — a human can act on an hourly alert |
| Trending signal method | Donchian 20/55 channel breakout | Chart-pattern geometry (current); Keltner/ATR channel; MA crossover | Trend following is the most-replicated edge across 100+ years and every asset class; 20/55 is canonical, not fitted. Pattern geometry has no peer-reviewed net-of-cost support and lost on all 3 symbols |
| Ranging signal method | Keep fade, but re-qualify under the new risk model | Drop it now; replace it | Least-bad measured numbers (win rate 20–24%, best PFs) and its excursion-extreme stop is already structurally correct. It has never been tested fairly |
| Regime classifier | **Keep unchanged** | Retune; replace; drop | The one measured-healthy component: non-degenerate occupancy, correct trailing-window Wilder math, used defensively to suppress rather than predictively |
| North-star metric | Sharpe ≥ 1.0, max DD ≤ 25%, net of costs | "~1% good day / ≤1% bad day" (current); raw annual return | No market pays a fixed daily rate; compounded, the old framing exceeds every published result by an order of magnitude |
| How 30–50% is reached | **Sizing**: 12–20% unlevered × vol targeting at 25% | Demand it from signal win rate | No published unlevered signal strategy on 3 majors delivers 30–50% at ≤25% DD. Vol targeting also lifts Sortino +30–40% and cut max DD −30.8%→−13.8% in comparable portfolios |
| Cost model | Taker 0.05% + 0.02% slip per side + funding, **frozen forever** | 0.04% (current); swept as a parameter | The old figure understated taker friction 25% (~5%/yr of phantom return at current turnover). Optimistic costs are a silent overfitting channel |
| Statistical power strategy | **Pool 3 symbols** into every fold | Per-symbol folds (current); longer history; finer grid | 3× the sample at zero additional degrees of freedom, and it enforces the cross-symbol consistency gate for free. The only free lunch on the list |
| `WF_MIN_TRADES` | 30 (up from 5) | Keep 5; 50+ | Selecting max-expectancy from 18 combos at n=5 is noise-fitting by construction |
| Robustness metric | Positive-neighbour fraction + absolute neighbour spread | `mean(neighbours)/best` (current) | Dividing by a near-zero `best` is numerically unstable — it reported catastrophic fragility (−11.3) for mildly noisy neighbourhoods |
| Selection-bias correction | Add Deflated Sharpe Ratio | Raw Sharpe; Bonferroni; CPCV | DSR exists for exactly this multiple-testing situation. CPCV is a stronger option and is noted as a future consideration, not v1 scope |
| Gate-failure response | Pre-committed escalation to Option B | Re-tune and re-run | Re-tuning on a failed one-shot holdout spends the only honest OOS data left. Pre-committing removes the temptation |
| Build order | Falsification-first: cheap config fixes (1–3) gate-tested before the new engine (5) | Build Donchian first | Phases 1–3 applied to the *existing* methods answer "was the risk model the whole problem?" for almost no cost. That question is worth answering before a new engine |
| Sizing & alerting position | Strictly **after** the gate | Build alerting in parallel (previous PRD's order) | They are the product but carry zero validation information. Building them first is exactly the sunk cost this pivot exists to stop |
| Option C (carry) | Deferred, `Won't` for v1 | Include as a v1 sleeve | Sharpe 6.45 → 4.06 → **negative in 2025**; needs funding-rate ingestion and conflicts with alert-and-execute's discrete-entry shape |

---

## Research Summary

**Market Context**

- **Realistic capability**: a well-configured retail bot nets **5–25%** annually above buy-and-hold; well-managed max drawdown runs 15–20%, industry max 25%; Sharpe 1.0/1.5/2.0+ maps to acceptable/good/excellent. Against this, 30–50% is aggressive-but-not-absurd *if* Sharpe ≈1.0 comes first and leverage is deliberate.
- **Live degradation of 20–40% versus backtest** is the expected case, not the pessimistic one. Curve fitting is the documented dominant failure mode.
- **Trend following** is the deepest-evidenced directional edge: Sharpe 0.5–1.5 for MA/Donchian on BTC over a decade ([arXiv 2009.12155](https://arxiv.org/pdf/2009.12155)); 20/55 Donchian profitable through 2017–2023 bull *and* bear cycles at 30–40% win rate with 3–5× winner/loser ratio.
- **Volatility targeting is the highest-leverage sizing decision available**: inverse-vol weighting lifted Sharpe 0.99 → 1.54 and cut max DD −30.8% → −13.8%; vol targeting adds Sortino +30–40% at 20–50% targets ([Van Hemert et al.](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf)).
- **The highest-Sharpe published crypto trend work** is Zarattini, Pagani & Barbon ([SSRN 5209907](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907)): net-of-fees Sharpe **>1.5**, 10.8% annualized alpha vs BTC — via a Donchian lookback ensemble on daily bars across a **wider universe** with inverse-vol sizing. That is Option B, and it is the pre-committed escalation path.
- **Timeframe is a first-order cost decision**: mean Sharpe across 81 walk-forward configs was −12.71 at 1-minute and +0.791 at 60-minute, all 81 configs positive at 60m ([arXiv 2602.10785](https://arxiv.org/html/2602.10785)).
- **Chart-pattern geometry has no comparable evidence base** — the 60–65% hit-rate claims are vendor-side, on daily bars, without cost modeling or an out-of-sample protocol. The old PRD flagged this as a HIGH risk; it was right.
- **Funding carry is real but crowded**: Sharpe 3–6 in 2020–2023 with DD <5%, decaying to 4.06 from 2024 and **negative in 2025** ([arXiv 2510.14435](https://arxiv.org/pdf/2510.14435)).
- **Combining orthogonal sleeves is what pushes portfolio Sharpe above 1.0** (>1.0 combined vs 0.4–0.7 standalone) — the structural argument for Option D's shape.

**Technical Context**

- The engineering spine is sound and reusable: correct trailing-window Wilder indicators, a defensively-used regime classifier with healthy occupancy, a bar-by-bar engine that replays the *exact* production components so backtest and live cannot drift, explicit no-lookahead guarantees, idempotent storage, and 3,300 lines of tests.
- **The pivot is mostly specification, not architecture.** The risk model is one function (`setup.build_signal`); the cost model is one constant plus one term; the tier shift is a config change plus a hardcoded-literal audit; Donchian slots into a trigger interface (`check_breakout`) that is already generic over `(level, direction, end_ts)`.
- **Three genuinely new pieces of work**: 1D bar ingestion (blocked on Binance connectivity), Sharpe/DSR metrics on a time-indexed equity curve (the equal-size assumption is baked through `metrics.py`), and pooling `walk_forward()` across symbols (a signature change, not a flag).
- **The harness was validating the wrong parameters.** Its grid is the three regime thresholds — the healthiest layer — while the actual determinants of outcome (`MAX_RISK_PCT`, `MIN_REWARD_PCT`, `BREAKOUT_STOP_BUFFER_PCT`, `MAX_HOLD_BARS_15M`) sat frozen at unvalidated defaults and were never swept. The existing "FAIL" verdict is therefore a verdict on one arbitrary risk configuration, not on the strategy family — which is the strongest available reason to believe this pivot has room to work.

---

*Generated: 2026-07-26*
*Status: DRAFT — needs validation*
