# v0.3.0 — Self-Learning Pattern Strategy Framework

## Problem Statement

A solo trader running personal capital on Binance crypto futures has built two bot versions (v0.1.0, v0.2.0) that both missed the 30–50% annual return target — v0.2.0 measured **+3.45% annualized vs +29.4% for a buy-and-hold basket**, losing to inaction on every axis. Each iteration rebuilt the engine from scratch, the strategy engine cannot express rich technical signals (RSI, divergence, most chart patterns), and there is no standardized workflow for improving a strategy between versions. Without fixing this, every future iteration pays the same rebuild cost and explores the same narrow signal space.

## Evidence

- [KNOWN-LIMITATIONS.md](../reports/KNOWN-LIMITATIONS.md) §0: v0.2.0 pooled result 1.11× / +3.45% ann. / Sharpe 0.282 / 56% max DD vs BTC buy-and-hold 2.21× / +30.3% / 0.80 / 53% DD. "There is no axis on which it wins."
- KNOWN-LIMITATIONS §0c: the parameter search **never explored the entry or the feature set** — MACD, RSI, stochastic, divergence, market structure, and volume gating were absent entirely; volume is computed on every signal but gates nothing.
- The brute-force strategy explorer (v0.2.x side effort) could not incorporate indicators outside its hardcoded set — RSI and most chart patterns were inexpressible, confirming the engine-flexibility gap.
- Signals are hardcoded modules dispatched by a fixed regime switch ([scan.py](../../../src/trading_bot/signals/scan.py)) — adding a detector means editing the engine.

## Proposed Solution

Build v0.3.0 as a **plug-and-play strategy framework** with three first-class module types — **Data gathering**, **Strategy**, and **Feedback loop** — fronted by an interactive builder UI, so new detectors, confirmations, and rules register against stable contracts instead of requiring engine rewrites. The MVP is one **thin vertical slice** through all three modules: OHLCV data → pattern detection (from the [technical-pattern.md](../../technical-pattern.md) catalog) → breakout confirmation via volume + MACD → position/entry/TP/SL decision → ≥1:2 risk-to-reward-after-costs filter → trade → closed-trade review → automated strategy refinement. Self-learning is realized as **population-based evolution**: hundreds of lightweight parameterized strategy variants trained on partial data, tournament-selected per generation for return-vs-risk, versioned, and re-tested on unseen forward data. This approach (over true deep RL) fits Mac-only compute, keeps every strategy variant auditable, and preserves the reward/punishment loop the pivot guide asks for.

## Key Hypothesis

We believe a plug-in engine expressing the full technical-pattern catalog with volume/MACD confirmation and an R:R-after-costs filter, evolved by a population-based feedback loop, will produce a strategy that beats buy-and-hold and reaches the northstar return for a solo Binance-futures trader.
We'll know we're right when a walk-forward out-of-sample run shows **>50% annualized return with lower drawdown and higher risk-adjusted return than the buy-and-hold basket**, passing the validation gate (sample adequacy, DSR) that v0.2.0 failed.

**Honesty clause:** this target exceeds what most published RL/pattern crypto systems demonstrate out-of-sample. The hypothesis is falsifiable and may well be falsified; the framework and workflow retain their value (cheaper iterations) even if this particular strategy family fails the gate.

## What We're NOT Building

- **Live order execution** — signal/alert-level only; no automated order placement with real funds.
- **News/sentiment data pipelines** — data-gathering MVP is OHLCV only; new data sources are a *discovered* addition the feedback loop may propose later (the framework contract must allow it, the MVP does not include it).
- **Multi-exchange support** — Binance futures only; no exchange abstraction work.
- **Position sizing / portfolio management** — equal-notional per trade stands; vol-targeted sizing remains deferred (as in v0.2.0 Phase 8).
- **True deep RL (neural policies)** — deferred; population-based evolution first. Revisit only if evolution plateaus with evidence.

## Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| OOS annualized return | >50% | Walk-forward out-of-sample, pooled, after costs |
| Beats buy-and-hold null | Return AND Sharpe above the equal-weight basket over the same span | Benchmark computed in the same run (KNOWN-LIMITATIONS §0 requirement) |
| Validation gate | All 5 conditions pass (≥30 OOS trades, Sharpe ≥1.0, DSR >0.95 with honest trial counting, DD ≤25%, all symbols positive expectancy) | [walkforward.py](../../../src/trading_bot/backtest/walkforward.py) gate, extended with the buy-and-hold null |
| Risk:Reward discipline | 100% of taken positions pass ≥1:2 R:R after costs at entry | Engine-enforced filter, audited in trade log |
| Iteration cost (framework goal) | New detector or rule added with zero engine-core edits | Code review of first 3 added plug-ins |
| Feedback-loop liveness | Every closed trade produces a TP/SL/result review record; each generation produces a versioned strategy candidate | Feedback-loop audit log |

## Open Questions

- [ ] Is >50% annualized OOS achievable at all in this asset class with technical signals? (The buy-and-hold basket did +29% with brutal drawdowns — the gate may prove the honest answer is "no.")
- [ ] How many of the ~150 catalog patterns have detectable, non-negative edge after costs? Detection ≠ edge; the catalog's own reliability tiers suggest most candlestick patterns alone are weak.
- [ ] How is DSR trial-counting kept honest when a population evolves thousands of variants per generation? (Degrees-of-freedom accounting must be automatic, or the gate becomes theater.)
- [ ] Which uncorrelated symbols get backfilled, and are enough available on Binance futures to break the ~0.76 BTC-beta correlation trap (§0b)?
- [ ] Geometric/structural patterns (Wyckoff, harmonics, SMC, Elliott) have no mature open-source detector — what subset is buildable with acceptable precision, and how is detector correctness itself validated?
- [ ] UI stack choice (no frontend exists in the repo) — local web app framework and how the builder serializes strategies into the plug-in contract.
- [ ] "Trust the signal" multi-position accounting: margin/liquidation modeling for overlapping long+short positions on the same symbol is unbuilt.

---

## Users & Context

**Primary User**
- **Who**: The project owner — a solo quant/builder trading their own capital on Binance crypto futures, technical, running everything on one Mac.
- **Current behavior**: Runs the v0.2.0 alert-level bot; executes trades manually; iterates strategy versions by rebuilding the engine with Claude's help each time.
- **Trigger**: A strategy version fails the validation gate or underperforms buy-and-hold, and improving it requires expressing signals the engine cannot represent.
- **Success state**: Opens the builder UI, composes or reviews an evolved strategy, sees walk-forward evidence against the buy-and-hold null, and trusts the resulting signals enough to execute them.

**Job to Be Done**
When a strategy iteration underperforms, I want to compose and evolve richer strategies from reusable data/strategy/feedback modules without rebuilding the engine, so I can search the signal space fast enough to find (or honestly rule out) a >50%-annual-return edge.

**Non-Users**
Anyone else. No multi-tenancy, no product polish, no docs-for-strangers, no equities/forex users. Single operator, single machine.

---

## Solution Detail

### Core Capabilities (MoSCoW)

| Priority | Capability | Rationale |
|----------|------------|-----------|
| Must | Plug-in framework: Data / Strategy / Feedback module contracts with registration, no engine-core edits to add a module | The scalability gap is the root cause of iteration cost |
| Must | Pattern detection pipeline from the technical-pattern.md catalog (thin slice first, expanding coverage) | v0.2.0's entry carried "almost no predictive content"; expressiveness is the alpha ceiling |
| Must | Breakout confirmation: volume-on-breakout + MACD, then position/entry/TP/SL, then ≥1:2 R:R-after-costs filter | Pivot guide Strategy Steps 2–5; volume machinery exists but gates nothing today |
| Must | Feedback loop: closed-trade TP/SL/result review vs the >50% target, improvement detection, strategy versioning, forward re-test | Pivot guide Feedback Steps 1–6; the standardized improvement workflow |
| Must | Population-based evolution: parallel strategy variants, partial-data training, tournament selection on return-vs-risk | The self-learning method, sized to Mac-only compute |
| Must | Interactive builder UI structuring the three modules (compose, run, monitor, inspect results) | User decision: UI is v1 must-have, the framework's forcing function |
| Must | Validation integrity: buy-and-hold null hypothesis in the gate, automatic trial counting for DSR, holding-day P&L attribution (fixes MEDIUM-5) | Without this the feedback loop optimizes into overfit; KNOWN-LIMITATIONS "next four things" #1–2 |
| Should | Backfill uncorrelated symbols beyond BTC/ETH/SOL | §4/§0b: 23 OOS trades can't clear a floor of 30; more BTC-beta adds rows, not information |
| Should | Multi-position management (overlapping long/short per "trust the signal" principle) | Pivot-guide principle; needs margin accounting design |
| Could | Detector-level edge reports (per-pattern hit rate, expectancy after costs) surfaced in the UI | Turns the catalog into evidence; guides evolution |
| Won't | Live execution, news data, multi-exchange, position sizing, deep RL | See "What We're NOT Building" |

### MVP Scope

One **thin full loop** on the existing 3 symbols: plug-in framework skeleton → a small pattern subset (the already-built H&S/triangle/flag detectors refactored into plug-ins, plus MACD and volume-gate confirmations) → position/TP/SL/R:R pipeline → backtested trades → feedback review records → one evolution generation producing a versioned strategy v2 → forward test. Every later phase widens a dimension (patterns, symbols, population size, UI depth) of a loop that already runs end-to-end.

### User Flow

1. Open builder UI → see the three module lanes (Data / Strategy / Feedback).
2. Compose a strategy: pick pattern detectors, confirmations (volume, MACD), R:R threshold — or load an evolved candidate.
3. Run backtest/walk-forward → see equity vs buy-and-hold, gate verdict, trade-level review records.
4. Start an evolution run → monitor generations (population fitness, best candidate, trial count) → promote a candidate to "active".
5. Active strategy emits signals; user executes manually; closed trades feed the next feedback cycle.

---

## Technical Approach

**Feasibility**: MEDIUM — strong reusable backtest/data core; greenfield UI and evolution engine; the statistics, not the code, are the hard part.

**Architecture Notes**
- **Reuse the honest core**: cost model, no-lookahead enforcement, shared scan/backtest dispatch, walk-forward gate, SQLite OHLCV store (112 MB), ccxt backfill ([engine.py](../../../src/trading_bot/backtest/engine.py), [storage.py](../../../src/trading_bot/data/storage.py)). These are v0.2.0's proven assets; the framework wraps them rather than replacing them.
- **Plug-in contracts**: `DataSource` (bars/series by symbol+timeframe), `Detector` (pattern events with direction/level), `Confirmation` (gate on a candidate event), `PositionPolicy` (entry/TP/SL/direction), `Filter` (R:R-after-costs), `Reviewer` (closed-trade analysis), `Mutator` (strategy-variant generation). A strategy = serializable graph of registered plug-ins — this serialization is also the UI's file format and the evolution engine's genome.
- **Existing detectors migrate, not rewrite**: patterns.py (H&S, triangles, flags), pivots.py, donchian.py, bollinger sit behind the `Detector` contract as the first plug-ins.
- **Evolution over deep RL**: population of strategy graphs + parameter vectors; fitness = risk-adjusted OOS return vs buy-and-hold; partial-data exposure per pivot-guide method §2; process-parallel on Mac cores. Every variant is a readable config, so "why did it trade" stays answerable.
- **Gate as a service**: the walk-forward gate (extended with the buy-and-hold null and automatic trial counting) is the *only* fitness oracle the evolution loop can query — one code path, no side-channel evaluation that dodges DSR accounting.
- **UI**: local web app (Python backend serving the framework API + a lightweight frontend); reads/writes the same strategy-graph serialization. Stack choice is an open question resolved in the UI phase.
- **Environment**: Homebrew python@3.11 venv (pandas-ta unavailable on PyPI; TA-Lib would need its C library — prefer extending the hand-rolled indicator set unless coverage demands otherwise).

**Technical Risks**

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Overfitting: evolution + 150 detectors explodes degrees of freedom; gate becomes theater | **H** | Gate-as-only-oracle; automatic trial counting into DSR; partial-data training; final holdout never seen by evolution; buy-and-hold null |
| >50% target is unreachable; project "fails" despite a working framework | **H** | Frame as hypothesis (see honesty clause); framework/workflow value is independent; gate gives an honest "no" cheaply |
| Geometric/structural pattern detectors (Wyckoff, harmonics, SMC) are research projects with no reference implementation | M | Thin-slice ordering: ship reliability-tier-1 patterns first; detector correctness gets its own fixture tests; full catalog is a direction, not a v1 gate |
| Statistical validation still starved: 3 correlated symbols → <30 OOS trades regardless of strategy | M | Backfill uncorrelated symbols early (Phase 2); fapi.binance.com reachable as of 2026-07-26, re-ping first |
| Mac-only compute caps population size/generations | M | Lightweight variants (vectorized backtests), process pool, overnight runs; cloud explicitly deferred until loop shows life |
| UI scope eats the project (greenfield frontend) | M | UI consumes the same serialization the engine already needs; build after the loop runs headless; MVP = compose/run/inspect, not drag-and-drop polish |
| Multi-position margin/liquidation accounting subtly wrong | M | Explicit design doc + tests before enabling overlapping positions; until then engine holds positions independently with stated assumptions |

---

## Implementation Phases

<!--
  STATUS: pending | in-progress | complete
  PARALLEL: phases that can run concurrently
  DEPENDS: phases that must complete first
  PRP: link to generated plan file once created
-->

| # | Phase | Description | Status | Parallel | Depends | PRP Plan |
|---|-------|-------------|--------|----------|---------|----------|
| 1 | Validation integrity | Buy-and-hold null in the gate; holding-day P&L attribution (MEDIUM-5); automatic trial counting | pending | with 2 | - | [phase1](../plans/v0.3.0/phase1-validation-integrity.plan.md) |
| 2 | Data breadth | Backfill uncorrelated Binance futures symbols; correlation report; storage capacity check | pending | with 1 | - | [phase2](../plans/v0.3.0/phase2-data-breadth.plan.md) |
| 3 | Plug-in framework core | Module contracts, registry, strategy-graph serialization; migrate existing detectors/indicators as first plug-ins | pending | - | 1 | [phase3](../plans/v0.3.0/phase3-plugin-framework-core.plan.md) |
| 4 | Strategy pipeline (thin slice) | Pattern→volume/MACD confirmation→position/TP/SL→R:R-after-costs filter as composed plug-ins; backtest parity with v0.2.0 harness | pending | - | 3 | [phase4](../plans/v0.3.0/phase4-strategy-pipeline-thin-slice.plan.md) |
| 5 | Feedback loop MVP | Closed-trade Reviewer (TP/SL/result vs target), review records, strategy versioning, forward re-test protocol | pending | with 6 | 4 | [phase5](../plans/v0.3.0/phase5-feedback-loop-mvp.plan.md) |
| 6 | Evolution engine | Population-based variant generation, partial-data training, tournament selection via the gate oracle, parallel execution | pending | with 5 | 4 | [phase6](../plans/v0.3.0/phase6-evolution-engine.plan.md) |
| 7 | Builder UI | Local web app: compose strategies from registered plug-ins, run/monitor backtests and evolution, inspect reviews vs buy-and-hold | pending | - | 5, 6 | [phase7](../plans/v0.3.0/phase7-builder-ui.plan.md) |
| 8 | Pattern coverage expansion | Widen detector library through the catalog by reliability tier, with per-detector fixture tests and edge reports | pending | with 7 | 4 | [phase8](../plans/v0.3.0/phase8-pattern-coverage-expansion.plan.md) |
| 9 | Walk-forward campaign | Full evolution campaign on broadened data; final untouched holdout; gate verdict vs northstar | pending | - | 7, 8 | [phase9](../plans/v0.3.0/phase9-walkforward-campaign.plan.md) |

> **Plans generated 2026-07-27.** All nine share the binding
> [shared architecture contract](../plans/v0.3.0/_shared-architecture-contract.md), which fixes
> every cross-phase interface and records measured corrections to this PRD — see especially §0a
> (20 symbols are already stored, not 3, so Phase 2's description above overstates the backfill
> work) and §4 (the gate has 7 conditions once the buy-and-hold null is added, not the 5 named in
> Success Metrics). Statuses remain `pending`: a plan exists, no implementation has started.

### Phase Details

**Phase 1: Validation integrity**
- **Goal**: Make the measuring stick honest *before* anything optimizes against it.
- **Scope**: Buy-and-hold basket computed in every walk-forward run and added to the gate; spread trade P&L across holding days (fixes MEDIUM-5, attacks the kurtosis blocking DSR); trial-count ledger API that every evaluation increments.
- **Success signal**: Re-running v0.2.0's walk-forward reproduces KNOWN-LIMITATIONS §0's verdict from one command; DSR inputs no longer show 30+ kurtosis.

**Phase 2: Data breadth**
- **Goal**: Break the 0.76-correlation / 23-trade sample starvation.
- **Scope**: Re-ping fapi.binance.com; select candidate symbols by liquidity and low BTC correlation; backfill via existing machinery; produce a correlation matrix report.
- **Success signal**: ≥8 symbols stored with pairwise daily-return correlation materially below the BTC-beta cluster, gap-checked.

**Phase 3: Plug-in framework core**
- **Goal**: The scalability contract — new modules without engine edits.
- **Scope**: Contracts for DataSource/Detector/Confirmation/PositionPolicy/Filter/Reviewer/Mutator; registry; strategy-graph serialization; migrate patterns.py, pivots.py, donchian, bollinger, wilder behind the contracts; existing tests keep passing.
- **Success signal**: v0.2.0-equivalent strategy expressed as a serialized graph reproduces its backtest numbers through the new dispatch.

**Phase 4: Strategy pipeline (thin slice)**
- **Goal**: The pivot guide's Strategy Steps 1–5 running end-to-end.
- **Scope**: MACD indicator plug-in; volume-on-breakout Confirmation (machinery exists, currently gates nothing); pattern-outcome PositionPolicy (direction/entry/TP/SL); ≥1:2 R:R-after-costs Filter; composed thin-slice strategy backtestable on 3+ symbols.
- **Success signal**: A composed strategy produces filtered, confirmed trades in backtest with every position logged against its R:R justification.

**Phase 5: Feedback loop MVP**
- **Goal**: The pivot guide's Feedback Steps 1–6 as a standardized workflow.
- **Scope**: Reviewer plug-in scoring each closed trade (TP quality, SL quality, pace vs >50% ann.); persistent review records; strategy version registry; forward-test protocol for candidate strategies on unseen data.
- **Success signal**: A full loop iteration — trade, close, review, refine, version, forward-test — executes without manual glue.

**Phase 6: Evolution engine**
- **Goal**: Self-learning via population-based evolution within honest statistics.
- **Scope**: Mutator plug-ins (parameter jitter, detector swap/add/remove); partial-data training windows; tournament selection scored only through the gate oracle; process-parallel runner sized to the Mac; generation audit log with trial counts.
- **Success signal**: An overnight run evolves a population whose best candidate beats the seed strategy OOS, with a DSR that accounts for every variant evaluated.

**Phase 7: Builder UI**
- **Goal**: The structured thinking framework made visible and operable.
- **Scope**: Local web app over the framework API: three module lanes, strategy composition from registered plug-ins, run/monitor backtests and evolution generations, equity-vs-buy-and-hold and review-record views.
- **Success signal**: A strategy composed entirely in the UI round-trips through serialization, backtests, and displays its gate verdict.

**Phase 8: Pattern coverage expansion**
- **Goal**: Widen expressiveness toward the full technical-pattern.md catalog.
- **Scope**: Detectors added by reliability tier (tier-1 first: cup & handle, H&S refinements, Wyckoff acc/dist, double top/bottom, flags, triangles, wedges, RSI divergence), each with geometry fixture tests and a per-detector edge report; candlestick/oscillator/SMC tiers follow.
- **Success signal**: Each new detector lands as a plug-in with zero engine edits, passing fixtures, and an edge report; catalog coverage tracked, not assumed.

**Phase 9: Walk-forward campaign**
- **Goal**: The verdict on the key hypothesis.
- **Scope**: Full evolution campaign on the broadened symbol set; a final holdout span never touched by any evolution generation; gate verdict including the buy-and-hold null and honest DSR.
- **Success signal**: Either the gate passes at >50% ann. OOS (northstar met) or the framework delivers a statistically honest "no" — both are completed outcomes; only an ambiguous result is failure.

### Parallelism Notes

Phases 1 and 2 are independent (statistics hardening vs data acquisition) and can run concurrently. Phases 5 and 6 both build on the Phase 4 pipeline and can proceed in parallel once its contracts are stable. Phase 8 (more detectors) only needs Phase 4's Detector contract, so it can run alongside the UI build (7). Phase 9 requires everything.

---

## Decisions Log

| Decision | Choice | Alternatives | Rationale |
|----------|--------|--------------|-----------|
| Self-learning method | Population-based evolution | True deep RL (small scale); analytical re-grid only | Mac-only compute; auditable variants; preserves reward/selection loop; deep RL's overfitting record in the literature |
| MVP shape | Thin full loop | Full pattern catalog first; UI first | Tests the riskiest integration (feedback loop) earliest; every phase then widens a working loop |
| UI in v1 | Yes — full interactive builder, built after headless loop works | Config-files + HTML reports first | User decision: UI is a must-have forcing function; sequencing after Phase 5/6 caps scope risk |
| Success evidence | Walk-forward OOS pass | Paper trading period; live capital | User decision; consistent with v0.2.0's gate discipline, now with buy-and-hold null |
| Execution | Signal/alert-level only | Automated order placement | Out of scope by user decision; also a safety boundary |
| Data scope | OHLCV only for MVP | Include news/sentiment now | Framework contract allows later data sources; feedback loop should *discover* the need |
| Null hypothesis | Buy-and-hold basket | Zero-return null (v0.2.0's mistake) | KNOWN-LIMITATIONS §0: the old gate could bless strategies worse than inaction |
| Pattern coverage | Full catalog as direction, reliability-tier order | ALL patterns before first loop | Detection ≠ edge; tiered delivery keeps the loop running while coverage grows |

---

## Research Summary

**Market Context**
- No mature single library covers the full chart-pattern catalog; TA-Lib handles ~60 candlestick patterns, geometric/structural patterns (H&S, wedges, Wyckoff, harmonics, SMC) are fragmented across small projects ([chart-patterns topic](https://github.com/topics/chart-patterns), [candlestick-patterns-detection topic](https://github.com/topics/candlestick-patterns-detection)); some approaches use vision models (YOLO) rather than geometry.
- RL trading frameworks exist (FinRL, [Freqtrade FreqAI-RL](https://docs.freqtrade.io/en/2026.2/freqai-reinforcement-learning/)) but the literature's consistent finding is severe backtest overfitting; [FinRL_Crypto](https://github.com/berendgort/FinRL_Crypto) exists specifically to mitigate it and the [underlying paper](https://arxiv.org/pdf/2209.05559) argues single-validation walk-forward is itself insufficient — supporting partial-data training and a never-touched final holdout.
- No surveyed open system credibly demonstrates >50% annualized OOS in crypto with technical signals — the northstar is a hypothesis at the optimistic edge of the field.

**Technical Context**
- Reusable core from v0.2.0: cost-honest backtest engine, no-lookahead enforcement, shared scan/backtest dispatch, walk-forward gate, SQLite OHLCV store + ccxt backfill, hand-rolled Wilder/Bollinger/Donchian indicators, and working H&S/triangle/flag detectors ([patterns.py](../../../src/trading_bot/signals/patterns.py)) currently retired from dispatch.
- Binding constraints from KNOWN-LIMITATIONS: 3 symbols at ~0.76 correlation → ~1.2 effective independent samples; 23 OOS trades vs a 30 floor; exit-day P&L attribution distorting Sharpe/DSR (MEDIUM-5, unfixed); volume computed but never gating; entry/feature space never searched.
- Environment: Homebrew python@3.11 venv required; pandas-ta gone from PyPI; no frontend stack exists in the repo; fapi.binance.com reachable again as of 2026-07-26 (re-verify before backfill).

---

*Generated: 2026-07-27*
*Status: DRAFT - needs validation*
