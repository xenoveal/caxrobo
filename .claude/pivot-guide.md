Problem: this current version failed to achieve 30-50% annual return.

> **STATUS 2026-07-28 — this document is the v0.3.0 CHARTER, and v0.3.0 is now COMPLETE
> and ANSWERED NO.** Its northstar (">50% annual return") was tested and missed: the
> champion returned **−45.47%** over a 182-day untouched holdout, failing 5 of the gate's 7
> conditions. Read this file as the stated *intent* of v0.3.0, not as a description of
> current capability — for what was actually built and measured, see
> [PRPs/reports/KNOWN-LIMITATIONS-v0.3.0.md](PRPs/reports/KNOWN-LIMITATIONS-v0.3.0.md).
>
> Two parts of the "Method" section below were **not** built as written and should not be
> read as shipped: reinforcement learning (§1) — the engine uses an evolutionary tournament
> scored by walk-forward, never RL — and "1000 simultaneous model" (§3), where the measured
> campaign was 24 × 8 = 192 evaluations, sized down deliberately because every evaluation is
> also a multiple-testing trial.
>
> Of the two v0.1.0 HTML reports linked below, `backtest-result-initial.html` (7.5 MB) is
> **LOCAL-ONLY as of 2026-07-28** — untracked from git as a generated artifact. It remains on
> disk and is recoverable; see
> [PRPs/reports/DEPRECATED-ARTIFACTS.md](PRPs/reports/DEPRECATED-ARTIFACTS.md).
> `bad-backtest-result-investigation-result.html` (68 KB) is still tracked.

Previous iteration
* INITIAL (v0.1.0) - .claude/PRPs/prds/initial-requirements.md
    * This is the initial strategy we tried
    * Result was documented in .claude/PRPs/reports/backtest-result-initial.html
    * We investigated the bad result in .claude/PRPs/reports/bad-backtest-result-investigation-result.html
* v0.2.0 - .claude/PRPs/prds/hybrid-trend-voltarget.prd.md
    * Result was below target as defined in .claude/PRPs/reports/KNOWN-LIMITATIONS.md
    * I've tried to find another way using bruteforce-strategy to explore the best indicators for this use case but the engine itself failed to incorporate every possible indicators (example: RSI was not added, some chart patterns are not there).

I noticed there are several gaps
1. The strategy engine needs to be built from scratch instead of using plug n play mechanism which gives the user flexibility to create/explore a new strategy in a more scalable way.
2. There's no standardization workflow to improve the model. Every iteration makes the engine from scratch with some help from the previous solution.
3. The performance itself was still bad.


Therefore, I'd like to initiate the v0.3.0.
Northstar metric: >50% annual return.
Optimization strategy: 
1. bot opimizes the strategy parameter to achieve northstar metrics with the least risk possible (lowest drawdown, good risk to reward ratio).
2. self-learning: bot improves their engine when a new data being feed up (feedback loop).

Key solution:
1. Build a UI which structure this new framework of thinking. There will be 3 core functionalities
    * Data gathering: logic to get the data source such as prices, news, etc.
    * Strategy: logic to make decision such as the entry point, TP/SL point, position (long/short/no-entry).
    * Feedback loop: logic to analyze the previous strategy result after the data come, then re-iterate the system by explore potential strategy -> introduce new data pipeline if not exist -> build & refine the strategy.
2. Build the MVP logic for data gathering which is OLCHV data.
3. Build the MVP logic for Strategy which are
    * Step 1. Identify if there's any technical patterns present (ALL OF THEM) that I've listed in .claude/technical-pattern.md
    * Step 2. Upon breakout, combine the technical pattern with "volume on breakout" and MACD indicators. Usually, breakout has the highest volume (or at least a quite significant volume)
    * Step 3. Decide the position based on the chart pattern outcome (long/short/no-entry) and its entry, and TP/SL point.
    * Step 4. Caculate potential return (after costs - already built in prev version) and risk (SL). After that, filter the position that only fulfill a minimum 1:2 risk to reward ratio.
    * Step 5. Proceed with the strategy output if fulfill step 4 filter.
4. Build the MVP logic for feedback loop which are
    * Step 1. Open the position if the Strategy decided to proceed. 
    * Step 2. Wait until the Strategy pipeline recommended to close the position.
    * Step 3. When a position is closed, review the "result" vs "prediction" for these factors: 
        * TP: was the TP successfully identify the best price to exit by considering a reasonable risk to achieve maximum gain?
        * SL: was the SL successfully guard a reasonable loss whilst not over-conservative to avoid high return opportunity
        * Result: does the outcome of the trade for this timeperiod can potentially achieve >50% annual return? -> definitely "not" if the result was a loss but even if it's profitable, it might be performing less than target thus also not good.
    * Step 4. System self detect a potential improvement in strategy. For example: considering new data points (e.g., news), considering new strategy to try, considering new risk profile.
    * Step 5. Model finetune itself to refine the strategy and creating a new version of strategy.
    * Step 6. Model test its new strategy using future data.

Method: 
1. use reinforcement learning that gives Strategy model a reward vs punishment for every loop.
2. start with some data and allow model to not see the full data to avoid overfitting.
3. let the model run gradually from stupid to smart. run these simulation in parallel e.g., 1000 simultaneous model.
4. model can battle each other to build the best return with the least risk profile possible.

Principles
1. Trust and do not intervene the signal. For example, step 3 generated "long" on 1AM and "short" on 2AM despite the TP for the "long" position wasn't reached (overlapping signal). In this scenario, system will keep 2 positions. Model needs to be able to manage multiple positions.