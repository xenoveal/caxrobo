=== DIAGNOSTIC -- NOT A GATE VERDICT =====================================
 span 2023-07-27..2025-07-26 (TUNING span; holdout guard: 150d reserved)
 symbols BTCUSDT,ETHUSDT,SOLUSDT
 fixed exit policy, NOTHING SWEPT.  every detector at its ParamSpec defaults.
 ledger campaign="detector-report" rows=+16 (campaign total 16)
 SELECTION COST IF USED: if any detector is dropped from a campaign because of
 this table, add n_trials += 16 to that campaign. Reading it is free; letting it
 inform selection is not.
 rows sorted by TIER then NAME -- never by expectancy, because sorting by
 expectancy IS selection.
==========================================================================

detector                     tier  trades    win%      exp%     pf   maxdd%       c  verdict
--------------------------------------------------------------------------------------------------------
cup-and-handle                  1      31   41.9%  +2.9618%   2.88   20.39%   0.068  MEASURED
head-and-shoulders              1      29   24.1%  -1.0696%   0.60   46.76%   0.052  MEASURED
inverse-head-and-shoulders      1      28   17.9%  +0.1343%   1.06   34.38%   0.067  MEASURED
wyckoff-spring                  1       5      --        --     --       --      --  INSUFFICIENT(<20)
wyckoff-upthrust                1       5      --        --     --       --      --  INSUFFICIENT(<20)
ascending-triangle              2       1      --        --     --       --      --  INSUFFICIENT(<20)
bear-flag                       2     242   30.6%  -0.6860%   0.72  181.98%   0.051  MEASURED
bull-flag                       2     276   37.0%  +0.5323%   1.25   51.55%   0.056  MEASURED
double-bottom                   2     131   35.1%  +1.0901%   1.62   31.69%   0.065  MEASURED
double-top                      2     100   36.0%  +0.6500%   1.31   36.20%   0.053  MEASURED
falling-wedge                   2       0      --        --     --       --      --  INSUFFICIENT(<20)
rising-wedge                    2       0      --        --     --       --      --  INSUFFICIENT(<20)
rsi-divergence                  2      29   24.1%  -0.1897%   0.93   34.79%   0.050  MEASURED
descending-triangle            --       1      --        --     --       --      --  INSUFFICIENT(<20)
inverse-cup-and-handle         --      42   19.0%  -1.1095%   0.56   46.60%   0.050  MEASURED
symmetrical-triangle           --       1      --        --     --       --      --  INSUFFICIENT(<20)
--------------------------------------------------------------------------------------------------------
 c = mean round-trip cost / median risk_pct, against config.COST_RATIO_CEILING = 0.1. A detector whose c exceeds
 the ceiling is STRUCTURALLY unable to pay for itself regardless of win rate.
 exp% is NET of costs: run_graph_backtest charged 2*(FEE_PCT+SLIPPAGE_PCT) + FUNDING_PCT_PER_DAY*days,
 exactly as production charges it. No second P&L path exists in this command.

catalog: 144 rows / 18 families   covered 19  deferred 6  out-of-scope 119
 contract §9 concepts -- tier 1: 2/3  (Wyckoff Accumulation / Distribution DEFERRED: unmeasurable, see catalog.py)   tier 2: 6/6
 ledger rows carrying a tier   -- tier 1: 3/5   tier 2: 11/11

  # family                            rows  cov  def  oos
--------------------------------------------------------
  1 Reversal Patterns                   16    4    2   10
  2 Continuation Patterns               15    9    2    4
  3 Bullish Candlestick Patterns        14    0    0   14
  4 Bearish Candlestick Patterns        14    0    0   14
  5 Indecision Candlestick Patterns      5    0    0    5
  6 Gap Patterns                         4    0    0    4
  7 Harmonic Patterns                    8    0    0    8
  8 Elliott Wave                         2    0    0    2
  9 Wyckoff Structures                   4    2    2    0
 10 Volume-Based Patterns                4    0    0    4
 11 Support & Resistance Patterns        6    0    0    6
 12 Trendline Patterns                   4    0    0    4
 13 Moving Average Patterns              6    0    0    6
 14 Oscillator Signals                  14    4    0   10
 15 Fibonacci Patterns                   7    0    0    7
 16 Market Structure                     9    0    0    9
 17 Smart Money Concepts (SMC)           8    0    0    8
 18 Volatility Patterns                  4    0    0    4
--------------------------------------------------------
16 detector key(s) referenced by covered rows
