"""
Configuration for trading bot.

Timestamps stored are ccxt/Binance candle OPEN times in epoch milliseconds, UTC.
Each timeframe is fetched natively from the exchange, never resampled.
"""

import os
from datetime import datetime, timezone

SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
# Phase 1 (v1.0 PRD): "1d" added so the regime tier can move up to daily bars
# in Phase 4. "1m" is deliberately NOT here — PRD Open Question #4 (1-minute
# exit resolution) is unresolved, and 1m would be ~1.84M bars/symbol/3.5yr.
# If it is ever resolved to "yes": add "1m" here and "1m": 60_000 to
# storage.TIMEFRAME_MS, plus a per-minute entry in poller._CRON_BY_TIMEFRAME.
TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h", "1d")
BACKFILL_START = "2023-01-01"
DB_PATH = "data/ohlcv.db"

# Number of intervals a series may lag behind now before gap-report flags it stale.
STALENESS_INTERVALS = 2

# Regime classifier thresholds (unvalidated defaults; tuned empirically in Phase 5).
# Phase 4: the regime tier is 1D.
# NOTE on ATR_PERCENTILE_WINDOW: 180 bars was ~30 days at the old 4H tier and is
# ~180 days at 1D. The window is deliberately NOT rescaled — the PRD keeps the
# regime classifier unchanged (it is the one measured-healthy component), a
# 30-sample percentile has too coarse a resolution for a 0.90 gate (1/30, making
# the gate effectively "top 3 bars"), and retuning it here would consume a degree
# of freedom for no measured benefit. The extreme-vol gate's meaning does change:
# it now asks "extreme against 180 days of context" rather than 30.
REGIME_TIMEFRAME = "1d"
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25.0
ATR_PERCENTILE_WINDOW = 180
ATR_EXTREME_PERCENTILE = 0.90
# 207 bars. At the 1D tier this is 207 CALENDAR DAYS of warmup: with history from
# 2023-01-01, the first non-"uncertain" label lands ~2023-07-27. Any backtest or
# walk-forward window starting before then produces zero trades.
REGIME_MIN_BARS = 2 * ADX_PERIOD - 1 + ATR_PERCENTILE_WINDOW

# Phase 3: chart-pattern breakout signal method (trending regime only).
# All thresholds are unvalidated defaults; tuned empirically in Phase 5.
# Phase 4 tiers: 1D regime / 4H setup / 1H trigger.
SIGNAL_PATTERN_TIMEFRAME = "4h"  # setups detected on setup-timeframe bars
SIGNAL_TRIGGER_TIMEFRAME = "1h"  # breakout trigger confirmed on trigger-timeframe bars
PIVOT_SPAN = 3  # bars on each side that a fractal pivot must dominate
PATTERN_LOOKBACK_BARS = 180  # 1H bars scanned for pattern geometry
PATTERN_MAX_AGE_BARS = 12  # pattern's last pivot must be within this many 1H bars

# Head-and-shoulders tolerances (fractions of price).
HS_SHOULDER_TOLERANCE = 0.03  # max relative height difference between shoulders
HS_HEAD_MIN_PROMINENCE = 0.01  # head must exceed both shoulders by at least this

# Triangle: converging trendlines through pivot highs/lows.
TRIANGLE_MIN_PIVOTS_PER_SIDE = 2  # need >= 2 pivot highs and >= 2 pivot lows (floor 2)
TRIANGLE_MIN_CONVERGENCE = 0.25  # end range must contract by at least this fraction
# The trendlines must span a plausible triangle, not the whole lookback: only
# pivots inside the last TRIANGLE_MAX_WIDTH_BARS are fitted, and the fitted
# structure must be at least TRIANGLE_MIN_WIDTH_BARS wide.
TRIANGLE_MAX_WIDTH_BARS = 80
TRIANGLE_MIN_WIDTH_BARS = 20
# Bars may pierce their own trendline by this fraction of price before the
# geometry is rejected as not actually bounded by the lines.
TRIANGLE_CONTAINMENT_TOL = 0.005

# Flag: impulse pole followed by a shallow consolidation drift.
FLAG_POLE_WINDOW_BARS = 12  # max 1H bars for the impulse pole
FLAG_POLE_MIN_PCT = 0.03  # pole must move at least this fraction of price
FLAG_CONSOL_MIN_BARS = 4  # consolidation length bounds (1H bars)
FLAG_CONSOL_MAX_BARS = 24
FLAG_MAX_RETRACE = 0.5  # consolidation may retrace at most this fraction of pole

# Breakout volume confirmation (graded confidence input, never a hard block).
VOLUME_LOOKBACK = 20  # trigger-timeframe bars in the rolling volume average
VOLUME_HIGH_RATIO = 1.5  # trigger volume >= ratio * average => "notably high"

# How many of the most recent closed trigger-timeframe bars may supply the crossing.
# 1 = the latest bar only (entry reference is always the freshest close). Raise
# it to let an ad-hoc scan still see a crossing it arrived too late for, at the
# cost of a staler entry price.
BREAKOUT_TRIGGER_LOOKBACK_BARS = 1

# Phase 4: mean-reversion fade signal method (ranging regime only).
# All thresholds are unvalidated defaults; tuned empirically in Phase 5.
BB_PERIOD = 20  # Bollinger middle-band SMA period (setup-tier bars)
BB_STD = 2.0  # band width in rolling standard deviations
FADE_STRETCH_MAX_AGE_BARS = 6  # band-stretch bar must be within this many setup-tier bars

# Phase 6: fade re-qualification. The ranging sleeve is kept behind an
# explicit switch so a DROP verdict is a recorded decision rather than a code
# deletion — the method stays tested and a future re-test costs nothing.
# Honored by BOTH dispatch paths (signals/scan.py and backtest/engine.py);
# flipping it must change live and backtest behavior identically.
# Set from .claude/PRPs/reports/fade-requalification.md's verdict.
# DROPPED per .claude/PRPs/reports/fade-requalification.md (2026-07-27): measured
# on the tuning span under the Phase 2 risk model and Phase 4 tiers, the sleeve
# failed three independent DROP clauses — pooled expectancy -0.3883% (n=297),
# 0 of 3 symbols positive, and cost ratio c above the 0.10 ceiling on all three
# (0.1812/0.1147/0.1001). Ranging now produces no signals, which the PRD accepts.
# meanrev.py, bollinger.py and their tests are intentionally left intact and
# green so this decision stays reversible and a future re-test costs nothing.
FADE_ENABLED = False

# Phase 5: Donchian trend engine (A-core), trending regime only.
# 20/55 are CANONICAL (Donchian/Turtle lineage), not fitted here, and are
# frozen: they must never appear in a walk-forward grid in this phase. See
# the plan's Stated Assumptions A1/A2 — channels compute on the SETUP tier
# (config.SIGNAL_PATTERN_TIMEFRAME), the 20-bar channel supplies both the
# entry level and the opposite-channel exit, and the 55-bar MID-LINE is the
# trend filter (nothing else uses 55).
DONCHIAN_ENTRY_PERIOD = 20   # bars in the entry / opposite-exit channel
DONCHIAN_TREND_PERIOD = 55   # bars in the mid-line trend filter

# Phase 5 exit management, REPAIRED per
# .claude/PRPs/reports/code review/phase4-7-code-review.md (measured 2026-07-27).
#
# TRAIL_ATR_MULTIPLE is deliberately a SEPARATE constant from
# ATR_STOP_MULTIPLE. Reusing k for both made the ratchet trail exactly as
# tight as the entry stop, which closed 91.3% of all trades at a median
# 10-hour hold on a system whose regime tier is 1D — and preempted the
# opposite-channel exit so completely that deleting that exit produced a
# bit-identical backtest. Widening the shared k to 3.0 measured WORSE than
# either 1.5 or no trail at all, so the trail is not mis-parameterised; it is
# structurally wrong for a trend system and defaults OFF.
#
# With the trail off and the target off, exits are: initial ATR stop,
# opposite-channel touch, time stop — the canonical Turtle shape. Both flags
# are walk-forward GRID AXES (backtest.walkforward.DEFAULT_GRID), so THE GATE
# arbitrates them on out-of-sample data rather than these defaults doing it.
TRAIL_ENABLED = False
TRAIL_ATR_MULTIPLE = 3.0  # only read when TRAIL_ENABLED; independent of k
# Whether Donchian trades carry the measured-move target (level +/- channel
# width) as an EXIT. The reward:risk screen in setup.build_signal always uses
# that target and is unaffected by this flag — this governs the exit only.
DONCHIAN_TARGET_ENABLED = False
# Warmup: the channel needs PERIOD prior bars (trailing, current bar excluded)
# and ADX(14) needs 2*14-1 = 27 bars. 55 dominates. On 4H bars that is ~9.2 days.
DONCHIAN_MIN_BARS = max(DONCHIAN_TREND_PERIOD + 1, 2 * ADX_PERIOD - 1)

# Phase 5: backtesting & walk-forward validation.
# Time-stop, in SIGNAL_TRIGGER_TIMEFRAME bars: 96 * 1h = 4 days.
# Renamed in Phase 4 from its old tier-specific name. The BAR COUNT is unchanged, so the
# tier shift rescales the holding limit 24h -> 4 days for free without
# introducing a new free parameter. Changing this number is a consumed degree of
# freedom and must be logged as one (PRD: trial-log discipline).
MAX_HOLD_BARS_TRIGGER = 96
WF_TRAIN_DAYS = 180  # walk-forward training window
WF_TEST_DAYS = 60  # walk-forward test window (fold step)
WF_OOS_DAYS = 90  # final untouched out-of-sample holdout
WF_MIN_TRADES = 30  # minimum trades for a parameter combo / gate to count (Phase 7: was 5, noise-fit)

# ---------------------------------------------------------------------------
# Phase 2: honest cost & risk model. Stop distance is now derived from
# volatility (ATR), never from a fixed percentage of entry. MAX_RISK_PCT
# below is retained ONLY as documentation of the account-risk budget the
# human discharges via position sizing (Phase 8) — it is not read by any
# signal or filter code from this phase forward.
# ---------------------------------------------------------------------------
MAX_RISK_PCT = 0.005  # ACCOUNT-RISK BUDGET (human sizing), NOT a stop-distance rule

ATR_STOP_PERIOD = 14  # Wilder ATR period for the stop distance
# k in stop = k * ATR. Inherited as a conventional default, then CONFIRMED
# against the cost constraint at the Phase 4 setup tier (4H) and re-frozen —
# derived from c <= COST_RATIO_CEILING alone, never from PnL. Measured
# 2026-07-27 on stored 4H history, median risk_pct at k = 1.5:
#   BTCUSDT 1.968% -> c = 0.0711 | ETHUSDT 2.679% -> c = 0.0523
#   SOLUSDT 3.752% -> c = 0.0373                     (ceiling 0.10, all PASS)
# k was NOT adjusted, so no degree of freedom was consumed. Note this ceiling
# was NOT satisfiable at the old 1H setup tier (c was 0.28/0.18/0.13 there):
# clearing it needs risk_pct >= 1.4%, and raising ATR by coarsening the setup
# tier is what delivered it — not widening k, which would fit the stop to the
# fee schedule instead of to volatility.
ATR_STOP_MULTIPLE = 1.5
# Minimum reward:risk. Breakout gates on the gross ratio (its risk has an ATR
# floor); fade gates on risk.atr_stop.net_rr, the cost-adjusted ratio, because
# its structural stop has no such floor. Replaces the old absolute band.
RR_FLOOR = 1.5

FEE_PCT = 0.0005  # Binance USDT-M VIP-0 taker fee per side (was 0.0004 — understated)
SLIPPAGE_PCT = 0.0002  # assumed slippage per side (unchanged)
FUNDING_PCT_PER_DAY = 0.0001  # frozen pessimistic placeholder; real ingestion is Option C
COST_RATIO_CEILING = 0.10  # c = cost / risk_pct; asserted in tests/reports, not enforced at runtime

# ---------------------------------------------------------------------------
# v0.3.0 Phase 2: data breadth. RESEARCH_SYMBOLS is a SEPARATE, WIDER universe
# from SYMBOLS above (production stays a 3-symbol product, per
# scripts/bruteforce/universe.py:3-6). Nothing in the live signal path reads
# it: every existing cli.py branch (backtest/walkforward/regime/signal/
# gap-report) still defaults to SYMBOLS. Phase 9 is the intended consumer,
# passing RESEARCH_SYMBOLS explicitly. Correlation is measured on DAILY
# returns from the 1d series over the benchmark span, so this phase's report
# and Phase 1's benchmark report are comparable.
# ---------------------------------------------------------------------------
CORRELATION_TIMEFRAME = "1d"  # frozen; KNOWN-LIMITATIONS §0b's anchors are daily figures
# The benchmark span start -- the first date the 1D regime tier's 207-calendar
# -day warmup permits a non-"uncertain" label. Deliberately NOT BACKFILL_START.
CORRELATION_START = "2023-07-27"
CORRELATION_MIN_OVERLAP_BARS = 365  # fewer aligned daily returns => r reported as None
CORRELATION_ANCHOR_SYMBOL = "BTCUSDT"  # beta reference; pinned into every selection
CORRELATION_SELECT_N = 8  # non-anchor symbols selected; PRD's stated success floor is >=8
# Candidate screens. Both measured NON-BINDING on the 20 currently stored
# (measured minimum median quote volume $51.7M = ATOMUSDT; measured maximum
# beta among selected 1.285 = UNIUSDT) -- pre-registered so a future
# low-correlation expansion cannot quietly buy correlation reduction with
# illiquidity the 2bps slippage assumption cannot support, or with leveraged
# BTC proxies (DOGE beta 1.433, NEAR beta 1.378) whose drawdowns compound BTC's.
CORRELATION_MIN_QUOTE_VOLUME_USD = 50_000_000.0
CORRELATION_MAX_BTC_BETA = 1.35
# Decision rule D1's threshold (a pre-registered, deliberately modest bar;
# measured expectation 1.54x). Never tune this after seeing the measurement
# -- that is the "gate becomes theater" failure this phase exists to avoid.
CORRELATION_EFFECTIVE_N_MIN_RATIO = 1.5

# RESEARCH_SYMBOLS: the MEASURED output of `python -m trading_bot.cli
# correlation-report --out .claude/PRPs/reports/phase2-correlation-report.md`,
# run 2026-07-27 against data/ohlcv.db (20 symbols, 1d, 2023-01-01 ->
# 2026-07-26). NEVER REORDER -- the report and any downstream table key off
# this exact ordering (anchor first, then ascending mean-pairwise-r rank).
# Regenerate with the command above; do not hand-edit this tuple.
#
# Measured: 1095 daily returns, span 2023-07-27 -> 2026-07-26. r_bar=0.4885,
# effective N (Kish)=1.834 / (participation ratio)=2.925, against
# config.SYMBOLS' r_bar=0.7574, effective N (Kish)=1.193 / (participation
# ratio)=1.395. Ratio 1.834/1.193 = 1.537 >= CORRELATION_EFFECTIVE_N_MIN_RATIO
# (1.5) -> decision D1, by a 2.5% margin -- see
# .claude/PRPs/reports/phase2-correlation-report.md for the full report.
#
# Honest reading (contract §0a/§0b), do not let this be mistaken for
# "the correlation trap is solved": 3 -> 9 symbols is 3x the rows and only
# ~1.5x the independent information. Within Binance USDT-M perps there is no
# genuinely uncorrelated crypto; 19 of 20 stored liquid majors sit at BTC
# beta/r 0.56-0.81, and the one exception (TRXUSDT, r 0.222) is also among
# the least liquid. Phase 9 MUST read its DSR as governed by effective
# N ~= 1.8, never by len(RESEARCH_SYMBOLS) = 9.
RESEARCH_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",   # anchor (pinned); L1 major; deepest book in the universe
    "TRXUSDT",   # L1; the single most-detached symbol measured (r vs BTC 0.222)
    "BCHUSDT",   # payments
    "BNBUSDT",   # exchange
    "XRPUSDT",   # payments
    "UNIUSDT",   # DeFi
    "OPUSDT",    # L2
    "AAVEUSDT",  # DeFi
    "FILUSDT",   # storage
)


# ---------------------------------------------------------------------------
# v0.3.0 Phase 4: strategy pipeline thin slice -- MACD, volume gating, and the
# >=1:2-after-costs reward:risk target. Reserved prefixes per the v0.3.0 shared
# architecture contract §7: MACD_*, VOLUME_CONFIRM_*, RR_TARGET_MIN.
# ---------------------------------------------------------------------------

# MACD (Appel 1979). Canonical 12/26/9 -- inherited, NOT fitted here, and never
# swept in this phase. Computed on the SETUP tier (SIGNAL_PATTERN_TIMEFRAME):
# risk (1.5*ATR) and reward (channel width) are both setup-tier quantities, so
# confirming momentum on a different tier would put signal and risk on different
# volatility scales -- the same argument as signals/donchian.py's Stated
# Assumption A1.
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
# Warmup, DERIVED (never a magic number), mirroring DONCHIAN_MIN_BARS above: the
# MACD line is first defined at positional index MACD_SLOW_PERIOD - 1, and the
# signal line needs MACD_SIGNAL_PERIOD defined line values on top of that, so the
# first defined index is MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD - 2 and the
# required BAR COUNT is one more. MEASURED against pandas 3.0.3's
# ewm(adjust=False, min_periods=period): line idx 25, signal/hist idx 33 at
# 12/26/9. 34 setup bars is ~5.7 days at the 4H tier -- dominated by
# REGIME_MIN_BARS (207 daily bars), so MACD costs no usable history, and
# 34 << PATTERN_LOOKBACK_BARS so no lookback constant changes.
MACD_MIN_BARS = MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD - 1  # 34

# MACD confirmation threshold on the PRICE-NORMALISED histogram
# ((line - signal) / close, so the value is portable across symbols). 0.0 makes
# it a pure SIGN test, which introduces no fitted number. The constant exists so
# the Phase 6 Mutator can jitter it within declared bounds -- not so this phase
# can tune it.
MACD_CONFIRM_MIN_HIST = 0.0

# Volume-on-breakout confirmation, Phase 4. THIS IS A BEHAVIOR CHANGE:
# VOLUME_LOOKBACK / VOLUME_HIGH_RATIO have existed since v0.2.0 and gated
# NOTHING (KNOWN-LIMITATIONS §0c: "volume is computed on every signal but gates
# nothing"). The Confirmation plug-in makes it a HARD GATE, for graph-composed
# strategies only -- the legacy signals/* path is unchanged.
#
# The threshold is defined BY REFERENCE to the existing constant, so activating
# the gate invents no new number and consumes no additional degree of freedom.
# VOLUME_HIGH_RATIO itself is unchanged (contract §7: do not modify existing
# constants). A re-typed 1.5 here would be the exact drift trap
# TRAIL_ATR_MULTIPLE was split out of.
VOLUME_CONFIRM_MIN_RATIO = VOLUME_HIGH_RATIO
# Undefined volume (NaN ratio: fewer than VOLUME_LOOKBACK prior bars, or a
# non-positive rolling mean -- see signals/breakout.py:131-137) REJECTS.
# "Unknown" must never read as "confirmed". Cost: the first ~VOLUME_LOOKBACK+1
# trigger bars of each series cannot trade, negligible against REGIME_MIN_BARS'
# 207 days.
VOLUME_CONFIRM_REQUIRE_DEFINED = True

# The PRD's ">=1:2 risk:reward AFTER COSTS" requirement, as a NET floor consumed
# by plugins/filters/rr_after_costs.py via risk.atr_stop.net_rr.
#
# This is a NEW constant, deliberately NOT a change to RR_FLOOR = 1.5, which
# stays the legacy breakout path's GROSS floor so its measured behavior is not
# silently altered.
#
# Requiring net_rr >= X is equivalent to requiring gross_rr >= X + (X+1)*c,
# where c = cost/risk_pct (risk.atr_stop.cost_ratio). At X = 2.0 and the median
# risk_pct measured at the 4H setup tier and recorded above (BTC 1.968% -> c
# 0.0711, ETH 2.679% -> 0.0523, SOL 3.752% -> 0.0373), that is a GROSS floor of
# 2.2134 / 2.1568 / 2.1119. walkforward.py records that the MINIMUM planned
# gross R:R across every trade the engine ever took was 1.56 -- whose net_rr is
# 1.3900 / 1.4329 / 1.4679. This floor is therefore expected to reject the large
# majority of plans. That is measured (cli graph-backtest --rr-report),
# reported, and NOT a licence to weaken it: derived from costs, never fitted to
# returns, exactly as ATR_STOP_MULTIPLE was.
RR_TARGET_MIN = 2.0


def date_to_ms(date_str: str) -> int:
    """Parse a %Y-%m-%d date string as UTC midnight and return epoch milliseconds.

    Args:
        date_str: ISO date string in format %Y-%m-%d (e.g., "2023-01-01").

    Returns:
        Epoch milliseconds since Unix epoch, always in UTC.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# v0.3.0 Phase 1: validation integrity. Nothing here changes the strategy —
# these constants govern how it is MEASURED.
# ---------------------------------------------------------------------------
# Second SQLite DB for framework state (trial ledger; later review records /
# strategy versions / populations). Deliberately NOT ohlcv.db: a corrupt
# experiment log must never endanger 117 MB of irreplaceable price history.
STATE_DB_PATH = "data/state.db"

# Buy-and-hold null (KNOWN-LIMITATIONS §0: the old gate compared against ZERO,
# so it "could have blessed a strategy worse than inaction").
# BENCHMARK_REBALANCE is "daily" because that is what MEASURED 2026-07-27
# reproduces §0's published basket (2.1628x / +29.32% / 0.7284 / 64.32% DD).
# "none" (buy-once-hold) measures 2.0809x / 0.6998 / 66.91% and matches none of
# the four published figures; it stays reachable so the choice remains a
# measurement rather than a belief.
BENCHMARK_TIMEFRAME = "1d"  # returns tier; must exist in storage.TIMEFRAME_MS
BENCHMARK_REBALANCE = "daily"  # "daily" | "none"
BENCHMARK_CHARGE_FEES = True  # one round-trip; NO funding — see benchmark.py docstring

# MEDIUM-5 (KNOWN-LIMITATIONS §3). "spread" books each trade's pnl_pct evenly
# across the UTC days it was open, entry_ts -> exit_ts inclusive. "exit_day" is
# v0.2.0's behavior, kept reachable so the moment delta is MEASURED, not
# asserted. Measured on the 23 OOS trades: kurtosis 31.2449 -> 15.5429, skew
# 3.7883 -> 1.4034, annualized Sharpe 1.1750 -> 1.5120 (the Sharpe RISES: the
# exit-day series' zero/spike alternation inflates its own variance).
PNL_ATTRIBUTION_MODE = "spread"  # "spread" | "exit_day"

# ---------------------------------------------------------------------------
# v0.3.0 Phase 3: plug-in framework core. Reserved prefixes STRATEGY_DIR /
# FRAMEWORK_* per the v0.3.0 shared architecture contract §7.
#
# None of these is a strategy parameter and none may ever appear in a
# walk-forward grid or a ParamSpec: they configure where graphs live and
# whether memoization is on, not what a strategy does.
# ---------------------------------------------------------------------------
STRATEGY_DIR = "data/strategies"  # serialized StrategyGraphs: <name>.strategy.json
# The package framework.registry.load_all() walks. A string (not the module
# object) so tests can point load_all at a throwaway package.
FRAMEWORK_PLUGIN_PACKAGE = "trading_bot.plugins"
# Kill switch for the framework's indicator/candidate memo. Exists as a
# MEASUREMENT tool, not a tuning knob: a graph backtest run with caching off
# must be bit-identical to one run with it on, and that is asserted in
# tests/test_framework_parity.py. If the two ever differ, the cache key is
# missing a parameter and every cached result since is suspect.
FRAMEWORK_CACHE_ENABLED = True

# ---------------------------------------------------------------------------
# Phase 5 (v0.3.0): feedback loop. REVIEW_* are DIAGNOSTIC thresholds — they
# label a closed trade so a human or Phase 6's mutators can read the pattern.
# They never select a strategy: THE GATE is the only fitness oracle (§4). None
# is a walk-forward grid axis; none may become one.
# ---------------------------------------------------------------------------
TARGET_ANN_RETURN = 0.50  # the northstar, stated once; everything reads it from here

# None => resolve SIGNAL_TRIGGER_TIMEFRAME at CALL time (never import time), so a
# tier shift moves the review with the engine — same discipline as FADE_ENABLED.
REVIEW_TIMEFRAME = None
# Minimum fraction of expected bars present before MFE/MAE are trusted. A gap
# understates both, and an understated MAE reads as "the stop was over-wide" —
# exactly backwards.
REVIEW_MIN_BAR_COVERAGE = 0.90

REVIEW_TP_CAPTURE_GOOD = 0.50  # realized favourable / MFE at or above this = good exit
REVIEW_TP_UNREACHABLE_FRAC = 0.50  # MFE below this fraction of target distance = never approached
REVIEW_SL_SLACK_MAX = 0.50  # MAE/stop distance below this = less than half the risk used
REVIEW_SL_NEAR_MISS = 0.90  # at or above, without stopping out = the stop barely held

# Pace (KNOWN-LIMITATIONS §2: a +68% headline was a 90-day extrapolation from 23
# trades). A pace CLAIM needs BOTH floors; below them diagnose() reports
# "insufficient-sample" and emits NO refinement suggestions.
REVIEW_PACE_MIN_TRADES = 30  # deliberately equal to WF_MIN_TRADES today
REVIEW_PACE_MIN_DAYS = 180

REVIEW_STOP_DOMINANCE = 0.50  # fraction of trades exiting on the stop
REVIEW_TIME_DOMINANCE = 0.33  # fraction exiting on the time stop
REVIEW_DEAD_WEIGHT_COVERAGE = 0.98  # a Confirmation passing on ~every trade carries no
# information (§0c: volume is computed but gates nothing)

REVIEW_REFINE_STEP_FRAC = 0.25  # one bounded step, as a fraction of a ParamSpec's range
REVIEW_FORWARD_MIN_DAYS = 30  # a shorter forward window is refused, not reported
REVIEW_LEGACY_VERSION_ID = "legacy-engine"

# ---------------------------------------------------------------------------
# v0.3.0 Phase 8: pattern coverage expansion. Reserved prefixes per the v0.3.0
# shared architecture contract §7 row 8: DETECTOR_* plus per-pattern tolerance
# names prefixed by pattern (CUP_, DOUBLE_, HS_, TRIANGLE_, WEDGE_, RSI_,
# WYCKOFF_).
#
# APPEND-ONLY BY CONSTRUCTION. §7 forbids modifying an existing constant, and
# two of the prefixes below (HS_*, TRIANGLE_*) already exist further up this
# file for the v0.2.0 geometry. The NEW names live here rather than beside
# their older siblings so the diff stays purely additive and the frozen
# signals/patterns.py path keeps its measured behaviour.
#
# Every constant here is the DEFAULT of a declared ParamSpec on a detector
# plug-in, i.e. sweepable by Phase 6's mutator within declared bounds. None is
# frozen, and none was tuned against the Phase 8 edge report -- the geometry
# choices were made from the classical pattern definitions BEFORE any
# measurement (see .claude/PRPs/reports/phase8-detector-edge-report.md's
# degrees-of-freedom section).
# ---------------------------------------------------------------------------

# --- the edge report itself (a diagnostic, never a fitness oracle: §4) ------
# Below this trade count the report prints INSUFFICIENT and SUPPRESSES every
# rate and expectancy figure. KNOWN-LIMITATIONS §1 records 23 trades failing a
# floor of 30; printing an expectancy off 9 trades repeats that error one level
# down, and suppression is what stops the table being mined.
DETECTOR_MIN_EVENTS_FOR_REPORT = 20  # trades, not events
# The diagnostic REFUSES to read any span ending inside this many days of now:
# Phase 9 owns a holdout no diagnostic may see. Derived from the existing
# walk-forward windows rather than re-typed, so there is one definition of
# "reserved tail". FOLLOW-UP: repoint at Phase 9's HOLDOUT_* when it lands.
DETECTOR_REPORT_HOLDOUT_GUARD_DAYS = WF_OOS_DAYS + WF_TEST_DAYS  # 150 days
# Trial-ledger campaign the report writes its own rows under (§4.2: nothing
# scores a candidate without a ledger handle, and this report is not exempt).
DETECTOR_REPORT_CAMPAIGN = "detector-report"

# --- RSI (Wilder), indicators/rsi.py ---------------------------------------
RSI_PERIOD = 14  # bars in the Wilder-smoothed gain/loss average

# --- cup & handle (tier 1; no donor exists anywhere in this repo) -----------
CUP_MIN_WIDTH_BARS = 20  # setup-tier bars from left rim to right rim, minimum
CUP_MAX_WIDTH_BARS = 90  # ... and maximum
CUP_MIN_DEPTH = 0.08  # cup depth as a fraction of the higher rim; under this is noise
CUP_MAX_DEPTH = 0.50  # over this is a crash with a bounce, not a cup
CUP_RIM_TOLERANCE = 0.05  # max relative height difference between the two rims
CUP_ROUND_BAND = 0.25  # fraction of depth defining the "base band" near the bottom
CUP_MIN_BASE_BARS = 5  # bars whose low sits inside that band -- the V-vs-cup test
CUP_HANDLE_MIN_BARS = 3  # bars after the right rim forming the handle, minimum
CUP_HANDLE_MAX_BARS = 20  # ... and maximum
CUP_HANDLE_MAX_RETRACE = 0.40  # handle may give back at most this fraction of depth

# --- double top / double bottom (tier 2) -----------------------------------
DOUBLE_TOLERANCE = 0.02  # max relative height difference between the two extremes
DOUBLE_MAX_GAP_BARS = 60  # max setup-tier bars between them (at 4h, 10 days)
DOUBLE_MIN_SEPARATION_BARS = 8  # min setup-tier bars between them
DOUBLE_MIN_TROUGH_DEPTH = 0.03  # intervening dip, as a fraction of the mean extreme
# v0.3.2 correctness gates (code review 2026-07-29): the three additions below
# are what turns "two comparable pivots with something between them" into the
# textbook shape. Each is a ParamSpec default, sweepable, chosen from the
# classical pattern definition rather than tuned to preserve v0.3.0/v0.3.1
# output -- moving detection counts is the point, not a regression.
DOUBLE_MAX_TROUGH_DEPTH = 0.20  # sibling ceiling to MIN_TROUGH_DEPTH: above this
# fraction the intervening swing is a trend leg, not a bump -- see reversal.py's
# D3 rejection test for the fixture (+39% / 32-bar swing) that motivated it.
DOUBLE_DOMINANCE_TOL = 0.001  # bar `high`/`low` may exceed either extreme by at
# most this fraction before the "nothing more extreme in between" guard fires;
# small enough to absorb float noise only, not a real intervening spike.
DOUBLE_PRIOR_TREND_LOOKBACK_BARS = 20  # setup bars looked back from the FIRST
# extreme to measure the move the pattern claims to reverse.
DOUBLE_PRIOR_TREND_MIN_MOVE = 0.05  # min fractional decline/advance into the
# first extreme over that lookback, as a fraction of the reference price --
# below this there is no prior trend to reverse.

# --- head & shoulders refinements (tier 1) ---------------------------------
# All three default to the v0.2.0 behaviour -- False / 1.0 permit everything --
# which is exactly what makes the H&S parity test possible: at defaults the
# refined detector must reproduce signals/patterns.py bit-for-bit.
HS_NECKLINE_SLOPED = False  # True fits the neckline through both troughs
HS_TIME_SYMMETRY_TOL = 1.0  # |(head-left)-(right-head)| / (right-left); 1.0 = no-op
HS_VOLUME_TAPER_REQUIRED = False  # True demands right-shoulder volume < left's

# --- triangle / wedge slope classification (tier 2) ------------------------
# On the NORMALISED per-bar slope ((y2-y1)/((x2-x1)*y1)), so 0.001 means 0.1%
# of price per bar and the tolerance is comparable across symbols priced from
# $0.10 to $100k. A raw slope tolerance is meaningless across a universe.
TRIANGLE_FLAT_SLOPE_TOL = 0.001  # |slope| <= this counts as a FLAT boundary
WEDGE_MIN_SLOPE = 0.001  # both boundaries must exceed this, same sign, for a wedge

# --- RSI divergence (tier 2) ----------------------------------------------
RSI_DIV_LOOKBACK_BARS = 120  # setup-tier bars scanned for the pivot pair
RSI_DIV_MIN_SEPARATION_BARS = 6  # min bars between the two price pivots
RSI_DIV_MAX_SEPARATION_BARS = 60  # ... and maximum
RSI_DIV_PIVOT_MATCH_BARS = 3  # an RSI pivot must sit within this many bars of a price pivot
# Deliberately NOT 70/30: pairing two RSI highs demands the FIRST be elevated,
# not extreme, and a 70 floor eliminates most real pairs. Both are ParamSpecs
# with bounds so the choice stays testable rather than assumed.
RSI_DIV_OVERBOUGHT = 60.0  # first RSI high must be at least this
RSI_DIV_OVERSOLD = 40.0  # first RSI low must be at most this

# --- Wyckoff spring / upthrust (the buildable subset; see A1/A2) -----------
WYCKOFF_RANGE_BARS = 30  # trailing setup-tier bars defining the range (probe bar excluded)
WYCKOFF_RANGE_MAX_WIDTH_PCT = 0.08  # (hi-lo)/close ceiling -- a real consolidation
WYCKOFF_PROBE_MIN_PCT = 0.003  # probe must pierce the level by at least this fraction
WYCKOFF_PROBE_VOL_RATIO = 1.5  # probe volume / trailing mean; a HARD condition here

# ---------------------------------------------------------------------------
# v0.3.0 Phase 6: evolution engine. Reserved prefix EVO_* (contract §7).
#
# THE GATE IS THE ONLY FITNESS ORACLE (contract §4). Nothing in this block may
# soften trial counting, and no constant here is a gate threshold, a cost, or a
# regime parameter — those live above and are frozen. These values size and
# seed a SEARCH; they do not change what a strategy does.
# ---------------------------------------------------------------------------
EVO_TRAIN_START = "2023-07-27"  # FROZEN: first date with a non-"uncertain" regime
                                # label (REGIME_MIN_BARS = 207 daily warmup bars
                                # from BACKFILL_START). Earlier windows trade zero.
EVO_TRAIN_END = "2026-01-26"    # FROZEN CEILING, never None and never "now": an
                                # implicit end would swallow Phase 9's holdout.
                                # MEASURED 2026-07-27: the last CLOSED 1d bar
                                # opens 2026-07-26 (the 2026-07-27 bar is still
                                # forming), so this reserves 181 days that no
                                # generation may score. When Phase 9 lands
                                # HOLDOUT_START_MS the ceiling becomes min(both).
EVO_WINDOW_DAYS = 540           # >= WF_TRAIN+TEST+OOS (180+60+90 = 330), or
                                # walk_forward_pooled raises. At 540: 4 folds +
                                # a 90-day window-OOS, and (914-540) = 374 days
                                # of start jitter inside the training span.
EVO_WINDOW_JITTER = True        # partial-data training (pivot guide method §2).
                                # False pins every generation to the LATEST legal
                                # window, which is reproducible but trains on one
                                # regime.
EVO_POPULATION = 24             # MEASURED, NOT GUESSED. From `cli evolve
                                # --calibrate --repeats 3 --seed-graph
                                # data/strategies/thin-slice.strategy.json` on
                                # 2026-07-27, M1 8-core, 3 symbols, 540-day
                                # window: cold 18.15 s, warm-SAME-graph 2.48 s,
                                # warm-DISTINCT-graph 7.78 s, 6 workers, 8.0 h
                                # -> capacity 22209 evaluations.
                                # 24 is the FLOOR the tournament needs
                                # (8 * EVO_TOURNAMENT_K), not the capacity.
EVO_GENERATIONS = 8             # 24 x 8 = 192 evaluations against a 22209
                                # capacity. SIZED DOWN ON PURPOSE: capacity is a
                                # wall-clock bound and every evaluation is also a
                                # DSR trial (contract §4). Going to 22209 trials
                                # would raise the expected-max-Sharpe bar by
                                # ~sqrt(log N) and buy nothing else — the honest
                                # levers are more independent observations, not a
                                # bigger search. Raising either constant is a
                                # consumed degree of freedom and must be logged.
                                # NOTE the same-graph figure (2.48 s) measures the
                                # candidate memo, not a campaign: every member of
                                # a real generation is a different graph. Sizing
                                # on it would overstate throughput ~3.1x.
EVO_BUDGET_HOURS = 8.0          # overnight wall-clock budget for the sizing math
EVO_WORKERS = max(1, (os.cpu_count() or 4) - 2)  # 6 here; mirrors
                                # scripts/bruteforce/runner.py's default. The M1's
                                # 8 cores are 4 performance + 4 efficiency, so
                                # throughput does not scale linearly past ~4-6.
EVO_ELITES = 2                  # carried unchanged but RE-SCORED, so still
                                # charged a trial: carrying an elite is not free
EVO_TOURNAMENT_K = 3            # k-way tournament, sampled WITH replacement
EVO_GRAPH_EDIT_SHARE = 0.35     # share of offspring bred by graph-edit vs jitter
EVO_JITTER_NODES = 1            # nodes touched per param-jitter mutation
EVO_JITTER_SIGMA = 0.15         # gaussian sigma as a fraction of a bound width
EVO_BOOL_FLIP_P = 0.25          # probability a bool param flips
EVO_DEDUP_MAX_REDRAWS = 8       # duplicate graph_hash -> redraw, then accept + log
EVO_MIN_UNIQUE_FRACTION = 0.5   # below this the diversity guard fires
EVO_FINALISTS = 5               # audit-round size, plus the seed (A3)
EVO_STRICT_MUTATORS = True      # a Mutator that cannot produce a VALID graph is
                                # a BUG: abort the campaign. Never silently skip.
EVO_DB_RETRIES = 5              # ledger write retries on transient errors
EVO_DB_RETRY_SLEEP_S = 0.25     # linear backoff base
EVO_DB_BUSY_TIMEOUT_MS = 5000   # per-connection PRAGMA in each worker
# Deliberately ABSENT: any drawdown-penalty coefficient. Sharpe is already
# risk-adjusted and drawdown enters through the gate tier; a weight would be one
# more fitted degree of freedom for no measured benefit (same reasoning as
# ATR_STOP_MULTIPLE's "k was NOT adjusted" note above).

# ---------------------------------------------------------------------------
# v0.3.0 Phase 7: builder UI. Reserved names UI_HOST / UI_PORT (contract §7).
# Exactly two constants — everything else the UI tunes (SSE poll interval,
# heartbeat, run-dir name, log caps, Tier A concurrency, body size cap) is a
# MODULE constant in ui/api.py or ui/server.py, never here, so no third name
# collides with another phase's reserved prefix.
#
# Stack: Python stdlib http.server.ThreadingHTTPServer + one vanilla
# HTML/CSS/JS page under ui/static/, zero new dependencies, SSE for progress
# (contract §10 row 6 — FastAPI+uvicorn, React/Vite and Streamlit were
# considered and rejected: this machine has no frontend build today and
# pandas-ta's disappearance from PyPI is the standing lesson about what a new
# dependency costs).
#
# UI_HOST is LOOPBACK ONLY AND NOT SWEEPABLE: this machine holds the trading
# logic and an irreplaceable 117 MB price store, the server has no auth, and
# ui.server refuses a non-loopback host with no override flag — deliberately,
# so a future edit cannot quietly expose it to a LAN.
# ---------------------------------------------------------------------------
UI_HOST = "127.0.0.1"
UI_PORT = 8770  # misses the usual dev ports (3000/5000/8000/8080)


# ---------------------------------------------------------------------------
# v0.3.0 Phase 9: the walk-forward CAMPAIGN protocol. Reserved prefixes
# CAMPAIGN_* / HOLDOUT_* (contract §7).
#
# EVERY VALUE IN THIS BLOCK IS PRE-REGISTERED: it is committed before the
# campaign runs so the result cannot be rationalised afterwards. Changing any
# of them after a holdout has been consumed does not produce a better verdict,
# it produces a different (and no longer clean) experiment — see
# campaign.holdout_is_consumed() and the --force-holdout-rerun audit path.
# ---------------------------------------------------------------------------

# THE HOLDOUT. Never seen by any evolution generation (contract §4.4).
#
# START — DEVIATION FROM THE PHASE 9 PLAN, AND THE MOST IMPORTANT NUMBER HERE.
# The plan declared 2026-01-01 (1767225600000). That would have been a
# LOOKAHEAD BUG: Phase 6 froze EVO_TRAIN_END = "2026-01-26" and its ledger
# proves generations scored bars right up to that ceiling. MEASURED:
#   sqlite3 data/phase6-state.db "select max(end_ms), sum(end_ms>1769385600000)
#     from trial_ledger"                       -> 1769385600000 | 0
# i.e. 397 ledger rows, max end_ms exactly 2026-01-26, ZERO rows past it. So
# the genuinely unseen span begins at 2026-01-26, not 2026-01-01; the plan's
# value would have handed evolution 25 days of "holdout" it had already
# scored. The holdout start is therefore pinned to Phase 6's frozen ceiling.
#
# END — EXCLUSIVE UPPER BOUND, and it is the last bar that is both CLOSED and
# COMPLETE, not MAX(ts). MEASURED 2026-07-28T01:09Z:
#   1d MAX(ts) = 1785110400000 = 2026-07-27T00:00:00Z
#   ... but the poller last ran at 2026-07-27T06:00Z (1h MAX(ts) =
#   2026-07-27T06:00:00Z), so that daily bar was UPSERTED WHILE FORMING and
#   holds only 7 hours of trade: BTCUSDT 1d 2026-07-27 volume 25535.7 against
#   the previous day's 41197.3, and its high/low (65722.5 / 64872.0) are
#   exactly the max/min of the seven 1h bars 00:00-06:00 that exist for that
#   day. It is a PARTIAL BAR persisted in the store.
# The last complete daily bar therefore OPENS 2026-07-26 and CLOSES
# 2026-07-27T00:00:00Z, so the exclusive end of trustworthy daily history is
# 1785110400000. Deriving this from MAX(ts) read as closed would put a
# 7-hour stub day in the verdict — a lookahead bug that flatters.
#
# Consequence for every consumer: pass HOLDOUT_END_MS as the exclusive end_ms
# of a half-open span [start, end). storage.load_candles' bounds are
# INCLUSIVE (contract §1), so a direct load_candles call for the holdout must
# use HOLDOUT_END_MS - 1, never HOLDOUT_END_MS.
HOLDOUT_START_MS = 1769385600000  # 2026-01-26T00:00:00Z, INCLUSIVE
HOLDOUT_END_MS = 1785110400000  # 2026-07-27T00:00:00Z, EXCLUSIVE
# 182 days. Asserted, never assumed:
#   HOLDOUT_END_MS - HOLDOUT_DAYS * 86_400_000 == HOLDOUT_START_MS
# The plan declared 207 days on its (wrong) 2026-01-01 start. 182 is what the
# corrected barrier leaves, and it is NOT a chosen number: both endpoints are
# forced — the start by Phase 6's frozen ceiling, the end by the last complete
# daily bar. Lengthening the holdout is the only honest lever on the required
# Sharpe, and it is already at its maximum given those two constraints.
# MEASURED consequence (no price data involved, so not a peek), via
# equity.deflated_sharpe on 182 daily observations at the post-MEDIUM-5
# moments (skew 1.4034, kurt 15.5429): dsr > 0.95 needs annualised Sharpe
# 2.20 at n_trials=1, 6.00 at 198, 6.38 at 417.
HOLDOUT_DAYS = 182
# A committed tripwire: campaign.py refuses the holdout stage when False.
# Flipping it is a recorded decision, not a convenience.
HOLDOUT_LOCKED = True

# Evolution span start: 2023-07-27 == date_to_ms("2023-07-27"). NOT
# BACKFILL_START — REGIME_MIN_BARS = 207 daily warmup bars means the first
# non-"uncertain" regime label lands ~2023-07-27, so earlier windows produce
# zero trades and would silently dilute every fold. Identical to
# EVO_TRAIN_START, deliberately: the campaign and the evolution it drives must
# not disagree about where usable history begins.
# [CAMPAIGN_EVOLVE_START_MS, HOLDOUT_START_MS) is 914 days -> 12 folds at
# WF_TRAIN_DAYS=180 / WF_TEST_DAYS=60 (asserted, not assumed, in campaign.py).
CAMPAIGN_EVOLVE_START_MS = 1690416000000

# The symbol set for the HOLDOUT GATE is FROZEN before the run. Sourced from
# Phase 2's measured, gap-verified, low-correlation selection (9 symbols,
# Kish effN 1.834) — never from SYMBOLS (3 correlated majors, effN 1.193).
# NOTE, stated in advance because it cuts both ways: Phase 6's evolution ran on
# SYMBOLS (3), so the gate evaluates the champion on a WIDER universe than it
# was selected on. That makes per_symbol_expectancy strictly harder (9-of-9,
# six of them never in any training pool) and the raw sample larger. Both
# effects are reported; neither is adjusted for.
CAMPAIGN_SYMBOLS: tuple[str, ...] = RESEARCH_SYMBOLS

# Phase 9's OWN evolution top-up, deliberately TINY. The substantive search is
# Phase 6's completed campaign (192 candidates + 6 audit = 198 evaluations,
# ZERO of which reached tier A). Contract §4 names exactly two honest levers:
# more independent observations, and A SMALLER PRE-REGISTERED SEARCH. The
# search-size lever points DOWN. Launching a second few-hundred-evaluation
# search would raise the Sharpe the DSR demands (6.00 -> 6.4+ annualised) in
# exchange for candidates drawn from a pool Phase 6 already measured to be
# barren, so Phase 9 adds only enough to exercise the stage under real
# conditions and charges the union to the DSR.
CAMPAIGN_POPULATION_SIZE = 6  # >= EVO_TOURNAMENT_K (3), or a bracket cannot be sampled
CAMPAIGN_GENERATIONS = 2
CAMPAIGN_WALL_CLOCK_BUDGET_HOURS = 1.0
# Stopping rule, declared in advance so "stop when it looks good" is
# impossible: stop at the FIRST of (a) CAMPAIGN_GENERATIONS reached,
# (b) budget exhausted, (c) no improvement in best fitness for
# CAMPAIGN_PATIENCE_GENERATIONS consecutive generations.
CAMPAIGN_PATIENCE_GENERATIONS = 5
CAMPAIGN_SEED = 20260728  # one seed for the whole campaign, recorded in state.db
CAMPAIGN_CHECKPOINT_EVERY = 1  # generations between checkpoints; 1 = every one

# THE PRE-REGISTERED TRADE FLOOR FOR CHAMPION ELIGIBILITY (contract §0a's
# trade-floor analysis, and the open issue Phase 6 found and deliberately did
# not fix).
#
# THE PROBLEM, MEASURED BY PHASE 6: fitness is `excess_sharpe`, which rewards
# NOT TRADING. Tier B averaged 14.9 trades against tier C's 20.1, and
# generation 7's winner had ONE trade at Sharpe 4.608 and still ranked tier B.
# `sample_adequacy` catches this at verdict time but selection does not, so the
# population drifts toward degenerate low-trade graphs. Phase 6 declined to
# retune fitness after observing that, correctly: tuning a search
# hyper-parameter against an observed outcome is an unpriced degree of freedom.
#
# THE DECISION, fixed before this campaign ran and before any member trade
# count was queried: a member is ELIGIBLE to be champion only if it recorded
# at least WF_MIN_TRADES trades on its own evaluation window.
#   * The number is NOT NEW. It is the gate's existing sample floor, so this
#     introduces no fitted quantity. Any other value (15, or 30/effN) would be
#     a fresh number chosen with Phase 6's results already in view — exactly
#     the unpriced degree of freedom above.
#   * It is applied at CHAMPION SELECTION, not inside evolution's fitness.
#     Changing fitness would change the search dynamics of a search already
#     performed, which is unrepeatable; filtering at selection is inspectable,
#     reversible, and consumes no additional evaluation.
#   * Justification in one line: a candidate that cannot clear the gate's own
#     sample floor on the window it was SELECTED on cannot honestly be put
#     forward as the campaign's champion — the gate would reject it on
#     sample_adequacy anyway, and the verdict would then be about the selection
#     rule rather than about the strategy.
#   * DECLARED FALLBACK, so this can never become a reason to lower a
#     threshold: if NO member clears the floor, the highest-fitness member is
#     still put through the holdout, the report is stamped
#     CHAMPION_BELOW_TRADE_FLOOR, and sample_adequacy is expected to FAIL.
#     The floor is never lowered and the holdout is never shrunk
#     (KNOWN-LIMITATIONS §4 forbids both by name).
CAMPAIGN_MIN_CHAMPION_TRADES = WF_MIN_TRADES

# Reported beside the gate, NOT a gate condition: contract §4's seven
# GATE_CONDITIONS contain no return threshold, while the PRD's northstar is
# >50% annualised OOS. Referencing Phase 5's constant keeps one source of truth
# rather than a second literal 0.50.
CAMPAIGN_NORTHSTAR_ANN_RETURN = TARGET_ANN_RETURN

# Where the verdict report and the v0.3.0 limitations document are written.
CAMPAIGN_REPORT_DIR = ".claude/PRPs/reports"
