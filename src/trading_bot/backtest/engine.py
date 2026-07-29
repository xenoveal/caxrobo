"""
Deterministic bar-by-bar backtest engine over the Phases 2-4 pipeline.

Replays history reusing the EXACT production components — classify_series,
detect_donchian_setups / detect_fade_setups, check_breakout, build_signal /
build_fade_signal — so backtest and live behavior cannot drift apart.

Regime -> method dispatch, identical to signals/scan.py:
  - trending           -> Donchian channel breakout method (Phase 5)
  - ranging            -> mean-reversion fade method, subject to
                          config.FADE_ENABLED (Phase 6 kill switch; read at
                          call time so live and backtest suppress together)
  - extreme-volatility -> no method
  - uncertain          -> no method

No-lookahead guarantees (stated in tier roles, not in fixed timeframes — the
tiers are config.REGIME_TIMEFRAME / SIGNAL_PATTERN_TIMEFRAME /
SIGNAL_TRIGGER_TIMEFRAME, currently 1d / 4h / 1h):
  - Regime labels come from classify_series, whose indicators use only
    trailing windows; the label applied at time t is the last REGIME bar
    CLOSED by t.
  - Candidates at a TRIGGER bar use only SETUP bars closed by that bar's
    close, and pivots are only confirmed with PIVOT_SPAN closed bars after
    them (same as live). Coarser tiers do not change this: regime and setup
    bar closes fall exactly on trigger-bar boundaries, so searchsorted's
    side="right" includes only bars genuinely closed at or before the trigger
    close.
  - The TRIGGER slice ends at the bar under evaluation.

Simulation rules:
  - One open trade per symbol at a time (the human executes alerts serially).
    When several candidates trigger on the same bar, the one taken is chosen by
    signals.setup.rank_signals — the same ranking live scanning presents — so
    measured performance is not an artifact of candidate iteration order.
  - Entry at the trigger bar close; exits evaluated from the NEXT bar.
  - Conservative same-bar rule: if a bar touches both stop and target, the
    stop is assumed to fill first.
  - Time-stop after MAX_HOLD_BARS_TRIGGER trigger bars; unresolved trades at
    data end are closed at the last close (outcome "end").
  - Costs: fee + slippage per side, deducted as 2*(FEE_PCT+SLIPPAGE_PCT), plus
    a funding term (FUNDING_PCT_PER_DAY * holding days) scaled by how long the
    trade was open (Phase 2: honest cost model).
  - Exits (Phase 5, DONCHIAN TRADES ONLY): the initial stop is the Signal's
    ATR stop. A Donchian trade exits on a touch of the trailing opposite
    DONCHIAN_ENTRY_PERIOD channel, and — only when params.trail_enabled — on a
    ratchet trail at (extreme since entry -/+ params.trail_atr_multiple *
    ATR(setup TF)) that never loosens. The measured-move target is an exit only
    when params.target_enabled; the reward:risk screen in setup.build_signal
    uses that target either way. Priority on a bar that touches several levels:
    stop/trail, then channel, then target (conservative). The ratchet is applied
    only after the bar's exits are resolved, so the trail can never fire on the
    same bar that produced its own extreme.

    The trail and target both DEFAULT OFF. With ATR_STOP_MULTIPLE shared
    between entry stop and trail, the trail closed 91.3% of trades at a median
    10-hour hold and preempted the channel exit entirely (removing the channel
    exit produced a bit-identical backtest). See config.py's exit-management
    block and .claude/PRPs/reports/code review/phase4-7-code-review.md (HIGH-2).

    Fade (ranging-regime) trades are DELIBERATELY EXCLUDED from the trail and
    channel exits — see the MANDATORY DEVIATION note at the exit loop.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot import config
from trading_bot.data import storage
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.regime.classifier import classify_series
from trading_bot.signals.breakout import check_breakout
from trading_bot.signals.donchian import (
    DONCHIAN_KIND,
    channel_exit_levels,
    detect_donchian_setups,
)
from trading_bot.signals.meanrev import (
    _to_trigger_candidate,
    build_fade_signal,
    detect_fade_setups,
)
from trading_bot.signals.setup import build_signal, rank_signals

logger = logging.getLogger("trading_bot")


@dataclass(frozen=True)
class BacktestParams:
    """Sweepable strategy parameters (defaults = the unvalidated config values)."""

    adx_trend_threshold: float = config.ADX_TREND_THRESHOLD
    atr_extreme_percentile: float = config.ATR_EXTREME_PERCENTILE
    bb_num_std: float = config.BB_STD
    rr_floor: float = config.RR_FLOOR
    # Phase 5 exit modes, now sweepable rather than hardcoded. trail_enabled
    # and target_enabled are walk-forward grid axes; trail_atr_multiple is
    # INDEPENDENT of ATR_STOP_MULTIPLE so the trail can never again be pinned
    # to the entry-stop distance. See config.py's exit-management block and
    # .claude/PRPs/reports/code review/phase4-7-code-review.md (HIGH-2).
    trail_enabled: bool = config.TRAIL_ENABLED
    trail_atr_multiple: float = config.TRAIL_ATR_MULTIPLE
    target_enabled: bool = config.DONCHIAN_TARGET_ENABLED


@dataclass(frozen=True)
class Trade:
    """One simulated round-trip.

    pnl_pct is net of costs, in fraction-of-entry terms (equal-sized trades).
    outcome is one of "stop", "trail", "channel", "target", "time", "end".
    "trail" and "channel" only occur for Donchian (trending) trades — see the
    engine's exit-loop MANDATORY DEVIATION note. stop keeps the trade's
    INITIAL stop (a frozen record of the setup), not the ratcheted level the
    trade may have exited at.

    THE LAST THREE FIELDS ARE THE AUDIT TRAIL, appended by v0.3.0 Phase 4 and
    populated ONLY on the graph path (framework/execute.py). run_backtest leaves
    all three at their defaults, so the legacy path is bit-identical and
    tests/test_framework_parity.py compares the SIMULATION fields explicitly
    rather than whole dataclasses (contract §5).

    planned_rr is the NET, cost-adjusted ratio from risk.atr_stop.net_rr — NOT
    the gross Signal.rr / PositionPlan.rr. The two are easy to confuse and mean
    different things: gross reward_pct/risk_pct is dimensionless and cost-blind,
    while net charges round-trip cost to both legs. filter.rr-after-costs gates
    on the net number, so this is the value that answers "did this trade meet the
    >=1:2-after-costs requirement".

    confirmations is not redundant with the graph, even though a hard-gating
    Confirmation means a trade exists only if every one passed: it survives
    serialization, it distinguishes graphs after Phase 6 mutates node sets, and
    it stays correct when a future Confirmation is advisory (passed=True always,
    score varying). Sorted, so the value is deterministic across runs — Phase 6
    hashes graph results.
    """

    symbol: str
    regime: str
    pattern: str
    direction: str
    entry_ts: int
    entry: float
    stop: float
    target: float
    exit_ts: int
    exit_price: float
    outcome: str
    pnl_pct: float
    volume_high: bool
    # v0.3.0 Phase 4, APPENDED WITH DEFAULTS per the shared architecture contract
    # §5. Existing fields keep their positions and meanings: never reorder, never
    # rename. Both keyword construction sites in the tests omit these, and
    # engine.run_backtest's own close_out leaves them at their defaults.
    planned_rr: float = 0.0  # R:R AFTER COSTS at entry; 0.0 on legacy-engine trades
    confirmations: tuple[str, ...] = ()  # registry keys that PASSED, sorted
    strategy_version: str = ""  # set by Phase 5's version registry; empty here
    # v0.3.2 WS-B (D4), APPENDED WITH DEFAULTS per the same rule the block above
    # establishes: GRAPH-PATH ONLY, defaults are the "no geometry" case, and
    # run_backtest never sets them, so test_framework_parity.py stays untouched
    # (PARITY_EXACT/PARITY_CLOSE do not name these fields, exactly as they
    # already omit planned_rr/confirmations/strategy_version).
    #
    # These four carry the DetectedEvent's own structural facts — not a
    # re-derivation, not a redraw-time guess — through to the replay chart, so
    # evoDrawTrade (app.js) can draw the pattern that was actually detected
    # instead of the hardcoded 20-bar/34px box it drew before (D4). start_ts/
    # end_ts/level are copied verbatim from the winning branch's DetectedEvent
    # (framework/contracts.py); pattern_meta is that event's `meta` mapping,
    # frozen into a sorted tuple of pairs because Trade is a frozen dataclass
    # and dict is unhashable. 0 / 0.0 / () are the "no geometry" sentinel the
    # UI must render WITHOUT a zone (D4's "degrade honestly" requirement) —
    # epoch-ms timestamps are never legitimately 0, so pattern_start_ts == 0
    # is an unambiguous "not populated" test, cheaper than a sentinel None on
    # a field two other int fields already use 0 as a real value for.
    pattern_start_ts: int = 0
    pattern_end_ts: int = 0
    pattern_level: float = 0.0
    pattern_meta: tuple[tuple[str, float], ...] = ()


# Memo of per-(symbol, tier) quantities that are IDENTICAL across grid combos.
#
# The walk-forward grid varies only exit parameters and the hold limit, none of
# which touch an indicator — yet every run_backtest recomputed classify_series,
# Wilder ATR, the exit channels, and a Wilder ADX per setup bar inside
# detect_donchian_setups. With ~870 run_backtest calls per pooled walk-forward
# that dominated runtime completely. Entries are keyed by a CONTENT fingerprint
# (see _fingerprint), so a different series that happens to share a symbol and
# timeframe — a re-backfill, or two test fixtures on separate databases — cannot
# be served a stale value. Parameters that DO change a cached value are part of
# its key.
_CACHE: dict = {}


def clear_caches() -> None:
    """Drop every memoized series.

    Only needed when stored candles are mutated inside a live process (the
    fingerprint already covers ordinary cases). Cheap and always safe to call.
    """
    _CACHE.clear()


def _fingerprint(df: pd.DataFrame) -> tuple:
    """Content fingerprint of a loaded OHLCV frame.

    Bar count and end timestamps alone are not enough: distinct fixtures
    routinely share them while holding different prices, which would collide.
    The column sums make the key sensitive to the actual values at negligible
    cost next to the indicators being cached.
    """
    ts = df.index.to_numpy()
    return (
        len(ts),
        int(ts[0]),
        int(ts[-1]),
        round(float(df["high"].sum()), 6),
        round(float(df["low"].sum()), 6),
        round(float(df["close"].sum()), 6),
    )


def _df(conn, symbol: str, timeframe: str) -> pd.DataFrame:
    rows = storage.load_candles(conn, symbol, timeframe)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if len(df):
        df["ts"] = df["ts"].astype(int)
    return df.set_index("ts")


def _assert_interval(df: pd.DataFrame, timeframe: str, symbol: str, role: str) -> int:
    """Verify a loaded series' bar spacing matches its configured timeframe.

    Every no-lookahead guarantee in this module assumes df_trig really is
    SIGNAL_TRIGGER_TIMEFRAME bars, close_setup lands on trigger-bar
    boundaries, and max_hold counts bars of the configured duration. If config
    and stored data disagree — a partial tier migration, a series backfilled
    under the wrong key, a fixture seeded at the old interval — all of them
    fail SILENTLY, in the flattering direction. Raise instead.

    Uses the MEDIAN inter-bar spacing so a handful of missing candles cannot
    trip the check; a wholesale interval mismatch always shifts the median.

    Args:
        df: Loaded OHLCV frame, epoch-ms index, ascending.
        timeframe: The config timeframe key this series was loaded under.
        symbol: For the error message.
        role: "regime" | "setup" | "trigger", for the error message.

    Returns:
        The expected interval in milliseconds (storage.TIMEFRAME_MS[timeframe]).

    Raises:
        ValueError: If the observed median spacing differs from the expected
            interval.
    """
    expected = storage.TIMEFRAME_MS[timeframe]
    ts = df.index.to_numpy()
    if len(ts) >= 2:
        observed = int(np.median(np.diff(ts)))
        if observed != expected:
            raise ValueError(
                f"{symbol} {role} series spacing is {observed} ms but config "
                f"names timeframe {timeframe!r} ({expected} ms). The tier "
                f"configuration and the stored data disagree; every "
                f"no-lookahead guarantee in this engine would be void."
            )
    return expected


def run_backtest(
    conn,
    symbol: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    params: BacktestParams | None = None,
    fee_pct: float | None = None,
    slippage_pct: float | None = None,
    funding_pct_per_day: float | None = None,
    max_hold_bars: int | None = None,
) -> list[Trade]:
    """
    Replay the regime-switched signal pipeline over stored history.

    Args:
        conn: Database connection.
        symbol: Trading pair symbol.
        start_ms: Only trigger bars closing at/after this time can trigger
            entries (default: start of data).
        end_ms: Only trigger bars closing at/before this time are simulated
            (default: end of data).
        params: Strategy parameter overrides (default BacktestParams()).
        fee_pct / slippage_pct: Per-side costs (default config values).
        funding_pct_per_day: Funding cost per day held (default
            config.FUNDING_PCT_PER_DAY).
        max_hold_bars: Time-stop in trigger bars (default
            config.MAX_HOLD_BARS_TRIGGER).

    Returns:
        List of Trade in entry-time order.
    """
    if params is None:
        params = BacktestParams()
    fee = config.FEE_PCT if fee_pct is None else fee_pct
    slip = config.SLIPPAGE_PCT if slippage_pct is None else slippage_pct
    funding = config.FUNDING_PCT_PER_DAY if funding_pct_per_day is None else funding_pct_per_day
    max_hold = config.MAX_HOLD_BARS_TRIGGER if max_hold_bars is None else max_hold_bars
    cost = 2 * (fee + slip)

    df_regime = _df(conn, symbol, config.REGIME_TIMEFRAME)
    df_setup = _df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME)
    df_trig = _df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME)
    if df_regime.empty or df_setup.empty or df_trig.empty:
        return []

    regime_ms = _assert_interval(df_regime, config.REGIME_TIMEFRAME, symbol, "regime")
    setup_ms = _assert_interval(df_setup, config.SIGNAL_PATTERN_TIMEFRAME, symbol, "setup")
    trigger_ms = _assert_interval(df_trig, config.SIGNAL_TRIGGER_TIMEFRAME, symbol, "trigger")

    fp_regime = _fingerprint(df_regime)
    fp_setup = _fingerprint(df_setup)

    # Regime labels depend on the two classifier thresholds, so both are in the key.
    labels_key = (
        "labels", symbol, config.REGIME_TIMEFRAME, fp_regime,
        params.adx_trend_threshold, params.atr_extreme_percentile,
    )
    if labels_key not in _CACHE:
        _CACHE[labels_key] = classify_series(
            df_regime,
            adx_trend_threshold=params.adx_trend_threshold,
            atr_extreme_percentile=params.atr_extreme_percentile,
        )
    labels = _CACHE[labels_key]

    close_regime = df_regime.index.to_numpy() + regime_ms
    close_setup = df_setup.index.to_numpy() + setup_ms
    atr_key = ("atr_setup", symbol, config.SIGNAL_PATTERN_TIMEFRAME, fp_setup)
    if atr_key not in _CACHE:
        _CACHE[atr_key] = wilder_atr(df_setup, period=config.ATR_STOP_PERIOD).to_numpy()
    atr_setup_vals = _CACHE[atr_key]
    # Phase 5: trailing opposite-channel exit levels, computed once on the
    # SETUP tier (df_setup) beside the ATR series above. Donchian trades only
    # — see the MANDATORY DEVIATION note at the exit loop.
    chan_key = ("exit_ch", symbol, config.SIGNAL_PATTERN_TIMEFRAME, fp_setup)
    if chan_key not in _CACHE:
        ch = channel_exit_levels(df_setup)
        _CACHE[chan_key] = (ch["lower"].to_numpy(), ch["upper"].to_numpy())
    exit_lower, exit_upper = _CACHE[chan_key]
    ts_trig = df_trig.index.to_numpy()
    close_trig = ts_trig + trigger_ms

    start = int(close_trig[0]) if start_ms is None else start_ms
    end = int(close_trig[-1]) if end_ms is None else end_ms

    highs = df_trig["high"].to_numpy()
    lows = df_trig["low"].to_numpy()
    closes = df_trig["close"].to_numpy()

    def regime_at(t: int) -> str:
        k = int(np.searchsorted(close_regime, t, side="right")) - 1
        return str(labels.iloc[k]) if k >= 0 else "uncertain"

    # (regime, [(method, candidate), ...]) per setup bar, computed lazily once.
    #
    # Shared across run_backtest calls via _CACHE: this is the single most
    # expensive thing in the engine, because detect_donchian_setups computes a
    # Wilder ADX over a PATTERN_LOOKBACK_BARS window for every setup bar. The
    # key carries everything that can change a candidate: the classifier
    # thresholds (via the regime label that selects the method), the fade band
    # width, and FADE_ENABLED — which is read at call time, so a live flip must
    # not be served a cached pre-flip candidate list.
    cand_key = (
        "cands", symbol, config.SIGNAL_PATTERN_TIMEFRAME, fp_setup,
        params.adx_trend_threshold, params.atr_extreme_percentile,
        params.bb_num_std, bool(config.FADE_ENABLED),
    )
    cand_cache: dict[int, tuple[str, list]] = _CACHE.setdefault(cand_key, {})

    def candidates_for(h_idx: int) -> tuple[str, list]:
        if h_idx not in cand_cache:
            t = int(close_setup[h_idx])
            reg = regime_at(t)
            window = df_setup.iloc[max(0, h_idx + 1 - config.PATTERN_LOOKBACK_BARS) : h_idx + 1]
            cands: list = []
            if reg == "trending":
                cands = [("donchian", c) for c in detect_donchian_setups(window)]
            elif reg == "ranging" and config.FADE_ENABLED:
                cands = [
                    ("fade", c)
                    for c in detect_fade_setups(window, num_std=params.bb_num_std)
                ]
            cand_cache[h_idx] = (reg, cands)
        return cand_cache[h_idx]

    trades: list[Trade] = []
    open_trade: dict | None = None

    def close_out(j: int, price: float, outcome: str) -> None:
        nonlocal open_trade
        s = open_trade["signal"]
        sign = 1.0 if s.direction == "long" else -1.0
        gross = sign * (price - s.entry) / s.entry
        hold_days = (int(ts_trig[j]) - s.ts) / 86_400_000.0
        funding_cost = funding * hold_days
        trades.append(
            Trade(
                symbol=symbol,
                regime=open_trade["regime"],
                pattern=s.pattern,
                direction=s.direction,
                entry_ts=s.ts,
                entry=s.entry,
                stop=s.stop,
                target=s.target,
                exit_ts=int(ts_trig[j]),
                exit_price=price,
                outcome=outcome,
                pnl_pct=gross - cost - funding_cost,
                volume_high=s.volume_high,
            )
        )
        open_trade = None

    last_j = -1
    for j in range(len(ts_trig)):
        bc = int(close_trig[j])
        if bc > end:
            break
        last_j = j

        if open_trade is not None:
            if j <= open_trade["entry_j"]:
                continue
            s = open_trade["signal"]

            # MANDATORY DEVIATION from the plan's Task 6: fade (ranging-regime)
            # trades keep EXACTLY today's stop/target/time/end behavior,
            # reading s.stop (the Signal's frozen stop) directly. Only
            # Donchian trades get the mutable trail and the opposite-channel
            # exit — see the entry-site comment and the module docstring's
            # Simulation rules block for why.
            if not open_trade["is_donchian"]:
                if s.direction == "long":
                    if lows[j] <= s.stop:
                        close_out(j, s.stop, "stop")  # conservative: stop first
                    elif highs[j] >= s.target:
                        close_out(j, s.target, "target")
                else:
                    if highs[j] >= s.stop:
                        close_out(j, s.stop, "stop")
                    elif lows[j] <= s.target:
                        close_out(j, s.target, "target")
                if open_trade is not None and j - open_trade["entry_j"] >= max_hold:
                    close_out(j, float(closes[j]), "time")
                continue

            stop = open_trade["stop"]
            # Setup-tier bar closed by this trigger bar: the opposite-channel
            # level is a trailing value, same lookup rule as the regime label.
            s_idx = int(np.searchsorted(close_setup, int(close_trig[j]), side="right")) - 1
            if s.direction == "long":
                chan = float(exit_lower[s_idx]) if s_idx >= 0 else float("nan")
                if lows[j] <= stop:
                    close_out(j, stop, "trail" if open_trade["trailed"] else "stop")
                elif not np.isnan(chan) and lows[j] <= chan:
                    close_out(j, chan, "channel")
                elif params.target_enabled and highs[j] >= s.target:
                    close_out(j, s.target, "target")
            else:
                chan = float(exit_upper[s_idx]) if s_idx >= 0 else float("nan")
                if highs[j] >= stop:
                    close_out(j, stop, "trail" if open_trade["trailed"] else "stop")
                elif not np.isnan(chan) and highs[j] >= chan:
                    close_out(j, chan, "channel")
                elif params.target_enabled and lows[j] <= s.target:
                    close_out(j, s.target, "target")
            if open_trade is not None and j - open_trade["entry_j"] >= max_hold:
                close_out(j, float(closes[j]), "time")
            # Ratchet AFTER this bar's exits are resolved: a stop derived from
            # bar j's own extreme, tested against bar j's own low, is
            # intra-bar lookahead. The trail only ever binds from bar j+1
            # onward.
            if open_trade is not None and params.trail_enabled and open_trade["atr"] > 0:
                trail_dist = params.trail_atr_multiple * open_trade["atr"]
                if s.direction == "long":
                    open_trade["extreme"] = max(open_trade["extreme"], float(highs[j]))
                    new_stop = open_trade["extreme"] - trail_dist
                    if new_stop > open_trade["stop"]:
                        open_trade["stop"] = new_stop
                        open_trade["trailed"] = True
                else:
                    open_trade["extreme"] = min(open_trade["extreme"], float(lows[j]))
                    new_stop = open_trade["extreme"] + trail_dist
                    if new_stop < open_trade["stop"]:
                        open_trade["stop"] = new_stop
                        open_trade["trailed"] = True
            continue

        if bc < start or j < 1:
            continue

        # Setup bar that had already CLOSED when this trigger bar OPENED.
        #
        # Selecting by the trigger bar's close (`bc`) instead silently discarded
        # one trigger bar in every setup window: on the last trigger bar of a
        # window, `bc` lands on the NEXT setup bar's close, so h_idx advanced to
        # a setup bar whose end_ts postdates this bar's open, and
        # check_breakout's `ts < candidate.end_ts` guard rejected it. With a 4H
        # setup and 1H trigger that was 25% of all trigger opportunities. Keying
        # on the open gives each setup bar exactly the trigger bars that follow
        # its close, and is if anything MORE conservative about lookahead: the
        # setup bar is now required to have closed before the trigger bar even
        # opened. See .claude/PRPs/reports/code review/phase4-7-code-review.md (MEDIUM-2).
        #
        # MEASURED CONSEQUENCE, recorded because it is counter-intuitive and
        # must not be silently re-broken: recovering those bars ADDS trades that
        # were unprofitable in-sample (142 -> 158 trades, pooled Sharpe 0.431 ->
        # 0.255, annualised 11.0% -> 0.25% on stored history). The old index was
        # therefore acting as an accidental trigger-recency filter, and that
        # filter was helping. The bug is still fixed rather than preserved: an
        # entry rule must be an explicit, pre-registered decision, not an
        # artifact of two code paths disagreeing about which setup bar is
        # current. If trigger recency is worth filtering on, it belongs on the
        # walk-forward grid as its own axis, priced as a degree of freedom.
        h_idx = int(np.searchsorted(close_setup, int(ts_trig[j]), side="right")) - 1
        if h_idx < 0:
            continue
        reg, cands = candidates_for(h_idx)
        if not cands:
            continue

        # Same slice shape production uses: volume window + crossing pair.
        window_trig = df_trig.iloc[max(0, j - (config.VOLUME_LOOKBACK + 1)) : j + 1]
        # lookback_bars=1: only the bar under evaluation may trigger, so the
        # entry price is always this bar's close (no stale fill).
        atr_value = (
            float(atr_setup_vals[h_idx]) if h_idx < len(atr_setup_vals) else float("nan")
        )
        bar_signals: list = []
        for method, cand in cands:
            if method == "donchian":
                event = check_breakout(window_trig, cand, lookback_bars=1, interval_ms=trigger_ms)
                sig = (
                    build_signal(symbol, cand, event, atr_value, rr_floor=params.rr_floor)
                    if event and atr_value > 0
                    else None
                )
            else:
                event = check_breakout(
                    window_trig,
                    _to_trigger_candidate(cand),
                    lookback_bars=1,
                    interval_ms=trigger_ms,
                )
                sig = (
                    build_fade_signal(symbol, cand, event, rr_floor=params.rr_floor)
                    if event
                    else None
                )
            if sig is not None:
                bar_signals.append(sig)
        # Only one trade at a time, so pick by the SAME rule live scanning ranks
        # by. Taking the first candidate instead made measured performance a
        # function of alphabetical pattern-kind order.
        if bar_signals:
            sig = rank_signals(bar_signals)[0]
            # MANDATORY DEVIATION from the plan's Task 6: the trail/channel
            # exits are gated on the winning signal being a Donchian trade
            # (sig.pattern == DONCHIAN_KIND), not applied to every trade.
            # Applying them to fade trades too would silently change the
            # ranging-regime sleeve's exit semantics, which contradicts
            # Phase 6 (measuring the fade sleeve on the premise its stop
            # placement/exits are untouched) and this module's own docstring.
            is_donchian = sig.pattern == DONCHIAN_KIND
            open_trade = {
                "signal": sig,
                "entry_j": j,
                "regime": reg,
                "is_donchian": is_donchian,
                "stop": sig.stop,          # MUTABLE for Donchian: ratchets with the ATR trail
                "atr": atr_value,          # ATR on the setup tier at entry, frozen
                "extreme": sig.entry,      # best price seen since entry (Donchian only)
                "trailed": False,          # True once the trail has moved the stop
                "h_idx": h_idx,
            }

    if open_trade is not None and last_j > open_trade["entry_j"]:
        close_out(last_j, float(closes[last_j]), "end")

    return trades
