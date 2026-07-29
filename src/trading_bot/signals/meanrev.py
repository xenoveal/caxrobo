"""
Mean-reversion fade signal method (Phase 4, ranging regime only).

Structural mirror of the Phase 3 breakout method, with fade-to-mean logic:

  1. Regime gate — active ONLY in the "ranging" regime (trending belongs to
     the pattern-breakout method; extreme-volatility and uncertain suppress
     everything).
  2. Setup on SETUP bars — a recent "stretch" bar closed outside a Bollinger band
     (below lower => long fade, above upper => short fade). The stretch must
     be within FADE_STRETCH_MAX_AGE_BARS of the latest closed bar.
  3. Trigger on TRIGGER bars — reuses check_breakout: a fade fires when the
     latest closed trigger bar freshly crosses BACK through the band (e.g. closes above
     the lower band after being below), confirming rejection of the extreme.
     Volume ratio rides along as a graded confidence input, same as Phase 3.
  4. SL/TP — stop at the excursion extreme (a new low/high beyond the stretch
     invalidates the fade thesis); target at the middle band (the mean).
  5. R:R ratio screen (Phase 2) — reward_pct / risk_pct must be >= RR_FLOOR,
     the same floor build_signal applies.
"""

import logging
import time
from dataclasses import dataclass

import pandas as pd

from trading_bot import config
from trading_bot.data import storage
from trading_bot.indicators.bollinger import bollinger
from trading_bot.regime.classifier import current_regime
from trading_bot.risk.atr_stop import net_rr
from trading_bot.signals.breakout import BreakoutEvent, check_breakout
from trading_bot.signals.patterns import PatternCandidate
from trading_bot.signals.setup import Signal, _load_df

logger = logging.getLogger("trading_bot")

FADE_KIND = "bollinger-fade"


@dataclass(frozen=True)
class FadeCandidate:
    """A band-stretch setup awaiting a trigger-bar re-cross.

    Attributes:
        direction: "long" (stretch below lower band) or "short" (above upper).
        trigger_level: Current band value the trigger bar must close back through.
        stop_level: The excursion extreme (min low / max high since the first
            stretch bar) — a close beyond it invalidates the fade.
        target: Current middle band (the mean being faded to).
        start_ts: Epoch-ms of the first stretch bar in the window.
        end_ts: Epoch-ms of the most recent stretch bar (trigger bars must not
            predate this).
    """

    direction: str
    trigger_level: float
    stop_level: float
    target: float
    start_ts: int
    end_ts: int


def detect_fade_setups(
    df: pd.DataFrame,
    *,
    period: int | None = None,
    num_std: float | None = None,
    max_age_bars: int | None = None,
) -> list[FadeCandidate]:
    """
    Scan recent SETUP bars for Bollinger band stretches.

    A bar is a stretch when its close sits outside the band computed AT that
    bar (no lookahead). At most one candidate per direction is emitted; the
    stretch window runs from the first to the most recent stretch bar within
    max_age_bars of the latest closed bar. Trigger level and target use the
    LATEST bar's band values — the levels a re-cross would actually happen at.

    Args:
        df: Setup-timeframe OHLCV DataFrame (closed bars only, ascending epoch-ms index).
        period: Bollinger period (default config.BB_PERIOD).
        num_std: Band width (default config.BB_STD).
        max_age_bars: Stretch recency bound (default config.FADE_STRETCH_MAX_AGE_BARS).

    Returns:
        List of 0-2 FadeCandidate (at most one long, one short).
    """
    if period is None:
        period = config.BB_PERIOD
    if num_std is None:
        num_std = config.BB_STD
    if max_age_bars is None:
        max_age_bars = config.FADE_STRETCH_MAX_AGE_BARS

    n = len(df)
    if n < period:
        return []

    bands = bollinger(df, period=period, num_std=num_std)
    last = bands.iloc[-1]
    if pd.isna(last["middle"]):
        return []

    # Window of recent bars eligible to be the stretch.
    window = range(max(period - 1, n - max_age_bars), n)

    out: list[FadeCandidate] = []
    closes = df["close"].to_numpy()
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    ts_vals = df.index.to_numpy()

    for direction in ("long", "short"):
        stretch_idx = [
            i
            for i in window
            if not pd.isna(bands["middle"].iloc[i])
            and (
                closes[i] < bands["lower"].iloc[i]
                if direction == "long"
                else closes[i] > bands["upper"].iloc[i]
            )
        ]
        if not stretch_idx:
            continue

        first = stretch_idx[0]
        latest = stretch_idx[-1]
        if direction == "long":
            stop_level = float(lows[first:].min())
            trigger_level = float(last["lower"])
        else:
            stop_level = float(highs[first:].max())
            trigger_level = float(last["upper"])

        target = float(last["middle"])
        # Sanity: the fade must have room between stop, trigger, and mean.
        if direction == "long" and not (stop_level < trigger_level < target):
            continue
        if direction == "short" and not (stop_level > trigger_level > target):
            continue

        out.append(
            FadeCandidate(
                direction=direction,
                trigger_level=trigger_level,
                stop_level=stop_level,
                target=target,
                start_ts=int(ts_vals[first]),
                end_ts=int(ts_vals[latest]),
            )
        )

    return out


def _to_trigger_candidate(candidate: FadeCandidate) -> PatternCandidate:
    """Adapt a FadeCandidate for check_breakout's crossing/freshness logic.

    A long fade fires on a fresh close back ABOVE the lower band — exactly
    check_breakout's "long" crossing of breakout_level. target_height is
    unused by check_breakout and set to the fade distance for transparency.
    """
    return PatternCandidate(
        kind=FADE_KIND,
        direction=candidate.direction,
        breakout_level=candidate.trigger_level,
        target_height=abs(candidate.target - candidate.trigger_level),
        start_ts=candidate.start_ts,
        end_ts=candidate.end_ts,
    )


def build_fade_signal(
    symbol: str,
    candidate: FadeCandidate,
    event: BreakoutEvent,
    *,
    rr_floor: float | None = None,
) -> Signal | None:
    """
    Compute SL/TP for a triggered fade and apply the R:R ratio floor.

    Stop stays at the excursion extreme (candidate.stop_level) — the extreme
    the fade thesis is invalidated by — rather than an ATR-derived distance.

    The floor is applied to the COST-ADJUSTED ratio, not the gross one. Unlike
    the breakout path, whose risk is k * ATR and so has a volatility floor,
    this structural stop can sit arbitrarily close to entry when price
    re-crosses right next to the extreme. A dimensionless gross ratio cannot
    see that: entry 100.00 / stop 99.95 / target 100.075 scores rr = 1.5 while
    the ~0.14% round-trip cost exceeds the 0.075% reward, so even a perfect
    win loses money. risk.atr_stop.net_rr charges cost to both legs, which
    rejects those setups without reintroducing an absolute reward floor.

    Signal.rr remains the GROSS ratio so it stays comparable with the breakout
    path and with historical output; only the gate uses the net ratio.

    Args:
        symbol: Trading pair symbol.
        candidate: The fade setup that re-crossed.
        event: The trigger-bar re-cross event (entry reference).
        rr_floor: Minimum acceptable cost-adjusted reward:risk
            (default config.RR_FLOOR).

    Returns:
        Signal (pattern="bollinger-fade") if the setup passes, else None.
    """
    if rr_floor is None:
        rr_floor = config.RR_FLOOR

    entry = event.price
    stop = candidate.stop_level
    target = candidate.target
    if entry <= 0:
        return None

    if candidate.direction == "long":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target

    if risk <= 0 or reward <= 0:
        return None

    risk_pct = risk / entry
    reward_pct = reward / entry
    rr = reward_pct / risk_pct
    if net_rr(reward_pct, risk_pct, config.FEE_PCT, config.SLIPPAGE_PCT) < rr_floor:
        return None

    return Signal(
        symbol=symbol,
        ts=event.ts,
        direction=candidate.direction,
        pattern=FADE_KIND,
        entry=entry,
        stop=stop,
        target=target,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr=rr,
        volume_ratio=event.volume_ratio,
        volume_high=event.volume_high,
    )


def scan_fade_signals(conn, symbol: str, now_ms: int) -> list[Signal]:
    """
    Fade-scan one symbol WITHOUT a regime check (dispatcher applies the gate).

    Args:
        conn: Database connection.
        symbol: Trading pair symbol.
        now_ms: Evaluation time in epoch milliseconds.

    Returns:
        List of screened fade Signals (possibly empty).
    """
    df_setup = _load_df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME, now_ms)
    df_trig = _load_df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME, now_ms)
    if df_setup.empty or df_trig.empty:
        return []

    df_setup = df_setup.tail(config.PATTERN_LOOKBACK_BARS)
    df_trig = df_trig.tail(config.VOLUME_LOOKBACK + 2)

    # interval_ms mirrors the breakout path (setup.py): without it,
    # check_breakout's fresh-crossing test compares against a "preceding close"
    # that may be arbitrarily stale, so a gapped pair can read as a fresh
    # re-cross. That hazard grows at coarser tiers — a gap is now hours, not
    # minutes.
    trigger_interval = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]

    signals: list[Signal] = []
    for candidate in detect_fade_setups(df_setup):
        event = check_breakout(
            df_trig, _to_trigger_candidate(candidate), interval_ms=trigger_interval
        )
        if event is None:
            continue
        signal = build_fade_signal(symbol, candidate, event)
        if signal is not None:
            signals.append(signal)

    return signals


def current_fade_signals(
    conn,
    symbol: str,
    *,
    now_ms: int | None = None,
) -> tuple[str, list[Signal]]:
    """
    Scan one symbol for mean-reversion fade signals as of now_ms.

    Args:
        conn: Database connection.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        now_ms: Evaluation time in epoch milliseconds (default: current time).

    Returns:
        Tuple of (regime_label, signals); signals is empty unless the regime
        is "ranging" and a triggered fade passed the R:R band.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    regime_label, _, _ = current_regime(conn, symbol, now_ms=now_ms)
    if regime_label != "ranging":
        return (regime_label, [])

    return (regime_label, scan_fade_signals(conn, symbol, now_ms))
