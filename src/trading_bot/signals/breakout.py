"""
Trigger-timeframe breakout with rolling-volume confirmation.

A pattern candidate fires when a closed trigger bar closes beyond the candidate's
breakout level AND the crossing is fresh (the preceding bar's close was not
already beyond the level) — this prevents re-triggering on every bar of an
extended move. The trigger bar must also close at or after the pattern's
completion (candidate.end_ts).

Only the last `lookback_bars` closed bars are eligible, newest first, so the
returned event is always the most recent crossing. The default of 1 means the
latest bar only, which keeps the entry reference as fresh as possible; a caller
that cannot guarantee it runs on the trigger-bar boundary can widen the window and
accept a staler entry price rather than silently miss the breakout entirely.

Bars are assumed contiguous. Pass `interval_ms` to have that verified: a gap
between the two bars of a crossing pair makes "the preceding close" a stale
reference, which can read as a fresh crossing when hours in fact elapsed.

Volume is a graded confidence input, never a hard block (per PRD): the trigger
bar's volume is compared against the rolling mean of the VOLUME_LOOKBACK bars
preceding it. `volume_high` marks a notably-high-volume breakout; a low ratio
lowers downstream confidence but does not suppress the event.
"""

import logging
import math
from dataclasses import dataclass

import pandas as pd

from trading_bot import config
from trading_bot.signals.patterns import PatternCandidate

logger = logging.getLogger("trading_bot")


@dataclass(frozen=True)
class BreakoutEvent:
    """A confirmed breakout of a pattern's level on the trigger timeframe.

    Attributes:
        ts: Epoch-ms of the trigger bar (the latest closed one).
        price: Trigger bar close — the signal's entry reference.
        level: The breakout level that was crossed.
        direction: "long" (closed above level) or "short" (closed below).
        volume_ratio: Trigger volume / rolling mean of the prior
            VOLUME_LOOKBACK bars; NaN if the average is undefined.
        volume_high: True when volume_ratio >= VOLUME_HIGH_RATIO.
    """

    ts: int
    price: float
    level: float
    direction: str
    volume_ratio: float
    volume_high: bool


def check_breakout(
    df: pd.DataFrame,
    candidate: PatternCandidate,
    *,
    volume_lookback: int | None = None,
    volume_high_ratio: float | None = None,
    lookback_bars: int | None = None,
    interval_ms: int | None = None,
) -> BreakoutEvent | None:
    """
    Check whether a recent closed trigger bar breaks the candidate's level.

    Args:
        df: Trigger-timeframe OHLCV DataFrame of CLOSED bars only (columns open/high/low/
            close/volume, epoch-ms index, ascending). The last row is the most
            recent trigger bar under evaluation.
        candidate: Pattern candidate providing level and direction.
        volume_lookback: Bars in the rolling volume mean
            (default config.VOLUME_LOOKBACK).
        volume_high_ratio: Ratio marking notably high volume
            (default config.VOLUME_HIGH_RATIO).
        lookback_bars: How many of the most recent bars may supply the crossing,
            newest first (default config.BREAKOUT_TRIGGER_LOOKBACK_BARS; 1 =
            latest bar only). Callers replaying history bar-by-bar must leave
            this at 1 so the entry price is the bar being evaluated.
        interval_ms: Expected spacing between bars. When given, a crossing pair
            that is not exactly this far apart is skipped — the gap makes the
            preceding close a stale reference for the fresh-crossing test.

    Returns:
        BreakoutEvent for the most recent bar that closes beyond the level in
        the candidate's direction with a fresh crossing, else None.
    """
    if volume_lookback is None:
        volume_lookback = config.VOLUME_LOOKBACK
    if volume_high_ratio is None:
        volume_high_ratio = config.VOLUME_HIGH_RATIO
    if lookback_bars is None:
        lookback_bars = config.BREAKOUT_TRIGGER_LOOKBACK_BARS

    n = len(df)
    if n < 2:
        return None

    level = candidate.breakout_level
    ts_vals = df.index.to_numpy()
    closes = df["close"].to_numpy()

    # Newest bar first: the most recent crossing is the actionable one.
    oldest = max(1, n - max(1, lookback_bars))
    for i in range(n - 1, oldest - 1, -1):
        ts = int(ts_vals[i])
        # Trigger bar must not predate pattern completion.
        if ts < candidate.end_ts:
            continue
        if interval_ms is not None and ts - int(ts_vals[i - 1]) != interval_ms:
            logger.debug(
                "skipping non-contiguous crossing pair at ts=%d (gap %d ms, expected %d)",
                ts,
                ts - int(ts_vals[i - 1]),
                interval_ms,
            )
            continue

        if candidate.direction == "long":
            crossed = closes[i] > level and closes[i - 1] <= level
        else:
            crossed = closes[i] < level and closes[i - 1] >= level
        if not crossed:
            continue

        # Rolling volume mean over the bars preceding the trigger bar.
        prior_vol = df["volume"].iloc[:i].tail(volume_lookback)
        if len(prior_vol) < volume_lookback or float(prior_vol.mean()) <= 0:
            volume_ratio = float("nan")
        else:
            volume_ratio = float(df["volume"].to_numpy()[i]) / float(prior_vol.mean())

        volume_high = (not math.isnan(volume_ratio)) and volume_ratio >= volume_high_ratio

        return BreakoutEvent(
            ts=ts,
            price=float(closes[i]),
            level=float(level),
            direction=candidate.direction,
            volume_ratio=volume_ratio,
            volume_high=volume_high,
        )

    return None
