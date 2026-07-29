"""
Fractal swing-point (pivot) detection on OHLCV bars.

A pivot high at bar i is a bar whose high strictly exceeds the highs of the
`span` bars on each side; a pivot low is symmetric on lows. Ties reject the
candidate — strict comparison keeps detection deterministic and unambiguous.

No lookahead leaks into consumers: a pivot at bar i is only knowable once
`span` bars have closed after it, and detection only emits pivots with a full
window on both sides. Callers evaluating "as of bar t" must treat pivots with
index > t - span as not yet confirmed (find_pivots on a df sliced up to t
already guarantees this).
"""

import logging
from dataclasses import dataclass

import pandas as pd

from trading_bot import config

logger = logging.getLogger("trading_bot")


@dataclass(frozen=True)
class Pivot:
    """A confirmed swing point.

    Attributes:
        index: Positional index of the bar within the DataFrame it was found in.
        ts: Epoch-ms timestamp of the bar (DataFrame index value).
        price: The pivot price (bar high for kind="high", bar low for kind="low").
        kind: "high" or "low".
    """

    index: int
    ts: int
    price: float
    kind: str


def find_pivots(df: pd.DataFrame, *, span: int | None = None) -> list[Pivot]:
    """
    Detect fractal pivot highs and lows in an OHLCV DataFrame.

    Args:
        df: DataFrame with columns open, high, low, close, volume indexed by
            epoch-ms ts, ordered ascending.
        span: Bars on each side the pivot must strictly dominate
              (default config.PIVOT_SPAN).

    Returns:
        List of Pivot ordered by bar index ascending. Strict comparison means a
        bar contributes at most one pivot of each kind, and flat ties never
        produce pivots. Bars within `span` of either end are never pivots.

        Note an outside bar that dominates both sides yields BOTH a high and a
        low at the same index, ordered high-then-low by the sort key even though
        their true intra-bar order is unknowable from OHLC. Consumers reading
        pivots as a time sequence must not treat two entries sharing an index as
        separated in time (see patterns._strictly_ordered).
    """
    if span is None:
        span = config.PIVOT_SPAN

    n = len(df)
    if n < 2 * span + 1:
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    ts_vals = df.index.to_numpy()

    pivots: list[Pivot] = []
    for i in range(span, n - span):
        window_h = highs[i - span : i + span + 1]
        window_l = lows[i - span : i + span + 1]
        h, low = highs[i], lows[i]

        # Strict domination: the center bar must be the unique extreme.
        if h > window_h[:span].max() and h > window_h[span + 1 :].max():
            pivots.append(Pivot(index=i, ts=int(ts_vals[i]), price=float(h), kind="high"))
        if low < window_l[:span].min() and low < window_l[span + 1 :].min():
            pivots.append(Pivot(index=i, ts=int(ts_vals[i]), price=float(low), kind="low"))

    pivots.sort(key=lambda p: (p.index, p.kind))
    return pivots
