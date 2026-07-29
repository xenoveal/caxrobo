"""MARKET STRUCTURE family: support/resistance, ranges, pivots, liquidity levels.

WHY THIS FAMILY, AND WHY IT IS THE RISKIEST ONE TO AUTHOR
--------------------------------------------------------
The PRD's post-mortem indicts the *risk model*, not the signal: a fixed
0.5%-of-price stop sat inside the noise floor (median 15m bar range 0.251% /
0.339% / 0.520% for BTC/ETH/SOL) and unconditional adverse-excursion probability
inside the hold window was 91-95%. A market-structure stop -- "below the swing
low that defines this trade" -- is the principled replacement, because the level
is *why* the trade exists, so its violation is *why* the trade is wrong.

But this family is also where the previous system died. The dropped Bollinger
fade sleeve failed with ``cost_ratio`` 0.18/0.11/0.10 against a 0.10 ceiling
precisely because a nearby band/level yields a tiny stop. A structural stop
0.3% from entry reproduces the old failure exactly, with better narrative cover.

So EVERY stop here goes through ``_structural_stop``: the structural distance is
padded by a buffer and then floored at ``atr_floor * ATR(4H)``. The floor is not
cosmetic -- on the sweep/retest strategies it binds on the majority of bars, and
that is the intended behaviour. ``median_risk_pct`` and ``cost_ratio`` are
reported for every combo so the reader can verify which term won.

CAUSALITY
---------
Three devices carry state forward in time, and all three are strictly backward
looking:

- ``ta.pivot_high``/``pivot_low`` place a pivot ``span`` bars LATE (the flag sits
  at t+span carrying the price at t). That shift is the whole point and is never
  removed here.
- ``_ffill_at`` freezes a value observed on a trigger bar and carries it forward.
  Forward-fill imports nothing.
- ``_bars_since`` counts bars since the most recent True using a running maximum
  of indices -- it can only see the past.

Coarse timeframes enter only via ``ctx.align``. No ``shift(-n)``, no
``center=True``, no ``bfill``, no full-sample statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ta
from core import LONG, SHORT, Plan
from registry import register

# ---------------------------------------------------------------------------
# Causal state helpers
# ---------------------------------------------------------------------------


def _ffill_at(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Freeze ``values`` on bars where ``mask`` is True, then carry forward.

    Used to remember the level that was broken, rather than re-reading a level
    series that has since moved on. Forward-fill only, so bar i reflects the
    latest qualifying event at or before i.
    """
    out = np.full(len(mask), np.nan, dtype=float)
    sel = mask & np.isfinite(values)
    out[sel] = values[sel]
    return pd.Series(out).ffill().to_numpy()


def _bars_since(mask: np.ndarray) -> np.ndarray:
    """Bars elapsed since the most recent True in ``mask`` (0 on the True bar).

    A large sentinel where no True has occurred yet, so range comparisons fail
    closed rather than matching everything.
    """
    idx = np.arange(len(mask), dtype=np.int64)
    last = np.maximum.accumulate(np.where(mask, idx, -1))
    return np.where(last < 0, 1 << 40, idx - last)


def _first_of_run(mask: np.ndarray) -> np.ndarray:
    """True only on the first bar of each contiguous run of True.

    A breakout that stays broken for 30 bars is ONE event; without this, every
    bar of the run re-arms the retest window and the strategy silently becomes
    "price is above a level" rather than "price just crossed a level".
    """
    prev = np.concatenate(([False], mask[:-1]))
    return mask & ~prev


def _prev_pivot(sparse: pd.Series) -> pd.Series:
    """The pivot BEFORE the latest confirmed one, forward-filled.

    ``sparse.shift(1)`` would shift by one BAR, and since a pivot series is
    forward-filled that yields "same pivot" on almost every bar -- silently
    turning a higher-low test into a never-true condition. Shifting among the
    non-NaN observations instead steps back one PIVOT. Reindex + ffill keeps it
    causal: bar i sees only pivots confirmed at or before i.
    """
    vals = sparse.dropna()
    return vals.shift(1).reindex(sparse.index).ffill()


def _structural_stop(
    struct_dist: np.ndarray, atr: np.ndarray, buf: float, atr_floor: float
) -> np.ndarray:
    """Buffered structural distance, floored at ``atr_floor * ATR``.

    ``struct_dist`` is entry-to-level distance in price units and may be tiny or
    negative-by-rounding; ``np.maximum`` against the ATR floor is what stops a
    coincidentally-nearby level from manufacturing a noise-floor stop. Returns
    NaN where ATR is unknown, which suppresses the entry in ``core.simulate``.
    """
    padded = np.where(np.isfinite(struct_dist), struct_dist * buf, 0.0)
    return np.maximum(padded, atr_floor * atr)


def _empty(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int8)


# ---------------------------------------------------------------------------
# 1. Support/resistance retest
# ---------------------------------------------------------------------------


@register(
    family="structure",
    grid={
        "span": [3, 6],
        "retest_within": [12, 24],
        "atr_floor": [1.0, 1.5],
        "rr": [1.5, 2.5],
    },
    rationale=(
        "A broken swing level flips polarity: former resistance becomes support. "
        "Entering on the PULLBACK to the level rather than on the break itself "
        "buys a much better entry price relative to the same invalidation point, "
        "which is the only lever that raises R:R without demanding more edge. "
        "Stop sits beyond the flipped level, ATR(4H)-floored."
    ),
)
def st_sr_retest(ctx, span, retest_within, atr_floor, rr):
    f4 = ctx.frame("4h")
    sw = ta.swing_levels(f4, span)
    res = ctx.align(sw["res"], "4h")
    sup = ctx.align(sw["sup"], "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    low = ctx.trigger["low"].to_numpy(dtype=float)
    high = ctx.trigger["high"].to_numpy(dtype=float)

    broke_up = _first_of_run(close > res)
    broke_dn = _first_of_run(close < sup)
    lvl_up = _ffill_at(broke_up, res)   # the resistance that was broken
    lvl_dn = _ffill_at(broke_dn, sup)

    age_up, age_dn = _bars_since(broke_up), _bars_since(broke_dn)
    # Retest: price came back DOWN to the flipped level but closed above it.
    long_ok = (
        (age_up >= 1) & (age_up <= retest_within)
        & (low <= lvl_up) & (close > lvl_up)
    )
    short_ok = (
        (age_dn >= 1) & (age_dn <= retest_within)
        & (high >= lvl_dn) & (close < lvl_dn)
    )

    entry = _empty(ctx.n)
    entry[long_ok] = LONG
    entry[short_ok] = SHORT

    struct = np.where(long_ok, close - lvl_up, np.where(short_ok, lvl_dn - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.5, atr_floor=atr_floor)
    return Plan(entry=entry, stop_dist=stop, target_dist=rr * stop, note="sr-retest")


# ---------------------------------------------------------------------------
# 2. Range fade (mean reversion between boundaries, exit at the mid)
# ---------------------------------------------------------------------------


@register(
    family="structure",
    max_hold_bars=48,
    grid={
        "period": [20, 40],
        "max_width": [0.06, 0.10],
        "edge": [0.15, 0.25],
        "atr_floor": [1.0, 1.5],
    },
    rationale=(
        "Inside a confirmed range, the boundaries are where resting liquidity "
        "sits and the mid is the fair-value magnet. The width filter is load "
        "bearing: it is what makes the boundary-to-mid target large relative to "
        "the ATR-floored stop, and it is the discipline the old Bollinger fade "
        "sleeve lacked when it posted cost_ratio 0.18."
    ),
)
def st_range_fade(ctx, period, max_width, edge, atr_floor):
    f4 = ctx.frame("4h")
    hi_s = ta.rolling_high(f4, period, exclude_current=False)
    lo_s = ta.rolling_low(f4, period, exclude_current=False)
    hi = ctx.align(hi_s, "4h")
    lo = ctx.align(lo_s, "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    mid = 0.5 * (hi + lo)
    width = (hi - lo) / np.where(close > 0, close, np.nan)
    pos = (close - lo) / np.where(hi > lo, hi - lo, np.nan)
    tight = width <= max_width

    long_ok = tight & (pos <= edge)
    short_ok = tight & (pos >= 1.0 - edge)
    entry = _empty(ctx.n)
    entry[long_ok] = LONG
    entry[short_ok] = SHORT

    struct = np.where(long_ok, close - lo, np.where(short_ok, hi - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.5, atr_floor=atr_floor)
    # Target is the range mid, not an R multiple: the thesis is reversion to
    # fair value, and a fixed R target would either overshoot the range or
    # leave money inside it.
    tgt = np.where(long_ok, mid - close, np.where(short_ok, close - mid, np.nan))
    tgt = np.where(np.isfinite(tgt) & (tgt > 0), tgt, np.nan)
    return Plan(entry=entry, stop_dist=stop, target_dist=tgt, note="range-fade")


# ---------------------------------------------------------------------------
# 3. Immediate range breakout
# ---------------------------------------------------------------------------


@register(
    family="structure",
    grid={
        "period": [20, 40, 60],
        "max_width": [0.04, 0.06, 0.10],
        "atr_floor": [1.0, 1.5],
        "rr": [2.0, 3.0],
    },
    rationale=(
        "Compression resolves into expansion; a break of a demonstrably TIGHT "
        "range is the cleanest version of that because the pre-break range "
        "supplies both a small, defined invalidation and a measured objective. "
        "Registered as the control arm against st_range_breakout_retest -- the "
        "immediate-vs-retest question is the open one in this family."
    ),
)
def st_range_breakout(ctx, period, max_width, atr_floor, rr):
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, period), "4h")
    lo = ctx.align(ta.rolling_low(f4, period), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    width = (hi - lo) / np.where(close > 0, close, np.nan)
    tight = width <= max_width
    up = _first_of_run(tight & (close > hi))
    dn = _first_of_run(tight & (close < lo))

    entry = _empty(ctx.n)
    entry[up] = LONG
    entry[dn] = SHORT

    # A break back to the MIDDLE of the vacated range invalidates the thesis --
    # a stop just under the broken boundary is inside the break's own noise.
    mid = 0.5 * (hi + lo)
    struct = np.where(up, close - mid, np.where(dn, mid - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.0, atr_floor=atr_floor)
    return Plan(entry=entry, stop_dist=stop, target_dist=rr * stop, note="range-brk")


# ---------------------------------------------------------------------------
# 4. Range breakout with retest confirmation
# ---------------------------------------------------------------------------


@register(
    family="structure",
    grid={
        "period": [20, 40],
        "max_width": [0.06, 0.10],
        "retest_within": [8, 16],
        "atr_floor": [1.0, 1.5],
        "rr": [2.0, 3.0],
    },
    rationale=(
        "Same event as st_range_breakout, but demanding the break hold on a "
        "return to the boundary. This trades hit-rate of participation (many "
        "breakouts never retest) for entry quality, and isolates whether the "
        "crypto perp breakout edge lives in immediacy or in confirmation."
    ),
)
def st_range_breakout_retest(ctx, period, max_width, retest_within, atr_floor, rr):
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, period), "4h")
    lo = ctx.align(ta.rolling_low(f4, period), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    low = ctx.trigger["low"].to_numpy(dtype=float)
    high = ctx.trigger["high"].to_numpy(dtype=float)

    width = (hi - lo) / np.where(close > 0, close, np.nan)
    tight = width <= max_width
    up = _first_of_run(tight & (close > hi))
    dn = _first_of_run(tight & (close < lo))
    lvl_up, lvl_dn = _ffill_at(up, hi), _ffill_at(dn, lo)
    age_up, age_dn = _bars_since(up), _bars_since(dn)

    long_ok = (age_up >= 1) & (age_up <= retest_within) & (low <= lvl_up) & (close > lvl_up)
    short_ok = (age_dn >= 1) & (age_dn <= retest_within) & (high >= lvl_dn) & (close < lvl_dn)
    entry = _empty(ctx.n)
    entry[long_ok] = LONG
    entry[short_ok] = SHORT

    struct = np.where(long_ok, close - lvl_up, np.where(short_ok, lvl_dn - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.5, atr_floor=atr_floor)
    return Plan(entry=entry, stop_dist=stop, target_dist=rr * stop, note="brk-retest")


# ---------------------------------------------------------------------------
# 5. Liquidity sweep / stop-hunt reversal
# ---------------------------------------------------------------------------


@register(
    family="structure",
    max_hold_bars=48,
    grid={
        "span": [3, 6],
        "pierce": [0.001, 0.003],
        "atr_floor": [1.0, 1.5],
        "rr": [1.5, 2.5],
    },
    rationale=(
        "Resting stops and liquidations cluster just beyond obvious swing "
        "extremes; a bar that pierces a prior swing low and closes back above it "
        "is the signature of that liquidity being taken and rejected. Unlike most "
        "patterns the invalidation is unambiguous and genuinely close -- the "
        "sweep's own extreme -- so it is the one place a tight stop is defensible "
        "rather than arbitrary. It is still ATR-floored."
    ),
)
def st_liquidity_sweep(ctx, span, pierce, atr_floor, rr):
    f1 = ctx.trigger
    sw = ta.swing_levels(f1, span)
    # shift(1): the level must predate the bar that sweeps it, otherwise a bar
    # can be judged against a pivot it helped define.
    sup = sw["sup"].shift(1).to_numpy(dtype=float)
    res = sw["res"].shift(1).to_numpy(dtype=float)
    atr = ctx.align(ta.atr(ctx.frame("4h"), 14), "4h")
    close = f1["close"].to_numpy(dtype=float)
    low = f1["low"].to_numpy(dtype=float)
    high = f1["high"].to_numpy(dtype=float)

    long_ok = (low < sup * (1.0 - pierce)) & (close > sup)
    short_ok = (high > res * (1.0 + pierce)) & (close < res)
    entry = _empty(ctx.n)
    entry[long_ok] = LONG
    entry[short_ok] = SHORT

    struct = np.where(long_ok, close - low, np.where(short_ok, high - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.15, atr_floor=atr_floor)
    return Plan(entry=entry, stop_dist=stop, target_dist=rr * stop, note="sweep")


# ---------------------------------------------------------------------------
# 6. Break of structure / change of character
# ---------------------------------------------------------------------------


@register(
    family="structure",
    grid={
        "span": [3, 6],
        "atr_floor": [1.0, 1.5, 2.0],
        "rr": [1.5, 2.5],
        "require_hl": [True, False],
    },
    rationale=(
        "A trend IS a sequence of higher highs and higher lows; taking out the "
        "prior swing high while the swing lows are also rising is the formal "
        "confirmation of continuation (break of structure). The require_hl axis "
        "tests whether the higher-low condition adds anything over the bare "
        "level break -- i.e. whether structure or momentum is doing the work."
    ),
)
def st_bos(ctx, span, atr_floor, rr, require_hl):
    f4 = ctx.frame("4h")
    ph, pl = ta.pivot_high(f4, span), ta.pivot_low(f4, span)
    res = ctx.align(ph.ffill(), "4h")
    sup = ctx.align(pl.ffill(), "4h")
    # "Previous pivot" must step back one PIVOT, not one bar -- see _prev_pivot.
    sup_prev = ctx.align(_prev_pivot(pl), "4h")
    res_prev = ctx.align(_prev_pivot(ph), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    hl = (sup > sup_prev) if require_hl else np.ones(ctx.n, dtype=bool)
    lh = (res < res_prev) if require_hl else np.ones(ctx.n, dtype=bool)

    up = _first_of_run(close > res) & hl
    dn = _first_of_run(close < sup) & lh
    entry = _empty(ctx.n)
    entry[up] = LONG
    entry[dn] = SHORT

    # Structure is broken if the swing that defined it gives way.
    struct = np.where(up, close - sup, np.where(dn, res - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.0, atr_floor=atr_floor)
    return Plan(entry=entry, stop_dist=stop, target_dist=rr * stop, note="bos")


# ---------------------------------------------------------------------------
# 7. Multi-timeframe level confluence
# ---------------------------------------------------------------------------


@register(
    family="structure",
    grid={
        "span": [2, 3],
        "tol": [0.002, 0.005],
        "atr_floor": [1.0, 1.5],
        "rr": [1.5, 2.5],
    },
    rationale=(
        "A daily swing level is watched by far more capital than a 1H one, so "
        "the reaction at it is larger relative to noise. Executing the touch on "
        "the 1H grid gives a stop measured against the daily level while paying "
        "only 1H-scale distance to it -- the cheapest available form of the "
        "asymmetry this whole family is looking for."
    ),
)
def st_mtf_level(ctx, span, tol, atr_floor, rr):
    f1d = ctx.frame("1d")
    sw = ta.swing_levels(f1d, span)
    sup = ctx.align(sw["sup"], "1d")
    res = ctx.align(sw["res"], "1d")
    atr = ctx.align(ta.atr(ctx.frame("4h"), 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    low = ctx.trigger["low"].to_numpy(dtype=float)
    high = ctx.trigger["high"].to_numpy(dtype=float)

    # Touch-and-hold: the bar traded into the level's tolerance band but closed
    # on the correct side of it.
    long_ok = (low <= sup * (1.0 + tol)) & (close > sup) & (low > sup * (1.0 - 3 * tol))
    short_ok = (high >= res * (1.0 - tol)) & (close < res) & (high < res * (1.0 + 3 * tol))
    entry = _empty(ctx.n)
    entry[long_ok] = LONG
    entry[short_ok] = SHORT

    struct = np.where(long_ok, close - sup, np.where(short_ok, res - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.5, atr_floor=atr_floor)
    return Plan(entry=entry, stop_dist=stop, target_dist=rr * stop, note="mtf-level")


# ---------------------------------------------------------------------------
# 8. Prior-day high/low break (crypto opening-range analogue)
# ---------------------------------------------------------------------------


@register(
    family="structure",
    grid={
        "adx_min": [0.0, 20.0],
        "atr_floor": [1.0, 1.5],
        "rr": [1.5, 2.5, 3.5],
    },
    rationale=(
        "Previous-day high/low are the two levels every desk marks, and in "
        "24/7 crypto they are the only calendar-anchored reference points that "
        "exist -- the analogue of an equity opening-range break. The adx_min=0 "
        "arm is the unfiltered control so the report can attribute any edge to "
        "the level rather than to the trend filter."
    ),
)
def st_prior_day_break(ctx, adx_min, atr_floor, rr):
    f1d = ctx.frame("1d")
    # exclude_current=False + align: align only exposes CLOSED daily bars, so
    # this is exactly "the last completed day's high".
    pdh = ctx.align(ta.rolling_high(f1d, 1, exclude_current=False), "1d")
    pdl = ctx.align(ta.rolling_low(f1d, 1, exclude_current=False), "1d")
    adx = ctx.align(ta.adx(f1d, 14), "1d")
    atr = ctx.align(ta.atr(ctx.frame("4h"), 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    ok = adx >= adx_min
    up = _first_of_run(close > pdh) & ok
    dn = _first_of_run(close < pdl) & ok
    entry = _empty(ctx.n)
    entry[up] = LONG
    entry[dn] = SHORT

    # Failed break = back through the opposite prior-day level's midpoint.
    mid = 0.5 * (pdh + pdl)
    struct = np.where(up, close - mid, np.where(dn, mid - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.0, atr_floor=atr_floor)
    return Plan(entry=entry, stop_dist=stop, target_dist=rr * stop, note="pdh-pdl")


# ---------------------------------------------------------------------------
# 9. Round-number level reclaim
# ---------------------------------------------------------------------------


@register(
    family="structure",
    max_hold_bars=48,
    grid={
        "step_mult": [1.0, 2.0],
        "tol": [0.001, 0.003],
        "atr_floor": [1.0, 1.5],
        "rr": [1.5, 2.5],
    },
    rationale=(
        "Limit and stop orders cluster at round numbers (the price-clustering "
        "effect is documented across equities and FX), so they act as levels "
        "with no prior-price memory at all. The grid is defined RELATIVE to each "
        "symbol's own order of magnitude -- one decade below the price, scaled by "
        "step_mult -- so no specific historical price (BTC 100000) is ever "
        "hardcoded. That scale-free construction is the anti-curve-fit control: "
        "the same rule applies to a $0.10 coin and a $100k one, and if the edge "
        "were an artifact of one symbol's price path it could not survive the "
        "20-symbol universe."
    ),
)
def st_round_number(ctx, step_mult, tol, atr_floor, rr):
    close = ctx.trigger["close"].to_numpy(dtype=float)
    low = ctx.trigger["low"].to_numpy(dtype=float)
    high = ctx.trigger["high"].to_numpy(dtype=float)
    atr = ctx.align(ta.atr(ctx.frame("4h"), 14), "4h")

    with np.errstate(divide="ignore", invalid="ignore"):
        decade = np.floor(np.log10(np.where(close > 0, close, np.nan)))
    step = step_mult * np.power(10.0, decade - 1.0)
    lvl = np.round(close / step) * step

    # Reclaim from below: the bar dipped through a round level and closed above.
    long_ok = (low <= lvl * (1.0 - tol)) & (close > lvl)
    # Rejection from above: poked above and closed back below.
    short_ok = (high >= lvl * (1.0 + tol)) & (close < lvl)
    entry = _empty(ctx.n)
    entry[long_ok] = LONG
    entry[short_ok] = SHORT

    struct = np.where(long_ok, close - low, np.where(short_ok, high - close, np.nan))
    stop = _structural_stop(struct, atr, buf=1.15, atr_floor=atr_floor)
    return Plan(entry=entry, stop_dist=stop, target_dist=rr * stop, note="round-num")
