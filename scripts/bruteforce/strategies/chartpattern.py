"""Chart-pattern family: a fair RE-TEST of geometry the PRD retired.

WHY THIS MODULE EXISTS
----------------------
`.claude/PRPs/prds/hybrid-trend-voltarget.prd.md` retires chart-pattern geometry
outright: "Triangle + flag were ~98% of trade volume and both lose on all three
symbols; H&S never accumulated a judgeable sample (n=5-27)". That verdict was
measured under the OLD risk model -- fixed 0.5%-of-price stops sitting inside the
noise floor, where the round-trip cost consumed 58-68% of the risk unit. The user
has explicitly asked for chart patterns to be re-tested; this module gives them a
fair test under ATR-derived stops and reports honestly either way.

TWO DESIGN DECISIONS THAT ADDRESS THE PRD'S SPECIFIC CRITICISMS
--------------------------------------------------------------
1. **Trigger on the level break, not on pattern confirmation.** A double bottom
   or an (inverse) head-and-shoulders is only tradeable when price actually
   clears the neckline: the pattern is the context, the break is the event.
   Entering on confirmation alone is what made the retired implementation fire
   constantly and lose. Every pattern here carries an explicit neckline and an
   expiry, and a trade happens only on the crossing bar.
2. **Every stop is `max(k * ATR(setup), structural distance)`.** The ATR floor is
   what keeps `cost_ratio = round_trip_cost / risk` under the 0.10 ceiling; the
   structural term is what makes the stop mean something geometrically (below the
   second low, beyond the head). Taking the max of the two never produces a stop
   inside the noise floor, which is precisely the failure the PRD diagnosed.

CAUSALITY
---------
All geometry is built from ``ta.pivot_high``/``ta.pivot_low``, which place a
pivot ``span`` bars LATE because that is when it becomes knowable. The local
helpers below only ever read bars at or before the confirmation bar, and
``_carry`` propagates a pattern FORWARD (never backward) for a bounded number of
bars. Coarse-timeframe values reach the 1H trigger grid solely via ``ctx.align``.
Setup geometry lives on 4H and regime filters on 1D, matching the PRD's tier
shift; nothing is sub-hourly, which the PRD places on the wrong side of the cost
frontier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ta
from core import LONG, SHORT, Plan
from registry import register

# ---------------------------------------------------------------------------
# Local geometry helpers.
#
# These live here rather than in indicators.py (which is frozen) and are all
# "mark at the confirmation bar, then carry forward" constructions: a pattern
# becomes visible on the bar its last pivot confirms and stays actionable for a
# bounded window, never earlier.
# ---------------------------------------------------------------------------


def _carry(index: pd.Index, marks: list[tuple[int, float]], expiry: int) -> pd.Series:
    """Place values at their confirmation bars and forward-fill for ``expiry`` bars.

    Forward-fill is causal (it repeats a value already known); the ``limit``
    makes a pattern go stale instead of arming a trigger months later. A later
    mark simply overwrites the carry, which is the desired "most recent pattern
    wins" behaviour.
    """
    s = pd.Series(np.nan, index=index, dtype=float)
    for i, value in marks:
        s.iloc[i] = value
    return s.ffill(limit=expiry)


def _double_pattern(
    df: pd.DataFrame, *, span: int, tol: float, max_gap: int, expiry: int, top: bool
) -> pd.DataFrame:
    """Double top/bottom, returning the ``neck`` line and the ``guard`` extreme.

    ``neck`` is the intervening counter-pivot the break must clear (the peak
    between two lows, the trough between two highs); ``guard`` is the extreme of
    the two matched pivots, i.e. where a structural stop belongs. Both are
    computed from bars <= the second pivot's confirmation bar.
    """
    piv = ta.pivot_high(df, span) if top else ta.pivot_low(df, span)
    vals = piv.to_numpy()
    idx = np.flatnonzero(piv.notna().to_numpy())
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    neck_marks: list[tuple[int, float]] = []
    guard_marks: list[tuple[int, float]] = []
    for a, b in zip(idx, idx[1:]):
        gap = b - a
        if gap > max_gap or gap < 2 * span:
            continue
        if abs(vals[b] - vals[a]) > tol * abs(vals[a]):
            continue
        if top:
            neck = float(np.min(lows[a : b + 1]))
            guard = float(max(vals[a], vals[b]))
        else:
            neck = float(np.max(highs[a : b + 1]))
            guard = float(min(vals[a], vals[b]))
        neck_marks.append((int(b), neck))
        guard_marks.append((int(b), guard))
    return pd.DataFrame(
        {
            "neck": _carry(df.index, neck_marks, expiry),
            "guard": _carry(df.index, guard_marks, expiry),
        }
    )


def _hs_pattern(
    df: pd.DataFrame, *, span: int, tol: float, prominence: float,
    max_gap: int, expiry: int, inverse: bool,
) -> pd.DataFrame:
    """(Inverse) head-and-shoulders with an explicit neckline and guard level.

    Three consecutive same-side pivots, outer two within ``tol`` of each other,
    middle one more extreme by at least ``prominence``. ``neck`` is the extreme
    counter-move inside the formation (the conservative horizontal neckline the
    production ``signals/patterns.py`` also used); ``guard`` is the head.
    """
    piv = ta.pivot_low(df, span) if inverse else ta.pivot_high(df, span)
    vals = piv.to_numpy()
    idx = np.flatnonzero(piv.notna().to_numpy())
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    neck_marks: list[tuple[int, float]] = []
    guard_marks: list[tuple[int, float]] = []
    for a, b, c in zip(idx, idx[1:], idx[2:]):
        if c - a > max_gap:
            continue
        left, head, right = float(vals[a]), float(vals[b]), float(vals[c])
        if abs(right - left) > tol * abs(left):
            continue
        if inverse:
            if head > min(left, right) * (1.0 - prominence):
                continue
            neck = float(np.max(highs[a : c + 1]))
        else:
            if head < max(left, right) * (1.0 + prominence):
                continue
            neck = float(np.min(lows[a : c + 1]))
        neck_marks.append((int(c), neck))
        guard_marks.append((int(c), head))
    return pd.DataFrame(
        {
            "neck": _carry(df.index, neck_marks, expiry),
            "guard": _carry(df.index, guard_marks, expiry),
        }
    )


def _cross_up(close: np.ndarray, level: np.ndarray) -> np.ndarray:
    """True where close moves from at-or-below ``level`` to above it."""
    prev_c = np.roll(close, 1)
    prev_l = np.roll(level, 1)
    prev_c[0], prev_l[0] = np.nan, np.nan
    return (close > level) & (prev_c <= prev_l)


def _cross_down(close: np.ndarray, level: np.ndarray) -> np.ndarray:
    prev_c = np.roll(close, 1)
    prev_l = np.roll(level, 1)
    prev_c[0], prev_l[0] = np.nan, np.nan
    return (close < level) & (prev_c >= prev_l)


def _stop(close: np.ndarray, atr: np.ndarray, k: float, guard: np.ndarray) -> np.ndarray:
    """``max(k*ATR, distance to the structural guard)``.

    The ATR term is a FLOOR, never a cap: it is what holds ``cost_ratio`` under
    the PRD's 0.10 ceiling. NaN guards fall back to the ATR stop.
    """
    struct = np.abs(close - guard)
    return np.fmax(k * atr, np.nan_to_num(struct, nan=0.0))


def _blank(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int8)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@register(
    family="chartpattern",
    grid={"span": [3, 5], "tol": [0.015, 0.03], "k": [1.5, 2.0], "expiry": [12, 24]},
    rationale=(
        "Double bottom / double top traded on the NECKLINE BREAK rather than on "
        "pattern confirmation: two failures to make a new extreme establish that "
        "one side is exhausted, and the break of the intervening counter-pivot is "
        "the moment the other side takes control. Stop sits beyond the matched "
        "pivot pair, floored at k*ATR(4H) so the risk unit stays outside the "
        "noise band that sank the retired implementation."
    ),
)
def cp_double_neckline(ctx, span, tol, k, expiry):
    f4 = ctx.frame("4h")
    bottom = _double_pattern(
        f4, span=span, tol=tol, max_gap=60, expiry=expiry, top=False
    )
    top = _double_pattern(f4, span=span, tol=tol, max_gap=60, expiry=expiry, top=True)
    atr = ctx.align(ta.atr(f4, 14), "4h")
    b_neck = ctx.align(bottom["neck"], "4h")
    b_guard = ctx.align(bottom["guard"], "4h")
    t_neck = ctx.align(top["neck"], "4h")
    t_guard = ctx.align(top["guard"], "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = _blank(ctx.n)
    long_sig = _cross_up(close, b_neck)
    short_sig = _cross_down(close, t_neck)
    entry[long_sig] = LONG
    entry[short_sig & ~long_sig] = SHORT
    guard = np.where(entry == LONG, b_guard, t_guard)
    return Plan(
        entry=entry,
        stop_dist=_stop(close, atr, k, guard),
        target_dist=2.0 * _stop(close, atr, k, guard),
        note="double-top/bottom neckline break",
    )


@register(
    family="chartpattern",
    grid={"span": [3, 5], "prominence": [0.01, 0.02], "k": [1.5, 2.0], "expiry": [12, 24]},
    rationale=(
        "(Inverse) head-and-shoulders on the neckline break. The PRD's complaint "
        "about H&S was sample size (n=5-27), not sign, so this is deliberately the "
        "coarsest defensible definition -- pivot span down to 3, prominence down to "
        "1% -- and is intended to be judged on the 20-symbol universe where the "
        "pooled count can actually reach a judgeable n."
    ),
)
def cp_hs_neckline(ctx, span, prominence, k, expiry):
    f4 = ctx.frame("4h")
    inv = _hs_pattern(
        f4, span=span, tol=0.03, prominence=prominence, max_gap=90,
        expiry=expiry, inverse=True,
    )
    hs = _hs_pattern(
        f4, span=span, tol=0.03, prominence=prominence, max_gap=90,
        expiry=expiry, inverse=False,
    )
    atr = ctx.align(ta.atr(f4, 14), "4h")
    i_neck = ctx.align(inv["neck"], "4h")
    i_guard = ctx.align(inv["guard"], "4h")
    h_neck = ctx.align(hs["neck"], "4h")
    h_guard = ctx.align(hs["guard"], "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = _blank(ctx.n)
    long_sig = _cross_up(close, i_neck)
    short_sig = _cross_down(close, h_neck)
    entry[long_sig] = LONG
    entry[short_sig & ~long_sig] = SHORT
    guard = np.where(entry == LONG, i_guard, h_guard)
    sd = _stop(close, atr, k, guard)
    return Plan(entry=entry, stop_dist=sd, target_dist=2.0 * sd, note="H&S neckline break")


@register(
    family="chartpattern",
    grid={"period": [30, 40, 55], "max_width": [0.04, 0.06, 0.09], "k": [1.5, 2.0]},
    rationale=(
        "Break out of a TIGHT trailing range -- the most sample-rich and most "
        "defensible member of the family, and the one with an independent "
        "evidence base (it is Donchian with a consolidation-width filter). The "
        "width gate is what makes the risk unit small relative to the move, which "
        "is the only mechanism by which pattern geometry could beat a plain "
        "channel break."
    ),
)
def cp_range_break(ctx, period, max_width, k):
    f4 = ctx.frame("4h")
    brk = ta.breakout_of_range(f4, period=period, max_width=max_width)
    atr = ctx.align(ta.atr(f4, 14), "4h")
    up = ctx.align(brk["up"].astype(float), "4h", fill=0.0) > 0.5
    dn = ctx.align(brk["down"].astype(float), "4h", fill=0.0) > 0.5
    lo = ctx.align(ta.rolling_low(f4, period), "4h")
    hi = ctx.align(ta.rolling_high(f4, period), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    # Only the first trigger bar of each aligned 4H breakout state, so a single
    # 4H break does not re-arm on all four of its 1H children.
    first_up = up & ~np.roll(up, 1)
    first_dn = dn & ~np.roll(dn, 1)
    first_up[0] = first_dn[0] = False

    entry = _blank(ctx.n)
    entry[first_up] = LONG
    entry[first_dn & ~first_up] = SHORT
    guard = np.where(entry == LONG, lo, hi)
    sd = _stop(close, atr, k, guard)
    return Plan(entry=entry, stop_dist=sd, target_dist=2.0 * sd, note="tight-range breakout")


@register(
    family="chartpattern",
    grid={"period": [30, 40, 55], "contraction": [0.4, 0.55, 0.7], "k": [1.5, 2.0]},
    rationale=(
        "Triangle/coil compression release: when the recent range is a small "
        "fraction of its own immediate past, the eventual expansion has to pick a "
        "direction, and the first close beyond the coil's extreme is that choice. "
        "This is the measurement-based form of the retired triangle detector, "
        "with no fitted trendlines to overfit."
    ),
)
def cp_squeeze_expand(ctx, period, contraction, k):
    f4 = ctx.frame("4h")
    coil = ta.triangle_squeeze(f4, period=period, min_contraction=contraction)
    half = max(2, period // 2)
    hi = ta.rolling_high(f4, half)
    lo = ta.rolling_low(f4, half)
    # The coil must have been present on the PRIOR bar; the current bar is the
    # release. shift(1) here is a backward shift -- strictly causal.
    armed = coil.shift(1).fillna(False)
    up = ctx.align((armed & (f4["close"] > hi)).astype(float), "4h", fill=0.0) > 0.5
    dn = ctx.align((armed & (f4["close"] < lo)).astype(float), "4h", fill=0.0) > 0.5
    atr = ctx.align(ta.atr(f4, 14), "4h")
    a_hi = ctx.align(hi, "4h")
    a_lo = ctx.align(lo, "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    first_up = up & ~np.roll(up, 1)
    first_dn = dn & ~np.roll(dn, 1)
    first_up[0] = first_dn[0] = False
    entry = _blank(ctx.n)
    entry[first_up] = LONG
    entry[first_dn & ~first_up] = SHORT
    guard = np.where(entry == LONG, a_lo, a_hi)
    sd = _stop(close, atr, k, guard)
    return Plan(entry=entry, stop_dist=sd, target_dist=2.0 * sd, note="coil release")


@register(
    family="chartpattern",
    grid={"pole": [8, 12], "consol": [8, 12], "pole_min": [0.03, 0.05], "k": [1.5, 2.0]},
    rationale=(
        "Bull/bear flag as trend CONTINUATION, gated by the 1D EMA(50) so the "
        "flag is only taken in the direction the higher timeframe already "
        "favours. The retired implementation had no such gate, which is the "
        "mechanical reason a flag detector fires constantly and fades the "
        "prevailing trend half the time; the trend filter is the specific fix "
        "being re-tested here."
    ),
)
def cp_flag_trend(ctx, pole, consol, pole_min, k):
    f4 = ctx.frame("4h")
    f1d = ctx.frame("1d")
    bull = ta.bull_flag(f4, pole=pole, consol=consol, pole_min=pole_min)
    bear = ta.bear_flag(f4, pole=pole, consol=consol, pole_min=pole_min)
    # Breakout of the consolidation is the trigger, not the flag itself.
    up = ctx.align(
        (bull & (f4["close"] > ta.rolling_high(f4, consol))).astype(float), "4h", fill=0.0
    ) > 0.5
    dn = ctx.align(
        (bear & (f4["close"] < ta.rolling_low(f4, consol))).astype(float), "4h", fill=0.0
    ) > 0.5
    trend = ctx.align(ta.ema(f1d["close"], 50), "1d")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    guard_lo = ctx.align(ta.rolling_low(f4, consol), "4h")
    guard_hi = ctx.align(ta.rolling_high(f4, consol), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    first_up = up & ~np.roll(up, 1) & (close > trend)
    first_dn = dn & ~np.roll(dn, 1) & (close < trend)
    first_up[0] = first_dn[0] = False
    entry = _blank(ctx.n)
    entry[first_up] = LONG
    entry[first_dn & ~first_up] = SHORT
    guard = np.where(entry == LONG, guard_lo, guard_hi)
    sd = _stop(close, atr, k, guard)
    return Plan(entry=entry, stop_dist=sd, target_dist=2.0 * sd, note="flag continuation")


@register(
    family="chartpattern",
    grid={"period": [30, 40, 55], "max_width": [0.06, 0.09], "within": [3, 6], "k": [1.5, 2.0]},
    rationale=(
        "Failed-breakout reversal: a close beyond a tight range that is REJECTED "
        "back inside within a few bars traps the breakout crowd, and the stops "
        "they leave behind fuel the move the other way. Same geometry as "
        "cp_range_break but the opposite side of it -- if the family's breakouts "
        "lose money, this is where that money should be, and testing both sides "
        "is what makes the re-test informative rather than one-sided."
    ),
)
def cp_failed_break(ctx, period, max_width, within, k):
    f4 = ctx.frame("4h")
    brk = ta.breakout_of_range(f4, period=period, max_width=max_width)
    hi = ta.rolling_high(f4, period)
    lo = ta.rolling_low(f4, period)
    # Carry the broken level forward for `within` bars; a close back inside that
    # level while the carry is live is the failure.
    up_marks = [(int(i), float(hi.to_numpy()[i])) for i in np.flatnonzero(brk["up"].to_numpy())]
    dn_marks = [(int(i), float(lo.to_numpy()[i])) for i in np.flatnonzero(brk["down"].to_numpy())]
    up_lvl = ctx.align(_carry(f4.index, up_marks, within), "4h")
    dn_lvl = ctx.align(_carry(f4.index, dn_marks, within), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    a_hi = ctx.align(hi, "4h")
    a_lo = ctx.align(lo, "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    short_sig = _cross_down(close, up_lvl)
    long_sig = _cross_up(close, dn_lvl)
    entry = _blank(ctx.n)
    entry[long_sig] = LONG
    entry[short_sig & ~long_sig] = SHORT
    # Stop beyond the extreme the failed breakout printed.
    guard = np.where(entry == LONG, a_lo, a_hi)
    sd = _stop(close, atr, k, guard)
    return Plan(entry=entry, stop_dist=sd, target_dist=2.0 * sd, note="failed breakout")


@register(
    family="chartpattern",
    grid={"span": [3, 5, 8], "near_atr": [0.3, 0.5], "k": [1.5, 2.0], "trend": [50, 100]},
    rationale=(
        "Support/resistance retest from confirmed swing levels: buy the first "
        "close back above a support pivot that price has just probed, sell the "
        "mirror. This is the structure family's cleanest idea -- the level is an "
        "observable price other participants also see -- and the 1D EMA gate "
        "keeps a support buy out of an established downtrend."
    ),
)
def cp_sr_retest(ctx, span, near_atr, k, trend):
    f4 = ctx.frame("4h")
    lv = ta.swing_levels(f4, span)
    atr4 = ta.atr(f4, 14)
    # "Probed": the bar's low reached within near_atr*ATR of support but the bar
    # closed back above it. Completes at the bar's own close -> causal.
    touch_sup = (f4["low"] <= lv["sup"] + near_atr * atr4) & (f4["close"] > lv["sup"])
    touch_res = (f4["high"] >= lv["res"] - near_atr * atr4) & (f4["close"] < lv["res"])
    up = ctx.align(touch_sup.astype(float), "4h", fill=0.0) > 0.5
    dn = ctx.align(touch_res.astype(float), "4h", fill=0.0) > 0.5
    ema_t = ctx.align(ta.ema(ctx.frame("1d")["close"], trend), "1d")
    atr = ctx.align(atr4, "4h")
    sup = ctx.align(lv["sup"], "4h")
    res = ctx.align(lv["res"], "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    first_up = up & ~np.roll(up, 1) & (close > ema_t)
    first_dn = dn & ~np.roll(dn, 1) & (close < ema_t)
    first_up[0] = first_dn[0] = False
    entry = _blank(ctx.n)
    entry[first_up] = LONG
    entry[first_dn & ~first_up] = SHORT
    guard = np.where(entry == LONG, sup, res)
    sd = _stop(close, atr, k, guard)
    return Plan(entry=entry, stop_dist=sd, target_dist=2.0 * sd, note="S/R retest")


@register(
    family="chartpattern",
    grid={"period": [30, 40, 55], "max_width": [0.06, 0.09], "vol_min": [1.3, 1.8], "k": [1.5, 2.0]},
    rationale=(
        "cp_range_break plus a volume-expansion gate: a consolidation break on "
        "below-average volume is the classic false break, so requiring "
        "vol_ratio(4H) above ~1.5 (production's VOLUME_HIGH_RATIO notion) should "
        "cut exactly the trades that lose. Registered alongside the ungated "
        "version so the volume filter's contribution is measurable rather than "
        "assumed."
    ),
)
def cp_range_break_vol(ctx, period, max_width, vol_min, k):
    f4 = ctx.frame("4h")
    brk = ta.breakout_of_range(f4, period=period, max_width=max_width)
    vr = ta.vol_ratio(f4, 20)
    up = ctx.align((brk["up"] & (vr >= vol_min)).astype(float), "4h", fill=0.0) > 0.5
    dn = ctx.align((brk["down"] & (vr >= vol_min)).astype(float), "4h", fill=0.0) > 0.5
    atr = ctx.align(ta.atr(f4, 14), "4h")
    lo = ctx.align(ta.rolling_low(f4, period), "4h")
    hi = ctx.align(ta.rolling_high(f4, period), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    first_up = up & ~np.roll(up, 1)
    first_dn = dn & ~np.roll(dn, 1)
    first_up[0] = first_dn[0] = False
    entry = _blank(ctx.n)
    entry[first_up] = LONG
    entry[first_dn & ~first_up] = SHORT
    guard = np.where(entry == LONG, lo, hi)
    sd = _stop(close, atr, k, guard)
    return Plan(
        entry=entry, stop_dist=sd, target_dist=2.0 * sd, note="tight-range breakout + volume"
    )


@register(
    family="chartpattern",
    grid={"span": [3, 5], "tol": [0.015, 0.03], "k": [1.5, 2.0], "adx_max": [20, 25, 30]},
    rationale=(
        "Double top/bottom neckline break, but only when the 1D ADX says the "
        "market is NOT already trending. Reversal geometry is a claim that a move "
        "is exhausted; taking it against a strong 1D trend is the single most "
        "likely reason the retired detector lost. If chart patterns survive "
        "anywhere, a regime gate is where."
    ),
)
def cp_double_ranging(ctx, span, tol, k, adx_max):
    f4 = ctx.frame("4h")
    bottom = _double_pattern(f4, span=span, tol=tol, max_gap=60, expiry=18, top=False)
    top = _double_pattern(f4, span=span, tol=tol, max_gap=60, expiry=18, top=True)
    adx = ctx.align(ta.adx(ctx.frame("1d"), 14), "1d")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    b_neck = ctx.align(bottom["neck"], "4h")
    b_guard = ctx.align(bottom["guard"], "4h")
    t_neck = ctx.align(top["neck"], "4h")
    t_guard = ctx.align(top["guard"], "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    ranging = adx < adx_max
    entry = _blank(ctx.n)
    long_sig = _cross_up(close, b_neck) & ranging
    short_sig = _cross_down(close, t_neck) & ranging
    entry[long_sig] = LONG
    entry[short_sig & ~long_sig] = SHORT
    guard = np.where(entry == LONG, b_guard, t_guard)
    sd = _stop(close, atr, k, guard)
    return Plan(entry=entry, stop_dist=sd, target_dist=2.0 * sd, note="double pattern, ranging only")
