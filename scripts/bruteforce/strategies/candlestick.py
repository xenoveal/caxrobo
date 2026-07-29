"""CANDLESTICK family: single- and multi-bar price-action patterns.

THE PRIOR, STATED UP FRONT
--------------------------
This is the weakest-evidenced family in the search and the module is written to
give it a fair test rather than a flattering one. Candlestick patterns are
extremely sample-rich, which means a naive sweep will always surface something
that "works"; the defensible questions are narrower:

1. **Does a pattern carry information at all once LOCATION is required?** A
   hammer in the middle of a range is a coin flip. A hammer printed at a 60-bar
   trailing low, on expanded volume, is a statement about who ran out of sellers.
   Every reversal strategy here is therefore location-conditioned; none of them
   fires on the pattern alone.
2. **Do patterns ADD anything as a confirmation filter on top of a trend
   signal?** ``cs_donchian_plain`` and ``cs_donchian_candle_confirm`` are a
   matched pair: identical channel, trend filter, stop, target and grid, with the
   ONLY difference being that the second additionally demands a decisive
   one-sided (marubozu-ish) or engulfing trigger bar. The Sharpe delta between
   the two IS the measured value of candlestick confirmation. If the delta is
   ~zero or negative, that is the finding and it should be reported as such.

WHY 4H IS THE TRIGGER TIER HERE
-------------------------------
A 1H engulfing bar is mostly microstructure; a 4H one represents a session's
worth of positioning, and the PRD's cost evidence says coarser is the right side
of the frontier (mean Sharpe -12.71 at 1m vs +0.791 at 60m). Every strategy in
this module triggers on 4H with 1D used only for regime/level context. That also
keeps ``cost_ratio`` survivable: ATR(14) on 4H is roughly 1.5-2.5% of price on
majors, versus ~0.4% on 1H.

STOPS: THE FAILURE MODE THIS FAMILY MUST AVOID
----------------------------------------------
The natural candlestick stop is "just beyond the wick", and on a small bar that
is a 0.3% stop -- exactly the adverse-selection machine that killed the previous
system (costs were 58-68% of the risk unit). Every stop here is
``max(wick_distance * BUFFER, K_STOP * ATR)``, so the pattern may WIDEN the risk
unit but can never shrink it below the frozen 1.5*ATR floor. ``median_risk_pct``
and ``cost_ratio`` are therefore structurally comparable to the trend family.

CAUSALITY
---------
Every detector in ``indicators`` used here completes at the current bar's close.
Coarse (1D) data enters only through ``ctx.align``. The only local helpers are
backward shifts. Nothing is centered, back-filled, or full-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ta
from core import LONG, SHORT, Plan
from registry import register

# Frozen risk unit and buffer. K_STOP is NOT a grid axis anywhere in the search
# (the PRD names sweeping it a consumed degree of freedom); WICK_BUFFER is fixed
# at a conventional 10% pad rather than swept, because it only ever matters when
# the wick already exceeds the ATR floor.
K_STOP = 1.5
ATR_LEN = 14
WICK_BUFFER = 1.1

TFS_4H = ("1d", "4h")


# ---------------------------------------------------------------------------
# Local causal helpers. Both look strictly backward.
# ---------------------------------------------------------------------------


def _prev(a: np.ndarray, n: int = 1) -> np.ndarray:
    """Value from ``n`` bars ago, NaN in the warmup. Never looks forward."""
    out = np.full(len(a), np.nan, dtype=float)
    if n < len(a):
        out[n:] = np.asarray(a, dtype=float)[: len(a) - n]
    return out


def _b(s: pd.Series) -> np.ndarray:
    """A pandas boolean/NaN Series as a plain bool array (NaN -> False)."""
    return s.fillna(False).to_numpy(dtype=bool)


def _stop_from_wick(wick: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """ATR-floored structural stop: never tighter than ``K_STOP * ATR``.

    This is the single most important line in the module -- see the module
    docstring. ``wick`` is the raw distance from entry to beyond the pattern's
    extreme; the floor is what stops the cost ratio exploding on small bars.
    """
    return np.maximum(np.asarray(wick, dtype=float) * WICK_BUFFER, K_STOP * atr)


def _trend_1d(ctx, period: int) -> tuple[np.ndarray, np.ndarray]:
    """1D trend state on the trigger grid: (up, down) by close vs EMA(period)."""
    f1d = ctx.frame("1d")
    ema = ctx.align(ta.ema(f1d["close"], period), "1d")
    c1d = ctx.align(f1d["close"], "1d")
    return c1d > ema, c1d < ema


# ---------------------------------------------------------------------------
# 1. Engulfing at a trailing extreme
# ---------------------------------------------------------------------------


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    max_hold_bars=30,
    grid={"look": [20, 40, 60], "vol_min": [1.0, 1.5], "rr": [1.5, 2.5, 3.5]},
    rationale=(
        "An engulfing bar is only informative where there is something to "
        "engulf: at a trailing extreme it means the marginal seller (buyer) who "
        "drove the low (high) was fully absorbed within one 4H session, which is "
        "a change in who is price-setting rather than a shape. Requiring the bar "
        "to print at a ``look``-bar low/high supplies that location, and the "
        "volume axis tests whether absorption needs participation to matter. "
        "The prior is weak: this is the most defensible version of a family the "
        "evidence does not support."
    ),
)
def cs_engulf_extreme(ctx, look, vol_min, rr):
    f = ctx.trigger
    atr = ta.atr(f, ATR_LEN).to_numpy(dtype=float)
    close = f["close"].to_numpy(dtype=float)
    low = f["low"].to_numpy(dtype=float)
    high = f["high"].to_numpy(dtype=float)

    # Location is tested on the bar's EXTREME, not its close: a bullish
    # engulfing bar by definition closes well off its low, so a close-based test
    # is unsatisfiable and yields zero trades (measured). "The bar probed the
    # trailing low and then engulfed" is the intended event.
    at_low = low <= ta.rolling_low(f, look).to_numpy(dtype=float) * 1.005
    at_high = high >= ta.rolling_high(f, look).to_numpy(dtype=float) * 0.995
    vol_ok = ta.vol_ratio(f, 20).to_numpy(dtype=float) >= vol_min

    bull = _b(ta.bullish_engulfing(f)) & at_low & vol_ok
    bear = _b(ta.bearish_engulfing(f)) & at_high & vol_ok

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[bull] = LONG
    entry[bear] = SHORT
    wick = np.where(entry == LONG, close - low, high - close)
    stop = _stop_from_wick(wick, atr)
    return Plan(
        entry=entry, stop_dist=stop, target_dist=rr * stop,
        note=f"engulf@extreme{look}",
    )


# ---------------------------------------------------------------------------
# 2. Pin bar (hammer / shooting star) with volume, WITH the 1D trend
# ---------------------------------------------------------------------------


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    max_hold_bars=30,
    grid={"ema": [50, 100], "vol_min": [1.0, 1.5], "rr": [1.5, 2.5, 3.5]},
    rationale=(
        "A hammer is a rejected probe lower. Taken as a standalone reversal it "
        "is noise, but taken as a PULLBACK-FAILURE inside an established 1D "
        "uptrend it is the classic 'buyers defended the dip' event and is "
        "directionally aligned with the only edge the PRD has evidence for "
        "(trend). Volume expansion is required because a rejection on no "
        "participation is just a thin print. This is a trend-continuation "
        "strategy that uses a candle for timing, not a reversal bet."
    ),
)
def cs_pinbar_trend(ctx, ema, vol_min, rr):
    f = ctx.trigger
    atr = ta.atr(f, ATR_LEN).to_numpy(dtype=float)
    close = f["close"].to_numpy(dtype=float)
    low = f["low"].to_numpy(dtype=float)
    high = f["high"].to_numpy(dtype=float)
    up, down = _trend_1d(ctx, ema)
    vol_ok = ta.vol_ratio(f, 20).to_numpy(dtype=float) >= vol_min

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[_b(ta.hammer(f)) & up & vol_ok] = LONG
    entry[_b(ta.shooting_star(f)) & down & vol_ok] = SHORT
    wick = np.where(entry == LONG, close - low, high - close)
    stop = _stop_from_wick(wick, atr)
    return Plan(
        entry=entry, stop_dist=stop, target_dist=rr * stop, note=f"pin+trend{ema}",
    )


# ---------------------------------------------------------------------------
# 3. Inside-bar compression break
# ---------------------------------------------------------------------------


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    max_hold_bars=30,
    grid={"n_inside": [1, 2], "ema": [50, 100], "rr": [2.0, 3.0, 4.0]},
    rationale=(
        "An inside bar is a one-bar volatility contraction: the market failed to "
        "extend either side of the prior bar's range. Volatility clusters, so a "
        "contraction is followed by an expansion more often than by another "
        "contraction, and the direction of the resolution is the tradeable part. "
        "Trading the break of the inside bar's own range, filtered to the 1D "
        "trend direction, is the price-action expression of the squeeze-release "
        "idea -- distinct from the volatility family because the trigger is a "
        "bar-relationship, not a band statistic."
    ),
)
def cs_inside_break(ctx, n_inside, ema, rr):
    f = ctx.trigger
    atr = ta.atr(f, ATR_LEN).to_numpy(dtype=float)
    close = f["close"].to_numpy(dtype=float)
    high = f["high"].to_numpy(dtype=float)
    low = f["low"].to_numpy(dtype=float)
    inside = ta.inside_bar(f)
    # Require n_inside consecutive inside bars ENDING on the previous bar, so the
    # compression is complete before the break bar is evaluated.
    compressed = _b(inside.rolling(n_inside, min_periods=n_inside).sum() == n_inside)
    compressed = _b(pd.Series(compressed, index=f.index).shift(1))

    up, down = _trend_1d(ctx, ema)
    prev_hi, prev_lo = _prev(high), _prev(low)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[compressed & (close > prev_hi) & up] = LONG
    entry[compressed & (close < prev_lo) & down] = SHORT
    # Risk is the opposite side of the compression range, ATR-floored.
    wick = np.where(entry == LONG, close - prev_lo, prev_hi - close)
    stop = _stop_from_wick(wick, atr)
    return Plan(
        entry=entry, stop_dist=stop, target_dist=rr * stop,
        note=f"inside{n_inside}-break",
    )


# ---------------------------------------------------------------------------
# 4. Three-bar reversal at a 1D swing level
# ---------------------------------------------------------------------------


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    max_hold_bars=30,
    grid={"span": [3, 5], "tol": [0.02, 0.04, 0.06], "rr": [2.0, 3.0]},
    rationale=(
        "The three-bar reversal already encodes both exhaustion and confirmation "
        "(two bars one way, then a bar that closes beyond the middle bar's "
        "extreme), so it needs less filtering than a single candle. What it still "
        "lacks is a reason for the turn to happen HERE, which the nearest "
        "confirmed 1D swing level supplies -- a level other participants can also "
        "see and defend. ``tol`` is the proximity band to that level and is kept "
        "to three conventional values."
    ),
)
def cs_three_bar_level(ctx, span, tol, rr):
    f = ctx.trigger
    atr = ta.atr(f, ATR_LEN).to_numpy(dtype=float)
    close = f["close"].to_numpy(dtype=float)
    low = f["low"].to_numpy(dtype=float)
    high = f["high"].to_numpy(dtype=float)

    lv = ta.swing_levels(ctx.frame("1d"), span)
    sup = ctx.align(lv["sup"], "1d")
    res = ctx.align(lv["res"], "1d")
    near_sup = np.abs(close - sup) <= tol * close
    near_res = np.abs(close - res) <= tol * close

    tbr = ta.three_bar_reversal(f)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[_b(tbr["bull"]) & near_sup] = LONG
    entry[_b(tbr["bear"]) & near_res] = SHORT
    # Stop beyond the 3-bar pattern's own extreme.
    lo3 = pd.Series(low, index=f.index).rolling(3, min_periods=3).min().to_numpy()
    hi3 = pd.Series(high, index=f.index).rolling(3, min_periods=3).max().to_numpy()
    wick = np.where(entry == LONG, close - lo3, hi3 - close)
    stop = _stop_from_wick(wick, atr)
    return Plan(
        entry=entry, stop_dist=stop, target_dist=rr * stop, note=f"3bar@1d-swing{span}",
    )


# ---------------------------------------------------------------------------
# 5. Marubozu / wide-range ignition
# ---------------------------------------------------------------------------


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    max_hold_bars=30,
    grid={"body_min": [0.8, 0.9], "range_mult": [1.0, 1.5, 2.0], "rr": [2.0, 3.0]},
    rationale=(
        "A marubozu is a bar with no meaningful wick: price opened, went one way "
        "and never traded back. That is a mechanical signature of one-sided order "
        "flow (an aggressor working a size), and unlike a reversal pattern it "
        "requires no assumption about who is exhausted. Requiring the bar's range "
        "to exceed ``range_mult * ATR`` separates a genuine ignition bar from a "
        "small quiet bar that happens to be one-sided -- the distinction the raw "
        "body-fraction detector cannot make. Traded as continuation."
    ),
)
def cs_marubozu_ignition(ctx, body_min, range_mult, rr):
    f = ctx.trigger
    atr = ta.atr(f, ATR_LEN).to_numpy(dtype=float)
    close = f["close"].to_numpy(dtype=float)
    open_ = f["open"].to_numpy(dtype=float)
    high = f["high"].to_numpy(dtype=float)
    low = f["low"].to_numpy(dtype=float)

    solid = _b(ta.marubozu(f, body_min=body_min))
    wide = (high - low) >= range_mult * atr
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[solid & wide & (close > open_)] = LONG
    entry[solid & wide & (close < open_)] = SHORT
    # Stop at the ignition bar's opposite extreme (which IS the bar's range here).
    wick = np.where(entry == LONG, close - low, high - close)
    stop = _stop_from_wick(wick, atr)
    return Plan(
        entry=entry, stop_dist=stop, target_dist=rr * stop,
        note=f"marubozu{body_min}x{range_mult}atr",
    )


# ---------------------------------------------------------------------------
# 6. Wick rejection at a 1D level
# ---------------------------------------------------------------------------


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    max_hold_bars=30,
    grid={"look": [30, 60], "wick_frac": [0.5, 0.6], "rr": [1.5, 2.5, 3.5]},
    rationale=(
        "This isolates the one mechanically meaningful part of a pin bar: price "
        "traded THROUGH a level that the whole market can see (a 1D-scale "
        "trailing extreme) and then closed back inside it. That is a failed "
        "breakout with a visible trapped side, and the stop is defined by the "
        "rejected wick rather than by a guess. ``wick_frac`` requires the "
        "rejected tail to be at least that fraction of the bar's range, so a bar "
        "that merely closed mid-range does not qualify."
    ),
)
def cs_wick_reject_level(ctx, look, wick_frac, rr):
    f = ctx.trigger
    atr = ta.atr(f, ATR_LEN).to_numpy(dtype=float)
    close = f["close"].to_numpy(dtype=float)
    high = f["high"].to_numpy(dtype=float)
    low = f["low"].to_numpy(dtype=float)
    rng = np.where(high > low, high - low, np.nan)

    hi_lvl = ta.rolling_high(f, look).to_numpy(dtype=float)
    lo_lvl = ta.rolling_low(f, look).to_numpy(dtype=float)

    upper = (high - np.maximum(close, f["open"].to_numpy(dtype=float))) / rng
    lower = (np.minimum(close, f["open"].to_numpy(dtype=float)) - low) / rng

    # Poked above the trailing high but closed back below it -> failed breakout.
    bear = (high > hi_lvl) & (close < hi_lvl) & (upper >= wick_frac)
    bull = (low < lo_lvl) & (close > lo_lvl) & (lower >= wick_frac)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[np.nan_to_num(bull, nan=0.0).astype(bool)] = LONG
    entry[np.nan_to_num(bear, nan=0.0).astype(bool)] = SHORT
    wick = np.where(entry == LONG, close - low, high - close)
    stop = _stop_from_wick(wick, atr)
    return Plan(
        entry=entry, stop_dist=stop, target_dist=rr * stop, note=f"wick-reject{look}",
    )


# ---------------------------------------------------------------------------
# 7. Consecutive-close streak, faded
# ---------------------------------------------------------------------------


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    max_hold_bars=18,
    grid={"n": [4, 5, 6], "rsi_ext": [70, 75], "rr": [1.0, 1.5]},
    rationale=(
        "Not really a 'pattern' but the same object: a run of N same-direction "
        "closes. It is included because it is the cleanest measurable version of "
        "the exhaustion premise every reversal candle relies on, so it acts as a "
        "control -- if streak-fading has no edge, the reversal candles' premise is "
        "unsupported regardless of shape. Paired with an RSI extreme so the run "
        "also has to be stretched, and a short hold, because a fade that needs "
        "weeks is a trend bet in disguise."
    ),
)
def cs_streak_fade(ctx, n, rsi_ext, rr):
    f = ctx.trigger
    atr = ta.atr(f, ATR_LEN).to_numpy(dtype=float)
    close_s = f["close"]
    up_bar = (close_s.diff() > 0).astype(float)
    dn_bar = (close_s.diff() < 0).astype(float)
    run_up = _b(up_bar.rolling(n, min_periods=n).sum() == n)
    run_dn = _b(dn_bar.rolling(n, min_periods=n).sum() == n)
    r = ta.rsi(close_s, 14).to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[run_up & (r >= rsi_ext)] = SHORT
    entry[run_dn & (r <= 100 - rsi_ext)] = LONG
    stop = K_STOP * atr
    return Plan(
        entry=entry, stop_dist=stop, target_dist=rr * stop, note=f"streak{n}-fade",
    )


# ---------------------------------------------------------------------------
# 8 & 9. THE MATCHED PAIR: does a candle filter add anything?
#
# These two must stay byte-for-byte identical apart from ``_candle_confirm``.
# Any other divergence destroys the comparison, which is the single most useful
# output of this module.
# ---------------------------------------------------------------------------


def _donchian_core(ctx, chan, ema, rr, confirm: np.ndarray | None):
    """Shared 4H Donchian-break engine. ``confirm`` gates entries if given."""
    f = ctx.trigger
    atr = ta.atr(f, ATR_LEN).to_numpy(dtype=float)
    close = f["close"].to_numpy(dtype=float)
    hi = ta.rolling_high(f, chan).to_numpy(dtype=float)
    lo = ta.rolling_low(f, chan).to_numpy(dtype=float)
    up, down = _trend_1d(ctx, ema)

    long_sig = (close > hi) & up
    short_sig = (close < lo) & down
    if confirm is not None:
        long_sig = long_sig & confirm
        short_sig = short_sig & confirm

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[long_sig] = LONG
    entry[short_sig] = SHORT
    stop = K_STOP * atr
    return entry, stop, rr * stop


def _candle_confirm(ctx) -> np.ndarray:
    """Trigger bar is DECISIVE: one-sided body, or an engulfing of the prior bar.

    Both readings are of the breakout bar itself and complete at its close.
    """
    f = ctx.trigger
    decisive = _b(ta.marubozu(f, body_min=0.7))
    engulf = _b(ta.bullish_engulfing(f)) | _b(ta.bearish_engulfing(f))
    return decisive | engulf


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    grid={"chan": [20, 40], "ema": [50, 100], "rr": [2.0, 3.0]},
    rationale=(
        "CONTROL ARM, deliberately not a candlestick idea: a plain 4H Donchian "
        "break in the direction of the 1D EMA trend, frozen 1.5*ATR stop. It "
        "exists so ``cs_donchian_candle_confirm`` has an honest baseline to be "
        "measured against; without it, any Sharpe the filtered version posts is "
        "uninterpretable. Reported in the candlestick family on purpose."
    ),
)
def cs_donchian_plain(ctx, chan, ema, rr):
    entry, stop, target = _donchian_core(ctx, chan, ema, rr, None)
    return Plan(entry=entry, stop_dist=stop, target_dist=target, note="control-unfiltered")


@register(
    family="candlestick",
    trigger_tf="4h",
    timeframes=TFS_4H,
    grid={"chan": [20, 40], "ema": [50, 100], "rr": [2.0, 3.0]},
    rationale=(
        "TREATMENT ARM: identical to ``cs_donchian_plain`` in every respect "
        "except that the breakout bar must also be decisive -- a body filling "
        ">=70% of its range, or an engulfing of the prior bar. The premise is "
        "that a channel break on a bar that closed mid-range is a probe, while "
        "one that closed on its extreme is a commitment, so the candle acts as a "
        "conviction proxy that the channel rule cannot see. Because the grids and "
        "stops match exactly, the per-symbol Sharpe difference between the two "
        "arms is a direct estimate of what candlestick confirmation is worth."
    ),
)
def cs_donchian_candle_confirm(ctx, chan, ema, rr):
    entry, stop, target = _donchian_core(ctx, chan, ema, rr, _candle_confirm(ctx))
    return Plan(entry=entry, stop_dist=stop, target_dist=target, note="candle-filtered")
