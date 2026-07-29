"""TREND family: breakouts, moving-average structure, channel following.

WHY THIS FAMILY AT ALL
---------------------
Trend following is the deepest-evidenced directional edge in the literature
(Sharpe 0.5-1.5 for MA/Donchian on BTC over a decade, arXiv 2009.12155), and
the PRD's chosen A-core is a regime-gated Donchian breakout. The baseline module
already pins the exact production rule; everything here is a variant that
changes ONE structural idea at a time so the report can attribute any difference
to that idea rather than to a parameter.

DESIGN CONSTRAINTS OBSERVED THROUGHOUT
--------------------------------------
- **Stops are frozen at ``K_STOP * ATR(4H)``**, k = 1.5. The PRD names sweeping
  ``k`` as a consumed degree of freedom that already came back empty, and Phase 4
  re-derived 1.5 from the ``c <= 0.10`` cost constraint alone. Sweeping it here
  would both re-spend that DoF and move ``cost_ratio`` around, making the
  cost gate a fitted quantity. Trail and exit-channel multiples ARE swept --
  they are exit geometry, not the risk unit, and do not touch ``median_risk_pct``.
- **4H setup / 1H trigger / 1D regime.** 1m and 15m are on the wrong side of the
  cost frontier (mean Sharpe -12.71 vs +0.791 at 60m across 81 WF configs).
- **Grids stay small.** Every combo is charged against the Deflated Sharpe Ratio,
  so each strategy is <= 18 combos and axes carry 2-3 values centred on textbook
  settings (Donchian 20/55, ATR 14, ADX 25, EMA 21/50/100).
- **Causality.** Coarse data enters only via ``ctx.align``. Every derived
  quantity below is either an aligned trailing indicator or a function of the
  current and STRICTLY EARLIER trigger bars (``_prev``, ``_cross_up``,
  ``_recent``). No centered windows, no bfill, no full-sample statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ta
from core import LONG, SHORT, Plan
from registry import register

# Frozen risk unit. See the module docstring: this is NOT a grid axis anywhere.
K_STOP = 1.5
ATR_LEN = 14


# ---------------------------------------------------------------------------
# Small causal helpers.
#
# All three read the current bar and earlier bars only. They are written on the
# TRIGGER grid, which is legitimate for arrays that arrived through ctx.align:
# align maps bar i to the last coarse bar CLOSED by i, so comparing bar i's
# aligned value with bar i-1's compares two already-settled coarse readings.
# ---------------------------------------------------------------------------


def _prev(a: np.ndarray) -> np.ndarray:
    """The previous trigger bar's value, NaN at bar 0. Never looks forward."""
    out = np.full(len(a), np.nan, dtype=float)
    out[1:] = np.asarray(a, dtype=float)[:-1]
    return out


def _cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True on the first bar where ``a`` moves from <= b to > b."""
    pa, pb = _prev(a), _prev(b)
    return (a > b) & (pa <= pb)


def _cross_dn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True on the first bar where ``a`` moves from >= b to < b."""
    pa, pb = _prev(a), _prev(b)
    return (a < b) & (pa >= pb)


def _recent(mask: np.ndarray, window: int) -> np.ndarray:
    """True if ``mask`` was True on any of the previous ``window`` bars.

    The current bar is EXCLUDED (the window is shifted), which is what a
    "a break already happened, now wait for the pullback" rule needs: the break
    bar and the entry bar must be different bars.
    """
    s = pd.Series(np.asarray(mask, dtype=float))
    return s.rolling(window, min_periods=1).max().shift(1).fillna(0.0).to_numpy() > 0


def _setup_atr(ctx) -> np.ndarray:
    """ATR(14) of the 4H setup tier, on the trigger grid. The risk unit."""
    return ctx.align(ta.atr(ctx.frame("4h"), period=ATR_LEN), "4h")


def _regime_adx(ctx) -> np.ndarray:
    """Wilder ADX(14) on the 1D regime tier -- the one measured-healthy layer."""
    return ctx.align(ta.adx(ctx.frame("1d"), 14), "1d")


# ---------------------------------------------------------------------------
# 1. Donchian breakout: channel length and the trend filter
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"chan": [20, 34, 55], "adx_min": [20, 25, 30], "use_mid": [True, False]},
    rationale=(
        "The canonical Turtle breakout: a close beyond the N-bar extreme of the "
        "4H setup tier means the market has cleared every price at which recent "
        "sellers were willing to transact, so the remaining supply must be found "
        "at higher prices. The ADX gate exists because the same break inside a "
        "range is a liquidity-provision event, not a trend start. This variant "
        "asks only two questions the production rule fixes by fiat: does channel "
        "length matter, and does the 55-mid trend filter earn its keep?"
    ),
)
def trend_donchian_len(ctx, chan, adx_min, use_mid):
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    atr = _setup_atr(ctx)
    adx = _regime_adx(ctx)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    if use_mid:
        mid = ctx.align(
            (ta.rolling_high(f4, 55, exclude_current=False)
             + ta.rolling_low(f4, 55, exclude_current=False)) / 2.0,
            "4h",
        )
        up_ok, dn_ok = close > mid, close < mid
    else:
        up_ok = dn_ok = np.ones(ctx.n, dtype=bool)

    trending = adx > adx_min
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(close > hi) & up_ok & trending] = LONG
    entry[(close < lo) & dn_ok & trending] = SHORT
    return Plan(entry=entry, stop_dist=K_STOP * atr, note=f"donchian{chan}")


# ---------------------------------------------------------------------------
# 2. Donchian retest: buy the pullback, not the break
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"chan": [20, 55], "wait": [6, 12, 24], "adx_min": [20, 25]},
    rationale=(
        "Immediate breakout entries pay the worst price of the move and are the "
        "side of the trade that stop-hunting liquidity sweeps are designed to "
        "harvest. If the break is real, the broken level becomes support, so "
        "waiting for price to return to it converts the same thesis into a "
        "cheaper entry with the stop no further away -- which raises R per "
        "trade without needing a better hit rate. The cost is missed runners "
        "that never look back; ``wait`` bounds how long we are willing to miss."
    ),
)
def trend_donchian_retest(ctx, chan, wait, adx_min):
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    atr = _setup_atr(ctx)
    adx = _regime_adx(ctx)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    broke_up = close > hi
    broke_dn = close < lo
    trending = adx > adx_min

    # Pullback: a break happened in the last `wait` bars (current bar excluded)
    # and price has now come back INSIDE the channel but is still above/below
    # the level's midpoint side, i.e. it has not given the whole move back.
    band = 0.5 * atr
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[_recent(broke_up, wait) & (close <= hi) & (close >= hi - band) & trending] = LONG
    entry[_recent(broke_dn, wait) & (close >= lo) & (close <= lo + band) & trending] = SHORT
    return Plan(entry=entry, stop_dist=K_STOP * atr, note=f"retest{chan}/{wait}")


# ---------------------------------------------------------------------------
# 3. Donchian + volume confirmation
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"chan": [20, 55], "vol_min": [1.2, 1.5, 2.0], "adx_min": [20, 25]},
    rationale=(
        "A breakout on thin volume is one participant lifting a thin book; a "
        "breakout on 1.5x average volume means real size had to be absorbed at "
        "the new price. Volume is the only direct evidence of participation we "
        "have, and the production engine already grades it, so testing it as a "
        "hard gate rather than a label is a one-line structural question."
    ),
)
def trend_donchian_volume(ctx, chan, vol_min, adx_min):
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    atr = _setup_atr(ctx)
    adx = _regime_adx(ctx)
    # Volume ratio on the TRIGGER tier: the confirming bar is the one that broke.
    vr = ta.vol_ratio(ctx.trigger, 24).to_numpy(dtype=float)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    trending = (adx > adx_min) & (vr >= vol_min)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(close > hi) & trending] = LONG
    entry[(close < lo) & trending] = SHORT
    return Plan(entry=entry, stop_dist=K_STOP * atr, note=f"donch-vol{chan}")


# ---------------------------------------------------------------------------
# 4. Turtle: wide entry channel, narrow exit channel
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"chan": [20, 55], "exit_chan": [10, 20], "trail_k": [2.0, 3.0, 99.0]},
    long_only=True,
    rationale=(
        "The original Turtle system's asymmetry: enter on an N-bar extreme but "
        "exit on a SHORTER OPPOSITE-channel extreme. The asymmetry is the whole "
        "point -- it keeps you in while the trend merely pauses and removes you "
        "when it makes a genuine lower low, which is the only way a 30-40% "
        "hit-rate system with 3-5x winners survives. ``trail_k=99`` disables the "
        "ATR trail so the exit channel can be judged on its own; the trail "
        "multiple is exit geometry and does not move the risk unit. "
        "LONG-ONLY BY NECESSITY, not by preference: ``Plan.exit_signal`` is "
        "direction-agnostic by contract, so 'exit longs on the N-bar low' and "
        "'exit shorts on the N-bar high' cannot both be expressed in one Plan -- "
        "an entry-side breakout always satisfies the opposite side's exit "
        "immediately, which flattens the position on the very next bar. "
        "Restricting to longs makes the exit channel unambiguous and the test "
        "honest; the short leg needs a harness that carries direction into the "
        "exit array, and is out of scope here."
    ),
)
def trend_turtle(ctx, chan, exit_chan, trail_k):
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    ex_lo = ctx.align(ta.rolling_low(f4, exit_chan), "4h")
    atr = _setup_atr(ctx)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[close > hi] = LONG
    # Long-only, so the exit channel is unambiguously the LOW side.
    exit_sig = close < ex_lo
    trail = np.full(ctx.n, np.nan) if trail_k > 10 else trail_k * atr
    return Plan(
        entry=entry, stop_dist=K_STOP * atr, exit_signal=exit_sig,
        trail_atr=trail, note=f"turtle{chan}/{exit_chan}",
    )


# ---------------------------------------------------------------------------
# 5. EMA crossover with optional 1D agreement
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"fast": [12, 21], "slow": [50, 100], "confirm_1d": [True, False]},
    rationale=(
        "A fast/slow MA cross is the oldest published trend rule and the one "
        "with the least to overfit: it states that the recent average price has "
        "moved decisively away from the longer average, which is the definition "
        "of a change in the drift term. Its known weakness is whipsaw in ranges, "
        "so the variant tests whether requiring the 1D EMA structure to agree "
        "(dual-timeframe trend agreement) removes enough of them to pay for the "
        "trades it forfeits."
    ),
)
def trend_ema_cross(ctx, fast, slow, confirm_1d):
    f4, f1d = ctx.frame("4h"), ctx.frame("1d")
    ef = ctx.align(ta.ema(f4["close"], fast), "4h")
    es = ctx.align(ta.ema(f4["close"], slow), "4h")
    atr = _setup_atr(ctx)

    if confirm_1d:
        d_f = ctx.align(ta.ema(f1d["close"], 20), "1d")
        d_s = ctx.align(ta.ema(f1d["close"], 50), "1d")
        up_ok, dn_ok = d_f > d_s, d_f < d_s
    else:
        up_ok = dn_ok = np.ones(ctx.n, dtype=bool)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[_cross_up(ef, es) & up_ok] = LONG
    entry[_cross_dn(ef, es) & dn_ok] = SHORT
    # Re-cross closes the position: the thesis that produced it has expired.
    exit_sig = _cross_dn(ef, es) | _cross_up(ef, es)
    return Plan(
        entry=entry, stop_dist=K_STOP * atr, exit_signal=exit_sig,
        note=f"ema{fast}/{slow}",
    )


# ---------------------------------------------------------------------------
# 6. MA ribbon alignment across 1D and 4H
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"short": [10, 21], "long": [50, 100], "adx_min": [0, 25]},
    rationale=(
        "Rather than trading the cross event, this trades the STATE: enter only "
        "while both the 1D and 4H ribbons are stacked in the same direction, "
        "and time the entry with a pullback-and-reclaim of the 4H fast MA. "
        "Requiring agreement across a 6x timeframe ratio is a cheap way to "
        "demand that the drift be visible at two horizons, which noise rarely "
        "manages; using the reclaim as the trigger avoids buying the extension."
    ),
)
def trend_ribbon(ctx, short, long, adx_min):
    f4, f1d = ctx.frame("4h"), ctx.frame("1d")
    s4 = ctx.align(ta.ema(f4["close"], short), "4h")
    l4 = ctx.align(ta.ema(f4["close"], long), "4h")
    s1 = ctx.align(ta.ema(f1d["close"], short), "1d")
    l1 = ctx.align(ta.ema(f1d["close"], long), "1d")
    atr = _setup_atr(ctx)
    adx = _regime_adx(ctx)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    trending = adx > adx_min if adx_min > 0 else np.ones(ctx.n, dtype=bool)
    stacked_up = (s4 > l4) & (s1 > l1)
    stacked_dn = (s4 < l4) & (s1 < l1)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[stacked_up & trending & _cross_up(close, s4)] = LONG
    entry[stacked_dn & trending & _cross_dn(close, s4)] = SHORT
    # Leave when the 4H ribbon itself unstacks -- the state, not the price, is
    # what authorised the trade.
    exit_sig = ~(stacked_up | stacked_dn)
    return Plan(
        entry=entry, stop_dist=K_STOP * atr, exit_signal=exit_sig,
        note=f"ribbon{short}/{long}",
    )


# ---------------------------------------------------------------------------
# 7. Keltner channel following with an ATR trail
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"period": [20, 50], "mult": [1.5, 2.0, 2.5], "trail_k": [2.0, 3.0]},
    rationale=(
        "Keltner is the volatility-normalised cousin of Donchian: the trigger is "
        "'price is more than m ATR above its own EMA', so the same threshold "
        "means the same statistical surprise in a quiet and a violent market, "
        "whereas an N-bar high means very different things in each. Pairing it "
        "with a ratcheting ATR trail rather than a fixed target is the classic "
        "let-winners-run structure, and the trail is the only exit that can "
        "capture a move whose size we did not have to predict in advance."
    ),
)
def trend_keltner_trail(ctx, period, mult, trail_k):
    f4 = ctx.frame("4h")
    kc = ta.keltner(f4, period=period, atr_period=ATR_LEN, mult=mult)
    up = ctx.align(kc["upper"], "4h")
    dn = ctx.align(kc["lower"], "4h")
    atr = _setup_atr(ctx)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[close > up] = LONG
    entry[close < dn] = SHORT
    return Plan(
        entry=entry, stop_dist=K_STOP * atr, trail_atr=trail_k * atr,
        note=f"keltner{period}x{mult}",
    )


# ---------------------------------------------------------------------------
# 8. Price-vs-MA distance in ATR units
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"ma_len": [50, 100], "dist": [0.5, 1.0, 1.5], "adx_min": [20, 25]},
    rationale=(
        "A dimensionless statement of trend strength: price is ``dist`` ATRs "
        "away from its own moving average. Unlike a breakout it does not need a "
        "specific level to be cleared, so it fires during steady grinding trends "
        "that never print a clean N-bar high -- the regime where channel systems "
        "are structurally blind. Measuring the gap in ATR units keeps the same "
        "threshold meaningful on BTC and on DOGE."
    ),
)
def trend_ma_distance(ctx, ma_len, dist, adx_min):
    f4 = ctx.frame("4h")
    ma = ctx.align(ta.ema(f4["close"], ma_len), "4h")
    atr = _setup_atr(ctx)
    adx = _regime_adx(ctx)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    trending = adx > adx_min
    gap = (close - ma) / atr
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(gap >= dist) & trending] = LONG
    entry[(gap <= -dist) & trending] = SHORT
    # Exit when price returns to its average: the displacement that defined the
    # trade has been fully retraced.
    exit_sig = np.abs(gap) < 0.1
    return Plan(
        entry=entry, stop_dist=K_STOP * atr, exit_signal=exit_sig,
        note=f"madist{ma_len}@{dist}",
    )


# ---------------------------------------------------------------------------
# 9. Time-series momentum on the 1D tier
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"lookback": [30, 60, 90], "slope_len": [20, 40], "trail_k": [2.0, 3.0]},
    rationale=(
        "Time-series momentum -- hold long while the trailing multi-month return "
        "is positive -- is the single most replicated anomaly in the asset-"
        "pricing literature (Moskowitz-Ooi-Pedersen, 58 instruments, 25 years) "
        "and needs no level, channel or crossover to define it. The linear-"
        "regression slope filter is added because a positive lookback return can "
        "be an artefact of where the window happens to start; requiring the "
        "fitted slope to agree demands that the path, not just the endpoints, "
        "trends."
    ),
)
def trend_tsmom(ctx, lookback, slope_len, trail_k):
    f1d = ctx.frame("1d")
    mom = ctx.align(ta.momentum(f1d["close"], lookback), "1d")
    slp = ctx.align(ta.linreg_slope(f1d["close"], slope_len), "1d")
    atr = _setup_atr(ctx)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(mom > 0) & (slp > 0)] = LONG
    entry[(mom < 0) & (slp < 0)] = SHORT
    # No exit_signal: the momentum STATE that authorises a long is identical to
    # the condition that would have to flip to close it, and a direction-agnostic
    # exit array cannot distinguish "long thesis expired" from "short thesis
    # confirmed". The ATR trail plus the stop carry the exits instead, which is
    # also how published time-series-momentum implementations handle it.
    return Plan(
        entry=entry, stop_dist=K_STOP * atr, trail_atr=trail_k * atr,
        note=f"tsmom{lookback}",
    )


# ---------------------------------------------------------------------------
# 10. Dual-timeframe agreement: 1D direction, 4H Donchian entry
# ---------------------------------------------------------------------------


@register(
    family="trend",
    grid={"chan": [20, 55], "d_chan": [20, 55], "adx_min": [20, 25]},
    rationale=(
        "Separates the two jobs a trend system does: the 1D channel decides "
        "WHICH WAY we are allowed to trade (the slow, statistically reliable "
        "question) and the 4H channel decides WHEN (the fast, noisy one). "
        "Forcing the direction decision onto the slower tier is the cheapest "
        "known defence against the failure mode that kills breakout systems -- "
        "taking shorts inside an uptrend's pullbacks."
    ),
)
def trend_dual_tf(ctx, chan, d_chan, adx_min):
    f4, f1d = ctx.frame("4h"), ctx.frame("1d")
    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    d_mid = ctx.align(
        (ta.rolling_high(f1d, d_chan, exclude_current=False)
         + ta.rolling_low(f1d, d_chan, exclude_current=False)) / 2.0,
        "1d",
    )
    atr = _setup_atr(ctx)
    adx = _regime_adx(ctx)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    trending = adx > adx_min
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(close > hi) & (close > d_mid) & trending] = LONG
    entry[(close < lo) & (close < d_mid) & trending] = SHORT
    return Plan(entry=entry, stop_dist=K_STOP * atr, note=f"dual{chan}/{d_chan}")
