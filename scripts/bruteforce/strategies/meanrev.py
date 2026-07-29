"""Mean-reversion strategy family for the brute-force search.

WHY THIS FAMILY IS BEING RE-TESTED AT ALL
-----------------------------------------
The production Bollinger fade sleeve was DROPPED in Phase 6
(``.claude/PRPs/reports/fade-requalification.md``): pooled expectancy -0.3883%
on n=297, 0 of 3 symbols positive, and cost ratio ``c`` above the 0.10 ceiling
on all three (0.1812 / 0.1147 / 0.1001). The report's own diagnosis is that the
sleeve's *structural* stop -- the excursion extreme, a median of only
0.63-0.75 x ATR(4H) -- was too tight, so the fixed round-trip cost toll ate the
risk unit. It also found the R:R floor was ADVERSELY SELECTING the sleeve: the
setups it admitted were systematically the shallow ones with the tightest stops.

Every strategy in this module therefore places its stop as

    stop_dist = max(k * ATR(setup tf), floor_pct * close)

which is the remedy the Phase 6 report explicitly named as a legitimate Phase 7
grid axis ("an ATR floor on the fade stop"). ``ROUND_TRIP_COST`` is 0.0014, so a
median risk of 1.4% is exactly the ``c = 0.10`` ceiling; the ``floor_pct`` axis
is swept only over values at or above that boundary, so no combination in this
module can be admitted with a structurally cost-doomed stop. That is a design
constraint, not a fitted parameter -- it is derived from the frozen cost model
the same way the PRD derives ``k = 1.5``.

CONTRARY PRIOR, STATED UP FRONT
-------------------------------
Mean reversion is where cost ratio goes to die. Widening the stop fixes ``c``
but mechanically lowers R per winner, so the honest question this family answers
is whether the reversion edge survives being paid for at a *wide* stop. The
answer may well be no; a negative result here is a real result and is reported
as one.

DESIGN NOTES COMMON TO EVERY STRATEGY BELOW
-------------------------------------------
- Signals are computed on the 4H setup tier (or 1D regime tier) and reach the 1H
  trigger grid only through ``ctx.align``. Nothing is computed on the 1H tier
  itself except the execution close, so no strategy can act on information
  faster than the tier that produced it.
- Entries are gated by ``_fresh_bar``, which fires only on the FIRST 1H bar
  after a new setup-tier bar closes. Without it a persistent 4H condition would
  re-arm on all four 1H bars inside the window, quadrupling the apparent signal
  count and biasing entry timing toward whichever 1H bar happened to be cheapest.
- Exits prefer ``target_dist`` (directional) over ``exit_signal`` for
  mean-touch logic, because ``Plan.exit_signal`` is direction-agnostic: a
  "close above the mid-band" flag would correctly exit a long and incorrectly
  exit a short on the same bar. ``exit_signal`` is used only for genuinely
  symmetric conditions such as "the oscillator returned to its neutral zone".
"""

from __future__ import annotations

import numpy as np

import indicators as ta
from core import LONG, SHORT, Plan
from registry import register

# ---------------------------------------------------------------------------
# Shared helpers
#
# COST_FLOORS is deliberately a module constant, not a grid axis with wide
# values: 0.0014 round-trip cost / 0.014 risk = the 0.10 ceiling exactly, so
# these three values sit AT and just above the cost frontier. Sweeping below
# 1.4% would only manufacture combinations that the cost gate must reject.
# ---------------------------------------------------------------------------
COST_FLOORS = [0.014, 0.018, 0.024]


def _fresh_bar(ctx, timeframe: str) -> np.ndarray:
    """True on the first trigger bar after a new ``timeframe`` bar closed.

    Built from ``ctx.align`` on the coarse bar ordinal, so it inherits align's
    "last coarse bar whose close <= this trigger bar's close" rule and cannot
    see a partially formed coarse bar. Causal by construction: the value at
    trigger bar i depends only on which coarse bars had closed by i.
    """
    n_coarse = len(ctx.frames[timeframe])
    ordinal = ctx.align(np.arange(n_coarse, dtype=float), timeframe, fill=-1.0)
    prev = np.concatenate(([-1.0], ordinal[:-1]))
    return ordinal > prev


def _stop(atr_aligned: np.ndarray, close: np.ndarray, k: float, floor_pct: float) -> np.ndarray:
    """ATR-scaled stop with a hard percentage floor. The Phase 6 remedy.

    The ``max`` is what breaks the documented failure mode: a fade whose
    structural stop collapses to 0.6 x ATR in quiet tape can no longer be
    entered with that stop, so the cost ratio cannot silently blow out.
    """
    return np.maximum(k * atr_aligned, floor_pct * close)


def _blank(ctx) -> np.ndarray:
    return np.zeros(ctx.n, dtype=np.int8)


# ---------------------------------------------------------------------------
# 1-2. Connors-style RSI extremes
# ---------------------------------------------------------------------------


@register(
    family="meanrev",
    grid={"rsi_p": [2, 4], "lo": [5.0, 10.0, 15.0], "k": [1.5, 2.5], "floor_pct": COST_FLOORS},
    rationale=(
        "Connors RSI-2: a 2-4 period RSI at a single-digit reading means every "
        "recent bar closed down, which in a market with no forced seller is "
        "liquidation-driven rather than information-driven. The counterparty is "
        "a leveraged long being force-closed by the exchange's liquidation "
        "engine, and the reversion force is market makers who absorbed that "
        "flow re-hedging back to flat once the cascade stops. The 200-period 1D "
        "trend filter is what makes the counterparty story hold: buying an "
        "oversold reading in a structural downtrend is not absorbing panic, it "
        "is standing in front of real selling."
    ),
)
def mr_rsi2_trend(ctx, rsi_p, lo, k, floor_pct):
    f4 = ctx.frame("4h")
    f1d = ctx.frame("1d")
    rsi = ctx.align(ta.rsi(f4["close"], rsi_p), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    ma = ctx.align(ta.sma(f1d["close"], 200), "1d")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    up = close > ma
    entry = _blank(ctx)
    entry[fresh & up & (rsi <= lo)] = LONG
    entry[fresh & ~up & (rsi >= 100.0 - lo) & np.isfinite(ma)] = SHORT
    # Symmetric neutral-zone exit: works for both directions, so exit_signal is
    # legitimate here (see module docstring).
    exit_sig = (rsi >= 45.0) & (rsi <= 55.0)
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        exit_signal=exit_sig,
        note=f"rsi{rsi_p}-trendfiltered",
    )


@register(
    family="meanrev",
    grid={"rsi_p": [2, 4], "lo": [5.0, 10.0, 15.0], "k": [1.5, 2.5], "floor_pct": COST_FLOORS},
    rationale=(
        "The same RSI extreme WITHOUT the trend filter, registered as the "
        "controlled comparison rather than as an independent idea. If the "
        "filtered version is not materially better, the 200-MA gate is not "
        "buying anything and the reported edge of mr_rsi2_trend is trend "
        "exposure rather than reversion. Publishing both is how that "
        "attribution question gets answered instead of assumed."
    ),
)
def mr_rsi_raw(ctx, rsi_p, lo, k, floor_pct):
    f4 = ctx.frame("4h")
    rsi = ctx.align(ta.rsi(f4["close"], rsi_p), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    entry = _blank(ctx)
    entry[fresh & (rsi <= lo)] = LONG
    entry[fresh & (rsi >= 100.0 - lo)] = SHORT
    exit_sig = (rsi >= 45.0) & (rsi <= 55.0)
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        exit_signal=exit_sig,
        note=f"rsi{rsi_p}-nofilter",
    )


# ---------------------------------------------------------------------------
# 3-4. Bollinger fade: the dropped sleeve, rebuilt on an ATR-floored stop
# ---------------------------------------------------------------------------


@register(
    family="meanrev",
    grid={"bb_p": [20, 40], "bb_std": [2.0, 2.5], "k": [1.5, 2.5, 3.5], "floor_pct": COST_FLOORS},
    rationale=(
        "The DROPPED production sleeve, rebuilt with the one change the Phase 6 "
        "post-mortem identified: the stop is max(k*ATR, floor) instead of the "
        "excursion extreme, so it can no longer collapse to 0.6xATR and pay "
        "double the cost toll per unit risk. Economically, a 2-sigma excursion "
        "on a 20-bar window with no trend behind it is an inventory event -- a "
        "large taker order walking the book -- and the counterparty forced to "
        "revert it is the market maker who filled that order and is now short "
        "gamma against their own quote. The target is the mid-band, which is "
        "the maker's flat-inventory price. This registration exists to answer "
        "the live PRD question of whether the sleeve deserves reinstatement, "
        "and it is registered expecting it may still fail."
    ),
)
def mr_bb_fade_atr(ctx, bb_p, bb_std, k, floor_pct):
    f4 = ctx.frame("4h")
    bb = ta.bollinger(f4["close"], bb_p, bb_std)
    upper = ctx.align(bb["upper"], "4h")
    lower = ctx.align(bb["lower"], "4h")
    mid = ctx.align(bb["mid"], "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    c4 = ctx.align(f4["close"], "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    entry = _blank(ctx)
    entry[fresh & (c4 > upper)] = SHORT
    entry[fresh & (c4 < lower)] = LONG
    # Directional target: distance to the mid-band from the actual entry price.
    target = np.abs(mid - close)
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        target_dist=target,
        note="bb-fade-atr-floored",
    )


@register(
    family="meanrev",
    grid={"bb_p": [20, 40], "bb_std": [2.0, 2.5], "k": [1.5, 2.5, 3.5], "floor_pct": COST_FLOORS},
    rationale=(
        "'Walk the band' variant: fade only once the setup bar has closed back "
        "INSIDE the band after a prior close outside it. The economic argument "
        "is selection, not prediction -- a price that keeps closing outside the "
        "band is a trend printing new information, and fading it is fading "
        "real flow; a price that closes back inside has demonstrated the "
        "excursion was absorbed, so the reverting counterparty has already "
        "shown up. This is the standard fix for the fade's worst failure mode "
        "(being run over by the first leg of a trend) and costs one bar of edge "
        "to buy it."
    ),
)
def mr_bb_walkback(ctx, bb_p, bb_std, k, floor_pct):
    f4 = ctx.frame("4h")
    bb = ta.bollinger(f4["close"], bb_p, bb_std)
    c = f4["close"]
    out_up = (c > bb["upper"])
    out_dn = (c < bb["lower"])
    inside = (~out_up) & (~out_dn) & bb["upper"].notna()
    # shift(1) only: strictly past bars.
    reenter_short = (inside & out_up.shift(1).fillna(False)).to_numpy()
    reenter_long = (inside & out_dn.shift(1).fillna(False)).to_numpy()

    mid = ctx.align(bb["mid"], "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    rs = ctx.align(reenter_short.astype(float), "4h", fill=0.0) > 0.5
    rl = ctx.align(reenter_long.astype(float), "4h", fill=0.0) > 0.5
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    entry = _blank(ctx)
    entry[fresh & rs] = SHORT
    entry[fresh & rl] = LONG
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        target_dist=np.abs(mid - close),
        note="bb-walkback",
    )


# ---------------------------------------------------------------------------
# 5. Z-score reversion against a moving average
# ---------------------------------------------------------------------------


@register(
    family="meanrev",
    grid={"z_p": [24, 48, 96], "z_min": [2.0, 2.5, 3.0], "k": [2.0, 3.0], "floor_pct": COST_FLOORS},
    rationale=(
        "Bollinger bands with the band geometry removed: enter when price is "
        "|z| >= z_min standard deviations from its own trailing mean. Same "
        "counterparty as the fade (inventory-driven excursion absorbed by "
        "makers) but the threshold is expressed in units of realised dispersion "
        "rather than a fixed sigma multiple on a fixed window, so it "
        "self-calibrates across the 20-symbol universe where a single sigma "
        "level means very different things on BTC and DOGE. Target is the mean "
        "itself, i.e. z = 0."
    ),
)
def mr_zscore(ctx, z_p, z_min, k, floor_pct):
    f4 = ctx.frame("4h")
    z = ctx.align(ta.zscore(f4["close"], z_p), "4h")
    mu = ctx.align(ta.sma(f4["close"], z_p), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    entry = _blank(ctx)
    entry[fresh & (z <= -z_min)] = LONG
    entry[fresh & (z >= z_min)] = SHORT
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        target_dist=np.abs(mu - close),
        note="zscore-reversion",
    )


# ---------------------------------------------------------------------------
# 6. Keltner touch, faded only in a genuinely ranging tape (low ADX)
# ---------------------------------------------------------------------------


@register(
    family="meanrev",
    grid={"kc_mult": [1.5, 2.0, 2.5], "adx_max": [18.0, 22.0], "k": [2.0, 3.0], "floor_pct": COST_FLOORS},
    rationale=(
        "The direct complement to the trend sleeve: fade the Keltner band only "
        "when 1D ADX is BELOW adx_max, i.e. precisely the regime the Donchian "
        "engine refuses to trade (it requires ADX > 25). In a low-ADX tape "
        "there is by construction no directional flow to be run over by, so the "
        "counterparty at the band is a maker with unwanted inventory rather "
        "than an informed buyer. This is the one strategy here whose edge claim "
        "is about REGIME rather than about the entry trigger, and it is the "
        "cheapest test of whether the classifier's unserved 'ranging' bucket "
        "contains anything at all. Keltner rather than Bollinger because its "
        "width is ATR-based, so band touches and the ATR-floored stop are "
        "denominated in the same volatility unit and cannot drift apart."
    ),
)
def mr_keltner_lowadx(ctx, kc_mult, adx_max, k, floor_pct):
    f4 = ctx.frame("4h")
    kc = ta.keltner(f4, period=20, atr_period=14, mult=kc_mult)
    upper = ctx.align(kc["upper"], "4h")
    lower = ctx.align(kc["lower"], "4h")
    mid = ctx.align(kc["mid"], "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    adx = ctx.align(ta.adx(ctx.frame("1d"), 14), "1d")
    h4 = ctx.align(f4["high"], "4h")
    l4 = ctx.align(f4["low"], "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    ranging = adx < adx_max
    entry = _blank(ctx)
    entry[fresh & ranging & (h4 >= upper)] = SHORT
    entry[fresh & ranging & (l4 <= lower)] = LONG
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        target_dist=np.abs(mid - close),
        note="keltner-touch-lowadx",
    )


# ---------------------------------------------------------------------------
# 7. Stochastic extreme with a confirmation bar
# ---------------------------------------------------------------------------


@register(
    family="meanrev",
    grid={"st_p": [14, 28], "lo": [10.0, 20.0], "k": [2.0, 3.0], "floor_pct": COST_FLOORS},
    rationale=(
        "Stochastic %K at an extreme says the close is pinned at the edge of "
        "its own recent range; requiring the NEXT setup bar to turn back "
        "(%K rising off the low, still below the mid) is a cheap, "
        "parameter-free confirmation that the pin has failed. The confirmation "
        "bar is the whole point: the un-confirmed version of this trade is "
        "identical to the dropped fade sleeve's failure mode of entering while "
        "the excursion is still extending. Counterparty is the stop-loss flow "
        "clustered just beyond a well-defined range edge, which is mechanical "
        "selling that stops when the stop orders are exhausted."
    ),
)
def mr_stoch_confirm(ctx, st_p, lo, k, floor_pct):
    f4 = ctx.frame("4h")
    st = ta.stochastic(f4, period=st_p, smooth_k=3, smooth_d=3)
    kk = st["k"]
    prev = kk.shift(1)
    long_ok = ((prev <= lo) & (kk > prev) & (kk < 50.0)).to_numpy()
    short_ok = ((prev >= 100.0 - lo) & (kk < prev) & (kk > 50.0)).to_numpy()

    mid = ctx.align(ta.ema(f4["close"], 20), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    lo_a = ctx.align(long_ok.astype(float), "4h", fill=0.0) > 0.5
    hi_a = ctx.align(short_ok.astype(float), "4h", fill=0.0) > 0.5
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    entry = _blank(ctx)
    entry[fresh & lo_a] = LONG
    entry[fresh & hi_a] = SHORT
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        target_dist=np.abs(mid - close),
        note="stoch-confirmed",
    )


# ---------------------------------------------------------------------------
# 8. Rolling-VWAP stretch, measured in ATR units
# ---------------------------------------------------------------------------


@register(
    family="meanrev",
    grid={"vwap_p": [24, 48], "stretch": [1.5, 2.0, 3.0], "k": [2.0, 3.0], "floor_pct": COST_FLOORS},
    rationale=(
        "Rolling VWAP is the volume-weighted average price real participants "
        "actually paid over the window, so it is the reference an execution desk "
        "is benchmarked against. Price stretched stretch x ATR away from it "
        "means the marginal trade is being done far from everyone else's cost "
        "basis, and the reverting force is explicit: VWAP-benchmarked "
        "algorithmic execution is mandated to lean against its own slippage. "
        "This is the most institutionally grounded reversion story available "
        "without funding-rate data, and it is the only one here whose anchor is "
        "volume-weighted rather than time-weighted."
    ),
)
def mr_vwap_stretch(ctx, vwap_p, stretch, k, floor_pct):
    f4 = ctx.frame("4h")
    vw = ctx.align(ta.vwap_session(f4, period=vwap_p), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    c4 = ctx.align(f4["close"], "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    with np.errstate(invalid="ignore", divide="ignore"):
        dist = (c4 - vw) / atr
    entry = _blank(ctx)
    entry[fresh & (dist <= -stretch)] = LONG
    entry[fresh & (dist >= stretch)] = SHORT
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        target_dist=np.abs(vw - close),
        note="vwap-stretch",
    )


# ---------------------------------------------------------------------------
# 9. Williams %R extreme on the 1D regime tier
# ---------------------------------------------------------------------------


@register(
    family="meanrev",
    grid={"wr_p": [10, 20], "lo": [-95.0, -90.0], "k": [1.5, 2.5], "floor_pct": COST_FLOORS},
    rationale=(
        "The same range-edge idea as mr_stoch_confirm but measured on the 1D "
        "regime tier, which lengthens the holding horizon and therefore raises "
        "R per winner -- the single lever that most directly relieves the cost "
        "ratio that killed the production sleeve. Economically, a daily close "
        "at the very bottom of a 10-20 day range is a multi-session "
        "capitulation, and the reverting counterparty is the slower, "
        "mandate-driven buyer (spot accumulators, basis desks) who cannot chase "
        "but will bid a dislocation. Registered specifically to test whether "
        "reversion edge is a function of horizon rather than of oscillator."
    ),
    # The default 96-bar (4-day) time stop would cut a 1D-tier trade off before
    # its thesis had a chance to resolve; 240 bars = 10 days ~ half the 20-day
    # mean the target is measured against. Set from the signal's horizon, not
    # fitted on returns.
    max_hold_bars=240,
)
def mr_wr_daily(ctx, wr_p, lo, k, floor_pct):
    f1d = ctx.frame("1d")
    wr = ctx.align(ta.williams_r(f1d, wr_p), "1d")
    atr = ctx.align(ta.atr(f1d, 14), "1d")
    mid = ctx.align(ta.sma(f1d["close"], 20), "1d")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "1d")

    entry = _blank(ctx)
    entry[fresh & (wr <= lo)] = LONG
    entry[fresh & (wr >= -100.0 - lo)] = SHORT
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        target_dist=np.abs(mid - close),
        note="williams-r-daily",
    )


# ---------------------------------------------------------------------------
# 10. Pullback-in-trend: the one refinement the TRAIN evidence actually asked for
# ---------------------------------------------------------------------------


@register(
    family="meanrev",
    grid={"bb_std": [2.0, 2.5], "k": [1.5, 2.5], "floor_pct": COST_FLOORS},
    rationale=(
        "Added AFTER the first core-symbol TRAIN pass, and the reasoning is "
        "recorded so the added degree of freedom is auditable: of nine "
        "candidates only mr_rsi2_trend was positive, its unfiltered twin "
        "mr_rsi_raw was not, and mr_bb_walkback was positive on BTC/ETH while "
        "every unconfirmed fade lost everywhere. Both surviving signals share "
        "two features and nothing else -- a long-side trend filter and a "
        "requirement that the excursion be demonstrably absorbed before entry. "
        "This strategy is the minimal conjunction of exactly those two, and "
        "nothing else: go long only when a 4H bar closes back INSIDE the lower "
        "Bollinger band while the 1D close is above a RISING 200-day mean. "
        "Economically that is not band fading at all -- it is buying a "
        "liquidation flush inside an established uptrend, where the reverting "
        "counterparty is the trend-follower who was stopped out and must "
        "re-establish. It is registered long-only because the short-side "
        "mirror has no equivalent counterparty in a market with a structural "
        "long bias, and it is deliberately the smallest grid in the module (12 "
        "combos) because it was specified after seeing TRAIN."
    ),
    long_only=True,
)
def mr_pullback_trend(ctx, bb_std, k, floor_pct):
    f4 = ctx.frame("4h")
    f1d = ctx.frame("1d")
    bb = ta.bollinger(f4["close"], 20, bb_std)
    c = f4["close"]
    below = c < bb["lower"]
    inside = (~below) & (c <= bb["upper"]) & bb["lower"].notna()
    reenter = (inside & below.shift(1).fillna(False)).to_numpy()

    mid = ctx.align(bb["mid"], "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    ma = ta.sma(f1d["close"], 200)
    up = ctx.align(((f1d["close"] > ma) & (ma > ma.shift(20))).astype(float), "1d", fill=0.0) > 0.5
    rl = ctx.align(reenter.astype(float), "4h", fill=0.0) > 0.5
    close = ctx.trigger["close"].to_numpy(dtype=float)
    fresh = _fresh_bar(ctx, "4h")

    entry = _blank(ctx)
    entry[fresh & rl & up] = LONG
    return Plan(
        entry=entry,
        stop_dist=_stop(atr, close, k, floor_pct),
        target_dist=np.abs(mid - close),
        note="pullback-in-uptrend",
    )
