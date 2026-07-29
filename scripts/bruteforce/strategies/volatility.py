"""VOLATILITY family: compression-then-expansion, vol-regime conditioning, vol trailing.

WHY THIS FAMILY AT ALL
----------------------
Volatility is what killed the previous system. The PRD's evidence section is
unambiguous: a fixed 0.5%-of-price stop sat inside the noise floor (median 15m
bar range 0.251% BTC / 0.339% ETH / 0.520% SOL) and unconditional adverse
excursion inside the hold window ran 91-95%. The old model treated volatility as
a nuisance to be normalised away. This family treats it as the traded object.

Three distinct hypotheses are separated here on purpose, because they have
different priors and should be judged separately:

1. COMPRESSION -> EXPANSION. Realised volatility is strongly autocorrelated and
   mean-reverting in level, so an unusually quiet window is followed by a
   noisier one more often than chance. That is the best-documented effect in the
   family (TTM squeeze, NR7, inside bars, Bollinger-bandwidth troughs). The
   direction is NOT predicted by the compression; it is taken from which side of
   the coiled range price actually leaves. Entering low-vol and letting the
   breakout choose the side is also the cheapest place to be: the ATR-derived
   stop is small in absolute terms, but the harness's fixed-fractional-risk
   sizing then buys MORE size, so the reward-to-cost arithmetic is unchanged and
   the entry is simply nearer the structure that invalidates it.

2. EXPANSION ITSELF AS THE SIGNAL. ATR percentile crossing up, or short-window
   realised vol overtaking long-window (vol term structure inverting), marks a
   regime handover. Whether the resulting move is worth following in the
   direction of the expansion is exactly the open question the PRD raises about
   the classifier's extreme-volatility bucket.

3. VOL-REGIME CONDITIONING OF A FIXED RULE. ``vol_tercile_break`` and
   ``vol_extreme_gate_probe`` hold the entry rule constant and vary ONLY the
   volatility permission, which is the clean experimental design for answering
   "should extreme-volatility keep suppressing signals?". Comparing two
   different strategies would confound the gate with the rule; comparing one
   rule under three gates does not.

CAUSALITY NOTES SPECIFIC TO THIS FAMILY
---------------------------------------
The classic lookahead in a volatility family is a FULL-SAMPLE quantile: "trade
when vol is in the bottom third of the sample" silently tells bar 100 what vol
did in year three. Every threshold here is a TRAILING ``ta.percentile_rank``
over an explicit window, so a bar's classification uses only bars <= itself and
can never be revised. Coarse-timeframe values reach the 1H grid only through
``ctx.align``. ``vol_opening_range`` needs a per-UTC-day statistic; it uses a
within-day ``cummax``/``cummin`` (never a within-day ``max``), so the level at
any bar depends only on that day's earlier bars.

STOP SIZING AND THE COST FRONTIER
---------------------------------
Stops are ``k * ATR(14)`` of the 4H setup tier unless noted. Measured median
ATR(14)/close on 4H is 1.31% (BTC) / 1.79% (ETH) / 2.50% (SOL), and the frozen
round-trip cost is 0.14%, so ``cost_ratio = 0.0014 / (k * atr_pct)``. Every grid
here starts at ``k >= 1.2``, which puts BTC -- the tightest of the three -- at
``0.0014 / 0.0157 = 0.089``, inside the 0.10 ceiling with a little room. ``k``
values below 1.2 are deliberately absent: they are not cheap trades, they are
trades that fail the cost gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ta
from core import LONG, SHORT, Plan
from registry import register

# --- shared helpers --------------------------------------------------------
# Kept tiny and local. Anything reusable enough to belong in indicators.py
# would mean editing indicators.py, which this family is forbidden to do.

_ATR_P = 14          # ATR period, matching the production/PRD convention
_BB_P = 20           # Bollinger period for squeeze/bandwidth work
_BARS_PER_YEAR_1D = 365
_BARS_PER_YEAR_1H = 8760


def _atr4(ctx) -> np.ndarray:
    """ATR(14) of the 4H setup tier, projected onto the 1H trigger grid."""
    return ctx.align(ta.atr(ctx.frame("4h"), _ATR_P), "4h")


def _flag(ctx, mask: pd.Series, tf: str) -> np.ndarray:
    """Align a boolean coarse-tf Series onto the trigger grid as a bool array.

    Booleans are aligned as floats (``align`` is float-typed) and thresholded at
    0.5. NaN -- meaning "no coarse bar has closed yet" -- becomes False, which is
    the correct reading of "not knowable, so not permitted".
    """
    arr = ctx.align(mask.fillna(False).astype(float), tf, fill=0.0)
    return arr > 0.5


def _fresh(mask: pd.Series) -> pd.Series:
    """First bar of each run of True -- the EVENT, not the state."""
    return mask.fillna(False) & ~mask.fillna(False).shift(1, fill_value=False)


def _recent(mask: pd.Series, window: int) -> pd.Series:
    """True if ``mask`` was True on any of the previous ``window`` bars.

    Shifted by one so the current bar is excluded: "the squeeze WAS on, and is
    now off" needs the two clauses to read different bars.
    """
    prev = mask.fillna(False).astype(float).shift(1, fill_value=0.0)
    return prev.rolling(window, min_periods=1).max().fillna(0.0) > 0.5


# ---------------------------------------------------------------------------
# 1. TTM squeeze release
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"kc_mult": [1.0, 1.5, 2.0], "chan": [10, 20, 40], "k": [1.2, 1.8, 2.5]},
    rationale=(
        "TTM squeeze release. Bollinger bands contracting inside the Keltner "
        "channel is a measured statement that recent close-to-close dispersion "
        "has fallen below recent true-range dispersion -- coiled energy. The "
        "tradeable event is the RELEASE, not the squeeze, and the direction is "
        "not forecast: it is read off whichever side of the trailing 4H channel "
        "price actually leaves within a few bars of the release. Compression "
        "followed by expansion is the best-documented effect in this family and "
        "the one worth spending the most grid on."
    ),
)
def vol_squeeze_release(ctx, kc_mult, chan, k):
    f4 = ctx.frame("4h")
    sq = ta.squeeze_on(f4, _BB_P, 2.0, _BB_P, kc_mult)
    # Off now, on within the last 3 setup bars => a fresh release with a short
    # window in which the breakout still counts as caused by the release.
    released = (~sq.fillna(False)) & _recent(sq, 3)
    ok = _flag(ctx, released, "4h")

    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[ok & (close > hi)] = LONG
    entry[ok & (close < lo)] = SHORT
    return Plan(entry=entry, stop_dist=k * _atr4(ctx), note="squeeze-release")


# ---------------------------------------------------------------------------
# 2. Bollinger-bandwidth trough, then a directional break
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"pr_win": [180, 360, 540], "pr_max": [0.05, 0.15, 0.30], "chan": [10, 20]},
    rationale=(
        "Bollinger bandwidth at a multi-month TRAILING low, then a channel "
        "break. Same compression prior as the squeeze but measured on a "
        "continuous scale rather than a binary band-inside-channel test, so it "
        "can express 'quietest 5% of the last 90 days' instead of just "
        "'quiet'. The percentile windows are 180/360/540 four-hour bars = "
        "30/60/90 days, chosen to bracket the horizon over which crypto vol "
        "regimes actually persist. Trailing percentile, never a sample "
        "quantile: a full-sample vol quantile is the canonical lookahead here."
    ),
)
def vol_bbw_trough_break(ctx, pr_win, pr_max, chan):
    f4 = ctx.frame("4h")
    bbw = ta.bbwidth(f4["close"], _BB_P, 2.0)
    quiet = ta.percentile_rank(bbw, pr_win) <= pr_max
    ok = _flag(ctx, quiet, "4h")

    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[ok & (close > hi)] = LONG
    entry[ok & (close < lo)] = SHORT
    return Plan(entry=entry, stop_dist=1.8 * _atr4(ctx), note="bbw-trough")


# ---------------------------------------------------------------------------
# 3. ATR expansion, entered with the expanding move
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"pr_win": [120, 360], "pr_min": [0.80, 0.90, 0.95], "mom": [3, 6, 12]},
    rationale=(
        "The mirror image of the compression trade, included precisely so the "
        "two can be compared under one cost model: enter WHEN volatility is "
        "already expanding, in the direction the expansion is going. ATR "
        "percentile crossing UP through a trailing threshold is the event; "
        "sign of the last `mom` setup-bar return supplies the side. pr_min "
        "0.90 with a 180-ish window is deliberately the production "
        "classifier's own extreme-volatility definition, so a positive result "
        "here is direct evidence that suppressing that bucket discards edge."
    ),
)
def vol_atr_expansion(ctx, pr_win, pr_min, mom):
    f4 = ctx.frame("4h")
    pr = ta.percentile_rank(ta.atr_pct(f4, _ATR_P), pr_win)
    # Cross UP through the threshold: state alone would fire for a whole regime.
    cross = (pr >= pr_min) & (pr.shift(1) < pr_min)
    ok = _flag(ctx, cross, "4h")

    ret = f4["close"] - f4["close"].shift(mom)
    up = _flag(ctx, ret > 0, "4h")
    down = _flag(ctx, ret < 0, "4h")

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[ok & up] = LONG
    entry[ok & down] = SHORT
    return Plan(entry=entry, stop_dist=1.8 * _atr4(ctx), note="atr-expansion")


# ---------------------------------------------------------------------------
# 4. THE EXPERIMENT: one breakout rule, three volatility permissions
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"band": ["low", "mid", "high"], "chan": [20, 55], "k": [1.2, 2.0]},
    rationale=(
        "A controlled experiment, not a strategy idea. One fixed Donchian "
        "breakout is run under three mutually exclusive volatility "
        "permissions -- bottom, middle and top tercile of the 1D realised-vol "
        "trailing percentile -- with everything else identical. Because the "
        "entry rule is held constant, any Sharpe difference between the three "
        "bands is attributable to the volatility gate alone, which is what "
        "makes this the right instrument for the PRD's open question about the "
        "extreme-volatility regime suppressing all signals."
    ),
)
def vol_tercile_break(ctx, band, chan, k):
    f1d = ctx.frame("1d")
    rv = ta.realized_vol(f1d["close"], 20, _BARS_PER_YEAR_1D)
    prv = ta.percentile_rank(rv, 252)  # ~1 year of daily bars, trailing
    if band == "low":
        permit = prv <= 1.0 / 3.0
    elif band == "mid":
        permit = (prv > 1.0 / 3.0) & (prv <= 2.0 / 3.0)
    else:
        permit = prv > 2.0 / 3.0
    ok = _flag(ctx, permit, "1d")

    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[ok & (close > hi)] = LONG
    entry[ok & (close < lo)] = SHORT
    return Plan(entry=entry, stop_dist=k * _atr4(ctx), note=f"vol-{band}")


@register(
    family="volatility",
    grid={"gate": ["none", "suppress", "only"], "thresh": [0.85, 0.90]},
    rationale=(
        "The suppression question asked against the PRODUCTION rule rather "
        "than a generic breakout: 20-bar 4H Donchian with the 55-mid filter "
        "and 1D ADX>25, at the frozen 1.5*ATR stop, under three settings of "
        "the classifier's own extreme-volatility gate -- absent, suppressing "
        "(today's behaviour), or inverted so ONLY the extreme bucket trades. "
        "The gate reproduces classifier.py exactly: relative-ATR percentile "
        "over a 180-bar trailing window on the 1D regime tier. 'none' should "
        "reproduce baseline.donchian_production up to the missing gate, which "
        "also makes this a sanity check on the comparison."
    ),
)
def vol_extreme_gate_probe(ctx, gate, thresh):
    f1d = ctx.frame("1d")
    # classifier.py: percentile rank of ATR/close over a 180-bar trailing window.
    atr_pctl = ta.percentile_rank(ta.atr_pct(f1d, _ATR_P), 180)
    extreme = atr_pctl >= thresh
    if gate == "none":
        permit = pd.Series(True, index=f1d.index) & atr_pctl.notna()
    elif gate == "suppress":
        permit = ~extreme & atr_pctl.notna()
    else:
        permit = extreme
    ok = _flag(ctx, permit, "1d")

    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, 20), "4h")
    lo = ctx.align(ta.rolling_low(f4, 20), "4h")
    mid = ctx.align(
        (ta.rolling_high(f4, 55, exclude_current=False)
         + ta.rolling_low(f4, 55, exclude_current=False)) / 2.0,
        "4h",
    )
    adx = ctx.align(ta.adx(f1d, 14), "1d")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    trending = adx > 25.0
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[ok & trending & (close > hi) & (close > mid)] = LONG
    entry[ok & trending & (close < lo) & (close < mid)] = SHORT
    return Plan(entry=entry, stop_dist=1.5 * _atr4(ctx), note=f"gate-{gate}")


# ---------------------------------------------------------------------------
# 5. Narrowest-range-in-N (NR7 family) break of the coiled bar
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"nr": [7, 14, 28], "w": [1, 3], "k": [1.2, 2.0]},
    rationale=(
        "NR7 adapted to the 4H setup tier: the narrowest true range in the last "
        "N bars marks a bar whose participants agreed on price, and the break "
        "of THAT bar's extreme -- not a long channel -- is the first evidence "
        "of disagreement. The distinguishing feature versus the squeeze is the "
        "reference level: an NR bar's own high/low is a much closer, much more "
        "precisely dated level than a 20-bar channel, so the break is timed "
        "rather than merely permitted. `w` bounds how many bars the level stays "
        "live, so a stale coil from a day ago cannot trigger today."
    ),
)
def vol_nr_break(ctx, nr, w, k):
    f4 = ctx.frame("4h")
    rng = f4["high"] - f4["low"]
    narrow = rng <= rng.rolling(nr, min_periods=nr).min()
    # Shift by one so the trigger bar is never the NR bar itself, then let the
    # level survive at most `w` further bars. ffill(limit=) is causal.
    hi_lvl = f4["high"].where(narrow).shift(1).ffill(limit=w)
    lo_lvl = f4["low"].where(narrow).shift(1).ffill(limit=w)
    hi = ctx.align(hi_lvl, "4h")
    lo = ctx.align(lo_lvl, "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[np.isfinite(hi) & (close > hi)] = LONG
    entry[np.isfinite(lo) & (close < lo)] = SHORT
    return Plan(entry=entry, stop_dist=k * _atr4(ctx), note="nr-break")


# ---------------------------------------------------------------------------
# 6. Inside-bar compression stack
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"nconsec": [1, 2, 3], "k": [1.2, 1.8, 2.5]},
    rationale=(
        "Consecutive inside bars are the purest available statement of "
        "contracting range: each bar's entire high-low is contained by its "
        "predecessor's, so the mother bar's extremes bound every subsequent "
        "print. Breaking the mother bar's high or low is therefore both the "
        "expansion event and a structurally defined level. Stacking `nconsec` "
        "inside bars trades frequency for compression quality, which is the "
        "one axis worth sweeping."
    ),
)
def vol_inside_bar_break(ctx, nconsec, k):
    f4 = ctx.frame("4h")
    ib = ta.inside_bar(f4).fillna(False).astype(float)
    coiled = ib.rolling(nconsec, min_periods=nconsec).min() > 0.5
    # Mother bar = the bar before the run of inside bars, so its extremes are
    # the max/min over the run plus one.
    hi_lvl = f4["high"].rolling(nconsec + 1, min_periods=nconsec + 1).max()
    lo_lvl = f4["low"].rolling(nconsec + 1, min_periods=nconsec + 1).min()
    ok = _flag(ctx, coiled, "4h")
    hi = ctx.align(hi_lvl, "4h")
    lo = ctx.align(lo_lvl, "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[ok & (close > hi)] = LONG
    entry[ok & (close < lo)] = SHORT
    return Plan(entry=entry, stop_dist=k * _atr4(ctx), note="inside-bar")


# ---------------------------------------------------------------------------
# 7. Vol term structure: short-window realised vol vs long-window
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"short": [12, 24, 48], "ratio_min": [1.2, 1.5, 2.0], "k": [1.2, 2.0]},
    rationale=(
        "Volatility term structure inverting. Short-window realised vol rising "
        "above a multiple of its own long-window value (fixed at 8x the short "
        "window, so only one lookback is fitted) says the current regime is "
        "noisier than the regime it sits inside -- a handover, not a wiggle. "
        "This is the vol-of-vol reading of the same expansion event that "
        "vol_atr_expansion reads through ATR percentile; running both and "
        "comparing tells us whether the effect is robust to the estimator or "
        "an artefact of one. Computed natively on the 1H trigger tier, so no "
        "alignment is involved and no coarse-bar staleness can help it."
    ),
)
def vol_term_structure(ctx, short, ratio_min, k):
    close_s = ctx.trigger["close"]
    rv_s = ta.realized_vol(close_s, short, _BARS_PER_YEAR_1H)
    rv_l = ta.realized_vol(close_s, short * 8, _BARS_PER_YEAR_1H)
    ratio = rv_s / rv_l.replace(0.0, np.nan)
    cross = ((ratio >= ratio_min) & (ratio.shift(1) < ratio_min)).to_numpy()

    ret = (close_s - close_s.shift(short)).to_numpy(dtype=float)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[cross & (ret > 0)] = LONG
    entry[cross & (ret < 0)] = SHORT
    return Plan(entry=entry, stop_dist=k * _atr4(ctx), note="vol-term")


# ---------------------------------------------------------------------------
# 8. Chandelier-exit trend riding
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"per": [20, 40], "mult": [2.0, 3.0, 4.0], "k": [1.2, 2.0]},
    rationale=(
        "The exit side of the family. A breakout is entered conventionally, but "
        "the trade is held by a ratcheting ATR chandelier rather than a fixed "
        "target, so the hold length adapts to volatility instead of being cut "
        "by a time stop at an arbitrary bar count. If volatility is the thing "
        "that killed the previous system, letting volatility set the exit "
        "distance is the most direct available remedy -- and it is the one "
        "component that could rescue an otherwise mediocre entry, which is why "
        "the entry here is deliberately plain."
    ),
)
def vol_chandelier_ride(ctx, per, mult, k):
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, per), "4h")
    lo = ctx.align(ta.rolling_low(f4, per), "4h")
    atr4 = _atr4(ctx)
    close = ctx.trigger["close"].to_numpy(dtype=float)

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[close > hi] = LONG
    entry[close < lo] = SHORT
    return Plan(
        entry=entry,
        stop_dist=k * atr4,
        trail_atr=mult * atr4,
        note="chandelier-ride",
    )


# ---------------------------------------------------------------------------
# 9. Opening-range expansion, adapted to 24h crypto
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"or_len": [4, 6, 8], "min_range": [0.0, 0.005], "k": [1.2, 2.0]},
    rationale=(
        "Opening-range breakout, transplanted to a market with no open. The "
        "window is anchored at 00:00 UTC because that boundary is not "
        "arbitrary in crypto perps: it is the daily-bar boundary the whole "
        "1D regime tier is built on, and it brackets the 00:00 UTC funding "
        "settlement, which is the one recurring time-of-day event with real "
        "flow behind it. The first `or_len` hours define the day's range; a "
        "break of it later in the same day is the expansion trade. "
        "`min_range` optionally requires the opening range to be non-trivial, "
        "so a dead 4 hours cannot manufacture a level a single tick away. "
        "The level uses a within-day cummax/cummin, never a within-day max, so "
        "it is knowable at every bar that uses it."
    ),
)
def vol_opening_range(ctx, or_len, min_range, k):
    trig = ctx.trigger
    ts = trig.index.to_numpy()
    day = ts // 86_400_000
    hour = (ts % 86_400_000) // 3_600_000

    in_or = pd.Series(hour < or_len, index=trig.index)
    # cummax/cummin within the day: at any bar, only that day's EARLIER opening
    # bars contribute. A groupby().max() would be a within-day lookahead.
    day_s = pd.Series(day, index=trig.index)
    or_hi = trig["high"].where(in_or).groupby(day_s).cummax().groupby(day_s).ffill()
    or_lo = trig["low"].where(in_or).groupby(day_s).cummin().groupby(day_s).ffill()

    hi = or_hi.to_numpy(dtype=float)
    lo = or_lo.to_numpy(dtype=float)
    close = trig["close"].to_numpy(dtype=float)
    wide = (hi - lo) >= min_range * close
    after = (hour >= or_len) & np.isfinite(hi) & np.isfinite(lo) & wide

    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[after & (close > hi)] = LONG
    entry[after & (close < lo)] = SHORT
    return Plan(entry=entry, stop_dist=k * _atr4(ctx), note="opening-range")


# ---------------------------------------------------------------------------
# 10. Squeeze release, but only from a genuinely low vol REGIME
# ---------------------------------------------------------------------------


@register(
    family="volatility",
    grid={"pr_max": [0.20, 0.40, 0.60], "chan": [10, 20], "k": [1.2, 2.0]},
    rationale=(
        "Two-tier compression: a 1D realised-vol trailing percentile below "
        "`pr_max` establishes that the whole REGIME is quiet, and a 4H "
        "Bollinger-bandwidth trough establishes that the setup tier is quiet "
        "too. The hypothesis being separated from vol_bbw_trough_break is "
        "whether local compression matters more when the macro regime agrees; "
        "if the two-tier version is no better, the extra condition is just "
        "sample reduction and should be dropped rather than kept for the story."
    ),
)
def vol_two_tier_compression(ctx, pr_max, chan, k):
    f1d = ctx.frame("1d")
    prv = ta.percentile_rank(
        ta.realized_vol(f1d["close"], 20, _BARS_PER_YEAR_1D), 252
    )
    regime_quiet = _flag(ctx, prv <= pr_max, "1d")

    f4 = ctx.frame("4h")
    bbw = ta.bbwidth(f4["close"], _BB_P, 2.0)
    setup_quiet = _flag(ctx, ta.percentile_rank(bbw, 360) <= 0.30, "4h")

    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    ok = regime_quiet & setup_quiet
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[ok & (close > hi)] = LONG
    entry[ok & (close < lo)] = SHORT
    return Plan(entry=entry, stop_dist=k * _atr4(ctx), note="two-tier-compression")
