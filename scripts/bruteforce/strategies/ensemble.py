"""ENSEMBLE family: the entry-vs-exit factorial, and composites of what survived.

This module has two halves with two different jobs.

PART 1 -- ``fac_*``: THE ENTRY-VS-EXIT FACTORIAL
------------------------------------------------
Three families independently produced the same shape of result:

- trend: every Donchian/N-bar-extreme entry loses on the 20-symbol SELECT split
  (-0.87 to -1.14 mean Sharpe), *including* the frozen production baseline
  ``donchian_production``. Five structurally different implementations of the
  same entry failed identically, which indicts the shared entry, not any
  variant's parameters.
- trend: every strategy positive on both splits has a ratcheting ATR trail
  (``Plan.trail_atr``), and no failing Donchian variant has one.
  ``trend_turtle`` survives better with the trail than without it.
- volatility: ``vol_chandelier_ride`` -- a deliberately plain 20-bar 4H breakout
  plus a ratcheting chandelier exit -- is that family's only both-splits
  survivor, and its author attributed the edge to the exit rather than the entry.

Those are three observational claims, all confounded: the trailed strategies also
had different entries. ``fac_entry_exit`` is the controlled test. It crosses

    entry in {N-bar extreme (Donchian), ATR channel (Keltner)}
      x  exit in {ratcheting ATR trail, fixed ATR target, return-to-mean}

with everything else held constant: stop frozen at ``K_STOP * ATR(14)`` of the 4H
setup tier, 1H trigger, the harness default 96-bar time stop, both directions
allowed, identical lookback axis, identical ATR-multiple axis. 24 combos, which
is a cheap decisive experiment rather than another search.

Two design choices that make the factorial readable:

- **The Keltner multiple is FROZEN at 2.0, not swept.** In a factorial the two
  entry arms must be equally tuned or the comparison measures search effort
  rather than structure. 2.0 is the textbook value and the centre of
  ``trend_keltner_trail``'s grid.
- **The no-trail arm is split in two.** "No trail" is not one thing: a fixed
  target caps the winner, whereas a return-to-mean exit does not. Testing both
  separates "the ratchet matters" from "capping winners is what hurts". Both
  no-trail arms keep the identical fixed ``1.5 * ATR`` stop.

``geom`` is the shared ATR-multiple axis and means the same kind of thing in
every arm: the trail ratchets ``geom`` ATR behind the extreme, the target sits
``geom`` ATR away, and the return-to-mean band is ``geom/2`` ATR around the EMA
(so the loosest exit in one arm is the loosest in all three). It is exit geometry
only and never touches ``median_risk_pct``, so the cost ratio is a property of
the market here exactly as it is in ``trend.py``.

PART 2 -- ``ens_*``: COMPOSITES OF WHAT ACTUALLY WORKED
------------------------------------------------------
Built only on findings that replicated across two splits on the 20-symbol
universe, and each one is registered to answer a specific question:

- ``ens_lowvol_keltner_trail`` -- the volatility family found the Sharpe gradient
  runs monotonically across the WHOLE realised-vol distribution (low +0.34 /
  mid +0.12 / high -0.28 on TRAIN), not just at the top decile the production
  classifier vetoes. This applies a low-vol *preference* to the search's single
  best rule.
- ``ens_vol_gate_ladder`` -- the direct three-arm comparison of no gate vs the
  production-style extreme veto vs a low-vol preference, on one fixed rule.
- ``ens_regime_dispatch`` -- trend engine in quiet tape, momentum engine in
  strongly trending tape, flat otherwise.
- ``ens_vote2_kelt_mom`` / ``ens_vote3_majority`` -- vote-combining survivors
  from different families (channel break, 1D momentum acceleration, break of
  structure), which are as close to uncorrelated as this search produced.
- ``ens_mom_accel_trail`` -- the trail overlay applied to a *momentum* entry
  (``mom_accel``, the momentum family's #1). If Part 1 says the trail is the
  active ingredient, this is where it should generalise.
- ``ens_bos_trail`` -- the same overlay on ``st_bos``, the structure family's
  only both-splits survivor and the best cost headroom in the search.

DELIBERATELY NOT BUILT: boundary mean reversion. The structure family showed it
fails on *payoff shape* (win rate 0.47-0.59 with profit factor 0.69-0.76 -- wins
small, loses the whole range width), which no stop or filter fixes.
Candlestick confirmation filters are also absent: measured across 20 symbols they
carried no value, so adding one would only spend degrees of freedom.

CAUSALITY
---------
Coarse data enters only via ``ctx.align``. Every helper reads the current and
strictly earlier bars. Volatility percentiles use ``ta.percentile_rank`` over an
explicit trailing window -- a full-sample vol quantile is the classic lookahead
in this family and is absent by construction. Pivot levels are confirmed ``span``
bars late by ``ta.pivot_high``/``pivot_low`` and forward-filled only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ta
from core import LONG, SHORT, Plan
from registry import register

# Frozen risk unit for every 1H-trigger strategy here, identical to trend.py so
# results are comparable across modules. NOT a grid axis anywhere.
K_STOP = 1.5
ATR_LEN = 14

# Frozen Keltner width for the factorial: see the module docstring on equal
# tuning effort between the two entry arms.
FAC_KC_MULT = 2.0

# Momentum tier, matching momentum.py so ens_mom_accel_trail is comparable to
# the mom_accel result it is built on.
_MOM_TRIG = "4h"
_MOM_TFS = ("1d", "4h", "1h")
_MOM_HOLD = 180

# The production regime classifier's own window: relative-ATR percentile over a
# 180-bar trailing window on the 1D tier (classifier.py). Reused rather than
# re-chosen so the "is the binary veto the right shape?" question is asked
# against the real thing.
VOL_PR_WIN = 180


# ---------------------------------------------------------------------------
# Causal helpers
# ---------------------------------------------------------------------------


def _setup_atr(ctx) -> np.ndarray:
    """ATR(14) of the 4H setup tier, on the trigger grid. The risk unit."""
    return ctx.align(ta.atr(ctx.frame("4h"), period=ATR_LEN), "4h")


def _regime_adx(ctx) -> np.ndarray:
    """Wilder ADX(14) on the 1D regime tier."""
    return ctx.align(ta.adx(ctx.frame("1d"), 14), "1d")


def _vol_pr(ctx, win: int = VOL_PR_WIN) -> np.ndarray:
    """Trailing percentile rank of 1D realised volatility, on the trigger grid.

    0 = quietest in the trailing window, 1 = most violent. Trailing by
    construction: ``percentile_rank`` compares the current value against the
    previous ``win`` observations only, so no future volatility informs the
    reading. NaN during warmup, which suppresses gated entries rather than
    silently admitting them.
    """
    f1d = ctx.frame("1d")
    rv = ta.realized_vol(f1d["close"], 20, bars_per_year=365)
    return ctx.align(ta.percentile_rank(rv, win), "1d")


def _keltner_break(ctx, period: int, mult: float) -> np.ndarray:
    """+1 above the upper ATR channel, -1 below the lower, else 0."""
    kc = ta.keltner(ctx.frame("4h"), period=period, atr_period=ATR_LEN, mult=mult)
    up = ctx.align(kc["upper"], "4h")
    dn = ctx.align(kc["lower"], "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    sig = np.zeros(ctx.n, dtype=np.int8)
    sig[close > up] = LONG
    sig[close < dn] = SHORT
    return sig


def _donchian_break(ctx, chan: int) -> np.ndarray:
    """+1 above the N-bar 4H high, -1 below the N-bar low, else 0."""
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, chan), "4h")
    lo = ctx.align(ta.rolling_low(f4, chan), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    sig = np.zeros(ctx.n, dtype=np.int8)
    sig[close > hi] = LONG
    sig[close < lo] = SHORT
    return sig


def _accel_state(ctx, lookback: int, gap: int) -> np.ndarray:
    """``mom_accel``'s state: 1D momentum and its change agree in sign.

    Reproduces ``momentum.mom_accel`` exactly (the momentum family's #1: TRAIN
    +0.712 / SELECT +0.430 on the 20-symbol universe) so the composite inherits
    a measured component rather than a new one.
    """
    f1d = ctx.frame("1d")
    mom = ta.momentum(f1d["close"], lookback)
    dmom = mom - mom.shift(gap)
    m = ctx.align(mom, "1d")
    d = ctx.align(dmom, "1d")
    sig = np.zeros(ctx.n, dtype=np.int8)
    sig[(m > 0.0) & (d > 0.0)] = LONG
    sig[(m < 0.0) & (d < 0.0)] = SHORT
    return sig


def _prev_pivot(sparse: pd.Series) -> pd.Series:
    """The pivot BEFORE the latest confirmed one, forward-filled.

    Same helper as ``structure.py``: ``sparse.shift(1)`` would shift by one BAR
    on a forward-filled series and yield "same pivot" almost everywhere.
    """
    vals = sparse.dropna()
    return vals.shift(1).reindex(sparse.index).ffill()


def _bos_state(ctx, span: int) -> tuple[np.ndarray, np.ndarray]:
    """``st_bos`` direction plus the structural distance behind it.

    Returns ``(signal, struct_dist)``. Reproduces ``structure.st_bos`` at
    ``require_hl=False`` -- the universe-optimal arm, and the arm whose author
    concluded the level break rather than the swing sequence does the work.
    ``struct_dist`` is entry-to-defining-swing distance in price units, NaN
    where there is no signal.
    """
    f4 = ctx.frame("4h")
    ph, pl = ta.pivot_high(f4, span), ta.pivot_low(f4, span)
    res = ctx.align(ph.ffill(), "4h")
    sup = ctx.align(pl.ffill(), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)

    up = close > res
    dn = close < sup
    sig = np.zeros(ctx.n, dtype=np.int8)
    sig[up] = LONG
    sig[dn] = SHORT
    struct = np.where(up, close - sup, np.where(dn, res - close, np.nan))
    return sig, struct


def _mean_back(ctx, period: int, band_atr: float) -> np.ndarray:
    """True when price has returned within ``band_atr`` ATR of its 4H EMA.

    This is the no-trail arm's *uncapped* exit. It is direction-agnostic on
    purpose, which matters: ``Plan.exit_signal`` carries no direction by
    contract, so an "exit longs on the N-bar low" rule cannot coexist with its
    short-side mirror in one Plan (the bug that forced ``trend_turtle``
    long-only). "Price is back at its mean" is symmetric and expresses the same
    idea -- the displacement that authorised the trade has been given back --
    without excluding the short leg from the factorial.
    """
    ema4 = ctx.align(ta.ema(ctx.frame("4h")["close"], period), "4h")
    close = ctx.trigger["close"].to_numpy(dtype=float)
    atr = _setup_atr(ctx)
    return np.abs(close - ema4) <= band_atr * atr


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


# ---------------------------------------------------------------------------
# PART 1: the 2x2 (really 2x3) entry-vs-exit factorial
# ---------------------------------------------------------------------------


@register(
    family="ensemble",
    grid={
        "entry": ["donchian", "keltner"],
        "exit": ["trail", "target", "meanback"],
        "look": [20, 55],
        "geom": [2.0, 3.0],
    },
    rationale=(
        "THE DISCRIMINATING EXPERIMENT for the search's biggest open question: "
        "is the edge in the entry or in the exit? Every strategy positive on "
        "both splits across all six families has a ratcheting ATR trail, and no "
        "failing Donchian variant has one -- but those observations are "
        "confounded, because the trailed survivors also had different entries. "
        "This crosses the two entries (N-bar extreme vs volatility-normalised "
        "ATR channel) against three exits (ratcheting trail, fixed ATR target, "
        "return-to-mean) with the stop frozen at 1.5*ATR(4H), one trigger tier, "
        "one time stop and one shared ATR-multiple axis, so the marginal effect "
        "of each factor is readable off the four cells. 24 combos: a cheap "
        "decisive test, not another search. If the trail carries the edge, the "
        "PRD has spent two iterations tuning the wrong component."
    ),
)
def fac_entry_exit(ctx, entry, exit, look, geom):
    atr = _setup_atr(ctx)
    sig = (
        _donchian_break(ctx, look)
        if entry == "donchian"
        else _keltner_break(ctx, look, FAC_KC_MULT)
    )

    trail = target = None
    exit_sig = None
    if exit == "trail":
        trail = geom * atr
    elif exit == "target":
        target = geom * atr
    else:  # meanback
        exit_sig = _mean_back(ctx, look, geom / 2.0)

    return Plan(
        entry=sig,
        stop_dist=K_STOP * atr,
        target_dist=target,
        exit_signal=exit_sig,
        trail_atr=trail,
        note=f"fac-{entry}-{exit}{geom}-{look}",
    )


# ---------------------------------------------------------------------------
# PART 2: composites
# ---------------------------------------------------------------------------


@register(
    family="ensemble",
    grid={"mult": [2.0, 2.5], "trail_k": [2.0, 3.0], "pr_max": [0.4, 0.6, 0.8]},
    rationale=(
        "The search's best single rule (Keltner channel + ratcheting ATR trail, "
        "the only trend strategy positive on both splits universe-wide) with a "
        "low-volatility PREFERENCE rather than the production classifier's "
        "binary 90th-percentile extreme veto. The volatility family measured a "
        "monotone Sharpe gradient across the whole realised-vol distribution "
        "(low +0.34 / mid +0.12 / high -0.28 on TRAIN, and steeper on SELECT), "
        "not a cliff at the top decile, so a ceiling well below 0.9 should "
        "capture strictly more of the effect than the veto does. pr_max sweeps "
        "how much of the distribution to keep."
    ),
)
def ens_lowvol_keltner_trail(ctx, mult, trail_k, pr_max):
    atr = _setup_atr(ctx)
    sig = _keltner_break(ctx, 20, mult)
    quiet = _vol_pr(ctx) <= pr_max
    sig = np.where(quiet, sig, 0).astype(np.int8)
    return Plan(
        entry=sig,
        stop_dist=K_STOP * atr,
        trail_atr=trail_k * atr,
        note=f"lowvol{pr_max}-kelt{mult}-tr{trail_k}",
    )


@register(
    family="ensemble",
    grid={"gate": ["none", "veto", "lowpref"], "trail_k": [2.0, 3.0]},
    rationale=(
        "The controlled three-arm test of the volatility-gate SHAPE, on one "
        "fixed rule (Keltner 20 x 2.0 + ATR trail) so the gate is the only thing "
        "varying. 'none' is the unfiltered control, 'veto' reproduces the "
        "production classifier's binary suppression above the 90th percentile of "
        "trailing 1D relative volatility, and 'lowpref' keeps only the quieter "
        "half of the distribution. The volatility family recommended this change "
        "but tested it on its own Donchian-based instrument; this asks the same "
        "question of the rule that actually survived, which is the one that "
        "would ship."
    ),
)
def ens_vol_gate_ladder(ctx, gate, trail_k):
    atr = _setup_atr(ctx)
    sig = _keltner_break(ctx, 20, 2.0)
    pr = _vol_pr(ctx)
    if gate == "veto":
        ok = ~(pr > 0.90)          # NaN warmup is not "extreme": matches the
    elif gate == "lowpref":        # classifier, which cannot label without data
        ok = pr <= 0.50
    else:
        ok = np.ones(ctx.n, dtype=bool)
    sig = np.where(ok, sig, 0).astype(np.int8)
    return Plan(
        entry=sig, stop_dist=K_STOP * atr, trail_atr=trail_k * atr,
        note=f"gate-{gate}-tr{trail_k}",
    )


@register(
    family="ensemble",
    grid={"adx_min": [20, 25], "pr_max": [0.5, 0.7], "trail_k": [2.0, 3.0]},
    rationale=(
        "Regime dispatch: the two engines that survived their own families work "
        "in different tape, so let the 1D regime choose between them instead of "
        "running one everywhere. In strongly trending tape (1D ADX > adx_min) "
        "trade 1D momentum acceleration (mom_accel, the momentum family's only "
        "both-splits survivor); otherwise, if the tape is quiet (vol percentile "
        "<= pr_max), trade the Keltner channel break; otherwise stay flat. Both "
        "legs share the ratcheting ATR trail, so the dispatch is the only thing "
        "being tested. Momentum takes precedence because its evidence is "
        "state-based and slower-moving, and because a strong ADX day is exactly "
        "where a channel break is most likely to be a late entry."
    ),
)
def ens_regime_dispatch(ctx, adx_min, pr_max, trail_k):
    atr = _setup_atr(ctx)
    adx = _regime_adx(ctx)
    pr = _vol_pr(ctx)

    mom_leg = _accel_state(ctx, 30, 20)
    kelt_leg = _keltner_break(ctx, 20, 2.0)

    trending = adx > adx_min
    quiet = pr <= pr_max
    sig = np.where(trending, mom_leg, np.where(quiet, kelt_leg, 0)).astype(np.int8)
    return Plan(
        entry=sig, stop_dist=K_STOP * atr, trail_atr=trail_k * atr,
        note=f"dispatch-adx{adx_min}-pr{pr_max}",
    )


@register(
    family="ensemble",
    grid={"mult": [2.0, 2.5], "lookback": [30, 60], "trail_k": [2.0, 3.0]},
    rationale=(
        "Two-vote agreement between the two least-related survivors in the "
        "search: a 4H volatility-normalised channel break (a fast, "
        "price-location statement) and 1D momentum acceleration (a slow, "
        "second-derivative statement). Requiring both to point the same way "
        "should raise per-trade quality if their errors are genuinely "
        "independent, and should merely cut trade count if they are not -- which "
        "is the diagnostic. The trend family already found that 1D-EMA "
        "confirmation cut trades in half without improving quality, so this is "
        "the same test with a confirmer that has its own measured edge rather "
        "than a redundant one."
    ),
)
def ens_vote2_kelt_mom(ctx, mult, lookback, trail_k):
    atr = _setup_atr(ctx)
    kelt = _keltner_break(ctx, 20, mult)
    mom = _accel_state(ctx, lookback, 20)
    agree = (kelt != 0) & (kelt == mom)
    sig = np.where(agree, kelt, 0).astype(np.int8)
    return Plan(
        entry=sig, stop_dist=K_STOP * atr, trail_atr=trail_k * atr,
        note=f"vote2-kelt{mult}-mom{lookback}",
    )


@register(
    family="ensemble",
    grid={"vote_min": [2, 3], "span": [3, 6], "trail_k": [2.0, 3.0]},
    rationale=(
        "Three-signal vote across three different families' survivors: Keltner "
        "channel break (trend), 1D momentum acceleration (momentum), and break "
        "of structure (structure -- the only structure strategy positive on both "
        "splits, and the best cost headroom in the search at c = 0.014-0.028). "
        "vote_min=2 is a majority rule and vote_min=3 is unanimity, so the axis "
        "measures directly whether agreement buys quality or just starves the "
        "sample. Voting on DIRECTION rather than averaging signals is "
        "deliberate: the three components disagree about position sizing "
        "entirely, but they all emit a sign."
    ),
)
def ens_vote3_majority(ctx, vote_min, span, trail_k):
    atr = _setup_atr(ctx)
    kelt = _keltner_break(ctx, 20, 2.0).astype(np.int16)
    mom = _accel_state(ctx, 30, 20).astype(np.int16)
    bos, _ = _bos_state(ctx, span)
    votes = kelt + mom + bos.astype(np.int16)
    sig = np.zeros(ctx.n, dtype=np.int8)
    sig[votes >= vote_min] = LONG
    sig[votes <= -vote_min] = SHORT
    return Plan(
        entry=sig, stop_dist=K_STOP * atr, trail_atr=trail_k * atr,
        note=f"vote3-min{vote_min}-span{span}",
    )


@register(
    family="ensemble",
    trigger_tf=_MOM_TRIG,
    timeframes=_MOM_TFS,
    max_hold_bars=_MOM_HOLD,
    grid={"lookback": [30, 60], "gap": [10, 20], "trail_k": [2.0, 3.0]},
    rationale=(
        "The trail overlay applied to a MOMENTUM entry. mom_accel was the "
        "momentum family's #1 (TRAIN +0.712 / SELECT +0.430 universe-wide, "
        "positive on all three core coins on both splits) and it runs with no "
        "trail at all -- its exits are the stop and the 30-day time stop. If the "
        "factorial says the ratchet is the active ingredient, the effect must "
        "generalise beyond channel entries, and this is the cleanest place to "
        "check: identical tier (1D signal / 4H trigger), identical 1.5*ATR(1D) "
        "stop, identical 180-bar hold, one array added."
    ),
)
def ens_mom_accel_trail(ctx, lookback, gap, trail_k):
    atr_1d = ctx.align(ta.atr(ctx.frame("1d"), ATR_LEN), "1d")
    sig = _accel_state(ctx, lookback, gap)
    return Plan(
        entry=sig,
        stop_dist=K_STOP * atr_1d,
        trail_atr=trail_k * atr_1d,
        note=f"accel{lookback}/{gap}-tr{trail_k}",
    )


@register(
    family="ensemble",
    grid={"span": [3, 6], "atr_floor": [1.5, 2.0], "trail_k": [2.0, 3.0]},
    rationale=(
        "The same trail overlay on st_bos -- break of structure with a "
        "structural stop floored at atr_floor*ATR(4H). st_bos was the structure "
        "family's only both-splits survivor and has the best cost headroom in "
        "the whole search (c = 0.014-0.048, two to seven times inside the "
        "ceiling), which means it has the most room to absorb worse-than-modelled "
        "slippage. As registered it exits on a fixed R:R target; replacing that "
        "target with a ratchet tests the factorial's conclusion on a "
        "structurally different entry AND on the family with the widest risk "
        "unit, where a trail has the most distance to work with."
    ),
)
def ens_bos_trail(ctx, span, atr_floor, trail_k):
    atr = _setup_atr(ctx)
    sig, struct = _bos_state(ctx, span)
    # Structural distance, floored at atr_floor*ATR: the floor is what stops a
    # coincidentally-nearby swing manufacturing a noise-floor stop, and the
    # structure family measured every cost-ratio breach at floor=1.0.
    padded = np.where(np.isfinite(struct), struct, 0.0)
    stop = np.maximum(padded, atr_floor * atr)
    return Plan(
        entry=sig,
        stop_dist=stop,
        trail_atr=trail_k * atr,
        note=f"bos{span}-floor{atr_floor}-tr{trail_k}",
    )
