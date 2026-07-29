"""MOMENTUM family: time-series momentum, cross-sectional relative strength.

WHY THIS FAMILY EXISTS
----------------------
The PRD names universe expansion (Option B) as the pre-committed escalation
path, on the strength of published net-of-fee Sharpe >1.5 for a daily-bar,
wide-universe momentum/trend ensemble. That published edge is driven largely by
CROSS-SECTIONAL breadth -- ranking 20 assets against each other -- which the
3-symbol production universe cannot express at all. This module is the honest
test of that thesis: plain time-series momentum first (the most replicated
effect in the asset-pricing literature), then the cross-sectional variants that
only exist because 20 symbols are now backfilled.

EXECUTION TIER
--------------
Every strategy here signals off 1D bars and executes on the 4H trigger grid.
Rationale: momentum is a multi-week effect, and the harness measured mean Sharpe
+0.791 at 60m execution vs -12.71 at 1m, so the cost frontier rewards coarse
triggers. Stops are ``k * ATR(14, 1D)``, which puts median risk_pct in the
3-7% band -- an order of magnitude clear of the ``cost_ratio <= 0.10`` ceiling
that the old fixed-0.5% stop model failed.

PEER-DATA CAUSALITY (read this before editing)
----------------------------------------------
``core.assert_causal`` truncates only the CURRENT symbol's frames. A peer frame
loaded via ``core.load_frame`` is therefore the FULL history in both the full
and truncated runs, so the audit cannot catch a peer-data leak. Peer safety here
rests on two structural properties instead, not on the audit:

1. **Close-time alignment.** ``_align_peer`` maps each of MY trigger bars to the
   last peer bar whose CLOSE time is ``<= my bar's close time``, using exactly
   ``np.searchsorted(peer_close_times, my_close_times, side="right") - 1`` --
   the same rule as ``Ctx.align`` and the production engine. A peer bar that has
   not closed can never be selected, and the index chosen for MY bar i is a
   function of bar i's close time alone, so it is invariant to how much future
   data exists in either frame.
2. **Trailing-only peer statistics.** Peer values are ``ta.momentum`` /
   rolling-std over trailing windows on the peer's own 1D closes. No
   full-sample mean, quantile, z-score, ``bfill``, or ``shift(-n)`` is applied
   to a peer series, so peer value at peer-bar j depends only on peer bars
   ``<= j``. Combined with (1), MY bar i sees only peer information that had
   already printed by bar i's close.

The cross-sectional rank is a rank across the 20 aligned peer values AT MY BAR
i only -- never across time -- so it introduces no additional time dependence.
"""

from __future__ import annotations

import numpy as np

import core
import indicators as ta
from core import LONG, SHORT, Plan
from registry import register
from universe import UNIVERSE

# 4H trigger, 1D signal tier. 180 trigger bars = 30 days, the horizon a 30-90
# day momentum signal actually needs; the 96-bar default would truncate winners.
_TFS = ("1d", "4h")
_TRIG = "4h"
_HOLD = 180

# Rank is meaningless on a handful of names; require most of the universe.
_MIN_PEERS = 12

# Peer series cache: (symbol, kind, period) -> (close_times, values). Populated
# once per worker process; core.load_frame already caches the raw frames.
_PEER_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray] | None] = {}


def _peer_signal(symbol: str, kind: str, period: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Trailing 1D momentum (or vol-scaled momentum) for one symbol.

    Returns ``(close_times, values)`` with close_times ascending, or None if the
    symbol has no 1D data. Values are trailing-window only -- see the module
    docstring's causality note.
    """
    key = (symbol, kind, period)
    if key in _PEER_CACHE:
        return _PEER_CACHE[key]
    df = core.load_frame(symbol, "1d")
    if df.empty or len(df) < period + 2:
        _PEER_CACHE[key] = None
        return None
    close = df["close"]
    mom = ta.momentum(close, period)
    if kind == "mom":
        vals = mom
    elif kind == "sharpe":
        # Return per unit of trailing realised vol: a Sharpe-ranked rather than
        # return-ranked signal. The annualisation constant is omitted because a
        # positive constant cannot change a cross-sectional ordering.
        lr = np.log(close / close.shift(1))
        sd = lr.rolling(period, min_periods=period).std(ddof=1)
        vals = mom / sd.replace(0.0, np.nan)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown peer signal kind {kind!r}")
    out = (
        df.index.to_numpy() + core.TIMEFRAME_MS["1d"],
        np.asarray(vals.to_numpy(), dtype=float),
    )
    _PEER_CACHE[key] = out
    return out


def _align_peer(peer: tuple[np.ndarray, np.ndarray], my_close_times: np.ndarray) -> np.ndarray:
    """Project a peer series onto MY trigger grid by bar CLOSE time.

    Identical rule to ``Ctx._index_map``: the last peer bar CLOSED at or before
    my bar's close. NaN where no peer bar had closed yet.
    """
    peer_ct, peer_vals = peer
    idx = np.searchsorted(peer_ct, my_close_times, side="right") - 1
    out = np.full(len(my_close_times), np.nan, dtype=float)
    ok = idx >= 0
    out[ok] = peer_vals[idx[ok]]
    return out


def _own_signal(ctx, kind: str, period: int) -> np.ndarray:
    """The same statistic as ``_peer_signal``, but off the CTX's own 1D frame.

    Using ctx.frame("1d") rather than the peer cache means the own-symbol leg is
    genuinely truncated by ``assert_causal``, so the audit does bite on it.
    """
    f1 = ctx.frame("1d")
    close = f1["close"]
    mom = ta.momentum(close, period)
    if kind == "sharpe":
        lr = np.log(close / close.shift(1))
        sd = lr.rolling(period, min_periods=period).std(ddof=1)
        mom = mom / sd.replace(0.0, np.nan)
    return ctx.align(mom, "1d")


def _xs_rank(ctx, kind: str, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-sectional percentile rank of this symbol against the other 19.

    Returns ``(rank, n_valid)``. ``rank`` is the fraction of peers with a
    STRICTLY SMALLER value at that bar (0 = worst, 1 = best), NaN where fewer
    than ``_MIN_PEERS`` peers have a value or where our own value is NaN.
    """
    my_close_times = ctx.close_times(ctx.trigger_tf)
    mine = _own_signal(ctx, kind, period)

    cols = []
    for sym in UNIVERSE:
        if sym == ctx.symbol:
            continue
        peer = _peer_signal(sym, kind, period)
        if peer is None:
            continue
        cols.append(_align_peer(peer, my_close_times))
    if not cols:
        nan = np.full(ctx.n, np.nan)
        return nan, np.zeros(ctx.n)

    mat = np.vstack(cols)                      # (n_peers, n_bars)
    valid = np.isfinite(mat)
    n_valid = valid.sum(axis=0).astype(float)
    below = (valid & (mat < mine[None, :])).sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rank = below / np.where(n_valid > 0, n_valid, np.nan)
    rank[~np.isfinite(mine)] = np.nan
    rank[n_valid < _MIN_PEERS] = np.nan
    return rank, n_valid


def _stop(ctx, k: float) -> np.ndarray:
    """``k * ATR(14)`` of the 1D tier, on the trigger grid."""
    return k * ctx.align(ta.atr(ctx.frame("1d"), 14), "1d")


# ---------------------------------------------------------------------------
# 1. Time-series momentum -- the plain, most-replicated form
# ---------------------------------------------------------------------------


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"lookback": [30, 60, 90], "min_abs": [0.0, 0.05], "k_atr": [1.5, 2.5]},
    rationale=(
        "Time-series momentum: long when the trailing N-day log return is "
        "positive, short when negative. The most replicated directional effect "
        "in the asset-pricing literature (Moskowitz-Ooi-Pedersen) and the "
        "single-asset baseline any cleverer momentum variant must beat before it "
        "earns its complexity. ``min_abs`` adds a dead band so near-zero drift "
        "does not generate churn that costs fund the spread."
    ),
)
def mom_tsmom(ctx, lookback, min_abs, k_atr):
    mom = _own_signal(ctx, "mom", lookback)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[mom >= min_abs] = LONG
    entry[mom <= -min_abs] = SHORT
    if min_abs == 0.0:
        # With no dead band the two masks overlap only at exactly 0; resolve to
        # flat so a zero reading is not silently a short.
        entry[mom == 0.0] = 0
    # Direction-agnostic exit: the signal has decayed back into the dead band
    # (or gone NaN), so the reason for holding no longer exists. With
    # min_abs == 0 this reduces to an exact sign flip.
    exit_sig = ~(np.abs(mom) > min_abs) if min_abs > 0.0 else ~np.isfinite(mom)
    return Plan(
        entry=entry, stop_dist=_stop(ctx, k_atr), exit_signal=exit_sig,
        note=f"tsmom{lookback}d",
    )


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"lookback": [60, 90], "trail_k": [2.0, 3.0], "k_atr": [2.0, 3.0]},
    rationale=(
        "Time-series momentum with a ratcheting ATR trail instead of a fixed "
        "stop and a time exit. Momentum's published return profile is strongly "
        "right-skewed -- few large winners fund many small losers -- so the exit "
        "rule, not the entry rule, is where most of its Sharpe is won or lost. "
        "Separated from ``mom_tsmom`` so the entry and exit questions are "
        "answered independently rather than confounded in one grid."
    ),
)
def mom_tsmom_trail(ctx, lookback, trail_k, k_atr):
    mom = _own_signal(ctx, "mom", lookback)
    atr = ctx.align(ta.atr(ctx.frame("1d"), 14), "1d")
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[mom > 0.0] = LONG
    entry[mom < 0.0] = SHORT
    return Plan(
        entry=entry, stop_dist=k_atr * atr, trail_atr=trail_k * atr,
        note=f"tsmom{lookback}d-trail",
    )


# ---------------------------------------------------------------------------
# 2. Cross-sectional relative strength -- the Option B thesis
# ---------------------------------------------------------------------------


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"lookback": [30, 60, 90], "top_q": [0.2, 0.3], "k_atr": [1.5, 2.5]},
    rationale=(
        "Cross-sectional relative strength across the 20-symbol universe: long "
        "when this symbol's trailing N-day return ranks in the top quintile of "
        "its peers, short in the bottom. This is the mechanism the PRD's Option "
        "B is betting on -- published Sharpe >1.5 comes from breadth and "
        "dispersion, not from a better single-asset filter -- and it is "
        "untestable on the 3-symbol production universe."
    ),
)
def mom_xsec_rs(ctx, lookback, top_q, k_atr):
    rank, _ = _xs_rank(ctx, "mom", lookback)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[rank >= 1.0 - top_q] = LONG
    entry[rank <= top_q] = SHORT
    return Plan(entry=entry, stop_dist=_stop(ctx, k_atr), note=f"xsrs{lookback}d")


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"lookback": [30, 60, 90], "top_q": [0.2, 0.3], "k_atr": [1.5, 2.5]},
    rationale=(
        "Sharpe-ranked cross-sectional momentum: rank on trailing return DIVIDED "
        "BY trailing realised vol rather than raw return. Raw-return ranking is "
        "structurally biased toward the highest-vol names, which in a 20-perp "
        "crypto universe means it is partly a bet on volatility rather than on "
        "momentum. Vol-scaling removes that confound, which is also why "
        "inverse-vol weighting is the sizing layer the PRD already adopted."
    ),
)
def mom_xsec_sharpe(ctx, lookback, top_q, k_atr):
    rank, _ = _xs_rank(ctx, "sharpe", lookback)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[rank >= 1.0 - top_q] = LONG
    entry[rank <= top_q] = SHORT
    return Plan(entry=entry, stop_dist=_stop(ctx, k_atr), note=f"xssharpe{lookback}d")


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"lookback": [30, 60, 90], "top_q": [0.2, 0.3], "k_atr": [1.5, 2.5]},
    rationale=(
        "Dual momentum: absolute AND relative agreement required -- long only "
        "when trailing return is positive AND the symbol ranks top-quintile, "
        "short only when negative AND bottom-quintile. Pure cross-sectional "
        "momentum still buys the best-performing name in a market that is "
        "falling as a whole; the absolute leg is the standard fix, and it is "
        "cheap because it consumes no extra parameter."
    ),
)
def mom_dual(ctx, lookback, top_q, k_atr):
    mom = _own_signal(ctx, "mom", lookback)
    rank, _ = _xs_rank(ctx, "mom", lookback)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(rank >= 1.0 - top_q) & (mom > 0.0)] = LONG
    entry[(rank <= top_q) & (mom < 0.0)] = SHORT
    return Plan(entry=entry, stop_dist=_stop(ctx, k_atr), note=f"dual{lookback}d")


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"lookback": [30, 60, 90], "edge": [0.0, 0.05], "k_atr": [1.5, 2.5]},
    rationale=(
        "Beta-relative momentum: long an alt that is outperforming BTC by at "
        "least ``edge`` while BTC's own trailing momentum is positive, short it "
        "when it underperforms BTC in a BTC downtrend. Crypto returns are "
        "dominated by a single common factor, so BTC is the natural benchmark; "
        "conditioning on the BTC leg is a one-asset, one-parameter proxy for the "
        "full cross-section and a useful control on whether breadth per se adds "
        "anything beyond 'beat the market factor'."
    ),
)
def mom_vs_btc(ctx, lookback, edge, k_atr):
    mine = _own_signal(ctx, "mom", lookback)
    btc = _peer_signal("BTCUSDT", "mom", lookback)
    if btc is None:  # pragma: no cover - BTC is always present in this DB
        return Plan(entry=np.zeros(ctx.n, np.int8), stop_dist=_stop(ctx, k_atr))
    bmom = _align_peer(btc, ctx.close_times(ctx.trigger_tf))
    rel = mine - bmom
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(rel >= edge) & (bmom > 0.0)] = LONG
    entry[(rel <= -edge) & (bmom < 0.0)] = SHORT
    return Plan(entry=entry, stop_dist=_stop(ctx, k_atr), note=f"vsbtc{lookback}d")


# ---------------------------------------------------------------------------
# 3. Momentum shape: MACD, acceleration, crash protection
# ---------------------------------------------------------------------------


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"zwin": [60, 120], "thr": [0.5, 1.0, 1.5], "k_atr": [1.5, 2.5]},
    rationale=(
        "MACD histogram on 1D closes, price-normalised and then z-scored over a "
        "trailing window, entered when it exceeds ``thr`` standard deviations. "
        "The raw histogram's scale is symbol- and era-specific, so a fixed "
        "threshold is untradeable across 20 perps; the trailing z-score is the "
        "causal way to ask 'is momentum unusually strong FOR THIS SYMBOL RIGHT "
        "NOW' without a full-sample statistic."
    ),
)
def mom_macd_hist(ctx, zwin, thr, k_atr):
    f1 = ctx.frame("1d")
    hist = ta.macd(f1["close"], 12, 26, 9)["hist"]
    z = ctx.align(ta.zscore(hist, zwin), "1d")
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[z >= thr] = LONG
    entry[z <= -thr] = SHORT
    return Plan(entry=entry, stop_dist=_stop(ctx, k_atr), note=f"macdz{zwin}")


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"lookback": [30, 60], "gap": [10, 20], "k_atr": [1.5, 2.5]},
    rationale=(
        "Acceleration: trade the CHANGE in momentum rather than its level -- "
        "enter long when the N-day return is both positive and higher than it "
        "was ``gap`` days ago. Momentum's level is a slow, heavily "
        "autocorrelated state; its first difference is the part that actually "
        "carries new information, and it turns earlier at trend inflections. "
        "Tested plainly so the added derivative can be judged against "
        "``mom_tsmom`` on identical stops and horizon."
    ),
)
def mom_accel(ctx, lookback, gap, k_atr):
    f1 = ctx.frame("1d")
    mom = ta.momentum(f1["close"], lookback)
    dmom = mom - mom.shift(gap)
    m = ctx.align(mom, "1d")
    d = ctx.align(dmom, "1d")
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(m > 0.0) & (d > 0.0)] = LONG
    entry[(m < 0.0) & (d < 0.0)] = SHORT
    return Plan(entry=entry, stop_dist=_stop(ctx, k_atr), note=f"accel{lookback}/{gap}")


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"long_lb": [60, 90], "short_lb": [10, 20], "k_atr": [1.5, 2.5]},
    rationale=(
        "Momentum-crash protection: take the long-horizon momentum signal, but "
        "REFUSE the trade and exit an open one whenever short-horizon momentum "
        "disagrees in sign. Momentum crashes are the effect's documented tail "
        "risk (Daniel-Moskowitz), and they announce themselves as a fast "
        "reversal against a still-positive slow signal. The exit_signal leg is "
        "what makes this different from an extra entry filter."
    ),
)
def mom_crash_guard(ctx, long_lb, short_lb, k_atr):
    slow = _own_signal(ctx, "mom", long_lb)
    fast = _own_signal(ctx, "mom", short_lb)
    agree_up = (slow > 0.0) & (fast > 0.0)
    agree_dn = (slow < 0.0) & (fast < 0.0)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[agree_up] = LONG
    entry[agree_dn] = SHORT
    # Direction-agnostic exit: fire wherever slow and fast no longer agree.
    diverge = ~(agree_up | agree_dn)
    return Plan(
        entry=entry, stop_dist=_stop(ctx, k_atr), exit_signal=diverge,
        note=f"crashguard{long_lb}/{short_lb}",
    )


@register(
    family="momentum",
    trigger_tf=_TRIG,
    timeframes=_TFS,
    max_hold_bars=_HOLD,
    grid={"lookback": [30, 60, 90], "top_q": [0.2, 0.3], "k_atr": [1.5, 2.5]},
    long_only=True,
    rationale=(
        "Long-only dual momentum. Crypto perp shorts pay funding against a "
        "market with a strong secular upward drift over the sample, and the "
        "cross-sectional short leg is where a breadth strategy most plausibly "
        "loses its published edge. Registering the long-only twin makes that "
        "attributable: if it beats ``mom_dual``, the short leg is a cost, not "
        "an edge, and the Option B verdict must say so."
    ),
)
def mom_dual_long_only(ctx, lookback, top_q, k_atr):
    mom = _own_signal(ctx, "mom", lookback)
    rank, _ = _xs_rank(ctx, "mom", lookback)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(rank >= 1.0 - top_q) & (mom > 0.0)] = LONG
    entry[(rank <= top_q) & (mom < 0.0)] = SHORT  # suppressed by long_only
    return Plan(entry=entry, stop_dist=_stop(ctx, k_atr), note=f"dualLO{lookback}d")
