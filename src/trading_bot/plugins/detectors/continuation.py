"""
Catalog family 2 (Continuation Patterns) as registered Detector plug-ins:
`cup-and-handle`, `inverse-cup-and-handle`, `bull-flag`, `bear-flag`,
`ascending-triangle`, `descending-triangle`, `symmetrical-triangle`,
`falling-wedge`, `rising-wedge`.

NINE of the catalog's fifteen continuation rows. Bull/Bear Pennant are
`deferred` in `plugins/detectors/catalog.py` (a flag with a CONVERGING rather
than parallel consolidation — a genuine small build, held back to keep tier 2
closed); the remaining four are `out-of-scope` with a reason each.

ATTRIBUTION (contract §0b):
  * Triangles and wedges refine `signals/patterns.py:272-370`, whose single
    `triangle` kind conflated FIVE distinct shapes with different directional
    priors, and whose line 327 made both wedges structurally unreachable. The
    line fit lives in `_geometry.fit_converging_lines`; the slope signs live in
    `_classify` here. `scripts/bruteforce/indicators.py:533`'s
    `triangle_squeeze` is deliberately NOT used: it is a range-contraction
    MEASUREMENT stand-in with no trendlines, so it cannot supply a level.
  * Flags port `signals/patterns.py:388-464`, which is strictly stronger than
    the `scripts/bruteforce/indicators.py:552` donor — the donor measures the
    pole from two `shift()`ed closes and has NO containment test, so it accepts
    a pole that already gave back its gain.
  * Cup & handle is genuinely NEW. No donor exists anywhere in this repo; it is
    the one tier-1 shape with no implementation to port.

WHAT THESE DETECTORS DELIBERATELY DO NOT DECIDE. The catalog says a symmetrical
triangle continues "in the prevailing trend", and lists cup & handle as a
continuation pattern — but nothing here tests for a prior trend. That is a
`Confirmation`'s job (Phase 4), and baking it in would hide the decision inside
geometry where no report could see it. The symmetrical triangle therefore emits
BOTH a long and a short candidate and lets the trigger bar pick at most one,
exactly as `patterns.py:347-349` did.
"""

import logging

from trading_bot import config
from trading_bot.framework.contracts import DetectedEvent, ParamSpec
from trading_bot.framework.registry import register
from trading_bot.plugins.detectors import _geometry as g
from trading_bot.signals.pivots import find_pivots

logger = logging.getLogger("trading_bot")

SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME

# The five shapes one converging line fit can describe. Ordered as `_classify`
# tests them, because that order is load-bearing (see `_classify`).
TRIANGLE_SHAPES = (
    "symmetrical-triangle",
    "ascending-triangle",
    "descending-triangle",
    "falling-wedge",
    "rising-wedge",
)


def _scan_params() -> dict:
    return {
        "pivot_span": ParamSpec(
            kind="int",
            default=config.PIVOT_SPAN,
            bounds=(2, 8),
            doc="Bars each side a fractal pivot must strictly dominate",
        ),
        "max_age_bars": ParamSpec(
            kind="int",
            default=config.PATTERN_MAX_AGE_BARS,
            bounds=(2, 60),
            doc="Freshness bound on the pattern's last pivot (setup bars)",
        ),
        "lookback_bars": ParamSpec(
            kind="int",
            default=config.PATTERN_LOOKBACK_BARS,
            bounds=(60, 500),
            doc="Setup bars the detector may see",
        ),
    }


# --------------------------------------------------------------------------- #
# Triangles and wedges — the classification layer over one shared line fit.
# --------------------------------------------------------------------------- #


def _classify(fit, *, flat_tol: float, min_slope: float) -> str | None:
    """Which of TRIANGLE_SHAPES a LineFit's two slopes describe, or None.

    THIS FUNCTION IS WHERE `signals/patterns.py:327`'s WORK WENT. That line
    rejected any fit whose upper side rose or lower side fell — which is every
    wedge. `fit_converging_lines` no longer rejects them, so something must
    decide what the signed slopes mean; deleting the guard without this would
    relabel every rising wedge a "symmetrical triangle" and emit a LONG
    candidate for a bearish shape.

    Slopes are `_geometry`'s NORMALISED per-bar slopes, so `flat_tol = 0.001`
    means "0.1% of price per bar" and is comparable across symbols priced from
    $0.10 to $100k. A raw-slope tolerance would be meaningless across a universe.

    BRANCH ORDER IS LOAD-BEARING and this returns the FIRST match, which is what
    makes the five shapes mutually exclusive BY CONSTRUCTION rather than by
    arithmetic coincidence. A falling wedge and a descending triangle differ
    ONLY by whether the lower line is flat within `flat_tol`; if a caller sets
    `min_slope < flat_tol` the two predicates can both be true, and the more
    specific one (a FLAT boundary — a level being repeatedly tested) must win.
    Pinned by `test_shape_classification_is_mutually_exclusive`.

    Args:
        fit: A `_geometry.LineFit`.
        flat_tol: |slope| <= this counts as a flat boundary.
        min_slope: Both boundaries must exceed this, same sign, for a wedge.

    Returns:
        One of TRIANGLE_SHAPES, or None when the slopes describe none of them.
    """
    u, l = fit.upper_slope, fit.lower_slope
    if u < -flat_tol and l > flat_tol:
        return "symmetrical-triangle"
    if abs(u) <= flat_tol and l > flat_tol:
        return "ascending-triangle"
    if u < -flat_tol and abs(l) <= flat_tol:
        return "descending-triangle"
    if u < -min_slope and l < -min_slope:
        return "falling-wedge"
    if u > min_slope and l > min_slope:
        return "rising-wedge"
    return None


def _fit(
    ctx,
    *,
    pivot_span: int,
    max_age_bars: int,
    lookback_bars: int,
    min_pivots_per_side: int,
    min_convergence: float,
    containment_tol: float,
    min_width_bars: int,
    max_width_bars: int,
):
    """(df, LineFit | None) for the current setup bar. Shared by all five shapes."""
    df = ctx.window(SETUP_TF, lookback_bars)
    if len(df) == 0:
        return df, None
    fresh = g.fresh_factory(len(df), max_age_bars)
    fit = g.fit_converging_lines(
        df,
        find_pivots(df, span=pivot_span),
        min_pivots_per_side=min_pivots_per_side,
        min_width_bars=min_width_bars,
        max_width_bars=max_width_bars,
        min_convergence=min_convergence,
        containment_tol=containment_tol,
        fresh=fresh,
    )
    return df, fit


def _triangle_meta(fit) -> dict:
    return {
        "upper_slope": float(fit.upper_slope),
        "lower_slope": float(fit.lower_slope),
        "start_range": float(fit.start_range),
        "end_range": float(fit.end_range),
        "width_bars": float(fit.end_x - fit.start_x),
    }


def _emit_shape(df, fit, shape: str) -> list[DetectedEvent]:
    """DetectedEvents for a classified fit.

    `target_height = fit.start_range` for every shape: the widest vertical
    extent of the structure is the classical measured move. NOT
    `max(start_range, end_range)` — convergence already guarantees
    `end_range < start_range`, and if they ever invert that is a bug to
    surface, not to paper over.

    `end_ts = df.index[fit.end_x]` — the bar the breakout levels are evaluated
    at, matching `patterns.py:345-346`. These shapes are line-terminated, not
    pivot-terminated, so `confirmation_ts` does not apply: the levels are read
    at the latest CLOSED bar, which is already knowable.
    """
    start_ts = int(df.index[fit.start_x])
    end_ts = int(df.index[fit.end_x])
    meta = _triangle_meta(fit)
    height = float(fit.start_range)

    def ev(direction: str, level: float) -> DetectedEvent:
        return DetectedEvent(
            kind=shape,
            direction=direction,
            level=float(level),
            target_height=height,
            start_ts=start_ts,
            end_ts=end_ts,
            meta=meta,
        )

    if shape == "symmetrical-triangle":
        # Direction is decided by which side actually breaks; emit both and let
        # the trigger bar pick at most one (patterns.py:347-349).
        return [ev("long", fit.upper_end), ev("short", fit.lower_end)]
    if shape in ("ascending-triangle", "falling-wedge"):
        return [ev("long", fit.upper_end)]
    return [ev("short", fit.lower_end)]


_TRIANGLE_PARAMS = {
    **_scan_params(),
    "min_pivots_per_side": ParamSpec(
        kind="int",
        default=config.TRIANGLE_MIN_PIVOTS_PER_SIDE,
        bounds=(2, 6),
        doc="Pivot highs and pivot lows required (floored at 2: a line needs two points)",
    ),
    "min_convergence": ParamSpec(
        kind="float",
        default=config.TRIANGLE_MIN_CONVERGENCE,
        bounds=(0.05, 0.90),
        doc="Required fractional contraction of the vertical range",
    ),
    "containment_tol": ParamSpec(
        kind="float",
        default=config.TRIANGLE_CONTAINMENT_TOL,
        bounds=(0.0, 0.05),
        doc="Fraction of price a bar may pierce its own trendline by",
    ),
    "min_width_bars": ParamSpec(
        kind="int",
        default=config.TRIANGLE_MIN_WIDTH_BARS,
        bounds=(8, 200),
        doc="Minimum width of the fitted structure, in setup bars",
    ),
    "max_width_bars": ParamSpec(
        kind="int",
        default=config.TRIANGLE_MAX_WIDTH_BARS,
        bounds=(20, 400),
        doc="Only pivots inside the last this-many bars are fitted",
    ),
    "flat_tol": ParamSpec(
        kind="float",
        default=config.TRIANGLE_FLAT_SLOPE_TOL,
        bounds=(0.0, 0.02),
        doc="Normalised per-bar |slope| at or below which a boundary is FLAT",
    ),
    "min_slope": ParamSpec(
        kind="float",
        default=config.WEDGE_MIN_SLOPE,
        bounds=(0.0, 0.02),
        doc="Normalised per-bar slope both wedge boundaries must exceed, same sign",
    ),
}


def _shape_detector(ctx, shape: str, *, flat_tol: float, min_slope: float, **fit_kw):
    """Fit once, classify once, emit only when the classification is `shape`."""
    df, fit = _fit(ctx, **fit_kw)
    if fit is None:
        return []
    if _classify(fit, flat_tol=flat_tol, min_slope=min_slope) != shape:
        return []
    return g.dedupe_events(_emit_shape(df, fit, shape))


@register(
    "detector",
    name="symmetrical-triangle",
    params=_TRIANGLE_PARAMS,
    rationale=(
        "Both boundaries converging toward each other — the catalog's "
        "'continuation in prevailing trend'. NOT in contract §9's tier list and "
        "no success criterion depends on it: it is a near-zero-marginal-cost "
        "mirror sharing the same line fit as the tier-2 shapes, registered so "
        "the five shapes signals/patterns.py conflated under one 'triangle' "
        "kind can finally be measured apart. Emits BOTH directions because the "
        "detector deliberately does not determine the prevailing trend."
    ),
    timeframes=(SETUP_TF,),
)
def symmetrical_triangle(ctx, **params) -> list[DetectedEvent]:
    """Falling upper boundary, rising lower boundary. Long and short candidates."""
    return _shape_detector(ctx, "symmetrical-triangle", **params)


@register(
    "detector",
    name="ascending-triangle",
    params=_TRIANGLE_PARAMS,
    rationale=(
        "Contract §9 tier 2 (the catalog rates Ascending Triangle four stars). "
        "Horizontal resistance repeatedly tested while lows rise is the "
        "classical accumulation-into-breakout shape; refined from "
        "signals/patterns.py:272, whose single 'triangle' kind conflated five "
        "distinct shapes with different directional priors and so could never "
        "show which of them carried the edge."
    ),
    timeframes=(SETUP_TF,),
    tier=2,
)
def ascending_triangle(ctx, **params) -> list[DetectedEvent]:
    """Flat upper boundary, rising lower boundary. Long only."""
    return _shape_detector(ctx, "ascending-triangle", **params)


@register(
    "detector",
    name="descending-triangle",
    params=_TRIANGLE_PARAMS,
    rationale=(
        "Horizontal support repeatedly tested while highs fall — the mirror of "
        "the ascending triangle and free once its line fit exists. NOT in "
        "contract §9's tier list and no success criterion depends on it; the "
        "ledger records it with tier=None for exactly that reason."
    ),
    timeframes=(SETUP_TF,),
)
def descending_triangle(ctx, **params) -> list[DetectedEvent]:
    """Falling upper boundary, flat lower boundary. Short only."""
    return _shape_detector(ctx, "descending-triangle", **params)


@register(
    "detector",
    name="falling-wedge",
    params=_TRIANGLE_PARAMS,
    rationale=(
        "Contract §9 tier 2 (the catalog rates Falling Wedge four stars). Two "
        "falling, converging boundaries mean sellers are losing ground faster "
        "than buyers, which is why the classical reading is bullish. NOT "
        "emittable by signals/patterns.py at all: its expanding-side guard "
        "(line 327) rejects any shape whose lower line falls, so this is one of "
        "the two tier-2 patterns that were structurally unreachable before this "
        "phase relaxed the guard into _classify."
    ),
    timeframes=(SETUP_TF,),
    tier=2,
)
def falling_wedge(ctx, **params) -> list[DetectedEvent]:
    """Both boundaries falling and converging. Long."""
    return _shape_detector(ctx, "falling-wedge", **params)


# --------------------------------------------------------------------------- #
# rising-wedge — THE ZERO-ENGINE-EDIT PROOF DETECTOR.
#
# Implemented LAST, after every shared helper existed, and it reads NO
# `config` attribute at all: every default below is an inline literal. That is
# not stylistic. The PRD's headline success metric is "a new detector or rule
# added with ZERO engine-core edits", and this detector is how that gets
# MEASURED rather than claimed: adding it changes exactly two paths,
#   src/trading_bot/plugins/detectors/continuation.py
#   tests/test_detectors_continuation.py
# and nothing in config.py, cli.py, framework/, signals/ or backtest/. Reading a
# config constant would have added a third path and destroyed the proof.
#
# THE COST OF THAT, stated so it is not discovered later: these literals
# DUPLICATE the config defaults the other four shapes read, and could drift from
# them. `test_rising_wedge_defaults_match_the_shared_ones` is the drift guard.
# --------------------------------------------------------------------------- #


@register(
    "detector",
    name="rising-wedge",
    params={
        "pivot_span": ParamSpec(
            kind="int", default=3, bounds=(2, 8),
            doc="Bars each side a fractal pivot must strictly dominate",
        ),
        "max_age_bars": ParamSpec(
            kind="int", default=12, bounds=(2, 60),
            doc="Freshness bound on the pattern's last pivot (setup bars)",
        ),
        "lookback_bars": ParamSpec(
            kind="int", default=180, bounds=(60, 500),
            doc="Setup bars the detector may see",
        ),
        "min_pivots_per_side": ParamSpec(
            kind="int", default=2, bounds=(2, 6),
            doc="Pivot highs and pivot lows required (floored at 2)",
        ),
        "min_convergence": ParamSpec(
            kind="float", default=0.25, bounds=(0.05, 0.90),
            doc="Required fractional contraction of the vertical range",
        ),
        "containment_tol": ParamSpec(
            kind="float", default=0.005, bounds=(0.0, 0.05),
            doc="Fraction of price a bar may pierce its own trendline by",
        ),
        "min_width_bars": ParamSpec(
            kind="int", default=20, bounds=(8, 200),
            doc="Minimum width of the fitted structure, in setup bars",
        ),
        "max_width_bars": ParamSpec(
            kind="int", default=80, bounds=(20, 400),
            doc="Only pivots inside the last this-many bars are fitted",
        ),
        "flat_tol": ParamSpec(
            kind="float", default=0.001, bounds=(0.0, 0.02),
            doc="Normalised per-bar |slope| at or below which a boundary is FLAT",
        ),
        "min_slope": ParamSpec(
            kind="float", default=0.001, bounds=(0.0, 0.02),
            doc="Normalised per-bar slope both boundaries must exceed, same sign",
        ),
    },
    rationale=(
        "Contract §9 tier 2 (the catalog rates Rising Wedge four stars). Two "
        "RISING, converging boundaries mean buyers are making progress at a "
        "shrinking rate, which is why the classical reading is bearish. Like the "
        "falling wedge it was NOT emittable by signals/patterns.py at all — its "
        "expanding-side guard (line 327) rejects any shape whose upper line "
        "rises, and a rising wedge's does by definition. Registered LAST and "
        "reading no config attribute, so the diff footprint of adding a detector "
        "is measured rather than asserted."
    ),
    timeframes=(SETUP_TF,),
    tier=2,
)
def rising_wedge(ctx, **params) -> list[DetectedEvent]:
    """Both boundaries rising and converging. Short."""
    return _shape_detector(ctx, "rising-wedge", **params)


# --------------------------------------------------------------------------- #
# Flags — ported from signals/patterns.py:388-464.
# --------------------------------------------------------------------------- #

_FLAG_PARAMS = {
    "lookback_bars": _scan_params()["lookback_bars"],
    "pole_window_bars": ParamSpec(
        kind="int",
        default=config.FLAG_POLE_WINDOW_BARS,
        bounds=(3, 60),
        doc="Setup bars the impulse pole is measured over",
    ),
    "pole_min_pct": ParamSpec(
        kind="float",
        default=config.FLAG_POLE_MIN_PCT,
        bounds=(0.005, 0.40),
        doc="Pole must move at least this fraction of price",
    ),
    "consol_min_bars": ParamSpec(
        kind="int",
        default=config.FLAG_CONSOL_MIN_BARS,
        bounds=(2, 40),
        doc="Shortest consolidation considered (bars)",
    ),
    "consol_max_bars": ParamSpec(
        kind="int",
        default=config.FLAG_CONSOL_MAX_BARS,
        bounds=(3, 80),
        doc="Longest consolidation considered (bars)",
    ),
    "max_retrace": ParamSpec(
        kind="float",
        default=config.FLAG_MAX_RETRACE,
        bounds=(0.1, 0.95),
        doc="Consolidation may retrace at most this fraction of the pole",
    ),
}

_FLAG_RATIONALE_TAIL = (
    "Contract §9 tier 2 (the catalog rates Bull/Bear Flag four stars). Ported "
    "from signals/patterns.py:388, which is strictly stronger than the "
    "scripts/bruteforce/indicators.py:552 donor: the donor measures the pole "
    "from two shift()ed closes and has no containment test, so it accepts a "
    "pole that already gave back its gain."
)


@register(
    "detector",
    name="bull-flag",
    params=_FLAG_PARAMS,
    rationale="Impulse up, then a shallow drift that holds most of the gain. "
    + _FLAG_RATIONALE_TAIL,
    timeframes=(SETUP_TF,),
    tier=2,
)
def bull_flag(ctx, **params) -> list[DetectedEvent]:
    """Rising pole + shallow consolidation ending at the latest bar. Long."""
    return _detect_flag(ctx, direction="long", **params)


@register(
    "detector",
    name="bear-flag",
    params=_FLAG_PARAMS,
    rationale="Impulse down, then a shallow bounce that keeps most of the loss. "
    + _FLAG_RATIONALE_TAIL,
    timeframes=(SETUP_TF,),
    tier=2,
)
def bear_flag(ctx, **params) -> list[DetectedEvent]:
    """Falling pole + shallow consolidation ending at the latest bar. Short."""
    return _detect_flag(ctx, direction="short", **params)


def _detect_flag(
    ctx,
    *,
    direction: str,
    lookback_bars: int,
    pole_window_bars: int,
    pole_min_pct: float,
    consol_min_bars: int,
    consol_max_bars: int,
    max_retrace: float,
) -> list[DetectedEvent]:
    """Impulse pole + shallow contained consolidation ending at the latest bar.

    For each consolidation length c (SMALLEST FIRST, so the tightest
    consolidation wins), the pole is the move into the bar just before the
    consolidation, measured over the `pole_window_bars` bars ending there.

    THREE DELIBERATE OMISSIONS, each with a reason:
      * FRESHNESS is skipped. A flag's consolidation ends at the latest bar by
        construction, so it is inherently fresh (patterns.py:96-98). Threading
        `fresh` in would reject every flag whose POLE is older than
        `PATTERN_MAX_AGE_BARS` — i.e. most of them.
      * `level_unbroken` is skipped. The consolidation extreme by definition has
        not been closed through, or the consolidation would have ended there.
      * `confirmation_ts` does not apply: a flag is bar-terminated, not
        pivot-terminated, so `end_ts` is the latest closed bar.

    Containment is tested against the pole's extreme over the WHOLE pole window,
    the same window the pole height is measured over — checking only the final
    pole bar's high rejected valid flags whose pole peaked a bar or two earlier
    (patterns.py:402-404). The pole slice is INCLUSIVE, so `pole_start`
    subtracts `pole_window_bars - 1`, not `pole_window_bars`.
    """
    df = ctx.window(SETUP_TF, lookback_bars)
    n = len(df)
    out: list[DetectedEvent] = []
    if n == 0:
        return out

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    ts_vals = df.index.to_numpy()
    kind = "bull-flag" if direction == "long" else "bear-flag"

    for c in range(consol_min_bars, consol_max_bars + 1):
        pole_end = n - 1 - c
        if pole_end < 1:
            break
        pole_start = max(0, pole_end - pole_window_bars + 1)
        pole_high = highs[pole_start : pole_end + 1].max()
        pole_low = lows[pole_start : pole_end + 1].min()
        consol_high = highs[pole_end + 1 :].max()
        consol_low = lows[pole_end + 1 :].min()

        if direction == "long":
            pole_height = closes[pole_end] - pole_low
            ok = (
                closes[pole_end] > 0
                and pole_height / closes[pole_end] >= pole_min_pct
                and consol_low >= closes[pole_end] - max_retrace * pole_height
                and consol_high <= pole_high
            )
            level = float(consol_high)
        else:
            pole_height = pole_high - closes[pole_end]
            ok = (
                closes[pole_end] > 0
                and pole_height / closes[pole_end] >= pole_min_pct
                and consol_high <= closes[pole_end] + max_retrace * pole_height
                and consol_low >= pole_low
            )
            level = float(consol_low)

        if not ok:
            continue
        out.append(
            DetectedEvent(
                kind=kind,
                direction=direction,
                level=level,
                target_height=float(pole_height),
                start_ts=int(ts_vals[pole_start]),
                end_ts=int(ts_vals[n - 1]),
                meta={
                    "pole_height": float(pole_height),
                    "pole_high": float(pole_high),
                    "pole_low": float(pole_low),
                    "consol_bars": float(c),
                },
            )
        )
        break  # smallest-first: the tightest consolidation wins

    return g.dedupe_events(out)


# --------------------------------------------------------------------------- #
# Cup & handle — the one genuinely new tier-1 build.
# --------------------------------------------------------------------------- #

_CUP_PARAMS = {
    **_scan_params(),
    "min_width_bars": ParamSpec(
        kind="int",
        default=config.CUP_MIN_WIDTH_BARS,
        bounds=(10, 200),
        doc="Minimum bars from left rim to right rim",
    ),
    "max_width_bars": ParamSpec(
        kind="int",
        default=config.CUP_MAX_WIDTH_BARS,
        bounds=(20, 400),
        doc="Maximum bars from left rim to right rim",
    ),
    "min_depth": ParamSpec(
        kind="float",
        default=config.CUP_MIN_DEPTH,
        bounds=(0.01, 0.40),
        doc="Cup depth as a fraction of the higher rim; under this is noise",
    ),
    "max_depth": ParamSpec(
        kind="float",
        default=config.CUP_MAX_DEPTH,
        bounds=(0.10, 0.90),
        doc="Over this it is a crash with a bounce, not a cup",
    ),
    "rim_tolerance": ParamSpec(
        kind="float",
        default=config.CUP_RIM_TOLERANCE,
        bounds=(0.0, 0.20),
        doc="Max relative height difference between the two rims",
    ),
    "round_band": ParamSpec(
        kind="float",
        default=config.CUP_ROUND_BAND,
        bounds=(0.05, 0.60),
        doc="Fraction of depth defining the base band near the bottom",
    ),
    "min_base_bars": ParamSpec(
        kind="int",
        default=config.CUP_MIN_BASE_BARS,
        bounds=(2, 40),
        doc="Bars whose extreme sits inside that band — the V-versus-cup test",
    ),
    "handle_min_bars": ParamSpec(
        kind="int",
        default=config.CUP_HANDLE_MIN_BARS,
        bounds=(1, 20),
        doc="Shortest handle, in bars after the right rim",
    ),
    "handle_max_bars": ParamSpec(
        kind="int",
        default=config.CUP_HANDLE_MAX_BARS,
        bounds=(3, 60),
        doc="Longest handle, in bars after the right rim",
    ),
    "handle_max_retrace": ParamSpec(
        kind="float",
        default=config.CUP_HANDLE_MAX_RETRACE,
        bounds=(0.05, 0.80),
        doc="Handle may give back at most this fraction of the cup's depth",
    ),
}

_CUP_RATIONALE_TAIL = (
    "Contract §9 tier 1 (five stars — the highest entry in the catalog's "
    "reliability table) and the only tier-1 shape with NO implementation "
    "anywhere in this repo, donor included. The roundness test — a minimum bar "
    "count whose extreme sits inside the bottom quartile of the depth — is what "
    "distinguishes it from a V reversal; without that test this detector is "
    "just a slow double bottom. The alternative (fitting a parabola and "
    "bounding its residuals) adds two fitted parameters for no measured benefit."
)


@register(
    "detector",
    name="cup-and-handle",
    params=_CUP_PARAMS,
    rationale="Rounded base between two comparable rims, then a shallow handle. "
    + _CUP_RATIONALE_TAIL,
    timeframes=(SETUP_TF,),
    tier=1,
)
def cup_and_handle(ctx, **params) -> list[DetectedEvent]:
    """Bullish cup & handle. Long on the higher rim."""
    return _detect_cup(ctx, inverse=False, **params)


@register(
    "detector",
    name="inverse-cup-and-handle",
    params=_CUP_PARAMS,
    rationale=(
        "Rounded DOME between two comparable rims, then a shallow upward handle "
        "— the mirror of the cup and nearly free once the cup's geometry exists. "
        "NOT in contract §9's tier list (only the UPRIGHT Cup & Handle is rated), "
        "so it carries tier=None in both the registry and the ledger and NO "
        "success criterion depends on it. The roundness test — a minimum bar "
        "count whose extreme sits inside the top quartile of the dome's height — "
        "is what distinguishes it from an inverted V; without that test this "
        "detector is just a slow double top."
    ),
    timeframes=(SETUP_TF,),
)
def inverse_cup_and_handle(ctx, **params) -> list[DetectedEvent]:
    """Bearish inverse cup & handle. Short on the lower rim."""
    return _detect_cup(ctx, inverse=True, **params)


def _detect_cup(
    ctx,
    *,
    inverse: bool,
    pivot_span: int,
    max_age_bars: int,
    lookback_bars: int,
    min_width_bars: int,
    max_width_bars: int,
    min_depth: float,
    max_depth: float,
    rim_tolerance: float,
    round_band: float,
    min_base_bars: int,
    handle_min_bars: int,
    handle_max_bars: int,
    handle_max_retrace: float,
) -> list[DetectedEvent]:
    """Cup & handle (and its inverse) over (rim, base, rim) pivot triples.

    Seven independent tests, in this order:
      1. Width — `min_width_bars <= R.index - L.index <= max_width_bars`.
      2. Rim symmetry — the two rims within `rim_tolerance` of each other.
      3. Depth — between `min_depth` and `max_depth` of the higher rim. Under
         8% is noise; over 50% is a crash with a bounce.
      4. ROUNDNESS — the only novel geometry here, and the test that separates a
         cup from a V. Count bars in `[L.index, R.index]` whose low sits inside
         the bottom `round_band` of the depth; require `min_base_bars`. A V
         bottom has one or two such bars; a rounded base has many. Decidable,
         cheap, and testable by construction.
      5. Bottom centrality — `B.index` inside the middle half of the span, so a
         base hugging one rim is rejected.
      6. Handle — length in `[handle_min_bars, handle_max_bars]`; its extreme
         holds within `handle_max_retrace` of the depth below the rim AND stays
         beyond the cup's midpoint (a handle giving back more than half the cup
         is a failed cup); its opposite extreme must not exceed the rim, or the
         rim already broke.
      7. `level_unbroken` from the right rim onward.

    FRESHNESS IS SKIPPED, for the flag's reason (patterns.py:96-98): the handle
    ends at the latest bar by construction, so the structure is inherently
    fresh. `max_age_bars` is still declared — it bounds nothing here today, and
    is kept in the spec only so the shared `_scan_params()` block stays one
    thing rather than two; see the phase report's degrees-of-freedom note.

    NO PRIOR-TREND TEST. The catalog lists cup & handle as a continuation
    pattern, but requiring a prior uptrend is a `Confirmation`'s job (Phase 4).
    Baking it in would hide the decision. Stated here so nobody adds it silently.

    COMPLEXITY. With `PIVOT_SPAN = 3` a 20-90 bar cup usually holds several
    noise pivots near the base, and `collapse_runs` merges only ADJACENT
    same-kind pivots, so an alternating micro-high/micro-low base survives and
    offers many (L, B, R) triples. The `R` scan is bounded to the most recent
    `pivot_span * 4` rims and the `L` loop breaks once the span exceeds
    `max_width_bars`; unbounded this is O(p^3) over a 180-bar lookback.
    """
    df = ctx.window(SETUP_TF, lookback_bars)
    n = len(df)
    out: list[DetectedEvent] = []
    if n == 0:
        return out

    pivots = g.collapse_runs(find_pivots(df, span=pivot_span))
    rim_kind = "low" if inverse else "high"
    base_kind = "high" if inverse else "low"
    kind = "inverse-cup-and-handle" if inverse else "cup-and-handle"
    direction = "short" if inverse else "long"

    rims = [p for p in pivots if p.kind == rim_kind]
    bases = [p for p in pivots if p.kind == base_kind]
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    # Bounded scan — see the docstring's COMPLEXITY note.
    for right in rims[-max(1, pivot_span * 4) :]:
        handle_bars = (n - 1) - right.index
        if not (handle_min_bars <= handle_bars <= handle_max_bars):
            continue
        for left in rims:
            width = right.index - left.index
            if width < min_width_bars:
                break  # rims are index-ordered: every later `left` is closer
            if width > max_width_bars:
                continue
            if abs(right.price - left.price) / max(right.price, left.price) > rim_tolerance:
                continue

            inner = [b for b in bases if left.index < b.index < right.index]
            if not inner:
                continue
            base = (
                max(inner, key=lambda p: p.price)
                if inverse
                else min(inner, key=lambda p: p.price)
            )
            if not g.strictly_ordered([left, base, right]):
                continue

            if inverse:
                rim_ref = min(left.price, right.price)
                depth = base.price - rim_ref
            else:
                rim_ref = max(left.price, right.price)
                depth = rim_ref - base.price
            if depth <= 0 or rim_ref <= 0:
                continue
            depth_pct = depth / rim_ref
            if not (min_depth <= depth_pct <= max_depth):
                continue

            # 4 — ROUNDNESS.
            band = round_band * depth
            if inverse:
                base_bars = int(
                    (highs[left.index : right.index + 1] >= base.price - band).sum()
                )
            else:
                base_bars = int(
                    (lows[left.index : right.index + 1] <= base.price + band).sum()
                )
            if base_bars < min_base_bars:
                continue

            # 5 — bottom centrality.
            if not (
                left.index + 0.25 * width <= base.index <= left.index + 0.75 * width
            ):
                continue

            # 6 — the handle.
            h_lows = lows[right.index + 1 :]
            h_highs = highs[right.index + 1 :]
            if len(h_lows) == 0:
                continue
            if inverse:
                handle_extreme = float(h_highs.max())
                if handle_extreme > right.price + handle_max_retrace * depth:
                    continue
                if handle_extreme >= base.price - depth / 2:
                    continue
                if float(h_lows.min()) < right.price:
                    continue
            else:
                handle_extreme = float(h_lows.min())
                if handle_extreme < right.price - handle_max_retrace * depth:
                    continue
                if handle_extreme <= base.price + depth / 2:
                    continue
                if float(h_highs.max()) > right.price:
                    continue

            level = float(rim_ref)
            if not g.level_unbroken(df, level, direction, right.index):
                continue

            out.append(
                DetectedEvent(
                    kind=kind,
                    direction=direction,
                    level=level,
                    target_height=float(depth),
                    start_ts=int(left.ts),
                    end_ts=int(df.index[n - 1]),
                    meta={
                        "depth_pct": float(depth_pct),
                        "base_bars": float(base_bars),
                        "handle_bars": float(handle_bars),
                        "rim_asymmetry": float(
                            abs(right.price - left.price)
                            / max(right.price, left.price)
                        ),
                    },
                )
            )

    return g.dedupe_events(out)
