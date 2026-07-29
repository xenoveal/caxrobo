"""
Catalog family 1 (Reversal Patterns) as registered Detector plug-ins:
`head-and-shoulders`, `inverse-head-and-shoulders`, `double-top`, `double-bottom`.

FOUR of the catalog's sixteen reversal rows. What is NOT here, and why, is
enumerated in `plugins/detectors/catalog.py` — Triple Top/Bottom are `deferred`
(a one-line generalisation of `_detect_double`, held back so this phase's
degrees of freedom stay countable) and the remaining ten are `out-of-scope`
with a stated reason each.

ATTRIBUTION (contract §0b — port, do not re-derive):
  * H&S geometry is `src/trading_bot/signals/patterns.py:191-257`, reproduced
    bit-identically at ParamSpec defaults and then EXTENDED with three
    refinements that each default to the legacy behaviour. The parity is the
    point: `plugins/detectors/legacy_patterns.py` wraps the same geometry and
    the two must agree, so what is under test here is the REFINEMENTS, not the
    base shape.
  * Double top/bottom is `scripts/bruteforce/indicators.py:519` / `:500`. The
    donor returns a boolean `pd.Series` with NO level and NO target, so the
    neckline and the measured move are genuinely new work here, as is the
    minimum-trough-depth test the donor lacks.

THE DONOR'S LOOKAHEAD TRAP, avoided deliberately. `bruteforce.pivot_high`
(indicators.py:370) flags a pivot at bar `t+span` carrying the price from `t`,
because its consumers index a Series. Production `find_pivots` returns
`index = t` and relies on the caller passing a frame sliced to the evaluation
bar (pivots.py:9-12). Reading `find_pivots`' `index` as a confirmation bar
creates a `span`-bar lookahead that *improves* every backtest and therefore
does not look like a bug. Every pivot-terminated detector in this module takes
`end_ts` from `_geometry.confirmation_ts` — except the H&S pair, which keeps
`window[-1].ts` for legacy parity (see `_detect_hs`).
"""

import logging

from trading_bot import config
from trading_bot.framework.contracts import DetectedEvent, ParamSpec
from trading_bot.framework.registry import register
from trading_bot.plugins.detectors import _geometry as g
from trading_bot.signals.pivots import find_pivots

logger = logging.getLogger("trading_bot")

SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME

# Shared ParamSpecs. Written as a factory rather than a module constant so each
# detector owns its own dict and a future per-detector bound change cannot leak
# sideways into another registration.
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
# Head & shoulders — contract §9 tier 1, refined.
# --------------------------------------------------------------------------- #

_HS_PARAMS = {
    **_scan_params(),
    "shoulder_tol": ParamSpec(
        kind="float",
        default=config.HS_SHOULDER_TOLERANCE,
        bounds=(0.0, 0.20),
        doc="Max relative height difference between the two shoulders",
    ),
    "head_prominence": ParamSpec(
        kind="float",
        default=config.HS_HEAD_MIN_PROMINENCE,
        bounds=(0.0, 0.20),
        doc="Head must exceed both shoulders by at least this fraction",
    ),
    "neckline_sloped": ParamSpec(
        kind="bool",
        default=config.HS_NECKLINE_SLOPED,
        doc="Fit the neckline through both troughs instead of taking the stricter one",
    ),
    "time_symmetry_tol": ParamSpec(
        kind="float",
        default=config.HS_TIME_SYMMETRY_TOL,
        bounds=(0.0, 1.0),
        doc="Max |(head-left)-(right-head)| / (right-left); 1.0 disables the test",
    ),
    "volume_taper": ParamSpec(
        kind="bool",
        default=config.HS_VOLUME_TAPER_REQUIRED,
        doc="Require the right shoulder's bar volume below the left shoulder's",
    ),
}

_HS_RATIONALE = (
    "Contract §9 tier 1 by reliability (the catalog's own table gives Head & "
    "Shoulders five stars). The v0.2.0 geometry measured as a loser under the "
    "old absolute-percentage risk model and was retired from dispatch "
    "(KNOWN-LIMITATIONS §8); it is re-offered here under the ATR risk model "
    "with sloping-neckline, time-symmetry and volume-taper options, all "
    "defaulting OFF. So what is being tested is the REFINEMENTS, not the base "
    "shape, and at defaults this detector reproduces signals/patterns.py "
    "bit-for-bit — pinned by TestHeadAndShouldersParity."
)


@register(
    "detector",
    name="head-and-shoulders",
    params=_HS_PARAMS,
    rationale=_HS_RATIONALE,
    timeframes=(SETUP_TF,),
    tier=1,
)
def head_and_shoulders(ctx, **params) -> list[DetectedEvent]:
    """Bearish H&S: high / low / HIGHER high / low / high, breaking a neckline down."""
    return _detect_hs(ctx, inverse=False, **params)


@register(
    "detector",
    name="inverse-head-and-shoulders",
    params=_HS_PARAMS,
    rationale=_HS_RATIONALE,
    timeframes=(SETUP_TF,),
    tier=1,
)
def inverse_head_and_shoulders(ctx, **params) -> list[DetectedEvent]:
    """Bullish inverse H&S: low / high / LOWER low / high / low, breaking up."""
    return _detect_hs(ctx, inverse=True, **params)


def _detect_hs(
    ctx,
    *,
    inverse: bool,
    pivot_span: int,
    max_age_bars: int,
    lookback_bars: int,
    shoulder_tol: float,
    head_prominence: float,
    neckline_sloped: bool,
    time_symmetry_tol: float,
    volume_taper: bool,
) -> list[DetectedEvent]:
    """H&S / inverse H&S over alternating pivot 5-tuples.

    Test order is `patterns.py:191-257`'s, deliberately unchanged: collapse
    same-kind runs first so noise pivots do not hide the geometry, require the
    window to be strictly ordered in time, then prominence / symmetry /
    neckline, then freshness and the unbroken-level rule LAST.

    end_ts KEEPS `window[-1].ts` rather than using
    `_geometry.confirmation_ts`. That is a knowing exception to this phase's own
    convention: `legacy_patterns.py` wraps the same geometry and bit-exact
    parity with it is load-bearing for Phase 3's frozen gate. The *newer*
    convention is stricter (a pivot is knowable `span` bars later), and
    reconciling `legacy_patterns.py` is recorded as a follow-up in the phase
    report. Do not unify it here — that would move a frozen number.

    THE BEARISH WINDOW IS (high, low, high, low, high). Get the order wrong and
    the detector silently finds nothing, which reads like a tolerance problem.
    """
    df = ctx.window(SETUP_TF, lookback_bars)
    n = len(df)
    out: list[DetectedEvent] = []
    if n == 0:
        return out

    fresh = g.fresh_factory(n, max_age_bars)
    seq = g.collapse_runs(find_pivots(df, span=pivot_span))
    volumes = df["volume"].to_numpy()
    want = (
        ("low", "high", "low", "high", "low")
        if inverse
        else ("high", "low", "high", "low", "high")
    )
    kind = "inverse-head-and-shoulders" if inverse else "head-and-shoulders"
    direction = "long" if inverse else "short"

    for i in range(len(seq) - 4):
        window = seq[i : i + 5]
        if tuple(p.kind for p in window) != want:
            continue
        if not g.strictly_ordered(window):
            continue
        left, tr1, head_p, tr2, right = window
        last_index = window[-1].index

        # --- neckline ---------------------------------------------------- #
        # Legacy default: the STRICTER of the two troughs, a horizontal line —
        # patterns.py:209/234's CONSERVATIVE_LEVEL_CHOICE. Refinement: the line
        # through both troughs, which is what a classical neckline actually is;
        # the horizontal min()/max() is its conservative special case.
        if neckline_sloped:
            def neck_at(x: float) -> float:
                return g.line_value(tr1.index, tr1.price, tr2.index, tr2.price, x)
        else:
            flat = min(tr1.price, tr2.price) if not inverse else max(tr1.price, tr2.price)

            def neck_at(x: float) -> float:
                return flat

        neckline = neck_at(last_index)
        # A sloped neckline extrapolated to last_index can land outside any
        # sane price range when the troughs are far apart and steeply offset.
        # Clamp NOTHING: reject instead. The shape-defining inequalities below
        # (head beyond the neckline, both shoulders on the correct side of it,
        # each evaluated at ITS OWN index) are what make an extrapolated level
        # sane; positivity is the one extra guard they cannot express.
        if not (neckline > 0):
            continue

        # --- time symmetry (refinement 2; 1.0 is a no-op) ----------------- #
        span_total = right.index - left.index
        left_leg = head_p.index - left.index
        right_leg = right.index - head_p.index
        asymmetry = (
            abs(left_leg - right_leg) / span_total if span_total > 0 else 1.0
        )
        if asymmetry > time_symmetry_tol:
            continue

        # --- volume taper (refinement 3) --------------------------------- #
        # ALWAYS recorded in meta, whether or not it gates: a measurement that
        # only exists when it is switched on cannot be used to decide whether
        # switching it on is worthwhile.
        vol_left = float(volumes[left.index])
        vol_right = float(volumes[right.index])
        taper_ratio = (vol_right / vol_left) if vol_left > 0 else float("nan")
        if volume_taper and not (vol_right < vol_left):
            continue

        if inverse:
            t1, p1, head, p2, t3 = (p.price for p in window)
            if not (
                head < t1 * (1 - head_prominence)
                and head < t3 * (1 - head_prominence)
                and abs(t1 - t3) / max(t1, t3) <= shoulder_tol
                and head < neckline
                # Both shoulders sit below the neckline they break up through,
                # each tested against the neckline AT ITS OWN BAR so a sloped
                # line is evaluated where the shoulder actually is.
                and t1 < neck_at(left.index)
                and t3 < neck_at(right.index)
                and fresh(last_index)
                and g.level_unbroken(df, neckline, "long", last_index)
            ):
                continue
            target_height = neckline - head
        else:
            p1, t1, head, t2, p3 = (p.price for p in window)
            if not (
                head > p1 * (1 + head_prominence)
                and head > p3 * (1 + head_prominence)
                and abs(p1 - p3) / max(p1, p3) <= shoulder_tol
                and head > neckline
                # Both shoulders stand above the neckline they break down
                # through — see the inverse branch's note on the sloped case.
                and p1 > neck_at(left.index)
                and p3 > neck_at(right.index)
                and fresh(last_index)
                and g.level_unbroken(df, neckline, "short", last_index)
            ):
                continue
            target_height = head - neckline

        out.append(
            DetectedEvent(
                kind=kind,
                direction=direction,
                level=float(neckline),
                target_height=float(target_height),
                start_ts=int(window[0].ts),
                end_ts=int(window[-1].ts),
                meta={
                    "head": float(head_p.price),
                    "left_shoulder": float(left.price),
                    "right_shoulder": float(right.price),
                    # <price_key>_ts siblings, epoch-ms as float (Mapping[str,
                    # float] takes it with no contract change): WS-B's replay
                    # chart pairs a price vertex with its x-axis position by
                    # this exact suffix, generically across detectors.
                    "head_ts": float(head_p.ts),
                    "left_shoulder_ts": float(left.ts),
                    "right_shoulder_ts": float(right.ts),
                    "time_asymmetry": float(asymmetry),
                    "volume_taper_ratio": float(taper_ratio),
                    "neckline_sloped": 1.0 if neckline_sloped else 0.0,
                },
            )
        )

    return g.dedupe_events(out)


# --------------------------------------------------------------------------- #
# Double top / double bottom — contract §9 tier 2.
# --------------------------------------------------------------------------- #

_DOUBLE_PARAMS = {
    **_scan_params(),
    "tolerance": ParamSpec(
        kind="float",
        default=config.DOUBLE_TOLERANCE,
        bounds=(0.0, 0.10),
        doc="Max relative height difference between the two extremes",
    ),
    "max_gap_bars": ParamSpec(
        kind="int",
        default=config.DOUBLE_MAX_GAP_BARS,
        bounds=(5, 200),
        doc="Max SETUP-TIER bars between the two extremes (60 is 10 days at 4h)",
    ),
    "min_separation_bars": ParamSpec(
        kind="int",
        default=config.DOUBLE_MIN_SEPARATION_BARS,
        bounds=(2, 60),
        doc="Min setup-tier bars between the two extremes",
    ),
    "min_trough_depth": ParamSpec(
        kind="float",
        default=config.DOUBLE_MIN_TROUGH_DEPTH,
        bounds=(0.0, 0.30),
        doc="Intervening dip/bump, as a fraction of the mean extreme",
    ),
    "max_trough_depth": ParamSpec(
        kind="float",
        default=config.DOUBLE_MAX_TROUGH_DEPTH,
        bounds=(0.05, 0.35),
        doc="Sibling ceiling to min_trough_depth: above this fraction the "
        "intervening swing is a trend leg (a range), not a bump — a W's "
        "middle bounce, not a fresh impulse. CORRECTNESS GATE, not a free "
        "tuning axis: the ceiling is capped at 0.35, BELOW the D3 evidence "
        "fixture's measured 0.399, so no value the mutator (param_jitter.py) "
        "can draw fully reopens that defect — see graph.py:141-148 on why a "
        "gate's bound is its safety envelope, not a suggestion",
    ),
    "dominance_tol": ParamSpec(
        kind="float",
        default=config.DOUBLE_DOMINANCE_TOL,
        bounds=(0.0, 0.005),
        doc="Fraction by which a bar's high/low between the two extremes may "
        "exceed either extreme before the pair is rejected as not dominant — "
        "checked on candle high/low, not pivots, so an unconfirmed spike that "
        "never became a fractal pivot still disqualifies the pair. This is "
        "float-noise slack, not a tuning axis: the ceiling is capped at 0.5% "
        "so the mutator cannot widen it into a real intervening-spike "
        "tolerance",
    ),
    "prior_trend_lookback_bars": ParamSpec(
        kind="int",
        default=config.DOUBLE_PRIOR_TREND_LOOKBACK_BARS,
        bounds=(5, 200),
        doc="Setup bars looked back from the FIRST extreme to measure the "
        "move the pattern claims to reverse. Floored at 5 (not 0): a pair "
        "whose first extreme sits closer to the start of the frame than "
        "this many bars is rejected outright rather than measured over a "
        "silently shortened window (insufficient history is not 'no prior "
        "trend'). CORRECTNESS GATE: the floor is deliberately kept away from "
        "0, which would let the mutator disable D1 outright — see "
        "graph.py:141-148",
    ),
    "prior_trend_min_move": ParamSpec(
        kind="float",
        default=config.DOUBLE_PRIOR_TREND_MIN_MOVE,
        bounds=(0.01, 0.50),
        doc="Min fractional decline (bottom) / advance (top) into the first "
        "extreme over prior_trend_lookback_bars — a reversal must reverse "
        "something; a shape with no antecedent move is a range, not a W/M. "
        "Floored at 0.01, not 0.0: a zero floor would let the mutator "
        "disable D1 independently of prior_trend_lookback_bars",
    ),
}

_DOUBLE_RATIONALE_TAIL = (
    "Contract §9 tier 2 (the catalog rates Double Top/Bottom four stars). "
    "Ported from scripts/bruteforce/indicators.py:519 per §0b; the donor "
    "returned a boolean Series with no level and no target, so the neckline "
    "(the intervening swing that must break for the reversal to be more than a "
    "wiggle) and the measured move are added here, together with the "
    "minimum-trough-depth test the donor lacks — without which two highs "
    "within tolerance and a negligible dip between them qualify as an M when "
    "they are a flat range. v0.3.2 (code review 2026-07-29) added three more "
    "gates the donor also lacks: a candle-range dominance guard (no bar between "
    "the extremes may exceed them), a max-trough-depth ceiling (the sibling of "
    "min-trough-depth), and a prior-trend requirement (the pattern must reverse "
    "a real antecedent move, not punctuate an ongoing one)."
)


@register(
    "detector",
    name="double-top",
    params=_DOUBLE_PARAMS,
    rationale="Two comparable highs separated by a real trough. " + _DOUBLE_RATIONALE_TAIL,
    timeframes=(SETUP_TF,),
    tier=2,
)
def double_top(ctx, **params) -> list[DetectedEvent]:
    """Bearish M: two comparable pivot highs, short on the intervening trough."""
    return _detect_double(ctx, inverse=False, **params)


@register(
    "detector",
    name="double-bottom",
    params=_DOUBLE_PARAMS,
    rationale="Two comparable lows separated by a real bump. " + _DOUBLE_RATIONALE_TAIL,
    timeframes=(SETUP_TF,),
    tier=2,
)
def double_bottom(ctx, **params) -> list[DetectedEvent]:
    """Bullish W: two comparable pivot lows, long on the intervening peak."""
    return _detect_double(ctx, inverse=True, **params)


def _detect_double(
    ctx,
    *,
    inverse: bool,
    pivot_span: int,
    max_age_bars: int,
    lookback_bars: int,
    tolerance: float,
    max_gap_bars: int,
    min_separation_bars: int,
    min_trough_depth: float,
    max_trough_depth: float,
    dominance_tol: float,
    prior_trend_lookback_bars: int,
    prior_trend_min_move: float,
) -> list[DetectedEvent]:
    """Double top / double bottom with the level-and-target adapter the donor lacks.

    EVERY ORDERED PAIR of same-kind pivots is considered, not just consecutive
    ones (`bruteforce:527`'s `zip(idx, idx[1:])`). A noise pivot high between
    two real tops is exactly what `collapse_runs` absorbs when it is ADJACENT,
    but an alternating micro-low/micro-high sequence survives collapsing, and
    consecutive-only pairing is why the donor misses most real double tops.

    The tolerance is relative to `max(a, b)`, following `patterns.py:213`. The
    donor divides by `vals[a]` (`bruteforce:528`), which is asymmetric in pair
    order — the same two peaks pass or fail depending on which is called first.

    THREE CORRECTNESS GATES beyond the donor's own tests (v0.3.2 code review):

      * DOMINANCE. A double bottom's two lows must be the two lowest points in
        the window — no bar between them may trade lower than either (within
        `dominance_tol`). Checked against `df["low"]`/`df["high"]` OVER THE BAR
        RANGE, not the pivot list: an intervening spike that never confirmed as
        a fractal pivot (too shallow, wrong side of `pivot_span`) still means a
        third, more extreme point exists, and a pivot-only check is blind to it.
      * MAX TROUGH DEPTH. `min_trough_depth`'s sibling ceiling: a bump/dip that
        retraces most of the move between the two extremes is a trend leg, not
        a W/M's modest middle bounce.
      * PRIOR TREND. A reversal must reverse something. The first extreme `a`
        must sit at the bottom (top) of a measurable decline (advance) over
        `prior_trend_lookback_bars` — otherwise "two similar lows with a bump"
        is just a consolidation inside an ongoing move, not a reversal.
    """
    df = ctx.window(SETUP_TF, lookback_bars)
    n = len(df)
    out: list[DetectedEvent] = []
    if n == 0:
        return out

    fresh = g.fresh_factory(n, max_age_bars)
    pivots = g.collapse_runs(find_pivots(df, span=pivot_span))
    same_kind = "low" if inverse else "high"
    other_kind = "high" if inverse else "low"
    kind = "double-bottom" if inverse else "double-top"
    direction = "long" if inverse else "short"

    # Candle range, not closes and not pivots: the dominance guard and the
    # prior-trend reference both need the actual intrabar extreme, which is
    # where a spike that never confirmed as a pivot would show up.
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()

    extremes = [p for p in pivots if p.kind == same_kind]
    others = [p for p in pivots if p.kind == other_kind]

    for ai in range(len(extremes)):
        a = extremes[ai]
        for bi in range(ai + 1, len(extremes)):
            b = extremes[bi]
            gap = b.index - a.index
            if gap < min_separation_bars:
                continue
            if gap > max_gap_bars:
                break  # extremes are index-ordered: everything later is farther
            if abs(b.price - a.price) / max(a.price, b.price) > tolerance:
                continue

            # --- D2: dominance guard ------------------------------------- #
            # No bar strictly between the two extremes may trade beyond
            # either of them by more than `dominance_tol` — an unconfirmed
            # spike still means a and b were never the pattern's real
            # extremes.
            if inverse:
                floor = min(a.price, b.price) * (1 - dominance_tol)
                if lows[a.index : b.index + 1].min() < floor:
                    continue
            else:
                ceiling = max(a.price, b.price) * (1 + dominance_tol)
                if highs[a.index : b.index + 1].max() > ceiling:
                    continue

            # THE ADAPTER. The intervening opposite-kind pivot is the level:
            # two peaks with no trough between them are one peak, not an M.
            between = [m for m in others if a.index < m.index < b.index]
            if not between:
                continue
            m = (
                max(between, key=lambda p: p.price)
                if inverse
                else min(between, key=lambda p: p.price)
            )

            extreme_mean = (a.price + b.price) / 2.0
            depth = abs(extreme_mean - m.price)
            depth_frac = depth / extreme_mean
            if depth_frac < min_trough_depth:
                continue
            # --- D3: maximum trough depth --------------------------------- #
            if depth_frac > max_trough_depth:
                continue

            # --- D1: prior-trend requirement ------------------------------ #
            # A double bottom must reverse a real decline into `a`; a double
            # top must reverse a real advance. Measured from the OPPOSITE
            # candle extreme (high for a bottom's reference, low for a top's)
            # so the reference is a price that genuinely traded, not a close
            # that could understate the move.
            #
            # FAIL CLOSED on insufficient history. If `a` is closer to the
            # start of the frame than `prior_trend_lookback_bars`, there is
            # not enough history to judge whether a prior trend existed at
            # all — "trend unknown" must never read as "trend confirmed", so
            # the pair is rejected outright rather than measured over a
            # silently shortened window. (Matches WS-C's trend-context
            # Confirmation, which fails closed on the same condition.)
            if a.index < prior_trend_lookback_bars:
                continue
            ref_index = a.index - prior_trend_lookback_bars
            # ANCHOR NEIGHBOURHOOD, not the whole window. The reference point
            # is fixed `prior_trend_lookback_bars` back from `a` — NOT the
            # max/min over the whole a-ward window, which would just re-find
            # the local dip/bump and pass even when the broader trend never
            # reversed (the straight-line-rally-then-consolidation defect
            # this gate exists to catch). But reading `ref_index`'s single bar
            # is fragile the other way: if that one bar happens to be a spike
            # the decline reads as bigger than it is, and if it is unusually
            # quiet a genuine reversal can be rejected. Taking the extreme
            # over a SMALL neighbourhood of just that anchor (`pivot_span`
            # bars either side, the same "one bar shouldn't decide a
            # reading" scale `_geometry.fit_converging_lines` uses for its
            # pivot fit) fixes the single-bar fragility without reopening the
            # window-wide loophole: the neighbourhood is centered on the
            # fixed anchor and cannot drift toward `a` to re-find the dip.
            nb_lo = max(0, ref_index - pivot_span)
            nb_hi = min(n - 1, ref_index + pivot_span)
            if inverse:
                ref_price = float(highs[nb_lo : nb_hi + 1].max())
                prior_move = (ref_price - a.price) / ref_price if ref_price > 0 else 0.0
            else:
                ref_price = float(lows[nb_lo : nb_hi + 1].min())
                prior_move = (a.price - ref_price) / ref_price if ref_price > 0 else 0.0
            if prior_move < prior_trend_min_move:
                continue

            level = float(m.price)
            if not fresh(b.index):
                continue
            if not g.level_unbroken(df, level, direction, b.index):
                continue

            out.append(
                DetectedEvent(
                    kind=kind,
                    direction=direction,
                    level=level,
                    target_height=float(depth),
                    start_ts=int(a.ts),
                    end_ts=g.confirmation_ts(df, b.index, pivot_span),
                    meta={
                        "peak_a": float(a.price),
                        "peak_b": float(b.price),
                        "trough": level,
                        # <price_key>_ts siblings — see _detect_hs's identical
                        # note. Additive only: no gate, level, target or
                        # end_ts reads these: they exist for WS-B's replay
                        # chart to place each vertex on the x-axis.
                        "peak_a_ts": float(a.ts),
                        "peak_b_ts": float(b.ts),
                        "trough_ts": float(m.ts),
                        "separation_bars": float(gap),
                    },
                )
            )

    return g.dedupe_events(out)
