"""
Shared private geometry helpers for the Phase 8 detector plug-ins.

ATTRIBUTION AND SCOPE. `collapse_runs`, `level_unbroken`, `strictly_ordered`
and `line_value` are ported VERBATIM (docstrings included) from
`src/trading_bot/signals/patterns.py:144 / 167 / 181 / 260`;
`fit_converging_lines` is that module's `_detect_triangles` (patterns.py:272-370)
split into a pure line fit, with the direction/emission decisions removed and
one guard deliberately dropped. `dedupe_events` is `patterns._dedupe`
(patterns.py:125-141) restated on the contract's field names.

TWO GEOMETRY IMPLEMENTATIONS COEXIST, ON PURPOSE — do not "clean this up".
The v0.3.0 shared architecture contract §9 records the decision:

  * `signals/patterns.py`, reached through
    `plugins/detectors/legacy_patterns.py`, exists to reproduce v0.2.0
    EXACTLY. Its behaviour is frozen by the Phase 3 parity test.
  * this module exists to be CORRECT.

They are allowed to disagree. Converging them is only safe once the parity test
is retired. That is why nothing here edits `signals/patterns.py`.

THE ONE DROPPED GUARD, and why relocating its work is mandatory.
`signals/patterns.py:327` reads:

    if upper_end > upper_start or lower_end < lower_start:
        return out  # a side is expanding: not a triangle

A rising wedge has BOTH trendlines rising and a falling wedge has BOTH falling,
so that line makes two of contract §9's tier-2 patterns structurally
UNREACHABLE — no parameter setting can produce one. `fit_converging_lines`
below omits it. Omitting it WITHOUT relocating its work would relabel every
wedge a "symmetrical triangle", so the guard's real job — deciding what shape
the two slopes describe — moves into
`plugins/detectors/continuation.py::_classify`, which is guarded by
`test_shape_classification_is_mutually_exclusive`.

NAMING. Underscore-prefixed module: `framework.registry.load_all()` skips
modules whose leaf name starts with "_" (registry.py:282), so this file
registers nothing and adds nothing to REGISTRY even though it lives inside the
walked package.
"""

from dataclasses import dataclass

import pandas as pd

from trading_bot.framework.contracts import DetectedEvent
from trading_bot.signals.pivots import Pivot


# --------------------------------------------------------------------------- #
# Ported verbatim from signals/patterns.py — see the module docstring.
# --------------------------------------------------------------------------- #


def collapse_runs(pivots: list[Pivot]) -> list[Pivot]:
    """Reduce each maximal run of same-kind pivots to its extreme.

    Fractal pivots do not strictly alternate — a noise pivot high between a
    shoulder and its trough is common. Scanning the raw list for adjacent
    alternating 5-tuples therefore skips most real structure, so each run of
    consecutive same-kind pivots collapses to its highest high / lowest low.
    """
    out: list[Pivot] = []
    i = 0
    while i < len(pivots):
        j = i
        while j + 1 < len(pivots) and pivots[j + 1].kind == pivots[i].kind:
            j += 1
        run = pivots[i : j + 1]
        if pivots[i].kind == "high":
            out.append(max(run, key=lambda p: p.price))
        else:
            out.append(min(run, key=lambda p: p.price))
        i = j + 1
    return out


def level_unbroken(
    df: pd.DataFrame, level: float, direction: str, after_index: int
) -> bool:
    """True if no bar after `after_index` has closed through `level`.

    A pattern whose level was already breached is spent: the move it predicted
    has begun, so a later crossing is a retest re-break, not the breakout.
    """
    closes = df["close"].to_numpy()[after_index + 1 :]
    if len(closes) == 0:
        return True
    if direction == "long":
        return bool((closes <= level).all())
    return bool((closes >= level).all())


def strictly_ordered(window: list[Pivot]) -> bool:
    """True if every pivot in the window sits on a strictly later bar.

    An outside bar can be both a pivot high and a pivot low, which would
    otherwise let a shoulder and its adjacent trough be the SAME bar — a
    time-degenerate shape that is not a pattern.
    """
    return all(window[k + 1].index > window[k].index for k in range(len(window) - 1))


def line_value(x1: int, y1: float, x2: int, y2: float, x: float) -> float:
    """Value at x of the line through (x1, y1) and (x2, y2).

    Raises:
        ValueError: if x1 == x2 — one point determines no line.
    """
    if x2 == x1:
        raise ValueError(f"line_value needs distinct x, got x1 == x2 == {x1}")
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


# --------------------------------------------------------------------------- #
# The shared discipline every Phase 8 detector ends with / opts out of.
# --------------------------------------------------------------------------- #


def dedupe_events(events: list[DetectedEvent]) -> list[DetectedEvent]:
    """Keep one event per (kind, direction): freshest, then largest target.

    `patterns._dedupe` (patterns.py:125-141) on the contract's field names
    (`level`, `target_height`), with its reasoning unchanged: overlapping
    windows of the same pattern type (several 5-pivot H&S windows sharing a
    head, say) describe one setup, not several. Emitting them all raises
    duplicate signals for the same symbol on the same bar.

    Two events with the same `kind` and DIFFERENT `direction` both survive, and
    that is deliberate — the symmetrical triangle emits a long and a short
    candidate and lets the trigger bar pick at most one, which is precisely why
    the key is a pair.

    Returns the survivors sorted by (kind, direction), matching
    `detect_patterns`' documented ordering so a caller's iteration order is
    stable across runs.
    """
    best: dict[tuple[str, str], DetectedEvent] = {}
    for e in events:
        key = (e.kind, e.direction)
        incumbent = best.get(key)
        if incumbent is None or (e.end_ts, e.target_height) > (
            incumbent.end_ts,
            incumbent.target_height,
        ):
            best[key] = e
    return sorted(best.values(), key=lambda e: (e.kind, e.direction))


def fresh_factory(n: int, max_age_bars: int):
    """The freshness closure, `patterns.py:113-114`.

    Geometry whose last confirming pivot is older than `max_age_bars` is stale
    and ignored, so signals never trigger off long-dead structure. Where a
    pattern has two independent sides (the two triangle trendlines) BOTH must
    be fresh, which is why callers pass `min(...)` of the two last indices.

    Args:
        n: Number of bars in the frame under evaluation.
        max_age_bars: Bars of slack allowed after the last confirming pivot.

    Returns:
        fresh(last_pivot_index) -> bool.
    """

    def fresh(last_pivot_index: int) -> bool:
        return (n - 1) - last_pivot_index <= max_age_bars

    return fresh


def confirmation_ts(df: pd.DataFrame, pivot_index: int, span: int) -> int:
    """Epoch-ms of the bar on which a pivot at `pivot_index` first becomes knowable.

    `signals.pivots.find_pivots` returns `index = t` for a pivot at bar t, but
    a fractal pivot is only knowable once `span` bars have CLOSED after it
    (pivots.py:9-12). Reading the pivot's own bar as the confirmation bar is a
    `span`-bar lookahead — and, being a lookahead, it *improves* every backtest,
    so it will not look like a bug. Every NEW pivot-terminated detector in
    Phase 8 therefore takes its `end_ts` from here.

    Clamped to the last bar of the frame: `EvalContext` already truncates at
    the last closed bar, so a pivot inside the final `span` bars cannot be
    returned by `find_pivots` at all, and the clamp is a belt-and-braces guard
    against a caller passing an unsliced frame rather than an expected path.

    NOT used by the H&S pair: those keep `window[-1].ts` for bit-exact parity
    with `signals/patterns.py`, which `legacy_patterns.py` also wraps. That
    exception is recorded in the phase report, and reconciling it is a
    follow-up, not something to unify here.
    """
    return int(df.index[min(len(df) - 1, pivot_index + span)])


# --------------------------------------------------------------------------- #
# The relaxed converging-line fit.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LineFit:
    """Two endpoint-fitted trendlines and the measurements a classifier needs.

    Pure geometry: it decides NOTHING about direction and emits no event. That
    separation is what lets five differently-signed shapes share one fit and
    one containment test.

    Attributes:
        start_x / end_x: Positional bar indices the fit spans.
        upper_start / upper_end: Upper line's value at start_x / end_x.
        lower_start / lower_end: Lower line's value at start_x / end_x.
        upper_slope / lower_slope: NORMALISED per-bar slopes,
            (y2 - y1) / ((x2 - x1) * y1) — the scale-free device of
            `scripts/bruteforce/indicators.py:100-105`, so one tolerance is
            meaningful for a symbol at $0.10 and one at $100k. A raw slope
            tolerance is not comparable across a universe.
        start_range / end_range: Vertical extent between the lines at each end.
            `start_range` is the classical measured move for these shapes.
    """

    start_x: int
    end_x: int
    upper_start: float
    upper_end: float
    lower_start: float
    lower_end: float
    upper_slope: float
    lower_slope: float
    start_range: float
    end_range: float


def fit_converging_lines(
    df: pd.DataFrame,
    pivots: list[Pivot],
    *,
    min_pivots_per_side: int,
    min_width_bars: int,
    max_width_bars: int,
    min_convergence: float,
    containment_tol: float,
    fresh,
) -> LineFit | None:
    """Fit converging upper/lower trendlines through recent pivots.

    The fit is deliberately simple and auditable, exactly as
    `patterns._detect_triangles` describes it: the upper line passes through the
    first and last pivot highs, the lower line through the first and last pivot
    lows, both taken from the last `max_width_bars` bars only.

    A fit must clear five independent tests, because convergence alone is
    satisfied by almost any pair of lines fitted over a long enough window:

      1. Width — the fitted structure spans at least `min_width_bars` and
         (by construction) at most `max_width_bars` bars.
      2. Freshness — BOTH lines are anchored on recent structure. `min()` of
         the two last pivot indices, never `max()`: with `max()` one line could
         be fitted to pivots long out of date and still set a level.
      3. Convergence — the vertical range at the latest bar has contracted by
         at least `min_convergence`, without the lines crossing.
      4. Containment — every bar the lines span stays inside them within
         `containment_tol`. Without this, two lines that merely converge
         qualify even when price spent the window well outside them.
      5. Price still inside at `end_x` — price already outside means the
         structure is broken, not pending.

    NOT tested, and this is the one deliberate difference from
    `patterns._detect_triangles`: the expanding-side guard at patterns.py:327.
    See the module docstring. Its work is relocated to `_classify`.

    Args:
        df: Setup-timeframe OHLCV frame, closed bars only.
        pivots: Confirmed pivots over `df`.
        min_pivots_per_side: Pivot highs and pivot lows required. Floored at 2
            — a line needs two points and no config knob can lower that.
        min_width_bars / max_width_bars: Bounds on the fitted structure's width.
        min_convergence: Required fractional contraction of the vertical range.
        containment_tol: Fraction of price a bar may pierce its own line by.
        fresh: `fresh_factory`'s closure.

    Returns:
        LineFit, or None when any test fails.
    """
    min_per_side = max(2, min_pivots_per_side)
    n = len(df)
    end_x = n - 1
    if end_x < min_width_bars:
        return None

    earliest = max(0, end_x - max_width_bars)
    ph = [p for p in pivots if p.kind == "high" and p.index >= earliest]
    pl = [p for p in pivots if p.kind == "low" and p.index >= earliest]
    if len(ph) < min_per_side or len(pl) < min_per_side:
        return None
    if ph[0].index == ph[-1].index or pl[0].index == pl[-1].index:
        return None

    # BOTH trendlines must be anchored on recent structure: with max() here, one
    # line could be fitted to pivots long out of date and still set a level.
    if not fresh(min(ph[-1].index, pl[-1].index)):
        return None

    start_x = min(ph[0].index, pl[0].index)
    if end_x - start_x < min_width_bars:
        return None

    def upper(x: float) -> float:
        return line_value(ph[0].index, ph[0].price, ph[-1].index, ph[-1].price, x)

    def lower(x: float) -> float:
        return line_value(pl[0].index, pl[0].price, pl[-1].index, pl[-1].price, x)

    upper_start, upper_end = upper(start_x), upper(end_x)
    lower_start, lower_end = lower(start_x), lower(end_x)

    start_range = upper_start - lower_start
    end_range = upper_end - lower_end
    if start_range <= 0 or end_range <= 0:
        return None  # degenerate or already-crossed lines
    if end_range > start_range * (1 - min_convergence):
        return None  # not converging enough

    if not _contains(df, upper, lower, start_x, end_x, containment_tol):
        return None

    last_close = float(df["close"].to_numpy()[end_x])
    if not (lower_end < last_close < upper_end):
        return None  # price already outside the structure: broken, not pending

    return LineFit(
        start_x=int(start_x),
        end_x=int(end_x),
        upper_start=float(upper_start),
        upper_end=float(upper_end),
        lower_start=float(lower_start),
        lower_end=float(lower_end),
        upper_slope=_norm_slope(
            ph[0].index, ph[0].price, ph[-1].index, ph[-1].price
        ),
        lower_slope=_norm_slope(
            pl[0].index, pl[0].price, pl[-1].index, pl[-1].price
        ),
        start_range=float(start_range),
        end_range=float(end_range),
    )


def _norm_slope(x1: int, y1: float, x2: int, y2: float) -> float:
    """Per-bar slope normalised by the level: (y2-y1) / ((x2-x1) * y1).

    Scale-free, so one tolerance is comparable across symbols priced from $0.10
    to $100k — the device of `scripts/bruteforce/indicators.py:100-105`. Guards
    a non-positive anchor price (impossible in stored OHLCV, but a fixture can
    produce one) by returning 0.0, i.e. "flat", which is the conservative
    reading: a shape is never promoted to a wedge on an undefined slope.
    """
    if x2 == x1 or not (y1 > 0):
        return 0.0
    return (y2 - y1) / ((x2 - x1) * y1)


def _contains(
    df: pd.DataFrame, upper, lower, start_x: int, end_x: int, tol: float
) -> bool:
    """True if every bar in [start_x, end_x] stays inside the trendlines.

    Verbatim in intent from `patterns._triangle_contains` (patterns.py:373-385),
    with the tolerance passed in rather than read from config so it can be a
    declared ParamSpec: without this test, two lines that merely converge
    qualify as a triangle even when price spent the window well outside them.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    for x in range(start_x, end_x + 1):
        if highs[x] > upper(x) * (1 + tol) or lows[x] < lower(x) * (1 - tol):
            return False
    return True
