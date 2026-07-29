"""
Geometric chart-pattern detection on setup-timeframe pivot structure.

Detectors are explicit, parameterized geometric rules over confirmed pivots
(no ML, no visual matching), per the PRD's anti-false-positive mitigation:

  - head-and-shoulders (short) / inverse head-and-shoulders (long):
    three alternating same-side pivots with a dominant head, shoulders within
    a symmetry tolerance, and a conservative horizontal neckline (the stricter
    of the two trough/peak levels).
  - triangle (long and short candidates): trendlines through the first/last
    pivot highs and first/last pivot lows must converge by a minimum fraction
    without either side expanding, must actually contain the bars they span,
    and must span a bounded width; breakout level is each line's value at the
    latest bar.
  - flag (long or short): an impulse pole followed by a shallow, contained
    consolidation; breakout level is the consolidation extreme in the pole
    direction.

Two rules apply across detectors:

  - Freshness — geometry whose last confirming pivot is older than
    PATTERN_MAX_AGE_BARS is stale and ignored, so signals never trigger off
    long-dead structure. Where a pattern has two independent sides (the two
    triangle trendlines), BOTH must be fresh.
  - Unbroken level — a pattern whose breakout level has already been closed
    through since the geometry completed is spent, and is not emitted. Without
    this, the trigger's fresh-crossing rule fires on the re-break after a
    retest, entering a move already underway while still projecting the original
    measured move.

Same-kind pivot runs are collapsed to their extreme before the H&S scan:
fractal pivots do not strictly alternate, and requiring adjacency in the raw
pivot list hides most real geometry.

Target heights are classical measured-move distances, consumed by setup.py for
TP computation.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from trading_bot import config
from trading_bot.signals.pivots import Pivot, find_pivots

logger = logging.getLogger("trading_bot")

PATTERN_KINDS = (
    "head-and-shoulders",
    "inverse-head-and-shoulders",
    "triangle",
    "flag",
)


@dataclass(frozen=True)
class PatternCandidate:
    """A completed pattern awaiting a breakout trigger.

    Attributes:
        kind: One of PATTERN_KINDS.
        direction: "long" or "short" — the side a breakout would trade.
        breakout_level: Price the trigger bar must close beyond.
        target_height: Measured-move distance (price units) for TP computation.
        start_ts: Epoch-ms of the first bar/pivot forming the pattern.
        end_ts: Epoch-ms of the last bar/pivot forming the pattern.
    """

    kind: str
    direction: str
    breakout_level: float
    target_height: float
    start_ts: int
    end_ts: int


def detect_patterns(
    df: pd.DataFrame,
    pivots: list[Pivot] | None = None,
    *,
    max_age_bars: int | None = None,
) -> list[PatternCandidate]:
    """
    Run all pattern detectors over an OHLCV DataFrame.

    Args:
        df: Setup-timeframe OHLCV DataFrame (columns open/high/low/close/volume, epoch-ms
            index, ascending). Callers should pass only closed bars, sliced to
            config.PATTERN_LOOKBACK_BARS.
        pivots: Pre-computed pivots (default: find_pivots(df)).
        max_age_bars: Freshness bound on the pattern's last pivot
            (default config.PATTERN_MAX_AGE_BARS). Flags are inherently fresh
            (their consolidation ends at the latest bar) and skip this check.

    Returns:
        List of PatternCandidate ordered by (kind, direction), at most one per
        (kind, direction) pair — overlapping geometry of the same type is
        deduplicated in favour of the freshest, largest-target candidate, so a
        single bar cannot raise the same setup several times.
    """
    if max_age_bars is None:
        max_age_bars = config.PATTERN_MAX_AGE_BARS
    if pivots is None:
        pivots = find_pivots(df)

    candidates: list[PatternCandidate] = []
    n = len(df)
    if n == 0:
        return candidates

    def fresh(last_pivot_index: int) -> bool:
        return (n - 1) - last_pivot_index <= max_age_bars

    candidates.extend(_detect_head_and_shoulders(df, pivots, fresh))
    candidates.extend(_detect_triangles(df, pivots, fresh))
    candidates.extend(_detect_flags(df))

    candidates = _dedupe(candidates)
    candidates.sort(key=lambda c: (c.kind, c.direction))
    return candidates


def _dedupe(candidates: list[PatternCandidate]) -> list[PatternCandidate]:
    """Keep one candidate per (kind, direction): freshest, then largest target.

    Overlapping windows of the same pattern type (several 5-pivot H&S windows
    sharing a head, say) describe one setup, not several. Emitting them all
    raises duplicate signals for the same symbol on the same bar.
    """
    best: dict[tuple[str, str], PatternCandidate] = {}
    for c in candidates:
        key = (c.kind, c.direction)
        incumbent = best.get(key)
        if incumbent is None or (c.end_ts, c.target_height) > (
            incumbent.end_ts,
            incumbent.target_height,
        ):
            best[key] = c
    return list(best.values())


def _collapse_runs(pivots: list[Pivot]) -> list[Pivot]:
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


def _level_unbroken(df: pd.DataFrame, level: float, direction: str, after_index: int) -> bool:
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


def _strictly_ordered(window: list[Pivot]) -> bool:
    """True if every pivot in the window sits on a strictly later bar.

    An outside bar can be both a pivot high and a pivot low, which would
    otherwise let a shoulder and its adjacent trough be the SAME bar — a
    time-degenerate shape that is not a pattern.
    """
    return all(window[k + 1].index > window[k].index for k in range(len(window) - 1))


def _detect_head_and_shoulders(df, pivots, fresh) -> list[PatternCandidate]:
    """H&S (short) and inverse H&S (long) over alternating pivot 5-tuples."""
    out: list[PatternCandidate] = []
    shoulder_tol = config.HS_SHOULDER_TOLERANCE
    head_prom = config.HS_HEAD_MIN_PROMINENCE

    # Collapse same-kind runs first so noise pivots don't hide the geometry.
    seq = _collapse_runs(pivots)

    for i in range(len(seq) - 4):
        window = seq[i : i + 5]
        kinds = tuple(p.kind for p in window)
        if not _strictly_ordered(window):
            continue
        last_index = window[-1].index

        if kinds == ("high", "low", "high", "low", "high"):
            p1, t1, head, t2, p3 = (p.price for p in window)
            neckline = min(t1, t2)  # stricter (lower) trough: conservative trigger
            if (
                head > p1 * (1 + head_prom)
                and head > p3 * (1 + head_prom)
                and abs(p1 - p3) / max(p1, p3) <= shoulder_tol
                and head > neckline
                # Both shoulders stand above the neckline they break down through.
                and p1 > neckline
                and p3 > neckline
                and fresh(last_index)
                and _level_unbroken(df, neckline, "short", last_index)
            ):
                out.append(
                    PatternCandidate(
                        kind="head-and-shoulders",
                        direction="short",
                        breakout_level=neckline,
                        target_height=head - neckline,
                        start_ts=window[0].ts,
                        end_ts=window[-1].ts,
                    )
                )

        elif kinds == ("low", "high", "low", "high", "low"):
            t1, p1, head, p2, t3 = (p.price for p in window)
            neckline = max(p1, p2)  # stricter (higher) peak: conservative trigger
            if (
                head < t1 * (1 - head_prom)
                and head < t3 * (1 - head_prom)
                and abs(t1 - t3) / max(t1, t3) <= shoulder_tol
                and head < neckline
                # Both shoulders sit below the neckline they break up through.
                and t1 < neckline
                and t3 < neckline
                and fresh(last_index)
                and _level_unbroken(df, neckline, "long", last_index)
            ):
                out.append(
                    PatternCandidate(
                        kind="inverse-head-and-shoulders",
                        direction="long",
                        breakout_level=neckline,
                        target_height=neckline - head,
                        start_ts=window[0].ts,
                        end_ts=window[-1].ts,
                    )
                )

    return out


def _line_value(x1: int, y1: float, x2: int, y2: float, x: int) -> float:
    """Value at x of the line through (x1, y1) and (x2, y2).

    Raises:
        ValueError: if x1 == x2 — one point determines no line.
    """
    if x2 == x1:
        raise ValueError(f"_line_value needs distinct x, got x1 == x2 == {x1}")
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


def _detect_triangles(df, pivots, fresh) -> list[PatternCandidate]:
    """Converging endpoint-fit trendlines through pivot highs and pivot lows.

    The fit is deliberately simple and auditable: the upper line passes through
    the first and last pivot highs, the lower line through the first and last
    pivot lows, both taken from the last TRIANGLE_MAX_WIDTH_BARS bars only.

    A candidate must clear four independent tests, because convergence alone is
    satisfied by almost any pair of lines fitted over a long enough window:

      1. Width — the fitted structure spans at least TRIANGLE_MIN_WIDTH_BARS and
         (by construction) at most TRIANGLE_MAX_WIDTH_BARS bars.
      2. Non-expanding sides — the upper line must not rise and the lower line
         must not fall, so the shape is a genuine wedge/triangle rather than two
         lines that happen to close part of their gap.
      3. Convergence — the vertical range at the latest bar has contracted by at
         least TRIANGLE_MIN_CONVERGENCE, without the lines crossing.
      4. Containment — every bar the lines span stays inside them within
         TRIANGLE_CONTAINMENT_TOL, and the latest close is still between them.
         Price already outside means the structure is broken, not pending.
    """
    out: list[PatternCandidate] = []
    # A line needs two points; the config knob cannot lower that.
    min_per_side = max(2, config.TRIANGLE_MIN_PIVOTS_PER_SIDE)
    n = len(df)
    end_x = n - 1
    if end_x < config.TRIANGLE_MIN_WIDTH_BARS:
        return out

    earliest = max(0, end_x - config.TRIANGLE_MAX_WIDTH_BARS)
    ph = [p for p in pivots if p.kind == "high" and p.index >= earliest]
    pl = [p for p in pivots if p.kind == "low" and p.index >= earliest]
    if len(ph) < min_per_side or len(pl) < min_per_side:
        return out
    if ph[0].index == ph[-1].index or pl[0].index == pl[-1].index:
        return out

    # BOTH trendlines must be anchored on recent structure: with max() here, one
    # line could be fitted to pivots long out of date and still set a level.
    if not fresh(min(ph[-1].index, pl[-1].index)):
        return out

    start_x = min(ph[0].index, pl[0].index)
    if end_x - start_x < config.TRIANGLE_MIN_WIDTH_BARS:
        return out

    def upper(x: int) -> float:
        return _line_value(ph[0].index, ph[0].price, ph[-1].index, ph[-1].price, x)

    def lower(x: int) -> float:
        return _line_value(pl[0].index, pl[0].price, pl[-1].index, pl[-1].price, x)

    upper_start, upper_end = upper(start_x), upper(end_x)
    lower_start, lower_end = lower(start_x), lower(end_x)

    if upper_end > upper_start or lower_end < lower_start:
        return out  # a side is expanding: not a triangle

    start_range = upper_start - lower_start
    end_range = upper_end - lower_end
    if start_range <= 0 or end_range <= 0:
        return out  # degenerate or already-crossed lines
    if end_range > start_range * (1 - config.TRIANGLE_MIN_CONVERGENCE):
        return out  # not converging enough

    if not _triangle_contains(df, upper, lower, start_x, end_x):
        return out

    last_close = float(df["close"].to_numpy()[end_x])
    if not (lower_end < last_close < upper_end):
        return out  # price already outside the structure: broken, not pending

    start_ts = int(df.index[start_x])
    # Anchored at end_x, the same bar the breakout levels are evaluated at.
    end_ts = int(df.index[end_x])

    # Direction is decided by which side actually breaks; emit both candidates
    # and let the trigger bar pick at most one.
    out.append(
        PatternCandidate(
            kind="triangle",
            direction="long",
            breakout_level=upper_end,
            target_height=start_range,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    )
    out.append(
        PatternCandidate(
            kind="triangle",
            direction="short",
            breakout_level=lower_end,
            target_height=start_range,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    )
    return out


def _triangle_contains(df, upper, lower, start_x: int, end_x: int) -> bool:
    """True if every bar in [start_x, end_x] stays inside the trendlines.

    Without this, two lines that merely converge qualify as a triangle even when
    price spent the window well outside them.
    """
    tol = config.TRIANGLE_CONTAINMENT_TOL
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    for x in range(start_x, end_x + 1):
        if highs[x] > upper(x) * (1 + tol) or lows[x] < lower(x) * (1 - tol):
            return False
    return True


def _detect_flags(df) -> list[PatternCandidate]:
    """Impulse pole + shallow contained consolidation ending at the latest bar.

    For each consolidation length c (smallest first), the pole is the move into
    the bar just before the consolidation, measured over the
    FLAG_POLE_WINDOW_BARS bars ending there. A bull flag requires the pole to
    rise at least FLAG_POLE_MIN_PCT, the consolidation lows to hold above the
    FLAG_MAX_RETRACE retracement, and the consolidation highs to stay at or
    under the pole's high. Bear flags mirror this. At most one candidate per
    direction (the tightest consolidation wins).

    Containment is tested against the pole's extreme over the whole pole window,
    the same window the pole height is measured over — checking only the final
    pole bar's high rejected valid flags whose pole peaked a bar or two earlier.
    """
    out: list[PatternCandidate] = []
    n = len(df)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    ts_vals = df.index.to_numpy()

    found_long = False
    found_short = False
    for c in range(config.FLAG_CONSOL_MIN_BARS, config.FLAG_CONSOL_MAX_BARS + 1):
        if found_long and found_short:
            break
        pole_end = n - 1 - c
        if pole_end < 1:
            break
        # Inclusive slice, so subtract one less to span exactly WINDOW bars.
        pole_start = max(0, pole_end - config.FLAG_POLE_WINDOW_BARS + 1)
        pole_high = highs[pole_start : pole_end + 1].max()
        pole_low = lows[pole_start : pole_end + 1].min()
        consol_high = highs[pole_end + 1 :].max()
        consol_low = lows[pole_end + 1 :].min()

        if not found_long:
            pole_height = closes[pole_end] - pole_low
            if (
                pole_height / closes[pole_end] >= config.FLAG_POLE_MIN_PCT
                and consol_low >= closes[pole_end] - config.FLAG_MAX_RETRACE * pole_height
                and consol_high <= pole_high
            ):
                out.append(
                    PatternCandidate(
                        kind="flag",
                        direction="long",
                        breakout_level=float(consol_high),
                        target_height=float(pole_height),
                        start_ts=int(ts_vals[pole_start]),
                        end_ts=int(ts_vals[n - 1]),
                    )
                )
                found_long = True

        if not found_short:
            pole_height = pole_high - closes[pole_end]
            if (
                pole_height / closes[pole_end] >= config.FLAG_POLE_MIN_PCT
                and consol_high <= closes[pole_end] + config.FLAG_MAX_RETRACE * pole_height
                and consol_low >= pole_low
            ):
                out.append(
                    PatternCandidate(
                        kind="flag",
                        direction="short",
                        breakout_level=float(consol_low),
                        target_height=float(pole_height),
                        start_ts=int(ts_vals[pole_start]),
                        end_ts=int(ts_vals[n - 1]),
                    )
                )
                found_short = True

    return out
