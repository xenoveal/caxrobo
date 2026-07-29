"""
Catalog family 9 (Wyckoff Structures), narrowed to the two mechanically
decidable EVENTS: `wyckoff-spring` and `wyckoff-upthrust`.

=============================================================================
THIS MODULE'S DOCSTRING IS A DELIVERABLE. Read the non-claim before the code.
=============================================================================

WHAT IS DETECTED. A probe below (spring) or above (upthrust) an established,
tight trailing range that CLOSES BACK INSIDE the range on elevated volume.
That is a geometric, decidable rule over OHLCV bars.

WHAT IS **NOT** DETECTED. These detectors do **NOT** detect accumulation or
distribution, and they cannot distinguish a spring from an ordinary failed
breakout. That distinction lives in the surrounding multi-week phase sequence
(PS -> SC -> AR -> ST -> Spring -> LPS -> SOS), whose distinguishing evidence is
effort-versus-result judged across weeks.

WHY WYCKOFF ACCUMULATION / DISTRIBUTION IS **DEFERRED**, not built (stated
assumption A1, and the reason is not effort). There is no agreed numeric
definition, no reference implementation (PRD Research Summary), and no labelled
dataset. Achievable precision is therefore not merely low — it is
**unmeasurable**. A phase-labelled structure built to a made-up numeric
definition and validated against bars drawn to satisfy it would report a
precision figure that means nothing, and would enter Phase 9's campaign wearing
a five-star reliability label it had not earned. The contract's §9 tier-1 row
lands 2 of 3 for exactly this reason, and
`plugins/detectors/catalog.py` records both rows as `deferred` with this text.

WHAT CORRECTNESS VALIDATION IS THEREFORE LIMITED TO. Geometry-by-construction
fixtures only: bars are built so the answer is known by arithmetic, and the
tests validate THE RULE. They do not, and cannot, validate THE CONCEPT.
`test_docstring_states_the_non_claim` pins the words above so the honesty
requirement survives a refactor.

DO NOT add "Phase A/B/C/D" labels to `meta`. Labelling phases is precisely the
claim A2 forbids.

ONE DELIBERATE DEPARTURE FROM HOUSE STYLE. `signals/breakout.py:20-24` treats
volume as a GRADED confidence input that never blocks. Here volume is a HARD
condition, because a probe without a volume expansion is exactly the ordinary
failed breakout this detector otherwise has no way to exclude. The departure is
named so it reads as a decision rather than an inconsistency.
"""

import logging

import numpy as np

from trading_bot import config
from trading_bot.framework.contracts import DetectedEvent, ParamSpec
from trading_bot.framework.registry import register
from trading_bot.plugins.detectors import _geometry as g

logger = logging.getLogger("trading_bot")

SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME

_PROBE_PARAMS = {
    "lookback_bars": ParamSpec(
        kind="int",
        default=config.PATTERN_LOOKBACK_BARS,
        bounds=(40, 500),
        doc="Setup bars the detector may see",
    ),
    "range_bars": ParamSpec(
        kind="int",
        default=config.WYCKOFF_RANGE_BARS,
        bounds=(8, 120),
        doc="Trailing bars defining the range; the probe bar itself is EXCLUDED",
    ),
    "range_max_width_pct": ParamSpec(
        kind="float",
        default=config.WYCKOFF_RANGE_MAX_WIDTH_PCT,
        bounds=(0.01, 0.40),
        doc="(hi-lo)/close ceiling — only a real consolidation qualifies",
    ),
    "probe_min_pct": ParamSpec(
        kind="float",
        default=config.WYCKOFF_PROBE_MIN_PCT,
        bounds=(0.0005, 0.05),
        doc="Probe must pierce the level by at least this fraction",
    ),
    "probe_vol_ratio": ParamSpec(
        kind="float",
        default=config.WYCKOFF_PROBE_VOL_RATIO,
        bounds=(1.0, 5.0),
        doc="Probe volume / trailing mean volume; a HARD condition here",
    ),
}

_PROBE_RATIONALE_HEAD = (
    "One event Wyckoff schematics place inside Phase C of an accumulation. It "
    "does NOT detect accumulation, and cannot distinguish a spring from an "
    "ordinary failed breakout, because that distinction lives in a multi-week "
    "phase sequence with no agreed numeric definition and no reference "
    "implementation to check against (contract §9, PRD Research Summary). "
    "Registered so the buildable subset is measurable; the STRUCTURE itself is "
    "DEFERRED in catalog.py."
)


@register(
    "detector",
    name="wyckoff-spring",
    params=_PROBE_PARAMS,
    rationale=_PROBE_RATIONALE_HEAD,
    timeframes=(SETUP_TF,),
    tier=1,
)
def wyckoff_spring(ctx, **params) -> list[DetectedEvent]:
    """A probe BELOW a tight range that closes back inside on elevated volume. Long."""
    return _detect_probe(ctx, inverse=False, **params)


@register(
    "detector",
    name="wyckoff-upthrust",
    params=_PROBE_PARAMS,
    rationale=(
        "The mirror of wyckoff-spring, inside Phase C of a DISTRIBUTION. "
        + _PROBE_RATIONALE_HEAD
    ),
    timeframes=(SETUP_TF,),
    tier=1,
)
def wyckoff_upthrust(ctx, **params) -> list[DetectedEvent]:
    """A probe ABOVE a tight range that closes back inside on elevated volume. Short."""
    return _detect_probe(ctx, inverse=True, **params)


def _detect_probe(
    ctx,
    *,
    inverse: bool,
    lookback_bars: int,
    range_bars: int,
    range_max_width_pct: float,
    probe_min_pct: float,
    probe_vol_ratio: float,
) -> list[DetectedEvent]:
    """The spring / upthrust rule. See the module docstring for the non-claim.

    Five conditions, in this order:

      1. RANGE, over the trailing `range_bars` bars ending at `n-2` — the probe
         bar is EXCLUDED. Without that exclusion the probe's own low defines the
         support it is supposedly probing and the detector never fires.
         `hi = max(high)`, `lo = min(low)`, and `(hi-lo)/close[-1]` must not
         exceed `range_max_width_pct`. That width filter is
         `scripts/bruteforce/indicators.py:626-628`'s tight-range idea: only a
         range narrow enough to be a real consolidation qualifies, so the probe
         has a defined, small risk.
      2. NO PRIOR ACCEPTANCE of the level. No bar in the range window may have
         CLOSED at or through the floor being probed (`min(close) > lo`, and
         `max(close) < hi` for an upthrust): a bar that closed on the floor
         ACCEPTED that price, so the floor is not a level being defended, and a
         probe of a level price has already left is a continuation rather than
         a spring. (The plan's literal wording — "no close BELOW lo" — is
         vacuous, because `close >= low >= lo` always holds when `lo` is the
         window's minimum low. The strict-inequality version above is the
         non-vacuous reading; the deviation is recorded in the phase report.)
      3. THE PROBE BAR is the latest bar: `low < lo * (1 - probe_min_pct)` AND
         `close > lo` — the reclaim. STRICTLY greater: a close exactly at
         support is ambiguous, and `pivots.py:5-6` sets the house convention
         that ties reject.
      4. VOLUME: `volume[-1] / mean(volume[-1-VOLUME_LOOKBACK:-1]) >=
         probe_vol_ratio`, reusing `config.VOLUME_LOOKBACK` rather than adding a
         second lookback constant. A short volume history makes the mean NaN and
         `NaN >= ratio` is False, so the detector correctly emits nothing — that
         is asserted rather than guarded against.
      5. `level = lo` (or `hi`); `target_height = hi - lo`, a STATED
         CONVENTION: the range the market is expected to leave. There is no
         classical measured move for a spring.

    FRESHNESS is skipped: the probe IS the latest bar, so the structure cannot
    be stale. `level_unbroken` is subsumed by condition 2, which is the same
    "the level is not already spent" idea evaluated over the range window.
    """
    df = ctx.window(SETUP_TF, lookback_bars)
    n = len(df)
    if n < range_bars + 1:
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    volumes = df["volume"].to_numpy()
    ts_vals = df.index.to_numpy()

    start = n - 1 - range_bars
    w_high = highs[start : n - 1]
    w_low = lows[start : n - 1]
    w_close = closes[start : n - 1]

    hi = float(w_high.max())
    lo = float(w_low.min())
    last_close = float(closes[-1])
    if last_close <= 0 or hi <= lo:
        return []
    width_pct = (hi - lo) / last_close
    if width_pct > range_max_width_pct:
        return []

    level = hi if inverse else lo
    # 2 — no prior acceptance of the level.
    if inverse:
        if not float(w_close.max()) < hi:
            return []
    else:
        if not float(w_close.min()) > lo:
            return []

    # 3 — the probe bar.
    if inverse:
        probed = float(highs[-1]) > hi * (1 + probe_min_pct)
        reclaimed = last_close < hi
    else:
        probed = float(lows[-1]) < lo * (1 - probe_min_pct)
        reclaimed = last_close > lo
    if not (probed and reclaimed):
        return []

    # 4 — volume. NaN mean => NaN ratio => False, and nothing is emitted.
    vol_window = volumes[max(0, n - 1 - config.VOLUME_LOOKBACK) : n - 1]
    mean_vol = float(np.mean(vol_window)) if len(vol_window) else float("nan")
    ratio = float(volumes[-1]) / mean_vol if mean_vol > 0 else float("nan")
    if not (ratio >= probe_vol_ratio):
        return []

    probe_extreme = float(highs[-1]) if inverse else float(lows[-1])
    probe_depth_pct = abs(probe_extreme - level) / level

    return g.dedupe_events(
        [
            DetectedEvent(
                kind="wyckoff-upthrust" if inverse else "wyckoff-spring",
                direction="short" if inverse else "long",
                level=float(level),
                target_height=float(hi - lo),
                start_ts=int(ts_vals[start]),
                end_ts=int(ts_vals[n - 1]),
                meta={
                    "range_high": hi,
                    "range_low": lo,
                    "range_width_pct": float(width_pct),
                    "probe_depth_pct": float(probe_depth_pct),
                    "volume_ratio": float(ratio),
                },
            )
        ]
    )
