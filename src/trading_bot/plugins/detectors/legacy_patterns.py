"""
detector.legacy-patterns — v0.2.0's retired chart-pattern geometry, behind the
Detector contract.

This is where signals/pivots.py and signals/patterns.py land behind the
contracts. Neither file is edited: their behavior is frozen by the Phase 3 parity
gate, and Phase 8 owns re-authoring these detectors as first-class plug-ins.
"""

import logging

from trading_bot import config
from trading_bot.framework import contracts
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.registry import register
from trading_bot.signals.patterns import detect_patterns
from trading_bot.signals.pivots import find_pivots

logger = logging.getLogger("trading_bot")


@register(
    "detector",
    name="legacy-patterns",
    params={
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
    },
    rationale=(
        "H&S / triangle / flag geometry, retired from v0.2.0's dispatch in Phase 5 "
        "because triangle and flag were ~98% of trade volume and lost on all three "
        "symbols. Migrated to keep the reference implementation reachable and to "
        "give Phase 8 a baseline to improve on; it is NOT in the v0.2.0 parity "
        "graph."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
    tier=2,
)
def legacy_patterns(
    ctx, *, pivot_span: int, max_age_bars: int, lookback_bars: int
) -> list:
    """Wrap signals.pivots.find_pivots + signals.patterns.detect_patterns.

    `pivot_span` is pivots.py's one parameter, and pivot confirmation needs no
    extra guard here because ctx.window() already ends at the last CLOSED setup
    bar, so find_pivots can only emit pivots with a full span of closed bars on
    both sides (pivots.py:7-13).

    Unlike the Donchian detector's 0-or-1, detect_patterns can return SEVERAL
    events at once — at most one per (kind, direction), ordered by
    (kind, direction) (patterns.py:96-101). The executor handles that with
    setup.rank_signals, which is what TestTieBreak exercises through this
    detector rather than through an invented stub.

    SCOPE NOTE: the ~14 geometry tolerances (HS_SHOULDER_TOLERANCE,
    TRIANGLE_MIN_CONVERGENCE, FLAG_POLE_MIN_PCT, ...) are NOT exposed as
    ParamSpecs. patterns.py's private detectors read them from config directly, so
    exposing them means editing a v0.2.0 module this phase must not fork. Phase 8
    owns re-authoring these detectors with real ParamSpecs; until then this
    plug-in honestly declares three parameters rather than pretending to declare
    seventeen.
    """
    window = ctx.window(config.SIGNAL_PATTERN_TIMEFRAME, lookback_bars)
    pivots = find_pivots(window, span=pivot_span)
    return [
        contracts.event_from_candidate(c)
        for c in detect_patterns(window, pivots, max_age_bars=max_age_bars)
    ]
