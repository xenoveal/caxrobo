"""
detector.bollinger-fade — the ranging-regime mean-reversion setup, behind the
Detector contract.

A7 — config.FADE_ENABLED IS READ HERE, AT CALL TIME, and the detector returns []
when it is false. This is the faithful port of engine.py:343 and scan.py:58: the
kill switch must suppress live and backtest identically, and putting the check in
the DETECTOR rather than in the executor means it keeps working through ANY
executor, and lets a graph carry the fade branch unconditionally.
framework.context's candidate memo carries bool(config.FADE_ENABLED) in its key
for the reason engine.py:322-327 states — a live flip must not be served a cached
pre-flip candidate list.
"""

import logging

from trading_bot import config
from trading_bot.framework.contracts import DetectedEvent, ParamSpec
from trading_bot.framework.registry import register
from trading_bot.signals.meanrev import FADE_KIND, detect_fade_setups

logger = logging.getLogger("trading_bot")


@register(
    "detector",
    name="bollinger-fade",
    params={
        "period": ParamSpec(
            kind="int",
            default=config.BB_PERIOD,
            bounds=(5, 100),
            doc="Bollinger middle-band SMA period (setup-tier bars)",
        ),
        "num_std": ParamSpec(
            kind="float",
            default=config.BB_STD,
            bounds=(0.5, 4.0),
            doc="Band width in rolling standard deviations",
        ),
        "max_age_bars": ParamSpec(
            kind="int",
            default=config.FADE_STRETCH_MAX_AGE_BARS,
            bounds=(1, 48),
            doc="Band-stretch recency bound (setup-tier bars)",
        ),
        "lookback_bars": ParamSpec(
            kind="int",
            default=config.PATTERN_LOOKBACK_BARS,
            bounds=(60, 500),
            doc="Setup bars the detector may see",
        ),
    },
    rationale=(
        "Bollinger band-stretch fade: a close outside the band, then a fresh "
        "re-cross back through it, faded to the mean. THE SLEEVE IS DROPPED "
        "(config.FADE_ENABLED = False) on a measured Phase 6 verdict — pooled "
        "expectancy -0.3883% on n=297, 0 of 3 symbols positive, cost ratio above "
        "the 0.10 ceiling on all three. It is migrated anyway, and kept tested, so "
        "the decision stays reversible and a future re-test costs nothing. Its "
        "presence here is NOT an endorsement."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
    tier=2,
)
def bollinger_fade(
    ctx,
    *,
    period: int,
    num_std: float,
    max_age_bars: int,
    lookback_bars: int,
) -> list:
    """Wrap signals.meanrev.detect_fade_setups behind the Detector contract.

    Reproduces meanrev._to_trigger_candidate (meanrev.py:159-173) field for field,
    which is the adapter idiom DetectedEvent generalizes.

    GOTCHA — `end_ts` ASYMMETRY, deliberately preserved. The Donchian detector's
    end_ts is the setup bar's CLOSE (ts + interval); the fade's is the most recent
    STRETCH bar's ts (meanrev.py:152, passed through unchanged by
    _to_trigger_candidate). Do NOT "make them consistent": check_breakout's
    `ts < end_ts` skip means changing either one changes which trigger bars are
    eligible, and the fade's looser rule is v0.2.0's measured behavior.

    `meta` carries stop_level and target because build_fade_signal needs the
    FadeCandidate's structural stop and mean target, and the six fixed
    DetectedEvent fields cannot express them. The detector runs BEFORE the trigger
    fires, so meta here holds none of the trigger facts — the executor stamps
    those in (contracts.with_trigger).
    """
    if not config.FADE_ENABLED:
        return []
    window = ctx.window(config.SIGNAL_PATTERN_TIMEFRAME, lookback_bars)
    out: list[DetectedEvent] = []
    for c in detect_fade_setups(
        window, period=period, num_std=num_std, max_age_bars=max_age_bars
    ):
        out.append(
            DetectedEvent(
                kind=FADE_KIND,
                direction=c.direction,
                level=c.trigger_level,  # meanrev.py:169
                target_height=abs(c.target - c.trigger_level),  # meanrev.py:170
                start_ts=c.start_ts,
                end_ts=c.end_ts,  # the stretch bar's ts, NOT +interval
                meta={
                    "stop_level": float(c.stop_level),
                    "target": float(c.target),
                    "trigger_level": float(c.trigger_level),
                },
            )
        )
    return out
