"""
detector.donchian-breakout — v0.2.0's only live signal method, behind the
Detector contract.

Zero logic of its own: the window comes from EvalContext, the detection from
signals.donchian.detect_donchian_setups, and the adaptation from
contracts.event_from_candidate.
"""

import logging

from trading_bot import config
from trading_bot.framework import contracts
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.registry import register
from trading_bot.signals.donchian import detect_donchian_setups

logger = logging.getLogger("trading_bot")


@register(
    "detector",
    name="donchian-breakout",
    params={
        "entry_period": ParamSpec(
            kind="int",
            default=config.DONCHIAN_ENTRY_PERIOD,
            bounds=(5, 200),
            doc="Entry/exit channel lookback (bars)",
        ),
        "trend_period": ParamSpec(
            kind="int",
            default=config.DONCHIAN_TREND_PERIOD,
            bounds=(10, 400),
            doc="Mid-line trend-filter lookback (bars)",
        ),
        "adx_period": ParamSpec(
            kind="int",
            default=config.ADX_PERIOD,
            bounds=(5, 50),
            doc="Wilder ADX period",
        ),
        "adx_min": ParamSpec(
            kind="float",
            default=config.ADX_TREND_THRESHOLD,
            bounds=(0.0, 60.0),
            doc="Minimum setup-tier ADX",
        ),
        "lookback_bars": ParamSpec(
            kind="int",
            default=config.PATTERN_LOOKBACK_BARS,
            bounds=(60, 500),
            doc="Setup bars the detector may see",
        ),
    },
    rationale=(
        "Canonical Donchian channel breakout (Donchian 1960s, Turtle lineage), "
        "ADX-confirmed on the setup tier and filtered by the 55-bar mid-line. The "
        "most-replicated directional edge in systematic trading, and v0.2.0's "
        "only live signal method."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
    tier=2,
)
def donchian_breakout(
    ctx,
    *,
    entry_period: int,
    trend_period: int,
    adx_period: int,
    adx_min: float,
    lookback_bars: int,
) -> list:
    """Wrap signals.donchian.detect_donchian_setups behind the Detector contract.

    A reimplementation here would be a second Donchian detector to keep in sync,
    and parity would be testing this file rather than the migration.

    20 and 55 remain CANONICAL. They are exposed as ParamSpecs because Phase 6
    needs legal bounds and Phase 7 needs a control, NOT because this phase sweeps
    them; the v0.2.0 note that they "must never appear in a walk-forward grid"
    (config.py:105-111) is a decision about THAT phase's grid, and spending them
    later is a logged degree of freedom.

    The timeframe is read from config.SIGNAL_PATTERN_TIMEFRAME rather than from
    ctx.tiers[1] because the WRAPPED function does, and the two must not be able
    to disagree.

    `end_ts` is left exactly as detect_donchian_setups sets it — the setup bar's
    CLOSE (donchian.py:127-138). Do not "normalize" it in the adapter:
    check_breakout's `ts < end_ts` skip depends on it.
    """
    window = ctx.window(config.SIGNAL_PATTERN_TIMEFRAME, lookback_bars)
    return [
        contracts.event_from_candidate(c, meta={"channel_width": c.target_height})
        for c in detect_donchian_setups(
            window,
            entry_period=entry_period,
            trend_period=trend_period,
            adx_period=adx_period,
            adx_min=adx_min,
        )
    ]
