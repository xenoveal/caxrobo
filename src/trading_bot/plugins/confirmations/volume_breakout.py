"""
confirmation.volume-breakout — THIS PLUG-IN TURNS A COMPUTED-BUT-UNUSED NUMBER
INTO A HARD GATE.

`config.VOLUME_LOOKBACK` / `VOLUME_HIGH_RATIO`, `Signal.volume_ratio` /
`volume_high` and `Trade.volume_high` have existed since v0.2.0 and gated
NOTHING. KNOWN-LIMITATIONS §0c states it plainly: "volume is computed on every
signal but gates nothing", and signals/breakout.py:19-23 documents the old policy
as a design decision ("a graded confidence input, never a hard block"). This
Confirmation deliberately REVERSES that policy — for graph-composed strategies
only. The legacy signals/* path is untouched and still gates nothing, so v0.2.0's
measured behavior is not silently altered and the parity graph (which has no
Confirmation nodes) is unaffected.

WHY A GATE AND NOT A GRADED WEIGHT (the rejected alternative, recorded):

 1. A graded confidence has nothing to grade. Contract §10 keeps v0.2.0's
    equal-notional, one-open-trade-per-symbol rule and rules position sizing out
    of all nine phases; `rank_signals` already orders by R:R. With no sizing and
    no ranking contest to influence, a confidence score can change behavior only
    through a threshold — which IS a hard gate. "Graded" without sizing would be
    decorative, and decoration that looks like a feature is how §0c happened.
 2. A gate is measurable; a weight is not. A gate's cost is a trade-count and
    expectancy delta from one ablation run against an identical graph with the
    Confirmation nodes removed. A weight's effect is entangled with everything
    downstream.
 3. The information is not thrown away. `ConfirmationVerdict.score` carries the
    measured ratio in EVERY branch, pass or fail, so the graded value is in the
    audit trail for Phase 5's Reviewer and Phase 6's Mutator without Phase 4
    pretending to use it now.

WHERE THE RATIO COMES FROM, and why not from `ctx`. The ratio is read from
`event.meta["volume_ratio"]`, which `framework/execute.py` stamps in via
`contracts.with_trigger` from the `BreakoutEvent` that `signals/breakout.py`
produced. That is not a convenience: a Confirmation runs with a context bound to
the TRIGGER BAR'S OPEN (framework/context.py's ROLES block), so the trigger bar
itself is ABSENT from `ctx.frame(trigger_tf)` and the ratio is not computable from
the context at all. Reading the stamped value also means there is exactly ONE
volume-window definition in the system — signals/breakout.py:131-137, trigger
volume over the mean of the `TriggerSpec.volume_lookback` bars STRICTLY PRECEDING
it — shared with `Trade.volume_high`. A second computation here could drift from
the number the trade records.

Consequently this plug-in declares NO `lookback` parameter: the window is a
graph-level `TriggerSpec` field, and a per-plug-in lookback it could not actually
honor would be a lie in the registry.

NaN REJECTS (fail closed). signals/breakout.py:131-137 yields
`volume_ratio = NaN` when fewer than `volume_lookback` prior bars exist or their
mean is non-positive. "Unknown" must never read as "confirmed". Cost: the first
~VOLUME_LOOKBACK+1 trigger bars of each series cannot trade — negligible against
REGIME_MIN_BARS' 207-day warmup.
"""

import logging
import math

from trading_bot import config
from trading_bot.framework import contracts
from trading_bot.framework.contracts import ConfirmationVerdict, ParamSpec
from trading_bot.framework.registry import register

logger = logging.getLogger("trading_bot")

VOLUME_CONFIRM_NAME = "confirmation.volume-breakout"


@register(
    "confirmation",
    name="volume-breakout",
    params={
        "min_ratio": ParamSpec(
            kind="float",
            default=config.VOLUME_CONFIRM_MIN_RATIO,
            bounds=(0.0, 10.0),
            doc="Minimum trigger volume / prior-window mean",
        ),
        "require_defined": ParamSpec(
            kind="bool",
            default=config.VOLUME_CONFIRM_REQUIRE_DEFINED,
            doc="Reject when the volume ratio is undefined (fail closed)",
        ),
    },
    rationale=(
        "A breakout on volume below its own recent average is the classic false "
        "break: without participation there is no new supply/demand imbalance to "
        "sustain the move. Volume has been COMPUTED on every v0.2.0 signal and has "
        "gated nothing (KNOWN-LIMITATIONS §0c); this plug-in makes it a real gate "
        "so its cost in trades and expectancy is measurable for the first time. "
        "The threshold is VOLUME_HIGH_RATIO BY REFERENCE, so activating the gate "
        "invents no new number."
    ),
    timeframes=(config.SIGNAL_TRIGGER_TIMEFRAME,),
)
def volume_breakout(ctx, event, *, min_ratio: float, require_defined: bool):
    """Gate an event on the trigger bar's volume ratio.

    Never mutates `event` (contract §3): it returns a verdict and nothing else.
    `score` carries the measured ratio in every branch, including the failures,
    so a rejection is diagnosable from the verdict alone.

    Args:
        ctx: EvalContext bound to the trigger bar's open. Unused — the ratio
            lives in event.meta, for the reason the module docstring gives.
        event: The DetectedEvent, already stamped with the trigger facts.
        min_ratio: Minimum ratio to pass. Boundary is `>=`, matching
            signals/breakout.py:138's `volume_ratio >= volume_high_ratio`.
        require_defined: When True an undefined (NaN) ratio REJECTS.

    Returns:
        ConfirmationVerdict.

    Raises:
        ContractError: From contracts.trigger_from_meta, if the event was never
            stamped — which would mean a confirmation ran before the trigger
            fired, an executor bug rather than a rejection.
    """
    ratio = float(contracts.trigger_from_meta(event).volume_ratio)

    if math.isnan(ratio):
        if require_defined:
            logger.debug(
                "%s %s %s rejected: volume ratio undefined",
                ctx.symbol,
                event.kind,
                event.direction,
            )
            return ConfirmationVerdict(
                passed=False,
                name=VOLUME_CONFIRM_NAME,
                score=ratio,
                reason=(
                    "volume ratio undefined (fewer prior bars than the trigger "
                    "spec's volume_lookback, or a non-positive prior mean); "
                    "'unknown' must never read as 'confirmed'"
                ),
            )
        return ConfirmationVerdict(
            passed=True,
            name=VOLUME_CONFIRM_NAME,
            score=ratio,
            reason="volume ratio undefined but require_defined is False",
        )

    if ratio >= min_ratio:
        return ConfirmationVerdict(
            passed=True,
            name=VOLUME_CONFIRM_NAME,
            score=ratio,
            reason=f"volume ratio {ratio:.4f} >= {min_ratio:.4f}",
        )

    logger.debug(
        "%s %s %s rejected: volume ratio %.4f below %.4f",
        ctx.symbol,
        event.kind,
        event.direction,
        ratio,
        min_ratio,
    )
    return ConfirmationVerdict(
        passed=False,
        name=VOLUME_CONFIRM_NAME,
        score=ratio,
        reason=f"volume ratio {ratio:.4f} < {min_ratio:.4f}",
    )
