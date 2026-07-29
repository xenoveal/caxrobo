"""
policy.measured-move — the pivot guide's Strategy Step 3, behind the
PositionPolicy contract: "decide the position based on the chart pattern outcome
(long/short/no-entry) and its entry, and TP/SL point."

This is signals/setup.py's `build_signal` body re-expressed against the contracts,
WITH ONE DELIBERATE OMISSION: it performs NO reward:risk rejection at all.

WHY THE R:R SCREEN IS NOT HERE. `build_signal` conflates SL/TP computation with the
`rr < rr_floor` screen (setup.py:154-156). The contracts split those roles — a
`PositionPolicy` DECIDES, a `Filter` ACCEPTS — and Phase 4's whole measurement is
the distribution of net R:R over the plans that reach the filter. Porting the
gross RR_FLOOR = 1.5 screen into this policy would pre-filter that distribution
before `filter.rr-after-costs` ever sees it, making `graph-backtest --rr-report` a
report on a truncated sample. policy.atr-stop-measured-move (Phase 3's migration
of build_signal) keeps the screen, because ITS job is v0.2.0 parity; this policy's
job is being composable.

TWO RATIOS WITH THE SAME NAME, and which is which — the single most likely
confusion in this phase:

  - `PositionPlan.rr` (set here) is the GROSS ratio `reward_pct / risk_pct`,
    matching `Signal.rr` (setup.py:54). Cost-blind by construction.
  - `Trade.planned_rr` is the NET, cost-adjusted ratio from
    `risk.atr_stop.net_rr`, computed at the graph->Trade seam. That is the number
    `filter.rr-after-costs` gates on and the number the audit trail records.

THE ATR COMES FROM THE SETUP TIER, and must. engine.py computes the entry stop's
ATR on config.SIGNAL_PATTERN_TIMEFRAME; a trigger-tier ATR would shrink every
stop by roughly sqrt(4) at the current 4h/1h tiers and silently blow through
config.COST_RATIO_CEILING — config.py:169-173 records that the ceiling was NOT
satisfiable at the old 1H setup tier. It is also the RIGHT setup bar by
construction rather than by coincidence: this policy runs with a context bound to
the trigger bar's OPEN (framework/context.py's ROLES block), so
`ctx.atr(setup_tf, period)[-1]` is exactly engine.py's `atr_setup_vals[h_idx]`
under MEDIUM-2.
"""

import logging

from trading_bot import config
from trading_bot.framework import contracts
from trading_bot.framework.contracts import ParamSpec, PositionPlan
from trading_bot.framework.registry import register
from trading_bot.risk.atr_stop import compute_atr_stop

logger = logging.getLogger("trading_bot")

MEASURED_MOVE_NAME = "policy.measured-move"


@register(
    "policy",
    name="measured-move",
    params={
        "atr_multiple": ParamSpec(
            kind="float",
            default=config.ATR_STOP_MULTIPLE,
            bounds=(0.5, 6.0),
            doc="Stop distance in ATRs",
        ),
        "atr_period": ParamSpec(
            kind="int",
            default=config.ATR_STOP_PERIOD,
            bounds=(5, 50),
            doc="Wilder ATR period (setup tier)",
        ),
    },
    rationale=(
        "Stop k*ATR(setup tier) from entry -- a volatility-derived, "
        "market-structure distance, never a fixed percentage of price; target at "
        "the measured move (level +/- target_height), the classic projection of a "
        "resolved range. Identical arithmetic to v0.2.0's setup.build_signal "
        "MINUS its gross reward:risk screen, which now lives in "
        "filter.rr-after-costs as a NET floor so the cost-adjusted distribution "
        "can be measured rather than pre-truncated."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
)
def measured_move(ctx, event, *, atr_multiple: float, atr_period: int):
    """Derive direction/entry/stop/target for a confirmed, triggered event.

    Args:
        ctx: EvalContext bound to the trigger bar's open.
        event: The DetectedEvent, stamped with the trigger facts.
        atr_multiple: k in `stop = entry -/+ k * ATR`.
        atr_period: Wilder ATR period, on the setup tier.

    Returns:
        PositionPlan, or None on rejection — an undefined/non-positive ATR, a
        non-positive entry or level, or non-positive risk or reward. A rejection
        is normal, high-frequency control flow and must NOT raise.
    """
    setup_tf = config.SIGNAL_PATTERN_TIMEFRAME
    trigger = contracts.trigger_from_meta(event)

    def reject(reason: str, *args) -> None:
        logger.debug(
            "%s %s %s rejected: " + reason,
            ctx.symbol,
            event.kind,
            event.direction,
            *args,
        )

    entry = float(trigger.price)
    if entry <= 0 or event.level <= 0:
        return None

    atr_arr = ctx.atr(setup_tf, atr_period)
    if len(atr_arr) == 0:
        reject("no setup-tier ATR value is available yet")
        return None
    atr_value = float(atr_arr[-1])

    # NaN comparisons are always False, so `atr_value <= 0` alone would let a
    # NaN ATR (Wilder warmup) silently produce a stop = nan plan instead of
    # being rejected. `not (atr_value > 0)` catches NaN, zero, and negative.
    # Verbatim from signals/setup.py:131-136, comment included.
    if not (atr_value > 0):
        reject("atr_value %s is undefined or non-positive", atr_value)
        return None

    stop = compute_atr_stop(entry, event.direction, atr_value, atr_multiple)
    if event.direction == "long":
        target = event.level + event.target_height
        reward = target - entry
    else:
        target = event.level - event.target_height
        reward = entry - target

    risk = abs(entry - stop)
    if risk <= 0 or reward <= 0:
        reject("risk %s or reward %s is non-positive", risk, reward)
        return None

    risk_pct = risk / entry
    reward_pct = reward / entry
    # GROSS, deliberately: the NET ratio is filter.rr-after-costs' output and
    # lands on Trade.planned_rr. See the module docstring.
    rr = reward_pct / risk_pct

    return PositionPlan(
        symbol=ctx.symbol,
        ts=int(trigger.ts),
        direction=event.direction,
        entry=entry,
        stop=stop,
        target=target,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr=rr,
        source=event.kind,
    )
