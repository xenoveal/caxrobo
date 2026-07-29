"""
The two v0.2.0 PositionPolicies, behind the PositionPolicy contract.

A8 — THE R:R FLOOR STAYS INSIDE THE MIGRATED POLICIES FOR PARITY; filter.*
plug-ins arrive in Phase 4. In v0.2.0 the floor is applied INSIDE
setup.build_signal (setup.py:154-156, on the GROSS ratio) and inside
meanrev.build_fade_signal (meanrev.py:233-234, on net_rr, the COST-ADJUSTED
ratio), and the two use different definitions on purpose. Extracting them into a
Filter node would change which setups survive and break parity. So the v0.2.0
parity graph has ZERO Filter nodes and ZERO Confirmation nodes — which is the
honest description of v0.2.0 (KNOWN-LIMITATIONS §0c: "volume is computed on every
signal but gates nothing"). The Filter and Confirmation contracts are authored and
tested in Phase 3 with in-test stubs; their first production instances are Phase
4's filters/rr_after_costs.py (RR_TARGET_MIN = 2.0) and
confirmations/{volume_breakout,macd}.py.

Both policies live in one module deliberately: it keeps the two halves of
v0.2.0's R:R story in one place, where the asymmetry is visible.
"""

import logging

from trading_bot import config
from trading_bot.framework import contracts
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.registry import register
from trading_bot.signals.meanrev import FadeCandidate, build_fade_signal
from trading_bot.signals.setup import build_signal

logger = logging.getLogger("trading_bot")


@register(
    "policy",
    name="atr-stop-measured-move",
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
        "rr_floor": ParamSpec(
            kind="float",
            default=config.RR_FLOOR,
            bounds=(1.0, 5.0),
            doc="Minimum GROSS reward:risk",
        ),
    },
    rationale=(
        "v0.2.0's entry/stop/target rule: stop k*ATR(setup tier) from entry "
        "(volatility-derived, never a fixed percentage), target at the measured "
        "move (level +/- target_height), screened on gross reward:risk. Wraps "
        "setup.build_signal unchanged so the migrated path cannot disagree with "
        "the live one."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
)
def atr_stop_measured_move(
    ctx, event, *, atr_multiple: float, atr_period: int, rr_floor: float
):
    """Delegate wholly to signals.setup.build_signal, then adapt.

    Returns None on rejection — warmup NaN ATR, non-positive risk or reward, or rr
    below the floor — exactly as build_signal does. A rejection is not an error and
    must not raise.

    THE ATR MUST BE THE SAME NUMBER engine.py:492-494 uses: atr_setup_vals[h_idx],
    i.e. the ATR at the SETUP bar, not at the trigger bar. It is, by construction
    rather than by coincidence: this policy runs with a context bound to the
    trigger bar's OPEN (framework/context.py's ROLES block), so
    ctx.bar_index(setup_tf) is exactly MEDIUM-2's h_idx (engine.py:481) and [-1] of
    a setup-tier series is that bar. TestAtrAtEntry asserts it on every entry
    rather than assuming it.

    build_signal takes atr_value POSITIONALLY (setup.py:76) and rejects
    NaN/non-positive itself, so it is passed through without a caller-side check;
    the executor separately reproduces engine.py:500-503's `atr_value > 0` skip, so
    the guard exists in both places, deliberately.
    """
    setup_tf = config.SIGNAL_PATTERN_TIMEFRAME
    atr_arr = ctx.atr(setup_tf, atr_period)
    if len(atr_arr) == 0:
        return None
    atr_value = float(atr_arr[-1])
    sig = build_signal(
        ctx.symbol,
        contracts.candidate_from_event(event),
        contracts.trigger_from_meta(event),
        atr_value,
        atr_multiple=atr_multiple,
        rr_floor=rr_floor,
    )
    return None if sig is None else contracts.plan_from_signal(sig, source=event.kind)


@register(
    "policy",
    name="fade-structural-stop",
    params={
        "rr_floor": ParamSpec(
            kind="float",
            default=config.RR_FLOOR,
            bounds=(1.0, 5.0),
            doc="Minimum COST-ADJUSTED reward:risk (risk.atr_stop.net_rr)",
        ),
    },
    rationale=(
        "v0.2.0's fade entry rule: stop at the excursion extreme (the structural "
        "level the reversion thesis is invalidated by), target at the middle band. "
        "Its floor is applied to risk.atr_stop.net_rr, the COST-ADJUSTED ratio, not "
        "the gross one, because a structural stop has no ATR floor and can sit "
        "arbitrarily close to entry (meanrev.py:186-198). Wraps "
        "meanrev.build_fade_signal unchanged."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
)
def fade_structural_stop(ctx, event, *, rr_floor: float):
    """Delegate wholly to signals.meanrev.build_fade_signal, then adapt.

    The FadeCandidate is reconstructed from the fields detector.bollinger-fade
    stamped into event.meta — stop_level and target, which the six fixed
    DetectedEvent fields cannot express — plus event.level / start_ts / end_ts.

    Note this policy's floor and atr-stop-measured-move's floor are applied to
    DIFFERENT ratios (cost-adjusted here, gross there). That asymmetry is v0.2.0's
    measured behavior and is deliberately preserved: converging them would change
    which setups survive on both paths at once.

    Deliberately declares NO atr_period: its stop is structural, so the executor's
    "a policy that declares an atr_period needs a defined ATR" guard correctly does
    not apply — which is how engine.py:504-515's asymmetry (no `atr_value > 0`
    check on the fade path) is expressed generically.
    """
    try:
        candidate = FadeCandidate(
            direction=event.direction,
            trigger_level=float(event.meta["trigger_level"]),
            stop_level=float(event.meta["stop_level"]),
            target=float(event.meta["target"]),
            start_ts=int(event.start_ts),
            end_ts=int(event.end_ts),
        )
    except KeyError as exc:
        raise contracts.ContractError(  # type: ignore[attr-defined]
            f"policy.fade-structural-stop needs {exc} in event.meta; only "
            f"detector.bollinger-fade supplies it"
        ) from exc
    sig = build_fade_signal(
        ctx.symbol,
        candidate,
        contracts.trigger_from_meta(event),
        rr_floor=rr_floor,
    )
    return None if sig is None else contracts.plan_from_signal(sig, source=event.kind)
