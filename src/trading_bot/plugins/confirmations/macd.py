"""
confirmation.macd — momentum agreement on the SETUP tier.

Passes when the price-normalised MACD histogram at the last closed setup bar
AGREES with the event's direction: `hist > min_hist` for a long,
`hist < -min_hist` for a short.

WHY THE SETUP TIER (config.SIGNAL_PATTERN_TIMEFRAME, currently 4h). The same
argument signals/donchian.py records as its Stated Assumption A1, in four parts:

 1. RISK AND REWARD ARE BOTH SETUP-TIER QUANTITIES. The stop is
    ATR_STOP_MULTIPLE * ATR(setup tf); the target is level +/- target_height, a
    setup-tier channel width. Confirming momentum on a different tier would
    confirm on one volatility scale and size risk on another — exactly the
    mistake A1 was written to prevent.
 2. EVERY EXISTING DETECTOR ALREADY LIVES THERE. detect_donchian_setups,
    detect_fade_setups and detect_patterns all take a setup-tier window, so a
    setup-tier MACD is one code path, not a fourth tier.
 3. THE 1D REGIME TIER WOULD BE WRONG. A daily MACD changes state once a day and
    is therefore near-constant across a setup window; a gate that is on or off
    for days at a time is a regime proxy, and the regime layer already exists and
    is the ONE measured-healthy component (contract §1). Duplicating it would
    repeat KNOWN-LIMITATIONS §0's error: "three of its four conditions are the
    same trend-strength idea measured three ways."
 4. THE 1H TRIGGER TIER WOULD BE WRONG. MACD state could flip INSIDE a setup
    window, decoupling the confirmation from the geometry it confirms, and would
    make MACD the fastest-moving input in a system whose regime is daily.

REJECTED: multi-tier MACD agreement (1d AND 4h). It doubles the gate's degrees of
freedom for an unmeasured benefit, and belongs on a Phase 6 grid axis if anywhere.

SYMMETRY. With `min_hist = 0.0` (the default, and a PURE SIGN TEST, so no fitted
number enters) the two branches are `hist > 0` and `hist < 0`. A positive
`min_hist` makes the gate symmetric AROUND ZERO rather than biased long — hence
`-min_hist` on the short side. Using `min_hist` there instead would make the short
gate permissive while looking symmetric, which is why a mirrored-data test pins it.

NORMALISATION makes the threshold portable across symbols: `hist` is
`(line - signal) / close`, so it is in fraction-of-price units — "a raw MACD
threshold that works on BTC is meaningless on DOGE". The divisor is positive, so
the sign test is identical normalised or not (tests/test_macd.py pins that).

MACD IS COMPUTED OVER FULL HISTORY through `EvalContext.series`, not over a
lookback window; see indicators/macd.py's CONSEQUENCE paragraph. MACD is
implemented in exactly one place — indicators/macd.py — and is not re-derived
here.

NaN REJECTS (fail closed): a warmup histogram is "not knowable yet", never
"confirmed". The NaN guard is `not math.isfinite(...)` rather than a bare
comparison, because NaN comparisons are always False and would silently read as
"momentum disagrees" instead of "momentum unknown" (signals/setup.py:131-136's
idiom).
"""

import logging
import math

from trading_bot import config
from trading_bot.framework.contracts import ConfirmationVerdict, ParamSpec
from trading_bot.framework.registry import register
from trading_bot.indicators.macd import macd_hist

logger = logging.getLogger("trading_bot")

MACD_CONFIRM_NAME = "confirmation.macd"


@register(
    "confirmation",
    name="macd",
    params={
        "fast": ParamSpec(
            kind="int",
            default=config.MACD_FAST_PERIOD,
            bounds=(2, 100),
            doc="Fast EMA span",
        ),
        "slow": ParamSpec(
            kind="int",
            default=config.MACD_SLOW_PERIOD,
            bounds=(3, 200),
            doc="Slow EMA span",
        ),
        "signal": ParamSpec(
            kind="int",
            default=config.MACD_SIGNAL_PERIOD,
            bounds=(2, 100),
            doc="Signal-line EMA span",
        ),
        "min_hist": ParamSpec(
            kind="float",
            default=config.MACD_CONFIRM_MIN_HIST,
            bounds=(0.0, 0.05),
            doc="Minimum |normalised histogram| in the event's direction",
        ),
    },
    rationale=(
        "Momentum must agree with the structure being traded: a breakout long "
        "while the MACD histogram is still negative is price moving against the "
        "prevailing momentum regime, the configuration that most often retraces. "
        "Computed on the SETUP tier so momentum and risk share one volatility "
        "scale (signals/donchian.py's Stated Assumption A1). Default min_hist=0.0 "
        "makes it a pure SIGN test, so it introduces no fitted number."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
)
def macd_confirmation(
    ctx, event, *, fast: int, slow: int, signal: int, min_hist: float
):
    """Gate an event on setup-tier MACD histogram sign.

    Never mutates `event` (contract §3). `score` carries the measured histogram
    in every branch — the graded value, reported and not used, the same
    discipline confirmation.volume-breakout follows.

    Args:
        ctx: EvalContext bound to the trigger bar's open, so
            ctx.bar_index(setup_tf) is the setup bar that had already CLOSED when
            the trigger bar opened (MEDIUM-2) — the same bar the policy's ATR
            comes from.
        event: The DetectedEvent under evaluation.
        fast / slow / signal: MACD spans.
        min_hist: Magnitude threshold on the normalised histogram.

    Returns:
        ConfirmationVerdict.
    """
    setup_tf = config.SIGNAL_PATTERN_TIMEFRAME
    hist = ctx.series(setup_tf, "macd-hist", macd_hist, fast=fast, slow=slow, signal=signal)

    if len(hist) == 0 or not math.isfinite(float(hist[-1])):
        logger.debug(
            "%s %s %s rejected: MACD histogram undefined",
            ctx.symbol,
            event.kind,
            event.direction,
        )
        return ConfirmationVerdict(
            passed=False,
            name=MACD_CONFIRM_NAME,
            score=float("nan"),
            reason=(
                f"MACD histogram undefined (warmup: needs "
                f"{slow + signal - 1} setup bars, saw {len(hist)})"
            ),
        )

    value = float(hist[-1])
    if event.direction == "long":
        passed = value > min_hist
        threshold = min_hist
        comparison = ">"
    else:
        passed = value < -min_hist
        threshold = -min_hist
        comparison = "<"

    reason = (
        f"MACD hist {value:.6f} {comparison if passed else 'not ' + comparison} "
        f"{threshold:.6f} for a {event.direction}"
    )
    if not passed:
        logger.debug(
            "%s %s %s rejected: %s",
            ctx.symbol,
            event.kind,
            event.direction,
            reason,
        )
    return ConfirmationVerdict(
        passed=passed, name=MACD_CONFIRM_NAME, score=value, reason=reason
    )
