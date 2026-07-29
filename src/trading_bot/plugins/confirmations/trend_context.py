"""
confirmation.trend-context — THE `Confirmation` THREE DOCSTRINGS PROMISED AND
NOBODY WROTE.

`plugins/detectors/continuation.py:27` states plainly: "nothing here tests for
a prior trend. That is a `Confirmation`'s job (Phase 4)". The same deferral
appears at `continuation.py:759` (flags) and in `reversal.py`'s reversal-kind
docstrings. `plugins/confirmations/` shipped with exactly two plug-ins —
`macd`, `volume-breakout` — neither of which is directional in the sense
needed here: `RegimeGate` (`framework/graph.py`) classifies TRENDING-vs-RANGING
off ADX/ATR, not UP-vs-DOWN. Nothing in the pipeline has ever checked that a
"double bottom" actually reversed a decline, or that a "bull flag" actually
continued an advance. This plug-in is that check.

WHAT IT DOES. For a `DetectedEvent` whose `kind` is one of the thirteen catalog
chart-pattern rows this repo has registered, it looks up whether that kind's
own classical definition (`.claude/technical-pattern.md`, families 1-2) makes a
prior-trend claim, and if so, in which direction, then measures the setup-tier
OHLC over a declared lookback and gates on whether that claimed trend is
actually there.

THE LOOKUP IS AN EXPLICIT TABLE, NOT A STRING HEURISTIC. `REVERSAL_KINDS`,
`CONTINUATION_KINDS` and `NO_TREND_CLAIM_KINDS` below are the entire
vocabulary this plug-in understands, built directly off the `@register` names
in `plugins/detectors/reversal.py` and `plugins/detectors/continuation.py` (and
cross-checked against `plugins/detectors/catalog.py`'s families 1-2). Nothing
here parses `event.kind` with `.startswith` or `"bear" in kind` — a kind is
either in exactly one of the three sets or it is UNKNOWN, and both of those
are handled by one small function (`_required_trend_direction`) that a test
can enumerate exhaustively (`test_lookup_covers_every_registered_chart_pattern_kind_exactly_once`).

DIRECTION SEMANTICS.
  * REVERSAL kinds (`double-bottom`, `double-top`, `head-and-shoulders`,
    `inverse-head-and-shoulders`, `cup-and-handle`, `inverse-cup-and-handle`)
    reverse a move — the OPPOSITE of `event.direction`'s own bias must have
    been the recent prior trend. A `double-bottom` (direction "long") must
    have been preceded by a DECLINE; a `double-top` (direction "short") must
    have been preceded by an ADVANCE.
  * CONTINUATION kinds (`bull-flag`, `bear-flag`, `ascending-triangle`,
    `descending-triangle`, `falling-wedge`, `rising-wedge`) continue a move —
    the SAME bias as `event.direction` must have been the recent prior trend.
    A `bull-flag` (direction "long") must have been preceded by an ADVANCE.

SYMMETRICAL TRIANGLE IS DELIBERATELY EXEMPT. `continuation.py`'s module
docstring is explicit that a symmetrical triangle "emits BOTH a long and a
short candidate and lets the trigger bar pick at most one" and the catalog
text gives it no single prevailing-trend claim to check (unlike the other
eight rows in that file, which each name one). Inventing a directional
requirement here that the pattern's own definition does not state would not
be "closing the gap" the three docstrings point at — it would be fabricating
a rule and hiding it inside a Confirmation, the exact anti-pattern
`continuation.py:27` calls out ("baking it in would hide the decision inside
geometry where no report could see it"). `symmetrical-triangle` is therefore
in `NO_TREND_CLAIM_KINDS` and PASSES UNCONDITIONALLY, with `score = nan` and a
`reason` that says so, so the pass-through is visible in the audit trail
rather than silently indistinguishable from "trend measured, gate satisfied".

UNKNOWN KINDS PASS THROUGH FOR THE SAME REASON, NOT THE OPPOSITE ONE. A
`DetectedEvent` from any detector this file has never heard of — e.g.
`donchian-breakout`, `rsi-divergence`, a future Phase-8-family-3 addition — has
no entry in any of the three sets. Failing closed on "unknown" is the right
default for a MEASURED quantity gone missing (a NaN indicator reading below is
exactly that, and DOES fail closed) but it is the WRONG default for a kind
this plug-in was never scoped to judge: `trend-context` only understands
chart-pattern families 1-2, and rejecting every event from an unrelated
detector that happens to share a graph with this Confirmation would silently
break strategies that mix pattern detectors with, say, `donchian-breakout`,
which has no prevailing-trend claim in its own right either. Pass-through
keeps the failure mode "this plug-in has nothing to say about this kind"
honestly reported (`reason` names the unknown kind) rather than disguised as
a trend rejection.

HOW THE TREND IS MEASURED, and why not from closes alone. The lookback window
(setup tier, `lookback` bars ending at the last setup bar closed before the
trigger bar opened — the same instant `macd_confirmation` reads, so momentum
and trend context share one clock) is split into a leading half and a
trailing half. Each half is reduced to one "level":

    level(half) = mean( (bar.high + bar.low) / 2  for bar in half )

i.e. the average of each bar's OWN high/low midpoint across the half — using
the full range every bar prints, not its close, so a decline that closes
mid-range each bar but prints lower lows on rejection wicks still counts
(the plan's OHLC mandate). The net move is

    net = (level(trailing) - level(leading)) / level(leading)

and `_trend_fraction(window, "up")` is `net`, `_trend_fraction(window, "down")`
is `-net`. THIS IS THE PROPERTY THAT MATTERS: "up" and "down" are exact
negatives of each other, so for any window and any `min_trend_frac > 0` AT
MOST ONE of the two can clear the gate
(`test_up_and_down_can_never_both_pass_the_same_window` pins this over a set
of adversarial windows).

REJECTED FORMULATION, kept here because it shipped briefly and was wrong: an
earlier version measured "up" as
`(trailing.high.max() - leading.low.min()) / leading.low.min()` and "down" as
the mirror using `leading.high.max()` / `trailing.low.min()`. Those are two
INDEPENDENT one-sided reaches, not a single directional measurement, and nothing
stops both from clearing a positive threshold on the SAME bars — e.g. a window
whose leading half ranges [90, 100] and trailing half ranges [95, 110] scores
`up = (110-90)/90 = 0.222` AND `down = (100-95)/100 = 0.050`, both comfortably
past this plug-in's own default `min_trend_frac=0.03`. A gate that confirms a
prior uptrend and a prior downtrend on the same window is measuring volatility,
not trend — precisely the defect class (D1/D2/D3) this whole plan exists to
close, and it would have been weakest exactly in the choppy conditions where a
directional gate matters most. The mean-midpoint / net-displacement formulation
above still reads high and low (never close) but combines them into ONE signed
number per window, which is what makes the two directions mutually exclusive
rather than independently satisfiable.

NaN / INSUFFICIENT-HISTORY REJECTS (fail closed), unconditionally, with no
`require_defined` escape hatch (unlike `volume-breakout`, which has one for a
stated legacy-parity reason that does not apply to a brand-new plug-in).
Fewer than `lookback` setup bars closed, or a non-positive divisor (a base
price of zero or less, which real OHLCV never produces but a synthetic test
fixture could), reads as "trend unknown", never "trend confirmed" — the same
idiom `macd.py` and `volume_breakout.py` both use.

THE DEFAULTS (`lookback=20`, `min_trend_frac=0.03`) ARE UNMEASURED STARTING
POINTS, NOT CALIBRATED VALUES — stated plainly rather than presented as a
finding. On the 4h setup tier, `lookback=20` is 3.3 days of history, and
`min_trend_frac=0.03` is a 3% net move over that stretch; on an asset with
BTC's typical 4h volatility a 3% directional drift over 3.3 days is common in
EITHER direction, so at these defaults the gate may reject only the most
extreme counter-trend cases (exactly D1's +49%/20-bar rally) and pass through
most ordinary chop unchallenged. That is a real risk of the gate being close
to toothless at its shipped settings. Fixing it requires an ablation
(`volume_breakout.py`'s own argument: "a gate's cost is a trade-count and
expectancy delta from one ablation run") that is out of scope for registering
the plug-in, and both numbers stay swept `ParamSpec`s specifically so that
measurement can happen without a code change.

WHY EVERY DEFAULT BELOW IS AN INLINE LITERAL, NOT `config.PY_SOMETHING`. This
is the `rising-wedge` precedent (`continuation.py:346-361`): duplicating that
proof here is deliberate, not an oversight. `lookback` and `min_trend_frac`
have no existing `config.py` constant to mirror (this Confirmation is new, so
there is nothing to duplicate FROM) and, this cycle, `config.py` is owned by
WS-A's `reversal.py` hardening — a shared edit here would be a merge conflict
for no functional gain. The one exception is the `timeframes=` metadata below,
which is not a default and carries no threshold: `config.SIGNAL_PATTERN_TIMEFRAME`
identifies WHICH tier this plug-in reads (exactly as `macd.py` and
`volume_breakout.py` already declare their own `timeframes=`), and hardcoding
the literal "4h" in its place would be the tier-name anti-pattern contract §8
forbids ("tier-derived, never hardcoded"), not a step toward the zero-config-read
proof. `_required_trend_direction`, `_trend_fraction` and every `ParamSpec`
default remain literal.

NOT WIRED INTO ANY DEFAULT STRATEGY GRAPH. Registering this plug-in makes it
available to any graph that names it; deciding WHICH graphs should gate on it
is a separate, measured decision (an ablation run, per `volume_breakout.py`'s
own argument for why a gate's cost must be measured) and is out of scope here.
"""

import logging
import math

import pandas as pd

from trading_bot import config
from trading_bot.framework.contracts import ConfirmationVerdict, ParamSpec
from trading_bot.framework.registry import register

logger = logging.getLogger("trading_bot")

TREND_CONTEXT_NAME = "confirmation.trend-context"

# The thirteen catalog chart-pattern kinds this plug-in understands, taken
# verbatim from the `@register(name=...)` calls in reversal.py and
# continuation.py (cross-checked against catalog.py families 1-2). Anything
# not listed in one of these three sets is UNKNOWN (see module docstring).

# Reversal kinds: the pattern reverses a prior move, so it needs the OPPOSITE
# of event.direction's own bias as the prior trend.
REVERSAL_KINDS = frozenset(
    {
        "double-top",
        "double-bottom",
        "head-and-shoulders",
        "inverse-head-and-shoulders",
        "cup-and-handle",
        "inverse-cup-and-handle",
    }
)

# Continuation kinds: the pattern continues a prior move, so it needs the SAME
# bias as event.direction as the prior trend.
CONTINUATION_KINDS = frozenset(
    {
        "bull-flag",
        "bear-flag",
        "ascending-triangle",
        "descending-triangle",
        "falling-wedge",
        "rising-wedge",
    }
)

# Registered chart-pattern kinds with NO single prevailing-trend claim to
# check — see the module docstring's SYMMETRICAL TRIANGLE section. Passes
# unconditionally.
NO_TREND_CLAIM_KINDS = frozenset({"symmetrical-triangle"})

# Every kind this plug-in has an opinion about. Used only by tests to prove
# the three sets above are disjoint and match the registry.
KNOWN_KINDS = REVERSAL_KINDS | CONTINUATION_KINDS | NO_TREND_CLAIM_KINDS


def _required_trend_direction(kind: str, direction: str) -> str | None:
    """"up", "down", or None (no claim to check: unknown kind or NO_TREND_CLAIM_KINDS).

    Pure lookup, enumerable by a test — no substring matching on `kind`.
    """
    if kind in NO_TREND_CLAIM_KINDS or kind not in KNOWN_KINDS:
        return None
    own_bias = "up" if direction == "long" else "down"
    if kind in CONTINUATION_KINDS:
        return own_bias
    # kind in REVERSAL_KINDS: the opposite of the event's own bias.
    return "down" if own_bias == "up" else "up"


def _level(half: pd.DataFrame) -> float:
    """Mean high/low midpoint across `half` — the single number that stands in
    for "where price was" over that stretch, built from every bar's own range
    rather than its close (see module docstring's HOW THE TREND IS MEASURED)."""
    return float(((half["high"] + half["low"]) / 2.0).mean())


def _trend_fraction(window: pd.DataFrame, wanted: str) -> float:
    """Signed net directional displacement of `window`, read as "up" or "down".

    `window` is split into a leading and trailing half; `net` is the trailing
    half's level minus the leading half's level, as a fraction of the leading
    level. Returns `net` for `wanted == "up"` and `-net` for `wanted == "down"`
    — this sign symmetry is what makes the two directions MUTUALLY EXCLUSIVE
    for any `min_trend_frac > 0` (see module docstring's REJECTED FORMULATION
    paragraph for the bug this replaced, and
    `test_up_and_down_can_never_both_pass_the_same_window` for the pin).

    Returns NaN when the leading level is non-positive — real OHLCV never
    produces that, but it is the correct "undefined" reading if it ever did,
    per this plug-in's fail-closed idiom.
    """
    n = len(window)
    half = max(1, n // 2)
    leading = window.iloc[:half]
    trailing = window.iloc[-half:]
    leading_level = _level(leading)
    if not (leading_level > 0.0):
        return float("nan")
    net = (_level(trailing) - leading_level) / leading_level
    return net if wanted == "up" else -net


@register(
    "confirmation",
    name="trend-context",
    params={
        "lookback": ParamSpec(
            kind="int",
            default=20,
            bounds=(4, 200),
            doc="Setup-tier bars examined for the prior trend, ending at the "
            "last bar closed before the trigger",
        ),
        "min_trend_frac": ParamSpec(
            kind="float",
            default=0.03,
            bounds=(0.0, 1.0),
            doc="Minimum measured high/low trend magnitude (fraction of "
            "price) required in the direction the pattern's kind claims to "
            "need",
        ),
    },
    rationale=(
        "A reversal pattern that reverses nothing, or a continuation pattern "
        "with no prior move to continue, is not the shape its name claims — "
        "the systemic gap three detector docstrings deferred to 'a "
        "Confirmation's job (Phase 4)' and nobody built. This plug-in reads "
        "the required direction off an explicit kind lookup (reversal wants "
        "the opposite of the event's own bias, continuation wants the same) "
        "and measures it from setup-tier OHLC extremes, not closes, so "
        "rejection wicks count."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
)
def trend_context(ctx, event, *, lookback: int, min_trend_frac: float):
    """Gate an event on whether its pattern class's claimed prior trend is real.

    Never mutates `event` (contract §3). `score` carries the measured trend
    fraction in every branch that measured one — the same discipline
    `macd_confirmation` and `volume_breakout` follow — and is `nan` on the two
    pass-through paths (no-claim kind, unknown kind) where nothing was
    measured, so a `nan` score is distinguishable from a poorly-trending 0.0.

    Args:
        ctx: EvalContext bound to the trigger bar's open, so
            `ctx.window(setup_tf, lookback)` is the `lookback` setup bars that
            had already closed when the trigger bar opened — the same instant
            `macd_confirmation` reads (MEDIUM-2).
        event: The DetectedEvent under evaluation.
        lookback: Setup-tier bars examined.
        min_trend_frac: Minimum required trend magnitude, fraction of price.

    Returns:
        ConfirmationVerdict.
    """
    wanted = _required_trend_direction(event.kind, event.direction)
    if wanted is None:
        reason = (
            f"no prior-trend claim to check for kind={event.kind!r} "
            "(symmetrical-triangle names no single prevailing trend, and any "
            "other unlisted kind is out of this plug-in's scope) — passing "
            "through"
        )
        logger.debug(
            "%s %s %s trend-context pass-through: %s",
            ctx.symbol,
            event.kind,
            event.direction,
            reason,
        )
        return ConfirmationVerdict(
            passed=True, name=TREND_CONTEXT_NAME, score=float("nan"), reason=reason
        )

    setup_tf = config.SIGNAL_PATTERN_TIMEFRAME
    window = ctx.window(setup_tf, lookback)

    if len(window) < lookback:
        reason = (
            f"insufficient setup history: needs {lookback} bars, saw "
            f"{len(window)} (warmup)"
        )
        logger.debug(
            "%s %s %s rejected: %s", ctx.symbol, event.kind, event.direction, reason
        )
        return ConfirmationVerdict(
            passed=False, name=TREND_CONTEXT_NAME, score=float("nan"), reason=reason
        )

    fraction = _trend_fraction(window, wanted)
    if not math.isfinite(fraction):
        reason = (
            f"prior {wanted}-trend undefined over the last {lookback} setup "
            "bars (non-positive base price)"
        )
        logger.debug(
            "%s %s %s rejected: %s", ctx.symbol, event.kind, event.direction, reason
        )
        return ConfirmationVerdict(
            passed=False, name=TREND_CONTEXT_NAME, score=fraction, reason=reason
        )

    passed = fraction >= min_trend_frac
    cls = "reversal" if event.kind in REVERSAL_KINDS else "continuation"
    reason = (
        f"{event.kind} ({cls}, {event.direction}) needs a prior {wanted}-trend "
        f">= {min_trend_frac:.4f}; measured {fraction:.4f} over the last "
        f"{lookback} setup bars"
    )
    if not passed:
        logger.debug(
            "%s %s %s rejected: %s", ctx.symbol, event.kind, event.direction, reason
        )
    return ConfirmationVerdict(
        passed=passed, name=TREND_CONTEXT_NAME, score=fraction, reason=reason
    )
