"""
detector.macd-cross — a MACD histogram zero-crossing on the setup tier.

TIER 3 (***) in .claude/technical-pattern.md's reliability table, and contract §9
makes tiers 3-4 "a direction, not a gate": NO success criterion in Phase 4
depends on this detector having edge. It exists because the pivot guide's
Strategy Step 2 names MACD, and because the thin slice must prove the framework
composes MORE THAN ONE detector. Its edge is measured and reported, not assumed.

WHY A CROSS NEEDS A LEVEL AND A HEIGHT. `DetectedEvent` requires `level` (a price
a trigger bar must close beyond) and `target_height` (the measured move), and a
MACD cross has neither naturally. Both are chosen PARAMETER-FREE:

  - `level` = the crossing setup bar's HIGH (long) / LOW (short). The trigger bar
    must close beyond the extreme of the bar on which the cross completed. This
    makes the whole existing trigger machinery reusable UNCHANGED —
    check_breakout's fresh-crossing rule, its contiguity check, its volume
    computation — which is contract §3's stated reason for making DetectedEvent
    field-compatible with PatternCandidate.
  - `target_height` = the trailing DONCHIAN_ENTRY_PERIOD channel width
    (upper - lower) at the crossing bar. Rationale quoted from
    signals/donchian.py: "the 20-bar CHANNEL WIDTH — a derived quantity, the
    range the market has just resolved, adding no parameter beyond the canonical
    20."

REJECTED: `target_height = k * ATR`. That injects a fitted parameter into the
exact quantity the R:R filter measures, and fitting the target to clear the
filter is precisely the dishonesty config.py:162-174 guards against.

CONSEQUENCE THAT MAKES PHASE 4'S CENTRAL MEASUREMENT VALID: this detector and
detector.donchian-breakout produce the SAME reward and risk construction
(channel-width target, k*ATR stop), so their net-R:R distributions are directly
comparable and the survival measurement is not confounded by two different TP
conventions.

MACD IS COMPUTED OVER FULL HISTORY, not over a lookback window — see
indicators/macd.py's CONSEQUENCE paragraph. `EvalContext.series` memoizes one
causal array per (frame content, parameters) and hands back a read-only slice
truncated at the current bar, so the value at a bar never depends on how large a
window the caller happened to ask for. That differs from
signals/donchian.py's per-window Wilder ADX, deliberately: the ADX convention is
frozen by the parity gate, MACD is new and gets the better convention.
"""

import logging
import math

from trading_bot import config
from trading_bot.framework.contracts import DetectedEvent, ParamSpec
from trading_bot.framework.registry import register
from trading_bot.indicators.donchian import donchian
from trading_bot.indicators.macd import macd_hist

logger = logging.getLogger("trading_bot")

MACD_CROSS_KIND = "macd-cross"


def _channel_width(df, *, period: int):
    """Trailing Donchian channel width as an array, for EvalContext.series.

    `donchian()` EXCLUDES the current bar (it .shift(1)s), so the first defined
    value is at positional index `period`, not `period - 1`. Trailing-only, so it
    is legal through the series seam.
    """
    ch = donchian(df, period=period)
    return (ch["upper"] - ch["lower"]).to_numpy()


@register(
    "detector",
    name=MACD_CROSS_KIND,
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
            doc="Minimum |normalised histogram| for a cross to count",
        ),
        "entry_period": ParamSpec(
            kind="int",
            default=config.DONCHIAN_ENTRY_PERIOD,
            bounds=(5, 200),
            doc="Channel lookback supplying target_height (bars)",
        ),
    },
    rationale=(
        "MACD histogram zero-crossing (Appel 1979) on the setup tier: the fast "
        "EMA overtaking the slow one is the canonical momentum-regime change, and "
        "the pivot guide's Strategy Step 2 names it. HONEST PRIOR: MACD Cross is "
        "TIER 3 (***) in the pattern catalog's reliability table -- a lagging "
        "indicator on a derivative of price, prone to whipsaw in chop. No Phase 4 "
        "success criterion depends on it having edge; it is here to prove the "
        "graph composes more than one detector, and its measured edge is reported "
        "either way."
    ),
    timeframes=(config.SIGNAL_PATTERN_TIMEFRAME,),
    tier=3,
)
def macd_cross(
    ctx,
    *,
    fast: int,
    slow: int,
    signal: int,
    min_hist: float,
    entry_period: int,
) -> list:
    """Emit at most one MACD-cross event for the latest closed setup bar.

    Mirrors signals/donchian.py:65-140's shape end to end: warmup guard -> NaN
    guard -> threshold -> degenerate-geometry guard -> direction -> a single
    candidate with a DERIVED target_height and an interval-derived end_ts.

    At most ONE event, like detect_donchian_setups, so the executor's
    rank_signals stage has a well-defined single candidate per detector per bar.

    Returns:
        A list of 0 or 1 DetectedEvent. Long and short are mutually exclusive
        (the histogram cannot cross zero in both directions on one bar).
    """
    setup_tf = config.SIGNAL_PATTERN_TIMEFRAME

    # Warmup, derived from BOTH constraints. Do not assume MACD_MIN_BARS
    # dominates: donchian() excludes the current bar, so it needs entry_period+1.
    min_bars = max(slow + signal - 1, entry_period + 1)

    hist = ctx.series(setup_tf, "macd-hist", macd_hist, fast=fast, slow=slow, signal=signal)
    if len(hist) < min_bars or len(hist) < 2:
        return []

    prev, now = float(hist[-2]), float(hist[-1])
    # Fail closed on an undefined histogram (warmup). `not math.isfinite` catches
    # NaN and both infinities; a NaN comparison is always False, so the crossing
    # tests below would silently read as "no cross" rather than as "unknown".
    if not (math.isfinite(prev) and math.isfinite(now)):
        return []

    # Fresh-crossing shape, matching signals/breakout.py:124-127: strict on the
    # current bar, non-strict on the previous one, so one cross is one event and
    # there is no re-trigger while the histogram stays on one side.
    if prev <= 0.0 < now and now > min_hist:
        direction = "long"
    elif prev >= 0.0 > now and now < -min_hist:
        direction = "short"
    else:
        return []

    width_arr = ctx.series(setup_tf, "donchian-width", _channel_width, period=entry_period)
    if len(width_arr) == 0:
        return []
    width = float(width_arr[-1])
    if not (width > 0):  # NaN (warmup), zero or negative: degenerate channel
        return []

    window = ctx.window(setup_tf, entry_period + 1)
    if len(window) < 2:
        return []
    bar = window.iloc[-1]
    level = float(bar["high"]) if direction == "long" else float(bar["low"])
    if not (level > 0):
        return []

    ts_vals = window.index.to_numpy()
    interval = int(ts_vals[-1]) - int(ts_vals[-2])
    return [
        DetectedEvent(
            kind=MACD_CROSS_KIND,
            direction=direction,
            level=level,
            target_height=width,
            start_ts=int(ts_vals[0]),
            # end_ts is the setup bar's CLOSE time, so check_breakout's
            # `ts < event.end_ts` skip guarantees the trigger bar opens at or
            # after the setup bar closed — no intra-bar lookahead. Identical
            # convention to signals/donchian.py:127.
            end_ts=int(ts_vals[-1]) + interval,
            meta={
                "hist": now,
                "hist_prev": prev,
                "channel_width": width,
            },
        )
    ]
