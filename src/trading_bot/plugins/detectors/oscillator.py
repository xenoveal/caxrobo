"""
Catalog family 14 (Oscillator Signals), RSI subsection: `rsi-divergence`.

FOUR of the six RSI rows — Bullish Divergence, Bearish Divergence, Hidden
Bullish Divergence, Hidden Bearish Divergence — served by ONE detector with a
`mode` choice parameter and a direction derived from which side diverges.
`RSI > 70` and `RSI < 30` are `out-of-scope` in
`plugins/detectors/catalog.py` with the reason that they are Confirmation
material, not tradeable events: an overbought reading has no level and no
target, so it cannot become a `PositionPlan` on its own.

WHY THIS ONE MATTERS MOST. KNOWN-LIMITATIONS §0c records RSI and divergence as
**absent entirely** from every v0.2.0 sweep — not measured and rejected,
never expressible. Price making a new extreme while momentum does not is the
classical exhaustion read, and the intervening swing gives it a level, so the
event is a structure break rather than an opinion.

MACD divergence is deliberately NOT here: MACD Cross is contract §9 tier 3 and
`indicators/macd.py` is Phase 4's file. `macd-divergence` is a ledger row, not
a task.

ONE FRACTAL IMPLEMENTATION, NOT TWO. Divergence needs pivots in the oscillator
as well as in price. Rather than write a second swing-finder that can drift from
`signals.pivots.find_pivots`, `indicators.rsi.rsi_frame` presents the RSI as a
degenerate OHLCV frame and the SAME function runs over it — same strict
domination, same confirmation lag, same tie rejection.
"""

import logging

import numpy as np

from trading_bot import config
from trading_bot.framework.contracts import DetectedEvent, ParamSpec
from trading_bot.framework.registry import register
from trading_bot.indicators.rsi import rsi_frame
from trading_bot.plugins.detectors import _geometry as g
from trading_bot.signals.pivots import find_pivots

logger = logging.getLogger("trading_bot")

SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
DIVERGENCE_MODES = ("regular", "hidden")


@register(
    "detector",
    name="rsi-divergence",
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
            doc="Freshness bound on the second price pivot (setup bars)",
        ),
        "lookback_bars": ParamSpec(
            kind="int",
            default=config.RSI_DIV_LOOKBACK_BARS,
            bounds=(40, 400),
            doc="Setup bars scanned, and the window the RSI recursion runs over",
        ),
        "rsi_period": ParamSpec(
            kind="int",
            default=config.RSI_PERIOD,
            bounds=(2, 50),
            doc="Wilder RSI period",
        ),
        "min_separation_bars": ParamSpec(
            kind="int",
            default=config.RSI_DIV_MIN_SEPARATION_BARS,
            bounds=(2, 60),
            doc="Min bars between the two price pivots",
        ),
        "max_separation_bars": ParamSpec(
            kind="int",
            default=config.RSI_DIV_MAX_SEPARATION_BARS,
            bounds=(5, 200),
            doc="Max bars between the two price pivots",
        ),
        "pivot_match_bars": ParamSpec(
            kind="int",
            default=config.RSI_DIV_PIVOT_MATCH_BARS,
            bounds=(0, 12),
            doc="An RSI pivot must sit within this many bars of its price pivot",
        ),
        "overbought": ParamSpec(
            kind="float",
            default=config.RSI_DIV_OVERBOUGHT,
            bounds=(50.0, 95.0),
            doc="The more extreme of the two RSI highs must be at least this",
        ),
        "oversold": ParamSpec(
            kind="float",
            default=config.RSI_DIV_OVERSOLD,
            bounds=(5.0, 50.0),
            doc="The more extreme of the two RSI lows must be at most this",
        ),
        "mode": ParamSpec(
            kind="choice",
            default="regular",
            choices=DIVERGENCE_MODES,
            doc="regular = classical divergence; hidden = the continuation variant",
        ),
    },
    rationale=(
        "Contract §9 tier 2 (the catalog rates RSI Divergence four stars) and "
        "the single largest hole in v0.2.0's search: KNOWN-LIMITATIONS §0c "
        "records RSI and divergence as ABSENT ENTIRELY from every sweep, so "
        "this is not a re-test of something already found wanting. Price making "
        "a new extreme while momentum does not is the classical exhaustion "
        "read; the intervening swing is the confirmation level, which makes the "
        "event a structure break rather than an opinion. Overbought/oversold "
        "default to 60/40 rather than 70/30 because pairing two RSI highs only "
        "needs the extreme one ELEVATED, and a 70 floor eliminates most real "
        "pairs; both are bounded ParamSpecs so that choice stays testable."
    ),
    timeframes=(SETUP_TF,),
    tier=2,
)
def rsi_divergence(
    ctx,
    *,
    pivot_span: int,
    max_age_bars: int,
    lookback_bars: int,
    rsi_period: int,
    min_separation_bars: int,
    max_separation_bars: int,
    pivot_match_bars: int,
    overbought: float,
    oversold: float,
    mode: str,
) -> list[DetectedEvent]:
    """Bullish and bearish RSI divergence, regular or hidden.

    DEFINITIONS USED (the classical ones):
      * regular bearish — HIGHER price high, LOWER RSI high. Short.
      * regular bullish — LOWER price low, HIGHER RSI low. Long.
      * hidden bearish — LOWER price high, HIGHER RSI high. Short.
      * hidden bullish — HIGHER price low, LOWER RSI low. Long.

    (The plan's prose said `mode="hidden"` "reverses the price inequality only"
    and then defined hidden bearish as "lower price high with a higher RSI
    high", which reverses BOTH. The parenthetical is the classical definition
    and is what is implemented; the deviation is recorded in the phase report.)

    LEVEL AND TARGET ARE A STATED CONVENTION, not a classical measured move.
    `level` is the price of the intervening opposite-kind PRICE pivot between
    the two — for a bearish event, the swing LOW. That is the structure which
    must break for the reversal to be more than a wiggle, i.e. exactly the
    logical role an H&S neckline plays, and it makes the event triggerable by
    the same `check_breakout`. `target_height = |second pivot - level|`.
    REJECTED ALTERNATIVE: an ATR multiple, which would put a fitted parameter
    inside a detector.

    WHERE THE LOOKAHEAD HIDES, and the rule that closes it. TWO pivots are
    involved — one in price, one in the oscillator — and they can confirm on
    DIFFERENT bars, because each is only knowable `pivot_span` closed bars after
    its own bar (pivots.py:9-12). `end_ts` is therefore
    `confirmation_ts(max(price_pivot, osc_pivot))`, the LATER of the two. Taking
    the price pivot alone would let `check_breakout` fire on a bar where the RSI
    pivot was not yet knowable — a lookahead that *improves* the backtest.
    Pinned by `test_end_ts_is_the_later_of_the_two_pivot_confirmations`.

    A NOTE ON PLATEAUS, so nobody "fixes" it. `find_pivots` rejects ties
    (pivots.py:5-6, 81-84), and RSI sits at exactly 100.0 through a pure
    uptrend, so NO RSI pivot is emitted there. That is correct — a flat
    oscillator has no swing — and it is asserted by
    `test_rsi_plateau_yields_no_oscillator_pivot` so a future loosening to `>=`
    is caught.

    WARMUP is sliced off by `rsi_frame`, never filled: NaN makes every
    comparison False, so leaving it in would silently suppress pivots near the
    start of the series while looking like a geometry problem.

    The RSI recursion runs over the `lookback_bars` window, not over full
    history, and a Wilder recursion over a window is NOT the tail of one over
    full history — the same property `framework/context.py` records for ADX.
    That makes `lookback_bars` a real parameter, which is why it is declared.
    """
    df = ctx.window(SETUP_TF, lookback_bars)
    out: list[DetectedEvent] = []
    n = len(df)
    if n == 0:
        return out

    osc = rsi_frame(df["close"], period=rsi_period)
    if len(osc) == 0:
        return out

    fresh = g.fresh_factory(n, max_age_bars)
    price_pivots = g.collapse_runs(find_pivots(df, span=pivot_span))
    osc_pivots = g.collapse_runs(find_pivots(osc, span=pivot_span))
    if not price_pivots or not osc_pivots:
        return out

    # Oscillator pivot indices are positional within `osc`, which had its warmup
    # rows dropped. Map through the shared epoch-ms timestamps so both sides
    # speak the same coordinate system as `df`.
    pos_of_ts = {int(ts): i for i, ts in enumerate(df.index.to_numpy())}
    osc_by_kind: dict[str, list[tuple[int, float]]] = {"high": [], "low": []}
    for p in osc_pivots:
        pos = pos_of_ts.get(int(p.ts))
        if pos is not None:
            osc_by_kind[p.kind].append((pos, float(p.price)))

    for bearish in (True, False):
        kind_wanted = "high" if bearish else "low"
        direction = "short" if bearish else "long"
        pivots = [p for p in price_pivots if p.kind == kind_wanted]
        others = [p for p in price_pivots if p.kind != kind_wanted]
        osc_side = osc_by_kind[kind_wanted]

        for i_p, j_p in zip(pivots, pivots[1:]):
            sep = j_p.index - i_p.index
            if not (min_separation_bars <= sep <= max_separation_bars):
                continue

            # Price inequality.
            if mode == "regular":
                price_ok = j_p.price > i_p.price if bearish else j_p.price < i_p.price
            else:
                price_ok = j_p.price < i_p.price if bearish else j_p.price > i_p.price
            if not price_ok:
                continue

            oi = _nearest(osc_side, i_p.index, pivot_match_bars)
            oj = _nearest(osc_side, j_p.index, pivot_match_bars)
            if oi is None or oj is None or oi[0] == oj[0]:
                continue

            # Oscillator inequality: opposite to price for regular divergence,
            # the SAME direction as price for hidden divergence.
            if mode == "regular":
                osc_ok = oj[1] < oi[1] if bearish else oj[1] > oi[1]
            else:
                osc_ok = oj[1] > oi[1] if bearish else oj[1] < oi[1]
            if not osc_ok:
                continue

            # The more extreme of the two readings must be elevated / depressed.
            # For mode="regular" this is identical to testing the FIRST reading,
            # because the second is less extreme by construction.
            if bearish:
                if max(oi[1], oj[1]) < overbought:
                    continue
            else:
                if min(oi[1], oj[1]) > oversold:
                    continue

            between = [m for m in others if i_p.index < m.index < j_p.index]
            if not between:
                continue
            m = (
                min(between, key=lambda p: p.price)
                if bearish
                else max(between, key=lambda p: p.price)
            )
            level = float(m.price)
            target_height = abs(float(j_p.price) - level)
            if target_height <= 0:
                continue
            if not fresh(j_p.index):
                continue
            if not g.level_unbroken(df, level, direction, j_p.index):
                continue

            out.append(
                DetectedEvent(
                    kind="rsi-divergence",
                    direction=direction,
                    level=level,
                    target_height=target_height,
                    start_ts=int(i_p.ts),
                    end_ts=g.confirmation_ts(
                        df, max(j_p.index, oj[0]), pivot_span
                    ),
                    meta={
                        "rsi_i": float(oi[1]),
                        "rsi_j": float(oj[1]),
                        "rsi_delta": float(oj[1] - oi[1]),
                        "price_delta_pct": float(
                            (j_p.price - i_p.price) / i_p.price
                        )
                        if i_p.price
                        else float("nan"),
                        "separation_bars": float(sep),
                        "mode": 1.0 if mode == "hidden" else 0.0,
                    },
                )
            )

    return g.dedupe_events(out)


def _nearest(
    candidates: list[tuple[int, float]], target: int, tol: int
) -> tuple[int, float] | None:
    """The (position, value) closest to ``target`` within ``tol`` bars, or None.

    Ties break toward the EARLIER bar via `np.argmin`'s first-minimum rule,
    which is the conservative choice: an earlier oscillator pivot confirms
    earlier, so it can never make `end_ts` optimistic.
    """
    if not candidates:
        return None
    dists = np.array([abs(pos - target) for pos, _ in candidates])
    k = int(np.argmin(dists))
    if dists[k] > tol:
        return None
    return candidates[k]
