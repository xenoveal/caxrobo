"""
reviewer.trade-quality — the Reviewer plug-in (v0.3.0 Phase 5, contract §3).

THE ORACLE BOUNDARY (contract §4, restated): this module never runs a
backtest and never returns a number that ranks a strategy. It measures a
CLOSED trade against its own prediction (was the TP the best exit available?
was the SL over-conservative? did this trade's pace clear the northstar?) and
returns a ReviewRecord whose fields are per-axis diagnostics, never combined.

IMPORT BAN (test-enforced): this module may not import backtest.walkforward,
backtest.engine, backtest.trials, framework.execute, or anything under
evolution/. `trade` is accepted DUCK-TYPED — attribute access only
(symbol/regime/pattern/direction/entry_ts/entry/stop/target/exit_ts/
exit_price/outcome/pnl_pct/planned_rr/confirmations/strategy_version) — which
is what makes the ban possible: this module cannot construct a Trade, so it
cannot cause one to exist.

A3 — POST-HOC, THEREFORE NOT LOOKAHEAD. Legitimate only because nothing in
this review path can reach an entry decision: this package cannot produce
trades (the import ban above), the reviewer runs on ALREADY-CLOSED trades, and
no Detector/Confirmation/Policy/Filter imports `feedback` or `plugins.reviewers`
(a one-way dependency). If that ever stops being true, every verdict here
becomes a lookahead bug.

THE EXCURSION WINDOW (A2), the phase's most likely defect: the window is the
bar set the ENGINE'S EXIT LOOP ACTUALLY EVALUATED. engine.py:388's
`if j <= open_trade["entry_j"]: continue` skips the entry bar, because a
trade fills at the entry bar's CLOSE and its range mostly precedes entry —
including it would count pre-entry intra-bar movement as excursion and bias
every TP verdict toward "left money". So this module's window is
(entry_ts, exit_ts] in bar-OPEN terms: `entry_ts + interval` through
`exit_ts` inclusive, both bounds passed to storage.load_candles (which is
itself both-bounds-inclusive) — that reproduces exactly the bars
`j > entry_j ... j_exit` of the exit loop. The exit bar IS included: it was
evaluated (that is how the trade closed).

TIER CHOICE (A1). Excursions are measured on the TRIGGER tier
(config.SIGNAL_TRIGGER_TIMEFRAME, resolved via config.REVIEW_TIMEFRAME or that
default, AT CALL TIME never at import time) — the tier the exit loop actually
evaluates. A finer tier judges the engine against exits it never had; a
coarser tier hides excursions it saw.

No pandas: every ratio here is written as explicit arithmetic
(`(x - entry) / entry`), never a pandas percentage-change helper — this phase
is clear of pandas 3.0.3's unsafe Series.pct_change() by construction, the
same reason backtest/equity.py avoids it.
"""

import logging
import statistics

from trading_bot import config
from trading_bot.data import storage
from trading_bot.feedback.records import _record_id
from trading_bot.feedback.records import ReviewRecord
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.registry import register

logger = logging.getLogger("trading_bot")

TRADE_QUALITY_KEY = "reviewer.trade-quality"


def _load_window(ctx, trade) -> tuple[list, int, int]:
    """The bars the engine's exit loop evaluated for this trade — see module
    docstring's window definition and engine.py:388 citation.

    Returns:
        (rows, bars_expected, interval_ms). rows is [] when bars_expected is 0
        (no interval mismatch check is meaningful on an empty window).

    Raises:
        ValueError: If the loaded window's median bar spacing does not match
            interval_ms — the INTERVAL_ASSERTION guarantee (contract §1) every
            new data path must keep. A single missing candle must not raise
            (that is bar-coverage's job below); a wholesale mismatch must.
    """
    review_tf = ctx.review_tf
    interval = storage.TIMEFRAME_MS[review_tf]
    bars_expected = max(0, (trade.exit_ts - trade.entry_ts) // interval)
    if bars_expected == 0:
        return [], 0, interval

    start = trade.entry_ts + interval  # exclude the entry bar (A2)
    end = trade.exit_ts  # inclusive: the exit bar IS evaluated
    rows = storage.load_candles(ctx.conn, trade.symbol, review_tf, start_ms=start, end_ms=end)

    if len(rows) >= 2:
        ts = [r[0] for r in rows]
        diffs = [b - a for a, b in zip(ts, ts[1:])]
        observed = statistics.median(diffs)
        if observed != interval:
            raise ValueError(
                f"{trade.symbol} review window spacing is {observed} ms but "
                f"config names review_tf {review_tf!r} ({interval} ms); "
                f"entry_ts={trade.entry_ts} exit_ts={trade.exit_ts}"
            )
    return rows, bars_expected, interval


def measure_excursions(ctx, trade) -> tuple[float | None, float | None, int, int]:
    """Maximum favourable / adverse excursion over the review window, as
    POSITIVE fractions of entry (never went favourable = 0.0, not negative).

    Returns:
        (mfe_pct, mae_pct, bars_reviewed, bars_expected). Both pct values are
        None — never NaN, never 0.0-as-missing — when bars_expected is 0 or
        coverage falls below config.REVIEW_MIN_BAR_COVERAGE (contract-style
        None-for-undefined; a gap understates both and an understated MAE
        reads as "the stop was over-wide", exactly backwards).
    """
    rows, bars_expected, _interval = _load_window(ctx, trade)
    bars_reviewed = len(rows)

    if bars_expected == 0 or bars_reviewed == 0 or (
        bars_reviewed / bars_expected < config.REVIEW_MIN_BAR_COVERAGE
    ):
        coverage = bars_reviewed / bars_expected if bars_expected else 0.0
        logger.warning(
            "%s entry_ts=%s: insufficient bar coverage %d/%d (%.2f < %.2f); "
            "excursions undefined",
            trade.symbol, trade.entry_ts, bars_reviewed, bars_expected,
            coverage, config.REVIEW_MIN_BAR_COVERAGE,
        )
        return None, None, bars_reviewed, bars_expected

    sign = 1.0 if trade.direction == "long" else -1.0
    mfe = 0.0
    mae = 0.0
    for _ts, _o, high, low, _c, _v in rows:
        fav_extreme = high if trade.direction == "long" else low
        adv_extreme = low if trade.direction == "long" else high
        mfe = max(mfe, sign * (fav_extreme - trade.entry) / trade.entry)
        mae = max(mae, -sign * (adv_extreme - trade.entry) / trade.entry)
    return max(0.0, mfe), max(0.0, mae), bars_reviewed, bars_expected


@register(
    "reviewer",
    name="trade-quality",
    params={
        "tp_capture_good": ParamSpec(
            kind="float", default=config.REVIEW_TP_CAPTURE_GOOD, bounds=(0.1, 0.9),
            doc="TP capture ratio (realized / MFE) at or above which an exit reads 'good'",
        ),
        "sl_slack_max": ParamSpec(
            kind="float", default=config.REVIEW_SL_SLACK_MAX, bounds=(0.1, 0.9),
            doc="MAE/stop-distance ratio below which a stop reads 'over-wide'",
        ),
    },
    rationale=(
        "v0.2.0 produced trades and no record of whether their exits were any good; "
        "KNOWN-LIMITATIONS §0c shows the search never learned from them. Measures TP "
        "capture against max favourable excursion, SL headroom against max adverse "
        "excursion, and pace against TARGET_ANN_RETURN — diagnostics only; THE GATE "
        "decides. Exposing tp_capture_good/sl_slack_max lets a Mutator jitter the "
        "REVIEW THRESHOLDS (which labels change), never the fitness (which strategy wins)."
    ),
)
def review(
    trade, context, *,
    tp_capture_good: float = config.REVIEW_TP_CAPTURE_GOOD,
    sl_slack_max: float = config.REVIEW_SL_SLACK_MAX,
) -> ReviewRecord:
    """Assemble one closed trade's ReviewRecord. Pure — no DB writes (A7);
    the caller (feedback.protocol.run_loop_iteration) persists via
    feedback.records.insert_records.
    """
    mfe_pct, mae_pct, bars_reviewed, bars_expected = measure_excursions(context, trade)

    sign = 1.0 if trade.direction == "long" else -1.0
    realized_pct = sign * (trade.exit_price - trade.entry) / trade.entry
    target_distance_pct = abs(trade.target - trade.entry) / trade.entry
    stop_distance_pct = abs(trade.entry - trade.stop) / trade.entry

    tp_capture_ratio = (
        None if (mfe_pct is None or mfe_pct <= 0) else max(0.0, realized_pct) / mfe_pct
    )
    sl_headroom_ratio = (
        None if (mae_pct is None or stop_distance_pct <= 0) else mae_pct / stop_distance_pct
    )

    if mfe_pct is None:
        tp_verdict = "n/a"
    elif mfe_pct <= 0:
        tp_verdict = "never-favoured"  # an entry problem, not a TP problem
    elif mfe_pct < target_distance_pct * config.REVIEW_TP_UNREACHABLE_FRAC:
        tp_verdict = "target-too-far"
    elif tp_capture_ratio is not None and tp_capture_ratio < tp_capture_good:
        tp_verdict = "left-money"
    else:
        tp_verdict = "good"

    if sl_headroom_ratio is None:
        sl_verdict = "n/a"
    elif trade.outcome in ("stop", "trail"):
        sl_verdict = "hit"  # whether it SHOULD have bound is a post-exit
        # question this phase refuses (no counterfactuals — see NOT Building)
    elif sl_headroom_ratio < sl_slack_max:
        sl_verdict = "over-wide"
    elif sl_headroom_ratio >= config.REVIEW_SL_NEAR_MISS:
        sl_verdict = "tight"
    else:
        sl_verdict = "ok"

    # Pace — the honest per-trade version and nothing more.
    # pace_ratio is a PER-TRADE DIAGNOSTIC WITH NO STATISTICAL SIGNIFICANCE and
    # it ignores idle capital — at most one position per symbol is held, so a
    # portfolio of pace_ratio > 1 trades does NOT imply >50% annual. The only
    # honest pace statement in this codebase is diagnose()'s, computed by
    # equity.compute_equity_metrics over a calendar-complete series with a
    # sample-adequacy flag. Annualising a single trade over its holding days
    # is the error KNOWN-LIMITATIONS §2 records.
    holding_days = (trade.exit_ts - trade.entry_ts) / 86_400_000.0  # engine.py:359
    required = None
    if holding_days > 0:
        # Required return over the SAME holding days at the northstar rate,
        # COMPOUNDED — consistent with equity.compute_equity_metrics'
        # equity ** (365/n) - 1. ~18% LESS demanding than linear pro-rata at
        # 30 days; chosen for consistency with the aggregate path, not for
        # flattery, and recorded so it is mistaken for neither.
        required = (1.0 + context.target_ann_return) ** (holding_days / 365.0) - 1.0
    pace_ratio = (
        None if (required is None or required <= 0) else trade.pnl_pct / required
    )

    if pace_ratio is None:
        pace_verdict = "n/a"
    elif trade.pnl_pct <= 0:
        pace_verdict = "loss"
    elif pace_ratio >= 1:
        pace_verdict = "on-pace"
    elif pace_ratio > 0:
        pace_verdict = "behind"
    else:  # pragma: no cover - unreachable given the guards above
        pace_verdict = "n/a"

    confirmations = tuple(getattr(trade, "confirmations", ()) or ())
    planned_rr = getattr(trade, "planned_rr", None)
    coverage = bars_reviewed / bars_expected if bars_expected else 0.0
    notes = f"mfe_bars={bars_reviewed} bars_expected={bars_expected} coverage={coverage:.2f}"

    record_id = _record_id(
        context.strategy_version, TRADE_QUALITY_KEY, context.span_class,
        trade.symbol, trade.entry_ts, trade.exit_ts, context.review_tf,
    )

    return ReviewRecord(
        record_id=record_id,
        strategy_version=context.strategy_version,
        reviewer=TRADE_QUALITY_KEY,
        span_class=context.span_class,
        symbol=trade.symbol,
        regime=trade.regime,
        pattern=trade.pattern,
        direction=trade.direction,
        outcome=trade.outcome,
        entry_ts=trade.entry_ts,
        exit_ts=trade.exit_ts,
        created_ts=context.created_ts,
        entry=trade.entry,
        stop=trade.stop,
        target=trade.target,
        exit_price=trade.exit_price,
        pnl_pct=trade.pnl_pct,
        holding_days=holding_days,
        review_tf=context.review_tf,
        bars_reviewed=bars_reviewed,
        bars_expected=bars_expected,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        target_distance_pct=target_distance_pct,
        stop_distance_pct=stop_distance_pct,
        tp_capture_ratio=tp_capture_ratio,
        tp_verdict=tp_verdict,
        sl_headroom_ratio=sl_headroom_ratio,
        sl_verdict=sl_verdict,
        pace_ratio=pace_ratio,
        pace_verdict=pace_verdict,
        confirmations=confirmations,
        planned_rr=planned_rr,
        notes=notes,
    )
