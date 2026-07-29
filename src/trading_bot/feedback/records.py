"""
ReviewRecord / ReviewContext, the review_records table, and diagnose() —
v0.3.0 Phase 5 (shared architecture contract §3, §6).

THE ORACLE BOUNDARY (contract §4, restated here because this is the module
most likely to breach it): A REVIEW NEVER RUNS A BACKTEST. A REVIEW NEVER
RETURNS A NUMBER THAT RANKS A STRATEGY. Fitness comes only from
walkforward.walk_forward_pooled, charged to Phase 1's trial ledger.
Concretely: ReviewRecord has NO aggregate field (no fitness/score/reward/
objective/rank — every quantitative field carries units and an axis, and the
axes are never combined). diagnose()'s Diagnosis is a structured description,
not a ranking, and its one aggregate (pace) deliberately omits Sharpe/DSR,
which belong to the gate alone.

IMPORT BAN (test-enforced by test_feedback_records.py::TestOracleBoundary):
this module may not import backtest.walkforward, backtest.engine,
backtest.trials, framework.execute, or anything under evolution/. It consumes
already-closed trades (duck-typed — see below) and stored bars: it cannot
cause a trade to exist, therefore it cannot evaluate a strategy.
backtest.equity is a DELIBERATE EXCEPTION: compute_equity_metrics is pure
measurement over trades that already happened, consumes no trial, and is
needed by diagnose()'s pace figure.

TIMESTAMPS. Every *_ts column is epoch MILLISECONDS, UTC — never a formatted
date string (contract §6). holding_days is a derived float MEASUREMENT, not a
timestamp.

DUCK-TYPED Trade. Every function here accepts "trade" via attribute access
only (symbol/regime/pattern/direction/entry_ts/entry/stop/target/exit_ts/
exit_price/outcome/pnl_pct/planned_rr/confirmations/strategy_version) and
never imports backtest.engine.Trade's class. That absence is exactly what
makes the import ban enforceable: nothing here can construct a Trade, so
nothing here can cause one to exist.

CONFIRMATION COVERAGE IS UNMEASURABLE FOR A HARD GATE, AND THAT IS A MISSING
CAPABILITY, NOT A DEAD-WEIGHT FINDING. framework/execute.py's confirmation
loop (`if not verdict.passed: return None`) means a Trade exists only if
EVERY one of its confirmations passed, so `Trade.confirmations` contains all
of them on every closed trade a hard-gating Confirmation ever let through.
confirmation_coverage is therefore NECESSARILY ~1.0 for such a Confirmation
BY CONSTRUCTION — it measures the gate's own definition, not its usefulness —
and confirmation_pnl_delta is NECESSARILY None, because the "without it" group
is always empty: no rejected event survived to become a trade to compare
against. There is no counterfactual in the data; the gate destroyed it before
a Trade was ever constructed. Phase 4 flagged exactly this gap on its own
work ("the volume-ratio distribution among rejected events is not measured",
reports/phase4-thin-slice.md §H.2) — recording rejected (failed) confirmation
verdicts, not just passed ones, is the prerequisite capability for ever
deciding a hard-gating Confirmation's value honestly. Until that exists,
diagnose() labels full-coverage-with-no-delta confirmations "unmeasurable"
and refuses to suggest dropping them.

This is not a hypothetical risk: Phase 4 MEASURED what happens when you act
on coverage alone. Dropping BTCUSDT/ETHUSDT/SOLUSDT's volume+MACD
confirmations moves pooled trades 276 -> 477, expectancy_pct +0.2028% ->
-0.2523%, and pooled Sharpe +0.53 -> -0.86 (reports/phase4-thin-slice.md §B) —
the sleeve flips negative on every symbol. A `drop-confirmation:` suggestion
that fires on coverage alone would have recommended exactly that regression.

A genuine dead-weight verdict stays reachable once a counterfactual DOES
exist — e.g. a Confirmation declared on some branches of a graph but not
others, so trades from a branch that never carried it form a real "without"
group and confirmation_pnl_delta is not None even at high coverage. Contract
§3 anticipated this shape ("NEVER mutates the event" leaves room for a future
advisory Confirmation whose score varies while passed stays informational
elsewhere); diagnose() below decides "dead weight" ONLY on that decidable
case, never on coverage alone.
"""

import hashlib
import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Mapping

from trading_bot import config
from trading_bot.backtest.equity import compute_equity_metrics  # pure measurement,
# consumes no trial — see module docstring's IMPORT BAN exception.
from trading_bot.data import statestore

logger = logging.getLogger("trading_bot")

TABLE = "review_records"

SPAN_CLASSES = ("in-sample", "forward")

TP_VERDICTS = ("n/a", "never-favoured", "target-too-far", "left-money", "good")
SL_VERDICTS = ("n/a", "hit", "over-wide", "tight", "ok")
PACE_VERDICTS = ("n/a", "loss", "on-pace", "behind")

# Closed vocabulary (Task 5). Phase 6's mutators and Phase 7's UI consume this
# mechanically, so it can never become free text. Validate a suggestion string
# with `s.split(":", 1)[0] in SUGGESTIONS`.
SUGGESTIONS = (
    "insufficient-sample",
    "widen-target",
    "narrow-target",
    "widen-stop",
    "tighten-stop",
    "raise-max-hold",
    "lower-max-hold",
    "drop-confirmation",  # emitted parametrised: "drop-confirmation:<name>"
)


@dataclass(frozen=True)
class ReviewContext:
    """Everything one review BATCH shares — one value per call, not per trade.

    Attributes:
        conn: OHLCV connection (data/ohlcv.db). READ ONLY — nothing in
            feedback/ writes to it.
        review_tf: The timeframe excursions are measured on (A1). Recorded
            per-record too (`ReviewRecord.review_tf`) so a later tier shift
            cannot silently change what an old record means.
        strategy_version: The version being reviewed. Used for every record in
            the batch rather than trusting each trade's own
            (possibly-unstamped) `strategy_version`, so a batch is
            self-consistent even over legacy trades.
        span_class: One of SPAN_CLASSES. "in-sample" trades came from a span
            available to tuning; "forward" trades came from THE GATE via
            feedback.protocol.evaluate_forward. This label is part of the
            record's identity (see _record_id) and persists forever.
        target_ann_return: The northstar rate pace_ratio is measured against
            (config.TARGET_ANN_RETURN by default, injectable for tests).
        created_ts: Epoch ms, UTC. One value for the whole batch so every
            record in one review run shares an audit timestamp.
    """

    conn: object
    review_tf: str
    strategy_version: str
    span_class: str
    target_ann_return: float
    created_ts: int

    def __post_init__(self) -> None:
        if self.span_class not in SPAN_CLASSES:
            raise ValueError(
                f"span_class {self.span_class!r} must be one of {SPAN_CLASSES}"
            )


@dataclass(frozen=True)
class ReviewRecord:
    """One closed trade's per-axis review verdict. NO AGGREGATE FIELD —
    contract §4's structural lock #1. Fields are listed in the DDL's column
    order.
    """

    record_id: str
    strategy_version: str
    reviewer: str
    span_class: str
    symbol: str
    regime: str
    pattern: str
    direction: str
    outcome: str
    entry_ts: int
    exit_ts: int
    created_ts: int
    entry: float
    stop: float
    target: float
    exit_price: float
    pnl_pct: float
    holding_days: float
    review_tf: str
    bars_reviewed: int
    bars_expected: int
    mfe_pct: float | None
    mae_pct: float | None
    target_distance_pct: float
    stop_distance_pct: float
    tp_capture_ratio: float | None
    tp_verdict: str
    sl_headroom_ratio: float | None
    sl_verdict: str
    pace_ratio: float | None
    pace_verdict: str
    confirmations: tuple[str, ...] = ()
    planned_rr: float | None = None
    notes: str = ""


_COLUMNS = (
    "record_id", "strategy_version", "reviewer", "span_class", "symbol",
    "regime", "pattern", "direction", "outcome", "entry_ts", "exit_ts",
    "created_ts", "entry", "stop", "target", "exit_price", "pnl_pct",
    "holding_days", "review_tf", "bars_reviewed", "bars_expected",
    "mfe_pct", "mae_pct", "target_distance_pct", "stop_distance_pct",
    "tp_capture_ratio", "tp_verdict", "sl_headroom_ratio", "sl_verdict",
    "pace_ratio", "pace_verdict", "confirmations", "planned_rr", "notes",
)


def _record_id(
    strategy_version: str, reviewer: str, span_class: str, symbol: str,
    entry_ts: int, exit_ts: int, review_tf: str,
) -> str:
    """Content-derived id so re-reviewing the same trade is idempotent.

    span_class is part of the identity ON PURPOSE (Task 2): an in-sample
    diagnostic and a forward test of the SAME trade are different claims, and
    both must be able to survive in the table simultaneously.
    """
    payload = "|".join(
        str(x) for x in
        (strategy_version, reviewer, span_class, symbol, entry_ts, exit_ts, review_tf)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def ensure_schema(conn) -> None:
    """Create review_records and its indexes if absent (idempotent).

    DDL lives here, not in statestore.py: contract §6 gives each table's
    schema to the module that owns it. Uses statestore's shared state.db lock
    (NOT a second, module-local lock — state.db writes must serialize through
    ONE lock per database, mirroring trials.py's use of the same lock object).
    """
    with statestore._db_lock:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                record_id TEXT PRIMARY KEY,
                strategy_version TEXT NOT NULL, reviewer TEXT NOT NULL, span_class TEXT NOT NULL,
                symbol TEXT NOT NULL, regime TEXT NOT NULL, pattern TEXT NOT NULL,
                direction TEXT NOT NULL, outcome TEXT NOT NULL,
                entry_ts INTEGER NOT NULL, exit_ts INTEGER NOT NULL, created_ts INTEGER NOT NULL,
                entry REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL,
                exit_price REAL NOT NULL, pnl_pct REAL NOT NULL, holding_days REAL NOT NULL,
                review_tf TEXT NOT NULL, bars_reviewed INTEGER NOT NULL, bars_expected INTEGER NOT NULL,
                mfe_pct REAL, mae_pct REAL,
                target_distance_pct REAL NOT NULL, stop_distance_pct REAL NOT NULL,
                tp_capture_ratio REAL, tp_verdict TEXT NOT NULL,
                sl_headroom_ratio REAL, sl_verdict TEXT NOT NULL,
                pace_ratio REAL, pace_verdict TEXT NOT NULL,
                confirmations TEXT NOT NULL,
                planned_rr REAL,
                notes TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_review_version ON {TABLE} (strategy_version, exit_ts)"
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_review_exit ON {TABLE} (exit_ts)")
        conn.commit()


def _to_row(r: ReviewRecord) -> tuple:
    return (
        r.record_id, r.strategy_version, r.reviewer, r.span_class, r.symbol,
        r.regime, r.pattern, r.direction, r.outcome, r.entry_ts, r.exit_ts,
        r.created_ts, r.entry, r.stop, r.target, r.exit_price, r.pnl_pct,
        r.holding_days, r.review_tf, r.bars_reviewed, r.bars_expected,
        r.mfe_pct, r.mae_pct, r.target_distance_pct, r.stop_distance_pct,
        r.tp_capture_ratio, r.tp_verdict, r.sl_headroom_ratio, r.sl_verdict,
        r.pace_ratio, r.pace_verdict, json.dumps(list(r.confirmations)),
        r.planned_rr, r.notes,
    )


def _from_row(row: tuple) -> ReviewRecord:
    d = dict(zip(_COLUMNS, row))
    d["confirmations"] = tuple(json.loads(d["confirmations"]))
    return ReviewRecord(**d)


def insert_records(conn, records: list) -> int:
    """INSERT OR REPLACE every record (idempotent re-review). Returns the
    number of records written (not necessarily new rows)."""
    records = list(records)
    if not records:
        return 0
    ensure_schema(conn)
    placeholders = ", ".join(["?"] * len(_COLUMNS))
    with statestore._db_lock:
        conn.executemany(
            f"INSERT OR REPLACE INTO {TABLE} ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            [_to_row(r) for r in records],
        )
        conn.commit()
    logger.info("wrote %d review record(s) to %s", len(records), TABLE)
    return len(records)


def load_records(
    conn, *, strategy_version: str | None = None, symbol: str | None = None,
    span_class: str | None = None, start_ms: int | None = None,
    end_ms: int | None = None,
) -> list:
    """Load review_records, optionally filtered. Ordered by exit_ts ASC,
    mirroring storage.load_candles' WHERE-building convention."""
    ensure_schema(conn)
    query = f"SELECT {', '.join(_COLUMNS)} FROM {TABLE} WHERE 1=1"
    params: list = []
    if strategy_version is not None:
        query += " AND strategy_version = ?"
        params.append(strategy_version)
    if symbol is not None:
        query += " AND symbol = ?"
        params.append(symbol)
    if span_class is not None:
        query += " AND span_class = ?"
        params.append(span_class)
    if start_ms is not None:
        query += " AND exit_ts >= ?"
        params.append(start_ms)
    if end_ms is not None:
        query += " AND exit_ts <= ?"
        params.append(end_ms)
    query += " ORDER BY exit_ts ASC"
    with statestore._db_lock:
        rows = conn.execute(query, params).fetchall()
    return [_from_row(row) for row in rows]


# --------------------------------------------------------------------------- #
# diagnose() — pivot-guide Step 4.
# --------------------------------------------------------------------------- #


# A confirmation's decidability status, one of:
#   ""            — coverage below REVIEW_DEAD_WEIGHT_COVERAGE: not a candidate.
#   "unmeasurable" — full coverage AND no counterfactual (confirmation_pnl_delta
#                    is None). This is the HARD-GATE case (module docstring):
#                    every closed trade passed it by construction, so there is
#                    no "without it" group to compare against. NEVER produces
#                    a drop-confirmation suggestion.
#   "dead-weight"  — full coverage AND a real counterfactual exists
#                    (confirmation_pnl_delta is not None) that shows no benefit.
#                    Only this status may produce drop-confirmation.
CONFIRMATION_STATUSES = ("", "unmeasurable", "dead-weight")


@dataclass(frozen=True)
class Diagnosis:
    """A structured description of what these trades did.

    It contains NO fitness value and cannot be ordered against another
    Diagnosis. confirmation_pnl_delta is in-sample and undeflated — a hint
    about where to look, never evidence. confirmation_status distinguishes a
    DECIDED "dead-weight" verdict from an "unmeasurable" one where a hard gate
    left no counterfactual to decide from (module docstring) — only the
    former may appear in `suggestions` as a drop-confirmation.
    """

    n_records: int
    span_start_ms: int
    span_end_ms: int
    span_class: str | None
    strategy_version: str | None
    by_outcome: Mapping[str, int] = field(default_factory=dict)
    outcome_mean_pnl: Mapping[str, float | None] = field(default_factory=dict)
    dominant_outcome: str | None = None
    tp_verdicts: Mapping[str, int] = field(default_factory=dict)
    sl_verdicts: Mapping[str, int] = field(default_factory=dict)
    pace_verdicts: Mapping[str, int] = field(default_factory=dict)
    median_tp_capture: float | None = None
    median_sl_headroom: float | None = None
    confirmation_coverage: Mapping[str, float] = field(default_factory=dict)
    confirmation_pnl_delta: Mapping[str, float | None] = field(default_factory=dict)
    confirmation_status: Mapping[str, str] = field(default_factory=dict)
    pace: Mapping[str, object] = field(default_factory=dict)
    suggestions: tuple[str, ...] = ()
    digest: str = ""


def _digest(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def diagnose(records: list, *, start_ms: int, end_ms: int) -> Diagnosis:
    """Turn a batch of ReviewRecords into a closed-vocabulary Diagnosis.

    Deliberately NOT an LLM, NOT free text, NOT a scalar (see the NOT
    Building list, Phase 5 plan): every field is either a count, a median, a
    ratio, or a member of SUGGESTIONS.
    """
    records = list(records)
    n = len(records)
    span_class = records[0].span_class if records else None
    strategy_version = records[0].strategy_version if records else None

    if n == 0:
        d = Diagnosis(
            n_records=0, span_start_ms=start_ms, span_end_ms=end_ms,
            span_class=span_class, strategy_version=strategy_version,
            pace={
                "ann_return_pct": None, "target_ann_return": config.TARGET_ANN_RETURN,
                "n_trades": 0, "n_days": None, "sample_adequate": False,
            },
            suggestions=("insufficient-sample",),
        )
        return _with_digest(d)

    by_outcome: dict[str, int] = {}
    outcome_pnls: dict[str, list] = {}
    for r in records:
        by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
        outcome_pnls.setdefault(r.outcome, []).append(r.pnl_pct)
    outcome_mean_pnl = {k: statistics.fmean(v) for k, v in outcome_pnls.items()}
    dominant_outcome = max(by_outcome, key=lambda k: by_outcome[k])

    tp_verdicts = {v: 0 for v in TP_VERDICTS}
    sl_verdicts = {v: 0 for v in SL_VERDICTS}
    pace_verdicts = {v: 0 for v in PACE_VERDICTS}
    for r in records:
        tp_verdicts[r.tp_verdict] = tp_verdicts.get(r.tp_verdict, 0) + 1
        sl_verdicts[r.sl_verdict] = sl_verdicts.get(r.sl_verdict, 0) + 1
        pace_verdicts[r.pace_verdict] = pace_verdicts.get(r.pace_verdict, 0) + 1

    tp_caps = [r.tp_capture_ratio for r in records if r.tp_capture_ratio is not None]
    sl_heads = [r.sl_headroom_ratio for r in records if r.sl_headroom_ratio is not None]
    median_tp_capture = statistics.median(tp_caps) if tp_caps else None
    median_sl_headroom = statistics.median(sl_heads) if sl_heads else None

    all_names: set = set()
    for r in records:
        all_names.update(r.confirmations)
    confirmation_coverage = {
        name: sum(1 for r in records if name in r.confirmations) / n
        for name in sorted(all_names)
    }
    confirmation_pnl_delta: dict[str, float | None] = {}
    for name in sorted(all_names):
        with_it = [r.pnl_pct for r in records if name in r.confirmations]
        without_it = [r.pnl_pct for r in records if name not in r.confirmations]
        confirmation_pnl_delta[name] = (
            statistics.fmean(with_it) - statistics.fmean(without_it)
            if with_it and without_it else None
        )
    # Decidability status (module docstring): a hard-gating Confirmation at
    # full coverage has NO counterfactual (pnl_delta is None) and is
    # "unmeasurable", never "dead-weight" — see the Phase 4 measured
    # regression cited there before changing this.
    confirmation_status: dict[str, str] = {}
    for name in sorted(all_names):
        if confirmation_coverage[name] >= config.REVIEW_DEAD_WEIGHT_COVERAGE:
            confirmation_status[name] = (
                "dead-weight" if confirmation_pnl_delta[name] is not None else "unmeasurable"
            )
        else:
            confirmation_status[name] = ""

    # pace, the one aggregate. Deliberately omits Sharpe/DSR (Task 5): they
    # belong to the gate, and printing them here would make `review` look
    # like an evaluation.
    em = compute_equity_metrics(records, start_ms, end_ms)
    n_days = em["n_days"]
    sample_adequate = n >= config.REVIEW_PACE_MIN_TRADES and n_days >= config.REVIEW_PACE_MIN_DAYS
    pace = {
        "ann_return_pct": em["ann_return_pct"],
        "target_ann_return": config.TARGET_ANN_RETURN,
        "n_trades": n,
        "n_days": n_days,
        "sample_adequate": sample_adequate,
    }

    suggestions = _suggest(
        n=n, by_outcome=by_outcome, tp_verdicts=tp_verdicts, sl_verdicts=sl_verdicts,
        median_tp_capture=median_tp_capture, median_sl_headroom=median_sl_headroom,
        confirmation_status=confirmation_status,
    )

    d = Diagnosis(
        n_records=n, span_start_ms=start_ms, span_end_ms=end_ms,
        span_class=span_class, strategy_version=strategy_version,
        by_outcome=by_outcome, outcome_mean_pnl=outcome_mean_pnl,
        dominant_outcome=dominant_outcome, tp_verdicts=tp_verdicts,
        sl_verdicts=sl_verdicts, pace_verdicts=pace_verdicts,
        median_tp_capture=median_tp_capture, median_sl_headroom=median_sl_headroom,
        confirmation_coverage=confirmation_coverage,
        confirmation_pnl_delta=confirmation_pnl_delta,
        confirmation_status=confirmation_status, pace=pace,
        suggestions=suggestions,
    )
    return _with_digest(d)


def _suggest(
    *, n: int, by_outcome: dict, tp_verdicts: dict, sl_verdicts: dict,
    median_tp_capture: float | None, median_sl_headroom: float | None,
    confirmation_status: dict,
) -> tuple[str, ...]:
    """Deterministic, fixed-order suggestion rules (Task 5).

    drop-confirmation fires ONLY on a "dead-weight" status — i.e. only when a
    real counterfactual exists. "unmeasurable" (the hard-gate case: full
    coverage, no counterfactual) NEVER produces this suggestion — see
    records.py's module docstring for why coverage alone cannot decide it,
    and the Phase 4 measurement (+0.2028% -> -0.2523% expectancy) that a
    coverage-only rule would have acted against.
    """
    if n < config.REVIEW_PACE_MIN_TRADES:
        # No refinement from a noise sample — the single rule that stops
        # Step 4 becoming an overfitting engine.
        return ("insufficient-sample",)

    out: list[str] = []

    if by_outcome.get("time", 0) / n >= config.REVIEW_TIME_DOMINANCE:
        out.append("raise-max-hold")

    # Mutually exclusive by construction: widen-stop requires the stop to be
    # DOMINANT AND not over-wide; tighten-stop requires it to be over-wide on
    # a majority of records. Both cannot fire from the same verdict mix.
    if (
        by_outcome.get("stop", 0) / n >= config.REVIEW_STOP_DOMINANCE
        and median_sl_headroom is not None
        and median_sl_headroom >= 1.0
    ):
        out.append("widen-stop")
    elif sl_verdicts.get("over-wide", 0) / n > 0.5:
        out.append("tighten-stop")

    if tp_verdicts.get("target-too-far", 0) / n > 0.5:
        out.append("narrow-target")
    elif median_tp_capture is not None and median_tp_capture < config.REVIEW_TP_CAPTURE_GOOD:
        out.append("widen-target")

    if len(confirmation_status) > 1:
        for name, status in confirmation_status.items():
            if status == "dead-weight":
                out.append(f"drop-confirmation:{name}")

    return tuple(out)


def _with_digest(d: Diagnosis) -> Diagnosis:
    payload = diagnosis_to_dict(d)
    payload.pop("digest", None)
    return Diagnosis(**{**d.__dict__, "digest": _digest(payload)})


def diagnosis_to_dict(d: Diagnosis) -> dict:
    """Plain dict for Phase 7's UI and provenance (JSON-primitive values)."""
    payload = dict(d.__dict__)
    payload["suggestions"] = list(payload["suggestions"])
    return payload
