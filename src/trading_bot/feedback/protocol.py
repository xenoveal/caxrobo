"""
Forward-test protocol and the deterministic refinement loop — v0.3.0 Phase 5
(contract §4, §5).

THIS IS THE ONLY MODULE IN feedback/ PERMITTED TO CAUSE TRADES TO EXIST, and
only through walkforward.walk_forward_pooled (lock #3 of the oracle boundary
— see feedback/records.py's docstring for the full statement). Every other
module in this package (records.py, versioning.py, plugins/reviewers/
trade_quality.py) is import-banned from the oracle precisely so this module is
the single seam. A review never runs a backtest; a forward test always does,
and it is always THE GATE.

Fitness comes only from walk_forward_pooled, charged to Phase 1's persistent
trial ledger (backtest/trials.py). evaluate_forward here calls it EXACTLY
ONCE per evaluation, with a FROZEN one-combo grid (a forward test evaluates a
CHOSEN candidate; it must not tune) and an n_trials value read from the
ledger's cumulative count for the campaign — never `len(combos) * len(folds)`.
"""

import logging
import time
from dataclasses import dataclass, replace

from trading_bot import config
from trading_bot.backtest import trials
from trading_bot.backtest.walkforward import walk_forward_pooled
from trading_bot.feedback import records, versioning
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.errors import RegistryError
from trading_bot.framework.execute import run_graph_backtest

logger = logging.getLogger("trading_bot")

DAY_MS = 86_400_000


class ForwardSpanError(ValueError):
    """A forward span that is inverted, too short, or touches the locked
    final holdout. A dedicated type so the CLI can print a clean message
    (mirrors _walkforward_command's `except ValueError`)."""


@dataclass(frozen=True)
class ForwardSpan:
    history_start_ms: int
    forward_start_ms: int
    forward_end_ms: int
    n_forward_days: int


def forward_span(*, history_start_ms: int, forward_start_ms: int, forward_end_ms: int) -> ForwardSpan:
    """Validate and construct a ForwardSpan. Three checks, in order (Task 8):

    1. Ordering: forward_end_ms > forward_start_ms > history_start_ms.
    2. Minimum window: below config.REVIEW_FORWARD_MIN_DAYS is REFUSED, not
       reported (KNOWN-LIMITATIONS §2: a 90-day window from 23 trades was
       over-read as evidence; a shorter one buys even less).
    3. The LOCKED final holdout (Phase 9's config.HOLDOUT_START/END, contract
       §7) is untouchable. Read AT CALL TIME (not import time), because Phase
       9 has not landed as of this phase's implementation: when the constants
       are absent, this WARNS and the guard is INERT — that must be stated
       plainly in any report using this function, not silently assumed fixed
       later.

    Raises:
        ForwardSpanError: On any of the three checks failing.
    """
    if not (forward_end_ms > forward_start_ms > history_start_ms):
        raise ForwardSpanError(
            f"forward span must satisfy forward_end_ms({forward_end_ms}) > "
            f"forward_start_ms({forward_start_ms}) > history_start_ms({history_start_ms})"
        )

    span_ms = forward_end_ms - forward_start_ms
    if span_ms % DAY_MS != 0:
        raise ForwardSpanError(
            f"forward_start_ms..forward_end_ms is {span_ms} ms, not a whole "
            f"number of days; forward_start_ms must equal "
            f"forward_end_ms - n_forward_days * DAY_MS exactly, or the gate's "
            f"holdout and the declared window differ by hours"
        )
    n_forward_days = span_ms // DAY_MS

    if n_forward_days < config.REVIEW_FORWARD_MIN_DAYS:
        raise ForwardSpanError(
            f"forward window is {n_forward_days} days, below "
            f"config.REVIEW_FORWARD_MIN_DAYS={config.REVIEW_FORWARD_MIN_DAYS}; "
            f"a window this short is refused, not reported (KNOWN-LIMITATIONS §2)"
        )

    holdout_start = getattr(config, "HOLDOUT_START", None)
    holdout_end = getattr(config, "HOLDOUT_END", None)
    if holdout_start is not None and holdout_end is not None:
        # Inclusive both sides (load_candles' convention); the checked span
        # includes history_start_ms, not just the forward window, per Task 8.
        if not (forward_end_ms < holdout_start or history_start_ms > holdout_end):
            raise ForwardSpanError(
                f"forward span {history_start_ms}..{forward_end_ms} intersects "
                f"the LOCKED final holdout {holdout_start}..{holdout_end} "
                f"(config.HOLDOUT_START/END). Phase 9 owns that span; no "
                f"feedback iteration, forward test or evolution generation "
                f"may see it."
            )
    else:
        logger.warning(
            "config.HOLDOUT_START/END are not set; the forward-span holdout "
            "guard is INERT for this call (holdout_guard=absent)"
        )

    return ForwardSpan(
        history_start_ms=history_start_ms, forward_start_ms=forward_start_ms,
        forward_end_ms=forward_end_ms, n_forward_days=n_forward_days,
    )


@dataclass(frozen=True)
class ForwardTestResult:
    version_id: str
    symbols: tuple
    span: ForwardSpan
    gate: dict
    passed: bool
    n_trials_used: int
    oos_metrics: dict
    oos_equity: dict
    benchmark: object
    trades: tuple
    config_drift: dict
    notes: str


def _frozen_max_hold_bars(graph) -> int:
    """The single max_hold_bars value a forward test freezes (Task 9: "build
    {axis: (value,)} from the version's own parameters"). max_hold_bars is a
    per-BRANCH ExitPolicySpec field (graph-level structure), not a registered
    plug-in ParamSpec, so there is no single canonical "the graph's value" —
    this reads the first enabled branch's explicit override, falling back to
    config.MAX_HOLD_BARS_TRIGGER (run_graph_backtest's own default) when every
    branch defers. A documented choice, not a discovered one: Task 0 found no
    ParamSpec named max_hold_bars in the registry to consult instead.
    """
    for branch in graph.ordered_branches():
        if branch.enabled and branch.exits.max_hold_bars is not None:
            return branch.exits.max_hold_bars
    return config.MAX_HOLD_BARS_TRIGGER


def evaluate_forward(
    conn, state_conn, *, version_id: str, symbols, span: ForwardSpan,
    n_trials: int | None = None, campaign: str | None = None,
) -> ForwardTestResult:
    """Forward-test a REGISTERED version through THE GATE and nothing else.

    Design's central move: the forward window IS the gate's one-shot OOS
    holdout, so `gate`, `benchmark` (the buy-and-hold null, contract §4.3) and
    `n_trials_used` describe exactly the unseen span. This is not a second
    evaluation path; it is walk_forward_pooled pointed at a later window.

    Ledger discipline (Task 0's Q2, answered by reading walkforward.py):
    walk_forward_pooled does NOT record to any ledger unless a `ledger=`
    object is explicitly passed to it, and even then it would record ONE row
    per grid combo per fold (train sweep + test evaluation), which is the
    wrong granularity for "one candidate, one forward evaluation". So this
    function does NOT pass `ledger=` to walk_forward_pooled. Instead it
    records EXACTLY ONE row itself (below) and passes the resulting
    cumulative count explicitly via `n_trials=` — the two mechanisms are
    mutually exclusive by construction, never both, as Task 9 requires.

    Args:
        conn: OHLCV connection.
        state_conn: state.db connection (for load_graph/verify_reproducible
            and the trial ledger).
        version_id: A version registered via versioning.register_version.
        symbols: Symbols to pool.
        span: A ForwardSpan from forward_span() — already holdout-checked.
        n_trials: Override for the DSR's cumulative trial count. When None
            (the normal path), one ledger row is recorded here and its
            POST-INSERT cumulative count is used.
        campaign: Ledger campaign name. Defaults to f"review:{version_id}" —
            each version accumulates its own forward-test trial count,
            distinct from Phase 6's evolution campaigns.

    Returns:
        ForwardTestResult. `trades` are the OOS trades, stamped with
        version_id, for the caller to review — WalkForwardResult exposes no
        trade list (only aggregate dicts), so they are RE-DERIVED here via one
        run_graph_backtest call per symbol over the OOS span the gate already
        computed. This is RECORD-GENERATION, not scoring: the verdict already
        came from the gate call above and this adds no trial.

    Raises:
        ForwardSpanError: If the span is too short for a fold before the OOS
            holdout (wrapping walk_forward_pooled's ValueError).
        KeyError: If version_id is unknown.
    """
    graph = versioning.load_graph(state_conn, version_id)
    config_drift = versioning.verify_reproducible(state_conn, version_id)
    if not config_drift["config_hash_matches"]:
        logger.warning(
            "config drift since version %s creation (%d key(s) differ)",
            version_id, len(config_drift["diff"]),
        )

    max_hold_bars = _frozen_max_hold_bars(graph)
    grid = {"max_hold_bars": (max_hold_bars,)}  # ONE combo: no tuning happens,
    # structurally (walkforward._neighbors returns [] for a 1-tuple axis).

    campaign_name = campaign or f"review:{version_id}"
    graph_hash_val = fgraph.graph_hash(graph)
    params_hash_val = trials.stable_hash(grid)

    if n_trials is None:
        ledger = trials.TrialLedger(state_conn, campaign_name)
        n_trials_resolved = ledger.record(
            graph_hash=graph_hash_val, params_hash=params_hash_val,
            start_ms=span.history_start_ms, end_ms=span.forward_end_ms,
            ts=int(time.time() * 1000),
        )
    else:
        n_trials_resolved = n_trials

    try:
        wf = walk_forward_pooled(
            conn, list(symbols),
            start_ms=span.history_start_ms, end_ms=span.forward_end_ms,
            grid=grid, oos_days=span.n_forward_days,
            n_trials=n_trials_resolved, strategy=graph,
        )
    except ValueError as exc:
        min_history_days = config.WF_TRAIN_DAYS + config.WF_TEST_DAYS
        raise ForwardSpanError(
            f"{exc}; history must precede the forward window by at least "
            f"config.WF_TRAIN_DAYS + config.WF_TEST_DAYS = {min_history_days} days"
        ) from exc

    oos_trades = []
    for symbol in symbols:
        oos_trades.extend(
            run_graph_backtest(
                conn, graph, symbol, start_ms=wf.oos_start, end_ms=wf.oos_end,
                max_hold_bars=max_hold_bars,
            )
        )
    stamped = versioning.stamp_trades(oos_trades, version_id)

    return ForwardTestResult(
        version_id=version_id, symbols=tuple(symbols), span=span,
        gate=wf.gate, passed=wf.passed, n_trials_used=wf.n_trials_used,
        oos_metrics=wf.oos_metrics, oos_equity=wf.oos_equity,
        benchmark=wf.benchmark, trades=stamped, config_drift=config_drift,
        notes="",
    )


# --------------------------------------------------------------------------- #
# apply_suggestion — deterministic, bounded, diagnosis-driven single-step
# edits. NOT a mutator: no RNG, no population, one step per suggestion.
# Phase 6's plugins/mutators/* are the stochastic, population-driven
# counterpart; if Phase 6 publishes an equivalent bounded edit, delegate
# rather than keep two implementations (Risks table).
# --------------------------------------------------------------------------- #

# Task 0 finding, recorded here rather than silently patched: grepping every
# ParamSpec name registered across plugins/*/*.py (as of this phase) finds NO
# "target_multiple" and NO "max_hold_bars" parameter anywhere. Target distance
# comes from detector geometry (DetectedEvent.target_height), not a tunable
# node parameter, and max_hold_bars is a graph-level ExitPolicySpec field with
# no ParamSpec bounds to step within. Per the plan's fail-closed rule ("never
# invent a parameter name — exact names come from Phase 4's registered
# policy; if they differ, update _SUGGESTION_EDIT, do not guess"),
# widen-target/narrow-target/raise-max-hold/lower-max-hold are deliberately
# NOT mapped below and always resolve to None + WARNING until a future phase
# registers a corresponding bounded parameter. Only widen-stop/tighten-stop
# (atr_multiple, declared by policy.measured-move and
# policy.atr-stop-measured-move) are implementable today.
_SUGGESTION_EDIT = {
    "widen-stop": ("atr_multiple", 1),
    "tighten-stop": ("atr_multiple", -1),
}


def apply_suggestion(graph, suggestion: str):
    """One bounded parameter step, or a confirmation removal, or None.

    Returns:
        A NEW StrategyGraph, or None when the suggestion cannot be safely
        applied (unknown suggestion, no backing ParamSpec anywhere in the
        graph, or the step would be a no-op because every matching node is
        already at its bound). The input graph is never mutated (it is
        frozen) and a None return leaves the caller's graph reference
        untouched either way.
    """
    if suggestion.startswith("drop-confirmation:"):
        name = suggestion.split(":", 1)[1]
        return _drop_confirmation(graph, name)

    edit = _SUGGESTION_EDIT.get(suggestion)
    if edit is None:
        logger.warning(
            "apply_suggestion: %r has no backing ParamSpec in the current "
            "plug-in set; refusing rather than inventing one", suggestion,
        )
        return None
    param_name, direction = edit
    return _step_param(graph, param_name, direction)


def _drop_confirmation(graph, name: str):
    """Remove the confirmation node named `name`.

    MEASURED (not assumed): every Confirmation plug-in shipped by Phase 4
    (plugins/confirmations/{macd,volume_breakout}.py) sets
    ConfirmationVerdict.name to its own FULL REGISTRY KEY (e.g.
    "confirmation.macd", not the bare "macd" — see MACD_CONFIRM_NAME /
    VOLUME_CONFIRM_NAME), and execute.py stores exactly that string on
    Trade.confirmations (`confirmed.append(verdict.name or spec.name)`). So
    the string a Diagnosis's confirmation_coverage is keyed by, and the
    "drop-confirmation:<name>" suggestion this function receives, is that
    SAME full key — which is also NodeSpec.key. Comparing against `c.key`
    directly (not a stripped plug-in name) is therefore correct, not a
    shortcut; an earlier version of this function stripped the kind prefix
    and never matched anything as a result.
    """
    matched_any = False
    changed_any = False
    new_branches = []
    for b in graph.branches:
        keep = tuple(c for c in b.confirmations if c.key != name)
        if len(keep) != len(b.confirmations):
            matched_any = True
            if len(keep) == 0:
                # Refuse to drop the LAST confirmation on a branch: a bounded
                # single-step edit should not turn "gated" into "ungated" in
                # one move.
                keep = b.confirmations
            else:
                changed_any = True
        new_branches.append(replace(b, confirmations=keep))

    if not matched_any:
        logger.warning("apply_suggestion: no confirmation named %r found in the graph", name)
        return None
    if not changed_any:
        logger.warning(
            "apply_suggestion: dropping %r would leave a branch with zero "
            "confirmations; refusing rather than removing the last gate", name,
        )
        return None
    return replace(graph, branches=tuple(new_branches))


def _step_param(graph, param_name: str, direction: int):
    found = False
    changed = False

    def edit_node(node):
        nonlocal found, changed
        try:
            spec = registry.get(node.key)
        except RegistryError:
            return node
        pspec = spec.params.get(param_name)
        if pspec is None or pspec.bounds is None:
            return node
        found = True
        current = spec.resolve(node.params)[param_name]
        lo, hi = pspec.bounds
        step = config.REVIEW_REFINE_STEP_FRAC * (hi - lo)
        candidate = current + direction * step
        clamped = pspec.clamp(candidate)
        if clamped == current:
            return node
        changed = True
        new_params = dict(node.params)
        new_params[param_name] = clamped
        return replace(node, params=new_params)

    new_branches = tuple(
        replace(
            b, detector=edit_node(b.detector), policy=edit_node(b.policy),
            confirmations=tuple(edit_node(c) for c in b.confirmations),
        )
        for b in graph.branches
    )
    new_data = edit_node(graph.data)
    new_filters = tuple(edit_node(f) for f in graph.filters)

    if not found:
        logger.warning(
            "apply_suggestion: no plug-in in the graph declares parameter %r", param_name
        )
        return None
    if not changed:
        logger.warning(
            "apply_suggestion: parameter %r is already at its bound on every "
            "node that declares it; refusing a no-op edit", param_name,
        )
        return None
    return replace(graph, data=new_data, branches=new_branches, filters=new_filters)


# --------------------------------------------------------------------------- #
# run_loop_iteration — Steps 3-6, the single CLI entry point.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoopIterationResult:
    parent_version_id: str
    records_written: int
    diagnosis: object
    suggestions_applied: tuple
    child_version_id: str | None
    forward: ForwardTestResult | None
    notes: str


def run_loop_iteration(
    conn, state_conn, *, version_id: str, symbols, review_start_ms: int,
    review_end_ms: int, span_class: str = "in-sample",
    forward: ForwardSpan | None = None, reviewer: str = "reviewer.trade-quality",
    persist: bool = True,
) -> LoopIterationResult:
    """Steps 3-6 in one call: review -> persist -> diagnose -> refine ->
    version -> (optionally) forward-test the child.

    Args:
        span_class: "in-sample" (default) reviews version_id's OWN trades
            over [review_start_ms, review_end_ms] via run_graph_backtest — an
            ALREADY-SEEN span, so the output is banner-labelled and makes no
            new claim. "forward" instead reviews version_id via
            evaluate_forward over `forward` (i.e. from THE GATE) — used when
            an operator wants a forward verdict on an EXISTING version
            without first producing a child.
        forward: When given AND a child is produced (Step 5), the CHILD is
            forward-tested over this SAME span and its records persist with
            span_class="forward". Never the parent by default (contract §4:
            "forward-testing" an already-seen span is exactly the side
            channel the gate exists to prevent).

    Returns:
        LoopIterationResult. child_version_id is None, with `notes` saying
        why, when no suggestion could be applied — an iteration that changes
        nothing is a valid outcome, never disguised as progress.
    """
    if span_class not in records.SPAN_CLASSES:
        raise ValueError(f"span_class must be one of {records.SPAN_CLASSES}, got {span_class!r}")

    registry.load_all()
    graph = versioning.load_graph(state_conn, version_id)
    reviewer_key = reviewer if "." in reviewer else f"reviewer.{reviewer}"
    reviewer_spec = registry.get(reviewer_key)

    review_tf = config.REVIEW_TIMEFRAME or config.SIGNAL_TRIGGER_TIMEFRAME

    if span_class == "in-sample":
        raw_trades = []
        for symbol in symbols:
            raw_trades.extend(
                run_graph_backtest(conn, graph, symbol, start_ms=review_start_ms, end_ms=review_end_ms)
            )
        trades = versioning.stamp_trades(raw_trades, version_id)
    else:  # "forward"
        if forward is None:
            raise ValueError('span_class="forward" requires a ForwardSpan via `forward=`')
        fwd = evaluate_forward(conn, state_conn, version_id=version_id, symbols=symbols, span=forward)
        trades = fwd.trades

    created_ts = int(time.time() * 1000)
    ctx = records.ReviewContext(
        conn=conn, review_tf=review_tf, strategy_version=version_id,
        span_class=span_class, target_ann_return=config.TARGET_ANN_RETURN,
        created_ts=created_ts,
    )
    review_records = [reviewer_spec.impl(t, ctx) for t in trades]

    written = 0
    if persist:
        written = records.insert_records(state_conn, review_records)

    diagnosis = records.diagnose(review_records, start_ms=review_start_ms, end_ms=review_end_ms)

    suggestions_applied: list = []
    child_graph = graph
    for s in diagnosis.suggestions:
        edited = apply_suggestion(child_graph, s)
        if edited is not None:
            child_graph = edited
            suggestions_applied.append(s)

    child_version_id = None
    notes = ""
    if suggestions_applied and fgraph.graph_hash(child_graph) != fgraph.graph_hash(graph):
        child = versioning.register_version(
            state_conn, child_graph, parent_id=version_id,
            provenance={
                "source": "diagnosis", "diagnosis_digest": diagnosis.digest,
                "suggestions": list(suggestions_applied),
            },
        )
        child_version_id = child.version_id
    else:
        notes = "no suggestion produced a graph edit; iteration recorded a diagnosis only"

    forward_result = None
    if forward is not None and child_version_id is not None:
        forward_result = evaluate_forward(
            conn, state_conn, version_id=child_version_id, symbols=symbols, span=forward,
        )
        if persist:
            fwd_ctx = replace(
                ctx, span_class="forward", strategy_version=child_version_id,
                created_ts=int(time.time() * 1000),
            )
            fwd_records = [reviewer_spec.impl(t, fwd_ctx) for t in forward_result.trades]
            records.insert_records(state_conn, fwd_records)

    return LoopIterationResult(
        parent_version_id=version_id, records_written=written, diagnosis=diagnosis,
        suggestions_applied=tuple(suggestions_applied), child_version_id=child_version_id,
        forward=forward_result, notes=notes,
    )
