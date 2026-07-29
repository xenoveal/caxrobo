"""
Walk-forward validation harness — THE GATE (PRD Phase 7).

Implements the repaired anti-overfitting protocol:

  1. Rolling folds, POOLED across symbols: train TRAIN_DAYS, test the
     following TEST_DAYS, step by TEST_DAYS. Every combo/fold is scored on
     the CONCATENATED trade list across all symbols (pooled BEFORE scoring,
     not after) — 3x the sample per fold at zero additional degrees of
     freedom. Parameters are chosen ONLY on train data (best pooled
     expectancy among combos with >= WF_MIN_TRADES; otherwise defaults) and
     scored on the unseen test window.
  2. Robustness (plateau, not spike): each fold's winning combo is compared
     against its grid neighbors (one step on one axis) on TRAIN data via
     positive_neighbour_fraction (fraction of neighbours with positive
     expectancy) and neighbour_spread (max-min neighbour expectancy, an
     absolute-scale companion that never divides by a fitted value).
  3. One-shot out-of-sample: the final WF_OOS_DAYS of data are excluded from
     ALL tuning. The per-axis median of fold-winning parameters is run once
     over the holdout, pooled across symbols, and scored against THE GATE:
     sample size, Sharpe, DSR, max drawdown, mandatory per-symbol positive
     expectancy (3-of-3, not an average), and — new in v0.3.0 Phase 1 — beating
     the equal-weight BUY-AND-HOLD null on both annualized return and Sharpe
     over the identical span. The verdict is a per-condition dict
     (GATE_CONDITIONS); `passed` is all() of it.

Costs (fees + slippage) are inherited from the engine on every run. The
grid sweeps only the levers that actually bind — the two Phase 5 exit modes
and the hold-time limit (see DEFAULT_GRID for the measurements that retired
the old R:R-floor axis). The regime-classifier thresholds (the one
measured-healthy layer) and the frozen cost / ATR-stop-multiple /
Donchian-canonical constants are never swept here.
"""

import itertools
import logging
import statistics
from dataclasses import dataclass

from trading_bot import config
from trading_bot.backtest import trials
from trading_bot.backtest.benchmark import BenchmarkResult, buy_and_hold
from trading_bot.backtest.engine import BacktestParams, run_backtest
from trading_bot.backtest.equity import compute_equity_metrics
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.framework.execute import run_graph_backtest
from trading_bot.framework.graph import StrategyGraph

logger = logging.getLogger("trading_bot")

DAY_MS = 86_400_000

# v0.3.0 Phase 3 (contract §5). Grid axes that are run_backtest KWARGS rather
# than BacktestParams fields, and therefore still meaningful when a graph carries
# the strategy.
_RUN_LEVEL_AXES = frozenset({"max_hold_bars"})

# Small, coarse grid by design — a fine grid is an overfitting machine.
# Grid contents changed per PRD Phase 7: sweep the levers that actually bind
# (R:R floor, hold-time limit), NOT the regime-classifier thresholds (the
# one measured-healthy layer, kept fixed at config defaults) and NOT the
# cost model, ATR stop multiple k, or Donchian canonical lookbacks (frozen
# per the PRD Decisions Log).
#
# REPAIRED per .claude/PRPs/reports/code review/phase4-7-code-review.md (HIGH-3). The
# previous grid swept rr_floor and max_hold_bars, both of which were MEASURED
# NON-BINDING on stored history:
#   - minimum planned R:R across every trade the engine took was 1.56, above
#     the RR_FLOOR = 1.5 default, so that axis was unreachable at its own
#     default (100% of trades cleared 1.25 AND 1.5);
#   - 0 of 183 trades ever reached max_hold_bars = 96 or 144.
# Ten of twelve combos therefore produced IDENTICAL trade lists, fold winners
# were decided by itertools.product ordering over tied expectancies, and the
# DSR was deflated for 12 configurations per fold that were never distinct.
#
# The axes below are the ones that demonstrably move the result: the two exit
# modes (see config.py's exit-management block). max_hold_bars is KEPT because
# it starts binding hard once the trail is off — with trail and target both
# disabled, 49 of 142 trades exited on the time stop — so it is a real lever in
# the configuration this grid now explores, unlike in the old one.
DEFAULT_GRID: dict[str, tuple] = {
    "trail_enabled": (False, True),
    "target_enabled": (False, True),
    "max_hold_bars": (48, 96, 144),  # in trigger-timeframe bars; Phase 4 rescales
}

# Gate thresholds — mirrors the PRD Success Metrics table exactly. Not
# sweepable; changing these is a decision about the north star, not a
# tuning knob, and must not live inside the grid.
GATE_MIN_SHARPE = 1.0
GATE_MIN_DSR = 0.95
GATE_MAX_DRAWDOWN = 0.25

# The gate's condition set, in report order (v0.3.0 contract §4 fixes these
# names). _evaluate_gate returns a dict keyed by exactly this tuple, and
# passed == all(gate.values()). The two beats_benchmark_* conditions are new in
# v0.3.0 Phase 1: v0.2.0 compared the strategy against ZERO, so the gate
# "could have blessed a strategy worse than inaction" (KNOWN-LIMITATIONS §0).
# Adding a name here without adding it in _evaluate_gate is caught by a test.
GATE_CONDITIONS: tuple[str, ...] = (
    "sample_adequacy",  # n_trades >= min_trades
    "sharpe",  # >= GATE_MIN_SHARPE
    "dsr",  # >  GATE_MIN_DSR
    "max_drawdown",  # <= GATE_MAX_DRAWDOWN
    "per_symbol_expectancy",  # EVERY symbol > 0 (AND, not average)
    "beats_benchmark_return",  # NEW: ann_return_pct > basket ann_return_pct
    "beats_benchmark_sharpe",  # NEW: sharpe        > basket sharpe
)


@dataclass(frozen=True)
class FoldResult:
    """One train/test fold's outcome."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: BacktestParams
    train_expectancy: float | None
    positive_neighbour_fraction: float | None  # REPLACES the old mean/best ratio metric
    neighbour_spread: float | None  # NEW: max-min neighbour expectancy
    test_metrics: dict


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate walk-forward outcome and the one-shot OOS gate verdict."""

    folds: list[FoldResult]
    final_params: BacktestParams
    # The winning max_hold_bars is NOT a BacktestParams field (it is a
    # run_backtest kwarg), so it is recorded separately. Without it the
    # one-shot OOS verdict would be unreproducible: the run would depend on a
    # swept value that appeared nowhere in the result or the printout.
    final_max_hold_bars: int | None
    oos_start: int
    oos_end: int
    oos_metrics: dict  # from compute_metrics — n_trades, expectancy, etc.
    oos_equity: dict  # from compute_equity_metrics — sharpe, sortino, dsr, max_drawdown_pct
    per_symbol_expectancy: dict[str, float | None]  # symbol -> OOS expectancy_pct
    # v0.3.0 Phase 1 additions, APPENDED before `passed` so nothing is
    # reordered or renamed and every existing keyword-arg caller still works.
    gate: dict[str, bool]  # per-condition verdicts, keyed by GATE_CONDITIONS
    benchmark: BenchmarkResult  # the buy-and-hold null over the SAME OOS span
    n_trials_used: int  # what the DSR was actually charged
    passed: bool  # == all(gate.values())


def _combos(grid: dict[str, tuple]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _neighbors(grid: dict[str, tuple], combo: dict) -> list[dict]:
    """Combos one grid step away on exactly one axis."""
    out = []
    for axis, values in grid.items():
        i = values.index(combo[axis])
        for step in (-1, 1):
            if 0 <= i + step < len(values):
                out.append({**combo, axis: values[i + step]})
    return out


def _default_combo(grid: dict[str, tuple]) -> dict:
    """Config-default combo for the current grid axes.

    Rebuilt fresh for Phase 7: the old default_combo referenced regime
    thresholds (adx_trend_threshold, atr_extreme_percentile, bb_num_std)
    that are no longer grid axes. Built from config.RR_FLOOR /
    config.MAX_HOLD_BARS_TRIGGER instead.
    """
    return {axis: getattr(config, _CONFIG_DEFAULT_ATTR[axis]) for axis in grid}


_CONFIG_DEFAULT_ATTR = {
    "rr_floor": "RR_FLOOR",
    "max_hold_bars": "MAX_HOLD_BARS_TRIGGER",
    "trail_enabled": "TRAIL_ENABLED",
    "trail_atr_multiple": "TRAIL_ATR_MULTIPLE",
    "target_enabled": "DONCHIAN_TARGET_ENABLED",
}


def _combo_to_kwargs(combo: dict) -> tuple[BacktestParams, dict]:
    """Split a grid combo dict into BacktestParams fields and top-level kwargs.

    max_hold_bars is a run_backtest kwarg, not a BacktestParams field, so it
    must be split out here rather than passed into BacktestParams(**combo).
    """
    combo = dict(combo)
    max_hold_bars = combo.pop("max_hold_bars", None)
    extra_kwargs = {} if max_hold_bars is None else {"max_hold_bars": max_hold_bars}
    return BacktestParams(**combo), extra_kwargs


def _run_one(conn, symbol: str, *, start: int, end: int, params, extra_kwargs, strategy):
    """One symbol's trades for a fold — engine path or graph path.

    strategy=None keeps the legacy engine.run_backtest path bit-for-bit, so every
    pre-existing test is unaffected (contract §5). strategy=graph routes through
    framework.execute.run_graph_backtest, whose signature mirrors
    engine.run_backtest argument-for-argument minus `params` (which the graph
    carries), so this dispatch is the only place the two differ.
    """
    if strategy is None:
        return run_backtest(
            conn, symbol, start_ms=start, end_ms=end, params=params, **extra_kwargs
        )
    return run_graph_backtest(
        conn, strategy, symbol, start_ms=start, end_ms=end, **extra_kwargs
    )


def _pooled_expectancy(
    conn,
    symbols: list[str],
    combo: dict,
    start: int,
    end: int,
    ledger=None,
    strategy: StrategyGraph | None = None,
) -> tuple[float | None, int, list]:
    """Concatenate trades across all symbols for one combo/fold; return
    (pooled expectancy, pooled n_trades, pooled trade list).

    When a trials.TrialLedger is supplied, records ONE row per pooled
    evaluation — one configuration on one span, not one per symbol, because
    pooling is a single evaluation (contract §4).
    """
    if ledger is not None:
        ledger.record(
            graph_hash=trials.LEGACY_GRAPH_HASH,
            params_hash=trials.stable_hash(combo),
            start_ms=start,
            end_ms=end,
        )
    params, extra_kwargs = _combo_to_kwargs(combo)
    all_trades = []
    for symbol in symbols:
        all_trades.extend(
            _run_one(
                conn,
                symbol,
                start=start,
                end=end,
                params=params,
                extra_kwargs=extra_kwargs,
                strategy=strategy,
            )
        )
    m = compute_metrics(all_trades)
    return m["expectancy_pct"], m["n_trades"], all_trades


def _positive_neighbour_stats(
    conn,
    symbols: list[str],
    grid: dict,
    best_combo: dict,
    best_exp: float | None,
    start: int,
    end: int,
    ledger=None,
    strategy: StrategyGraph | None = None,
) -> tuple[float | None, float | None]:
    """
    Robustness check around the winning combo, computed on TRAIN data.

    Replaces the old `mean(neighbours) / best` ratio metric, which divides
    by a value that can be arbitrarily close to zero and reports numerically
    meaningless "fragility" (e.g. -11.3) for ordinary noisy neighbourhoods.

    Returns:
        (positive_neighbour_fraction, neighbour_spread) — the fraction of
        one-step neighbours with positive expectancy (near 1.0 = broad
        plateau of profitability; near 0 = an isolated spike), and the
        absolute spread (max - min) among neighbour expectancies (an
        absolute-scale companion that never divides by a fitted value).
        (None, None) if best_exp is None (no combo reached min_trades) or
        there are no neighbours to probe.

    Note: unlike the old ratio metric (which only ran when best_exp > 0),
    this computes stats whenever best_exp is not None, positive or
    negative — "is this a fluke or a broad negative region" is equally
    informative either way. This is a deliberate behavior change from the
    original code, not a bug.
    """
    if best_exp is None:
        return None, None
    neigh_exps = []
    for nb in _neighbors(grid, best_combo):
        # ledger forwarded deliberately: an uncounted neighbour probe
        # understates the search, the exact dishonesty the ledger prevents.
        exp, _, _ = _pooled_expectancy(conn, symbols, nb, start, end, ledger, strategy)
        if exp is not None:
            neigh_exps.append(exp)
    if not neigh_exps:
        return None, None
    pos_frac = sum(1 for e in neigh_exps if e > 0) / len(neigh_exps)
    spread = max(neigh_exps) - min(neigh_exps)
    return pos_frac, spread


def _evaluate_gate(
    oos_metrics: dict,
    oos_equity: dict,
    per_symbol_expectancy: dict[str, float | None],
    min_trades: int,
    benchmark: BenchmarkResult,
) -> dict[str, bool]:
    """
    THE GATE. All conditions are mandatory (AND, not average):
      1. sample_adequacy:        n_trades >= min_trades
      2. sharpe:                 not None and >= GATE_MIN_SHARPE
      3. dsr:                    not None and >  GATE_MIN_DSR (p < 0.05)
      4. max_drawdown:           not None and <= GATE_MAX_DRAWDOWN
      5. per_symbol_expectancy:  EVERY symbol not None and > 0
         (mandatory 3-of-3 cross-symbol gate, not an average)
      6. beats_benchmark_return: ann_return_pct > the basket buy-and-hold's
      7. beats_benchmark_sharpe: sharpe         > the basket buy-and-hold's

    Returns the per-condition verdict dict keyed by GATE_CONDITIONS; the
    caller's overall verdict is all(...) of its values.

    Fails safely (records False) whenever a required metric is None; never
    raises. A MISSING benchmark is a FAIL, not a pass — if the null could not
    be computed, the comparison was not made.
    """
    b = benchmark.basket
    sharpe = oos_equity["sharpe"]
    ann = oos_equity["ann_return_pct"]
    dd = oos_equity["max_drawdown_pct"]
    dsr = oos_equity["dsr"]
    return {
        "sample_adequacy": oos_metrics["n_trades"] >= min_trades,
        "sharpe": sharpe is not None and sharpe >= GATE_MIN_SHARPE,
        "dsr": dsr is not None and dsr > GATE_MIN_DSR,
        "max_drawdown": dd is not None and dd <= GATE_MAX_DRAWDOWN,
        # bool(...) first: `all(... for _ in {}.values())` is vacuously True,
        # and zero symbols is not "every symbol positive".
        "per_symbol_expectancy": bool(per_symbol_expectancy)
        and all(e is not None and e > 0 for e in per_symbol_expectancy.values()),
        "beats_benchmark_return": (
            ann is not None
            and b["ann_return_pct"] is not None
            and ann > b["ann_return_pct"]
        ),
        "beats_benchmark_sharpe": (
            sharpe is not None and b["sharpe"] is not None and sharpe > b["sharpe"]
        ),
    }


def walk_forward_pooled(
    conn,
    symbols: list[str],
    *,
    start_ms: int,
    end_ms: int,
    grid: dict[str, tuple] | None = None,
    train_days: int | None = None,
    test_days: int | None = None,
    oos_days: int | None = None,
    min_trades: int | None = None,
    n_trials: int | None = None,
    ledger=None,
    strategy: StrategyGraph | None = None,
) -> WalkForwardResult:
    """
    Pooled multi-symbol walk-forward: identical fold/train/test/OOS mechanics
    to the original per-symbol walk_forward, except every combo is scored on
    the CONCATENATED trade list across all `symbols` rather than one symbol
    at a time. This triples the sample per fold at zero additional degrees
    of freedom (Decisions Log: "the only free lunch on the list") and makes
    cross-symbol consistency checkable on the exact trades the gate scores.

    Args:
        conn: Database connection.
        symbols: Trading pair symbols, pooled into every fold and the OOS run.
        start_ms / end_ms: Overall data span (epoch ms). The final oos_days
            of the span are reserved for the one-shot holdout.
        grid: Parameter grid (default DEFAULT_GRID).
        train_days / test_days / oos_days / min_trades: Protocol knobs
            (default config.WF_*).
        n_trials: Total configurations evaluated during the tuning search,
            for the DSR correction (default: len(combos) * len(folds), or the
            ledger's cumulative count when a ledger is supplied).
        ledger: Optional trials.TrialLedger. When given, every pooled
            evaluation (including the robustness neighbour probes) is recorded
            in state.db and the DSR is charged the campaign's CUMULATIVE count,
            which survives process restarts (contract §4). None reproduces
            v0.2.0's behavior exactly.
        strategy: Optional serialized StrategyGraph (v0.3.0 Phase 3, contract
            §5). None keeps the legacy engine.run_backtest path bit-for-bit; a
            graph routes every run through
            framework.execute.run_graph_backtest instead. Because a graph
            carries its own parameters, `grid` may then contain only run-level
            axes (_RUN_LEVEL_AXES) — see the guard below.

    Returns:
        WalkForwardResult.

    Raises:
        ValueError: If the span is too short for at least one fold plus the
            OOS holdout, or if a graph is supplied alongside grid axes it cannot
            honor.
    """
    if grid is None:
        grid = DEFAULT_GRID

    if strategy is not None:
        unsupported = sorted(set(grid) - _RUN_LEVEL_AXES)
        if unsupported:
            raise ValueError(
                f"a graph strategy carries its own parameters, so grid axes "
                f"{unsupported} would be silently ignored — producing "
                f"{len(_combos(grid))} identical trade lists per fold and "
                f"deflating the DSR for configurations that were never distinct "
                f"(the exact pathology repaired as HIGH-3, see DEFAULT_GRID's "
                f"note). Pass grid={{'max_hold_bars': (...)}} or sweep graphs by "
                f"mutation (Phase 6), not by grid."
            )
    train_ms = (config.WF_TRAIN_DAYS if train_days is None else train_days) * DAY_MS
    test_ms = (config.WF_TEST_DAYS if test_days is None else test_days) * DAY_MS
    oos_ms = (config.WF_OOS_DAYS if oos_days is None else oos_days) * DAY_MS
    if min_trades is None:
        min_trades = config.WF_MIN_TRADES

    tune_end = end_ms - oos_ms
    if start_ms + train_ms + test_ms > tune_end:
        raise ValueError(
            "span too short: need at least one train+test fold before the OOS holdout"
        )

    combos = _combos(grid)

    folds: list[FoldResult] = []
    best_combos: list[dict] = []  # raw combo dicts, parallel to `folds` — needed
    # because FoldResult.best_params (a BacktestParams) doesn't carry
    # max_hold_bars (that's a run_backtest kwarg, not a dataclass field), so
    # the per-axis median below can't be recovered via getattr on it alone.
    t0 = start_ms
    while t0 + train_ms + test_ms <= tune_end:
        train_start, train_end = t0, t0 + train_ms
        test_start, test_end = train_end, train_end + test_ms

        # Grid sweep on TRAIN only, pooled across symbols.
        scored = []
        for combo in combos:
            exp, n, _ = _pooled_expectancy(
                conn, symbols, combo, train_start, train_end, ledger, strategy
            )
            if exp is not None and n >= min_trades:
                scored.append((exp, combo))

        if scored:
            best_exp, best_combo = max(scored, key=lambda x: x[0])
        else:
            best_exp, best_combo = None, _default_combo(grid)
            logger.info("fold %d: no combo reached min_trades; using defaults", len(folds))

        pos_frac, spread = _positive_neighbour_stats(
            conn, symbols, grid, best_combo, best_exp, train_start, train_end,
            ledger, strategy,
        )

        _, _, test_trades = _pooled_expectancy(
            conn, symbols, best_combo, test_start, test_end, ledger, strategy
        )
        best_params, _ = _combo_to_kwargs(best_combo)
        best_combos.append(best_combo)
        folds.append(
            FoldResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_expectancy=best_exp,
                positive_neighbour_fraction=pos_frac,
                neighbour_spread=spread,
                test_metrics=compute_metrics(test_trades),
            )
        )
        t0 += test_ms

    # Final parameters: per-axis median of fold winners (robust to outlier folds).
    #
    # median_low, NOT median: plain median INTERPOLATES on an even number of
    # folds, producing values no fold ever evaluated — [1.25, 2.0] -> 1.625,
    # [48, 96] -> 72 — so the one-shot OOS, the whole point of the protocol,
    # could be run at an unvalidated configuration that is not even on the grid.
    # median_low always returns an actual observed fold winner. It also keeps
    # boolean axes boolean. See the report's MEDIUM-4.
    final_combo = {
        axis: statistics.median_low(c[axis] for c in best_combos) for axis in grid
    }
    for axis, value in final_combo.items():
        if value not in grid[axis]:  # pragma: no cover - defensive invariant
            raise AssertionError(
                f"final parameter {axis}={value!r} is not on the grid "
                f"{grid[axis]!r}; the one-shot OOS would be unvalidated"
            )
    final_params, final_extra_kwargs = _combo_to_kwargs(final_combo)

    # n_trials for DSR: every combo evaluated per fold. Resolved HERE, after
    # the fold loop, because it needs len(folds). Deliberately does NOT
    # include the neighbour probes from _positive_neighbour_stats (those
    # examine robustness, not selection, and adding them would double-count
    # in a way that's hard to justify precisely) — a conservative-but-
    # approximate choice; n_trials is an explicit override for a future,
    # more careful count.
    #
    # With a ledger, the DSR charge is instead the campaign's CUMULATIVE
    # evaluation count, persisted in state.db so it survives restarts and
    # overnight runs (contract §4) — and it INCLUDES the neighbour probes the
    # ledgerless default deliberately omits. The ledgered number is therefore
    # strictly larger and the DSR strictly worse. That is the point.
    if n_trials is not None:
        n_trials_used = n_trials
    elif ledger is not None:
        n_trials_used = ledger.count()
    else:
        n_trials_used = len(combos) * max(1, len(folds))

    oos_start, oos_end = tune_end, end_ms
    oos_trades = []
    per_symbol_oos: dict[str, list] = {}
    for symbol in symbols:
        t = _run_one(
            conn, symbol, start=oos_start, end=oos_end,
            params=final_params, extra_kwargs=final_extra_kwargs, strategy=strategy,
        )
        per_symbol_oos[symbol] = t
        oos_trades.extend(t)

    oos_metrics = compute_metrics(oos_trades)
    oos_equity = compute_equity_metrics(
        oos_trades, oos_start, oos_end, n_trials=n_trials_used,
        attribution=config.PNL_ATTRIBUTION_MODE,
    )
    per_symbol_expectancy = {s: compute_metrics(t)["expectancy_pct"] for s, t in per_symbol_oos.items()}

    # The null, on the SAME span the strategy is scored on. Over a 90-day
    # holdout its ann_return_pct is a 90-day extrapolation exactly as the
    # strategy's is — identical annualization on an identical span keeps the
    # COMPARISON valid even though neither LEVEL is quotable.
    benchmark = buy_and_hold(conn, symbols, start_ms=oos_start, end_ms=oos_end)
    gate = _evaluate_gate(
        oos_metrics, oos_equity, per_symbol_expectancy, min_trades, benchmark
    )
    passed = all(gate.values())

    return WalkForwardResult(
        folds=folds,
        final_params=final_params,
        final_max_hold_bars=final_extra_kwargs.get("max_hold_bars"),
        oos_start=oos_start,
        oos_end=oos_end,
        oos_metrics=oos_metrics,
        oos_equity=oos_equity,
        per_symbol_expectancy=per_symbol_expectancy,
        gate=gate,
        benchmark=benchmark,
        n_trials_used=n_trials_used,
        passed=passed,
    )
