"""
THE ONE FITNESS ORACLE (v0.3.0 Phase 6; contract §4).

THE HONEST-ACCOUNTING POSITION, stated here because this is the file that could
quietly abandon it:

 1. ONE CODE PATH. Fitness comes only from walkforward.walk_forward_pooled.
    There is no second scoring function. scripts/bruteforce/core.score is NOT an
    oracle — it exists for the retired research sweep and must never be wired
    into evolution. `walk_forward_pooled` is imported HERE and nowhere else under
    evolution/, and a test walks the package's source to keep that true.
 2. EVERY EVALUATION INCREMENTS THE LEDGER. `ledger` is GateOracle.__init__'s
    first POSITIONAL parameter with no default. There is no set_ledger(), no
    None fallback, and no module-level function that scores a graph. The row is
    written BEFORE scoring (A6), so a crash costs a trial: over-charging is the
    only safe error. Duplicates are prevented at breeding time, never discounted
    at scoring time.
 3. THE FINAL HOLDOUT IS NEVER SEEN. Any window ending at or after
    `train_end_ms` raises HoldoutViolation BEFORE the ledger is charged and
    before the gate is called. Phase 9 owns the span beyond that ceiling.
 4. FITNESS IS NOT DSR (A2). DSR moves with n_trials, which grows through a
    campaign, so a candidate scored in generation 1 would not be comparable to
    one scored in generation 40. Selection ranks `excess_sharpe` (this
    candidate's Sharpe minus the buy-and-hold basket's, on the identical span),
    which is trial-count free. DSR decides the VERDICT, never the SELECTION.
 5. THE COUNT IS NEVER SOFTENED. Cumulative counting pushes n_trials into the
    hundreds or thousands and `expected_max_sharpe` grows with it. Measured
    figures this phase must not argue with (contract §4, Phase 9's post-MEDIUM-5
    moments, 207-day holdout): DSR = 0.8829 at n_trials=1, 0.0753 at 132,
    0.0090 at 3011. So the multiple-testing penalty is precisely what converts a
    near-pass into a hopeless one once the return distribution is repaired.
    FORBIDDEN: softening the count, passing n_trials=1, returning a cached score
    without a ledger row, shrinking the window-OOS, lowering WF_MIN_TRADES,
    counting "distinct genomes", or scoring through anything but this oracle. If
    honest accounting makes the gate unpassable inside a campaign, that is the
    FINDING, not a bug (PRD honesty clause).

WHY ONE LEDGER ROW PER CANDIDATE, NOT ONE PER FOLD. `evo_grid()` is DEGENERATE —
a single combo (A1). The 9 pooled evaluations `walk_forward_pooled` performs
internally per candidate (4 folds x train+test, plus the one-shot OOS) are the
SAME configuration on different spans, not 9 distinct configurations, so they
are not 9 multiple-testing trials. What the search selects among is candidates,
and a candidate costs exactly one row. The ledger is therefore NOT passed down
into walk_forward_pooled (that would add 9 rows per candidate and make the
next candidate's charge depend on the previous one's fold count); `n_trials` is
passed explicitly instead, as the campaign's cumulative post-charge count.
"""

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from trading_bot import config
from trading_bot.backtest import trials
from trading_bot.backtest.walkforward import (
    GATE_CONDITIONS,
    _RUN_LEVEL_AXES,
    walk_forward_pooled,
)
from trading_bot.framework.errors import FrameworkError
from trading_bot.framework.graph import graph_hash

logger = logging.getLogger("trading_bot")

__all__ = (
    "GateOracle",
    "HoldoutViolation",
    "LedgerHandle",
    "OracleResult",
    "TrialLedger",
    "evo_grid",
)


class HoldoutViolation(RuntimeError):
    """A requested span would cross the training ceiling into Phase 9's holdout.

    An exception rather than a printed refusal because this is a library:
    scripts/bruteforce/runner.py's guard prints and returns 3, which is right for
    a CLI and wrong here — a caller that ignored a return code would spend the
    holdout silently.
    """


def evo_grid() -> dict[str, tuple]:
    """The DEGENERATE one-combo grid every candidate is evaluated with (A1).

    The graph is the genome. Sweeping DEFAULT_GRID on top of it would double-
    search: the same candidate would be charged 12 configurations it never
    varied, and ~870 backtests instead of ~27 (engine.py's cache note).

    Two invariants a test pins, both of which fail SILENTLY if broken:
      - every axis is in walkforward._RUN_LEVEL_AXES, or walk_forward_pooled
        raises because a graph carries its own parameters;
      - the single value on each axis is EXACTLY the config default, because
        walkforward._default_combo() falls back to config defaults whenever no
        combo reaches min_trades and then ASSERTS the result is on the grid — a
        one-combo grid holding anything else raises AssertionError the first time
        a fold comes up short.
    """
    grid = {"max_hold_bars": (config.MAX_HOLD_BARS_TRIGGER,)}
    assert not set(grid) - _RUN_LEVEL_AXES  # pragma: no cover - structural invariant
    return grid


class LedgerHandle(Protocol):
    """What the oracle needs of a trial ledger, and nothing more.

    Narrow on purpose: a handle cannot be satisfied by an object that merely
    *looks* like a ledger and forgets to persist, because `charge` must return
    the post-insert cumulative count that is then handed to the DSR.
    """

    def charge(self, *, graph_hash: str, params_hash: str, start_ms: int,
               end_ms: int) -> int: ...

    def count(self) -> int: ...


class TrialLedger:
    """The only concrete LedgerHandle: an adapter over Phase 1's trial ledger.

    Thin by design. Phase 1 owns backtest/trials.py and its exact method names;
    isolating them here means a rename upstream changes this adapter and nothing
    else in the phase.
    """

    def __init__(self, state_conn, campaign_id: str):
        self._ledger = trials.TrialLedger(state_conn, campaign_id)
        self.campaign_id = campaign_id

    def charge(self, *, graph_hash: str, params_hash: str, start_ms: int,
               end_ms: int) -> int:
        """Persist one evaluation and return the campaign's POST-insert count."""
        return self._ledger.record(
            graph_hash=graph_hash, params_hash=params_hash,
            start_ms=start_ms, end_ms=end_ms,
        )

    def count(self) -> int:
        return self._ledger.count()

    def distinct_count(self) -> int:
        """REPORTED ONLY. Contract §4 forbids charging the DSR this number."""
        return self._ledger.distinct_count()


@dataclass(frozen=True)
class OracleResult:
    """One candidate's flattened gate outcome. Plain, picklable, frozen.

    Every field is either a primitive, None, or a dict[str, bool], so
    dataclasses.asdict() crosses a process boundary without a custom reducer
    (trap 3: under spawn, every payload must be picklable).

    sharpe / dsr / ann_return_pct / max_drawdown_pct are None when UNDEFINED
    (fewer than two daily observations, no trades). They are never coerced to
    0.0: a missing metric must not read as a mediocre one.

    `gate` is Phase 1's dict verbatim, keyed by GATE_CONDITIONS. Tiering reads it
    rather than re-deriving any condition — two implementations of one condition
    is how they drift.

    `error` is non-empty only for a LEGITIMATE evaluation failure (a span too
    short for a fold). A malformed graph is a mutator BUG and propagates.
    """

    graph_hash: str
    params_hash: str
    window_start_ms: int
    window_end_ms: int
    oos_start_ms: int | None
    oos_end_ms: int | None
    n_trials_used: int
    n_trades: int
    sharpe: float | None
    dsr: float | None
    ann_return_pct: float | None
    max_drawdown_pct: float | None
    bench_sharpe: float | None
    bench_ann_return_pct: float | None
    gate: dict
    passed: bool
    eval_seconds: float
    error: str = ""


class GateOracle:
    """Scores a StrategyGraph through THE GATE, and charges a trial for doing so.

    The ledger is the FIRST POSITIONAL argument with no default. That is the
    whole design: `GateOracle()` is a TypeError, `GateOracle(ohlcv_conn=...)` is a
    TypeError, and there is no module-level function that takes a graph and
    returns a number. Bypassing the ledger is structurally impossible rather
    than discouraged.
    """

    def __init__(
        self,
        ledger: LedgerHandle,
        *,
        ohlcv_conn,
        symbols,
        train_start_ms: int,
        train_end_ms: int,
        train_days: int | None = None,
        test_days: int | None = None,
        oos_days: int | None = None,
        min_trades: int | None = None,
    ) -> None:
        if ledger is None:
            raise ValueError(
                "GateOracle requires a ledger handle: contract §4.2 says nothing "
                "may score a candidate without charging the trial ledger"
            )
        for attr in ("charge", "count"):
            if not callable(getattr(ledger, attr, None)):
                raise TypeError(
                    f"ledger does not satisfy LedgerHandle: missing callable "
                    f"{attr!r}. A stand-in that cannot charge is exactly the "
                    f"bypass contract §4.2 forbids."
                )
        if train_end_ms <= train_start_ms:
            raise ValueError(
                f"train span is empty: train_start_ms={train_start_ms} >= "
                f"train_end_ms={train_end_ms}"
            )
        self._ledger = ledger
        self._conn = ohlcv_conn
        self._symbols = tuple(symbols)
        if not self._symbols:
            raise ValueError("GateOracle needs at least one symbol")
        self.train_start_ms = int(train_start_ms)
        self.train_end_ms = int(train_end_ms)
        self._train_days = train_days
        self._test_days = test_days
        self._oos_days = oos_days
        self._min_trades = min_trades

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def count(self) -> int:
        """The campaign's cumulative charged trials."""
        return self._ledger.count()

    def _check_span(self, window_start_ms: int, window_end_ms: int) -> None:
        """Refuse any span outside the declared training bounds.

        Runs FIRST, before the ledger is touched and before the gate is called:
        a refused evaluation must cost nothing, or a caller probing the boundary
        would inflate its own DSR penalty for evaluations that never happened.
        `>` on the end bound: walk_forward_pooled's end_ms is INCLUSIVE, so
        end_ms == train_end_ms scores the ceiling bar itself, which is the last
        bar Phase 9 does NOT own.
        """
        if window_start_ms < self.train_start_ms or window_end_ms > self.train_end_ms:
            raise HoldoutViolation(
                f"span [{window_start_ms}, {window_end_ms}] leaves the declared "
                f"training window [{self.train_start_ms}, {self.train_end_ms}]. "
                f"Evolution may never score a bar beyond the ceiling — that span "
                f"is Phase 9's holdout and it is evaluated ONCE, on a final "
                f"pre-committed shortlist, by Phase 9 alone."
            )
        if window_start_ms >= window_end_ms:
            raise HoldoutViolation(
                f"span [{window_start_ms}, {window_end_ms}] is empty or inverted"
            )

    def evaluate(self, graph, *, window_start_ms: int, window_end_ms: int) -> OracleResult:
        """Score one candidate. Charges exactly one trial, before scoring.

        Args:
            graph: The StrategyGraph to score.
            window_start_ms / window_end_ms: The evaluation span, inclusive, which
                walk_forward_pooled splits into folds plus a window-OOS.

        Returns:
            OracleResult. On a legitimate ValueError from walk_forward_pooled (a
            span too short for one fold plus the holdout) the result carries
            `error` and undefined metrics — and the ledger row STAYS, because the
            evaluation was attempted (A6).

        Raises:
            HoldoutViolation: If the span leaves the training window. Nothing is
                charged.
            FrameworkError: If the graph is malformed. That is a mutator bug and
                must abort the campaign, not score as a safe zero.
        """
        self._check_span(window_start_ms, window_end_ms)

        grid = evo_grid()
        g_hash = graph_hash(graph)
        # params_hash is the degenerate combo's digest and is therefore CONSTANT
        # across candidates. That is exactly right: it records that ONE
        # configuration was evaluated per candidate, not twelve.
        p_hash = trials.stable_hash(
            {axis: values[0] for axis, values in grid.items()}
        )

        n_trials = self._ledger.charge(
            graph_hash=g_hash, params_hash=p_hash,
            start_ms=window_start_ms, end_ms=window_end_ms,
        )

        t0 = time.perf_counter()
        try:
            wf = walk_forward_pooled(
                self._conn,
                list(self._symbols),
                start_ms=window_start_ms,
                end_ms=window_end_ms,
                grid=grid,
                train_days=self._train_days,
                test_days=self._test_days,
                oos_days=self._oos_days,
                min_trades=self._min_trades,
                n_trials=n_trials,
                strategy=graph,
            )
        except ValueError as exc:
            # A span too short for one fold plus the holdout. A real, expected
            # outcome of a jittered window near the edge of the training span —
            # NOT a graph bug, so it is recorded rather than raised, and the
            # ledger row stays charged.
            elapsed = time.perf_counter() - t0
            logger.info("oracle: evaluation refused by walk-forward (%s)", exc)
            return OracleResult(
                graph_hash=g_hash, params_hash=p_hash,
                window_start_ms=window_start_ms, window_end_ms=window_end_ms,
                oos_start_ms=None, oos_end_ms=None,
                n_trials_used=n_trials, n_trades=0,
                sharpe=None, dsr=None, ann_return_pct=None, max_drawdown_pct=None,
                bench_sharpe=None, bench_ann_return_pct=None,
                gate={}, passed=False, eval_seconds=elapsed, error=str(exc),
            )
        except FrameworkError:
            # A malformed graph reached the executor. Under EVO_STRICT_MUTATORS
            # this aborts the campaign: a broken graph takes zero trades, which
            # would score as an honest tier-D rejection while hiding a mutator
            # bug behind a plausible number.
            raise

        elapsed = time.perf_counter() - t0
        eq = wf.oos_equity
        basket = wf.benchmark.basket
        return OracleResult(
            graph_hash=g_hash,
            params_hash=p_hash,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            oos_start_ms=wf.oos_start,
            oos_end_ms=wf.oos_end,
            n_trials_used=wf.n_trials_used,
            n_trades=wf.oos_metrics["n_trades"],
            sharpe=eq["sharpe"],
            dsr=eq["dsr"],
            ann_return_pct=eq["ann_return_pct"],
            max_drawdown_pct=eq["max_drawdown_pct"],
            bench_sharpe=basket["sharpe"],
            bench_ann_return_pct=basket["ann_return_pct"],
            # dict(...) so the result owns its own copy across a pickle boundary.
            gate={name: bool(wf.gate[name]) for name in GATE_CONDITIONS},
            passed=wf.passed,
            eval_seconds=elapsed,
            error="",
        )
