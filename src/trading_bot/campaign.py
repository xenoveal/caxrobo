"""
The v0.3.0 walk-forward CAMPAIGN — Phase 9, the verdict on the key hypothesis.

Protocol (LOCKED before the run; see config.py's Phase 9 block):

  1. EVOLUTION runs on [CAMPAIGN_EVOLVE_START_MS, HOLDOUT_START_MS). Phase 6's
     runner carves its own internal validation window inside that span, which it
     scores thousands of times — that window is a SELECTION signal, NOT evidence,
     and the report labels it so.
  2. THE HOLDOUT is [HOLDOUT_START_MS, HOLDOUT_END_MS) and is never seen by any
     generation. Enforced twice: by construction (the runner is handed
     end_ms = HOLDOUT_START_MS, which is also Phase 6's frozen EVO_TRAIN_END
     ceiling) and by assertion (assert_no_holdout_overlap on every span this
     module builds or forwards).
  3. THE CHAMPION is chosen by a deterministic query with a declared tie-break
     and a PRE-REGISTERED trade floor (config.CAMPAIGN_MIN_CHAMPION_TRADES),
     then evaluated on the holdout EXACTLY ONCE through
     walkforward.walk_forward_pooled with a DEGENERATE grid, so no parameter is
     re-selected on holdout data.
  4. DSR is charged the CUMULATIVE trial count (contract §4): every row in
     state.db's trial_ledger whose end_ms lies at or before the holdout barrier,
     plus the gate run's own evaluations. This is expected to make the gate
     unpassable on the `dsr` condition; that is the finding, not a bug.
  5. The holdout is CONSUMED-ONCE: a holdout_consumption row is written BEFORE
     any holdout bar is read, so killing the process still burns the peek. A
     second run requires an explicit override that is itself logged.

This module DRIVES walkforward.py and evolution/ and edits neither. It contains
no gate threshold of its own: every threshold printed or compared is read from
walkforward.GATE_MIN_* or config.WF_MIN_TRADES.
"""

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field

from trading_bot import config
from trading_bot.backtest import trials, walkforward
from trading_bot.backtest.equity import (
    daily_returns,
    deflated_sharpe,
    max_drawdown,
)
from trading_bot.backtest.walkforward import DAY_MS, walk_forward_pooled
from trading_bot.data import correlation
from trading_bot.evolution import mutate as evo_mutate
from trading_bot.evolution import oracle as evo_oracle
from trading_bot.evolution import population as evo_population
from trading_bot.evolution import runner as evo_runner
from trading_bot.framework import registry
from trading_bot.framework.execute import run_graph_backtest

logger = logging.getLogger("trading_bot")

# Verdict taxonomy, declared in the Phase 9 plan and implemented here as code
# rather than judgement.
VERDICTS = ("A", "A_PRIME", "B1", "B2", "B3", "B4", "C")
BENCHMARK_CONTEXTS = ("BENCHMARK_POSITIVE", "BENCHMARK_NEGATIVE")

# Roles considered for champion selection. Finalist rows are audit RE-SCORES of
# graphs that are already present as offspring (verified: all 6 of Phase 6's
# finalist graph_hashes appear among its offspring), so including them would let
# one graph enter selection twice and let a re-score decide the winner.
CHAMPION_ROLES = ("offspring", "elite", "seed")


class CampaignError(RuntimeError):
    """Any protocol violation that must abort the campaign."""


class HoldoutViolation(CampaignError):
    """A span that would let evolution touch the final holdout."""


# ---------------------------------------------------------------------------
# Spans and the structural holdout barrier
# ---------------------------------------------------------------------------


def holdout_span() -> tuple[int, int]:
    """(start_ms, end_ms) of the final holdout. end_ms is EXCLUSIVE."""
    start, end = config.HOLDOUT_START_MS, config.HOLDOUT_END_MS
    if end - config.HOLDOUT_DAYS * DAY_MS != start:
        raise CampaignError(
            f"HOLDOUT_DAYS={config.HOLDOUT_DAYS} disagrees with [{start}, {end}); "
            f"the declared span and its length must match exactly or "
            f"walk_forward_pooled's tune_end lands off-boundary"
        )
    return start, end


def evolution_span() -> tuple[int, int]:
    """(start_ms, end_ms) evolution may see. end_ms IS the holdout start."""
    start = config.CAMPAIGN_EVOLVE_START_MS
    end, _ = holdout_span()
    if start >= end:
        raise CampaignError("evolution span is empty; check CAMPAIGN_EVOLVE_START_MS")
    return start, end


def assert_no_holdout_overlap(start_ms: int, end_ms: int, *, what: str) -> None:
    """Raise unless [start_ms, end_ms) is entirely before the holdout.

    This is the mechanism, not the convention. Called on every span this module
    builds and on every span it forwards to Phase 6's runner, so a config edit,
    a mutator that widens a window, or a hand-typed --start cannot quietly reach
    holdout bars. Any half-open interval that INTERSECTS the holdout is a
    violation, including one that merely straddles the boundary by 1 ms.
    """
    h_start, h_end = holdout_span()
    if start_ms < h_end and end_ms > h_start:
        raise HoldoutViolation(
            f"{what} span [{start_ms}, {end_ms}) intersects the holdout "
            f"[{h_start}, {h_end}); evolution must end at or before {h_start}. "
            f"Refusing to run."
        )


def fold_count() -> int:
    """Folds the gate run will form, computed the way walkforward.py does.

    Asserted rather than assumed (the plan's GOTCHA #2): if history moves, the
    fold count changes and the report must show the real number.
    """
    start, tune_end = evolution_span()
    train_ms = config.WF_TRAIN_DAYS * DAY_MS
    test_ms = config.WF_TEST_DAYS * DAY_MS
    n = 0
    t0 = start
    while t0 + train_ms + test_ms <= tune_end:
        n += 1
        t0 += test_ms
    if n < 1:
        raise CampaignError(
            f"evolution span [{start}, {tune_end}) forms zero folds at "
            f"WF_TRAIN_DAYS={config.WF_TRAIN_DAYS} / WF_TEST_DAYS="
            f"{config.WF_TEST_DAYS}"
        )
    return n


# ---------------------------------------------------------------------------
# The consume-once holdout ledger (DDL owned here, contract §6)
# ---------------------------------------------------------------------------

_HOLDOUT_DDL = """
CREATE TABLE IF NOT EXISTS holdout_consumption (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    run_index INTEGER NOT NULL,
    holdout_start_ms INTEGER NOT NULL,
    holdout_end_ms INTEGER NOT NULL,
    graph_hash TEXT,
    strategy_version TEXT,
    n_trials_charged INTEGER,
    opened_ts INTEGER NOT NULL,
    completed_ts INTEGER,
    gate_json TEXT,
    outcome_json TEXT,
    override_reason TEXT,
    detail TEXT
)
"""


def _ensure_schema(state_conn) -> None:
    """Idempotent DDL, owned here (contract §6: the module that uses the table
    owns its CREATE TABLE IF NOT EXISTS, not a central migration file)."""
    with evo_population._db_lock:
        state_conn.execute(_HOLDOUT_DDL)
        state_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_holdout_span "
            "ON holdout_consumption(holdout_start_ms, holdout_end_ms, kind)"
        )
        state_conn.commit()


def holdout_consumption_rows(state_conn, *, kind: str = "consume") -> list[dict]:
    """Every recorded touch of the CURRENTLY DECLARED holdout span, oldest first.

    Keyed on the SPAN, not the campaign: a new campaign id does not grant a
    fresh look at the same bars.
    """
    _ensure_schema(state_conn)
    h_start, h_end = holdout_span()
    with evo_population._db_lock:
        cur = state_conn.execute(
            "SELECT id, campaign_id, kind, run_index, holdout_start_ms, "
            "holdout_end_ms, graph_hash, strategy_version, n_trials_charged, "
            "opened_ts, completed_ts, gate_json, outcome_json, override_reason, "
            "detail FROM holdout_consumption WHERE holdout_start_ms = ? AND "
            "holdout_end_ms = ? AND kind = ? ORDER BY id ASC",
            (h_start, h_end, kind),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def holdout_is_consumed(state_conn) -> bool:
    """True when the declared holdout span already has a 'consume' row."""
    return bool(holdout_consumption_rows(state_conn, kind="consume"))


def open_holdout(
    state_conn,
    *,
    campaign_id: str,
    graph_hash: str,
    strategy_version: str,
    n_trials: int,
    override_reason: str | None = None,
) -> int:
    """Record the consumption BEFORE the holdout is read; return (id, run_index).

    Write-then-run is deliberate: if the process dies mid-evaluation the holdout
    is still burned. Otherwise `kill -9` after a glimpse of the numbers would be
    a free peek and the one-shot guarantee would be advisory.
    """
    if not config.HOLDOUT_LOCKED:
        raise CampaignError(
            "config.HOLDOUT_LOCKED is False; the holdout stage refuses to run. "
            "Flipping that flag is a recorded decision, not a convenience."
        )
    _ensure_schema(state_conn)
    h_start, h_end = holdout_span()
    prior = holdout_consumption_rows(state_conn, kind="consume")
    run_index = len(prior) + 1
    if run_index > 1 and not (override_reason or "").strip():
        raise CampaignError(
            f"holdout [{h_start}, {h_end}) already consumed "
            f"{len(prior)} time(s) (run_index={prior[-1]['run_index']}, opened "
            f"{prior[-1]['opened_ts']}). Re-running requires "
            f"--force-holdout-rerun REASON, which is recorded and stamps the "
            f"report NOT A CLEAN HOLDOUT."
        )
    now = int(time.time() * 1000)
    with evo_population._db_lock:
        cur = state_conn.execute(
            "INSERT INTO holdout_consumption (campaign_id, kind, run_index, "
            "holdout_start_ms, holdout_end_ms, graph_hash, strategy_version, "
            "n_trials_charged, opened_ts, override_reason) "
            "VALUES (?, 'consume', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                campaign_id, run_index, h_start, h_end, graph_hash,
                strategy_version, int(n_trials), now, override_reason,
            ),
        )
        state_conn.commit()
        row_id = cur.lastrowid
    logger.info(
        "*** HOLDOUT CONSUMED — recorded, run_index=%d, n_trials_charged=%d ***",
        run_index, n_trials,
    )
    return row_id, run_index


def close_holdout(state_conn, row_id: int, *, gate: dict, outcome: dict) -> None:
    """Attach the gate verdict, the full outcome payload and completed_ts."""
    with evo_population._db_lock:
        state_conn.execute(
            "UPDATE holdout_consumption SET completed_ts = ?, gate_json = ?, "
            "outcome_json = ? WHERE id = ?",
            (
                int(time.time() * 1000),
                json.dumps(gate, sort_keys=True),
                json.dumps(outcome, sort_keys=True, default=str),
                row_id,
            ),
        )
        state_conn.commit()


def record_violation(state_conn, *, campaign_id: str, start_ms: int, end_ms: int,
                     detail: str) -> None:
    """Audit a HoldoutViolation attempt (kind='violation', run_index=0).

    Violations do NOT consume the holdout — no bar was read — but they are
    recorded, because "the code tried to train on the holdout once" is exactly
    the kind of thing a report must not be able to omit.
    """
    _ensure_schema(state_conn)
    h_start, h_end = config.HOLDOUT_START_MS, config.HOLDOUT_END_MS
    with evo_population._db_lock:
        state_conn.execute(
            "INSERT INTO holdout_consumption (campaign_id, kind, run_index, "
            "holdout_start_ms, holdout_end_ms, opened_ts, detail) "
            "VALUES (?, 'violation', 0, ?, ?, ?, ?)",
            (campaign_id, h_start, h_end, int(time.time() * 1000),
             f"[{start_ms}, {end_ms}) {detail}"),
        )
        state_conn.commit()
    logger.warning("recorded holdout VIOLATION attempt: %s", detail)


def latest_outcome(state_conn) -> dict | None:
    """The most recent COMPLETED holdout outcome for the declared span, or None.

    The report generator's only data source: it must never re-run the gate.
    """
    rows = [r for r in holdout_consumption_rows(state_conn) if r["outcome_json"]]
    if not rows:
        return None
    payload = json.loads(rows[-1]["outcome_json"])
    payload["run_index"] = rows[-1]["run_index"]
    payload["override_reason"] = rows[-1]["override_reason"]
    payload["opened_ts"] = rows[-1]["opened_ts"]
    payload["completed_ts"] = rows[-1]["completed_ts"]
    return payload


# ---------------------------------------------------------------------------
# Cumulative trial accounting (contract §4)
# ---------------------------------------------------------------------------


def ledger_breakdown(state_conn) -> list[dict]:
    """Per-campaign row counts in state.db's trial_ledger, with span ceilings.

    Reported in full so the trial charge is inspectable rather than asserted.
    """
    trials.ensure_schema(state_conn)
    with evo_population._db_lock:
        rows = state_conn.execute(
            f"SELECT campaign, COUNT(*), MIN(start_ms), MAX(end_ms) FROM "
            f"{trials.LEDGER_TABLE} GROUP BY campaign ORDER BY COUNT(*) DESC, "
            f"campaign ASC"
        ).fetchall()
    return [
        {"campaign": r[0], "n": r[1], "min_start_ms": r[2], "max_end_ms": r[3],
         "past_barrier": r[3] > config.HOLDOUT_START_MS}
        for r in rows
    ]


def cumulative_trials(state_conn) -> int:
    """Every oracle evaluation performed on pre-barrier data (contract §4).

    THE PRE-REGISTERED RULE, chosen because it is mechanical rather than a
    judgement call: count every row in state.db's trial_ledger whose `end_ms`
    lies at or before HOLDOUT_START_MS — i.e. every evaluation this project ever
    performed on data the champion could have been selected on. It deliberately
    over-counts (it includes diagnostic re-scorings and other phases' report
    campaigns), because contract §4 says over-counting errs pessimistically and
    a narrower rule would require deciding, after the fact, which of our own
    evaluations "really" counted.

    NEVER counts in-process, and never creates a second ledger: a ledger that
    resets on restart is worse than none, because it looks authoritative.
    """
    trials.ensure_schema(state_conn)
    with evo_population._db_lock:
        cur = state_conn.execute(
            f"SELECT COUNT(*) FROM {trials.LEDGER_TABLE} WHERE end_ms <= ?",
            (config.HOLDOUT_START_MS,),
        )
        return int(cur.fetchone()[0])


def predicted_gate_trials(grid: dict, n_folds: int) -> int:
    """The gate run's OWN ledger rows: per fold, one train eval + one test eval.

    With the degenerate grid there are no neighbour probes
    (walkforward._neighbors returns [] for a one-value axis), so the count is
    exactly 2 * n_folds. Omitting these would make the ledger a lie about its
    own last step.
    """
    return 2 * len(walkforward._combos(grid)) * max(1, n_folds)


# ---------------------------------------------------------------------------
# Deterministic champion selection and the degenerate grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChampionRef:
    campaign_id: str
    member_id: str
    graph_hash: str
    generation: int
    fitness: float
    tier: str
    n_trades: int
    graph_json: str
    below_trade_floor: bool
    n_eligible: int
    n_scored: int


def contributing_campaigns(state_conn) -> list[dict]:
    """Evolution campaigns whose training ceiling respects the holdout barrier.

    This is the barrier proof at the campaign level: any campaign whose
    train_end_ms sits past HOLDOUT_START_MS is REFUSED as a champion source, so
    a champion can never come from a search that saw holdout bars.
    """
    evo_population.ensure_schema(state_conn)
    with evo_population._db_lock:
        rows = state_conn.execute(
            "SELECT campaign_id, seed, seed_graph_json, symbols_json, "
            "train_start_ms, train_end_ms, population, generations, status "
            "FROM campaigns ORDER BY started_ts ASC"
        ).fetchall()
    out = []
    for r in rows:
        if r[5] > config.HOLDOUT_START_MS:
            raise HoldoutViolation(
                f"campaign {r[0]!r} has train_end_ms={r[5]}, past the holdout "
                f"barrier {config.HOLDOUT_START_MS}; it cannot supply a champion"
            )
        out.append(
            {"campaign_id": r[0], "seed": r[1], "seed_graph_json": r[2],
             "symbols": json.loads(r[3]), "train_start_ms": r[4],
             "train_end_ms": r[5], "population": r[6], "generations": r[7],
             "status": r[8]}
        )
    if not out:
        raise CampaignError(
            "no evolution campaign in state.db; the champion pool is empty. "
            "Run `cli campaign --stage evolve` (or `cli evolve`) first."
        )
    return out


def select_champion(state_conn) -> ChampionRef:
    """The campaign's champion, chosen deterministically.

    Eligibility (PRE-REGISTERED, config.CAMPAIGN_MIN_CHAMPION_TRADES): a member
    must have recorded at least that many trades on its own evaluation window.
    See config.py's Phase 9 block for why the number is WF_MIN_TRADES and not a
    fresh one, and for the declared fallback when nobody clears it.

    Declared tie-break, in this order (committed BEFORE the run so a tie cannot
    be resolved by whoever is looking at the numbers):
      1. highest fitness (the gate oracle's score — contract §4.1: there is no
         other scoring function);
      2. then EARLIEST generation — an equally-fit earlier candidate has
         survived more selection rounds and consumed fewer degrees of freedom;
      3. then lexicographically smallest graph_hash — arbitrary but total, so
         the result never depends on SQLite row order or dict iteration.
    """
    campaigns = contributing_campaigns(state_conn)
    ids = [c["campaign_id"] for c in campaigns]
    placeholders = ", ".join("?" for _ in ids)
    roles = ", ".join("?" for _ in CHAMPION_ROLES)
    with evo_population._db_lock:
        rows = state_conn.execute(
            f"SELECT campaign_id, member_id, graph_hash, gen_index, fitness, "
            f"tier, n_trades, graph_json FROM population_members "
            f"WHERE campaign_id IN ({placeholders}) AND fitness IS NOT NULL "
            f"AND role IN ({roles}) "
            f"ORDER BY fitness DESC, gen_index ASC, graph_hash ASC",
            (*ids, *CHAMPION_ROLES),
        ).fetchall()
    if not rows:
        raise CampaignError(
            f"campaigns {ids} have no scored members; a campaign with no "
            f"champion is verdict C (ambiguous), never a silent fallback to the "
            f"seed strategy"
        )
    floor = config.CAMPAIGN_MIN_CHAMPION_TRADES
    eligible = [r for r in rows if (r[6] or 0) >= floor]
    below = not eligible
    pool = eligible or rows
    if below:
        logger.warning(
            "CHAMPION_BELOW_TRADE_FLOOR: no member of %d scored candidates "
            "reached the pre-registered floor of %d trades; falling back to the "
            "highest-fitness member and expecting sample_adequacy to FAIL. The "
            "floor is NOT lowered.",
            len(rows), floor,
        )
    best = pool[0]
    return ChampionRef(
        campaign_id=best[0], member_id=best[1], graph_hash=best[2],
        generation=int(best[3]), fitness=float(best[4]), tier=best[5] or "",
        n_trades=int(best[6] or 0), graph_json=best[7],
        below_trade_floor=below, n_eligible=len(eligible), n_scored=len(rows),
    )


def champion_graph(champion: ChampionRef):
    """The champion's StrategyGraph, rebuilt from its persisted JSON."""
    registry.load_all()
    return evo_mutate.graph_from_json(champion.graph_json)


def frozen_grid() -> dict[str, tuple]:
    """A DEGENERATE grid: exactly one value per axis, at the config default.

    walk_forward_pooled takes final parameters as the per-axis median_low of
    fold winners. With one value per axis, median_low provably returns that
    value, the on-grid assertion passes trivially, and the holdout is evaluated
    at EXACTLY the configuration evolution produced.

    Because a StrategyGraph carries its own node parameters, the only legal axes
    are walkforward._RUN_LEVEL_AXES; this reuses Phase 6's `evo_grid()` so the
    holdout run and every evolution evaluation are pinned to the same axis and
    the same value, rather than to two independently-typed literals.

    Rejected alternative: pass DEFAULT_GRID and let the folds re-tune. That
    would run the holdout at parameters evolution never selected, consume
    additional degrees of freedom for the verdict itself, and make "what was
    tested?" unanswerable.
    """
    return evo_oracle.evo_grid()


# ---------------------------------------------------------------------------
# The evolution stage — budget probe, run/resume, declared stopping rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetProbe:
    seconds_per_eval: float
    n_probe_evals: int
    projected_hours: float
    within_budget: bool
    population: int
    generations: int
    trials_charged: int


def probe_budget(conn, state_conn, *, campaign_id: str, n_evals: int = 3,
                 symbols=None) -> BudgetProbe:
    """MEASURE the cost of one oracle evaluation; never estimate it.

    Runs n_evals gate evaluations of the seed strategy on the TRAINING span only
    (assert_no_holdout_overlap first), times them, and projects
    CAMPAIGN_POPULATION_SIZE * CAMPAIGN_GENERATIONS * seconds_per_eval. These
    probe evaluations are REAL evaluations and ARE charged to the ledger — a
    measurement that dodges the ledger is exactly the side-channel contract §4.2
    forbids.

    Declared response to within_budget=False: reduce CAMPAIGN_GENERATIONS (and
    re-commit config) — never shorten the holdout, never shrink the fold
    windows, never lower WF_MIN_TRADES.
    """
    start, end = evolution_span()
    assert_no_holdout_overlap(start, end, what="probe")
    registry.load_all()
    campaigns = contributing_campaigns(state_conn)
    graph = evo_mutate.graph_from_json(campaigns[0]["seed_graph_json"])
    symbols = tuple(symbols or campaigns[0]["symbols"])
    window_days = config.EVO_WINDOW_DAYS
    window_start = end - window_days * DAY_MS
    assert_no_holdout_overlap(window_start, end, what="probe window")

    ledger = evo_oracle.TrialLedger(state_conn, campaign_id)
    oracle = evo_oracle.GateOracle(
        ledger, ohlcv_conn=conn, symbols=symbols,
        train_start_ms=start, train_end_ms=end,
    )
    elapsed = []
    for _ in range(max(1, n_evals)):
        t0 = time.monotonic()
        oracle.evaluate(graph, window_start_ms=window_start, window_end_ms=end)
        elapsed.append(time.monotonic() - t0)
    # median-free: the FIRST evaluation is the cold one and is kept in the mean,
    # deliberately — an overnight run pays cold costs too.
    per_eval = sum(elapsed) / len(elapsed)
    planned = config.CAMPAIGN_POPULATION_SIZE * config.CAMPAIGN_GENERATIONS
    hours = per_eval * planned / 3600.0
    return BudgetProbe(
        seconds_per_eval=per_eval, n_probe_evals=len(elapsed),
        projected_hours=hours,
        within_budget=hours <= config.CAMPAIGN_WALL_CLOCK_BUDGET_HOURS,
        population=config.CAMPAIGN_POPULATION_SIZE,
        generations=config.CAMPAIGN_GENERATIONS,
        trials_charged=ledger.count(),
    )


@dataclass(frozen=True)
class EvolutionStageResult:
    campaign_id: str
    last_generation: int
    stop_reason: str
    wall_seconds: float
    trials_before: int
    trials_after: int
    generations_planned: int
    resumed: bool


def _phase9_campaign_row(state_conn, campaign_id: str):
    return evo_population.load_campaign(state_conn, campaign_id)


def run_evolution_stage(conn, state_conn, *, campaign_id: str | None = None,
                        resume: bool = True, progress=logger.info,
                        symbols=None) -> EvolutionStageResult:
    """Run (or resume) Phase 9's own evolution top-up on the training span.

    Every span forwarded to Phase 6's runner passes through
    assert_no_holdout_overlap first, and `end` is evolution_span()[1] — i.e.
    HOLDOUT_START_MS exactly, which is also Phase 6's frozen EVO_TRAIN_END, so
    the runner's own ceiling check agrees with ours.

    Stopping rule, evaluated in this fixed order (declared in config, so "stop
    when the numbers look good" is not reachable):
      1. generation == CAMPAIGN_GENERATIONS       -> COMPLETE
      2. elapsed >= budget                        -> BUDGET_EXHAUSTED
      3. best fitness unimproved for
         CAMPAIGN_PATIENCE_GENERATIONS            -> CONVERGED
    All three are completed evolution stages; the holdout runs after any of
    them, and the report records WHICH one fired.

    Resume: state lives in Phase 6's `generations` / `population_members` tables
    in state.db, so a campaign that dies mid-run continues at the last
    checkpointed generation with the SAME CAMPAIGN_SEED.
    """
    start, end = evolution_span()
    assert_no_holdout_overlap(start, end, what="evolution")
    registry.load_all()
    prior = contributing_campaigns(state_conn)
    seed_graph = evo_mutate.graph_from_json(prior[0]["seed_graph_json"])
    # Fitness comparability: the union champion pool is ranked on one number, so
    # the top-up must be scored on the SAME symbol set the existing campaign
    # used. Scoring it on the 9-symbol holdout set would make its fitness
    # incomparable with Phase 6's 194 members.
    symbols = tuple(symbols or prior[0]["symbols"])

    existing = None
    if campaign_id and resume:
        existing = _phase9_campaign_row(state_conn, campaign_id)
    if existing is not None and existing.seed != config.CAMPAIGN_SEED:
        raise CampaignError(
            f"campaign {campaign_id!r} was seeded {existing.seed}, but "
            f"config.CAMPAIGN_SEED is {config.CAMPAIGN_SEED}. A resume with a "
            f"different seed is a NEW experiment, not a continuation."
        )

    ledger_before = cumulative_trials(state_conn)
    t0 = time.monotonic()
    summary = evo_runner.run_campaign(
        seed_graph=None if existing is not None else seed_graph,
        seed=config.CAMPAIGN_SEED,
        symbols=symbols,
        population_size=config.CAMPAIGN_POPULATION_SIZE,
        generations=config.CAMPAIGN_GENERATIONS,
        train_start_ms=start,
        train_end_ms=end,
        resume_campaign_id=campaign_id if existing is not None else None,
        state_conn=state_conn,
        progress=progress,
    )
    wall = time.monotonic() - t0
    cid = summary["campaign_id"]
    last_done = evo_population.last_completed_generation(state_conn, cid)
    last_done = -1 if last_done is None else last_done

    stop_reason = "COMPLETE"
    if last_done + 1 < config.CAMPAIGN_GENERATIONS:
        if wall / 3600.0 >= config.CAMPAIGN_WALL_CLOCK_BUDGET_HOURS:
            stop_reason = "BUDGET_EXHAUSTED"
        else:
            fits = [
                r["best_fitness"] for r in
                evo_population.generation_rows(state_conn, cid)
                if r.get("best_fitness") is not None
            ]
            tail = fits[-config.CAMPAIGN_PATIENCE_GENERATIONS:]
            if (len(tail) >= config.CAMPAIGN_PATIENCE_GENERATIONS
                    and max(tail) <= tail[0]):
                stop_reason = "CONVERGED"
            else:
                stop_reason = "ABORTED"
    return EvolutionStageResult(
        campaign_id=cid, last_generation=last_done, stop_reason=stop_reason,
        wall_seconds=wall, trials_before=ledger_before,
        trials_after=cumulative_trials(state_conn),
        generations_planned=config.CAMPAIGN_GENERATIONS,
        resumed=existing is not None,
    )


# ---------------------------------------------------------------------------
# Sample-adequacy projection (measured on TRAINING data only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleProjection:
    n_symbols: int
    evolution_days: int
    measured_trades_per_symbol_day: float
    measured_trades_total: int
    holdout_days: int
    projected_raw_trades: float
    mean_pairwise_correlation: float
    effective_n: float
    projected_independent_trades: float
    required_rate_for_independent_floor: float
    raw_floor: int
    raw_floor_reachable: bool
    independent_floor_reachable: bool
    reference_rate_v020_realized: float = 0.085
    reference_rate_kl4_pessimistic: float = 0.050


def sample_adequacy_projection(conn, *, champion_graph_obj,
                               symbols=None) -> SampleProjection:
    """Arithmetic, not hope (KNOWN-LIMITATIONS §4).

    The rate is MEASURED by running the CHAMPION over the evolution span — a
    span it has already seen, so this costs no holdout information — and
    dividing trades by symbols by days.

    effective_n is READ from Phase 2's data/correlation.py (Kish effective N
    over the full pairwise matrix), never re-derived here from a single mean.

    The identity worth internalising: with n symbols each producing k trades,
    raw pooled trades = n*k but independent-equivalent trades ~= k *
    effective_n — so independent-equivalent does NOT grow with symbol count,
    only the raw count does. §0b's "rows, not information", literally.

    This projection is REPORTED and LOGGED. It is never a gate, and its failure
    NEVER triggers lowering WF_MIN_TRADES or shrinking the holdout.
    """
    symbols = tuple(symbols or config.CAMPAIGN_SYMBOLS)
    e_start, e_end = evolution_span()
    assert_no_holdout_overlap(e_start, e_end, what="sample projection")
    n_trades = 0
    for symbol in symbols:
        n_trades += len(
            run_graph_backtest(
                conn, champion_graph_obj, symbol,
                start_ms=e_start, end_ms=e_end - 1,
            )
        )
    evo_days = (e_end - e_start) // DAY_MS
    rate = n_trades / len(symbols) / evo_days if evo_days else 0.0

    returns = correlation.daily_return_frame(
        conn, symbols, start_ms=config.date_to_ms(config.CORRELATION_START),
        end_ms=e_end - 1,
    )
    corr = correlation.correlation_matrix(returns)
    _, r_bar, eff_n = correlation.effective_n(corr, list(symbols))

    k = rate * config.HOLDOUT_DAYS
    raw = k * len(symbols)
    indep = k * eff_n
    floor = config.WF_MIN_TRADES
    return SampleProjection(
        n_symbols=len(symbols), evolution_days=int(evo_days),
        measured_trades_per_symbol_day=rate, measured_trades_total=n_trades,
        holdout_days=config.HOLDOUT_DAYS, projected_raw_trades=raw,
        mean_pairwise_correlation=r_bar, effective_n=eff_n,
        projected_independent_trades=indep,
        required_rate_for_independent_floor=floor / eff_n / config.HOLDOUT_DAYS,
        raw_floor=floor, raw_floor_reachable=raw >= floor,
        independent_floor_reachable=indep >= floor,
    )


# ---------------------------------------------------------------------------
# What the gate does NOT see
# ---------------------------------------------------------------------------


def benchmark_context(benchmark) -> str:
    """How much a "beats the benchmark" PASS is actually worth.

    BENCHMARK_POSITIVE iff the basket's ann_return_pct > 0 AND sharpe > 0;
    otherwise BENCHMARK_NEGATIVE.

    Measured motivation: Phase 1 found the equal-weight basket's Sharpe over
    v0.2.0's 90-day OOS window was -1.303 — buy-and-hold LOST MONEY there, so
    both beats_benchmark_* conditions PASS trivially. In a bear window almost
    anything beats inaction, and KNOWN-LIMITATIONS §0's complaint was about
    blessing something WORSE than inaction. This label keeps that visible.

    DELIBERATELY NOT A GATE CONDITION. GATE_CONDITIONS stays the contract's
    seven; requiring a positive benchmark would be moving the gate, and beating
    a falling market is a real if weaker result.
    """
    b = benchmark.basket if hasattr(benchmark, "basket") else benchmark
    ann, sharpe = b.get("ann_return_pct"), b.get("sharpe")
    if ann is not None and sharpe is not None and ann > 0 and sharpe > 0:
        return "BENCHMARK_POSITIVE"
    return "BENCHMARK_NEGATIVE"


def collect_diagnostics(conn, state_conn, result, champion, graph_obj,
                        projection) -> dict:
    """Everything a single PASS/FAIL bit hides. Every key MEASURED.

    Requires the holdout to already be OPEN: full_span_max_drawdown_pct spans
    the holdout, so computing it before open_holdout would be a free peek.
    """
    if not holdout_is_consumed(state_conn):
        raise CampaignError(
            "collect_diagnostics touches holdout bars and must never be "
            "reachable before open_holdout() has burned the holdout"
        )
    e_start, _ = evolution_span()
    h_start, h_end = holdout_span()
    symbols = tuple(config.CAMPAIGN_SYMBOLS)

    full_trades: list = []
    per_symbol_end: dict[str, float | None] = {}
    for symbol in symbols:
        t = run_graph_backtest(
            conn, graph_obj, symbol, start_ms=e_start, end_ms=h_end - 1,
            max_hold_bars=config.MAX_HOLD_BARS_TRIGGER,
        )
        full_trades.extend(t)
        equity = 1.0
        for tr in sorted(t, key=lambda x: x.exit_ts):
            equity *= 1.0 + tr.pnl_pct
        per_symbol_end[symbol] = equity
    full_rets = daily_returns(full_trades, e_start, h_end,
                              attribution=config.PNL_ATTRIBUTION_MODE)
    full_dd = max_drawdown(full_rets)

    # The holdout moments come from result.oos_equity, NOT from a second series
    # built here: two implementations of one number is how a report ends up
    # disagreeing with the gate it is reporting on.
    eq = result.oos_equity
    skew, kurt = eq["skew"], eq["kurtosis"]
    sr_daily, n_obs = eq["daily_sharpe"], eq["n_days"]
    dsr_1 = (deflated_sharpe(sr_daily, 1, n_obs, skew, kurt)
             if sr_daily is not None and n_obs >= 2 else None)
    default_charge = len(walkforward._combos(frozen_grid())) * max(1, len(result.folds))
    dsr_default = (deflated_sharpe(sr_daily, default_charge, n_obs, skew, kurt)
                   if sr_daily is not None and n_obs >= 2 else None)

    fold_exp = [f.test_metrics["expectancy_pct"] for f in result.folds]
    n_neg = sum(1 for e in fold_exp if e is not None and e < 0)
    fell_back = sum(1 for f in result.folds if f.train_expectancy is None)

    basket = dict(result.benchmark.basket)
    realized_rate = (
        result.oos_metrics["n_trades"] / len(symbols) / config.HOLDOUT_DAYS
    )
    ann = eq["ann_return_pct"]
    round_trip = 2 * (config.FEE_PCT + config.SLIPPAGE_PCT)
    return {
        "full_span_start_ms": e_start,
        "full_span_end_ms": h_end,
        "full_span_n_trades": len(full_trades),
        "full_span_max_drawdown_pct": full_dd,
        "holdout_max_drawdown_pct": eq["max_drawdown_pct"],
        "per_symbol_end_state": per_symbol_end,
        "fold_test_expectancy": fold_exp,
        "fold_n_negative": n_neg,
        "fold_n_total": len(result.folds),
        "folds_fell_back_to_defaults": fell_back,
        "benchmark_basket_absolute": basket,
        "benchmark_per_symbol_absolute": {
            s: dict(v) for s, v in result.benchmark.per_symbol.items()
        },
        "benchmark_context": benchmark_context(result.benchmark),
        "holdout_daily_n_obs": n_obs,
        "holdout_daily_skew": skew,
        "holdout_daily_kurtosis": kurt,
        "phase1_measured_skew_after_fix": 1.4034,
        "phase1_measured_kurtosis_after_fix": 15.5429,
        "pnl_attribution_mode": config.PNL_ATTRIBUTION_MODE,
        "dsr_at_n_trials_1": dsr_1,
        "dsr_at_grid_x_folds": dsr_default,
        "dsr_charged": eq["dsr"],
        "n_trials_charged": result.n_trials_used,
        "n_trials_walkforward_default": default_charge,
        "realized_holdout_rate": realized_rate,
        "projected_holdout_rate": projection.measured_trades_per_symbol_day,
        "mean_cost_pct_per_trade_round_trip": round_trip,
        "northstar_ann_return_gap": (
            None if ann is None else ann - config.CAMPAIGN_NORTHSTAR_ANN_RETURN
        ),
        "unsized_drawdown_note": (
            "Every drawdown here is UNSIZED: equal notional, one open trade per "
            "symbol, no position sizing (KNOWN-LIMITATIONS §7)."
        ),
        "fold_evidence_note": (
            "Fold test metrics are DIAGNOSIS, not evidence: the fold sweep runs "
            "inside the evolution span, on data the search has already seen. "
            "Only result.oos_* is evidence."
        ),
    }


# ---------------------------------------------------------------------------
# Verdict classifier
# ---------------------------------------------------------------------------


def classify_verdict(gate: dict, oos_equity: dict, projection=None) -> str:
    """Map the 7-condition gate onto the taxonomy declared in the plan.

    Returns the id only. Callers pair it with benchmark_context() and render
    '<id> / <context>'; the two are never collapsed into one label.

    Code, not judgement — the order of tests is fixed so exactly one id applies.
    B3 is tested before B4 and B1 because losing to buy-and-hold makes the
    significance question moot: §0's lesson is that a gate which cannot see
    inaction can bless something worse than it.
    """
    missing = [c for c in walkforward.GATE_CONDITIONS if c not in gate]
    if missing:
        return "C"
    if all(gate.values()):
        ann = oos_equity.get("ann_return_pct")
        if ann is not None and ann > config.CAMPAIGN_NORTHSTAR_ANN_RETURN:
            return "A"
        return "A_PRIME"
    if not gate["beats_benchmark_return"] or not gate["beats_benchmark_sharpe"]:
        return "B3"
    if not gate["sample_adequacy"]:
        return "B4"
    failed = [c for c, ok in gate.items() if not ok]
    if failed == ["dsr"]:
        return "B1"
    if "dsr" in failed:
        return "B2"
    return "C"


# ---------------------------------------------------------------------------
# The one-shot holdout stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutOutcome:
    campaign_id: str
    champion: ChampionRef
    run_index: int
    n_trials_charged: int
    n_folds: int
    result: object
    gate: dict
    benchmark: object
    diagnostics: dict
    projection: SampleProjection
    verdict: str
    benchmark_context: str
    stop_reason: str
    wall_seconds: float
    ledger_breakdown: list = field(default_factory=list)


def run_holdout_gate(conn, state_conn, *, campaign_id: str,
                     force_reason: str | None = None,
                     stop_reason: str | None = None) -> HoldoutOutcome:
    """THE ONE-SHOT RUN. Everything before this point is preparation.

    Order of operations IS the protocol and must not be rearranged:
      1. refuse unless config.HOLDOUT_LOCKED (inside open_holdout);
      2. select_champion (deterministic; no champion => CampaignError);
      3. cumulative_trials + the gate's own predicted rows -> n_trials;
      4. sample_adequacy_projection on the TRAINING span — logged, never a gate;
      5. open_holdout(...)  <-- the holdout is BURNED HERE, before any holdout
         bar is read;
      6. walk_forward_pooled with the degenerate grid and the champion graph;
      7. ASSERT the harness's own boundary (oos_start/oos_end);
      8. collect_diagnostics; 9. classify_verdict; 10. close_holdout.

    min_trades is NOT passed: the floor stays config.WF_MIN_TRADES. There is no
    code path in this module that lowers it.
    """
    t_start = time.monotonic()
    if stop_reason is None:
        stop_reason = evolution_stop_reason(state_conn)
    h_start, h_end = holdout_span()
    n_folds = fold_count()
    champion = select_champion(state_conn)
    graph_obj = champion_graph(champion)
    grid = frozen_grid()

    prior_trials = cumulative_trials(state_conn)
    gate_trials = predicted_gate_trials(grid, n_folds)
    n_trials = prior_trials + gate_trials
    breakdown = ledger_breakdown(state_conn)

    projection = sample_adequacy_projection(
        conn, champion_graph_obj=graph_obj, symbols=config.CAMPAIGN_SYMBOLS
    )
    logger.info(
        "sample projection: %d symbols x %.4f tr/symbol/day x %d d = %.1f raw; "
        "independent-equiv %.1f (effN %.4f measured) vs floor %d",
        projection.n_symbols, projection.measured_trades_per_symbol_day,
        projection.holdout_days, projection.projected_raw_trades,
        projection.projected_independent_trades, projection.effective_n,
        projection.raw_floor,
    )

    row_id, run_index = open_holdout(
        state_conn, campaign_id=campaign_id, graph_hash=champion.graph_hash,
        strategy_version=champion.member_id, n_trials=n_trials,
        override_reason=force_reason,
    )

    # BOTH `ledger=` and `n_trials=` are passed, deliberately:
    #   * n_trials= fixes what the DSR is charged (the cumulative pre-barrier
    #     count plus this run's own evaluations), so the in-process default
    #     (len(combos) * len(folds) == 12) can never apply and flatter the
    #     result by ~2.5 annualised Sharpe points of required threshold;
    #   * ledger= makes the gate run's OWN evaluations persist as rows, so the
    #     ledger is not a lie about its own last step. walkforward resolves
    #     n_trials first, so supplying both is unambiguous.
    gate_ledger = trials.TrialLedger(state_conn, f"{campaign_id}-gate")
    result = walk_forward_pooled(
        conn, list(config.CAMPAIGN_SYMBOLS),
        start_ms=config.CAMPAIGN_EVOLVE_START_MS,
        end_ms=config.HOLDOUT_END_MS,
        grid=grid,
        oos_days=config.HOLDOUT_DAYS,
        n_trials=n_trials,
        ledger=gate_ledger,
        strategy=graph_obj,
    )
    if result.oos_start != h_start or result.oos_end != h_end:
        raise CampaignError(
            f"the harness placed the one-shot OOS at [{result.oos_start}, "
            f"{result.oos_end}), not the declared holdout [{h_start}, {h_end}). "
            f"The boundary is a CHECK, not a convention."
        )
    if len(result.folds) != n_folds:
        raise CampaignError(
            f"expected {n_folds} folds, the harness formed {len(result.folds)}; "
            f"history moved under the pre-registered spans"
        )

    diagnostics = collect_diagnostics(
        conn, state_conn, result, champion, graph_obj, projection
    )
    verdict = classify_verdict(result.gate, result.oos_equity, projection)
    context = benchmark_context(result.benchmark)
    outcome = HoldoutOutcome(
        campaign_id=campaign_id, champion=champion, run_index=run_index,
        n_trials_charged=n_trials, n_folds=len(result.folds), result=result,
        gate=dict(result.gate), benchmark=result.benchmark,
        diagnostics=diagnostics, projection=projection, verdict=verdict,
        benchmark_context=context, stop_reason=stop_reason,
        wall_seconds=time.monotonic() - t_start, ledger_breakdown=breakdown,
    )
    close_holdout(state_conn, row_id, gate=dict(result.gate),
                  outcome=serialize_outcome(outcome))
    return outcome


def serialize_outcome(outcome: HoldoutOutcome) -> dict:
    """A JSON-able snapshot of everything the report needs.

    Persisted so `campaign --stage report` reads state.db and NEVER re-runs the
    gate: a report generator that re-runs the gate is a second peek at the
    holdout wearing a reporting hat.
    """
    r = outcome.result
    return {
        "campaign_id": outcome.campaign_id,
        "champion": asdict(outcome.champion),
        "run_index": outcome.run_index,
        "n_trials_charged": outcome.n_trials_charged,
        "n_folds": outcome.n_folds,
        "gate": outcome.gate,
        "gate_conditions": list(walkforward.GATE_CONDITIONS),
        "passed": r.passed,
        "verdict": outcome.verdict,
        "benchmark_context": outcome.benchmark_context,
        "stop_reason": outcome.stop_reason,
        "wall_seconds": outcome.wall_seconds,
        "spans": {
            "evolve_start_ms": config.CAMPAIGN_EVOLVE_START_MS,
            "holdout_start_ms": config.HOLDOUT_START_MS,
            "holdout_end_ms": config.HOLDOUT_END_MS,
            "holdout_days": config.HOLDOUT_DAYS,
            "oos_start": r.oos_start,
            "oos_end": r.oos_end,
        },
        "symbols": list(config.CAMPAIGN_SYMBOLS),
        "seed": config.CAMPAIGN_SEED,
        "oos_metrics": r.oos_metrics,
        "oos_equity": r.oos_equity,
        "per_symbol_expectancy": r.per_symbol_expectancy,
        "final_max_hold_bars": r.final_max_hold_bars,
        "benchmark": {
            "basket": dict(r.benchmark.basket),
            "per_symbol": {s: dict(v) for s, v in r.benchmark.per_symbol.items()},
            "start_ms": r.benchmark.start_ms,
            "end_ms": r.benchmark.end_ms,
        },
        "diagnostics": outcome.diagnostics,
        "projection": asdict(outcome.projection),
        "ledger_breakdown": outcome.ledger_breakdown,
        "thresholds": {
            "GATE_MIN_SHARPE": walkforward.GATE_MIN_SHARPE,
            "GATE_MIN_DSR": walkforward.GATE_MIN_DSR,
            "GATE_MAX_DRAWDOWN": walkforward.GATE_MAX_DRAWDOWN,
            "WF_MIN_TRADES": config.WF_MIN_TRADES,
            "CAMPAIGN_MIN_CHAMPION_TRADES": config.CAMPAIGN_MIN_CHAMPION_TRADES,
            "CAMPAIGN_NORTHSTAR_ANN_RETURN": config.CAMPAIGN_NORTHSTAR_ANN_RETURN,
        },
    }


def required_annual_sharpe(n_trials: int, n_obs: int, skew: float = 0.0,
                           kurt: float = 3.0, target: float | None = None) -> float:
    """The annualised Sharpe `dsr > target` demands at this trial count.

    Bisection over equity.deflated_sharpe — no price data involved, so this is
    never a holdout peek. `target` defaults to walkforward.GATE_MIN_DSR so the
    threshold is read from source, never retyped.
    """
    if target is None:
        target = walkforward.GATE_MIN_DSR
    lo, hi = 0.0, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        d = deflated_sharpe(mid, n_trials, n_obs, skew, kurt)
        lo, hi = (mid, hi) if (d is None or d < target) else (lo, mid)
    return hi * math.sqrt(365)


def northstar_gap(oos_equity: dict) -> tuple[float | None, bool]:
    """(ann_return_pct, met) against CAMPAIGN_NORTHSTAR_ANN_RETURN.

    REPORTED, NOT GATED: contract §4's seven conditions contain no return
    threshold, so a passing gate does not imply the northstar and vice versa.
    """
    ann = oos_equity.get("ann_return_pct")
    return ann, (ann is not None and ann > config.CAMPAIGN_NORTHSTAR_ANN_RETURN)


def evolution_stop_reason(state_conn) -> str:
    """The stop reason of the most recent contributing campaign, from state.db.

    Needed because the stages are separate processes: `--stage holdout` invoked
    on its own cannot be handed the in-memory stop reason from an earlier
    `--stage evolve`, and re-deriving it from the persisted `campaigns` /
    `generations` rows is more honest than persisting "NOT_RUN". Reads rows; runs
    nothing.
    """
    campaigns = contributing_campaigns(state_conn)
    latest = campaigns[-1]
    last_done = evo_population.last_completed_generation(
        state_conn, latest["campaign_id"]
    )
    last_done = -1 if last_done is None else last_done
    if latest["status"] == "done" and last_done + 1 >= latest["generations"]:
        return f"COMPLETE ({latest['campaign_id']}, {last_done + 1} generations)"
    return f"{latest['status'].upper()} ({latest['campaign_id']}, gen {last_done})"


def default_campaign_id(now_ms: int | None = None) -> str:
    """A new campaign id derived from the UTC date, mirroring Phase 6's shape."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    return time.strftime("phase9-%Y%m%d", time.gmtime(now_ms / 1000))
