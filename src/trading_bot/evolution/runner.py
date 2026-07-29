"""
The campaign loop: spawn-safe process pool, generation cycle, audit round,
calibration and resume (v0.3.0 Phase 6).

THE THREE macOS PROCESS-PARALLELISM TRAPS THIS FILE EXISTS TO HANDLE:

 1. sqlite3.Connection is neither fork-safe nor shareable. No connection is ever
    a task argument or a Campaign field; each worker connects itself in
    _worker_init. ohlcv.db is opened READ-ONLY (file:...?mode=ro, uri=True) so a
    worker physically cannot write price history — the identical call is proven on
    this database by scripts/bruteforce/core.py, and WAL makes concurrent readers
    safe. state.db: the PARENT is the only writer of campaigns / generations /
    population_members; workers write only trial_ledger rows, on their own
    connection, with PRAGMA busy_timeout and retries classified by
    storage.is_transient_db_error, so a locked file retries and a corrupt one
    propagates. `generations.db_retries` records the cost, so "contention is
    negligible" stays a measurement.

 2. engine._CACHE is PER-PROCESS, so the memo that makes a pooled walk-forward
    tractable is worker-local. Therefore: a PERSISTENT pool, and
    max_tasks_per_child is never set. The first task per worker pays the cold cost
    (frame decode, regime labels, ATR, channels) and later tasks are warm; one
    member per task is right BECAUSE workers persist, since batching would only
    save IPC microseconds against a multi-second evaluation. engine.clear_caches()
    is never called in a worker. calibrate() reports cold and warm separately so
    this stays measured rather than assumed.

 3. macOS defaults to spawn. mp_context is passed EXPLICITLY so behaviour matches
    on Linux too, and the consequences are accepted: workers re-import everything,
    __main__ never runs, module state starts empty, every payload must be
    picklable. Hence initializer=_worker_init, a belt-and-braces re-init inside
    the task, and PLAIN DICTS across the boundary — never a StrategyGraph,
    DataFrame, Connection, random.Random or registry object.

RESULTS ARE MERGED IN MEMBER-INDEX ORDER, never as_completed order. as_completed
yields by completion time, so ranking on it would make selection depend on
machine speed; rank_key's -member_index is the second line of defence.
"""

import dataclasses
import json
import logging
import math
import multiprocessing
import os
import signal
import sqlite3
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from trading_bot import config
from trading_bot.data import statestore, storage
from trading_bot.evolution import mutate, oracle, population, tournament
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.graph import StrategyGraph

logger = logging.getLogger("trading_bot")

DAY_MS = 86_400_000

__all__ = ("calibrate", "run_campaign", "training_span", "window_arithmetic")

# Worker-local state. Empty in a freshly spawned process; _worker_init fills it
# exactly once per worker, and _evaluate_task re-checks (belt and braces) because
# under spawn there is no guarantee an initializer ran for a re-created worker.
_W: dict = {}


# --------------------------------------------------------------------------- #
# Span arithmetic — derived from config, never guessed
# --------------------------------------------------------------------------- #


def training_span(train_start_ms: int | None = None,
                  train_end_ms: int | None = None) -> tuple[int, int]:
    """The campaign's training bounds, defaulting to config.EVO_TRAIN_*.

    EVO_TRAIN_END is a FROZEN CEILING, never None and never "now". An implicit
    end would swallow Phase 9's holdout, and because today's 1d bar is still
    forming, "now" would also treat an unfinished candle as closed — a lookahead
    bug that fails in the flattering direction.
    """
    start = (
        config.date_to_ms(config.EVO_TRAIN_START)
        if train_start_ms is None
        else int(train_start_ms)
    )
    end = (
        config.date_to_ms(config.EVO_TRAIN_END)
        if train_end_ms is None
        else int(train_end_ms)
    )
    if end <= start:
        raise ValueError(
            f"training span is empty: {start} >= {end}. EVO_TRAIN_START/END are "
            f"{config.EVO_TRAIN_START} / {config.EVO_TRAIN_END}."
        )
    return start, end


def window_arithmetic(*, window_days: int | None = None,
                      train_start_ms: int | None = None,
                      train_end_ms: int | None = None) -> dict:
    """Every derived number the window policy depends on, in one place.

    Returned rather than printed so a test can pin it: the minimum window
    (WF_TRAIN + WF_TEST + WF_OOS) is what walk_forward_pooled raises below, the
    fold count follows from the tuning span, and `jitter_days` is how much of the
    training span a generation's start may move within.
    """
    start, end = training_span(train_start_ms, train_end_ms)
    window_days = config.EVO_WINDOW_DAYS if window_days is None else int(window_days)
    min_window = config.WF_TRAIN_DAYS + config.WF_TEST_DAYS + config.WF_OOS_DAYS
    if window_days < min_window:
        raise ValueError(
            f"EVO_WINDOW_DAYS={window_days} is below the walk-forward minimum "
            f"{min_window} (WF_TRAIN {config.WF_TRAIN_DAYS} + WF_TEST "
            f"{config.WF_TEST_DAYS} + WF_OOS {config.WF_OOS_DAYS}); "
            f"walk_forward_pooled would raise for every candidate"
        )
    span_days = (end - start) // DAY_MS
    tune_days = window_days - config.WF_OOS_DAYS
    n_folds = 0
    t0 = 0
    while t0 + config.WF_TRAIN_DAYS + config.WF_TEST_DAYS <= tune_days:
        n_folds += 1
        t0 += config.WF_TEST_DAYS
    return {
        "train_start_ms": start,
        "train_end_ms": end,
        "span_days": int(span_days),
        "window_days": window_days,
        "min_window_days": min_window,
        "tune_days": tune_days,
        "n_folds": n_folds,
        "oos_days": config.WF_OOS_DAYS,
        "jitter_days": int(span_days) - window_days,
    }


def _draw_window(rng, arith: dict) -> tuple[int, int]:
    """One generation's evaluation window, shared by every member (A3 fairness).

    Jitter is partial-data training (pivot guide method §2): each generation sees
    a different slice, so a candidate cannot be tuned to one regime and still win
    repeatedly. With EVO_WINDOW_JITTER False the window is pinned to the LATEST
    legal position — reproducible, but one regime.
    """
    window_ms = arith["window_days"] * DAY_MS
    latest_start = arith["train_end_ms"] - window_ms
    if not config.EVO_WINDOW_JITTER or latest_start <= arith["train_start_ms"]:
        start = latest_start
    else:
        # Snapped to whole days so a window boundary always lands on a 1d bar
        # open, never mid-candle.
        jitter = rng.randrange(0, arith["jitter_days"] + 1)
        start = arith["train_start_ms"] + jitter * DAY_MS
    return int(start), int(start + window_ms)


def _audit_window(arith: dict) -> tuple[int, int]:
    """The FIXED audit window, declared at campaign start (A3).

    The latest legal window inside the training span, so the audit round scores
    finalists on the most recent data evolution was allowed to see — and it is
    written into the campaigns row up front precisely so it cannot later be chosen
    to flatter a winner.
    """
    window_ms = arith["window_days"] * DAY_MS
    return int(arith["train_end_ms"] - window_ms), int(arith["train_end_ms"])


# --------------------------------------------------------------------------- #
# Worker side — pure, no RNG, no selection, no campaign-table writes
# --------------------------------------------------------------------------- #


def _connect_ohlcv_readonly(db_path: str):
    """Read-only URI connection, with a logged fallback.

    A worker that physically cannot write ohlcv.db cannot corrupt 117 MB of
    irreplaceable history no matter what a future edit does. The fallback exists
    because a URI open can fail on an exotic filesystem; it WARNs rather than
    failing silently, so a run that lost the guarantee says so.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                               check_same_thread=False)
        conn.execute("SELECT 1 FROM ohlcv LIMIT 1")
        return conn
    except sqlite3.OperationalError as exc:
        logger.warning(
            "read-only open of %s failed (%s); falling back to a writable "
            "connection — the physical write guarantee is LOST for this worker",
            db_path, exc,
        )
        return storage.connect(db_path)


def _worker_init(campaign_json: str) -> None:
    """Runs once per worker: registry, connections, oracle. Idempotent.

    Import errors from plugins.load_all() are FATAL here exactly as they are in
    the registry: a family silently missing from a population would read as
    "tested and found wanting".
    """
    if _W.get("campaign_json") == campaign_json:
        return
    spec = json.loads(campaign_json)
    registry.load_all()
    ohlcv = _connect_ohlcv_readonly(spec["ohlcv_path"])
    state = statestore.connect(spec["state_path"])
    state.execute(f"PRAGMA busy_timeout = {int(config.EVO_DB_BUSY_TIMEOUT_MS)}")
    ledger = oracle.TrialLedger(state, spec["campaign_id"])
    _W.clear()
    _W.update(
        {
            "campaign_json": campaign_json,
            "ohlcv": ohlcv,
            "state": state,
            "ledger": ledger,
            "oracle": oracle.GateOracle(
                ledger,
                ohlcv_conn=ohlcv,
                symbols=spec["symbols"],
                train_start_ms=spec["train_start_ms"],
                train_end_ms=spec["train_end_ms"],
                train_days=spec.get("train_days"),
                test_days=spec.get("test_days"),
                oos_days=spec.get("oos_days"),
                min_trades=spec.get("min_trades"),
            ),
        }
    )


def _evaluate_task(payload: dict) -> dict:
    """Score one member. PURE: no RNG, no selection, no campaign-table write.

    Returns a plain dict (asdict of the OracleResult plus member_id) because that
    is all that can be guaranteed picklable across a spawn boundary.
    """
    if not _W:  # spawn re-import: the initializer may not have run for this worker
        _worker_init(payload["campaign_json"])
    graph = StrategyGraph.from_dict(payload["graph"])
    result = _W["oracle"].evaluate(
        graph,
        window_start_ms=payload["window_start_ms"],
        window_end_ms=payload["window_end_ms"],
    )
    out = dataclasses.asdict(result)
    out["member_id"] = payload["member_id"]
    out["member_index"] = payload["member_index"]
    return out


# --------------------------------------------------------------------------- #
# Parent side
# --------------------------------------------------------------------------- #


def _worker_spec(campaign: population.Campaign, *, ohlcv_path: str, state_path: str,
                 wf_knobs: dict) -> str:
    """The JSON a worker needs to rebuild its own oracle. No connections in it."""
    return json.dumps(
        {
            "campaign_id": campaign.campaign_id,
            "symbols": list(campaign.symbols),
            "train_start_ms": campaign.train_start_ms,
            "train_end_ms": campaign.train_end_ms,
            "ohlcv_path": ohlcv_path,
            "state_path": state_path,
            **wf_knobs,
        },
        sort_keys=True,
    )


def _result_values(result_dict: dict, fitness) -> dict:
    """The population_members result columns for one evaluated member."""
    return {
        "fitness": None if fitness.score == -math.inf else fitness.score,
        "tier": fitness.tier,
        "excess_sharpe": fitness.excess_sharpe,
        "excess_ann_return": fitness.excess_ann_return,
        "sharpe": result_dict["sharpe"],
        "dsr": result_dict["dsr"],
        "ann_return_pct": result_dict["ann_return_pct"],
        "max_drawdown_pct": result_dict["max_drawdown_pct"],
        "n_trades": result_dict["n_trades"],
        "bench_sharpe": result_dict["bench_sharpe"],
        "bench_ann_return_pct": result_dict["bench_ann_return_pct"],
        "n_trials_used": result_dict["n_trials_used"],
        "gate_json": json.dumps(result_dict["gate"], sort_keys=True),
        "window_start_ms": result_dict["window_start_ms"],
        "window_end_ms": result_dict["window_end_ms"],
        "eval_seconds": result_dict["eval_seconds"],
        "error": result_dict["error"],
    }


def _run_tasks(payloads, *, workers: int, campaign_json: str, progress) -> dict:
    """Evaluate payloads, returning {member_index: result_dict}.

    workers == 1 runs IN-PROCESS: a single-worker pool would pay spawn cost for no
    parallelism, and running in-process is what lets the test suite exercise this
    function without spawning children (which under pytest re-imports the test
    module in every child and can deadlock under output capture).
    """
    out: dict[int, dict] = {}
    total = len(payloads)
    started = time.perf_counter()
    if workers <= 1:
        _worker_init(campaign_json)
        for i, payload in enumerate(payloads, 1):
            res = _evaluate_task(payload)
            out[res["member_index"]] = res
            _tick(progress, i, total, started)
        return out

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=_worker_init, initargs=(campaign_json,),
    ) as pool:
        futures = {pool.submit(_evaluate_task, p): p for p in payloads}
        done = 0
        for fut in as_completed(futures):
            payload = futures[fut]
            done += 1
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 — one bad candidate is not a bad campaign
                logger.warning("member %s raised: %s", payload["member_id"], exc)
                res = _error_result(payload, exc)
            out[res["member_index"]] = res
            _tick(progress, done, total, started)
    return out


def _tick(progress, done: int, total: int, started: float) -> None:
    """done/total with a rate and an ETA (mirrors bruteforce/runner.py's idiom)."""
    if progress is None:
        return
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("nan")
    if done == total or done % max(1, total // 10) == 0:
        progress(f"    {done}/{total} evaluated  {rate:.2f}/s  eta {eta:.0f}s")


def _error_result(payload: dict, exc: BaseException) -> dict:
    """A tier-D result for a task that raised.

    The ledger row was already written inside the worker before scoring (A6), so
    the trial stays charged: over-charging is the only safe error.
    """
    return {
        "member_id": payload["member_id"],
        "member_index": payload["member_index"],
        "graph_hash": "",
        "params_hash": "",
        "window_start_ms": payload["window_start_ms"],
        "window_end_ms": payload["window_end_ms"],
        "oos_start_ms": None, "oos_end_ms": None,
        "n_trials_used": 0, "n_trades": 0,
        "sharpe": None, "dsr": None, "ann_return_pct": None,
        "max_drawdown_pct": None, "bench_sharpe": None,
        "bench_ann_return_pct": None,
        "gate": {}, "passed": False, "eval_seconds": 0.0,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _score_generation(members, results):
    """[(member_index, Fitness, Member)] in MEMBER-INDEX order, then ranked.

    Built from members (a deterministic list) rather than from `results` (a dict
    filled in completion order), so nothing downstream can depend on which worker
    finished first.
    """
    scored = []
    for m in members:
        res = results.get(m.member_index)
        if res is None:  # pragma: no cover - only reachable on a lost future
            continue
        fitness = tournament.score_member(_as_result(res))
        scored.append((m.member_index, fitness, m))
    return scored


class _ResultView:
    """Attribute access over a result dict, so tournament sees one shape.

    tournament.score_member is written against OracleResult; rebuilding a full
    frozen OracleResult from a dict would be an extra place for a field rename to
    diverge, and this view has no fields of its own to drift.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


def _as_result(d: dict):
    return _ResultView(d)


def _campaign_id(seed: int, seed_graph_hash: str, *, now=None) -> str:
    """"<YYYYMMDD>-<6 hex>", stable for a given (date, seed, seed graph)."""
    stamp = time.strftime("%Y%m%d", time.gmtime(now if now is not None else time.time()))
    tag = population.derive_seed("campaign", seed, seed_graph_hash)
    return f"{stamp}-{tag & 0xFFFFFF:06x}"


def _evo_config_snapshot(**overrides) -> dict:
    """Every EVO_* value ACTUALLY used, so a report cannot quote an unused default."""
    snap = {
        name: getattr(config, name)
        for name in sorted(dir(config))
        if name.startswith("EVO_")
    }
    snap.update({k: v for k, v in overrides.items() if v is not None})
    return snap


def run_campaign(
    *,
    seed_graph=None,
    seed: int | None = None,
    symbols=None,
    population_size: int | None = None,
    generations: int | None = None,
    workers: int | None = None,
    window_days: int | None = None,
    train_start_ms: int | None = None,
    train_end_ms: int | None = None,
    resume_campaign_id: str | None = None,
    label: str | None = None,
    strategy_name: str | None = None,
    extend_generations: int | None = None,
    state_conn=None,
    state_path: str | None = None,
    ohlcv_path: str | None = None,
    wf_knobs: dict | None = None,
    dry_run: bool = False,
    progress=print,
) -> dict:
    """Run (or resume) one evolution campaign. Returns a summary dict.

    Args:
        seed_graph: The StrategyGraph generation 0 descends from. Required unless
            resuming (a resumed campaign carries its seed graph inline).
        seed: The ONE number a replay needs. Defaults to a time-derived value,
            which is then RECORDED, so even an unseeded run is replayable.
        symbols: Pooled symbols. Part of a campaign's identity, stored in
            campaigns.symbols_json.
        population_size / generations / workers / window_days: default to EVO_*.
        train_start_ms / train_end_ms: default to EVO_TRAIN_*. train_end_ms is
            REFUSED if later than EVO_TRAIN_END — a keyword argument must not be
            able to spend Phase 9's holdout.
        resume_campaign_id: Continue an existing campaign from its last COMPLETED
            generation. Accepts a campaign_id OR an operator label. The ledger is
            persistent, so n_trials continues from the true total across
            restarts, exactly as contract §4 requires.
        label: The operator's durable handle for a NEW campaign. campaign_id
            embeds its creation date, so it is not a name a human can retype
            tomorrow to continue the same search; the label is.
        strategy_name: The strategy this campaign evolves. A campaign is an
            evolution history OF one strategy (one strategy, many campaigns), and
            recording it is what lets a continuation refuse a mismatched graph.
        extend_generations: On a resume, guarantee at least this many FURTHER
            generations beyond the work already completed, raising the campaign's
            ceiling if needed. This is how "evolve it again" becomes "keep
            improving the population I already paid for" instead of a fresh
            search under a new seed — the parents, the seen-hash set and the
            ledger all carry over.
        dry_run: Breed and print generation 0 with ZERO oracle calls and ZERO
            trials charged.
        progress: Injected printer, so this module never calls print itself.

    Returns:
        A summary dict: campaign_id, per-generation rows, the audit round, the
        best member, trials charged, and the gate verdict.
    """
    t_campaign = time.perf_counter()
    own_conn = state_conn is None
    conn = statestore.connect(state_path) if own_conn else state_conn
    population.ensure_schema(conn)
    ohlcv_path = ohlcv_path or config.DB_PATH
    wf_knobs = dict(wf_knobs or {})

    registry.load_all()

    if resume_campaign_id:
        campaign = population.load_campaign(conn, resume_campaign_id)
        if campaign is None:
            # A label is what the UI and an operator actually hold; falling back
            # to it here means "continue my campaign" works with either handle.
            campaign = population.campaign_by_label(conn, resume_campaign_id)
        if campaign is None:
            raise ValueError(f"no campaign {resume_campaign_id!r} in state.db")
        seed_graph = mutate.graph_from_json(campaign.seed_graph_json)
        # Explicit `is None`, never `or -1`: generation 0 is FALSY, so a campaign
        # interrupted after its first generation would restart at 0 and pay for
        # every member of it a second time.
        last_done = population.last_completed_generation(conn, campaign.campaign_id)
        start_gen = 0 if last_done is None else last_done + 1
        if extend_generations:
            # Measured from work COMPLETED, not from the old ceiling: a campaign
            # that stopped early still owes the operator the full batch they
            # asked for, and one that finished gets exactly that many more.
            wanted = start_gen + int(extend_generations)
            if wanted > campaign.generations:
                population.set_generations(conn, campaign.campaign_id, wanted)
                campaign = population.load_campaign(conn, campaign.campaign_id)
        if start_gen >= campaign.generations:
            raise ValueError(
                f"campaign {campaign.campaign_id} has already completed all "
                f"{campaign.generations} of its generations. Pass "
                f"extend_generations to keep improving it, which raises the "
                f"ceiling and continues from the population it already evolved."
            )
        seen_hashes = population.seen_graph_hashes(conn, campaign.campaign_id)
        arith = window_arithmetic(
            window_days=json.loads(campaign.config_json).get(
                "EVO_WINDOW_DAYS", config.EVO_WINDOW_DAYS
            ),
            train_start_ms=campaign.train_start_ms,
            train_end_ms=campaign.train_end_ms,
        )
        resumed_scored = _reload_scored(conn, campaign, start_gen - 1)
        progress(
            f"resuming campaign {campaign.campaign_id} at generation {start_gen} "
            f"({len(seen_hashes)} graph hashes already bred, "
            f"{len(resumed_scored)} scored parents reloaded from generation "
            f"{start_gen - 1})"
        )
    else:
        if seed_graph is None:
            raise ValueError("run_campaign needs seed_graph unless resuming")
        arith_start, arith_end = training_span(train_start_ms, train_end_ms)
        ceiling = config.date_to_ms(config.EVO_TRAIN_END)
        if arith_end > ceiling:
            raise oracle.HoldoutViolation(
                f"train_end {arith_end} is later than config.EVO_TRAIN_END "
                f"({config.EVO_TRAIN_END} = {ceiling}). A caller must not be able "
                f"to spend Phase 9's holdout."
            )
        arith = window_arithmetic(
            window_days=window_days, train_start_ms=arith_start, train_end_ms=arith_end
        )
        if seed is None:
            seed = population.derive_seed("clock", int(time.time() * 1000))
        seed_hash = population.graph_hash(seed_graph)
        symbols = tuple(symbols or config.SYMBOLS)
        pop = int(population_size or config.EVO_POPULATION)
        gens = int(generations or config.EVO_GENERATIONS)
        audit_start, audit_end = _audit_window(arith)
        campaign = population.Campaign(
            campaign_id=_campaign_id(seed, seed_hash),
            seed=int(seed),
            seed_graph_hash=seed_hash,
            seed_graph_json=json.dumps(seed_graph.to_dict(), sort_keys=True),
            config_json=json.dumps(
                _evo_config_snapshot(
                    EVO_POPULATION=pop, EVO_GENERATIONS=gens,
                    EVO_WINDOW_DAYS=arith["window_days"],
                    EVO_WORKERS=workers,
                ),
                sort_keys=True, default=str,
            ),
            symbols=symbols,
            train_start_ms=arith["train_start_ms"],
            train_end_ms=arith["train_end_ms"],
            audit_start_ms=audit_start,
            audit_end_ms=audit_end,
            population=pop,
            generations=gens,
            started_ts=int(time.time() * 1000),
            status="running",
            label=label or "",
            strategy_name=strategy_name or "",
        )
        population.insert_campaign(conn, campaign)
        if campaign.label:
            # Gate runs made under this label BEFORE the campaign existed were
            # charged to the label, there being no campaign_id yet. Adopt them
            # so the campaign carries one honest cumulative trial count (§4)
            # rather than two half-histories.
            adopted = population.adopt_ledger_rows(
                conn, from_key=campaign.label, to_key=campaign.campaign_id
            )
            if adopted:
                progress(
                    f"  adopted {adopted} earlier trial(s) charged to label "
                    f"{campaign.label!r} — they count against this campaign's DSR"
                )
        start_gen = 0
        seen_hashes = set()
        resumed_scored = []

    workers = int(workers or config.EVO_WORKERS)
    if campaign.population < config.EVO_TOURNAMENT_K:
        raise ValueError(
            f"population {campaign.population} is below EVO_TOURNAMENT_K "
            f"({config.EVO_TOURNAMENT_K}); a tournament cannot sample a bracket"
        )
    campaign_json = _worker_spec(
        campaign, ohlcv_path=ohlcv_path,
        state_path=state_path or config.STATE_DB_PATH, wf_knobs=wf_knobs,
    )
    ledger = oracle.TrialLedger(conn, campaign.campaign_id)
    trials_at_start = ledger.count()

    progress(
        f"campaign {campaign.campaign_id}"
        + (f"  label {campaign.label}" if campaign.label else "")
        + (f"  strategy {campaign.strategy_name}" if campaign.strategy_name else "")
        + f"  seed {campaign.seed}  symbols {','.join(campaign.symbols)}"
    )
    progress(
        f"  train {_d(campaign.train_start_ms)} -> {_d(campaign.train_end_ms)} "
        f"({arith['span_days']} d)  window {arith['window_days']} d  "
        f"folds {arith['n_folds']}  window-OOS {arith['oos_days']} d  "
        f"jitter {arith['jitter_days']} d"
    )
    progress(
        f"  audit window (FIXED, declared now) {_d(campaign.audit_start_ms)} -> "
        f"{_d(campaign.audit_end_ms)}"
    )
    progress(
        f"  population {campaign.population} x generations {campaign.generations}  "
        f"workers {workers}  ledger at start {trials_at_start}"
    )

    stop = {"requested": False}

    def _on_sigint(signum, frame):  # pragma: no cover - interactive path
        stop["requested"] = True
        progress("\nSIGINT: finishing this generation, then writing status='aborted'")

    try:
        previous = signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:  # pragma: no cover - non-main thread
        previous = None

    gen_rows: list[dict] = []
    last_scored: list = list(resumed_scored)
    status = "done"
    try:
        for gen_index in range(start_gen, campaign.generations):
            gen_rng = population.member_rng(campaign.seed, gen_index, -1)
            win_start, win_end = _draw_window(gen_rng, arith)
            t_gen = time.perf_counter()
            gen = population.Generation(
                campaign_id=campaign.campaign_id, gen_index=gen_index,
                window_start_ms=win_start, window_end_ms=win_end,
                population_size=campaign.population,
                started_ts=int(time.time() * 1000),
            )
            population.insert_generation(conn, gen)

            if gen_index == 0 or not last_scored:
                members = mutate.seed_members(
                    campaign, seed_graph, seen_hashes=seen_hashes,
                    population_size=campaign.population,
                )
                if gen_index != 0:
                    # A resumed campaign whose previous generation is unavailable
                    # in memory: re-seed rather than invent a parent set.
                    members = [
                        dataclasses.replace(m, gen_index=gen_index,
                                            member_id=mutate.member_id_for(
                                                campaign.campaign_id, gen_index,
                                                m.member_index))
                        for m in members
                    ]
            else:
                elite_entries = tournament.elites(last_scored)
                if tournament.all_tier_d(last_scored):
                    logger.warning(
                        "generation %d: every member was tier D; re-breeding from "
                        "the previous elites with the diversity guard on",
                        gen_index,
                    )
                parents = tournament.select_parents(
                    last_scored, gen_rng,
                    n=campaign.population - len(elite_entries),
                )
                members = mutate.breed(
                    campaign, gen_index,
                    elites=[m for _, _, m in elite_entries],
                    parents=[m for _, _, m in parents],
                    seen_hashes=seen_hashes,
                )
            population.insert_members(conn, members)
            unique = tournament.unique_fraction([m.graph_hash for m in members])
            if tournament.diversity_guard_needed([m.graph_hash for m in members]):
                logger.warning(
                    "generation %d: unique graph fraction %.2f is below "
                    "EVO_MIN_UNIQUE_FRACTION %.2f — the population is collapsing "
                    "and every duplicate still costs a trial",
                    gen_index, unique, config.EVO_MIN_UNIQUE_FRACTION,
                )

            progress(
                f"gen {gen_index}  window {_d(win_start)}->{_d(win_end)}  "
                f"members {len(members)}  unique {unique:.2f}"
            )
            if dry_run:
                progress(
                    "  --dry-run: bred and recorded, ZERO oracle calls, ZERO "
                    f"trials charged (ledger still {ledger.count()})"
                )
                gen_rows.append(
                    {"gen_index": gen_index, "window_start_ms": win_start,
                     "window_end_ms": win_end, "n_evaluated": 0, "n_errors": 0,
                     "n_unique_graphs": len(set(m.graph_hash for m in members)),
                     "trials_cumulative": ledger.count(), "best_member_id": None,
                     "best_fitness": None, "db_retries": 0, "wall_seconds": 0.0,
                     "population_size": len(members)}
                )
                status = "dry-run"
                break

            payloads = [
                {
                    "member_id": m.member_id, "member_index": m.member_index,
                    "graph": json.loads(m.graph_json),
                    "window_start_ms": win_start, "window_end_ms": win_end,
                    "campaign_json": campaign_json,
                }
                for m in members
            ]
            results = _run_tasks(
                payloads, workers=workers, campaign_json=campaign_json,
                progress=progress,
            )

            retries = 0
            n_errors = 0
            scored = _score_generation(members, results)
            for idx, fitness, member in scored:
                res = results[idx]
                if res["error"]:
                    n_errors += 1
                retries += population.update_member_result(
                    conn, member.member_id, _result_values(res, fitness)
                )
            last_scored = scored
            order = tournament.ranked(scored)
            best_idx, best_fit, best_member = order[0] if order else (None, None, None)
            trials_cum = ledger.count()
            wall = time.perf_counter() - t_gen
            population.finish_generation(
                conn, campaign.campaign_id, gen_index,
                n_evaluated=len(results), n_errors=n_errors,
                n_unique_graphs=len(set(m.graph_hash for m in members)),
                trials_cumulative=trials_cum,
                best_member_id=best_member.member_id if best_member else None,
                best_fitness=(
                    None if best_fit is None or best_fit.score == -math.inf
                    else best_fit.score
                ),
                db_retries=retries, wall_seconds=wall,
            )
            gen_rows.append(
                {"gen_index": gen_index, "window_start_ms": win_start,
                 "window_end_ms": win_end, "n_evaluated": len(results),
                 "n_errors": n_errors,
                 "n_unique_graphs": len(set(m.graph_hash for m in members)),
                 "trials_cumulative": trials_cum,
                 "best_member_id": best_member.member_id if best_member else None,
                 "best_fitness": None if best_fit is None else best_fit.score,
                 "db_retries": retries, "wall_seconds": wall,
                 "population_size": len(members)}
            )
            if best_fit is not None:
                res = results[best_idx]
                progress(
                    f"  best {best_member.member_id}  tier {best_fit.tier}  "
                    f"exSharpe {_f(best_fit.excess_sharpe)}  "
                    f"gate {sum(1 for v in res['gate'].values() if v)}/"
                    f"{len(res['gate']) or 7}  dsr {_f(res['dsr'], 4)}  "
                    f"trades {res['n_trades']}  trials {trials_cum}"
                )
            if stop["requested"]:
                status = "aborted"
                break
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)

    audit = []
    verdict: dict = {}
    if status == "done":
        audit, verdict = _finalize(
            conn, campaign, seed_graph, workers=workers,
            campaign_json=campaign_json, ledger=ledger, progress=progress,
        )
    population.finish_campaign(conn, campaign.campaign_id, status=status)

    summary = {
        "campaign_id": campaign.campaign_id,
        "label": campaign.label,
        "strategy_name": campaign.strategy_name,
        "seed": campaign.seed,
        "symbols": list(campaign.symbols),
        "status": status,
        "train_start_ms": campaign.train_start_ms,
        "train_end_ms": campaign.train_end_ms,
        "audit_start_ms": campaign.audit_start_ms,
        "audit_end_ms": campaign.audit_end_ms,
        "arithmetic": arith,
        "generations": gen_rows,
        "audit": audit,
        "verdict": verdict,
        "trials_charged": ledger.count() - trials_at_start,
        "ledger_total": ledger.count(),
        "ledger_distinct": ledger.distinct_count(),
        "wall_seconds": time.perf_counter() - t_campaign,
    }
    if own_conn:
        conn.close()
    return summary


def _finalize(conn, campaign, seed_graph, *, workers, campaign_json, ledger, progress):
    """The AUDIT ROUND (A3): finalists AND the seed, on ONE fixed window.

    Fitness inside a generation is comparable (all members share a window), but
    windows differ BETWEEN generations, so "the best beat the seed" is only
    answerable by re-scoring finalists and the seed together on the window that
    was declared at campaign start. Each re-scoring charges a trial: an audit
    evaluation is an evaluation.
    """
    # One extra row is requested and the SEED's own genome filtered out: the seed
    # is member 0 of generation 0, so if it ranks in the top EVO_FINALISTS it would
    # otherwise be audited twice — two identical rows, one of them not marked
    # is_seed, and one extra trial charged for a comparison already being made.
    shortlist = [
        row
        for row in population.top_members(
            conn, campaign.campaign_id, limit=config.EVO_FINALISTS + 1
        )
        if row["graph_hash"] != campaign.seed_graph_hash
    ][: config.EVO_FINALISTS]
    graphs = [(row["member_id"], mutate.graph_from_json(row["graph_json"]))
              for row in shortlist]
    graphs.append(("seed", seed_graph))

    progress(
        f"AUDIT ROUND (fixed window {_d(campaign.audit_start_ms)}->"
        f"{_d(campaign.audit_end_ms)}): {len(shortlist)} finalist(s) + the seed"
    )

    members = []
    for i, (origin, graph) in enumerate(graphs):
        members.append(
            population.Member(
                member_id=f"{campaign.campaign_id}:audit:{i:04d}",
                campaign_id=campaign.campaign_id,
                gen_index=-1,
                member_index=i,
                graph_hash=population.graph_hash(graph),
                graph_json=json.dumps(graph.to_dict(), sort_keys=True,
                                      separators=(",", ":")),
                rng_seed=0,
                role="finalist",
                parent_member_id=None if origin == "seed" else origin,
                mutator="",
                mutation_json=json.dumps({"origin": origin}),
                graph=graph,
            )
        )
    population.insert_members(conn, members)

    payloads = [
        {
            "member_id": m.member_id, "member_index": m.member_index,
            "graph": json.loads(m.graph_json),
            "window_start_ms": campaign.audit_start_ms,
            "window_end_ms": campaign.audit_end_ms,
            "campaign_json": campaign_json,
        }
        for m in members
    ]
    results = _run_tasks(
        payloads, workers=workers, campaign_json=campaign_json, progress=progress
    )
    scored = _score_generation(members, results)
    for idx, fitness, member in scored:
        population.update_member_result(
            conn, member.member_id, _result_values(results[idx], fitness)
        )

    rows = []
    for idx, fitness, member in scored:
        res = results[idx]
        origin = json.loads(member.mutation_json).get("origin", "")
        rows.append(
            {
                "member_id": member.member_id, "origin": origin,
                "is_seed": origin == "seed", "tier": fitness.tier,
                "excess_sharpe": fitness.excess_sharpe,
                "excess_ann_return": fitness.excess_ann_return,
                "score": fitness.score, "sharpe": res["sharpe"],
                "dsr": res["dsr"], "n_trades": res["n_trades"],
                "ann_return_pct": res["ann_return_pct"],
                "max_drawdown_pct": res["max_drawdown_pct"],
                "bench_sharpe": res["bench_sharpe"],
                "bench_ann_return_pct": res["bench_ann_return_pct"],
                "n_trials_used": res["n_trials_used"],
                "gate": res["gate"], "passed": res["passed"],
                "graph_hash": member.graph_hash,
            }
        )
    order = tournament.ranked(scored)
    seed_row = next((r for r in rows if r["is_seed"]), None)
    best_entry = order[0] if order else None
    best_row = None
    if best_entry is not None:
        best_id = best_entry[2].member_id
        best_row = next(r for r in rows if r["member_id"] == best_id)

    beats_seed = False
    if best_row is not None and seed_row is not None:
        beats_seed = tournament.rank_key(
            _fit_of(best_row), 0
        ) > tournament.rank_key(_fit_of(seed_row), 0) and not best_row["is_seed"]

    for r in rows:
        progress(
            f"    {'SEED ' if r['is_seed'] else '     '}{r['member_id']}  "
            f"tier {r['tier']}  exSharpe {_f(r['excess_sharpe'])}  "
            f"sharpe {_f(r['sharpe'])}  dsr {_f(r['dsr'], 4)}  "
            f"trades {r['n_trades']}"
            f"{'  <- best' if best_row is not None and r['member_id'] == best_row['member_id'] else ''}"
        )
    verdict = {
        "best": best_row,
        "seed": seed_row,
        "beats_seed": bool(beats_seed),
        "gate_passed": bool(best_row["passed"]) if best_row else False,
        "failed_conditions": (
            [k for k, v in best_row["gate"].items() if not v] if best_row else []
        ),
    }
    return rows, verdict


def _reload_scored(conn, campaign, gen_index: int) -> list:
    """Rebuild [(member_index, Fitness, Member)] for one completed generation.

    So `--resume` continues the SEARCH and not merely the bookkeeping. Members
    whose evaluation errored or was never scored are skipped: they carry no
    genome worth breeding from, and including them would let a tier-D row become
    a parent purely because it survived a restart.
    """
    if gen_index < 0:
        return []
    out = []
    for row in population.generation_members(conn, campaign.campaign_id, gen_index):
        if row["fitness"] is None and row["tier"] in (None, "", population.TIER_UNKNOWN):
            continue
        try:
            graph = mutate.graph_from_json(row["graph_json"])
        except Exception as exc:  # noqa: BLE001 — a corrupt row must not kill a resume
            logger.warning("resume: member %s has an unusable graph (%s)",
                           row["member_id"], exc)
            continue
        member = population.Member(
            member_id=row["member_id"], campaign_id=row["campaign_id"],
            gen_index=row["gen_index"], member_index=row["member_index"],
            graph_hash=row["graph_hash"], graph_json=row["graph_json"],
            rng_seed=row["rng_seed"], role=row["role"],
            parent_member_id=row["parent_member_id"], mutator=row["mutator"] or "",
            mutation_json=row["mutation_json"] or "{}", graph=graph,
        )
        result = dict(row)
        result["gate"] = json.loads(row["gate_json"] or "{}")
        result["error"] = row["error"] or ""
        result["n_trades"] = row["n_trades"] or 0
        fitness = tournament.score_member(_as_result(result))
        out.append((row["member_index"], fitness, member))
    return out


def _fit_of(row) -> tournament.Fitness:
    """Rebuild a Fitness from an audit row so ranking uses ONE comparison rule."""
    score = row["score"]
    return tournament.Fitness(
        tier=row["tier"],
        score=-math.inf if score is None else score,
        excess_sharpe=row["excess_sharpe"],
        excess_ann_return=row["excess_ann_return"],
        n_trades=int(row["n_trades"] or 0),
    )


# --------------------------------------------------------------------------- #
# Calibration — measure the per-candidate cost, then DERIVE the size
# --------------------------------------------------------------------------- #


def calibrate(*, seed_graph, symbols=None, repeats: int = 3, workers: int | None = None,
              budget_hours: float | None = None, window_days: int | None = None,
              state_conn=None, state_path: str | None = None,
              ohlcv_path: str | None = None, wf_knobs: dict | None = None,
              progress=print) -> dict:
    """Measure cold and warm per-candidate cost, then print legal (pop, gens) pairs.

    Nothing in this phase's config asserts a population size; this function
    produces it. Calibration runs SINGLE-PROCESS and in-process on purpose, so
    cache effects are attributable (trap 2): the first evaluation in a fresh
    process pays frame decode, regime labels, ATR and channel warm-up, and the
    rest are warm.

    Calibration evaluations ARE charged to a ledger — under a throwaway campaign
    id, but charged, because they are evaluations. Including them is the safe
    direction (A6).
    """
    own_conn = state_conn is None
    conn = statestore.connect(state_path) if own_conn else state_conn
    population.ensure_schema(conn)
    registry.load_all()
    symbols = tuple(symbols or config.SYMBOLS)
    workers = int(workers or config.EVO_WORKERS)
    budget_hours = float(budget_hours or config.EVO_BUDGET_HOURS)
    arith = window_arithmetic(window_days=window_days)
    win_start, win_end = _audit_window(arith)

    campaign_id = f"calibrate-{int(time.time())}"
    ledger = oracle.TrialLedger(conn, campaign_id)
    ohlcv = _connect_ohlcv_readonly(ohlcv_path or config.DB_PATH)
    knobs = dict(wf_knobs or {})
    orc = oracle.GateOracle(
        ledger, ohlcv_conn=ohlcv, symbols=symbols,
        train_start_ms=arith["train_start_ms"], train_end_ms=arith["train_end_ms"],
        train_days=knobs.get("train_days"), test_days=knobs.get("test_days"),
        oos_days=knobs.get("oos_days"), min_trades=knobs.get("min_trades"),
    )

    timings = []
    for i in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        res = orc.evaluate(seed_graph, window_start_ms=win_start, window_end_ms=win_end)
        timings.append(time.perf_counter() - t0)
        progress(f"  eval {i} (SAME graph): {timings[-1]:.2f} s  "
                 f"trades {res.n_trades}  sharpe {_f(res.sharpe)}")

    # THE NUMBER THAT ACTUALLY SIZES A CAMPAIGN. Re-evaluating one graph measures
    # the framework's candidate memo (framework.context keys detector candidates
    # by branch_hash), not a campaign: every member of a real generation is a
    # DIFFERENT graph and misses that memo. Reporting only the same-graph "warm"
    # figure would overstate throughput by several times and size a population
    # that cannot finish. Measured separately, on real mutated children.
    distinct = []
    seen: set[str] = set()
    for i in range(max(1, int(repeats))):
        child_rng = population.member_rng(0, 0, i)
        try:
            child, _key, _diff, h, _ = mutate._breed_one(
                seed_graph, child_rng, seen_hashes=seen
            )
        except Exception as exc:  # noqa: BLE001 — calibration must not abort a run
            logger.warning("calibrate: could not breed a distinct child (%s)", exc)
            break
        seen.add(h)
        t0 = time.perf_counter()
        orc.evaluate(child, window_start_ms=win_start, window_end_ms=win_end)
        distinct.append(time.perf_counter() - t0)
        progress(f"  eval {i} (DISTINCT graph): {distinct[-1]:.2f} s")

    cold = timings[0]
    warm = statistics.median(timings[1:]) if len(timings) > 1 else timings[0]
    warm_distinct = statistics.median(distinct) if distinct else cold
    backtests_per_eval = (arith["n_folds"] * 2 + 1) * len(symbols)
    # Sized on warm_distinct, never on warm: the honest, pessimistic direction.
    capacity = (
        int((workers * budget_hours * 3600) // warm_distinct) if warm_distinct > 0 else 0
    )

    pairs = []
    floor_pop = max(8 * config.EVO_TOURNAMENT_K, config.EVO_TOURNAMENT_K)
    for pop in (24, 32, 48, 64, 96, 128, 192, 288):
        if pop < floor_pop:
            continue
        gens = capacity // pop if pop else 0
        if gens >= 1:
            pairs.append((pop, int(gens)))

    progress(
        f"  cold {cold:.2f} s  warm-SAME-graph {warm:.2f} s  "
        f"warm-DISTINCT-graph {warm_distinct:.2f} s  <- sizing uses the DISTINCT "
        f"figure; the same-graph one only measures the candidate memo"
    )
    progress(
        f"  workers {workers}  budget {budget_hours:.1f} h -> capacity "
        f"{capacity} evaluations (= workers x budget / warm-distinct). Capacity is "
        f"a WALL-CLOCK bound, NOT a recommendation: every evaluation is also a DSR "
        f"trial (contract §4), so spending it all buys ~sqrt(log N) more expected "
        f"max Sharpe to beat and nothing else."
    )
    progress(
        f"  cost per candidate: ({arith['n_folds']} folds x 2 + 1 OOS) x "
        f"{len(symbols)} symbols = {backtests_per_eval} graph backtests, "
        f"one degenerate grid combo (A1)"
    )
    if pairs:
        progress(
            "  suggested: "
            + "  |  ".join(f"population {p} x generations {g}" for p, g in pairs[:4])
        )
    thousand_needed = (
        workers * budget_hours * 3600 / (1000 * max(1, config.EVO_GENERATIONS))
    )
    progress(
        f"  pivot-guide reconciliation: '~1000 simultaneous models' per generation "
        f"needs warm-distinct <= {thousand_needed:.1f} s/candidate at {workers} "
        f"workers / {budget_hours:.1f} h / {config.EVO_GENERATIONS} generations. "
        f"Measured {warm_distinct:.2f} s -> "
        f"{'REACHABLE on wall clock' if warm_distinct <= thousand_needed else 'NOT REACHABLE'}"
        f". Reachable on wall clock is NOT the same as advisable: 1000 x "
        f"{config.EVO_GENERATIONS} evaluations is also 8000 DSR trials."
    )
    progress(
        f"  calibration charged {ledger.count()} trial(s) under throwaway campaign "
        f"{campaign_id}"
    )

    out = {
        "cold_seconds": cold, "warm_seconds": warm,
        "warm_distinct_seconds": warm_distinct, "distinct_timings": distinct,
        "timings": timings,
        "workers": workers, "budget_hours": budget_hours, "capacity": capacity,
        "backtests_per_eval": backtests_per_eval, "pairs": pairs,
        "arithmetic": arith, "campaign_id": campaign_id,
        "trials_charged": ledger.count(),
        "thousand_models_threshold_s": thousand_needed,
    }
    ohlcv.close()
    if own_conn:
        conn.close()
    return out


# --------------------------------------------------------------------------- #
# Small formatting helpers (this module never calls print directly)
# --------------------------------------------------------------------------- #


def _d(ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


def _f(value, places: int = 2) -> str:
    """Formats None as 'n/a' — never as 0.00, which would read as a real value."""
    if value is None:
        return "n/a"
    return f"{value:+.{places}f}"


def worker_count_default() -> int:
    """config.EVO_WORKERS, exposed so the CLI's help text cannot drift from it."""
    return max(1, (os.cpu_count() or 4) - 2)
