"""
The campaign loop: determinism, pickle safety, persistence, ordering, the audit
round, calibration and the window arithmetic (v0.3.0 Phase 6).

NO REAL PROCESSES ARE SPAWNED HERE. Under `spawn`, pytest re-imports the test
module in every child, which is slow and can deadlock under output capture. So
_worker_init / _evaluate_task are exercised directly in-process (which is also the
production path at workers=1), and pool wiring is asserted against a
monkeypatched ProcessPoolExecutor.

Tiny walk-forward knobs (train_days=10, test_days=5, oos_days=5) mirror
tests/test_backtest.py so a synthetic campaign runs in milliseconds.
"""

import dataclasses
import json
import pickle
import random
import sqlite3

import pytest

from trading_bot import config
from trading_bot.backtest import engine, trials
from trading_bot.backtest.walkforward import GATE_CONDITIONS
from trading_bot.data import statestore
from trading_bot.evolution import mutate, oracle, population, runner, tournament
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry

START = 1_700_000_000_000
DAY_MS = 86_400_000
SEED_GRAPH_PATH = "data/strategies/thin-slice.strategy.json"

# Tier constants derived from config, never hardcoded (contract §8).
TINY_KNOBS = {"train_days": 10, "test_days": 5, "oos_days": 5, "min_trades": 1}


@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    """Mirrors tests/test_backtest.py: test isolation must not DEPEND on a
    fingerprint argument being right."""
    engine.clear_caches()
    yield
    engine.clear_caches()


@pytest.fixture(autouse=True)
def _registry_loaded():
    registry.load_all()
    yield


@pytest.fixture(autouse=True)
def _clean_worker_state():
    runner._W.clear()
    yield
    runner._W.clear()


@pytest.fixture
def state_conn(tmp_path):
    conn = statestore.connect(str(tmp_path / "s.db"))
    population.ensure_schema(conn)
    trials.ensure_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def seed_graph():
    return fgraph.load(SEED_GRAPH_PATH)


@pytest.fixture
def stub_oracle(monkeypatch):
    """Replace the GATE with a deterministic, graph-hash-derived score.

    The LEDGER IS STILL REAL and still charged: the point of these tests is the
    loop's bookkeeping, and stubbing the ledger too would make ledger-parity
    assertions vacuous.
    """
    calls: list = []
    real_evaluate = oracle.GateOracle.evaluate

    def _fake(self, graph, *, window_start_ms, window_end_ms):
        self._check_span(window_start_ms, window_end_ms)
        h = fgraph.graph_hash(graph)
        n_trials = self._ledger.charge(
            graph_hash=h, params_hash="stub", start_ms=window_start_ms,
            end_ms=window_end_ms,
        )
        calls.append((h, window_start_ms, window_end_ms))
        # Deterministic pseudo-metric from the hash, so ranking is stable and
        # reproducible without touching the database.
        score = (int(h[:8], 16) % 1000) / 1000.0
        return oracle.OracleResult(
            graph_hash=h, params_hash="stub",
            window_start_ms=window_start_ms, window_end_ms=window_end_ms,
            oos_start_ms=window_start_ms, oos_end_ms=window_end_ms,
            n_trials_used=n_trials, n_trades=40,
            sharpe=1.0 + score, dsr=0.01, ann_return_pct=0.3 + score,
            max_drawdown_pct=0.1, bench_sharpe=0.7, bench_ann_return_pct=0.29,
            gate=dict.fromkeys(GATE_CONDITIONS, True), passed=True,
            eval_seconds=0.001, error="",
        )

    monkeypatch.setattr(oracle.GateOracle, "evaluate", _fake)
    return {"calls": calls, "real": real_evaluate}


def _run(tmp_path, seed_graph, **kwargs):
    args = dict(
        seed_graph=seed_graph, seed=42, symbols=("BTCUSDT",),
        population_size=6, generations=3, workers=1,
        state_path=str(tmp_path / "camp.db"), progress=lambda *_a, **_k: None,
        wf_knobs=TINY_KNOBS,
    )
    args.update(kwargs)
    return runner.run_campaign(**args)


class TestCampaignIdentity:
    """A campaign is a NAMED, strategy-bound history that keeps improving.

    campaign_id embeds its creation date, so it cannot be the handle an operator
    retypes tomorrow; the label is. These pin that the label round-trips, that
    continuing extends the SAME campaign rather than seeding a new one, and that
    the ledger stays one cumulative count.
    """

    def test_label_and_strategy_round_trip_and_resolve(self, tmp_path, seed_graph,
                                                       stub_oracle):
        path = str(tmp_path / "lbl.db")
        summary = _run(tmp_path, seed_graph, state_path=path, population_size=4,
                       generations=1, label="my-campaign", strategy_name="thin-slice")
        conn = statestore.connect(path)
        by_label = population.campaign_by_label(conn, "my-campaign")
        assert by_label is not None
        assert by_label.campaign_id == summary["campaign_id"]
        assert by_label.strategy_name == "thin-slice"
        # Resolvable by either handle, since the CLI and UI carry different ones.
        assert population.load_campaign(conn, summary["campaign_id"]).label == "my-campaign"
        conn.close()

    def test_resuming_by_label_extends_instead_of_restarting(
        self, tmp_path, seed_graph, stub_oracle
    ):
        path = str(tmp_path / "ext.db")
        first = _run(tmp_path, seed_graph, state_path=path, population_size=4,
                     generations=2, label="keep-going", strategy_name="thin-slice")
        conn = statestore.connect(path)
        trials_after_first = population.list_campaigns(conn)[0]["trials"]

        second = _run(tmp_path, seed_graph, state_path=path, population_size=4,
                      resume_campaign_id="keep-going", extend_generations=2)
        # SAME campaign, not a new one under a fresh seed.
        assert second["campaign_id"] == first["campaign_id"]
        rows = population.list_campaigns(conn)
        assert len(rows) == 1, "continuing must not create a second campaign"
        assert rows[0]["generations_done"] == 4
        # The generations it bred on from are the ones already paid for, so the
        # cumulative ledger only ever grows.
        assert rows[0]["trials"] > trials_after_first
        gen_indexes = [
            r["gen_index"] for r in population.generation_rows(conn, first["campaign_id"])
        ]
        assert gen_indexes == [0, 1, 2, 3]
        conn.close()

    def test_a_finished_campaign_refuses_to_resume_without_extend(
        self, tmp_path, seed_graph, stub_oracle
    ):
        """Silently doing nothing is the failure mode this replaces: the loop is
        range(start_gen, generations), which is EMPTY once a campaign finishes."""
        path = str(tmp_path / "done.db")
        _run(tmp_path, seed_graph, state_path=path, population_size=4,
             generations=1, label="finished", strategy_name="thin-slice")
        with pytest.raises(ValueError, match="already completed"):
            _run(tmp_path, seed_graph, state_path=path,
                 resume_campaign_id="finished")

    def test_list_campaigns_filters_by_strategy_and_counts_finished_generations(
        self, tmp_path, seed_graph, stub_oracle
    ):
        path = str(tmp_path / "list.db")
        _run(tmp_path, seed_graph, state_path=path, population_size=4, generations=2,
             label="alpha", strategy_name="strat-a")
        _run(tmp_path, seed_graph, state_path=path, population_size=4, generations=1,
             seed=99, label="beta", strategy_name="strat-b")
        conn = statestore.connect(path)
        assert {c["label"] for c in population.list_campaigns(conn)} == {"alpha", "beta"}
        only_a = population.list_campaigns(conn, strategy_name="strat-a")
        assert [c["label"] for c in only_a] == ["alpha"]
        assert only_a[0]["generations_done"] == 2
        conn.close()

    def test_gate_trials_charged_to_the_label_are_adopted_by_the_campaign(
        self, tmp_path, seed_graph, stub_oracle
    ):
        """A gate run before the first evolve had no campaign_id to charge, so it
        used the label. Those evaluations still happened and contract §4 deflates
        by the CUMULATIVE count — they must not be left behind."""
        path = str(tmp_path / "adopt.db")
        conn = statestore.connect(path)
        population.ensure_schema(conn)
        trials.ensure_schema(conn)
        early = trials.TrialLedger(conn, "adopt-me")
        for i in range(3):
            early.record(graph_hash=f"h{i}", params_hash="p", start_ms=1, end_ms=2)
        conn.commit()
        conn.close()

        summary = _run(tmp_path, seed_graph, state_path=path, population_size=4,
                       generations=1, label="adopt-me", strategy_name="thin-slice")
        conn = statestore.connect(path)
        left_behind = conn.execute(
            "SELECT COUNT(*) FROM trial_ledger WHERE campaign=?", ("adopt-me",)
        ).fetchone()[0]
        assert left_behind == 0, "label-keyed rows should have been adopted"
        on_campaign = conn.execute(
            "SELECT COUNT(*) FROM trial_ledger WHERE campaign=?",
            (summary["campaign_id"],),
        ).fetchone()[0]
        assert on_campaign >= 3 + 4
        conn.close()

    def test_migration_adds_columns_to_a_pre_label_database(self, tmp_path):
        """ensure_schema's CREATE TABLE IF NOT EXISTS is a no-op on an existing
        table, so a state.db written before these columns needs the ALTER."""
        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE campaigns (campaign_id TEXT PRIMARY KEY, seed INTEGER "
            "NOT NULL, seed_graph_hash TEXT NOT NULL, seed_graph_json TEXT NOT "
            "NULL, config_json TEXT NOT NULL, symbols_json TEXT NOT NULL, "
            "train_start_ms INTEGER NOT NULL, train_end_ms INTEGER NOT NULL, "
            "audit_start_ms INTEGER NOT NULL, audit_end_ms INTEGER NOT NULL, "
            "population INTEGER NOT NULL, generations INTEGER NOT NULL, "
            "started_ts INTEGER NOT NULL, finished_ts INTEGER, status TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO campaigns VALUES ('20260101-aaaaaa',1,'h','{}','{}',"
            "'[\"BTCUSDT\"]',1,2,3,4,4,2,100,NULL,'done')"
        )
        conn.commit()
        conn.close()

        conn = statestore.connect(path)
        population.ensure_schema(conn)
        loaded = population.load_campaign(conn, "20260101-aaaaaa")
        assert loaded is not None
        assert loaded.label == "" and loaded.strategy_name == ""
        conn.close()


class TestWindowArithmetic:
    """Derived from config, and pinned rather than trusted."""

    def test_the_training_span_is_derived_from_the_frozen_dates(self):
        start, end = runner.training_span()
        assert start == config.date_to_ms(config.EVO_TRAIN_START)
        assert end == config.date_to_ms(config.EVO_TRAIN_END)

    def test_the_measured_arithmetic(self):
        """MEASURED 2026-07-27 against config's frozen dates. The plan predicted
        913 days / 373 days of jitter; the real figures are 914 / 374, because the
        last CLOSED 1d bar opens 2026-07-26 (2026-07-27's is still forming)."""
        a = runner.window_arithmetic()
        assert a["span_days"] == 914
        assert a["window_days"] == config.EVO_WINDOW_DAYS == 540
        assert a["min_window_days"] == 330
        assert a["n_folds"] == 4
        assert a["oos_days"] == config.WF_OOS_DAYS == 90
        assert a["jitter_days"] == 374

    def test_the_minimum_window_is_the_walk_forward_sum(self):
        a = runner.window_arithmetic()
        assert a["min_window_days"] == (
            config.WF_TRAIN_DAYS + config.WF_TEST_DAYS + config.WF_OOS_DAYS
        )

    def test_a_window_below_the_minimum_is_refused_up_front(self):
        """walk_forward_pooled would raise for EVERY candidate; refusing here says
        so once instead of N times."""
        with pytest.raises(ValueError, match="below the walk-forward minimum"):
            runner.window_arithmetic(window_days=329)

    def test_an_empty_training_span_is_refused(self):
        with pytest.raises(ValueError, match="training span is empty"):
            runner.training_span(START, START)

    def test_the_audit_window_is_the_latest_legal_one(self):
        a = runner.window_arithmetic()
        s, e = runner._audit_window(a)
        assert e == a["train_end_ms"]
        assert (e - s) == a["window_days"] * DAY_MS

    def test_jitter_never_leaves_the_training_span(self):
        a = runner.window_arithmetic()
        for seed in range(500):
            s, e = runner._draw_window(random.Random(seed), a)
            assert a["train_start_ms"] <= s < e <= a["train_end_ms"]
            assert (s - a["train_start_ms"]) % DAY_MS == 0, "snap to a 1d bar open"

    def test_jitter_off_pins_the_window(self, monkeypatch):
        monkeypatch.setattr(config, "EVO_WINDOW_JITTER", False)
        a = runner.window_arithmetic()
        windows = {runner._draw_window(random.Random(s), a) for s in range(20)}
        assert len(windows) == 1
        assert windows.pop() == runner._audit_window(a)


class TestWorkerInit:
    def test_it_builds_an_oracle_and_is_idempotent(self, tmp_path):
        spec = json.dumps(
            {
                "campaign_id": "c1", "symbols": ["BTCUSDT"],
                "train_start_ms": START, "train_end_ms": START + 900 * DAY_MS,
                "ohlcv_path": config.DB_PATH,
                "state_path": str(tmp_path / "w.db"),
                **TINY_KNOBS,
            },
            sort_keys=True,
        )
        runner._worker_init(spec)
        first = runner._W["oracle"]
        runner._worker_init(spec)
        assert runner._W["oracle"] is first, "the initializer must be idempotent"
        assert isinstance(runner._W["ledger"], oracle.TrialLedger)

    def test_the_ohlcv_connection_is_physically_read_only(self, tmp_path):
        """A worker that CANNOT write ohlcv.db cannot corrupt 117 MB of
        irreplaceable price history, whatever a future edit does (trap 1)."""
        conn = runner._connect_ohlcv_readonly(config.DB_PATH)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE should_not_exist (x INTEGER)")
        conn.close()

    def test_a_failed_readonly_open_warns_rather_than_failing_silently(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging

        real_connect = sqlite3.connect

        def _fail(*args, **kwargs):
            if kwargs.get("uri"):
                raise sqlite3.OperationalError("unable to open database file")
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", _fail)
        with caplog.at_level(logging.WARNING, logger="trading_bot"):
            conn = runner._connect_ohlcv_readonly(str(tmp_path / "x.db"))
        assert "write guarantee is LOST" in caplog.text
        conn.close()


class TestPickleSafety:
    """Trap 3: under spawn, every payload must be picklable."""

    def test_task_payloads_round_trip(self, seed_graph):
        payload = {
            "member_id": "c:000:0000", "member_index": 0,
            "graph": seed_graph.to_dict(),
            "window_start_ms": START, "window_end_ms": START + DAY_MS,
            "campaign_json": "{}",
        }
        back = pickle.loads(pickle.dumps(payload))
        assert fgraph.graph_hash(
            fgraph.StrategyGraph.from_dict(back["graph"])
        ) == fgraph.graph_hash(seed_graph)

    def test_no_payload_carries_a_connection_frame_graph_or_rng(
        self, tmp_path, seed_graph, stub_oracle
    ):
        """Asserted on the REAL payloads a campaign builds, not on a hand-written
        example, so a future field addition is covered."""
        captured: list = []
        real = runner._run_tasks

        def _spy(payloads, **kwargs):
            captured.extend(payloads)
            return real(payloads, **kwargs)

        import pandas as pd

        runner._run_tasks = _spy
        try:
            _run(tmp_path, seed_graph, generations=1, population_size=4)
        finally:
            runner._run_tasks = real
        assert captured
        for payload in captured:
            for value in payload.values():
                assert not isinstance(
                    value,
                    (sqlite3.Connection, pd.DataFrame, random.Random,
                     fgraph.StrategyGraph),
                ), type(value)
            pickle.dumps(payload)  # would raise on anything unpicklable

    def test_oracle_results_round_trip_as_plain_dicts(self):
        res = oracle.OracleResult(
            graph_hash="h", params_hash="p", window_start_ms=1, window_end_ms=2,
            oos_start_ms=1, oos_end_ms=2, n_trials_used=3, n_trades=4,
            sharpe=1.0, dsr=0.1, ann_return_pct=0.2, max_drawdown_pct=0.3,
            bench_sharpe=0.4, bench_ann_return_pct=0.5,
            gate=dict.fromkeys(GATE_CONDITIONS, True), passed=True,
            eval_seconds=0.1,
        )
        d = dataclasses.asdict(res)
        assert json.loads(json.dumps(d)) == d


class TestDeterminism:
    def test_the_same_seed_reproduces_the_whole_campaign(
        self, tmp_path, seed_graph, stub_oracle
    ):
        a = _run(tmp_path / "a", seed_graph)
        b = _run(tmp_path / "b", seed_graph)
        assert [g["window_start_ms"] for g in a["generations"]] == [
            g["window_start_ms"] for g in b["generations"]
        ]
        assert [g["best_fitness"] for g in a["generations"]] == [
            g["best_fitness"] for g in b["generations"]
        ]
        assert a["verdict"]["best"]["graph_hash"] == b["verdict"]["best"]["graph_hash"]

    def test_a_different_seed_gives_a_different_campaign(
        self, tmp_path, seed_graph, stub_oracle
    ):
        a = _run(tmp_path / "a", seed_graph, seed=1)
        b = _run(tmp_path / "b", seed_graph, seed=2)
        assert a["campaign_id"] != b["campaign_id"]

    def test_worker_count_does_not_change_the_population(
        self, tmp_path, seed_graph, stub_oracle, monkeypatch
    ):
        """The whole point of keeping all randomness in the parent: worker count,
        completion order and machine speed cannot change the population."""
        one = _run(tmp_path / "one", seed_graph, workers=1)
        # workers=4 without spawning: run the pool branch's payloads serially.
        real = runner._run_tasks
        seen_workers: list[int] = []

        def _serial(payloads, *, workers, campaign_json, progress):
            seen_workers.append(workers)
            return real(payloads, workers=1, campaign_json=campaign_json,
                        progress=progress)

        monkeypatch.setattr(runner, "_run_tasks", _serial)
        four = _run(tmp_path / "four", seed_graph, workers=4)
        assert seen_workers and set(seen_workers) == {4}
        assert [g["best_fitness"] for g in one["generations"]] == [
            g["best_fitness"] for g in four["generations"]
        ]


class TestOrdering:
    def test_results_are_merged_in_member_index_order(self, seed_graph):
        """as_completed yields by completion TIME; ranking on it would make
        selection machine-speed-dependent."""
        c = population.Campaign(
            campaign_id="c1", seed=1, seed_graph_hash="h", seed_graph_json="{}",
            config_json="{}", symbols=("BTCUSDT",), train_start_ms=1,
            train_end_ms=2, audit_start_ms=1, audit_end_ms=2, population=4,
            generations=1, started_ts=0,
        )
        members = mutate.seed_members(c, seed_graph, seen_hashes=set(),
                                     population_size=4)
        results = {
            m.member_index: {
                "sharpe": 1.0, "bench_sharpe": 0.7, "ann_return_pct": 0.5,
                "bench_ann_return_pct": 0.29, "n_trades": 40, "dsr": 0.01,
                "gate": dict.fromkeys(GATE_CONDITIONS, True), "error": "",
            }
            for m in members
        }
        # Reverse the dict's insertion order: the merge must not notice.
        reversed_results = dict(reversed(list(results.items())))
        assert [i for i, _f, _m in runner._score_generation(members, results)] == [
            i for i, _f, _m in runner._score_generation(members, reversed_results)
        ] == [0, 1, 2, 3]

    def test_a_task_that_raises_becomes_a_recorded_tier_d(self):
        payload = {"member_id": "m", "member_index": 3,
                   "window_start_ms": 1, "window_end_ms": 2}
        res = runner._error_result(payload, RuntimeError("boom"))
        fitness = tournament.score_member(runner._as_result(res))
        assert fitness.tier == "D"
        assert "RuntimeError: boom" in res["error"]


class TestPersistence:
    def test_ensure_schema_is_idempotent(self, tmp_path):
        conn = statestore.connect(str(tmp_path / "i.db"))
        population.ensure_schema(conn)
        population.ensure_schema(conn)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"campaigns", "generations", "population_members"} <= tables
        conn.close()

    def test_a_campaign_writes_one_row_per_generation_and_member(
        self, tmp_path, seed_graph, stub_oracle
    ):
        path = str(tmp_path / "p.db")
        summary = _run(tmp_path, seed_graph, state_path=path,
                       population_size=4, generations=2)
        conn = statestore.connect(path)
        cid = summary["campaign_id"]
        gens = conn.execute(
            "SELECT COUNT(*) FROM generations WHERE campaign_id=?", (cid,)
        ).fetchone()[0]
        members = conn.execute(
            "SELECT COUNT(*) FROM population_members WHERE campaign_id=? "
            "AND role != 'finalist'", (cid,)
        ).fetchone()[0]
        assert gens == 2
        assert members == 8
        conn.close()

    def test_trials_cumulative_is_non_decreasing_and_matches_the_ledger(
        self, tmp_path, seed_graph, stub_oracle
    ):
        path = str(tmp_path / "t.db")
        summary = _run(tmp_path, seed_graph, state_path=path,
                       population_size=4, generations=3)
        rows = [g["trials_cumulative"] for g in summary["generations"]]
        assert rows == sorted(rows)
        assert rows == [4, 8, 12]
        # The last generation's cumulative count plus the audit round equals the
        # campaign's ledger total.
        assert summary["ledger_total"] == rows[-1] + len(summary["audit"])

    def test_every_ledger_row_stays_inside_the_training_ceiling(
        self, tmp_path, seed_graph, stub_oracle
    ):
        """Contract §4.4, asserted the same way the phase report proves it."""
        path = str(tmp_path / "h.db")
        summary = _run(tmp_path, seed_graph, state_path=path)
        conn = statestore.connect(path)
        over = conn.execute(
            "SELECT COUNT(*) FROM trial_ledger WHERE campaign=? AND end_ms > ?",
            (summary["campaign_id"], summary["train_end_ms"]),
        ).fetchone()[0]
        under = conn.execute(
            "SELECT COUNT(*) FROM trial_ledger WHERE campaign=? AND start_ms < ?",
            (summary["campaign_id"], summary["train_start_ms"]),
        ).fetchone()[0]
        assert over == 0
        assert under == 0
        conn.close()

    def test_the_campaign_row_records_the_config_actually_used(
        self, tmp_path, seed_graph, stub_oracle
    ):
        path = str(tmp_path / "c.db")
        summary = _run(tmp_path, seed_graph, state_path=path, population_size=5,
                       generations=2)
        conn = statestore.connect(path)
        camp = population.load_campaign(conn, summary["campaign_id"])
        cfg = json.loads(camp.config_json)
        assert cfg["EVO_POPULATION"] == 5, "not the module default"
        assert cfg["EVO_GENERATIONS"] == 2
        assert camp.status == "done" and camp.finished_ts
        conn.close()

    def test_last_completed_generation_ignores_unfinished_rows(self, state_conn):
        """finished_ts, not MAX(gen_index): resuming from a generation interrupted
        mid-flight would double-count members it already paid trials for."""
        for i in range(3):
            population.insert_generation(
                state_conn,
                population.Generation(
                    campaign_id="c", gen_index=i, window_start_ms=1,
                    window_end_ms=2, population_size=1, started_ts=0,
                ),
            )
        assert population.last_completed_generation(state_conn, "c") is None
        population.finish_generation(
            state_conn, "c", 1, n_evaluated=1, n_errors=0, n_unique_graphs=1,
            trials_cumulative=1, best_member_id="m", best_fitness=0.1,
            db_retries=0, wall_seconds=0.1,
        )
        assert population.last_completed_generation(state_conn, "c") == 1

    def test_resume_continues_and_the_ledger_keeps_climbing(
        self, tmp_path, seed_graph, stub_oracle
    ):
        path = str(tmp_path / "r.db")
        first = _run(tmp_path, seed_graph, state_path=path, generations=1,
                     population_size=4)
        assert first["ledger_total"] == 4 + len(first["audit"])
        conn = statestore.connect(path)
        # Reopen the campaign so the resume path has somewhere to go.
        conn.execute(
            "UPDATE campaigns SET generations = 3, status='running' WHERE campaign_id=?",
            (first["campaign_id"],),
        )
        conn.commit()
        second = runner.run_campaign(
            resume_campaign_id=first["campaign_id"], state_conn=conn,
            state_path=path, workers=1, wf_knobs=TINY_KNOBS,
            progress=lambda *_a, **_k: None,
        )
        assert [g["gen_index"] for g in second["generations"]] == [1, 2]
        assert second["ledger_total"] > first["ledger_total"]
        conn.close()

    def test_resume_rebuilds_seen_hashes(self, tmp_path, seed_graph, stub_oracle):
        path = str(tmp_path / "sh.db")
        first = _run(tmp_path, seed_graph, state_path=path, generations=1,
                     population_size=6)
        conn = statestore.connect(path)
        seen = population.seen_graph_hashes(conn, first["campaign_id"])
        assert len(seen) >= 6, (
            "a resumed campaign must not re-breed duplicates it already paid for"
        )
        conn.close()

    def test_resuming_an_unknown_campaign_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no campaign"):
            runner.run_campaign(
                resume_campaign_id="nope", state_path=str(tmp_path / "n.db"),
                progress=lambda *_a, **_k: None,
            )


class TestRetryClassification:
    """Trap 1: locked/busy retries, everything else propagates immediately."""

    def test_a_locked_error_is_retried(self):
        attempts = {"n": 0}

        def _flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result, retries = population.write_with_retry(
            _flaky, retries=5, sleep_s=0.0, sleep=lambda _s: None
        )
        assert result == "ok" and retries == 2

    def test_a_permanent_error_propagates_immediately(self):
        attempts = {"n": 0}

        def _broken():
            attempts["n"] += 1
            raise sqlite3.OperationalError("no such table: population_members")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            population.write_with_retry(
                _broken, retries=5, sleep_s=0.0, sleep=lambda _s: None
            )
        assert attempts["n"] == 1, "a permanent failure must not be retried"

    def test_the_retry_budget_is_finite(self):
        def _always_locked():
            raise sqlite3.OperationalError("database is busy")

        with pytest.raises(sqlite3.OperationalError):
            population.write_with_retry(
                _always_locked, retries=2, sleep_s=0.0, sleep=lambda _s: None
            )

    def test_update_member_result_rejects_an_unknown_column(self, state_conn):
        with pytest.raises(ValueError, match="unknown column"):
            population.update_member_result(state_conn, "m", {"nonsense": 1})


class TestAuditRound:
    def test_finalists_and_the_seed_share_one_fixed_window(
        self, tmp_path, seed_graph, stub_oracle
    ):
        """A3: fitness on jittered windows is not comparable across generations, so
        'the best beat the seed' is only answerable on one pre-declared window."""
        summary = _run(tmp_path, seed_graph, population_size=6, generations=2)
        audit = summary["audit"]
        assert audit, "the audit round produced no rows"
        assert sum(1 for r in audit if r["is_seed"]) == 1
        assert len(audit) <= config.EVO_FINALISTS + 1
        calls = stub_oracle["calls"]
        audit_spans = {
            (s, e) for _h, s, e in calls
            if (s, e) == (summary["audit_start_ms"], summary["audit_end_ms"])
        }
        assert audit_spans == {(summary["audit_start_ms"], summary["audit_end_ms"])}

    def test_the_audit_window_is_declared_before_any_generation_runs(
        self, tmp_path, seed_graph, stub_oracle
    ):
        """Declared in the campaigns row up front precisely so it cannot later be
        chosen to flatter a winner."""
        path = str(tmp_path / "a.db")
        summary = _run(tmp_path, seed_graph, state_path=path, generations=2)
        conn = statestore.connect(path)
        camp = population.load_campaign(conn, summary["campaign_id"])
        assert camp.audit_start_ms == summary["audit_start_ms"]
        assert camp.audit_end_ms == summary["audit_end_ms"]
        assert camp.started_ts <= (camp.finished_ts or camp.started_ts)
        conn.close()

    def test_each_audit_evaluation_charges_a_trial(
        self, tmp_path, seed_graph, stub_oracle
    ):
        summary = _run(tmp_path, seed_graph, population_size=4, generations=2)
        assert summary["ledger_total"] == 8 + len(summary["audit"])

    def test_the_verdict_names_the_failed_conditions(
        self, tmp_path, seed_graph, monkeypatch, stub_oracle
    ):
        real = oracle.GateOracle.evaluate

        def _failing(self, graph, **kwargs):
            res = real(self, graph, **kwargs)
            gate = dict(res.gate)
            gate["dsr"] = False
            gate["sample_adequacy"] = False
            return dataclasses.replace(res, gate=gate, passed=False)

        monkeypatch.setattr(oracle.GateOracle, "evaluate", _failing)
        summary = _run(tmp_path, seed_graph, population_size=4, generations=1)
        assert summary["verdict"]["gate_passed"] is False
        assert set(summary["verdict"]["failed_conditions"]) == {
            "dsr", "sample_adequacy"
        }

    def test_beats_seed_is_false_when_the_seed_wins(
        self, tmp_path, seed_graph, monkeypatch
    ):
        """The seed must be able to WIN — otherwise 'beats seed' is decoration."""
        seed_hash = fgraph.graph_hash(seed_graph)

        def _seed_favouring(self, graph, *, window_start_ms, window_end_ms):
            h = fgraph.graph_hash(graph)
            n = self._ledger.charge(graph_hash=h, params_hash="s",
                                    start_ms=window_start_ms, end_ms=window_end_ms)
            sharpe = 5.0 if h == seed_hash else 0.8
            return oracle.OracleResult(
                graph_hash=h, params_hash="s", window_start_ms=window_start_ms,
                window_end_ms=window_end_ms, oos_start_ms=window_start_ms,
                oos_end_ms=window_end_ms, n_trials_used=n, n_trades=40,
                sharpe=sharpe, dsr=0.01, ann_return_pct=0.3,
                max_drawdown_pct=0.1, bench_sharpe=0.7, bench_ann_return_pct=0.29,
                gate=dict.fromkeys(GATE_CONDITIONS, True), passed=True,
                eval_seconds=0.0, error="",
            )

        monkeypatch.setattr(oracle.GateOracle, "evaluate", _seed_favouring)
        summary = _run(tmp_path, seed_graph, population_size=4, generations=1)
        assert summary["verdict"]["beats_seed"] is False


class TestDryRun:
    def test_a_dry_run_charges_nothing(self, tmp_path, seed_graph):
        """A dry run that charged trials would be a trap."""
        path = str(tmp_path / "d.db")
        summary = _run(tmp_path, seed_graph, state_path=path, dry_run=True)
        assert summary["status"] == "dry-run"
        assert summary["trials_charged"] == 0
        assert summary["ledger_total"] == 0
        assert summary["audit"] == []
        conn = statestore.connect(path)
        members = conn.execute(
            "SELECT COUNT(*) FROM population_members WHERE campaign_id=?",
            (summary["campaign_id"],),
        ).fetchone()[0]
        assert members == 6, "generation 0 is still bred and recorded"
        conn.close()


class TestGuards:
    def test_a_train_end_past_the_frozen_ceiling_is_refused(self, tmp_path, seed_graph):
        """A caller must not be able to spend Phase 9's holdout."""
        ceiling = config.date_to_ms(config.EVO_TRAIN_END)
        with pytest.raises(oracle.HoldoutViolation, match="EVO_TRAIN_END"):
            _run(tmp_path, seed_graph, train_end_ms=ceiling + DAY_MS)

    def test_a_population_below_the_tournament_size_is_refused(
        self, tmp_path, seed_graph
    ):
        with pytest.raises(ValueError, match="EVO_TOURNAMENT_K"):
            _run(tmp_path, seed_graph, population_size=config.EVO_TOURNAMENT_K - 1)

    def test_run_campaign_needs_a_seed_graph(self, tmp_path):
        with pytest.raises(ValueError, match="needs seed_graph"):
            runner.run_campaign(
                state_path=str(tmp_path / "x.db"), progress=lambda *_a, **_k: None
            )

    def test_a_seedless_run_still_records_a_seed(self, tmp_path, seed_graph,
                                                 stub_oracle):
        """Even an unseeded run is replayable, because the derived seed is stored."""
        summary = _run(tmp_path, seed_graph, seed=None, generations=1,
                       population_size=4)
        assert isinstance(summary["seed"], int) and summary["seed"] > 0


class TestPoolWiring:
    """Trap 3: mp_context is spawn EXPLICITLY, and max_tasks_per_child is never set."""

    def test_the_pool_is_spawn_and_persistent(self, tmp_path, seed_graph, monkeypatch):
        captured: dict = {}

        class _FakePool:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, payload):
                class _F:
                    def result(self_inner):
                        return runner._error_result(payload, RuntimeError("stub"))

                return _F()

        monkeypatch.setattr(runner, "ProcessPoolExecutor", _FakePool)
        monkeypatch.setattr(runner, "as_completed", lambda futures: list(futures))
        runner._run_tasks(
            [{"member_id": "m", "member_index": 0, "graph": {},
              "window_start_ms": 1, "window_end_ms": 2, "campaign_json": "{}"}],
            workers=4, campaign_json="{}", progress=None,
        )
        assert captured["max_workers"] == 4
        assert captured["initializer"] is runner._worker_init
        assert captured["mp_context"].get_start_method() == "spawn"
        assert "max_tasks_per_child" not in captured, (
            "engine._CACHE is per-process; recycling workers would throw away the "
            "memo that makes a pooled walk-forward tractable (trap 2)"
        )

    def test_workers_one_runs_in_process(self, tmp_path, seed_graph, monkeypatch):
        def _boom(**kwargs):
            raise AssertionError("workers=1 must not construct a pool")

        monkeypatch.setattr(runner, "ProcessPoolExecutor", _boom)
        spec = json.dumps(
            {
                "campaign_id": "c", "symbols": ["BTCUSDT"],
                "train_start_ms": START, "train_end_ms": START + 900 * DAY_MS,
                "ohlcv_path": config.DB_PATH,
                "state_path": str(tmp_path / "w.db"), **TINY_KNOBS,
            },
            sort_keys=True,
        )
        payload = {
            "member_id": "m", "member_index": 0, "graph": seed_graph.to_dict(),
            "window_start_ms": START, "window_end_ms": START + 30 * DAY_MS,
            "campaign_json": spec,
        }
        out = runner._run_tasks([payload], workers=1, campaign_json=spec,
                                progress=None)
        assert set(out) == {0}


class TestCalibration:
    def test_capacity_uses_the_distinct_graph_figure(
        self, tmp_path, seed_graph, monkeypatch
    ):
        """Re-evaluating ONE graph measures the framework's candidate memo, not a
        campaign: every member of a real generation is a different graph. Sizing on
        the same-graph figure overstates throughput."""
        timings = iter([10.0, 1.0, 1.0, 10.0, 4.0, 4.0])
        base = 0.0

        class _Clock:
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                try:
                    self.t += next(timings)
                except StopIteration:
                    self.t += 1.0
                return self.t

        # perf_counter is called twice per evaluation (before and after), so a
        # monotonic stub that advances by the intended duration on the SECOND call
        # is what produces the timings above.
        seq = [0.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 22.0, 22.0, 26.0, 26.0, 30.0]
        it = iter(seq)
        monkeypatch.setattr(runner.time, "perf_counter", lambda: next(it, 30.0))
        monkeypatch.setattr(
            oracle.GateOracle, "evaluate",
            lambda self, graph, **kw: oracle.OracleResult(
                graph_hash="h", params_hash="p", window_start_ms=1,
                window_end_ms=2, oos_start_ms=1, oos_end_ms=2, n_trials_used=1,
                n_trades=1, sharpe=1.0, dsr=0.1, ann_return_pct=0.1,
                max_drawdown_pct=0.1, bench_sharpe=0.1,
                bench_ann_return_pct=0.1,
                gate=dict.fromkeys(GATE_CONDITIONS, True), passed=True,
                eval_seconds=0.0,
            ),
        )
        out = runner.calibrate(
            seed_graph=seed_graph, symbols=("BTCUSDT",), repeats=3, workers=6,
            budget_hours=1.0, state_path=str(tmp_path / "cal.db"),
            progress=lambda *_a, **_k: None,
        )
        assert out["cold_seconds"] == pytest.approx(10.0)
        assert out["warm_seconds"] == pytest.approx(1.0)
        assert out["warm_distinct_seconds"] == pytest.approx(4.0)
        assert out["capacity"] == int(6 * 1.0 * 3600 // 4.0)
        assert out["capacity"] < int(6 * 1.0 * 3600 // out["warm_seconds"]), (
            "sizing on the same-graph figure would overstate capacity"
        )

    def test_backtests_per_eval_is_explainable(self, tmp_path, seed_graph,
                                               monkeypatch):
        monkeypatch.setattr(
            oracle.GateOracle, "evaluate",
            lambda self, graph, **kw: oracle.OracleResult(
                graph_hash="h", params_hash="p", window_start_ms=1, window_end_ms=2,
                oos_start_ms=1, oos_end_ms=2, n_trials_used=1, n_trades=1,
                sharpe=1.0, dsr=0.1, ann_return_pct=0.1, max_drawdown_pct=0.1,
                bench_sharpe=0.1, bench_ann_return_pct=0.1,
                gate=dict.fromkeys(GATE_CONDITIONS, True), passed=True,
                eval_seconds=0.0,
            ),
        )
        out = runner.calibrate(
            seed_graph=seed_graph, symbols=("BTCUSDT", "ETHUSDT"), repeats=2,
            state_path=str(tmp_path / "cal2.db"), progress=lambda *_a, **_k: None,
        )
        a = out["arithmetic"]
        assert out["backtests_per_eval"] == (a["n_folds"] * 2 + 1) * 2

    def test_capacity_bounds_every_suggested_pair(self, tmp_path, seed_graph,
                                                  monkeypatch):
        monkeypatch.setattr(
            oracle.GateOracle, "evaluate",
            lambda self, graph, **kw: oracle.OracleResult(
                graph_hash="h", params_hash="p", window_start_ms=1, window_end_ms=2,
                oos_start_ms=1, oos_end_ms=2, n_trials_used=1, n_trades=1,
                sharpe=1.0, dsr=0.1, ann_return_pct=0.1, max_drawdown_pct=0.1,
                bench_sharpe=0.1, bench_ann_return_pct=0.1,
                gate=dict.fromkeys(GATE_CONDITIONS, True), passed=True,
                eval_seconds=0.0,
            ),
        )
        out = runner.calibrate(
            seed_graph=seed_graph, symbols=("BTCUSDT",), repeats=2,
            state_path=str(tmp_path / "cal3.db"), progress=lambda *_a, **_k: None,
        )
        for pop, gens in out["pairs"]:
            assert pop * gens <= out["capacity"]
            assert pop >= 8 * config.EVO_TOURNAMENT_K

    def test_calibration_charges_its_own_trials_under_a_throwaway_campaign(
        self, tmp_path, seed_graph, monkeypatch
    ):
        """Calibration evaluations ARE evaluations. Charging them is the safe
        direction (A6), and a throwaway campaign id keeps them out of a real
        campaign's DSR."""
        monkeypatch.setattr(
            oracle.GateOracle, "evaluate",
            lambda self, graph, **kw: (
                self._ledger.charge(graph_hash="h", params_hash="p", start_ms=1,
                                    end_ms=2),
                oracle.OracleResult(
                    graph_hash="h", params_hash="p", window_start_ms=1,
                    window_end_ms=2, oos_start_ms=1, oos_end_ms=2,
                    n_trials_used=1, n_trades=1, sharpe=1.0, dsr=0.1,
                    ann_return_pct=0.1, max_drawdown_pct=0.1, bench_sharpe=0.1,
                    bench_ann_return_pct=0.1,
                    gate=dict.fromkeys(GATE_CONDITIONS, True), passed=True,
                    eval_seconds=0.0,
                ),
            )[1],
        )
        out = runner.calibrate(
            seed_graph=seed_graph, symbols=("BTCUSDT",), repeats=2,
            state_path=str(tmp_path / "cal4.db"), progress=lambda *_a, **_k: None,
        )
        assert out["campaign_id"].startswith("calibrate-")
        assert out["trials_charged"] >= 2

    def test_the_seed_is_audited_exactly_once(self, tmp_path, seed_graph,
                                              stub_oracle):
        """The seed is member 0 of generation 0, so if its own row ranks in the
        shortlist it must not be audited twice: two identical rows, one of them not
        marked is_seed, and one extra trial charged for a comparison already made."""
        summary = _run(tmp_path, seed_graph, population_size=4, generations=1)
        seed_hash = summary["audit"][0]["graph_hash"] and next(
            r["graph_hash"] for r in summary["audit"] if r["is_seed"]
        )
        assert sum(1 for r in summary["audit"] if r["is_seed"]) == 1
        assert sum(1 for r in summary["audit"] if r["graph_hash"] == seed_hash) == 1

    def test_resume_reloads_scored_parents_and_does_not_reseed(
        self, tmp_path, seed_graph, stub_oracle
    ):
        """--resume must continue the SEARCH, not only the bookkeeping. Without
        reloading the previous generation's scored members, a resumed campaign
        re-seeds from the original graph and silently throws away the evolutionary
        progress it already paid trials for."""
        path = str(tmp_path / "rs.db")
        first = _run(tmp_path, seed_graph, state_path=path, generations=2,
                     population_size=6)
        conn = statestore.connect(path)
        reloaded = runner._reload_scored(
            conn, population.load_campaign(conn, first["campaign_id"]), 1
        )
        assert len(reloaded) == 6
        assert all(m.graph is not None for _i, _f, m in reloaded)
        assert [i for i, _f, _m in reloaded] == list(range(6))
        assert all(f.tier in ("A", "B", "C") for _i, f, _m in reloaded)

        conn.execute(
            "UPDATE campaigns SET generations=3, status='running' WHERE campaign_id=?",
            (first["campaign_id"],),
        )
        conn.commit()
        second = runner.run_campaign(
            resume_campaign_id=first["campaign_id"], state_conn=conn,
            state_path=path, workers=1, wf_knobs=TINY_KNOBS,
            progress=lambda *_a, **_k: None,
        )
        gen2 = population.generation_members(conn, first["campaign_id"], 2)
        # Generation 2 was BRED (elites carried + offspring mutated), not re-seeded:
        # a re-seeded generation would have exactly one role='seed' row.
        assert {r["role"] for r in gen2} == {"elite", "offspring"}
        assert [g["gen_index"] for g in second["generations"]] == [2]
        conn.close()

    def test_reload_skips_unscored_and_corrupt_rows(self, tmp_path, seed_graph,
                                                    stub_oracle):
        path = str(tmp_path / "rc.db")
        first = _run(tmp_path, seed_graph, state_path=path, generations=1,
                     population_size=6)
        conn = statestore.connect(path)
        conn.execute(
            "UPDATE population_members SET fitness=NULL, tier='' "
            "WHERE campaign_id=? AND gen_index=0 AND member_index IN (0,1)",
            (first["campaign_id"],),
        )
        conn.execute(
            "UPDATE population_members SET graph_json='{not json' "
            "WHERE campaign_id=? AND gen_index=0 AND member_index=2",
            (first["campaign_id"],),
        )
        conn.commit()
        reloaded = runner._reload_scored(
            conn, population.load_campaign(conn, first["campaign_id"]), 0
        )
        assert [i for i, _f, _m in reloaded] == [3, 4, 5]
        conn.close()
