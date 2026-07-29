"""
Tests for the persistent trial ledger (v0.3.0 Phase 1, contract §4/§6).

The ledger's whole purpose is that DSR's n_trials is HONEST: it counts every
evaluation the oracle performs, cumulatively, on disk, so an overnight run that
restarts cannot reset its own degrees-of-freedom count.
"""

import pytest

from trading_bot import config
from trading_bot.backtest import trials, walkforward
from trading_bot.backtest.engine import BacktestParams
from trading_bot.backtest.walkforward import DAY_MS, walk_forward_pooled
from trading_bot.data import statestore

from tests.test_backtest import _beatable_benchmark, per_symbol_fake_run_factory

CAMPAIGN = "test-campaign"


@pytest.fixture
def state_conn(tmp_path):
    """File-backed, NOT ":memory:" — the reconnect test needs a real file."""
    conn = statestore.connect(str(tmp_path / "state.db"))
    yield conn
    conn.close()


@pytest.fixture
def ledger(state_conn):
    return trials.TrialLedger(state_conn, CAMPAIGN)


class TestSchema:
    def test_connect_creates_no_tables(self, state_conn):
        """DDL belongs to the module that owns the table (contract §6), so
        statestore.connect() must not create anything."""
        rows = state_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert rows == []

    def test_ensure_schema_is_idempotent(self, state_conn):
        trials.ensure_schema(state_conn)
        trials.ensure_schema(state_conn)
        names = {
            r[0]
            for r in state_conn.execute("SELECT name FROM sqlite_master").fetchall()
        }
        assert trials.LEDGER_TABLE in names
        assert f"idx_{trials.LEDGER_TABLE}_campaign" in names

    def test_state_db_path_is_a_second_database(self):
        assert config.STATE_DB_PATH == "data/state.db"
        assert config.STATE_DB_PATH != config.DB_PATH


class TestCounting:
    def test_record_returns_post_insert_count(self, ledger):
        assert ledger.record(
            graph_hash="g", params_hash="p", start_ms=0, end_ms=DAY_MS
        ) == 1
        assert ledger.record(
            graph_hash="g", params_hash="q", start_ms=0, end_ms=DAY_MS
        ) == 2

    def test_repeat_evaluations_are_separate_rows(self, ledger):
        """No UNIQUE constraint: the ledger counts evaluations PERFORMED, and
        deduping would understate n_trials in the flattering direction.
        distinct_count() serves whoever wants the other number."""
        for _ in range(3):
            ledger.record(graph_hash="g", params_hash="p", start_ms=0, end_ms=DAY_MS)
        assert ledger.count() == 3
        assert ledger.distinct_count() == 1

    def test_campaigns_are_isolated(self, state_conn):
        a = trials.TrialLedger(state_conn, "campaign-a")
        b = trials.TrialLedger(state_conn, "campaign-b")
        a.record(graph_hash="g", params_hash="p", start_ms=0, end_ms=1)
        a.record(graph_hash="g", params_hash="q", start_ms=0, end_ms=1)
        b.record(graph_hash="g", params_hash="p", start_ms=0, end_ms=1)
        assert a.count() == 2
        assert b.count() == 1

    def test_empty_campaign_raises(self, state_conn):
        with pytest.raises(ValueError):
            trials.TrialLedger(state_conn, "")

    def test_records_round_trip(self, ledger):
        ledger.record(
            graph_hash=trials.LEGACY_GRAPH_HASH, params_hash="p",
            start_ms=10, end_ms=20, ts=1_700_000_000_000,
        )
        (rec,) = ledger.records()
        assert rec == trials.TrialRecord(
            campaign=CAMPAIGN, graph_hash=trials.LEGACY_GRAPH_HASH,
            params_hash="p", start_ms=10, end_ms=20, ts=1_700_000_000_000,
        )


class TestPersistence:
    def test_count_survives_reconnect(self, tmp_path):
        """The whole reason the ledger is on disk: an overnight run that
        restarts must not reset its own degrees-of-freedom count."""
        path = str(tmp_path / "state.db")
        first = statestore.connect(path)
        trials.TrialLedger(first, CAMPAIGN).record(
            graph_hash="g", params_hash="p", start_ms=0, end_ms=1
        )
        first.close()

        second = statestore.connect(path)
        assert trials.TrialLedger(second, CAMPAIGN).count() == 1
        second.close()

    def test_ts_is_epoch_milliseconds(self, ledger):
        """Contract §6: epoch MILLISECONDS, UTC, never formatted strings."""
        ledger.record(graph_hash="g", params_hash="p", start_ms=0, end_ms=1)
        (rec,) = ledger.records()
        assert isinstance(rec.ts, int)
        assert rec.ts > 1_600_000_000_000  # seconds would be ~1.7e9, not 1.7e12


class TestHashing:
    def test_stable_hash_is_order_independent(self):
        assert trials.stable_hash({"a": 1, "b": 2}) == trials.stable_hash({"b": 2, "a": 1})

    def test_stable_hash_is_reproducible_across_processes(self):
        """PINNED LITERAL. The digest must be reproducible ACROSS PROCESSES;
        PYTHONHASHSEED randomizes str hashing, so a ledger keyed on Python's
        builtin hash() would double-count the same configuration after a
        restart — precisely the failure this module exists to prevent.

        The literal is the first 16 hex chars of sha256 over the canonical JSON
        '"x"', verified independently of the implementation:
            printf '"x"' | shasum -a 256   ->  ba2df4903a2c14e8...
        """
        assert trials.stable_hash("x") == "ba2df4903a2c14e8"

    def test_params_hash_of_a_dataclass_matches_its_field_dict(self):
        p = BacktestParams()
        import dataclasses

        expected = trials.stable_hash(
            {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}
        )
        assert trials.params_hash(p) == expected

    def test_params_hash_of_none_is_the_empty_mapping(self):
        assert trials.params_hash(None) == trials.stable_hash({})

    def test_params_hash_accepts_a_mapping(self):
        assert trials.params_hash({"a": 1}) == trials.stable_hash({"a": 1})

    def test_default_str_collides_objects_sharing_a_repr(self):
        """DOCUMENTED LIMITATION, not a bug: default=str makes stable_hash
        total rather than raising mid-run, at the cost that two non-JSONable
        objects with the same repr collide. Acceptable — every params value in
        this codebase is bool/int/float/str."""

        class A:
            def __repr__(self):
                return "same"

        class B:
            def __repr__(self):
                return "same"

        assert trials.stable_hash(A()) == trials.stable_hash(B())


class TestWalkForwardIntegration:
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    SPAN = dict(start_ms=0, end_ms=25 * DAY_MS)
    KNOBS = dict(train_days=10, test_days=5, oos_days=5, min_trades=12)

    @pytest.fixture(autouse=True)
    def _stub_walkforward(self, monkeypatch):
        monkeypatch.setattr(
            walkforward, "run_backtest", per_symbol_fake_run_factory()
        )
        monkeypatch.setattr(
            walkforward, "buy_and_hold", lambda conn, syms, **kw: _beatable_benchmark()
        )

    def test_ledgerless_run_reproduces_the_legacy_count(self):
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        n_combos = len(walkforward._combos(walkforward.DEFAULT_GRID))
        assert result.n_trials_used == n_combos * len(result.folds)

    def test_ledgered_run_charges_the_cumulative_count(self, state_conn):
        led = trials.TrialLedger(state_conn, CAMPAIGN)
        result = walk_forward_pooled(
            None, self.SYMBOLS, **self.SPAN, **self.KNOBS, ledger=led
        )
        assert result.n_trials_used == led.count()

    def test_ledgered_count_exceeds_the_legacy_count(self, state_conn):
        """The ledger counts the robustness NEIGHBOUR PROBES that the ledgerless
        default deliberately omits, so the ledgered n_trials is strictly larger
        and the DSR strictly worse. That is the point (contract §4)."""
        led = trials.TrialLedger(state_conn, CAMPAIGN)
        result = walk_forward_pooled(
            None, self.SYMBOLS, **self.SPAN, **self.KNOBS, ledger=led
        )
        n_combos = len(walkforward._combos(walkforward.DEFAULT_GRID))
        assert result.n_trials_used > n_combos * len(result.folds)

    def test_ledger_records_the_legacy_graph_sentinel(self, state_conn):
        led = trials.TrialLedger(state_conn, CAMPAIGN)
        walk_forward_pooled(
            None, self.SYMBOLS, **self.SPAN, **self.KNOBS, ledger=led
        )
        assert {r.graph_hash for r in led.records()} == {trials.LEGACY_GRAPH_HASH}
