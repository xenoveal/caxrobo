"""
The ARCHITECTURAL-INVARIANT test module for v0.3.0 Phase 6.

Four of the six honesty properties in contract §4 are mechanical claims about
this package's source and signatures, so they are asserted from inside pytest
rather than trusted to review: there is no linter or type checker configured in
this repo (KNOWN-LIMITATIONS §8), so a test is the only enforcement that runs.
"""

import inspect
import json
import pathlib
import random

import pytest

from trading_bot import config
from trading_bot.backtest import trials
from trading_bot.backtest import walkforward
from trading_bot.data import statestore
from trading_bot.evolution import oracle
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry

START = 1_700_000_000_000
DAY_MS = 86_400_000

# Derived from config, never hardcoded: a future tier shift must not leave these
# tests green on the old timeframes while production moves (contract §8).
TRAIN_START = START
TRAIN_END = START + 900 * DAY_MS

EVOLUTION_DIR = pathlib.Path(oracle.__file__).parent


@pytest.fixture(autouse=True)
def _registry_loaded():
    registry.load_all()
    yield


@pytest.fixture
def state_conn(tmp_path):
    """A throwaway state.db. NEVER the real data/state.db."""
    conn = statestore.connect(str(tmp_path / "s.db"))
    trials.ensure_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def ledger(state_conn):
    return oracle.TrialLedger(state_conn, "test-campaign")


@pytest.fixture
def seed_graph():
    return fgraph.load("data/strategies/thin-slice.strategy.json")


def _stub_wf(monkeypatch, *, calls: list, sharpe=1.2, dsr=0.4, n_trades=42,
             gate_overrides=None, raises=None):
    """Stub walk_forward_pooled AT THE NAME BOUND INSIDE oracle.py.

    Patching backtest.walkforward.walk_forward_pooled would NOT intercept
    oracle.py's already-imported reference, so a ledger-parity assertion would
    silently exercise the real gate over the real database instead of the stub.
    """
    verdicts = dict.fromkeys(walkforward.GATE_CONDITIONS, True)
    verdicts.update(gate_overrides or {})

    class _Bench:
        basket = {"sharpe": 0.7, "ann_return_pct": 0.29}

    class _Result:
        oos_start = 1
        oos_end = 2
        n_trials_used = 0
        oos_metrics = {"n_trades": n_trades}
        oos_equity = {
            "sharpe": sharpe, "dsr": dsr, "ann_return_pct": 0.5,
            "max_drawdown_pct": 0.1,
        }
        benchmark = _Bench()
        gate = verdicts
        passed = all(verdicts.values())

    def _fake(conn, symbols, **kwargs):
        calls.append(kwargs)
        if raises is not None:
            raise raises
        res = _Result()
        res.n_trials_used = kwargs.get("n_trials", 0)
        return res

    monkeypatch.setattr(oracle, "walk_forward_pooled", _fake)


class TestLedgerIsRequired:
    """Contract §4.2: nothing may score a candidate without a ledger handle."""

    def test_ledger_is_the_first_positional_parameter_with_no_default(self):
        params = list(inspect.signature(oracle.GateOracle.__init__).parameters.values())
        assert params[1].name == "ledger"
        assert params[1].default is inspect.Parameter.empty
        assert params[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        # Everything else is keyword-only, so a caller cannot slide a connection
        # into the ledger slot positionally.
        for p in params[2:]:
            assert p.kind is inspect.Parameter.KEYWORD_ONLY, p.name

    def test_constructing_without_a_ledger_is_a_TypeError(self):
        with pytest.raises(TypeError):
            oracle.GateOracle()  # type: ignore[call-arg]

    def test_constructing_with_only_keywords_is_a_TypeError(self, state_conn):
        with pytest.raises(TypeError):
            oracle.GateOracle(  # type: ignore[call-arg]
                ohlcv_conn=state_conn, symbols=("BTCUSDT",),
                train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
            )

    def test_a_none_ledger_is_refused(self, state_conn):
        with pytest.raises(ValueError, match="requires a ledger handle"):
            oracle.GateOracle(
                None, ohlcv_conn=state_conn, symbols=("BTCUSDT",),
                train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
            )

    def test_a_stand_in_that_cannot_charge_is_refused(self, state_conn):
        class NotALedger:
            def count(self):
                return 0

        with pytest.raises(TypeError, match="does not satisfy LedgerHandle"):
            oracle.GateOracle(
                NotALedger(), ohlcv_conn=state_conn, symbols=("BTCUSDT",),
                train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
            )

    def test_no_free_scoring_function_is_exported(self):
        assert oracle.__all__ == (
            "GateOracle", "HoldoutViolation", "LedgerHandle", "OracleResult",
            "TrialLedger", "evo_grid",
        )
        # No module-level callable takes a graph and returns a number.
        for name in oracle.__all__:
            obj = getattr(oracle, name)
            if inspect.isfunction(obj):
                assert "graph" not in inspect.signature(obj).parameters, name


class TestNoSecondOracle:
    """Contract §4.1: one code path. Enforced by scanning this package's source.

    The scan parses each module with `ast` rather than grepping text, so a
    docstring that NAMES the forbidden path (oracle.py's does, deliberately —
    "scripts/bruteforce/core.score is NOT an oracle") is not mistaken for a call
    to it. A grep-based version of this test fails on its own documentation, and
    the temptation would then be to delete the documentation.
    """

    def _sources(self):
        return sorted(EVOLUTION_DIR.glob("*.py"))

    def _identifiers(self, path) -> set[str]:
        """Every name the CODE (never a comment or string) can reach."""
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.update(a.name.split("."))
                    if a.asname:
                        names.add(a.asname)
            elif isinstance(node, ast.ImportFrom):
                names.update((node.module or "").split("."))
                for a in node.names:
                    names.add(a.name)
                    if a.asname:
                        names.add(a.asname)
        return names

    def test_walk_forward_pooled_is_reachable_from_exactly_one_module(self):
        importers = [
            p.name for p in self._sources()
            if "walk_forward_pooled" in self._identifiers(p)
        ]
        assert importers == ["oracle.py"], (
            f"walk_forward_pooled is reachable from {importers}; contract §4.1 "
            f"allows it in oracle.py alone"
        )

    @pytest.mark.parametrize(
        "forbidden",
        ["bruteforce", "run_backtest", "run_graph_backtest",
         "compute_equity_metrics", "compute_metrics"],
    )
    def test_no_second_scoring_path_is_reachable(self, forbidden):
        offenders = [
            p.name for p in self._sources() if forbidden in self._identifiers(p)
        ]
        assert offenders == [], (
            f"{forbidden!r} is reachable from {offenders}; evolution/ must reach "
            f"the engine only through walk_forward_pooled (contract §4.1)"
        )

    def test_bruteforce_core_score_is_never_called(self):
        """`core.score` specifically: the retired research sweep's own scorer.
        `Fitness.score` is a legitimate attribute of this package's own type, so
        the bare name cannot be banned — the MODULE-QUALIFIED call is what is
        forbidden (contract §0b, §4.1)."""
        import ast

        for p in self._sources():
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "score"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("core", "bruteforce")
                ):
                    pytest.fail(f"{p.name}:{node.lineno} reaches core.score")

    def test_builtin_hash_is_never_called(self):
        """PYTHONHASHSEED salts str hashing per process; a spawned worker gets a
        different salt, so builtin hash() would make a campaign unreproducible in
        a way that looks exactly like nondeterministic code."""
        import ast

        for p in self._sources():
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "hash"
                ):
                    pytest.fail(f"{p.name}:{node.lineno} calls builtin hash()")

    def test_no_other_phases_constants_are_read(self):
        """Contract §7 reserves EVO_* for this phase and other prefixes for others."""
        import ast

        for p in self._sources():
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "config"
                ):
                    for prefix in ("HOLDOUT_", "CAMPAIGN_", "REVIEW_", "UI_"):
                        assert not node.attr.startswith(prefix), (
                            f"{p.name}:{node.lineno} reads config.{node.attr}, "
                            f"which belongs to another phase (contract §7)"
                        )

    def test_walkforward_py_is_not_edited_by_this_phase(self):
        """Contract §7: 'Phase 6 does not edit this file; it wraps it.' Asserted on
        the file's CONTENT — no EVO_* name and no evolution import may appear in
        it — because `git diff` is not available inside a test."""
        text = pathlib.Path(walkforward.__file__).read_text(encoding="utf-8")
        assert "EVO_" not in text
        assert "evolution" not in text


class TestEvoGrid:
    """A1 + the GRID_DEGENERACY_TRAP at walkforward.py:500-505."""

    def test_exactly_one_combo(self):
        grid = oracle.evo_grid()
        assert len(walkforward._combos(grid)) == 1

    def test_every_axis_is_run_level(self):
        """A graph carries its own parameters, so walk_forward_pooled REFUSES any
        other axis rather than silently ignoring it."""
        assert set(oracle.evo_grid()) <= walkforward._RUN_LEVEL_AXES

    def test_the_single_value_is_the_config_default(self):
        """walkforward._default_combo() falls back to config defaults when no
        combo reaches min_trades, then ASSERTS the result is on the grid. A
        one-combo grid holding anything else raises AssertionError the first time
        a fold comes up short — silently, until it does."""
        grid = oracle.evo_grid()
        assert walkforward._default_combo(grid) == {
            axis: values[0] for axis, values in grid.items()
        }
        assert grid["max_hold_bars"] == (config.MAX_HOLD_BARS_TRIGGER,)

    def test_the_grid_is_accepted_by_walk_forward_pooled_with_a_graph(self, seed_graph):
        """The complement of the above: DEFAULT_GRID is refused with a graph, and
        evo_grid() is not."""
        with pytest.raises(ValueError, match="carries its own parameters"):
            walkforward.walk_forward_pooled(
                None, ["BTCUSDT"], start_ms=0, end_ms=1,
                grid=walkforward.DEFAULT_GRID, strategy=seed_graph,
            )
        # evo_grid() gets past the guard and fails later, on the span, proving the
        # grid itself was accepted.
        with pytest.raises(ValueError, match="span too short"):
            walkforward.walk_forward_pooled(
                None, ["BTCUSDT"], start_ms=0, end_ms=1,
                grid=oracle.evo_grid(), strategy=seed_graph,
            )


class TestHoldoutCeiling:
    """Contract §4.4 / A5: the final holdout is never seen by evolution."""

    def _oracle(self, ledger, state_conn):
        return oracle.GateOracle(
            ledger, ohlcv_conn=state_conn, symbols=("BTCUSDT",),
            train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
        )

    def test_a_span_past_the_ceiling_raises(self, ledger, state_conn, seed_graph):
        orc = self._oracle(ledger, state_conn)
        with pytest.raises(oracle.HoldoutViolation, match="Phase 9's holdout"):
            orc.evaluate(
                seed_graph, window_start_ms=TRAIN_START,
                window_end_ms=TRAIN_END + 1,
            )

    def test_the_refusal_charges_nothing(self, ledger, state_conn, seed_graph):
        """A refused evaluation must cost nothing, or probing the boundary would
        inflate the DSR penalty for evaluations that never happened."""
        orc = self._oracle(ledger, state_conn)
        before = ledger.count()
        with pytest.raises(oracle.HoldoutViolation):
            orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                         window_end_ms=TRAIN_END + DAY_MS)
        assert ledger.count() == before == 0

    def test_the_refusal_happens_before_the_gate_is_called(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        calls: list = []
        _stub_wf(monkeypatch, calls=calls)
        orc = self._oracle(ledger, state_conn)
        with pytest.raises(oracle.HoldoutViolation):
            orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                         window_end_ms=TRAIN_END + DAY_MS)
        assert calls == []

    def test_a_span_before_the_start_also_raises(self, ledger, state_conn, seed_graph):
        orc = self._oracle(ledger, state_conn)
        with pytest.raises(oracle.HoldoutViolation):
            orc.evaluate(seed_graph, window_start_ms=TRAIN_START - 1,
                         window_end_ms=TRAIN_END)

    def test_the_ceiling_bar_itself_is_allowed(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        """end_ms == train_end_ms is legal: walk_forward_pooled's bounds are
        INCLUSIVE, and the ceiling bar is the last one Phase 9 does NOT own."""
        calls: list = []
        _stub_wf(monkeypatch, calls=calls)
        orc = self._oracle(ledger, state_conn)
        orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                     window_end_ms=TRAIN_END)
        assert len(calls) == 1


class TestLedgerParity:
    """Contract §4.2: ledger.count() == the number of oracle evaluations."""

    def _oracle(self, ledger, state_conn):
        return oracle.GateOracle(
            ledger, ohlcv_conn=state_conn, symbols=("BTCUSDT", "ETHUSDT"),
            train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
        )

    def test_every_evaluation_including_failures_is_charged(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        calls: list = []
        _stub_wf(monkeypatch, calls=calls)
        orc = self._oracle(ledger, state_conn)
        for i in range(5):
            orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                         window_end_ms=TRAIN_START + (400 + i) * DAY_MS)
        assert ledger.count() == 5 == len(calls)

        # A ValueError from a short span is a recorded result, and the row STAYS.
        _stub_wf(monkeypatch, calls=calls, raises=ValueError("span too short"))
        res = orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                           window_end_ms=TRAIN_START + DAY_MS)
        assert res.error and res.sharpe is None
        assert ledger.count() == 6, "a failed evaluation is still an evaluation (A6)"

    def test_charge_happens_before_scoring(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        """The row is written first, so a crash costs a trial: over-charging is
        the only safe error (A6)."""
        observed: list[int] = []

        def _fake(conn, symbols, **kwargs):
            observed.append(ledger.count())
            raise ValueError("boom, mid-scoring")

        monkeypatch.setattr(oracle, "walk_forward_pooled", _fake)
        orc = self._oracle(ledger, state_conn)
        orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                     window_end_ms=TRAIN_START + 400 * DAY_MS)
        assert observed == [1], "the ledger was not charged before scoring"

    def test_ledger_rows_record_the_real_span_and_graph(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        _stub_wf(monkeypatch, calls=[])
        orc = self._oracle(ledger, state_conn)
        end = TRAIN_START + 500 * DAY_MS
        orc.evaluate(seed_graph, window_start_ms=TRAIN_START, window_end_ms=end)
        (row,) = ledger._ledger.records()
        assert row.graph_hash == fgraph.graph_hash(seed_graph)
        assert (row.start_ms, row.end_ms) == (TRAIN_START, end)
        assert row.campaign == "test-campaign"

    def test_params_hash_is_constant_across_candidates(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        """One configuration was evaluated per candidate, not twelve — the
        params_hash records exactly that."""
        _stub_wf(monkeypatch, calls=[])
        orc = self._oracle(ledger, state_conn)
        rng = random.Random(7)
        spec = registry.get("mutator.param-jitter")
        child, _ = spec.impl(seed_graph, rng, **spec.defaults())
        orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                     window_end_ms=TRAIN_START + 400 * DAY_MS)
        orc.evaluate(child, window_start_ms=TRAIN_START,
                     window_end_ms=TRAIN_START + 400 * DAY_MS)
        a, b = ledger._ledger.records()
        assert a.params_hash == b.params_hash
        assert a.graph_hash != b.graph_hash


class TestNTrialsPassthrough:
    """Contract §4: n_trials is the campaign's CUMULATIVE count, and it grows."""

    def test_n_trials_is_the_post_charge_cumulative_count(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        calls: list = []
        _stub_wf(monkeypatch, calls=calls)
        orc = oracle.GateOracle(
            ledger, ohlcv_conn=state_conn, symbols=("BTCUSDT",),
            train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
        )
        seen = []
        for _ in range(3):
            res = orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                               window_end_ms=TRAIN_START + 400 * DAY_MS)
            seen.append(res.n_trials_used)
        assert seen == [1, 2, 3], "n_trials must be strictly increasing"
        assert [c["n_trials"] for c in calls] == [1, 2, 3]
        assert seen[-1] == ledger.count()

    def test_the_ledger_is_not_also_passed_down(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        """Passing `ledger=` into walk_forward_pooled too would add one row per
        FOLD evaluation, making the next candidate's charge depend on the previous
        one's fold count. n_trials is passed explicitly instead."""
        calls: list = []
        _stub_wf(monkeypatch, calls=calls)
        orc = oracle.GateOracle(
            ledger, ohlcv_conn=state_conn, symbols=("BTCUSDT",),
            train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
        )
        orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                     window_end_ms=TRAIN_START + 400 * DAY_MS)
        assert "ledger" not in calls[0]

    def test_a_persistent_ledger_continues_across_processes(self, tmp_path, seed_graph):
        """The count survives a restart, which is why an overnight run that
        crashes and resumes cannot reset its own degrees-of-freedom count."""
        path = str(tmp_path / "persist.db")
        c1 = statestore.connect(path)
        led1 = oracle.TrialLedger(c1, "camp")
        led1.charge(graph_hash="g", params_hash="p", start_ms=1, end_ms=2)
        c1.close()
        c2 = statestore.connect(path)
        led2 = oracle.TrialLedger(c2, "camp")
        assert led2.count() == 1
        assert led2.charge(graph_hash="g", params_hash="p", start_ms=1, end_ms=2) == 2
        c2.close()


class TestOracleResultShape:
    """The result must cross a spawn boundary and must never fabricate a metric."""

    def test_result_is_picklable_and_asdict_able(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        import dataclasses
        import pickle

        _stub_wf(monkeypatch, calls=[])
        orc = oracle.GateOracle(
            ledger, ohlcv_conn=state_conn, symbols=("BTCUSDT",),
            train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
        )
        res = orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                           window_end_ms=TRAIN_START + 400 * DAY_MS)
        assert pickle.loads(pickle.dumps(res)) == res
        d = dataclasses.asdict(res)
        assert json.loads(json.dumps(d))["gate"].keys() == set(
            walkforward.GATE_CONDITIONS
        )

    def test_undefined_metrics_stay_none(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        _stub_wf(monkeypatch, calls=[], sharpe=None, dsr=None, n_trades=0)
        orc = oracle.GateOracle(
            ledger, ohlcv_conn=state_conn, symbols=("BTCUSDT",),
            train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
        )
        res = orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                           window_end_ms=TRAIN_START + 400 * DAY_MS)
        assert res.sharpe is None and res.dsr is None
        assert res.n_trades == 0

    def test_gate_keys_are_exactly_phase_1s_conditions(
        self, ledger, state_conn, seed_graph, monkeypatch
    ):
        """Keyed off walkforward.GATE_CONDITIONS, not literal strings, so a
        Phase 1 rename fails loudly instead of silently mis-tiering everything."""
        _stub_wf(monkeypatch, calls=[])
        orc = oracle.GateOracle(
            ledger, ohlcv_conn=state_conn, symbols=("BTCUSDT",),
            train_start_ms=TRAIN_START, train_end_ms=TRAIN_END,
        )
        res = orc.evaluate(seed_graph, window_start_ms=TRAIN_START,
                           window_end_ms=TRAIN_START + 400 * DAY_MS)
        assert tuple(res.gate) == walkforward.GATE_CONDITIONS
