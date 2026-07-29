"""
Tests for feedback/protocol.py (v0.3.0 Phase 5, contract §4/§5/§8) and the
`review` CLI subcommand.

Every test in this file that reaches a forward test STUBS
protocol.walk_forward_pooled and/or protocol.run_graph_backtest — protocol.py
is the one module permitted to call the oracle (lock #3), and these tests
verify HOW it calls it (grid shape, n_trials source, call count) without
needing real OHLCV history or the real gate's runtime cost.
"""

import logging

import pytest

from trading_bot import config
from trading_bot.backtest import trials
from trading_bot.backtest.benchmark import BenchmarkResult
from trading_bot.backtest.engine import BacktestParams, Trade
from trading_bot.backtest.walkforward import GATE_CONDITIONS, WalkForwardResult
from trading_bot.data import statestore, storage
from trading_bot.feedback import protocol, records, versioning
from trading_bot.framework import registry
from trading_bot.plugins import build_v020_graph

SYMBOL = "BTCUSDT"
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
DAY_MS = protocol.DAY_MS
START = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _load_registry():
    registry.load_all()


@pytest.fixture
def state_conn():
    conn = statestore.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def ohlcv_conn():
    conn = storage.connect(":memory:")
    yield conn
    conn.close()


def _fake_bench():
    bundle = {
        "total_return": 1.0, "ann_return_pct": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "max_drawdown_pct": 0.0, "n_days": 5,
    }
    return BenchmarkResult(per_symbol={}, basket=bundle, start_ms=0, end_ms=1)


def _fake_oos_metrics():
    return {
        "n_trades": 0, "win_rate": None, "expectancy_pct": None,
        "avg_win_pct": None, "avg_loss_pct": None, "profit_factor": None,
        "max_drawdown_pct": None, "by_bucket": {},
    }


def _fake_oos_equity():
    return {
        "n_days": 5, "sharpe": None, "sortino": None, "max_drawdown_pct": None,
        "ann_return_pct": None, "daily_sharpe": None, "dsr": None,
        "skew": None, "kurtosis": None, "attribution": "spread",
    }


@pytest.fixture
def spy_walk_forward(monkeypatch):
    """Stub protocol.walk_forward_pooled AND protocol.run_graph_backtest
    (used to re-derive OOS trades) with a recording spy. Returns the list of
    captured call kwargs."""
    calls = []

    def fake_wf(conn, symbols, *, start_ms, end_ms, grid=None, oos_days=None,
                n_trials=None, strategy=None, **kw):
        calls.append(dict(
            conn=conn, symbols=symbols, start_ms=start_ms, end_ms=end_ms,
            grid=grid, oos_days=oos_days, n_trials=n_trials, strategy=strategy,
        ))
        return WalkForwardResult(
            folds=[], final_params=BacktestParams(), final_max_hold_bars=96,
            oos_start=end_ms - (oos_days or 0) * DAY_MS, oos_end=end_ms,
            oos_metrics=_fake_oos_metrics(), oos_equity=_fake_oos_equity(),
            per_symbol_expectancy={}, gate={c: True for c in GATE_CONDITIONS},
            benchmark=_fake_bench(), n_trials_used=n_trials, passed=True,
        )

    monkeypatch.setattr(protocol, "walk_forward_pooled", fake_wf)
    monkeypatch.setattr(protocol, "run_graph_backtest", lambda *a, **k: [])
    return calls


class TestForwardSpan:
    def test_inverted_span_raises(self):
        with pytest.raises(protocol.ForwardSpanError, match="forward_end_ms"):
            protocol.forward_span(
                history_start_ms=100, forward_start_ms=50, forward_end_ms=200 * DAY_MS,
            )

    def test_zero_span_raises(self):
        with pytest.raises(protocol.ForwardSpanError):
            protocol.forward_span(
                history_start_ms=0, forward_start_ms=100 * DAY_MS, forward_end_ms=100 * DAY_MS,
            )

    def test_below_min_days_raises(self):
        with pytest.raises(protocol.ForwardSpanError, match="REVIEW_FORWARD_MIN_DAYS"):
            protocol.forward_span(
                history_start_ms=0, forward_start_ms=100 * DAY_MS,
                forward_end_ms=100 * DAY_MS + 5 * DAY_MS,
            )

    def test_n_forward_days_round_trips_exactly(self):
        span = protocol.forward_span(
            history_start_ms=0, forward_start_ms=100 * DAY_MS,
            forward_end_ms=140 * DAY_MS,
        )
        assert span.n_forward_days == 40
        assert span.forward_start_ms == span.forward_end_ms - span.n_forward_days * DAY_MS

    def test_non_whole_day_span_raises(self):
        with pytest.raises(protocol.ForwardSpanError, match="whole number of days"):
            protocol.forward_span(
                history_start_ms=0, forward_start_ms=100 * DAY_MS,
                forward_end_ms=140 * DAY_MS + 3_600_000,
            )

    @pytest.mark.parametrize("holdout_start_days,holdout_end_days", [
        (100, 140),   # contained
        (90, 110),    # left overlap
        (130, 150),   # right overlap
        (0, 200),     # spans the whole thing (also catches history_start_ms)
    ])
    def test_locked_holdout_is_refused(self, monkeypatch, holdout_start_days, holdout_end_days):
        monkeypatch.setattr(config, "HOLDOUT_START", holdout_start_days * DAY_MS, raising=False)
        monkeypatch.setattr(config, "HOLDOUT_END", holdout_end_days * DAY_MS, raising=False)
        with pytest.raises(protocol.ForwardSpanError, match="LOCKED final holdout"):
            protocol.forward_span(
                history_start_ms=0, forward_start_ms=100 * DAY_MS, forward_end_ms=140 * DAY_MS,
            )

    def test_span_strictly_before_holdout_is_accepted(self, monkeypatch):
        monkeypatch.setattr(config, "HOLDOUT_START", 200 * DAY_MS, raising=False)
        monkeypatch.setattr(config, "HOLDOUT_END", 240 * DAY_MS, raising=False)
        span = protocol.forward_span(
            history_start_ms=0, forward_start_ms=100 * DAY_MS, forward_end_ms=140 * DAY_MS,
        )
        assert span.n_forward_days == 40

    def test_absent_holdout_constants_warn_but_do_not_block(self, monkeypatch, caplog):
        monkeypatch.delattr(config, "HOLDOUT_START", raising=False)
        monkeypatch.delattr(config, "HOLDOUT_END", raising=False)
        with caplog.at_level(logging.WARNING, logger="trading_bot"):
            span = protocol.forward_span(
                history_start_ms=0, forward_start_ms=100 * DAY_MS, forward_end_ms=140 * DAY_MS,
            )
        assert span.n_forward_days == 40
        assert any("INERT" in r.message for r in caplog.records)


class TestEvaluateForward:
    def test_calls_walk_forward_pooled_exactly_once_with_frozen_grid(self, state_conn, spy_walk_forward):
        g = build_v020_graph()
        v = versioning.register_version(state_conn, g)
        span = protocol.forward_span(
            history_start_ms=0, forward_start_ms=100 * DAY_MS, forward_end_ms=130 * DAY_MS,
        )
        result = protocol.evaluate_forward(
            None, state_conn, version_id=v.version_id, symbols=[SYMBOL], span=span,
        )
        assert len(spy_walk_forward) == 1
        call = spy_walk_forward[0]
        assert call["strategy"] is not None
        assert all(len(vals) == 1 for vals in call["grid"].values())
        assert call["oos_days"] == 30
        # n_trials is the ledger's cumulative count, never len(combos)*len(folds).
        assert call["n_trials"] == 1
        assert result.n_trials_used == 1
        assert result.passed is True

    def test_ledger_gets_exactly_one_row_per_call(self, state_conn, spy_walk_forward):
        g = build_v020_graph()
        v = versioning.register_version(state_conn, g)
        span = protocol.forward_span(
            history_start_ms=0, forward_start_ms=100 * DAY_MS, forward_end_ms=130 * DAY_MS,
        )
        protocol.evaluate_forward(None, state_conn, version_id=v.version_id, symbols=[SYMBOL], span=span)
        protocol.evaluate_forward(None, state_conn, version_id=v.version_id, symbols=[SYMBOL], span=span)
        ledger = trials.TrialLedger(state_conn, f"review:{v.version_id}")
        assert ledger.count() == 2  # one per evaluate_forward call, not per fold

    def test_short_history_raises_forward_span_error(self, state_conn, monkeypatch):
        g = build_v020_graph()
        v = versioning.register_version(state_conn, g)

        def raise_short(*a, **k):
            raise ValueError("span too short: need at least one train+test fold before the OOS holdout")

        monkeypatch.setattr(protocol, "walk_forward_pooled", raise_short)
        span = protocol.forward_span(
            history_start_ms=0, forward_start_ms=100 * DAY_MS, forward_end_ms=130 * DAY_MS,
        )
        with pytest.raises(protocol.ForwardSpanError, match="WF_TRAIN_DAYS \\+ config.WF_TEST_DAYS"):
            protocol.evaluate_forward(None, state_conn, version_id=v.version_id, symbols=[SYMBOL], span=span)

    def test_config_drift_is_reported_not_fatal(self, state_conn, spy_walk_forward, monkeypatch):
        g = build_v020_graph()
        v = versioning.register_version(state_conn, g)
        monkeypatch.setattr(config, "FEE_PCT", 0.001)
        span = protocol.forward_span(
            history_start_ms=0, forward_start_ms=100 * DAY_MS, forward_end_ms=130 * DAY_MS,
        )
        result = protocol.evaluate_forward(
            None, state_conn, version_id=v.version_id, symbols=[SYMBOL], span=span,
        )
        assert result.config_drift["config_hash_matches"] is False  # reported
        assert result.passed is True  # and NOT fatal


class TestApplySuggestion:
    def test_step_stays_inside_bounds(self):
        g = build_v020_graph()
        before = protocol.fgraph.graph_hash(g)
        edited = protocol.apply_suggestion(g, "widen-stop")
        assert edited is not None
        assert protocol.fgraph.graph_hash(edited) != before
        for b in edited.branches:
            spec = registry.get(b.policy.key)
            if "atr_multiple" in spec.params:
                resolved = spec.resolve(b.policy.params)
                lo, hi = spec.params["atr_multiple"].bounds
                assert lo <= resolved["atr_multiple"] <= hi

    def test_param_already_at_bound_gives_none(self):
        g = build_v020_graph()
        spec = registry.get("policy.atr-stop-measured-move")
        lo, hi = spec.params["atr_multiple"].bounds
        pinned_branches = tuple(
            protocol.replace(
                b, policy=protocol.replace(b.policy, params={**b.policy.params, "atr_multiple": hi})
            )
            if b.policy.key == "policy.atr-stop-measured-move" else b
            for b in g.branches
        )
        g_at_bound = protocol.replace(g, branches=pinned_branches)
        result = protocol.apply_suggestion(g_at_bound, "widen-stop")
        assert result is None

    def test_unknown_suggestion_gives_none_and_warns(self, caplog):
        g = build_v020_graph()
        with caplog.at_level(logging.WARNING, logger="trading_bot"):
            result = protocol.apply_suggestion(g, "widen-target")
        assert result is None
        assert any("no backing ParamSpec" in r.message for r in caplog.records)

    def test_int_kind_param_stays_integral(self):
        g = build_v020_graph()
        edited = protocol._step_param(g, "atr_period", 1)
        assert edited is not None
        for b in edited.branches:
            if "atr_period" in registry.get(b.policy.key).params:
                v = registry.get(b.policy.key).resolve(b.policy.params)["atr_period"]
                assert isinstance(v, int)

    def test_drop_confirmation_on_single_confirmation_branch_gives_none(self):
        from trading_bot.framework.graph import Branch, NodeSpec, StrategyGraph

        g = build_v020_graph(include_fade=False)
        branch = g.branches[0]
        with_one_conf = protocol.replace(
            branch,
            confirmations=(NodeSpec(id="c1", key="confirmation.macd"),),
        )
        g1 = protocol.replace(g, branches=(with_one_conf,))
        result = protocol.apply_suggestion(g1, "drop-confirmation:confirmation.macd")
        assert result is None

    def test_input_graph_is_unchanged(self):
        g = build_v020_graph()
        before = protocol.fgraph.graph_hash(g)
        protocol.apply_suggestion(g, "widen-stop")
        assert protocol.fgraph.graph_hash(g) == before  # frozen; never mutated


class TestFullLoop:
    """PRD Phase 5 success signal: trade -> close -> review -> refine ->
    version -> forward-test with no manual glue.

    protocol.run_graph_backtest is stubbed (not walk_forward_pooled's callee,
    protocol's OWN direct call) so this test needs no real multi-year OHLCV
    history — it supplies a canned trade list and lets the REAL review,
    diagnose, apply_suggestion, register_version and evaluate_forward-ledger
    code run unstubbed. walk_forward_pooled itself IS stubbed (that is the
    oracle; contract §4 forbids a review test from being the thing that
    exercises it end to end).
    """

    N_TRADES = 40

    @pytest.fixture
    def seeded_ohlcv(self, ohlcv_conn):
        # A long, gently-adverse-biased 1h series so every canned trade's
        # excursion window has real bars: deep low wick (big MAE), shallow
        # high wick (small MFE) -> "stop"-dominant, "target-too-far" review.
        n_bars = self.N_TRADES * 4 + 10
        rows = [
            [START + i * D_TRIG, 100.0, 100.2, 95.0, 99.8, 1.0]
            for i in range(n_bars)
        ]
        storage.upsert_candles(ohlcv_conn, SYMBOL, TRIGGER_TF, rows)
        return ohlcv_conn

    @pytest.fixture
    def canned_trades(self):
        trades = []
        for i in range(self.N_TRADES):
            entry_ts = START + i * 2 * D_TRIG
            trades.append(Trade(
                symbol=SYMBOL, regime="trending", pattern="donchian-breakout",
                direction="long", entry_ts=entry_ts, entry=100.0, stop=99.5,
                target=110.0, exit_ts=entry_ts + D_TRIG, exit_price=99.5,
                outcome="stop", pnl_pct=-0.005, volume_high=False,
            ))
        return trades

    def test_one_iteration_executes_without_manual_glue(
        self, seeded_ohlcv, state_conn, canned_trades, spy_walk_forward, monkeypatch,
    ):
        monkeypatch.setattr(protocol, "run_graph_backtest", lambda *a, **k: canned_trades)

        seed_graph = build_v020_graph()
        seed_version = versioning.register_version(state_conn, seed_graph)

        review_start_ms = START
        review_end_ms = START + self.N_TRADES * 2 * D_TRIG
        forward = protocol.forward_span(
            history_start_ms=review_start_ms, forward_start_ms=review_end_ms,
            forward_end_ms=review_end_ms + 30 * DAY_MS,
        )

        result = protocol.run_loop_iteration(
            seeded_ohlcv, state_conn, version_id=seed_version.version_id,
            symbols=[SYMBOL], review_start_ms=review_start_ms,
            review_end_ms=review_end_ms, span_class="in-sample", forward=forward,
        )

        # Parent records, span_class="in-sample".
        parent_records = records.load_records(
            state_conn, strategy_version=seed_version.version_id, span_class="in-sample",
        )
        assert len(parent_records) == self.N_TRADES
        assert result.records_written == self.N_TRADES

        d = result.diagnosis
        assert all(s.split(":", 1)[0] in records.SUGGESTIONS for s in d.suggestions)

        assert result.child_version_id is not None
        child = versioning.get_version(state_conn, result.child_version_id)
        assert child.parent_id == seed_version.version_id
        assert child.provenance["diagnosis_digest"] == d.digest

        chain = versioning.lineage(state_conn, result.child_version_id)
        assert len(chain) == 2
        assert chain[0].version_id == seed_version.version_id

        assert result.forward is not None
        assert set(result.forward.gate) == set(GATE_CONDITIONS)

        forward_records = records.load_records(
            state_conn, strategy_version=result.child_version_id, span_class="forward",
        )
        assert len(forward_records) == len(result.forward.trades)

        ledger = trials.TrialLedger(state_conn, f"review:{result.child_version_id}")
        assert ledger.count() == 1  # exactly one evaluation, from evaluate_forward alone

    def test_no_suggestion_yields_no_child_and_says_so(self, seeded_ohlcv, state_conn, monkeypatch):
        # A tiny sample (< REVIEW_PACE_MIN_TRADES) yields ONLY
        # "insufficient-sample", which apply_suggestion cannot act on.
        few_trades = [
            Trade(
                symbol=SYMBOL, regime="trending", pattern="donchian-breakout",
                direction="long", entry_ts=START + i * 2 * D_TRIG, entry=100.0,
                stop=99.5, target=110.0, exit_ts=START + i * 2 * D_TRIG + D_TRIG,
                exit_price=99.5, outcome="stop", pnl_pct=-0.005, volume_high=False,
            )
            for i in range(5)
        ]
        monkeypatch.setattr(protocol, "run_graph_backtest", lambda *a, **k: few_trades)
        seed_graph = build_v020_graph()
        seed_version = versioning.register_version(state_conn, seed_graph)
        result = protocol.run_loop_iteration(
            seeded_ohlcv, state_conn, version_id=seed_version.version_id,
            symbols=[SYMBOL], review_start_ms=START, review_end_ms=START + 20 * D_TRIG,
        )
        assert result.child_version_id is None
        assert result.notes != ""


class TestReviewCli:
    def test_register_prints_version_id(self, tmp_path, state_conn, ohlcv_conn):
        from trading_bot.cli import _review_command

        g = build_v020_graph(name="cli-test-graph")
        path = tmp_path / "g.strategy.json"
        import json
        path.write_text(json.dumps(g.to_dict()))

        code = _review_command(
            ohlcv_conn, state_conn, [SYMBOL], strategy_path=str(path), version_id=None,
            register=True, start_ms=0, end_ms=1, diagnose_only=False, loop=False,
            forward=None, reviewer="trade-quality", persist=True,
        )
        assert code == 0

    def test_unknown_version_prints_error_and_returns_1(self, state_conn, ohlcv_conn, capsys):
        from trading_bot.cli import _review_command

        code = _review_command(
            ohlcv_conn, state_conn, [SYMBOL], strategy_path=None, version_id="nope",
            register=False, start_ms=0, end_ms=1, diagnose_only=True, loop=False,
            forward=None, reviewer="trade-quality", persist=True,
        )
        assert code == 1
        assert "ERROR" in capsys.readouterr().out

    def test_diagnose_only_with_no_stored_records_returns_1(self, state_conn, ohlcv_conn, tmp_path, capsys):
        from trading_bot.cli import _review_command
        import json

        g = build_v020_graph(name="cli-test-graph-2")
        path = tmp_path / "g2.strategy.json"
        path.write_text(json.dumps(g.to_dict()))
        code = _review_command(
            ohlcv_conn, state_conn, [SYMBOL], strategy_path=str(path), version_id=None,
            register=False, start_ms=0, end_ms=1, diagnose_only=True, loop=False,
            forward=None, reviewer="trade-quality", persist=True,
        )
        assert code == 1
        assert "no stored review_records" in capsys.readouterr().out
