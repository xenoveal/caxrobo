"""
Tests for the v0.3.0 Phase 9 walk-forward campaign (src/trading_bot/campaign.py).

GOTCHA that governs this whole file: no test may touch the real
data/state.db. A test that writes a holdout_consumption row into the real
state database would burn the project's one-shot holdout — a catastrophic false
positive. Every state connection here is sqlite3.connect(":memory:").
"""

import json
import sqlite3

import pytest

from trading_bot import campaign, config
from trading_bot.backtest import trials, walkforward
from trading_bot.backtest.engine import clear_caches
from trading_bot.backtest.walkforward import DAY_MS, GATE_CONDITIONS
from trading_bot.cli import _campaign_command
from trading_bot.evolution import population as evo_population

# Tier-derived, never hardcoded: these follow config forever, so a future tier
# shift cannot leave the tests on the old timeframes while production moves.
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME

H_START = config.HOLDOUT_START_MS
H_END = config.HOLDOUT_END_MS


@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    """Clear the engine's indicator memo around every test."""
    clear_caches()
    yield
    clear_caches()


@pytest.fixture
def state():
    """An in-memory state.db with every schema this module reads."""
    conn = sqlite3.connect(":memory:")
    trials.ensure_schema(conn)
    evo_population.ensure_schema(conn)
    campaign._ensure_schema(conn)
    yield conn
    conn.close()


_SEED_GRAPH_JSON = json.dumps(
    {"schema_version": 1, "name": "stub", "data": {"id": "d", "key": "data.ohlcv"}}
)


def insert_campaign_row(conn, campaign_id="c1", *, train_end_ms=None, seed=1):
    train_end_ms = H_START if train_end_ms is None else train_end_ms
    conn.execute(
        "INSERT INTO campaigns (campaign_id, seed, seed_graph_hash, "
        "seed_graph_json, config_json, symbols_json, train_start_ms, "
        "train_end_ms, audit_start_ms, audit_end_ms, population, generations, "
        "started_ts, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (campaign_id, seed, "seedhash", _SEED_GRAPH_JSON, "{}",
         json.dumps(["BTCUSDT"]), config.CAMPAIGN_EVOLVE_START_MS, train_end_ms,
         config.CAMPAIGN_EVOLVE_START_MS, train_end_ms, 6, 2, 1, "done"),
    )
    conn.commit()


def insert_member(conn, member_id, *, campaign_id="c1", gen=0, graph_hash="aaa",
                  fitness=1.0, n_trades=40, role="offspring", tier="B"):
    conn.execute(
        "INSERT INTO population_members (member_id, campaign_id, gen_index, "
        "member_index, graph_hash, graph_json, rng_seed, role, fitness, tier, "
        "n_trades) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (member_id, campaign_id, gen, 0, graph_hash, _SEED_GRAPH_JSON, 1, role,
         fitness, tier, n_trades),
    )
    conn.commit()


class TestSpans:
    def test_config_holdout_arithmetic_is_consistent(self):
        assert H_END - config.HOLDOUT_DAYS * DAY_MS == H_START

    def test_holdout_span_rejects_inconsistent_days(self, monkeypatch):
        monkeypatch.setattr(config, "HOLDOUT_DAYS", config.HOLDOUT_DAYS + 1)
        with pytest.raises(campaign.CampaignError):
            campaign.holdout_span()

    def test_evolution_span_ends_exactly_at_holdout_start(self):
        start, end = campaign.evolution_span()
        assert end == H_START
        assert start == config.CAMPAIGN_EVOLVE_START_MS

    def test_evolution_span_matches_phase6_frozen_ceiling(self):
        """The barrier is Phase 6's own ceiling, not a Phase 9 invention.

        If these ever diverge, evolution's runner and the campaign disagree
        about which bars are spendable — the lookahead bug this pins.
        """
        assert config.date_to_ms(config.EVO_TRAIN_END) == H_START

    def test_overlap_passes_for_the_evolution_span(self):
        campaign.assert_no_holdout_overlap(
            config.CAMPAIGN_EVOLVE_START_MS, H_START, what="evolution"
        )

    def test_overlap_raises_on_the_straddle_case(self):
        with pytest.raises(campaign.HoldoutViolation):
            campaign.assert_no_holdout_overlap(H_START - 1, H_START + 1, what="x")

    def test_overlap_raises_strictly_inside_the_holdout(self):
        with pytest.raises(campaign.HoldoutViolation):
            campaign.assert_no_holdout_overlap(
                H_START + DAY_MS, H_START + 2 * DAY_MS, what="x"
            )

    def test_overlap_passes_for_the_empty_span_at_the_boundary(self):
        campaign.assert_no_holdout_overlap(H_START, H_START, what="x")

    def test_fold_count_is_twelve_on_the_declared_spans(self):
        assert campaign.fold_count() == 12


class TestHoldoutLedger:
    def test_first_open_is_run_index_one_and_marks_consumed(self, state):
        assert campaign.holdout_is_consumed(state) is False
        _, idx = campaign.open_holdout(
            state, campaign_id="c", graph_hash="g", strategy_version="v",
            n_trials=10,
        )
        assert idx == 1
        assert campaign.holdout_is_consumed(state) is True

    def test_second_open_without_a_reason_raises(self, state):
        campaign.open_holdout(state, campaign_id="c", graph_hash="g",
                             strategy_version="v", n_trials=10)
        with pytest.raises(campaign.CampaignError, match="already consumed"):
            campaign.open_holdout(state, campaign_id="c2", graph_hash="g",
                                 strategy_version="v", n_trials=10)

    def test_second_open_with_a_reason_records_run_index_two(self, state):
        campaign.open_holdout(state, campaign_id="c", graph_hash="g",
                             strategy_version="v", n_trials=10)
        _, idx = campaign.open_holdout(
            state, campaign_id="c", graph_hash="g", strategy_version="v",
            n_trials=10, override_reason="auditor asked",
        )
        assert idx == 2
        rows = campaign.holdout_consumption_rows(state)
        assert rows[-1]["override_reason"] == "auditor asked"

    def test_violation_row_does_not_mark_consumed(self, state):
        campaign.record_violation(state, campaign_id="c", start_ms=0, end_ms=1,
                                 detail="tried")
        assert campaign.holdout_is_consumed(state) is False
        assert len(campaign.holdout_consumption_rows(state, kind="violation")) == 1

    def test_row_survives_a_raising_gate_with_completed_ts_null(self, state):
        """Prevents the `kill -9` free peek: write-then-run ordering."""
        campaign.open_holdout(state, campaign_id="c", graph_hash="g",
                             strategy_version="v", n_trials=10)
        rows = campaign.holdout_consumption_rows(state)
        assert len(rows) == 1
        assert rows[0]["completed_ts"] is None
        assert campaign.holdout_is_consumed(state) is True

    def test_holdout_locked_false_refuses(self, state, monkeypatch):
        monkeypatch.setattr(config, "HOLDOUT_LOCKED", False)
        with pytest.raises(campaign.CampaignError, match="HOLDOUT_LOCKED"):
            campaign.open_holdout(state, campaign_id="c", graph_hash="g",
                                 strategy_version="v", n_trials=10)

    def test_close_holdout_attaches_gate_and_outcome(self, state):
        row_id, _ = campaign.open_holdout(
            state, campaign_id="c", graph_hash="g", strategy_version="v",
            n_trials=10,
        )
        campaign.close_holdout(state, row_id, gate={"dsr": False},
                               outcome={"verdict": "B1", "passed": False})
        payload = campaign.latest_outcome(state)
        assert payload["verdict"] == "B1"
        assert payload["run_index"] == 1

    def test_latest_outcome_is_none_before_any_run(self, state):
        assert campaign.latest_outcome(state) is None


class TestTrialAccounting:
    def _row(self, conn, campaign_name, end_ms):
        conn.execute(
            f"INSERT INTO {trials.LEDGER_TABLE} (campaign, graph_hash, "
            "params_hash, start_ms, end_ms, ts) VALUES (?,?,?,?,?,?)",
            (campaign_name, "g", "p", 0, end_ms, 1),
        )
        conn.commit()

    def test_counts_every_pre_barrier_row_across_campaigns(self, state):
        for _ in range(5):
            self._row(state, "camp-a", H_START)
        for _ in range(3):
            self._row(state, "camp-b", H_START - DAY_MS)
        assert campaign.cumulative_trials(state) == 8

    def test_excludes_rows_past_the_barrier(self, state):
        self._row(state, "camp-a", H_START)
        self._row(state, "leaky", H_START + 1)
        assert campaign.cumulative_trials(state) == 1

    def test_breakdown_flags_a_past_barrier_campaign(self, state):
        self._row(state, "leaky", H_START + 1)
        rows = campaign.ledger_breakdown(state)
        assert rows[0]["campaign"] == "leaky"
        assert rows[0]["past_barrier"] is True

    def test_predicted_gate_trials_is_two_per_fold_on_a_degenerate_grid(self):
        assert campaign.predicted_gate_trials(campaign.frozen_grid(), 12) == 24

    def test_frozen_grid_has_one_value_per_run_level_axis(self):
        grid = campaign.frozen_grid()
        assert set(grid) <= walkforward._RUN_LEVEL_AXES
        assert all(len(v) == 1 for v in grid.values())
        assert walkforward._neighbors(grid, {"max_hold_bars": grid["max_hold_bars"][0]}) == []

    def test_median_low_over_the_degenerate_grid_returns_that_value(self):
        import statistics

        grid = campaign.frozen_grid()
        for axis, values in grid.items():
            assert statistics.median_low([values[0]] * 12) == values[0]
            assert values[0] in grid[axis]

    def test_n_trials_reaching_the_gate_is_the_cumulative_count(self, state,
                                                               monkeypatch):
        """REGRESSION: the in-process default (len(combos)*len(folds) == 12)
        would flatter the DSR by ~5 annualised Sharpe points."""
        for _ in range(200):
            self._row(state, "camp-a", H_START)
        insert_campaign_row(state)
        insert_member(state, "m1")
        captured = {}

        def fake_wf(conn, symbols, **kw):
            captured.update(kw)
            raise RuntimeError("stop after capture")

        monkeypatch.setattr(campaign, "walk_forward_pooled", fake_wf)
        monkeypatch.setattr(campaign, "champion_graph", lambda ch: object())
        monkeypatch.setattr(
            campaign, "sample_adequacy_projection",
            lambda *a, **k: _fake_projection(),
        )
        with pytest.raises(RuntimeError, match="stop after capture"):
            campaign.run_holdout_gate(None, state, campaign_id="p9")
        assert captured["n_trials"] == 200 + 24
        assert captured["oos_days"] == config.HOLDOUT_DAYS
        assert "min_trades" not in captured

    def test_gate_run_is_handed_a_ledger_so_its_own_rows_persist(self, state,
                                                                monkeypatch):
        """REGRESSION: charging n_trials for the gate's own evaluations while
        NOT persisting them makes the ledger a lie about its own last step."""
        insert_campaign_row(state)
        insert_member(state, "m1")
        captured = {}

        def fake_wf(conn, symbols, **kw):
            captured.update(kw)
            raise RuntimeError("stop after capture")

        monkeypatch.setattr(campaign, "walk_forward_pooled", fake_wf)
        monkeypatch.setattr(campaign, "champion_graph", lambda ch: object())
        monkeypatch.setattr(
            campaign, "sample_adequacy_projection",
            lambda *a, **k: _fake_projection(),
        )
        with pytest.raises(RuntimeError):
            campaign.run_holdout_gate(None, state, campaign_id="p9")
        ledger = captured["ledger"]
        assert ledger is not None
        assert callable(ledger.record) and callable(ledger.count)


class TestChampionSelection:
    def test_highest_fitness_wins(self, state):
        insert_campaign_row(state)
        insert_member(state, "lo", fitness=1.0, graph_hash="b")
        insert_member(state, "hi", fitness=2.0, graph_hash="a")
        assert campaign.select_champion(state).member_id == "hi"

    def test_tie_breaks_on_earliest_generation(self, state):
        insert_campaign_row(state)
        insert_member(state, "late", fitness=2.0, gen=7, graph_hash="a")
        insert_member(state, "early", fitness=2.0, gen=4, graph_hash="b")
        assert campaign.select_champion(state).member_id == "early"

    def test_tie_breaks_on_smallest_graph_hash(self, state):
        insert_campaign_row(state)
        insert_member(state, "zz", fitness=2.0, gen=4, graph_hash="zzz")
        insert_member(state, "aa", fitness=2.0, gen=4, graph_hash="aaa")
        assert campaign.select_champion(state).member_id == "aa"

    def test_trade_floor_excludes_a_degenerate_low_trade_winner(self, state):
        """The pre-registered floor, and the exact pathology it exists for:
        Phase 6 measured a tier-B generation-7 winner with ONE trade at Sharpe
        4.608."""
        insert_campaign_row(state)
        insert_member(state, "degenerate", fitness=7.67, n_trades=1,
                      graph_hash="a")
        insert_member(state, "real", fitness=3.43, n_trades=34, graph_hash="b")
        champ = campaign.select_champion(state)
        assert champ.member_id == "real"
        assert champ.below_trade_floor is False
        assert champ.n_eligible == 1
        assert champ.n_scored == 2

    def test_floor_is_the_gates_own_sample_floor(self):
        assert config.CAMPAIGN_MIN_CHAMPION_TRADES == config.WF_MIN_TRADES

    def test_floor_unmet_falls_back_and_is_flagged_never_lowered(self, state):
        insert_campaign_row(state)
        insert_member(state, "a", fitness=7.0, n_trades=12, graph_hash="a")
        insert_member(state, "b", fitness=1.0, n_trades=20, graph_hash="b")
        champ = campaign.select_champion(state)
        assert champ.member_id == "a"
        assert champ.below_trade_floor is True
        assert champ.n_eligible == 0
        # The floor itself is untouched by the fallback.
        assert config.CAMPAIGN_MIN_CHAMPION_TRADES == 30

    def test_finalist_rescores_are_excluded_from_selection(self, state):
        insert_campaign_row(state)
        insert_member(state, "offspring", fitness=3.0, graph_hash="a")
        insert_member(state, "audit", fitness=9.0, graph_hash="a",
                      role="finalist", gen=-1)
        assert campaign.select_champion(state).member_id == "offspring"

    def test_empty_population_raises_never_falls_back_to_seed(self, state):
        insert_campaign_row(state)
        with pytest.raises(campaign.CampaignError, match="no scored members"):
            campaign.select_champion(state)

    def test_no_campaign_raises(self, state):
        with pytest.raises(campaign.CampaignError, match="no evolution campaign"):
            campaign.select_champion(state)

    def test_a_campaign_past_the_barrier_is_refused_as_a_champion_source(self, state):
        insert_campaign_row(state, "leaky", train_end_ms=H_START + DAY_MS)
        insert_member(state, "m", campaign_id="leaky")
        with pytest.raises(campaign.HoldoutViolation, match="past the holdout"):
            campaign.select_champion(state)


def _fake_projection(rate=0.085, eff_n=1.834, n_symbols=9):
    k = rate * config.HOLDOUT_DAYS
    return campaign.SampleProjection(
        n_symbols=n_symbols, evolution_days=914,
        measured_trades_per_symbol_day=rate,
        measured_trades_total=int(rate * n_symbols * 914),
        holdout_days=config.HOLDOUT_DAYS, projected_raw_trades=k * n_symbols,
        mean_pairwise_correlation=0.4885, effective_n=eff_n,
        projected_independent_trades=k * eff_n,
        required_rate_for_independent_floor=(
            config.WF_MIN_TRADES / eff_n / config.HOLDOUT_DAYS
        ),
        raw_floor=config.WF_MIN_TRADES,
        raw_floor_reachable=k * n_symbols >= config.WF_MIN_TRADES,
        independent_floor_reachable=k * eff_n >= config.WF_MIN_TRADES,
    )


class TestSampleProjection:
    def test_analytic_effective_n_reproduces_the_published_anchor(self):
        """n=3, r_bar=0.7574 -> 1.193 (KNOWN-LIMITATIONS §0b AND Phase 2)."""
        n, r = 3, 0.7574
        assert round(n / (1 + (n - 1) * r), 3) == 1.193

    def test_independent_equivalent_is_k_times_eff_n(self):
        p = _fake_projection(rate=0.085, eff_n=1.834)
        assert round(p.projected_independent_trades, 1) == 28.4
        assert p.independent_floor_reachable is False

    def test_independent_equivalent_is_invariant_to_symbol_count(self):
        """"rows, not information": only the raw count moves with symbols."""
        p8 = _fake_projection(n_symbols=8)
        p9 = _fake_projection(n_symbols=9)
        assert p8.projected_independent_trades == p9.projected_independent_trades
        assert p8.projected_raw_trades != p9.projected_raw_trades

    def test_break_even_rate_identity(self):
        p = _fake_projection()
        assert round(p.required_rate_for_independent_floor, 4) == round(
            config.WF_MIN_TRADES / 1.834 / config.HOLDOUT_DAYS, 4
        )
        assert round(p.required_rate_for_independent_floor, 4) == 0.0899

    def test_pessimistic_rate_does_not_clear_the_independent_floor(self):
        p = _fake_projection(rate=0.050)
        assert p.independent_floor_reachable is False
        assert round(p.projected_independent_trades, 1) == 16.7

    def test_raw_floor_compares_against_unmodified_wf_min_trades(self):
        assert config.WF_MIN_TRADES == 30
        assert _fake_projection().raw_floor == 30

    def test_campaign_module_never_overrides_the_floor_or_the_window(self):
        """REGRESSION (KNOWN-LIMITATIONS §4): neither WF_MIN_TRADES nor the
        holdout length may be written by this phase."""
        from pathlib import Path

        src = Path(campaign.__file__).read_text()
        assert "min_trades=" not in src
        assert "WF_MIN_TRADES =" not in src
        assert src.count("oos_days=") == 1
        assert "oos_days=config.HOLDOUT_DAYS" in src


class TestVerdictClassifier:
    def _gate(self, **overrides):
        gate = dict.fromkeys(GATE_CONDITIONS, True)
        gate.update(overrides)
        return gate

    def test_all_pass_above_northstar_is_A(self):
        assert campaign.classify_verdict(
            self._gate(), {"ann_return_pct": 0.9}
        ) == "A"

    def test_all_pass_below_northstar_is_A_prime(self):
        assert campaign.classify_verdict(
            self._gate(), {"ann_return_pct": 0.1}
        ) == "A_PRIME"

    def test_only_dsr_fails_is_B1(self):
        assert campaign.classify_verdict(
            self._gate(dsr=False), {"ann_return_pct": 0.2}
        ) == "B1"

    def test_dsr_plus_another_is_B2(self):
        assert campaign.classify_verdict(
            self._gate(dsr=False, per_symbol_expectancy=False),
            {"ann_return_pct": 0.2},
        ) == "B2"

    def test_benchmark_failure_takes_precedence_over_B1_and_B4(self):
        assert campaign.classify_verdict(
            self._gate(dsr=False, sample_adequacy=False,
                       beats_benchmark_sharpe=False),
            {"ann_return_pct": 0.2},
        ) == "B3"

    def test_sample_adequacy_failure_is_B4(self):
        assert campaign.classify_verdict(
            self._gate(dsr=False, sample_adequacy=False), {"ann_return_pct": 0.2}
        ) == "B4"

    def test_missing_condition_is_C(self):
        gate = self._gate()
        gate.pop("dsr")
        assert campaign.classify_verdict(gate, {"ann_return_pct": 0.2}) == "C"

    def test_exactly_one_id_over_the_whole_gate_space(self):
        import itertools

        for bits in itertools.product((True, False), repeat=len(GATE_CONDITIONS)):
            gate = dict(zip(GATE_CONDITIONS, bits))
            verdict = campaign.classify_verdict(gate, {"ann_return_pct": 0.2})
            assert verdict in campaign.VERDICTS
            if all(bits):
                assert verdict == "A_PRIME"
            else:
                assert verdict != "A" and verdict != "A_PRIME"

    def test_benchmark_context_negative_on_phase1s_measured_case(self):
        """Phase 1 measured the v0.2.0 OOS basket at Sharpe -1.303."""
        ctx = campaign.benchmark_context(
            _FakeBenchmark({"ann_return_pct": -0.5, "sharpe": -1.303})
        )
        assert ctx == "BENCHMARK_NEGATIVE"

    def test_benchmark_context_positive(self):
        assert campaign.benchmark_context(
            _FakeBenchmark({"ann_return_pct": 0.29, "sharpe": 0.73})
        ) == "BENCHMARK_POSITIVE"

    def test_benchmark_context_needs_both_signs(self):
        assert campaign.benchmark_context(
            _FakeBenchmark({"ann_return_pct": 0.2, "sharpe": -0.1})
        ) == "BENCHMARK_NEGATIVE"
        assert campaign.benchmark_context(
            _FakeBenchmark({"ann_return_pct": -0.2, "sharpe": 0.1})
        ) == "BENCHMARK_NEGATIVE"

    def test_context_is_not_a_gate_condition(self):
        """REGRESSION: a BENCHMARK_NEGATIVE all-pass run still yields A/A' —
        the gate was not moved."""
        assert "benchmark_context" not in GATE_CONDITIONS
        assert len(GATE_CONDITIONS) == 7
        assert campaign.classify_verdict(
            self._gate(), {"ann_return_pct": 0.9}
        ) == "A"

    def test_required_annual_sharpe_grows_with_trials(self):
        low = campaign.required_annual_sharpe(1, config.HOLDOUT_DAYS)
        high = campaign.required_annual_sharpe(417, config.HOLDOUT_DAYS)
        assert 2.0 < low < 3.0
        assert high > low
        assert 6.0 < high < 8.0

    def test_northstar_is_reported_not_gated(self):
        ann, met = campaign.northstar_gap({"ann_return_pct": 0.6})
        assert ann == 0.6 and met is True
        assert campaign.northstar_gap({"ann_return_pct": None}) == (None, False)


class _FakeBenchmark:
    def __init__(self, basket, per_symbol=None):
        self.basket = basket
        self.per_symbol = per_symbol or {}
        self.start_ms = H_START
        self.end_ms = H_END


class _FakeFold:
    def __init__(self, exp, train_exp=0.01):
        self.test_metrics = {"expectancy_pct": exp}
        self.train_expectancy = train_exp


class _FakeResult:
    def __init__(self, *, gate=None, passed=False, n_trades=140, sharpe=1.5,
                 dsr=0.009, ann=0.18, dd=0.21):
        self.gate = gate or dict.fromkeys(GATE_CONDITIONS, True) | {"dsr": False}
        self.passed = passed
        self.oos_start, self.oos_end = H_START, H_END
        self.oos_metrics = {"n_trades": n_trades, "win_rate": 0.5,
                            "expectancy_pct": 0.004, "profit_factor": 1.2,
                            "max_drawdown_pct": dd}
        self.oos_equity = {
            "n_days": config.HOLDOUT_DAYS, "sharpe": sharpe, "sortino": 1.2,
            "max_drawdown_pct": dd, "ann_return_pct": ann,
            "daily_sharpe": sharpe / 19.1, "dsr": dsr, "skew": 1.4,
            "kurtosis": 15.5, "attribution": config.PNL_ATTRIBUTION_MODE,
        }
        self.per_symbol_expectancy = {s: 0.003 for s in config.CAMPAIGN_SYMBOLS}
        self.benchmark = _FakeBenchmark(
            {"total_return": 0.8, "ann_return_pct": -0.3, "sharpe": -1.1,
             "max_drawdown_pct": 0.4, "n_days": config.HOLDOUT_DAYS},
            {s: {"total_return": 0.8, "ann_return_pct": -0.3, "sharpe": -1.1,
                 "max_drawdown_pct": 0.4, "n_days": config.HOLDOUT_DAYS}
             for s in config.CAMPAIGN_SYMBOLS},
        )
        self.n_trials_used = 441
        self.folds = [_FakeFold(0.01), _FakeFold(-0.02), _FakeFold(0.0, None)]
        self.final_params = None
        self.final_max_hold_bars = config.MAX_HOLD_BARS_TRIGGER


def _fake_outcome(result=None, *, verdict="B1", run_index=1):
    result = result or _FakeResult()
    champ = campaign.ChampionRef(
        campaign_id="c1", member_id="m1", graph_hash="deadbeefcafe",
        generation=6, fitness=3.4296, tier="B", n_trades=34,
        graph_json=_SEED_GRAPH_JSON, below_trade_floor=False, n_eligible=10,
        n_scored=194,
    )
    return campaign.HoldoutOutcome(
        campaign_id="p9", champion=champ, run_index=run_index,
        n_trials_charged=441, n_folds=12, result=result, gate=dict(result.gate),
        benchmark=result.benchmark, diagnostics={},
        projection=_fake_projection(), verdict=verdict,
        benchmark_context=campaign.benchmark_context(result.benchmark),
        stop_reason="COMPLETE", wall_seconds=1.0, ledger_breakdown=[],
    )


class TestDiagnostics:
    def test_collect_refuses_before_the_holdout_is_opened(self, state):
        with pytest.raises(campaign.CampaignError, match="before open_holdout"):
            campaign.collect_diagnostics(
                None, state, _FakeResult(), None, None, _fake_projection()
            )

    def test_dsr_at_n_trials_1_exceeds_dsr_at_the_cumulative_count(self):
        from trading_bot.backtest.equity import deflated_sharpe

        args = (config.HOLDOUT_DAYS, 1.4034, 15.5429)
        one = deflated_sharpe(0.0785, 1, *args)
        many = deflated_sharpe(0.0785, 441, *args)
        assert one > many

    def test_folds_fell_back_counts_none_train_expectancy(self):
        folds = _FakeResult().folds
        assert sum(1 for f in folds if f.train_expectancy is None) == 1

    def test_serialize_outcome_is_json_able_and_carries_the_gate(self):
        payload = campaign.serialize_outcome(_fake_outcome())
        text = json.dumps(payload)
        assert '"verdict": "B1"' in text
        assert payload["gate_conditions"] == list(GATE_CONDITIONS)
        assert payload["thresholds"]["WF_MIN_TRADES"] == 30
        assert payload["spans"]["holdout_days"] == config.HOLDOUT_DAYS


class TestCampaignCli:
    def test_pass_stub_exits_zero(self, state, monkeypatch, capsys):
        result = _FakeResult(gate=dict.fromkeys(GATE_CONDITIONS, True),
                             passed=True, ann=0.9)
        monkeypatch.setattr(
            campaign, "run_holdout_gate",
            lambda *a, **k: _fake_outcome(result, verdict="A"),
        )
        assert _campaign_command(None, state, stage="holdout") == 0
        out = capsys.readouterr().out
        assert "GATE: PASS" in out
        assert "VERDICT: A / BENCHMARK_NEGATIVE" in out

    def test_fail_stub_exits_one_and_prints_all_seven_conditions(
            self, state, monkeypatch, capsys):
        monkeypatch.setattr(campaign, "run_holdout_gate",
                            lambda *a, **k: _fake_outcome())
        assert _campaign_command(None, state, stage="holdout") == 1
        out = capsys.readouterr().out
        for name in GATE_CONDITIONS:
            assert name in out
        assert "GATE: FAIL" in out

    def test_benchmark_absolutes_print_beside_the_bits(self, state, monkeypatch,
                                                      capsys):
        monkeypatch.setattr(campaign, "run_holdout_gate",
                            lambda *a, **k: _fake_outcome())
        _campaign_command(None, state, stage="holdout")
        out = capsys.readouterr().out
        assert "buy-and-hold basket ABSOLUTE" in out
        assert "benchmark context: BENCHMARK_NEGATIVE" in out
        assert "vs basket" in out

    def test_northstar_is_labelled_reported_not_gated(self, state, monkeypatch,
                                                      capsys):
        monkeypatch.setattr(campaign, "run_holdout_gate",
                            lambda *a, **k: _fake_outcome())
        _campaign_command(None, state, stage="holdout")
        assert "REPORTED, NOT GATED" in capsys.readouterr().out

    def test_not_a_clean_holdout_stamp_on_a_rerun(self, state, monkeypatch,
                                                  capsys):
        monkeypatch.setattr(
            campaign, "run_holdout_gate",
            lambda *a, **k: _fake_outcome(run_index=2),
        )
        _campaign_command(None, state, stage="holdout",
                          force_reason="auditor asked")
        assert "NOT A CLEAN HOLDOUT" in capsys.readouterr().out

    def test_already_consumed_without_override_exits_two(self, state, monkeypatch,
                                                         capsys):
        def boom(*a, **k):
            raise campaign.CampaignError("holdout already consumed")

        monkeypatch.setattr(campaign, "run_holdout_gate", boom)
        assert _campaign_command(None, state, stage="holdout") == 2
        assert "ERROR:" in capsys.readouterr().out

    def test_holdout_violation_exits_two_and_records_a_violation_row(
            self, state, monkeypatch, capsys):
        def boom(*a, **k):
            raise campaign.HoldoutViolation("span intersects the holdout")

        monkeypatch.setattr(campaign, "run_holdout_gate", boom)
        assert _campaign_command(None, state, stage="holdout") == 2
        assert campaign.holdout_is_consumed(state) is False
        assert len(campaign.holdout_consumption_rows(state, kind="violation")) == 1

    def test_report_stage_without_a_run_exits_two(self, state, capsys):
        assert _campaign_command(None, state, stage="report") == 2
        assert "no completed campaign" in capsys.readouterr().out

    def test_report_stage_reads_persisted_state_and_never_reruns(self, state,
                                                                 monkeypatch):
        row_id, _ = campaign.open_holdout(
            state, campaign_id="p9", graph_hash="g", strategy_version="v",
            n_trials=441,
        )
        campaign.close_holdout(
            state, row_id, gate={"dsr": False},
            outcome=campaign.serialize_outcome(_fake_outcome()),
        )

        def boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("the report stage re-ran the gate")

        monkeypatch.setattr(campaign, "run_holdout_gate", boom)
        assert _campaign_command(None, state, stage="report") == 1

    def test_evolution_stop_reason_is_rederived_from_persisted_rows(self, state):
        """The stages are separate processes, so the stop reason must be
        recoverable from state.db rather than from an in-memory hand-off."""
        insert_campaign_row(state, "c1")
        state.execute(
            "INSERT INTO generations (campaign_id, gen_index, window_start_ms, "
            "window_end_ms, population_size, n_evaluated, n_unique_graphs, "
            "trials_cumulative, started_ts, finished_ts) "
            "VALUES ('c1', 0, 1, 2, 6, 6, 6, 6, 1, 2)"
        )
        state.execute(
            "INSERT INTO generations (campaign_id, gen_index, window_start_ms, "
            "window_end_ms, population_size, n_evaluated, n_unique_graphs, "
            "trials_cumulative, started_ts, finished_ts) "
            "VALUES ('c1', 1, 1, 2, 6, 6, 6, 12, 1, 2)"
        )
        state.commit()
        assert campaign.evolution_stop_reason(state).startswith("COMPLETE")

    def test_evolution_stop_reason_reports_an_incomplete_campaign(self, state):
        insert_campaign_row(state, "c1")
        assert campaign.evolution_stop_reason(state).startswith("DONE")

    def test_span_header_prints_both_epoch_and_iso(self, state, monkeypatch,
                                                   capsys):
        monkeypatch.setattr(campaign, "run_holdout_gate",
                            lambda *a, **k: _fake_outcome())
        _campaign_command(None, state, stage="holdout")
        out = capsys.readouterr().out
        assert str(H_START) in out and "2026-01-26" in out
        assert str(H_END) in out and "2026-07-27" in out
        assert "end EXCLUSIVE" in out
