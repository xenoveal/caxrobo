"""
Tests for `cli.py detector-report` — its METHODOLOGY, not its numbers.

The numbers belong to `.claude/PRPs/reports/phase8-detector-edge-report.md` and
change with the data. What must never change is the set of mechanisms that keep
this command a DIAGNOSTIC rather than a second fitness oracle, which contract §4
forbids. Each test below pins one of them, and the risk table's top row — "the
edge report becomes a fitness oracle; the gate becomes theater" — is the reason
they exist.
"""

import inspect
import time

import pytest

from trading_bot import cli
from trading_bot import config
from trading_bot.backtest import trials
from trading_bot.backtest.engine import Trade
from trading_bot.data import statestore
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import registry

START = 1_700_000_000_000
SYMBOL = "BTCUSDT"
DAY_MS = 86_400_000


@pytest.fixture(autouse=True)
def _clear_framework_caches():
    fcontext.clear_caches()
    yield
    fcontext.clear_caches()


def _trade(i: int, pnl: float) -> Trade:
    """One synthetic closed trade. Only the fields the report reads are meaningful."""
    return Trade(
        symbol=SYMBOL,
        regime="trending",
        pattern="probe",
        direction="long",
        entry_ts=START + i * DAY_MS,
        entry=100.0,
        stop=98.0,
        target=104.0,
        exit_ts=START + i * DAY_MS + DAY_MS,
        exit_price=100.0 * (1 + pnl),
        outcome="target" if pnl > 0 else "stop",
        pnl_pct=pnl,
        volume_high=False,
    )


@pytest.fixture
def fake_backtest(monkeypatch):
    """Replace the graph->Trade seam with a per-detector trade-count table.

    Monkeypatching the collaborator (the shape tests/test_signals.py's
    TestSignalCommand uses) keeps these tests about the REPORT: they must hold
    whatever the detectors happen to find in whatever data is present.
    """
    counts: dict[str, int] = {}
    calls: list[tuple[str, str]] = []

    def fake(conn, graph, symbol, **kwargs):
        key = graph.branches[0].detector.key
        calls.append((key, symbol))
        # Costs MUST be left unset by the caller: contract §1's "any new
        # execution path charges costs identically or it is lying".
        assert kwargs.get("fee_pct") is None
        assert kwargs.get("slippage_pct") is None
        assert kwargs.get("funding_pct_per_day") is None
        n = counts.get(key, 0)
        return [_trade(i, 0.02 if i % 2 else -0.01) for i in range(n)]

    monkeypatch.setattr(cli, "run_graph_backtest", fake)
    return {"counts": counts, "calls": calls}


@pytest.fixture
def dbs(tmp_path, monkeypatch):
    """An empty OHLCV db and a per-test state db, both on disk under tmp_path."""
    conn = storage.connect(str(tmp_path / "ohlcv.db"))
    state_path = str(tmp_path / "state.db")
    monkeypatch.setattr(config, "STATE_DB_PATH", state_path)
    yield {"conn": conn, "state_path": state_path}
    conn.close()


def _ledger_count(state_path: str) -> int:
    conn = statestore.connect(state_path)
    try:
        ledger = trials.TrialLedger(conn, config.DETECTOR_REPORT_CAMPAIGN)
        return ledger.count()
    finally:
        conn.close()


class TestHoldoutGuard:
    def test_holdout_guard_rejects_recent_end(self, dbs, capsys, fake_backtest):
        """MECHANISM, NOT PROMISE. Phase 9 owns a span no diagnostic may see, so
        an --end inside the reserved tail returns 2 and measures NOTHING.

        `end_ms` is computed from time.time() at test time rather than written as
        a literal, or the test starts passing for the wrong reason once the clock
        moves past the literal.
        """
        now = int(time.time() * 1000)
        code = cli._detector_report_command(
            dbs["conn"], (SYMBOL,), start_ms=now - 400 * DAY_MS, end_ms=now - DAY_MS,
            state_db=dbs["state_path"],
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "REFUSED" in err
        assert "holdout" in err
        assert "DETECTOR_REPORT_HOLDOUT_GUARD_DAYS" in err
        assert fake_backtest["calls"] == []
        assert _ledger_count(dbs["state_path"]) == 0

    def test_guard_boundary_is_exactly_the_configured_day_count(self, dbs, capsys):
        """One millisecond past the boundary must refuse; one before must not."""
        now = START
        guard = config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS * DAY_MS
        assert (
            cli._detector_report_command(
                dbs["conn"], (SYMBOL,), start_ms=now - 400 * DAY_MS,
                end_ms=now - guard + 1, state_db=dbs["state_path"], now_ms=now,
            )
            == 2
        )
        capsys.readouterr()

    def test_missing_end_is_refused_rather_than_defaulted_to_now(self, dbs, capsys):
        assert (
            cli._detector_report_command(
                dbs["conn"], (SYMBOL,), start_ms=START, end_ms=None,
                state_db=dbs["state_path"],
            )
            == 2
        )
        capsys.readouterr()


class TestCoverageFlag:
    def test_coverage_flag_touches_neither_db_nor_ledger(self, tmp_path, capsys):
        """A read-only ledger listing that creates a database as a side effect is
        a trap — the same reason `plugins` takes no connection."""
        state_path = str(tmp_path / "never-created.db")
        code = cli._detector_report_command(
            None, None, start_ms=None, end_ms=None, coverage=True, state_db=state_path
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "144 rows / 18 families" in out
        assert "tier 1: 2/3" in out
        assert "tier 2: 6/6" in out
        assert not (tmp_path / "never-created.db").exists()

    def test_coverage_can_be_written_to_a_file(self, tmp_path, capsys):
        target = tmp_path / "coverage.md"
        assert (
            cli._detector_report_command(
                None, None, start_ms=None, end_ms=None, coverage=True,
                out_path=str(target),
            )
            == 0
        )
        capsys.readouterr()
        assert "covered 19" in target.read_text()


class TestReportBody:
    SPAN = {"start_ms": START - 400 * DAY_MS, "end_ms": START}

    def _run(self, dbs, capsys, *, detectors=None):
        code = cli._detector_report_command(
            dbs["conn"], (SYMBOL,), detectors=detectors,
            state_db=dbs["state_path"], now_ms=START + 400 * DAY_MS, **self.SPAN
        )
        return code, capsys.readouterr().out

    def test_insufficient_suppresses_expectancy(self, dbs, capsys, fake_backtest):
        """KNOWN-LIMITATIONS §1's lesson one level down: 23 trades could not clear
        a floor of 30, so printing an expectancy off 9 trades repeats that error.
        Suppression is what stops the table being mined."""
        fake_backtest["counts"]["detector.falling-wedge"] = 9
        code, out = self._run(dbs, capsys, detectors=["detector.falling-wedge"])
        assert code == 0
        line = next(l for l in out.splitlines() if l.startswith("falling-wedge"))
        assert f"INSUFFICIENT(<{config.DETECTOR_MIN_EVENTS_FOR_REPORT})" in line
        assert "%" not in line  # every rate and expectancy figure suppressed
        assert line.split()[2] == "9"  # the trade COUNT is still shown

    def test_sufficient_rows_print_measured_figures(self, dbs, capsys, fake_backtest):
        fake_backtest["counts"]["detector.falling-wedge"] = 40
        code, out = self._run(dbs, capsys, detectors=["detector.falling-wedge"])
        assert code == 0
        line = next(l for l in out.splitlines() if l.startswith("falling-wedge"))
        assert "MEASURED" in line
        assert "%" in line

    def test_a_detector_with_no_edge_is_a_result_not_an_error(
        self, dbs, capsys, fake_backtest
    ):
        """Mirrors _backtest_command's "a backtest with zero trades is a result"."""
        fake_backtest["counts"]["detector.falling-wedge"] = 0
        code, out = self._run(dbs, capsys, detectors=["detector.falling-wedge"])
        assert code == 0
        assert "INSUFFICIENT" in out

    def test_one_ledger_row_per_detector(self, dbs, capsys, fake_backtest):
        """Contract §4.2: nothing scores a candidate without a ledger handle, and
        this report is not exempt. One row per (detector, params, span) —
        deliberately NOT one per (detector, symbol), because the measured row is
        the pooled evaluation."""
        keys = ["detector.falling-wedge", "detector.double-top"]
        for k in keys:
            fake_backtest["counts"][k] = 25
        code, out = self._run(dbs, capsys, detectors=keys)
        assert code == 0
        assert _ledger_count(dbs["state_path"]) == len(keys)
        assert f"rows=+{len(keys)}" in out

    def test_ledger_rows_accumulate_across_runs(self, dbs, capsys, fake_backtest):
        """The ledger measures evaluations PERFORMED; re-running is a legitimate
        second row, and silently deduping would understate n_trials in the
        flattering direction."""
        fake_backtest["counts"]["detector.double-top"] = 25
        self._run(dbs, capsys, detectors=["detector.double-top"])
        self._run(dbs, capsys, detectors=["detector.double-top"])
        assert _ledger_count(dbs["state_path"]) == 2

    def test_banner_states_selection_cost(self, dbs, capsys, fake_backtest):
        """Reading the table is free; letting it inform selection is not, and the
        banner prints the number so the debt cannot be incurred silently."""
        keys = ["detector.falling-wedge", "detector.double-top", "detector.bull-flag"]
        for k in keys:
            fake_backtest["counts"][k] = 25
        code, out = self._run(dbs, capsys, detectors=keys)
        assert code == 0
        assert "DIAGNOSTIC -- NOT A GATE VERDICT" in out
        assert "SELECTION COST IF USED" in out
        assert f"n_trials += {len(keys)}" in out
        assert "NOTHING SWEPT" in out
        assert "TUNING span" in out
        # No gate verdict, ever.
        for forbidden in ("PASS", "FAIL", "gate:", "sharpe", "dsr"):
            assert forbidden not in out

    def test_report_is_sorted_by_tier_then_name_not_by_expectancy(
        self, dbs, capsys, fake_backtest
    ):
        """SORTING BY EXPECTANCY *IS* SELECTION. This pins the anti-mining rule so
        a later 'helpful' sort cannot slip in."""
        registry.load_all()
        code, out = self._run(dbs, capsys)
        assert code == 0
        body = out.split("-" * 104)[1].strip().splitlines()
        names = [l.split()[0] for l in body]
        tiers = [l.split()[1] for l in body]
        expected = [
            (99 if registry.get(f"detector.{n}").tier is None
             else registry.get(f"detector.{n}").tier, n)
            for n in names
        ]
        assert expected == sorted(expected)
        assert tiers == ["1"] * tiers.count("1") + ["2"] * tiers.count("2") + [
            "--"
        ] * tiers.count("--")

    def test_every_phase8_detector_gets_a_row(self, dbs, capsys, fake_backtest):
        registry.load_all()
        code, out = self._run(dbs, capsys)
        assert code == 0
        names = {
            registry.get(k).name for k in cli._phase8_detector_keys()
        }
        printed = {l.split()[0] for l in out.split("-" * 104)[1].strip().splitlines()}
        assert printed == names

    def test_unknown_detector_key_is_a_clean_error_not_a_traceback(
        self, dbs, capsys, fake_backtest
    ):
        code = cli._detector_report_command(
            dbs["conn"], (SYMBOL,), detectors=["detector.no-such-thing"],
            state_db=dbs["state_path"], now_ms=START + 400 * DAY_MS, **self.SPAN
        )
        assert code == 1
        assert "unknown detector key" in capsys.readouterr().err

    def test_a_non_phase8_detector_is_refused(self, dbs, capsys, fake_backtest):
        """Phase 3's and Phase 4's detectors are not re-measured here under a
        fixed exit policy that was never designed for them."""
        code = cli._detector_report_command(
            dbs["conn"], (SYMBOL,), detectors=["detector.donchian-breakout"],
            state_db=dbs["state_path"], now_ms=START + 400 * DAY_MS, **self.SPAN
        )
        assert code == 1
        assert "not a Phase 8 detector" in capsys.readouterr().err

    def test_report_can_be_written_to_a_file(self, dbs, capsys, tmp_path, fake_backtest):
        fake_backtest["counts"]["detector.double-top"] = 25
        target = tmp_path / "report.md"
        code = cli._detector_report_command(
            dbs["conn"], (SYMBOL,), detectors=["detector.double-top"],
            out_path=str(target), state_db=dbs["state_path"],
            now_ms=START + 400 * DAY_MS, **self.SPAN
        )
        capsys.readouterr()
        assert code == 0
        text = target.read_text()
        assert "DIAGNOSTIC" in text and "SELECTION COST IF USED" in text


class TestStructuralGuards:
    def test_no_second_pnl_path(self):
        """Contract §1: "any new execution path charges costs identically or it is
        lying". A crude but effective structural guard — the command must go
        through run_graph_backtest and must not do arithmetic on the cost
        constants. `_cost_columns` is a separate, clearly-named REPORTING helper
        that recomputes the cost for a display column and computes no P&L."""
        src = inspect.getsource(cli._detector_report_command)
        assert "run_graph_backtest(" in src
        # The docstring NAMES the constants in order to explain that it must not
        # touch them, so the guard scans CODE only. Comment lines are stripped for
        # the same reason.
        body = src.split('"""')[2]
        code = "\n".join(
            l for l in body.splitlines() if not l.strip().startswith("#")
        )
        for constant in ("FEE_PCT", "SLIPPAGE_PCT", "FUNDING_PCT_PER_DAY"):
            assert constant not in code, constant

    def test_no_parameter_override_surface_exists(self):
        """A --param flag, or any loop over parameter values, would convert this
        diagnostic into a second fitness oracle."""
        params = inspect.signature(cli._detector_report_command).parameters
        assert "param" not in params and "params" not in params
        assert "grid" not in params
        src = inspect.getsource(cli._diagnostic_graph)
        # The graph is built from ParamSpec DEFAULTS: it passes no params at all.
        assert "params=" not in src

    def test_diagnostic_graph_has_no_confirmations_and_no_filters(self):
        registry.load_all()
        graph = cli._diagnostic_graph("detector.double-top")
        assert graph.filters == ()
        assert graph.branches[0].confirmations == ()
        assert graph.branches[0].policy.key == "policy.measured-move"
        assert graph.data.key == "data.ohlcv"
        assert graph.meta["diagnostic"] is True

    def test_diagnostic_graph_validates(self):
        from trading_bot.framework.graph import validate

        registry.load_all()
        for key in cli._phase8_detector_keys():
            validate(cli._diagnostic_graph(key))

    def test_diagnostic_graph_hash_is_stable_per_detector(self):
        """The ledger keys on the graph hash, so two runs of one detector must
        agree and two different detectors must not collide."""
        from trading_bot.framework.graph import graph_hash

        registry.load_all()
        a1 = graph_hash(cli._diagnostic_graph("detector.double-top"))
        a2 = graph_hash(cli._diagnostic_graph("detector.double-top"))
        b = graph_hash(cli._diagnostic_graph("detector.double-bottom"))
        assert a1 == a2 != b

    def test_cost_ratio_is_measured_against_the_config_ceiling(self):
        trades = [_trade(i, 0.01) for i in range(10)]
        cols = cli._cost_columns(trades)
        # risk_pct = |100 - 98| / 100 = 0.02; one day held.
        expected_cost = 2 * (config.FEE_PCT + config.SLIPPAGE_PCT) + (
            config.FUNDING_PCT_PER_DAY * 1.0
        )
        assert cols["median_risk_pct"] == pytest.approx(0.02)
        assert cols["mean_cost_pct"] == pytest.approx(expected_cost)
        assert cols["cost_ratio"] == pytest.approx(expected_cost / 0.02)
        assert config.COST_RATIO_CEILING == 0.10

    def test_cost_columns_are_none_without_trades(self):
        assert cli._cost_columns([]) == {
            "mean_cost_pct": None,
            "median_risk_pct": None,
            "cost_ratio": None,
        }
