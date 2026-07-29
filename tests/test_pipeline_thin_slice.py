"""End-to-end tests for the Phase 4 thin slice (v0.3.0).

Mirrors tests/test_backtest.py wholesale, which contract §8 names as the reference
module: tier-derived constants, autouse cache clearing, in-memory-equivalent
SQLite seeded bar-by-bar, class-per-behavior, direct CLI-handler invocation.

NO TEST IN THIS FILE ASSERTS PROFITABILITY. Expectancy, Sharpe and the gate are
Phase 1's and Phase 9's business; asserting a synthetic fixture is profitable is
how a test becomes a lie.
"""

import dataclasses
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from trading_bot import config
from trading_bot.backtest import engine
from trading_bot.backtest.engine import Trade, run_backtest
from trading_bot.cli import _graph_backtest_command
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.execute import run_graph_backtest
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.plugins import build_v020_graph
from trading_bot.risk.atr_stop import net_rr
from trading_bot.signals.donchian import DONCHIAN_KIND

SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded (contract §8): a future tier shift cannot leave
# these fixtures on the old timeframes while production moves.
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_REG = storage.TIMEFRAME_MS[REGIME_TF]
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000

STRATEGY_DIR = Path(config.STRATEGY_DIR)
THIN_SLICE = STRATEGY_DIR / "thin-slice.strategy.json"
THIN_SLICE_NOCONFIRM = STRATEGY_DIR / "thin-slice-noconfirm.strategy.json"

# Volume: the prior-window bars carry LOW_VOL and the breakout bar HIGH_VOL, so
# the ratio is HIGH_VOL / LOW_VOL = 3.0, comfortably above
# config.VOLUME_CONFIRM_MIN_RATIO (1.5). Derived from config so a threshold change
# cannot leave the fixture silently on the wrong side of the gate.
LOW_VOL = 10.0
HIGH_VOL = LOW_VOL * (config.VOLUME_CONFIRM_MIN_RATIO + 1.5)


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Clear BOTH memos around every test, and make sure the plug-ins are loaded."""
    registry.load_all()
    engine.clear_caches()
    fcontext.clear_caches()
    yield
    engine.clear_caches()
    fcontext.clear_caches()


def seed(conn, timeframe, rows, start=START, interval=D_SET):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    storage.upsert_candles(conn, SYMBOL, timeframe, data)
    return data


def patch_trending(monkeypatch, label="trending"):
    """Force a regime label on BOTH dispatch paths.

    engine.py and framework/context.py each import classify_series BY NAME, so
    patching only one leaves the two paths seeing different regimes — the single
    most likely cause of a confusing initial red (test_framework_parity.py:182).
    """
    stub = lambda df, **kw: pd.Series(label, index=df.index, dtype=object)  # noqa: E731
    monkeypatch.setattr(engine, "classify_series", stub)
    monkeypatch.setattr(fcontext, "classify_series", stub)


# --------------------------------------------------------------------------- #
# Fixture A — a clean Donchian long whose NET R:R clears RR_TARGET_MIN.
#
# 60 setup bars, close = 100 + i, high = close + 1, low = close - 1. True range is
# a constant 3.0 on every bar (high-low = 3 dominates both gap terms), so
# ATR(14) = 3.0 exactly. At the last bar (i = 59, close 160.0) the trailing 20-bar
# channel over bars 39..58 is upper = 102 + 58 = 160.0 and lower = 99 + 39 = 138.0,
# so close == upper (a Donchian long setup) and width = 22.0.
#
# Hand-computed geometry at a trigger entry of 161.0:
#   level  = 160.0 ; target = level + width = 182.0
#   ATR    = 3.0   ; stop   = 161.0 - 1.5 * 3.0 = 156.5
#   risk   = 4.5   -> risk_pct   = 4.5 / 161 = 0.0279503
#   reward = 21.0  -> reward_pct = 21.0 / 161 = 0.1304348
#   gross_rr = 21.0 / 4.5                      = 4.6667
#   net_rr   = (0.1304348 - 0.0014) / (0.0279503 + 0.0014) = 4.3966  >= 2.0
# This is the same ramp tests/test_backtest.py and test_framework_parity.py use.
# --------------------------------------------------------------------------- #


def donchian_rows():
    return [[100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, LOW_VOL] for i in range(60)]


# --------------------------------------------------------------------------- #
# Fixture B — a Donchian long whose NET R:R lands BETWEEN 1.5 and 2.0, so the
# R:R filter binds exactly where the legacy gross RR_FLOOR = 1.5 would not.
#
# Same close ramp (close = 101 + i) so the breakout still triggers, but the bars
# are much TALLER: high = close + 1, low = close - 10. True range is a constant
# 11.0, so ATR(14) = 11.0 exactly and the stop is 1.5 * 11 = 16.5 wide.
#
# MEASURED on this frame (not derived): trailing 20-bar channel at the last bar is
# upper 160.0, lower 130.0, width 30.0; close is 160.0 (== upper, a Donchian long
# setup); 55-bar mid 127.5; ATR(14) 11.0; ADX 100.0.
#   level = 160.0 ; target = level + width = 190.0 ; entry = 161.0
#   risk   = 16.5 -> risk_pct   = 16.5 / 161 = 0.1024845
#   reward = 29.0 -> reward_pct = 29.0 / 161 = 0.1801242
#   gross_rr = 29.0 / 16.5 = 1.7576   -> ABOVE config.RR_FLOOR (1.5)
#   net_rr = (0.1801242 - 0.0014) / (0.1024845 + 0.0014) = 1.7205 -> BELOW 2.0
# So RR_FLOOR would have let this trade through and RR_TARGET_MIN does not.
# --------------------------------------------------------------------------- #

MID_RR_LEVEL = 160.0
MID_RR_ENTRY = 161.0
MID_RR_WIDTH = 30.0
MID_RR_ATR = 11.0


def mid_rr_rows():
    return [[100.0 + i, 102.0 + i, 91.0 + i, 101.0 + i, LOW_VOL] for i in range(60)]


def trigger_rows(level, entry, outcome_rows):
    """21 bars sitting just below `level`, then a high-volume breakout bar.

    21 = config.VOLUME_LOOKBACK + 1, so the breakout bar has a FULL prior volume
    window and its ratio is defined (an undefined ratio fails the volume gate
    closed). Derived from config, not hardcoded.
    """
    below = level - 0.8
    rows = [[below, below + 0.2, below - 0.2, below, LOW_VOL]] * (
        config.VOLUME_LOOKBACK + 1
    )
    rows.append([below, entry + 0.1, below - 0.2, entry, HIGH_VOL])
    return rows + list(outcome_rows)


def seed_scenario(
    conn, *, setup_rows=None, level=160.0, entry=161.0, outcome_rows=None,
    breakout_volume=None,
):
    """Seed all three tiers so exactly one Donchian long breakout is available."""
    setup_rows = donchian_rows() if setup_rows is None else setup_rows
    outcome_rows = (
        [[entry + 20, entry + 30.0, entry + 21.0, entry + 29.0, LOW_VOL]]
        if outcome_rows is None
        else outcome_rows
    )
    seed(conn, REGIME_TF, [[100, 111, 99, 105, LOW_VOL]] * 12, interval=D_REG)
    seed(conn, SETUP_TF, setup_rows, interval=D_SET)
    last_setup_ts = START + (len(setup_rows) - 1) * D_SET
    rows_trig = trigger_rows(level, entry, outcome_rows)
    if breakout_volume is not None:
        # The breakout bar is the one right after the 21 prior bars.
        idx = config.VOLUME_LOOKBACK + 1
        rows_trig[idx] = list(rows_trig[idx])
        rows_trig[idx][4] = breakout_volume
    seed(conn, TRIGGER_TF, rows_trig, start=last_setup_ts + D_SET, interval=D_TRIG)
    return conn


def load_thin_slice():
    return fgraph.load(THIN_SLICE)


class TestFixtureGeometry:
    """Pins the fixtures' own premises, so a later failure points at the pipeline
    rather than at arithmetic in a comment."""

    def test_fixture_a_atr_and_channel(self):
        df = pd.DataFrame(donchian_rows(), columns=["open", "high", "low", "close", "volume"])
        assert math.isclose(float(wilder_atr(df, period=config.ATR_STOP_PERIOD).iloc[-1]), 3.0)
        assert float(df["close"].iloc[-1]) == 160.0

    def test_fixture_b_net_rr_is_between_1_5_and_2_0(self):
        df = pd.DataFrame(mid_rr_rows(), columns=["open", "high", "low", "close", "volume"])
        atr = float(wilder_atr(df, period=config.ATR_STOP_PERIOD).iloc[-1])
        assert math.isclose(atr, MID_RR_ATR)
        risk = config.ATR_STOP_MULTIPLE * atr
        target = MID_RR_LEVEL + MID_RR_WIDTH
        reward = target - MID_RR_ENTRY
        gross = reward / risk
        net = net_rr(
            reward / MID_RR_ENTRY, risk / MID_RR_ENTRY, config.FEE_PCT, config.SLIPPAGE_PCT
        )
        assert gross > config.RR_FLOOR, "RR_FLOOR must NOT have rejected this"
        assert config.RR_FLOOR < net < config.RR_TARGET_MIN, (
            f"fixture must land between the two floors: gross {gross:.4f}, net {net:.4f}"
        )


class TestThinSliceProducesTrades:
    def test_full_graph_takes_a_position(self, tmp_path, monkeypatch):
        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        trades = run_graph_backtest(conn, load_thin_slice(), SYMBOL)
        assert len(trades) >= 1
        assert all(t.pattern == DONCHIAN_KIND for t in trades)

    def test_every_taken_position_clears_RR_TARGET_MIN(self, tmp_path, monkeypatch):
        """The PRD's Success Metrics row 4 -- "100% of taken positions pass >=1:2
        R:R after costs at entry" -- as an assertion."""
        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        trades = run_graph_backtest(conn, load_thin_slice(), SYMBOL)
        assert trades
        for t in trades:
            assert t.planned_rr >= config.RR_TARGET_MIN, (
                f"planned_rr {t.planned_rr!r} < {config.RR_TARGET_MIN}"
            )

    def test_planned_rr_is_the_net_ratio_not_the_gross_one(self, tmp_path, monkeypatch):
        """Hand-computed: net 4.3966, gross 4.6667. The two must not be confused."""
        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        t = run_graph_backtest(conn, load_thin_slice(), SYMBOL)[0]
        risk_pct = abs(t.entry - t.stop) / t.entry
        reward_pct = abs(t.target - t.entry) / t.entry
        expected_net = net_rr(reward_pct, risk_pct, config.FEE_PCT, config.SLIPPAGE_PCT)
        gross = reward_pct / risk_pct
        assert math.isclose(t.planned_rr, expected_net, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(expected_net, 4.3966, rel_tol=0.0, abs_tol=1e-3)
        assert math.isclose(gross, 4.6667, rel_tol=0.0, abs_tol=1e-3)
        assert t.planned_rr < gross, "net must be strictly below gross at positive cost"

    def test_planned_rr_equals_the_filters_own_measured_number(self, tmp_path, monkeypatch):
        """The seam computes planned_rr itself rather than lifting it out of the
        filter's verdict, so the two are pinned equal at default costs. If they
        ever diverge, one of the two definitions moved."""
        from trading_bot.plugins.filters import rr_after_costs

        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        with rr_after_costs.recording() as tape:
            trades = run_graph_backtest(conn, load_thin_slice(), SYMBOL)
        accepted = [v for v in tape if v.accepted]
        assert trades and accepted
        assert math.isclose(
            trades[0].planned_rr, accepted[0].measured["net_rr"],
            rel_tol=0.0, abs_tol=1e-12,
        )

    def test_confirmations_are_populated_and_sorted(self, tmp_path, monkeypatch):
        """D8: registry KEYS of the Confirmations that passed, SORTED for
        determinism -- Phase 6 hashes graph results."""
        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        t = run_graph_backtest(conn, load_thin_slice(), SYMBOL)[0]
        assert isinstance(t.confirmations, tuple), "must be a tuple: Trade is frozen"
        assert t.confirmations == tuple(sorted(t.confirmations))
        assert set(t.confirmations) == {
            "confirmation.macd", "confirmation.volume-breakout",
        }

    def test_strategy_version_is_left_empty_for_phase_5(self, tmp_path, monkeypatch):
        """D7: contract §5 assigns strategy_version to Phase 5's version registry.
        Phase 4 declares the field and does NOT populate it, so Phase 5 has exactly
        one writer and no ambiguity about who owns the value."""
        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        trades = run_graph_backtest(conn, load_thin_slice(), SYMBOL)
        assert trades
        assert all(t.strategy_version == "" for t in trades)


class TestVolumeGateActuallyGates:
    """THE PROOF THAT KNOWN-LIMITATIONS §0c IS CLOSED.

    §0c: "volume is computed on every signal but gates nothing." The identical
    fixture, with only the breakout bar's VOLUME changed, must now produce a
    different number of trades. If it does not, the gate is decorative -- which is
    exactly the failure mode §0c records.
    """

    def test_low_volume_breakout_produces_no_trades(self, tmp_path, monkeypatch):
        conn = seed_scenario(
            storage.connect(str(tmp_path / "t.db")),
            # ratio = 0.9 * LOW_VOL / LOW_VOL = 0.9, below the 1.5 threshold
            breakout_volume=LOW_VOL * 0.9,
        )
        patch_trending(monkeypatch)
        assert run_graph_backtest(conn, load_thin_slice(), SYMBOL) == []

    def test_the_same_low_volume_fixture_trades_without_the_confirmations(
        self, tmp_path, monkeypatch
    ):
        """The ablation twin: identical graph minus the two Confirmation nodes.

        This is the half that makes the test above mean something -- it shows the
        zero-trade result is the GATE, not a broken fixture.
        """
        conn = seed_scenario(
            storage.connect(str(tmp_path / "t.db")), breakout_volume=LOW_VOL * 0.9
        )
        patch_trending(monkeypatch)
        trades = run_graph_backtest(conn, fgraph.load(THIN_SLICE_NOCONFIRM), SYMBOL)
        assert len(trades) >= 1
        assert trades[0].confirmations == (), "no Confirmation nodes => empty tuple"

    def test_undefined_volume_ratio_fails_closed(self, tmp_path, monkeypatch):
        """Too few prior trigger bars for a defined ratio => reject, never confirm."""
        setup_rows = donchian_rows()
        seedable = storage.connect(str(tmp_path / "t.db"))
        seed(seedable, REGIME_TF, [[100, 111, 99, 105, LOW_VOL]] * 12, interval=D_REG)
        seed(seedable, SETUP_TF, setup_rows, interval=D_SET)
        last_setup_ts = START + (len(setup_rows) - 1) * D_SET
        below, entry = 159.2, 161.0
        # Only 3 prior bars, far fewer than VOLUME_LOOKBACK, so the ratio is NaN.
        rows_trig = [[below, below + 0.2, below - 0.2, below, LOW_VOL]] * 3
        rows_trig.append([below, entry + 0.1, below - 0.2, entry, HIGH_VOL])
        rows_trig += [[entry + 20, entry + 30.0, entry + 21.0, entry + 29.0, LOW_VOL]]
        seed(seedable, TRIGGER_TF, rows_trig, start=last_setup_ts + D_SET, interval=D_TRIG)
        patch_trending(monkeypatch)
        assert run_graph_backtest(seedable, load_thin_slice(), SYMBOL) == []
        # ... and the no-confirmation twin still trades on the same bars.
        assert run_graph_backtest(seedable, fgraph.load(THIN_SLICE_NOCONFIRM), SYMBOL)


class TestMacdGateActuallyGates:
    """The MACD Confirmation must be able to veto a trade the rest of the pipeline
    would take.

    The veto is driven by RAISING min_hist above the fixture's measured histogram
    rather than by inverting the price series: a falling setup series would not
    produce a Donchian LONG breakout at all, so the zero-trade result would prove
    nothing about the gate. The SIGN semantics (long needs hist > min_hist, short
    needs hist < -min_hist, and the two are mirror images) are unit-tested in
    tests/test_plugins_confirmations.py.
    """

    def _with_macd_min_hist(self, graph, value):
        branches = []
        for b in graph.branches:
            confs = tuple(
                dataclasses.replace(c, params={**dict(c.params), "min_hist": value})
                if c.key == "confirmation.macd" else c
                for c in b.confirmations
            )
            branches.append(dataclasses.replace(b, confirmations=confs))
        return dataclasses.replace(graph, branches=tuple(branches))

    def test_an_unreachable_min_hist_vetoes_every_trade(self, tmp_path, monkeypatch):
        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        base = load_thin_slice()
        assert run_graph_backtest(conn, base, SYMBOL), "baseline must trade"
        # 0.05 is the ParamSpec's upper bound, far above any realistic normalised
        # histogram on this fixture.
        vetoed = self._with_macd_min_hist(base, 0.05)
        fcontext.clear_caches()
        assert run_graph_backtest(conn, vetoed, SYMBOL) == []


class TestRrFilterActuallyGates:
    """The filter must bind exactly where the legacy gross RR_FLOOR would not.

    Fixture B's geometry (see its comment block): gross_rr = 1.7576, which is ABOVE
    config.RR_FLOOR = 1.5, while net_rr = 1.7207, which is BELOW
    config.RR_TARGET_MIN = 2.0. So a pipeline gating on the legacy gross floor
    takes this trade and one gating on the new net floor does not.

    This also guards Task 8's GOTCHA 2: if policy.measured-move had silently
    re-applied the gross RR_FLOOR screen, the no-filter variant below would ALSO
    trade (gross clears 1.5) and this test would still pass -- so the assertion
    that the no-filter variant DOES trade is what proves the filter, not the
    policy, is doing the rejecting.
    """

    def test_a_plan_between_the_two_floors_is_rejected(self, tmp_path, monkeypatch):
        conn = seed_scenario(
            storage.connect(str(tmp_path / "t.db")),
            setup_rows=mid_rr_rows(),
            level=MID_RR_LEVEL,
            entry=MID_RR_ENTRY,
            outcome_rows=[[191.0, 201.0, 190.0, 200.0, LOW_VOL]],
        )
        patch_trending(monkeypatch)
        graph = load_thin_slice()
        assert run_graph_backtest(conn, graph, SYMBOL) == []

        # Same graph, filters removed: the trade appears, so the zero above is the
        # FILTER and not a broken fixture or a hidden screen in the policy.
        no_filter = dataclasses.replace(graph, filters=())
        fcontext.clear_caches()
        trades = run_graph_backtest(conn, no_filter, SYMBOL)
        assert len(trades) == 1
        t = trades[0]
        assert config.RR_FLOOR < t.planned_rr < config.RR_TARGET_MIN
        gross = abs(t.target - t.entry) / abs(t.entry - t.stop)
        assert gross > config.RR_FLOOR


class TestParityFieldsExcluded:
    """Contract §5 / D6: the three appended Trade fields are excluded from parity
    BY DESIGN, and this test is the guard against a future "tighten parity to `==`"
    change silently making the graph path unmergeable.

    Contract §5 defines parity as "same count, same entry/exit timestamps, same
    pnl_pct to floating-point tolerance" -- deliberately field-wise. Once the graph
    path populates planned_rr and confirmations, a graph Trade is NOT `==` an
    engine Trade even for an identical strategy.
    """

    def test_simulation_matches_while_the_audit_trail_differs(self, tmp_path, monkeypatch):
        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        free = dict(fee_pct=0.0, slippage_pct=0.0, funding_pct_per_day=0.0)
        want = run_backtest(conn, SYMBOL, **free)
        got = run_graph_backtest(conn, build_v020_graph(), SYMBOL, **free)

        assert len(want) == 1, "fixture must produce exactly one engine trade"
        assert len(got) == len(want)
        for g, e in zip(got, want):
            # The SIMULATION is identical...
            assert g.entry_ts == e.entry_ts
            assert g.exit_ts == e.exit_ts
            assert g.outcome == e.outcome
            assert math.isclose(g.pnl_pct, e.pnl_pct, rel_tol=0.0, abs_tol=1e-12)
            # ...and the AUDIT TRAIL is not, deliberately.
            assert e.planned_rr == 0.0, "engine.run_backtest never computes it"
            assert e.confirmations == ()
            assert g.planned_rr > 0.0, "the graph path records the net ratio"
            assert g != e, "whole-dataclass equality must NOT hold (contract §5)"

    def test_the_parity_helper_does_not_compare_the_new_fields(self):
        """Belt and braces: the parity module's own field lists must exclude them,
        so this phase cannot be broken by a rename there."""
        from tests import test_framework_parity as parity

        for field in ("planned_rr", "confirmations", "strategy_version"):
            assert field not in parity.PARITY_EXACT
            assert field not in parity.PARITY_CLOSE

    def test_trade_field_order_is_unchanged_with_three_appended(self):
        """Contract §5: appended at the END, each with a default; never reordered.

        v0.3.2 WS-B (D4) appends FOUR more fields after strategy_version, by
        the identical rule: end of the list, all defaulted, so this pinning
        test is updated (not violated) to include them. They carry the
        detected pattern's geometry to the replay chart -- see engine.Trade's
        docstring and ui/static/app.js's evoDrawTrade.
        """
        names = [f.name for f in dataclasses.fields(Trade)]
        assert names[:13] == [
            "symbol", "regime", "pattern", "direction", "entry_ts", "entry",
            "stop", "target", "exit_ts", "exit_price", "outcome", "pnl_pct",
            "volume_high",
        ]
        assert names[13:16] == ["planned_rr", "confirmations", "strategy_version"]
        assert names[16:] == [
            "pattern_start_ts", "pattern_end_ts", "pattern_level", "pattern_meta",
        ]
        defaults = {f.name: f.default for f in dataclasses.fields(Trade)}
        assert defaults["planned_rr"] == 0.0
        assert defaults["confirmations"] == ()
        assert defaults["strategy_version"] == ""
        assert defaults["pattern_start_ts"] == 0
        assert defaults["pattern_end_ts"] == 0
        assert defaults["pattern_level"] == 0.0
        assert defaults["pattern_meta"] == ()


class TestGraphFilesRoundTrip:
    """Catches hand-editing drift in the committed artifacts (Task 13 gotcha 1)."""

    @pytest.mark.parametrize("path", [THIN_SLICE, THIN_SLICE_NOCONFIRM], ids=lambda p: p.name)
    def test_loads_and_reserializes_identically(self, path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        g = fgraph.StrategyGraph.from_dict(payload)
        fgraph.validate(g)
        assert g.to_dict() == payload

    def test_noconfirm_differs_only_by_the_confirmation_nodes(self):
        full = json.loads(THIN_SLICE.read_text(encoding="utf-8"))
        nc = json.loads(THIN_SLICE_NOCONFIRM.read_text(encoding="utf-8"))
        assert sum(len(b["confirmations"]) for b in full["branches"]) == 4
        assert sum(len(b["confirmations"]) for b in nc["branches"]) == 0
        # Strip name/meta (which legitimately differ) and the confirmations, and
        # the two graphs must be byte-for-byte the same structure.
        def strip(payload):
            out = {k: v for k, v in payload.items() if k not in ("name", "meta")}
            out["branches"] = [
                {k: v for k, v in b.items() if k != "confirmations"}
                for b in out["branches"]
            ]
            return out

        assert strip(full) == strip(nc)

    def test_both_graphs_carry_the_rr_filter_and_use_config_defaults(self):
        for path in (THIN_SLICE, THIN_SLICE_NOCONFIRM):
            g = fgraph.load(path)
            assert [f.key for f in g.filters] == ["filter.rr-after-costs"]
            # NOTHING in either graph was chosen by search: every node's params
            # override nothing, so every value is its registry ParamSpec default.
            for _owner, node in fgraph._iter_nodes(g):
                assert dict(node.params) == {}, f"{node.id} overrides a default"

    def test_the_thin_slice_composes_more_than_one_detector(self):
        g = fgraph.load(THIN_SLICE)
        keys = {b.detector.key for b in g.branches}
        assert keys == {"detector.donchian-breakout", "detector.macd-cross"}


class TestGraphBacktestCommand:
    """Direct CLI-handler invocation, as tests/test_backtest.py does for
    _backtest_command / _walkforward_command."""

    def test_returns_zero_and_exercises_audit_and_rr_report(
        self, tmp_path, monkeypatch, capsys
    ):
        conn = seed_scenario(storage.connect(str(tmp_path / "t.db")))
        patch_trending(monkeypatch)
        rc = _graph_backtest_command(
            conn, [SYMBOL], graph_path=str(THIN_SLICE),
            start_ms=START, end_ms=START + 200 * D_SET,
            audit=True, rr_report=True,
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "graph: thin-slice" in out
        assert "filter.rr-after-costs" in out
        assert "planned_rr=" in out          # the audit line
        assert "survival_rate" in out        # the rr-report
        assert "POOLED:" in out
        assert f"planned_rr >= RR_TARGET_MIN ({config.RR_TARGET_MIN})" in out

    def test_zero_trades_is_a_result_not_an_error(self, tmp_path, monkeypatch, capsys):
        """cli.py's rule, inherited from _backtest_command: "a backtest with zero
        trades is a result, not an error"."""
        conn = seed_scenario(
            storage.connect(str(tmp_path / "t.db")), breakout_volume=LOW_VOL * 0.9
        )
        patch_trending(monkeypatch)
        rc = _graph_backtest_command(
            conn, [SYMBOL], graph_path=str(THIN_SLICE),
            start_ms=START, end_ms=START + 200 * D_SET, audit=True,
        )
        assert rc == 0
        assert "(no positions taken)" in capsys.readouterr().out

    def test_missing_graph_returns_one(self, tmp_path, capsys):
        conn = storage.connect(str(tmp_path / "t.db"))
        rc = _graph_backtest_command(
            conn, [SYMBOL], graph_path=str(tmp_path / "nope.json"),
            start_ms=START, end_ms=START + D_SET,
        )
        assert rc == 1
        assert "ERROR:" in capsys.readouterr().out

    def test_invalid_graph_returns_one(self, tmp_path, capsys):
        bad = tmp_path / "bad.strategy.json"
        bad.write_text(json.dumps({"name": "bad", "data": {"id": "d", "key": "nope.x"},
                                   "branches": []}), encoding="utf-8")
        conn = storage.connect(str(tmp_path / "t.db"))
        rc = _graph_backtest_command(
            conn, [SYMBOL], graph_path=str(bad),
            start_ms=START, end_ms=START + D_SET,
        )
        assert rc == 1
        assert "ERROR:" in capsys.readouterr().out


class TestZeroEngineCoreEdits:
    """PRD success metric / contract §12.4: a new plug-in costs zero engine-core
    edits.

    Encoded as a STRUCTURAL assertion rather than a `git diff` shell-out: a test
    that shells out to git fails in a clean checkout, in a worktree, and after a
    commit, so it would measure the environment rather than the code. The
    reviewable evidence is `git diff --stat src/trading_bot/` in the phase report;
    what is asserted here is the property that matters and is stable -- that all
    five new plug-ins reach the executor through the REGISTRY, with no reference
    to any of them anywhere in backtest/ or framework/.
    """

    NEW_PLUGINS = (
        "detector.macd-cross",
        "confirmation.volume-breakout",
        "confirmation.macd",
        "policy.measured-move",
        "filter.rr-after-costs",
    )

    @staticmethod
    def _code_string_literals(path: Path) -> set[str]:
        """Every string constant in a module EXCEPT docstrings.

        Docstrings and comments are prose: framework/execute.py's own docstrings
        legitimately name filter.rr-after-costs when explaining what planned_rr
        means, and that is documentation, not a dependency. What must not exist is
        an executable reference — a dispatch table, an `if key == ...`, an import.
        ast drops comments for free; docstrings are stripped explicitly.
        """
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                body = getattr(node, "body", None)
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))
        return {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        }

    def test_every_new_plugin_is_reachable_only_through_the_registry(self):
        import trading_bot.backtest.engine as eng
        import trading_bot.framework.execute as ex
        import trading_bot.framework.graph as gr

        literals: set[str] = set()
        for mod in (eng, ex, gr):
            literals |= self._code_string_literals(Path(mod.__file__))
        for key in self.NEW_PLUGINS:
            assert registry.get(key) is not None
            kind, name = key.split(".", 1)
            for literal in (key, name):
                assert literal not in literals, (
                    f"{literal!r} appears as an executable string literal in "
                    f"engine/executor/graph; a plug-in must be reachable only "
                    f"through the registry"
                )

    def test_every_new_plugin_has_a_non_empty_rationale(self):
        for key in self.NEW_PLUGINS:
            spec = registry.get(key)
            assert spec.rationale.strip(), key
            assert len(spec.rationale) > 80, f"{key}'s rationale is a placeholder"

    def test_every_new_plugin_declares_only_ParamSpecs(self):
        from trading_bot.framework.contracts import ParamSpec

        for key in self.NEW_PLUGINS:
            for pname, spec in registry.get(key).params.items():
                assert isinstance(spec, ParamSpec), f"{key}.{pname}"
                assert spec.doc.strip(), f"{key}.{pname} has no doc"
