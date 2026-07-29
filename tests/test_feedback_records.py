"""
Tests for feedback/records.py and plugins/reviewers/trade_quality.py
(v0.3.0 Phase 5, contract §3/§4/§6/§8).
"""

import ast
import math
from pathlib import Path

import pytest

from trading_bot import config
from trading_bot.backtest.engine import Trade
from trading_bot.data import statestore, storage
from trading_bot.feedback import records
from trading_bot.framework import registry
from trading_bot.plugins.reviewers import trade_quality

SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded (contract §8): review_tf resolves to
# SIGNAL_TRIGGER_TIMEFRAME by default (config.REVIEW_TIMEFRAME is None).
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _load_registry():
    registry.load_all()


@pytest.fixture
def ohlcv_conn():
    conn = storage.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def state_conn():
    conn = statestore.connect(":memory:")
    yield conn
    conn.close()


def seed(conn, rows, *, start, interval=D_TRIG, timeframe=TRIGGER_TF, symbol=SYMBOL):
    """rows: list of (open, high, low, close, volume) tuples."""
    storage.upsert_candles(
        conn, symbol, timeframe,
        [[start + i * interval] + list(r) for i, r in enumerate(rows)],
    )


def make_trade(
    *, direction="long", entry=100.0, stop=95.0, target=110.0,
    entry_ts=START, exit_ts=None, exit_price=100.0, outcome="target",
    pnl_pct=0.0, regime="trending", pattern="donchian-breakout",
    planned_rr=1.5, confirmations=(), strategy_version="v1", symbol=SYMBOL,
):
    exit_ts = START + 3 * D_TRIG if exit_ts is None else exit_ts
    return Trade(
        symbol=symbol, regime=regime, pattern=pattern, direction=direction,
        entry_ts=entry_ts, entry=entry, stop=stop, target=target,
        exit_ts=exit_ts, exit_price=exit_price, outcome=outcome, pnl_pct=pnl_pct,
        volume_high=False, planned_rr=planned_rr, confirmations=confirmations,
        strategy_version=strategy_version,
    )


def make_ctx(conn, *, review_tf=TRIGGER_TF, strategy_version="v1", span_class="in-sample",
             target_ann_return=0.5, created_ts=START):
    return records.ReviewContext(
        conn=conn, review_tf=review_tf, strategy_version=strategy_version,
        span_class=span_class, target_ann_return=target_ann_return, created_ts=created_ts,
    )


def make_record(**overrides):
    """A minimal, valid ReviewRecord for diagnose()-only tests (no DB, no
    trade_quality involved)."""
    defaults = dict(
        record_id="r0", strategy_version="v1", reviewer="reviewer.trade-quality",
        span_class="in-sample", symbol=SYMBOL, regime="trending",
        pattern="donchian-breakout", direction="long", outcome="target",
        entry_ts=START, exit_ts=START + D_TRIG, created_ts=START,
        entry=100.0, stop=95.0, target=110.0, exit_price=105.0, pnl_pct=0.05,
        holding_days=1.0, review_tf=TRIGGER_TF, bars_reviewed=1, bars_expected=1,
        mfe_pct=0.06, mae_pct=0.01, target_distance_pct=0.10, stop_distance_pct=0.05,
        tp_capture_ratio=0.83, tp_verdict="good", sl_headroom_ratio=0.2, sl_verdict="over-wide",
        pace_ratio=1.2, pace_verdict="on-pace", confirmations=(), planned_rr=1.5, notes="",
    )
    defaults.update(overrides)
    return records.ReviewRecord(**defaults)


class TestSchema:
    def test_connect_creates_no_tables(self, state_conn):
        rows = state_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        assert rows == []

    def test_ensure_schema_is_idempotent(self, state_conn):
        records.ensure_schema(state_conn)
        records.ensure_schema(state_conn)
        names = {r[0] for r in state_conn.execute("SELECT name FROM sqlite_master").fetchall()}
        assert records.TABLE in names
        assert "idx_review_version" in names
        assert "idx_review_exit" in names

    def test_ts_columns_are_integer(self, state_conn):
        records.ensure_schema(state_conn)
        cols = {
            row[1]: row[2]
            for row in state_conn.execute(f"PRAGMA table_info({records.TABLE})").fetchall()
        }
        for name in ("entry_ts", "exit_ts", "created_ts"):
            assert cols[name] == "INTEGER"

    def test_roundtrip_preserves_every_field(self, state_conn):
        rec = make_record(confirmations=("confirmation.macd",))
        records.insert_records(state_conn, [rec])
        loaded = records.load_records(state_conn)
        assert len(loaded) == 1
        assert loaded[0] == rec

    def test_double_insert_is_one_row(self, state_conn):
        rec = make_record()
        records.insert_records(state_conn, [rec])
        records.insert_records(state_conn, [rec])
        rows = state_conn.execute(f"SELECT COUNT(*) FROM {records.TABLE}").fetchone()
        assert rows[0] == 1

    def test_no_aggregate_field(self):
        """Contract §4 structural lock #1: no field name may suggest a
        combined fitness value."""
        import re
        forbidden = re.compile(r"fitness|score|reward|objective|rank")
        for name in records.ReviewRecord.__dataclass_fields__:
            assert not forbidden.search(name), f"forbidden aggregate-like field: {name}"


class TestExcursions:
    def test_entry_bar_is_excluded(self, ohlcv_conn):
        """The phase's key invariant (engine.py:388). Bar 0 (the entry bar)
        carries a huge favourable wick that must NOT be counted."""
        rows = [
            (100.0, 200.0, 99.0, 100.0, 1.0),   # bar 0: entry bar, huge wick, EXCLUDED
            (100.0, 105.0, 99.0, 101.0, 1.0),   # bar 1
            (101.0, 103.0, 98.0, 100.0, 1.0),   # bar 2
            (100.0, 110.0, 95.0, 105.0, 1.0),   # bar 3: exit bar
        ]
        seed(ohlcv_conn, rows, start=START)
        trade = make_trade(direction="long", entry=100.0, entry_ts=START, exit_ts=START + 3 * D_TRIG)
        ctx = make_ctx(ohlcv_conn)
        mfe, mae, reviewed, expected = trade_quality.measure_excursions(ctx, trade)
        assert expected == 3
        assert reviewed == 3
        assert mfe == pytest.approx(0.10)  # from bar 3's high=110, NOT bar 0's 200
        assert mae == pytest.approx(0.05)  # from bar 3's low=95

    def test_exit_bar_is_included(self, ohlcv_conn):
        rows = [
            (100.0, 101.0, 99.0, 100.0, 1.0),   # bar 1
            (100.0, 101.0, 99.0, 100.0, 1.0),   # bar 2
            (100.0, 120.0, 80.0, 105.0, 1.0),   # bar 3: exit bar, the only extreme
        ]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(direction="long", entry=100.0, entry_ts=START, exit_ts=START + 3 * D_TRIG)
        ctx = make_ctx(ohlcv_conn)
        mfe, mae, _, _ = trade_quality.measure_excursions(ctx, trade)
        assert mfe == pytest.approx(0.20)
        assert mae == pytest.approx(0.20)

    def test_mfe_mae_hand_computed_long(self, ohlcv_conn):
        rows = [(100.0, 108.0, 97.0, 102.0, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(direction="long", entry=100.0, entry_ts=START, exit_ts=START + D_TRIG)
        ctx = make_ctx(ohlcv_conn)
        mfe, mae, reviewed, expected = trade_quality.measure_excursions(ctx, trade)
        assert (reviewed, expected) == (1, 1)
        assert mfe == pytest.approx((108.0 - 100.0) / 100.0)
        assert mae == pytest.approx((100.0 - 97.0) / 100.0)

    def test_mfe_mae_hand_computed_short(self, ohlcv_conn):
        rows = [(100.0, 104.0, 90.0, 95.0, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(direction="short", entry=100.0, entry_ts=START, exit_ts=START + D_TRIG)
        ctx = make_ctx(ohlcv_conn)
        mfe, mae, _, _ = trade_quality.measure_excursions(ctx, trade)
        # short: favourable is price falling, adverse is price rising.
        assert mfe == pytest.approx((100.0 - 90.0) / 100.0)
        assert mae == pytest.approx((104.0 - 100.0) / 100.0)

    def test_never_favourable_gives_zero_mfe_and_none_capture(self, ohlcv_conn):
        rows = [(100.0, 100.0, 90.0, 95.0, 1.0), (95.0, 96.0, 85.0, 90.0, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(
            direction="long", entry=100.0, entry_ts=START, exit_ts=START + 2 * D_TRIG,
            outcome="stop", exit_price=90.0,
        )
        ctx = make_ctx(ohlcv_conn)
        rec = trade_quality.review(trade, ctx)
        assert rec.mfe_pct == 0.0
        assert rec.tp_capture_ratio is None
        assert rec.tp_verdict == "never-favoured"

    def test_coverage_below_threshold_returns_none(self, ohlcv_conn, caplog):
        # bars_expected = 10, but only 1 bar is actually stored -> coverage 0.1
        rows = [(100.0, 101.0, 99.0, 100.0, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(direction="long", entry=100.0, entry_ts=START, exit_ts=START + 10 * D_TRIG)
        ctx = make_ctx(ohlcv_conn)
        import logging
        with caplog.at_level(logging.WARNING, logger="trading_bot"):
            mfe, mae, reviewed, expected = trade_quality.measure_excursions(ctx, trade)
        assert (mfe, mae) == (None, None)
        assert expected == 10
        assert reviewed == 1
        assert any("coverage" in r.message for r in caplog.records)

    def test_interval_mismatch_raises(self, ohlcv_conn):
        # Seed bars at 2x the review_tf's interval -> median spacing mismatch.
        rows = [(100.0, 101.0, 99.0, 100.0, 1.0)] * 3
        seed(ohlcv_conn, rows, start=START + D_TRIG, interval=2 * D_TRIG)
        trade = make_trade(direction="long", entry=100.0, entry_ts=START, exit_ts=START + 6 * D_TRIG)
        ctx = make_ctx(ohlcv_conn)
        with pytest.raises(ValueError, match="spacing"):
            trade_quality.measure_excursions(ctx, trade)


class TestVerdicts:
    def test_verdict_tuples_are_declared(self):
        assert records.TP_VERDICTS == ("n/a", "never-favoured", "target-too-far", "left-money", "good")
        assert records.SL_VERDICTS == ("n/a", "hit", "over-wide", "tight", "ok")
        assert records.PACE_VERDICTS == ("n/a", "loss", "on-pace", "behind")

    def test_target_too_far(self, ohlcv_conn):
        # MFE never gets within REVIEW_TP_UNREACHABLE_FRAC of the target distance.
        rows = [(100.0, 101.0, 99.5, 100.0, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(
            direction="long", entry=100.0, target=110.0, entry_ts=START,
            exit_ts=START + D_TRIG, outcome="time", exit_price=100.5,
        )
        rec = trade_quality.review(trade, make_ctx(ohlcv_conn))
        assert rec.tp_verdict == "target-too-far"

    def test_good_exit(self, ohlcv_conn):
        rows = [(100.0, 106.0, 99.0, 105.5, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(
            direction="long", entry=100.0, target=106.0, entry_ts=START,
            exit_ts=START + D_TRIG, outcome="target", exit_price=106.0,
        )
        rec = trade_quality.review(trade, make_ctx(ohlcv_conn))
        assert rec.tp_verdict == "good"

    def test_sl_hit_regardless_of_headroom(self, ohlcv_conn):
        rows = [(100.0, 100.0, 94.0, 95.0, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(
            direction="long", entry=100.0, stop=95.0, entry_ts=START,
            exit_ts=START + D_TRIG, outcome="stop", exit_price=95.0, pnl_pct=-0.05,
        )
        rec = trade_quality.review(trade, make_ctx(ohlcv_conn))
        assert rec.sl_verdict == "hit"

    def test_sl_over_wide(self, ohlcv_conn):
        rows = [(100.0, 101.0, 99.5, 100.5, 1.0)]  # MAE tiny vs a wide stop
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(
            direction="long", entry=100.0, stop=90.0, entry_ts=START,
            exit_ts=START + D_TRIG, outcome="time", exit_price=100.5,
        )
        rec = trade_quality.review(trade, make_ctx(ohlcv_conn))
        assert rec.sl_verdict == "over-wide"

    def test_verdict_strings_are_declared(self, ohlcv_conn):
        rows = [(100.0, 106.0, 94.0, 105.0, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(entry_ts=START, exit_ts=START + D_TRIG, outcome="target", exit_price=106.0)
        rec = trade_quality.review(trade, make_ctx(ohlcv_conn))
        assert rec.tp_verdict in records.TP_VERDICTS
        assert rec.sl_verdict in records.SL_VERDICTS
        assert rec.pace_verdict in records.PACE_VERDICTS


class TestPace:
    def test_pace_ratio_hand_computed(self, ohlcv_conn):
        holding_days = 30
        interval = D_TRIG
        n_bars = int(holding_days * 86_400_000 / interval)
        rows = [(100.0, 100.5, 99.5, 100.0, 1.0)] * n_bars
        seed(ohlcv_conn, rows, start=START + interval)
        trade = make_trade(
            direction="long", entry=100.0, entry_ts=START,
            exit_ts=START + n_bars * interval, exit_price=100.0, pnl_pct=0.05,
        )
        ctx = make_ctx(ohlcv_conn, target_ann_return=0.5)
        rec = trade_quality.review(trade, ctx)
        required = (1.5) ** (30 / 365) - 1.0
        assert rec.holding_days == pytest.approx(30.0)
        assert rec.pace_ratio == pytest.approx(0.05 / required, rel=1e-9)
        assert rec.pace_ratio == pytest.approx(1.4747, rel=1e-3)

    def test_pace_ratio_none_when_holding_days_zero(self, ohlcv_conn):
        trade = make_trade(entry_ts=START, exit_ts=START, outcome="end")
        ctx = make_ctx(ohlcv_conn)
        rec = trade_quality.review(trade, ctx)
        assert rec.pace_ratio is None
        assert rec.pace_verdict == "n/a"

    def test_single_trade_never_produces_an_annualised_figure(self, ohlcv_conn, state_conn):
        # "ann" alone is too broad a substring: "planned_rr" legitimately
        # contains it ("pl-ANN-ed"). The regression this pins is specifically
        # about an ANNUALIZED metric (KNOWN-LIMITATIONS §2's ann_return_pct
        # error), so check for that pattern precisely.
        assert not any(
            "ann_return" in name or "annuali" in name
            for name in records.ReviewRecord.__dataclass_fields__
        )
        rows = [(100.0, 106.0, 94.0, 105.0, 1.0)]
        seed(ohlcv_conn, rows, start=START + D_TRIG)
        trade = make_trade(entry_ts=START, exit_ts=START + D_TRIG, outcome="target", exit_price=106.0)
        rec = trade_quality.review(trade, make_ctx(ohlcv_conn))
        d = records.diagnose([rec], start_ms=START, end_ms=START + D_TRIG)
        assert d.pace["sample_adequate"] is False
        assert d.suggestions == ("insufficient-sample",)


class TestDiagnosis:
    def test_outcome_mix_and_dominance(self):
        recs = (
            [make_record(record_id=str(i), outcome="stop", pnl_pct=-0.02) for i in range(5)]
            + [make_record(record_id=f"t{i}", outcome="target", pnl_pct=0.05) for i in range(2)]
        )
        d = records.diagnose(recs, start_ms=START, end_ms=START + D_TRIG)
        assert d.n_records == 7
        assert d.by_outcome == {"stop": 5, "target": 2}
        assert d.dominant_outcome == "stop"

    def test_full_coverage_hard_gate_is_unmeasurable_not_dead_weight(self):
        """A hard-gating Confirmation (framework/execute.py:291-295's
        `if not verdict.passed: return None`) is present on 100% of closed
        trades BY CONSTRUCTION -- there is no rejected-event counterfactual,
        so it must be labelled "unmeasurable" and must NEVER produce a
        drop-confirmation suggestion.

        This regression is pinned because acting on coverage alone is
        MEASURABLY HARMFUL: Phase 4 measured dropping volume+MACD moves
        pooled trades 276 -> 477 and expectancy_pct +0.2028% -> -0.2523%,
        Sharpe +0.53 -> -0.86 (reports/phase4-thin-slice.md §B) -- the sleeve
        flips negative on every symbol. The old (buggy) version of this test
        asserted the opposite: that full coverage alone should produce a
        drop-confirmation suggestion.
        """
        recs = [
            make_record(
                record_id=str(i),
                confirmations=("confirmation.macd", "confirmation.volume-breakout"),
            )
            for i in range(35)
        ]
        d = records.diagnose(recs, start_ms=START, end_ms=START + 200 * D_TRIG)
        assert d.confirmation_coverage["confirmation.macd"] == 1.0
        assert d.confirmation_pnl_delta["confirmation.macd"] is None  # no counterfactual
        assert d.confirmation_status["confirmation.macd"] == "unmeasurable"
        assert d.confirmation_status["confirmation.volume-breakout"] == "unmeasurable"
        assert not any(s.startswith("drop-confirmation:") for s in d.suggestions)

    def test_advisory_confirmation_with_a_real_counterfactual_is_dead_weight(self):
        """Once a counterfactual DOES exist -- here, one record lacks
        "confirmation.advisory" while the rest carry it, so a real "without
        it" group exists even at 49/50 = 0.98 coverage -- dead-weight IS
        decidable and DOES produce a drop-confirmation suggestion.
        "confirmation.hard-gate" is present on every record (no
        counterfactual) and stays "unmeasurable" in the SAME diagnosis,
        proving the two statuses are decided independently per name.
        """
        recs = [
            make_record(
                record_id=str(i), outcome="target", pnl_pct=0.05,
                confirmations=("confirmation.hard-gate", "confirmation.advisory"),
            )
            for i in range(49)
        ] + [
            make_record(
                record_id="49", outcome="target", pnl_pct=0.05,
                confirmations=("confirmation.hard-gate",),
            )
        ]
        d = records.diagnose(recs, start_ms=START, end_ms=START + 200 * D_TRIG)
        assert d.confirmation_coverage["confirmation.advisory"] == pytest.approx(49 / 50)
        assert d.confirmation_pnl_delta["confirmation.advisory"] is not None
        assert d.confirmation_status["confirmation.advisory"] == "dead-weight"
        assert d.confirmation_status["confirmation.hard-gate"] == "unmeasurable"
        assert "drop-confirmation:confirmation.advisory" in d.suggestions
        assert "drop-confirmation:confirmation.hard-gate" not in d.suggestions

    def test_widen_and_tighten_stop_are_mutually_exclusive(self):
        # Scenario A: stop-dominant AND wide headroom -> widen-stop, never tighten-stop.
        wide = [
            make_record(record_id=str(i), outcome="stop", sl_headroom_ratio=1.5, sl_verdict="ok")
            for i in range(20)
        ] + [make_record(record_id=f"t{i}", outcome="target") for i in range(15)]
        d_wide = records.diagnose(wide, start_ms=START, end_ms=START + 400 * D_TRIG)
        assert "widen-stop" in d_wide.suggestions
        assert "tighten-stop" not in d_wide.suggestions

        # Scenario B: majority over-wide SL, but stop is NOT dominant -> tighten-stop only.
        narrow = [
            make_record(record_id=str(i), outcome="target", sl_headroom_ratio=0.1, sl_verdict="over-wide")
            for i in range(25)
        ] + [make_record(record_id=f"s{i}", outcome="stop", sl_headroom_ratio=0.9, sl_verdict="ok") for i in range(5)]
        d_narrow = records.diagnose(narrow, start_ms=START, end_ms=START + 400 * D_TRIG)
        assert "tighten-stop" in d_narrow.suggestions
        assert "widen-stop" not in d_narrow.suggestions

    def test_suggestions_are_from_the_closed_vocabulary(self):
        recs = [make_record(record_id=str(i), outcome="stop") for i in range(40)]
        d = records.diagnose(recs, start_ms=START, end_ms=START + 400 * D_TRIG)
        for s in d.suggestions:
            assert s.split(":", 1)[0] in records.SUGGESTIONS

    def test_below_min_trades_emits_only_insufficient_sample(self):
        recs = [make_record(record_id=str(i)) for i in range(5)]
        d = records.diagnose(recs, start_ms=START, end_ms=START + D_TRIG)
        assert d.suggestions == ("insufficient-sample",)

    def test_digest_is_stable_and_content_sensitive(self):
        recs = [make_record(record_id=str(i)) for i in range(40)]
        d1 = records.diagnose(recs, start_ms=START, end_ms=START + 400 * D_TRIG)
        d2 = records.diagnose(recs, start_ms=START, end_ms=START + 400 * D_TRIG)
        assert d1.digest == d2.digest
        recs2 = list(recs)
        recs2[0] = make_record(record_id="0", outcome="stop", pnl_pct=-0.5)
        d3 = records.diagnose(recs2, start_ms=START, end_ms=START + 400 * D_TRIG)
        assert d3.digest != d1.digest

    def test_empty_records_gives_insufficient_sample(self):
        d = records.diagnose([], start_ms=START, end_ms=START + D_TRIG)
        assert d.n_records == 0
        assert d.suggestions == ("insufficient-sample",)
        assert d.pace["sample_adequate"] is False


class TestOracleBoundary:
    """Contract §4's structural lock #2: a review that can run a backtest is
    a second fitness oracle. This test IS the lock."""

    _FORBIDDEN = ("backtest.walkforward", "backtest.engine", "backtest.trials",
                  "framework.execute", "evolution")

    @pytest.mark.parametrize("relpath", [
        "src/trading_bot/feedback/records.py",
        "src/trading_bot/feedback/versioning.py",
        "src/trading_bot/plugins/reviewers/trade_quality.py",
    ])
    def test_review_modules_do_not_import_the_oracle(self, relpath):
        repo_root = Path(__file__).resolve().parents[1]
        tree = ast.parse((repo_root / relpath).read_text(encoding="utf-8"))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
                names.extend(f"{node.module}.{alias.name}" for alias in node.names)
        for forbidden in self._FORBIDDEN:
            for name in names:
                assert forbidden not in name, (
                    f"{relpath} imports {name!r}, which names the forbidden "
                    f"oracle path {forbidden!r}"
                )
