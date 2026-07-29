"""
Tests for the buy-and-hold null (v0.3.0 Phase 1).

Includes THE ACCEPTANCE ANCHOR: TestKnownLimitationsSection0Anchor reproduces
the per-symbol and equal-weight-basket table already published in
KNOWN-LIMITATIONS §0 from stored 1d bars, network-free. That reproduction is
the strongest available check that this benchmark path is honest.
"""

import sys
from pathlib import Path

import pytest

from trading_bot import cli, config
from trading_bot.backtest.benchmark import (
    METRIC_KEYS,
    REBALANCE_DAILY,
    REBALANCE_NONE,
    BenchmarkResult,
    _metrics_from_returns,
    buy_and_hold,
)
from trading_bot.data.storage import TIMEFRAME_MS, connect, upsert_candles

# Tier-derived, never hardcoded: the fixtures follow config forever.
BENCH_TF = config.BENCHMARK_TIMEFRAME
D_BENCH = TIMEFRAME_MS[BENCH_TF]
START = 1_700_000_000_000
ROUND_TRIP = 2.0 * (config.FEE_PCT + config.SLIPPAGE_PCT)


def seed_closes(conn, symbol, closes, start_ms=START):
    """One BENCH_TF bar per close, with o=h=l=c.

    Deliberately degenerate bars: with distinct highs/lows an off-by-one on the
    close column index (load_candles returns tuples, close is index 4) would
    still produce plausible numbers and the bug would ship.
    """
    rows = [
        [start_ms + i * D_BENCH, c, c, c, c, 1.0] for i, c in enumerate(closes)
    ]
    upsert_candles(conn, symbol, BENCH_TF, rows)


@pytest.fixture
def db(tmp_path):
    conn = connect(str(tmp_path / "ohlcv.db"))
    yield conn
    conn.close()


SPAN = dict(start_ms=START, end_ms=START + 100 * D_BENCH)


class TestSyntheticSingleSymbol:
    def test_doubling_over_two_bars(self, db):
        seed_closes(db, "BTCUSDT", [100.0, 200.0])
        m = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        # One return of +100%, then ONE round-trip of fees on that first day.
        assert m["total_return"] == pytest.approx(2.0 * (1.0 - ROUND_TRIP))
        assert m["n_days"] == 1

    def test_flat_series_has_no_sharpe(self, db, monkeypatch):
        """Zero variance -> None, never a fabricated 0.0 (metrics.py's rule).
        Fees off: the round-trip is booked on day 0, which would itself make an
        otherwise-flat series non-constant."""
        monkeypatch.setattr(config, "BENCHMARK_CHARGE_FEES", False)
        seed_closes(db, "BTCUSDT", [100.0] * 5)
        m = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        assert m["sharpe"] is None
        assert m["sortino"] is None

    def test_single_bar_is_all_none(self, db):
        seed_closes(db, "BTCUSDT", [100.0])
        m = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        assert m["n_days"] == 0
        assert all(m[k] is None for k in METRIC_KEYS if k != "n_days")

    def test_missing_symbol_is_all_none(self, db):
        m = buy_and_hold(db, ["NOPEUSDT"], **SPAN).per_symbol["NOPEUSDT"]
        assert m["n_days"] == 0
        assert m["total_return"] is None

    def test_n_days_is_bars_minus_one(self, db):
        """load_candles' bounds are BOTH INCLUSIVE, so N bars give N-1 returns.
        Do not subtract a day "to make it exclusive"."""
        seed_closes(db, "BTCUSDT", [100.0 + i for i in range(11)])
        m = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        assert m["n_days"] == 10

    def test_wipeout_gives_total_loss_not_a_blowup(self):
        """Mirrors tests/test_equity.py's wipe-out guard: an even number of
        non-positive daily factors would otherwise multiply back to a
        spuriously large positive equity (4.0 for two days of r = -3.0) and the
        power law would explode to an absurd positive annualized return.

        Exercised on _metrics_from_returns directly: a return <= -1.0 is
        UNREACHABLE from positive stored closes, so seeding bars cannot get here
        (a zero close is refused as corrupt — see
        test_non_positive_close_is_refused_not_divided_by)."""
        m = _metrics_from_returns([-3.0, -3.0])
        assert m["ann_return_pct"] == -1.0
        assert m["total_return"] == 0.0

    def test_non_positive_close_is_refused_not_divided_by(self, db):
        """A zero close is corrupt data, not a price: refuse (all-None, which
        FAILS the gate) rather than raise ZeroDivisionError mid-run."""
        seed_closes(db, "BTCUSDT", [100.0, 0.0, 100.0])
        m = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        assert m["n_days"] == 0
        assert m["total_return"] is None


class TestBasketConstruction:
    def test_daily_rebalance_is_the_mean_of_daily_returns(self, db, monkeypatch):
        monkeypatch.setattr(config, "BENCHMARK_CHARGE_FEES", False)
        monkeypatch.setattr(config, "BENCHMARK_REBALANCE", REBALANCE_DAILY)
        seed_closes(db, "AAAUSDT", [100.0, 110.0, 121.0])  # +10%, +10%
        seed_closes(db, "BBBUSDT", [100.0, 100.0, 100.0])  # 0%, 0%
        basket = buy_and_hold(db, ["AAAUSDT", "BBBUSDT"], **SPAN).basket
        # mean(+0.10, 0.0) = +0.05 each day -> 1.05^2
        assert basket["total_return"] == pytest.approx(1.05 ** 2)

    def test_buy_once_hold_differs_from_daily_rebalance(self, db, monkeypatch):
        monkeypatch.setattr(config, "BENCHMARK_CHARGE_FEES", False)
        seed_closes(db, "AAAUSDT", [100.0, 110.0, 121.0])
        seed_closes(db, "BBBUSDT", [100.0, 100.0, 100.0])
        syms = ["AAAUSDT", "BBBUSDT"]
        monkeypatch.setattr(config, "BENCHMARK_REBALANCE", REBALANCE_DAILY)
        daily = buy_and_hold(db, syms, **SPAN).basket["total_return"]
        monkeypatch.setattr(config, "BENCHMARK_REBALANCE", REBALANCE_NONE)
        once = buy_and_hold(db, syms, **SPAN).basket["total_return"]
        # buy-once: 0.5*1.21 + 0.5*1.0 = 1.105; daily-rebalanced: 1.05^2 = 1.1025
        assert once == pytest.approx(1.105)
        assert daily == pytest.approx(1.1025)
        assert once != pytest.approx(daily)

    def test_unknown_rebalance_mode_raises(self, db, monkeypatch):
        monkeypatch.setattr(config, "BENCHMARK_REBALANCE", "weekly")
        with pytest.raises(ValueError):
            buy_and_hold(db, ["BTCUSDT"], **SPAN)

    def test_symbol_without_data_is_excluded_not_zero_padded(self, db, monkeypatch):
        """A 2-symbol honest basket beats a 3-symbol one padded with zeros."""
        monkeypatch.setattr(config, "BENCHMARK_CHARGE_FEES", False)
        seed_closes(db, "AAAUSDT", [100.0, 110.0])
        seed_closes(db, "BBBUSDT", [100.0, 110.0])
        b = buy_and_hold(db, ["AAAUSDT", "BBBUSDT", "NOPEUSDT"], **SPAN)
        assert b.per_symbol["NOPEUSDT"]["total_return"] is None
        assert b.basket["total_return"] == pytest.approx(1.10)  # not diluted to 1.0667

    def test_ragged_bar_counts_leave_the_basket_undefined(self, db):
        """Refuse rather than zip-truncate: truncating would silently shorten
        the span the gate compares on."""
        seed_closes(db, "AAAUSDT", [100.0, 110.0, 120.0])
        seed_closes(db, "BBBUSDT", [100.0, 110.0])
        basket = buy_and_hold(db, ["AAAUSDT", "BBBUSDT"], **SPAN).basket
        assert basket["n_days"] == 0
        assert basket["sharpe"] is None
        assert basket["total_return"] is None


class TestCostAssumption:
    def test_round_trip_fee_is_charged_exactly_once(self, db, monkeypatch):
        seed_closes(db, "BTCUSDT", [100.0] * 6 + [200.0])
        with_fee = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        monkeypatch.setattr(config, "BENCHMARK_CHARGE_FEES", False)
        no_fee = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        assert no_fee["total_return"] == pytest.approx(2.0)
        assert with_fee["total_return"] == pytest.approx(2.0 * (1.0 - ROUND_TRIP))

    def test_funding_does_not_move_the_benchmark(self, db, monkeypatch):
        """The null is a SPOT-EQUIVALENT hold. Charging the frozen pessimistic
        FUNDING_PCT_PER_DAY over 1095 days would cost it ~11.6% and hand the
        strategy an ~11-point head start a real operator could dodge by buying
        spot."""
        seed_closes(db, "BTCUSDT", [100.0 + i for i in range(30)])
        before = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        monkeypatch.setattr(config, "FUNDING_PCT_PER_DAY", config.FUNDING_PCT_PER_DAY * 100)
        after = buy_and_hold(db, ["BTCUSDT"], **SPAN).per_symbol["BTCUSDT"]
        assert after == before


class TestBenchmarkCli:
    def _stub(self, monkeypatch, result):
        monkeypatch.setattr(cli, "buy_and_hold", lambda conn, syms, **kw: result)

    def test_prints_basket_and_exits_zero(self, monkeypatch, capsys):
        full = dict(zip(METRIC_KEYS, (2.0, 0.3, 0.8, 1.2, 0.5, 1095)))
        self._stub(
            monkeypatch,
            BenchmarkResult(per_symbol={"BTCUSDT": dict(full)}, basket=dict(full),
                            start_ms=0, end_ms=1),
        )
        code = cli._benchmark_command(None, ["BTCUSDT"], start_ms=0, end_ms=1)
        out = capsys.readouterr().out
        assert code == 0
        assert "BASKET" in out
        assert "2.0000x" in out
        assert "NOTE:" not in out  # 1095 days >= 365, no extrapolation warning

    def test_short_span_prints_the_extrapolation_note(self, monkeypatch, capsys):
        short = dict(zip(METRIC_KEYS, (1.02, 0.9, 0.5, 0.7, 0.02, 6)))
        self._stub(
            monkeypatch,
            BenchmarkResult(per_symbol={"BTCUSDT": dict(short)}, basket=dict(short),
                            start_ms=0, end_ms=1),
        )
        assert cli._benchmark_command(None, ["BTCUSDT"], start_ms=0, end_ms=1) == 0
        assert "NOTE: ann is a 6-day extrapolation" in capsys.readouterr().out

    def test_incomplete_bundle_exits_one(self, monkeypatch, capsys):
        empty = {**dict.fromkeys(METRIC_KEYS, None), "n_days": 0}
        self._stub(
            monkeypatch,
            BenchmarkResult(per_symbol={"BTCUSDT": dict(empty)}, basket=dict(empty),
                            start_ms=0, end_ms=1),
        )
        assert cli._benchmark_command(None, ["BTCUSDT"], start_ms=0, end_ms=1) == 1
        assert "INCOMPLETE" in capsys.readouterr().out

    def test_main_benchmark_argv(self, tmp_path, monkeypatch, capsys):
        """Drives cli.main() through sys.argv to exercise the argparse wiring,
        not just the inner helper. An empty DB has no bars -> exit 1."""
        db_path = tmp_path / "empty.db"
        monkeypatch.setattr(
            sys, "argv",
            ["trading-bot", "--db", str(db_path), "benchmark",
             "--symbol", "BTCUSDT", "--start", "2023-07-27", "--end", "2026-07-26"],
        )
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "INCOMPLETE" in capsys.readouterr().out


@pytest.mark.skipif(
    not Path(config.DB_PATH).exists(),
    reason="requires the real OHLCV store (data/ohlcv.db)",
)
class TestKnownLimitationsSection0Anchor:
    """THE ACCEPTANCE TEST: reproduce KNOWN-LIMITATIONS §0 from stored bars.

    TOLERANCES, each ONE ROUNDING UNIT of the published figure — the tightest
    bound that cannot fail on the fee-booking choice (measured no-fee vs
    with-fee differ by at most 0.0031 on total_return and 0.0010 on Sharpe).
    Tighter would pin an assumption the published table never made; looser
    would stop being evidence.
        total_return      abs=0.01   ("2.21x"  is 2 d.p.)
        ann_return_pct    abs=0.002  ("+30.3%" is 0.1 pp)
        sharpe            abs=0.01   ("0.80"   is 2 d.p.)
        max_drawdown_pct  abs=0.001  ("53.0%"  is 0.1 pp)
    """

    SPAN = dict(
        start_ms=config.date_to_ms("2023-07-27"),
        end_ms=config.date_to_ms("2026-07-26"),
    )

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        conn = connect()
        try:
            return buy_and_hold(conn, list(config.SYMBOLS), **cls.SPAN)
        finally:
            conn.close()

    def test_span_is_1095_daily_returns(self, result):
        """1096 inclusive bars -> 1095 returns -> 365/1095 = 1/3 EXACTLY, so
        §0's annualized column is a true 3-year CAGR, NOT an extrapolation."""
        assert result.basket["n_days"] == 1095
        for m in result.per_symbol.values():
            assert m["n_days"] == 1095

    @pytest.mark.parametrize(
        "symbol,total,ann,sharpe,dd",
        [
            ("BTCUSDT", 2.21, 0.303, 0.80, 0.530),
            ("ETHUSDT", 1.03, 0.009, 0.34, 0.676),
            ("SOLUSDT", 3.00, 0.443, 0.85, 0.763),
        ],
    )
    def test_per_symbol_reproduces_section_0(self, result, symbol, total, ann, sharpe, dd):
        m = result.per_symbol[symbol]
        assert m["total_return"] == pytest.approx(total, abs=0.01)
        assert m["ann_return_pct"] == pytest.approx(ann, abs=0.002)
        assert m["sharpe"] == pytest.approx(sharpe, abs=0.01)
        assert m["max_drawdown_pct"] == pytest.approx(dd, abs=0.001)

    def test_basket_reproduces_section_0(self, result):
        b = result.basket
        assert b["total_return"] == pytest.approx(2.17, abs=0.01)
        assert b["ann_return_pct"] == pytest.approx(0.294, abs=0.002)
        assert b["sharpe"] == pytest.approx(0.73, abs=0.01)
        assert b["max_drawdown_pct"] == pytest.approx(0.643, abs=0.001)

    def test_buy_once_hold_does_NOT_reproduce_section_0(self, monkeypatch):
        """The measurement that DECIDED the basket construction, pinned so a
        future "simplification" to buy-once-hold cannot pass the anchor above."""
        monkeypatch.setattr(config, "BENCHMARK_REBALANCE", REBALANCE_NONE)
        conn = connect()
        try:
            b = buy_and_hold(conn, list(config.SYMBOLS), **self.SPAN).basket
        finally:
            conn.close()
        assert b["total_return"] == pytest.approx(2.081, abs=0.01)
        assert b["total_return"] != pytest.approx(2.17, abs=0.01)

    def test_ninety_day_oos_benchmark_is_negative(self):
        """§0's other half: over the gate's holdout the null LOSES, which is why
        v0.2.0 passes both new gate conditions while still failing on sample
        size and DSR. Pinned so that fact is a test, not a memory."""
        end_ms = self.SPAN["end_ms"]
        start_ms = end_ms - config.WF_OOS_DAYS * 86_400_000
        conn = connect()
        try:
            b = buy_and_hold(conn, list(config.SYMBOLS), start_ms=start_ms, end_ms=end_ms).basket
        finally:
            conn.close()
        assert b["sharpe"] == pytest.approx(-1.303, abs=0.01)
        assert b["ann_return_pct"] < 0
