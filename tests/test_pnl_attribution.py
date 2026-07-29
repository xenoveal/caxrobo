"""
Tests for MEDIUM-5: holding-day P&L attribution (v0.3.0 Phase 1).

KNOWN-LIMITATIONS §3, finding MEDIUM-5: equity.daily_returns booked each
trade's whole pnl_pct on its EXIT day, producing daily skew 3.79 / kurtosis
31.24 on 23 trades, which inflates PSR's denominator and blocks every
significance test. This file pins both the synthetic mechanics of the fix and
the measured before/after moments on the real OOS trades.
"""

import inspect
import math
from pathlib import Path

import pytest

from trading_bot import config
from trading_bot.backtest import engine
from trading_bot.backtest import equity as equity_module
from trading_bot.backtest.engine import BacktestParams, Trade, run_backtest
from trading_bot.backtest.equity import (
    ATTRIBUTION_EXIT_DAY,
    ATTRIBUTION_SPREAD,
    DAY_MS,
    _skew_kurt,
    compute_equity_metrics,
    daily_returns,
)
from trading_bot.data.storage import connect

SYMBOL = "BTCUSDT"


def trade(pnl, *, entry_ts, exit_ts):
    return Trade(
        symbol=SYMBOL, regime="trending", pattern="flag", direction="long",
        entry_ts=entry_ts, entry=100.0, stop=99.0, target=101.0,
        exit_ts=exit_ts, exit_price=100.0, outcome="target",
        pnl_pct=pnl, volume_high=False,
    )


class TestSpreadAttribution:
    def test_three_day_hold_splits_evenly(self):
        trades = [trade(0.03, entry_ts=10, exit_ts=2 * DAY_MS + 10)]
        rets = daily_returns(trades, 0, 4 * DAY_MS)
        assert len(rets) == 4
        for r in rets[:3]:
            assert math.isclose(r, 0.01)
        assert rets[3] == 0.0

    def test_same_day_hold_is_identical_to_exit_day(self):
        trades = [trade(0.02, entry_ts=DAY_MS + 1, exit_ts=DAY_MS + 999)]
        assert daily_returns(trades, 0, 3 * DAY_MS) == daily_returns(
            trades, 0, 3 * DAY_MS, attribution=ATTRIBUTION_EXIT_DAY
        )

    def test_sum_is_conserved(self):
        """THE invariant: attribution redistributes P&L across days, it never
        creates or destroys any. Measured identical to 17 significant figures on
        the real 23 OOS trades (0.16509935688760607 both ways)."""
        trades = [
            trade(0.03, entry_ts=10, exit_ts=2 * DAY_MS + 10),
            trade(-0.01, entry_ts=DAY_MS + 10, exit_ts=4 * DAY_MS + 10),
        ]
        span = (0, 6 * DAY_MS)
        spread = sum(daily_returns(trades, *span))
        exit_day = sum(daily_returns(trades, *span, attribution=ATTRIBUTION_EXIT_DAY))
        assert math.isclose(spread, 0.02)
        assert math.isclose(spread, exit_day)

    def test_entry_before_span_conserves_whole_pnl(self):
        """A trade entered before the window still contributes its WHOLE
        pnl_pct, divided by its IN-SPAN days only — otherwise total reported
        P&L would depend on where the window is cut."""
        trades = [trade(0.02, entry_ts=-2 * DAY_MS + 10, exit_ts=DAY_MS + 10)]
        rets = daily_returns(trades, 0, 3 * DAY_MS)
        assert math.isclose(sum(rets), 0.02)
        assert rets[2] == 0.0

    def test_exit_after_span_now_contributes(self):
        """BEHAVIOR CHANGE, deliberate: exit-day mode DROPPED a trade whose
        exit fell past end_ms even though the trade was open during the window.
        Spread mode books its in-span days — a trade open during the window did
        affect equity during the window."""
        trades = [trade(0.02, entry_ts=10, exit_ts=10 * DAY_MS)]
        span = (0, 3 * DAY_MS)
        assert math.isclose(sum(daily_returns(trades, *span)), 0.02)
        assert daily_returns(trades, *span, attribution=ATTRIBUTION_EXIT_DAY) == [0.0] * 3

    def test_no_overlap_is_dropped(self):
        trades = [trade(0.05, entry_ts=9 * DAY_MS, exit_ts=10 * DAY_MS)]
        assert daily_returns(trades, 0, 3 * DAY_MS) == [0.0, 0.0, 0.0]

    def test_unknown_mode_raises(self):
        """A silent fallback would make the gate's definition of Sharpe depend
        on a typo."""
        with pytest.raises(ValueError):
            daily_returns([], 0, DAY_MS, attribution="exitday")

    def test_spread_lowers_kurtosis_on_a_synthetic_spiky_series(self):
        trades = [
            trade(0.05, entry_ts=i * 8 * DAY_MS, exit_ts=i * 8 * DAY_MS + 3 * DAY_MS)
            for i in range(5)
        ]
        span = (0, 40 * DAY_MS)
        _, kurt_spread = _skew_kurt(daily_returns(trades, *span))
        _, kurt_exit = _skew_kurt(
            daily_returns(trades, *span, attribution=ATTRIBUTION_EXIT_DAY)
        )
        assert kurt_spread < kurt_exit


class TestConfigDefault:
    def test_config_default_mode_is_spread(self):
        assert config.PNL_ATTRIBUTION_MODE == ATTRIBUTION_SPREAD

    def test_metrics_report_the_moments_and_the_mode(self):
        trades = [trade(0.02, entry_ts=10, exit_ts=2 * DAY_MS + 10)]
        em = compute_equity_metrics(trades, 0, 10 * DAY_MS)
        for key in ("attribution", "skew", "kurtosis"):
            assert key in em
        assert em["attribution"] == ATTRIBUTION_SPREAD
        assert compute_equity_metrics(
            trades, 0, 10 * DAY_MS, attribution=ATTRIBUTION_EXIT_DAY
        )["attribution"] == ATTRIBUTION_EXIT_DAY

    def test_equity_module_has_no_config_coupling(self):
        """equity.py's docstring promises "no config coupling", so the default
        is the MODULE constant; config.PNL_ATTRIBUTION_MODE is read by the
        already-config-coupled callers (walkforward.py, cli.py)."""
        src = Path(inspect.getsourcefile(equity_module)).read_text()
        assert "config." not in src
        assert "import config" not in src


@pytest.mark.skipif(
    not Path(config.DB_PATH).exists(),
    reason="requires the real OHLCV store (data/ohlcv.db)",
)
class TestMedium5MeasuredDelta:
    """The finding, pinned against the REAL 23 OOS trades KNOWN-LIMITATIONS §3
    measured. Parameters below are §1's GATE-SELECTED ones: change any and the
    trade count leaves 23 and the pinned moments become meaningless."""

    OOS_END = config.date_to_ms("2026-07-26")
    OOS_START = OOS_END - config.WF_OOS_DAYS * DAY_MS

    @pytest.fixture(scope="class")
    @classmethod
    def trades(cls):
        engine.clear_caches()
        conn = connect()
        out = []
        for symbol in config.SYMBOLS:
            out.extend(
                run_backtest(
                    conn, symbol,
                    start_ms=cls.OOS_START, end_ms=cls.OOS_END,
                    params=BacktestParams(trail_enabled=False, target_enabled=False),
                    max_hold_bars=48,
                )
            )
        conn.close()
        return out

    def test_exit_day_reproduces_the_published_moments(self, trades):
        """MEDIUM-5's "before": §3's skew 3.79 / kurtosis 31.24, to 4 s.f.
        Reproducing it is the proof this harness is wired to the same trades."""
        assert len(trades) == 23
        rets = daily_returns(
            trades, self.OOS_START, self.OOS_END, attribution=ATTRIBUTION_EXIT_DAY
        )
        skew, kurt = _skew_kurt(rets)
        assert skew == pytest.approx(3.7883, rel=1e-3)
        assert kurt == pytest.approx(31.2449, rel=1e-3)
        assert sum(1 for r in rets if r == 0.0) == 73

    def test_spread_drops_kurtosis_below_thirty(self, trades):
        """MEDIUM-5's "after". The PRD success signal is literally "no longer
        30+ kurtosis"; it does NOT reach normality (3.0) and must not be
        claimed to — holds are 1-3 calendar days, so there is only so much
        smearing available."""
        rets = daily_returns(trades, self.OOS_START, self.OOS_END)
        skew, kurt = _skew_kurt(rets)
        assert skew == pytest.approx(1.4034, rel=1e-3)
        assert kurt == pytest.approx(15.5429, rel=1e-3)
        assert kurt < 30.0

    def test_total_pnl_is_identical_across_modes(self, trades):
        span = (self.OOS_START, self.OOS_END)
        assert sum(daily_returns(trades, *span)) == pytest.approx(
            sum(daily_returns(trades, *span, attribution=ATTRIBUTION_EXIT_DAY)),
            rel=1e-12,
        )

    def test_spread_raises_sharpe_and_dsr_still_fails(self, trades):
        """The fix moves Sharpe in the FLATTERING direction and does NOT rescue
        DSR; both recorded so neither is discovered later as a surprise."""
        span = (self.OOS_START, self.OOS_END)
        old = compute_equity_metrics(
            trades, *span, 36, attribution=ATTRIBUTION_EXIT_DAY
        )
        new = compute_equity_metrics(trades, *span, 36)
        assert old["sharpe"] == pytest.approx(1.1750, rel=1e-3)
        assert new["sharpe"] == pytest.approx(1.5120, rel=1e-3)
        assert new["sharpe"] > old["sharpe"]
        assert old["dsr"] < 0.95
        assert new["dsr"] < 0.95
