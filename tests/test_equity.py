"""Tests for the Phase 3 equity metrics module (Sharpe-first)."""

import math
import statistics

import pytest

from trading_bot.backtest.engine import Trade
from trading_bot.backtest.equity import (
    ATTRIBUTION_SPREAD,
    DAY_MS,
    compute_equity_metrics,
    daily_returns,
    deflated_sharpe,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe,
    sharpe_ratio,
    sortino_ratio,
)

SYMBOL = "BTCUSDT"
START = 1_700_000_000_000  # arbitrary epoch ms, day-aligned-ish


def make_trade(pnl, exit_ts, entry_ts=None):
    """entry_ts defaults to exit_ts — a ZERO-LENGTH hold, which is why almost
    all of this file survived the MEDIUM-5 attribution change unchanged."""
    return Trade(
        symbol=SYMBOL, regime="trending", pattern="flag", direction="long",
        entry_ts=exit_ts if entry_ts is None else entry_ts,
        entry=100.0, stop=99.0, target=101.0,
        exit_ts=exit_ts, exit_price=100.0, outcome="target",
        pnl_pct=pnl, volume_high=False,
    )


class TestDailyReturns:
    def test_zero_length_hold_books_on_its_single_day(self):
        """A trade whose entry and exit land on the same UTC day books there
        under BOTH attribution modes — which is why this class survived the
        MEDIUM-5 fix unchanged. Multi-day attribution is covered in
        tests/test_pnl_attribution.py."""
        start = 0
        end = 3 * DAY_MS
        trades = [make_trade(0.02, exit_ts=DAY_MS + 1234)]
        assert daily_returns(trades, start, end) == [0.0, 0.02, 0.0]

    def test_module_default_is_spread_and_a_multi_day_hold_splits(self):
        """Signpost (MEDIUM-5): exit-day booking is NO LONGER the convention.
        Do not conclude otherwise from the zero-length-hold tests above."""
        from trading_bot.backtest import equity

        assert equity.daily_returns.__defaults__ is None  # keyword-only
        start = 0
        end = 3 * DAY_MS
        trades = [make_trade(0.03, exit_ts=2 * DAY_MS + 10, entry_ts=10)]
        rets = daily_returns(trades, start, end)
        assert all(math.isclose(r, 0.01) for r in rets), rets
        assert daily_returns(trades, start, end, attribution=ATTRIBUTION_SPREAD) == rets

    def test_two_trades_same_day_are_summed(self):
        start = 0
        end = 3 * DAY_MS
        trades = [
            make_trade(0.01, exit_ts=DAY_MS + 100),
            make_trade(0.02, exit_ts=DAY_MS + 5000),
        ]
        rets = daily_returns(trades, start, end)
        assert math.isclose(rets[1], 0.03)
        assert rets[0] == 0.0
        assert rets[2] == 0.0

    def test_trade_outside_span_dropped(self):
        start = 0
        end = 3 * DAY_MS
        trades = [make_trade(0.05, exit_ts=10 * DAY_MS)]
        rets = daily_returns(trades, start, end)
        assert rets == [0.0, 0.0, 0.0]

    def test_empty_span_returns_empty(self):
        assert daily_returns([], 100, 100) == []
        assert daily_returns([], 100, 50) == []


class TestRatios:
    def test_sharpe_matches_hand_computed(self):
        # PRD success signal: closed-form literal, derived independently of
        # the implementation (no statistics.stdev / sqrt(365) call in this
        # test — see derivation below).
        #
        # rets = [0.01, -0.005, 0.02, 0.0, -0.01]
        #   mu = 0.015 / 5 = 0.003 exactly
        #   deviations = [0.007, -0.008, 0.017, -0.003, -0.013]
        #   squared deviations sum = 4.9e-5 + 6.4e-5 + 2.89e-4 + 9e-6 + 1.69e-4
        #                          = 5.8e-4 exactly
        #   sample variance (n-1=4) = 5.8e-4 / 4 = 1.45e-4 exactly
        #   sd = sqrt(1.45e-4) = 0.012041594578792296
        #   sharpe = (0.003 / sd) * sqrt(365) = 4.7597449946...
        rets = [0.01, -0.005, 0.02, 0.0, -0.01]
        expected = 4.7597449946
        assert math.isclose(sharpe_ratio(rets), expected, rel_tol=1e-9)

    def test_sharpe_zero_variance_is_none(self):
        assert sharpe_ratio([0.01, 0.01, 0.01]) is None

    def test_sharpe_single_obs_is_none(self):
        assert sharpe_ratio([0.01]) is None

    def test_sortino_no_downside_is_none(self):
        assert sortino_ratio([0.01, 0.02, 0.03]) is None

    def test_sortino_matches_hand_computed(self):
        # Pins the non-obvious convention documented in sortino_ratio's
        # docstring: downside deviation is sqrt(mean(min(r,0)**2)) over ALL
        # observations (denominator 5), NOT the stdev of only the losing
        # subset (which would divide by 2). An implementation that silently
        # switched to the losing-subset convention would NOT satisfy this
        # literal.
        #
        # rets = [0.01, -0.005, 0.02, 0.0, -0.01]; mu = 0.003 (as above)
        #   negative returns: -0.005, -0.01; squares = 2.5e-5, 1e-4
        #   sum over ALL 5 obs (wins contribute 0) = 1.25e-4
        #   mean over ALL 5 obs = 1.25e-4 / 5 = 2.5e-5
        #   dd = sqrt(2.5e-5) = 0.005 exactly
        #   sortino = (0.003 / 0.005) * sqrt(365) = 11.4629839047...
        rets = [0.01, -0.005, 0.02, 0.0, -0.01]
        expected = 11.4629839047
        assert math.isclose(sortino_ratio(rets), expected, rel_tol=1e-9)

    def test_max_drawdown_compounds_not_sums(self):
        # equity: 1.0 -> 1.10 -> 0.99; peak 1.10; dd = 1 - 0.99/1.10 = 0.1
        dd = max_drawdown([0.10, -0.10])
        assert math.isclose(dd, 1 - 0.99 / 1.10)
        assert math.isclose(dd, 0.1)

    def test_max_drawdown_constant_negative_returns(self):
        r = -0.01
        n = 5
        dd = max_drawdown([r] * n)
        expected = 1 - (1 + r) ** n
        assert math.isclose(dd, expected)

    def test_max_drawdown_empty_is_none(self):
        assert max_drawdown([]) is None


class TestDSR:
    def test_psr_at_benchmark_is_half(self):
        assert probabilistic_sharpe(sr=0.0, sr_benchmark=0.0, n_obs=100,
                                     skew=0.3, kurt=4.0) == pytest.approx(0.5)

    def test_expected_max_sharpe_single_trial_is_zero(self):
        assert expected_max_sharpe(1, 0.01) == 0.0

    def test_expected_max_sharpe_monotone_in_n_trials(self):
        low = expected_max_sharpe(2, 0.01)
        high = expected_max_sharpe(100, 0.01)
        assert high > low > 0.0

    def test_deflated_sharpe_penalizes_more_search(self):
        sr = 0.05
        n_obs = 100
        skew, kurt = 0.0, 3.0
        v = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
        sr_var = v / (n_obs - 1)
        dsr_1 = deflated_sharpe(sr, 1, n_obs, skew, kurt, sr_var=sr_var)
        dsr_100 = deflated_sharpe(sr, 100, n_obs, skew, kurt, sr_var=sr_var)
        assert dsr_100 < dsr_1

    def test_normal_fixture_denominator(self):
        # skew=0, kurt=3 -> denom = sqrt(1 + sr^2/2)
        sr = 0.1
        n_obs = 50
        expected_denom = math.sqrt(1 + sr * sr / 2)
        z_manual = (sr - 0.0) * math.sqrt(n_obs - 1) / expected_denom
        from statistics import NormalDist
        expected_psr = NormalDist().cdf(z_manual)
        assert math.isclose(
            probabilistic_sharpe(sr, 0.0, n_obs, skew=0.0, kurt=3.0), expected_psr
        )

    def test_psr_none_below_two_obs(self):
        assert probabilistic_sharpe(0.1, 0.0, 1, 0.0, 3.0) is None

    def test_deflated_sharpe_none_below_two_obs(self):
        assert deflated_sharpe(0.1, 10, 1, 0.0, 3.0) is None


class TestComputeEquityMetrics:
    def test_end_to_end_keys_present(self):
        start = 0
        end = 10 * DAY_MS
        trades = [
            make_trade(0.02, exit_ts=DAY_MS + 100),
            make_trade(-0.01, exit_ts=3 * DAY_MS + 100),
            make_trade(0.015, exit_ts=7 * DAY_MS + 100),
        ]
        em = compute_equity_metrics(trades, start, end)
        for key in ("n_days", "sharpe", "sortino", "max_drawdown_pct",
                    "ann_return_pct", "daily_sharpe", "dsr"):
            assert key in em
        assert em["n_days"] == 10
        assert em["ann_return_pct"] > 0  # net pnl is positive

    def test_end_to_end_negative_pnl_gives_negative_ann_return(self):
        start = 0
        end = 10 * DAY_MS
        trades = [
            make_trade(-0.02, exit_ts=DAY_MS + 100),
            make_trade(-0.01, exit_ts=3 * DAY_MS + 100),
        ]
        em = compute_equity_metrics(trades, start, end)
        assert em["ann_return_pct"] < 0

    def test_empty_trades_all_none_ratios(self):
        em = compute_equity_metrics([], 0, 10 * DAY_MS)
        assert em["sharpe"] is None
        assert em["sortino"] is None
        assert em["dsr"] is None

    def test_wipeout_series_gives_total_loss_not_blowup(self):
        # Two days of r = -3.0 each: factor = 1 + r = -2.0 per day. The old
        # code only checked the FINAL compounded equity > 0, so an even
        # number of such factors multiplies back positive (4.0) and the
        # power law equity**(365/n) - 1 explodes to an absurd positive
        # "annualized return" (~7.5e109 measured against the pre-fix logic)
        # instead of reporting the total loss it actually is. Note: the
        # PRP's suggested example of two days at -1.5 does NOT reproduce the
        # bug — factor = -0.5 has magnitude < 1, so its square underflows
        # toward 0 and the old formula coincidentally still lands on -1.0.
        # -3.0 (magnitude > 1 after the +1 shift) is required to demonstrate
        # the blowup.
        start = 0
        end = 2 * DAY_MS
        trades = [
            make_trade(-3.0, exit_ts=100),
            make_trade(-3.0, exit_ts=DAY_MS + 100),
        ]
        em = compute_equity_metrics(trades, start, end)
        assert em["ann_return_pct"] == -1.0
