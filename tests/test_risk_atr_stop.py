"""Tests for the Phase 2 pure ATR-stop / cost-ratio helper functions."""

import math

from trading_bot.risk.atr_stop import (
    compute_atr_stop,
    cost_ratio,
    net_rr,
    round_trip_cost_pct,
)


def test_long_stop_is_below_entry():
    assert compute_atr_stop(100.0, "long", atr_value=2.0, k=1.5) == 97.0


def test_short_stop_is_above_entry():
    assert compute_atr_stop(100.0, "short", atr_value=2.0, k=1.5) == 103.0


def test_cost_ratio_basic():
    # risk_pct=0.02, round-trip cost = 2*(0.0005+0.0002) = 0.0014 -> c = 0.07
    assert math.isclose(cost_ratio(0.02, 0.0005, 0.0002), 0.07)


def test_cost_ratio_zero_risk_is_infinite():
    assert cost_ratio(0.0, 0.0005, 0.0002) == float("inf")


def test_round_trip_cost_is_both_sides():
    # Two sides, each paying fee + slippage: 2 * (0.0005 + 0.0002).
    assert math.isclose(round_trip_cost_pct(0.0005, 0.0002), 0.0014)


def test_net_rr_is_below_gross_rr():
    # reward 2%, risk 1% -> gross rr = 2.00 exactly. Charging the 0.14%
    # round-trip cost to both legs: (0.02 - 0.0014) / (0.01 + 0.0014)
    #   = 0.0186 / 0.0114 = 31/19 = 1.6315789473684210...
    # Hand-derived as an exact rational, NOT by re-running the formula.
    assert math.isclose(
        net_rr(0.02, 0.01, 0.0005, 0.0002), 1.6315789473684210, rel_tol=1e-12
    )


def test_net_rr_is_zero_when_reward_exactly_equals_cost():
    # reward == cost -> numerator is 0, so the ratio is 0 regardless of risk.
    # This is the break-even knife edge: any RR_FLOOR > 0 rejects it.
    assert net_rr(0.0014, 0.01, 0.0005, 0.0002) == 0.0


def test_net_rr_is_negative_when_reward_cannot_cover_cost():
    # reward 0.10% < 0.14% cost -> a perfect win still loses money, so the
    # ratio must go negative and be rejected by any positive floor.
    #   (0.0010 - 0.0014) / (0.005 + 0.0014) = -0.0004 / 0.0064 = -1/16
    assert math.isclose(net_rr(0.0010, 0.005, 0.0005, 0.0002), -0.0625, rel_tol=1e-12)
