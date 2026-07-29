"""Tests for filter.rr-after-costs (v0.3.0 Phase 4).

The load-bearing test is TestIdentity: the whole phase rests on the algebra
`net_rr >= X  <=>  gross_rr >= X + (X + 1) * c`, and that identity is asserted
exactly (1e-12), not approximately. A loose tolerance would hide an `(X+1)`
vs `X` error at small c, which is exactly where the two expressions nearly agree.

The filter is a pure function of a PositionPlan and the cost constants, so these
tests need no EvalContext and no database; `None` is passed as ctx deliberately,
which also pins that the filter never consults market data (a risk screen that
did could be fitted to it).
"""

import math

import pytest

from trading_bot import config
from trading_bot.framework.contracts import FilterVerdict, PositionPlan
from trading_bot.plugins.filters import rr_after_costs
from trading_bot.plugins.filters.rr_after_costs import (
    RR_REPORT_THRESHOLDS,
    rr_after_costs as accept,
    rr_distribution_report,
)
from trading_bot.risk.atr_stop import cost_ratio, net_rr

FEE = config.FEE_PCT
SLIP = config.SLIPPAGE_PCT
DEFAULTS = dict(
    rr_target_min=config.RR_TARGET_MIN, fee_pct=FEE, slippage_pct=SLIP
)


def plan(*, risk_pct, reward_pct, symbol="BTCUSDT", source="donchian-breakout"):
    """A PositionPlan carrying only the fields the filter reads.

    entry/stop/target are set consistently with risk_pct/reward_pct for a long so
    the fixture is not self-contradictory, but the filter reads only risk_pct,
    reward_pct and rr.
    """
    entry = 100.0
    return PositionPlan(
        symbol=symbol,
        ts=1_700_000_000_000,
        direction="long",
        entry=entry,
        stop=entry * (1.0 - risk_pct),
        target=entry * (1.0 + reward_pct),
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr=(reward_pct / risk_pct) if risk_pct else float("inf"),
        source=source,
    )


class TestIdentity:
    """`net_rr >= X` is exactly `gross_rr >= X + (X + 1) * c`.

    Construct reward_pct = (X + (X+1)*c) * risk_pct and net_rr must come out at
    EXACTLY X. Derivation:
        net = (reward - cost) / (risk + cost)
        reward = X*risk + (X+1)*cost
          => net = (X*risk + (X+1)*cost - cost) / (risk + cost)
                 = (X*risk + X*cost) / (risk + cost)
                 = X
    """

    @pytest.mark.parametrize("X", [1.5, 2.0])
    @pytest.mark.parametrize("risk_pct", [0.005, 0.01968, 0.02679, 0.03752, 0.10])
    def test_identity_is_exact_to_1e_12(self, X, risk_pct):
        """The algebra itself, to 1e-12.

        NOTE ON THE EXACT BOUNDARY, measured rather than assumed: in exact
        arithmetic this construction gives net_rr == X, but in IEEE-754 the
        round-trip through (X + (X+1)*c) * risk_pct lands a few ulps either side
        of X depending on c, so `net_rr >= X` is NOT reliably True at the
        constructed boundary (measured False at risk_pct 0.005/X=2.0 and at
        risk_pct 0.10 for both X). That is a property of floating point, not of
        the filter, so this test asserts the ALGEBRA to 1e-12 and the two tests
        below assert the DECISION just above and just below the boundary. An
        exact-equality boundary case that does land on X exactly is covered by
        TestWhyNet::test_boundary_equality_accepts.
        """
        c = cost_ratio(risk_pct, FEE, SLIP)
        reward_pct = (X + (X + 1.0) * c) * risk_pct
        got = net_rr(reward_pct, risk_pct, FEE, SLIP)
        assert abs(got - X) < 1e-12, f"net_rr {got!r} != {X!r} (c={c!r})"

        v = accept(None, plan(risk_pct=risk_pct, reward_pct=reward_pct),
                   rr_target_min=X, fee_pct=FEE, slippage_pct=SLIP)
        assert abs(v.measured["net_rr"] - X) < 1e-12
        assert abs(v.measured["gross_rr_required"] - (X + (X + 1.0) * c)) < 1e-12

    @pytest.mark.parametrize("X", [1.5, 2.0])
    @pytest.mark.parametrize("risk_pct", [0.005, 0.01968, 0.02679, 0.03752, 0.10])
    def test_nudging_the_reward_up_accepts(self, X, risk_pct):
        c = cost_ratio(risk_pct, FEE, SLIP)
        reward_pct = (X + (X + 1.0) * c) * risk_pct * (1.0 + 1e-9)
        v = accept(None, plan(risk_pct=risk_pct, reward_pct=reward_pct),
                   rr_target_min=X, fee_pct=FEE, slippage_pct=SLIP)
        assert v.accepted is True

    @pytest.mark.parametrize("X", [1.5, 2.0])
    @pytest.mark.parametrize("risk_pct", [0.005, 0.01968, 0.03752])
    def test_shaving_the_reward_rejects(self, X, risk_pct):
        c = cost_ratio(risk_pct, FEE, SLIP)
        reward_pct = (X + (X + 1.0) * c) * risk_pct - 1e-9 * risk_pct
        v = accept(None, plan(risk_pct=risk_pct, reward_pct=reward_pct),
                   rr_target_min=X, fee_pct=FEE, slippage_pct=SLIP)
        assert v.accepted is False


class TestRecordedRegression:
    """Pins backtest/walkforward.py's measured minimum planned R:R of 1.56.

    walkforward.py records that the MINIMUM planned GROSS R:R across every trade
    the v0.2.0 engine ever took was 1.56, and that 100% of trades cleared both
    1.25 and 1.5. At BTC's measured median risk_pct (1.968%, config.py:166-168)
    that trade's net_rr is 1.3900 -- it clears neither 2.0 NOR 1.5 net. The
    1.5-GROSS floor was measured non-binding; a 2.0-NET floor is a different bar.
    """

    @pytest.mark.parametrize(
        "risk_pct,expected_net",
        [(0.01968, 1.389981), (0.02679, 1.432863), (0.03752, 1.467914)],
        ids=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )
    def test_gross_1_56_fails_at_2_0_and_at_1_5_net(self, risk_pct, expected_net):
        p = plan(risk_pct=risk_pct, reward_pct=1.56 * risk_pct)
        assert math.isclose(p.rr, 1.56, rel_tol=0.0, abs_tol=1e-12)
        v = accept(None, p, **DEFAULTS)
        assert math.isclose(v.measured["net_rr"], expected_net, rel_tol=0.0, abs_tol=1e-6)
        assert v.accepted is False
        v15 = accept(None, p, rr_target_min=1.5, fee_pct=FEE, slippage_pct=SLIP)
        assert v15.accepted is False

    @pytest.mark.parametrize(
        "risk_pct,required",
        [(0.01968, 2.213415), (0.02679, 2.156775), (0.03752, 2.111940)],
        ids=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )
    def test_per_symbol_required_gross_table(self, risk_pct, required):
        """The 2.2134 / 2.1568 / 2.1119 prediction, pinned at the three recorded
        median risk_pct values from config.py:166-168."""
        v = accept(None, plan(risk_pct=risk_pct, reward_pct=3.0 * risk_pct), **DEFAULTS)
        assert math.isclose(
            v.measured["gross_rr_required"], required, rel_tol=0.0, abs_tol=1e-3
        )


class TestWhyNet:
    def test_the_dimensionless_trap_in_one_test(self, ):
        """risk/atr_stop.py:52-56's scenario: "a setup risking 0.05% to make 0.10%
        scores a healthy 2.0 while round-trip cost exceeds the entire reward."

        This single assertion is the clearest possible demonstration of why the
        filter is net: gross clears 2.0 and the plan is still rejected.
        """
        p = plan(risk_pct=0.0005, reward_pct=0.0010)
        assert p.rr >= 2.0  # a "healthy" GROSS ratio
        v = accept(None, p, **DEFAULTS)
        assert v.accepted is False
        assert v.measured["net_rr"] <= 0.0  # reward does not even cover cost
        assert v.measured["cost_ratio"] > 2.0  # cost is >2x the risk unit

    def test_boundary_equality_accepts(self):
        """`>=`, matching setup.py:154's `rr < floor` rejection semantics."""
        risk_pct = 0.02
        c = cost_ratio(risk_pct, FEE, SLIP)
        X = config.RR_TARGET_MIN
        reward_pct = (X + (X + 1.0) * c) * risk_pct
        v = accept(None, plan(risk_pct=risk_pct, reward_pct=reward_pct), **DEFAULTS)
        assert v.measured["net_rr"] == pytest.approx(X, abs=1e-12)
        assert v.accepted is True


class TestDegenerate:
    def test_zero_risk_pct_gives_infinite_cost_ratio_without_raising(self):
        """cost_ratio returns +inf at risk_pct <= 0 (risk/atr_stop.py:98-99).

        MEASURED GAP, pinned rather than papered over: at risk_pct == 0 this
        filter ACCEPTS. net_rr(0.05, 0, ...) = (0.05 - 0.0014) / 0.0014 = 34.71,
        because net_rr's `risk + cost` denominator floors the risk at the cost
        term, so a degenerate zero-risk plan scores spectacularly instead of
        being rejected. cost_ratio correctly reports +inf alongside it, and the
        two disagree.

        This is UNREACHABLE THROUGH THE PIPELINE: policy.measured-move rejects
        `risk <= 0` before any filter runs
        (tests/test_plugins_policies.py::TestMeasuredMoveRejections), so no
        zero-risk plan ever reaches here. It is pinned as the actual behavior
        rather than "fixed" because Task 10's GOTCHA 1 forbids adding floors to
        this filter, and because a reader who assumes the filter guards its own
        denominator should be corrected by a test rather than by a surprise.
        """
        v = accept(None, plan(risk_pct=0.0, reward_pct=0.05), **DEFAULTS)
        assert v.accepted is True, "documented gap: net_rr floors risk at cost"
        assert math.isinf(v.measured["cost_ratio"])
        assert math.isinf(v.measured["gross_rr_required"])
        assert isinstance(v.reason, str) and v.reason
        assert v.measured["risk_pct"] == 0.0

    def test_negative_costs_give_minus_inf_net_rr_without_raising(self):
        """net_rr returns -inf when adjusted risk is non-positive
        (risk/atr_stop.py:74-75). `-inf >= 2.0` is False, which is correct, but
        the reason f-string must not crash formatting it."""
        v = accept(None, plan(risk_pct=0.01, reward_pct=0.05),
                   rr_target_min=2.0, fee_pct=-0.01, slippage_pct=0.0)
        assert v.accepted is False
        assert math.isinf(v.measured["net_rr"]) and v.measured["net_rr"] < 0
        assert isinstance(v.reason, str) and "-inf" in v.reason

    def test_measured_is_all_floats(self):
        """Contract §3 makes `measured` a Mapping[str, float]: no strings, no None."""
        v = accept(None, plan(risk_pct=0.02, reward_pct=0.06), **DEFAULTS)
        assert isinstance(v, FilterVerdict)
        assert v.measured
        for key, value in v.measured.items():
            assert isinstance(key, str)
            assert isinstance(value, float), f"{key} is {type(value).__name__}"


class TestVerdictShape:
    def test_reason_prints_both_ratios_and_the_required_gross_floor(self):
        v = accept(None, plan(risk_pct=0.02, reward_pct=0.03), **DEFAULTS)
        assert v.accepted is False
        assert "net_rr" in v.reason and "gross" in v.reason and "c " in v.reason
        assert v.name == "filter.rr-after-costs"

    def test_name_is_the_full_registry_key(self):
        v = accept(None, plan(risk_pct=0.02, reward_pct=0.06), **DEFAULTS)
        assert v.name.startswith("filter.")


class TestRrDistributionReport:
    def test_empty_list_has_no_zero_division(self):
        r = rr_distribution_report([])
        assert r["n_plans"] == 0
        assert r["survival_rate"] is None
        assert all(r["n_pass"][t] == 0 for t in RR_REPORT_THRESHOLDS)
        assert r["net"]["n"] == 0 and r["net"]["median"] is None

    def test_hand_built_ten_verdicts_give_exact_counts(self):
        """Ten plans whose NET ratios are, by construction, 1.0 .. 5.5 in steps
        of 0.5. Counts are then exact by inspection:
            >= 2.00 -> 1.0 is out, 1.5 is out, 2.0..5.5 in  => 8
            >= 1.75 -> 1.0 out, 1.5 out                     => 8
            >= 1.50 -> 1.0 out                              => 9
            >= 1.25 -> 1.0 out                              => 9
            >= 1.00 -> all                                  => 10
        """
        risk_pct = 0.02
        c = cost_ratio(risk_pct, FEE, SLIP)
        targets = [1.0 + 0.5 * k for k in range(10)]
        verdicts = []
        for X in targets:
            reward_pct = (X + (X + 1.0) * c) * risk_pct
            verdicts.append(
                accept(None, plan(risk_pct=risk_pct, reward_pct=reward_pct), **DEFAULTS)
            )
        nets = [v.measured["net_rr"] for v in verdicts]
        for got, want in zip(nets, targets):
            assert abs(got - want) < 1e-12

        r = rr_distribution_report(verdicts)
        assert r["n_plans"] == 10
        assert r["n_pass"][2.0] == 8
        assert r["n_pass"][1.75] == 8
        assert r["n_pass"][1.5] == 9
        assert r["n_pass"][1.25] == 9
        assert r["n_pass"][1.0] == 10
        assert r["survival_rate"] == r["n_pass_target"] / 10
        assert r["n_pass_target"] == r["n_pass"][config.RR_TARGET_MIN]
        assert abs(r["net"]["min"] - 1.0) < 1e-12
        assert abs(r["net"]["max"] - 5.5) < 1e-12
        assert abs(r["net"]["median"] - 3.25) < 1e-12  # mean of 3.0 and 3.5
        assert len(r["net"]["deciles"]) == 9
        assert r["gross"]["n"] == 10

    def test_non_finite_net_is_counted_and_excluded_from_quantiles(self):
        good = accept(None, plan(risk_pct=0.02, reward_pct=0.10), **DEFAULTS)
        bad = accept(None, plan(risk_pct=0.01, reward_pct=0.05),
                     rr_target_min=2.0, fee_pct=-0.01, slippage_pct=0.0)
        r = rr_distribution_report([good, bad])
        assert r["n_plans"] == 2
        assert r["n_non_finite_net"] == 1
        assert r["net"]["n"] == 1  # the -inf is excluded, not swallowed


class TestRecording:
    """The opt-in tape that `graph-backtest --rr-report` uses.

    Off by default in every production run; the DECISION never reads it, so the
    filter stays a pure function of (plan, costs) — only the diagnostic tape is
    stateful, and it is scoped by the context manager.
    """

    def test_off_by_default(self):
        assert rr_after_costs._RECORDER is None
        accept(None, plan(risk_pct=0.02, reward_pct=0.06), **DEFAULTS)
        assert rr_after_costs._RECORDER is None

    def test_records_inside_the_block_and_stops_after(self):
        with rr_after_costs.recording() as tape:
            accept(None, plan(risk_pct=0.02, reward_pct=0.06), **DEFAULTS)
            accept(None, plan(risk_pct=0.02, reward_pct=0.03), **DEFAULTS)
            assert len(tape) == 2
        assert rr_after_costs._RECORDER is None
        accept(None, plan(risk_pct=0.02, reward_pct=0.06), **DEFAULTS)
        assert len(tape) == 2, "recording must stop when the block exits"

    def test_an_exception_still_restores_the_previous_recorder(self):
        with pytest.raises(RuntimeError):
            with rr_after_costs.recording():
                raise RuntimeError("boom")
        assert rr_after_costs._RECORDER is None

    def test_records_rejections_too(self):
        """The report is over every plan that REACHED the filter, not over the
        accepted ones — otherwise survival_rate would be 100% by construction."""
        with rr_after_costs.recording() as tape:
            accept(None, plan(risk_pct=0.02, reward_pct=0.03), **DEFAULTS)
        assert len(tape) == 1 and tape[0].accepted is False


class TestFrozenSurface:
    def test_rr_floor_and_rr_target_min_are_distinct_constants(self):
        """Contract §7: "Do not modify existing constants." RR_FLOOR stays the
        legacy breakout path's GROSS floor; RR_TARGET_MIN is the NEW net floor.
        A literal pin, so a future edit that "harmonises" them fails loudly."""
        assert config.RR_FLOOR == 1.5
        assert config.RR_TARGET_MIN == 2.0
        assert config.RR_FLOOR is not config.RR_TARGET_MIN

    def test_volume_confirm_min_ratio_is_defined_by_reference(self):
        """It must track VOLUME_HIGH_RATIO, not be a re-typed 1.5 — the exact
        drift trap TRAIL_ATR_MULTIPLE was split out of."""
        assert config.VOLUME_CONFIRM_MIN_RATIO == config.VOLUME_HIGH_RATIO

    def test_funding_caveat_cannot_be_deleted_silently(self):
        """net_rr charges fee + slippage only, so this filter UNDERSTATES realized
        cost and is if anything too permissive. The caveat is load-bearing honesty,
        so its presence is asserted."""
        doc = (rr_after_costs.__doc__ or "").lower()
        assert "funding" in doc
        assert "permissive" in doc

    def test_the_decision_rule_is_recorded_in_the_module(self):
        """Pre-registration only works if it cannot be quietly rewritten."""
        doc = rr_after_costs.__doc__ or ""
        assert "PRE-REGISTERED DECISION RULE" in doc
        assert "FORBIDDEN" in doc
