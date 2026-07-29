"""
filter.rr-after-costs — the PRD's ">=1:2 reward:risk AFTER COSTS" requirement.

PRD Success Metrics row 4: "100% of taken positions pass >=1:2 R:R after costs at
entry." This Filter is the mechanism, and `config.RR_TARGET_MIN = 2.0` is the
floor, applied to `risk.atr_stop.net_rr` — the COST-ADJUSTED ratio.

WHY NET AND NOT GROSS, quoting risk/atr_stop.py:49-58 directly: the gross ratio
`reward_pct / risk_pct` is DIMENSIONLESS, so it is blind to absolute cost — "a
setup risking 0.05% to make 0.10% scores a healthy 2.0 while round-trip cost
(~0.14% at config defaults) exceeds the entire reward, making a perfect win a
guaranteed net loss." `net_rr` charges cost to both legs: the winner nets less and
the loser costs more, which is what actually happens.

THE ALGEBRA, so the floor's real severity is visible rather than asserted.
`net_rr(reward, risk, fee, slip) = (reward - cost) / (risk + cost)` with
`cost = 2 * (fee + slip) = 0.0014` at config defaults. Requiring `net_rr >= X` is
equivalent, in GROSS terms, to

    reward_pct >= X * risk_pct + (X + 1) * cost
    divide by risk_pct, and let c = cost / risk_pct  (= risk.atr_stop.cost_ratio)
    gross_rr >= X + (X + 1) * c

So a NET floor of 2.0 is a GROSS floor of `2 + 3c`. At the median `risk_pct`
already measured and recorded in config.py:166-168:

    symbol    median risk_pct     c         gross R:R needed for NET 2.0
    BTCUSDT   1.968%              0.0711    2.2134
    ETHUSDT   2.679%              0.0523    2.1568
    SOLUSDT   3.752%              0.0373    2.1119

And the fact that makes this a genuinely new bar: backtest/walkforward.py records
that the MINIMUM planned GROSS R:R across every trade the v0.2.0 engine ever took
was 1.56, and 100% of trades cleared both 1.25 and 1.5. A trade at gross 1.56 has
`net_rr` of 1.3900 / 1.4329 / 1.4679 (BTC/ETH/SOL) — it clears neither 2.0 nor
even 1.5 NET. The 1.5-GROSS floor was measured non-binding; a 2.0-NET floor is not.

THIS FLOOR IS EXPECTED TO REJECT MOST PLANS. That is stated here, before
measurement, so a near-zero survivor count reads as a CONFIRMED PREDICTION rather
than as a surprise to be engineered away. The survival rate is measured by
`cli graph-backtest --rr-report`, which reports `n_plans`, `n_pass` at each of
RR_REPORT_THRESHOLDS, and the net/gross deciles.

PRE-REGISTERED DECISION RULE, reproduced verbatim from the phase plan so it cannot
be quietly reinterpreted after the measurement. Let `N2 = n_pass(2.0)` pooled
across the production symbols over the full stored span:

  - `survival_rate >= 0.20` AND `N2 >= 30`: proceed. RR_TARGET_MIN stays 2.0.
  - `0 < N2 < 30`: RR_TARGET_MIN STAYS 2.0. Report: "the >=1:2-after-costs
    requirement is satisfiable but starves the sample." The remedy is MORE
    CANDIDATES — Phase 2's symbol breadth, Phase 8's detector breadth — not a
    lower floor. Phase 4 is still done: the pipeline works; the sample does not
    clear WF_MIN_TRADES = 30.
  - `N2 == 0`: RR_TARGET_MIN STAYS 2.0. Report as a FINDING: "the PRD's
    >=1:2-after-costs requirement is unsatisfiable by the thin slice's TP/SL
    geometry (target = channel width, stop = 1.5*ATR)." Escalate with exactly
    three non-weakening remedies: (a) detectors with a structurally larger
    target_height, (b) a tighter stop with a STRUCTURAL basis, not a fitted one,
    (c) accept that the requirement falsifies this strategy family — which the
    PRD's honesty clause explicitly permits.

FORBIDDEN under every branch without an explicit user decision logged as a
consumed degree of freedom: lowering RR_TARGET_MIN; switching to gross R:R;
dropping funding or slippage from the cost term; modifying `RR_FLOOR` (contract
§7 — it is UNTOUCHED at 1.5 and still governs the legacy breakout path); or adding
an `or` escape clause. A run that produces zero trades is a result, not an error
(cli.py's `_backtest_command` already codifies that).

ITS OWN LIMITATION, inherited from `net_rr`: FUNDING IS NOT CHARGED.
risk/atr_stop.py:82-88 states it, and `round_trip_cost_pct`'s docstring explains
why — funding is time-dependent and holding duration is UNKNOWABLE at signal time.
The engine does charge `FUNDING_PCT_PER_DAY * hold_days` on the realized trade
(engine.py's close_out). So the ratio this Filter computes UNDERSTATES realized
cost, and the Filter is, if anything, TOO PERMISSIVE — never too strict. Inventing
an assumed hold to close the gap would be a new fitted parameter; saying so is the
honest posture.
"""

import logging
import math
from contextlib import contextmanager

from trading_bot import config
from trading_bot.framework.contracts import FilterVerdict, ParamSpec
from trading_bot.framework.registry import register
from trading_bot.risk.atr_stop import cost_ratio, net_rr

logger = logging.getLogger("trading_bot")

RR_FILTER_NAME = "filter.rr-after-costs"

# Printed FOR DIAGNOSIS ONLY. Printing 1.5 is not permission to use 1.5 — see the
# pre-registered decision rule above.
RR_REPORT_THRESHOLDS = (2.0, 1.75, 1.5, 1.25, 1.0)


# --------------------------------------------------------------------------- #
# Opt-in verdict recorder, for the measure-only `--rr-report` run.
# --------------------------------------------------------------------------- #
#
# The plan pre-registers a report over "every PositionPlan that reaches the
# filter". The executor discards rejected plans, so that population is not
# recoverable from the returned Trade list, and the alternatives are worse: adding
# an observer hook to framework/execute.py would be a second cross-ownership edit
# to Phase 3's file, and re-running with a permissive floor would measure a
# DIFFERENT population (one open trade per symbol means the accepted set changes
# which later plans are ever reached).
#
# So the recorder lives here, in the plug-in that owns the number, and is:
#   - OFF by default; `_RECORDER is None` in every production run;
#   - enabled only inside the `recording()` context manager;
#   - never read by `accept()` — the DECISION stays pure, which is the part of
#     contract §3's purity clause that matters. Only the diagnostic tape is
#     stateful, and it is scoped.
_RECORDER: list | None = None


@contextmanager
def recording():
    """Collect every FilterVerdict this filter produces inside the block.

    Yields:
        The list being appended to. Restores the previous recorder on exit, so
        nesting is safe and an exception cannot leave recording switched on.
    """
    global _RECORDER
    previous = _RECORDER
    tape: list = []
    _RECORDER = tape
    try:
        yield tape
    finally:
        _RECORDER = previous


@register(
    "filter",
    name="rr-after-costs",
    params={
        "rr_target_min": ParamSpec(
            kind="float",
            default=config.RR_TARGET_MIN,
            bounds=(0.0, 10.0),
            doc="Minimum COST-ADJUSTED reward:risk (risk.atr_stop.net_rr)",
        ),
        "fee_pct": ParamSpec(
            kind="float",
            default=config.FEE_PCT,
            bounds=(0.0, 0.01),
            doc="Per-side taker fee",
        ),
        "slippage_pct": ParamSpec(
            kind="float",
            default=config.SLIPPAGE_PCT,
            bounds=(0.0, 0.01),
            doc="Per-side assumed slippage",
        ),
    },
    rationale=(
        "The PRD's '>=1:2 reward:risk AFTER COSTS' requirement, applied to "
        "risk.atr_stop.net_rr rather than to the dimensionless gross ratio, which "
        "is blind to absolute cost (a setup risking 0.05% to make 0.10% scores a "
        "healthy gross 2.0 while round-trip cost exceeds the whole reward). "
        "DERIVED FROM COST ALGEBRA, NEVER FITTED TO RETURNS: net >= X is gross >= "
        "X + (X+1)*c. Predicted to reject most plans, and that prediction is "
        "measured and reported rather than engineered away. Caveat: net_rr charges "
        "fee + slippage only, so it understates realized cost and this filter is "
        "if anything too permissive."
    ),
)
def rr_after_costs(ctx, plan, *, rr_target_min: float, fee_pct: float, slippage_pct: float):
    """Accept a plan only when its cost-adjusted reward:risk clears the floor.

    Args:
        ctx: EvalContext bound to the trigger bar's open. Unused: the decision is
            a pure function of the plan and the cost constants, which is the point
            — a risk screen that consulted market data could be fitted to it.
        plan: The PositionPlan under evaluation.
        rr_target_min: NET floor (default config.RR_TARGET_MIN = 2.0).
        fee_pct / slippage_pct: Per-side costs. Do NOT compute costs here: the one
            definition is risk.atr_stop.round_trip_cost_pct, which net_rr and
            cost_ratio both call (contract §1 — the cost model is frozen, and any
            new path charges costs identically or it is lying).

    Returns:
        FilterVerdict. `measured` carries net_rr, gross_rr, cost_ratio, risk_pct,
        reward_pct and gross_rr_required, all floats (contract §3 makes `measured`
        a Mapping[str, float]), so a rejection is auditable rather than a bare
        False. The boundary is `>=`: equality PASSES, matching setup.py:154's
        `rr < floor` rejection semantics.
    """
    net = net_rr(plan.reward_pct, plan.risk_pct, fee_pct, slippage_pct)
    c = cost_ratio(plan.risk_pct, fee_pct, slippage_pct)
    gross_required = rr_target_min + (rr_target_min + 1.0) * c
    accepted = net >= rr_target_min

    verdict = FilterVerdict(
        accepted=accepted,
        name=RR_FILTER_NAME,
        # `net` can be -inf (only reachable with nonsensical negative costs) and
        # `c` can be +inf (risk_pct == 0). Python's format spec renders both
        # without raising, so the f-string is safe; a test covers each.
        reason=(
            f"net_rr {net:.4f} {'>=' if accepted else '<'} {rr_target_min:.2f} "
            f"(gross {plan.rr:.4f}, c {c:.4f}, gross floor needed "
            f"{gross_required:.4f})"
        ),
        measured={
            "net_rr": float(net),
            "gross_rr": float(plan.rr),
            "cost_ratio": float(c),
            "risk_pct": float(plan.risk_pct),
            "reward_pct": float(plan.reward_pct),
            "gross_rr_required": float(gross_required),
        },
    )
    if _RECORDER is not None:
        _RECORDER.append(verdict)
    if not accepted:
        logger.debug("%s %s rejected: %s", plan.symbol, plan.source, verdict.reason)
    return verdict


def _quantiles(values: list[float]) -> dict:
    """min / deciles / median / max of a finite-filtered sorted sample.

    Deciles are the linearly-interpolated 10th..90th percentiles, computed
    without numpy so this stays a plain-Python reporting helper. Non-finite
    values (a -inf net_rr) are EXCLUDED from the quantiles and counted
    separately by the caller, because a -inf would swallow every decile.
    """
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return {"n": 0, "min": None, "median": None, "max": None, "deciles": []}

    def q(p: float) -> float:
        if len(finite) == 1:
            return finite[0]
        pos = p * (len(finite) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(finite) - 1)
        frac = pos - lo
        return finite[lo] + (finite[hi] - finite[lo]) * frac

    return {
        "n": len(finite),
        "min": finite[0],
        "median": q(0.5),
        "max": finite[-1],
        "deciles": [q(k / 10.0) for k in range(1, 10)],
    }


def rr_distribution_report(verdicts: list) -> dict:
    """Survival counts and deciles over every plan that reached the filter.

    Returns `n_plans`, `n_pass` per threshold in RR_REPORT_THRESHOLDS,
    `survival_rate` at config.RR_TARGET_MIN, and min/deciles/median/max for BOTH
    net and gross R:R so the gross-vs-net gap is measured, not asserted.

    The sub-target thresholds are printed FOR DIAGNOSIS ONLY. Printing 1.5 is not
    permission to use 1.5 — see the module docstring's decision rule.

    Args:
        verdicts: FilterVerdicts, e.g. the tape from `recording()`.

    Returns:
        dict with keys n_plans, n_pass (threshold -> count), survival_rate
        (None when n_plans == 0, never a ZeroDivisionError), net, gross,
        n_non_finite_net, and gross_rr_required (quantiles of the per-plan
        required gross floor, so the 2.11-2.21 prediction is checkable).
    """
    nets = [float(v.measured.get("net_rr", float("nan"))) for v in verdicts]
    grosses = [float(v.measured.get("gross_rr", float("nan"))) for v in verdicts]
    required = [
        float(v.measured.get("gross_rr_required", float("nan"))) for v in verdicts
    ]
    n_plans = len(verdicts)
    n_pass = {t: sum(1 for v in nets if v >= t) for t in RR_REPORT_THRESHOLDS}
    target = config.RR_TARGET_MIN
    n_target = sum(1 for v in nets if v >= target)
    return {
        "n_plans": n_plans,
        "n_pass": n_pass,
        "n_pass_target": n_target,
        "rr_target_min": target,
        "survival_rate": (n_target / n_plans) if n_plans else None,
        "n_non_finite_net": sum(1 for v in nets if not math.isfinite(v)),
        "net": _quantiles(nets),
        "gross": _quantiles(grosses),
        "gross_rr_required": _quantiles(required),
    }
