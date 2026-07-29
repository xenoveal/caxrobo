"""
ATR-scaled stop distance and cost-ratio helpers (Phase 2: honest risk model).

Stop distance is a market-structure quantity (how far price plausibly moves
against the setup before invalidating the thesis), governed by volatility —
NOT an account-risk sizing rule. These are pure functions with no I/O.

compute_atr_stop is used by the breakout path (setup.build_signal) only. The
fade path (meanrev.build_fade_signal) never calls it — it keeps its own
structural stop, the excursion extreme (the stretch's low/high), by design:
that stop marks where the reversion thesis is invalidated.

Because that structural stop has no volatility floor, the fade path's risk_pct
can collapse toward zero, which the dimensionless gross reward:risk ratio
cannot detect. The fade path therefore imports net_rr from this module and
gates on the cost-adjusted ratio instead. See net_rr's docstring.
"""


def compute_atr_stop(entry: float, direction: str, atr_value: float, k: float) -> float:
    """Stop price = k * ATR away from entry, in the direction that invalidates the trade.

    Args:
        entry: Entry price.
        direction: "long" or "short".
        atr_value: Current ATR (already computed by the caller; NaN is the
            caller's responsibility to check before calling).
        k: ATR multiple (config.ATR_STOP_MULTIPLE).

    Returns:
        Stop price. Always on the losing side of entry.
    """
    distance = k * atr_value
    return entry - distance if direction == "long" else entry + distance


def round_trip_cost_pct(fee_pct: float, slippage_pct: float) -> float:
    """Total fee + slippage cost of opening AND closing one position, as a
    fraction of entry price. Two sides, each paying fee + slippage.

    Excludes funding (time-dependent, so it cannot be known at signal time).
    """
    return 2 * (fee_pct + slippage_pct)


def net_rr(
    reward_pct: float, risk_pct: float, fee_pct: float, slippage_pct: float
) -> float:
    """Cost-aware reward:risk ratio — (reward - cost) / (risk + cost).

    The gross ratio reward_pct / risk_pct is DIMENSIONLESS, so it is blind to
    absolute costs: a setup risking 0.05% to make 0.10% scores a healthy 2.0
    while round-trip cost (~0.14% at config defaults) exceeds the entire
    reward, making a perfect win a guaranteed net loss. Charging cost to both
    legs restores the scale that the ratio threw away — the winner nets less
    and the loser costs more, which is what actually happens.

    Returns a value <= 0 when reward_pct does not even cover cost, so a single
    `net_rr >= floor` comparison rejects unprofitable-by-construction setups
    without needing a separate absolute reward floor.

    Args:
        reward_pct: |target - entry| / entry.
        risk_pct: |entry - stop| / entry.
        fee_pct: Per-side taker fee.
        slippage_pct: Per-side assumed slippage.

    Returns:
        Cost-adjusted ratio. float('-inf') if the adjusted risk is
        non-positive (only reachable with nonsensical negative costs).
    """
    cost = round_trip_cost_pct(fee_pct, slippage_pct)
    net_risk = risk_pct + cost
    if net_risk <= 0:
        return float("-inf")
    return (reward_pct - cost) / net_risk


def cost_ratio(risk_pct: float, fee_pct: float, slippage_pct: float) -> float:
    """c = round-trip cost / risk unit. Used to assert the strategy sits on
    the right side of the cost frontier (config.COST_RATIO_CEILING), never to
    gate an individual candidate.

    NOTE: this counts only fee + slippage. engine.run_backtest also charges
    FUNDING_PCT_PER_DAY * hold_days, which this function has no way to see.
    The returned c therefore UNDERSTATES the engine's realized cost — it is
    a lower bound, not the true all-in cost ratio (~7% relative at a 24h
    hold, at config's default FUNDING_PCT_PER_DAY).

    Args:
        risk_pct: |entry - stop| / entry.
        fee_pct: Per-side taker fee.
        slippage_pct: Per-side assumed slippage.

    Returns:
        cost / risk_pct. Undefined (returns float('inf')) if risk_pct <= 0.
    """
    if risk_pct <= 0:
        return float("inf")
    round_trip_cost = 2 * (fee_pct + slippage_pct)
    return round_trip_cost / risk_pct
