"""
Trade-list performance metrics.

Pure computation over the engine's Trade list. Expectancy and drawdown are in
percent-of-entry terms (each trade is treated as equal-sized, consistent with
the alert-only, human-sized execution model). The per-bucket breakdown
(regime+pattern) is the raw material for Phase 6 confidence calibration.
"""

import logging

logger = logging.getLogger("trading_bot")


def compute_metrics(trades) -> dict:
    """
    Compute summary statistics for a list of Trades.

    Args:
        trades: Iterable of engine.Trade (needs pnl_pct, regime, pattern).

    Returns:
        Dict with n_trades, win_rate, expectancy_pct, avg_win_pct,
        avg_loss_pct, profit_factor, max_drawdown_pct, and by_bucket — a dict
        keyed "regime/pattern" of the same stats (without nested buckets).
        Ratios are None when undefined (no trades / no losses).
    """
    trades = list(trades)
    stats = _stats([t.pnl_pct for t in trades])

    buckets: dict[str, list] = {}
    for t in trades:
        buckets.setdefault(f"{t.regime}/{t.pattern}", []).append(t.pnl_pct)
    stats["by_bucket"] = {k: _stats(v) for k, v in sorted(buckets.items())}
    return stats


def _stats(pnls: list[float]) -> dict:
    """Summary stats for a sequence of per-trade pnl percentages."""
    n = len(pnls)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": None,
            "expectancy_pct": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
        }

    wins = [p for p in pnls if p > 0]
    # Strictly negative: an exactly-zero trade is neither a win nor a loss, and
    # bucketing it as a loss depressed avg_loss_pct and profit_factor.
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)

    # Max drawdown on the cumulative equal-size pnl curve.
    cum = peak = max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {
        "n_trades": n,
        "win_rate": len(wins) / n,
        "expectancy_pct": sum(pnls) / n,
        "avg_win_pct": gross_win / len(wins) if wins else None,
        "avg_loss_pct": -gross_loss / len(losses) if losses else None,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "max_drawdown_pct": max_dd,
    }
