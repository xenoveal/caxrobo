"""
Evolution engine (v0.3.0 Phase 6) — population search over StrategyGraphs.

THE ONE THING TO KNOW ABOUT THIS PACKAGE: exactly one function in it can turn a
strategy into a number — evolution.oracle.GateOracle.evaluate — and it charges
Phase 1's persistent trial ledger BEFORE it scores. `walk_forward_pooled` is
imported in oracle.py and nowhere else under evolution/; nothing here imports
scripts/bruteforce, engine.run_backtest, run_graph_backtest, compute_metrics or
compute_equity_metrics. A change that lets a candidate be scored without a
ledger row is wrong regardless of how much faster it is.

This package WRAPS backtest/walkforward.py. It never edits it (contract §7).
"""

from trading_bot.evolution.oracle import (
    GateOracle,
    HoldoutViolation,
    OracleResult,
    TrialLedger,
    evo_grid,
)
from trading_bot.evolution.population import Campaign, Generation, Member
from trading_bot.evolution.runner import calibrate, run_campaign

__all__ = (
    "Campaign",
    "GateOracle",
    "Generation",
    "HoldoutViolation",
    "Member",
    "OracleResult",
    "TrialLedger",
    "calibrate",
    "evo_grid",
    "run_campaign",
)
