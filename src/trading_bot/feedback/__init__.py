"""
Feedback loop MVP (v0.3.0 Phase 5, contract §2 owns this package).

Thin re-exports only; no logic lives here. See records.py for the oracle
boundary this whole package is built around.
"""

from trading_bot.feedback.records import Diagnosis, ReviewContext, ReviewRecord
from trading_bot.feedback.versioning import StrategyVersion
from trading_bot.feedback.protocol import run_loop_iteration

__all__ = [
    "ReviewRecord",
    "ReviewContext",
    "Diagnosis",
    "StrategyVersion",
    "run_loop_iteration",
]
