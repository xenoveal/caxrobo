"""
Regime-switched signal dispatcher.

Per the PRD, exactly one signal method is active at a time, selected by the
current regime-timeframe regime:

  - trending           -> Donchian channel breakout (Phase 5, donchian.py)
  - ranging            -> mean-reversion fade method (Phase 4, meanrev.py),
                          subject to config.FADE_ENABLED
  - extreme-volatility -> no method (defensive suppression)
  - uncertain          -> no method (insufficient data)

The regime is computed once here; the per-method scan functions do not
re-check it.
"""

import logging
import time

from trading_bot import config
from trading_bot.regime.classifier import current_regime
from trading_bot.signals.donchian import scan_donchian_signals
from trading_bot.signals.meanrev import scan_fade_signals
from trading_bot.signals.setup import Signal

logger = logging.getLogger("trading_bot")


def scan_symbol(
    conn,
    symbol: str,
    *,
    now_ms: int | None = None,
) -> tuple[str, list[Signal]]:
    """
    Scan one symbol with the regime-matched signal method as of now_ms.

    Args:
        conn: Database connection.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        now_ms: Evaluation time in epoch milliseconds (default: current time).

    Returns:
        Tuple of (regime_label, signals). signals come from the breakout
        method when trending, the fade method when ranging (unless
        config.FADE_ENABLED is false), and are empty for extreme-volatility or
        uncertain regimes. The label is always the true classification — a
        suppressed method is not a misclassification.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    regime_label, _, _ = current_regime(conn, symbol, now_ms=now_ms)

    if regime_label == "trending":
        return (regime_label, scan_donchian_signals(conn, symbol, now_ms))
    if regime_label == "ranging":
        if not config.FADE_ENABLED:
            return (regime_label, [])
        return (regime_label, scan_fade_signals(conn, symbol, now_ms))
    return (regime_label, [])
