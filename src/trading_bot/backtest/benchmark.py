"""
The buy-and-hold null hypothesis (v0.3.0 Phase 1).

v0.2.0's gate compared the strategy against ZERO, so — in its own words —
"THE GATE could have blessed a strategy worse than inaction"
(KNOWN-LIMITATIONS §0). This module computes what doing nothing clever would
have earned over the identical span, per symbol and as an equal-weight basket,
using the SAME metric functions (equity.sharpe_ratio / sortino_ratio /
max_drawdown) the strategy is scored with. A benchmark Sharpe from a different
formula is not comparable to a strategy Sharpe.

THREE STATED ASSUMPTIONS.

1. COSTS: one round-trip of fees + slippage, 2*(FEE_PCT + SLIPPAGE_PCT), and
   NO funding. The null is a spot-equivalent hold, not a perpetual position.
   Charging the frozen pessimistic FUNDING_PCT_PER_DAY = 0.0001 over 1095 days
   would cost the null ~11.6% and hand the strategy an ~11-point head start a
   real operator could dodge by simply buying spot. That placeholder exists to
   make the STRATEGY's costs pessimistic; applying it to the benchmark makes
   the comparison flattering — the opposite of its purpose. The fee IS charged
   so the null is not costless, and it is measurably immaterial (BTC 2.2135x ->
   2.2104x, Sharpe 0.8002 -> 0.7992, measured 2026-07-27).

2. BASKET: equal-weight, DAILY-REBALANCED — its daily return is the arithmetic
   mean of the constituents' daily returns. Measured 2026-07-27, that
   reproduces all four of KNOWN-LIMITATIONS §0's published basket figures
   (2.1628x / +29.32% / 0.7284 / 64.32% DD) while buy-once-hold (2.0809x /
   0.6998 / 66.91%) matches none of them. config.BENCHMARK_REBALANCE = "none"
   keeps buy-once reachable so the choice stays a measurement, not a belief.

3. ANNUALIZATION: ann_return_pct is a true CAGR on a multi-year span (§0's is
   exactly 1095 days = 3 years) but a 90-DAY EXTRAPOLATION on a walk-forward
   holdout — exactly as the strategy's is. The comparison stays valid because
   both sides are annualized identically over the identical span; the LEVEL is
   not quotable, and callers printing a short-span figure must say so.

Risk-free rate is 0, matching equity.py, or the two Sharpes would not be
comparable.
"""

import logging
import statistics
from dataclasses import dataclass

from trading_bot import config
from trading_bot.backtest.equity import (
    PERIODS_PER_YEAR,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from trading_bot.data.storage import load_candles

logger = logging.getLogger("trading_bot")

REBALANCE_DAILY = "daily"
REBALANCE_NONE = "none"
REBALANCE_MODES = (REBALANCE_DAILY, REBALANCE_NONE)

# Contract §4 fixes this metric set. total_return is an equity MULTIPLE
# (2.21 means 2.21x); ann_return_pct and max_drawdown_pct are FRACTIONS
# (0.303 == 30.3%), matching compute_equity_metrics' established naming so
# cli._fmt's '.2%' specs work unchanged.
METRIC_KEYS = (
    "total_return",
    "ann_return_pct",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "n_days",
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Buy-and-hold outcome over one span."""

    per_symbol: dict[str, dict]  # symbol -> METRIC_KEYS bundle; None where undefined
    basket: dict  # equal-weight, daily-rebalanced by default, same keys
    start_ms: int
    end_ms: int


def _metrics_from_returns(rets: list[float]) -> dict:
    """METRIC_KEYS bundle from a daily return series.

    Undefined metrics are None, never a fabricated 0.0 (metrics.py's
    convention). The compounding loop mirrors equity.compute_equity_metrics'
    wipe-out guard verbatim: a non-positive daily factor means one day's loss
    consumed the whole account, and continuing to multiply would let an even
    number of such factors flip the sign back to a spuriously large positive
    equity.
    """
    n = len(rets)
    if n == 0:
        return {**dict.fromkeys(METRIC_KEYS, None), "n_days": 0}

    equity = 1.0
    wiped_out = False
    for r in rets:
        factor = 1.0 + r
        if factor <= 0.0:
            equity = 0.0
            wiped_out = True
            break
        equity *= factor

    if wiped_out:
        total_return: float | None = 0.0
        ann_return: float | None = -1.0
    elif equity > 0:
        total_return = equity
        ann_return = equity ** (PERIODS_PER_YEAR / n) - 1.0
    else:  # pragma: no cover - unreachable given the guard above
        total_return = None
        ann_return = None

    return {
        "total_return": total_return,
        "ann_return_pct": ann_return,
        "sharpe": sharpe_ratio(rets),
        "sortino": sortino_ratio(rets),
        "max_drawdown_pct": max_drawdown(rets),
        "n_days": n,
    }


def _close_returns(conn, symbol: str, *, start_ms: int, end_ms: int,
                   timeframe: str, charge_fees: bool) -> list[float]:
    """Close-to-close returns for one symbol over an INCLUSIVE-bounds span.

    load_candles' bounds are both inclusive, so N stored bars yield N-1
    returns. Rows are tuples and close is index 4 (storage.py:230) — an
    off-by-one here would silently benchmark the low.
    """
    rows = load_candles(conn, symbol, timeframe, start_ms=start_ms, end_ms=end_ms)
    if len(rows) < 2:
        logger.warning(
            "benchmark: %s has %d %s bars in span; buy-and-hold undefined",
            symbol, len(rows), timeframe,
        )
        return []
    closes = [r[4] for r in rows]
    if any(c is None or c <= 0.0 for c in closes):
        # A non-positive close is corrupt data, not a price. Refuse (all-None
        # metrics, which FAIL the gate) rather than crash on the division or
        # invent a return — same spirit as equity.py's wipe-out guard.
        logger.warning(
            "benchmark: %s has a non-positive %s close in span; buy-and-hold undefined",
            symbol, timeframe,
        )
        return []
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    if charge_fees:
        # ONE round-trip, booked on the first day. Not per day, not per side
        # twice: a hold pays to get in and to get out, once each.
        rt = 2.0 * (config.FEE_PCT + config.SLIPPAGE_PCT)
        rets[0] = (1.0 + rets[0]) * (1.0 - rt) - 1.0
    return rets


def buy_and_hold(conn, symbols, *, start_ms: int, end_ms: int) -> BenchmarkResult:
    """Buy-and-hold null for `symbols` over [start_ms, end_ms] (inclusive).

    config.BENCHMARK_TIMEFRAME / BENCHMARK_CHARGE_FEES / BENCHMARK_REBALANCE
    are read at CALL time, so a test or an operator can flip them and see the
    benchmark change — the discipline FADE_ENABLED relies on.

    Args:
        conn: Database connection (ohlcv.db).
        symbols: Iterable of symbols to hold.
        start_ms / end_ms: Span in epoch ms, both bounds inclusive.

    Returns:
        BenchmarkResult. A symbol with fewer than 2 bars gets an all-None
        bundle and is EXCLUDED from the basket — a 2-symbol honest basket
        beats a 3-symbol one padded with zeros.

    Raises:
        ValueError: On an unknown config.BENCHMARK_REBALANCE.
    """
    timeframe = config.BENCHMARK_TIMEFRAME
    charge_fees = config.BENCHMARK_CHARGE_FEES
    rebalance = config.BENCHMARK_REBALANCE
    if rebalance not in REBALANCE_MODES:
        raise ValueError(
            f"unknown BENCHMARK_REBALANCE {rebalance!r}; expected one of {REBALANCE_MODES}"
        )

    per_symbol: dict[str, dict] = {}
    series: dict[str, list[float]] = {}
    for symbol in symbols:
        rets = _close_returns(
            conn, symbol, start_ms=start_ms, end_ms=end_ms,
            timeframe=timeframe, charge_fees=charge_fees,
        )
        per_symbol[symbol] = _metrics_from_returns(rets)
        if rets:
            series[symbol] = rets

    basket_rets: list[float] = []
    if not series:
        logger.warning("benchmark: no symbol produced a return series; basket undefined")
    else:
        lengths = {len(r) for r in series.values()}
        if len(lengths) > 1:
            # Refuse rather than zip-truncate to the shortest: truncating would
            # silently shorten the span the gate compares on.
            logger.warning(
                "benchmark: symbols disagree on bar count %s; basket undefined "
                "(run gap-report)",
                sorted(lengths),
            )
        elif rebalance == REBALANCE_DAILY:
            n = lengths.pop()
            keys = list(series)
            basket_rets = [
                statistics.fmean([series[s][i] for s in keys]) for i in range(n)
            ]
        else:  # REBALANCE_NONE — buy once, never rebalance
            n = lengths.pop()
            keys = list(series)
            weight = 1.0 / len(keys)
            path = [1.0]
            equities = {s: 1.0 for s in keys}
            for i in range(n):
                for s in keys:
                    equities[s] *= 1.0 + series[s][i]
                path.append(sum(weight * equities[s] for s in keys))
            basket_rets = [
                path[i] / path[i - 1] - 1.0 for i in range(1, len(path))
            ]

    return BenchmarkResult(
        per_symbol=per_symbol,
        basket=_metrics_from_returns(basket_rets),
        start_ms=start_ms,
        end_ms=end_ms,
    )
