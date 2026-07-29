"""
Time-indexed equity metrics (Phase 3: Sharpe-first).

Builds a calendar-complete daily return series from the engine's Trade list
and computes Sharpe / Sortino / equity-curve max drawdown / Deflated Sharpe
Ratio on it. Each trade contributes its net pnl_pct SPREAD EVENLY across the
UTC days it was open, entry_ts -> exit_ts inclusive (equal notional per trade
— the vol-targeted curve arrives in Phase 8).

v0.2.0 booked the whole pnl_pct on the EXIT day, producing daily skew 3.79 /
kurtosis 31.24 on 23 trades and blocking every significance test
(KNOWN-LIMITATIONS §3, finding MEDIUM-5); pass
attribution=ATTRIBUTION_EXIT_DAY to reproduce it, so the moment delta stays
measurable rather than asserted.

Days with no open trade are zero-return days and are INCLUDED: idle time is
real time, and excluding it inflates Sharpe.

Pure computation, no I/O, no config coupling. Pools naturally across
symbols: pass trades from several symbols and their same-day pnls sum.
"""

import logging
import math
import statistics
from statistics import NormalDist

logger = logging.getLogger("trading_bot")

DAY_MS = 86_400_000
PERIODS_PER_YEAR = 365  # crypto perps trade every calendar day

_EULER_GAMMA = 0.5772156649015329

# P&L attribution modes (MEDIUM-5). "spread" is this module's default; the
# config constant that flips it in production is read by the already
# config-coupled callers (walkforward.py, cli.py), never here — this module's
# docstring promises no config coupling.
ATTRIBUTION_SPREAD = "spread"
ATTRIBUTION_EXIT_DAY = "exit_day"
ATTRIBUTION_MODES = (ATTRIBUTION_SPREAD, ATTRIBUTION_EXIT_DAY)


def daily_returns(trades, start_ms: int, end_ms: int, *,
                  attribution: str = ATTRIBUTION_SPREAD) -> list[float]:
    """Calendar-complete daily return series over [start_ms, end_ms).

    Args:
        trades: Iterable of engine.Trade (needs entry_ts, exit_ts, pnl_pct).
        start_ms / end_ms: Span in epoch ms; every UTC day in the span gets
            an entry (0.0 if no trade was open that day).
        attribution: ATTRIBUTION_SPREAD (default) books each trade's pnl_pct
            evenly across the UTC days it was open, entry_ts -> exit_ts
            inclusive. ATTRIBUTION_EXIT_DAY reproduces v0.2.0 by booking the
            whole pnl_pct on the exit day (MEDIUM-5).

    Returns:
        List of daily returns (fractions), one per UTC day, oldest first.
        Empty list if the span is empty or inverted.

    Raises:
        ValueError: On an unknown attribution mode. A silent fallback would
            make the gate's definition of Sharpe depend on a typo.
    """
    if attribution not in ATTRIBUTION_MODES:
        raise ValueError(
            f"unknown attribution mode {attribution!r}; expected one of {ATTRIBUTION_MODES}"
        )
    if end_ms <= start_ms:
        return []
    first_day = start_ms // DAY_MS
    n_days = (end_ms - 1) // DAY_MS - first_day + 1
    rets = [0.0] * n_days
    for t in trades:
        if attribution == ATTRIBUTION_EXIT_DAY:
            d = t.exit_ts // DAY_MS - first_day
            if 0 <= d < n_days:
                rets[d] += t.pnl_pct
            else:
                logger.warning("trade exit_ts %d outside metrics span; dropped", t.exit_ts)
            continue
        # ATTRIBUTION_SPREAD. entry_ts/exit_ts already exist on Trade
        # (engine.py:124,128) — NO schema change is needed.
        d_first = t.entry_ts // DAY_MS - first_day
        d_last = t.exit_ts // DAY_MS - first_day
        if d_last < d_first:  # pragma: no cover - defensive invariant
            logger.warning(
                "trade exit_ts %d precedes entry_ts %d; booked on the exit day",
                t.exit_ts, t.entry_ts,
            )
            d_first = d_last
        lo, hi = max(d_first, 0), min(d_last, n_days - 1)
        if hi < lo:
            logger.warning(
                "trade %d->%d does not overlap the metrics span; dropped",
                t.entry_ts, t.exit_ts,
            )
            continue
        # Divide by the CLIPPED day count, not the full holding length, so a
        # trade straddling a span boundary still contributes its WHOLE
        # pnl_pct: sum() is then invariant to the attribution mode for any
        # overlapping trade (measured: 0.16509935688760607 both ways on the 23
        # OOS trades). Dividing by full length and dropping the out-of-span
        # share would make total reported P&L depend on where the window is cut.
        share = t.pnl_pct / (hi - lo + 1)
        for d in range(lo, hi + 1):
            rets[d] += share
    return rets


def sharpe_ratio(returns: list[float], periods_per_year: int = PERIODS_PER_YEAR) -> float | None:
    """Annualized Sharpe (risk-free rate 0). None if < 2 obs or zero variance."""
    if len(returns) < 2:
        return None
    mu = statistics.fmean(returns)
    sd = statistics.stdev(returns)  # sample stdev (n-1)
    if sd == 0:
        return None
    return (mu / sd) * math.sqrt(periods_per_year)


def sortino_ratio(returns: list[float], periods_per_year: int = PERIODS_PER_YEAR) -> float | None:
    """Annualized Sortino: mean / downside deviation (all obs in denominator).

    Downside deviation = sqrt(mean(min(r, 0)^2)) over ALL returns — the
    full-series convention, not stdev of the losing subset.
    None if < 2 obs or no downside at all.
    """
    if len(returns) < 2:
        return None
    mu = statistics.fmean(returns)
    dd = math.sqrt(statistics.fmean([min(r, 0.0) ** 2 for r in returns]))
    if dd == 0:
        return None
    return (mu / dd) * math.sqrt(periods_per_year)


def max_drawdown(returns: list[float]) -> float | None:
    """Peak-to-trough drawdown (positive fraction) on the COMPOUNDED equity
    curve — not the sum-of-percentages proxy metrics.py uses. None if empty."""
    if not returns:
        return None
    equity = peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        max_dd = max(max_dd, 1.0 - equity / peak)
    return max_dd


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int,
                          skew: float, kurt: float) -> float | None:
    """PSR: probability the true (per-period) Sharpe exceeds sr_benchmark.

    PSR = Phi( (sr - sr*) * sqrt(n_obs - 1)
               / sqrt(1 - skew*sr + ((kurt - 1) / 4) * sr^2) )

    All Sharpe values PER-PERIOD (daily), NOT annualized. kurt is the raw
    (non-excess) kurtosis: 3.0 for a normal distribution.
    """
    if n_obs < 2:
        return None
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0:
        return None  # pathological higher moments; refuse rather than lie
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return NormalDist().cdf(z)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """E[max SR] under n_trials independent zero-true-Sharpe trials
    (Bailey & Lopez de Prado eq. for the expected maximum):

    SR0 = sqrt(sr_var) * ((1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N*e)))
    """
    if n_trials <= 1 or sr_var <= 0:
        return 0.0
    z = NormalDist().inv_cdf
    return math.sqrt(sr_var) * (
        (1.0 - _EULER_GAMMA) * z(1.0 - 1.0 / n_trials)
        + _EULER_GAMMA * z(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(sr: float, n_trials: int, n_obs: int,
                     skew: float, kurt: float,
                     sr_var: float | None = None) -> float | None:
    """DSR: PSR evaluated against the expected-max Sharpe of the search.

    Args:
        sr: Observed PER-PERIOD (daily) Sharpe of the selected strategy.
        n_trials: TOTAL configurations evaluated during the search
            (grid combos x folds + neighbor probes — the caller counts).
        n_obs: Length of the daily return series.
        skew / kurt: Sample skewness and raw kurtosis of the daily returns.
        sr_var: Variance of Sharpe estimates across trials. Default: the
            SR estimator variance (1 - skew*sr + (kurt-1)/4*sr^2)/(n_obs-1)
            — a conservative stand-in when per-trial SRs weren't retained.

    Returns:
        Probability in [0, 1]; DSR > 0.95 == significant at p < 0.05.
        None when undefined (n_obs < 2 or pathological moments).
    """
    if n_obs < 2:
        return None
    if sr_var is None:
        v = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
        if v <= 0:
            return None
        sr_var = v / (n_obs - 1)
    sr0 = expected_max_sharpe(n_trials, sr_var)
    return probabilistic_sharpe(sr, sr0, n_obs, skew, kurt)


def _skew_kurt(returns: list[float]) -> tuple[float, float]:
    """Sample skewness and RAW kurtosis (normal => 3.0). (0.0, 3.0) if degenerate."""
    n = len(returns)
    if n < 2:
        return 0.0, 3.0
    mu = statistics.fmean(returns)
    m2 = statistics.fmean([(r - mu) ** 2 for r in returns])
    if m2 == 0:
        return 0.0, 3.0
    m3 = statistics.fmean([(r - mu) ** 3 for r in returns])
    m4 = statistics.fmean([(r - mu) ** 4 for r in returns])
    return m3 / m2 ** 1.5, m4 / m2 ** 2


def compute_equity_metrics(trades, start_ms: int, end_ms: int,
                            n_trials: int = 1, *,
                            attribution: str = ATTRIBUTION_SPREAD) -> dict:
    """One-stop equity metrics for a trade list over a span.

    Returns dict with: n_days, sharpe (annualized), sortino (annualized),
    max_drawdown_pct, ann_return_pct (compounded, reported-not-gated per
    PRD), dsr (probability), daily_sharpe, skew, kurtosis (the moments the DSR
    was computed on, so a report can quote them without recomputing), and
    attribution (which mode produced the series). None values where undefined.
    """
    rets = daily_returns(trades, start_ms, end_ms, attribution=attribution)
    n = len(rets)
    sr_daily = None
    if n >= 2:
        sd = statistics.stdev(rets)
        sr_daily = (statistics.fmean(rets) / sd) if sd > 0 else None
    skew, kurt = _skew_kurt(rets)
    equity = 1.0
    wiped_out = False
    for r in rets:
        factor = 1.0 + r
        # A non-positive factor means a single day's loss consumed the whole
        # account (or more). The account is terminally wiped from that day
        # forward: stop compounding immediately, since an even number of such
        # factors would otherwise multiply back to a spuriously large
        # positive equity (e.g. two days of r <= -1 flips the sign twice and
        # yields an absurd positive "ann_return_pct").
        if factor <= 0.0:
            equity = 0.0
            wiped_out = True
            break
        equity *= factor
    if n == 0:
        ann_return = None
    elif wiped_out:
        ann_return = -1.0
    elif equity > 0:
        ann_return = equity ** (PERIODS_PER_YEAR / n) - 1.0
    else:
        ann_return = None
    return {
        "n_days": n,
        "sharpe": sharpe_ratio(rets),
        "sortino": sortino_ratio(rets),
        "max_drawdown_pct": max_drawdown(rets),
        "ann_return_pct": ann_return,
        "daily_sharpe": sr_daily,
        "dsr": deflated_sharpe(sr_daily, n_trials, n, skew, kurt)
               if sr_daily is not None else None,
        "skew": skew,
        "kurtosis": kurt,
        "attribution": attribution,
    }
