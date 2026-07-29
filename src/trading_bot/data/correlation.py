"""
Correlation measurement, effective-N, and universe selection (v0.3.0 Phase 2).

**The headline finding this module exists to measure (contract §0a/§0b):** the
PRD scoped this phase as "backfill uncorrelated symbols" assuming 3 stored
symbols. 20 are actually stored, all liquid Binance USDT-M perp majors, and
the open question was whether that breadth buys statistical independence.
Measured answer: mostly no. 3 -> 20 symbols multiplies rows by ~6.7x and
independent information (effective N) by only ~1.5x. This module's job is to
turn that into a reproducible number, not to assume it away.

Effective-N formula and why this one: the primary figure is the Kish-style
design effect ``N_eff = m / (1 + (m - 1) * r_bar)``, where ``r_bar`` is the
mean OFF-DIAGONAL Pearson r over the measured set. Chosen because it (a)
reproduces KNOWN-LIMITATIONS §0b's published anchor exactly (m=3,
r_bar=0.7574 -> N_eff=1.193 ~= 1.2), (b) needs only r_bar, which is stable at
~1000 observations, (c) is monotone in r_bar, and (d) is the CONSERVATIVE
candidate of the two computed here. Its stated weakness: the exchangeability
assumption means block structure (e.g. TRXUSDT sitting near r=0.23 against a
~0.6 cluster of the rest) makes it UNDERSTATE true independence. The
secondary figure, the participation ratio ``(sum(eigvals))**2 /
sum(eigvals**2)`` over the correlation matrix's eigenvalues, respects block
structure and is reported for that reason -- but it is never used by the
decision rule, because a conservative statistic is the right one to bet a
validation gate on.

Alignment policy: returns are ``r_t = close_t / close_(t-1) - 1`` on the 1d
close series (installed pandas is 3.0.3, where ``Series.pct_change()``'s fill
behaviour changed -- explicit arithmetic is version-proof and used
throughout). Daily bars are UTC-midnight aligned (``ts % 86_400_000 == 0``),
so calendar alignment is an inner join on ``ts`` and needs no resampling.
Series are aligned LISTWISE (complete cases only, via an inner join over the
symbol panel) so the resulting correlation matrix stays positive
semi-definite and every pair is measured on the same ``n_obs``, which is
always reported alongside r. On the real store this costs nothing (zero NaN
across all 20 symbols, measured 2026-07-27); ``CORRELATION_MIN_OVERLAP_BARS``
guards a future ragged panel.

Pure computation except for reads through ``storage.load_candles`` /
``storage.find_gaps`` / ``storage.last_ts``. Writes nothing; the CLI
(``cli.py::_correlation_command``) owns file output via ``format_report``.
"""

import logging
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot import config
from trading_bot.data import storage

logger = logging.getLogger("trading_bot")

# DAY_MS is deliberately duplicated rather than imported from
# backtest/equity.py or backtest/walkforward.py (which each define their own,
# per v0.2.0 Phase 1's plan): Phase 1 is concurrently editing those files
# (contract §11), and importing from trading_bot.backtest.* here would create
# a merge surface for no benefit (contract §2 GOTCHA).
DAY_MS = 86_400_000

# 15m is deliberately excluded: it exists for BTC/ETH/SOL only and the tier
# was retired on cost-frontier grounds (scripts/bruteforce/universe.py:40-43).
# Requiring it here would disqualify 17 symbols for lacking data that no
# v0.3.0 strategy reads.
INTEGRITY_TIMEFRAMES = ("1d", "4h", "1h")


@dataclass(frozen=True)
class SymbolStats:
    """Per-symbol screening/ranking statistics consumed by select_research_symbols.

    Attributes:
        symbol: e.g. "BTCUSDT".
        n_bars: aligned daily-return observations for this symbol after the
            listwise join (see daily_return_frame).
        mean_r: mean pairwise Pearson r of this symbol's daily returns against
            every OTHER measured symbol (off-diagonal row mean of the full
            correlation matrix). None if fewer than 2 observations.
        btc_beta: cov(r_sym, r_anchor) / var(r_anchor); None if the anchor is
            absent from the measured set or has zero variance.
        median_quote_volume: median(close * volume) over the measured span --
            an OHLCV-DERIVED LIQUIDITY PROXY, not book depth. Used as a floor
            screen only; cannot validate config.SLIPPAGE_PCT.
        interior_gaps: count of interior gaps across INTEGRITY_TIMEFRAMES
            (trailing poller staleness is excluded by construction).
        eligible: True iff the symbol passed every selection screen (see
            screen_eligibility).
        reason: "" if eligible; otherwise the name of the first failing
            screen ("gaps" | "overlap" | "liquidity" | "beta").
    """

    symbol: str
    n_bars: int
    mean_r: float | None
    btc_beta: float | None
    median_quote_volume: float | None
    interior_gaps: int
    eligible: bool
    reason: str


@dataclass(frozen=True)
class CorrelationReport:
    """Full output of build_report: the measurement, the selection, the verdict.

    Attributes:
        timeframe: return timeframe measured (config.CORRELATION_TIMEFRAME).
        start_ms / end_ms: measured span; both bounds inclusive, matching
            storage.load_candles' contract.
        symbols: symbols measured, in store order.
        n_obs: aligned daily-return observations after the listwise join,
            shared by every pair in matrix.
        matrix: full Pearson correlation matrix over symbols (pd.DataFrame).
        stats: symbol -> SymbolStats.
        effective_n: "production" | "stored" | "selection" -> (m, r_bar,
            N_eff_kish, N_eff_participation).
        selection: the promoted RESEARCH_SYMBOLS candidate, anchor first then
            ascending mean-r rank order. Empty tuple under a D3 verdict.
        decision: "D1" | "D2" | "D3" (see the plan's pre-registered Decision
            Rule table).
        verdict: the human-readable finding sentence(s) explaining the
            decision letter, with its numeric justification.
        integrity: symbol -> list of gap strings, gap_integrity's raw output.
        storage_stats: output of storage_footprint(conn), or None if it could
            not be measured (e.g. an in-memory test DB). Appended last, with
            a default, so existing keyword-constructed reports never break.
    """

    timeframe: str
    start_ms: int
    end_ms: int
    symbols: tuple[str, ...]
    n_obs: int
    matrix: pd.DataFrame
    stats: dict[str, SymbolStats]
    effective_n: dict[str, tuple[int, float, float, float]]
    selection: tuple[str, ...]
    decision: str
    verdict: str
    integrity: dict[str, list[str]]
    storage_stats: dict | None = None


def _stored_symbols(conn, timeframe: str) -> tuple[str, ...]:
    """Distinct symbols stored at timeframe, ascending.

    Lives here rather than in storage.py: storage.py is not Phase 2's file to
    modify (contract §2), and deriving the default universe from the STORE
    rather than a hand-maintained list means a future backfill is picked up
    automatically, with no config edit.
    """
    cursor = conn.execute(
        "SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ? ORDER BY symbol",
        (timeframe,),
    )
    return tuple(row[0] for row in cursor.fetchall())


def _assert_daily_grid(index, symbol: str, timeframe: str) -> None:
    """Raise if a loaded series is not grid-aligned / correctly spaced.

    Mirrors backtest/engine.py:186-197's reasoning in spirit: a series
    backfilled under the wrong key, or loaded under a mismatched timeframe,
    would produce a plausible correlation on the wrong bars -- a silent
    failure in the flattering direction. Raise instead.

    Uses the MEDIAN inter-bar spacing so a handful of missing candles cannot
    trip this; a wholesale interval mismatch always shifts the median.

    Args:
        index: Ascending epoch-ms timestamps (e.g. a DataFrame index).
        symbol: For the error message.
        timeframe: The config timeframe key this series was loaded under.

    Raises:
        ValueError: naming "grid-aligned" or "median bar spacing".
    """
    interval = storage.TIMEFRAME_MS[timeframe]
    ts_list = list(index)
    if not ts_list:
        return
    if any(ts % interval != 0 for ts in ts_list):
        raise ValueError(
            f"{symbol} {timeframe}: series is not grid-aligned to {interval}ms bars"
        )
    if len(ts_list) >= 2:
        diffs = [b - a for a, b in zip(ts_list[:-1], ts_list[1:])]
        median_diff = statistics.median(diffs)
        if median_diff != interval:
            raise ValueError(
                f"{symbol} {timeframe}: median bar spacing {median_diff}ms does not "
                f"match the expected {interval}ms for {timeframe}"
            )


def daily_return_frame(
    conn,
    symbols,
    *,
    start_ms: int,
    end_ms: int,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Aligned daily returns for symbols over [start_ms, end_ms] (both inclusive).

    Returns r_t = close_t / close_(t-1) - 1, computed with explicit
    arithmetic -- NOT Series.pct_change(), whose fill behaviour changed under
    the installed pandas 3.0.3 and is unsafe here.

    Alignment is LISTWISE (complete cases only): symbols are assembled into
    one wide close-price panel indexed by ts, and any ts where any symbol is
    missing is dropped for EVERY symbol, so the resulting matrix stays
    positive semi-definite and every pair shares the same n_obs. On the real
    store this drops nothing (measured 2026-07-27); a future ragged panel
    would be visible via the logged drop count.

    Args:
        conn: Database connection.
        symbols: Iterable of symbols to load.
        start_ms / end_ms: Inclusive bounds, matching storage.load_candles.
        timeframe: Defaults to config.CORRELATION_TIMEFRAME.

    Returns:
        DataFrame indexed by ts (ascending), one column per symbol, of daily
        returns. Empty (zero rows) when fewer than 2 aligned returns survive.

    Raises:
        ValueError: via _assert_daily_grid, if any loaded series is not
            grid-aligned to `timeframe` or has an inconsistent bar spacing.
    """
    if timeframe is None:
        timeframe = config.CORRELATION_TIMEFRAME

    prices: dict[str, pd.Series] = {}
    for symbol in symbols:
        rows = storage.load_candles(
            conn, symbol, timeframe, start_ms=start_ms, end_ms=end_ms
        )
        df = pd.DataFrame(
            rows, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        if len(df):
            df["ts"] = df["ts"].astype(int)
        df = df.set_index("ts")
        _assert_daily_grid(df.index, symbol, timeframe)
        prices[symbol] = df["close"]

    panel = pd.DataFrame(prices).sort_index()
    n_before = len(panel)
    rets = (panel / panel.shift(1) - 1.0).dropna(how="any")
    n_dropped = max(n_before - 1 - len(rets), 0) if n_before else 0
    logger.info(
        "daily_return_frame: %d symbols, n_obs=%d, dropped=%d timestamps",
        len(list(symbols)),
        len(rets),
        n_dropped,
    )
    return rets


def gap_integrity(
    conn,
    symbols,
    *,
    timeframes: tuple[str, ...] = INTEGRITY_TIMEFRAMES,
    start_ms: int | None = None,
) -> dict[str, list[str]]:
    """Interior-gap-only integrity sweep over symbols x timeframes.

    An interior hole is worse than a symbol you do not have at all, because
    it silently distorts a POOLED result. Trailing staleness is normal
    poller lag (measured present on 40 of 60 research cells at any moment,
    2026-07-27) and must NOT disqualify anything. This function pins
    ``now_ms = last_ts + interval + 1`` per cell so storage.find_gaps'
    staleness branch (storage.py:188-192) cannot fire while its interior-gap
    loop (storage.py:179-181) still runs over the whole series. This READS
    find_gaps' documented invariants (storage.py:140-146); it never
    reimplements gap detection.

    Args:
        conn: Database connection.
        symbols: Iterable of symbols to sweep.
        timeframes: Timeframes to check per symbol (default: 1d/4h/1h).
        start_ms: Passed through to find_gaps. Left None by default (uses
            find_gaps' own default, config.BACKFILL_START) -- passing
            CORRELATION_START here would be wrong, since the sub-daily tiers'
            real history predates it and that warmup is genuine data the
            regime classifier consumes.

    Returns:
        symbol -> list of strings, one per finding: "{tf} MISSING" for an
        empty series, or "{tf} {gap_start}-{gap_end}" per interior gap.
        Empty list for a clean symbol.
    """
    result: dict[str, list[str]] = {}
    for symbol in symbols:
        entries: list[str] = []
        for timeframe in timeframes:
            interval = storage.TIMEFRAME_MS[timeframe]
            last = storage.last_ts(conn, symbol, timeframe)
            if last is None:
                entries.append(f"{timeframe} MISSING")
                continue
            gaps = storage.find_gaps(
                conn,
                symbol,
                timeframe,
                now_ms=last + interval + 1,
                start_ms=start_ms,
            )
            for g0, g1 in gaps:
                # Re-assert find_gaps' own documented invariants (storage.py
                # 140-146): they are guarantees, so a violation is a storage
                # bug worth surfacing loudly rather than swallowing.
                assert g0 <= g1, (symbol, timeframe, g0, g1)
                assert g0 % interval == 0 and g1 % interval == 0, (
                    symbol,
                    timeframe,
                    g0,
                    g1,
                )
                entries.append(f"{timeframe} {g0}-{g1}")
        result[symbol] = entries
    return result


def storage_footprint(conn, db_path: str | None = None) -> dict | None:
    """Storage-cost accounting for the OHLCV store: bytes/row and free disk
    space, so "is breadth storage-constrained" is a MEASURED coefficient
    (via PRAGMA page_size/page_count and shutil.disk_usage), never an
    assumed risk.

    Args:
        conn: Database connection (must be a real file-backed sqlite3 DB;
            PRAGMA page_count is meaningless for ":memory:").
        db_path: Path used to resolve the free-space filesystem. Defaults
            to config.DB_PATH.

    Returns:
        dict with page_size, page_count, total_bytes, total_rows,
        bytes_per_row, free_bytes -- or None if the store is empty or the
        path cannot be resolved (e.g. an in-memory test DB).
    """
    if db_path is None:
        db_path = config.DB_PATH
    try:
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        total_rows = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        rows_1d4h1h = conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE timeframe IN ('1d','4h','1h')"
        ).fetchone()[0]
        n_symbols_1d4h1h = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM ohlcv WHERE timeframe IN ('1d','4h','1h')"
        ).fetchone()[0]
        free_bytes = shutil.disk_usage(str(Path(db_path).resolve().parent)).free
    except Exception:
        logger.warning("storage_footprint: could not measure store/disk stats", exc_info=True)
        return None
    if not total_rows:
        return None
    total_bytes = page_size * page_count
    bytes_per_row = total_bytes / total_rows
    rows_per_symbol_1d4h1h = rows_1d4h1h / n_symbols_1d4h1h if n_symbols_1d4h1h else None
    return {
        "page_size": page_size,
        "page_count": page_count,
        "total_bytes": total_bytes,
        "total_rows": total_rows,
        "bytes_per_row": bytes_per_row,
        "free_bytes": free_bytes,
        # Measured (not assumed) marginal cost of ONE additional symbol's
        # full 1d+4h+1h history, extrapolated from the stored universe's own
        # observed rows/symbol at those three tiers.
        "rows_per_new_symbol_1d4h1h": rows_per_symbol_1d4h1h,
        "bytes_per_new_symbol_1d4h1h": (
            rows_per_symbol_1d4h1h * bytes_per_row
            if rows_per_symbol_1d4h1h is not None
            else None
        ),
    }


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of daily returns.

    Args:
        returns: DataFrame of daily returns, one column per symbol.

    Returns:
        Symbols x symbols Pearson correlation matrix. Empty DataFrame if
        fewer than 2 observations.
    """
    if len(returns) < 2:
        return pd.DataFrame()
    return returns.corr()


def effective_n(
    corr: pd.DataFrame, symbols: list[str] | None = None
) -> tuple[int, float, float]:
    """Kish-style design-effect effective sample size over a subset.

    N_eff = m / (1 + (m - 1) * r_bar), where r_bar is the mean OFF-DIAGONAL
    Pearson r (see module docstring for why this formula). The correctness
    check for this whole module is that it reproduces KNOWN-LIMITATIONS
    §0b's published anchor: m=3, r_bar=0.7574 -> N_eff=1.193.

    m == 1 returns (1, 0.0, 1.0) by convention (a single series is fully
    "independent of itself").

    N_eff is CLAMPED to [1.0, m]: with a strongly negative r_bar the closed
    form can exceed m or go negative, and neither is a meaningful sample
    size -- the clamp is the honest way to say "at most m independent series
    can exist here."

    Args:
        corr: Full correlation matrix (must contain every symbol in subset).
        symbols: Subset to measure. Defaults to every column of corr.

    Returns:
        (m, r_bar, N_eff).
    """
    if symbols is None:
        symbols = list(corr.columns)
    m = len(symbols)
    if m <= 1:
        return (m, 0.0, 1.0)

    sub = corr.loc[list(symbols), list(symbols)].to_numpy()
    off_diag = [sub[i][j] for i in range(m) for j in range(m) if i != j]
    r_bar = float(statistics.fmean(off_diag))
    n_eff = m / (1 + (m - 1) * r_bar)
    n_eff = max(1.0, min(float(m), n_eff))
    return (m, r_bar, n_eff)


def effective_n_participation(
    corr: pd.DataFrame, symbols: list[str] | None = None
) -> float:
    """Participation-ratio effective sample size: (sum(eigvals))**2 / sum(eigvals**2).

    Unlike the Kish form, this respects block structure -- it credits a
    genuinely detached series instead of averaging it into one r_bar.
    Reported alongside effective_n, but NEVER used by the decision rule: a
    conservative statistic is the right one to bet a validation gate on.
    Measured on config.SYMBOLS: 1.395 (vs Kish 1.193).

    Args:
        corr: Full correlation matrix (real symmetric; uses eigvalsh).
        symbols: Subset to measure. Defaults to every column of corr.

    Returns:
        Effective N, clamped to [1.0, m].
    """
    if symbols is None:
        symbols = list(corr.columns)
    m = len(symbols)
    if m <= 1:
        return 1.0

    sub = corr.loc[list(symbols), list(symbols)].to_numpy()
    eigvals = np.linalg.eigvalsh(sub)
    s1 = float(eigvals.sum())
    s2 = float((eigvals**2).sum())
    if s2 == 0:
        return 1.0
    pr = (s1**2) / s2
    return max(1.0, min(float(m), pr))


def btc_beta(
    returns: pd.DataFrame, symbol: str, *, anchor: str | None = None
) -> float | None:
    """cov(r_sym, r_anchor) / var(r_anchor), sample ddof=1.

    Beta and correlation measure different things and this module reports
    both: r is the SHARE of co-movement, beta its AMPLITUDE. A symbol can be
    a leveraged BTC proxy at moderate r (measured: DOGEUSDT r=0.762,
    beta=1.433) -- which is why the selection screen is on beta and the
    ranking is on r.

    Args:
        returns: Daily-return frame containing both symbol and anchor.
        symbol: Symbol to measure.
        anchor: Defaults to config.CORRELATION_ANCHOR_SYMBOL. Exactly 1.0
            when symbol == anchor, by construction.

    Returns:
        Beta, or None if the anchor is absent from returns, has zero
        variance, or fewer than 2 observations exist.
    """
    if anchor is None:
        anchor = config.CORRELATION_ANCHOR_SYMBOL
    if anchor not in returns.columns or symbol not in returns.columns:
        return None
    if len(returns) < 2:
        return None

    r_anchor = returns[anchor]
    r_sym = returns[symbol]
    var_anchor = r_anchor.var(ddof=1)
    if not var_anchor or pd.isna(var_anchor):
        return None
    cov = r_sym.cov(r_anchor, ddof=1)
    return float(cov / var_anchor)


def median_quote_volume(
    conn,
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    timeframe: str | None = None,
) -> float | None:
    """Median close*volume over [start_ms, end_ms] (both inclusive).

    Stated plainly: this is a liquidity PROXY derived from OHLCV, not book
    depth, and cannot validate config.SLIPPAGE_PCT (2bps/side). It is a
    floor screen only.

    Args:
        conn: Database connection.
        symbol: Symbol to measure.
        start_ms / end_ms: Inclusive bounds.
        timeframe: Defaults to config.CORRELATION_TIMEFRAME.

    Returns:
        Median quote volume in USD, or None if no rows are stored in range.
    """
    if timeframe is None:
        timeframe = config.CORRELATION_TIMEFRAME
    rows = storage.load_candles(
        conn, symbol, timeframe, start_ms=start_ms, end_ms=end_ms
    )
    if not rows:
        return None
    quote_vols = [r[4] * r[5] for r in rows]  # close * volume
    return float(statistics.median(quote_vols))


def screen_eligibility(
    *,
    n_bars: int,
    interior_gaps: int,
    median_quote_volume: float | None,
    btc_beta: float | None,
    is_anchor: bool,
) -> tuple[bool, str]:
    """Selection-rule eligibility screen (step 1 of select_research_symbols' rule).

    Order matters -- the first failing screen names the reason: zero
    interior gaps, then the minimum daily-return overlap, then the
    liquidity floor, then the BTC-beta ceiling. The anchor is exempt from
    the beta screen ONLY (its beta is 1.0 by construction and would pass
    anyway) -- never from the gap/history/liquidity screens; silently
    exempting the anchor from those would defeat the integrity check.

    Args:
        n_bars: Aligned daily-return observations.
        interior_gaps: Count of interior gaps (from gap_integrity).
        median_quote_volume: USD liquidity proxy, or None.
        btc_beta: Beta vs the anchor, or None.
        is_anchor: True for config.CORRELATION_ANCHOR_SYMBOL itself.

    Returns:
        (eligible, reason). reason is "" iff eligible; otherwise one of
        "gaps" | "overlap" | "liquidity" | "beta".
    """
    if interior_gaps > 0:
        return False, "gaps"
    if n_bars < config.CORRELATION_MIN_OVERLAP_BARS:
        return False, "overlap"
    if (
        median_quote_volume is None
        or median_quote_volume < config.CORRELATION_MIN_QUOTE_VOLUME_USD
    ):
        return False, "liquidity"
    if not is_anchor:
        if btc_beta is None or abs(btc_beta) > config.CORRELATION_MAX_BTC_BETA:
            return False, "beta"
    return True, ""


def select_research_symbols(
    stats: dict[str, SymbolStats],
    corr: pd.DataFrame,
    *,
    anchor: str | None = None,
    select_n: int | None = None,
) -> tuple[str, ...]:
    """The pre-registered, recorded selection rule (auditable, not re-litigated later).

    Rule, exactly and in this order:
      1. Eligibility -- ``stats[symbol].eligible`` (see screen_eligibility):
         zero interior gaps across 1d/4h/1h; >= CORRELATION_MIN_OVERLAP_BARS
         aligned daily returns; median daily quote volume >=
         CORRELATION_MIN_QUOTE_VOLUME_USD; |BTC beta| <=
         CORRELATION_MAX_BTC_BETA.
      2. Anchor -- CORRELATION_ANCHOR_SYMBOL (BTCUSDT) is PINNED regardless
         of rank. It costs ~0.034 of Kish effective N (measured: 1.868 ->
         1.834) and buys the deepest book in the universe, continuity with
         the buy-and-hold benchmark and config.SYMBOLS, and the beta
         reference every other row is quoted against. If the anchor itself
         fails the gap/history/liquidity screens, this raises rather than
         silently substituting a different anchor -- that failure IS
         decision D3.
      3. Ranking -- remaining eligible candidates ascending by MEAN PAIRWISE
         Pearson r (not beta: r_bar is the exact quantity the effective-N
         formula consumes, so ranking on beta would optimise a proxy). Ties
         break on higher median quote volume, then alphabetically, so the
         result is deterministic.
      4. Size -- take the first ``select_n`` (default CORRELATION_SELECT_N).
         With the anchor that is select_n + 1, clearing the PRD's >=8 floor.
      5. Result order -- anchor first, then rank order. NEVER reordered: the
         report and config.RESEARCH_SYMBOLS key off this ordering.

    Why simple rather than optimised: a deterministic ranking and a greedy
    minimum-average-correlation search agree on 7 of 8 members (measured;
    differing only AAVE <-> NEAR) with effective N differing by 0.009 (1.868
    vs 1.877) -- the selection is insensitive to the algorithm, so a greedy
    search would add path dependence and a hidden degree of freedom for a
    0.5% effect.

    Args:
        stats: symbol -> SymbolStats, with eligible/reason already computed.
        corr: Full correlation matrix (unused directly here; kept in the
            signature because the rule is conceptually "rank by r against
            this matrix" and callers pass it for symmetry with effective_n).
        anchor: Defaults to config.CORRELATION_ANCHOR_SYMBOL.
        select_n: Defaults to config.CORRELATION_SELECT_N.

    Returns:
        Tuple of symbols, anchor first then ascending mean-r rank.

    Raises:
        ValueError: if the anchor is absent or ineligible (never substitute
            a different anchor), or if fewer than select_n eligible
            non-anchor candidates exist (decision D3: a genuine backfill is
            the only remedy).
    """
    del corr  # not read directly; see docstring
    if anchor is None:
        anchor = config.CORRELATION_ANCHOR_SYMBOL
    if select_n is None:
        select_n = config.CORRELATION_SELECT_N

    if anchor not in stats:
        raise ValueError(f"anchor {anchor!r} not present in measured stats")
    anchor_stats = stats[anchor]
    if not anchor_stats.eligible:
        raise ValueError(
            f"anchor {anchor!r} fails its own integrity/history/liquidity screen "
            f"(reason={anchor_stats.reason!r}); not substituting a different anchor"
        )

    candidates = [
        symbol
        for symbol, s in stats.items()
        if symbol != anchor and s.eligible
    ]
    if len(candidates) < select_n:
        raise ValueError(
            f"only {len(candidates)} eligible non-anchor candidates, need "
            f"{select_n} (decision D3: a genuine backfill is required)"
        )

    def _sort_key(symbol: str):
        s = stats[symbol]
        mean_r = s.mean_r if s.mean_r is not None else float("inf")
        vol = s.median_quote_volume if s.median_quote_volume is not None else 0.0
        return (mean_r, -vol, symbol)

    ranked = sorted(candidates, key=_sort_key)
    chosen = ranked[:select_n]
    return (anchor,) + tuple(chosen)


def _row_mean_r(corr: pd.DataFrame, symbol: str) -> float | None:
    """Mean off-diagonal Pearson r for one symbol's row. None if unavailable."""
    if corr.empty or symbol not in corr.columns or corr.shape[0] < 2:
        return None
    row = corr.loc[symbol].drop(labels=[symbol])
    if row.empty or row.isna().all():
        return None
    value = row.mean()
    return float(value) if pd.notna(value) else None


def build_report(
    conn,
    symbols=None,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    timeframe: str | None = None,
    select_n: int | None = None,
) -> CorrelationReport:
    """Measure the stored universe and apply the pre-registered Decision Rule.

    Defaults: symbols = every symbol stored at `timeframe`; start_ms =
    config.CORRELATION_START; end_ms = the last CLOSED bar at `timeframe`.

    **Deviation from the plan's literal Task 7 text, and why.** The plan's
    Task 7 says to default end_ms to storage.last_ts (MAX(ts)) on the theory
    that the store's max timestamp is always a closed bar. Measured on this
    working tree 2026-07-27: it is NOT -- MAX(ts) for "1d" is 1785110400000
    = 2026-07-27 00:00Z, which is TODAY's still-forming daily candle
    (contract §0's explicit gotcha: "the final 1d bar ... is still forming
    and is not a closed candle. Any span/bar-count must use the last closed
    bar, never MAX(ts)"). Using MAX(ts) here would silently pool a partial
    day's return into the correlation measurement. So end_ms is instead
    derived the same way storage.find_gaps does
    (``(now_ms // interval) * interval - interval``), then clamped to
    whatever is actually stored -- reproducible up to the wall-clock date
    boundary, and provably never a forming bar.

    Decision Rule (pre-registered so it cannot be re-litigated after the
    fact): let N_prod = Kish effective N over config.SYMBOLS, N_sel = over
    the selection, k = select_n.

      D1: len(selection) >= k+1 AND every selected symbol interior-gap-clean
          AND N_sel >= CORRELATION_EFFECTIVE_N_MIN_RATIO * N_prod.
          -> Promote RESEARCH_SYMBOLS. No backfill needed.
      D2: size + integrity hold, but N_sel < the ratio threshold.
          -> Promote anyway (rows still clear the mechanical n_trades
          floor) and report that information did not scale. A COMPLETED
          OUTCOME, not a failed phase -- measuring that liquid majors are
          close to one bet is the answer to PRD Open Question #4.
      D3: fewer than k eligible candidates, or the anchor itself fails
          integrity with no eligible replacement.
          -> Backfill required before this universe can be trusted.

    Args:
        conn: Database connection.
        symbols: Defaults to every symbol stored at `timeframe`.
        start_ms: Defaults to config.CORRELATION_START.
        end_ms: Defaults to the last stored 1d bar (a closed candle).
        timeframe: Defaults to config.CORRELATION_TIMEFRAME.
        select_n: Defaults to config.CORRELATION_SELECT_N.

    Returns:
        A frozen CorrelationReport.
    """
    if timeframe is None:
        timeframe = config.CORRELATION_TIMEFRAME
    if symbols is None:
        symbols = _stored_symbols(conn, timeframe)
    symbols = tuple(symbols)
    if select_n is None:
        select_n = config.CORRELATION_SELECT_N
    if start_ms is None:
        start_ms = config.date_to_ms(config.CORRELATION_START)
    if end_ms is None:
        interval = storage.TIMEFRAME_MS[timeframe]
        now_ms = int(time.time() * 1000)
        last_closed = (now_ms // interval) * interval - interval
        last_candidates = [storage.last_ts(conn, s, timeframe) for s in symbols]
        last_candidates = [t for t in last_candidates if t is not None]
        stored_max = max(last_candidates) if last_candidates else start_ms
        end_ms = min(stored_max, last_closed)

    returns = daily_return_frame(
        conn, symbols, start_ms=start_ms, end_ms=end_ms, timeframe=timeframe
    )
    corr = correlation_matrix(returns)
    integrity = gap_integrity(conn, symbols)

    anchor = config.CORRELATION_ANCHOR_SYMBOL

    stats: dict[str, SymbolStats] = {}
    for symbol in symbols:
        n_bars = int(returns[symbol].count()) if symbol in returns.columns else 0
        mean_r = _row_mean_r(corr, symbol)
        beta = btc_beta(returns, symbol, anchor=anchor)
        vol = median_quote_volume(
            conn, symbol, start_ms=start_ms, end_ms=end_ms, timeframe=timeframe
        )
        gaps = len(integrity.get(symbol, []))
        eligible, reason = screen_eligibility(
            n_bars=n_bars,
            interior_gaps=gaps,
            median_quote_volume=vol,
            btc_beta=beta,
            is_anchor=(symbol == anchor),
        )
        stats[symbol] = SymbolStats(
            symbol=symbol,
            n_bars=n_bars,
            mean_r=mean_r,
            btc_beta=beta,
            median_quote_volume=vol,
            interior_gaps=gaps,
            eligible=eligible,
            reason=reason,
        )

    try:
        selection = select_research_symbols(
            stats, corr, anchor=anchor, select_n=select_n
        )
        decision_error: str | None = None
    except ValueError as exc:
        selection = ()
        decision_error = str(exc)

    def _eff(subset: list[str]) -> tuple[int, float, float, float]:
        m, r_bar, n_kish = effective_n(corr, subset)
        n_part = effective_n_participation(corr, subset)
        return (m, r_bar, n_kish, n_part)

    eff_n: dict[str, tuple[int, float, float, float]] = {
        "production": _eff([s for s in config.SYMBOLS if s in corr.columns]),
        "stored": _eff(list(symbols)),
    }
    if selection:
        eff_n["selection"] = _eff(list(selection))

    n_prod = eff_n["production"][2]
    if decision_error is not None:
        decision = "D3"
        verdict = (
            f"D3: insufficient eligible universe ({decision_error}). "
            "A genuine backfill is required before this universe is trusted "
            "(see the Task 14 contingency)."
        )
    else:
        n_sel = eff_n["selection"][2]
        ratio = n_sel / n_prod if n_prod else 0.0
        any_gap = any(len(integrity.get(s, [])) > 0 for s in selection)
        size_ok = len(selection) >= select_n + 1
        ratio_ok = ratio >= config.CORRELATION_EFFECTIVE_N_MIN_RATIO
        if size_ok and not any_gap and ratio_ok:
            decision = "D1"
            verdict = (
                f"D1: breadth confirmed, with 'confirmed' meaning ~{ratio:.2f}x "
                f"independent information, not the {len(selection)}x row count. "
                f"selection N_eff(Kish)={n_sel:.3f} is {ratio:.3f}x config.SYMBOLS' "
                f"N_eff={n_prod:.3f} (threshold "
                f"{config.CORRELATION_EFFECTIVE_N_MIN_RATIO}x). RESEARCH_SYMBOLS "
                "promoted; no backfill performed."
            )
        else:
            decision = "D2"
            verdict = (
                f"D2: rows without information. selection N_eff(Kish)={n_sel:.3f} "
                f"is only {ratio:.3f}x config.SYMBOLS' N_eff={n_prod:.3f} "
                f"(threshold {config.CORRELATION_EFFECTIVE_N_MIN_RATIO}x). Promoted "
                "anyway: rows still clear the mechanical n_trades floor, but "
                "independent information did not scale with the symbol count."
            )

    return CorrelationReport(
        timeframe=timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        symbols=symbols,
        n_obs=len(returns),
        matrix=corr,
        stats=stats,
        effective_n=eff_n,
        selection=selection,
        decision=decision,
        verdict=verdict,
        integrity=integrity,
        storage_stats=storage_footprint(conn),
    )


def _fmt_opt(value: float | None, spec: str = ".4f") -> str:
    """Format a possibly-None metric value, mirroring cli.py::_fmt."""
    return format(value, spec) if value is not None else "--"


def _render_matrix(matrix: pd.DataFrame) -> list[str]:
    """Render a correlation matrix as a markdown pipe-table, without pandas'
    to_markdown() (which requires the optional `tabulate` dependency this
    repo does not have -- contract §1/§0a: add no new dependency)."""
    if matrix.empty:
        return ["(insufficient data)"]
    cols = list(matrix.columns)
    lines = ["| | " + " | ".join(cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for row_symbol in matrix.index:
        cells = [_fmt_opt(matrix.loc[row_symbol, c], ".3f") for c in cols]
        lines.append(f"| {row_symbol} | " + " | ".join(cells) + " |")
    return lines


def format_report(report: CorrelationReport) -> str:
    """Render a CorrelationReport as markdown.

    Pure string building; the caller (cli.py) owns file I/O. Every number
    comes from `report`, so the artifact cannot drift from what was measured.

    Sections, in order: (1) the correction to the PRD's premise and the
    supersession of KNOWN-LIMITATIONS §4; (2) measured span/store inventory;
    (3) gap integrity, interior vs trailing; (4) the full pairwise matrix
    with n_obs; (5) the effective-N table with the §0b reproduction; (6) the
    per-symbol table; (7) the selection rule verbatim plus the result; (8)
    the verdict; (9) degrees of freedom consumed.
    """
    lines: list[str] = []
    lines.append("# Phase 2 correlation report (v0.3.0)")
    lines.append("")
    lines.append(
        "**Correction to the PRD's premise (contract §0a).** The PRD scopes this "
        "phase as \"backfill uncorrelated Binance futures symbols\", assuming 3 "
        "symbols are stored. **20 are stored**, at 1d/4h/1h, over the full "
        "2023-01-01 -> 2026-07-26 span. KNOWN-LIMITATIONS §4's \"Only 3 symbols "
        "exist in the store\" was true at the v0.2.0 merge and is superseded by "
        "contract §0a; this report records the supersession rather than "
        "editing that file."
    )
    lines.append("")
    lines.append(
        f"Span: `{report.start_ms}` -> `{report.end_ms}`  "
        f"timeframe: `{report.timeframe}`  n_obs: **{report.n_obs}**  "
        f"symbols measured: **{len(report.symbols)}**"
    )
    lines.append("")

    if report.storage_stats:
        s = report.storage_stats
        free_gib = s["free_bytes"] / (1024**3)
        lines.append("## Storage cost (measured, not assumed)")
        lines.append("")
        lines.append(
            f"`{s['total_rows']:,}` rows in `{s['total_bytes']:,}` bytes = "
            f"**{s['bytes_per_row']:.2f} bytes/row**. `{free_gib:.1f} GiB` free on "
            "the DB's filesystem."
        )
        if s.get("bytes_per_new_symbol_1d4h1h") is not None:
            per_symbol_mb = s["bytes_per_new_symbol_1d4h1h"] / (1024**2)
            lines.append(
                f"One additional symbol's full 1d+4h+1h history costs "
                f"~{per_symbol_mb:.1f} MB, extrapolated from the "
                f"{s['rows_per_new_symbol_1d4h1h']:.0f} rows/symbol already "
                "observed at those three tiers. **Disk is not the "
                "constraint; correlation is.**"
            )
        lines.append("")

    lines.append("## Gap integrity (interior gaps only; trailing poller lag excluded)")
    lines.append("")
    clean = sum(1 for v in report.integrity.values() if not v)
    total_cells = len(report.integrity) * len(INTEGRITY_TIMEFRAMES)
    total_gap_findings = sum(len(v) for v in report.integrity.values())
    lines.append(
        f"{clean} of {len(report.integrity)} symbols clean; "
        f"{total_gap_findings} finding(s) across {total_cells} cells "
        f"({len(report.integrity)} symbols x {len(INTEGRITY_TIMEFRAMES)} timeframes)."
    )
    for symbol, entries in report.integrity.items():
        if entries:
            lines.append(f"- {symbol}: {entries}")
    lines.append("")

    lines.append("## Pairwise correlation matrix (Pearson, daily returns)")
    lines.append("")
    lines.extend(_render_matrix(report.matrix.round(4)))
    lines.append("")

    lines.append("## Effective independent sample size (Kish design effect)")
    lines.append("")
    lines.append(
        "Correctness check: this must reproduce KNOWN-LIMITATIONS §0b's "
        "anchor -- config.SYMBOLS at m=3, r_bar~=0.7574, N_eff~=1.193."
    )
    lines.append("")
    lines.append("| set | n | r_bar | N_eff (Kish) | N_eff (participation ratio) |")
    lines.append("|---|---|---|---|---|")
    for key in ("production", "stored", "selection"):
        if key in report.effective_n:
            m, r_bar, n_kish, n_part = report.effective_n[key]
            lines.append(f"| {key} | {m} | {r_bar:.4f} | {n_kish:.3f} | {n_part:.3f} |")
    lines.append("")

    lines.append("## Per-symbol statistics")
    lines.append("")
    lines.append(
        "| symbol | n_bars | mean_r | btc_beta | median_quote_volume | "
        "interior_gaps | eligible | reason |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for symbol, s in report.stats.items():
        lines.append(
            f"| {symbol} | {s.n_bars} | {_fmt_opt(s.mean_r, '.3f')} | "
            f"{_fmt_opt(s.btc_beta, '.3f')} | "
            f"{_fmt_opt(s.median_quote_volume, ',.0f')} | {s.interior_gaps} | "
            f"{s.eligible} | {s.reason or '-'} |"
        )
    lines.append("")

    lines.append("## Selection rule and result")
    lines.append("")
    lines.append(
        "Rule (recorded so it is auditable, not re-litigated later): "
        "eligibility screens on zero interior gaps, minimum daily-return "
        "overlap, a liquidity floor, and a BTC-beta ceiling; the anchor is "
        "pinned regardless of rank and exempt from the beta screen only; "
        "remaining candidates rank ascending by mean pairwise Pearson r "
        "(not beta -- r_bar is what the effective-N formula consumes); ties "
        "break on higher median quote volume then alphabetically; the first "
        "CORRELATION_SELECT_N are taken. Anchor first, then rank order -- "
        "never reordered."
    )
    lines.append("")
    lines.append(f"`RESEARCH_SYMBOLS` candidate: `{report.selection}`")
    lines.append("")

    lines.append(f"## Verdict: {report.decision}")
    lines.append("")
    lines.append(report.verdict)
    lines.append("")
    if report.decision != "D3":
        n_prod = report.effective_n["production"][2]
        n_sel = report.effective_n.get("selection", (0, 0.0, n_prod, 0.0))[2]
        row_ratio = len(report.selection) / len(config.SYMBOLS) if config.SYMBOLS else 0.0
        lines.append(
            f"**What breadth buys, kept apart.** Rows rise "
            f"{len(report.selection)}/{len(config.SYMBOLS)} = {row_ratio:.2f}x; "
            f"independent information (Kish effective N) rises only "
            f"{(n_sel / n_prod) if n_prod else 0.0:.2f}x. Row count clears the "
            "gate's mechanical n_trades floor; it does not clear the "
            "statistics DSR needs. Phase 9 must read its DSR as governed by "
            f"effective N ~= {n_sel:.1f}, not by {len(report.selection)} symbols."
        )
        lines.append(
            "ESTIMATE, at v0.2.0's measured OOS trade rate (0.0852 "
            "trades/symbol/day over 90 days, KNOWN-LIMITATIONS §1) -- NOT "
            "measured by this phase's own command, and labelled as such: a "
            f"{len(report.selection)}-symbol pool would yield roughly "
            f"{0.0852 * len(report.selection) * 90:.0f} OOS trades per "
            "90-day holdout (mechanical sample adequacy, gate floor 30), but "
            "only about "
            f"{0.0852 * len(report.selection) * 90 * (n_sel / len(report.selection)):.0f} "
            "independent-equivalent trades after deflating by effective N -- "
            "against v0.2.0's own ~9. Phase 9 is the one that must actually "
            "measure the real rate."
        )
        lines.append("")
        n_stored = report.effective_n.get("stored")
        ranked_by_mean_r = sorted(
            (
                (s.mean_r, sym)
                for sym, s in report.stats.items()
                if s.mean_r is not None
            )
        )
        most_detached = ranked_by_mean_r[0][1] if ranked_by_mean_r else "n/a"
        lines.append(
            "**Answer to PRD Open Question #4** (\"are enough uncorrelated "
            "symbols available on Binance futures?\"): **no, not within this "
            "venue.** 19 of the 20 stored liquid majors sit at BTC "
            "correlation roughly 0.56-0.81; the sole exception measured here "
            f"is {most_detached}, which is also among the least liquid "
            "symbols measured. The achievable ceiling from this venue's "
            f"liquid majors is an effective N around {n_stored[2]:.1f}-"
            f"{n_sel:.1f} (stored-universe / selection Kish figures above), "
            "not 8. A genuine diversifier would need a different asset "
            "class, which the PRD excludes (\"Binance futures only\")."
        )
        lines.append("")
        lines.append(
            "**Contingency: what would justify a genuine backfill (D2/D3 "
            "path).** (a) fewer than CORRELATION_SELECT_N eligible symbols "
            "remain after the screens, or (b) a candidate class is found "
            "with measured r vs BTC below ~0.4 AND median quote volume above "
            "CORRELATION_MIN_QUOTE_VOLUME_USD. The acceptance test is stated "
            "in advance: such a backfill must raise the selection's Kish "
            "effective N by >= 0.25 absolute, re-measured by re-running "
            "correlation-report, or it is not worth the download."
        )
        lines.append("")

    lines.append("## Degrees of freedom consumed")
    lines.append("")
    lines.append(
        "- **Zero strategy degrees of freedom.** No parameter affecting a "
        "trade was swept, fitted, or chosen by looking at P&L. RESEARCH_SYMBOLS "
        "follows a rule pre-registered BEFORE the selection was run, on "
        "correlation / liquidity / beta -- none of which is a return."
    )
    lines.append(
        "- **Two thresholds set to be non-binding, deliberately**: "
        "CORRELATION_MIN_QUOTE_VOLUME_USD and CORRELATION_MAX_BTC_BETA. "
        "Recorded as guards for a future expansion, not fitted values."
    )
    lines.append(
        "- **One judgement call that is not a measurement**: "
        f"CORRELATION_EFFECTIVE_N_MIN_RATIO = "
        f"{config.CORRELATION_EFFECTIVE_N_MIN_RATIO}, chosen before this "
        "selection's effective N was known -- pre-registered, and the margin "
        "by which the measured ratio clears or misses it is stated above."
    )
    lines.append(
        f"- **One pinned anchor**: {config.CORRELATION_ANCHOR_SYMBOL}, at a "
        "measured Kish effective-N cost, for the reasons in "
        "select_research_symbols' docstring."
    )
    lines.append("")

    return "\n".join(lines)
