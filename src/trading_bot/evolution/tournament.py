"""
Fitness, gate tiering and selection (v0.3.0 Phase 6).

Pure functions over OracleResult. No I/O, no RNG creation — `rng` is always
passed in, because all randomness lives in the parent process (that single
decision is what makes parallelism and reproducibility compatible).

TWO RULES THAT LOOK LIKE DETAILS AND ARE NOT:

A2 — FITNESS IS `excess_sharpe`, NEVER `dsr`. DSR depends on n_trials, which
grows through a campaign, so a DSR fitness would make an UNCHANGED candidate
look worse in generation 30 than in generation 1 for reasons unrelated to the
candidate. Selection ranks a candidate property; DSR decides the verdict.

A4 — FAILING "BEATS BUY-AND-HOLD" DEMOTES, IT DOES NOT ELIMINATE. The v0.2.0
seed itself fails it (+3.45% annualized vs the basket's +29.4%,
KNOWN-LIMITATIONS §0). Hard elimination makes generation 0 extinct, so a
beats_benchmark_* failure is a rank TIER (C), and tier C is selectable.

The ranking key is a TOTAL order ending in -member_index. walkforward.py:60-63
records what happens without one: ten of twelve combos produced identical trade
lists, fold winners were decided by itertools.product ordering over tied
expectancies, and the DSR was deflated for configurations that were never
distinct. Here the failure mode would be worse — under a process pool, ties
broken on completion order make SELECTION depend on machine speed.
"""

import logging
import math
from dataclasses import dataclass

from trading_bot import config
from trading_bot.backtest.walkforward import GATE_CONDITIONS

logger = logging.getLogger("trading_bot")

__all__ = (
    "Fitness",
    "TIER_ORDER",
    "diversity_guard_needed",
    "elites",
    "rank_key",
    "ranked",
    "score_member",
    "select_parents",
    "unique_fraction",
)

# Best first. A: every gate condition passes. B: both beats_benchmark_* pass but
# another condition fails (the honest "has an edge, not yet significant" tier).
# C: valid metrics but a beats_benchmark_* fails — the seed's own tier. D:
# undefined (no trades, no Sharpe, or an evaluation error).
TIER_ORDER = ("A", "B", "C", "D")

_BENCH_CONDITIONS = ("beats_benchmark_return", "beats_benchmark_sharpe")


@dataclass(frozen=True)
class Fitness:
    """One candidate's selectable score.

    `score` is `excess_sharpe` and is -inf for tier D, so an undefined candidate
    can never out-rank a defined one no matter how the tie-breakers fall.
    `excess_ann_return` is a SECONDARY key only: annualized return over a 90-day
    window-OOS is a 90-day extrapolation and is not quotable as a level, but the
    COMPARISON against the identically-annualized basket over the identical span
    is valid, which is all a tie-break needs.
    """

    tier: str
    score: float
    excess_sharpe: float | None
    excess_ann_return: float | None
    n_trades: int

    @property
    def tier_rank(self) -> int:
        return TIER_ORDER.index(self.tier)


def _excess(value: float | None, baseline: float | None) -> float | None:
    """value - baseline, or None if either side is undefined.

    Never substitutes 0.0 for a missing benchmark: "the null could not be
    computed" is not "the null returned nothing".
    """
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _tier(result) -> str:
    """The gate tier of one OracleResult. Reads `gate`; re-derives nothing."""
    if result.error:
        return "D"
    if result.sharpe is None or not result.n_trades:
        return "D"
    gate = result.gate or {}
    missing = [name for name in GATE_CONDITIONS if name not in gate]
    if missing:
        # A short gate dict means Phase 1 renamed a condition or the oracle
        # returned an error shape without setting `error`. Either is a bug worth
        # a loud tier-D rather than a silent mis-tiering.
        logger.warning(
            "gate dict is missing condition(s) %s; tiering as D", missing
        )
        return "D"
    if all(gate.values()):
        return "A"
    if all(gate[name] for name in _BENCH_CONDITIONS):
        return "B"
    return "C"


def score_member(result) -> Fitness:
    """Fitness for one OracleResult.

    Benchmark-relative and trial-count free by construction (A2): the basket's
    Sharpe is measured on the SAME span by the same code path, so the difference
    is comparable across generations even though the windows differ (A3 still
    forbids comparing across generations for a VERDICT — that is what the audit
    round is for).
    """
    tier = _tier(result)
    ex_sharpe = _excess(result.sharpe, result.bench_sharpe)
    ex_ann = _excess(result.ann_return_pct, result.bench_ann_return_pct)
    if tier == "D" or ex_sharpe is None:
        score = -math.inf
    else:
        score = ex_sharpe
    return Fitness(
        tier=tier,
        score=score,
        excess_sharpe=ex_sharpe,
        excess_ann_return=ex_ann,
        n_trades=int(result.n_trades or 0),
    )


def rank_key(fitness: Fitness, member_index: int) -> tuple:
    """A TOTAL order, best first under reverse=True.

    (-tier_rank, score, excess_ann_return, n_trades, -member_index):
      - tier dominates score, so a tier-A candidate outranks a tier-B one with a
        higher excess Sharpe. The gate is the oracle; score only orders within a
        tier.
      - the final -member_index guarantees no tie is ever broken by dict
        iteration order, future completion order, or list order. Lower index
        wins, deterministically.
    """
    ex_ann = fitness.excess_ann_return
    return (
        -fitness.tier_rank,
        fitness.score,
        -math.inf if ex_ann is None else ex_ann,
        fitness.n_trades,
        -int(member_index),
    )


def ranked(scored) -> list:
    """Sort [(member_index, Fitness, payload), ...] best first.

    `payload` is whatever the caller wants carried along (a Member, a dict); this
    module never inspects it, so ranking cannot depend on anything but the
    Fitness and the index.
    """
    return sorted(scored, key=lambda item: rank_key(item[1], item[0]), reverse=True)


def elites(scored, *, n: int | None = None) -> list:
    """The best `n` entries, ranked. Elites are carried UNCHANGED — and re-scored
    next generation on that generation's window, which charges a trial (A6):
    carrying an elite is not free."""
    if n is None:
        n = config.EVO_ELITES
    if n <= 0:
        return []
    return ranked(scored)[:n]


def select_parents(scored, rng, *, k: int | None = None, n: int) -> list:
    """k-way tournament selection, sampled WITH replacement.

    `rng.choices`, not `rng.sample`: with replacement is the intended selection
    pressure, and swapping in `sample` would silently change it (and fail
    outright once k exceeds the population).

    Tier C is SELECTABLE (A4). Eliminating it would empty generation 0, because
    the seed itself fails beats_benchmark_*. A population that is entirely tier D
    is reported by the caller via diversity_guard_needed and re-bred from the
    previous generation's elites; it is not a crash here.

    Args:
        scored: [(member_index, Fitness, payload), ...].
        rng: A seeded random.Random, owned by the parent.
        k: Tournament size (default config.EVO_TOURNAMENT_K).
        n: How many parents to return.
    """
    if not scored:
        raise ValueError("select_parents needs a non-empty scored population")
    if k is None:
        k = config.EVO_TOURNAMENT_K
    k = max(1, min(int(k), len(scored)))
    out = []
    for _ in range(max(0, int(n))):
        bracket = rng.choices(scored, k=k)
        out.append(ranked(bracket)[0])
    return out


def unique_fraction(graph_hashes) -> float:
    """Distinct graph hashes / total. 1.0 for an empty population (nothing has
    collapsed yet), so the guard cannot fire on a generation that does not exist."""
    hashes = list(graph_hashes)
    if not hashes:
        return 1.0
    return len(set(hashes)) / len(hashes)


def diversity_guard_needed(graph_hashes, *, minimum: float | None = None) -> bool:
    """Whether the population has collapsed onto too few distinct genomes.

    A collapsed population still charges a full trial per member, so it buys DSR
    penalty with no search — the most expensive way to learn nothing.
    """
    if minimum is None:
        minimum = config.EVO_MIN_UNIQUE_FRACTION
    return unique_fraction(graph_hashes) < minimum


def all_tier_d(scored) -> bool:
    """True when nothing in the generation produced a defined metric."""
    return bool(scored) and all(f.tier == "D" for _, f, _ in scored)
