"""Strategy registry and parameter-grid expansion.

A strategy is declared once, as a ``build`` function plus a grid of parameter
values, and the runner takes care of everything else: causality checking, the
per-symbol sweep, split enforcement, trial counting, and the leaderboard.

Declaring one looks like this::

    from registry import register
    import indicators as ta
    from core import Plan, LONG, SHORT

    @register(
        family="trend",
        grid={"chan": [20, 55], "k": [1.5, 2.5], "adx_min": [20, 25]},
        rationale="Canonical Donchian breakout, ADX-confirmed.",
    )
    def donchian_adx(ctx, chan, k, adx_min):
        f4 = ctx.frame("4h")
        hi = ctx.align(ta.rolling_high(f4, chan), "4h")
        atr = ctx.align(ta.atr(f4, 14), "4h")
        adx = ctx.align(ta.adx(ctx.frame("1d"), 14), "1d")
        close = ctx.trigger["close"].to_numpy()
        entry = np.zeros(ctx.n, np.int8)
        entry[(close > hi) & (adx > adx_min)] = LONG
        return Plan(entry=entry, stop_dist=k * atr)

TRIAL COUNTING
--------------
``n_trials`` for the Deflated Sharpe Ratio is the TOTAL number of
(strategy, param-combo) evaluations performed across the whole search, not the
number for one strategy. A leaderboard's top entry was selected out of all of
them, and DSR exists precisely to charge for that selection. The runner sums
``combo_count()`` over every registered strategy and passes the total.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

# Families are just labels for grouping the report, but keeping them to a known
# set stops the leaderboard fragmenting into 30 one-member groups.
FAMILIES = (
    "trend",         # breakouts, MA structure, channel following
    "momentum",      # ROC/relative strength, time-series momentum
    "meanrev",       # band fades, oscillator extremes, z-score reversion
    "volatility",    # squeeze release, ATR expansion, vol regime shifts
    "structure",     # support/resistance, ranges, pivots
    "candlestick",   # single/multi-bar reversal and continuation bars
    "chartpattern",  # double top/bottom, triangle, flag, head & shoulders
    "ensemble",      # regime-switched or vote-combined composites
    "carry",         # funding/basis (needs data the DB lacks; placeholder)
)


@dataclass(frozen=True)
class Strategy:
    """One registered, sweepable strategy.

    Attributes:
        name: Unique identifier; becomes the leaderboard key.
        build: ``build(ctx, **params) -> Plan``.
        family: One of FAMILIES.
        grid: param name -> list of values to sweep. The cartesian product is
            the strategy's trial count.
        trigger_tf: Timeframe the Plan is indexed on (execution granularity).
        timeframes: Every timeframe ``build`` reads. Must include trigger_tf.
        rationale: WHY this should work, in one or two sentences. Required --
            an unmotivated strategy in a 10,000-combo sweep is just noise with
            a name, and the report needs to state the prior.
        long_only: Suppress short entries (some edges are genuinely one-sided).
        max_hold_bars: Override the default time stop, in trigger bars.
    """

    name: str
    build: Callable
    family: str
    grid: dict[str, list] = field(default_factory=dict)
    trigger_tf: str = "1h"
    timeframes: tuple[str, ...] = ("1d", "4h", "1h")
    rationale: str = ""
    long_only: bool = False
    max_hold_bars: int | None = None

    def combos(self) -> list[dict]:
        """Every parameter combination, as a list of kwargs dicts."""
        if not self.grid:
            return [{}]
        keys = list(self.grid)
        return [
            dict(zip(keys, values))
            for values in itertools.product(*(self.grid[k] for k in keys))
        ]

    def combo_count(self) -> int:
        n = 1
        for values in self.grid.values():
            n *= len(values)
        return n


ALL: dict[str, Strategy] = {}


def register(
    *,
    family: str,
    grid: dict[str, list] | None = None,
    trigger_tf: str = "1h",
    timeframes: tuple[str, ...] = ("1d", "4h", "1h"),
    rationale: str,
    long_only: bool = False,
    max_hold_bars: int | None = None,
    name: str | None = None,
):
    """Decorator registering a build function as a sweepable strategy.

    Raises:
        ValueError: On a duplicate name, an unknown family, a trigger timeframe
            missing from ``timeframes``, an empty rationale, or a grid axis with
            no values -- all of which would otherwise fail confusingly deep
            inside the sweep.
    """

    def deco(fn: Callable) -> Callable:
        key = name or fn.__name__
        if key in ALL:
            raise ValueError(
                f"strategy {key!r} is already registered (by "
                f"{ALL[key].build.__module__}); pick a distinct name"
            )
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
        if trigger_tf not in timeframes:
            raise ValueError(
                f"{key}: trigger_tf {trigger_tf!r} must appear in timeframes {timeframes}"
            )
        if not rationale.strip():
            raise ValueError(f"{key}: a rationale is required")
        for axis, values in (grid or {}).items():
            if not values:
                raise ValueError(f"{key}: grid axis {axis!r} has no values")
        ALL[key] = Strategy(
            name=key, build=fn, family=family, grid=dict(grid or {}),
            trigger_tf=trigger_tf, timeframes=tuple(timeframes),
            rationale=rationale.strip(), long_only=long_only,
            max_hold_bars=max_hold_bars,
        )
        return fn

    return deco


def total_trials(strategies=None) -> int:
    """Sum of parameter combos across strategies -- the DSR trial count."""
    pool = ALL.values() if strategies is None else strategies
    return sum(s.combo_count() for s in pool)


def load_all() -> dict[str, Strategy]:
    """Import every module in ``strategies/`` so their decorators run.

    Import errors are FATAL rather than skipped: a family silently missing from
    the leaderboard would read as "tested and found wanting".
    """
    import importlib
    import pkgutil
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent / "strategies"
    if not pkg_dir.is_dir():
        return ALL
    for mod in pkgutil.iter_modules([str(pkg_dir)]):
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"strategies.{mod.name}")
    return ALL
