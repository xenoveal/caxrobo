"""
Framework plug-ins: one module per plug-in family.

framework.registry.load_all() walks this package with pkgutil and IMPORT ERRORS
ARE FATAL, never skipped — a family silently missing from a report reads as
"tested and found wanting" (scripts/bruteforce/registry.py:162-167).

Modules whose last name segment starts with "_" are skipped by load_all, which
is how a scratch or helper module stays out of the registry.

This package's __init__ deliberately does NOT import its sibling plug-in
modules: load_all() does that, and an eager import here would make one broken
plug-in break build_v020_graph too. build_v020_graph only names registry keys as
strings.
"""

from trading_bot import config
from trading_bot.framework.graph import (
    Branch,
    ExitPolicySpec,
    NodeSpec,
    RegimeGate,
    StrategyGraph,
    TriggerSpec,
)


def build_v020_graph(
    *,
    name: str = "donchian-v020",
    trail_enabled: bool = config.TRAIL_ENABLED,
    trail_atr_multiple: float = config.TRAIL_ATR_MULTIPLE,
    target_enabled: bool = config.DONCHIAN_TARGET_ENABLED,
    rr_floor: float = config.RR_FLOOR,
    adx_trend_threshold: float = config.ADX_TREND_THRESHOLD,
    atr_extreme_percentile: float = config.ATR_EXTREME_PERCENTILE,
    bb_num_std: float = config.BB_STD,
    include_fade: bool = True,
) -> StrategyGraph:
    """The v0.2.0 strategy expressed as a graph.

    This is the parity fixture, the `cli.py graph-validate` example, and Phase 6's
    seed genome.

    Every default is read from config, so this function TRACKS v0.2.0 rather than
    freezing a copy of it: if TRAIL_ENABLED ever flips, the parity fixture follows
    and the parity test keeps testing the CURRENT engine defaults instead of a
    historical snapshot. The keyword arguments exist so the parity test can sweep
    the same axes walkforward.DEFAULT_GRID sweeps and prove parity for all of
    them, not just the defaults.

    Branch ids are "range" and "trend". Order is irrelevant to results (the two
    regime sets are disjoint, so at most one branch is active per setup bar and no
    rank_signals tie is possible) but it IS fixed by ordered_branches(), so the
    JSON fully determines behavior.

    The fade branch is included even though config.FADE_ENABLED is False, because
    A7 puts that check inside the detector at CALL TIME: including the branch is
    what proves the kill switch still suppresses identically through the graph
    path.

    Args:
        name: Graph name; becomes <name>.strategy.json.
        trail_enabled / trail_atr_multiple / target_enabled: The trend branch's
            exit modes — walkforward.DEFAULT_GRID's two boolean axes.
        rr_floor: Both policies' reward:risk floor. NOTE the two policies apply it
            to DIFFERENT ratios (gross for the ATR-stop path, cost-adjusted for
            the fade path); that asymmetry is v0.2.0's measured behavior and is
            deliberately preserved (A8).
        adx_trend_threshold / atr_extreme_percentile: RegimeGate thresholds.
        bb_num_std: The fade detector's band width.
        include_fade: Drop the fade branch entirely. Off only for tests that want
            a single-branch graph; the default INCLUDES it, because suppression
            through the kill switch is one of the things parity proves.

    Returns:
        A StrategyGraph. Call framework.graph.validate() on it (and
        registry.load_all() first) before use.
    """
    trend = Branch(
        id="trend",
        detector=NodeSpec(id="trend-detector", key="detector.donchian-breakout"),
        policy=NodeSpec(
            id="trend-policy",
            key="policy.atr-stop-measured-move",
            params={"rr_floor": rr_floor},
        ),
        regimes=("trending",),
        exits=ExitPolicySpec(
            stop=True,
            target_enabled=target_enabled,
            trail_enabled=trail_enabled,
            trail_atr_multiple=trail_atr_multiple,
            trail_atr_period=config.ATR_STOP_PERIOD,
            channel_exit=True,
            channel_period=config.DONCHIAN_ENTRY_PERIOD,
            max_hold_bars=None,  # => run_graph_backtest's value
        ),
    )
    fade = Branch(
        id="range",
        detector=NodeSpec(
            id="range-detector",
            key="detector.bollinger-fade",
            params={"num_std": bb_num_std},
        ),
        policy=NodeSpec(
            id="range-policy",
            key="policy.fade-structural-stop",
            params={"rr_floor": rr_floor},
        ),
        regimes=("ranging",),
        # The MANDATORY DEVIATION at engine.py:392-411 as data: fade trades keep
        # frozen stop/target/time/end behavior — no trail, no opposite channel.
        exits=ExitPolicySpec(
            stop=True,
            target_enabled=True,
            trail_enabled=False,
            channel_exit=False,
            max_hold_bars=None,
        ),
    )
    branches = (trend, fade) if include_fade else (trend,)

    return StrategyGraph(
        name=name,
        data=NodeSpec(id="ohlcv", key="data.ohlcv"),
        branches=branches,
        regime=RegimeGate(
            timeframe=config.REGIME_TIMEFRAME,
            adx_trend_threshold=adx_trend_threshold,
            atr_extreme_percentile=atr_extreme_percentile,
            enabled=True,
        ),
        trigger=TriggerSpec(),
        filters=(),  # v0.2.0 has NO Filter and NO Confirmation nodes — A8
        meta={
            "provenance": "v0.2.0 Phases 1-7, see .claude/PRPs/reports/KNOWN-LIMITATIONS.md",
            "gate": "FAILING — 2 of 5 conditions (KNOWN-LIMITATIONS §1)",
        },
    )
