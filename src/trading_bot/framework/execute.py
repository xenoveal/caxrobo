"""
run_graph_backtest: the ONE seam that turns a StrategyGraph into engine.Trade
objects (v0.3.0 shared architecture contract §5).

This module is a faithful re-expression of backtest/engine.py:225-546. Where the
two disagree, engine.py is right and this file is broken.
tests/test_framework_parity.py is the proof, and it is the acceptance gate for
PRD Phase 3. Do NOT import run_backtest here: if this function can ever delegate
to it, the parity test is vacuous.

Stated assumptions, reproduced so they survive without the plan:

A3 — THE BREAKOUT TRIGGER IS EXECUTOR MACHINERY, parameterized by a graph-level
TriggerSpec, not a plug-in kind. signals.breakout.check_breakout is the SHARED
live/backtest crossing rule and it is where two no-lookahead guarantees live: the
`ts < candidate.end_ts` skip (breakout.py:113-114) and the interval_ms contiguity
check (breakout.py:115-122). Making it swappable per branch would let a mutated
graph disable the freshness test or the gap check and produce a flattering
backtest that still validates.

A6 — signals/scan.py IS NOT MIGRATED TO GRAPH DISPATCH IN THIS PHASE. The live
path returns list[Signal] for alerting and has no EvalSession, no
start_ms/end_ms and no Trade; this phase's gate is BACKTEST parity, and rewiring
live dispatch before a single graph has passed that gate would put an unvalidated
code path in front of the only thing the operator acts on. THE DRIFT RISK IS REAL
AND IS NAMED: there are now two dispatch tables — scan.py:55-61 (regime ->
scanner) and Branch.regimes — and they can silently disagree. Three mitigations
ship in this phase: (i) config.FADE_ENABLED is read AT CALL TIME inside the fade
detector plug-in (A7), so the kill switch cannot desync; (ii)
test_framework_parity.py::TestScanDriftGuard asserts that for every label in
classifier.REGIMES the v0.2.0 graph's active branch corresponds to what
scan.scan_symbol dispatches; (iii) full-history parity is itself a drift
detector, since both tables feed the same detectors. Migration is RECOMMENDED
FOR PHASE 4, which already owns graph-backtest and the thin slice.

A9 — THE EXECUTOR'S INTERNAL CURRENCY IS signals.setup.Signal, and the
multi-candidate tie-break calls signals.setup.rank_signals UNCHANGED. A
PositionPolicy returns a PositionPlan per contract §3; the executor adapts it to
a Signal via contracts.plan_to_signal BEFORE ranking, so ties break by the
identical (-rr, pattern, direction) key the engine uses (setup.py:227) and Trade
construction is field-for-field the same code shape as engine.close_out.
PositionPlan.source carries the detector event's kind, so Trade.pattern — and
therefore metrics.by_bucket's "regime/pattern" keys (metrics.py:31) — are
unchanged.

A10 — THE COST ARITHMETIC IN _close_out IS A DELIBERATE DUPLICATION of
engine.run_backtest.close_out (engine.py:354-378), pinned by a test, not a
refactor. Contract §1 says do not rewrite, wrap, or fork engine.py; extracting a
shared helper IS an edit to it, and the test baseline plus _CACHE semantics make
even a pure extraction a nonzero risk for zero benefit at this stage. WHEN ONE
CHANGES, BOTH MUST, and test_framework_parity.py::TestCostArithmetic will say so.

A11 — _close_out IS THE DESIGNATED EXTENSION POINT for Phase 4's three appended
Trade fields, and is shaped for that up front. See its docstring.
"""

import logging

import numpy as np

from trading_bot import config
from trading_bot.backtest.engine import Trade
from trading_bot.framework import contracts, registry
from trading_bot.framework.context import EvalSession
from trading_bot.framework.context import clear_caches as _clear_context_caches
from trading_bot.framework.errors import ContractError
from trading_bot.framework.graph import ANY_REGIME, StrategyGraph, branch_hash, validate
from trading_bot.risk.atr_stop import net_rr
from trading_bot.signals.breakout import check_breakout
from trading_bot.signals.setup import rank_signals

logger = logging.getLogger("trading_bot")


def clear_caches() -> None:
    """Drop the framework's memos, so a test needs one call rather than two."""
    _clear_context_caches()


def run_graph_backtest(
    conn,
    graph: StrategyGraph,
    symbol: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    fee_pct: float | None = None,
    slippage_pct: float | None = None,
    funding_pct_per_day: float | None = None,
    max_hold_bars: int | None = None,
) -> list[Trade]:
    """
    Replay a serialized strategy graph over stored history.

    Signature mirrors engine.run_backtest argument-for-argument minus `params`,
    which the graph carries, so the two are drop-in interchangeable
    (contract §5).

    Args:
        conn: Database connection.
        graph: The strategy to replay. Validated before anything is loaded.
        symbol: Trading pair symbol.
        start_ms: Only trigger bars closing at/after this time can trigger
            entries (default: start of data).
        end_ms: Only trigger bars closing at/before this time are simulated
            (default: end of data).
        fee_pct / slippage_pct: Per-side costs (default config values).
        funding_pct_per_day: Funding cost per day held (default
            config.FUNDING_PCT_PER_DAY).
        max_hold_bars: Time-stop in trigger bars, used by any branch whose
            ExitPolicySpec leaves max_hold_bars None (default
            config.MAX_HOLD_BARS_TRIGGER).

    Returns:
        List of Trade in entry-time order — the SAME dataclass
        engine.run_backtest returns, so metrics, equity, walk-forward, the
        benchmark, reviewers and the UI all work unchanged.

    Raises:
        GraphError: If the graph is malformed (validated first).
        ContractError: If the data plug-in does not satisfy the DataSource
            protocol.
        ValueError: From engine._assert_interval, when stored bar spacing
            disagrees with the configured tier.
    """
    validate(graph)

    fee = config.FEE_PCT if fee_pct is None else fee_pct
    slip = config.SLIPPAGE_PCT if slippage_pct is None else slippage_pct
    funding = config.FUNDING_PCT_PER_DAY if funding_pct_per_day is None else funding_pct_per_day
    max_hold = config.MAX_HOLD_BARS_TRIGGER if max_hold_bars is None else max_hold_bars
    cost = 2 * (fee + slip)

    regime_tf = graph.regime.timeframe
    setup_tf = config.SIGNAL_PATTERN_TIMEFRAME
    trigger_tf = config.SIGNAL_TRIGGER_TIMEFRAME

    data_spec = registry.get(graph.data.key)
    source = data_spec.impl(conn, **data_spec.resolve(graph.data.params))
    if not isinstance(source, contracts.DataSource):
        raise ContractError(
            f"{graph.data.key} returned {type(source).__name__}, which does not "
            f"satisfy the DataSource protocol (needs frame() and timeframes())"
        )

    session = EvalSession(
        source, symbol, tiers=(regime_tf, setup_tf, trigger_tf), regime_gate=graph.regime
    )
    if session.empty:
        return []

    df_trig = session.frame_of(trigger_tf)
    ts_trig = df_trig.index.to_numpy()
    close_trig = session.closes(trigger_tf)
    close_setup = session.closes(setup_tf)
    trigger_ms = session.interval_ms(trigger_tf)

    start = int(close_trig[0]) if start_ms is None else start_ms
    end = int(close_trig[-1]) if end_ms is None else end_ms

    highs = df_trig["high"].to_numpy()
    lows = df_trig["low"].to_numpy()
    closes = df_trig["close"].to_numpy()

    # Resolve every branch ONCE. Iterate ordered_branches(), never
    # graph.branches (A5): canonical hash order is also runtime order, so two
    # hash-equal graphs cannot break a rank_signals tie differently.
    branches = [b for b in graph.ordered_branches() if b.enabled]
    det_specs, det_params, pol_specs, pol_params = {}, {}, {}, {}
    conf_specs, conf_params, cand_keys, atr_periods = {}, {}, {}, {}
    for b in branches:
        det_specs[b.id] = registry.get(b.detector.key)
        det_params[b.id] = det_specs[b.id].resolve(b.detector.params)
        pol_specs[b.id] = registry.get(b.policy.key)
        pol_params[b.id] = pol_specs[b.id].resolve(b.policy.params)
        conf_specs[b.id] = [registry.get(c.key) for c in b.confirmations]
        conf_params[b.id] = [
            registry.get(c.key).resolve(c.params) for c in b.confirmations
        ]
        cand_keys[b.id] = branch_hash(graph, b)
        # The ATR the policy's stop is derived from AND the trail ratchets on —
        # engine.py:537 uses ONE atr_value for both, and graph validation rule 12
        # keeps the two declarations equal. A policy that declares no atr_period
        # (the fade policy: its stop is structural) still gets an atr recorded on
        # the open trade, exactly as engine.py:537 does for fade trades.
        atr_periods[b.id] = pol_params[b.id].get(
            "atr_period", b.exits.trail_atr_period
        )
    filter_specs = [registry.get(f.key) for f in graph.filters]
    filter_params = [registry.get(f.key).resolve(f.params) for f in graph.filters]

    trades: list[Trade] = []
    open_trade: dict | None = None

    def _close_out(rec: dict, j: int, price: float, outcome: str) -> None:
        """Append one Trade.

        Deliberate duplication of engine.run_backtest's close_out
        (engine.py:354-378) — see A10. When one changes, both must, and
        tests/test_framework_parity.py::TestCostArithmetic will say so.

        PHASE 4 EXTENSION POINT (contract §5), NOW TAKEN. planned_rr and
        confirmations are populated here, on the graph path only —
        engine.run_backtest never will, which is why
        tests/test_framework_parity.py compares the simulation fields explicitly
        instead of whole dataclasses.

        planned_rr is the NET, cost-adjusted ratio (risk.atr_stop.net_rr),
        computed from the plan and THIS RUN'S resolved fee/slippage. Contract §5
        defines the field as "R:R after costs at entry", and A11's original
        sketch of `rec["plan"].rr` would have recorded the GROSS ratio, which is
        the one number the field exists to not be. Computing it here rather than
        lifting it out of filter.rr-after-costs' verdict means every graph carries
        the audit trail, including graphs with no R:R filter at all — a legacy
        parity graph would otherwise record a meaningless 0.0. The cost term comes
        from risk.atr_stop, the single frozen definition (contract §1), never
        recomputed locally. When the filter IS present at default costs the two
        numbers are identical, which tests/test_pipeline_thin_slice.py pins.

        strategy_version is deliberately left at its "" default: contract §5
        assigns it to Phase 5's version registry, so Phase 5 has exactly one
        writer and no ambiguity about who owns the value.

        pattern_start_ts/pattern_end_ts/pattern_level/pattern_meta (v0.3.2 WS-B,
        D4) are the winning branch's ORIGINAL DetectedEvent — not `stamped`
        (the with_trigger copy carried on rec["event"] would also work for
        start_ts/end_ts/level, since with_trigger only adds to meta, but using
        the pre-stamp event keeps pattern_meta exactly what the detector
        emitted, free of the trigger_ts/trigger_price/... keys that describe
        the EXECUTION rather than the SHAPE). Carried verbatim, never
        re-derived: the whole point of persisting this is that the UI draws
        what was actually detected, not a plausible-looking approximation.

        Trade is constructed with KEYWORD ARGUMENTS ONLY, in field order — do not
        collapse it to positional.

        Note `s.stop` is the INITIAL stop even when the trail moved it:
        Trade.stop is "a frozen record of the setup" (engine.py:114-117), and
        rec["stop"] is the live one. Getting this backwards changes Trade.stop on
        every trailed trade.
        """
        nonlocal open_trade
        s = rec["signal"]
        event = rec["event"]
        sign = 1.0 if s.direction == "long" else -1.0
        gross = sign * (price - s.entry) / s.entry
        hold_days = (int(ts_trig[j]) - s.ts) / 86_400_000.0
        funding_cost = funding * hold_days
        trades.append(
            Trade(
                symbol=symbol,
                regime=rec["regime"],
                pattern=s.pattern,
                direction=s.direction,
                entry_ts=s.ts,
                entry=s.entry,
                stop=s.stop,
                target=s.target,
                exit_ts=int(ts_trig[j]),
                exit_price=price,
                outcome=outcome,
                pnl_pct=gross - cost - funding_cost,
                volume_high=s.volume_high,
                planned_rr=net_rr(
                    rec["plan"].reward_pct, rec["plan"].risk_pct, fee, slip
                ),
                confirmations=tuple(sorted(rec["confirmed"])),
                # strategy_version left at "" — Phase 5 owns it (contract §5).
                pattern_start_ts=event.start_ts,
                pattern_end_ts=event.end_ts,
                pattern_level=event.level,
                pattern_meta=tuple(sorted(event.meta.items())),
            )
        )
        open_trade = None

    def _detect(branch, h_idx: int, reg: str, setup_ctx) -> list:
        """One branch's events at a setup bar, memoized via session.candidates.

        Applies the regime gate INSIDE the memo, exactly as engine.py:341-347
        gates on `reg == "trending"` inside candidates_for — so an inactive
        branch's empty list is cached too, which is where the engine's
        performance actually comes from.
        """

        def produce() -> list:
            if not (ANY_REGIME in branch.regimes or reg in branch.regimes):
                return []
            events = det_specs[branch.id].impl(setup_ctx, **det_params[branch.id])
            return list(events)

        return session.candidates(cand_keys[branch.id], h_idx, produce)

    def _plan(branch, event, trig, ctx):
        """Confirmations -> policy -> filters -> Signal.

        Returns (plan, signal, confirmed_names) or None at the FIRST rejection.
        A rejection is not an error and must not raise.

        Order is fixed: confirmations run on the DetectedEvent AFTER the trigger
        fires and after the trigger facts are stamped into meta; filters run on
        the PositionPlan, BEFORE plan_to_signal.
        """
        stamped = contracts.with_trigger(event, trig)
        confirmed: list[str] = []
        for spec, params in zip(conf_specs[branch.id], conf_params[branch.id]):
            verdict = spec.impl(ctx, stamped, **params)
            if not verdict.passed:
                return None
            confirmed.append(verdict.name or spec.name)

        plan = pol_specs[branch.id].impl(ctx, stamped, **pol_params[branch.id])
        if plan is None:
            return None

        for spec, params in zip(filter_specs, filter_params):
            verdict = spec.impl(ctx, plan, **params)
            if not verdict.accepted:
                return None

        return plan, contracts.plan_to_signal(plan, trig), tuple(confirmed)

    last_j = -1
    for j in range(len(ts_trig)):
        bc = int(close_trig[j])
        if bc > end:
            break
        last_j = j

        if open_trade is not None:
            if j <= open_trade["entry_j"]:
                continue
            s = open_trade["signal"]
            ex = open_trade["exits"]
            stop = open_trade["stop"]

            # ONE generic exit block reproduces BOTH legacy branches from
            # ExitPolicySpec flags, with no `if is_donchian` (A4):
            #   fade branch (channel_exit=False, target_enabled=True,
            #     trail_enabled=False): chan is NaN so the channel arm is
            #     skipped, the target arm runs unconditionally, and `trailed`
            #     stays False forever so the label is "stop" and never "trail"
            #     -> identical to engine.py:398-408.
            #   Donchian branch (channel_exit=True, target/trail from the graph)
            #     -> identical to engine.py:417-432.
            # Priority is stop/trail -> channel -> target, the conservative
            # same-bar rule (engine.py:36-37). Reordering this elif chain is what
            # TestExitMix's outcome counts detect.
            if ex.channel_exit:
                exit_lower, exit_upper = session.channels(setup_tf, ex.channel_period)
                # Setup-tier bar closed by this trigger bar: the opposite-channel
                # level is a trailing value, same lookup rule as the regime label.
                s_idx = (
                    int(np.searchsorted(close_setup, int(close_trig[j]), side="right")) - 1
                )
            else:
                exit_lower = exit_upper = None
                s_idx = -1

            branch_max_hold = max_hold if ex.max_hold_bars is None else ex.max_hold_bars

            if s.direction == "long":
                chan = (
                    float(exit_lower[s_idx])
                    if (ex.channel_exit and s_idx >= 0)
                    else float("nan")
                )
                if lows[j] <= stop:
                    _close_out(open_trade, j, stop, "trail" if open_trade["trailed"] else "stop")
                elif not np.isnan(chan) and lows[j] <= chan:
                    _close_out(open_trade, j, chan, "channel")
                elif ex.target_enabled and highs[j] >= s.target:
                    _close_out(open_trade, j, s.target, "target")
            else:
                chan = (
                    float(exit_upper[s_idx])
                    if (ex.channel_exit and s_idx >= 0)
                    else float("nan")
                )
                if highs[j] >= stop:
                    _close_out(open_trade, j, stop, "trail" if open_trade["trailed"] else "stop")
                elif not np.isnan(chan) and highs[j] >= chan:
                    _close_out(open_trade, j, chan, "channel")
                elif ex.target_enabled and lows[j] <= s.target:
                    _close_out(open_trade, j, s.target, "target")

            if open_trade is not None and j - open_trade["entry_j"] >= branch_max_hold:
                _close_out(open_trade, j, float(closes[j]), "time")
            # Ratchet AFTER this bar's exits are resolved: a stop derived from
            # bar j's own extreme, tested against bar j's own low, is intra-bar
            # lookahead. The trail only ever binds from bar j+1 onward.
            if open_trade is not None and ex.trail_enabled and open_trade["atr"] > 0:
                trail_dist = ex.trail_atr_multiple * open_trade["atr"]
                if s.direction == "long":
                    open_trade["extreme"] = max(open_trade["extreme"], float(highs[j]))
                    new_stop = open_trade["extreme"] - trail_dist
                    if new_stop > open_trade["stop"]:
                        open_trade["stop"] = new_stop
                        open_trade["trailed"] = True
                else:
                    open_trade["extreme"] = min(open_trade["extreme"], float(lows[j]))
                    new_stop = open_trade["extreme"] + trail_dist
                    if new_stop < open_trade["stop"]:
                        open_trade["stop"] = new_stop
                        open_trade["trailed"] = True
            continue

        if bc < start or j < 1:
            continue

        # Setup bar that had already CLOSED when this trigger bar OPENED.
        #
        # Selecting by the trigger bar's close (`bc`) instead silently discarded
        # one trigger bar in every setup window: on the last trigger bar of a
        # window, `bc` lands on the NEXT setup bar's close, so h_idx advanced to
        # a setup bar whose end_ts postdates this bar's open, and
        # check_breakout's `ts < candidate.end_ts` guard rejected it. With a 4H
        # setup and 1H trigger that was 25% of all trigger opportunities.
        # See .claude/PRPs/reports/code review/phase4-7-code-review.md (MEDIUM-2).
        #
        # MEASURED CONSEQUENCE, recorded because it is counter-intuitive and must
        # not be silently re-broken: recovering those bars ADDS trades that were
        # unprofitable in-sample (142 -> 158 trades, pooled Sharpe 0.431 ->
        # 0.255, annualised 11.0% -> 0.25% on stored history). The fix was kept
        # because an entry rule must be an explicit, pre-registered decision, not
        # an artifact of two code paths disagreeing about which setup bar is
        # current. DO NOT SILENTLY RE-BREAK IT — TestMedium2 pins it.
        h_idx = int(np.searchsorted(close_setup, int(ts_trig[j]), side="right")) - 1
        if h_idx < 0:
            continue

        setup_ctx = session.context("setup", int(close_setup[h_idx]))
        reg = setup_ctx.regime()
        trig_ctx = None

        # Same slice shape production uses: volume window + crossing pair.
        window_trig = df_trig.iloc[
            max(0, j - (graph.trigger.volume_lookback + 1)) : j + 1
        ]

        bar_signals: list = []
        for branch in branches:
            events = _detect(branch, h_idx, reg, setup_ctx)
            if not events:
                continue
            atr_arr = session.atr(setup_tf, atr_periods[branch.id])
            atr_value = float(atr_arr[h_idx]) if h_idx < len(atr_arr) else float("nan")
            # engine.py:499-503 skips build_signal entirely when atr_value <= 0
            # for the ATR-stop method, and does NOT for the structural-stop fade
            # method. The generic expression of that: a policy that declares an
            # atr_period needs a defined ATR. `not (atr > 0)` catches NaN
            # (Wilder warmup), zero and negative — setup.py:131-136's idiom.
            needs_atr = "atr_period" in pol_params[branch.id]
            if needs_atr and not (atr_value > 0):
                continue
            for event in events:
                trig = check_breakout(
                    window_trig,
                    contracts.candidate_from_event(event),
                    volume_lookback=graph.trigger.volume_lookback,
                    volume_high_ratio=graph.trigger.volume_high_ratio,
                    lookback_bars=graph.trigger.lookback_bars,
                    interval_ms=trigger_ms,
                )
                if trig is None:
                    continue
                if trig_ctx is None:
                    trig_ctx = session.context("trigger", int(ts_trig[j]))
                planned = _plan(branch, event, trig, trig_ctx)
                if planned is None:
                    continue
                plan, sig, confirmed = planned
                # `event` (the pre-trigger-stamp DetectedEvent) rides along so
                # _close_out can record the winning candidate's real geometry
                # (pattern_start_ts/end_ts/level/meta, v0.3.2 WS-B) on the Trade
                # it eventually builds — see A9 for why `sig` alone is not enough.
                bar_signals.append((branch, plan, sig, confirmed, atr_value, event))

        # Only one trade at a time, so pick by the SAME rule live scanning ranks
        # by (setup.rank_signals) — taking the first candidate instead would make
        # measured performance a function of branch iteration order.
        if bar_signals:
            winner = rank_signals([t[2] for t in bar_signals])[0]
            # `is` identity, not `==`: two branches can legitimately produce
            # equal-valued Signals, and `==` would pick the wrong branch's
            # ExitPolicySpec.
            branch, plan, sig, confirmed, atr_value, event = next(
                t for t in bar_signals if t[2] is winner
            )
            open_trade = {
                "signal": sig,        # Trade is built from this
                "plan": plan,         # carries .rr for Phase 4's planned_rr
                "confirmed": confirmed,  # for Phase 4's Trade.confirmations
                "event": event,        # for v0.3.2 WS-B's Trade.pattern_* fields
                "branch": branch,     # the winning Branch
                "exits": branch.exits,  # drives the exit loop
                "entry_j": j,
                "regime": reg,
                "stop": sig.stop,     # MUTABLE when exits.trail_enabled
                "atr": atr_value,     # setup-tier ATR at entry, frozen
                "extreme": sig.entry,  # best price seen since entry
                "trailed": False,
                "h_idx": h_idx,
            }

    if open_trade is not None and last_j > open_trade["entry_j"]:
        _close_out(open_trade, last_j, float(closes[last_j]), "end")

    return trades
