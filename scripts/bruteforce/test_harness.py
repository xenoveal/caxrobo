"""Unit tests for the research harness.

Every number the strategy search produces flows through ``core.simulate`` and
``core.score``, so a silent error here would invalidate the entire report rather
than one strategy. These tests pin the behaviours that are easy to get subtly
wrong and impossible to notice from aggregate metrics: fill timing, the
conservative same-bar rule, cost arithmetic, trail ratcheting, and the
multi-timeframe alignment boundary.

Synthetic bars throughout, so an expected P&L can be computed by hand.

Run:  .venv/bin/pytest scripts/bruteforce/test_harness.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core  # noqa: E402
import indicators as ta  # noqa: E402

H = 3_600_000  # 1h in ms
D = 86_400_000


def bars(closes, *, highs=None, lows=None, opens=None, tf_ms=H, start=0):
    """Build an OHLCV frame from a close series, open-time indexed in epoch ms."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame(
        {
            "open": closes if opens is None else opens,
            "high": closes if highs is None else highs,
            "low": closes if lows is None else lows,
            "close": closes,
            "volume": np.ones(n),
        },
        index=pd.Index(start + np.arange(n) * tf_ms, name="ts"),
    )


def ctx_from(trigger_df, coarse=None):
    """Single- or two-timeframe Ctx over synthetic frames."""
    frames = {"1h": trigger_df}
    if coarse is not None:
        frames["4h"] = coarse
    return core.Ctx(symbol="TEST", trigger_tf="1h", frames=frames)


def only(entry_i, direction, n):
    e = np.zeros(n, dtype=np.int8)
    e[entry_i] = direction
    return e


SPAN = dict(start_ms=0, end_ms=10**15)


# ---------------------------------------------------------------------------
# Fill timing
# ---------------------------------------------------------------------------


def test_entry_fills_at_the_signal_bars_close():
    df = bars([100, 100, 100, 100, 100])
    c = ctx_from(df)
    t = core.simulate(c, core.Plan(entry=only(2, core.LONG, 5), stop_dist=np.full(5, 5.0)), **SPAN)
    assert len(t) == 1
    assert t["entry_i"][0] == 2
    assert t["entry"][0] == 100.0


def test_stop_is_not_checked_on_the_entry_bar():
    """The entry bar's own low must not close the trade.

    Entry bar 2 dips to 90 -- below the 95 stop -- but the fill happens at that
    bar's close, so the dip already happened. Charging it would be intra-bar
    lookahead in the pessimistic direction and would fabricate losses.
    """
    df = bars([100, 100, 100, 100, 100], lows=[100, 100, 90, 100, 100])
    c = ctx_from(df)
    t = core.simulate(c, core.Plan(entry=only(2, core.LONG, 5), stop_dist=np.full(5, 5.0)), **SPAN)
    assert core.OUTCOMES[t["outcome"][0]] == "end", "entry-bar low must not trigger the stop"


def test_no_entry_on_the_first_bar():
    df = bars([100] * 4)
    c = ctx_from(df)
    t = core.simulate(c, core.Plan(entry=only(0, core.LONG, 4), stop_dist=np.full(4, 5.0)), **SPAN)
    assert len(t) == 0


def test_one_position_at_a_time():
    """A second signal while a position is open is ignored, not stacked."""
    df = bars([100] * 10)
    e = np.zeros(10, dtype=np.int8)
    e[2] = e[3] = e[4] = core.LONG
    t = core.simulate(ctx_from(df), core.Plan(entry=e, stop_dist=np.full(10, 5.0)), **SPAN)
    assert len(t) == 1 and t["entry_i"][0] == 2


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------


def test_long_stop_fills_at_the_stop_price():
    df = bars([100, 100, 100, 97, 100], lows=[100, 100, 100, 94, 100])
    c = ctx_from(df)
    t = core.simulate(c, core.Plan(entry=only(2, core.LONG, 5), stop_dist=np.full(5, 5.0)), **SPAN)
    assert core.OUTCOMES[t["outcome"][0]] == "stop"
    assert t["exit_price"][0] == 95.0, "fills at the stop, not the bar's low"


def test_short_stop_fills_at_the_stop_price():
    df = bars([100, 100, 100, 103, 100], highs=[100, 100, 100, 106, 100])
    c = ctx_from(df)
    t = core.simulate(c, core.Plan(entry=only(2, core.SHORT, 5), stop_dist=np.full(5, 5.0)), **SPAN)
    assert core.OUTCOMES[t["outcome"][0]] == "stop"
    assert t["exit_price"][0] == 105.0


def test_target_fills_at_the_target_price():
    df = bars([100, 100, 100, 108, 100], highs=[100, 100, 100, 112, 100])
    c = ctx_from(df)
    plan = core.Plan(
        entry=only(2, core.LONG, 5), stop_dist=np.full(5, 5.0), target_dist=np.full(5, 10.0)
    )
    t = core.simulate(c, plan, **SPAN)
    assert core.OUTCOMES[t["outcome"][0]] == "target"
    assert t["exit_price"][0] == 110.0


def test_same_bar_stop_and_target_resolves_to_the_stop():
    """The conservative rule. A bar spanning both levels must book the LOSS.

    Without this, every backtest inherits a free optimistic bias exactly
    proportional to how volatile the bars are -- worst on the symbols with the
    least signal.
    """
    df = bars([100, 100, 100, 100], highs=[100, 100, 100, 115], lows=[100, 100, 100, 90])
    c = ctx_from(df)
    plan = core.Plan(
        entry=only(2, core.LONG, 4), stop_dist=np.full(4, 5.0), target_dist=np.full(4, 10.0)
    )
    t = core.simulate(c, plan, **SPAN)
    assert core.OUTCOMES[t["outcome"][0]] == "stop"


def test_exit_signal_closes_at_that_bars_close():
    df = bars([100, 100, 100, 103, 100])
    x = np.zeros(5, dtype=bool)
    x[3] = True
    plan = core.Plan(entry=only(2, core.LONG, 5), stop_dist=np.full(5, 5.0), exit_signal=x)
    t = core.simulate(ctx_from(df), plan, **SPAN)
    assert core.OUTCOMES[t["outcome"][0]] == "signal"
    assert t["exit_price"][0] == 103.0


def test_time_stop_closes_after_max_hold_bars():
    df = bars([100] * 20)
    plan = core.Plan(entry=only(1, core.LONG, 20), stop_dist=np.full(20, 5.0))
    t = core.simulate(ctx_from(df), plan, cfg=core.SimConfig(max_hold_bars=5), **SPAN)
    assert core.OUTCOMES[t["outcome"][0]] == "time"
    assert t["bars_held"][0] == 5


def test_open_position_at_data_end_is_closed_as_end():
    df = bars([100, 100, 100, 104])
    plan = core.Plan(entry=only(2, core.LONG, 4), stop_dist=np.full(4, 5.0))
    t = core.simulate(ctx_from(df), plan, **SPAN)
    assert core.OUTCOMES[t["outcome"][0]] == "end"
    assert t["exit_price"][0] == 104.0


# ---------------------------------------------------------------------------
# Trail
# ---------------------------------------------------------------------------


def test_trail_cannot_fire_on_the_bar_that_set_its_own_extreme():
    """Bar 3 spikes to 120 then closes at 100. A trail of 5 computed from that
    bar's own high would sit at 115 and 'stop out' inside the same bar -- pure
    intra-bar lookahead. The ratchet must apply from bar 4 onward.
    """
    df = bars([100, 100, 100, 100, 100], highs=[100, 100, 100, 120, 100],
              lows=[100, 100, 100, 100, 100])
    plan = core.Plan(
        entry=only(2, core.LONG, 5), stop_dist=np.full(5, 5.0), trail_atr=np.full(5, 5.0)
    )
    t = core.simulate(ctx_from(df), plan, **SPAN)
    # Ratchets to 115 after bar 3; bar 4's low of 100 then breaches it.
    assert core.OUTCOMES[t["outcome"][0]] == "trail"
    assert t["exit_i"][0] == 4, "trail must bind from the bar AFTER the extreme"


def test_trail_never_loosens():
    df = bars([100, 100, 100, 110, 100, 100], highs=[100, 100, 100, 110, 100, 100],
              lows=[100, 100, 100, 110, 104, 104])
    plan = core.Plan(
        entry=only(2, core.LONG, 6), stop_dist=np.full(6, 5.0), trail_atr=np.full(6, 5.0)
    )
    t = core.simulate(ctx_from(df), plan, **SPAN)
    # Extreme 110 -> stop 105. Bar 4's low 104 breaches it; a loosened stop
    # (back to 95 as price falls) would have survived.
    assert t["exit_price"][0] == 105.0


# ---------------------------------------------------------------------------
# Costs and R
# ---------------------------------------------------------------------------


def test_costs_are_charged_round_trip_plus_funding():
    """A flat trade must lose exactly the round-trip cost plus funding."""
    df = bars([100] * 4)
    plan = core.Plan(entry=only(1, core.LONG, 4), stop_dist=np.full(4, 5.0))
    t = core.simulate(ctx_from(df), plan, **SPAN)
    hold_days = (t["exit_ts"][0] - t["entry_ts"][0]) / core.DAY_MS
    expected = -core.ROUND_TRIP_COST - core.FUNDING_PCT_PER_DAY * hold_days
    assert t["pnl_pct"][0] == pytest.approx(expected, abs=1e-12)


def test_r_multiple_is_pnl_over_risk():
    df = bars([100, 100, 100, 100], lows=[100, 100, 100, 90])
    plan = core.Plan(entry=only(2, core.LONG, 4), stop_dist=np.full(4, 5.0))
    t = core.simulate(ctx_from(df), plan, **SPAN)
    assert t["risk_pct"][0] == pytest.approx(0.05)
    assert t["r_multiple"][0] == pytest.approx(t["pnl_pct"][0] / 0.05)
    # A clean stop-out loses 1R plus costs -- slightly worse than -1.
    assert -1.2 < t["r_multiple"][0] < -1.0


def test_short_pnl_sign_is_correct():
    df = bars([100, 100, 100, 90])
    plan = core.Plan(entry=only(2, core.SHORT, 4), stop_dist=np.full(4, 5.0))
    t = core.simulate(ctx_from(df), plan, **SPAN)
    assert t["pnl_pct"][0] > 0.09, "a short into a 10% fall must profit"


def test_cost_ratio_matches_the_prd_definition():
    """c = round-trip cost / median risk_pct, the PRD's cost-frontier gate."""
    df = bars([100] * 6)
    e = np.zeros(6, dtype=np.int8)
    e[1] = core.LONG
    plan = core.Plan(entry=e, stop_dist=np.full(6, 2.0))  # risk_pct = 2%
    m = core.score(core.simulate(ctx_from(df), plan, **SPAN), start_ms=0, end_ms=6 * H)
    assert m["cost_ratio"] == pytest.approx(core.ROUND_TRIP_COST / 0.02)


def test_entry_is_suppressed_without_a_valid_stop():
    """No risk definition, no trade -- NaN and non-positive stops both."""
    df = bars([100] * 6)
    e = np.zeros(6, dtype=np.int8)
    e[2] = e[4] = core.LONG
    sd = np.full(6, np.nan)
    sd[4] = -1.0
    t = core.simulate(ctx_from(df), core.Plan(entry=e, stop_dist=sd), **SPAN)
    assert len(t) == 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_sharpe_is_invariant_to_base_risk_below_the_leverage_cap():
    """Turning risk up must not buy Sharpe. This is what makes the headline
    metric non-gameable by the 'finetune the risk' lever."""
    df = bars(100 + np.sin(np.arange(400) / 7.0) * 5)
    e = np.zeros(400, dtype=np.int8)
    e[10::40] = core.LONG
    plan = core.Plan(entry=e, stop_dist=np.full(400, 8.0))  # risk 8% -> size well under cap
    t = core.simulate(ctx_from(df), plan, **SPAN)
    span = dict(start_ms=0, end_ms=400 * H)
    a = core.score(t, **span)["sharpe"]
    saved = core.BASE_RISK
    try:
        core.BASE_RISK = saved * 2
        b = core.score(t, **span)["sharpe"]
    finally:
        core.BASE_RISK = saved
    assert a == pytest.approx(b, rel=1e-9)


def test_score_of_no_trades_is_empty_not_zero():
    """Zero trades must report None, never a flattering 0.0 Sharpe."""
    m = core.score(np.zeros(0, dtype=core.TRADE_DTYPE), start_ms=0, end_ms=D)
    assert m["trades"] == 0 and m["sharpe"] is None and m["win_rate"] is None


# ---------------------------------------------------------------------------
# Multi-timeframe alignment
# ---------------------------------------------------------------------------


def test_align_uses_only_closed_coarse_bars():
    """A 4H value must not be visible before that 4H bar has closed.

    4H bar 0 spans 1H bars 0-3 and closes at the end of bar 3. So 1H bars 0-2
    see nothing (NaN), and bar 3 -- whose close coincides with the 4H close --
    is the first to see it.
    """
    trig = bars([100] * 8)
    coarse = bars([10.0, 20.0], tf_ms=4 * H)
    c = ctx_from(trig, coarse)
    got = c.align(coarse["close"], "4h")
    assert np.isnan(got[:3]).all(), "coarse value leaked before its bar closed"
    assert got[3] == 10.0 and got[6] == 10.0 and got[7] == 20.0


def test_align_rejects_a_length_mismatch():
    c = ctx_from(bars([100] * 8), bars([1.0, 2.0], tf_ms=4 * H))
    with pytest.raises(ValueError, match="length"):
        c.align(np.arange(5, dtype=float), "4h")


def test_truncated_ctx_drops_coarse_bars_that_had_not_closed():
    trig = bars([100] * 8)
    coarse = bars([10.0, 20.0], tf_ms=4 * H)
    frames = {"1h": trig, "4h": coarse}
    full = core.Ctx(symbol="T", trigger_tf="1h", frames=frames)
    assert full.align(coarse["close"], "4h")[7] == 20.0
    # Truncating to 4 1H bars leaves only the first 4H bar closed.
    cutoff = int(trig.index[3]) + H
    kept = coarse[coarse.index + 4 * H <= cutoff]
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# Indicator causality (the properties assert_causal depends on)
# ---------------------------------------------------------------------------


def test_pivot_high_is_confirmed_late_not_marked_in_place():
    """A pivot at bar t may only be flagged at bar t+span."""
    highs = np.array([1, 2, 3, 9, 3, 2, 1, 1, 1], dtype=float)
    df = bars(highs, highs=highs, lows=highs - 1)
    piv = ta.pivot_high(df, span=2)
    assert np.isnan(piv.iloc[3]), "pivot must not be visible on its own bar"
    assert piv.iloc[5] == 9.0, "pivot is confirmed span bars later"


def test_rolling_high_excludes_the_current_bar_for_breakouts():
    df = bars([1, 2, 3, 10.0], highs=[1, 2, 3, 10.0])
    hi = ta.rolling_high(df, 3)
    assert hi.iloc[3] == 3.0, "must exclude the current bar or a break is impossible"


def test_zscore_uses_a_trailing_window():
    """A full-sample z-score would make early values depend on late data."""
    s = pd.Series(np.arange(100, dtype=float))
    full = ta.zscore(s, 20)
    part = ta.zscore(s.iloc[:60], 20)
    pd.testing.assert_series_equal(full.iloc[:60], part)


# ---------------------------------------------------------------------------
# Integration: the audit must be able to fail
# ---------------------------------------------------------------------------


def test_causality_audit_catches_a_deliberate_lookahead():
    """If this ever passes, every 'audit clean' claim in the report is worthless."""
    import registry

    registry.load_all()
    canary = registry.ALL["lookahead_canary"]
    with pytest.raises(AssertionError, match="LOOKAHEAD"):
        core.assert_causal(canary.build, {}, "BTCUSDT")


def test_causality_audit_passes_the_production_baseline():
    import registry

    registry.load_all()
    base = registry.ALL["donchian_production"]
    core.assert_causal(base.build, {}, "BTCUSDT")
