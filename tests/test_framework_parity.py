"""THE PHASE 3 ACCEPTANCE GATE: run_graph_backtest reproduces run_backtest exactly.

Parity is the proof that the framework WRAPS v0.2.0's honest core rather than
replacing it (contract §5). Phase 4 onward may only extend from a green parity
test. When parity is red, fix the EXECUTOR to match the engine — never the engine
to match the executor, and never the test to match both.

Beyond the aggregate comparison, nine localizing sub-parity classes each isolate
one row of the plan's parity table, because a bare failure on 150 trades is
nearly uninformative whereas "regime labels match, candidate lists match, but the
exit outcome mix differs" points at one block.
"""

import collections
import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot import config
from trading_bot.backtest import engine, walkforward
from trading_bot.backtest.engine import BacktestParams, run_backtest
from trading_bot.data import storage
from trading_bot.framework import context as fcontext
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.execute import run_graph_backtest
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.plugins import build_v020_graph
from trading_bot.regime.classifier import REGIMES
from trading_bot.signals import scan
from trading_bot.signals.donchian import DONCHIAN_KIND, detect_donchian_setups
from trading_bot.signals.meanrev import FADE_KIND, detect_fade_setups
from trading_bot.signals.patterns import PatternCandidate

SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded (contract §8).
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_REG = storage.TIMEFRAME_MS[REGIME_TF]
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000

# The fields that constitute a simulated round-trip. Deliberately EXPLICIT and
# deliberately NOT `Trade == Trade`:
#
# contract §5 has Phase 4 append planned_rr / confirmations / strategy_version
# to Trade and populate them on the GRAPH path only — engine.run_backtest never
# will. Dataclass equality would therefore start failing the moment Phase 4
# lands, and would read as a graph-executor regression rather than as the
# intended divergence it is. Parity is about the SIMULATION, not about the
# audit trail attached to it.
#
# Do not replace this with `==`. If a new field belongs in parity, add its name
# here explicitly, with a reason.
PARITY_EXACT = (
    "symbol", "regime", "pattern", "direction",
    "entry_ts", "exit_ts", "outcome", "volume_high",
)
PARITY_CLOSE = ("entry", "stop", "target", "exit_price", "pnl_pct")

# Bit-identity is the EXPECTED outcome: the executor performs the same
# arithmetic in the same order on the same floats. abs_tol exists so a genuine
# reassociation reports as a tiny delta instead of an opaque False, and it is
# tight enough that a real behavioural difference (a different exit level, a
# different cost term) can never hide under it. If a diff shows up at 1e-13,
# something reordered — investigate, do not widen the tolerance.
PARITY_ABS_TOL = 1e-12


def assert_trades_match(got, want, *, abs_tol=PARITY_ABS_TOL):
    """Compare a graph trade list against an engine trade list, field by field."""
    assert len(got) == len(want), (
        f"trade COUNT differs: graph {len(got)} vs engine {len(want)}; "
        f"graph entries {[t.entry_ts for t in got][:5]} ... "
        f"engine entries {[t.entry_ts for t in want][:5]}"
    )
    for i, (g, e) in enumerate(zip(got, want)):
        for f in PARITY_EXACT:
            assert getattr(g, f) == getattr(e, f), (
                f"trade {i} field {f}: {getattr(g, f)!r} != {getattr(e, f)!r}"
            )
        for f in PARITY_CLOSE:
            gv, ev = getattr(g, f), getattr(e, f)
            assert math.isclose(gv, ev, rel_tol=0.0, abs_tol=abs_tol), (
                f"trade {i} field {f}: {gv!r} vs {ev!r} (delta {gv - ev:.3e})"
            )


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Clear BOTH memos around every test, and make sure the plug-ins are loaded."""
    registry.load_all()
    engine.clear_caches()
    fcontext.clear_caches()
    yield
    engine.clear_caches()
    fcontext.clear_caches()


# --------------------------------------------------------------------------- #
# Fixture builders, COPIED from tests/test_backtest.py:83-215. The suite has no
# conftest.py and duplicates fixtures across modules by convention; follow it.
# --------------------------------------------------------------------------- #


def seed(conn, timeframe, rows, start=START, interval=D_SET):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    storage.upsert_candles(conn, SYMBOL, timeframe, data)
    return data


def donchian_rows():
    """Setup-timeframe ramp: 60 bars, ADX(14) >> 25 after warmup, last bar closes
    exactly at the trailing 20-bar upper channel (160.0) and above the 55-bar mid
    (131.5). Constant true range (3) gives an exact ATR(14) = 3.0."""
    return [[100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 10.0] for i in range(60)]


def donchian_atr_at_entry():
    """The ATR value both paths must compute at the Donchian breakout's h_idx."""
    df_setup = pd.DataFrame(
        donchian_rows(), columns=["open", "high", "low", "close", "volume"]
    )
    return float(wilder_atr(df_setup, period=config.ATR_STOP_PERIOD).iloc[-1])


def seed_scenario(conn, outcome_rows_trig):
    """Donchian ramp + a trigger breakout above the 20-bar upper channel (160.0),
    then outcome bars. Entry = 161.0, stop = 156.5, target = 182.0."""
    seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 12, interval=D_REG)
    rows_setup = donchian_rows()
    seed(conn, SETUP_TF, rows_setup, interval=D_SET)
    last_setup_ts = START + (len(rows_setup) - 1) * D_SET
    level = 160.0
    below = level - 0.8
    entry = level + 1.0
    rows_trig = [[below, below + 0.2, below - 0.2, below, 10.0]] * 21
    rows_trig.append([below, entry + 0.1, below - 0.2, entry, 30.0])  # breakout bar
    rows_trig += outcome_rows_trig
    seed(conn, TRIGGER_TF, rows_trig, start=last_setup_ts + D_SET, interval=D_TRIG)


def fade_setup_rows():
    rows = []
    for i in range(40):
        close = 99.5 if i % 2 == 0 else 100.5
        rows.append([close, close + 0.1, close - 0.1, close, 10.0])
    rows.append([99.0, 99.0, 98.7, 98.8, 10.0])  # stretch: close below lower band
    rows.append([99.2, 99.4, 98.9, 99.3, 10.0])  # recovery, back inside the bands
    return rows


def fade_candidate():
    rows = fade_setup_rows()
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.Index([START + i * D_SET for i in range(len(rows))], name="ts")
    longs = [c for c in detect_fade_setups(df) if c.direction == "long"]
    assert longs, "fixture must produce a long fade candidate"
    return longs[0]


def seed_fade_scenario(conn):
    seed(conn, REGIME_TF, [[100, 101, 99, 100, 10.0]] * 12, interval=D_REG)
    rows_setup = fade_setup_rows()
    seed(conn, SETUP_TF, rows_setup, interval=D_SET)
    last_setup_ts = START + (len(rows_setup) - 1) * D_SET
    level = fade_candidate().trigger_level
    below = level - 0.01
    entry = level + 0.05
    rows_trig = [[below, below + 0.005, below - 0.005, below, 10.0]] * 21
    rows_trig.append([below, entry + 0.02, below - 0.005, entry, 30.0])  # re-cross
    rows_trig += [[entry, entry + 0.03, entry - 0.03, entry, 10.0]] * 3
    seed(conn, TRIGGER_TF, rows_trig, start=last_setup_ts + D_SET, interval=D_TRIG)


def patch_trending(monkeypatch, label="trending"):
    """Force a regime label on BOTH dispatch paths.

    THE SINGLE MOST LIKELY CAUSE OF A CONFUSING INITIAL RED: engine.py and
    framework/context.py each import classify_series by name, so patching only
    one leaves the two paths seeing different regimes and every parity test fails
    for the wrong reason.
    """
    stub = lambda df, **kw: pd.Series(label, index=df.index, dtype=object)  # noqa: E731
    monkeypatch.setattr(engine, "classify_series", stub)
    monkeypatch.setattr(fcontext, "classify_series", stub)


FREE = dict(fee_pct=0.0, slippage_pct=0.0, funding_pct_per_day=0.0)


def both_paths(conn, *, params=None, graph=None, **kwargs):
    """Run engine and graph over the same span/costs and return (graph, engine)."""
    params = params or BacktestParams()
    graph = graph if graph is not None else build_v020_graph()
    engine.clear_caches()
    fcontext.clear_caches()
    want = run_backtest(conn, SYMBOL, params=params, **kwargs)
    got = run_graph_backtest(conn, graph, SYMBOL, **kwargs)
    return got, want


class TestSyntheticParity:
    def test_donchian_scenario_parity(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        got, want = both_paths(conn, **FREE)
        assert len(want) == 1, "fixture must produce exactly one engine trade"
        assert_trades_match(got, want)

    def test_donchian_stop_scenario_parity(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[110, 190.0, 100.0, 189.0, 10.0]])
        got, want = both_paths(conn, **FREE)
        assert want and want[0].outcome == "stop"
        assert_trades_match(got, want)

    def test_fade_scenario_parity(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        monkeypatch.setattr(config, "FADE_ENABLED", True)
        patch_trending(monkeypatch, label="ranging")
        seed_fade_scenario(conn)
        got, want = both_paths(conn, **FREE)
        assert want and want[0].pattern == FADE_KIND
        assert_trades_match(got, want)

    def test_no_trades_scenario_parity(self, tmp_path, monkeypatch):
        """A flat trigger series never crosses the channel, so both paths see the
        setup and take nothing."""
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 12, interval=D_REG)
        seed(conn, SETUP_TF, donchian_rows(), interval=D_SET)
        last = START + 59 * D_SET
        seed(conn, TRIGGER_TF, [[100, 100.2, 99.8, 100.0, 10.0]] * 40,
             start=last + D_SET, interval=D_TRIG)
        got, want = both_paths(conn, **FREE)
        assert want == []
        assert_trades_match(got, want)

    def test_empty_db_parity(self, tmp_path):
        conn = storage.connect(str(tmp_path / "empty.db"))
        got, want = both_paths(conn)
        assert got == [] and want == []

    def test_one_empty_tier_parity(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed(conn, SETUP_TF, donchian_rows(), interval=D_SET)  # no regime, no trigger
        got, want = both_paths(conn)
        assert got == [] and want == []

    @pytest.mark.parametrize(
        "trail_enabled,target_enabled,max_hold_bars",
        [
            (t, g, m)
            for t in walkforward.DEFAULT_GRID["trail_enabled"]
            for g in walkforward.DEFAULT_GRID["target_enabled"]
            for m in walkforward.DEFAULT_GRID["max_hold_bars"]
        ],
    )
    def test_parity_across_every_default_grid_combo(
        self, tmp_path, monkeypatch, trail_enabled, target_enabled, max_hold_bars
    ):
        """Every configuration THE GATE could actually select, not just defaults.

        This is the test most likely to catch an ExitPolicySpec mis-mapping, and
        it is what makes the graph safe for `walkforward --graph`.
        """
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        # A long, wandering outcome tail so trail / channel / target / time can
        # each plausibly bind depending on the combo.
        tail = []
        for i in range(120):
            base = 161.0 + 12.0 * math.sin(i / 7.0) + i * 0.15
            tail.append([base, base + 3.0, base - 3.0, base, 10.0])
        seed_scenario(conn, tail)
        got, want = both_paths(
            conn,
            params=BacktestParams(
                trail_enabled=trail_enabled, target_enabled=target_enabled
            ),
            graph=build_v020_graph(
                trail_enabled=trail_enabled, target_enabled=target_enabled
            ),
            max_hold_bars=max_hold_bars,
            **FREE,
        )
        assert want, "fixture produced no engine trade for this combo"
        assert_trades_match(got, want)

    def test_parity_with_cache_disabled(self, tmp_path, monkeypatch):
        """A run with caching off must be BIT-IDENTICAL to one with it on. If the
        two ever differ, the cache key is missing a parameter and every cached
        result since is suspect."""
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        g = build_v020_graph()
        with_cache = run_graph_backtest(conn, g, SYMBOL, **FREE)
        fcontext.clear_caches()
        monkeypatch.setattr(config, "FRAMEWORK_CACHE_ENABLED", False)
        without = run_graph_backtest(conn, g, SYMBOL, **FREE)
        assert_trades_match(without, with_cache, abs_tol=0.0)
        engine.clear_caches()
        want = run_backtest(conn, SYMBOL, **FREE)
        assert_trades_match(without, want)


STORED_SPAN_DAYS = 365


def _stored_span(conn):
    """The last STORED_SPAN_DAYS of stored trigger history, ending at the last
    CLOSED trigger bar.

    Uses the last stored bar's OPEN as the inclusive end bound, which is what
    run_backtest's `end_ms` compares against bar CLOSES — so the currently-forming
    bar (today is 2026-07-27, and the final 1d bar is still forming) can never
    enter either path. Both paths get the identical bound, so the comparison is
    valid regardless.
    """
    rows = storage.load_candles(conn, SYMBOL, TRIGGER_TF)
    end = int(rows[-1][0])
    return end - STORED_SPAN_DAYS * 86_400_000, end


@pytest.mark.skipif(
    not Path(config.DB_PATH).exists(), reason="stored OHLCV DB not present"
)
class TestStoredHistoryParity:
    """Synthetic fixtures cannot produce warmup, gaps, or regime transitions."""

    def test_parity_one_year_all_symbols(self):
        conn = storage.connect()
        try:
            start, end = _stored_span(conn)
            total = 0
            for symbol in config.SYMBOLS:
                engine.clear_caches()
                fcontext.clear_caches()
                want = run_backtest(conn, symbol, start_ms=start, end_ms=end)
                got = run_graph_backtest(
                    conn, build_v020_graph(), symbol, start_ms=start, end_ms=end
                )
                assert_trades_match(got, want)
                assert want, f"{symbol}: no trades in the {self.SPAN_DAYS}-day span"
                total += len(want)
            assert total > 0, f"pooled trade count over the span: {total}"
        finally:
            conn.close()

    # NOTE: the FULL-SPAN, all-symbols parity run is a MANUAL validation step
    # (see the phase report), not a unit test. It costs ~30 s of engine replay per
    # symbol and the one-year slice above already exercises warmup, gaps and
    # regime transitions on real data.


class TestRegimeLabels:
    """Isolates the classify_series call + the searchsorted rule."""

    def test_labels_match_engine(self, tmp_path):
        conn = storage.connect(str(tmp_path / "t.db"))
        seed(conn, REGIME_TF, [[100 + i, 111 + i, 99 + i, 105 + i, 10.0] for i in range(300)],
             interval=D_REG)
        seed(conn, SETUP_TF, donchian_rows(), interval=D_SET)
        seed(conn, TRIGGER_TF, [[100, 101, 99, 100, 10.0]] * 40,
             start=START + 60 * D_SET, interval=D_TRIG)
        from trading_bot.framework.context import EvalSession
        from trading_bot.plugins.data.ohlcv import OhlcvSource

        session = EvalSession(
            OhlcvSource(conn), SYMBOL,
            tiers=(REGIME_TF, SETUP_TF, TRIGGER_TF), regime_gate=fgraph.RegimeGate(),
        )
        df_regime = session.frame_of(REGIME_TF)
        labels = engine.classify_series(
            df_regime,
            adx_trend_threshold=config.ADX_TREND_THRESHOLD,
            atr_extreme_percentile=config.ATR_EXTREME_PERCENTILE,
        )
        close_regime = df_regime.index.to_numpy() + D_REG
        for k in range(0, 300, 7):
            t = START + k * D_REG + D_REG
            j = int(np.searchsorted(close_regime, t, side="right")) - 1
            want = str(labels.iloc[j]) if j >= 0 else "uncertain"
            assert session.context(f"c{k}", t).regime() == want


class TestCandidateLists:
    """Isolates the detector window + the regime gate.

    Compares candidate_from_event(e) against the EXACT expression
    engine.candidates_for uses (engine.py:339-342) for every setup bar.
    """

    def test_events_match_per_setup_bar(self, tmp_path):
        conn = storage.connect(str(tmp_path / "t.db"))
        seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 12, interval=D_REG)
        rows_setup = donchian_rows()
        seed(conn, SETUP_TF, rows_setup, interval=D_SET)
        seed(conn, TRIGGER_TF, [[100, 101, 99, 100, 10.0]] * 40,
             start=START + 60 * D_SET, interval=D_TRIG)
        from trading_bot.framework.context import EvalSession
        from trading_bot.plugins.data.ohlcv import OhlcvSource

        session = EvalSession(
            OhlcvSource(conn), SYMBOL,
            tiers=(REGIME_TF, SETUP_TF, TRIGGER_TF), regime_gate=fgraph.RegimeGate(),
        )
        df_setup = session.frame_of(SETUP_TF)
        spec = registry.get("detector.donchian-breakout")
        params = spec.resolve({})
        for h_idx in range(len(rows_setup)):
            ctx = session.context(f"s{h_idx}", int(df_setup.index[h_idx]) + D_SET)
            got = [
                _to_candidate(e) for e in spec.impl(ctx, **params)
            ]
            window = df_setup.iloc[
                max(0, h_idx + 1 - config.PATTERN_LOOKBACK_BARS) : h_idx + 1
            ]
            want = detect_donchian_setups(window)
            assert got == want, f"setup bar {h_idx}"


def _to_candidate(event) -> PatternCandidate:
    from trading_bot.framework.contracts import candidate_from_event

    return candidate_from_event(event)


class TestMedium2:
    """The MEDIUM-2 repair (engine.py:458-481, KNOWN-LIMITATIONS §6).

    Selecting the setup bar by the trigger bar's CLOSE instead of its OPEN
    silently discarded one trigger bar in every setup window — 25% of all trigger
    opportunities at a 4H setup / 1H trigger. It hides as GOOD NEWS: fixing it
    ADDED trades that were unprofitable in-sample (142 -> 158 trades, pooled
    Sharpe 0.431 -> 0.255). Both paths must take the breakout that lands on the
    LAST trigger bar of a setup window.
    """

    def test_last_trigger_bar_of_a_setup_window_can_trigger(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 12, interval=D_REG)
        rows_setup = donchian_rows()
        seed(conn, SETUP_TF, rows_setup, interval=D_SET)
        last_setup_ts = START + (len(rows_setup) - 1) * D_SET
        # Place the breakout on the FINAL trigger bar before the next setup bar
        # would close: bars_per_setup - 1 bars after the setup window opens.
        bars_per_setup = D_SET // D_TRIG
        level, below, entry = 160.0, 159.2, 161.0
        rows_trig = [[below, below + 0.2, below - 0.2, below, 10.0]] * 21
        # Pad so the breakout bar is the last trigger bar of a setup window.
        trig_start = last_setup_ts + D_SET
        while ((len(rows_trig) + 1) % bars_per_setup) != 0:
            rows_trig.append([below, below + 0.2, below - 0.2, below, 10.0])
        rows_trig.append([below, entry + 0.1, below - 0.2, entry, 30.0])
        rows_trig += [[entry, entry + 0.2, entry - 0.2, entry, 10.0]] * 4
        seed(conn, TRIGGER_TF, rows_trig, start=trig_start, interval=D_TRIG)

        got, want = both_paths(conn, **FREE)
        assert want, "the last trigger bar of a setup window must be able to trigger"
        assert_trades_match(got, want)
        # And the entry really is on that final bar of the window.
        entry_j = [i for i, r in enumerate(rows_trig) if r[3] == entry][0]
        assert want[0].entry_ts == trig_start + entry_j * D_TRIG
        assert ((entry_j + 1) % bars_per_setup) == 0, "fixture placement"
        assert level == 160.0


@pytest.mark.skipif(
    not Path(config.DB_PATH).exists(), reason="stored OHLCV DB not present"
)
class TestAtrAtEntry:
    """Task 14 GOTCHA b: the policy's ATR must be engine.py:492's
    atr_setup_vals[h_idx] — the ATR at the SETUP bar, keyed on the trigger bar's
    OPEN (MEDIUM-2), not at the trigger bar."""

    def test_policy_atr_equals_engine_atr_setup_vals(self):
        conn = storage.connect()
        try:
            df_setup = engine._df(conn, SYMBOL, SETUP_TF)
            atr_vals = wilder_atr(df_setup, period=config.ATR_STOP_PERIOD).to_numpy()
            close_setup = df_setup.index.to_numpy() + D_SET
            start, end = _stored_span(conn)
            trades = run_graph_backtest(
                conn, build_v020_graph(), SYMBOL, start_ms=start, end_ms=end
            )
            assert trades
            for t in trades:
                h_idx = int(np.searchsorted(close_setup, t.entry_ts, side="right")) - 1
                implied = abs(t.entry - t.stop) / config.ATR_STOP_MULTIPLE
                assert math.isclose(
                    implied, float(atr_vals[h_idx]), rel_tol=0.0, abs_tol=1e-9
                ), f"entry_ts {t.entry_ts}: implied ATR {implied} != {atr_vals[h_idx]}"
        finally:
            conn.close()


@pytest.mark.skipif(
    not Path(config.DB_PATH).exists(), reason="stored OHLCV DB not present"
)
class TestExitMix:
    """The generic exit block vs both legacy branches: a reordered elif chain is
    exactly what an outcome-count comparison detects."""

    def test_outcome_counts_match(self):
        conn = storage.connect()
        try:
            start, end = _stored_span(conn)
            engine.clear_caches()
            fcontext.clear_caches()
            want = run_backtest(conn, SYMBOL, start_ms=start, end_ms=end)
            got = run_graph_backtest(
                conn, build_v020_graph(), SYMBOL, start_ms=start, end_ms=end
            )
            counts = collections.Counter(t.outcome for t in got)
            assert counts == collections.Counter(t.outcome for t in want)
            assert sum(counts.values()) > 0, "no trades to compare outcomes over"
        finally:
            conn.close()


class TestExitAsymmetry:
    """The MANDATORY DEVIATION (engine.py:392-411, 523-530): applying the trend
    exits to the fade sleeve would silently change what Phase 6's DROP verdict
    measured."""

    def test_fade_trades_never_trail_or_channel(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        monkeypatch.setattr(config, "FADE_ENABLED", True)
        patch_trending(monkeypatch, label="ranging")
        seed_fade_scenario(conn)
        got, want = both_paths(
            conn,
            params=BacktestParams(trail_enabled=True, target_enabled=True),
            graph=build_v020_graph(trail_enabled=True, target_enabled=True),
            **FREE,
        )
        assert want and all(t.pattern == FADE_KIND for t in want)
        for t in got:
            assert t.outcome not in ("trail", "channel"), t.outcome
        assert_trades_match(got, want)

    def test_fade_branch_exits_are_structurally_incapable_of_trailing(self):
        g = build_v020_graph(trail_enabled=True)
        fade = next(b for b in g.branches if b.id == "range")
        assert fade.exits.trail_enabled is False
        assert fade.exits.channel_exit is False
        assert fade.exits.target_enabled is True


class TestRatchetOrdering:
    """The intra-bar lookahead guard (engine.py:435-452): the trail can never fire
    on the same bar that produced its own extreme."""

    def test_trail_cannot_fire_on_the_bar_that_set_its_extreme(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        # Entry 161.0, ATR 3.0, trail_atr_multiple 3.0 -> trail distance 9.0.
        # Bar A: high 200 (extreme), low 190. new_stop = 191 > low 190, so a
        # ratchet applied BEFORE this bar's exits would close the trade at 191 on
        # bar A itself. Applied after, it can only bind from bar B onward.
        seed_scenario(
            conn,
            [
                [190.0, 200.0, 190.0, 199.0, 10.0],   # bar A
                [195.0, 196.0, 185.0, 186.0, 10.0],   # bar B: touches the 191 trail
            ],
        )
        got, want = both_paths(
            conn,
            params=BacktestParams(trail_enabled=True),
            graph=build_v020_graph(trail_enabled=True),
            **FREE,
        )
        assert want and want[0].outcome == "trail"
        assert_trades_match(got, want)
        # The exit lands on bar B, not bar A.
        entry_ts = want[0].entry_ts
        assert want[0].exit_ts == entry_ts + 2 * D_TRIG
        assert math.isclose(want[0].exit_price, 191.0, abs_tol=1e-9)

    def test_trade_stop_records_the_initial_stop_not_the_ratcheted_one(
        self, tmp_path, monkeypatch
    ):
        """engine.py:114-117 — Trade.stop is a frozen record of the setup."""
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(
            conn,
            [[190.0, 200.0, 190.0, 199.0, 10.0], [195.0, 196.0, 185.0, 186.0, 10.0]],
        )
        got, _ = both_paths(
            conn,
            params=BacktestParams(trail_enabled=True),
            graph=build_v020_graph(trail_enabled=True),
            **FREE,
        )
        assert math.isclose(got[0].stop, 156.5, abs_tol=1e-9)
        assert math.isclose(got[0].exit_price, 191.0, abs_tol=1e-9)


class TestTieBreak:
    """Multi-event branches must resolve by setup.rank_signals, not by branch
    iteration order. Uses detector.legacy-patterns, which emits SEVERAL events at
    once, rather than an invented stub."""

    def test_multi_event_branch_ranks_by_rank_signals(self, tmp_path, monkeypatch):
        from trading_bot.framework import execute as fexecute
        from trading_bot.signals.setup import rank_signals as real_rank

        seen: list = []

        def spy(signals):
            ranked = real_rank(signals)
            seen.append((list(signals), ranked))
            return ranked

        monkeypatch.setattr(fexecute, "rank_signals", spy)

        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[161, 200.0, 150.0, 199.0, 10.0]] * 30)

        # Both the Donchian branch and the legacy-pattern branch on "trending", so
        # one bar can carry several signals from different branches.
        base = build_v020_graph(include_fade=False)
        legacy = fgraph.Branch(
            id="legacy",
            detector=fgraph.NodeSpec(id="lg-det", key="detector.legacy-patterns"),
            policy=fgraph.NodeSpec(id="lg-pol", key="policy.atr-stop-measured-move"),
            regimes=("trending",),
        )
        g = dataclasses.replace(base, branches=base.branches + (legacy,))
        fgraph.validate(g)
        trades = run_graph_backtest(conn, g, SYMBOL, **FREE)

        assert seen, "rank_signals was never consulted"
        for signals, ranked in seen:
            assert ranked[0] == max(signals, key=lambda s: (s.rr, -ord(s.pattern[0])))
        # Every opened trade corresponds to some rank_signals winner.
        winners = {(r[0].ts, r[0].pattern) for _, r in seen}
        for t in trades:
            assert (t.entry_ts, t.pattern) in winners

    def test_two_branches_still_yield_one_open_trade_at_a_time(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[161, 200.0, 150.0, 199.0, 10.0]] * 30)
        base = build_v020_graph(include_fade=False)
        legacy = fgraph.Branch(
            id="legacy",
            detector=fgraph.NodeSpec(id="lg-det", key="detector.legacy-patterns"),
            policy=fgraph.NodeSpec(id="lg-pol", key="policy.atr-stop-measured-move"),
            regimes=("trending",),
        )
        g = dataclasses.replace(base, branches=base.branches + (legacy,))
        trades = run_graph_backtest(conn, g, SYMBOL, **FREE)
        spans = sorted((t.entry_ts, t.exit_ts) for t in trades)
        for (_, a_exit), (b_entry, _) in zip(spans, spans[1:]):
            assert b_entry >= a_exit, "two trades overlapped on one symbol"


class TestCostArithmetic:
    """A10's duplication of engine.py:354-378, pinned by hand-computed arithmetic."""

    def test_pnl_matches_hand_computed(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        fee, slip, funding = 0.0005, 0.0002, 0.0001
        g = build_v020_graph(target_enabled=True)
        got = run_graph_backtest(
            conn, g, SYMBOL, fee_pct=fee, slippage_pct=slip,
            funding_pct_per_day=funding,
        )
        engine.clear_caches()
        want = run_backtest(
            conn, SYMBOL, params=BacktestParams(target_enabled=True),
            fee_pct=fee, slippage_pct=slip, funding_pct_per_day=funding,
        )
        assert len(got) == 1
        t = got[0]
        gross = (t.exit_price - t.entry) / t.entry
        hold_days = (t.exit_ts - t.entry_ts) / 86_400_000.0
        expected = gross - 2 * (fee + slip) - funding * hold_days
        assert math.isclose(t.pnl_pct, expected, rel_tol=0.0, abs_tol=1e-15)
        assert_trades_match(got, want)

    def test_round_trip_cost_is_two_sided(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        g = build_v020_graph(target_enabled=True)
        free = run_graph_backtest(conn, g, SYMBOL, **FREE)
        fcontext.clear_caches()
        charged = run_graph_backtest(
            conn, g, SYMBOL, fee_pct=0.001, slippage_pct=0.0,
            funding_pct_per_day=0.0,
        )
        assert math.isclose(
            free[0].pnl_pct - charged[0].pnl_pct, 0.002, rel_tol=0.0, abs_tol=1e-15
        )


class TestFadeEnabled:
    """A7 + the candidate cache key: FADE_ENABLED is read at CALL TIME inside the
    detector, so a live flip must not be served a cached pre-flip list."""

    def test_flag_read_at_call_time(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch, label="ranging")
        seed_fade_scenario(conn)
        g = build_v020_graph()  # the SAME graph object for both runs

        monkeypatch.setattr(config, "FADE_ENABLED", False)
        off = run_graph_backtest(conn, g, SYMBOL, **FREE)
        monkeypatch.setattr(config, "FADE_ENABLED", True)
        on = run_graph_backtest(conn, g, SYMBOL, **FREE)
        assert off == [], "the kill switch must suppress the sleeve"
        assert on, "flipping the flag mid-process must change the candidate set"

    def test_parity_under_both_settings(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch, label="ranging")
        seed_fade_scenario(conn)
        for enabled in (False, True):
            monkeypatch.setattr(config, "FADE_ENABLED", enabled)
            got, want = both_paths(conn, **FREE)
            assert_trades_match(got, want)
            assert bool(want) is enabled

    def test_flag_is_still_false_in_config(self):
        """FADE_ENABLED = False stays false: migrating the detector is not
        re-enabling the sleeve (config.py:96-103)."""
        assert config.FADE_ENABLED is False


class TestAssertIntervalEnforced:
    """A partial tier migration must fail LOUDLY, not silently in the flattering
    direction (engine.py:185-222)."""

    def test_both_paths_raise_the_same_message(self, tmp_path):
        conn = storage.connect(str(tmp_path / "bad.db"))
        seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 12, interval=D_REG)
        seed(conn, SETUP_TF, donchian_rows(), interval=D_SET)
        # 15m bars stored under the 1h trigger key.
        seed(conn, TRIGGER_TF, [[100, 101, 99, 100, 10.0]] * 40,
             start=START + 60 * D_SET, interval=storage.TIMEFRAME_MS["15m"])
        with pytest.raises(ValueError) as engine_exc:
            run_backtest(conn, SYMBOL)
        with pytest.raises(ValueError) as graph_exc:
            run_graph_backtest(conn, build_v020_graph(), SYMBOL)
        assert str(engine_exc.value) == str(graph_exc.value)


class TestScanDriftGuard:
    """A6's mitigation, and an honest alarm rather than a fix.

    signals/scan.py is NOT yet graph-dispatched, so live and backtest now share
    DETECTORS but not DISPATCH. There are two dispatch tables — scan.py:55-61 and
    Branch.regimes — and they can silently disagree. This test is the drift alarm
    until scan.py is migrated (recommended for Phase 4).
    """

    KIND_BY_BRANCH = {"trend": DONCHIAN_KIND, "range": FADE_KIND}

    def test_graph_branches_match_scan_dispatch_for_every_regime(self, monkeypatch):
        monkeypatch.setattr(config, "FADE_ENABLED", True)
        monkeypatch.setattr(scan, "scan_donchian_signals",
                            lambda conn, symbol, now_ms: ["DONCHIAN"])
        monkeypatch.setattr(scan, "scan_fade_signals",
                            lambda conn, symbol, now_ms: ["FADE"])
        g = build_v020_graph()
        expected_sentinel = {DONCHIAN_KIND: "DONCHIAN", FADE_KIND: "FADE"}

        for label in REGIMES:
            monkeypatch.setattr(
                scan, "current_regime",
                lambda conn, symbol, now_ms=None, _l=label: (_l, 0.0, 0.0),
            )
            _, signals = scan.scan_symbol(None, SYMBOL, now_ms=START)
            branches = [
                b for b in g.ordered_branches()
                if b.enabled and (label in b.regimes or "any" in b.regimes)
            ]
            if not signals:
                assert branches == [], (
                    f"regime {label!r}: scan dispatches NOTHING but the graph "
                    f"activates {[b.id for b in branches]} — the two dispatch "
                    f"tables have drifted"
                )
                continue
            assert len(branches) == 1, f"regime {label!r}: {[b.id for b in branches]}"
            kind = self.KIND_BY_BRANCH[branches[0].id]
            assert signals == [expected_sentinel[kind]], f"regime {label!r}"

    def test_kill_switch_suppresses_on_both_sides(self, monkeypatch):
        monkeypatch.setattr(config, "FADE_ENABLED", False)
        monkeypatch.setattr(scan, "scan_fade_signals",
                            lambda conn, symbol, now_ms: ["FADE"])
        monkeypatch.setattr(
            scan, "current_regime", lambda conn, symbol, now_ms=None: ("ranging", 0.0, 0.0)
        )
        _, signals = scan.scan_symbol(None, SYMBOL, now_ms=START)
        assert signals == []
        spec = registry.get("detector.bollinger-fade")
        assert spec.impl(object(), **spec.resolve({})) == []


class TestWalkForwardGraphPath:
    """The `strategy=` keyword (contract §5), and the guard that refuses a grid a
    graph would silently ignore."""

    SPAN_DAYS = 150  # train 60 + test 30 + oos 30 needs 120; 150 gives one fold

    def _long_db(self, tmp_path):
        conn = storage.connect(str(tmp_path / "wf.db"))
        n_reg = self.SPAN_DAYS + 20
        seed(conn, REGIME_TF,
             [[100 + i, 111 + i, 99 + i, 105 + i, 10.0] for i in range(n_reg)],
             interval=D_REG)
        n_set = n_reg * (D_REG // D_SET)
        seed(conn, SETUP_TF,
             [[100.0 + (i % 40), 102.0 + (i % 40), 99.0 + (i % 40), 101.0 + (i % 40), 10.0]
              for i in range(n_set)],
             interval=D_SET)
        n_trig = n_reg * (D_REG // D_TRIG)
        seed(conn, TRIGGER_TF,
             [[100.0 + (i % 60), 103.0 + (i % 60), 98.0 + (i % 60), 101.0 + (i % 60), 10.0]
              for i in range(n_trig)],
             interval=D_TRIG)
        return conn

    def test_unsupported_grid_axis_raises(self, tmp_path):
        conn = self._long_db(tmp_path)
        with pytest.raises(ValueError, match="carries its own parameters"):
            walkforward.walk_forward_pooled(
                conn, [SYMBOL], start_ms=START,
                end_ms=START + self.SPAN_DAYS * 86_400_000,
                strategy=build_v020_graph(),
            )

    def test_graph_path_oos_trades_equal_the_engine_path(self, tmp_path, monkeypatch):
        conn = self._long_db(tmp_path)
        patch_trending(monkeypatch)
        end = START + self.SPAN_DAYS * 86_400_000
        grid = {"max_hold_bars": (48, 96)}
        kw = dict(
            start_ms=START, end_ms=end, grid=grid,
            train_days=60, test_days=30, oos_days=30, min_trades=1,
        )
        got = walkforward.walk_forward_pooled(conn, [SYMBOL], strategy=build_v020_graph(), **kw)
        engine.clear_caches()
        fcontext.clear_caches()
        want = walkforward.walk_forward_pooled(conn, [SYMBOL], **kw)

        assert got.final_max_hold_bars == want.final_max_hold_bars
        assert got.oos_start == want.oos_start and got.oos_end == want.oos_end
        assert got.oos_metrics["n_trades"] == want.oos_metrics["n_trades"]

        # And the OOS trade LISTS themselves, at the same final parameters.
        engine.clear_caches()
        fcontext.clear_caches()
        e_trades = run_backtest(
            conn, SYMBOL, start_ms=got.oos_start, end_ms=got.oos_end,
            params=want.final_params, max_hold_bars=want.final_max_hold_bars,
        )
        g_trades = run_graph_backtest(
            conn, build_v020_graph(), SYMBOL,
            start_ms=got.oos_start, end_ms=got.oos_end,
            max_hold_bars=got.final_max_hold_bars,
        )
        assert_trades_match(g_trades, e_trades)

    def test_cli_loads_the_registry_before_validating_the_graph(self, tmp_path, monkeypatch):
        """REGRESSION: _walkforward_command validated a graph with an EMPTY
        registry, so a perfectly good committed graph was reported as five
        unknown plug-ins and the run aborted. graph.validate resolves every node
        against the registry, so load_all() must come first.
        """
        from trading_bot.cli import _walkforward_command

        conn = self._long_db(tmp_path)
        patch_trending(monkeypatch)
        monkeypatch.setattr(config, "STRATEGY_DIR", str(tmp_path / "s"))
        path = fgraph.save(build_v020_graph())
        with registry.temporary_registry():
            registry.REGISTRY.clear()  # safe: temporary_registry restores it
            monkeypatch.setattr(config, "WF_TRAIN_DAYS", 60)
            monkeypatch.setattr(config, "WF_TEST_DAYS", 30)
            monkeypatch.setattr(config, "WF_OOS_DAYS", 30)
            monkeypatch.setattr(config, "WF_MIN_TRADES", 1)
            rc = _walkforward_command(
                conn, [SYMBOL], start_ms=START,
                end_ms=START + self.SPAN_DAYS * 86_400_000,
                graph_path=str(path),
            )
        # The gate verdict is irrelevant here (0 or 1); what must NOT happen is an
        # abort on "unknown plug-in".
        assert rc in (0, 1)
        assert registry.REGISTRY, "load_all() must have repopulated the registry"

    def test_strategy_none_is_the_default(self):
        """Every pre-existing caller must be unaffected: `strategy` is the LAST
        keyword-only parameter and defaults to None."""
        import inspect

        sig = inspect.signature(walkforward.walk_forward_pooled)
        params = list(sig.parameters)
        assert params[-1] == "strategy"
        assert sig.parameters["strategy"].default is None
        assert sig.parameters["strategy"].kind is inspect.Parameter.KEYWORD_ONLY
