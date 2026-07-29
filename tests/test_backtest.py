"""Tests for the Phase 5 backtest engine, metrics, and walk-forward harness."""

import dataclasses
import math
import statistics

import pandas as pd
import pytest

from trading_bot import config
from trading_bot.backtest import engine, walkforward
from trading_bot.backtest.benchmark import METRIC_KEYS, BenchmarkResult
from trading_bot.backtest.engine import BacktestParams, Trade, run_backtest
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.backtest.walkforward import DAY_MS, WalkForwardResult, walk_forward_pooled
from trading_bot.cli import _backtest_command, _walkforward_command
from trading_bot.data import storage
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.risk.atr_stop import compute_atr_stop, cost_ratio
from trading_bot.signals.donchian import DONCHIAN_KIND
from trading_bot.signals.meanrev import detect_fade_setups
from trading_bot.signals.patterns import PatternCandidate

SYMBOL = "BTCUSDT"
# Tier-derived, never hardcoded: these fixtures follow config forever, so a
# future tier shift cannot leave the tests on the old timeframes while
# production moves (the classic way a tier change passes CI while being wrong).
REGIME_TF = config.REGIME_TIMEFRAME
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_REG = storage.TIMEFRAME_MS[REGIME_TF]
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    """Clear the engine's indicator memo around every test.

    engine._CACHE is keyed by a content fingerprint, so cross-fixture collisions
    should be impossible — but test isolation must not DEPEND on that argument
    being right. Clearing makes each test independent of execution order.
    """
    engine.clear_caches()
    yield
    engine.clear_caches()


def make_trade(pnl, regime="trending", pattern="flag", exit_ts=None, entry_ts=None):
    """entry_ts defaults to START, preserving every pre-existing call site.

    Any fake that MOVES exit_ts off START must pass entry_ts too: under
    equity.py's spread attribution (MEDIUM-5) a stale entry_ts far outside the
    scored span makes the trade non-overlapping, and the metrics layer drops it.
    """
    exit_ts = START + D_TRIG if exit_ts is None else exit_ts
    return Trade(
        symbol=SYMBOL, regime=regime, pattern=pattern, direction="long",
        entry_ts=START if entry_ts is None else entry_ts,
        entry=100.0, stop=99.0, target=101.0,
        exit_ts=exit_ts, exit_price=100.0, outcome="target",
        pnl_pct=pnl, volume_high=False,
    )


def _beatable_benchmark(ann_return_pct=0.0, sharpe=0.0):
    """A BENCHMARK the strategy can beat, for the conn=None walk-forward tests.

    buy_and_hold reads SQLite, and every existing walk-forward test passes
    conn=None, so the null must be stubbed at walkforward's module level (the
    seam run_backtest already uses). A BEATABLE null keeps those tests failing
    for the reasons their names claim — an unbeatable one would make
    test_gate_requires_sharpe_and_dsr pass for the wrong reason.
    """
    bundle = {
        "total_return": 1.0, "ann_return_pct": ann_return_pct, "sharpe": sharpe,
        "sortino": sharpe, "max_drawdown_pct": 0.0, "n_days": 5,
    }
    return BenchmarkResult(
        per_symbol={SYMBOL: dict(bundle)}, basket=dict(bundle), start_ms=0, end_ms=1
    )


class TestComputeMetrics:
    def test_empty_trades(self):
        m = compute_metrics([])
        assert m["n_trades"] == 0
        assert m["win_rate"] is None
        assert m["by_bucket"] == {}

    def test_basic_stats(self):
        m = compute_metrics([make_trade(0.01), make_trade(-0.005), make_trade(0.02)])
        assert m["n_trades"] == 3
        assert math.isclose(m["win_rate"], 2 / 3)
        assert math.isclose(m["expectancy_pct"], 0.025 / 3)
        assert math.isclose(m["profit_factor"], 0.03 / 0.005)
        # Cumulative curve: 0.01 -> 0.005 -> 0.025; dd = 0.005 from the peak.
        assert math.isclose(m["max_drawdown_pct"], 0.005)

    def test_buckets_split_by_regime_and_pattern(self):
        m = compute_metrics(
            [make_trade(0.01), make_trade(-0.01, regime="ranging", pattern="bollinger-fade")]
        )
        assert set(m["by_bucket"]) == {"trending/flag", "ranging/bollinger-fade"}
        assert m["by_bucket"]["trending/flag"]["n_trades"] == 1


def seed(conn, timeframe, rows, start=START, interval=D_SET):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    storage.upsert_candles(conn, SYMBOL, timeframe, data)
    return data


def donchian_rows():
    """Setup-timeframe ramp: 60 bars, ADX(14) >> 25 after warmup, last bar
    closes exactly at the trailing 20-bar upper channel (160.0) and above the
    55-bar mid (131.5) — a clean Donchian long setup. Constant true range (3)
    gives an exact, easy-to-check ATR(14) = 3.0.
    """
    return [[100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 10.0] for i in range(60)]


def donchian_atr_at_entry():
    """The ATR value the engine computes at the Donchian breakout's h_idx.

    Mirrors engine.run_backtest exactly: ATR is Wilder-smoothed over the same
    setup df (donchian_rows), and the breakout's h_idx is the last setup bar
    (the trigger entry bar closes well after it), so this is simply the last
    ATR value.
    """
    df_setup = pd.DataFrame(
        donchian_rows(), columns=["open", "high", "low", "close", "volume"]
    )
    return float(wilder_atr(df_setup, period=config.ATR_STOP_PERIOD).iloc[-1])


def seed_scenario(conn, outcome_rows_trig):
    """Seed regime/setup/trigger data: Donchian ramp + trigger breakout above
    the 20-bar upper channel (160.0), then outcome bars.

    Entry = 161.0, stop = compute_atr_stop(161.0, "long", 3.0, ATR_STOP_MULTIPLE)
    = 156.5, target = 160.0 + 22.0 (20-bar channel width) = 182.0.
    """
    seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 12, interval=D_REG)
    rows_setup = donchian_rows()
    seed(conn, SETUP_TF, rows_setup, interval=D_SET)
    last_setup_ts = START + (len(rows_setup) - 1) * D_SET
    level = 160.0
    below = level - 0.8
    entry = level + 1.0
    rows_trig = [[below, below + 0.2, below - 0.2, below, 10.0]] * 21
    rows_trig.append([below, entry + 0.1, below - 0.2, entry, 30.0])  # breakout bar (entry)
    rows_trig += outcome_rows_trig
    # The trigger series must begin AFTER the last setup bar closes, so the
    # trigger bar is genuinely post-pattern.
    seed(conn, TRIGGER_TF, rows_trig, start=last_setup_ts + D_SET, interval=D_TRIG)


def fade_setup_rows():
    """Setup-tier ranging bars: 40 alternating, one stretch below the lower band, one recovery.

    Mirrors tests/test_meanrev.py's alternating_rows/long_stretch_df shape.
    Produces exactly one long FadeCandidate (see fade_candidate()).
    """
    rows = []
    for i in range(40):
        close = 99.5 if i % 2 == 0 else 100.5
        rows.append([close, close + 0.1, close - 0.1, close, 10.0])
    rows.append([99.0, 99.0, 98.7, 98.8, 10.0])  # stretch: close below lower band
    rows.append([99.2, 99.4, 98.9, 99.3, 10.0])  # recovery, back inside the bands
    return rows


def fade_candidate():
    """The long FadeCandidate the engine detects from fade_setup_rows().

    Re-calls the production detector rather than hardcoding levels, so the
    trigger fixture below cannot drift away from what the engine actually sees.
    """
    rows = fade_setup_rows()
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.Index([START + i * D_SET for i in range(len(rows))], name="ts")
    longs = [c for c in detect_fade_setups(df) if c.direction == "long"]
    assert longs, "fixture must produce a long fade candidate"
    return longs[0]


def seed_fade_scenario(conn):
    """Seed regime/setup/trigger data for one ranging fade entry.

    The trigger series sits below the candidate's trigger level for 21 bars,
    then closes freshly above it (the re-cross that fires the fade), then
    meanders without reaching stop or target — so the trade resolves at data
    end and lands in the ranging/bollinger-fade bucket.
    """
    seed(conn, REGIME_TF, [[100, 101, 99, 100, 10.0]] * 12, interval=D_REG)
    rows_setup = fade_setup_rows()
    seed(conn, SETUP_TF, rows_setup, interval=D_SET)
    last_setup_ts = START + (len(rows_setup) - 1) * D_SET

    level = fade_candidate().trigger_level
    below = level - 0.01  # under the band, but above the 98.7 excursion stop
    entry = level + 0.05
    rows_trig = [[below, below + 0.005, below - 0.005, below, 10.0]] * 21
    rows_trig.append([below, entry + 0.02, below - 0.005, entry, 30.0])  # re-cross
    rows_trig += [[entry, entry + 0.03, entry - 0.03, entry, 10.0]] * 3
    seed(conn, TRIGGER_TF, rows_trig, start=last_setup_ts + D_SET, interval=D_TRIG)


def patch_trending(monkeypatch, label="trending"):
    monkeypatch.setattr(
        engine, "classify_series",
        lambda df, **kw: pd.Series(label, index=df.index, dtype=object),
    )


class TestRunBacktest:
    def test_target_exit(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        # Target = 160.0 + 22.0 (channel width) = 182.0; bar reaches high 190
        # without hitting the ATR-based stop (156.5, see donchian_atr_at_entry).
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        # The measured-move target is an exit only when target_enabled; it
        # defaults OFF (see config's exit-management block), so this
        # target-mechanics test opts in explicitly.
        trades = run_backtest(
            conn, SYMBOL, fee_pct=0.001, slippage_pct=0.0, funding_pct_per_day=0.0,
            params=BacktestParams(target_enabled=True),
        )
        assert len(trades) == 1
        t = trades[0]
        assert t.outcome == "target"
        assert t.entry == 161.0
        expected_stop = compute_atr_stop(
            t.entry, "long", donchian_atr_at_entry(), config.ATR_STOP_MULTIPLE
        )
        assert math.isclose(t.stop, expected_stop)
        gross = (t.target - t.entry) / t.entry
        assert math.isclose(t.pnl_pct, gross - 0.002)  # 2 * fee, no slippage, no funding
        assert t.regime == "trending"
        assert t.pattern == DONCHIAN_KIND

    def test_stop_exit_conservative_same_bar(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        # Bar touches BOTH the ATR-based stop (low well beneath it) and target
        # (high 190): stop wins by the engine's conservative same-bar rule.
        seed_scenario(conn, [[110, 190.0, 100.0, 189.0, 10.0]])
        trades = run_backtest(conn, SYMBOL)
        assert len(trades) == 1
        t = trades[0]
        assert t.outcome == "stop"
        expected_stop = compute_atr_stop(
            t.entry, "long", donchian_atr_at_entry(), config.ATR_STOP_MULTIPLE
        )
        assert math.isclose(t.exit_price, expected_stop)
        assert t.pnl_pct < 0

    def test_time_exit(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        meander = [[161.3, 161.5, 161.1, 161.3, 10.0]] * 3
        seed_scenario(conn, meander)
        trades = run_backtest(
            conn, SYMBOL, max_hold_bars=2, fee_pct=0.0, slippage_pct=0.0,
            funding_pct_per_day=0.0,
        )
        assert len(trades) == 1
        t = trades[0]
        assert t.outcome == "time"
        assert math.isclose(t.pnl_pct, (161.3 - 161.0) / 161.0)

    def test_unresolved_trade_closed_at_data_end(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[161.2, 161.5, 160.8, 161.2, 10.0]])
        trades = run_backtest(conn, SYMBOL)  # default max_hold far larger than data
        assert len(trades) == 1
        assert trades[0].outcome == "end"
        assert trades[0].exit_price == 161.2

    def test_inactive_regime_produces_no_trades(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch, label="extreme-volatility")
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        assert run_backtest(conn, SYMBOL) == []

    def test_empty_db_returns_empty(self, tmp_path):
        conn = storage.connect(str(tmp_path / "t.db"))
        assert run_backtest(conn, SYMBOL) == []


def donchian_exit_scenario(
    monkeypatch,
    tmp_path,
    *,
    exit_lower,
    exit_upper,
    outcome_rows_trig,
    target_height=1000.0,
    params=None,
):
    """Engine-mechanics fixture for the Phase 5 trail/channel exit tests.

    detect_donchian_setups and channel_exit_levels are monkeypatched directly
    on the engine module (same pattern as patch_trending) so the exit tests
    control the candidate's level/target and the opposite-channel value
    exactly, rather than fighting to derive them from organic geometry. ATR is
    still the REAL wilder_atr computed from a flat 30-bar setup series (true
    range constant at 2.0 => ATR(14) = 2.0), so the ATR-trail arithmetic below
    is genuine, not injected.

    Entry = 101.0, initial stop = compute_atr_stop(101.0, "long", 2.0,
    ATR_STOP_MULTIPLE) = 98.0.

    These are ENGINE-MECHANICS tests, so both optional exit modes are enabled by
    default here (production defaults them off — see config's exit-management
    block). The trail multiple is pinned to ATR_STOP_MULTIPLE so the hand-checked
    arithmetic in the trail tests (200 - 1.5*2 = 197) stays valid now that the
    trail has its own, wider, independent multiple.
    """
    if params is None:
        params = BacktestParams(
            trail_enabled=True,
            trail_atr_multiple=config.ATR_STOP_MULTIPLE,
            target_enabled=True,
        )
    patch_trending(monkeypatch)
    candidate = PatternCandidate(
        kind=DONCHIAN_KIND,
        direction="long",
        breakout_level=100.0,
        target_height=target_height,
        start_ts=START,
        end_ts=START,
    )
    monkeypatch.setattr(engine, "detect_donchian_setups", lambda window: [candidate])

    def fake_channel_exit_levels(df, *, period=None):
        return pd.DataFrame(
            {
                "upper": exit_upper,
                "lower": exit_lower,
                "mid": (exit_upper + exit_lower) / 2.0,
            },
            index=df.index,
        )

    monkeypatch.setattr(engine, "channel_exit_levels", fake_channel_exit_levels)

    conn = storage.connect(str(tmp_path / "t.db"))
    setup_rows = [[100, 101, 99, 100, 10.0]] * 30  # constant true range 2.0
    seed(conn, REGIME_TF, [[100, 111, 99, 105, 10.0]] * 30, interval=D_REG)
    seed(conn, SETUP_TF, setup_rows, interval=D_SET)
    last_setup_ts = START + (len(setup_rows) - 1) * D_SET

    below = 99.5
    entry = 101.0
    rows_trig = [[below, below + 0.1, below - 0.1, below, 10.0]] * 21
    rows_trig.append([below, entry + 0.1, below - 0.1, entry, 30.0])  # breakout bar (entry)
    rows_trig += outcome_rows_trig
    seed(conn, TRIGGER_TF, rows_trig, start=last_setup_ts + D_SET, interval=D_TRIG)

    return run_backtest(
        conn, SYMBOL, fee_pct=0.0, slippage_pct=0.0, funding_pct_per_day=0.0,
        params=params,
    )


class TestDonchianExits:
    """The Phase 5 trail ratchet and opposite-channel exit, DONCHIAN TRADES
    ONLY (see the MANDATORY DEVIATION note in engine.py's exit loop — fade
    trades never reach this code path).
    """

    def test_trail_ratchets_and_exits_above_initial_stop(self, tmp_path, monkeypatch):
        trades = donchian_exit_scenario(
            monkeypatch,
            tmp_path,
            exit_lower=0.0,
            exit_upper=300.0,
            outcome_rows_trig=[
                [110, 110.0, 109.0, 110.0, 10.0],  # favorable: extreme -> 110, stop ratchets to 107
                [107, 108.0, 106.0, 107.0, 10.0],  # pullback: low 106 <= ratcheted stop 107
            ],
        )
        assert len(trades) == 1
        t = trades[0]
        assert t.outcome == "trail"
        assert t.exit_price == 107.0
        assert t.exit_price > t.stop  # t.stop is the FROZEN initial stop (98.0)
        assert t.pnl_pct > 0

    def test_trail_never_loosens(self, tmp_path, monkeypatch):
        trades = donchian_exit_scenario(
            monkeypatch,
            tmp_path,
            exit_lower=0.0,
            exit_upper=300.0,
            outcome_rows_trig=[
                [110, 110.0, 109.0, 110.0, 10.0],  # favorable: extreme -> 110, stop ratchets to 107
                [90, 105.0, 90.0, 95.0, 10.0],  # deep adverse bar, low 90 far below both stops
            ],
        )
        assert len(trades) == 1
        t = trades[0]
        assert t.outcome == "trail"
        # Exit is at the RATCHETED level (107.0), not the initial stop (98.0)
        # and not the adverse bar's own low (90.0) — the trail never loosens.
        assert t.exit_price == 107.0
        assert t.exit_price > t.stop

    def test_trail_cannot_fire_on_the_bar_that_set_its_own_extreme(self, tmp_path, monkeypatch):
        trades = donchian_exit_scenario(
            monkeypatch,
            tmp_path,
            exit_lower=0.0,
            exit_upper=300.0,
            outcome_rows_trig=[
                # This bar's own low (196.9) is BELOW what its own extreme
                # would ratchet the stop to (200 - 1.5*2 = 197). If the trail
                # fired intra-bar, the trade would close HERE at 197.
                [200, 200.0, 196.9, 199.0, 10.0],
                # Only on this SECOND bar does the ratcheted stop (197, set at
                # the END of the previous bar) actually bind.
                [196, 198.0, 196.5, 197.0, 10.0],
            ],
        )
        assert len(trades) == 1
        t = trades[0]
        assert t.outcome == "trail"
        assert t.exit_price == 197.0
        assert t.exit_ts > t.entry_ts + D_TRIG  # closed on the SECOND bar, not the first

    def test_opposite_channel_touch_exits(self, tmp_path, monkeypatch):
        trades = donchian_exit_scenario(
            monkeypatch,
            tmp_path,
            exit_lower=99.5,  # above the initial stop (98.0): channel wins first
            exit_upper=300.0,
            outcome_rows_trig=[
                [99.4, 101.2, 99.4, 100.0, 10.0],  # low 99.4 touches the channel, not the stop
            ],
        )
        assert len(trades) == 1
        t = trades[0]
        assert t.outcome == "channel"
        assert t.exit_price == 99.5

    def test_stop_wins_over_channel_and_target_on_the_same_bar(self, tmp_path, monkeypatch):
        trades = donchian_exit_scenario(
            monkeypatch,
            tmp_path,
            exit_lower=99.0,
            exit_upper=300.0,
            target_height=6.0,  # target = 106.0, reachable within this bar
            outcome_rows_trig=[
                [90, 110.0, 90.0, 105.0, 10.0],  # touches stop (98), channel (99), and target (106)
            ],
        )
        assert len(trades) == 1
        t = trades[0]
        assert t.outcome in {"stop", "trail"}
        assert t.exit_price == 98.0


class TestFadeEnabledSwitch:
    """Phase 6 kill switch on the BACKTEST dispatch path.

    The live path is covered in tests/test_meanrev.py::TestFadeEnabledSwitch.
    Both must honor the switch identically, or backtest and live drift — the
    one thing engine.py's docstring promises never happens.
    """

    def test_fade_enabled_produces_a_ranging_bucket(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch, label="ranging")
        monkeypatch.setattr(config, "FADE_ENABLED", True)
        seed_fade_scenario(conn)
        m = compute_metrics(run_backtest(conn, SYMBOL))
        assert "ranging/bollinger-fade" in m["by_bucket"]

    def test_fade_disabled_produces_no_ranging_trades(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch, label="ranging")
        monkeypatch.setattr(config, "FADE_ENABLED", False)
        seed_fade_scenario(conn)
        m = compute_metrics(run_backtest(conn, SYMBOL))
        assert not [k for k in m["by_bucket"] if k.startswith("ranging/")]
        assert m["n_trades"] == 0


def per_symbol_fake_run_factory(peak_exp=0.01, n_trades=8, bad_symbol=None):
    """run_backtest stub whose expectancy peaks at rr_floor=1.5, and where
    `bad_symbol` (if given) always returns negative expectancy — lets tests
    exercise the mandatory per-symbol gate.

    Trade exit_ts are spread evenly across [start_ms, end_ms) (rather than
    pinned to a fixed timestamp) so compute_equity_metrics' calendar-day
    bucketing (backtest/equity.py's daily_returns) actually sees these
    trades inside the queried span — required for Sharpe/DSR to be
    non-trivially computed on the OOS window instead of silently None.

    entry_ts must be spread the same way, and for the same reason. Leaving it
    at make_trade's default START = 1_700_000_000_000 while the test span is
    0 .. 25*DAY_MS puts the entry DAY (19675) far after the exit day, so under
    spread attribution (MEDIUM-5) every fake trade fails the overlap test and
    is dropped — the folds would score an all-zero series and this class would
    keep passing for a degenerate reason.
    """

    def fake_run(conn, symbol, *, start_ms=None, end_ms=None, params=None, **kw):
        params = params or BacktestParams()
        base = -0.01 if symbol == bad_symbol else peak_exp
        exp = base * (1 - abs(params.rr_floor - 1.5) / 1.5) if base > 0 else base
        span = (end_ms or 0) - (start_ms or 0)
        step = max(1, span // (n_trades + 1))
        return [
            make_trade(exp, exit_ts=ts, entry_ts=ts)
            for ts in ((start_ms or 0) + step * (i + 1) for i in range(n_trades))
        ]

    return fake_run


class TestWalkForwardPooled:
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    SPAN = dict(start_ms=0, end_ms=25 * DAY_MS)
    # n_trades=8/symbol x 3 symbols = 24 pooled trades per fold/combo, so
    # min_trades=12 clears easily without falling back to _default_combo
    # (the original per-symbol tests bit this exact gotcha at min_trades=5).
    KNOBS = dict(train_days=10, test_days=5, oos_days=5, min_trades=12)

    @pytest.fixture(autouse=True)
    def _stub_benchmark(self, monkeypatch):
        """conn is None here, so buy_and_hold cannot read SQLite. Stub a
        BEATABLE null (Sharpe 0.0, ann 0.0) so the existing tests keep failing
        for the reasons their names claim — an unbeatable null would make
        test_gate_requires_sharpe_and_dsr pass for the wrong reason, and its
        docstring says a False verdict "can only come from Sharpe/DSR/max_dd"."""
        monkeypatch.setattr(
            walkforward, "buy_and_hold", lambda conn, syms, **kw: _beatable_benchmark()
        )

    def test_pools_across_symbols(self, monkeypatch):
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        assert len(result.folds) == 2
        for fold in result.folds:
            assert fold.best_params.rr_floor == 1.5
            # 8 trades/symbol x 3 symbols = 24 per fold's test window.
            assert fold.test_metrics["n_trades"] == 24
        assert result.final_params.rr_floor == 1.5

    def test_cross_symbol_gate_fails_on_one_bad_symbol(self, monkeypatch):
        monkeypatch.setattr(
            walkforward, "run_backtest",
            per_symbol_fake_run_factory(bad_symbol="SOLUSDT"),
        )
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        assert result.per_symbol_expectancy["SOLUSDT"] < 0
        assert result.per_symbol_expectancy["BTCUSDT"] > 0
        assert result.per_symbol_expectancy["ETHUSDT"] > 0
        assert result.passed is False  # even though BTC/ETH are positive

    def test_gate_requires_sharpe_and_dsr(self, monkeypatch):
        """Positive expectancy on every symbol and enough trades to clear
        the sample-size gate, but wildly-varying per-trade pnl (one huge
        winner, many small losers) should still fail the pooled gate on
        Sharpe/DSR — proving those conditions are actually enforced, not
        just the sample-size/per-symbol ones."""

        def volatile_fake_run(conn, symbol, *, start_ms=None, end_ms=None, params=None, **kw):
            span = (end_ms or 0) - (start_ms or 0)
            n = 12
            step = max(1, span // (n + 1))
            pnls = [-0.01] * (n - 1) + [0.6]  # one huge winner, many small losers
            # entry_ts=exit_ts for the reason in per_symbol_fake_run_factory's
            # docstring: a stale entry_ts drops every trade under spread
            # attribution.
            return [
                make_trade(p, exit_ts=(start_ms or 0) + step * (i + 1),
                           entry_ts=(start_ms or 0) + step * (i + 1))
                for i, p in enumerate(pnls)
            ]

        monkeypatch.setattr(walkforward, "run_backtest", volatile_fake_run)
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        # Sanity: sample size and per-symbol expectancy both clear on their
        # own, so a False verdict here can only come from Sharpe/DSR/max_dd.
        assert result.oos_metrics["n_trades"] >= self.KNOBS["min_trades"]
        for exp in result.per_symbol_expectancy.values():
            assert exp is not None and exp > 0
        assert result.oos_equity["sharpe"] is None or result.oos_equity["sharpe"] < walkforward.GATE_MIN_SHARPE or (
            result.oos_equity["dsr"] is None or result.oos_equity["dsr"] <= walkforward.GATE_MIN_DSR
        )
        assert result.passed is False

    def test_positive_neighbour_fraction_replaces_old_ratio_metric(self, monkeypatch):
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        expected_fields = {
            "train_start", "train_end", "test_start", "test_end", "best_params",
            "train_expectancy", "positive_neighbour_fraction", "neighbour_spread",
            "test_metrics",
        }
        for fold in result.folds:
            assert fold.positive_neighbour_fraction is None or (
                0.0 <= fold.positive_neighbour_fraction <= 1.0
            )
            # Exact field set: the old mean/best ratio field is fully gone,
            # not merely unused.
            assert {f.name for f in dataclasses.fields(fold)} == expected_fields

    def test_min_trades_30_default(self):
        assert config.WF_MIN_TRADES == 30

    def test_too_few_trades_uses_defaults_and_fails_gate(self, monkeypatch):
        monkeypatch.setattr(
            walkforward, "run_backtest", per_symbol_fake_run_factory(n_trades=1)
        )
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        # 1 trade/symbol x 3 symbols = 3 < min_trades=12: no combo reaches
        # min_trades, so folds fall back to config defaults.
        assert result.folds[0].best_params.rr_floor == config.RR_FLOOR
        assert result.folds[0].train_expectancy is None
        assert result.passed is False

    def test_span_too_short_raises(self, monkeypatch):
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        with pytest.raises(ValueError):
            walk_forward_pooled(
                None, self.SYMBOLS, start_ms=0, end_ms=10 * DAY_MS, **self.KNOBS
            )

    # --- v0.3.0 Phase 1: the gate is a per-condition dict with a real null ---

    def test_gate_is_a_dict_keyed_by_gate_conditions(self, monkeypatch):
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        assert set(result.gate) == set(walkforward.GATE_CONDITIONS)
        assert len(result.gate) == 7
        assert all(isinstance(v, bool) for v in result.gate.values())

    def test_passed_is_still_a_bool_and_equals_all_of_gate(self, monkeypatch):
        """Existing callers assert `result.passed is False`, which is
        identity-sensitive, so `passed` must stay a plain bool."""
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        assert isinstance(result.passed, bool)
        assert result.passed is all(result.gate.values())

    def test_beats_benchmark_return_fails_against_an_unbeatable_null(self, monkeypatch):
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        # Measure the strategy's own annualized return first (this fake's is
        # enormous: 24 winning trades compounded over a 5-day OOS window), then
        # give the null one point more. Pinning a literal here would silently
        # stop testing the comparison the day the fake's arithmetic changes.
        baseline = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        assert baseline.gate["beats_benchmark_return"] is True
        unbeatable = baseline.oos_equity["ann_return_pct"] + 1.0
        monkeypatch.setattr(
            walkforward, "buy_and_hold",
            lambda conn, syms, **kw: _beatable_benchmark(ann_return_pct=unbeatable),
        )
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        assert result.gate["beats_benchmark_return"] is False
        assert result.passed is False

    def test_beats_benchmark_sharpe_fails_against_an_unbeatable_null(self, monkeypatch):
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        monkeypatch.setattr(
            walkforward, "buy_and_hold",
            lambda conn, syms, **kw: _beatable_benchmark(sharpe=99.0),
        )
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        assert result.gate["beats_benchmark_sharpe"] is False
        assert result.passed is False

    def test_missing_benchmark_fails_the_gate(self, monkeypatch):
        """An UNCOMPUTABLE null is a FAIL, not a pass: if the comparison was
        not made, the strategy has not been shown to beat inaction."""
        empty = BenchmarkResult(
            per_symbol={}, basket={**dict.fromkeys(METRIC_KEYS, None), "n_days": 0},
            start_ms=0, end_ms=1,
        )
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        monkeypatch.setattr(walkforward, "buy_and_hold", lambda conn, syms, **kw: empty)
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        assert result.gate["beats_benchmark_return"] is False
        assert result.gate["beats_benchmark_sharpe"] is False
        assert result.passed is False

    def test_n_trials_used_is_reported(self, monkeypatch):
        """The ledgerless default is len(combos) * len(folds), unchanged from
        v0.2.0 — it is now RECORDED on the result instead of discarded."""
        monkeypatch.setattr(walkforward, "run_backtest", per_symbol_fake_run_factory())
        result = walk_forward_pooled(None, self.SYMBOLS, **self.SPAN, **self.KNOBS)
        n_combos = len(walkforward._combos(walkforward.DEFAULT_GRID))
        assert result.n_trials_used == n_combos * len(result.folds)

    def test_empty_symbol_dict_fails_per_symbol_condition(self):
        """`all(... for _ in {}.values())` is vacuously True, so the old gate
        passed condition 5 on ZERO symbols. Zero symbols is not "every symbol
        positive"."""
        gate = walkforward._evaluate_gate(
            {"n_trades": 100},
            {"sharpe": 2.0, "dsr": 0.99, "max_drawdown_pct": 0.01,
             "ann_return_pct": 1.0},
            {},
            10,
            _beatable_benchmark(),
        )
        assert gate["per_symbol_expectancy"] is False


class TestBacktestCli:
    def test_backtest_command_prints_metrics(self, monkeypatch, capsys):
        import trading_bot.cli as cli

        monkeypatch.setattr(cli, "run_backtest", lambda conn, s, **kw: [make_trade(0.01)])
        code = _backtest_command(None, [SYMBOL], start_ms=0, end_ms=DAY_MS)
        out = capsys.readouterr().out
        assert code == 0
        assert "trades=1" in out
        assert "trending/flag" in out
        assert "sharpe=" in out

    def test_walkforward_command_exit_codes(self, monkeypatch, capsys):
        import trading_bot.cli as cli

        def make_result(passed):
            return WalkForwardResult(
                folds=[], final_params=BacktestParams(),
                final_max_hold_bars=config.MAX_HOLD_BARS_TRIGGER,
                oos_start=0, oos_end=1,
                oos_metrics=compute_metrics([make_trade(0.01)] * 5),
                oos_equity={
                    "sharpe": 1.5, "sortino": 1.5, "dsr": 0.99,
                    "max_drawdown_pct": 0.05, "ann_return_pct": 0.1,
                },
                per_symbol_expectancy={SYMBOL: 0.01},
                gate={c: passed for c in walkforward.GATE_CONDITIONS},
                benchmark=BenchmarkResult(
                    per_symbol={SYMBOL: dict.fromkeys(METRIC_KEYS, 0.0)},
                    basket=dict.fromkeys(METRIC_KEYS, 0.0),
                    start_ms=0, end_ms=1,
                ),
                n_trials_used=1,
                passed=passed,
            )

        monkeypatch.setattr(cli, "walk_forward_pooled", lambda conn, syms, **kw: make_result(True))
        assert _walkforward_command(None, [SYMBOL], start_ms=0, end_ms=DAY_MS) == 0
        assert "GATE: PASS" in capsys.readouterr().out

        monkeypatch.setattr(cli, "walk_forward_pooled", lambda conn, syms, **kw: make_result(False))
        assert _walkforward_command(None, [SYMBOL], start_ms=0, end_ms=DAY_MS) == 1
        assert "GATE: FAIL" in capsys.readouterr().out

    def test_walkforward_command_handles_short_span(self, monkeypatch, capsys):
        import trading_bot.cli as cli

        def raise_short(conn, syms, **kw):
            raise ValueError("span too short")

        monkeypatch.setattr(cli, "walk_forward_pooled", raise_short)
        assert _walkforward_command(None, [SYMBOL], start_ms=0, end_ms=DAY_MS) == 1
        assert "ERROR" in capsys.readouterr().out


class TestPhase2CostAndRisk:
    """Mechanics-only checks for the honest cost & risk model (Task 8).

    These assert the risk model's wiring — funding scales with holding time,
    and the realized stop distance actually reflects ATR_STOP_MULTIPLE * ATR
    — not the strategy's profitability, which is a Phase 7 gate concern.
    """

    def test_funding_cost_scales_with_holding_duration(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        # 100 flat meander bars: never touches the (far-below) ATR stop, the
        # trail (ratchets at most to 157.0, still below the meander lows), the
        # opposite channel, or the measured-move target, so both runs below
        # resolve by time-stop only, closing at the identical price —
        # isolating the funding term.
        meander = [[161.3, 161.5, 161.1, 161.3, 10.0]] * 100
        seed_scenario(conn, meander)

        short = run_backtest(conn, SYMBOL, max_hold_bars=1, fee_pct=0.0, slippage_pct=0.0)
        long_ = run_backtest(conn, SYMBOL, max_hold_bars=96, fee_pct=0.0, slippage_pct=0.0)
        assert len(short) == 1
        assert len(long_) == 1
        s, l = short[0], long_[0]
        assert s.outcome == "time"
        assert l.outcome == "time"
        assert s.exit_price == l.exit_price  # identical gross pnl basis

        short_days = (s.exit_ts - s.entry_ts) / 86_400_000.0
        long_days = (l.exit_ts - l.entry_ts) / 86_400_000.0
        assert long_days > short_days  # the 96-bar hold really is longer

        # Only funding differs between the two runs (fee/slippage both zeroed
        # and gross pnl identical), so the pnl_pct gap is exactly the extra
        # funding accrued by the longer hold.
        expected_gap = config.FUNDING_PCT_PER_DAY * (long_days - short_days)
        assert math.isclose(s.pnl_pct - l.pnl_pct, expected_gap, abs_tol=1e-12)
        assert l.pnl_pct < s.pnl_pct  # the longer hold is strictly more negative

    def test_median_realized_stop_distance_meets_atr_floor(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        trades = run_backtest(conn, SYMBOL, fee_pct=0.001, slippage_pct=0.0)
        assert len(trades) == 1

        atr_value = donchian_atr_at_entry()
        risk_distances = [abs(t.entry - t.stop) for t in trades]
        median_risk = statistics.median(risk_distances)
        # Proves R1 landed: the realized stop distance tracks ATR_STOP_MULTIPLE
        # (>= 1.0) * ATR(setup timeframe), not a fixed percentage of entry.
        assert median_risk >= 1.0 * atr_value
        assert config.ATR_STOP_MULTIPLE >= 1.0
        assert math.isclose(median_risk, config.ATR_STOP_MULTIPLE * atr_value)

        # c is measured and reported per symbol (this phase's success signal),
        # not asserted <= COST_RATIO_CEILING — full closure is Phase 4's job.
        median_risk_pct = median_risk / trades[0].entry
        c = cost_ratio(median_risk_pct, config.FEE_PCT, config.SLIPPAGE_PCT)
        assert c > 0 and math.isfinite(c)


class TestPhase47ReviewRepairs:
    """Regressions for .claude/PRPs/reports/code review/phase4-7-code-review.md.

    Each test here pins one repaired defect. They deliberately use PRODUCTION
    defaults (bare BacktestParams()) where the point is what the shipped
    configuration does, rather than the exit-mechanics fixtures above which opt
    into every exit mode to test its arithmetic.
    """

    # --- HIGH-2: the trail and target are off by default and independent ---

    def test_trail_is_off_by_default(self, tmp_path, monkeypatch):
        """Bars that WOULD ratchet a stop and exit on it must not, at defaults.

        Mirrors test_trail_ratchets_and_exits_above_initial_stop, which exits
        "trail" at 202.0 once the trail is enabled.
        """
        trades = donchian_exit_scenario(
            monkeypatch,
            tmp_path,
            exit_lower=0.0,
            exit_upper=300.0,
            params=BacktestParams(),
            outcome_rows_trig=[
                [200, 205.0, 199.0, 204.0, 10.0],  # extreme 205 -> trail would be 202
                [204, 205.0, 201.0, 201.5, 10.0],  # low 201 would trip a 202 trail
            ],
        )
        assert all(t.outcome != "trail" for t in trades)

    def test_target_is_not_an_exit_by_default(self, tmp_path, monkeypatch):
        """Same bar that exits "target" in test_target_exit must not, at defaults."""
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        trades = run_backtest(
            conn, SYMBOL, fee_pct=0.0, slippage_pct=0.0, funding_pct_per_day=0.0
        )
        assert all(t.outcome != "target" for t in trades)

    def test_trail_multiple_is_independent_of_the_entry_stop(self):
        """Sharing k made the trail exactly as tight as the entry stop, which
        closed 91.3% of trades at a median 10-hour hold on a 1D-regime system."""
        assert config.TRAIL_ATR_MULTIPLE != config.ATR_STOP_MULTIPLE
        p = BacktestParams()
        assert p.trail_atr_multiple == config.TRAIL_ATR_MULTIPLE
        assert p.trail_enabled is False
        assert p.target_enabled is False

    def test_target_still_screens_reward_to_risk_when_disabled_as_an_exit(
        self, tmp_path, monkeypatch
    ):
        """target_enabled governs the EXIT only. The R:R screen in build_signal
        must still use the measured move, so a trade is still produced and still
        carries a target and an rr."""
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        trades = run_backtest(
            conn, SYMBOL, fee_pct=0.0, slippage_pct=0.0, funding_pct_per_day=0.0
        )
        assert len(trades) == 1
        assert trades[0].target > trades[0].entry  # recorded, just not an exit

    # --- HIGH-3: the grid sweeps levers that bind ---

    def test_grid_sweeps_exit_modes_not_the_unreachable_rr_floor(self):
        grid = walkforward.DEFAULT_GRID
        # rr_floor was unreachable at its own default: the minimum planned R:R
        # across every trade the engine took was 1.56 > RR_FLOOR = 1.5.
        assert "rr_floor" not in grid
        assert grid["trail_enabled"] == (False, True)
        assert grid["target_enabled"] == (False, True)
        # Every axis must resolve a config default, or _default_combo raises
        # KeyError on the no-combo-reached-min_trades fallback path.
        assert set(walkforward._default_combo(grid)) == set(grid)

    def test_every_grid_axis_reaches_the_engine(self):
        """An axis that BacktestParams cannot accept, and that is not a
        run_backtest kwarg, would be silently ignored — a grid that sweeps
        nothing is exactly the defect this repair addresses."""
        param_fields = {f.name for f in dataclasses.fields(BacktestParams)}
        for axis in walkforward.DEFAULT_GRID:
            assert axis in param_fields or axis == "max_hold_bars"

    # --- MEDIUM-4: the one-shot OOS must run an on-grid configuration ---

    def test_final_params_stay_on_grid_with_an_even_fold_count(self, monkeypatch):
        """statistics.median INTERPOLATES: fold winners [48, 144] gave 96, and
        [1.25, 2.0] gave 1.625 — configurations no fold ever evaluated, used for
        the one-shot out-of-sample run that is the whole point of the protocol.
        """
        grid = {"max_hold_bars": (48, 144)}

        def fold_dependent_fake(
            conn, symbol, *, start_ms=None, end_ms=None, params=None,
            max_hold_bars=None, **kw
        ):
            # Fold 0 (train starts at 0) prefers 48; fold 1 (train starts at
            # 5 days) prefers 144, so the winners straddle and median would
            # interpolate to the off-grid 96.
            prefer = 48 if (start_ms or 0) < 5 * DAY_MS else 144
            exp = 0.02 if max_hold_bars == prefer else 0.01
            n = 12
            span = (end_ms or 0) - (start_ms or 0)
            step = max(1, span // (n + 1))
            return [
                make_trade(exp, exit_ts=(start_ms or 0) + step * (i + 1),
                           entry_ts=(start_ms or 0) + step * (i + 1))
                for i in range(n)
            ]

        monkeypatch.setattr(walkforward, "run_backtest", fold_dependent_fake)
        # conn is None here too, so the SQLite-reading null must be stubbed.
        monkeypatch.setattr(
            walkforward, "buy_and_hold", lambda conn, syms, **kw: _beatable_benchmark()
        )
        result = walk_forward_pooled(
            None, ["BTCUSDT"], start_ms=0, end_ms=25 * DAY_MS, grid=grid,
            train_days=10, test_days=5, oos_days=5, min_trades=12,
        )
        # The interpolation-prone case: an even number of straddling winners.
        assert len(result.folds) == 2
        winners = [f.test_metrics for f in result.folds]
        assert len(winners) == 2
        assert statistics.median([48, 144]) not in grid["max_hold_bars"]  # the old bug
        assert result.final_max_hold_bars in grid["max_hold_bars"]

    # --- MEDIUM-2: no trigger bar in a setup window is unreachable ---

    def test_last_trigger_bar_of_a_setup_window_can_trigger(self):
        """h_idx keys on the trigger bar's OPEN, not its close.

        Keying on the close made the final trigger bar of every setup window
        select the NEXT setup bar, whose end_ts postdates that bar, so
        check_breakout rejected it: 25% of trigger opportunities at a 4H setup /
        1H trigger tier. Reproduces the index arithmetic directly.
        """
        setup_ms, trigger_ms = D_SET, D_TRIG
        per_window = setup_ms // trigger_ms
        close_setup = [START + setup_ms, START + 2 * setup_ms]
        eligible = 0
        # Every trigger bar opening within the first setup window's aftermath.
        for k in range(per_window):
            ts = close_setup[0] + k * trigger_ms
            h_idx = _searchsorted_right(close_setup, ts) - 1
            assert h_idx >= 0
            # check_breakout's guard: the setup bar must have closed by ts.
            if ts >= close_setup[h_idx]:
                eligible += 1
        assert eligible == per_window, "every trigger bar in the window must be usable"


def _searchsorted_right(sorted_values, needle):
    """Mirror of numpy.searchsorted(..., side="right") for the index test."""
    return sum(1 for v in sorted_values if v <= needle)


class TestIndicatorMemo:
    """The indicator memo must be a pure speedup: identical trades, no leakage.

    The walk-forward makes ~870 run_backtest calls that vary only exit
    parameters, so caching the Wilder indicators is what makes the gate runnable
    (LOW-3 in the review). A cache that changed a single trade would be far
    worse than a slow one.
    """

    def _run(self, conn, **kw):
        return run_backtest(
            conn, SYMBOL, fee_pct=0.0, slippage_pct=0.0, funding_pct_per_day=0.0, **kw
        )

    def test_repeated_runs_are_identical(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        cold = self._run(conn)
        warm = self._run(conn)  # served from _CACHE
        assert engine._CACHE, "the memo must actually be populated"
        assert cold == warm

    def test_clear_caches_restores_a_cold_run(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        cold = self._run(conn)
        engine.clear_caches()
        assert engine._CACHE == {}
        assert self._run(conn) == cold

    def test_differing_params_do_not_share_a_cache_entry(self, tmp_path, monkeypatch):
        """target_enabled changes the outcome, so a params-blind cache would
        return the first run's trades for the second."""
        conn = storage.connect(str(tmp_path / "t.db"))
        patch_trending(monkeypatch)
        seed_scenario(conn, [[180, 190.0, 181.0, 189.0, 10.0]])
        without = self._run(conn, params=BacktestParams(target_enabled=False))
        with_target = self._run(conn, params=BacktestParams(target_enabled=True))
        assert [t.outcome for t in without] != [t.outcome for t in with_target]

    def test_same_symbol_different_data_is_not_confused(self, tmp_path, monkeypatch):
        """Two databases, same symbol and timeframes, different prices. Keyed on
        bar count and timestamps alone these would collide; the content
        fingerprint must keep them apart."""
        patch_trending(monkeypatch)
        conn_a = storage.connect(str(tmp_path / "a.db"))
        seed_scenario(conn_a, [[180, 190.0, 181.0, 189.0, 10.0]])
        first = self._run(conn_a, params=BacktestParams(target_enabled=True))

        conn_b = storage.connect(str(tmp_path / "b.db"))
        # Same shape, but the outcome bar never reaches the 182.0 target.
        seed_scenario(conn_b, [[161, 161.5, 160.5, 161.0, 10.0]])
        second = self._run(conn_b, params=BacktestParams(target_enabled=True))

        assert first and second
        assert first[0].outcome == "target"
        assert second[0].outcome != "target"
