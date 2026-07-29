"""Tests for the Phase 4 mean-reversion fade signal method."""

import math

import pandas as pd

from trading_bot import config
from trading_bot.data import storage
from trading_bot.indicators.bollinger import bollinger
from trading_bot.signals import meanrev, scan
from trading_bot.signals.breakout import BreakoutEvent
from trading_bot.signals.meanrev import (
    FadeCandidate,
    build_fade_signal,
    current_fade_signals,
    detect_fade_setups,
)
from trading_bot.risk.atr_stop import net_rr, round_trip_cost_pct
from trading_bot.signals.scan import scan_symbol

SYMBOL = "BTCUSDT"
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME
D_SET = storage.TIMEFRAME_MS[SETUP_TF]
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000  # arbitrary epoch-ms base


def make_df(rows, start=START, interval=D_SET):
    """Build an OHLCV DataFrame from [open, high, low, close, volume] rows."""
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype(int)
    return df.set_index("ts")


def alternating_rows(n, lo=99.5, hi=100.5):
    """n ranging bars alternating between lo and hi closes."""
    rows = []
    for i in range(n):
        close = lo if i % 2 == 0 else hi
        rows.append([close, close + 0.1, close - 0.1, close, 10.0])
    return rows


def long_stretch_df(recovery_bars=1):
    """40 ranging bars, one stretch bar below the lower band, then recovery.

    Stretch bar: close 98.8, low 98.7 (below the ~98.85 lower band at that
    bar). Recovery bars close 99.3 back inside the bands.
    """
    rows = alternating_rows(40)
    rows.append([99.0, 99.0, 98.7, 98.8, 10.0])
    rows += [[99.2, 99.4, 98.9, 99.3, 10.0]] * recovery_bars
    return make_df(rows)


class TestBollinger:
    def test_nan_during_warmup(self):
        df = make_df(alternating_rows(config.BB_PERIOD + 5))
        bands = bollinger(df)
        assert bands["middle"].iloc[: config.BB_PERIOD - 1].isna().all()
        assert not pd.isna(bands["middle"].iloc[-1])

    def test_constant_series_bands_collapse(self):
        df = make_df([[100, 100.1, 99.9, 100.0, 10.0]] * 30)
        bands = bollinger(df)
        assert math.isclose(bands["middle"].iloc[-1], 100.0)
        assert math.isclose(bands["upper"].iloc[-1], 100.0)
        assert math.isclose(bands["lower"].iloc[-1], 100.0)

    def test_alternating_series_known_bands(self):
        # 20-bar window holds 10 each of 99.5/100.5: mean 100, std 0.5 (ddof=0).
        df = make_df(alternating_rows(40))
        bands = bollinger(df)
        assert math.isclose(bands["middle"].iloc[-1], 100.0)
        assert math.isclose(bands["upper"].iloc[-1], 101.0)
        assert math.isclose(bands["lower"].iloc[-1], 99.0)


class TestDetectFadeSetups:
    def test_detects_long_fade_after_lower_stretch(self):
        df = long_stretch_df()
        setups = detect_fade_setups(df)
        assert len(setups) == 1
        c = setups[0]
        assert c.direction == "long"
        assert c.stop_level == 98.7  # excursion extreme low
        bands = bollinger(df)
        assert math.isclose(c.trigger_level, float(bands["lower"].iloc[-1]))
        assert math.isclose(c.target, float(bands["middle"].iloc[-1]))
        assert c.stop_level < c.trigger_level < c.target
        assert c.end_ts == int(df.index[40])  # the stretch bar

    def test_detects_short_fade_after_upper_stretch(self):
        rows = alternating_rows(40)
        rows.append([101.0, 101.3, 101.0, 101.2, 10.0])
        rows.append([100.8, 101.1, 100.6, 100.7, 10.0])
        df = make_df(rows)
        setups = detect_fade_setups(df)
        assert len(setups) == 1
        c = setups[0]
        assert c.direction == "short"
        assert c.stop_level == 101.3  # excursion extreme high
        assert c.stop_level > c.trigger_level > c.target

    def test_no_setup_when_price_inside_bands(self):
        df = make_df(alternating_rows(40))
        assert detect_fade_setups(df) == []

    def test_stale_stretch_rejected(self):
        # Stretch followed by more recovery bars than FADE_STRETCH_MAX_AGE_BARS.
        df = long_stretch_df(recovery_bars=config.FADE_STRETCH_MAX_AGE_BARS + 2)
        assert detect_fade_setups(df) == []

    def test_insufficient_bars_returns_empty(self):
        df = make_df(alternating_rows(config.BB_PERIOD - 1))
        assert detect_fade_setups(df) == []


def make_fade(direction="long", trigger=98.79, stop=98.7, target=99.9):
    return FadeCandidate(
        direction=direction,
        trigger_level=trigger,
        stop_level=stop,
        target=target,
        start_ts=START,
        end_ts=START,
    )


def make_event(price, direction="long", level=98.79):
    return BreakoutEvent(
        ts=START + 10 * D_TRIG,
        price=price,
        level=level,
        direction=direction,
        volume_ratio=2.0,
        volume_high=True,
    )


class TestBuildFadeSignal:
    def test_long_fade_passes_band(self):
        signal = build_fade_signal(SYMBOL, make_fade(), make_event(98.95))
        assert signal is not None
        assert signal.pattern == "bollinger-fade"
        assert signal.direction == "long"
        assert signal.stop == 98.7
        assert signal.target == 99.9
        assert signal.rr >= config.RR_FLOOR
        assert math.isclose(signal.rr, signal.reward_pct / signal.risk_pct)

    def test_net_rr_exactly_at_floor_is_accepted(self):
        # The gate is `net_rr(...) < rr_floor`, so a setup landing exactly ON
        # the floor must be accepted (>= semantics). No tidy geometry puts the
        # NET ratio on a round number — the cost term (0.0014) is not a binary
        # fraction — so instead of solving for one, pass the floor this
        # geometry's net ratio lands on exactly. That tests the comparison at
        # true equality with zero float slop, in both directions: one
        # representable tick above the same value must be rejected.
        entry, stop, target = 99.0, 98.5, 99.75
        fade = make_fade(stop=stop, target=target)
        event = make_event(entry)
        exact = net_rr(
            (target - entry) / entry,
            (entry - stop) / entry,
            config.FEE_PCT,
            config.SLIPPAGE_PCT,
        )
        assert build_fade_signal(SYMBOL, fade, event, rr_floor=exact) is not None
        assert (
            build_fade_signal(
                SYMBOL, fade, event, rr_floor=math.nextafter(exact, math.inf)
            )
            is None
        )

    def test_gross_rr_passes_but_reward_under_cost_rejected(self):
        # The case approach B exists to catch. entry 100.00 / stop 99.95 /
        # target 100.10 gives risk 0.05% and reward 0.10%, so the GROSS ratio
        # is 2.00 — comfortably over RR_FLOOR (1.5), and the gross gate
        # accepted it. But round-trip cost is 2*(0.05% + 0.02%) = 0.14%, more
        # than the entire 0.10% reward: hitting the target exactly still loses
        # ~0.04%. There is no outcome in which this trade makes money, and the
        # dimensionless gross ratio cannot see that because it is scale-free.
        entry, stop, target = 100.0, 99.95, 100.10
        risk_pct = (entry - stop) / entry
        reward_pct = (target - entry) / entry
        assert reward_pct / risk_pct > config.RR_FLOOR  # old gate would pass it
        assert reward_pct < round_trip_cost_pct(config.FEE_PCT, config.SLIPPAGE_PCT)
        fade = make_fade(trigger=100.04, stop=stop, target=target)
        assert build_fade_signal(SYMBOL, fade, make_event(entry, level=100.04)) is None

    def test_wider_stop_drops_rr_below_floor_rejected(self):
        # Stop 98.2 widens risk to ~0.758% (vs ~0.253% at the default stop
        # 98.7) while reward stays ~0.96%, dropping rr to ~1.27 < RR_FLOOR
        # (1.5). There is no separate risk-percentage band anymore (Phase 2
        # removed MAX_RISK_PCT as a filter) — this is rejected by the same
        # single rr >= RR_FLOOR screen as every other rejection here.
        assert build_fade_signal(SYMBOL, make_fade(stop=98.2), make_event(98.95)) is None

    def test_thinner_reward_drops_rr_below_floor_rejected(self):
        # The reward-side counterpart to the test above: target 99.3 shrinks
        # reward to ~0.35% while risk stays ~0.253% at the default stop, giving
        # rr ~1.38 < RR_FLOOR (1.5). No absolute reward floor exists anymore
        # (Phase 2 removed MIN_REWARD_PCT); this is the same single
        # rr >= RR_FLOOR screen, exercised by shrinking the numerator rather
        # than growing the denominator.
        assert build_fade_signal(SYMBOL, make_fade(target=99.3), make_event(98.95)) is None

    def test_entry_beyond_target_rejected(self):
        fade = make_fade(stop=99.6, target=99.9)
        assert build_fade_signal(SYMBOL, fade, make_event(99.95)) is None

    def test_entry_beyond_stop_rejected(self):
        assert build_fade_signal(SYMBOL, make_fade(), make_event(98.65)) is None

    def test_short_fade_passes_band(self):
        fade = make_fade(direction="short", trigger=101.21, stop=101.3, target=100.1)
        event = make_event(101.05, direction="short", level=101.21)
        signal = build_fade_signal(SYMBOL, fade, event)
        assert signal is not None
        assert signal.direction == "short"
        assert signal.stop == 101.3


def seed_candles(conn, symbol, timeframe, df):
    rows = [
        [int(ts), r["open"], r["high"], r["low"], r["close"], r["volume"]]
        for ts, r in df.iterrows()
    ]
    storage.upsert_candles(conn, symbol, timeframe, rows)


class TestCurrentFadeSignals:
    def test_empty_db_is_uncertain_with_no_signals(self, tmp_path):
        conn = storage.connect(str(tmp_path / "test.db"))
        regime, signals = current_fade_signals(conn, SYMBOL, now_ms=START)
        assert regime == "uncertain"
        assert signals == []

    def test_non_ranging_regime_gates_signals(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "test.db"))
        monkeypatch.setattr(
            meanrev, "current_regime", lambda *a, **k: ("trending", 30.0, 0.5)
        )
        regime, signals = current_fade_signals(conn, SYMBOL, now_ms=START)
        assert regime == "trending"
        assert signals == []

    def test_ranging_stretch_recross_produces_signal(self, tmp_path, monkeypatch):
        conn = storage.connect(str(tmp_path / "test.db"))
        monkeypatch.setattr(
            meanrev, "current_regime", lambda *a, **k: ("ranging", 15.0, 0.4)
        )

        df_setup = long_stretch_df()  # long fade: stop 98.7, trigger ~98.79
        seed_candles(conn, SYMBOL, SETUP_TF, df_setup)
        trigger = detect_fade_setups(df_setup)[0].trigger_level

        # Trigger series ending after the last setup bar: fresh re-cross of the band.
        last_setup_ts = int(df_setup.index[-1])
        rows_trig = [[98.75, 98.78, 98.72, 98.75, 10.0]] * 21
        rows_trig.append([98.76, 99.0, 98.74, 98.95, 30.0])
        df_trig = make_df(rows_trig, start=last_setup_ts - 17 * D_TRIG, interval=D_TRIG)
        seed_candles(conn, SYMBOL, TRIGGER_TF, df_trig)
        assert rows_trig[-2][3] <= trigger < rows_trig[-1][3]  # fresh cross holds

        now_ms = int(df_trig.index[-1]) + D_TRIG
        regime, signals = current_fade_signals(conn, SYMBOL, now_ms=now_ms)
        assert regime == "ranging"
        assert len(signals) == 1
        s = signals[0]
        assert s.pattern == "bollinger-fade"
        assert s.direction == "long"
        assert s.entry == 98.95
        assert s.stop == 98.7
        assert s.volume_high is True
        assert s.rr >= config.RR_FLOOR

    def test_fade_path_passes_the_trigger_interval_to_check_breakout(
        self, tmp_path, monkeypatch
    ):
        """The fade path must pass interval_ms, matching the breakout path.

        check_breakout only enforces its fresh-crossing contiguity rule when
        given interval_ms; without it, a non-adjacent pair reads as a fresh
        re-cross off a stale reference close. That hazard grows at coarser tiers
        (a gap is now hours, not minutes).

        This asserts the argument is actually passed rather than seeding a
        gapped series, because _load_df already trims to a contiguous tail
        (setup.py::_contiguous_tail) — so a gapped fixture can never reach
        check_breakout through this path, and a test built that way would pass
        whether or not interval_ms was supplied. The interval_ms here is
        defense-in-depth against a caller that bypasses the trim, and it is the
        pass-through itself that needs pinning.
        """
        conn = storage.connect(str(tmp_path / "test.db"))
        monkeypatch.setattr(
            meanrev, "current_regime", lambda *a, **k: ("ranging", 15.0, 0.4)
        )
        seen = []
        real = meanrev.check_breakout

        def spy(df, candidate, **kw):
            seen.append(kw)
            return real(df, candidate, **kw)

        monkeypatch.setattr(meanrev, "check_breakout", spy)

        df_setup = long_stretch_df()
        seed_candles(conn, SYMBOL, SETUP_TF, df_setup)
        last_setup_ts = int(df_setup.index[-1])
        rows_trig = [[98.75, 98.78, 98.72, 98.75, 10.0]] * 21
        rows_trig.append([98.76, 99.0, 98.74, 98.95, 30.0])
        df_trig = make_df(rows_trig, start=last_setup_ts - 17 * D_TRIG, interval=D_TRIG)
        seed_candles(conn, SYMBOL, TRIGGER_TF, df_trig)

        current_fade_signals(conn, SYMBOL, now_ms=int(df_trig.index[-1]) + D_TRIG)
        assert seen, "check_breakout was never called; fixture no longer exercises the path"
        assert all(kw.get("interval_ms") == D_TRIG for kw in seen), seen


class TestScanSymbol:
    def _patch(self, monkeypatch, regime):
        monkeypatch.setattr(
            scan, "current_regime", lambda *a, **k: (regime, 20.0, 0.5)
        )
        monkeypatch.setattr(
            scan, "scan_donchian_signals", lambda conn, sym, now: ["donchian-sentinel"]
        )
        monkeypatch.setattr(
            scan, "scan_fade_signals", lambda conn, sym, now: ["fade-sentinel"]
        )

    def test_trending_dispatches_to_donchian_method(self, monkeypatch):
        self._patch(monkeypatch, "trending")
        assert scan_symbol(None, SYMBOL, now_ms=START) == ("trending", ["donchian-sentinel"])

    def test_ranging_dispatches_to_fade_method(self, monkeypatch):
        # This class tests ROUTING, so the Phase 6 kill switch is forced on:
        # the sleeve ships disabled, but the ranging branch must still route to
        # the fade method when it is enabled. Suppression is covered by
        # TestFadeEnabledSwitch.
        self._patch(monkeypatch, "ranging")
        monkeypatch.setattr(config, "FADE_ENABLED", True)
        assert scan_symbol(None, SYMBOL, now_ms=START) == ("ranging", ["fade-sentinel"])

    def test_extreme_volatility_suppresses_all_methods(self, monkeypatch):
        self._patch(monkeypatch, "extreme-volatility")
        assert scan_symbol(None, SYMBOL, now_ms=START) == ("extreme-volatility", [])

    def test_uncertain_suppresses_all_methods(self, monkeypatch):
        self._patch(monkeypatch, "uncertain")
        assert scan_symbol(None, SYMBOL, now_ms=START) == ("uncertain", [])


class TestFadeEnabledSwitch:
    """Phase 6 kill switch on the LIVE dispatch path.

    config.FADE_ENABLED must be read at call time, which is what makes
    monkeypatch.setattr(config, ...) bite. If the guard in scan.py is reverted,
    test_disabled_suppresses_fade_but_reports_regime fails.
    """

    def _patch(self, monkeypatch, enabled):
        monkeypatch.setattr(
            scan, "current_regime", lambda *a, **k: ("ranging", 20.0, 0.5)
        )
        monkeypatch.setattr(
            scan, "scan_fade_signals", lambda conn, sym, now: ["fade-sentinel"]
        )
        monkeypatch.setattr(config, "FADE_ENABLED", enabled)

    def test_enabled_dispatches_to_fade(self, monkeypatch):
        self._patch(monkeypatch, True)
        assert scan_symbol(None, SYMBOL, now_ms=START) == ("ranging", ["fade-sentinel"])

    def test_disabled_suppresses_fade_but_reports_regime(self, monkeypatch):
        self._patch(monkeypatch, False)
        # Label stays truthful: suppression is not a misclassification.
        assert scan_symbol(None, SYMBOL, now_ms=START) == ("ranging", [])

    def test_disabled_leaves_trending_path_untouched(self, monkeypatch):
        monkeypatch.setattr(
            scan, "current_regime", lambda *a, **k: ("trending", 30.0, 0.5)
        )
        monkeypatch.setattr(
            scan, "scan_donchian_signals", lambda conn, sym, now: ["donchian-sentinel"]
        )
        monkeypatch.setattr(config, "FADE_ENABLED", False)
        assert scan_symbol(None, SYMBOL, now_ms=START) == (
            "trending",
            ["donchian-sentinel"],
        )

    def test_ships_disabled_per_the_phase_6_verdict(self):
        """The switch's VALUE is the Phase 6 decision, not a default.

        DROPPED per .claude/PRPs/reports/fade-requalification.md (2026-07-27):
        pooled expectancy -0.3883% (n=297), 0 of 3 symbols positive, cost ratio
        above the ceiling on all three. If this assertion is ever flipped back,
        it must be because a NEW measurement re-qualified the sleeve and the
        report records it — not because a test was inconvenient.
        """
        assert config.FADE_ENABLED is False
