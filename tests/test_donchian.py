"""Tests for the Phase 5 Donchian trend engine (indicator + signal method)."""

import math

import pandas as pd

from trading_bot import config
from trading_bot.data import storage
from trading_bot.indicators.donchian import donchian
from trading_bot.signals import donchian as donchian_mod
from trading_bot.signals.donchian import (
    DONCHIAN_KIND,
    detect_donchian_setups,
    scan_donchian_signals,
)

SYMBOL = "BTCUSDT"
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
TRIG_TF = config.SIGNAL_TRIGGER_TIMEFRAME
SETUP_MS = storage.TIMEFRAME_MS[SETUP_TF]
TRIG_MS = storage.TIMEFRAME_MS[TRIG_TF]
START = 1_700_000_000_000


def make_df(rows, start=START, interval=SETUP_MS):
    """Build an OHLCV DataFrame from [open, high, low, close, volume] rows."""
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype(int)
    return df.set_index("ts")


def seed_candles(conn, symbol, timeframe, df):
    rows = [
        [int(ts), r["open"], r["high"], r["low"], r["close"], r["volume"]]
        for ts, r in df.iterrows()
    ]
    storage.upsert_candles(conn, symbol, timeframe, rows)


def uptrend_rows(n):
    """Monotone ramp: high/low/close all strictly rising by 1 per bar.

    ADX(14) climbs well past 25 after warmup; the last bar closes at the
    trailing 20-bar upper channel and above the 55-bar mid.
    """
    return [[100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 10.0] for i in range(n)]


def downtrend_rows(n):
    """Mirror of uptrend_rows: a monotone decline."""
    return [[100.0 - i, 99.0 - i, 96.0 - i, 97.0 - i, 10.0] for i in range(n)]


def choppy_rows(n):
    """Oscillating series with no persistent direction: ADX stays low."""
    rows = []
    for i in range(n):
        base = 100.0
        close = base + (1.0 if i % 2 == 0 else -1.0)
        rows.append([base, close + 0.5, close - 0.5, close, 10.0])
    return rows


class TestDonchianIndicator:
    def test_leading_nans_until_period_bars(self):
        df = make_df(uptrend_rows(30))
        ch = donchian(df, period=20)
        for i in range(20):
            assert pd.isna(ch["upper"].iloc[i])
            assert pd.isna(ch["lower"].iloc[i])
            assert pd.isna(ch["mid"].iloc[i])
        assert pd.notna(ch["upper"].iloc[20])
        assert pd.notna(ch["lower"].iloc[20])
        assert pd.notna(ch["mid"].iloc[20])

    def test_current_bar_excluded_from_its_own_channel(self):
        rows = [[100, 100, 100, 100, 10.0]] * 20
        rows.append([130, 130, 130, 130, 10.0])
        df = make_df(rows)
        ch = donchian(df, period=20)
        assert ch["upper"].iloc[-1] == 100.0

    def test_upper_lower_mid_arithmetic(self):
        df = make_df(uptrend_rows(30))
        ch = donchian(df, period=20)
        defined = ch.dropna()
        assert len(defined) > 0
        for upper, lower, mid in zip(defined["upper"], defined["lower"], defined["mid"]):
            assert math.isclose(mid, (upper + lower) / 2.0)

    def test_trailing_only_matches_incremental(self):
        df = make_df(uptrend_rows(40))
        full = donchian(df, period=20)
        for i in range(20, len(df)):
            partial = donchian(df.iloc[: i + 1], period=20)
            assert partial["upper"].iloc[-1] == full["upper"].iloc[i]
            assert partial["lower"].iloc[-1] == full["lower"].iloc[i]
            assert partial["mid"].iloc[-1] == full["mid"].iloc[i]

    def test_short_series_all_nan(self):
        df = make_df(uptrend_rows(15))
        ch = donchian(df, period=20)
        assert ch["upper"].isna().all()
        assert ch["lower"].isna().all()
        assert ch["mid"].isna().all()


class TestDetectDonchianSetups:
    def test_long_candidate_shape(self):
        rows = uptrend_rows(60)
        df = make_df(rows)
        cands = detect_donchian_setups(df)
        assert len(cands) == 1
        c = cands[0]
        assert c.kind == DONCHIAN_KIND
        assert c.direction == "long"
        expected_upper = donchian(df, period=20)["upper"].iloc[-1]
        expected_width = expected_upper - donchian(df, period=20)["lower"].iloc[-1]
        assert math.isclose(c.breakout_level, expected_upper)
        assert math.isclose(c.target_height, expected_width)
        assert c.end_ts == int(df.index[-1]) + SETUP_MS

    def test_short_candidate_on_downtrend(self):
        rows = downtrend_rows(60)
        df = make_df(rows)
        cands = detect_donchian_setups(df)
        assert len(cands) == 1
        c = cands[0]
        assert c.kind == DONCHIAN_KIND
        assert c.direction == "short"
        expected_lower = donchian(df, period=20)["lower"].iloc[-1]
        expected_width = donchian(df, period=20)["upper"].iloc[-1] - expected_lower
        assert math.isclose(c.breakout_level, expected_lower)
        assert math.isclose(c.target_height, expected_width)
        assert c.end_ts == int(df.index[-1]) + SETUP_MS

    def test_no_candidate_below_adx_floor(self):
        df = make_df(choppy_rows(60))
        cands = detect_donchian_setups(df)
        assert cands == []

    def test_no_candidate_on_the_mid_line(self):
        """Isolates the 55-mid filter (adx_min=0.0): a close exactly on the
        55-bar mid-line has no side, so no candidate is emitted even though
        ADX and the channel are both well-defined.
        """
        rows = downtrend_rows(60)
        df = make_df(rows)
        mid = donchian(df, period=55)["mid"].iloc[-1]
        rows[-1][3] = mid  # force this bar's close onto the mid-line exactly
        df = make_df(rows)
        cands = detect_donchian_setups(df, adx_min=0.0)
        assert cands == []

    def test_warmup_returns_empty(self):
        df = make_df(uptrend_rows(40))
        assert detect_donchian_setups(df) == []

    def test_no_geometry_params_referenced(self):
        import inspect

        src = inspect.getsource(donchian_mod)
        assert "TRIANGLE" not in src
        assert "FLAG_" not in src
        assert "HS_" not in src


class TestScanDonchianSignals:
    def _seed(self, conn):
        rows_setup = uptrend_rows(60)
        df_setup = make_df(rows_setup)
        seed_candles(conn, SYMBOL, SETUP_TF, df_setup)
        last_setup_ts = int(df_setup.index[-1])

        candidate = detect_donchian_setups(df_setup)[0]
        level = candidate.breakout_level
        below = level - 1.0
        entry = level + 1.0
        rows_trig = [[below, below + 0.1, below - 0.1, below, 10.0]] * 21
        rows_trig.append([below, entry + 0.1, below - 0.1, entry, 30.0])
        df_trig = make_df(
            rows_trig, start=last_setup_ts + SETUP_MS, interval=TRIG_MS
        )
        seed_candles(conn, SYMBOL, TRIG_TF, df_trig)
        return df_setup, df_trig, candidate

    def test_scan_produces_signal(self, tmp_path):
        conn = storage.connect(str(tmp_path / "t.db"))
        df_setup, df_trig, candidate = self._seed(conn)
        now_ms = int(df_trig.index[-1]) + TRIG_MS

        signals = scan_donchian_signals(conn, SYMBOL, now_ms)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.pattern == DONCHIAN_KIND
        assert sig.entry == float(df_trig["close"].iloc[-1])
        assert sig.rr >= config.RR_FLOOR

    def test_empty_db_returns_empty(self, tmp_path):
        conn = storage.connect(str(tmp_path / "t.db"))
        assert scan_donchian_signals(conn, SYMBOL, START) == []

    def test_warns_and_returns_empty_when_atr_undefined(self, tmp_path, caplog):
        conn = storage.connect(str(tmp_path / "t.db"))
        # Fewer bars than ATR_STOP_PERIOD (14): ATR is NaN (warmup), so the
        # scan must warn and bail before ever calling detect_donchian_setups.
        rows_setup = uptrend_rows(10)
        df_setup = make_df(rows_setup)
        seed_candles(conn, SYMBOL, SETUP_TF, df_setup)
        last_setup_ts = int(df_setup.index[-1])
        rows_trig = [[100, 100.5, 99.5, 100, 10.0]] * 22
        df_trig = make_df(rows_trig, start=last_setup_ts + SETUP_MS, interval=TRIG_MS)
        seed_candles(conn, SYMBOL, TRIG_TF, df_trig)
        now_ms = int(df_trig.index[-1]) + TRIG_MS

        with caplog.at_level("WARNING", logger="trading_bot"):
            signals = scan_donchian_signals(conn, SYMBOL, now_ms)
        assert signals == []
        assert "ATR undefined" in caplog.text
