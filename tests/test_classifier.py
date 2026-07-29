"""Tests for market regime classification."""

import math
import time

import pandas as pd

from trading_bot import config
from trading_bot.cli import _regime_command
from trading_bot.data import storage
from trading_bot.indicators import wilder
from trading_bot.regime import classifier

# Test fixtures
SYMBOL = "BTCUSDT"
TF = config.REGIME_TIMEFRAME  # follows the tier config, never hardcoded
INTERVAL = storage.TIMEFRAME_MS[TF]
START = 1_700_000_000_000  # arbitrary epoch-ms base


def make_conn(tmp_path):
    return storage.connect(str(tmp_path / "test.db"))


def make_ohlcv_rows(
    n, start=START, interval=INTERVAL, close=100.0, high_offset=2.0, low_offset=2.0
):
    """Generate n OHLCV rows as [ts, open, high, low, close, volume]."""
    return [
        [start + i * interval, close, close + high_offset, close - low_offset, close, 10.0]
        for i in range(n)
    ]


def make_dataframe(rows):
    """Convert OHLCV rows to a DataFrame with proper column names and index."""
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype(int)
    df = df.set_index("ts")
    return df


class TestClassifySeries:
    """Tests for classify_series() function."""

    def test_insufficient_data_all_uncertain(self):
        """Series with < REGIME_MIN_BARS bars should all be 'uncertain'."""
        n = config.REGIME_MIN_BARS - 5
        rows = make_ohlcv_rows(n)
        df = make_dataframe(rows)

        labels = classifier.classify_series(df)
        assert len(labels) == n
        assert all(label == "uncertain" for label in labels)

    def test_warmup_period_marked_uncertain(self):
        """First REGIME_MIN_BARS bars should be 'uncertain', even in a long series."""
        n = config.REGIME_MIN_BARS + 50
        rows = make_ohlcv_rows(n)
        df = make_dataframe(rows)

        labels = classifier.classify_series(df)
        # First REGIME_MIN_BARS should be uncertain
        assert all(label == "uncertain" for label in labels[: config.REGIME_MIN_BARS])
        # Some later bars should not be uncertain (they may trend or range)
        assert not all(label == "uncertain" for label in labels[config.REGIME_MIN_BARS :])

    def test_monotonic_uptrend_classified_trending(self):
        """Monotonic uptrend with rising close should classify as trending."""
        n = config.REGIME_MIN_BARS + 50
        rows = make_ohlcv_rows(n)
        # Make close monotonically increasing
        for i in range(n):
            rows[i][4] = 100.0 + i * 0.5  # close increases by 0.5 each bar
            rows[i][1] = 100.0 + i * 0.5  # open same as close (easier ADX computation)
            rows[i][2] = rows[i][4] + 2.0  # high
            rows[i][3] = rows[i][4] - 2.0  # low

        df = make_dataframe(rows)
        labels = classifier.classify_series(df)

        # After warmup, most bars should be trending (ADX should rise for a trend)
        post_warmup_labels = labels[config.REGIME_MIN_BARS :]
        trending_count = sum(1 for label in post_warmup_labels if label == "trending")
        # With a strong uptrend, many should be trending
        assert trending_count > len(post_warmup_labels) * 0.5

    def test_oscillating_range_classified_ranging(self):
        """Oscillating close around a mean should classify as ranging."""
        n = config.REGIME_MIN_BARS + 50
        rows = make_ohlcv_rows(n)
        # Make close oscillate around 100.0
        for i in range(n):
            # Oscillate between 98 and 102
            close_val = 100.0 + 2.0 * math.sin(2 * math.pi * i / 20)
            rows[i][4] = close_val
            rows[i][1] = close_val
            rows[i][2] = close_val + 2.0
            rows[i][3] = close_val - 2.0

        df = make_dataframe(rows)
        labels = classifier.classify_series(df)

        # After warmup, most bars should be ranging (low ADX in oscillation)
        post_warmup_labels = labels[config.REGIME_MIN_BARS :]
        ranging_count = sum(1 for label in post_warmup_labels if label == "ranging")
        # With a range, most should be ranging
        assert ranging_count > len(post_warmup_labels) * 0.5

    def test_extreme_volatility_override_with_spike(self):
        """ATR spike should produce 'extreme-volatility' labels, overriding trending."""
        # Need enough bars for warmup (and enough past warmup for spike to be classified)
        # Use a smaller window to allow for shorter series
        window = 30
        n = config.REGIME_MIN_BARS + window + 10
        rows = make_ohlcv_rows(n)

        # Create an uptrend
        for i in range(n):
            rows[i][4] = 100.0 + i * 0.3  # close increases
            rows[i][1] = 100.0 + i * 0.3
            rows[i][2] = rows[i][4] + 1.0
            rows[i][3] = rows[i][4] - 1.0

        # Inject a volatility spike after warmup
        spike_idx = config.REGIME_MIN_BARS + 5
        if spike_idx < n:
            # Create a large gap (spike in ATR)
            rows[spike_idx][2] = rows[spike_idx][4] + 20.0  # high spike
            rows[spike_idx][3] = rows[spike_idx][4] - 20.0  # low spike

        df = make_dataframe(rows)
        labels = classifier.classify_series(
            df, atr_percentile_window=window, atr_extreme_percentile=0.85
        )

        # The spike bar should be extreme-volatility
        if spike_idx < n:
            spike_label = labels.iloc[spike_idx]
            assert spike_label == "extreme-volatility"

    def test_custom_thresholds_used(self):
        """Passing custom thresholds should affect classification."""
        n = config.REGIME_MIN_BARS + 30
        rows = make_ohlcv_rows(n)
        # Monotonic uptrend
        for i in range(n):
            rows[i][4] = 100.0 + i * 0.5
            rows[i][1] = 100.0 + i * 0.5
            rows[i][2] = rows[i][4] + 2.0
            rows[i][3] = rows[i][4] - 2.0

        df = make_dataframe(rows)

        # With a low threshold, most post-warmup bars should be trending
        labels_low_threshold = classifier.classify_series(
            df, adx_trend_threshold=5.0
        )
        post_warmup_low = labels_low_threshold[config.REGIME_MIN_BARS :]
        trending_count_low = sum(1 for label in post_warmup_low if label == "trending")

        # With a higher threshold (default 25), fewer bars should be trending
        labels_default_threshold = classifier.classify_series(df)
        post_warmup_default = labels_default_threshold[config.REGIME_MIN_BARS :]
        trending_count_default = sum(1 for label in post_warmup_default if label == "trending")

        # Low threshold should have at least as many trending bars as default
        assert trending_count_low >= trending_count_default


class TestCurrentRegime:
    """Tests for current_regime() function."""

    def test_current_regime_empty_db(self, tmp_path):
        """Empty DB should return ('uncertain', nan, nan)."""
        conn = make_conn(tmp_path)
        label, adx, atr_pct = classifier.current_regime(conn, SYMBOL)
        assert label == "uncertain"
        assert pd.isna(adx)
        assert pd.isna(atr_pct)

    def test_current_regime_insufficient_bars(self, tmp_path):
        """DB with < REGIME_MIN_BARS bars should return ('uncertain', nan, nan)."""
        conn = make_conn(tmp_path)
        rows = make_ohlcv_rows(config.REGIME_MIN_BARS - 10)
        storage.upsert_candles(conn, SYMBOL, TF, rows)

        label, adx, atr_pct = classifier.current_regime(conn, SYMBOL)
        assert label == "uncertain"

    def test_current_regime_excludes_open_candle(self, tmp_path):
        """current_regime should use the last CLOSED bar, not the open one."""
        conn = make_conn(tmp_path)
        # Create enough bars for classification
        n = config.REGIME_MIN_BARS + 10
        rows = make_ohlcv_rows(n)
        storage.upsert_candles(conn, SYMBOL, TF, rows)

        # Make the final (still-open) bar a violent spike: if it leaked into the
        # computation, the ADX for the "latest closed bar" would shift
        last_ts = rows[-1][0]
        rows[-1] = [last_ts, 100.0, 1000.0, 10.0, 500.0, 10.0]
        storage.upsert_candles(conn, SYMBOL, TF, [rows[-1]])

        # now_ms halfway through the last bar, so that bar is still open
        now_ms = last_ts + INTERVAL // 2

        label, adx, atr_pct = classifier.current_regime(conn, SYMBOL, now_ms=now_ms)

        # Expected values come from the closed bars only (all rows but the last)
        closed_df = make_dataframe(rows[:-1])
        expected_adx = wilder.adx(closed_df, period=config.ADX_PERIOD).iloc[-1]
        expected_labels = classifier.classify_series(closed_df)

        assert label == expected_labels.iloc[-1]
        assert math.isclose(adx, expected_adx, abs_tol=1e-9) or (
            math.isnan(adx) and math.isnan(expected_adx)
        )

    def test_current_regime_returns_valid_values(self, tmp_path):
        """current_regime should return valid ADX and ATR percentile values."""
        conn = make_conn(tmp_path)
        n = config.REGIME_MIN_BARS + 20
        rows = make_ohlcv_rows(n)
        # Make an uptrend
        for i in range(n):
            rows[i][4] = 100.0 + i * 0.5
            rows[i][1] = 100.0 + i * 0.5
            rows[i][2] = rows[i][4] + 2.0
            rows[i][3] = rows[i][4] - 2.0

        storage.upsert_candles(conn, SYMBOL, TF, rows)

        # Use now_ms just after the last closed bar
        last_ts = rows[-1][0]
        now_ms = last_ts + INTERVAL

        label, adx, atr_pct = classifier.current_regime(conn, SYMBOL, now_ms=now_ms)

        # Should return valid values (not NaN) for a trending series
        assert not pd.isna(adx) or label == "uncertain"
        assert not pd.isna(atr_pct) or label == "uncertain"


class TestRegimeCommand:
    """Tests for _regime_command() CLI helper."""

    def test_regime_command_empty_db_returns_1(self, tmp_path, capsys):
        """_regime_command should return 1 if DB is empty (all uncertain)."""
        conn = make_conn(tmp_path)
        symbols = [SYMBOL]

        exit_code = _regime_command(conn, symbols)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Symbol" in captured.out
        assert SYMBOL in captured.out
        assert "uncertain" in captured.out

    def test_regime_command_prints_table(self, tmp_path, capsys):
        """_regime_command should print an aligned table with Symbol, ADX, ATR%ile, Regime."""
        conn = make_conn(tmp_path)
        # Create sufficient data for classification
        n = config.REGIME_MIN_BARS + 20
        rows = make_ohlcv_rows(n)
        storage.upsert_candles(conn, SYMBOL, TF, rows)

        symbols = [SYMBOL]
        now_ms = rows[-1][0] + INTERVAL

        exit_code = _regime_command(conn, symbols, now_ms=now_ms)

        captured = capsys.readouterr()
        # Should have header row
        assert "Symbol" in captured.out
        assert "ADX" in captured.out
        assert "ATR%ile" in captured.out
        assert "Regime" in captured.out
        # Should have symbol name
        assert SYMBOL in captured.out
        # Should have one of the valid regimes
        assert any(regime in captured.out for regime in classifier.REGIMES)

    def test_regime_command_multiple_symbols(self, tmp_path, capsys):
        """_regime_command should handle multiple symbols."""
        conn = make_conn(tmp_path)
        symbols = ["BTCUSDT", "ETHUSDT"]

        # Seed data for both
        n = config.REGIME_MIN_BARS + 15
        for symbol in symbols:
            rows = make_ohlcv_rows(n)
            storage.upsert_candles(conn, symbol, TF, rows)

        exit_code = _regime_command(conn, symbols)

        captured = capsys.readouterr()
        # Both symbols should appear in output
        for symbol in symbols:
            assert symbol in captured.out

    def test_regime_command_valid_data_returns_0(self, tmp_path):
        """_regime_command should return 0 if all symbols have valid (non-uncertain) regimes."""
        conn = make_conn(tmp_path)
        symbols = [SYMBOL]

        # Create data with strong trend (should classify as trending, not uncertain)
        n = config.REGIME_MIN_BARS + 40
        rows = make_ohlcv_rows(n)
        for i in range(n):
            rows[i][4] = 100.0 + i * 0.5  # uptrend
            rows[i][1] = 100.0 + i * 0.5
            rows[i][2] = rows[i][4] + 2.0
            rows[i][3] = rows[i][4] - 2.0

        storage.upsert_candles(conn, SYMBOL, TF, rows)

        now_ms = rows[-1][0] + INTERVAL

        exit_code = _regime_command(conn, symbols, now_ms=now_ms)

        # With a strong trend past warmup, should get 0 (all symbols classified)
        assert exit_code == 0


class TestRelativeAtrPercentile:
    """Regression for HIGH-1 in .claude/PRPs/reports/code review/phase4-7-code-review.md.

    The extreme-volatility gate must rank ATR as a FRACTION OF CLOSE. Ranking
    raw ATR — denominated in quote currency — meant that in any trending window
    absolute ATR sat near its own trailing maximum simply because price had
    risen, so a steady trend was labelled extreme-volatility indefinitely and
    all trading was suppressed exactly when the trend method should have run.
    """

    @staticmethod
    def _steady_growth_rows(n=400):
        """Price compounding 0.4%/bar with CONSTANT relative range (TR/close).

        Absolute TR grows with price; relative TR does not. A correct gate sees
        flat volatility here; the raw-ATR gate sees a new maximum on every bar.
        """
        rows = []
        close = 100.0
        for i in range(n):
            rows.append(
                [START + i * INTERVAL, close, close * 1.01, close * 0.99, close, 10.0]
            )
            close *= 1.004
        return rows

    def test_steady_relative_volatility_is_not_extreme(self):
        df = make_dataframe(self._steady_growth_rows())
        labels = classifier.classify_series(df)
        post_warmup = labels.iloc[config.REGIME_MIN_BARS :]
        assert "extreme-volatility" not in set(post_warmup)
        # A relentless uptrend is what ADX is for; the label must survive to it.
        assert labels.iloc[-1] == "trending"

    def test_relative_atr_divides_by_close(self):
        df = make_dataframe(self._steady_growth_rows(n=60))
        raw = wilder.atr(df, period=config.ADX_PERIOD)
        rel = classifier._relative_atr(df, config.ADX_PERIOD)
        assert rel.iloc[-1] == raw.iloc[-1] / df["close"].iloc[-1]
        # Raw ATR grows with the price level; the relative series does not.
        assert raw.iloc[-1] > raw.iloc[config.ADX_PERIOD]
        assert rel.iloc[-1] < raw.iloc[-1]

    def test_current_regime_percentile_matches_classify_series(self, tmp_path):
        """current_regime's reported atr_percentile must be computed the same
        way as the label it ships with — both on relative ATR."""
        conn = make_conn(tmp_path)
        rows = self._steady_growth_rows()
        storage.upsert_candles(conn, SYMBOL, TF, rows)
        now_ms = rows[-1][0] + 2 * INTERVAL
        label, _, pct = classifier.current_regime(conn, SYMBOL, now_ms=now_ms)
        assert label != "extreme-volatility"
        assert not math.isnan(pct)
        assert pct < config.ATR_EXTREME_PERCENTILE


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
