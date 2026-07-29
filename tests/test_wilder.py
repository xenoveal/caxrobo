"""Tests for Wilder's indicators (ATR, DI, ADX)."""

import pandas as pd

from trading_bot.data import storage
from trading_bot.indicators import wilder

# Test fixtures
SYMBOL = "BTCUSDT"
TF = "4h"
INTERVAL = storage.TIMEFRAME_MS[TF]
START = 1_700_000_000_000


def make_conn(tmp_path):
    return storage.connect(str(tmp_path / "test.db"))


def make_ohlcv_rows(n, start=START, interval=INTERVAL, high_offset=2.0, low_offset=2.0, close=100.0):
    """
    Generate n OHLCV rows as [ts, open, high, low, close, volume].

    Args:
        n: Number of rows.
        start: Starting timestamp (epoch-ms).
        interval: Time interval between rows.
        high_offset: Amount above close for high.
        low_offset: Amount below close for low.
        close: Base close price.
    """
    return [
        [start + i * interval, close, close + high_offset, close - low_offset, close, 10.0]
        for i in range(n)
    ]


def make_dataframe(rows):
    """Convert OHLCV rows to a DataFrame with proper column names and index."""
    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    df["ts"] = df["ts"].astype(int)
    df = df.set_index("ts")
    return df


class TestTrueRange:
    """Tests for true_range() calculation."""

    def test_tr_first_bar_is_high_minus_low(self):
        """First bar has no previous close; TR is simply high-low."""
        rows = make_ohlcv_rows(3, close=100.0, high_offset=5.0, low_offset=3.0)
        df = make_dataframe(rows)
        tr = wilder.true_range(df)

        # TR[0] = high - low = (100+5) - (100-3) = 8
        assert tr.iloc[0] == 8.0

    def test_tr_handles_gap_up(self):
        """TR includes gap-up component |high - prev_close|."""
        rows = [
            [START, 100.0, 102.0, 98.0, 100.0, 10.0],  # gap up to 102, range 4
            [START + INTERVAL, 105.0, 110.0, 104.0, 109.0, 10.0],  # high 110, low 104, prev_close 100
        ]
        df = make_dataframe(rows)
        tr = wilder.true_range(df)

        # TR[0] = 110 - 104 = 6 (high - low)
        # But we need prev_close at index 1
        # TR[1] = max(110-104, |110-100|, |104-100|) = max(6, 10, 4) = 10
        assert tr.iloc[1] == 10.0

    def test_tr_handles_gap_down(self):
        """TR includes gap-down component |low - prev_close|."""
        rows = [
            [START, 100.0, 102.0, 98.0, 100.0, 10.0],
            [START + INTERVAL, 95.0, 98.0, 90.0, 92.0, 10.0],  # gap down to 90, range 8
        ]
        df = make_dataframe(rows)
        tr = wilder.true_range(df)

        # TR[1] = max(98-90, |98-100|, |90-100|) = max(8, 2, 10) = 10
        assert tr.iloc[1] == 10.0


class TestWilderSmooth:
    """Tests for wilder_smooth() calculation."""

    def test_wilder_smooth_first_value_at_period_minus_one(self):
        """First smoothed value appears at index period-1."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=range(5))
        smoothed = wilder.wilder_smooth(series, period=3)

        assert pd.isna(smoothed.iloc[0])
        assert pd.isna(smoothed.iloc[1])
        assert pd.notna(smoothed.iloc[2])

    def test_wilder_smooth_seeds_with_sma(self):
        """First smoothed value is the SMA of first period values."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=range(5))
        smoothed = wilder.wilder_smooth(series, period=3)

        # smoothed[2] = mean([1, 2, 3]) = 2.0
        expected_first = (1.0 + 2.0 + 3.0) / 3.0
        assert abs(smoothed.iloc[2] - expected_first) < 0.001

    def test_wilder_smooth_recursive_formula(self):
        """Subsequent values follow the recursive formula."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=range(5))
        smoothed = wilder.wilder_smooth(series, period=3)

        # smoothed[2] = (1+2+3)/3 = 2.0
        # smoothed[3] = (2.0 * 2 + 4) / 3 = 8/3 = 2.6667
        # smoothed[4] = (2.6667 * 2 + 5) / 3 = (5.3333 + 5) / 3 = 10.3333 / 3 = 3.4444
        assert abs(smoothed.iloc[3] - 8.0 / 3.0) < 0.001
        assert abs(smoothed.iloc[4] - 10.3333 / 3.0) < 0.001

    def test_wilder_smooth_short_series(self):
        """Series shorter than period returns all NaN."""
        series = pd.Series([1.0, 2.0], index=range(2))
        smoothed = wilder.wilder_smooth(series, period=3)

        assert smoothed.isna().all()


class TestATR:
    """Tests for atr() calculation."""

    def test_atr_simple_uptrend(self):
        """ATR with simple uptrend and stable range."""
        rows = [
            [START + i * INTERVAL, 100.0, 102.0 + i, 98.0 + i, 100.0 + i, 10.0]
            for i in range(6)
        ]
        df = make_dataframe(rows)
        atr = wilder.atr(df, period=3)

        # TR should be roughly constant (range of 4 + some gaps)
        assert pd.isna(atr.iloc[0])
        assert pd.isna(atr.iloc[1])
        assert pd.notna(atr.iloc[2])  # First value at index period-1

    def test_atr_matches_tr_when_tr_constant(self):
        """ATR approaches TR if TR values are constant."""
        # Create rows where TR is constant at 4.0
        tr_val = 4.0
        rows = [
            [START + i * INTERVAL, 100.0, 102.0, 98.0, 100.0, 10.0]
            for i in range(10)
        ]
        df = make_dataframe(rows)
        atr = wilder.atr(df, period=3)
        tr = wilder.true_range(df)

        # All TR values should be 4.0
        for i in range(1, len(tr)):
            assert abs(tr.iloc[i] - 4.0) < 0.01

        # ATR should converge to 4.0
        assert abs(atr.iloc[2] - 4.0) < 0.01
        assert abs(atr.iloc[-1] - 4.0) < 0.01


class TestDI:
    """Tests for plus_di() and minus_di() calculations."""

    def test_di_simple_uptrend(self):
        """+DI rises in uptrend, -DI stays low."""
        # Create a clear uptrend
        rows = [
            [START + i * INTERVAL, 100.0 + i, 102.0 + i, 98.0 + i, 100.5 + i, 10.0]
            for i in range(15)
        ]
        df = make_dataframe(rows)

        pdi = wilder.plus_di(df, period=5)
        mdi = wilder.minus_di(df, period=5)

        # First values appear at index period-1 = 4 for period=5
        assert pd.isna(pdi.iloc[3])
        assert pd.notna(pdi.iloc[4])

        # In an uptrend, +DI should be higher than -DI
        assert pdi.iloc[-1] > mdi.iloc[-1]

    def test_di_simple_downtrend(self):
        """-DI rises in downtrend, +DI stays low."""
        rows = [
            [START + i * INTERVAL, 100.0 - i, 102.0 - i, 98.0 - i, 100.5 - i, 10.0]
            for i in range(15)
        ]
        df = make_dataframe(rows)

        pdi = wilder.plus_di(df, period=5)
        mdi = wilder.minus_di(df, period=5)

        # In a downtrend, -DI should be higher than +DI
        assert pd.notna(pdi.iloc[8])
        assert pd.notna(mdi.iloc[8])
        assert mdi.iloc[-1] > pdi.iloc[-1]


class TestADX:
    """Tests for adx() calculation."""

    def test_adx_warmup_period(self):
        """ADX first appears at index 2*period-2."""
        rows = make_ohlcv_rows(50)
        df = make_dataframe(rows)
        adx = wilder.adx(df, period=3)

        # First value at index 2*3-2 = 4
        # (period-1 to get +DI/-DI, then another period-1 to smooth DX)
        assert pd.isna(adx.iloc[3])
        assert pd.notna(adx.iloc[4])

    def test_adx_for_period_14(self):
        """ADX(14) needs 27 bars before first value."""
        rows = make_ohlcv_rows(50)
        df = make_dataframe(rows)
        adx = wilder.adx(df, period=14)

        # First value at index 2*14-2 = 26
        assert pd.isna(adx.iloc[25])
        assert pd.notna(adx.iloc[26])

    def test_adx_uptrend_high(self):
        """ADX is high in a strong uptrend."""
        rows = [
            [START + i * INTERVAL, 100.0 + i*2, 103.0 + i*2, 99.0 + i*2, 102.0 + i*2, 10.0]
            for i in range(50)
        ]
        df = make_dataframe(rows)
        adx = wilder.adx(df, period=5)

        # After warmup, ADX should be high in this consistent uptrend
        assert pd.notna(adx.iloc[8])
        assert adx.iloc[-1] > 20.0  # Should be reasonably high for a clear trend

    def test_adx_range_low(self):
        """ADX is low in a ranging market."""
        # Oscillating close around a midpoint
        rows = []
        for i in range(50):
            close = 100.0 + (5.0 if i % 2 == 0 else -5.0)
            rows.append([START + i * INTERVAL, close, close + 0.5, close - 0.5, close, 10.0])

        df = make_dataframe(rows)
        adx = wilder.adx(df, period=5)

        # In a ranging market, ADX should be low
        assert pd.notna(adx.iloc[8])
        assert adx.iloc[-1] < 20.0

    def test_adx_is_between_0_and_100(self):
        """ADX values are always between 0 and 100."""
        rows = make_ohlcv_rows(50)
        df = make_dataframe(rows)
        adx = wilder.adx(df, period=5)

        valid_adx = adx.dropna()
        assert (valid_adx >= 0.0).all()
        assert (valid_adx <= 100.0).all()


class TestMonotonicUptrend:
    """Test indicators on a monotonic uptrend that should yield strong signals."""

    def test_monotonic_uptrend_adx_above_25(self):
        """Consistent uptrend eventually produces ADX > 25."""
        # Create a strictly increasing price
        rows = [
            [START + i * INTERVAL, 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 10.0]
            for i in range(40)
        ]
        df = make_dataframe(rows)
        adx = wilder.adx(df, period=14)

        # After warmup, ADX should exceed 25 in a strong uptrend
        valid_adx = adx[adx.notna()]
        assert len(valid_adx) > 0
        assert valid_adx.iloc[-1] > 25.0


class TestLeadingNaNs:
    """Test that leading NaNs are preserved."""

    def test_tr_first_nan(self):
        """True Range has NaN for components that depend on previous close."""
        rows = make_ohlcv_rows(5)
        df = make_dataframe(rows)
        tr = wilder.true_range(df)

        # First row should have valid TR (it's just high-low)
        assert pd.notna(tr.iloc[0])

    def test_atr_leading_nans(self):
        """ATR has NaN for first period-1 bars."""
        rows = make_ohlcv_rows(20)
        df = make_dataframe(rows)
        atr = wilder.atr(df, period=5)

        for i in range(4):  # period - 1 = 4
            assert pd.isna(atr.iloc[i])
        assert pd.notna(atr.iloc[4])

    def test_adx_leading_nans(self):
        """ADX has NaN until 2*period-1 bars have been processed."""
        rows = make_ohlcv_rows(30)
        df = make_dataframe(rows)

        for period in [3, 5, 14]:
            adx = wilder.adx(df, period=period)
            min_bars_needed = 2 * period - 1

            # All values before index min_bars_needed - 1 should be NaN
            for i in range(min_bars_needed - 1):
                assert pd.isna(adx.iloc[i])

            # Value at index min_bars_needed - 1 should be defined
            if min_bars_needed - 1 < len(adx):
                assert pd.notna(adx.iloc[min_bars_needed - 1])


class TestHandComputedValues:
    """Test against hand-computed reference values for correctness."""

    def test_tr_hand_computed(self):
        """True Range matches hand-computed values."""
        # Simple series where TR values are easy to calculate
        rows = [
            [START, 100.0, 102.0, 98.0, 100.0, 10.0],      # TR = 4
            [START + INTERVAL, 100.0, 105.0, 99.0, 102.0, 10.0],  # TR = max(6, 5, 1) = 6
            [START + 2*INTERVAL, 102.0, 104.0, 100.0, 103.0, 10.0],  # TR = max(4, 2, 2) = 4
        ]
        df = make_dataframe(rows)
        tr = wilder.true_range(df)

        assert abs(tr.iloc[0] - 4.0) < 0.01
        assert abs(tr.iloc[1] - 6.0) < 0.01
        assert abs(tr.iloc[2] - 4.0) < 0.01

    def test_atr_hand_computed_simple(self):
        """ATR matches hand-computed values for simple series."""
        # Constant range, so ATR should be that range value
        rows = [
            [START + i * INTERVAL, 100.0, 102.0, 98.0, 100.0, 10.0]
            for i in range(7)
        ]
        df = make_dataframe(rows)
        atr = wilder.atr(df, period=3)

        # All TR values are 4.0
        # atr[2] = (4 + 4 + 4) / 3 = 4.0
        # atr[3] = (4 * 2 + 4) / 3 = 4.0
        # ... all should be 4.0
        assert abs(atr.iloc[2] - 4.0) < 0.01
        assert abs(atr.iloc[-1] - 4.0) < 0.01
