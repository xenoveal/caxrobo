"""Tests for correlation.py: correlation matrix, effective-N, gap integrity,
and the RESEARCH_SYMBOLS selection rule (v0.3.0 Phase 2).

correlation.py holds no cache (pure computation over DB reads), so no
engine.clear_caches()-style autouse fixture is needed here -- its absence is
deliberate, not an oversight.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot import config
from trading_bot.data import correlation, storage

TF = config.CORRELATION_TIMEFRAME
DAY = storage.TIMEFRAME_MS[TF]
START = 1_700_000_000_000 // DAY * DAY  # arbitrary epoch-ms base, grid-aligned


def make_conn(tmp_path, name="test.db"):
    return storage.connect(str(tmp_path / name))


def make_rows(n, start=START, interval=DAY, close=100.0):
    return [[start + i * interval, 99.0, 101.0, 98.0, close, 10.0] for i in range(n)]


def seed_closes(conn, symbol, closes, *, start=START, interval=DAY, volume=10.0, timeframe=TF):
    rows = [
        [start + i * interval, c, c, c, c, volume] for i, c in enumerate(closes)
    ]
    storage.upsert_candles(conn, symbol, timeframe, rows)
    return rows


def prices_from_returns(returns, start_price=100.0):
    """Compound a return sequence into a price sequence: prices[0]=start_price,
    prices[i+1] = prices[i] * (1 + returns[i]). daily_return_frame recovers
    `returns` exactly (to floating-point tolerance) from these prices."""
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


def seed_integrity_ok(conn, symbol, n_days, *, start=START):
    """Seed dummy 4h/1h data (values irrelevant to correlation, only to
    timestamps) so gap_integrity reports the symbol clean; the 1d series
    (seeded separately) carries the test's actual correlation structure."""
    for timeframe, bars_per_day in (("4h", 6), ("1h", 24)):
        interval = storage.TIMEFRAME_MS[timeframe]
        n = n_days * bars_per_day
        rows = [
            [start + i * interval, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(n)
        ]
        storage.upsert_candles(conn, symbol, timeframe, rows)


class TestDailyReturnFrame:
    """daily_return_frame: alignment, grid assertions, inclusive bounds."""

    def test_hand_computed_returns(self, tmp_path):
        """closes 100 -> 110 -> 99 => returns +0.10, -0.10 (explicit arithmetic,
        not Series.pct_change() -- pandas 3.0.3's fill behaviour is unsafe)."""
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0, 110.0, 99.0])
        r = correlation.daily_return_frame(
            conn, ["AAAUSDT"], start_ms=START, end_ms=START + 2 * DAY
        )
        assert r.shape == (2, 1)
        vals = r["AAAUSDT"].tolist()
        assert abs(vals[0] - 0.10) < 1e-9
        assert abs(vals[1] - (-0.10)) < 1e-9

    def test_shape_is_n_bars_minus_one_by_n_symbols(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0, 101.0, 102.0, 103.0, 104.0])
        seed_closes(conn, "BBBUSDT", [50.0, 51.0, 52.0, 53.0, 54.0])
        r = correlation.daily_return_frame(
            conn, ["AAAUSDT", "BBBUSDT"], start_ms=START, end_ms=START + 4 * DAY
        )
        assert r.shape == (4, 2)

    def test_missing_interior_bar_drops_ts_from_both_columns(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0, 101.0, 102.0, 103.0, 104.0])
        rows_b = make_rows(5, close=50.0)
        del rows_b[2]  # BBBUSDT missing the interior bar at index 2
        storage.upsert_candles(conn, "BBBUSDT", TF, rows_b)

        r = correlation.daily_return_frame(
            conn, ["AAAUSDT", "BBBUSDT"], start_ms=START, end_ms=START + 4 * DAY
        )
        # ts index 2 and 3 both depend on the missing price at index 2, so
        # BOTH become NaN for BBBUSDT and both timestamps are dropped from
        # the listwise join -- including AAAUSDT's otherwise-valid values.
        dropped_ts = {START + 2 * DAY, START + 3 * DAY}
        assert not dropped_ts & set(r.index)
        assert len(r) == 2  # only ts=START+1*DAY and START+4*DAY survive

    def test_wrong_median_spacing_raises(self, tmp_path):
        """A series stored under the '1d' key but spaced every 2 days is
        grid-aligned (every ts % DAY == 0) yet its median spacing (2*DAY)
        disagrees with the declared timeframe -- must raise naming spacing."""
        conn = make_conn(tmp_path)
        rows = [[START + i * 2 * DAY, 1, 1, 1, 100.0 + i, 1.0] for i in range(5)]
        storage.upsert_candles(conn, "AAAUSDT", TF, rows)
        with pytest.raises(ValueError, match="median bar spacing"):
            correlation.daily_return_frame(
                conn, ["AAAUSDT"], start_ms=START, end_ms=START + 8 * DAY
            )

    def test_non_grid_ts_raises(self, tmp_path):
        conn = make_conn(tmp_path)
        rows = make_rows(5)
        rows[2][0] += 12345  # knock one timestamp off the daily grid
        storage.upsert_candles(conn, "AAAUSDT", TF, rows)
        with pytest.raises(ValueError, match="grid-aligned"):
            correlation.daily_return_frame(
                conn, ["AAAUSDT"], start_ms=START, end_ms=START + 4 * DAY
            )

    def test_fewer_than_two_bars_returns_empty_no_raise(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0])
        r = correlation.daily_return_frame(
            conn, ["AAAUSDT"], start_ms=START, end_ms=START
        )
        assert len(r) == 0

    def test_inclusive_bounds_include_first_and_last_ts(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0, 101.0, 102.0, 103.0, 104.0])
        first_ts, last_ts = START, START + 4 * DAY
        r = correlation.daily_return_frame(
            conn, ["AAAUSDT"], start_ms=first_ts, end_ms=last_ts
        )
        assert len(r) == 4  # 5 bars -> 4 returns, both endpoints included


class TestCorrelationMatrix:
    def test_proportional_series_gives_r_one(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0, 110.0, 121.0, 108.9])
        seed_closes(conn, "BBBUSDT", [50.0, 55.0, 60.5, 54.45])  # identical % moves
        r = correlation.daily_return_frame(
            conn, ["AAAUSDT", "BBBUSDT"], start_ms=START, end_ms=START + 3 * DAY
        )
        m = correlation.correlation_matrix(r)
        assert abs(m.loc["AAAUSDT", "BBBUSDT"] - 1.0) < 1e-9

    def test_anti_proportional_series_gives_r_minus_one(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0, 110.0, 99.0, 108.9])
        seed_closes(conn, "BBBUSDT", [100.0, 90.0, 99.9, 89.91])  # mirrored % moves
        r = correlation.daily_return_frame(
            conn, ["AAAUSDT", "BBBUSDT"], start_ms=START, end_ms=START + 3 * DAY
        )
        m = correlation.correlation_matrix(r)
        assert abs(m.loc["AAAUSDT", "BBBUSDT"] - (-1.0)) < 1e-9

    def test_constant_column_gives_nan_and_effective_n_does_not_raise(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0, 101.0, 102.0, 103.0])
        seed_closes(conn, "FLATUSDT", [50.0, 50.0, 50.0, 50.0])  # zero variance
        r = correlation.daily_return_frame(
            conn, ["AAAUSDT", "FLATUSDT"], start_ms=START, end_ms=START + 3 * DAY
        )
        m = correlation.correlation_matrix(r)
        assert pd.isna(m.loc["AAAUSDT", "FLATUSDT"])
        correlation.effective_n(m)  # must not raise

    def test_symmetry_and_unit_diagonal(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "AAAUSDT", [100.0, 101.0, 99.0, 103.0, 98.0])
        seed_closes(conn, "BBBUSDT", [50.0, 52.0, 49.0, 51.0, 53.0])
        r = correlation.daily_return_frame(
            conn, ["AAAUSDT", "BBBUSDT"], start_ms=START, end_ms=START + 4 * DAY
        )
        m = correlation.correlation_matrix(r)
        assert m.loc["AAAUSDT", "AAAUSDT"] == 1.0
        assert m.loc["BBBUSDT", "BBBUSDT"] == 1.0
        assert abs(m.loc["AAAUSDT", "BBBUSDT"] - m.loc["BBBUSDT", "AAAUSDT"]) < 1e-12

    def test_fewer_than_two_observations_gives_empty_frame(self):
        assert correlation.correlation_matrix(pd.DataFrame({"A": [0.01]})).empty


class TestEffectiveN:
    """Hand-computed reference values (contract §8's HAND_COMPUTED_NUMERIC_TEST)."""

    def _corr(self, symbols, off_diag):
        m = len(symbols)
        arr = np.full((m, m), off_diag)
        np.fill_diagonal(arr, 1.0)
        return pd.DataFrame(arr, index=symbols, columns=symbols)

    def test_m3_rbar_0p5_gives_1p5(self):
        c = self._corr(["A", "B", "C"], 0.5)
        m, r_bar, n_eff = correlation.effective_n(c)
        assert m == 3
        assert abs(r_bar - 0.5) < 1e-12
        assert abs(n_eff - 1.5) < 1e-9

    def test_m2_rbar_0_gives_2p0(self):
        c = self._corr(["A", "B"], 0.0)
        m, r_bar, n_eff = correlation.effective_n(c)
        assert m == 2
        assert r_bar == 0.0
        assert n_eff == 2.0

    def test_m4_rbar_1_gives_1p0(self):
        c = self._corr(["A", "B", "C", "D"], 1.0)
        m, r_bar, n_eff = correlation.effective_n(c)
        assert m == 4
        assert r_bar == 1.0
        assert n_eff == 1.0

    def test_m1_gives_1_0_1(self):
        c = pd.DataFrame({"A": [1.0]}, index=["A"])
        assert correlation.effective_n(c) == (1, 0.0, 1.0)

    def test_negative_rbar_clamped_into_1_m(self):
        c = self._corr(["A", "B", "C"], -0.9)
        m, r_bar, n_eff = correlation.effective_n(c)
        assert 1.0 <= n_eff <= 3.0
        assert n_eff == 1.0  # closed form goes negative; clamp floors it at 1

    def test_known_limitations_anchor_reproduction(self):
        """The values.mean()-bug guard: an all-off-diagonals-equal 3x3 at
        r_bar=0.7574 must give N_eff ~= 1.1929, matching KNOWN-LIMITATIONS
        §0b exactly. Using corr.values.mean() (which wrongly includes the
        diagonal's three 1.0s) would give r_bar ~= 0.8383 and N_eff ~= 1.09
        -- a plausible-looking WRONG number."""
        c = self._corr(["BTCUSDT", "ETHUSDT", "SOLUSDT"], 0.7574)
        m, r_bar, n_eff = correlation.effective_n(c)
        assert abs(r_bar - 0.7574) < 1e-9
        assert abs(n_eff - 1.1929) < 1e-4

    def test_participation_ratio_identity_gives_m(self):
        m = 4
        c = pd.DataFrame(np.eye(m), index=list("ABCD"), columns=list("ABCD"))
        assert abs(correlation.effective_n_participation(c) - m) < 1e-9

    def test_participation_ratio_all_ones_gives_1(self):
        m = 4
        c = pd.DataFrame(np.ones((m, m)), index=list("ABCD"), columns=list("ABCD"))
        assert abs(correlation.effective_n_participation(c) - 1.0) < 1e-9


class TestBtcBeta:
    def test_beta_two_times_anchor(self, tmp_path):
        conn = make_conn(tmp_path)
        anchor_returns = [0.01, -0.02, 0.03, -0.01]
        sym_returns = [2 * r for r in anchor_returns]
        seed_closes(conn, "BTCUSDT", prices_from_returns(anchor_returns))
        seed_closes(conn, "DBLUSDT", prices_from_returns(sym_returns))
        r = correlation.daily_return_frame(
            conn, ["BTCUSDT", "DBLUSDT"], start_ms=START, end_ms=START + 4 * DAY
        )
        beta = correlation.btc_beta(r, "DBLUSDT", anchor="BTCUSDT")
        assert abs(beta - 2.0) < 1e-9
        corr_matrix = correlation.correlation_matrix(r)
        assert abs(corr_matrix.loc["BTCUSDT", "DBLUSDT"] - 1.0) < 1e-9

    def test_anchor_vs_itself_is_exactly_one(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "BTCUSDT", [100.0, 101.0, 99.0, 103.0, 98.0])
        r = correlation.daily_return_frame(
            conn, ["BTCUSDT"], start_ms=START, end_ms=START + 4 * DAY
        )
        assert correlation.btc_beta(r, "BTCUSDT", anchor="BTCUSDT") == 1.0

    def test_zero_variance_anchor_gives_none(self, tmp_path):
        conn = make_conn(tmp_path)
        seed_closes(conn, "BTCUSDT", [100.0, 100.0, 100.0, 100.0, 100.0])
        seed_closes(conn, "AAAUSDT", [50.0, 51.0, 49.0, 52.0, 48.0])
        r = correlation.daily_return_frame(
            conn, ["BTCUSDT", "AAAUSDT"], start_ms=START, end_ms=START + 4 * DAY
        )
        assert correlation.btc_beta(r, "AAAUSDT", anchor="BTCUSDT") is None


class TestGapIntegrity:
    def test_interior_gap_reports_one_entry(self, tmp_path):
        conn = make_conn(tmp_path)
        rows = make_rows(10)
        removed = rows.pop(5)
        storage.upsert_candles(conn, "AAAUSDT", "1d", rows)
        result = correlation.gap_integrity(conn, ["AAAUSDT"], timeframes=("1d",))
        assert result["AAAUSDT"] == [f"1d {removed[0]}-{removed[0]}"]
        assert removed[0] % DAY == 0

    def test_stale_no_hole_gives_empty_list(self, tmp_path):
        """The most important test in this module: a series that is merely
        STALE (normal poller lag) must NOT be reported as having an interior
        gap. Without this, all 20 real symbols would be disqualified for
        ordinary trailing lag at any moment (measured: 40 of 60 cells)."""
        conn = make_conn(tmp_path)
        # Seeded once, long ago relative to "now" -- guaranteed stale by the
        # time this test runs, and stays stale for the life of the repo.
        storage.upsert_candles(conn, "AAAUSDT", "1d", make_rows(5, start=START))
        result = correlation.gap_integrity(conn, ["AAAUSDT"], timeframes=("1d",))
        assert result["AAAUSDT"] == []

    def test_missing_timeframe_reports_missing(self, tmp_path):
        conn = make_conn(tmp_path)
        storage.upsert_candles(conn, "AAAUSDT", "1d", make_rows(5))
        result = correlation.gap_integrity(conn, ["AAAUSDT"], timeframes=("1d", "4h"))
        assert "4h MISSING" in result["AAAUSDT"]
        assert not any(entry.startswith("1d ") for entry in result["AAAUSDT"])


class TestSelectionRule:
    def _stats(self, **overrides):
        base = dict(
            symbol="XUSDT",
            n_bars=1000,
            mean_r=0.5,
            btc_beta=0.5,
            median_quote_volume=1_000_000_000.0,
            interior_gaps=0,
            eligible=True,
            reason="",
        )
        base.update(overrides)
        return correlation.SymbolStats(**base)

    def test_screen_below_liquidity_floor_excluded_for_volume(self, monkeypatch):
        monkeypatch.setattr(config, "CORRELATION_MIN_QUOTE_VOLUME_USD", 1_000.0)
        eligible, reason = correlation.screen_eligibility(
            n_bars=1000, interior_gaps=0, median_quote_volume=1.0,
            btc_beta=0.5, is_anchor=False,
        )
        assert not eligible
        assert reason == "liquidity"

    def test_screen_above_beta_ceiling_excluded_for_beta(self, monkeypatch):
        monkeypatch.setattr(config, "CORRELATION_MAX_BTC_BETA", 1.0)
        eligible, reason = correlation.screen_eligibility(
            n_bars=1000, interior_gaps=0, median_quote_volume=1e9,
            btc_beta=5.0, is_anchor=False,
        )
        assert not eligible
        assert reason == "beta"

    def test_screen_with_interior_gap_excluded_for_gaps(self):
        eligible, reason = correlation.screen_eligibility(
            n_bars=1000, interior_gaps=1, median_quote_volume=1e9,
            btc_beta=0.5, is_anchor=False,
        )
        assert not eligible
        assert reason == "gaps"

    def test_anchor_exempt_from_beta_screen_only(self, monkeypatch):
        monkeypatch.setattr(config, "CORRELATION_MAX_BTC_BETA", 1.0)
        eligible, reason = correlation.screen_eligibility(
            n_bars=1000, interior_gaps=0, median_quote_volume=1e9,
            btc_beta=999.0, is_anchor=True,
        )
        assert eligible
        assert reason == ""
        # but NOT exempt from gaps:
        eligible, reason = correlation.screen_eligibility(
            n_bars=1000, interior_gaps=1, median_quote_volume=1e9,
            btc_beta=999.0, is_anchor=True,
        )
        assert not eligible
        assert reason == "gaps"

    def test_selection_anchor_first_then_ascending_rank(self):
        stats = {
            "BTCUSDT": self._stats(symbol="BTCUSDT", mean_r=None),
            "LOWUSDT": self._stats(symbol="LOWUSDT", mean_r=0.1),
            "MIDUSDT": self._stats(symbol="MIDUSDT", mean_r=0.3),
            "HIUSDT": self._stats(symbol="HIUSDT", mean_r=0.9),
        }
        result = correlation.select_research_symbols(
            stats, pd.DataFrame(), anchor="BTCUSDT", select_n=3
        )
        assert result == ("BTCUSDT", "LOWUSDT", "MIDUSDT", "HIUSDT")

    def test_ineligible_candidates_excluded(self):
        stats = {
            "BTCUSDT": self._stats(symbol="BTCUSDT", mean_r=None),
            "OKUSDT": self._stats(symbol="OKUSDT", mean_r=0.2),
            "GAPUSDT": self._stats(
                symbol="GAPUSDT", mean_r=0.05, eligible=False, reason="gaps",
                interior_gaps=1,
            ),
        }
        result = correlation.select_research_symbols(
            stats, pd.DataFrame(), anchor="BTCUSDT", select_n=1
        )
        assert result == ("BTCUSDT", "OKUSDT")
        assert "GAPUSDT" not in result

    def test_determinism_under_shuffled_input_and_ties(self):
        stats = {
            "BTCUSDT": self._stats(symbol="BTCUSDT", mean_r=None),
            "AAAUSDT": self._stats(symbol="AAAUSDT", mean_r=0.4, median_quote_volume=100.0),
            "BBBUSDT": self._stats(symbol="BBBUSDT", mean_r=0.4, median_quote_volume=100.0),
        }
        r1 = correlation.select_research_symbols(
            stats, pd.DataFrame(), anchor="BTCUSDT", select_n=2
        )
        reordered = dict(reversed(list(stats.items())))
        r2 = correlation.select_research_symbols(
            reordered, pd.DataFrame(), anchor="BTCUSDT", select_n=2
        )
        assert r1 == r2 == ("BTCUSDT", "AAAUSDT", "BBBUSDT")  # tie -> alphabetical

    def test_fewer_than_select_n_eligible_raises(self):
        stats = {
            "BTCUSDT": self._stats(symbol="BTCUSDT", mean_r=None),
            "ONLYUSDT": self._stats(symbol="ONLYUSDT", mean_r=0.2),
        }
        with pytest.raises(ValueError, match="decision D3"):
            correlation.select_research_symbols(
                stats, pd.DataFrame(), anchor="BTCUSDT", select_n=2
            )

    def test_ineligible_anchor_raises_never_substitutes(self):
        stats = {
            "BTCUSDT": self._stats(symbol="BTCUSDT", eligible=False, reason="gaps"),
            "AAAUSDT": self._stats(symbol="AAAUSDT", mean_r=0.2),
        }
        with pytest.raises(ValueError, match="BTCUSDT"):
            correlation.select_research_symbols(
                stats, pd.DataFrame(), anchor="BTCUSDT", select_n=1
            )


class TestBuildReportDecision:
    """Synthetic stores engineered (via Hadamard-orthogonal return vectors,
    exact by construction -- no randomness) to land on each decision letter."""

    def _seed_hadamard_universe(self, tmp_path, *, candidates_orthogonal: bool):
        """BTC/ETH/SOL (config.SYMBOLS) share one Hadamard column (r=1
        pairwise, the minimal possible N_eff=1.0 for m=3). Two extra
        candidates get either the SAME column (r=1 -> low N_sel, D2) or an
        ORTHOGONAL column (r=0 -> high N_sel, D1)."""
        conn = make_conn(tmp_path, name="hadamard.db")
        d = 0.01
        core = [d, d, -d, -d]
        orth1 = [d, -d, d, -d]
        orth2 = [d, -d, -d, d]

        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            seed_closes(conn, symbol, prices_from_returns(core))
            seed_integrity_ok(conn, symbol, n_days=4)

        cand_returns = core if not candidates_orthogonal else orth1
        cand2_returns = core if not candidates_orthogonal else orth2
        seed_closes(conn, "C1USDT", prices_from_returns(cand_returns))
        seed_integrity_ok(conn, "C1USDT", n_days=4)
        seed_closes(conn, "C2USDT", prices_from_returns(cand2_returns))
        seed_integrity_ok(conn, "C2USDT", n_days=4)
        return conn

    def _relax_screens(self, monkeypatch):
        monkeypatch.setattr(config, "CORRELATION_MIN_OVERLAP_BARS", 4)
        monkeypatch.setattr(config, "CORRELATION_MIN_QUOTE_VOLUME_USD", 0.0)
        monkeypatch.setattr(config, "CORRELATION_MAX_BTC_BETA", 1e9)

    def test_d1_orthogonal_candidates(self, tmp_path, monkeypatch):
        self._relax_screens(monkeypatch)
        conn = self._seed_hadamard_universe(tmp_path, candidates_orthogonal=True)
        symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "C1USDT", "C2USDT")
        report = correlation.build_report(
            conn, symbols, start_ms=START, end_ms=START + 4 * DAY, select_n=2
        )
        assert report.decision == "D1"
        assert report.selection[0] == "BTCUSDT"

    def test_d2_identical_candidates(self, tmp_path, monkeypatch):
        self._relax_screens(monkeypatch)
        conn = self._seed_hadamard_universe(tmp_path, candidates_orthogonal=False)
        symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "C1USDT", "C2USDT")
        report = correlation.build_report(
            conn, symbols, start_ms=START, end_ms=START + 4 * DAY, select_n=2
        )
        assert report.decision == "D2"
        assert "rows" in report.verdict and "information" in report.verdict

    def test_d3_insufficient_candidates(self, tmp_path, monkeypatch):
        self._relax_screens(monkeypatch)
        conn = make_conn(tmp_path, name="d3.db")
        core = [0.01, 0.01, -0.01, -0.01]
        seed_closes(conn, "BTCUSDT", prices_from_returns(core))
        seed_integrity_ok(conn, "BTCUSDT", n_days=4)
        seed_closes(conn, "ETHUSDT", prices_from_returns(core))
        seed_integrity_ok(conn, "ETHUSDT", n_days=4)
        symbols = ("BTCUSDT", "ETHUSDT")  # only 1 non-anchor candidate
        report = correlation.build_report(
            conn, symbols, start_ms=START, end_ms=START + 4 * DAY, select_n=2
        )
        assert report.decision == "D3"
        assert report.selection == ()


class TestFormatReport:
    def test_contains_all_sections_and_key_strings(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CORRELATION_MIN_OVERLAP_BARS", 4)
        monkeypatch.setattr(config, "CORRELATION_MIN_QUOTE_VOLUME_USD", 0.0)
        monkeypatch.setattr(config, "CORRELATION_MAX_BTC_BETA", 1e9)
        conn = make_conn(tmp_path)
        core = [0.01, 0.01, -0.01, -0.01]
        orth = [0.01, -0.01, 0.01, -0.01]
        for symbol, seq in (
            ("BTCUSDT", core), ("ETHUSDT", core), ("SOLUSDT", core),
            ("C1USDT", orth), ("C2USDT", orth),
        ):
            seed_closes(conn, symbol, prices_from_returns(seq))
            seed_integrity_ok(conn, symbol, n_days=4)
        report = correlation.build_report(
            conn,
            ("BTCUSDT", "ETHUSDT", "SOLUSDT", "C1USDT", "C2USDT"),
            start_ms=START, end_ms=START + 4 * DAY, select_n=2,
        )
        md = correlation.format_report(report)
        for heading in (
            "# Phase 2 correlation report",
            "## Gap integrity",
            "## Pairwise correlation matrix",
            "## Effective independent sample size",
            "## Per-symbol statistics",
            "## Selection rule and result",
            "## Verdict",
            "## Degrees of freedom consumed",
        ):
            assert heading in md
        assert str(report.n_obs) in md
        assert report.decision in md
        assert "RESEARCH_SYMBOLS" in md


class TestKnownLimitationsAnchors:
    """The correctness gate. Pins KNOWN-LIMITATIONS §0b: 3 symbols at mean
    r ~0.76 -> effective N ~1.2. Skipped on a fresh clone without the real
    OHLCV store so the suite stays green without data/ohlcv.db."""

    @pytest.mark.skipif(
        not Path("data/ohlcv.db").exists(),
        reason="requires the real data/ohlcv.db store",
    )
    def test_reproduces_known_limitations_0b(self):
        import sqlite3

        conn = sqlite3.connect("data/ohlcv.db")
        start_ms = config.date_to_ms(config.CORRELATION_START)
        end_ms = config.date_to_ms("2026-07-26")
        r = correlation.daily_return_frame(
            conn, list(config.SYMBOLS), start_ms=start_ms, end_ms=end_ms
        )
        m = correlation.correlation_matrix(r)
        assert abs(m.loc["BTCUSDT", "ETHUSDT"] - 0.809) < 0.001
        assert abs(m.loc["BTCUSDT", "SOLUSDT"] - 0.744) < 0.001
        assert abs(m.loc["ETHUSDT", "SOLUSDT"] - 0.719) < 0.001
        _, r_bar, n_eff = correlation.effective_n(m)
        assert abs(r_bar - 0.7574) < 0.0005
        assert abs(n_eff - 1.193) < 0.005
        conn.close()


class TestExchangeReachability:
    """The @pytest.mark.network smoke test lives HERE rather than in a new
    file: contract §8 gives Phase 2 only test_correlation.py plus additions
    to test_storage.py, and the exchange probe is Phase 2's backfill
    precondition (decision D3's contingency), not a storage concern."""

    @pytest.mark.network
    @pytest.mark.skipif(
        not os.getenv("RUN_NETWORK_TESTS"),
        reason="Network tests disabled by default; set RUN_NETWORK_TESTS=1 to enable",
    )
    def test_fetch_anchor_daily_bars_network_smoke(self):
        from trading_bot.exchange.binance_client import fetch_ohlcv_page, to_ccxt_symbol

        result = fetch_ohlcv_page(
            symbol=to_ccxt_symbol(config.CORRELATION_ANCHOR_SYMBOL),
            timeframe=config.CORRELATION_TIMEFRAME,
            since_ms=config.date_to_ms(config.BACKFILL_START),
            limit=5,
        )
        assert len(result) > 0
        for i in range(len(result) - 1):
            assert result[i][0] < result[i + 1][0]
