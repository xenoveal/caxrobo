"""Tier-configuration invariants (Phase 4: tier shift to 1D/4H/1H).

These tests exist so that a future edit cannot silently reintroduce a
hardcoded timeframe assumption. They assert properties of the CONFIGURATION
and of the engine's tier guard, not of any strategy outcome.
"""

import pathlib
import re

import pandas as pd
import pytest

from trading_bot import config
from trading_bot.backtest.engine import _assert_interval
from trading_bot.data import storage

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "trading_bot"


def _frame(interval_ms, n=10):
    ts = [1_700_000_000_000 + i * interval_ms for i in range(n)]
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.Index(ts, name="ts"),
    )


class TestTierConfiguration:
    def test_all_three_tiers_are_known_timeframes(self):
        for tf in (
            config.REGIME_TIMEFRAME,
            config.SIGNAL_PATTERN_TIMEFRAME,
            config.SIGNAL_TRIGGER_TIMEFRAME,
        ):
            assert tf in storage.TIMEFRAME_MS, f"{tf} missing from TIMEFRAME_MS"

    def test_tiers_are_strictly_coarse_to_fine(self):
        """Regime must be coarser than setup, setup coarser than trigger."""
        reg = storage.TIMEFRAME_MS[config.REGIME_TIMEFRAME]
        setup = storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME]
        trig = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
        assert reg > setup > trig

    def test_coarser_closes_land_on_trigger_boundaries(self):
        """The no-lookahead searchsorted alignment depends on this."""
        trig = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
        assert storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME] % trig == 0
        assert storage.TIMEFRAME_MS[config.REGIME_TIMEFRAME] % trig == 0

    def test_tiers_are_the_prd_phase_4_values(self):
        assert config.REGIME_TIMEFRAME == "1d"
        assert config.SIGNAL_PATTERN_TIMEFRAME == "4h"
        assert config.SIGNAL_TRIGGER_TIMEFRAME == "1h"

    def test_hold_limit_is_named_for_the_trigger_tier(self):
        assert hasattr(config, "MAX_HOLD_BARS_TRIGGER")
        assert not hasattr(config, "MAX_HOLD_BARS_15M")

    def test_every_tier_is_an_ingested_timeframe(self):
        """A tier the poller/backfill never fetches would silently stay empty."""
        for tf in (
            config.REGIME_TIMEFRAME,
            config.SIGNAL_PATTERN_TIMEFRAME,
            config.SIGNAL_TRIGGER_TIMEFRAME,
        ):
            assert tf in config.TIMEFRAMES, f"{tf} is a tier but is never ingested"


class TestAssertInterval:
    def test_matching_interval_returns_expected_ms(self):
        tf = config.SIGNAL_TRIGGER_TIMEFRAME
        ms = storage.TIMEFRAME_MS[tf]
        assert _assert_interval(_frame(ms), tf, "BTCUSDT", "trigger") == ms

    def test_mismatched_interval_raises(self):
        tf = config.SIGNAL_TRIGGER_TIMEFRAME
        wrong = storage.TIMEFRAME_MS[tf] // 4
        with pytest.raises(ValueError, match="spacing"):
            _assert_interval(_frame(wrong), tf, "BTCUSDT", "trigger")

    def test_single_bar_series_is_not_checked(self):
        tf = config.SIGNAL_TRIGGER_TIMEFRAME
        ms = storage.TIMEFRAME_MS[tf]
        assert _assert_interval(_frame(ms, n=1), tf, "BTCUSDT", "trigger") == ms

    def test_a_few_missing_bars_do_not_trip_the_median(self):
        tf = config.SIGNAL_TRIGGER_TIMEFRAME
        ms = storage.TIMEFRAME_MS[tf]
        df = _frame(ms, n=12).drop(index=[1_700_000_000_000 + 5 * ms])
        assert _assert_interval(df, tf, "BTCUSDT", "trigger") == ms


class TestNoHardcodedTierLiterals:
    """Grep-audit-as-a-test: the PRD's named mitigation, made permanent."""

    SIGNAL_PATH = (
        "signals/setup.py",
        "signals/meanrev.py",
        "signals/breakout.py",
        "signals/scan.py",
        "signals/patterns.py",
        "signals/pivots.py",
        "backtest/engine.py",
        "regime/classifier.py",
    )

    def test_no_15m_mentions_on_the_signal_path(self):
        offenders = []
        for rel in self.SIGNAL_PATH:
            text = (SRC / rel).read_text()
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(r"15m|15M|900_000|900000", line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        assert not offenders, "hardcoded 15m assumption reintroduced:\n" + "\n".join(
            offenders
        )

    def test_signal_path_never_hardcodes_a_timeframe_string(self):
        """Timeframes must be read from config, never inlined."""
        offenders = []
        for rel in self.SIGNAL_PATH:
            for i, line in enumerate((SRC / rel).read_text().splitlines(), 1):
                if re.search(r'''["'](?:1m|5m|15m|30m|1h|2h|4h|6h|12h|1d|1w)["']''', line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        assert not offenders, "inline timeframe literal:\n" + "\n".join(offenders)
