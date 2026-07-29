"""
Tests for feedback/versioning.py (v0.3.0 Phase 5, contract §3/§6/§8).
"""

import json

import pytest

from trading_bot import config
from trading_bot.backtest.engine import Trade
from trading_bot.data import statestore
from trading_bot.feedback import versioning
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.plugins import build_v020_graph

START = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _load_registry():
    registry.load_all()


@pytest.fixture
def state_conn():
    conn = statestore.connect(":memory:")
    yield conn
    conn.close()


def make_trade(strategy_version="", entry_ts=START, exit_ts=None):
    exit_ts = START + 3_600_000 if exit_ts is None else exit_ts
    return Trade(
        symbol="BTCUSDT", regime="trending", pattern="donchian-breakout",
        direction="long", entry_ts=entry_ts, entry=100.0, stop=95.0, target=110.0,
        exit_ts=exit_ts, exit_price=105.0, outcome="target", pnl_pct=0.05,
        volume_high=False, strategy_version=strategy_version,
    )


class TestRegister:
    def test_same_graph_gives_same_version_id(self, state_conn):
        g = build_v020_graph()
        v1 = versioning.register_version(state_conn, g)
        v2 = versioning.register_version(state_conn, g)
        assert v1.version_id == v2.version_id

    def test_key_order_does_not_change_the_id(self, state_conn):
        g1 = build_v020_graph(rr_floor=2.0, adx_trend_threshold=25.0)
        g2 = build_v020_graph(adx_trend_threshold=25.0, rr_floor=2.0)
        assert versioning.version_id_for(g1) == versioning.version_id_for(g2)

    def test_different_params_give_different_ids(self, state_conn):
        g1 = build_v020_graph(rr_floor=1.5)
        g2 = build_v020_graph(rr_floor=2.0)
        assert versioning.version_id_for(g1) != versioning.version_id_for(g2)

    def test_reregistering_is_idempotent(self, state_conn):
        g = build_v020_graph()
        v1 = versioning.register_version(state_conn, g, created_ts=START)
        v2 = versioning.register_version(state_conn, g, created_ts=START + 999_999)
        assert v2.created_ts == START  # unchanged, NOT overwritten
        rows = state_conn.execute(
            f"SELECT COUNT(*) FROM {versioning.TABLE}"
        ).fetchone()
        assert rows[0] == 1

    def test_self_parent_is_refused(self, state_conn):
        g = build_v020_graph()
        vid = versioning.version_id_for(g)
        with pytest.raises(ValueError, match="own parent"):
            versioning.register_version(state_conn, g, parent_id=vid)

    def test_graph_json_matches_the_stored_hash(self, state_conn):
        g = build_v020_graph()
        v = versioning.register_version(state_conn, g)
        reloaded = fgraph.StrategyGraph.from_dict(json.loads(v.graph_json))
        assert fgraph.graph_hash(reloaded) == v.graph_hash

    def test_unknown_version_raises_keyerror(self, state_conn):
        with pytest.raises(KeyError):
            versioning.get_version(state_conn, "does-not-exist")


class TestLineage:
    def test_lineage_is_root_first(self, state_conn):
        root_g = build_v020_graph(rr_floor=1.5)
        root = versioning.register_version(state_conn, root_g, created_ts=START)
        child_g = build_v020_graph(rr_floor=1.6)
        child = versioning.register_version(
            state_conn, child_g, parent_id=root.version_id, created_ts=START + 1
        )
        grandchild_g = build_v020_graph(rr_floor=1.7)
        grandchild = versioning.register_version(
            state_conn, grandchild_g, parent_id=child.version_id, created_ts=START + 2
        )
        chain = versioning.lineage(state_conn, grandchild.version_id)
        assert [v.version_id for v in chain] == [
            root.version_id, child.version_id, grandchild.version_id,
        ]

    def test_lineage_stops_on_a_cycle(self, state_conn):
        versioning.ensure_schema(state_conn)
        # Hand-insert two rows whose parent_id points at each other.
        for vid, parent in (("a", "b"), ("b", "a")):
            state_conn.execute(
                f"INSERT INTO {versioning.TABLE} "
                "(version_id, parent_id, label, graph_json, graph_hash, "
                "schema_version, config_hash, config_snapshot, provenance, created_ts) "
                "VALUES (?, ?, '', '{}', 'h', '1', 'c', '{}', '{}', 0)",
                (vid, parent),
            )
        state_conn.commit()
        with pytest.raises(ValueError, match="cycle"):
            versioning.lineage(state_conn, "a")


class TestLoadGraph:
    def test_roundtrip_through_phase3_from_dict(self, state_conn):
        g = build_v020_graph()
        v = versioning.register_version(state_conn, g)
        loaded = versioning.load_graph(state_conn, v.version_id)
        assert versioning.version_id_for(loaded) == v.version_id

    def test_unknown_version_raises_keyerror(self, state_conn):
        with pytest.raises(KeyError):
            versioning.load_graph(state_conn, "nope")


class TestConfigSnapshot:
    def test_every_config_constant_is_classified(self):
        """Fails when a later phase adds a constant without deciding whether
        it changes what a strategy version does — the whole mitigation for
        config.py being mutable global state."""
        classified = set(versioning.VERSION_CONFIG_KEYS) | set(versioning.VERSION_CONFIG_IGNORED)
        missing = []
        for name in dir(config):
            if not name.isupper():
                continue
            value = getattr(config, name)
            if callable(value):
                continue
            if name not in classified:
                missing.append(name)
        assert missing == [], f"unclassified config constants: {missing}"

    def test_verify_reproducible_detects_drift(self, state_conn, monkeypatch):
        g = build_v020_graph()
        v = versioning.register_version(state_conn, g)
        original_fee = config.FEE_PCT
        monkeypatch.setattr(config, "FEE_PCT", 0.001)
        report = versioning.verify_reproducible(state_conn, v.version_id)
        assert report["config_hash_matches"] is False
        assert report["diff"]["FEE_PCT"] == {"recorded": original_fee, "live": 0.001}

    def test_verify_reproducible_clean_when_nothing_changed(self, state_conn):
        g = build_v020_graph()
        v = versioning.register_version(state_conn, g)
        report = versioning.verify_reproducible(state_conn, v.version_id)
        assert report["config_hash_matches"] is True
        assert report["diff"] == {}

    def test_stamp_trades_sets_the_version_without_mutating_the_original(self):
        t = make_trade(strategy_version="")
        stamped = versioning.stamp_trades([t], "abc123")
        assert stamped[0].strategy_version == "abc123"
        assert t.strategy_version == ""  # original untouched (frozen dataclass)
