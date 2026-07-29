"""Tests for framework/registry.py (v0.3.0 Phase 3).

Every registration test runs inside `with registry.temporary_registry():`.
NEVER call REGISTRY.clear(): importlib caches modules, so the decorators would
not re-run and every later test in the session would see an empty registry and
fail for a reason unrelated to itself.

TestRegisteredPluginsAreWellFormed is the meta-test every later phase inherits
for free: Phase 4's and Phase 8's plug-ins are validated the moment they land,
with no new test code.
"""

import importlib

import pytest

from trading_bot import config
from trading_bot.framework import registry
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.errors import ContractError, RegistryError
from trading_bot.framework.registry import (
    KINDS,
    REGISTRY,
    TIERS,
    _NAME_RE,
    by_kind,
    get,
    load_all,
    register,
    temporary_registry,
)

PRODUCTION_KEYS = {
    "data.ohlcv",
    "detector.bollinger-fade",
    "detector.donchian-breakout",
    "detector.legacy-patterns",
    "policy.atr-stop-measured-move",
    "policy.fade-structural-stop",
}

GOOD = dict(rationale="A sufficiently motivated reason to test this plug-in.")


def _detector(ctx, *, period=1):
    return []


class TestRegisterValidation:
    def test_duplicate_key_names_the_claiming_module(self):
        with temporary_registry():
            register("detector", name="dup", **GOOD)(_detector)
            claimant = REGISTRY["detector.dup"].module
            assert claimant  # the module string is what the message must carry
            with pytest.raises(RegistryError, match=claimant.replace(".", r"\.")):
                register("detector", name="dup", **GOOD)(_detector)

    def test_unknown_kind_raises(self):
        with temporary_registry():
            with pytest.raises(RegistryError, match="unknown kind"):
                register("indicator", name="x", **GOOD)(_detector)

    @pytest.mark.parametrize("name", ["camelCase", "snake_case", "Leading-Caps", "trailing-"])
    def test_bad_name_casing_raises_with_the_legal_form(self, name):
        with temporary_registry():
            with pytest.raises(RegistryError, match="lowercase-hyphen"):
                register("detector", name=name, **GOOD)(_detector)

    def test_good_names_accepted(self):
        for name in ("donchian-breakout", "macd", "head-and-shoulders", "rsi2"):
            assert _NAME_RE.match(name), name

    @pytest.mark.parametrize("rationale", ["", "   ", "\n\t"])
    def test_empty_rationale_raises(self, rationale):
        with temporary_registry():
            with pytest.raises(RegistryError, match="rationale is required"):
                register("detector", name="x", rationale=rationale)(_detector)

    def test_non_paramspec_value_raises(self):
        with temporary_registry():
            with pytest.raises(RegistryError, match="must be a ParamSpec"):
                register("detector", name="x", params={"period": 20}, **GOOD)(_detector)

    def test_unknown_timeframe_raises(self):
        with temporary_registry():
            with pytest.raises(RegistryError, match="unknown timeframe"):
                register("detector", name="x", timeframes=("7h",), **GOOD)(_detector)

    def test_bad_tier_raises(self):
        with temporary_registry():
            with pytest.raises(RegistryError, match="tier 5"):
                register("detector", name="x", tier=5, **GOOD)(_detector)

    def test_bad_callable_shape_raises(self):
        with temporary_registry():
            with pytest.raises(ContractError):
                register("detector", name="x", **GOOD)(lambda df, ctx: [])

    def test_returns_the_function_unchanged_and_still_callable(self):
        with temporary_registry():
            returned = register("detector", name="x", **GOOD)(_detector)
            assert returned is _detector
            assert returned(None, period=3) == []

    def test_rationale_is_stripped(self):
        with temporary_registry():
            register("detector", name="x", rationale="  a good reason  ")(_detector)
            assert REGISTRY["detector.x"].rationale == "a good reason"


class TestLookup:
    def test_get_missing_key_suggests_close_matches(self):
        load_all()
        with pytest.raises(RegistryError, match="donchian-breakout"):
            get("detector.donchian-breakou")

    def test_get_unknown_key_raises(self):
        with pytest.raises(RegistryError, match="unknown plug-in"):
            get("detector.no-such-thing-at-all")

    def test_by_kind_is_sorted_by_name(self):
        """The assertion is SORTEDNESS plus containment, not an exhaustive list.

        Widened in Phase 4, which legitimately adds detector.macd-cross: an
        exhaustive equality here makes every later phase that registers a plug-in
        edit this test, which trains people to edit it without thinking. The
        forward-compatible shape is the one TestLoadAll already uses
        (`PRODUCTION_KEYS <= set(reg)`).
        """
        load_all()
        names = list(by_kind("detector"))
        assert names == sorted(names)
        assert {"bollinger-fade", "donchian-breakout", "legacy-patterns"} <= set(names)

    def test_by_kind_unknown_kind_raises(self):
        with pytest.raises(RegistryError, match="unknown kind"):
            by_kind("indicator")

    def test_every_kind_is_queryable(self):
        load_all()
        for kind in KINDS:
            assert isinstance(by_kind(kind), dict)


class TestLoadAll:
    def test_returns_all_six_production_keys(self):
        reg = load_all()
        assert PRODUCTION_KEYS <= set(reg)

    def test_is_idempotent(self):
        """importlib caches modules, so the decorators run once and a second
        call is a no-op. If this ever raises 'already registered', something is
        importing a module under two names."""
        first = dict(load_all())
        second = dict(load_all())
        assert set(first) == set(second)

    def test_import_error_is_fatal_not_skipped(self, monkeypatch):
        """scripts/bruteforce/registry.py:164-167 — a family silently missing
        from a report reads as 'tested and found wanting'."""
        real = importlib.import_module
        target = "trading_bot.plugins.detectors.donchian"

        def boom(name, *a, **kw):
            if name == target:
                raise ImportError("synthetic failure")
            return real(name, *a, **kw)

        monkeypatch.setattr(registry.importlib, "import_module", boom)
        with pytest.raises(RegistryError, match=target.replace(".", r"\.")):
            load_all()

    def test_broken_package_import_is_fatal(self, monkeypatch):
        def boom(name, *a, **kw):
            raise ImportError("no such package")

        monkeypatch.setattr(registry.importlib, "import_module", boom)
        with pytest.raises(RegistryError, match="failed to import"):
            load_all("trading_bot.plugins")

    def test_underscore_prefixed_modules_are_skipped(self, tmp_path, monkeypatch):
        pkg = tmp_path / "throwaway_plugins"
        pkg.mkdir()
        (pkg / "__init__.py").write_text('"""throwaway."""\n')
        (pkg / "_private.py").write_text("raise RuntimeError('must not be imported')\n")
        (pkg / "visible.py").write_text(
            "from trading_bot.framework.registry import register\n"
            "@register('detector', name='throwaway-visible',\n"
            "          rationale='A throwaway plug-in used only to test load_all.')\n"
            "def v(ctx):\n"
            "    return []\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        with temporary_registry():
            reg = load_all("throwaway_plugins")
            assert "detector.throwaway-visible" in reg

    def test_default_package_is_config_driven(self):
        assert config.FRAMEWORK_PLUGIN_PACKAGE == "trading_bot.plugins"


class TestPluginSpec:
    def test_defaults(self):
        load_all()
        spec = get("detector.donchian-breakout")
        assert spec.defaults() == {
            "entry_period": config.DONCHIAN_ENTRY_PERIOD,
            "trend_period": config.DONCHIAN_TREND_PERIOD,
            "adx_period": config.ADX_PERIOD,
            "adx_min": config.ADX_TREND_THRESHOLD,
            "lookback_bars": config.PATTERN_LOOKBACK_BARS,
        }

    def test_resolve_none_equals_defaults(self):
        load_all()
        spec = get("detector.donchian-breakout")
        assert spec.resolve(None) == spec.defaults()
        assert spec.resolve({}) == spec.defaults()

    def test_resolve_unknown_key_lists_the_legal_ones(self):
        load_all()
        spec = get("detector.donchian-breakout")
        with pytest.raises(RegistryError, match="unknown parameter 'entryperiod'"):
            spec.resolve({"entryperiod": 20})

    def test_resolve_out_of_bounds_names_the_parameter(self):
        load_all()
        spec = get("detector.donchian-breakout")
        with pytest.raises(
            ContractError, match=r"detector\.donchian-breakout\.entry_period"
        ):
            spec.resolve({"entry_period": 9999})

    def test_resolve_coerces_int_valued_floats_where_legal(self):
        load_all()
        spec = get("policy.atr-stop-measured-move")
        got = spec.resolve({"atr_multiple": 2})
        assert isinstance(got["atr_multiple"], float) and got["atr_multiple"] == 2.0

    def test_combo_count_on_a_known_spec(self):
        load_all()
        # 5 bounded numeric axes at the default 3 probe points each.
        assert get("detector.donchian-breakout").combo_count() == 3**5
        # One 2-way choice axis.
        assert get("data.ohlcv").combo_count() == 2

    def test_combo_count_counts_bool_as_two(self):
        with temporary_registry():
            register(
                "detector",
                name="bool-axis",
                params={"on": ParamSpec(kind="bool", default=False, doc="on")},
                **GOOD,
            )(_detector)
            assert get("detector.bool-axis").combo_count() == 2

    def test_key_is_kind_dot_name(self):
        load_all()
        for key, spec in REGISTRY.items():
            assert key == f"{spec.kind}.{spec.name}"


class TestRegisteredPluginsAreWellFormed:
    """The meta-test every later phase inherits for free."""

    def test_every_registered_plugin_is_well_formed(self):
        load_all()
        assert REGISTRY, "load_all() registered nothing"
        for key, spec in REGISTRY.items():
            assert spec.rationale.strip(), f"{key}: empty rationale"
            assert len(spec.rationale) >= 40, (
                f"{key}: rationale is {len(spec.rationale)} chars; a one-word "
                f"rationale does not state a prior"
            )
            assert key == f"{spec.kind}.{spec.name}"
            assert spec.kind in KINDS
            assert _NAME_RE.match(spec.name), f"{key}: bad name casing"
            assert spec.tier is None or spec.tier in TIERS
            for pname, p in spec.params.items():
                assert isinstance(p, ParamSpec), f"{key}.{pname} is not a ParamSpec"
                assert p.doc.strip(), f"{key}.{pname} has no doc"
                assert p.is_legal(p.default), f"{key}.{pname} default is illegal"
            registry.check_callable_shape(spec.kind, spec.impl, key=key)

    def test_describe_is_json_shaped(self):
        load_all()
        d = registry.describe(get("detector.donchian-breakout"))
        assert d["key"] == "detector.donchian-breakout"
        assert d["tier"] == 2
        assert d["params"]["entry_period"]["bounds"] == [5, 200]
        assert d["dof"] == 3**5
