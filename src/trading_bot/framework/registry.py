"""
Plug-in registry: the @register decorator, REGISTRY, and load_all().

Direct descendant of scripts/bruteforce/registry.py, and its four hard-won
lessons are carried forward deliberately rather than rediscovered:

  1. `rationale` is MANDATORY and non-empty. Quoting the ancestor: "an
     unmotivated strategy in a 10,000-combo sweep is just noise with a name,
     and the report needs to state the prior." Under Phase 6's population this
     matters more, not less.
  2. Duplicate names RAISE, naming the module that already claimed the key.
  3. Import errors during load_all() are FATAL, never skipped — "a family
     silently missing from the leaderboard would read as 'tested and found
     wanting'."
  4. Trial counting is first-class. Here it is `PluginSpec.combo_count()` over
     ParamSpec bounds/choices, reported by `cli.py plugins`, so the degrees of
     freedom a graph exposes are visible BEFORE Phase 6 spends them. It is NOT
     a DSR trial count: the ledger (backtest/trials.py) counts evaluations
     actually performed (contract §4).
"""

import contextlib
import difflib
import importlib
import logging
import pkgutil
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trading_bot import config
from trading_bot.data import storage
from trading_bot.framework.contracts import ParamSpec, check_callable_shape
from trading_bot.framework.errors import RegistryError

logger = logging.getLogger("trading_bot")

KINDS = ("data", "detector", "confirmation", "policy", "filter", "reviewer", "mutator")
# Registry names are lowercase-hyphen, matching DONCHIAN_KIND = "donchian-breakout"
# and PATTERN_KINDS already in the codebase (contract §3). Never camelCase,
# never snake_case, never Leading-Caps.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TIERS = (1, 2, 3, 4)  # contract §9 reliability tiers


@dataclass(frozen=True)
class PluginSpec:
    """One registered plug-in and everything the framework knows about it.

    Attributes:
        kind: One of KINDS.
        name: lowercase-hyphen identifier, unique within its kind.
        key: f"{kind}.{name}" — the registry key and the graph's node reference.
        impl: The registered callable, returned UNCHANGED by the decorator so it
            stays directly callable and unit-testable.
        params: Declared parameters, name -> ParamSpec.
        rationale: WHY this should work. Mandatory and non-empty.
        timeframes: Timeframes the plug-in reads, for reporting.
        tier: contract §9 reliability tier, or None when the concept does not
            apply (a data source has no reliability tier).
        module: impl.__module__, so a duplicate-key error can name the claimant.
    """

    kind: str
    name: str
    key: str
    impl: Callable
    params: Mapping[str, ParamSpec] = field(default_factory=dict)
    rationale: str = ""
    timeframes: tuple[str, ...] = ()
    tier: int | None = None
    module: str = ""

    def defaults(self) -> dict:
        """Every declared parameter at its ParamSpec default."""
        return {name: spec.default for name, spec in self.params.items()}

    def resolve(self, overrides: Mapping | None = None) -> dict:
        """defaults() merged with ``overrides``, every value ParamSpec-checked.

        Unknown keys raise, listing the legal ones — a typo'd parameter must
        never be silently ignored, which is how a sweep ends up measuring the
        default 200 times.

        Raises:
            RegistryError: On an unknown parameter name.
            ContractError: On an illegal value (from ParamSpec.check).
        """
        out = self.defaults()
        for name, value in (overrides or {}).items():
            if name not in self.params:
                raise RegistryError(
                    f"{self.key}: unknown parameter {name!r}; declared "
                    f"parameters are {sorted(self.params) or '(none)'}. A typo'd "
                    f"parameter must never be silently ignored — that is how a "
                    f"sweep ends up measuring the default N times."
                )
            out[name] = self.params[name].check(value, where=f"{self.key}.{name}")
        # Re-check the untouched defaults too, so a graph is always fully legal.
        for name, spec in self.params.items():
            out[name] = spec.check(out[name], where=f"{self.key}.{name}")
        return out

    def combo_count(self, *, points: int = 3) -> int:
        """Rough declared degrees of freedom.

        len(choices) per choice axis, 2 per bool axis, ``points`` per bounded
        numeric axis. Reported by `cli.py plugins`, NEVER used as a DSR trial
        count — the ledger counts evaluations actually performed (contract §4).
        """
        n = 1
        for spec in self.params.values():
            if spec.kind == "bool":
                n *= 2
            elif spec.kind == "choice":
                n *= max(1, len(spec.choices or ()))
            else:
                low, high = spec.bounds  # type: ignore[misc]
                n *= 1 if low == high else points
        return n


REGISTRY: dict[str, PluginSpec] = {}


def register(
    kind: str,
    *,
    name: str,
    params: Mapping[str, ParamSpec] | None = None,
    rationale: str,
    timeframes: tuple[str, ...] = (),
    tier: int | None = None,
) -> Callable:
    """Decorator registering a plain function as a framework plug-in.

    Returns the function UNCHANGED (prior art: scripts/bruteforce/registry.py:151)
    so the plug-in stays directly callable and unit-testable without the registry.

    Args:
        kind: One of KINDS.
        name: lowercase-hyphen identifier, unique within the whole registry
            once prefixed by kind.
        params: Declared parameters, name -> ParamSpec.
        rationale: WHY this should work, in one or two sentences. Required.
        timeframes: Timeframes the plug-in reads (keys of storage.TIMEFRAME_MS).
        tier: contract §9 reliability tier, or None.

    Raises:
        RegistryError: On an unknown kind, a badly-cased name, an empty
            rationale, a non-ParamSpec parameter, an unknown timeframe, an
            out-of-range tier, or a duplicate key.
        ContractError: When the callable's leading parameters do not match the
            contract for ``kind`` (from contracts.check_callable_shape).
    """
    params = dict(params or {})

    def deco(fn: Callable) -> Callable:
        key = f"{kind}.{name}"
        if kind not in KINDS:
            raise RegistryError(f"{key}: unknown kind {kind!r}; expected one of {KINDS}")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise RegistryError(
                f"{key}: name {name!r} must be lowercase-hyphen, e.g. "
                f"'donchian-breakout'. camelCase, snake_case, Leading-Caps and a "
                f"trailing hyphen are all rejected (contract §3)."
            )
        if key in REGISTRY:
            raise RegistryError(
                f"{key!r} is already registered (by {REGISTRY[key].module}); "
                f"pick a distinct name"
            )
        if not rationale.strip():
            raise RegistryError(
                f"{key}: a rationale is required — an unmotivated plug-in in a "
                f"population search is just noise with a name"
            )
        for pname, spec in params.items():
            if not isinstance(spec, ParamSpec):
                raise RegistryError(
                    f"{key}: parameter {pname!r} must be a ParamSpec, got "
                    f"{type(spec).__name__}; a bare default cannot tell the UI "
                    f"what control to draw nor the mutator what is legal"
                )
        for tf in timeframes:
            if tf not in storage.TIMEFRAME_MS:
                raise RegistryError(
                    f"{key}: unknown timeframe {tf!r}; expected one of "
                    f"{tuple(storage.TIMEFRAME_MS)}"
                )
        if tier is not None and tier not in TIERS:
            raise RegistryError(f"{key}: tier {tier!r} must be None or one of {TIERS}")

        check_callable_shape(kind, fn, key=key)

        REGISTRY[key] = PluginSpec(
            kind=kind,
            name=name,
            key=key,
            impl=fn,
            params=params,
            rationale=rationale.strip(),
            timeframes=tuple(timeframes),
            tier=tier,
            module=getattr(fn, "__module__", ""),
        )
        return fn

    return deco


def get(key: str) -> PluginSpec:
    """Look one plug-in up by "<kind>.<name>".

    Raises:
        RegistryError: When the key is unknown, with close-match suggestions —
            a mistyped detector name in a serialized graph is otherwise a very
            confusing "produces no trades".
    """
    if key in REGISTRY:
        return REGISTRY[key]
    close = difflib.get_close_matches(key, sorted(REGISTRY), n=3)
    hint = f"; did you mean {close}?" if close else ""
    raise RegistryError(
        f"unknown plug-in {key!r}{hint} ({len(REGISTRY)} registered; call "
        f"registry.load_all() first if you have not)"
    )


def by_kind(kind: str) -> dict[str, PluginSpec]:
    """Every plug-in of one kind, keyed by NAME (not key), sorted by name.

    Sorted so `cli.py plugins` output is stable and diffable.

    Raises:
        RegistryError: On an unknown kind.
    """
    if kind not in KINDS:
        raise RegistryError(f"unknown kind {kind!r}; expected one of {KINDS}")
    return {
        spec.name: spec
        for spec in sorted(
            (s for s in REGISTRY.values() if s.kind == kind), key=lambda s: s.name
        )
    }


def load_all(package: str | None = None) -> dict[str, PluginSpec]:
    """Import every module under ``package`` so its decorators run.

    Idempotent: importlib caches modules, so the decorators run once and a
    second call is a no-op. If a second call ever raises "already registered",
    something is importing a module under two names.

    Import errors are FATAL rather than skipped: a family silently missing from
    a report would read as "tested and found wanting".

    Args:
        package: Dotted package name (default config.FRAMEWORK_PLUGIN_PACKAGE).

    Returns:
        REGISTRY itself.

    Raises:
        RegistryError: If the package or any module under it fails to import.
    """
    pkg_name = config.FRAMEWORK_PLUGIN_PACKAGE if package is None else package
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception as exc:  # noqa: BLE001 — fatal by design
        raise RegistryError(
            f"plug-in package {pkg_name!r} failed to import; import errors are "
            f"fatal, never skipped, because a silently missing family reads as "
            f"'tested and found wanting'"
        ) from exc

    # walk_packages, not iter_modules: plug-ins live at plugins/<sub>/<mod>.py.
    # It IMPORTS each package to read __path__, so a broken subpackage
    # __init__.py raises there rather than at the leaf — hence the module name
    # in the wrapped message, without which the traceback is unreadable.
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if mod.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 — fatal by design
            raise RegistryError(
                f"plug-in module {mod.name!r} failed to import; import errors "
                f"are fatal, never skipped, because a silently missing family "
                f"reads as 'tested and found wanting'"
            ) from exc
    return REGISTRY


@contextlib.contextmanager
def temporary_registry():
    """Snapshot REGISTRY, yield, restore. For tests that register throwaway plug-ins.

    Deliberately a snapshot/restore rather than a clear(): emptying the
    registry cannot be undone by re-importing, because importlib caches
    modules and the decorators would never run again — a clear() would leave
    every subsequent test in the session looking at an empty registry and
    failing for a reason unrelated to itself.
    """
    saved: dict[str, PluginSpec] = dict(REGISTRY)
    try:
        yield REGISTRY
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)


def describe(spec: PluginSpec) -> dict[str, Any]:
    """One plug-in as a plain dict, for `cli.py plugins` and Phase 7's UI."""
    return {
        "key": spec.key,
        "kind": spec.kind,
        "name": spec.name,
        "tier": spec.tier,
        "timeframes": list(spec.timeframes),
        "params": {
            pname: {
                "kind": p.kind,
                "default": p.default,
                "bounds": list(p.bounds) if p.bounds is not None else None,
                "choices": list(p.choices) if p.choices is not None else None,
                "step": p.step,
                "doc": p.doc,
            }
            for pname, p in spec.params.items()
        },
        "rationale": spec.rationale,
        "dof": spec.combo_count(),
        "module": spec.module,
    }
