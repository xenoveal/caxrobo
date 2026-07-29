"""
The seven plug-in contracts (v0.3.0 shared architecture contract §3).

UNIVERSAL CONVENTIONS, true of every type and every plug-in here:
  - All timestamps are epoch MILLISECONDS, UTC, candle OPEN time — the same
    convention as data/ohlcv.db and config.py's header. Never a formatted
    date string, never seconds.
  - All prices are float. All frames are pd.DataFrame indexed by ts
    (ascending) with columns open/high/low/close/volume.
  - Every contract is PURE and side-effect free except DataSource (reads
    SQLite) and Reviewer (writes review records).
  - Plug-ins are plain FUNCTIONS registered by decorator, not subclasses.
    The Protocols below document the call shape and support isinstance()
    for object-shaped contracts; @runtime_checkable checks member PRESENCE
    only, never signatures, so the six function-shaped kinds are validated
    at register time by check_callable_shape().

A1 — THERE IS NO INDICATOR CONTRACT. indicators/{wilder,bollinger,donchian}.py
stay pure pandas functions with no Protocol of their own; "migrated behind
the contracts" means every one of them is reachable from a strategy graph
only through a registered plug-in that declares its parameters as
ParamSpecs. The PRD's Phase 3 row implies an eighth contract; the shared
architecture contract defines seven, and it wins. An eighth Protocol would
have one implementation shape and no consumer.

GOTCHA, binding on every module in trading_bot: `ParamSpec` below SHADOWS
typing.ParamSpec (PEP 612). The name is fixed by contract §3 and must not be
renamed, so never write `import typing` with attribute access anywhere in this
package — import named symbols from typing only.
"""

import inspect
import math
import random  # noqa: F401 — referenced by the Mutator protocol's annotation
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import pandas as pd

from trading_bot.framework.errors import ContractError
from trading_bot.signals.breakout import BreakoutEvent
from trading_bot.signals.patterns import PatternCandidate
from trading_bot.signals.setup import Signal

if TYPE_CHECKING:  # runtime import would be circular / needlessly heavy
    from trading_bot.backtest.engine import Trade
    from trading_bot.framework.context import EvalContext
    from trading_bot.framework.graph import StrategyGraph

# Phase 5 owns the concrete ReviewRecord/ReviewContext (feedback/records.py).
# Aliased to Any here so the Reviewer protocol is expressible now WITHOUT
# Phase 3 pre-empting Phase 5's schema and WITHOUT Phase 5 needing to edit
# framework/, which is Phase 3's exclusively (contract §2).
ReviewContext: TypeAlias = Any
ReviewRecord: TypeAlias = Any

PARAM_KINDS = ("int", "float", "bool", "choice")


@dataclass(frozen=True)
class ParamSpec:
    """One declared, legal-bounded plug-in parameter.

    This replaces scripts/bruteforce/registry.py's ``grid: dict[str, list]``
    and is the single reason the grid becomes a spec: a list of values can
    be swept, but it cannot tell a UI what control to draw nor a mutator
    what the legal range is BETWEEN the listed values.

    Attributes:
        kind: One of PARAM_KINDS.
        default: The value used when a graph omits this parameter. MUST be
            legal under this spec — checked here, at import time, so a bad
            default is a startup failure rather than a silent mid-sweep one.
        bounds: (low, high) INCLUSIVE. Required for int/float, forbidden
            otherwise. This is what Phase 6's mutator jitters within.
        choices: Allowed values. Required for choice, forbidden otherwise.
        step: Control granularity for the UI and the mutation quantum for
            the mutator. Defaults to 1 for int, None for float.
        doc: One-line control label / tooltip. Required non-empty — an
            unlabelled knob in a builder UI is an invitation to sweep
            something nobody understands.
    """

    kind: str
    default: Any
    bounds: tuple[float, float] | None = None
    choices: tuple[Any, ...] | None = None
    step: float | None = None
    doc: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ContractError(
                f"ParamSpec kind {self.kind!r} is not one of {PARAM_KINDS}"
            )
        if not self.doc.strip():
            raise ContractError(
                f"ParamSpec(kind={self.kind!r}) needs a non-empty doc: an "
                f"unlabelled knob in a builder UI is an invitation to sweep "
                f"something nobody understands"
            )
        if self.bounds is not None and not isinstance(self.bounds, tuple):
            raise ContractError(
                f"ParamSpec bounds must be a TUPLE, got {type(self.bounds).__name__}; "
                f"a list makes the frozen dataclass unhashable and order-unstable"
            )
        if self.choices is not None and not isinstance(self.choices, tuple):
            raise ContractError(
                f"ParamSpec choices must be a TUPLE, got {type(self.choices).__name__}; "
                f"a list makes the frozen dataclass unhashable and order-unstable"
            )
        if self.kind in ("int", "float"):
            if self.bounds is None:
                raise ContractError(
                    f"ParamSpec(kind={self.kind!r}) requires bounds=(low, high): "
                    f"Phase 6's mutator has no legal range to jitter within without them"
                )
            low, high = self.bounds
            if not (low <= high):
                raise ContractError(
                    f"ParamSpec bounds {self.bounds!r} are inverted; expected low <= high"
                )
            if self.choices is not None:
                raise ContractError(
                    f"ParamSpec(kind={self.kind!r}) must not carry choices; "
                    f"use kind='choice' for an enumerated parameter"
                )
        elif self.kind == "choice":
            if not self.choices:
                raise ContractError(
                    "ParamSpec(kind='choice') requires a non-empty choices tuple"
                )
            if self.bounds is not None:
                raise ContractError(
                    "ParamSpec(kind='choice') must not carry bounds"
                )
        else:  # bool
            if self.bounds is not None or self.choices is not None:
                raise ContractError(
                    "ParamSpec(kind='bool') must not carry bounds or choices; "
                    "its legal set is exactly (False, True)"
                )
        if self.step is None and self.kind == "int":
            object.__setattr__(self, "step", 1)
        if self.step is not None and not (self.step > 0):
            raise ContractError(f"ParamSpec step must be > 0, got {self.step!r}")
        if not self.is_legal(self.default):
            raise ContractError(
                f"ParamSpec default {self.default!r} is illegal under "
                f"kind={self.kind!r} bounds={self.bounds!r} choices={self.choices!r}"
            )

    def is_legal(self, value: Any) -> bool:
        """Whether ``value`` is an admissible value for this parameter.

        Note the explicit bool rejection for int/float: bool is a subclass of
        int in Python, so without it `entry_period=True` would validate as the
        integer 1 and a graph could carry a boolean where a period belongs.
        NaN is rejected via math.isfinite rather than `v == v` so the intent
        reads plainly (mirrors setup.py:131-136's NaN-safe guard idiom).
        """
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                return False
            low, high = self.bounds  # type: ignore[misc]
            return low <= value <= high
        if self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False
            low, high = self.bounds  # type: ignore[misc]
            return low <= float(value) <= high
        if self.kind == "bool":
            return isinstance(value, bool)
        return value in (self.choices or ())

    def check(self, value: Any, *, where: str) -> Any:
        """Return ``value`` coerced to this spec's type, or raise.

        Args:
            value: The candidate value.
            where: Dotted location for the error message, e.g.
                "detector.donchian-breakout.entry_period".

        Returns:
            The value, coerced to float for kind="float" so 20 and 20.0 cannot
            produce two graph hashes for one strategy.

        Raises:
            ContractError: naming ``where``, the value, and the legal set.
        """
        if self.kind == "float" and self.is_legal(value):
            return float(value)
        if self.is_legal(value):
            return value
        legal = (
            f"bounds {self.bounds!r}"
            if self.bounds is not None
            else f"choices {self.choices!r}"
            if self.choices is not None
            else "(False, True)"
        )
        raise ContractError(
            f"{where}: {value!r} is not a legal {self.kind} value ({legal})"
        )

    def clamp(self, value: Any) -> Any:
        """Nearest legal value.

        int/float clamp to bounds and snap to step; choice falls back to the
        default when the value is not among the choices; bool coerces via
        bool(). Phase 6's mutator calls this so a jitter can never produce an
        illegal graph.
        """
        if self.kind == "bool":
            return bool(value)
        if self.kind == "choice":
            return value if value in (self.choices or ()) else self.default
        low, high = self.bounds  # type: ignore[misc]
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self.default
        if not math.isfinite(v):
            return self.default
        if self.step is not None:
            v = low + round((v - low) / self.step) * self.step
        v = min(max(v, low), high)
        if self.kind == "int":
            iv = int(round(v))
            return min(max(iv, int(math.ceil(low))), int(math.floor(high)))
        return float(v)


@dataclass(frozen=True)
class DetectedEvent:
    """A detector's structural finding, awaiting a trigger.

    DELIBERATELY field-compatible with signals.patterns.PatternCandidate
    (kind/direction/breakout_level->level/target_height/start_ts/end_ts) so
    migrating a v0.2.0 detector is an adapter, not a rewrite, and
    signals.breakout.check_breakout can be reused against it unchanged.

    Attributes:
        kind: Registry-style event kind, e.g. "donchian-breakout".
        direction: "long" or "short".
        level: Price a trigger bar must close beyond (PatternCandidate's
            breakout_level).
        target_height: Measured-move distance in price units.
        start_ts: Epoch-ms of the first bar forming the structure.
        end_ts: Epoch-ms the trigger bar must not predate.
        meta: Whatever the matching PositionPolicy needs that the six fixed
            fields cannot express — the fade policy's stop_level and target, a
            detector's ADX reading — plus the trigger facts the executor stamps
            in before calling confirmations and the policy (execute.py:
            trigger_ts / trigger_price / trigger_level / volume_ratio /
            volume_high). Values are floats: booleans are 0.0/1.0.
    """

    kind: str
    direction: str
    level: float
    target_height: float
    start_ts: int
    end_ts: int
    meta: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfirmationVerdict:
    """One Confirmation's answer about a candidate event.

    Attributes:
        passed: True lets the event through. A Confirmation NEVER mutates it.
        name: The confirmation's registry name, recorded on the trade.
        score: Optional strength reading, for reporting only.
        reason: Human-readable explanation, most useful when passed is False.
    """

    passed: bool
    name: str
    score: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class PositionPlan:
    """direction + entry + TP + SL derived from a confirmed event.

    Attributes:
        symbol: Trading pair symbol.
        ts: Epoch-ms of the trigger bar (becomes Trade.entry_ts).
        direction: "long" or "short".
        entry: Entry price (the trigger bar's close).
        stop: Stop-loss price.
        target: Take-profit price.
        risk_pct: |entry - stop| / entry.
        reward_pct: |target - entry| / entry.
        rr: reward_pct / risk_pct. Phase 4 persists this as Trade.planned_rr.
        source: The DetectedEvent.kind — becomes Trade.pattern, so
            metrics.by_bucket's "regime/pattern" keys are unchanged.
    """

    symbol: str
    ts: int
    direction: str
    entry: float
    stop: float
    target: float
    risk_pct: float
    reward_pct: float
    rr: float
    source: str


@dataclass(frozen=True)
class FilterVerdict:
    """One Filter's accept/reject decision on a PositionPlan.

    Attributes:
        accepted: True lets the plan through.
        name: The filter's registry name.
        reason: Why it was rejected (or accepted).
        measured: The numbers the decision was made on, so a rejection is
            auditable rather than a bare False.
    """

    accepted: bool
    name: str
    reason: str = ""
    measured: Mapping[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The seven Protocols (contract §3). @runtime_checkable supports isinstance()
# but checks member PRESENCE only — never signatures — so only DataSource,
# which is object-shaped, is isinstance-checked at run time. The six
# function-shaped kinds are validated at REGISTER time by
# check_callable_shape() below.
# --------------------------------------------------------------------------- #


@runtime_checkable
class DataSource(Protocol):
    """Bars/series by symbol + timeframe. MVP implementation is OHLCV over storage.py."""

    def frame(
        self,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None,
        end_ms: int | None,
    ) -> pd.DataFrame: ...

    def timeframes(self) -> tuple[str, ...]: ...


@runtime_checkable
class Detector(Protocol):
    """Pattern/structure events. Returns PatternCandidate-compatible events."""

    def detect(self, ctx: "EvalContext", **params) -> list[DetectedEvent]: ...


@runtime_checkable
class Confirmation(Protocol):
    """Gate on a candidate event. True = let it through. NEVER mutates the event."""

    def confirm(
        self, ctx: "EvalContext", event: DetectedEvent, **params
    ) -> ConfirmationVerdict: ...


@runtime_checkable
class PositionPolicy(Protocol):
    """direction + entry + TP + SL from a confirmed event."""

    def decide(
        self, ctx: "EvalContext", event: DetectedEvent, **params
    ) -> PositionPlan | None: ...


@runtime_checkable
class Filter(Protocol):
    """Accept/reject a PositionPlan. The R:R-after-costs filter lives here."""

    def accept(
        self, ctx: "EvalContext", plan: PositionPlan, **params
    ) -> FilterVerdict: ...


@runtime_checkable
class Reviewer(Protocol):
    """Closed-trade analysis -> a persisted ReviewRecord (Phase 5 owns both types)."""

    def review(
        self, trade: "Trade", context: ReviewContext, **params
    ) -> ReviewRecord: ...


@runtime_checkable
class Mutator(Protocol):
    """Strategy-variant generation for the evolution engine (Phase 6)."""

    def mutate(
        self, graph: "StrategyGraph", rng: "random.Random", **params
    ) -> "StrategyGraph": ...


# --------------------------------------------------------------------------- #
# The four adapters — the migration seam.
# --------------------------------------------------------------------------- #


def event_from_candidate(
    candidate: PatternCandidate, *, meta: Mapping[str, float] | None = None
) -> DetectedEvent:
    """PatternCandidate -> DetectedEvent.

    breakout_level -> level; every other field carries across by name.
    """
    return DetectedEvent(
        kind=candidate.kind,
        direction=candidate.direction,
        level=float(candidate.breakout_level),
        target_height=float(candidate.target_height),
        start_ts=int(candidate.start_ts),
        end_ts=int(candidate.end_ts),
        meta=dict(meta or {}),
    )


def candidate_from_event(event: DetectedEvent) -> PatternCandidate:
    """DetectedEvent -> PatternCandidate.

    So signals.breakout.check_breakout works against it unchanged.
    check_breakout reads only breakout_level, direction and end_ts
    (breakout.py:104, 113, 124-127), so this round-trip is lossless for
    everything the trigger consumes. ``meta`` is dropped — PatternCandidate has
    nowhere to put it and check_breakout never reads it.
    """
    return PatternCandidate(
        kind=event.kind,
        direction=event.direction,
        breakout_level=float(event.level),
        target_height=float(event.target_height),
        start_ts=int(event.start_ts),
        end_ts=int(event.end_ts),
    )


# The trigger facts the executor stamps into DetectedEvent.meta once a trigger
# bar fires, and the ONLY channel by which a Confirmation, PositionPolicy or
# Filter learns about the bar under evaluation (EvalContext deliberately
# truncates it away — see context.py's ROLES block). Booleans are 0.0/1.0
# because meta values are floats.
TRIGGER_META_KEYS = (
    "trigger_ts",
    "trigger_price",
    "trigger_level",
    "volume_ratio",
    "volume_high",
)


def with_trigger(event: DetectedEvent, trigger: BreakoutEvent) -> DetectedEvent:
    """Return a copy of ``event`` with the trigger bar's facts in ``meta``.

    A new object, never a mutation: DetectedEvent is frozen and Confirmations
    are contractually forbidden from changing an event, so stamping in place
    would be the one write that breaks that promise.
    """
    meta = dict(event.meta)
    meta.update(
        {
            "trigger_ts": float(trigger.ts),
            "trigger_price": float(trigger.price),
            "trigger_level": float(trigger.level),
            "volume_ratio": float(trigger.volume_ratio),
            "volume_high": 1.0 if trigger.volume_high else 0.0,
        }
    )
    return DetectedEvent(
        kind=event.kind,
        direction=event.direction,
        level=event.level,
        target_height=event.target_height,
        start_ts=event.start_ts,
        end_ts=event.end_ts,
        meta=meta,
    )


def trigger_from_meta(event: DetectedEvent) -> BreakoutEvent:
    """Rebuild the BreakoutEvent with_trigger() stamped into ``event.meta``.

    Lossless: BreakoutEvent carries exactly ts / price / level / direction /
    volume_ratio / volume_high, and direction comes from the event itself.

    Raises:
        ContractError: If the event was never stamped — which means a policy or
            confirmation ran before the trigger fired, an executor bug.
    """
    missing = [k for k in TRIGGER_META_KEYS if k not in event.meta]
    if missing:
        raise ContractError(
            f"event {event.kind!r} has no trigger facts in meta (missing {missing}); "
            f"a policy or confirmation ran before the trigger fired"
        )
    return BreakoutEvent(
        ts=int(event.meta["trigger_ts"]),
        price=float(event.meta["trigger_price"]),
        level=float(event.meta["trigger_level"]),
        direction=event.direction,
        volume_ratio=float(event.meta["volume_ratio"]),
        volume_high=bool(event.meta["volume_high"]),
    )


def plan_to_signal(plan: PositionPlan, event: BreakoutEvent) -> Signal:
    """PositionPlan + trigger event -> signals.setup.Signal (A9).

    Signal is the executor's internal currency, so ties break by the identical
    (-rr, pattern, direction) key setup.rank_signals uses and Trade
    construction is field-for-field the same shape as engine.close_out.

    The trigger event is a REQUIRED argument rather than optional because
    volume_ratio/volume_high live only on it: PositionPlan has no volume
    fields, and defaulting them would silently zero Trade.volume_high and
    break parity in a way no arithmetic check would catch.
    ``Signal.pattern = plan.source``, so metrics' "regime/pattern" buckets
    (metrics.py:31) are unchanged.
    """
    return Signal(
        symbol=plan.symbol,
        ts=plan.ts,
        direction=plan.direction,
        pattern=plan.source,
        entry=plan.entry,
        stop=plan.stop,
        target=plan.target,
        risk_pct=plan.risk_pct,
        reward_pct=plan.reward_pct,
        rr=plan.rr,
        volume_ratio=event.volume_ratio,
        volume_high=event.volume_high,
    )


def plan_from_signal(sig: Signal, *, source: str) -> PositionPlan:
    """Signal -> PositionPlan.

    For policies that wrap a v0.2.0 builder which already returns a Signal
    (setup.build_signal, meanrev.build_fade_signal).
    ``plan_to_signal(plan_from_signal(s, source=s.pattern), event) == s`` for
    every field, given the event that produced s — pinned by
    test_framework_contracts.py::TestAdapters.
    """
    return PositionPlan(
        symbol=sig.symbol,
        ts=sig.ts,
        direction=sig.direction,
        entry=sig.entry,
        stop=sig.stop,
        target=sig.target,
        risk_pct=sig.risk_pct,
        reward_pct=sig.reward_pct,
        rr=sig.rr,
        source=source,
    )


# --------------------------------------------------------------------------- #
# Register-time structural check.
# --------------------------------------------------------------------------- #

_EXPECTED_FIRST_PARAMS = {
    "data": ("conn",),
    "detector": ("ctx",),
    "confirmation": ("ctx", "event"),
    "policy": ("ctx", "event"),
    "filter": ("ctx", "plan"),
    "reviewer": ("trade", "context"),
    "mutator": ("graph", "rng"),
}


def check_callable_shape(kind: str, fn: Callable, *, key: str) -> None:
    """Verify a plug-in's leading positional parameters are named as its
    contract requires, and that every remaining parameter is keyword-capable.

    @runtime_checkable Protocols check member PRESENCE only, never
    signatures, so isinstance() would accept a detector whose first argument
    is a DataFrame. This is the cheap structural substitute, run once at
    import time. It checks NAMES, not types (no annotations are required
    anywhere in this repo), which is enough to catch the mistakes that
    actually happen: wrong argument order and a forgotten ctx.

    Args:
        kind: One of registry.KINDS.
        fn: The plug-in callable.
        key: "<kind>.<name>", for the error message.

    Raises:
        ContractError: naming ``key``, the expected leading parameters, and
            what was found.
    """
    expected = _EXPECTED_FIRST_PARAMS.get(kind)
    if expected is None:
        raise ContractError(f"{key}: unknown plug-in kind {kind!r}")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:  # pragma: no cover - exotic callables
        raise ContractError(f"{key}: signature is not introspectable: {exc}") from exc

    params = list(sig.parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    found = tuple(p.name for p in positional)
    if found[: len(expected)] != expected:
        raise ContractError(
            f"{key}: a {kind} plug-in must take {expected} as its leading "
            f"positional parameters, found {found or '()'}. Names are checked "
            f"(not types) because argument ORDER is the mistake that actually happens."
        )
    if len(positional) > len(expected):
        raise ContractError(
            f"{key}: parameters {found[len(expected):]} must be KEYWORD-ONLY "
            f"(declare them after a bare `*`) — the executor resolves plug-in "
            f"parameters from the graph by name, never by position"
        )
    for p in params[len(expected) :]:
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ContractError(
                f"{key}: parameter {p.name!r} is positional-only; plug-in "
                f"parameters must be reachable by keyword"
            )
