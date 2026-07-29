"""
Plug-in framework core (v0.3.0 Phase 3).

Five modules and one seam:

  errors.py     FrameworkError / ContractError / RegistryError / GraphError
  contracts.py  the seven Protocols, ParamSpec, the payload dataclasses, adapters
  registry.py   @register + REGISTRY + load_all()
  graph.py      StrategyGraph, NodeSpec, to_dict/from_dict, graph_hash
  context.py    EvalSession / EvalContext — no-lookahead bar access
  execute.py    run_graph_backtest() — THE graph->Trade seam (contract §5)

IMPORT DIRECTION IS ONE-WAY. Importing this package imports backtest.engine (via
context and execute), which imports every signal module — so `framework` must
NEVER be imported from engine.py, signals/*, indicators/*, regime/* or data/* or
the cycle closes. walkforward.py importing it is fine: nothing in framework
imports walkforward.
"""

from trading_bot.framework.errors import (  # noqa: F401
    ContractError,
    FrameworkError,
    GraphError,
    RegistryError,
)
from trading_bot.framework.contracts import (  # noqa: F401
    PARAM_KINDS,
    Confirmation,
    ConfirmationVerdict,
    DataSource,
    DetectedEvent,
    Detector,
    Filter,
    FilterVerdict,
    Mutator,
    ParamSpec,
    PositionPlan,
    PositionPolicy,
    Reviewer,
    candidate_from_event,
    check_callable_shape,
    event_from_candidate,
    plan_from_signal,
    plan_to_signal,
    trigger_from_meta,
    with_trigger,
)
from trading_bot.framework.registry import (  # noqa: F401
    KINDS,
    REGISTRY,
    TIERS,
    PluginSpec,
    by_kind,
    get,
    load_all,
    register,
    temporary_registry,
)
from trading_bot.framework.graph import (  # noqa: F401
    ANY_REGIME,
    SCHEMA_VERSION,
    Branch,
    ExitPolicySpec,
    NodeSpec,
    RegimeGate,
    StrategyGraph,
    TriggerSpec,
    canonical_dict,
    graph_hash,
    load,
    save,
    short_hash,
    validate,
)
from trading_bot.framework.context import (  # noqa: F401
    EvalContext,
    EvalSession,
    assert_trailing_only,
    cache_stats,
)
from trading_bot.framework.execute import clear_caches, run_graph_backtest  # noqa: F401
