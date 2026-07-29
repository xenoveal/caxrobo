"""
Framework exception hierarchy.

Every error class inherits BOTH FrameworkError and ValueError. The double
inheritance is deliberate: the repo's established convention is that a
programmer/config error raises ValueError (storage.py, engine._assert_interval
at engine.py:216, scripts/bruteforce/registry.py:130), and existing callers
and tests catch ValueError. Inheriting it means `except ValueError` keeps
working while `except FrameworkError` becomes possible, and no caller has to
learn a new base class to keep behaving correctly.
"""


class FrameworkError(Exception):
    """Base for every framework error. Never raised directly."""


class ContractError(FrameworkError, ValueError):
    """A plug-in or value violates a contract in framework/contracts.py.

    Raised at REGISTER time for a malformed callable or ParamSpec, and at
    RUN time when a plug-in returns the wrong type.
    """


class RegistryError(FrameworkError, ValueError):
    """A registration or lookup failed.

    Duplicate key, unknown kind, bad name casing, empty rationale, unknown
    plug-in key, or a fatal plug-in import during load_all().
    """


class GraphError(FrameworkError, ValueError):
    """A StrategyGraph is malformed.

    Raised when a graph references an unknown plug-in, carries an illegal
    parameter value, or was serialized under a different SCHEMA_VERSION.
    """
