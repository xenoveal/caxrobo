"""
Shared pytest helpers for the whole test tree.

OWNERSHIP AND THE ONE RULE. This file is Phase 8's (v0.3.0 shared architecture
contract §8, assigned 2026-07-27 after the cross-plan audit found it unowned).
It is deliberately PURELY ADDITIVE: no `autouse` fixture, no `pytest_*` hook, no
collection or reporting customisation. Nothing here changes the behaviour of a
single pre-existing test, and a run of the suite with this file deleted differs
only in that the Phase 8 detector modules cannot collect.

Any later phase that needs a conftest hook APPENDS to this file. Never create a
second `conftest.py` and never rewrite this one — a broken conftest breaks every
test run for every phase at once, which is the exact reason it has an owner.

WHAT IS HERE
  * `pattern_fixture` — loads a hand-drawn OHLCV CSV from
    tests/fixtures/patterns/ and asserts its index is a clean, evenly-spaced
    setup-tier series.
  * `setup_context` / `make_setup_context` — build a real
    `framework.context.EvalContext` over an in-memory frame, so a detector test
    exercises the plug-in through exactly the surface the executor gives it.
    Four Phase 8 test modules need this; putting it in each of them would be
    four copies of a no-lookahead harness, which is the last thing that should
    be duplicated. (A deviation from the plan's "exactly one fixture", recorded
    in the phase report.)
"""

from pathlib import Path

import pandas as pd
import pytest

from trading_bot import config
from trading_bot.data import storage
from trading_bot.framework.context import EvalSession
from trading_bot.framework.graph import RegimeGate

PATTERN_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "patterns"

_SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
_REGIME_TF = config.REGIME_TIMEFRAME
_TRIGGER_TF = config.SIGNAL_TRIGGER_TIMEFRAME


def load_pattern_fixture(name: str) -> pd.DataFrame:
    """Read one hand-drawn pattern CSV into an OHLCV frame.

    The CSV carries `ts,open,high,low,close,volume` with `#` comment lines; by
    convention its FIRST line is a comment stating the shape and the expected
    verdict, so the expected outcome is derivable by reading the numbers rather
    than by running the detector (which would be circular).

    Asserts the index is strictly increasing and evenly spaced by
    `storage.TIMEFRAME_MS[config.SIGNAL_PATTERN_TIMEFRAME]`. A hand-edited CSV
    that silently introduces a gap makes every bar-count tolerance meaningless,
    and the step is derived from config so a future tier shift cannot leave
    these fixtures green on the old timeframe (contract §8).

    Args:
        name: File name inside tests/fixtures/patterns/.

    Returns:
        DataFrame indexed by int epoch-ms ts.
    """
    path = PATTERN_FIXTURE_DIR / name
    df = pd.read_csv(path, comment="#")
    df["ts"] = df["ts"].astype("int64")
    df = df.set_index("ts")
    step = storage.TIMEFRAME_MS[_SETUP_TF]
    ts = df.index.to_numpy()
    assert len(ts) >= 2, f"{name}: a pattern fixture needs at least two bars"
    diffs = set((ts[1:] - ts[:-1]).tolist())
    assert diffs == {step}, (
        f"{name}: index steps {sorted(diffs)} but the setup tier "
        f"({_SETUP_TF}) is {step} ms. A gap in a hand-drawn fixture makes every "
        f"bar-count tolerance in the detector meaningless."
    )
    return df


@pytest.fixture
def pattern_fixture():
    """Factory fixture: `pattern_fixture("cup_and_handle_positive.csv") -> df`."""
    return load_pattern_fixture


def make_setup_context(df: pd.DataFrame, *, symbol: str = "BTCUSDT", now_ms=None):
    """An `EvalContext` whose setup tier is exactly ``df``.

    Builds a real `EvalSession` — not a stub — so a detector under test sees the
    same truncation, the same `engine._assert_interval` check and the same
    pivot-confirmation guarantee the executor gives it. The regime and trigger
    tiers are minimal correctly-spaced two-bar frames: no Phase 8 detector reads
    them, but `EvalSession` loads all three tiers and asserts each one's
    spacing, so they have to be real.

    Args:
        df: Setup-tier OHLCV frame, epoch-ms index spaced at the setup interval.
        symbol: Symbol name, for error messages only.
        now_ms: Evaluation instant. Defaults to the LAST setup bar's CLOSE, i.e.
            every bar in ``df`` is closed and visible — which is what a
            geometry fixture means by "as of the end of this frame".

    Returns:
        EvalContext bound to role "setup".
    """
    setup_step = storage.TIMEFRAME_MS[_SETUP_TF]
    if now_ms is None:
        now_ms = int(df.index[-1]) + setup_step

    def _pad(timeframe: str) -> pd.DataFrame:
        step = storage.TIMEFRAME_MS[timeframe]
        start = int(df.index[0])
        rows = [
            (start + i * step, 100.0, 100.0, 100.0, 100.0, 1.0)
            for i in range(2)
        ]
        pad = pd.DataFrame(
            rows, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        return pad.set_index("ts")

    frames = {_SETUP_TF: df}

    class _Source:
        def frame(self, sym, timeframe, *, start_ms=None, end_ms=None):
            return frames.get(timeframe) if timeframe in frames else _pad(timeframe)

        def timeframes(self):
            return (_REGIME_TF, _SETUP_TF, _TRIGGER_TF)

    session = EvalSession(
        _Source(),
        symbol,
        tiers=(_REGIME_TF, _SETUP_TF, _TRIGGER_TF),
        regime_gate=RegimeGate(enabled=False),
    )
    return session.context("setup", now_ms)


@pytest.fixture
def setup_context():
    """Factory fixture wrapping `make_setup_context`."""
    return make_setup_context


def detect_events(key: str, df: pd.DataFrame, **overrides):
    """Run the REGISTERED detector ``key`` over ``df`` and return its events.

    Resolves the plug-in out of `REGISTRY` **by key** rather than importing the
    function, so a missing or misspelled `@register` fails loudly instead of the
    test quietly exercising an unregistered function. Parameters go through
    `PluginSpec.resolve`, so an override that is not a declared ParamSpec — or is
    out of its bounds — raises rather than being silently ignored.
    """
    from trading_bot.framework import registry

    registry.load_all()
    spec = registry.get(key)
    return spec.impl(make_setup_context(df), **spec.resolve(overrides))
