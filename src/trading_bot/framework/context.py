"""
EvalContext: a read-only, time-bounded view of market data for ONE
evaluation instant, and EvalSession, which owns the frames and the memos.

Every no-lookahead guarantee engine.py documents at engine.py:16-52 is
preserved here STRUCTURALLY rather than by convention. A plug-in author
cannot see the future by accident, and cannot see it on purpose without
obviously reaching around this class:

  1. A plug-in never receives `conn` or a full frame. It receives an
     EvalContext and nothing else.
  2. The master frames live on the EvalSession in a private attribute; the
     EvalContext's public API has no accessor that returns one. Every
     frame/series accessor slices at now_ms INSIDE the context.
  3. Returned numpy arrays are zero-copy slices with flags.writeable
     = False; returned DataFrames are pandas copy-on-write slices, so a
     plug-in cannot mutate what the next plug-in sees.
  4. now_ms is set by the executor from BAR TIMES only. There is no setter and
     no wall clock anywhere in this module — nothing calls time.time().
  5. now_ms is non-decreasing PER ROLE ("regime"/"setup"/"trigger"), which
     catches an executor bug that walks backwards. Per-role rather than
     global because a setup-bar context legitimately trails the previous
     trigger-bar context.
  6. Every loaded series passes engine._assert_interval — the SAME function,
     imported, with the same error text — so a partial tier migration
     cannot fail silently in the flattering direction.
  7. series() factories must be TRAILING-ONLY (value at i depends on bars
     [0..i]). assert_trailing_only() proves it for a factory; truncation is
     the second line of defence, not the first.
  8. regime() is the single implementation of the "last regime bar CLOSED by
     t" rule (engine.py:315-317). A plug-in cannot roll its own from a
     regime frame, because frame(regime_tf) is truncated too.
  9. Pivot confirmation is INHERITED from truncation: find_pivots on a frame
     ending at the last closed bar can only emit pivots with a full
     PIVOT_SPAN window on both sides (pivots.py:7-13). Nothing extra is
     needed, and nothing may bypass the truncation to "get one more pivot".
 10. Warmup NaNs are never filled. A plug-in that gets NaN during warmup
     must return no event, exactly as detect_donchian_setups does
     (donchian.py:112-113).

ROLES AND THEIR now_ms, fixed here because parity depends on it:
  - "setup"   -> the setup bar's CLOSE time. bar_index(setup_tf) is then
                 exactly engine.py:337's h_idx and regime() is exactly
                 engine.py:338's regime_at(close_setup[h_idx]).
  - "trigger" -> the trigger bar's OPEN time (ts_trig[j]), NOT its close.
                 This is MEDIUM-2 (engine.py:458-481) expressed as a context:
                 bar_index(setup_tf) is then the setup bar that had already
                 closed when the trigger bar OPENED, so a policy reading
                 ctx.series(setup_tf, "atr", ...)[-1] gets exactly
                 engine.py:492's atr_setup_vals[h_idx] BY CONSTRUCTION rather
                 than by a tier-boundary coincidence. The trigger bar under
                 evaluation is therefore absent from frame(trigger_tf); the
                 executor stamps that bar's facts into DetectedEvent.meta,
                 which is the only channel by which a plug-in learns them.
"""

import logging

import numpy as np
import pandas as pd

from trading_bot import config
from trading_bot.backtest.engine import _assert_interval, _fingerprint
from trading_bot.data import storage
from trading_bot.framework.errors import ContractError
from trading_bot.indicators.wilder import atr as wilder_atr
from trading_bot.regime.classifier import classify_series
from trading_bot.signals.donchian import channel_exit_levels

logger = logging.getLogger("trading_bot")

# Process-wide memo, mirroring engine._CACHE's role and keying discipline: every
# entry is keyed by a CONTENT fingerprint (engine._fingerprint), never by
# (symbol, timeframe) alone, so two fixtures on separate databases cannot be
# served each other's values.
_CACHE: dict = {}
_STATS = {"hits": 0, "misses": 0}


def clear_caches() -> None:
    """Drop every memoized series, channel, label set and candidate list.

    Cheap and always safe. Phase 6 should call this between generations: the
    candidate memo grows without bound across a population run, and
    cache_stats() is how that is measured rather than guessed.
    """
    _CACHE.clear()
    _STATS["hits"] = 0
    _STATS["misses"] = 0


def cache_stats() -> dict:
    """Measured memo statistics, so performance claims are numbers not adjectives.

    Returns:
        dict with `entries` (top-level keys), `hits`, `misses`, `hit_rate`, and
        `approx_bytes` (summed nbytes of cached numpy arrays; dict-valued
        candidate memos are counted as entries, not bytes).
    """
    total = _STATS["hits"] + _STATS["misses"]
    approx = 0
    for value in _CACHE.values():
        if isinstance(value, np.ndarray):
            approx += int(value.nbytes)
        elif isinstance(value, tuple):
            approx += sum(int(v.nbytes) for v in value if isinstance(v, np.ndarray))
    return {
        "entries": len(_CACHE),
        "hits": _STATS["hits"],
        "misses": _STATS["misses"],
        "hit_rate": (_STATS["hits"] / total) if total else None,
        "approx_bytes": approx,
    }


def _memo(key, produce):
    """Look ``key`` up, computing with ``produce`` on a miss.

    config.FRAMEWORK_CACHE_ENABLED is read AT CALL TIME, not captured at import:
    it is a measurement switch, and a run with caching off must be bit-identical
    to one with it on (asserted in tests/test_framework_parity.py).
    """
    if not config.FRAMEWORK_CACHE_ENABLED:
        _STATS["misses"] += 1
        return produce()
    if key in _CACHE:
        _STATS["hits"] += 1
        return _CACHE[key]
    _STATS["misses"] += 1
    value = produce()
    _CACHE[key] = value
    return value


def _readonly(arr) -> np.ndarray:
    """Return ``arr`` as a numpy array that cannot be written through.

    Set on the CACHED array as well as the handed-out slice: cache immutable,
    hand out immutable views, so one plug-in cannot poison the next.
    """
    out = np.asarray(arr)
    try:
        out.flags.writeable = False
    except ValueError:  # pragma: no cover - a non-owning view already frozen
        pass
    return out


def assert_trailing_only(factory, df: pd.DataFrame, *, sample: int = 8, **params) -> None:
    """Prove a series factory is causal: value at i depends only on bars [0..i].

    Recomputes the factory on truncated frames and compares against the
    full-history value at the same index. This is the FIRST line of defence;
    EvalContext's truncation is the second and cannot save a non-causal factory
    (a centred rolling window poisons arr[i] itself). Port of the idea behind
    scripts/bruteforce/core.assert_causal.

    Args:
        factory: factory(df, **params) -> np.ndarray | pd.Series.
        df: A frame long enough to have defined values.
        sample: How many trailing indices to probe.
        **params: Passed through to the factory.

    Raises:
        ContractError: naming the FIRST offending index and both values.
    """
    full = np.asarray(pd.Series(factory(df, **params)).to_numpy(), dtype=float)
    n = len(full)
    if n == 0:
        return
    idxs = sorted({max(0, n - 1 - k) for k in range(max(1, sample))})
    for i in idxs:
        partial = np.asarray(
            pd.Series(factory(df.iloc[: i + 1], **params)).to_numpy(), dtype=float
        )
        if len(partial) != i + 1:
            raise ContractError(
                f"factory is not trailing-only: on a frame truncated to {i + 1} "
                f"bars it returned {len(partial)} values; a causal factory returns "
                f"one value per input bar"
            )
        a, b = float(full[i]), float(partial[i])
        same = (np.isnan(a) and np.isnan(b)) or a == b
        if not same:
            raise ContractError(
                f"factory is not trailing-only: value at index {i} is {a!r} over "
                f"full history but {b!r} when the frame ends at that bar. A "
                f"non-causal factory (shift(-1), a centred window) cannot be "
                f"rescued by truncation — the poisoned value IS arr[i]."
            )


class EvalSession:
    """Owns the loaded frames, the derived close-time arrays, the regime labels,
    and the memos. One per (symbol, graph, run).

    Frames are loaded UNBOUNDED and the replay loop is bounded instead — exactly
    what engine.run_backtest does (engine.py:265-267 loads everything, 308-309
    bounds start/end). Loading [start_ms, end_ms] would silently truncate
    indicator warmup and change every early candidate.
    """

    def __init__(self, source, symbol: str, *, tiers: tuple[str, str, str], regime_gate):
        self.symbol = symbol
        self.tiers = tuple(tiers)
        self.regime_gate = regime_gate
        regime_tf, setup_tf, trigger_tf = self.tiers

        self._frames: dict[str, pd.DataFrame] = {}
        self._intervals: dict[str, int] = {}
        self._closes: dict[str, np.ndarray] = {}
        self._fps: dict[str, tuple] = {}

        for tf, role in ((regime_tf, "regime"), (setup_tf, "setup"), (trigger_tf, "trigger")):
            if tf in self._frames:
                continue
            df = source.frame(symbol, tf, start_ms=None, end_ms=None)
            self._frames[tf] = df
            if df.empty:
                self._intervals[tf] = storage.TIMEFRAME_MS[tf]
                self._closes[tf] = np.array([], dtype="int64")
                continue
            interval = _assert_interval(df, tf, symbol, role)
            self._intervals[tf] = interval
            self._closes[tf] = df.index.to_numpy() + interval
            self._fps[tf] = _fingerprint(df)

        self._last_now: dict[str, int] = {}

    # -- introspection used by the executor, not by plug-ins ---------------- #

    @property
    def empty(self) -> bool:
        """True when any tier has no bars at all (engine.py:268-269's guard)."""
        return any(df.empty for df in self._frames.values())

    def frame_of(self, timeframe: str) -> pd.DataFrame:
        """The untruncated master frame. Executor-only — never handed to a plug-in."""
        return self._frames[timeframe]

    def interval_ms(self, timeframe: str) -> int:
        return self._intervals[timeframe]

    def closes(self, timeframe: str) -> np.ndarray:
        """Bar CLOSE times (index + interval) for one tier."""
        return self._closes[timeframe]

    def fingerprint(self, timeframe: str) -> tuple:
        return self._fps[timeframe]

    # -- memoized derived series -------------------------------------------- #

    def labels(self) -> pd.Series:
        """Regime labels over the whole regime frame.

        Byte-identical key shape to engine.py:279-282, and the same call:
        adx_period and atr_percentile_window come from config, exactly as
        engine.run_backtest leaves them (A2 — the classifier's thresholds are
        the one measured-healthy layer and are never swept).
        """
        regime_tf = self.tiers[0]
        df = self._frames[regime_tf]
        key = (
            "labels",
            self.symbol,
            regime_tf,
            self._fps.get(regime_tf),
            self.regime_gate.adx_trend_threshold,
            self.regime_gate.atr_extreme_percentile,
        )
        return _memo(
            key,
            lambda: classify_series(
                df,
                adx_trend_threshold=self.regime_gate.adx_trend_threshold,
                atr_extreme_percentile=self.regime_gate.atr_extreme_percentile,
            ),
        )

    def atr(self, timeframe: str, period: int) -> np.ndarray:
        """Wilder ATR over one tier's full history.

        `period` IS in the key. engine.py:293 omits it because ATR_STOP_PERIOD is
        config-fixed there; the graph exposes atr_period as a ParamSpec, so
        omitting it would serve a 14-period ATR to a graph asking for 21 — a
        silent wrong-number bug parity would not catch (the parity graph uses 14).
        A deliberate improvement on the engine's key, not a divergence.
        """
        df = self._frames[timeframe]
        key = ("atr", self.symbol, timeframe, self._fps.get(timeframe), int(period))
        return _memo(
            key, lambda: _readonly(wilder_atr(df, period=period).to_numpy())
        )

    def channels(self, timeframe: str, period: int) -> tuple[np.ndarray, np.ndarray]:
        """Trailing opposite-channel exit levels: (lower, upper). Same key note as atr()."""
        df = self._frames[timeframe]
        key = ("chan", self.symbol, timeframe, self._fps.get(timeframe), int(period))

        def produce():
            ch = channel_exit_levels(df, period=period)
            return (_readonly(ch["lower"].to_numpy()), _readonly(ch["upper"].to_numpy()))

        return _memo(key, produce)

    def series(self, timeframe: str, name: str, factory, **params) -> np.ndarray:
        """Memoized, read-only indicator array over one tier's FULL history.

        CACHE KEY — everything that can change a value, and nothing that cannot:
            ("series", name, symbol, timeframe,
             engine._fingerprint(master_frame),      # CONTENT, not (symbol, tf)
             tuple(sorted(params.items())))
        In the key because it changes the numbers: the frame content, every
        factory parameter (period, num_std, ...). NOT in the key because it
        cannot: start_ms/end_ms (they bound the loop, not the series),
        fee/slippage/funding, max_hold_bars, and the graph's name or meta.
        """
        df = self._frames[timeframe]
        key = (
            "series",
            name,
            self.symbol,
            timeframe,
            self._fps.get(timeframe),
            tuple(sorted(params.items())),
        )
        return _memo(
            key,
            lambda: _readonly(pd.Series(factory(df, **params)).to_numpy()),
        )

    def candidates(self, branch_key: str, setup_idx: int, produce) -> list:
        """Per-(branch, setup-bar) memo of detector output.

        This — not the indicator memo — is what engine.py's cand_cache
        (engine.py:328-349) actually relies on, and why: detect_donchian_setups
        computes a Wilder ADX over a PATTERN_LOOKBACK_BARS window for EVERY
        setup bar, and a Wilder recursion over a 180-bar window is NOT the tail
        of one over full history, so that cost cannot be cached away by the
        indicator seam. Memoizing the CANDIDATE instead is exact and it is
        shared across every graph and every parameter combo that leaves the
        branch untouched — which is the whole performance argument for a
        population: 200 graphs that share a donchian branch share its candidates.

        CACHE KEY:
            ("cands", branch_content_hash, symbol, setup_tf,
             fingerprint(df_setup), regime thresholds, bool(config.FADE_ENABLED))
        branch_content_hash is the canonical hash of the branch's RESOLVED
        detector node, so any parameter change invalidates. FADE_ENABLED is in
        the key for the reason engine.py:322-327 states: it is read at call
        time, so a live flip must not be served a cached pre-flip list.
        """
        setup_tf = self.tiers[1]
        key = (
            "cands",
            branch_key,
            self.symbol,
            setup_tf,
            self._fps.get(setup_tf),
            self.regime_gate.adx_trend_threshold,
            self.regime_gate.atr_extreme_percentile,
            bool(config.FADE_ENABLED),
        )
        if not config.FRAMEWORK_CACHE_ENABLED:
            _STATS["misses"] += 1
            return produce()
        per_bar: dict[int, list] = _CACHE.setdefault(key, {})
        if setup_idx in per_bar:
            _STATS["hits"] += 1
        else:
            _STATS["misses"] += 1
            per_bar[setup_idx] = produce()
        return per_bar[setup_idx]

    # -- context factory ---------------------------------------------------- #

    def context(self, role: str, now_ms: int) -> "EvalContext":
        """An EvalContext bound to ``now_ms``.

        Asserts now_ms is non-decreasing for this role, which catches an
        executor bug that walks backwards. Per-role rather than global because a
        setup-bar context legitimately trails the previous trigger-bar context.

        Raises:
            ContractError: naming the role, when now_ms goes backwards.
        """
        previous = self._last_now.get(role)
        if previous is not None and now_ms < previous:
            raise ContractError(
                f"{self.symbol}: evaluation time went BACKWARDS for role {role!r} "
                f"({previous} -> {now_ms}). A replay loop must advance; a "
                f"backwards step means some later bar's data was already seen."
            )
        self._last_now[role] = int(now_ms)
        return EvalContext(self, role, int(now_ms))


class EvalContext:
    """The entire surface a plug-in may ask for, and nothing else.

    WHAT A PLUG-IN MAY NOT ASK FOR, and why:
      - `conn`: a plug-in must not be able to query arbitrary history or write.
      - an untruncated frame: every accessor slices at now_ms inside this class.
      - wall-clock time: now_ms is data-derived; nothing here calls time.time().
      - `random`: Mutators receive their own seeded rng (contract §3); a
        detector that samples randomness is not reproducible.
      - the network: no phase of this framework fetches at evaluation time.
      - another symbol: cross-sectional strategies are a deliberate future
        extension needing their own no-lookahead argument, not an accident of
        API surface.
    """

    __slots__ = ("_session", "_role", "_now_ms")

    def __init__(self, session: EvalSession, role: str, now_ms: int):
        self._session = session
        self._role = role
        self._now_ms = int(now_ms)

    @property
    def symbol(self) -> str:
        return self._session.symbol

    @property
    def role(self) -> str:
        """"regime" | "setup" | "trigger" — which evaluation instant this is."""
        return self._role

    @property
    def now_ms(self) -> int:
        """The evaluation instant. Data-derived, never a clock.

        See the module docstring's ROLES block: for role "setup" this is the
        setup bar's CLOSE; for role "trigger" it is the trigger bar's OPEN
        (MEDIUM-2).
        """
        return self._now_ms

    @property
    def tiers(self) -> tuple[str, str, str]:
        """(regime_tf, setup_tf, trigger_tf) — names, so a plug-in can ask for
        'the setup tier' without hardcoding '4h'."""
        return self._session.tiers

    def interval_ms(self, timeframe: str) -> int:
        """Bar spacing of one tier, in milliseconds."""
        return self._session.interval_ms(timeframe)

    def bar_index(self, timeframe: str) -> int:
        """Positional index of the last bar CLOSED by now_ms; -1 if none.

        A bar is closed when ts + interval <= now_ms, which with
        close_times = index + interval is exactly
        searchsorted(close_times, now_ms, side="right") - 1 — the same rule
        engine.py:316 uses for the regime label, classifier.current_regime uses
        at classifier.py:181-184, and setup._load_df uses at setup.py:209-210.
        """
        closes = self._session.closes(timeframe)
        if len(closes) == 0:
            return -1
        return int(np.searchsorted(closes, self._now_ms, side="right")) - 1

    def frame(self, timeframe: str) -> pd.DataFrame:
        """All CLOSED bars: df.iloc[: bar_index + 1]."""
        k = self.bar_index(timeframe)
        return self._session.frame_of(timeframe).iloc[: k + 1]

    def window(self, timeframe: str, bars: int) -> pd.DataFrame:
        """The trailing ``bars`` closed bars.

        Equals engine.py:339's df_setup.iloc[max(0, h_idx+1-N) : h_idx+1]
        exactly — same bar count, same last bar. Wilder ADX is recursive from
        the window start, so a window one bar longer changes adx_now and can
        flip the adx_min gate.
        """
        k = self.bar_index(timeframe)
        return self._session.frame_of(timeframe).iloc[max(0, k + 1 - bars) : k + 1]

    def latest(self, timeframe: str) -> pd.Series | None:
        """The last CLOSED bar of one tier, or None during warmup."""
        k = self.bar_index(timeframe)
        if k < 0:
            return None
        return self._session.frame_of(timeframe).iloc[k]

    def regime(self) -> str:
        """Label of the last regime bar CLOSED by now_ms; 'uncertain' if none.

        The single implementation of engine.py:315-317's rule.
        """
        gate = self._session.regime_gate
        if not gate.enabled:
            return "any"
        k = self.bar_index(self._session.tiers[0])
        if k < 0:
            return "uncertain"
        return str(self._session.labels().iloc[k])

    def series(self, timeframe: str, name: str, factory, **params) -> np.ndarray:
        """Memoized indicator array, TRUNCATED at the last bar closed by now_ms.

        factory(full_frame, **params) -> np.ndarray | pd.Series is computed ONCE
        over full history per key and cached process-wide; the array handed back
        is a zero-copy, READ-ONLY slice arr[: bar_index + 1]. That is what makes
        the seam both fast and safe: the computation sees all the bars (so it is
        O(n) per run, not O(n^2)), and the caller cannot index into the future.

        The factory MUST be trailing-only. Truncation is not a substitute: a
        centred rolling window would poison arr[i] itself. Use
        assert_trailing_only() in the plug-in's tests.
        """
        arr = self._session.series(timeframe, name, factory, **params)
        k = self.bar_index(timeframe)
        return arr[: k + 1]

    def atr(self, timeframe: str, period: int) -> np.ndarray:
        """Wilder ATR, truncated at the current bar. Convenience over series()
        that shares engine.py:293-296's memo shape (with `period` added, see
        EvalSession.atr)."""
        arr = self._session.atr(timeframe, period)
        k = self.bar_index(timeframe)
        return arr[: k + 1]
