"""Vectorized multi-timeframe research harness for the brute-force strategy search.

WHY THIS EXISTS ALONGSIDE trading_bot.backtest.engine
-----------------------------------------------------
The production engine hardcodes one pipeline (Donchian/fade setups ->
check_breakout -> build_signal). It cannot express "RSI-2 mean reversion on 4H"
or "cross-sectional momentum" at all. This harness inverts the control flow: a
strategy is any function that turns bars into a ``Plan`` of per-bar decisions,
and the simulator is generic.

Everything that was already correct in production is IMPORTED, not reimplemented:
Wilder ATR/ADX, the cost model, and every equity metric (Sharpe, Sortino,
drawdown, Deflated Sharpe). The only new logic here is (a) multi-timeframe
alignment, (b) the generic exit simulator, and (c) the causality check.

THE STRATEGY CONTRACT
---------------------
A strategy receives a ``Ctx`` and returns a ``Plan`` whose arrays are indexed on
the TRIGGER timeframe grid. Index ``i`` means "decided using information
available at the close of trigger bar i, executed at that same close". The
simulator never looks at ``plan`` values beyond the bar it is currently on, and
never fills earlier than the bar after entry.

Two rules make lookahead structurally hard rather than merely discouraged:

1. Coarser-timeframe values reach the trigger grid only through ``Ctx.align``,
   which maps trigger bar i to the last coarse bar whose CLOSE time is <= bar
   i's close. A 4H value can therefore never be seen before it exists.
2. ``assert_causal`` re-runs the strategy on truncated history and requires the
   prefix of every array to be bit-identical. Centered windows, full-sample
   normalisation, ``bfill``, and ``shift(-1)`` all fail this test loudly. Every
   registered strategy is checked before its results are allowed onto the
   leaderboard.

COST AND SIZING MODEL
---------------------
Costs are the frozen PRD values and are NEVER swept: taker fee + slippage per
side, charged twice, plus funding per day held. Position sizing is fixed
FRACTIONAL RISK -- each trade risks ``base_risk`` of equity, so size is
``base_risk / risk_pct`` capped by ``lev_cap``. This is the honest analogue of
the PRD's volatility targeting: because the stop is ATR-derived, sizing to a
constant risk fraction automatically shrinks positions when volatility rises.

Sharpe is invariant to ``base_risk`` (it is a constant scale factor), which is
deliberate: it means the headline metric cannot be inflated by turning up risk.
Return and drawdown ARE scale-dependent, so both are additionally reported at
the PRD's 25% volatility target via an ex-post scale factor.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trading_bot import config  # noqa: E402
from trading_bot.backtest import equity  # noqa: E402
from trading_bot.data.storage import TIMEFRAME_MS  # noqa: E402

DB_PATH = str(Path(__file__).resolve().parents[2] / "data" / "ohlcv.db")
DAY_MS = 86_400_000

# ---------------------------------------------------------------------------
# Frozen cost model. These are the PRD's pessimistic values and are deliberately
# module constants rather than parameters: the PRD names "any parameter sweep of
# the cost model" as a silent overfitting channel worth ~5%/yr of phantom return.
# ---------------------------------------------------------------------------
FEE_PCT = config.FEE_PCT                        # 0.0005 taker, per side
SLIPPAGE_PCT = config.SLIPPAGE_PCT              # 0.0002, per side
FUNDING_PCT_PER_DAY = config.FUNDING_PCT_PER_DAY
ROUND_TRIP_COST = 2.0 * (FEE_PCT + SLIPPAGE_PCT)
COST_RATIO_CEILING = config.COST_RATIO_CEILING   # 0.10

# ---------------------------------------------------------------------------
# The 3-way split. Fixed once, here, so no strategy author can quietly redefine
# it. HOLDOUT is evaluated exactly once, on the final shortlist only.
# ---------------------------------------------------------------------------
SPLITS: dict[str, tuple[str, str]] = {
    "TRAIN": ("2023-08-01", "2025-07-01"),   # sweep freely (starts after 1D warmup)
    "SELECT": ("2025-07-01", "2026-01-01"),  # rank + shortlist
    "HOLDOUT": ("2026-01-01", "2026-07-24"), # ONE shot, shortlist only
}


def split_ms(name: str) -> tuple[int, int]:
    """Epoch-ms bounds of a named split. Raises KeyError on an unknown name."""
    start, end = SPLITS[name]
    return config.date_to_ms(start), config.date_to_ms(end)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

_FRAME_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def load_frame(symbol: str, timeframe: str, *, db_path: str = DB_PATH) -> pd.DataFrame:
    """Load one full OHLCV series, ascending, indexed by bar OPEN time (epoch ms).

    Cached per process: the sweep re-reads the same 60 series thousands of
    times, and SQLite decode dominates runtime otherwise.
    """
    key = (symbol, timeframe)
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        return cached
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol=? AND timeframe=? ORDER BY ts",
            (symbol, timeframe),
        ).fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype("int64")
    df = df.set_index("ts")
    _FRAME_CACHE[key] = df
    return df


def available_symbols(*, db_path: str = DB_PATH, timeframe: str = "1h") -> list[str]:
    """Symbols with data stored at ``timeframe``, alphabetical."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT symbol FROM ohlcv WHERE timeframe=? ORDER BY symbol",
                (timeframe,),
            )
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Context: multi-timeframe view of one symbol, plus the alignment primitive
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    """One symbol's bars across timeframes, on a fixed trigger grid.

    Attributes:
        symbol: e.g. "BTCUSDT".
        trigger_tf: The timeframe whose grid every Plan array is indexed on.
        frames: timeframe -> OHLCV frame (open-time indexed, ascending).
    """

    symbol: str
    trigger_tf: str
    frames: dict[str, pd.DataFrame]
    _align_cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # -- convenience accessors, so strategies read like TA code ------------
    def frame(self, timeframe: str | None = None) -> pd.DataFrame:
        return self.frames[timeframe or self.trigger_tf]

    @property
    def trigger(self) -> pd.DataFrame:
        return self.frames[self.trigger_tf]

    @property
    def n(self) -> int:
        return len(self.trigger)

    def close_times(self, timeframe: str) -> np.ndarray:
        """Bar CLOSE times of a timeframe: open time + one interval."""
        return self.frames[timeframe].index.to_numpy() + TIMEFRAME_MS[timeframe]

    def _index_map(self, timeframe: str) -> np.ndarray:
        """For each trigger bar, the index of the last ``timeframe`` bar CLOSED by it.

        -1 where no bar of that timeframe has closed yet. This is the single
        place coarse-timeframe information may enter the trigger grid, and it is
        the same ``searchsorted(..., side="right") - 1`` rule the production
        engine uses for regime labels (engine.py:316).
        """
        cached = self._align_cache.get(timeframe)
        if cached is not None:
            return cached
        if timeframe == self.trigger_tf:
            idx = np.arange(self.n, dtype=np.int64)
        else:
            idx = (
                np.searchsorted(
                    self.close_times(timeframe), self.close_times(self.trigger_tf),
                    side="right",
                ).astype(np.int64)
                - 1
            )
        self._align_cache[timeframe] = idx
        return idx

    def align(self, values, timeframe: str, *, fill: float = np.nan) -> np.ndarray:
        """Project a coarse-timeframe series onto the trigger grid.

        Args:
            values: array/Series indexed on ``timeframe``'s bars, same length
                as ``frames[timeframe]``.
            timeframe: Which timeframe ``values`` lives on.
            fill: Value used for trigger bars before the first coarse close.

        Returns:
            float64 array of length ``self.n``.
        """
        arr = np.asarray(
            values.to_numpy() if isinstance(values, pd.Series) else values, dtype=float
        )
        expected = len(self.frames[timeframe])
        if len(arr) != expected:
            raise ValueError(
                f"align(): {self.symbol} {timeframe} values have length {len(arr)} "
                f"but the {timeframe} frame has {expected} bars"
            )
        if timeframe == self.trigger_tf:
            return arr.copy()
        idx = self._index_map(timeframe)
        out = np.full(self.n, fill, dtype=float)
        valid = idx >= 0
        out[valid] = arr[idx[valid]]
        return out


def make_ctx(
    symbol: str,
    *,
    trigger_tf: str = "1h",
    timeframes: tuple[str, ...] = ("1d", "4h", "1h"),
    truncate_to: int | None = None,
    db_path: str = DB_PATH,
) -> Ctx | None:
    """Build a Ctx, or None if any requested series is missing/empty.

    Args:
        truncate_to: If set, keep only the first N trigger bars (and every
            coarse bar that closed by the last kept trigger bar's close). Used
            by ``assert_causal``; never used in scoring runs.
    """
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        df = load_frame(symbol, timeframe, db_path=db_path)
        if df.empty:
            return None
        frames[timeframe] = df

    if truncate_to is not None:
        trig = frames[trigger_tf].iloc[:truncate_to]
        if trig.empty:
            return None
        cutoff = int(trig.index[-1]) + TIMEFRAME_MS[trigger_tf]
        frames = {trigger_tf: trig} | {
            tf: df[df.index + TIMEFRAME_MS[tf] <= cutoff]
            for tf, df in frames.items()
            if tf != trigger_tf
        }
        if any(df.empty for df in frames.values()):
            return None

    return Ctx(symbol=symbol, trigger_tf=trigger_tf, frames=frames)


# ---------------------------------------------------------------------------
# The Plan: what a strategy returns
# ---------------------------------------------------------------------------

LONG, SHORT, FLAT = 1, -1, 0


@dataclass
class Plan:
    """Per-trigger-bar trading decisions.

    All arrays have length ``ctx.n`` and are read only at or before the bar the
    simulator is currently evaluating.

    Attributes:
        entry: +1 open long / -1 open short / 0 nothing, at this bar's CLOSE.
        stop_dist: Absolute price distance from entry to the initial stop.
            Must be > 0 on any bar where ``entry != 0``; a non-positive or NaN
            stop suppresses the entry (a strategy with no risk definition does
            not get to trade).
        target_dist: Absolute price distance to the take-profit. NaN means no
            target; the trade then relies on stop / exit signal / time stop.
        exit_signal: Optional. True on bars where an open position should be
            closed at that bar's CLOSE (e.g. an MA re-cross). Direction-agnostic.
        trail_atr: Optional per-bar ratchet-trail distance in price units
            (typically ``k * ATR``). NaN disables the trail on that bar.
        note: Free-text label recorded on every trade, for attribution.
    """

    entry: np.ndarray
    stop_dist: np.ndarray
    target_dist: np.ndarray | None = None
    exit_signal: np.ndarray | None = None
    trail_atr: np.ndarray | None = None
    note: str = ""

    def validate(self, n: int) -> None:
        for name in ("entry", "stop_dist", "target_dist", "exit_signal", "trail_atr"):
            arr = getattr(self, name)
            if arr is None:
                continue
            if len(arr) != n:
                raise ValueError(
                    f"Plan.{name} has length {len(arr)} but the trigger grid has {n} bars"
                )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

TRADE_DTYPE = np.dtype(
    [
        ("entry_i", "i8"), ("exit_i", "i8"),
        ("entry_ts", "i8"), ("exit_ts", "i8"),
        ("direction", "i1"),
        ("entry", "f8"), ("stop", "f8"), ("target", "f8"), ("exit_price", "f8"),
        ("risk_pct", "f8"), ("pnl_pct", "f8"), ("r_multiple", "f8"),
        ("bars_held", "i8"), ("outcome", "u1"),
    ]
)

OUTCOMES = ("stop", "target", "signal", "trail", "time", "end")
_OUT = {name: i for i, name in enumerate(OUTCOMES)}


@dataclass(frozen=True)
class SimConfig:
    """Non-strategy simulation settings.

    ``max_hold_bars`` is in TRIGGER bars. ``allow_short`` lets a family be
    tested long-only without editing the strategy.
    """

    max_hold_bars: int = config.MAX_HOLD_BARS_TRIGGER  # 96 => 4 days at 1H
    allow_short: bool = True
    allow_long: bool = True


def simulate(
    ctx: Ctx, plan: Plan, *, start_ms: int, end_ms: int, cfg: SimConfig | None = None
) -> np.ndarray:
    """Replay a Plan bar by bar. Returns a structured array of trades.

    Simulation rules (deliberately identical in spirit to
    ``trading_bot.backtest.engine.run_backtest`` so results stay comparable):

    - One position at a time; entries fill at the trigger bar's CLOSE.
    - Exits are evaluated from the bar AFTER entry onward, never same-bar.
    - Conservative same-bar rule: a bar touching both stop and target is
      assumed to have hit the STOP first.
    - Exit priority: stop/trail, then target, then exit_signal (at close),
      then the time stop.
    - The trail ratchets only AFTER the current bar's exits resolve, so a stop
      derived from bar j's own extreme can never fire on bar j.
    - An open position at ``end_ms`` (or at data end) closes at the last close
      with outcome "end".
    """
    cfg = cfg or SimConfig()
    plan.validate(ctx.n)

    trig = ctx.trigger
    ts = trig.index.to_numpy()
    highs = trig["high"].to_numpy(dtype=float)
    lows = trig["low"].to_numpy(dtype=float)
    closes = trig["close"].to_numpy(dtype=float)
    interval = TIMEFRAME_MS[ctx.trigger_tf]
    close_t = ts + interval

    entry_sig = np.asarray(plan.entry, dtype=np.int8)
    stop_dist = np.asarray(plan.stop_dist, dtype=float)
    target_dist = (
        np.full(ctx.n, np.nan) if plan.target_dist is None
        else np.asarray(plan.target_dist, dtype=float)
    )
    exit_sig = (
        np.zeros(ctx.n, dtype=bool) if plan.exit_signal is None
        else np.asarray(plan.exit_signal, dtype=bool)
    )
    trail_atr = (
        np.full(ctx.n, np.nan) if plan.trail_atr is None
        else np.asarray(plan.trail_atr, dtype=float)
    )

    out: list[tuple] = []
    # Open position state. dir_ == 0 means flat.
    dir_ = 0
    e_i = e_ts = 0
    e_px = e_stop = e_target = e_risk = 0.0
    extreme = 0.0
    trailed = False
    last_j = -1

    def close_out(j: int, price: float, outcome: str) -> None:
        nonlocal dir_
        gross = dir_ * (price - e_px) / e_px
        hold_days = (int(ts[j]) - e_ts) / DAY_MS
        pnl = gross - ROUND_TRIP_COST - FUNDING_PCT_PER_DAY * hold_days
        out.append(
            (
                e_i, j, e_ts, int(ts[j]), dir_,
                e_px, e_stop, e_target, price,
                e_risk, pnl, pnl / e_risk if e_risk > 0 else np.nan,
                j - e_i, _OUT[outcome],
            )
        )
        dir_ = 0

    for j in range(ctx.n):
        bar_close_t = int(close_t[j])
        if bar_close_t > end_ms:
            break
        last_j = j

        if dir_ != 0:
            if j <= e_i:
                continue
            if dir_ == LONG:
                if lows[j] <= e_stop:
                    close_out(j, e_stop, "trail" if trailed else "stop")
                elif not np.isnan(e_target) and highs[j] >= e_target:
                    close_out(j, e_target, "target")
                elif exit_sig[j]:
                    close_out(j, float(closes[j]), "signal")
            else:
                if highs[j] >= e_stop:
                    close_out(j, e_stop, "trail" if trailed else "stop")
                elif not np.isnan(e_target) and lows[j] <= e_target:
                    close_out(j, e_target, "target")
                elif exit_sig[j]:
                    close_out(j, float(closes[j]), "signal")
            if dir_ != 0 and j - e_i >= cfg.max_hold_bars:
                close_out(j, float(closes[j]), "time")
            # Ratchet last: see docstring. Never loosens.
            if dir_ != 0 and not np.isnan(trail_atr[j]) and trail_atr[j] > 0:
                if dir_ == LONG:
                    extreme = max(extreme, float(highs[j]))
                    cand = extreme - trail_atr[j]
                    if cand > e_stop:
                        e_stop, trailed = cand, True
                else:
                    extreme = min(extreme, float(lows[j]))
                    cand = extreme + trail_atr[j]
                    if cand < e_stop:
                        e_stop, trailed = cand, True
            continue

        # Flat: consider an entry. j >= 1 mirrors the production engine, which
        # never enters on the very first bar of a series.
        if bar_close_t < start_ms or j < 1:
            continue
        sig = int(entry_sig[j])
        if sig == 0:
            continue
        if sig == LONG and not cfg.allow_long:
            continue
        if sig == SHORT and not cfg.allow_short:
            continue
        sd = float(stop_dist[j])
        px = float(closes[j])
        if not np.isfinite(sd) or sd <= 0 or not np.isfinite(px) or px <= 0:
            continue  # no valid risk definition => no trade
        risk_pct = sd / px
        if risk_pct >= 1.0:
            continue  # a stop at or beyond zero is not a stop
        td = float(target_dist[j]) if target_dist is not None else np.nan

        dir_, e_i, e_ts, e_px, e_risk = sig, j, int(ts[j]), px, risk_pct
        e_stop = px - sd if sig == LONG else px + sd
        e_target = (
            np.nan if not np.isfinite(td) or td <= 0
            else (px + td if sig == LONG else px - td)
        )
        extreme, trailed = px, False

    if dir_ != 0 and last_j > e_i:
        close_out(last_j, float(closes[last_j]), "end")

    return np.array(out, dtype=TRADE_DTYPE) if out else np.zeros(0, dtype=TRADE_DTYPE)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

BASE_RISK = 0.01     # fraction of equity risked per trade
LEV_CAP = 5.0        # hard cap on notional/equity, per the PRD's leverage-cap note
TARGET_VOL = 0.25    # PRD's 25% annualized volatility target
MAX_VOL_SCALE = 3.0  # ceiling on the ex-post vol-target scale factor


class _R:
    """Minimal shim so trading_bot.backtest.equity can consume our trades.

    equity.daily_returns only needs ``exit_ts`` and ``pnl_pct``; passing a
    lightweight object keeps that module reused verbatim instead of copied.
    """

    __slots__ = ("exit_ts", "pnl_pct")

    def __init__(self, exit_ts: int, pnl_pct: float) -> None:
        self.exit_ts = exit_ts
        self.pnl_pct = pnl_pct


def score(trades: np.ndarray, *, start_ms: int, end_ms: int, n_trials: int = 1) -> dict:
    """Compute the full metric set for one trade list over one span.

    Sizing: each trade risks BASE_RISK of equity, so its equity return is
    ``min(LEV_CAP, BASE_RISK / risk_pct) * pnl_pct``. Sharpe is unaffected by
    BASE_RISK (constant scaling) unless LEV_CAP binds, which is the point: risk
    dialled up cannot buy Sharpe.

    Returns a flat dict of floats/ints (None where undefined) suitable for a
    CSV row. ``ann_return_at_target_pct`` and ``max_dd_at_target_pct`` restate
    return and drawdown after scaling daily returns to TARGET_VOL, capped at
    MAX_VOL_SCALE -- the PRD's "derived return (outcome, not a gate)".
    """
    n = int(len(trades))
    base: dict = {
        "trades": n, "win_rate": None, "expectancy_R": None, "profit_factor": None,
        "avg_win_R": None, "avg_loss_R": None, "cost_ratio": None,
        "median_risk_pct": None, "sharpe": None, "sortino": None,
        "max_dd_pct": None, "ann_return_pct": None, "dsr": None, "n_days": 0,
        "ann_vol_pct": None, "vol_scale": None,
        "ann_return_at_target_pct": None, "max_dd_at_target_pct": None,
        "avg_bars_held": None, "long_share": None,
        "stop_rate": None, "target_rate": None, "time_rate": None,
    }
    if n == 0:
        return base

    r = trades["r_multiple"]
    r = r[np.isfinite(r)]
    pnl = trades["pnl_pct"]
    risk = trades["risk_pct"]
    wins, losses = r[r > 0], r[r <= 0]

    base["win_rate"] = float(len(wins) / len(r)) if len(r) else None
    base["expectancy_R"] = float(np.mean(r)) if len(r) else None
    base["avg_win_R"] = float(np.mean(wins)) if len(wins) else None
    base["avg_loss_R"] = float(np.mean(losses)) if len(losses) else None
    gain, loss = float(pnl[pnl > 0].sum()), float(-pnl[pnl <= 0].sum())
    base["profit_factor"] = (gain / loss) if loss > 0 else None
    med_risk = float(np.median(risk))
    base["median_risk_pct"] = med_risk
    # c = round-trip cost / risk, the PRD's cost-frontier gate (ceiling 0.10).
    base["cost_ratio"] = (ROUND_TRIP_COST / med_risk) if med_risk > 0 else None
    base["avg_bars_held"] = float(np.mean(trades["bars_held"]))
    base["long_share"] = float(np.mean(trades["direction"] == LONG))
    for name, key in (("stop", "stop_rate"), ("target", "target_rate"), ("time", "time_rate")):
        base[key] = float(np.mean(trades["outcome"] == _OUT[name]))

    size = np.minimum(LEV_CAP, BASE_RISK / np.maximum(risk, 1e-9))
    sized = [_R(int(t), float(p)) for t, p in zip(trades["exit_ts"], size * pnl)]
    m = equity.compute_equity_metrics(sized, start_ms, end_ms, n_trials=n_trials)
    base.update(
        {
            "n_days": m["n_days"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd_pct": m["max_drawdown_pct"], "ann_return_pct": m["ann_return_pct"],
            "dsr": m["dsr"],
        }
    )

    rets = equity.daily_returns(sized, start_ms, end_ms)
    if len(rets) >= 2:
        vol = float(np.std(rets, ddof=1) * np.sqrt(equity.PERIODS_PER_YEAR))
        base["ann_vol_pct"] = vol
        if vol > 0:
            scale = min(MAX_VOL_SCALE, TARGET_VOL / vol)
            base["vol_scale"] = scale
            scaled = [x * scale for x in rets]
            base["max_dd_at_target_pct"] = equity.max_drawdown(scaled)
            eq = 1.0
            for x in scaled:
                eq *= 1.0 + x
                if eq <= 0:
                    eq = 0.0
                    break
            base["ann_return_at_target_pct"] = (
                eq ** (equity.PERIODS_PER_YEAR / len(scaled)) - 1.0 if eq > 0 else -1.0
            )
    return base


# ---------------------------------------------------------------------------
# Causality check -- the anti-lookahead net
# ---------------------------------------------------------------------------


def assert_causal(
    build, params: dict, symbol: str, *,
    trigger_tf: str = "1h",
    timeframes: tuple[str, ...] = ("1d", "4h", "1h"),
    cut_fracs: tuple[float, ...] = (0.55, 0.8),
    atol: float = 1e-9,
    tail_skip: int = 0,
) -> None:
    """Prove a strategy's Plan depends only on past bars. Raises on violation.

    Method (prefix invariance): build the Plan on the full series, then rebuild
    it on history truncated to the first K trigger bars. If the strategy is
    causal, the truncated Plan must equal the first K entries of the full Plan
    exactly -- INCLUDING the final bar. Anything that consults the future --
    ``shift(-1)``, ``np.roll(x, -1)``, a centered rolling window, a full-sample
    mean/quantile, ``bfill()``, resampling that peeks past a bar's close --
    changes the prefix and is caught here.

    WHY THE COMPARISON RUNS TO THE VERY LAST BAR (``tail_skip=0``)
    -------------------------------------------------------------
    An earlier version of this function excluded a slack window of
    ``2 * coarse_bars + 8`` trigger bars before the cut, reasoning that a
    partially-formed coarse bar would legitimately differ there. That reasoning
    was wrong, and it silently defeated the whole check: a one-bar lookahead
    only manifests on the LAST bar of the truncated series, which the slack
    window discarded. The ``lookahead_canary`` strategy -- which reads next
    bar's close on purpose -- passed the audit.

    There is in fact no legitimate difference to excuse. ``Ctx.align`` selects
    the last coarse bar whose CLOSE time is <= the trigger bar's close, so a
    partially-formed coarse bar is never visible in either run; and a pivot that
    needs N confirming bars is placed N bars late by construction, so its value
    at bar i depends only on bars <= i. A causal strategy is therefore identical
    on the entire prefix, and any tolerance here is a hole to hide in.

    ``tail_skip`` remains available for the rare strategy with a documented,
    deliberate reason to differ at the boundary, but it must be justified in the
    strategy's rationale -- it is not a knob for making a red audit go green.

    Args:
        build: ``build(ctx, **params) -> Plan``.
        params: Parameters passed to ``build``.
        symbol: Symbol to check on.
        cut_fracs: Truncation points as fractions of the full trigger series.
        atol: Absolute tolerance for float comparison.
        tail_skip: Bars before the cut to exclude. Keep at 0.

    Raises:
        AssertionError: With the offending array, index, and both values.
    """
    full_ctx = make_ctx(symbol, trigger_tf=trigger_tf, timeframes=timeframes)
    if full_ctx is None:
        raise AssertionError(f"assert_causal: no data for {symbol}")
    full = build(full_ctx, **params)
    full.validate(full_ctx.n)

    for frac in cut_fracs:
        k = int(full_ctx.n * frac)
        if k <= tail_skip + 50:
            continue
        cut_ctx = make_ctx(
            symbol, trigger_tf=trigger_tf, timeframes=timeframes, truncate_to=k
        )
        if cut_ctx is None:
            continue
        part = build(cut_ctx, **params)
        part.validate(cut_ctx.n)
        m = min(cut_ctx.n, k) - tail_skip
        if m <= 0:
            continue
        for name in ("entry", "stop_dist", "target_dist", "exit_signal", "trail_atr"):
            a, b = getattr(full, name), getattr(part, name)
            if a is None and b is None:
                continue
            if (a is None) != (b is None):
                raise AssertionError(
                    f"LOOKAHEAD in {build.__module__}.{build.__name__} "
                    f"({symbol}, cut={frac}): Plan.{name} is present in one run "
                    f"and absent in the other -- the array's existence depends "
                    f"on how much future data was supplied."
                )
            x = np.asarray(a[:m], dtype=float)
            y = np.asarray(b[:m], dtype=float)
            bad = ~(
                (np.isnan(x) & np.isnan(y))
                | (np.abs(np.nan_to_num(x) - np.nan_to_num(y)) <= atol)
            )
            if bad.any():
                i = int(np.argmax(bad))
                raise AssertionError(
                    f"LOOKAHEAD in {build.__module__}.{build.__name__} "
                    f"({symbol}, cut={frac}): Plan.{name}[{i}] is {x[i]!r} with "
                    f"full history but {y[i]!r} when history stops at bar {k}. "
                    f"Settled bars must not change when future data is removed "
                    f"({int(bad.sum())} of {m} bars differ)."
                )
