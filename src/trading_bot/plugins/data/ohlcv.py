"""
data.ohlcv — the MVP DataSource, over data/ohlcv.db.

Wraps storage.load_candles behind the DataSource contract so a later phase can
add a second source (funding, sentiment, order-book) without touching the
executor.
"""

import logging

import pandas as pd

from trading_bot import config
from trading_bot.backtest.engine import _assert_interval
from trading_bot.data import storage
from trading_bot.framework.contracts import ParamSpec
from trading_bot.framework.registry import register

logger = logging.getLogger("trading_bot")

# Which engine role string each configured tier carries, so _assert_interval's
# error message is IDENTICAL to the one engine.run_backtest produces
# (engine.py:271-273). A test pins that both paths raise the same text.
_ROLE_BY_TIMEFRAME = {
    config.REGIME_TIMEFRAME: "regime",
    config.SIGNAL_PATTERN_TIMEFRAME: "setup",
    config.SIGNAL_TRIGGER_TIMEFRAME: "trigger",
}


class OhlcvSource:
    """DataSource over data/ohlcv.db.

    frame() returns EVERY stored bar for (symbol, timeframe) within
    [start_ms, end_ms] — both bounds INCLUSIVE, matching storage.load_candles'
    contract (storage.py:216-217). Truncation to "closed by now_ms" is
    EvalContext's job, not this class's: a DataSource that also truncated would
    give two places to get the closed-bar rule wrong.

    engine._assert_interval runs on every loaded frame — the SAME function with
    the same error text — so a series backfilled under the wrong timeframe key
    raises here rather than silently voiding every no-lookahead guarantee
    downstream (engine.py:185-222).

    `conn` is captured, never opened here: the CLI owns connection lifecycle and
    storage._db_lock serializes access.
    """

    def __init__(self, conn, *, expose: str = "tiers"):
        self._conn = conn
        self._expose = expose
        self._cache: dict[tuple, pd.DataFrame] = {}

    def frame(
        self,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Load one (symbol, timeframe) series as an OHLCV frame.

        Built exactly as engine._df does (engine.py:177-182): columns
        ts/open/high/low/close/volume, ts cast to int, ts as the index.

        Memoized per (symbol, timeframe, start_ms, end_ms) for this source's
        lifetime, which is one run_graph_backtest call.
        """
        key = (symbol, timeframe, start_ms, end_ms)
        if key in self._cache:
            return self._cache[key]
        rows = storage.load_candles(
            self._conn, symbol, timeframe, start_ms=start_ms, end_ms=end_ms
        )
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        if len(df):
            df["ts"] = df["ts"].astype(int)
        df = df.set_index("ts")
        if not df.empty:
            _assert_interval(
                df, timeframe, symbol, _ROLE_BY_TIMEFRAME.get(timeframe, timeframe)
            )
        self._cache[key] = df
        return df

    def timeframes(self) -> tuple[str, ...]:
        """The timeframes this source exposes.

        "tiers" (the default) is the three configured tiers, coarsest first.
        "all" is every key of storage.TIMEFRAME_MS actually present in the DB —
        a DB probe, so it reports what exists rather than what could exist.
        """
        if self._expose == "tiers":
            return (
                config.REGIME_TIMEFRAME,
                config.SIGNAL_PATTERN_TIMEFRAME,
                config.SIGNAL_TRIGGER_TIMEFRAME,
            )
        rows = self._conn.execute("SELECT DISTINCT timeframe FROM ohlcv").fetchall()
        present = {r[0] for r in rows}
        return tuple(tf for tf in storage.TIMEFRAME_MS if tf in present)


@register(
    "data",
    name="ohlcv",
    params={
        "timeframes": ParamSpec(
            kind="choice",
            default="tiers",
            choices=("tiers", "all"),
            doc="Which stored timeframes to expose",
        )
    },
    rationale=(
        "The stored OHLCV series is the only data v0.3.0 has. Wrapping "
        "storage.load_candles behind DataSource is what lets a later phase add a "
        "second source (funding, sentiment) without touching the executor."
    ),
    timeframes=(),
)
def ohlcv(conn, *, timeframes: str = "tiers") -> OhlcvSource:
    """Factory returning an OhlcvSource bound to ``conn``.

    IMPORTANT (the single easiest way to fail parity by 5-10 trades):
    run_graph_backtest passes start_ms/end_ms to bound the LOOP, not the load.
    engine.run_backtest loads the WHOLE series and then bounds start/end inside
    the loop (engine.py:265-267 vs 308-309), which matters because indicator
    warmup needs bars BEFORE start_ms. EvalSession therefore calls frame() with
    both bounds None; loading [start_ms, end_ms] instead would silently truncate
    warmup and change every early candidate.
    """
    return OhlcvSource(conn, expose=timeframes)
