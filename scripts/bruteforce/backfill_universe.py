"""Backfill the research universe (20 perps x 1d/4h/1h) into data/ohlcv.db.

Reuses ``trading_bot.data.backfill.backfill_series`` verbatim rather than
reimplementing paging/resume semantics -- the only thing this script adds is
the loop over ``universe.UNIVERSE`` instead of production's 3 symbols.

Idempotent: upserts are keyed on (symbol, timeframe, ts) and each series
resumes from its own last stored bar, so re-running after an interruption
costs only the missing tail.

Usage:
    python scripts/bruteforce/backfill_universe.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading_bot.data import storage  # noqa: E402
from trading_bot.data.backfill import backfill_series  # noqa: E402
from universe import RESEARCH_TIMEFRAMES, UNIVERSE  # noqa: E402

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
)


def main() -> int:
    conn = storage.connect()
    failures: list[tuple[str, str, str]] = []
    started = time.time()

    for i, symbol in enumerate(UNIVERSE, 1):
        for timeframe in RESEARCH_TIMEFRAMES:
            t0 = time.time()
            result = backfill_series(conn, symbol, timeframe)
            have = conn.execute(
                "SELECT COUNT(*), MIN(ts), MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
            status = "ok" if result.complete else f"INCOMPLETE({result.reason})"
            print(
                f"[{i:2d}/{len(UNIVERSE)}] {symbol:14s} {timeframe:3s} "
                f"+{result.rows:6d} rows  total={have[0]:6d}  "
                f"{status}  {time.time() - t0:5.1f}s",
                flush=True,
            )
            if not result.complete:
                failures.append((symbol, timeframe, result.reason or "unknown"))

    print(f"\nElapsed {time.time() - started:.0f}s")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for symbol, timeframe, reason in failures:
            print(f"  {symbol} {timeframe}: {reason}")
        return 1
    print("All series complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
