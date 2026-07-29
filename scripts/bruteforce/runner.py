"""Sweep driver: run every registered strategy x param combo x symbol.

Usage:
    python scripts/bruteforce/runner.py --split TRAIN
    python scripts/bruteforce/runner.py --split TRAIN --only rsi2_fade,donchian_adx
    python scripts/bruteforce/runner.py --split SELECT --symbols BTCUSDT,ETHUSDT
    python scripts/bruteforce/runner.py --causal-only        # lookahead audit, no scoring

Guarantees this driver enforces so no individual strategy has to:

- **Causality first.** Every strategy passes ``core.assert_causal`` before ANY
  of its rows reach the leaderboard. A failure disqualifies the whole strategy
  and is reported loudly rather than skipped quietly.
- **HOLDOUT is locked.** Running ``--split HOLDOUT`` requires
  ``--i-am-spending-the-holdout`` and an explicit ``--only`` shortlist. The
  holdout is a depleting resource; spending it must be a deliberate act.
- **Honest trial counting.** DSR is computed against the total combo count
  across every loaded strategy, so a winner is charged for the whole search.
- **Parallel by symbol.** Each worker process loads its own frames; the sweep is
  embarrassingly parallel and CPU-bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core  # noqa: E402
import registry  # noqa: E402
from universe import CORE, UNIVERSE  # noqa: E402

REPORT_DIR = (
    Path(__file__).resolve().parents[2]
    / ".claude" / "PRPs" / "reports" / "bruteforce-strategy"
)
RESULTS_DIR = REPORT_DIR / "results"

FIELDS = [
    "strategy", "family", "symbol", "split", "trigger_tf", "params_json",
    "trades", "win_rate", "expectancy_R", "profit_factor",
    "sharpe", "sortino", "max_dd_pct", "ann_return_pct",
    "ann_return_at_target_pct", "max_dd_at_target_pct", "ann_vol_pct", "vol_scale",
    "cost_ratio", "median_risk_pct", "dsr",
    "avg_bars_held", "long_share", "stop_rate", "target_rate", "time_rate", "n_days",
]


def _run_one_symbol(args: tuple) -> tuple[str, list[dict], str | None]:
    """Score every combo of one strategy on one symbol. Runs in a worker process.

    Returns ``(label, rows, error)``. A raised exception is returned rather than
    propagated so one bad strategy cannot abort a multi-hour sweep -- but it IS
    surfaced in the summary, never swallowed.
    """
    strat_name, symbol, split, n_trials = args
    # Belt-and-braces against an empty registry in a worker. The pool's
    # `initializer` normally handles this; on the `spawn` start method (the macOS
    # default) a worker re-imports this module fresh, `main()` is behind the
    # __main__ guard, so nothing would otherwise populate registry.ALL and every
    # job would die with KeyError. load_all() is idempotent and cheap after the
    # first call, so re-checking here costs nothing and makes the function safe
    # to call directly in-process too.
    if strat_name not in registry.ALL:
        registry.load_all()
    strat = registry.ALL[strat_name]
    label = f"{strat_name}/{symbol}"
    try:
        ctx = core.make_ctx(
            symbol, trigger_tf=strat.trigger_tf, timeframes=strat.timeframes
        )
        if ctx is None:
            return label, [], f"no data for {symbol}"
        start_ms, end_ms = core.split_ms(split)
        cfg = core.SimConfig(
            max_hold_bars=strat.max_hold_bars or core.SimConfig().max_hold_bars,
            allow_short=not strat.long_only,
        )
        rows: list[dict] = []
        for params in strat.combos():
            plan = strat.build(ctx, **params)
            trades = core.simulate(ctx, plan, start_ms=start_ms, end_ms=end_ms, cfg=cfg)
            m = core.score(trades, start_ms=start_ms, end_ms=end_ms, n_trials=n_trials)
            rows.append(
                {
                    "strategy": strat_name, "family": strat.family, "symbol": symbol,
                    "split": split, "trigger_tf": strat.trigger_tf,
                    "params_json": json.dumps(params, sort_keys=True),
                    **{k: m.get(k) for k in FIELDS if k in m},
                }
            )
        return label, rows, None
    except Exception:
        return label, [], traceback.format_exc(limit=6)


def check_causality(strat_names: list[str], symbol: str = "BTCUSDT") -> dict[str, str]:
    """Run the lookahead audit. Returns ``{strategy: error}`` for failures only.

    One symbol is enough: lookahead is a property of the code, not the data. The
    first combo of each grid is checked (the cheapest sufficient probe -- a
    strategy that peeks does so for every parameter value).
    """
    failures: dict[str, str] = {}
    for name in strat_names:
        strat = registry.ALL[name]
        try:
            core.assert_causal(
                strat.build, strat.combos()[0], symbol,
                trigger_tf=strat.trigger_tf, timeframes=strat.timeframes,
            )
        except AssertionError as e:
            failures[name] = str(e)
        except Exception:
            failures[name] = traceback.format_exc(limit=4)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="TRAIN", choices=list(core.SPLITS))
    ap.add_argument("--only", default="", help="comma-separated strategy names")
    ap.add_argument("--family", default="", help="comma-separated families")
    ap.add_argument(
        "--symbols", default="", help="comma-separated; default = the 20-symbol universe"
    )
    ap.add_argument("--core-only", action="store_true", help="just BTC/ETH/SOL")
    ap.add_argument("--causal-only", action="store_true", help="audit lookahead, don't score")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="", help="output CSV path (default: results/<SPLIT>.csv)")
    ap.add_argument(
        "--i-am-spending-the-holdout", action="store_true",
        help="required to run --split HOLDOUT; see the module docstring",
    )
    args = ap.parse_args()

    registry.load_all()
    names = sorted(registry.ALL)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        missing = [w for w in wanted if w not in registry.ALL]
        if missing:
            print(f"unknown strategies: {missing}\nregistered: {names}", file=sys.stderr)
            return 2
        names = wanted
    if args.family:
        fams = {s.strip() for s in args.family.split(",")}
        names = [n for n in names if registry.ALL[n].family in fams]
    if not names:
        print("no strategies registered/selected", file=sys.stderr)
        return 2

    if args.split == "HOLDOUT":
        # The holdout is spent once, on a pre-committed shortlist. Refusing the
        # unguarded case is the whole point: an accidental holdout run cannot be
        # undone, and every later "out-of-sample" claim would be false.
        if not args.i_am_spending_the_holdout:
            print(
                "REFUSING: --split HOLDOUT needs --i-am-spending-the-holdout.\n"
                "The holdout may be evaluated ONCE, on the final shortlist only.",
                file=sys.stderr,
            )
            return 3
        if not args.only:
            print(
                "REFUSING: --split HOLDOUT needs an explicit --only shortlist.\n"
                "Sweeping the whole registry on the holdout destroys it.",
                file=sys.stderr,
            )
            return 3

    selected = [registry.ALL[n] for n in names]
    print(f"strategies: {len(names)}  combos: {registry.total_trials(selected)}")

    print("causality audit...", flush=True)
    t0 = time.time()
    failures = check_causality(names)
    for name, err in failures.items():
        print(f"  DISQUALIFIED {name}:\n    {err.splitlines()[0]}", file=sys.stderr)
    names = [n for n in names if n not in failures]
    print(f"  {len(names)} passed, {len(failures)} disqualified ({time.time() - t0:.1f}s)")
    if args.causal_only:
        return 1 if failures else 0
    if not names:
        return 1

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else (list(CORE) if args.core_only else list(UNIVERSE))
    )
    # Charged against the FULL registry, not the selected subset: the search
    # that produced any winner is the whole thing, and DSR must know that.
    n_trials = registry.total_trials() * len(symbols)

    jobs = [(n, sym, args.split, n_trials) for n in names for sym in symbols]
    print(f"{len(jobs)} (strategy, symbol) jobs on {args.workers} workers", flush=True)

    rows: list[dict] = []
    errors: list[tuple[str, str]] = []
    done = 0
    t0 = time.time()
    # initializer: import the strategy modules ONCE per worker rather than once
    # per job. Without it, workers started via `spawn` have an empty registry.
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=registry.load_all
    ) as pool:
        futures = {pool.submit(_run_one_symbol, j): j for j in jobs}
        for fut in as_completed(futures):
            label, got, err = fut.result()
            done += 1
            if err:
                errors.append((label, err))
            rows.extend(got)
            if done % 25 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                print(
                    f"  {done}/{len(jobs)}  {len(rows)} rows  "
                    f"{rate:.1f} job/s  eta {(len(jobs) - done) / max(rate, 1e-9):.0f}s",
                    flush=True,
                )

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_path}")

    if errors:
        print(f"\n{len(errors)} job errors:", file=sys.stderr)
        for label, err in errors[:10]:
            print(f"  {label}: {err.splitlines()[-1]}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} strategies disqualified for lookahead.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
