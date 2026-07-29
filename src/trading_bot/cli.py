"""
CLI entry point for the trading bot.

Provides subcommands for backfilling historical data and other operations.
"""

import argparse
import sys
import time

import pandas as pd

from trading_bot import campaign, config
from trading_bot.backtest import trials, walkforward
from trading_bot.backtest.benchmark import METRIC_KEYS, buy_and_hold
from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.equity import compute_equity_metrics
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.backtest.walkforward import (
    DEFAULT_GRID,
    GATE_CONDITIONS,
    walk_forward_pooled,
)
from trading_bot.data import correlation
from trading_bot.data.backfill import backfill_all
from trading_bot.data.poller import run_polling_loop
from trading_bot.data.statestore import connect as connect_state
from trading_bot.data.storage import connect, find_gaps
from trading_bot.evolution import oracle as evo_oracle
from trading_bot.evolution import population as evo_population
from trading_bot.evolution import runner as evo_runner
from trading_bot.feedback import protocol as feedback_protocol
from trading_bot.feedback import records as feedback_records
from trading_bot.feedback import versioning as feedback_versioning
from trading_bot.framework import graph as fgraph
from trading_bot.framework import registry
from trading_bot.framework.errors import FrameworkError
from trading_bot.framework.execute import run_graph_backtest
from trading_bot.plugins.detectors.catalog import coverage_summary, format_coverage
from trading_bot.plugins.filters import rr_after_costs
from trading_bot.regime.classifier import current_regime
from trading_bot.signals.scan import scan_symbol


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Trading bot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (default: from config)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Backfill subcommand
    backfill_parser = subparsers.add_parser(
        "backfill", help="Backfill historical OHLCV data"
    )
    backfill_parser.add_argument(
        "--symbol",
        action="append",
        help="Symbol to backfill (repeatable, e.g., --symbol BTCUSDT --symbol ETHUSDT)",
    )
    backfill_parser.add_argument(
        "--timeframe",
        action="append",
        help="Timeframe to backfill (repeatable, e.g., --timeframe 15m --timeframe 1h)",
    )
    backfill_parser.add_argument(
        "--start",
        type=_date_arg,
        help="ISO date for start time (e.g., 2023-01-01), overrides BACKFILL_START for empty series",
    )

    # Poll subcommand
    subparsers.add_parser(
        "poll", help="Run live polling loop for latest candles"
    )

    # Gap-report subcommand
    gap_report_parser = subparsers.add_parser(
        "gap-report", help="Report gaps in stored OHLCV data"
    )
    gap_report_parser.add_argument(
        "--as-of",
        type=_date_arg,
        help="Report gaps as of this UTC date (YYYY-MM-DD); default is now",
    )
    gap_report_parser.add_argument(
        "--start",
        type=_date_arg,
        help="ISO date for the expected start of the series (e.g., 2023-01-01), "
        "overrides BACKFILL_START; use if series were backfilled from a later date",
    )

    # Regime subcommand
    regime_parser = subparsers.add_parser(
        "regime", help="Classify market regime for symbols"
    )
    regime_parser.add_argument(
        "--symbol",
        action="append",
        help="Symbol to classify (repeatable, e.g., --symbol BTCUSDT --symbol ETHUSDT); default is all symbols",
    )
    regime_parser.add_argument(
        "--as-of",
        type=_date_arg,
        help="Classify regime as of this UTC date (YYYY-MM-DD); default is now",
    )

    # Signal subcommand
    signal_parser = subparsers.add_parser(
        "signal",
        help="Scan for signals with the regime-matched method "
        "(trending: Donchian channel breakout; ranging: mean-reversion fade)",
    )
    signal_parser.add_argument(
        "--symbol",
        action="append",
        help="Symbol to scan (repeatable, e.g., --symbol BTCUSDT --symbol ETHUSDT); default is all symbols",
    )
    signal_parser.add_argument(
        "--as-of",
        type=_date_arg,
        help="Scan for signals as of this UTC date (YYYY-MM-DD); default is now",
    )

    # Backtest subcommand
    backtest_parser = subparsers.add_parser(
        "backtest", help="Replay the signal pipeline over stored history"
    )
    backtest_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to backtest (repeatable); default is all symbols",
    )
    backtest_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD (default: BACKFILL_START)",
    )
    backtest_parser.add_argument(
        "--end", type=_date_arg,
        help="UTC end date YYYY-MM-DD (default: now)",
    )

    # Walkforward subcommand
    wf_parser = subparsers.add_parser(
        "walkforward",
        help="Walk-forward validation with one-shot OOS gate (Phase 5 MVP gate)",
    )
    wf_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to validate (repeatable); default is all symbols",
    )
    wf_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD (default: BACKFILL_START)",
    )
    wf_parser.add_argument(
        "--end", type=_date_arg,
        help="UTC end date YYYY-MM-DD (default: now)",
    )
    wf_parser.add_argument(
        "--graph",
        help="Run the walk-forward through a serialized strategy graph instead "
        "of the legacy engine path. The sweep is reduced to the single "
        "run-level axis max_hold_bars, because a graph carries its own "
        "parameters (contract §5)",
    )

    # Plug-in and graph subcommands (v0.3.0 Phase 3)
    plugins_parser = subparsers.add_parser(
        "plugins", help="List registered framework plug-ins"
    )
    plugins_parser.add_argument(
        "--kind", choices=registry.KINDS, help="Only this kind; default all"
    )
    gv_parser = subparsers.add_parser(
        "graph-validate", help="Validate serialized strategy graphs"
    )
    gv_parser.add_argument(
        "path", nargs="+", help="Path(s) to <name>.strategy.json"
    )

    # Benchmark subcommand (v0.3.0 Phase 1)
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Buy-and-hold null hypothesis: per-symbol and equal-weight basket "
        "return / Sharpe / Sortino / max drawdown over a span",
    )
    benchmark_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to hold (repeatable); default is all symbols",
    )
    benchmark_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD (default: BACKFILL_START)",
    )
    benchmark_parser.add_argument(
        "--end", type=_date_arg,
        help="UTC end date YYYY-MM-DD (default: now)",
    )

    # Correlation-report subcommand (v0.3.0 Phase 2)
    correlation_parser = subparsers.add_parser(
        "correlation-report",
        help="Measure pairwise correlation / effective independent sample size "
        "over the stored universe and select config.RESEARCH_SYMBOLS",
    )
    correlation_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to measure (repeatable); default is every symbol stored "
        "at --timeframe",
    )
    correlation_parser.add_argument(
        "--timeframe", default=config.CORRELATION_TIMEFRAME,
        help="Return timeframe to measure correlation on (default: "
        "config.CORRELATION_TIMEFRAME). A single string, not repeatable -- "
        "correlation is defined on one return series.",
    )
    correlation_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD (default: config.CORRELATION_START)",
    )
    correlation_parser.add_argument(
        "--end", type=_date_arg,
        help="UTC end date YYYY-MM-DD (default: the last CLOSED bar stored)",
    )
    correlation_parser.add_argument(
        "--select-n", type=int, default=config.CORRELATION_SELECT_N,
        help="Non-anchor symbols to select (default: config.CORRELATION_SELECT_N)",
    )
    correlation_parser.add_argument(
        "--out", default=None,
        help="Write the markdown report to this path in addition to stdout",
    )
    correlation_parser.add_argument(
        "--check-exchange", action="store_true",
        help="Probe fapi.binance.com reachability before measuring (opt-in; "
        "the default path is fully offline)",
    )

    # Graph-backtest subcommand (v0.3.0 Phase 4)
    gb_parser = subparsers.add_parser(
        "graph-backtest",
        help="Replay a serialized strategy graph over stored history, with an "
        "optional per-position R:R audit trail and R:R survival report",
    )
    gb_parser.add_argument(
        "--graph", required=True,
        help="Path to a <name>.strategy.json. REQUIRED: there is no default "
        "graph, because a default would make the audit trail ambiguous about "
        "which strategy produced a number.",
    )
    gb_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to backtest (repeatable); default is config.SYMBOLS",
    )
    gb_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD (default: BACKFILL_START)",
    )
    gb_parser.add_argument(
        "--end", type=_date_arg,
        help="UTC end date YYYY-MM-DD (default: now)",
    )
    gb_parser.add_argument(
        "--audit", action="store_true",
        help="Print every taken position with its R:R justification (the PRD's "
        "'every position logged against its R:R justification')",
    )
    gb_parser.add_argument(
        "--rr-report", action="store_true",
        help="Measure the cost-adjusted R:R distribution over every plan that "
        "reached filter.rr-after-costs, per symbol and pooled",
    )

    # Review subcommand (v0.3.0 Phase 5)
    review_parser = subparsers.add_parser(
        "review",
        help="Feedback loop MVP: review closed trades against their own "
        "prediction, register/version strategy graphs, and forward-test a "
        "candidate through THE GATE (contract §4 — reviews inform, the gate "
        "decides)",
    )
    review_group = review_parser.add_mutually_exclusive_group(required=True)
    review_group.add_argument(
        "--strategy", help="Path to a <name>.strategy.json to register (use with --register)"
    )
    review_group.add_argument(
        "--version", help="A previously registered strategy_version id"
    )
    review_parser.add_argument(
        "--register", action="store_true",
        help="Register --strategy and exit, printing the resulting version_id",
    )
    review_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to review (repeatable); default is config.SYMBOLS",
    )
    review_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD for the review span (default: BACKFILL_START)",
    )
    review_parser.add_argument(
        "--end", type=_date_arg,
        help="UTC end date YYYY-MM-DD for the review span (default: now)",
    )
    review_parser.add_argument(
        "--diagnose-only", action="store_true",
        help="Aggregate STORED review_records for --version; runs nothing",
    )
    review_parser.add_argument(
        "--loop", action="store_true",
        help="Explicitly run the refine step (diagnosis -> apply_suggestion -> "
        "child version), on top of the review + diagnosis every mode performs",
    )
    review_parser.add_argument(
        "--forward-start", type=_date_arg,
        help="UTC start date YYYY-MM-DD of the forward (unseen) window",
    )
    review_parser.add_argument(
        "--forward-end", type=_date_arg,
        help="UTC end date YYYY-MM-DD of the forward (unseen) window",
    )
    review_parser.add_argument(
        "--reviewer", default="trade-quality",
        help="Registered reviewer plug-in name or key (default: trade-quality)",
    )
    review_parser.add_argument(
        "--no-persist", action="store_true",
        help="Do not write review_records to state.db (diagnosis printed, nothing stored)",
    )
    review_parser.add_argument(
        "--state-db", default=None,
        help="Path to the framework state database (default: config.STATE_DB_PATH). "
        "On this subcommand only, so Phases 1/6/7 cannot collide over a global flag",
    )

    # Evolve subcommand (v0.3.0 Phase 6)
    evolve_parser = subparsers.add_parser(
        "evolve",
        help="Population search over strategy graphs, scored ONLY through THE "
        "GATE, with every evaluation charged to the persistent trial ledger",
    )
    evolve_parser.add_argument(
        "--seed-graph",
        help="Path to the <name>.strategy.json generation 0 descends from "
        "(required unless --resume or --report)",
    )
    evolve_parser.add_argument(
        "--seed", type=int,
        help="Campaign RNG seed — the ONE number a replay needs. Default: derived "
        "from the clock and RECORDED, so even an unseeded run is replayable",
    )
    evolve_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to pool (repeatable); default config.SYMBOLS. Pass "
        "config.RESEARCH_SYMBOLS explicitly for the wider set — the symbol set is "
        "part of a campaign's identity and is stored in campaigns.symbols_json",
    )
    evolve_parser.add_argument(
        "--population", type=int, default=None,
        help=f"Members per generation (default config.EVO_POPULATION = "
        f"{config.EVO_POPULATION})",
    )
    evolve_parser.add_argument(
        "--generations", type=int, default=None,
        help=f"Generations to run (default config.EVO_GENERATIONS = "
        f"{config.EVO_GENERATIONS})",
    )
    evolve_parser.add_argument(
        "--workers", type=int, default=None,
        help=f"Process pool size (default config.EVO_WORKERS = "
        f"{config.EVO_WORKERS}); 1 runs in-process",
    )
    evolve_parser.add_argument(
        "--window-days", type=int, default=None,
        help=f"Evaluation window per generation (default config.EVO_WINDOW_DAYS = "
        f"{config.EVO_WINDOW_DAYS}); must be >= WF_TRAIN+TEST+OOS",
    )
    evolve_parser.add_argument(
        "--train-start", type=_date_arg,
        help=f"UTC YYYY-MM-DD (default config.EVO_TRAIN_START = "
        f"{config.EVO_TRAIN_START})",
    )
    evolve_parser.add_argument(
        "--train-end", type=_date_arg,
        help=f"UTC YYYY-MM-DD ceiling (default config.EVO_TRAIN_END = "
        f"{config.EVO_TRAIN_END}). REFUSED if later: a CLI flag must not be able "
        f"to spend Phase 9's holdout",
    )
    evolve_parser.add_argument(
        "--budget-hours", type=float, default=None,
        help=f"Wall-clock budget for --calibrate's sizing math (default "
        f"config.EVO_BUDGET_HOURS = {config.EVO_BUDGET_HOURS})",
    )
    evolve_parser.add_argument(
        "--calibrate", action="store_true",
        help="MEASURE cold/warm per-candidate cost and print legal "
        "(population x generations) pairs. Charges trials under a throwaway "
        "campaign id — they are evaluations, so they are counted",
    )
    evolve_parser.add_argument(
        "--repeats", type=int, default=3,
        help="Evaluations --calibrate times (default 3; the first is the cold one)",
    )
    evolve_parser.add_argument(
        "--resume", default=None,
        help="Continue a campaign from its last COMPLETED generation. Accepts a "
        "campaign_id OR a --label. The ledger is persistent, so n_trials "
        "continues from the true total",
    )
    evolve_parser.add_argument(
        "--label", default=None,
        help="Durable operator name for a NEW campaign (e.g. cup-and-handle-v1). "
        "campaign_id embeds its creation date, so it is not a handle you can "
        "retype tomorrow to continue the same search — the label is. Must be "
        "unique across campaigns",
    )
    evolve_parser.add_argument(
        "--strategy-name", default=None,
        help="The strategy this campaign evolves. A campaign is an evolution "
        "history OF one strategy (one strategy, many campaigns); recording it "
        "is what lets a later continuation refuse a mismatched graph",
    )
    evolve_parser.add_argument(
        "--extend", type=int, default=None,
        help="With --resume: run at least this many FURTHER generations beyond "
        "the work already completed, raising the campaign's ceiling if needed. "
        "This is how a finished campaign keeps IMPROVING — parents, seen-hash "
        "set and ledger all carry over — instead of restarting under a new seed",
    )
    evolve_parser.add_argument(
        "--report", default=None,
        help="Reprint a finished campaign_id from state.db. Evaluates NOTHING and "
        "charges NOTHING",
    )
    evolve_parser.add_argument(
        "--dry-run", action="store_true",
        help="Breed and print generation 0 with ZERO oracle calls and ZERO trials "
        "charged. A dry run that charged trials would be a trap",
    )
    evolve_parser.add_argument(
        "--state-db", default=None,
        help="Path to the framework state database (default: config.STATE_DB_PATH)",
    )

    # Detector-report subcommand (v0.3.0 Phase 8). DELIBERATELY has no --param
    # override and no sort option: see _detector_report_command's docstring.
    dr_parser = subparsers.add_parser(
        "detector-report",
        help="DIAGNOSTIC, not a gate verdict: per-detector event/trade counts, "
        "hit rate, after-costs expectancy and cost ratio over the TUNING span, "
        "at fixed ParamSpec defaults, plus the 144-row coverage ledger",
    )
    dr_parser.add_argument(
        "--symbol", action="append",
        help="Symbol to measure (repeatable); default is config.SYMBOLS",
    )
    dr_parser.add_argument(
        "--start", type=_date_arg,
        help="UTC start date YYYY-MM-DD (default: BACKFILL_START)",
    )
    dr_parser.add_argument(
        "--end", type=_date_arg,
        help=f"UTC end date YYYY-MM-DD. MUST end at least "
        f"config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS "
        f"({config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS}) days before now: Phase 9 "
        f"owns a holdout no diagnostic may see",
    )
    dr_parser.add_argument(
        "--detector", action="append",
        help="Registry key to measure (repeatable, e.g. --detector "
        "detector.falling-wedge); default is every Phase 8 detector",
    )
    dr_parser.add_argument(
        "--coverage", action="store_true",
        help="Print the coverage ledger summary and exit 0 without running "
        "anything. Touches neither the OHLCV database nor the trial ledger",
    )
    dr_parser.add_argument(
        "--out", default=None,
        help="Write the markdown report to this path in addition to stdout",
    )
    dr_parser.add_argument(
        "--state-db", default=None,
        help="Path to the framework state database (default: config.STATE_DB_PATH)",
    )

    # UI subcommand (v0.3.0 Phase 7).
    ui_parser = subparsers.add_parser(
        "ui", help="Serve the local builder UI (loopback only, no auth)"
    )
    ui_parser.add_argument(
        "--host", default=None,
        help=f"Bind host (default config.UI_HOST = {config.UI_HOST!r}); loopback "
        f"only — a non-loopback host is refused",
    )
    ui_parser.add_argument(
        "--port", type=int, default=None,
        help=f"Bind port (default config.UI_PORT = {config.UI_PORT})",
    )

    # Campaign subcommand (v0.3.0 Phase 9) — the pre-registered walk-forward
    # campaign and its ONE-SHOT holdout gate. Registered last, in phase order.
    campaign_parser = subparsers.add_parser(
        "campaign",
        help="Run the pre-registered v0.3.0 walk-forward campaign (Phase 9): "
        "evolution on the training span, then the one-shot holdout gate",
    )
    campaign_parser.add_argument(
        "--stage", choices=("probe", "evolve", "holdout", "report", "all"),
        default="all", help="Campaign stage to run (default: all)",
    )
    campaign_parser.add_argument(
        "--campaign-id", default=None,
        help="Resume an existing campaign; default derives a new id from the "
        "UTC date",
    )
    campaign_parser.add_argument(
        "--no-resume", action="store_true",
        help="Start a fresh campaign instead of continuing the latest one",
    )
    campaign_parser.add_argument(
        "--force-holdout-rerun", metavar="REASON", default=None,
        help="Re-run an already-consumed holdout. The reason is RECORDED and "
        "the report is stamped NOT A CLEAN HOLDOUT. Not a routine flag.",
    )
    campaign_parser.add_argument(
        "--state-db", default=None,
        help="Path to the framework state database (default: config.STATE_DB_PATH)",
    )

    args = parser.parse_args()

    if args.command == "backfill":
        conn = connect(args.db)

        # args.start is already epoch-ms (or None) thanks to type=_date_arg
        results = backfill_all(
            conn, symbols=args.symbol, timeframes=args.timeframe, start_ms=args.start
        )

        # Print results in tabular format
        print(f"{'Symbol':<10} {'Timeframe':<10} {'Rows':>8} {'Status':<24}")
        print("-" * 54)
        any_incomplete = False
        for (symbol, timeframe), result in results.items():
            if result.complete:
                status = "OK"
            else:
                any_incomplete = True
                status = f"INCOMPLETE({result.reason})"
            print(f"{symbol:<10} {timeframe:<10} {result.rows:>8} {status:<24}")

        conn.close()

        if any_incomplete:
            sys.exit(1)

    elif args.command == "poll":
        run_polling_loop(conn=connect(args.db))

    elif args.command == "gap-report":
        conn = connect(args.db)
        # args.as_of / args.start are already epoch-ms (or None) thanks to type=_date_arg
        exit_code = _gap_report_command(conn, now_ms=args.as_of, start_ms=args.start)
        conn.close()
        sys.exit(exit_code)

    elif args.command == "regime":
        conn = connect(args.db)
        # args.as_of is already epoch-ms (or None) thanks to type=_date_arg
        symbols = args.symbol if args.symbol else config.SYMBOLS
        exit_code = _regime_command(conn, symbols, now_ms=args.as_of)
        conn.close()
        sys.exit(exit_code)

    elif args.command == "signal":
        conn = connect(args.db)
        # args.as_of is already epoch-ms (or None) thanks to type=_date_arg
        symbols = args.symbol if args.symbol else config.SYMBOLS
        exit_code = _signal_command(conn, symbols, now_ms=args.as_of)
        conn.close()
        sys.exit(exit_code)

    # plugins / graph-validate deliberately take NO DB connection: a read-only
    # listing that creates data/ohlcv.db as a side effect is a trap.
    elif args.command == "plugins":
        sys.exit(_plugins_command(kind=args.kind))

    elif args.command == "graph-validate":
        sys.exit(_graph_validate_command(args.path))

    elif args.command in ("backtest", "walkforward", "benchmark"):
        conn = connect(args.db)
        symbols = args.symbol if args.symbol else config.SYMBOLS
        start_ms = args.start if args.start else config.date_to_ms(config.BACKFILL_START)
        end_ms = args.end if args.end else int(time.time() * 1000)
        if args.command == "backtest":
            exit_code = _backtest_command(conn, symbols, start_ms=start_ms, end_ms=end_ms)
        elif args.command == "benchmark":
            exit_code = _benchmark_command(conn, symbols, start_ms=start_ms, end_ms=end_ms)
        else:
            exit_code = _walkforward_command(
                conn, symbols, start_ms=start_ms, end_ms=end_ms,
                graph_path=args.graph,
            )
        conn.close()
        sys.exit(exit_code)

    elif args.command == "graph-backtest":
        conn = connect(args.db)
        symbols = args.symbol if args.symbol else config.SYMBOLS
        start_ms = args.start if args.start else config.date_to_ms(config.BACKFILL_START)
        end_ms = args.end if args.end else int(time.time() * 1000)
        exit_code = _graph_backtest_command(
            conn, symbols,
            graph_path=args.graph, start_ms=start_ms, end_ms=end_ms,
            audit=args.audit, rr_report=args.rr_report,
        )
        conn.close()
        sys.exit(exit_code)

    elif args.command == "correlation-report":
        conn = connect(args.db)
        symbols = args.symbol if args.symbol else None
        exit_code = _correlation_command(
            conn,
            symbols,
            timeframe=args.timeframe,
            start_ms=args.start,
            end_ms=args.end,
            select_n=args.select_n,
            out=args.out,
            check_exchange=args.check_exchange,
        )
        conn.close()
        sys.exit(exit_code)

    elif args.command == "review":
        conn = connect(args.db)
        state_conn = connect_state(args.state_db)
        symbols = args.symbol if args.symbol else config.SYMBOLS
        start_ms = args.start if args.start else config.date_to_ms(config.BACKFILL_START)
        end_ms = args.end if args.end else int(time.time() * 1000)
        forward_span = None
        if args.forward_start is not None and args.forward_end is not None:
            try:
                forward_span = feedback_protocol.forward_span(
                    history_start_ms=start_ms, forward_start_ms=args.forward_start,
                    forward_end_ms=args.forward_end,
                )
            except feedback_protocol.ForwardSpanError as exc:
                print(f"ERROR: {exc}")
                conn.close()
                state_conn.close()
                sys.exit(1)
        try:
            exit_code = _review_command(
                conn, state_conn, symbols,
                strategy_path=args.strategy, version_id=args.version,
                register=args.register, start_ms=start_ms, end_ms=end_ms,
                diagnose_only=args.diagnose_only, loop=args.loop,
                forward=forward_span, reviewer=args.reviewer,
                persist=not args.no_persist,
            )
        finally:
            conn.close()
            state_conn.close()
        sys.exit(exit_code)

    elif args.command == "evolve":
        # Deliberately NOT folded into the backtest/walkforward branch: that one
        # defaults end_ms to NOW, and this phase needs the frozen training
        # ceiling. An implicit "now" here would swallow Phase 9's holdout.
        sys.exit(
            _evolve_command(
                seed_graph_path=args.seed_graph,
                seed=args.seed,
                symbols=args.symbol,
                population=args.population,
                generations=args.generations,
                workers=args.workers,
                window_days=args.window_days,
                train_start_ms=args.train_start,
                train_end_ms=args.train_end,
                budget_hours=args.budget_hours,
                calibrate=args.calibrate,
                repeats=args.repeats,
                resume=args.resume,
                report=args.report,
                dry_run=args.dry_run,
                label=args.label,
                strategy_name=args.strategy_name,
                extend=args.extend,
                state_db=args.state_db,
                ohlcv_db=args.db,
            )
        )

    # detector-report (v0.3.0 Phase 8). --coverage opens NO database: a
    # read-only ledger listing that creates data/ohlcv.db as a side effect is a
    # trap, the same reason `plugins` takes no connection.
    elif args.command == "detector-report":
        if args.coverage:
            sys.exit(
                _detector_report_command(
                    None, None, start_ms=None, end_ms=None, coverage=True,
                    out_path=args.out,
                )
            )
        conn = connect(args.db)
        symbols = args.symbol if args.symbol else config.SYMBOLS
        start_ms = args.start if args.start else config.date_to_ms(config.BACKFILL_START)
        end_ms = args.end if args.end else int(time.time() * 1000)
        exit_code = _detector_report_command(
            conn, symbols,
            start_ms=start_ms, end_ms=end_ms,
            detectors=args.detector, out_path=args.out,
            state_db=args.state_db,
        )
        conn.close()
        sys.exit(exit_code)

    elif args.command == "ui":
        sys.exit(_ui_command(host=args.host, port=args.port))

    # campaign (v0.3.0 Phase 9). Needs TWO connections and keeps them separate:
    # contract §6's whole point is that a corrupt experiment log cannot endanger
    # 117 MB of irreplaceable price history, so state_conn is never passed where
    # a bar-loading function expects conn.
    elif args.command == "campaign":
        conn = connect(args.db)
        state_conn = connect_state(args.state_db)
        exit_code = _campaign_command(
            conn, state_conn,
            stage=args.stage,
            campaign_id=args.campaign_id,
            resume=not args.no_resume,
            force_reason=args.force_holdout_rerun,
        )
        state_conn.close()
        conn.close()
        sys.exit(exit_code)

    else:
        parser.print_help()


def _ui_command(*, host: str | None = None, port: int | None = None) -> int:
    """Serve the builder UI until interrupted.

    Deliberately opens NO shared connection: ui.api opens and closes its own
    per request and per job, since ThreadingHTTPServer gives each request its
    own thread and one sqlite cursor cannot be shared across them.

    The import is function-local on purpose: every other subcommand would
    otherwise pay for the http.server/socketserver chain on `cli backfill`.

    Returns:
        0 on clean shutdown (Ctrl-C), 2 on a refused bind (non-loopback host
        or port already in use).
    """
    from trading_bot.ui.server import serve

    return serve(host=host, port=port)


def _date_arg(value: str) -> int:
    """Parse a CLI date argument into epoch milliseconds.

    Args:
        value: ISO date string in format YYYY-MM-DD.

    Returns:
        Epoch milliseconds since Unix epoch, UTC.

    Raises:
        argparse.ArgumentTypeError: If value is not a valid YYYY-MM-DD date;
            argparse turns this into a clean usage error (exit code 2)
            instead of an uncaught traceback.
    """
    try:
        return config.date_to_ms(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        )


def _gap_report_command(
    conn, *, now_ms: int | None = None, start_ms: int | None = None
) -> int:
    """
    Run gap-report logic: check all symbol/timeframe combinations for gaps.

    Args:
        conn: Database connection.
        now_ms: Current time in epoch milliseconds for staleness detection.
                If None, uses current time.
        start_ms: Expected start of each series, in epoch milliseconds.
                If None, find_gaps defaults to config.date_to_ms(config.BACKFILL_START).

    Returns:
        0 if no gaps found anywhere, 1 if any gaps exist.
    """
    has_gaps = False
    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            gaps = find_gaps(conn, symbol, timeframe, now_ms=now_ms, start_ms=start_ms)
            if gaps:
                # Format gaps: [start_ts-end_ts, ...]
                gap_ranges = [f"{g[0]}-{g[1]}" for g in gaps]
                print(f"{symbol} {timeframe}: {len(gaps)} gap(s): {gap_ranges}")
                has_gaps = True
            else:
                print(f"{symbol} {timeframe}: OK")
    return 1 if has_gaps else 0


def _regime_command(conn, symbols: list[str] | tuple[str, ...], *, now_ms: int | None = None) -> int:
    """
    Classify market regime for given symbols.

    Prints an aligned table with Symbol, ADX, ATR%ile, and Regime columns.
    Returns 1 if any symbol has insufficient data ("uncertain" regime), 0 otherwise.

    Args:
        conn: Database connection.
        symbols: Iterable of symbols to classify.
        now_ms: Current time in epoch milliseconds. If None, uses current time.

    Returns:
        0 if all symbols have valid regimes, 1 if any symbol is "uncertain" due to insufficient data.
    """
    # Print table header
    print(f"{'Symbol':<10} {'ADX':>8} {'ATR%ile':>8} {'Regime':<18}")
    print("-" * 45)

    any_uncertain = False
    for symbol in symbols:
        label, adx_val, atr_pct_val = current_regime(conn, symbol, now_ms=now_ms)

        # Format ADX and ATR percentile; use "--" for NaN
        adx_str = f"{adx_val:.1f}" if not pd.isna(adx_val) else "--"
        atr_pct_str = f"{atr_pct_val:.2f}" if not pd.isna(atr_pct_val) else "--"

        print(f"{symbol:<10} {adx_str:>8} {atr_pct_str:>8} {label:<18}")

        if label == "uncertain":
            any_uncertain = True

    return 1 if any_uncertain else 0


def _signal_command(
    conn, symbols: list[str] | tuple[str, ...], *, now_ms: int | None = None
) -> int:
    """
    Scan symbols with the regime-matched signal method and print a table.

    Dispatches per symbol via scan_symbol (trending: Donchian channel breakout;
    ranging: mean-reversion fade; other regimes: no method). One row per
    signal; symbols with no signal print a single placeholder row showing why
    (inactive regime, or no triggered setup). Mirrors
    _regime_command's exit-code semantics: 1 if any symbol's regime is
    "uncertain" (insufficient data), 0 otherwise.

    Args:
        conn: Database connection.
        symbols: Iterable of symbols to scan.
        now_ms: Evaluation time in epoch milliseconds. If None, uses current time.

    Returns:
        0 if all symbols were scanned with valid regimes, 1 if any was "uncertain".
    """
    header = (
        f"{'Symbol':<10} {'Regime':<18} {'Pattern':<28} {'Dir':<6} "
        f"{'Entry':>12} {'SL':>12} {'TP':>12} {'R:R':>6} {'Vol×':>6}"
    )
    print(header)
    print("-" * len(header))

    any_uncertain = False
    for symbol in symbols:
        regime_label, signals = scan_symbol(conn, symbol, now_ms=now_ms)
        if regime_label == "uncertain":
            any_uncertain = True

        if not signals:
            note = "--" if regime_label in ("trending", "ranging") else "(inactive)"
            print(
                f"{symbol:<10} {regime_label:<18} {note:<28} {'--':<6} "
                f"{'--':>12} {'--':>12} {'--':>12} {'--':>6} {'--':>6}"
            )
            continue

        for s in signals:
            vol_str = f"{s.volume_ratio:.2f}" if not pd.isna(s.volume_ratio) else "--"
            pattern = s.pattern + ("*" if s.volume_high else "")
            print(
                f"{s.symbol:<10} {regime_label:<18} {pattern:<28} {s.direction:<6} "
                f"{s.entry:>12.4f} {s.stop:>12.4f} {s.target:>12.4f} "
                f"{s.rr:>6.2f} {vol_str:>6}"
            )

    return 1 if any_uncertain else 0


def _fmt(v, spec=".4f") -> str:
    """Format a possibly-None metric value."""
    return format(v, spec) if v is not None else "--"


def _print_metrics(m: dict, indent: str = "") -> None:
    """Print one metrics dict (as produced by compute_metrics)."""
    print(
        f"{indent}trades={m['n_trades']}  win_rate={_fmt(m['win_rate'], '.2%')}  "
        f"expectancy={_fmt(m['expectancy_pct'], '.4%')}  "
        f"profit_factor={_fmt(m['profit_factor'], '.2f')}  "
        f"max_dd={_fmt(m['max_drawdown_pct'], '.4%')}"
    )


def _backtest_command(conn, symbols, *, start_ms: int, end_ms: int) -> int:
    """
    Run a single backtest per symbol and print metrics with bucket breakdown.

    Returns:
        0 always (a backtest with zero trades is a result, not an error).
    """
    for symbol in symbols:
        trades = run_backtest(conn, symbol, start_ms=start_ms, end_ms=end_ms)
        m = compute_metrics(trades)
        print(f"{symbol}:")
        _print_metrics(m, indent="  ")
        for bucket, bm in m["by_bucket"].items():
            print(f"    {bucket}:")
            _print_metrics(bm, indent="      ")
        em = compute_equity_metrics(trades, start_ms, end_ms)
        print(
            f"  equity: sharpe={_fmt(em['sharpe'], '.2f')}  "
            f"sortino={_fmt(em['sortino'], '.2f')}  "
            f"max_dd={_fmt(em['max_drawdown_pct'], '.2%')}  "
            f"ann_return={_fmt(em['ann_return_pct'], '.2%')}"
        )
    return 0


def _print_benchmark(b, indent: str = "") -> None:
    """Print one BenchmarkResult as a per-symbol + BASKET table."""
    print(
        f"{indent}{'symbol':<10} {'total':>10} {'ann':>9} {'sharpe':>8} "
        f"{'sortino':>8} {'max_dd':>8} {'n_days':>7}"
    )
    print(f"{indent}{'-' * 64}")
    for name, m in list(b.per_symbol.items()) + [("BASKET", b.basket)]:
        total = f"{m['total_return']:.4f}x" if m["total_return"] is not None else "--"
        print(
            f"{indent}{name:<10} {total:>10} {_fmt(m['ann_return_pct'], '+.2%'):>9} "
            f"{_fmt(m['sharpe'], '.3f'):>8} {_fmt(m['sortino'], '.3f'):>8} "
            f"{_fmt(m['max_drawdown_pct'], '.2%'):>8} {m['n_days']:>7}"
        )


def _benchmark_command(conn, symbols, *, start_ms: int, end_ms: int) -> int:
    """
    Print the buy-and-hold null (per symbol and equal-weight basket) over a span.

    Returns:
        0 when every requested symbol produced a full metric set, 1 otherwise
        (mirrors _regime_command's "1 when a symbol had insufficient data").
    """
    b = buy_and_hold(conn, symbols, start_ms=start_ms, end_ms=end_ms)
    _print_benchmark(b)

    n_days = b.basket["n_days"]
    if 0 < n_days < 365:
        # An annualized figure from a sub-year window is an extrapolation, not
        # a compound annual rate. Say so rather than let it be quoted.
        print(f"NOTE: ann is a {n_days}-day extrapolation, not a compound annual rate.")

    incomplete = [
        name
        for name, m in b.per_symbol.items()
        if any(m[k] is None for k in METRIC_KEYS)
    ]
    if incomplete:
        print(f"INCOMPLETE: {', '.join(incomplete)} (insufficient data in span)")
        return 1
    return 0


def _walkforward_command(
    conn, symbols, *, start_ms: int, end_ms: int, graph_path: str | None = None
) -> int:
    """
    Run the POOLED walk-forward protocol across all symbols and print fold,
    per-symbol OOS, and pooled Sharpe/DSR results.

    Args:
        conn: Database connection.
        symbols: Symbols to pool.
        start_ms / end_ms: Overall span (epoch ms).
        graph_path: Optional path to a serialized StrategyGraph. When given, every
            run routes through framework.execute.run_graph_backtest and the sweep
            is reduced to the single run-level axis max_hold_bars — a graph
            carries its own parameters, and walk_forward_pooled REFUSES a grid it
            would silently ignore (contract §5). The reduction is printed, because
            an operator must know the sweep changed shape.

    Returns:
        0 if the pooled gate passes, 1 otherwise (including a span too
        short to form a single fold, an unloadable graph, or a refused grid).
    """
    kwargs: dict = {}
    if graph_path:
        try:
            # load_all() FIRST: graph.validate resolves every node against the
            # registry, so an unloaded registry reports a valid graph as five
            # unknown plug-ins.
            registry.load_all()
            strategy = fgraph.load(graph_path)
        except FrameworkError as exc:
            print(f"ERROR: {exc}")
            return 1
        kwargs["strategy"] = strategy
        kwargs["grid"] = {"max_hold_bars": DEFAULT_GRID["max_hold_bars"]}
        print(
            f"graph: {strategy.name} {fgraph.short_hash(strategy)} "
            f"branches={len(strategy.branches)}"
        )
        print(
            f"grid REDUCED to {sorted(kwargs['grid'])} — a graph carries its own "
            f"parameters, so the exit-mode axes are expressed in the graph, not swept"
        )

    try:
        result = walk_forward_pooled(
            conn, symbols, start_ms=start_ms, end_ms=end_ms, **kwargs
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    for i, fold in enumerate(result.folds):
        tm = fold.test_metrics
        print(
            f"fold {i}: params={fold.best_params}  "
            f"train_exp={_fmt(fold.train_expectancy, '.4%')}  "
            f"pos_neighbours={_fmt(fold.positive_neighbour_fraction, '.2f')}  "
            f"neighbour_spread={_fmt(fold.neighbour_spread, '.4%')}  "
            f"test_exp={_fmt(tm['expectancy_pct'], '.4%')}  "
            f"test_trades={tm['n_trades']}"
        )
    print(
        f"final params: {result.final_params}  "
        f"max_hold_bars={result.final_max_hold_bars}"
    )
    print("one-shot OOS (pooled):")
    _print_metrics(result.oos_metrics, indent="  ")
    em = result.oos_equity
    print(
        f"  sharpe={_fmt(em['sharpe'], '.2f')}  sortino={_fmt(em['sortino'], '.2f')}  "
        f"dsr={_fmt(em['dsr'], '.4f')}  max_dd={_fmt(em['max_drawdown_pct'], '.2%')}  "
        f"ann_return={_fmt(em['ann_return_pct'], '.2%')}"
    )
    print("per-symbol OOS expectancy:")
    for symbol, exp in result.per_symbol_expectancy.items():
        flag = "OK" if (exp is not None and exp > 0) else "FAIL"
        print(f"  {symbol}: {_fmt(exp, '.4%')}  [{flag}]")
    print("benchmark (equal-weight basket, same OOS span):")
    _print_benchmark(result.benchmark, indent="  ")
    print(f"n_trials charged to DSR: {result.n_trials_used}")
    print("gate conditions:")
    # Iterate GATE_CONDITIONS, not result.gate: the tuple is the canonical
    # report order, and iterating it makes a missing key an immediate KeyError
    # rather than a quietly short report.
    for name in GATE_CONDITIONS:
        print(f"  {name:<24} {'PASS' if result.gate[name] else 'FAIL'}")
    print(f"GATE: {'PASS' if result.passed else 'FAIL'}")

    return 0 if result.passed else 1


def _plugins_command(*, kind: str | None = None) -> int:
    """
    List every registered framework plug-in with its declared parameters,
    degrees of freedom, and rationale.

    This is where the PRD's success metric "a new plug-in costs zero engine-core
    edits" is verified: a newly-added module under trading_bot/plugins/ appears
    here with no other file touched.

    The rationale is printed on its own continuation line rather than truncated
    into a column. The whole point of making it mandatory is that it is readable.

    Args:
        kind: Only list this registry kind (default: all).

    Returns:
        0 on success, 1 if load_all() failed (import errors are fatal by design).
    """
    try:
        registry.load_all()
    except FrameworkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    specs = [s for s in registry.REGISTRY.values() if kind is None or s.kind == kind]
    specs.sort(key=lambda s: (s.kind, s.name))

    header = f"{'key':<38} {'tier':>4} {'timeframes':<12} {'dof':>5}  params"
    print(header)
    print("-" * len(header))
    for s in specs:
        tier = "--" if s.tier is None else str(s.tier)
        tfs = ",".join(s.timeframes) if s.timeframes else "--"
        params = (
            ", ".join(f"{n}={p.default!r}" for n, p in s.params.items()) or "(none)"
        )
        print(f"{s.key:<38} {tier:>4} {tfs:<12} {s.combo_count():>5}  {params}")
        print(f"    rationale: {s.rationale}")

    by_kind_counts = {}
    for s in specs:
        by_kind_counts[s.kind] = by_kind_counts.get(s.kind, 0) + 1
    total_dof = sum(s.combo_count() for s in specs)
    print("-" * len(header))
    print(
        f"{len(specs)} plug-in(s): "
        + ", ".join(f"{n} {k}" for k, n in sorted(by_kind_counts.items()))
    )
    print(f"summed declared degrees of freedom: {total_dof}")
    print(
        "  degrees of freedom a graph could expose; the DSR is charged for "
        "evaluations actually performed, not for this number (contract §4)."
    )
    return 0


def _graph_validate_command(paths: list[str]) -> int:
    """
    Validate one or more serialized strategy graphs.

    Phase 7's UI writes these files; this is the pre-flight check. Prints the
    content hash so an operator can tell two graphs apart without diffing JSON.

    Args:
        paths: Paths to <name>.strategy.json files.

    Returns:
        0 if every graph validated, 1 if any failed.
    """
    try:
        registry.load_all()
    except FrameworkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        try:
            g = fgraph.load(path)
        except FrameworkError as exc:
            failed = True
            print(f"FAIL {path}")
            for line in str(exc).splitlines():
                print(f"     {line}")
            continue
        n_nodes = 1 + len(g.filters) + sum(2 + len(b.confirmations) for b in g.branches)
        print(
            f"OK   {g.name}  {fgraph.short_hash(g)}  branches={len(g.branches)}  "
            f"nodes={n_nodes}  {path}"
        )
    return 1 if failed else 0


def _print_graph_header(g) -> None:
    """One-line-per-kind summary of a graph's nodes, so the audit output below it
    is unambiguous about which strategy produced it."""
    n_nodes = 1 + len(g.filters) + sum(2 + len(b.confirmations) for b in g.branches)
    print(
        f"graph: {g.name}  hash={fgraph.short_hash(g)}  "
        f"schema={g.schema_version}  branches={len(g.branches)}  nodes={n_nodes}"
    )
    print(f"  data       | {g.data.key}")
    for b in g.ordered_branches():
        state = "" if b.enabled else "  [DISABLED]"
        confs = ", ".join(c.key for c in b.confirmations) or "(none)"
        print(
            f"  branch {b.id!r} regimes={','.join(b.regimes)}{state}\n"
            f"    detector      | {b.detector.key}\n"
            f"    confirmations | {confs}\n"
            f"    policy        | {b.policy.key}"
        )
    filters = ", ".join(f.key for f in g.filters) or "(none)"
    print(f"  filters    | {filters}  (RR_TARGET_MIN={config.RR_TARGET_MIN})")


def _print_audit(trades, indent: str = "    ") -> None:
    """One line per taken position with its R:R justification.

    risk_pct / reward_pct are derived from the Trade's own frozen entry/stop/
    target rather than carried as extra fields, so the audit cannot disagree with
    what the trade actually recorded.
    """
    if not trades:
        print(f"{indent}(no positions taken)")
        return
    for t in trades:
        risk_pct = abs(t.entry - t.stop) / t.entry if t.entry else float("nan")
        reward_pct = abs(t.target - t.entry) / t.entry if t.entry else float("nan")
        confs = ",".join(t.confirmations) if t.confirmations else "(none)"
        print(
            f"{indent}ts={t.entry_ts} {t.direction:<5} {t.pattern} "
            f"entry={t.entry:.6g} stop={t.stop:.6g} target={t.target:.6g}"
        )
        print(
            f"{indent}  risk={risk_pct:.4%} reward={reward_pct:.4%} "
            f"planned_rr={t.planned_rr:.4f} outcome={t.outcome} "
            f"pnl={t.pnl_pct:.4%}"
        )
        print(f"{indent}  confirmations={confs}")


def _print_rr_report(report: dict, indent: str = "    ") -> None:
    """Print rr_distribution_report's output.

    The sub-target thresholds are printed FOR DIAGNOSIS ONLY. Printing 1.5 is not
    permission to use 1.5 — see plugins/filters/rr_after_costs.py's
    pre-registered decision rule.
    """
    n = report["n_plans"]
    rate = report["survival_rate"]
    print(f"{indent}n_plans={n}  (every PositionPlan that reached the filter)")
    if n == 0:
        print(f"{indent}no plans reached the filter — nothing to summarise")
        return
    passes = "  ".join(
        f"{t:.2f}:{report['n_pass'][t]}" for t in sorted(report["n_pass"], reverse=True)
    )
    print(f"{indent}n_pass by NET threshold (diagnosis only): {passes}")
    print(
        f"{indent}survival_rate at RR_TARGET_MIN={report['rr_target_min']:.2f}: "
        f"{rate:.4%}  ({report['n_pass_target']}/{n})"
    )
    if report["n_non_finite_net"]:
        print(f"{indent}non-finite net_rr excluded from quantiles: "
              f"{report['n_non_finite_net']}")
    for label in ("net", "gross", "gross_rr_required"):
        q = report[label]
        if q["n"] == 0:
            print(f"{indent}{label:<18} (no finite values)")
            continue
        deciles = " ".join(f"{v:.3f}" for v in q["deciles"])
        print(
            f"{indent}{label:<18} min={q['min']:.4f} median={q['median']:.4f} "
            f"max={q['max']:.4f}"
        )
        print(f"{indent}{'':<18} deciles(10..90%)= {deciles}")


def _graph_backtest_command(
    conn,
    symbols,
    *,
    graph_path: str,
    start_ms: int,
    end_ms: int,
    audit: bool = False,
    rr_report: bool = False,
) -> int:
    """
    Replay a serialized strategy graph over stored history, per symbol and pooled.

    This is a BACKTEST, not a validation run. Contract §4: fitness comes only from
    walkforward.walk_forward_pooled, and nothing here queries that oracle or
    increments Phase 1's trial ledger. Numbers printed here are in-sample
    diagnostics and must never be reported as validated results.

    Args:
        conn: Database connection.
        symbols: Symbols to replay.
        start_ms / end_ms: Span (epoch ms).
        audit: Print every taken position with its R:R justification.
        rr_report: Measure the cost-adjusted R:R distribution over every plan
            that reached filter.rr-after-costs.

    Returns:
        0 for any successful run, ZERO TRADES INCLUDED — quoting
        _backtest_command's rule: "a backtest with zero trades is a result, not
        an error". 1 only for an unreadable or invalid graph.
    """
    try:
        # load_all() FIRST: graph.validate resolves every node against the
        # registry, so an unloaded registry reports a valid graph as N unknown
        # plug-ins. Import errors are FATAL, never skipped (contract §3).
        registry.load_all()
        g = fgraph.load(graph_path)
    except FrameworkError as exc:
        print(f"ERROR: {exc}")
        return 1

    _print_graph_header(g)

    pooled: list = []
    pooled_verdicts: list = []
    for symbol in symbols:
        if rr_report:
            with rr_after_costs.recording() as tape:
                trades = run_graph_backtest(
                    conn, g, symbol, start_ms=start_ms, end_ms=end_ms
                )
            verdicts = list(tape)
        else:
            trades = run_graph_backtest(
                conn, g, symbol, start_ms=start_ms, end_ms=end_ms
            )
            verdicts = []
        pooled.extend(trades)
        pooled_verdicts.extend(verdicts)

        m = compute_metrics(trades)
        print(f"{symbol}:")
        _print_metrics(m, indent="  ")
        for bucket, bm in m["by_bucket"].items():
            print(f"    {bucket}:")
            _print_metrics(bm, indent="      ")
        em = compute_equity_metrics(trades, start_ms, end_ms)
        print(
            f"  equity: sharpe={_fmt(em['sharpe'], '.2f')}  "
            f"sortino={_fmt(em['sortino'], '.2f')}  "
            f"max_dd={_fmt(em['max_drawdown_pct'], '.2%')}  "
            f"ann_return={_fmt(em['ann_return_pct'], '.2%')}"
        )
        if audit:
            print("  audit (every taken position, with its R:R justification):")
            _print_audit(trades)
        if rr_report:
            print("  rr-report:")
            _print_rr_report(rr_after_costs.rr_distribution_report(verdicts))

    print("POOLED:")
    pm = compute_metrics(pooled)
    _print_metrics(pm, indent="  ")
    # Pooled per-DETECTOR breakdown. by_bucket keys are "regime/pattern", so
    # re-bucketing on pattern alone is what answers "does this detector have an
    # edge" across regimes and symbols at once — contract §9's edge report,
    # produced by a committed command rather than by hand.
    patterns = sorted({t.pattern for t in pooled})
    if len(patterns) > 1:
        print("  by detector (pooled across symbols and regimes):")
        for pattern in patterns:
            subset = [t for t in pooled if t.pattern == pattern]
            print(f"    {pattern}:")
            _print_metrics(compute_metrics(subset), indent="      ")
    pem = compute_equity_metrics(pooled, start_ms, end_ms)
    print(
        f"  equity: sharpe={_fmt(pem['sharpe'], '.2f')}  "
        f"sortino={_fmt(pem['sortino'], '.2f')}  "
        f"max_dd={_fmt(pem['max_drawdown_pct'], '.2%')}  "
        f"ann_return={_fmt(pem['ann_return_pct'], '.2%')}"
    )
    if pooled:
        below = [t for t in pooled if t.planned_rr < config.RR_TARGET_MIN]
        print(
            f"  planned_rr >= RR_TARGET_MIN ({config.RR_TARGET_MIN}) on "
            f"{len(pooled) - len(below)}/{len(pooled)} taken positions"
            + (f"  VIOLATIONS: {len(below)}" if below else "")
        )
    if rr_report:
        print("  rr-report (pooled):")
        _print_rr_report(rr_after_costs.rr_distribution_report(pooled_verdicts))
    return 0


def _correlation_command(
    conn,
    symbols: list[str] | None,
    *,
    timeframe: str,
    start_ms: int | None,
    end_ms: int | None,
    select_n: int,
    out: str | None,
    check_exchange: bool,
) -> int:
    """
    Measure correlation / effective-N over the stored universe, select
    RESEARCH_SYMBOLS candidates, print a compact summary, and optionally
    write the full markdown report.

    --as-of is deliberately NOT offered here (contrast gap-report /
    regime / signal): correlation.gap_integrity neutralises poller staleness
    internally by pinning now_ms per cell, so an as-of date would have no
    effect and offering one would imply otherwise.

    Returns:
        1 if --check-exchange was requested and the probe failed, or the
        decision is D3, or any SELECTED symbol has an interior gap; 0
        otherwise. D1 and D2 both exit 0 -- D2 is an honest finding, not a
        command failure, and exiting 1 there would train the operator to
        ignore the exit code.
    """
    exchange_failed = False
    if check_exchange:
        import ccxt

        from trading_bot.exchange.binance_client import fetch_ohlcv_page, to_ccxt_symbol

        try:
            rows = fetch_ohlcv_page(
                to_ccxt_symbol(config.CORRELATION_ANCHOR_SYMBOL),
                timeframe,
                since_ms=config.date_to_ms(config.BACKFILL_START),
                limit=5,
            )
            print(f"exchange: OK ({len(rows)} rows, first ts {rows[0][0] if rows else '--'})")
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            print(f"exchange: UNREACHABLE ({type(exc).__name__}: {exc})")
            exchange_failed = True
    else:
        print("exchange: (not checked; pass --check-exchange)")

    report = correlation.build_report(
        conn,
        symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        timeframe=timeframe,
        select_n=select_n,
    )

    print(
        f"span: {report.start_ms} -> {report.end_ms}  timeframe: {report.timeframe}  "
        f"returns: {report.n_obs}  symbols: {len(report.symbols)}"
    )
    print()
    clean = sum(1 for v in report.integrity.values() if not v)
    print(f"integrity (interior gaps over {'/'.join(correlation.INTEGRITY_TIMEFRAMES)}):")
    print(
        f"  {clean} of {len(report.integrity)} symbols clean, "
        f"{sum(len(v) for v in report.integrity.values())} interior gaps in "
        f"{len(report.integrity) * len(correlation.INTEGRITY_TIMEFRAMES)} cells"
    )
    print()
    print("effective independent sample size (Kish, r_bar-based):")
    labels = {"production": "config.SYMBOLS", "stored": "all stored", "selection": "selection"}
    for key in ("production", "stored", "selection"):
        if key in report.effective_n:
            m, r_bar, n_kish, n_part = report.effective_n[key]
            print(
                f"  {labels[key]:<16} n={m:<4} r_bar={r_bar:.4f}   N_eff={n_kish:.3f}   "
                f"(participation ratio {n_part:.3f})"
            )
    print()
    print("selection (rule: liquidity floor, beta ceiling, rank by mean pairwise r, "
          "BTC pinned):")
    print(f"  {' '.join(report.selection) if report.selection else '(none: see verdict)'}")
    print()
    print(f"VERDICT: {report.decision}")
    print(f"  {report.verdict}")

    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(correlation.format_report(report))
        print(f"wrote {out}")

    selected_has_gap = any(
        len(report.integrity.get(s, [])) > 0 for s in report.selection
    )
    if exchange_failed or report.decision == "D3" or selected_has_gap:
        return 1
    return 0


def _print_review_table(review_records) -> None:
    """One line per closed-trade review, mirroring the UX example in the
    Phase 5 plan. `_fmt` handles every possibly-None value; no second
    formatter is added."""
    print(
        f"{'Symbol':<10} {'Dir':<5} {'Outcome':<8} {'pnl%':>8}  {'MFE%':>6} "
        f"{'MAE%':>6}  {'TPcap':>6} {'TPverdict':<15} {'SLhead':>6} "
        f"{'SLverdict':<10} {'Pace':<8}"
    )
    for r in review_records:
        print(
            f"{r.symbol:<10} {r.direction:<5} {r.outcome:<8} "
            f"{r.pnl_pct * 100:>8.4f}  {_fmt(r.mfe_pct, '.2%'):>6} "
            f"{_fmt(r.mae_pct, '.2%'):>6}  {_fmt(r.tp_capture_ratio, '.2f'):>6} "
            f"{r.tp_verdict:<15} {_fmt(r.sl_headroom_ratio, '.2f'):>6} "
            f"{r.sl_verdict:<10} {r.pace_verdict:<8}"
        )


def _print_diagnosis(d) -> None:
    n = d.n_records
    print(f"diagnosis (n={n}, {d.span_class}):")
    outcome_bits = "  ".join(
        f"{k} {v} ({_fmt(d.outcome_mean_pnl.get(k), '.2%')})" for k, v in sorted(d.by_outcome.items())
    )
    print(f"  outcome mix  : {outcome_bits or '(none)'}")
    print(
        f"  TP           : median capture {_fmt(d.median_tp_capture, '.2f')}  "
        + "  ".join(f"{k} {v}" for k, v in d.tp_verdicts.items() if v)
    )
    print(
        f"  SL           : median headroom {_fmt(d.median_sl_headroom, '.2f')}  "
        + "  ".join(f"{k} {v}" for k, v in d.sl_verdicts.items() if v)
    )
    p = d.pace
    print(
        f"  pace         : ann_return={_fmt(p.get('ann_return_pct'), '.2%')} "
        f"target={_fmt(p.get('target_ann_return'), '.2%')} n={p.get('n_trades')} "
        f"over {p.get('n_days')}d sample_adequate={p.get('sample_adequate')}"
    )
    if d.confirmation_coverage:
        _STATUS_LABEL = {
            "dead-weight": " (DEAD WEIGHT)",
            # Hard-gate case (records.py module docstring): full coverage by
            # construction, no rejected-event counterfactual to decide
            # "dead weight" from -- NOT the same claim, and deliberately
            # never turns into a drop-confirmation suggestion.
            "unmeasurable": " (UNMEASURABLE — no rejected-event counterfactual)",
            "": "",
        }
        conf_bits = "  ".join(
            f"{name} coverage={cov:.2f}"
            + _STATUS_LABEL.get(d.confirmation_status.get(name, ""), "")
            for name, cov in d.confirmation_coverage.items()
        )
        print(f"  confirmations: {conf_bits}")
    print(f"  suggestions  : {', '.join(d.suggestions) if d.suggestions else '(none)'}")
    print(f"  digest       : {d.digest}")


def _print_forward_result(result) -> None:
    for name in ("sample_adequacy", "sharpe", "dsr", "max_drawdown",
                 "per_symbol_expectancy", "beats_benchmark_return", "beats_benchmark_sharpe"):
        print(f"  {name:<24} {'PASS' if result.gate.get(name) else 'FAIL'}")
    print(f"  GATE: {'PASS' if result.passed else 'FAIL'}  n_trials={result.n_trials_used}")


def _review_command(
    conn, state_conn, symbols, *,
    strategy_path: str | None, version_id: str | None, register: bool,
    start_ms: int, end_ms: int, diagnose_only: bool, loop: bool,
    forward, reviewer: str, persist: bool,
) -> int:
    """Feedback loop MVP dispatch (v0.3.0 Phase 5). See cli.py's `review`
    subparser help and the phase plan's UX section for the exact modes.

    Returns:
        0 on a successful run (including zero records — cli.py's rule that
        an empty result is a result, not an error — EXCEPT --diagnose-only,
        where zero stored records is something the operator should notice).
        1 on an unreadable/invalid graph, an unknown version, a refused
        forward span, or (forward mode) a failed gate.
    """
    # load_all() FIRST: resolving --reviewer or validating a graph against an
    # unloaded registry reports valid plug-ins as unknown. Import errors are
    # FATAL, never skipped (contract §3) — no try/except around this call.
    registry.load_all()

    if register:
        if not strategy_path:
            print("ERROR: --register requires --strategy PATH")
            return 1
        try:
            g = fgraph.load(strategy_path)
        except FrameworkError as exc:
            print(f"ERROR: {exc}")
            return 1
        v = feedback_versioning.register_version(state_conn, g, label=g.name)
        print(
            f"registered strategy_version={v.version_id}  "
            f"parent={v.parent_id or '--'}  provenance={v.provenance.get('source', '--')}"
        )
        return 0

    try:
        if strategy_path:
            g = fgraph.load(strategy_path)
            v = feedback_versioning.register_version(state_conn, g, label=g.name)
            resolved_version_id = v.version_id
        else:
            resolved_version_id = version_id
            feedback_versioning.get_version(state_conn, resolved_version_id)  # existence check
    except FrameworkError as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 1

    reviewer_key = reviewer if "." in reviewer else f"reviewer.{reviewer}"

    if diagnose_only:
        review_records = feedback_records.load_records(
            state_conn, strategy_version=resolved_version_id, span_class="in-sample",
            start_ms=start_ms, end_ms=end_ms,
        )
        if not review_records:
            print(f"no stored review_records for strategy_version={resolved_version_id}")
            return 1
        d = feedback_records.diagnose(review_records, start_ms=start_ms, end_ms=end_ms)
        if d.span_class == "in-sample":
            print("IN-SAMPLE DIAGNOSTIC — NOT EVIDENCE (span was available to tuning)")
        _print_review_table(review_records)
        _print_diagnosis(d)
        return 0

    _ = loop  # documented no-op today (Task 11): default and --loop both run
    # the full review+diagnose+refine protocol; see protocol.py's
    # run_loop_iteration docstring.
    try:
        result = feedback_protocol.run_loop_iteration(
            conn, state_conn, version_id=resolved_version_id, symbols=symbols,
            review_start_ms=start_ms, review_end_ms=end_ms, span_class="in-sample",
            forward=forward, reviewer=reviewer_key, persist=persist,
        )
    except feedback_protocol.ForwardSpanError as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("IN-SAMPLE DIAGNOSTIC — NOT EVIDENCE (span was available to tuning)")
    if persist:
        parent_records = feedback_records.load_records(
            state_conn, strategy_version=resolved_version_id, span_class="in-sample",
            start_ms=start_ms, end_ms=end_ms,
        )
        _print_review_table(parent_records)
    print(f"{result.records_written} record(s) written to state.db")
    _print_diagnosis(result.diagnosis)

    if result.child_version_id:
        print(
            f"refine: applied {', '.join(result.suggestions_applied)} -> "
            f"child {result.child_version_id} (parent {result.parent_version_id})"
        )
    else:
        print(f"refine: {result.notes}")

    if result.forward is not None:
        fwd = result.forward
        print(
            f"forward test {fwd.span.forward_start_ms}..{fwd.span.forward_end_ms} "
            f"via THE GATE (n_trials={fwd.n_trials_used}, cumulative)"
        )
        _print_forward_result(fwd)
        print(f"{len(fwd.trades)} forward review record(s) written" if persist else
              f"{len(fwd.trades)} forward trade(s) (not persisted)")
        return 0 if fwd.passed else 1

    return 0


def _evolve_command(
    *,
    seed_graph_path: str | None = None,
    seed: int | None = None,
    symbols=None,
    population: int | None = None,
    generations: int | None = None,
    workers: int | None = None,
    window_days: int | None = None,
    train_start_ms: int | None = None,
    train_end_ms: int | None = None,
    budget_hours: float | None = None,
    calibrate: bool = False,
    repeats: int = 3,
    resume: str | None = None,
    report: str | None = None,
    dry_run: bool = False,
    label: str | None = None,
    strategy_name: str | None = None,
    extend: int | None = None,
    state_db: str | None = None,
    ohlcv_db: str | None = None,
) -> int:
    """Run, resume, calibrate or reprint an evolution campaign (v0.3.0 Phase 6).

    Exit codes:
        0 — completed AND the audit round shows the best candidate beating the
            seed (or a successful --calibrate / --report / --dry-run).
        1 — completed with no improvement over the seed.
        2 — aborted, or a fatal GraphError / HoldoutViolation / bad arguments.

    Note the gate verdict is NOT the exit code. "Beat the seed" and "passed THE
    GATE" are different questions, and a campaign that improves on the seed while
    still failing the gate on dsr/sample_adequacy is the EXPECTED outcome
    (KNOWN-LIMITATIONS §1: 0.742 even at n_trials=1). Both are printed.
    """
    state_conn = connect_state(state_db)
    try:
        if report:
            return _evolve_report(state_conn, report)

        if calibrate:
            if not seed_graph_path:
                print("ERROR: --calibrate needs --seed-graph")
                return 2
            try:
                registry.load_all()
                graph = fgraph.load(seed_graph_path)
            except FrameworkError as exc:
                print(f"ERROR: {exc}")
                return 2
            print(f"calibrating on {graph.name} {fgraph.short_hash(graph)}")
            try:
                evo_runner.calibrate(
                    seed_graph=graph, symbols=symbols, repeats=repeats,
                    workers=workers, budget_hours=budget_hours,
                    window_days=window_days, state_conn=state_conn,
                    ohlcv_path=ohlcv_db, progress=print,
                )
            except ValueError as exc:
                print(f"ERROR: {exc}")
                return 2
            return 0

        graph = None
        if not resume:
            if not seed_graph_path:
                print("ERROR: --seed-graph is required unless --resume or --report")
                return 2
            try:
                # load_all() FIRST: graph.validate resolves every node against the
                # registry, so an unloaded registry reports a valid graph as N
                # unknown plug-ins.
                registry.load_all()
                graph = fgraph.load(seed_graph_path)
            except FrameworkError as exc:
                print(f"ERROR: {exc}")
                return 2

        try:
            summary = evo_runner.run_campaign(
                seed_graph=graph, seed=seed, symbols=symbols,
                population_size=population, generations=generations,
                workers=workers, window_days=window_days,
                train_start_ms=train_start_ms, train_end_ms=train_end_ms,
                resume_campaign_id=resume, state_conn=state_conn,
                label=label, strategy_name=strategy_name,
                extend_generations=extend,
                state_path=state_db, ohlcv_path=ohlcv_db,
                dry_run=dry_run, progress=print,
            )
        except (evo_oracle.HoldoutViolation, FrameworkError) as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            return 2
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2

        print(
            f"trials charged {summary['trials_charged']}   "
            f"ledger total {summary['ledger_total']}   "
            f"(distinct {summary['ledger_distinct']}, REPORTED ONLY — contract §4 "
            f"charges the DSR every evaluation performed)"
        )
        if summary["status"] != "done":
            print(f"CAMPAIGN {summary['status'].upper()}: resume with --resume "
                  f"{summary['campaign_id']}")
            return 0 if summary["status"] == "dry-run" else 2

        verdict = summary["verdict"]
        best = verdict.get("best")
        if best is not None:
            print("best member's gate (audit window):")
            for name in GATE_CONDITIONS:
                print(f"  {name:<24} {'PASS' if best['gate'].get(name) else 'FAIL'}")
            print(
                f"  benchmark over the SAME span: ann_return="
                f"{_fmt(best['bench_ann_return_pct'], '.2%')} "
                f"sharpe={_fmt(best['bench_sharpe'], '.2f')}  <- a "
                f"'beats buy-and-hold' pass earned against a basket that LOST money "
                f"is not the protection contract §4 asks for"
            )
            print(
                f"  dsr={_fmt(best['dsr'], '.4f')} at n_trials="
                f"{best['n_trials_used']}  (fitness never uses dsr — contract §4)"
            )
        print(f"BEST BEATS SEED: {'YES' if verdict.get('beats_seed') else 'NO'}")
        print(
            f"GATE: {'PASS' if verdict.get('gate_passed') else 'FAIL'}"
            + (
                f" ({', '.join(verdict.get('failed_conditions') or [])})"
                if not verdict.get("gate_passed")
                else ""
            )
        )
        return 0 if verdict.get("beats_seed") else 1
    finally:
        state_conn.close()


def _evolve_report(state_conn, campaign_id: str) -> int:
    """Reprint a stored campaign. Evaluates nothing and charges nothing."""
    evo_population.ensure_schema(state_conn)
    campaign = evo_population.load_campaign(state_conn, campaign_id)
    if campaign is None:
        campaign = evo_population.campaign_by_label(state_conn, campaign_id)
    if campaign is None:
        print(f"ERROR: no campaign {campaign_id!r} in state.db (tried both "
              f"campaign_id and label)")
        return 2
    campaign_id = campaign.campaign_id
    print(f"campaign {campaign.campaign_id}  seed {campaign.seed}  "
          f"status {campaign.status}")
    if campaign.label or campaign.strategy_name:
        print(f"  label {campaign.label or '-'}  "
              f"strategy {campaign.strategy_name or '-'}")
    print(f"  symbols {','.join(campaign.symbols)}")
    print(f"  train {_ms_date(campaign.train_start_ms)} -> "
          f"{_ms_date(campaign.train_end_ms)}")
    print(f"  audit {_ms_date(campaign.audit_start_ms)} -> "
          f"{_ms_date(campaign.audit_end_ms)} (declared at campaign start)")
    print(f"  population {campaign.population} x generations {campaign.generations}")
    print(f"{'gen':>4} {'window':<25} {'evals':>6} {'err':>4} {'uniq':>5} "
          f"{'trials':>7} {'best fitness':>13} {'retries':>8} {'wall s':>8}")
    for row in evo_population.generation_rows(state_conn, campaign_id):
        window = (f"{_ms_date(row['window_start_ms'])}->"
                  f"{_ms_date(row['window_end_ms'])}")
        print(
            f"{row['gen_index']:>4} {window:<25} {row['n_evaluated']:>6} "
            f"{row['n_errors']:>4} {row['n_unique_graphs']:>5} "
            f"{row['trials_cumulative']:>7} "
            f"{_fmt(row['best_fitness'], '+.4f'):>13} {row['db_retries']:>8} "
            f"{_fmt(row['wall_seconds'], '.1f'):>8}"
        )
    finalists = evo_population.finalist_rows(state_conn, campaign_id)
    if finalists:
        print("audit round:")
        for row in finalists:
            print(
                f"  {row['member_id']}  tier {row['tier']}  "
                f"exSharpe {_fmt(row['excess_sharpe'], '+.4f')}  "
                f"sharpe {_fmt(row['sharpe'], '.4f')}  "
                f"dsr {_fmt(row['dsr'], '.4f')}  trades {row['n_trades']}  "
                f"n_trials {row['n_trials_used']}"
            )
    ledger = evo_oracle.TrialLedger(state_conn, campaign_id)
    print(f"ledger total {ledger.count()} (distinct {ledger.distinct_count()})")
    over = state_conn.execute(
        "SELECT COUNT(*) FROM trial_ledger WHERE campaign = ? AND end_ms > ?",
        (campaign_id, campaign.train_end_ms),
    ).fetchone()[0]
    print(
        f"ledger rows beyond the training ceiling: {over}  "
        f"(MUST be 0 — proof no generation touched Phase 9's holdout)"
    )
    return 0


def _ms_date(ms: int) -> str:
    """Epoch ms -> UTC YYYY-MM-DD, for report tables."""
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


# --------------------------------------------------------------------------- #
# v0.3.0 Phase 8: the per-detector edge report.
# --------------------------------------------------------------------------- #

# Detector modules this phase owns. Used ONLY to pick the default detector set,
# so Phase 3's and Phase 4's detectors are not silently re-measured here under a
# fixed exit policy that was never designed for them.
PHASE8_DETECTOR_MODULES = (
    "trading_bot.plugins.detectors.reversal",
    "trading_bot.plugins.detectors.continuation",
    "trading_bot.plugins.detectors.oscillator",
    "trading_bot.plugins.detectors.wyckoff_events",
)

DETECTOR_REPORT_BANNER = "=== DIAGNOSTIC -- NOT A GATE VERDICT " + "=" * 37


def _phase8_detector_keys() -> list[str]:
    """Every registered detector from a Phase 8 module, sorted by (tier, name).

    SORTED BY TIER THEN NAME, NEVER BY EXPECTANCY. Sorting a table of measured
    expectancies by expectancy IS selection, and selection that is not charged
    to the trial ledger is exactly how the gate becomes theater (contract §4).
    Pinned by test_report_is_sorted_by_tier_then_name_not_by_expectancy.
    """
    specs = [
        s
        for s in registry.REGISTRY.values()
        if s.kind == "detector" and s.module in PHASE8_DETECTOR_MODULES
    ]
    specs.sort(key=lambda s: (99 if s.tier is None else s.tier, s.name))
    return [s.key for s in specs]


def _diagnostic_graph(detector_key: str) -> "fgraph.StrategyGraph":
    """The canonical single-detector diagnostic graph.

    ohlcv DataSource -> ONE detector at its ParamSpec DEFAULTS ->
    policy.measured-move -> NO confirmations, NO filters.

    Nothing is swept and no parameter can be overridden from the command line:
    there is deliberately no --param flag, because a loop over parameter values
    would turn this diagnostic into a second fitness oracle and contract §4
    allows exactly one. The regime gate is DISABLED so a detector is measured on
    every bar it fires on rather than on the subset one regime label admits —
    the report is about the DETECTOR, not about the regime router.
    """
    name = detector_key.split(".", 1)[1]
    return fgraph.StrategyGraph(
        name=f"detector-report-{name}",
        data=fgraph.NodeSpec(id="ohlcv", key="data.ohlcv"),
        regime=fgraph.RegimeGate(enabled=False),
        branches=(
            fgraph.Branch(
                id="probe",
                detector=fgraph.NodeSpec(id="det", key=detector_key),
                policy=fgraph.NodeSpec(id="pol", key="policy.measured-move"),
                regimes=(fgraph.ANY_REGIME,),
            ),
        ),
        meta={"purpose": "phase8-detector-report", "diagnostic": True},
    )


def _cost_columns(trades) -> dict:
    """Measured mean round-trip cost and the cost ratio c, per contract §1.

    `c = mean_cost_pct / median_risk_pct` against config.COST_RATIO_CEILING.
    A detector whose c exceeds the ceiling is STRUCTURALLY unable to pay for
    itself regardless of win rate — which makes this the most useful column in
    the table, and it is measured from the trades rather than asserted.

    The cost is recomputed here from config exactly as engine.py charges it —
    `2*(FEE_PCT+SLIPPAGE_PCT) + FUNDING_PCT_PER_DAY*holding_days` — because
    Trade records the P&L net of cost but not the cost itself. This is a
    REPORTING column only: no P&L is computed here, and
    test_no_second_pnl_path pins that.
    """
    if not trades:
        return {"mean_cost_pct": None, "median_risk_pct": None, "cost_ratio": None}
    round_trip = 2 * (config.FEE_PCT + config.SLIPPAGE_PCT)
    costs, risks = [], []
    for t in trades:
        days = max(0.0, (t.exit_ts - t.entry_ts) / 86_400_000.0)
        costs.append(round_trip + config.FUNDING_PCT_PER_DAY * days)
        if t.entry:
            risks.append(abs(t.entry - t.stop) / t.entry)
    mean_cost = sum(costs) / len(costs)
    med_risk = float(pd.Series(risks).median()) if risks else None
    ratio = (mean_cost / med_risk) if (med_risk and med_risk > 0) else None
    return {
        "mean_cost_pct": mean_cost,
        "median_risk_pct": med_risk,
        "cost_ratio": ratio,
    }


def _detector_report_command(
    conn,
    symbols,
    *,
    start_ms: int | None,
    end_ms: int | None,
    detectors: list[str] | None = None,
    coverage: bool = False,
    out_path: str | None = None,
    state_db: str | None = None,
    now_ms: int | None = None,
) -> int:
    """
    Measure each Phase 8 detector's events, trades, hit rate and after-costs
    expectancy over the TUNING span, at fixed defaults.

    THIS IS A DIAGNOSTIC AND MUST NOT BECOME A FITNESS ORACLE. Six independent
    mechanisms keep it one, and removing any of them reopens the hole:

      1. The HOLDOUT GUARD runs first, before any work: an `end_ms` inside
         config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS of now returns 2 and measures
         nothing. Phase 9 owns a span no diagnostic may see. Mechanism, not
         promise.
      2. FIXED DEFAULTS. Every detector runs at its ParamSpec defaults; there is
         no --param flag and no loop over values.
      3. ONE P&L PATH. Trades come from framework.execute.run_graph_backtest with
         every cost argument left at None, so config.FEE_PCT / SLIPPAGE_PCT /
         FUNDING_PCT_PER_DAY apply exactly as production applies them. Passing
         explicit fees "for the report" would let report and production disagree.
      4. INSUFFICIENT SUPPRESSION. Below config.DETECTOR_MIN_EVENTS_FOR_REPORT
         trades, every rate and expectancy figure is suppressed.
         KNOWN-LIMITATIONS §1 records 23 trades failing a floor of 30; printing
         an expectancy off 9 trades repeats that error one level down.
      5. TRIAL LEDGER ROWS. One row per (detector, params, span) under
         config.DETECTOR_REPORT_CAMPAIGN. Contract §4.2: nothing scores a
         candidate without a ledger handle, and this report is not exempt.
      6. THE BANNER, which prints the SELECTION COST: reading this table is
         free, but the moment it informs which detectors enter a campaign, that
         campaign owes n_trials += <number of detectors evaluated>.

    Output is sorted by tier then name. Never by expectancy — see
    _phase8_detector_keys.

    Args:
        conn: OHLCV database connection. May be None when coverage=True.
        symbols: Symbols to measure. May be None when coverage=True.
        start_ms / end_ms: Tuning span, inclusive.
        detectors: Registry keys to restrict to (default: every Phase 8 detector).
        coverage: Print the ledger summary and exit without measuring anything.
        out_path: Also write the report as markdown here.
        state_db: Trial-ledger database path (default config.STATE_DB_PATH).
        now_ms: Injectable clock for the holdout guard, for tests only.

    Returns:
        0 when a report was produced — a detector with no edge is a RESULT, not
        an error, mirroring _backtest_command's "a backtest with zero trades is a
        result". 1 on a data or configuration error. 2 on the holdout guard.
    """
    try:
        registry.load_all()
    except FrameworkError as exc:
        # Import errors are fatal by contract §3: a family silently missing from
        # this table would read as "tested and found wanting".
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if coverage:
        text = format_coverage()
        print(text)
        if out_path:
            with open(out_path, "w") as fh:
                fh.write(text + "\n")
        return 0

    now = int(time.time() * 1000) if now_ms is None else now_ms
    boundary = now - config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS * 86_400_000
    if end_ms is None or end_ms > boundary:
        print(
            f"REFUSED: --end {_ms_date(end_ms) if end_ms else '(none)'} is inside the "
            f"reserved holdout. The latest permitted end date is "
            f"{_ms_date(boundary)} (now - "
            f"{config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS} days = "
            f"config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS). Phase 9 owns a span no "
            f"diagnostic may see, so this is a mechanism rather than a promise.",
            file=sys.stderr,
        )
        return 2

    keys = _phase8_detector_keys()
    if detectors:
        unknown = [k for k in detectors if k not in registry.REGISTRY]
        if unknown:
            print(
                f"ERROR: unknown detector key(s) {unknown}; registered Phase 8 "
                f"detectors are {keys}",
                file=sys.stderr,
            )
            return 1
        keys = [k for k in keys if k in set(detectors)]
        extra = [k for k in detectors if k not in keys]
        if extra:
            print(
                f"ERROR: {extra} is registered but is not a Phase 8 detector "
                f"module; this report measures only {PHASE8_DETECTOR_MODULES}",
                file=sys.stderr,
            )
            return 1
    if not keys:
        print("ERROR: no Phase 8 detectors are registered", file=sys.stderr)
        return 1

    state_conn = connect_state(state_db)
    ledger = trials.TrialLedger(state_conn, config.DETECTOR_REPORT_CAMPAIGN)
    before = ledger.count()

    rows = []
    for key in keys:
        spec = registry.get(key)
        graph = _diagnostic_graph(key)
        all_trades = []
        for symbol in symbols:
            all_trades.extend(
                run_graph_backtest(
                    conn, graph, symbol, start_ms=start_ms, end_ms=end_ms
                )
            )
        ledger.record(
            graph_hash=fgraph.graph_hash(graph),
            params_hash=trials.params_hash(spec.defaults()),
            start_ms=start_ms,
            end_ms=end_ms,
        )
        m = compute_metrics(all_trades)
        rows.append(
            {
                "key": key,
                "name": spec.name,
                "tier": spec.tier,
                "metrics": m,
                "costs": _cost_columns(all_trades),
                "sufficient": m["n_trades"] >= config.DETECTOR_MIN_EVENTS_FOR_REPORT,
            }
        )
    after = ledger.count()
    state_conn.close()

    text = _format_detector_report(
        rows,
        symbols=symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        ledger_rows=after - before,
        ledger_total=after,
    )
    print(text)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
    return 0


def _format_detector_report(
    rows, *, symbols, start_ms, end_ms, ledger_rows, ledger_total
) -> str:
    """The banner, the table, and the coverage ledger, as one string."""
    n = len(rows)
    out = [
        DETECTOR_REPORT_BANNER,
        f" span {_ms_date(start_ms)}..{_ms_date(end_ms)} (TUNING span; holdout guard: "
        f"{config.DETECTOR_REPORT_HOLDOUT_GUARD_DAYS}d reserved)",
        f" symbols {','.join(symbols)}",
        f" fixed exit policy, NOTHING SWEPT.  every detector at its ParamSpec "
        f"defaults.",
        f' ledger campaign="{config.DETECTOR_REPORT_CAMPAIGN}" rows=+{ledger_rows} '
        f"(campaign total {ledger_total})",
        f" SELECTION COST IF USED: if any detector is dropped from a campaign "
        f"because of",
        f" this table, add n_trials += {n} to that campaign. Reading it is free; "
        f"letting it",
        f" inform selection is not.",
        f" rows sorted by TIER then NAME -- never by expectancy, because sorting "
        f"by",
        f" expectancy IS selection.",
        "=" * len(DETECTOR_REPORT_BANNER),
        "",
        f"{'detector':<28} {'tier':>4} {'trades':>7} {'win%':>7} {'exp%':>9} "
        f"{'pf':>6} {'maxdd%':>8} {'c':>7}  verdict",
        "-" * 104,
    ]
    for r in rows:
        m, c = r["metrics"], r["costs"]
        tier = "--" if r["tier"] is None else str(r["tier"])
        if not r["sufficient"]:
            # Suppression, not rounding: rates off a handful of trades are the
            # error KNOWN-LIMITATIONS §1 records, one level down.
            out.append(
                f"{r['name']:<28} {tier:>4} {m['n_trades']:>7} {'--':>7} "
                f"{'--':>9} {'--':>6} {'--':>8} {'--':>7}  "
                f"INSUFFICIENT(<{config.DETECTOR_MIN_EVENTS_FOR_REPORT})"
            )
            continue
        exceeds = (
            c["cost_ratio"] is not None
            and c["cost_ratio"] > config.COST_RATIO_CEILING
        )
        verdict = "MEASURED" + (" COST-BOUND" if exceeds else "")
        out.append(
            f"{r['name']:<28} {tier:>4} {m['n_trades']:>7} "
            f"{_fmt(m['win_rate'], '.1%'):>7} "
            f"{_fmt(m['expectancy_pct'], '+.4%'):>9} "
            f"{_fmt(m['profit_factor'], '.2f'):>6} "
            f"{_fmt(m['max_drawdown_pct'], '.2%'):>8} "
            f"{_fmt(c['cost_ratio'], '.3f'):>7}  {verdict}"
        )
    out += [
        "-" * 104,
        f" c = mean round-trip cost / median risk_pct, against "
        f"config.COST_RATIO_CEILING = {config.COST_RATIO_CEILING}. A detector "
        f"whose c exceeds",
        f" the ceiling is STRUCTURALLY unable to pay for itself regardless of "
        f"win rate.",
        f" exp% is NET of costs: run_graph_backtest charged "
        f"2*(FEE_PCT+SLIPPAGE_PCT) + FUNDING_PCT_PER_DAY*days,",
        f" exactly as production charges it. No second P&L path exists in this "
        f"command.",
        "",
        format_coverage(),
    ]
    return "\n".join(out)


def _campaign_command(
    conn,
    state_conn,
    *,
    stage: str = "all",
    campaign_id: str | None = None,
    resume: bool = True,
    force_reason: str | None = None,
) -> int:
    """Run one stage of the pre-registered v0.3.0 campaign (Phase 9).

    Exit codes:
        0 — gate PASS (verdict A or A').
        1 — gate FAIL. **A COMPLETED, VALID OUTCOME** (B1-B4). This command
            reports the GATE, not whether the phase succeeded: a statistically
            honest "no" exits 1 by design, and contract §4 says explicitly that
            if the cumulative trial count makes the gate unpassable, "that is
            the finding".
        2 — ambiguous / aborted (verdict C): HoldoutViolation, a holdout already
            consumed without an override, a missing dependency, no champion, or
            HOLDOUT_LOCKED false. **Only exit 2 means the phase failed to
            produce a verdict.**
    """
    cid = campaign_id or campaign.default_campaign_id()
    try:
        h_start, h_end = campaign.holdout_span()
        e_start, e_end = campaign.evolution_span()
    except campaign.CampaignError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(
        f"campaign {cid}  seed={config.CAMPAIGN_SEED}  "
        f"symbols={len(config.CAMPAIGN_SYMBOLS)} ({','.join(config.CAMPAIGN_SYMBOLS)})"
    )
    print(
        f"  evolution span {_ms(e_start)} -> {_ms(e_end)} "
        f"({(e_end - e_start) // 86_400_000} d, {campaign.fold_count()} folds)"
    )
    print(
        f"  HOLDOUT        {_ms(h_start)} -> {_ms(h_end)} "
        f"({config.HOLDOUT_DAYS} d, end EXCLUSIVE)  locked={config.HOLDOUT_LOCKED}"
    )

    stop_reason = None  # None => campaign.py derives it from state.db
    try:
        if stage in ("probe", "all"):
            probe = campaign.probe_budget(
                conn, state_conn, campaign_id=f"{cid}-probe"
            )
            print(
                f"  probe: {probe.n_probe_evals} real evaluations, "
                f"{probe.seconds_per_eval:.2f} s/eval -> "
                f"{probe.population}x{probe.generations} projects "
                f"{probe.projected_hours:.2f} h vs budget "
                f"{config.CAMPAIGN_WALL_CLOCK_BUDGET_HOURS:.2f} h  "
                f"[{'WITHIN' if probe.within_budget else 'OVER'} BUDGET]"
            )
            if stage == "probe":
                return 0

        if stage in ("evolve", "all"):
            evo = campaign.run_evolution_stage(
                conn, state_conn, campaign_id=campaign_id, resume=resume,
                progress=print,
            )
            stop_reason = evo.stop_reason
            print(
                f"  evolve: campaign {evo.campaign_id} last_generation="
                f"{evo.last_generation} stop_reason={evo.stop_reason} "
                f"wall={evo.wall_seconds:.1f}s trials {evo.trials_before} -> "
                f"{evo.trials_after}"
            )
            if stage == "evolve":
                return 0

        if stage in ("holdout", "all"):
            outcome = campaign.run_holdout_gate(
                conn, state_conn, campaign_id=cid, force_reason=force_reason,
                stop_reason=stop_reason,
            )
            _print_campaign_outcome(outcome)
            if stage == "holdout":
                return 0 if outcome.result.passed else 1

        if stage in ("report", "all"):
            payload = campaign.latest_outcome(state_conn)
            if payload is None:
                print(
                    "ERROR: no completed campaign; run "
                    "`campaign --stage holdout` first"
                )
                return 2
            print(
                f"  report: verdict {payload['verdict']} / "
                f"{payload['benchmark_context']}  run_index="
                f"{payload['run_index']}  n_trials_charged="
                f"{payload['n_trials_charged']}"
            )
            print(
                "  render with: .venv/bin/python scripts/build_campaign_report.py "
                f"--out {config.CAMPAIGN_REPORT_DIR}/phase9-campaign-verdict.md"
            )
            return 0 if payload["passed"] else 1
    except campaign.HoldoutViolation as exc:
        campaign.record_violation(
            state_conn, campaign_id=cid, start_ms=e_start, end_ms=e_end,
            detail=str(exc),
        )
        print(f"ERROR (holdout violation): {exc}")
        return 2
    except campaign.CampaignError as exc:
        print(f"ERROR: {exc}")
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


def _ms(ms: int) -> str:
    """Epoch ms -> 'YYYY-MM-DD (ms)', so every printed span carries both forms."""
    return f"{pd.Timestamp(ms, unit='ms', tz='UTC').date()} ({ms})"


def _print_campaign_outcome(outcome) -> None:
    """The verdict, per condition, with the benchmark's ABSOLUTE numbers beside
    the two beats_benchmark_* bits — never the bit alone (contract §4 point 3)."""
    r = outcome.result
    eq, m = r.oos_equity, r.oos_metrics
    basket = outcome.benchmark.basket
    ch = outcome.champion
    print(
        f"  champion member={ch.member_id} graph={ch.graph_hash[:12]} "
        f"gen={ch.generation} tier={ch.tier} fitness={ch.fitness:.4f} "
        f"n_trades(window)={ch.n_trades}"
    )
    print(
        f"    eligible-by-trade-floor: {ch.n_eligible} of {ch.n_scored} scored "
        f"members cleared CAMPAIGN_MIN_CHAMPION_TRADES="
        f"{config.CAMPAIGN_MIN_CHAMPION_TRADES}"
        + ("  *** CHAMPION_BELOW_TRADE_FLOOR ***" if ch.below_trade_floor else "")
    )
    if outcome.run_index > 1:
        print(f"  *** NOT A CLEAN HOLDOUT — run_index={outcome.run_index} ***")
    print(f"  n_trials charged: {outcome.n_trials_charged}")
    print(f"GATE ({len(GATE_CONDITIONS)} conditions):")
    rows = [
        ("sample_adequacy", f"{m['n_trades']} trades",
         f">= {config.WF_MIN_TRADES}"),
        ("sharpe", _fmt(eq["sharpe"], ".4f"),
         f">= {walkforward.GATE_MIN_SHARPE}"),
        ("dsr", _fmt(eq["dsr"], ".6f"), f"> {walkforward.GATE_MIN_DSR}"),
        ("max_drawdown", _fmt(eq["max_drawdown_pct"], ".4%"),
         f"<= {walkforward.GATE_MAX_DRAWDOWN:.0%}"),
        ("per_symbol_expectancy",
         f"{sum(1 for v in r.per_symbol_expectancy.values() if v is not None and v > 0)}"
         f" of {len(r.per_symbol_expectancy)} positive", "all > 0"),
        ("beats_benchmark_return",
         f"{_fmt(eq['ann_return_pct'], '.2%')} vs basket "
         f"{_fmt(basket['ann_return_pct'], '.2%')}", "> basket"),
        ("beats_benchmark_sharpe",
         f"{_fmt(eq['sharpe'], '.4f')} vs basket {_fmt(basket['sharpe'], '.4f')}",
         "> basket"),
    ]
    for name, value, threshold in rows:
        ok = outcome.gate[name]
        print(f"  {name:<24} {value:<38} {threshold:<14} "
              f"{'PASS' if ok else 'FAIL'}")
    print(
        f"  buy-and-hold basket ABSOLUTE over the SAME holdout: "
        f"total={_fmt(basket['total_return'], '.4f')}x  "
        f"ann={_fmt(basket['ann_return_pct'], '.2%')}  "
        f"sharpe={_fmt(basket['sharpe'], '.4f')}  "
        f"maxDD={_fmt(basket['max_drawdown_pct'], '.2%')}  "
        f"n_days={basket['n_days']}"
    )
    print(f"  benchmark context: {outcome.benchmark_context}")
    print("per-symbol OOS expectancy:")
    for symbol, exp in r.per_symbol_expectancy.items():
        flag = "OK" if (exp is not None and exp > 0) else "FAIL"
        print(f"  {symbol}: {_fmt(exp, '.4%')}  [{flag}]")
    ann, met = campaign.northstar_gap(eq)
    print(
        f"northstar (REPORTED, NOT GATED): ann_return={_fmt(ann, '.2%')} vs "
        f"target >{config.CAMPAIGN_NORTHSTAR_ANN_RETURN:.0%} -> "
        f"{'MET' if met else 'NOT MET'}"
    )
    print(f"GATE: {'PASS' if r.passed else 'FAIL'}")
    print(f"VERDICT: {outcome.verdict} / {outcome.benchmark_context}")


if __name__ == "__main__":
    main()
