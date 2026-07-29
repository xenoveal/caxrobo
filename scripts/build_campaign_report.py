"""
Render the v0.3.0 Phase 9 campaign verdict as Markdown, from persisted state.

READS state.db AND NOTHING ELSE. It never re-runs the gate: a report generator
that re-runs the gate is a second peek at the holdout wearing a reporting hat.
Every figure it prints comes out of the outcome_json blob that
campaign.run_holdout_gate wrote inside the same transaction as the holdout
consumption row, and every figure appears in the provenance table with the
command and span that produced it (contract §12.5).

Companion to build_performance_chart.py, whose shape this mirrors
(collect() -> _scorecard() -> build() -> main() with --out required). Markdown,
not HTML: the verdict is a table of measured numbers with provenance, and it
does not need SVG to be honest.

Usage:
    .venv/bin/python scripts/build_campaign_report.py \
        --out .claude/PRPs/reports/phase9-campaign-verdict.md
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")

from trading_bot import campaign, config  # noqa: E402
from trading_bot.backtest import walkforward  # noqa: E402
from trading_bot.data import statestore  # noqa: E402

logger = logging.getLogger("build_campaign_report")

# The taxonomy's meaning + recommended action, copied verbatim from the Phase 9
# plan. The classifier in campaign.py owns the label; this table only renders it.
VERDICT_TEXT = {
    "A": (
        "Northstar met. Hypothesis survived.",
        "Freeze the champion graph and version; do NOT tune further; move to a "
        "forward paper period before any capital.",
    ),
    "A_PRIME": (
        "Gate passed, northstar missed.",
        "Completed outcome. Report the gap honestly and decide separately "
        "whether a ~20-30% honest edge is worth sizing.",
    ),
    "B1": (
        "The economically interesting \"no\": beat the basket out of sample but "
        "cannot be proven significant under honest multiple-testing correction "
        "at this sample size.",
        "Report both DSR figures and the basket's absolute numbers. Extend the "
        "holdout in CALENDAR TIME by waiting for real forward data — never by "
        "re-running with a lower n_trials.",
    ),
    "B2": (
        "Honest \"no\" with a diagnosed second weakness.",
        "Action follows the OTHER failure: per-symbol failure => the edge is "
        "concentrated, revisit Phase 8 detector selection; drawdown failure => "
        "the equity path is unacceptable regardless of mean.",
    ),
    "B3": (
        "Same verdict as v0.2.0: LOSES TO INACTION.",
        "Abandon this strategy family. The framework and workflow keep their "
        "value (PRD honesty clause). Do not tune.",
    ),
    "B4": (
        "Data problem, unchanged from v0.2.0 §4.",
        "Record the measured trade rate. NEVER lower WF_MIN_TRADES and NEVER "
        "shrink the holdout. The honest responses are more genuinely "
        "uncorrelated symbols, a higher trade rate, or more calendar time.",
    ),
    "C": (
        "AMBIGUOUS — the run did not produce a reproducible verdict. This is the "
        "only outcome that counts as Phase 9 failing.",
        "Diagnose which ambiguity fired (see the plan's prevention table) and "
        "re-run the protocol, not the numbers.",
    ),
}


def _iso(ms) -> str:
    if ms is None:
        return "--"
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _d(ms) -> str:
    if ms is None:
        return "--"
    return f"{_iso(ms)[:10]} ({int(ms)})"


def _n(v, spec=".4f") -> str:
    """Format a possibly-None figure. Mirrors cli._fmt: None reads as '--'."""
    if v is None:
        return "--"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def collect(state_conn) -> dict | None:
    """The persisted campaign outcome, or None when no holdout has been run.

    One field is re-derived rather than trusted: `stop_reason`. The stages run as
    separate processes, so a `--stage holdout` invoked on its own persists
    "NOT_RUN"/"UNKNOWN" for a stage that did in fact complete earlier. Reading it
    back out of the `campaigns` / `generations` rows is honest (it is persisted
    state, not a re-run) and avoids a report that understates what happened.
    """
    payload = campaign.latest_outcome(state_conn)
    if payload is None:
        return None
    if str(payload.get("stop_reason", "")).split()[0] in ("NOT_RUN", "UNKNOWN", ""):
        payload["stop_reason"] = (
            f"{campaign.evolution_stop_reason(state_conn)} "
            f"[re-derived from state.db; the holdout stage ran as a separate "
            f"process and persisted {payload.get('stop_reason')!r}]"
        )
    return payload


def _scorecard(data) -> list[tuple[str, str, str, bool]]:
    """The SEVEN gate conditions as (label, value, threshold, ok) rows.

    `ok` is taken from the persisted gate dict, never recomputed: two
    implementations of a threshold is how a report ends up disagreeing with the
    gate it reports on. Every threshold is read from walkforward.GATE_MIN_* or
    config, never retyped as a literal.
    """
    gate, eq, m = data["gate"], data["oos_equity"], data["oos_metrics"]
    per_sym = data["per_symbol_expectancy"]
    basket = data["benchmark"]["basket"]
    n_pos = sum(1 for v in per_sym.values() if v is not None and v > 0)
    return [
        ("sample_adequacy", f"{m['n_trades']} trades",
         f">= {config.WF_MIN_TRADES}", gate["sample_adequacy"]),
        ("sharpe (annualised)", _n(eq["sharpe"]),
         f">= {walkforward.GATE_MIN_SHARPE}", gate["sharpe"]),
        ("dsr", _n(eq["dsr"], ".6f"), f"> {walkforward.GATE_MIN_DSR}",
         gate["dsr"]),
        ("max_drawdown", _n(eq["max_drawdown_pct"], ".2%"),
         f"<= {walkforward.GATE_MAX_DRAWDOWN:.0%}", gate["max_drawdown"]),
        ("per_symbol_expectancy", f"{n_pos} of {len(per_sym)} positive",
         "ALL > 0 (AND, not average)", gate["per_symbol_expectancy"]),
        ("beats_benchmark_return",
         f"{_n(eq['ann_return_pct'], '.2%')} vs basket "
         f"{_n(basket['ann_return_pct'], '.2%')}",
         "> basket ann_return_pct", gate["beats_benchmark_return"]),
        ("beats_benchmark_sharpe",
         f"{_n(eq['sharpe'])} vs basket {_n(basket['sharpe'])}",
         "> basket sharpe", gate["beats_benchmark_sharpe"]),
    ]


def build(data: dict) -> str:
    out: list[str] = []
    add = out.append

    ch = data["champion"]
    sp = data["spans"]
    eq, m = data["oos_equity"], data["oos_metrics"]
    diag = data["diagnostics"]
    proj = data["projection"]
    basket = data["benchmark"]["basket"]
    verdict, context = data["verdict"], data["benchmark_context"]
    meaning, action = VERDICT_TEXT[verdict]

    add("# Phase 9 — the v0.3.0 walk-forward campaign verdict")
    add("")
    add(
        "GENERATED by `scripts/build_campaign_report.py` from `data/state.db`. "
        "This generator re-runs nothing; every figure below was measured by the "
        "one-shot holdout run recorded in `holdout_consumption`."
    )
    add("")
    if data["run_index"] > 1:
        add("> ## *** NOT A CLEAN HOLDOUT ***")
        add(">")
        add(
            f"> This is holdout run_index **{data['run_index']}**. Override "
            f"reason, verbatim: `{data.get('override_reason')}`"
        )
        add("")

    # ---- 1. Header -------------------------------------------------------
    add("## 1. Header — what was pre-registered")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| campaign id | `{data['campaign_id']}` |")
    add(f"| seed | {data['seed']} |")
    add(f"| symbols ({len(data['symbols'])}) | {', '.join(data['symbols'])} |")
    add(
        f"| evolution span | {_d(sp['evolve_start_ms'])} -> "
        f"{_d(sp['holdout_start_ms'])} "
        f"({(sp['holdout_start_ms'] - sp['evolve_start_ms']) // 86_400_000} d, "
        f"{data['n_folds']} folds) |"
    )
    add(
        f"| **HOLDOUT (end EXCLUSIVE)** | **{_d(sp['holdout_start_ms'])} -> "
        f"{_d(sp['holdout_end_ms'])} ({sp['holdout_days']} d)** |"
    )
    add(f"| harness-reported OOS window | {_d(sp['oos_start'])} -> {_d(sp['oos_end'])} |")
    add(f"| holdout run_index | {data['run_index']} |")
    add(f"| evolution stop reason | {data['stop_reason']} |")
    add(f"| champion member | `{ch['member_id']}` (campaign `{ch['campaign_id']}`) |")
    add(f"| champion graph hash | `{ch['graph_hash']}` |")
    add(
        f"| champion generation / tier / fitness | {ch['generation']} / "
        f"{ch['tier']} / {ch['fitness']:.4f} |"
    )
    add(
        f"| champion trades on its selection window | {ch['n_trades']} |"
    )
    add(
        f"| trade-floor eligibility | {ch['n_eligible']} of {ch['n_scored']} "
        f"scored members cleared `CAMPAIGN_MIN_CHAMPION_TRADES` = "
        f"{data['thresholds']['CAMPAIGN_MIN_CHAMPION_TRADES']}"
        + ("  **CHAMPION_BELOW_TRADE_FLOOR**" if ch["below_trade_floor"] else "")
        + " |"
    )
    add(f"| holdout opened / completed | {_iso(data['opened_ts'])} / {_iso(data['completed_ts'])} |")
    add("")
    add(
        "The holdout start is Phase 6's frozen `EVO_TRAIN_END`. The holdout end "
        "is the exclusive close of the last **complete** daily bar, not "
        "`MAX(ts)` — see `config.py`'s Phase 9 block for the measurement."
    )
    add("")

    # ---- 2. The gate, per condition -------------------------------------
    add(f"## 2. THE GATE — all {len(walkforward.GATE_CONDITIONS)} conditions")
    add("")
    add("| Condition | Measured | Threshold | Verdict |")
    add("|---|---|---|---|")
    for label, value, threshold, ok in _scorecard(data):
        add(f"| `{label}` | {value} | {threshold} | {'PASS' if ok else '**FAIL**'} |")
    add("")
    add(f"**GATE: {'PASS' if data['passed'] else 'FAIL'}** "
        f"(`passed == all(gate.values())`)")
    add("")
    add("Per-symbol OOS expectancy (the AND, spelled out):")
    add("")
    add("| Symbol | OOS expectancy | |")
    add("|---|---|---|")
    for symbol, exp in data["per_symbol_expectancy"].items():
        flag = "OK" if (exp is not None and exp > 0) else "FAIL"
        add(f"| {symbol} | {_n(exp, '.4%')} | {flag} |")
    add("")

    # ---- 3. Buy-and-hold beside it, in absolute terms -------------------
    add("## 3. The buy-and-hold null, in ABSOLUTE terms")
    add("")
    add(
        f"**benchmark context: `{context}`.** The two `beats_benchmark_*` bits "
        "above may never be read without these numbers next to them: Phase 1 "
        "measured the basket's Sharpe over v0.2.0's 90-day OOS window at "
        "**-1.303**, i.e. both conditions can PASS against a basket that LOST "
        "MONEY, which is not the protection KNOWN-LIMITATIONS §0 asked for."
    )
    add("")
    add(
        "Every annualised figure in this section is an **EXTRAPOLATION** from a "
        f"{sp['holdout_days']}-day window, not a CAGR. (§0's +29.4% basket "
        "column IS a genuine 3-year CAGR — 1095 days — and the two must never "
        "share a column.)"
    )
    add("")
    add("| Series | total return | annualised (extrapolated) | Sharpe | max DD | n_days |")
    add("|---|---|---|---|---|---|")
    add(
        f"| **equal-weight basket** | {_n(basket['total_return'])}x | "
        f"{_n(basket['ann_return_pct'], '.2%')} | {_n(basket['sharpe'])} | "
        f"{_n(basket['max_drawdown_pct'], '.2%')} | {basket['n_days']} |"
    )
    for symbol, b in data["benchmark"]["per_symbol"].items():
        add(
            f"| {symbol} | {_n(b['total_return'])}x | "
            f"{_n(b['ann_return_pct'], '.2%')} | {_n(b['sharpe'])} | "
            f"{_n(b['max_drawdown_pct'], '.2%')} | {b['n_days']} |"
        )
    add(
        f"| **the champion (strategy)** | -- | {_n(eq['ann_return_pct'], '.2%')} "
        f"| {_n(eq['sharpe'])} | {_n(eq['max_drawdown_pct'], '.2%')} | "
        f"{eq['n_days']} |"
    )
    add("")
    if context == "BENCHMARK_NEGATIVE":
        add(
            "> Because the context is `BENCHMARK_NEGATIVE`, a "
            "`beats_benchmark_*` PASS here means **\"beat a basket that lost "
            "money\"** and is NOT reported as northstar evidence. Under `B3` "
            "the same context makes the result WORSE, not merely weaker: "
            "failing to beat a basket that itself lost money is a strong "
            "negative signal."
        )
        add("")

    # ---- 4. Trial accounting --------------------------------------------
    add("## 4. Trial accounting — what the DSR was charged")
    add("")
    add("| Figure | Value |")
    add("|---|---|")
    add(f"| **n_trials CHARGED** | **{data['n_trials_charged']}** |")
    add(
        f"| walk-forward's in-process default (NOT used) | "
        f"{diag['n_trials_walkforward_default']} |"
    )
    add(f"| DSR at the charged count | {_n(diag['dsr_charged'], '.6f')} |")
    add(f"| DSR at `n_trials=1` (correction OFF) | {_n(diag['dsr_at_n_trials_1'], '.6f')} |")
    add(
        f"| DSR at grid x folds ({diag['n_trials_walkforward_default']}) | "
        f"{_n(diag['dsr_at_grid_x_folds'], '.6f')} |"
    )
    add("")
    recorded = sum(r["n"] for r in data["ledger_breakdown"] if not r["past_barrier"])
    add(
        f"The charge decomposes as **{recorded} recorded pre-barrier ledger "
        f"rows + {data['n_trials_charged'] - recorded} evaluations performed by "
        f"this gate run's own fold sweep** "
        f"(`2 x combos x folds` on a degenerate grid = 2 x 1 x "
        f"{data['n_folds']}). The rule is mechanical rather than a judgement "
        f"call: count every ledger row whose `end_ms` lies at or before the "
        f"holdout barrier, so no decision is made after the fact about which of "
        f"our own evaluations \"really\" counted. It deliberately OVER-counts "
        f"(diagnostic re-scorings and other phases' report campaigns are "
        f"included) because over-counting errs pessimistically."
    )
    add("")
    add("Per-campaign ledger rows in `data/state.db`:")
    add("")
    add("| Ledger campaign | rows | max end_ms | past barrier? |")
    add("|---|---|---|---|")
    for row in data["ledger_breakdown"]:
        add(
            f"| `{row['campaign']}` | {row['n']} | {_d(row['max_end_ms'])} | "
            f"{'**YES**' if row['past_barrier'] else 'no'} |"
        )
    add("")
    add("Annualised Sharpe that `dsr > "
        f"{walkforward.GATE_MIN_DSR}` demands at each candidate trial count, on "
        f"{eq['n_days']} daily observations at this run's MEASURED moments "
        f"(skew {_n(diag['holdout_daily_skew'])}, kurtosis "
        f"{_n(diag['holdout_daily_kurtosis'])}):")
    add("")
    add("| n_trials | required annualised Sharpe | observed |")
    add("|---|---|---|")
    for nt in (1, diag["n_trials_walkforward_default"], 198,
               data["n_trials_charged"]):
        req = campaign.required_annual_sharpe(
            nt, eq["n_days"] or config.HOLDOUT_DAYS,
            diag["holdout_daily_skew"], diag["holdout_daily_kurtosis"],
        )
        add(f"| {nt} | {req:.2f} | {_n(eq['sharpe'])} |")
    add("")

    # ---- 5. Northstar ---------------------------------------------------
    add("## 5. The northstar — REPORTED, NOT GATED")
    add("")
    add(
        "Contract §4's seven `GATE_CONDITIONS` contain **no return threshold**, "
        "so a passing gate does not imply the northstar and a failing gate does "
        "not by itself refute it. They are reported separately, on purpose."
    )
    add("")
    add("| Figure | Value |")
    add("|---|---|")
    add(f"| holdout annualised return (extrapolated) | {_n(eq['ann_return_pct'], '.2%')} |")
    add(
        f"| target | > "
        f"{data['thresholds']['CAMPAIGN_NORTHSTAR_ANN_RETURN']:.0%} |"
    )
    add(f"| gap | {_n(diag['northstar_ann_return_gap'], '.2%')} |")
    met = (eq["ann_return_pct"] is not None
           and eq["ann_return_pct"] > data["thresholds"]["CAMPAIGN_NORTHSTAR_ANN_RETURN"])
    add(f"| **northstar** | **{'MET' if met else 'NOT MET'}** |")
    add("")

    # ---- 6. Sample adequacy --------------------------------------------
    add("## 6. Sample adequacy — raw AND independent-equivalent")
    add("")
    add("| Figure | Value |")
    add("|---|---|")
    add(f"| symbols | {proj['n_symbols']} |")
    add(
        f"| trade rate MEASURED on the evolution span only | "
        f"{proj['measured_trades_per_symbol_day']:.4f} trades/symbol/day "
        f"({proj['measured_trades_total']} trades over "
        f"{proj['evolution_days']} d) |"
    )
    add(f"| projected raw holdout trades | {proj['projected_raw_trades']:.1f} |")
    add(f"| REALIZED raw holdout trades | {m['n_trades']} |")
    add(f"| realized holdout rate | {diag['realized_holdout_rate']:.4f} trades/symbol/day |")
    add(f"| mean pairwise r (measured) | {proj['mean_pairwise_correlation']:.4f} |")
    add(f"| **Kish effective N (measured)** | **{proj['effective_n']:.4f}** |")
    add(
        f"| independent-equivalent trades (= per-symbol trades x effN) | "
        f"{(m['n_trades'] / proj['n_symbols']) * proj['effective_n']:.1f} |"
    )
    add(
        f"| break-even rate for {config.WF_MIN_TRADES} independent-equivalent | "
        f"{proj['required_rate_for_independent_floor']:.4f} trades/symbol/day |"
    )
    add(f"| raw floor (`WF_MIN_TRADES`, unmodified) | {proj['raw_floor']} |")
    add("")
    add(
        "**The correlation trap is MITIGATED, NOT SOLVED.** An effective N of "
        f"{proj['effective_n']:.3f} on {proj['n_symbols']} symbols means the "
        "pooled sample carries roughly the information of **fewer than two "
        "independent instruments**; every stored symbol is a liquid major and "
        "all are BTC beta. DSR's near-iid assumption is still violated, just "
        "less severely. The raw count is **never** evidence of independence."
    )
    add("")
    add(
        "`WF_MIN_TRADES` was **not lowered** and the holdout window was **not "
        "shrunk** (KNOWN-LIMITATIONS §4 forbids both by name). The "
        "independent-equivalent figure is a reported caveat, never a gate "
        "condition."
    )
    add("")

    # ---- 7. What the gate does not see ---------------------------------
    add("## 7. What the gate does NOT see")
    add("")
    add(f"> {diag['fold_evidence_note']}")
    add("")
    add("| Diagnostic | Value |")
    add("|---|---|")
    add(
        f"| full-span max drawdown ({_d(diag['full_span_start_ms'])} -> "
        f"{_d(diag['full_span_end_ms'])}) | "
        f"{_n(diag['full_span_max_drawdown_pct'], '.2%')} |"
    )
    add(f"| holdout-window max drawdown (what the gate saw) | {_n(diag['holdout_max_drawdown_pct'], '.2%')} |")
    add(f"| full-span trades | {diag['full_span_n_trades']} |")
    add(
        f"| fold test expectancy: negative folds | "
        f"{diag['fold_n_negative']} of {diag['fold_n_total']} |"
    )
    add(f"| folds that fell back to config defaults | {diag['folds_fell_back_to_defaults']} of {diag['fold_n_total']} |")
    add(f"| holdout daily observations | {diag['holdout_daily_n_obs']} |")
    add(
        f"| holdout daily skew / kurtosis | "
        f"{_n(diag['holdout_daily_skew'])} / {_n(diag['holdout_daily_kurtosis'])} "
        f"(Phase 1 measured {diag['phase1_measured_skew_after_fix']} / "
        f"{diag['phase1_measured_kurtosis_after_fix']} on v0.2.0's series after "
        f"the MEDIUM-5 fix) |"
    )
    add(f"| P&L attribution mode | `{diag['pnl_attribution_mode']}` |")
    add(f"| round-trip cost charged per trade | {diag['mean_cost_pct_per_trade_round_trip']:.4%} |")
    add("")
    add("Per-symbol equity multiple over the FULL span (evolution + holdout):")
    add("")
    add("| Symbol | equity multiple |")
    add("|---|---|")
    for symbol, mult in diag["per_symbol_end_state"].items():
        add(f"| {symbol} | {_n(mult, '.4f')}x |")
    add("")
    add("Per-fold test expectancy (DIAGNOSIS, not evidence):")
    add("")
    add("| Fold | test expectancy |")
    add("|---|---|")
    for i, exp in enumerate(diag["fold_test_expectancy"]):
        add(f"| {i} | {_n(exp, '.4%')} |")
    add("")
    add(f"> {diag['unsized_drawdown_note']}")
    add("")

    # ---- 8. Verdict -----------------------------------------------------
    add("## 8. VERDICT")
    add("")
    add(f"# `{verdict}` / `{context}`")
    add("")
    add(f"**Meaning.** {meaning}")
    add("")
    add(f"**Pre-committed next action.** {action}")
    add("")

    # ---- 9. Provenance --------------------------------------------------
    add("## 9. Provenance — every figure with its command and span")
    add("")
    add("| Figure | Value | Command | Span |")
    add("|---|---|---|---|")
    holdout_span = f"{_d(sp['holdout_start_ms'])} -> {_d(sp['holdout_end_ms'])}"
    evo_span = f"{_d(sp['evolve_start_ms'])} -> {_d(sp['holdout_start_ms'])}"
    full_span = f"{_d(diag['full_span_start_ms'])} -> {_d(diag['full_span_end_ms'])}"
    rows = [
        ("all 7 gate conditions", "see §2",
         "`cli campaign --stage holdout`", holdout_span),
        ("OOS trades", str(m["n_trades"]),
         "`cli campaign --stage holdout` -> `walk_forward_pooled.oos_metrics`",
         holdout_span),
        ("OOS Sharpe / DSR / annualised / max DD",
         f"{_n(eq['sharpe'])} / {_n(eq['dsr'], '.6f')} / "
         f"{_n(eq['ann_return_pct'], '.2%')} / {_n(eq['max_drawdown_pct'], '.2%')}",
         "`walk_forward_pooled.oos_equity` (`compute_equity_metrics`, "
         f"attribution=`{diag['pnl_attribution_mode']}`)", holdout_span),
        ("buy-and-hold basket + per symbol", "see §3",
         "`backtest.benchmark.buy_and_hold` via `walk_forward_pooled`",
         holdout_span),
        ("n_trials charged", str(data["n_trials_charged"]),
         "`campaign.cumulative_trials` (SQL over `trial_ledger`) + "
         "`campaign.predicted_gate_trials`",
         f"every ledger row with end_ms <= {sp['holdout_start_ms']}"),
        ("DSR at n_trials=1 and at grid x folds",
         f"{_n(diag['dsr_at_n_trials_1'], '.6f')} / "
         f"{_n(diag['dsr_at_grid_x_folds'], '.6f')}",
         "`equity.deflated_sharpe` on the SAME daily series", holdout_span),
        ("required annualised Sharpe table", "see §4",
         "`campaign.required_annual_sharpe` (bisection over "
         "`equity.deflated_sharpe`; no price data)", "n/a"),
        ("champion identity + tie-break", f"`{ch['member_id']}`",
         "`campaign.select_champion` (SQL over `population_members`)",
         evo_span),
        ("trade rate + projection",
         f"{proj['measured_trades_per_symbol_day']:.4f}/symbol/day",
         "`campaign.sample_adequacy_projection` -> `run_graph_backtest`",
         evo_span),
        ("mean pairwise r + Kish effN",
         f"{proj['mean_pairwise_correlation']:.4f} / {proj['effective_n']:.4f}",
         "`data.correlation.effective_n(correlation_matrix(daily_return_frame))`",
         f"{config.CORRELATION_START} -> {_d(sp['holdout_start_ms'] - 1)}"),
        ("full-span max drawdown",
         _n(diag["full_span_max_drawdown_pct"], ".2%"),
         "`campaign.collect_diagnostics` -> `run_graph_backtest` + "
         "`equity.max_drawdown`", full_span),
        ("per-symbol full-span equity multiple", "see §7",
         "`campaign.collect_diagnostics`", full_span),
        ("fold test expectancy / fallbacks", "see §7",
         "`walk_forward_pooled.folds` (DIAGNOSIS, not evidence)", evo_span),
        ("holdout skew / kurtosis",
         f"{_n(diag['holdout_daily_skew'])} / "
         f"{_n(diag['holdout_daily_kurtosis'])}",
         "`walk_forward_pooled.oos_equity` (`equity._skew_kurt`)", holdout_span),
        ("verdict id + benchmark context", f"{verdict} / {context}",
         "`campaign.classify_verdict` + `campaign.benchmark_context`",
         holdout_span),
    ]
    for figure, value, command, span in rows:
        add(f"| {figure} | {value} | {command} | {span} |")
    add("")
    add(
        "No figure in this document was typed by hand and none is derived from "
        "another (contract §12.5; `git log`: \"Use measured rather than derived "
        "figures in the benchmark table\")."
    )
    add("")
    return "\n".join(out) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output Markdown path")
    ap.add_argument("--state-db", default=None,
                    help=f"default config.STATE_DB_PATH = {config.STATE_DB_PATH}")
    args = ap.parse_args()

    state_conn = statestore.connect(args.state_db)
    data = collect(state_conn)
    state_conn.close()
    if data is None:
        logger.error(
            "no completed campaign; run `cli campaign --stage holdout` first"
        )
        return 1

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(build(data))
    logger.info(
        "wrote %s — verdict %s / %s, gate %s, n_trials %d",
        args.out, data["verdict"], data["benchmark_context"],
        "PASS" if data["passed"] else "FAIL", data["n_trials_charged"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
