# Plan: Fade Re-qualification (PRD Phase 6)

## Summary
Give the ranging-regime Bollinger fade sleeve a fair, one-time test under the new risk model (Phase 2) and the new tiers (Phase 4), then **keep it or drop it on written evidence**. This phase deliberately writes almost no strategy code — Phase 2 already swapped the fade's filter to `RR_FLOOR` and Phase 4 already moved it to 4H setup / 1H trigger. What it adds is (a) a `FADE_ENABLED` kill switch so "drop" is a reversible one-line decision instead of a code deletion, (b) a measurement script that produces the candidate funnel and per-symbol metrics the decision needs, and (c) a dated decision report with the numbers behind it.

## User Story
As the bot's sole user, I want the ranging-regime sleeve judged on measured evidence under the corrected risk model rather than carried forward on sentiment or deleted on suspicion, so that Phase 7's gate runs against a signal set I can defend, and so "ranging produces no signals" is an outcome I chose with numbers rather than an accident.

## Problem → Solution
**Current**: the fade method lost money on all 3 symbols (PF 0.49 / 0.76 / 0.86 at n=203/163/155 — `market-research-capability-benchmark.md:110`) but it lost money *under the same broken risk model that sank the structurally unrelated breakout method*, and it was the least-bad of the two (win rate 20–24% vs 7–11%; best PFs) precisely because its stop already sits at the excursion extreme rather than a hairline buffer. It has never been measured under a correct risk model. There is no mechanism to disable it short of editing the dispatcher, and no written record of what its numbers were when the decision was made.
**Solution**: measure it once, on the tuning span only, with every parameter frozen; compare against a decision rule **pre-committed in this plan before any number is seen**; record the verdict and the numbers in `.claude/PRPs/reports/fade-requalification.md`; flip `config.FADE_ENABLED` accordingly. Because nothing is swept, the measurement consumes zero degrees of freedom and the OOS holdout stays untouched.

## Metadata
- **Complexity**: Medium — the diff is small (1 config constant, 2 two-line dispatch guards, 1 script, 1 test class) but the analysis is the deliverable and the sequencing discipline is strict
- **Source PRD**: `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`
- **PRD Phase**: Phase 6 — Fade re-qualification
- **Estimated Files**: 8 (3 new, 5 modified)
- **Depends on**: Phase 2 (`phase2-honest-cost-and-risk-model.plan.md` — `RR_FLOOR`, frozen costs, funding term) and Phase 4 (`phase4-tier-shift-1d-4h-1h.plan.md` — 4H setup / 1H trigger, `MAX_HOLD_BARS_TRIGGER`, the `interval_ms` fix). **Both must be landed and green before Task 3 runs**, or the measurement describes a state that will never ship.
- **Parallel with**: Phase 5 (Donchian). Different regime, different module, one dispatch table shared — see Task 2's GOTCHA.
- **Blocks**: Phase 7. The gate needs the ranging sleeve settled (`PRD:265`).

---

## UX Design

N/A — internal change. No user-facing surface exists until Phase 9. The only observable outputs are the CLI's existing `backtest` bucket breakdown (a `ranging/bollinger-fade` bucket that either exists or does not), a new script's stdout, and the decision report.

### Interaction Changes
| Touchpoint | Before | After | Notes |
|---|---|---|---|
| `cli.py backtest` bucket list | always contains `ranging/bollinger-fade` when ranging bars exist | contains it only when `FADE_ENABLED` is true | Absence is now a *decision*, legible in config |
| `scan_symbol` in a ranging regime | returns fade signals | returns `[]` when `FADE_ENABLED` is false | Regime label still returned unchanged — suppression, not misclassification |
| New | — | `python scripts/fade_requalification.py` | Prints the funnel + per-symbol metrics the decision is made on |

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/trading_bot/signals/meanrev.py` | 1-153 | `detect_fade_setups` — the setup logic being judged. Note the stretch window (104), the per-direction stretch scan (112-141), and that `stop_level` is `lows[first:].min()` over bars **from the first stretch bar to the latest bar**, not just the stretch bar |
| P0 | `src/trading_bot/signals/meanrev.py` | 173-241 | `build_fade_signal` — **Phase 2 rewrites this** to `rr >= rr_floor` and keeps the stop at `candidate.stop_level`. Read Phase 2's Task 5 for its post-Phase-2 body; do not re-derive it |
| P0 | `.claude/PRPs/plans/v0.2.0/phase2-honest-cost-and-risk-model.plan.md` | Task 5 (~407-473), Files to Change (~150-165) | Establishes that fade's stop stays at the excursion extreme and only its *filter* changed. This phase must not move the stop |
| P0 | `.claude/PRPs/plans/v0.2.0/phase4-tier-shift-1d-4h-1h.plan.md` | §C5 (~143-151), §D `FADE_STRETCH_MAX_AGE_BARS` row (~182), NOT Building (~307) | Phase 4 renames `df_1h`/`df_15m` → `df_setup`/`df_trig`, fixes the missing `interval_ms` at `meanrev.py:266`, and explicitly defers any fade retune to this phase |
| P0 | `src/trading_bot/signals/scan.py` | 44-54 | The live dispatch point the kill switch hooks into (`regime_label == "ranging"` at 52) |
| P0 | `src/trading_bot/backtest/engine.py` | 154-171 | `candidates_for` — the backtest dispatch point (`elif reg == "ranging"` at 166). The kill switch must land in **both** paths or backtest and live drift, which is the one thing this engine's docstring (1-28) promises never happens |
| P1 | `src/trading_bot/backtest/metrics.py` | 15-35 | `by_bucket` keyed `"{regime}/{pattern}"` gives fade attribution for free — `"ranging/bollinger-fade"`. No new metrics plumbing needed for the decision |
| P1 | `src/trading_bot/backtest/engine.py` | 51-58, 60-81 | `BacktestParams` carries `bb_num_std`; `Trade` carries `regime`/`pattern`/`entry`/`stop`/`target`/`pnl_pct` — everything the funnel report needs except ATR, which the script computes itself |
| P1 | `src/trading_bot/indicators/bollinger.py` | 19-52 | `bollinger(df, period=, num_std=)` → DataFrame of `middle`/`upper`/`lower`, NaN for the first `period-1` bars. Population std (`ddof=0`) — the classical definition; do not "fix" it to sample std |
| P1 | `src/trading_bot/backtest/walkforward.py` | 36-40, 126-136 | `bb_num_std` is in `DEFAULT_GRID` **today** — meaning `BB_STD` has already been swept on BTCUSDT 2023–2026 and has consumed a degree of freedom. Also the source of the tuning/OOS split arithmetic (`tune_end = end_ms - oos_ms`) the script must reuse verbatim |
| P1 | `tests/test_meanrev.py` | 1-52, 140-175, 232-259 | Test idioms: `make_df`/`alternating_rows`/`long_stretch_df` builders, `make_fade`/`make_event` factories, and `TestScanSymbol`'s monkeypatch-the-dispatcher pattern that the kill-switch tests mirror exactly |
| P1 | `src/trading_bot/indicators/wilder.py` | 95-109 | `atr(df, period)` — needed by the script to express median fade stop distance in ATR multiples and to compute the cost ratio `c` |
| P2 | `.claude/PRPs/reports/market-research-capability-benchmark.md` | 95-120, 250 | The baseline numbers this phase is measured against, and the explicit "re-test it under R1/R2 before judging it" recommendation |
| P2 | `scripts/export_bar_annotations.py` | 1-30 | The house style for an analysis script: module docstring with a `Usage:` block, `argparse`, and the rule that it re-calls production builders rather than reimplementing screening logic |
| P2 | `src/trading_bot/cli.py` | 355-385 | `_fmt` / `_print_metrics` — the output formatting the script should reuse in spirit (`None` renders as `--`) |

## External Documentation

No external research needed. Bollinger (1980) `period=20`, `num_std=2.0` are the canonical defaults already implemented in `indicators/bollinger.py`; every threshold in this phase is either canonical, frozen by Phase 2, or a bar count carried over unchanged from Phase 4. The published evidence context (fade/mean-reversion has no comparable trend-following literature base) is already captured in `market-research-capability-benchmark.md` §2.7 and needs no re-gathering.

---

## Patterns to Mirror

### NAMING_CONVENTION
```python
# SOURCE: config.py:87-91 — SCREAMING_SNAKE constants under a phase-banner
# comment; unit or tier stated inline in the trailing comment
# Phase 4: mean-reversion fade signal method (ranging regime only).
# All thresholds are unvalidated defaults; tuned empirically in Phase 5.
BB_PERIOD = 20  # Bollinger middle-band SMA period (1H bars)
BB_STD = 2.0  # band width in rolling standard deviations
FADE_STRETCH_MAX_AGE_BARS = 6  # band-stretch bar must be within this many 1H bars
```
The new constant follows the same shape under a `# Phase 6:` banner. Note the existing comments say "1H bars" — Phase 4 rewords those to the setup tier; if it hasn't, reword only the `FADE_*`/`BB_*` lines you touch, and don't expand scope.

### REGIME_DISPATCH (live)
```python
# SOURCE: scan.py:48-54 — regime computed once, one method per label, empty
# list for suppressed labels. The regime label is ALWAYS returned truthfully,
# even when the method is suppressed.
    regime_label, _, _ = current_regime(conn, symbol, now_ms=now_ms)

    if regime_label == "trending":
        return (regime_label, scan_breakout_signals(conn, symbol, now_ms))
    if regime_label == "ranging":
        return (regime_label, scan_fade_signals(conn, symbol, now_ms))
    return (regime_label, [])
```

### REGIME_DISPATCH (backtest — must stay behaviorally identical)
```python
# SOURCE: engine.py:162-170 — the same dispatch, lazily cached per 1H bar.
# Divergence between this and scan.py is the failure mode the engine's
# docstring (engine.py:2-6) exists to prevent.
            cands: list = []
            if reg == "trending":
                cands = [("breakout", c) for c in detect_patterns(window)]
            elif reg == "ranging":
                cands = [
                    ("fade", c)
                    for c in detect_fade_setups(window, num_std=params.bb_num_std)
                ]
            cand_cache[h_idx] = (reg, cands)
```

### BUCKET_ATTRIBUTION
```python
# SOURCE: metrics.py:31-34 — per-(regime, pattern) stats come free from the
# Trade list. "ranging/bollinger-fade" is the fade sleeve's whole P&L story;
# no separate fade-only backtest run is required.
    buckets: dict[str, list] = {}
    for t in trades:
        buckets.setdefault(f"{t.regime}/{t.pattern}", []).append(t.pnl_pct)
    stats["by_bucket"] = {k: _stats(v) for k, v in sorted(buckets.items())}
```

### TUNING_VS_OOS_SPLIT
```python
# SOURCE: walkforward.py:126-136 — the ONLY definition of where the untouched
# holdout begins. Any analysis that must not spend the holdout computes its
# end bound exactly this way.
    oos_ms = (config.WF_OOS_DAYS if oos_days is None else oos_days) * DAY_MS
    tune_end = end_ms - oos_ms
```

### ANALYSIS_SCRIPT_SHAPE
```python
# SOURCE: scripts/export_bar_annotations.py:1-30 — docstring states what is
# replayed and why it cannot disagree with the backtest, ends with a Usage:
# block; argparse for options; production builders re-called rather than
# reimplemented ("this script owns no screening logic of its own").
"""
Export per-bar pipeline annotations for the interactive review chart.
...
Usage:
    python scripts/export_bar_annotations.py --out annotations.json
"""

import argparse
import json
import logging
import sys
```

### TEST_STRUCTURE (dispatcher suppression)
```python
# SOURCE: tests/test_meanrev.py:232-258 — class per dispatch concern, a _patch
# helper installing sentinels so the test asserts ROUTING, not signal content
class TestScanSymbol:
    def _patch(self, monkeypatch, regime):
        monkeypatch.setattr(
            scan, "current_regime", lambda *a, **k: (regime, 20.0, 0.5)
        )
        monkeypatch.setattr(
            scan, "scan_fade_signals", lambda conn, sym, now: ["fade-sentinel"]
        )

    def test_ranging_dispatches_to_fade_method(self, monkeypatch):
        self._patch(monkeypatch, "ranging")
        assert scan_symbol(None, SYMBOL, now_ms=START) == ("ranging", ["fade-sentinel"])
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/config.py` | UPDATE | Add `FADE_ENABLED = True` under a Phase 6 banner; reword the `BB_*`/`FADE_*` tier comments if Phase 4 left them saying "1H bars" |
| `src/trading_bot/signals/scan.py` | UPDATE | Guard the `ranging` branch on `config.FADE_ENABLED` (live path) |
| `src/trading_bot/backtest/engine.py` | UPDATE | Guard the `elif reg == "ranging"` branch on `config.FADE_ENABLED` (backtest path) — same semantics, or backtest and live drift |
| `scripts/fade_requalification.py` | CREATE | The measurement harness: candidate funnel, per-symbol + pooled fade metrics, median stop distance in ATR multiples, measured cost ratio `c`, rr distribution, stretch-depth bias check |
| `tests/test_meanrev.py` | UPDATE | Add `TestFadeEnabledSwitch` — live path suppressed when false, and unchanged when true |
| `tests/test_backtest.py` | UPDATE | One test: `FADE_ENABLED=False` produces no `ranging/bollinger-fade` bucket from a ranging fixture |
| `.claude/PRPs/reports/fade-requalification.md` | CREATE | The deliverable: pre-committed rule, measured numbers, verdict, and the trial-log entry recording what this phase spent |
| `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md` | UPDATE | Record the keep/drop outcome in the Decisions Log (one row) — the PRD's Phase 6 success signal is literally "a documented keep/drop decision" |

## NOT Building

- **Any change to the fade's stop placement.** The excursion extreme is structurally correct per the PRD Decisions Log (`PRD:280`) and Phase 2 deliberately left it alone. Moving it to `k × ATR` here would (a) contradict two prior phases and (b) make this measurement a measurement of a third thing.
- **Any retune of `BB_PERIOD`, `BB_STD`, or `FADE_STRETCH_MAX_AGE_BARS`.** 20/2.0 are canonical Bollinger; `FADE_STRETCH_MAX_AGE_BARS = 6` is carried over as a **bar count** (6 bars = 24h at the 4H setup tier, up from 6h at 1H), which is the zero-degrees-of-freedom choice — the same reasoning Phase 4 used for `MAX_HOLD_BARS_TRIGGER`. Retuning any of them is fitting on returns, which is the exact behavior the PRD forbids.
- **Sweeping anything.** No grid, no folds, no parameter selection. This phase's entire epistemic claim rests on sweeping nothing (see Notes).
- **Touching the OOS holdout.** Every number in this phase comes from `BACKFILL_START → end − WF_OOS_DAYS`. A keep/drop decision made on holdout data spends the only honest OOS data left (`PRD:243`, `PRD:78`).
- **Running or repairing the walk-forward gate** (Phase 7) — no pooled folds, no `WF_MIN_TRADES` change, no DSR, no `plateau_ratio` replacement.
- **Sharpe / Sortino / equity-curve metrics** (Phase 3). If Phase 3 has landed, the script may *report* Sharpe as supporting colour, but the decision rule below is stated in expectancy/PF/`c` terms so this phase does not acquire a dependency the PRD didn't give it (`PRD:201` lists Phase 6's dependency as Phase 4 only).
- **A replacement ranging method.** If fade fails, ranging becomes flat/no-signal. "Ranging regime produces no signals" is an acceptable result (`PRD:238`), and designing a substitute is not in v1.
- **Deleting `meanrev.py`, `bollinger.py`, or their tests** on a DROP verdict. The kill switch is the drop mechanism; the code stays, tested, so a later re-test costs nothing and the decision stays auditable.
- **`patterns.py` / `pivots.py` retirement** (Phase 5) and the Donchian engine itself.

---

## Step-by-Step Tasks

### Task 1: Add the `FADE_ENABLED` kill switch to config
- **ACTION**: Edit `src/trading_bot/config.py`, appending after the existing `FADE_STRETCH_MAX_AGE_BARS` line (currently `config.py:91`).
- **IMPLEMENT**:
  ```python
  # Phase 6: fade re-qualification. The ranging sleeve is kept behind an
  # explicit switch so a DROP verdict is a recorded decision rather than a code
  # deletion — the method stays tested and a future re-test costs nothing.
  # Honored by BOTH dispatch paths (signals/scan.py and backtest/engine.py);
  # flipping it must change live and backtest behavior identically.
  # Set from .claude/PRPs/reports/fade-requalification.md's verdict.
  FADE_ENABLED = True
  ```
  While here: if Phase 4 has **not** already reworded them, change `config.py:89` `# Bollinger middle-band SMA period (1H bars)` → `(setup-tier bars)` and `config.py:91` `# ... within this many 1H bars` → `... setup-tier bars`. If Phase 4 already did it, change nothing.
- **MIRROR**: NAMING_CONVENTION — phase-banner comment, SCREAMING_SNAKE, unit/tier in the trailing comment.
- **IMPORTS**: None (same file).
- **GOTCHA**: Ship it as `True`. It is a switch whose *value* is the Task 6 verdict; defaulting it to `False` would silently drop the sleeve before the measurement that decides its fate has run.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot import config; print(config.FADE_ENABLED)"` prints `True`.

### Task 2: Guard both dispatch paths on the switch
- **ACTION**: Edit `src/trading_bot/signals/scan.py:52-53` and `src/trading_bot/backtest/engine.py:166-169`.
- **IMPLEMENT**: In `scan.py`, `scan_symbol`:
  ```python
      if regime_label == "ranging":
          if not config.FADE_ENABLED:
              return (regime_label, [])
          return (regime_label, scan_fade_signals(conn, symbol, now_ms))
  ```
  In `engine.py`, `candidates_for`:
  ```python
              elif reg == "ranging" and config.FADE_ENABLED:
                  cands = [
                      ("fade", c)
                      for c in detect_fade_setups(window, num_std=params.bb_num_std)
                  ]
  ```
  Also extend the two dispatch docstrings that enumerate the regime→method mapping — `scan.py:5-13` and `engine.py:2-6` — with one line each: `ranging -> mean-reversion fade method (Phase 4), subject to config.FADE_ENABLED`.
- **MIRROR**: REGIME_DISPATCH (live) and REGIME_DISPATCH (backtest) above — keep the early-return shape in `scan.py` and the `elif`-guard shape in `engine.py`; do not restructure either function.
- **IMPORTS**: `scan.py` does **not** currently import config (it imports `current_regime`, `scan_fade_signals`, `Signal`, `scan_breakout_signals`) — add `from trading_bot import config`. No circularity: `config` imports only `datetime`. `engine.py` already imports `config` (`engine.py:36`).
- **GOTCHA**: Read `config.FADE_ENABLED` **at call time** (as written above), never capture it at import time into a module-level default — `tests/test_meanrev.py` and `test_backtest.py` flip it via `monkeypatch.setattr(config, "FADE_ENABLED", False)` and an import-time capture makes those tests silently pass while asserting nothing. Second gotcha: keep returning the *true* `regime_label` when suppressed; the classifier is used defensively and a suppressed method is not a misclassification. Third: Phase 5 edits the `if reg == "trending"` branch of this same `candidates_for` function — coordinate the merge, the two branches are adjacent lines.
- **VALIDATE**: `.venv/bin/pytest tests/ -q -m "not network"` still green (188 passed baseline); `grep -n "FADE_ENABLED" src/trading_bot/signals/scan.py src/trading_bot/backtest/engine.py src/trading_bot/config.py` returns exactly 3 files, and `grep -rn "FADE_ENABLED" src/ | wc -l` is 3 — one definition, two consumers.

### Task 3: Write `scripts/fade_requalification.py` — the measurement harness
- **ACTION**: Create `scripts/fade_requalification.py`.
- **IMPLEMENT**: A script that runs `run_backtest` once per symbol over the **tuning span only**, filters to fade trades, and prints five blocks. Structure:
  ```python
  """
  Measure the ranging-regime Bollinger fade sleeve under the Phase 2 risk model
  and the Phase 4 tiers, for the Phase 6 keep/drop decision.

  Runs the production run_backtest per symbol over the TUNING span only
  (BACKFILL_START -> end - WF_OOS_DAYS), so this measurement cannot spend the
  one-shot OOS holdout. Nothing is swept: every parameter is the frozen config
  default, which is why running on the tuning span consumes no degrees of
  freedom.

  Owns no screening logic — fade trades are whatever run_backtest produced with
  pattern == meanrev.FADE_KIND, so these numbers cannot disagree with the
  backtest the decision is written from.

  Usage:
      python scripts/fade_requalification.py
      python scripts/fade_requalification.py --start 2023-07-27 --symbol BTCUSDT
  """
  ```
  Blocks to print, in order:
  1. **Span banner** — resolved `start`/`tune_end` as ISO dates plus `WF_OOS_DAYS`, and an explicit line `OOS holdout NOT touched: <iso> -> <iso>`. Fail loudly (`sys.exit(2)`) if `--end` would exceed `tune_end`.
  2. **Per-symbol fade metrics + pooled** — `compute_metrics([t for t in trades if t.pattern == meanrev.FADE_KIND])`, printed with the same field set as `cli._print_metrics` (reuse it: `from trading_bot.cli import _print_metrics`). Pool by concatenating the three filtered trade lists and calling `compute_metrics` once.
  3. **Risk-model conformance** — for each fade trade, `risk_pct = abs(entry - stop) / entry`; report the median, and the median as a multiple of `ATR(setup_tf)` at entry (compute the ATR series once per symbol from the setup-tier DataFrame via `wilder.atr(df, config.ATR_STOP_PERIOD)` and index it with `np.searchsorted(close_setup, entry_ts, side="right") - 1`, the same alignment the engine uses at `engine.py:150-152`). Then `c = round_trip_cost / median_risk_pct` where `round_trip_cost = 2 * (config.FEE_PCT + config.SLIPPAGE_PCT)` plus the funding term Phase 2 added, using the median holding duration in days. Print `c` per symbol against `config.COST_RATIO_CEILING`.
  4. **Candidate funnel** — the count at each stage, per symbol: setups detected → triggered (`check_breakout` returned an event) → passed the `RR_FLOOR` filter → actually traded (the rest lost the `rank_signals` tie or arrived while a trade was open). Get stages 1–3 by re-walking the setup-tier bars and re-calling `detect_fade_setups` / `check_breakout` / `build_fade_signal` exactly as `engine.candidates_for` does — **re-call the production functions, never reimplement the screen** — and stage 4 from the `Trade` list. Also print the `rr` distribution (min / p25 / median / p75 / max) of *triggered* candidates, both accepted and rejected.
  5. **Adverse-selection check** — for each triggered candidate, `stretch_depth = abs(candidate.stop_level - candidate.trigger_level) / candidate.trigger_level`. Print median stretch depth for `RR_FLOOR`-accepted vs `RR_FLOOR`-rejected candidates. See the GOTCHA — this is the block most likely to change the verdict's meaning.
  Use `argparse` with `--symbol` (repeatable), `--start`, `--end` mirroring `cli.py:114-125`, and `config.date_to_ms` for date parsing.
- **MIRROR**: ANALYSIS_SCRIPT_SHAPE (`scripts/export_bar_annotations.py:1-30`) for the docstring/`argparse`/no-own-logic rule; TUNING_VS_OOS_SPLIT (`walkforward.py:126-132`) for the span arithmetic — copy `tune_end = end_ms - oos_ms` rather than inventing a split; BUCKET_ATTRIBUTION for why no fade-only engine mode is needed.
- **IMPORTS**: `argparse`, `logging`, `statistics`, `sys`, `numpy as np`, `pandas as pd`; `from trading_bot import config`, `from trading_bot.data import storage`, `from trading_bot.backtest.engine import run_backtest, _df`, `from trading_bot.backtest.metrics import compute_metrics`, `from trading_bot.cli import _print_metrics`, `from trading_bot.indicators.wilder import atr`, `from trading_bot.signals import meanrev`, `from trading_bot.signals.breakout import check_breakout`.
- **GOTCHA**: Four, in descending order of how badly they bite.
  1. **Start the span at 2023-07-27, not `BACKFILL_START`.** Phase 4's plan measured that `REGIME_MIN_BARS = 207` becomes 207 *calendar days* at the 1D regime tier, so no non-`uncertain` label exists before ~2023-07-27 and a span starting at `BACKFILL_START` reports a wall of zero-trade months. Default `--start` to `2023-07-27` in this script and say so in its `--help`.
  2. **The `RR_FLOOR` filter may be adversely selecting fades, and block 5 is what detects it.** Fade reward is bounded by band geometry (`target` = middle band, so reward ≈ the stretch-to-mean distance) while risk = entry − excursion extreme. A *deep* stretch has large risk and therefore small `rr` → rejected; a *shallow* stretch passes. If block 5 shows accepted candidates are systematically shallower, `RR_FLOOR ≥ 1.5` is doing to fade a milder version of what `MAX_RISK_PCT` did to breakout — admitting the freakishly tight and discarding the structurally sound. That finding belongs in the report **even on a KEEP verdict**, and it is a genuine Phase 7 input, not a Phase 6 fix.
  3. **`_df` and `_print_metrics` are underscore-private.** Importing them from a script is acceptable here (this is analysis tooling, not library code, and `export_bar_annotations.py` already reaches into engine internals the same way) but do not promote them to public API as a drive-by.
  4. `detect_fade_setups` returns **0–2** candidates (at most one long, one short — `meanrev.py:112`), so funnel stage 1 counts candidates, not bars. Don't report it as "bars with a stretch".
- **VALIDATE**: `.venv/bin/python scripts/fade_requalification.py --symbol BTCUSDT` exits 0 and prints all five blocks with a non-empty funnel; re-running with `--end` past `tune_end` exits 2 with the holdout-protection message. Cross-check block 2 against `.venv/bin/python -m trading_bot.cli backtest --symbol BTCUSDT --start 2023-07-27 --end <tune_end>` — the script's fade numbers must equal that run's `ranging/bollinger-fade` bucket exactly. If they differ, the script grew its own screening logic; delete the difference, don't reconcile it.

### Task 4: Kill-switch tests (live path)
- **ACTION**: Edit `tests/test_meanrev.py`, adding a class after `TestScanSymbol` (currently ends at line 258).
- **IMPLEMENT**:
  ```python
  class TestFadeEnabledSwitch:
      def _patch(self, monkeypatch, enabled):
          monkeypatch.setattr(scan, "current_regime", lambda *a, **k: ("ranging", 20.0, 0.5))
          monkeypatch.setattr(scan, "scan_fade_signals", lambda conn, sym, now: ["fade-sentinel"])
          monkeypatch.setattr(config, "FADE_ENABLED", enabled)

      def test_enabled_dispatches_to_fade(self, monkeypatch):
          self._patch(monkeypatch, True)
          assert scan_symbol(None, SYMBOL, now_ms=START) == ("ranging", ["fade-sentinel"])

      def test_disabled_suppresses_fade_but_reports_regime(self, monkeypatch):
          self._patch(monkeypatch, False)
          assert scan_symbol(None, SYMBOL, now_ms=START) == ("ranging", [])
  ```
- **MIRROR**: TEST_STRUCTURE — `TestScanSymbol._patch`'s sentinel technique, verbatim shape.
- **IMPORTS**: `config` and `scan` are already imported at `tests/test_meanrev.py:7,10`; `scan_symbol` at line 18. Nothing new.
- **GOTCHA**: `monkeypatch.setattr(config, "FADE_ENABLED", ...)` patches the module attribute, which only works because Task 2 reads it at call time. If `test_disabled_suppresses_fade_but_reports_regime` passes when you *revert* Task 2's `scan.py` guard, the test is asserting nothing — check that inversion once, deliberately.
- **VALIDATE**: `.venv/bin/pytest tests/test_meanrev.py -v` — all pass, 2 new tests visible in the output.

### Task 5: Kill-switch test (backtest path)
- **ACTION**: Edit `tests/test_backtest.py`, adding one test beside the existing ranging/fade coverage.
- **IMPLEMENT**: Using the file's existing ranging fixture and its `patch_trending`-style monkeypatch helper (`tests/test_backtest.py:83-87` patches `engine.classify_series` wholesale — use the analogous ranging patch), assert that with `monkeypatch.setattr(config, "FADE_ENABLED", False)` the resulting `compute_metrics(trades)["by_bucket"]` contains **no** key starting `"ranging/"`, and that with it `True` the same fixture does produce `ranging/bollinger-fade`. Name it `test_fade_disabled_produces_no_ranging_trades`.
- **MIRROR**: the bucket assertion at `tests/test_backtest.py:50-52` (`assert set(m["by_bucket"]) == {"trending/flag", "ranging/bollinger-fade"}`) — same style, adapted to an absence check.
- **IMPORTS**: `config` is already imported in `tests/test_backtest.py`; add nothing.
- **GOTCHA**: `tests/test_backtest.py:73` seeds only 12 regime bars against `REGIME_MIN_BARS = 207` — that is fine and must stay fine, because `classify_series` is monkeypatched wholesale so the warmup gate is never reached (per Phase 4's Task 8 note). Do **not** inflate the fixture. Also do not touch `test_backtest.py:101,116`, which assert Phase-2-retired constants and are Phase 2's to rewrite.
- **VALIDATE**: `.venv/bin/pytest tests/test_backtest.py -v` — all pass; the new test fails if you revert Task 2's `engine.py` guard.

### Task 6: Run the measurement and write the decision report
- **ACTION**: Run Task 3's script for all 3 symbols, then create `.claude/PRPs/reports/fade-requalification.md`.
- **IMPLEMENT**: Read the pre-committed decision rule below **before** looking at any output, then write the report with these sections, in this order (the order matters — the rule is stated before the numbers so the document itself shows the rule wasn't moved):
  1. **Pre-committed decision rule** — copied verbatim from this plan.
  2. **What changed since the last measurement** — Phase 2's `RR_FLOOR` + frozen costs + funding term; Phase 4's 4H setup / 1H trigger + `interval_ms` fix. State that the stop placement did **not** change.
  3. **Measured numbers** — the script's five blocks, per symbol and pooled.
  4. **Verdict** — KEEP or DROP, with the rule's clause that decided it named explicitly.
  5. **Adverse-selection finding** — block 5's result and what it implies for Phase 7, regardless of verdict.
  6. **Trial-log entry** — what this phase spent: *zero new degrees of freedom* (nothing swept), one tuning-span read of the fade sleeve. Also record the **pre-existing** debt this phase discovered: `bb_num_std ∈ {1.75, 2.0, 2.25}` is in `walkforward.py:39`'s live grid, so `BB_STD` has already been fitted on BTCUSDT 2023–2026 by the old Phase 5 sweep. Phase 7 removes `bb_num_std` from the grid; until it does, that consumed DoF is real and belongs in the log.

  **Pre-committed decision rule** (evaluated on the tuning span, pooled across BTCUSDT/ETHUSDT/SOLUSDT, all parameters frozen):
  - **KEEP** if *all four* hold: pooled fade `n_trades ≥ 30`; pooled `expectancy_pct > 0`; `expectancy_pct > 0` on **at least 2 of 3** symbols; measured `c ≤ config.COST_RATIO_CEILING` on all 3 symbols.
  - **DROP** if pooled `expectancy_pct ≤ 0`, **or** pooled `n_trades < 30`, **or** only 1 of 3 symbols is positive, **or** `c > COST_RATIO_CEILING` on any symbol.
  - The 2-of-3 bar is deliberately looser than Phase 7's mandatory 3-of-3 gate: this is a keep-for-further-testing decision on a droppable `Should` sleeve, not the gate. A sleeve that clears 2-of-3 here still has to clear 3-of-3 in Phase 7, where it can still be dropped.
  - `n_trades < 30` resolving to DROP is intentional. An unjudgeable sample is not a reason to carry a sleeve into the gate; "ranging produces no signals" is an acceptable outcome (`PRD:238`).
  - **No re-running with a different `--start`, symbol subset, or span to change the answer.** If the span genuinely needs correcting (e.g. the regime-warmup date was wrong), record both runs and the reason in the trial log.
- **MIRROR**: `market-research-capability-benchmark.md`'s house style — measured tables with explicit `n`, source citations by `file:line`, and conclusions stated as decisions rather than suggestions.
- **IMPORTS**: N/A (document).
- **GOTCHA**: Do not start this task before Phases 2 and 4 are both landed and `.venv/bin/pytest tests/ -q -m "not network"` is green. Measuring a half-migrated state produces a number that describes nothing shippable, and re-running later to "confirm" turns one honest read into two, which is how a tuning span quietly becomes a fitted one.
- **VALIDATE**: The report exists, states the verdict, and every number in it is reproducible by re-running the exact command line recorded in the report.

### Task 7: Apply the verdict
- **ACTION**: Set `config.FADE_ENABLED` to the Task 6 verdict; add one row to the PRD's Decisions Log; update the PRD's Phase 6 row.
- **IMPLEMENT**: On DROP, `FADE_ENABLED = False` and extend its comment with `# DROPPED per .claude/PRPs/reports/fade-requalification.md (<date>): <one-line reason>`. On KEEP, leave it `True` and add the analogous `# KEPT per ...` line. Then in `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`, add a Decisions Log row: `| Ranging sleeve (fade) | KEEP/DROP | ... | <the deciding clause + the numbers> |`, replacing the current provisional row at `PRD:280` (`Ranging signal method | Keep fade, but re-qualify ...`) — that row's job was to defer this decision, and it is now made.
- **MIRROR**: The Decisions Log table's 4-column shape (`PRD:272-293`): Decision / Choice / Alternatives / Rationale, rationale carrying the numbers.
- **IMPORTS**: N/A.
- **GOTCHA**: On DROP, leave `meanrev.py`, `bollinger.py`, `tests/test_meanrev.py`, and the `bollinger`/`detect_fade_setups` tests fully intact and passing. Deleting them saves nothing and converts a reversible, auditable decision into an irreversible one — and Phase 10's escalation paths may want the ranging sleeve back.
- **VALIDATE**: `.venv/bin/pytest tests/ -q -m "not network"` green with the verdict applied; on DROP, `.venv/bin/python -m trading_bot.cli backtest --symbol BTCUSDT --start 2023-07-27` shows no `ranging/` bucket.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| `test_enabled_dispatches_to_fade` | ranging regime, `FADE_ENABLED=True` | `("ranging", ["fade-sentinel"])` | No |
| `test_disabled_suppresses_fade_but_reports_regime` | ranging regime, `FADE_ENABLED=False` | `("ranging", [])` — label truthful, list empty | Yes — suppression must not masquerade as a different regime |
| `test_fade_disabled_produces_no_ranging_trades` | ranging fixture, `FADE_ENABLED=False` | no `"ranging/"` key in `by_bucket` | Yes — the backtest/live parity guarantee |
| existing `test_ranging_dispatches_to_fade_method` | unchanged | unchanged | Regression: default-on behavior preserved |
| existing `TestDetectFadeSetups` (5 tests) | unchanged | unchanged | Regression: this phase touches no detection logic |
| existing `TestBuildFadeSignal` (6 tests) | unchanged (Phase 2 owns their rewrite) | unchanged | Regression: do not re-touch Phase 2's `RR_FLOOR` assertions |

### Edge Cases Checklist
- [ ] `FADE_ENABLED=False` in a **ranging** regime → empty signals, correct label (both paths)
- [ ] `FADE_ENABLED=False` in a **trending** regime → breakout path completely unaffected
- [ ] `FADE_ENABLED=True` → byte-for-byte identical behavior to pre-Task-2 (the 188-test baseline stays green)
- [ ] Script on a symbol with zero fade trades → prints `--` for undefined ratios (via `_print_metrics`), does not divide by zero, exits 0
- [ ] Script with `--end` beyond `tune_end` → exits 2 without printing metrics
- [ ] Script when the setup-tier ATR is NaN during warmup → those trades excluded from the ATR-multiple median, count of exclusions reported
- [ ] Config read at call time, not import time (verified by deliberately reverting a guard and watching the test fail)
- N/A: concurrent access (single-process CLI), network failure (no network in this phase), permission denied (local SQLite read)

---

## Validation Commands

### Static Analysis
```bash
.venv/bin/python -m compileall -q src/trading_bot scripts
```
EXPECT: no output (no syntax errors). **No linter and no type checker are configured in this repo** — `pyproject.toml` declares only `pytest` under `[project.optional-dependencies].dev`, and there is no `Makefile`, `noxfile`, `ruff`/`mypy`/`flake8` config. Do not invent a lint step.

### Unit Tests
```bash
.venv/bin/pytest tests/test_meanrev.py tests/test_backtest.py -v
```
EXPECT: all pass, including the 3 new kill-switch tests.

### Full Test Suite
```bash
.venv/bin/pytest tests/ -q -m "not network"
```
EXPECT: `191 passed, 1 deselected` (baseline measured on this branch: `188 passed, 1 deselected in 2.51s`, Python 3.11.6). The `network` marker is declared at `pyproject.toml:[tool.pytest.ini_options].markers`.

### Switch-Wiring Verification
```bash
grep -rn "FADE_ENABLED" src/ tests/
```
EXPECT: exactly one definition (`config.py`), exactly two production consumers (`signals/scan.py`, `backtest/engine.py`), and test references only in `tests/test_meanrev.py` / `tests/test_backtest.py`. Any other production read means the switch has leaked into logic it shouldn't gate.

### Measurement Run
```bash
.venv/bin/python scripts/fade_requalification.py
.venv/bin/python -m trading_bot.cli backtest --start 2023-07-27 --end <tune_end>
```
EXPECT: the script's per-symbol fade block equals the CLI run's `ranging/bollinger-fade` bucket, symbol for symbol. A mismatch means the script reimplemented screening — fix the script, not the reconciliation.

### Manual Validation
- [ ] Phases 2 and 4 are landed and the suite is green **before** the measurement run
- [ ] The script's span banner shows an OOS holdout that was not touched
- [ ] The decision rule in the report is textually identical to the one in this plan
- [ ] The verdict names the specific clause that decided it
- [ ] The adverse-selection block is reported even on a KEEP
- [ ] The trial-log entry records zero new degrees of freedom **and** the pre-existing `bb_num_std` debt
- [ ] `config.FADE_ENABLED` matches the verdict, with the report cited in its comment
- [ ] The PRD's provisional "keep fade, but re-qualify" Decisions Log row is replaced by the actual decision

---

## Acceptance Criteria
- [ ] `FADE_ENABLED` exists, defaults `True`, and is honored identically by the live and backtest dispatch paths
- [ ] 3 new tests, all failing if their corresponding guard is reverted
- [ ] `scripts/fade_requalification.py` runs on all 3 symbols, reproduces the CLI's fade bucket exactly, and refuses to read the OOS holdout
- [ ] `.claude/PRPs/reports/fade-requalification.md` exists with rule-before-numbers ordering, a verdict, and the trial-log entry
- [ ] `FADE_ENABLED` set to the verdict, citing the report
- [ ] PRD Decisions Log records the outcome
- [ ] Full suite green; no fade detection, stop placement, or threshold changed anywhere

## Completion Checklist
- [ ] Code follows the discovered dispatch patterns (early-return in `scan.py`, `elif`-guard in `engine.py`)
- [ ] No new logging added (neither dispatcher logs suppression today; suppression is expected control flow, not an event)
- [ ] Tests follow `TestScanSymbol`'s sentinel/monkeypatch idiom
- [ ] No hardcoded values — the span comes from `config.WF_OOS_DAYS`, costs from `config.FEE_PCT`/`SLIPPAGE_PCT`, the ceiling from `config.COST_RATIO_CEILING`
- [ ] `meanrev.py` / `bollinger.py` / their tests intact even on a DROP
- [ ] Nothing swept; the OOS holdout untouched
- [ ] No Phase 5 (Donchian), Phase 7 (gate), or Phase 3 (Sharpe) work absorbed
- [ ] Self-contained — no questions needed during implementation

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Measurement is run before Phases 2 and 4 land, describing a state that never ships | **High** (both are prerequisites and Phase 2 is `in-progress`) | High — the verdict would be about the wrong strategy | Task 6's GOTCHA blocks on both phases + a green suite; Tasks 1, 2, 4, 5 are implementable and testable *now* and don't depend on either |
| `RR_FLOOR ≥ 1.5` adversely selects shallow stretches, so a KEEP verdict keeps a subtly broken sleeve | Medium-High | High — it is a milder replay of the exact defect the PRD exists to fix | Block 5 measures it explicitly and the report carries the finding regardless of verdict; the remedy (an rr floor that respects band geometry) is deliberately deferred to Phase 7's grid, not improvised here |
| Wider stops + coarser tiers shrink fade to an unjudgeable sample | Medium-High | Medium | Pool all 3 symbols (the PRD's only free lunch); `n < 30` resolves to DROP by pre-commitment rather than to a judgement call |
| Pressure to retune `FADE_STRETCH_MAX_AGE_BARS` because 6 bars is now 24h instead of 6h | Medium | High — it is fitting on returns, and Phase 4 explicitly parked the temptation here | NOT Building states the bar count is carried over unchanged as the zero-DoF choice; any retune is a Phase 7 grid axis with a logged DoF, never a Phase 6 edit |
| Fade's excursion-extreme stop fails `c ≤ 0.10` on a symbol, since it is not ATR-derived and inherits none of Phase 2's guarantee | Medium | Medium | `c` is measured per symbol in block 3 and is a hard DROP clause; failing it is a legitimate verdict, not a bug to fix |
| Verdict is re-litigated by re-running with a different span/symbol subset | Medium | High — turns one honest read into a search | The rule forbids it and requires both runs logged if a span correction is genuinely needed |
| Kill switch read at import time, so both new tests pass vacuously | Low-Medium | Medium | Task 2's GOTCHA + the deliberate revert-and-watch-it-fail check in the edge-case list |
| Phase 5 edits the adjacent `trending` branch of `candidates_for`, causing a merge conflict | Medium | Low | Phases 5 and 6 are declared parallel with one shared dispatch table (`PRD:264`); the two branches are adjacent lines — merge, don't rebase blindly |

## Notes

**This phase's central claim is epistemic, not mechanical.** It sweeps nothing: every parameter is a frozen config default, no grid, no folds, no selection. That is exactly why it may honestly read the *tuning* span — a measurement that selects nothing consumes no degrees of freedom, so the tuning span behaves like out-of-sample data for this one question. The moment any threshold is adjusted in response to the output, that property is gone and the read is retroactively contaminated. The pre-committed rule and the rule-before-numbers report ordering exist to make that violation visible if it happens.

**Why a switch rather than a deletion.** The PRD calls the ranging sleeve a `Should` (`PRD:116`) and says dropping it is acceptable — but it also pre-commits Phase 10 escalation paths that may want it back, and a DROP here is made on a tuning-span read, not the gate. A one-line switch keeps the code tested, keeps the decision auditable next to its evidence, and makes "we turned it off on this date for these reasons" a fact in the repository rather than an absence in the git history.

**Why the fade was always the more interesting of the two methods.** The benchmark's own reading (`market-research-capability-benchmark.md:114`): fade win rates ran roughly double the breakout method's, and fade had the best profit factors, because its stop sits at the excursion extreme — structurally *further away* than a hairline buffer past a breakout level. Wider stop, better outcome, exactly as the noise math predicts. That is the same insight the entire Phase 2 pivot rests on, observed in fade's numbers before anyone acted on it. It is a reason to test the sleeve fairly, and not a reason to expect it to pass.

**Pre-existing degree-of-freedom debt found while planning this.** `walkforward.py:39` has `"bb_num_std": (1.75, 2.0, 2.25)` in the live `DEFAULT_GRID`, and `BacktestParams.bb_num_std` (`engine.py:57`) threads it into `detect_fade_setups` (`engine.py:168`). So the old Phase 5 sweep already fitted the fade's band width on BTCUSDT 2023–2026. Phase 7 removes it from the grid; this phase freezes it at the canonical `2.0` and records the spend. Worth noting that the fade's *only* previously-swept parameter is the one this phase declines to touch.

**Dependency posture on Phase 3.** The PRD lists Phase 6's dependency as Phase 4 only, so the decision rule is stated in expectancy / profit-factor / cost-ratio terms, all available from today's `compute_metrics`. If Phase 3 has landed by the time Task 6 runs, report Sharpe as supporting colour — but do not make it a rule clause, or Phase 6 acquires a barrier the PRD didn't give it and the parallelism with Phase 5 is lost.
