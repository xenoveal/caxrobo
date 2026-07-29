# Implementation Report: Phase 3 Breakout Detection Fixes

**Date**: 2026-07-26
**Branch**: `fix/phase3-breakout-detection` (from `master` @ `dcb22d0`)
**Source plan**: `.claude/PRPs/reports/code review/phase3-breakout-detection-review.md`
**Status**: all 15 findings addressed (5 HIGH, 7 MEDIUM, 3 LOW)

## Summary

Fixed every finding from the Phase 3 review. The bulk of the work is in
`patterns.py`: the triangle detector gained the four constraints it was missing,
the H&S detector now collapses same-kind pivot runs before scanning, and no
detector emits a pattern whose level has already been closed through. The trigger
and signal layers gained a contiguity guard, an explicit entry-extension cap with
logged rejections, and a single shared ranking rule so the backtest stops
selecting candidates alphabetically.

Behavior is verified on the full stored history for all three symbols, not only
on fixtures.

## Measured effect (1H, 4,432 windows per symbol, 2023-01-01 → 2026-07-23)

Pattern emissions per symbol, before → after:

| Pattern | BTC before | BTC after | ETH after | SOL after |
|---|---|---|---|---|
| triangle long / short | 1,433 (32% of bars) | **19 (0.4%)** | 10 (0.2%) | 9 (0.2%) |
| head-and-shoulders | 54 | **154 (2.9×)** | 221 | 216 |
| inverse-H&S | 67 | **156 (2.3×)** | 220 | 253 |
| flag long | 433 | **491** | 664 | 949 |
| flag short | 387 | **414** | 674 | 943 |
| **emissions with price already past the level** | **354** | **0** | **0** | **0** |

Reading: the triangle detector stopped being an always-on noise source (32% →
0.4%), H&S detection roughly tripled as the run-collapse analysis predicted
(87 → 269 geometries over full history), the flag containment fix recovered ~13%
more bull flags, and the already-broken-level class is eliminated outright on all
three symbols.

BTC full backtest still runs end-to-end: 384 trades, and the H&S/triangle
patterns that alphabetical selection had made nearly unreachable now appear
(18 H&S, 11 triangle). Win rate moved 12.66% → 14.84%. These are correctness
fixes, not tuning — the method remains below break-even, which is the Phase 6
cost-ratio work.

## Findings → fixes

| ID | Fix | Location |
|---|---|---|
| **H1** | Triangle now requires four tests beyond convergence: bounded width (`TRIANGLE_MIN/MAX_WIDTH_BARS`), non-expanding sides, bar-by-bar containment within `TRIANGLE_CONTAINMENT_TOL`, and the latest close still inside the lines | `patterns.py:_detect_triangles`, `_triangle_contains` |
| **H2** | `_collapse_runs` reduces each same-kind pivot run to its extreme before the 5-tuple scan | `patterns.py:_collapse_runs` |
| **H3** | Freshness gate uses `min(...)` of the two sides' last pivots, so both trendlines must be current | `patterns.py:_detect_triangles` |
| **H4** | `_level_unbroken` rejects any pattern whose level was closed through after the geometry completed | `patterns.py:_level_unbroken` |
| **H5** | `rank_signals` (highest R:R, then pattern for determinism) is the one selection rule; live scanning returns signals in that order and the engine picks `rank_signals(...)[0]` instead of the first candidate | `setup.py:rank_signals`, `engine.py` |
| **M1** | `_strictly_ordered` requires strictly increasing bar indices across a pattern window; `find_pivots` docstring documents the dual-kind outside bar | `patterns.py`, `pivots.py` |
| **M2** | `check_breakout` takes `lookback_bars` (default 1) and scans newest-first, returning the most recent crossing; `scan_breakout_signals` warns when 15m data is behind `now_ms` | `breakout.py`, `setup.py` |
| **M3** | `_contiguous_tail` trims loaded frames to the longest evenly-spaced run and logs the drop; `check_breakout` takes `interval_ms` and skips non-adjacent crossing pairs | `setup.py`, `breakout.py` |
| **M4** | `_line_value` raises `ValueError` on coincident x; triangle floors the pivot count at 2 and checks distinct indices | `patterns.py` |
| **M5** | `pole_start = pole_end - FLAG_POLE_WINDOW_BARS + 1`, so the inclusive slice spans exactly the configured 12 bars | `patterns.py:_detect_flags` |
| **M6** | Flag containment tests against the pole's extreme over the whole pole window, matching where `pole_height` is measured | `patterns.py:_detect_flags` |
| **M7** | `BREAKOUT_MAX_ENTRY_EXTENSION_PCT` makes the previously implicit cap explicit and separately tunable; all three band rejections now log | `setup.py:build_signal`, `config.py` |
| **L1** | `_dedupe` keeps one candidate per (kind, direction) — freshest, then largest target | `patterns.py:_dedupe` |
| **L2** | Triangle `end_ts` anchored at `end_x`, the bar its levels are evaluated at | `patterns.py:_detect_triangles` |
| **L3** | Both shoulders must sit on the correct side of the neckline | `patterns.py:_detect_head_and_shoulders` |

## Validation

| Level | Result |
|---|---|
| Static analysis | `compileall` clean; all modules import. No linter/type-checker is configured (`pyproject.toml` declares only `pytest`), so lint and type-check are **skipped, not passed** |
| Unit tests | **188 passed, 1 skipped** (was 165 + 1) — 23 tests added |
| Build | N/A — no build step |
| Integration | `python -m trading_bot.cli signal` exits 0 and renders all three symbols; the new staleness warning correctly reports the store is ~3 days behind |
| Edge cases | Real-data sweep over 3 symbols × 4,432 windows (table above); full BTC backtest completes |

## Files changed

| File | Action | Lines |
|---|---|---|
| `src/trading_bot/signals/patterns.py` | UPDATED | +237 / −56 |
| `tests/test_signals.py` | UPDATED | +273 / −7 |
| `src/trading_bot/signals/breakout.py` | UPDATED | +126 / −38 |
| `src/trading_bot/signals/setup.py` | UPDATED | +110 / −14 |
| `src/trading_bot/config.py` | UPDATED | +24 / −3 |
| `src/trading_bot/backtest/engine.py` | UPDATED | +22 / −8 |
| `src/trading_bot/signals/pivots.py` | UPDATED | +6 |

## Tests written

| Area | Tests | What they pin |
|---|---|---|
| H&S | 5 | already-broken neckline (both directions), detection through a same-kind pivot run, same-bar shoulder/trough rejection, overlap dedup |
| Triangle geometry | 6 | contained wedge detected; pierced trendline, last-close-outside, expanding side, sub-minimum width, and one-stale-side all rejected |
| Flags | 2 | pole window spans exactly 12 bars; consolidation may reach a pole high set before the pole's last bar |
| Trigger | 3 | default lookback sees only the latest bar; widened lookback returns the crossing bar's own close and volume; gapped crossing pair skipped only when `interval_ms` is passed |
| Bands | 2 | entry-extension rejection, and that lifting it alone still trips the risk band (the level-anchored-stop coupling) |
| Ranking | 2 | R:R beats pattern name; ties break deterministically |
| Contiguity | 3 | untouched when even, trims after the last gap, warns |
| Guards | 1 | `_line_value` raises on coincident x |

## Deviations from the plan

1. **Two existing tests were changed, not just added to.**
   `test_detects_hs_short` and `test_detects_inverse_hs_long` appended a tail that
   closed *through* the neckline (94.8/94.4/94.0 against a 95 neckline), so they
   asserted detection of a pattern whose breakout had already completed — the
   exact H4 defect. Their fixtures now stop short of the neckline, and the
   original fixtures were kept as two new tests asserting rejection.

2. **M7 stops short of the review's own recommendation.** The review suggested
   replacing the level-anchored stop with a structural or ATR stop. That changes
   every Phase 5 number, so it is a tuning decision rather than a bug fix:
   instead the implicit cap is now an explicit, separately tunable, logged
   rejection, and `config.py` records why. The structural stop is left to Phase 6.

3. **Plan not archived.** The source document is a review artifact in `reports/`,
   referenced by path in the request; moving it to `plans/completed/` would break
   that reference. It stays where it is.

## Follow-ups (not bugs introduced here)

- **Triangles may now be too strict** at 0.2–0.4% of bars. The geometry is sound,
  but `TRIANGLE_CONTAINMENT_TOL` and `TRIANGLE_MIN_CONVERGENCE` deserve a sweep in
  Phase 6 to confirm the detector is not over-suppressed.
- **The fade path (`meanrev.py:266`) still calls `check_breakout` with defaults**,
  so it gets no `interval_ms` contiguity guard. Phase 4 was outside this review's
  scope and its behavior is unchanged, but the same one-line guard applies.
- **No scheduled signal scan exists.** `BREAKOUT_TRIGGER_LOOKBACK_BARS` and the
  staleness warning make late scans visible rather than silent, but the real fix
  is a 15m-boundary job in Phase 7 alerting.
- **Phase 5 numbers are stale.** H5 changed which candidate the backtest takes, so
  the walk-forward results and the cost-ratio analysis should be re-run before
  further tuning conclusions are drawn from them.

## Next steps

- [ ] Re-run walk-forward validation across all three symbols on the fixed engine
- [ ] Review the diff and commit
