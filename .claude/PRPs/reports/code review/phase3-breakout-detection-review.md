# Code Review: Phase 3 — Chart-Pattern Breakout Detection

**Reviewed**: 2026-07-26
**Scope**: `signals/pivots.py`, `signals/patterns.py`, `signals/breakout.py`, `signals/setup.py`
(+ `backtest/engine.py` candidate selection, in scope as a correlated cause)
**Baseline**: commit `f6fa289` (working tree clean)
**Decision**: REQUEST CHANGES — 5 HIGH findings

## Summary

The trigger layer (`breakout.py`) and the SL/TP/R:R layer (`setup.py`) are sound: the
fresh-crossing rule, closed-bar handling, NaN-safe volume ratio, and `end_ts` ordering
check all hold up. The defects are concentrated in the **pattern geometry**
(`patterns.py`): the triangle detector is effectively a 7-day range-contraction test with
no containment constraint (fires on 32% of bars, always in both directions), and the H&S
detector misses roughly two thirds of real geometries because it requires strict
adjacency-alternation in the raw pivot list.

All measurements below are on BTCUSDT 1H, 31,198 bars (2023-01-01 → present), sampled
every 7 bars over 4,432 windows unless stated otherwise.

---

## HIGH

### H1 — Triangle detector has no containment or slope constraint (false breakouts)
`patterns.py:167-230`

The "triangle" is two lines through the **first and last** pivot high / pivot low in the
entire 180-bar lookback. Nothing requires the highs to descend, the lows to ascend, or the
lines to actually bound the intervening price action. The only test is that the vertical
gap contracted by ≥25% from `start_x` to `n-1`.

Measured consequences:

| Metric | Value |
|---|---|
| Windows emitting a triangle pair | 1,433 / 4,432 (**32%**) |
| Emissions where both directions are emitted | 100% (by construction, lines 210-229) |
| Median trendline pivot span | **159 bars** (~7 days) |
| Emissions where the last close is already outside the lines | 330 / 1,433 (**23%**) |
| Median `target_height / price` | 2.5% (p90 5.7%, max 19.8%) |

So on a third of all bars the system carries a long *and* a short level derived from lines
that frequently do not contain price at all, and `target_height = start_range` — the gap
between two extrapolated lines ~160 bars back — becomes the measured-move TP. With
`BREAKOUT_STOP_BUFFER_PCT = 0.001` the risk leg is tiny, so these candidates sail through
the R:R band in `build_signal` and dominate the signal population.

**Fix**: require monotonic convergence (upper pivot highs non-increasing, lower pivot lows
non-decreasing), require the lines to contain the intervening bars within a tolerance, and
bound the pattern width (e.g. 20-80 bars) instead of letting it span the whole lookback.

### H2 — H&S requires strict adjacency in the raw pivot list (missed patterns)
`patterns.py:112-116`

The scan only accepts 5 *consecutive entries* of `pivots` alternating exactly
`high,low,high,low,high`. Real fractal output does not alternate: on BTC 1H there are 993
runs of 2 same-kind pivots, 136 of 3, and 21 of 4. Only **36.4%** (2,313 / 6,359) of
5-pivot windows alternate at all — any noise pivot between the shoulder and the trough
silently deletes that whole region from H&S consideration.

Collapsing each same-kind run to its extreme (the standard approach) before scanning:

| Pivot sequence | H&S + inverse-H&S geometries found |
|---|---|
| Raw list (current code) | 87 |
| Same-kind runs collapsed | **269** (3.1×) |

**Fix**: collapse same-kind runs to the max high / min low before the 5-tuple scan.

### H3 — Triangle freshness gate only checks the fresher side (false levels)
`patterns.py:184-185`

```python
last_pivot_index = max(ph[-1].index, pl[-1].index)
if not fresh(last_pivot_index):
```

`max` means one trendline can be anchored on an arbitrarily old pivot while the other is
recent. Age of the *staler* side's last pivot across triangle-eligible windows: median 10
bars, p90 17, **max 48**, and it exceeds `PATTERN_MAX_AGE_BARS` (12) in **32%** of cases.
That stale line is still extrapolated to `n-1` and used as a tradeable breakout level.

**Fix**: use `min(...)`, or gate each side independently.

### H4 — H&S candidates emitted after the neckline was already broken (false breakouts)
`patterns.py:118-124`, `patterns.py:139-146`, `breakout.py:91-94`

`fresh()` bounds only the last pivot's age; nothing checks that price has not already
closed through the neckline between `end_ts` and the current bar. Measured on emission:

- head-and-shoulders / short: 9 / 54 emissions already had the close **below** the neckline
- inverse-H&S / long: 15 / 67 emissions already had the close **above** the neckline
- triangle: 330 / 1,433 (see H1)

`check_breakout`'s fresh-crossing rule (`prev close` on the correct side) then makes these
fire on the *re-break* after a retest — i.e. a "breakout" of a pattern whose thesis was
already consumed, entered late with the measured-move target still computed from the
original geometry.

**Fix**: reject a candidate whose level has already been closed through by any bar after
the pattern's last pivot.

### H5 — Backtest picks candidates alphabetically, so Phase 5 measures a biased subset
`backtest/engine.py:238-242` + `patterns.py:101`

`detect_patterns` sorts by `(kind, direction)` and the engine `break`s on the first
candidate that yields a signal. The sort order is therefore
`flag/long → flag/short → head-and-shoulders → inverse-head-and-shoulders → triangle/long
→ triangle/short`. Flags always win; `triangle/short` is only ever reachable when
everything else failed. Live `scan_breakout_signals` (`setup.py:190-199`) returns **all**
signals with no such preference.

This is Phase 5 code, but it is in scope: it means the Phase 5 win-rate numbers driving the
current tuning work do not describe the breakout method as it behaves live, and the
per-pattern attribution in those reports is skewed by name ordering rather than by which
breakout actually fired first.

**Fix**: pick deterministically by an explicit rule (earliest trigger, then best R:R), and
make live and backtest use the same selection.

---

## MEDIUM

### M1 — A bar can be both a pivot high and a pivot low, and land inside one pattern
`pivots.py:75-80`, `patterns.py:112`

`find_pivots` emits both kinds for an outside bar that dominates both sides — **78 such
bars** in BTC 1H history. The sort key `(index, kind)` places them adjacent with `"high"`
first, giving a time-degenerate `high,low` step. 3 of the 87 passing H&S geometries are
built on a window where a shoulder and its adjacent trough are the **same bar**. There is
no distinct-index or minimum-bar-separation requirement between window pivots.

**Fix**: require strictly increasing indices (ideally a minimum separation) across the
5-tuple.

### M2 — Live trigger inspects only the last 15m row, with no scheduler backing it
`breakout.py:82-97`, `setup.py:185`, `cli.py:331`

`check_breakout` evaluates only `df.iloc[-1]`. `signals scan` is a manual CLI command —
`poller.py:_CRON_BY_TIMEFRAME` schedules data ingestion only, nothing schedules scanning.
Any invocation more than one 15m bar after the crossing silently drops the breakout, with
no log line saying so. Cause (1) "not detected breakout" in practice, even though the logic
is correct.

**Fix**: either schedule the scan on the 15m boundary, or let `check_breakout` scan the
last *k* bars and return the most recent crossing.

### M3 — No bar-contiguity check on either frame
`setup.py:151-163`, `setup.py:183-185`, `breakout.py:83`

`df_1h.tail(180)` and `df_15m.tail(22)` assume no missing bars. A gap silently distorts
pivot spacing and pattern width, and makes `prev = df.iloc[-2]` a stale reference for the
fresh-cross test — the same slice can then be read as a fresh crossing when hours passed
between the two "adjacent" bars. Currently latent: BTC 1H is exactly gapless (all 31,197
diffs = 3,600,000 ms). Given the recorded `fapi.binance.com` connectivity problem, gaps
are a realistic future state.

**Fix**: assert expected spacing on the loaded slices and log/skip on violation.

### M4 — `_line_value` divides by zero if `TRIANGLE_MIN_PIVOTS_PER_SIDE` is 1
`patterns.py:161-164`

Guarded only incidentally: with the default `2`, `ph[0].index != ph[-1].index`. Setting the
config knob to 1 — a plausible tuning move in Phase 5/6 — makes `x2 == x1` and raises
`ZeroDivisionError` mid-scan.

**Fix**: guard `x2 == x1` in `_line_value`, or floor the config at 2.

### M5 — Flag pole window is 13 bars, not the documented 12
`patterns.py:259`, `config.py:43`

`pole_start = max(0, pole_end - FLAG_POLE_WINDOW_BARS)` and the slice
`lows[pole_start : pole_end + 1]` are both inclusive, giving
`FLAG_POLE_WINDOW_BARS + 1 = 13` bars against a config comment reading "max 12 1H bars for
the impulse pole".

### M6 — Flag containment mixes close and high references (missed patterns)
`patterns.py:264-269`, `patterns.py:283-288`

`pole_height` is measured to `closes[pole_end]`, but containment is tested against
`highs[pole_end]` / `lows[pole_end]`. When the pole's extreme sits on an earlier bar than
`pole_end`, a textbook flag whose consolidation sits comfortably under the true pole high
but above `highs[pole_end]` is rejected outright.

**Fix**: test containment against the pole's extreme over `[pole_start, pole_end]`, the
same window `pole_height` uses.

### M7 — Stop buffer vs risk band silently discards the most decisive breakouts
`setup.py:113-131`, `config.py:55`, `config.py:63`

For a long, `risk_pct ≈ (entry − level)/entry + 0.001`. With `MAX_RISK_PCT = 0.005`, any
trigger bar closing more than ~0.4% beyond the level is rejected. The filter therefore
removes the *strongest* breakout bars and keeps the marginal ones — the opposite of the
intended selection — and does so with no log line. This is arithmetic, not a measurement;
it interacts directly with the cost-ratio tuning already identified as the key lever.

**Fix**: decide the stop from pattern structure (e.g. the opposite pattern extreme or an
ATR multiple) rather than from the level the entry sits fractionally beyond, and log band
rejections.

---

## LOW

- **L1** — No dedup of overlapping candidates. Several overlapping H&S windows, or
  triangle + flag on the same side, can each produce a `Signal` for the same symbol and
  bar in live output (`setup.py:190-199`). Only the backtest collapses them (via H5's
  `break`).
- **L2** — Triangle `breakout_level` is computed at `end_x = n-1` but `end_ts` is
  `df.index[last_pivot_index]` (`patterns.py:194-206`), so the reported pattern window does
  not match the level's anchor point.
- **L3** — No check that H&S shoulders sit above the neckline (`patterns.py:119-124`).
  Latent only: 0 occurrences on BTC 1H, but the invariant is unstated and unenforced.

## Validation

| Check | Result |
|---|---|
| Tests (`pytest -q`) | Pass — 165 passed, 1 skipped |
| Lint | Skipped (no linter configured) |
| Type check | Skipped (no mypy/pyright config) |
| Build | N/A (no build step) |

Note: the suite passes with every finding above present. No test covers same-kind pivot
runs, dual-kind pivot bars, triangle containment, staler-side freshness, or
already-broken-level rejection.

## Files Reviewed

| File | Status |
|---|---|
| `src/trading_bot/signals/pivots.py` | Reviewed — M1 |
| `src/trading_bot/signals/patterns.py` | Reviewed — H1, H2, H3, H4, M1, M4, M5, M6, L2, L3 |
| `src/trading_bot/signals/breakout.py` | Reviewed — H4, M2, M3 |
| `src/trading_bot/signals/setup.py` | Reviewed — M2, M3, M7, L1 |
| `src/trading_bot/backtest/engine.py` | Reviewed (correlated) — H5 |
| `src/trading_bot/config.py` | Reviewed (thresholds) — M4, M5, M7 |
