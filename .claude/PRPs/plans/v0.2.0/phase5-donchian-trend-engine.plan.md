# Plan: Donchian Trend Engine — A-core (PRD Phase 5)

## Summary
Replace chart-pattern geometry (triangle / flag / head-and-shoulders) with a canonical Donchian channel breakout as the trending-regime signal engine. Two new modules — `indicators/donchian.py` (trailing-only upper/lower/mid over N bars) and `signals/donchian.py` (candidate producer + live scanner) — plus a dispatch swap in `signals/scan.py` and `backtest/engine.py`, and one genuinely new capability in the backtest engine: a ratcheting ATR trail and an opposite-channel exit, neither of which the engine supports today. `signals/patterns.py` and `signals/pivots.py` are retired from the active dispatch path but **not deleted** (PRD: "Retired, not paused … leave the active path" — the modules and their tests stay green as reference).

## User Story
As the bot's sole user, I want the trending-regime signals to come from the most-replicated directional edge in systematic trading (Donchian channel breakout, canonical 20/55, Turtle lineage) instead of chart-pattern geometry that has no peer-reviewed net-of-cost support and lost money on all three symbols, so that when the Phase 7 gate delivers its verdict it is a verdict on a strategy family worth believing.

## Problem → Solution
**Current**: `signals/setup.scan_breakout_signals` calls `find_pivots` + `detect_patterns` on the setup tier, producing `PatternCandidate`s from ~14 fitted geometry parameters (`HS_SHOULDER_TOLERANCE`, `TRIANGLE_MIN_CONVERGENCE`, `FLAG_POLE_MIN_PCT`, …). Triangle + flag were ~98% of trade volume; both lose on all 3 symbols. The engine's only exit modes are fixed stop, fixed target, time-stop and end-of-data.
**Solution**: A `PatternCandidate` whose `breakout_level` is the trailing 20-bar Donchian extreme on the **4H setup tier**, gated by ADX(14) > 25 on the same tier and by price on the correct side of the 55-bar mid-line, with `target_height` = the 20-bar channel width (a derived quantity, not a fitted one). Exit adds an opposite-channel touch and a 1.5×ATR ratchet trail, both reusing constants that are already canonical or frozen. Zero new fitted parameters.

## Metadata
- **Complexity**: Medium-High (9 files touched, ~520 net new/changed lines including tests; no new dependencies; one new engine capability — mutable per-bar stop — that touches the no-lookahead invariant)
- **Source PRD**: `.claude/PRPs/prds/hybrid-trend-voltarget.prd.md`
- **PRD Phase**: Phase 5 — Donchian trend engine (A-core). Parallel with Phase 6. Depends on Phase 4 (tiers) and, transitively, Phase 2 (ATR risk model).
- **Estimated Files**: 9 (3 new, 6 modified)

---

## Stated Assumptions (resolving PRD Open Questions)

The PRD leaves two things unpinned. Both are resolved here, deliberately and once.

**A1 — Channels compute on the 4H setup tier**, i.e. `config.SIGNAL_PATTERN_TIMEFRAME` once Phase 4 sets it to `4h`. (PRD Open Question: *"on which timeframe do the channels compute, 4H setup bars or 1D?"*) Reasoning: (a) 4H is the tier the Phase 4 map assigns to *setup* structure, and the Donchian channel *is* the setup structure; (b) the `c ≤ 0.10` gate is only satisfied there — PRD Open Question #2 measures `c` = 7.1/5.2/3.7% (BTC/ETH/SOL) all-taker at `k = 1.5 × ATR(4H)` versus BTC failing at 14.9% on ATR(1H). Since the stop is `1.5 × ATR(setup TF)` and the target derives from the channel width, computing the channel on a different tier than ATR would put stop and target on different volatility scales; (c) 55 4H bars ≈ 9.2 days of warmup versus 55 days on 1D — and 1D bars **do not exist in the DB yet**. Never re-decided or swept.

**A2 — "20/55 entry/exit channels" is read as: 20 = entry *and* exit channel, 55 = trend filter.** Entry level = trailing 20-bar channel extreme; opposite-channel exit = trailing 20-bar extreme on the other side; trend filter = the 55-bar mid-line. Reasoning: the alternative Turtle-System-2 reading (55 entry, 20 exit) makes the trend filter degenerate — for a long, the 55-mid is *always* below a fresh 55-bar high, so "price above the 55-mid" is tautologically true and carries zero information. Under A2 both canonical numbers do non-redundant work and no third lookback is invented. **The alternative reading is pre-registered as the one legitimate Phase-7 grid axis** (PRD Phase 7: "Donchian lookback pair if genuinely uncertain") — it is *not* explored here.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/trading_bot/signals/patterns.py` | 58-77 | The `PatternCandidate` dataclass your Donchian candidate must reproduce **exactly**: `kind`, `direction`, `breakout_level`, `target_height`, `start_ts`, `end_ts`. Six fields, all required, frozen |
| P0 | `src/trading_bot/signals/breakout.py` | 60-149 | `check_breakout` — the trigger interface you must satisfy. Reads only `candidate.breakout_level`, `candidate.direction`, `candidate.end_ts`. Note the fresh-crossing test (124-129), the `ts < candidate.end_ts` skip (113-114), the `interval_ms` contiguity check (115-122), and the volume grading (131-138) |
| P0 | `src/trading_bot/indicators/bollinger.py` | 1-52 | The single-indicator-module exemplar: module docstring stating the math and the NaN rule, one public function, keyword-only `| None = None` params falling back to config, returns a `pd.DataFrame` indexed like the input with named columns. Mirror this shape file-for-file in `donchian.py` |
| P0 | `src/trading_bot/indicators/wilder.py` | 95-109, 207-250 | `atr(df, period=14)` and `adx(df, period=14)` — the exact API. Both return `pd.Series` indexed like `df` with **leading NaNs preserved, never filled**. ATR's first defined value is at index `period-1`; ADX's at `2*period-2` (27 bars for period 14) |
| P0 | `src/trading_bot/backtest/engine.py` | 150-260 | The replay loop. `candidates_for(h_idx)` at 157-171 is the dispatch site (line 164 is the `detect_patterns` call to replace). Exits at 207-223 — **this is the code that must learn a mutable stop**. `close_out` at 176-198 builds the frozen `Trade` |
| P0 | `.claude/PRPs/plans/v0.2.0/phase2-honest-cost-and-risk-model.plan.md` | Tasks 3, 7 | Phase 2 rewrites `build_signal` to `build_signal(symbol, candidate, event, atr_value, *, atr_multiple=None, rr_floor=None)` with `atr_value` a **required positional**. Your candidates flow into *that* signature, not today's |
| P1 | `src/trading_bot/signals/setup.py` | 189-293 | `_contiguous_tail` (189-214) and `_load_df` (217-230) are reused verbatim by the new scanner. `rank_signals` (233-242). `scan_breakout_signals` (245-293) is the function whose *role* you replace — lines 281-282 (`find_pivots` / `detect_patterns`) are the geometry that leaves the path |
| P1 | `src/trading_bot/signals/scan.py` | 44-54 | The dispatch table proper. `regime_label == "trending"` → `scan_breakout_signals` becomes → `scan_donchian_signals` |
| P1 | `src/trading_bot/signals/meanrev.py` | 156-170, 244-273 | Parallel structure to copy: `_to_trigger_candidate` shows how a non-geometry setup is adapted into a `PatternCandidate` purely to reuse `check_breakout`; `scan_fade_signals` shows the scanner shape |
| P1 | `tests/test_signals.py` | 22-40, 316-333, 500-547 | Test idioms: `make_df(rows, start, interval)`, `path_df(anchors)`, `make_candidate()`, `breakout_df(prev_close, last_close)`, `seed_candles(conn, symbol, timeframe, df)`, and the `monkeypatch.setattr(setup_mod, "current_regime", ...)` pattern |
| P1 | `tests/test_backtest.py` | 60-90, 92-135 | `seed(conn, timeframe, rows, start, interval)`, `seed_scenario`, `patch_trending(monkeypatch)` — reuse these for the new exit tests; do not invent new fixture shapes |
| P2 | `tests/test_wilder.py` | 285-323 | The leading-NaN / warmup assertion idiom (`TestLeadingNaNs`) your `donchian.py` tests must mirror |
| P2 | `tests/test_meanrev.py` | 232-258 | `TestScanSymbol` patches `scan.scan_breakout_signals` by name — this breaks when the dispatch swaps and must be updated |
| P2 | `src/trading_bot/config.py` | 18-32, 87-100 | Constant-grouping-by-phase-comment convention; `SIGNAL_PATTERN_TIMEFRAME`/`SIGNAL_TRIGGER_TIMEFRAME` (28-29); `ADX_PERIOD`/`ADX_TREND_THRESHOLD` (20-21) which you **reuse rather than duplicate** |
| P2 | `src/trading_bot/backtest/metrics.py` | 15-34 | Buckets key on `f"{regime}/{pattern}"`, so `candidate.kind` becomes a metrics bucket name. `outcome` is not read by metrics — adding new outcome values is safe |

## External Documentation

No external research needed. Donchian channels (Richard Donchian, 1960s; Turtle system 1983) are a rolling max/min — three lines of pandas. The 20/55 pair is canonical, sourced in the PRD from the Turtle lineage and [arXiv 2009.12155](https://arxiv.org/pdf/2009.12155) / [SSRN 5209907](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907). ADX > 25 and ATR are already implemented and measured-correct in `indicators/wilder.py`.

---

## Dependencies on Other Phases

| Dependency | Artifact | What breaks without it |
|---|---|---|
| **Phase 2** (in-progress) | `build_signal(symbol, candidate, event, atr_value, *, atr_multiple=None, rr_floor=None)` — `atr_value` required positional; `MAX_RISK_PCT`/`MIN_REWARD_PCT`/`BREAKOUT_STOP_BUFFER_PCT` gone from stop logic | Under *today's* `build_signal` a Donchian candidate is rejected almost always: the level-anchored stop is `level*(1-0.001)` while the entry sits a full 1H bar's move past a 20-bar channel extreme, so `risk_pct` blows past `MAX_RISK_PCT = 0.005` immediately. Donchian is **not testable** against the pre-Phase-2 filter |
| **Phase 2** | `config.ATR_STOP_MULTIPLE = 1.5`, `config.ATR_STOP_PERIOD = 14`, `config.RR_FLOOR = 1.5` | The ATR trail reuses `ATR_STOP_MULTIPLE` verbatim so the trail introduces no new parameter. If Phase 2 named it differently, use its name — do **not** add a second multiple |
| **Phase 2** | `risk/atr_stop.compute_atr_stop` | Not called directly here (`build_signal` calls it), but the trail arithmetic must match its sign convention: long stop below entry, short above |
| **Phase 2** | Engine computes an ATR series on the setup tier and threads `atr_value` into `build_signal` (Phase 2 Task 7) | You reuse that exact series for the trail. If Phase 2 computed ATR on the 1H df, Phase 4 moves it to 4H; assert it is the setup tier before wiring the trail |
| **Phase 4** | `config.SIGNAL_PATTERN_TIMEFRAME == "4h"`, `SIGNAL_TRIGGER_TIMEFRAME == "1h"`, `REGIME_TIMEFRAME == "1d"`; `MAX_HOLD_BARS_15M` renamed/rescaled | A1 assumes the setup tier *is* 4H. If Phase 4 has not landed, `donchian.py` still computes on `SIGNAL_PATTERN_TIMEFRAME` correctly but the channel would be a 20-**hour** channel and the `c ≤ 0.10` argument does not hold |
| **Phase 4** | Engine locals renamed away from `df_15m`/`m15`/`close_15m`/`window_15m` | Task 6 edits those lines. Read the current names in `engine.py` before editing — this plan cites the *pre-Phase-4* names (`engine.py:139-141, 236, 242`) because Phase 4's plan does not exist yet |
| **Phase 1** (pending, externally blocked) | 1D bars in `storage.TIMEFRAME_MS` + DB | Only matters via Phase 4's regime tier. **Confirmed as of this writing: `data/ohlcv.db` holds only `15m`/`1h`/`4h`** (7,799 4H bars per symbol, 2023-01-01 → 2026-07). Nothing in Phase 5 needs 1D directly |

---

## Patterns to Mirror

### SINGLE_INDICATOR_MODULE
```python
# SOURCE: indicators/bollinger.py:19-52 — one public function, keyword-only
# optional params with config fallback, returns a DataFrame indexed like df.
def bollinger(
    df: pd.DataFrame, *, period: int | None = None, num_std: float | None = None
) -> pd.DataFrame:
    if period is None:
        period = config.BB_PERIOD
    close = df["close"]
    middle = close.rolling(period, min_periods=period).mean()
    return pd.DataFrame({"middle": middle, "upper": ..., "lower": ...}, index=df.index)
```

### TRAILING_WINDOW_NO_LOOKAHEAD
```python
# SOURCE: regime/classifier.py:36-53 — rolling(window, min_periods=window),
# NaN during warmup, never filled, window ENDS at the current bar.
def _atr_percentile_rank(atr_vals: pd.Series, window: int) -> pd.Series:
    return atr_vals.rolling(window, min_periods=window).rank(pct=True)

# SOURCE: indicators/wilder.py:14-18 (docstring contract) — "All outputs are
# pd.Series indexed identically, with leading NaNs preserved (never filled)."
```

### CANDIDATE_ADAPTER_FOR_REUSING_check_breakout
```python
# SOURCE: signals/meanrev.py:156-170 — a non-geometry setup is wrapped in a
# PatternCandidate purely so check_breakout's crossing/freshness/volume logic
# can be reused unchanged. This is exactly what donchian.py does.
def _to_trigger_candidate(candidate: FadeCandidate) -> PatternCandidate:
    return PatternCandidate(
        kind=FADE_KIND,
        direction=candidate.direction,
        breakout_level=candidate.trigger_level,
        target_height=abs(candidate.target - candidate.trigger_level),
        start_ts=candidate.start_ts,
        end_ts=candidate.end_ts,
    )
```

### SCANNER_SHAPE
`signals/setup.py:245-293` — load both tiers via `_load_df`, bail on empty, `tail()` each to its required window, warn on staleness, loop candidates → `check_breakout` → `build_signal`, `return rank_signals(...)`. Task 4 reproduces it line-for-line, so it is not duplicated here.

### ENGINE_CANDIDATE_DISPATCH
```python
# SOURCE: backtest/engine.py:157-171 — lazily cached per setup-tier bar, keyed
# by bar index, returning (regime, [(method, candidate), ...]).
def candidates_for(h_idx: int) -> tuple[str, list]:
    if h_idx not in cand_cache:
        t = int(close_1h[h_idx])
        reg = regime_at(t)
        window = df_1h.iloc[max(0, h_idx + 1 - config.PATTERN_LOOKBACK_BARS) : h_idx + 1]
        cands: list = []
        if reg == "trending":
            cands = [("breakout", c) for c in detect_patterns(window)]   # <-- line 164
        elif reg == "ranging":
            cands = [("fade", c) for c in detect_fade_setups(window, num_std=params.bb_num_std)]
        cand_cache[h_idx] = (reg, cands)
    return cand_cache[h_idx]
```

### ENGINE_EXIT_LOOP (the code that gains a mutable stop)
```python
# SOURCE: backtest/engine.py:207-223 — the ONLY exit modes today. s.stop and
# s.target are read off the FROZEN Signal on every bar; there is no per-bar
# mutation, and nothing consults the setup tier during a trade.
if open_trade is not None:
    if j <= open_trade["entry_j"]:
        continue
    s = open_trade["signal"]
    if s.direction == "long":
        if lows[j] <= s.stop:
            close_out(j, s.stop, "stop")  # conservative: stop first
        elif highs[j] >= s.target:
            close_out(j, s.target, "target")
    else:
        ...  # mirrored
    if open_trade is not None and j - open_trade["entry_j"] >= max_hold:
        close_out(j, float(closes[j]), "time")
    continue
```

### TEST_FIXTURE_BUILDERS
```python
# SOURCE: tests/test_signals.py:22-28 and tests/test_backtest.py:60-63
def make_df(rows, start=START, interval=H1):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype(int)
    return df.set_index("ts")

def seed(conn, timeframe, rows, start=START, interval=H1):
    data = [[start + i * interval] + list(r) for i, r in enumerate(rows)]
    storage.upsert_candles(conn, SYMBOL, timeframe, data)
    return data
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/indicators/donchian.py` | CREATE | Trailing-only channel: `donchian(df, *, period) -> DataFrame[upper, lower, mid]`. Pure, no I/O, mirrors `bollinger.py` |
| `src/trading_bot/signals/donchian.py` | CREATE | `DONCHIAN_KIND`, `detect_donchian_setups(df, ...) -> list[PatternCandidate]`, `scan_donchian_signals(conn, symbol, now_ms) -> list[Signal]`, `channel_exit_levels(df, period)` helper reused by the engine |
| `src/trading_bot/config.py` | UPDATE | Add `DONCHIAN_ENTRY_PERIOD = 20`, `DONCHIAN_TREND_PERIOD = 55`, `DONCHIAN_MIN_BARS`. **No new ADX / ATR / multiple constants** — reuse `ADX_PERIOD`, `ADX_TREND_THRESHOLD`, `ATR_STOP_PERIOD`, `ATR_STOP_MULTIPLE` |
| `src/trading_bot/signals/scan.py` | UPDATE | Dispatch swap at 50-51: trending → `scan_donchian_signals`. Docstring updated (line 7 names the retired method) |
| `src/trading_bot/signals/setup.py` | UPDATE | Docstring banners marking `scan_breakout_signals` and `current_signals` RETIRED / off the active dispatch path. `build_signal`, `rank_signals`, `_load_df`, `_contiguous_tail`, `Signal` all stay and are reused |
| `src/trading_bot/backtest/engine.py` | UPDATE | Replace `detect_patterns` at line 164 with `detect_donchian_setups`; add the mutable-stop ATR trail and the opposite-channel exit to the exit loop; two new `outcome` values |
| `tests/test_donchian.py` | CREATE | Indicator tests (trailing-only / no-lookahead / warmup NaN) + candidate tests (ADX gate, mid-line filter, level/width, `end_ts`) + scanner integration test |
| `tests/test_backtest.py` | UPDATE | New tests for the trail exit, the channel exit, and their interaction with the conservative same-bar rule; existing flag-based scenarios re-pointed or explicitly patched |
| `tests/test_meanrev.py` | UPDATE | `TestScanSymbol._patch` (232-242) patches `scan.scan_breakout_signals` by name — rename to `scan_donchian_signals` |

## NOT Building

- **Walk-forward repair and THE GATE (Phase 7).** Do not run `python -m trading_bot.cli walkforward`, do not touch `walkforward.py`, do not add `DONCHIAN_ENTRY_PERIOD`/`DONCHIAN_TREND_PERIOD` to `DEFAULT_GRID` (`walkforward.py:36-40`). Running the gate here spends the one-shot OOS holdout.
- **Fade re-qualification (Phase 6).** `meanrev.py` logic is untouched; only the one monkeypatch name in `tests/test_meanrev.py` changes. Phases 5 and 6 collide *only* in `scan.py`'s dispatch — merge that file carefully.
- **Sweeping anything.** 20, 55, `k = 1.5`, ADX > 25 are canonical or Phase-2-derived and **frozen**. No `DONCHIAN_*` constant may be added to any grid in this phase. Do not "check whether 30/70 works better" — that is a consumed degree of freedom.
- **Donchian lookback ensembles (20/55/100) and universe expansion** — Phase 10 / Option B.
- **Volatility-target sizing (Phase 8)** — no sizing fields on `Signal`.
- **Alert delivery / Discord (Phase 9)** — the exit rule is enforced by the engine only; nothing publishes "your trail is now at X".
- **Sharpe / Sortino / DSR metrics (Phase 3)** — `metrics.py` untouched.
- **Deleting `patterns.py` / `pivots.py` or their tests.** PRD: retired from the active path, not removed. `tests/test_signals.py`'s pivot/pattern/`TestBuildSignal` classes stay green.
- **1D bars, backfill, poller, storage, classifier, `binance_client`, `wilder.py`, `bollinger.py`** — no changes.
- **A new exit-resolution timeframe (1m).** The trail and channel exits are evaluated on trigger-tier bars with the existing conservative same-bar rule.

---

## Step-by-Step Tasks

### Task 1: Config constants
- **ACTION**: Edit `src/trading_bot/config.py`, appending a new phase-banner block after the Phase 4 mean-reversion block (currently ends line 91).
- **IMPLEMENT**:
  ```python
  # Phase 5: Donchian trend engine (A-core), trending regime only.
  # 20/55 are CANONICAL (Donchian/Turtle lineage), not fitted here, and are
  # frozen: they must never appear in a walk-forward grid in this phase. See
  # the plan's Stated Assumptions A1/A2 — channels compute on the SETUP tier
  # (config.SIGNAL_PATTERN_TIMEFRAME), the 20-bar channel supplies both the
  # entry level and the opposite-channel exit, and the 55-bar MID-LINE is the
  # trend filter (nothing else uses 55).
  DONCHIAN_ENTRY_PERIOD = 20   # bars in the entry / opposite-exit channel
  DONCHIAN_TREND_PERIOD = 55   # bars in the mid-line trend filter
  # Warmup: the channel needs PERIOD prior bars (trailing, current bar excluded)
  # and ADX(14) needs 2*14-1 = 27 bars. 55 dominates. On 4H bars that is ~9.2 days.
  DONCHIAN_MIN_BARS = max(DONCHIAN_TREND_PERIOD + 1, 2 * ADX_PERIOD - 1)
  ```
  Add **nothing else**. The ADX gate reuses `ADX_PERIOD` (20) and `ADX_TREND_THRESHOLD` (21); the ATR trail reuses `ATR_STOP_PERIOD` / `ATR_STOP_MULTIPLE` introduced by Phase 2.
- **MIRROR**: `config.py:26-32` and `87-91` — banner comment naming the phase and the regime it serves, then constants with inline `#` rationale.
- **IMPORTS**: None (same file). `DONCHIAN_MIN_BARS` references `ADX_PERIOD` defined at line 20, so the block must sit *after* it.
- **GOTCHA**: `PATTERN_LOOKBACK_BARS = 180` (line 31) is the setup-tier window the engine and scanner slice to. 180 ≥ 56, so no lookback change is needed — but do **not** reduce it: the engine's `candidates_for` window (`engine.py:161`) uses it and a shorter window would silently NaN the 55-channel on early bars.
- **VALIDATE**: `.venv/bin/python -c "from trading_bot import config; print(config.DONCHIAN_ENTRY_PERIOD, config.DONCHIAN_TREND_PERIOD, config.DONCHIAN_MIN_BARS)"` → `20 55 56`.

### Task 2: `indicators/donchian.py`
- **ACTION**: Create `src/trading_bot/indicators/donchian.py`.
- **IMPLEMENT**:
  ```python
  """
  Donchian channels, pure pandas.

  Upper band is the highest high and lower band the lowest low over the
  `period` bars PRECEDING each bar; mid is their average. The current bar is
  deliberately EXCLUDED from its own channel: a channel including the current
  bar's high can never be closed above (max >= close by construction), so a
  self-inclusive channel makes breakout detection impossible while still
  reading as a working indicator. NaN until `period` prior bars exist.

  Trailing-only, no lookahead: the value at bar i depends on bars
  [i-period, i-1] and nothing later, so a series computed over full history is
  identical bar-for-bar to one computed incrementally as bars close.
  """

  import logging

  import pandas as pd

  from trading_bot import config

  logger = logging.getLogger("trading_bot")


  def donchian(df: pd.DataFrame, *, period: int | None = None) -> pd.DataFrame:
      """
      Compute trailing Donchian channels over an OHLCV DataFrame.

      Args:
          df: DataFrame with high/low columns, indexed by epoch-ms ts, ascending.
          period: Channel lookback in bars, EXCLUDING the current bar
              (default config.DONCHIAN_ENTRY_PERIOD).

      Returns:
          DataFrame indexed like df with columns upper, lower, mid. All three
          are NaN for the first `period` bars (period prior bars are required,
          so the first defined value is at positional index `period`).
      """
      if period is None:
          period = config.DONCHIAN_ENTRY_PERIOD

      upper = df["high"].rolling(period, min_periods=period).max().shift(1)
      lower = df["low"].rolling(period, min_periods=period).min().shift(1)
      return pd.DataFrame(
          {"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0},
          index=df.index,
      )
  ```
- **MIRROR**: `indicators/bollinger.py:19-52` exactly — same signature shape, same config fallback, same DataFrame return, same NaN-during-warmup docstring sentence.
- **IMPORTS**: `logging`, `pandas as pd`, `from trading_bot import config`.
- **GOTCHA**: The `.shift(1)` is the whole correctness argument of this module. `rolling(period).max()` alone includes the current bar. Without the shift, `close > upper` is unsatisfiable for a long and the engine would report zero trades while every unit test on the raw max still passed. Test this explicitly (Task 7). Second gotcha: `min_periods=period` **before** the shift means the first defined value lands at positional index `period`, not `period-1` — one bar later than Bollinger's convention. Say so in the docstring (done above) so nobody "fixes" it.
- **VALIDATE**: `pytest tests/test_donchian.py::TestDonchianIndicator -v`.

### Task 3: `signals/donchian.py` — candidate detection
- **ACTION**: Create `src/trading_bot/signals/donchian.py` with the module docstring, `DONCHIAN_KIND`, `detect_donchian_setups`, and `channel_exit_levels`.
- **IMPLEMENT**:
  ```python
  """
  Donchian channel breakout signal method (Phase 5, trending regime only).

  Replaces the retired Phase 3 chart-pattern geometry. Structural mirror of
  meanrev.py: the setup is detected on the SETUP tier and wrapped in a
  PatternCandidate so breakout.check_breakout supplies the trigger-tier
  crossing/freshness/volume logic unchanged.

    1. Regime gate — "trending" only (the dispatcher applies it).
    2. Setup on config.SIGNAL_PATTERN_TIMEFRAME (4H after the Phase 4 tier
       shift): the trailing DONCHIAN_ENTRY_PERIOD (20) channel extreme is the
       level a trigger bar must close beyond, with two canonical confirmations
       on the same tier — ADX(ADX_PERIOD) >= ADX_TREND_THRESHOLD (25), the same
       Wilder ADX the 1D regime classifier uses applied to the finer tier; and
       the latest close on the correct side of the DONCHIAN_TREND_PERIOD (55)
       MID-LINE (long above, short below).
    3. Trigger — check_breakout on the trigger tier.
    4. SL/TP — delegated wholly to setup.build_signal: stop =
       ATR_STOP_MULTIPLE * ATR(setup TF) (Phase 2); target = level +/-
       target_height, where target_height is the 20-bar CHANNEL WIDTH
       (upper - lower) — a derived quantity, the range the market has just
       resolved, adding no parameter beyond the canonical 20.
    5. R:R ratio floor — build_signal's rr >= RR_FLOOR (Phase 2).

  Exits (opposite-channel touch, ATR ratchet trail) are not expressible in the
  frozen Signal dataclass and are enforced by the backtest engine; see
  channel_exit_levels() and backtest/engine.py.

  No geometry parameters exist here. 20 and 55 are canonical and frozen; ADX 25
  and the ATR multiple are reused from existing config, never redeclared.
  """

  import logging

  import pandas as pd

  from trading_bot import config
  from trading_bot.data import storage
  from trading_bot.indicators.donchian import donchian
  from trading_bot.indicators.wilder import adx as wilder_adx
  from trading_bot.indicators.wilder import atr as wilder_atr
  from trading_bot.signals.breakout import check_breakout
  from trading_bot.signals.patterns import PatternCandidate
  from trading_bot.signals.setup import Signal, _load_df, build_signal, rank_signals

  logger = logging.getLogger("trading_bot")

  DONCHIAN_KIND = "donchian-breakout"


  def detect_donchian_setups(
      df: pd.DataFrame,
      *,
      entry_period: int | None = None,
      trend_period: int | None = None,
      adx_period: int | None = None,
      adx_min: float | None = None,
  ) -> list[PatternCandidate]:
      """
      Emit at most one Donchian breakout candidate for the latest closed bar.

      Args:
          df: Setup-tier OHLCV DataFrame, CLOSED bars only, ascending epoch-ms
              index. Callers pass the last config.PATTERN_LOOKBACK_BARS bars.
          entry_period / trend_period / adx_period / adx_min: defaults
              config.DONCHIAN_ENTRY_PERIOD / DONCHIAN_TREND_PERIOD /
              ADX_PERIOD / ADX_TREND_THRESHOLD.

      Returns:
          A list of 0 or 1 PatternCandidate. Long and short are mutually
          exclusive here (a close cannot sit both above and below the 55-mid),
          unlike the triangle detector which emitted both sides.
      """
      if entry_period is None:
          entry_period = config.DONCHIAN_ENTRY_PERIOD
      if trend_period is None:
          trend_period = config.DONCHIAN_TREND_PERIOD
      if adx_period is None:
          adx_period = config.ADX_PERIOD
      if adx_min is None:
          adx_min = config.ADX_TREND_THRESHOLD

      n = len(df)
      if n < max(trend_period + 1, 2 * adx_period - 1):
          return []

      entry_ch = donchian(df, period=entry_period)
      trend_ch = donchian(df, period=trend_period)
      adx_vals = wilder_adx(df, period=adx_period)

      i = n - 1  # latest CLOSED setup bar
      upper = float(entry_ch["upper"].iloc[i])
      lower = float(entry_ch["lower"].iloc[i])
      mid = float(trend_ch["mid"].iloc[i])
      adx_now = float(adx_vals.iloc[i])
      close = float(df["close"].to_numpy()[i])

      if pd.isna(upper) or pd.isna(lower) or pd.isna(mid) or pd.isna(adx_now):
          return []  # warmup
      if adx_now < adx_min:
          return []  # trend not confirmed on the setup tier
      width = upper - lower
      if width <= 0:
          return []  # degenerate channel

      if close > mid:
          direction, level = "long", upper
      elif close < mid:
          direction, level = "short", lower
      else:
          return []  # exactly on the mid-line: no side

      interval = int(df.index.to_numpy()[i]) - int(df.index.to_numpy()[i - 1])
      return [
          PatternCandidate(
              kind=DONCHIAN_KIND,
              direction=direction,
              breakout_level=level,
              target_height=width,
              start_ts=int(df.index.to_numpy()[max(0, i - entry_period)]),
              # end_ts is the setup bar's CLOSE time, so check_breakout's
              # `ts < candidate.end_ts` skip guarantees the trigger bar opens
              # at or after the setup bar closed — no intra-bar lookahead.
              end_ts=int(df.index.to_numpy()[i]) + interval,
          )
      ]


  def channel_exit_levels(
      df: pd.DataFrame, *, period: int | None = None
  ) -> pd.DataFrame:
      """Trailing opposite-channel exit levels for the setup tier.

      A long exits when price touches the trailing `period`-bar LOW; a short
      when it touches the trailing `period`-bar HIGH. Identical to the entry
      channel by construction (same period, other side) — exposed as its own
      function so the backtest engine does not import the indicator directly
      and the "same 20 bars, other side" contract is stated in one place.

      Returns:
          DataFrame indexed like df with columns upper, lower, mid (NaN during
          the first `period` bars).
      """
      return donchian(df, period=period or config.DONCHIAN_ENTRY_PERIOD)
  ```
- **MIRROR**: `meanrev.detect_fade_setups` (`meanrev.py:62-153`) — same optional-override header, same `n < period` early return, same "at most one per direction" contract, same `float(...)` coercion off numpy.
- **IMPORTS**: as listed. `_load_df` is a deliberate private-name import from `setup.py` — `meanrev.py:32` already does exactly this, so it is the established convention, not a new violation.
- **GOTCHA**: (a) `end_ts` must be the setup bar's **close** time, not its index (open) time. With `end_ts = index[i]`, a trigger bar *inside* the setup bar would pass the freshness test and enter on information the setup bar had not yet published. (b) `adx_min` reuses `ADX_TREND_THRESHOLD` — the same numeric threshold the 1D regime classifier uses, applied here to setup-tier bars. That is deliberate reuse of a canonical value, **not** a second copy of a tunable: do not add `DONCHIAN_ADX_MIN`. (c) Long and short are mutually exclusive; do not "emit both and let the trigger decide" as `_detect_triangles` does (`patterns.py:350-369`) — the mid-line filter is the whole point. (d) `df.index.to_numpy()` twice for `interval` assumes ≥2 bars; the `n < trend_period + 1` guard already ensures ≥56.

### Task 4: `signals/donchian.py` — live scanner
- **ACTION**: Append `scan_donchian_signals` to `src/trading_bot/signals/donchian.py`.
- **IMPLEMENT**:
  ```python
  def scan_donchian_signals(conn, symbol: str, now_ms: int) -> list[Signal]:
      """
      Donchian-scan one symbol WITHOUT a regime check (dispatcher applies it).

      Args:
          conn: Database connection.
          symbol: Trading pair symbol.
          now_ms: Evaluation time in epoch milliseconds.

      Returns:
          Screened Signals, best-R:R first (0 or 1 in practice, since
          detect_donchian_setups emits at most one candidate).
      """
      df_setup = _load_df(conn, symbol, config.SIGNAL_PATTERN_TIMEFRAME, now_ms)
      df_trig = _load_df(conn, symbol, config.SIGNAL_TRIGGER_TIMEFRAME, now_ms)
      if df_setup.empty or df_trig.empty:
          return []

      df_setup = df_setup.tail(config.PATTERN_LOOKBACK_BARS)
      trigger_bars = config.VOLUME_LOOKBACK + 1 + max(1, config.BREAKOUT_TRIGGER_LOOKBACK_BARS)
      df_trig = df_trig.tail(trigger_bars)

      interval_trig = storage.TIMEFRAME_MS[config.SIGNAL_TRIGGER_TIMEFRAME]
      latest_close = int(df_trig.index[-1]) + interval_trig
      if now_ms - latest_close >= interval_trig:
          # Copy setup.py:272-279's staleness warning verbatim, with the
          # timeframe name substituted for the hardcoded "15m".
          logger.warning(
              "%s %s data is %d ms behind now_ms; a breakout may already be older "
              "than the %d-bar trigger window",
              symbol, config.SIGNAL_TRIGGER_TIMEFRAME, now_ms - latest_close,
              config.BREAKOUT_TRIGGER_LOOKBACK_BARS,
          )

      atr_series = wilder_atr(df_setup, period=config.ATR_STOP_PERIOD)
      atr_value = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
      if not (atr_value > 0):  # covers NaN (warmup) and zero/negative
          logger.warning("%s: ATR undefined on %s bars; no signals", symbol,
                         config.SIGNAL_PATTERN_TIMEFRAME)
          return []

      signals: list[Signal] = []
      for candidate in detect_donchian_setups(df_setup):
          event = check_breakout(df_trig, candidate, interval_ms=interval_trig)
          if event is None:
              continue
          signal = build_signal(symbol, candidate, event, atr_value)
          if signal is not None:
              signals.append(signal)

      return rank_signals(signals)
  ```
- **MIRROR**: `setup.scan_breakout_signals` (`setup.py:245-293`) line-for-line, including the staleness warning, plus the ATR block Phase 2 Task 4 adds to that same function.
- **IMPORTS**: already added in Task 3.
- **GOTCHA**: `build_signal(symbol, candidate, event, atr_value)` — `atr_value` is a **positional** argument in the post-Phase-2 signature. If Phase 2 has not landed, this call is a `TypeError`; that is the intended hard failure, not something to paper over with a default. Do not pass `atr_multiple`/`rr_floor`: they must come from frozen config.
- **VALIDATE**: `pytest tests/test_donchian.py::TestScanDonchianSignals -v`.

### Task 5: Dispatch swap and retirement banners
- **ACTION**: Edit `src/trading_bot/signals/scan.py` (lines 7, 20-21, 50-51) and add retirement docstring banners in `src/trading_bot/signals/setup.py`.
- **IMPLEMENT**: In `scan.py`, change the docstring line 7 from `trending -> chart-pattern breakout method (Phase 3, setup.py)` to `trending -> Donchian channel breakout (Phase 5, donchian.py)`; swap the import
  ```python
  from trading_bot.signals.donchian import scan_donchian_signals
  from trading_bot.signals.setup import Signal
  ```
  and the branch
  ```python
      if regime_label == "trending":
          return (regime_label, scan_donchian_signals(conn, symbol, now_ms))
  ```
  In `setup.py`, prepend to the docstrings of `scan_breakout_signals` (245) and `current_signals` (296):
  ```
  RETIRED (PRD Phase 5): chart-pattern geometry has left the active dispatch
  path — signals.scan now routes "trending" to signals.donchian. This function
  and signals/patterns.py + signals/pivots.py are kept, unreferenced by the
  dispatcher, as the reference implementation behind the retirement decision;
  they are NOT deleted and their tests stay green. Do not re-register them.
  ```
  `build_signal`, `rank_signals`, `Signal`, `_load_df`, `_contiguous_tail` remain fully live and are imported by `donchian.py`.
- **MIRROR**: `scan.py`'s existing four-line regime table docstring.
- **IMPORTS**: `scan.py` no longer needs `scan_breakout_signals`; it still needs `Signal` for the return annotation.
- **GOTCHA**: (a) This is the **one file Phase 6 also edits**. Coordinate the merge: Phase 6 touches the `ranging` branch, Phase 5 the `trending` branch. (b) The PRD points at `setup.py:281-293` as a dispatch site — those lines are `find_pivots` / `detect_patterns` / the candidate loop inside `scan_breakout_signals`. They are honored by taking the *whole function* off the active path rather than editing geometry out of it in place; the new scanner is a sibling module, which keeps the retired implementation readable and the diff reviewable. (c) `cli.py:307` has a stale docstring mentioning "pattern breakout" — update that string too (one line) so the CLI help does not lie.
- **VALIDATE**: `grep -rn "detect_patterns\|find_pivots" src/trading_bot/signals/scan.py src/trading_bot/backtest/engine.py` → no matches after Task 6.

### Task 6: Backtest engine — dispatch, ATR trail, opposite-channel exit
- **ACTION**: Edit `src/trading_bot/backtest/engine.py`: imports (39-46), the module docstring's "Simulation rules" block (18-27), `candidates_for` (157-171), the entry site (239-255), and the exit loop (207-223).
- **IMPLEMENT**:
  1. **Imports**: drop `from trading_bot.signals.patterns import detect_patterns`; add
     ```python
     from trading_bot.signals.donchian import channel_exit_levels, detect_donchian_setups
     ```
  2. **Candidate dispatch** — replace line 164:
     ```python
             if reg == "trending":
                 cands = [("donchian", c) for c in detect_donchian_setups(window)]
     ```
     and at the entry site (241) change `if method == "breakout":` to `if method == "donchian":`. The `else` branch (fade) is untouched.
  3. **Precompute the exit channel** once per run, beside the ATR series Phase 2 added:
     ```python
     exit_ch = channel_exit_levels(df_1h)          # df_1h == SETUP tier df
     exit_lower = exit_ch["lower"].to_numpy()
     exit_upper = exit_ch["upper"].to_numpy()
     ```
  4. **Record trail state at entry** — replace the `open_trade = {...}` assignment (255):
     ```python
     if bar_signals:
         sig = rank_signals(bar_signals)[0]
         open_trade = {
             "signal": sig,
             "entry_j": j,
             "regime": reg,
             "stop": sig.stop,          # MUTABLE: ratchets with the ATR trail
             "atr": atr_value,          # ATR on the setup tier at entry, frozen
             "extreme": sig.entry,      # best price seen since entry
             "trailed": False,          # True once the trail has moved the stop
             "h_idx": h_idx,
         }
     ```
  5. **Exit loop** — replace `engine.py:207-223` with:
     ```python
     if open_trade is not None:
         if j <= open_trade["entry_j"]:
             continue
         s = open_trade["signal"]
         stop = open_trade["stop"]
         # Setup-tier bar closed by this trigger bar: the opposite-channel
         # level is a trailing value, same lookup rule as the regime label.
         s_idx = int(np.searchsorted(close_1h, int(close_15m[j]), side="right")) - 1
         if s.direction == "long":
             chan = float(exit_lower[s_idx]) if s_idx >= 0 else float("nan")
             if lows[j] <= stop:
                 close_out(j, stop, "trail" if open_trade["trailed"] else "stop")
             elif not np.isnan(chan) and lows[j] <= chan:
                 close_out(j, chan, "channel")
             elif highs[j] >= s.target:
                 close_out(j, s.target, "target")
         else:
             chan = float(exit_upper[s_idx]) if s_idx >= 0 else float("nan")
             if highs[j] >= stop:
                 close_out(j, stop, "trail" if open_trade["trailed"] else "stop")
             elif not np.isnan(chan) and highs[j] >= chan:
                 close_out(j, chan, "channel")
             elif lows[j] <= s.target:
                 close_out(j, s.target, "target")
         if open_trade is not None and j - open_trade["entry_j"] >= max_hold:
             close_out(j, float(closes[j]), "time")
         # Ratchet AFTER this bar's exits are resolved: a stop derived from bar
         # j's own extreme, tested against bar j's own low, is intra-bar
         # lookahead. The trail only ever binds from bar j+1 onward.
         if open_trade is not None and open_trade["atr"] > 0:
             trail_dist = config.ATR_STOP_MULTIPLE * open_trade["atr"]
             if s.direction == "long":
                 open_trade["extreme"] = max(open_trade["extreme"], float(highs[j]))
                 new_stop = open_trade["extreme"] - trail_dist
                 if new_stop > open_trade["stop"]:
                     open_trade["stop"] = new_stop
                     open_trade["trailed"] = True
             else:
                 open_trade["extreme"] = min(open_trade["extreme"], float(lows[j]))
                 new_stop = open_trade["extreme"] + trail_dist
                 if new_stop < open_trade["stop"]:
                     open_trade["stop"] = new_stop
                     open_trade["trailed"] = True
         continue
     ```
  6. **Docstring**: extend the "Simulation rules" list (`engine.py:22-27`) with
     ```
       - Exits (Phase 5): the initial stop is the Signal's ATR stop; it ratchets
         to (extreme since entry -/+ ATR_STOP_MULTIPLE * ATR(setup TF)) and never
         loosens. A trade also exits on a touch of the trailing opposite
         DONCHIAN_ENTRY_PERIOD channel. Priority on a bar that touches several
         levels: stop/trail, then channel, then target (conservative). The
         ratchet is applied only after the bar's exits are resolved, so the
         trail can never fire on the same bar that produced its own extreme.
     ```
     and note the two new `outcome` values on the `Trade` docstring (61-66): `"stop", "trail", "channel", "target", "time", "end"`. `Trade.stop` keeps the **initial** stop (a frozen record of the setup), not the ratcheted one — document that on the field.
- **MIRROR**: `regime_at` (`engine.py:150-152`) for the `searchsorted(..., "right") - 1` trailing-lookup idiom; `engine.py:121-123` for config resolution.
- **IMPORTS**: `numpy as np` and `config` are already imported (33, 36).
- **GOTCHA**: (a) **Local names**: this task cites the pre-Phase-4 names `df_1h`/`close_1h` (setup tier) and `df_15m`/`close_15m`/`m15`/`highs`/`lows` (trigger tier). Phase 4 may have renamed them — read the file first and keep whatever it left. Do not confuse the tiers: `exit_ch` must be built from the **setup-tier** df. (b) `exit_lower` is indexed by *setup-tier* position (`s_idx`), `lows`/`highs` by *trigger-tier* position (`j`). Mixing them silently produces garbage that still runs. (c) `atr_value` in step 4 is the variable Phase 2 Task 7 introduces at the entry site; if it is only computed inside a conditional, hoist it. (d) Exit-priority ordering matters for the honesty of the result: stop before channel before target preserves the existing conservative same-bar rule. (e) The trail uses `config.ATR_STOP_MULTIPLE` — the Phase-2 frozen `k = 1.5`. Adding a separate trail multiple would introduce a fitted parameter this phase is forbidden from introducing. (f) `cand_cache` (155) is keyed on setup-bar index and stores `(regime, candidates)`; the exit-channel lookup happens outside it and needs no memoization — leave the cache alone.
- **VALIDATE**: `pytest tests/test_backtest.py -v`.

### Task 7: `tests/test_donchian.py`
- **ACTION**: Create `tests/test_donchian.py`.
- **IMPLEMENT**: Three classes, reusing the `make_df` / `seed_candles` builders copied from `tests/test_signals.py:22-28` and `498-504` (copy them into this file's header as those tests do — the suite has no shared `conftest.py`).
  ```python
  """Tests for the Phase 5 Donchian trend engine (indicator + signal method)."""

  import math

  import pandas as pd

  from trading_bot import config
  from trading_bot.data import storage
  from trading_bot.indicators.donchian import donchian
  from trading_bot.signals import donchian as donchian_mod
  from trading_bot.signals.donchian import (
      DONCHIAN_KIND,
      detect_donchian_setups,
      scan_donchian_signals,
  )

  SYMBOL = "BTCUSDT"
  SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
  TRIG_TF = config.SIGNAL_TRIGGER_TIMEFRAME
  SETUP_MS = storage.TIMEFRAME_MS[SETUP_TF]
  TRIG_MS = storage.TIMEFRAME_MS[TRIG_TF]
  START = 1_700_000_000_000
  ```
  `class TestDonchianIndicator`:
  - `test_leading_nans_until_period_bars` — first `period` values NaN, first defined at positional index `period` (mirrors `test_wilder.py::TestLeadingNaNs`).
  - `test_current_bar_excluded_from_its_own_channel` — **the load-bearing test**: 20 flat bars at high 100, then one bar with high 130; assert `upper.iloc[-1] == 100.0`, not 130. Without `.shift(1)` this is 130 and no breakout can ever fire.
  - `test_upper_lower_mid_arithmetic` — `mid == (upper + lower) / 2` on every defined row.
  - `test_trailing_only_matches_incremental` — for each `i` in `[period, n)`, `donchian(df.iloc[:i+1]).iloc[-1] == donchian(df).iloc[i]`. The no-lookahead invariant; catches any future `center=True` / `bfill` regression.
  - `test_short_series_all_nan` — `len(df) <= period` → all NaN.

  `class TestDetectDonchianSetups`: a rising series of ≥60 bars whose last bar closes above the trailing 20-bar high and the 55-mid with ADX > 25 (a monotone ramp achieves this — `test_wilder.py::TestMonotonicUptrend` already proves the fixture shape).
  - `test_long_candidate_shape` — exactly one candidate; `kind == DONCHIAN_KIND`; `direction == "long"`; `breakout_level == trailing 20-bar high`; `target_height == upper - lower`; `end_ts == last bar ts + SETUP_MS`.
  - `test_short_candidate_on_downtrend` — mirror.
  - `test_no_candidate_below_adx_floor` — choppy series, ADX < 25 → `[]`, asserting the ADX precondition directly so it cannot pass for the wrong reason.
  - `test_no_candidate_when_close_on_wrong_side_of_mid` — pass `adx_min=0.0` to isolate the filter; close below the 55-mid but above the 20-upper → `[]`.
  - `test_warmup_returns_empty` — 40 bars (< 56) → `[]`.
  - `test_no_geometry_params_referenced` — `import inspect; src = inspect.getsource(donchian_mod); assert "TRIANGLE" not in src and "FLAG_" not in src and "HS_" not in src`. Cheap, and it encodes the phase's actual success criterion.

  `class TestScanDonchianSignals`: seed both tiers via `seed_candles`, no monkeypatching (the scanner has no regime check); assert one `Signal` with `pattern == DONCHIAN_KIND`, `entry` == the trigger bar close, `stop == entry - config.ATR_STOP_MULTIPLE * atr`, `rr >= config.RR_FLOOR`. Plus `test_empty_db_returns_empty` and `test_warns_and_returns_empty_when_atr_undefined` (`caplog.at_level("WARNING", logger="trading_bot")`, mirroring `test_signals.py::TestContiguousTail::test_warns_when_trimming`).
- **MIRROR**: `tests/test_signals.py` header + helper-builder style; `tests/test_wilder.py::TestLeadingNaNs` for warmup assertions.
- **IMPORTS**: as listed; `math` for `isclose`.
- **GOTCHA**: Use `config.SIGNAL_PATTERN_TIMEFRAME` / `SIGNAL_TRIGGER_TIMEFRAME` rather than hardcoded `"4h"`/`"1h"` so the file survives any further tier change. And note the trigger df needs at least `VOLUME_LOOKBACK + 2 = 22` bars for `check_breakout` to grade volume rather than return NaN — `breakout_df`'s `n_prior=21` default exists for exactly this reason.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_donchian.py -v`.

### Task 8: Backtest exit tests
- **ACTION**: Edit `tests/test_backtest.py`.
- **IMPLEMENT**: Keep `TestComputeMetrics`, `TestWalkForward` and `TestBacktestCli` untouched (they use `make_trade` stubs / monkeypatched `run_backtest` and are unaffected). Rework `seed_scenario` (78-90) and `TestRunBacktest`: the flag fixture (`flag_rows`, 68-75) no longer produces candidates once geometry leaves the dispatch path, so replace it with `donchian_rows()` — a ramp long enough to clear the 56-bar setup warmup that puts the last setup bar above its trailing 20-bar high and 55-mid with ADX > 25 (reuse the fixture built in Task 7; factor it into `tests/test_backtest.py` locally rather than importing across test modules, matching the suite's existing duplication style). Then add:
  - `test_trail_ratchets_and_exits_above_initial_stop` — a run of favourable bars, then a pullback of more than `1.5 × ATR` from the extreme; assert `outcome == "trail"`, `exit_price > signal-initial stop`, and `pnl_pct > 0`.
  - `test_trail_never_loosens` — a favourable excursion followed by a deep adverse bar; assert the exit price equals the ratcheted level, not the (lower) initial stop.
  - `test_trail_cannot_fire_on_the_bar_that_set_its_own_extreme` — a single bar with a very high high and a low below `high − 1.5×ATR`; assert the trade does **not** exit on that bar (this is the intra-bar lookahead guard from Task 6's ratchet ordering).
  - `test_opposite_channel_touch_exits` — drive price down to the trailing 20-bar low with the trail still slack; assert `outcome == "channel"` and `exit_price` equals that channel value.
  - `test_stop_wins_over_channel_and_target_on_the_same_bar` — one bar touching all three; assert `outcome in {"stop", "trail"}` (the existing conservative same-bar rule, extended).
  - Keep `test_time_exit`, `test_unresolved_trade_closed_at_data_end`, `test_inactive_regime_produces_no_trades`, `test_empty_db_returns_empty` — update only their fixtures.
- **MIRROR**: `seed`/`seed_scenario`/`patch_trending` (`test_backtest.py:60-90`) — extend, do not replace.
- **IMPORTS**: no new imports.
- **GOTCHA**: `patch_trending` monkeypatches `engine.classify_series` to label *every* bar; keep using it so the regime layer never confounds an exit test. And with `fee_pct=0.0, slippage_pct=0.0` (plus Phase 2's `funding_pct_per_day=0.0`) the arithmetic assertions stay exact — the existing tests already pass costs explicitly for this reason (`test_backtest.py:110, 130`).
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_backtest.py -v`.

### Task 9: Fix the dispatch monkeypatch in `tests/test_meanrev.py`
- **ACTION**: Edit `tests/test_meanrev.py:232-246`.
- **IMPLEMENT**: In `TestScanSymbol._patch`, change
  ```python
  monkeypatch.setattr(scan, "scan_breakout_signals", lambda conn, sym, now: ["breakout-sentinel"])
  ```
  to
  ```python
  monkeypatch.setattr(scan, "scan_donchian_signals", lambda conn, sym, now: ["donchian-sentinel"])
  ```
  and update `test_trending_dispatches_to_breakout_method` (rename to `test_trending_dispatches_to_donchian_method`) to expect `("trending", ["donchian-sentinel"])`.
- **MIRROR**: The existing `_patch` helper shape.
- **IMPORTS**: none.
- **GOTCHA**: `monkeypatch.setattr` with a name that no longer exists on the module raises `AttributeError` — so this test **fails loudly** rather than silently passing, which is why it is a separate task and not a footnote. It is also the only place Phase 5 must touch a Phase 6-owned test file; coordinate.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_meanrev.py -v`.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
Full per-test specs are in Tasks 7 and 8. The five that carry the phase's correctness are:

| Test | Input | Expected Output | Why it matters |
|---|---|---|---|
| `test_current_bar_excluded_from_its_own_channel` | 20 flat bars @high 100, then high 130 | `upper.iloc[-1] == 100.0` | Without `.shift(1)`: zero signals forever |
| `test_trailing_only_matches_incremental` | 80-bar series | `donchian(df[:i+1]).iloc[-1] == donchian(df).iloc[i]` ∀ i | The no-lookahead invariant |
| `test_no_candidate_when_close_on_wrong_side_of_mid` | close > upper20 but < mid55, `adx_min=0` | `[]` | Isolates the 55-filter from the ADX gate |
| `test_trail_cannot_fire_on_the_bar_that_set_its_own_extreme` | one bar: high far up, low below high−1.5×ATR | no exit that bar | Intra-bar lookahead guard |
| `test_stop_wins_over_channel_and_target_on_the_same_bar` | bar touching all three | `outcome in {"stop","trail"}` | Conservative same-bar rule preserved |

### Edge Cases Checklist
- [x] Channel excludes the current bar (`.shift(1)`)
- [x] Warmup NaN semantics for channel (period bars) and ADX (2·period−1)
- [x] Degenerate channel (`upper == lower`) → no candidate
- [x] Close exactly on the 55-mid → no candidate (no arbitrary side)
- [x] Long/short symmetry in candidate, trail, and channel exit
- [x] `end_ts` set to the setup bar's close, not its open
- [x] Trail ratchet ordering (no intra-bar lookahead)
- [x] Trail monotonicity (never loosens)
- [x] Exit priority on a multi-touch bar
- [x] Setup-tier vs trigger-tier index confusion in the engine (`s_idx` vs `j`)
- [ ] Concurrent access — N/A (single-threaded replay; pure functions)
- [ ] Network failure — N/A (no network calls in this phase)

---

## Validation Commands

The project has **no linter and no type checker** — `pyproject.toml` declares only `ccxt`, `pandas`, `apscheduler`, `python-dotenv` and a `dev` extra of `pytest`, with a `[tool.pytest.ini_options]` block registering a `network` marker and `testpaths = ["tests"]`. There is no `Makefile`. Do not invent a lint/typecheck step.

**Interpreter**: the repo venv is Python 3.11.6 (Homebrew `python@3.11`) at `.venv/`. Always invoke it explicitly.

### Static Analysis
```bash
cd /Users/ttaa/Documents/Project.nosync/InvestmentBot/trading
.venv/bin/python -m py_compile \
  src/trading_bot/indicators/donchian.py \
  src/trading_bot/signals/donchian.py \
  src/trading_bot/signals/scan.py \
  src/trading_bot/signals/setup.py \
  src/trading_bot/backtest/engine.py \
  src/trading_bot/config.py
```
EXPECT: zero output.

### Unit Tests
```bash
.venv/bin/python -m pytest tests/test_donchian.py tests/test_backtest.py tests/test_meanrev.py -v
```
EXPECT: all pass.

### Full Suite
```bash
.venv/bin/python -m pytest tests/ -q
```
EXPECT: no regressions. `test_wilder.py` (22 tests), `test_classifier.py`, `test_storage.py`, `test_backfill.py`, `test_poller.py`, `test_binance_client.py` are untouched by this phase. `test_signals.py`'s pivot/pattern/`TestBuildSignal` classes must **stay green** — retirement is a dispatch decision, not a deletion.

### Retirement Verification
```bash
grep -rn "detect_patterns\|find_pivots" src/trading_bot/signals/scan.py src/trading_bot/backtest/engine.py src/trading_bot/signals/donchian.py
```
EXPECT: no matches — geometry is off the active path.
```bash
grep -rn "DONCHIAN_ENTRY_PERIOD\|DONCHIAN_TREND_PERIOD" src/trading_bot/backtest/walkforward.py
```
EXPECT: no matches — the canonical parameters are not in any grid.

### Manual Validation (the phase's own success signal)
```bash
.venv/bin/python -m trading_bot.cli backtest --symbol BTCUSDT
.venv/bin/python -m trading_bot.cli backtest --symbol ETHUSDT
.venv/bin/python -m trading_bot.cli backtest --symbol SOLUSDT
```
- [ ] All three symbols produce a **non-zero** trade count in the `trending/donchian-breakout` bucket. Zero trades on any symbol almost always means the `.shift(1)` is wrong in one direction or the `end_ts` freshness test is rejecting every trigger bar.
- [ ] Rate is plausible. The DB holds **7,799 4H bars per symbol** spanning 2023-01-01 → 2026-07 ≈ 1,300 days ≈ 186 weeks (verified: `sqlite3 -readonly data/ohlcv.db "select timeframe, symbol, count(*), min(ts), max(ts) from ohlcv group by 1,2"` — timeframes present are `15m`/`1h`/`4h` only; there is **no `1d`** yet). With the ~36–43% trending occupancy the classifier measures, one-open-trade-at-a-time, and multi-day holds, expect roughly **100–600 entries per symbol** over the full span (≈0.5–3 per week). **Above ~1,500 per symbol (≈8/week) the phase has failed its own criterion** ("a few per week, not a few per day") — the usual cause is a trigger tier re-firing on an extended move, i.e. `lookback_bars` not pinned to 1 or the fresh-crossing pair spanning a data gap.
- [ ] Warmup cost is as expected: no entry before roughly the 56th setup bar (≈9.4 days of 4H bars) plus the regime classifier's own `REGIME_MIN_BARS`.
- [ ] Exit-mix sanity from the trade list: all of `stop`, `trail`, `channel`, `target`, `time` appear. If `trail` never appears, the ratchet is dead code; if `target` never appears, `target_height` (channel width) is too wide for the hold limit and that is worth reporting to Phase 7, **not** worth retuning here.
- [ ] Do **not** run `walkforward`. Do not report expectancy or Sharpe as a verdict — Phase 7 owns the verdict.

---

## Acceptance Criteria
- [ ] All 9 tasks completed
- [ ] All validation commands pass
- [ ] Tests written and passing (new `tests/test_donchian.py`; extended `tests/test_backtest.py`; fixed `tests/test_meanrev.py`)
- [ ] `tests/test_signals.py` still green (retired modules keep their coverage)
- [ ] No type errors — N/A (no type checker configured)
- [ ] No lint errors — N/A (no linter configured; match surrounding style by inspection)
- [ ] Matches UX design — N/A, internal change (the CLI's `signal` table gains a new `pattern` value, `donchian-breakout`, and nothing else)

## Completion Checklist
- [ ] Code follows discovered patterns (keyword-only optional overrides with config fallback, `logging.getLogger("trading_bot")`, frozen dataclasses, `return None`/`[]` instead of raising on a failed filter)
- [ ] Error handling matches codebase style — warmup and filter failures return empty, never raise; only genuinely impossible geometry warns
- [ ] Logging follows conventions — DEBUG for per-candidate rejections (inside `build_signal`), WARNING for stale data / undefined ATR
- [ ] Tests follow test patterns (`make_df` / `seed` / `seed_scenario` / `patch_trending` reused, not reinvented; `caplog.at_level(..., logger="trading_bot")` for log assertions)
- [ ] No hardcoded values — 20, 55, ADX floor, ATR period and multiple all resolve through `config`
- [ ] No new fitted parameters — verified by `TestDetectDonchianSetups::test_no_geometry_params_referenced` and by the `walkforward.py` grep
- [ ] Documentation updated — module docstrings on both new files; `engine.py`'s simulation-rules block extended with the two new exit modes; `scan.py` and `cli.py:307` regime-table docstrings corrected; retirement banners added in `setup.py`
- [ ] Stated Assumptions A1 (4H tier) and A2 (20 entry+exit / 55 filter) reproduced verbatim in `signals/donchian.py`'s module docstring so the decision survives without the plan
- [ ] Self-contained — no questions needed during implementation

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `.shift(1)` omitted or reintroduced later → channel includes the current bar → **zero signals forever**, or (if shifted the wrong way) a lookahead-biased backtest that looks excellent | Medium | **Critical** | `test_current_bar_excluded_from_its_own_channel` and `test_trailing_only_matches_incremental` are both mandatory and both named in Task 7; the module docstring states the reason |
| The mutable stop in `engine.py` breaks the no-lookahead invariant: a trail computed from bar *j*'s own high, tested against bar *j*'s own low | Medium | **High** (silently inflates every result) | Ratchet runs *after* the bar's exits are resolved (Task 6 step 5, with the rationale as an inline comment) and is pinned by `test_trail_cannot_fire_on_the_bar_that_set_its_own_extreme` |
| Setup-tier vs trigger-tier index confusion (`exit_lower[s_idx]` vs `lows[j]`) — runs fine, produces nonsense | Medium | High | Task 6 GOTCHA (b); the channel-exit test asserts the exit price *equals a specific channel value*, which fails loudly on misalignment |
| Phase 2 has not landed (or landed with different names), so `build_signal(…, atr_value)` / `ATR_STOP_MULTIPLE` do not exist | Medium | High | Dependencies table lists every Phase-2 artifact by name; the failure mode is an import/`TypeError` at first test run, not silent. Do **not** add a default `atr_value` to work around it |
| Phase 4 has not landed, so the "4H setup tier" assumption is false and the channel is a 20-*hour* channel | Medium | Medium | A1 states the dependency; every reference goes through `config.SIGNAL_PATTERN_TIMEFRAME`, so the code is correct-by-construction once Phase 4 lands — only the `c ≤ 0.10` claim is premature |
| Signal frequency lands far outside "a few per week" (either ~0 or several per day) | Medium | Medium | Explicit numeric plausibility band and the two usual root causes are in Manual Validation. **Not** a licence to adjust 20/55 |
| A2's channel-role reading is wrong, and the Turtle-System-2 assignment (55 entry / 20 exit) is what the strategy doc meant | Medium | Medium | A2 gives the falsifying argument (a 55-entry makes the 55-mid filter tautological) and pre-registers the alternative as a legitimate *Phase 7* grid axis, so it is testable later without spending a degree of freedom now |
| `target_height` = channel width produces targets so distant that `target` exits never occur, making the R:R floor and the target academic | Medium | Low-Medium | Reported, not fixed: exit-mix sanity is a Manual Validation checklist item, and the finding is handed to Phase 7. An ATR-multiple target would be a new fitted parameter — forbidden here |
| Phase 5 and Phase 6 collide in `signals/scan.py` and `tests/test_meanrev.py` | Medium | Low | Both files are called out in Tasks 5 and 9 with the merge boundary named (trending branch vs ranging branch) |
| Retiring geometry loses coverage that was catching real bugs | Low | Low-Medium | Nothing is deleted: `patterns.py`, `pivots.py`, `setup.scan_breakout_signals` and all their tests stay, only unreferenced by the dispatcher |

## Notes
- **The two new exit modes are the only genuinely new capability here.** Everything else substitutes into interfaces that already exist: `check_breakout` is generic over `(breakout_level, direction, end_ts)`, `build_signal` already computes stop/target/R:R, `rank_signals` already orders, `engine.candidates_for` already dispatches by regime. The exit loop, by contrast, has only ever read two immutable numbers off a frozen `Signal` — teaching it a ratcheting stop is where this phase's real risk lives, which is why three of the new engine tests are lookahead guards rather than behavior tests.
- **Frequency is the success criterion, not profitability.** Expectancy, Sharpe and the verdict belong to Phase 7. If the backtest's expectancy looks bad, that is information for Phase 7; if it looks good, acting on it is how the OOS holdout gets spent.
- **Two canonical numbers, zero fitted ones.** Every number here is canonical (20, 55, ADX 25) or already frozen by Phase 2 (`k = 1.5`, `RR_FLOOR = 1.5`, `ATR_STOP_PERIOD = 14`). Needing a new one means the design has drifted — re-read A1/A2 rather than adding a constant.
- Carried over from the PRD: branch `fix/phase3-breakout-detection` touched `engine.py`, `config.py`, `signals/*`, `tests/test_signals.py`, and its premise is superseded by this phase retiring that method entirely. Verify `git status` is clean of it before Task 1.
