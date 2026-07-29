# v0.3.0 — Shared Architecture Contract

**Status: BINDING on all nine phase plans in this directory.**

This file is the coherence spine for the nine `/ecc:prp-plan` outputs generated from
[self-learning-pattern-framework.prd.md](../../prds/self-learning-pattern-framework.prd.md).
Nine plans were authored in parallel; every cross-phase name, path, signature, and
sequencing decision below was fixed BEFORE they were written so they compose instead of
colliding. A plan that contradicts this file is wrong; this file is not a suggestion.

Read this first, then the phase plan you are implementing.

---

## 0. Verified ground truth (measured 2026-07-27, not assumed)

These were measured against the working tree at the time of planning. Re-verify anything
you are about to depend on; do not re-derive what is already here.

| Fact | Value | How verified |
|---|---|---|
| Python | 3.11.6 (`.venv/bin/python`, Homebrew python@3.11) | `.venv/bin/python -V` |
| pandas / numpy / ccxt | 3.0.3 / 2.4.6 / 4.5.64 | import check |
| **pandas 3.x gotcha** | `Series.pct_change()` behavior changed and is **unsafe here** — compute returns with explicit arithmetic (`s / s.shift(1) - 1.0`) and say so. Found by the Phase 2 plan; applies to any phase computing returns. | Phase 2 measurement |
| pytest | 9.1.1, **286 tests collected** | `pytest --collect-only` |
| Test invocation | `.venv/bin/python -m pytest` (no linter, no type checker configured) | KNOWN-LIMITATIONS §8 |
| Source LOC | ~5.4k in `src/trading_bot`, ~4.6k in `tests`, ~6.4k in `scripts/bruteforce` | `wc -l` |
| OHLCV store | `data/ohlcv.db`, 117 MB | `ls -la data/` |
| **Symbols stored** | **20** — BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, TRX, BCH, NEAR, ATOM, UNI, AAVE, FIL, OP, APT (all `…USDT`) | `sqlite3 data/ohlcv.db` GROUP BY |
| Timeframe coverage | all 20 symbols have 1d (1304 bars), 4h (7820), 1h (31279). **15m exists for BTC/ETH/SOL only** (125 096 bars each) | same query |
| Exact stored span (**CORRECTED** 2026-07-27) | last bar OPEN time per tier: **1d 2026-07-27 00:00**, 4h 2026-07-27 04:00, 1h 2026-07-27 06:00, 15m 2026-07-26 18:45. First 1d bar 2023-01-01 00:00; sub-daily tiers start earlier, at 2022-12-31 17:00–20:00 UTC. | `SELECT timeframe, MIN(ts), MAX(ts), datetime(…) FROM ohlcv GROUP BY timeframe` |
| Production symbol set | `config.SYMBOLS = ("BTCUSDT","ETHUSDT","SOLUSDT")` — still 3, despite 20 being stored | `config.py:10` |
| `fapi.binance.com` | blocked 2026-07-05, reachable 2026-07-26. **Re-ping before any backfill.** | KNOWN-LIMITATIONS §4 |

**Gotcha on the last bar, binding on every span calculation.** Today is 2026-07-27, so the
final 1d bar (open 2026-07-27 00:00) is **still forming** and is not a closed candle.
`storage.find_gaps` already excludes the currently-forming candle by construction
(`storage.py:164-166`: `expected_end = (now_ms // interval) * interval - interval`). Any phase
that derives a span, a holdout boundary, or a bar count must use the last **closed** bar, not
`MAX(ts)`, and must state which it used. Treating a forming bar as closed is a lookahead bug
that fails in the flattering direction.

### 0a. The single most important correction to the PRD

The PRD's Phase 2 ("backfill uncorrelated symbols") and its Technical Risks table both
assume **3 symbols are stored**. That is stale: **17 extra symbols were already backfilled**
by the `scripts/bruteforce/` side effort at 1d/4h/1h over the full span. Phase 2 is therefore
NOT primarily a backfill phase — it is a **selection, correlation-measurement, gap-verification
and promotion** phase. Its plan reflects this. Do not spend a day re-downloading data you have.

The unchanged part of §0b's problem: all 20 stored symbols are **liquid majors**, which are
plausibly *all* high BTC beta. Phase 2 must measure this rather than assume the correlation
trap is solved by row count.

**MEASURED 2026-07-27 by the Phase 2 plan — use these numbers, do not re-derive them.**
On 1095 daily returns (2023-07-27 → 2026-07-26, zero missing bars):

| Symbol set | mean pairwise r | Kish effective N |
|---|---|---|
| 3 core (BTC/ETH/SOL) | 0.7574 | **1.193** ← reproduces §0b exactly |
| best 8-symbol low-correlation subset | — | **1.868** |
| pre-registered selection (BTC pinned + lowest-8 by mean r) | — | **1.834** |

Pairwise anchors reproduce §0b to three decimals: BTC/ETH +0.809, BTC/SOL +0.744, ETH/SOL +0.719.

**So 3 → 20 stored symbols buys 6.7× the rows and only ~1.5× the independent information.**
§0b's trap is **mitigated, not solved**. Every phase whose arithmetic depends on sample
adequacy — Phase 6's population sizing, Phase 9's holdout length and trade-floor analysis —
must use **effN ≈ 1.83** for the broadened set, not 1.19 (the 3-symbol figure) and never the
raw symbol count. Independent-equivalent trades ≈ (per-symbol trades) × effN.

Two further measured facts from the same pass: `fapi.binance.com/fapi/v1/ping` returned HTTP
200 in 0.19 s today; and there are **zero interior gaps in all 60 research cells**
(20 symbols × 3 timeframes), with trailing poller lag present on 40 of 60 — so `now_ms` must be
pinned per cell or lag masquerades as a hole. Storage measures 98.84 bytes/row, so ten more
symbols would cost ~39.9 MB against 256 GiB free: **disk is not the constraint, correlation is.**

### 0b. The second most important correction

`scripts/bruteforce/` is a **6.4k-line reusable donor**, not dead weight, and the PRD
under-describes it. It already contains, working and causality-checked:

- `indicators.py` (631 lines): `macd`, `rsi`, `stochastic`, `cci`, `williams_r`, `roc`,
  `obv`, `mfi`, `vwap_session`, `keltner`, `squeeze_on`, `zscore`, `percentile_rank`,
  `swing_levels`, `pivot_high/low`, plus detectors for `double_top`, `double_bottom`,
  `triangle_squeeze`, `bull_flag`, `bear_flag`, `head_and_shoulders`,
  `inverse_head_and_shoulders`, `breakout_of_range`, and 11 candlestick patterns.
- `registry.py` (179 lines): a `@register(family=…, grid=…, rationale=…)` decorator,
  `combo_count()`, `total_trials()`, and `load_all()` via `pkgutil` — **the closest existing
  analogue to the Phase 3 plug-in registry, and its lessons (mandatory `rationale`,
  duplicate-name rejection, fatal import errors, trial counting as a first-class concept)
  are to be carried forward, not reinvented.**
- `core.py` (700 lines): `Ctx` multi-timeframe alignment, `Plan`, a **vectorized** `simulate()`
  over a numpy `TRADE_DTYPE`, `score()` including DSR, named `SPLITS`, and `assert_causal()`.
- `universe.py`: the documented 20-symbol universe with sector labels and its selection rule.

**Rule: do not re-derive an indicator or detector that exists in `scripts/bruteforce/`.**
Port it (with attribution in the module docstring) behind the Phase 3 contracts. The PRD's
claim that this explorer "could not incorporate RSI" is imprecise — the inexpressiveness was
in `src/trading_bot`'s hardcoded regime switch, not in the research scripts. Correcting this
saves Phases 4 and 8 substantial work.

Caveat that keeps this honest: `scripts/bruteforce/` is **research-grade, outside the tested
package**, has its own cost/score path, and is NOT covered by the 286 production tests. Ported
code enters `src/trading_bot/` only with tests written to production standards, and the
production cost model / `engine.Trade` / gate remain the single source of truth. Never call
`scripts.bruteforce.core.score` as a fitness oracle (see §4).

---

## 1. What every plan reuses unchanged (the honest core)

Do not rewrite, wrap, or fork these. v0.3.0's whole premise is that they are v0.2.0's proven assets.

| Component | Path | Contract you must honor |
|---|---|---|
| OHLCV store | `src/trading_bot/data/storage.py` | `connect()`, `upsert_candles()`, `load_candles(conn, symbol, tf, *, start_ms, end_ms)` (both bounds **inclusive**), `find_gaps()`, `TIMEFRAME_MS`. All access serialized by `_db_lock`. Timestamps are candle **OPEN** times, epoch ms, UTC. |
| Cost model | `src/trading_bot/config.py` + `engine.py` | `2*(FEE_PCT+SLIPPAGE_PCT)` round-trip plus `FUNDING_PCT_PER_DAY * holding_days`. **Frozen.** Any new execution path charges costs identically or it is lying. |
| Backtest engine | `src/trading_bot/backtest/engine.py` | `run_backtest(conn, symbol, *, start_ms, end_ms, params, fee_pct, slippage_pct, funding_pct_per_day, max_hold_bars) -> list[Trade]`; `BacktestParams`; `Trade`; `clear_caches()`; `_CACHE` keyed by content `_fingerprint`. |
| No-lookahead enforcement | `engine.py:185` `_assert_interval` | Raises when stored bar spacing disagrees with the configured tier. Every new data path that loads bars for simulation must keep this guarantee. |
| Trade metrics | `backtest/metrics.py` | `compute_metrics(trades) -> dict` with `n_trades`, `win_rate`, `expectancy_pct`, `avg_win_pct`, `avg_loss_pct`, `profit_factor`, `max_drawdown_pct`, `by_bucket`. |
| Equity metrics | `backtest/equity.py` | `daily_returns`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `probabilistic_sharpe`, `expected_max_sharpe`, `deflated_sharpe`, `compute_equity_metrics(trades, start_ms, end_ms, n_trials) -> dict`. |
| The gate | `backtest/walkforward.py` | `walk_forward_pooled(...) -> WalkForwardResult`; `DEFAULT_GRID`; `GATE_MIN_SHARPE/DSR/MAX_DRAWDOWN`; `_evaluate_gate`. Extended in Phase 1 (§4). |
| Indicators | `src/trading_bot/indicators/{wilder,bollinger,donchian}.py` | Hand-rolled Wilder `atr/adx/plus_di/minus_di`, `bollinger`, `donchian`. `pandas-ta` is **gone from PyPI**; TA-Lib needs a C library. **Extend the hand-rolled set; add no new indicator dependency.** |
| Regime classifier | `regime/classifier.py` | `classify_series`, `current_regime`, `REGIMES`. The one measured-healthy layer — thresholds are never swept. |
| Existing detectors | `signals/{patterns,pivots,donchian,breakout,meanrev}.py` | `PatternCandidate`, `Pivot`, `BreakoutEvent`, `Signal`, `check_breakout`, `find_pivots`, `detect_patterns`, `detect_donchian_setups`, `channel_exit_levels`, `build_signal`, `rank_signals`, `DONCHIAN_KIND`. |
| Live/backtest parity | `signals/scan.py` ↔ `backtest/engine.py` | Both dispatch the same detectors; `FADE_ENABLED` is read at call time so both suppress together. **The graph executor must preserve one-code-path parity the same way.** |

---

## 2. Canonical file layout for everything NEW

Every new module goes exactly here. This table is the deconfliction mechanism — no two
phases create the same file, and no phase invents a sibling directory for the same concept.

```
src/trading_bot/
  framework/                      # PHASE 3 owns this package
    __init__.py                   #   re-exports contracts, registry, graph API
    contracts.py                  #   the 7 Protocols (§3)
    registry.py                   #   @register decorator + REGISTRY + load_all()
    graph.py                      #   StrategyGraph, NodeSpec, to_dict/from_dict, SCHEMA_VERSION
    context.py                    #   EvalContext — multi-timeframe, no-lookahead bar access
    execute.py                    #   run_graph_backtest() — the ONE graph→Trade seam (§5)
    errors.py                     #   FrameworkError, ContractError, RegistryError, GraphError
  plugins/                        # PHASE 3 creates the package + migrations;
    __init__.py                   #   load_all() imports every submodule so decorators run
    data/ohlcv.py                 #   P3  DataSource over storage.load_candles
    detectors/legacy_patterns.py  #   P3  H&S / triangle / flag  (wraps signals/patterns.py)
    detectors/donchian.py         #   P3  wraps signals/donchian.detect_donchian_setups
    detectors/bollinger_fade.py   #   P3  wraps signals/meanrev.detect_fade_setups
    detectors/macd_cross.py       #   P4
    detectors/<family>.py         #   P8  one module per catalog family (§9)
    confirmations/volume_breakout.py  # P4
    confirmations/macd.py         #   P4
    policies/measured_move.py     #   P4  entry/TP/SL from pattern outcome
    filters/rr_after_costs.py     #   P4  the >=1:2-after-costs filter
    reviewers/trade_quality.py    #   P5
    mutators/param_jitter.py      #   P6
    mutators/graph_edit.py        #   P6  detector swap / add / remove
  indicators/macd.py              # PHASE 4  (beside wilder.py; port from bruteforce)
  indicators/rsi.py               # PHASE 8
  backtest/benchmark.py           # PHASE 1  buy-and-hold null
  backtest/trials.py              # PHASE 1  trial-count ledger
  data/statestore.py              # PHASE 1  second SQLite DB for framework state (§6)
  data/correlation.py             # PHASE 2  correlation matrix + effective-N
  feedback/                       # PHASE 5 owns this package
    __init__.py  records.py  versioning.py  protocol.py
  evolution/                      # PHASE 6 owns this package
    __init__.py  population.py  mutate.py  tournament.py  oracle.py  runner.py
  ui/                             # PHASE 7 owns this package
    __init__.py  server.py  api.py  static/{index.html,app.js,app.css}
  campaign.py                     # PHASE 9 owns this module (added 2026-07-27: the
                                  #   original table assigned Phase 9 no src/ module, which
                                  #   left the campaign driver homeless. Phase 9 claims
                                  #   exactly this one file, plus scripts/build_campaign_report.py)
data/
  ohlcv.db                        # existing, untouched schema
  state.db                        # NEW (P1) — framework state; see §6
  strategies/<name>.strategy.json # NEW (P3) — serialized graphs
tests/
  <per-phase files, see §8>
```

**Phase ownership is exclusive.** If your plan needs to change a file another phase owns,
it must instead depend on that phase (check the dependency graph in §11) and consume the
published interface. The only files legitimately touched by several phases are
`config.py`, `cli.py`, and `walkforward.py` — see §7 for how that is deconflicted.

---

## 3. The seven plug-in contracts (Phase 3 authors, everyone else consumes)

Defined in `framework/contracts.py` as `typing.Protocol` classes with
`@runtime_checkable`, **not** ABCs — plug-ins are plain functions/dataclasses registered by
decorator, exactly as `scripts/bruteforce/registry.py` does it, so nothing is forced to
inherit. Every contract is **pure and side-effect free** except `DataSource` (reads SQLite)
and `Reviewer` (writes review records).

```python
# All timestamps: epoch ms, UTC, candle OPEN time. All prices: float.
# All frames: pd.DataFrame indexed by ts (ascending), columns open/high/low/close/volume.

class DataSource(Protocol):
    """Bars/series by symbol + timeframe. MVP implementation is OHLCV over storage.py."""
    def frame(self, symbol: str, timeframe: str, *, start_ms: int | None, end_ms: int | None) -> pd.DataFrame: ...
    def timeframes(self) -> tuple[str, ...]: ...

class Detector(Protocol):
    """Pattern/structure events. Returns PatternCandidate-compatible events."""
    def detect(self, ctx: EvalContext, **params) -> list[DetectedEvent]: ...

class Confirmation(Protocol):
    """Gate on a candidate event. True = let it through. NEVER mutates the event."""
    def confirm(self, ctx: EvalContext, event: DetectedEvent, **params) -> ConfirmationVerdict: ...

class PositionPolicy(Protocol):
    """direction + entry + TP + SL from a confirmed event."""
    def decide(self, ctx: EvalContext, event: DetectedEvent, **params) -> PositionPlan | None: ...

class Filter(Protocol):
    """Accept/reject a PositionPlan. The R:R-after-costs filter lives here."""
    def accept(self, ctx: EvalContext, plan: PositionPlan, **params) -> FilterVerdict: ...

class Reviewer(Protocol):
    """Closed-trade analysis -> a persisted ReviewRecord."""
    def review(self, trade: Trade, context: ReviewContext, **params) -> ReviewRecord: ...

class Mutator(Protocol):
    """Strategy-variant generation for the evolution engine."""
    def mutate(self, graph: StrategyGraph, rng: random.Random, **params) -> StrategyGraph: ...
```

Supporting frozen dataclasses, all in `framework/contracts.py`:

| Type | Fields (canonical) |
|---|---|
| `DetectedEvent` | `kind: str`, `direction: str` (`"long"`/`"short"`), `level: float`, `target_height: float`, `start_ts: int`, `end_ts: int`, `meta: Mapping[str, float]` |
| `ConfirmationVerdict` | `passed: bool`, `name: str`, `score: float`, `reason: str` |
| `PositionPlan` | `symbol: str`, `ts: int`, `direction: str`, `entry: float`, `stop: float`, `target: float`, `risk_pct: float`, `reward_pct: float`, `rr: float`, `source: str` |
| `FilterVerdict` | `accepted: bool`, `name: str`, `reason: str`, `measured: Mapping[str, float]` |
| `ReviewRecord` | defined by Phase 5; persisted to `state.db` |
| `ReviewContext` | **defined by Phase 5** in `feedback/records.py`, not by Phase 3 (gap found and closed 2026-07-27: the `Reviewer` Protocol above names this type but the original table assigned it no owner). Phase 3 declares the Protocol referencing it; Phase 5 owns the concrete type, exactly as with `ReviewRecord`. If Phase 3's plan also defines it, Phase 5's definition wins and Phase 3's is deleted at implementation time. |

**`DetectedEvent` is deliberately field-compatible with the existing
`signals.patterns.PatternCandidate`** (`kind`/`direction`/`breakout_level`→`level`/
`target_height`/`start_ts`/`end_ts`) so migration is an adapter, not a rewrite, and
`signals/breakout.check_breakout` can be reused against it.

### Registry

```python
# framework/registry.py
REGISTRY: dict[str, PluginSpec]          # key = "<kind>.<name>", e.g. "detector.head-and-shoulders"
KINDS = ("data", "detector", "confirmation", "policy", "filter", "reviewer", "mutator")

def register(kind: str, *, name: str, params: dict[str, ParamSpec], rationale: str,
             timeframes: tuple[str, ...] = (), tier: int | None = None) -> Callable
def get(key: str) -> PluginSpec
def by_kind(kind: str) -> dict[str, PluginSpec]
def load_all() -> dict[str, PluginSpec]   # pkgutil-walks trading_bot.plugins; import errors are FATAL
```

Carried forward from `scripts/bruteforce/registry.py`, deliberately:
- **`rationale` is mandatory and non-empty** — "an unmotivated strategy in a 10 000-combo
  sweep is just noise with a name."
- **Duplicate names raise**, naming the module that already claimed the key.
- **Import errors during `load_all()` are fatal, never skipped** — a family silently missing
  from a report reads as "tested and found wanting."
- `params: dict[str, ParamSpec]` replaces bruteforce's `grid: dict[str, list]`. `ParamSpec`
  carries `default`, `bounds`/`choices`, and `kind` (`"int"|"float"|"bool"|"choice"`) so the
  **UI can render a control and the Mutator can jitter within legal bounds from the same
  declaration.** This one change is why the grid becomes a spec.

### Plug-in naming

Registry names are **lowercase-hyphen**, matching `DONCHIAN_KIND = "donchian-breakout"` and
`PATTERN_KINDS = ("head-and-shoulders", …)` already in the codebase. Module names are
snake_case. Python identifiers stay snake_case. Never `camelCase` anywhere.

---

## 4. THE GATE is the only fitness oracle

This is the anti-overfitting spine and the PRD's hardest constraint. Stated once, here:

1. **One code path.** Fitness comes only from `walkforward.walk_forward_pooled`. There is no
   second scoring function. `scripts/bruteforce/core.score` is **not** an oracle — it exists
   for the retired research sweep and must not be wired into evolution.
2. **Every evaluation increments the ledger.** `backtest/trials.py` (Phase 1) owns a persistent
   trial ledger in `state.db`. The oracle increments it; nothing may score a candidate without
   passing a ledger handle. Phase 6's `evolution/oracle.py` is a thin wrapper that *cannot*
   bypass it.
3. **Buy-and-hold is the null.** Zero-return nulls were v0.2.0's documented mistake
   (KNOWN-LIMITATIONS §0: "THE GATE could have blessed a strategy worse than inaction").
4. **The final holdout is never seen by evolution.** Phase 6 trains on partial data; Phase 9
   owns a span no generation ever touched.

### The extended gate (Phase 1 delivers; Phases 6 and 9 consume)

```python
# backtest/walkforward.py — ADDITIVE changes only; existing field names keep their meaning.
GATE_MIN_SHARPE      = 1.0     # unchanged
GATE_MIN_DSR         = 0.95    # unchanged
GATE_MAX_DRAWDOWN    = 0.25    # unchanged

GATE_CONDITIONS = (
    "sample_adequacy",          # n_trades >= min_trades
    "sharpe",                   # >= GATE_MIN_SHARPE
    "dsr",                      # >  GATE_MIN_DSR
    "max_drawdown",             # <= GATE_MAX_DRAWDOWN
    "per_symbol_expectancy",    # EVERY symbol > 0 (AND, not average)
    "beats_benchmark_return",   # NEW: ann_return_pct > basket ann_return_pct
    "beats_benchmark_sharpe",   # NEW: sharpe        > basket sharpe
)

def _evaluate_gate(...) -> dict[str, bool]      # CHANGED: was -> bool
```

`WalkForwardResult` gains, appended, never reordered:

```python
gate: dict[str, bool]            # per-condition verdicts, keyed by GATE_CONDITIONS
benchmark: BenchmarkResult       # from backtest/benchmark.py
n_trials_used: int               # what the DSR was actually charged
passed: bool                     # KEPT, == all(gate.values()) — every existing caller still works
```

```python
# backtest/benchmark.py  (Phase 1)
@dataclass(frozen=True)
class BenchmarkResult:
    per_symbol: dict[str, dict]   # symbol -> {total_return, ann_return_pct, sharpe, sortino,
                                  #            max_drawdown_pct, n_days}
    basket: dict                  # equal-weight, daily-rebalanced, same keys
    start_ms: int
    end_ms: int

def buy_and_hold(conn, symbols, *, start_ms, end_ms) -> BenchmarkResult
```
**MEASURED 2026-07-27 by the Phase 1 plan — three corrections, use these.**

1. **Basket construction is decided, not open**: equal-weight **daily-rebalanced** reproduces all
   four §0 basket figures (2.1628× / +29.32% / 0.7284 / 64.32%); **buy-once-hold does not**
   (2.0809× / 0.6998 / 66.91%). Implement daily rebalancing.
2. **§0's annualized column is a genuine 3-year CAGR, not an extrapolation.** Its span is 1095
   days = exactly 3 years. (The extrapolation warning belongs to §2's +68% OOS headline, which is
   90 days from 23 trades — a different number. Earlier guidance conflating the two was wrong.)
3. **The buy-and-hold null is necessary but NOT sufficient, and can be toothless in a bear
   window.** On v0.2.0's 90-day OOS holdout the basket Sharpe was **−1.303**, so both new gate
   conditions (`beats_benchmark_return`, `beats_benchmark_sharpe`) **PASS trivially** and the
   verdict still fails on sample adequacy + DSR. Any phase reporting a gate verdict must state the
   benchmark's own absolute performance over the same span alongside the pass/fail bit — a
   "beats buy-and-hold" pass earned against a basket that lost money is not the protection §0 asks
   for. Phase 9 in particular must not let a bear holdout manufacture two free gate passes.
4. **The MEDIUM-5 fix eases a gate condition and that must be recorded, never buried**: spreading
   P&L across holding days measures **kurtosis 31.2449 → 15.5429, skew 3.7883 → 1.4034**, and
   **raises OOS Sharpe 1.175 → 1.512**. A correctness fix that flatters the result is exactly the
   kind of change that needs its direction logged (cf. KNOWN-LIMITATIONS §6, where a correctness
   fix *hurt* in-sample and was kept anyway).

Reproducibility target: on 2023-07-27 → 2026-07-26 this must reproduce KNOWN-LIMITATIONS §0
(BTC 2.21× / +30.3% / 0.80 / 53.0% DD; equal-weight basket 2.17× / +29.4% / 0.73 / 64.3% DD)
**from one command.** That is Phase 1's acceptance test, and it is the strongest available
check that the benchmark path is honest. Costs: buy-and-hold pays entry+exit fees once, no
funding (spot-equivalent hold) — state the assumption in the module docstring.

### MEDIUM-5, the highest-leverage fix in the project

`equity.daily_returns` books each trade's whole `pnl_pct` on its **exit day**, producing
daily skew 3.79 / kurtosis 31.24 on 23 trades, which inflates PSR's denominator and blocks
every significance test (KNOWN-LIMITATIONS §3). Phase 1 spreads each trade's P&L across the
days it was open, `entry_ts`→`exit_ts` inclusive. Both fields already exist on `Trade`; **no
schema change is needed.** Keep the old behavior reachable behind a keyword argument so the
before/after kurtosis delta is measurable rather than asserted, and record the measured delta
in the phase report.

### Trial counting under a population

The PRD's open question — "how is DSR trial-counting kept honest when a population evolves
thousands of variants per generation?" — is resolved as: **the ledger counts every
(graph, params) evaluation the oracle ever performs, across generations, persisted in
`state.db` so it survives process restarts and overnight runs.** `n_trials` passed to
`deflated_sharpe` is that cumulative count for the campaign. This will produce brutal DSR
values. That is the correct, honest answer and must not be softened; if it makes the gate
unpassable, that is the finding (see the PRD's honesty clause).

**REFINEMENT, and a corrected reading — both the sample and the search bind, and the earlier
"trial count doesn't matter" framing was too strong.** Two plans computed this independently:

- **Phase 6**: `expected_max_sharpe` grows as roughly √log N, so 12 → 9 216 trials raises the
  required annualized Sharpe only from **≈3.3 to ≈7.7**. Read alone, that suggests the search
  size is cheap and the ~90 daily observations are the whole problem — which is how
  KNOWN-LIMITATIONS §1's "fails at 0.742 even with the correction off" reads on v0.2.0's numbers.
- **Phase 9**, computing with Phase 1's *measured post-MEDIUM-5* moments (skew 1.4034, kurt
  15.5429) at the improved observed Sharpe of 1.512, over a 207-day holdout: DSR =
  **0.8829 at `n_trials=1`**, **0.0753 at 132**, **0.0090 at 3011**. Required annualized Sharpe
  6.78 (softened from 7.16 by the attribution fix).

**Reconcile them this way, and do not restate either half alone.** Post-fix, `n_trials=1` sits at
0.8829 — *near* the 0.95 threshold, not hopeless. So the multiple-testing penalty is precisely
what converts a near-pass into 0.0090: **the search size does do the killing, once the return
distribution has been repaired.** Phase 6's √log N point remains true and still forbids
"shrink the population to rescue DSR" as a strategy (the exponent is brutal in the other
direction: buying back one order of magnitude of trials is worth very little Sharpe). But the
sample is no longer the *sole* binding constraint the pre-fix numbers implied.

Consequences, binding and unchanged:
- Do **not** shrink the population, sample the ledger, or count "distinct genomes" to rescue DSR.
  It is not honest, and √log N means it barely helps.
- The honest levers are **more independent observations** (longer holdout; genuinely uncorrelated
  symbols — and §0a measures effN ≈ 1.83, so that one is nearly exhausted) and **a smaller,
  pre-registered search**. Phase 9 owns the trade-off and reports it either way.
- Record that the MEDIUM-5 fix moved DSR at `n_trials=1` from **0.742 → 0.8829**: evidence the
  repair was correct, independent of whether the gate ever passes.

Two consequences, binding:
- Do **not** propose shrinking the population, sampling the ledger, or "counting distinct
  genomes" as a way to rescue DSR. It buys almost nothing (√log N) and costs the honesty
  property the ledger exists for.
- The only honest lever is **more independent observations** — a longer holdout and/or genuinely
  uncorrelated symbols (and §0a measures that the stored majors give effN ≈ 1.83, so the second
  lever is nearly exhausted). Phase 9 owns that trade-off.

**Fitness must not be DSR.** DSR moves with `n_trials`, so a candidate scored in generation 1
is not comparable to one scored in generation 40. Use a benchmark-relative, trial-count-free
statistic (Phase 6 uses `excess_sharpe`) and keep DSR for the final gate verdict only.
Relatedly: failing `beats_benchmark_*` must **demote, not eliminate** — the v0.2.0 seed itself
fails it (+3.45% vs +29.4%), so hard elimination makes generation 0 extinct.

---

## 5. The one graph→Trade seam

Exactly one function turns a serialized strategy into trades, and it returns the **existing**
`engine.Trade` so every downstream consumer (metrics, equity, walk-forward, benchmark,
reviewers, UI) works unchanged:

```python
# framework/execute.py  (Phase 3)
def run_graph_backtest(
    conn, graph: StrategyGraph, symbol: str, *,
    start_ms: int | None = None, end_ms: int | None = None,
    fee_pct: float | None = None, slippage_pct: float | None = None,
    funding_pct_per_day: float | None = None, max_hold_bars: int | None = None,
) -> list[Trade]: ...
```

Signature deliberately mirrors `engine.run_backtest` argument-for-argument (minus `params`,
which the graph carries) so the two are drop-in interchangeable.

`walk_forward_pooled` gains one keyword-only parameter:

```python
def walk_forward_pooled(conn, symbols, *, ..., strategy: StrategyGraph | None = None)
# strategy=None  -> legacy engine.run_backtest path (all 286 existing tests unaffected)
# strategy=graph -> framework.execute.run_graph_backtest
```

**Phase 3's acceptance test is parity:** the v0.2.0 Donchian strategy expressed as a
serialized graph must reproduce `engine.run_backtest`'s trade list **exactly** — same count,
same entry/exit timestamps, same `pnl_pct` to floating-point tolerance — on stored history.
Parity is the proof the framework wraps the honest core rather than replacing it. Phase 4
onward may only extend from a green parity test.

### Trade dataclass evolution

`Trade` is frozen with positional construction used across tests. **New fields are appended
at the end and must have defaults.** Never reorder or rename existing fields. Phase 4 adds:

```python
planned_rr: float = 0.0                  # R:R after costs at entry — the audit trail
confirmations: tuple[str, ...] = ()      # names of Confirmations that passed
strategy_version: str = ""               # set by Phase 5's version registry
```

---

## 6. Persistent state: `data/state.db`

OHLCV stays in `data/ohlcv.db` with its schema untouched. Everything the framework needs to
remember goes in a **second** database so a corrupt experiment log can never endanger 117 MB
of irreplaceable price history.

```python
# data/statestore.py  (Phase 1 creates; later phases add tables)
def connect(db_path: str | None = None) -> sqlite3.Connection   # default config.STATE_DB_PATH
```
Mirrors `storage.connect()` exactly: WAL mode, `check_same_thread=False`, a module-level
`_db_lock` serializing access, and **idempotent `CREATE TABLE IF NOT EXISTS` DDL owned by the
module that uses the table** (not one central migration file). `config.STATE_DB_PATH = "data/state.db"`.

| Table | Owner | Purpose |
|---|---|---|
| `trial_ledger` | P1 `backtest/trials.py` | one row per oracle evaluation: campaign, graph hash, params hash, span, ts |
| `review_records` | P5 `feedback/records.py` | per-closed-trade TP/SL/pace review |
| `strategy_versions` | P5 `feedback/versioning.py` | version id, parent, graph JSON, created_ts, provenance |
| `generations` | P6 `evolution/population.py` | generation index, campaign, population size, best member |
| `population_members` | P6 `evolution/population.py` | member id, generation, graph hash, fitness, gate verdicts |
| `campaigns` | P6 | campaign id, seed, config, started_ts — the unit the trial ledger keys on |

All timestamps in `state.db` use the same convention as `ohlcv.db`: **epoch milliseconds, UTC**
(e.g. `1672531200000` = 2023-01-01T00:00:00Z). Never store formatted date strings.

`.gitignore` must ignore `data/state.db*` (including the `-wal`/`-shm` sidecars). It is
derived, reproducible, and must not be committed.

---

## 7. Deconflicting the three shared files

`config.py`, `cli.py`, and `walkforward.py` are touched by several phases. Rules:

**`config.py`** — each phase appends its own clearly-headed block at the end, in phase order,
following the existing style: a comment naming the phase, why the value exists, and whether it
is frozen or sweepable. Reserved constant prefixes, so no two phases collide:

| Phase | Prefix / names |
|---|---|
| 1 | `STATE_DB_PATH`, `BENCHMARK_*`, `PNL_ATTRIBUTION_*` |
| 2 | `RESEARCH_SYMBOLS`, `CORRELATION_*` |
| 3 | `STRATEGY_DIR`, `FRAMEWORK_*` |
| 4 | `MACD_*`, `VOLUME_CONFIRM_*`, `RR_TARGET_MIN` |
| 5 | `REVIEW_*`, `TARGET_ANN_RETURN` |
| 6 | `EVO_*` |
| 7 | `UI_HOST`, `UI_PORT` |
| 8 | `DETECTOR_*`, per-pattern tolerance names prefixed by pattern |
| 9 | `CAMPAIGN_*`, `HOLDOUT_*` |

Do **not** modify existing constants. `RR_FLOOR = 1.5` stays as the legacy breakout floor;
the PRD's ≥1:2 requirement lands as a **new** `RR_TARGET_MIN = 2.0` consumed by the Phase 4
filter, so the legacy path's measured behavior is not silently altered.

**`cli.py`** — one flat subcommand per phase, registered in phase order in `main()`, each
following the existing `subparsers.add_parser(...)` + `_<name>_command(...) -> int` pattern
(exit code 0 = pass). Reserved names:

| Phase | Subcommand | Handler |
|---|---|---|
| 1 | `benchmark` | `_benchmark_command` |
| 2 | `correlation-report` | `_correlation_command` |
| 3 | `plugins`, `graph-validate` | `_plugins_command`, `_graph_validate_command` |
| 4 | `graph-backtest` | `_graph_backtest_command` |
| 5 | `review` | `_review_command` |
| 6 | `evolve` | `_evolve_command` |
| 7 | `ui` | `_ui_command` |
| 8 | `detector-report` | `_detector_report_command` |
| 9 | `campaign` | `_campaign_command` |

Existing subcommands (`backfill`, `poll`, `gap-report`, `regime`, `signal`, `backtest`,
`walkforward`) keep their behavior. `walkforward` gains an optional `--graph <path>`.

**`walkforward.py`** — Phase 1 owns the gate extension (§4). Phase 3 adds only the
`strategy=` keyword (§5). Phase 6 does **not** edit this file; it wraps it. Phase 9 does not
edit it either; it drives it.

---

## 8. Test conventions and per-phase test files

Mirror `tests/test_backtest.py` — it is the reference:

- One module per area: `tests/test_<area>.py`. Class-based grouping: `class TestComputeMetrics:`.
- **Tier constants are derived from config, never hardcoded** — `SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME`,
  `D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]` — so a future tier shift cannot leave tests
  green on the old timeframes while production moves.
- `@pytest.fixture(autouse=True)` clearing module caches around every test
  (`engine.clear_caches()`); test isolation must not *depend* on a fingerprint argument being right.
- Synthetic in-memory SQLite fixtures seeded bar-by-bar; `START = 1_700_000_000_000`.
- Network tests marked `@pytest.mark.network` (deselect with `-m "not network"`).
- Hand-computed expected values for numeric code (see `tests/test_wilder.py::TestHandComputedValues`).
- Regression tests **pin** each repaired finding, with the finding id in the test name or docstring.

Reserved test files, so nine plans never claim the same path:

| Phase | New test files | Extends |
|---|---|---|
| 1 | `test_benchmark.py`, `test_trials.py`, `test_pnl_attribution.py` | `test_equity.py`, `test_backtest.py` |
| 2 | `test_correlation.py` | `test_storage.py` |
| 3 | `test_framework_contracts.py`, `test_framework_registry.py`, `test_framework_graph.py`, `test_framework_parity.py` | — |
| 4 | `test_macd.py`, `test_plugins_confirmations.py`, `test_plugins_policies.py`, `test_plugins_filters.py`, `test_pipeline_thin_slice.py` | — |
| 5 | `test_feedback_records.py`, `test_feedback_versioning.py`, `test_feedback_protocol.py` | — |
| 6 | `test_evolution_mutate.py`, `test_evolution_tournament.py`, `test_evolution_oracle.py`, `test_evolution_runner.py` | — |
| 7 | `test_ui_api.py`, `test_ui_roundtrip.py` | — |
| 8 | `test_detectors_<family>.py`, fixtures under `tests/fixtures/patterns/` | — |
| 9 | `test_campaign.py` | — |

Two shared test-tree files, assigned 2026-07-27 after the cross-plan audit found them unowned:
`tests/conftest.py` is **Phase 8's** (it needs shared pattern-fixture helpers; no other phase
claimed it), and `tests/fixtures/patterns/` is Phase 8's by §9. Any later phase needing a
conftest hook adds to Phase 8's file rather than creating a second one.

**The 286 existing tests must stay green at every phase boundary.** A phase that breaks them
is not done. Report the count in each phase's validation section.

---

## 9. Pattern coverage: tiers and honesty

`.claude/technical-pattern.md` lists ~150 patterns across 18 families with an explicit
reliability table. Ordering for Phase 8, straight from that table — **not** re-litigated per plan:

| Tier | Patterns | Reliability |
|---|---|---|
| 1 | Cup & Handle, Head & Shoulders (refined), Wyckoff Accumulation/Distribution | ⭐⭐⭐⭐⭐ |
| 2 | Double Top/Bottom, Bull/Bear Flag, Ascending Triangle, Falling Wedge, Rising Wedge, RSI Divergence | ⭐⭐⭐⭐☆ |
| 3 | Golden Cross, MACD Cross, Harmonic patterns | ⭐⭐⭐☆☆ |
| 4 | Elliott Wave, candlestick-alone | ⭐⭐☆☆☆ |

Rules binding on Phases 4 and 8:
- **Detection ≠ edge.** Every detector ships with geometry **fixture tests** (hand-built bars
  where the answer is known by construction) *and* an edge report (hit rate, expectancy after
  costs). Detector correctness is validated separately from detector profitability.
- Tiers 1–2 are the v0.3.0 commitment. Tier 3–4, harmonics, Elliott, and SMC are a
  **direction, not a gate** — the PRD says so explicitly, and no phase's success criteria may
  depend on them.
- Reuse `scripts/bruteforce/indicators.py` implementations for `double_top`, `double_bottom`,
  `bull_flag`, `bear_flag`, `triangle_squeeze`, `head_and_shoulders`,
  `inverse_head_and_shoulders`, `rsi`, `macd`, `swing_levels` (§0b).
- Wyckoff / harmonics / SMC have **no mature reference implementation** (PRD Research Summary).
  Any plan proposing them must say what precision is achievable and how correctness is checked,
  or defer them explicitly.

**OUTCOME of that rule, decided by the Phase 8 plan 2026-07-27 — tier 1 lands 2 of 3, not 3 of 3.**
**Wyckoff Accumulation/Distribution is DEFERRED.** Not because precision is low but because it is
*unmeasurable*: no agreed numeric definition, no reference implementation, no labelled dataset. In
its place Phase 8 ships the two mechanically decidable events, `wyckoff-spring` and
`wyckoff-upthrust`, whose registry `rationale` states the non-claim in words, pinned by a test. So
§9's tier-1 commitment is **Cup & Handle + H&S-refined delivered, Wyckoff acc/dist deferred**;
tier 2 lands 6 of 6. Do not read this section as promising three tier-1 detectors.

Measured catalog size, so coverage is tracked not assumed: **144 pattern rows across 18 families**
(159 table rows minus the 15-row reliability table, verified by parsing the document, with a test
asserting the ledger against it so it cannot drift). Phase 8 delivers **19/144 (13.2%)**, 6 deferred
with reasons, 119 out of scope with one of five stated reasons.

**A blocking defect found in the existing geometry, and how it is handled without breaking parity.**
`signals/patterns.py:327` reads `if upper_end > upper_start or lower_end < lower_start: return out`,
which makes **rising *and* falling wedges structurally unreachable** — a rising wedge has both
trendlines rising. Two tier-2 detectors are impossible while it stands. Phase 8 does **not** edit
`signals/patterns.py` (Phase 3 owns it as a migration target and its behavior is frozen by the
parity gate). Instead it ports the helpers into `plugins/detectors/_geometry.py` with the guard
relaxed and its work relocated into slope classification, guarded by a mutual-exclusivity test.

Accepted consequence, recorded so nobody "cleans it up" later: **two geometry implementations will
coexist** — the frozen legacy path behind `plugins/detectors/legacy_patterns.py` (whose job is
reproducing v0.2.0 exactly) and `_geometry.py` (whose job is being correct). They are allowed to
disagree. Converging them is only safe once the Phase 3 parity test is retired.

---

## 10. Resolved open questions (decided here so nine plans agree)

The PRD leaves seven open questions. Five are resolved below as **stated planning
assumptions**; two are empirical and stay open by design. Where a decision was the author's to
make rather than the PRD's, it is flagged ⚠ — the user can overturn any of these by editing
one plan, and none is load-bearing on more than the phase named.

| # | Question | Resolution |
|---|---|---|
| 1 | Is >50% ann. OOS achievable? | **Stays open.** It is the falsifiable hypothesis; Phase 9 answers it. An honest "no" is a completed outcome. |
| 2 | How many catalog patterns have edge? | **Stays open.** Phase 8's per-detector edge reports answer it empirically, tier by tier. |
| 3 | Honest DSR trial counting under a population | **Resolved** — cumulative persistent ledger, §4. |
| 4 | Which uncorrelated symbols? | **Resolved into a measurement**: 20 are already stored (§0a); Phase 2 measures pairwise correlation + effective-N and selects the low-beta subset. No new backfill unless the measurement shows the majors are all one bet. |
| 5 | Geometric/structural detectors | **Resolved by tiering** — §9. Tier 1–2 only; Wyckoff gets a precision statement or a deferral. |
| 6 | UI stack | ⚠ **Resolved: Python stdlib `http.server.ThreadingHTTPServer` + a single vanilla HTML/CSS/JS page under `ui/static/`, zero new dependencies, SSE for run/generation progress.** Rationale: single local operator on one Mac, no frontend stack exists in the repo, `pandas-ta`'s disappearance is a live lesson in dependency risk, and the PRD's own risk table says "UI scope eats the project." Rejected: FastAPI+uvicorn (two new deps for one user), React/Vite (npm toolchain for a single page), Streamlit (fights the compose-a-graph interaction). |
| 7 | Multi-position margin/liquidation accounting | ⚠ **Out of scope for all nine phases.** It is a PRD "Should" with no phase assigned, and the pivot guide's "trust the signal" principle conflicts with the engine's one-open-trade-per-symbol rule. v0.3.0 keeps **one open trade per symbol**, equal notional. Flagged to the user as the known gap between the pivot guide and this build. |

Also inherited unchanged from v0.2.0 and **not** reopened by any phase: alert-only (no order
placement), no position sizing (equal notional; `MAX_RISK_PCT` documents a human budget),
`FADE_ENABLED = False` (Phase 6 DROP verdict, kept reversible), funding as a frozen
pessimistic placeholder.

---

## 11. Dependency graph and parallelism

```
   P1 ─┬─────────────► P3 ──► P4 ─┬──► P5 ─┬──► P7 ──┬──► P9
       │  (validation      (thin  │        │         │
   P2 ─┘   integrity        slice)│        │         │
        (both independent,        ├──► P6 ─┘         │
         run concurrently)        │                  │
                                  └──► P8 ───────────┘
```

| Phase | Depends on | Runs parallel with |
|---|---|---|
| 1 Validation integrity | — | 2 |
| 2 Data breadth | — | 1 |
| 3 Plug-in framework core | 1 | — |
| 4 Strategy pipeline (thin slice) | 3 | — |
| 5 Feedback loop MVP | 4 | 6 |
| 6 Evolution engine | 4 | 5 |
| 7 Builder UI | 5, 6 | 8 |
| 8 Pattern coverage expansion | 4 | 7 |
| 9 Walk-forward campaign | 7, 8 | — |

Phase 1 gates Phase 3 for a reason worth restating: **make the measuring stick honest before
anything optimizes against it.** Phase 2 gates nothing but feeds Phase 9's breadth.

---

## 12. Definition of done, every phase

1. All 286 pre-existing tests still pass, plus the phase's own new tests.
2. `.venv/bin/python -m pytest -q` is green; `python -m py_compile` clean (no linter/type
   checker exists — say so rather than inventing a command).
3. New plug-ins register and appear in `cli.py plugins` output with a non-empty rationale.
4. **Zero engine-core edits to add a plug-in** — the framework's whole reason to exist. Verified
   by code review of the phase's added plug-ins (PRD success metric).
5. Every number quoted in a phase report is **measured by a committed command**, never derived
   or estimated. This repo has already committed once to using measured rather than derived
   figures (`git log`: "Use measured rather than derived figures in the benchmark table") —
   keep that discipline.
6. Degrees of freedom consumed by the phase are recorded, per the trial-log discipline
   (KNOWN-LIMITATIONS §9).

---

## 13. Source map for plan authors

| Need | Read |
|---|---|
| PRD | `.claude/PRPs/prds/self-learning-pattern-framework.prd.md` |
| Why v0.3.0 exists | `.claude/pivot-guide.md` |
| Pattern catalog + reliability tiers | `.claude/technical-pattern.md` |
| Honest accounting of v0.2.0 | `.claude/PRPs/reports/KNOWN-LIMITATIONS.md` |
| Plan format reference | `.claude/PRPs/plans/v0.2.0/phase7-validation-protocol-repair-gate.plan.md` |
| Registry prior art | `scripts/bruteforce/registry.py` |
| Indicator/detector donors | `scripts/bruteforce/indicators.py` |
| Vectorized sim + splits + causality | `scripts/bruteforce/core.py` |
| Research universe + selection rule | `scripts/bruteforce/universe.py` |

---

*Authored 2026-07-27 as the coherence contract for the nine v0.3.0 phase plans.*
