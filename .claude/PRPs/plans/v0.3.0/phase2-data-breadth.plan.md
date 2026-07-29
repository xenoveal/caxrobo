# Plan: Data Breadth — measure the correlation trap, do not assume it away (v0.3.0 Phase 2)

## Summary

**The PRD's premise for this phase is stale, and the correction is the headline.** The PRD scopes
Phase 2 as "backfill uncorrelated Binance futures symbols", assuming 3 symbols are stored.
**20 are stored** — every one at 1d/4h/1h over 2023-01-01 → 2026-07-26, loaded by the
`scripts/bruteforce/` side effort (contract §0a). Phase 2 is therefore **verify → measure →
select → promote → report**, not backfill.

Deliverables: `src/trading_bot/data/correlation.py` (pairwise daily-return correlation matrix +
a stated **effective independent sample size**), a `correlation-report` CLI subcommand, a
gap-integrity sweep over the stored universe via the existing `storage.find_gaps`, a
pre-registered selection rule producing `config.RESEARCH_SYMBOLS`, and a markdown report artifact
under `.claude/PRPs/reports/`.

**The measurement was run once during planning and the answer is uncomfortable.** Span
2023-07-27 → 2026-07-26, 1095 daily returns, zero missing bars:

| Set | n | mean pairwise r | effective N (Kish) | effective N (participation ratio) |
|---|---|---|---|---|
| `config.SYMBOLS` (BTC/ETH/SOL) | 3 | **0.7574** | **1.193** | 1.395 |
| all 20 stored | 20 | 0.5905 | 1.637 | 2.505 |
| best low-r subset of 8 | 8 | 0.4689 | 1.868 | 2.983 |
| pre-registered selection (BTC + lowest-8) | 9 | 0.4885 | 1.834 | 2.925 |

BTC/ETH **+0.809**, BTC/SOL **+0.744**, ETH/SOL **+0.719**, mean r 0.7574 → effective N
**1.193 ≈ 1.2**, reproducing KNOWN-LIMITATIONS §0b's anchors exactly. That reproduction is this
phase's correctness check and it passes.

The honest finding, which this plan makes a **first-class reportable result rather than a failure
to paper over**: 3 → 20 stored symbols multiplies rows by **6.7×** and independent information by
about **1.5×**. All 20 are liquid majors; 19 carry BTC correlation between +0.56 and +0.81, and
only TRXUSDT (+0.222) is meaningfully detached. §0b's trap is **mitigated, not solved**.

## User Story

As the bot's sole operator, I want the stored universe measured for pairwise correlation,
effective independent sample size, BTC beta, liquidity and gap integrity, and the
low-correlation subset promoted into config as `RESEARCH_SYMBOLS`, so Phase 9 pools over a
universe whose statistical independence is a **measured number** — and so "we added symbols" can
never again be mistaken for "we added information."

## Problem → Solution

**Current**: `config.SYMBOLS` holds 3 symbols at ~0.76 correlation → ~1.2 effective independent
samples; 23 OOS trades cannot clear a floor of 30 (KNOWN-LIMITATIONS §1, §4). 17 extra symbols
already sit in `data/ohlcv.db` but nothing in `src/trading_bot/` knows they exist, nothing has
gap-verified them, and nothing has measured whether they add information or just rows.

**Solution**: `data/correlation.py` measures the stored universe on daily returns from the 1d
series; a pre-registered rule (liquidity floor + BTC-beta ceiling + rank by mean pairwise r, BTC
pinned) selects ≥8 symbols; the selection is promoted to `config.RESEARCH_SYMBOLS` with
production `config.SYMBOLS` **untouched**, exactly as `scripts/bruteforce/universe.py:3-6`
already argues; `correlation-report` emits the matrix, the effective-N table and the verdict as a
committed artifact.

## Metadata

- **Complexity**: **Medium** — 1 new module (~330 lines), 2 shared files appended to, 2 test
  files (1 new, 1 extended), 1 generated report. The work is in the statistics being *stated*
  rather than assumed, and in the decision rule for a negative result.
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md`, Phase 2
- **Depends on**: nothing. Parallel with Phase 1. Gates nothing; feeds Phase 9's breadth.
- **Estimated Files**: 6 (2 CREATE, 3 UPDATE, 1 generated artifact)
- **Binding contract**: `_shared-architecture-contract.md` §0, §0a, §1, §2, §7, §8, §11, §12

---

## THE CORRECTION, stated prominently

| The PRD says | Verified 2026-07-27 | Consequence |
|---|---|---|
| "Backfill uncorrelated Binance futures symbols" | 20 symbols stored at 1d/4h/1h, full span, **zero interior gaps in 60 cells** | No backfill performed. It becomes a contingency gated on decision D3 (Task 14) |
| KNOWN-LIMITATIONS §4: "Only 3 symbols exist in the store" | True at v0.2.0 merge; `bruteforce/backfill_universe.py` has since loaded 17 more | §4 is superseded by contract §0a. This plan **records** the supersession; it does not edit that file |
| Open Question #4: "are enough uncorrelated symbols available to break the ~0.76 trap?" | **Measured answer: no.** Best achievable effective N from the stored liquid universe is ~1.9, not ~8 | The negative answer is a deliverable. Task 13 writes it down; Task 14 states what would justify a real backfill |
| Success signal: "≥8 symbols with correlation materially below the BTC-beta cluster" | ≥8 *is* achievable (mean r 0.47 vs 0.76 — materially below); the **information** gain is only ~1.5× | Criterion met on its literal terms, and still reported as a partial win. Both facts go in the report |

Where PRD and contract disagree, the contract wins. This correction repeats in Notes and opens
the report artifact.

---

## Measured ground truth (2026-07-27, this working tree, read-only)

Every number was produced by a command run while writing this plan.

**Environment**: Python 3.11.6 · pandas **3.0.3** / numpy 2.4.6 · pytest **286 tests collected**
(`.venv/bin/python -m pytest --collect-only -q`) · `fapi.binance.com/fapi/v1/ping` → **HTTP 200
in 0.19 s** · 256 GiB free on `/Users/ttaa`.

**The store** — `data/ohlcv.db`, 116,961,280 bytes (page_size 4096 × page_count 28,555):

| timeframe | symbols | rows | rows/symbol | min ts | max ts |
|---|---|---|---|---|---|
| `1d` | 20 | 26,080 | 1,304 | 1672531200000 (2023-01-01Z) | 1785110400000 (2026-07-26Z) |
| `4h` | 20 | 156,403 | ~7,820 | 1672516800000 | 1785124800000 |
| `1h` | 20 | 625,601 | ~31,280 | 1672506000000 | 1785132000000 |
| `15m` | **3** (BTC/ETH/SOL) | 375,288 | 125,096 | — | — |
| **total** | | **1,183,372** | | | |

**Storage cost, computed not guessed: 116,961,280 / 1,183,372 = 98.84 bytes/row** (PK is
`(symbol, timeframe, ts)`, `storage.py:68`).

| Hypothetical expansion | rows | MB | resulting DB |
|---|---|---|---|
| +1 symbol at 1d+4h+1h | 40,404 | **4.0** | 121 MB |
| +10 symbols at 1d+4h+1h | 404,040 | **39.9** | 157 MB (+34%) |
| 15m for the 17 symbols lacking it | 2,126,632 | **210** | 327 MB (+180%) |

**Storage is not a constraint** — 256 GiB free against a worst case of ~330 MB. The PRD's
"capacity check" is answered "unconstrained, and here is the coefficient." The binding constraint
on breadth is **correlation, not disk.**

**Gap integrity, measured over all 60 research cells** with `now_ms` pinned per cell to
`last_ts + interval + 1` so trailing staleness cannot masquerade as a hole:
**`interior-gap cells: 0 of 60`** (20 symbols × {1d, 4h, 1h}).

At wall-clock `now`, 40 of 60 cells plus all 3 `15m` cells report a *trailing* gap (1h lag ~5 h;
4h lag 1 bar). That is poller lag against `STALENESS_INTERVALS = 2`, **not missing history** —
which is why the integrity check must separate the two exactly as `storage.find_gaps` already
does: interior gaps come from the `zip(timestamps, timestamps[1:])` loop at `storage.py:179-181`,
the trailing gap from the staleness branch at `storage.py:188-192`.

**The correlation measurement** (1096 aligned 1d closes → 1095 daily returns, **zero NaN across
all 20 symbols** — the series share an identical timestamp set, so listwise alignment discards
nothing):

| symbol | mean pairwise r | r vs BTC | BTC beta | median daily quote volume |
|---|---|---|---|---|
| TRXUSDT | **0.231** | 0.222 | 0.327 | $67.3 M |
| BCHUSDT | 0.519 | 0.615 | 1.087 | $158.5 M |
| BNBUSDT | 0.555 | 0.661 | 0.739 | $397.3 M |
| XRPUSDT | 0.565 | 0.627 | 1.027 | $840.4 M |
| UNIUSDT | 0.570 | 0.580 | 1.285 | $90.9 M |
| OPUSDT | 0.593 | 0.560 | 1.217 | $106.9 M |
| AAVEUSDT | 0.594 | 0.623 | 1.230 | $109.2 M |
| FILUSDT | 0.595 | 0.567 | 1.227 | $128.0 M |
| LTCUSDT | 0.597 | 0.612 | 0.911 | $168.2 M |
| NEARUSDT | 0.601 | 0.621 | 1.378 | $135.5 M |
| APTUSDT | 0.606 | 0.624 | 1.207 | $92.9 M |
| SOLUSDT | 0.615 | 0.744 | 1.329 | $2,610.9 M |
| BTCUSDT | 0.629 | 1.000 | 1.000 | $13,720.5 M |
| ADAUSDT | 0.630 | 0.687 | 1.338 | $252.9 M |
| DOGEUSDT | 0.632 | 0.762 | 1.433 | $709.4 M |
| ATOMUSDT | 0.636 | 0.618 | 1.007 | $51.7 M |
| LINKUSDT | 0.651 | 0.676 | 1.232 | $238.5 M |
| AVAXUSDT | 0.652 | 0.696 | 1.321 | $233.9 M |
| ETHUSDT | 0.668 | 0.809 | 1.140 | $8,753.2 M |
| DOTUSDT | 0.671 | 0.650 | 1.142 | $123.5 M |

Full-span cross-check (2023-01-01 →, 1303 returns): BTC/ETH 0.812, BTC/SOL 0.723, ETH/SOL 0.707,
core r̄ 0.7473 → effective N 1.203. The core anchor is stable to ±0.01 across both spans, so the
reproduction check is not span-fragile.

**Selection-rule robustness**: ranking by mean pairwise r and a greedy minimum-average-correlation
search agree on **7 of 8** members (differing only AAVE ↔ NEAR) with effective N differing by
0.009 (1.868 vs 1.877). The selection is insensitive to the algorithm — which is why Task 6
specifies the simple deterministic ranking: no path dependence, no hidden degree of freedom.

### What breadth actually buys — two different things, kept apart

This distinction is the analytical core of the phase and must survive into the report.

1. **Mechanical sample adequacy** — the gate's `n_trades >= config.WF_MIN_TRADES` (30). At
   v0.2.0's measured rate (23 OOS trades / 90 days / 3 symbols = 0.0852 trades/symbol/day), a
   9-symbol pool yields **≈69 trades per 90-day holdout**. Phase 2 fixes this condition.
   *Labelled an ESTIMATE at the v0.2.0 rate; the actual rate is Phase 9's to measure.*
2. **Statistical independence** — what the Deflated Sharpe needs. Deflating by effective N:
   v0.2.0 had 23 × (1.193/3) ≈ **9 independent-equivalent trades**; the 9-symbol selection gives
   69 × (1.834/9) ≈ **14**. Better; still under 30 in independent terms.

Row count clears the gate's arithmetic. Information does not clear the statistics. Neither is
allowed to stand in for the other.

---

## UX Design

Internal / CLI change. No web surface exists yet (Phase 7 owns that).

**Before**: `gap-report` covers 3 symbols × 4 timeframes; nothing in `src/trading_bot/` knows the
other 17 stored symbols exist, and no command anywhere measures correlation.

**After**:
```
$ python -m trading_bot.cli correlation-report
exchange: (not checked; pass --check-exchange)
span: 2023-07-27 -> 2026-07-26   timeframe: 1d   returns: 1095   symbols: 20

integrity (interior gaps over 1d/4h/1h):
  20 of 20 symbols clean, 0 interior gaps in 60 cells

effective independent sample size (Kish, r_bar-based):
  config.SYMBOLS   n=3   r_bar=0.7574   N_eff=1.193   (participation ratio 1.395)
  all stored       n=20  r_bar=0.5905   N_eff=1.637   (participation ratio 2.505)
  selection        n=9   r_bar=0.4885   N_eff=1.834   (participation ratio 2.925)

selection (rule: liquidity floor, beta ceiling, rank by mean pairwise r, BTC pinned):
  BTCUSDT TRXUSDT BCHUSDT BNBUSDT XRPUSDT UNIUSDT OPUSDT AAVEUSDT FILUSDT

VERDICT: D1  N_eff ratio 1.54x vs config.SYMBOLS (threshold 1.50x)
wrote .claude/PRPs/reports/phase2-correlation-report.md
```

| Touchpoint | Before | After | Notes |
|---|---|---|---|
| `cli.py` subcommands | 7 | 8 (`correlation-report`) | Contract §7 reserved name → `_correlation_command` |
| `config.SYMBOLS` | 3-tuple | **unchanged** | Production product stays 3 symbols. Non-negotiable |
| config symbol sets | 1 | 2 (`SYMBOLS`, `RESEARCH_SYMBOLS`) | Mirrors `universe.py`'s `CORE` / `UNIVERSE` split |
| Reports dir | no correlation artifact | `phase2-correlation-report.md` | Written by `--out`, mirroring `build_performance_chart.py:690` |
| `gap-report` | 12 lines | **unchanged** | Contract §7. Universe-wide integrity lives inside `correlation-report` |

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `_shared-architecture-contract.md` | §0, §0a, §2, §7, §8, §12 | Binding. §0a overturns the PRD's premise; §7 reserves `RESEARCH_SYMBOLS`/`CORRELATION_*` and `correlation-report`/`_correlation_command`; §8 reserves `test_correlation.py` |
| P0 | `src/trading_bot/data/storage.py` | 117-194 | `find_gaps` — the INVARIANTS at **140-146**, interior-gap loop 179-181, staleness branch 188-192, `expected_end = (now_ms // interval) * interval - interval` at 166. Mirror the invariants; never loosen them |
| P0 | `src/trading_bot/data/storage.py` | 205-244 | `load_candles` — `list[tuple]` of `(ts,open,high,low,close,volume)`, **both bounds inclusive**, `ts ASC`. The only sanctioned read path. Note it does **not** take `_db_lock` while `find_gaps` does — do not "fix" that |
| P0 | `src/trading_bot/data/storage.py` | 22-32 | `TIMEFRAME_MS` (`1d`: 86_400_000), `_db_lock` |
| P0 | `scripts/bruteforce/universe.py` | 1-43 | The prior art this phase formalizes: research-vs-production split (3-6), the **recorded selection-rule docstring** (8-12) whose style Task 6 must match, sector labels, why 15m is excluded (40-43) |
| P0 | `.claude/PRPs/reports/KNOWN-LIMITATIONS.md` | §0b, §4 | The anchors to reproduce (0.809 / 0.744 / 0.719 → ~1.2) and the backfill mandate §0a supersedes. §4's "only 3 symbols" is stale — record, do not edit |
| P1 | `src/trading_bot/cli.py` | 237-264 | `_gap_report_command` — the exact shape for `_correlation_command`: keyword-only time args, one line per cell, `return 1 if has_gaps else 0` |
| P1 | `src/trading_bot/cli.py` | 39-56, 64-77, 176-209, 215-234 | `add_parser` style, `type=_date_arg` (epoch-ms at parse time, clean exit 2 on bad dates), the `elif args.command` chain, `conn.close()` then `sys.exit(exit_code)` |
| P1 | `src/trading_bot/config.py` | 10-21, 153-183 | `SYMBOLS`/`TIMEFRAMES`/`BACKFILL_START`/`STALENESS_INTERVALS`; 153-183 is the **appended-phase-block** comment style (ruled header, phase named, why the value exists, what does *not* read it) |
| P1 | `src/trading_bot/backtest/engine.py` | 177-182, 185-206 | `_df`'s canonical frame shape; `_assert_interval`'s rationale (a wrongly-keyed series fails *silently in the flattering direction*). Both are private — mirror, never import (contract §2) |
| P1 | `tests/test_storage.py` | 1-31, 98-133, 210-275, 373-411 | Module constants (`SYMBOL`, `TF`, `INTERVAL`, `START = 1_700_000_000_000`), `make_rows`, the `check_invariants` helper at **112-116**, `class TestStalenessDetection`, the v0.2.0 1d gap tests |
| P1 | `tests/test_binance_client.py` | 146-170 | The **only** network-test pattern in the repo. Copy verbatim |
| P2 | `src/trading_bot/backtest/equity.py` | 1-63 | Pure-computation module docstring conventions, `DAY_MS = 86_400_000` at 22, and the "return `None`, never a fake number" convention at 54-63 |
| P2 | `scripts/build_performance_chart.py` | 687-729 | The live report-writing reference: `--out` + `open(..., encoding="utf-8")` + one `logger.info`. (`build_review_chart.py` is stale and will not run — KNOWN-LIMITATIONS §8) |
| P2 | `scripts/bruteforce/backfill_universe.py` | 1-60 | The D3 contingency path: reuses `backfill_series` verbatim, idempotent, one line per cell |
| P2 | `src/trading_bot/data/backfill.py` | 130-166 | `backfill_all(conn, *, exchange, symbols, timeframes, start_ms)` already accepts an arbitrary symbol list — a contingency backfill needs zero code change |
| P2 | `pyproject.toml` | 25-29 | Exactly one marker, `network`. No linter, no type checker, no `Makefile` |

## External Documentation

| Topic | Source | Key takeaway |
|---|---|---|
| Kish design effect | Standard survey-statistics result: `N_eff = m / (1 + (m−1)·r̄)` | Reproduces §0b's published 1.2 exactly at m=3, r̄=0.7574 → 1.193. **Verified numerically during planning.** No library needed |
| Participation ratio | `N_eff = (Σλ)² / Σλ²` over correlation-matrix eigenvalues | Secondary, less pessimistic figure that respects block structure. `numpy.linalg.eigvalsh` (numpy 2.4.6, already present). Measured: core3 1.395, selection 2.925 |
| pandas 3.0 `pct_change` | Installed pandas is **3.0.3** | Fill behaviour changed in 3.x. **Do not rely on the default** — compute `close / close.shift(1) - 1.0`. Verified `pd.Series([1,2,4]).pct_change() → [nan,1.0,1.0]`, correct today, but explicit arithmetic is version-proof |
| Binance reachability | `GET https://fapi.binance.com/fapi/v1/ping` | **HTTP 200, 0.19 s, 2026-07-27.** Blocked 2026-07-05, reachable 2026-07-26 and again today. Reached only via `ccxt` here |

**No new dependency.** pandas/numpy are present; `statistics`/`math` cover the rest. Contract §1's
"add no new indicator dependency" discipline applies — `pandas-ta`'s disappearance from PyPI is
the live lesson.

---

## Patterns to Mirror

Every snippet is copied from a real file at the stated lines.

### RECORDED_SELECTION_RULE_DOCSTRING
```python
# SOURCE: scripts/bruteforce/universe.py:1-13 — the exact register Task 6 must write in.
"""Research universe for the brute-force strategy search.

Deliberately separate from ``trading_bot.config.SYMBOLS``: production stays a
3-symbol product while research explores a wider universe. ...

Selection rule (recorded so it is auditable, not re-litigated later): the 20
highest-liquidity USDT-M perps that (a) have full history and (b) span
distinct sectors -- L1 majors, L2, DeFi, payments, memecoin, storage -- so
cross-sectional strategies see genuine dispersion rather than 20 proxies for
beta.
"""
```

### FRAME_FROM_STORAGE
```python
# SOURCE: src/trading_bot/backtest/engine.py:177-182 — canonical OHLCV frame shape.
# PRIVATE to engine.py; mirror it, do NOT import it (contract §2).
def _df(conn, symbol: str, timeframe: str) -> pd.DataFrame:
    rows = storage.load_candles(conn, symbol, timeframe)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if len(df):
        df["ts"] = df["ts"].astype(int)
    return df.set_index("ts")
```

### SILENT_MISALIGNMENT_MUST_RAISE
```python
# SOURCE: src/trading_bot/backtest/engine.py:186-197 — the reasoning Task 3 mirrors.
"""Verify a loaded series' bar spacing matches its configured timeframe.
... If config and stored data disagree — a partial tier migration, a series
backfilled under the wrong key, a fixture seeded at the old interval — all of
them fail SILENTLY, in the flattering direction. Raise instead.

Uses the MEDIAN inter-bar spacing so a handful of missing candles cannot
trip the check; a wholesale interval mismatch always shifts the median.
"""
```

### NONE_NOT_A_FAKE_NUMBER
```python
# SOURCE: src/trading_bot/backtest/equity.py:54-58 — insufficient data returns None,
# never 0.0. Every correlation / beta / effective-N function follows this.
def sharpe_ratio(returns: list[float], periods_per_year: int = PERIODS_PER_YEAR) -> float | None:
    """Annualized Sharpe (risk-free rate 0). None if < 2 obs or zero variance."""
    if len(returns) < 2:
        return None
```

### CLI_REPORT_COMMAND
```python
# SOURCE: src/trading_bot/cli.py:237-264 — keyword-only time args, one printed line per
# cell, exit 1 when the report found something wrong. Copy this contract exactly.
def _gap_report_command(
    conn, *, now_ms: int | None = None, start_ms: int | None = None
) -> int:
    has_gaps = False
    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            gaps = find_gaps(conn, symbol, timeframe, now_ms=now_ms, start_ms=start_ms)
            if gaps:
                gap_ranges = [f"{g[0]}-{g[1]}" for g in gaps]
                print(f"{symbol} {timeframe}: {len(gaps)} gap(s): {gap_ranges}")
                has_gaps = True
            else:
                print(f"{symbol} {timeframe}: OK")
    return 1 if has_gaps else 0
```

### CLI_SUBPARSER_AND_DISPATCH
```python
# SOURCE: src/trading_bot/cli.py:64-71 (parser) and 176-181 (dispatch).
    gap_report_parser = subparsers.add_parser(
        "gap-report", help="Report gaps in stored OHLCV data"
    )
    gap_report_parser.add_argument(
        "--as-of", type=_date_arg,
        help="Report gaps as of this UTC date (YYYY-MM-DD); default is now",
    )
    ...
    elif args.command == "gap-report":
        conn = connect(args.db)
        # args.as_of / args.start are already epoch-ms (or None) thanks to type=_date_arg
        exit_code = _gap_report_command(conn, now_ms=args.as_of, start_ms=args.start)
        conn.close()
        sys.exit(exit_code)
```

### CONFIG_APPENDED_PHASE_BLOCK
```python
# SOURCE: src/trading_bot/config.py:153-160 — ruled header, phase named, WHY the value
# exists, and an explicit statement of what does NOT read it.
# ---------------------------------------------------------------------------
# Phase 2: honest cost & risk model. Stop distance is now derived from
# volatility (ATR), never from a fixed percentage of entry. MAX_RISK_PCT
# below is retained ONLY as documentation of the account-risk budget the
# human discharges via position sizing (Phase 8) — it is not read by any
# signal or filter code from this phase forward.
# ---------------------------------------------------------------------------
MAX_RISK_PCT = 0.005  # ACCOUNT-RISK BUDGET (human sizing), NOT a stop-distance rule
```

### CONFIG_TUPLE_SHAPE
```python
# SOURCE: src/trading_bot/config.py:10 and scripts/bruteforce/universe.py:15-16 —
# annotated module-level tuples; universe.py adds the "never reorder" contract.
SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
# The three the PRD gate is defined on. Never reorder: reports key off this.
CORE = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
```

### TEST_MODULE_SHAPE
```python
# SOURCE: tests/test_storage.py:9-24 — module constants derived from the real maps, a
# synthetic-row factory, arbitrary epoch base. Reuse in test_correlation.py.
SYMBOL = "BTCUSDT"
TF = "15m"
INTERVAL = storage.TIMEFRAME_MS[TF]
START = 1_700_000_000_000  # arbitrary epoch-ms base


def make_rows(n, start=START, interval=INTERVAL, close=100.0):
    return [
        [start + i * interval, 99.0, 101.0, 98.0, close, 10.0]
        for i in range(n)
    ]
```

### GAP_INVARIANT_TEST
```python
# SOURCE: tests/test_storage.py:112-116 — the invariant sweep helper. Task 10's
# research-grid tests reuse this exact assertion triple, with INTERVAL per timeframe.
    def check_invariants(gaps):
        for gap_start, gap_end in gaps:
            assert gap_start <= gap_end
            assert gap_start % INTERVAL == 0
            assert gap_end % INTERVAL == 0
```

### NETWORK_TEST_GATING
```python
# SOURCE: tests/test_binance_client.py:146-156 — the repo's ONLY network-test pattern.
@pytest.mark.network
@pytest.mark.skipif(
    not os.getenv("RUN_NETWORK_TESTS"),
    reason="Network tests disabled by default; set RUN_NETWORK_TESTS=1 to enable",
)
def test_fetch_ohlcv_page_network_smoke():
    """Network smoke test: fetch BTC/USDT:USDT 15m candles from live Binance.

    This test is skipped by default unless RUN_NETWORK_TESTS=1 is set.
    """
```

### HAND_COMPUTED_NUMERIC_TEST
```python
# SOURCE: tests/test_wilder.py:325-334 — numeric code is tested against values a human
# computed, with the arithmetic shown in a comment. test_correlation.py does the same.
class TestHandComputedValues:
    """Test against hand-computed reference values for correctness."""

    def test_tr_hand_computed(self):
        """True Range matches hand-computed values."""
        rows = [
            [START, 100.0, 102.0, 98.0, 100.0, 10.0],      # TR = 4
            [START + INTERVAL, 100.0, 105.0, 99.0, 102.0, 10.0],  # TR = max(6, 5, 1) = 6
        ]
```

### IDEMPOTENT_UNIVERSE_BACKFILL
```python
# SOURCE: scripts/bruteforce/backfill_universe.py:36-48 — the D3 contingency path.
# Reuses backfill_series verbatim; adds only the loop over a wider symbol tuple.
    for i, symbol in enumerate(UNIVERSE, 1):
        for timeframe in RESEARCH_TIMEFRAMES:
            result = backfill_series(conn, symbol, timeframe)
            have = conn.execute(
                "SELECT COUNT(*), MIN(ts), MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
            status = "ok" if result.complete else f"INCOMPLETE({result.reason})"
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/data/correlation.py` | **CREATE** | Contract §2 assigns this exact path to Phase 2 |
| `src/trading_bot/config.py` | UPDATE (append) | Phase 2's reserved block: `CORRELATION_*` (Task 1) + `RESEARCH_SYMBOLS` (Task 12). Appended at end; no existing constant edited (contract §7) |
| `src/trading_bot/cli.py` | UPDATE | `correlation-report` subparser + `_correlation_command`, registered after Phase 1's `benchmark` (contract §7) |
| `tests/test_correlation.py` | **CREATE** | Contract §8 reserves this name. Offline except one `@pytest.mark.network` smoke |
| `tests/test_storage.py` | UPDATE (append) | Contract §8's sanctioned "deliberate addition": research-grid gap invariants |
| `.claude/PRPs/reports/phase2-correlation-report.md` | **CREATE** (generated) | The report artifact, written by `--out`. Committed, because its numbers are the evidence Phase 9 keys off |

## NOT Building

- **No backfill of the stored 20.** They are complete, gap-free, full-span (measured). Contract
  §0a: "Do not spend a day re-downloading data you have."
- **No change to `config.SYMBOLS`.** Widening it would silently change what `gap-report`,
  `regime`, `signal`, `backtest`, `walkforward` and `build_performance_chart.py` iterate.
- **No change to `gap-report`.** Contract §7: existing subcommands keep their behavior. The
  universe-wide integrity sweep lives inside `correlation-report`, where it belongs anyway
  (a gap is a *selection criterion*). *Rejected alternative*: `--symbol`/`--research` flags on
  `gap-report` — the contract permits exactly one flag addition to an existing subcommand across
  all of v0.3.0 (`walkforward --graph`), and a second would erode that discipline for no gain.
- **No new `scripts/` file.** The reserved surface is the CLI subcommand; the markdown renderer
  is a pure `format_report(report) -> str` in `correlation.py`, so the artifact is reproducible
  from the same command that produced the numbers.
- **No 15m backfill for the 17 symbols lacking it.** Measured 210 MB, and the tier was retired
  on cost-frontier grounds (`universe.py:40-43`). Extension point recorded, not built.
- **No correlation on 4h or 1h returns.** §0b's anchors are daily figures; higher-frequency
  correlation is a different, microstructure-contaminated quantity. `--timeframe` exists and
  defaults to `config.CORRELATION_TIMEFRAME = "1d"`.
- **No rolling / regime-conditional correlation.** Genuinely informative (correlation spikes in
  crashes) but it needs a window parameter — a degree of freedom this phase has no budget for.
  Recorded in Notes as the obvious follow-up.
- **No cross-sectional strategy, BTC-beta neutralisation, or relative-strength signal.**
  KNOWN-LIMITATIONS §0c lists these as unexplored; they are Phase 4/8 strategy work. This phase
  only makes the data for them selectable.
- **No `state.db` usage** — Phase 1's file (contract §6). Nothing here needs persistence.
- **No edits to** `storage.py`, `backfill.py`, `engine.py`, `walkforward.py`, `equity.py`,
  `poller.py`, `binance_client.py`, or anything under `signals/`, `regime/`, `indicators/`.
- **No editing of the PRD, the shared contract, or KNOWN-LIMITATIONS.** §4's stale claim is
  *recorded* as superseded in the report and in Notes.

---

## Step-by-Step Tasks

### Task 1: Append the Phase 2 config block (`CORRELATION_*` knobs)

- **ACTION**: Append a ruled block at the end of `src/trading_bot/config.py`'s constants
  (after `COST_RATIO_CEILING`, line 183, before `date_to_ms` at 186).
- **IMPLEMENT**: A block headed `# v0.3.0 Phase 2: data breadth`, recording that
  `RESEARCH_SYMBOLS` is a **separate, wider** universe from `SYMBOLS` (production stays a
  3-symbol product, per `universe.py:3-6`), that **nothing in the live signal path reads it**
  (every `cli.py` branch defaults to `SYMBOLS`), that Phase 9 is the intended consumer, and that
  correlation is measured on **daily returns from the 1d series over the benchmark span** so the
  Phase 1 and Phase 2 reports are comparable. Constants:
  - `CORRELATION_TIMEFRAME = "1d"` — frozen; §0b's anchors are daily figures
  - `CORRELATION_START = "2023-07-27"` — the benchmark span start (the first date the 1D regime
    tier's 207-calendar-day warmup permits a non-`uncertain` label), **not** `BACKFILL_START`
  - `CORRELATION_MIN_OVERLAP_BARS = 365` — fewer aligned bars ⇒ r reported as `None`
  - `CORRELATION_ANCHOR_SYMBOL = "BTCUSDT"` — beta reference, pinned into every selection
  - `CORRELATION_SELECT_N = 8` — non-anchor symbols; the PRD's success floor is ≥8
  - `CORRELATION_MIN_QUOTE_VOLUME_USD = 50_000_000.0` and `CORRELATION_MAX_BTC_BETA = 1.35` —
    candidate screens, with a comment recording that **both are non-binding on the 20 currently
    stored** (measured minimum median quote volume $51.7 M = ATOMUSDT; maximum beta among
    selected 1.285 = UNIUSDT). They are pre-registered so a future low-correlation expansion
    cannot quietly buy correlation reduction with illiquidity the 2 bps slippage assumption
    cannot support, or with leveraged BTC proxies (DOGE 1.433, NEAR 1.378) whose drawdowns
    compound BTC's.
  - `CORRELATION_EFFECTIVE_N_MIN_RATIO = 1.5` — decision rule D1's threshold; a pre-registered,
    deliberately modest bar (measured expectation 1.54×)
- **MIRROR**: CONFIG_APPENDED_PHASE_BLOCK, CONFIG_TUPLE_SHAPE.
- **IMPORTS**: none.
- **GOTCHA**: `RESEARCH_SYMBOLS` is deliberately **not** defined yet — it is the *output* of the
  measurement and lands in Task 12. Defining it now with a guessed membership is exactly the
  "derived rather than measured figure" the repo already committed against (`git log`: "Use
  measured rather than derived figures in the benchmark table"). Task 3's default symbol set is
  discovered from the DB, so nothing depends on it existing earlier. **Do not touch any existing
  constant** — `SYMBOLS`, `TIMEFRAMES`, `RR_FLOOR`, `WF_MIN_TRADES` stay exactly as they are
  (contract §7).
- **VALIDATE**:
  ```bash
  .venv/bin/python -c "
  from trading_bot import config
  assert config.SYMBOLS == ('BTCUSDT','ETHUSDT','SOLUSDT'), config.SYMBOLS
  assert config.date_to_ms(config.CORRELATION_START) == 1690416000000
  assert 1690416000000 % 86_400_000 == 0
  print('config block OK; SYMBOLS untouched')"
  ```

### Task 2: Create `data/correlation.py` — docstring, constants, dataclasses

- **ACTION**: Create `src/trading_bot/data/correlation.py`.
- **IMPLEMENT**: Module docstring carrying (a) the phase, (b) the **stated effective-N formula
  and why that one**, (c) the alignment policy, (d) the honest headline. Specifically:
  - Returns are `r_t = close_t / close_{t-1} - 1` on the 1d close series. Daily bars are
    UTC-midnight aligned (`ts % 86_400_000 == 0`), so calendar alignment is an inner join on `ts`
    and needs no resampling. Series are aligned **listwise** (complete cases only) so the matrix
    stays positive semi-definite and every pair is measured on the same `n_obs`, which is always
    reported alongside r.
  - Primary effective-N figure: **Kish-style design effect**
    `N_eff = m / (1 + (m − 1) · r̄)`, r̄ = mean off-diagonal Pearson r. Chosen because it
    produces §0b's published anchor (m=3, r̄=0.7574 → 1.193 ≈ 1.2), needs only r̄ (stable at
    ~1000 obs), is monotone in r̄, and is the **conservative** candidate. Weakness stated
    honestly: the exchangeability assumption means block structure (TRX at 0.23 against a ~0.6
    cluster) makes it **understate** independence.
  - Secondary: **participation ratio** `(Σλ)² / Σλ²`. Reported, never used by the decision rule,
    because a conservative statistic is the right one to bet a validation gate on.
  - "Pure computation except for reads through `storage.load_candles` / `storage.find_gaps`.
    Writes nothing; the CLI owns file output."

  Then module constants — `DAY_MS = 86_400_000` and
  `INTEGRITY_TIMEFRAMES = ("1d", "4h", "1h")` (15m excluded deliberately: it exists for
  BTC/ETH/SOL only and the tier was retired, so requiring it would disqualify 17 symbols for
  lacking data no v0.3.0 strategy reads) — and two frozen dataclasses with Attributes docstrings:
  - `SymbolStats(symbol, n_bars, mean_r, btc_beta, median_quote_volume, interior_gaps, eligible,
    reason)`. Document `median_quote_volume` as an **OHLCV-derived liquidity proxy, not book
    depth**, since it is used as a screen.
  - `CorrelationReport(timeframe, start_ms, end_ms, symbols, n_obs, matrix, stats, effective_n,
    selection, decision, verdict, integrity)`, where `effective_n` maps
    `"production" | "stored" | "selection"` → `(m, r_bar, N_eff_kish, N_eff_participation)` and
    `decision` is `"D1" | "D2" | "D3"`.
- **MIRROR**: `equity.py:1-14` for the pure-computation docstring register; `BackfillResult`
  (`data/backfill.py:23-37`) for frozen-dataclass + Attributes style.
- **IMPORTS**: `logging`, `statistics`, `dataclasses.dataclass/field`, `numpy as np`,
  `pandas as pd`, `from trading_bot import config`, `from trading_bot.data import storage`,
  `logger = logging.getLogger("trading_bot")`.
- **GOTCHA**: Import nothing from `trading_bot.backtest.*`. Phase 1 is concurrently editing
  `equity.py` and `walkforward.py` (contract §11); an import creates a merge surface for no
  benefit. `DAY_MS` is duplicated on purpose — `equity.py:22` and `walkforward.py:44` each define
  their own, and v0.2.0 Phase 1's plan recorded that duplication as correct.
- **VALIDATE**: `.venv/bin/python -m py_compile src/trading_bot/data/correlation.py` → clean.

### Task 3: `daily_return_frame` — aligned daily returns with a hard grid assertion

- **ACTION**: Add to `data/correlation.py`.
- **IMPLEMENT**: three functions.
  - `_stored_symbols(conn, timeframe) -> tuple[str, ...]` — a direct
    `SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ? ORDER BY symbol`. Docstring records
    why it is here rather than in `storage.py`: `storage.py` is not Phase 2's to modify (contract
    §2), and making the default universe a function of the **store** rather than a hand-maintained
    list means a future backfill is picked up without editing config.
  - `_assert_daily_grid(index, symbol, timeframe) -> None` — raises `ValueError` if any `ts % interval != 0`
    ("grid-aligned") or if `median(diff(ts)) != interval` ("median bar spacing"). Docstring
    mirrors `engine.py:186-197`'s reasoning verbatim in spirit: a series backfilled under the
    wrong key would produce a plausible correlation on the wrong bars — a silent failure in the
    flattering direction. Median spacing so a handful of missing candles cannot trip it.
  - `daily_return_frame(conn, symbols, *, start_ms, end_ms, timeframe=None) -> pd.DataFrame` —
    per symbol, `storage.load_candles(...)` → frame per FRAME_FROM_STORAGE → `_assert_daily_grid`
    → `["close"]`; assemble, `sort_index()`, then
    `rets = (prices / prices.shift(1) - 1.0).dropna(how="any")`. Docstring states **both bounds
    inclusive** (matching `storage.py:216-217`), complete-cases-only alignment, and returns an
    empty frame when fewer than 2 aligned returns survive. Log at INFO: symbol count, `n_obs`,
    and how many timestamps the join dropped (`len(prices) - 1 - len(rets)`) — a non-zero drop is
    exactly the silent data problem this phase exists to surface.
- **MIRROR**: FRAME_FROM_STORAGE, SILENT_MISALIGNMENT_MUST_RAISE.
- **IMPORTS**: already added in Task 2.
- **GOTCHA (three)**:
  1. **Do not use `pct_change()`** — installed pandas is 3.0.3 where fill behaviour changed;
     `close / close.shift(1) - 1.0` is explicit, version-proof and self-documenting.
  2. `load_candles` bounds are **both inclusive** and it does **not** take `_db_lock` (contrast
     `find_gaps`, `storage.py:171`). Do not add locking; do not fix that asymmetry.
  3. **Listwise, not pairwise.** Pairwise completion gives each pair a different n and can yield
     a non-PSD matrix with negative eigenvalues, silently corrupting the participation ratio. On
     the real store the two are identical (zero NaN across 20 symbols × 1096 bars), so listwise
     costs nothing and buys correctness. **Record the measured drop count in the report** so a
     future expansion that does lose bars is visible.
- **VALIDATE**:
  ```bash
  PYTHONPATH=src .venv/bin/python -c "
  import sqlite3
  from trading_bot import config
  from trading_bot.data import correlation as c
  conn = sqlite3.connect('data/ohlcv.db')
  r = c.daily_return_frame(conn, ['BTCUSDT','ETHUSDT','SOLUSDT'],
        start_ms=config.date_to_ms('2023-07-27'), end_ms=config.date_to_ms('2026-07-26'))
  assert r.shape == (1095, 3), r.shape
  print('aligned daily returns OK', r.shape)"
  ```
  EXPECT: `aligned daily returns OK (1095, 3)`.

### Task 4: `gap_integrity` — interior gaps only, over the research grid

- **ACTION**: Add `gap_integrity(conn, symbols, *, timeframes=INTEGRITY_TIMEFRAMES,
  start_ms=None) -> dict[str, list[str]]` to `data/correlation.py`.
- **IMPLEMENT**: For each symbol × timeframe: `last = storage.last_ts(...)`; if `None`, record
  `f"{timeframe} MISSING"`; else call
  `storage.find_gaps(conn, symbol, timeframe, now_ms=last + interval + 1, start_ms=start_ms)` and
  record each tuple as `f"{timeframe} {g0}-{g1}"`. Assert the documented invariants on every
  returned tuple (`g0 <= g1`, both `% interval == 0`) and raise on violation — they are
  guarantees, so a violation is a storage bug worth surfacing loudly. Returns `{}`-valued empty
  lists for clean symbols.

  Docstring must say **why** the `now_ms` pin exists: a symbol with an interior hole is worse
  than a symbol you do not have, because the hole silently distorts a **pooled** result; whereas
  trailing staleness is normal poller lag present on 40 of 60 cells at any moment (measured
  2026-07-27) and must not disqualify anything. Pinning `now_ms` inside the freshness window
  keeps the staleness branch (`storage.py:188-192`) from firing while the interior-gap loop
  (`storage.py:179-181`) still runs over the whole series. This **reads** `find_gaps`' documented
  invariants (`storage.py:140-146`); it never reimplements gap detection.
- **MIRROR**: CLI_REPORT_COMMAND's per-cell loop; GAP_INVARIANT_TEST's assertion triple.
- **IMPORTS**: none new.
- **GOTCHA**: `find_gaps`' `start_ms` defaults to `config.date_to_ms(config.BACKFILL_START)` =
  2023-01-01 (`storage.py:160-161`). The stored 1h/4h series begin *before* that (1h min ts
  `1672506000000` = 2022-12-31 17:00Z, measured) and extra leading data creates no gap, so the
  default is correct — leave it alone. Passing `CORRELATION_START` here would be **wrong**: it
  would hide any hole in 2023-01→2023-07, and that history is real warmup the regime classifier
  consumes. Default `start_ms=None`; expose it only for tests.
- **VALIDATE**:
  ```bash
  PYTHONPATH=src .venv/bin/python -c "
  import sqlite3
  from trading_bot.data import correlation as c
  conn = sqlite3.connect('data/ohlcv.db')
  syms = c._stored_symbols(conn, '1d')
  bad = {k: v for k, v in c.gap_integrity(conn, syms).items() if v}
  print(len(syms), 'symbols;', len(bad), 'with interior gaps'); assert not bad, bad"
  ```
  EXPECT: `20 symbols; 0 with interior gaps` — matching the planning measurement.

### Task 5: `correlation_matrix`, `effective_n`, `effective_n_participation`, `btc_beta`, `median_quote_volume`

- **ACTION**: Add to `data/correlation.py`.
- **IMPLEMENT**:
  - `correlation_matrix(returns) -> pd.DataFrame` — Pearson; empty frame if < 2 observations.
  - `effective_n(corr, symbols=None) -> tuple[int, float, float]` returning `(m, r̄, N_eff)`.
    Docstring states the formula, names the §0b reproduction as the correctness check, and
    documents: `m == 1 → (1, 0.0, 1.0)`; `N_eff` **clamped to `[1.0, m]`** because with a
    strongly negative r̄ the closed form can exceed m or go negative and neither is a meaningful
    sample size — the clamp is the honest way to say "at most m independent series exist."
  - `effective_n_participation(corr, symbols=None) -> float` — `numpy.linalg.eigvalsh` (the
    matrix is real symmetric), same clamp. Docstring records that unlike the Kish form it
    respects block structure, so it credits a genuinely detached series instead of averaging it
    into one r̄; measured `config.SYMBOLS` 1.395, selection 2.925; reported, never used by the
    decision rule.
  - `btc_beta(returns, symbol, *, anchor=None) -> float | None` —
    `cov(r_sym, r_anchor) / var(r_anchor)`, sample `ddof=1`; exactly `1.0` for the anchor;
    `None` if the anchor is absent or has zero variance. Docstring records that **beta and
    correlation measure different things and both are reported**: r is the *share* of
    co-movement, beta its *amplitude* — TRX has r 0.222 / beta 0.327 while DOGE has r 0.762 /
    beta 1.433, so a symbol can be a leveraged BTC proxy at moderate r. That is why the screen is
    on beta and the ranking is on r.
  - `median_quote_volume(conn, symbol, *, start_ms, end_ms, timeframe=None) -> float | None` —
    median of `close * volume`. Docstring states plainly it is a proxy, cannot validate
    `config.SLIPPAGE_PCT` (2 bps/side), and is a floor screen only.
- **MIRROR**: NONE_NOT_A_FAKE_NUMBER — every one returns `None`, never `0.0`, on insufficient data.
- **IMPORTS**: none new.
- **GOTCHA**: the off-diagonal mean must **exclude the diagonal explicitly**. `corr.values.mean()`
  silently includes m ones and biases r̄ upward by `(1 − r̄)/m`, which at m=3 turns 0.757 into
  0.838 and N_eff into 1.09 — a plausible-looking wrong number, the worst kind. Use
  `off = [M[i][j] for i in range(m) for j in range(m) if i != j]`, or equivalently
  `(M.sum() - m) / (m * (m - 1))`.
- **VALIDATE**:
  ```bash
  PYTHONPATH=src .venv/bin/python -c "
  import sqlite3
  from trading_bot import config
  from trading_bot.data import correlation as c
  conn = sqlite3.connect('data/ohlcv.db')
  R = c.daily_return_frame(conn, list(config.SYMBOLS),
        start_ms=config.date_to_ms('2023-07-27'), end_ms=config.date_to_ms('2026-07-26'))
  C = c.correlation_matrix(R)
  print('BTC/ETH %.3f BTC/SOL %.3f ETH/SOL %.3f' % (C.loc['BTCUSDT','ETHUSDT'],
        C.loc['BTCUSDT','SOLUSDT'], C.loc['ETHUSDT','SOLUSDT']))
  m, rbar, n = c.effective_n(C)
  print('m=%d r_bar=%.4f N_eff=%.3f pr=%.3f' % (m, rbar, n, c.effective_n_participation(C)))"
  ```
  EXPECT exactly `BTC/ETH 0.809 BTC/SOL 0.744 ETH/SOL 0.719` and
  `m=3 r_bar=0.7574 N_eff=1.193 pr=1.395`. **If these differ, stop and find out why** — §0b's
  anchors are the only external check this phase has.

### Task 6: `select_research_symbols` — the pre-registered, recorded selection rule

- **ACTION**: Add
  `select_research_symbols(stats, corr, *, anchor=None, select_n=None) -> tuple[str, ...]`.
  Its docstring **is** the audit trail; write it in `universe.py:8-12`'s register, including the
  phrase "recorded so it is auditable, not re-litigated later".
- **IMPLEMENT** the rule, exactly and in this order:
  1. **Eligibility** — zero **interior** gaps across 1d/4h/1h; ≥ `CORRELATION_MIN_OVERLAP_BARS`
     aligned daily returns; median daily quote volume ≥ `CORRELATION_MIN_QUOTE_VOLUME_USD`;
     `|BTC beta| <= CORRELATION_MAX_BTC_BETA`.
  2. **Anchor** — `CORRELATION_ANCHOR_SYMBOL` (BTCUSDT) is **pinned** regardless of rank. It
     costs 0.034 of Kish effective N (1.868 → 1.834, measured) and buys the deepest book in the
     universe, continuity with the buy-and-hold benchmark and `config.SYMBOLS`, and the beta
     reference every other row is quoted against. A crypto research universe containing no BTC
     would be an odd artefact of an optimiser.
  3. **Ranking** — remaining eligible candidates ascending by **mean pairwise Pearson r** of
     daily returns. Mean r, not beta, because r̄ is the exact quantity the effective-N formula
     consumes; ranking on beta would optimise a proxy. Ties break on higher median quote volume,
     then alphabetically, so the result is deterministic.
  4. **Size** — take the first `CORRELATION_SELECT_N` (8). With the anchor that is 9, clearing
     the PRD's ≥8 floor.
  5. **Result order** — anchor first, then rank order. **Never reorder**: the report and
     `RESEARCH_SYMBOLS` key off this ordering.

  Also record *why simple and not optimised*: the deterministic ranking and a greedy
  minimum-average-correlation search agree on 7 of 8 members (differing only AAVE ↔ NEAR) with
  effective N differing by 0.009 (1.868 vs 1.877), so the selection is insensitive to the
  algorithm and a greedy search would add path dependence and a hidden degree of freedom for a
  0.5% effect. Raises `ValueError` when fewer than `select_n` eligible non-anchor candidates
  exist — that is decision **D3**, the only condition under which a genuine backfill is justified.
- **MIRROR**: RECORDED_SELECTION_RULE_DOCSTRING.
- **IMPORTS**: none new.
- **GOTCHA**: the anchor is exempt from the **beta** screen (its beta is 1.0 by construction and
  passes anyway) but **not** from the gap/history screens — if BTC ever had an interior hole,
  silently pinning it is exactly the failure the integrity check exists to catch. Raise; do not
  substitute a different anchor.
- **VALIDATE**: covered by Task 8's CLI run. Expected selection from the planning measurement:
  `BTCUSDT TRXUSDT BCHUSDT BNBUSDT XRPUSDT UNIUSDT OPUSDT AAVEUSDT FILUSDT` — 9 symbols,
  r̄ 0.4885, Kish 1.834, participation ratio 2.925.

### Task 7: `build_report` + `format_report` — the decision rule and the artifact

- **ACTION**: Add `build_report(conn, symbols=None, *, start_ms=None, end_ms=None,
  timeframe=None, select_n=None) -> CorrelationReport` and
  `format_report(report) -> str` to `data/correlation.py`.
- **IMPLEMENT**:
  - `build_report` defaults: `symbols = _stored_symbols(conn, timeframe)`,
    `start_ms = config.CORRELATION_START`, `end_ms =` **the last stored bar at `timeframe`**
    (`storage.last_ts`). Deriving `end_ms` from the store rather than wall clock keeps the report
    reproducible and cannot include a forming bar.
  - The **decision rule**, in the docstring verbatim, with `N_prod` = effective N over
    `config.SYMBOLS` and `N_sel` = effective N over the selection — see the Decision Rule section
    below for the three letters and their actions. The docstring must state that **a D2 verdict
    is a completed outcome, not a failed phase**: measuring that 20 liquid majors are close to
    one bet is the answer to PRD Open Question #4.
  - `format_report` renders markdown, sections in order: (1) the correction to the PRD's premise
    and the supersession of KNOWN-LIMITATIONS §4; (2) measured environment, store inventory,
    bytes/row, expansion cost; (3) gap integrity, interior vs trailing, per symbol; (4) the full
    pairwise matrix with `n_obs`; (5) the effective-N table with the §0b reproduction; (6) the
    per-symbol table (mean r, r vs anchor, beta, median quote volume, eligibility, reason);
    (7) the selection rule verbatim plus the result; (8) the verdict — decision letter, the
    two-things-breadth-buys distinction with the independent-equivalent-trade arithmetic, and what
    happens next; (9) degrees of freedom consumed. Pure string building; the caller owns file I/O,
    and **every number comes from the report object** so the artifact cannot drift from what was
    measured.
- **MIRROR**: `build_performance_chart.py:174-214` (`_scorecard`: condition rows as
  `(label, value, threshold, ok)`) for the verdict table shape.
- **IMPORTS**: none new.
- **GOTCHA**: `end_ms` from `last_ts`, **not** `time.time()`. A wall-clock end makes the artifact
  irreproducible and can include a forming daily bar — the hazard v0.2.0 Phase 1's plan flagged
  at its Task 6. The last stored 1d bar is `1785110400000` = 2026-07-26 00:00Z, a *closed* bar.
- **VALIDATE**: covered by Tasks 8 and 11.

### Task 8: `cli.py` — the `correlation-report` subcommand

- **ACTION**: Edit `src/trading_bot/cli.py`: add `from trading_bot.data import correlation` to
  the import block (13-22); add the subparser after the `walkforward` block (129-144) and
  **after** Phase 1's `benchmark` parser (contract §7 registration order); add an `elif` branch
  after the backtest/walkforward branch (199-209); add `_correlation_command`.
- **IMPLEMENT**:
  - Parser `"correlation-report"` with flags: `--symbol` (`action="append"`, default = every
    symbol stored at the correlation timeframe), `--timeframe` (default
    `config.CORRELATION_TIMEFRAME`), `--start` / `--end` (`type=_date_arg`), `--select-n`
    (`type=int`, default `config.CORRELATION_SELECT_N`), `--out` (write the markdown report in
    addition to stdout), `--check-exchange` (`action="store_true"`, probe reachability first).
  - Dispatch branch: `conn = connect(args.db)` → `_correlation_command(...)` → `conn.close()` →
    `sys.exit(exit_code)`.
  - `_correlation_command(conn, symbols, *, timeframe, start_ms, end_ms, select_n, out,
    check_exchange) -> int`: when `check_exchange`, call
    `fetch_ohlcv_page(to_ccxt_symbol(config.CORRELATION_ANCHOR_SYMBOL), timeframe,
    since_ms=config.date_to_ms(config.BACKFILL_START), limit=5)` inside
    `try/except (ccxt.NetworkError, ccxt.ExchangeError)`, printing
    `exchange: OK (<n> rows, first ts <ts>)` or `exchange: UNREACHABLE (<class>: <msg>)`. Then
    `report = correlation.build_report(...)`, print the compact summary from the UX section, and
    when `out` is set write `correlation.format_report(report)` and print `wrote <path>`.
  - **Exit code**: `1` if the exchange probe was requested and failed, **or** the decision is
    `D3`, **or** any *selected* symbol has an interior gap. `0` for D1 **and D2** — D2 is an
    honest finding, not a command failure, and exiting 1 would train the operator to ignore the
    exit code. Document that choice in the docstring.
- **MIRROR**: CLI_SUBPARSER_AND_DISPATCH, CLI_REPORT_COMMAND, and
  `build_performance_chart.py:717-724` for the `--out` write.
- **IMPORTS**: `from trading_bot.data import correlation` at module level; `import ccxt` and
  `from trading_bot.exchange.binance_client import fetch_ohlcv_page, to_ccxt_symbol`
  **function-local**, so the default offline path never pays the ccxt import — mirroring how
  `test_cli.py:362` treats ccxt as a boundary. **Read-only use of `binance_client`; no edit.**
- **GOTCHA**: `--timeframe` is a **plain string**, not `action="append"` as `backfill` uses
  (`cli.py:47-51`). Correlation is defined on one return series; a list would silently mean "last
  one wins". Also: `--as-of` is deliberately **not** offered — `gap_integrity` neutralises
  staleness internally, so an as-of date would have no effect and offering it would imply
  otherwise.
- **VALIDATE**:
  ```bash
  .venv/bin/python -m trading_bot.cli correlation-report --help
  .venv/bin/python -m trading_bot.cli correlation-report; echo "exit=$?"
  ```
  EXPECT: help lists all seven flags; the report prints; `exit=0`.

### Task 9: `tests/test_correlation.py`

- **ACTION**: Create the file. Everything offline except one network smoke. Module constants
  derived from config and `storage.TIMEFRAME_MS` — contract §8 forbids hardcoding the tier:
  `TF = config.CORRELATION_TIMEFRAME`, `DAY = storage.TIMEFRAME_MS[TF]`,
  `START = 1_700_000_000_000 // DAY * DAY`.
- **IMPLEMENT** these classes and cases:
  - `TestDailyReturnFrame` — hand-computed returns (`100 → 110 → 99` ⇒ `+0.10, −0.10`), shape
    `(n_bars − 1, n_symbols)`; one symbol missing an interior bar ⇒ that ts dropped from **both**
    columns; a 4h-spaced series loaded as `1d` ⇒ `ValueError` naming "median bar spacing"; a
    non-grid ts ⇒ `ValueError` naming "grid-aligned"; fewer than 2 bars ⇒ empty frame, no raise;
    **inclusive bounds** — `start_ms` == first ts and `end_ms` == last ts include both, pinning
    `load_candles`' documented contract.
  - `TestCorrelationMatrix` — proportional series ⇒ r = 1.0 (1e-12); anti-proportional ⇒ −1.0;
    a constant column ⇒ `NaN` row/col and `effective_n` does not raise; symmetry and unit diagonal.
  - `TestEffectiveN` (hand-computed, per HAND_COMPUTED_NUMERIC_TEST) — m=3 r̄=0.5 ⇒ exactly 1.5;
    m=2 r̄=0 ⇒ 2.0; m=4 r̄=1 ⇒ 1.0; m=1 ⇒ `(1, 0.0, 1.0)`; r̄=−0.9 m=3 ⇒ clamped into `[1, 3]`;
    **an all-off-diagonals-equal 3×3 at 0.7574 ⇒ r̄ ≈ 0.7574 and N_eff ≈ 1.1929 (abs 1e-4)** —
    the test that would have caught the `values.mean()` bug in Task 5's GOTCHA; participation
    ratio of the identity matrix of size m ⇒ exactly m, of the all-ones matrix ⇒ 1.0.
  - `TestBtcBeta` — a series constructed as exactly 2× the anchor ⇒ beta 2.0, r 1.0; anchor vs
    itself ⇒ exactly 1.0; zero-variance anchor ⇒ `None`.
  - `TestGapIntegrity` — synthetic 1d series with one dropped interior bar ⇒ exactly one entry
    formatted `"1d <ts>-<ts>"`, with the `check_invariants` triple holding on the raw tuple; a
    **stale** series with no interior hole ⇒ **empty list** (the single most important test in the
    module — without it the phase would disqualify all 20 symbols for normal poller lag); a
    symbol with no rows at one timeframe ⇒ `"<tf> MISSING"`.
  - `TestSelectionRule` — a synthetic 6-symbol structure whose winners are obvious by
    construction ⇒ anchor-first then rank order; a candidate below the liquidity floor / above
    the beta ceiling / with an interior gap ⇒ excluded with a `reason` naming volume / beta /
    gaps respectively; determinism under shuffled input and ties; fewer than `select_n` eligible
    ⇒ `ValueError`.
  - `TestBuildReportDecision` — three synthetic stores yielding `"D1"`, `"D2"` (whose `verdict`
    string must contain the words *rows* and *information*), and `"D3"`.
  - `TestFormatReport` — non-empty markdown containing every section heading, the measured
    `n_obs`, the decision letter, and the string `RESEARCH_SYMBOLS` so the artifact always tells
    a reader what to do with the result.
  - `TestKnownLimitationsAnchors` — **the correctness gate**, guarded by
    `@pytest.mark.skipif(not Path("data/ohlcv.db").exists(), ...)` so a fresh clone stays green:
    BTC/ETH 0.809, BTC/SOL 0.744, ETH/SOL 0.719 (abs 0.001), r̄ 0.7574 (abs 0.0005), Kish N_eff
    1.193 (abs 0.005) over `CORRELATION_START → 2026-07-26`. Docstring names the finding it pins:
    *"KNOWN-LIMITATIONS §0b: 3 symbols at mean r ~0.76 → effective N ~1.2."*
  - `TestExchangeReachability` — one test per NETWORK_TEST_GATING, calling `fetch_ohlcv_page` for
    the anchor at `1d` and asserting non-empty + ascending timestamps. Its docstring records
    **why it lives here**: contract §8 gives Phase 2 only `test_correlation.py` plus additions to
    `test_storage.py`, and the exchange probe is Phase 2's backfill precondition (decision D3),
    not a storage concern.
- **MIRROR**: TEST_MODULE_SHAPE, GAP_INVARIANT_TEST, HAND_COMPUTED_NUMERIC_TEST,
  NETWORK_TEST_GATING.
- **IMPORTS**: `os`, `sqlite3`, `pathlib.Path`, `pytest`, `pandas as pd`,
  `from trading_bot import config`, `from trading_bot.data import correlation, storage`.
- **GOTCHA**: build fixtures with `storage.connect(str(tmp_path / "x.db"))` +
  `storage.upsert_candles`, seeded bar-by-bar (contract §8) — **not** raw SQL — so the tests
  exercise the same write path production uses and cannot drift from the schema. No
  `engine.clear_caches()` autouse fixture is needed because `correlation.py` holds no cache; add
  a comment saying so, so its absence reads as deliberate.
- **VALIDATE**:
  ```bash
  .venv/bin/python -m pytest tests/test_correlation.py -q
  RUN_NETWORK_TESTS=1 .venv/bin/python -m pytest tests/test_correlation.py -m network -q
  ```

### Task 10: Deliberate additions to `tests/test_storage.py`

- **ACTION**: Append a `class TestResearchGridGaps` — contract §8 sanctions exactly this file as
  Phase 2's extension target.
- **IMPLEMENT**: four cases covering the multi-symbol/multi-timeframe shape no existing test does:
  - **Per-symbol isolation** — symbol A complete, symbol B holed, same timeframe, same DB ⇒ B
    reported, A `[]`. Pins that the `WHERE symbol = ? AND timeframe = ?` predicate at
    `storage.py:173` really scopes per cell, the assumption the 60-cell sweep rests on.
  - **Per-timeframe isolation** — same symbol, `1d` complete and `4h` holed ⇒ `1d` clean.
  - **Interior gap survives a pinned fresh `now_ms`** — with `now_ms = last_ts + interval + 1`
    the staleness branch cannot fire but the interior gap is still reported. This is exactly the
    technique `correlation.gap_integrity` depends on, pinned so a change to
    `config.STALENESS_INTERVALS` or `storage.py:188-192` breaks a test rather than silently
    changing which symbols are eligible.
  - **Invariants across all three research timeframes** — sweep `("1d","4h","1h")` × several
    `now_ms` values applying the `check_invariants` triple from `test_storage.py:112-116`, with
    `INTERVAL` taken per-timeframe from `storage.TIMEFRAME_MS`.
- **MIRROR**: GAP_INVARIANT_TEST; `class TestStalenessDetection` (`test_storage.py:210-275`) for
  class grouping and fixture style.
- **IMPORTS**: none new — the module already imports `config` and `storage` (lines 6-7).
- **GOTCHA**: the module-level `TF = "15m"` / `INTERVAL` constants (`test_storage.py:10-11`) are
  used by the existing tests — **do not repoint them**. Use per-test locals for the research
  timeframes, exactly as the v0.2.0 1d tests already do (`test_storage.py:373-411`). The
  staleness comparison at `storage.py:189` is **strict `>`**, so lag of exactly
  `STALENESS_INTERVALS` intervals is not stale; `+ interval + 1` gives 1 ms of margin above one
  interval and stays well inside the window.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_storage.py -q` → all pass, including the
  ~30 pre-existing cases.

### Task 11: Run the measurement and reproduce §0b's anchors

- **ACTION**: Run the committed command against the real store. No code change. **This is the
  phase's correctness gate.**
- **IMPLEMENT**:
  ```bash
  .venv/bin/python -m trading_bot.cli correlation-report | head -20
  .venv/bin/python -m trading_bot.cli correlation-report \
    --out .claude/PRPs/reports/phase2-correlation-report.md; echo "exit=$?"
  sqlite3 data/ohlcv.db "SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(ts), MAX(ts), \
    SUM(ts % 86400000 <> 0) AS misaligned FROM ohlcv WHERE timeframe='1d';"
  ```
  EXPECT, reproducing the planning measurement exactly:
  - `BTC/ETH 0.809  BTC/SOL 0.744  ETH/SOL 0.719`
  - `config.SYMBOLS   n=3   r_bar=0.7574   N_eff=1.193   (participation ratio 1.395)`
  - `all stored       n=20  r_bar=0.5905   N_eff=1.637   (participation ratio 2.505)`
  - `selection        n=9   r_bar=0.4885   N_eff=1.834   (participation ratio 2.925)`
  - `20 of 20 symbols clean, 0 interior gaps in 60 cells`
  - selection `BTCUSDT TRXUSDT BCHUSDT BNBUSDT XRPUSDT UNIUSDT OPUSDT AAVEUSDT FILUSDT`
  - SQL: `20 | 26080 | 1672531200000 | 1785110400000 | 0`
  - decision: `1.834 / 1.193 = 1.537 >= 1.50` → **D1**, marginally.
- **GOTCHA**: the D1/D2 boundary is decided by a **2.5% margin** (1.537 vs 1.50). **Do not tune
  `CORRELATION_EFFECTIVE_N_MIN_RATIO` to change the letter.** If the measurement lands under
  1.50, the verdict is D2 and is reported as such; adjusting a threshold after seeing the number
  is precisely the "gate becomes theater" failure the PRD's top technical risk names. Second: if
  the selection membership differs from the expectation above, that is *information*, not a bug —
  record what differed and why (a screen bit, or an AAVE/NEAR rank flip whose r̄ differ by
  0.001). Do not force the expected list.
- **VALIDATE**: anchors match to the stated tolerances; `exit=0`; the artifact is non-empty.

### Task 12: Promote the selection into `config.RESEARCH_SYMBOLS`

- **ACTION**: Append to the Task 1 config block, using the **measured** membership from Task 11 —
  never this plan's expectation.
- **IMPLEMENT**: an annotated tuple, anchor first then ascending mean-r rank, with a
  one-line-per-symbol sector comment in `universe.py:18-36`'s style, preceded by a comment block
  recording: the regeneration command
  (`python -m trading_bot.cli correlation-report --out <path>`), **NEVER REORDER** (report and
  downstream tables key off the ordering), the measured figures (1095 daily returns,
  2023-07-27 → 2026-07-26, r̄ 0.4885, effective N 1.834 Kish / 2.925 participation ratio, against
  `config.SYMBOLS`' 0.7574 and 1.193 / 1.395), and the **honest reading**: 3 → 9 symbols is 3× the
  rows and ~1.5× the independent information; §0b's trap is **mitigated, not solved**; Phase 9
  must read its DSR as governed by effective N ≈ 1.8, not by 9.
- **MIRROR**: CONFIG_TUPLE_SHAPE; `universe.py:18-36` sector comments.
- **IMPORTS**: none.
- **GOTCHA**: `config.SYMBOLS` stays `("BTCUSDT","ETHUSDT","SOLUSDT")`, and **nothing in
  `cli.py`'s existing branches may be repointed at `RESEARCH_SYMBOLS`** — they default to
  `config.SYMBOLS` at `cli.py:186`, `194`, `201`, `254`, and changing them would silently widen
  the production product and rewrite what every committed number under `.claude/PRPs/reports/`
  refers to. Phase 9 passes `RESEARCH_SYMBOLS` explicitly via `--symbol`; that is the whole
  intended coupling.
- **VALIDATE**:
  ```bash
  .venv/bin/python -c "
  from trading_bot import config
  assert config.SYMBOLS == ('BTCUSDT','ETHUSDT','SOLUSDT')
  assert len(config.RESEARCH_SYMBOLS) >= 9
  assert config.RESEARCH_SYMBOLS[0] == config.CORRELATION_ANCHOR_SYMBOL
  print('promoted;', len(config.RESEARCH_SYMBOLS), 'symbols')"
  grep -rn "config.SYMBOLS" src/ | wc -l    # unchanged vs before the phase
  ```

### Task 13: Review that the artifact states the honest verdict

- **ACTION**: The artifact is generated by Task 11's `--out`; this task is the review that it
  says the true thing, then commit it.
- **IMPLEMENT**: confirm the markdown contains, explicitly:
  1. The correction — 20 stored not 3 — and that KNOWN-LIMITATIONS §4's "only 3 symbols exist" is
     superseded by contract §0a (with the pointer, since this phase does not edit that file).
  2. The reproduction of §0b's three anchors and the ~1.2 effective N, named as the correctness
     check.
  3. The two-things-breadth-buys section with the arithmetic: *mechanical* —
     0.0852 trades/symbol/day × 9 × 90 days ≈ **69 OOS trades** vs a floor of 30 (labelled an
     ESTIMATE at the v0.2.0 rate); *informational* — 69 × (1.834/9) ≈ **14
     independent-equivalent trades** against v0.2.0's 23 × (1.193/3) ≈ **9**.
  4. The answer to PRD Open Question #4, plainly: **within Binance USDT-M perps there is no
     genuinely uncorrelated crypto.** 19 of 20 liquid majors sit at r 0.56–0.81 to BTC; the one
     exception (TRXUSDT, r 0.222) is also among the least liquid. The achievable ceiling from this
     venue is an effective N around **2–3**, not 8. A true diversifier would need a different
     asset class, which the PRD excludes ("Binance futures only").
  5. The storage table (98.84 bytes/row, 256 GiB free) and the conclusion that **disk is not the
     constraint; correlation is.**
  6. The decision letter with its numeric justification, plus the D2/D3 contingency — *what would
     justify a genuine backfill*: (a) fewer than `CORRELATION_SELECT_N` eligible symbols, or (b) a
     candidate class with measured r vs BTC below ~0.4 **and** median quote volume above
     `CORRELATION_MIN_QUOTE_VOLUME_USD`. The acceptance test is stated in advance: such a backfill
     must raise the selection's Kish effective N by **≥0.25 absolute**, measured by re-running
     `correlation-report`, or it is not worth the 4.0 MB and the download.
  7. Degrees of freedom consumed (Task 15).
- **MIRROR**: the register of `KNOWN-LIMITATIONS.md` — verdict first, measured numbers in tables,
  no hedging, an explicit "what IS established" separation.
- **GOTCHA**: **do not soften the finding to make the phase look successful.** Contract §12.5:
  every number is measured by a committed command. The PRD's honesty clause makes an
  unflattering, correctly-measured result a completed outcome. The failure mode to avoid is the
  opposite one — reporting "20 symbols now available, correlation trap addressed" and letting
  Phase 9 spend a DSR budget it does not have.
- **VALIDATE**:
  ```bash
  test -s .claude/PRPs/reports/phase2-correlation-report.md
  grep -n "0.809\|0.744\|0.719\|1.19\|RESEARCH_SYMBOLS\|D1\|D2" \
    .claude/PRPs/reports/phase2-correlation-report.md | head
  ```

### Task 14: CONTINGENCY — exchange re-ping and a genuine backfill (only under D3, or D2 + an approved candidate class)

- **ACTION**: **Skip entirely under D1.** Run only if Task 11 decides `D3`, or the operator
  approves a D2 expansion against Task 13's acceptance test.
- **IMPLEMENT**:
  ```bash
  # 14a. Reachability gate. Verified HTTP 200 in 0.19s on 2026-07-27; it was BLOCKED
  # on 2026-07-05, so re-verify rather than trusting this plan.
  curl -s -o /dev/null -w "fapi:%{http_code}\n" --max-time 15 https://fapi.binance.com/fapi/v1/ping
  .venv/bin/python -m trading_bot.cli correlation-report --check-exchange | head -3
  RUN_NETWORK_TESTS=1 .venv/bin/python -m pytest tests/test_correlation.py -m network -q

  # 14b. NEW symbols only, research timeframes, existing machinery (no code change:
  # backfill_all already accepts an arbitrary symbol list, data/backfill.py:130-137).
  .venv/bin/python -m trading_bot.cli backfill \
    --symbol <NEW1> --symbol <NEW2> --timeframe 1d --timeframe 4h --timeframe 1h

  # 14c. Re-measure. The backfill is justified only if effective N moved.
  .venv/bin/python -m trading_bot.cli correlation-report \
    --out .claude/PRPs/reports/phase2-correlation-report.md
  ```
- **MIRROR**: IDEMPOTENT_UNIVERSE_BACKFILL — upserts keyed on `(symbol, timeframe, ts)`, so a
  re-run after an interruption costs only the missing tail.
- **GOTCHA (four)**:
  1. **Do not add new symbols to `config.SYMBOLS` or `config.TIMEFRAMES`** to make `backfill`
     pick them up — pass `--symbol`. Widening `TIMEFRAMES` also widens `gap-report` and six loops
     in `tests/test_cli.py` (documented in the v0.2.0 Phase 1 plan).
  2. New symbols have **shorter history**. `universe.py:4-7` selected only symbols with 1d
     history at or before 2022-01-01 precisely so the research span plus warmup is covered. A
     symbol listed in 2024 cannot be pooled without truncating everyone's span:
     `daily_return_frame`'s listwise alignment would silently discard ~half the observations for
     *every* symbol. `CORRELATION_MIN_OVERLAP_BARS` is the guard — check `n_obs` after any
     backfill and reject the candidate if it dropped.
  3. A newly-backfilled series holds a **forming** current bar (`INSERT OR REPLACE`,
     `storage.py:94`) and `build_report` derives `end_ms` from `last_ts`, so a same-day
     re-measure could include a partial daily bar. Re-run after 00:00 UTC or pass `--end`.
  4. If the ping fails, **stop and escalate**. A verified offline fallback exists
     (`data.binance.vision` monthly ZIPs, URLs recorded in
     `.claude/PRPs/plans/v0.2.0/phase1-data-tier-extension.plan.md` Task 7) but building it is out
     of scope, and under D1 no backfill is needed — a `fapi` block does not block this phase.
- **VALIDATE**: `n_obs` has not fallen; the selection's Kish effective N rose by ≥0.25 absolute;
  `gap_integrity` reports the new symbols clean.

### Task 15: Full-suite regression and the degrees-of-freedom log

- **ACTION**: Run everything; record what the phase consumed.
- **IMPLEMENT**:
  ```bash
  .venv/bin/python -m py_compile src/trading_bot/data/correlation.py \
    src/trading_bot/config.py src/trading_bot/cli.py
  .venv/bin/python -m pytest -q
  ```
  Then append to the artifact's final section, per contract §12.6 and KNOWN-LIMITATIONS §9:
  - **Zero strategy degrees of freedom consumed.** No parameter affecting a trade was swept,
    fitted, or chosen by looking at P&L. `RESEARCH_SYMBOLS` follows a rule pre-registered *before*
    the selection was run, on correlation / liquidity / beta — none of which is a return.
  - **Two thresholds set to be non-binding, deliberately**:
    `CORRELATION_MIN_QUOTE_VOLUME_USD` ($50 M vs a measured minimum of $51.7 M) and
    `CORRELATION_MAX_BTC_BETA` (1.35 vs a measured selected maximum of 1.285). Both bind on
    nothing today; recorded as guards for a future expansion so a later reader does not mistake
    them for fitted values.
  - **One judgement call that is not a measurement**: `CORRELATION_EFFECTIVE_N_MIN_RATIO = 1.5`,
    chosen before the selection's effective N was known — pre-registered, but a threshold on a
    statistic that the measured 1.537 clears by only 2.5%. **Record the margin**, so anyone
    arguing the D1/D2 boundary has the number in front of them.
  - **One pinned anchor**: BTCUSDT, costing 0.034 of Kish effective N (measured), for the three
    reasons in Task 6's rule.
- **GOTCHA**: use `.venv/bin/python -m pytest`, never bare `pytest` — the venv must be the
  Homebrew python@3.11 one (recorded repo quirk: `python3 -m venv` with default Homebrew
  3.12/3.14 on this host produces a broken `pyexpat`/`ensurepip`). Do not recreate `.venv`.
- **VALIDATE**: `py_compile` clean; suite green at the counts below.

---

## Decision Rule (pre-registered so it cannot be re-litigated after the fact)

`N_prod` = Kish effective N over `config.SYMBOLS`; `N_sel` = over the selection;
`k = config.CORRELATION_SELECT_N` (8).

| Letter | Condition | Action | Reported as |
|---|---|---|---|
| **D1** | `len(selection) >= k+1` **and** every selected symbol interior-gap-clean **and** `N_sel >= 1.5 * N_prod` | Promote `RESEARCH_SYMBOLS`. **No backfill.** | Breadth confirmed — with "confirmed" meaning ~1.5×, not 3× |
| **D2** | Size + integrity hold, `N_sel < 1.5 * N_prod` | Promote anyway (rows still clear the mechanical `n_trades` floor) **and** report that information did not scale. Name the candidate class + acceptance test for a genuine backfill | Rows without information — a completed outcome, **exit 0** |
| **D3** | Fewer than `k` eligible candidates, or a selected symbol fails integrity with no eligible replacement | **Backfill required** (Task 14) before Phase 9 | Insufficient universe, **exit 1** |

Measured expectation: `1.834 / 1.193 = 1.537` → **D1 by a 2.5% margin.** This plan pre-commits to
reporting D2 honestly if the implemented measurement lands below 1.50, and forbids moving the
threshold to move the letter.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected | Edge case? |
|---|---|---|---|
| Hand-computed returns | closes `100 → 110 → 99` | `+0.10, −0.10` | — |
| Listwise alignment | symbol B missing one interior bar | ts dropped from **both** columns; drop count logged | Yes |
| Inclusive bounds | `start_ms` == first ts, `end_ms` == last ts | both bars included | Yes — pins `load_candles`' contract |
| Wrong-interval series | 4h bars loaded as `1d` | `ValueError` naming "median bar spacing" | Yes — silent-failure guard |
| Non-grid ts | `ts % 86_400_000 != 0` | `ValueError` naming "grid-aligned" | Yes |
| < 2 bars | 1 bar | empty frame, no raise | Yes |
| Perfect / anti correlation | proportional / anti-proportional | r = 1.0 / −1.0 | — |
| Zero-variance column | constant closes | `NaN` row/col; `effective_n` no raise | Yes |
| Kish m=3 r̄=0.5 | — | exactly 1.5 | Hand-computed |
| **Kish m=3 r̄=0.7574** | all off-diagonals equal | r̄ 0.7574, N_eff 1.1929 | **The `values.mean()` bug guard** |
| Kish m=4 r̄=1.0 / m=1 | — | 1.0 / `(1, 0.0, 1.0)` | Boundary |
| Kish r̄=−0.9, m=3 | — | clamped into `[1, 3]` | Yes — closed form misbehaves |
| Participation ratio | identity size m / all-ones | exactly m / 1.0 | Boundary |
| Beta 2× anchor / anchor vs itself / zero-var anchor | constructed | 2.0 & r=1.0 / exactly 1.0 / `None` | Yes |
| Interior gap | 1d series, bar 10 dropped | one `"1d ts-ts"`; invariants hold | — |
| **Stale, no hole** | last bar far behind now | **empty list** | **Yes — most important test in the module** |
| Missing timeframe | no rows at `4h` | `"4h MISSING"` | Yes |
| Selection screens | under liquidity floor / over beta ceiling / interior hole | excluded, `reason` names volume / beta / gaps | — |
| Selection determinism | shuffled input, ties present | identical tuple both times | Yes |
| Selection shortfall | fewer than `k` eligible | `ValueError` → D3 | Yes |
| Decision D1 / D2 / D3 | three synthetic stores | correct letter; D2's verdict names *rows* and *information* | — |
| `format_report` | any report | all headings, `n_obs`, decision letter, `RESEARCH_SYMBOLS` | — |
| **§0b anchors** (real DB, skip-if-absent) | `config.SYMBOLS`, 2023-07-27 → 2026-07-26 | 0.809 / 0.744 / 0.719; r̄ 0.7574; N_eff 1.193 | **The correctness gate** |
| Per-symbol / per-timeframe gap isolation | A clean & B holed; `1d` clean & `4h` holed | scoped correctly | Yes |
| Research-grid invariants | `1d`/`4h`/`1h` × several `now_ms` | `check_invariants` holds for every tuple | Yes |
| Exchange reachability | `RUN_NETWORK_TESTS=1` | non-empty, ascending ts | Network-marked, skipped by default |

### Edge Cases Checklist
- [x] Empty input — no symbols, no bars, single bar → empty frame, no exception
- [x] Zero-variance series → `NaN`, not a crash
- [x] Negative mean correlation → effective N clamped to `[1, m]`
- [x] Ragged panel → listwise alignment + `CORRELATION_MIN_OVERLAP_BARS` guard
- [x] Trailing staleness must **not** disqualify a symbol; interior gaps must
- [x] Series stored under the wrong timeframe key → raises, never silently correlates
- [x] Forming (unclosed) daily bar → `end_ms` from `last_ts` (Task 14 GOTCHA 3)
- [x] Both `load_candles` bounds inclusive — asserted, not assumed
- [x] Network unavailable → default path fully offline; probe opt-in and marked
- [x] Fresh clone without `data/ohlcv.db` → anchor test skips, suite green
- [x] Determinism of the selection under ties and input reordering
- [ ] Concurrent access — N/A beyond the existing `storage._db_lock`, unchanged
- [ ] Permission denied — N/A (one `--out` write, to an operator-chosen path)

---

## Validation Commands

### Static Analysis
```bash
.venv/bin/python -m py_compile src/trading_bot/data/correlation.py \
  src/trading_bot/config.py src/trading_bot/cli.py
```
EXPECT: zero syntax errors. **No linter and no type checker exist in this repo** —
`pyproject.toml` declares `ccxt`/`pandas`/`apscheduler`/`python-dotenv` plus `pytest` as the sole
dev extra, and one pytest marker (`network`). There is no `Makefile`, `noxfile.py`, `tox.ini`, or
`setup.cfg`. Do not invent a lint or typecheck step (contract §12.2, KNOWN-LIMITATIONS §8).

### Unit Tests
```bash
.venv/bin/python -m pytest tests/test_correlation.py tests/test_storage.py -q
```

### Full Test Suite
```bash
.venv/bin/python -m pytest -q
```
**Baseline before this phase: 286 tests collected** (verified 2026-07-27 via
`.venv/bin/python -m pytest --collect-only -q`; contract §0). After this phase: **≥ 286 plus the
new cases (~40 in `test_correlation.py`, ~4 in `test_storage.py`), zero failures, and the 286
pre-existing tests unchanged and green.** A phase that breaks any of the 286 is not done
(contract §8, §12.1).

Regression check that nothing else moved:
```bash
.venv/bin/python -m pytest tests/test_cli.py tests/test_backtest.py \
  tests/test_walkforward.py tests/test_equity.py -q
```
EXPECT: unchanged pass counts — `config.SYMBOLS` and `config.TIMEFRAMES` are untouched, so none
of the loops iterating them changes shape.

### Network Smoke (opt-in)
```bash
RUN_NETWORK_TESTS=1 .venv/bin/python -m pytest tests/test_correlation.py -m network -q
curl -s -o /dev/null -w "fapi:%{http_code}\n" --max-time 15 https://fapi.binance.com/fapi/v1/ping
```
EXPECT: 1 passed; `fapi:200`. Deselected from the default suite so it stays offline-clean
(`pyproject.toml:26-28`).

### Manual Validation (the phase's success signal)
```bash
.venv/bin/python -m trading_bot.cli correlation-report \
  --out .claude/PRPs/reports/phase2-correlation-report.md; echo "exit=$?"
.venv/bin/python -c "
from trading_bot import config
print('SYMBOLS         ', config.SYMBOLS)
print('RESEARCH_SYMBOLS', config.RESEARCH_SYMBOLS)"
```
- [ ] `exit=0`
- [ ] `BTC/ETH 0.809`, `BTC/SOL 0.744`, `ETH/SOL 0.719` — §0b reproduced
- [ ] `config.SYMBOLS  n=3  r_bar=0.7574  N_eff=1.193` — the ~1.2 anchor reproduced
- [ ] `0 interior gaps in 60 cells`
- [ ] `len(RESEARCH_SYMBOLS) >= 9`, anchor first, `config.SYMBOLS` **unchanged**
- [ ] The artifact exists, is non-empty, states the decision letter with its ratio, and states
      the negative finding: 6.7× rows, ~1.5× information

---

## Acceptance Criteria
- [ ] Tasks 1–13 and 15 complete; Task 14 skipped under D1 (record which path ran)
- [ ] All validation commands pass
- [ ] `data/correlation.py`, `correlation-report` → `_correlation_command`,
      `config.RESEARCH_SYMBOLS` + `CORRELATION_*` match contract §2/§7 names exactly
- [ ] Tests in `tests/test_correlation.py` (new) and `tests/test_storage.py` (extended) — the
      only two test paths Phase 2 owns (contract §8)
- [ ] The **286 pre-existing tests are green and unmodified**
- [ ] No type errors / no lint errors — neither tool is configured, N/A
- [ ] `config.SYMBOLS` byte-identical to its pre-phase value
- [ ] `gap-report`, `backfill`, `poll`, `regime`, `signal`, `backtest`, `walkforward` behave
      identically (contract §7)
- [ ] ≥8 symbols selected with mean pairwise r materially below the BTC-beta cluster (0.49 vs
      0.76 measured), every one gap-checked — the PRD's stated success signal
- [ ] The artifact records the decision letter, the effective-N ratio, and the honest
      "rows ≠ information" finding, every number produced by a committed command

## Completion Checklist
- [ ] Code follows discovered patterns (frame shape from `engine._df`, `None`-not-fake-number
      from `equity.py`, exit-code contract from `_gap_report_command`, appended config block)
- [ ] Error handling matches codebase style — `ValueError` with an actionable message for
      misaligned/insufficient data; `ccxt` exceptions caught only at the CLI boundary
- [ ] Logging follows conventions — `logging.getLogger("trading_bot")`, INFO for counts and
      dropped observations, no prints outside `cli.py`
- [ ] Tests follow test patterns — synthetic bar-by-bar SQLite fixtures, `START` base,
      hand-computed numerics, `@pytest.mark.network` gating, tier constants from config
- [ ] No hardcoded values — timeframes from `config.CORRELATION_TIMEFRAME` /
      `storage.TIMEFRAME_MS`, thresholds from `config.CORRELATION_*`
- [ ] Documentation updated — the selection rule is recorded in the function docstring
      (`universe.py` style) and the report artifact is committed
- [ ] No unnecessary scope additions — no rolling correlation, no 4h/1h correlation, no
      cross-sectional strategy, no `gap-report` change, no `scripts/` file
- [ ] Self-contained — no questions needed during implementation
- [ ] Degrees of freedom recorded (contract §12.6): **zero strategy DoF**; two deliberately
      non-binding screens; one pre-registered threshold with its 2.5% margin stated

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **The honest answer is unwelcome**: 20 liquid majors are close to one bet, so Phase 9's DSR stays brutal and the northstar stays out of reach on statistical grounds | **High — already measured** | **High** — it constrains what Phase 9 can ever prove | Make it a first-class reported result (Task 13) with the decision rule stated in advance. PRD honesty clause + contract §12.5 require this. Do **not** let "20 symbols stored" read as "trap solved" |
| The D1/D2 boundary turns on a 2.5% margin and someone tunes the threshold after seeing it | Medium | High — the "gate becomes theater" failure | Threshold set in Task 1, before Task 11 measures. Task 11's GOTCHA forbids adjusting it. The margin is recorded in the DoF log so the fragility is visible |
| Effective-N formula is contestable (Kish 1.834 vs participation ratio 2.925 for the same set) | Medium | Medium — a reader could quote the flattering one | Publish **both**; state which the rule uses (Kish) and why (reproduces §0b, conservative). Never quote the participation ratio alone |
| Pairwise-vs-listwise alignment silently changes the matrix on a future ragged panel | Low today (zero NaN measured), Medium after a backfill | Medium — non-PSD matrix corrupts the eigenvalue figure | Listwise enforced in `daily_return_frame`; dropped-observation count logged and printed; `CORRELATION_MIN_OVERLAP_BARS` rejects short-history candidates |
| pandas 3.0.3 `pct_change` semantics differ from 2.x code elsewhere | Low | Medium — a wrong-but-plausible return series | Explicit `close/close.shift(1) - 1.0`; hand-computed test pins `100→110→99 ⇒ +0.10, −0.10` |
| Trailing staleness misread as missing history, disqualifying all 20 symbols | Medium (40 of 60 cells show trailing gaps at any moment) | High — spurious D3 and a pointless backfill | `gap_integrity` pins `now_ms = last_ts + interval + 1`; the "stale, no hole → empty list" test is called out as the module's most important case |
| Off-diagonal mean computed as `corr.values.mean()`, deflating N_eff to a plausible wrong number | Medium (easy mistake) | High — corrupts the headline | Task 5 GOTCHA names it; a dedicated test asserts r̄ 0.7574 → N_eff 1.1929 |
| Scope creep into widening `config.SYMBOLS` because 20 symbols now "exist" in config's view | Medium | High — silently changes what every committed report refers to and what the live poller/scan iterate | NOT Building names it; Task 12's VALIDATE asserts `SYMBOLS` byte-identical and counts `config.SYMBOLS` references |
| `config.py` / `cli.py` merge collision with Phase 1, which runs concurrently (contract §11) | Medium | Low | Both append clearly-headed blocks at the end in phase order (§7); names are disjoint (P1 owns `STATE_DB_PATH`/`BENCHMARK_*`/`PNL_ATTRIBUTION_*`). Register `correlation-report` after `benchmark` |
| `fapi.binance.com` regresses to the 2026-07-05 blocked state | Low-Medium | **Low for this phase** | Under D1 no network is needed. The probe is opt-in; the test is marked. A verified offline fallback exists (v0.2.0 Phase 1 plan Task 7) but is out of scope |
| A future backfill of a recently-listed symbol truncates every symbol's span via listwise alignment | Medium if Task 14 runs | High — would silently halve `n_obs` for all | Task 14 GOTCHA 2; `CORRELATION_MIN_OVERLAP_BARS`; the report prints `n_obs`; the acceptance test requires effective N to *rise* by ≥0.25 |
| Report artifact drifts from the code that produced it | Low | Medium | `format_report` renders **only** from the frozen `CorrelationReport`; regeneration is one command, recorded in the config comment above `RESEARCH_SYMBOLS` |

## Notes

**THE HEADLINE CORRECTION (contract §0a).** The PRD scopes Phase 2 as "backfill uncorrelated
Binance futures symbols" and its Technical Risks table assumes 3 stored symbols. **20 are
stored**, all at 1d/4h/1h over 2023-01-01 → 2026-07-26, with **zero interior gaps across all 60
cells** (measured 2026-07-27). Phase 2 is therefore a verification, measurement, selection and
promotion phase; backfill is a contingency gated on decision D3. KNOWN-LIMITATIONS §4's "Only 3
symbols exist in the store" was true at the v0.2.0 merge and is superseded — this plan records
the supersession rather than editing that file.

**And the correction cuts both ways.** PRD Open Question #4 asks whether enough uncorrelated
symbols exist on Binance futures to break the ~0.76 trap. **Measured answer: no.** 19 of the 20
stored liquid majors carry BTC correlation between +0.56 and +0.81; the single exception,
TRXUSDT at +0.222, is also among the least liquid. Going 3 → 20 symbols multiplies rows by 6.7×
and independent information by ~1.5×. The best 8-symbol subset reaches an effective N of 1.87
against the core three's 1.19. **§0b's correlation trap is mitigated, not solved**, and within
this venue an effective N around 2–3 appears to be the ceiling. A true diversifier would require
a different asset class, which the PRD excludes.

**What this phase can and cannot fix, kept strictly apart.** Breadth fixes the gate's
*mechanical* `n_trades >= 30` condition — 9 symbols at v0.2.0's trade rate gives ~69 OOS trades
per 90-day holdout, comfortably over the floor. It barely moves *statistical independence*:
independent-equivalent OOS trades rise from ~9 to ~14, still under 30. Phase 9 must read its DSR
as governed by effective N ≈ 1.8, not by symbol count. Conflating the two is how "we added
symbols" becomes "we added evidence."

**Why the selection rule is deterministic rather than optimised.** Ranking by mean pairwise r and
a greedy minimum-average-correlation search agree on 7 of 8 members and differ by 0.009 in
effective N (measured). The selection is insensitive to the algorithm, so the simple rule wins:
no path dependence, no hidden degree of freedom, and a docstring a human can audit in thirty
seconds — the standard `universe.py:8-12` already set.

**Storage is a non-issue and the report should say so.** Measured 98.84 bytes/row across
1,183,372 rows in a 116,961,280-byte file; ten more symbols at 1d/4h/1h costs 39.9 MB against
256 GiB free. The binding constraint on breadth is correlation, not disk — which reframes the
PRD's "storage capacity check" from a risk into a one-line coefficient.

**Cross-phase coupling is deliberately thin.** Phase 2 depends on nothing and gates nothing
(contract §11). Its outputs are `config.RESEARCH_SYMBOLS`, one CLI subcommand, one module, one
report. Phase 9 consumes `RESEARCH_SYMBOLS` by passing it explicitly; Phase 1's
`benchmark.buy_and_hold(conn, symbols, ...)` already takes a symbol list, so the buy-and-hold
null can be computed over the **same** universe the strategy trades — which it must be, or the
comparison is between two different bets. **Flag to Phase 9: pass the same tuple to both.**

**No import from `trading_bot.backtest.*`.** Phase 1 is concurrently editing `equity.py` and
`walkforward.py`; `correlation.py` duplicates `DAY_MS` rather than importing, in keeping with the
existing duplication at `equity.py:22` and `walkforward.py:44` that v0.2.0 Phase 1's plan already
recorded as correct.

**Deliberate deviations from the letter of the brief, each stated:**
1. The brief says select "by an explicit rule combining liquidity and low BTC beta". The
   implemented rule *screens* on liquidity and beta but *ranks* on mean pairwise correlation,
   because r̄ is the exact quantity the effective-N formula consumes while beta is a proxy. Both
   screens are measured non-binding on the stored universe — recorded in the DoF log so they are
   not mistaken for fitted values.
2. Universe-wide gap verification lives in `correlation-report`, not `gap-report`. Contract §7
   permits exactly one flag addition to an existing subcommand across all of v0.3.0, and
   integrity is a selection criterion, so it belongs beside the selection.
3. The `@pytest.mark.network` reachability check lives in `test_correlation.py` because contract
   §8 gives Phase 2 only that file plus additions to `test_storage.py`, and the exchange probe is
   Phase 2's backfill precondition rather than a storage concern. The rationale goes in the
   test's docstring so the placement reads as deliberate.

**Obvious follow-ups deliberately not taken**: rolling / regime-conditional correlation
(correlation spikes in crashes, exactly when a pooled result matters most — but it needs a window
parameter this phase has no budget for); BTC-beta-neutralised returns as a signal input
(KNOWN-LIMITATIONS §0c lists it as never explored — Phase 4/8 territory); 15m for the 17 symbols
lacking it (measured 210 MB, retired tier).

**Memory note**: `~/.claude/projects/…/memory/binance-fapi-unreachable.md` records the 2026-07-05
block and the 2026-07-26 recovery. Re-verified **HTTP 200 in 0.19 s on 2026-07-27**; the entry is
accurate and needs only the new date appended if the operator wants the log current. It is not
load-bearing here — under D1 no network access is required at all.
