# trading-bot — maintainer's guide

A crypto strategy research engine: a plug-in framework for composing trading strategies, a
walk-forward validation gate that tries to reject them, and an evolutionary search that hunts
for one that passes.

**Version 0.3.0.** Nine phases shipped.

> ## Read this before anything else
>
> **This build is NOT VALIDATED. Do not commit capital on the basis of it.**
>
> v0.3.0 set out to test whether >50% annualized out-of-sample return is achievable by this
> strategy family. It answered **no**. Measured on 182 days of genuinely unseen data, the
> champion strategy returned **−45.47%** and failed **5 of the gate's 7 conditions**.
>
> **It is also alert-only. Nothing here places an order.** There is no exchange write path, no
> position sizing, and no margin or liquidation accounting. The engine reads public price data
> and prints signals.
>
> The full accounting is
> [`.claude/PRPs/reports/KNOWN-LIMITATIONS-v0.3.0.md`](.claude/PRPs/reports/KNOWN-LIMITATIONS-v0.3.0.md).
> Read it before trusting any number this tool prints. v0.2.0's
> [`KNOWN-LIMITATIONS.md`](.claude/PRPs/reports/KNOWN-LIMITATIONS.md) is **not** superseded and
> still stands.

---

## Setup

Python **3.11** specifically — the venv here is built on Homebrew's `python@3.11`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Every command below assumes `.venv/bin/python`. There is no console-script entry point, so the
CLI is always invoked as a module:

```bash
.venv/bin/python -m trading_bot.cli --help
```

Verify the install:

```bash
.venv/bin/python -m pytest -q -m "not network"     # expect: 1252 passed, 2 deselected
```

**No linter and no type checker are configured.** The test suite is the whole safety net. That
is a stated limitation, not an oversight — see KNOWN-LIMITATIONS-v0.3.0 §8.

### Credentials (usually unnecessary)

`.env` is **never loaded** — there is no `load_dotenv()` call anywhere. The only two variables
read are `BINANCE_API_KEY` / `BINANCE_API_SECRET` (`src/trading_bot/exchange/binance_client.py:71`),
both defaulting to empty, and **public OHLCV endpoints need no credentials**. If you ever do
need them, `export` them into the real environment; copying `.env.example` to `.env` will
silently do nothing.

### Two databases, deliberately separate

| Path | Holds | If you lose it |
|---|---|---|
| `data/ohlcv.db` | ~117 MB of price history | **Irreplaceable in practice** — a full re-backfill is hours of API calls |
| `data/state.db` | Trial ledger, review records, strategy versions, populations | Painful but rebuildable |

They are separate so a corrupt experiment log can never endanger the price store. This has
already paid for itself once: `state.db` was truncated to 0 bytes by a concurrent process and
`ohlcv.db` was untouched. **Do not point framework state at `ohlcv.db`.**

Both are gitignored. So is `data/ui_runs/`.

---

## The 17 commands, grouped by what you actually want to do

```
backfill  poll  gap-report          → keep data fresh
regime  signal                      → what would it trade right now
backtest  graph-backtest  benchmark → test over history
walkforward                         → the honest verdict
plugins  graph-validate             → inspect the framework
ui                                  → build strategies visually
review                              → learn from closed trades
evolve  campaign                    → search for a better strategy
detector-report  correlation-report → diagnostics
```

---

## Use case 1 — Keep the price data fresh

**Do this first, every time.** Every other command reads stored candles; stale data silently
produces stale conclusions.

```bash
# 1. What's missing or stale right now?
.venv/bin/python -m trading_bot.cli gap-report

# 2. Fill the gaps (needs network; resumes where it left off)
.venv/bin/python -m trading_bot.cli backfill

# 3. Confirm clean
.venv/bin/python -m trading_bot.cli gap-report
```

Narrow the work when you only need one series:

```bash
.venv/bin/python -m trading_bot.cli backfill --symbol BTCUSDT --timeframe 1h --start 2024-01-01
```

`--symbol` and `--timeframe` are **repeatable**, not comma-separated.

To keep a running process topping up the latest candles on a schedule:

```bash
.venv/bin/python -m trading_bot.cli poll
```

**Gotchas.**
- Both `backfill` and `poll` need `fapi.binance.com` reachable. It has been intermittently
  blocked from this machine before; check with
  `curl -sS -o /dev/null -w '%{http_code}\n' https://fapi.binance.com/fapi/v1/ping`
  before assuming the code is broken.
- **`poll` can persist a partial daily bar.** If polling stops mid-day, the current `1d` row
  holds only the hours seen so far, and `MAX(ts)` will hand you a bar that looks closed but
  isn't. This is a real lookahead trap the project hit five separate times. Treat the last bar
  as suspect; `gap-report` is how you spot it.
- Stored timestamps are candle **OPEN** times, epoch **milliseconds**, **UTC**. Every timeframe
  is fetched natively — nothing is resampled.

---

## Use case 2 — Ask what it would trade right now

```bash
# Which regime is each symbol in?
.venv/bin/python -m trading_bot.cli regime

# Any signals under the regime-matched method?
.venv/bin/python -m trading_bot.cli signal
```

Both accept `--symbol` (repeatable) and `--as-of YYYY-MM-DD` to ask historically.

**Gotchas.**
- The regime classifier needs **207 daily bars** of warmup. With history from 2023-01-01, the
  first non-`uncertain` label lands **2023-07-27**. Any span starting earlier yields zero
  trades — not a bug.
- **Ranging regimes currently produce no signals at all.** The mean-reversion fade sleeve was
  measured and dropped (`FADE_ENABLED = False`); the code is kept intact and tested so the
  decision stays reversible. Trending regimes are the only ones that fire.
- Output is an alert. Nothing is placed.

---

## Use case 3 — Test a strategy over stored history

Three tools, and picking the wrong one is the most common mistake:

| Command | Runs | Use when |
|---|---|---|
| `backtest` | The **frozen legacy** v0.2.0 signal path | You want v0.2.0's measured behaviour reproduced |
| `graph-backtest` | A **serialized strategy graph** | Almost always — this is the framework path |
| `benchmark` | Buy-and-hold, no strategy | To find out whether the strategy beats doing nothing |

```bash
# The framework path — --graph is required
.venv/bin/python -m trading_bot.cli graph-backtest --graph data/strategies/thin-slice.strategy.json

# With the per-position reward:risk audit trail
.venv/bin/python -m trading_bot.cli graph-backtest \
  --graph data/strategies/tech-pattern.strategy.json --audit --rr-report

# The null hypothesis — always run this next to any result
.venv/bin/python -m trading_bot.cli benchmark --start 2023-07-27 --end 2026-07-27

# The frozen legacy path — only to reproduce v0.2.0's measured behaviour
.venv/bin/python -m trading_bot.cli backtest --start 2023-07-27 --end 2026-01-26
```

**Always run `benchmark` beside a backtest.** v0.2.0's central error was reporting returns with
no null to compare against; the gate "passed" a strategy that lost to holding. A
`beats_benchmark` pass means only *"lost less than holding"* — and over the v0.3.0 holdout the
basket itself lost 36%.

Four strategies ship in `data/strategies/`:

| Strategy | Branches | Editable in the UI |
|---|---|---|
| `donchian-v020` | 2 | no |
| `thin-slice` | 2 | no |
| `thin-slice-noconfirm` | 2 | no |
| `tech-pattern` | 1 | **yes** |

---

## Use case 4 — Get an honest verdict (the gate)

This is the one that tries to **reject** your strategy.

```bash
.venv/bin/python -m trading_bot.cli walkforward \
  --graph data/strategies/thin-slice.strategy.json \
  --start 2023-07-27 --end 2026-01-26
```

It trains on rolling windows, tests out-of-sample, and scores **7 conditions**
(`src/trading_bot/backtest/walkforward.py:100`): sample adequacy, Sharpe ≥ 1.0, deflated Sharpe
> 0.95, max drawdown ≤ 25%, per-symbol expectancy all positive, and the two benchmark
comparisons.

**The thing to understand about this gate:** every evaluation you run is charged as a
multiple-testing **trial**, and the deflated-Sharpe requirement rises with the trial count. At
1 trial the gate wants annualized Sharpe 2.39; at 462 trials it wants 7.58. **Searching harder
makes the bar higher.** That is the honest arithmetic of trying many strategies, and it is why
the two legitimate ways to improve a verdict are *more independent observations* and *a
smaller pre-registered search* — never one more generation.

Trials are persisted to `data/state.db`, so the count survives restarts and you cannot reset it
by rerunning.

**Gotcha.** Passing `--graph` **changes what gets swept**: the parameter grid collapses to the
single run-level axis `max_hold_bars`, because a graph already carries its own parameters.
Omitting `--graph` runs the legacy engine path with the full default grid instead. Those are two
different experiments, so do not compare their outputs directly.

---

## Use case 5 — Build and edit strategies visually

```bash
.venv/bin/python -m trading_bot.cli ui
# then open http://127.0.0.1:8770
```

The dashboard lists plug-ins and data coverage, composes strategies, launches backtests and
gate runs with live progress over SSE, and browses reviews and evolution generations.

> **[Full visual guide with screenshots →](docs/ui/README.md)** — all six views, and a
> use-case-by-use-case table of what the UI can and cannot do.

**The UI covers 2 of these 9 use cases completely** (this one and the gate), 5 partially, and 2
not at all. It launches exactly three run kinds — `backtest`, `gate`, `evolve`. It deliberately
cannot write price data (use case 1) or run the campaign (use case 8), because those are the
hardest actions to undo. Most importantly: **the UI shows no benchmark leg on a backtest**, so
the buy-and-hold null in use case 3 is CLI-only.

**Loopback only, and not configurable past that.** There is no auth, and the server refuses a
non-loopback host with no override flag — this machine holds the trading logic and the
irreplaceable price store. Do not try to expose it to a LAN.

**Gotchas.**
- **Only *uniform* graphs are editable.** The composer maps one detector stage to one branch, so
  a multi-detector strategy stays editable as long as every branch agrees on policy,
  confirmations (same keys, same order), regimes and exits. A graph whose branches genuinely
  diverge — the per-branch confirmations an evolved champion grows, or a disabled branch — still
  renders read-only, and a POST that would flatten it is refused with HTTP 409 rather than
  silently discarding the other branches. All three shipped multi-branch strategies
  (`donchian-v020`, `thin-slice`, `thin-slice-noconfirm`) diverge and remain read-only;
  `tech-pattern`, `tech-pattern-double-confirmations` and `cup-and-handle` are editable and
  round-trip to a bit-identical graph hash, so editing them is lossless.
- **One heavy run at a time.** A second concurrent backtest or gate run gets HTTP 429.
- Run bookkeeping lands in `data/ui_runs/` — gitignored, derived, safe to delete.

---

## Use case 6 — Learn from closed trades

`--strategy` and `--version` are mutually exclusive and one is required. The normal flow is
two steps: register a graph to get a version id, then work with that id.

```bash
# 1. Register the graph. This EXITS immediately, printing the version_id —
#    it does not run a review.
.venv/bin/python -m trading_bot.cli review \
  --strategy data/strategies/thin-slice.strategy.json --register

# 2. Review that version's closed trades. Start read-only.
.venv/bin/python -m trading_bot.cli review --version <id> --diagnose-only

# 3. Forward-test a refined candidate through the gate on unseen-by-it data
.venv/bin/python -m trading_bot.cli review --version <id> \
  --forward-start 2026-01-26 --forward-end 2026-07-27
```

Each closed trade is scored against its own prediction — did the take-profit capture the move,
did the stop guard a sensible loss, was the pace consistent with the target.

**Reviews inform; the gate decides.** No `REVIEW_*` threshold ever selects a strategy. Start
with `--diagnose-only`, and use `--no-persist` when experimenting.

**Gotchas.**
- Pace claims need **≥ 30 trades and ≥ 180 days**. Below either floor you get
  `insufficient-sample` and no suggestions — deliberately, because v0.2.0 published a +68%
  headline extrapolated from 23 trades.
- `--loop` acts on suggestions automatically. It has produced a **measurably harmful** change
  before (dropping confirmations that were worth +0.45%/trade). That specific bug is fixed, but
  run it with your eyes open.

---

## Use case 7 — Search for a better strategy

```bash
# Size the search against your hardware FIRST
.venv/bin/python -m trading_bot.cli evolve --calibrate --repeats 3 \
  --seed-graph data/strategies/thin-slice.strategy.json

# Then run it
.venv/bin/python -m trading_bot.cli evolve \
  --seed-graph data/strategies/thin-slice.strategy.json \
  --population 24 --generations 8 --report out.md

# Resume an interrupted campaign
.venv/bin/python -m trading_bot.cli evolve --resume <campaign-id>
```

A population of strategy graphs, mutated by parameter jitter and graph edits, scored **only**
through the gate. Scoring without a ledger handle is structurally impossible — the ledger is
the first positional argument of `GateOracle` with no default, so bypassing it is a `TypeError`
rather than a discouraged practice.

Use `--dry-run` to see the plan without spending evaluations.

**Gotchas.**
- **Every evaluation raises the bar for the eventual verdict** (see use case 4). 192
  evaluations is not "more thorough" than 24 for free.
- **Fitness rewards not trading.** `excess_sharpe` is easier to win by barely trading against a
  falling benchmark; one generation winner had a *single* trade at Sharpe 4.608. A champion
  trade floor patches selection, but the search itself is still biased. This is a known,
  unresolved design flaw — if you change it, pre-register the change before searching.
- Zero of 192 candidates have ever passed all seven conditions.

---

## Use case 8 — Run the pre-registered campaign

```bash
.venv/bin/python -m trading_bot.cli campaign --stage probe     # cheap preflight
.venv/bin/python -m trading_bot.cli campaign --stage evolve
.venv/bin/python -m trading_bot.cli campaign --stage holdout   # ← spends the holdout
.venv/bin/python -m trading_bot.cli campaign --stage report
```

Exit codes are **not** pass/fail in the usual sense:

| Code | Meaning |
|---|---|
| 0 | Gate passed |
| 1 | Gate failed — **a valid, completed outcome.** An honest "no" exits 1 by design |
| 2 | Ambiguous or aborted. **Only this means the run failed to produce a verdict** |

### The holdout is a one-shot resource

`[2026-01-26, 2026-07-27)` — 182 days no evolution generation has ever scored. **It has already
been consumed once.** Running `--stage holdout` again is refused unless you pass
`--force-holdout-rerun REASON`, which is recorded and stamps the result `NOT A CLEAN HOLDOUT`.

Both endpoints were forced by measurement, not chosen: the start is the frozen ceiling of the
evolution span, the end is the last *complete* daily bar.

**Explicitly forbidden as a reaction to a bad verdict:** a lower trial count, a lower trade
floor, a shorter holdout, a different champion rule, or "one more generation." A second run is
a **new campaign** and must be charged the combined trial count. This is the difference between
a measurement and a story.

---

## Use case 9 — Diagnostics

```bash
.venv/bin/python -m trading_bot.cli plugins                    # what's registered
.venv/bin/python -m trading_bot.cli plugins --kind detector
.venv/bin/python -m trading_bot.cli graph-validate data/strategies/thin-slice.strategy.json
.venv/bin/python -m trading_bot.cli detector-report --coverage # 144-row pattern ledger
.venv/bin/python -m trading_bot.cli correlation-report --out report.md
```

`detector-report` **suppresses** every rate and expectancy figure below 20 trades and refuses
to read any span ending within 150 days of now — the holdout belongs to the campaign, and no
diagnostic may peek at it.

`correlation-report` regenerates `config.RESEARCH_SYMBOLS`. **Never hand-edit that tuple or
reorder it** — downstream tables key off the exact ordering.

**The finding worth internalising:** 9 symbols carry a Kish effective N of **1.87**. Going from
3 to 9 symbols bought ~1.54× the independent information, not 3×. Within Binance perps there is
no genuinely uncorrelated crypto, so read every sample size as "fewer than two independent
instruments." Adding more liquid majors adds rows, not information.

---

## Adding a plug-in

Drop a module under the right `src/trading_bot/plugins/` subpackage and it is discovered
automatically — `registry.load_all()` walks the package. **No engine-core edit is required**;
that property is enforced by test, and `engine.py`, `signals/`, `regime/`, `indicators/`,
`risk/`, `metrics.py` and `equity.py` showed an empty diff across the phase that introduced the
framework.

Kinds: `data`, `detector`, `confirmation`, `policy`, `filter`, `reviewer`, `mutator`.

**Constraints that will bite you:**
- **No new indicator dependency, ever.** `pandas-ta` vanished from PyPI and TA-Lib needs a C
  library; every indicator here is hand-rolled. This is a standing rule, not a preference.
- `pandas.Series.pct_change()` is **unsafe** in this codebase — pandas 3.x changed its
  behaviour. Compute returns as `s / s.shift(1) - 1.0`.
- Declare `ParamSpec` bounds so the mutator can jitter your parameters legally.
- Two geometry implementations coexist on purpose: `signals/patterns.py` is frozen by a parity
  test and reproduces v0.2.0 exactly; `plugins/detectors/_geometry.py` is the corrected one.
  **They are allowed to disagree.** Do not "fix" the frozen one.

---

## Config

`src/trading_bot/config.py` is the single source of truth and is **densely commented with the
measurement behind each constant** — read the comment before changing the value. Constants are
grouped by the phase that introduced them and namespaced by prefix (`EVO_*`, `CAMPAIGN_*`,
`HOLDOUT_*`, `REVIEW_*`, `DETECTOR_*`, `UI_*`).

Some values are **frozen** and say so: Donchian 20/55 are canonical, MACD 12/26/9 is canonical,
`ATR_STOP_MULTIPLE = 1.5` was derived from a cost constraint rather than from returns. Changing
any of them consumes a degree of freedom that must be logged.

---

## Where the truth lives

| Document | What it is |
|---|---|
| [`KNOWN-LIMITATIONS-v0.3.0.md`](.claude/PRPs/reports/KNOWN-LIMITATIONS-v0.3.0.md) | **Start here.** The honest v0.3.0 accounting |
| [`KNOWN-LIMITATIONS.md`](.claude/PRPs/reports/KNOWN-LIMITATIONS.md) | v0.2.0's, still current |
| [`_shared-architecture-contract.md`](.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md) | The invariants every phase had to honour |
| [`technical-pattern.md`](.claude/technical-pattern.md) | The 144-row pattern catalog — **machine-parsed by a test; edits change test expectations** |
| [`pivot-guide.md`](.claude/pivot-guide.md) | Why v0.3.0 exists (its charter; northstar since answered *no*) |
| [`DEPRECATED-ARTIFACTS.md`](.claude/PRPs/reports/DEPRECATED-ARTIFACTS.md) | Artifacts untracked from git, and how to recover them |
| `.claude/PRPs/plans/v0.3.0/` | Per-phase plans |

A guided code tour of the same ground is in `.tours/` (VS Code CodeTour extension).

---

## Before you commit

```bash
.venv/bin/python -m pytest -q -m "not network"    # 1252 passed, 2 deselected
```

The 2 deselected are network tests needing a reachable `fapi.binance.com`.

**Maintainer's rule of thumb, inherited from this project's own discipline:** correctness is
never negotiated against results. Two fixes in this repo's history moved performance in the
*unfavourable* direction and were kept anyway; one moved it favourably and had that direction
recorded precisely *because* it flattered. When a change makes the numbers look better, that is
when to write down why.
