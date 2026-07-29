# Plan: Builder UI (v0.3.0 Phase 7)

## Summary
A local single-page web app — Python **stdlib** `http.server.ThreadingHTTPServer` plus one hand-written HTML/CSS/JS page, **zero new dependencies** — that makes the three-lane framework (Data / Strategy / Feedback) visible and operable. It composes strategies from the Phase 3 registry, writes the *same* `data/strategies/<name>.strategy.json` serialization the engine reads, launches backtest / gate / evolution runs without blocking the server, streams progress over SSE, and renders the gate verdict **per condition** beside the buy-and-hold benchmark. Started with `python -m trading_bot.cli ui`, bound to `127.0.0.1` only.

## User Story
As the sole operator, I want to compose a strategy from registered plug-ins in a browser, run it against the gate, and see its equity curve next to buy-and-hold with every gate condition itemised, so that the framework is something I can *operate* instead of a set of CLI flags to remember — and so a failing candidate tells me *which* condition failed.

## Problem → Solution
**Current**: every capability is a CLI subcommand printing fixed-width text (`cli.py:396-439`). A strategy is a `BacktestParams` dataclass edited in Python (`engine.py:91-106`). Seeing performance means a separate script that regenerates a whole HTML file (`scripts/build_performance_chart.py`). No frontend of any kind exists in the repo, and `pandas-ta` vanishing from PyPI is the standing lesson about what a new dependency costs.
**Solution**: `src/trading_bot/ui/` — `api.py` (pure request→JSON over `framework`/`feedback`/`evolution`; imports nothing from `http`) + `server.py` (transport only) + three static files. Three lanes on one page. Controls render from the *same* `ParamSpec` declarations the registry exposes and Phase 6's mutators jitter; persistence goes through the *same* `StrategyGraph.to_dict()` the executor reads. Long work runs as a job (thread for gate runs, **subprocess** for campaigns) whose state lives on disk, so a reload or a closed browser loses nothing.

## Metadata
- **Complexity**: **Large** — 8 new files + 3 modified, ~1,300 net new lines (≈750 Python, ≈550 HTML/CSS/JS), one new architectural surface (HTTP), zero new dependencies.
- **Source PRD**: `.claude/PRPs/prds/self-learning-pattern-framework.prd.md` · **Phase**: 7 — Builder UI
- **Binding contract**: `.claude/PRPs/plans/v0.3.0/_shared-architecture-contract.md` (§3, §4, §5, §6, §7, §8, §10 row 6, §12)
- **Estimated Files**: 11 (8 CREATE, 3 UPDATE) · **Tasks**: 16
- **Depends on**: Phase 5 (`feedback/`), Phase 6 (`evolution/`); transitively 1, 3, 4. **Gates Phase 9.** **Parallel with** Phase 8 (no shared files).
- **Test baseline**: **286 tests collected** (`.venv/bin/python -m pytest --collect-only -q`, verified 2026-07-27). All 286 must stay green.

---

## UX Design

### Before

```
$ python -m trading_bot.cli walkforward
one-shot OOS (pooled):
  trades=23  win_rate=43.48%  expectancy=0.0412%  profit_factor=1.31
  sharpe=1.17  sortino=1.44  dsr=0.0210  max_dd=19.13%  ann_return=68.20%
GATE: FAIL     <-- WHICH condition? Diff the numbers against constants in
                   another file, by hand. No benchmark in this output at all.
                   To change the strategy: edit engine.py. To see a chart:
                   re-run the whole walk-forward through
                   scripts/build_performance_chart.py (~minutes).
```

### After

```
$ python -m trading_bot.cli ui
trading_bot.ui listening on http://127.0.0.1:8770  (loopback only, no auth)
  strategies: data/strategies   state: data/state.db   runs: data/ui_runs
  registry: 27 plug-ins across 7 kinds

+- 1 . DATA ------+ +- 2 . STRATEGY -----+ +- 3 . FEEDBACK ------------+
| source          | | name [hs-vol-macd] | | [Backtest] [Run gate]     |
|  (o) data.ohlcv | |                    | | [Start evolution]         |
| symbols         | | DETECTOR           | | campaign [ui-smoke      ] |
|  [x] BTCUSDT    | |  [head-and-should.]| | ! adds ~156 trials to     |
|  [x] ETHUSDT    | |   shoulder_tol .03 | |   this campaign's ledger  |
|  [x] SOLUSDT    | |     |---o------|   | +- runs ------------------+ |
|  [ ] AVAXUSDT   | |  [+ detector]      | | * gate    running 2/13   | |
|  [ ] LINKUSDT+15| | CONFIRMATION       | | o evolve  gen 8/40       | |
| span            | |  [volume-breakout] | | v backtest done          | |
|  [2023-07-27]   | |   ratio 1.50|--o-| | +--------------------------+ |
|  [2026-07-26]   | |  [macd-cross     ] | (also below: review records
| coverage        | |  [+ confirmation]  |  and a generations table)
|  1d    1304 ok  | | POLICY (exactly 1) |
|  4h    7820 ok  | |  [measured-move  ] |
|  1h   31279 ok  | | FILTER             |
|  0 gaps         | |  [rr-after-costs ] |
| (read-only:     | |   rr_target_min 2.0|
|  backfill is a  | | [Validate] [Save]  |
|  CLI job)       | | [Load v] schema v1 |
+-----------------+ +--------------------+

+- THE GATE - 7 conditions, all mandatory (AND, not average) --------------+
| sample_adequacy        23 trades        >= 30       x FAIL               |
| sharpe                 1.17             >= 1.0      v PASS               |
| dsr                    0.0210           >  0.95     x FAIL               |
| max_drawdown           19.13%           <= 25%      v PASS               |
| per_symbol_expectancy  3 of 3 positive  all > 0     v PASS               |
| beats_benchmark_return +68.2%  vs  +29.4%           v PASS               |
| beats_benchmark_sharpe   1.17  vs    0.73           v PASS               |
| -- n_trials charged to DSR: 156 (campaign ui-smoke, cumulative 156)      |
+-------------------------------------------------------------------------+

+- Equity vs buy-and-hold ------------------------------------------------+
| 2.4x|                             ,--.    .... basket   2.17x           |
|     |                 ....,-------'  `-   ---- strategy 1.11x           |
| 1.6x|     ...,--------'                   #### one-shot OOS             |
| 1.0x+---,-'        \___                                                 |
|     +--+---------+---------+---------+------####--                      |
|     2023Q4    2024Q3    2025Q2    2026Q1                                |
|  [table view v]   (every series also printed as numbers)                |
+-------------------------------------------------------------------------+
```

### Compose flow — what a click actually does

```
GET /api/plugins -> {kind: [{key, name, rationale, params:{ParamSpec}}]}
  app.js renders ONE control PER ParamSpec entry:
    int->number step1 . float->number+range . bool->checkbox . choice->select
    min/max straight from ParamSpec.bounds -- the SAME bounds Phase 6 jitters in
[Save]  POST /api/strategies/<name>  {name, stages:[{kind,key,params}]}
  -> _stages_to_graph(): NodeSpec x N -> StrategyGraph -> to_dict()
  -> data/strategies/<name>.strategy.json     (Phase 3 owns the format)
[Run gate] POST /api/runs {kind:"gate", strategy, symbols, start, end, campaign}
  -> 202 {run_id} immediately; Tier A job thread runs
     walk_forward_pooled(conn, symbols, strategy=graph)
     -> trial_ledger rows -> run dir result.json
GET /api/runs/<id>/events (SSE) -> log + progress + done
```

### Run / monitor — the three states

```
STARTING            RUNNING (reloaded twice, nothing lost)     FINISHED / TAB CLOSED
POST /api/runs      GET .../events?offset=41904                Tier A: thread ends,
 -> 202 {run_id}      retry: 2000                               result.json written,
status.json           id: <byte offset>                         done event, verdict
 {state:"running",    event: log      data: "fold 4/13 ..."     Tier B: subprocess
  tier:"A"|"B",       event: progress data:{gen:8,pop:64,...}    keeps running,
  pid:null|12345,     :heartbeat                  (every 15s)    detached from any
  argv:[...],         event: done     data:{exit:1}              HTTP connection
  started_ms:...}   reload -> re-GET /api/runs, see "running",  restart: Tier A ->
                    reopen SSE at the byte offset consumed.      "orphaned";
                    No line lost or duplicated.                  Tier B -> pid probe
```

### Interaction Changes

| Touchpoint | Before | After | Notes |
|---|---|---|---|
| Compose a strategy | Edit `engine.py`/`config.py`, restart | Pick plug-ins + params in lane 2, Save | Writes Phase 3's serialization, not a UI format |
| Param bounds | Tribal knowledge in comments | Rendered from `ParamSpec.bounds` | Same declaration the mutators jitter (§3) |
| Gate verdict | `GATE: FAIL`, one word | 7 rows: condition · measured · threshold · ✓/✗ | §4; fixes KNOWN-LIMITATIONS §0's blind spot |
| Benchmark | Nowhere in CLI output | Basket curve + 2 explicit gate rows | KNOWN-LIMITATIONS §0 is the whole reason |
| Long run | Blocks the terminal | 202 + run id; disk-backed; reload-safe | A closed browser does not kill a campaign |
| Trial cost of a run | Invisible | Shown *before* launch and after | §4.2 — the ledger is not decoration |
| Chart | Re-run a script, regenerate a file | Redraw from JSON in the open page | `build_performance_chart.py` stays for reports |
| Access | n/a | `http://127.0.0.1:8770`, loopback, no auth | Not a security boundary — Task 8 |

### UX edge cases

- **Empty registry**: lane 2 says "no plug-ins registered — run `cli plugins`", not empty selects. **Zero stored symbols**: lane 1 says so; run buttons disable.
- **A graph the linear editor cannot represent** (branching, if Phase 3's graph is a real DAG): loads **read-only** with the reason. It must **never silently flatten and re-save**.
- **A saved graph naming a plug-in that no longer registers**: that stage renders red with the missing key; Save disabled. (Same instinct as §3's "import errors are FATAL": a silently-dropped stage reads as "tested and found wanting".)
- **Run already in progress** (Tier A limit 1): button reads "busy — 1 gate run at a time"; `POST /api/runs` → `429`.
- **High contrast / no `prefers-color-scheme`**: every verdict carries a mark **and** a word (`✓ PASS` / `✗ FAIL`), never colour alone; every chart ships a table view — the relief already used at `build_performance_chart.py:388-394`, `:619-624`.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `_shared-architecture-contract.md` | §3,4,5,6,7,8,10 | BINDING. §3 = the `ParamSpec` your controls render. §4 = 7 gate conditions + `BenchmarkResult`. §5 = the one graph→Trade seam. §7 = reserved `UI_HOST`/`UI_PORT` + `ui` subcommand |
| P0 | `framework/graph.py` | all (P3) | `StrategyGraph`, `NodeSpec`, `to_dict`/`from_dict`, `SCHEMA_VERSION`. **The file format. The UI never hand-builds this JSON** |
| P0 | `framework/registry.py` | all (P3) | `REGISTRY`, `KINDS`, `register`, `get`, `by_kind`, `load_all()`. `PluginSpec.params: dict[str, ParamSpec]` is what lane 2 renders |
| P0 | `framework/contracts.py` | `ParamSpec` | Fields (`default`, `bounds`/`choices`, `kind`) map 1:1 onto HTML controls. Verify names before writing `_paramspec_to_dict` |
| P0 | `backtest/walkforward.py` | 84-114 + P1's `GATE_CONDITIONS`, `_evaluate_gate`, `strategy=` | What you serialise. P1 made `_evaluate_gate` return `dict[str, bool]`; P3 added `strategy=`. Verify both landed |
| P0 | `backtest/benchmark.py` | all (P1) | `BenchmarkResult.per_symbol`/`.basket` — the second equity series and the 2 benchmark gate rows |
| P0 | `scripts/build_performance_chart.py` | 85-137, 174-214, 432-448, 454-535, 642-682 | Charting prior art: SVG path/tick/label helpers, gate scorecard shape, JSON embed, light/dark tokens, tooltip wiring. Self-contained-HTML conventions live here |
| P0 | `src/trading_bot/cli.py` | 39-56, 146-213, 215-234, 356-368 | `add_parser` + `main()` dispatch + `_date_arg` + `_fmt`. Your `ui` subcommand follows this exactly |
| P1 | `data/storage.py` | 30-72 | `_db_lock`, `connect()` (WAL, `check_same_thread=False`). Read before opening a connection from a thread |
| P1 | `data/statestore.py` | all (P1) | `connect()` for `state.db`; the module using a table owns its `CREATE TABLE IF NOT EXISTS` |
| P1 | `evolution/population.py`, `runner.py` | all (P6) | Where `generations`/`population_members` are written, and the accessor to read them by. **No accessor = a Phase 6 gap to report** (Task 5) |
| P1 | `feedback/records.py`, `versioning.py` | all (P5) | `ReviewRecord` fields; the version registry the review table renders |
| P1 | `tests/test_backtest.py` | 27-57 | Tier constants from config, autouse cache fixture, `make_trade` — both test files mirror this |
| P1 | `src/trading_bot/config.py` | 10, 186-196 | `SYMBOLS`, `date_to_ms`. Your `UI_HOST`/`UI_PORT` block appends at the end, phase-ordered (§7) |
| P2 | `scripts/bruteforce/registry.py` | 28-35, 107-153, 162-179 | Why `rationale` is mandatory and import errors fatal. `/api/plugins` **displays** the rationale — that is what stops it being a docstring |
| P2 | `.claude/PRPs/reports/KNOWN-LIMITATIONS.md` | §0, §8 | §0: equity-vs-buy-and-hold is the view that matters. §8: `build_review_chart.py` is stale and will not run — do not port or call it |
| P2 | PRD | User Flow, Technical Risks | Your spec, and the "UI scope eats the project" risk the NOT-Building list contains |

## External Documentation

| Topic | Source | Key Takeaway |
|---|---|---|
| `http.server` | Py3.11 stdlib `http.server`, `socketserver` | `ThreadingHTTPServer` = `HTTPServer` + `ThreadingMixIn`: thread per request; `daemon_threads = True` so an open SSE stream cannot hold Ctrl-C hostage. **`SimpleHTTPRequestHandler` deliberately unused** — it maps URLs onto the filesystem relative to cwd, a traversal surface. Subclass `BaseHTTPRequestHandler`, serve a fixed allowlist |
| Server-Sent Events | WHATWG HTML §server-sent-events | `text/event-stream`; `event:`/`data:`/`id:`/`retry:` lines, `\n\n`-terminated; `:comment` = heartbeat. Browser `EventSource` auto-reconnects and replays `Last-Event-ID`. **No library either side** |
| `dataclasses.asdict` | Py3.11 stdlib | Serialises `ParamSpec`, `FoldResult`, `BenchmarkResult`. Tuples→lists is fine; `float('nan')` is **not** valid JSON (Task 5) |

`KEY_INSIGHT`: the entire client is three files served from a Python dict — no framework research needed. `APPLIES_TO`: Tasks 10–12. `GOTCHA`: "no network at runtime" also rules out webfonts, icon sets and `<script src>` — `system-ui` and inline SVG only, as `build_performance_chart.py:482`, `:636-639` already do.

---

## Patterns to Mirror

### CLI_SUBCOMMAND_AND_HANDLER
```python
# SOURCE: src/trading_bot/cli.py:128-144 (parser) and :199-209 (dispatch)
    wf_parser = subparsers.add_parser("walkforward", help="Walk-forward validation ...")
    wf_parser.add_argument("--start", type=_date_arg, help="UTC start date YYYY-MM-DD ...")
    elif args.command in ("backtest", "walkforward"):
        conn = connect(args.db)
        exit_code = _walkforward_command(conn, symbols, start_ms=start_ms, end_ms=end_ms)
        conn.close()
        sys.exit(exit_code)
```
`_date_arg` (`cli.py:215-234`) converts dates to epoch-ms **at the argparse boundary**; `api.py` does the same at the HTTP boundary.

### CONFIG_APPEND_BLOCK
```python
# SOURCE: src/trading_bot/config.py:141-151 — comment-headed per phase, says WHY
# the value exists and whether it is frozen or sweepable.
WF_MIN_TRADES = 30  # minimum trades for a combo / gate to count (Phase 7: was 5, noise-fit)
```

### DB_CONNECT_AND_LOCK
```python
# SOURCE: src/trading_bot/data/storage.py:30-33, :52-57
_db_lock = threading.Lock()   # serializes all DB access across callers sharing a conn
conn = sqlite3.connect(str(db_file), check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
```
**The UI holds no long-lived connection.** Every route and job opens its own via `storage.connect()`/`statestore.connect()` and closes it in `finally` — a request thread and a job thread sharing one cursor is the classic `ThreadingHTTPServer` `sqlite3.ProgrammingError`.

### SVG_GEOMETRY_HELPERS
```python
# SOURCE: scripts/build_performance_chart.py:85-88, :90-94, :97-107, :121-137
def _path(values, x_of, y_of):
    return "M" + " L".join(f"{x_of(i):.2f},{y_of(v):.2f}" for i, v in enumerate(values))

def _nice_ticks(lo, hi, count=5):
    # The magnitude MUST come from log10, not the digit count of int(raw): for a
    # range like 0.70-2.20 that floored the step at 0.01 and rendered 16 gridlines
    # instead of 5. Bug found and fixed once -- do not reintroduce it in JS.
    raw = (hi - lo) / max(1, count)
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0

def _spread(labels, min_gap=13.0, top=0.0, bottom=1e9):
    # Nudge overlapping end-of-line labels apart, preserving order. Needed again:
    # strategy and basket curves converge.
```
Port `_path`, `_area_path`, `_nice_ticks`, `_spread` into `app.js` as `svgPath`, `areaPath`, `niceTicks`, `spreadLabels` — same algorithms, comments carried over.

### GATE_SCORECARD_ROW_SHAPE
```python
# SOURCE: scripts/build_performance_chart.py:174-214 — (label, value, threshold, ok)
        ("Sharpe (annualised)", _fmt_num(eq["sharpe"]), f"≥ {walkforward.GATE_MIN_SHARPE}",
         eq["sharpe"] is not None and eq["sharpe"] >= walkforward.GATE_MIN_SHARPE),
# and :388-394 — every verdict ships a MARK and a WORD, never colour alone:
        f'<span class="chip {"good" if ok else "crit"}">'
        f'{"✓" if ok else "✗"} {"PASS" if ok else "FAIL"}</span>'
```
Yours has **7** rows and **reads `result.gate: dict[str, bool]`** keyed by `walkforward.GATE_CONDITIONS` (§4) rather than recomputing the booleans — recomputing would be a second source of truth for the verdict itself.

### THEME_TOKENS_LIGHT_AND_DARK
```css
/* SOURCE: scripts/build_performance_chart.py:454-477 — copy verbatim into app.css */
:root { color-scheme: light dark; }
.viz-root {
  --surface-1:#fcfcfb; --plane:#f9f9f7; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --border:rgba(11,11,11,0.10); --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --s4:#eda100; --good:#0ca30c; --crit:#d03b3b; --oos:rgba(42,120,214,0.07);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root { --surface-1:#1a1a19; /* ... */ } }
:root[data-theme="dark"] .viz-root { --surface-1:#1a1a19; /* ... */ }
```
Already contrast-validated; identical tokens make the live UI and the committed report chart read as one system.

### ESCAPE_EVERY_INTERPOLATED_STRING
```python
# SOURCE: scripts/build_performance_chart.py:383-387
        f'<div class="tile"><div class="tl">{html.escape(t)}</div>'
```
In `app.js` the equivalent is `textContent`/`createElement` for every plug-in name, rationale, symbol, error and log line. `innerHTML` only for markup you author with no interpolation.

### FROZEN_DATACLASS_RESULT
```python
# SOURCE: src/trading_bot/backtest/walkforward.py:98-114
@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]
    final_params: BacktestParams
    final_max_hold_bars: int | None   # :104-107 — a swept value absent from the
                                      # result makes the verdict unreproducible
    oos_start: int; oos_end: int
    oos_metrics: dict; oos_equity: dict
    per_symbol_expectancy: dict[str, float | None]
    passed: bool
```
Your serialiser carries **every** field, including ones no view renders yet, for exactly the reason in that comment.

### TEST_SETUP_AND_ISOLATION
```python
# SOURCE: tests/test_backtest.py:27-34 — tier constants from config, never hardcoded
SETUP_TF = config.SIGNAL_PATTERN_TIMEFRAME
D_TRIG = storage.TIMEFRAME_MS[TRIGGER_TF]
START = 1_700_000_000_000

# SOURCE: tests/test_backtest.py:37-47 — isolation must not DEPEND on a
# fingerprint argument being right.
@pytest.fixture(autouse=True)
def _isolate_engine_caches():
    engine.clear_caches(); yield; engine.clear_caches()
```
Your autouse fixture additionally points `config.STRATEGY_DIR`, `config.STATE_DB_PATH` and `api.RUN_ROOT` at `tmp_path` and drains the job table — a leaked job thread poisons every test after it.

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/trading_bot/ui/__init__.py` | CREATE | Package marker; re-exports `serve` and `handle` so `cli.py` imports one name |
| `src/trading_bot/ui/api.py` | CREATE | Pure request→JSON over `framework`/`feedback`/`evolution`. **No `http`/`socket` import.** Routes, validation, serialisers, job table |
| `src/trading_bot/ui/server.py` | CREATE | Transport only: `ThreadingHTTPServer`, handler, static allowlist, SSE, loopback enforcement |
| `src/trading_bot/ui/static/index.html` | CREATE | One page, three lanes. No inline JS/CSS |
| `src/trading_bot/ui/static/app.css` | CREATE | Tokens ported from `build_performance_chart.py:454-535`; three-lane grid |
| `src/trading_bot/ui/static/app.js` | CREATE | Fetch + render, ParamSpec→control, SVG chart, SSE client. Vanilla, no bundler |
| `src/trading_bot/config.py` | UPDATE | Append Phase 7 block: `UI_HOST`, `UI_PORT` only (§7) |
| `src/trading_bot/cli.py` | UPDATE | `ui` subparser + `_ui_command` (§7 reserved names) |
| `.gitignore` | UPDATE | One line: `data/ui_runs/` — run logs are derived |
| `tests/test_ui_api.py` | CREATE | Handlers called directly, no socket (§8) |
| `tests/test_ui_roundtrip.py` | CREATE | Success signal: compose → serialize → deserialize → backtest → gate verdict |

## NOT Building

The PRD names "UI scope eats the project" as a Medium risk. This list is the containment. **MVP = compose / run / inspect.** Each item names the cheapest thing that satisfies the corresponding User Flow step.

**Composition** — no drag-and-drop, node canvas, bezier edges, auto-layout, zoom/pan or minimap. *Cheapest that satisfies "compose a strategy": four labelled stage groups (Detector / Confirmation / Policy / Filter), each a `<select>` plus `[+]`/`[×]`, in pipeline order; reordering is `[↑]`/`[↓]`, not dragging.* No graph shape beyond a linear pipeline — no branching, fan-in, conditionals or sub-graphs even if Phase 3 supports them; a non-linear graph loads **read-only** with a visible reason. No new `ParamSpec` capabilities (no units, help text, dependent params, or validation beyond `bounds`/`choices`) — a missing one is a **Phase 3 gap to report**, never a UI-local extension. No plug-in authoring in the browser: no code editor, no eval, no "new detector" wizard; adding a plug-in stays a file in `plugins/` (§12.4). No delete/rename from the UI — `mv`/`rm`.

**Runs** — no cancellation of Tier A (gate/backtest) jobs: `walk_forward_pooled` has no cooperative cancel point and adding one edits Phase 1's file. Tier B subprocesses get SIGTERM because that costs nothing. Documented, not hidden. No run queue, scheduling, cron or retries; Tier A concurrency is **1**, a second request gets `429`. No worker pool of our own — Phase 6 owns process parallelism. No run comparison, diffing, leaderboard sorting or A/B view; one run's results at a time, the generations table being the only multi-candidate view. No editing a run's parameters after launch.

**Rendering** — no charting library (not Chart.js, Plotly, d3, uPlot, or a vendored copy); hand-rolled inline SVG from the ported helpers. No candlestick / bar-geometry / pattern-overlay chart: that is `scripts/build_review_chart.py`'s job and per KNOWN-LIMITATIONS §8 it is **stale and will not run** (dispatches to retired `detect_patterns`, hardcodes `STEP_S = 900`) — do not port it, import it, or "fix it while we're here". No CDN, webfont, icon font, sprite sheet or remote image; `system-ui` plus text marks (`✓ ✗ ● ○`). No npm, node, package.json, bundler, minifier, transpiler, TypeScript, JSX, Sass, Tailwind or CSS-in-JS; no watch mode, no hot reload. No design system, component library, storybook or theme-toggle UI (`prefers-color-scheme` plus the `data-theme` attribute the report chart already honours). No responsive work beyond "does not break below 1000px" — single operator, one Mac; wide tables scroll inside their card (`overflow-x:auto`, as at `build_performance_chart.py:489-492`). No animation or loading skeletons beyond the existing 80ms tooltip fade.

**Platform** — no auth, users, sessions, cookies, CSRF tokens, TLS or rate limiting; loopback bind is the boundary (Task 7 states exactly what the POST-only/JSON-only/Origin discipline does and does not buy). No WebSockets — SSE is one-directional and sufficient, and its client half is a browser built-in. No non-loopback bind, `0.0.0.0`, tunnel, reverse proxy or LAN access; `_ui_command` refuses a non-loopback host with no override flag. No order placement, position management or "go live" button, ever (§10: alert-only is inherited, not reopened). **No "active strategy" flag**: User Flow step 4 says "promote a candidate to active", but no engine component reads such a flag today, so a UI-only marker is precisely the second source of truth this plan avoids — promotion in the MVP is *save the evolved graph under a name* (the Save path) plus display of Phase 5's `strategy_versions` provenance; a real active concept belongs to Phase 9's campaign. **Reported as a deliberate deferral, not an oversight.** User Flow step 5 (active strategy emits signals, executed manually) stays `cli.py signal`; the UI shows the command as copyable text. No `pyproject.toml` change — verified editable install (`__editable__.trading_bot-0.1.0.pth` → `.../trading/src`), so `Path(__file__).parent / "static"` resolves; a wheel install would need `[tool.setuptools.package-data]` (Risks). No browser automation in tests — no Playwright, Selenium or headless Chrome; none is configured here, and §8 gives two pure-Python test files.

---

## Step-by-Step Tasks

### Task 1: Reserve `UI_HOST` / `UI_PORT` in config
- **ACTION**: Append a Phase 7 block at the end of `config.py`. **Two constants, nothing else** (§7).
- **IMPLEMENT**: `UI_HOST = "127.0.0.1"`, `UI_PORT = 8770`, under a comment block stating: stdlib http.server, zero new dependencies (§10 row 6); **loopback only and not sweepable**, because this machine holds the trading logic and an irreplaceable 117 MB price store and the server has no auth; `ui.server` refuses a non-loopback host and no override flag exists deliberately; 8770 misses the usual dev ports (3000/5000/8000/8080).
- **MIRROR**: `CONFIG_APPEND_BLOCK`.
- **GOTCHA**: everything else the UI tunes (SSE poll interval, heartbeat, run-dir name, max log bytes, Tier A concurrency, body cap) is a **module constant in `ui/api.py`/`ui/server.py`**, not config. Only two names are reserved; a third collides with another phase's prefix.
- **VALIDATE**: `python -c "from trading_bot import config; print(config.UI_HOST, config.UI_PORT)"`; `grep -c "^UI_" src/trading_bot/config.py` → `2`.

### Task 2: Package skeleton, run-dir layout, module constants
- **ACTION**: Create `ui/__init__.py` (docstring stating the api/server split; re-export `handle`, `serve`) and the constants block at the top of `api.py`.
- **IMPLEMENT**:
  ```python
  RUN_ROOT = Path("data/ui_runs")     # monkeypatched in tests
  # data/ui_runs/<run_id>/
  #   cmd.json     what was asked for (kind, strategy, symbols, span, campaign)
  #   status.json  {state: queued|running|done|failed|orphaned, tier, pid,
  #                 started_ms, finished_ms, exit_code}
  #   stdout.log   append-only; the SSE `log` stream tails it by byte offset
  #   result.json  Tier A only: the serialised result
  MAX_TIER_A_JOBS = 1                            # 2nd concurrent gate run -> 429
  MAX_BODY_BYTES = 1 << 20
  RUN_ID_RE = re.compile(r"^[0-9]{17}-[0-9a-f]{6}$")     # <ms>-<rand>, sortable
  NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")   # strategy + campaign ids
  logger = logging.getLogger("trading_bot")
  ```
- **MIRROR**: `DB_CONNECT_AND_LOCK` — the run dir exists so no UI state is memory-only.
- **GOTCHA**: log under `"trading_bot"` (as `storage.py:16`, `walkforward.py:43`), not a `"trading_bot.ui"` child the existing logging config knows nothing about.
- **VALIDATE**: `py_compile` both; `python -c "import trading_bot.ui.api as a; print(a.RUN_ROOT, a.MAX_TIER_A_JOBS)"`.

### Task 3: `api.py` — read-only surface (`/api/plugins`, `/api/data/coverage`, `/api/config`)
- **ACTION**: Three GET handlers driving lanes 1 and 2.
- **IMPLEMENT**: `get_plugins()` calls `registry.load_all()` (import errors are FATAL by §3 → 500), then groups `REGISTRY` by the `kind` prefix of each `"<kind>.<name>"` key into `{key, name, rationale, timeframes, tier, params}` where `params = {p: _paramspec_to_dict(ps)}`. The **rationale is included and displayed** — bruteforce/registry.py: *"an unmotivated strategy in a 10,000-combo sweep is just noise with a name"*; the UI is where that stops being a docstring.
  ```python
  def _paramspec_to_dict(ps) -> dict:
      """ParamSpec -> an HTML-control description. The SAME declaration Phase 6
      jitters within (§3), so a control can never offer a value the mutator
      considers illegal."""
      d = {"kind": ps.kind, "default": ps.default}
      if getattr(ps, "choices", None):
          d["choices"] = list(ps.choices)
      if getattr(ps, "bounds", None):
          d["min"], d["max"] = ps.bounds
          # `step` is NOT a ParamSpec field: derived, not invented as schema.
          d["step"] = 1 if ps.kind == "int" else max((d["max"] - d["min"]) / 100.0, 1e-6)
      return d
  ```
  `get_data_coverage()`: per (symbol, timeframe) bar count, first/last ts, gap count — read-only; **lane 1 reports, it does not fetch** (backfill stays a CLI job). `get_ui_config()`: `config.SYMBOLS`, every stored symbol, `TIMEFRAMES`, `BACKFILL_START`, `STRATEGY_DIR`, `GATE_CONDITIONS` **with thresholds**, `SCHEMA_VERSION`, `MAX_TIER_A_JOBS`.
- **MIRROR**: `bruteforce/registry.py:162-179` (`load_all`, fatal imports), `:140-141` (non-empty rationale).
- **GOTCHA #1**: send `GATE_CONDITIONS` **and** thresholds so `app.js` never hardcodes `1.0`/`0.95`/`0.25`. A threshold duplicated in JS is a second source of truth that goes stale silently.
- **GOTCHA #2**: `bounds` may be `None` for `bool`/`choice` — then `min`/`max`/`step` must be **absent**, not `null`, or `app.js` renders a number input for a checkbox.
- **GOTCHA #3**: verify `ParamSpec`'s real field names against `contracts.py` first. `_paramspec_to_dict` is the **only** place in the UI that knows them.
- **VALIDATE**: `pytest tests/test_ui_api.py::TestPlugins -v`; `curl -s localhost:8770/api/plugins | python -m json.tool | head -40`.

### Task 4: `api.py` — strategy read/write over the Phase 3 serialization
- **ACTION**: `list_strategies`, `get_strategy`, `post_strategy`, `post_validate`, plus two mapping functions.
- **IMPLEMENT**: `_stages_to_graph(name, stages)` builds `NodeSpec(kind=…, key=…, params=…)` per stage into a `StrategyGraph`. **`stages` is a linear projection that exists only in HTTP bodies, never on disk** — the file is always `graph.to_dict()`; this is the single mapping site. `_graph_to_stages(g) -> (stages, editable)` returns `editable=False` when the graph's shape is not representable in the linear composer, so the page renders it read-only; silently flattening would corrupt the graph on Save.
  ```python
  def post_strategy(name, body) -> ApiResponse:
      if not NAME_RE.match(name):
          return _err(400, f"invalid strategy name {name!r}; expected {NAME_RE.pattern}")
      target = (Path(config.STRATEGY_DIR) / f"{name}.strategy.json").resolve()
      if Path(config.STRATEGY_DIR).resolve() not in target.parents:
          return _err(400, "resolved path escapes STRATEGY_DIR")     # belt and braces
      try:
          payload = _stages_to_graph(name, body["stages"]).to_dict()
          # Round-trip BEFORE writing: a file the executor cannot read back is
          # worse than a rejection.
          if graph.from_dict(payload).to_dict() != payload:
              return _err(500, "graph failed to_dict/from_dict round-trip")
      except (KeyError, TypeError, errors.GraphError, errors.RegistryError) as exc:
          return _err(400, str(exc))
      tmp = target.with_suffix(".json.tmp")          # atomic: never a half file
      tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
      tmp.replace(target)
      return _ok({"name": name, "schema_version": payload.get("schema_version")})
  ```
- **MIRROR**: `cli.py:215-234` — convert dates to epoch-ms once, at the boundary.
- **GOTCHA #1**: **`SCHEMA_VERSION` is written by `to_dict()`, never by the UI.** If it is absent, that is a Phase 3 gap to report — do not inject the key.
- **GOTCHA #2**: reject unknown plug-in keys via `registry.get(key)` before writing; a graph naming a nonexistent plug-in must not reach disk.
- **GOTCHA #3**: verify `NodeSpec`'s real field names; adapt here only.
- **VALIDATE**: `pytest tests/test_ui_api.py::TestStrategyIO -v` — traversal, uppercase name, unknown key, failing `from_dict`, happy round-trip.

### Task 5: `api.py` — result serialisers (gate rows, equity-vs-benchmark, reviews, generations)
- **ACTION**: Turn engine dataclasses into page JSON. **No metric is recomputed here.**
- **IMPLEMENT**: `_serialize_wf_result(result, curves)` emits `gate` (`{passed, conditions:[{name, ok, measured, threshold}]}` built by iterating `walkforward.GATE_CONDITIONS` and **reading `result.gate[c]`**), `n_trials_used`, `oos_metrics`, `oos_equity`, `per_symbol_expectancy`, `benchmark` (`asdict`), `folds` (`asdict`), `final_params`, `final_max_hold_bars`, `oos_start`/`oos_end`, and `curves`. Every field is carried, including ones no view renders — `walkforward.py:104-107` records why a swept value absent from the result makes a verdict unreproducible. `_jsonsafe(obj)` maps NaN/Inf → `None` recursively: `json.dumps` emits bare `NaN`, which is invalid JSON, so `JSON.parse` throws and the panel goes blank **with no error** — pandas/statistics paths produce NaN freely, so this is handled at the serialiser, not hoped away.
  Curves reuse `equity.daily_returns` for the strategy leg and `result.benchmark.basket` for the null, compounded as at `build_performance_chart.py:68-74`/`:159-161`, but emitted as **number arrays** (`{"strategy":[…], "basket":[…], "first_day_ms":…, "oos_day0":…}`): the page is static and must redraw for a different run. *(Alternative considered: server-rendered SVG path strings, as the report script does. Rejected — chart geometry in two places, and the server becomes both API and renderer.)* Also `get_reviews(limit=50, strategy_version=None)` and `get_generations(campaign)` (gen index, population size, best fitness, gate conditions met, cumulative trials).
- **MIRROR**: `GATE_SCORECARD_ROW_SHAPE`, extended 5→7 rows and reading `result.gate` — a UI that re-derived PASS/FAIL could disagree with the engine, the one failure mode a verdict view must not have.
- **GOTCHA #1 (verify first)**: `get_reviews`/`get_generations` must call **published read accessors** on `feedback.records`/`evolution.population`, not hand-written SQL over another phase's tables. §6 names the tables but not their columns; UI SQL would couple two phases through an unpublished schema. **No accessor = a Phase 5/6 gap to report.** Interim fallback only if the phase must proceed: exactly **one** read-only `SELECT` function per table, marked `# COUPLING: raw SQL over a Phase N table, pending an accessor`, so there is a single site to delete.
- **GOTCHA #2**: `result.gate`, `.benchmark`, `.n_trials_used` are Phase 1 additions (§4). If `_evaluate_gate` still returns `bool`, this task is **blocked on Phase 1** — a 7-row view synthesised from one boolean would be theatre.
- **GOTCHA #3**: `_gate_measured` shows the *pair* for benchmark conditions ("+68.2% vs +29.4%"). A benchmark row without the benchmark number is the exact hiding KNOWN-LIMITATIONS §0 is about.
- **VALIDATE**: `pytest tests/test_ui_api.py::TestSerializers -v` — construct a `WalkForwardResult` with all 7 gate keys; assert 7 rows, `passed == all(...)`, NaN → `null`.

### Task 6: `api.py` — starting and observing runs (Tier A thread, Tier B subprocess)
- **ACTION**: `post_run`, `get_runs`, `get_run`, `post_run_stop`, `recover_runs`, the job registry, run-dir writers.
- **IMPLEMENT**: Two tiers, one run-dir contract.
  - **Tier A — in-process job THREAD**, kinds `backtest` | `gate`. Minutes, single-process, and it *is* the oracle path (`run_graph_backtest` / `walk_forward_pooled`), so in-process keeps one code path and one ledger. `MAX_TIER_A_JOBS = 1`. Not cancellable (see NOT Building).
  - **Tier B — SUBPROCESS**, kind `evolve`. Hours, and Phase 6 already owns a process-parallel runner — nesting it in a request thread would fork a pool from inside the server:
    `Popen([sys.executable, "-m", "trading_bot.cli", "evolve", "--graph", p, "--campaign", c, …], stdout=log_fh, stderr=STDOUT, start_new_session=True, shell=False)`. `start_new_session` detaches it from the server's process group, so Ctrl-C on the server does **not** kill a campaign six hours in. **The UI observes**: progress comes from the `generations`/`population_members` rows Phase 6 writes to `state.db`, plus the log tail. SIGTERM to the recorded pid stops it.
  - `_RUN_KINDS: dict[str, _RunKind]` is an **allowlist** mapping kind → tier, target (`fn` for A, argv prefix for B) and permitted arg names. `post_run` validates **before** anything starts: kind in `_RUN_KINDS`; `strategy` matches `NAME_RE` and its file exists; every symbol is in the **stored** symbol set (not a free string); `start`/`end` parse via `config.date_to_ms`; `campaign` matches `NAME_RE`; ints are ints under a stated ceiling. Anything else → `400` naming the field. **Never** interpolate a request value into a shell string — `Popen` takes a list, `shell=False`.
  - Response is `202 {run_id, tier, trials_before, trials_estimated}`: a gate/evolve run **spends degrees of freedom** (§4.2), so its ledger cost is echoed, and the page shows it *before* the click too.
  - `recover_runs()` runs once at server start: any `status.json` still `running` → Tier B pid probe (`os.kill(pid, 0)`; alive stays `running`, dead → `failed`), Tier A → `orphaned` (its thread died with the old process). Without it a restarted server shows phantom runs forever.
- **MIRROR**: `cli.py:199-209` — open conn, work, close, integer exit code; a Tier A job body is that shape with `exit_code` landing in `status.json`.
- **GOTCHA #1**: a job thread opens its **own** sqlite connection inside the thread and closes it in `finally` (`DB_CONNECT_AND_LOCK`).
- **GOTCHA #2**: wrap the job body in `try/except BaseException`, write `state="failed"` plus the traceback into `stdout.log`, re-raise nothing. An unhandled exception in a daemon thread vanishes silently and the page shows "running" forever.
- **GOTCHA #3**: `status.json` writes must be atomic (`tmp` + `replace`) or an SSE poll reads a truncated file and `JSONDecodeError`s on a healthy run.
- **GOTCHA #4**: Tier A `gate` runs increment the **persistent** ledger, so manual validation must use a scratch `--campaign ui-smoke` — otherwise smoke-testing quietly inflates a real campaign's DSR denominator. Say so in the run form's help text, not only here.
- **VALIDATE**: `pytest tests/test_ui_api.py::TestRuns -v` with `RUN_ROOT` at `tmp_path` and job fns stubbed: 400 per bad field, 429 on the 2nd Tier A run, `recover_runs` reclassification, argv is a list.

### Task 7: `api.py` — the dispatcher (`handle`) and request-level discipline
- **ACTION**: `ApiResponse(status, payload: dict, headers)`, a `ROUTES` tuple of `(method, path regex, handler)`, and `handle()`.
- **IMPLEMENT**: routes are `GET /api/config`, `/api/plugins`, `/api/data/coverage`, `/api/strategies`, `/api/strategies/<name>`, `/api/runs`, `/api/runs/<rid>`, `/api/reviews`, `/api/generations`; `POST /api/strategies/<name>`, `/api/validate`, `/api/runs`, `/api/runs/<rid>/stop`.
  ```python
  def handle(method, path, query, body, headers=None) -> ApiResponse:
      """The single entry point. No socket, no http import -- tests call this.
      1. path matches a route     -> else 404
      2. method matches that path -> else 405 with Allow:
      3. POST: Origin, if present, must be http://127.0.0.1:<port> or
         http://localhost:<port>  -> else 403; Content-Type must be
         application/json         -> else 415.
         THIS IS NOT A SECURITY BOUNDARY. It stops a random web page the
         operator has open from POSTing here (HTML forms cannot send
         application/json; a cross-origin fetch carries a rejected Origin). It
         stops NOTHING local: any process or user on this Mac can drive every
         endpoint. The boundary is the loopback bind plus a single-user machine.
      4. dispatch; ValueError/KeyError from a handler -> 400 with its message
         (they are argument errors); anything else -> 500, logged, generic body.
      """
  ```
- **MIRROR**: `cli.py:405-409` — a bad-argument `ValueError` becomes a clean message and a nonzero code, never a traceback. `handle` is the HTTP form of that convention.
- **GOTCHA**: the Origin check needs the bound port, but `api.py` must not import `server.py` (circular, and it would drag a socket dependency into the testable half). `api.set_origin_port(port)` is called once by `serve()`, defaulting to `config.UI_PORT`, so a test that never calls it still asserts the 403.
- **VALIDATE**: `pytest tests/test_ui_api.py::TestDispatch -v` — 404, 405+`Allow`, 403 foreign Origin, 415, 400 from a raising handler.

### Task 8: `server.py` — transport, static allowlist, loopback enforcement
- **ACTION**: `serve()`, the handler subclass, the static allowlist.
- **IMPLEMENT**:
  ```python
  STATIC_DIR = Path(__file__).resolve().parent / "static"
  # A FIXED allowlist, not a directory walk: SimpleHTTPRequestHandler maps URLs
  # onto the filesystem and is a traversal surface; three entries cannot be
  # traversed. New file -> new entry, deliberately.
  STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
            "/app.css": ("app.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8")}
  SSE_POLL_SECONDS, SSE_HEARTBEAT_SECONDS = 1.0, 15.0
  SSE_MAX_CHUNK_BYTES, MAX_SSE_STREAMS = 64 * 1024, 8

  class _Handler(http.server.BaseHTTPRequestHandler):
      protocol_version = "HTTP/1.1"   # MANDATORY: 1.0 makes SSE buffer to EOF
      server_version = "trading_bot.ui"
      def log_message(self, fmt, *a): logger.info("ui %s", fmt % a)

  class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
      daemon_threads = True           # an open SSE stream must not block Ctrl-C
      allow_reuse_address = True

  def serve(host=None, port=None) -> int:
      host, port = host or config.UI_HOST, port or config.UI_PORT
      if not ipaddress.ip_address(socket.gethostbyname(host)).is_loopback:
          logger.error("refusing to bind %s: loopback only. This server has NO "
                       "auth and this machine holds the trading logic and the "
                       "OHLCV store.", host)
          return 2
      api.set_origin_port(port); api.recover_runs()
  ```
  Response discipline: always `Content-Length` (HTTP/1.1 keep-alive) except on SSE; `Cache-Control: no-store` on `/api/*`; read the POST body honouring `Content-Length` up to `api.MAX_BODY_BYTES` (no cap is an easy local OOM); `Connection: close` on any 4xx/5xx so a half-read body cannot desync the connection.
- **MIRROR**: `cli.py:396-404` for the documented return-code contract; `build_performance_chart.py:687-725` for "one entry point returning an int".
- **GOTCHA #1**: `protocol_version = "HTTP/1.1"` is mandatory — under the 1.0 default SSE cannot stream incrementally, the browser buffers to connection close, and the progress view appears frozen for the entire run.
- **GOTCHA #2**: on `OSError: [Errno 48]` print "port 8770 in use — another `cli ui` is probably running" and return 2. A raw traceback here is a daily annoyance.
- **GOTCHA #3**: read static files **from disk per request**, never cached at import. Editing `app.js` and reloading must show the change; a cache would waste an afternoon before anyone suspected it.
- **VALIDATE**: `py_compile`; `curl -sI localhost:8770/`; `curl -s localhost:8770/../config.py` → 404; `cli ui --host 0.0.0.0` → exit 2 with the refusal.

### Task 9: `server.py` — the SSE endpoint
- **ACTION**: `GET /api/runs/<rid>/events` — the only streaming route, in `server.py` so `api.py` stays socket-free. Headers: `Content-Type: text/event-stream`, `Cache-Control: no-store`, `X-Accel-Buffering: no` (harmless locally, correct if ever proxied).
- **IMPLEMENT**: frames are `retry: 2000` once; then `id: <byte offset>` + `event: log` + one `data:` line per log line; `event: progress` with a JSON payload (Tier B only); `: heartbeat`; and a terminal `event: done` carrying `{state, exit_code, has_result}`. Loop every `SSE_POLL_SECONDS`: (a) read `stdout.log` from the cursor, emit ≤ `SSE_MAX_CHUNK_BYTES`, advance; (b) Tier B — re-read the Phase 6 generation rows, emit `progress` **only on change**; (c) re-read `status.json`, and if terminal emit `done` and return; (d) heartbeat if nothing was sent.
  **Reconnect**: `EventSource` reconnects after `retry` ms and resends `Last-Event-ID`; the handler prefers that, falls back to `?offset=`, defaults to 0. Because the cursor is a **byte offset into a file on disk**, a reload, a network blip and a browser restart are one case, and no log line is lost or duplicated.
  **Browser closes mid-run**: writing to the dead socket raises `BrokenPipeError`/`ConnectionResetError` — catch, log at DEBUG, return. The **run is untouched**: Tier A's thread and Tier B's detached subprocess hold no reference to the request. Nothing is cancelled by a closed tab, which is the intended semantics for an overnight campaign.
- **MIRROR**: §6 (`generations`/`population_members` are Phase 6's) and Task 5 GOTCHA #1 — progress goes through the accessor, not raw SQL.
- **GOTCHA #1**: a `data:` payload may not contain a raw newline. Split multi-line text into one `data:` line each (the spec joins with `\n`) and strip `\r`; a stray newline silently truncates the event.
- **GOTCHA #2**: `self.wfile.flush()` after every event, or the OS buffer holds progress for a run that is visibly alive in the terminal but frozen in the browser.
- **GOTCHA #3**: each stream occupies a thread for the run's duration — `daemon_threads` keeps Ctrl-C working, `MAX_SSE_STREAMS` guards against a reload loop accumulating threads.
- **GOTCHA #4**: send **no** `Content-Length` here, and keep the generic keep-alive path from adding one.
- **VALIDATE**: no socket test (§8 gives two socket-free files). Framing is unit-tested via a pure `api.sse_frames(offset, log_text, status, progress) -> list[str]` — formatting in `api.py`, socket writing in `server.py`, which is the point of the split. Manual: `curl -N localhost:8770/api/runs/<rid>/events` during a real run.

### Task 10: `static/index.html` — three lanes, one page
- **ACTION**: The static skeleton. **No inline JS or CSS.** Every dynamic region is an empty container `app.js` fills: `#lane-data`, `#lane-strategy` (`#stages`, `#compose-actions`), `#lane-feedback` (`#run-form`, `#run-list`), then `#gate`, `#equity`, `#reviews`, `#generations`, `#tip`.
- **IMPLEMENT**: `<!doctype html>` with the note carried from `build_performance_chart.py:446-448` — *the doctype is not optional: without it browsers render in Quirks Mode, which changes box sizing and line layout out from under the CSS*. `<meta charset>`, `<meta viewport>`, `<title>trading_bot builder</title>`, `<link rel="stylesheet" href="/app.css">`, `<body class="viz-root">`, `<script src="/app.js">` at the end. A header banner states loopback-only / no-auth plus the strategy dir and `state.db` path. Each `<section>` gets `aria-labelledby` on an `<h2>`; the lanes are labelled with the pivot guide's own three words (**Data gathering / Strategy / Feedback loop**), so the UI teaches the framework's structure — the PRD's "forcing function".
- **MIRROR**: `build_performance_chart.py:536-641` for section/card/table structure and `role="img"` + `aria-label` on every SVG; `:530-533`, `:641` for the `#tip` element.
- **GOTCHA**: `aria-live="polite"` on `#gate` so a verdict announces itself; **not** `assertive`, which would interrupt on every progress tick.
- **VALIDATE**: manual — three lanes render with JS disabled (empty containers, no error); view-source has no `http` URL anywhere.

### Task 11: `static/app.css` — tokens and the lane grid
- **ACTION**: Copy the `:root`/`.viz-root`/`@media (prefers-color-scheme: dark)`/`:root[data-theme=…]` blocks **verbatim** from `build_performance_chart.py:454-535`, then add the grid and control styles.
- **IMPLEMENT**:
  ```css
  #lanes { display:grid; gap:12px; align-items:start;
           grid-template-columns:minmax(220px,1fr) minmax(320px,1.4fr) minmax(260px,1fr); }
  @media (max-width:1000px) { #lanes { grid-template-columns:1fr; } }
  .chip.good { color:var(--good); } .chip.crit { color:var(--crit); }
  .stage { border-left:3px solid var(--axis); padding-left:10px; }
  .stage.missing { border-left-color:var(--crit); }
  .param { display:grid; grid-template-columns:1fr auto; gap:6px; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  ```
- **MIRROR**: `build_performance_chart.py:478-535` — `box-sizing:border-box` reset, `system-ui` stack, tabular numerals on numeric cells, `overflow-x:auto` on `.card`.
- **GOTCHA**: keep `--s1..--s4` identical to the report chart's, or the same curve reads as two different things across the live UI and the committed report.
- **VALIDATE**: manual — toggle macOS light/dark; at 1000px the lanes stack rather than scroll the body horizontally.

### Task 12: `static/app.js` — controls, chart, SSE client
- **ACTION**: The whole client, one file, no modules, no framework, no build step. Six sections:
  1. **`api()`** — `fetch` wrapper: JSON in/out, non-2xx throws with the server's `error` string, every failure surfaced in a visible `#error` banner. No silent `catch {}`.
  2. **`renderLanes()`** — one `Promise.all` over `/api/config`, `/plugins`, `/data/coverage`, `/strategies`, `/runs`, then paint lanes 1 and 3.
  3. **`controlFor(name, spec)`** — the ParamSpec renderer, the heart of lane 2: `int` → `<input type=number step=1 min max>`; `float` → number + `<input type=range>` sharing one value; `bool` → checkbox; `choice` → `<select>` from `spec.choices`. Label is the param name **verbatim** (the name is the contract, not a UI string to prettify); `title` is the plug-in rationale. **No control exists that ParamSpec did not declare, and none accepts a value outside `bounds`.**
  4. **`collectStages()` / `loadStages(stages, editable)`** — DOM ↔ the `stages` wire format; `editable === false` renders read-only with the reason and disables Save.
  5. **`drawEquity(curves)` / `drawGate(gate)`** — inline SVG via `document.createElementNS`, using `svgPath`/`areaPath`/`niceTicks`/`spreadLabels` ported from `build_performance_chart.py:85-137` (keep the log10-magnitude comment). Two series minimum — **strategy and buy-and-hold basket** — plus the shaded OOS band (`:265-275`) and a `[table view]` `<details>` with the same numbers (`:619-624`). `drawGate` paints the 7 rows, each mark + word.
  6. **`watchRun(rid)`** — `new EventSource('/api/runs/'+rid+'/events')`; `log` appends to a **bounded** (last 500 lines) `<pre>`; `progress` updates the generations table; `done` closes the source and fetches the final result. On load, every run whose status is `running` is re-watched automatically — the whole reload story, needing no client-side persistence because the run list comes from disk.
- **MIRROR**: `build_performance_chart.py:642-682` — the `wire()` tooltip/crosshair function, its `viewBox`-to-client coordinate maths and its edge-flip logic; reuse nearly verbatim. Plus `ESCAPE_EVERY_INTERPOLATED_STRING`.
- **IMPORTS**: none. No `import`/`export`, no `<script type=module>` — a plain script works over `http://` with zero ceremony.
- **GOTCHA #1**: `textContent`/`createElement` for **every** server string. A rationale containing `<` would otherwise break the page (`:383` escapes for exactly this).
- **GOTCHA #2**: `EventSource` reconnects on its own; adding a manual retry loop produces two streams per run after a server restart.
- **GOTCHA #3**: an unbounded log `<pre>` eats gigabytes over a 6-hour campaign — cap at 500 lines and note "(older lines in `data/ui_runs/<id>/stdout.log`)".
- **GOTCHA #4**: `range` inputs yield strings — `parseFloat` before POSTing, or the graph stores `"1.5"` and `from_dict` may accept it and quietly produce a string-typed param.
- **VALIDATE**: `pytest tests/test_ui_roundtrip.py -v` covers the data path; the DOM path is the manual checklist.

### Task 13: `cli.py` — the `ui` subcommand
- **ACTION**: Register `ui` in `main()` in phase order (after Phase 6's `evolve`, before Phase 8's `detector-report`) and add `_ui_command`.
- **IMPLEMENT**: `subparsers.add_parser("ui", help="Serve the local builder UI (loopback only, no auth)")` with `--host` (default `None`, help naming `config.UI_HOST` and "loopback only") and `--port` (`type=int`, default `None`); dispatch `elif args.command == "ui": sys.exit(_ui_command(host=args.host, port=args.port))`.
  ```python
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
  ```
- **MIRROR**: `CLI_SUBCOMMAND_AND_HANDLER`.
- **GOTCHA**: `ui` takes no `--db`. The global flag exists (`cli.py:31-35`) but `api.py` resolves paths from `config` per request; threading one `--db` through would create a connection the UI must not hold. A non-default DB is an `api.py`-level override, not a flag here.
- **VALIDATE**: `cli ui --help`; `cli ui --host 10.0.0.5; echo $?` → `2`; `pytest tests/test_ui_api.py::TestCli -v`.

### Task 14: `tests/test_ui_api.py` — handlers called directly, no socket
- **ACTION**: The unit suite. **Never binds a port**; every test calls `api.*`. Module docstring: *"Calls api.py's handlers DIRECTLY. No socket, no server, no browser — the api/server split (§8) exists so this file can exist."*
- **IMPLEMENT**: an autouse `_isolate_ui(tmp_path, monkeypatch)` fixture patching `config.STRATEGY_DIR`, `config.STATE_DB_PATH` and `api.RUN_ROOT` into `tmp_path` and calling `api.reset_jobs()` before and after (asserting the table is empty on exit rather than trusting it — a leaked job thread poisons every later test; and a test writing into the real `data/strategies/` will one day overwrite a real strategy). Classes:
  - `TestPlugins` — every kind present; every rationale non-empty; bounds for numeric; choices for choice; **bool has no `min`/`max`/`step` keys**.
  - `TestStrategyIO` — traversal name → 400; uppercase → 400; unknown plug-in key → 400; `SCHEMA_VERSION` comes from `to_dict`; save-then-get identical; non-linear graph → `editable: false`.
  - `TestSerializers` — 7 rows == `GATE_CONDITIONS`; `passed == all()`; benchmark `measured` contains both sides; NaN → `null`.
  - `TestRuns` — unknown kind / unstored symbol / bad date → 400; 2nd Tier A → 429; trial cost reported; `recover_runs` marks Tier A `orphaned`; argv is a list.
  - `TestDispatch` — 404; 405 + `Allow`; 403 foreign Origin; 415.
  - `TestSseFraming` — multiline log → one `data:` line each; `id` == byte offset; `done` carries `exit_code`.
  - `TestCli` — `_ui_command` refuses a non-loopback host → 2.
- **MIRROR**: `TEST_SETUP_AND_ISOLATION` — autouse fixture with the reason spelled out; `GATE_CONDITIONS` from `walkforward` and symbols from `config`, never hardcoded.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_ui_api.py -v` — all pass, under ~2s, no port, no I/O outside `tmp_path`.

### Task 15: `tests/test_ui_roundtrip.py` — the success-signal test
- **ACTION**: One class proving the phase's success signal: *a strategy composed entirely in the UI round-trips through serialization, backtests, and displays its gate verdict.* Docstring: *"The strategy is composed the way the BROWSER composes one: read /api/plugins, take each ParamSpec's default, build `stages`, POST it. Nothing is hand-written as graph JSON, so a Phase 3 format change fails this test loudly instead of being silently duplicated in the UI."*
- **IMPLEMENT**: `TestComposeRoundTrip` with `test_compose_from_paramspec_defaults_and_save`, `test_saved_file_deserializes_through_from_dict`, `test_backtest_through_run_graph_backtest_returns_trades`, `test_curves_include_the_buy_and_hold_basket`, `test_engine_core_untouched` (§12.4 — the UI added no `BacktestParams` field and no `registry.KINDS` entry), and:
  ```python
      def test_gate_verdict_has_all_seven_conditions_rendered(self, conn):
          """walk_forward_pooled(strategy=graph) -> _serialize_wf_result ->
          exactly len(GATE_CONDITIONS) rows, each with name/ok/measured/
          threshold, plus a benchmark block.

          Asserts SHAPE AND REACHABILITY, never PASS. On a synthetic fixture the
          gate will fail, and a test demanding a pass would be a test applying
          pressure to the gate -- precisely what §4 forbids.
          """
  ```
  Fixture: synthetic in-memory SQLite seeded bar-by-bar for 2+ symbols at `REGIME_TF`/`SETUP_TF`/`TRIGGER_TF`, `START = 1_700_000_000_000`, spans just long enough for one fold plus the OOS window — same construction as `tests/test_backtest.py`, tier constants from `config` (`:27-34`).
- **MIRROR**: `tests/test_backtest.py:50-57` (`make_trade`); Phase 3's `test_framework_parity.py` graph fixture — reuse its helper if importable rather than writing a second one.
- **GOTCHA #1**: pass small explicit `min_trades`/`train_days`/`test_days`/`oos_days`, as `TestWalkForwardPooled` does; the production 180/60/90 defaults need years of bars and minutes of runtime.
- **GOTCHA #2**: `config.STATE_DB_PATH` must point at `tmp_path` — a gate run in a test must not write to the real trial ledger. A suite that inflates a campaign's DSR denominator is a correctness bug, not test noise.
- **GOTCHA #3**: this is the coupling detector for Phases 3/5/6. When it fails after a sibling change, the fix is usually in `api.py`'s two mapping functions, not in the test.
- **VALIDATE**: `.venv/bin/python -m pytest tests/test_ui_roundtrip.py -v`.

### Task 16: `.gitignore`, full-suite sweep, phase report
- **ACTION**: Append `data/ui_runs/` to `.gitignore`, then run the measured validation pass.
- **IMPLEMENT**: the report quotes **measured** figures only (§12.5), each with the command that produced it: test count before/after; plug-ins exposed by `/api/plugins` by kind with a count of empty rationales (must be 0); byte size of the three static files (`wc -c`) as the concrete "no build step" evidence; the zero-external-reference grep; the `pyproject.toml` diff proving zero new dependencies; and degrees of freedom — **the UI itself consumes none**, but record how many `trial_ledger` rows manual validation added and **under which scratch campaign** (`ui-smoke`), so the count is attributable and excluded from any real campaign's DSR.
- **MIRROR**: the repo's own committed discipline — `git log`: *"Use measured rather than derived figures in the benchmark table."*
- **GOTCHA**: `.gitignore` already has `data/*.db` (Phase 1's line covers `state.db` and its WAL sidecars), but `data/ui_runs/` is a **directory** of logs and JSON that pattern does not match. Verify with `git status --porcelain data/` after a run.
- **VALIDATE**: the full block below.

---

## Testing Strategy

### Unit Tests

| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| Every kind present | `get_plugins()` | all 7 `registry.KINDS` keys, even when empty | — |
| Rationale non-empty | every `PluginSpec` | 0 empty rationales | Registry invariant, surfaced in UI |
| ParamSpec → control | `int`/`float` with bounds | `min`, `max`, `step` present | — |
| ParamSpec → control | `bool` | **no** `min`/`max`/`step` keys | Wrong control otherwise |
| ParamSpec → control | `choice` | `choices`, no bounds | — |
| Name validation | `"../../etc/passwd"` | 400, path never touched | **Traversal** |
| Name validation | `"MyStrategy"` | 400 (lowercase-hyphen only) | §3 naming |
| Unknown plug-in key | stage `detector.nope` | 400, no file written | — |
| Save round-trip | stages from ParamSpec defaults | file exists, `SCHEMA_VERSION` set, `from_dict` OK | Success signal |
| Non-linear graph | branching graph on disk | `editable: false` + reason, Save disabled | **Never flatten-and-save** |
| Gate rows | fabricated `WalkForwardResult` | 7 rows == `GATE_CONDITIONS`, `passed == all()` | §4 |
| Benchmark rows | ditto | `measured` shows both sides ("… vs …") | KNOWN-LIMITATIONS §0 |
| NaN serialisation | `oos_equity` with NaN | `null`, valid JSON | **Silent blank panel** |
| Run kind allowlist | `{"kind":"rm -rf"}` | 400 | **Injection** |
| Symbol validation | `"BTC; drop"` | 400 (not in stored set) | **Injection** |
| Tier A concurrency | 2nd gate run | 429 | — |
| Trial cost echoed | gate run | `trials_before` + `trials_estimated` | §4.2 |
| `recover_runs` | stale `running` status.json | Tier A → `orphaned`, dead Tier B → `failed` | **Server restart** |
| Popen argv | evolve run | argv is a `list`, `shell=False` | **Injection** |
| Dispatch | unknown path / GET on POST route | 404 / 405 with `Allow` | — |
| Dispatch | foreign `Origin` / `text/plain` POST | 403 / 415 | Same-origin discipline |
| SSE framing | log text with newlines | one `data:` line per line | **Spec violation truncates** |
| SSE framing | terminal status | `done` with `exit_code` | — |
| Loopback | `_ui_command(host="10.0.0.5")` | returns 2, nothing bound | **Exposure** |
| Round-trip + gate | synthetic 2-symbol store | 7 conditions rendered; shape asserted, PASS not required | **Success signal** |
| Curves | ditto | both `strategy` and `basket` present | KNOWN-LIMITATIONS §0 |
| Engine untouched | after this phase | `BacktestParams` fields and `registry.KINDS` unchanged | §12.4 |

### Edge Cases Checklist
- [x] Empty input — empty registry, zero stored symbols, absent strategy dir, zero runs, zero reviews: each renders a stated reason, never a traceback or a blank panel
- [x] Maximum size input — POST body capped by `MAX_BODY_BYTES`; log `<pre>` capped at 500 lines; SSE chunk capped at 64 KiB
- [x] Invalid types — every field validated at the HTTP boundary; floats `parseFloat`'d client-side and type-checked server-side
- [x] Concurrent access — thread per request, **no shared sqlite connection**, Tier A limited to 1, `status.json` written atomically
- [x] Network failure — SSE drop is `EventSource`'s own reconnect with a byte-offset cursor; the run is unaffected
- [x] Permission denied — unwritable `STRATEGY_DIR`/`RUN_ROOT` → 500 naming the path, not a traceback in the browser
- [x] Path traversal — names regex-gated *and* path-containment checked; statics are a 3-entry allowlist, not a directory walk
- [x] Process death — restart reclassifies stale runs; a detached Tier B campaign survives; a closed tab cancels nothing
- [x] Stale schema — a graph naming a missing plug-in renders red with Save disabled
- [ ] Browser automation — **not tested**, by decision: nothing of the kind is configured here (§8). The DOM path is a manual checklist

---

## Validation Commands

### Static Analysis
```bash
.venv/bin/python -m py_compile \
  src/trading_bot/ui/__init__.py src/trading_bot/ui/api.py src/trading_bot/ui/server.py \
  src/trading_bot/cli.py src/trading_bot/config.py
```
EXPECT: clean. **No linter and no type checker are configured** (KNOWN-LIMITATIONS §8, §12.2) — `py_compile` plus `pytest` *is* the validation surface. No frontend toolchain exists, so the three static files have no automated check beyond the greps below plus the manual checklist.

### Unit Tests and Full Suite
```bash
.venv/bin/python -m pytest tests/test_ui_api.py tests/test_ui_roundtrip.py -v
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest --collect-only -q | tail -1
```
EXPECT: new tests pass with no port bound, no browser, no network; the **286 pre-existing tests still green** (baseline measured 2026-07-27). §8: a phase that breaks the 286 is not done. Report the new total.

### Zero-dependency / zero-network proof
```bash
diff <(git show HEAD:pyproject.toml) pyproject.toml            # EXPECT: no output
grep -rnE 'https?://|cdn\.|unpkg|jsdelivr|@import url|<script src="http' \
  src/trading_bot/ui/static/                                   # EXPECT: no matches
grep -rn "import \|require(" src/trading_bot/ui/static/app.js  # EXPECT: no matches
grep -rn "innerHTML" src/trading_bot/ui/static/app.js          # EXPECT: only non-interpolated literals
grep -rn "SimpleHTTPRequestHandler\|shell=True\|eval(" src/trading_bot/ui/   # EXPECT: no matches
wc -c src/trading_bot/ui/static/*                              # measured page weight
```

### Loopback and route discipline
```bash
.venv/bin/python -m trading_bot.cli ui --host 0.0.0.0; echo "exit=$?"   # EXPECT: refusal, exit=2
.venv/bin/python -m trading_bot.cli ui &        # then, in another shell:
curl -s -o /dev/null -w '%{http_code}\n' localhost:8770/                        # 200
curl -s -o /dev/null -w '%{http_code}\n' localhost:8770/../../config.py         # 404
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8770/api/runs \
     -H 'Origin: https://evil.example' -H 'Content-Type: application/json' -d '{}'   # 403
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8770/api/runs \
     -H 'Content-Type: text/plain' -d 'x'                                        # 415
curl -s localhost:8770/api/plugins | .venv/bin/python -c \
  "import json,sys; d=json.load(sys.stdin); print(d['n'],'plugins'); \
   print('empty rationales:', sum(1 for v in d['plugins'].values() \
                                  for p in v if not p['rationale'].strip()))"
```
EXPECT: `200 / 404 / 403 / 415`, and **0 empty rationales**.

### Database Validation
```bash
sqlite3 data/state.db ".tables"
sqlite3 data/state.db "select count(*) from trial_ledger where campaign='ui-smoke';"
git status --porcelain data/       # EXPECT: no output
```
EXPECT: Phase 1/5/6 tables exist; the UI created **no table of its own** (its run state is the filesystem); nothing under `data/` is stageable.

### Manual Validation
```bash
.venv/bin/python -m trading_bot.cli ui        # open http://127.0.0.1:8770/
```
- [ ] Three lanes render, labelled **Data / Strategy / Feedback**; lane 2 lists every registered plug-in per kind with its rationale on hover
- [ ] Every control's min/max matches its `ParamSpec.bounds`; no control exists that ParamSpec did not declare
- [ ] Compose → **Save** → reload → **Load** → identical stages; the file passes `cli graph-validate` and carries `SCHEMA_VERSION`
- [ ] **Run gate** (`--campaign ui-smoke`) returns immediately; the run shows `running`; progress streams
- [ ] **Reload mid-run** → stream resumes with no duplicated or missing lines
- [ ] **Close the tab** mid-run, reopen: still running (Tier B) or completed (Tier A) — nothing cancelled
- [ ] Gate panel shows **7 rows**, including `beats_benchmark_return`/`beats_benchmark_sharpe` with *both* numbers
- [ ] Equity chart shows strategy **and** buy-and-hold basket, OOS band shaded, plus a table view
- [ ] `n_trials_used` and the campaign's cumulative ledger count are visible
- [ ] Light/dark both legible; every verdict has a mark **and** a word
- [ ] Ctrl-C with an SSE stream open exits immediately (daemon threads); on restart there are no phantom `running` runs
- [ ] DevTools Network: **zero** requests to any host but `127.0.0.1:8770`

---

## Acceptance Criteria
- [ ] All 16 tasks complete; every validation command passes
- [ ] **Zero new dependencies** — `pyproject.toml` byte-identical to HEAD
- [ ] **Zero network references** in the static files; the page works with Wi-Fi off
- [ ] Loopback bind only; a non-loopback `UI_HOST` is refused with exit code 2
- [ ] `api.py` imports nothing from `http`/`socket`; `test_ui_api.py` opens no port
- [ ] A strategy composed entirely in the UI round-trips: save → `from_dict` → `run_graph_backtest` → `walk_forward_pooled` → **gate verdict rendered per condition** (the success signal)
- [ ] Strategy files written **only** via `graph.to_dict()`; no UI-authored JSON shape on disk
- [ ] Controls generated **only** from `ParamSpec`; a missing field is reported as a Phase 3 gap, not added locally
- [ ] Gate view shows all 7 `GATE_CONDITIONS`, reading `result.gate`, never recomputing
- [ ] Buy-and-hold basket rendered alongside strategy equity
- [ ] An evolution campaign runs as a detached subprocess; reload, closed tab or server restart never kills it
- [ ] Trial-ledger cost shown before launch and after completion
- [ ] `config.py` gained exactly two constants; `cli.py` gained exactly one subcommand
- [ ] **286 pre-existing tests still pass**; no file owned by another phase was modified (§2)

## Completion Checklist
- [ ] Follows discovered patterns (frozen dataclasses, `_<name>_command() -> int`, None-safe formatting, `logging.getLogger("trading_bot")`)
- [ ] Error handling matches codebase style: argument errors → clean message + non-2xx, never a traceback to the browser; no bare `except: pass`
- [ ] Every path opens and closes its own sqlite connection; nothing shared across threads
- [ ] Tests mirror `tests/test_backtest.py` — class grouping, autouse isolation, config-derived constants
- [ ] No hardcoded gate thresholds, symbols, timeframes or paths in `app.js` — all via `/api/config`
- [ ] Static assets read from disk per request (editable without restart)
- [ ] No unnecessary scope additions — the NOT Building list held
- [ ] Phase report quotes only **measured** figures, each with its command
- [ ] Degrees of freedom recorded: 0 consumed by the UI; validation ledger rows attributed to campaign `ui-smoke`
- [ ] Self-contained — the only flagged unknowns are the four sibling-phase field-name checks (Tasks 3, 4, 5), each localised to one function

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Scope creep** — the PRD names this the phase's own risk; a canvas or "one more view" doubles the phase | **High** | **High** | A long, specific NOT Building list paired with the cheapest satisfying alternative per User Flow step. MVP = compose / run / inspect; anything off the task list is a follow-up |
| `ParamSpec`/`NodeSpec`/`WalkForwardResult.gate` field names differ from the contract's sketch | Medium | Medium | Each is read in **exactly one** function (`_paramspec_to_dict`, `_stages_to_graph`, `_serialize_wf_result`); verify against the real files first. A missing *capability* (not a rename) is a Phase 3 gap to report |
| Phase 5/6 publish no read accessor, tempting raw SQL over their tables | Medium | Medium | Task 5 GOTCHA #1: prefer the accessor, report the gap, and if unavoidable confine read-only SQL to one commented function per table |
| A Tier A gate run in the server process makes the UI sluggish for minutes | Medium | Low | `MAX_TIER_A_JOBS = 1`; pandas/numpy release the GIL through most of a backtest; statics and other routes are on separate threads. If it bites, promote `gate` to Tier B — the run-dir contract is tier-agnostic |
| SSE progress depends on Phase 6 writing `generations` rows *during* a campaign, not at the end | Medium | Medium | The log tail is the always-works fallback (same stdout the CLI prints). If Phase 6 batches its writes the table lags but nothing breaks — report it rather than papering over it |
| A no-auth local server driven by something else on the machine | Low | Medium | Loopback-only bind with no override, POST-only + JSON-only + Origin check, `Popen` argv allowlist, no shell. **Stated plainly as not a security boundary**: any local process can drive it; the machine is the boundary |
| Trial-ledger inflation from casual clicking — a UI makes spending DoF *easy* | Medium | **High** (it is the anti-overfitting spine) | Cost shown before and after; `campaign` is required so attribution is never implicit; manual validation pinned to `ui-smoke`; the phase report records the rows added |
| A non-editable install breaks `Path(__file__).parent / "static"` | Low | Medium | Verified editable today (`__editable__.trading_bot-0.1.0.pth` → `.../trading/src`). A wheel install would need `[tool.setuptools.package-data]` for `trading_bot.ui` → `static/*`; not done — no phase owns `pyproject.toml` and nothing needs it |
| Hand-rolled SVG is more code than importing a chart library | High (certain) | Low | Deliberate. `build_performance_chart.py` proves the approach and donates four tested helpers; `pandas-ta`'s disappearance (§1) is the standing lesson about the alternative's cost |
| The 7-condition view makes failure legible and the reaction is to loosen a threshold | Low | **High** | Thresholds are read-only in the UI, arriving from `/api/config`; no control edits them. Changing one stays a `config.py`/`walkforward.py` edit with a logged rationale |

## Notes

**The UI stack is decided, not open.** PRD Open Question #6 is **resolved by contract §10 row 6** and is not reopened: stdlib `ThreadingHTTPServer` plus one vanilla HTML/CSS/JS page under `ui/static/`, zero new dependencies, SSE for progress. Recorded as considered-and-rejected:

| Considered | Rejected because |
|---|---|
| **FastAPI + uvicorn** | Two new dependencies (plus starlette, pydantic, click, h11…) for one local user. `pandas-ta` vanishing from PyPI is the live lesson in dependency risk (§1) |
| **React / Vite** | An npm toolchain, a `node_modules` and a build step for a **single page**. Nothing here has a frontend build, and the PRD's risk table says "UI scope eats the project" |
| **Streamlit** | Fights the interaction: its rerun-the-script model is awkward for composing a graph and for observing an hours-long subprocess, and it is another heavy dependency |
| **Static HTML reports only** (v0.2.0 status quo) | Already exists as `build_performance_chart.py` and is kept for committed reports — but it cannot compose a strategy or start a run, which is this phase's point |

**Where the PRD and the contract disagree, the contract wins** — two instances, both noted:
1. The PRD calls the UI stack an open question (Open Questions #6; Technical Approach "Stack choice is an open question resolved in the UI phase"); the contract resolves it. Implemented as resolved.
2. The PRD's Success Metrics table says the gate has **5** conditions; §4 extends it to **7** (`beats_benchmark_return`, `beats_benchmark_sharpe`, delivered by Phase 1). The UI renders **7**. This is the most important disagreement to get right: a 5-row scorecard would omit exactly the buy-and-hold comparison KNOWN-LIMITATIONS §0 exists to enforce.

**Also deliberately deferred, and reported rather than quietly dropped**: User Flow step 4's "promote a candidate to *active*" and step 5's signal emission. No engine component reads an active-strategy flag today, so a UI-only marker would be the second source of truth this plan is built to avoid. Promotion in the MVP is *save the graph under a name* plus display of Phase 5's `strategy_versions` provenance; a real active concept belongs with Phase 9's campaign. Signal emission stays `cli.py signal`.

**The single load-bearing idea**, worth restating because every design choice follows from it: **the UI is a view and a launcher, never a store.** Strategies live in Phase 3's serialization. Verdicts live in Phase 1's gate. Progress lives in Phase 6's `state.db` tables. Trials live in Phase 1's ledger. Reviews live in Phase 5's records. The UI's *only* private state is a run-bookkeeping directory under `data/ui_runs/` holding a pid, an argv, a status and a log — deliberately not a database table, and containing no result any other component would need to trust. **Delete `data/ui_runs/` and nothing of value is lost.** That is the test for whether a proposed UI feature belongs in this phase.

**Phase 8 runs in parallel and is free acceptance evidence.** Every detector Phase 8 registers appears in lane 2's dropdown with its rationale and ParamSpec controls, with **zero UI edits**. If a Phase 8 detector needs a UI change to be composable, §12.4 ("zero engine-core edits to add a plug-in") has been violated somewhere — note it, do not patch around it in `app.js`.

**This phase gates Phase 9**, which drives a full campaign and a never-touched holdout and needs a way to launch, observe and read out a campaign that is not a terminal the operator must keep open for six hours. That is what Task 6's Tier B and Task 9's SSE provide.
