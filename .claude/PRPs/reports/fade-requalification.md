# Fade Re-qualification — Decision Report (PRD Phase 6)

**Date**: 2026-07-27
**Verdict**: **DROP**
**Applied as**: `config.FADE_ENABLED = False`
**Measured by**: `scripts/fade_requalification.py` (reproducible command in §3)

---

## 1. Pre-committed decision rule

Copied verbatim from `.claude/PRPs/plans/v0.2.0/phase6-fade-requalification.plan.md` Task 6, which
was written before any number below was seen. Evaluated on the tuning span, pooled across
BTCUSDT/ETHUSDT/SOLUSDT, all parameters frozen:

- **KEEP** if *all four* hold: pooled fade `n_trades >= 30`; pooled `expectancy_pct > 0`;
  `expectancy_pct > 0` on **at least 2 of 3** symbols; measured `c <= config.COST_RATIO_CEILING`
  on all 3 symbols.
- **DROP** if pooled `expectancy_pct <= 0`, **or** pooled `n_trades < 30`, **or** only 1 of 3
  symbols is positive, **or** `c > COST_RATIO_CEILING` on any symbol.
- The 2-of-3 bar is deliberately looser than Phase 7's mandatory 3-of-3 gate: this is a
  keep-for-further-testing decision on a droppable `Should` sleeve, not the gate.
- `n_trades < 30` resolving to DROP is intentional. An unjudgeable sample is not a reason to
  carry a sleeve into the gate; "ranging produces no signals" is an acceptable outcome (`PRD:238`).
- **No re-running with a different `--start`, symbol subset, or span to change the answer.**

The measurement was run **once**, on the default span, and the first result is the one recorded
here. No span, symbol subset, or threshold was adjusted after seeing output.

## 2. What changed since the last measurement

| Change | Phase | Effect on the fade sleeve |
|---|---|---|
| `RR_FLOOR` replaces the old absolute reward band; gate applies to the **cost-adjusted** `net_rr` | 2 | New admission filter |
| Frozen honest costs: `FEE_PCT` 0.0004 → 0.0005, plus a funding term | 2 | Every trade is charged more |
| Setup tier 1H → **4H**, trigger tier 15m → **1H**; regime tier → 1D | 4 | Coarser structure, wider stops, fewer setups |
| `interval_ms` now passed to `check_breakout` on the fade path | 4 | Gapped bar pairs no longer read as fresh re-crosses |
| Trending method replaced by Donchian; new trail/channel exits | 5 | Indirect only — see the coupling note below |

**The stop placement did NOT change.** It remains the excursion extreme (`candidate.stop_level`),
per the PRD Decisions Log (`PRD:280`) and Phase 2's deliberate choice. Only the *filter* changed.

**Coupling note (honest caveat).** The engine permits one open trade per symbol at a time, so
Donchian trades occupying that slot can block fade entries. Phase 6's plan declared itself
parallel to Phase 5, but the two are in fact coupled through that slot. The measurement was
therefore run **after** Phase 5 landed, so these numbers describe the state that will actually
ship rather than a Phase-5-less state that never will. Phase 5's new trail and opposite-channel
exits are gated to Donchian trades only and do **not** touch fade exit behavior (verified by
`tests/test_backtest.py::TestFadeEnabledSwitch` and by code inspection of the exit loop).

## 3. Measured numbers

Reproduce with:

```
.venv/bin/python scripts/fade_requalification.py
```

Span: `2023-07-27 -> 2026-04-27` (`WF_OOS_DAYS=90`). **OOS holdout NOT touched: 2026-04-27 -> 2026-07-26.**
The 2023-07-27 start is the 1D regime tier's warmup boundary (`REGIME_MIN_BARS = 207` calendar
days from 2023-01-01); nothing before it carries a non-`uncertain` label.

### Block 2 — fade metrics, per symbol and pooled

| Symbol | trades | win_rate | expectancy | profit_factor | max_dd |
|---|---|---|---|---|---|
| BTCUSDT | 94 | 26.60% | **−0.0603%** | 0.91 | 24.60% |
| ETHUSDT | 67 | 19.40% | **−0.5530%** | 0.58 | 43.38% |
| SOLUSDT | 136 | 22.06% | **−0.5339%** | 0.63 | 90.66% |
| **POOLED** | **297** | 22.90% | **−0.3883%** | **0.67** | 146.28% |

Cross-checked: the BTCUSDT row equals `.venv/bin/python -m trading_bot.cli backtest --symbol
BTCUSDT --start 2023-07-27 --end 2026-04-27`'s `ranging/bollinger-fade` bucket exactly
(`trades=94 win_rate=26.60% expectancy=-0.0603% profit_factor=0.91`), confirming the script
reimplements no screening logic.

### Block 3 — risk-model conformance

| Symbol | median risk_pct | median ATR multiple | measured `c` | ceiling | result |
|---|---|---|---|---|---|
| BTCUSDT | 0.7944% | 0.63x | **0.1812** | 0.10 | **FAIL** |
| ETHUSDT | 1.2574% | 0.75x | **0.1147** | 0.10 | **FAIL** |
| SOLUSDT | 1.4283% | 0.65x | **0.1001** | 0.10 | **FAIL** |

No trades were excluded for undefined ATR (`excluded_nan_atr=0` on all three). The median fade
stop sits at only **0.63–0.75 × ATR(4H)** — less than half the breakout path's frozen
`k = 1.5 × ATR` — which is precisely why `c` fails: cost is a fixed toll, so a stop half as wide
pays twice the toll per unit of risk. Phase 2 confirmed `k = 1.5` clears the ceiling
(c = 0.071/0.052/0.037); the fade's structural stop inherits none of that guarantee, exactly as
the plan's risk table predicted.

### Block 4 — candidate funnel

| Symbol | setups | triggered | passed RR | traded | rr distribution of triggered (min/p25/median/p75/max) |
|---|---|---|---|---|---|
| BTCUSDT | 1009 | 224 | 110 | 94 | 0.03 / 1.18 / 1.85 / 2.87 / 22.00 |
| ETHUSDT | 757 | 149 | 75 | 67 | 0.00 / 1.10 / 1.75 / 3.04 / 6.83 |
| SOLUSDT | 921 | 245 | 153 | 136 | 0.28 / 1.27 / 2.07 / 3.24 / 19.64 |

Roughly half of all triggered candidates are rejected by the R:R floor, and ~85–90% of accepted
ones become trades (the remainder lose the `rank_signals` tie or arrive while a trade is open).
The sample is ample: pooled `n = 297` is ~10× the `n >= 30` clause, so this verdict is **not** a
small-sample artifact.

Footnote on reading the funnel: the "triggered" stage counts candidates that fired *and* were
geometrically valid — the stage is measured by calling `build_fade_signal` with the floor
disabled, and that function independently rejects `risk <= 0` / `reward <= 0` regardless of the
floor. So "triggered" is slightly narrower than "check_breakout returned an event".

### Block 5 — adverse-selection check

Median `stretch_depth = |stop_level − trigger_level| / trigger_level`:

| Symbol | RR-accepted | RR-rejected | accepted shallower? |
|---|---|---|---|
| BTCUSDT | 0.6576% (n=110) | 0.9575% (n=111) | **yes, by 1.46×** |
| ETHUSDT | 0.9414% (n=75) | 1.2564% (n=73) | **yes, by 1.33×** |
| SOLUSDT | 1.1166% (n=153) | 1.9010% (n=91) | **yes, by 1.70×** |

## 4. Verdict: DROP

**Three independent clauses of the DROP rule fire**, any one of which is sufficient:

1. **Pooled `expectancy_pct <= 0`** — measured −0.3883%.
2. **Fewer than 2 of 3 symbols positive** — in fact **0 of 3** are positive.
3. **`c > COST_RATIO_CEILING` on at least one symbol** — it fails on **all three** (0.1812 / 0.1147 / 0.1001).

The one clause that does *not* fire is sample size (`n = 297 >= 30`), which strengthens rather
than weakens the verdict: the sleeve had every opportunity to show edge on an ample sample under
a corrected risk model, and did not. Profit factor 0.67 pooled is worse than the 0.49/0.76/0.86
per-symbol figures recorded pre-Phase-2 (`market-research-capability-benchmark.md:110`) — the
honest cost model and coarser tiers made it look worse, not better.

Applied: `config.FADE_ENABLED = False`. `meanrev.py`, `bollinger.py` and all their tests remain
intact and green, so this decision is reversible and auditable, and a future re-test costs nothing.

## 5. Adverse-selection finding (reported regardless of verdict)

**The `RR_FLOOR` filter is adversely selecting the fade sleeve, and the effect is unambiguous.**
On all three symbols the candidates the floor *accepts* have systematically **shallower** stretches
than the ones it *rejects* (1.33×–1.70× shallower at the median, consistent in direction across
every symbol).

The mechanism is the one the plan predicted: fade reward is bounded by band geometry (target = the
middle band), while risk = entry − excursion extreme. A **deep** stretch therefore has large risk
and small `rr` and gets rejected; a **shallow** stretch passes. So `RR_FLOOR >= 1.5` is doing to
fade a milder version of what `MAX_RISK_PCT` did to breakout: admitting the freakishly tight setups
and discarding the structurally sound ones. Block 3 corroborates it from the other direction — the
median *accepted* stop is only 0.63–0.75 × ATR, i.e. the survivors are the ones with stops too
tight to survive noise.

**Implication for Phase 7**: this is a genuine input, not a Phase 6 fix. A ratio floor that
respects band geometry (or an ATR floor on the fade stop) is a legitimate Phase 7 grid axis with a
logged degree of freedom. It was deliberately **not** improvised here. Note also that this finding
is *independent* of the DROP verdict: even had the sleeve passed, it would have been passing for a
suspect reason, and the same remedy would be owed.

## 6. Trial-log entry

| Item | Value |
|---|---|
| Degrees of freedom consumed by this phase | **Zero.** Nothing swept, no grid, no folds, no parameter selection. Every value is a frozen config default. |
| Data read | One tuning-span read (2023-07-27 → 2026-04-27) of the fade sleeve, 3 symbols. Run **once**. |
| OOS holdout | **Untouched.** 2026-04-27 → 2026-07-26 not read. The script hard-exits 2 if `--end` would cross `tune_end`. |
| Decision rule | Pre-committed in the plan before any number was observed; reproduced verbatim in §1 above. |

**Pre-existing degree-of-freedom debt discovered while planning this phase** (not spent by it):
`walkforward.py`'s `DEFAULT_GRID` contains `"bb_num_std": (1.75, 2.0, 2.25)`, so `BB_STD` has
**already** been fitted on BTCUSDT 2023–2026 by the earlier sweep. This phase declined to touch it
and froze it at the canonical 2.0. Phase 7 removes `bb_num_std` from the grid; until it does, that
consumed DoF is real and belongs in this log. Noteworthy that the fade's only previously-swept
parameter is the one this phase refused to re-touch.

## 7. What this phase did NOT do

- Did not change fade stop placement, `BB_PERIOD`, `BB_STD`, or `FADE_STRETCH_MAX_AGE_BARS`.
- Did not delete `meanrev.py`, `bollinger.py`, or any of their tests — the kill switch is the drop
  mechanism, so the decision stays reversible and the code stays covered.
- Did not design a replacement ranging method. Ranging now produces no signals, which the PRD
  accepts (`PRD:238`).
- Did not run or repair the walk-forward gate (Phase 7), and did not read the OOS holdout.
