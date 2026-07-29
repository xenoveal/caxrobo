# Deprecated artifacts — untracked from git 2026-07-28

Five machine-generated report artifacts were removed from git tracking during the
v0.3.0 documentation audit. **Every one of them still exists on disk, and every one
is permanently recoverable from git history.**

## Why

They were **generated output, not authored documentation** — 17.1 MB across five
files, which was **81% of the entire 21.3 MB tracked repository**. Two of them
(13.9 MB combined) are v0.1.0-era HTML dumps that no file in the repo references.

The line drawn was deliberate and mechanical, so it can be audited rather than
argued with:

> Untrack a **generated** artifact if it exceeds 100 KB. Keep every authored
> document, every test fixture, and every small artifact that a live document
> cites as evidence.

That rule keeps `bad-backtest-result-investigation-result.html` (68 KB, cited by
`pivot-guide.md`), `results/HOLDOUT_shortlist.csv` (3 KB, cited by
`bruteforce-strategy/FINDINGS.md`) and `phase8-detector-report.out.md` (4.5 KB,
cited by `phase8-detector-edge-report.md`) tracked. It also leaves
`src/trading_bot/ui/static/index.html` and `tests/fixtures/patterns/*.csv` alone —
those are source and test inputs, not generated output.

## Nothing was lost, and here is why that is a fact rather than a reassurance

**Git history was NOT rewritten.** No `filter-repo`, no `BFG`, no force-push. The
blobs remain in the object database, reachable by the commit that last contained
them. Recovery is a one-liner per file, listed below.

The honest consequence of that same decision, stated plainly: **a fresh `git clone`
still downloads all 17.1 MB**, because the objects are still in history. Untracking
stops these paths from growing the repo further and removes them from the index — it
does not shrink the clone. Reclaiming the 17.1 MB would require rewriting shared
history on `feat/v0.3.0`, which was deliberately not done unprompted.

## The five artifacts

| Artifact | Size | Last commit | Referenced by | Regenerable |
|---|---|---|---|---|
| `backtest-result-initial.html` | 7,526,603 B | `7f692ac` 2026-07-26 | `pivot-guide.md:6` | no (v0.1.0 engine is gone) |
| `signal-report-after-breakout-fix.html` | 6,754,536 B | `7f692ac` 2026-07-26 | **nothing** | no (v0.1.0 engine is gone) |
| `phase7-walkforward-result.chart.html` | 152,857 B | `fa81718` 2026-07-27 | `KNOWN-LIMITATIONS.md:12` | **yes** — see below |
| `bruteforce-strategy/results/volatility_TRAIN.csv` | 1,377,673 B | `48ef27a` 2026-07-27 | **nothing** | via `scripts/bruteforce/runner.py` |
| `bruteforce-strategy/results/volatility_SELECT.csv` | 1,311,646 B | `48ef27a` 2026-07-27 | **nothing** | via `scripts/bruteforce/runner.py` |

### sha256 — verify a recovered copy against these

```
88735e15148bfb9f4bbda77b351238c3e0cbc675cdc469bc844b5a4e000f88ba  backtest-result-initial.html
74cb8a5fc312d592e5a8ef22341a2b0062fecb1e78db7e6465832fadde2e8f29  signal-report-after-breakout-fix.html
1be23b51a81cfe26bc85b370d3843644104ad30fbad4146956521a5fca216920  phase7-walkforward-result.chart.html
7fd901329ad29cde0bf0097ecfd611298c78cbf83dff923c15e6ba001454064f  results/volatility_TRAIN.csv
58fd1faf50a7ba766c81cbbd12c49c088a18f7afca5edf5583e4b07967c1bcd0  results/volatility_SELECT.csv
```

## Recovery

Restore any one of them from history (run from the repo root):

```bash
git show 7f692ac:.claude/PRPs/reports/backtest-result-initial.html \
  > .claude/PRPs/reports/backtest-result-initial.html

git show 7f692ac:.claude/PRPs/reports/signal-report-after-breakout-fix.html \
  > .claude/PRPs/reports/signal-report-after-breakout-fix.html

git show fa81718:.claude/PRPs/reports/phase7-walkforward-result.chart.html \
  > .claude/PRPs/reports/phase7-walkforward-result.chart.html

git show 48ef27a:'.claude/PRPs/reports/bruteforce-strategy/results/volatility_TRAIN.csv' \
  > .claude/PRPs/reports/bruteforce-strategy/results/volatility_TRAIN.csv

git show 48ef27a:'.claude/PRPs/reports/bruteforce-strategy/results/volatility_SELECT.csv' \
  > .claude/PRPs/reports/bruteforce-strategy/results/volatility_SELECT.csv
```

Then verify with `shasum -a 256 <path>` against the list above.

The walk-forward chart is the one artifact cheaper to regenerate than to recover —
`scripts/build_performance_chart.py:15` documents the command:

```bash
.venv/bin/python scripts/build_performance_chart.py \
  --out .claude/PRPs/reports/phase7-walkforward-result.chart.html
```

## If you want the 17.1 MB back out of the clone

That is a separate, deliberate decision, because it rewrites published history on
`feat/v0.3.0` and every collaborator must re-clone or hard-reset:

```bash
git filter-repo --invert-paths \
  --path '.claude/PRPs/reports/backtest-result-initial.html' \
  --path '.claude/PRPs/reports/signal-report-after-breakout-fix.html' \
  --path '.claude/PRPs/reports/phase7-walkforward-result.chart.html' \
  --path '.claude/PRPs/reports/bruteforce-strategy/results/volatility_TRAIN.csv' \
  --path '.claude/PRPs/reports/bruteforce-strategy/results/volatility_SELECT.csv'
```

**Do this only after the recovered copies are verified and stored outside the repo** —
`filter-repo` destroys the very blobs this document's recovery section depends on, so
running it makes the commands above useless.
