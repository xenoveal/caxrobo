# Phase 2 correlation report (v0.3.0)

**Correction to the PRD's premise (contract §0a).** The PRD scopes this phase as "backfill uncorrelated Binance futures symbols", assuming 3 symbols are stored. **20 are stored**, at 1d/4h/1h, over the full 2023-01-01 -> 2026-07-26 span. KNOWN-LIMITATIONS §4's "Only 3 symbols exist in the store" was true at the v0.2.0 merge and is superseded by contract §0a; this report records the supersession rather than editing that file.

Span: `1690416000000` -> `1785024000000`  timeframe: `1d`  n_obs: **1095**  symbols measured: **20**

## Storage cost (measured, not assumed)

`1,183,372` rows in `116,961,280` bytes = **98.84 bytes/row**. `255.6 GiB` free on the DB's filesystem.
One additional symbol's full 1d+4h+1h history costs ~3.8 MB, extrapolated from the 40404 rows/symbol already observed at those three tiers. **Disk is not the constraint; correlation is.**

## Gap integrity (interior gaps only; trailing poller lag excluded)

20 of 20 symbols clean; 0 finding(s) across 60 cells (20 symbols x 3 timeframes).

## Pairwise correlation matrix (Pearson, daily returns)

| | AAVEUSDT | ADAUSDT | APTUSDT | ATOMUSDT | AVAXUSDT | BCHUSDT | BNBUSDT | BTCUSDT | DOGEUSDT | DOTUSDT | ETHUSDT | FILUSDT | LINKUSDT | LTCUSDT | NEARUSDT | OPUSDT | SOLUSDT | TRXUSDT | UNIUSDT | XRPUSDT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAVEUSDT | 1.000 | 0.628 | 0.595 | 0.621 | 0.649 | 0.513 | 0.553 | 0.623 | 0.601 | 0.639 | 0.741 | 0.575 | 0.715 | 0.563 | 0.581 | 0.624 | 0.610 | 0.227 | 0.667 | 0.564 |
| ADAUSDT | 0.628 | 1.000 | 0.608 | 0.689 | 0.725 | 0.520 | 0.558 | 0.687 | 0.711 | 0.753 | 0.711 | 0.608 | 0.720 | 0.620 | 0.610 | 0.592 | 0.689 | 0.236 | 0.572 | 0.738 |
| APTUSDT | 0.595 | 0.608 | 1.000 | 0.679 | 0.697 | 0.483 | 0.564 | 0.624 | 0.647 | 0.715 | 0.663 | 0.651 | 0.671 | 0.602 | 0.647 | 0.689 | 0.636 | 0.244 | 0.578 | 0.517 |
| ATOMUSDT | 0.621 | 0.689 | 0.679 | 1.000 | 0.711 | 0.545 | 0.586 | 0.618 | 0.674 | 0.794 | 0.668 | 0.686 | 0.708 | 0.670 | 0.682 | 0.669 | 0.617 | 0.260 | 0.607 | 0.610 |
| AVAXUSDT | 0.649 | 0.725 | 0.697 | 0.711 | 1.000 | 0.539 | 0.576 | 0.696 | 0.706 | 0.771 | 0.708 | 0.652 | 0.754 | 0.646 | 0.707 | 0.664 | 0.742 | 0.220 | 0.605 | 0.622 |
| BCHUSDT | 0.513 | 0.520 | 0.483 | 0.545 | 0.539 | 1.000 | 0.520 | 0.615 | 0.587 | 0.552 | 0.588 | 0.517 | 0.559 | 0.602 | 0.480 | 0.481 | 0.520 | 0.241 | 0.511 | 0.493 |
| BNBUSDT | 0.553 | 0.558 | 0.564 | 0.586 | 0.576 | 0.520 | 1.000 | 0.661 | 0.594 | 0.605 | 0.673 | 0.527 | 0.579 | 0.548 | 0.543 | 0.530 | 0.600 | 0.323 | 0.519 | 0.482 |
| BTCUSDT | 0.623 | 0.687 | 0.624 | 0.618 | 0.696 | 0.615 | 0.661 | 1.000 | 0.762 | 0.650 | 0.809 | 0.567 | 0.676 | 0.612 | 0.621 | 0.560 | 0.744 | 0.222 | 0.580 | 0.627 |
| DOGEUSDT | 0.601 | 0.711 | 0.647 | 0.674 | 0.706 | 0.587 | 0.594 | 0.762 | 1.000 | 0.720 | 0.743 | 0.619 | 0.667 | 0.639 | 0.632 | 0.604 | 0.677 | 0.202 | 0.598 | 0.624 |
| DOTUSDT | 0.639 | 0.753 | 0.715 | 0.794 | 0.771 | 0.552 | 0.605 | 0.650 | 0.720 | 1.000 | 0.715 | 0.759 | 0.743 | 0.708 | 0.746 | 0.679 | 0.679 | 0.232 | 0.650 | 0.637 |
| ETHUSDT | 0.741 | 0.711 | 0.663 | 0.668 | 0.708 | 0.588 | 0.673 | 0.809 | 0.743 | 0.715 | 1.000 | 0.632 | 0.749 | 0.655 | 0.656 | 0.722 | 0.719 | 0.225 | 0.680 | 0.628 |
| FILUSDT | 0.575 | 0.608 | 0.651 | 0.686 | 0.652 | 0.517 | 0.527 | 0.567 | 0.619 | 0.759 | 0.632 | 1.000 | 0.635 | 0.644 | 0.692 | 0.644 | 0.562 | 0.215 | 0.575 | 0.535 |
| LINKUSDT | 0.715 | 0.720 | 0.671 | 0.708 | 0.754 | 0.559 | 0.579 | 0.676 | 0.667 | 0.743 | 0.749 | 0.635 | 1.000 | 0.659 | 0.660 | 0.664 | 0.689 | 0.228 | 0.637 | 0.649 |
| LTCUSDT | 0.563 | 0.620 | 0.602 | 0.670 | 0.646 | 0.602 | 0.548 | 0.612 | 0.639 | 0.708 | 0.655 | 0.644 | 0.659 | 1.000 | 0.593 | 0.584 | 0.568 | 0.226 | 0.555 | 0.651 |
| NEARUSDT | 0.581 | 0.610 | 0.647 | 0.682 | 0.707 | 0.480 | 0.543 | 0.621 | 0.632 | 0.746 | 0.656 | 0.692 | 0.660 | 0.593 | 1.000 | 0.614 | 0.635 | 0.203 | 0.575 | 0.535 |
| OPUSDT | 0.624 | 0.592 | 0.689 | 0.669 | 0.664 | 0.481 | 0.530 | 0.560 | 0.604 | 0.679 | 0.722 | 0.644 | 0.664 | 0.584 | 0.614 | 1.000 | 0.611 | 0.201 | 0.613 | 0.515 |
| SOLUSDT | 0.610 | 0.689 | 0.636 | 0.617 | 0.742 | 0.520 | 0.600 | 0.744 | 0.677 | 0.679 | 0.719 | 0.562 | 0.689 | 0.568 | 0.635 | 0.611 | 1.000 | 0.242 | 0.556 | 0.586 |
| TRXUSDT | 0.227 | 0.236 | 0.244 | 0.260 | 0.220 | 0.241 | 0.323 | 0.222 | 0.202 | 0.232 | 0.225 | 0.215 | 0.228 | 0.226 | 0.203 | 0.201 | 0.242 | 1.000 | 0.239 | 0.207 |
| UNIUSDT | 0.667 | 0.572 | 0.578 | 0.607 | 0.605 | 0.511 | 0.519 | 0.580 | 0.598 | 0.650 | 0.680 | 0.575 | 0.637 | 0.555 | 0.575 | 0.613 | 0.556 | 0.239 | 1.000 | 0.517 |
| XRPUSDT | 0.564 | 0.738 | 0.517 | 0.610 | 0.622 | 0.493 | 0.482 | 0.627 | 0.624 | 0.637 | 0.628 | 0.535 | 0.649 | 0.651 | 0.535 | 0.515 | 0.586 | 0.207 | 0.517 | 1.000 |

## Effective independent sample size (Kish design effect)

Correctness check: this must reproduce KNOWN-LIMITATIONS §0b's anchor -- config.SYMBOLS at m=3, r_bar~=0.7574, N_eff~=1.193.

| set | n | r_bar | N_eff (Kish) | N_eff (participation ratio) |
|---|---|---|---|---|
| production | 3 | 0.7574 | 1.193 | 1.395 |
| stored | 20 | 0.5905 | 1.637 | 2.505 |
| selection | 9 | 0.4885 | 1.834 | 2.925 |

## Per-symbol statistics

| symbol | n_bars | mean_r | btc_beta | median_quote_volume | interior_gaps | eligible | reason |
|---|---|---|---|---|---|---|---|
| AAVEUSDT | 1095 | 0.594 | 1.230 | 109,179,660 | 0 | True | - |
| ADAUSDT | 1095 | 0.630 | 1.338 | 252,868,656 | 0 | True | - |
| APTUSDT | 1095 | 0.606 | 1.207 | 92,898,081 | 0 | True | - |
| ATOMUSDT | 1095 | 0.636 | 1.007 | 51,744,932 | 0 | True | - |
| AVAXUSDT | 1095 | 0.652 | 1.321 | 233,879,011 | 0 | True | - |
| BCHUSDT | 1095 | 0.519 | 1.087 | 158,503,022 | 0 | True | - |
| BNBUSDT | 1095 | 0.555 | 0.739 | 397,270,472 | 0 | True | - |
| BTCUSDT | 1095 | 0.629 | 1.000 | 13,720,529,471 | 0 | True | - |
| DOGEUSDT | 1095 | 0.632 | 1.433 | 709,419,496 | 0 | False | beta |
| DOTUSDT | 1095 | 0.671 | 1.142 | 123,461,750 | 0 | True | - |
| ETHUSDT | 1095 | 0.668 | 1.140 | 8,753,160,661 | 0 | True | - |
| FILUSDT | 1095 | 0.595 | 1.227 | 128,014,776 | 0 | True | - |
| LINKUSDT | 1095 | 0.651 | 1.232 | 238,483,667 | 0 | True | - |
| LTCUSDT | 1095 | 0.597 | 0.911 | 168,163,688 | 0 | True | - |
| NEARUSDT | 1095 | 0.601 | 1.378 | 135,527,516 | 0 | False | beta |
| OPUSDT | 1095 | 0.593 | 1.217 | 106,897,971 | 0 | True | - |
| SOLUSDT | 1095 | 0.615 | 1.329 | 2,610,898,511 | 0 | True | - |
| TRXUSDT | 1095 | 0.231 | 0.327 | 67,306,661 | 0 | True | - |
| UNIUSDT | 1095 | 0.570 | 1.285 | 90,925,246 | 0 | True | - |
| XRPUSDT | 1095 | 0.565 | 1.027 | 840,354,261 | 0 | True | - |

## Selection rule and result

Rule (recorded so it is auditable, not re-litigated later): eligibility screens on zero interior gaps, minimum daily-return overlap, a liquidity floor, and a BTC-beta ceiling; the anchor is pinned regardless of rank and exempt from the beta screen only; remaining candidates rank ascending by mean pairwise Pearson r (not beta -- r_bar is what the effective-N formula consumes); ties break on higher median quote volume then alphabetically; the first CORRELATION_SELECT_N are taken. Anchor first, then rank order -- never reordered.

`RESEARCH_SYMBOLS` candidate: `('BTCUSDT', 'TRXUSDT', 'BCHUSDT', 'BNBUSDT', 'XRPUSDT', 'UNIUSDT', 'OPUSDT', 'AAVEUSDT', 'FILUSDT')`

## Verdict: D1

D1: breadth confirmed, with 'confirmed' meaning ~1.54x independent information, not the 9x row count. selection N_eff(Kish)=1.834 is 1.537x config.SYMBOLS' N_eff=1.193 (threshold 1.5x). RESEARCH_SYMBOLS promoted; no backfill performed.

**What breadth buys, kept apart.** Rows rise 9/3 = 3.00x; independent information (Kish effective N) rises only 1.54x. Row count clears the gate's mechanical n_trades floor; it does not clear the statistics DSR needs. Phase 9 must read its DSR as governed by effective N ~= 1.8, not by 9 symbols.
ESTIMATE, at v0.2.0's measured OOS trade rate (0.0852 trades/symbol/day over 90 days, KNOWN-LIMITATIONS §1) -- NOT measured by this phase's own command, and labelled as such: a 9-symbol pool would yield roughly 69 OOS trades per 90-day holdout (mechanical sample adequacy, gate floor 30), but only about 14 independent-equivalent trades after deflating by effective N -- against v0.2.0's own ~9. Phase 9 is the one that must actually measure the real rate.

**Answer to PRD Open Question #4** ("are enough uncorrelated symbols available on Binance futures?"): **no, not within this venue.** 19 of the 20 stored liquid majors sit at BTC correlation roughly 0.56-0.81; the sole exception measured here is TRXUSDT, which is also among the least liquid symbols measured. The achievable ceiling from this venue's liquid majors is an effective N around 1.6-1.8 (stored-universe / selection Kish figures above), not 8. A genuine diversifier would need a different asset class, which the PRD excludes ("Binance futures only").

**Contingency: what would justify a genuine backfill (D2/D3 path).** (a) fewer than CORRELATION_SELECT_N eligible symbols remain after the screens, or (b) a candidate class is found with measured r vs BTC below ~0.4 AND median quote volume above CORRELATION_MIN_QUOTE_VOLUME_USD. The acceptance test is stated in advance: such a backfill must raise the selection's Kish effective N by >= 0.25 absolute, re-measured by re-running correlation-report, or it is not worth the download.

## Degrees of freedom consumed

- **Zero strategy degrees of freedom.** No parameter affecting a trade was swept, fitted, or chosen by looking at P&L. RESEARCH_SYMBOLS follows a rule pre-registered BEFORE the selection was run, on correlation / liquidity / beta -- none of which is a return.
- **Two thresholds set to be non-binding, deliberately**: CORRELATION_MIN_QUOTE_VOLUME_USD and CORRELATION_MAX_BTC_BETA. Recorded as guards for a future expansion, not fitted values.
- **One judgement call that is not a measurement**: CORRELATION_EFFECTIVE_N_MIN_RATIO = 1.5, chosen before this selection's effective N was known -- pre-registered, and the margin by which the measured ratio clears or misses it is stated above.
- **One pinned anchor**: BTCUSDT, at a measured Kish effective-N cost, for the reasons in select_research_symbols' docstring.
