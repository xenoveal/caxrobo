"""
The coverage ledger: every one of the catalog's 144 pattern rows, enumerated in
code, with a status and — for anything not covered — a stated reason.

WHY THIS FILE EXISTS. Without it, "Phase 8 covers tiers 1-2" is a claim. With
it, coverage is TRACKED: `tests/test_detectors_catalog.py` parses
`.claude/technical-pattern.md`, collects every table row's left-column text from
the 18 numbered family sections (excluding the reliability table's 15 rows), and
asserts set equality against `{e.pattern for e in CATALOG}` plus
`len(CATALOG) == 144`. The catalog document IS the specification, and a ledger
that cannot drift from its spec is the only kind worth having.

MEASURED SIZE, not estimated: 144 pattern rows across 18 families = 159 total
markdown table rows minus the 15-row `# Highest Historical Reliability` table.

WHAT THE THREE STATUSES MEAN
  * `covered` — a registered Phase 8 detector serves this row. 19 rows.
    `detector_key` is deliberately NOT unique: `detector.rsi-divergence` serves
    four rows (regular and hidden, bullish and bearish). The uniqueness
    invariant runs the other way — (family_no, pattern) is unique.
  * `deferred` — buildable, deliberately not built now, with the reason. 6 rows.
  * `out-of-scope` — 119 rows, each carrying ONE of the five reasons in
    OUT_OF_SCOPE_REASONS. Those five are distinct and load-bearing: a single
    copy-pasted reason would defeat the whole point of enumerating, which is why
    `test_non_covered_entries_state_a_reason` bounds how often one reason may
    repeat.

THE HEADLINE, stated plainly because 13.2% reads as failure to anyone who
skipped the PRD: the full catalog is "a direction, not a v1 gate". This phase
closes contract §9's tier 2 completely (6 of 6) and lands tier 1 at 2 of 3, with
Wyckoff Accumulation/Distribution `deferred` for the reason in their entries.

TIER SEMANTICS. `tier` is read off the catalog's own reliability table, mapped
onto the rows that table names, and is None where the table is silent. Two
mappings are worth stating because they are judgements rather than lookups:
"Head & Shoulders" in the reliability table is taken to cover BOTH the upright
and inverse rows (one shape, mirrored), and "RSI Divergence" is taken to cover
all four divergence rows. The reliability table's "Cup & Handle" is taken to
name only the UPRIGHT row, so `Inverse Cup and Handle` carries tier=None — it is
a free-rider mirror and no success criterion depends on it.
"""

from dataclasses import dataclass

# The five distinct out-of-scope reasons. Every out-of-scope entry's `reason`
# begins with exactly one of these, followed by ": " and row-specific detail.
# Pinned by tests/test_detectors_catalog.py.
OUT_OF_SCOPE_REASONS = (
    "tier 3-4 by contract §9",
    "belongs to another plug-in kind",
    "no reference implementation and no labelled data",
    "owned by another phase",
    "already covered by another registered plug-in",
)

STATUSES = ("covered", "deferred", "out-of-scope")

# Contract §9's reliability tiers are stated as CONCEPTS ("Double Top/Bottom",
# "Bull/Bear Flag", "RSI Divergence"), each of which maps onto one or more
# catalog ROWS. Both counts are reported, because the two differ and quoting
# only one invites a misreading: §9's commitment is "tier 1 lands 2 of 3, tier 2
# lands 6 of 6", which is a CONCEPT count, while the ledger's `tier` field
# counts rows. Mapping fixed here so the scorecard cannot drift.
SECTION9_TIER1_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Cup & Handle", ("Cup and Handle",)),
    ("Head & Shoulders", ("Head & Shoulders",)),
    ("Wyckoff Accumulation/Distribution", ("Accumulation", "Distribution")),
)
SECTION9_TIER2_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Double Top/Bottom", ("Double Top", "Double Bottom")),
    ("Bull/Bear Flag", ("Bull Flag", "Bear Flag")),
    ("Ascending Triangle", ("Ascending Triangle",)),
    ("Falling Wedge", ("Falling Wedge",)),
    ("Rising Wedge", ("Rising Wedge",)),
    (
        "RSI Divergence",
        (
            "Bullish Divergence",
            "Bearish Divergence",
            "Hidden Bullish Divergence",
            "Hidden Bearish Divergence",
        ),
    ),
)

# Measured by parsing .claude/technical-pattern.md, not estimated from the PRD's
# "~150". Asserted against the document by test_catalog_matches_the_source_document.
CATALOG_ROW_COUNT = 144
CATALOG_FAMILY_COUNT = 18


@dataclass(frozen=True)
class CatalogEntry:
    """One row of `.claude/technical-pattern.md`, with this phase's verdict on it.

    Attributes:
        family_no: 1..18, matching the document's `# <n>. <name>` headings.
        family: The heading text.
        pattern: The EXACT left-column text. A paraphrase fails
            test_catalog_matches_the_source_document, which is the point.
        tier: contract §9 reliability tier, or None where the reliability table
            is silent about this row.
        status: One of STATUSES.
        detector_key: The registry key serving this row, when covered.
        reason: Required unless covered. For out-of-scope rows it begins with one
            of OUT_OF_SCOPE_REASONS.
        subsection: The `## <name>` sub-heading, where the family has them. Only
            family 14 (Oscillator Signals) does — RSI / MACD / Stochastic — and
            it is REQUIRED there rather than cosmetic: the MACD and Stochastic
            subsections each contain a row literally named "Bullish Cross" and
            one named "Bearish Cross", so (family_no, pattern) is NOT unique and
            (family_no, subsection, pattern) is.
    """

    family_no: int
    family: str
    pattern: str
    tier: int | None
    status: str
    detector_key: str | None = None
    reason: str = ""
    subsection: str = ""


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Head & Shoulders",
        tier=1,
        status="covered",
        detector_key="detector.head-and-shoulders",
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Inverse Head & Shoulders",
        tier=1,
        status="covered",
        detector_key="detector.inverse-head-and-shoulders",
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Double Top",
        tier=2,
        status="covered",
        detector_key="detector.double-top",
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Double Bottom",
        tier=2,
        status="covered",
        detector_key="detector.double-bottom",
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Triple Top",
        tier=None,
        status="deferred",
        detector_key=None,
        reason=(
            "DEFERRED: a double-top generalisation -- one extra pivot in "
            "_detect_double's pairing loop. Left for a later tier so this phase's "
            "degrees of freedom stay countable; the code it needs already exists in "
            "plugins/detectors/reversal.py."
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Triple Bottom",
        tier=None,
        status="deferred",
        detector_key=None,
        reason=(
            "DEFERRED: a double-top generalisation -- one extra pivot in "
            "_detect_double's pairing loop. Left for a later tier so this phase's "
            "degrees of freedom stay countable; the code it needs already exists in "
            "plugins/detectors/reversal.py."
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Rounded Top",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: a dome with no handle. "
            "The cup detector's roundness test would port cheaply, but 'rounded' has no "
            "agreed numeric definition beyond the base-bar count chosen here and no "
            "labelled data to calibrate that count against"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Rounded Bottom (Saucer)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: a cup without a handle; "
            "same missing calibration as Rounded Top, and without the handle there is "
            "no bar at which the structure is complete"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Diamond Top",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: a broadening formation "
            "followed by a converging one; no donor, and the transition bar between the "
            "two phases has no numeric definition"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Diamond Bottom",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: mirror of Diamond Top, "
            "and unbuildable for the same reason"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Island Reversal Top",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: requires an exhaustion "
            "gap on both sides of the island. Crypto perpetuals trade 24/7, so true "
            "gaps are rare in the stored history and there is no labelled set of them"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Island Reversal Bottom",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: mirror of Island "
            "Reversal Top; gaps are rare in 24/7 perp data"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Bump and Run Top",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: defined by a lead-in "
            "trendline angle and a 'bump' whose steepness threshold is drawn by eye in "
            "every published description; no donor and nothing to calibrate against"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Bump and Run Bottom",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: mirror of Bump and Run "
            "Top; the same undefined steepness threshold"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Broadening Top",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: an EXPANDING structure: "
            "_geometry.fit_converging_lines requires convergence by construction, so "
            "this needs a second, diverging-line fit that has no donor in the repo"
        ),
    ),
    CatalogEntry(
        family_no=1,
        family="Reversal Patterns",
        pattern="Broadening Bottom",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: mirror of Broadening "
            "Top; needs the same diverging-line fit"
        ),
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Bull Flag",
        tier=2,
        status="covered",
        detector_key="detector.bull-flag",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Bear Flag",
        tier=2,
        status="covered",
        detector_key="detector.bear-flag",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Bull Pennant",
        tier=None,
        status="deferred",
        detector_key=None,
        reason=(
            "DEFERRED: a flag whose consolidation CONVERGES rather than running "
            "parallel. It needs _geometry.fit_converging_lines called inside the flag "
            "scan -- a genuine small build, deferred to keep contract §9's tier 2 "
            "closed at 6 of 6 rather than opened at 7 of 8."
        ),
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Bear Pennant",
        tier=None,
        status="deferred",
        detector_key=None,
        reason=(
            "DEFERRED: a flag whose consolidation CONVERGES rather than running "
            "parallel. It needs _geometry.fit_converging_lines called inside the flag "
            "scan -- a genuine small build, deferred to keep contract §9's tier 2 "
            "closed at 6 of 6 rather than opened at 7 of 8."
        ),
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Ascending Triangle",
        tier=2,
        status="covered",
        detector_key="detector.ascending-triangle",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Descending Triangle",
        tier=None,
        status="covered",
        detector_key="detector.descending-triangle",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Symmetrical Triangle",
        tier=None,
        status="covered",
        detector_key="detector.symmetrical-triangle",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Rising Wedge",
        tier=2,
        status="covered",
        detector_key="detector.rising-wedge",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Falling Wedge",
        tier=2,
        status="covered",
        detector_key="detector.falling-wedge",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Rectangle (Trading Range)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: a horizontal range "
            "breakout, which detector.donchian-breakout already emits on the setup tier "
            "with a channel-width measured move"
        ),
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Ascending Channel",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: two PARALLEL rising "
            "boundaries. fit_converging_lines rejects non-converging boundaries by "
            "construction, and no parallel-channel fit exists anywhere in the repo to "
            "port"
        ),
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Descending Channel",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: mirror of Ascending "
            "Channel; needs the same parallel-line fit"
        ),
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Cup and Handle",
        tier=1,
        status="covered",
        detector_key="detector.cup-and-handle",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Inverse Cup and Handle",
        tier=None,
        status="covered",
        detector_key="detector.inverse-cup-and-handle",
    ),
    CatalogEntry(
        family_no=2,
        family="Continuation Patterns",
        pattern="Megaphone Continuation",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: an expanding structure, "
            "like Broadening Top, and additionally requires a prevailing-trend "
            "judgement this phase deliberately leaves to a Confirmation"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Hammer",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. A donor exists at "
            "scripts/bruteforce/indicators.py:442, so a later port is cheap"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Inverted Hammer",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Bullish Engulfing",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. A donor exists at "
            "scripts/bruteforce/indicators.py:426, so a later port is cheap"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Piercing Line",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Morning Star",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Morning Doji Star",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Three White Soldiers",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Tweezer Bottom",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Dragonfly Doji",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Bullish Harami",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Bullish Kicker",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Three Inside Up",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Three Outside Up",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=3,
        family="Bullish Candlestick Patterns",
        pattern="Abandoned Baby Bottom",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Hanging Man",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Shooting Star",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. A donor exists at "
            "scripts/bruteforce/indicators.py:448, so a later port is cheap"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Bearish Engulfing",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. A donor exists at "
            "scripts/bruteforce/indicators.py:434, so a later port is cheap"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Dark Cloud Cover",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Evening Star",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Evening Doji Star",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Three Black Crows",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Tweezer Top",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Gravestone Doji",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Bearish Harami",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Bearish Kicker",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Three Inside Down",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Three Outside Down",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=4,
        family="Bearish Candlestick Patterns",
        pattern="Abandoned Baby Top",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=5,
        family="Indecision Candlestick Patterns",
        pattern="Doji",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. A donor exists at "
            "scripts/bruteforce/indicators.py:456, so a later port is cheap"
        ),
    ),
    CatalogEntry(
        family_no=5,
        family="Indecision Candlestick Patterns",
        pattern="Long-legged Doji",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=5,
        family="Indecision Candlestick Patterns",
        pattern="Spinning Top",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=5,
        family="Indecision Candlestick Patterns",
        pattern="High Wave Candle",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. No donor exists; a "
            "later port would start from scratch"
        ),
    ),
    CatalogEntry(
        family_no=5,
        family="Indecision Candlestick Patterns",
        pattern="Marubozu",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates 'Candlestick Pattern Alone' two "
            "stars, the lowest band in the catalog's own reliability table, and a "
            "single-bar shape carries no level and no measured move. A donor exists at "
            "scripts/bruteforce/indicators.py:461, so a later port is cheap"
        ),
    ),
    CatalogEntry(
        family_no=6,
        family="Gap Patterns",
        pattern="Breakaway Gap",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: crypto perpetuals trade "
            "24/7, so true gaps are rare in the stored history and there is no labelled "
            "set of them to validate against; additionally the gap must be measured "
            "against a prior consolidation"
        ),
    ),
    CatalogEntry(
        family_no=6,
        family="Gap Patterns",
        pattern="Runaway (Measuring) Gap",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: crypto perpetuals trade "
            "24/7, so true gaps are rare in the stored history and there is no labelled "
            "set of them to validate against; additionally the measured move is a "
            "multiple of the prior leg, itself undefined"
        ),
    ),
    CatalogEntry(
        family_no=6,
        family="Gap Patterns",
        pattern="Exhaustion Gap",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: crypto perpetuals trade "
            "24/7, so true gaps are rare in the stored history and there is no labelled "
            "set of them to validate against; additionally distinguishing it from a "
            "breakaway gap needs the subsequent bars"
        ),
    ),
    CatalogEntry(
        family_no=6,
        family="Gap Patterns",
        pattern="Common Gap",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: crypto perpetuals trade "
            "24/7, so true gaps are rare in the stored history and there is no labelled "
            "set of them to validate against; additionally by definition it carries no "
            "directional information"
        ),
    ),
    CatalogEntry(
        family_no=7,
        family="Harmonic Patterns",
        pattern="Gartley",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Gartley is defined by "
            "Fibonacci retracement ratios between five swing points with tolerances no "
            "two sources agree on, there is no reference implementation, and no "
            "labelled dataset exists to measure precision against (PRD Research "
            "Summary)"
        ),
    ),
    CatalogEntry(
        family_no=7,
        family="Harmonic Patterns",
        pattern="Butterfly",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Butterfly is defined by "
            "Fibonacci retracement ratios between five swing points with tolerances no "
            "two sources agree on, there is no reference implementation, and no "
            "labelled dataset exists to measure precision against (PRD Research "
            "Summary)"
        ),
    ),
    CatalogEntry(
        family_no=7,
        family="Harmonic Patterns",
        pattern="Bat",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Bat is defined by "
            "Fibonacci retracement ratios between five swing points with tolerances no "
            "two sources agree on, there is no reference implementation, and no "
            "labelled dataset exists to measure precision against (PRD Research "
            "Summary)"
        ),
    ),
    CatalogEntry(
        family_no=7,
        family="Harmonic Patterns",
        pattern="Crab",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Crab is defined by "
            "Fibonacci retracement ratios between five swing points with tolerances no "
            "two sources agree on, there is no reference implementation, and no "
            "labelled dataset exists to measure precision against (PRD Research "
            "Summary)"
        ),
    ),
    CatalogEntry(
        family_no=7,
        family="Harmonic Patterns",
        pattern="Deep Crab",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Deep Crab is defined by "
            "Fibonacci retracement ratios between five swing points with tolerances no "
            "two sources agree on, there is no reference implementation, and no "
            "labelled dataset exists to measure precision against (PRD Research "
            "Summary)"
        ),
    ),
    CatalogEntry(
        family_no=7,
        family="Harmonic Patterns",
        pattern="Shark",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Shark is defined by "
            "Fibonacci retracement ratios between five swing points with tolerances no "
            "two sources agree on, there is no reference implementation, and no "
            "labelled dataset exists to measure precision against (PRD Research "
            "Summary)"
        ),
    ),
    CatalogEntry(
        family_no=7,
        family="Harmonic Patterns",
        pattern="Cypher",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Cypher is defined by "
            "Fibonacci retracement ratios between five swing points with tolerances no "
            "two sources agree on, there is no reference implementation, and no "
            "labelled dataset exists to measure precision against (PRD Research "
            "Summary)"
        ),
    ),
    CatalogEntry(
        family_no=7,
        family="Harmonic Patterns",
        pattern="ABCD",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: ABCD is defined by "
            "Fibonacci retracement ratios between five swing points with tolerances no "
            "two sources agree on, there is no reference implementation, and no "
            "labelled dataset exists to measure precision against (PRD Research "
            "Summary)"
        ),
    ),
    CatalogEntry(
        family_no=8,
        family="Elliott Wave",
        pattern="Impulse Wave",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Impulse Wave labelling "
            "is recursive and famously non-unique -- two analysts label the same bars "
            "differently -- so there is no reference implementation and nothing to "
            "score against; contract §9 also rates Elliott Wave two stars"
        ),
    ),
    CatalogEntry(
        family_no=8,
        family="Elliott Wave",
        pattern="Corrective Wave",
        tier=4,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Corrective Wave "
            "labelling is recursive and famously non-unique -- two analysts label the "
            "same bars differently -- so there is no reference implementation and "
            "nothing to score against; contract §9 also rates Elliott Wave two stars"
        ),
    ),
    CatalogEntry(
        family_no=9,
        family="Wyckoff Structures",
        pattern="Accumulation",
        tier=1,
        status="deferred",
        detector_key=None,
        reason=(
            "DEFERRED per stated assumption A1: an accumulation is a SEQUENCE OF PHASES "
            "(PS -> SC -> AR -> ST -> Spring -> LPS -> SOS) whose distinguishing "
            "evidence is effort-versus-result judged across weeks. There is no agreed "
            "numeric definition, no reference implementation, and no labelled dataset, "
            "so achievable precision is not merely low -- it is UNMEASURABLE. The "
            "buildable subset ships as detector.wyckoff-spring and "
            "detector.wyckoff-upthrust, whose rationales state the non-claim. Contract "
            "§9's tier-1 row therefore lands 2 of 3, and that is reported rather than "
            "hidden."
        ),
    ),
    CatalogEntry(
        family_no=9,
        family="Wyckoff Structures",
        pattern="Distribution",
        tier=1,
        status="deferred",
        detector_key=None,
        reason=(
            "DEFERRED per stated assumption A1: an accumulation is a SEQUENCE OF PHASES "
            "(PS -> SC -> AR -> ST -> Spring -> LPS -> SOS) whose distinguishing "
            "evidence is effort-versus-result judged across weeks. There is no agreed "
            "numeric definition, no reference implementation, and no labelled dataset, "
            "so achievable precision is not merely low -- it is UNMEASURABLE. The "
            "buildable subset ships as detector.wyckoff-spring and "
            "detector.wyckoff-upthrust, whose rationales state the non-claim. Contract "
            "§9's tier-1 row therefore lands 2 of 3, and that is reported rather than "
            "hidden."
        ),
    ),
    CatalogEntry(
        family_no=9,
        family="Wyckoff Structures",
        pattern="Spring",
        tier=None,
        status="covered",
        detector_key="detector.wyckoff-spring",
    ),
    CatalogEntry(
        family_no=9,
        family="Wyckoff Structures",
        pattern="Upthrust",
        tier=None,
        status="covered",
        detector_key="detector.wyckoff-upthrust",
    ),
    CatalogEntry(
        family_no=10,
        family="Volume-Based Patterns",
        pattern="High Volume Breakout",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: "
            "confirmation.volume-breakout (Phase 4) already gates a breakout on the "
            "same volume ratio, using config.VOLUME_LOOKBACK / VOLUME_HIGH_RATIO"
        ),
    ),
    CatalogEntry(
        family_no=10,
        family="Volume-Based Patterns",
        pattern="Volume Climax",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a volume reading has no level and no "
            "measured move, so it cannot become a PositionPlan on its own; it is "
            "Confirmation material"
        ),
    ),
    CatalogEntry(
        family_no=10,
        family="Volume-Based Patterns",
        pattern="Low Volume Pullback",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a pullback-quality reading, i.e. a "
            "Confirmation on an existing setup rather than an event with its own entry"
        ),
    ),
    CatalogEntry(
        family_no=10,
        family="Volume-Based Patterns",
        pattern="High Volume Rejection",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: the rejection needs a LEVEL supplied by "
            "some other detector; on its own it is a Confirmation, which is why "
            "detector.wyckoff-upthrust supplies the level and the volume test together"
        ),
    ),
    CatalogEntry(
        family_no=11,
        family="Support & Resistance Patterns",
        pattern="Resistance Breakout",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: detector.donchian-breakout, "
            "whose level IS the trailing channel high"
        ),
    ),
    CatalogEntry(
        family_no=11,
        family="Support & Resistance Patterns",
        pattern="Support Breakdown",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: detector.donchian-breakout "
            "on the short side, whose level is the trailing channel low"
        ),
    ),
    CatalogEntry(
        family_no=11,
        family="Support & Resistance Patterns",
        pattern="False Breakout (Bull Trap)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: geometrically the same "
            "event as detector.wyckoff-upthrust: a probe above an established range "
            "that closes back inside. Recorded here so the duplication is visible "
            "rather than counted twice"
        ),
    ),
    CatalogEntry(
        family_no=11,
        family="Support & Resistance Patterns",
        pattern="False Breakdown (Bear Trap)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: geometrically the same "
            "event as detector.wyckoff-spring"
        ),
    ),
    CatalogEntry(
        family_no=11,
        family="Support & Resistance Patterns",
        pattern="Retest & Hold",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: entry timing on an already-detected "
            "level, which is what signals/breakout.py's crossing rule and a "
            "Confirmation do; it has no structure of its own"
        ),
    ),
    CatalogEntry(
        family_no=11,
        family="Support & Resistance Patterns",
        pattern="Rejection at Level",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a Confirmation on a level some other "
            "detector supplied"
        ),
    ),
    CatalogEntry(
        family_no=12,
        family="Trendline Patterns",
        pattern="Trendline Bounce",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a Confirmation on a trendline the "
            "triangle/wedge fit already produces"
        ),
    ),
    CatalogEntry(
        family_no=12,
        family="Trendline Patterns",
        pattern="Trendline Break",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: the five triangle/wedge "
            "detectors' `level` IS a fitted trendline's value at the latest bar, so a "
            "break of it is exactly what those events trigger on"
        ),
    ),
    CatalogEntry(
        family_no=12,
        family="Trendline Patterns",
        pattern="Channel Breakout",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: detector.donchian-breakout "
            "-- literally a channel breakout"
        ),
    ),
    CatalogEntry(
        family_no=12,
        family="Trendline Patterns",
        pattern="Trend Exhaustion",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: detector.rsi-divergence is "
            "the measurable form of exhaustion: a new price extreme that momentum does "
            "not confirm"
        ),
    ),
    CatalogEntry(
        family_no=13,
        family="Moving Average Patterns",
        pattern="Golden Cross",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: contract §9 rates Golden Cross three stars, below "
            "the tier 1-2 commitment, and no success criterion may depend on it"
        ),
    ),
    CatalogEntry(
        family_no=13,
        family="Moving Average Patterns",
        pattern="Death Cross",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "tier 3-4 by contract §9: the bearish mirror of Golden Cross and equally "
            "tier 3"
        ),
    ),
    CatalogEntry(
        family_no=13,
        family="Moving Average Patterns",
        pattern="EMA Bounce",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a moving average is a level, not an "
            "event; gating on it is Confirmation work"
        ),
    ),
    CatalogEntry(
        family_no=13,
        family="Moving Average Patterns",
        pattern="MA Compression",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a volatility reading, and "
            "detector.bollinger-fade plus indicators/bollinger.py already expose the "
            "squeeze form of it"
        ),
    ),
    CatalogEntry(
        family_no=13,
        family="Moving Average Patterns",
        pattern="MA Fan Expansion",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a trend-strength reading with no entry "
            "level; the regime classifier's ADX already serves this role"
        ),
    ),
    CatalogEntry(
        family_no=13,
        family="Moving Average Patterns",
        pattern="MA Flattening",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a trend-weakening reading with no level "
            "or target of its own"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="RSI",
        pattern="Bullish Divergence",
        tier=2,
        status="covered",
        detector_key="detector.rsi-divergence",
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="RSI",
        pattern="Bearish Divergence",
        tier=2,
        status="covered",
        detector_key="detector.rsi-divergence",
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="RSI",
        pattern="Hidden Bullish Divergence",
        tier=2,
        status="covered",
        detector_key="detector.rsi-divergence",
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="RSI",
        pattern="Hidden Bearish Divergence",
        tier=2,
        status="covered",
        detector_key="detector.rsi-divergence",
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="RSI",
        pattern="RSI > 70",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: an overbought reading has no level and no "
            "target, so it cannot become a PositionPlan; it is Confirmation material "
            "over indicators/rsi.py, which this phase does deliver"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="RSI",
        pattern="RSI < 30",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: the oversold mirror of RSI > 70, and "
            "Confirmation material for the same reason"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="MACD",
        pattern="Bullish Cross",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "owned by another phase: MACD Bullish Cross is detector.macd-cross, Phase "
            "4's module, and contract §9 rates MACD Cross three stars. (This row is the "
            "MACD subsection's; the Stochastic subsection has a row with identical "
            "text, which is why the ledger's uniqueness key is (family_no, "
            "subsection, pattern))"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="MACD",
        pattern="Bearish Cross",
        tier=3,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "owned by another phase: MACD Bearish Cross is the short side of "
            "detector.macd-cross, Phase 4's module"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="MACD",
        pattern="Histogram Increasing",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: confirmation.macd (Phase 4) already reads "
            "the price-normalised histogram; a histogram slope has no level and no "
            "target"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="MACD",
        pattern="Histogram Decreasing",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: the mirror of Histogram Increasing, and "
            "Confirmation material for the same reason"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="Stochastic",
        pattern="Bullish Cross",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a stochastic %K/%D cross has no level and "
            "no measured move, so it cannot become a PositionPlan on its own; and no "
            "indicators/stochastic.py exists in this repo to build it over"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="Stochastic",
        pattern="Bearish Cross",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: the short side of the stochastic cross, "
            "Confirmation material for the same reason, and equally without an "
            "indicators/stochastic.py to build over"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="Stochastic",
        pattern="Above 80",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a stochastic overbought reading: no "
            "level, no target, Confirmation material"
        ),
    ),
    CatalogEntry(
        family_no=14,
        family="Oscillator Signals",
        subsection="Stochastic",
        pattern="Below 20",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a stochastic oversold reading: no level, "
            "no target, Confirmation material"
        ),
    ),
    CatalogEntry(
        family_no=15,
        family="Fibonacci Patterns",
        pattern="38.2% Retracement Bounce",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: 38.2% Retracement Bounce names a price "
            "ZONE derived from a prior swing, not an event: it has no trigger bar and "
            "no direction on its own, so it is Confirmation / level material"
        ),
    ),
    CatalogEntry(
        family_no=15,
        family="Fibonacci Patterns",
        pattern="50% Retracement Bounce",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: 50% Retracement Bounce names a price ZONE "
            "derived from a prior swing, not an event: it has no trigger bar and no "
            "direction on its own, so it is Confirmation / level material"
        ),
    ),
    CatalogEntry(
        family_no=15,
        family="Fibonacci Patterns",
        pattern="61.8% Golden Pocket",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: 61.8% Golden Pocket names a price ZONE "
            "derived from a prior swing, not an event: it has no trigger bar and no "
            "direction on its own, so it is Confirmation / level material"
        ),
    ),
    CatalogEntry(
        family_no=15,
        family="Fibonacci Patterns",
        pattern="78.6% Retracement",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: 78.6% Retracement names a price ZONE "
            "derived from a prior swing, not an event: it has no trigger bar and no "
            "direction on its own, so it is Confirmation / level material"
        ),
    ),
    CatalogEntry(
        family_no=15,
        family="Fibonacci Patterns",
        pattern="127.2% Extension",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: 127.2% Extension names a price ZONE "
            "derived from a prior swing, not an event: it has no trigger bar and no "
            "direction on its own, so it is Confirmation / level material"
        ),
    ),
    CatalogEntry(
        family_no=15,
        family="Fibonacci Patterns",
        pattern="161.8% Extension",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: 161.8% Extension names a price ZONE "
            "derived from a prior swing, not an event: it has no trigger bar and no "
            "direction on its own, so it is Confirmation / level material"
        ),
    ),
    CatalogEntry(
        family_no=15,
        family="Fibonacci Patterns",
        pattern="Fib Cluster",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: Fib Cluster names a price ZONE derived "
            "from a prior swing, not an event: it has no trigger bar and no direction "
            "on its own, so it is Confirmation / level material"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Higher High (HH)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: Higher High (HH) is a market-STRUCTURE "
            "input, not a tradeable event: signals/pivots.py already produces the swing "
            "points it is computed from, and using it means gating some other "
            "detector's event on trend agreement -- Confirmation work"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Higher Low (HL)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: Higher Low (HL) is a market-STRUCTURE "
            "input, not a tradeable event: signals/pivots.py already produces the swing "
            "points it is computed from, and using it means gating some other "
            "detector's event on trend agreement -- Confirmation work"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Lower High (LH)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: Lower High (LH) is a market-STRUCTURE "
            "input, not a tradeable event: signals/pivots.py already produces the swing "
            "points it is computed from, and using it means gating some other "
            "detector's event on trend agreement -- Confirmation work"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Lower Low (LL)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: Lower Low (LL) is a market-STRUCTURE "
            "input, not a tradeable event: signals/pivots.py already produces the swing "
            "points it is computed from, and using it means gating some other "
            "detector's event on trend agreement -- Confirmation work"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Break of Structure (BoS)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: no reference "
            "implementation and no labelled data: which swing counts as 'the' structure "
            "is chosen by eye"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Change of Character (CHoCH)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: no reference "
            "implementation and no labelled data: the same swing-selection problem, "
            "plus a regime judgement"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Liquidity Sweep",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: no reference "
            "implementation and no labelled data: identical geometry to "
            "detector.wyckoff-spring but with an SMC narrative attached"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Equal Highs",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: no reference "
            "implementation and no labelled data: 'equal' needs a tolerance nobody "
            "agrees on, and the event is a liquidity claim rather than a price "
            "prediction"
        ),
    ),
    CatalogEntry(
        family_no=16,
        family="Market Structure",
        pattern="Equal Lows",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: no reference "
            "implementation and no labelled data: the mirror of Equal Highs, with the "
            "same undefined tolerance"
        ),
    ),
    CatalogEntry(
        family_no=17,
        family="Smart Money Concepts (SMC)",
        pattern="Order Block",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Order Block is a Smart "
            "Money Concepts construct: the PRD's Research Summary records SMC as having "
            "no mature reference implementation, and there is no labelled dataset, so "
            "any precision figure would be a measurement of the author's own drawing"
        ),
    ),
    CatalogEntry(
        family_no=17,
        family="Smart Money Concepts (SMC)",
        pattern="Fair Value Gap (FVG)",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Fair Value Gap (FVG) is "
            "a Smart Money Concepts construct: the PRD's Research Summary records SMC "
            "as having no mature reference implementation, and there is no labelled "
            "dataset, so any precision figure would be a measurement of the author's "
            "own drawing"
        ),
    ),
    CatalogEntry(
        family_no=17,
        family="Smart Money Concepts (SMC)",
        pattern="Breaker Block",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Breaker Block is a Smart "
            "Money Concepts construct: the PRD's Research Summary records SMC as having "
            "no mature reference implementation, and there is no labelled dataset, so "
            "any precision figure would be a measurement of the author's own drawing"
        ),
    ),
    CatalogEntry(
        family_no=17,
        family="Smart Money Concepts (SMC)",
        pattern="Mitigation Block",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Mitigation Block is a "
            "Smart Money Concepts construct: the PRD's Research Summary records SMC as "
            "having no mature reference implementation, and there is no labelled "
            "dataset, so any precision figure would be a measurement of the author's "
            "own drawing"
        ),
    ),
    CatalogEntry(
        family_no=17,
        family="Smart Money Concepts (SMC)",
        pattern="Liquidity Grab",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Liquidity Grab is a "
            "Smart Money Concepts construct: the PRD's Research Summary records SMC as "
            "having no mature reference implementation, and there is no labelled "
            "dataset, so any precision figure would be a measurement of the author's "
            "own drawing"
        ),
    ),
    CatalogEntry(
        family_no=17,
        family="Smart Money Concepts (SMC)",
        pattern="Inducement",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Inducement is a Smart "
            "Money Concepts construct: the PRD's Research Summary records SMC as having "
            "no mature reference implementation, and there is no labelled dataset, so "
            "any precision figure would be a measurement of the author's own drawing"
        ),
    ),
    CatalogEntry(
        family_no=17,
        family="Smart Money Concepts (SMC)",
        pattern="Premium Zone",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Premium Zone is a Smart "
            "Money Concepts construct: the PRD's Research Summary records SMC as having "
            "no mature reference implementation, and there is no labelled dataset, so "
            "any precision figure would be a measurement of the author's own drawing"
        ),
    ),
    CatalogEntry(
        family_no=17,
        family="Smart Money Concepts (SMC)",
        pattern="Discount Zone",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "no reference implementation and no labelled data: Discount Zone is a Smart "
            "Money Concepts construct: the PRD's Research Summary records SMC as having "
            "no mature reference implementation, and there is no labelled dataset, so "
            "any precision figure would be a measurement of the author's own drawing"
        ),
    ),
    CatalogEntry(
        family_no=18,
        family="Volatility Patterns",
        pattern="Bollinger Band Squeeze",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "already covered by another registered plug-in: indicators/bollinger.py "
            "plus detector.bollinger-fade already express band-width compression and "
            "the fade it sets up (kept reachable behind config.FADE_ENABLED)"
        ),
    ),
    CatalogEntry(
        family_no=18,
        family="Volatility Patterns",
        pattern="Bollinger Expansion",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: a volatility reading with no level or "
            "target; the regime classifier's ATR percentile covers the same ground"
        ),
    ),
    CatalogEntry(
        family_no=18,
        family="Volatility Patterns",
        pattern="ATR Expansion",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: indicators/wilder.py's ATR is already the "
            "regime classifier's input; expansion is a Confirmation, not an event"
        ),
    ),
    CatalogEntry(
        family_no=18,
        family="Volatility Patterns",
        pattern="ATR Compression",
        tier=None,
        status="out-of-scope",
        detector_key=None,
        reason=(
            "belongs to another plug-in kind: the mirror of ATR Expansion, and "
            "Confirmation material for the same reason"
        ),
    ),
)


def coverage_summary() -> dict:
    """Coverage counted by status, by tier and by family.

    Returns:
        dict with:
          `rows` / `families` — the ledger's own size,
          `by_status` — status -> count,
          `by_tier` — tier (int or None) -> {status -> count},
          `by_family` — family_no -> {"family", "rows", status counts},
          `detectors` — sorted distinct detector keys referenced,
          `tier1` / `tier2` — (covered, total) pairs for the two committed tiers.

    Pure: reads nothing but this module. `cli.py detector-report --coverage`
    calls it and must touch neither the database nor the trial ledger.
    """
    by_status: dict[str, int] = {s: 0 for s in STATUSES}
    by_tier: dict[int | None, dict[str, int]] = {}
    by_family: dict[int, dict] = {}
    for e in CATALOG:
        by_status[e.status] += 1
        by_tier.setdefault(e.tier, {s: 0 for s in STATUSES})[e.status] += 1
        fam = by_family.setdefault(
            e.family_no,
            {"family": e.family, "rows": 0, **{s: 0 for s in STATUSES}},
        )
        fam["rows"] += 1
        fam[e.status] += 1
    return {
        "rows": len(CATALOG),
        "families": len({e.family_no for e in CATALOG}),
        "by_status": by_status,
        "by_tier": by_tier,
        "by_family": by_family,
        "detectors": sorted(
            {e.detector_key for e in CATALOG if e.detector_key is not None}
        ),
        "tier1_rows": (
            by_tier.get(1, {}).get("covered", 0),
            sum(by_tier.get(1, {}).values()),
        ),
        "tier2_rows": (
            by_tier.get(2, {}).get("covered", 0),
            sum(by_tier.get(2, {}).values()),
        ),
        "section9": section9_scorecard(),
    }


def status_of(pattern: str) -> str:
    """The status of the single ledger row whose `pattern` text is ``pattern``.

    Raises:
        KeyError: when the text names no row, or more than one (the MACD and
            Stochastic subsections both contain "Bullish Cross", so those two
            must be looked up by (family_no, pattern) instead).
    """
    hits = [e for e in CATALOG if e.pattern == pattern]
    if len(hits) != 1:
        raise KeyError(
            f"{pattern!r} matches {len(hits)} ledger rows; ambiguous or unknown"
        )
    return hits[0].status


def section9_scorecard() -> dict:
    """Contract §9's tier commitments counted in CONCEPTS, not rows.

    A concept counts as covered only when EVERY row it names is covered — so
    "Double Top/Bottom" needs both rows, and "Wyckoff Accumulation/Distribution"
    is not covered while either row is deferred. That is what makes tier 1 read
    2 of 3 rather than being rounded up.
    """

    def score(concepts):
        detail = {}
        for name, patterns in concepts:
            detail[name] = (
                "covered"
                if all(status_of(p) == "covered" for p in patterns)
                else status_of(patterns[0])
            )
        covered = sum(1 for v in detail.values() if v == "covered")
        return {"covered": covered, "total": len(detail), "detail": detail}

    return {
        "tier1": score(SECTION9_TIER1_CONCEPTS),
        "tier2": score(SECTION9_TIER2_CONCEPTS),
    }


def format_coverage(summary: dict | None = None) -> str:
    """The `--coverage` report as a string. Sorted by family number, never by count."""
    s = summary or coverage_summary()
    st = s["by_status"]
    out = [
        f"catalog: {s['rows']} rows / {s['families']} families   "
        f"covered {st['covered']}  deferred {st['deferred']}  "
        f"out-of-scope {st['out-of-scope']}",
        f" contract §9 concepts -- tier 1: {s['section9']['tier1']['covered']}/"
        f"{s['section9']['tier1']['total']}  (Wyckoff Accumulation / Distribution "
        f"DEFERRED: unmeasurable, see catalog.py)"
        f"   tier 2: {s['section9']['tier2']['covered']}/"
        f"{s['section9']['tier2']['total']}",
        f" ledger rows carrying a tier   -- tier 1: {s['tier1_rows'][0]}/"
        f"{s['tier1_rows'][1]}   tier 2: {s['tier2_rows'][0]}/{s['tier2_rows'][1]}",
        "",
        f"{'#':>3} {'family':<32} {'rows':>5} {'cov':>4} {'def':>4} {'oos':>4}",
        "-" * 56,
    ]
    for fam_no in sorted(s["by_family"]):
        f = s["by_family"][fam_no]
        out.append(
            f"{fam_no:>3} {f['family'][:32]:<32} {f['rows']:>5} "
            f"{f['covered']:>4} {f['deferred']:>4} {f['out-of-scope']:>4}"
        )
    out.append("-" * 56)
    out.append(f"{len(s['detectors'])} detector key(s) referenced by covered rows")
    return "\n".join(out)
