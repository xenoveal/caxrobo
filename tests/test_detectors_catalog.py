"""
Ledger invariants for plugins/detectors/catalog.py.

WITHOUT THIS MODULE, catalog.py IS A COMMENT. Invariant 4 in particular is the
only thing that makes coverage TRACKED rather than claimed: it parses
`.claude/technical-pattern.md` — the document that DEFINES the catalog — and
asserts the ledger's pattern set equals the document's, row for row.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

from trading_bot.framework import registry
from trading_bot.plugins.detectors.catalog import (
    CATALOG,
    CATALOG_FAMILY_COUNT,
    CATALOG_ROW_COUNT,
    OUT_OF_SCOPE_REASONS,
    STATUSES,
    coverage_summary,
    format_coverage,
    section9_scorecard,
)

CATALOG_DOC = Path(__file__).resolve().parents[1] / ".claude" / "technical-pattern.md"

# Detector modules owned by earlier phases. Excluded BY NAME from invariant 2:
# they are legitimately absent from a catalog that enumerates Phase 8's coverage.
NON_PHASE8_MODULE_LEAVES = ("legacy_patterns", "donchian", "bollinger_fade", "macd_cross")


def _parse_catalog_document() -> list[tuple[int, str, str]]:
    """(family_no, subsection, left-column text) for every row of the 18 NUMBERED
    sections.

    Skips the `# Highest Historical Reliability` table: including its 15 rows
    gives 159 instead of 144, which is exactly the drift this test exists to
    catch. Anchoring on the `# <n>. ` headings is what excludes it — the
    reliability table's heading carries no number.

    Rows containing parentheses (`Rounded Bottom (Saucer)`,
    `Runaway (Measuring) Gap`) are compared as EXACT literals; only surrounding
    pipes and whitespace are stripped.
    """
    rows: list[tuple[int, str, str]] = []
    current: int | None = None
    subsection = ""
    for line in CATALOG_DOC.read_text().splitlines():
        heading = re.match(r"^# (\d+)\.\s+(.*)$", line.strip())
        if heading:
            current = int(heading.group(1))
            subsection = ""
            continue
        if line.startswith("# ") and not re.match(r"^# \d+\.", line):
            current = None  # e.g. "# Highest Historical Reliability"
            continue
        sub = re.match(r"^## (.+)$", line.strip())
        if sub:
            subsection = sub.group(1).strip()
            continue
        if current is None:
            continue
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2:
            continue
        left = cells[0]
        if left.lower() in ("pattern", "signal", "structure"):
            continue  # header row
        if set(left) <= set("-: "):
            continue  # separator row
        rows.append((current, subsection, left))
    return rows


class TestCatalogLedger:
    def test_every_covered_entry_has_a_registered_detector(self):
        registry.load_all()
        for e in CATALOG:
            if e.status != "covered":
                continue
            assert e.detector_key, e.pattern
            assert e.detector_key in registry.REGISTRY, (e.pattern, e.detector_key)

    def test_every_phase8_detector_appears_in_the_catalog(self):
        """No orphan detectors: a detector nobody can find in the ledger is
        coverage that is not tracked."""
        registry.load_all()
        keys = set()
        for spec in registry.REGISTRY.values():
            if spec.kind != "detector":
                continue
            if not spec.module.startswith("trading_bot.plugins.detectors."):
                continue
            if spec.module.rsplit(".", 1)[-1] in NON_PHASE8_MODULE_LEAVES:
                continue
            keys.add(spec.key)
        listed = {e.detector_key for e in CATALOG if e.detector_key}
        assert keys - listed == set(), f"orphan detector(s): {sorted(keys - listed)}"
        assert listed - keys == set(), f"ledger names unregistered key(s): {sorted(listed - keys)}"

    def test_non_covered_entries_state_a_reason(self):
        reasons = []
        for e in CATALOG:
            if e.status == "covered":
                assert not e.reason, e.pattern
                continue
            assert len(e.reason) >= 20, (e.pattern, e.reason)
            reasons.append(e.reason)
            if e.status == "out-of-scope":
                assert any(
                    e.reason.startswith(r) for r in OUT_OF_SCOPE_REASONS
                ), (e.pattern, e.reason[:60])
        # A blanket copy-paste would defeat the whole point of enumerating.
        worst = Counter(reasons).most_common(1)[0]
        assert worst[1] <= 40, f"reason repeated {worst[1]} times: {worst[0][:80]}"

    def test_catalog_matches_the_source_document(self):
        """THE INVARIANT THAT MAKES COVERAGE TRACKED RATHER THAN CLAIMED.

        The catalog document IS the specification, so this test deliberately
        reads a file outside the package. A source distribution has no
        `.claude/`, hence the skip.
        """
        if not CATALOG_DOC.exists():
            pytest.skip(
                f"{CATALOG_DOC} is absent (a source distribution carries no "
                f".claude/); the ledger cannot be checked against its spec"
            )
        doc_rows = _parse_catalog_document()
        assert len(doc_rows) == CATALOG_ROW_COUNT == 144, len(doc_rows)
        assert len({f for f, _, _ in doc_rows}) == CATALOG_FAMILY_COUNT == 18
        assert len(CATALOG) == 144
        assert {p for _, _, p in doc_rows} == {e.pattern for e in CATALOG}
        # (family, subsection, pattern) triples, so the MACD and Stochastic
        # subsections' shared "Bullish Cross" / "Bearish Cross" rows are checked
        # individually rather than collapsing into one another.
        assert sorted(doc_rows) == sorted(
            (e.family_no, e.subsection, e.pattern) for e in CATALOG
        )

    def test_tier_scorecard(self):
        """PINS SCOPE DECISION A1 against silent drift.

        Contract §9's tiers are stated as CONCEPTS, so they are counted as
        concepts: tier 1 lands 2 of 3 and tier 2 lands 6 of 6. If someone later
        covers Wyckoff, this test tells them to update A1 rather than letting the
        scorecard quietly improve.
        """
        s = section9_scorecard()
        assert s["tier2"] == {
            "covered": 6,
            "total": 6,
            "detail": {
                "Double Top/Bottom": "covered",
                "Bull/Bear Flag": "covered",
                "Ascending Triangle": "covered",
                "Falling Wedge": "covered",
                "Rising Wedge": "covered",
                "RSI Divergence": "covered",
            },
        }
        assert s["tier1"]["covered"] == 2
        assert s["tier1"]["total"] == 3
        assert s["tier1"]["detail"]["Cup & Handle"] == "covered"
        assert s["tier1"]["detail"]["Head & Shoulders"] == "covered"
        assert s["tier1"]["detail"]["Wyckoff Accumulation/Distribution"] == "deferred"


class TestCatalogStructure:
    def test_counts_are_exactly_the_committed_ledger(self):
        st = coverage_summary()["by_status"]
        assert st == {"covered": 19, "deferred": 6, "out-of-scope": 119}
        assert sum(st.values()) == 144

    def test_family_subsection_and_pattern_triples_are_unique(self):
        """(family_no, pattern) is NOT unique — the MACD and Stochastic
        subsections each carry a "Bullish Cross" and a "Bearish Cross" row — so
        the subsection is part of the key, not decoration."""
        triples = [(e.family_no, e.subsection, e.pattern) for e in CATALOG]
        assert len(triples) == len(set(triples)) == 144
        pairs = {(e.family_no, e.pattern) for e in CATALOG}
        assert len(pairs) == 142  # the two colliding cross rows

    def test_only_family_14_has_subsections(self):
        for e in CATALOG:
            if e.family_no == 14:
                assert e.subsection in ("RSI", "MACD", "Stochastic"), e.pattern
            else:
                assert e.subsection == "", e.pattern

    def test_detector_key_is_deliberately_not_unique(self):
        """rsi-divergence serves four rows; the uniqueness invariant runs the
        other way (family+pattern)."""
        keys = [e.detector_key for e in CATALOG if e.detector_key]
        assert Counter(keys)["detector.rsi-divergence"] == 4

    def test_statuses_and_tiers_are_legal(self):
        for e in CATALOG:
            assert e.status in STATUSES
            assert e.tier is None or e.tier in (1, 2, 3, 4)
            assert e.family_no in range(1, 19)
            assert e.pattern.strip() == e.pattern

    def test_all_five_out_of_scope_reason_categories_are_used(self):
        """The five categories are distinct and load-bearing; an unused one would
        mean the taxonomy is decoration."""
        used = set()
        for e in CATALOG:
            if e.status != "out-of-scope":
                continue
            for r in OUT_OF_SCOPE_REASONS:
                if e.reason.startswith(r):
                    used.add(r)
        assert used == set(OUT_OF_SCOPE_REASONS)

    def test_families_match_the_document_headings(self):
        by_no = {}
        for e in CATALOG:
            by_no.setdefault(e.family_no, set()).add(e.family)
        for no, names in by_no.items():
            assert len(names) == 1, (no, names)

    def test_format_coverage_reports_both_row_and_concept_counts(self):
        text = format_coverage()
        assert "144 rows / 18 families" in text
        assert "covered 19" in text
        assert "deferred 6" in text
        assert "out-of-scope 119" in text
        assert "tier 1: 2/3" in text
        assert "tier 2: 6/6" in text

    def test_coverage_summary_is_pure(self):
        """`--coverage` must touch neither the OHLCV database nor the ledger; the
        function it calls reads nothing but this module."""
        first = coverage_summary()
        assert coverage_summary() == first
