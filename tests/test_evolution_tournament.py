"""
Tiering, ranking and selection (v0.3.0 Phase 6).

Every gate dict here is built from walkforward.GATE_CONDITIONS rather than from
literal strings, so a Phase 1 rename fails loudly instead of silently mis-tiering
every candidate in every campaign.
"""

import math
import random

import pytest

from trading_bot import config
from trading_bot.backtest.walkforward import GATE_CONDITIONS
from trading_bot.evolution import oracle, tournament


def _gate(**overrides) -> dict:
    """A full gate dict keyed by GATE_CONDITIONS, all True unless overridden."""
    verdicts = dict.fromkeys(GATE_CONDITIONS, True)
    unknown = set(overrides) - set(GATE_CONDITIONS)
    assert not unknown, f"not a gate condition: {unknown}"
    verdicts.update(overrides)
    return verdicts


def _result(*, sharpe=1.5, bench_sharpe=0.7, ann=0.5, bench_ann=0.29,
            n_trades=42, dsr=0.01, gate=None, error="") -> oracle.OracleResult:
    return oracle.OracleResult(
        graph_hash="h", params_hash="p",
        window_start_ms=1, window_end_ms=2,
        oos_start_ms=1, oos_end_ms=2,
        n_trials_used=1, n_trades=n_trades,
        sharpe=sharpe, dsr=dsr, ann_return_pct=ann, max_drawdown_pct=0.1,
        bench_sharpe=bench_sharpe, bench_ann_return_pct=bench_ann,
        gate=_gate() if gate is None else gate,
        passed=all((gate or _gate()).values()),
        eval_seconds=0.1, error=error,
    )


class TestTiers:
    def test_tier_a_is_every_condition_passing(self):
        f = tournament.score_member(_result(gate=_gate()))
        assert f.tier == "A"

    def test_tier_b_is_both_benchmark_conditions_passing_but_not_all(self):
        f = tournament.score_member(
            _result(gate=_gate(dsr=False, sample_adequacy=False))
        )
        assert f.tier == "B"

    @pytest.mark.parametrize(
        "failing", ["beats_benchmark_return", "beats_benchmark_sharpe"]
    )
    def test_tier_c_is_a_benchmark_failure(self, failing):
        f = tournament.score_member(_result(gate=_gate(**{failing: False})))
        assert f.tier == "C"

    def test_tier_d_on_undefined_sharpe(self):
        assert tournament.score_member(_result(sharpe=None)).tier == "D"

    def test_tier_d_on_zero_trades(self):
        assert tournament.score_member(_result(n_trades=0)).tier == "D"

    def test_tier_d_on_an_evaluation_error(self):
        assert tournament.score_member(_result(error="span too short")).tier == "D"

    def test_tier_d_on_a_short_gate_dict(self, caplog):
        """A missing condition means Phase 1 renamed one, or the oracle returned an
        error shape without setting `error`. Either is a bug worth a loud tier D
        rather than a silent mis-tiering."""
        import logging

        partial = {k: True for k in GATE_CONDITIONS[:3]}
        with caplog.at_level(logging.WARNING, logger="trading_bot"):
            f = tournament.score_member(_result(gate=partial))
        assert f.tier == "D"
        assert "missing condition" in caplog.text

    def test_tiering_reads_the_gate_and_re_derives_nothing(self):
        """Two implementations of one condition is how they drift: a result whose
        METRICS look great but whose gate says the benchmark beat it is tier C."""
        f = tournament.score_member(
            _result(sharpe=9.0, bench_sharpe=0.1,
                    gate=_gate(beats_benchmark_sharpe=False))
        )
        assert f.tier == "C"


class TestFitness:
    def test_score_is_excess_sharpe_never_dsr(self):
        """A2: dsr moves with n_trials, so a dsr fitness would make an UNCHANGED
        candidate look worse in generation 30 than in generation 1."""
        a = tournament.score_member(_result(sharpe=1.5, bench_sharpe=0.7, dsr=0.9))
        b = tournament.score_member(_result(sharpe=1.5, bench_sharpe=0.7, dsr=0.001))
        assert a.score == b.score == pytest.approx(0.8)
        assert a.excess_sharpe == pytest.approx(0.8)

    def test_tier_d_scores_negative_infinity(self):
        assert tournament.score_member(_result(sharpe=None)).score == -math.inf

    def test_a_missing_benchmark_is_not_treated_as_zero(self):
        """'The null could not be computed' is not 'the null returned nothing'."""
        f = tournament.score_member(_result(bench_sharpe=None))
        assert f.excess_sharpe is None
        assert f.score == -math.inf

    def test_excess_can_be_negative_and_is_still_a_real_score(self):
        f = tournament.score_member(
            _result(sharpe=0.1, bench_sharpe=0.7,
                    gate=_gate(beats_benchmark_sharpe=False))
        )
        assert f.tier == "C"
        assert f.score == pytest.approx(-0.6)
        assert f.score > -math.inf


class TestRanking:
    def test_tier_dominates_score(self):
        """The gate is the oracle; score only orders WITHIN a tier."""
        weak_a = tournament.score_member(_result(sharpe=0.75, bench_sharpe=0.7))
        strong_b = tournament.score_member(
            _result(sharpe=5.0, bench_sharpe=0.7, gate=_gate(dsr=False))
        )
        assert tournament.rank_key(weak_a, 0) > tournament.rank_key(strong_b, 1)

    def test_ranking_is_a_total_order_under_1000_shuffles(self):
        """Pins walkforward.py:60-63's precedent: ten of twelve combos produced
        identical trade lists and fold winners were decided by
        itertools.product ordering over tied expectancies. Under a process pool
        the same bug would make SELECTION depend on machine speed."""
        tied = [
            (i, tournament.score_member(_result(sharpe=1.5, bench_sharpe=0.7)), f"m{i}")
            for i in range(12)
        ]
        expected = [item[0] for item in tournament.ranked(tied)]
        rng = random.Random(0)
        for _ in range(1000):
            shuffled = list(tied)
            rng.shuffle(shuffled)
            assert [item[0] for item in tournament.ranked(shuffled)] == expected
        assert expected == list(range(12)), "lower member_index must win a tie"

    def test_secondary_keys_break_ties_before_the_index(self):
        lower_ann = tournament.score_member(
            _result(sharpe=1.5, bench_sharpe=0.7, ann=0.30, bench_ann=0.29)
        )
        higher_ann = tournament.score_member(
            _result(sharpe=1.5, bench_sharpe=0.7, ann=0.90, bench_ann=0.29)
        )
        # index 5 loses to index 0 on nothing else, but wins on excess_ann_return.
        assert tournament.rank_key(higher_ann, 5) > tournament.rank_key(lower_ann, 0)

    def test_a_none_excess_ann_return_sorts_last_not_first(self):
        defined = tournament.score_member(
            _result(sharpe=1.5, bench_sharpe=0.7, ann=0.5, bench_ann=0.29)
        )
        missing = tournament.score_member(
            _result(sharpe=1.5, bench_sharpe=0.7, ann=None, bench_ann=0.29)
        )
        assert tournament.rank_key(defined, 9) > tournament.rank_key(missing, 0)


class TestSelection:
    def _population(self, tiers):
        out = []
        for i, tier in enumerate(tiers):
            gate = {
                "A": _gate(),
                "B": _gate(dsr=False),
                "C": _gate(beats_benchmark_return=False),
            }.get(tier)
            res = (
                _result(sharpe=None) if tier == "D"
                else _result(sharpe=1.0 + i * 0.1, gate=gate)
            )
            out.append((i, tournament.score_member(res), f"m{i}"))
        return out

    def test_tier_c_is_selectable(self):
        """A4: the v0.2.0 seed itself fails beats_benchmark_* (+3.45% annualized vs
        the basket's +29.4%), so eliminating tier C empties generation 0."""
        pop = self._population(["C"] * 6)
        assert all(f.tier == "C" for _, f, _ in pop)
        parents = tournament.select_parents(pop, random.Random(0), k=3, n=6)
        assert len(parents) == 6
        assert all(p[1].tier == "C" for p in parents)

    def test_selection_is_reproducible_from_the_seed(self):
        pop = self._population(["A", "B", "C", "C", "B", "A"])
        a = tournament.select_parents(pop, random.Random(7), k=3, n=10)
        b = tournament.select_parents(pop, random.Random(7), k=3, n=10)
        assert [p[0] for p in a] == [p[0] for p in b]

    def test_sampling_is_with_replacement(self):
        """rng.choices, not rng.sample: with replacement IS the intended selection
        pressure, and rng.sample would also fail once k exceeds the population."""
        pop = self._population(["A", "D", "D", "D"])
        parents = tournament.select_parents(pop, random.Random(1), k=2, n=8)
        assert len(parents) == 8
        assert len({p[0] for p in parents}) < 8

    def test_k_is_clamped_to_the_population(self):
        pop = self._population(["A", "B"])
        parents = tournament.select_parents(pop, random.Random(1), k=99, n=3)
        assert len(parents) == 3

    def test_an_empty_population_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            tournament.select_parents([], random.Random(0), k=3, n=2)

    def test_elites_are_the_ranked_head(self):
        pop = self._population(["C", "A", "B", "A"])
        top = tournament.elites(pop, n=2)
        assert [t[1].tier for t in top] == ["A", "A"]
        assert [t[0] for t in top] == [3, 1], "higher score first within a tier"

    def test_zero_elites_is_legal(self):
        assert tournament.elites(self._population(["A"]), n=0) == []

    def test_elites_default_to_the_config_value(self):
        pop = self._population(["A"] * 5)
        assert len(tournament.elites(pop)) == config.EVO_ELITES

    def test_an_all_tier_d_population_is_detected_not_crashed(self):
        pop = self._population(["D"] * 5)
        assert tournament.all_tier_d(pop)
        parents = tournament.select_parents(pop, random.Random(0), k=3, n=5)
        assert len(parents) == 5

    def test_all_tier_d_is_false_for_a_mixed_population(self):
        assert not tournament.all_tier_d(self._population(["D", "C"]))
        assert not tournament.all_tier_d([])


class TestDiversity:
    def test_unique_fraction(self):
        assert tournament.unique_fraction(["a", "b", "c"]) == 1.0
        assert tournament.unique_fraction(["a", "a", "b", "b"]) == 0.5
        assert tournament.unique_fraction([]) == 1.0

    def test_the_guard_fires_below_the_configured_minimum(self):
        assert tournament.diversity_guard_needed(["a"] * 10)
        assert not tournament.diversity_guard_needed([str(i) for i in range(10)])

    def test_the_guard_never_fires_on_an_empty_generation(self):
        assert not tournament.diversity_guard_needed([])

    def test_the_guard_uses_the_config_threshold(self):
        hashes = ["a", "a", "b", "c"]  # 0.75 unique
        assert not tournament.diversity_guard_needed(hashes, minimum=0.5)
        assert tournament.diversity_guard_needed(hashes, minimum=0.8)
        assert config.EVO_MIN_UNIQUE_FRACTION == 0.5


class TestNoThresholdDuplication:
    """Contract §12: no gate threshold is re-typed here.

    Tiering must read Phase 1's `gate` dict, so this module has no business
    knowing GATE_MIN_SHARPE, GATE_MIN_DSR or GATE_MAX_DRAWDOWN.
    """

    def test_no_gate_threshold_appears_in_this_module(self):
        import pathlib

        text = pathlib.Path(tournament.__file__).read_text(encoding="utf-8")
        for name in ("GATE_MIN_SHARPE", "GATE_MIN_DSR", "GATE_MAX_DRAWDOWN"):
            assert name not in text, f"{name} is duplicated into tournament.py"

    def test_the_benchmark_conditions_are_taken_from_phase_1s_tuple(self):
        assert set(tournament._BENCH_CONDITIONS) <= set(GATE_CONDITIONS)
