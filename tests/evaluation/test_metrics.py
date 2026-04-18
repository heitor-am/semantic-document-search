from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank


class TestRecallAtK:
    def test_all_relevant_in_top_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_half_relevant_in_top_k(self) -> None:
        assert recall_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_none_relevant_in_top_k(self) -> None:
        assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_relevant_outside_top_k_is_missed(self) -> None:
        # b is at position 3 but k=2 — misses it
        assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5

    def test_empty_relevant_set_is_vacuously_1(self) -> None:
        # No relevant docs → recall is 1.0 (nothing to miss)
        assert recall_at_k(["a", "b"], [], k=2) == 1.0

    def test_zero_k_raises(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k(["a"], ["a"], k=0)


class TestReciprocalRank:
    def test_first_result_relevant(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_result_relevant(self) -> None:
        assert reciprocal_rank(["x", "a", "y"], ["a"]) == 0.5

    def test_third_result_relevant(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_relevant_in_retrieved(self) -> None:
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_earliest_relevant_wins(self) -> None:
        # Both a and b are relevant, but b is at a better rank
        assert reciprocal_rank(["c", "b", "a"], ["a", "b"]) == 0.5


class TestNdcgAtK:
    def test_perfect_ordering_is_1(self) -> None:
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_any_permutation_of_relevant_items_is_bounded(self) -> None:
        # With binary relevance and 2 relevant items packed in the top 2
        # slots, any permutation yields the same DCG as the ideal one.
        # Verifying the bound and full recovery, not "reversed is worse".
        assert ndcg_at_k(["b", "a"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_irrelevant_dilutes(self) -> None:
        perfect = ndcg_at_k(["a", "b"], ["a", "b"], k=2)
        diluted = ndcg_at_k(["a", "x"], ["a", "b"], k=2)
        assert diluted < perfect

    def test_no_relevant_retrieved_is_0(self) -> None:
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_empty_relevant_is_1(self) -> None:
        assert ndcg_at_k(["a"], [], k=1) == 1.0

    def test_zero_k_raises(self) -> None:
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], ["a"], k=0)

    def test_matches_known_value(self) -> None:
        # retrieved=[a, x, b], relevant={a, b}, k=3
        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG: 2 relevant packed at top → 1/log2(2) + 1/log2(3) = 1 + 0.6309
        score = ndcg_at_k(["a", "x", "b"], ["a", "b"], k=3)
        expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        assert score == pytest.approx(expected)


class TestMean:
    def test_nonempty(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty_is_0(self) -> None:
        assert mean([]) == 0.0

    def test_generator_input(self) -> None:
        assert mean(x for x in [1.0, 2.0]) == 1.5
