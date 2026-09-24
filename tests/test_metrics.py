from __future__ import annotations

import pytest

from eval.metrics import (
    is_refusal,
    mean_reciprocal_rank,
    mean_recall_at_k,
    p50,
    p95,
    percentile,
    recall_at_k,
    reciprocal_rank,
    refusal_accuracy,
)


def test_recall_at_k_true_when_expected_doc_in_top_k() -> None:
    retrieved = ["docB", "docA", "docC", "docD", "docE"]
    assert recall_at_k(retrieved, "docA", k=5) is True


def test_recall_at_k_false_when_expected_doc_missing() -> None:
    retrieved = ["docB", "docC", "docD", "docE", "docF"]
    assert recall_at_k(retrieved, "docA", k=5) is False


def test_recall_at_k_false_when_expected_doc_beyond_k() -> None:
    retrieved = ["docB", "docC", "docD", "docE", "docA"]
    assert recall_at_k(retrieved, "docA", k=3) is False


def test_mean_recall_at_k_averages_across_questions() -> None:
    results = [
        (["docA", "docB"], "docA"),  # hit
        (["docC", "docD"], "docA"),  # miss
        (["docA", "docE"], "docA"),  # hit
        (["docF", "docG"], "docA"),  # miss
    ]
    assert mean_recall_at_k(results, k=5) == 0.5


def test_mean_recall_at_k_empty_results_is_zero() -> None:
    assert mean_recall_at_k([], k=5) == 0.0


def test_reciprocal_rank_matches_build_plan_worked_examples() -> None:
    # BUILD_PLAN.md: "right chunk ranked 1st scores 1.0, 2nd scores 0.5,
    # 3rd scores 0.33"
    assert reciprocal_rank(["docA"], "docA") == pytest.approx(1.0)
    assert reciprocal_rank(["docX", "docA"], "docA") == pytest.approx(0.5)
    assert reciprocal_rank(["docX", "docY", "docA"], "docA") == pytest.approx(
        0.333, abs=0.01
    )


def test_reciprocal_rank_zero_when_not_found_in_top_k() -> None:
    assert reciprocal_rank(["docX", "docY"], "docA", k=5) == 0.0


def test_mean_reciprocal_rank_averages_across_questions() -> None:
    results = [
        (["docA"], "docA"),  # rank 1 -> 1.0
        (["docX", "docA"], "docA"),  # rank 2 -> 0.5
    ]
    assert mean_reciprocal_rank(results) == pytest.approx(0.75)


def test_is_refusal_detects_the_exact_prompt_phrase() -> None:
    assert is_refusal("I don't have that information in my knowledge base.")


def test_is_refusal_case_insensitive_and_substring() -> None:
    answer = "Sorry -- i don't have that information in my knowledge base. Try rephrasing."
    assert is_refusal(answer)


def test_is_refusal_false_on_a_real_answer() -> None:
    answer = "Under section 75, the answer must set out the defendant's case."
    assert not is_refusal(answer)


def test_refusal_accuracy_counts_correct_refusals() -> None:
    answers = [
        "I don't have that information in my knowledge base.",
        "The Penal Code prescribes a fine or imprisonment.",  # fabricated, not refused
        "I don't have that information in my knowledge base.",
    ]
    assert refusal_accuracy(answers) == pytest.approx(2 / 3)


def test_percentile_nearest_rank_on_known_values() -> None:
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 100


def test_percentile_single_value() -> None:
    assert percentile([42], 50) == 42
    assert percentile([42], 95) == 42


def test_percentile_empty_raises() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_p50_and_p95_are_percentile_shortcuts() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert p50(values) == percentile(values, 50)
    assert p95(values) == percentile(values, 95)
