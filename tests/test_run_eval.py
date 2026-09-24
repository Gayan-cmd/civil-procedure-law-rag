from __future__ import annotations

from cpa.config import Settings
from eval.run_eval import print_table


def _fake_result(
    category: str,
    answerable: bool,
    expected_doc: str | None,
    retrieved_slugs: list[str],
    answer: str,
) -> dict:
    return {
        "id": "qX",
        "category": category,
        "answerable": answerable,
        "expected_doc": expected_doc,
        "retrieved_slugs": retrieved_slugs,
        "answer": answer,
        "retrieval_latency": 0.1,
        "generation_latency": 0.5,
    }


def test_print_table_computes_recall_over_answerable_only(capsys) -> None:
    results = [
        _fake_result("basic_lookup", True, "docA", ["docA", "docB"], "an answer"),
        _fake_result("basic_lookup", True, "docA", ["docC", "docD"], "an answer"),
        _fake_result(
            "unanswerable",
            False,
            None,
            [],
            "I don't have that information in my knowledge base.",
        ),
    ]

    recall, refusal_acc = print_table(results, Settings(), k=5)

    # 1 hit out of 2 answerable questions
    assert recall == 0.5
    # the one unanswerable question was correctly refused
    assert refusal_acc == 1.0


def test_print_table_refusal_accuracy_penalizes_fabricated_answers(capsys) -> None:
    results = [
        _fake_result(
            "unanswerable", False, None, [], "The penalty is five years imprisonment."
        ),
        _fake_result(
            "unanswerable",
            False,
            None,
            [],
            "I don't have that information in my knowledge base.",
        ),
    ]

    _, refusal_acc = print_table(results, Settings(), k=5)

    assert refusal_acc == 0.5


def test_print_table_prints_a_table(capsys) -> None:
    results = [
        _fake_result("basic_lookup", True, "docA", ["docA"], "an answer"),
    ]

    print_table(results, Settings(), k=5)

    captured = capsys.readouterr()
    assert "Recall@5" in captured.out
    assert "MRR" in captured.out
    assert "Refusal accuracy" in captured.out
