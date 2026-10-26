from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cpa.config import Settings
from cpa.generation import get_prompt
from cpa.store import build_vectorstore
from conftest import FakeEmbeddings
from eval.ablation import (
    evaluate_config,
    load_existing_baseline,
    naive_dense_retrieve,
    summarize,
)


def test_naive_dense_retrieve_applies_no_filtering_at_all(
    fake_embeddings: FakeEmbeddings,
) -> None:
    settings = Settings()
    chunks = [
        Document(
            page_content="current in-force text",
            metadata={
                "slug": "cpc-consolidated",
                "content_role": "primary_consolidated_text",
            },
        ),
        Document(
            page_content="amending instrument text",
            metadata={
                "slug": "cpc-amend-43-2024",
                "content_role": "amending_instrument",
            },
        ),
    ]
    vectorstore = build_vectorstore(chunks, fake_embeddings, settings)

    # Unlike get_retriever, this must be able to return BOTH documents --
    # proving there's genuinely no content_role filter applied.
    results = naive_dense_retrieve(vectorstore, "any question", k=2)

    slugs = {doc.metadata["slug"] for doc in results}
    assert slugs == {"cpc-consolidated", "cpc-amend-43-2024"}


def test_summarize_computes_recall_and_refusal_separately() -> None:
    per_question = [
        {
            "id": "q1",
            "answerable": True,
            "expected_doc": "docA",
            "retrieved_slugs": ["docA", "docB"],
            "retrieval_latency": 0.1,
            "answer": None,
        },
        {
            "id": "q2",
            "answerable": True,
            "expected_doc": "docA",
            "retrieved_slugs": ["docC"],
            "retrieval_latency": 0.1,
            "answer": None,
        },
        {
            "id": "q3",
            "answerable": False,
            "expected_doc": None,
            "retrieved_slugs": [],
            "retrieval_latency": 0.1,
            "answer": "I don't have that information in my knowledge base.",
        },
    ]

    summary = summarize("test-config", per_question, k=5)

    assert summary["recall_at_k"] == 0.5  # 1 of 2 answerable hit
    assert summary["refusal_accuracy"] == 1.0
    assert summary["name"] == "test-config"


def test_load_existing_baseline_matches_summarize(tmp_path: Path) -> None:
    import json

    fake_results = [
        {
            "id": "q1",
            "answerable": True,
            "expected_doc": "docA",
            "retrieved_slugs": ["docA"],
            "retrieval_latency": 0.2,
            "answer": "some answer",
        },
    ]
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(fake_results), encoding="utf-8")

    summary = load_existing_baseline("+ Act-number filtering", path, k=5)

    assert summary["recall_at_k"] == 1.0
    assert summary["name"] == "+ Act-number filtering"


def test_evaluate_config_only_generates_for_unanswerable_questions() -> None:
    call_count = 0
    real_llm = FakeListChatModel(responses=["fake refusal"])

    class _CountingLLM:
        def invoke(self, formatted):
            nonlocal call_count
            call_count += 1
            return real_llm.invoke(formatted)

    doc = Document(page_content="text", metadata={"slug": "docA"})
    gold_set = [
        {"id": "q1", "answerable": True, "expected_doc": "docA", "question": "q1?"},
        {"id": "q2", "answerable": True, "expected_doc": "docA", "question": "q2?"},
        {"id": "q3", "answerable": False, "expected_doc": None, "question": "q3?"},
    ]

    summary = evaluate_config(
        "test",
        lambda q: [doc],
        gold_set,
        get_prompt(),
        _CountingLLM(),
        k=5,
    )

    # Only 1 of the 3 questions is unanswerable -- generation should run
    # exactly once, not three times.
    assert call_count == 1
    assert summary["recall_at_k"] == 1.0
