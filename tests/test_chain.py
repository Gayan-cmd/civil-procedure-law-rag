from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import cpa.chain as chain_module
from cpa.chain import ask, build_chain, format_docs
from cpa.config import Settings
from cpa.store import build_vectorstore
from conftest import FakeEmbeddings


def test_format_docs_labels_each_source() -> None:
    docs = [
        Document(page_content="alpha text", metadata={"title": "Doc A"}),
        Document(page_content="beta text", metadata={"title": "Doc B"}),
    ]

    formatted = format_docs(docs)

    assert "[Source: Doc A]\nalpha text" in formatted
    assert "[Source: Doc B]\nbeta text" in formatted
    assert formatted.count("---") == 1


def test_format_docs_handles_missing_title() -> None:
    docs = [Document(page_content="text", metadata={})]

    formatted = format_docs(docs)

    assert "[Source: Unknown source]" in formatted


def _sample_vectorstore(tmp_path: Path, fake_embeddings: FakeEmbeddings):
    settings = Settings(chroma_dir=tmp_path / "chroma")
    documents = [
        Document(
            page_content="An answer must be filed within 21 days.",
            metadata={
                "slug": "cpc-consolidated",
                "content_role": "primary_consolidated_text",
                "act_number": "",
                "year": "",
            },
        ),
    ]
    return build_vectorstore(documents, fake_embeddings, settings)


def test_chain_retrieves_exactly_once_per_question(
    tmp_path: Path, fake_embeddings: FakeEmbeddings, monkeypatch
) -> None:
    settings = Settings(chroma_dir=tmp_path / "chroma", retrieval_k=5)
    vectorstore = _sample_vectorstore(tmp_path, fake_embeddings)

    call_count = 0
    real_get_retriever = chain_module.get_retriever

    def counting_get_retriever(vs, question, settings=None):
        nonlocal call_count
        call_count += 1
        return real_get_retriever(vs, question, settings)

    monkeypatch.setattr(chain_module, "get_retriever", counting_get_retriever)
    monkeypatch.setattr(
        chain_module,
        "get_llm",
        lambda settings=None: FakeListChatModel(responses=["fake answer"]),
    )

    chain = build_chain(vectorstore, settings)
    result = chain.invoke("What is the time limit for filing an answer?")

    assert call_count == 1
    assert result["answer"] == "fake answer"
    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["slug"] == "cpc-consolidated"


def test_ask_returns_answer_and_documents_together(
    tmp_path: Path, fake_embeddings: FakeEmbeddings, monkeypatch
) -> None:
    settings = Settings(chroma_dir=tmp_path / "chroma", retrieval_k=5)
    vectorstore = _sample_vectorstore(tmp_path, fake_embeddings)

    monkeypatch.setattr(
        chain_module,
        "get_llm",
        lambda settings=None: FakeListChatModel(responses=["21 days"]),
    )

    result = ask(vectorstore, "What is the time limit?", settings)

    assert result["answer"] == "21 days"
    assert len(result["documents"]) == 1
