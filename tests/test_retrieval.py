from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from cpa.config import Settings
from cpa.retrieval import extract_act_reference, get_retriever
from cpa.store import build_vectorstore
from conftest import FakeEmbeddings


def test_extracts_full_citation() -> None:
    assert extract_act_reference("Act No. 43 of 2024") == ("43", 2024)


def test_returns_none_on_garbage() -> None:
    assert extract_act_reference("hello world") == (None, None)


def test_handles_slash_format() -> None:
    assert extract_act_reference("Act 43/2024") == ("43", 2024)


def _sample_vectorstore(tmp_path: Path, fake_embeddings: FakeEmbeddings):
    settings = Settings(chroma_dir=tmp_path / "chroma")
    documents = [
        Document(
            page_content="The consolidated code: an answer must be filed"
            " within a set period.",
            metadata={
                "slug": "cpc-consolidated-lankalaw",
                "doc_type": "principal_act",
                "act_number": "",
                "year": "",
                "content_role": "primary_consolidated_text",
                "superseded_by": "",
            },
        ),
        Document(
            page_content="Act No. 43 of 2024: in section 46, substitute"
            " the following.",
            metadata={
                "slug": "cpc-amend-43-2024",
                "doc_type": "amendment_act",
                "act_number": "43",
                "year": 2024,
                "content_role": "amending_instrument",
                "superseded_by": "cpc-consolidated-lankalaw",
            },
        ),
    ]
    return build_vectorstore(documents, fake_embeddings, settings)


def test_act_specific_query_returns_amending_instrument(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    vectorstore = _sample_vectorstore(tmp_path, fake_embeddings)
    settings = Settings(chroma_dir=tmp_path / "chroma", retrieval_k=5)

    retriever = get_retriever(
        vectorstore, "What did Act No. 43 of 2024 change?", settings
    )
    results = retriever.invoke("What did Act No. 43 of 2024 change?")

    assert {doc.metadata["slug"] for doc in results} == {"cpc-amend-43-2024"}


def test_general_query_excludes_amending_instrument(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    vectorstore = _sample_vectorstore(tmp_path, fake_embeddings)
    settings = Settings(chroma_dir=tmp_path / "chroma", retrieval_k=5)

    retriever = get_retriever(
        vectorstore, "What is the time limit for filing an answer?", settings
    )
    results = retriever.invoke("What is the time limit for filing an answer?")

    slugs = {doc.metadata["slug"] for doc in results}
    assert slugs == {"cpc-consolidated-lankalaw"}
    assert "cpc-amend-43-2024" not in slugs


def test_cited_act_not_in_corpus_falls_back_to_general_default(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    vectorstore = _sample_vectorstore(tmp_path, fake_embeddings)
    settings = Settings(chroma_dir=tmp_path / "chroma", retrieval_k=5)

    # Act 99 of 2099 doesn't exist in this store -- should fall back to the
    # general default rather than returning nothing.
    retriever = get_retriever(
        vectorstore, "What did Act 99 of 2099 change?", settings
    )
    results = retriever.invoke("What did Act 99 of 2099 change?")

    assert {doc.metadata["slug"] for doc in results} == {"cpc-consolidated-lankalaw"}
