from __future__ import annotations

from collections import Counter

from langchain_core.documents import Document

from cpa.chunking import chunk_documents
from cpa.config import Settings
from cpa.corpus import load_documents

_TEST_SETTINGS = Settings(chunk_size=800, chunk_overlap=100)


def test_short_document_produces_one_chunk() -> None:
    doc = Document(page_content="short text", metadata={"slug": "x"})

    chunks = chunk_documents([doc], _TEST_SETTINGS)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"


def test_long_document_splits_into_multiple_chunks() -> None:
    long_text = "The plaintiff shall file the plaint. " * 100  # ~3,800 chars
    doc = Document(page_content=long_text, metadata={"slug": "x"})

    chunks = chunk_documents([doc], _TEST_SETTINGS)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= _TEST_SETTINGS.chunk_size


def test_chunks_inherit_source_document_metadata() -> None:
    doc = Document(
        page_content="Section 46. " * 200,
        metadata={"slug": "cpc-amend-4-2005", "doc_type": "amendment_act"},
    )

    chunks = chunk_documents([doc], _TEST_SETTINGS)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["slug"] == "cpc-amend-4-2005"
        assert chunk.metadata["doc_type"] == "amendment_act"


def test_real_corpus_chunk_count_is_in_expected_range() -> None:
    # BUILD_PLAN.md estimates ~1,900 chunks at 800/100 -- a wide band here
    # catches a badly broken splitter without being brittle to small text
    # edits in the corpus.
    documents = load_documents()
    chunks = chunk_documents(documents, _TEST_SETTINGS)

    assert 1000 < len(chunks) < 3000

    counts = Counter(c.metadata["slug"] for c in chunks)
    assert len(counts) == len(documents)
