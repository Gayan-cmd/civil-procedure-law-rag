from __future__ import annotations

from langchain_core.documents import Document

from cpa.config import Settings
from cpa.store import build_vectorstore, load_vectorstore
from conftest import TEST_DATABASE_URL, FakeEmbeddings


def test_build_vectorstore_persists_all_chunks(
    fake_embeddings: FakeEmbeddings,
) -> None:
    settings = Settings(database_url=TEST_DATABASE_URL)
    chunks = [
        Document(page_content="alpha", metadata={"slug": "a"}),
        Document(page_content="beta", metadata={"slug": "b"}),
    ]

    store = build_vectorstore(chunks, fake_embeddings, settings)

    assert store.count() == 2


def test_load_vectorstore_sees_previously_built_data(
    fake_embeddings: FakeEmbeddings,
) -> None:
    settings = Settings(database_url=TEST_DATABASE_URL)
    chunks = [Document(page_content="alpha", metadata={"slug": "a"})]
    build_vectorstore(chunks, fake_embeddings, settings)

    reopened = load_vectorstore(fake_embeddings, settings)

    assert reopened.count() == 1


def test_build_vectorstore_replaces_rather_than_appends(
    fake_embeddings: FakeEmbeddings,
) -> None:
    settings = Settings(database_url=TEST_DATABASE_URL)
    chunks = [Document(page_content="alpha", metadata={"slug": "a"})]

    build_vectorstore(chunks, fake_embeddings, settings)
    store = build_vectorstore(chunks, fake_embeddings, settings)

    # Re-running ingestion on an unchanged corpus must not double the
    # chunk count -- that's the whole point of a rebuild-from-scratch job.
    assert store.count() == 1


def test_load_vectorstore_on_empty_table_starts_empty(
    fake_embeddings: FakeEmbeddings,
) -> None:
    settings = Settings(database_url=TEST_DATABASE_URL)
    build_vectorstore([], fake_embeddings, settings)  # truncates, inserts nothing

    store = load_vectorstore(fake_embeddings, settings)

    assert store.count() == 0
