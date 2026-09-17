from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from cpa.config import Settings
from cpa.store import build_vectorstore, load_vectorstore
from conftest import FakeEmbeddings


def test_build_vectorstore_persists_all_chunks(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    settings = Settings(chroma_dir=tmp_path / "chroma")
    chunks = [
        Document(page_content="alpha", metadata={"slug": "a"}),
        Document(page_content="beta", metadata={"slug": "b"}),
    ]

    store = build_vectorstore(chunks, fake_embeddings, settings)

    assert store._collection.count() == 2


def test_load_vectorstore_sees_previously_built_data(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    settings = Settings(chroma_dir=tmp_path / "chroma")
    chunks = [Document(page_content="alpha", metadata={"slug": "a"})]
    build_vectorstore(chunks, fake_embeddings, settings)

    reopened = load_vectorstore(fake_embeddings, settings)

    assert reopened._collection.count() == 1


def test_load_vectorstore_on_empty_dir_starts_empty(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    settings = Settings(chroma_dir=tmp_path / "chroma")

    store = load_vectorstore(fake_embeddings, settings)

    assert store._collection.count() == 0
