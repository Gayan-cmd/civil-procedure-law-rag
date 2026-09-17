from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cpa.config import Settings
from cpa.store import build_vectorstore, load_vectorstore


class _FakeEmbeddings(Embeddings):
    """Deterministic, download-free stand-in for a real embedding model.

    Store/load round-tripping doesn't care whether the numbers mean
    anything semantically -- only that the same text always maps to the
    same vector, so we don't need the real ~90MB model for these tests.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), float(text.count("a")), 1.0]


def test_build_vectorstore_persists_all_chunks(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path / "chroma")
    chunks = [
        Document(page_content="alpha", metadata={"slug": "a"}),
        Document(page_content="beta", metadata={"slug": "b"}),
    ]

    store = build_vectorstore(chunks, _FakeEmbeddings(), settings)

    assert store._collection.count() == 2


def test_load_vectorstore_sees_previously_built_data(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path / "chroma")
    chunks = [Document(page_content="alpha", metadata={"slug": "a"})]
    build_vectorstore(chunks, _FakeEmbeddings(), settings)

    reopened = load_vectorstore(_FakeEmbeddings(), settings)

    assert reopened._collection.count() == 1


def test_load_vectorstore_on_empty_dir_starts_empty(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path / "chroma")

    store = load_vectorstore(_FakeEmbeddings(), settings)

    assert store._collection.count() == 0
