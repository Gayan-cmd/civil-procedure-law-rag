from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic, download-free stand-in for a real embedding model.

    Good enough for anything that only cares about wiring (does the store
    persist, does a filter include/exclude the right documents) rather than
    semantic search quality.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), float(text.count("a")), 1.0]


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
