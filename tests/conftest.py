from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic, download-free stand-in for a real embedding model.

    Good enough for anything that only cares about wiring (does the store
    persist, does a filter include/exclude the right documents) rather than
    semantic search quality.

    Padded to 384 dimensions -- store.py's Postgres `chunks` table has a
    fixed vector(384) column (tied to the real embedding model), so even a
    fake vector must be exactly that length or the insert fails outright.
    Only the first 3 values carry any real signal; the rest are zeros,
    which doesn't change any two fake vectors' relative distance from each
    other (padding both sides of a comparison with the same zeros changes
    nothing about which one is closer).
    """

    _DIMENSIONS = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        signal = [float(len(text) % 7), float(text.count("a")), 1.0]
        return signal + [0.0] * (self._DIMENSIONS - len(signal))


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
