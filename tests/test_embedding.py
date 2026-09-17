from __future__ import annotations

from cpa.embedding import get_embeddings


def test_embed_query_returns_normalized_384_dim_vector() -> None:
    embeddings = get_embeddings()

    vector = embeddings.embed_query(
        "What is the time limit for filing an answer?"
    )

    assert len(vector) == 384
    norm = sum(component * component for component in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-3
