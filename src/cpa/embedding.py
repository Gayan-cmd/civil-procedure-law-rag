"""Embedding model wrapper."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from cpa.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> HuggingFaceEmbeddings:
    """Build the embedding model used for both ingestion and querying.

    Normalizing embeddings to unit length is what makes cosine similarity
    and dot-product similarity equivalent -- Chroma's default distance
    metric relies on this.
    """
    settings = settings or get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
