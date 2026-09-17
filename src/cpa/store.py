"""Vector store read/write, backed by Chroma."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cpa.config import Settings, get_settings

COLLECTION_NAME = "sri_lanka_civil_procedure"


def build_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    settings: Settings | None = None,
) -> Chroma:
    """Embed every chunk and write it to disk, replacing whatever was there.

    This is the ingestion job's job: a full rebuild from the current corpus,
    not an incremental append. Run it after any change to the corpus or to
    chunking.
    """
    settings = settings or get_settings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(settings.chroma_dir),
    )


def load_vectorstore(
    embeddings: Embeddings, settings: Settings | None = None
) -> Chroma:
    """Open an already-built vector store without re-embedding anything.

    This is what the API/retrieval side calls -- it must never re-embed the
    corpus just to answer a question.
    """
    settings = settings or get_settings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
