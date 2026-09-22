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
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    # Chroma.from_documents() would just ADD to whatever's already in this
    # collection -- calling it twice with the same corpus silently doubles
    # every chunk. reset_collection() clears it first so this function's
    # docstring promise ("replacing whatever was there") is actually true.
    store.reset_collection()
    store.add_documents(chunks)
    return store


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
