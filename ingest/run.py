"""Ingestion job: rebuild the vector store from the current corpus.

This is a batch job, not a service -- run it occasionally, whenever the
corpus or chunking changes, not on every API start. See BUILD_PLAN.md's
explanation of why ingestion and serving are kept separate: the API only
ever loads what this script already built (cpa.store.load_vectorstore),
so restarting the API never means re-embedding the whole corpus.

Usage:
    python ingest/run.py
"""

from __future__ import annotations

import time

from cpa.chunking import chunk_documents
from cpa.config import get_settings
from cpa.corpus import load_documents
from cpa.embedding import get_embeddings
from cpa.store import build_vectorstore


def main() -> None:
    settings = get_settings()
    start = time.monotonic()

    print("Loading corpus...")
    documents = load_documents(settings)
    print(f"  {len(documents)} documents loaded")

    print("Chunking...")
    chunks = chunk_documents(documents, settings)
    print(f"  {len(chunks)} chunks "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    print(f"Loading embedding model ({settings.embedding_model})...")
    embeddings = get_embeddings(settings)

    print(f"Building vector store at {settings.chroma_dir}...")
    store = build_vectorstore(chunks, embeddings, settings)

    actual = store._collection.count()
    if actual != len(chunks):
        raise RuntimeError(
            f"Vector count ({actual}) doesn't match chunk count ({len(chunks)}) "
            "-- ingestion did not complete cleanly."
        )

    elapsed = time.monotonic() - start
    print(f"Done: {actual} vectors stored in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
