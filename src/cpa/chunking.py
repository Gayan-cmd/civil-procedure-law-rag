"""Split Documents into retrieval-sized chunks."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cpa.config import Settings, get_settings

# Prefer splitting on paragraph breaks, then lines, then sentences, then
# words -- in that order -- so a chunk boundary lands somewhere readable
# rather than mid-word.
_SEPARATORS = ["\n\n", "\n", ".", " "]


def get_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    settings = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
    )


def chunk_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Document]:
    """Split each Document into overlapping chunks, inheriting its metadata.

    Every chunk of a given document carries the same slug/title/doc_type/etc.
    -- there's no per-chunk position info yet, which is fine until retrieval
    needs to tell chunks of the same document apart.
    """
    splitter = get_splitter(settings)
    return splitter.split_documents(documents)
