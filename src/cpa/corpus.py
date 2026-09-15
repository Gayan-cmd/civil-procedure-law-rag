"""Load the corpus manifest and build LangChain Documents from it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from cpa.config import Settings, get_settings


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Replace None values with "" -- Chroma's metadata store rejects None."""
    return {key: ("" if value is None else value) for key, value in metadata.items()}


def load_manifest(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    with open(settings.corpus_manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _build_document(entry: dict[str, Any], corpus_root: Path) -> Document:
    text_path = corpus_root / entry["text_path"]
    text = text_path.read_text(encoding="utf-8")

    metadata = sanitize_metadata(
        {
            "slug": entry["slug"],
            "title": entry["title"],
            "doc_type": entry["doc_type"],
            "act_number": entry["act_number"],
            "year": entry["year"],
            "amends": entry["amends"],
            "source_url": entry["source_url"],
            "content_role": entry.get("content_role"),
            "superseded_by": entry.get("superseded_by"),
        }
    )
    return Document(page_content=text, metadata=metadata)


def load_documents(settings: Settings | None = None) -> list[Document]:
    """Build a Document for every manifest entry that downloaded successfully.

    Entries with `download_status != "ok"` (failed downloads, mislabeled
    sources) have no text file and exist in the manifest only as a record
    of what was attempted -- they're skipped, not an error.
    """
    settings = settings or get_settings()
    manifest = load_manifest(settings)
    return [
        _build_document(entry, settings.corpus_root)
        for entry in manifest
        if entry["download_status"] == "ok"
    ]
