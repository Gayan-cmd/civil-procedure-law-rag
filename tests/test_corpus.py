from __future__ import annotations

from pathlib import Path

from cpa.corpus import _build_document, load_documents, sanitize_metadata


def test_sanitize_metadata_replaces_none() -> None:
    assert sanitize_metadata({"a": None, "b": 1, "c": "x"}) == {
        "a": "",
        "b": 1,
        "c": "x",
    }


def test_build_document_sanitizes_none_fields(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("hello world", encoding="utf-8")
    entry = {
        "slug": "doc",
        "title": "Doc",
        "doc_type": "principal_act",
        "act_number": None,
        "year": None,
        "amends": None,
        "source_url": "http://example.test",
        "text_path": "doc.txt",
    }

    doc = _build_document(entry, tmp_path)

    assert doc.page_content == "hello world"
    assert doc.metadata["act_number"] == ""
    assert doc.metadata["year"] == ""
    assert doc.metadata["slug"] == "doc"


def test_build_document_defaults_missing_content_role(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("text", encoding="utf-8")
    entry = {
        "slug": "doc",
        "title": "Doc",
        "doc_type": "principal_act",
        "act_number": None,
        "year": None,
        "amends": None,
        "source_url": "http://example.test",
        "text_path": "doc.txt",
    }

    doc = _build_document(entry, tmp_path)

    assert doc.metadata["content_role"] == ""
    assert doc.metadata["superseded_by"] == ""


def test_load_documents_excludes_failed_downloads() -> None:
    documents = load_documents()
    slugs = {doc.metadata["slug"] for doc in documents}

    assert len(documents) == 18
    assert "cpc-amend-50-2024" not in slugs
    assert "cpc-consolidated-commonlii" not in slugs
    assert "cpc-consolidated-lawnet" not in slugs


def test_load_documents_metadata_has_no_none_values() -> None:
    for doc in load_documents():
        assert None not in doc.metadata.values()


def test_load_documents_content_role_matches_duplicate_policy() -> None:
    by_slug = {doc.metadata["slug"]: doc for doc in load_documents()}

    assert by_slug["cpc-consolidated-lankalaw"].metadata["content_role"] == (
        "primary_consolidated_text"
    )
    assert by_slug["cpc-amend-43-2024"].metadata["content_role"] == (
        "amending_instrument"
    )
    assert (
        by_slug["cpc-amend-43-2024"].metadata["superseded_by"]
        == "cpc-consolidated-lankalaw"
    )
