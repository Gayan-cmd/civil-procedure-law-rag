from __future__ import annotations

import json

from cpa.config import get_settings
from cpa.corpus import load_manifest

GOLD_SET_PATH = get_settings().corpus_root / "eval" / "gold_set.json"

EXPECTED_CATEGORY_COUNTS = {
    "basic_lookup": 20,
    "act_citation": 8,
    "cross_document": 8,
    "amendment_aware": 6,
    "unanswerable": 8,
}

REQUIRED_KEYS = {
    "id",
    "question",
    "expected_doc",
    "expected_section",
    "category",
    "answerable",
}


def _load_gold_set() -> list[dict]:
    with open(GOLD_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _real_slugs() -> set[str]:
    manifest = load_manifest()
    return {entry["slug"] for entry in manifest if entry["download_status"] == "ok"}


def test_gold_set_has_fifty_questions() -> None:
    assert len(_load_gold_set()) == 50


def test_gold_set_category_counts_match_build_plan() -> None:
    gold_set = _load_gold_set()
    counts: dict[str, int] = {}
    for entry in gold_set:
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1

    assert counts == EXPECTED_CATEGORY_COUNTS


def test_every_entry_has_required_keys() -> None:
    for entry in _load_gold_set():
        assert set(entry.keys()) == REQUIRED_KEYS, entry["id"]


def test_ids_are_unique() -> None:
    ids = [entry["id"] for entry in _load_gold_set()]
    assert len(ids) == len(set(ids))


def test_answerable_questions_have_a_real_expected_doc() -> None:
    real_slugs = _real_slugs()
    for entry in _load_gold_set():
        if entry["answerable"]:
            assert entry["expected_doc"] in real_slugs, entry["id"]


def test_unanswerable_questions_have_no_expected_doc() -> None:
    for entry in _load_gold_set():
        if not entry["answerable"]:
            assert entry["expected_doc"] is None, entry["id"]
            assert entry["expected_section"] is None, entry["id"]


def test_categories_are_one_of_the_five_known_values() -> None:
    for entry in _load_gold_set():
        assert entry["category"] in EXPECTED_CATEGORY_COUNTS, entry["id"]
