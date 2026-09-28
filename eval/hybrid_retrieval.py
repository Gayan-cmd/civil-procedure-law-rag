"""BM25 keyword search fused with dense vector search, via Reciprocal Rank
Fusion (RRF) -- an ablation-only retrieval strategy compared against
production retrieval.py in the eval harness, not a change to it.

Why this exists: dense (embedding) search measures semantic similarity,
which can be fooled by a document that's thematically close but wrong
(see the real q002 miss from sub-phase 2.3 -- "jurisdiction of the court"
pulled every result toward the Judicature Act, whose whole subject is
court jurisdiction, over the one specific CPC section that actually
answers the question). BM25 instead scores exact keyword/term overlap,
which doesn't get fooled the same way -- it doesn't care what a document
is "about," only whether the query's actual words appear in it. Fusing
the two catches cases where either one alone would miss.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from cpa.config import Settings, get_settings
from cpa.retrieval import extract_act_reference, get_retriever


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokens -- good enough for BM25, which
    only needs consistent word boundaries, not linguistic correctness.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(documents: Sequence[Document]) -> BM25Okapi:
    return BM25Okapi([_tokenize(doc.page_content) for doc in documents])


def _eligible_documents(
    all_chunks: Sequence[Document], question: str
) -> list[Document]:
    """Mirrors get_retriever's duplicate-avoidance policy (sub-phase 1.5),
    applied to a plain list of Documents.

    BM25 has no native metadata filter to push this into the way Chroma's
    `where` clause does -- so eligibility is decided here in Python,
    before scoring, using the exact same two-branch decision:
    act-specific questions get that act's chunks; general questions get
    only the current in-force text (content_role == primary_consolidated_text).
    """
    act_number, year = extract_act_reference(question)
    if act_number and year:
        matches = [
            doc
            for doc in all_chunks
            if doc.metadata.get("act_number") == act_number
            and doc.metadata.get("year") == year
        ]
        if matches:
            return matches
    return [
        doc
        for doc in all_chunks
        if doc.metadata.get("content_role") == "primary_consolidated_text"
    ]


def bm25_search(
    all_chunks: Sequence[Document], question: str, k: int
) -> list[Document]:
    """BM25 keyword search over the eligible chunk pool for this question."""
    eligible = _eligible_documents(all_chunks, question)
    if not eligible:
        return []
    index = build_bm25_index(eligible)
    scores = index.get_scores(_tokenize(question))
    ranked_indices = sorted(range(len(eligible)), key=lambda i: scores[i], reverse=True)
    return [eligible[i] for i in ranked_indices[:k]]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k_constant: int = 60
) -> list[str]:
    """Fuse multiple ranked lists of item IDs into one ranking.

    score(item) = sum, over every list it appears in, of 1/(k_constant + rank)
    -- rank counted from 1. An item missing from a list simply contributes
    0 from that list, so items appearing in both lists (even at middling
    ranks in each) can out-rank an item that's 1st in only one list.
    k_constant=60 matches BUILD_PLAN.md's formula and the standard RRF
    default.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_constant + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


def hybrid_retrieve(
    vectorstore: Chroma,
    all_chunks: Sequence[Document],
    question: str,
    settings: Settings | None = None,
    candidate_k: int = 10,
) -> list[Document]:
    """Dense search + BM25 keyword search, fused by RRF.

    Each side retrieves `candidate_k` candidates (wider than the final
    result count, so RRF has real overlap to work with) before fusion
    narrows back down to settings.retrieval_k.
    """
    settings = settings or get_settings()
    k = settings.retrieval_k
    wide_settings = replace(settings, retrieval_k=candidate_k)

    dense_docs = get_retriever(vectorstore, question, wide_settings).invoke(question)
    bm25_docs = bm25_search(all_chunks, question, candidate_k)

    # page_content as the fusion key: chunking never produces two chunks
    # with identical text, so it's a reliable stand-in for chunk identity
    # without needing a dedicated per-chunk ID that doesn't exist yet.
    by_content = {doc.page_content: doc for doc in dense_docs + bm25_docs}
    dense_ranking = [doc.page_content for doc in dense_docs]
    bm25_ranking = [doc.page_content for doc in bm25_docs]

    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
    return [by_content[content] for content in fused[:k]]
