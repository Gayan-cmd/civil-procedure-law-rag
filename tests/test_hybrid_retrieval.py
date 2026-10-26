from __future__ import annotations

from langchain_core.documents import Document

from cpa.chunking import chunk_documents
from cpa.config import Settings, get_settings
from cpa.corpus import load_documents
from cpa.embedding import get_embeddings
from cpa.store import build_vectorstore, load_vectorstore
from conftest import FakeEmbeddings
from eval.hybrid_retrieval import (
    _eligible_documents,
    _tokenize,
    bm25_search,
    hybrid_retrieve,
    reciprocal_rank_fusion,
)


def test_tokenize_lowercases_and_splits_on_non_alphanumeric() -> None:
    assert _tokenize("Act No. 43/2024") == ["act", "no", "43", "2024"]


def test_reciprocal_rank_fusion_matches_the_formula() -> None:
    # X: rank 1 in list1 -> 1/61. Y: rank 2 in list1 -> 1/62.
    # Z: rank 1 in list2 -> 1/61 (tied with X, and sorted() is stable, so
    # X -- appearing first in the input -- stays ahead of Z on a tie).
    fused = reciprocal_rank_fusion([["X", "Y"], ["Z"]])
    assert fused == ["X", "Z", "Y"]


def test_reciprocal_rank_fusion_promotes_items_ranked_in_both_lists() -> None:
    # "A" is only 3rd in list1 and 2nd in list2, but appearing in BOTH
    # should out-rank "X"/"B", which are 1st but only in a single list.
    fused = reciprocal_rank_fusion([["X", "Y", "A"], ["B", "A", "Z"]])
    assert fused[0] == "A"


def test_reciprocal_rank_fusion_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_eligible_documents_general_question_excludes_amending_instrument() -> None:
    chunks = [
        Document(
            page_content="current text",
            metadata={"content_role": "primary_consolidated_text"},
        ),
        Document(
            page_content="amending text",
            metadata={"content_role": "amending_instrument", "act_number": "43", "year": 2024},
        ),
    ]
    eligible = _eligible_documents(chunks, "What is the time limit for filing an answer?")
    assert len(eligible) == 1
    assert eligible[0].page_content == "current text"


def test_eligible_documents_act_specific_question_returns_that_act_only() -> None:
    chunks = [
        Document(
            page_content="current text",
            metadata={"content_role": "primary_consolidated_text"},
        ),
        Document(
            page_content="amending text",
            metadata={"content_role": "amending_instrument", "act_number": "43", "year": 2024},
        ),
    ]
    eligible = _eligible_documents(chunks, "What did Act No. 43 of 2024 change?")
    assert len(eligible) == 1
    assert eligible[0].page_content == "amending text"


def test_bm25_search_finds_exact_keyword_match() -> None:
    # A third, unrelated distractor document matters here: with only two
    # documents where a term appears in exactly one of them, BM25's
    # classic IDF formula -- log((N - n + 0.5) / (n + 0.5)) -- evaluates
    # to log(1) == 0 (N=2, n=1), which zeroes out the score entirely and
    # makes every document tie at 0. A third document breaks that
    # degenerate case, same as any real corpus would.
    chunks = [
        Document(
            page_content="The plaintiff shall file the plaint within the district.",
            metadata={"content_role": "primary_consolidated_text"},
        ),
        Document(
            page_content="Every District Court shall have unlimited original jurisdiction.",
            metadata={"content_role": "primary_consolidated_text"},
        ),
        Document(
            page_content="An arbitral award shall be final and binding on the parties.",
            metadata={"content_role": "primary_consolidated_text"},
        ),
    ]
    results = bm25_search(chunks, "unlimited original jurisdiction", k=1)
    assert len(results) == 1
    assert "unlimited original jurisdiction" in results[0].page_content


def test_hybrid_retrieve_respects_eligibility_and_k(
    fake_embeddings: FakeEmbeddings,
) -> None:
    settings = Settings(retrieval_k=1)
    chunks = [
        Document(
            page_content="current in-force answer text",
            metadata={
                "slug": "cpc-consolidated",
                "content_role": "primary_consolidated_text",
                "act_number": "",
                "year": "",
            },
        ),
        Document(
            page_content="amending instrument text",
            metadata={
                "slug": "cpc-amend-43-2024",
                "content_role": "amending_instrument",
                "act_number": "43",
                "year": 2024,
            },
        ),
    ]
    vectorstore = build_vectorstore(chunks, fake_embeddings, settings)

    results = hybrid_retrieve(
        vectorstore, chunks, "What is the time limit for filing an answer?", settings
    )

    assert len(results) == 1
    assert results[0].metadata["slug"] == "cpc-consolidated"


def test_hybrid_retrieve_fixes_the_real_q002_miss() -> None:
    """Spot check against the real corpus, not synthetic data.

    Sub-phase 2.3's baseline run found a genuine dense-search miss: asking
    about a plaint stating "the jurisdiction of the court" retrieved 5/5
    chunks from judicature-act-1978 (whose entire subject is jurisdiction)
    instead of the CPC's specific section 45. BM25 doesn't get pulled by
    thematic similarity -- it only cares about literal term overlap, and
    the corpus's actual section 45 text does contain these exact query
    words. This proves the hybrid variant actually fixes a real,
    previously-documented failure, not just a hypothetical one.
    """
    question = (
        "Must a plaint state facts establishing the jurisdiction of the "
        "court in which it is filed?"
    )
    settings = get_settings()

    documents = load_documents(settings)
    chunks = chunk_documents(documents, settings)
    embeddings = get_embeddings(settings)
    vectorstore = load_vectorstore(embeddings, settings)

    results = hybrid_retrieve(vectorstore, chunks, question, settings)
    retrieved_slugs = [doc.metadata.get("slug") for doc in results]

    assert "cpc-consolidated-lankalaw" in retrieved_slugs
