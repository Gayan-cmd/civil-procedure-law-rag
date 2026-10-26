"""Cross-encoder reranking -- the last ablation step, stacked on top of
hybrid_retrieval.py's BM25+dense fusion, per BUILD_PLAN.md's Phase 2
ablation table (naive dense -> +Act filtering -> +BM25 hybrid -> +rerank).

Dense search and BM25 both score a query against a document *without*
looking at them together -- a dense embedding is computed once per
document, independent of any particular question. A cross-encoder scores
the (question, document) *pair* directly: question and candidate text go
into the model together, so it can judge relevance far more precisely.
The cost is exactly why it only runs on a small shortlist (candidate_k
candidates) instead of the whole corpus -- scoring every chunk this way
would be far too slow to be usable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from cpa.config import Settings, get_settings
from cpa.store import PgVectorStore
from eval.hybrid_retrieval import hybrid_retrieve

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder(model_name: str = DEFAULT_MODEL) -> CrossEncoder:
    """Load the cross-encoder once; reuse it across every question.

    Mirrors cpa.generation.get_llm()/cpa.embedding.get_embeddings() --
    loading a model is expensive, so every caller in this project loads
    once and passes the loaded object down, rather than reloading per call.
    """
    return CrossEncoder(model_name)


def rerank(
    cross_encoder: CrossEncoder,
    documents: Sequence[Document],
    question: str,
    k: int,
) -> list[Document]:
    """Re-score each candidate against the question directly, keep the best k.

    documents is expected to be a cheaply-retrieved shortlist (tens of
    candidates), not the whole corpus -- see the module docstring for why.
    """
    if not documents:
        return []
    pairs = [[question, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in ranked[:k]]


def reranked_retrieve(
    vectorstore: PgVectorStore,
    all_chunks: Sequence[Document],
    question: str,
    cross_encoder: CrossEncoder,
    settings: Settings | None = None,
    candidate_k: int = 25,
) -> list[Document]:
    """Retrieve a wide candidate pool via hybrid_retrieve, then rerank it
    down to the final result count.

    candidate_k=25 matches BUILD_PLAN.md's "retrieve 25 cheaply" --
    passed through to hybrid_retrieve as its own candidate_k too, so each
    side of the BM25/dense fusion contributes enough for the fused pool to
    actually reach 25, not just settings.retrieval_k.
    """
    settings = settings or get_settings()
    k = settings.retrieval_k
    wide_settings = replace(settings, retrieval_k=candidate_k)

    candidates = hybrid_retrieve(
        vectorstore, all_chunks, question, wide_settings, candidate_k=candidate_k
    )
    return rerank(cross_encoder, candidates, question, k)
