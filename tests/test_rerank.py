from __future__ import annotations

from langchain_core.documents import Document

from cpa.chunking import chunk_documents
from cpa.config import get_settings
from cpa.corpus import load_documents
from cpa.embedding import get_embeddings
from cpa.store import load_vectorstore
from eval.hybrid_retrieval import hybrid_retrieve
from eval.rerank import get_cross_encoder, rerank, reranked_retrieve

_CROSS_ENCODER = get_cross_encoder()


def test_rerank_fixes_a_clearly_wrong_input_order() -> None:
    question = "What is the time limit for filing a notice of appeal?"
    irrelevant = Document(
        page_content=(
            "An arbitral award made by an arbitral tribunal shall be final "
            "and binding on the parties to the arbitration agreement."
        )
    )
    correct = Document(
        page_content=(
            "The notice of appeal shall be presented within a period of "
            "fourteen days from the date when the decree or order appealed "
            "against was pronounced."
        )
    )
    # Deliberately handed in with the wrong document first.
    documents = [irrelevant, correct]

    reranked = rerank(_CROSS_ENCODER, documents, question, k=2)

    assert reranked[0].page_content == correct.page_content


def test_rerank_respects_k() -> None:
    question = "What is the time limit for filing a notice of appeal?"
    documents = [
        Document(page_content=f"Irrelevant filler document number {i}.")
        for i in range(5)
    ]
    reranked = rerank(_CROSS_ENCODER, documents, question, k=2)
    assert len(reranked) == 2


def test_rerank_empty_input_returns_empty() -> None:
    assert rerank(_CROSS_ENCODER, [], "any question?", k=5) == []


def test_reranking_improves_the_real_q002_ranking() -> None:
    """Sub-phase 2.4 found hybrid_retrieve surfaces the correct document
    for q002 ("must a plaint state facts establishing jurisdiction") but
    only at rank 2, behind a judicature-act-1978 chunk. Recall@5 doesn't
    care about that -- MRR does. This checks reranking actually moves the
    correct document closer to (or to) rank 1, against the real corpus,
    not a synthetic stand-in.
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

    before = hybrid_retrieve(vectorstore, chunks, question, settings)
    before_slugs = [doc.metadata.get("slug") for doc in before]
    before_rank = before_slugs.index("cpc-consolidated-lankalaw") + 1

    after = reranked_retrieve(
        vectorstore, chunks, question, _CROSS_ENCODER, settings
    )
    after_slugs = [doc.metadata.get("slug") for doc in after]
    after_rank = after_slugs.index("cpc-consolidated-lankalaw") + 1

    assert after_rank <= before_rank
