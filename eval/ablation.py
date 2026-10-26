"""Run BUILD_PLAN.md's full Phase 2 ablation: all four retrieval
configurations against the same 50 gold-set questions, producing the
one table the plan calls "the single most valuable artifact in the
project for your CV."

    Naive dense, k=5
    + Act-number filtering
    + BM25 hybrid (RRF)
    + Cross-encoder rerank

Each row adds one more technique on top of the row before it -- this
script doesn't introduce any new retrieval logic, it only runs what
sub-phases 2.3-2.5 already built, side by side, on the same questions.

Scope of what's measured, and why: Recall@5, MRR, and retrieval p95
latency are computed over all 50 questions for every config -- they're
cheap (no LLM call) and they're the entire point of comparing retrieval
strategies. Refusal accuracy additionally needs generation, but only for
the 8 unanswerable questions per config (32 new LLM calls total across
the 3 new configs, not 150) -- a retrieval change could plausibly affect
whether the model refuses (e.g. BM25 pulling in a tangentially related
chunk that tempts the model to answer instead of refusing), so it's
still worth checking per config, just not worth re-running all 42
answerable questions' generation four times over to get it. Row 2
("+ Act-number filtering") reuses sub-phase 2.3's full baseline run
(eval/results/baseline.json) instead of repeating it -- that run IS this
row, exactly as it already exists.

Usage (must be run with -m, same reason as run_eval.py):
    python -m eval.ablation
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from cpa.chain import format_docs
from cpa.chunking import chunk_documents
from cpa.config import get_settings
from cpa.corpus import load_documents
from cpa.embedding import get_embeddings
from cpa.generation import get_llm, get_prompt
from cpa.store import load_vectorstore

from eval.groq_retry import invoke_with_retry
from eval.hybrid_retrieval import hybrid_retrieve
from eval.metrics import mean_reciprocal_rank, mean_recall_at_k, p95, refusal_accuracy
from eval.rerank import get_cross_encoder, reranked_retrieve
from eval.run_eval import RESULTS_PATH as BASELINE_PATH

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "ablation.json"


def naive_dense_retrieve(
    vectorstore: Chroma, question: str, k: int
) -> list[Document]:
    """Row 1: plain similarity search, no Act-number/content_role
    filtering at all.

    This is the one configuration that doesn't already exist as code --
    every other retrieval function built so far (get_retriever,
    hybrid_retrieve, reranked_retrieve) already applies the 1.5
    duplicate-avoidance filter, since that's what the real system
    actually needs. This function exists only to give the ablation
    table an honest "before any of that" starting point.
    """
    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    ).invoke(question)


def evaluate_config(
    name: str,
    retrieve_fn: Callable[[str], list[Document]],
    gold_set: list[dict[str, Any]],
    prompt: Any,
    llm: Any,
    k: int,
) -> dict[str, Any]:
    """Run one retrieval strategy across the full gold set.

    Retrieval is timed and scored for every question. Generation only
    runs for unanswerable questions -- see the module docstring for why.
    """
    per_question = []
    for entry in gold_set:
        question = entry["question"]

        start = time.monotonic()
        documents = retrieve_fn(question)
        retrieval_latency = time.monotonic() - start
        retrieved_slugs = [doc.metadata.get("slug", "") for doc in documents]

        answer = None
        if not entry["answerable"]:
            context = format_docs(documents)
            formatted = prompt.invoke({"context": context, "question": question})
            response, _ = invoke_with_retry(llm, formatted)
            answer = response.content

        per_question.append(
            {
                "id": entry["id"],
                "answerable": entry["answerable"],
                "expected_doc": entry["expected_doc"],
                "retrieved_slugs": retrieved_slugs,
                "retrieval_latency": retrieval_latency,
                "answer": answer,
            }
        )

    return summarize(name, per_question, k)


def summarize(name: str, per_question: list[dict[str, Any]], k: int) -> dict[str, Any]:
    answerable = [r for r in per_question if r["answerable"]]
    unanswerable = [r for r in per_question if not r["answerable"]]

    recall_pairs = [(r["retrieved_slugs"], r["expected_doc"]) for r in answerable]

    return {
        "name": name,
        "recall_at_k": mean_recall_at_k(recall_pairs, k=k),
        "mrr": mean_reciprocal_rank(recall_pairs, k=k),
        "refusal_accuracy": refusal_accuracy(
            [r["answer"] for r in unanswerable if r["answer"] is not None]
        ),
        "retrieval_p95": p95([r["retrieval_latency"] for r in per_question]),
        "per_question": per_question,
    }


def load_existing_baseline(name: str, path: Path, k: int) -> dict[str, Any]:
    """Row 2: reuse sub-phase 2.3's already-complete run instead of
    repeating 50 questions' worth of retrieval and generation that
    would produce the exact same thing.
    """
    results = json.loads(path.read_text(encoding="utf-8"))
    return summarize(name, results, k=k)


def print_ablation_table(configs: list[dict[str, Any]]) -> None:
    print()
    print(f"{'Config':<28}{'Recall@5':>10}{'MRR':>8}{'Refusal':>10}{'Retr p95':>10}")
    print("-" * 66)
    for config in configs:
        print(
            f"{config['name']:<28}"
            f"{config['recall_at_k']:>10.3f}"
            f"{config['mrr']:>8.3f}"
            f"{config['refusal_accuracy']:>10.3f}"
            f"{config['retrieval_p95']:>9.3f}s"
        )


def main() -> int:
    settings = get_settings()
    k = settings.retrieval_k

    gold_set_path = settings.corpus_root / "eval" / "gold_set.json"
    gold_set = json.loads(gold_set_path.read_text(encoding="utf-8"))

    print("Loading embedding model, vector store, corpus chunks, cross-encoder, and LLM...")
    embeddings = get_embeddings(settings)
    vectorstore = load_vectorstore(embeddings, settings)
    all_chunks = chunk_documents(load_documents(settings), settings)
    cross_encoder = get_cross_encoder()
    prompt = get_prompt()
    llm = get_llm(settings)

    configs = []

    print("\n[1/4] Naive dense, k=5 ...")
    configs.append(
        evaluate_config(
            "Naive dense, k=5",
            lambda q: naive_dense_retrieve(vectorstore, q, k),
            gold_set,
            prompt,
            llm,
            k,
        )
    )

    print("[2/4] + Act-number filtering (reusing sub-phase 2.3's baseline run) ...")
    configs.append(
        load_existing_baseline("+ Act-number filtering", BASELINE_PATH, k)
    )

    print("[3/4] + BM25 hybrid (RRF) ...")
    configs.append(
        evaluate_config(
            "+ BM25 hybrid (RRF)",
            lambda q: hybrid_retrieve(vectorstore, all_chunks, q, settings),
            gold_set,
            prompt,
            llm,
            k,
        )
    )

    print("[4/4] + Cross-encoder rerank ...")
    configs.append(
        evaluate_config(
            "+ Cross-encoder rerank",
            lambda q: reranked_retrieve(
                vectorstore, all_chunks, q, cross_encoder, settings
            ),
            gold_set,
            prompt,
            llm,
            k,
        )
    )

    print_ablation_table(configs)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    print(f"\nFull results written to {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
