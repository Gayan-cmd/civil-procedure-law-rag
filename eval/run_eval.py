"""Baseline evaluation: run the full RAG pipeline over eval/gold_set.json
and report Recall@k, MRR, refusal accuracy, and retrieval/generation
latency (p50/p95), split apart per BUILD_PLAN.md's Phase 2.

This is the "naive dense + Act-number filtering" row of the eventual
ablation table -- it exercises the current production retrieval.py as-is
(dense similarity search plus the content_role/act-number filter from
sub-phase 1.5), not a stripped-down "no filtering at all" baseline. The
later ablation sub-phases (2.4-2.6) add separate BM25/rerank variants for
comparison, without changing this script.

Usage (must be run with -m, not as a bare script -- see note in the
README/sub-phase explanation about why):
    python -m eval.run_eval [--recall-threshold 0.90] [--k 5]

Exits nonzero if Recall@k (over the answerable questions) falls below
the threshold. That exit code is what Phase 5's CI pipeline will consume.
Default threshold (0.90) is calibrated against this project's own
observed baseline (0.976 on the real corpus, 2026-09-24) with a margin
for run-to-run noise -- not a number chosen before any real run existed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import groq

from cpa.chain import format_docs
from cpa.config import Settings, get_settings
from cpa.embedding import get_embeddings
from cpa.generation import get_llm, get_prompt
from cpa.retrieval import get_retriever
from cpa.store import load_vectorstore

from eval.metrics import (
    mean_reciprocal_rank,
    mean_recall_at_k,
    p50,
    p95,
    refusal_accuracy,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "baseline.json"


def _invoke_with_retry(
    llm: Any, formatted: Any, max_retries: int = 5
) -> tuple[Any, float]:
    """Call llm.invoke(), retrying on Groq's free-tier rate limit.

    50 full-pipeline questions back to back is enough to trip the free
    tier's tokens-per-minute limit (observed directly: a real run hit a
    429 after ~34 questions). Groq's error response includes a
    "Retry-After" header with the exact wait time -- use that instead of
    guessing, falling back to a fixed delay only if it's ever missing.

    Returns (response, latency) where latency times only the successful
    call itself -- rate-limit backoff waiting is deliberately excluded,
    so a throttled question doesn't show up as a bogus latency spike.
    """
    for attempt in range(max_retries):
        try:
            start = time.monotonic()
            response = llm.invoke(formatted)
            return response, time.monotonic() - start
        except groq.RateLimitError as error:
            if attempt == max_retries - 1:
                raise
            retry_after = error.response.headers.get("retry-after")
            delay = float(retry_after) if retry_after else 5.0 * (2**attempt)
            print(f"    rate limited, waiting {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def evaluate_question(
    entry: dict[str, Any],
    vectorstore: Any,
    prompt: Any,
    llm: Any,
    settings: Settings,
) -> dict[str, Any]:
    """Run one gold-set question through retrieval, then generation,
    timing each stage independently.
    """
    question = entry["question"]

    retrieval_start = time.monotonic()
    retriever = get_retriever(vectorstore, question, settings)
    documents = retriever.invoke(question)
    retrieval_latency = time.monotonic() - retrieval_start

    retrieved_slugs = [doc.metadata.get("slug", "") for doc in documents]

    context = format_docs(documents)
    formatted = prompt.invoke({"context": context, "question": question})
    response, generation_latency = _invoke_with_retry(llm, formatted)

    return {
        "id": entry["id"],
        "category": entry["category"],
        "answerable": entry["answerable"],
        "expected_doc": entry["expected_doc"],
        "retrieved_slugs": retrieved_slugs,
        "answer": response.content,
        "retrieval_latency": retrieval_latency,
        "generation_latency": generation_latency,
    }


def print_table(
    results: list[dict[str, Any]], settings: Settings, k: int
) -> tuple[float, float]:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    recall_pairs = [(r["retrieved_slugs"], r["expected_doc"]) for r in answerable]
    recall = mean_recall_at_k(recall_pairs, k=k)
    mrr = mean_reciprocal_rank(recall_pairs, k=k)
    refusal_acc = refusal_accuracy([r["answer"] for r in unanswerable])

    retrieval_latencies = [r["retrieval_latency"] for r in results]
    generation_latencies = [r["generation_latency"] for r in results]

    print()
    print(f"{'Metric':<28}{'Value':>12}")
    print("-" * 40)
    print(f"{'Recall@' + str(k):<28}{recall:>12.3f}")
    print(f"{'MRR':<28}{mrr:>12.3f}")
    print(f"{'Refusal accuracy':<28}{refusal_acc:>12.3f}")
    print(f"{'Retrieval p50 (s)':<28}{p50(retrieval_latencies):>12.3f}")
    print(f"{'Retrieval p95 (s)':<28}{p95(retrieval_latencies):>12.3f}")
    print(f"{'Generation p50 (s)':<28}{p50(generation_latencies):>12.3f}")
    print(f"{'Generation p95 (s)':<28}{p95(generation_latencies):>12.3f}")

    print()
    print(f"{'Category':<20}{'Recall@' + str(k):>12}{'n':>6}")
    print("-" * 38)
    categories = sorted({r["category"] for r in answerable})
    for category in categories:
        cat_pairs = [
            (r["retrieved_slugs"], r["expected_doc"])
            for r in answerable
            if r["category"] == category
        ]
        cat_recall = mean_recall_at_k(cat_pairs, k=k)
        print(f"{category:<20}{cat_recall:>12.3f}{len(cat_pairs):>6}")

    return recall, refusal_acc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=0.90,
        help="Exit nonzero if Recall@k falls below this (default: 0.90).",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Override retrieval_k from Settings."
    )
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.retrieval_k

    gold_set_path = settings.corpus_root / "eval" / "gold_set.json"
    gold_set = json.loads(gold_set_path.read_text(encoding="utf-8"))

    print("Loading embedding model and vector store...")
    embeddings = get_embeddings(settings)
    vectorstore = load_vectorstore(embeddings, settings)
    prompt = get_prompt()
    llm = get_llm(settings)

    print(f"Running {len(gold_set)} gold-set questions...")
    results = []
    for i, entry in enumerate(gold_set, start=1):
        result = evaluate_question(entry, vectorstore, prompt, llm, settings)
        results.append(result)
        print(
            f"  [{i}/{len(gold_set)}] {entry['id']} "
            f"(retrieval={result['retrieval_latency']:.2f}s, "
            f"generation={result['generation_latency']:.2f}s)"
        )

    recall, refusal_acc = print_table(results, settings, k)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRaw results written to {RESULTS_PATH}")

    if recall < args.recall_threshold:
        print(
            f"\nFAIL: Recall@{k} ({recall:.3f}) is below threshold "
            f"({args.recall_threshold})"
        )
        return 1

    print(f"\nPASS: Recall@{k} ({recall:.3f}) meets threshold ({args.recall_threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
