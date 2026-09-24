"""Pure, tested metric functions for the evaluation harness.

Nothing here calls retrieval or generation -- these functions only ever
operate on already-collected results (retrieved document slugs, generated
answer text, latency measurements). That's what makes them testable with
small hand-crafted inputs instead of the real corpus and a live LLM.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from cpa.generation import REFUSAL_PHRASE


def recall_at_k(retrieved_docs: Sequence[str], expected_doc: str, k: int = 5) -> bool:
    """Did expected_doc appear anywhere in the first k retrieved doc slugs?

    Document-level, matching BUILD_PLAN.md's own definition of Recall@5
    ("did the correct document appear in the top 5 chunks") -- chunks
    aren't checked individually, only which document each came from.
    """
    return expected_doc in retrieved_docs[:k]


def mean_recall_at_k(
    results: Sequence[tuple[Sequence[str], str]], k: int = 5
) -> float:
    """Recall@k averaged over many questions.

    results: (retrieved_doc_slugs, expected_doc) pairs, one per answerable
    question. Returns 0.0 for an empty results list rather than raising --
    an eval run over zero questions has no recall to report, not an error.
    """
    if not results:
        return 0.0
    hits = sum(
        recall_at_k(retrieved, expected, k) for retrieved, expected in results
    )
    return hits / len(results)


def reciprocal_rank(
    retrieved_docs: Sequence[str], expected_doc: str, k: int = 5
) -> float:
    """1 / (rank of expected_doc's first occurrence in the first k results).

    0.0 if it doesn't appear in the first k at all -- same "not found"
    convention as recall_at_k, so the two metrics agree on what "top k"
    means.
    """
    for rank, slug in enumerate(retrieved_docs[:k], start=1):
        if slug == expected_doc:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    results: Sequence[tuple[Sequence[str], str]], k: int = 5
) -> float:
    if not results:
        return 0.0
    return sum(
        reciprocal_rank(retrieved, expected, k) for retrieved, expected in results
    ) / len(results)


def is_refusal(answer: str) -> bool:
    """Did the model refuse rather than answer -- checked against the exact
    phrase generation.py's own prompt instructs it to use, so this can
    never silently drift out of sync with what the prompt actually says.
    """
    return REFUSAL_PHRASE.lower() in answer.lower()


def refusal_accuracy(answers: Sequence[str]) -> float:
    """Of these answers (expected to be for unanswerable questions), what
    fraction were refusals rather than fabricated answers?
    """
    if not answers:
        return 0.0
    return sum(is_refusal(answer) for answer in answers) / len(answers)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of a list of measurements (pct in [0, 100]).

    Nearest-rank rather than interpolated: simpler to explain and to test
    exactly, and with eval runs this small (dozens of latency samples, not
    thousands) interpolation wouldn't be meaningfully more accurate.
    """
    if not values:
        raise ValueError("percentile() needs at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def p50(values: Sequence[float]) -> float:
    return percentile(values, 50)


def p95(values: Sequence[float]) -> float:
    return percentile(values, 95)
