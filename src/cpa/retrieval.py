"""Search logic: Act-number citation detection and the retriever it drives."""

from __future__ import annotations

import re

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from cpa.config import Settings, get_settings

# Matches both "Act No. 43 of 2024" and the slash form "Act 43/2024".
_ACT_PATTERN = re.compile(
    r"Act\s+(?:No\.?\s*)?(\d+)\s*(?:of\s+(\d{4})|/(\d{4}))",
    re.IGNORECASE,
)


def extract_act_reference(question: str) -> tuple[str | None, int | None]:
    """Pull an act number and year out of a question, if it names one.

    Returns (None, None) when no citation is found -- this is how
    get_retriever decides whether a question is "about a specific act" or
    a general question.
    """
    match = _ACT_PATTERN.search(question)
    if not match:
        return None, None
    act_number = match.group(1)
    year = match.group(2) or match.group(3)
    return act_number, int(year)


def get_retriever(
    vectorstore: VectorStore,
    question: str,
    settings: Settings | None = None,
) -> BaseRetriever:
    """Pick the right retriever for a question, applying the duplicate policy.

    - Question names a specific act (e.g. "What did Act 43/2024 change?"):
      the amending instrument itself is what they're asking about -- its
      legislative language and reasoning -- so filter to that act/year.
    - Otherwise: default to the current in-force text only
      (content_role == "primary_consolidated_text"), excluding amending
      instruments, so a general question can never surface both the final
      text and the (out-of-context) instruction that produced it.

    If an act is named but nothing matches it (e.g. a typo, or an act not
    in the corpus), falls back to the general case rather than returning
    nothing.
    """
    settings = settings or get_settings()
    k = settings.retrieval_k

    act_number, year = extract_act_reference(question)
    if act_number and year:
        where_filter = {"$and": [{"act_number": act_number}, {"year": year}]}
        filtered = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": where_filter},
        )
        if filtered.invoke(question):
            return filtered

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"content_role": "primary_consolidated_text"},
        },
    )
