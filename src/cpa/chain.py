"""Retrieval + generation wired together, retrieving exactly once per question."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStore

from cpa.config import Settings, get_settings
from cpa.generation import get_llm, get_prompt
from cpa.retrieval import get_retriever


class RagResult(TypedDict):
    answer: str
    documents: list[Document]


def format_docs(documents: list[Document]) -> str:
    """Render retrieved chunks as one string, each labeled with its source."""
    formatted = []
    for doc in documents:
        title = doc.metadata.get("title", "Unknown source")
        formatted.append(f"[Source: {title}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_chain(
    vectorstore: VectorStore, settings: Settings | None = None
) -> Runnable[str, RagResult]:
    """Build the RAG chain: retrieves once per question, returns both the
    generated answer and the documents it was generated from.

    The notebook's ask() ran retrieval inside the chain to build the answer,
    then ran it a SECOND time afterward just to display sources -- doubling
    retrieval latency, and risking the two calls returning different chunks
    if the store changed in between. This fans out once (RunnableParallel),
    keeping the retrieved documents alongside the generated answer instead
    of discarding and re-fetching them.
    """
    settings = settings or get_settings()
    prompt = get_prompt()
    llm = get_llm(settings)

    retrieve_documents: Runnable[str, list[Document]] = RunnableLambda(
        lambda question: get_retriever(vectorstore, question, settings).invoke(
            question
        )
    )

    generate_answer = (
        RunnableLambda(
            lambda inputs: {
                "context": format_docs(inputs["documents"]),
                "question": inputs["question"],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    fan_out_retrieval: Runnable[str, dict[str, Any]] = RunnableParallel(
        documents=retrieve_documents,
        question=RunnablePassthrough(),
    )

    return fan_out_retrieval | RunnableParallel(
        answer=generate_answer,
        documents=lambda inputs: inputs["documents"],
    )


def ask(
    vectorstore: VectorStore, question: str, settings: Settings | None = None
) -> RagResult:
    """Convenience wrapper: build the chain and answer one question with it.

    Builds a fresh chain per call -- cheap, since get_llm() just constructs
    an API client rather than loading a model. Prefer build_chain() directly
    when answering many questions (an eval run, an API server) so the chain
    is only assembled once.
    """
    chain = build_chain(vectorstore, settings)
    return chain.invoke(question)
