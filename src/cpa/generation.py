"""LLM wrapper: a hosted chat model behind a LangChain Runnable.

Uses Groq's API rather than running a model locally. BUILD_PLAN.md's own
Phase 4 already calls for this swap in production ("Replace Qwen 0.5B --
half a billion parameters cannot reliably follow a refusal instruction"),
and a local 0.5B model needs ~2GB+ of RAM just for its weights on CPU --
enough to be a real constraint on a development machine. Pulling that swap
forward avoids fighting local memory for a model that was getting replaced
regardless.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from cpa.config import Settings, get_settings

SYSTEM_PROMPT = """You are a legal research assistant specializing in Sri Lankan civil procedure law.
Use ONLY the information in the context below to answer the question.

Rules:
- Every factual claim must cite its source act by name (e.g. "under the Civil Procedure Code (Amendment) Act No. 43 of 2024").
- If the context includes a section number, cite it. If it doesn't, say the section number isn't specified in the retrieved text -- do not invent one.
- If the answer is not in the context, say "I don't have that information in my knowledge base."
- Keep your answer clear and concise."""


def get_prompt() -> ChatPromptTemplate:
    """Build the RAG prompt as chat turns, not one string ending in "Answer:".

    The notebook's version was a single completion-style string with a
    literal "Answer:" label at the end, which a small local model would
    sometimes echo back -- patched over with `answer.split("Answer:")[-1]`.
    A real chat API takes system/user turns directly and returns only the
    assistant's reply, so there's no label for it to echo and nothing to
    strip out afterward.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )


def get_llm(settings: Settings | None = None) -> ChatGroq:
    """Build the chat model. Reads the API key from the GROQ_API_KEY env var.

    Temperature 0 for the same reason the notebook used do_sample=False:
    deterministic, repeatable answers over a legal corpus, not creative
    variation.
    """
    settings = settings or get_settings()
    return ChatGroq(model=settings.llm_model, temperature=0)
