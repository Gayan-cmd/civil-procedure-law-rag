from __future__ import annotations

import os

import pytest

from cpa.generation import get_llm, get_prompt


def test_prompt_has_no_answer_label_to_leak() -> None:
    prompt = get_prompt()
    formatted = prompt.format_messages(context="some context", question="a question?")

    assert len(formatted) == 2
    assert formatted[0].type == "system"
    assert formatted[1].type == "human"
    # The notebook's completion-style prompt ended in a literal "Answer:"
    # label that a small model could echo back. A chat prompt has no such
    # label in the first place.
    assert "Answer:" not in formatted[1].content


def test_prompt_fills_in_context_and_question() -> None:
    prompt = get_prompt()
    formatted = prompt.format_messages(
        context="Section 75: file within 21 days.",
        question="What is the time limit?",
    )

    human_message = formatted[1].content
    assert "Section 75: file within 21 days." in human_message
    assert "What is the time limit?" in human_message


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)
def test_llm_answers_from_context_only() -> None:
    chain = get_prompt() | get_llm()

    result = chain.invoke(
        {
            "context": "Section 75: An answer must be filed within 21 days"
            " of service of the plaint.",
            "question": "What is the time limit for filing an answer?",
        }
    )

    assert "21 days" in result.content
    assert "Answer:" not in result.content
