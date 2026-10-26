"""Shared retry wrapper for Groq's free-tier rate limit.

Used by every eval script that calls the LLM across many gold-set
questions in a row (run_eval.py, ablation.py) -- enough real calls back
to back trips the free tier's tokens-per-minute limit (observed directly
in sub-phase 2.3: a real run hit a 429 after ~34 questions), so this
lives in one shared place instead of being copied into each script that
needs it.
"""

from __future__ import annotations

import time
from typing import Any

import groq


def invoke_with_retry(
    llm: Any, formatted: Any, max_retries: int = 5
) -> tuple[Any, float]:
    """Call llm.invoke(), retrying on Groq's free-tier rate limit.

    Groq's error response includes a "Retry-After" header with the exact
    wait time -- use that instead of guessing, falling back to a fixed
    delay only if it's ever missing.

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
