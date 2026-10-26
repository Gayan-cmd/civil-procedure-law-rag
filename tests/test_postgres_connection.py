from __future__ import annotations

import psycopg
import pytest

from cpa.config import get_settings


def _can_connect(database_url: str) -> bool:
    try:
        with psycopg.connect(database_url, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


_SETTINGS = get_settings()
_REACHABLE = _can_connect(_SETTINGS.database_url)


@pytest.mark.skipif(
    not _REACHABLE,
    reason="Postgres not reachable at CPA_DATABASE_URL -- is the container running?",
)
def test_vector_extension_is_installed() -> None:
    with psycopg.connect(_SETTINGS.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()

    assert row is not None
    assert row[0] == "vector"
