"""GET /health -- the one endpoint BUILD_PLAN.md's Phase 4 lists as a plain
liveness probe, used here to also prove the API layer can actually reach
the same Postgres database cpa.store talks to, via cpa.config's Settings.

Deliberately does not use PgVectorStore (which would load the embedding
model) -- a health check should be cheap enough to hit frequently.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from cpa.config import Settings, get_settings

router = APIRouter()


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except psycopg.OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="database unreachable"
        ) from exc

    return {"status": "ok", "database": "reachable"}
