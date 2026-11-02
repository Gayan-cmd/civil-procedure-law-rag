"""FastAPI application entrypoint.

Run locally with:
    fastapi dev api/main.py

This module, and everything under api/, is the only place in the repo
allowed to import FastAPI/Starlette -- src/cpa/ stays a plain importable
package (see its own module docstrings), callable from a script, the
ingestion job, or the eval harness with no HTTP framework involved at all.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routers import auth, health

app = FastAPI(title="Civil Procedure Law Assistant API")

app.include_router(health.router)
app.include_router(auth.router)
