"""Shared FastAPI dependencies -- right now just the database connection.
Grows in 4.4+ to include whichever dependency resolves the authenticated
user from an access token.
"""

from __future__ import annotations

from collections.abc import Generator

import psycopg
from fastapi import Depends

from cpa.config import Settings, get_settings


def get_db_connection(
    settings: Settings = Depends(get_settings),
) -> Generator[psycopg.Connection, None, None]:
    with psycopg.connect(settings.database_url) as conn:
        yield conn
