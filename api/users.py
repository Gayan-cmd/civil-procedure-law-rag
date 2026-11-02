"""Plain SQL access to the users table -- same choice of psycopg over an
ORM that store.py already made, for the same reason: a schema and its
queries should be something you can read directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class User:
    id: int
    email: str
    password_hash: str


def create_user(conn: psycopg.Connection, email: str, password_hash: str) -> User:
    """Raises psycopg.errors.UniqueViolation if the email is already
    registered -- the users.email UNIQUE constraint is the real guarantee
    here, not a pre-check in Python, which would still race under
    concurrent signups for the same address.
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s) "
            "RETURNING id, email, password_hash",
            (email, password_hash),
        )
        row = cur.fetchone()
        conn.commit()
    assert row is not None
    return User(id=row[0], email=row[1], password_hash=row[2])


def get_user_by_email(conn: psycopg.Connection, email: str) -> User | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s", (email,)
        )
        row = cur.fetchone()
    return User(id=row[0], email=row[1], password_hash=row[2]) if row else None


def get_user_by_id(conn: psycopg.Connection, user_id: int) -> User | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, password_hash FROM users WHERE id = %s", (user_id,)
        )
        row = cur.fetchone()
    return User(id=row[0], email=row[1], password_hash=row[2]) if row else None
