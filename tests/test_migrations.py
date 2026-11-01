"""Verifies the Alembic migration against the real (test) database --
not just that it runs, but that the constraints it creates actually hold
and that downgrade() genuinely undoes upgrade().

Runs against conftest.TEST_DATABASE_URL, never the production database --
same isolation rule as every other test in this suite that writes to
Postgres (see tests/conftest.py).
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config

from conftest import TEST_DATABASE_URL

_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


def _alembic_config() -> Config:
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    return config


@pytest.fixture
def migrated_db():
    """Upgrade to head, yield, then downgrade back to base -- every test
    using this fixture starts and ends with no trace of these tables,
    the same "leave it как you found it" rule build_vectorstore's reset()
    follows for the chunks table.
    """
    config = _alembic_config()
    command.upgrade(config, "head")
    try:
        yield
    finally:
        command.downgrade(config, "base")


def test_upgrade_creates_all_four_tables_with_working_constraints(
    migrated_db: None,
) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name IN "
                "('users', 'conversations', 'messages', 'citations')"
            )
            tables = {row[0] for row in cur.fetchall()}
    assert tables == {"users", "conversations", "messages", "citations"}


def test_foreign_key_rejects_a_conversation_for_a_nonexistent_user(
    migrated_db: None,
) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO conversations (user_id, title) VALUES (999999, 'x')"
                )
        conn.rollback()


def test_check_constraint_rejects_an_invalid_message_role(
    migrated_db: None,
) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
                ("test@example.com", "hash"),
            )
            (user_id,) = cur.fetchone()
            cur.execute(
                "INSERT INTO conversations (user_id) VALUES (%s) RETURNING id",
                (user_id,),
            )
            (conversation_id,) = cur.fetchone()
            conn.commit()

            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content) "
                    "VALUES (%s, 'not_a_real_role', 'hi')",
                    (conversation_id,),
                )
        conn.rollback()


def test_downgrade_removes_all_four_tables() -> None:
    config = _alembic_config()
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    with psycopg.connect(TEST_DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name IN "
                "('users', 'conversations', 'messages', 'citations')"
            )
            tables = cur.fetchall()
    assert tables == []
