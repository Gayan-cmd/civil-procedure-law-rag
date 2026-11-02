from __future__ import annotations

import psycopg
import pytest

from api.users import create_user, get_user_by_email, get_user_by_id
from conftest import TEST_DATABASE_URL


def test_create_user_persists_email_and_hash(migrated_db: None) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        user = create_user(conn, "alice@example.com", "some-hash")

    assert user.id is not None
    assert user.email == "alice@example.com"
    assert user.password_hash == "some-hash"


def test_create_user_rejects_a_duplicate_email(migrated_db: None) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        create_user(conn, "bob@example.com", "hash-one")

        with pytest.raises(psycopg.errors.UniqueViolation):
            create_user(conn, "bob@example.com", "hash-two")


def test_get_user_by_email_finds_an_existing_user(migrated_db: None) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        created = create_user(conn, "carol@example.com", "some-hash")

        found = get_user_by_email(conn, "carol@example.com")

    assert found == created


def test_get_user_by_email_returns_none_for_an_unknown_address(
    migrated_db: None,
) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        assert get_user_by_email(conn, "nobody@example.com") is None


def test_get_user_by_id_finds_an_existing_user(migrated_db: None) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        created = create_user(conn, "dave@example.com", "some-hash")

        found = get_user_by_id(conn, created.id)

    assert found == created


def test_get_user_by_id_returns_none_for_an_unknown_id(migrated_db: None) -> None:
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        assert get_user_by_id(conn, 999999) is None
