"""End-to-end tests for /auth/signup, /auth/login, /auth/refresh --
through the real FastAPI app and a real (test) database, not by calling
api/users.py or api/security.py directly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from cpa.config import Settings, get_settings
from conftest import TEST_DATABASE_URL


@pytest.fixture
def client(migrated_db: None) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_url=TEST_DATABASE_URL
    )
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_signup_creates_a_user_and_never_returns_the_password_hash(
    client: TestClient,
) -> None:
    response = client.post(
        "/auth/signup", json={"email": "new@example.com", "password": "a-long-password"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new@example.com"
    assert "password" not in body
    assert "password_hash" not in body


def test_signup_rejects_an_email_already_in_use(client: TestClient) -> None:
    client.post(
        "/auth/signup", json={"email": "dup@example.com", "password": "a-long-password"}
    )

    response = client.post(
        "/auth/signup", json={"email": "dup@example.com", "password": "another-password"}
    )

    assert response.status_code == 409


def test_signup_is_case_insensitive_on_email(client: TestClient) -> None:
    client.post(
        "/auth/signup", json={"email": "Case@Example.com", "password": "a-long-password"}
    )

    response = client.post(
        "/auth/signup", json={"email": "case@example.com", "password": "another-password"}
    )

    assert response.status_code == 409


def test_signup_rejects_a_short_password(client: TestClient) -> None:
    response = client.post(
        "/auth/signup", json={"email": "short@example.com", "password": "short"}
    )

    assert response.status_code == 422


def test_login_with_correct_credentials_returns_both_tokens(
    client: TestClient,
) -> None:
    client.post(
        "/auth/signup", json={"email": "login@example.com", "password": "a-long-password"}
    )

    response = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "a-long-password"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["token_type"] == "bearer"


def test_login_with_wrong_password_is_rejected(client: TestClient) -> None:
    client.post(
        "/auth/signup", json={"email": "login2@example.com", "password": "a-long-password"}
    )

    response = client.post(
        "/auth/login", json={"email": "login2@example.com", "password": "wrong-password"}
    )

    assert response.status_code == 401


def test_login_with_unknown_email_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/auth/login", json={"email": "nobody@example.com", "password": "whatever"}
    )

    assert response.status_code == 401


def test_refresh_with_a_valid_refresh_token_returns_new_tokens(
    client: TestClient,
) -> None:
    client.post(
        "/auth/signup", json={"email": "refresh@example.com", "password": "a-long-password"}
    )
    login_response = client.post(
        "/auth/login", json={"email": "refresh@example.com", "password": "a-long-password"}
    )
    refresh_token = login_response.json()["refresh_token"]

    response = client.post("/auth/refresh", json={"refresh_token": refresh_token})

    assert response.status_code == 200
    assert response.json()["access_token"]


def test_refresh_rejects_an_access_token_used_in_place_of_a_refresh_token(
    client: TestClient,
) -> None:
    client.post(
        "/auth/signup", json={"email": "wrongtoken@example.com", "password": "a-long-password"}
    )
    login_response = client.post(
        "/auth/login", json={"email": "wrongtoken@example.com", "password": "a-long-password"}
    )
    access_token = login_response.json()["access_token"]

    response = client.post("/auth/refresh", json={"refresh_token": access_token})

    assert response.status_code == 401


def test_refresh_rejects_garbage(client: TestClient) -> None:
    # FastAPI resolves the route's DB-connection dependency before the
    # route body runs, even though a garbage token never reaches the
    # point of using it -- so this still goes through the `client`
    # fixture rather than a bare TestClient(app), to stay on the test
    # database like every other test here.
    response = client.post("/auth/refresh", json={"refresh_token": "not-a-real-token"})

    assert response.status_code == 401
