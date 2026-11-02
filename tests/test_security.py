from __future__ import annotations

import jwt
import pytest

from api.security import (
    InvalidToken,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_hash_password_does_not_store_the_plaintext() -> None:
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"


def test_verify_password_accepts_the_right_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed) is True


def test_verify_password_rejects_the_wrong_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password("wrong password", hashed) is False


def test_access_and_refresh_tokens_round_trip_to_the_same_user_id() -> None:
    access = create_access_token(42)
    refresh = create_refresh_token(42)

    assert decode_token(access, expected_type="access") == 42
    assert decode_token(refresh, expected_type="refresh") == 42


def test_an_access_token_cannot_be_used_as_a_refresh_token() -> None:
    access = create_access_token(42)

    with pytest.raises(InvalidToken):
        decode_token(access, expected_type="refresh")


def test_a_refresh_token_cannot_be_used_as_an_access_token() -> None:
    refresh = create_refresh_token(42)

    with pytest.raises(InvalidToken):
        decode_token(refresh, expected_type="access")


def test_decode_token_rejects_an_expired_token(monkeypatch) -> None:
    import api.security as security

    monkeypatch.setattr(security, "ACCESS_TOKEN_EXPIRE_MINUTES", -1)
    expired = create_access_token(42)

    with pytest.raises(InvalidToken):
        decode_token(expired, expected_type="access")


def test_decode_token_rejects_a_token_signed_with_a_different_secret() -> None:
    forged = jwt.encode(
        {"sub": "42", "type": "access"},
        "a-completely-different-secret-32-bytes-long",
        algorithm="HS256",
    )

    with pytest.raises(InvalidToken):
        decode_token(forged, expected_type="access")
