"""Password hashing and JWT issuing/verification for the auth endpoints.

CPA_JWT_SECRET_KEY is read lazily, straight from the environment, not
through cpa.config.Settings -- same reasoning generation.py already
applies to GROQ_API_KEY: a secret only one specific piece of
functionality needs shouldn't be a required field on the one Settings
object every unrelated script and test also constructs.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Literal

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

import cpa.config  # noqa: F401  -- import side effect: loads .env into os.environ

ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30
JWT_ALGORITHM = "HS256"

_hasher = PasswordHasher()


class InvalidToken(Exception):
    """Raised by decode_token for any reason a token should be rejected --
    bad signature, expired, or the right kind of token used in the wrong
    place (a refresh token presented where an access token belongs).
    Callers map this to a single 401 without needing to know which case
    it was.
    """


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def _secret_key() -> str:
    key = os.environ.get("CPA_JWT_SECRET_KEY")
    if not key:
        raise RuntimeError("CPA_JWT_SECRET_KEY must be set to issue or verify tokens")
    return key


def _create_token(
    user_id: int, token_type: Literal["access", "refresh"], expires_delta: timedelta
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, _secret_key(), algorithm=JWT_ALGORITHM)


def create_access_token(user_id: int) -> str:
    return _create_token(
        user_id, "access", timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )


def create_refresh_token(user_id: int) -> str:
    return _create_token(user_id, "refresh", timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))


def decode_token(token: str, expected_type: Literal["access", "refresh"]) -> int:
    """Returns the user id if the token is valid and of the expected type.

    The "type" claim is what stops a refresh token (same secret, same
    signature scheme, long-lived) from being usable directly as an access
    token -- without it, stealing a refresh token would be as good as
    stealing an access token, for thirty days instead of fifteen minutes.
    """
    try:
        payload = jwt.decode(token, _secret_key(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc

    if payload.get("type") != expected_type:
        raise InvalidToken(f"expected a {expected_type} token")

    return int(payload["sub"])
