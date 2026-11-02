"""POST /auth/signup, POST /auth/login, POST /auth/refresh."""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_db_connection
from api.schemas import (
    LoginRequest,
    RefreshRequest,
    SignupRequest,
    SignupResponse,
    TokenResponse,
)
from api.security import (
    InvalidToken,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from api.users import create_user, get_user_by_email, get_user_by_id

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=SignupResponse, status_code=201)
def signup(
    body: SignupRequest, conn: psycopg.Connection = Depends(get_db_connection)
) -> SignupResponse:
    email = body.email.lower()
    try:
        user = create_user(conn, email, hash_password(body.password))
    except psycopg.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=409, detail="email already registered")

    return SignupResponse(id=user.id, email=user.email)


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest, conn: psycopg.Connection = Depends(get_db_connection)
) -> TokenResponse:
    user = get_user_by_email(conn, body.email.lower())
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="incorrect email or password")

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    body: RefreshRequest, conn: psycopg.Connection = Depends(get_db_connection)
) -> TokenResponse:
    try:
        user_id = decode_token(body.refresh_token, expected_type="refresh")
    except InvalidToken:
        raise HTTPException(status_code=401, detail="invalid refresh token")

    if get_user_by_id(conn, user_id) is None:
        # The user existed when this refresh token was issued but doesn't
        # now -- treat it the same as an invalid token rather than a
        # different error, so this endpoint never confirms or denies
        # which user ids exist.
        raise HTTPException(status_code=401, detail="invalid refresh token")

    return TokenResponse(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
    )
