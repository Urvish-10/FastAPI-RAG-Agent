import pytest
from fastapi import HTTPException

from app.core.security import (
    create_access_token, decode_token, hash_password, verify_password,
)


def test_password_roundtrip():
    h = hash_password("hunter2hunter2")
    assert verify_password("hunter2hunter2", h)
    assert not verify_password("wrong", h)


def test_token_roundtrip():
    claims = decode_token(create_access_token("uid-1", "a@b.com"))
    assert claims["sub"] == "uid-1" and claims["email"] == "a@b.com"


def test_bad_token_rejected():
    with pytest.raises(HTTPException) as e:
        decode_token("not.a.token")
    assert e.value.status_code == 401


def test_short_secret_rejected():
    from app.core.config import Settings
    with pytest.raises(ValueError):
        Settings(SECRET_KEY="short")
