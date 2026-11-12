import pytest
from fastapi import HTTPException

from app.core.rate_limit import rate_limit


class _Req:
    client = type("C", (), {"host": "1.2.3.4"})()


async def test_blocks_after_limit():
    dep = rate_limit("t", lambda: 2)
    await dep(_Req())
    await dep(_Req())
    with pytest.raises(HTTPException) as e:
        await dep(_Req())
    assert e.value.status_code == 429


async def test_zero_disables():
    dep = rate_limit("t", lambda: 0)
    for _ in range(5):
        await dep(_Req())
