"""Minimal in-process sliding-window rate limiter (per client IP).

Good enough for a single instance; use Redis-backed limiting (e.g. slowapi)
if you run multiple API replicas.
"""
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import HTTPException, Request, status


def rate_limit(name: str, limit_getter: Callable[[], int], window_s: int = 60):
    hits: Dict[str, Deque[float]] = defaultdict(deque)

    async def dependency(request: Request) -> None:
        limit = limit_getter()
        if limit <= 0:
            return
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        q = hits[key]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {name}",
                headers={"Retry-After": str(window_s)},
            )
        q.append(now)

    return dependency
