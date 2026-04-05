"""Auth middleware: rate limiting and optional JWT/API-key verification."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Simple in-process rate limiter (requests per minute per IP)
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_RPM: int = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed RATE_LIMIT_RPM per client IP."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0
        timestamps = _rate_limit_store[client_ip]
        # Purge old entries
        _rate_limit_store[client_ip] = [t for t in timestamps if t > window_start]
        if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_RPM:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return Response(
                content='{"detail":"Too Many Requests"}',
                status_code=429,
                media_type="application/json",
            )
        _rate_limit_store[client_ip].append(now)
        return await call_next(request)
