"""Auth Pydantic models."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
