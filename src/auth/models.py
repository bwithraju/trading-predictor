"""Pydantic models for authentication."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    is_admin: bool = False


class UserInDB(User):
    hashed_password: str


class APIKey(BaseModel):
    key: str = Field(..., description="API key value")
    name: str = Field(..., description="Human-readable label")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True


class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
