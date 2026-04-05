"""WebSocket connection manager for real-time price/signal updates."""
from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._active.append(websocket)
        logger.info("WebSocket connected; total=%d", len(self._active))

    def disconnect(self, websocket: WebSocket) -> None:
        self._active.remove(websocket)
        logger.info("WebSocket disconnected; total=%d", len(self._active))

    async def send_personal(self, data: Any, websocket: WebSocket) -> None:
        await websocket.send_text(json.dumps(data))

    async def broadcast(self, data: Any) -> None:
        payload = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._active)
