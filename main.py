"""FastAPI application entry point."""
from __future__ import annotations

import time
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import config
from src.api.endpoints import router
from src.auth.middleware import RateLimitMiddleware
from src.db.models import init_db
from src.utils.logger import get_logger
from src.websocket.manager import ConnectionManager

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response as FastAPIResponse

    _PROMETHEUS_AVAILABLE = True
    REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint"])
    REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency")
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False

logger = get_logger(__name__)
ws_manager = ConnectionManager()

app = FastAPI(
    title="Trading Predictor API",
    description=(
        "A production-ready trading prediction system with technical analysis, "
        "ML predictions (UP/DOWN/NO_SIGNAL) and risk management."
    ),
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.add_middleware(RateLimitMiddleware)

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Initializing database …")
    init_db()
    logger.info("Trading Predictor API started on %s:%s", config.api.HOST, config.api.PORT)


@app.on_event("shutdown")
def on_shutdown() -> None:
    logger.info("Trading Predictor API shutting down.")


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Trading Predictor API is running"}


@app.get("/health", tags=["Health"])
def health() -> dict[str, Any]:
    """Health check with DB connectivity status."""
    db_status = "ok"
    try:
        from sqlalchemy import text
        from src.db.models import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"

    return {
        "status": "healthy" if db_status == "ok" else "degraded",
        "db": db_status,
        "websocket_connections": ws_manager.connection_count,
    }


if _PROMETHEUS_AVAILABLE:

    @app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
    def metrics() -> FastAPIResponse:  # type: ignore[misc]
        return FastAPIResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket) -> None:
    """Real-time trading signal updates over WebSocket."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await ws_manager.send_personal({"echo": data, "ts": time.time()}, websocket)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.HOST,
        port=config.api.PORT,
        reload=config.api.DEBUG,
    )
