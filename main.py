"""FastAPI application entry point."""
import uvicorn
from fastapi import FastAPI

from config import config
from src.api.endpoints import router
from src.db.models import init_db
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Trading Predictor API",
    description=(
        "A production-ready trading prediction system with technical analysis, "
        "ML predictions (UP/DOWN/NO_SIGNAL) and risk management."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def on_startup():
    logger.info("Initializing database …")
    init_db()
    logger.info("Trading Predictor API started on %s:%s", config.api.HOST, config.api.PORT)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Trading Predictor API is running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.HOST,
        port=config.api.PORT,
        reload=config.api.DEBUG,
    )
