import os
from dotenv import load_dotenv

load_dotenv()


class APIConfig:
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))


class DataConfig:
    # yfinance settings
    DEFAULT_STOCK_INTERVAL: str = os.getenv("STOCK_INTERVAL", "1d")
    DEFAULT_STOCK_PERIOD: str = os.getenv("STOCK_PERIOD", "1y")

    # Binance / CCXT settings
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")
    DEFAULT_CRYPTO_EXCHANGE: str = os.getenv("CRYPTO_EXCHANGE", "binance")
    DEFAULT_CRYPTO_TIMEFRAME: str = os.getenv("CRYPTO_TIMEFRAME", "1d")
    DEFAULT_CRYPTO_LIMIT: int = int(os.getenv("CRYPTO_LIMIT", "365"))

    # Storage
    DB_URL: str = os.getenv("DB_URL", "sqlite:///trading_predictor.db")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "cache")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))


class ModelConfig:
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    MIN_INDICATOR_ALIGNMENT: int = int(os.getenv("MIN_INDICATOR_ALIGNMENT", "3"))
    TRAIN_TEST_SPLIT: float = float(os.getenv("TRAIN_TEST_SPLIT", "0.8"))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    N_ESTIMATORS: int = int(os.getenv("N_ESTIMATORS", "100"))


class RiskConfig:
    DEFAULT_RISK_PERCENT: float = float(os.getenv("DEFAULT_RISK_PERCENT", "2.0"))
    DEFAULT_REWARD_RATIO: float = float(os.getenv("DEFAULT_REWARD_RATIO", "2.0"))
    SL_PERCENT_LONG: float = float(os.getenv("SL_PERCENT_LONG", "2.0"))
    SL_PERCENT_SHORT: float = float(os.getenv("SL_PERCENT_SHORT", "2.0"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.20"))


class Config:
    api = APIConfig()
    data = DataConfig()
    model = ModelConfig()
    risk = RiskConfig()


config = Config()
