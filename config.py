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


class AlpacaConfig:
    """Alpaca broker API settings."""
    API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    # Paper trading base URL (default); switch to live URL for real trading
    BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    PAPER_MODE: bool = os.getenv("ALPACA_PAPER_MODE", "true").lower() == "true"


class PaperTradingConfig:
    """Paper (simulated) trading settings."""
    INITIAL_CAPITAL: float = float(os.getenv("PAPER_INITIAL_CAPITAL", "10000.0"))
    MAX_OPEN_POSITIONS: int = int(os.getenv("PAPER_MAX_OPEN_POSITIONS", "5"))
    COMMISSION_RATE: float = float(os.getenv("PAPER_COMMISSION_RATE", "0.001"))


class BacktestConfig:
    """Historical backtesting parameters."""
    DEFAULT_START_DATE: str = os.getenv("BACKTEST_START_DATE", "2023-01-01")
    DEFAULT_END_DATE: str = os.getenv("BACKTEST_END_DATE", "2026-04-05")
    DEFAULT_INITIAL_CAPITAL: float = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000.0"))
    STOCK_SYMBOLS: list = [
        s.strip()
        for s in os.getenv(
            "BACKTEST_STOCK_SYMBOLS", "AAPL,MSFT,GOOGL,TSLA,AMZN,SPY,QQQ"
        ).split(",")
    ]
    CRYPTO_SYMBOLS: list = [
        s.strip()
        for s in os.getenv(
            "BACKTEST_CRYPTO_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT"
        ).split(",")
    ]
    TIMEFRAMES: list = [
        s.strip()
        for s in os.getenv("BACKTEST_TIMEFRAMES", "1d").split(",")
    ]


class LiveTradingConfig:
    """Live trading risk controls."""
    ENABLED: bool = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
    MAX_RISK_PER_TRADE_PCT: float = float(os.getenv("LIVE_MAX_RISK_PCT", "2.0"))
    DAILY_LOSS_LIMIT_PCT: float = float(os.getenv("LIVE_DAILY_LOSS_LIMIT_PCT", "5.0"))
    MAX_OPEN_POSITIONS: int = int(os.getenv("LIVE_MAX_OPEN_POSITIONS", "5"))
    REQUIRE_MANUAL_APPROVAL: bool = (
        os.getenv("LIVE_REQUIRE_MANUAL_APPROVAL", "true").lower() == "true"
    )


class SuccessMetrics:
    """Validation success thresholds."""
    MIN_WIN_RATE: float = float(os.getenv("MIN_WIN_RATE", "0.55"))
    MIN_PROFIT_FACTOR: float = float(os.getenv("MIN_PROFIT_FACTOR", "1.5"))
    MIN_SHARPE_RATIO: float = float(os.getenv("MIN_SHARPE_RATIO", "1.0"))
    MAX_DRAWDOWN_PCT: float = float(os.getenv("MAX_DRAWDOWN_PCT", "20.0"))
    MIN_RECOVERY_FACTOR: float = float(os.getenv("MIN_RECOVERY_FACTOR", "2.0"))


class Config:
    api = APIConfig()
    data = DataConfig()
    model = ModelConfig()
    risk = RiskConfig()
    alpaca = AlpacaConfig()
    paper = PaperTradingConfig()
    backtest = BacktestConfig()
    live = LiveTradingConfig()
    metrics = SuccessMetrics()


config = Config()
