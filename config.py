# Configuration classes

class APIConfig:
    BASE_URL = 'https://api.example.com'
    TIMEOUT = 30

class DataConfig:
    DATA_SOURCE = 'data_source'
    DATA_PATH = 'data/data.csv'

class ModelConfig:
    MODEL_PATH = 'models/model.pkl'
    INPUT_FEATURES = ['feature1', 'feature2', 'feature3']

class LeverageConfig:
    # Default risk parameters
    DEFAULT_RISK_PERCENT = 2.0          # % of account risked per trade
    DEFAULT_STOP_LOSS_PERCENT = 5.0     # % distance to stop-loss from entry
    DEFAULT_DAILY_LOSS_LIMIT_PCT = 5.0  # % max daily loss

    # Hard caps
    MAX_LEVERAGE = 20                   # absolute maximum allowed
    MIN_ACCOUNT_SIZE = 10.0             # minimum account size in USD

    # Safety thresholds
    MAX_DRAWDOWN_THRESHOLD = 0.25       # 25% drawdown triggers warning
    LIQUIDATION_RISK_CRITICAL = 40.0    # % liquidation risk → CRITICAL
    LIQUIDATION_RISK_WARNING = 20.0     # % liquidation risk → WARNING

    # Model paths (relative to project root)
    MODEL_DIR = 'src/models'
    MODEL_FILE = 'leverage_model.pkl'
    SCALER_FILE = 'leverage_scaler.pkl'
