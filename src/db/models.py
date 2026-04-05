from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import config

engine = create_engine(
    config.data.DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in config.data.DB_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    asset_type = Column(String, nullable=False)  # 'stock' or 'crypto'
    prediction = Column(String, nullable=False)  # 'UP', 'DOWN', 'NO_SIGNAL'
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    max_loss = Column(Float, nullable=True)
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    direction = Column(String, nullable=False)  # 'LONG' or 'SHORT'
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    position_size = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    outcome = Column(String, nullable=True)  # 'WIN', 'LOSS', 'OPEN'
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
