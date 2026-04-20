"""Walk-forward backtesting engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import config
from src.analysis.signals import SignalGenerator
from src.models.features import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.risk.calculator import RiskCalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)

signal_gen = SignalGenerator()
fe = FeatureEngineer()


@dataclass
class TradeLog:
    entry_index: int
    entry_price: float
    direction: str
    stop_loss: float
    take_profit: float
    exit_price: Optional[float] = None
    exit_index: Optional[int] = None
    outcome: str = "OPEN"  # 'WIN', 'LOSS', 'OPEN'
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    total_return_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[TradeLog] = field(default_factory=list)


class BacktestEngine:
    """Simple walk-forward backtester using the trained ML model + signals."""

    def __init__(
        self,
        lookback: int = 50,
        confidence_threshold: float = None,
        min_indicator_alignment: int = None,
    ):
        self.lookback = lookback
        self.confidence_threshold = confidence_threshold or config.model.CONFIDENCE_THRESHOLD
        self.min_indicator_alignment = min_indicator_alignment or config.model.MIN_INDICATOR_ALIGNMENT

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        risk_percent: float = 2.0,
        account_size: float = 10_000.0,
    ) -> BacktestResult:
        """Run a walk-forward backtest on *df*.

        For each bar (after the warm-up *lookback* period) the model is
        queried with the preceding *lookback* bars.  When a signal is
        generated a simulated trade is opened and tracked until SL or TP
        is hit.
        """
        logger.info("Starting backtest for %s (%d bars)", symbol, len(df))
        trainer = ModelTrainer()
        if not trainer.load_model(symbol):
            # Train on the first 70 % of data before backtesting the rest
            train_end = int(len(df) * 0.7)
            try:
                trainer.train(df.iloc[:train_end], symbol)
            except ValueError as exc:
                logger.warning("Could not train model for backtest: %s", exc)

        trades: List[TradeLog] = []
        equity_curve: List[float] = [account_size]
        current_equity = account_size

        for i in range(self.lookback, len(df)):
            window = df.iloc[max(0, i - self.lookback): i]
            signals = signal_gen.generate_from_df(window)

            # ML prediction
            ml_dir, confidence = self._ml_predict(window, trainer)
            if ml_dir == "NONE" or confidence < self.confidence_threshold:
                continue

            aligned = signals.get("bull_count", 0) if ml_dir == "UP" else signals.get("bear_count", 0)
            if aligned < self.min_indicator_alignment:
                continue

            direction = "LONG" if ml_dir == "UP" else "SHORT"
            entry_price = float(df["close"].iloc[i])
            risk_calc = RiskCalculator()
            risk = risk_calc.calculate(
                entry_price=entry_price,
                direction=direction,
                risk_percent=risk_percent,
                account_size=current_equity,
            )

            # Simulate forward bars to find exit
            trade = TradeLog(
                entry_index=i,
                entry_price=entry_price,
                direction=direction,
                stop_loss=risk.stop_loss,
                take_profit=risk.take_profit,
            )

            exit_index = min(i + 20, len(df) - 1)
            for j in range(i + 1, min(i + 20, len(df))):
                bar_high = float(df["high"].iloc[j])
                bar_low = float(df["low"].iloc[j])
                if direction == "LONG":
                    if bar_low <= risk.stop_loss:
                        trade.exit_price = risk.stop_loss
                        trade.outcome = "LOSS"
                        exit_index = j
                        break
                    if bar_high >= risk.take_profit:
                        trade.exit_price = risk.take_profit
                        trade.outcome = "WIN"
                        exit_index = j
                        break
                else:  # SHORT
                    if bar_high >= risk.stop_loss:
                        trade.exit_price = risk.stop_loss
                        trade.outcome = "LOSS"
                        exit_index = j
                        break
                    if bar_low <= risk.take_profit:
                        trade.exit_price = risk.take_profit
                        trade.outcome = "WIN"
                        exit_index = j
                        break

            if trade.exit_price is None:
                trade.exit_price = float(df["close"].iloc[exit_index])
                trade.outcome = "OPEN"

            trade.exit_index = exit_index
            if direction == "LONG":
                trade.pnl_pct = (trade.exit_price - entry_price) / entry_price * 100
            else:
                trade.pnl_pct = (entry_price - trade.exit_price) / entry_price * 100

            pnl_abs = current_equity * trade.pnl_pct / 100
            current_equity += pnl_abs
            equity_curve.append(current_equity)
            trades.append(trade)

        return self._compute_metrics(symbol, trades, equity_curve, account_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ml_predict(window: pd.DataFrame, trainer: ModelTrainer) -> tuple[str, float]:
        if trainer.model is None:
            return "NONE", 0.0
        X = fe.get_feature_matrix(window)
        if X.empty:
            return "NONE", 0.0
        available = [f for f in trainer.feature_names if f in X.columns]
        if not available:
            return "NONE", 0.0
        X = X[available].iloc[[-1]]
        proba = trainer.model.predict_proba(X)[0]
        classes = list(trainer.model.classes_)
        up_conf = proba[classes.index(1)] if 1 in classes else 0.0
        down_conf = proba[classes.index(0)] if 0 in classes else 0.0
        if up_conf >= down_conf:
            return "UP", float(up_conf)
        return "DOWN", float(down_conf)

    @staticmethod
    def _compute_metrics(
        symbol: str,
        trades: List[TradeLog],
        equity_curve: List[float],
        initial_equity: float,
    ) -> BacktestResult:
        total = len(trades)
        if total == 0:
            return BacktestResult(
                symbol=symbol,
                total_trades=0,
                win_trades=0,
                loss_trades=0,
                win_rate=0.0,
                total_return_pct=0.0,
                profit_factor=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                trades=[],
            )

        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        win_rate = len(wins) / total

        gross_profit = sum(t.pnl_pct for t in wins)
        gross_loss = abs(sum(t.pnl_pct for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        final_equity = equity_curve[-1]
        total_return_pct = (final_equity - initial_equity) / initial_equity * 100

        # Max drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak * 100
        max_drawdown = float(drawdown.max())

        # Sharpe (annualised, assume daily bars, 252 trading days)
        returns = np.diff(eq) / eq[:-1]
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252))

        return BacktestResult(
            symbol=symbol,
            total_trades=total,
            win_trades=len(wins),
            loss_trades=len(losses),
            win_rate=round(win_rate, 4),
            total_return_pct=round(total_return_pct, 2),
            profit_factor=round(profit_factor, 4),
            max_drawdown_pct=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe, 4),
            trades=trades,
        )
