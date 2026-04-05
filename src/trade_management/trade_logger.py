"""Trade journaling: log every trade with entry/exit reasons."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeEntry:
    trade_id: str
    symbol: str
    direction: str          # 'LONG' or 'SHORT'
    asset_type: str         # 'stock' or 'crypto'
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    entry_reason: str
    entry_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    outcome: Optional[str] = None  # 'WIN', 'LOSS', 'OPEN'
    mode: str = "paper"     # 'paper' or 'live'
    tags: List[str] = field(default_factory=list)


class TradeLogger:
    """Persist trade entries to a JSON-lines journal file."""

    def __init__(self, journal_path: str = "trade_journal.jsonl"):
        self.journal_path = Path(journal_path)

    def log_entry(self, trade: TradeEntry) -> None:
        """Append a new trade entry to the journal."""
        with self.journal_path.open("a") as fh:
            fh.write(json.dumps(asdict(trade)) + "\n")
        logger.info("Trade logged: %s %s %s", trade.mode.upper(), trade.direction, trade.symbol)

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
    ) -> bool:
        """Update an existing trade entry with exit information."""
        trades = self.load_all()
        updated = False
        for trade in trades:
            if trade["trade_id"] == trade_id:
                trade["exit_price"] = exit_price
                trade["exit_time"] = datetime.utcnow().isoformat()
                trade["exit_reason"] = exit_reason
                trade["pnl"] = round(pnl, 2)
                trade["pnl_pct"] = round(pnl_pct, 4)
                trade["outcome"] = "WIN" if pnl > 0 else "LOSS"
                updated = True
                break

        if updated:
            with self.journal_path.open("w") as fh:
                for trade in trades:
                    fh.write(json.dumps(trade) + "\n")
            logger.info("Trade exit logged: %s pnl=%.2f", trade_id, pnl)
        return updated

    def load_all(self) -> List[dict]:
        """Load all trade entries from the journal."""
        if not self.journal_path.exists():
            return []
        trades = []
        with self.journal_path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return trades

    def load_open_trades(self) -> List[dict]:
        """Return only trades that are still open (no exit_price)."""
        return [t for t in self.load_all() if t.get("outcome") is None]

    def load_closed_trades(self) -> List[dict]:
        """Return only closed trades."""
        return [t for t in self.load_all() if t.get("outcome") is not None]
