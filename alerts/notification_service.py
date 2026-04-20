"""Alert notification service."""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)

_alert_file_handler = logging.FileHandler("logs/alerts.log")
_alert_file_handler.setLevel(logging.INFO)
_alert_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

_alerts_logger = logging.getLogger("alerts")
_alerts_logger.setLevel(logging.INFO)
_alerts_logger.addHandler(_alert_file_handler)


class AlertType(str, Enum):
    DAILY_LOSS = "DAILY_LOSS"
    UNUSUAL_ACTIVITY = "UNUSUAL_ACTIVITY"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    PERFORMANCE_MILESTONE = "PERFORMANCE_MILESTONE"


class NotificationService:
    """Handles alert notifications via logging, optional email, and optional Slack."""

    # Default threshold values
    DAILY_LOSS_THRESHOLD: float = float(os.getenv("LIVE_DAILY_LOSS_LIMIT_PCT", "5.0")) / 100
    WIN_RATE_MILESTONE: float = 0.60
    SHARPE_MILESTONE: float = 1.5

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []
        self._email_enabled = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
        self._slack_enabled = os.getenv("ALERT_SLACK_ENABLED", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(
        self,
        alert_type: AlertType | str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an alert and dispatch via configured channels."""
        alert_type_str = alert_type.value if isinstance(alert_type, AlertType) else str(alert_type)
        entry: dict[str, Any] = {
            "type": alert_type_str,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._history.append(entry)
        _alerts_logger.info("[%s] %s | data=%s", alert_type_str, message, data)

        if self._email_enabled:
            self._send_email(alert_type_str, message)

        if self._slack_enabled:
            self._send_slack(alert_type_str, message)

        return entry

    def check_daily_loss(self, current_loss_pct: float) -> bool:
        """Trigger DAILY_LOSS alert if loss exceeds threshold. Returns True if alert fired."""
        if current_loss_pct >= self.DAILY_LOSS_THRESHOLD:
            self.send_alert(
                AlertType.DAILY_LOSS,
                f"Daily loss {current_loss_pct:.2%} exceeded threshold {self.DAILY_LOSS_THRESHOLD:.2%}",
                {"current_loss_pct": current_loss_pct, "threshold": self.DAILY_LOSS_THRESHOLD},
            )
            return True
        return False

    def check_performance_milestone(self, metrics: dict[str, float]) -> bool:
        """Trigger PERFORMANCE_MILESTONE alert if metrics exceed thresholds. Returns True if fired."""
        win_rate = metrics.get("win_rate", 0.0)
        sharpe = metrics.get("sharpe_ratio", 0.0)
        if win_rate >= self.WIN_RATE_MILESTONE or sharpe >= self.SHARPE_MILESTONE:
            self.send_alert(
                AlertType.PERFORMANCE_MILESTONE,
                f"Performance milestone reached: win_rate={win_rate:.2%}, sharpe={sharpe:.2f}",
                metrics,
            )
            return True
        return False

    def get_alert_history(self) -> list[dict[str, Any]]:
        """Return all recorded alerts."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send_email(self, alert_type: AlertType | str, message: str) -> None:
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        recipient = os.getenv("ALERT_EMAIL_TO", "")

        if not all([smtp_user, smtp_password, recipient]):
            logger.warning("Email alert skipped: missing SMTP credentials.")
            return

        try:
            msg = MIMEText(message)
            msg["Subject"] = f"[Trading Alert] {alert_type}"
            msg["From"] = smtp_user
            msg["To"] = recipient
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send email alert: %s", exc)

    def _send_slack(self, alert_type: AlertType | str, message: str) -> None:
        webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        if not webhook_url:
            logger.warning("Slack alert skipped: SLACK_WEBHOOK_URL not set.")
            return

        try:
            import json
            import urllib.request

            payload = json.dumps({"text": f"*[{alert_type}]* {message}"}).encode()
            req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send Slack alert: %s", exc)
