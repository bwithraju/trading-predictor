"""Safety validation before issuing a leverage recommendation.

:class:`SafetyChecker` runs a series of checks and returns a list of
:class:`SafetyIssue` objects.  The caller decides whether to proceed
based on the severity of detected issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Severity(str, Enum):
    """Issue severity."""

    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class SafetyIssue:
    """A single safety concern returned by :class:`SafetyChecker`."""

    check: str
    severity: Severity
    message: str
    passed: bool


@dataclass
class SafetyReport:
    """Aggregated safety report."""

    issues: list[SafetyIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only if no CRITICAL issues exist."""
        return not any(i.severity == Severity.CRITICAL for i in self.issues)

    @property
    def max_severity(self) -> Severity:
        """Highest severity level across all issues."""
        for sev in (Severity.CRITICAL, Severity.WARNING, Severity.OK):
            if any(i.severity == sev for i in self.issues):
                return sev
        return Severity.OK

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "max_severity": self.max_severity.value,
            "issues": [
                {
                    "check": i.check,
                    "severity": i.severity.value,
                    "message": i.message,
                    "passed": i.passed,
                }
                for i in self.issues
            ],
        }


class SafetyChecker:
    """Run pre-recommendation safety checks.

    All thresholds have sensible defaults but can be overridden via
    constructor arguments for testing or custom deployments.
    """

    def __init__(
        self,
        min_account_size: float = 10.0,
        max_drawdown_threshold: float = 0.25,
        volatility_spike_threshold: float = 2.0,
        min_data_length: int = 30,
        max_safe_leverage: int = 20,
        liquidation_risk_critical_threshold: float = 40.0,
        liquidation_risk_warning_threshold: float = 20.0,
    ) -> None:
        self.min_account_size = min_account_size
        self.max_drawdown_threshold = max_drawdown_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        self.min_data_length = min_data_length
        self.max_safe_leverage = max_safe_leverage
        self.liquidation_risk_critical_threshold = liquidation_risk_critical_threshold
        self.liquidation_risk_warning_threshold = liquidation_risk_warning_threshold

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_account_size(self, account_size: float) -> SafetyIssue:
        passed = account_size >= self.min_account_size
        return SafetyIssue(
            check="account_size",
            severity=Severity.CRITICAL if not passed else Severity.OK,
            message=(
                f"Account size ${account_size:.2f} is below minimum ${self.min_account_size:.2f}"
                if not passed
                else f"Account size ${account_size:.2f} meets minimum requirement"
            ),
            passed=passed,
        )

    def _check_data_availability(self, data_length: int) -> SafetyIssue:
        passed = data_length >= self.min_data_length
        return SafetyIssue(
            check="data_availability",
            severity=Severity.CRITICAL if not passed else Severity.OK,
            message=(
                f"Insufficient data: {data_length} bars (need {self.min_data_length})"
                if not passed
                else f"Sufficient data available: {data_length} bars"
            ),
            passed=passed,
        )

    def _check_volatility_data(self, volatility_ratio: float) -> SafetyIssue:
        is_nan = np.isnan(volatility_ratio)
        return SafetyIssue(
            check="volatility_data",
            severity=Severity.CRITICAL if is_nan else Severity.OK,
            message=(
                "Volatility data unavailable – cannot compute ratio"
                if is_nan
                else "Volatility data available"
            ),
            passed=not is_nan,
        )

    def _check_volatility_spike(self, volatility_spike: bool) -> SafetyIssue:
        return SafetyIssue(
            check="volatility_spike",
            severity=Severity.WARNING if volatility_spike else Severity.OK,
            message=(
                "Volatility spike detected – consider reducing leverage"
                if volatility_spike
                else "No volatility spike detected"
            ),
            passed=not volatility_spike,
        )

    def _check_drawdown(self, current_drawdown: float) -> SafetyIssue:
        pct = current_drawdown * 100
        exceeded = current_drawdown > self.max_drawdown_threshold
        return SafetyIssue(
            check="drawdown",
            severity=Severity.WARNING if exceeded else Severity.OK,
            message=(
                f"Current drawdown {pct:.1f}% exceeds threshold "
                f"{self.max_drawdown_threshold * 100:.0f}% – consider reducing exposure"
                if exceeded
                else f"Current drawdown {pct:.1f}% within acceptable range"
            ),
            passed=not exceeded,
        )

    def _check_leverage_cap(self, requested_leverage: int) -> SafetyIssue:
        exceeded = requested_leverage > self.max_safe_leverage
        return SafetyIssue(
            check="leverage_cap",
            severity=Severity.CRITICAL if exceeded else Severity.OK,
            message=(
                f"Requested leverage {requested_leverage}× exceeds hard cap {self.max_safe_leverage}×"
                if exceeded
                else f"Leverage {requested_leverage}× within hard cap"
            ),
            passed=not exceeded,
        )

    def _check_liquidation_risk(self, liquidation_risk_pct: float) -> SafetyIssue:
        if liquidation_risk_pct >= self.liquidation_risk_critical_threshold:
            severity = Severity.CRITICAL
            passed = False
            message = f"Liquidation risk {liquidation_risk_pct:.1f}% is CRITICAL"
        elif liquidation_risk_pct >= self.liquidation_risk_warning_threshold:
            severity = Severity.WARNING
            passed = False
            message = f"Liquidation risk {liquidation_risk_pct:.1f}% is elevated"
        else:
            severity = Severity.OK
            passed = True
            message = f"Liquidation risk {liquidation_risk_pct:.1f}% is acceptable"

        return SafetyIssue(
            check="liquidation_risk",
            severity=severity,
            message=message,
            passed=passed,
        )

    # ------------------------------------------------------------------
    # Composite check runner
    # ------------------------------------------------------------------

    def run_checks(
        self,
        account_size: float,
        data_length: int,
        volatility_ratio: float,
        volatility_spike: bool,
        current_drawdown: float,
        requested_leverage: int,
        liquidation_risk_pct: float,
    ) -> SafetyReport:
        """Run all safety checks and return an aggregated :class:`SafetyReport`.

        Args:
            account_size: Account equity in USD.
            data_length: Number of OHLCV bars available.
            volatility_ratio: Current / historical volatility ratio.
            volatility_spike: Whether a spike has been detected.
            current_drawdown: Current drawdown as a fraction (0–1).
            requested_leverage: Leverage value to validate.
            liquidation_risk_pct: Estimated liquidation risk (0–100).

        Returns:
            :class:`SafetyReport` with all issues listed.
        """
        issues = [
            self._check_account_size(account_size),
            self._check_data_availability(data_length),
            self._check_volatility_data(volatility_ratio),
            self._check_volatility_spike(volatility_spike),
            self._check_drawdown(current_drawdown),
            self._check_leverage_cap(requested_leverage),
            self._check_liquidation_risk(liquidation_risk_pct),
        ]
        return SafetyReport(issues=issues)
