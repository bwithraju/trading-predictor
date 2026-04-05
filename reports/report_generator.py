"""Comprehensive report generator for backtest, paper trading, and risk analysis."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any


def _html_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html><head>"
        f"<meta charset='utf-8'><title>{title}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
        "th{background:#f4f4f4}</style>"
        f"</head><body><h1>{title}</h1>"
        f"<p>Generated: {datetime.utcnow().isoformat()} UTC</p>"
        f"{body}</body></html>"
    )


def _metrics_table(data: dict[str, Any]) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items())
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


class ComprehensiveReportGenerator:
    """Generates HTML and Markdown reports for trading analysis."""

    def generate_backtest_html(
        self,
        backtest_results: dict[str, Any],
        output_path: str = "reports/backtest_results.html",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        body = _metrics_table(backtest_results)
        html = _html_page("Backtest Results", body)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return output_path

    def generate_validation_md(
        self,
        validation_results: dict[str, Any],
        output_path: str = "reports/validation_report.md",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lines = [
            "# Validation Report",
            f"\n_Generated: {datetime.utcnow().isoformat()} UTC_\n",
            "| Check | Result |",
            "|-------|--------|",
        ]
        for check, result in validation_results.items():
            status = "✅ Pass" if result else "❌ Fail"
            lines.append(f"| {check} | {status} |")
        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return output_path

    def generate_paper_trading_html(
        self,
        paper_results: dict[str, Any],
        output_path: str = "reports/paper_trading_summary.html",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        body = _metrics_table(paper_results)
        html = _html_page("Paper Trading Summary", body)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return output_path

    def generate_risk_analysis_html(
        self,
        risk_data: dict[str, Any],
        output_path: str = "reports/risk_analysis.html",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        body = _metrics_table(risk_data)
        html = _html_page("Risk Analysis", body)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return output_path

    def generate_recommendations_md(
        self,
        analysis: dict[str, Any],
        output_path: str = "reports/recommendations.md",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lines = [
            "# Trading Recommendations",
            f"\n_Generated: {datetime.utcnow().isoformat()} UTC_\n",
        ]
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            for rec in recommendations:
                lines.append(f"- {rec}")
        else:
            lines.append("_No recommendations at this time._")
        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return output_path

    def generate_all_reports(
        self,
        backtest_results: dict[str, Any],
        paper_results: dict[str, Any],
        validation_results: dict[str, Any],
    ) -> dict[str, str]:
        """Generate all reports and return a dict of report_name -> file_path."""
        return {
            "backtest": self.generate_backtest_html(backtest_results),
            "paper_trading": self.generate_paper_trading_html(paper_results),
            "validation": self.generate_validation_md(validation_results),
            "risk_analysis": self.generate_risk_analysis_html(backtest_results),
            "recommendations": self.generate_recommendations_md({}),
        }
