"""Cost tracking for LLM API usage.

Tracks token usage and calculates costs for different LLM providers.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Model pricing per 1K tokens (input/output in USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1-preview": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},

    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},

    # Google
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},

    # Groq (free tier, estimate for paid)
    "llama-3.1-70b": {"input": 0.0, "output": 0.0},
    "llama-3.1-8b": {"input": 0.0, "output": 0.0},
    "mixtral-8x7b": {"input": 0.0, "output": 0.0},

    # Default for unknown models
    "default": {"input": 0.001, "output": 0.002},
}


@dataclass
class UsageRecord:
    """A single usage record"""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    session_id: str | None = None
    tool_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageRecord":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class SessionCostSummary:
    """Cost summary for a session"""
    session_id: str
    model: str
    start_time: str
    end_time: str | None = None
    total_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    records: list[UsageRecord] = field(default_factory=list)

    def add_usage(self, record: UsageRecord) -> None:
        """Add a usage record"""
        self.records.append(record)
        self.total_requests += 1
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.total_tokens += record.total_tokens
        self.total_cost += record.total_cost
        self.end_time = record.timestamp


class CostTracker:
    """Tracks LLM usage costs across sessions"""

    USAGE_LOG_FILE = Path.home() / ".sepilot" / "usage_log.jsonl"
    MONTHLY_STATS_FILE = Path.home() / ".sepilot" / "monthly_stats.json"

    def __init__(self):
        """Initialize cost tracker"""
        self._current_session: SessionCostSummary | None = None
        self._load_monthly_stats()

    def _load_monthly_stats(self) -> None:
        """Load monthly statistics"""
        self._monthly_stats: dict[str, dict[str, Any]] = {}

        if self.MONTHLY_STATS_FILE.exists():
            try:
                with open(self.MONTHLY_STATS_FILE, encoding="utf-8") as f:
                    self._monthly_stats = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load monthly stats: {e}")

    def _save_monthly_stats(self) -> None:
        """Save monthly statistics"""
        try:
            self.MONTHLY_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.MONTHLY_STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._monthly_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save monthly stats: {e}")

    def start_session(self, session_id: str, model: str) -> None:
        """Start tracking a new session

        Args:
            session_id: Session identifier
            model: Model name
        """
        self._current_session = SessionCostSummary(
            session_id=session_id,
            model=model,
            start_time=datetime.now().isoformat(),
        )
        logger.debug(f"Started cost tracking for session: {session_id}")

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tool_name: str | None = None,
    ) -> UsageRecord:
        """Record token usage

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tool_name: Optional tool name

        Returns:
            UsageRecord with cost information
        """
        # Calculate costs
        pricing = self._get_pricing(model)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            session_id=self._current_session.session_id if self._current_session else None,
            tool_name=tool_name,
        )

        # Add to current session
        if self._current_session:
            self._current_session.add_usage(record)

        # Log to file
        self._log_usage(record)

        # Update monthly stats
        self._update_monthly_stats(record)

        return record

    def _get_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model

        Args:
            model: Model name

        Returns:
            Pricing dictionary with input/output rates
        """
        model_lower = model.lower()

        # Exact match
        if model_lower in MODEL_PRICING:
            return MODEL_PRICING[model_lower]

        # Partial match
        for key in MODEL_PRICING:
            if key in model_lower or model_lower in key:
                return MODEL_PRICING[key]

        return MODEL_PRICING["default"]

    def _log_usage(self, record: UsageRecord) -> None:
        """Log usage to file

        Args:
            record: Usage record
        """
        try:
            self.USAGE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.USAGE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log usage: {e}")

    def _update_monthly_stats(self, record: UsageRecord) -> None:
        """Update monthly statistics

        Args:
            record: Usage record
        """
        month_key = datetime.now().strftime("%Y-%m")

        if month_key not in self._monthly_stats:
            self._monthly_stats[month_key] = {
                "total_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_model": {},
            }

        stats = self._monthly_stats[month_key]
        stats["total_requests"] += 1
        stats["input_tokens"] += record.input_tokens
        stats["output_tokens"] += record.output_tokens
        stats["total_tokens"] += record.total_tokens
        stats["total_cost"] += record.total_cost

        # Track by model
        if record.model not in stats["by_model"]:
            stats["by_model"][record.model] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
            }
        stats["by_model"][record.model]["requests"] += 1
        stats["by_model"][record.model]["tokens"] += record.total_tokens
        stats["by_model"][record.model]["cost"] += record.total_cost

        self._save_monthly_stats()

    def get_session_summary(self) -> SessionCostSummary | None:
        """Get current session cost summary

        Returns:
            Session cost summary or None
        """
        return self._current_session

    def get_monthly_summary(self, month: str | None = None) -> dict[str, Any]:
        """Get monthly cost summary

        Args:
            month: Month in YYYY-MM format (defaults to current month)

        Returns:
            Monthly statistics dictionary
        """
        if month is None:
            month = datetime.now().strftime("%Y-%m")

        return self._monthly_stats.get(month, {
            "total_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "by_model": {},
        })

    def get_all_time_summary(self) -> dict[str, Any]:
        """Get all-time cost summary

        Returns:
            All-time statistics dictionary
        """
        summary = {
            "total_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "by_model": {},
            "by_month": {},
        }

        for month, stats in self._monthly_stats.items():
            summary["total_requests"] += stats["total_requests"]
            summary["input_tokens"] += stats["input_tokens"]
            summary["output_tokens"] += stats["output_tokens"]
            summary["total_tokens"] += stats["total_tokens"]
            summary["total_cost"] += stats["total_cost"]

            summary["by_month"][month] = {
                "requests": stats["total_requests"],
                "tokens": stats["total_tokens"],
                "cost": stats["total_cost"],
            }

            for model, model_stats in stats.get("by_model", {}).items():
                if model not in summary["by_model"]:
                    summary["by_model"][model] = {
                        "requests": 0,
                        "tokens": 0,
                        "cost": 0.0,
                    }
                summary["by_model"][model]["requests"] += model_stats["requests"]
                summary["by_model"][model]["tokens"] += model_stats["tokens"]
                summary["by_model"][model]["cost"] += model_stats["cost"]

        return summary

    def format_summary(self, summary: dict[str, Any]) -> str:
        """Format summary for display

        Args:
            summary: Statistics dictionary

        Returns:
            Formatted string
        """
        lines = [
            f"Total Requests: {summary['total_requests']:,}",
            f"Total Tokens: {summary['total_tokens']:,}",
            f"  - Input: {summary['input_tokens']:,}",
            f"  - Output: {summary['output_tokens']:,}",
            f"Total Cost: ${summary['total_cost']:.4f}",
        ]

        if summary.get("by_model"):
            lines.append("\nBy Model:")
            for model, stats in sorted(
                summary["by_model"].items(),
                key=lambda x: x[1]["cost"],
                reverse=True
            ):
                lines.append(
                    f"  {model}: {stats['requests']:,} requests, "
                    f"{stats['tokens']:,} tokens, ${stats['cost']:.4f}"
                )

        return "\n".join(lines)

    def end_session(self) -> SessionCostSummary | None:
        """End the current session and return summary

        Returns:
            Session cost summary
        """
        summary = self._current_session
        self._current_session = None
        return summary


# Singleton instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker

    Returns:
        CostTracker instance
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def record_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    tool_name: str | None = None,
) -> UsageRecord:
    """Convenience function to record usage

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        tool_name: Optional tool name

    Returns:
        Usage record
    """
    return get_cost_tracker().record_usage(model, input_tokens, output_tokens, tool_name)
