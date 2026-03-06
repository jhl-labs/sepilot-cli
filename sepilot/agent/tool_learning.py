"""Tool Learning - Tool Usage Optimization System

This module implements tool learning capabilities:
- Tracks tool usage patterns and success rates
- Learns optimal tool sequences for task types
- Recommends tools based on context
- Adapts tool selection over time

Inspired by:
- Toolformer: Language Models Can Teach Themselves to Use Tools
- ART: Automatic multi-step Reasoning and Tool-use
- Tool Learning with Foundation Models
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sepilot.agent.enhanced_state import AgentStrategy, EnhancedAgentState


@dataclass
class ToolUsageRecord:
    """Record of a single tool usage."""
    tool_name: str
    timestamp: datetime
    task_type: str  # e.g., "bug_fix", "feature", "refactor"
    context_keywords: list[str]
    success: bool
    duration: float  # seconds
    error_message: str | None = None
    preceded_by: str | None = None  # Previous tool in sequence
    followed_by: str | None = None  # Next tool in sequence
    file_types: list[str] = field(default_factory=list)  # e.g., [".py", ".ts"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat(),
            "task_type": self.task_type,
            "context_keywords": self.context_keywords,
            "success": self.success,
            "duration": self.duration,
            "error_message": self.error_message,
            "preceded_by": self.preceded_by,
            "followed_by": self.followed_by,
            "file_types": self.file_types
        }


@dataclass
class ToolStatistics:
    """Aggregated statistics for a tool."""
    tool_name: str
    total_uses: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration: float = 0.0
    by_task_type: dict[str, dict[str, int]] = field(default_factory=dict)
    common_predecessors: dict[str, int] = field(default_factory=dict)
    common_successors: dict[str, int] = field(default_factory=dict)
    common_errors: dict[str, int] = field(default_factory=dict)
    file_type_affinity: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.success_count / self.total_uses if self.total_uses > 0 else 0.0

    @property
    def avg_duration(self) -> float:
        """Calculate average duration."""
        return self.total_duration / self.total_uses if self.total_uses > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "total_uses": self.total_uses,
            "success_rate": self.success_rate,
            "avg_duration": self.avg_duration,
            "by_task_type": self.by_task_type,
            "top_predecessors": dict(sorted(
                self.common_predecessors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            "top_successors": dict(sorted(
                self.common_successors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }


@dataclass
class ToolSequence:
    """A learned tool sequence pattern."""
    sequence: list[str]  # Ordered list of tools
    task_type: str
    occurrence_count: int = 1
    success_count: int = 0
    avg_task_completion_rate: float = 0.0

    @property
    def effectiveness(self) -> float:
        """Calculate sequence effectiveness."""
        if self.occurrence_count == 0:
            return 0.0
        return (self.success_count / self.occurrence_count) * self.avg_task_completion_rate


@dataclass
class ToolRecommendation:
    """A tool recommendation."""
    tool_name: str
    confidence: float  # 0.0 to 1.0
    reason: str
    expected_success_rate: float
    suggested_sequence: list[str] | None = None


class ToolLearningSystem:
    """Learns and optimizes tool usage patterns.

    Features:
    - Tracks tool usage history
    - Identifies effective tool sequences
    - Recommends tools based on context
    - Adapts to project-specific patterns
    """

    MAX_HISTORY = 500  # Maximum usage records to keep

    def __init__(
        self,
        storage_path: str | Path | None = None,
        project_id: str | None = None
    ):
        """Initialize tool learning system.

        Args:
            storage_path: Path for persistence
            project_id: Project identifier
        """
        if storage_path is None:
            storage_path = Path.home() / ".sepilot" / "tool_learning"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.project_id = project_id or "default"
        self.usage_history: list[ToolUsageRecord] = []
        self.tool_stats: dict[str, ToolStatistics] = {}
        self.learned_sequences: dict[str, list[ToolSequence]] = defaultdict(list)

        self._load_data()

    def _get_data_file(self) -> Path:
        """Get path to data file."""
        return self.storage_path / f"{self.project_id}_tool_data.json"

    def _load_data(self) -> None:
        """Load persisted data."""
        data_file = self._get_data_file()
        if data_file.exists():
            try:
                with open(data_file, encoding="utf-8") as f:
                    data = json.load(f)
                    # Load stats
                    for tool_name, stats_data in data.get("tool_stats", {}).items():
                        self.tool_stats[tool_name] = ToolStatistics(
                            tool_name=tool_name,
                            total_uses=stats_data.get("total_uses", 0),
                            success_count=stats_data.get("success_count", 0),
                            failure_count=stats_data.get("failure_count", 0),
                            total_duration=stats_data.get("total_duration", 0.0),
                            by_task_type=stats_data.get("by_task_type", {}),
                            common_predecessors=stats_data.get("common_predecessors", {}),
                            common_successors=stats_data.get("common_successors", {}),
                            common_errors=stats_data.get("common_errors", {}),
                            file_type_affinity=stats_data.get("file_type_affinity", {})
                        )
                    # Load sequences
                    for task_type, sequences in data.get("learned_sequences", {}).items():
                        for seq_data in sequences:
                            self.learned_sequences[task_type].append(ToolSequence(
                                sequence=seq_data["sequence"],
                                task_type=task_type,
                                occurrence_count=seq_data.get("occurrence_count", 1),
                                success_count=seq_data.get("success_count", 0),
                                avg_task_completion_rate=seq_data.get("avg_task_completion_rate", 0.0)
                            ))
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_data(self) -> None:
        """Persist data to disk."""
        data = {
            "project_id": self.project_id,
            "updated_at": datetime.now().isoformat(),
            "tool_stats": {
                name: {
                    "total_uses": stats.total_uses,
                    "success_count": stats.success_count,
                    "failure_count": stats.failure_count,
                    "total_duration": stats.total_duration,
                    "by_task_type": stats.by_task_type,
                    "common_predecessors": stats.common_predecessors,
                    "common_successors": stats.common_successors,
                    "common_errors": stats.common_errors,
                    "file_type_affinity": stats.file_type_affinity
                }
                for name, stats in self.tool_stats.items()
            },
            "learned_sequences": {
                task_type: [
                    {
                        "sequence": seq.sequence,
                        "occurrence_count": seq.occurrence_count,
                        "success_count": seq.success_count,
                        "avg_task_completion_rate": seq.avg_task_completion_rate
                    }
                    for seq in sequences
                ]
                for task_type, sequences in self.learned_sequences.items()
            }
        }
        with open(self._get_data_file(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_tool_usage(
        self,
        tool_name: str,
        success: bool,
        duration: float,
        task_type: str = "general",
        context_keywords: list[str] | None = None,
        error_message: str | None = None,
        file_types: list[str] | None = None
    ) -> None:
        """Record a tool usage event.

        Args:
            tool_name: Name of the tool used
            success: Whether the tool call succeeded
            duration: Duration in seconds
            task_type: Type of task being performed
            context_keywords: Keywords from task context
            error_message: Error message if failed
            file_types: File types involved
        """
        # Get previous tool from history
        preceded_by = None
        if self.usage_history:
            preceded_by = self.usage_history[-1].tool_name

        record = ToolUsageRecord(
            tool_name=tool_name,
            timestamp=datetime.now(),
            task_type=task_type,
            context_keywords=context_keywords or [],
            success=success,
            duration=duration,
            error_message=error_message,
            preceded_by=preceded_by,
            file_types=file_types or []
        )

        # Update previous record's followed_by
        if self.usage_history:
            self.usage_history[-1].followed_by = tool_name

        self.usage_history.append(record)

        # Update statistics
        self._update_statistics(record)

        # Prune history if needed
        if len(self.usage_history) > self.MAX_HISTORY:
            self.usage_history = self.usage_history[-self.MAX_HISTORY:]

        self._save_data()

    def _update_statistics(self, record: ToolUsageRecord) -> None:
        """Update tool statistics with new record."""
        tool_name = record.tool_name

        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolStatistics(tool_name=tool_name)

        stats = self.tool_stats[tool_name]
        stats.total_uses += 1
        stats.total_duration += record.duration

        if record.success:
            stats.success_count += 1
        else:
            stats.failure_count += 1
            if record.error_message:
                error_key = record.error_message[:50]
                stats.common_errors[error_key] = stats.common_errors.get(error_key, 0) + 1

        # Task type breakdown
        if record.task_type not in stats.by_task_type:
            stats.by_task_type[record.task_type] = {"success": 0, "failure": 0}
        if record.success:
            stats.by_task_type[record.task_type]["success"] += 1
        else:
            stats.by_task_type[record.task_type]["failure"] += 1

        # Predecessor/successor tracking
        if record.preceded_by:
            stats.common_predecessors[record.preceded_by] = \
                stats.common_predecessors.get(record.preceded_by, 0) + 1

        # File type affinity
        for ft in record.file_types:
            stats.file_type_affinity[ft] = stats.file_type_affinity.get(ft, 0) + 1

    def record_from_state(
        self,
        state: EnhancedAgentState,
        task_completed: bool = False
    ) -> None:
        """Record tool usage from agent state.

        Args:
            state: Current agent state
            task_completed: Whether the overall task completed
        """
        tool_history = state.get("tool_call_history", [])
        if not tool_history:
            return

        # Determine task type from strategy
        strategy = state.get("current_strategy", AgentStrategy.IMPLEMENT)
        task_type = strategy.value if strategy else "general"

        # Get file types from changes
        file_changes = state.get("file_changes", [])
        file_types = list({
            Path(fc.file_path).suffix
            for fc in file_changes
            if fc.file_path
        })

        # Record each tool usage
        for tc in tool_history:
            self.record_tool_usage(
                tool_name=tc.tool_name,
                success=tc.success,
                duration=tc.duration,
                task_type=task_type,
                error_message=tc.error if hasattr(tc, 'error') else None,
                file_types=file_types
            )

        # Learn sequence if task completed
        if task_completed and len(tool_history) >= 2:
            sequence = [tc.tool_name for tc in tool_history]
            self._learn_sequence(sequence, task_type, success=True)

    def _learn_sequence(
        self,
        sequence: list[str],
        task_type: str,
        success: bool
    ) -> None:
        """Learn a tool sequence pattern."""
        # Normalize sequence (remove consecutive duplicates)
        normalized = []
        for tool in sequence:
            if not normalized or normalized[-1] != tool:
                normalized.append(tool)

        if len(normalized) < 2:
            return

        # Check if sequence already exists
        sequence_key = "->".join(normalized)
        for existing in self.learned_sequences[task_type]:
            if "->".join(existing.sequence) == sequence_key:
                existing.occurrence_count += 1
                if success:
                    existing.success_count += 1
                return

        # Add new sequence
        self.learned_sequences[task_type].append(ToolSequence(
            sequence=normalized,
            task_type=task_type,
            occurrence_count=1,
            success_count=1 if success else 0
        ))

    def recommend_tools(
        self,
        task_description: str,
        current_tool: str | None = None,
        strategy: AgentStrategy | None = None,
        file_types: list[str] | None = None,
        limit: int = 3
    ) -> list[ToolRecommendation]:
        """Get tool recommendations for current context.

        Args:
            task_description: Description of the task
            current_tool: Currently executing tool (for sequence prediction)
            strategy: Current agent strategy
            file_types: File types being worked on
            limit: Maximum recommendations

        Returns:
            List of ToolRecommendation objects
        """
        recommendations = []
        task_type = strategy.value if strategy else "general"

        # Score each tool
        tool_scores: dict[str, float] = {}

        for tool_name, stats in self.tool_stats.items():
            score = 0.0

            # Base score from success rate
            score += stats.success_rate * 30

            # Task type affinity
            if task_type in stats.by_task_type:
                type_stats = stats.by_task_type[task_type]
                type_success_rate = type_stats["success"] / (type_stats["success"] + type_stats["failure"]) \
                    if (type_stats["success"] + type_stats["failure"]) > 0 else 0
                score += type_success_rate * 20

            # Sequence prediction (if current tool known)
            if current_tool and current_tool in self.tool_stats:
                current_stats = self.tool_stats[current_tool]
                if tool_name in current_stats.common_successors:
                    successor_freq = current_stats.common_successors[tool_name]
                    total_successors = sum(current_stats.common_successors.values())
                    score += (successor_freq / total_successors) * 25 if total_successors > 0 else 0

            # File type affinity
            if file_types:
                for ft in file_types:
                    if ft in stats.file_type_affinity:
                        score += 5

            # Usage frequency (slight preference for commonly used tools)
            score += min(stats.total_uses / 10, 10)

            tool_scores[tool_name] = score

        # Sort and create recommendations
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)

        for tool_name, score in sorted_tools[:limit]:
            stats = self.tool_stats.get(tool_name)
            if stats:
                # Find relevant sequence
                suggested_sequence = None
                if task_type in self.learned_sequences:
                    for seq in sorted(
                        self.learned_sequences[task_type],
                        key=lambda s: s.effectiveness,
                        reverse=True
                    ):
                        if tool_name in seq.sequence:
                            suggested_sequence = seq.sequence
                            break

                recommendations.append(ToolRecommendation(
                    tool_name=tool_name,
                    confidence=min(score / 100, 1.0),
                    reason=self._generate_recommendation_reason(tool_name, stats, task_type),
                    expected_success_rate=stats.success_rate,
                    suggested_sequence=suggested_sequence
                ))

        return recommendations

    def _generate_recommendation_reason(
        self,
        tool_name: str,
        stats: ToolStatistics,
        task_type: str
    ) -> str:
        """Generate human-readable reason for recommendation."""
        reasons = []

        if stats.success_rate >= 0.8:
            reasons.append(f"high success rate ({stats.success_rate:.0%})")

        if task_type in stats.by_task_type:
            type_stats = stats.by_task_type[task_type]
            if type_stats["success"] > type_stats["failure"]:
                reasons.append(f"effective for {task_type} tasks")

        if stats.total_uses >= 10:
            reasons.append(f"frequently used ({stats.total_uses} times)")

        return ", ".join(reasons) if reasons else "general recommendation"

    def get_optimal_sequence(
        self,
        task_type: str,
        min_occurrences: int = 2
    ) -> list[str] | None:
        """Get the most effective tool sequence for a task type.

        Args:
            task_type: Type of task
            min_occurrences: Minimum times sequence must have occurred

        Returns:
            Optimal tool sequence or None
        """
        if task_type not in self.learned_sequences:
            return None

        valid_sequences = [
            seq for seq in self.learned_sequences[task_type]
            if seq.occurrence_count >= min_occurrences
        ]

        if not valid_sequences:
            return None

        best = max(valid_sequences, key=lambda s: s.effectiveness)
        return best.sequence if best.effectiveness > 0.5 else None

    def format_recommendations_for_prompt(
        self,
        recommendations: list[ToolRecommendation]
    ) -> str:
        """Format recommendations for LLM prompt.

        Args:
            recommendations: List of recommendations

        Returns:
            Formatted string
        """
        if not recommendations:
            return ""

        lines = ["## Tool Recommendations (based on learned patterns):\n"]

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. **{rec.tool_name}** (confidence: {rec.confidence:.0%})")
            lines.append(f"   - Reason: {rec.reason}")
            lines.append(f"   - Expected success: {rec.expected_success_rate:.0%}")
            if rec.suggested_sequence:
                lines.append(f"   - Suggested sequence: {' → '.join(rec.suggested_sequence)}")
            lines.append("")

        return "\n".join(lines)

    def get_statistics_summary(self) -> dict[str, Any]:
        """Get summary of tool learning statistics."""
        return {
            "total_tools_tracked": len(self.tool_stats),
            "total_usages_recorded": sum(s.total_uses for s in self.tool_stats.values()),
            "learned_sequences": sum(len(seqs) for seqs in self.learned_sequences.values()),
            "top_tools": sorted(
                [(name, stats.success_rate, stats.total_uses)
                 for name, stats in self.tool_stats.items()],
                key=lambda x: x[1] * x[2],
                reverse=True
            )[:5],
            "task_types_covered": list(self.learned_sequences.keys())
        }


class ToolLearningNode:
    """LangGraph node for tool learning integration."""

    def __init__(
        self,
        tool_learning: ToolLearningSystem,
        console: Any | None = None,
        verbose: bool = False
    ):
        """Initialize tool learning node.

        Args:
            tool_learning: ToolLearningSystem instance
            console: Rich console
            verbose: Verbose output flag
        """
        self.tool_learning = tool_learning
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Apply tool learning to current state.

        Args:
            state: Current agent state

        Returns:
            State updates with tool recommendations
        """
        # Get task context
        messages = state.get("messages", [])
        task_description = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task_description = getattr(msg, "content", "")
                break

        # Get current context
        strategy = state.get("current_strategy")
        file_changes = state.get("file_changes", [])
        file_types = list({
            Path(fc.file_path).suffix
            for fc in file_changes
            if fc.file_path
        })

        # Get last tool used
        tool_history = state.get("tool_call_history", [])
        current_tool = tool_history[-1].tool_name if tool_history else None

        # Get recommendations
        recommendations = self.tool_learning.recommend_tools(
            task_description=task_description,
            current_tool=current_tool,
            strategy=strategy,
            file_types=file_types
        )

        # Get optimal sequence
        task_type = strategy.value if strategy else "general"
        optimal_sequence = self.tool_learning.get_optimal_sequence(task_type)

        updates: dict[str, Any] = {}

        if recommendations:
            updates["tool_recommendations"] = [
                {
                    "tool": rec.tool_name,
                    "confidence": rec.confidence,
                    "reason": rec.reason
                }
                for rec in recommendations
            ]

            if self.console and self.verbose:
                self.console.print(
                    f"[dim cyan]🔧 Tool recommendations: "
                    f"{', '.join(r.tool_name for r in recommendations[:3])}[/dim cyan]"
                )

        if optimal_sequence:
            updates["suggested_tool_sequence"] = optimal_sequence

        return updates


def create_tool_learning_system(
    storage_path: str | Path | None = None,
    project_id: str | None = None
) -> ToolLearningSystem:
    """Factory function to create ToolLearningSystem.

    Args:
        storage_path: Path for storage
        project_id: Project identifier

    Returns:
        Configured ToolLearningSystem
    """
    return ToolLearningSystem(storage_path=storage_path, project_id=project_id)
