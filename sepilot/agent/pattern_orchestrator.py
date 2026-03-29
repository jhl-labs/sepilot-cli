"""Adaptive Pattern Orchestrator - Smart Pattern Selection

This module coordinates all agent patterns and decides which to activate
based on task type, complexity, and context.

Patterns managed:
- Reflection: Self-critique and improvement
- Memory Bank: Experience learning
- Backtracking: State rollback
- Hierarchical Planning: Multi-level task decomposition
- Tool Learning: Tool usage optimization
- Debate: Multi-perspective analysis

The orchestrator analyzes the task and automatically enables
the most appropriate patterns for optimal performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseChatModel

from sepilot.agent.enhanced_state import EnhancedAgentState
from sepilot.agent.execution_context import get_current_user_query


class TaskType(str, Enum):
    """Types of tasks the agent handles."""
    BUG_FIX = "bug_fix"
    NEW_FEATURE = "new_feature"
    REFACTOR = "refactor"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    EXPLORATION = "exploration"
    GENERAL = "general"


class PatternType(str, Enum):
    """Available agent patterns."""
    REFLECTION = "reflection"
    MEMORY_BANK = "memory_bank"
    BACKTRACKING = "backtracking"
    HIERARCHICAL_PLANNING = "hierarchical_planning"
    TOOL_LEARNING = "tool_learning"
    DEBATE = "debate"


@dataclass
class PatternConfig:
    """Configuration for a single pattern."""
    pattern_type: PatternType
    enabled: bool = True
    priority: int = 5  # 1-10, higher = more important
    trigger_conditions: list[str] = field(default_factory=list)
    task_type_affinity: dict[TaskType, float] = field(default_factory=dict)

    def should_activate(self, task_type: TaskType, complexity: int) -> bool:
        """Determine if pattern should activate."""
        if not self.enabled:
            return False

        # Check task type affinity
        affinity = self.task_type_affinity.get(task_type, 0.5)
        if affinity < 0.3:
            return False

        # Higher complexity favors more patterns
        if complexity >= 3 and affinity >= 0.5:
            return True

        return affinity >= 0.7


@dataclass
class OrchestrationPlan:
    """Plan for which patterns to use."""
    task_type: TaskType
    complexity: int
    active_patterns: list[PatternType]
    pattern_configs: dict[PatternType, dict[str, Any]]
    reasoning: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type.value,
            "complexity": self.complexity,
            "active_patterns": [p.value for p in self.active_patterns],
            "reasoning": self.reasoning
        }


class TaskAnalyzer:
    """Analyzes tasks to determine type and complexity."""

    # Keywords for task type detection
    TASK_KEYWORDS = {
        TaskType.BUG_FIX: [
            "bug", "fix", "error", "issue", "broken", "crash",
            "not working", "fails", "exception", "버그", "수정", "오류"
        ],
        TaskType.NEW_FEATURE: [
            "add", "new", "create", "implement", "build", "feature",
            "추가", "새로운", "구현", "기능"
        ],
        TaskType.REFACTOR: [
            "refactor", "clean", "reorganize", "improve", "optimize",
            "restructure", "리팩토링", "개선", "최적화"
        ],
        TaskType.CODE_REVIEW: [
            "review", "check", "verify", "analyze", "audit",
            "security", "리뷰", "검토", "분석"
        ],
        TaskType.DOCUMENTATION: [
            "document", "docs", "readme", "comment", "explain",
            "문서", "설명"
        ],
        TaskType.TESTING: [
            "test", "spec", "coverage", "unit test", "integration",
            "테스트"
        ],
        TaskType.EXPLORATION: [
            "find", "search", "where", "what", "how does",
            "understand", "explore", "찾기", "검색", "어디"
        ]
    }

    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        "high": [
            "multiple files", "across", "entire", "all", "complex",
            "architecture", "system", "refactor", "여러", "전체"
        ],
        "medium": [
            "some", "few", "update", "modify", "change",
            "몇 개", "변경"
        ],
        "low": [
            "simple", "small", "one", "single", "quick",
            "간단", "작은", "하나"
        ]
    }

    def analyze(self, task_description: str) -> tuple[TaskType, int]:
        """Analyze task to determine type and complexity.

        Args:
            task_description: The task description

        Returns:
            Tuple of (TaskType, complexity 1-5)
        """
        task_lower = task_description.lower()

        # Determine task type
        task_type = TaskType.GENERAL
        max_score = 0

        for ttype, keywords in self.TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > max_score:
                max_score = score
                task_type = ttype

        # Determine complexity
        complexity = 3  # Default medium

        high_count = sum(1 for ind in self.COMPLEXITY_INDICATORS["high"] if ind in task_lower)
        low_count = sum(1 for ind in self.COMPLEXITY_INDICATORS["low"] if ind in task_lower)

        if high_count > low_count:
            complexity = 4 if high_count >= 2 else 3
        elif low_count > high_count:
            complexity = 2 if low_count >= 2 else 2

        # Adjust based on length
        word_count = len(task_description.split())
        if word_count > 50:
            complexity = min(5, complexity + 1)
        elif word_count < 10:
            complexity = max(1, complexity - 1)

        return task_type, complexity


class PatternSelector:
    """Selects appropriate patterns based on task analysis."""

    # Default pattern configurations
    DEFAULT_CONFIGS: dict[PatternType, PatternConfig] = {
        PatternType.REFLECTION: PatternConfig(
            pattern_type=PatternType.REFLECTION,
            enabled=True,
            priority=8,
            task_type_affinity={
                TaskType.BUG_FIX: 0.9,
                TaskType.NEW_FEATURE: 0.7,
                TaskType.REFACTOR: 0.8,
                TaskType.CODE_REVIEW: 0.6,
                TaskType.DOCUMENTATION: 0.4,
                TaskType.TESTING: 0.7,
                TaskType.EXPLORATION: 0.5,
                TaskType.GENERAL: 0.6
            }
        ),
        PatternType.MEMORY_BANK: PatternConfig(
            pattern_type=PatternType.MEMORY_BANK,
            enabled=True,
            priority=9,
            task_type_affinity={
                TaskType.BUG_FIX: 0.9,
                TaskType.NEW_FEATURE: 0.8,
                TaskType.REFACTOR: 0.7,
                TaskType.CODE_REVIEW: 0.8,
                TaskType.DOCUMENTATION: 0.5,
                TaskType.TESTING: 0.6,
                TaskType.EXPLORATION: 0.6,
                TaskType.GENERAL: 0.7
            }
        ),
        PatternType.BACKTRACKING: PatternConfig(
            pattern_type=PatternType.BACKTRACKING,
            enabled=True,
            priority=7,
            task_type_affinity={
                TaskType.BUG_FIX: 0.9,
                TaskType.NEW_FEATURE: 0.8,
                TaskType.REFACTOR: 0.9,
                TaskType.CODE_REVIEW: 0.3,
                TaskType.DOCUMENTATION: 0.2,
                TaskType.TESTING: 0.5,
                TaskType.EXPLORATION: 0.2,
                TaskType.GENERAL: 0.5
            }
        ),
        PatternType.HIERARCHICAL_PLANNING: PatternConfig(
            pattern_type=PatternType.HIERARCHICAL_PLANNING,
            enabled=True,
            priority=6,
            task_type_affinity={
                TaskType.BUG_FIX: 0.5,
                TaskType.NEW_FEATURE: 0.9,
                TaskType.REFACTOR: 0.8,
                TaskType.CODE_REVIEW: 0.4,
                TaskType.DOCUMENTATION: 0.6,
                TaskType.TESTING: 0.7,
                TaskType.EXPLORATION: 0.3,
                TaskType.GENERAL: 0.5
            }
        ),
        PatternType.TOOL_LEARNING: PatternConfig(
            pattern_type=PatternType.TOOL_LEARNING,
            enabled=True,
            priority=5,
            task_type_affinity={
                TaskType.BUG_FIX: 0.7,
                TaskType.NEW_FEATURE: 0.7,
                TaskType.REFACTOR: 0.6,
                TaskType.CODE_REVIEW: 0.5,
                TaskType.DOCUMENTATION: 0.4,
                TaskType.TESTING: 0.6,
                TaskType.EXPLORATION: 0.8,
                TaskType.GENERAL: 0.6
            }
        ),
        PatternType.DEBATE: PatternConfig(
            pattern_type=PatternType.DEBATE,
            enabled=True,
            priority=4,
            task_type_affinity={
                TaskType.BUG_FIX: 0.5,
                TaskType.NEW_FEATURE: 0.6,
                TaskType.REFACTOR: 0.6,
                TaskType.CODE_REVIEW: 0.95,
                TaskType.DOCUMENTATION: 0.3,
                TaskType.TESTING: 0.4,
                TaskType.EXPLORATION: 0.2,
                TaskType.GENERAL: 0.4
            }
        )
    }

    def __init__(self, custom_configs: dict[PatternType, PatternConfig] | None = None):
        """Initialize pattern selector.

        Args:
            custom_configs: Override default configurations
        """
        self.configs = dict(self.DEFAULT_CONFIGS)
        if custom_configs:
            self.configs.update(custom_configs)

    def select_patterns(
        self,
        task_type: TaskType,
        complexity: int,
        max_patterns: int = 4
    ) -> list[PatternType]:
        """Select patterns for a task.

        Args:
            task_type: Type of task
            complexity: Task complexity (1-5)
            max_patterns: Maximum patterns to activate

        Returns:
            List of PatternType to activate
        """
        candidates = []

        for pattern_type, config in self.configs.items():
            if config.should_activate(task_type, complexity):
                affinity = config.task_type_affinity.get(task_type, 0.5)
                score = config.priority * affinity * (1 + complexity / 10)
                candidates.append((pattern_type, score))

        # Sort by score and return top patterns
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in candidates[:max_patterns]]


class AdaptiveOrchestrator:
    """Orchestrates pattern selection and execution.

    Main coordinator that:
    1. Analyzes incoming tasks
    2. Selects appropriate patterns
    3. Configures pattern execution
    4. Monitors and adapts during execution
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        console: Any | None = None,
        verbose: bool = False,
        custom_configs: dict[PatternType, PatternConfig] | None = None
    ):
        """Initialize orchestrator.

        Args:
            llm: Language model for advanced analysis
            console: Rich console for output
            verbose: Verbose output flag
            custom_configs: Custom pattern configurations
        """
        self.llm = llm
        self.console = console
        self.verbose = verbose

        self.task_analyzer = TaskAnalyzer()
        self.pattern_selector = PatternSelector(custom_configs)

        self.current_plan: OrchestrationPlan | None = None
        self.execution_history: list[OrchestrationPlan] = []

    def create_orchestration_plan(
        self,
        state: EnhancedAgentState
    ) -> OrchestrationPlan:
        """Create an orchestration plan for current task.

        Args:
            state: Current agent state

        Returns:
            OrchestrationPlan with selected patterns
        """
        # Extract task description
        task_description = get_current_user_query(state)

        # Analyze task
        task_type, complexity = self.task_analyzer.analyze(task_description)

        # Select patterns
        active_patterns = self.pattern_selector.select_patterns(
            task_type=task_type,
            complexity=complexity
        )

        # Build pattern-specific configs
        pattern_configs = self._build_pattern_configs(
            task_type=task_type,
            complexity=complexity,
            active_patterns=active_patterns
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            task_type=task_type,
            complexity=complexity,
            active_patterns=active_patterns
        )

        plan = OrchestrationPlan(
            task_type=task_type,
            complexity=complexity,
            active_patterns=active_patterns,
            pattern_configs=pattern_configs,
            reasoning=reasoning
        )

        self.current_plan = plan
        self.execution_history.append(plan)

        if self.console and self.verbose:
            self.console.print(
                f"[bold cyan]🎯 Orchestration Plan:[/bold cyan] "
                f"Task={task_type.value}, Complexity={complexity}/5"
            )
            self.console.print(
                f"   Active patterns: {', '.join(p.value for p in active_patterns)}"
            )

        return plan

    def _build_pattern_configs(
        self,
        task_type: TaskType,
        complexity: int,
        active_patterns: list[PatternType]
    ) -> dict[PatternType, dict[str, Any]]:
        """Build pattern-specific configurations."""
        configs = {}

        for pattern in active_patterns:
            config: dict[str, Any] = {"enabled": True}

            if pattern == PatternType.REFLECTION:
                config["max_iterations"] = 2 if complexity <= 3 else 3
                config["confidence_threshold"] = 0.6

            elif pattern == PatternType.MEMORY_BANK:
                config["max_memories"] = 3 if complexity <= 3 else 5
                config["include_failures"] = task_type == TaskType.BUG_FIX

            elif pattern == PatternType.BACKTRACKING:
                config["auto_checkpoint"] = complexity >= 3
                config["checkpoint_interval"] = 3 if complexity <= 3 else 2

            elif pattern == PatternType.HIERARCHICAL_PLANNING:
                config["max_levels"] = 2 if complexity <= 2 else 3
                config["detailed_operational"] = complexity >= 4

            elif pattern == PatternType.TOOL_LEARNING:
                config["recommendation_count"] = 3
                config["use_sequences"] = complexity >= 3

            elif pattern == PatternType.DEBATE:
                config["max_rounds"] = 2 if complexity <= 3 else 3
                config["min_confidence"] = 0.7

            configs[pattern] = config

        return configs

    def _generate_reasoning(
        self,
        task_type: TaskType,
        complexity: int,
        active_patterns: list[PatternType]
    ) -> str:
        """Generate reasoning for pattern selection."""
        reasons = []

        task_reasons = {
            TaskType.BUG_FIX: "Bug fix requires careful verification and rollback capability",
            TaskType.NEW_FEATURE: "New feature benefits from hierarchical planning",
            TaskType.REFACTOR: "Refactoring needs backtracking and reflection",
            TaskType.CODE_REVIEW: "Code review benefits from debate pattern",
            TaskType.DOCUMENTATION: "Documentation is straightforward",
            TaskType.TESTING: "Testing needs tool learning for optimal execution",
            TaskType.EXPLORATION: "Exploration relies on memory and tool learning"
        }
        reasons.append(task_reasons.get(task_type, "General task handling"))

        if complexity >= 4:
            reasons.append("High complexity enables all relevant patterns")
        elif complexity <= 2:
            reasons.append("Low complexity limits active patterns")

        return "; ".join(reasons)

    def get_active_patterns(self) -> list[PatternType]:
        """Get currently active patterns."""
        if self.current_plan:
            return self.current_plan.active_patterns
        return []

    def is_pattern_active(self, pattern: PatternType) -> bool:
        """Check if a specific pattern is active."""
        return pattern in self.get_active_patterns()

    def get_pattern_config(self, pattern: PatternType) -> dict[str, Any]:
        """Get configuration for a specific pattern."""
        if self.current_plan and pattern in self.current_plan.pattern_configs:
            return self.current_plan.pattern_configs[pattern]
        return {}

    def adapt_during_execution(
        self,
        state: EnhancedAgentState
    ) -> dict[str, Any]:
        """Adapt pattern selection during execution.

        Called during execution to potentially adjust patterns based on progress.

        Args:
            state: Current agent state

        Returns:
            Any state updates needed
        """
        if not self.current_plan:
            return {}

        updates: dict[str, Any] = {}

        # Check if we should enable additional patterns
        iteration = state.get("iteration_count", 0)
        error_count = len(state.get("error_history", []))

        # Enable backtracking if encountering errors
        if error_count >= 2 and PatternType.BACKTRACKING not in self.current_plan.active_patterns:
            self.current_plan.active_patterns.append(PatternType.BACKTRACKING)
            updates["orchestration_adapted"] = True
            updates["patterns_added"] = ["backtracking"]

            if self.console and self.verbose:
                self.console.print(
                    "[yellow]⚙️ Orchestrator: Enabling backtracking due to errors[/yellow]"
                )

        # Enable reflection if stuck
        if iteration >= 5 and PatternType.REFLECTION not in self.current_plan.active_patterns:
            self.current_plan.active_patterns.append(PatternType.REFLECTION)
            updates["orchestration_adapted"] = True
            updates["patterns_added"] = updates.get("patterns_added", []) + ["reflection"]

            if self.console and self.verbose:
                self.console.print(
                    "[yellow]⚙️ Orchestrator: Enabling reflection due to many iterations[/yellow]"
                )

        return updates

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of orchestration execution."""
        return {
            "total_plans": len(self.execution_history),
            "current_plan": self.current_plan.to_dict() if self.current_plan else None,
            "task_type_distribution": self._get_task_type_distribution(),
            "pattern_usage": self._get_pattern_usage()
        }

    def _get_task_type_distribution(self) -> dict[str, int]:
        """Get distribution of task types."""
        distribution: dict[str, int] = {}
        for plan in self.execution_history:
            key = plan.task_type.value
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def _get_pattern_usage(self) -> dict[str, int]:
        """Get pattern usage counts."""
        usage: dict[str, int] = {}
        for plan in self.execution_history:
            for pattern in plan.active_patterns:
                key = pattern.value
                usage[key] = usage.get(key, 0) + 1
        return usage


class OrchestratorNode:
    """LangGraph node for adaptive orchestration."""

    def __init__(
        self,
        orchestrator: AdaptiveOrchestrator,
        console: Any | None = None,
        verbose: bool = False
    ):
        """Initialize orchestrator node.

        Args:
            orchestrator: AdaptiveOrchestrator instance
            console: Rich console
            verbose: Verbose output flag
        """
        self.orchestrator = orchestrator
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Execute orchestration.

        Args:
            state: Current agent state

        Returns:
            State updates with orchestration plan
        """
        # Check if we need to create a new plan
        existing_plan = state.get("orchestration_plan")

        if not existing_plan:
            # Create new plan
            plan = self.orchestrator.create_orchestration_plan(state)

            return {
                "orchestration_plan": plan.to_dict(),
                "active_patterns": [p.value for p in plan.active_patterns],
                "task_type": plan.task_type.value,
                "task_complexity": plan.complexity,
                "planning_notes": [f"Orchestration: {plan.reasoning}"]
            }
        else:
            # Adapt existing plan
            return self.orchestrator.adapt_during_execution(state)


def create_adaptive_orchestrator(
    llm: BaseChatModel | None = None,
    console: Any | None = None,
    verbose: bool = False,
    custom_configs: dict[PatternType, PatternConfig] | None = None
) -> AdaptiveOrchestrator:
    """Factory function to create AdaptiveOrchestrator.

    Args:
        llm: Language model
        console: Rich console
        verbose: Verbose output
        custom_configs: Custom pattern configurations

    Returns:
        Configured AdaptiveOrchestrator
    """
    return AdaptiveOrchestrator(
        llm=llm,
        console=console,
        verbose=verbose,
        custom_configs=custom_configs
    )
