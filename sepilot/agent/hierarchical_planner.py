"""Hierarchical Planning - Multi-Level Task Decomposition (Claude Code Style)

This module implements hierarchical planning with DYNAMIC DEPTH:
- Strategic Level: High-level goals and approach (complex tasks only)
- Tactical Level: Mid-level task breakdown (medium+ tasks)
- Operational Level: Concrete tool actions (all tasks)

Claude Code Style Enhancements:
- Dynamic plan depth based on task complexity
- Skip LLM for simple tasks (direct plan generation)
- 2-level planning for medium tasks
- Full 3-level planning for complex tasks only

Inspired by:
- Claude Code's efficient task handling
- HTN (Hierarchical Task Network) Planning
- Plan-and-Solve Prompting
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from sepilot.agent.enhanced_state import AgentStrategy, EnhancedAgentState


class PlanDepth(str, Enum):
    """Plan depth based on task complexity (Claude Code style)."""
    MINIMAL = "minimal"      # 1 level: Operational only (simple tasks, no LLM)
    TACTICAL = "tactical"    # 2 levels: Tactical + Operational (medium tasks)
    FULL = "full"           # 3 levels: Strategic + Tactical + Operational (complex)


class PlanLevel(str, Enum):
    """Hierarchical plan levels."""
    STRATEGIC = "strategic"    # High-level goals (what to achieve)
    TACTICAL = "tactical"      # Mid-level tasks (how to approach)
    OPERATIONAL = "operational"  # Low-level actions (specific tools)


class TaskStatus(str, Enum):
    """Status of a hierarchical task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class HierarchicalTask:
    """A task at any level of the hierarchy."""
    task_id: str
    level: PlanLevel
    description: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    strategy: AgentStrategy | None = None
    tools_required: list[str] = field(default_factory=list)
    files_involved: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    estimated_complexity: int = 1  # 1-5 scale
    actual_iterations: int = 0
    notes: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "level": self.level.value,
            "description": self.description,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "status": self.status.value,
            "strategy": self.strategy.value if self.strategy else None,
            "tools_required": self.tools_required,
            "files_involved": self.files_involved,
            "dependencies": self.dependencies,
            "estimated_complexity": self.estimated_complexity,
            "actual_iterations": self.actual_iterations,
            "notes": self.notes
        }


@dataclass
class HierarchicalPlan:
    """Complete hierarchical plan."""
    plan_id: str
    original_task: str
    strategic_goals: list[HierarchicalTask]
    tactical_tasks: list[HierarchicalTask]
    operational_actions: list[HierarchicalTask]
    current_focus: str | None = None  # Current task_id being executed
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_all_tasks(self) -> list[HierarchicalTask]:
        """Get all tasks across all levels."""
        return self.strategic_goals + self.tactical_tasks + self.operational_actions

    def get_task_by_id(self, task_id: str) -> HierarchicalTask | None:
        """Find task by ID."""
        for task in self.get_all_tasks():
            if task.task_id == task_id:
                return task
        return None

    def get_next_pending_task(self, level: PlanLevel | None = None) -> HierarchicalTask | None:
        """Get next pending task, optionally filtered by level."""
        tasks = self.get_all_tasks()
        if level:
            tasks = [t for t in tasks if t.level == level]

        for task in tasks:
            if task.status == TaskStatus.PENDING:
                # Check dependencies
                deps_met = all(
                    self.get_task_by_id(dep_id) and
                    self.get_task_by_id(dep_id).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_met:
                    return task
        return None

    def get_progress(self) -> dict[str, Any]:
        """Get plan execution progress."""
        all_tasks = self.get_all_tasks()
        completed = sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)

        return {
            "total_tasks": len(all_tasks),
            "completed": completed,
            "failed": failed,
            "pending": len(all_tasks) - completed - failed,
            "progress_percent": (completed / len(all_tasks) * 100) if all_tasks else 0,
            "by_level": {
                level.value: {
                    "total": sum(1 for t in all_tasks if t.level == level),
                    "completed": sum(1 for t in all_tasks if t.level == level and t.status == TaskStatus.COMPLETED)
                }
                for level in PlanLevel
            }
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "original_task": self.original_task,
            "strategic_goals": [t.to_dict() for t in self.strategic_goals],
            "tactical_tasks": [t.to_dict() for t in self.tactical_tasks],
            "operational_actions": [t.to_dict() for t in self.operational_actions],
            "current_focus": self.current_focus,
            "progress": self.get_progress()
        }


class HierarchicalPlanner:
    """Creates and manages hierarchical plans (Claude Code Style).

    Dynamic depth planning based on task complexity:
    - MINIMAL (simple): Direct plan, no LLM call
    - TACTICAL (medium): 2-level planning (Tactical + Operational)
    - FULL (complex): 3-level planning (Strategic + Tactical + Operational)
    """

    # Full 3-level decomposition (complex tasks only)
    FULL_DECOMPOSITION_PROMPT = """You are a HIERARCHICAL TASK PLANNER for a coding agent.

Your task: Decompose a COMPLEX task into a 3-level hierarchy.

## Levels:
1. STRATEGIC (1-3 goals): High-level objectives
2. TACTICAL (2-5 tasks per goal): Approach for each goal
3. OPERATIONAL (1-4 actions per task): Specific tool calls

## Available Tools:
find_file, file_read, file_edit, file_write, bash_execute, web_search

## Output Format (JSON):
{
    "strategic_goals": [{"id": "S1", "description": "...", "strategy": "explore|implement|debug|refactor", "complexity": 1-5}],
    "tactical_tasks": [{"id": "T1", "parent": "S1", "description": "...", "strategy": "...", "complexity": 1-5, "dependencies": []}],
    "operational_actions": [{"id": "O1", "parent": "T1", "description": "...", "tools": ["tool1"], "files": ["path"], "dependencies": []}]
}

Be specific, consider dependencies, estimate complexity realistically."""

    # 2-level decomposition (medium tasks)
    TACTICAL_DECOMPOSITION_PROMPT = """You are a TASK PLANNER for a coding agent.

Your task: Decompose a task into 2 levels (NO strategic level needed).

## Levels:
1. TACTICAL (1-3 tasks): Approach steps
2. OPERATIONAL (1-2 actions per task): Specific tool calls

## Available Tools:
find_file, file_read, file_edit, file_write, bash_execute

## Output Format (JSON):
{
    "tactical_tasks": [{"id": "T1", "description": "...", "strategy": "explore|implement|debug|refactor", "complexity": 1-3}],
    "operational_actions": [{"id": "O1", "parent": "T1", "description": "...", "tools": ["tool1"], "files": ["path"]}]
}

Keep it simple and actionable."""

    # Backward compatibility alias
    DECOMPOSITION_PROMPT = FULL_DECOMPOSITION_PROMPT

    # Simple task indicators (no LLM needed)
    SIMPLE_TASK_PATTERNS = [
        "읽어", "보여", "열어", "확인해",
        "read", "show", "display", "open", "view", "cat",
        "status", "log", "list", "ls", "pwd",
    ]

    # Complex task indicators (full planning needed)
    COMPLEX_TASK_PATTERNS = [
        "리팩토링", "리팩터", "전체", "모든", "아키텍처",
        "refactor", "restructure", "all", "entire", "architecture",
        "design", "migrate", "upgrade", "overhaul",
        "그리고", "또한", "다음에", "and then", "also", "after that",
    ]

    # Pre-built plan templates for common task types
    PLAN_TEMPLATES = {
        "bug_fix": {
            "keywords": ["fix", "bug", "수정", "버그", "error", "오류", "broken", "not working"],
            "tactical_tasks": [
                {"id": "T1", "description": "Locate and read the buggy code", "strategy": "explore", "complexity": 2},
                {"id": "T2", "description": "Identify root cause and apply fix", "strategy": "implement", "complexity": 3},
                {"id": "T3", "description": "Verify the fix works correctly", "strategy": "test", "complexity": 2},
            ],
            "operational_actions": [
                {"id": "O1", "parent": "T1", "description": "Search for relevant files", "tools": ["find_file", "search_content"]},
                {"id": "O2", "parent": "T1", "description": "Read the target file", "tools": ["file_read"]},
                {"id": "O3", "parent": "T2", "description": "Edit the file to fix the bug", "tools": ["file_edit"]},
                {"id": "O4", "parent": "T3", "description": "Run tests or verify output", "tools": ["bash_execute"]},
            ],
        },
        "new_feature": {
            "keywords": ["implement", "add", "create", "new", "구현", "추가", "생성", "만들"],
            "tactical_tasks": [
                {"id": "T1", "description": "Explore existing code structure", "strategy": "explore", "complexity": 2},
                {"id": "T2", "description": "Implement the feature", "strategy": "implement", "complexity": 3},
                {"id": "T3", "description": "Test and verify implementation", "strategy": "test", "complexity": 2},
            ],
            "operational_actions": [
                {"id": "O1", "parent": "T1", "description": "Find related files and understand structure", "tools": ["find_file", "file_read"]},
                {"id": "O2", "parent": "T2", "description": "Create or edit files for the feature", "tools": ["file_write", "file_edit"]},
                {"id": "O3", "parent": "T3", "description": "Run tests to verify", "tools": ["bash_execute"]},
            ],
        },
        "refactor": {
            "keywords": ["refactor", "리팩토링", "리팩터", "restructure", "reorganize", "clean up"],
            "tactical_tasks": [
                {"id": "T1", "description": "Analyze current code structure", "strategy": "explore", "complexity": 2},
                {"id": "T2", "description": "Refactor code incrementally", "strategy": "refactor", "complexity": 4},
                {"id": "T3", "description": "Verify no regressions", "strategy": "test", "complexity": 2},
            ],
            "operational_actions": [
                {"id": "O1", "parent": "T1", "description": "Read files to understand dependencies", "tools": ["file_read", "search_content"]},
                {"id": "O2", "parent": "T2", "description": "Edit files for refactoring", "tools": ["file_edit"]},
                {"id": "O3", "parent": "T3", "description": "Run tests after refactoring", "tools": ["bash_execute"]},
            ],
        },
        "code_review": {
            "keywords": ["review", "check", "analyze", "검토", "리뷰", "분석"],
            "tactical_tasks": [
                {"id": "T1", "description": "Read and understand the code", "strategy": "explore", "complexity": 2},
                {"id": "T2", "description": "Analyze code quality and issues", "strategy": "explore", "complexity": 2},
            ],
            "operational_actions": [
                {"id": "O1", "parent": "T1", "description": "Read target files", "tools": ["file_read"]},
                {"id": "O2", "parent": "T2", "description": "Search for patterns and issues", "tools": ["search_content"]},
            ],
        },
    }

    def _match_plan_template(self, task_description: str) -> dict[str, Any] | None:
        """Match task description to a pre-built plan template.

        Returns template dict if matched, None otherwise.
        """
        task_lower = task_description.lower()
        for _template_name, template in self.PLAN_TEMPLATES.items():
            if any(kw in task_lower for kw in template["keywords"]):
                return template
        return None

    def __init__(
        self,
        llm: BaseChatModel,
        console: Any | None = None,
        verbose: bool = False
    ):
        """Initialize hierarchical planner.

        Args:
            llm: Language model for planning
            console: Rich console for output
            verbose: Verbose output flag
        """
        self.llm = llm
        self.console = console
        self.verbose = verbose
        self._plan_counter = 0

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        self._plan_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"plan_{timestamp}_{self._plan_counter:04d}"

    def assess_plan_depth(self, task_description: str) -> PlanDepth:
        """Assess optimal plan depth based on task complexity (Claude Code style).

        Returns:
            PlanDepth.MINIMAL - Simple tasks (no LLM)
            PlanDepth.TACTICAL - Medium tasks (2-level LLM)
            PlanDepth.FULL - Complex tasks (3-level LLM)
        """
        task_lower = task_description.lower()

        # Check for simple task patterns
        simple_count = sum(1 for p in self.SIMPLE_TASK_PATTERNS if p in task_lower)

        # Check for complex task patterns
        complex_count = sum(1 for p in self.COMPLEX_TASK_PATTERNS if p in task_lower)

        # Length-based heuristic
        word_count = len(task_description.split())

        # Decision logic
        if simple_count > 0 and complex_count == 0 and word_count < 30:
            return PlanDepth.MINIMAL
        elif complex_count > 0 or word_count > 100:
            return PlanDepth.FULL
        else:
            return PlanDepth.TACTICAL

    def create_plan(
        self,
        task_description: str,
        context: dict[str, Any] | None = None,
        depth: PlanDepth | str | None = None
    ) -> HierarchicalPlan:
        """Create a hierarchical plan with dynamic depth (Claude Code style).

        Args:
            task_description: Original task description
            context: Additional context (files, codebase info)
            depth: Plan depth (auto-detected if None)

        Returns:
            HierarchicalPlan with appropriate depth
        """
        plan_id = self._generate_plan_id()

        # Auto-detect depth if not specified
        if depth is None:
            depth = self.assess_plan_depth(task_description)
        elif isinstance(depth, str):
            depth = PlanDepth(depth)

        # MINIMAL depth: No LLM call, direct plan generation
        if depth == PlanDepth.MINIMAL:
            if self.console and self.verbose:
                self.console.print("[dim cyan]⚡ Simple task → direct plan (no LLM)[/dim cyan]")
            return self._create_direct_plan(plan_id, task_description, context)

        # TACTICAL depth: try template first (skip LLM if template matches)
        if depth == PlanDepth.TACTICAL:
            template = self._match_plan_template(task_description)
            if template:
                if self.console and self.verbose:
                    self.console.print("[dim cyan]⚡ Template matched → skip LLM planning[/dim cyan]")
                return self._create_plan_from_template(plan_id, task_description, template, context)

        # Build context string for LLM
        context_str = self._build_context_string(context)

        # Select prompt based on depth
        if depth == PlanDepth.TACTICAL:
            prompt = self.TACTICAL_DECOMPOSITION_PROMPT
            user_prompt = f"Task: {task_description}\n{context_str}\nCreate tactical and operational steps."
        else:  # FULL
            prompt = self.FULL_DECOMPOSITION_PROMPT
            user_prompt = f"Task: {task_description}\n{context_str}\nCreate strategic, tactical, and operational plan."

        try:
            from sepilot.agent.output_validator import OutputValidator

            original_messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(original_messages)

            # Validate JSON response with OutputValidator
            import json
            raw_content = response.content.strip()
            is_valid, plan_data = OutputValidator.validate_json(raw_content)

            if not is_valid:
                # Retry once with correction prompt
                corrected = OutputValidator.retry_with_correction(
                    llm=self.llm,
                    original_messages=original_messages,
                    original_response=raw_content,
                    error_desc="Expected valid JSON with keys: tactical_tasks, operational_actions",
                    max_retries=1,
                )
                if corrected:
                    is_valid, plan_data = OutputValidator.validate_json(corrected)

                if not is_valid:
                    raise json.JSONDecodeError("OutputValidator: JSON extraction failed", raw_content, 0)

            # Build tasks based on depth
            if depth == PlanDepth.FULL:
                strategic_goals = self._build_strategic_tasks(plan_data.get("strategic_goals", []))
            else:
                strategic_goals = []  # No strategic level for TACTICAL

            tactical_tasks = self._build_tactical_tasks(plan_data.get("tactical_tasks", []))
            operational_actions = self._build_operational_tasks(plan_data.get("operational_actions", []))

            # Link parent-child relationships
            self._link_hierarchy(strategic_goals, tactical_tasks, operational_actions)

            plan = HierarchicalPlan(
                plan_id=plan_id,
                original_task=task_description,
                strategic_goals=strategic_goals,
                tactical_tasks=tactical_tasks,
                operational_actions=operational_actions,
                metadata={"context": context, "depth": depth.value}
            )

            if self.console and self.verbose:
                progress = plan.get_progress()
                depth_label = "Full" if depth == PlanDepth.FULL else "Tactical"
                self.console.print(
                    f"[cyan]📋 {depth_label} plan: {progress['total_tasks']} tasks "
                    f"(S:{len(strategic_goals)}, T:{len(tactical_tasks)}, O:{len(operational_actions)})[/cyan]"
                )

            return plan

        except json.JSONDecodeError as e:
            # JSON parsing failed - try simpler depth first
            if depth == PlanDepth.FULL:
                if self.console:
                    self.console.print("[yellow]⚠️ Full plan parsing failed. Trying tactical depth...[/yellow]")
                return self.create_plan(task_description, context, depth=PlanDepth.TACTICAL)
            elif depth == PlanDepth.TACTICAL:
                if self.console:
                    self.console.print("[yellow]⚠️ Tactical plan parsing failed. Using direct plan...[/yellow]")
                return self._create_direct_plan(plan_id, task_description, context)
            else:
                return self._create_direct_plan(plan_id, task_description, context)

        except Exception as e:
            # Other errors - fall back to direct plan
            if self.console:
                self.console.print(f"[yellow]⚠️ Planning failed ({type(e).__name__}): {str(e)[:50]}. Using direct plan.[/yellow]")
            return self._create_direct_plan(plan_id, task_description, context)

    def _build_context_string(self, context: dict[str, Any] | None) -> str:
        """Build context string for LLM prompt."""
        if not context:
            return ""

        parts = []
        if context.get("exploration_context"):
            parts.append(context["exploration_context"])
        if context.get("memory_context"):
            parts.append(context["memory_context"])
        if context.get("files"):
            parts.append(f"Files: {', '.join(context['files'][:10])}")
        if context.get("recent_errors"):
            parts.append(f"Errors: {context['recent_errors']}")

        return "\n".join(parts)

    def _create_direct_plan(
        self,
        plan_id: str,
        task_description: str,
        context: dict[str, Any] | None = None
    ) -> HierarchicalPlan:
        """Create a direct plan without LLM (for simple tasks).

        Claude Code style: Simple tasks don't need hierarchical decomposition.
        """
        # Detect likely tools from task description
        task_lower = task_description.lower()
        tools = []

        if any(p in task_lower for p in ["읽", "read", "show", "보여", "열어", "open", "view"]):
            tools = ["file_read"]
        elif any(p in task_lower for p in ["수정", "변경", "edit", "change", "modify", "fix"]):
            tools = ["file_read", "file_edit"]
        elif any(p in task_lower for p in ["생성", "만들", "create", "write", "new"]):
            tools = ["file_write"]
        elif any(p in task_lower for p in ["실행", "run", "execute", "bash", "명령"]):
            tools = ["bash_execute"]
        elif any(p in task_lower for p in ["찾", "find", "search", "검색"]):
            tools = ["find_file", "file_read"]
        else:
            tools = ["file_read", "file_edit"]  # Default

        # Extract file hints from context
        files = []
        if context and context.get("files"):
            files = context["files"][:3]

        # Create minimal plan structure
        operational = HierarchicalTask(
            task_id="O1",
            level=PlanLevel.OPERATIONAL,
            description=f"Execute: {task_description[:100]}",
            tools_required=tools,
            files_involved=files,
            estimated_complexity=1
        )

        return HierarchicalPlan(
            plan_id=plan_id,
            original_task=task_description,
            strategic_goals=[],  # No strategic for simple tasks
            tactical_tasks=[],   # No tactical for simple tasks
            operational_actions=[operational],
            metadata={"context": context, "depth": PlanDepth.MINIMAL.value}
        )

    def _create_plan_from_template(
        self,
        plan_id: str,
        task_description: str,
        template: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> HierarchicalPlan:
        """Create a plan from a pre-built template (no LLM call).

        For TACTICAL depth tasks that match a known template.
        """
        tactical_tasks = self._build_tactical_tasks(template["tactical_tasks"])
        operational_actions = self._build_operational_tasks(template["operational_actions"])
        self._link_hierarchy([], tactical_tasks, operational_actions)

        # Enrich with context files if available
        if context and context.get("files"):
            for action in operational_actions:
                if not action.files_involved:
                    action.files_involved = context["files"][:3]

        return HierarchicalPlan(
            plan_id=plan_id,
            original_task=task_description,
            strategic_goals=[],
            tactical_tasks=tactical_tasks,
            operational_actions=operational_actions,
            metadata={"context": context, "depth": PlanDepth.TACTICAL.value, "source": "template"}
        )

    def _build_strategic_tasks(self, goals_data: list[dict]) -> list[HierarchicalTask]:
        """Build strategic level tasks."""
        tasks = []
        for i, goal in enumerate(goals_data):
            task = HierarchicalTask(
                task_id=goal.get("id", f"S{i+1}"),
                level=PlanLevel.STRATEGIC,
                description=goal.get("description", "Unknown goal"),
                strategy=self._parse_strategy(goal.get("strategy")),
                estimated_complexity=goal.get("complexity", 3)
            )
            tasks.append(task)
        return tasks

    def _build_tactical_tasks(self, tasks_data: list[dict]) -> list[HierarchicalTask]:
        """Build tactical level tasks."""
        tasks = []
        for i, task_data in enumerate(tasks_data):
            task = HierarchicalTask(
                task_id=task_data.get("id", f"T{i+1}"),
                level=PlanLevel.TACTICAL,
                description=task_data.get("description", "Unknown task"),
                parent_id=task_data.get("parent"),
                strategy=self._parse_strategy(task_data.get("strategy")),
                estimated_complexity=task_data.get("complexity", 2),
                dependencies=task_data.get("dependencies", [])
            )
            tasks.append(task)
        return tasks

    def _build_operational_tasks(self, actions_data: list[dict]) -> list[HierarchicalTask]:
        """Build operational level tasks."""
        tasks = []
        for i, action in enumerate(actions_data):
            task = HierarchicalTask(
                task_id=action.get("id", f"O{i+1}"),
                level=PlanLevel.OPERATIONAL,
                description=action.get("description", "Unknown action"),
                parent_id=action.get("parent"),
                tools_required=action.get("tools", []),
                files_involved=action.get("files", []),
                dependencies=action.get("dependencies", []),
                estimated_complexity=1
            )
            tasks.append(task)
        return tasks

    def _parse_strategy(self, strategy_str: str | None) -> AgentStrategy | None:
        """Parse strategy string to enum."""
        if not strategy_str:
            return None
        try:
            return AgentStrategy(strategy_str.lower())
        except ValueError:
            return AgentStrategy.IMPLEMENT

    def _link_hierarchy(
        self,
        strategic: list[HierarchicalTask],
        tactical: list[HierarchicalTask],
        operational: list[HierarchicalTask]
    ) -> None:
        """Link parent-child relationships."""
        # Build lookup
        all_tasks = {t.task_id: t for t in strategic + tactical + operational}

        # Link children to parents
        for task in tactical + operational:
            if task.parent_id and task.parent_id in all_tasks:
                parent = all_tasks[task.parent_id]
                if task.task_id not in parent.children_ids:
                    parent.children_ids.append(task.task_id)

    def _create_fallback_plan(self, plan_id: str, task_description: str) -> HierarchicalPlan:
        """Create a simple fallback plan (backward compatibility wrapper)."""
        # Delegate to _create_direct_plan for consistency
        return self._create_direct_plan(plan_id, task_description)

    def update_task_status(
        self,
        plan: HierarchicalPlan,
        task_id: str,
        status: TaskStatus,
        notes: list[str] | None = None
    ) -> None:
        """Update status of a task.

        Args:
            plan: The plan containing the task
            task_id: ID of task to update
            status: New status
            notes: Optional notes to add
        """
        task = plan.get_task_by_id(task_id)
        if task:
            task.status = status
            if notes:
                task.notes.extend(notes)
            if status == TaskStatus.COMPLETED:
                task.completed_at = datetime.now()

            # Propagate status up if all children completed
            if task.parent_id:
                parent = plan.get_task_by_id(task.parent_id)
                if parent:
                    children = [plan.get_task_by_id(cid) for cid in parent.children_ids]
                    if all(c and c.status == TaskStatus.COMPLETED for c in children):
                        parent.status = TaskStatus.COMPLETED

    def get_current_focus(self, plan: HierarchicalPlan) -> HierarchicalTask | None:
        """Get the current task to focus on.

        Args:
            plan: Current hierarchical plan

        Returns:
            Next task to execute
        """
        # First, try to get operational task (most concrete)
        op_task = plan.get_next_pending_task(PlanLevel.OPERATIONAL)
        if op_task:
            return op_task

        # Then tactical
        tac_task = plan.get_next_pending_task(PlanLevel.TACTICAL)
        if tac_task:
            return tac_task

        # Finally strategic
        return plan.get_next_pending_task(PlanLevel.STRATEGIC)

    def format_plan_for_prompt(self, plan: HierarchicalPlan) -> str:
        """Format plan for inclusion in agent prompt.

        Args:
            plan: Hierarchical plan

        Returns:
            Formatted string
        """
        lines = ["## Current Execution Plan\n"]

        # Strategic level
        lines.append("### Strategic Goals:")
        for goal in plan.strategic_goals:
            status_icon = self._status_icon(goal.status)
            lines.append(f"  {status_icon} [{goal.task_id}] {goal.description}")

        # Tactical level
        lines.append("\n### Tactical Tasks:")
        for task in plan.tactical_tasks:
            status_icon = self._status_icon(task.status)
            lines.append(f"  {status_icon} [{task.task_id}] {task.description}")
            if task.strategy:
                lines.append(f"      Strategy: {task.strategy.value}")

        # Operational level (current focus)
        lines.append("\n### Current Actions:")
        for action in plan.operational_actions:
            if action.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                status_icon = self._status_icon(action.status)
                lines.append(f"  {status_icon} [{action.task_id}] {action.description}")
                if action.tools_required:
                    lines.append(f"      Tools: {', '.join(action.tools_required)}")

        # Progress
        progress = plan.get_progress()
        lines.append(f"\n### Progress: {progress['completed']}/{progress['total_tasks']} tasks completed")

        return "\n".join(lines)

    def _status_icon(self, status: TaskStatus) -> str:
        """Get icon for task status."""
        icons = {
            TaskStatus.PENDING: "⬜",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.BLOCKED: "🚫",
            TaskStatus.SKIPPED: "⏭️"
        }
        return icons.get(status, "❓")


class HierarchicalPlanningNode:
    """LangGraph node for hierarchical planning."""

    def __init__(
        self,
        planner: HierarchicalPlanner,
        console: Any | None = None,
        verbose: bool = False
    ):
        """Initialize planning node.

        Args:
            planner: HierarchicalPlanner instance
            console: Rich console
            verbose: Verbose output flag
        """
        self.planner = planner
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Execute hierarchical planning.

        Args:
            state: Current agent state

        Returns:
            State updates with hierarchical plan
        """
        # Get task from messages
        messages = state.get("messages", [])
        task_description = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task_description = getattr(msg, "content", "")
                break

        if not task_description:
            return {}

        # Check if we already have a hierarchical plan
        existing_plan = state.get("hierarchical_plan")
        if existing_plan:
            # Update focus
            plan = existing_plan  # Would need deserialization in real impl
            return {"hierarchical_plan_focus": self.planner.get_current_focus(plan)}

        # Create new hierarchical plan
        context = {
            "files": [fc.file_path for fc in state.get("file_changes", [])],
            "recent_errors": [e.message for e in state.get("error_history", [])[-3:]]
        }

        plan = self.planner.create_plan(task_description, context)

        # Convert to flat plan_steps for compatibility
        flat_steps = []
        for goal in plan.strategic_goals:
            flat_steps.append(f"[STRATEGIC] {goal.description}")
        for task in plan.tactical_tasks:
            flat_steps.append(f"[TACTICAL] {task.description}")
        for action in plan.operational_actions:
            tools = ", ".join(action.tools_required) if action.tools_required else "determine tools"
            flat_steps.append(f"[OPERATIONAL] {action.description} (tools: {tools})")

        return {
            "hierarchical_plan": plan.to_dict(),
            "plan_steps": flat_steps,
            "plan_created": True,
            "current_plan_step": 0,
            "planning_notes": [
                f"Hierarchical plan created with {len(plan.get_all_tasks())} tasks",
                self.planner.format_plan_for_prompt(plan)
            ]
        }


def create_hierarchical_planner(
    llm: BaseChatModel,
    console: Any | None = None,
    verbose: bool = False
) -> HierarchicalPlanner:
    """Factory function to create HierarchicalPlanner.

    Args:
        llm: Language model
        console: Rich console
        verbose: Verbose output

    Returns:
        Configured HierarchicalPlanner
    """
    return HierarchicalPlanner(llm=llm, console=console, verbose=verbose)
