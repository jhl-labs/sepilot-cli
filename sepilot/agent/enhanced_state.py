"""Enhanced state management for production-level coding agent

This module provides a rich state structure similar to Claude Code, Cursor, and other
production coding agents. It tracks tasks, file changes, errors, tool usage, and more.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from sepilot.config.constants import (
    FILE_CHANGE_MERGE_WINDOW_SECONDS,
    MAX_ERROR_HISTORY,
    MAX_FILE_CHANGES_HISTORY,
    MAX_PLANNING_NOTES,
    MAX_REFLECTION_NOTES,
    MAX_SCRATCHPAD_ENTRIES,
    MAX_STRATEGY_HISTORY,
    MAX_TASK_HISTORY,
    MAX_TOOL_CALL_HISTORY,
    MAX_VERIFICATION_NOTES,
)

# ============================================================================
# Enums
# ============================================================================

class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class FileAction(str, Enum):
    """File operation types"""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    READ = "read"


class ProcessStatus(str, Enum):
    """Process execution status"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class ErrorLevel(str, Enum):
    """Error/Warning levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AgentMode(str, Enum):
    """Agent operational mode - controls tool availability.

    Orthogonal to AgentStrategy:
    - AgentMode: "어떤 도구를 사용할 수 있는가" — tool access gate
    - AgentStrategy: "어떤 전략적 의도인가" — LLM thinking guide
    """
    PLAN = "plan"    # Read-only: explore, understand, strategize
    CODE = "code"    # Read + Write + Bash: edit files, implement changes
    EXEC = "exec"    # Read + Execute: run commands, verify, test
    AUTO = "auto"    # System auto-decides (default, backward-compatible)


class AgentStrategy(str, Enum):
    """Agent's current working strategy"""
    EXPLORE = "explore"          # Exploring codebase
    IMPLEMENT = "implement"      # Implementing new feature
    DEBUG = "debug"              # Debugging issues
    REFACTOR = "refactor"        # Refactoring code
    TEST = "test"                # Writing/running tests
    DOCUMENT = "document"        # Writing documentation


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TaskContext:
    """Context for current task being executed"""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    subtasks: list[str] = field(default_factory=list)
    parent_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "subtasks": self.subtasks,
            "parent_task_id": self.parent_task_id,
            "metadata": self.metadata
        }


@dataclass
class FileChange:
    """File modification tracking (Git-like)"""
    file_path: str
    action: FileAction
    old_content: str | None = None
    new_content: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    tool_used: str | None = None
    diff: str | None = None  # Unified diff format
    lines_added: int = 0
    lines_removed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_path": self.file_path,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
            "tool_used": self.tool_used,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "has_diff": self.diff is not None
        }


@dataclass
class ProcessInfo:
    """Running process information"""
    process_id: str
    command: str
    started_at: datetime = field(default_factory=datetime.now)
    status: ProcessStatus = ProcessStatus.RUNNING
    output: str = ""
    error_output: str = ""
    exit_code: int | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "process_id": self.process_id,
            "command": self.command,
            "started_at": self.started_at.isoformat(),
            "status": self.status.value,
            "exit_code": self.exit_code,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class ErrorRecord:
    """Error/Warning record"""
    timestamp: datetime
    level: ErrorLevel
    message: str
    source: str  # "tool", "llm", "system", "user"
    context: dict[str, Any] = field(default_factory=dict)
    stack_trace: str | None = None
    resolved: bool = False
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "context": self.context,
            "resolved": self.resolved,
            "resolution": self.resolution
        }


@dataclass
class ToolCallRecord:
    """Tool invocation record"""
    tool_name: str
    timestamp: datetime
    args: dict[str, Any]
    result: Any
    duration: float  # seconds
    success: bool
    error: str | None = None
    tokens_used: int = 0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat(),
            "args": {k: str(v)[:100] for k, v in self.args.items()},  # Truncate
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "cached": self.cached
        }


@dataclass
class ApprovalRequest:
    """User approval request for sensitive operations"""
    request_id: str
    action: str
    description: str
    preview: str
    risk_level: str  # "low", "medium", "high"
    created_at: datetime = field(default_factory=datetime.now)
    approved: bool | None = None
    user_response: str | None = None


# ============================================================================
# State Reducers
# ============================================================================

def add_file_change(existing: list[FileChange], new: FileChange | list[FileChange]) -> list[FileChange]:
    """Add file change(s) to the list

    Merges changes to the same file within a short time window.

    Performance: O(n) instead of O(n²) using dict for fast lookup
    """
    if isinstance(new, FileChange):
        new = [new]

    # Create a dict for O(1) lookup by file_path
    # Key: file_path, Value: (index, FileChange)
    file_map = {change.file_path: (i, change) for i, change in enumerate(existing)}

    result = existing.copy()

    for change in new:
        # O(1) lookup instead of O(n) iteration
        if change.file_path in file_map:
            idx, existing_change = file_map[change.file_path]
            # Check if within time window
            if (change.timestamp - existing_change.timestamp).total_seconds() < FILE_CHANGE_MERGE_WINDOW_SECONDS:
                # Merge the changes (replace)
                result[idx] = change
                file_map[change.file_path] = (idx, change)
            else:
                # Time window expired, add as new change
                result.append(change)
                file_map[change.file_path] = (len(result) - 1, change)
        else:
            # New file, add to result
            result.append(change)
            file_map[change.file_path] = (len(result) - 1, change)

    # Keep only last N changes
    if len(result) > MAX_FILE_CHANGES_HISTORY:
        result = result[-MAX_FILE_CHANGES_HISTORY:]

    return result


def add_error_record(existing: list[ErrorRecord], new: ErrorRecord | list[ErrorRecord]) -> list[ErrorRecord]:
    """Add error record(s) to the list

    Keeps only the most recent 100 errors.
    """
    if isinstance(new, ErrorRecord):
        new = [new]

    result = existing + new

    # Keep only last N errors
    if len(result) > MAX_ERROR_HISTORY:
        result = result[-MAX_ERROR_HISTORY:]

    return result


def add_tool_call(existing: list[ToolCallRecord], new: ToolCallRecord | list[ToolCallRecord]) -> list[ToolCallRecord]:
    """Add tool call record(s) to the list

    Keeps only the most recent 200 tool calls.
    """
    if isinstance(new, ToolCallRecord):
        new = [new]

    result = existing + new

    # Keep only last N tool calls
    if len(result) > MAX_TOOL_CALL_HISTORY:
        result = result[-MAX_TOOL_CALL_HISTORY:]

    return result


def _bounded_list_reducer(max_size: int):
    """Create a list reducer that appends new items and trims to max_size."""
    def reducer(old: list, new: list) -> list:
        result = old + new
        if len(result) > max_size:
            result = result[-max_size:]
        return result
    return reducer


# ============================================================================
# Enhanced State Definition
# ============================================================================

class EnhancedAgentState(TypedDict, total=False):
    """Enhanced state for production-level coding agent

    This state structure is inspired by Claude Code, Cursor, and other
    production coding agents. It provides comprehensive tracking of:
    - Tasks and their progress
    - File modifications (Git-like)
    - Running processes
    - Errors and warnings
    - Tool usage and caching
    - User interactions
    - Session metadata

    All fields are optional (total=False) to allow gradual migration.
    """

    # ===== Core (from original AgentState) =====
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ===== Task Management =====
    current_task: TaskContext | None
    """Currently executing task"""

    task_history: Annotated[list[TaskContext], _bounded_list_reducer(MAX_TASK_HISTORY)]
    """History of completed/failed tasks"""

    # ===== File System =====
    working_directory: str
    """Current working directory"""

    open_files: dict[str, str]
    """Currently open files (path -> content)"""

    file_changes: Annotated[list[FileChange], add_file_change]
    """All file modifications tracked (Git-like)"""

    staged_changes: list[FileChange]
    """Changes staged for commit"""

    # ===== Execution Environment =====
    active_processes: dict[str, ProcessInfo]
    """Currently running processes"""

    environment_vars: dict[str, str]
    """Environment variables"""

    # ===== Error & Warning Management =====
    error_history: Annotated[list[ErrorRecord], add_error_record]
    """Complete error history (last 100)"""

    recent_errors: list[ErrorRecord]
    """Most recent errors (last 10)"""

    warning_count: int
    """Total warning count"""

    # ===== Tool Usage & Caching =====
    tool_call_history: Annotated[list[ToolCallRecord], add_tool_call]
    """Complete tool call history (last 200)"""

    tool_results_cache: dict[str, Any]
    """Cache for expensive tool results"""

    tools_used_count: dict[str, int]
    """Count of each tool usage"""

    # ===== User Interaction =====
    pending_user_input: str | None
    """Waiting for user input/clarification"""

    user_feedback: list[str]
    """User feedback messages"""

    pending_approvals: list[ApprovalRequest]
    """Operations awaiting user approval"""

    # ===== Session Metadata =====
    session_id: str
    """Unique session identifier"""

    session_start: datetime
    """Session start time"""

    iteration_count: int
    """Number of agent iterations"""

    max_iterations: int
    """Maximum allowed iterations"""

    total_tokens_used: int
    """Total tokens consumed"""

    estimated_cost: float
    """Estimated cost in USD"""

    plan_created: bool
    """Whether the initial execution plan has been generated"""

    plan_steps: list[str]
    """Parsed individual steps from the plan (e.g., ['1. find_file', '2. file_read', '3. Provide analysis'])"""

    current_plan_step: int
    """Index of the current step being executed (0-based)"""

    planning_notes: Annotated[list[str], _bounded_list_reducer(MAX_PLANNING_NOTES)]
    """History of planning notes / outlines"""

    verification_notes: Annotated[list[str], _bounded_list_reducer(MAX_VERIFICATION_NOTES)]
    """Post-execution verification notes"""

    needs_additional_iteration: bool
    """Flag set by verifier when another pass is required"""

    required_files: list[str]
    """Files explicitly mentioned in the prompt that must be updated"""

    plan_execution_pending: bool
    """True when a plan exists but 실행 단계가 아직 진행되지 않은 상태"""

    missing_tasks: list[str]
    """Tasks identified by verifier that still need completion"""

    last_approval_status: str | None
    """Status code from the last human approval step"""

    force_termination: bool
    """Set by iteration guard when execution must stop"""

    consecutive_llm_errors: int
    """Count of consecutive LLM invocation failures (reset on success)"""

    _skip_compaction: bool
    """Skip context compaction for this iteration (e.g., after injecting nudge message)"""

    # ===== Think / Scratchpad =====
    scratchpad_entries: Annotated[list[dict], _bounded_list_reducer(MAX_SCRATCHPAD_ENTRIES)]
    """Agent's intermediate thoughts (category, content, timestamp)"""

    # ===== Memory & Context =====
    conversation_summary: str | None
    """Summary for long conversations"""

    important_context: list[str]
    """Key information to retain"""

    triage_decision: str | None
    """Initial routing decision (direct_response vs graph)"""

    triage_reason: str | None
    """Explanation for the triage decision"""

    # ===== Agent Mode (PLAN/CODE/EXEC) =====
    current_mode: AgentMode
    """Current operational mode — controls tool availability"""

    mode_locked: bool
    """Whether user manually locked the mode (disables auto-transition)"""

    mode_history: Annotated[list[str], lambda old, new: old + new]
    """Mode transition history (e.g., 'iter=3: plan→code')"""

    mode_iteration_count: int
    """Iterations spent in current mode (reset on mode transition)"""

    # ===== Agent Behavior =====
    current_strategy: AgentStrategy
    """Current working strategy"""

    confidence_level: float
    """Confidence in current approach (0.0 - 1.0)"""

    needs_clarification: bool
    """Whether agent needs user clarification"""

    use_enhanced_state: bool
    """Whether Enhanced State tracking is enabled"""

    # ===== Query Analysis =====
    query_complexity: str | None
    """Query complexity classification: 'simple' or 'complex'"""

    expected_tool_count: int
    """Estimated number of tool calls needed to complete the task"""

    _task_complete_pending_response: bool
    """Flag indicating task is complete but awaiting agent's summary response"""

    _patch_review_done: bool
    """Flag indicating patch self-review has been performed (prevent repeated reviews)"""

    # ===== Codebase Exploration =====
    exploration_performed: bool
    """Whether automatic codebase exploration was performed"""

    exploration_skipped: bool
    """Whether exploration was skipped (explicit files provided)"""

    exploration_context: str
    """Formatted exploration results for planning context"""

    exploration_results: dict[str, Any]
    """Raw exploration results (files, matches, etc.)"""

    exploration_hints: list[str]
    """Keywords/hints extracted for exploration"""

    explicit_files: list[str]
    """Explicitly mentioned file paths from user request"""

    project_type: str | None
    """Detected project type (python, javascript, etc.)"""

    # ===== Self-Reflection (Reflexion Pattern) =====
    reflection_count: int
    """Number of reflection iterations performed"""

    reflection_decision: str | None
    """Most recent reflection decision: revise_plan, refine_strategy, proceed, escalate"""

    reflection_notes: Annotated[list[str], _bounded_list_reducer(MAX_REFLECTION_NOTES)]
    """History of reflection insights and decisions"""

    failure_patterns: list[str]
    """Currently detected failure patterns"""

    strategy_adjustment_history: Annotated[list[str], _bounded_list_reducer(MAX_STRATEGY_HISTORY)]
    """History of strategy adjustments made during execution"""

    last_reflection_insight: str | None
    """Most recent reflection insight for context"""

    # ===== Execution Tracking =====
    task_complexity: str | None
    """Task complexity from unified triage: 'simple' or 'complex'"""

    repetition_detected: bool
    """Whether tool repetition was detected by recursion detector"""

    repetition_info: dict[str, Any] | None
    """Details about detected repetition pattern"""

    stagnation_detected: bool
    """Whether agent stagnation was detected"""

    consecutive_denials: int
    """Count of consecutive user approval denials"""

    task_type_detected: str | None
    """Detected task type from orchestrator"""

    tests_requested: bool
    """Whether testing was requested by user or plan"""

    lint_requested: bool
    """Whether linting was requested by user or plan"""

    debate_decision: str | None
    """Decision from internal debate evaluation"""

    _last_compaction_iter: int
    """Iteration number of last context compaction (cooldown tracking)"""

    _initial_message_count: int
    """Number of messages in the initial state (system + context + prompt).
    Used by reporter to distinguish context messages from current execution messages.
    WARNING: This integer index is unreliable when LangGraph checkpoint accumulates
    messages across execute() calls. Use _execution_boundary_msg_id instead."""

    _execution_boundary_msg_id: str | None
    """ID of the HumanMessage prompt that starts the current execution.
    Used to find the execution boundary in messages list, robust against
    checkpoint message accumulation via add_messages reducer."""

    _execution_deadline_monotonic: float | None
    """Absolute monotonic deadline for a single execute() call."""

    # ===== Backtracking State =====
    backtrack_decision: str | None
    """Decision from backtracking evaluation"""

    backtrack_performed: bool
    """Whether a backtrack was performed"""

    backtrack_reason: str | None
    """Reason for backtracking"""

    files_restored: list[str]
    """Files restored during backtracking"""

    active_patterns: list[str]
    """Active failure patterns detected during execution"""

    # ===== File Tracking =====
    modified_files: list[str]
    """Files modified during tool execution (from git snapshot)"""

    file_changes_count: int
    """Count of file changes during execution"""


# ============================================================================
# Factory Functions
# ============================================================================

def create_initial_state(
    session_id: str,
    working_directory: str = ".",
    *,
    max_iterations: int
) -> EnhancedAgentState:
    """Create initial enhanced state with sensible defaults"""
    return EnhancedAgentState(
        messages=[],
        current_task=None,
        task_history=[],
        working_directory=working_directory,
        open_files={},
        file_changes=[],
        staged_changes=[],
        active_processes={},
        environment_vars={},
        error_history=[],
        recent_errors=[],
        warning_count=0,
        tool_call_history=[],
        tool_results_cache={},
        tools_used_count={},
        pending_user_input=None,
        user_feedback=[],
        pending_approvals=[],
        session_id=session_id,
        session_start=datetime.now(),
        iteration_count=0,
        max_iterations=max_iterations,
        total_tokens_used=0,
        estimated_cost=0.0,
        plan_created=False,
        plan_steps=[],
        current_plan_step=0,
        planning_notes=[],
        verification_notes=[],
        needs_additional_iteration=False,
        required_files=[],
        plan_execution_pending=False,
        last_approval_status=None,
        force_termination=False,
        consecutive_llm_errors=0,
        scratchpad_entries=[],
        conversation_summary=None,
        important_context=[],
        # Agent mode
        current_mode=AgentMode.AUTO,
        mode_locked=False,
        mode_history=[],
        mode_iteration_count=0,
        # Agent behavior
        current_strategy=AgentStrategy.EXPLORE,
        confidence_level=0.8,
        needs_clarification=False,
        use_enhanced_state=True,
        triage_decision=None,
        triage_reason=None,
        query_complexity=None,
        expected_tool_count=5,
        _task_complete_pending_response=False,
        _patch_review_done=False,
        # Reflection state
        reflection_count=0,
        reflection_decision=None,
        reflection_notes=[],
        failure_patterns=[],
        strategy_adjustment_history=[],
        last_reflection_insight=None,
        # Execution tracking
        missing_tasks=[],
        _skip_compaction=False,
        exploration_performed=False,
        exploration_skipped=False,
        exploration_context="",
        exploration_results={},
        exploration_hints=[],
        explicit_files=[],
        project_type=None,
        task_complexity=None,
        repetition_detected=False,
        repetition_info=None,
        stagnation_detected=False,
        consecutive_denials=0,
        task_type_detected=None,
        tests_requested=False,
        lint_requested=False,
        debate_decision=None,
        _last_compaction_iter=0,
        _initial_message_count=0,
        _execution_boundary_msg_id=None,
        _execution_deadline_monotonic=None,
        # Backtracking state
        backtrack_decision=None,
        backtrack_performed=False,
        backtrack_reason=None,
        files_restored=[],
        active_patterns=[],
        # File tracking
        modified_files=[],
        file_changes_count=0,
    )


def state_to_summary(state: EnhancedAgentState) -> dict[str, Any]:
    """Convert state to a summary dictionary for logging/display"""
    return {
        "session_id": state.get("session_id", "unknown"),
        "iteration": state.get("iteration_count", 0),
        "current_task": state.get("current_task").to_dict() if state.get("current_task") else None,
        "file_changes_count": len(state.get("file_changes", [])),
        "active_processes_count": len(state.get("active_processes", {})),
        "error_count": len(state.get("error_history", [])),
        "tool_calls_count": len(state.get("tool_call_history", [])),
        "total_tokens": state.get("total_tokens_used", 0),
        "estimated_cost": state.get("estimated_cost", 0.0),
        "current_mode": state.get("current_mode", AgentMode.AUTO).value,
        "mode_locked": state.get("mode_locked", False),
        "current_strategy": state.get("current_strategy", AgentStrategy.EXPLORE).value,
        "confidence": state.get("confidence_level", 0.0),
        "reflection_count": state.get("reflection_count", 0),
        "reflection_decision": state.get("reflection_decision")
    }


def print_session_summary(state: EnhancedAgentState, console=None):
    """Print enhanced session summary with Rich formatting

    This is designed for stream output (GitHub Actions compatible):
    - No live updates or screen clearing
    - Static output with ANSI colors
    - Sequential printing to stdout

    Args:
        state: Enhanced agent state
        console: Rich console instance (optional)
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    if console is None:
        console = Console()

    summary = state_to_summary(state)

    # Main summary panel
    console.print()
    console.print(Panel.fit(
        "[bold cyan]📊 Session Summary[/bold cyan]",
        border_style="cyan"
    ))

    # Basic statistics table
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", style="cyan")

    stats_table.add_row("Session ID", summary['session_id'])
    stats_table.add_row("Iterations", str(summary['iteration']))
    stats_table.add_row("File Changes", str(summary['file_changes_count']))
    stats_table.add_row("Tool Calls", str(summary['tool_calls_count']))
    stats_table.add_row("Errors", str(summary['error_count']))
    stats_table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
    stats_table.add_row("Estimated Cost", f"${summary['estimated_cost']:.4f}")
    stats_table.add_row("Strategy", summary['current_strategy'])

    console.print(stats_table)

    # File changes details
    file_changes = state.get("file_changes", [])
    if file_changes:
        console.print()
        console.print("[bold yellow]📁 File Changes:[/bold yellow]")

        for i, fc in enumerate(file_changes, 1):
            action_color = {
                "create": "green",
                "modify": "yellow",
                "delete": "red",
                "read": "blue"
            }.get(fc.action.value, "white")

            console.print(f"  {i}. [{action_color}]{fc.action.value.upper()}[/{action_color}]: {fc.file_path}")
            console.print(f"     Tool: {fc.tool_used or 'unknown'}")

            if fc.lines_added or fc.lines_removed:
                console.print(f"     [green]+{fc.lines_added}[/green] [red]-{fc.lines_removed}[/red] lines")

            # Show diff preview (first 5 lines)
            if fc.diff:
                diff_lines = fc.diff.split('\n')[:5]
                for line in diff_lines:
                    if line.startswith('+'):
                        console.print(f"     [green]{line[:80]}[/green]")
                    elif line.startswith('-'):
                        console.print(f"     [red]{line[:80]}[/red]")
                    elif line.startswith('@@'):
                        console.print(f"     [blue]{line[:80]}[/blue]")
                if len(fc.diff.split('\n')) > 5:
                    console.print("     [dim]...[/dim]")

    # Tool call statistics
    tool_calls = state.get("tool_call_history", [])
    if tool_calls:
        console.print()
        console.print("[bold magenta]🔧 Tool Call Statistics:[/bold magenta]")

        # Group by tool name
        tool_stats = {}
        for tc in tool_calls:
            name = tc.tool_name
            if name not in tool_stats:
                tool_stats[name] = {"count": 0, "success": 0, "duration": 0.0}
            tool_stats[name]["count"] += 1
            if tc.success:
                tool_stats[name]["success"] += 1
            tool_stats[name]["duration"] += tc.duration

        # Create table
        tools_table = Table(show_header=True, box=None, padding=(0, 1))
        tools_table.add_column("Tool", style="bold")
        tools_table.add_column("Calls", justify="right")
        tools_table.add_column("Success", justify="right")
        tools_table.add_column("Duration", justify="right")

        for tool_name, stats in sorted(tool_stats.items()):
            success_rate = (stats["success"] / stats["count"] * 100) if stats["count"] > 0 else 0
            success_color = "green" if success_rate == 100 else "yellow" if success_rate >= 80 else "red"

            tools_table.add_row(
                tool_name,
                str(stats["count"]),
                f"[{success_color}]{stats['success']}/{stats['count']} ({success_rate:.0f}%)[/{success_color}]",
                f"{stats['duration']:.2f}s"
            )

        console.print(tools_table)

    # Errors summary
    errors = state.get("error_history", [])
    if errors:
        console.print()
        console.print("[bold red]❌ Errors:[/bold red]")

        for i, error in enumerate(errors[-5:], 1):  # Show last 5 errors
            resolved = error.get("resolved", False) if isinstance(error, dict) else getattr(error, "resolved", False)
            level = error.get("level", "error") if isinstance(error, dict) else getattr(error, "level", "error")
            message = error.get("message", "") if isinstance(error, dict) else getattr(error, "message", "")
            resolution = error.get("resolution") if isinstance(error, dict) else getattr(error, "resolution", None)

            status = "[green]Resolved[/green]" if resolved else "[red]Unresolved[/red]"
            level_str = level.value if hasattr(level, 'value') else str(level)
            console.print(f"  {i}. [{level_str}] {str(message)[:100]}")
            console.print(f"     Status: {status}")
            if resolution:
                console.print(f"     Resolution: {str(resolution)[:80]}")

    console.print()
