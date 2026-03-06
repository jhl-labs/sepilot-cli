"""Helper functions for updating EnhancedAgentState

These utility functions make it easy to update the enhanced state
without directly manipulating the state dictionary.
"""

import difflib
import uuid
from datetime import datetime
from typing import Any

from .enhanced_state import (
    EnhancedAgentState,
    ErrorLevel,
    ErrorRecord,
    FileAction,
    FileChange,
    ProcessInfo,
    ProcessStatus,
    TaskContext,
    TaskStatus,
    ToolCallRecord,
)

# ============================================================================
# Task Management
# ============================================================================

def start_task(
    state: EnhancedAgentState,
    description: str,
    task_id: str | None = None,
    parent_task_id: str | None = None
) -> EnhancedAgentState:
    """Start a new task

    Args:
        state: Current agent state
        description: Task description
        task_id: Optional custom task ID
        parent_task_id: Optional parent task ID for subtasks

    Returns:
        Updated state
    """
    # Complete previous task if exists
    if state.get("current_task") and state["current_task"].status == TaskStatus.IN_PROGRESS:
        complete_task(state, TaskStatus.PAUSED)

    task = TaskContext(
        task_id=task_id or f"task_{uuid.uuid4().hex[:8]}",
        description=description,
        status=TaskStatus.IN_PROGRESS,
        started_at=datetime.now(),
        parent_task_id=parent_task_id
    )

    state["current_task"] = task
    return state


def complete_task(
    state: EnhancedAgentState,
    status: TaskStatus = TaskStatus.COMPLETED,
    metadata: dict[str, Any] | None = None
) -> EnhancedAgentState:
    """Complete the current task

    Args:
        state: Current agent state
        status: Final status (COMPLETED, FAILED, PAUSED)
        metadata: Optional metadata to attach

    Returns:
        Updated state
    """
    if not state.get("current_task"):
        return state

    task = state["current_task"]
    task.status = status
    task.completed_at = datetime.now()

    if metadata:
        task.metadata.update(metadata)

    # Add to history
    if "task_history" not in state:
        state["task_history"] = []
    state["task_history"].append(task)

    state["current_task"] = None
    return state


def add_subtask(
    state: EnhancedAgentState,
    subtask_description: str
) -> EnhancedAgentState:
    """Add a subtask to the current task

    Args:
        state: Current agent state
        subtask_description: Description of the subtask

    Returns:
        Updated state
    """
    if not state.get("current_task"):
        return state

    state["current_task"].subtasks.append(subtask_description)
    return state


# ============================================================================
# File Change Tracking
# ============================================================================

def record_file_change(
    state: EnhancedAgentState,
    file_path: str,
    action: FileAction,
    old_content: str | None = None,
    new_content: str | None = None,
    tool_used: str | None = None,
    return_delta: bool = False
) -> EnhancedAgentState | dict[str, Any]:
    """Record a file modification

    Args:
        state: Current agent state
        file_path: Path to the file
        action: Type of action (CREATE, MODIFY, DELETE)
        old_content: Previous file content (for MODIFY/DELETE)
        new_content: New file content (for CREATE/MODIFY)
        tool_used: Name of tool that made the change

    Returns:
        Updated state
    """
    # Calculate diff if both old and new content exist
    diff = None
    lines_added = 0
    lines_removed = 0

    if old_content and new_content:
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}"
        ))
        diff = ''.join(diff_lines)

        # Count lines
        for line in diff_lines:
            if line.startswith('+') and not line.startswith('+++'):
                lines_added += 1
            elif line.startswith('-') and not line.startswith('---'):
                lines_removed += 1

    change = FileChange(
        file_path=file_path,
        action=action,
        old_content=old_content,
        new_content=new_content,
        timestamp=datetime.now(),
        tool_used=tool_used,
        diff=diff,
        lines_added=lines_added,
        lines_removed=lines_removed
    )

    if return_delta:
        # Delta mode: do NOT mutate state; let LangGraph reducers handle it
        return {"file_changes": [change]}

    # Direct mutation mode (non-delta)
    if "file_changes" not in state:
        state["file_changes"] = []

    state["file_changes"] = add_file_change(state["file_changes"], change)
    return state


def stage_change(
    state: EnhancedAgentState,
    file_path: str
) -> EnhancedAgentState:
    """Stage a file change for commit (Git-like)

    Args:
        state: Current agent state
        file_path: Path to the file to stage

    Returns:
        Updated state
    """
    # Find the most recent change to this file
    for change in reversed(state.get("file_changes", [])):
        if change.file_path == file_path:
            if "staged_changes" not in state:
                state["staged_changes"] = []

            # Remove from staged if already there
            state["staged_changes"] = [
                c for c in state["staged_changes"]
                if c.file_path != file_path
            ]

            # Add to staged
            state["staged_changes"].append(change)
            break

    return state


# ============================================================================
# Process Management
# ============================================================================

def start_process(
    state: EnhancedAgentState,
    command: str,
    process_id: str | None = None,
    return_delta: bool = False
) -> EnhancedAgentState | dict[str, Any]:
    """Start tracking a process

    Args:
        state: Current agent state
        command: Command being executed
        process_id: Optional custom process ID

    Returns:
        Updated state
    """
    pid = process_id or f"proc_{uuid.uuid4().hex[:8]}"

    process = ProcessInfo(
        process_id=pid,
        command=command,
        started_at=datetime.now(),
        status=ProcessStatus.RUNNING
    )

    if return_delta:
        # Delta mode: do NOT mutate state
        current = state.get("active_processes", {}).copy()
        current[pid] = process
        return {"active_processes": current}

    if "active_processes" not in state:
        state["active_processes"] = {}
    state["active_processes"][pid] = process
    return state


def complete_process(
    state: EnhancedAgentState,
    process_id: str,
    exit_code: int,
    output: str = "",
    error_output: str = "",
    return_delta: bool = False
) -> EnhancedAgentState | dict[str, Any]:
    """Complete a process

    Args:
        state: Current agent state
        process_id: Process ID
        exit_code: Process exit code
        output: Standard output
        error_output: Error output

    Returns:
        Updated state
    """
    if "active_processes" not in state or process_id not in state["active_processes"]:
        if return_delta:
            return {}
        return state

    if return_delta:
        # Delta mode: do NOT mutate state; deep copy ProcessInfo to avoid side effects
        from copy import copy
        current = {k: copy(v) for k, v in state.get("active_processes", {}).items()}
        proc = current.pop(process_id, None)
        if proc:
            proc.status = ProcessStatus.COMPLETED if exit_code == 0 else ProcessStatus.FAILED
            proc.exit_code = exit_code
            proc.output = output
            proc.error_output = error_output
            proc.completed_at = datetime.now()
        return {"active_processes": current}

    process = state["active_processes"][process_id]
    process.status = ProcessStatus.COMPLETED if exit_code == 0 else ProcessStatus.FAILED
    process.exit_code = exit_code
    process.output = output
    process.error_output = error_output
    process.completed_at = datetime.now()
    del state["active_processes"][process_id]
    return state


# ============================================================================
# Error Management
# ============================================================================

def record_error(
    state: EnhancedAgentState,
    message: str,
    level: ErrorLevel = ErrorLevel.ERROR,
    source: str = "system",
    context: dict[str, Any] | None = None,
    stack_trace: str | None = None,
    return_delta: bool = False
) -> EnhancedAgentState | dict[str, Any]:
    """Record an error or warning

    Args:
        state: Current agent state
        message: Error message
        level: Error level (INFO, WARNING, ERROR, CRITICAL)
        source: Source of error (tool, llm, system, user)
        context: Additional context
        stack_trace: Optional stack trace

    Returns:
        Updated state
    """
    error = ErrorRecord(
        timestamp=datetime.now(),
        level=level,
        message=message,
        source=source,
        context=context or {},
        stack_trace=stack_trace
    )

    if return_delta:
        # Delta mode: do NOT mutate state; let LangGraph reducers handle it
        # NOTE: recent_errors and warning_count are NOT included in delta.
        # - recent_errors: derived from error_history after reducer runs.
        #   Including it causes merge_updates to concatenate stale snapshots
        #   when multiple errors are recorded in the same node.
        # - warning_count: derived from error_history at read time.
        #   Including it as absolute value causes undercounting when multiple
        #   warnings are recorded in the same node (same snapshot + 1).
        return {"error_history": [error]}

    # Direct mutation mode (non-delta)
    if "error_history" not in state:
        state["error_history"] = []

    state["error_history"] = add_error_record(state["error_history"], error)
    state["recent_errors"] = state["error_history"][-10:]

    if level == ErrorLevel.WARNING:
        state["warning_count"] = state.get("warning_count", 0) + 1

    return state


def resolve_error(
    state: EnhancedAgentState,
    error_index: int,
    resolution: str
) -> EnhancedAgentState:
    """Mark an error as resolved

    Args:
        state: Current agent state
        error_index: Index in error_history
        resolution: Description of how it was resolved

    Returns:
        Updated state
    """
    if "error_history" not in state or error_index >= len(state["error_history"]):
        return state

    target = state["error_history"][error_index]
    if isinstance(target, dict):
        target["resolved"] = True
        target["resolution"] = resolution
    else:
        target.resolved = True
        target.resolution = resolution

    return state


# ============================================================================
# Tool Usage Tracking
# ============================================================================

def record_tool_call(
    state: EnhancedAgentState,
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    duration: float,
    success: bool,
    error: str | None = None,
    tokens_used: int = 0,
    cached: bool = False,
    return_delta: bool = False
) -> EnhancedAgentState | dict[str, Any]:
    """Record a tool invocation

    Args:
        state: Current agent state
        tool_name: Name of the tool
        args: Tool arguments
        result: Tool result
        duration: Execution time in seconds
        success: Whether the call succeeded
        error: Error message if failed
        tokens_used: Tokens consumed
        cached: Whether result was cached

    Returns:
        Updated state
    """
    record = ToolCallRecord(
        tool_name=tool_name,
        timestamp=datetime.now(),
        args=args,
        result=result,
        duration=duration,
        success=success,
        error=error,
        tokens_used=tokens_used,
        cached=cached
    )

    if return_delta:
        # Delta mode: do NOT mutate state; let LangGraph reducers handle it
        current_counts = state.get("tools_used_count", {}).copy()
        current_counts[tool_name] = current_counts.get(tool_name, 0) + 1
        return {
            "tool_call_history": [record],
            "tools_used_count": current_counts,
            "total_tokens_used": state.get("total_tokens_used", 0) + tokens_used
        }

    # Direct mutation mode (non-delta)
    if "tool_call_history" not in state:
        state["tool_call_history"] = []

    state["tool_call_history"] = add_tool_call(state["tool_call_history"], record)

    if "tools_used_count" not in state:
        state["tools_used_count"] = {}

    state["tools_used_count"][tool_name] = state["tools_used_count"].get(tool_name, 0) + 1
    state["total_tokens_used"] = state.get("total_tokens_used", 0) + tokens_used

    return state


def cache_tool_result(
    state: EnhancedAgentState,
    cache_key: str,
    result: Any,
    ttl: int | None = None,
    return_delta: bool = False
) -> EnhancedAgentState | dict[str, Any]:
    """Cache a tool result

    Args:
        state: Current agent state
        cache_key: Key for caching
        result: Result to cache
        ttl: Time to live in seconds (not implemented yet)

    Returns:
        Updated state
    """
    if return_delta:
        # Delta mode: do NOT mutate state
        current = state.get("tool_results_cache", {}).copy()
        current[cache_key] = result
        return {"tool_results_cache": current}

    if "tool_results_cache" not in state:
        state["tool_results_cache"] = {}
    state["tool_results_cache"][cache_key] = result
    return state


def get_cached_result(
    state: EnhancedAgentState,
    cache_key: str
) -> Any | None:
    """Get a cached tool result

    Args:
        state: Current agent state
        cache_key: Key for caching

    Returns:
        Cached result or None
    """
    return state.get("tool_results_cache", {}).get(cache_key)


# ============================================================================
# Session Management
# ============================================================================

def increment_iteration(
    state: EnhancedAgentState,
    return_delta: bool = False
) -> EnhancedAgentState | dict[str, Any]:
    """Increment the iteration count

    Args:
        state: Current agent state

    Returns:
        Updated state
    """
    new_count = state.get("iteration_count", 0) + 1

    if return_delta:
        return {"iteration_count": new_count}
    state["iteration_count"] = new_count
    return state


def update_cost(
    state: EnhancedAgentState,
    tokens_used: int,
    cost_per_1k_tokens: float = 0.03
) -> EnhancedAgentState:
    """Update estimated cost

    Args:
        state: Current agent state
        tokens_used: Tokens used in this call
        cost_per_1k_tokens: Cost per 1000 tokens

    Returns:
        Updated state
    """
    state["total_tokens_used"] = state.get("total_tokens_used", 0) + tokens_used
    state["estimated_cost"] = (state["total_tokens_used"] / 1000) * cost_per_1k_tokens
    return state


def set_strategy(
    state: EnhancedAgentState,
    strategy: str,
    confidence: float = 0.8
) -> EnhancedAgentState:
    """Set the agent's current strategy

    Args:
        state: Current agent state
        strategy: Strategy name (explore, implement, debug, etc.)
        confidence: Confidence level (0.0 - 1.0)

    Returns:
        Updated state
    """
    from .enhanced_state import AgentStrategy

    # Convert string to enum
    try:
        strategy_enum = AgentStrategy(strategy.lower())
        state["current_strategy"] = strategy_enum
    except ValueError:
        state["current_strategy"] = AgentStrategy.EXPLORE

    state["confidence_level"] = max(0.0, min(1.0, confidence))
    return state


# Helper function from enhanced_state for use here
def add_file_change(existing, new):
    """Import from enhanced_state"""
    from .enhanced_state import add_file_change as _add_file_change
    return _add_file_change(existing, new)


def add_error_record(existing, new):
    """Import from enhanced_state"""
    from .enhanced_state import add_error_record as _add_error_record
    return _add_error_record(existing, new)


def add_tool_call(existing, new):
    """Import from enhanced_state"""
    from .enhanced_state import add_tool_call as _add_tool_call
    return _add_tool_call(existing, new)


# ============================================================================
# Task Manager Integration (Claude Code style)
# ============================================================================

def sync_task_manager_to_state(
    state: EnhancedAgentState,
    task_manager: "TaskManager"
) -> EnhancedAgentState:
    """Sync TaskManager state to EnhancedAgentState.

    This enables Claude Code-style unified task tracking.

    Args:
        state: Current agent state
        task_manager: TaskManager instance

    Returns:
        Updated state with synced task information
    """
    from sepilot.agent.task_manager import TodoStatus

    # Sync current todo as current_task
    current_todo = task_manager.get_current_todo()
    if current_todo:
        state["current_task"] = TaskContext(
            task_id=current_todo.id,
            description=current_todo.content,
            status=_map_todo_to_task_status(current_todo.status),
            started_at=current_todo.started_at or datetime.now(),
            metadata={"active_form": current_todo.active_form}
        )
    else:
        state["current_task"] = None

    # Sync todos as plan_steps
    root_todos = task_manager.get_root_todos()
    state["plan_steps"] = [todo.content for todo in root_todos]

    # Calculate current step index
    for i, todo in enumerate(root_todos):
        if todo.status == TodoStatus.IN_PROGRESS:
            state["current_plan_step"] = i
            break
        elif todo.status == TodoStatus.PENDING:
            state["current_plan_step"] = max(0, i - 1)
            break
    else:
        state["current_plan_step"] = len(root_todos) - 1 if root_todos else 0

    state["plan_created"] = len(root_todos) > 0

    return state


def sync_state_to_task_manager(
    state: EnhancedAgentState,
    task_manager: "TaskManager"
) -> None:
    """Sync EnhancedAgentState to TaskManager.

    Args:
        state: Current agent state
        task_manager: TaskManager instance to update
    """

    # Get plan_steps from state and create todos if needed
    plan_steps = state.get("plan_steps", [])
    if not plan_steps:
        return

    existing_contents = {t.content for t in task_manager.get_all_todos()}

    for step in plan_steps:
        if step not in existing_contents:
            task_manager.add_todo(step)


def _map_todo_to_task_status(todo_status) -> TaskStatus:
    """Map TodoStatus to TaskStatus."""
    from sepilot.agent.task_manager import TodoStatus

    mapping = {
        TodoStatus.PENDING: TaskStatus.PENDING,
        TodoStatus.IN_PROGRESS: TaskStatus.IN_PROGRESS,
        TodoStatus.COMPLETED: TaskStatus.COMPLETED,
        TodoStatus.FAILED: TaskStatus.FAILED,
        TodoStatus.SKIPPED: TaskStatus.PAUSED,
    }
    return mapping.get(todo_status, TaskStatus.PENDING)


def create_state_sync_callback(state: EnhancedAgentState) -> callable:
    """Create a callback function for TaskManager state sync.

    Args:
        state: State to sync to

    Returns:
        Callback function for TaskManager
    """
    def callback(update: dict[str, Any]) -> None:
        if update.get("current_task"):
            task_data = update["current_task"]
            if task_data.get("task_id"):
                state["current_task"] = TaskContext(
                    task_id=task_data["task_id"],
                    description=task_data.get("description", ""),
                    status=TaskStatus(task_data.get("status", "pending"))
                )
            else:
                state["current_task"] = None

        if "plan_steps" in update:
            state["plan_steps"] = update["plan_steps"]

        if "current_plan_step" in update:
            state["current_plan_step"] = update["current_plan_step"]

    return callback


# ============================================================================
# Task Registry Integration
# ============================================================================

def sync_task_registry_to_state(
    state: EnhancedAgentState,
    registry: "TaskRegistry"
) -> EnhancedAgentState:
    """Sync TaskRegistry to EnhancedAgentState.

    Args:
        state: Current agent state
        registry: TaskRegistry instance

    Returns:
        Updated state
    """

    # Get registry stats
    stats = registry.get_stats()

    # Store in metadata
    if "parallel_task_stats" not in state:
        state["parallel_task_stats"] = {}

    state["parallel_task_stats"] = {
        "total": stats["total"],
        "running": stats["running"],
        "completed": stats["completed"],
        "failed": stats["failed"],
        "success_rate": stats["success_rate"]
    }

    # Get running tasks for context
    running = registry.get_running_tasks()
    if running:
        state["active_parallel_tasks"] = [
            {"task_id": t.task_id, "name": t.name, "progress": t.progress}
            for t in running
        ]
    else:
        state["active_parallel_tasks"] = []

    return state


def get_parallel_task_context(state: EnhancedAgentState) -> str:
    """Get formatted parallel task context for prompts.

    Args:
        state: Agent state

    Returns:
        Formatted context string
    """
    stats = state.get("parallel_task_stats", {})
    if not stats:
        return ""

    active = state.get("active_parallel_tasks", [])

    lines = ["## Parallel Task Status"]
    lines.append(f"- Total: {stats.get('total', 0)}")
    lines.append(f"- Completed: {stats.get('completed', 0)}")
    lines.append(f"- Running: {stats.get('running', 0)}")

    if active:
        lines.append("\n### Currently Running:")
        for task in active:
            progress = task.get('progress', 0)
            lines.append(f"  - {task['name']}: {progress:.0%}")

    return "\n".join(lines)


# Type hints for documentation (imported at runtime)
if False:  # TYPE_CHECKING equivalent that doesn't require import
    from sepilot.agent.task_manager import TaskManager
    from sepilot.agent.task_registry import TaskRegistry
