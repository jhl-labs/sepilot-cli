"""Tool execution tracking for Enhanced State

This module provides helper functions that allow tools to integrate with
Enhanced State tracking without direct state access.

Usage:
    # In tool implementations:
    from sepilot.agent.tool_tracker import record_file_change_if_enabled

    record_file_change_if_enabled(
        file_path="main.py",
        action=FileAction.MODIFY,
        old_content=old,
        new_content=new,
        tool_used="file_edit"
    )
"""

import threading
from typing import Any

from sepilot.agent import state_helpers
from sepilot.agent.enhanced_state import FileAction

# Thread-local storage for current state
_thread_local = threading.local()
_global_lock = threading.Lock()
_global_state: dict[str, Any] | None = None
_global_enabled: bool = False


def queue_delta(delta: dict[str, Any] | None) -> None:
    """Store state delta generated during tool execution.

    Public API — used by tool implementations (e.g. think_tools) to push
    state updates that will be flushed by the tool executor after execution.
    """
    if not delta:
        return
    pending = getattr(_thread_local, "pending_deltas", [])
    pending.append(delta)
    _thread_local.pending_deltas = pending


def flush_pending_deltas() -> list[dict[str, Any]]:
    """Return and clear accumulated state deltas."""
    pending = getattr(_thread_local, "pending_deltas", [])
    _thread_local.pending_deltas = []
    return pending


def set_current_state(state: dict[str, Any] | None):
    """Set the current state for this thread

    Called by Agent before tool execution.

    Args:
        state: Current EnhancedAgentState or None
    """
    global _global_state, _global_enabled

    _thread_local.state = state
    _thread_local.pending_deltas = []

    enabled = False
    if state is not None:
        try:
            getter = state.get  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback: treat unknown mapping types as enhanced to avoid losing tracking
            enabled = True
        else:
            # Default to True so Enhanced State stays active even if the flag is missing
            enabled = bool(getter("use_enhanced_state", True))

    _thread_local.enhanced_enabled = enabled
    with _global_lock:
        _global_state = state
        _global_enabled = enabled


def get_current_state() -> dict[str, Any] | None:
    """Get the current state for this thread

    Returns:
        Current state or None if not set
    """
    state = getattr(_thread_local, 'state', None)
    if state is not None:
        return state
    with _global_lock:
        return _global_state


def clear_current_state():
    """Clear the current state for this thread

    Called by Agent after tool execution.
    """
    global _global_state, _global_enabled

    _thread_local.state = None
    _thread_local.pending_deltas = []
    _thread_local.enhanced_enabled = False
    with _global_lock:
        _global_state = None
        _global_enabled = False


def is_enhanced_state_enabled() -> bool:
    """Check if Enhanced State tracking is enabled

    Returns:
        True if Enhanced State is enabled, False otherwise
    """
    enabled_flag = getattr(_thread_local, "enhanced_enabled", None)
    if enabled_flag is not None:
        return bool(enabled_flag)
    with _global_lock:
        if _global_state is not None:
            return bool(_global_enabled)

    state = get_current_state()
    if state is None:
        return False

    try:
        return bool(state.get('use_enhanced_state', True))  # type: ignore[attr-defined]
    except AttributeError:
        # Fall back to True so Enhanced State capable agents don't silently disable tracking
        return True


# ============================================================================
# File Change Tracking
# ============================================================================

def record_file_change_if_enabled(
    file_path: str,
    action: FileAction,
    old_content: str | None = None,
    new_content: str | None = None,
    tool_used: str | None = None
) -> bool:
    """Record a file change if Enhanced State is enabled

    Args:
        file_path: Path to the file
        action: Type of file action (CREATE, MODIFY, DELETE, READ)
        old_content: Previous file content (for MODIFY/DELETE)
        new_content: New file content (for CREATE/MODIFY)
        tool_used: Name of tool that made the change

    Returns:
        True if change was recorded, False otherwise
    """
    if not is_enhanced_state_enabled():
        return False

    try:
        state = get_current_state()
        if state is None:
            return False
        delta = state_helpers.record_file_change(
            state,
            file_path=file_path,
            action=action,
            old_content=old_content,
            new_content=new_content,
            tool_used=tool_used,
            return_delta=True
        )
        queue_delta(delta)
        return True
    except Exception:
        # Silently fail to avoid breaking tool execution
        return False


# ============================================================================
# Process Tracking
# ============================================================================

def start_process_if_enabled(command: str, process_id: str) -> bool:
    """Record process start if Enhanced State is enabled

    Args:
        command: Command being executed
        process_id: Unique process identifier

    Returns:
        True if process start was recorded, False otherwise
    """
    if not is_enhanced_state_enabled():
        return False

    try:
        state = get_current_state()
        delta = state_helpers.start_process(state, command, process_id, return_delta=True)
        queue_delta(delta)
        return True
    except Exception:
        return False


def complete_process_if_enabled(
    process_id: str,
    exit_code: int,
    output: str = "",
    error_output: str = ""
) -> bool:
    """Record process completion if Enhanced State is enabled

    Args:
        process_id: Process identifier
        exit_code: Process exit code
        output: Standard output
        error_output: Error output

    Returns:
        True if process completion was recorded, False otherwise
    """
    if not is_enhanced_state_enabled():
        return False

    try:
        state = get_current_state()
        delta = state_helpers.complete_process(
            state,
            process_id=process_id,
            exit_code=exit_code,
            output=output,
            error_output=error_output,
            return_delta=True
        )
        queue_delta(delta)
        return True
    except Exception:
        return False


# ============================================================================
# Tool Call Tracking
# ============================================================================

def record_tool_call_if_enabled(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    duration: float,
    success: bool,
    error: str | None = None,
    tokens_used: int = 0,
    cached: bool = False
) -> bool:
    """Record a tool call if Enhanced State is enabled

    Args:
        tool_name: Name of the tool
        args: Tool arguments
        result: Tool result
        duration: Execution time in seconds
        success: Whether the call succeeded
        error: Error message if failed
        tokens_used: Tokens consumed
        cached: Whether result was cached

    Returns:
        True if tool call was recorded, False otherwise
    """
    if not is_enhanced_state_enabled():
        return False

    try:
        state = get_current_state()
        delta = state_helpers.record_tool_call(
            state,
            tool_name=tool_name,
            args=args,
            result=result,
            duration=duration,
            success=success,
            error=error,
            tokens_used=tokens_used,
            cached=cached,
            return_delta=True
        )
        queue_delta(delta)
        return True
    except Exception:
        return False


# ============================================================================
# Error Tracking
# ============================================================================

def record_error_if_enabled(
    message: str,
    level: str = "error",
    source: str = "tool",
    context: dict[str, Any] | None = None,
    stack_trace: str | None = None
) -> bool:
    """Record an error if Enhanced State is enabled

    Args:
        message: Error message
        level: Error level (info, warning, error, critical)
        source: Source of error (tool, llm, system, user)
        context: Additional context
        stack_trace: Optional stack trace

    Returns:
        True if error was recorded, False otherwise
    """
    if not is_enhanced_state_enabled():
        return False

    try:
        from sepilot.agent.enhanced_state import ErrorLevel

        # Convert string to ErrorLevel
        level_map = {
            "info": ErrorLevel.INFO,
            "warning": ErrorLevel.WARNING,
            "error": ErrorLevel.ERROR,
            "critical": ErrorLevel.CRITICAL
        }
        error_level = level_map.get(level.lower(), ErrorLevel.ERROR)

        state = get_current_state()
        delta = state_helpers.record_error(
            state,
            message=message,
            level=error_level,
            source=source,
            context=context or {},
            stack_trace=stack_trace,
            return_delta=True
        )
        queue_delta(delta)
        return True
    except Exception:
        return False


# ============================================================================
# Caching
# ============================================================================

def cache_tool_result_if_enabled(cache_key: str, result: Any) -> bool:
    """Cache a tool result if Enhanced State is enabled

    Args:
        cache_key: Key for caching
        result: Result to cache

    Returns:
        True if result was cached, False otherwise
    """
    if not is_enhanced_state_enabled():
        return False

    try:
        state = get_current_state()
        delta = state_helpers.cache_tool_result(state, cache_key, result, return_delta=True)
        queue_delta(delta)
        return True
    except Exception:
        return False


def get_cached_result_if_enabled(cache_key: str) -> Any | None:
    """Get a cached tool result if Enhanced State is enabled

    Args:
        cache_key: Key for caching

    Returns:
        Cached result or None
    """
    if not is_enhanced_state_enabled():
        return None

    try:
        state = get_current_state()
        return state_helpers.get_cached_result(state, cache_key)
    except Exception:
        return None
