"""Agent module for SE Pilot

This module provides Claude Code-style task management and execution:
- TaskRegistry: Centralized parallel task state management
- TaskExecutor: Parallel execution with cancel/retry support
- TaskManager: Unified todo, state, and execution integration
- TaskProgressUI: Rich Live-based real-time status display
"""

from sepilot.agent.task_executor import (
    ExecutionConfig,
    ExecutionResult,
    RetryableTask,
    RetryPolicy,
    TaskExecutor,
    execute_with_progress,
)
from sepilot.agent.task_manager import (
    TaskManager,
    TodoItem,
    TodoStatus,
    get_task_manager,
    reset_task_manager,
)
from sepilot.agent.task_registry import (
    TaskInfo,
    TaskPriority,
    TaskRegistry,
    TaskState,
    get_task_registry,
    reset_task_registry,
)
from sepilot.agent.task_ui import (
    ParallelTaskUI,
    SimpleProgressBar,
    TaskProgressUI,
    print_task_summary,
    task_progress_context,
)

__all__ = [
    # Task Registry
    "TaskRegistry",
    "TaskInfo",
    "TaskState",
    "TaskPriority",
    "get_task_registry",
    "reset_task_registry",
    # Task Executor
    "TaskExecutor",
    "ExecutionConfig",
    "ExecutionResult",
    "RetryPolicy",
    "RetryableTask",
    "execute_with_progress",
    # Task Manager
    "TaskManager",
    "TodoItem",
    "TodoStatus",
    "get_task_manager",
    "reset_task_manager",
    # Task UI
    "TaskProgressUI",
    "ParallelTaskUI",
    "SimpleProgressBar",
    "task_progress_context",
    "print_task_summary",
]
