"""Task Registry - Centralized parallel task state management.

Provides Claude Code-style task state tracking with:
- Centralized state registry for all parallel tasks
- Real-time state updates and queries
- Task lifecycle management (create, start, complete, cancel, fail)
- State persistence support for checkpoint integration
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """Task execution states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskInfo:
    """Information about a registered task."""
    task_id: str
    name: str
    description: str
    state: TaskState = TaskState.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    parent_id: str | None = None
    dependencies: list[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Execution tracking
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0  # 5 minutes default

    # Results
    result: Any = None
    error: str | None = None
    error_traceback: str | None = None

    # Metrics
    progress: float = 0.0  # 0.0 to 1.0
    tokens_used: int = 0
    duration_seconds: float = 0.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime references (not serialized)
    _asyncio_task: asyncio.Task | None = field(default=None, repr=False)

    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.TIMEOUT
        )

    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.state in (TaskState.RUNNING, TaskState.RETRYING)

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.state == TaskState.FAILED and
            self.retry_count < self.max_retries
        )

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "priority": self.priority.value,
            "parent_id": self.parent_id,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "error": self.error,
            "progress": self.progress,
            "tokens_used": self.tokens_used,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskInfo:
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data["description"],
            state=TaskState(data["state"]),
            priority=TaskPriority(data.get("priority", 1)),
            parent_id=data.get("parent_id"),
            dependencies=data.get("dependencies", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            error=data.get("error"),
            progress=data.get("progress", 0.0),
            tokens_used=data.get("tokens_used", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            metadata=data.get("metadata", {})
        )


class TaskRegistry:
    """Centralized registry for all parallel tasks.

    Thread-safe and async-safe task state management.

    Example:
        registry = TaskRegistry()

        # Register tasks
        task1 = registry.register("Task 1", "Description 1")
        task2 = registry.register("Task 2", "Description 2", dependencies=[task1.task_id])

        # Update state
        registry.start(task1.task_id)
        registry.update_progress(task1.task_id, 0.5)
        registry.complete(task1.task_id, result="Done")

        # Query
        running = registry.get_running_tasks()
        pending = registry.get_ready_tasks()  # Dependencies met
    """

    def __init__(self):
        """Initialize the task registry."""
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.RLock()
        self._listeners: list[Callable[[TaskInfo, str], None]] = []
        self._session_id: str = str(uuid.uuid4())[:8]

    # ==================== Registration ====================

    def register(
        self,
        name: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        parent_id: str | None = None,
        dependencies: list[str] | None = None,
        max_retries: int = 3,
        timeout_seconds: float = 300.0,
        metadata: dict[str, Any] | None = None
    ) -> TaskInfo:
        """Register a new task.

        Args:
            name: Short task name
            description: Detailed description
            priority: Task priority
            parent_id: Parent task ID (for hierarchical tasks)
            dependencies: Task IDs that must complete first
            max_retries: Maximum retry attempts
            timeout_seconds: Task timeout
            metadata: Additional metadata

        Returns:
            Registered TaskInfo
        """
        task_id = f"{self._session_id}_{len(self._tasks):04d}"

        task = TaskInfo(
            task_id=task_id,
            name=name,
            description=description or name,
            priority=priority,
            parent_id=parent_id,
            dependencies=dependencies or [],
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {}
        )

        with self._lock:
            self._tasks[task_id] = task

        self._notify_listeners(task, "registered")
        logger.debug(f"Task registered: {task_id} - {name}")

        return task

    def register_batch(
        self,
        tasks: list[dict[str, Any]]
    ) -> list[TaskInfo]:
        """Register multiple tasks at once.

        Args:
            tasks: List of task definitions (name, description, etc.)

        Returns:
            List of registered TaskInfo
        """
        registered = []
        for task_def in tasks:
            task = self.register(**task_def)
            registered.append(task)
        return registered

    # ==================== State Transitions ====================

    def start(self, task_id: str) -> TaskInfo | None:
        """Mark task as started/running.

        Args:
            task_id: Task ID to start

        Returns:
            Updated TaskInfo or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            if task.state not in (TaskState.PENDING, TaskState.QUEUED, TaskState.RETRYING):
                logger.warning(f"Cannot start task {task_id} in state {task.state}")
                return task

            task.state = TaskState.RUNNING
            task.started_at = datetime.now()

        self._notify_listeners(task, "started")
        return task

    def complete(
        self,
        task_id: str,
        result: Any = None,
        tokens_used: int = 0
    ) -> TaskInfo | None:
        """Mark task as completed.

        Args:
            task_id: Task ID
            result: Task result
            tokens_used: Tokens consumed

        Returns:
            Updated TaskInfo or None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            task.state = TaskState.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            task.tokens_used = tokens_used
            task.progress = 1.0
            task.duration_seconds = task.elapsed_time()

        self._notify_listeners(task, "completed")
        return task

    def fail(
        self,
        task_id: str,
        error: str,
        traceback: str | None = None
    ) -> TaskInfo | None:
        """Mark task as failed.

        Args:
            task_id: Task ID
            error: Error message
            traceback: Optional traceback

        Returns:
            Updated TaskInfo or None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            task.state = TaskState.FAILED
            task.completed_at = datetime.now()
            task.error = error
            task.error_traceback = traceback
            task.duration_seconds = task.elapsed_time()

        self._notify_listeners(task, "failed")
        return task

    def cancel(self, task_id: str, reason: str = "") -> TaskInfo | None:
        """Cancel a task.

        Also cancels the underlying asyncio task if running.

        Args:
            task_id: Task ID to cancel
            reason: Cancellation reason

        Returns:
            Updated TaskInfo or None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            if task.is_terminal():
                return task

            # Cancel asyncio task if running
            if task._asyncio_task and not task._asyncio_task.done():
                task._asyncio_task.cancel()

            task.state = TaskState.CANCELLED
            task.completed_at = datetime.now()
            task.error = reason or "Cancelled by user"
            task.duration_seconds = task.elapsed_time()

        self._notify_listeners(task, "cancelled")
        return task

    def timeout(self, task_id: str) -> TaskInfo | None:
        """Mark task as timed out.

        Args:
            task_id: Task ID

        Returns:
            Updated TaskInfo or None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            if task._asyncio_task and not task._asyncio_task.done():
                task._asyncio_task.cancel()

            task.state = TaskState.TIMEOUT
            task.completed_at = datetime.now()
            task.error = f"Task timed out after {task.timeout_seconds}s"
            task.duration_seconds = task.elapsed_time()

        self._notify_listeners(task, "timeout")
        return task

    def begin_retry(self, task_id: str) -> TaskInfo | None:
        """Transition a running/failed task into retrying state.

        Args:
            task_id: Task ID to retry

        Returns:
            Updated TaskInfo or None if not retryable
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            if task.retry_count >= task.max_retries:
                logger.warning(f"Task {task_id} cannot be retried")
                return task

            if task.state not in (TaskState.RUNNING, TaskState.FAILED, TaskState.RETRYING):
                logger.warning(f"Task {task_id} cannot begin retry from state {task.state}")
                return task

            task.state = TaskState.RETRYING
            task.retry_count += 1
            task.completed_at = None
            task.error = None
            task.error_traceback = None
            task.progress = 0.0

        self._notify_listeners(task, "retrying")
        return task

    def retry(self, task_id: str) -> TaskInfo | None:
        """Backward-compatible alias for begin_retry()."""
        return self.begin_retry(task_id)

    def update_progress(
        self,
        task_id: str,
        progress: float,
        metadata: dict[str, Any] | None = None
    ) -> TaskInfo | None:
        """Update task progress.

        Args:
            task_id: Task ID
            progress: Progress value (0.0 to 1.0)
            metadata: Additional metadata to update

        Returns:
            Updated TaskInfo or None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            task.progress = max(0.0, min(1.0, progress))
            if metadata:
                task.metadata.update(metadata)

        self._notify_listeners(task, "progress")
        return task

    def set_asyncio_task(
        self,
        task_id: str,
        asyncio_task: asyncio.Task
    ) -> None:
        """Associate an asyncio.Task with a registered task.

        Args:
            task_id: Registered task ID
            asyncio_task: The asyncio.Task to associate
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task._asyncio_task = asyncio_task

    # ==================== Queries ====================

    def get(self, task_id: str) -> TaskInfo | None:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all(self) -> list[TaskInfo]:
        """Get all registered tasks."""
        with self._lock:
            return list(self._tasks.values())

    def get_by_state(self, state: TaskState) -> list[TaskInfo]:
        """Get tasks by state."""
        with self._lock:
            return [t for t in self._tasks.values() if t.state == state]

    def get_running_tasks(self) -> list[TaskInfo]:
        """Get all currently running tasks."""
        with self._lock:
            return [t for t in self._tasks.values() if t.is_running()]

    def get_pending_tasks(self) -> list[TaskInfo]:
        """Get all pending tasks."""
        return self.get_by_state(TaskState.PENDING)

    def get_ready_tasks(self) -> list[TaskInfo]:
        """Get pending tasks with all dependencies met."""
        with self._lock:
            ready = []
            for task in self._tasks.values():
                if task.state != TaskState.PENDING:
                    continue

                # Check dependencies
                deps_met = all(
                    self._tasks.get(dep_id) and
                    self._tasks[dep_id].state == TaskState.COMPLETED
                    for dep_id in task.dependencies
                )

                if deps_met:
                    ready.append(task)

            # Sort by priority (highest first)
            ready.sort(key=lambda t: t.priority.value, reverse=True)
            return ready

    def get_failed_tasks(self) -> list[TaskInfo]:
        """Get all failed tasks."""
        return self.get_by_state(TaskState.FAILED)

    def get_retryable_tasks(self) -> list[TaskInfo]:
        """Get failed tasks that can be retried."""
        with self._lock:
            return [t for t in self._tasks.values() if t.can_retry()]

    def get_children(self, parent_id: str) -> list[TaskInfo]:
        """Get child tasks of a parent."""
        with self._lock:
            return [t for t in self._tasks.values() if t.parent_id == parent_id]

    # ==================== Statistics ====================

    def get_stats(self) -> dict[str, Any]:
        """Get overall statistics."""
        with self._lock:
            tasks = list(self._tasks.values())

            by_state = {}
            for state in TaskState:
                by_state[state.value] = len([t for t in tasks if t.state == state])

            total = len(tasks)
            completed = by_state.get(TaskState.COMPLETED.value, 0)
            failed = by_state.get(TaskState.FAILED.value, 0)
            running = by_state.get(TaskState.RUNNING.value, 0)

            return {
                "total": total,
                "by_state": by_state,
                "completed": completed,
                "failed": failed,
                "running": running,
                "pending": by_state.get(TaskState.PENDING.value, 0),
                "success_rate": (completed / total * 100) if total > 0 else 0.0,
                "completion_rate": ((completed + failed) / total * 100) if total > 0 else 0.0,
                "total_tokens": sum(t.tokens_used for t in tasks),
                "total_duration": sum(t.duration_seconds for t in tasks if t.duration_seconds > 0)
            }

    def get_progress_summary(self) -> str:
        """Get a formatted progress summary."""
        stats = self.get_stats()

        parts = []
        parts.append(f"Tasks: {stats['completed']}/{stats['total']} completed")

        if stats['running'] > 0:
            parts.append(f"{stats['running']} running")
        if stats['failed'] > 0:
            parts.append(f"{stats['failed']} failed")

        return " | ".join(parts)

    # ==================== Listeners ====================

    def add_listener(self, callback: Callable[[TaskInfo, str], None]) -> None:
        """Add a state change listener.

        Args:
            callback: Function called with (task, event_type)
                     event_type: registered, started, completed, failed, etc.
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[TaskInfo, str], None]) -> None:
        """Remove a state change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self, task: TaskInfo, event: str) -> None:
        """Notify all listeners of a state change."""
        for listener in self._listeners:
            try:
                listener(task, event)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    # ==================== Persistence ====================

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry to dictionary."""
        with self._lock:
            return {
                "session_id": self._session_id,
                "tasks": {
                    task_id: task.to_dict()
                    for task_id, task in self._tasks.items()
                }
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRegistry:
        """Deserialize registry from dictionary."""
        registry = cls()
        registry._session_id = data.get("session_id", registry._session_id)

        for task_id, task_data in data.get("tasks", {}).items():
            task = TaskInfo.from_dict(task_data)
            registry._tasks[task_id] = task

        return registry

    def clear(self) -> None:
        """Clear all tasks."""
        with self._lock:
            self._tasks.clear()

    def clear_completed(self) -> int:
        """Clear completed tasks and return count removed."""
        with self._lock:
            to_remove = [
                task_id for task_id, task in self._tasks.items()
                if task.is_terminal()
            ]
            for task_id in to_remove:
                del self._tasks[task_id]
            return len(to_remove)


# Singleton instance
_registry: TaskRegistry | None = None


def get_task_registry() -> TaskRegistry:
    """Get the global TaskRegistry instance."""
    global _registry
    if _registry is None:
        _registry = TaskRegistry()
    return _registry


def reset_task_registry() -> None:
    """Reset the global TaskRegistry instance."""
    global _registry
    if _registry:
        _registry.clear()
    _registry = TaskRegistry()
