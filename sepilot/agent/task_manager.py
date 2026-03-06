"""Unified Task Manager - Claude Code style todo, state, and execution integration.

Provides:
- Unified todo tracking with agent state synchronization
- Automatic todo generation for complex tasks
- Real-time progress tracking
- Integration with TaskRegistry and TaskExecutor
- Checkpoint persistence support
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sepilot.agent.task_executor import ExecutionConfig, ExecutionResult, TaskExecutor
from sepilot.agent.task_registry import (
    TaskInfo,
    TaskRegistry,
    TaskState,
    get_task_registry,
)

if TYPE_CHECKING:
    from sepilot.agent.enhanced_state import EnhancedAgentState

logger = logging.getLogger(__name__)


class TodoStatus(str, Enum):
    """Todo item status aligned with Claude Code."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TodoItem:
    """A todo item in the unified task manager."""
    id: str
    content: str
    active_form: str  # Present continuous form for display
    status: TodoStatus = TodoStatus.PENDING
    task_id: str | None = None  # Link to TaskRegistry task
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    priority: int = 1  # 1-5 scale
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "active_form": self.active_form,
            "status": self.status.value,
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoItem:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            active_form=data.get("active_form", data["content"]),
            status=TodoStatus(data.get("status", "pending")),
            task_id=data.get("task_id"),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            priority=data.get("priority", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )


class TaskManager:
    """Unified task manager integrating todo, state, and execution.

    Claude Code-style features:
    - Todo items automatically sync with agent state
    - Progress is visible in real-time
    - Parallel execution is transparent
    - State persists across sessions

    Example:
        manager = TaskManager()

        # Create todos from task description
        manager.auto_create_todos("Implement user authentication with JWT")

        # Or manually create todos
        manager.add_todo("Analyze current auth implementation", "Analyzing auth")
        manager.add_todo("Design JWT token structure", "Designing JWT")
        manager.add_todo("Implement token generation", "Implementing tokens")

        # Start working on a todo
        manager.start_todo("todo_1")

        # Complete with result
        manager.complete_todo("todo_1", result=analysis_result)

        # Execute todos as parallel tasks
        async with manager:
            results = await manager.execute_parallel([
                ("todo_2", coro1),
                ("todo_3", coro2)
            ])
    """

    def __init__(
        self,
        registry: TaskRegistry | None = None,
        persist_path: str | Path | None = None,
        auto_persist: bool = True,
        state_sync_callback: Callable[[dict[str, Any]], None] | None = None
    ):
        """Initialize task manager.

        Args:
            registry: Task registry (uses global if None)
            persist_path: Path for persistence (default: ~/.sepilot/task_manager.json)
            auto_persist: Auto-save on changes
            state_sync_callback: Callback to sync with EnhancedAgentState
        """
        self.registry = registry or get_task_registry()
        self.auto_persist = auto_persist
        self.state_sync_callback = state_sync_callback

        if persist_path is None:
            persist_path = Path.home() / ".sepilot" / "task_manager.json"
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self._todos: dict[str, TodoItem] = {}
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._todo_counter: int = 0
        self._executor: TaskExecutor | None = None
        self._current_todo_id: str | None = None

        # Progress listeners
        self._progress_listeners: list[Callable[[TodoItem], None]] = []

        # Load persisted state if exists
        self._load()

        # Set up registry listener
        self.registry.add_listener(self._on_task_state_change)

    # ==================== Todo Management ====================

    def add_todo(
        self,
        content: str,
        active_form: str | None = None,
        priority: int = 1,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> TodoItem:
        """Add a new todo item.

        Args:
            content: Todo description (imperative form: "Run tests")
            active_form: Active form for display ("Running tests")
            priority: Priority 1-5
            parent_id: Parent todo ID for hierarchical todos
            metadata: Additional metadata

        Returns:
            Created TodoItem
        """
        self._todo_counter += 1
        todo_id = f"todo_{self._session_id}_{self._todo_counter:04d}"

        if active_form is None:
            # Auto-generate active form (simple heuristic)
            active_form = self._generate_active_form(content)

        todo = TodoItem(
            id=todo_id,
            content=content,
            active_form=active_form,
            priority=priority,
            parent_id=parent_id,
            metadata=metadata or {}
        )

        self._todos[todo_id] = todo

        if parent_id and parent_id in self._todos:
            self._todos[parent_id].children_ids.append(todo_id)

        self._notify_change(todo)
        self._persist_if_auto()

        logger.debug(f"Added todo: {todo_id} - {content}")
        return todo

    def add_todos(
        self,
        todos: list[dict[str, Any]]
    ) -> list[TodoItem]:
        """Add multiple todos at once.

        Args:
            todos: List of todo definitions with 'content' and 'activeForm'

        Returns:
            List of created TodoItems
        """
        created = []
        for todo_def in todos:
            item = self.add_todo(
                content=todo_def.get("content", ""),
                active_form=todo_def.get("activeForm") or todo_def.get("active_form"),
                priority=todo_def.get("priority", 1),
                parent_id=todo_def.get("parent_id"),
                metadata=todo_def.get("metadata")
            )
            created.append(item)
        return created

    def start_todo(self, todo_id: str) -> TodoItem | None:
        """Mark a todo as in progress.

        Args:
            todo_id: Todo ID to start

        Returns:
            Updated TodoItem or None
        """
        todo = self._todos.get(todo_id)
        if not todo:
            return None

        if todo.status != TodoStatus.PENDING:
            logger.warning(f"Cannot start todo {todo_id} in status {todo.status}")
            return todo

        todo.status = TodoStatus.IN_PROGRESS
        todo.started_at = datetime.now()
        self._current_todo_id = todo_id

        self._notify_change(todo)
        self._persist_if_auto()
        self._sync_to_state()

        return todo

    def complete_todo(
        self,
        todo_id: str,
        result: Any = None
    ) -> TodoItem | None:
        """Mark a todo as completed.

        Args:
            todo_id: Todo ID to complete
            result: Optional result data

        Returns:
            Updated TodoItem or None
        """
        todo = self._todos.get(todo_id)
        if not todo:
            return None

        todo.status = TodoStatus.COMPLETED
        todo.completed_at = datetime.now()
        if result is not None:
            todo.metadata["result"] = str(result)[:1000]

        if self._current_todo_id == todo_id:
            self._current_todo_id = None

        self._notify_change(todo)
        self._persist_if_auto()
        self._sync_to_state()

        # Auto-complete parent if all children done
        if todo.parent_id:
            self._check_parent_completion(todo.parent_id)

        return todo

    def fail_todo(
        self,
        todo_id: str,
        error: str
    ) -> TodoItem | None:
        """Mark a todo as failed.

        Args:
            todo_id: Todo ID to fail
            error: Error message

        Returns:
            Updated TodoItem or None
        """
        todo = self._todos.get(todo_id)
        if not todo:
            return None

        todo.status = TodoStatus.FAILED
        todo.completed_at = datetime.now()
        todo.error = error

        if self._current_todo_id == todo_id:
            self._current_todo_id = None

        self._notify_change(todo)
        self._persist_if_auto()
        self._sync_to_state()

        return todo

    def skip_todo(self, todo_id: str, reason: str = "") -> TodoItem | None:
        """Skip a todo.

        Args:
            todo_id: Todo ID to skip
            reason: Skip reason

        Returns:
            Updated TodoItem or None
        """
        todo = self._todos.get(todo_id)
        if not todo:
            return None

        todo.status = TodoStatus.SKIPPED
        todo.completed_at = datetime.now()
        if reason:
            todo.metadata["skip_reason"] = reason

        self._notify_change(todo)
        self._persist_if_auto()

        return todo

    def update_progress(
        self,
        todo_id: str,
        progress: float = 0.0,
        metadata: dict[str, Any] | None = None
    ) -> TodoItem | None:
        """Update todo progress.

        Args:
            todo_id: Todo ID
            progress: Progress 0.0 to 1.0
            metadata: Additional metadata

        Returns:
            Updated TodoItem or None
        """
        todo = self._todos.get(todo_id)
        if not todo:
            return None

        todo.metadata["progress"] = max(0.0, min(1.0, progress))
        if metadata:
            todo.metadata.update(metadata)

        self._notify_change(todo)
        return todo

    # ==================== Todo Queries ====================

    def get_todo(self, todo_id: str) -> TodoItem | None:
        """Get todo by ID."""
        return self._todos.get(todo_id)

    def get_all_todos(self) -> list[TodoItem]:
        """Get all todos."""
        return list(self._todos.values())

    def get_pending_todos(self) -> list[TodoItem]:
        """Get pending todos."""
        return [t for t in self._todos.values() if t.status == TodoStatus.PENDING]

    def get_in_progress_todos(self) -> list[TodoItem]:
        """Get in-progress todos."""
        return [t for t in self._todos.values() if t.status == TodoStatus.IN_PROGRESS]

    def get_current_todo(self) -> TodoItem | None:
        """Get currently active todo."""
        if self._current_todo_id:
            return self._todos.get(self._current_todo_id)
        return None

    def get_root_todos(self) -> list[TodoItem]:
        """Get top-level todos (no parent)."""
        return [t for t in self._todos.values() if t.parent_id is None]

    # ==================== Auto Todo Generation ====================

    def auto_create_todos(
        self,
        task_description: str,
        complexity: str = "auto"
    ) -> list[TodoItem]:
        """Automatically generate todos from task description.

        Args:
            task_description: Task to break down
            complexity: "simple", "medium", "complex", or "auto"

        Returns:
            List of created todos
        """
        task_lower = task_description.lower()

        # Detect complexity if auto
        if complexity == "auto":
            if len(task_description) < 50 and any(
                p in task_lower for p in ["읽", "보여", "read", "show", "list"]
            ):
                complexity = "simple"
            elif any(
                p in task_lower for p in ["리팩토링", "전체", "refactor", "all", "entire"]
            ):
                complexity = "complex"
            else:
                complexity = "medium"

        todos = []

        if complexity == "simple":
            # Single todo
            todos = [{"content": task_description, "activeForm": self._generate_active_form(task_description)}]

        elif complexity == "medium":
            # Standard breakdown
            if any(p in task_lower for p in ["수정", "변경", "edit", "fix", "modify"]):
                todos = [
                    {"content": "Find and read the target file", "activeForm": "Finding target file"},
                    {"content": "Make the requested changes", "activeForm": "Making changes"},
                    {"content": "Verify the changes work correctly", "activeForm": "Verifying changes"},
                ]
            elif any(p in task_lower for p in ["구현", "만들", "create", "implement", "add"]):
                todos = [
                    {"content": "Analyze requirements and existing code", "activeForm": "Analyzing requirements"},
                    {"content": "Implement the feature", "activeForm": "Implementing feature"},
                    {"content": "Test the implementation", "activeForm": "Testing implementation"},
                ]
            elif any(p in task_lower for p in ["분석", "이해", "analyze", "understand", "explain"]):
                todos = [
                    {"content": "Find relevant files", "activeForm": "Finding relevant files"},
                    {"content": "Read and analyze the code", "activeForm": "Analyzing code"},
                    {"content": "Summarize findings", "activeForm": "Summarizing findings"},
                ]
            else:
                todos = [
                    {"content": "Understand the task requirements", "activeForm": "Understanding requirements"},
                    {"content": "Execute the main task", "activeForm": "Executing task"},
                    {"content": "Verify completion", "activeForm": "Verifying completion"},
                ]

        else:  # complex
            todos = [
                {"content": "Analyze scope and requirements", "activeForm": "Analyzing scope"},
                {"content": "Design the solution approach", "activeForm": "Designing solution"},
                {"content": "Identify affected files and components", "activeForm": "Identifying components"},
                {"content": "Implement core changes", "activeForm": "Implementing core changes"},
                {"content": "Implement secondary changes", "activeForm": "Implementing secondary changes"},
                {"content": "Run tests and fix issues", "activeForm": "Running tests"},
                {"content": "Review and finalize", "activeForm": "Reviewing changes"},
            ]

        return self.add_todos(todos)

    def _generate_active_form(self, content: str) -> str:
        """Generate active form from imperative form."""
        # Simple heuristic: Add -ing for English, 중 for Korean
        if content.endswith("..."):
            return content

        words = content.split()
        if not words:
            return content

        first_word = words[0].lower()

        # English patterns
        ing_map = {
            "find": "Finding",
            "read": "Reading",
            "write": "Writing",
            "create": "Creating",
            "update": "Updating",
            "delete": "Deleting",
            "run": "Running",
            "test": "Testing",
            "fix": "Fixing",
            "add": "Adding",
            "remove": "Removing",
            "implement": "Implementing",
            "analyze": "Analyzing",
            "search": "Searching",
            "check": "Checking",
            "verify": "Verifying",
            "review": "Reviewing",
            "build": "Building",
            "deploy": "Deploying",
            "install": "Installing",
            "configure": "Configuring",
            "design": "Designing",
            "refactor": "Refactoring",
            "optimize": "Optimizing",
            "debug": "Debugging",
        }

        if first_word in ing_map:
            words[0] = ing_map[first_word]
            return " ".join(words)

        # Default: add "Working on"
        return f"Working on: {content}"

    # ==================== Parallel Execution ====================

    async def execute_parallel(
        self,
        todo_coroutines: list[tuple[str, Coroutine]],
        max_concurrent: int = 3
    ) -> list[ExecutionResult]:
        """Execute todos as parallel tasks.

        Args:
            todo_coroutines: List of (todo_id, coroutine) tuples
            max_concurrent: Maximum concurrent executions

        Returns:
            List of ExecutionResults
        """
        if not self._executor:
            config = ExecutionConfig(max_concurrent=max_concurrent)
            self._executor = TaskExecutor(
                registry=self.registry,
                config=config,
                progress_callback=self._on_executor_progress
            )

        # Submit todos as tasks
        for todo_id, coro in todo_coroutines:
            todo = self._todos.get(todo_id)
            if not todo:
                continue

            task_id = self._executor.submit(
                name=todo.content,
                coro=coro,
                description=todo.active_form,
                metadata={"todo_id": todo_id}
            )

            # Link todo to task
            todo.task_id = task_id
            self.start_todo(todo_id)

        # Wait for completion
        results = await self._executor.wait_all()

        # Update todos based on results
        for result in results:
            todo_id = None
            task_info = self.registry.get(result.task_id)
            if task_info and task_info.metadata.get("todo_id"):
                todo_id = task_info.metadata["todo_id"]

            if todo_id:
                if result.success:
                    self.complete_todo(todo_id, result.result)
                else:
                    self.fail_todo(todo_id, result.error or "Unknown error")

        return results

    def _on_executor_progress(self, task_info: TaskInfo) -> None:
        """Handle executor progress updates."""
        todo_id = task_info.metadata.get("todo_id")
        if not todo_id:
            return

        todo = self._todos.get(todo_id)
        if not todo:
            return

        # Sync task state to todo
        if task_info.state == TaskState.RUNNING:
            todo.status = TodoStatus.IN_PROGRESS
        elif task_info.state == TaskState.COMPLETED:
            todo.status = TodoStatus.COMPLETED
        elif task_info.state == TaskState.FAILED:
            todo.status = TodoStatus.FAILED
            todo.error = task_info.error

        todo.metadata["progress"] = task_info.progress

        self._notify_change(todo)

    # ==================== Registry Integration ====================

    def _on_task_state_change(self, task_info: TaskInfo, event: str) -> None:
        """Handle task registry state changes."""
        todo_id = task_info.metadata.get("todo_id")
        if not todo_id or todo_id not in self._todos:
            return

        todo = self._todos[todo_id]

        # Map task state to todo status
        state_map = {
            TaskState.PENDING: TodoStatus.PENDING,
            TaskState.RUNNING: TodoStatus.IN_PROGRESS,
            TaskState.COMPLETED: TodoStatus.COMPLETED,
            TaskState.FAILED: TodoStatus.FAILED,
            TaskState.CANCELLED: TodoStatus.SKIPPED,
            TaskState.TIMEOUT: TodoStatus.FAILED,
        }

        new_status = state_map.get(task_info.state)
        if new_status and new_status != todo.status:
            todo.status = new_status

            if task_info.state == TaskState.FAILED:
                todo.error = task_info.error

            self._notify_change(todo)

    # ==================== State Synchronization ====================

    def _sync_to_state(self) -> None:
        """Sync todos to EnhancedAgentState."""
        if not self.state_sync_callback:
            return

        current = self.get_current_todo()
        update = {
            "current_task": {
                "task_id": current.id if current else None,
                "description": current.content if current else None,
                "status": current.status.value if current else None,
            } if current else None,
            "plan_steps": [t.content for t in self.get_root_todos()],
            "current_plan_step": self._get_current_step_index(),
        }

        try:
            self.state_sync_callback(update)
        except Exception as e:
            logger.error(f"State sync error: {e}")

    def _get_current_step_index(self) -> int:
        """Get current step index for plan_steps."""
        todos = self.get_root_todos()
        for i, todo in enumerate(todos):
            if todo.status == TodoStatus.IN_PROGRESS:
                return i
            if todo.status == TodoStatus.PENDING:
                return max(0, i - 1)
        return len(todos) - 1 if todos else 0

    def sync_from_state(self, state: EnhancedAgentState) -> None:
        """Sync from EnhancedAgentState.

        Args:
            state: Agent state to sync from
        """
        # Get plan_steps from state
        plan_steps = state.get("plan_steps", [])
        if not plan_steps:
            return

        # Check if we need to create todos
        existing_contents = {t.content for t in self._todos.values()}
        for step in plan_steps:
            if step not in existing_contents:
                self.add_todo(step)

    # ==================== Progress Listeners ====================

    def add_progress_listener(self, callback: Callable[[TodoItem], None]) -> None:
        """Add a progress change listener."""
        self._progress_listeners.append(callback)

    def remove_progress_listener(self, callback: Callable[[TodoItem], None]) -> None:
        """Remove a progress listener."""
        if callback in self._progress_listeners:
            self._progress_listeners.remove(callback)

    def _notify_change(self, todo: TodoItem) -> None:
        """Notify listeners of todo change."""
        for listener in self._progress_listeners:
            try:
                listener(todo)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    # ==================== Parent/Child Management ====================

    def _check_parent_completion(self, parent_id: str) -> None:
        """Check if parent should be marked complete."""
        parent = self._todos.get(parent_id)
        if not parent:
            return

        children = [self._todos.get(cid) for cid in parent.children_ids]
        if all(
            c and c.status in (TodoStatus.COMPLETED, TodoStatus.SKIPPED)
            for c in children
        ):
            parent.status = TodoStatus.COMPLETED
            parent.completed_at = datetime.now()
            self._notify_change(parent)
            self._persist_if_auto()

    # ==================== Persistence ====================

    def _persist_if_auto(self) -> None:
        """Persist if auto_persist is enabled."""
        if self.auto_persist:
            self.persist()

    def persist(self) -> None:
        """Save current state to disk."""
        try:
            data = {
                "session_id": self._session_id,
                "todo_counter": self._todo_counter,
                "current_todo_id": self._current_todo_id,
                "todos": {
                    tid: todo.to_dict()
                    for tid, todo in self._todos.items()
                }
            }

            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Persist error: {e}")

    def _load(self) -> None:
        """Load state from disk."""
        if not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, encoding='utf-8') as f:
                data = json.load(f)

            self._session_id = data.get("session_id", self._session_id)
            self._todo_counter = data.get("todo_counter", 0)
            self._current_todo_id = data.get("current_todo_id")

            for tid, todo_data in data.get("todos", {}).items():
                self._todos[tid] = TodoItem.from_dict(todo_data)

        except Exception as e:
            logger.error(f"Load error: {e}")

    # ==================== Statistics ====================

    def get_stats(self) -> dict[str, Any]:
        """Get todo statistics."""
        todos = list(self._todos.values())

        by_status = {}
        for status in TodoStatus:
            by_status[status.value] = len([t for t in todos if t.status == status])

        total = len(todos)
        completed = by_status.get("completed", 0)

        return {
            "total": total,
            "by_status": by_status,
            "completed": completed,
            "in_progress": by_status.get("in_progress", 0),
            "pending": by_status.get("pending", 0),
            "failed": by_status.get("failed", 0),
            "completion_rate": (completed / total * 100) if total > 0 else 0.0,
            "current_todo": self._current_todo_id
        }

    def format_progress(self) -> str:
        """Format progress for display."""
        stats = self.get_stats()
        current = self.get_current_todo()

        lines = []
        lines.append(f"Progress: {stats['completed']}/{stats['total']} completed")

        if current:
            lines.append(f"Current: {current.active_form}")

        return " | ".join(lines)

    def format_todos(self) -> str:
        """Format todos list for display."""
        lines = []
        stats = self.get_stats()

        lines.append(f"📋 Tasks: {stats['completed']}/{stats['total']} completed\n")

        for todo in self.get_root_todos():
            icon = {
                TodoStatus.PENDING: "⏳",
                TodoStatus.IN_PROGRESS: "🔄",
                TodoStatus.COMPLETED: "✅",
                TodoStatus.FAILED: "❌",
                TodoStatus.SKIPPED: "⏭️"
            }.get(todo.status, "❓")

            lines.append(f"{icon} {todo.content}")

            # Show children
            for child_id in todo.children_ids:
                child = self._todos.get(child_id)
                if child:
                    child_icon = {
                        TodoStatus.PENDING: "⏳",
                        TodoStatus.IN_PROGRESS: "🔄",
                        TodoStatus.COMPLETED: "✅",
                        TodoStatus.FAILED: "❌",
                        TodoStatus.SKIPPED: "⏭️"
                    }.get(child.status, "❓")
                    lines.append(f"  {child_icon} {child.content}")

        return "\n".join(lines)

    # ==================== Cleanup ====================

    def clear(self) -> None:
        """Clear all todos."""
        self._todos.clear()
        self._current_todo_id = None
        self._todo_counter = 0
        self._persist_if_auto()

    def clear_completed(self) -> int:
        """Clear completed todos."""
        to_remove = [
            tid for tid, todo in self._todos.items()
            if todo.status in (TodoStatus.COMPLETED, TodoStatus.SKIPPED)
        ]
        for tid in to_remove:
            del self._todos[tid]

        self._persist_if_auto()
        return len(to_remove)

    # ==================== Context Manager ====================

    async def __aenter__(self) -> TaskManager:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.persist()


# Singleton instance
_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get the global TaskManager instance."""
    global _manager
    if _manager is None:
        _manager = TaskManager()
    return _manager


def reset_task_manager() -> None:
    """Reset the global TaskManager instance."""
    global _manager
    if _manager:
        _manager.clear()
    _manager = TaskManager()
