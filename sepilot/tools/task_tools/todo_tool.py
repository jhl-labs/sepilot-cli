"""Todo/Task management tool"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sepilot.tools.base_tool import BaseTool


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TodoTool(BaseTool):
    """Tool for managing todos and tasks"""

    name = "todo"
    description = "Manage todos and tasks"
    parameters = {
        "action": "Action to perform (add/list/update/remove/clear) (required)",
        "task": "Task description (for add action)",
        "task_id": "Task ID (for update/remove actions)",
        "status": "Task status (pending/in_progress/completed/cancelled)",
        "priority": "Task priority (1-5, 1 is highest)"
    }

    def __init__(self, logger=None):
        super().__init__(logger)
        self.todo_file = Path.home() / ".sepilot" / "todos.json"
        self.todo_file.parent.mkdir(parents=True, exist_ok=True)
        self.todos = self._load_todos()

    def _load_todos(self) -> list[dict[str, Any]]:
        """Load todos from file"""
        if self.todo_file.exists():
            try:
                with open(self.todo_file) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_todos(self) -> None:
        """Save todos to file"""
        with open(self.todo_file, 'w') as f:
            json.dump(self.todos, f, indent=2, default=str)

    def execute(
        self,
        action: str,
        task: str | None = None,
        task_id: int | None = None,
        status: str | None = None,
        priority: int | None = None
    ) -> str:
        """Execute todo management actions"""
        self.validate_params(action=action)

        try:
            if action == "add":
                return self._add_task(task, priority)
            elif action == "list":
                return self._list_tasks(status)
            elif action == "update":
                return self._update_task(task_id, status, priority)
            elif action == "remove":
                return self._remove_task(task_id)
            elif action == "clear":
                return self._clear_tasks(status)
            else:
                return f"Error: Unknown action: {action}"

        except Exception as e:
            return f"Todo error: {str(e)}"

    def _add_task(self, task: str | None, priority: int | None) -> str:
        """Add a new task"""
        if not task:
            return "Error: Task description is required"

        new_task = {
            "id": len(self.todos) + 1,
            "task": task,
            "status": TaskStatus.PENDING.value,
            "priority": priority or 3,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        self.todos.append(new_task)
        self._save_todos()

        return f"✅ Added task #{new_task['id']}: {task}\nPriority: {new_task['priority']}/5"

    def _list_tasks(self, status: str | None) -> str:
        """List tasks"""
        if not self.todos:
            return "No tasks found. Add some with 'todo add <task>'"

        # Filter by status if provided
        tasks = self.todos
        if status:
            try:
                status_enum = TaskStatus(status)
                tasks = [t for t in tasks if t["status"] == status_enum.value]
            except ValueError:
                return f"Error: Invalid status. Use: {', '.join([s.value for s in TaskStatus])}"

        if not tasks:
            return f"No tasks with status: {status}"

        # Sort by priority and status
        status_order = {
            TaskStatus.IN_PROGRESS.value: 0,
            TaskStatus.PENDING.value: 1,
            TaskStatus.COMPLETED.value: 2,
            TaskStatus.CANCELLED.value: 3
        }
        tasks.sort(key=lambda x: (status_order.get(x["status"], 99), x["priority"]))

        # Format output
        result = ["📋 Tasks:\n"]

        # Group by status
        for status_value in [TaskStatus.IN_PROGRESS.value, TaskStatus.PENDING.value,
                             TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]:
            status_tasks = [t for t in tasks if t["status"] == status_value]
            if status_tasks:
                status_emoji = {
                    TaskStatus.PENDING.value: "⏳",
                    TaskStatus.IN_PROGRESS.value: "🔄",
                    TaskStatus.COMPLETED.value: "✅",
                    TaskStatus.CANCELLED.value: "❌"
                }[status_value]

                result.append(f"\n{status_emoji} {status_value.upper()}:")
                for task in status_tasks:
                    priority_stars = "⭐" * (6 - task["priority"])
                    result.append(
                        f"  #{task['id']} {priority_stars} {task['task']}"
                    )
                    # Show time info for recent tasks
                    created = datetime.fromisoformat(task["created_at"])
                    age = datetime.now() - created
                    if age.days == 0:
                        result.append(f"      Created: {age.seconds//3600}h ago")

        # Add summary
        total = len(tasks)
        completed = len([t for t in tasks if t["status"] == TaskStatus.COMPLETED.value])
        pending = len([t for t in tasks if t["status"] == TaskStatus.PENDING.value])
        in_progress = len([t for t in tasks if t["status"] == TaskStatus.IN_PROGRESS.value])

        result.append(f"\n📊 Summary: {total} total | {in_progress} in progress | {pending} pending | {completed} completed")

        return "\n".join(result)

    def _update_task(self, task_id: int | None, status: str | None, priority: int | None) -> str:
        """Update a task"""
        if task_id is None:
            return "Error: Task ID is required"

        task = next((t for t in self.todos if t["id"] == task_id), None)
        if not task:
            return f"Task #{task_id} not found"

        updated = []
        if status:
            try:
                status_enum = TaskStatus(status)
                task["status"] = status_enum.value
                updated.append(f"status → {status}")
            except ValueError:
                return f"Error: Invalid status. Use: {', '.join([s.value for s in TaskStatus])}"

        if priority is not None:
            if 1 <= priority <= 5:
                task["priority"] = priority
                updated.append(f"priority → {priority}")
            else:
                return "Error: Priority must be between 1 and 5"

        if updated:
            task["updated_at"] = datetime.now().isoformat()
            self._save_todos()
            return f"Updated task #{task_id}: {', '.join(updated)}\n{task['task']}"
        else:
            return "No changes made"

    def _remove_task(self, task_id: int | None) -> str:
        """Remove a task"""
        if task_id is None:
            return "Error: Task ID is required"

        task = next((t for t in self.todos if t["id"] == task_id), None)
        if not task:
            return f"Task #{task_id} not found"

        self.todos.remove(task)
        self._save_todos()

        return f"Removed task #{task_id}: {task['task']}"

    def _clear_tasks(self, status: str | None) -> str:
        """Clear tasks"""
        if status:
            try:
                status_enum = TaskStatus(status)
                before_count = len(self.todos)
                self.todos = [t for t in self.todos if t["status"] != status_enum.value]
                removed = before_count - len(self.todos)
                self._save_todos()
                return f"Cleared {removed} {status} tasks"
            except ValueError:
                return f"Error: Invalid status. Use: {', '.join([s.value for s in TaskStatus])}"
        else:
            count = len(self.todos)
            self.todos = []
            self._save_todos()
            return f"Cleared all {count} tasks"
