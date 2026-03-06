"""Task management tools for LangChain agent.

Provides plan, todo_manage tools.
"""

import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


@tool
def plan(task: str, approach: str = "sequential") -> str:
    """Create a step-by-step plan for completing a complex task.

    Args:
        task: The complex task to plan for
        approach: Planning approach (sequential/parallel/iterative)
            - sequential: Step-by-step sequential plan (default)
            - parallel: Identify parallelizable tasks
            - iterative: Iterative refinement approach

    Returns:
        Structured plan with numbered steps and recommendations

    Examples:
        # Create sequential plan
        plan(task="Refactor authentication module to use JWT tokens")

        # Identify parallel tasks
        plan(task="Set up CI/CD pipeline", approach="parallel")

        # Iterative approach
        plan(task="Optimize database performance", approach="iterative")
    """
    try:
        result = []
        result.append(f"📋 Plan for: \"{task}\"\n")
        result.append(f"Approach: {approach}\n")

        if approach == "sequential":
            result.append("Steps (execute in order):")
            result.append("")

            if "refactor" in task.lower():
                result.extend([
                    "1. Analyze current implementation",
                    "   - Review existing code",
                    "   - Identify dependencies",
                    "   - Document current behavior",
                    "",
                    "2. Design new structure",
                    "   - Plan architecture changes",
                    "   - Identify breaking changes",
                    "   - Design migration path",
                    "",
                    "3. Implement changes",
                    "   - Make code modifications",
                    "   - Update related components",
                    "   - Maintain backward compatibility if needed",
                    "",
                    "4. Test thoroughly",
                    "   - Run existing tests",
                    "   - Add new tests for changes",
                    "   - Manual verification",
                    "",
                    "5. Document and deploy",
                    "   - Update documentation",
                    "   - Code review",
                    "   - Deploy changes",
                ])

            elif "create" in task.lower() or "add" in task.lower():
                result.extend([
                    "1. Plan structure",
                    "   - Define requirements",
                    "   - Design architecture",
                    "   - Choose technologies",
                    "",
                    "2. Set up foundation",
                    "   - Create project structure",
                    "   - Set up dependencies",
                    "   - Configure environment",
                    "",
                    "3. Implement core functionality",
                    "   - Write main logic",
                    "   - Add error handling",
                    "   - Implement features",
                    "",
                    "4. Add tests",
                    "   - Write unit tests",
                    "   - Add integration tests",
                    "   - Test edge cases",
                    "",
                    "5. Finalize",
                    "   - Write documentation",
                    "   - Code cleanup",
                    "   - Final review",
                ])

            elif "optimize" in task.lower() or "improve" in task.lower():
                result.extend([
                    "1. Profile and measure",
                    "   - Identify bottlenecks",
                    "   - Measure current performance",
                    "   - Set optimization goals",
                    "",
                    "2. Analyze issues",
                    "   - Review code for inefficiencies",
                    "   - Check resource usage",
                    "   - Identify quick wins",
                    "",
                    "3. Implement optimizations",
                    "   - Apply performance improvements",
                    "   - Optimize algorithms",
                    "   - Reduce resource usage",
                    "",
                    "4. Verify improvements",
                    "   - Measure new performance",
                    "   - Compare before/after",
                    "   - Ensure correctness maintained",
                ])

            else:
                result.extend([
                    "1. Understand requirements",
                    "   - Analyze the task",
                    "   - Identify deliverables",
                    "   - Clarify constraints",
                    "",
                    "2. Research and design",
                    "   - Research solutions",
                    "   - Design approach",
                    "   - Plan implementation",
                    "",
                    "3. Implement solution",
                    "   - Write code",
                    "   - Handle edge cases",
                    "   - Add error handling",
                    "",
                    "4. Test and verify",
                    "   - Test implementation",
                    "   - Verify requirements met",
                    "   - Fix any issues",
                    "",
                    "5. Document and finalize",
                    "   - Add documentation",
                    "   - Clean up code",
                    "   - Final review",
                ])

        elif approach == "parallel":
            result.extend([
                "Parallelizable tasks:",
                "",
                "Track A (Foundation):",
                "  1. Set up project structure",
                "  2. Configure dependencies",
                "",
                "Track B (Implementation):",
                "  1. Implement core features",
                "  2. Add business logic",
                "",
                "Track C (Quality):",
                "  1. Write tests",
                "  2. Add documentation",
                "",
                "Synchronization points:",
                "  - After Track A: Tracks B & C can start",
                "  - Final integration: Merge all tracks",
            ])

        elif approach == "iterative":
            result.extend([
                "Iteration plan (incremental refinement):",
                "",
                "Iteration 1 (MVP):",
                "  - Minimal working version",
                "  - Core functionality only",
                "  - Quick validation",
                "",
                "Iteration 2 (Enhancement):",
                "  - Add key features",
                "  - Improve usability",
                "  - Basic error handling",
                "",
                "Iteration 3 (Polish):",
                "  - Optimize performance",
                "  - Complete error handling",
                "  - Add documentation",
                "",
                "Iteration 4 (Production-ready):",
                "  - Full testing",
                "  - Security review",
                "  - Deployment prep",
            ])

        result.extend([
            "",
            "─" * 50,
            "",
            "💡 Recommendation:",
            "Use 'todo_manage' tool to track progress:",
            "",
            "  todo_manage(action='create',",
            "              task_description=\"" + task[:50] + "\",",
            "              todos=[",
            "                  {\"task\": \"Step 1\", \"status\": \"pending\"},",
            "                  {\"task\": \"Step 2\", \"status\": \"pending\"},",
            "                  ...",
            "              ])",
        ])

        return "\n".join(result)

    except Exception as e:
        return f"Planning error: {str(e)}"


@tool
def todo_manage(action: str, todos: list[dict[str, str]] | None = None, task_description: str = "") -> str:
    """Manage todo list for tracking complex tasks and progress.

    Args:
        action: Action to perform
            - create: Create new todo list from scratch
            - update: Update existing todos (change status, add/remove items)
            - list: Show current todos
            - clear: Clear all todos
        todos: List of todos (for create/update actions)
            Format: [{"task": "Task description", "status": "pending|in_progress|completed"}]
        task_description: Overall task description (for create action)

    Returns:
        Success message or current todo list

    Examples:
        # Create todo list
        todo_manage(action="create", task_description="Refactor auth module", todos=[
            {"task": "Analyze current code", "status": "pending"},
            {"task": "Design new structure", "status": "pending"}
        ])

        # Update progress
        todo_manage(action="update", todos=[
            {"task": "Analyze current code", "status": "completed"},
            {"task": "Design new structure", "status": "in_progress"}
        ])

        # List current todos
        todo_manage(action="list")

        # Clear all
        todo_manage(action="clear")
    """
    todo_file = Path.home() / ".sepilot" / "current_todos.json"
    todo_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_todos():
        if todo_file.exists():
            try:
                with open(todo_file) as f:
                    return json.load(f)
            except Exception:
                return {"todos": [], "task_description": "", "created_at": None}
        return {"todos": [], "task_description": "", "created_at": None}

    def _save_todos(data):
        data["updated_at"] = datetime.now().isoformat()
        with open(todo_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _format_todos(data):
        if not data.get("todos"):
            return "No todos found. Create some with action='create'."

        todos_list = data["todos"]
        total = len(todos_list)
        completed = len([t for t in todos_list if t.get("status") == "completed"])
        in_progress = len([t for t in todos_list if t.get("status") == "in_progress"])
        pending = len([t for t in todos_list if t.get("status") == "pending"])

        result = []
        if data.get("task_description"):
            result.append(f"📋 Task: {data['task_description']}\n")

        result.append(f"Progress: {completed}/{total} completed")
        result.append(f"({in_progress} in progress, {pending} pending)\n")

        for i, todo in enumerate(todos_list, 1):
            status = todo.get("status", "pending")
            task = todo.get("task", "")

            if status == "completed":
                result.append(f"✅ {i}. {task} [completed]")
            elif status == "in_progress":
                result.append(f"🔄 {i}. {task} [in progress]")
            else:
                result.append(f"⏳ {i}. {task} [pending]")

        return "\n".join(result)

    try:
        if action == "create":
            if not todos:
                return "Error: 'todos' parameter required for create action"

            data = {
                "task_description": task_description,
                "created_at": datetime.now().isoformat(),
                "todos": []
            }

            for i, todo in enumerate(todos, 1):
                data["todos"].append({
                    "id": i,
                    "task": todo.get("task", ""),
                    "status": todo.get("status", "pending"),
                    "created_at": datetime.now().isoformat()
                })

            _save_todos(data)
            return f"✅ Created todo list with {len(todos)} tasks\n\n" + _format_todos(data)

        elif action == "update":
            if not todos:
                return "Error: 'todos' parameter required for update action"

            data = _load_todos()
            if not data.get("todos"):
                return "Error: No existing todos to update. Use action='create' first."

            existing = {t["task"]: t for t in data["todos"]}

            for new_todo in todos:
                task_name = new_todo.get("task", "")
                if task_name in existing:
                    existing[task_name]["status"] = new_todo.get("status", "pending")
                    if new_todo["status"] == "completed" and "completed_at" not in existing[task_name]:
                        existing[task_name]["completed_at"] = datetime.now().isoformat()
                    elif new_todo["status"] == "in_progress" and "started_at" not in existing[task_name]:
                        existing[task_name]["started_at"] = datetime.now().isoformat()
                else:
                    new_id = max([t["id"] for t in data["todos"]]) + 1
                    data["todos"].append({
                        "id": new_id,
                        "task": task_name,
                        "status": new_todo.get("status", "pending"),
                        "created_at": datetime.now().isoformat()
                    })

            _save_todos(data)
            return "✅ Updated todo list\n\n" + _format_todos(data)

        elif action == "list":
            data = _load_todos()
            return _format_todos(data)

        elif action == "clear":
            if todo_file.exists():
                todo_file.unlink()
            return "✅ Cleared all todos"

        else:
            return f"Error: Unknown action '{action}'. Valid: create, update, list, clear"

    except Exception as e:
        return f"Todo management error: {str(e)}"


__all__ = ['plan', 'todo_manage']
