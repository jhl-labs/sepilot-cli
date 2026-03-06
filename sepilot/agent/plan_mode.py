"""Plan Mode - Interactive planning mode for complex tasks.

Provides EnterPlanMode/ExitPlanMode functionality similar to Claude Code.
Plan mode allows the agent to explore the codebase and design implementation
approaches before writing code.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """A single step in the plan."""
    description: str
    status: str = "pending"  # pending, in_progress, completed, skipped
    notes: str = ""
    files_involved: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """A complete plan for a task."""
    task_description: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    status: str = "draft"  # draft, approved, in_progress, completed
    approach: str = ""
    considerations: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class PlanModeManager:
    """Manages plan mode state and operations."""

    def __init__(self, persist_dir: str | None = None):
        """Initialize Plan Mode Manager.

        Args:
            persist_dir: Directory to store plan files
        """
        if persist_dir is None:
            persist_dir = str(Path.home() / ".sepilot" / "plans")

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._active_plan: Plan | None = None
        self._plan_mode_active: bool = False
        self._plan_file: Path | None = None

        logger.info(f"PlanModeManager initialized: {persist_dir}")

    @property
    def is_active(self) -> bool:
        """Check if plan mode is currently active."""
        return self._plan_mode_active

    @property
    def current_plan(self) -> Plan | None:
        """Get the current active plan."""
        return self._active_plan

    def enter_plan_mode(self, task_description: str = "") -> dict[str, Any]:
        """Enter plan mode for designing implementation approach.

        Args:
            task_description: Description of the task to plan

        Returns:
            Status information about entering plan mode
        """
        if self._plan_mode_active:
            return {
                "success": False,
                "message": "Already in plan mode. Use exit_plan_mode() first.",
                "current_plan": self._format_plan()
            }

        # Create new plan
        self._active_plan = Plan(task_description=task_description)
        self._plan_mode_active = True

        # Create plan file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._plan_file = self.persist_dir / f"plan_{timestamp}.json"
        self._save_plan()

        logger.info(f"Entered plan mode for: {task_description[:50]}...")

        return {
            "success": True,
            "message": "Entered plan mode. You can now explore the codebase and design your approach.",
            "plan_file": str(self._plan_file),
            "instructions": self._get_plan_mode_instructions()
        }

    def exit_plan_mode(self, approve: bool = True) -> dict[str, Any]:
        """Exit plan mode, optionally approving the plan.

        Args:
            approve: Whether to approve the plan for implementation

        Returns:
            Status information about exiting plan mode
        """
        if not self._plan_mode_active:
            return {
                "success": False,
                "message": "Not currently in plan mode."
            }

        if not self._active_plan:
            self._plan_mode_active = False
            return {
                "success": False,
                "message": "No active plan found."
            }

        if approve:
            self._active_plan.status = "approved"
            message = "Plan approved. Ready for implementation."
        else:
            self._active_plan.status = "cancelled"
            message = "Plan cancelled."

        self._active_plan.updated_at = datetime.now().isoformat()
        self._save_plan()

        result = {
            "success": True,
            "message": message,
            "plan": self._format_plan(),
            "plan_file": str(self._plan_file) if self._plan_file else None
        }

        # Clear active plan mode
        self._plan_mode_active = False
        logger.info(f"Exited plan mode: {self._active_plan.status}")

        return result

    def update_plan(
        self,
        approach: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        considerations: list[str] | None = None
    ) -> dict[str, Any]:
        """Update the current plan.

        Args:
            approach: Overall approach description
            steps: List of plan steps
            considerations: List of considerations/trade-offs

        Returns:
            Updated plan information
        """
        if not self._plan_mode_active or not self._active_plan:
            return {
                "success": False,
                "message": "Not in plan mode. Use enter_plan_mode() first."
            }

        if approach:
            self._active_plan.approach = approach

        if steps:
            self._active_plan.steps = [
                PlanStep(
                    description=s.get("description", ""),
                    status=s.get("status", "pending"),
                    notes=s.get("notes", ""),
                    files_involved=s.get("files_involved", [])
                )
                for s in steps
            ]

        if considerations:
            self._active_plan.considerations = considerations

        self._active_plan.updated_at = datetime.now().isoformat()
        self._save_plan()

        return {
            "success": True,
            "message": "Plan updated.",
            "plan": self._format_plan()
        }

    def add_step(
        self,
        description: str,
        files_involved: list[str] | None = None,
        notes: str = ""
    ) -> dict[str, Any]:
        """Add a step to the current plan.

        Args:
            description: Step description
            files_involved: Files this step will touch
            notes: Additional notes

        Returns:
            Updated plan information
        """
        if not self._plan_mode_active or not self._active_plan:
            return {
                "success": False,
                "message": "Not in plan mode."
            }

        step = PlanStep(
            description=description,
            files_involved=files_involved or [],
            notes=notes
        )
        self._active_plan.steps.append(step)
        self._active_plan.updated_at = datetime.now().isoformat()
        self._save_plan()

        return {
            "success": True,
            "message": f"Added step: {description}",
            "total_steps": len(self._active_plan.steps)
        }

    def get_plan_summary(self) -> str:
        """Get a formatted summary of the current plan."""
        if not self._active_plan:
            return "No active plan."

        return self._format_plan()

    def _format_plan(self) -> str:
        """Format the plan for display."""
        if not self._active_plan:
            return "No plan available."

        plan = self._active_plan
        lines = []

        lines.append("=" * 60)
        lines.append("IMPLEMENTATION PLAN")
        lines.append("=" * 60)
        lines.append("")

        if plan.task_description:
            lines.append(f"Task: {plan.task_description}")
            lines.append("")

        if plan.approach:
            lines.append("Approach:")
            lines.append(f"  {plan.approach}")
            lines.append("")

        if plan.considerations:
            lines.append("Considerations:")
            for cons in plan.considerations:
                lines.append(f"  - {cons}")
            lines.append("")

        if plan.steps:
            lines.append("Steps:")
            for i, step in enumerate(plan.steps, 1):
                status_icon = {
                    "pending": "[ ]",
                    "in_progress": "[*]",
                    "completed": "[x]",
                    "skipped": "[-]"
                }.get(step.status, "[ ]")

                lines.append(f"  {status_icon} {i}. {step.description}")
                if step.files_involved:
                    lines.append(f"       Files: {', '.join(step.files_involved)}")
                if step.notes:
                    lines.append(f"       Note: {step.notes}")
            lines.append("")

        lines.append(f"Status: {plan.status}")
        lines.append(f"Created: {plan.created_at}")
        lines.append(f"Updated: {plan.updated_at}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _get_plan_mode_instructions(self) -> str:
        """Get instructions for plan mode."""
        return """
Plan Mode Instructions:
-----------------------
In plan mode, you should:

1. EXPLORE the codebase:
   - Use file_read, search_content, find_file to understand existing code
   - Identify patterns and conventions used in the project
   - Find relevant files that will need modification

2. DESIGN your approach:
   - Use update_plan() to set your overall approach
   - Use add_step() to add implementation steps
   - Consider edge cases and potential issues

3. DOCUMENT considerations:
   - List architectural trade-offs
   - Note any assumptions
   - Identify potential risks

4. GET APPROVAL:
   - When ready, use exit_plan_mode(approve=True)
   - The user will review and approve your plan
   - Only then should you start implementation

Available tools in plan mode:
- file_read, search_content, find_file (exploration)
- update_plan, add_step (planning)
- exit_plan_mode (finish planning)

Do NOT modify files while in plan mode!
"""

    def _save_plan(self):
        """Save the current plan to file."""
        if not self._active_plan or not self._plan_file:
            return

        try:
            data = {
                "task_description": self._active_plan.task_description,
                "approach": self._active_plan.approach,
                "status": self._active_plan.status,
                "created_at": self._active_plan.created_at,
                "updated_at": self._active_plan.updated_at,
                "considerations": self._active_plan.considerations,
                "steps": [asdict(s) for s in self._active_plan.steps]
            }

            with open(self._plan_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save plan: {e}")

    def load_plan(self, plan_file: str) -> dict[str, Any]:
        """Load a plan from file.

        Args:
            plan_file: Path to the plan file

        Returns:
            Loaded plan information
        """
        try:
            path = Path(plan_file)
            if not path.exists():
                return {"success": False, "message": f"Plan file not found: {plan_file}"}

            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            plan_status = data.get("status", "draft")
            self._active_plan = Plan(
                task_description=data.get("task_description", ""),
                approach=data.get("approach", ""),
                status=plan_status,
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
                considerations=data.get("considerations", []),
                steps=[
                    PlanStep(**s) for s in data.get("steps", [])
                ]
            )
            self._plan_file = path
            # Only activate plan mode for actionable plans
            self._plan_mode_active = plan_status in ("draft", "approved", "in_progress")

            return {
                "success": True,
                "message": f"Loaded plan from {plan_file}",
                "plan": self._format_plan()
            }

        except Exception as e:
            return {"success": False, "message": f"Failed to load plan: {e}"}


# Thread-isolated plan mode managers
_plan_mode_managers: dict[str, PlanModeManager] = {}
_default_plan_mode_manager: PlanModeManager | None = None


def get_plan_mode_manager(thread_id: str | None = None) -> PlanModeManager:
    """Get or create a Plan Mode manager instance for the given thread.

    Args:
        thread_id: Thread ID for isolation. If None, returns a default instance.

    Returns:
        PlanModeManager instance for the given thread.
    """
    global _default_plan_mode_manager

    if thread_id is None:
        if _default_plan_mode_manager is None:
            _default_plan_mode_manager = PlanModeManager()
        return _default_plan_mode_manager

    if thread_id not in _plan_mode_managers:
        _plan_mode_managers[thread_id] = PlanModeManager()
    return _plan_mode_managers[thread_id]
