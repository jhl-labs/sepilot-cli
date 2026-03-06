"""Plan Mode tools for LangChain agent.

Provides enter_plan_mode, exit_plan_mode, update_plan tools.
These enable Claude Code-style interactive planning before implementation.
"""

from typing import Any

from langchain_core.tools import tool


@tool
def enter_plan_mode(task_description: str = "") -> str:
    """Enter plan mode to design implementation approach before writing code.

    Use this tool when you're about to start a non-trivial implementation task.
    Getting user sign-off on your approach before writing code prevents wasted
    effort and ensures alignment.

    When to use:
    - New feature implementation
    - Multiple valid approaches exist
    - Code modifications affecting existing behavior
    - Architectural decisions required
    - Multi-file changes
    - Unclear requirements needing exploration

    When NOT to use:
    - Single-line or few-line fixes
    - Tasks with very specific, detailed instructions
    - Pure research/exploration tasks

    Args:
        task_description: Description of the task to plan for (optional)

    Returns:
        Status and instructions for plan mode

    Examples:
        # Enter plan mode for a new feature
        enter_plan_mode(task_description="Add user authentication with JWT tokens")

        # Enter plan mode for refactoring
        enter_plan_mode(task_description="Refactor database layer to use async")
    """
    from sepilot.agent.plan_mode import get_plan_mode_manager

    manager = get_plan_mode_manager()
    result = manager.enter_plan_mode(task_description)

    if result["success"]:
        output = [
            "=" * 60,
            "PLAN MODE ACTIVATED",
            "=" * 60,
            "",
            f"Task: {task_description}" if task_description else "No task specified yet.",
            "",
            result["instructions"],
            "",
            f"Plan file: {result.get('plan_file', 'N/A')}",
            "=" * 60
        ]
        return "\n".join(output)
    else:
        return f"Failed to enter plan mode: {result['message']}"


@tool
def exit_plan_mode(approve: bool = True, launch_swarm: bool = False) -> str:
    """Exit plan mode after completing the plan design.

    Call this when you have finished writing your plan and are ready for
    user approval. The plan should include:
    - Overall approach
    - Step-by-step implementation plan
    - Files that will be modified
    - Any considerations or trade-offs

    Args:
        approve: Whether to mark the plan as approved (default: True)
        launch_swarm: Whether to launch parallel agents for implementation (default: False)

    Returns:
        Final plan summary and status

    Examples:
        # Exit and approve plan
        exit_plan_mode(approve=True)

        # Cancel plan
        exit_plan_mode(approve=False)

        # Exit and launch parallel implementation
        exit_plan_mode(approve=True, launch_swarm=True)
    """
    from sepilot.agent.plan_mode import get_plan_mode_manager

    manager = get_plan_mode_manager()
    result = manager.exit_plan_mode(approve=approve)

    if result["success"]:
        output = [
            "=" * 60,
            "EXITING PLAN MODE",
            "=" * 60,
            "",
            result["message"],
            "",
            result.get("plan", ""),
        ]

        if launch_swarm and approve:
            output.extend([
                "",
                "Swarm mode requested - parallel agents will be launched.",
                "Note: Swarm execution requires user approval."
            ])

        return "\n".join(output)
    else:
        return f"Failed to exit plan mode: {result['message']}"


@tool
def update_plan(
    approach: str = "",
    steps: list[dict[str, Any]] | None = None,
    considerations: list[str] | None = None
) -> str:
    """Update the current plan with approach, steps, or considerations.

    Use this tool while in plan mode to document your implementation plan.

    Args:
        approach: Overall approach/strategy description
        steps: List of implementation steps, each with:
            - description: What this step does (required)
            - files_involved: List of files to modify (optional)
            - notes: Additional notes (optional)
        considerations: List of considerations, trade-offs, or risks

    Returns:
        Updated plan summary

    Examples:
        # Set approach
        update_plan(approach="Use decorator pattern for authentication middleware")

        # Add steps
        update_plan(steps=[
            {"description": "Create auth middleware", "files_involved": ["src/middleware/auth.py"]},
            {"description": "Add JWT validation", "files_involved": ["src/utils/jwt.py"]},
            {"description": "Update routes", "files_involved": ["src/routes/*.py"]}
        ])

        # Add considerations
        update_plan(considerations=[
            "Need to handle token refresh",
            "Consider rate limiting",
            "Backward compatibility with existing sessions"
        ])
    """
    from sepilot.agent.plan_mode import get_plan_mode_manager

    manager = get_plan_mode_manager()

    if not manager.is_active:
        return "Not in plan mode. Use enter_plan_mode() first."

    result = manager.update_plan(
        approach=approach if approach else None,
        steps=steps,
        considerations=considerations
    )

    if result["success"]:
        return f"Plan updated:\n\n{result['plan']}"
    else:
        return f"Failed to update plan: {result['message']}"


@tool
def add_plan_step(
    description: str,
    files_involved: list[str] | None = None,
    notes: str = ""
) -> str:
    """Add a single step to the current plan.

    Use this to incrementally build your plan as you explore the codebase.

    Args:
        description: What this step accomplishes (required)
        files_involved: List of files this step will modify (optional)
        notes: Additional notes or considerations (optional)

    Returns:
        Confirmation and current step count

    Examples:
        # Add a simple step
        add_plan_step(description="Create database migration for users table")

        # Add step with files
        add_plan_step(
            description="Implement user model",
            files_involved=["src/models/user.py", "src/models/__init__.py"]
        )

        # Add step with notes
        add_plan_step(
            description="Add authentication routes",
            files_involved=["src/routes/auth.py"],
            notes="Need to handle both session and token auth"
        )
    """
    from sepilot.agent.plan_mode import get_plan_mode_manager

    manager = get_plan_mode_manager()

    if not manager.is_active:
        return "Not in plan mode. Use enter_plan_mode() first."

    result = manager.add_step(
        description=description,
        files_involved=files_involved,
        notes=notes
    )

    if result["success"]:
        return f"Step added: {description}\nTotal steps: {result['total_steps']}"
    else:
        return f"Failed to add step: {result['message']}"


@tool
def get_plan_status() -> str:
    """Get the current plan status and summary.

    Returns the current plan's status, steps, and overall progress.

    Returns:
        Formatted plan summary or status message
    """
    from sepilot.agent.plan_mode import get_plan_mode_manager

    manager = get_plan_mode_manager()

    if not manager.is_active:
        return "Not in plan mode. No active plan."

    return manager.get_plan_summary()


__all__ = [
    'enter_plan_mode',
    'exit_plan_mode',
    'update_plan',
    'add_plan_step',
    'get_plan_status'
]
