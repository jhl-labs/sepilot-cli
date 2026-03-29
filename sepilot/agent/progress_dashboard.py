"""Progress Dashboard — fact-based progress info for LLM self-assessment.

Instead of hardcoded nudges ("YOU MUST EDIT NOW!"), we provide the LLM with
a factual progress dashboard and let it decide what to do next.
"""

from __future__ import annotations

from typing import Any


def build_progress_dashboard(
    state: dict[str, Any],
    context_window: int,
    is_swe_bench: bool,
) -> str:
    """Build a fact-based progress dashboard injected into the execution specialist message.

    Returns a compact (≤32K) or detailed (>32K) dashboard string.
    """
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 50)
    pct = int(iteration / max_iter * 100) if max_iter else 0

    tool_history = state.get("tool_call_history", [])
    tool_count = len(tool_history)

    # Recent tools (last 5)
    recent_tools = [tc.tool_name for tc in tool_history[-5:]] if tool_history else []

    # File edit tracking
    file_changes = state.get("file_changes", [])
    modified_files = state.get("modified_files", [])
    edit_count = len(file_changes) + len(modified_files)
    edited_paths = list({
        *(fc.file_path for fc in file_changes if hasattr(fc, "file_path")),
        *modified_files,
    })

    # Error tracking
    error_history = state.get("error_history", [])
    unresolved_errors = len(error_history)

    # Plan tracking
    plan_steps = state.get("plan_steps", [])
    current_step = state.get("current_plan_step", 0)

    compact = context_window <= 32768

    if compact:
        # ≤32K: minimal (~150 tokens)
        lines = [
            "═══ PROGRESS ═══",
            f"Iter: {iteration}/{max_iter} ({pct}%) | Tools: {tool_count} | Edits: {edit_count} | Errors: {unresolved_errors}",
        ]
        if recent_tools:
            lines.append(f"Recent: {' → '.join(recent_tools[-3:])}")
        if plan_steps:
            lines.append(f"Plan: step {current_step + 1}/{len(plan_steps)}")
    else:
        # >32K: detailed (~300 tokens)
        lines = [
            "═══ PROGRESS ═══",
            f"Iteration: {iteration}/{max_iter} ({pct}%)",
            f"Tools called: {tool_count} | Recent: {' → '.join(recent_tools) if recent_tools else '(none)'}",
        ]

        if tool_history:
            unique_tools = sorted({tc.tool_name for tc in tool_history})
            lines.append(f"Unique tools: {', '.join(unique_tools)}")

        if edited_paths:
            display_paths = [p.split("/")[-1] for p in edited_paths[:5]]
            lines.append(f"Files edited: {edit_count} ({', '.join(display_paths)})")
        else:
            lines.append("Files edited: 0 (none)")

        if unresolved_errors:
            last_err = error_history[-1]
            err_msg = (
                last_err.message[:60]
                if hasattr(last_err, "message")
                else str(last_err)[:60]
            )
            lines.append(f"Errors: {unresolved_errors} unresolved (last: {err_msg})")
        else:
            lines.append("Errors: 0")

        if plan_steps:
            lines.append(f"Plan: step {current_step + 1}/{len(plan_steps)}")

    return "\n".join(lines)


def build_self_assessment(
    state: dict[str, Any],
    is_swe_bench: bool,
    context_window: int,
) -> str:
    """Build a self-assessment prompt that replaces hardcoded nudges.

    Adapts urgency based on iteration progress and edit status.
    """
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 50)
    pct = int(iteration / max_iter * 100) if max_iter else 0

    file_changes = state.get("file_changes", [])
    modified_files = state.get("modified_files", [])
    has_edits = bool(file_changes) or bool(modified_files)

    compact = context_window <= 32768

    # Urgency tiers based on progress
    if has_edits:
        # Already editing — light guidance
        action_line = "Continue verifying and refining your fix."
    elif pct >= 40:
        # 40%+ budget used without edits — strong directive
        action_line = (
            f"WARNING: {pct}% budget used, 0 files edited. "
            "Use file_read on the relevant file then file_edit NOW. "
            "Stop exploring — fix the bug immediately."
        )
    elif iteration >= 3:
        # 3+ iterations without edits — moderate directive
        action_line = (
            f"Note: {iteration} iterations used, 0 files edited. "
            "You should use file_read on the relevant file then file_edit to fix the bug now."
        )
    else:
        # Early iterations — workflow guidance
        action_line = (
            "Workflow: find_file → file_read → file_edit. "
            "Once you find the relevant file, read it and edit it."
        )

    if compact:
        return (
            "═══ NEXT ACTION ═══\n"
            f"{action_line}"
        )
    else:
        lines = [
            "═══ SELF-ASSESSMENT ═══",
            f"{action_line}",
            "Expected workflow: find_file/search_content → file_read → file_edit → bash_execute (test)",
        ]
        return "\n".join(lines)
