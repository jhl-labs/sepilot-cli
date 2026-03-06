"""Agent Mode Manager — PLAN / CODE / EXEC mode system.

Controls tool availability per mode and manages automatic mode transitions.
Inspired by OpenCode / Claude Code workflow patterns.

AgentMode vs AgentStrategy — orthogonal axes:
- AgentMode (PLAN/CODE/EXEC): "which tools are available" — tool access gate
- AgentStrategy (EXPLORE/IMPLEMENT/DEBUG/...): "what strategic intent" — LLM thinking guide
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sepilot.agent.enhanced_state import AgentMode, AgentStrategy

if TYPE_CHECKING:
    from sepilot.agent.enhanced_state import EnhancedAgentState


def _is_error_resolved(error: Any) -> bool:
    """Support both ErrorRecord objects and plain dict entries."""
    if isinstance(error, dict):
        return bool(error.get("resolved", False))
    return bool(getattr(error, "resolved", False))

# Maximum iterations per mode before forced transition/completion
EXEC_MAX_ITERATIONS = 3  # EXEC: verify quickly then finish

# ============================================================================
# Mode-specific allowed tools
# ============================================================================

_PLAN_TOOLS = frozenset({
    # Read / explore
    "file_read", "find_file", "search_content", "codebase",
    "get_structure", "find_definition", "list_directory",
    "get_file_info", "code_analyze", "web_search", "web_fetch",
    "image_read", "pdf_read", "multimedia_info",
    # Planning helpers
    "enter_plan_mode", "exit_plan_mode", "update_plan",
    "add_plan_step", "get_plan_status",
    "plan", "todo_manage", "ask_user", "slash_command", "skill",
    "think",
})

_SWE_PLAN_TOOLS = _PLAN_TOOLS | frozenset({
    "bash_execute", "bash_background", "bash_output",  # Bug reproduction
})

MODE_TOOLS: dict[str, frozenset[str] | None] = {
    "plan": _PLAN_TOOLS,
    "code": _PLAN_TOOLS | frozenset({
        # Write tools
        "file_edit", "file_write", "apply_patch", "notebook_edit",
        # Bash / execution
        "bash_execute", "bash_background", "bash_output",
        "kill_shell", "list_shells", "git",
    }),
    "exec": _PLAN_TOOLS | frozenset({
        # Bash / execution (no file_edit/file_write)
        "bash_execute", "bash_background", "bash_output",
        "kill_shell", "list_shells", "git",
    }),
    "auto": None,  # None = no filter, all tools available
}


def get_mode_filtered_tools(all_tools: list, mode: AgentMode, prompt_profile: str = "default") -> list:
    """Filter tool list based on current mode.

    Args:
        all_tools: Complete list of LangChain tools
        mode: Current agent mode
        prompt_profile: Profile name ("swe_bench" enables bash in PLAN mode)

    Returns:
        Filtered tool list (or full list for AUTO mode)
    """
    if mode == AgentMode.AUTO:
        return all_tools
    if mode == AgentMode.PLAN and prompt_profile.startswith("swe_bench"):
        allowed = _SWE_PLAN_TOOLS
    else:
        allowed = MODE_TOOLS.get(mode.value)
    if allowed is None:
        return all_tools
    return [t for t in all_tools if t.name in allowed]


def is_tool_allowed(tool_name: str, mode: AgentMode) -> bool:
    """Check if a specific tool is allowed in the given mode."""
    if mode == AgentMode.AUTO:
        return True
    allowed = MODE_TOOLS.get(mode.value)
    if allowed is None:
        return True
    return tool_name in allowed


# ============================================================================
# Mode prompts injected into system messages
# ============================================================================

MODE_PROMPTS: dict[str, str] = {
    "plan": (
        "[MODE: PLAN — 탐색·분석]\n"
        "Read-only mode. 코드를 읽고 구조를 파악하세요.\n"
        "Available: file_read, find_file, search_content, codebase, get_structure\n"
        "NOT available: file_edit, file_write, bash_execute\n"
        "Goal: 문제를 이해하고 수정 계획을 수립하세요."
    ),
    "code": (
        "[MODE: CODE — 구현·수정]\n"
        "Implementation mode. 파일을 수정하고 구현하세요.\n"
        "Available: read tools + file_edit, file_write, bash_execute\n"
        "NOT available: (없음 — 대부분의 도구 사용 가능)\n"
        "Goal: 최소한의 정확한 변경을 만드세요. file_edit 사용이 핵심입니다."
    ),
    "exec": (
        "[MODE: EXEC — 실행·검증]\n"
        "Verification mode. 명령을 실행하고 결과를 검증하세요.\n"
        "Available: read tools + bash_execute, git\n"
        "NOT available: file_edit, file_write\n"
        "Goal: 테스트를 실행하고 변경사항을 검증하세요."
    ),
}


SWE_MODE_PROMPTS: dict[str, str] = {
    "plan": (
        "[MODE: PLAN — Bug Analysis]\n"
        "Explore the codebase to understand the bug. "
        "Use file_read, find_file, search_content, bash_execute.\n"
        "Goal: Identify the root cause and the exact file+line to fix."
    ),
    "code": (
        "[MODE: CODE — Fix Implementation]\n"
        "You can now edit files. "
        "Available: all tools including file_edit, file_write, bash_execute.\n"
        "Goal: Make the minimal correct change to fix the bug."
    ),
    "exec": (
        "[MODE: EXEC — Verification]\n"
        "Run tests to verify your fix. Use bash_execute.\n"
        "If tests fail, you can return to CODE mode to fix."
    ),
}


def get_mode_prompt(mode: AgentMode, prompt_profile: str = "default") -> str | None:
    """Get mode-specific prompt injection. Returns None for AUTO mode."""
    if mode == AgentMode.AUTO:
        return None
    prompts = SWE_MODE_PROMPTS if prompt_profile.startswith("swe_bench") else MODE_PROMPTS
    return prompts.get(mode.value)


# ============================================================================
# Mode Transition Engine
# ============================================================================

class ModeTransitionEngine:
    """Suggests automatic mode transitions based on agent state.

    Design philosophy (inspired by Claude Code / OpenCode):
    - Modes control tool availability, not workflow stages
    - Trust the LLM to decide when to transition
    - Only provide gentle safety nets (max iteration limits)
    - Same rules for all profiles — no profile-specific forcing

    If mode_locked=True, no transitions are suggested.
    """

    def suggest_transition(
        self,
        state: EnhancedAgentState,
        prompt_profile: str = "default",
    ) -> AgentMode | None:
        """Analyze current state and suggest a mode transition.

        Args:
            state: Current agent state
            prompt_profile: Reserved for future use (currently unused)

        Returns:
            Suggested new mode, or None if no transition needed.
        """
        if state.get("mode_locked", False):
            return None

        current_mode = state.get("current_mode", AgentMode.AUTO)
        mode_iter = state.get("mode_iteration_count", 0)
        strategy = state.get("current_strategy", AgentStrategy.EXPLORE)
        plan_created = state.get("plan_created", False)
        file_changes = state.get("file_changes", [])
        modified_files = state.get("modified_files", [])
        has_file_changes = bool(file_changes) or bool(modified_files)
        tool_history = state.get("tool_call_history", [])

        if current_mode == AgentMode.PLAN:
            return self._suggest_from_plan(mode_iter, strategy, plan_created, tool_history)
        elif current_mode == AgentMode.CODE:
            return self._suggest_from_code(mode_iter, strategy, has_file_changes, tool_history)
        elif current_mode == AgentMode.EXEC:
            return self._suggest_from_exec(mode_iter, state)
        elif current_mode == AgentMode.AUTO:
            return self._suggest_from_auto(strategy, has_file_changes, tool_history)
        return None

    def _suggest_from_plan(
        self,
        mode_iter: int,
        strategy: AgentStrategy,
        plan_created: bool,
        tool_history: list | None = None,
    ) -> AgentMode | None:
        """PLAN mode transition rules."""
        # Plan created and explored enough → CODE
        if plan_created and mode_iter >= 2:
            return AgentMode.CODE
        # Strategy shifted to implementation intent → CODE
        impl_strategies = {AgentStrategy.IMPLEMENT, AgentStrategy.REFACTOR}
        if strategy in impl_strategies and mode_iter >= 3:
            return AgentMode.CODE
        # Evidence-based: sufficient file reads indicate understanding achieved
        if mode_iter >= 4 and tool_history:
            recent = tool_history[-min(len(tool_history), 8):]
            read_tools = {"file_read", "search_content", "find_file", "codebase"}
            reads = sum(1 for tc in recent if getattr(tc, "tool_name", "") in read_tools)
            if reads >= 3:
                return AgentMode.CODE
        # Maximum exploration limit → CODE (safety net)
        if mode_iter >= 5:
            return AgentMode.CODE
        return None

    def _suggest_from_code(
        self,
        mode_iter: int,
        strategy: AgentStrategy,
        has_file_changes: bool,
        tool_history: list | None = None,
    ) -> AgentMode | None:
        """CODE mode transition rules."""
        if not has_file_changes:
            return None
        # File changes exist and strategy is TEST → EXEC
        if strategy == AgentStrategy.TEST:
            return AgentMode.EXEC
        # Evidence-based: file edits + bash execution attempted → ready for EXEC
        if tool_history:
            recent = tool_history[-min(len(tool_history), 6):]
            has_edits = any(
                getattr(tc, "tool_name", "") in {"file_edit", "file_write"}
                for tc in recent
            )
            has_exec = any(
                getattr(tc, "tool_name", "") == "bash_execute"
                for tc in recent
            )
            if has_edits and has_exec:
                return AgentMode.EXEC
        # Fallback: spent enough iterations with file changes → EXEC
        if mode_iter >= 3:
            return AgentMode.EXEC
        return None

    def _suggest_from_exec(
        self,
        mode_iter: int,
        state: EnhancedAgentState,
    ) -> AgentMode | None:
        """EXEC mode transition rules."""
        # Check for unresolved errors requiring code changes
        error_history = state.get("error_history", [])
        unresolved_errors = [e for e in error_history if not _is_error_resolved(e)]
        if unresolved_errors:
            return AgentMode.CODE
        # No errors and spent enough iterations → stay (completion signal)
        # Note: EXEC_MAX_ITERATIONS enforcement is in _iteration_guard_node
        return None

    @staticmethod
    def should_force_exec_completion(mode_iter: int) -> bool:
        """Check if EXEC mode has exceeded max iterations and should complete."""
        return mode_iter >= EXEC_MAX_ITERATIONS

    def _suggest_from_auto(
        self,
        strategy: AgentStrategy,
        has_file_changes: bool,
        tool_history: list | None = None,
    ) -> AgentMode | None:
        """AUTO mode: suggest effective mode based on strategy."""
        if strategy in {AgentStrategy.EXPLORE, AgentStrategy.DOCUMENT}:
            return AgentMode.PLAN
        if strategy in {AgentStrategy.IMPLEMENT, AgentStrategy.REFACTOR}:
            return AgentMode.CODE
        if strategy in {AgentStrategy.TEST, AgentStrategy.DEBUG}:
            if has_file_changes:
                return AgentMode.EXEC
            # DEBUG without file changes: need to explore first
            if strategy == AgentStrategy.DEBUG:
                return AgentMode.PLAN
        return None


# ============================================================================
# Mode-aware calibration helpers
# ============================================================================

def mode_calibration_cap(
    mode: AgentMode,
    is_modification: bool,
    plan_created: bool,
    has_file_changes: bool,
    exec_tools_used: bool,
) -> float | None:
    """Return a confidence cap based on mode-specific heuristics.

    Returns None if no cap should be applied.
    """
    if mode == AgentMode.PLAN:
        # PLAN mode: completing without a plan → very low
        if is_modification and not plan_created:
            return 0.25
    elif mode == AgentMode.CODE:
        # CODE mode: completing without file changes → low
        if is_modification and not has_file_changes:
            return 0.35
    elif mode == AgentMode.EXEC:
        # EXEC mode: completing without running commands → moderate
        if not exec_tools_used:
            return 0.4
    return None
