"""Agent Mode slash commands for Interactive Mode.

Commands:
  /plan  — Switch to PLAN mode (read-only exploration)
  /code  — Switch to CODE mode (implementation)
  /exec  — Switch to EXEC mode (verification)
  /auto  — Return to AUTO mode (system decides)
  /mode  — Show current mode status
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console

from sepilot.agent.enhanced_state import AgentMode

if TYPE_CHECKING:
    pass


def handle_mode_command(
    console: Console,
    agent: Any,
    mode_input: str,
) -> None:
    """Show current mode status.

    Args:
        console: Rich console
        agent: ReactAgent instance
        mode_input: Raw input after /mode
    """
    # Try to read mode from agent's pending update or last known state
    pending = getattr(agent, '_pending_mode_update', None)
    if pending and "current_mode" in pending:
        current_mode = pending["current_mode"]
        locked = pending.get("mode_locked", False)
    else:
        # Fallback: read from last graph state
        current_mode, locked = _get_current_mode(agent)

    mode_name = current_mode.value if isinstance(current_mode, AgentMode) else str(current_mode)
    lock_icon = " 🔒" if locked else " 🔄"

    mode_descriptions = {
        "plan": "PLAN — 탐색·분석 (read-only)",
        "code": "CODE — 구현·수정 (read + write + bash)",
        "exec": "EXEC — 실행·검증 (read + bash)",
        "auto": "AUTO — 시스템 자동 결정",
    }

    desc = mode_descriptions.get(mode_name, mode_name.upper())
    console.print(f"\n[bold cyan]🎯 Current Mode:[/bold cyan] {desc}{lock_icon}")
    if locked:
        console.print("[dim]  (수동 고정 — 자동 전환 비활성화. /auto로 해제)[/dim]")
    else:
        console.print("[dim]  (자동 전환 활성화)[/dim]")
    console.print()


def handle_plan_mode(console: Console, agent: Any) -> None:
    """Switch to PLAN mode (locked)."""
    _set_mode(console, agent, AgentMode.PLAN, locked=True)


def handle_code_mode(console: Console, agent: Any) -> None:
    """Switch to CODE mode (locked)."""
    _set_mode(console, agent, AgentMode.CODE, locked=True)


def handle_exec_mode(console: Console, agent: Any) -> None:
    """Switch to EXEC mode (locked)."""
    _set_mode(console, agent, AgentMode.EXEC, locked=True)


def handle_auto_mode(console: Console, agent: Any) -> None:
    """Return to AUTO mode (unlocked)."""
    _set_mode(console, agent, AgentMode.AUTO, locked=False)


def _set_mode(
    console: Console,
    agent: Any,
    mode: AgentMode,
    *,
    locked: bool,
) -> None:
    """Set agent mode immediately when possible, with pending-update fallback."""
    mode_update = {
        "current_mode": mode,
        "mode_locked": locked,
        "mode_iteration_count": 0,
    }
    # Persist mode selection across turns so execute() initial-state reset
    # cannot unexpectedly revert user-selected mode.
    agent._session_mode_override = dict(mode_update)
    agent._pending_mode_update = mode_update

    # Best effort: apply mode directly to graph state so next user turn
    # doesn't run with stale mode due to delayed pending-update handling.
    try:
        thread_mgr = getattr(agent, "_thread_manager", None)
        graph = getattr(agent, "graph", None)
        thread_id = getattr(thread_mgr, "thread_id", None)
        if graph is not None and thread_id:
            config = {"configurable": {"thread_id": thread_id}}
            graph.update_state(config, mode_update)
    except Exception:
        # Fallback to pending update only
        pass

    lock_str = " [dim](locked)[/dim]" if locked else ""
    console.print(f"[bold green]✅ Mode set to {mode.value.upper()}{lock_str}[/bold green]")


def _get_current_mode(agent: Any) -> tuple[AgentMode, bool]:
    """Read current mode from agent's last graph state."""
    try:
        thread_mgr = getattr(agent, '_thread_manager', None)
        if thread_mgr:
            config = {"configurable": {"thread_id": thread_mgr.thread_id}}
            snapshot = agent.graph.get_state(config)
            if snapshot and snapshot.values:
                mode = snapshot.values.get("current_mode", AgentMode.AUTO)
                locked = snapshot.values.get("mode_locked", False)
                return mode, locked
    except Exception:
        pass
    return AgentMode.AUTO, False
