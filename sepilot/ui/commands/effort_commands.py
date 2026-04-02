"""Effort level slash commands for Interactive Mode.

Commands:
  /effort       — Show current effort level
  /effort fast  — Switch to fast mode (5-node ReAct loop, small model friendly)
  /effort high  — Switch to high mode (full 17-node enhanced pipeline)
"""

from __future__ import annotations

from typing import Any

from rich.console import Console


EFFORT_LEVELS = {
    "fast": {
        "graph_mode": "fast",
        "label": "⚡ FAST",
        "description": "5-node ReAct loop — 소형 모델 (qwen3:8b 등) 최적화",
        "nodes": 5,
    },
    "high": {
        "graph_mode": "enhanced",
        "label": "🧠 HIGH",
        "description": "17-node 풀 파이프라인 — 계획·검증·반성·토론 포함",
        "nodes": 17,
    },
}


def handle_effort_command(
    console: Console,
    agent: Any,
    input_text: str = "",
) -> None:
    """Show or change effort level.

    Args:
        console: Rich console for output
        agent: ReactAgent instance
        input_text: Raw input after /effort (e.g., "fast", "high", or "")
    """
    arg = input_text.strip().lower()

    if not arg:
        _show_current_effort(console, agent)
        return

    if arg not in EFFORT_LEVELS:
        console.print(f"[yellow]⚠️  Unknown effort level: '{arg}'[/yellow]")
        console.print("[dim]사용법: /effort fast | /effort high[/dim]")
        return

    _set_effort(console, agent, arg)


def _show_current_effort(console: Console, agent: Any) -> None:
    """Display current effort level."""
    current_mode = getattr(getattr(agent, "settings", None), "graph_mode", "enhanced")

    console.print("\n[bold cyan]⚙️  Effort Level[/bold cyan]\n")

    for key, info in EFFORT_LEVELS.items():
        is_current = info["graph_mode"] == current_mode
        marker = " [bold green]← current[/bold green]" if is_current else ""
        console.print(
            f"  {info['label']}  ({info['nodes']} nodes) — {info['description']}{marker}"
        )

    console.print("\n[dim]변경: /effort fast | /effort high[/dim]\n")


def _set_effort(console: Console, agent: Any, level: str) -> None:
    """Change effort level by rebuilding the graph at runtime."""
    info = EFFORT_LEVELS[level]
    new_graph_mode = info["graph_mode"]

    settings = getattr(agent, "settings", None)
    if not settings:
        console.print("[red]❌ Agent settings not available[/red]")
        return

    old_graph_mode = getattr(settings, "graph_mode", "enhanced")
    if old_graph_mode == new_graph_mode:
        console.print(f"[dim]이미 {info['label']} 모드입니다.[/dim]")
        return

    # Update settings
    settings.graph_mode = new_graph_mode

    # Re-initialize patterns if switching to/from fast
    if new_graph_mode == "fast":
        agent._init_stub_patterns()
    elif old_graph_mode == "fast":
        agent._init_agent_patterns()

    # Rebuild graph
    agent.graph = agent._build_graph()

    # Update thread manager's graph reference
    thread_mgr = getattr(agent, "_thread_manager", None)
    if thread_mgr:
        thread_mgr.graph = agent.graph

    console.print(
        f"[bold green]✅ Effort: {info['label']} ({info['nodes']} nodes)[/bold green]"
    )
