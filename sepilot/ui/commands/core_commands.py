"""Core commands: help, exit, status, history, license.

This module follows the Single Responsibility Principle (SRP) by handling
only the core informational and control commands.
"""

import os
import subprocess
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from sepilot.agent.execution_context import (
    is_user_turn_boundary_message,
    is_user_visible_conversation_message,
)


def handle_help(console: Console, show_dynamic: bool = True) -> None:
    """Show help message.

    Args:
        console: Rich console for output
        show_dynamic: If True, show dynamic skills/commands info
    """
    help_text = """
# File References (Claude Code Style!)

**Usage:**
1. Type `@` → File list appears instantly
2. Type letters → Real-time filtering (e.g., `@sep` → `sepilot/`)
3. Type `/` → Enter directory automatically
4. Arrow keys ↑↓ → Select item
5. **First Enter** → Insert file path (NOT execute!)
6. **ESC** → Cancel completion menu, return to normal input
7. Continue typing → Add your instruction
8. **Second Enter** → Execute the command

**Important:**
- When completion menu is open, Enter only selects the file
- Press ESC to cancel the completion menu (Claude Code style)
- You need to press Enter again (after typing your instruction) to execute

**Examples:**
- `@README.md explain this` - Reference and ask
- `@src/utils.py add tests` - Reference and instruct
- `@main.py @config.json compare` - Multiple files

# Reference Types (Claude Code Style!)

- `@file.py` - Single file content
- `@folder/` - Directory listing with file contents
- `@**/*.py` - Glob pattern (all matching files)
- `@url:https://...` - Web page content (text extracted)
- `@git:diff` - Unstaged changes
- `@git:staged` - Staged changes
- `@git:status` - Git status
- `@git:log` - Recent commits (last 20)
- `@git:branch` - List branches
- `@git:show:HASH` - Show specific commit
- `@git:diff:main` - Diff from main branch

# Core Commands

- `/help` - Show this help message
- `/exit`, `/quit` - Exit interactive mode
- `/clearscreen` - Clear screen (Ctrl+K also works)
- `/history` - Show conversations with response summaries (`/history cmd` for commands only)
- `/status` - Show current session status
- `/license` - Show license information (use `/license summary` for quick overview)
- `/resume` - Resume previous conversation threads (type `/resume` to list all)
- `/new` - Start a new conversation (fresh thread, preserves old one for `/resume`)
- `/rewind` - Rewind conversation (`/rewind N` for N exchanges, `/rewind all` to see history)
- `/undo` - Undo last exchange (removes message + file changes, Git-backed)
- `/redo` - Redo undone exchange (restores message + file changes)
- `/reset` - Reset session statistics (command count and token usage)
- `/multiline` - Toggle multi-line input mode
- `/yolo` - Toggle YOLO mode (auto-approve all tool calls)
- `/graph` - Visualize LangGraph workflow as ASCII (use --xray for details)
- `/model` - Dynamic model configuration (type `/model show` for current settings)
- `/rag` - RAG URL management for context retrieval (type `/rag help` for details)
- `/mcp` - MCP server management
- `/tools` - List all available LLM tools (use `/tools list` for compact view)
- `/compact` - Compact conversation context (use `/compact <focus>` for focused summary)
- `/context` - Show context usage visualization grid (Claude Code style)
- `/cost` - Show estimated session cost
- `/clear` - Clear all conversation history (start fresh conversation)

**Auto-Compact (Claude Code style):**
- Context 80%: Warning message shown
- Context 92%: Auto-compact triggered (token-based compression)
- Compresses to 60% of max tokens (leaves room for next interaction)
- Env: `SEPILOT_AUTO_COMPACT=0` to disable
- Env: `SEPILOT_COMPACT_THRESHOLD=0.92` (trigger threshold)
- Env: `SEPILOT_TARGET_RATIO=0.60` (compression target)
- `/skill` - Skills system for specialized capabilities (e.g., `/skill code-review`)
- `/commands` - Custom user commands (e.g., `/commands list` to see available)
- `/theme` - Theme management (e.g., `/theme list`, `/theme set dark`)
- `/stats` - Usage statistics and cost tracking (e.g., `/stats monthly`, `/stats all`)
- `/permissions` - Permission rules management (e.g., `/permissions list`, `/permissions add`)
- `/session` - Session export/import (e.g., `/session export conv.json`, `/session import conv.json`)
- `!<cmd>` - Run a shell command (e.g., `!pytest`)
- `#note` - Save sticky memory, `#list` to view notes
    """
    console.print("[bold cyan]Help[/bold cyan]")
    console.print(Markdown(help_text))

    # Show dynamic skills and commands
    if show_dynamic:
        _show_available_skills(console)
        _show_available_commands(console)

    keyboard_text = """
# Keyboard Shortcuts

- `Ctrl+C` - Cancel current input
- `Ctrl+D` - Exit interactive mode
- `Ctrl+O` - Show last execution result
- `Ctrl+K` - Clear screen
- `Ctrl+X Ctrl+P` - Open slash-command search
- `ESC` - Cancel file completion menu (Claude Code style!)
- `ESC+ESC` - Quick rewind (double-tap within 0.5s)
- `F1` - Show this help message
- `F2` - Show session status
- `Ctrl+R` - Reverse search history (when prompt_toolkit installed)
- `Up/Down` - Navigate command history
- `Alt+Enter` - Submit multi-line input

# Examples

```
🤖 sepilot> @main.py explain what this file does
🤖 sepilot> @src/utils.py @tests/test_utils.py add more unit tests
🤖 sepilot> Create a new Python file called hello.py with a hello world function
🤖 sepilot> Run the tests in the tests/ directory
🤖 sepilot> /status
🤖 sepilot> /exit
```
    """
    console.print(Markdown(keyboard_text))


def _show_available_skills(console: Console) -> None:
    """Show available builtin skills."""
    try:
        from sepilot.skills import get_skill_manager

        skill_manager = get_skill_manager()
        skills = skill_manager.list_skills()

        if skills:
            console.print("\n[bold cyan]Available Skills[/bold cyan]")
            console.print("[dim]Use `/skill <name>` to activate or `/skill list` for details[/dim]\n")

            for metadata in skills:
                trigger_hint = ""
                if metadata.triggers:
                    trigger_hint = f" [dim](triggers: {', '.join(metadata.triggers[:3])})[/dim]"
                console.print(f"  • [green]{metadata.name}[/green] - {metadata.description}{trigger_hint}")

            console.print()
    except Exception:
        pass  # Skills not available


def _show_available_commands(console: Console) -> None:
    """Show available custom commands."""
    try:
        from sepilot.commands import get_command_manager

        command_manager = get_command_manager()
        commands = command_manager.list_commands()

        if commands:
            console.print("[bold cyan]Custom Commands[/bold cyan]")
            console.print("[dim]Use `/<command>` to execute or `/commands list` for details[/dim]\n")

            for cmd in commands:
                source = "[project]" if cmd.is_project else "[user]"
                console.print(f"  • [yellow]/{cmd.name}[/yellow] - {cmd.description} [dim]{source}[/dim]")

            console.print()
    except Exception:
        pass  # Commands not available


def handle_clearscreen(console: Console) -> None:
    """Clear the terminal screen.

    Args:
        console: Rich console for output
    """
    import sys

    def _clear_fallback() -> None:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "cls"], check=False)
        else:
            subprocess.run(["clear"], check=False)

    try:
        if console:
            console.clear()
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
        else:
            _clear_fallback()
    except Exception:
        _clear_fallback()


def handle_status(
    console: Console,
    session_start: datetime,
    command_count: int,
    total_tokens: int,
    memory_count: int,
    memory_file: str,
    history_file: str,
    agent: Any | None = None,
    context_usage_formatter: Any | None = None,
) -> None:
    """Show session status.

    Args:
        console: Rich console for output
        session_start: When session started
        command_count: Number of commands executed
        total_tokens: Total tokens used
        memory_count: Number of memory entries
        memory_file: Path to memory file
        history_file: Path to history file
        agent: Optional agent for additional metrics
        context_usage_formatter: Optional callable to format context usage
    """
    from sepilot.agent.enhanced_state import state_to_summary

    elapsed = datetime.now() - session_start
    elapsed_str = str(elapsed).split('.')[0]

    # Try to get agent metrics
    agent_metrics = None
    if agent and hasattr(agent, "graph"):
        thread_config = getattr(agent, "thread_config", None)
        if thread_config:
            try:
                state = agent.graph.get_state(thread_config)
                if state and getattr(state, "values", None):
                    agent_metrics = state_to_summary(state.values)
            except Exception:
                pass

    token_total = agent_metrics["total_tokens"] if agent_metrics else total_tokens
    token_str = f"{token_total:,}"

    # Check for prompt_toolkit
    try:
        import prompt_toolkit  # noqa: F401
        has_prompt_toolkit = True
    except ImportError:
        has_prompt_toolkit = False

    lines = [
        "[bold cyan]SE Pilot Status[/bold cyan]",
        f"[cyan]Session start:[/cyan] {session_start.strftime('%Y-%m-%d %H:%M:%S')}",
        f"[cyan]Duration:[/cyan] {elapsed_str}",
        f"[cyan]Commands:[/cyan] {command_count}",
        f"[cyan]Total tokens:[/cyan] {token_str}",
        "",
    ]

    # Add context usage if formatter provided
    if context_usage_formatter:
        lines.append(context_usage_formatter(compact=False))
        lines.append("")

    lines.extend([
        f"[cyan]Memory notes:[/cyan] {memory_count} ({memory_file})",
        f"[cyan]History file:[/cyan] {history_file}",
        f"[cyan]prompt_toolkit:[/cyan] {'✅' if has_prompt_toolkit else '❌'}",
    ])

    if agent_metrics:
        cost = agent_metrics.get("estimated_cost", 0.0)
        lines.append(f"[cyan]Estimated cost:[/cyan] ${cost:.4f}")
        lines.append(
            f"[cyan]Iterations:[/cyan] {agent_metrics.get('iteration', 0)}"
            f" · [cyan]Tool calls:[/cyan] {agent_metrics.get('tool_calls_count', 0)}"
        )
        lines.append(
            f"[cyan]Errors:[/cyan] {agent_metrics.get('error_count', 0)}"
            f" · [cyan]File changes:[/cyan] {agent_metrics.get('file_changes_count', 0)}"
        )
        strategy = agent_metrics.get("current_strategy", "")
        if strategy:
            lines.append(f"[cyan]Strategy:[/cyan] {strategy}")

    console.print("\n".join(lines))


def handle_history(
    console: Console,
    history_file: str,
    conversation_context: list,
    input_text: str,
    agent: Any | None = None,
) -> None:
    """Show command/conversation history.

    Args:
        console: Rich console for output
        history_file: Path to command history file
        conversation_context: List of conversation messages
        input_text: Original command input for parsing args
        agent: Optional memory-backed agent for persisted thread history
    """
    args = input_text.strip().split()

    # Parse arguments
    show_cmd_only = False
    limit = 10

    for arg in args:
        if arg == "cmd":
            show_cmd_only = True
        elif arg.isdigit():
            limit = int(arg)

    # Legacy mode: show only commands from history file
    if show_cmd_only:
        if not os.path.exists(history_file):
            console.print("[yellow]No command history yet[/yellow]")
            return

        try:
            with open(history_file) as f:
                lines = f.readlines()
            recent = lines[-20:]
            console.print("[bold cyan]Recent Commands (last 20)[/bold cyan]")
            console.print("\n".join(f"{i+1}. {line.strip()}" for i, line in enumerate(recent)))
        except Exception as e:
            console.print(f"[red]Error reading history: {e}[/red]")
        return

    display_context = list(conversation_context)
    if agent and getattr(agent, "enable_memory", False) is True:
        try:
            if hasattr(agent, "get_conversation_messages"):
                persisted_messages = agent.get_conversation_messages()
                if persisted_messages:
                    display_context = list(persisted_messages)
        except Exception:
            pass

    # New mode: show conversations with response summaries
    if not display_context:
        console.print("[yellow]No conversation history in current session[/yellow]")
        console.print("[dim]Use /history cmd to see command history from file[/dim]")
        return

    # Extract user/assistant pairs
    conversations = _extract_conversations(display_context)

    if not conversations:
        console.print("[yellow]No complete conversations found[/yellow]")
        return

    # Show recent conversations
    recent_convs = conversations[-limit:]
    console.print(f"[bold cyan]Recent Conversations (last {len(recent_convs)})[/bold cyan]\n")

    for i, conv in enumerate(recent_convs, 1):
        user_display = conv['user']
        if len(user_display) > 80:
            user_display = user_display[:77] + "..."

        assistant_content = conv['assistant']
        assistant_lines = [line.strip() for line in assistant_content.split('\n') if line.strip()]
        if assistant_lines:
            assistant_summary = assistant_lines[0]
            if len(assistant_summary) > 150:
                assistant_summary = assistant_summary[:147] + "..."
        else:
            assistant_summary = assistant_content[:150] + "..." if len(assistant_content) > 150 else assistant_content

        console.print(f"[cyan]{i}.[/cyan] [bold white]{user_display}[/bold white]")
        console.print(f"   [dim]→ {assistant_summary}[/dim]")
        console.print()


def _extract_conversations(conversation_context: list) -> list[dict[str, str]]:
    """Extract user/assistant pairs from conversation context."""
    system_markers = ['# User Memory', '# System', '# Context', '# Instructions', '# Settings']

    def extract_user_query(content: str) -> str:
        if not content:
            return ""

        content = content.strip()
        has_system_section = any(content.startswith(marker) for marker in system_markers)

        if has_system_section:
            paragraphs = content.split('\n\n')
            for para in reversed(paragraphs):
                para = para.strip()
                if para and not any(para.startswith(marker) for marker in system_markers):
                    if len(para) < 500 and not para.startswith('-'):
                        return para
            return ""

        return content

    conversations = []
    current_user_msg = None

    for msg in conversation_context:
        role = getattr(msg, 'type', None) or getattr(msg, 'role', None)
        content = getattr(msg, 'content', str(msg))

        if role in ('human', 'user'):
            if role == 'human' and not is_user_turn_boundary_message(msg):
                continue
            user_query = extract_user_query(content)
            current_user_msg = user_query or None
        elif role in ('ai', 'assistant') and current_user_msg:
            if not is_user_visible_conversation_message(msg):
                continue
            conversations.append({
                'user': current_user_msg,
                'assistant': content
            })
            current_user_msg = None

    return conversations


def handle_license(console: Console, input_text: str) -> None:
    """Show license information.

    Args:
        console: Rich console for output
        input_text: Original command for parsing args
    """
    # Find LICENSE.md file
    possible_paths = [
        os.path.join(os.getcwd(), "LICENSE.md"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "LICENSE.md"),
        os.path.expanduser("~/.sepilot/LICENSE.md"),
    ]

    license_file = None
    for path in possible_paths:
        if os.path.exists(path):
            license_file = path
            break

    if not license_file:
        _show_license_fallback(console, possible_paths)
        return

    try:
        with open(license_file, encoding='utf-8') as f:
            license_content = f.read()

        args = input_text.strip().split()
        if len(args) > 1 and args[1] in ['--summary', '-s', 'summary']:
            _show_license_summary(console)
        else:
            _show_license_full(console, license_file, license_content)

    except Exception as e:
        console.print(f"[red]❌ Error reading LICENSE.md: {e}[/red]")


def _show_license_fallback(console: Console, paths: list[str]) -> None:
    """Show fallback license info when file not found."""
    console.print("[yellow]⚠️  LICENSE.md file not found[/yellow]")
    console.print("[dim]Searched in:[/dim]")
    for path in paths:
        console.print(f"[dim]  - {path}[/dim]")

    console.print("\n[bold cyan]SEPilot3 License Summary:[/bold cyan]")
    console.print("[yellow]Source-Available Proprietary License[/yellow]")
    console.print("• ✅ Free for personal, educational, and non-profit use")
    console.print("• ⚠️  Commercial use requires a commercial license")
    console.print("• ❌ Modification and redistribution prohibited")
    console.print("• ✅ Pull requests and contributions welcome")
    console.print("\n[dim]For full license text, see LICENSE.md in the project repository[/dim]")


def _show_license_summary(console: Console) -> None:
    """Show license summary."""
    summary_text = """
**License Type:** Source-Available Proprietary License v1.0

**Quick Reference:**

| Activity | Individual/Non-Profit | Commercial Entity |
|----------|----------------------|-------------------|
| View source code | ✅ Allowed | ✅ Allowed |
| Use the software | ✅ Free | ⚠️ License required |
| Modify source code | ❌ Prohibited | ❌ Prohibited |
| Redistribute | ❌ Prohibited | ❌ Prohibited |
| Submit Pull Requests | ✅ Allowed | ✅ Allowed |
| Commercial use | N/A | ⚠️ License required |

**Key Points:**
• Free for personal, educational, research, and non-profit use
• Commercial entities require a commercial license
• Source code visible but not modifiable or redistributable
• Contributions via Pull Requests are welcome
• Based on permissive open-source dependencies (MIT, BSD, Apache 2.0)

**For Full License:**
Type `/license full` or see LICENSE.md in the project directory

**Contact:** See LICENSE.md for licensing inquiries
    """
    console.print("[bold cyan]📜 SEPilot3 License Summary[/bold cyan]\n")
    console.print(Markdown(summary_text))


def _show_license_full(console: Console, license_file: str, content: str) -> None:
    """Show full license."""
    console.print("[bold cyan]📜 SEPilot3 License[/bold cyan]")
    console.print(f"[dim]Reading from: {license_file}[/dim]\n")
    console.print(Markdown(content))
    console.print(f"\n[dim]Full license text: {license_file}[/dim]")
    console.print("[dim]💡 Tip: Use '/license summary' for a quick overview[/dim]")


def handle_reset(
    console: Console,
    session_start: datetime,
    command_count: int,
    total_tokens: int,
) -> tuple[datetime, int, int]:
    """Reset session statistics.

    Args:
        console: Rich console for output
        session_start: Current session start time
        command_count: Current command count
        total_tokens: Current token count

    Returns:
        Tuple of (new_session_start, new_command_count, new_total_tokens)
    """
    old_duration = datetime.now() - session_start

    console.print("[bold cyan]🔄 Session Reset[/bold cyan]")
    console.print("[dim]Previous session:[/dim]")
    console.print(f"  [cyan]Duration:[/cyan] {str(old_duration).split('.')[0]}")
    console.print(f"  [cyan]Commands:[/cyan] {command_count}")
    console.print(f"  [cyan]Total tokens:[/cyan] {total_tokens:,}")
    console.print("\n[green]✓ Session statistics have been reset[/green]")

    return datetime.now(), 0, 0
