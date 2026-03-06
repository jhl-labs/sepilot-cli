"""Session management commands: resume, new, rewind.

This module follows the Single Responsibility Principle (SRP) by handling
only session and thread management commands.

Rewind command supports:
- /rewind                    Interactive mode
- /rewind N                  Rewind N exchanges
- /rewind --list             List all checkpoints
- /rewind --to <checkpoint>  Rewind to specific checkpoint
- /rewind --diff <cp1> <cp2> Compare two checkpoints
- /rewind --preview <cp>     Preview checkpoint before restoring
- /rewind --files-only       Only rewind files, keep conversation
- /rewind --env              Also restore environment snapshot
"""

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def handle_resume(
    console: Console,
    agent: Any | None,
    input_text: str,
    conversation_context: list | None = None,
) -> bool:
    """Resume a previous conversation thread.

    Args:
        console: Rich console for output
        agent: Agent with thread management capabilities
        input_text: Original command input for parsing args
        conversation_context: Conversation context to sync with resumed thread

    Returns:
        True if resume successful
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /resume command disabled[/yellow]")
        return False

    # Verify agent has required methods
    required_methods = ['list_available_threads', 'get_thread_id', 'switch_thread']
    missing_methods = [m for m in required_methods if not hasattr(agent, m)]
    if missing_methods:
        console.print(f"[yellow]⚠️  Agent missing required methods: {', '.join(missing_methods)}[/yellow]")
        console.print("[dim]Thread management may not be supported by this agent type[/dim]")
        return False

    # Parse arguments
    args = input_text.strip().split(maxsplit=1)
    thread_arg = args[1] if len(args) > 1 else None

    if not thread_arg:
        _show_available_threads(console, agent)
        return False

    # Support numeric index (e.g. /resume 1)
    thread_id = thread_arg
    if thread_arg.isdigit():
        idx = int(thread_arg)
        threads = agent.list_available_threads()
        if 1 <= idx <= len(threads):
            thread_id = threads[idx - 1].get("thread_id", thread_arg)
        else:
            console.print(f"[yellow]Invalid index: {idx}. Use /resume to see available threads.[/yellow]")
            return False

    # Check if already on this thread
    current_thread = agent.get_thread_id()
    if thread_id == current_thread:
        console.print(f"[yellow]Already on this thread: {thread_id}[/yellow]")
        return True

    return _switch_to_thread(console, agent, thread_id, conversation_context)


def _show_available_threads(console: Console, agent: Any) -> None:
    """Show list of available threads with conversation topic."""
    console.print("[bold cyan]📋 Available Conversation Threads[/bold cyan]\n")

    threads = agent.list_available_threads()

    if not threads:
        console.print("[yellow]No previous threads found[/yellow]")
        console.print("[dim]Start a new conversation and it will be saved automatically[/dim]")
        return

    current_thread_id = agent.get_thread_id()

    for i, thread in enumerate(threads, 1):
        thread_id = thread.get("thread_id", "Unknown")
        is_current = thread_id == current_thread_id

        # Timestamp
        ts = _format_timestamp(thread.get("updated_at", "") or thread.get("created_at", ""))

        # Message count
        msg_count = thread.get("message_count", 0)

        # Topic = first user message
        topic = thread.get("first_message_preview", "") or thread.get("last_message_preview", "") or "No messages"

        # Format display
        marker = "[bold green]►[/bold green] " if is_current else "  "
        current_tag = " [green](current)[/green]" if is_current else ""
        idx_str = f"[cyan]{i}.[/cyan]"

        console.print(f"{marker}{idx_str} {topic}{current_tag}")
        console.print(f"      [dim]{thread_id}  ·  {ts}  ·  {msg_count} msgs[/dim]")

    console.print()
    console.print("[dim]💡 /resume <thread-id> 또는 /resume <번호> 로 이전 대화를 이어갈 수 있습니다[/dim]")
    console.print(f"[dim]💡 현재 스레드: {current_thread_id}[/dim]")


def _format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    if not ts:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16] if ts else "Unknown"


def _switch_to_thread(
    console: Console,
    agent: Any,
    thread_id: str,
    conversation_context: list | None = None,
) -> bool:
    """Switch to a specific thread and sync conversation context."""
    console.print(f"[cyan]Switching to thread: {thread_id}[/cyan]")

    success = agent.switch_thread(thread_id)

    if success:
        console.print(f"[bold green]✓ Successfully resumed thread: {thread_id}[/bold green]")

        # Sync conversation_context with LangGraph messages
        if conversation_context is not None and hasattr(agent, 'get_conversation_messages'):
            messages = agent.get_conversation_messages()
            conversation_context.clear()
            for msg in messages:
                msg_type = getattr(msg, 'type', '')
                if msg_type in ('human', 'ai'):
                    conversation_context.append(msg)
            if conversation_context:
                console.print(f"[dim]Restored {len(conversation_context)} messages to context[/dim]")

        # Show session summary
        if hasattr(agent, 'get_session_summary'):
            summary = agent.get_session_summary()
            console.print(f"[dim]{summary}[/dim]")

        # Show last few messages
        _show_recent_messages(console, agent, thread_id)
        return True

    console.print(f"[bold red]✗ Failed to resume thread: {thread_id}[/bold red]")
    console.print("[dim]Use /resume to see available threads[/dim]")
    return False


def _show_recent_messages(console: Console, agent: Any, thread_id: str, max_exchanges: int = 5) -> None:
    """Show recent human/ai message exchanges from a thread."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.graph.get_state(config)

        if state and state.values:
            messages = state.values.get("messages", [])

            # Filter to human/ai messages only
            conversation = [
                msg for msg in messages
                if getattr(msg, 'type', '') in ('human', 'ai')
            ]

            if conversation:
                # Take last N exchanges (each exchange = human + ai pair)
                # Count human messages from the end to find the cutoff
                human_count = 0
                cutoff = len(conversation)
                for i in range(len(conversation) - 1, -1, -1):
                    if getattr(conversation[i], 'type', '') == 'human':
                        human_count += 1
                        if human_count > max_exchanges:
                            cutoff = i + 1
                            break

                recent = conversation[cutoff:]

                console.print("\n[bold]Recent conversation:[/bold]")
                for msg in recent:
                    role = "User" if getattr(msg, 'type', '') == "human" else "Assistant"
                    content = getattr(msg, 'content', str(msg))

                    if len(content) > 200:
                        content = content[:200] + "..."

                    console.print(f"[cyan]{role}:[/cyan] {content}")

                console.print()
                console.print("[dim]You can continue the conversation from here[/dim]")
    except Exception:
        pass


def handle_new(
    console: Console,
    agent: Any | None,
    conversation_context: list,
) -> tuple[str | None, int, int]:
    """Start a new conversation thread.

    Args:
        console: Rich console for output
        agent: Agent with thread management
        conversation_context: Current conversation context to clear

    Returns:
        Tuple of (new_thread_id, reset_command_count, reset_total_tokens)
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /new command disabled[/yellow]")
        return None, 0, 0

    # Verify agent has required methods
    required_methods = ['get_thread_id', 'create_new_thread']
    missing_methods = [m for m in required_methods if not hasattr(agent, m)]
    if missing_methods:
        console.print(f"[yellow]⚠️  Agent missing required methods: {', '.join(missing_methods)}[/yellow]")
        return None, 0, 0

    # Get old thread info
    old_thread_id = agent.get_thread_id()
    old_msg_count = len(conversation_context)

    # Create new thread
    try:
        new_thread_id = agent.create_new_thread()
    except Exception as e:
        console.print(f"[red]✗ Failed to create new thread: {e}[/red]")
        return None, 0, 0

    # Clear conversation context
    conversation_context.clear()

    # Display confirmation
    console.print()
    console.print("[bold green]✨ New conversation started![/bold green]")
    console.print()
    console.print(f"[dim]Previous thread: {old_thread_id} ({old_msg_count} messages)[/dim]")
    console.print(f"[cyan]New thread: {new_thread_id}[/cyan]")
    console.print()
    console.print("[dim]💡 Use /resume to return to previous conversations[/dim]")

    return new_thread_id, 0, 0


def handle_rewind(
    console: Console,
    agent: Any | None,
    conversation_context: list,
    input_text: str,
) -> dict[str, Any]:
    """Rewind conversation and/or code changes.

    Supports enhanced Claude Code-style options:
    - /rewind                    Interactive mode
    - /rewind N                  Rewind N exchanges
    - /rewind --list             List all checkpoints
    - /rewind --to <checkpoint>  Rewind to specific checkpoint
    - /rewind --diff <cp1> <cp2> Compare two checkpoints
    - /rewind --preview <cp>     Preview checkpoint before restoring
    - /rewind --files-only       Only rewind files, keep conversation
    - /rewind --env              Also restore environment snapshot

    Args:
        console: Rich console for output
        agent: Agent with rewind capabilities
        conversation_context: Current conversation context
        input_text: Original command for parsing args

    Returns:
        Dictionary with results: {conv_result, code_result}
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /rewind command disabled[/yellow]")
        return {}

    # Parse arguments
    args = input_text.strip().split()
    count = 1
    mode = "interactive"
    restore_env = False
    target_checkpoint = None

    # Handle new --options
    if len(args) > 1:
        arg = args[1].lower()

        # --list: Show all checkpoints with enhanced info
        if arg in ("--list", "-l"):
            _show_enhanced_checkpoints(console)
            return {}

        # --to <checkpoint>: Rewind to specific checkpoint
        elif arg in ("--to", "-t"):
            if len(args) < 3:
                console.print("[yellow]Usage: /rewind --to <checkpoint_id>[/yellow]")
                return {}
            target_checkpoint = args[2]
            mode = "to_checkpoint"

        # --diff <cp1> <cp2>: Compare two checkpoints
        elif arg in ("--diff", "-d"):
            if len(args) < 4:
                console.print("[yellow]Usage: /rewind --diff <checkpoint1> <checkpoint2>[/yellow]")
                return {}
            return _show_checkpoint_diff(console, args[2], args[3])

        # --preview <checkpoint>: Preview before restoring
        elif arg in ("--preview", "-p"):
            if len(args) < 3:
                console.print("[yellow]Usage: /rewind --preview <checkpoint_id>[/yellow]")
                return {}
            return _preview_checkpoint(console, args[2])

        # --files-only: Only rewind files
        elif arg in ("--files-only", "--files", "-f"):
            mode = "code"
            if len(args) > 2 and args[2].isdigit():
                count = int(args[2])

        # --env: Also restore environment
        elif arg in ("--env", "-e"):
            restore_env = True
            mode = "both"
            if len(args) > 2 and args[2].isdigit():
                count = int(args[2])

        # Original options
        elif arg in ("all", "list"):
            _show_rewind_options(console, agent)
            return {}
        elif arg in ("conv", "conversation"):
            mode = "conv"
        elif arg in ("code", "files"):
            mode = "code"
        elif arg == "both":
            mode = "both"
        elif arg.isdigit():
            count = int(arg)
            mode = "both"
        else:
            console.print(f"[yellow]Invalid argument: {arg}[/yellow]")
            _show_rewind_help(console)
            return {}

    if len(args) > 2 and args[2].isdigit() and mode not in ("to_checkpoint",):
        count = int(args[2])

    # Verify agent has required methods for conversation rewind
    if mode in ("conv", "both", "interactive"):
        required_methods = ['get_conversation_messages', 'rewind_messages']
        missing_methods = [m for m in required_methods if not hasattr(agent, m)]
        if missing_methods:
            console.print(f"[dim]Note: Conversation rewind unavailable (missing: {', '.join(missing_methods)})[/dim]")
            if mode == "conv":
                return {}
            mode = "code"

    # Handle to_checkpoint mode
    if mode == "to_checkpoint":
        return _rewind_to_checkpoint(console, target_checkpoint, restore_env)

    # Interactive mode: Claude Code style — select message, then action
    if mode == "interactive":
        return _interactive_rewind(console, agent, conversation_context)

    # Execute rewind
    results: dict[str, Any] = {}

    if mode in ("conv", "both"):
        results['conv_result'] = _rewind_conversation(agent, conversation_context, count)

    if mode in ("code", "both"):
        results['code_result'] = _rewind_code(count, restore_env)

    # Display results
    _display_rewind_results(console, results)

    return results


def _interactive_rewind(
    console: Console,
    agent: Any,
    conversation_context: list,
) -> dict[str, Any]:
    """Claude Code style interactive rewind.

    Step 1: Show conversation exchanges with file change summaries
    Step 2: User selects which exchange to rewind to
    Step 3: User selects restore action (code+conv / conv only / code only / cancel)
    """
    messages = agent.get_conversation_messages()
    if not messages:
        console.print("[yellow]No messages to rewind[/yellow]")
        return {}

    # Group messages into user-assistant exchanges
    exchanges = _build_exchange_list(messages)
    if not exchanges:
        console.print("[yellow]No exchanges to rewind[/yellow]")
        return {}

    # Step 1: Show exchanges
    console.print()
    console.print('[bold cyan]⏪ Restore conversation to the point before…[/bold cyan]')
    console.print()

    for i, ex in enumerate(exchanges, 1):
        user_text = ex["user_text"]
        if len(user_text) > 70:
            user_text = user_text[:67] + "..."
        user_text = user_text.replace('\n', ' ').strip()

        # File change summary (if available from tool calls)
        file_info = ""
        if ex["file_changes"]:
            parts = []
            for fc in ex["file_changes"][:3]:
                parts.append(f"[dim]{fc}[/dim]")
            file_info = "  " + " ".join(parts)
            if len(ex["file_changes"]) > 3:
                file_info += f" [dim]+{len(ex['file_changes']) - 3} more[/dim]"

        console.print(f"  [cyan]{i}.[/cyan] {user_text}{file_info}")

    console.print(f"  [dim]{len(exchanges) + 1}.[/dim] [dim]Cancel[/dim]")
    console.print()

    # Step 2: Select exchange
    try:
        choice = input(f"Select (1-{len(exchanges) + 1}): ").strip()
        if not choice.isdigit():
            console.print("[dim]Cancelled[/dim]")
            return {}
        idx = int(choice)
        if idx < 1 or idx > len(exchanges):
            console.print("[dim]Cancelled[/dim]")
            return {}
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled[/dim]")
        return {}

    selected_count = len(exchanges) - idx + 1  # How many exchanges to remove from the end

    # Step 3: Select restore action
    console.print()
    console.print(f"[bold]Rewind {selected_count} exchange(s):[/bold]")
    console.print()
    console.print("  [cyan]1.[/cyan] Restore code and conversation")
    console.print("  [cyan]2.[/cyan] Restore conversation only")
    console.print("  [cyan]3.[/cyan] Restore code only")
    console.print("  [cyan]4.[/cyan] Cancel")
    console.print()

    try:
        action = input("Select action (1-4): ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled[/dim]")
        return {}

    results: dict[str, Any] = {}

    if action == "1":
        results['conv_result'] = _rewind_conversation(agent, conversation_context, selected_count)
        results['code_result'] = _rewind_code(selected_count)
    elif action == "2":
        results['conv_result'] = _rewind_conversation(agent, conversation_context, selected_count)
    elif action == "3":
        results['code_result'] = _rewind_code(selected_count)
    else:
        console.print("[dim]Cancelled[/dim]")
        return {}

    _display_rewind_results(console, results)
    return results


def _build_exchange_list(messages: list) -> list[dict[str, Any]]:
    """Build a list of user-assistant exchanges from messages.

    Each exchange = { user_text, assistant_text, file_changes, pair_index }
    """
    exchanges = []
    i = 0
    pair_idx = 0

    while i < len(messages):
        msg = messages[i]
        msg_type = getattr(msg, 'type', '')

        if msg_type == 'human':
            user_text = getattr(msg, 'content', str(msg))
            assistant_text = ""
            file_changes = []

            # Collect subsequent non-human messages
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                next_type = getattr(next_msg, 'type', '')
                if next_type == 'human':
                    break
                if next_type == 'ai':
                    assistant_text = getattr(next_msg, 'content', '')
                    # Check for tool calls that modified files
                    tool_calls = getattr(next_msg, 'tool_calls', [])
                    for tc in (tool_calls or []):
                        name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                        args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                        if name in ('file_write', 'file_edit', 'bash_execute', 'git'):
                            path = args.get('path', args.get('file_path', ''))
                            if path:
                                fname = path.split('/')[-1] if '/' in path else path
                                if fname and fname not in file_changes:
                                    file_changes.append(fname)
                j += 1

            exchanges.append({
                "user_text": user_text,
                "assistant_text": assistant_text,
                "file_changes": file_changes,
                "pair_index": pair_idx,
            })
            pair_idx += 1
            i = j
        else:
            i += 1

    return exchanges


def _rewind_conversation(agent: Any, conversation_context: list, count: int) -> dict[str, Any]:
    """Rewind conversation messages."""
    messages_before = agent.get_conversation_messages()

    if not messages_before:
        return {"success": False, "error": "No messages to rewind"}

    result = agent.rewind_messages(count)

    if result.get("success"):
        pairs_to_remove = result.get("pairs_removed", count)
        msgs_to_remove = pairs_to_remove * 2

        if msgs_to_remove <= len(conversation_context):
            conversation_context[:] = conversation_context[:-msgs_to_remove]

    return result


def _rewind_code(count: int, restore_env: bool = False) -> dict[str, Any]:
    """Rewind code/file changes.

    Args:
        count: Number of checkpoints to rewind
        restore_env: Whether to also restore environment snapshot

    Returns:
        Result dictionary
    """
    try:
        from sepilot.memory.file_checkpoint import FileCheckpointManager
        manager = FileCheckpointManager()
    except ImportError:
        return {"success": False, "no_checkpoints": True}

    if not manager.checkpoints:
        return {"success": False, "no_checkpoints": True}

    result = manager.revert_by_count(count)

    if result is None:
        return {"success": False, "error": "Not enough checkpoints"}

    files_reverted = sum(1 for v in result.values() if v in ("restored", "deleted"))

    response = {
        "success": True,
        "files_reverted": files_reverted,
        "details": result
    }

    # Restore environment if requested
    if restore_env:
        env_result = _restore_environment_snapshot(count)
        response["env_result"] = env_result

    return response


def _restore_environment_snapshot(count: int) -> dict[str, Any]:
    """Restore environment snapshot from checkpoint.

    Args:
        count: Checkpoint index (from end)

    Returns:
        Result dictionary
    """
    try:
        from sepilot.memory.environment_snapshot import get_environment_manager
        from sepilot.memory.file_checkpoint import FileCheckpointManager

        manager = FileCheckpointManager()
        if not manager.checkpoints or len(manager.checkpoints) < count + 1:
            return {"success": False, "error": "No environment snapshot found"}

        checkpoint = manager.checkpoints[-(count + 1)]
        env_manager = get_environment_manager()

        # Get commands to restore environment
        # Note: We can't fully restore env in-process, just return commands
        return {
            "success": True,
            "checkpoint_id": checkpoint.id,
            "note": "Environment snapshot recorded but cannot be auto-restored in-process"
        }
    except ImportError:
        return {"success": False, "error": "Environment snapshot module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _display_rewind_results(console: Console, results: dict[str, Any]) -> None:
    """Display rewind results."""
    console.print()

    conv_result = results.get('conv_result')
    code_result = results.get('code_result')

    if conv_result and conv_result.get("success"):
        pairs = conv_result.get("pairs_removed", 0)
        remaining = conv_result.get("remaining_count", 0)
        console.print(f"[green]✓ Conversation:[/green] Rewound {pairs} exchange(s), {remaining} remaining")

    if code_result:
        if code_result.get("success"):
            files_reverted = code_result.get("files_reverted", 0)
            console.print(f"[green]✓ Code:[/green] Reverted {files_reverted} file(s)")
        elif code_result.get("no_checkpoints"):
            console.print("[dim]✓ Code: No file checkpoints to revert[/dim]")

    console.print()
    console.print("[dim]💡 Use /rewind again to go further back[/dim]")


def _show_rewind_options(console: Console, agent: Any) -> None:
    """Show conversation messages for selective rewind."""
    messages = agent.get_conversation_messages()

    if not messages:
        console.print("[yellow]No messages in conversation[/yellow]")
        return

    console.print()
    console.print("[bold cyan]📜 Conversation History[/bold cyan]")
    console.print("[dim]Messages are shown oldest to newest[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Type", width=10)
    table.add_column("Content", max_width=70)

    for i, msg in enumerate(messages):
        msg_type = getattr(msg, 'type', 'unknown')
        content = getattr(msg, 'content', str(msg))

        if msg_type == 'human':
            type_display = "[cyan]User[/cyan]"
        elif msg_type == 'ai':
            type_display = "[green]Assistant[/green]"
        elif msg_type == 'tool':
            type_display = "[yellow]Tool[/yellow]"
        else:
            type_display = f"[dim]{msg_type}[/dim]"

        if len(content) > 100:
            content = content[:97] + "..."
        content = content.replace('\n', ' ').strip()

        table.add_row(str(i + 1), type_display, content)

    console.print(table)
    console.print()

    human_count = sum(1 for m in messages if getattr(m, 'type', '') == 'human')
    console.print(f"[dim]Total: {len(messages)} messages, {human_count} exchanges[/dim]")
    console.print()
    console.print("[dim]💡 Use /rewind N to remove the last N exchanges[/dim]")
    console.print("[dim]💡 Use /rewind to remove just the last exchange[/dim]")


def handle_multiline(console: Console, session: Any | None) -> bool:
    """Toggle multi-line input mode.

    Args:
        console: Rich console for output
        session: prompt_toolkit session

    Returns:
        New multiline state
    """
    if session is None:
        console.print("[yellow]Multi-line mode requires prompt_toolkit[/yellow]")
        return False

    current = session.multiline
    session.multiline = not current

    if session.multiline:
        console.print("[green]Multi-line mode enabled[/green]")
        console.print("[dim]Press Alt+Enter or Meta+Enter to submit[/dim]")
    else:
        console.print("[yellow]Multi-line mode disabled[/yellow]")
        console.print("[dim]Press Enter to submit[/dim]")

    return session.multiline


def handle_yolo(console: Console, agent: Any | None) -> bool:
    """Toggle YOLO mode (auto-approve all tool calls).

    Args:
        console: Rich console for output
        agent: Agent with auto_approve setting

    Returns:
        New auto_approve state
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /yolo command disabled[/yellow]")
        return False

    current = agent.auto_approve
    agent.auto_approve = not current

    if agent.auto_approve:
        console.print("[bold red]🚀 YOLO MODE ENABLED![/bold red]")
        console.print("[yellow]⚠️  All tool executions will be auto-approved[/yellow]")
        console.print("[dim]Agent will run without human-in-the-loop approval[/dim]")
        console.print("[dim]Type /yolo again to disable[/dim]")
    else:
        console.print("[green]✓ YOLO MODE DISABLED[/green]")
        console.print("[dim]Human-in-the-loop approval restored[/dim]")

    return agent.auto_approve


# ============================================================================
# Enhanced Rewind Helper Functions (Claude Code style)
# ============================================================================

def _show_rewind_help(console: Console) -> None:
    """Show help for rewind command options."""
    console.print()
    console.print("[bold cyan]⏪ Rewind Command Options[/bold cyan]")
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan", width=30)
    table.add_column("Description")

    table.add_row("/rewind", "Interactive mode - choose what to rewind")
    table.add_row("/rewind N", "Rewind N exchanges (conversation + files)")
    table.add_row("/rewind --list", "List all available checkpoints")
    table.add_row("/rewind --to <cp>", "Rewind to specific checkpoint")
    table.add_row("/rewind --diff <cp1> <cp2>", "Compare two checkpoints")
    table.add_row("/rewind --preview <cp>", "Preview checkpoint before restoring")
    table.add_row("/rewind --files-only", "Only rewind files, keep conversation")
    table.add_row("/rewind --env", "Also restore environment snapshot")
    table.add_row("/rewind conv", "Rewind conversation only")
    table.add_row("/rewind code", "Rewind code/files only")
    table.add_row("/rewind both", "Rewind both conversation and files")
    table.add_row("/rewind all", "Show conversation history")

    console.print(table)
    console.print()


def _show_enhanced_checkpoints(console: Console) -> None:
    """Show all checkpoints with enhanced information."""
    console.print()
    console.print("[bold cyan]📋 Available Checkpoints[/bold cyan]")
    console.print()

    # Try to load from both file_checkpoint and backtracking
    checkpoints = []

    try:
        from sepilot.memory.file_checkpoint import FileCheckpointManager
        manager = FileCheckpointManager()
        for cp in manager.checkpoints:
            checkpoints.append({
                "id": cp.id,
                "timestamp": cp.timestamp,
                "type": "file",
                "files": len(cp.files),
                "prompt": cp.user_prompt[:50] if cp.user_prompt else "",
            })
    except ImportError:
        pass

    # Try project history
    try:
        import os
        from sepilot.memory.project_history import get_project_history_manager

        pm = get_project_history_manager()
        project_path = os.getcwd()
        sessions = pm.get_project_sessions(project_path)

        if sessions:
            console.print(f"[dim]Project sessions: {len(sessions)}[/dim]")
    except ImportError:
        pass

    if not checkpoints:
        console.print("[yellow]No checkpoints found[/yellow]")
        console.print("[dim]Checkpoints are created automatically during agent operations[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Checkpoint ID", style="cyan", width=25)
    table.add_column("Timestamp", style="dim", width=20)
    table.add_column("Files", justify="right", width=6)
    table.add_column("Prompt Preview", max_width=40)

    for i, cp in enumerate(reversed(checkpoints)):
        table.add_row(
            str(len(checkpoints) - i),
            cp["id"],
            _format_timestamp(cp["timestamp"]),
            str(cp["files"]),
            cp["prompt"]
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Total: {len(checkpoints)} checkpoints[/dim]")
    console.print()
    console.print("[dim]💡 Use /rewind --to <checkpoint_id> to rewind to specific point[/dim]")
    console.print("[dim]💡 Use /rewind --preview <checkpoint_id> to see details[/dim]")


def _show_checkpoint_diff(console: Console, cp1_id: str, cp2_id: str) -> dict[str, Any]:
    """Show diff between two checkpoints.

    Args:
        console: Rich console
        cp1_id: First checkpoint ID
        cp2_id: Second checkpoint ID

    Returns:
        Result dictionary
    """
    console.print()
    console.print(f"[bold cyan]📊 Checkpoint Diff: {cp1_id} vs {cp2_id}[/bold cyan]")
    console.print()

    try:
        from sepilot.memory.file_checkpoint import FileCheckpointManager
        manager = FileCheckpointManager()

        cp1 = None
        cp2 = None

        for cp in manager.checkpoints:
            if cp.id == cp1_id:
                cp1 = cp
            if cp.id == cp2_id:
                cp2 = cp

        if not cp1:
            console.print(f"[red]Checkpoint not found: {cp1_id}[/red]")
            return {"success": False, "error": f"Checkpoint not found: {cp1_id}"}

        if not cp2:
            console.print(f"[red]Checkpoint not found: {cp2_id}[/red]")
            return {"success": False, "error": f"Checkpoint not found: {cp2_id}"}

        # Compare files
        files1 = set(cp1.files.keys())
        files2 = set(cp2.files.keys())

        added = files2 - files1
        removed = files1 - files2
        common = files1 & files2

        modified = []
        for f in common:
            if cp1.files[f].hash != cp2.files[f].hash:
                modified.append(f)

        # Display results
        if added:
            console.print("[green]Added files:[/green]")
            for f in added:
                console.print(f"  [green]+ {f}[/green]")

        if removed:
            console.print("[red]Removed files:[/red]")
            for f in removed:
                console.print(f"  [red]- {f}[/red]")

        if modified:
            console.print("[yellow]Modified files:[/yellow]")
            for f in modified:
                console.print(f"  [yellow]~ {f}[/yellow]")

        if not (added or removed or modified):
            console.print("[dim]No differences between checkpoints[/dim]")

        console.print()
        console.print(f"[dim]Summary: +{len(added)} -{len(removed)} ~{len(modified)}[/dim]")

        return {
            "success": True,
            "added": list(added),
            "removed": list(removed),
            "modified": modified
        }

    except ImportError:
        console.print("[yellow]Checkpoint manager not available[/yellow]")
        return {"success": False, "error": "Checkpoint manager not available"}


def _preview_checkpoint(console: Console, checkpoint_id: str) -> dict[str, Any]:
    """Preview a checkpoint before restoring.

    Args:
        console: Rich console
        checkpoint_id: Checkpoint ID to preview

    Returns:
        Result dictionary
    """
    console.print()
    console.print(f"[bold cyan]🔍 Checkpoint Preview: {checkpoint_id}[/bold cyan]")
    console.print()

    try:
        from sepilot.memory.file_checkpoint import FileCheckpointManager
        manager = FileCheckpointManager()

        checkpoint = None
        for cp in manager.checkpoints:
            if cp.id == checkpoint_id:
                checkpoint = cp
                break

        if not checkpoint:
            console.print(f"[red]Checkpoint not found: {checkpoint_id}[/red]")
            return {"success": False, "error": "Checkpoint not found"}

        # Display checkpoint info
        console.print(f"[cyan]ID:[/cyan] {checkpoint.id}")
        console.print(f"[cyan]Timestamp:[/cyan] {checkpoint.timestamp}")
        console.print(f"[cyan]Message Index:[/cyan] {checkpoint.message_index}")
        console.print(f"[cyan]User Prompt:[/cyan] {checkpoint.user_prompt}")
        console.print()

        # List files
        console.print("[cyan]Files in checkpoint:[/cyan]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("File Path", style="dim")
        table.add_column("Status", width=10)
        table.add_column("Hash", width=18)

        for path, snapshot in checkpoint.files.items():
            status = "[green]exists[/green]" if snapshot.exists else "[red]deleted[/red]"
            table.add_row(path, status, snapshot.hash[:16] if snapshot.hash else "-")

        console.print(table)
        console.print()

        # Show what would change if restored
        console.print("[bold]What would change if restored:[/bold]")
        changes = manager.get_changed_files(checkpoint)

        if changes:
            for path, change in changes.items():
                change_type = change["type"]
                if change_type == "created":
                    console.print(f"  [red]Would delete:[/red] {path}")
                elif change_type == "deleted":
                    console.print(f"  [green]Would restore:[/green] {path}")
                elif change_type == "modified":
                    console.print(f"  [yellow]Would revert:[/yellow] {path}")
        else:
            console.print("  [dim]No changes needed - files match checkpoint[/dim]")

        console.print()
        console.print("[dim]💡 Use /rewind --to {checkpoint_id} to restore this checkpoint[/dim]")

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "timestamp": checkpoint.timestamp,
                "files": len(checkpoint.files)
            }
        }

    except ImportError:
        console.print("[yellow]Checkpoint manager not available[/yellow]")
        return {"success": False, "error": "Checkpoint manager not available"}


def _rewind_to_checkpoint(
    console: Console,
    checkpoint_id: str,
    restore_env: bool = False
) -> dict[str, Any]:
    """Rewind to a specific checkpoint.

    Args:
        console: Rich console
        checkpoint_id: Target checkpoint ID
        restore_env: Whether to also restore environment

    Returns:
        Result dictionary
    """
    console.print()
    console.print(f"[bold cyan]⏪ Rewinding to checkpoint: {checkpoint_id}[/bold cyan]")
    console.print()

    try:
        from sepilot.memory.file_checkpoint import FileCheckpointManager
        manager = FileCheckpointManager()

        checkpoint = None
        for cp in manager.checkpoints:
            if cp.id == checkpoint_id:
                checkpoint = cp
                break

        if not checkpoint:
            console.print(f"[red]Checkpoint not found: {checkpoint_id}[/red]")
            return {"success": False, "error": "Checkpoint not found"}

        # Confirm before proceeding
        changes = manager.get_changed_files(checkpoint)
        if changes:
            console.print("[yellow]The following changes will be made:[/yellow]")
            for path, change in changes.items():
                change_type = change["type"]
                if change_type == "created":
                    console.print(f"  [red]Delete:[/red] {path}")
                elif change_type == "deleted":
                    console.print(f"  [green]Restore:[/green] {path}")
                elif change_type == "modified":
                    console.print(f"  [yellow]Revert:[/yellow] {path}")
            console.print()

        # Execute rewind
        result = manager.revert_to_checkpoint(checkpoint)

        files_restored = sum(1 for v in result.values() if v in ("restored", "deleted"))
        files_skipped = sum(1 for v in result.values() if "skipped" in v)
        files_error = sum(1 for v in result.values() if "error" in v)

        console.print(f"[green]✓ Rewound to checkpoint {checkpoint_id}[/green]")
        console.print(f"  Files restored: {files_restored}")
        if files_skipped:
            console.print(f"  Files skipped: {files_skipped}")
        if files_error:
            console.print(f"  [red]Files with errors: {files_error}[/red]")

        response = {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "files_restored": files_restored,
            "details": result
        }

        # Restore environment if requested
        if restore_env:
            console.print()
            console.print("[dim]Checking environment snapshot...[/dim]")
            # Environment restoration is informational only
            response["env_note"] = "Environment snapshot captured but cannot be auto-restored"

        return response

    except ImportError:
        console.print("[yellow]Checkpoint manager not available[/yellow]")
        return {"success": False, "error": "Checkpoint manager not available"}
    except Exception as e:
        console.print(f"[red]Error during rewind: {e}[/red]")
        return {"success": False, "error": str(e)}


def handle_session_export(
    console: Console,
    agent: Any | None,
    input_text: str,
) -> bool:
    """Export current session to file.

    Usage:
        /session export <path>          - Export as JSON
        /session export <path> --md     - Export as Markdown
        /session export <path> --json   - Export as JSON (default)

    Args:
        console: Rich console for output
        agent: Agent with session manager
        input_text: Original command for parsing args

    Returns:
        True if export successful
    """
    from pathlib import Path
    from sepilot.memory.session import SessionManager

    # Parse arguments
    args = input_text.strip().split()

    # Find export path
    export_path = None
    export_format = "json"

    for i, arg in enumerate(args):
        if arg == "export" and i + 1 < len(args):
            export_path = args[i + 1]
        elif arg in ("--md", "--markdown"):
            export_format = "markdown"
        elif arg in ("--json",):
            export_format = "json"

    if not export_path:
        console.print("[yellow]Usage: /session export <path> [--md|--json][/yellow]")
        console.print()
        console.print("Examples:")
        console.print("  /session export conversation.json")
        console.print("  /session export conversation.md --md")
        return False

    # Get session manager from agent or create one
    session_manager = None
    if agent and hasattr(agent, "session_manager"):
        session_manager = agent.session_manager
    else:
        # Try to get from current conversation
        session_manager = SessionManager()
        # Start a temporary session with current context
        if agent and hasattr(agent, "get_thread_id"):
            session_manager.start_session(thread_id=agent.get_thread_id())

    if not session_manager or not session_manager.current_session:
        console.print("[yellow]No active session to export[/yellow]")
        return False

    try:
        output_path = Path(export_path).expanduser().resolve()

        # Add appropriate extension if not present
        if export_format == "markdown" and not output_path.suffix:
            output_path = output_path.with_suffix(".md")
        elif export_format == "json" and not output_path.suffix:
            output_path = output_path.with_suffix(".json")

        session_manager.export_session(output_path, format=export_format)

        console.print(f"[green]Session exported to:[/green] {output_path}")
        console.print(f"[dim]Format: {export_format}[/dim]")
        console.print(f"[dim]Messages: {len(session_manager.current_session.messages)}[/dim]")

        return True

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        return False


def handle_session_import(
    console: Console,
    agent: Any | None,
    input_text: str,
) -> bool:
    """Import session from file.

    Usage:
        /session import <path>  - Import session from JSON file

    Args:
        console: Rich console for output
        agent: Agent with session manager
        input_text: Original command for parsing args

    Returns:
        True if import successful
    """
    from pathlib import Path
    from sepilot.memory.session import SessionManager

    # Parse arguments
    args = input_text.strip().split()

    # Find import path
    import_path = None
    for i, arg in enumerate(args):
        if arg == "import" and i + 1 < len(args):
            import_path = args[i + 1]
            break

    if not import_path:
        console.print("[yellow]Usage: /session import <path>[/yellow]")
        console.print()
        console.print("Example:")
        console.print("  /session import conversation.json")
        return False

    # Resolve path
    file_path = Path(import_path).expanduser().resolve()

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return False

    # Get or create session manager
    session_manager = None
    if agent and hasattr(agent, "session_manager"):
        session_manager = agent.session_manager
    else:
        session_manager = SessionManager()

    try:
        session = session_manager.import_session(file_path)

        console.print(f"[green]Session imported:[/green] {session.session_id[:12]}...")
        console.print(f"[dim]Messages: {len(session.messages)}[/dim]")
        console.print(f"[dim]Started: {session.started_at[:19]}[/dim]")
        console.print(f"[dim]Updated: {session.updated_at[:19]}[/dim]")

        return True

    except Exception as e:
        console.print(f"[red]Import failed: {e}[/red]")
        return False


def handle_session(
    console: Console,
    agent: Any | None,
    input_text: str,
) -> bool:
    """Handle session commands.

    Usage:
        /session                    - Show current session info
        /session list               - List all sessions
        /session export <path>      - Export current session
        /session import <path>      - Import session from file

    Args:
        console: Rich console for output
        agent: Agent with session manager
        input_text: Original command for parsing args

    Returns:
        True if command handled successfully
    """
    args = input_text.strip().split()
    subcommand = args[1] if len(args) > 1 else "info"

    if subcommand == "export":
        return handle_session_export(console, agent, input_text)
    elif subcommand == "import":
        return handle_session_import(console, agent, input_text)
    elif subcommand == "list":
        return _handle_session_list(console, agent)
    else:
        return _handle_session_info(console, agent)


def _handle_session_info(console: Console, agent: Any | None) -> bool:
    """Show current session info."""
    from sepilot.memory.session import SessionManager

    session_manager = None
    if agent and hasattr(agent, "session_manager"):
        session_manager = agent.session_manager

    if not session_manager or not session_manager.current_session:
        console.print("[bold cyan]Current Session[/bold cyan]")
        console.print("[dim]No active session[/dim]")
        console.print()
        console.print("Commands:")
        console.print("  /session list     - List all sessions")
        console.print("  /session export   - Export current session")
        console.print("  /session import   - Import session from file")
        return True

    session = session_manager.current_session

    console.print("[bold cyan]Current Session[/bold cyan]\n")
    console.print(f"Session ID: {session.session_id}")
    console.print(f"Thread ID: {session.thread_id}")
    console.print(f"Started: {session.started_at[:19]}")
    console.print(f"Updated: {session.updated_at[:19]}")
    console.print(f"Messages: {len(session.messages)}")

    if session.context:
        console.print(f"Context keys: {', '.join(session.context.keys())}")

    return True


def _handle_session_list(console: Console, agent: Any | None) -> bool:
    """List all sessions."""
    from sepilot.memory.session import SessionHistory

    history = SessionHistory()
    sessions = history.list_sessions()

    console.print("[bold cyan]Saved Sessions[/bold cyan]\n")

    if not sessions:
        console.print("[dim]No sessions found[/dim]")
        return True

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Session ID", style="cyan", width=15)
    table.add_column("Started", width=20)
    table.add_column("Updated", width=20)
    table.add_column("Messages", justify="right", width=10)

    for s in sessions[:10]:  # Show last 10
        table.add_row(
            s["session_id"][:12] + "...",
            s["started_at"][:19],
            s["updated_at"][:19],
            str(s["message_count"]),
        )

    console.print(table)

    if len(sessions) > 10:
        console.print(f"\n[dim]... and {len(sessions) - 10} more sessions[/dim]")

    return True
