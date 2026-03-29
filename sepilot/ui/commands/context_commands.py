"""Context management commands: compact, clear, context, cost.

This module follows the Single Responsibility Principle (SRP) by handling
only context and cost-related commands.
"""

import os
from typing import Any

from rich.console import Console

from sepilot.agent.execution_context import (
    is_user_turn_boundary_message,
    is_user_visible_conversation_message,
)
from sepilot.ui.input_utils import prompt_confirm


def _agent_uses_memory(agent: Any | None) -> bool:
    """Return True only for real memory-enabled agents, not loose mocks."""
    return bool(agent is not None and getattr(agent, "enable_memory", False) is True)


def _sync_conversation_context_from_agent(agent: Any, conversation_context: list) -> None:
    """Refresh local conversation context from the active agent thread."""
    if not agent or not hasattr(agent, "get_conversation_messages"):
        return

    try:
        messages = agent.get_conversation_messages()
    except Exception:
        return

    conversation_context.clear()
    for msg in messages:
        if is_user_visible_conversation_message(msg):
            conversation_context.append(msg)


def handle_compact(
    console: Console,
    conversation_context: list,
    agent: Any | None,
    input_text: str,
) -> bool:
    """Compact conversation context by summarizing old messages.

    Args:
        console: Rich console for output
        conversation_context: Current conversation context
        agent: Agent with LLM for summarization
        input_text: Original command for parsing args

    Returns:
        True if compaction was performed
    """
    from sepilot.agent.context_manager import ContextManager

    has_agent_thread = bool(
        agent
        and _agent_uses_memory(agent)
        and hasattr(agent, "get_conversation_messages")
    )
    if not conversation_context and not has_agent_thread:
        console.print("[yellow]No conversation context to compact[/yellow]")
        return False

    # Parse focus instruction
    args = input_text.strip()
    focus_instruction = None
    if args and not args.startswith('/'):
        if args.lower().startswith('compact'):
            args = args[7:].strip()
        if args:
            focus_instruction = args

    if focus_instruction:
        console.print(f"[cyan]📦 Compacting with focus: [bold]{focus_instruction}[/bold][/cyan]")
    else:
        console.print("[cyan]📦 Compacting conversation context...[/cyan]")

    if _agent_uses_memory(agent) and hasattr(agent, "compact_conversation_context"):
        result = agent.compact_conversation_context(
            keep_recent=10,
            focus_instruction=focus_instruction,
        )
        if not result.get("success"):
            console.print(f"[yellow]{result.get('error', 'Context compaction failed')}[/yellow]")
            return False

        _sync_conversation_context_from_agent(agent, conversation_context)
        console.print("[green]✅ Context compacted successfully![/green]")
        console.print(
            f"  Messages: {result.get('messages_before', 0)} → {result.get('messages_after', 0)}"
        )
        if focus_instruction:
            console.print(f"  Focus: {focus_instruction}")
        return True

    # Get current stats
    tokens_before = _count_tokens(conversation_context)
    message_count_before = len(conversation_context)

    # Get max tokens
    max_tokens = int(os.getenv('MAX_TOKENS', '96000'))
    context_manager = ContextManager(max_context_tokens=max_tokens)

    # Get LLM
    llm = None
    if agent and hasattr(agent, 'llm'):
        llm = agent.llm

    # Perform compaction
    if llm:
        try:
            console.print("[dim]Using LLM to summarize conversation history...[/dim]")
            conversation_context[:] = context_manager.summarize_messages(
                conversation_context,
                llm,
                keep_recent=10,
                focus_instruction=focus_instruction
            )
            method = "summarized"
        except Exception as e:
            console.print(f"[yellow]Summarization failed ({e}), using simple compaction[/yellow]")
            conversation_context[:] = context_manager.compact_messages(
                conversation_context,
                keep_recent=10
            )
            method = "compacted"
    else:
        conversation_context[:] = context_manager.compact_messages(
            conversation_context,
            keep_recent=10
        )
        method = "compacted"

    # Get new stats
    tokens_after = _count_tokens(conversation_context)
    message_count_after = len(conversation_context)

    # Show results
    console.print(f"[green]✅ Context {method} successfully![/green]")
    console.print(f"  Messages: {message_count_before} → {message_count_after}")
    console.print(f"  Tokens: {tokens_before:,} → {tokens_after:,}")
    if tokens_before > 0:
        saved_pct = (tokens_before - tokens_after) / tokens_before * 100
        console.print(f"  Saved: {tokens_before - tokens_after:,} tokens ({saved_pct:.1f}%)")
    if focus_instruction:
        console.print(f"  Focus: {focus_instruction}")

    return True


def handle_clear_context(
    console: Console,
    conversation_context: list,
    agent: Any | None = None,
    session: Any | None = None,
) -> bool:
    """Clear all conversation context and agent LangGraph state.

    Args:
        console: Rich console for output
        conversation_context: Current conversation context to clear
        agent: Optional agent to also clear LangGraph checkpoint messages

    Returns:
        True if cleared
    """
    has_agent_thread = bool(
        agent
        and _agent_uses_memory(agent)
        and hasattr(agent, "get_conversation_messages")
    )
    if not conversation_context and not has_agent_thread:
        console.print("[yellow]Conversation context is already empty[/yellow]")
        return False

    message_count = len(conversation_context)
    if message_count == 0 and has_agent_thread:
        try:
            message_count = len(agent.get_conversation_messages())
        except Exception:
            message_count = 0

    # Ask for confirmation
    console.print(f"[yellow]⚠️  This will clear {message_count} messages from the conversation history.[/yellow]")
    console.print("[yellow]This action cannot be undone.[/yellow]")

    confirm = prompt_confirm("Are you sure?", session=session, default=False)
    if confirm is not True:
        console.print("[dim]Cancelled[/dim]")
        return False

    # Clear local conversation context
    conversation_context.clear()

    # Also clear agent's LangGraph checkpoint messages and plan state
    if agent:
        if _agent_uses_memory(agent) and hasattr(agent, "clear_conversation_messages"):
            try:
                agent.clear_conversation_messages()
            except Exception:
                pass  # Best-effort: local context is already cleared
        elif hasattr(agent, 'get_conversation_messages') and hasattr(agent, 'rewind_messages'):
            try:
                messages = agent.get_conversation_messages()
                if messages:
                    human_count = sum(1 for m in messages if getattr(m, 'type', '') == 'human')
                    if human_count > 0:
                        agent.rewind_messages(human_count)
            except Exception:
                pass  # Best-effort: local context is already cleared

        # Reset plan-related state so stale plans don't persist
        if hasattr(agent, 'reset_plan_state'):
            try:
                agent.reset_plan_state()
            except Exception:
                pass

    console.print(f"[green]✅ Cleared {message_count} messages from conversation history[/green]")
    console.print("[dim]Starting fresh conversation...[/dim]")

    return True


def handle_context(
    console: Console,
    conversation_context: list,
    agent: Any | None,
    input_text: str = "",
) -> None:
    """Show context usage visualization.

    Usage:
        /context           - Show context usage grid
        /context --detail  - Show detailed message breakdown

    Args:
        console: Rich console for output
        conversation_context: Current conversation context
        agent: Optional agent for additional metrics
        input_text: Original command input for parsing options
    """
    # Check for --detail flag
    args = input_text.strip().split() if input_text else []
    show_detail = any(arg in ("--detail", "-d", "--breakdown") for arg in args)

    if show_detail:
        # Use ContextDisplayManager for detailed breakdown
        from sepilot.ui.context_display import ContextDisplayManager
        display_manager = ContextDisplayManager(
            console=console,
            agent=agent,
            conversation_context=conversation_context
        )
        display_manager.display_detailed_breakdown()
        return

    from sepilot.ui.context_display import ContextDisplayManager

    display_manager = ContextDisplayManager(
        console=console,
        agent=agent,
        conversation_context=conversation_context,
    )
    usage = display_manager.get_context_usage_info()
    display_messages = display_manager._get_user_visible_messages_for_summary()
    max_context = usage.max_tokens
    current_tokens = usage.tokens_used
    usage_percent = usage.usage_percent

    # Create visual grid (40 cells)
    grid_size = 40
    filled_cells = int((usage_percent / 100) * grid_size)

    # Determine color
    if usage_percent < 50:
        color = "green"
    elif usage_percent < 70:
        color = "yellow"
    elif usage_percent < 85:
        color = "orange1"
    else:
        color = "red"

    # Build grid
    grid = ""
    for i in range(grid_size):
        grid += f"[{color}]█[/{color}]" if i < filled_cells else "[dim]░[/dim]"

    # Display
    console.print()
    console.print("[bold cyan]📊 Context Usage[/bold cyan]")
    console.print()
    console.print(f"  {grid}")
    console.print()
    console.print(f"  [{color}]{current_tokens:,}[/{color}] / {max_context:,} tokens ({usage_percent:.1f}%)")

    # Show breakdown
    if display_messages:
        msg_count = len(display_messages)
        human_count = sum(1 for m in display_messages if is_user_turn_boundary_message(m))
        ai_count = sum(1 for m in display_messages if getattr(m, 'type', '') in ('ai', 'assistant'))
        console.print(f"  [dim]Messages: {msg_count} ({human_count} user, {ai_count} assistant)[/dim]")

    # Enhanced: Show top 5 messages by token usage
    if display_messages:
        from sepilot.agent.context_manager import ContextManager as _ContextManager
        _cm = _ContextManager(max_context_tokens=max_context)
        _show_enhanced_context_stats(console, _cm, display_messages, current_tokens)

    # Recommendations
    console.print()
    if usage_percent >= 85:
        console.print("[red]⚠️  Context is nearly full! Use /compact or /clear[/red]")
    elif usage_percent >= 70:
        console.print("[yellow]💡 Consider using /compact to free up space[/yellow]")
    else:
        console.print("[dim]💡 Use /compact to summarize, /clear to reset[/dim]")


def handle_cost(
    console: Console,
    total_tokens: int,
    command_count: int,
    agent: Any | None,
) -> None:
    """Show estimated cost for the current session.

    Args:
        console: Rich console for output
        total_tokens: Total tokens used
        command_count: Number of commands executed
        agent: Optional agent for model info
    """
    console.print()
    console.print("[bold cyan]💰 Session Cost Estimate[/bold cyan]")
    console.print()

    # Estimate input/output split (roughly 60/40)
    input_tokens = int(total_tokens * 0.6)
    output_tokens = int(total_tokens * 0.4)

    # Get model for pricing
    model_name = "unknown"
    if agent and hasattr(agent, 'settings'):
        model_name = getattr(agent.settings, 'model', 'unknown')

    # Pricing per 1M tokens
    pricing = {
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "default": {"input": 3.0, "output": 15.0}
    }

    # Find matching pricing
    model_pricing = pricing["default"]
    for key in pricing:
        if key in model_name.lower():
            model_pricing = pricing[key]
            break

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    total_cost = input_cost + output_cost

    # Display
    console.print(f"  [cyan]Model:[/cyan] {model_name}")
    console.print(f"  [cyan]Commands:[/cyan] {command_count}")
    console.print()
    console.print(f"  [dim]Input tokens:[/dim]  {input_tokens:,} (${input_cost:.4f})")
    console.print(f"  [dim]Output tokens:[/dim] {output_tokens:,} (${output_cost:.4f})")
    console.print()
    console.print(f"  [bold green]Total:[/bold green] {total_tokens:,} tokens (~${total_cost:.4f})")
    console.print()
    console.print(f"[dim]* Prices based on {model_name} rates. Actual costs may vary.[/dim]")


def _show_enhanced_context_stats(
    console: Console,
    context_manager: Any,
    messages: list,
    current_tokens: int,
) -> None:
    """Show enhanced context statistics with Rich table.

    Displays:
    - Top 5 messages by token usage
    - Instructions/Rules token ratio
    - Estimated remaining turns
    """
    from rich.table import Table

    console.print()

    # Top 5 messages by token usage
    breakdown = context_manager.get_message_token_breakdown(messages)
    top5 = breakdown[:5]

    if top5:
        table = Table(title="Top 5 Messages by Token Usage", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Role", width=10)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("Preview", max_width=50)

        for entry in top5:
            pct = (entry['tokens'] / current_tokens * 100) if current_tokens > 0 else 0
            table.add_row(
                str(entry['index']),
                entry['role'],
                f"{entry['tokens']:,} ({pct:.0f}%)",
                entry['preview'][:50],
            )
        console.print(table)

    # Instructions ratio
    ratio_info = context_manager.get_instructions_token_ratio(messages)
    instr_pct = ratio_info['ratio'] * 100
    console.print(
        f"  [cyan]Instructions/Rules:[/cyan] {ratio_info['instruction_tokens']:,} tokens "
        f"({instr_pct:.1f}% of context)"
    )

    # Estimated remaining turns
    remaining_turns = context_manager.estimate_remaining_turns(messages, current_tokens)
    if remaining_turns > 20:
        turn_color = "green"
    elif remaining_turns > 5:
        turn_color = "yellow"
    else:
        turn_color = "red"
    console.print(
        f"  [cyan]Estimated remaining turns:[/cyan] [{turn_color}]~{remaining_turns}[/{turn_color}]"
    )


def _count_tokens(messages: list) -> int:
    """Count tokens in a list of messages.

    Args:
        messages: List of message objects

    Returns:
        Token count
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")

        total = 0
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                total += len(encoding.encode(str(msg.content)))
        return total
    except Exception:
        # Fallback: estimate 400 tokens per message
        return len(messages) * 400
