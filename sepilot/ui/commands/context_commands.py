"""Context management commands: compact, clear, context, cost.

This module follows the Single Responsibility Principle (SRP) by handling
only context and cost-related commands.
"""

import os
from typing import Any

from rich.console import Console


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

    if not conversation_context:
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
) -> bool:
    """Clear all conversation context and agent LangGraph state.

    Args:
        console: Rich console for output
        conversation_context: Current conversation context to clear
        agent: Optional agent to also clear LangGraph checkpoint messages

    Returns:
        True if cleared
    """
    if not conversation_context:
        console.print("[yellow]Conversation context is already empty[/yellow]")
        return False

    message_count = len(conversation_context)

    # Ask for confirmation
    console.print(f"[yellow]⚠️  This will clear {message_count} messages from the conversation history.[/yellow]")
    console.print("[yellow]This action cannot be undone.[/yellow]")

    try:
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            console.print("[dim]Cancelled[/dim]")
            return False
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled[/dim]")
        return False

    # Clear local conversation context
    conversation_context.clear()

    # Also clear agent's LangGraph checkpoint messages
    if agent and hasattr(agent, 'get_conversation_messages') and hasattr(agent, 'rewind_messages'):
        try:
            messages = agent.get_conversation_messages()
            if messages:
                human_count = sum(1 for m in messages if getattr(m, 'type', '') == 'human')
                if human_count > 0:
                    agent.rewind_messages(human_count)
        except Exception:
            pass  # Best-effort: local context is already cleared

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

    # Get max context window
    max_context = int(os.getenv('MAX_TOKENS', '96000'))

    # Calculate current token usage
    current_tokens = _count_tokens(conversation_context)

    # Also check agent state
    if current_tokens == 0 and agent and hasattr(agent, 'get_conversation_messages'):
        try:
            messages = agent.get_conversation_messages()
            current_tokens = _count_tokens(messages)
        except Exception:
            pass

    # Calculate percentage
    usage_percent = (current_tokens / max_context) * 100 if max_context > 0 else 0

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
    if conversation_context:
        msg_count = len(conversation_context)
        human_count = sum(1 for m in conversation_context if getattr(m, 'type', '') == 'human')
        ai_count = sum(1 for m in conversation_context if getattr(m, 'type', '') == 'ai')
        console.print(f"  [dim]Messages: {msg_count} ({human_count} user, {ai_count} assistant)[/dim]")

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
