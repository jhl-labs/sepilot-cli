"""Context usage display and auto-compaction (Claude Code style).

This module follows the Single Responsibility Principle (SRP) by handling
only context visualization and automatic compaction.

Provides:
- Token counting using tiktoken
- Context usage visualization (progress bars, grids)
- Auto-compact when context threshold reached
- Model context window lookup
"""

import os
from dataclasses import dataclass
from typing import Any

from rich.console import Console


@dataclass
class ContextUsageInfo:
    """Information about current context usage."""

    tokens_used: int
    max_tokens: int
    usage_percent: float
    tokens_remaining: int
    status: str  # 'low', 'medium', 'high'
    color: str  # Rich color for display
    model_name: str


class ContextDisplayManager:
    """Manages context usage display and auto-compaction.

    This class handles token counting, usage visualization, and
    automatic context compaction when thresholds are reached.
    """

    # Model context windows (in tokens)
    MODEL_CONTEXT_WINDOWS = {
        # STEP models
        "step-3.5-flash": 32768,
        "step-3.5": 65536,
        # OpenAI models
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4-1106": 128000,
        "gpt-4-0125": 128000,
        "gpt-4-32k": 32768,
        "gpt-4": 8192,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo": 4096,
        # O1 models
        "o1-preview": 128000,
        "o1-mini": 128000,
        # Anthropic Claude models
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-2": 100000,
        "claude": 100000,
        # Other hosted models
        "qwen3-vl-235b": 131072,
        "qwen3-coder": 131072,
        "glm-4.7-cloud": 131072,
        # Local models
        "ollama": 32768,
        "llama": 32768,
    }

    def __init__(
        self,
        console: Console | None = None,
        agent: Any | None = None,
        conversation_context: list | None = None,
    ):
        """Initialize context display manager.

        Args:
            console: Rich console for output
            agent: Reference to the agent for metrics
            conversation_context: Shared conversation history
        """
        self.console = console or Console()
        self.agent = agent
        # Keep the original shared list reference even when it's empty.
        # Using `or []` breaks linkage and context usage stays at 0.
        self.conversation_context = (
            conversation_context if conversation_context is not None else []
        )
        self._context_warnings: set[str] = set()

    def get_model_context_window(self, model_name: str) -> int:
        """Get context window size for a model.

        Args:
            model_name: Name of the LLM model

        Returns:
            Context window size in tokens
        """
        model_lower = model_name.lower()

        for key, window_size in self.MODEL_CONTEXT_WINDOWS.items():
            if key in model_lower:
                return window_size

        # Default fallback for modern models
        return 128000

    def count_tokens(self, text: str, model_name: str = "gpt-4") -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for
            model_name: Model name for encoding selection

        Returns:
            Token count
        """
        try:
            import tiktoken

            encoding_name = "cl100k_base"
            if "gpt-3.5" in model_name.lower() or "gpt-4" in model_name.lower():
                encoding_name = "cl100k_base"
            elif "gpt2" in model_name.lower():
                encoding_name = "gpt2"

            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: estimate ~4 chars per token
            return len(text) // 4

    def count_message_tokens(self, message: Any, model_name: str = "gpt-4") -> int:
        """Count tokens in a message object.

        Args:
            message: Message with content attribute
            model_name: Model name for encoding

        Returns:
            Token count
        """
        content = getattr(message, 'content', '')
        if content:
            return self.count_tokens(str(content), model_name)
        return 0

    def count_messages_tokens(
        self, messages: list, model_name: str = "gpt-4"
    ) -> int:
        """Count total tokens in a list of messages.

        Args:
            messages: List of message objects
            model_name: Model name for encoding

        Returns:
            Total token count
        """
        total = 0
        for msg in messages:
            total += self.count_message_tokens(msg, model_name)
        return total

    def get_context_usage_info(self) -> ContextUsageInfo:
        """Get context window usage information.

        Returns:
            ContextUsageInfo with usage details
        """
        # Get model name
        model_name = "gpt-4-turbo-preview"
        if self.agent and hasattr(self.agent, 'settings'):
            model_name = getattr(self.agent.settings, 'model', model_name)

        # Get max tokens from environment or model default
        max_tokens = self._get_max_tokens(model_name)

        # Calculate current token usage
        tokens_used = self._calculate_tokens_used(model_name)

        # Calculate usage metrics
        usage_percent = (tokens_used / max_tokens * 100) if max_tokens > 0 else 0
        tokens_remaining = max_tokens - tokens_used

        # Determine status and color
        if usage_percent < 50:
            status, color = 'low', 'green'
        elif usage_percent < 80:
            status, color = 'medium', 'yellow'
        else:
            status, color = 'high', 'red'

        return ContextUsageInfo(
            tokens_used=tokens_used,
            max_tokens=max_tokens,
            usage_percent=usage_percent,
            tokens_remaining=tokens_remaining,
            status=status,
            color=color,
            model_name=model_name,
        )

    def _get_max_tokens(self, model_name: str) -> int:
        """Get maximum tokens from env or model default."""
        for env_key in ['MAX_TOKENS', 'LLM_MAX_TOKENS', 'SEPILOT_MAX_TOKENS']:
            env_value = os.getenv(env_key)
            if env_value:
                try:
                    return int(env_value)
                except (ValueError, TypeError):
                    pass

        if self.agent and hasattr(self.agent, 'settings'):
            context_window = getattr(self.agent.settings, 'context_window', None)
            if isinstance(context_window, int) and context_window > 0:
                return context_window

            max_tokens = getattr(self.agent.settings, 'max_tokens', None)
            if isinstance(max_tokens, int) and max_tokens > 0:
                return max_tokens

        return self.get_model_context_window(model_name)

    def _calculate_tokens_used(self, model_name: str) -> int:
        """Calculate current token usage from various sources."""
        tokens_used = 0

        # Priority 1: From conversation_context
        if self.conversation_context:
            tokens_used = self.count_messages_tokens(self.conversation_context, model_name)

        # Priority 2: From agent state
        if tokens_used == 0 and self.agent and hasattr(self.agent, 'graph'):
            try:
                thread_config = getattr(self.agent, "thread_config", None)
                if thread_config:
                    state = self.agent.graph.get_state(thread_config)
                    if state and hasattr(state, 'values'):
                        messages = state.values.get('messages', [])
                        tokens_used = self.count_messages_tokens(messages, model_name)
            except Exception:
                pass

        return tokens_used

    def format_context_usage(self, compact: bool = False) -> str:
        """Format context usage for display.

        Args:
            compact: If True, return one-line format

        Returns:
            Formatted string with Rich markup
        """
        usage = self.get_context_usage_info()

        if compact:
            return (
                f"[{usage.color}]ctx {usage.tokens_used:,}/{usage.max_tokens:,} "
                f"({usage.usage_percent:.1f}%)[/{usage.color}]"
            )

        # Detailed format
        bar_width = 30
        filled = int(bar_width * usage.usage_percent / 100)
        empty = bar_width - filled

        bar = f"[{usage.color}]{'█' * filled}[/{usage.color}][dim]{'░' * empty}[/dim]"

        return (
            f"[cyan]Context Usage:[/cyan]\n"
            f"  {bar}\n"
            f"  [{usage.color}]{usage.tokens_used:,}[/{usage.color}] / "
            f"{usage.max_tokens:,} tokens "
            f"([{usage.color}]{usage.usage_percent:.1f}%[/{usage.color}])\n"
            f"  [dim]{usage.tokens_remaining:,} tokens remaining[/dim]"
        )

    def display_context_grid(self) -> None:
        """Display context usage as a visual grid."""
        usage = self.get_context_usage_info()

        # Create visual grid (40 cells)
        grid_size = 40
        filled_cells = int((usage.usage_percent / 100) * grid_size)

        # Determine color
        if usage.usage_percent < 50:
            color = "green"
        elif usage.usage_percent < 70:
            color = "yellow"
        elif usage.usage_percent < 85:
            color = "orange1"
        else:
            color = "red"

        # Build grid
        grid = ""
        for i in range(grid_size):
            grid += f"[{color}]█[/{color}]" if i < filled_cells else "[dim]░[/dim]"

        # Display
        self.console.print()
        self.console.print("[bold cyan]📊 Context Usage[/bold cyan]")
        self.console.print()
        self.console.print(f"  {grid}")
        self.console.print()
        self.console.print(f"  [{color}]{usage.tokens_used:,}[/{color}] / {usage.max_tokens:,} tokens ({usage.usage_percent:.1f}%)")

        # Show breakdown
        if self.conversation_context:
            msg_count = len(self.conversation_context)
            human_count = sum(1 for m in self.conversation_context if getattr(m, 'type', '') == 'human')
            ai_count = sum(1 for m in self.conversation_context if getattr(m, 'type', '') == 'ai')
            self.console.print(f"  [dim]Messages: {msg_count} ({human_count} user, {ai_count} assistant)[/dim]")

        # Recommendations
        self.console.print()
        if usage.usage_percent >= 85:
            self.console.print("[red]⚠️  Context is nearly full! Use /compact or /clear[/red]")
        elif usage.usage_percent >= 70:
            self.console.print("[yellow]💡 Consider using /compact to free up space[/yellow]")
        else:
            self.console.print("[dim]💡 Use /compact to summarize, /clear to reset[/dim]")

    def check_auto_compact(self, llm: Any | None = None) -> bool:
        """Check context usage and auto-compact if needed.

        Args:
            llm: LLM for summarization (optional)

        Returns:
            True if compaction was performed
        """
        if os.getenv('SEPILOT_AUTO_COMPACT', '1') == '0':
            return False

        if not self.conversation_context:
            return False

        # Get thresholds
        warn_threshold = float(os.getenv('SEPILOT_WARN_THRESHOLD', '0.80'))
        compact_threshold = float(os.getenv('SEPILOT_COMPACT_THRESHOLD', '0.92'))
        target_ratio = float(os.getenv('SEPILOT_TARGET_RATIO', '0.60'))

        usage = self.get_context_usage_info()

        # Check if we should auto-compact (92%+)
        if usage.usage_percent >= compact_threshold * 100:
            return self._perform_auto_compact(llm, target_ratio, usage)

        # Check if we should warn (80%+)
        if usage.usage_percent >= warn_threshold * 100:
            self._show_warning(usage)

        return False

    def _perform_auto_compact(
        self, llm: Any | None, target_ratio: float, usage: ContextUsageInfo
    ) -> bool:
        """Perform automatic context compaction.

        Args:
            llm: LLM for summarization
            target_ratio: Target ratio of max tokens
            usage: Current usage info

        Returns:
            True if compaction was successful
        """
        from sepilot.agent.context_manager import ContextManager

        self.console.print()
        self.console.print(
            f"[bold yellow]⚠️  Context {usage.usage_percent:.0f}% 도달 "
            f"({usage.tokens_used:,}/{usage.max_tokens:,} 토큰) - 자동 압축 시작...[/bold yellow]"
        )

        tokens_before = usage.tokens_used
        message_count_before = len(self.conversation_context)
        target_tokens = int(usage.max_tokens * target_ratio)

        self.console.print(f"[dim]목표: {target_tokens:,} 토큰 ({target_ratio*100:.0f}%)[/dim]")

        context_manager = ContextManager(max_context_tokens=usage.max_tokens)

        method = "압축"
        if llm:
            try:
                self.conversation_context[:] = context_manager.summarize_to_token_limit(
                    self.conversation_context, llm, target_tokens=target_tokens
                )
                method = "요약"
            except Exception as e:
                self.console.print(f"[dim]요약 실패 ({e}), 단순 압축 사용[/dim]")
                self.conversation_context[:] = context_manager.compact_to_token_limit(
                    self.conversation_context, target_tokens=target_tokens, min_keep=4
                )
        else:
            self.conversation_context[:] = context_manager.compact_to_token_limit(
                self.conversation_context, target_tokens=target_tokens, min_keep=4
            )

        # Calculate new stats
        tokens_after = self.count_messages_tokens(self.conversation_context, usage.model_name)
        message_count_after = len(self.conversation_context)
        tokens_saved = tokens_before - tokens_after
        new_usage = (tokens_after / usage.max_tokens) * 100 if usage.max_tokens > 0 else 0

        self.console.print(f"[green]✅ 자동 {method} 완료![/green]")
        self.console.print(f"   [dim]메시지: {message_count_before} → {message_count_after}[/dim]")
        self.console.print(f"   [dim]토큰: {tokens_before:,} → {tokens_after:,} ({tokens_saved:,} 절약)[/dim]")
        self.console.print(f"   [dim]사용량: {usage.usage_percent:.1f}% → {new_usage:.1f}%[/dim]")
        self.console.print()

        return True

    def display_detailed_breakdown(self) -> None:
        """Display detailed message-by-message token breakdown (OpenCode style)."""
        usage = self.get_context_usage_info()

        self.console.print()
        self.console.print("[bold cyan]📊 Context Breakdown[/bold cyan]")
        self.console.print()

        # Overall stats
        self.console.print(f"[dim]Model: {usage.model_name}[/dim]")
        self.console.print(f"[dim]Context window: {usage.max_tokens:,} tokens[/dim]")
        self.console.print()

        if not self.conversation_context:
            self.console.print("[yellow]No messages in context[/yellow]")
            return

        # Create table
        from rich.table import Table
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Type", width=10)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("%", justify="right", width=6)
        table.add_column("Preview", max_width=50)

        total_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        tool_tokens = 0

        for i, msg in enumerate(self.conversation_context, 1):
            msg_type = getattr(msg, 'type', 'unknown')
            content = str(getattr(msg, 'content', ''))
            tokens = self.count_tokens(content, usage.model_name)
            total_tokens += tokens
            percent = (tokens / usage.max_tokens * 100) if usage.max_tokens > 0 else 0

            # Track by type
            if msg_type == 'human':
                user_tokens += tokens
                type_display = "[cyan]User[/cyan]"
            elif msg_type == 'ai':
                assistant_tokens += tokens
                type_display = "[green]Assistant[/green]"
            elif msg_type == 'tool':
                tool_tokens += tokens
                type_display = "[yellow]Tool[/yellow]"
            else:
                type_display = f"[dim]{msg_type}[/dim]"

            # Preview (truncate)
            preview = content.replace('\n', ' ').strip()
            if len(preview) > 47:
                preview = preview[:47] + "..."

            table.add_row(
                str(i),
                type_display,
                f"{tokens:,}",
                f"{percent:.1f}%",
                preview
            )

        self.console.print(table)

        # Summary by type
        self.console.print()
        self.console.print("[bold]Token Distribution:[/bold]")
        total = max(total_tokens, 1)
        self.console.print(f"  [cyan]User:[/cyan] {user_tokens:,} ({user_tokens/total*100:.1f}%)")
        self.console.print(f"  [green]Assistant:[/green] {assistant_tokens:,} ({assistant_tokens/total*100:.1f}%)")
        if tool_tokens > 0:
            self.console.print(f"  [yellow]Tool:[/yellow] {tool_tokens:,} ({tool_tokens/total*100:.1f}%)")

        # Cost estimate
        cost = estimate_cost(total_tokens, usage.model_name)
        self.console.print()
        self.console.print(f"[bold]Estimated Cost:[/bold] ${cost['total_cost']:.4f}")
        self.console.print(f"  [dim]Input: {cost['input_tokens']:,} tokens (${cost['input_cost']:.4f})[/dim]")
        self.console.print(f"  [dim]Output: {cost['output_tokens']:,} tokens (${cost['output_cost']:.4f})[/dim]")

        # Recommendations
        self.console.print()
        if usage.usage_percent >= 85:
            self.console.print("[red]⚠️  High usage! Use /compact or /clear to free space[/red]")
        elif usage.usage_percent >= 70:
            self.console.print("[yellow]💡 Consider using /compact to summarize context[/yellow]")
        else:
            self.console.print("[dim]💡 Use /context for quick view, /context --detail for this breakdown[/dim]")

    def _show_warning(self, usage: ContextUsageInfo) -> None:
        """Show context usage warning."""
        warn_key = f"warned_{int(usage.usage_percent // 10) * 10}"

        if warn_key not in self._context_warnings:
            self._context_warnings.add(warn_key)
            self.console.print()
            self.console.print(
                f"[yellow]💡 Context {usage.usage_percent:.0f}% 사용 중 - "
                "/compact로 압축하거나 계속 진행하세요[/yellow]"
            )
            self.console.print("[dim]   92% 도달 시 자동 압축됩니다. /context로 상세 보기[/dim]")


# Cost estimation
PRICING_PER_MILLION = {
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "default": {"input": 3.0, "output": 15.0},
}


def estimate_cost(
    total_tokens: int,
    model_name: str,
    input_ratio: float = 0.6,
) -> dict[str, float]:
    """Estimate cost for token usage.

    Args:
        total_tokens: Total tokens used
        model_name: Model name for pricing
        input_ratio: Ratio of input to total tokens

    Returns:
        Dictionary with input_cost, output_cost, total_cost
    """
    input_tokens = int(total_tokens * input_ratio)
    output_tokens = total_tokens - input_tokens

    # Find matching pricing
    model_pricing = PRICING_PER_MILLION["default"]
    for key in PRICING_PER_MILLION:
        if key in model_name.lower():
            model_pricing = PRICING_PER_MILLION[key]
            break

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }
