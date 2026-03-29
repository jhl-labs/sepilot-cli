"""Streaming output handler for LLM responses

Provides real-time token-by-token output for better user experience.
"""

import time
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class StreamingHandler:
    """Handle streaming output from LLM"""

    def __init__(
        self,
        console: Console | None = None,
        show_panel: bool = True,
        panel_title: str = "AI Response"
    ):
        """
        Initialize streaming handler

        Args:
            console: Rich console for output
            show_panel: Whether to wrap output in a panel
            panel_title: Title for the panel
        """
        self.console = console or Console()
        self.show_panel = show_panel
        self.panel_title = panel_title
        self.current_text = ""
        self.live: Live | None = None
        self.start_time: float = 0
        self.chunk_count: int = 0

    def start(self):
        """Start streaming output"""
        self.current_text = ""
        self.chunk_count = 0
        self.start_time = time.time()

        if self.show_panel:
            self.live = Live(
                self._render_panel(),
                console=self.console,
                refresh_per_second=10,
                transient=False
            )
            self.live.start()

    def update(self, token: str):
        """
        Add a new streaming chunk to the output

        Args:
            token: New chunk from LLM (may contain multiple tokens)
        """
        self.current_text += token
        self.chunk_count += 1

        if self.live:
            self.live.update(self._render_panel())
        elif not self.show_panel:
            # Direct streaming without panel
            self.console.print(token, end="")

    def finish(self):
        """Finish streaming and clean up"""
        if self.live:
            self.live.stop()
            self.live = None

        # Show final output if not using live display
        if not self.show_panel:
            self.console.print()  # Newline at end

        # Calculate stats
        elapsed = time.time() - self.start_time
        chunks_per_sec = self.chunk_count / elapsed if elapsed > 0 else 0

        # Show stats in dim
        if self.console and elapsed > 0:
            self.console.print(
                f"[dim]Streamed {self.chunk_count} chunks in {elapsed:.1f}s "
                f"({chunks_per_sec:.1f} chunks/sec)[/dim]"
            )

    def get_text(self) -> str:
        """Get accumulated text"""
        return self.current_text

    def _render_panel(self) -> Panel:
        """Render current text in a panel"""
        text = Text(self.current_text)

        # Add blinking cursor at end
        text.append("▌", style="blink")

        return Panel(
            text,
            title=f"[bold]{self.panel_title}[/bold]",
            border_style="cyan",
            padding=(1, 2)
        )


class CallbackStreamingHandler:
    """Streaming handler that uses callbacks instead of direct console output

    Useful for integrating with custom UI components or logging systems.
    """

    def __init__(
        self,
        on_token: Callable[[str], None] | None = None,
        on_start: Callable[[], None] | None = None,
        on_finish: Callable[[str, dict[str, Any]], None] | None = None
    ):
        """
        Initialize callback-based streaming handler

        Args:
            on_token: Called for each token
            on_start: Called when streaming starts
            on_finish: Called when streaming finishes with (full_text, stats)
        """
        self.on_token = on_token or (lambda t: None)
        self.on_start = on_start or (lambda: None)
        self.on_finish = on_finish or (lambda t, s: None)

        self.current_text = ""
        self.start_time: float = 0
        self.chunk_count: int = 0

    def start(self):
        """Start streaming"""
        self.current_text = ""
        self.chunk_count = 0
        self.start_time = time.time()
        self.on_start()

    def update(self, token: str):
        """Add chunk"""
        self.current_text += token
        self.chunk_count += 1
        self.on_token(token)

    def finish(self):
        """Finish streaming"""
        elapsed = time.time() - self.start_time
        stats = {
            'chunk_count': self.chunk_count,
            'elapsed_seconds': elapsed,
            'chunks_per_second': self.chunk_count / elapsed if elapsed > 0 else 0
        }
        self.on_finish(self.current_text, stats)

    def get_text(self) -> str:
        """Get accumulated text"""
        return self.current_text


def stream_llm_response(
    llm_invoke_fn: Callable,
    messages: list,
    console: Console | None = None,
    show_panel: bool = True
) -> str:
    """
    Stream LLM response with real-time output

    Args:
        llm_invoke_fn: LLM streaming invoke function (e.g., llm.stream)
        messages: Messages to send to LLM
        console: Rich console
        show_panel: Whether to show response in panel

    Returns:
        Complete response text

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(streaming=True)
        >>> messages = [HumanMessage(content="Hello!")]
        >>> response = stream_llm_response(llm.stream, messages)
    """
    handler = StreamingHandler(console=console, show_panel=show_panel)
    handler.start()

    try:
        # Stream tokens from LLM
        for chunk in llm_invoke_fn(messages):
            # Extract content from chunk
            if hasattr(chunk, 'content'):
                token = chunk.content
            elif isinstance(chunk, str):
                token = chunk
            else:
                token = str(chunk)

            if token:
                handler.update(token)

        handler.finish()
        return handler.get_text()

    except Exception as e:
        handler.finish()
        raise e


def stream_with_callbacks(
    llm_invoke_fn: Callable,
    messages: list,
    on_token: Callable[[str], None] | None = None,
    on_start: Callable[[], None] | None = None,
    on_finish: Callable[[str, dict[str, Any]], None] | None = None
) -> str:
    """
    Stream LLM response using callbacks

    Args:
        llm_invoke_fn: LLM streaming invoke function
        messages: Messages to send to LLM
        on_token: Callback for each token
        on_start: Callback when streaming starts
        on_finish: Callback when streaming finishes

    Returns:
        Complete response text

    Example:
        >>> def on_token(token):
        ...     print(token, end="", flush=True)
        >>> response = stream_with_callbacks(llm.stream, messages, on_token=on_token)
    """
    handler = CallbackStreamingHandler(
        on_token=on_token,
        on_start=on_start,
        on_finish=on_finish
    )
    handler.start()

    try:
        for chunk in llm_invoke_fn(messages):
            if hasattr(chunk, 'content'):
                token = chunk.content
            elif isinstance(chunk, str):
                token = chunk
            else:
                token = str(chunk)

            if token:
                handler.update(token)

        handler.finish()
        return handler.get_text()

    except Exception as e:
        handler.finish()
        raise e
