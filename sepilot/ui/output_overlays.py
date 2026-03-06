"""Output overlay displays for execution results and logs.

This module follows the Single Responsibility Principle (SRP) by handling
only the display of execution results and logs in overlay views.

Provides:
- Full-screen result overlay (Ctrl+O)
- Full-screen log viewer (Ctrl+L)
- Inline fallback displays
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, TextArea

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent


@dataclass
class ExecutionResult:
    """Holds the result of a command execution."""

    content: str = ""
    meta: str = "최근 실행 결과 없음"
    timestamp: datetime = field(default_factory=datetime.now)

    def is_empty(self) -> bool:
        """Check if result has content."""
        return not self.content.strip()


class OutputOverlayManager:
    """Manages output overlay displays.

    This class handles showing execution results and logs in
    full-screen overlay views with scrolling support.
    """

    def __init__(self, console: Console | None = None):
        """Initialize overlay manager.

        Args:
            console: Rich console for fallback output
        """
        self.console = console or Console()
        self.last_result = ExecutionResult()
        self.execution_logs: list[str] = []
        self.capturing_logs = False

    def update_result(self, content: str, meta: str | None = None) -> None:
        """Update the last execution result.

        Args:
            content: Result content
            meta: Result metadata/description
        """
        self.last_result = ExecutionResult(
            content=content.strip() if content else "",
            meta=meta or f"Agent response · {datetime.now().strftime('%H:%M:%S')}",
            timestamp=datetime.now(),
        )

    def add_log(self, text: str) -> None:
        """Add a log entry.

        Args:
            text: Log text
        """
        if text and text.strip():
            self.execution_logs.append(text)

    def clear_logs(self) -> None:
        """Clear execution logs."""
        self.execution_logs.clear()

    def show_last_result(self) -> None:
        """Display the last execution result."""
        if self.last_result.is_empty():
            self.console.print("[yellow]최근 실행 결과가 아직 없습니다.[/yellow]")
            return

        if HAS_PROMPT_TOOLKIT:
            self._open_result_overlay()
        else:
            self._show_result_inline()

    def show_execution_logs(self) -> None:
        """Display captured execution logs."""
        if not self.execution_logs:
            self.console.print("[yellow]실행 로그가 아직 없습니다.[/yellow]")
            return

        if HAS_PROMPT_TOOLKIT:
            self._open_logs_overlay()
        else:
            self._show_logs_inline()

    def _show_result_inline(self) -> None:
        """Fallback inline display for result."""
        self.console.print("[bold magenta]! Execution Results[/bold magenta]")
        self.console.print(f"[dim]{self.last_result.meta}[/dim]")
        self.console.print(self.last_result.content)

    def _show_logs_inline(self) -> None:
        """Fallback inline display for logs."""
        self.console.print("[bold cyan]📋 Execution Logs[/bold cyan]")
        for log_line in self.execution_logs[-100:]:
            self.console.print(log_line)

    def _open_result_overlay(self) -> None:
        """Open full-screen result overlay with scrolling."""
        overlay_text = (
            f"{self.last_result.meta}\n\n"
            f"{self.last_result.content}\n\n"
            "────────────────────────────\n"
            "Ctrl+O 또는 Esc 로 닫습니다 · ↑/↓ 로 스크롤 · 입력은 닫은 뒤 다시 가능합니다"
        )

        text_area = TextArea(
            text=overlay_text,
            read_only=True,
            scrollbar=True,
            wrap_lines=False,
        )

        frame = Frame(
            text_area,
            title="! Execution Results",
            style="class:result-frame",
        )

        kb = KeyBindings()

        @kb.add('escape')
        @kb.add('c-o')
        def close_overlay(event: 'KeyPressEvent') -> None:
            event.app.exit()

        style = Style.from_dict({
            "result-frame": "bg:#020617 #ffffff",
            "frame.label": "bg:#0f172a #94a3b8",
            "frame.border": "#334155",
            "textarea": "bg:#020617 #e2e8f0",
        })

        app = Application(
            layout=Layout(frame),
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
            style=style,
        )
        app.run()

    def _open_logs_overlay(self) -> None:
        """Open full-screen logs overlay with scrolling."""
        overlay_text = (
            "\n".join(self.execution_logs) +
            "\n\n────────────────────────────\n"
            "Ctrl+L 또는 Esc 로 닫습니다 · ↑/↓ 로 스크롤 · 입력은 닫은 뒤 다시 가능합니다"
        )

        text_area = TextArea(
            text=overlay_text,
            read_only=True,
            scrollbar=True,
            wrap_lines=False,
        )

        frame = Frame(
            text_area,
            title="📋 Execution Logs",
            style="class:log-frame",
        )

        kb = KeyBindings()

        @kb.add('escape')
        @kb.add('c-l')
        def close_overlay(event: 'KeyPressEvent') -> None:
            event.app.exit()

        style = Style.from_dict({
            "log-frame": "bg:#0c1821 #ffffff",
            "frame.label": "bg:#1b3a4b #94a3b8",
            "frame.border": "#2a5673",
            "textarea": "bg:#0c1821 #e2e8f0",
        })

        app = Application(
            layout=Layout(frame),
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
            style=style,
        )
        app.run()


class TeeOutput:
    """Capture output while also displaying it in real-time.

    This class implements a tee-like pattern where output is both
    displayed immediately and captured for later viewing.
    """

    def __init__(self, original_stream: Any, log_list: list[str]):
        """Initialize tee output.

        Args:
            original_stream: Original stdout/stderr
            log_list: List to append captured output to
        """
        self.original = original_stream
        self.log_list = log_list

    def write(self, text: str) -> None:
        """Write to both original stream and log.

        Args:
            text: Text to write
        """
        if self.original:
            self.original.write(text)
            self.original.flush()

        if text and text.strip():
            self.log_list.append(text)

    def flush(self) -> None:
        """Flush the original stream."""
        if self.original:
            self.original.flush()

    def isatty(self) -> bool:
        """Proxy isatty() to original stream (required by Rich for terminal detection)."""
        return hasattr(self.original, 'isatty') and self.original.isatty()

    def fileno(self) -> int:
        """Proxy fileno() to original stream."""
        if self.original and hasattr(self.original, 'fileno'):
            return self.original.fileno()
        import io
        raise io.UnsupportedOperation("fileno")

    @property
    def encoding(self) -> str:
        """Proxy encoding to original stream."""
        return getattr(self.original, 'encoding', 'utf-8')

    def writable(self) -> bool:
        """Report this stream as writable."""
        return True


# Convenience instance
_default_overlay_manager: OutputOverlayManager | None = None


def get_overlay_manager(console: Console | None = None) -> OutputOverlayManager:
    """Get or create an overlay manager instance.

    Args:
        console: Optional console for output

    Returns:
        OutputOverlayManager instance
    """
    global _default_overlay_manager
    if _default_overlay_manager is None or console is not None:
        _default_overlay_manager = OutputOverlayManager(console=console)
    return _default_overlay_manager
