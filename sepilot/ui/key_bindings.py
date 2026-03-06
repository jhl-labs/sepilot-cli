"""Key bindings configuration for SE Pilot interactive mode.

This module follows the Single Responsibility Principle (SRP) by handling
only key binding creation and configuration.

Provides:
- Custom key bindings for prompt_toolkit
- @ reference completion triggers
- Shortcut keys (Ctrl+O, Ctrl+L, Ctrl+K, F1, F2)
- Double-escape rewind trigger
"""

import os
import subprocess
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

try:
    from prompt_toolkit.key_binding import KeyBindings
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    KeyBindings = None

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent


class KeyBindingsManager:
    """Manages key bindings for the interactive prompt.

    This class encapsulates all key binding logic, making it easy to
    extend or modify keyboard shortcuts without touching the main
    interactive mode code.
    """

    def __init__(
        self,
        on_show_result: Callable[[], None] | None = None,
        on_show_logs: Callable[[], None] | None = None,
        on_rewind: Callable[[str], None] | None = None,
        on_command_palette: Callable[[], None] | None = None,
    ):
        """Initialize key bindings manager.

        Args:
            on_show_result: Callback to show last execution result (Ctrl+O)
            on_show_logs: Callback to show execution logs (Ctrl+L)
            on_rewind: Callback to trigger rewind command (Esc+Esc)
            on_command_palette: Callback to show command palette (Ctrl+K)
        """
        self._on_show_result = on_show_result
        self._on_show_logs = on_show_logs
        self._on_rewind = on_rewind
        self._on_command_palette = on_command_palette
        self._last_escape_time = 0.0

    def create_key_bindings(self) -> 'KeyBindings | None':
        """Create custom key bindings for prompt_toolkit.

        Returns:
            KeyBindings instance or None if prompt_toolkit not available
        """
        if not HAS_PROMPT_TOOLKIT:
            return None

        kb = KeyBindings()

        # @ key: Start file reference and trigger completion
        @kb.add('@')
        def handle_at(event: 'KeyPressEvent') -> None:
            """When @ is pressed, insert it and trigger completion menu."""
            event.current_buffer.insert_text('@')
            event.current_buffer.start_completion(select_first=False)

        # Ctrl+O: Show last execution output
        @kb.add('c-o')
        def handle_ctrl_o(event: 'KeyPressEvent') -> None:
            """Show last execution output (Claude Code style)."""
            if self._on_show_result:
                self._on_show_result()

        # Ctrl+L: Show execution logs
        @kb.add('c-l')
        def handle_ctrl_l(event: 'KeyPressEvent') -> None:
            """Show execution logs with scroll support."""
            if self._on_show_logs:
                self._on_show_logs()

        # Esc: Double-escape to trigger rewind
        @kb.add('escape')
        def handle_escape(event: 'KeyPressEvent') -> None:
            """Double-escape to trigger rewind menu."""
            current_time = time.time()
            buffer = event.current_buffer

            # If completion menu is open, just close it
            if buffer.complete_state is not None:
                buffer.complete_state = None
                self._last_escape_time = 0
                return

            # If two escapes within 0.5 seconds, trigger rewind
            # Use "both" mode directly to avoid raw input() during prompt_toolkit session
            if current_time - self._last_escape_time < 0.5:
                buffer.reset()
                if self._on_rewind:
                    self._on_rewind("/rewind both")
                self._last_escape_time = 0
            else:
                self._last_escape_time = current_time

        # Alphanumeric keys: Re-trigger completion if in @ context
        for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:':
            @kb.add(char)
            def handle_char(event: 'KeyPressEvent', char: str = char) -> None:
                """When typing after @, keep completion menu active."""
                buffer = event.current_buffer
                buffer.insert_text(char)

                if _is_in_file_reference_context(buffer):
                    buffer.start_completion(select_first=False)

        # / key: Directory navigation in @ context
        @kb.add('/')
        def handle_slash(event: 'KeyPressEvent') -> None:
            """When / is pressed after @, enter directory and refresh completion."""
            text = event.current_buffer.document.text
            cursor_pos = event.current_buffer.cursor_position

            if '@' in text and text.rfind('@') < cursor_pos:
                event.current_buffer.insert_text('/')
                event.current_buffer.start_completion(select_first=False)
            else:
                event.current_buffer.insert_text('/')

        # Enter: Only submit if completion menu is NOT active
        @kb.add('enter')
        def handle_enter(event: 'KeyPressEvent') -> None:
            """Handle Enter key: insert completion if menu is open, otherwise submit."""
            buffer = event.current_buffer

            if buffer.complete_state is not None:
                current_completion = buffer.complete_state.current_completion
                if current_completion:
                    buffer.apply_completion(current_completion)
                    buffer.complete_state = None

                    # If completed a directory, show contents
                    if buffer.document.text.rstrip().endswith('/'):
                        buffer.start_completion(select_first=False)
            else:
                buffer.validate_and_handle()

        # Alt+Enter: Submit multi-line input
        @kb.add('escape', 'enter')
        def handle_alt_enter(event: 'KeyPressEvent') -> None:
            """Submit multi-line input."""
            event.current_buffer.validate_and_handle()

        # Ctrl+K: Command palette (OpenCode style)
        @kb.add('c-k')
        def handle_ctrl_k(event: 'KeyPressEvent') -> None:
            """Open command palette (OpenCode style)."""
            if self._on_command_palette:
                self._on_command_palette()

        # Ctrl+Shift+K: Clear screen (moved from Ctrl+K)
        @kb.add('c-x', 'c-k')
        def handle_ctrl_x_k(event: 'KeyPressEvent') -> None:
            """Clear screen."""
            if os.name == 'nt':
                subprocess.run(["cmd", "/c", "cls"], check=False)
            else:
                subprocess.run(["clear"], check=False)

        # F1: Show help
        @kb.add('f1')
        def handle_f1(event: 'KeyPressEvent') -> None:
            """Show help via /help command."""
            event.current_buffer.text = '/help'
            event.current_buffer.validate_and_handle()

        # F2: Show status
        @kb.add('f2')
        def handle_f2(event: 'KeyPressEvent') -> None:
            """Show status via /status command."""
            event.current_buffer.text = '/status'
            event.current_buffer.validate_and_handle()

        return kb


def _is_in_file_reference_context(buffer: Any) -> bool:
    """Check if cursor is after an @ symbol for file reference.

    Args:
        buffer: prompt_toolkit buffer

    Returns:
        True if currently typing a file reference
    """
    text = buffer.document.text_before_cursor
    if '@' not in text:
        return False

    last_at_pos = text.rfind('@')
    after_at = text[last_at_pos + 1:]
    return ' ' not in after_at


def create_key_bindings(
    on_show_result: Callable[[], None] | None = None,
    on_show_logs: Callable[[], None] | None = None,
    on_rewind: Callable[[str], None] | None = None,
    on_command_palette: Callable[[], None] | None = None,
) -> 'KeyBindings | None':
    """Factory function to create key bindings.

    Convenience function that creates a KeyBindingsManager and returns
    the configured key bindings.

    Args:
        on_show_result: Callback for Ctrl+O
        on_show_logs: Callback for Ctrl+L
        on_rewind: Callback for Esc+Esc
        on_command_palette: Callback for Ctrl+K

    Returns:
        KeyBindings instance or None
    """
    manager = KeyBindingsManager(
        on_show_result=on_show_result,
        on_show_logs=on_show_logs,
        on_rewind=on_rewind,
        on_command_palette=on_command_palette,
    )
    return manager.create_key_bindings()
