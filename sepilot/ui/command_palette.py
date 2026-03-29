"""Command palette metadata and prompt_toolkit layout helpers.

Provides a fuzzy search interface for quickly finding and executing commands.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.widgets import Frame
from rich.console import Console

from sepilot.ui.command_catalog import iter_palette_commands


@dataclass
class PaletteCommand:
    """A command in the command palette"""
    name: str  # Command name (e.g., "/help")
    description: str  # Short description
    category: str = "General"  # Category for grouping
    handler: Callable | None = None  # Function to execute
    keywords: list[str] = field(default_factory=list)  # Additional search keywords


class CommandPalette:
    """Command palette with fuzzy search.

    Features:
    - Searchable command list
    - Fuzzy search across command names and descriptions
    - Arrow keys to navigate
    - Enter to execute
    - Escape to close
    """

    def __init__(
        self,
        commands: list[PaletteCommand] | None = None,
        console: Console | None = None,
    ):
        """Initialize command palette.

        Args:
            commands: List of available commands
            console: Rich console for output
        """
        self.commands = commands or []
        self.console = console or Console()
        self._visible = False
        self._selected_index = 0
        self._filtered_commands: list[PaletteCommand] = []
        self._search_buffer = Buffer(
            name="palette_search",
            on_text_changed=self._on_search_changed,
        )
        self._result_callback: Callable[[str], None] | None = None

    def add_command(self, command: PaletteCommand) -> None:
        """Add a command to the palette."""
        self.commands.append(command)

    def add_commands_from_dict(self, commands: dict[str, str]) -> None:
        """Add commands from a simple dict of name -> description."""
        for name, desc in commands.items():
            self.commands.append(PaletteCommand(name=name, description=desc))

    def show(self, callback: Callable[[str], None] | None = None) -> None:
        """Show the command palette."""
        self._visible = True
        self._selected_index = 0
        self._search_buffer.reset()
        self._filtered_commands = self.commands.copy()
        self._result_callback = callback

    def hide(self) -> None:
        """Hide the command palette."""
        self._visible = False
        self._search_buffer.reset()

    def is_visible(self) -> bool:
        """Check if palette is visible."""
        return self._visible

    def _on_search_changed(self, buffer: Buffer) -> None:
        """Handle search text changes."""
        query = buffer.text.lower().strip()
        if not query:
            self._filtered_commands = self.commands.copy()
        else:
            self._filtered_commands = self._fuzzy_search(query)
        self._selected_index = 0

    def _fuzzy_search(self, query: str) -> list[PaletteCommand]:
        """Fuzzy search commands.

        Args:
            query: Search query

        Returns:
            Filtered and sorted commands
        """
        results = []
        for cmd in self.commands:
            score = self._fuzzy_score(query, cmd)
            if score > 0:
                results.append((score, cmd))

        # Sort by score (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        return [cmd for _, cmd in results]

    def _fuzzy_score(self, query: str, cmd: PaletteCommand) -> int:
        """Calculate fuzzy match score.

        Args:
            query: Search query
            cmd: Command to score

        Returns:
            Match score (0 = no match)
        """
        score = 0
        name_lower = cmd.name.lower()
        desc_lower = cmd.description.lower()
        keywords = " ".join(cmd.keywords).lower()

        # Exact name match
        if query == name_lower:
            return 1000

        # Name starts with query
        if name_lower.startswith(query):
            score += 100

        # Name contains query
        if query in name_lower:
            score += 50

        # Description contains query
        if query in desc_lower:
            score += 25

        # Keywords contain query
        if query in keywords:
            score += 30

        # Character-by-character fuzzy match
        query_idx = 0
        for char in name_lower:
            if query_idx < len(query) and char == query[query_idx]:
                query_idx += 1
                score += 5

        # All query chars found in order
        if query_idx == len(query):
            score += 20

        return score

    def _get_selected_command(self) -> PaletteCommand | None:
        """Get currently selected command."""
        if self._filtered_commands and 0 <= self._selected_index < len(self._filtered_commands):
            return self._filtered_commands[self._selected_index]
        return None

    def execute_selected(self) -> str | None:
        """Execute the selected command.

        Returns:
            Command name if executed, None otherwise
        """
        cmd = self._get_selected_command()
        if cmd:
            self.hide()
            if cmd.handler:
                cmd.handler()
            if self._result_callback:
                self._result_callback(cmd.name)
            return cmd.name
        return None

    def move_selection(self, delta: int) -> None:
        """Move selection up or down.

        Args:
            delta: -1 for up, +1 for down
        """
        if not self._filtered_commands:
            return
        self._selected_index = (self._selected_index + delta) % len(self._filtered_commands)

    def get_key_bindings(self) -> KeyBindings:
        """Get key bindings for the palette."""
        kb = KeyBindings()

        @kb.add('escape')
        def _(event):
            self.hide()

        @kb.add('enter')
        def _(event):
            self.execute_selected()

        @kb.add('up')
        @kb.add('c-p')
        def _(event):
            self.move_selection(-1)

        @kb.add('down')
        @kb.add('c-n')
        def _(event):
            self.move_selection(+1)

        return kb

    def get_layout(self) -> ConditionalContainer:
        """Get the prompt_toolkit layout for the palette."""
        # Create the results display
        def get_results_text():
            lines = []
            max_display = 10

            if not self._filtered_commands:
                return HTML('<ansigray>No commands found</ansigray>')

            for i, cmd in enumerate(self._filtered_commands[:max_display]):
                if i == self._selected_index:
                    # Selected item
                    lines.append(
                        f'<ansibrightcyan><b>> {cmd.name}</b></ansibrightcyan>'
                        f'  <ansiwhite>{cmd.description}</ansiwhite>'
                    )
                else:
                    lines.append(
                        f'<ansicyan>  {cmd.name}</ansicyan>'
                        f'  <ansigray>{cmd.description}</ansigray>'
                    )

            if len(self._filtered_commands) > max_display:
                remaining = len(self._filtered_commands) - max_display
                lines.append(f'<ansigray>  ... and {remaining} more</ansigray>')

            return HTML('\n'.join(lines))

        # Search input
        search_window = Window(
            content=BufferControl(buffer=self._search_buffer),
            height=1,
        )

        # Results list
        results_window = Window(
            content=FormattedTextControl(get_results_text),
            height=12,
        )

        # Combined layout in a frame
        palette_body = HSplit([
            Window(
                content=FormattedTextControl(HTML('<b>Command Palette</b> <ansigray>(type to search)</ansigray>')),
                height=1,
            ),
            Window(char='-', height=1),
            search_window,
            Window(char='-', height=1),
            results_window,
        ])

        palette_frame = Frame(
            body=palette_body,
            title='Ctrl+X Ctrl+P',
            width=60,
        )

        return ConditionalContainer(
            content=FloatContainer(
                content=Window(),  # Dummy background
                floats=[
                    Float(
                        content=palette_frame,
                        top=2,
                        left=10,
                    )
                ]
            ),
            filter=Condition(lambda: self._visible),
        )


def create_default_palette(command_handlers: dict[str, Callable] | None = None) -> CommandPalette:
    """Create a command palette with default SE Pilot commands.

    Args:
        command_handlers: Dict of command name -> handler function

    Returns:
        Configured CommandPalette
    """
    handlers = command_handlers or {}

    palette = CommandPalette()
    for entry in iter_palette_commands():
        palette.add_command(
            PaletteCommand(
                name=entry.name,
                description=entry.description,
                category=entry.category,
                handler=handlers.get(entry.name),
                keywords=list(entry.keywords),
            )
        )

    return palette
