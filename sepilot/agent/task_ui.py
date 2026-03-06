"""Task UI - Rich Live-based real-time task status display.

Claude Code-style features:
- Real-time progress bar and status updates
- Parallel task visualization
- Spinner animations for running tasks
- Clean summary on completion
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from sepilot.agent.task_manager import TaskManager, TodoItem, TodoStatus

if TYPE_CHECKING:
    from sepilot.agent.task_registry import TaskInfo, TaskRegistry

logger = logging.getLogger(__name__)


class TaskProgressUI:
    """Real-time task progress display using Rich Live.

    Example:
        manager = TaskManager()

        with TaskProgressUI(manager) as ui:
            manager.add_todo("Task 1", "Working on task 1")
            manager.start_todo("todo_1")
            await do_work()
            manager.complete_todo("todo_1")

        # Or with explicit control
        ui = TaskProgressUI(manager)
        ui.start()
        # ... do work ...
        ui.stop()
    """

    def __init__(
        self,
        manager: TaskManager,
        console: Console | None = None,
        refresh_rate: float = 4.0,  # Updates per second
        show_parallel: bool = True,
        compact: bool = False
    ):
        """Initialize task progress UI.

        Args:
            manager: TaskManager to display
            console: Rich console (creates new if None)
            refresh_rate: UI refresh rate (Hz)
            show_parallel: Show parallel task details
            compact: Use compact display mode
        """
        self.manager = manager
        self.console = console or Console()
        self.refresh_rate = refresh_rate
        self.show_parallel = show_parallel
        self.compact = compact

        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_ids: dict[str, TaskID] = {}
        self._running = False
        self._lock = threading.Lock()

        # Register as progress listener
        self.manager.add_progress_listener(self._on_todo_change)

    def start(self) -> None:
        """Start the live display."""
        if self._running:
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            refresh_per_second=self.refresh_rate
        )

        self._live = Live(
            self._generate_display(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
            vertical_overflow="visible"
        )

        self._running = True
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if not self._running:
            return

        self._running = False

        if self._live:
            self._live.stop()
            self._live = None

        # Print final summary
        self._print_summary()

    def update(self) -> None:
        """Force update the display."""
        if self._live and self._running:
            self._live.update(self._generate_display())

    def _generate_display(self) -> Panel:
        """Generate the current display panel."""
        if self.compact:
            return self._generate_compact_display()
        return self._generate_full_display()

    def _generate_full_display(self) -> Panel:
        """Generate full display with all details."""
        stats = self.manager.get_stats()
        current = self.manager.get_current_todo()

        # Build content
        content_parts = []

        # Header with progress
        header = self._create_progress_header(stats)
        content_parts.append(header)

        # Current task (if any)
        if current:
            current_text = Text()
            current_text.append("\n🔄 ", style="bold yellow")
            current_text.append(current.active_form, style="bold")
            content_parts.append(current_text)

        # Todo list
        if self.manager.get_all_todos():
            todo_table = self._create_todo_table()
            content_parts.append(todo_table)

        # Parallel tasks (if enabled and running)
        if self.show_parallel:
            parallel_info = self._create_parallel_info()
            if parallel_info:
                content_parts.append(parallel_info)

        # Combine all parts
        combined = Group(*content_parts)

        return Panel(
            combined,
            title="[bold cyan]📋 Task Progress[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    def _generate_compact_display(self) -> Panel:
        """Generate compact one-line display."""
        stats = self.manager.get_stats()
        current = self.manager.get_current_todo()

        text = Text()

        # Progress
        text.append(f"[{stats['completed']}/{stats['total']}] ", style="bold cyan")

        # Current task
        if current:
            text.append(current.active_form, style="bold")
            if current.metadata.get("progress"):
                progress = current.metadata["progress"]
                text.append(f" ({progress:.0%})", style="dim")
        else:
            text.append("Waiting...", style="dim")

        return Panel(text, border_style="cyan", padding=(0, 1))

    def _create_progress_header(self, stats: dict[str, Any]) -> Text:
        """Create the progress header."""
        text = Text()

        total = stats['total']
        completed = stats['completed']
        in_progress = stats['in_progress']
        failed = stats['failed']

        # Progress bar
        if total > 0:
            bar_width = 30
            filled = int((completed / total) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            text.append(f"[{bar}] ", style="cyan")

        # Stats
        text.append(f"{completed}/{total}", style="bold green")
        if in_progress > 0:
            text.append(f" | {in_progress} running", style="yellow")
        if failed > 0:
            text.append(f" | {failed} failed", style="red")

        return text

    def _create_todo_table(self) -> Table:
        """Create todo list table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Status", width=3)
        table.add_column("Task")

        for todo in self.manager.get_root_todos():
            icon, style = self._get_status_icon_and_style(todo.status)
            table.add_row(icon, Text(todo.content, style=style))

            # Show children with indent
            for child_id in todo.children_ids:
                child = self.manager.get_todo(child_id)
                if child:
                    child_icon, child_style = self._get_status_icon_and_style(child.status)
                    table.add_row(f"  {child_icon}", Text(child.content, style=child_style))

        return table

    def _get_status_icon_and_style(
        self,
        status: TodoStatus
    ) -> tuple[str, str]:
        """Get icon and style for status."""
        icons = {
            TodoStatus.PENDING: ("⏳", "dim"),
            TodoStatus.IN_PROGRESS: ("🔄", "bold yellow"),
            TodoStatus.COMPLETED: ("✅", "green"),
            TodoStatus.FAILED: ("❌", "red"),
            TodoStatus.SKIPPED: ("⏭️", "dim"),
        }
        return icons.get(status, ("❓", ""))

    def _create_parallel_info(self) -> Table | None:
        """Create parallel task information."""
        registry = self.manager.registry
        running = registry.get_running_tasks()

        if not running:
            return None

        table = Table(show_header=True, box=None, padding=(0, 1), title="Parallel Tasks")
        table.add_column("Task", style="bold")
        table.add_column("Progress", justify="right")
        table.add_column("Time", justify="right")

        for task in running:
            progress_str = f"{task.progress:.0%}" if task.progress > 0 else "..."
            elapsed = task.elapsed_time()
            time_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"

            table.add_row(
                task.name[:30],
                progress_str,
                time_str
            )

        return table

    def _on_todo_change(self, todo: TodoItem) -> None:
        """Handle todo change events."""
        if not self._running:
            return

        with self._lock:
            # Update progress task if exists
            if self._progress and todo.id in self._task_ids:
                task_id = self._task_ids[todo.id]
                progress = todo.metadata.get("progress", 0.0)

                if todo.status == TodoStatus.COMPLETED:
                    self._progress.update(task_id, completed=100)
                elif todo.status == TodoStatus.IN_PROGRESS:
                    self._progress.update(task_id, completed=progress * 100)

        self.update()

    def _print_summary(self) -> None:
        """Print final summary after stopping."""
        stats = self.manager.get_stats()

        self.console.print()
        self.console.print(Panel.fit(
            f"[bold green]✅ Completed[/bold green]: {stats['completed']}/{stats['total']} tasks",
            border_style="green"
        ))

        # Show failed tasks if any
        failed = [t for t in self.manager.get_all_todos() if t.status == TodoStatus.FAILED]
        if failed:
            self.console.print("[bold red]Failed tasks:[/bold red]")
            for todo in failed:
                self.console.print(f"  ❌ {todo.content}: {todo.error}")

    def __enter__(self) -> TaskProgressUI:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class ParallelTaskUI:
    """Specialized UI for parallel task execution.

    Shows each parallel task with its own progress bar.
    """

    def __init__(
        self,
        registry: TaskRegistry,
        console: Console | None = None,
        title: str = "Parallel Execution"
    ):
        """Initialize parallel task UI.

        Args:
            registry: Task registry to display
            console: Rich console
            title: Panel title
        """
        from sepilot.agent.task_registry import get_task_registry

        self.registry = registry or get_task_registry()
        self.console = console or Console()
        self.title = title

        self._progress: Progress | None = None
        self._task_map: dict[str, TaskID] = {}
        self._live: Live | None = None

    def start(self) -> None:
        """Start the parallel task display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[name]}"),
            BarColumn(bar_width=20),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
            console=self.console
        )

        self._live = Live(
            Panel(self._progress, title=f"[bold cyan]{self.title}[/bold cyan]"),
            console=self.console,
            refresh_per_second=4
        )

        # Register listener
        self.registry.add_listener(self._on_task_change)

        self._live.start()

    def stop(self) -> None:
        """Stop the display."""
        self.registry.remove_listener(self._on_task_change)

        if self._live:
            self._live.stop()

        # Print summary
        stats = self.registry.get_stats()
        self.console.print(f"\n[bold]Completed:[/bold] {stats['completed']}/{stats['total']}")
        if stats['failed'] > 0:
            self.console.print(f"[bold red]Failed:[/bold red] {stats['failed']}")

    def _on_task_change(self, task: TaskInfo, event: str) -> None:
        """Handle task state changes."""
        if not self._progress:
            return

        from sepilot.agent.task_registry import TaskState

        if event == "registered":
            # Add new task to progress
            task_id = self._progress.add_task(
                "",
                total=100,
                name=task.name[:25],
                status="pending"
            )
            self._task_map[task.task_id] = task_id

        elif task.task_id in self._task_map:
            progress_id = self._task_map[task.task_id]

            # Update status text
            status_text = {
                TaskState.PENDING: "[dim]pending[/dim]",
                TaskState.RUNNING: "[yellow]running[/yellow]",
                TaskState.COMPLETED: "[green]✓[/green]",
                TaskState.FAILED: "[red]✗[/red]",
                TaskState.CANCELLED: "[dim]cancelled[/dim]",
            }.get(task.state, "")

            # Update progress
            if task.state == TaskState.COMPLETED:
                self._progress.update(progress_id, completed=100, status=status_text)
            elif task.state == TaskState.RUNNING:
                self._progress.update(
                    progress_id,
                    completed=task.progress * 100,
                    status=status_text
                )
            else:
                self._progress.update(progress_id, status=status_text)

    def __enter__(self) -> ParallelTaskUI:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class SimpleProgressBar:
    """Simple progress bar for non-live contexts.

    Use when Live display isn't appropriate (CI, logging, etc.)
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        width: int = 30,
        console: Console | None = None
    ):
        """Initialize simple progress bar.

        Args:
            total: Total steps
            description: Progress description
            width: Bar width in characters
            console: Rich console
        """
        self.total = total
        self.description = description
        self.width = width
        self.console = console or Console()
        self.current = 0

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current = min(self.total, self.current + n)
        self._print()

    def set(self, value: int) -> None:
        """Set progress to specific value."""
        self.current = min(self.total, max(0, value))
        self._print()

    def _print(self) -> None:
        """Print current progress."""
        if self.total <= 0:
            return

        pct = self.current / self.total
        filled = int(pct * self.width)
        bar = "█" * filled + "░" * (self.width - filled)

        text = f"\r{self.description} [{bar}] {self.current}/{self.total} ({pct:.0%})"
        self.console.print(text, end="")

    def finish(self) -> None:
        """Finish and print newline."""
        self.current = self.total
        self._print()
        self.console.print()


@contextmanager
def task_progress_context(
    manager: TaskManager,
    console: Console | None = None,
    compact: bool = False
):
    """Context manager for task progress display.

    Example:
        with task_progress_context(manager) as ui:
            manager.add_todo("Task 1", "Working on task 1")
            await do_work()
    """
    ui = TaskProgressUI(manager, console=console, compact=compact)
    ui.start()
    try:
        yield ui
    finally:
        ui.stop()


def print_task_summary(manager: TaskManager, console: Console | None = None) -> None:
    """Print a static task summary.

    Args:
        manager: TaskManager to summarize
        console: Rich console
    """
    console = console or Console()

    console.print()
    console.print(Panel.fit(
        manager.format_todos(),
        title="[bold cyan]📋 Task Summary[/bold cyan]",
        border_style="cyan"
    ))

    stats = manager.get_stats()
    console.print(f"\n[bold]Completion:[/bold] {stats['completion_rate']:.1f}%")
