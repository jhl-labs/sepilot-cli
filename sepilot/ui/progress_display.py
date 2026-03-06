"""Enhanced progress display and status panel for SE Pilot"""

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


class ProgressDisplay:
    """Enhanced progress bar with multiple tasks support"""

    def __init__(self, console: Console | None = None):
        """
        Initialize progress display

        Args:
            console: Rich console (if None, creates new one)
        """
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False  # Keep progress visible after completion
        )
        self.tasks: dict[str, Any] = {}
        self.started = False

    def start(self):
        """Start the progress display"""
        if not self.started:
            self.progress.start()
            self.started = True

    def stop(self):
        """Stop the progress display"""
        if self.started:
            self.progress.stop()
            self.started = False

    def add_task(self, name: str, description: str, total: int = 100) -> Any:
        """
        Add a new task to track

        Args:
            name: Unique task identifier
            description: Human-readable task description
            total: Total units of work (default: 100 for percentage)

        Returns:
            Task ID from rich.progress
        """
        task_id = self.progress.add_task(description, total=total)
        self.tasks[name] = {
            'id': task_id,
            'description': description,
            'total': total,
            'completed': 0
        }
        return task_id

    def update_task(self, name: str, advance: int = 1, description: str | None = None):
        """
        Update task progress

        Args:
            name: Task identifier
            advance: Amount to advance (default: 1)
            description: Optional new description
        """
        if name in self.tasks:
            task = self.tasks[name]
            task['completed'] += advance

            update_kwargs = {'advance': advance}
            if description:
                update_kwargs['description'] = description
                task['description'] = description

            self.progress.update(task['id'], **update_kwargs)

    def complete_task(self, name: str):
        """Mark task as complete"""
        if name in self.tasks:
            task = self.tasks[name]
            remaining = task['total'] - task['completed']
            if remaining > 0:
                self.progress.update(task['id'], advance=remaining)
            task['completed'] = task['total']

    def remove_task(self, name: str):
        """Remove a task from tracking"""
        if name in self.tasks:
            self.progress.remove_task(self.tasks[name]['id'])
            del self.tasks[name]


class StatusPanel:
    """Live status panel showing agent state, memory, and statistics"""

    def __init__(self, console: Console | None = None):
        """
        Initialize status panel

        Args:
            console: Rich console (if None, creates new one)
        """
        self.console = console or Console()
        self.layout = Layout()
        self.live: Live | None = None

        # Status data
        self.session_id: str = ""
        self.iteration: int = 0
        self.max_iterations: int = 0
        self.current_task: str = "Idle"
        self.memory_mb: float = 0.0
        self.cache_stats: dict[str, Any] = {}
        self.tool_stats: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.start_time: datetime = datetime.now()

    def start(self):
        """Start live display"""
        if self.live is None:
            self.live = Live(
                self._render_panel(),
                console=self.console,
                refresh_per_second=2,
                transient=False
            )
            self.live.start()

    def stop(self):
        """Stop live display"""
        if self.live:
            self.live.stop()
            self.live = None

    def update(
        self,
        session_id: str | None = None,
        iteration: int | None = None,
        max_iterations: int | None = None,
        current_task: str | None = None,
        memory_mb: float | None = None,
        cache_stats: dict[str, Any] | None = None,
        tool_stats: list[dict[str, Any]] | None = None,
        error: str | None = None
    ):
        """
        Update status panel data

        Args:
            session_id: Session identifier
            iteration: Current iteration
            max_iterations: Maximum iterations
            current_task: Current task description
            memory_mb: Memory usage in MB
            cache_stats: Cache hit/miss statistics
            tool_stats: Recent tool call statistics
            error: New error message to add
        """
        if session_id is not None:
            self.session_id = session_id
        if iteration is not None:
            self.iteration = iteration
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if current_task is not None:
            self.current_task = current_task
        if memory_mb is not None:
            self.memory_mb = memory_mb
        if cache_stats is not None:
            self.cache_stats = cache_stats
        if tool_stats is not None:
            self.tool_stats = tool_stats
        if error is not None:
            self.errors.append(error)
            # Keep only last 5 errors
            if len(self.errors) > 5:
                self.errors = self.errors[-5:]

        # Update live display
        if self.live:
            self.live.update(self._render_panel())

    def _render_panel(self) -> Panel:
        """Render the status panel"""
        # Calculate elapsed time
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds

        # Create main table
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")

        # Session info
        table.add_row("Session:", self.session_id or "N/A")
        table.add_row("Elapsed:", elapsed_str)

        # Progress info
        if self.max_iterations > 0:
            progress_pct = (self.iteration / self.max_iterations) * 100
            table.add_row(
                "Progress:",
                f"{self.iteration}/{self.max_iterations} ({progress_pct:.0f}%)"
            )
        else:
            table.add_row("Iteration:", str(self.iteration))

        # Current task
        table.add_row("Task:", self.current_task)

        # Memory usage
        if self.memory_mb > 0:
            color = "green" if self.memory_mb < 500 else "yellow" if self.memory_mb < 1000 else "red"
            table.add_row("Memory:", f"[{color}]{self.memory_mb:.1f} MB[/{color}]")

        # Cache stats
        if self.cache_stats:
            hit_rate = self.cache_stats.get('hit_rate_pct', 0)
            color = "green" if hit_rate > 50 else "yellow" if hit_rate > 25 else "white"
            table.add_row(
                "Cache:",
                f"[{color}]{hit_rate:.1f}% hit rate[/{color}] ({self.cache_stats.get('hits', 0)} hits, {self.cache_stats.get('misses', 0)} misses)"
            )

        # Tool stats (recent 3)
        if self.tool_stats:
            recent_tools = self.tool_stats[-3:]
            tool_summary = ", ".join(f"{t['name']}({t.get('duration', 0):.1f}s)" for t in recent_tools)
            table.add_row("Recent Tools:", tool_summary)

        # Errors
        if self.errors:
            error_text = Text()
            error_text.append("Last Errors:\n", style="bold red")
            for i, error in enumerate(self.errors[-3:], 1):
                error_text.append(f"  {i}. {error[:60]}...\n" if len(error) > 60 else f"  {i}. {error}\n", style="red")
            table.add_row("", error_text)

        return Panel(
            table,
            title="[bold]SE Pilot Status[/bold]",
            border_style="blue",
        )

    def __enter__(self):
        """Context manager support"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()
        return False
