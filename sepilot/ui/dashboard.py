"""TUI Dashboard for SE Pilot using Textual

Provides a rich terminal user interface with:
- Real-time status display
- Log output area
- Interactive controls
- Session information
"""

import contextlib
from collections.abc import Callable
from datetime import datetime

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Label, RichLog, Static
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


if HAS_TEXTUAL:
    class StatusWidget(Static):
        """Widget displaying agent status"""

        session_id: reactive[str] = reactive("N/A")
        iteration: reactive[int] = reactive(0)
        max_iterations: reactive[int] = reactive(0)
        current_task: reactive[str] = reactive("Idle")
        memory_mb: reactive[float] = reactive(0.0)
        cache_hit_rate: reactive[float] = reactive(0.0)

        def compose(self) -> ComposeResult:
            """Create child widgets"""
            yield Label("📊 Status", id="status-title")
            yield Label(id="session-info")
            yield Label(id="progress-info")
            yield Label(id="task-info")
            yield Label(id="resource-info")

        def watch_session_id(self, session_id: str) -> None:
            """Update when session_id changes"""
            self.query_one("#session-info", Label).update(f"Session: {session_id}")

        def watch_iteration(self, iteration: int) -> None:
            """Update when iteration changes"""
            self._update_progress()

        def watch_max_iterations(self, max_iterations: int) -> None:
            """Update when max_iterations changes"""
            self._update_progress()

        def watch_current_task(self, current_task: str) -> None:
            """Update when current_task changes"""
            self.query_one("#task-info", Label).update(f"Task: {current_task}")

        def watch_memory_mb(self, memory_mb: float) -> None:
            """Update when memory_mb changes"""
            self._update_resources()

        def watch_cache_hit_rate(self, cache_hit_rate: float) -> None:
            """Update when cache_hit_rate changes"""
            self._update_resources()

        def _update_progress(self) -> None:
            """Update progress display"""
            if self.max_iterations > 0:
                pct = (self.iteration / self.max_iterations) * 100
                self.query_one("#progress-info", Label).update(
                    f"Progress: {self.iteration}/{self.max_iterations} ({pct:.0f}%)"
                )
            else:
                self.query_one("#progress-info", Label).update(
                    f"Iteration: {self.iteration}"
                )

        def _update_resources(self) -> None:
            """Update resource display"""
            parts = []
            if self.memory_mb > 0:
                parts.append(f"Memory: {self.memory_mb:.1f}MB")
            if self.cache_hit_rate > 0:
                parts.append(f"Cache: {self.cache_hit_rate:.0f}%")

            if parts:
                self.query_one("#resource-info", Label).update(" | ".join(parts))
            else:
                self.query_one("#resource-info", Label).update("Resources: N/A")


    class DashboardApp(App):
        """TUI Dashboard application"""

        CSS = """
        Screen {
            layout: grid;
            grid-size: 2 3;
            grid-rows: auto 1fr auto;
        }

        Header {
            column-span: 2;
        }

        Footer {
            column-span: 2;
        }

        #status-panel {
            width: 30;
            border: solid green;
            padding: 1;
        }

        #log-panel {
            border: solid blue;
            padding: 1;
        }

        #status-title {
            text-style: bold;
            background: green;
            color: white;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("c", "clear_logs", "Clear Logs"),
            ("s", "save_logs", "Save Logs"),
        ]

        def __init__(
            self,
            session_id: str = "N/A",
            on_quit: Callable | None = None
        ):
            super().__init__()
            self.session_id = session_id
            self.on_quit_callback = on_quit
            self.start_time = datetime.now()

        def compose(self) -> ComposeResult:
            """Create child widgets"""
            yield Header(show_clock=True)

            # Status panel (left)
            with Container(id="status-panel"):
                yield StatusWidget(id="status")

            # Log panel (right)
            with Container(id="log-panel"):
                yield Label("📝 Logs", classes="panel-title")
                yield RichLog(id="logs", wrap=True, highlight=True, markup=True)

            yield Footer()

        def on_mount(self) -> None:
            """Called when app is mounted"""
            self.title = f"SE Pilot Dashboard - {self.session_id}"
            self.sub_title = f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"

            # Initialize status widget
            status = self.query_one("#status", StatusWidget)
            status.session_id = self.session_id

        def action_quit(self) -> None:
            """Quit the application"""
            if self.on_quit_callback:
                self.on_quit_callback()
            self.exit()

        def action_clear_logs(self) -> None:
            """Clear log panel"""
            logs = self.query_one("#logs", RichLog)
            logs.clear()

        def action_save_logs(self) -> None:
            """Save logs to file"""
            # TODO: Implement log saving
            self.add_log("[yellow]Log saving not yet implemented[/yellow]")

        def update_status(
            self,
            iteration: int | None = None,
            max_iterations: int | None = None,
            current_task: str | None = None,
            memory_mb: float | None = None,
            cache_hit_rate: float | None = None
        ) -> None:
            """Update status widget"""
            try:
                status = self.query_one("#status", StatusWidget)

                if iteration is not None:
                    status.iteration = iteration
                if max_iterations is not None:
                    status.max_iterations = max_iterations
                if current_task is not None:
                    status.current_task = current_task
                if memory_mb is not None:
                    status.memory_mb = memory_mb
                if cache_hit_rate is not None:
                    status.cache_hit_rate = cache_hit_rate
            except Exception:
                # Widget may not be mounted yet
                pass

        def add_log(self, message: str) -> None:
            """Add message to log panel"""
            try:
                logs = self.query_one("#logs", RichLog)
                logs.write(message)
            except Exception:
                # Widget may not be mounted yet
                pass


class DashboardManager:
    """Manages TUI dashboard in a separate thread"""

    def __init__(self, session_id: str = "N/A"):
        """
        Initialize dashboard manager

        Args:
            session_id: Session identifier
        """
        if not HAS_TEXTUAL:
            raise ImportError(
                "Textual is required for dashboard. "
                "Install with: pip install textual"
            )

        self.session_id = session_id
        self.app: DashboardApp | None = None
        self.running = False

    def start(self) -> None:
        """Start the dashboard application"""
        if self.running:
            return

        self.app = DashboardApp(
            session_id=self.session_id,
            on_quit=self.stop
        )
        self.running = True

    def stop(self) -> None:
        """Stop the dashboard application"""
        self.running = False
        if self.app:
            with contextlib.suppress(Exception):
                self.app.exit()

    def update_status(self, **kwargs) -> None:
        """Update dashboard status"""
        if self.app and self.running:
            self.app.update_status(**kwargs)

    def add_log(self, message: str) -> None:
        """Add log message to dashboard"""
        if self.app and self.running:
            self.app.add_log(message)

    def run(self) -> None:
        """Run the dashboard (blocking)"""
        if self.app:
            self.app.run()


def create_dashboard(session_id: str = "N/A") -> DashboardManager | None:
    """
    Create a dashboard manager

    Args:
        session_id: Session identifier

    Returns:
        DashboardManager instance or None if Textual not available
    """
    if not HAS_TEXTUAL:
        return None

    return DashboardManager(session_id=session_id)


def is_dashboard_available() -> bool:
    """Check if dashboard functionality is available"""
    return HAS_TEXTUAL
