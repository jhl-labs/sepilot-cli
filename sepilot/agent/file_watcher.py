"""File watcher for detecting external file changes during agent execution.

Uses watchdog to monitor the project directory and report changes that
were NOT made by the agent itself (i.e. changes from an external editor).
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# Default patterns to ignore (build artifacts, caches, VCS)
DEFAULT_IGNORE_PATTERNS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
    ".tox",
    ".checkpoints",
}

# File extensions to watch
WATCHED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".yaml", ".yml", ".json", ".toml",
    ".md", ".txt", ".sql", ".sh", ".bash", ".css", ".html",
}


class _ChangeCollector(FileSystemEventHandler):
    """Collects file change events, filtering out agent-owned writes."""

    def __init__(self, ignore_patterns: set[str], watched_extensions: set[str]):
        super().__init__()
        self._ignore_patterns = ignore_patterns
        self._watched_extensions = watched_extensions
        self._lock = threading.Lock()
        self._changes: dict[str, str] = {}  # path -> event_type
        # Paths the agent is currently writing — these are suppressed
        self._agent_paths: set[str] = set()

    def _should_ignore(self, path: str) -> bool:
        parts = Path(path).parts
        for part in parts:
            if part in self._ignore_patterns:
                return True
        ext = Path(path).suffix.lower()
        if ext and ext not in self._watched_extensions:
            return True
        return False

    # -- watchdog callbacks --------------------------------------------------

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._record(event.src_path, "modified")

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._record(event.src_path, "created")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._record(event.src_path, "deleted")

    # -- internal ------------------------------------------------------------

    def _record(self, path: str, event_type: str) -> None:
        if self._should_ignore(path):
            return
        resolved = str(Path(path).resolve())
        with self._lock:
            if resolved in self._agent_paths:
                return  # Agent-owned write, skip
            self._changes[resolved] = event_type

    # -- public API ----------------------------------------------------------

    def mark_agent_write(self, path: str) -> None:
        """Mark a path as being written by the agent (suppress next event)."""
        resolved = str(Path(path).resolve())
        with self._lock:
            self._agent_paths.add(resolved)

    def unmark_agent_write(self, path: str) -> None:
        """Remove the agent-write suppression for a path."""
        resolved = str(Path(path).resolve())
        with self._lock:
            self._agent_paths.discard(resolved)

    def drain_changes(self) -> dict[str, str]:
        """Return and clear all collected external changes.

        Returns:
            Dict mapping absolute file paths to event types
        """
        with self._lock:
            changes = dict(self._changes)
            self._changes.clear()
            return changes


class FileWatcher:
    """Watches a project directory for external file changes.

    Usage::

        watcher = FileWatcher("/path/to/project")
        watcher.start()

        # Before agent writes a file:
        watcher.mark_agent_write("/path/to/project/foo.py")
        # ... agent writes ...
        watcher.unmark_agent_write("/path/to/project/foo.py")

        # Periodically check for external changes:
        changes = watcher.get_external_changes()
        # {'src/bar.py': 'modified', 'src/new.py': 'created'}

        watcher.stop()
    """

    def __init__(
        self,
        project_path: str,
        ignore_patterns: set[str] | None = None,
        watched_extensions: set[str] | None = None,
    ):
        self._project_path = str(Path(project_path).resolve())
        self._ignore = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        self._extensions = watched_extensions or WATCHED_EXTENSIONS
        self._collector = _ChangeCollector(self._ignore, self._extensions)
        self._observer: Observer | None = None
        self._running = False

    def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return
        try:
            self._observer = Observer()
            self._observer.schedule(
                self._collector, self._project_path, recursive=True
            )
            self._observer.daemon = True
            self._observer.start()
            self._running = True
            logger.debug(f"FileWatcher started for {self._project_path}")
        except Exception as e:
            logger.warning(f"Failed to start FileWatcher: {e}")
            self._running = False

    def stop(self) -> None:
        """Stop watching."""
        if not self._running or not self._observer:
            return
        try:
            self._observer.stop()
            self._observer.join(timeout=3.0)
        except Exception:
            pass
        self._running = False
        self._observer = None
        logger.debug("FileWatcher stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def mark_agent_write(self, path: str) -> None:
        """Suppress change events for a path the agent is about to write."""
        self._collector.mark_agent_write(path)

    def unmark_agent_write(self, path: str) -> None:
        """Re-enable change detection for a path after agent write completes."""
        self._collector.unmark_agent_write(path)

    def get_external_changes(self) -> dict[str, str]:
        """Get all external file changes since last call.

        Returns:
            Dict of {absolute_path: event_type} for external changes
        """
        if not self._running:
            return {}
        return self._collector.drain_changes()

    def format_changes_for_agent(self, changes: dict[str, str] | None = None) -> str | None:
        """Format external changes as a human-readable notification.

        Args:
            changes: Changes dict (fetched automatically if None)

        Returns:
            Formatted string or None if no changes
        """
        if changes is None:
            changes = self.get_external_changes()
        if not changes:
            return None

        project = Path(self._project_path)
        lines = ["[External file changes detected]"]
        for abs_path, event_type in sorted(changes.items()):
            try:
                rel = str(Path(abs_path).relative_to(project))
            except ValueError:
                rel = abs_path
            lines.append(f"  {event_type}: {rel}")

        return "\n".join(lines)
