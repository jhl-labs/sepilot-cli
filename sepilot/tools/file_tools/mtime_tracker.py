"""File modification time tracker - OpenCode style file change detection.

Tracks file modification times to detect external changes and require re-reading
before editing. This prevents accidental overwrites of externally modified files.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FileState:
    """Tracked state of a file"""
    path: str
    mtime: float
    size: int
    read_time: float  # When the file was last read by the agent


class FileModificationTracker:
    """Tracks file modification times for conflict detection.

    OpenCode style:
    - Records mtime when file is read
    - Checks mtime before editing
    - Requires re-reading if file changed externally
    """

    def __init__(self):
        """Initialize tracker"""
        self._states: dict[str, FileState] = {}

    def record_read(self, file_path: str) -> FileState | None:
        """Record that a file was read.

        Args:
            file_path: Path to the file

        Returns:
            FileState if successful, None if file doesn't exist
        """
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return None

            stat = path.stat()
            state = FileState(
                path=str(path),
                mtime=stat.st_mtime,
                size=stat.st_size,
                read_time=stat.st_mtime,  # Use mtime as read_time
            )

            self._states[str(path)] = state
            logger.debug(f"Recorded file read: {file_path} (mtime: {stat.st_mtime})")

            return state

        except Exception as e:
            logger.warning(f"Failed to record file read: {file_path}: {e}")
            return None

    def check_for_changes(self, file_path: str) -> tuple[bool, str]:
        """Check if a file has been modified externally since last read.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (has_changed, message)
        """
        try:
            path = Path(file_path).resolve()
            path_str = str(path)

            # Check if we have recorded this file
            if path_str not in self._states:
                # File wasn't read yet - this is an issue
                return True, f"File has not been read: {file_path}. Read the file first before editing."

            recorded_state = self._states[path_str]

            # Check if file still exists
            if not path.exists():
                return True, f"File no longer exists: {file_path}"

            # Get current state
            stat = path.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size

            # Check for mtime change
            if current_mtime != recorded_state.mtime:
                return True, (
                    f"File has been modified externally: {file_path}\n"
                    f"  Recorded mtime: {recorded_state.mtime}\n"
                    f"  Current mtime: {current_mtime}\n"
                    f"Please re-read the file before editing."
                )

            # Check for size change (backup check)
            if current_size != recorded_state.size:
                return True, (
                    f"File size has changed: {file_path}\n"
                    f"  Recorded size: {recorded_state.size}\n"
                    f"  Current size: {current_size}\n"
                    f"Please re-read the file before editing."
                )

            return False, "File unchanged"

        except Exception as e:
            logger.warning(f"Failed to check file changes: {file_path}: {e}")
            return True, f"Error checking file: {e}"

    def update_after_write(self, file_path: str) -> FileState | None:
        """Update the recorded state after writing to a file.

        Args:
            file_path: Path to the file

        Returns:
            Updated FileState or None
        """
        return self.record_read(file_path)

    def forget(self, file_path: str) -> None:
        """Remove tracking for a file.

        Args:
            file_path: Path to the file
        """
        try:
            path = Path(file_path).resolve()
            self._states.pop(str(path), None)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all tracked file states."""
        self._states.clear()

    def get_state(self, file_path: str) -> FileState | None:
        """Get the recorded state for a file.

        Args:
            file_path: Path to the file

        Returns:
            FileState or None
        """
        try:
            path = Path(file_path).resolve()
            return self._states.get(str(path))
        except Exception:
            return None

    def list_tracked_files(self) -> list[str]:
        """List all tracked files.

        Returns:
            List of file paths
        """
        return list(self._states.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "tracked_files": len(self._states),
            "files": [
                {
                    "path": state.path,
                    "mtime": state.mtime,
                    "size": state.size,
                }
                for state in self._states.values()
            ],
        }


# Singleton instance
_tracker: FileModificationTracker | None = None


def get_file_tracker() -> FileModificationTracker:
    """Get or create the global file modification tracker.

    Returns:
        FileModificationTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = FileModificationTracker()
    return _tracker


def record_file_read(file_path: str) -> FileState | None:
    """Convenience function to record a file read.

    Args:
        file_path: Path to the file

    Returns:
        FileState or None
    """
    return get_file_tracker().record_read(file_path)


def check_file_modified(file_path: str) -> tuple[bool, str]:
    """Convenience function to check if a file was modified.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (has_changed, message)
    """
    return get_file_tracker().check_for_changes(file_path)


def update_file_after_write(file_path: str) -> FileState | None:
    """Convenience function to update tracking after write.

    Args:
        file_path: Path to the file

    Returns:
        FileState or None
    """
    return get_file_tracker().update_after_write(file_path)
