"""File change tracking utilities for the agent.

This module provides helpers for tracking file modifications during agent execution.
"""

from pathlib import Path
from typing import Any

_IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", "logs", "temp"}


def get_workspace_files(workspace_path: str | None = None) -> dict[str, Any]:
    """Create a snapshot of workspace files.

    Args:
        workspace_path: Path to workspace directory (defaults to ".")

    Returns:
        Dictionary mapping relative file paths to their metadata (size, mtime)
    """
    if not workspace_path:
        workspace_path = "."

    workspace = Path(workspace_path)
    if not workspace.exists():
        return {}

    files_info = {}
    try:
        for file_path in workspace.rglob("*"):
            if not file_path.is_file():
                continue
            # Skip heavy directories
            if _IGNORE_DIRS & set(file_path.relative_to(workspace).parts):
                continue
            rel_path = str(file_path.relative_to(workspace))
            try:
                st = file_path.stat()  # Single stat() call
            except OSError:
                continue
            files_info[rel_path] = {
                "size": st.st_size,
                "mtime": st.st_mtime,
            }
    except Exception:
        pass

    return files_info


def detect_file_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    """Detect file changes between two snapshots.

    Args:
        before: Snapshot before changes
        after: Snapshot after changes

    Returns:
        Dictionary with 'added', 'modified', and 'deleted' file lists
    """
    changes = {
        "added": [],
        "modified": [],
        "deleted": []
    }

    # Added files
    for file_path in after:
        if file_path not in before:
            changes["added"].append(file_path)

    # Modified files (check both mtime and size for robustness)
    for file_path in after:
        if file_path in before:
            a, b = after[file_path], before[file_path]
            if a["mtime"] != b["mtime"] or a["size"] != b["size"]:
                changes["modified"].append(file_path)

    # Deleted files
    for file_path in before:
        if file_path not in after:
            changes["deleted"].append(file_path)

    return changes
