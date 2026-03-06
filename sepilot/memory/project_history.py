"""Project-based History Management for SE Pilot (Claude Code style)

This module provides functionality to:
- Organize data by project directory (like Claude Code's ~/.claude/projects/)
- Manage per-project sessions and history
- Encode project paths safely for filesystem storage
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote


@dataclass
class ProjectInfo:
    """Metadata about a project"""
    project_path: str  # Original absolute path
    encoded_name: str  # Filesystem-safe encoded name
    created_at: str
    last_accessed: str
    session_count: int = 0
    total_events: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionInfo:
    """Summary information about a session"""
    session_id: str
    project_path: str
    started_at: str
    ended_at: str | None
    event_count: int
    summary: str | None = None
    git_branch: str | None = None


class ProjectHistoryManager:
    """Manages project-specific history storage (Claude Code style)

    Directory structure:
    ~/.sepilot/
    ├── projects/
    │   └── {encoded_project_path}/
    │       ├── project.json         # Project metadata
    │       ├── sessions/            # Session event logs
    │       │   └── {session_id}.jsonl
    │       ├── file-history/        # File snapshots per checkpoint
    │       │   └── {checkpoint_id}/
    │       │       └── {file_hash}.snapshot
    │       └── env-snapshots/       # Environment snapshots
    │           └── {checkpoint_id}.json
    """

    def __init__(self, base_dir: Path | None = None):
        """Initialize the project history manager.

        Args:
            base_dir: Base directory for storage (default: ~/.sepilot)
        """
        if base_dir is None:
            base_dir = Path.home() / ".sepilot"

        self.base_dir = Path(base_dir)
        self.projects_dir = self.base_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        # Cache for project info
        self._project_cache: dict[str, ProjectInfo] = {}

    def encode_project_path(self, path: str) -> str:
        """Encode a project path for filesystem storage.

        Similar to Claude Code's approach: replace / with - and handle special chars.

        Args:
            path: Absolute path to project directory

        Returns:
            Filesystem-safe encoded string
        """
        # Normalize and get absolute path
        abs_path = os.path.abspath(os.path.expanduser(path))

        # Replace path separators with dashes
        encoded = abs_path.replace(os.sep, '-')

        # Remove leading dash if present (from root /)
        if encoded.startswith('-'):
            encoded = encoded[1:]

        # URL-encode any remaining special characters
        # But keep alphanumeric, dash, underscore, dot
        encoded = re.sub(r'[^\w\-.]', lambda m: quote(m.group(0)), encoded)

        return encoded

    def decode_project_path(self, encoded: str) -> str:
        """Decode an encoded project path back to original.

        Args:
            encoded: Encoded project path

        Returns:
            Original absolute path
        """
        # URL-decode first
        decoded = unquote(encoded)

        # Replace dashes with path separator
        # Be careful: we need to distinguish between dashes that were
        # originally in the path vs dashes that replaced /
        # For simplicity, assume all dashes were /
        decoded = decoded.replace('-', os.sep)

        # Add leading separator (root)
        if not decoded.startswith(os.sep):
            decoded = os.sep + decoded

        return decoded

    def get_project_dir(self, project_path: str) -> Path:
        """Get the storage directory for a project.

        Args:
            project_path: Absolute path to project directory

        Returns:
            Path to project's storage directory
        """
        encoded = self.encode_project_path(project_path)
        project_dir = self.projects_dir / encoded

        # Create subdirectories
        (project_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (project_dir / "file-history").mkdir(parents=True, exist_ok=True)
        (project_dir / "env-snapshots").mkdir(parents=True, exist_ok=True)

        return project_dir

    def get_or_create_project(self, project_path: str) -> ProjectInfo:
        """Get or create project info for a directory.

        Args:
            project_path: Absolute path to project directory

        Returns:
            ProjectInfo for the project
        """
        abs_path = os.path.abspath(project_path)

        # Check cache first
        if abs_path in self._project_cache:
            return self._project_cache[abs_path]

        project_dir = self.get_project_dir(abs_path)
        project_file = project_dir / "project.json"

        if project_file.exists():
            # Load existing project info
            try:
                with open(project_file, encoding='utf-8') as f:
                    data = json.load(f)

                info = ProjectInfo(
                    project_path=data['project_path'],
                    encoded_name=data['encoded_name'],
                    created_at=data['created_at'],
                    last_accessed=datetime.now().isoformat(),
                    session_count=data.get('session_count', 0),
                    total_events=data.get('total_events', 0),
                    tags=data.get('tags', [])
                )
            except (json.JSONDecodeError, KeyError):
                # Corrupt file, recreate
                info = self._create_project_info(abs_path)
        else:
            # Create new project info
            info = self._create_project_info(abs_path)

        # Update last accessed and save
        info.last_accessed = datetime.now().isoformat()
        self._save_project_info(info)

        # Cache it
        self._project_cache[abs_path] = info

        return info

    def _create_project_info(self, project_path: str) -> ProjectInfo:
        """Create new project info.

        Args:
            project_path: Absolute path to project

        Returns:
            New ProjectInfo instance
        """
        encoded = self.encode_project_path(project_path)
        now = datetime.now().isoformat()

        return ProjectInfo(
            project_path=project_path,
            encoded_name=encoded,
            created_at=now,
            last_accessed=now,
            session_count=0,
            total_events=0,
            tags=[]
        )

    def _save_project_info(self, info: ProjectInfo) -> None:
        """Save project info to disk.

        Args:
            info: ProjectInfo to save
        """
        project_dir = self.get_project_dir(info.project_path)
        project_file = project_dir / "project.json"

        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(info), f, indent=2, ensure_ascii=False)

    def get_session_file(self, project_path: str, session_id: str) -> Path:
        """Get the path to a session's JSONL file.

        Args:
            project_path: Absolute path to project
            session_id: Session ID

        Returns:
            Path to session file
        """
        project_dir = self.get_project_dir(project_path)
        return project_dir / "sessions" / f"{session_id}.jsonl"

    def get_file_history_dir(self, project_path: str, checkpoint_id: str) -> Path:
        """Get the directory for file snapshots of a checkpoint.

        Args:
            project_path: Absolute path to project
            checkpoint_id: Checkpoint ID

        Returns:
            Path to checkpoint's file history directory
        """
        project_dir = self.get_project_dir(project_path)
        history_dir = project_dir / "file-history" / checkpoint_id
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir

    def get_env_snapshot_file(self, project_path: str, checkpoint_id: str) -> Path:
        """Get the path to an environment snapshot file.

        Args:
            project_path: Absolute path to project
            checkpoint_id: Checkpoint ID

        Returns:
            Path to environment snapshot file
        """
        project_dir = self.get_project_dir(project_path)
        return project_dir / "env-snapshots" / f"{checkpoint_id}.json"

    def list_projects(self) -> list[ProjectInfo]:
        """List all projects with stored history.

        Returns:
            List of ProjectInfo for all projects
        """
        projects = []

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_file = project_dir / "project.json"
            if not project_file.exists():
                continue

            try:
                with open(project_file, encoding='utf-8') as f:
                    data = json.load(f)

                projects.append(ProjectInfo(
                    project_path=data['project_path'],
                    encoded_name=data['encoded_name'],
                    created_at=data['created_at'],
                    last_accessed=data.get('last_accessed', data['created_at']),
                    session_count=data.get('session_count', 0),
                    total_events=data.get('total_events', 0),
                    tags=data.get('tags', [])
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by last accessed (most recent first)
        projects.sort(key=lambda p: p.last_accessed, reverse=True)

        return projects

    def get_project_sessions(self, project_path: str) -> list[SessionInfo]:
        """Get all sessions for a project.

        Args:
            project_path: Absolute path to project

        Returns:
            List of SessionInfo sorted by start time (newest first)
        """
        project_dir = self.get_project_dir(project_path)
        sessions_dir = project_dir / "sessions"

        sessions = []

        for session_file in sessions_dir.glob("*.jsonl"):
            session_id = session_file.stem

            # Read first and last lines to get timing info
            try:
                with open(session_file, encoding='utf-8') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                first_event = json.loads(lines[0])
                last_event = json.loads(lines[-1])

                sessions.append(SessionInfo(
                    session_id=session_id,
                    project_path=project_path,
                    started_at=first_event.get('timestamp', ''),
                    ended_at=last_event.get('timestamp') if len(lines) > 1 else None,
                    event_count=len(lines),
                    summary=first_event.get('data', {}).get('summary'),
                    git_branch=first_event.get('data', {}).get('git_branch')
                ))
            except (json.JSONDecodeError, IndexError):
                continue

        # Sort by start time (newest first)
        sessions.sort(key=lambda s: s.started_at, reverse=True)

        return sessions

    def increment_session_count(self, project_path: str) -> None:
        """Increment the session count for a project.

        Args:
            project_path: Absolute path to project
        """
        info = self.get_or_create_project(project_path)
        info.session_count += 1
        self._save_project_info(info)

    def add_event_count(self, project_path: str, count: int = 1) -> None:
        """Add to the total event count for a project.

        Args:
            project_path: Absolute path to project
            count: Number of events to add
        """
        info = self.get_or_create_project(project_path)
        info.total_events += count
        self._save_project_info(info)

    def cleanup_old_sessions(
        self,
        project_path: str | None = None,
        days: int = 30
    ) -> dict[str, int]:
        """Remove sessions older than specified days.

        Args:
            project_path: Specific project to clean (None for all projects)
            days: Keep sessions newer than this many days

        Returns:
            Dict of project_path -> removed session count
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        results: dict[str, int] = {}

        if project_path:
            projects = [self.get_or_create_project(project_path)]
        else:
            projects = self.list_projects()

        for project in projects:
            project_dir = self.get_project_dir(project.project_path)
            sessions_dir = project_dir / "sessions"
            removed = 0

            for session_file in sessions_dir.glob("*.jsonl"):
                if session_file.stat().st_mtime < cutoff:
                    session_file.unlink()
                    removed += 1

            if removed > 0:
                results[project.project_path] = removed
                # Update session count
                project.session_count = max(0, project.session_count - removed)
                self._save_project_info(project)

        return results

    def get_current_project_path(self) -> str:
        """Get the current working directory as project path.

        Returns:
            Absolute path to current directory
        """
        return os.path.abspath(os.getcwd())

    def delete_project_history(self, project_path: str) -> bool:
        """Delete all history for a project.

        Args:
            project_path: Absolute path to project

        Returns:
            True if deleted successfully
        """
        import shutil

        project_dir = self.get_project_dir(project_path)

        if project_dir.exists():
            shutil.rmtree(project_dir)

            # Remove from cache
            abs_path = os.path.abspath(project_path)
            if abs_path in self._project_cache:
                del self._project_cache[abs_path]

            return True

        return False


# Global instance for convenience
_default_manager: ProjectHistoryManager | None = None


def get_project_history_manager() -> ProjectHistoryManager:
    """Get the default project history manager instance.

    Returns:
        Default ProjectHistoryManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ProjectHistoryManager()
    return _default_manager
