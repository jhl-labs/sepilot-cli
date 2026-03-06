"""File Checkpointing System for tracking and reverting file changes (Claude Code style)

This module provides functionality to:
- Track file changes made during a session
- Create checkpoints at each user prompt
- Revert file changes to specific checkpoints
- Store checkpoints in project-specific directories (Claude Code style)
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sepilot.memory.project_history import ProjectHistoryManager


@dataclass
class FileSnapshot:
    """Snapshot of a file at a specific point in time"""
    path: str
    content: str | None  # None if file was deleted
    hash: str
    exists: bool
    timestamp: str


@dataclass
class Checkpoint:
    """A checkpoint containing file snapshots"""
    id: str
    timestamp: str
    message_index: int  # Index in conversation
    user_prompt: str  # First 100 chars of user prompt
    files: dict[str, FileSnapshot]  # path -> snapshot


class FileCheckpointManager:
    """Manages file checkpoints for rewind functionality.

    Supports two storage modes:
    1. Legacy: ~/.sepilot/checkpoints/ (default, backward compatible)
    2. Project-based: ~/.sepilot/projects/{project}/file-history/ (Claude Code style)
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        max_checkpoints: int = 50,
        project_path: str | None = None,
        use_project_storage: bool = False
    ):
        """Initialize the checkpoint manager.

        Args:
            storage_dir: Directory to store checkpoint data (legacy mode)
            max_checkpoints: Maximum number of checkpoints to keep
            project_path: Project directory path (for project-based storage)
            use_project_storage: If True, use project-based storage
        """
        self.use_project_storage = use_project_storage
        self.project_path = project_path or os.getcwd()
        self._project_manager: 'ProjectHistoryManager | None' = None

        # Determine storage directory
        if use_project_storage:
            self.storage_dir = self._get_project_storage_dir()
        elif storage_dir is None:
            self.storage_dir = Path.home() / ".sepilot" / "checkpoints"
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.checkpoints: list[Checkpoint] = []
        self.tracked_files: dict[str, FileSnapshot] = {}  # Current tracked files
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load existing checkpoints for this session
        self._load_session()

    def _get_project_manager(self) -> 'ProjectHistoryManager | None':
        """Lazy-load project history manager."""
        if self._project_manager is None:
            try:
                from sepilot.memory.project_history import get_project_history_manager
                self._project_manager = get_project_history_manager()
            except ImportError:
                pass
        return self._project_manager

    def _get_project_storage_dir(self) -> Path:
        """Get project-specific storage directory."""
        manager = self._get_project_manager()
        if manager:
            project_dir = manager.get_project_dir(self.project_path)
            return project_dir / "file-history"
        # Fallback to legacy storage
        return Path.home() / ".sepilot" / "checkpoints"

    def _compute_hash(self, content: str) -> str:
        """Compute hash of file content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def _snapshot_file(self, path: str) -> FileSnapshot:
        """Create a snapshot of a file"""
        abs_path = os.path.abspath(path)

        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            try:
                with open(abs_path, encoding='utf-8') as f:
                    content = f.read()
                return FileSnapshot(
                    path=abs_path,
                    content=content,
                    hash=self._compute_hash(content),
                    exists=True,
                    timestamp=datetime.now().isoformat()
                )
            except (OSError, UnicodeDecodeError):
                # Binary or unreadable file - track existence only
                return FileSnapshot(
                    path=abs_path,
                    content=None,
                    hash="binary",
                    exists=True,
                    timestamp=datetime.now().isoformat()
                )
        else:
            return FileSnapshot(
                path=abs_path,
                content=None,
                hash="",
                exists=False,
                timestamp=datetime.now().isoformat()
            )

    def track_file(self, path: str) -> None:
        """Start tracking a file for changes"""
        snapshot = self._snapshot_file(path)
        self.tracked_files[snapshot.path] = snapshot

    def track_files(self, paths: list[str]) -> None:
        """Track multiple files"""
        for path in paths:
            self.track_file(path)

    def create_checkpoint(self, message_index: int, user_prompt: str) -> Checkpoint:
        """Create a new checkpoint with current file states.

        Args:
            message_index: Index of the user message in conversation
            user_prompt: The user's prompt text

        Returns:
            The created checkpoint
        """
        checkpoint_id = f"cp_{datetime.now().strftime('%H%M%S')}_{len(self.checkpoints)}"

        # Update snapshots for all tracked files
        current_files = {}
        for path in self.tracked_files:
            snapshot = self._snapshot_file(path)
            current_files[path] = snapshot

        checkpoint = Checkpoint(
            id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            message_index=message_index,
            user_prompt=user_prompt[:100] if user_prompt else "",
            files=current_files
        )

        self.checkpoints.append(checkpoint)

        # Trim old checkpoints if exceeding limit
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]

        # Save session
        self._save_session()

        return checkpoint

    def get_changed_files(self, since_checkpoint: Checkpoint | None = None) -> dict[str, dict[str, Any]]:
        """Get files that changed since a checkpoint.

        Args:
            since_checkpoint: Compare against this checkpoint (default: last checkpoint)

        Returns:
            Dict of path -> {old_hash, new_hash, change_type}
        """
        if not since_checkpoint:
            if not self.checkpoints:
                return {}
            since_checkpoint = self.checkpoints[-1]

        changes = {}

        # Check all tracked files
        all_paths = set(self.tracked_files.keys()) | set(since_checkpoint.files.keys())

        for path in all_paths:
            old_snapshot = since_checkpoint.files.get(path)
            new_snapshot = self._snapshot_file(path)

            if old_snapshot is None and new_snapshot.exists:
                changes[path] = {
                    "type": "created",
                    "old_hash": None,
                    "new_hash": new_snapshot.hash
                }
            elif old_snapshot and not new_snapshot.exists:
                changes[path] = {
                    "type": "deleted",
                    "old_hash": old_snapshot.hash,
                    "new_hash": None
                }
            elif old_snapshot and new_snapshot.exists and old_snapshot.hash != new_snapshot.hash:
                changes[path] = {
                    "type": "modified",
                    "old_hash": old_snapshot.hash,
                    "new_hash": new_snapshot.hash
                }

        return changes

    def revert_to_checkpoint(self, checkpoint: Checkpoint) -> dict[str, str]:
        """Revert files to a checkpoint state.

        Args:
            checkpoint: The checkpoint to revert to

        Returns:
            Dict of path -> result message
        """
        results = {}

        for path, snapshot in checkpoint.files.items():
            try:
                if snapshot.exists and snapshot.content is not None:
                    # Restore file content
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(snapshot.content)
                    results[path] = "restored"
                elif not snapshot.exists:
                    # File didn't exist at checkpoint - delete if exists now
                    if os.path.exists(path):
                        os.remove(path)
                        results[path] = "deleted"
                    else:
                        results[path] = "unchanged"
                else:
                    # Binary file or no content - skip
                    results[path] = "skipped (binary)"
            except Exception as e:
                results[path] = f"error: {str(e)}"

        return results

    def revert_by_count(self, count: int = 1) -> dict[str, str] | None:
        """Revert to N checkpoints ago.

        Args:
            count: Number of checkpoints to go back

        Returns:
            Results dict or None if not enough checkpoints
        """
        if len(self.checkpoints) < count + 1:
            return None

        target_checkpoint = self.checkpoints[-(count + 1)]
        return self.revert_to_checkpoint(target_checkpoint)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints with summary info"""
        return [
            {
                "id": cp.id,
                "timestamp": cp.timestamp,
                "message_index": cp.message_index,
                "user_prompt": cp.user_prompt,
                "file_count": len(cp.files)
            }
            for cp in self.checkpoints
        ]

    def _save_session(self) -> None:
        """Save session data to disk"""
        session_file = self.storage_dir / f"session_{self.session_id}.json"

        data = {
            "session_id": self.session_id,
            "checkpoints": [
                {
                    "id": cp.id,
                    "timestamp": cp.timestamp,
                    "message_index": cp.message_index,
                    "user_prompt": cp.user_prompt,
                    "files": {
                        path: asdict(snapshot)
                        for path, snapshot in cp.files.items()
                    }
                }
                for cp in self.checkpoints
            ],
            "tracked_files": list(self.tracked_files.keys())
        }

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_session(self) -> None:
        """Load session data from disk (most recent)"""
        session_files = sorted(self.storage_dir.glob("session_*.json"), reverse=True)

        if not session_files:
            return

        # Load most recent session
        try:
            with open(session_files[0], encoding='utf-8') as f:
                data = json.load(f)

            self.session_id = data.get("session_id", self.session_id)

            for cp_data in data.get("checkpoints", []):
                files = {}
                for path, snap_data in cp_data.get("files", {}).items():
                    files[path] = FileSnapshot(**snap_data)

                checkpoint = Checkpoint(
                    id=cp_data["id"],
                    timestamp=cp_data["timestamp"],
                    message_index=cp_data["message_index"],
                    user_prompt=cp_data["user_prompt"],
                    files=files
                )
                self.checkpoints.append(checkpoint)

            for path in data.get("tracked_files", []):
                self.tracked_files[path] = self._snapshot_file(path)

        except Exception:
            # If loading fails, start fresh
            pass

    def cleanup_old_sessions(self, keep_days: int = 30) -> int:
        """Remove session files older than specified days.

        Args:
            keep_days: Keep sessions newer than this many days

        Returns:
            Number of sessions removed
        """
        cutoff = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        removed = 0

        for session_file in self.storage_dir.glob("session_*.json"):
            if session_file.stat().st_mtime < cutoff:
                session_file.unlink()
                removed += 1

        return removed

    def get_project_info(self) -> dict[str, Any]:
        """Get information about the project storage.

        Returns:
            Dict with project storage information
        """
        return {
            "storage_dir": str(self.storage_dir),
            "project_path": self.project_path,
            "use_project_storage": self.use_project_storage,
            "session_id": self.session_id,
            "checkpoint_count": len(self.checkpoints),
            "tracked_file_count": len(self.tracked_files)
        }


# Convenience functions

def create_project_checkpoint_manager(
    project_path: str | None = None,
    max_checkpoints: int = 50
) -> FileCheckpointManager:
    """Create a checkpoint manager with project-based storage.

    Args:
        project_path: Project directory (default: current directory)
        max_checkpoints: Maximum checkpoints to keep

    Returns:
        FileCheckpointManager with project-based storage
    """
    return FileCheckpointManager(
        project_path=project_path,
        max_checkpoints=max_checkpoints,
        use_project_storage=True
    )


def create_legacy_checkpoint_manager(
    storage_dir: Path | None = None,
    max_checkpoints: int = 50
) -> FileCheckpointManager:
    """Create a checkpoint manager with legacy storage.

    Args:
        storage_dir: Storage directory (default: ~/.sepilot/checkpoints)
        max_checkpoints: Maximum checkpoints to keep

    Returns:
        FileCheckpointManager with legacy storage
    """
    return FileCheckpointManager(
        storage_dir=storage_dir,
        max_checkpoints=max_checkpoints,
        use_project_storage=False
    )
