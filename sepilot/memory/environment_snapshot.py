"""Environment Variable Snapshot Management for SE Pilot

This module provides functionality to:
- Capture environment variable snapshots at checkpoints
- Mask sensitive values (API keys, passwords, tokens)
- Compare environment states between checkpoints
- Support environment restoration for debugging
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EnvironmentSnapshot:
    """Snapshot of environment state at a specific point"""
    checkpoint_id: str
    timestamp: str

    # Environment variables (masked for sensitive ones)
    variables: dict[str, str]
    masked_keys: list[str]  # Keys that were masked

    # Critical system info
    python_version: str
    platform_info: str
    working_directory: str

    # Git state
    git_branch: str | None = None
    git_commit: str | None = None
    git_dirty: bool = False

    # Virtual environment info
    virtual_env: str | None = None
    conda_env: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvironmentManager:
    """Manages environment variable snapshots

    Features:
    - Automatic detection and masking of sensitive variables
    - Capture critical environment info (Python version, git state, etc.)
    - Compare snapshots to detect changes
    - Safe storage without exposing secrets
    """

    # Patterns for detecting sensitive environment variable names
    SECRET_PATTERNS = [
        r'.*API[_-]?KEY.*',
        r'.*SECRET.*',
        r'.*PASSWORD.*',
        r'.*TOKEN.*',
        r'.*CREDENTIAL.*',
        r'.*PRIVATE[_-]?KEY.*',
        r'.*AUTH.*',
        r'.*ACCESS[_-]?KEY.*',
        r'.*AWS.*KEY.*',
        r'.*AZURE.*KEY.*',
        r'.*GCP.*KEY.*',
        r'.*DATABASE.*URL.*',
        r'.*DB[_-]?PASS.*',
        r'.*OPENAI.*',
        r'.*ANTHROPIC.*',
        r'.*GITHUB.*TOKEN.*',
        r'.*GITLAB.*TOKEN.*',
        r'.*NPM.*TOKEN.*',
        r'.*PYPI.*TOKEN.*',
    ]

    # Critical environment variables to always capture
    CRITICAL_VARS = [
        'PATH',
        'PYTHONPATH',
        'HOME',
        'USER',
        'SHELL',
        'LANG',
        'LC_ALL',
        'VIRTUAL_ENV',
        'CONDA_DEFAULT_ENV',
        'CONDA_PREFIX',
        'PWD',
        'TERM',
        'EDITOR',
        'VISUAL',
        'GIT_AUTHOR_NAME',
        'GIT_AUTHOR_EMAIL',
        'GIT_COMMITTER_NAME',
        'GIT_COMMITTER_EMAIL',
    ]

    # Variables to completely exclude (never store)
    EXCLUDE_VARS = [
        'LS_COLORS',
        'LESS_TERMCAP_*',
        'XDG_*',
        'DBUS_*',
        'SESSION_*',
        'DISPLAY',
        'WAYLAND_*',
        'SSH_*',  # Exclude SSH agent info
    ]

    def __init__(self, storage_dir: Path | None = None):
        """Initialize the environment manager.

        Args:
            storage_dir: Directory to store snapshots (default: ~/.sepilot/env-snapshots)
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".sepilot" / "env-snapshots"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Compile regex patterns for efficiency
        self._secret_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SECRET_PATTERNS
        ]
        self._exclude_patterns = [
            re.compile(pattern.replace('*', '.*'), re.IGNORECASE)
            for pattern in self.EXCLUDE_VARS
        ]

    def _is_secret(self, key: str) -> bool:
        """Check if a variable name looks like a secret.

        Args:
            key: Environment variable name

        Returns:
            True if the variable appears to contain sensitive data
        """
        for pattern in self._secret_patterns:
            if pattern.match(key):
                return True
        return False

    def _should_exclude(self, key: str) -> bool:
        """Check if a variable should be completely excluded.

        Args:
            key: Environment variable name

        Returns:
            True if the variable should not be stored
        """
        for pattern in self._exclude_patterns:
            if pattern.match(key):
                return True
        return False

    def _mask_value(self, value: str) -> str:
        """Mask a sensitive value while preserving some info.

        Args:
            value: Original value

        Returns:
            Masked value showing length and hash prefix
        """
        if not value:
            return "[EMPTY]"

        # Create a hash for comparison purposes
        value_hash = hashlib.sha256(value.encode()).hexdigest()[:8]

        # Show length and hash prefix
        return f"[MASKED:len={len(value)},hash={value_hash}]"

    def _get_git_info(self) -> tuple[str | None, str | None, bool]:
        """Get current git branch and commit.

        Returns:
            Tuple of (branch, commit_hash, is_dirty)
        """
        try:
            # Get current branch
            branch_result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

            # Get current commit
            commit_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            commit = commit_result.stdout.strip()[:12] if commit_result.returncode == 0 else None

            # Check if dirty
            dirty_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                timeout=5
            )
            is_dirty = bool(dirty_result.stdout.strip()) if dirty_result.returncode == 0 else False

            return branch, commit, is_dirty

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None, None, False

    def capture_snapshot(
        self,
        checkpoint_id: str,
        include_all: bool = False,
        working_dir: str | None = None
    ) -> EnvironmentSnapshot:
        """Capture current environment snapshot.

        Args:
            checkpoint_id: ID to associate with this snapshot
            include_all: If True, include all env vars (not just critical ones)
            working_dir: Working directory (default: current directory)

        Returns:
            EnvironmentSnapshot with current state
        """
        timestamp = datetime.now().isoformat()
        variables: dict[str, str] = {}
        masked_keys: list[str] = []

        # Get all environment variables
        for key, value in os.environ.items():
            # Skip excluded variables
            if self._should_exclude(key):
                continue

            # Determine if this variable should be included
            is_critical = key in self.CRITICAL_VARS

            if not include_all and not is_critical:
                # Skip non-critical variables unless include_all
                continue

            # Mask sensitive values
            if self._is_secret(key):
                variables[key] = self._mask_value(value)
                masked_keys.append(key)
            else:
                variables[key] = value

        # Get git info
        git_branch, git_commit, git_dirty = self._get_git_info()

        # Determine virtual environment
        virtual_env = os.environ.get('VIRTUAL_ENV')
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')

        return EnvironmentSnapshot(
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            variables=variables,
            masked_keys=masked_keys,
            python_version=sys.version,
            platform_info=platform.platform(),
            working_directory=working_dir or os.getcwd(),
            git_branch=git_branch,
            git_commit=git_commit,
            git_dirty=git_dirty,
            virtual_env=virtual_env,
            conda_env=conda_env,
            metadata={
                'python_executable': sys.executable,
                'python_path': sys.path[:5],  # First 5 entries
            }
        )

    def save_snapshot(
        self,
        snapshot: EnvironmentSnapshot,
        file_path: Path | None = None
    ) -> Path:
        """Save snapshot to disk.

        Args:
            snapshot: Snapshot to save
            file_path: Optional specific path (default: auto-generated)

        Returns:
            Path where snapshot was saved
        """
        if file_path is None:
            file_path = self.storage_dir / f"{snapshot.checkpoint_id}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(snapshot), f, indent=2, ensure_ascii=False)

        return file_path

    def load_snapshot(self, file_path: Path) -> EnvironmentSnapshot | None:
        """Load snapshot from disk.

        Args:
            file_path: Path to snapshot file

        Returns:
            EnvironmentSnapshot or None if not found/invalid
        """
        try:
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)

            return EnvironmentSnapshot(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None

    def compare_snapshots(
        self,
        snapshot1: EnvironmentSnapshot,
        snapshot2: EnvironmentSnapshot
    ) -> dict[str, dict[str, Any]]:
        """Compare two snapshots and find differences.

        Args:
            snapshot1: First (older) snapshot
            snapshot2: Second (newer) snapshot

        Returns:
            Dict with 'added', 'removed', 'changed' keys
        """
        vars1 = snapshot1.variables
        vars2 = snapshot2.variables

        keys1 = set(vars1.keys())
        keys2 = set(vars2.keys())

        added = keys2 - keys1
        removed = keys1 - keys2
        common = keys1 & keys2

        changed = {}
        for key in common:
            if vars1[key] != vars2[key]:
                changed[key] = {
                    'old': vars1[key],
                    'new': vars2[key]
                }

        # Also compare non-variable fields
        other_changes = {}

        if snapshot1.git_branch != snapshot2.git_branch:
            other_changes['git_branch'] = {
                'old': snapshot1.git_branch,
                'new': snapshot2.git_branch
            }

        if snapshot1.git_commit != snapshot2.git_commit:
            other_changes['git_commit'] = {
                'old': snapshot1.git_commit,
                'new': snapshot2.git_commit
            }

        if snapshot1.working_directory != snapshot2.working_directory:
            other_changes['working_directory'] = {
                'old': snapshot1.working_directory,
                'new': snapshot2.working_directory
            }

        return {
            'added': {k: vars2[k] for k in added},
            'removed': {k: vars1[k] for k in removed},
            'changed': changed,
            'other_changes': other_changes
        }

    def get_restore_commands(
        self,
        snapshot: EnvironmentSnapshot,
        current_env: dict[str, str] | None = None
    ) -> list[str]:
        """Generate shell commands to restore environment.

        Note: This only works for non-masked variables.

        Args:
            snapshot: Snapshot to restore to
            current_env: Current environment (default: os.environ)

        Returns:
            List of export commands
        """
        if current_env is None:
            current_env = dict(os.environ)

        commands = []

        for key, value in snapshot.variables.items():
            # Skip masked values
            if key in snapshot.masked_keys:
                continue

            # Check if value is different or missing
            if current_env.get(key) != value:
                # Escape single quotes in value
                escaped_value = value.replace("'", "'\"'\"'")
                commands.append(f"export {key}='{escaped_value}'")

        return commands

    def get_critical_diff(
        self,
        snapshot1: EnvironmentSnapshot,
        snapshot2: EnvironmentSnapshot
    ) -> dict[str, dict[str, str]]:
        """Get differences only in critical variables.

        Args:
            snapshot1: First snapshot
            snapshot2: Second snapshot

        Returns:
            Dict of critical variable changes
        """
        diff = self.compare_snapshots(snapshot1, snapshot2)

        critical_diff = {}

        for change_type in ['added', 'removed', 'changed']:
            for key in diff.get(change_type, {}):
                if key in self.CRITICAL_VARS:
                    critical_diff[key] = {
                        'type': change_type,
                        'value': diff[change_type][key]
                    }

        return critical_diff


# Global instance for convenience
_default_manager: EnvironmentManager | None = None


def get_environment_manager() -> EnvironmentManager:
    """Get the default environment manager instance.

    Returns:
        Default EnvironmentManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = EnvironmentManager()
    return _default_manager


def capture_env_snapshot(checkpoint_id: str) -> EnvironmentSnapshot:
    """Convenience function to capture environment snapshot.

    Args:
        checkpoint_id: Checkpoint ID

    Returns:
        EnvironmentSnapshot
    """
    return get_environment_manager().capture_snapshot(checkpoint_id)
