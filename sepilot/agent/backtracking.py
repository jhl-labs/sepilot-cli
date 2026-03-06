"""Backtracking - State Rollback and Recovery System

This module implements backtracking capabilities for the agent:
- Saves checkpoints before risky operations
- Detects failures and triggers rollback (test, lint, build failures)
- Restores previous stable state
- Enables exploration of alternative paths
- Environment variable snapshot integration
- Project-based history logging

Inspired by:
- MCTS (Monte Carlo Tree Search) in AlphaGo
- Git's branching and reset mechanisms
- Database transaction rollback
- Claude Code's rewind functionality
"""

import re
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sepilot.agent.enhanced_state import EnhancedAgentState
from sepilot.config.constants import (
    BACKTRACK_AUTO_INTERVAL,
    BACKTRACK_MAX_SCAN_LENGTH,
    MAX_BACKTRACK_CHECKPOINTS,
)

if TYPE_CHECKING:
    from sepilot.memory.environment_snapshot import EnvironmentSnapshot
    from sepilot.memory.file_checkpoint import FileCheckpointManager
    from sepilot.memory.history_writer import HistoryWriter


class CheckpointType(str, Enum):
    """Types of checkpoints."""
    AUTO = "auto"                    # Automatic checkpoint
    PRE_FILE_CHANGE = "pre_file_change"  # Before modifying files
    PRE_BASH = "pre_bash"            # Before bash commands
    PRE_GIT = "pre_git"              # Before git operations
    USER_REQUESTED = "user_requested"  # Manual checkpoint
    PLAN_START = "plan_start"        # Start of plan execution


class RollbackReason(str, Enum):
    """Reasons for triggering rollback."""
    TOOL_FAILURE = "tool_failure"
    TEST_FAILURE = "test_failure"
    LINT_FAILURE = "lint_failure"
    BUILD_FAILURE = "build_failure"
    USER_REQUESTED = "user_requested"
    REFLECTION_DECISION = "reflection_decision"
    ERROR_CASCADE = "error_cascade"


# Patterns to detect various failure types in tool outputs
FAILURE_PATTERNS = {
    RollbackReason.TEST_FAILURE: [
        r'FAILED',
        r'test.*fail',
        r'AssertionError',
        r'pytest.*error',
        r'ERRORS?:?\s*\d+',
        r'failures?:?\s*\d+',
        r'ERROR:.*test',
        r'\d+ failed',
    ],
    RollbackReason.LINT_FAILURE: [
        r'lint.*error',
        r'eslint.*error',
        r'pylint.*error',
        r'flake8.*error',
        r'mypy.*error',
        r'ruff.*error',
        r'Found \d+ error',
        r'E\d{3,4}:',  # pylint/flake8 error codes
        r'error TS\d+',  # TypeScript errors
    ],
    RollbackReason.BUILD_FAILURE: [
        r'build.*fail',
        r'compilation.*fail',
        r'compile.*error',
        r'BUILD FAILED',
        r'npm ERR!',
        r'error: cannot find',
        r'ModuleNotFoundError',
        r'ImportError',
        r'SyntaxError',
    ],
}


_FAILURE_QUICK_KEYWORDS = frozenset([
    'fail', 'error', 'assert', 'err!', 'syntaxerror', 'importerror', 'modulenot',
])

# Maximum output length to scan (avoid regex over huge outputs)
_MAX_SCAN_LENGTH = BACKTRACK_MAX_SCAN_LENGTH


def detect_failure_type(output: str) -> RollbackReason | None:
    """Detect the type of failure from tool output.

    Args:
        output: Tool output string to analyze

    Returns:
        RollbackReason if failure detected, None otherwise
    """
    if not output:
        return None

    # Quick pre-check: if no failure keyword at all, skip expensive regex
    output_lower = output[:_MAX_SCAN_LENGTH].lower()
    if not any(kw in output_lower for kw in _FAILURE_QUICK_KEYWORDS):
        return None

    # Limit scan length for regex patterns
    scan_text = output[:_MAX_SCAN_LENGTH]

    for failure_type, patterns in FAILURE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, scan_text, re.IGNORECASE):
                return failure_type

    return None


def analyze_tool_results_for_failures(
    tool_history: list,
    check_last_n: int = 3
) -> tuple[RollbackReason | None, str | None]:
    """Analyze recent tool results for test/lint/build failures.

    Args:
        tool_history: List of ToolCallRecord objects
        check_last_n: Number of recent tool calls to check

    Returns:
        Tuple of (failure_type, failure_message) or (None, None)
    """
    if not tool_history:
        return None, None

    # Check recent tool calls
    recent_tools = tool_history[-check_last_n:]

    for tc in recent_tools:
        result = getattr(tc, 'result', None) or ''

        # Skip successful or empty results
        if not result or getattr(tc, 'success', True):
            continue

        failure_type = detect_failure_type(result)
        if failure_type:
            # Extract relevant error message
            lines = result.split('\n')
            error_lines = [
                line for line in lines
                if any(keyword in line.lower()
                       for keyword in ['error', 'fail', 'assert'])
            ][:5]  # Limit to 5 lines
            failure_msg = '\n'.join(error_lines) if error_lines else result[:500]
            return failure_type, failure_msg

    return None, None


@dataclass
class Checkpoint:
    """A saved state checkpoint."""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    timestamp: datetime
    description: str
    state_snapshot: dict[str, Any]
    git_commit_hash: str | None  # Git HEAD at checkpoint time
    file_states: dict[str, str]  # file_path -> content at checkpoint
    working_directory: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "git_commit_hash": self.git_commit_hash,
            "working_directory": self.working_directory,
            "metadata": self.metadata,
            "files_tracked": list(self.file_states.keys())
        }


@dataclass
class EnhancedCheckpoint(Checkpoint):
    """Enhanced checkpoint with Claude Code-style features.

    Extends Checkpoint with:
    - Environment variable snapshots
    - Parent checkpoint tracking (for checkpoint chains)
    - Event range tracking (links to history events)
    - File diffs instead of just full content
    - Summary generation
    """
    # Environment snapshot reference
    environment_snapshot_id: str | None = None

    # Checkpoint chain (for tracking history)
    parent_checkpoint_id: str | None = None

    # History event references
    event_range_start: str | None = None  # First event ID in this checkpoint
    event_range_end: str | None = None    # Last event ID in this checkpoint

    # Files changed since parent checkpoint
    files_changed_since_parent: list[str] = field(default_factory=list)

    # File diffs (unified diff format) - more efficient than full content
    file_diffs: dict[str, str] = field(default_factory=dict)

    # AI-generated summary of changes
    summary: str | None = None

    # Session ID for correlation
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with enhanced fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "environment_snapshot_id": self.environment_snapshot_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "event_range_start": self.event_range_start,
            "event_range_end": self.event_range_end,
            "files_changed_since_parent": self.files_changed_since_parent,
            "has_diffs": bool(self.file_diffs),
            "summary": self.summary,
            "session_id": self.session_id
        })
        return base_dict

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        environment_snapshot_id: str | None = None,
        parent_checkpoint_id: str | None = None,
        session_id: str | None = None
    ) -> 'EnhancedCheckpoint':
        """Create EnhancedCheckpoint from a basic Checkpoint.

        Args:
            checkpoint: Base checkpoint
            environment_snapshot_id: ID of environment snapshot
            parent_checkpoint_id: ID of parent checkpoint
            session_id: Session ID

        Returns:
            EnhancedCheckpoint instance
        """
        return cls(
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_type=checkpoint.checkpoint_type,
            timestamp=checkpoint.timestamp,
            description=checkpoint.description,
            state_snapshot=checkpoint.state_snapshot,
            git_commit_hash=checkpoint.git_commit_hash,
            file_states=checkpoint.file_states,
            working_directory=checkpoint.working_directory,
            metadata=checkpoint.metadata,
            environment_snapshot_id=environment_snapshot_id,
            parent_checkpoint_id=parent_checkpoint_id,
            session_id=session_id
        )


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    checkpoint_id: str
    files_restored: list[str]
    git_reset_performed: bool
    error_message: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class BacktrackingManager:
    """Manages checkpoints and rollback operations.

    Features:
    - Automatic checkpointing before risky operations
    - File content backup and restoration
    - Git state tracking
    - Multiple checkpoint history
    - Environment variable snapshots
    - Integration with project history system
    """

    MAX_CHECKPOINTS = MAX_BACKTRACK_CHECKPOINTS
    AUTO_CHECKPOINT_INTERVAL = BACKTRACK_AUTO_INTERVAL
    MAX_FILE_SIZE_BYTES = 512 * 1024  # Skip files larger than 512KB for checkpoint
    MAX_FILES_PER_CHECKPOINT = 20  # Limit files per checkpoint to avoid memory bloat

    def __init__(
        self,
        working_directory: str | Path | None = None,
        enable_git_tracking: bool = True,
        console: Any | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        enable_history_logging: bool = True,
        use_file_checkpoint_manager: bool = True
    ):
        """Initialize backtracking manager.

        Args:
            working_directory: Working directory for operations
            enable_git_tracking: Whether to track git state
            console: Rich console for output
            verbose: Verbose output flag
            session_id: Current session ID for history correlation
            enable_history_logging: Whether to log to project history
            use_file_checkpoint_manager: Whether to use FileCheckpointManager for file tracking
        """
        self.working_directory = Path(working_directory or ".").resolve()
        self.enable_git_tracking = enable_git_tracking
        self.console = console
        self.verbose = verbose
        self.session_id = session_id
        self.enable_history_logging = enable_history_logging
        self.use_file_checkpoint_manager = use_file_checkpoint_manager

        self.checkpoints: list[Checkpoint] = []
        self.enhanced_checkpoints: list[EnhancedCheckpoint] = []
        self.rollback_history: list[RollbackResult] = []
        self._checkpoint_counter = 0
        self._last_checkpoint_id: str | None = None

        # Thread safety lock for checkpoint operations
        self._lock = threading.RLock()

        # Cached settings
        self._error_threshold: int | None = None

        # Lazy-loaded components
        self._history_writer: 'HistoryWriter | None' = None
        self._env_manager: Any = None
        self._project_manager: Any = None
        self._file_checkpoint_manager: 'FileCheckpointManager | None' = None

    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID."""
        self._checkpoint_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cp_{timestamp}_{self._checkpoint_counter:04d}"

    def _get_environment_manager(self) -> Any:
        """Lazy-load environment manager."""
        if self._env_manager is None:
            try:
                from sepilot.memory.environment_snapshot import get_environment_manager
                self._env_manager = get_environment_manager()
            except ImportError:
                pass
        return self._env_manager

    def _get_project_manager(self) -> Any:
        """Lazy-load project history manager."""
        if self._project_manager is None:
            try:
                from sepilot.memory.project_history import get_project_history_manager
                self._project_manager = get_project_history_manager()
            except ImportError:
                pass
        return self._project_manager

    def _get_file_checkpoint_manager(self) -> 'FileCheckpointManager | None':
        """Lazy-load file checkpoint manager.

        Returns:
            FileCheckpointManager instance or None
        """
        if not self.use_file_checkpoint_manager:
            return None

        if self._file_checkpoint_manager is None:
            try:
                from sepilot.memory.file_checkpoint import create_project_checkpoint_manager
                self._file_checkpoint_manager = create_project_checkpoint_manager(
                    project_path=str(self.working_directory),
                    max_checkpoints=self.MAX_CHECKPOINTS
                )
            except ImportError:
                pass
        return self._file_checkpoint_manager

    def sync_with_file_checkpoint_manager(self) -> None:
        """Synchronize state with FileCheckpointManager.

        Ensures both checkpoint systems track the same files.
        """
        fcm = self._get_file_checkpoint_manager()
        if not fcm:
            return

        # Track all files from our checkpoints
        for cp in self.checkpoints:
            fcm.track_files(list(cp.file_states.keys()))

    def _get_history_writer(self) -> 'HistoryWriter | None':
        """Lazy-load history writer."""
        if self._history_writer is None and self.enable_history_logging and self.session_id:
            try:
                from sepilot.memory.history_writer import get_history_writer
                self._history_writer = get_history_writer(
                    str(self.working_directory),
                    self.session_id
                )
            except ImportError:
                pass
        return self._history_writer

    def _capture_environment_snapshot(self, checkpoint_id: str) -> str | None:
        """Capture environment snapshot for a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to associate

        Returns:
            Environment snapshot ID or None
        """
        env_manager = self._get_environment_manager()
        if env_manager is None:
            return None

        try:
            snapshot = env_manager.capture_snapshot(
                checkpoint_id=checkpoint_id,
                working_dir=str(self.working_directory)
            )

            # Save to project-specific location
            project_manager = self._get_project_manager()
            if project_manager:
                snapshot_file = project_manager.get_env_snapshot_file(
                    str(self.working_directory),
                    checkpoint_id
                )
                env_manager.save_snapshot(snapshot, snapshot_file)

            return checkpoint_id
        except Exception:
            return None

    def _log_checkpoint_event(
        self,
        checkpoint_id: str,
        description: str,
        files: list[str]
    ) -> str | None:
        """Log checkpoint creation to history.

        Args:
            checkpoint_id: Checkpoint ID
            description: Checkpoint description
            files: List of tracked files

        Returns:
            Event ID or None
        """
        writer = self._get_history_writer()
        if writer is None:
            return None

        try:
            return writer.checkpoint_create(
                checkpoint_id=checkpoint_id,
                description=description,
                files=files
            )
        except Exception:
            return None

    def _log_rewind_event(
        self,
        target_checkpoint_id: str,
        files_restored: list[str]
    ) -> str | None:
        """Log rewind operation to history.

        Args:
            target_checkpoint_id: Target checkpoint ID
            files_restored: List of restored files

        Returns:
            Event ID or None
        """
        writer = self._get_history_writer()
        if writer is None:
            return None

        try:
            return writer.rewind(
                target_checkpoint_id=target_checkpoint_id,
                files_restored=files_restored
            )
        except Exception:
            return None

    def _get_git_head(self) -> str | None:
        """Get current git HEAD commit hash."""
        if not self.enable_git_tracking:
            return None

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _perform_git_reset(
        self,
        target_commit: str,
        hard_reset: bool = False
    ) -> tuple[bool, str | None]:
        """Perform git reset to a specific commit.

        Args:
            target_commit: Target commit hash
            hard_reset: If True, use --hard reset (WARNING: discards changes)

        Returns:
            Tuple of (success, error_message)
        """
        if not self.enable_git_tracking:
            return False, "Git tracking disabled"

        try:
            # First, check git status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=5
            )

            if status_result.returncode != 0:
                return False, "Git status check failed"

            has_changes = bool(status_result.stdout.strip())

            if hard_reset and has_changes:
                # Warn about hard reset with uncommitted changes
                if self.console and self.verbose:
                    self.console.print(
                        "[yellow]⚠️ Performing hard reset with uncommitted changes[/yellow]"
                    )

                # Hard reset - discards all local changes
                reset_result = subprocess.run(
                    ["git", "reset", "--hard", target_commit],
                    cwd=self.working_directory,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if reset_result.returncode != 0:
                    return False, f"Git reset --hard failed: {reset_result.stderr}"

                return True, None

            elif not hard_reset:
                # Soft approach: checkout files only
                checkout_result = subprocess.run(
                    ["git", "checkout", "--", "."],
                    cwd=self.working_directory,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if checkout_result.returncode != 0:
                    return False, f"Git checkout failed: {checkout_result.stderr}"

                return True, None

            else:
                # No changes to reset
                return True, None

        except subprocess.TimeoutExpired:
            return False, "Git operation timed out"
        except FileNotFoundError:
            return False, "Git command not found"
        except Exception as e:
            return False, f"Git operation failed: {e}"

    def _is_git_repo(self) -> bool:
        """Check if working directory is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _read_file_content(self, file_path: str) -> str | None:
        """Read file content safely."""
        try:
            full_path = self.working_directory / file_path
            if full_path.exists() and full_path.is_file():
                return full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
        return None

    def _write_file_content(self, file_path: str, content: str) -> bool:
        """Write file content safely."""
        try:
            full_path = self.working_directory / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return True
        except OSError:
            return False

    def _extract_state_snapshot(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Extract serializable snapshot from state."""
        return {
            "iteration_count": state.get("iteration_count", 0),
            "plan_steps": state.get("plan_steps", []),
            "current_plan_step": state.get("current_plan_step", 0),
            "current_strategy": state.get("current_strategy"),
            "confidence_level": state.get("confidence_level", 0.8),
            "error_count": len(state.get("error_history", [])),
            "tool_count": len(state.get("tool_call_history", [])),
            "file_change_count": len(state.get("file_changes", []))
        }

    def create_checkpoint(
        self,
        state: EnhancedAgentState,
        checkpoint_type: CheckpointType,
        description: str,
        files_to_track: list[str] | None = None
    ) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            state: Current agent state
            checkpoint_type: Type of checkpoint
            description: Description of checkpoint
            files_to_track: Specific files to backup (default: changed files)

        Returns:
            Created Checkpoint
        """
        with self._lock:
            return self._create_checkpoint_internal(
                state, checkpoint_type, description, files_to_track
            )

    def _create_checkpoint_internal(
        self,
        state: EnhancedAgentState,
        checkpoint_type: CheckpointType,
        description: str,
        files_to_track: list[str] | None = None
    ) -> Checkpoint:
        """Internal checkpoint creation (must be called with lock held)."""
        checkpoint_id = self._generate_checkpoint_id()

        # Determine files to track
        if files_to_track is None:
            # Track files that have been changed
            file_changes = state.get("file_changes", [])
            files_to_track = list({fc.file_path for fc in file_changes})

        # Backup file contents (with size and count limits)
        file_states = {}
        for file_path in files_to_track[:self.MAX_FILES_PER_CHECKPOINT]:
            try:
                fsize = Path(file_path).stat().st_size
                if fsize > self.MAX_FILE_SIZE_BYTES:
                    continue  # Skip large files
            except OSError:
                continue
            content = self._read_file_content(file_path)
            if content is not None:
                file_states[file_path] = content

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            timestamp=datetime.now(),
            description=description,
            state_snapshot=self._extract_state_snapshot(state),
            git_commit_hash=self._get_git_head(),
            file_states=file_states,
            working_directory=str(self.working_directory),
            metadata={
                "files_tracked": len(file_states),
                "state_iteration": state.get("iteration_count", 0)
            }
        )

        self.checkpoints.append(checkpoint)

        # Prune old checkpoints
        if len(self.checkpoints) > self.MAX_CHECKPOINTS:
            self.checkpoints = self.checkpoints[-self.MAX_CHECKPOINTS:]

        # Capture environment snapshot and log to history
        env_snapshot_id = self._capture_environment_snapshot(checkpoint_id)
        self._log_checkpoint_event(checkpoint_id, description, list(file_states.keys()))

        # Sync with FileCheckpointManager if available
        fcm = self._get_file_checkpoint_manager()
        if fcm and files_to_track:
            fcm.track_files(files_to_track)
            try:
                # Create checkpoint in FileCheckpointManager too
                fcm.create_checkpoint(
                    message_index=state.get("iteration_count", 0),
                    user_prompt=description
                )
            except Exception as e:
                if self.console and self.verbose:
                    self.console.print(f"[dim yellow]⚠️ FileCheckpointManager sync failed: {e}[/dim yellow]")

        # Create enhanced version
        enhanced = EnhancedCheckpoint.from_checkpoint(
            checkpoint,
            environment_snapshot_id=env_snapshot_id,
            parent_checkpoint_id=self._last_checkpoint_id,
            session_id=self.session_id
        )
        self.enhanced_checkpoints.append(enhanced)
        # Prune enhanced_checkpoints to match checkpoints limit
        if len(self.enhanced_checkpoints) > self.MAX_CHECKPOINTS:
            self.enhanced_checkpoints = self.enhanced_checkpoints[-self.MAX_CHECKPOINTS:]
        self._last_checkpoint_id = checkpoint_id

        if self.console and self.verbose:
            self.console.print(
                f"[dim cyan]📌 Checkpoint created: {checkpoint_id} "
                f"({len(file_states)} files tracked)[/dim cyan]"
            )

        return checkpoint

    def create_enhanced_checkpoint(
        self,
        state: EnhancedAgentState,
        checkpoint_type: CheckpointType,
        description: str,
        files_to_track: list[str] | None = None,
        summary: str | None = None,
        generate_diffs: bool = True
    ) -> EnhancedCheckpoint:
        """Create an enhanced checkpoint with full history integration.

        Args:
            state: Current agent state
            checkpoint_type: Type of checkpoint
            description: Description of checkpoint
            files_to_track: Specific files to backup
            summary: Optional AI-generated summary
            generate_diffs: Whether to generate file diffs

        Returns:
            Created EnhancedCheckpoint
        """
        # Create base checkpoint
        base_checkpoint = self.create_checkpoint(
            state=state,
            checkpoint_type=checkpoint_type,
            description=description,
            files_to_track=files_to_track
        )

        # Get the enhanced version we just created
        enhanced = self.enhanced_checkpoints[-1]
        enhanced.summary = summary

        # Calculate files changed since parent and generate diffs
        if enhanced.parent_checkpoint_id:
            parent = self.get_enhanced_checkpoint_by_id(enhanced.parent_checkpoint_id)
            if parent:
                parent_files = set(parent.file_states.keys())
                current_files = set(enhanced.file_states.keys())

                # Find changed files
                changed_files = []
                for f in current_files - parent_files:
                    changed_files.append(f)  # New files
                for f in parent_files & current_files:
                    if parent.file_states.get(f) != enhanced.file_states.get(f):
                        changed_files.append(f)  # Modified files

                enhanced.files_changed_since_parent = changed_files

                # Generate diffs if requested
                if generate_diffs and changed_files:
                    try:
                        from sepilot.memory.history_event import FileDiff, FileAction
                        for file_path in changed_files:
                            old_content = parent.file_states.get(file_path)
                            new_content = enhanced.file_states.get(file_path)

                            diff_obj = FileDiff.create_diff(
                                file_path=file_path,
                                old_content=old_content,
                                new_content=new_content
                            )
                            if diff_obj and diff_obj.unified_diff:
                                enhanced.file_diffs[file_path] = diff_obj.unified_diff
                    except ImportError:
                        pass  # Skip diff generation if module not available

        return enhanced

    def get_enhanced_checkpoint_by_id(self, checkpoint_id: str) -> EnhancedCheckpoint | None:
        """Get enhanced checkpoint by ID."""
        for cp in self.enhanced_checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                return cp
        return None

    def list_enhanced_checkpoints(self) -> list[dict[str, Any]]:
        """List all enhanced checkpoints with summary info."""
        return [cp.to_dict() for cp in self.enhanced_checkpoints]

    def create_auto_checkpoint(
        self,
        state: EnhancedAgentState
    ) -> Checkpoint | None:
        """Create automatic checkpoint if conditions are met.

        Args:
            state: Current agent state

        Returns:
            Created checkpoint or None
        """
        iteration = state.get("iteration_count", 0)

        # Auto-checkpoint at intervals
        if iteration > 0 and iteration % self.AUTO_CHECKPOINT_INTERVAL == 0:
            return self.create_checkpoint(
                state=state,
                checkpoint_type=CheckpointType.AUTO,
                description=f"Auto checkpoint at iteration {iteration}"
            )
        return None

    def create_pre_operation_checkpoint(
        self,
        state: EnhancedAgentState,
        operation_type: str,
        target_files: list[str]
    ) -> Checkpoint:
        """Create checkpoint before a risky operation.

        Args:
            state: Current agent state
            operation_type: Type of operation (file_edit, bash, git)
            target_files: Files that will be affected

        Returns:
            Created checkpoint
        """
        checkpoint_type_map = {
            "file_edit": CheckpointType.PRE_FILE_CHANGE,
            "file_write": CheckpointType.PRE_FILE_CHANGE,
            "bash": CheckpointType.PRE_BASH,
            "git": CheckpointType.PRE_GIT
        }
        checkpoint_type = checkpoint_type_map.get(
            operation_type, CheckpointType.AUTO
        )

        return self.create_checkpoint(
            state=state,
            checkpoint_type=checkpoint_type,
            description=f"Before {operation_type} on {', '.join(target_files[:3])}",
            files_to_track=target_files
        )

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Get the most recent checkpoint."""
        with self._lock:
            return self.checkpoints[-1] if self.checkpoints else None

    def get_checkpoint_by_id(self, checkpoint_id: str) -> Checkpoint | None:
        """Get checkpoint by ID."""
        with self._lock:
            for cp in self.checkpoints:
                if cp.checkpoint_id == checkpoint_id:
                    return cp
            return None

    def validate_checkpoint(self, checkpoint: Checkpoint) -> tuple[bool, str]:
        """Validate a checkpoint before rollback.

        Args:
            checkpoint: Checkpoint to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if checkpoint exists in our list
        if checkpoint not in self.checkpoints:
            return False, "Checkpoint not found in checkpoint list"

        # Check if checkpoint is not too old (max 24 hours)
        age = datetime.now() - checkpoint.timestamp
        if age.total_seconds() > 24 * 60 * 60:
            return False, f"Checkpoint is too old ({age.total_seconds() / 3600:.1f} hours)"

        # Check if file states exist
        if not checkpoint.file_states and not checkpoint.state_snapshot:
            return False, "Checkpoint has no file states or state snapshot"

        # Check if working directory matches
        if checkpoint.working_directory != str(self.working_directory):
            return False, f"Working directory mismatch: {checkpoint.working_directory} vs {self.working_directory}"

        return True, ""

    def rollback_to_checkpoint(
        self,
        checkpoint: Checkpoint,
        reason: RollbackReason,
        restore_files: bool = True,
        git_reset: bool = False,
        skip_validation: bool = False
    ) -> RollbackResult:
        """Rollback to a specific checkpoint.

        Args:
            checkpoint: Checkpoint to restore
            reason: Reason for rollback
            restore_files: Whether to restore file contents
            git_reset: Whether to perform git reset (dangerous)
            skip_validation: Skip checkpoint validation (use with caution)

        Returns:
            RollbackResult with operation status
        """
        with self._lock:
            return self._rollback_to_checkpoint_internal(
                checkpoint, reason, restore_files, git_reset, skip_validation
            )

    def _rollback_to_checkpoint_internal(
        self,
        checkpoint: Checkpoint,
        reason: RollbackReason,
        restore_files: bool = True,
        git_reset: bool = False,
        skip_validation: bool = False
    ) -> RollbackResult:
        """Internal rollback (must be called with lock held)."""
        files_restored = []
        git_reset_performed = False
        error_message = None

        # Validate checkpoint before rollback
        if not skip_validation:
            is_valid, validation_error = self.validate_checkpoint(checkpoint)
            if not is_valid:
                if self.console and self.verbose:
                    self.console.print(
                        f"[yellow]⚠️ Checkpoint validation failed: {validation_error}[/yellow]"
                    )
                return RollbackResult(
                    success=False,
                    checkpoint_id=checkpoint.checkpoint_id,
                    files_restored=[],
                    git_reset_performed=False,
                    error_message=f"Validation failed: {validation_error}"
                )

        try:
            # Restore file contents
            if restore_files:
                # Try FileCheckpointManager first for better tracking
                fcm = self._get_file_checkpoint_manager()
                if fcm:
                    try:
                        fcm_result = fcm.revert_by_count(count=1)
                        if fcm_result:
                            files_restored.extend([
                                path for path, status in fcm_result.items()
                                if status in ('restored', 'deleted')
                            ])
                    except Exception:
                        pass  # Fall through to manual restoration

                # Manual restoration for files not handled by FCM
                for file_path, content in checkpoint.file_states.items():
                    if file_path not in files_restored:
                        if self._write_file_content(file_path, content):
                            files_restored.append(file_path)
                        else:
                            if error_message:
                                error_message += f"; Failed to restore {file_path}"
                            else:
                                error_message = f"Failed to restore {file_path}"

            # Git reset (only if explicitly requested and safe)
            if git_reset and checkpoint.git_commit_hash:
                git_reset_performed, git_error = self._perform_git_reset(
                    checkpoint.git_commit_hash,
                    hard_reset=True
                )
                if git_error:
                    if error_message:
                        error_message += f"; {git_error}"
                    else:
                        error_message = git_error

            success = len(files_restored) > 0 or not checkpoint.file_states

            result = RollbackResult(
                success=success,
                checkpoint_id=checkpoint.checkpoint_id,
                files_restored=files_restored,
                git_reset_performed=git_reset_performed,
                error_message=error_message
            )

            self.rollback_history.append(result)

            # Log rewind event to history
            if success:
                self._log_rewind_event(checkpoint.checkpoint_id, files_restored)

            if self.console and self.verbose:
                status = "[green]✅" if success else "[red]❌"
                self.console.print(
                    f"{status} Rollback to {checkpoint.checkpoint_id}: "
                    f"{len(files_restored)} files restored[/]"
                )

            return result

        except Exception as e:
            return RollbackResult(
                success=False,
                checkpoint_id=checkpoint.checkpoint_id,
                files_restored=files_restored,
                git_reset_performed=False,
                error_message=str(e)
            )

    def rollback_to_latest(
        self,
        reason: RollbackReason,
        restore_files: bool = True
    ) -> RollbackResult | None:
        """Rollback to the latest checkpoint.

        Args:
            reason: Reason for rollback
            restore_files: Whether to restore files

        Returns:
            RollbackResult or None if no checkpoints
        """
        checkpoint = self.get_latest_checkpoint()
        if checkpoint:
            return self.rollback_to_checkpoint(
                checkpoint=checkpoint,
                reason=reason,
                restore_files=restore_files
            )
        return None

    def should_rollback(self, state: EnhancedAgentState) -> tuple[bool, RollbackReason | None]:
        """Determine if rollback should be triggered.

        Args:
            state: Current agent state

        Returns:
            Tuple of (should_rollback, reason)
        """
        error_threshold = self._get_error_threshold()

        # Check for error cascade
        error_history = state.get("error_history", [])
        if len(error_history) >= error_threshold:
            recent_errors = error_history[-error_threshold:]
            if all(
                not (e.get("resolved", False) if isinstance(e, dict) else getattr(e, "resolved", False))
                for e in recent_errors
            ):
                return True, RollbackReason.ERROR_CASCADE

        # Check reflection decision
        reflection_decision = state.get("reflection_decision")
        if reflection_decision == "revise_plan":
            # Reflection suggested going back
            return True, RollbackReason.REFLECTION_DECISION

        # Check for repeated tool failures
        tool_history = state.get("tool_call_history", [])
        if len(tool_history) >= error_threshold:
            recent_tools = tool_history[-error_threshold:]
            if all(not tc.success for tc in recent_tools):
                return True, RollbackReason.TOOL_FAILURE

        # Check for test/lint/build failures
        failure_type, failure_msg = analyze_tool_results_for_failures(tool_history)
        if failure_type:
            if self.console and self.verbose:
                self.console.print(
                    f"[yellow]⚠️ {failure_type.value} detected: {failure_msg[:100]}...[/yellow]"
                )
            return True, failure_type

        return False, None

    def _get_error_threshold(self) -> int:
        """Get error threshold from settings or default (cached after first read).

        Returns:
            Number of consecutive errors before triggering rollback
        """
        if self._error_threshold is not None:
            return self._error_threshold

        threshold = 3  # Default
        try:
            import json
            settings_path = Path.home() / ".sepilot" / "settings.json"
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = json.load(f)
                threshold = settings.get("error_threshold", 3)
        except Exception:
            pass
        self._error_threshold = threshold
        return threshold

    def get_state_for_rollback(
        self,
        checkpoint: Checkpoint,
        current_state: EnhancedAgentState
    ) -> dict[str, Any]:
        """Get state updates for rollback.

        Args:
            checkpoint: Checkpoint to restore to
            current_state: Current state

        Returns:
            State update dict
        """
        snapshot = checkpoint.state_snapshot

        return {
            "current_plan_step": snapshot.get("current_plan_step", 0),
            "confidence_level": snapshot.get("confidence_level", 0.8),
            "needs_additional_iteration": True,
            "backtrack_performed": True,
            "backtrack_checkpoint_id": checkpoint.checkpoint_id,
            "backtrack_reason": "Rollback to previous stable state"
        }

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        with self._lock:
            self.checkpoints = []
            self.enhanced_checkpoints = []
            self._last_checkpoint_id = None

    def get_stats(self) -> dict[str, Any]:
        """Get backtracking statistics."""
        return {
            "total_checkpoints": len(self.checkpoints),
            "total_rollbacks": len(self.rollback_history),
            "successful_rollbacks": sum(1 for r in self.rollback_history if r.success),
            "checkpoint_types": {
                cp_type.value: sum(1 for cp in self.checkpoints if cp.checkpoint_type == cp_type)
                for cp_type in CheckpointType
            }
        }


class BacktrackingNode:
    """LangGraph node for backtracking decisions."""

    def __init__(
        self,
        backtracking_manager: BacktrackingManager,
        console: Any | None = None,
        verbose: bool = False
    ):
        """Initialize backtracking node.

        Args:
            backtracking_manager: BacktrackingManager instance
            console: Rich console
            verbose: Verbose output flag
        """
        self.manager = backtracking_manager
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Execute backtracking check and potential rollback.

        Args:
            state: Current agent state

        Returns:
            State updates
        """
        updates: dict[str, Any] = {}

        # Check if rollback needed
        should_rollback, reason = self.manager.should_rollback(state)

        if should_rollback and reason:
            checkpoint = self.manager.get_latest_checkpoint()

            if checkpoint:
                result = self.manager.rollback_to_checkpoint(
                    checkpoint=checkpoint,
                    reason=reason,
                    restore_files=True
                )

                if result.success:
                    # Get state updates from checkpoint
                    updates = self.manager.get_state_for_rollback(checkpoint, state)
                    updates["backtrack_result"] = {
                        "success": True,
                        "files_restored": result.files_restored,
                        "reason": reason.value
                    }

                    if self.console:
                        self.console.print(
                            f"[yellow]⏪ Backtracked to checkpoint {checkpoint.checkpoint_id}[/yellow]"
                        )
                else:
                    updates["backtrack_result"] = {
                        "success": False,
                        "error": result.error_message
                    }
        else:
            # Create auto checkpoint if appropriate
            self.manager.create_auto_checkpoint(state)

        return updates


def create_backtracking_manager(
    working_directory: str | Path | None = None,
    enable_git_tracking: bool = True,
    console: Any | None = None,
    verbose: bool = False,
    session_id: str | None = None,
    enable_history_logging: bool = True,
    use_file_checkpoint_manager: bool = True
) -> BacktrackingManager:
    """Factory function to create BacktrackingManager.

    Args:
        working_directory: Working directory
        enable_git_tracking: Enable git state tracking
        console: Rich console
        verbose: Verbose output
        session_id: Current session ID for history correlation
        enable_history_logging: Enable project history logging
        use_file_checkpoint_manager: Enable FileCheckpointManager integration

    Returns:
        Configured BacktrackingManager
    """
    return BacktrackingManager(
        working_directory=working_directory,
        enable_git_tracking=enable_git_tracking,
        console=console,
        verbose=verbose,
        session_id=session_id,
        enable_history_logging=enable_history_logging,
        use_file_checkpoint_manager=use_file_checkpoint_manager
    )
