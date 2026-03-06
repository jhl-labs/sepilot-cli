"""Undo/Redo commands for conversation and file changes.

OpenCode-style undo/redo implementation:
- /undo: Remove the most recent user message, all subsequent responses, and associated file changes
- /redo: Restore a previously undone message and its corresponding file modifications

Both commands use Git for file state management when available.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class UndoEntry:
    """An entry in the undo/redo stack"""
    timestamp: str
    user_message: str
    assistant_response: str
    files_changed: list[str]  # List of changed file paths
    git_commit: str | None  # Git commit SHA if available
    checkpoint_id: str | None  # FileCheckpointManager checkpoint ID
    message_indices: tuple[int, int]  # (user_msg_index, assistant_msg_index)


class UndoRedoManager:
    """Manages undo/redo operations for conversation and files."""

    def __init__(self, project_path: str | None = None):
        """Initialize the undo/redo manager.

        Args:
            project_path: Project directory path (default: current directory)
        """
        self.project_path = project_path or os.getcwd()
        self._undo_stack: list[UndoEntry] = []
        self._redo_stack: list[UndoEntry] = []
        self._is_git_repo = self._check_git_repo()
        self._checkpoint_manager = None
        self._storage_path = self._get_storage_path()
        self._load_stacks()

    def _check_git_repo(self) -> bool:
        """Check if the project is a Git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_storage_path(self) -> Path:
        """Get path for persisting undo/redo stacks."""
        import hashlib
        path_hash = hashlib.sha256(self.project_path.encode()).hexdigest()[:16]
        project_name = Path(self.project_path).name
        storage_dir = Path.home() / ".sepilot" / "undo-redo"
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir / f"{project_name}_{path_hash}.json"

    def _save_stacks(self) -> None:
        """Save undo/redo stacks to disk."""
        try:
            data = {
                "project_path": self.project_path,
                "undo_stack": [self._entry_to_dict(e) for e in self._undo_stack],
                "redo_stack": [self._entry_to_dict(e) for e in self._redo_stack],
            }
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_stacks(self) -> None:
        """Load undo/redo stacks from disk.

        Validates that redo entries still have valid git stashes available,
        discarding entries whose stashes have already been popped or dropped.
        """
        try:
            if not self._storage_path.exists():
                return
            with open(self._storage_path, encoding='utf-8') as f:
                data = json.load(f)
            if data.get("project_path") != self.project_path:
                return
            self._undo_stack = [self._dict_to_entry(d) for d in data.get("undo_stack", [])]

            # Validate redo entries: only keep those with valid git stashes
            redo_candidates = [self._dict_to_entry(d) for d in data.get("redo_stack", [])]
            if self._is_git_repo and redo_candidates:
                stash_list = self._get_stash_list()
                validated = []
                for entry in redo_candidates:
                    if not entry.files_changed:
                        validated.append(entry)  # No files to restore, always valid
                    else:
                        stash_msg = f"sepilot_undo_{entry.timestamp.replace(':', '-')}"
                        if any(stash_msg in line for line in stash_list):
                            validated.append(entry)
                        # else: stash already popped, discard this entry
                self._redo_stack = validated
            else:
                self._redo_stack = redo_candidates
        except Exception:
            pass

    def _get_stash_list(self) -> list[str]:
        """Get current git stash list."""
        try:
            result = subprocess.run(
                ["git", "stash", "list"],
                cwd=self.project_path,
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return []

    @staticmethod
    def _entry_to_dict(entry: 'UndoEntry') -> dict:
        return {
            "timestamp": entry.timestamp,
            "user_message": entry.user_message,
            "assistant_response": entry.assistant_response,
            "files_changed": entry.files_changed,
            "git_commit": entry.git_commit,
            "checkpoint_id": entry.checkpoint_id,
            "message_indices": list(entry.message_indices),
        }

    @staticmethod
    def _dict_to_entry(d: dict) -> 'UndoEntry':
        return UndoEntry(
            timestamp=d["timestamp"],
            user_message=d["user_message"],
            assistant_response=d["assistant_response"],
            files_changed=d.get("files_changed", []),
            git_commit=d.get("git_commit"),
            checkpoint_id=d.get("checkpoint_id"),
            message_indices=tuple(d.get("message_indices", (0, 0))),
        )

    def _get_checkpoint_manager(self):
        """Lazy-load the checkpoint manager."""
        if self._checkpoint_manager is None:
            try:
                from sepilot.memory.file_checkpoint import FileCheckpointManager
                self._checkpoint_manager = FileCheckpointManager()
            except ImportError:
                pass
        return self._checkpoint_manager

    def _get_git_status(self) -> dict[str, list[str]]:
        """Get current Git status (modified, added, deleted files)."""
        if not self._is_git_repo:
            return {"modified": [], "added": [], "deleted": []}

        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return {"modified": [], "added": [], "deleted": []}

            modified = []
            added = []
            deleted = []

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                status = line[:2]
                filepath = line[3:]

                if status in (" M", "M ", "MM"):
                    modified.append(filepath)
                elif status in ("A ", " A", "??"):
                    added.append(filepath)
                elif status in ("D ", " D"):
                    deleted.append(filepath)

            return {"modified": modified, "added": added, "deleted": deleted}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {"modified": [], "added": [], "deleted": []}

    def _get_current_git_sha(self) -> str | None:
        """Get current Git HEAD SHA."""
        if not self._is_git_repo:
            return None

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _git_stash_changes(self, message: str) -> bool:
        """Stash current changes for potential redo."""
        if not self._is_git_repo:
            return False

        try:
            # First, stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.project_path,
                capture_output=True,
                timeout=10
            )

            # Stash with message
            result = subprocess.run(
                ["git", "stash", "push", "-m", f"sepilot_undo_{message}"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _git_stash_files(self, files: list[str], message: str) -> bool:
        """Stash only specific files for potential redo.

        Unlike _git_stash_changes which stashes everything, this only saves
        the specified files so unrelated work-in-progress is preserved.
        """
        if not self._is_git_repo or not files:
            return False

        try:
            # Stage only the specific files
            subprocess.run(
                ["git", "add", "--"] + files,
                cwd=self.project_path,
                capture_output=True,
                timeout=10
            )

            # Stash only staged changes (--staged requires git 2.35+)
            result = subprocess.run(
                ["git", "stash", "push", "--staged", "-m", f"sepilot_undo_{message}"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True

            # Fallback for older git: stash with pathspec
            # Unstage first to avoid polluting index
            subprocess.run(
                ["git", "reset", "HEAD", "--"] + files,
                cwd=self.project_path,
                capture_output=True,
                timeout=10
            )
            result = subprocess.run(
                ["git", "stash", "push", "-m", f"sepilot_undo_{message}", "--"] + files,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _git_restore_from_stash(self, stash_message: str) -> bool:
        """Restore changes from stash by message pattern."""
        if not self._is_git_repo:
            return False

        try:
            # List stashes to find matching one
            result = subprocess.run(
                ["git", "stash", "list"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return False

            # Find matching stash
            stash_ref = None
            for line in result.stdout.strip().split("\n"):
                if stash_message in line:
                    stash_ref = line.split(":")[0]
                    break

            if not stash_ref:
                return False

            # Apply stash
            result = subprocess.run(
                ["git", "stash", "pop", stash_ref],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _git_checkout_files(self, files: list[str]) -> bool:
        """Checkout (revert) specific files from HEAD."""
        if not self._is_git_repo or not files:
            return False

        try:
            result = subprocess.run(
                ["git", "checkout", "HEAD", "--"] + files,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def record_exchange(
        self,
        user_message: str,
        assistant_response: str,
        user_msg_index: int,
        assistant_msg_index: int,
        files_changed: list[str] | None = None
    ) -> None:
        """Record a conversation exchange for potential undo.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            user_msg_index: Index of user message in conversation
            assistant_msg_index: Index of assistant message in conversation
            files_changed: List of files that were changed
        """
        # Clear redo stack when new action is taken
        self._redo_stack.clear()

        # Get files changed from Git if not provided
        if files_changed is None:
            status = self._get_git_status()
            files_changed = status["modified"] + status["added"]

        # Get checkpoint ID if available
        checkpoint_id = None
        checkpoint_manager = self._get_checkpoint_manager()
        if checkpoint_manager and checkpoint_manager.checkpoints:
            checkpoint_id = checkpoint_manager.checkpoints[-1].id

        entry = UndoEntry(
            timestamp=datetime.now().isoformat(),
            user_message=user_message[:500],  # Truncate for storage
            assistant_response=assistant_response[:500],
            files_changed=files_changed,
            git_commit=self._get_current_git_sha(),
            checkpoint_id=checkpoint_id,
            message_indices=(user_msg_index, assistant_msg_index)
        )

        self._undo_stack.append(entry)

        # Keep stack size manageable
        if len(self._undo_stack) > 50:
            self._undo_stack = self._undo_stack[-50:]

        self._save_stacks()

    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return len(self._redo_stack) > 0

    def undo(self) -> tuple[UndoEntry | None, dict[str, Any]]:
        """Perform undo operation.

        Returns:
            Tuple of (UndoEntry, result_dict) or (None, error_dict)
        """
        if not self._undo_stack:
            return None, {"success": False, "error": "Nothing to undo"}

        entry = self._undo_stack.pop()
        result = {"success": True, "files_reverted": 0, "messages_removed": 2}

        # Revert file changes
        if entry.files_changed:
            if self._is_git_repo:
                # First, save current state of the changed files for redo via stash.
                # We only stash the specific files (not all changes) so unrelated
                # work-in-progress is not disturbed.
                stash_msg = entry.timestamp.replace(":", "-")
                self._git_stash_files(entry.files_changed, stash_msg)

                # Checkout the files from HEAD to revert them
                if self._git_checkout_files(entry.files_changed):
                    result["files_reverted"] = len(entry.files_changed)
            else:
                # Use checkpoint manager fallback
                checkpoint_manager = self._get_checkpoint_manager()
                if checkpoint_manager and entry.checkpoint_id:
                    # Find the checkpoint before this one
                    for i, cp in enumerate(checkpoint_manager.checkpoints):
                        if cp.id == entry.checkpoint_id and i > 0:
                            prev_cp = checkpoint_manager.checkpoints[i - 1]
                            revert_result = checkpoint_manager.revert_to_checkpoint(prev_cp)
                            result["files_reverted"] = sum(
                                1 for v in revert_result.values()
                                if v in ("restored", "deleted")
                            )
                            break

        # Move to redo stack
        self._redo_stack.append(entry)
        self._save_stacks()

        return entry, result

    def redo(self) -> tuple[UndoEntry | None, dict[str, Any]]:
        """Perform redo operation.

        Returns:
            Tuple of (UndoEntry, result_dict) or (None, error_dict)
        """
        if not self._redo_stack:
            return None, {"success": False, "error": "Nothing to redo"}

        entry = self._redo_stack.pop()
        result = {"success": True, "files_restored": 0, "messages_restored": 2}

        # Restore file changes
        if entry.files_changed:
            if self._is_git_repo:
                stash_msg = entry.timestamp.replace(":", "-")
                if self._git_restore_from_stash(f"sepilot_undo_{stash_msg}"):
                    result["files_restored"] = len(entry.files_changed)
            else:
                # Use checkpoint manager fallback
                checkpoint_manager = self._get_checkpoint_manager()
                if checkpoint_manager and entry.checkpoint_id:
                    for cp in checkpoint_manager.checkpoints:
                        if cp.id == entry.checkpoint_id:
                            revert_result = checkpoint_manager.revert_to_checkpoint(cp)
                            result["files_restored"] = sum(
                                1 for v in revert_result.values()
                                if v in ("restored", "deleted")
                            )
                            break

        # Move back to undo stack
        self._undo_stack.append(entry)
        self._save_stacks()

        return entry, result

    def get_undo_stack_info(self) -> list[dict[str, Any]]:
        """Get information about items in the undo stack."""
        return [
            {
                "timestamp": entry.timestamp,
                "user_message": entry.user_message[:100] + "..." if len(entry.user_message) > 100 else entry.user_message,
                "files_changed": len(entry.files_changed),
                "has_git": entry.git_commit is not None
            }
            for entry in reversed(self._undo_stack)
        ]

    def get_redo_stack_info(self) -> list[dict[str, Any]]:
        """Get information about items in the redo stack."""
        return [
            {
                "timestamp": entry.timestamp,
                "user_message": entry.user_message[:100] + "..." if len(entry.user_message) > 100 else entry.user_message,
                "files_changed": len(entry.files_changed),
                "has_git": entry.git_commit is not None
            }
            for entry in reversed(self._redo_stack)
        ]


# Global manager instance (per project)
_managers: dict[str, UndoRedoManager] = {}


def get_undo_redo_manager(project_path: str | None = None) -> UndoRedoManager:
    """Get or create an UndoRedoManager for the project.

    Args:
        project_path: Project directory path (default: current directory)

    Returns:
        UndoRedoManager instance
    """
    path = project_path or os.getcwd()
    if path not in _managers:
        _managers[path] = UndoRedoManager(path)
    return _managers[path]


def handle_undo(
    console: Console,
    agent: Any | None,
    conversation_context: list,
    input_text: str,
) -> dict[str, Any]:
    """Handle /undo command.

    Removes the most recent user message, all subsequent responses,
    and associated file changes.

    Args:
        console: Rich console for output
        agent: Agent with conversation management
        conversation_context: Current conversation context
        input_text: Original command for parsing args

    Returns:
        Dictionary with results
    """
    manager = get_undo_redo_manager()

    # Parse arguments
    args = input_text.strip().split()

    # Handle --list or -l
    if len(args) > 1 and args[1] in ("--list", "-l"):
        _show_undo_stack(console, manager)
        return {}

    if not manager.can_undo():
        console.print("[yellow]Nothing to undo[/yellow]")
        console.print("[dim]Undo is available after conversation exchanges are recorded[/dim]")
        return {"success": False, "error": "Nothing to undo"}

    # Perform undo
    entry, result = manager.undo()

    if not result.get("success"):
        console.print(f"[red]Undo failed: {result.get('error')}[/red]")
        return result

    # Remove messages from conversation context
    if entry and agent and hasattr(agent, 'rewind_messages'):
        agent.rewind_messages(1)

        # Also update conversation_context
        if len(conversation_context) >= 2:
            conversation_context.pop()  # Remove assistant message
            conversation_context.pop()  # Remove user message

    # Display results
    console.print()
    console.print("[bold cyan]Undo[/bold cyan]")
    console.print()

    if entry:
        console.print(f"[green]Removed:[/green] {entry.user_message[:80]}{'...' if len(entry.user_message) > 80 else ''}")

    files_reverted = result.get("files_reverted", 0)
    if files_reverted > 0:
        console.print(f"[green]Files reverted:[/green] {files_reverted}")
        if entry and entry.files_changed:
            for f in entry.files_changed[:5]:
                console.print(f"  [dim]{f}[/dim]")
            if len(entry.files_changed) > 5:
                console.print(f"  [dim]... and {len(entry.files_changed) - 5} more[/dim]")

    console.print()
    if manager.can_redo():
        console.print("[dim]Use /redo to restore this change[/dim]")
    if manager.can_undo():
        console.print("[dim]Use /undo again to go further back[/dim]")

    return result


def handle_redo(
    console: Console,
    agent: Any | None,
    conversation_context: list,
    input_text: str,
) -> dict[str, Any]:
    """Handle /redo command.

    Restores a previously undone message and its corresponding file modifications.

    Args:
        console: Rich console for output
        agent: Agent with conversation management
        conversation_context: Current conversation context
        input_text: Original command for parsing args

    Returns:
        Dictionary with results
    """
    manager = get_undo_redo_manager()

    # Parse arguments
    args = input_text.strip().split()

    # Handle --list or -l
    if len(args) > 1 and args[1] in ("--list", "-l"):
        _show_redo_stack(console, manager)
        return {}

    if not manager.can_redo():
        console.print("[yellow]Nothing to redo[/yellow]")
        console.print("[dim]Redo is available after using /undo[/dim]")
        return {"success": False, "error": "Nothing to redo"}

    # Perform redo
    entry, result = manager.redo()

    if not result.get("success"):
        console.print(f"[red]Redo failed: {result.get('error')}[/red]")
        return result

    # Restore messages to conversation context using proper LangChain message types
    if entry:
        conversation_context.append(HumanMessage(content=entry.user_message))
        conversation_context.append(AIMessage(content=entry.assistant_response))

    # Display results
    console.print()
    console.print("[bold cyan]Redo[/bold cyan]")
    console.print()

    if entry:
        console.print(f"[green]Restored:[/green] {entry.user_message[:80]}{'...' if len(entry.user_message) > 80 else ''}")

    files_restored = result.get("files_restored", 0)
    if files_restored > 0:
        console.print(f"[green]Files restored:[/green] {files_restored}")
        if entry and entry.files_changed:
            for f in entry.files_changed[:5]:
                console.print(f"  [dim]{f}[/dim]")
            if len(entry.files_changed) > 5:
                console.print(f"  [dim]... and {len(entry.files_changed) - 5} more[/dim]")

    console.print()
    if manager.can_undo():
        console.print("[dim]Use /undo to undo this change[/dim]")

    return result


def _show_undo_stack(console: Console, manager: UndoRedoManager) -> None:
    """Show items available for undo."""
    stack_info = manager.get_undo_stack_info()

    if not stack_info:
        console.print("[yellow]Undo stack is empty[/yellow]")
        return

    console.print("[bold cyan]Undo Stack[/bold cyan]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Message", max_width=60)
    table.add_column("Files", justify="right", width=6)
    table.add_column("Git", width=4)

    for i, info in enumerate(stack_info, 1):
        table.add_row(
            str(i),
            info["user_message"],
            str(info["files_changed"]),
            "[green]Yes[/green]" if info["has_git"] else "[dim]No[/dim]"
        )

    console.print(table)
    console.print()
    console.print(f"[dim]{len(stack_info)} item(s) available for undo[/dim]")


def _show_redo_stack(console: Console, manager: UndoRedoManager) -> None:
    """Show items available for redo."""
    stack_info = manager.get_redo_stack_info()

    if not stack_info:
        console.print("[yellow]Redo stack is empty[/yellow]")
        return

    console.print("[bold cyan]Redo Stack[/bold cyan]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Message", max_width=60)
    table.add_column("Files", justify="right", width=6)
    table.add_column("Git", width=4)

    for i, info in enumerate(stack_info, 1):
        table.add_row(
            str(i),
            info["user_message"],
            str(info["files_changed"]),
            "[green]Yes[/green]" if info["has_git"] else "[dim]No[/dim]"
        )

    console.print(table)
    console.print()
    console.print(f"[dim]{len(stack_info)} item(s) available for redo[/dim]")
