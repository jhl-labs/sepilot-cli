"""Memory management for persistent notes (#note system).

This module follows the Single Responsibility Principle (SRP) by handling
only memory/note persistence and commands.

Provides:
- Save persistent notes (#note)
- List saved notes (#list)
- Clear notes (#clear)
- View specific note (#<id>)
- Memory context injection for prompts
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console


@dataclass
class MemoryEntry:
    """A single memory/note entry."""

    content: str
    timestamp: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return {"content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> 'MemoryEntry':
        """Create from dictionary."""
        return cls(content=data["content"], timestamp=data["timestamp"])


class MemoryManager:
    """Manages persistent memory notes.

    This class handles loading, saving, and manipulating memory entries
    that persist across sessions and can be injected into prompts.
    """

    def __init__(
        self,
        memory_file: str | None = None,
        console: Console | None = None,
    ):
        """Initialize memory manager.

        Args:
            memory_file: Path to memory JSON file
            console: Rich console for output
        """
        self.console = console or Console()

        if memory_file is None:
            history_dir = os.path.join(os.path.expanduser("~"), ".sepilot")
            os.makedirs(history_dir, exist_ok=True)
            memory_file = os.path.join(history_dir, "memory.json")

        self.memory_file = memory_file
        self.entries: list[MemoryEntry] = self._load_memory()

    def _load_memory(self) -> list[MemoryEntry]:
        """Load memory entries from disk.

        Returns:
            List of MemoryEntry objects
        """
        if not os.path.exists(self.memory_file):
            return []

        try:
            with open(self.memory_file, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [MemoryEntry.from_dict(d) for d in data]
        except Exception:
            pass

        return []

    def _save_memory(self) -> bool:
        """Persist memory entries to disk.

        Returns:
            True if save successful
        """
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self.entries], f, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            self.console.print(f"[red]Failed to save memory: {exc}[/red]")
            return False

    def add_note(self, content: str) -> int:
        """Add a new memory note.

        Args:
            content: Note content

        Returns:
            Index of the new note (1-based)
        """
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.entries.append(entry)
        self._save_memory()
        return len(self.entries)

    def list_notes(self, limit: int = 20) -> list[tuple[int, MemoryEntry]]:
        """Get recent notes with indices.

        Args:
            limit: Maximum notes to return

        Returns:
            List of (1-based index, MemoryEntry) tuples
        """
        recent = self.entries[-limit:]
        start_idx = max(0, len(self.entries) - limit)
        return [(i + start_idx + 1, entry) for i, entry in enumerate(recent)]

    def get_note(self, index: int) -> MemoryEntry | None:
        """Get a specific note by 1-based index.

        Args:
            index: 1-based index

        Returns:
            MemoryEntry or None if not found
        """
        idx = index - 1
        if 0 <= idx < len(self.entries):
            return self.entries[idx]
        return None

    def clear_notes(self) -> int:
        """Clear all notes.

        Returns:
            Number of notes cleared
        """
        count = len(self.entries)
        self.entries.clear()
        self._save_memory()
        return count

    def delete_note(self, index: int) -> bool:
        """Delete a specific note by 1-based index.

        Args:
            index: 1-based index

        Returns:
            True if deleted
        """
        idx = index - 1
        if 0 <= idx < len(self.entries):
            del self.entries[idx]
            self._save_memory()
            return True
        return False

    def get_context_injection(self) -> str:
        """Get memory context for prompt injection.

        Returns:
            Formatted memory context string, empty if no entries
        """
        if not self.entries:
            return ""

        context = "\n# User Memory (Persistent Instructions)\n"
        context += "The user has saved the following important notes. Always follow these instructions:\n"

        for entry in self.entries:
            context += f"- {entry.content}\n"

        context += "\n# User Request\n"
        return context

    def handle_command(self, body: str) -> None:
        """Process a # memory command.

        Args:
            body: Command body after # (e.g., "note text", "list", "clear", "1")
        """
        if not body:
            self.console.print("[yellow]Usage:[/yellow] `#note text`, `#list`, `#clear`, `#<id>`")
            return

        lowered = body.lower()

        # List command
        if lowered in {"list", "ls"}:
            self._cmd_list()
            return

        # Clear command
        if lowered.startswith("clear"):
            count = self.clear_notes()
            self.console.print(f"[red]Memory cleared ({count} notes).[/red]")
            return

        # View specific note by ID
        if body.isdigit():
            self._cmd_view(int(body))
            return

        # Delete specific note
        if lowered.startswith("delete ") or lowered.startswith("del "):
            parts = body.split(maxsplit=1)
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1])
                if self.delete_note(idx):
                    self.console.print(f"[red]Deleted memory #{idx}[/red]")
                else:
                    self.console.print(f"[red]No memory entry #{idx}[/red]")
            return

        # Default: save as note
        idx = self.add_note(body)
        self.console.print(f"[green]Saved memory #{idx}[/green] {body}")

    def _cmd_list(self) -> None:
        """Show list of saved notes."""
        if not self.entries:
            self.console.print("[yellow]No saved memories yet.[/yellow]")
            return

        lines = []
        for idx, entry in self.list_notes():
            lines.append(
                f"[cyan]{idx:>2}[/cyan] {entry.content} [dim]({entry.timestamp})[/dim]"
            )

        self.console.print("[bold magenta]Memory Notes[/bold magenta]")
        self.console.print("\n".join(lines))

    def _cmd_view(self, index: int) -> None:
        """View a specific note."""
        entry = self.get_note(index)
        if entry:
            self.console.print(f"[bold magenta]Memory #{index}[/bold magenta]")
            self.console.print(entry.content)
            self.console.print(f"[dim]{entry.timestamp}[/dim]")
        else:
            self.console.print(f"[red]No memory entry #{index}[/red]")

    @property
    def count(self) -> int:
        """Get number of saved notes."""
        return len(self.entries)

    def __len__(self) -> int:
        """Get number of saved notes."""
        return len(self.entries)

    def __bool__(self) -> bool:
        """Check if any notes exist."""
        return len(self.entries) > 0


# Convenience function
def get_memory_manager(
    memory_file: str | None = None, console: Console | None = None
) -> MemoryManager:
    """Get or create a memory manager instance.

    Args:
        memory_file: Optional path to memory file
        console: Optional console for output

    Returns:
        MemoryManager instance
    """
    return MemoryManager(memory_file=memory_file, console=console)
