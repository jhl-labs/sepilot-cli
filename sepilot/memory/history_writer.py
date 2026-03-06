"""JSONL History Writer for SE Pilot

This module provides:
- Append-only JSONL file writing for event logs
- Atomic writes with file locking
- Event streaming and batching support
- History file reading and parsing
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterator

# Platform-compatible file locking
if sys.platform == "win32":
    import msvcrt

    def _lock_file(f):
        """Acquire exclusive lock on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_file(f):
        """Release lock on Windows."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock_file(f):
        """Acquire exclusive lock on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(f):
        """Release lock on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

from .history_event import EventType, HistoryEvent
from .project_history import ProjectHistoryManager, get_project_history_manager


class HistoryWriter:
    """Writes history events to JSONL files.

    Features:
    - Append-only writes for durability
    - File locking for concurrent access safety
    - Automatic sequence numbering
    - Event buffering for batch writes
    """

    def __init__(
        self,
        project_path: str,
        session_id: str,
        history_manager: ProjectHistoryManager | None = None,
        buffer_size: int = 10,
        include_file_content: bool = False
    ):
        """Initialize the history writer.

        Args:
            project_path: Path to the project directory
            session_id: Current session ID
            history_manager: ProjectHistoryManager instance
            buffer_size: Number of events to buffer before flushing
            include_file_content: Whether to include full file content in diffs
        """
        self.project_path = os.path.abspath(project_path)
        self.session_id = session_id
        self.history_manager = history_manager or get_project_history_manager()
        self.buffer_size = buffer_size
        self.include_file_content = include_file_content

        # Get the session file path
        self.session_file = self.history_manager.get_session_file(
            self.project_path,
            self.session_id
        )

        # Event buffer
        self._buffer: list[HistoryEvent] = []
        self._sequence_number = 0
        self._last_event_id: str | None = None

        # Load existing sequence number if resuming
        self._load_sequence_number()

    def _load_sequence_number(self) -> None:
        """Load the last sequence number from existing file."""
        if not self.session_file.exists():
            return

        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                # Read last line
                lines = f.readlines()
                if lines:
                    last_event = json.loads(lines[-1])
                    self._sequence_number = last_event.get('sequence_number', 0) + 1
                    self._last_event_id = last_event.get('event_id')
        except (json.JSONDecodeError, OSError):
            pass

    def _write_event_to_file(self, event: HistoryEvent) -> None:
        """Write a single event to the JSONL file with locking.

        Args:
            event: Event to write
        """
        event_dict = event.to_dict(include_file_content=self.include_file_content)
        event_line = json.dumps(event_dict, ensure_ascii=False) + '\n'

        # Ensure parent directory exists
        self.session_file.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode with locking
        with open(self.session_file, 'a', encoding='utf-8') as f:
            _lock_file(f)
            try:
                f.write(event_line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                _unlock_file(f)

    def write_event(self, event: HistoryEvent, flush: bool = False) -> str:
        """Write an event to the history.

        Args:
            event: Event to write
            flush: Force immediate write (bypass buffer)

        Returns:
            Event ID
        """
        # Set sequence number
        event.sequence_number = self._sequence_number
        self._sequence_number += 1

        # Set parent if not specified
        if event.parent_event_id is None and self._last_event_id:
            event.parent_event_id = self._last_event_id

        self._last_event_id = event.event_id

        if flush or self.buffer_size <= 1:
            self._write_event_to_file(event)
            # Update project event count
            self.history_manager.add_event_count(self.project_path, 1)
        else:
            self._buffer.append(event)
            if len(self._buffer) >= self.buffer_size:
                self.flush()

        return event.event_id

    def flush(self) -> int:
        """Flush buffered events to disk.

        Returns:
            Number of events flushed
        """
        if not self._buffer:
            return 0

        count = len(self._buffer)

        # Write all buffered events
        for event in self._buffer:
            self._write_event_to_file(event)

        # Update project event count
        self.history_manager.add_event_count(self.project_path, count)

        self._buffer.clear()
        return count

    def close(self) -> None:
        """Close the writer and flush remaining events."""
        self.flush()

    def __enter__(self) -> 'HistoryWriter':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # Convenience methods for common event types

    def session_start(
        self,
        project_path: str | None = None,
        git_branch: str | None = None,
        summary: str | None = None
    ) -> str:
        """Record session start event."""
        event = HistoryEvent.session_start(
            session_id=self.session_id,
            project_path=project_path or self.project_path,
            git_branch=git_branch,
            summary=summary
        )
        # Update project session count
        self.history_manager.increment_session_count(self.project_path)
        return self.write_event(event, flush=True)

    def session_end(self, summary: str | None = None) -> str:
        """Record session end event."""
        event = HistoryEvent.session_end(
            session_id=self.session_id,
            parent_event_id=self._last_event_id,
            summary=summary
        )
        return self.write_event(event, flush=True)

    def user_message(self, content: str) -> str:
        """Record a user message."""
        event = HistoryEvent.user_message(
            session_id=self.session_id,
            content=content,
            parent_event_id=self._last_event_id
        )
        return self.write_event(event, flush=True)

    def assistant_message(self, content: str) -> str:
        """Record an assistant message."""
        event = HistoryEvent.assistant_message(
            session_id=self.session_id,
            content=content,
            parent_event_id=self._last_event_id
        )
        return self.write_event(event)

    def tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Record a tool call."""
        event = HistoryEvent.tool_call(
            session_id=self.session_id,
            tool_name=tool_name,
            tool_args=tool_args,
            parent_event_id=self._last_event_id
        )
        return self.write_event(event)

    def tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: int,
        success: bool = True,
        error: str | None = None
    ) -> str:
        """Record a tool result."""
        event = HistoryEvent.tool_result(
            session_id=self.session_id,
            tool_name=tool_name,
            result=result,
            duration_ms=duration_ms,
            success=success,
            error=error,
            parent_event_id=self._last_event_id
        )
        return self.write_event(event)

    def file_change(
        self,
        file_path: str,
        old_content: str | None,
        new_content: str | None
    ) -> str:
        """Record a file change with diff."""
        event = HistoryEvent.file_change(
            session_id=self.session_id,
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            parent_event_id=self._last_event_id
        )
        return self.write_event(event)

    def checkpoint_create(
        self,
        checkpoint_id: str,
        description: str,
        files: list[str]
    ) -> str:
        """Record a checkpoint creation."""
        event = HistoryEvent.checkpoint_create(
            session_id=self.session_id,
            checkpoint_id=checkpoint_id,
            description=description,
            files=files,
            parent_event_id=self._last_event_id
        )
        return self.write_event(event, flush=True)

    def rewind(
        self,
        target_checkpoint_id: str,
        files_restored: list[str]
    ) -> str:
        """Record a rewind operation."""
        event = HistoryEvent.rewind(
            session_id=self.session_id,
            target_checkpoint_id=target_checkpoint_id,
            files_restored=files_restored,
            parent_event_id=self._last_event_id
        )
        return self.write_event(event, flush=True)

    def error(
        self,
        error_message: str,
        error_traceback: str | None = None
    ) -> str:
        """Record an error."""
        event = HistoryEvent.error(
            session_id=self.session_id,
            error_message=error_message,
            error_traceback=error_traceback,
            parent_event_id=self._last_event_id
        )
        return self.write_event(event, flush=True)


class HistoryReader:
    """Reads history events from JSONL files.

    Features:
    - Streaming reads for large files
    - Event filtering and searching
    - Timeline reconstruction
    """

    def __init__(
        self,
        project_path: str,
        history_manager: ProjectHistoryManager | None = None
    ):
        """Initialize the history reader.

        Args:
            project_path: Path to the project directory
            history_manager: ProjectHistoryManager instance
        """
        self.project_path = os.path.abspath(project_path)
        self.history_manager = history_manager or get_project_history_manager()

    def read_session(self, session_id: str) -> list[HistoryEvent]:
        """Read all events from a session.

        Args:
            session_id: Session ID to read

        Returns:
            List of HistoryEvent instances
        """
        session_file = self.history_manager.get_session_file(
            self.project_path,
            session_id
        )

        events = []
        if not session_file.exists():
            return events

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        event_dict = json.loads(line)
                        events.append(HistoryEvent.from_dict(event_dict))
        except (json.JSONDecodeError, OSError):
            pass

        return events

    def stream_session(self, session_id: str) -> Generator[HistoryEvent, None, None]:
        """Stream events from a session file.

        Args:
            session_id: Session ID to read

        Yields:
            HistoryEvent instances
        """
        session_file = self.history_manager.get_session_file(
            self.project_path,
            session_id
        )

        if not session_file.exists():
            return

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event_dict = json.loads(line)
                            yield HistoryEvent.from_dict(event_dict)
                        except json.JSONDecodeError:
                            continue
        except OSError:
            return

    def filter_events(
        self,
        session_id: str,
        event_types: list[EventType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        tool_name: str | None = None,
        file_path: str | None = None
    ) -> Generator[HistoryEvent, None, None]:
        """Filter events by criteria.

        Args:
            session_id: Session ID to read
            event_types: Filter by event types
            since: Filter events after this time
            until: Filter events before this time
            tool_name: Filter by tool name
            file_path: Filter by file path

        Yields:
            Matching HistoryEvent instances
        """
        for event in self.stream_session(session_id):
            # Filter by event type
            if event_types and event.event_type not in event_types:
                continue

            # Filter by time
            event_time = datetime.fromisoformat(event.timestamp)
            if since and event_time < since:
                continue
            if until and event_time > until:
                continue

            # Filter by tool name
            if tool_name:
                if not event.tool_execution or event.tool_execution.tool_name != tool_name:
                    continue

            # Filter by file path
            if file_path:
                if not event.file_diff or event.file_diff.file_path != file_path:
                    continue

            yield event

    def get_file_history(
        self,
        session_id: str,
        file_path: str
    ) -> list[HistoryEvent]:
        """Get all events related to a specific file.

        Args:
            session_id: Session ID
            file_path: Path to file

        Returns:
            List of events related to the file
        """
        return list(self.filter_events(
            session_id,
            event_types=[
                EventType.FILE_CREATE,
                EventType.FILE_MODIFY,
                EventType.FILE_DELETE,
                EventType.FILE_READ
            ],
            file_path=file_path
        ))

    def get_tool_history(
        self,
        session_id: str,
        tool_name: str | None = None
    ) -> list[HistoryEvent]:
        """Get all tool execution events.

        Args:
            session_id: Session ID
            tool_name: Optional specific tool name

        Returns:
            List of tool events
        """
        return list(self.filter_events(
            session_id,
            event_types=[EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.TOOL_ERROR],
            tool_name=tool_name
        ))

    def get_checkpoints(self, session_id: str) -> list[HistoryEvent]:
        """Get all checkpoint events.

        Args:
            session_id: Session ID

        Returns:
            List of checkpoint events
        """
        return list(self.filter_events(
            session_id,
            event_types=[EventType.CHECKPOINT_CREATE]
        ))

    def find_event_by_id(
        self,
        session_id: str,
        event_id: str
    ) -> HistoryEvent | None:
        """Find a specific event by ID.

        Args:
            session_id: Session ID
            event_id: Event ID to find

        Returns:
            HistoryEvent or None
        """
        for event in self.stream_session(session_id):
            if event.event_id == event_id:
                return event
        return None

    def get_event_chain(
        self,
        session_id: str,
        event_id: str
    ) -> list[HistoryEvent]:
        """Get the chain of events leading to a specific event.

        Args:
            session_id: Session ID
            event_id: Target event ID

        Returns:
            List of events from root to target
        """
        # First, load all events
        events = self.read_session(session_id)
        events_by_id = {e.event_id: e for e in events}

        # Build chain by following parent_event_id
        chain = []
        current_id = event_id

        while current_id:
            event = events_by_id.get(current_id)
            if not event:
                break
            chain.insert(0, event)
            current_id = event.parent_event_id

        return chain

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get a summary of a session.

        Args:
            session_id: Session ID

        Returns:
            Summary dict with counts and timing info
        """
        events = self.read_session(session_id)

        if not events:
            return {'event_count': 0}

        event_counts: dict[str, int] = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Find start and end times
        start_time = events[0].timestamp
        end_time = events[-1].timestamp

        # Count unique files
        files_modified = set()
        for event in events:
            if event.file_diff:
                files_modified.add(event.file_diff.file_path)

        # Count errors
        error_count = event_counts.get('error', 0) + event_counts.get('tool_error', 0)

        return {
            'event_count': len(events),
            'event_counts': event_counts,
            'start_time': start_time,
            'end_time': end_time,
            'files_modified': list(files_modified),
            'error_count': error_count
        }


# Convenience functions

_active_writers: dict[str, HistoryWriter] = {}


def get_history_writer(
    project_path: str,
    session_id: str
) -> HistoryWriter:
    """Get or create a history writer for a session.

    Args:
        project_path: Project path
        session_id: Session ID

    Returns:
        HistoryWriter instance
    """
    key = f"{project_path}:{session_id}"

    if key not in _active_writers:
        _active_writers[key] = HistoryWriter(project_path, session_id)

    return _active_writers[key]


def close_history_writer(project_path: str, session_id: str) -> None:
    """Close and remove a history writer.

    Args:
        project_path: Project path
        session_id: Session ID
    """
    key = f"{project_path}:{session_id}"

    if key in _active_writers:
        _active_writers[key].close()
        del _active_writers[key]


def get_history_reader(project_path: str) -> HistoryReader:
    """Get a history reader for a project.

    Args:
        project_path: Project path

    Returns:
        HistoryReader instance
    """
    return HistoryReader(project_path)
