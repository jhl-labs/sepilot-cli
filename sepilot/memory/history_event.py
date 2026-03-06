"""Unified History Event System for SE Pilot

This module provides:
- Standardized event types for all agent activities
- Event data structures with full context
- Support for event causality tracking (parent events)
- File change diffs and tool execution records
"""

import difflib
import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Types of events that can be recorded in history"""

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_RESUME = "session_resume"

    # Messages
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"

    # Tool execution
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # File operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_CREATE = "file_create"
    FILE_DELETE = "file_delete"
    FILE_MODIFY = "file_modify"

    # Checkpoints and state
    CHECKPOINT_CREATE = "checkpoint_create"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    REWIND = "rewind"

    # Environment
    ENV_CAPTURE = "env_capture"
    ENV_CHANGE = "env_change"

    # Errors and recovery
    ERROR = "error"
    RECOVERY = "recovery"

    # Git operations
    GIT_COMMIT = "git_commit"
    GIT_OPERATION = "git_operation"

    # Agent state
    AGENT_THINKING = "agent_thinking"
    AGENT_DECISION = "agent_decision"


class FileAction(str, Enum):
    """Types of file actions"""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    READ = "read"


@dataclass
class FileDiff:
    """Represents a diff between two versions of a file"""
    file_path: str
    action: FileAction
    old_content: str | None = None
    new_content: str | None = None
    unified_diff: str | None = None
    old_hash: str | None = None
    new_hash: str | None = None

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    @classmethod
    def create_diff(
        cls,
        file_path: str,
        old_content: str | None,
        new_content: str | None,
        context_lines: int = 3
    ) -> 'FileDiff':
        """Create a FileDiff from old and new content.

        Args:
            file_path: Path to the file
            old_content: Previous content (None if new file)
            new_content: Current content (None if deleted)
            context_lines: Number of context lines in diff

        Returns:
            FileDiff instance
        """
        # Determine action
        if old_content is None and new_content is not None:
            action = FileAction.CREATE
        elif old_content is not None and new_content is None:
            action = FileAction.DELETE
        elif old_content is not None and new_content is not None:
            action = FileAction.MODIFY
        else:
            action = FileAction.READ

        # Compute hashes
        old_hash = cls.compute_hash(old_content) if old_content else None
        new_hash = cls.compute_hash(new_content) if new_content else None

        # Generate unified diff
        unified_diff = None
        if old_content is not None and new_content is not None:
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)

            diff_lines = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                n=context_lines
            )
            unified_diff = ''.join(diff_lines)

        return cls(
            file_path=file_path,
            action=action,
            old_content=old_content,
            new_content=new_content,
            unified_diff=unified_diff,
            old_hash=old_hash,
            new_hash=new_hash
        )

    def to_dict(self, include_content: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            include_content: Whether to include full content (can be large)

        Returns:
            Dictionary representation
        """
        result = {
            'file_path': self.file_path,
            'action': self.action.value,
            'old_hash': self.old_hash,
            'new_hash': self.new_hash,
            'unified_diff': self.unified_diff
        }

        if include_content:
            result['old_content'] = self.old_content
            result['new_content'] = self.new_content

        return result


@dataclass
class ToolExecution:
    """Record of a tool execution"""
    tool_name: str
    tool_args: dict[str, Any]
    start_time: str
    end_time: str | None = None
    duration_ms: int | None = None
    result: Any = None
    error: str | None = None
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'tool_name': self.tool_name,
            'tool_args': self.tool_args,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'result': str(self.result)[:1000] if self.result else None,  # Truncate
            'error': self.error,
            'success': self.success
        }


@dataclass
class HistoryEvent:
    """A single event in the history timeline.

    Events form a causal chain through parent_event_id, allowing
    reconstruction of the sequence of actions that led to any state.
    """
    event_id: str
    event_type: EventType
    timestamp: str
    session_id: str

    # Core data
    data: dict[str, Any] = field(default_factory=dict)

    # Causality
    parent_event_id: str | None = None  # For tracking cause-effect chains
    sequence_number: int = 0  # Order within session

    # Tool execution context
    tool_execution: ToolExecution | None = None

    # File change context
    file_diff: FileDiff | None = None

    # Message context
    message_content: str | None = None
    message_role: str | None = None  # user, assistant, system

    # Checkpoint context
    checkpoint_id: str | None = None

    # Error context
    error_message: str | None = None
    error_traceback: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: EventType,
        session_id: str,
        data: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
        **kwargs
    ) -> 'HistoryEvent':
        """Factory method to create a new event.

        Args:
            event_type: Type of event
            session_id: Session this event belongs to
            data: Event-specific data
            parent_event_id: ID of the event that caused this one
            **kwargs: Additional fields

        Returns:
            New HistoryEvent instance
        """
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            data=data or {},
            parent_event_id=parent_event_id,
            **kwargs
        )

    @classmethod
    def session_start(
        cls,
        session_id: str,
        project_path: str,
        git_branch: str | None = None,
        summary: str | None = None
    ) -> 'HistoryEvent':
        """Create a session start event."""
        return cls.create(
            EventType.SESSION_START,
            session_id=session_id,
            data={
                'project_path': project_path,
                'git_branch': git_branch,
                'summary': summary
            }
        )

    @classmethod
    def session_end(
        cls,
        session_id: str,
        parent_event_id: str | None = None,
        summary: str | None = None
    ) -> 'HistoryEvent':
        """Create a session end event."""
        return cls.create(
            EventType.SESSION_END,
            session_id=session_id,
            data={'summary': summary},
            parent_event_id=parent_event_id
        )

    @classmethod
    def user_message(
        cls,
        session_id: str,
        content: str,
        parent_event_id: str | None = None
    ) -> 'HistoryEvent':
        """Create a user message event."""
        return cls.create(
            EventType.USER_MESSAGE,
            session_id=session_id,
            message_content=content,
            message_role='user',
            parent_event_id=parent_event_id
        )

    @classmethod
    def assistant_message(
        cls,
        session_id: str,
        content: str,
        parent_event_id: str | None = None
    ) -> 'HistoryEvent':
        """Create an assistant message event."""
        return cls.create(
            EventType.ASSISTANT_MESSAGE,
            session_id=session_id,
            message_content=content,
            message_role='assistant',
            parent_event_id=parent_event_id
        )

    @classmethod
    def tool_call(
        cls,
        session_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        parent_event_id: str | None = None
    ) -> 'HistoryEvent':
        """Create a tool call event."""
        tool_exec = ToolExecution(
            tool_name=tool_name,
            tool_args=tool_args,
            start_time=datetime.now().isoformat()
        )
        return cls.create(
            EventType.TOOL_CALL,
            session_id=session_id,
            tool_execution=tool_exec,
            parent_event_id=parent_event_id
        )

    @classmethod
    def tool_result(
        cls,
        session_id: str,
        tool_name: str,
        result: Any,
        duration_ms: int,
        success: bool = True,
        error: str | None = None,
        parent_event_id: str | None = None
    ) -> 'HistoryEvent':
        """Create a tool result event."""
        tool_exec = ToolExecution(
            tool_name=tool_name,
            tool_args={},
            start_time='',
            end_time=datetime.now().isoformat(),
            duration_ms=duration_ms,
            result=result,
            success=success,
            error=error
        )
        return cls.create(
            EventType.TOOL_RESULT if success else EventType.TOOL_ERROR,
            session_id=session_id,
            tool_execution=tool_exec,
            error_message=error,
            parent_event_id=parent_event_id
        )

    @classmethod
    def file_change(
        cls,
        session_id: str,
        file_path: str,
        old_content: str | None,
        new_content: str | None,
        parent_event_id: str | None = None
    ) -> 'HistoryEvent':
        """Create a file change event with diff."""
        diff = FileDiff.create_diff(file_path, old_content, new_content)

        # Determine event type
        if diff.action == FileAction.CREATE:
            event_type = EventType.FILE_CREATE
        elif diff.action == FileAction.DELETE:
            event_type = EventType.FILE_DELETE
        else:
            event_type = EventType.FILE_MODIFY

        return cls.create(
            event_type,
            session_id=session_id,
            file_diff=diff,
            parent_event_id=parent_event_id
        )

    @classmethod
    def checkpoint_create(
        cls,
        session_id: str,
        checkpoint_id: str,
        description: str,
        files: list[str],
        parent_event_id: str | None = None
    ) -> 'HistoryEvent':
        """Create a checkpoint creation event."""
        return cls.create(
            EventType.CHECKPOINT_CREATE,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            data={
                'description': description,
                'files': files
            },
            parent_event_id=parent_event_id
        )

    @classmethod
    def rewind(
        cls,
        session_id: str,
        target_checkpoint_id: str,
        files_restored: list[str],
        parent_event_id: str | None = None
    ) -> 'HistoryEvent':
        """Create a rewind event."""
        return cls.create(
            EventType.REWIND,
            session_id=session_id,
            checkpoint_id=target_checkpoint_id,
            data={
                'files_restored': files_restored
            },
            parent_event_id=parent_event_id
        )

    @classmethod
    def error(
        cls,
        session_id: str,
        error_message: str,
        error_traceback: str | None = None,
        parent_event_id: str | None = None
    ) -> 'HistoryEvent':
        """Create an error event."""
        return cls.create(
            EventType.ERROR,
            session_id=session_id,
            error_message=error_message,
            error_traceback=error_traceback,
            parent_event_id=parent_event_id
        )

    def to_dict(self, include_file_content: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            include_file_content: Whether to include full file content

        Returns:
            Dictionary representation
        """
        result = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'data': self.data,
            'parent_event_id': self.parent_event_id,
            'sequence_number': self.sequence_number,
        }

        # Add optional fields if present
        if self.tool_execution:
            result['tool_execution'] = self.tool_execution.to_dict()

        if self.file_diff:
            result['file_diff'] = self.file_diff.to_dict(include_content=include_file_content)

        if self.message_content:
            result['message_content'] = self.message_content
            result['message_role'] = self.message_role

        if self.checkpoint_id:
            result['checkpoint_id'] = self.checkpoint_id

        if self.error_message:
            result['error_message'] = self.error_message
            result['error_traceback'] = self.error_traceback

        if self.metadata:
            result['metadata'] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'HistoryEvent':
        """Create HistoryEvent from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            HistoryEvent instance
        """
        # Convert event_type string back to enum
        event_type = EventType(data['event_type'])

        # Reconstruct tool_execution if present
        tool_execution = None
        if 'tool_execution' in data and data['tool_execution']:
            tool_execution = ToolExecution(**data['tool_execution'])

        # Reconstruct file_diff if present
        file_diff = None
        if 'file_diff' in data and data['file_diff']:
            diff_data = data['file_diff']
            diff_data['action'] = FileAction(diff_data['action'])
            file_diff = FileDiff(**diff_data)

        return cls(
            event_id=data['event_id'],
            event_type=event_type,
            timestamp=data['timestamp'],
            session_id=data['session_id'],
            data=data.get('data', {}),
            parent_event_id=data.get('parent_event_id'),
            sequence_number=data.get('sequence_number', 0),
            tool_execution=tool_execution,
            file_diff=file_diff,
            message_content=data.get('message_content'),
            message_role=data.get('message_role'),
            checkpoint_id=data.get('checkpoint_id'),
            error_message=data.get('error_message'),
            error_traceback=data.get('error_traceback'),
            metadata=data.get('metadata', {})
        )
