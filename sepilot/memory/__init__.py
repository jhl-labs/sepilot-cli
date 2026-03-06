"""Memory and checkpoint system for SE Pilot

Includes:
- Session management
- LangGraph checkpointing
- File checkpointing with rewind support
- Project-based history (Claude Code style)
- Environment variable snapshots
- History event logging (JSONL)
"""

from .cache import CacheManager, PromptCache, ToolResultCache
from .checkpoint import CheckpointManager, MemoryCheckpointer, SqliteCheckpointer
from .environment_snapshot import (
    EnvironmentManager,
    EnvironmentSnapshot,
    capture_env_snapshot,
    get_environment_manager,
)
from .history_event import (
    EventType,
    FileAction,
    FileDiff,
    HistoryEvent,
    ToolExecution,
)
from .history_writer import (
    HistoryReader,
    HistoryWriter,
    close_history_writer,
    get_history_reader,
    get_history_writer,
)
from .project_history import (
    ProjectHistoryManager,
    ProjectInfo,
    SessionInfo,
    get_project_history_manager,
)
from .session import Session, SessionHistory, SessionManager

__all__ = [
    # Original exports
    'MemoryCheckpointer',
    'SqliteCheckpointer',
    'CheckpointManager',
    'SessionManager',
    'Session',
    'SessionHistory',
    'PromptCache',
    'ToolResultCache',
    'CacheManager',
    # Project history
    'ProjectHistoryManager',
    'ProjectInfo',
    'SessionInfo',
    'get_project_history_manager',
    # Environment snapshots
    'EnvironmentManager',
    'EnvironmentSnapshot',
    'get_environment_manager',
    'capture_env_snapshot',
    # History events
    'EventType',
    'FileAction',
    'FileDiff',
    'HistoryEvent',
    'ToolExecution',
    # History writer/reader
    'HistoryWriter',
    'HistoryReader',
    'get_history_writer',
    'get_history_reader',
    'close_history_writer',
]
