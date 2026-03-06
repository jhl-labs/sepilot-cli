"""Checkpoint system for persisting agent state"""

import base64
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def safe_serialize(obj: Any) -> bytes:
    """Safely serialize object to JSON with fallback to base64 encoding

    This avoids pickle's security vulnerabilities while maintaining compatibility.
    """
    def default_converter(o):
        """Convert non-JSON-serializable objects"""
        # Handle common types
        if hasattr(o, '__dict__'):
            return {'__type__': o.__class__.__name__, '__data__': o.__dict__}
        elif isinstance(o, (set, frozenset)):
            return {'__type__': 'set', '__data__': list(o)}
        elif isinstance(o, bytes):
            return {'__type__': 'bytes', '__data__': base64.b64encode(o).decode('ascii')}
        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        else:
            return str(o)

    try:
        json_str = json.dumps(obj, default=default_converter, ensure_ascii=False)
        return json_str.encode('utf-8')
    except Exception as e:
        # If JSON serialization fails, log warning and use fallback
        import logging
        logging.warning(f"JSON serialization failed, using fallback: {e}")
        # Return a simple representation
        return json.dumps({'error': 'Serialization failed', 'repr': repr(obj)}).encode('utf-8')


def safe_deserialize(data: bytes) -> Any:
    """Safely deserialize JSON data

    This avoids pickle's arbitrary code execution vulnerability.
    """
    def object_hook(dct):
        """Reconstruct objects from JSON"""
        if '__type__' in dct:
            obj_type = dct['__type__']
            obj_data = dct['__data__']

            if obj_type == 'set':
                return set(obj_data)
            elif obj_type == 'bytes':
                return base64.b64decode(obj_data.encode('ascii'))
            # For custom types, return as dict
            return dct
        return dct

    try:
        json_str = data.decode('utf-8')
        return json.loads(json_str, object_hook=object_hook)
    except Exception as e:
        import logging
        logging.error(f"JSON deserialization failed: {e}")
        return None


@dataclass
class Checkpoint:
    """Represents a checkpoint in the agent's execution"""
    thread_id: str
    checkpoint_id: str
    state: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary"""
        return asdict(self)


class MemoryCheckpointer:
    """In-memory checkpointer for agent state"""

    def __init__(self):
        self.checkpoints: dict[str, list[Checkpoint]] = {}
        self.current_states: dict[str, dict[str, Any]] = {}

    def save(self, config: dict[str, Any], state: dict[str, Any]) -> None:
        """Save a checkpoint"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = f"{thread_id}_{datetime.now().isoformat()}"

        checkpoint = Checkpoint(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            state=state,
            metadata=config.get("metadata", {}),
            created_at=datetime.now().isoformat()
        )

        if thread_id not in self.checkpoints:
            self.checkpoints[thread_id] = []
        self.checkpoints[thread_id].append(checkpoint)
        self.current_states[thread_id] = state

    def load(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Load the latest checkpoint for a thread"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")

        if thread_id in self.checkpoints and self.checkpoints[thread_id]:
            return self.checkpoints[thread_id][-1].state
        return None

    def get_history(self, thread_id: str) -> list[Checkpoint]:
        """Get checkpoint history for a thread"""
        return self.checkpoints.get(thread_id, [])

    def clear(self, thread_id: str) -> None:
        """Clear checkpoints for a thread"""
        if thread_id in self.checkpoints:
            del self.checkpoints[thread_id]
        if thread_id in self.current_states:
            del self.current_states[thread_id]


class SqliteCheckpointer:
    """SQLite-based persistent checkpointer"""

    def __init__(self, db_path: Path | None = None):
        """Initialize SQLite checkpointer

        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            # Default to user's home directory
            db_path = Path.home() / ".sepilot" / "checkpoints.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    state BLOB NOT NULL,
                    metadata TEXT,
                    parent_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(thread_id, checkpoint_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_id ON checkpoints(thread_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON checkpoints(created_at)
            """)

    def save(self, config: dict[str, Any], state: dict[str, Any]) -> None:
        """Save a checkpoint to the database"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id") or \
                       f"{thread_id}_{datetime.now().isoformat()}"
        parent_id = config.get("configurable", {}).get("parent_id")
        metadata = json.dumps(config.get("metadata", {}))

        # Serialize state using safe JSON serialization (avoids pickle vulnerabilities)
        state_blob = safe_serialize(state)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_id, state, metadata, parent_id)
                VALUES (?, ?, ?, ?, ?)
            """, (thread_id, checkpoint_id, state_blob, metadata, parent_id))

    def load(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Load the latest checkpoint for a thread"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        with sqlite3.connect(self.db_path) as conn:
            if checkpoint_id:
                # Load specific checkpoint
                cursor = conn.execute("""
                    SELECT state FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_id = ?
                """, (thread_id, checkpoint_id))
            else:
                # Load latest checkpoint
                cursor = conn.execute("""
                    SELECT state FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (thread_id,))

            row = cursor.fetchone()
            if row:
                return safe_deserialize(row[0])
        return None

    def get_history(self, thread_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get checkpoint history for a thread"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT checkpoint_id, metadata, created_at
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (thread_id, limit))

            history = []
            for row in cursor:
                history.append({
                    "checkpoint_id": row["checkpoint_id"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"]
                })
            return history

    def list_threads(self) -> list[str]:
        """List all thread IDs with checkpoints"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT thread_id
                FROM checkpoints
                ORDER BY thread_id
            """)
            return [row[0] for row in cursor]

    def list_threads_with_metadata(self) -> list[dict[str, Any]]:
        """List all threads with detailed metadata (Claude Code style)

        Returns:
            List of dictionaries containing:
                - thread_id: Thread identifier
                - created_at: First checkpoint timestamp
                - updated_at: Last checkpoint timestamp
                - checkpoint_count: Number of checkpoints
                - message_count: Number of messages (if available)
                - last_message_preview: Preview of last message
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    thread_id,
                    MIN(created_at) as created_at,
                    MAX(created_at) as updated_at,
                    COUNT(*) as checkpoint_count
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY updated_at DESC
            """)

            threads = []
            for row in cursor:
                thread_info = {
                    "thread_id": row["thread_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "checkpoint_count": row["checkpoint_count"],
                    "message_count": 0,
                    "last_message_preview": None
                }

                # Try to get message count and last message from latest checkpoint
                try:
                    state_cursor = conn.execute("""
                        SELECT state FROM checkpoints
                        WHERE thread_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (row["thread_id"],))

                    state_row = state_cursor.fetchone()
                    if state_row:
                        state = safe_deserialize(state_row[0])
                        if state and isinstance(state, dict):
                            messages = state.get("messages", [])
                            thread_info["message_count"] = len(messages)

                            # Get last message preview
                            if messages:
                                last_msg = messages[-1]
                                if hasattr(last_msg, "content"):
                                    content = last_msg.content
                                elif isinstance(last_msg, dict):
                                    content = last_msg.get("content", "")
                                else:
                                    content = str(last_msg)

                                # Truncate to 100 chars
                                if len(content) > 100:
                                    content = content[:100] + "..."
                                thread_info["last_message_preview"] = content
                except Exception as e:
                    # If we can't deserialize state, just skip message info
                    import logging
                    logging.debug(f"Could not extract message info for thread {row['thread_id']}: {e}")

                threads.append(thread_info)

            return threads

    def clear(self, thread_id: str) -> None:
        """Clear checkpoints for a thread"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))

    def clear_old(self, days: int = 30) -> int:
        """Clear checkpoints older than specified days

        Args:
            days: Number of days to keep

        Returns:
            Number of checkpoints deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM checkpoints
                WHERE created_at < datetime('now', '-' || ? || ' days')
            """, (days,))
            return cursor.rowcount


class CheckpointManager:
    """High-level checkpoint management"""

    def __init__(self, checkpointer: Any | None = None):
        """Initialize checkpoint manager

        Args:
            checkpointer: Checkpointer instance (Memory or SQLite)
        """
        self.checkpointer = checkpointer or MemoryCheckpointer()
        self._current_thread_id = None

    def start_thread(self, thread_id: str | None = None) -> str:
        """Start a new thread or resume existing one

        Args:
            thread_id: Optional thread ID to resume

        Returns:
            Thread ID (new or resumed)
        """
        if thread_id is None:
            import uuid
            thread_id = str(uuid.uuid4())

        self._current_thread_id = thread_id
        return thread_id

    def save_checkpoint(self, state: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        """Save a checkpoint for the current thread"""
        if not self._current_thread_id:
            raise ValueError("No active thread. Call start_thread() first.")

        config = {
            "configurable": {"thread_id": self._current_thread_id},
            "metadata": metadata or {}
        }
        self.checkpointer.save(config, state)

    def load_checkpoint(self, thread_id: str | None = None) -> dict[str, Any] | None:
        """Load the latest checkpoint"""
        thread_id = thread_id or self._current_thread_id
        if not thread_id:
            return None

        config = {"configurable": {"thread_id": thread_id}}
        return self.checkpointer.load(config)

    def get_config(self) -> dict[str, Any]:
        """Get current configuration for LangGraph"""
        if not self._current_thread_id:
            self.start_thread()

        return {
            "configurable": {
                "thread_id": self._current_thread_id,
                "checkpoint_id": f"{self._current_thread_id}_{datetime.now().isoformat()}"
            }
        }

    @property
    def current_thread_id(self) -> str | None:
        """Get current thread ID"""
        return self._current_thread_id
