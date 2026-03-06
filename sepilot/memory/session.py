"""Session management for maintaining conversation context"""

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Message:
    """Represents a message in the conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: dict[str, Any] | None = None


@dataclass
class Session:
    """Represents a conversation session"""
    session_id: str
    thread_id: str
    started_at: str
    updated_at: str
    messages: list[Message]
    context: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "messages": [asdict(msg) for msg in self.messages],
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create session from dictionary"""
        messages = [Message(**msg) for msg in data["messages"]]
        return cls(
            session_id=data["session_id"],
            thread_id=data["thread_id"],
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            messages=messages,
            context=data["context"],
            metadata=data["metadata"]
        )


class SessionHistory:
    """Manages conversation history across sessions"""

    def __init__(self, storage_path: Path | None = None):
        """Initialize session history

        Args:
            storage_path: Path to store session files
        """
        if storage_path is None:
            storage_path = Path.home() / ".sepilot" / "sessions"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: Session) -> None:
        """Save a session to disk with fsync for durability"""
        session_file = self.storage_path / f"{session.session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    def load_session(self, session_id: str) -> Session | None:
        """Load a session from disk"""
        session_file = self.storage_path / f"{session_id}.json"
        if not session_file.exists():
            return None

        with open(session_file, encoding='utf-8') as f:
            data = json.load(f)
            return Session.from_dict(data)

    def list_sessions(self) -> list[dict[str, str]]:
        """List all available sessions"""
        sessions = []
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data["session_id"],
                        "thread_id": data["thread_id"],
                        "started_at": data["started_at"],
                        "updated_at": data["updated_at"],
                        "message_count": len(data["messages"])
                    })
            except Exception as e:
                # Log error but continue processing other sessions
                import logging
                logging.warning(f"Failed to load session from {session_file}: {e}")
                continue
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def search_sessions(self, query: str) -> list[dict[str, Any]]:
        """Search sessions by content"""
        results = []
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, encoding='utf-8') as f:
                    data = json.load(f)
                    # Search in messages
                    for msg in data["messages"]:
                        if query.lower() in msg["content"].lower():
                            results.append({
                                "session_id": data["session_id"],
                                "thread_id": data["thread_id"],
                                "message": msg["content"][:200],
                                "timestamp": msg["timestamp"]
                            })
                            break
            except Exception as e:
                # Log error but continue searching other sessions
                import logging
                logging.warning(f"Failed to search in session {session_file}: {e}")
                continue
        return results


class SessionManager:
    """High-level session management with thread safety"""

    def __init__(self, history: SessionHistory | None = None, max_buffer_size: int = 1000):
        """Initialize session manager

        Args:
            history: SessionHistory instance
            max_buffer_size: Maximum number of messages to keep in buffer (prevents memory leak)
        """
        import threading
        self.history = history or SessionHistory()
        self.current_session: Session | None = None
        # Use deque with maxlen to prevent unbounded memory growth
        self._message_buffer: deque = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()

    def start_session(self, session_id: str | None = None, thread_id: str | None = None) -> Session:
        """Start a new session or resume existing one

        Args:
            session_id: Optional session ID to resume
            thread_id: Optional thread ID to associate

        Returns:
            Session object
        """
        if session_id:
            # Try to resume existing session
            session = self.history.load_session(session_id)
            if session:
                self.current_session = session
                return session

        # Create new session
        import uuid
        session_id = session_id or str(uuid.uuid4())
        thread_id = thread_id or str(uuid.uuid4())

        self.current_session = Session(
            session_id=session_id,
            thread_id=thread_id,
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            messages=[],
            context={},
            metadata={}
        )
        return self.current_session

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a message to the current session (thread-safe)"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        with self._lock:
            self.current_session.messages.append(message)
            self.current_session.updated_at = datetime.now().isoformat()

        # Also add to buffer for quick access
        self._message_buffer.append(message)

    def update_context(self, key: str, value: Any) -> None:
        """Update session context"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        self.current_session.context[key] = value
        self.current_session.updated_at = datetime.now().isoformat()

    def get_recent_messages(self, limit: int = 10) -> list[Message]:
        """Get recent messages from current session"""
        if not self.current_session:
            return []
        return self.current_session.messages[-limit:]

    def get_context_summary(self) -> str:
        """Get a summary of the current context"""
        if not self.current_session:
            return "No active session"

        messages = self.get_recent_messages(5)
        if not messages:
            return "No messages in session"

        summary_lines = [
            f"Session: {self.current_session.session_id[:8]}",
            f"Messages: {len(self.current_session.messages)}",
            "Recent conversation:"
        ]

        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_lines.append(f"  {role}: {content}")

        return "\n".join(summary_lines)

    def save_current_session(self) -> None:
        """Save the current session to disk"""
        if self.current_session:
            self.history.save_session(self.current_session)

    def end_session(self) -> None:
        """End the current session and save it"""
        if self.current_session:
            self.save_current_session()
            self.current_session = None
            self._message_buffer.clear()

    def get_session_for_langraph(self) -> dict[str, Any]:
        """Get session data formatted for LangGraph context"""
        if not self.current_session:
            return {}

        recent_messages = self.get_recent_messages(10)
        message_history = []
        for msg in recent_messages:
            if msg.role == "user":
                message_history.append(f"User: {msg.content}")
            else:
                message_history.append(f"Assistant: {msg.content}")

        return {
            "session_id": self.current_session.session_id,
            "thread_id": self.current_session.thread_id,
            "message_history": "\n".join(message_history),
            "context": self.current_session.context,
            "message_count": len(self.current_session.messages)
        }

    def export_session(
        self,
        output_path: Path | str,
        format: str = "json",
        include_metadata: bool = True,
    ) -> None:
        """Export current session to file

        Args:
            output_path: Path to export file
            format: Export format ("json" or "markdown")
            include_metadata: Include session metadata

        Raises:
            ValueError: If no active session
        """
        if not self.current_session:
            raise ValueError("No active session to export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._export_json(output_path, include_metadata)
        elif format == "markdown":
            self._export_markdown(output_path, include_metadata)
        else:
            raise ValueError(f"Unknown export format: {format}")

    def _export_json(self, output_path: Path, include_metadata: bool) -> None:
        """Export session as JSON"""
        data = self.current_session.to_dict()
        if not include_metadata:
            data.pop("metadata", None)
            data.pop("context", None)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_markdown(self, output_path: Path, include_metadata: bool) -> None:
        """Export session as Markdown"""
        session = self.current_session
        lines = [
            f"# Session: {session.session_id[:8]}",
            "",
            f"**Started:** {session.started_at}",
            f"**Updated:** {session.updated_at}",
            f"**Messages:** {len(session.messages)}",
            "",
            "---",
            "",
            "## Conversation",
            "",
        ]

        for msg in session.messages:
            role_emoji = "U" if msg.role == "user" else "A"
            timestamp = msg.timestamp[:19]  # Trim milliseconds
            lines.append(f"### [{role_emoji}] {timestamp}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")

        if include_metadata and session.context:
            lines.extend([
                "---",
                "",
                "## Context",
                "",
                "```json",
                json.dumps(session.context, indent=2, ensure_ascii=False),
                "```",
            ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def import_session(self, input_path: Path | str) -> Session:
        """Import session from file

        Args:
            input_path: Path to import file (JSON format)

        Returns:
            Imported Session object
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Session file not found: {input_path}")

        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        session = Session.from_dict(data)

        # Optionally set as current session
        self.current_session = session

        # Save to session history
        self.history.save_session(session)

        return session
