"""Thread and session management for LangGraph agents.

This module follows the Single Responsibility Principle (SRP) by handling
only thread/session management operations.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any

from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from rich.console import Console

from sepilot.agent.execution_context import (
    get_message_content,
    is_user_turn_boundary_message,
    is_user_visible_conversation_message,
)


class ThreadManager:
    """Manages conversation threads and sessions for LangGraph agents.

    This class handles:
    - Thread switching (resume)
    - Thread listing
    - Thread creation (new)
    - Message rewinding
    - Session state queries
    """

    def __init__(
        self,
        graph: Any,
        checkpointer: Any,
        logger: Any,
        console: Console | None = None,
        enable_memory: bool = True,
    ):
        """Initialize thread manager.

        Args:
            graph: LangGraph compiled graph
            checkpointer: LangGraph checkpointer instance
            logger: Logger instance
            console: Rich console for output
            enable_memory: Whether memory/persistence is enabled
        """
        self.graph = graph
        self.checkpointer = checkpointer
        self.logger = logger
        self.console = console
        self.enable_memory = enable_memory
        self._thread_id: str | None = None

    @property
    def thread_id(self) -> str | None:
        """Get current thread ID."""
        return self._thread_id

    @thread_id.setter
    def thread_id(self, value: str | None) -> None:
        """Set current thread ID."""
        self._thread_id = value

    def get_session_summary(self) -> str:
        """Get a summary of the current session using LangGraph checkpoints.

        Returns:
            Session summary string
        """
        if not self.enable_memory:
            return "No active session (memory disabled)"

        try:
            config = {"configurable": {"thread_id": self._thread_id}}
            state = self.graph.get_state(config)

            if state and state.values:
                messages = state.values.get("messages", [])
                visible_messages = [msg for msg in messages if is_user_visible_conversation_message(msg)]
                return f"Session has {len(visible_messages)} messages"
            return "No messages in current session"
        except Exception as e:
            return f"Error retrieving session: {str(e)}"

    def switch_thread(self, new_thread_id: str) -> bool:
        """Switch to a different thread (Claude Code style resume).

        Args:
            new_thread_id: Thread ID to switch to

        Returns:
            True if switch was successful, False otherwise
        """
        if not self.enable_memory:
            self.logger.log_error("Cannot switch thread: memory is disabled")
            return False

        try:
            # Verify thread exists by trying to get its state directly
            # (cheaper and more reliable than listing all threads)
            config = {"configurable": {"thread_id": new_thread_id}}
            state = self.graph.get_state(config)

            # StateSnapshot is always returned; check created_at to verify
            # the thread actually has checkpoints saved
            if not state or (not state.values and not getattr(state, 'created_at', None)):
                self.logger.log_error(f"Thread {new_thread_id} not found or empty")
                return False

            # Update thread ID
            old_thread_id = self._thread_id
            self._thread_id = new_thread_id

            # Log the switch
            self.logger.log_event(
                "thread_switch",
                {
                    "from_thread": old_thread_id,
                    "to_thread": new_thread_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

            return True

        except Exception as e:
            self.logger.log_error(f"Failed to switch thread: {str(e)}")
            import traceback
            self.logger.log_trace("switch_thread_error", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return False

    def list_available_threads(self) -> list[dict[str, Any]]:
        """List all available threads with metadata (Claude Code style).

        Returns:
            List of thread metadata dictionaries
        """
        if not self.enable_memory:
            return []

        try:
            # Use LangGraph's checkpointer.list() method
            if not hasattr(self.checkpointer, 'list'):
                return []

            # Get all checkpoints and group by thread
            threads_data = defaultdict(list)

            for checkpoint_tuple in self.checkpointer.list(None):
                thread_id = checkpoint_tuple.config.get('configurable', {}).get('thread_id')
                if thread_id:
                    threads_data[thread_id].append(checkpoint_tuple)

            # Process each thread to extract metadata
            result = []
            for thread_id, checkpoints in threads_data.items():
                if not checkpoints:
                    continue

                thread_info = self._extract_thread_info(thread_id, checkpoints)
                result.append(thread_info)

            # Sort by updated_at (most recent first)
            result.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

            return result

        except Exception as e:
            self.logger.log_error(f"Failed to list threads: {str(e)}")
            import traceback
            self.logger.log_trace("list_threads_error", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return []

    def _extract_thread_info(
        self, thread_id: str, checkpoints: list
    ) -> dict[str, Any]:
        """Extract thread metadata from checkpoints.

        Args:
            thread_id: Thread ID
            checkpoints: List of checkpoint tuples

        Returns:
            Thread info dictionary
        """
        latest_checkpoint = checkpoints[0]
        oldest_checkpoint = checkpoints[-1]

        thread_info = {
            "thread_id": thread_id,
            "checkpoint_count": len(checkpoints),
            "created_at": None,
            "updated_at": None,
            "message_count": 0,
            "first_message_preview": None,
            "last_message_preview": None,
        }

        # Extract timestamps from checkpoint metadata
        thread_info["updated_at"] = self._extract_checkpoint_timestamp(latest_checkpoint)
        thread_info["created_at"] = self._extract_checkpoint_timestamp(oldest_checkpoint)

        # Fallback: extract timestamp from thread_id (e.g. thread_20260306_073136_xxx)
        if not thread_info["created_at"]:
            thread_info["created_at"] = self._extract_timestamp_from_thread_id(thread_id)
        if not thread_info["updated_at"]:
            thread_info["updated_at"] = thread_info["created_at"]

        # Try to get message count and previews
        try:
            if isinstance(latest_checkpoint.checkpoint, dict):
                channel_values = latest_checkpoint.checkpoint.get('channel_values', {})
                messages = channel_values.get('messages', [])

                visible_messages = [
                    msg for msg in messages if is_user_visible_conversation_message(msg)
                ]
                thread_info["message_count"] = len(visible_messages)

                if visible_messages:
                    # First user message = conversation topic
                    for msg in visible_messages:
                        if is_user_turn_boundary_message(msg):
                            content = get_message_content(msg).replace('\n', ' ').strip()
                            if len(content) > 80:
                                content = content[:77] + "..."
                            thread_info["first_message_preview"] = content
                            break

                    # Last message preview
                    last_msg = visible_messages[-1]
                    content = get_message_content(last_msg)
                    if len(content) > 100:
                        content = content[:100] + "..."
                    thread_info["last_message_preview"] = content

        except Exception as e:
            self.logger.log_trace("message_extraction_failed", {
                "thread_id": thread_id,
                "error": str(e)
            })

        return thread_info

    def _extract_timestamp_from_thread_id(self, thread_id: str) -> str | None:
        """Extract timestamp from thread_id format: thread_YYYYMMDD_HHMMSS_xxx."""
        import re
        m = re.match(r'thread_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', thread_id)
        if m:
            try:
                dt = datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)), int(m.group(6))
                )
                return dt.isoformat()
            except ValueError:
                pass
        return None

    def _extract_checkpoint_timestamp(self, checkpoint_tuple: Any) -> str | None:
        """Extract timestamp from a checkpoint tuple.

        Tries multiple sources in order of preference:
        1. metadata.created_at or metadata.timestamp
        2. checkpoint_id (often contains timestamp info)
        3. Fallback to None

        Args:
            checkpoint_tuple: LangGraph checkpoint tuple

        Returns:
            ISO format timestamp string or None
        """
        try:
            # Try metadata first (most reliable)
            if checkpoint_tuple.metadata:
                metadata = checkpoint_tuple.metadata

                # Check common timestamp fields
                for field in ['created_at', 'timestamp', 'ts', 'time', 'updated_at']:
                    if field in metadata:
                        ts_value = metadata[field]
                        return self._normalize_timestamp(ts_value)

                # Check step field for relative ordering
                if 'step' in metadata:
                    # Step can help with ordering but not absolute time
                    pass

            # Try checkpoint_id (some implementations embed timestamp)
            if hasattr(checkpoint_tuple, 'config') and checkpoint_tuple.config:
                config = checkpoint_tuple.config
                checkpoint_id = config.get('configurable', {}).get('checkpoint_id')
                if checkpoint_id:
                    # Try to parse UUID v1 timestamp (if applicable)
                    ts = self._extract_timestamp_from_id(checkpoint_id)
                    if ts:
                        return ts

            # Try parent_config for additional info
            if hasattr(checkpoint_tuple, 'parent_config') and checkpoint_tuple.parent_config:
                parent_ts = checkpoint_tuple.parent_config.get('configurable', {}).get('checkpoint_ts')
                if parent_ts:
                    return self._normalize_timestamp(parent_ts)

        except Exception as e:
            self.logger.log_trace("timestamp_extraction_failed", {"error": str(e)})

        return None

    def _normalize_timestamp(self, ts_value: Any) -> str | None:
        """Normalize various timestamp formats to ISO string.

        Args:
            ts_value: Timestamp in various formats

        Returns:
            ISO format string or None
        """
        if ts_value is None:
            return None

        # Already a string
        if isinstance(ts_value, str):
            try:
                # Validate it's a valid ISO format
                datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
                return ts_value
            except ValueError:
                pass

        # datetime object
        if isinstance(ts_value, datetime):
            return ts_value.isoformat()

        # Unix timestamp (int or float)
        if isinstance(ts_value, (int, float)):
            try:
                # Handle both seconds and milliseconds
                if ts_value > 1e12:  # Likely milliseconds
                    ts_value = ts_value / 1000
                return datetime.fromtimestamp(ts_value).isoformat()
            except (ValueError, OSError):
                pass

        return None

    def _extract_timestamp_from_id(self, checkpoint_id: str) -> str | None:
        """Try to extract timestamp from checkpoint ID.

        Some checkpoint IDs are UUID v1 which contain timestamp info.

        Args:
            checkpoint_id: Checkpoint ID string

        Returns:
            ISO format string or None
        """
        try:
            import uuid
            parsed_uuid = uuid.UUID(checkpoint_id)

            # UUID v1 contains timestamp
            if parsed_uuid.version == 1:
                # UUID v1 timestamp is 100-nanosecond intervals since Oct 15, 1582
                timestamp = (parsed_uuid.time - 0x01b21dd213814000) / 1e7
                return datetime.fromtimestamp(timestamp).isoformat()
        except (ValueError, AttributeError, ImportError):
            pass

        return None

    def _extract_message_content(self, message: Any) -> str:
        """Extract content from a message object.

        Args:
            message: Message object or dict

        Returns:
            Message content string
        """
        if hasattr(message, 'content'):
            return message.content
        elif isinstance(message, dict):
            return message.get('content', str(message))
        return str(message)

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics from LangGraph checkpoints.

        Returns:
            Memory stats dictionary
        """
        if not self.enable_memory:
            return {}

        try:
            config = {"configurable": {"thread_id": self._thread_id}}
            state = self.graph.get_state(config)

            stats = {
                "thread_id": self._thread_id,
                "checkpoint_enabled": self.enable_memory,
            }

            if state:
                messages = state.values.get("messages", []) if state.values else []
                stats["message_count"] = sum(
                    1 for msg in messages if is_user_visible_conversation_message(msg)
                )

            return stats
        except Exception as e:
            return {"error": str(e)}

    def create_new_thread(self) -> str:
        """Create a new conversation thread (Claude Code style /new).

        Returns:
            The new thread ID
        """
        import uuid

        old_thread_id = self._thread_id
        new_thread_id = str(uuid.uuid4())

        self._thread_id = new_thread_id

        self.logger.log_event(
            "thread_created",
            {
                "old_thread": old_thread_id,
                "new_thread": new_thread_id,
                "timestamp": datetime.now().isoformat()
            }
        )

        return new_thread_id

    def rewind_messages(self, count: int = 1) -> dict[str, Any]:
        """Rewind conversation by removing recent messages (Claude Code style /rewind).

        Args:
            count: Number of user-assistant pairs to remove (default: 1)

        Returns:
            Dict with status and info about removed messages
        """
        if not self.enable_memory:
            return {"success": False, "error": "Memory is disabled"}

        try:
            config = {"configurable": {"thread_id": self._thread_id}}
            state = self.graph.get_state(config)

            if not state or not state.values:
                return {"success": False, "error": "No state found"}

            messages = state.values.get("messages", [])

            if not messages:
                return {"success": False, "error": "No messages to rewind"}

            # Calculate how many messages to remove
            removed_count, pairs_found, messages_to_keep = self._calculate_rewind(
                messages, count
            )

            # Replace the thread message list exactly so no stale tail survives.
            replacement_ops = [RemoveMessage(id=REMOVE_ALL_MESSAGES, content=""), *messages_to_keep]
            self.graph.update_state(config, {"messages": replacement_ops})

            return {
                "success": True,
                "removed_count": removed_count,
                "pairs_removed": pairs_found,
                "remaining_count": len(messages_to_keep)
            }

        except Exception as e:
            self.logger.log_error(f"Failed to rewind messages: {str(e)}")
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _calculate_rewind(
        self, messages: list, pairs_to_remove: int
    ) -> tuple[int, int, list]:
        """Calculate which messages to remove for rewind.

        Args:
            messages: Current messages list
            pairs_to_remove: Number of user-assistant pairs to remove

        Returns:
            Tuple of (removed_count, pairs_found, messages_to_keep)
        """
        removed_count = 0
        pairs_found = 0
        i = len(messages) - 1

        while i >= 0 and pairs_found < pairs_to_remove:
            msg = messages[i]
            msg_type = getattr(msg, 'type', None)

            # Count only real user turn boundaries, not internal control prompts.
            if msg_type == 'human' and is_user_turn_boundary_message(msg):
                pairs_found += 1

            removed_count += 1
            i -= 1

        messages_to_keep = messages[:len(messages) - removed_count]
        return removed_count, pairs_found, messages_to_keep

    def get_conversation_messages(self) -> list[Any]:
        """Get all messages from current conversation thread.

        Returns:
            List of messages in the current thread
        """
        if not self.enable_memory:
            return []

        try:
            config = {"configurable": {"thread_id": self._thread_id}}
            state = self.graph.get_state(config)

            if state and state.values:
                return state.values.get("messages", [])
            return []
        except Exception:
            return []
