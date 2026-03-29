"""Thread-safe Inbox message system for multi-agent orchestration."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import replace

from sepilot.agent.multi.models import Message, MessageType

logger = logging.getLogger(__name__)


class Inbox:
    """Thread-safe message inbox for inter-agent communication.

    Each registered agent gets a personal message queue. All operations
    are protected by a single lock for simplicity and correctness.
    """

    MAX_HISTORY = 1000  # per-agent history limit

    def __init__(self) -> None:
        self._boxes: dict[str, deque[Message]] = {}
        self._history: dict[str, list[Message]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, agent_id: str) -> None:
        """Register an agent. Idempotent -- re-registering preserves existing messages."""
        with self._lock:
            if agent_id not in self._boxes:
                self._boxes[agent_id] = deque()
            if agent_id not in self._history:
                self._history[agent_id] = []

    def unregister(self, agent_id: str) -> None:
        """Unregister an agent. History is preserved."""
        with self._lock:
            box = self._boxes.pop(agent_id, None)
            if box:
                history = self._history.setdefault(agent_id, [])
                history.extend(list(box))
                if len(history) > self.MAX_HISTORY:
                    del history[: len(history) - self.MAX_HISTORY]

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_message(message: Message) -> Message:
        """Return a detached copy so callers cannot mutate internal state."""
        return replace(message)

    def send(
        self,
        sender: str,
        receiver: str,
        content: str,
        msg_type: MessageType,
        reply_to: str | None = None,
    ) -> Message:
        """Send a message from *sender* to *receiver*. Returns the created Message."""
        msg = Message(
            sender=sender,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            reply_to=reply_to,
        )
        with self._lock:
            box = self._boxes.get(receiver)
            if box is not None:
                box.append(self._clone_message(msg))
            else:
                logger.warning("Inbox: 미등록 receiver '%s'에 메시지 전송 시도 (sender=%s)", receiver, sender)
        return self._clone_message(msg)

    def send_broadcast(
        self,
        sender: str,
        content: str,
        msg_type: MessageType,
    ) -> list[Message]:
        """Broadcast a message to all registered agents except *sender*.

        Each receiver gets its own distinct ``Message`` object (unique id).
        Returns the list of created messages.
        """
        messages: list[Message] = []
        with self._lock:
            for aid, box in self._boxes.items():
                if aid == sender:
                    continue
                msg = Message(
                    sender=sender,
                    receiver=aid,
                    content=content,
                    msg_type=msg_type,
                )
                box.append(self._clone_message(msg))
                messages.append(self._clone_message(msg))
        return messages

    # ------------------------------------------------------------------
    # Receive / Peek
    # ------------------------------------------------------------------

    def receive(self, agent_id: str) -> list[Message]:
        """Consume all pending messages for *agent_id* (moved to history).

        Returns ``[]`` if the agent is unregistered or has no messages.
        """
        with self._lock:
            box = self._boxes.get(agent_id)
            if box is None or len(box) == 0:
                return []
            messages = list(box)
            box.clear()
            history = self._history.get(agent_id)
            if history is not None:
                history.extend(messages)
                if len(history) > self.MAX_HISTORY:
                    del history[: len(history) - self.MAX_HISTORY]
        return [self._clone_message(message) for message in messages]

    def peek(self, agent_id: str) -> list[Message]:
        """Read pending messages without consuming them.

        Returns a *copy* of the queue so external mutation is safe.
        """
        with self._lock:
            box = self._boxes.get(agent_id)
            if box is None:
                return []
            return [self._clone_message(message) for message in box]

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, agent_id: str) -> list[Message]:
        """Return all consumed messages for *agent_id*. Returns ``[]`` if unknown."""
        with self._lock:
            history = self._history.get(agent_id)
            if history is None:
                return []
            return [self._clone_message(message) for message in history]
