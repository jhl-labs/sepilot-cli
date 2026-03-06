"""Session Handoff Manager for A2A Protocol.

This module manages session handoffs between agents, allowing one agent
to transfer control to another agent for direct user interaction.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class HandoffStatus(Enum):
    """Status of a session handoff."""

    PENDING = "pending"         # Waiting for target agent to accept
    ACTIVE = "active"           # Handoff in progress, target agent has control
    COMPLETED = "completed"     # Handoff completed successfully
    CANCELLED = "cancelled"     # Handoff was cancelled
    FAILED = "failed"           # Handoff failed
    RETURNED = "returned"       # Control returned to original agent


@dataclass
class HandoffContext:
    """Context for a session handoff.

    Contains all information needed to transfer control from one agent
    to another while preserving session state.

    Attributes:
        handoff_id: Unique handoff identifier
        from_agent: Agent initiating the handoff
        to_agent: Target agent receiving control
        task: Task description for the target agent
        original_context: Session context from the original agent
        started_at: Handoff start time
        completed_at: Handoff completion time (if completed)
        status: Current handoff status
        result: Result from the handoff (after completion)
        metadata: Additional handoff metadata
    """

    from_agent: str
    to_agent: str
    task: str
    handoff_id: str = field(default_factory=lambda: str(uuid4()))
    original_context: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    status: HandoffStatus = HandoffStatus.PENDING
    result: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Call chain for circular detection
    call_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert handoff context to dictionary."""
        return {
            "handoff_id": self.handoff_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "task": self.task,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "call_chain": self.call_chain
        }


class SessionHandoffManager:
    """Manages session handoffs between agents.

    Handles the lifecycle of session transfers, including:
    - Initiating handoffs
    - Tracking active handoffs
    - Completing or cancelling handoffs
    - Managing session context preservation

    Attributes:
        active_handoffs: Currently active handoffs
        completed_handoffs: History of completed handoffs
        handoff_callbacks: Registered callbacks for handoff events
    """

    def __init__(
        self,
        console: Any | None = None,
        max_history: int = 100
    ):
        """Initialize handoff manager.

        Args:
            console: Rich console for output
            max_history: Maximum completed handoffs to keep in history
        """
        self.console = console
        self.max_history = max_history

        # Handoff tracking
        self._active_handoffs: dict[str, HandoffContext] = {}
        self._completed_handoffs: list[HandoffContext] = []

        # Callbacks
        self._on_handoff_start: list[Callable] = []
        self._on_handoff_complete: list[Callable] = []
        self._on_handoff_cancel: list[Callable] = []

        # Statistics
        self._stats = {
            "total_handoffs": 0,
            "successful_handoffs": 0,
            "failed_handoffs": 0,
            "cancelled_handoffs": 0
        }

        logger.info("SessionHandoffManager initialized")

    async def initiate_handoff(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        context: dict[str, Any],
        call_chain: list[str] | None = None
    ) -> HandoffContext:
        """Initiate a session handoff.

        Args:
            from_agent: Agent initiating handoff
            to_agent: Target agent
            task: Task description
            context: Session context to transfer
            call_chain: Current call chain for circular detection

        Returns:
            HandoffContext for the new handoff

        Raises:
            ValueError: If circular handoff detected
        """
        # Check for circular handoff
        chain = call_chain or []
        if to_agent in chain:
            raise ValueError(
                f"Circular handoff detected: {' -> '.join(chain)} -> {to_agent}"
            )

        # Create handoff context
        handoff = HandoffContext(
            from_agent=from_agent,
            to_agent=to_agent,
            task=task,
            original_context=context.copy(),
            call_chain=chain + [from_agent]
        )

        # Track handoff
        self._active_handoffs[handoff.handoff_id] = handoff
        self._stats["total_handoffs"] += 1

        logger.info(
            f"Handoff initiated: {from_agent} -> {to_agent} "
            f"(id: {handoff.handoff_id})"
        )

        # Notify callbacks
        await self._notify_start(handoff)

        # Display to user
        if self.console:
            self.console.print(
                f"\n[bold cyan]🔄 세션 전환: {from_agent} → {to_agent}[/bold cyan]"
            )
            self.console.print(f"[dim]작업: {task}[/dim]\n")

        return handoff

    async def accept_handoff(self, handoff_id: str) -> HandoffContext | None:
        """Accept a pending handoff.

        Args:
            handoff_id: Handoff to accept

        Returns:
            HandoffContext if found and accepted, None otherwise
        """
        handoff = self._active_handoffs.get(handoff_id)
        if not handoff:
            logger.warning(f"Handoff not found: {handoff_id}")
            return None

        if handoff.status != HandoffStatus.PENDING:
            logger.warning(f"Handoff {handoff_id} is not pending")
            return None

        handoff.status = HandoffStatus.ACTIVE
        logger.info(f"Handoff accepted: {handoff_id}")

        return handoff

    async def complete_handoff(
        self,
        handoff_id: str,
        result: Any = None,
        return_to_caller: bool = False
    ) -> bool:
        """Complete a handoff.

        Args:
            handoff_id: Handoff to complete
            result: Result of the handoff
            return_to_caller: Whether to return control to original agent

        Returns:
            True if handoff was completed successfully
        """
        handoff = self._active_handoffs.get(handoff_id)
        if not handoff:
            logger.warning(f"Handoff not found: {handoff_id}")
            return False

        handoff.status = HandoffStatus.COMPLETED if not return_to_caller else HandoffStatus.RETURNED
        handoff.completed_at = datetime.now()
        handoff.result = result

        # Move to completed
        del self._active_handoffs[handoff_id]
        self._completed_handoffs.append(handoff)

        # Trim history
        if len(self._completed_handoffs) > self.max_history:
            self._completed_handoffs = self._completed_handoffs[-self.max_history:]

        self._stats["successful_handoffs"] += 1

        logger.info(f"Handoff completed: {handoff_id}")

        # Notify callbacks
        await self._notify_complete(handoff)

        # Display to user
        if self.console:
            status_text = "반환" if return_to_caller else "완료"
            self.console.print(
                f"\n[bold green]✓ 세션 전환 {status_text}: "
                f"{handoff.to_agent} → {handoff.from_agent}[/bold green]\n"
            )

        return True

    async def cancel_handoff(
        self,
        handoff_id: str,
        reason: str = "Cancelled by user"
    ) -> bool:
        """Cancel an active handoff.

        Args:
            handoff_id: Handoff to cancel
            reason: Cancellation reason

        Returns:
            True if handoff was cancelled
        """
        handoff = self._active_handoffs.get(handoff_id)
        if not handoff:
            logger.warning(f"Handoff not found: {handoff_id}")
            return False

        handoff.status = HandoffStatus.CANCELLED
        handoff.completed_at = datetime.now()
        handoff.error = reason

        # Move to completed
        del self._active_handoffs[handoff_id]
        self._completed_handoffs.append(handoff)

        self._stats["cancelled_handoffs"] += 1

        logger.info(f"Handoff cancelled: {handoff_id} - {reason}")

        # Notify callbacks
        await self._notify_cancel(handoff)

        # Display to user
        if self.console:
            self.console.print(
                f"\n[bold yellow]⚠️ 세션 전환 취소: {reason}[/bold yellow]\n"
            )

        return True

    async def fail_handoff(
        self,
        handoff_id: str,
        error: str
    ) -> bool:
        """Mark a handoff as failed.

        Args:
            handoff_id: Handoff that failed
            error: Error message

        Returns:
            True if handoff was marked as failed
        """
        handoff = self._active_handoffs.get(handoff_id)
        if not handoff:
            logger.warning(f"Handoff not found: {handoff_id}")
            return False

        handoff.status = HandoffStatus.FAILED
        handoff.completed_at = datetime.now()
        handoff.error = error

        # Move to completed
        del self._active_handoffs[handoff_id]
        self._completed_handoffs.append(handoff)

        self._stats["failed_handoffs"] += 1

        logger.error(f"Handoff failed: {handoff_id} - {error}")

        # Display to user
        if self.console:
            self.console.print(
                f"\n[bold red]❌ 세션 전환 실패: {error}[/bold red]\n"
            )

        return True

    def get_handoff(self, handoff_id: str) -> HandoffContext | None:
        """Get a handoff context by ID.

        Args:
            handoff_id: Handoff ID

        Returns:
            HandoffContext if found
        """
        # Check active first
        if handoff_id in self._active_handoffs:
            return self._active_handoffs[handoff_id]

        # Check completed
        for handoff in self._completed_handoffs:
            if handoff.handoff_id == handoff_id:
                return handoff

        return None

    def get_active_handoffs(self) -> list[HandoffContext]:
        """Get all active handoffs.

        Returns:
            List of active HandoffContext objects
        """
        return list(self._active_handoffs.values())

    def get_handoffs_for_agent(
        self,
        agent_id: str,
        include_completed: bool = False
    ) -> list[HandoffContext]:
        """Get handoffs involving a specific agent.

        Args:
            agent_id: Agent ID
            include_completed: Whether to include completed handoffs

        Returns:
            List of relevant HandoffContext objects
        """
        result = []

        # Check active
        for handoff in self._active_handoffs.values():
            if handoff.from_agent == agent_id or handoff.to_agent == agent_id:
                result.append(handoff)

        # Check completed if requested
        if include_completed:
            for handoff in self._completed_handoffs:
                if handoff.from_agent == agent_id or handoff.to_agent == agent_id:
                    result.append(handoff)

        return result

    def is_in_handoff(self, agent_id: str) -> bool:
        """Check if an agent is currently in a handoff.

        Args:
            agent_id: Agent ID to check

        Returns:
            True if agent is currently handling a handoff
        """
        for handoff in self._active_handoffs.values():
            if handoff.to_agent == agent_id and handoff.status == HandoffStatus.ACTIVE:
                return True
        return False

    def get_current_handoff_for_agent(self, agent_id: str) -> HandoffContext | None:
        """Get the current active handoff for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Active HandoffContext or None
        """
        for handoff in self._active_handoffs.values():
            if handoff.to_agent == agent_id and handoff.status == HandoffStatus.ACTIVE:
                return handoff
        return None

    # Callback registration
    def on_handoff_start(self, callback: Callable) -> None:
        """Register callback for handoff start events."""
        self._on_handoff_start.append(callback)

    def on_handoff_complete(self, callback: Callable) -> None:
        """Register callback for handoff completion events."""
        self._on_handoff_complete.append(callback)

    def on_handoff_cancel(self, callback: Callable) -> None:
        """Register callback for handoff cancellation events."""
        self._on_handoff_cancel.append(callback)

    async def _notify_start(self, handoff: HandoffContext) -> None:
        """Notify registered callbacks of handoff start."""
        for callback in self._on_handoff_start:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(handoff)
                else:
                    callback(handoff)
            except Exception as e:
                logger.error(f"Handoff start callback error: {e}")

    async def _notify_complete(self, handoff: HandoffContext) -> None:
        """Notify registered callbacks of handoff completion."""
        for callback in self._on_handoff_complete:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(handoff)
                else:
                    callback(handoff)
            except Exception as e:
                logger.error(f"Handoff complete callback error: {e}")

    async def _notify_cancel(self, handoff: HandoffContext) -> None:
        """Notify registered callbacks of handoff cancellation."""
        for callback in self._on_handoff_cancel:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(handoff)
                else:
                    callback(handoff)
            except Exception as e:
                logger.error(f"Handoff cancel callback error: {e}")

    def get_stats(self) -> dict[str, int]:
        """Get handoff statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            **self._stats,
            "active_handoffs": len(self._active_handoffs)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SessionHandoffManager(active={len(self._active_handoffs)}, "
            f"completed={len(self._completed_handoffs)})"
        )
