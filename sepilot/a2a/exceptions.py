"""A2A Exception Classes.

This module defines exceptions specific to the A2A protocol
for proper error handling in agent-to-agent communication.
"""

from typing import Any


class A2AError(Exception):
    """Base exception for all A2A errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "A2A_ERROR"
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for message payload."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class AgentNotFoundError(A2AError):
    """Raised when a target agent is not found in the registry."""

    def __init__(self, agent_id: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=f"Agent '{agent_id}' not found in registry",
            error_code="AGENT_NOT_FOUND",
            details={"agent_id": agent_id, **(details or {})}
        )
        self.agent_id = agent_id


class AgentNotAvailableError(A2AError):
    """Raised when a target agent exists but is not available."""

    def __init__(self, agent_id: str, reason: str = "Agent is not available"):
        super().__init__(
            message=f"Agent '{agent_id}' is not available: {reason}",
            error_code="AGENT_NOT_AVAILABLE",
            details={"agent_id": agent_id, "reason": reason}
        )
        self.agent_id = agent_id
        self.reason = reason


class MessageTimeoutError(A2AError):
    """Raised when message delivery or response times out."""

    def __init__(
        self,
        message_id: str,
        timeout: float,
        target_agent: str | None = None
    ):
        msg = f"Message '{message_id}' timed out after {timeout}s"
        if target_agent:
            msg += f" waiting for agent '{target_agent}'"

        super().__init__(
            message=msg,
            error_code="MESSAGE_TIMEOUT",
            details={
                "message_id": message_id,
                "timeout": timeout,
                "target_agent": target_agent
            }
        )
        self.message_id = message_id
        self.timeout = timeout
        self.target_agent = target_agent


class MessageDeliveryError(A2AError):
    """Raised when a message cannot be delivered."""

    def __init__(
        self,
        message_id: str,
        target_agent: str,
        reason: str = "Delivery failed"
    ):
        super().__init__(
            message=f"Failed to deliver message '{message_id}' to '{target_agent}': {reason}",
            error_code="DELIVERY_FAILED",
            details={
                "message_id": message_id,
                "target_agent": target_agent,
                "reason": reason
            }
        )
        self.message_id = message_id
        self.target_agent = target_agent
        self.reason = reason


class HandoffError(A2AError):
    """Base exception for session handoff errors."""

    def __init__(
        self,
        message: str,
        from_agent: str,
        to_agent: str,
        details: dict[str, Any] | None = None
    ):
        super().__init__(
            message=message,
            error_code="HANDOFF_ERROR",
            details={
                "from_agent": from_agent,
                "to_agent": to_agent,
                **(details or {})
            }
        )
        self.from_agent = from_agent
        self.to_agent = to_agent


class HandoffRejectedError(HandoffError):
    """Raised when a session handoff is rejected by the target agent."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        reason: str = "Handoff rejected"
    ):
        super().__init__(
            message=f"Handoff from '{from_agent}' to '{to_agent}' was rejected: {reason}",
            from_agent=from_agent,
            to_agent=to_agent,
            details={"reason": reason}
        )
        self.error_code = "HANDOFF_REJECTED"
        self.reason = reason


class HandoffTimeoutError(HandoffError):
    """Raised when a session handoff times out."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        timeout: float
    ):
        super().__init__(
            message=f"Handoff from '{from_agent}' to '{to_agent}' timed out after {timeout}s",
            from_agent=from_agent,
            to_agent=to_agent,
            details={"timeout": timeout}
        )
        self.error_code = "HANDOFF_TIMEOUT"
        self.timeout = timeout


class CapabilityNotFoundError(A2AError):
    """Raised when no agent with required capability is found."""

    def __init__(self, capability: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=f"No agent found with capability '{capability}'",
            error_code="CAPABILITY_NOT_FOUND",
            details={"capability": capability, **(details or {})}
        )
        self.capability = capability


class CircularCallError(A2AError):
    """Raised when circular agent calls are detected."""

    def __init__(self, call_chain: list):
        chain_str = " -> ".join(call_chain)
        super().__init__(
            message=f"Circular agent call detected: {chain_str}",
            error_code="CIRCULAR_CALL",
            details={"call_chain": call_chain}
        )
        self.call_chain = call_chain


class RouterNotInitializedError(A2AError):
    """Raised when A2A operations are attempted before router initialization."""

    def __init__(self, agent_id: str | None = None):
        msg = "A2A router not initialized"
        if agent_id:
            msg += f" for agent '{agent_id}'"

        super().__init__(
            message=msg,
            error_code="ROUTER_NOT_INITIALIZED",
            details={"agent_id": agent_id}
        )


class InvalidMessageError(A2AError):
    """Raised when a message is invalid or malformed."""

    def __init__(self, reason: str, message_data: dict[str, Any] | None = None):
        super().__init__(
            message=f"Invalid A2A message: {reason}",
            error_code="INVALID_MESSAGE",
            details={"reason": reason, "message_data": message_data}
        )
        self.reason = reason


class AgentBusyError(A2AError):
    """Raised when an agent is busy and cannot accept new requests."""

    def __init__(self, agent_id: str, current_task: str | None = None):
        msg = f"Agent '{agent_id}' is busy"
        if current_task:
            msg += f" (currently: {current_task})"

        super().__init__(
            message=msg,
            error_code="AGENT_BUSY",
            details={"agent_id": agent_id, "current_task": current_task}
        )
        self.agent_id = agent_id
        self.current_task = current_task


class HandoffExecutionError(HandoffError):
    """Raised when task execution fails during handoff."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        error: str
    ):
        super().__init__(
            message=f"Task execution failed during handoff to '{to_agent}': {error}",
            from_agent=from_agent,
            to_agent=to_agent,
            details={"task": task, "error": error}
        )
        self.error_code = "HANDOFF_EXECUTION_ERROR"
        self.task = task
        self.error = error


class UserRejectedHandoffError(HandoffError):
    """Raised when user rejects a session handoff."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        task: str
    ):
        super().__init__(
            message=f"User rejected handoff from '{from_agent}' to '{to_agent}'",
            from_agent=from_agent,
            to_agent=to_agent,
            details={"task": task}
        )
        self.error_code = "USER_REJECTED_HANDOFF"
        self.task = task


class A2AConfigurationError(A2AError):
    """Raised when A2A configuration is invalid or incomplete."""

    def __init__(self, reason: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=f"A2A configuration error: {reason}",
            error_code="A2A_CONFIGURATION_ERROR",
            details={"reason": reason, **(details or {})}
        )
        self.reason = reason
