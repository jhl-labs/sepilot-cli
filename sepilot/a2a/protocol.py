"""A2A (Agent-to-Agent) Protocol Definitions.

This module defines the core message protocol for agent-to-agent communication
in SEPilot. It enables agents to collaborate by sending messages through a
central router.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class A2AMessageType(Enum):
    """Types of A2A messages."""

    REQUEST = "request"           # Task request
    RESPONSE = "response"         # Task response
    HANDOFF = "handoff"           # Session handoff (for direct response)
    HANDOFF_ACCEPT = "handoff_accept"  # Handoff accepted
    HANDOFF_REJECT = "handoff_reject"  # Handoff rejected
    HANDOFF_COMPLETE = "handoff_complete"  # Handoff completed
    ACK = "ack"                   # Acknowledgment
    ERROR = "error"               # Error message
    HEARTBEAT = "heartbeat"       # Health check
    CAPABILITY_QUERY = "capability_query"  # Query agent capabilities
    CAPABILITY_RESPONSE = "capability_response"  # Capabilities response


class ResponseMode(Enum):
    """Response modes for A2A communication."""

    DIRECT = "direct"       # Called agent responds directly to user
    RETURN = "return"       # Result returned to calling agent


class MessagePriority(int, Enum):
    """Message priority levels (lower = higher priority)."""

    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 10


@dataclass
class A2AMessage:
    """Agent-to-Agent communication message.

    Attributes:
        message_id: Unique message identifier
        from_agent: Sender agent ID
        to_agent: Recipient agent ID
        message_type: Type of message
        payload: Message payload data
        timestamp: Message creation timestamp
        priority: Message priority (1=highest, 10=lowest)
        correlation_id: ID linking request-response pairs
        session_context: Session context for handoff
        response_mode: How the response should be delivered
        timeout: Response timeout in seconds
        metadata: Additional metadata
    """

    from_agent: str
    to_agent: str
    message_type: A2AMessageType
    payload: dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = MessagePriority.NORMAL
    correlation_id: str | None = None
    session_context: dict[str, Any] | None = None
    response_mode: ResponseMode = ResponseMode.DIRECT
    timeout: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def create_response(
        self,
        payload: dict[str, Any],
        message_type: A2AMessageType = A2AMessageType.RESPONSE
    ) -> "A2AMessage":
        """Create a response message to this message.

        Args:
            payload: Response payload
            message_type: Response message type

        Returns:
            Response A2AMessage
        """
        return A2AMessage(
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            message_type=message_type,
            payload=payload,
            correlation_id=self.message_id,
            response_mode=self.response_mode
        )

    def create_error_response(self, error: str, error_code: str | None = None) -> "A2AMessage":
        """Create an error response message.

        Args:
            error: Error message
            error_code: Optional error code

        Returns:
            Error A2AMessage
        """
        return A2AMessage(
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            message_type=A2AMessageType.ERROR,
            payload={
                "error": error,
                "error_code": error_code,
                "original_message_type": self.message_type.value
            },
            correlation_id=self.message_id,
            priority=MessagePriority.HIGH
        )

    def create_ack(self) -> "A2AMessage":
        """Create an acknowledgment message.

        Returns:
            Acknowledgment A2AMessage
        """
        return A2AMessage(
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            message_type=A2AMessageType.ACK,
            payload={"acknowledged": self.message_id},
            correlation_id=self.message_id,
            priority=MessagePriority.HIGH
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary.

        Returns:
            Dictionary representation of the message
        """
        return {
            "message_id": self.message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "correlation_id": self.correlation_id,
            "session_context": self.session_context,
            "response_mode": self.response_mode.value,
            "timeout": self.timeout,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "A2AMessage":
        """Create message from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            A2AMessage instance
        """
        return cls(
            message_id=data.get("message_id", str(uuid4())),
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            message_type=A2AMessageType(data["message_type"]),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            priority=data.get("priority", MessagePriority.NORMAL),
            correlation_id=data.get("correlation_id"),
            session_context=data.get("session_context"),
            response_mode=ResponseMode(data.get("response_mode", "direct")),
            timeout=data.get("timeout", 300.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentCapability:
    """Describes a capability that an agent provides.

    Attributes:
        name: Capability name (e.g., "code_review", "git_operations")
        description: Human-readable description
        input_schema: JSON schema for expected input
        output_schema: JSON schema for expected output
        priority: Agent's confidence in handling this capability (0-100)
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    priority: int = 50

    def to_dict(self) -> dict[str, Any]:
        """Convert capability to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCapability":
        """Create capability from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            priority=data.get("priority", 50)
        )


@dataclass
class AgentInfo:
    """Information about a registered agent.

    Attributes:
        agent_id: Unique agent identifier
        display_name: Human-readable name
        agent_type: Type of agent (e.g., "plugin", "base", "subagent")
        capabilities: List of agent capabilities
        status: Current agent status
        metadata: Additional agent metadata
    """

    agent_id: str
    display_name: str = ""
    agent_type: str = "agent"
    capabilities: list[AgentCapability] = field(default_factory=list)
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability.

        Args:
            capability_name: Name of capability to check

        Returns:
            True if agent has the capability
        """
        return any(cap.name == capability_name for cap in self.capabilities)

    def get_capability(self, capability_name: str) -> AgentCapability | None:
        """Get a specific capability.

        Args:
            capability_name: Name of capability

        Returns:
            AgentCapability or None if not found
        """
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert agent info to dictionary."""
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "agent_type": self.agent_type,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "status": self.status,
            "metadata": self.metadata
        }


class A2AResultStatus(Enum):
    """Status of an A2A call result."""

    SUCCESS = "success"           # Call completed successfully
    TIMEOUT = "timeout"           # Call timed out
    REJECTED = "rejected"         # Target agent rejected the request
    ERROR = "error"               # Error during execution
    CANCELLED = "cancelled"       # Call was cancelled
    USER_REJECTED = "user_rejected"  # User rejected the handoff


@dataclass
class A2ACallResult:
    """Result of an A2A call with comprehensive error information.

    This class provides a structured way to handle A2A call results,
    including success data, error information, and fallback suggestions.

    Attributes:
        status: Result status
        result: Actual result data (if successful)
        error: Error message (if failed)
        error_code: Specific error code
        target_agent: Agent that was called
        task: Original task description
        duration_ms: Call duration in milliseconds
        can_retry: Whether the call can be retried
        fallback_suggestions: Suggested fallback actions
        original_context: Context that was passed
    """

    status: A2AResultStatus
    result: Any | None = None
    error: str | None = None
    error_code: str | None = None
    target_agent: str = ""
    task: str = ""
    duration_ms: float = 0
    can_retry: bool = False
    fallback_suggestions: list[str] = field(default_factory=list)
    original_context: dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        """Check if the call was successful."""
        return self.status == A2AResultStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        """Check if the call resulted in an error."""
        return self.status in (
            A2AResultStatus.ERROR,
            A2AResultStatus.TIMEOUT,
            A2AResultStatus.REJECTED
        )

    @property
    def is_user_intervention_needed(self) -> bool:
        """Check if user intervention is needed."""
        return self.status == A2AResultStatus.USER_REJECTED

    def get_user_message(self) -> str:
        """Get a user-friendly message about the result."""
        if self.is_success:
            return f"✅ {self.target_agent}가 작업을 완료했습니다."

        messages = {
            A2AResultStatus.TIMEOUT: f"⏱️ {self.target_agent} 응답 시간 초과",
            A2AResultStatus.REJECTED: f"❌ {self.target_agent}가 요청을 거부했습니다: {self.error}",
            A2AResultStatus.ERROR: f"❌ {self.target_agent} 실행 오류: {self.error}",
            A2AResultStatus.CANCELLED: f"🚫 {self.target_agent} 호출이 취소되었습니다",
            A2AResultStatus.USER_REJECTED: f"🙅 사용자가 {self.target_agent}로의 세션 전환을 거부했습니다"
        }
        return messages.get(self.status, f"알 수 없는 상태: {self.status}")

    def get_fallback_message(self) -> str | None:
        """Get fallback suggestion message for the caller."""
        if self.is_success:
            return None

        if self.fallback_suggestions:
            suggestions = "\n".join(f"  - {s}" for s in self.fallback_suggestions)
            return f"대안 조치:\n{suggestions}"

        # Default fallback suggestions based on error type
        default_suggestions = {
            A2AResultStatus.TIMEOUT: "재시도하거나 더 간단한 작업으로 분리해보세요.",
            A2AResultStatus.REJECTED: "다른 에이전트를 시도하거나 직접 처리하세요.",
            A2AResultStatus.ERROR: "오류 내용을 확인하고 수정된 요청을 시도하세요.",
            A2AResultStatus.USER_REJECTED: "사용자가 원하는 방식으로 작업을 진행하세요."
        }
        return default_suggestions.get(self.status)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "error_code": self.error_code,
            "target_agent": self.target_agent,
            "task": self.task,
            "duration_ms": self.duration_ms,
            "can_retry": self.can_retry,
            "fallback_suggestions": self.fallback_suggestions
        }


@dataclass
class HandoffApprovalRequest:
    """Request for user approval before session handoff.

    Attributes:
        from_agent: Agent initiating the handoff
        to_agent: Target agent
        to_agent_display_name: Human-readable name of target agent
        task: Task to be handed off
        reason: Reason for handoff
        capabilities: Capabilities of target agent relevant to task
        estimated_duration: Estimated task duration
    """

    from_agent: str
    to_agent: str
    to_agent_display_name: str
    task: str
    reason: str = ""
    capabilities: list[str] = field(default_factory=list)
    estimated_duration: str | None = None

    def get_approval_message(self) -> str:
        """Generate user-facing approval message."""
        caps = ", ".join(self.capabilities[:3]) if self.capabilities else "일반"
        msg = (
            f"\n🔄 세션 전환 요청\n"
            f"  대상: {self.to_agent_display_name} ({caps})\n"
            f"  작업: {self.task}\n"
        )
        if self.reason:
            msg += f"  이유: {self.reason}\n"
        msg += "\n세션을 전환하시겠습니까? [Y/n]: "
        return msg
