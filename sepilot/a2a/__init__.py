"""A2A (Agent-to-Agent) Protocol for SEPilot.

This module provides inter-agent communication capabilities for SEPilot,
allowing agents to collaborate by sending messages and delegating tasks.

Main components:
- A2AMessage: Message protocol for agent communication
- A2AConnector: Per-agent communication interface
- A2ARouter: Central message routing hub
- SessionHandoffManager: Session handoff management

Example usage:
    from sepilot.a2a import A2ARouter, A2AConnector, A2AMessage

    # Create router
    router = A2ARouter()

    # Create connector for an agent
    connector = A2AConnector(
        agent_id="my_agent",
        capabilities=[AgentCapability(name="task_type")]
    )

    # Register with router
    router.register_agent(connector)

    # Send message to another agent
    await connector.send_request(
        to_agent="other_agent",
        task="do something",
        direct_response=True
    )
"""

from .connector import A2AConnector
from .exceptions import (
    A2AConfigurationError,
    A2AError,
    AgentNotFoundError,
    CapabilityNotFoundError,
    CircularCallError,
    HandoffError,
    HandoffExecutionError,
    HandoffRejectedError,
    HandoffTimeoutError,
    MessageDeliveryError,
    MessageTimeoutError,
    UserRejectedHandoffError,
)
from .handoff import (
    HandoffContext,
    HandoffStatus,
    SessionHandoffManager,
)
from .protocol import (
    A2ACallResult,
    A2AMessage,
    A2AMessageType,
    A2AResultStatus,
    AgentCapability,
    AgentInfo,
    HandoffApprovalRequest,
    MessagePriority,
    ResponseMode,
)
from .router import A2ARouter

__all__ = [
    # Protocol
    "A2AMessage",
    "A2AMessageType",
    "ResponseMode",
    "MessagePriority",
    "AgentCapability",
    "AgentInfo",
    "A2AResultStatus",
    "A2ACallResult",
    "HandoffApprovalRequest",
    # Connector
    "A2AConnector",
    # Router
    "A2ARouter",
    # Handoff
    "HandoffContext",
    "HandoffStatus",
    "SessionHandoffManager",
    # Exceptions
    "A2AError",
    "AgentNotFoundError",
    "CapabilityNotFoundError",
    "MessageTimeoutError",
    "MessageDeliveryError",
    "HandoffError",
    "HandoffRejectedError",
    "HandoffTimeoutError",
    "CircularCallError",
    "A2AConfigurationError",
    "HandoffExecutionError",
    "UserRejectedHandoffError",
]
