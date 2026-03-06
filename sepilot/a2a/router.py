"""A2A Router - Central message routing hub.

This module provides the A2ARouter class that manages message routing
between agents in the A2A protocol.
"""

import asyncio
import contextlib
import logging
from collections import defaultdict

from .connector import A2AConnector
from .exceptions import (
    AgentNotFoundError,
    MessageDeliveryError,
)
from .protocol import (
    A2AMessage,
    AgentInfo,
)

logger = logging.getLogger(__name__)


class A2ARouter:
    """Central A2A message router.

    The router manages agent registration and message routing.
    It maintains an index of agent capabilities for capability-based
    routing.

    Attributes:
        agents: Registered agent connectors
        capability_index: Capability to agent ID mapping
        message_queue: Priority queue for message processing
    """

    def __init__(self, max_queue_size: int = 10000, use_queue: bool = True):
        """Initialize A2A router.

        Args:
            max_queue_size: Maximum message queue size
            use_queue: Whether to use async queue processing (default True)
        """
        # Agent registry
        self._agents: dict[str, A2AConnector] = {}

        # Capability index: capability_name -> set of agent_ids
        self._capability_index: dict[str, set[str]] = defaultdict(set)

        # Message queue for async processing
        self._message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._use_queue = use_queue

        # Router state
        self._is_running = False
        self._process_task: asyncio.Task | None = None

        # Statistics
        self._stats = {
            "messages_routed": 0,
            "messages_failed": 0,
            "messages_queued": 0,
            "agents_registered": 0
        }

        logger.info("A2A Router initialized")

    def register_agent(self, connector: A2AConnector) -> bool:
        """Register an agent with the router.

        Args:
            connector: Agent's A2A connector

        Returns:
            True if registration successful
        """
        agent_id = connector.agent_id

        if agent_id in self._agents:
            logger.warning(f"Agent {agent_id} already registered, updating")

        # Register agent
        self._agents[agent_id] = connector
        connector.set_router(self)

        # Index capabilities
        for capability in connector.get_capabilities():
            self._capability_index[capability.name].add(agent_id)

        self._stats["agents_registered"] = len(self._agents)
        logger.info(f"Agent registered: {agent_id} with {len(connector.get_capabilities())} capabilities")

        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the router.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if unregistration successful
        """
        if agent_id not in self._agents:
            logger.warning(f"Agent {agent_id} not found in registry")
            return False

        connector = self._agents[agent_id]

        # Remove from capability index
        for capability in connector.get_capabilities():
            self._capability_index[capability.name].discard(agent_id)
            # Clean up empty sets
            if not self._capability_index[capability.name]:
                del self._capability_index[capability.name]

        # Remove from registry
        del self._agents[agent_id]
        connector.stop()

        self._stats["agents_registered"] = len(self._agents)
        logger.info(f"Agent unregistered: {agent_id}")

        return True

    def get_agent(self, agent_id: str) -> A2AConnector | None:
        """Get an agent connector by ID.

        Args:
            agent_id: Agent ID

        Returns:
            A2AConnector or None if not found
        """
        return self._agents.get(agent_id)

    def get_agent_info(self, agent_id: str) -> AgentInfo | None:
        """Get agent information.

        Args:
            agent_id: Agent ID

        Returns:
            AgentInfo or None if not found
        """
        connector = self._agents.get(agent_id)
        if connector:
            return connector.get_agent_info()
        return None

    def list_agents(self) -> list[AgentInfo]:
        """List all registered agents.

        Returns:
            List of AgentInfo objects
        """
        return [connector.get_agent_info() for connector in self._agents.values()]

    def list_agent_ids(self) -> list[str]:
        """List all registered agent IDs.

        Returns:
            List of agent IDs
        """
        return list(self._agents.keys())

    def find_agents_by_capability(self, capability: str) -> list[str]:
        """Find agents that have a specific capability.

        Args:
            capability: Capability name to search for

        Returns:
            List of agent IDs with the capability
        """
        return list(self._capability_index.get(capability, set()))

    def find_best_agent_for_capability(self, capability: str) -> str | None:
        """Find the best agent for a capability based on priority.

        Args:
            capability: Capability name

        Returns:
            Agent ID of best match, or None if not found
        """
        agent_ids = self.find_agents_by_capability(capability)

        if not agent_ids:
            return None

        # Find agent with highest capability priority
        best_agent = None
        best_priority = -1

        for agent_id in agent_ids:
            connector = self._agents.get(agent_id)
            if connector:
                for cap in connector.get_capabilities():
                    if cap.name == capability and cap.priority > best_priority:
                        best_priority = cap.priority
                        best_agent = agent_id

        return best_agent

    def list_capabilities(self) -> dict[str, list[str]]:
        """List all capabilities and their providing agents.

        Returns:
            Dict mapping capability names to agent IDs
        """
        return {
            cap: list(agents)
            for cap, agents in self._capability_index.items()
        }

    async def route_message(self, message: A2AMessage) -> bool:
        """Route a message to its target agent.

        Args:
            message: Message to route

        Returns:
            True if routing successful

        Raises:
            AgentNotFoundError: If target agent not found
            MessageDeliveryError: If delivery fails
        """
        # Use queue for async processing if enabled and router is running
        if self._use_queue and self._is_running:
            return await self._queue_message(message)
        else:
            return await self._deliver_message_direct(message)

    async def _queue_message(self, message: A2AMessage) -> bool:
        """Queue a message for async processing.

        Args:
            message: Message to queue

        Returns:
            True if queued successfully
        """
        try:
            # Priority queue uses (priority, counter, item) to break ties
            import time
            priority_item = (message.priority, time.time(), message)
            await asyncio.wait_for(
                self._message_queue.put(priority_item),
                timeout=5.0
            )
            self._stats["messages_queued"] += 1
            logger.debug(f"Message queued: {message.from_agent} -> {message.to_agent}")
            return True
        except asyncio.TimeoutError:
            logger.error("Message queue full, falling back to direct delivery")
            return await self._deliver_message_direct(message)

    async def _deliver_message_direct(self, message: A2AMessage) -> bool:
        """Deliver a message directly to target agent.

        Args:
            message: Message to deliver

        Returns:
            True if delivery successful
        """
        target_agent = message.to_agent

        # Check if target agent exists
        connector = self._agents.get(target_agent)
        if not connector:
            # Try to find by capability if target looks like a capability name
            connector = self._try_capability_routing(message)
            if not connector:
                self._stats["messages_failed"] += 1
                raise AgentNotFoundError(target_agent)

        # Deliver message
        try:
            await connector.deliver_message(message)
            self._stats["messages_routed"] += 1
            logger.debug(
                f"Message routed: {message.from_agent} -> {target_agent} "
                f"({message.message_type.value})"
            )
            return True

        except Exception as e:
            self._stats["messages_failed"] += 1
            logger.error(f"Message delivery failed: {e}")
            raise MessageDeliveryError(
                message_id=message.message_id,
                target_agent=target_agent,
                reason=str(e)
            ) from e

    def _try_capability_routing(self, message: A2AMessage) -> A2AConnector | None:
        """Try to route message based on capability.

        If target_agent is not found, treat it as a capability name
        and find the best agent for that capability.

        Args:
            message: Message to route

        Returns:
            A2AConnector if found, else None
        """
        capability = message.to_agent
        best_agent = self.find_best_agent_for_capability(capability)

        if best_agent:
            logger.info(
                f"Capability routing: {capability} -> {best_agent}"
            )
            return self._agents.get(best_agent)

        return None

    async def broadcast_message(
        self,
        message: A2AMessage,
        exclude: list[str] | None = None
    ) -> int:
        """Broadcast a message to all agents.

        Args:
            message: Message to broadcast
            exclude: List of agent IDs to exclude

        Returns:
            Number of agents message was sent to
        """
        exclude = exclude or []
        sent_count = 0

        for agent_id, connector in self._agents.items():
            if agent_id not in exclude and agent_id != message.from_agent:
                try:
                    broadcast_msg = A2AMessage(
                        from_agent=message.from_agent,
                        to_agent=agent_id,
                        message_type=message.message_type,
                        payload=message.payload,
                        priority=message.priority,
                        metadata={"broadcast": True}
                    )
                    await connector.deliver_message(broadcast_msg)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Broadcast to {agent_id} failed: {e}")

        logger.debug(f"Broadcast sent to {sent_count} agents")
        return sent_count

    async def query_capabilities(
        self,
        capability: str,
        from_agent: str
    ) -> list[AgentInfo]:
        """Query agents that can handle a capability.

        Args:
            capability: Capability to query
            from_agent: Requesting agent ID

        Returns:
            List of AgentInfo for agents with the capability
        """
        agent_ids = self.find_agents_by_capability(capability)

        result = []
        for agent_id in agent_ids:
            if agent_id != from_agent:  # Exclude requester
                info = self.get_agent_info(agent_id)
                if info:
                    result.append(info)

        return result

    async def start(self) -> None:
        """Start the router's message processing loop."""
        if self._is_running:
            logger.warning("Router already running")
            return

        self._is_running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("A2A Router started")

    async def stop(self) -> None:
        """Stop the router."""
        self._is_running = False

        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task

        # Stop all agent connectors
        for connector in self._agents.values():
            connector.stop()

        logger.info("A2A Router stopped")

    async def _process_loop(self) -> None:
        """Main message processing loop."""
        while self._is_running:
            try:
                # Process messages from queue
                if not self._message_queue.empty():
                    try:
                        priority_item = await asyncio.wait_for(
                            self._message_queue.get(),
                            timeout=0.1
                        )
                        _, _, message = priority_item
                        # Process message in background
                        asyncio.create_task(self._process_queued_message(message))
                    except asyncio.TimeoutError:
                        pass
                else:
                    # Yield to other tasks when queue is empty
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Router process loop error: {e}")

    async def _process_queued_message(self, message: A2AMessage) -> None:
        """Process a message from the queue.

        Args:
            message: Message to process
        """
        try:
            await self._deliver_message_direct(message)
        except Exception as e:
            logger.error(f"Failed to process queued message: {e}")
            # Try to send error response
            try:
                if message.from_agent in self._agents:
                    error_response = message.create_error_response(str(e))
                    await self._deliver_message_direct(error_response)
            except Exception:
                pass

    def get_stats(self) -> dict[str, int]:
        """Get router statistics.

        Returns:
            Dictionary with statistics
        """
        return self._stats.copy()

    def is_running(self) -> bool:
        """Check if router is running.

        Returns:
            True if router is running
        """
        return self._is_running

    def has_agent(self, agent_id: str) -> bool:
        """Check if an agent is registered.

        Args:
            agent_id: Agent ID to check

        Returns:
            True if agent is registered
        """
        return agent_id in self._agents

    def __repr__(self) -> str:
        """String representation of router."""
        return (
            f"A2ARouter(agents={len(self._agents)}, "
            f"capabilities={len(self._capability_index)}, "
            f"running={self._is_running})"
        )
