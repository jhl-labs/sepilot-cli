"""A2A Connector - Agent communication interface.

This module provides the A2AConnector class that agents use to communicate
with other agents through the A2A protocol.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from .exceptions import (
    MessageTimeoutError,
    RouterNotInitializedError,
)
from .protocol import (
    A2AMessage,
    A2AMessageType,
    AgentCapability,
    AgentInfo,
    MessagePriority,
    ResponseMode,
)

if TYPE_CHECKING:
    from .router import A2ARouter

logger = logging.getLogger(__name__)


class A2AConnector:
    """Agent's interface for A2A communication.

    Each agent has its own A2AConnector that provides methods for
    sending/receiving messages and managing session handoffs.

    Attributes:
        agent_id: Unique identifier for this agent
        display_name: Human-readable name
        router: Reference to the central A2A router
        inbox: Queue for incoming messages
        capabilities: List of agent capabilities
        message_handlers: Registered message handlers
    """

    def __init__(
        self,
        agent_id: str,
        display_name: str = "",
        router: Optional["A2ARouter"] = None,
        agent_type: str = "agent"
    ):
        """Initialize A2A connector.

        Args:
            agent_id: Unique agent identifier
            display_name: Human-readable name
            router: Central A2A router (can be set later)
            agent_type: Type of agent
        """
        self.agent_id = agent_id
        self.display_name = display_name or agent_id
        self.router = router
        self.agent_type = agent_type

        # Message handling
        self.inbox: asyncio.Queue[A2AMessage] = asyncio.Queue()
        self._pending_responses: dict[str, asyncio.Future] = {}
        self._message_handlers: dict[A2AMessageType, Callable] = {}

        # Capabilities
        self._capabilities: list[AgentCapability] = []

        # State
        self._is_active = False
        self._is_busy = False
        self._current_task: str | None = None
        self._call_stack: list[str] = []  # For circular call detection
        self._call_stack_lock = asyncio.Lock()  # Protect call_stack mutations
        self._process_task: asyncio.Task | None = None

        # Concurrency control
        self._task_lock = asyncio.Lock()
        self._max_concurrent_tasks = 1  # Default: one task at a time

        logger.debug(f"A2AConnector created for agent: {agent_id}")

    def set_router(self, router: "A2ARouter") -> None:
        """Set the A2A router reference.

        Args:
            router: Central A2A router
        """
        self.router = router
        logger.debug(f"Router set for agent: {self.agent_id}")

    def register_capability(
        self,
        name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        priority: int = 50
    ) -> None:
        """Register a capability for this agent.

        Args:
            name: Capability name
            description: Capability description
            input_schema: JSON schema for expected input
            output_schema: JSON schema for expected output
            priority: Agent's confidence in handling (0-100)
        """
        capability = AgentCapability(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            priority=priority
        )
        self._capabilities.append(capability)
        logger.debug(f"Agent {self.agent_id} registered capability: {name}")

    def get_capabilities(self) -> list[AgentCapability]:
        """Get list of agent capabilities.

        Returns:
            List of AgentCapability objects
        """
        return self._capabilities.copy()

    def get_agent_info(self) -> AgentInfo:
        """Get agent information.

        Returns:
            AgentInfo object describing this agent
        """
        return AgentInfo(
            agent_id=self.agent_id,
            display_name=self.display_name,
            agent_type=self.agent_type,
            capabilities=self._capabilities.copy(),
            status="active" if self._is_active else "inactive",
            metadata={"current_task": self._current_task}
        )

    def register_handler(
        self,
        message_type: A2AMessageType,
        handler: Callable[[A2AMessage], A2AMessage | None]
    ) -> None:
        """Register a message handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self._message_handlers[message_type] = handler
        logger.debug(f"Agent {self.agent_id} registered handler for: {message_type.value}")

    async def send_message(
        self,
        to_agent: str,
        message_type: A2AMessageType,
        payload: dict[str, Any],
        priority: int = MessagePriority.NORMAL,
        response_mode: ResponseMode = ResponseMode.DIRECT,
        timeout: float = 300.0,
        wait_for_response: bool = False,
        session_context: dict[str, Any] | None = None
    ) -> A2AMessage | None:
        """Send a message to another agent.

        Args:
            to_agent: Target agent ID
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            response_mode: How response should be delivered
            timeout: Response timeout in seconds
            wait_for_response: Whether to wait for response
            session_context: Optional session context for handoff

        Returns:
            Response message if wait_for_response is True, else None

        Raises:
            RouterNotInitializedError: If router is not set
            AgentNotFoundError: If target agent not found
            MessageTimeoutError: If response times out
        """
        if not self.router:
            raise RouterNotInitializedError(self.agent_id)

        message = A2AMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
            priority=priority,
            response_mode=response_mode,
            timeout=timeout,
            session_context=session_context
        )

        logger.debug(f"Agent {self.agent_id} sending {message_type.value} to {to_agent}")

        if wait_for_response:
            # Create future for response
            response_future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._pending_responses[message.message_id] = response_future

            try:
                # Route message
                await self.router.route_message(message)

                # Wait for response with timeout
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response

            except asyncio.TimeoutError as e:
                raise MessageTimeoutError(
                    message_id=message.message_id,
                    timeout=timeout,
                    target_agent=to_agent
                ) from e
            finally:
                self._pending_responses.pop(message.message_id, None)
        else:
            # Fire and forget
            await self.router.route_message(message)
            return None

    async def send_request(
        self,
        to_agent: str,
        task: str,
        context: dict[str, Any] | None = None,
        timeout: float = 300.0,
        direct_response: bool = True
    ) -> A2AMessage | None:
        """Send a task request to another agent.

        Convenience method for sending REQUEST messages.

        Args:
            to_agent: Target agent ID
            task: Task description
            context: Task context
            timeout: Response timeout
            direct_response: If True, target agent responds to user directly

        Returns:
            Response message if direct_response is False
        """
        payload = {
            "task": task,
            "context": context or {}
        }

        response_mode = ResponseMode.DIRECT if direct_response else ResponseMode.RETURN

        return await self.send_message(
            to_agent=to_agent,
            message_type=A2AMessageType.REQUEST,
            payload=payload,
            response_mode=response_mode,
            timeout=timeout,
            wait_for_response=not direct_response
        )

    async def handoff_session(
        self,
        to_agent: str,
        task: str,
        context: dict[str, Any],
        timeout: float = 30.0
    ) -> bool:
        """Request session handoff to another agent.

        The target agent will take over and respond directly to the user.

        Args:
            to_agent: Target agent ID
            task: Task to be handled
            context: Session context (console, settings, etc.)
            timeout: Timeout for handoff acceptance

        Returns:
            True if handoff was accepted, False otherwise
        """
        if not self.router:
            raise RouterNotInitializedError(self.agent_id)

        # Track call for circular detection (atomically)
        async with self._call_stack_lock:
            if to_agent in self._call_stack:
                logger.warning(f"Circular call detected: {' -> '.join(self._call_stack)} -> {to_agent}")
                return False
            self._call_stack.append(to_agent)

        try:
            message = A2AMessage(
                from_agent=self.agent_id,
                to_agent=to_agent,
                message_type=A2AMessageType.HANDOFF,
                payload={
                    "task": task,
                    "handoff_id": str(uuid4())
                },
                session_context=context,
                response_mode=ResponseMode.DIRECT,
                timeout=timeout
            )

            # Wait for handoff acceptance
            response_future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._pending_responses[message.message_id] = response_future

            try:
                await self.router.route_message(message)
                response = await asyncio.wait_for(response_future, timeout=timeout)

                if response.message_type == A2AMessageType.HANDOFF_ACCEPT:
                    logger.info(f"Handoff accepted by {to_agent}")
                    return True
                else:
                    logger.warning(f"Handoff rejected by {to_agent}: {response.payload}")
                    return False

            except asyncio.TimeoutError:
                logger.warning(f"Handoff to {to_agent} timed out")
                return False
            finally:
                self._pending_responses.pop(message.message_id, None)

        finally:
            async with self._call_stack_lock:
                if to_agent in self._call_stack:
                    self._call_stack.remove(to_agent)

    async def receive_message(self, timeout: float | None = None) -> A2AMessage | None:
        """Receive a message from the inbox.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Received message or None if timeout
        """
        try:
            if timeout:
                return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
            else:
                return await self.inbox.get()
        except asyncio.TimeoutError:
            return None

    async def deliver_message(self, message: A2AMessage) -> None:
        """Deliver a message to this agent.

        Called by the router to deliver messages.

        Args:
            message: Message to deliver
        """
        # Check if this is a response to a pending request
        if message.correlation_id and message.correlation_id in self._pending_responses:
            future = self._pending_responses[message.correlation_id]
            if not future.done():
                future.set_result(message)
            return

        # Otherwise, put in inbox for processing
        await self.inbox.put(message)
        logger.debug(f"Message delivered to {self.agent_id}: {message.message_type.value}")

    def start(self) -> None:
        """Start the message processing loop as a background task."""
        if self._is_active:
            logger.warning(f"Connector {self.agent_id} already running")
            return

        self._is_active = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"A2A Connector started for agent: {self.agent_id}")

    async def _process_loop(self) -> None:
        """Internal message processing loop."""
        while self._is_active:
            try:
                message = await self.receive_message(timeout=1.0)
                if message:
                    # Handle message in background to not block the loop
                    asyncio.create_task(self._handle_message_safe(message))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop for {self.agent_id}: {e}")

    async def _handle_message_safe(self, message: A2AMessage) -> None:
        """Handle message with error catching."""
        try:
            await self._handle_message(message)
        except Exception as e:
            logger.error(f"Error handling message in {self.agent_id}: {e}")
            # Send error response if possible
            if self.router and message.message_type in (
                A2AMessageType.REQUEST,
                A2AMessageType.HANDOFF
            ):
                error_response = message.create_error_response(str(e))
                with contextlib.suppress(Exception):
                    await self.router.route_message(error_response)

    async def process_messages(self) -> None:
        """Process messages from inbox using registered handlers.

        This is the main message processing loop.
        Prefer using start() for background processing.
        """
        self._is_active = True

        while self._is_active:
            try:
                message = await self.receive_message(timeout=1.0)
                if message:
                    await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message in {self.agent_id}: {e}")

    async def _handle_message(self, message: A2AMessage) -> None:
        """Handle a received message.

        Args:
            message: Message to handle
        """
        handler = self._message_handlers.get(message.message_type)

        if handler:
            try:
                response = await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
                if response and self.router:
                    await self.router.route_message(response)
            except Exception as e:
                logger.error(f"Handler error in {self.agent_id}: {e}")
                # Send error response
                if self.router:
                    error_response = message.create_error_response(str(e))
                    await self.router.route_message(error_response)
        else:
            logger.warning(f"No handler for {message.message_type.value} in {self.agent_id}")

    def stop(self) -> None:
        """Stop message processing."""
        self._is_active = False
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()
        logger.info(f"A2A Connector stopped for agent: {self.agent_id}")

    def is_busy(self) -> bool:
        """Check if agent is currently busy.

        Returns:
            True if agent has a current task
        """
        return self._is_busy or self._current_task is not None

    def set_busy(self, busy: bool, task: str | None = None) -> None:
        """Set the busy state of the agent.

        Args:
            busy: Whether agent is busy
            task: Current task description (if busy)
        """
        self._is_busy = busy
        if busy and task:
            self._current_task = task
        elif not busy:
            self._current_task = None

    def set_current_task(self, task: str | None) -> None:
        """Set the current task.

        Args:
            task: Task description or None to clear
        """
        self._current_task = task
        self._is_busy = task is not None

    def get_call_stack(self) -> list[str]:
        """Get current call stack for circular call detection.

        Returns:
            List of agent IDs in call stack
        """
        return self._call_stack.copy()

    async def acquire_task_slot(self, task: str, timeout: float = 30.0) -> bool:
        """Acquire a task slot for processing.

        Args:
            task: Task description
            timeout: Timeout for acquiring the lock

        Returns:
            True if slot acquired, False if busy or timeout
        """
        try:
            acquired = await asyncio.wait_for(
                self._task_lock.acquire(),
                timeout=timeout
            )
            if acquired:
                self.set_busy(True, task)
                return True
        except asyncio.TimeoutError:
            logger.warning(f"Agent {self.agent_id} busy, could not acquire task slot")
        return False

    def release_task_slot(self) -> None:
        """Release the task slot after processing."""
        self.set_busy(False)
        try:
            self._task_lock.release()
        except RuntimeError:
            pass  # Lock was not acquired
