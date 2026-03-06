"""MCP Integration Helpers

Utilities for integrating MCP tools into SEPilot agents.
"""

import asyncio
import concurrent.futures
import logging
import threading

from langchain_core.tools import BaseTool

from sepilot.mcp.client import MCPProtocolError, MCPToolRegistry
from sepilot.mcp.config_manager import MCPConfigManager

logger = logging.getLogger(__name__)

# Thread pool shared across calls to avoid re-creation overhead
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _run_async(coro):
    """Run async coroutine safely from sync context.

    Handles the case where we're already in an async context
    by using a shared thread pool executor.
    """
    try:
        asyncio.get_running_loop()
        # We're in an async context, use shared thread pool
        future = _thread_pool.submit(asyncio.run, coro)
        return future.result(timeout=120)
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)


def get_mcp_tools_for_agent(
    agent_name: str,
    config_manager: MCPConfigManager | None = None
) -> list[BaseTool]:
    """Get MCP tools that an agent can access

    This function:
    1. Loads MCP configuration
    2. Checks which servers the agent can access
    3. Returns LangChain tools from accessible servers

    Args:
        agent_name: Name of the agent (e.g., "github", "git", "se")
        config_manager: Optional MCP config manager (creates new if not provided)

    Returns:
        List of LangChain BaseTool objects from accessible MCP servers

    Example:
        >>> from sepilot.mcp.integration import get_mcp_tools_for_agent
        >>> tools = get_mcp_tools_for_agent("github")
        >>> print(f"GitHub agent has access to {len(tools)} MCP tools")
    """
    try:
        # Create config manager
        if config_manager is None:
            config_manager = MCPConfigManager()

        # Get accessible servers
        accessible_servers = config_manager.get_allowed_servers(agent_name)

        if not accessible_servers:
            logger.debug(f"Agent '{agent_name}' has no accessible MCP servers")
            return []

        logger.info(
            f"Agent '{agent_name}' can access {len(accessible_servers)} MCP servers: "
            f"{', '.join(s.name for s in accessible_servers)}"
        )

        # Create tool registry
        registry = MCPToolRegistry(agent_name, config_manager)

        # Get tools asynchronously using safe async runner
        tools = _run_async(registry.get_langchain_tools())

        logger.info(f"Loaded {len(tools)} MCP tools for agent '{agent_name}'")

        return tools

    except MCPProtocolError as e:
        logger.error(f"MCP protocol error for agent '{agent_name}': {e}")
        return []
    except TimeoutError as e:
        logger.error(f"Timeout loading MCP tools for agent '{agent_name}': {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load MCP tools for agent '{agent_name}': {e}")
        # Return empty list on error - don't break agent initialization
        return []


def check_mcp_access(agent_name: str, server_name: str) -> bool:
    """Check if an agent can access a specific MCP server

    Args:
        agent_name: Name of the agent
        server_name: Name of the MCP server

    Returns:
        True if agent can access the server, False otherwise

    Example:
        >>> from sepilot.mcp.integration import check_mcp_access
        >>> if check_mcp_access("github", "filesystem"):
        ...     print("GitHub agent can access filesystem MCP")
    """
    try:
        manager = MCPConfigManager()
        return manager.can_agent_access(server_name, agent_name)
    except Exception as e:
        logger.error(f"Error checking MCP access: {e}")
        return False


def list_available_mcp_servers(agent_name: str) -> list[str]:
    """List MCP servers available to an agent

    Args:
        agent_name: Name of the agent

    Returns:
        List of server names the agent can access

    Example:
        >>> from sepilot.mcp.integration import list_available_mcp_servers
        >>> servers = list_available_mcp_servers("github")
        >>> print(f"Available servers: {', '.join(servers)}")
    """
    try:
        manager = MCPConfigManager()
        servers = manager.get_allowed_servers(agent_name)
        return [s.name for s in servers]
    except Exception as e:
        logger.error(f"Error listing MCP servers: {e}")
        return []


_config_manager_instance: MCPConfigManager | None = None
_config_manager_lock = threading.Lock()


def get_mcp_config_manager() -> MCPConfigManager:
    """Get a shared MCP configuration manager instance (thread-safe singleton)

    Returns:
        MCPConfigManager instance

    Example:
        >>> from sepilot.mcp.integration import get_mcp_config_manager
        >>> manager = get_mcp_config_manager()
        >>> servers = manager.list_servers()
    """
    global _config_manager_instance
    if _config_manager_instance is None:
        with _config_manager_lock:
            if _config_manager_instance is None:
                _config_manager_instance = MCPConfigManager()
    return _config_manager_instance
