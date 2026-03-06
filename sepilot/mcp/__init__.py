"""MCP (Model Context Protocol) integration for SE Pilot"""

from .client import MCPClient, MCPToolRegistry
from .config_manager import (
    MCPAccessControl,
    MCPConfigManager,
    MCPServerConfig,
)
from .integration import (
    check_mcp_access,
    get_mcp_config_manager,
    get_mcp_tools_for_agent,
    list_available_mcp_servers,
)
from .server import SEPilotMCPServer, run_mcp_server

__all__ = [
    # Server
    "SEPilotMCPServer",
    "run_mcp_server",
    # Configuration
    "MCPConfigManager",
    "MCPServerConfig",
    "MCPAccessControl",
    # Client
    "MCPClient",
    "MCPToolRegistry",
    # Integration
    "get_mcp_tools_for_agent",
    "check_mcp_access",
    "list_available_mcp_servers",
    "get_mcp_config_manager",
]
