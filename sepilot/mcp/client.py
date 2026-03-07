"""MCP Client for connecting to MCP servers

This module provides a client for SEPilot agents to connect to
and use tools from MCP servers via stdio JSON-RPC protocol.
"""

import asyncio
import contextlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sepilot.mcp.config_manager import MCPConfigManager, MCPServerConfig

logger = logging.getLogger(__name__)


class MCPErrorCode(Enum):
    """MCP error codes for better error handling"""
    UNKNOWN = "unknown"
    SERVER_NOT_FOUND = "server_not_found"
    SERVER_DISABLED = "server_disabled"
    PERMISSION_DENIED = "permission_denied"
    COMMAND_NOT_FOUND = "command_not_found"
    CONNECTION_FAILED = "connection_failed"
    PROTOCOL_ERROR = "protocol_error"
    TIMEOUT = "timeout"
    TOOL_NOT_FOUND = "tool_not_found"
    INVALID_ARGUMENTS = "invalid_arguments"


@dataclass
class MCPServerProcess:
    """Represents a running MCP server process (local stdio)"""
    config: MCPServerConfig
    process: asyncio.subprocess.Process
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    request_id: int = 0
    initialized: bool = False
    server_info: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPRemoteConnection:
    """Represents a connection to a remote MCP server"""
    config: MCPServerConfig
    client: Any  # MCPHttpClient or MCPSSEClient
    initialized: bool = False
    server_info: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)


class MCPProtocolError(Exception):
    """MCP Protocol related errors with error codes and suggestions"""

    def __init__(
        self,
        message: str,
        code: MCPErrorCode = MCPErrorCode.UNKNOWN,
        suggestion: str | None = None
    ):
        super().__init__(message)
        self.code = code
        self.suggestion = suggestion or self._get_default_suggestion(code)

    def _get_default_suggestion(self, code: MCPErrorCode) -> str:
        """Get default suggestion for error code"""
        suggestions = {
            MCPErrorCode.SERVER_NOT_FOUND: "Check if the server is configured with '/mcp list'",
            MCPErrorCode.SERVER_DISABLED: "Enable the server with '/mcp <name> enable'",
            MCPErrorCode.PERMISSION_DENIED: "Check access control with '/mcp <name> access'",
            MCPErrorCode.COMMAND_NOT_FOUND: "Verify the server command is installed and in PATH",
            MCPErrorCode.CONNECTION_FAILED: "Check if the server process can start correctly",
            MCPErrorCode.PROTOCOL_ERROR: "The server may not be MCP-compatible",
            MCPErrorCode.TIMEOUT: "The server might be slow or unresponsive. Try again or increase timeout",
            MCPErrorCode.TOOL_NOT_FOUND: "Check available tools with '/mcp <name> tools'",
            MCPErrorCode.INVALID_ARGUMENTS: "Check the tool's input schema for required arguments",
            MCPErrorCode.UNKNOWN: "Check server logs for more details",
        }
        return suggestions.get(code, "")

    def __str__(self) -> str:
        base = super().__str__()
        if self.suggestion:
            return f"{base}\n  Tip: {self.suggestion}"
        return base


def _validate_server_command(command: str) -> tuple[bool, str]:
    """Validate that a server command exists and is executable.

    Args:
        command: Command to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command:
        return False, "Command is empty"

    # Check if command exists in PATH
    cmd_path = shutil.which(command)
    if cmd_path is None:
        return False, f"Command '{command}' not found in PATH"

    return True, ""


def _validate_tool_arguments(
    arguments: dict[str, Any],
    input_schema: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate tool arguments against schema.

    Args:
        arguments: Arguments to validate
        input_schema: JSON schema for the tool

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    schema_props = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    # Check required arguments
    for req_arg in required:
        if req_arg not in arguments:
            errors.append(f"Missing required argument: '{req_arg}'")
        elif arguments[req_arg] is None:
            errors.append(f"Required argument '{req_arg}' is null")

    # Check argument types
    for arg_name, arg_value in arguments.items():
        if arg_name not in schema_props:
            # Unknown argument - warn but don't fail
            logger.warning(f"Unknown argument '{arg_name}' not in schema")
            continue

        expected_type = schema_props[arg_name].get("type")
        if expected_type and arg_value is not None:
            type_mapping = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            expected_python_type = type_mapping.get(expected_type)
            if expected_python_type and not isinstance(arg_value, expected_python_type):
                errors.append(
                    f"Argument '{arg_name}' should be {expected_type}, got {type(arg_value).__name__}"
                )

    return len(errors) == 0, errors


class MCPClient:
    """Client for connecting to and using MCP servers via stdio JSON-RPC

    This client allows SEPilot agents to:
    1. List available MCP servers they can access
    2. List tools provided by MCP servers
    3. Execute tools on MCP servers

    Uses MCP protocol over stdio with JSON-RPC 2.0 message format.
    """

    def __init__(self, agent_name: str, config_manager: MCPConfigManager | None = None):
        """Initialize MCP client

        Args:
            agent_name: Name of the agent using this client (e.g., "github", "git")
            config_manager: MCP configuration manager (creates new one if not provided)
        """
        self.agent_name = agent_name.lower()
        self.config_manager = config_manager or MCPConfigManager()

        # Cache of available servers and their tools
        self._server_tools_cache: dict[str, list[dict[str, Any]]] = {}

        # Running server processes
        self._server_processes: dict[str, MCPServerProcess] = {}

        # Remote server connections
        self._remote_connections: dict[str, MCPRemoteConnection] = {}

        # Protocol version
        self.protocol_version = "2024-11-05"

    def get_available_servers(self) -> list[MCPServerConfig]:
        """Get list of MCP servers this agent can access

        Returns:
            List of accessible MCPServerConfig objects
        """
        return self.config_manager.get_allowed_servers(self.agent_name)

    async def _start_server_process(self, server: MCPServerConfig) -> MCPServerProcess:
        """Start an MCP server subprocess

        Args:
            server: Server configuration

        Returns:
            MCPServerProcess object

        Raises:
            MCPProtocolError: If server fails to start
        """
        if server.name in self._server_processes:
            existing = self._server_processes[server.name]
            if existing.process.returncode is None:  # Still running
                return existing
            else:
                # Process died, clean up
                del self._server_processes[server.name]

        # Validate command exists before trying to run
        is_valid, error_msg = _validate_server_command(server.command)
        if not is_valid:
            raise MCPProtocolError(
                f"Server '{server.name}' command not found: {error_msg}",
                code=MCPErrorCode.COMMAND_NOT_FOUND,
                suggestion=f"Install '{server.command}' or check your PATH"
            )

        # Prepare environment
        env = os.environ.copy()
        if server.env:
            env.update(server.env)

        # Build command
        cmd = [server.command] + (server.args or [])

        logger.info(f"Starting MCP server '{server.name}': {' '.join(cmd)}")

        try:
            # Start subprocess with stdio pipes
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # Check if process started successfully
            await asyncio.sleep(0.1)  # Give process time to fail
            if process.returncode is not None:
                # Process already exited
                stderr_output = ""
                if process.stderr:
                    stderr_output = (await process.stderr.read()).decode()[:500]
                raise MCPProtocolError(
                    f"Server '{server.name}' exited immediately (code {process.returncode}): {stderr_output}",
                    code=MCPErrorCode.CONNECTION_FAILED,
                    suggestion="Check server command and arguments"
                )

            server_process = MCPServerProcess(
                config=server,
                process=process,
                reader=process.stdout,
                writer=process.stdin
            )

            self._server_processes[server.name] = server_process

            # Initialize the connection
            await self._initialize_server(server_process)

            return server_process

        except MCPProtocolError:
            raise
        except FileNotFoundError as e:
            raise MCPProtocolError(
                f"Server command not found: {server.command}",
                code=MCPErrorCode.COMMAND_NOT_FOUND,
                suggestion=f"Install '{server.command}' and ensure it's in your PATH"
            ) from e
        except PermissionError as e:
            raise MCPProtocolError(
                f"Permission denied running '{server.command}'",
                code=MCPErrorCode.CONNECTION_FAILED,
                suggestion="Check file permissions or run with appropriate privileges"
            ) from e
        except Exception as e:
            logger.error(f"Failed to start MCP server '{server.name}': {e}")
            raise MCPProtocolError(
                f"Failed to start server '{server.name}': {e}",
                code=MCPErrorCode.CONNECTION_FAILED
            ) from e

    async def _initialize_server(self, server_process: MCPServerProcess) -> None:
        """Initialize MCP protocol with server

        Args:
            server_process: Server process to initialize
        """
        # Send initialize request
        response = await self._send_request(
            server_process,
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "sepilot",
                    "version": __import__("sepilot").__version__
                }
            }
        )

        if "error" in response:
            raise MCPProtocolError(f"Initialize failed: {response['error']}")

        result = response.get("result", {})
        server_process.server_info = result.get("serverInfo", {})
        server_process.capabilities = result.get("capabilities", {})

        # Send initialized notification
        await self._send_notification(server_process, "notifications/initialized", {})

        server_process.initialized = True
        logger.info(f"MCP server '{server_process.config.name}' initialized: {server_process.server_info}")

    async def _send_request(
        self,
        server_process: MCPServerProcess,
        method: str,
        params: dict[str, Any],
        timeout: float = 30.0
    ) -> dict[str, Any]:
        """Send JSON-RPC request and wait for response

        Args:
            server_process: Server process
            method: RPC method name
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            Response dictionary
        """
        server_process.request_id += 1
        request_id = server_process.request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        # Send request
        message = json.dumps(request) + "\n"
        server_process.writer.write(message.encode())
        await server_process.writer.drain()

        logger.debug(f"Sent request to '{server_process.config.name}': {method}")

        # Wait for response
        try:
            response = await asyncio.wait_for(
                self._read_response(server_process, request_id),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError as e:
            raise MCPProtocolError(
                f"Request '{method}' to server '{server_process.config.name}' timed out after {timeout}s",
                code=MCPErrorCode.TIMEOUT
            ) from e

    async def _send_notification(
        self,
        server_process: MCPServerProcess,
        method: str,
        params: dict[str, Any]
    ) -> None:
        """Send JSON-RPC notification (no response expected)

        Args:
            server_process: Server process
            method: Notification method name
            params: Notification parameters
        """
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        message = json.dumps(notification) + "\n"
        server_process.writer.write(message.encode())
        await server_process.writer.drain()

        logger.debug(f"Sent notification to '{server_process.config.name}': {method}")

    async def _read_response(
        self,
        server_process: MCPServerProcess,
        expected_id: int
    ) -> dict[str, Any]:
        """Read JSON-RPC response from server

        Args:
            server_process: Server process
            expected_id: Expected response ID

        Returns:
            Response dictionary
        """
        while True:
            line = await server_process.reader.readline()
            if not line:
                raise MCPProtocolError(
                    f"Server '{server_process.config.name}' closed connection unexpectedly",
                    code=MCPErrorCode.CONNECTION_FAILED,
                    suggestion="The server process may have crashed. Check server logs"
                )

            try:
                response = json.loads(line.decode().strip())
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from server: {line[:100]}")
                continue

            # Skip notifications
            if "id" not in response:
                logger.debug(f"Received notification: {response.get('method', 'unknown')}")
                continue

            if response.get("id") == expected_id:
                return response
            else:
                logger.warning(f"Unexpected response ID: {response.get('id')} (expected {expected_id})")

    async def list_tools_from_server(self, server_name: str) -> list[dict[str, Any]]:
        """List tools available from a specific MCP server

        Args:
            server_name: Name of the MCP server

        Returns:
            List of tool definitions

        Raises:
            MCPProtocolError: If access denied or communication fails
        """
        # Check access permission
        if not self.config_manager.can_agent_access(server_name, self.agent_name):
            raise MCPProtocolError(
                f"Agent '{self.agent_name}' does not have permission to access server '{server_name}'",
                code=MCPErrorCode.PERMISSION_DENIED
            )

        # Check cache
        if server_name in self._server_tools_cache:
            return self._server_tools_cache[server_name]

        # Get server config
        server = self.config_manager.get_server(server_name)
        if not server:
            raise MCPProtocolError(
                f"Server '{server_name}' not found",
                code=MCPErrorCode.SERVER_NOT_FOUND
            )

        if not server.enabled:
            raise MCPProtocolError(
                f"Server '{server_name}' is disabled",
                code=MCPErrorCode.SERVER_DISABLED
            )

        # Query server for tools
        try:
            tools = await self._query_server_tools(server)
            self._server_tools_cache[server.name] = tools
            return tools
        except MCPProtocolError:
            raise
        except Exception as e:
            logger.error(f"Failed to list tools from server '{server_name}': {e}")
            raise MCPProtocolError(
                f"Failed to communicate with server '{server_name}': {e}",
                code=MCPErrorCode.CONNECTION_FAILED
            ) from e

    async def list_all_available_tools(self) -> dict[str, list[dict[str, Any]]]:
        """List all tools from all accessible servers concurrently

        Returns:
            Dictionary mapping server name to list of tools
        """
        servers = [s for s in self.get_available_servers() if s.enabled]

        if not servers:
            return {}

        async def _get_tools(server: MCPServerConfig) -> tuple[str, list[dict[str, Any]]]:
            try:
                tools = await self.list_tools_from_server(server.name)
                return server.name, tools
            except Exception as e:
                logger.warning(f"Failed to get tools from '{server.name}': {e}")
                return server.name, []

        # Query all servers concurrently
        results = await asyncio.gather(
            *[_get_tools(s) for s in servers],
            return_exceptions=True
        )

        all_tools = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error querying server: {result}")
                continue
            name, tools = result
            all_tools[name] = tools

        return all_tools

    async def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        validate_args: bool = True
    ) -> dict[str, Any]:
        """Execute a tool on an MCP server

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            validate_args: Whether to validate arguments against schema

        Returns:
            Tool execution result

        Raises:
            MCPProtocolError: If access denied or execution fails
        """
        # Check access permission
        if not self.config_manager.can_agent_access(server_name, self.agent_name):
            raise MCPProtocolError(
                f"Agent '{self.agent_name}' does not have permission to access server '{server_name}'",
                code=MCPErrorCode.PERMISSION_DENIED
            )

        # Get server config
        server = self.config_manager.get_server(server_name)
        if not server:
            raise MCPProtocolError(
                f"Server '{server_name}' not found",
                code=MCPErrorCode.SERVER_NOT_FOUND
            )

        if not server.enabled:
            raise MCPProtocolError(
                f"Server '{server_name}' is disabled",
                code=MCPErrorCode.SERVER_DISABLED
            )

        # Validate arguments if requested and we have schema cached
        if validate_args and server_name in self._server_tools_cache:
            tools = self._server_tools_cache[server_name]
            tool_schema = next(
                (t.get("inputSchema", {}) for t in tools if t.get("name") == tool_name),
                None
            )
            if tool_schema:
                is_valid, errors = _validate_tool_arguments(arguments, tool_schema)
                if not is_valid:
                    raise MCPProtocolError(
                        f"Invalid arguments for tool '{tool_name}': {'; '.join(errors)}",
                        code=MCPErrorCode.INVALID_ARGUMENTS
                    )

        # Execute tool
        try:
            result = await self._execute_tool_on_server(server, tool_name, arguments)
            return result
        except MCPProtocolError:
            raise
        except Exception as e:
            logger.error(f"Failed to execute tool '{tool_name}' on server '{server_name}': {e}")
            raise MCPProtocolError(
                f"Tool execution failed: {e}",
                code=MCPErrorCode.UNKNOWN
            ) from e

    async def _query_server_tools(self, server: MCPServerConfig) -> list[dict[str, Any]]:
        """Query MCP server for available tools via JSON-RPC

        Args:
            server: Server configuration

        Returns:
            List of tool definitions
        """
        # Handle remote servers
        if server.is_remote():
            return await self._query_remote_server_tools(server)

        # Start/get server process (local stdio)
        server_process = await self._start_server_process(server)

        # Send tools/list request
        response = await self._send_request(
            server_process,
            "tools/list",
            {}
        )

        if "error" in response:
            error = response["error"]
            raise MCPProtocolError(f"tools/list failed: {error.get('message', error)}")

        result = response.get("result", {})
        tools = result.get("tools", [])

        logger.info(f"Got {len(tools)} tools from server '{server.name}'")
        return tools

    async def _query_remote_server_tools(self, server: MCPServerConfig) -> list[dict[str, Any]]:
        """Query remote MCP server for available tools

        Args:
            server: Remote server configuration

        Returns:
            List of tool definitions
        """
        client = await self._get_remote_client(server)
        tools = await client.client.list_tools()
        logger.info(f"Got {len(tools)} tools from remote server '{server.name}'")
        return tools

    async def _get_remote_client(self, server: MCPServerConfig) -> MCPRemoteConnection:
        """Get or create a remote client connection

        Args:
            server: Remote server configuration

        Returns:
            MCPRemoteConnection object
        """
        if server.name in self._remote_connections:
            conn = self._remote_connections[server.name]
            if conn.initialized:
                return conn

        # Import remote client
        from sepilot.mcp.remote_client import create_remote_client

        # Get headers (including OAuth if configured)
        headers = {}
        if server.oauth:
            from sepilot.mcp.oauth_handler import authenticate_mcp_server
            try:
                auth_headers = await authenticate_mcp_server(server.oauth)
                headers.update(auth_headers)
            except Exception as e:
                logger.warning(f"OAuth authentication failed for {server.name}: {e}")
                raise MCPProtocolError(
                    f"OAuth authentication failed: {e}",
                    code=MCPErrorCode.PERMISSION_DENIED
                ) from e

        # Create client
        client = create_remote_client(
            url=server.url,
            transport=server.transport,
            headers=headers,
        )

        # Connect
        try:
            server_info = await client.connect()
            conn = MCPRemoteConnection(
                config=server,
                client=client,
                initialized=True,
                server_info={"name": server_info.name, "version": server_info.version},
                capabilities=server_info.capabilities,
            )
            self._remote_connections[server.name] = conn
            logger.info(f"Connected to remote MCP server: {server.name} ({server.url})")
            return conn

        except Exception as e:
            raise MCPProtocolError(
                f"Failed to connect to remote server '{server.name}': {e}",
                code=MCPErrorCode.CONNECTION_FAILED
            ) from e

    async def _execute_tool_on_server(
        self,
        server: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool on an MCP server via JSON-RPC

        Args:
            server: Server configuration
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        # Handle remote servers
        if server.is_remote():
            conn = await self._get_remote_client(server)
            return await conn.client.call_tool(tool_name, arguments)

        # Start/get server process (local stdio)
        server_process = await self._start_server_process(server)

        # Send tools/call request
        response = await self._send_request(
            server_process,
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments
            },
            timeout=60.0  # Longer timeout for tool execution
        )

        if "error" in response:
            error = response["error"]
            error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            return {
                "success": False,
                "error": error_msg,
                "isError": True
            }

        result = response.get("result", {})

        # Extract content from MCP response
        content = result.get("content", [])
        is_error = result.get("isError", False)

        # Format response
        if is_error:
            error_text = ""
            for item in content:
                if item.get("type") == "text":
                    error_text += item.get("text", "")
            return {
                "success": False,
                "error": error_text or "Tool execution failed",
                "isError": True
            }

        # Extract text content
        output_text = ""
        for item in content:
            if item.get("type") == "text":
                output_text += item.get("text", "")
            elif item.get("type") == "resource":
                output_text += f"\n[Resource: {item.get('resource', {}).get('uri', 'unknown')}]"

        return {
            "success": True,
            "output": output_text,
            "content": content,
            "isError": False
        }

    async def stop_server(self, server_name: str) -> None:
        """Stop a running MCP server

        Args:
            server_name: Name of the server to stop
        """
        if server_name not in self._server_processes:
            return

        server_process = self._server_processes[server_name]

        try:
            # Send shutdown notification (if supported)
            with contextlib.suppress(Exception):
                await self._send_notification(server_process, "notifications/shutdown", {})

            # Terminate process
            server_process.process.terminate()

            # Wait for termination
            try:
                await asyncio.wait_for(server_process.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                server_process.process.kill()
                await server_process.process.wait()

            logger.info(f"Stopped MCP server '{server_name}'")

        except Exception as e:
            logger.warning(f"Error stopping server '{server_name}': {e}")

        finally:
            del self._server_processes[server_name]
            # Clear cache for this server
            if server_name in self._server_tools_cache:
                del self._server_tools_cache[server_name]

    async def stop_all_servers(self) -> None:
        """Stop all running MCP servers (local and remote)"""
        # Stop local servers
        server_names = list(self._server_processes.keys())
        for name in server_names:
            await self.stop_server(name)

        # Close remote connections
        remote_names = list(self._remote_connections.keys())
        for name in remote_names:
            await self._close_remote_connection(name)

    async def _close_remote_connection(self, server_name: str) -> None:
        """Close a remote server connection

        Args:
            server_name: Name of the server
        """
        if server_name not in self._remote_connections:
            return

        conn = self._remote_connections[server_name]
        try:
            await conn.client.close()
            logger.info(f"Closed remote connection: {server_name}")
        except Exception as e:
            logger.warning(f"Error closing remote connection '{server_name}': {e}")
        finally:
            del self._remote_connections[server_name]
            if server_name in self._server_tools_cache:
                del self._server_tools_cache[server_name]

    def clear_cache(self):
        """Clear the tools cache"""
        self._server_tools_cache.clear()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup servers"""
        await self.stop_all_servers()


class MCPToolRegistry:
    """Registry for converting MCP tools to LangChain tools

    This class helps integrate MCP tools into SEPilot's existing tool system.
    """

    def __init__(self, agent_name: str, config_manager: MCPConfigManager | None = None):
        """Initialize MCP tool registry

        Args:
            agent_name: Name of the agent
            config_manager: MCP configuration manager
        """
        self.client = MCPClient(agent_name, config_manager)

    async def get_langchain_tools(self) -> list[Any]:
        """Get MCP tools as LangChain tools

        This method:
        1. Lists all accessible MCP servers
        2. Gets tools from each server
        3. Wraps them as LangChain tools

        Returns:
            List of LangChain tool objects
        """
        from langchain_core.tools import StructuredTool
        from pydantic import Field, create_model

        langchain_tools = []

        # Get all available tools
        try:
            all_tools = await self.client.list_all_available_tools()
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            return []

        for server_name, tools in all_tools.items():
            for tool in tools:
                try:
                    tool_name = tool.get("name", "unknown")
                    description = tool.get("description", f"Tool from MCP server '{server_name}'")
                    input_schema = tool.get("inputSchema", {})

                    # Create unique tool name
                    full_tool_name = f"mcp_{server_name}_{tool_name}"

                    # Create Pydantic model from input schema
                    fields = {}
                    schema_props = input_schema.get("properties", {})
                    required = input_schema.get("required", [])

                    for prop_name, prop_schema in schema_props.items():
                        prop_type = str  # Default to string
                        if prop_schema.get("type") == "integer":
                            prop_type = int
                        elif prop_schema.get("type") == "number":
                            prop_type = float
                        elif prop_schema.get("type") == "boolean":
                            prop_type = bool
                        elif prop_schema.get("type") == "array":
                            prop_type = list
                        elif prop_schema.get("type") == "object":
                            prop_type = dict

                        prop_desc = prop_schema.get("description", "")

                        if prop_name in required:
                            fields[prop_name] = (prop_type, Field(description=prop_desc))
                        else:
                            fields[prop_name] = (prop_type | None, Field(default=None, description=prop_desc))

                    # Create input model
                    if fields:
                        InputModel = create_model(f"{full_tool_name}_Input", **fields)
                    else:
                        InputModel = create_model(f"{full_tool_name}_Input")

                    # Create closure with captured variables
                    def make_tool_func(srv_name: str, tl_name: str):
                        async def execute_mcp_tool(**kwargs) -> str:
                            try:
                                result = await self.client.execute_tool(srv_name, tl_name, kwargs)
                                if result.get("success"):
                                    return result.get("output", json.dumps(result))
                                else:
                                    return f"Error: {result.get('error', 'Unknown error')}"
                            except Exception as e:
                                return f"Error executing MCP tool: {e}"
                        return execute_mcp_tool

                    tool_func = make_tool_func(server_name, tool_name)

                    def make_sync_wrapper(afunc):
                        def sync_wrapper(**kwargs):
                            return asyncio.run(afunc(**kwargs))
                        return sync_wrapper

                    # Create LangChain tool
                    langchain_tool = StructuredTool.from_function(
                        func=make_sync_wrapper(tool_func),
                        name=full_tool_name,
                        description=f"[MCP:{server_name}] {description}",
                        args_schema=InputModel,
                        coroutine=tool_func
                    )

                    langchain_tools.append(langchain_tool)
                    logger.debug(f"Created LangChain tool: {full_tool_name}")

                except Exception as e:
                    logger.warning(f"Failed to create LangChain tool for '{tool.get('name', 'unknown')}': {e}")

        logger.info(f"Created {len(langchain_tools)} LangChain tools from MCP servers")
        return langchain_tools

    async def cleanup(self) -> None:
        """Cleanup MCP client resources"""
        await self.client.stop_all_servers()
