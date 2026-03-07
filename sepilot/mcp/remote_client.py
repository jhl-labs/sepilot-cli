"""MCP Remote Client for HTTP/SSE transport.

Supports connecting to remote MCP servers via:
- HTTP (JSON-RPC over HTTP POST)
- SSE (Server-Sent Events for streaming)

Network Configuration:
- Proxy support via HTTP_PROXY, HTTPS_PROXY environment variables
- SSL verification control via SSL_VERIFY environment variable
- Custom CA certificates via SSL_CERT_FILE environment variable
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


def _get_async_http_client_kwargs(timeout: float) -> dict:
    """Get async HTTP client kwargs with proxy and SSL configuration from environment.

    Args:
        timeout: Request timeout in seconds

    Returns:
        Dictionary of kwargs for httpx.AsyncClient
    """
    kwargs = {
        "timeout": httpx.Timeout(timeout),
    }

    # Proxy configuration from environment
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    if https_proxy or http_proxy:
        kwargs["proxy"] = https_proxy or http_proxy

    # SSL verification from environment
    ssl_verify = os.getenv("SSL_VERIFY", "true").lower() not in ("false", "0", "no")
    ssl_cert_file = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")

    if not ssl_verify:
        kwargs["verify"] = False
        logger.warning("SSL verification disabled for MCP remote client")
    elif ssl_cert_file:
        kwargs["verify"] = ssl_cert_file

    return kwargs


class MCPRemoteError(Exception):
    """Remote MCP connection error"""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


@dataclass
class MCPRemoteServerInfo:
    """Information about a remote MCP server"""
    url: str
    name: str = ""
    version: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "2024-11-05"


class MCPHttpClient:
    """MCP client using HTTP transport (JSON-RPC over HTTP POST)"""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        """Initialize HTTP client

        Args:
            base_url: Base URL of the MCP server
            headers: Additional HTTP headers (e.g., for authentication)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        }
        self.request_id = 0
        self.initialized = False
        self.server_info: MCPRemoteServerInfo | None = None

        # HTTP client
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with proxy/SSL support"""
        if self._client is None or self._client.is_closed:
            client_kwargs = _get_async_http_client_kwargs(self.timeout)
            client_kwargs["headers"] = self.headers
            self._client = httpx.AsyncClient(**client_kwargs)
        return self._client

    async def connect(self) -> MCPRemoteServerInfo:
        """Connect and initialize the MCP server

        Returns:
            Server information

        Raises:
            MCPRemoteError: If connection fails
        """
        # Send initialize request
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": "sepilot",
                    "version": __import__("sepilot").__version__,
                },
            },
        )

        if "error" in response:
            raise MCPRemoteError(f"Initialize failed: {response['error']}")

        result = response.get("result", {})
        self.server_info = MCPRemoteServerInfo(
            url=self.base_url,
            name=result.get("serverInfo", {}).get("name", "unknown"),
            version=result.get("serverInfo", {}).get("version", ""),
            capabilities=result.get("capabilities", {}),
        )

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

        self.initialized = True
        logger.info(f"Connected to remote MCP server: {self.server_info.name}")

        return self.server_info

    async def _send_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Send JSON-RPC request

        Args:
            method: RPC method name
            params: Method parameters

        Returns:
            Response dictionary
        """
        self.request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params,
        }

        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.base_url}/rpc",
                json=request,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            raise MCPRemoteError(
                f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise MCPRemoteError(f"Request failed: {e}") from e

    async def _send_notification(
        self, method: str, params: dict[str, Any]
    ) -> None:
        """Send JSON-RPC notification (no response expected)

        Args:
            method: Notification method name
            params: Notification parameters
        """
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        client = await self._get_client()

        try:
            await client.post(
                f"{self.base_url}/rpc",
                json=notification,
            )
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools

        Returns:
            List of tool definitions
        """
        if not self.initialized:
            await self.connect()

        response = await self._send_request("tools/list", {})

        if "error" in response:
            raise MCPRemoteError(f"tools/list failed: {response['error']}")

        return response.get("result", {}).get("tools", [])

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self.initialized:
            await self.connect()

        response = await self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )

        if "error" in response:
            return {
                "success": False,
                "error": response["error"].get("message", str(response["error"])),
                "isError": True,
            }

        result = response.get("result", {})
        content = result.get("content", [])
        is_error = result.get("isError", False)

        # Extract text content
        output_text = ""
        for item in content:
            if item.get("type") == "text":
                output_text += item.get("text", "")

        return {
            "success": not is_error,
            "output": output_text,
            "content": content,
            "isError": is_error,
        }

    async def close(self) -> None:
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self.initialized = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class MCPSSEClient:
    """MCP client using Server-Sent Events (SSE) transport"""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ):
        """Initialize SSE client

        Args:
            base_url: Base URL of the MCP server
            headers: Additional HTTP headers
            timeout: Connection timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Accept": "text/event-stream",
            **(headers or {}),
        }
        self.request_id = 0
        self.initialized = False
        self.server_info: MCPRemoteServerInfo | None = None

        # Pending requests waiting for responses
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._client: httpx.AsyncClient | None = None
        self._sse_task: asyncio.Task | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with proxy/SSL support"""
        if self._client is None or self._client.is_closed:
            client_kwargs = _get_async_http_client_kwargs(self.timeout)
            client_kwargs["headers"] = self.headers
            self._client = httpx.AsyncClient(**client_kwargs)
        return self._client

    async def connect(self) -> MCPRemoteServerInfo:
        """Connect to SSE endpoint and initialize

        Returns:
            Server information
        """
        # Start SSE listener
        self._sse_task = asyncio.create_task(self._listen_sse())

        # Wait for connection
        await asyncio.sleep(0.5)

        # Send initialize request
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sepilot", "version": __import__("sepilot").__version__},
            },
        )

        result = response.get("result", {})
        self.server_info = MCPRemoteServerInfo(
            url=self.base_url,
            name=result.get("serverInfo", {}).get("name", "unknown"),
            version=result.get("serverInfo", {}).get("version", ""),
            capabilities=result.get("capabilities", {}),
        )

        self.initialized = True
        return self.server_info

    async def _listen_sse(self) -> None:
        """Listen for SSE events"""
        client = await self._get_client()

        try:
            async with client.stream("GET", f"{self.base_url}/sse") as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            message = json.loads(data)
                            await self._handle_message(message)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid SSE data: {data}")

        except Exception as e:
            logger.error(f"SSE connection error: {e}")

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming SSE message

        Args:
            message: Parsed JSON message
        """
        # Check if it's a response to a pending request
        request_id = message.get("id")
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_result(message)

    async def _send_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Send request and wait for response

        Args:
            method: RPC method name
            params: Method parameters

        Returns:
            Response dictionary
        """
        self.request_id += 1
        request_id = self.request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        # Create future for response
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future

        # Send request
        client = await self._get_client()
        await client.post(
            f"{self.base_url}/rpc",
            json=request,
            headers={"Content-Type": "application/json"},
        )

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=self.timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise MCPRemoteError(f"Request timed out: {method}")

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools"""
        if not self.initialized:
            await self.connect()

        response = await self._send_request("tools/list", {})
        return response.get("result", {}).get("tools", [])

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool"""
        if not self.initialized:
            await self.connect()

        response = await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

        if "error" in response:
            return {
                "success": False,
                "error": response["error"].get("message", str(response["error"])),
                "isError": True,
            }

        result = response.get("result", {})
        content = result.get("content", [])
        output_text = "".join(
            item.get("text", "") for item in content if item.get("type") == "text"
        )

        return {
            "success": not result.get("isError", False),
            "output": output_text,
            "content": content,
            "isError": result.get("isError", False),
        }

    async def close(self) -> None:
        """Close the SSE client"""
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
            self._sse_task = None

        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

        self._pending_requests.clear()
        self.initialized = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def create_remote_client(
    url: str,
    transport: str = "http",
    headers: dict[str, str] | None = None,
) -> MCPHttpClient | MCPSSEClient:
    """Factory function to create appropriate remote client

    Args:
        url: Server URL
        transport: Transport type ("http" or "sse")
        headers: Additional headers

    Returns:
        Remote client instance
    """
    if transport == "sse":
        return MCPSSEClient(url, headers)
    return MCPHttpClient(url, headers)
