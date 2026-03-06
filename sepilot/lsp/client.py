"""LSP Client - JSON-RPC communication with language servers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from .models import (
    CallHierarchyIncomingCall,
    CallHierarchyItem,
    CallHierarchyOutgoingCall,
    Diagnostic,
    DocumentSymbol,
    HoverInfo,
    InitializeResult,
    Location,
    LocationLink,
    Position,
    SymbolInformation,
    TextDocumentIdentifier,
)
from .servers import LSPServerConfig

logger = logging.getLogger(__name__)


class LSPClient:
    """Client for communicating with Language Server Protocol servers.

    Handles JSON-RPC communication, request/response tracking, and
    server lifecycle management.
    """

    def __init__(self, config: LSPServerConfig, workspace_root: str):
        """Initialize LSP client.

        Args:
            config: Server configuration
            workspace_root: Root directory of the workspace
        """
        self.config = config
        self.workspace_root = Path(workspace_root).resolve()
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._read_thread: threading.Thread | None = None
        self._running = False
        self._capabilities: InitializeResult | None = None
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Diagnostics storage (uri -> list of Diagnostic)
        self._diagnostics: dict[str, list[Diagnostic]] = {}

    async def start(self) -> bool:
        """Start the language server.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        if not self.config.is_installed():
            logger.warning(f"Server {self.config.name} is not installed")
            return False

        try:
            cmd = self.config.get_full_command()
            logger.info(f"Starting LSP server: {' '.join(cmd)}")

            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workspace_root),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            self._running = True
            self._loop = asyncio.get_event_loop()

            # Start reading thread
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()

            # Initialize the server
            result = await self._initialize()
            if result:
                self._capabilities = result
                await self._initialized()
                return True
            else:
                await self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start LSP server: {e}")
            self._running = False
            return False

    async def stop(self) -> None:
        """Stop the language server."""
        if not self._running:
            return

        self._running = False

        try:
            # Send shutdown request
            await self._request("shutdown", None)
            # Send exit notification
            self._notify("exit", None)
        except Exception:
            pass

        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()

            self._process = None

        # Cancel pending requests
        for future in self._pending_requests.values():
            future.cancel()
        self._pending_requests.clear()

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running and self._process is not None

    @property
    def capabilities(self) -> InitializeResult | None:
        """Get server capabilities."""
        return self._capabilities

    # LSP Protocol Methods

    async def _initialize(self) -> InitializeResult | None:
        """Send initialize request to server."""
        params = {
            "processId": os.getpid(),
            "rootUri": f"file://{self.workspace_root}",
            "rootPath": str(self.workspace_root),
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didSave": True},
                    "completion": {"completionItem": {"snippetSupport": True}},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"linkSupport": True},
                    "references": {},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    "callHierarchy": {},
                },
                "workspace": {
                    "workspaceFolders": True,
                    "symbol": {},
                },
            },
            "workspaceFolders": [
                {
                    "uri": f"file://{self.workspace_root}",
                    "name": self.workspace_root.name,
                }
            ],
        }

        if self.config.init_options:
            params["initializationOptions"] = self.config.init_options

        result = await self._request("initialize", params, timeout=30)
        if result:
            return InitializeResult.from_dict(result)
        return None

    async def _initialized(self) -> None:
        """Send initialized notification."""
        self._notify("initialized", {})

        # Send workspace configuration if needed
        if self.config.workspace_config:
            self._notify(
                "workspace/didChangeConfiguration",
                {"settings": self.config.workspace_config},
            )

    async def open_document(self, file_path: str) -> None:
        """Open a document in the server.

        Args:
            file_path: Path to the file
        """
        path = Path(file_path)
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return

        # Determine language ID
        lang_id = self._get_language_id(path)

        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": f"file://{path}",
                    "languageId": lang_id,
                    "version": 1,
                    "text": content,
                }
            },
        )

    async def close_document(self, file_path: str) -> None:
        """Close a document in the server.

        Args:
            file_path: Path to the file
        """
        self._notify(
            "textDocument/didClose",
            {"textDocument": {"uri": f"file://{file_path}"}},
        )

    def _get_language_id(self, path: Path) -> str:
        """Get language ID for a file."""
        ext = path.suffix.lower()
        lang_map = {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".go": "go",
            ".rs": "rust",
        }
        return lang_map.get(ext, "plaintext")

    # High-level Operations

    async def goto_definition(
        self, file_path: str, line: int, character: int
    ) -> list[Location]:
        """Go to definition of symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            List of locations
        """
        if not self.is_running:
            return []

        await self.open_document(file_path)

        result = await self._request(
            "textDocument/definition",
            {
                "textDocument": {"uri": f"file://{file_path}"},
                "position": {"line": line, "character": character},
            },
        )

        if not result:
            return []

        return self._parse_locations(result)

    async def find_references(
        self, file_path: str, line: int, character: int, include_declaration: bool = True
    ) -> list[Location]:
        """Find all references to symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)
            include_declaration: Include the declaration

        Returns:
            List of locations
        """
        if not self.is_running:
            return []

        await self.open_document(file_path)

        result = await self._request(
            "textDocument/references",
            {
                "textDocument": {"uri": f"file://{file_path}"},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": include_declaration},
            },
        )

        if not result:
            return []

        return self._parse_locations(result)

    async def hover(self, file_path: str, line: int, character: int) -> HoverInfo | None:
        """Get hover information for symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            Hover information or None
        """
        if not self.is_running:
            return None

        await self.open_document(file_path)

        result = await self._request(
            "textDocument/hover",
            {
                "textDocument": {"uri": f"file://{file_path}"},
                "position": {"line": line, "character": character},
            },
        )

        if not result:
            return None

        return HoverInfo.from_dict(result)

    async def document_symbols(
        self, file_path: str
    ) -> list[DocumentSymbol | SymbolInformation]:
        """Get all symbols in a document.

        Args:
            file_path: Path to the file

        Returns:
            List of symbols
        """
        if not self.is_running:
            return []

        await self.open_document(file_path)

        result = await self._request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": f"file://{file_path}"}},
        )

        if not result:
            return []

        symbols = []
        for item in result:
            if "location" in item:
                # SymbolInformation
                symbols.append(SymbolInformation.from_dict(item))
            else:
                # DocumentSymbol
                symbols.append(DocumentSymbol.from_dict(item))

        return symbols

    async def workspace_symbols(self, query: str) -> list[SymbolInformation]:
        """Search for symbols across the workspace.

        Args:
            query: Search query

        Returns:
            List of matching symbols
        """
        if not self.is_running:
            return []

        result = await self._request(
            "workspace/symbol",
            {"query": query},
        )

        if not result:
            return []

        return [SymbolInformation.from_dict(item) for item in result]

    async def prepare_call_hierarchy(
        self, file_path: str, line: int, character: int
    ) -> list[CallHierarchyItem]:
        """Prepare call hierarchy at position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            List of call hierarchy items
        """
        if not self.is_running:
            return []

        await self.open_document(file_path)

        result = await self._request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": f"file://{file_path}"},
                "position": {"line": line, "character": character},
            },
        )

        if not result:
            return []

        return [CallHierarchyItem.from_dict(item) for item in result]

    async def incoming_calls(
        self, item: CallHierarchyItem
    ) -> list[CallHierarchyIncomingCall]:
        """Get incoming calls to a call hierarchy item.

        Args:
            item: Call hierarchy item

        Returns:
            List of incoming calls
        """
        if not self.is_running:
            return []

        result = await self._request(
            "callHierarchy/incomingCalls",
            {"item": item.to_dict()},
        )

        if not result:
            return []

        return [CallHierarchyIncomingCall.from_dict(c) for c in result]

    async def outgoing_calls(
        self, item: CallHierarchyItem
    ) -> list[CallHierarchyOutgoingCall]:
        """Get outgoing calls from a call hierarchy item.

        Args:
            item: Call hierarchy item

        Returns:
            List of outgoing calls
        """
        if not self.is_running:
            return []

        result = await self._request(
            "callHierarchy/outgoingCalls",
            {"item": item.to_dict()},
        )

        if not result:
            return []

        return [CallHierarchyOutgoingCall.from_dict(c) for c in result]

    # Helper Methods

    def _parse_locations(self, result: Any) -> list[Location]:
        """Parse locations from response."""
        if not result:
            return []

        if isinstance(result, dict):
            # Single location
            if "uri" in result:
                return [Location.from_dict(result)]
            # LocationLink
            if "targetUri" in result:
                return [LocationLink.from_dict(result).to_location()]
            return []

        locations = []
        for item in result:
            if "uri" in item:
                locations.append(Location.from_dict(item))
            elif "targetUri" in item:
                locations.append(LocationLink.from_dict(item).to_location())

        return locations

    # JSON-RPC Communication

    def _next_request_id(self) -> int:
        """Get next request ID."""
        with self._lock:
            self._request_id += 1
            return self._request_id

    async def _request(
        self, method: str, params: Any, timeout: float = 10.0
    ) -> Any:
        """Send a request and wait for response.

        Args:
            method: LSP method name
            params: Request parameters
            timeout: Timeout in seconds

        Returns:
            Response result or None
        """
        if not self.is_running or not self._process:
            return None

        request_id = self._next_request_id()
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future

        try:
            self._send_message(message)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"LSP request {method} timed out")
            return None
        except Exception as e:
            logger.error(f"LSP request {method} failed: {e}")
            return None
        finally:
            self._pending_requests.pop(request_id, None)

    def _notify(self, method: str, params: Any) -> None:
        """Send a notification (no response expected).

        Args:
            method: LSP method name
            params: Notification parameters
        """
        if not self.is_running or not self._process:
            return

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._send_message(message)

    def _send_message(self, message: dict) -> None:
        """Send a JSON-RPC message."""
        if not self._process or not self._process.stdin:
            return

        content = json.dumps(message)
        content_bytes = content.encode("utf-8")
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"
        data = header.encode("utf-8") + content_bytes

        try:
            self._process.stdin.write(data)
            self._process.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to send LSP message: {e}")

    def _read_loop(self) -> None:
        """Background thread for reading server responses."""
        while self._running and self._process and self._process.stdout:
            try:
                # Read headers
                content_length = 0
                while True:
                    line = self._process.stdout.readline()
                    if not line:
                        return

                    line = line.decode("utf-8").strip()
                    if not line:
                        break

                    if line.startswith("Content-Length:"):
                        content_length = int(line.split(":")[1].strip())

                if content_length == 0:
                    continue

                # Read content
                content = self._process.stdout.read(content_length)
                if not content:
                    continue

                try:
                    message = json.loads(content.decode("utf-8"))
                except json.JSONDecodeError as e:
                    logger.warning(f"LSP invalid JSON message: {e}")
                    continue

                self._handle_message(message)

            except Exception as e:
                if self._running:
                    logger.error(f"LSP read error: {e}")
                break

    def _handle_message(self, message: dict) -> None:
        """Handle incoming message."""
        if "id" in message:
            # Response
            request_id = message["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests[request_id]
                if "error" in message:
                    error = message["error"]
                    logger.warning(f"LSP error: {error.get('message', error)}")
                    if self._loop:
                        self._loop.call_soon_threadsafe(future.set_result, None)
                else:
                    result = message.get("result")
                    if self._loop:
                        self._loop.call_soon_threadsafe(future.set_result, result)
        elif "method" in message:
            # Notification or request from server
            self._handle_server_message(message)

    def _handle_server_message(self, message: dict) -> None:
        """Handle message from server."""
        method = message.get("method", "")

        if method == "window/logMessage":
            params = message.get("params", {})
            logger.debug(f"LSP: {params.get('message', '')}")
        elif method == "textDocument/publishDiagnostics":
            # Store diagnostics for the file
            params = message.get("params", {})
            uri = params.get("uri", "")
            diagnostics_data = params.get("diagnostics", [])
            diagnostics = [Diagnostic.from_dict(d) for d in diagnostics_data]
            with self._lock:
                self._diagnostics[uri] = diagnostics
            logger.debug(f"LSP: Received {len(diagnostics)} diagnostics for {uri}")
        elif method == "window/showMessage":
            params = message.get("params", {})
            logger.info(f"LSP: {params.get('message', '')}")

    def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Get diagnostics for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of diagnostics
        """
        uri = f"file://{Path(file_path).resolve()}"
        with self._lock:
            return self._diagnostics.get(uri, [])

    def get_all_diagnostics(self) -> dict[str, list[Diagnostic]]:
        """Get all diagnostics for all files.

        Returns:
            Dictionary mapping URIs to diagnostics
        """
        with self._lock:
            return dict(self._diagnostics)

    async def request_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Request diagnostics for a file by opening/saving it.

        This triggers the server to send publishDiagnostics.

        Args:
            file_path: Path to the file

        Returns:
            List of diagnostics (after waiting for server response)
        """
        if not self.is_running:
            return []

        # Open the document to trigger diagnostics
        await self.open_document(file_path)

        # Wait a bit for diagnostics to arrive
        await asyncio.sleep(0.5)

        return self.get_diagnostics(file_path)
