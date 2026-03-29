"""High-level LSP operations with caching and multi-language support."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from sepilot.tools.code_analysis.language_detector import get_language_detector

from .client import LSPClient
from .models import (
    Diagnostic,
    DocumentSymbol,
    HoverInfo,
    Location,
    SymbolInformation,
)

logger = logging.getLogger(__name__)


class LSPOperations:
    """High-level LSP operations with caching and error handling.

    Manages multiple language servers and provides a unified API.
    """

    def __init__(self, workspace_root: str | Path):
        """Initialize LSP operations.

        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = Path(workspace_root).resolve()
        self._clients: dict[str, LSPClient] = {}
        self._detector = get_language_detector()
        self._cache: dict[str, Any] = {}
        self._cache_ttl = 60  # seconds

    async def start_server(self, language: str) -> bool:
        """Start a language server.

        Args:
            language: Language identifier

        Returns:
            True if started successfully
        """
        if language in self._clients and self._clients[language].is_running:
            return True

        from .servers import get_server_config

        config = get_server_config(language)
        if not config:
            logger.warning(f"No server configuration for {language}")
            return False

        client = LSPClient(config, str(self.workspace_root))
        success = await client.start()

        if success:
            self._clients[language] = client
            logger.info(f"Started LSP server for {language}")
        else:
            logger.warning(f"Failed to start LSP server for {language}")

        return success

    async def stop_server(self, language: str) -> None:
        """Stop a language server.

        Args:
            language: Language identifier
        """
        if language in self._clients:
            await self._clients[language].stop()
            del self._clients[language]

    async def stop_all(self) -> None:
        """Stop all language servers."""
        for client in self._clients.values():
            await client.stop()
        self._clients.clear()

    def get_client_for_file(self, file_path: str) -> LSPClient | None:
        """Get the appropriate client for a file.

        Args:
            file_path: Path to the file

        Returns:
            LSPClient or None
        """
        language = self._detector.detect_from_path(file_path)
        if language.value in self._clients:
            return self._clients[language.value]
        return None

    async def ensure_server_for_file(self, file_path: str) -> LSPClient | None:
        """Ensure a server is running for a file.

        Args:
            file_path: Path to the file

        Returns:
            LSPClient or None
        """
        language = self._detector.detect_from_path(file_path)
        if language.value == "unknown":
            return None

        if language.value not in self._clients:
            success = await self.start_server(language.value)
            if not success:
                return None

        return self._clients.get(language.value)

    # High-level Operations

    async def goto_definition(
        self, file_path: str, line: int, character: int
    ) -> list[Location]:
        """Go to definition of symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed, as shown in editors)
            character: Character offset (1-indexed)

        Returns:
            List of definition locations
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return []

        # Convert to 0-indexed
        return await client.goto_definition(
            str(Path(file_path).resolve()),
            line - 1,
            character - 1,
        )

    async def find_references(
        self, file_path: str, line: int, character: int, include_declaration: bool = True
    ) -> list[Location]:
        """Find all references to symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            character: Character offset (1-indexed)
            include_declaration: Include the declaration

        Returns:
            List of reference locations
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return []

        return await client.find_references(
            str(Path(file_path).resolve()),
            line - 1,
            character - 1,
            include_declaration,
        )

    async def hover(
        self, file_path: str, line: int, character: int
    ) -> HoverInfo | None:
        """Get hover information for symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            character: Character offset (1-indexed)

        Returns:
            Hover information or None
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return None

        return await client.hover(
            str(Path(file_path).resolve()),
            line - 1,
            character - 1,
        )

    async def document_symbols(
        self, file_path: str
    ) -> list[DocumentSymbol | SymbolInformation]:
        """Get all symbols in a document.

        Args:
            file_path: Path to the file

        Returns:
            List of symbols
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return []

        return await client.document_symbols(str(Path(file_path).resolve()))

    async def workspace_symbols(self, query: str, language: str | None = None) -> list[SymbolInformation]:
        """Search for symbols across the workspace.

        Args:
            query: Search query
            language: Optional language filter

        Returns:
            List of matching symbols
        """
        results = []

        if language:
            if language in self._clients:
                results = await self._clients[language].workspace_symbols(query)
        else:
            # Search all active servers
            for client in self._clients.values():
                symbols = await client.workspace_symbols(query)
                results.extend(symbols)

        return results

    async def get_call_hierarchy(
        self, file_path: str, line: int, character: int, direction: str = "both"
    ) -> dict:
        """Get call hierarchy for symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            character: Character offset (1-indexed)
            direction: "incoming", "outgoing", or "both"

        Returns:
            Call hierarchy tree
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return {}

        items = await client.prepare_call_hierarchy(
            str(Path(file_path).resolve()),
            line - 1,
            character - 1,
        )

        if not items:
            return {}

        item = items[0]
        result = {
            "name": item.name,
            "kind": item.kind.name,
            "file": item.file_path,
            "line": item.range.start.line + 1,
        }

        if direction in ("incoming", "both"):
            incoming = await client.incoming_calls(item)
            result["callers"] = [
                {
                    "name": c.from_item.name,
                    "kind": c.from_item.kind.name,
                    "file": c.from_item.file_path,
                    "line": c.from_item.range.start.line + 1,
                }
                for c in incoming
            ]

        if direction in ("outgoing", "both"):
            outgoing = await client.outgoing_calls(item)
            result["callees"] = [
                {
                    "name": c.to_item.name,
                    "kind": c.to_item.kind.name,
                    "file": c.to_item.file_path,
                    "line": c.to_item.range.start.line + 1,
                }
                for c in outgoing
            ]

        return result

    # Convenience methods for common use cases

    async def find_symbol_definition(self, symbol_name: str) -> list[Location]:
        """Find definition of a symbol by name.

        Args:
            symbol_name: Name of the symbol

        Returns:
            List of definition locations
        """
        # Search workspace symbols
        all_symbols = []
        for client in self._clients.values():
            symbols = await client.workspace_symbols(symbol_name)
            all_symbols.extend(symbols)

        # Filter exact matches
        exact = [s for s in all_symbols if s.name == symbol_name]
        if exact:
            return [s.location for s in exact]

        # Return all matches if no exact match
        return [s.location for s in all_symbols]

    async def get_type_info(self, file_path: str, line: int, character: int) -> str | None:
        """Get type information for symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            character: Character offset (1-indexed)

        Returns:
            Type information string or None
        """
        hover = await self.hover(file_path, line, character)
        if hover and not hover.is_empty:
            return hover.contents
        return None

    async def get_signature(self, file_path: str, line: int, character: int) -> str | None:
        """Get function signature at position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            character: Character offset (1-indexed)

        Returns:
            Function signature or None
        """
        hover = await self.hover(file_path, line, character)
        if hover and not hover.is_empty:
            # Extract first line which usually contains the signature
            lines = hover.contents.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("---"):
                    # Remove markdown code fence markers
                    if line.startswith("```"):
                        continue
                    return line
        return None

    # Diagnostics methods

    async def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Get diagnostics for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of diagnostics
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return []

        return await client.request_diagnostics(str(Path(file_path).resolve()))

    async def check_file(self, file_path: str) -> dict:
        """Check a file for errors and warnings.

        This is useful after editing a file to see if there are any issues.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with diagnostics summary and details
        """
        diagnostics = await self.get_diagnostics(file_path)

        if not diagnostics:
            return {
                "file": file_path,
                "has_errors": False,
                "has_warnings": False,
                "error_count": 0,
                "warning_count": 0,
                "hint_count": 0,
                "info_count": 0,
                "diagnostics": [],
            }

        # Categorize diagnostics
        errors = [d for d in diagnostics if d.severity == 1]
        warnings = [d for d in diagnostics if d.severity == 2]
        info = [d for d in diagnostics if d.severity == 3]
        hints = [d for d in diagnostics if d.severity == 4]

        return {
            "file": file_path,
            "has_errors": len(errors) > 0,
            "has_warnings": len(warnings) > 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(info),
            "hint_count": len(hints),
            "diagnostics": [
                {
                    "severity": d.severity_name,
                    "line": d.range.start.line + 1,
                    "character": d.range.start.character + 1,
                    "message": d.message,
                    "source": d.source or "",
                    "code": d.code or "",
                }
                for d in diagnostics
            ],
        }

    async def check_files(self, file_paths: list[str]) -> dict:
        """Check multiple files for errors and warnings.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary with summary and per-file details
        """
        results = {}
        total_errors = 0
        total_warnings = 0

        for file_path in file_paths:
            result = await self.check_file(file_path)
            results[file_path] = result
            total_errors += result["error_count"]
            total_warnings += result["warning_count"]

        return {
            "files_checked": len(file_paths),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "has_errors": total_errors > 0,
            "has_warnings": total_warnings > 0,
            "details": results,
        }

    # Status methods

    def get_active_servers(self) -> list[str]:
        """Get list of active server languages.

        Returns:
            List of language identifiers
        """
        return [lang for lang, client in self._clients.items() if client.is_running]

    def get_server_capabilities(self, language: str) -> dict:
        """Get capabilities of a server.

        Args:
            language: Language identifier

        Returns:
            Capabilities dictionary
        """
        if language not in self._clients:
            return {}

        client = self._clients[language]
        if not client.capabilities:
            return {}

        caps = client.capabilities
        return {
            "definition": caps.supports_definition(),
            "references": caps.supports_references(),
            "hover": caps.supports_hover(),
            "documentSymbol": caps.supports_document_symbol(),
            "workspaceSymbol": caps.supports_workspace_symbol(),
            "callHierarchy": caps.supports_call_hierarchy(),
        }


# Singleton per workspace
_instances: dict[str, LSPOperations] = {}


def get_lsp_operations(workspace_root: str | Path) -> LSPOperations:
    """Get or create LSP operations for a workspace.

    Args:
        workspace_root: Root directory of the workspace

    Returns:
        LSPOperations instance
    """
    root = str(Path(workspace_root).resolve())
    if root not in _instances:
        _instances[root] = LSPOperations(root)
    return _instances[root]


# Synchronous wrappers for easy use

def goto_definition_sync(
    file_path: str, line: int, character: int, workspace_root: str | None = None
) -> list[Location]:
    """Synchronous wrapper for goto_definition.

    Args:
        file_path: Path to the file
        line: Line number (1-indexed)
        character: Character offset (1-indexed)
        workspace_root: Workspace root (defaults to file's directory)

    Returns:
        List of definition locations
    """
    if workspace_root is None:
        workspace_root = str(Path(file_path).parent)

    ops = get_lsp_operations(workspace_root)
    return asyncio.run(ops.goto_definition(file_path, line, character))


def find_references_sync(
    file_path: str, line: int, character: int, workspace_root: str | None = None
) -> list[Location]:
    """Synchronous wrapper for find_references.

    Args:
        file_path: Path to the file
        line: Line number (1-indexed)
        character: Character offset (1-indexed)
        workspace_root: Workspace root (defaults to file's directory)

    Returns:
        List of reference locations
    """
    if workspace_root is None:
        workspace_root = str(Path(file_path).parent)

    ops = get_lsp_operations(workspace_root)
    return asyncio.run(ops.find_references(file_path, line, character))


def hover_sync(
    file_path: str, line: int, character: int, workspace_root: str | None = None
) -> HoverInfo | None:
    """Synchronous wrapper for hover.

    Args:
        file_path: Path to the file
        line: Line number (1-indexed)
        character: Character offset (1-indexed)
        workspace_root: Workspace root (defaults to file's directory)

    Returns:
        Hover information or None
    """
    if workspace_root is None:
        workspace_root = str(Path(file_path).parent)

    ops = get_lsp_operations(workspace_root)
    return asyncio.run(ops.hover(file_path, line, character))


def check_file_sync(
    file_path: str, workspace_root: str | None = None
) -> dict:
    """Synchronous wrapper for check_file.

    Check a file for errors and warnings using LSP diagnostics.

    Args:
        file_path: Path to the file
        workspace_root: Workspace root (defaults to file's directory)

    Returns:
        Dictionary with diagnostics summary and details
    """
    if workspace_root is None:
        workspace_root = str(Path(file_path).parent)

    ops = get_lsp_operations(workspace_root)
    return asyncio.run(ops.check_file(file_path))


def get_diagnostics_sync(
    file_path: str, workspace_root: str | None = None
) -> list[Diagnostic]:
    """Synchronous wrapper for get_diagnostics.

    Get LSP diagnostics for a file.

    Args:
        file_path: Path to the file
        workspace_root: Workspace root (defaults to file's directory)

    Returns:
        List of diagnostics
    """
    if workspace_root is None:
        workspace_root = str(Path(file_path).parent)

    ops = get_lsp_operations(workspace_root)
    return asyncio.run(ops.get_diagnostics(file_path))
