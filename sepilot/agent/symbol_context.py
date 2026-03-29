"""Symbol Context - Gather context around code symbols."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sepilot.indexer import ProjectIndexer
    from sepilot.lsp import LSPOperations

logger = logging.getLogger(__name__)


@dataclass
class SymbolContext:
    """Context information about a code symbol."""

    name: str
    kind: str  # function, class, method, variable
    file_path: str
    line_start: int
    line_end: int
    signature: str | None = None
    docstring: str | None = None
    source_code: str | None = None
    type_info: str | None = None
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    parent: str | None = None  # For methods, the class name
    metadata: dict[str, Any] = field(default_factory=dict)

    def format(self, include_source: bool = True) -> str:
        """Format symbol context for LLM consumption."""
        lines = [f"## {self.kind.title()}: {self.name}"]
        lines.append(f"Location: {self.file_path}:{self.line_start}")

        if self.parent:
            lines.append(f"Parent: {self.parent}")

        if self.signature:
            lines.append(f"Signature: {self.signature}")

        if self.type_info:
            lines.append(f"Type: {self.type_info}")

        if self.docstring:
            lines.append(f"\nDocstring:\n{self.docstring}")

        if include_source and self.source_code:
            lines.append(f"\nSource:\n```\n{self.source_code}\n```")

        if self.callers:
            lines.append(f"\nCallers: {', '.join(self.callers[:5])}")
            if len(self.callers) > 5:
                lines.append(f"  (+{len(self.callers) - 5} more)")

        if self.callees:
            lines.append(f"\nCalls: {', '.join(self.callees[:5])}")
            if len(self.callees) > 5:
                lines.append(f"  (+{len(self.callees) - 5} more)")

        return "\n".join(lines)


class SymbolContextGatherer:
    """Gather rich context for code symbols.

    Combines information from:
    - Project indexer (symbols, call graph)
    - LSP (type info, hover)
    - Source files
    """

    def __init__(
        self,
        project_root: str | Path,
        indexer: ProjectIndexer | None = None,
        lsp: LSPOperations | None = None,
    ):
        """Initialize the gatherer.

        Args:
            project_root: Project root directory
            indexer: Optional project indexer
            lsp: Optional LSP operations
        """
        self.project_root = Path(project_root).resolve()
        self._indexer = indexer
        self._lsp = lsp
        self._file_cache: dict[str, list[str]] = {}

    def gather(
        self,
        symbol_name: str,
        file_path: str | None = None,
        include_source: bool = True,
        include_callers: bool = True,
        max_source_lines: int = 50,
    ) -> SymbolContext | None:
        """Gather context for a symbol.

        Args:
            symbol_name: Name of the symbol
            file_path: Optional file path for disambiguation
            include_source: Include source code
            include_callers: Include caller information
            max_source_lines: Maximum lines of source to include

        Returns:
            SymbolContext or None if not found
        """
        # Find symbol in indexer
        symbol = self._find_symbol(symbol_name, file_path)
        if not symbol:
            return None

        context = SymbolContext(
            name=symbol.name,
            kind=symbol.kind,
            file_path=symbol.file_path,
            line_start=symbol.line_start,
            line_end=symbol.line_end,
            signature=symbol.signature,
            docstring=symbol.docstring,
            parent=symbol.parent,
        )

        # Get source code
        if include_source:
            context.source_code = self._get_source(
                symbol.file_path,
                symbol.line_start,
                min(symbol.line_end, symbol.line_start + max_source_lines),
            )

        # Get type info from LSP
        if self._lsp:
            context.type_info = self._get_type_info(
                symbol.file_path, symbol.line_start
            )

        # Get call graph info
        if include_callers and self._indexer:
            context.callers = self._get_callers(symbol_name)
            context.callees = self._get_callees(symbol.file_path, symbol_name)

        return context

    def gather_multiple(
        self,
        symbols: list[str],
        max_per_symbol: int = 30,
    ) -> list[SymbolContext]:
        """Gather context for multiple symbols.

        Args:
            symbols: List of symbol names
            max_per_symbol: Maximum source lines per symbol

        Returns:
            List of SymbolContext
        """
        results = []
        for symbol in symbols:
            ctx = self.gather(symbol, max_source_lines=max_per_symbol)
            if ctx:
                results.append(ctx)
        return results

    def gather_for_file(
        self, file_path: str, max_symbols: int = 20
    ) -> list[SymbolContext]:
        """Gather context for all symbols in a file.

        Args:
            file_path: Path to the file
            max_symbols: Maximum symbols to return

        Returns:
            List of SymbolContext
        """
        if not self._indexer:
            return []

        results = []
        try:
            symbols = self._indexer.get_file_symbols(file_path)
            for symbol in symbols[:max_symbols]:
                ctx = self.gather(
                    symbol.name,
                    file_path=file_path,
                    include_source=False,  # Skip source to save tokens
                    include_callers=False,
                )
                if ctx:
                    results.append(ctx)
        except Exception as e:
            logger.debug(f"Error gathering file symbols: {e}")

        return results

    def _find_symbol(self, name: str, file_path: str | None):
        """Find symbol in indexer."""
        if not self._indexer:
            return None

        try:
            symbols = self._indexer.get_symbol(name)
            if not symbols:
                return None

            if file_path:
                for s in symbols:
                    if s.file_path == file_path:
                        return s

            return symbols[0]
        except Exception as e:
            logger.debug(f"Error finding symbol: {e}")
            return None

    def _get_source(self, file_path: str, start: int, end: int) -> str | None:
        """Get source code lines."""
        try:
            lines = self._read_file(file_path)
            if not lines:
                return None

            # Convert to 0-indexed
            start_idx = max(0, start - 1)
            end_idx = min(len(lines), end)

            return "\n".join(lines[start_idx:end_idx])
        except Exception as e:
            logger.debug(f"Error reading source: {e}")
            return None

    def _read_file(self, file_path: str) -> list[str]:
        """Read file with caching."""
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        try:
            content = Path(file_path).read_text(encoding="utf-8")
            lines = content.split("\n")
            self._file_cache[file_path] = lines
            return lines
        except Exception:
            return []

    def _get_type_info(self, file_path: str, line: int) -> str | None:
        """Get type information from LSP."""
        if not self._lsp:
            return None

        try:
            import asyncio

            hover = asyncio.run(self._lsp.hover(file_path, line, 0))
            if hover and not hover.is_empty:
                return hover.contents
        except Exception as e:
            logger.debug(f"Error getting type info: {e}")

        return None

    def _get_callers(self, symbol: str) -> list[str]:
        """Get callers of a symbol."""
        if not self._indexer:
            return []

        try:
            callers = self._indexer.get_callers(symbol)
            return [c[1] for c in callers]  # Return just the caller names
        except Exception:
            return []

    def _get_callees(self, file_path: str, symbol: str) -> list[str]:
        """Get symbols called by a function."""
        if not self._indexer:
            return []

        try:
            return self._indexer.get_callees(file_path, symbol)
        except Exception:
            return []

    def clear_cache(self) -> None:
        """Clear file cache."""
        self._file_cache.clear()


def gather_symbol_context(
    project_root: str | Path,
    symbol_name: str,
    file_path: str | None = None,
) -> SymbolContext | None:
    """Convenience function to gather symbol context.

    Args:
        project_root: Project root directory
        symbol_name: Name of the symbol
        file_path: Optional file path

    Returns:
        SymbolContext or None
    """
    gatherer = SymbolContextGatherer(project_root)
    return gatherer.gather(symbol_name, file_path)
