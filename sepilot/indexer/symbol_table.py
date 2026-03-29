"""Symbol Table - Fast symbol lookup and management."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator

from .models import IndexedSymbol

logger = logging.getLogger(__name__)


class SymbolTable:
    """In-memory symbol table for fast lookups.

    Provides O(1) lookups by name and efficient pattern matching.
    """

    def __init__(self):
        """Initialize empty symbol table."""
        # Primary index: name -> list of symbols
        self._by_name: dict[str, list[IndexedSymbol]] = defaultdict(list)

        # Secondary indexes
        self._by_file: dict[str, list[IndexedSymbol]] = defaultdict(list)
        self._by_kind: dict[str, list[IndexedSymbol]] = defaultdict(list)
        self._by_qualified_name: dict[str, IndexedSymbol] = {}

        # Statistics
        self._total_count = 0

    def add(self, symbol: IndexedSymbol) -> None:
        """Add a symbol to the table.

        Args:
            symbol: The symbol to add
        """
        self._by_name[symbol.name].append(symbol)
        self._by_file[symbol.file_path].append(symbol)
        self._by_kind[symbol.kind].append(symbol)

        # Qualified name should be unique per file
        key = f"{symbol.file_path}:{symbol.qualified_name}"
        self._by_qualified_name[key] = symbol

        self._total_count += 1

    def add_many(self, symbols: list[IndexedSymbol]) -> None:
        """Add multiple symbols efficiently.

        Args:
            symbols: List of symbols to add
        """
        for symbol in symbols:
            self.add(symbol)

    def remove_file(self, file_path: str) -> int:
        """Remove all symbols from a file.

        Args:
            file_path: Path to the file

        Returns:
            Number of symbols removed
        """
        symbols = self._by_file.get(file_path, [])
        count = len(symbols)

        for symbol in symbols:
            # Remove from name index
            if symbol.name in self._by_name:
                self._by_name[symbol.name] = [
                    s for s in self._by_name[symbol.name] if s.file_path != file_path
                ]
                if not self._by_name[symbol.name]:
                    del self._by_name[symbol.name]

            # Remove from kind index
            if symbol.kind in self._by_kind:
                self._by_kind[symbol.kind] = [
                    s for s in self._by_kind[symbol.kind] if s.file_path != file_path
                ]
                if not self._by_kind[symbol.kind]:
                    del self._by_kind[symbol.kind]

            # Remove from qualified name index
            key = f"{file_path}:{symbol.qualified_name}"
            if key in self._by_qualified_name:
                del self._by_qualified_name[key]

        # Remove from file index
        if file_path in self._by_file:
            del self._by_file[file_path]

        self._total_count -= count
        return count

    def get_by_name(self, name: str) -> list[IndexedSymbol]:
        """Get all symbols with a given name.

        Args:
            name: Symbol name to search for

        Returns:
            List of matching symbols
        """
        return self._by_name.get(name, [])

    def get_by_file(self, file_path: str) -> list[IndexedSymbol]:
        """Get all symbols in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of symbols in the file
        """
        return self._by_file.get(file_path, [])

    def get_by_kind(self, kind: str) -> list[IndexedSymbol]:
        """Get all symbols of a given kind.

        Args:
            kind: Symbol kind (function, class, etc.)

        Returns:
            List of matching symbols
        """
        return self._by_kind.get(kind, [])

    def get_by_qualified_name(
        self, qualified_name: str, file_path: str | None = None
    ) -> IndexedSymbol | None:
        """Get a symbol by its qualified name.

        Args:
            qualified_name: Fully qualified name (e.g., ClassName.method)
            file_path: Optional file path for disambiguation

        Returns:
            The symbol or None if not found
        """
        if file_path:
            key = f"{file_path}:{qualified_name}"
            return self._by_qualified_name.get(key)

        # Search all files
        for key, symbol in self._by_qualified_name.items():
            if key.endswith(f":{qualified_name}"):
                return symbol
        return None

    def search(
        self,
        query: str,
        kind: str | None = None,
        file_path: str | None = None,
        limit: int = 100,
    ) -> list[IndexedSymbol]:
        """Search for symbols matching a query.

        Args:
            query: Search query (supports partial matching)
            kind: Optional filter by symbol kind
            file_path: Optional filter by file
            limit: Maximum results to return

        Returns:
            List of matching symbols
        """
        results: list[IndexedSymbol] = []
        query_lower = query.lower()

        # Determine which index to search
        if file_path:
            candidates = self._by_file.get(file_path, [])
        elif kind:
            candidates = self._by_kind.get(kind, [])
        else:
            candidates = self._all_symbols()

        for symbol in candidates:
            if symbol.matches(query):
                if kind and symbol.kind != kind:
                    continue
                if file_path and symbol.file_path != file_path:
                    continue
                results.append(symbol)
                if len(results) >= limit:
                    break

        # Sort by relevance (exact matches first, then by name length)
        results.sort(
            key=lambda s: (
                0 if s.name.lower() == query_lower else 1,
                len(s.name),
                s.name,
            )
        )

        return results

    def find_definition(self, name: str, context_file: str | None = None) -> IndexedSymbol | None:
        """Find the definition of a symbol.

        Prioritizes symbols in the same file or imported modules.

        Args:
            name: Symbol name to find
            context_file: File where the reference occurs

        Returns:
            The definition symbol or None
        """
        candidates = self.get_by_name(name)
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Prioritize by context
        if context_file:
            # First, check same file
            for symbol in candidates:
                if symbol.file_path == context_file:
                    return symbol

            # TODO: Check imported modules using dependency graph

        # Return first public symbol
        for symbol in candidates:
            if symbol.visibility == "public":
                return symbol

        return candidates[0]

    def get_all_names(self) -> list[str]:
        """Get all unique symbol names.

        Returns:
            List of symbol names
        """
        return list(self._by_name.keys())

    def get_all_files(self) -> list[str]:
        """Get all indexed files.

        Returns:
            List of file paths
        """
        return list(self._by_file.keys())

    def get_statistics(self) -> dict[str, int]:
        """Get symbol table statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_symbols": self._total_count,
            "unique_names": len(self._by_name),
            "indexed_files": len(self._by_file),
            "by_kind": {k: len(v) for k, v in self._by_kind.items()},
        }

    def clear(self) -> None:
        """Clear all symbols from the table."""
        self._by_name.clear()
        self._by_file.clear()
        self._by_kind.clear()
        self._by_qualified_name.clear()
        self._total_count = 0

    def _all_symbols(self) -> Iterator[IndexedSymbol]:
        """Iterate over all symbols."""
        for symbols in self._by_file.values():
            yield from symbols

    def __len__(self) -> int:
        """Get total symbol count."""
        return self._total_count

    def __contains__(self, name: str) -> bool:
        """Check if a symbol name exists."""
        return name in self._by_name
