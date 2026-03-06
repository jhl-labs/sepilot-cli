"""Call Graph - Function call relationship tracking."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterator

from .models import CallEdge

logger = logging.getLogger(__name__)


class CallGraph:
    """Graph of function call relationships.

    Tracks which functions call which other functions.
    """

    def __init__(self):
        """Initialize empty call graph."""
        # caller -> list of call edges (outgoing calls)
        self._outgoing: dict[str, list[CallEdge]] = defaultdict(list)

        # callee -> list of call edges (incoming calls)
        self._incoming: dict[str, list[CallEdge]] = defaultdict(list)

        # file -> list of call edges
        self._by_file: dict[str, list[CallEdge]] = defaultdict(list)

    def add_call(self, edge: CallEdge) -> None:
        """Add a call edge to the graph.

        Args:
            edge: The call edge to add
        """
        caller_key = f"{edge.caller_file}:{edge.caller_symbol}"
        self._outgoing[caller_key].append(edge)
        self._incoming[edge.callee_symbol].append(edge)
        self._by_file[edge.caller_file].append(edge)

    def add_calls(self, edges: list[CallEdge]) -> None:
        """Add multiple call edges.

        Args:
            edges: List of call edges
        """
        for edge in edges:
            self.add_call(edge)

    def remove_file(self, file_path: str) -> int:
        """Remove all call edges from a file.

        Args:
            file_path: Path to the file

        Returns:
            Number of edges removed
        """
        edges = self._by_file.get(file_path, [])
        count = len(edges)

        for edge in edges:
            caller_key = f"{edge.caller_file}:{edge.caller_symbol}"

            # Remove from outgoing
            if caller_key in self._outgoing:
                self._outgoing[caller_key] = [
                    e for e in self._outgoing[caller_key] if e.caller_file != file_path
                ]
                if not self._outgoing[caller_key]:
                    del self._outgoing[caller_key]

            # Remove from incoming
            if edge.callee_symbol in self._incoming:
                self._incoming[edge.callee_symbol] = [
                    e for e in self._incoming[edge.callee_symbol] if e.caller_file != file_path
                ]
                if not self._incoming[edge.callee_symbol]:
                    del self._incoming[edge.callee_symbol]

        # Remove from file index
        if file_path in self._by_file:
            del self._by_file[file_path]

        return count

    def get_outgoing_calls(self, file_path: str, symbol: str) -> list[CallEdge]:
        """Get all functions called by a symbol.

        Args:
            file_path: File containing the caller
            symbol: Caller symbol name

        Returns:
            List of call edges (outgoing calls)
        """
        key = f"{file_path}:{symbol}"
        return self._outgoing.get(key, [])

    def get_incoming_calls(self, symbol: str) -> list[CallEdge]:
        """Get all callers of a symbol.

        Args:
            symbol: Callee symbol name

        Returns:
            List of call edges (incoming calls)
        """
        return self._incoming.get(symbol, [])

    def get_calls_in_file(self, file_path: str) -> list[CallEdge]:
        """Get all call edges in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of call edges
        """
        return self._by_file.get(file_path, [])

    def get_call_hierarchy(
        self, file_path: str, symbol: str, direction: str = "outgoing", max_depth: int = 5
    ) -> dict:
        """Get the call hierarchy for a symbol.

        Args:
            file_path: File containing the symbol
            symbol: Symbol name
            direction: "outgoing" for calls made, "incoming" for callers
            max_depth: Maximum depth to traverse

        Returns:
            Nested dictionary representing the call hierarchy
        """
        visited = set()

        def traverse(fp: str, sym: str, depth: int) -> dict | None:
            if depth > max_depth:
                return None

            key = f"{fp}:{sym}"
            if key in visited:
                return {"symbol": sym, "file": fp, "recursive": True}

            visited.add(key)

            result = {
                "symbol": sym,
                "file": fp,
                "children": [],
            }

            if direction == "outgoing":
                edges = self.get_outgoing_calls(fp, sym)
                for edge in edges:
                    child_file = edge.callee_file or fp
                    child = traverse(child_file, edge.callee_symbol, depth + 1)
                    if child:
                        child["line"] = edge.line_number
                        result["children"].append(child)
            else:  # incoming
                edges = self.get_incoming_calls(sym)
                for edge in edges:
                    child = traverse(edge.caller_file, edge.caller_symbol, depth + 1)
                    if child:
                        child["line"] = edge.line_number
                        result["children"].append(child)

            return result

        return traverse(file_path, symbol, 0)

    def get_callers(self, symbol: str) -> list[tuple[str, str]]:
        """Get all unique callers of a symbol.

        Args:
            symbol: Callee symbol name

        Returns:
            List of (file_path, caller_symbol) tuples
        """
        edges = self.get_incoming_calls(symbol)
        return list({(e.caller_file, e.caller_symbol) for e in edges})

    def get_callees(self, file_path: str, symbol: str) -> list[str]:
        """Get all unique symbols called by a function.

        Args:
            file_path: File containing the caller
            symbol: Caller symbol name

        Returns:
            List of called symbol names
        """
        edges = self.get_outgoing_calls(file_path, symbol)
        return list({e.callee_symbol for e in edges})

    def find_paths(
        self,
        from_file: str,
        from_symbol: str,
        to_symbol: str,
        max_length: int = 10,
    ) -> list[list[str]]:
        """Find all call paths between two symbols.

        Args:
            from_file: File containing the source symbol
            from_symbol: Source symbol name
            to_symbol: Target symbol name
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of symbol names)
        """
        paths = []
        visited = set()

        def dfs(file_path: str, symbol: str, path: list[str]) -> None:
            if len(path) > max_length:
                return

            if symbol == to_symbol:
                paths.append(path.copy())
                return

            key = f"{file_path}:{symbol}"
            if key in visited:
                return

            visited.add(key)
            path.append(symbol)

            for edge in self.get_outgoing_calls(file_path, symbol):
                callee_file = edge.callee_file or file_path
                dfs(callee_file, edge.callee_symbol, path)

            path.pop()
            visited.discard(key)

        dfs(from_file, from_symbol, [])
        return paths

    def get_hot_spots(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most called symbols.

        Args:
            limit: Maximum number of results

        Returns:
            List of (symbol, call_count) tuples, sorted by count
        """
        counts = {sym: len(edges) for sym, edges in self._incoming.items()}
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_counts[:limit]

    def get_statistics(self) -> dict[str, int]:
        """Get call graph statistics.

        Returns:
            Dictionary with statistics
        """
        total_edges = sum(len(edges) for edges in self._outgoing.values())
        unique_callers = len(self._outgoing)
        unique_callees = len(self._incoming)

        return {
            "total_call_edges": total_edges,
            "unique_callers": unique_callers,
            "unique_callees": unique_callees,
            "files_with_calls": len(self._by_file),
        }

    def clear(self) -> None:
        """Clear all call edges."""
        self._outgoing.clear()
        self._incoming.clear()
        self._by_file.clear()

    def __len__(self) -> int:
        """Get total number of call edges."""
        return sum(len(edges) for edges in self._outgoing.values())

    def __iter__(self) -> Iterator[CallEdge]:
        """Iterate over all call edges."""
        for edges in self._outgoing.values():
            yield from edges
