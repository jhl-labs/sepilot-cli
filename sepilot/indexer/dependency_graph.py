"""Dependency Graph - Import/export relationship tracking."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path

from .models import Dependency

logger = logging.getLogger(__name__)


class DependencyGraph:
    """Graph of import/export relationships between files.

    Tracks which files depend on which modules and provides
    traversal methods for impact analysis.
    """

    def __init__(self):
        """Initialize empty dependency graph."""
        # file -> list of dependencies (outgoing edges)
        self._dependencies: dict[str, list[Dependency]] = defaultdict(list)

        # module/file -> list of files that import it (incoming edges)
        self._dependents: dict[str, set[str]] = defaultdict(set)

        # Resolved module -> file mapping
        self._module_to_file: dict[str, str] = {}

    def add_dependency(self, dependency: Dependency) -> None:
        """Add a dependency relationship.

        Args:
            dependency: The dependency to add
        """
        self._dependencies[dependency.from_file].append(dependency)
        self._dependents[dependency.to_module].add(dependency.from_file)

    def add_dependencies(self, file_path: str, dependencies: list[Dependency]) -> None:
        """Add multiple dependencies for a file.

        Args:
            file_path: Source file path
            dependencies: List of dependencies
        """
        for dep in dependencies:
            self.add_dependency(dep)

    def remove_file(self, file_path: str) -> None:
        """Remove all dependencies from a file.

        Args:
            file_path: Path to the file
        """
        if file_path in self._dependencies:
            # Remove from dependents index
            for dep in self._dependencies[file_path]:
                if dep.to_module in self._dependents:
                    self._dependents[dep.to_module].discard(file_path)
                    if not self._dependents[dep.to_module]:
                        del self._dependents[dep.to_module]

            # Remove dependencies
            del self._dependencies[file_path]

    def get_dependencies(self, file_path: str) -> list[Dependency]:
        """Get all dependencies of a file.

        Args:
            file_path: Path to the file

        Returns:
            List of dependencies
        """
        return self._dependencies.get(file_path, [])

    def get_dependents(self, module: str) -> set[str]:
        """Get all files that depend on a module.

        Args:
            module: Module name or file path

        Returns:
            Set of file paths that import this module
        """
        return self._dependents.get(module, set())

    def get_imported_symbols(self, file_path: str, from_module: str) -> list[str]:
        """Get symbols imported from a specific module.

        Args:
            file_path: File that imports
            from_module: Module being imported

        Returns:
            List of imported symbol names
        """
        deps = self.get_dependencies(file_path)
        for dep in deps:
            if dep.to_module == from_module:
                return dep.symbols
        return []

    def get_all_imported_modules(self, file_path: str) -> list[str]:
        """Get all modules imported by a file.

        Args:
            file_path: Path to the file

        Returns:
            List of module names
        """
        deps = self.get_dependencies(file_path)
        return [dep.to_module for dep in deps]

    def get_external_dependencies(self, file_path: str) -> list[Dependency]:
        """Get only external (not in project) dependencies.

        Args:
            file_path: Path to the file

        Returns:
            List of external dependencies
        """
        return [dep for dep in self.get_dependencies(file_path) if dep.is_external]

    def get_internal_dependencies(self, file_path: str) -> list[Dependency]:
        """Get only internal (in project) dependencies.

        Args:
            file_path: Path to the file

        Returns:
            List of internal dependencies
        """
        return [dep for dep in self.get_dependencies(file_path) if not dep.is_external]

    def resolve_module(self, module: str, project_root: str) -> str | None:
        """Resolve a module name to a file path.

        Args:
            module: Module name (e.g., 'sepilot.tools.code_analysis')
            project_root: Project root directory

        Returns:
            Resolved file path or None
        """
        # Check cache
        if module in self._module_to_file:
            return self._module_to_file[module]

        # Try to resolve
        root = Path(project_root)

        # Python-style module resolution
        module_path = module.replace(".", "/")
        candidates = [
            root / f"{module_path}.py",
            root / module_path / "__init__.py",
        ]

        for candidate in candidates:
            if candidate.exists():
                resolved = str(candidate)
                self._module_to_file[module] = resolved
                return resolved

        return None

    def get_transitive_dependencies(
        self, file_path: str, max_depth: int = 10
    ) -> set[str]:
        """Get all transitive dependencies of a file.

        Args:
            file_path: Starting file
            max_depth: Maximum traversal depth

        Returns:
            Set of all transitively imported modules
        """
        visited = set()
        to_visit = deque([(file_path, 0)])

        while to_visit:
            current, depth = to_visit.popleft()
            if current in visited or depth > max_depth:
                continue

            visited.add(current)

            for dep in self.get_dependencies(current):
                if dep.to_module not in visited:
                    # Try to resolve to a file for further traversal
                    resolved = self._module_to_file.get(dep.to_module)
                    if resolved:
                        to_visit.append((resolved, depth + 1))
                    visited.add(dep.to_module)

        visited.discard(file_path)  # Remove starting file
        return visited

    def get_transitive_dependents(
        self, module: str, max_depth: int = 10
    ) -> set[str]:
        """Get all files that transitively depend on a module.

        Args:
            module: Module name or file path
            max_depth: Maximum traversal depth

        Returns:
            Set of all files that transitively import this module
        """
        visited = set()
        to_visit = deque([(module, 0)])

        while to_visit:
            current, depth = to_visit.popleft()
            if current in visited or depth > max_depth:
                continue

            visited.add(current)

            for dependent in self.get_dependents(current):
                if dependent not in visited:
                    to_visit.append((dependent, depth + 1))

        visited.discard(module)  # Remove starting module
        return visited

    def detect_cycles(self) -> list[list[str]]:
        """Detect circular dependencies.

        Returns:
            List of cycles (each cycle is a list of files/modules)
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in self.get_dependencies(node):
                next_node = dep.to_module
                if next_node not in visited:
                    dfs(next_node, path)
                elif next_node in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(next_node)
                    cycle = path[cycle_start:] + [next_node]
                    cycles.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for file_path in self._dependencies:
            if file_path not in visited:
                dfs(file_path, [])

        return cycles

    def get_statistics(self) -> dict[str, int]:
        """Get dependency graph statistics.

        Returns:
            Dictionary with statistics
        """
        total_deps = sum(len(deps) for deps in self._dependencies.values())
        external_deps = sum(
            1 for deps in self._dependencies.values() for d in deps if d.is_external
        )

        return {
            "files_with_dependencies": len(self._dependencies),
            "total_dependencies": total_deps,
            "external_dependencies": external_deps,
            "internal_dependencies": total_deps - external_deps,
            "modules_imported": len(self._dependents),
        }

    def clear(self) -> None:
        """Clear all dependencies."""
        self._dependencies.clear()
        self._dependents.clear()
        self._module_to_file.clear()

    def __len__(self) -> int:
        """Get total number of dependencies."""
        return sum(len(deps) for deps in self._dependencies.values())

    def __iter__(self) -> Iterator[Dependency]:
        """Iterate over all dependencies."""
        for deps in self._dependencies.values():
            yield from deps
