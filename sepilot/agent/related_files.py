"""Related Files Finder - Discover related files through dependencies and references."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sepilot.indexer import ProjectIndexer
    from sepilot.lsp import LSPOperations

logger = logging.getLogger(__name__)


@dataclass
class RelatedFile:
    """A file related to the current context."""

    file_path: str
    relation_type: str  # import, reference, test, sibling
    relevance_score: float
    reason: str

    def __lt__(self, other: RelatedFile) -> bool:
        return self.relevance_score < other.relevance_score


class RelatedFileFinder:
    """Find files related to given files or symbols.

    Uses multiple strategies:
    - Dependency graph (imports/exports)
    - LSP references
    - Test file discovery
    - Sibling files (same directory)
    - Similar naming patterns
    """

    def __init__(
        self,
        project_root: str | Path,
        indexer: ProjectIndexer | None = None,
        lsp: LSPOperations | None = None,
    ):
        """Initialize the finder.

        Args:
            project_root: Project root directory
            indexer: Optional project indexer for dependencies
            lsp: Optional LSP operations for references
        """
        self.project_root = Path(project_root).resolve()
        self._indexer = indexer
        self._lsp = lsp

    def find_related(
        self,
        files: list[str],
        symbols: list[str] | None = None,
        max_results: int = 10,
    ) -> list[RelatedFile]:
        """Find files related to the given files and symbols.

        Args:
            files: List of file paths
            symbols: Optional list of symbol names
            max_results: Maximum number of results

        Returns:
            List of RelatedFile sorted by relevance
        """
        related: dict[str, RelatedFile] = {}

        for file_path in files:
            # Find imports/dependencies
            deps = self._find_dependencies(file_path)
            for rel in deps:
                self._merge_related(related, rel)

            # Find dependents (files that import this one)
            dependents = self._find_dependents(file_path)
            for rel in dependents:
                self._merge_related(related, rel)

            # Find test files
            test_files = self._find_test_files(file_path)
            for rel in test_files:
                self._merge_related(related, rel)

            # Find sibling files
            siblings = self._find_siblings(file_path)
            for rel in siblings:
                self._merge_related(related, rel)

        # Find references to symbols
        if symbols:
            for symbol in symbols:
                refs = self._find_symbol_references(symbol)
                for rel in refs:
                    self._merge_related(related, rel)

        # Remove input files from results
        for f in files:
            related.pop(f, None)

        # Sort and limit results
        results = sorted(related.values(), reverse=True)
        return results[:max_results]

    def _merge_related(
        self, related: dict[str, RelatedFile], new: RelatedFile
    ) -> None:
        """Merge a related file into the results."""
        if new.file_path in related:
            existing = related[new.file_path]
            # Boost score for multiple relations
            existing.relevance_score = min(
                1.0, existing.relevance_score + new.relevance_score * 0.5
            )
            if new.relation_type not in existing.reason:
                existing.reason += f", {new.reason}"
        else:
            related[new.file_path] = new

    def _find_dependencies(self, file_path: str) -> list[RelatedFile]:
        """Find files imported by the given file."""
        related = []

        if not self._indexer:
            return related

        try:
            deps = self._indexer.get_dependencies(file_path)
            for dep in deps:
                # Try to resolve module to file
                resolved = self._resolve_module(dep.to_module)
                if resolved:
                    related.append(
                        RelatedFile(
                            file_path=resolved,
                            relation_type="import",
                            relevance_score=0.8,
                            reason=f"imported by {Path(file_path).name}",
                        )
                    )
        except Exception as e:
            logger.debug(f"Error finding dependencies: {e}")

        return related

    def _find_dependents(self, file_path: str) -> list[RelatedFile]:
        """Find files that import the given file."""
        related = []

        if not self._indexer:
            return related

        try:
            # Get module name for this file
            module = self._file_to_module(file_path)
            if module:
                dependents = self._indexer.get_dependents(module)
                for dep_file in dependents:
                    related.append(
                        RelatedFile(
                            file_path=dep_file,
                            relation_type="dependent",
                            relevance_score=0.7,
                            reason=f"imports {Path(file_path).name}",
                        )
                    )
        except Exception as e:
            logger.debug(f"Error finding dependents: {e}")

        return related

    def _find_test_files(self, file_path: str) -> list[RelatedFile]:
        """Find test files for the given file."""
        related = []
        path = Path(file_path)
        name = path.stem
        suffix = path.suffix

        # Common test file patterns
        patterns = [
            f"test_{name}{suffix}",
            f"{name}_test{suffix}",
            f"test_{name}.py",
            f"{name}_test.py",
            f"{name}.test.ts",
            f"{name}.spec.ts",
            f"{name}.test.js",
            f"{name}.spec.js",
            f"{name}_test.go",
        ]

        # Search in tests/ directory
        test_dirs = ["tests", "test", "__tests__", "spec"]
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                for pattern in patterns:
                    for found in test_path.rglob(pattern):
                        related.append(
                            RelatedFile(
                                file_path=str(found),
                                relation_type="test",
                                relevance_score=0.9,
                                reason=f"test for {name}",
                            )
                        )

        # Search in same directory
        for pattern in patterns:
            test_file = path.parent / pattern
            if test_file.exists():
                related.append(
                    RelatedFile(
                        file_path=str(test_file),
                        relation_type="test",
                        relevance_score=0.9,
                        reason=f"test for {name}",
                    )
                )

        return related

    def _find_siblings(self, file_path: str) -> list[RelatedFile]:
        """Find sibling files in the same directory."""
        related = []
        path = Path(file_path)
        suffix = path.suffix

        try:
            for sibling in path.parent.iterdir():
                if sibling.is_file() and sibling.suffix == suffix and sibling != path:
                    # Exclude test files here (handled separately)
                    if "test" in sibling.name.lower():
                        continue

                    related.append(
                        RelatedFile(
                            file_path=str(sibling),
                            relation_type="sibling",
                            relevance_score=0.4,
                            reason=f"same directory as {path.name}",
                        )
                    )
        except OSError:
            pass

        return related

    def _find_symbol_references(self, symbol: str) -> list[RelatedFile]:
        """Find files that reference a symbol."""
        related = []

        # Use indexer if available
        if self._indexer:
            try:
                callers = self._indexer.get_callers(symbol)
                for file_path, _caller in callers:
                    related.append(
                        RelatedFile(
                            file_path=file_path,
                            relation_type="reference",
                            relevance_score=0.6,
                            reason=f"calls {symbol}",
                        )
                    )
            except Exception as e:
                logger.debug(f"Error finding symbol references: {e}")

        return related

    def _resolve_module(self, module: str) -> str | None:
        """Resolve a module name to a file path."""
        # Skip external modules
        if not module.startswith(".") and "." in module:
            # Might be internal (e.g., sepilot.tools.code_analysis)
            module_path = module.replace(".", "/")
            candidates = [
                self.project_root / f"{module_path}.py",
                self.project_root / module_path / "__init__.py",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)

        return None

    def _file_to_module(self, file_path: str) -> str | None:
        """Convert a file path to a module name."""
        try:
            path = Path(file_path)
            rel_path = path.relative_to(self.project_root)

            # Remove suffix
            if rel_path.suffix == ".py":
                parts = list(rel_path.with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                return ".".join(parts)
        except ValueError:
            pass

        return None

    def find_by_pattern(self, pattern: str) -> list[RelatedFile]:
        """Find files matching a naming pattern.

        Args:
            pattern: Pattern to match (e.g., "*_handler.py")

        Returns:
            List of matching files
        """
        related = []

        import fnmatch

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and fnmatch.fnmatch(file_path.name, pattern):
                related.append(
                    RelatedFile(
                        file_path=str(file_path),
                        relation_type="pattern",
                        relevance_score=0.5,
                        reason=f"matches {pattern}",
                    )
                )

        return related


def find_related_files(
    project_root: str | Path,
    files: list[str],
    symbols: list[str] | None = None,
    max_results: int = 10,
) -> list[RelatedFile]:
    """Convenience function to find related files.

    Args:
        project_root: Project root directory
        files: List of file paths
        symbols: Optional list of symbol names
        max_results: Maximum number of results

    Returns:
        List of RelatedFile sorted by relevance
    """
    finder = RelatedFileFinder(project_root)
    return finder.find_related(files, symbols, max_results)
