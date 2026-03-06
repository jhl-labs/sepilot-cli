"""Project Indexer - Main indexing orchestrator."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

from sepilot.tools.code_analysis.language_detector import get_language_detector
from sepilot.tools.code_analysis.tree_sitter_parser import get_tree_sitter_parser
from sepilot.tools.code_analysis.unified_ast import Language, UnifiedAST

from .call_graph import CallGraph
from .dependency_graph import DependencyGraph
from .models import CallEdge, Dependency, IndexedFile, IndexedSymbol, IndexStatus
from .storage import IndexStorage, get_storage_for_project
from .symbol_table import SymbolTable

logger = logging.getLogger(__name__)


class ProjectIndexer:
    """Main project indexer.

    Provides background indexing, incremental updates, and query interface.
    """

    # Files/directories to exclude from indexing
    DEFAULT_EXCLUDES = {
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "venv",
        ".venv",
        "env",
        ".env",
        "build",
        "dist",
        ".eggs",
        "*.egg-info",
        ".tox",
        "coverage",
        ".coverage",
        "htmlcov",
        ".hypothesis",
    }

    def __init__(
        self,
        project_root: str | Path,
        storage: IndexStorage | None = None,
        excludes: set[str] | None = None,
    ):
        """Initialize the indexer.

        Args:
            project_root: Root directory of the project
            storage: Optional storage backend (default: SQLite)
            excludes: Additional patterns to exclude
        """
        self.project_root = Path(project_root).resolve()
        self.storage = storage or get_storage_for_project(str(self.project_root))

        self.excludes = self.DEFAULT_EXCLUDES.copy()
        if excludes:
            self.excludes.update(excludes)

        # In-memory structures
        self.symbol_table = SymbolTable()
        self.dependency_graph = DependencyGraph()
        self.call_graph = CallGraph()

        # Tools
        self._parser = get_tree_sitter_parser()
        self._detector = get_language_detector()

        # Background indexing
        self._indexing_thread: threading.Thread | None = None
        self._stop_indexing = threading.Event()
        self._indexing_progress: dict[str, int] = {}
        self._progress_lock = threading.Lock()

        # Callbacks
        self._on_file_indexed: list[Callable[[str, bool], None]] = []
        self._on_indexing_complete: list[Callable[[], None]] = []

    def index_project(
        self,
        background: bool = True,
        force: bool = False,
        max_workers: int = 4,
    ) -> None:
        """Index the entire project.

        Args:
            background: Run indexing in background thread
            force: Re-index even if already indexed
            max_workers: Number of parallel workers
        """
        if background:
            self._stop_indexing.clear()
            self._indexing_thread = threading.Thread(
                target=self._index_project_impl,
                args=(force, max_workers),
                daemon=True,
            )
            self._indexing_thread.start()
        else:
            self._index_project_impl(force, max_workers)

    def _index_project_impl(self, force: bool, max_workers: int) -> None:
        """Implementation of project indexing."""
        start_time = time.time()
        logger.info(f"Starting indexing of {self.project_root}")

        # Find all files to index
        files_to_index = self._find_files_to_index(force)
        total_files = len(files_to_index)

        self._indexing_progress = {
            "total": total_files,
            "indexed": 0,
            "errors": 0,
        }

        logger.info(f"Found {total_files} files to index")

        # Index files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._index_file, f): f for f in files_to_index
            }

            for future in as_completed(futures):
                if self._stop_indexing.is_set():
                    break

                file_path = futures[future]
                try:
                    success = future.result()
                    with self._progress_lock:
                        self._indexing_progress["indexed"] += 1
                        if not success:
                            self._indexing_progress["errors"] += 1

                    # Notify callbacks
                    for callback in self._on_file_indexed:
                        callback(file_path, success)

                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")
                    with self._progress_lock:
                        self._indexing_progress["errors"] += 1

        elapsed = time.time() - start_time
        logger.info(
            f"Indexing complete: {self._indexing_progress['indexed']} files "
            f"in {elapsed:.2f}s ({self._indexing_progress['errors']} errors)"
        )

        # Notify completion callbacks
        for callback in self._on_indexing_complete:
            callback()

    def _find_files_to_index(self, force: bool) -> list[str]:
        """Find all files that need indexing."""
        files = []

        for file_path in self.project_root.rglob("*"):
            if not file_path.is_file():
                continue

            # Check exclusions
            if self._should_exclude(file_path):
                continue

            # Check if supported language
            if not self._detector.is_supported(file_path):
                continue

            # Check if needs indexing
            if not force:
                existing = self.storage.get_file(str(file_path))
                if existing and not existing.needs_indexing:
                    # Check if file changed
                    current_hash = self._compute_file_hash(file_path)
                    if current_hash == existing.file_hash:
                        continue

            files.append(str(file_path))

        return files

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded from indexing."""
        parts = file_path.relative_to(self.project_root).parts

        for part in parts:
            if part in self.excludes:
                return True
            for pattern in self.excludes:
                if "*" in pattern:
                    import fnmatch
                    if fnmatch.fnmatch(part, pattern):
                        return True

        return False

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file content for change detection."""
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content, usedforsecurity=False).hexdigest()
        except OSError:
            return ""

    def _index_file(self, file_path: str) -> bool:
        """Index a single file.

        Args:
            file_path: Path to the file

        Returns:
            True if successful
        """
        path = Path(file_path)

        try:
            # Read content
            raw_bytes = path.read_bytes()
            file_hash = hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest()
            content = raw_bytes.decode("utf-8")

            # Parse file
            ast = self._parser.parse_file(file_path, content)

            if ast.errors and not (ast.functions or ast.classes):
                # Parse failed completely
                self._save_file_error(file_path, ast.language, file_hash, ast.errors[0])
                return False

            # Remove old data for this file
            self._remove_file_from_indexes(file_path)

            # Extract and save symbols
            symbols = self._extract_symbols(ast)
            if symbols:
                self.symbol_table.add_many(symbols)
                self.storage.save_symbols(symbols)

            # Extract and save dependencies
            dependencies = self._extract_dependencies(ast)
            if dependencies:
                self.dependency_graph.add_dependencies(file_path, dependencies)
                self.storage.save_dependencies(dependencies)

            # Extract and save call edges
            call_edges = self._extract_call_edges(ast)
            if call_edges:
                self.call_graph.add_calls(call_edges)
                self.storage.save_call_edges(call_edges)

            # Save file info
            file_info = IndexedFile(
                file_path=file_path,
                language=ast.language.value,
                status=IndexStatus.INDEXED,
                indexed_at=datetime.now(),
                file_hash=file_hash,
                symbol_count=len(symbols),
                import_count=len(dependencies),
            )
            self.storage.save_file(file_info)

            return True

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            self._save_file_error(file_path, Language.UNKNOWN, "", str(e))
            return False

    def _save_file_error(
        self, file_path: str, language: Language, file_hash: str, error: str
    ) -> None:
        """Save file with error status."""
        file_info = IndexedFile(
            file_path=file_path,
            language=language.value,
            status=IndexStatus.ERROR,
            indexed_at=datetime.now(),
            file_hash=file_hash,
            error_message=error,
        )
        self.storage.save_file(file_info)

    def _remove_file_from_indexes(self, file_path: str) -> None:
        """Remove a file from all in-memory indexes."""
        self.symbol_table.remove_file(file_path)
        self.dependency_graph.remove_file(file_path)
        self.call_graph.remove_file(file_path)

    def _extract_symbols(self, ast: UnifiedAST) -> list[IndexedSymbol]:
        """Extract IndexedSymbol objects from AST."""
        symbols = []

        # Functions
        for func in ast.functions:
            symbols.append(
                IndexedSymbol(
                    name=func.name,
                    kind=func.kind.value,
                    file_path=ast.file_path,
                    line_start=func.location.line_start,
                    line_end=func.location.line_end,
                    column_start=func.location.column_start,
                    column_end=func.location.column_end,
                    signature=func.signature,
                    docstring=func.docstring,
                    language=ast.language.value,
                    visibility=func.visibility.value,
                    is_async=func.is_async,
                    is_static=func.is_static,
                    decorators=func.decorators,
                )
            )

        # Classes
        for cls in ast.classes:
            symbols.append(
                IndexedSymbol(
                    name=cls.name,
                    kind=cls.kind.value,
                    file_path=ast.file_path,
                    line_start=cls.location.line_start,
                    line_end=cls.location.line_end,
                    column_start=cls.location.column_start,
                    column_end=cls.location.column_end,
                    docstring=cls.docstring,
                    language=ast.language.value,
                    visibility=cls.visibility.value,
                    decorators=cls.decorators,
                    generic_params=cls.generic_params,
                    metadata={
                        "base_classes": cls.base_classes,
                        "interfaces": cls.interfaces,
                    },
                )
            )

            # Methods
            for method in cls.methods:
                symbols.append(
                    IndexedSymbol(
                        name=method.name,
                        kind=method.kind.value,
                        file_path=ast.file_path,
                        line_start=method.location.line_start,
                        line_end=method.location.line_end,
                        signature=method.signature,
                        docstring=method.docstring,
                        language=ast.language.value,
                        visibility=method.visibility.value,
                        parent=cls.name,
                        is_async=method.is_async,
                        is_static=method.is_static,
                        decorators=method.decorators,
                    )
                )

        # Variables
        for var in ast.variables:
            symbols.append(
                IndexedSymbol(
                    name=var.name,
                    kind=var.kind.value,
                    file_path=ast.file_path,
                    line_start=var.location.line_start,
                    line_end=var.location.line_end,
                    language=ast.language.value,
                    visibility=var.visibility.value,
                    metadata={"type": var.type_annotation},
                )
            )

        return symbols

    def _extract_dependencies(self, ast: UnifiedAST) -> list[Dependency]:
        """Extract Dependency objects from AST."""
        dependencies = []

        for imp in ast.imports:
            dependencies.append(
                Dependency(
                    from_file=ast.file_path,
                    to_module=imp.module,
                    import_type=imp.import_type,
                    symbols=imp.symbols,
                    alias=imp.alias,
                    is_default=imp.is_default,
                    is_star=imp.is_star,
                )
            )

        return dependencies

    def _extract_call_edges(self, ast: UnifiedAST) -> list[CallEdge]:
        """Extract CallEdge objects from AST."""
        edges = []

        for func in ast.functions:
            for called in func.calls:
                edges.append(
                    CallEdge(
                        caller_file=ast.file_path,
                        caller_symbol=func.name,
                        callee_symbol=called,
                    )
                )

        for cls in ast.classes:
            for method in cls.methods:
                for called in method.calls:
                    edges.append(
                        CallEdge(
                            caller_file=ast.file_path,
                            caller_symbol=f"{cls.name}.{method.name}",
                            callee_symbol=called,
                        )
                    )

        return edges

    # Query interface

    def get_symbol(self, name: str) -> list[IndexedSymbol]:
        """Get symbols by name.

        Args:
            name: Symbol name

        Returns:
            List of matching symbols
        """
        return self.symbol_table.get_by_name(name)

    def search_symbols(
        self, query: str, kind: str | None = None, limit: int = 100
    ) -> list[IndexedSymbol]:
        """Search for symbols.

        Args:
            query: Search query
            kind: Optional filter by kind
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        return self.symbol_table.search(query, kind=kind, limit=limit)

    def get_file_symbols(self, file_path: str) -> list[IndexedSymbol]:
        """Get all symbols in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of symbols
        """
        return self.symbol_table.get_by_file(file_path)

    def get_dependencies(self, file_path: str) -> list[Dependency]:
        """Get dependencies of a file.

        Args:
            file_path: Path to the file

        Returns:
            List of dependencies
        """
        return self.dependency_graph.get_dependencies(file_path)

    def get_dependents(self, module: str) -> set[str]:
        """Get files that depend on a module.

        Args:
            module: Module name

        Returns:
            Set of file paths
        """
        return self.dependency_graph.get_dependents(module)

    def get_callers(self, symbol: str) -> list[tuple[str, str]]:
        """Get all callers of a symbol.

        Args:
            symbol: Symbol name

        Returns:
            List of (file_path, caller_symbol) tuples
        """
        return self.call_graph.get_callers(symbol)

    def get_callees(self, file_path: str, symbol: str) -> list[str]:
        """Get symbols called by a function.

        Args:
            file_path: File containing the caller
            symbol: Caller symbol name

        Returns:
            List of called symbol names
        """
        return self.call_graph.get_callees(file_path, symbol)

    def get_call_hierarchy(
        self, file_path: str, symbol: str, direction: str = "outgoing"
    ) -> dict:
        """Get call hierarchy for a symbol.

        Args:
            file_path: File containing the symbol
            symbol: Symbol name
            direction: "outgoing" or "incoming"

        Returns:
            Call hierarchy tree
        """
        return self.call_graph.get_call_hierarchy(file_path, symbol, direction)

    # File watching

    def update_file(self, file_path: str) -> bool:
        """Update index for a single file.

        Args:
            file_path: Path to the file

        Returns:
            True if successful
        """
        return self._index_file(file_path)

    def remove_file(self, file_path: str) -> None:
        """Remove a file from the index.

        Args:
            file_path: Path to the file
        """
        self._remove_file_from_indexes(file_path)
        self.storage.delete_file(file_path)

    # Callbacks

    def on_file_indexed(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for when a file is indexed.

        Args:
            callback: Function called with (file_path, success)
        """
        self._on_file_indexed.append(callback)

    def on_indexing_complete(self, callback: Callable[[], None]) -> None:
        """Register callback for when indexing is complete.

        Args:
            callback: Function called when complete
        """
        self._on_indexing_complete.append(callback)

    # Management

    def stop_indexing(self) -> None:
        """Stop background indexing."""
        self._stop_indexing.set()
        if self._indexing_thread:
            self._indexing_thread.join(timeout=5.0)

    def get_progress(self) -> dict[str, int]:
        """Get current indexing progress.

        Returns:
            Progress dictionary with total, indexed, errors
        """
        return self._indexing_progress.copy()

    def get_statistics(self) -> dict:
        """Get indexer statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "project_root": str(self.project_root),
            "symbol_table": self.symbol_table.get_statistics(),
            "dependency_graph": self.dependency_graph.get_statistics(),
            "call_graph": self.call_graph.get_statistics(),
            "storage": self.storage.get_statistics(),
        }

    def clear(self) -> None:
        """Clear all index data."""
        self.symbol_table.clear()
        self.dependency_graph.clear()
        self.call_graph.clear()
        self.storage.clear_all()

    def close(self) -> None:
        """Close the indexer and release resources."""
        self.stop_indexing()
        self.storage.close()


# Singleton instance per project
_indexers: dict[str, ProjectIndexer] = {}


def get_project_indexer(project_root: str | Path) -> ProjectIndexer:
    """Get or create an indexer for a project.

    Args:
        project_root: Project root directory

    Returns:
        ProjectIndexer instance
    """
    root = str(Path(project_root).resolve())
    if root not in _indexers:
        _indexers[root] = ProjectIndexer(root)
    return _indexers[root]
