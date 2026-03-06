"""Project Indexer - Background indexing and symbol table management.

This module provides project-wide indexing capabilities including:
- Symbol table for quick symbol lookup
- Dependency graph for import/export tracking
- Call graph for function call tracking
- File watching for incremental updates
"""

from sepilot.indexer.models import (
    CallEdge,
    Dependency,
    IndexedFile,
    IndexedSymbol,
    SymbolReference,
)
from sepilot.indexer.symbol_table import SymbolTable
from sepilot.indexer.dependency_graph import DependencyGraph
from sepilot.indexer.call_graph import CallGraph
from sepilot.indexer.indexer import ProjectIndexer, get_project_indexer
from sepilot.indexer.storage import IndexStorage

__all__ = [
    # Models
    "IndexedSymbol",
    "IndexedFile",
    "Dependency",
    "CallEdge",
    "SymbolReference",
    # Core components
    "SymbolTable",
    "DependencyGraph",
    "CallGraph",
    "IndexStorage",
    # Main indexer
    "ProjectIndexer",
    "get_project_indexer",
]
