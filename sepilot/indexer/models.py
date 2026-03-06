"""Data models for the project indexer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class IndexStatus(str, Enum):
    """Status of an indexed item."""

    INDEXED = "indexed"
    PENDING = "pending"
    ERROR = "error"
    STALE = "stale"


@dataclass
class IndexedSymbol:
    """An indexed symbol with its metadata.

    Represents a function, class, method, variable, etc. that has been
    indexed from the project.
    """

    name: str
    kind: str  # function, class, method, variable, etc.
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    signature: str | None = None
    docstring: str | None = None
    language: str = "unknown"
    visibility: str = "public"
    parent: str | None = None  # For methods, this is the class name
    is_async: bool = False
    is_static: bool = False
    generic_params: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name (e.g., ClassName.method_name)."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    @property
    def location_str(self) -> str:
        """Get location as file:line string."""
        return f"{self.file_path}:{self.line_start}"

    def matches(self, query: str, exact: bool = False) -> bool:
        """Check if this symbol matches a search query."""
        if exact:
            return self.name == query or self.qualified_name == query
        query_lower = query.lower()
        return (
            query_lower in self.name.lower()
            or query_lower in self.qualified_name.lower()
        )


@dataclass
class IndexedFile:
    """Metadata for an indexed file."""

    file_path: str
    language: str
    status: IndexStatus
    indexed_at: datetime
    file_hash: str  # Content hash for change detection
    symbol_count: int = 0
    import_count: int = 0
    error_message: str | None = None

    @property
    def is_stale(self) -> bool:
        """Check if file needs re-indexing."""
        return self.status == IndexStatus.STALE

    @property
    def needs_indexing(self) -> bool:
        """Check if file needs indexing."""
        return self.status in (IndexStatus.PENDING, IndexStatus.STALE)


@dataclass
class Dependency:
    """A dependency relationship between files/modules.

    Represents an import/require/use statement.
    """

    from_file: str
    to_module: str
    import_type: str  # import, from, require, use
    symbols: list[str] = field(default_factory=list)
    alias: str | None = None
    is_default: bool = False
    is_star: bool = False
    line_number: int = 0

    @property
    def is_external(self) -> bool:
        """Check if this is an external dependency (not in project)."""
        # Simple heuristic: if module doesn't start with . and isn't a relative path
        return not self.to_module.startswith(".") and "/" not in self.to_module

    def resolves_to(self, file_path: str) -> bool:
        """Check if this dependency resolves to a specific file."""
        # Simplified resolution - would need language-specific logic
        module_path = self.to_module.replace(".", "/")
        return file_path.endswith(module_path) or module_path in file_path


@dataclass
class CallEdge:
    """An edge in the call graph.

    Represents a function call relationship.
    """

    caller_file: str
    caller_symbol: str  # Qualified name of calling function
    callee_symbol: str  # Name of called function
    callee_file: str | None = None  # Resolved file, if known
    line_number: int = 0
    call_count: int = 1  # How many times this call appears

    @property
    def is_resolved(self) -> bool:
        """Check if the callee has been resolved to a file."""
        return self.callee_file is not None

    def __hash__(self) -> int:
        return hash((self.caller_file, self.caller_symbol, self.callee_symbol))


@dataclass
class SymbolReference:
    """A reference to a symbol.

    Represents where a symbol is used/referenced in code.
    """

    symbol_name: str
    file_path: str
    line_number: int
    column: int = 0
    reference_kind: str = "read"  # read, write, call, type
    context: str | None = None  # Surrounding code context

    @property
    def location_str(self) -> str:
        """Get location as file:line:column string."""
        return f"{self.file_path}:{self.line_number}:{self.column}"


@dataclass
class ProjectIndex:
    """Complete index for a project."""

    project_root: str
    project_hash: str  # Hash of project path for storage
    created_at: datetime
    updated_at: datetime
    file_count: int = 0
    symbol_count: int = 0
    dependency_count: int = 0
    call_edge_count: int = 0
    languages: list[str] = field(default_factory=list)
    status: IndexStatus = IndexStatus.PENDING

    @property
    def storage_path(self) -> Path:
        """Get the storage path for this project's index."""
        base = Path.home() / ".sepilot" / "indexes"
        return base / self.project_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_root": self.project_root,
            "project_hash": self.project_hash,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "file_count": self.file_count,
            "symbol_count": self.symbol_count,
            "dependency_count": self.dependency_count,
            "call_edge_count": self.call_edge_count,
            "languages": self.languages,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectIndex":
        """Create from dictionary."""
        return cls(
            project_root=data["project_root"],
            project_hash=data["project_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            file_count=data.get("file_count", 0),
            symbol_count=data.get("symbol_count", 0),
            dependency_count=data.get("dependency_count", 0),
            call_edge_count=data.get("call_edge_count", 0),
            languages=data.get("languages", []),
            status=IndexStatus(data.get("status", "pending")),
        )
