"""Data models for LSP protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


@dataclass
class Position:
    """A position in a text document (0-indexed)."""

    line: int
    character: int

    def to_dict(self) -> dict[str, int]:
        """Convert to LSP format."""
        return {"line": self.line, "character": self.character}

    @classmethod
    def from_dict(cls, data: dict) -> Position:
        """Create from LSP format."""
        return cls(line=data["line"], character=data["character"])


@dataclass
class Range:
    """A range in a text document."""

    start: Position
    end: Position

    def to_dict(self) -> dict:
        """Convert to LSP format."""
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> Range:
        """Create from LSP format."""
        return cls(
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
        )

    @property
    def is_empty(self) -> bool:
        """Check if range is empty (single position)."""
        return (
            self.start.line == self.end.line
            and self.start.character == self.end.character
        )


@dataclass
class Location:
    """A location in a document."""

    uri: str
    range: Range

    def to_dict(self) -> dict:
        """Convert to LSP format."""
        return {"uri": self.uri, "range": self.range.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> Location:
        """Create from LSP format."""
        return cls(uri=data["uri"], range=Range.from_dict(data["range"]))

    @property
    def file_path(self) -> str:
        """Get file path from URI."""
        if self.uri.startswith("file://"):
            return self.uri[7:]
        return self.uri

    @property
    def line_number(self) -> int:
        """Get 1-indexed line number."""
        return self.range.start.line + 1

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number}"


@dataclass
class LocationLink:
    """A link to a location with origin information."""

    origin_selection_range: Range | None
    target_uri: str
    target_range: Range
    target_selection_range: Range

    @classmethod
    def from_dict(cls, data: dict) -> LocationLink:
        """Create from LSP format."""
        return cls(
            origin_selection_range=(
                Range.from_dict(data["originSelectionRange"])
                if data.get("originSelectionRange")
                else None
            ),
            target_uri=data["targetUri"],
            target_range=Range.from_dict(data["targetRange"]),
            target_selection_range=Range.from_dict(data["targetSelectionRange"]),
        )

    def to_location(self) -> Location:
        """Convert to simple Location."""
        return Location(uri=self.target_uri, range=self.target_selection_range)


class SymbolKind(IntEnum):
    """LSP symbol kinds."""

    FILE = 1
    MODULE = 2
    NAMESPACE = 3
    PACKAGE = 4
    CLASS = 5
    METHOD = 6
    PROPERTY = 7
    FIELD = 8
    CONSTRUCTOR = 9
    ENUM = 10
    INTERFACE = 11
    FUNCTION = 12
    VARIABLE = 13
    CONSTANT = 14
    STRING = 15
    NUMBER = 16
    BOOLEAN = 17
    ARRAY = 18
    OBJECT = 19
    KEY = 20
    NULL = 21
    ENUM_MEMBER = 22
    STRUCT = 23
    EVENT = 24
    OPERATOR = 25
    TYPE_PARAMETER = 26


@dataclass
class SymbolInformation:
    """Information about a symbol."""

    name: str
    kind: SymbolKind
    location: Location
    container_name: str | None = None
    tags: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> SymbolInformation:
        """Create from LSP format."""
        return cls(
            name=data["name"],
            kind=SymbolKind(data["kind"]),
            location=Location.from_dict(data["location"]),
            container_name=data.get("containerName"),
            tags=data.get("tags", []),
        )

    @property
    def qualified_name(self) -> str:
        """Get qualified name."""
        if self.container_name:
            return f"{self.container_name}.{self.name}"
        return self.name


@dataclass
class DocumentSymbol:
    """A document symbol with hierarchy."""

    name: str
    kind: SymbolKind
    range: Range
    selection_range: Range
    detail: str | None = None
    children: list[DocumentSymbol] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> DocumentSymbol:
        """Create from LSP format."""
        return cls(
            name=data["name"],
            kind=SymbolKind(data["kind"]),
            range=Range.from_dict(data["range"]),
            selection_range=Range.from_dict(data["selectionRange"]),
            detail=data.get("detail"),
            children=[
                DocumentSymbol.from_dict(c) for c in data.get("children", [])
            ],
        )


@dataclass
class HoverInfo:
    """Hover information for a symbol."""

    contents: str
    range: Range | None = None
    language: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> HoverInfo:
        """Create from LSP format."""
        contents = data.get("contents", "")

        # Handle MarkupContent
        if isinstance(contents, dict):
            contents = contents.get("value", "")
        # Handle MarkedString array
        elif isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, dict):
                    parts.append(item.get("value", ""))
                else:
                    parts.append(str(item))
            contents = "\n\n".join(parts)

        return cls(
            contents=contents,
            range=Range.from_dict(data["range"]) if data.get("range") else None,
        )

    @property
    def is_empty(self) -> bool:
        """Check if hover has content."""
        return not self.contents.strip()


class DiagnosticSeverity(IntEnum):
    """Diagnostic severity levels."""

    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


@dataclass
class Diagnostic:
    """A diagnostic (error, warning, etc.)."""

    range: Range
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    code: str | int | None = None
    source: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Diagnostic:
        """Create from LSP format."""
        return cls(
            range=Range.from_dict(data["range"]),
            message=data["message"],
            severity=DiagnosticSeverity(data.get("severity", 1)),
            code=data.get("code"),
            source=data.get("source"),
        )

    @property
    def severity_name(self) -> str:
        """Get human-readable severity name."""
        severity_names = {
            1: "error",
            2: "warning",
            3: "info",
            4: "hint",
        }
        return severity_names.get(self.severity, "unknown")

    def __str__(self) -> str:
        return f"{self.severity.name}: {self.message}"


@dataclass
class CallHierarchyItem:
    """An item in a call hierarchy."""

    name: str
    kind: SymbolKind
    uri: str
    range: Range
    selection_range: Range
    detail: str | None = None
    tags: list[int] = field(default_factory=list)
    data: Any = None

    @classmethod
    def from_dict(cls, data: dict) -> CallHierarchyItem:
        """Create from LSP format."""
        return cls(
            name=data["name"],
            kind=SymbolKind(data["kind"]),
            uri=data["uri"],
            range=Range.from_dict(data["range"]),
            selection_range=Range.from_dict(data["selectionRange"]),
            detail=data.get("detail"),
            tags=data.get("tags", []),
            data=data.get("data"),
        )

    def to_dict(self) -> dict:
        """Convert to LSP format."""
        result = {
            "name": self.name,
            "kind": self.kind.value,
            "uri": self.uri,
            "range": self.range.to_dict(),
            "selectionRange": self.selection_range.to_dict(),
        }
        if self.detail:
            result["detail"] = self.detail
        if self.tags:
            result["tags"] = self.tags
        if self.data:
            result["data"] = self.data
        return result

    @property
    def file_path(self) -> str:
        """Get file path from URI."""
        if self.uri.startswith("file://"):
            return self.uri[7:]
        return self.uri


@dataclass
class CallHierarchyIncomingCall:
    """An incoming call in a call hierarchy."""

    from_item: CallHierarchyItem
    from_ranges: list[Range]

    @classmethod
    def from_dict(cls, data: dict) -> CallHierarchyIncomingCall:
        """Create from LSP format."""
        return cls(
            from_item=CallHierarchyItem.from_dict(data["from"]),
            from_ranges=[Range.from_dict(r) for r in data["fromRanges"]],
        )


@dataclass
class CallHierarchyOutgoingCall:
    """An outgoing call in a call hierarchy."""

    to_item: CallHierarchyItem
    from_ranges: list[Range]

    @classmethod
    def from_dict(cls, data: dict) -> CallHierarchyOutgoingCall:
        """Create from LSP format."""
        return cls(
            to_item=CallHierarchyItem.from_dict(data["to"]),
            from_ranges=[Range.from_dict(r) for r in data["fromRanges"]],
        )


@dataclass
class TextDocumentIdentifier:
    """Identifies a text document."""

    uri: str

    def to_dict(self) -> dict:
        """Convert to LSP format."""
        return {"uri": self.uri}

    @classmethod
    def from_file_path(cls, file_path: str) -> TextDocumentIdentifier:
        """Create from file path."""
        return cls(uri=f"file://{file_path}")


@dataclass
class TextDocumentPositionParams:
    """Position in a text document."""

    text_document: TextDocumentIdentifier
    position: Position

    def to_dict(self) -> dict:
        """Convert to LSP format."""
        return {
            "textDocument": self.text_document.to_dict(),
            "position": self.position.to_dict(),
        }


@dataclass
class InitializeResult:
    """Result of initialize request."""

    capabilities: dict[str, Any]
    server_info: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> InitializeResult:
        """Create from LSP format."""
        return cls(
            capabilities=data.get("capabilities", {}),
            server_info=data.get("serverInfo"),
        )

    def supports_definition(self) -> bool:
        """Check if server supports go to definition."""
        return bool(self.capabilities.get("definitionProvider"))

    def supports_references(self) -> bool:
        """Check if server supports find references."""
        return bool(self.capabilities.get("referencesProvider"))

    def supports_hover(self) -> bool:
        """Check if server supports hover."""
        return bool(self.capabilities.get("hoverProvider"))

    def supports_document_symbol(self) -> bool:
        """Check if server supports document symbol."""
        return bool(self.capabilities.get("documentSymbolProvider"))

    def supports_workspace_symbol(self) -> bool:
        """Check if server supports workspace symbol."""
        return bool(self.capabilities.get("workspaceSymbolProvider"))

    def supports_call_hierarchy(self) -> bool:
        """Check if server supports call hierarchy."""
        return bool(self.capabilities.get("callHierarchyProvider"))
