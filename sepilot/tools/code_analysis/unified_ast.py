"""Unified AST - Language-neutral AST representation

This module defines a common AST interface that works across all supported languages
(Python, JavaScript, TypeScript, Go, Rust). It provides a unified view of code structure
regardless of the source language.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


class SymbolKind(str, Enum):
    """Types of code symbols."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    STRUCT = "struct"
    INTERFACE = "interface"
    TRAIT = "trait"
    ENUM = "enum"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PROPERTY = "property"
    PARAMETER = "parameter"
    TYPE_ALIAS = "type_alias"
    MODULE = "module"
    NAMESPACE = "namespace"


class Visibility(str, Enum):
    """Symbol visibility/access modifiers."""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"  # Go package-level
    UNKNOWN = "unknown"


@dataclass
class Location:
    """Source code location."""
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_start}"


@dataclass
class Parameter:
    """Function/method parameter."""
    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_variadic: bool = False  # *args, ...rest
    is_keyword: bool = False   # **kwargs


@dataclass
class TypeInfo:
    """Type information."""
    name: str
    is_generic: bool = False
    generic_params: list[str] = field(default_factory=list)
    is_nullable: bool = False
    is_array: bool = False


@dataclass
class ImportInfo:
    """Import/require statement information."""
    module: str
    symbols: list[str] = field(default_factory=list)  # Specific imports
    alias: str | None = None
    is_default: bool = False  # Default import (JS/TS)
    is_star: bool = False  # import * as ...
    import_type: str = "import"  # import, from, require, use

    def __str__(self) -> str:
        if self.symbols:
            return f"from {self.module} import {', '.join(self.symbols)}"
        return f"import {self.module}"


@dataclass
class FunctionSymbol:
    """Function or method symbol."""
    name: str
    kind: SymbolKind
    location: Location
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    visibility: Visibility = Visibility.PUBLIC
    is_async: bool = False
    is_static: bool = False
    is_abstract: bool = False
    calls: list[str] = field(default_factory=list)  # Functions this calls
    complexity: int = 1  # Cyclomatic complexity

    @property
    def signature(self) -> str:
        """Generate function signature string."""
        params = ", ".join(
            f"{p.name}: {p.type_annotation}" if p.type_annotation else p.name
            for p in self.parameters
        )
        ret = f" -> {self.return_type}" if self.return_type else ""
        prefix = "async " if self.is_async else ""
        return f"{prefix}def {self.name}({params}){ret}"


@dataclass
class ClassSymbol:
    """Class, struct, or interface symbol."""
    name: str
    kind: SymbolKind
    location: Location
    base_classes: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)  # Implemented interfaces
    methods: list[FunctionSymbol] = field(default_factory=list)
    properties: list['VariableSymbol'] = field(default_factory=list)
    class_variables: list['VariableSymbol'] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None
    visibility: Visibility = Visibility.PUBLIC
    is_abstract: bool = False
    generic_params: list[str] = field(default_factory=list)


@dataclass
class VariableSymbol:
    """Variable or constant symbol."""
    name: str
    kind: SymbolKind
    location: Location
    type_annotation: str | None = None
    initial_value: str | None = None
    visibility: Visibility = Visibility.PUBLIC
    is_mutable: bool = True


@dataclass
class UnifiedAST:
    """Language-neutral AST representation.

    This is the main output of parsing any supported language.
    It provides a unified view of the code structure.
    """
    file_path: str
    language: Language
    functions: list[FunctionSymbol] = field(default_factory=list)
    classes: list[ClassSymbol] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    variables: list[VariableSymbol] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)  # Exported symbols
    errors: list[str] = field(default_factory=list)  # Parse errors
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_symbols(self) -> list[FunctionSymbol | ClassSymbol | VariableSymbol]:
        """Get all symbols in the file."""
        symbols: list = []
        symbols.extend(self.functions)
        symbols.extend(self.classes)
        symbols.extend(self.variables)
        return symbols

    @property
    def symbol_count(self) -> int:
        """Total number of symbols."""
        return len(self.functions) + len(self.classes) + len(self.variables)

    def get_function(self, name: str) -> FunctionSymbol | None:
        """Find function by name."""
        for func in self.functions:
            if func.name == name:
                return func
        return None

    def get_class(self, name: str) -> ClassSymbol | None:
        """Find class by name."""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None

    def get_method(self, class_name: str, method_name: str) -> FunctionSymbol | None:
        """Find method in a class."""
        cls = self.get_class(class_name)
        if cls:
            for method in cls.methods:
                if method.name == method_name:
                    return method
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "language": self.language.value,
            "functions": [
                {
                    "name": f.name,
                    "kind": f.kind.value,
                    "location": str(f.location),
                    "signature": f.signature,
                    "is_async": f.is_async,
                    "complexity": f.complexity,
                }
                for f in self.functions
            ],
            "classes": [
                {
                    "name": c.name,
                    "kind": c.kind.value,
                    "location": str(c.location),
                    "base_classes": c.base_classes,
                    "method_count": len(c.methods),
                }
                for c in self.classes
            ],
            "imports": [str(i) for i in self.imports],
            "variables": [
                {
                    "name": v.name,
                    "kind": v.kind.value,
                    "type": v.type_annotation,
                }
                for v in self.variables
            ],
            "symbol_count": self.symbol_count,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"File: {self.file_path}",
            f"Language: {self.language.value}",
            f"Functions: {len(self.functions)}",
            f"Classes: {len(self.classes)}",
            f"Imports: {len(self.imports)}",
            f"Variables: {len(self.variables)}",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


# Type aliases for convenience
Symbol = FunctionSymbol | ClassSymbol | VariableSymbol


def merge_asts(asts: list[UnifiedAST]) -> dict[str, Any]:
    """Merge multiple ASTs into a project-level summary.

    Args:
        asts: List of UnifiedAST objects

    Returns:
        Dictionary with merged statistics and symbols
    """
    all_functions: list[FunctionSymbol] = []
    all_classes: list[ClassSymbol] = []
    all_imports: list[ImportInfo] = []
    all_variables: list[VariableSymbol] = []
    all_errors: list[str] = []
    languages: set[Language] = set()

    for ast in asts:
        all_functions.extend(ast.functions)
        all_classes.extend(ast.classes)
        all_imports.extend(ast.imports)
        all_variables.extend(ast.variables)
        all_errors.extend(ast.errors)
        languages.add(ast.language)

    return {
        "file_count": len(asts),
        "languages": [lang.value for lang in languages],
        "total_functions": len(all_functions),
        "total_classes": len(all_classes),
        "total_imports": len(all_imports),
        "total_variables": len(all_variables),
        "total_errors": len(all_errors),
        "functions": all_functions,
        "classes": all_classes,
        "imports": all_imports,
        "variables": all_variables,
    }
