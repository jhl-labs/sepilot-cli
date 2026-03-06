"""Data models for code analysis results"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Parameter:
    """Function parameter information"""
    name: str
    type_hint: str | None = None
    default_value: str | None = None
    is_kwonly: bool = False
    is_vararg: bool = False
    is_kwarg: bool = False


@dataclass
class FunctionInfo:
    """Function analysis result"""
    name: str
    line_number: int
    end_line_number: int
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    complexity: int = 1
    is_async: bool = False
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    decorators: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # Functions called by this function

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "line_number": self.line_number,
            "end_line_number": self.end_line_number,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type_hint,
                    "default": p.default_value,
                    "is_kwonly": p.is_kwonly,
                    "is_vararg": p.is_vararg,
                    "is_kwarg": p.is_kwarg
                }
                for p in self.parameters
            ],
            "return_type": self.return_type,
            "docstring": self.docstring,
            "complexity": self.complexity,
            "is_async": self.is_async,
            "is_method": self.is_method,
            "is_classmethod": self.is_classmethod,
            "is_staticmethod": self.is_staticmethod,
            "decorators": self.decorators,
            "calls": self.calls
        }


@dataclass
class ClassInfo:
    """Class analysis result"""
    name: str
    line_number: int
    end_line_number: int
    base_classes: list[str] = field(default_factory=list)
    docstring: str | None = None
    methods: list[FunctionInfo] = field(default_factory=list)
    class_variables: list[str] = field(default_factory=list)
    instance_variables: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "line_number": self.line_number,
            "end_line_number": self.end_line_number,
            "base_classes": self.base_classes,
            "docstring": self.docstring,
            "methods": [m.to_dict() for m in self.methods],
            "class_variables": self.class_variables,
            "instance_variables": self.instance_variables,
            "decorators": self.decorators
        }


@dataclass
class Import:
    """Simple import statement (import x)"""
    module: str
    alias: str | None = None


@dataclass
class FromImport:
    """From import statement (from x import y)"""
    module: str
    names: list[dict[str, str | None]] = field(default_factory=list)  # [{"name": "foo", "alias": "f"}]


@dataclass
class ImportInfo:
    """Import analysis result"""
    imports: list[Import] = field(default_factory=list)
    from_imports: list[FromImport] = field(default_factory=list)

    def get_external_dependencies(self) -> list[str]:
        """Extract external package dependencies (not stdlib)"""
        import sys
        stdlib_modules = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()

        external = set()

        # Check imports
        for imp in self.imports:
            root_module = imp.module.split('.')[0]
            if root_module not in stdlib_modules and root_module not in ['__main__', '__future__']:
                external.add(root_module)

        # Check from imports
        for from_imp in self.from_imports:
            root_module = from_imp.module.split('.')[0]
            if root_module not in stdlib_modules and root_module not in ['__main__', '__future__']:
                external.add(root_module)

        return sorted(external)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "imports": [{"module": i.module, "alias": i.alias} for i in self.imports],
            "from_imports": [
                {"module": fi.module, "names": fi.names}
                for fi in self.from_imports
            ],
            "external_dependencies": self.get_external_dependencies()
        }


@dataclass
class ComplexityInfo:
    """Code complexity metrics"""
    cyclomatic_complexity: int
    lines_of_code: int
    max_nesting_depth: int
    parameter_count: int

    @property
    def complexity_level(self) -> str:
        """Determine complexity level"""
        if self.cyclomatic_complexity <= 5:
            return "low"
        elif self.cyclomatic_complexity <= 10:
            return "medium"
        else:
            return "high"

    def get_suggestions(self) -> list[str]:
        """Get improvement suggestions"""
        suggestions = []

        if self.cyclomatic_complexity > 10:
            suggestions.append(
                f"High cyclomatic complexity ({self.cyclomatic_complexity}). "
                "Consider splitting into smaller functions."
            )

        if self.lines_of_code > 50:
            suggestions.append(
                f"Function is long ({self.lines_of_code} lines). "
                "Consider splitting into smaller functions."
            )

        if self.max_nesting_depth > 3:
            suggestions.append(
                f"Deep nesting detected (depth: {self.max_nesting_depth}). "
                "Consider reducing nesting depth (recommended: <= 3)."
            )

        if self.parameter_count > 4:
            suggestions.append(
                f"Too many parameters ({self.parameter_count}). "
                "Consider using a configuration object (recommended: <= 4)."
            )

        return suggestions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "lines_of_code": self.lines_of_code,
            "max_nesting_depth": self.max_nesting_depth,
            "parameter_count": self.parameter_count,
            "complexity_level": self.complexity_level,
            "suggestions": self.get_suggestions()
        }


@dataclass
class FileAnalysis:
    """Complete file analysis result"""
    file_path: str
    total_lines: int
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: ImportInfo = field(default_factory=ImportInfo)
    global_variables: list[str] = field(default_factory=list)

    @property
    def total_functions(self) -> int:
        """Total number of functions (including methods)"""
        count = len(self.functions)
        for cls in self.classes:
            count += len(cls.methods)
        return count

    @property
    def total_classes(self) -> int:
        """Total number of classes"""
        return len(self.classes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "file_path": self.file_path,
            "summary": {
                "total_lines": self.total_lines,
                "total_functions": self.total_functions,
                "total_classes": self.total_classes,
                "total_imports": len(self.imports.imports) + len(self.imports.from_imports)
            },
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "imports": self.imports.to_dict(),
            "global_variables": self.global_variables
        }


@dataclass
class Reference:
    """Symbol reference/usage location"""
    file_path: str
    line_number: int
    context: str  # Line content or surrounding context
    reference_type: str  # "call", "import", "definition", "usage"
