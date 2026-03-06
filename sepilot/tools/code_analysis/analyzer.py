"""High-level code analyzer using AST parser"""

import json
from pathlib import Path
from typing import Any

from .ast_parser import ASTParser
from .models import ClassInfo, FileAnalysis, FunctionInfo, ImportInfo, Reference


class CodeAnalyzer:
    """High-level interface for code analysis"""

    def __init__(self):
        pass

    def analyze_file(self, file_path: str) -> FileAnalysis:
        """
        Analyze a Python file completely

        Args:
            file_path: Path to the Python file

        Returns:
            FileAnalysis object with all information

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file has syntax errors
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.suffix == '.py':
            raise ValueError(f"Not a Python file: {file_path}")

        with open(path, encoding='utf-8') as f:
            source_code = f.read()

        parser = ASTParser(source_code, str(path))
        return parser.parse()

    def list_functions(self, file_path: str) -> list[str]:
        """
        List all function names in a file

        Args:
            file_path: Path to the Python file

        Returns:
            List of function names
        """
        analysis = self.analyze_file(file_path)
        function_names = [f.name for f in analysis.functions]

        # Add methods from classes
        for cls in analysis.classes:
            for method in cls.methods:
                function_names.append(f"{cls.name}.{method.name}")

        return function_names

    def list_classes(self, file_path: str) -> list[str]:
        """
        List all class names in a file

        Args:
            file_path: Path to the Python file

        Returns:
            List of class names
        """
        analysis = self.analyze_file(file_path)
        return [c.name for c in analysis.classes]

    def find_function(self, file_path: str, function_name: str) -> FunctionInfo | None:
        """
        Find and analyze a specific function

        Args:
            file_path: Path to the Python file
            function_name: Name of the function to find

        Returns:
            FunctionInfo if found, None otherwise
        """
        path = Path(file_path)
        with open(path, encoding='utf-8') as f:
            source_code = f.read()

        parser = ASTParser(source_code, str(path))
        return parser.find_function(function_name)

    def find_class(self, file_path: str, class_name: str) -> ClassInfo | None:
        """
        Find and analyze a specific class

        Args:
            file_path: Path to the Python file
            class_name: Name of the class to find

        Returns:
            ClassInfo if found, None otherwise
        """
        path = Path(file_path)
        with open(path, encoding='utf-8') as f:
            source_code = f.read()

        parser = ASTParser(source_code, str(path))
        return parser.find_class(class_name)

    def analyze_imports(self, file_path: str) -> ImportInfo:
        """
        Analyze import statements in a file

        Args:
            file_path: Path to the Python file

        Returns:
            ImportInfo object
        """
        analysis = self.analyze_file(file_path)
        return analysis.imports

    def check_complexity(self, file_path: str, threshold: int = 10) -> list[dict[str, Any]]:
        """
        Check functions that exceed complexity threshold

        Args:
            file_path: Path to the Python file
            threshold: Complexity threshold (default: 10)

        Returns:
            List of complex functions with their metrics
        """
        path = Path(file_path)
        with open(path, encoding='utf-8') as f:
            source_code = f.read()

        parser = ASTParser(source_code, str(path))
        analysis = parser.parse()

        complex_functions = []

        # Check top-level functions
        for func in analysis.functions:
            if func.complexity > threshold:
                complexity_info = parser.calculate_complexity(func.name)
                if complexity_info:
                    complex_functions.append({
                        "name": func.name,
                        "type": "function",
                        "line": func.line_number,
                        "metrics": complexity_info.to_dict()
                    })

        # Check methods
        for cls in analysis.classes:
            for method in cls.methods:
                if method.complexity > threshold:
                    complexity_info = parser.calculate_complexity(method.name)
                    if complexity_info:
                        complex_functions.append({
                            "name": f"{cls.name}.{method.name}",
                            "type": "method",
                            "line": method.line_number,
                            "metrics": complexity_info.to_dict()
                        })

        return complex_functions

    def find_references(
        self,
        symbol_name: str,
        directory: str = ".",
        file_pattern: str = "*.py"
    ) -> list[Reference]:
        """
        Find all references to a symbol (function/class/variable) in a directory

        Args:
            symbol_name: Name of the symbol to find
            directory: Directory to search in
            file_pattern: File pattern to match (default: *.py)

        Returns:
            List of Reference objects
        """
        references = []
        search_path = Path(directory)

        # Find all Python files
        py_files = list(search_path.rglob(file_pattern))

        for py_file in py_files:
            try:
                with open(py_file, encoding='utf-8') as f:
                    lines = f.readlines()

                # Simple text-based search (can be improved with AST)
                for line_num, line_content in enumerate(lines, 1):
                    if symbol_name in line_content:
                        # Determine reference type
                        ref_type = "usage"
                        if f"def {symbol_name}" in line_content or f"class {symbol_name}" in line_content:
                            ref_type = "definition"
                        elif f"import {symbol_name}" in line_content or f"from .* import .*{symbol_name}" in line_content:
                            ref_type = "import"
                        elif f"{symbol_name}(" in line_content:
                            ref_type = "call"

                        references.append(Reference(
                            file_path=str(py_file),
                            line_number=line_num,
                            context=line_content.strip(),
                            reference_type=ref_type
                        ))

            except Exception:
                # Skip files that can't be read
                continue

        return references

    def format_file_analysis(self, file_path: str, format: str = "text") -> str:
        """
        Format file analysis result for display

        Args:
            file_path: Path to the Python file
            format: Output format ("text" or "json")

        Returns:
            Formatted string
        """
        analysis = self.analyze_file(file_path)

        if format == "json":
            return json.dumps(analysis.to_dict(), indent=2)

        # Text format
        lines = []
        lines.append(f"📄 File Analysis: {analysis.file_path}")
        lines.append("\n📊 Summary:")
        lines.append(f"  • Total lines: {analysis.total_lines}")
        lines.append(f"  • Functions: {analysis.total_functions}")
        lines.append(f"  • Classes: {analysis.total_classes}")
        lines.append(f"  • Imports: {len(analysis.imports.imports) + len(analysis.imports.from_imports)}")

        if analysis.functions:
            lines.append(f"\n🔧 Functions ({len(analysis.functions)}):")
            for func in analysis.functions:
                params = ", ".join([p.name for p in func.parameters])
                lines.append(f"  • {func.name}({params}) [line {func.line_number}]")
                if func.complexity > 5:
                    lines.append(f"    ⚠️  Complexity: {func.complexity}")

        if analysis.classes:
            lines.append(f"\n📦 Classes ({len(analysis.classes)}):")
            for cls in analysis.classes:
                lines.append(f"  • {cls.name} [line {cls.line_number}]")
                if cls.base_classes:
                    lines.append(f"    Inherits: {', '.join(cls.base_classes)}")
                if cls.methods:
                    lines.append(f"    Methods: {', '.join([m.name for m in cls.methods])}")

        external_deps = analysis.imports.get_external_dependencies()
        if external_deps:
            lines.append("\n📚 External Dependencies:")
            for dep in external_deps:
                lines.append(f"  • {dep}")

        return "\n".join(lines)

    def format_function_info(self, func_info: FunctionInfo, format: str = "text") -> str:
        """
        Format function information for display

        Args:
            func_info: FunctionInfo object
            format: Output format ("text" or "json")

        Returns:
            Formatted string
        """
        if format == "json":
            return json.dumps(func_info.to_dict(), indent=2)

        # Text format
        lines = []
        lines.append(f"🔧 Function: {func_info.name}")
        lines.append(f"  Location: line {func_info.line_number}-{func_info.end_line_number}")

        # Parameters
        if func_info.parameters:
            lines.append("\n  Parameters:")
            for param in func_info.parameters:
                param_str = f"    • {param.name}"
                if param.type_hint:
                    param_str += f": {param.type_hint}"
                if param.default_value:
                    param_str += f" = {param.default_value}"
                if param.is_vararg:
                    param_str = f"    • *{param.name}"
                elif param.is_kwarg:
                    param_str = f"    • **{param.name}"
                lines.append(param_str)

        # Return type
        if func_info.return_type:
            lines.append(f"\n  Returns: {func_info.return_type}")

        # Docstring
        if func_info.docstring:
            lines.append("\n  Description:")
            for line in func_info.docstring.split('\n'):
                lines.append(f"    {line}")

        # Complexity
        lines.append("\n  Metrics:")
        lines.append(f"    • Complexity: {func_info.complexity}")
        if func_info.complexity > 10:
            lines.append("      ⚠️  High complexity - consider refactoring")

        # Decorators
        if func_info.decorators:
            lines.append(f"    • Decorators: {', '.join(func_info.decorators)}")

        # Function calls
        if func_info.calls:
            lines.append(f"\n  Calls: {', '.join(func_info.calls)}")

        return "\n".join(lines)

    def format_class_info(self, class_info: ClassInfo, format: str = "text") -> str:
        """
        Format class information for display

        Args:
            class_info: ClassInfo object
            format: Output format ("text" or "json")

        Returns:
            Formatted string
        """
        if format == "json":
            return json.dumps(class_info.to_dict(), indent=2)

        # Text format
        lines = []
        lines.append(f"📦 Class: {class_info.name}")
        lines.append(f"  Location: line {class_info.line_number}-{class_info.end_line_number}")

        # Base classes
        if class_info.base_classes:
            lines.append(f"  Inherits from: {', '.join(class_info.base_classes)}")

        # Docstring
        if class_info.docstring:
            lines.append("\n  Description:")
            for line in class_info.docstring.split('\n'):
                lines.append(f"    {line}")

        # Methods
        if class_info.methods:
            lines.append(f"\n  Methods ({len(class_info.methods)}):")
            for method in class_info.methods:
                params = ", ".join([p.name for p in method.parameters])
                method_str = f"    • {method.name}({params})"
                if method.is_classmethod:
                    method_str += " [@classmethod]"
                elif method.is_staticmethod:
                    method_str += " [@staticmethod]"
                lines.append(method_str)
                if method.complexity > 5:
                    lines.append(f"      ⚠️  Complexity: {method.complexity}")

        # Variables
        if class_info.class_variables:
            lines.append(f"\n  Class variables: {', '.join(class_info.class_variables)}")
        if class_info.instance_variables:
            lines.append(f"  Instance variables: {', '.join(class_info.instance_variables)}")

        return "\n".join(lines)
