"""AST-based Python code parser"""

import ast

from .models import (
    ClassInfo,
    ComplexityInfo,
    FileAnalysis,
    FromImport,
    FunctionInfo,
    Import,
    ImportInfo,
    Parameter,
)


class ComplexityCalculator(ast.NodeVisitor):
    """Calculate cyclomatic complexity and nesting depth"""

    def __init__(self):
        self.complexity = 1  # Base complexity
        self.current_depth = 0
        self.max_depth = 0

    def visit_If(self, node):
        self.complexity += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_For(self, node):
        self.complexity += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_While(self, node):
        self.complexity += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # and/or adds complexity
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


class FunctionCallExtractor(ast.NodeVisitor):
    """Extract function calls from a function body"""

    def __init__(self):
        self.calls: set[str] = set()

    def visit_Call(self, node):
        # Extract function name
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # For method calls like obj.method()
            self.calls.add(node.func.attr)
        self.generic_visit(node)


class ASTParser:
    """Parse Python code using AST"""

    def __init__(self, source_code: str, file_path: str = "<string>"):
        """
        Initialize parser

        Args:
            source_code: Python source code to parse
            file_path: Path to the file (for error messages)
        """
        self.source_code = source_code
        self.file_path = file_path
        self.source_lines = source_code.split('\n')

        try:
            self.tree = ast.parse(source_code, filename=file_path)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}") from e

    def parse(self) -> FileAnalysis:
        """
        Parse the entire file

        Returns:
            FileAnalysis object with all extracted information
        """
        functions = []
        classes = []
        imports = self._extract_imports()
        global_vars = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # Only top-level functions (not methods)
                if self._is_top_level(node):
                    functions.append(self._parse_function(node))

            elif isinstance(node, ast.AsyncFunctionDef):
                if self._is_top_level(node):
                    func_info = self._parse_function(node)
                    func_info.is_async = True
                    functions.append(func_info)

            elif isinstance(node, ast.ClassDef):
                if self._is_top_level(node):
                    classes.append(self._parse_class(node))

            elif isinstance(node, ast.Assign):
                # Global variables
                if self._is_top_level(node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            global_vars.append(target.id)

        return FileAnalysis(
            file_path=self.file_path,
            total_lines=len(self.source_lines),
            functions=functions,
            classes=classes,
            imports=imports,
            global_variables=global_vars
        )

    def _is_top_level(self, node: ast.AST) -> bool:
        """Check if node is at top level (not nested in class/function)"""
        # Walk up the tree to check if there's a class or function parent
        for parent in ast.walk(self.tree):
            if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if parent == node:
                    continue
                # Check if node is in parent's body
                if hasattr(parent, 'body') and node in ast.walk(ast.Module(body=parent.body)):
                    return False
        return True

    def _parse_function(self, node: ast.FunctionDef) -> FunctionInfo:
        """Parse a function definition"""
        # Extract parameters
        parameters = self._extract_parameters(node.args)

        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Calculate complexity
        complexity_calc = ComplexityCalculator()
        complexity_calc.visit(node)

        # Extract function calls
        call_extractor = FunctionCallExtractor()
        call_extractor.visit(node)

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(f"@{decorator.id}")
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                decorators.append(f"@{decorator.func.id}")
            else:
                decorators.append(ast.unparse(decorator))

        # Check if it's a method
        is_method = False
        is_classmethod = False
        is_staticmethod = False

        for dec in decorators:
            if dec == "@classmethod":
                is_classmethod = True
                is_method = True
            elif dec == "@staticmethod":
                is_staticmethod = True
                is_method = True

        return FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            end_line_number=node.end_lineno or node.lineno,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            complexity=complexity_calc.complexity,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            decorators=decorators,
            calls=sorted(call_extractor.calls)
        )

    def _extract_parameters(self, args: ast.arguments) -> list[Parameter]:
        """Extract function parameters"""
        parameters = []

        # Regular positional/keyword arguments
        for i, arg in enumerate(args.args):
            type_hint = None
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)

            default_value = None
            # Defaults are aligned to the end of args
            default_offset = len(args.args) - len(args.defaults)
            if i >= default_offset:
                default_idx = i - default_offset
                default_value = ast.unparse(args.defaults[default_idx])

            parameters.append(Parameter(
                name=arg.arg,
                type_hint=type_hint,
                default_value=default_value
            ))

        # *args
        if args.vararg:
            type_hint = None
            if args.vararg.annotation:
                type_hint = ast.unparse(args.vararg.annotation)
            parameters.append(Parameter(
                name=args.vararg.arg,
                type_hint=type_hint,
                is_vararg=True
            ))

        # Keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            type_hint = None
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)

            default_value = None
            if i < len(args.kw_defaults) and args.kw_defaults[i]:
                default_value = ast.unparse(args.kw_defaults[i])

            parameters.append(Parameter(
                name=arg.arg,
                type_hint=type_hint,
                default_value=default_value,
                is_kwonly=True
            ))

        # **kwargs
        if args.kwarg:
            type_hint = None
            if args.kwarg.annotation:
                type_hint = ast.unparse(args.kwarg.annotation)
            parameters.append(Parameter(
                name=args.kwarg.arg,
                type_hint=type_hint,
                is_kwarg=True
            ))

        return parameters

    def _parse_class(self, node: ast.ClassDef) -> ClassInfo:
        """Parse a class definition"""
        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            else:
                base_classes.append(ast.unparse(base))

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                func_info = self._parse_function(item)
                func_info.is_method = True
                methods.append(func_info)
            elif isinstance(item, ast.AsyncFunctionDef):
                func_info = self._parse_function(item)
                func_info.is_async = True
                func_info.is_method = True
                methods.append(func_info)

        # Extract class variables
        class_variables = []
        instance_variables = []

        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_variables.append(target.id)

        # Extract instance variables from __init__
        for method in methods:
            if method.name == "__init__":
                # Look for self.x = ... assignments
                for item in ast.walk(self.tree):
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Attribute):
                                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    instance_variables.append(target.attr)

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(f"@{decorator.id}")
            else:
                decorators.append(ast.unparse(decorator))

        return ClassInfo(
            name=node.name,
            line_number=node.lineno,
            end_line_number=node.end_lineno or node.lineno,
            base_classes=base_classes,
            docstring=docstring,
            methods=methods,
            class_variables=list(set(class_variables)),
            instance_variables=list(set(instance_variables)),
            decorators=decorators
        )

    def _extract_imports(self) -> ImportInfo:
        """Extract all import statements"""
        imports = []
        from_imports = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(Import(
                        module=alias.name,
                        alias=alias.asname
                    ))

            elif isinstance(node, ast.ImportFrom):
                names = []
                for alias in node.names:
                    names.append({
                        "name": alias.name,
                        "alias": alias.asname
                    })

                from_imports.append(FromImport(
                    module=node.module or "",
                    names=names
                ))

        return ImportInfo(
            imports=imports,
            from_imports=from_imports
        )

    def find_function(self, function_name: str) -> FunctionInfo | None:
        """Find a specific function by name"""
        analysis = self.parse()

        # Check top-level functions
        for func in analysis.functions:
            if func.name == function_name:
                return func

        # Check methods in classes
        for cls in analysis.classes:
            for method in cls.methods:
                if method.name == function_name:
                    return method

        return None

    def find_class(self, class_name: str) -> ClassInfo | None:
        """Find a specific class by name"""
        analysis = self.parse()

        for cls in analysis.classes:
            if cls.name == class_name:
                return cls

        return None

    def calculate_complexity(self, function_name: str) -> ComplexityInfo | None:
        """Calculate complexity metrics for a specific function"""
        func_info = self.find_function(function_name)
        if not func_info:
            return None

        # Find the function node
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    # Calculate lines of code
                    loc = (node.end_lineno or node.lineno) - node.lineno + 1

                    # Calculate max nesting depth
                    calc = ComplexityCalculator()
                    calc.visit(node)

                    return ComplexityInfo(
                        cyclomatic_complexity=func_info.complexity,
                        lines_of_code=loc,
                        max_nesting_depth=calc.max_depth,
                        parameter_count=len(func_info.parameters)
                    )

        return None
