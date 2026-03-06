"""JavaScript language handler for tree-sitter parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..unified_ast import (
    ClassSymbol,
    FunctionSymbol,
    ImportInfo,
    Language,
    Location,
    Parameter,
    SymbolKind,
    UnifiedAST,
    VariableSymbol,
    Visibility,
)

if TYPE_CHECKING:
    import tree_sitter


class JavaScriptHandler:
    """Handler for JavaScript AST extraction."""

    language = Language.JAVASCRIPT

    def extract_ast(
        self, file_path: str, content: str, tree: "tree_sitter.Tree"
    ) -> UnifiedAST:
        """Extract unified AST from JavaScript parse tree."""
        ast = UnifiedAST(file_path=file_path, language=Language.JAVASCRIPT)

        self._extract_imports(ast, tree.root_node, content)
        self._extract_functions(ast, tree.root_node, content, file_path)
        self._extract_classes(ast, tree.root_node, content, file_path)
        self._extract_variables(ast, tree.root_node, content, file_path)
        self._extract_exports(ast, tree.root_node, content)

        return ast

    def _get_node_text(self, node: "tree_sitter.Node", content: str) -> str:
        """Get text content of a node."""
        return content[node.start_byte : node.end_byte]

    def _extract_imports(
        self, ast: UnifiedAST, root: "tree_sitter.Node", content: str
    ) -> None:
        """Extract import statements."""
        for node in self._find_nodes(root, ["import_statement"]):
            module = ""
            symbols = []
            is_default = False
            is_star = False
            alias = None

            source_node = node.child_by_field_name("source")
            if source_node:
                module = self._get_node_text(source_node, content).strip("'\"")

            for child in node.children:
                if child.type == "import_clause":
                    for clause_child in child.children:
                        if clause_child.type == "identifier":
                            # Default import
                            symbols.append(self._get_node_text(clause_child, content))
                            is_default = True
                        elif clause_child.type == "namespace_import":
                            # import * as name
                            is_star = True
                            name_node = clause_child.child_by_field_name("name")
                            if name_node:
                                alias = self._get_node_text(name_node, content)
                        elif clause_child.type == "named_imports":
                            # import { a, b, c }
                            for spec in clause_child.children:
                                if spec.type == "import_specifier":
                                    name_node = spec.child_by_field_name("name")
                                    if name_node:
                                        symbols.append(self._get_node_text(name_node, content))

            ast.imports.append(
                ImportInfo(
                    module=module,
                    symbols=symbols,
                    alias=alias,
                    is_default=is_default,
                    is_star=is_star,
                    import_type="import",
                )
            )

        # Handle require() calls
        for node in self._find_nodes(root, ["call_expression"]):
            func = node.child_by_field_name("function")
            if func and self._get_node_text(func, content) == "require":
                args = node.child_by_field_name("arguments")
                if args and args.children:
                    for arg in args.children:
                        if arg.type == "string":
                            module = self._get_node_text(arg, content).strip("'\"")
                            ast.imports.append(
                                ImportInfo(module=module, import_type="require")
                            )

    def _extract_functions(
        self, ast: UnifiedAST, root: "tree_sitter.Node", content: str, file_path: str
    ) -> None:
        """Extract function definitions."""
        # Function declarations
        for node in self._find_nodes(root, ["function_declaration"]):
            func = self._parse_function(node, content, file_path)
            if func:
                ast.functions.append(func)

        # Arrow functions assigned to variables
        for node in self._find_nodes(root, ["lexical_declaration", "variable_declaration"]):
            for declarator in self._find_nodes(node, ["variable_declarator"]):
                value = declarator.child_by_field_name("value")
                if value and value.type in ("arrow_function", "function"):
                    name_node = declarator.child_by_field_name("name")
                    if name_node:
                        func = self._parse_arrow_function(
                            name_node, value, content, file_path
                        )
                        if func:
                            ast.functions.append(func)

    def _parse_function(
        self, node: "tree_sitter.Node", content: str, file_path: str
    ) -> FunctionSymbol | None:
        """Parse a function declaration node."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, content)

        # Check if async
        is_async = any(
            child.type == "async" for child in node.children if hasattr(child, "type")
        )

        # Parse parameters
        params = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            params = self._parse_parameters(params_node, content)

        # Extract function calls
        calls = self._extract_calls(node, content)

        # Calculate complexity
        complexity = self._calculate_complexity(node)

        return FunctionSymbol(
            name=name,
            kind=SymbolKind.FUNCTION,
            location=Location(
                file_path=file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                column_start=node.start_point[1],
                column_end=node.end_point[1],
            ),
            parameters=params,
            is_async=is_async,
            calls=calls,
            complexity=complexity,
        )

    def _parse_arrow_function(
        self,
        name_node: "tree_sitter.Node",
        func_node: "tree_sitter.Node",
        content: str,
        file_path: str,
    ) -> FunctionSymbol | None:
        """Parse an arrow function assigned to a variable."""
        name = self._get_node_text(name_node, content)

        # Check if async
        is_async = any(
            child.type == "async"
            for child in func_node.children
            if hasattr(child, "type")
        )

        # Parse parameters
        params = []
        params_node = func_node.child_by_field_name("parameters")
        if params_node:
            params = self._parse_parameters(params_node, content)
        else:
            # Single parameter without parens
            param_node = func_node.child_by_field_name("parameter")
            if param_node:
                params = [Parameter(name=self._get_node_text(param_node, content))]

        # Extract function calls
        calls = self._extract_calls(func_node, content)

        return FunctionSymbol(
            name=name,
            kind=SymbolKind.FUNCTION,
            location=Location(
                file_path=file_path,
                line_start=name_node.start_point[0] + 1,
                line_end=func_node.end_point[0] + 1,
                column_start=name_node.start_point[1],
                column_end=func_node.end_point[1],
            ),
            parameters=params,
            is_async=is_async,
            calls=calls,
        )

    def _parse_parameters(
        self, params_node: "tree_sitter.Node", content: str
    ) -> list[Parameter]:
        """Parse function parameters."""
        params = []
        for child in params_node.children:
            if child.type == "identifier":
                params.append(Parameter(name=self._get_node_text(child, content)))
            elif child.type == "assignment_pattern":
                # Default parameter
                left = child.child_by_field_name("left")
                right = child.child_by_field_name("right")
                if left:
                    params.append(
                        Parameter(
                            name=self._get_node_text(left, content),
                            default_value=(
                                self._get_node_text(right, content) if right else None
                            ),
                        )
                    )
            elif child.type == "rest_pattern":
                # ...args
                name_node = child.children[0] if child.children else None
                if name_node:
                    params.append(
                        Parameter(
                            name=self._get_node_text(name_node, content),
                            is_variadic=True,
                        )
                    )
            elif child.type == "object_pattern" or child.type == "array_pattern":
                # Destructuring
                params.append(Parameter(name=self._get_node_text(child, content)))
        return params

    def _extract_classes(
        self, ast: UnifiedAST, root: "tree_sitter.Node", content: str, file_path: str
    ) -> None:
        """Extract class definitions."""
        for node in self._find_nodes(root, ["class_declaration", "class"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)

            # Parse extends
            base_classes = []
            heritage = node.child_by_field_name("heritage")
            if heritage:
                for child in heritage.children:
                    if child.type == "extends_clause":
                        for extends_child in child.children:
                            if extends_child.type == "identifier":
                                base_classes.append(
                                    self._get_node_text(extends_child, content)
                                )

            # Extract methods
            methods = []
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "method_definition":
                        method = self._parse_method(child, content, file_path)
                        if method:
                            methods.append(method)

            ast.classes.append(
                ClassSymbol(
                    name=name,
                    kind=SymbolKind.CLASS,
                    location=Location(
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                    ),
                    base_classes=base_classes,
                    methods=methods,
                )
            )

    def _parse_method(
        self, node: "tree_sitter.Node", content: str, file_path: str
    ) -> FunctionSymbol | None:
        """Parse a class method."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, content)

        # Check modifiers
        is_async = False
        is_static = False
        for child in node.children:
            if child.type == "async":
                is_async = True
            elif child.type == "static":
                is_static = True

        # Parse parameters
        params = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            params = self._parse_parameters(params_node, content)

        # Visibility (private with #)
        visibility = Visibility.PRIVATE if name.startswith("#") else Visibility.PUBLIC

        return FunctionSymbol(
            name=name,
            kind=SymbolKind.METHOD,
            location=Location(
                file_path=file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
            ),
            parameters=params,
            visibility=visibility,
            is_async=is_async,
            is_static=is_static,
        )

    def _extract_variables(
        self, ast: UnifiedAST, root: "tree_sitter.Node", content: str, file_path: str
    ) -> None:
        """Extract module-level variables."""
        for node in root.children:
            if node.type in ("lexical_declaration", "variable_declaration"):
                is_const = "const" in self._get_node_text(node, content).split()[0]
                for declarator in self._find_nodes(node, ["variable_declarator"]):
                    name_node = declarator.child_by_field_name("name")
                    value_node = declarator.child_by_field_name("value")

                    # Skip if value is a function (already extracted)
                    if value_node and value_node.type in ("arrow_function", "function"):
                        continue

                    if name_node and name_node.type == "identifier":
                        name = self._get_node_text(name_node, content)
                        ast.variables.append(
                            VariableSymbol(
                                name=name,
                                kind=SymbolKind.CONSTANT if is_const else SymbolKind.VARIABLE,
                                location=Location(
                                    file_path=file_path,
                                    line_start=node.start_point[0] + 1,
                                    line_end=node.end_point[0] + 1,
                                ),
                                is_mutable=not is_const,
                            )
                        )

    def _extract_exports(
        self, ast: UnifiedAST, root: "tree_sitter.Node", content: str
    ) -> None:
        """Extract exports."""
        for node in self._find_nodes(root, ["export_statement"]):
            for child in node.children:
                if child.type == "identifier":
                    ast.exports.append(self._get_node_text(child, content))
                elif child.type == "export_clause":
                    for spec in child.children:
                        if spec.type == "export_specifier":
                            name_node = spec.child_by_field_name("name")
                            if name_node:
                                ast.exports.append(self._get_node_text(name_node, content))

    def _extract_calls(self, node: "tree_sitter.Node", content: str) -> list[str]:
        """Extract function calls."""
        calls = []
        for call_node in self._find_nodes(node, ["call_expression"]):
            func = call_node.child_by_field_name("function")
            if func:
                if func.type == "identifier":
                    calls.append(self._get_node_text(func, content))
                elif func.type == "member_expression":
                    prop = func.child_by_field_name("property")
                    if prop:
                        calls.append(self._get_node_text(prop, content))
        return list(set(calls))

    def _calculate_complexity(self, node: "tree_sitter.Node") -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        branch_nodes = [
            "if_statement",
            "for_statement",
            "for_in_statement",
            "while_statement",
            "do_statement",
            "switch_case",
            "catch_clause",
            "ternary_expression",
            "binary_expression",  # && and ||
        ]
        for child in self._find_nodes(node, branch_nodes):
            if child.type == "binary_expression":
                op = child.child_by_field_name("operator")
                if op and self._get_node_text(op, node) in ("&&", "||"):
                    complexity += 1
            else:
                complexity += 1
        return complexity

    def _find_nodes(
        self, node: "tree_sitter.Node", types: list[str]
    ) -> list["tree_sitter.Node"]:
        """Find all nodes of given types recursively."""
        results = []
        if node.type in types:
            results.append(node)
        for child in node.children:
            results.extend(self._find_nodes(child, types))
        return results
