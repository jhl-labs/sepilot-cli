"""Go language handler for tree-sitter parsing."""

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


class GoHandler:
    """Handler for Go AST extraction."""

    language = Language.GO

    def extract_ast(
        self, file_path: str, content: str, tree: tree_sitter.Tree
    ) -> UnifiedAST:
        """Extract unified AST from Go parse tree."""
        ast = UnifiedAST(file_path=file_path, language=Language.GO)

        self._extract_package(ast, tree.root_node, content)
        self._extract_imports(ast, tree.root_node, content)
        self._extract_functions(ast, tree.root_node, content, file_path)
        self._extract_structs(ast, tree.root_node, content, file_path)
        self._extract_interfaces(ast, tree.root_node, content, file_path)
        self._extract_variables(ast, tree.root_node, content, file_path)

        return ast

    def _get_node_text(self, node: tree_sitter.Node, content: str) -> str:
        """Get text content of a node."""
        return content[node.start_byte : node.end_byte]

    def _extract_package(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str
    ) -> None:
        """Extract package name."""
        for node in self._find_nodes(root, ["package_clause"]):
            for child in node.children:
                if child.type == "package_identifier":
                    ast.metadata["package"] = self._get_node_text(child, content)
                    break

    def _extract_imports(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str
    ) -> None:
        """Extract import statements."""
        for node in self._find_nodes(root, ["import_declaration"]):
            for child in node.children:
                if child.type == "import_spec":
                    self._parse_import_spec(ast, child, content)
                elif child.type == "import_spec_list":
                    for spec in child.children:
                        if spec.type == "import_spec":
                            self._parse_import_spec(ast, spec, content)

    def _parse_import_spec(
        self, ast: UnifiedAST, node: tree_sitter.Node, content: str
    ) -> None:
        """Parse a single import spec."""
        path_node = node.child_by_field_name("path")
        name_node = node.child_by_field_name("name")

        if path_node:
            module = self._get_node_text(path_node, content).strip('"')
            alias = self._get_node_text(name_node, content) if name_node else None

            # Check for dot import or blank import
            is_star = alias == "."
            if alias == "_":
                alias = None  # Side-effect import

            ast.imports.append(
                ImportInfo(
                    module=module,
                    alias=alias if alias != "." else None,
                    is_star=is_star,
                    import_type="import",
                )
            )

    def _extract_functions(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract function and method definitions."""
        for node in self._find_nodes(root, ["function_declaration", "method_declaration"]):
            func = self._parse_function(node, content, file_path)
            if func:
                ast.functions.append(func)

    def _parse_function(
        self, node: tree_sitter.Node, content: str, file_path: str
    ) -> FunctionSymbol | None:
        """Parse a function or method declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, content)

        # Determine visibility (Go: uppercase = exported)
        visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.INTERNAL

        # Check for receiver (method)
        receiver_node = node.child_by_field_name("receiver")
        kind = SymbolKind.METHOD if receiver_node else SymbolKind.FUNCTION

        # Parse parameters
        params = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            params = self._parse_parameters(params_node, content)

        # Parse return type
        return_type = None
        result_node = node.child_by_field_name("result")
        if result_node:
            return_type = self._get_node_text(result_node, content)

        # Extract function calls
        body_node = node.child_by_field_name("body")
        calls = self._extract_calls(body_node, content) if body_node else []

        # Calculate complexity
        complexity = self._calculate_complexity(body_node) if body_node else 1

        return FunctionSymbol(
            name=name,
            kind=kind,
            location=Location(
                file_path=file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                column_start=node.start_point[1],
                column_end=node.end_point[1],
            ),
            parameters=params,
            return_type=return_type,
            visibility=visibility,
            calls=calls,
            complexity=complexity,
        )

    def _parse_parameters(
        self, params_node: tree_sitter.Node, content: str
    ) -> list[Parameter]:
        """Parse function parameters."""
        params = []
        for child in params_node.children:
            if child.type == "parameter_declaration":
                names = []
                type_node = child.child_by_field_name("type")
                type_str = self._get_node_text(type_node, content) if type_node else None

                for name_child in child.children:
                    if name_child.type == "identifier":
                        names.append(self._get_node_text(name_child, content))

                for name in names:
                    # Check for variadic
                    is_variadic = type_str and type_str.startswith("...")
                    params.append(
                        Parameter(
                            name=name,
                            type_annotation=type_str,
                            is_variadic=is_variadic,
                        )
                    )
        return params

    def _extract_structs(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract struct definitions."""
        for node in self._find_nodes(root, ["type_declaration"]):
            for spec in node.children:
                if spec.type == "type_spec":
                    type_node = spec.child_by_field_name("type")
                    if type_node and type_node.type == "struct_type":
                        self._parse_struct(ast, spec, type_node, content, file_path)

    def _parse_struct(
        self,
        ast: UnifiedAST,
        spec_node: tree_sitter.Node,
        struct_node: tree_sitter.Node,
        content: str,
        file_path: str,
    ) -> None:
        """Parse a struct definition."""
        name_node = spec_node.child_by_field_name("name")
        if not name_node:
            return

        name = self._get_node_text(name_node, content)
        visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.INTERNAL

        # Extract fields as properties
        properties = []
        field_list = struct_node.child_by_field_name("body")
        if field_list:
            for child in field_list.children:
                if child.type == "field_declaration":
                    field_names = []
                    type_node = child.child_by_field_name("type")
                    type_str = (
                        self._get_node_text(type_node, content) if type_node else None
                    )

                    for field_child in child.children:
                        if field_child.type == "field_identifier":
                            field_names.append(
                                self._get_node_text(field_child, content)
                            )

                    for field_name in field_names:
                        field_vis = (
                            Visibility.PUBLIC
                            if field_name[0].isupper()
                            else Visibility.INTERNAL
                        )
                        properties.append(
                            VariableSymbol(
                                name=field_name,
                                kind=SymbolKind.PROPERTY,
                                location=Location(
                                    file_path=file_path,
                                    line_start=child.start_point[0] + 1,
                                    line_end=child.end_point[0] + 1,
                                ),
                                type_annotation=type_str,
                                visibility=field_vis,
                            )
                        )

        # Find methods for this struct
        methods = self._find_struct_methods(ast, name, content, file_path)

        ast.classes.append(
            ClassSymbol(
                name=name,
                kind=SymbolKind.STRUCT,
                location=Location(
                    file_path=file_path,
                    line_start=spec_node.start_point[0] + 1,
                    line_end=struct_node.end_point[0] + 1,
                ),
                properties=properties,
                methods=methods,
                visibility=visibility,
            )
        )

    def _find_struct_methods(
        self, ast: UnifiedAST, struct_name: str, content: str, file_path: str
    ) -> list[FunctionSymbol]:
        """Find methods that have a receiver of this struct type."""
        methods = []
        for func in ast.functions:
            if func.kind == SymbolKind.METHOD:
                # Already extracted, check if it belongs to this struct
                # This is simplified - would need receiver info
                pass
        return methods

    def _extract_interfaces(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract interface definitions."""
        for node in self._find_nodes(root, ["type_declaration"]):
            for spec in node.children:
                if spec.type == "type_spec":
                    type_node = spec.child_by_field_name("type")
                    if type_node and type_node.type == "interface_type":
                        self._parse_interface(ast, spec, type_node, content, file_path)

    def _parse_interface(
        self,
        ast: UnifiedAST,
        spec_node: tree_sitter.Node,
        interface_node: tree_sitter.Node,
        content: str,
        file_path: str,
    ) -> None:
        """Parse an interface definition."""
        name_node = spec_node.child_by_field_name("name")
        if not name_node:
            return

        name = self._get_node_text(name_node, content)
        visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.INTERNAL

        # Extract method signatures
        methods = []
        body = interface_node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "method_spec":
                    method = self._parse_method_spec(child, content, file_path)
                    if method:
                        methods.append(method)

        # Extract embedded interfaces
        interfaces = []
        if body:
            for child in body.children:
                if child.type == "type_identifier":
                    interfaces.append(self._get_node_text(child, content))

        ast.classes.append(
            ClassSymbol(
                name=name,
                kind=SymbolKind.INTERFACE,
                location=Location(
                    file_path=file_path,
                    line_start=spec_node.start_point[0] + 1,
                    line_end=interface_node.end_point[0] + 1,
                ),
                methods=methods,
                interfaces=interfaces,
                visibility=visibility,
            )
        )

    def _parse_method_spec(
        self, node: tree_sitter.Node, content: str, file_path: str
    ) -> FunctionSymbol | None:
        """Parse an interface method specification."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, content)

        # Parse parameters
        params = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            params = self._parse_parameters(params_node, content)

        # Parse return type
        return_type = None
        result_node = node.child_by_field_name("result")
        if result_node:
            return_type = self._get_node_text(result_node, content)

        return FunctionSymbol(
            name=name,
            kind=SymbolKind.METHOD,
            location=Location(
                file_path=file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
            ),
            parameters=params,
            return_type=return_type,
            is_abstract=True,
        )

    def _extract_variables(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract package-level variables and constants."""
        # Variables
        for node in self._find_nodes(root, ["var_declaration"]):
            for spec in node.children:
                if spec.type == "var_spec":
                    self._parse_var_spec(ast, spec, content, file_path, is_const=False)

        # Constants
        for node in self._find_nodes(root, ["const_declaration"]):
            for spec in node.children:
                if spec.type == "const_spec":
                    self._parse_var_spec(ast, spec, content, file_path, is_const=True)

    def _parse_var_spec(
        self,
        ast: UnifiedAST,
        node: tree_sitter.Node,
        content: str,
        file_path: str,
        is_const: bool,
    ) -> None:
        """Parse a variable or constant specification."""
        type_node = node.child_by_field_name("type")
        value_node = node.child_by_field_name("value")

        type_str = self._get_node_text(type_node, content) if type_node else None
        value_str = self._get_node_text(value_node, content) if value_node else None

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, content)
                visibility = (
                    Visibility.PUBLIC if name[0].isupper() else Visibility.INTERNAL
                )

                ast.variables.append(
                    VariableSymbol(
                        name=name,
                        kind=SymbolKind.CONSTANT if is_const else SymbolKind.VARIABLE,
                        location=Location(
                            file_path=file_path,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                        ),
                        type_annotation=type_str,
                        initial_value=value_str,
                        visibility=visibility,
                        is_mutable=not is_const,
                    )
                )

    def _extract_calls(self, node: tree_sitter.Node, content: str) -> list[str]:
        """Extract function calls."""
        calls = []
        for call_node in self._find_nodes(node, ["call_expression"]):
            func = call_node.child_by_field_name("function")
            if func:
                if func.type == "identifier":
                    calls.append(self._get_node_text(func, content))
                elif func.type == "selector_expression":
                    field = func.child_by_field_name("field")
                    if field:
                        calls.append(self._get_node_text(field, content))
        return list(set(calls))

    def _calculate_complexity(self, node: tree_sitter.Node) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        branch_nodes = [
            "if_statement",
            "for_statement",
            "switch_statement",
            "select_statement",
            "expression_case",
            "default_case",
            "type_case",
        ]
        for _child in self._find_nodes(node, branch_nodes):
            complexity += 1
        return complexity

    def _find_nodes(
        self, node: tree_sitter.Node, types: list[str]
    ) -> list[tree_sitter.Node]:
        """Find all nodes of given types recursively."""
        results = []
        if node.type in types:
            results.append(node)
        for child in node.children:
            results.extend(self._find_nodes(child, types))
        return results
