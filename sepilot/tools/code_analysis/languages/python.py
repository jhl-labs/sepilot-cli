"""Python language handler for tree-sitter parsing."""

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


class PythonHandler:
    """Handler for Python AST extraction."""

    language = Language.PYTHON

    def extract_ast(
        self, file_path: str, content: str, tree: tree_sitter.Tree
    ) -> UnifiedAST:
        """Extract unified AST from Python parse tree."""
        ast = UnifiedAST(file_path=file_path, language=Language.PYTHON)
        _lines = content.split("\n")

        self._extract_imports(ast, tree.root_node, content)
        self._extract_functions(ast, tree.root_node, content, file_path)
        self._extract_classes(ast, tree.root_node, content, file_path)
        self._extract_variables(ast, tree.root_node, content, file_path)

        return ast

    def _get_node_text(self, node: tree_sitter.Node, content: str) -> str:
        """Get text content of a node."""
        return content[node.start_byte : node.end_byte]

    def _extract_imports(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str
    ) -> None:
        """Extract import statements."""
        for node in self._find_nodes(root, ["import_statement", "import_from_statement"]):
            if node.type == "import_statement":
                # import module
                for child in node.children:
                    if child.type == "dotted_name":
                        module = self._get_node_text(child, content)
                        ast.imports.append(
                            ImportInfo(module=module, import_type="import")
                        )
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node:
                            module = self._get_node_text(name_node, content)
                            alias = (
                                self._get_node_text(alias_node, content)
                                if alias_node
                                else None
                            )
                            ast.imports.append(
                                ImportInfo(module=module, alias=alias, import_type="import")
                            )

            elif node.type == "import_from_statement":
                # from module import ...
                module_node = node.child_by_field_name("module_name")
                module = self._get_node_text(module_node, content) if module_node else ""

                symbols = []
                is_star = False
                for child in node.children:
                    if child.type == "dotted_name" and child != module_node:
                        symbols.append(self._get_node_text(child, content))
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            symbols.append(self._get_node_text(name_node, content))
                    elif child.type == "wildcard_import":
                        is_star = True

                ast.imports.append(
                    ImportInfo(
                        module=module,
                        symbols=symbols,
                        is_star=is_star,
                        import_type="from",
                    )
                )

    def _extract_functions(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract function definitions."""
        for node in self._find_nodes(root, ["function_definition"]):
            # Skip methods (they'll be extracted with classes)
            if node.parent and node.parent.type == "block":
                parent_parent = node.parent.parent
                if parent_parent and parent_parent.type == "class_definition":
                    continue

            func = self._parse_function(node, content, file_path)
            if func:
                ast.functions.append(func)

    def _parse_function(
        self, node: tree_sitter.Node, content: str, file_path: str
    ) -> FunctionSymbol | None:
        """Parse a function definition node."""
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

        # Parse return type
        return_type = None
        return_node = node.child_by_field_name("return_type")
        if return_node:
            return_type = self._get_node_text(return_node, content)

        # Parse decorators
        decorators = []
        if node.parent:
            for sibling in node.parent.children:
                if sibling.type == "decorator" and sibling.end_byte <= node.start_byte:
                    decorator_text = self._get_node_text(sibling, content)
                    decorators.append(decorator_text.lstrip("@"))

        # Get docstring
        docstring = self._get_docstring(node, content)

        # Visibility
        visibility = Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC

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
            return_type=return_type,
            docstring=docstring,
            decorators=decorators,
            visibility=visibility,
            is_async=is_async,
            calls=calls,
            complexity=complexity,
        )

    def _parse_parameters(
        self, params_node: tree_sitter.Node, content: str
    ) -> list[Parameter]:
        """Parse function parameters."""
        params = []
        for child in params_node.children:
            if child.type in ("identifier", "typed_parameter", "default_parameter"):
                param = self._parse_single_param(child, content)
                if param and param.name not in ("self", "cls"):
                    params.append(param)
            elif child.type == "list_splat_pattern":
                name_node = child.children[0] if child.children else None
                if name_node:
                    params.append(
                        Parameter(
                            name=self._get_node_text(name_node, content),
                            is_variadic=True,
                        )
                    )
            elif child.type == "dictionary_splat_pattern":
                name_node = child.children[0] if child.children else None
                if name_node:
                    params.append(
                        Parameter(
                            name=self._get_node_text(name_node, content),
                            is_keyword=True,
                        )
                    )
        return params

    def _parse_single_param(
        self, node: tree_sitter.Node, content: str
    ) -> Parameter | None:
        """Parse a single parameter."""
        if node.type == "identifier":
            return Parameter(name=self._get_node_text(node, content))
        elif node.type == "typed_parameter":
            name_node = node.children[0] if node.children else None
            type_node = node.child_by_field_name("type")
            if name_node:
                return Parameter(
                    name=self._get_node_text(name_node, content),
                    type_annotation=(
                        self._get_node_text(type_node, content) if type_node else None
                    ),
                )
        elif node.type == "default_parameter":
            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")
            if name_node:
                return Parameter(
                    name=self._get_node_text(name_node, content),
                    default_value=(
                        self._get_node_text(value_node, content) if value_node else None
                    ),
                )
        return None

    def _extract_classes(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract class definitions."""
        for node in self._find_nodes(root, ["class_definition"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)

            # Parse base classes
            base_classes = []
            superclasses_node = node.child_by_field_name("superclasses")
            if superclasses_node:
                for child in superclasses_node.children:
                    if child.type in ("identifier", "attribute"):
                        base_classes.append(self._get_node_text(child, content))

            # Parse decorators
            decorators = []
            if node.parent:
                for sibling in node.parent.children:
                    if sibling.type == "decorator" and sibling.end_byte <= node.start_byte:
                        decorator_text = self._get_node_text(sibling, content)
                        decorators.append(decorator_text.lstrip("@"))

            # Get docstring
            docstring = self._get_docstring(node, content)

            # Extract methods
            methods = []
            body_node = node.child_by_field_name("body")
            if body_node:
                for child in self._find_nodes(body_node, ["function_definition"]):
                    method = self._parse_function(child, content, file_path)
                    if method:
                        method.kind = SymbolKind.METHOD
                        # Check for staticmethod/classmethod
                        for dec in method.decorators:
                            if "staticmethod" in dec:
                                method.is_static = True
                            elif "abstractmethod" in dec:
                                method.is_abstract = True
                        methods.append(method)

            # Extract class variables
            class_vars = []
            if body_node:
                for child in body_node.children:
                    if child.type == "expression_statement":
                        expr = child.children[0] if child.children else None
                        if expr and expr.type == "assignment":
                            left = expr.child_by_field_name("left")
                            if left and left.type == "identifier":
                                var_name = self._get_node_text(left, content)
                                class_vars.append(
                                    VariableSymbol(
                                        name=var_name,
                                        kind=SymbolKind.VARIABLE,
                                        location=Location(
                                            file_path=file_path,
                                            line_start=child.start_point[0] + 1,
                                            line_end=child.end_point[0] + 1,
                                        ),
                                    )
                                )

            is_abstract = any("ABC" in b or "Abstract" in b for b in base_classes)

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
                    class_variables=class_vars,
                    decorators=decorators,
                    docstring=docstring,
                    is_abstract=is_abstract,
                )
            )

    def _extract_variables(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract module-level variables."""
        for node in root.children:
            if node.type == "expression_statement":
                expr = node.children[0] if node.children else None
                if expr and expr.type == "assignment":
                    left = expr.child_by_field_name("left")
                    if left and left.type == "identifier":
                        name = self._get_node_text(left, content)
                        # Check for type annotation
                        type_node = expr.child_by_field_name("type")
                        type_ann = (
                            self._get_node_text(type_node, content) if type_node else None
                        )
                        # Determine if constant (ALL_CAPS)
                        is_constant = name.isupper()

                        ast.variables.append(
                            VariableSymbol(
                                name=name,
                                kind=SymbolKind.CONSTANT if is_constant else SymbolKind.VARIABLE,
                                location=Location(
                                    file_path=file_path,
                                    line_start=node.start_point[0] + 1,
                                    line_end=node.end_point[0] + 1,
                                ),
                                type_annotation=type_ann,
                                is_mutable=not is_constant,
                            )
                        )

    def _get_docstring(self, node: tree_sitter.Node, content: str) -> str | None:
        """Extract docstring from a function or class."""
        body = node.child_by_field_name("body")
        if not body or not body.children:
            return None

        first_stmt = body.children[0]
        if first_stmt.type == "expression_statement":
            expr = first_stmt.children[0] if first_stmt.children else None
            if expr and expr.type == "string":
                docstring = self._get_node_text(expr, content)
                # Remove quotes
                if docstring.startswith('"""') or docstring.startswith("'''"):
                    return docstring[3:-3].strip()
                elif docstring.startswith('"') or docstring.startswith("'"):
                    return docstring[1:-1].strip()
        return None

    def _extract_calls(self, node: tree_sitter.Node, content: str) -> list[str]:
        """Extract function calls from a function body."""
        calls = []
        for call_node in self._find_nodes(node, ["call"]):
            func_node = call_node.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    calls.append(self._get_node_text(func_node, content))
                elif func_node.type == "attribute":
                    # Get only the method name for attribute calls
                    attr_node = func_node.child_by_field_name("attribute")
                    if attr_node:
                        calls.append(self._get_node_text(attr_node, content))
        return list(set(calls))

    def _calculate_complexity(self, node: tree_sitter.Node) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        branch_nodes = [
            "if_statement",
            "elif_clause",
            "for_statement",
            "while_statement",
            "except_clause",
            "with_statement",
            "conditional_expression",
            "and",
            "or",
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
