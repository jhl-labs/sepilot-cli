"""Rust language handler for tree-sitter parsing."""

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


class RustHandler:
    """Handler for Rust AST extraction."""

    language = Language.RUST

    def extract_ast(
        self, file_path: str, content: str, tree: tree_sitter.Tree
    ) -> UnifiedAST:
        """Extract unified AST from Rust parse tree."""
        ast = UnifiedAST(file_path=file_path, language=Language.RUST)

        self._extract_imports(ast, tree.root_node, content)
        self._extract_functions(ast, tree.root_node, content, file_path)
        self._extract_structs(ast, tree.root_node, content, file_path)
        self._extract_traits(ast, tree.root_node, content, file_path)
        self._extract_enums(ast, tree.root_node, content, file_path)
        self._extract_impls(ast, tree.root_node, content, file_path)
        self._extract_variables(ast, tree.root_node, content, file_path)
        self._extract_modules(ast, tree.root_node, content)

        return ast

    def _get_node_text(self, node: tree_sitter.Node, content: str) -> str:
        """Get text content of a node."""
        return content[node.start_byte : node.end_byte]

    def _extract_imports(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str
    ) -> None:
        """Extract use statements."""
        for node in self._find_nodes(root, ["use_declaration"]):
            # Get the use tree
            for child in node.children:
                if child.type == "use_tree":
                    self._parse_use_tree(ast, child, content, "")
                elif child.type == "scoped_identifier":
                    module = self._get_node_text(child, content)
                    ast.imports.append(ImportInfo(module=module, import_type="use"))

    def _parse_use_tree(
        self, ast: UnifiedAST, node: tree_sitter.Node, content: str, prefix: str
    ) -> None:
        """Recursively parse use tree."""
        if node.type == "use_tree":
            for child in node.children:
                self._parse_use_tree(ast, child, content, prefix)
        elif node.type == "scoped_use_list":
            path = node.child_by_field_name("path")
            list_node = node.child_by_field_name("list")
            new_prefix = self._get_node_text(path, content) if path else prefix

            if list_node:
                for child in list_node.children:
                    if child.type in ("use_tree", "identifier", "use_as_clause"):
                        self._parse_use_tree(ast, child, content, new_prefix)
        elif node.type == "identifier":
            name = self._get_node_text(node, content)
            module = f"{prefix}::{name}" if prefix else name
            ast.imports.append(ImportInfo(module=module, import_type="use"))
        elif node.type == "use_as_clause":
            path = node.child_by_field_name("path")
            alias_node = node.child_by_field_name("alias")
            if path:
                module = self._get_node_text(path, content)
                alias = (
                    self._get_node_text(alias_node, content) if alias_node else None
                )
                full_module = f"{prefix}::{module}" if prefix else module
                ast.imports.append(
                    ImportInfo(module=full_module, alias=alias, import_type="use")
                )
        elif node.type == "use_wildcard":
            module = f"{prefix}::*" if prefix else "*"
            ast.imports.append(
                ImportInfo(module=module, is_star=True, import_type="use")
            )
        elif node.type == "scoped_identifier":
            module = self._get_node_text(node, content)
            full_module = f"{prefix}::{module}" if prefix else module
            ast.imports.append(ImportInfo(module=full_module, import_type="use"))

    def _extract_functions(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract function definitions."""
        for node in self._find_nodes(root, ["function_item"]):
            # Skip impl methods (handled separately)
            if node.parent and node.parent.type == "impl_item":
                continue

            func = self._parse_function(node, content, file_path)
            if func:
                ast.functions.append(func)

    def _parse_function(
        self, node: tree_sitter.Node, content: str, file_path: str
    ) -> FunctionSymbol | None:
        """Parse a function definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, content)

        # Check visibility
        visibility = Visibility.PRIVATE
        for child in node.children:
            if child.type == "visibility_modifier":
                vis_text = self._get_node_text(child, content)
                if "pub" in vis_text:
                    visibility = Visibility.PUBLIC
                break

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
            return_type = self._get_node_text(return_node, content).lstrip("-> ").strip()

        # Parse generic parameters
        generic_params = []
        type_params = node.child_by_field_name("type_parameters")
        if type_params:
            for child in type_params.children:
                if child.type == "type_identifier":
                    generic_params.append(self._get_node_text(child, content))

        # Extract function calls
        body_node = node.child_by_field_name("body")
        calls = self._extract_calls(body_node, content) if body_node else []

        # Calculate complexity
        complexity = self._calculate_complexity(body_node) if body_node else 1

        # Get docstring (/// or //! comments)
        docstring = self._get_doc_comment(node, content)

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
            if child.type == "parameter":
                pattern = child.child_by_field_name("pattern")
                type_node = child.child_by_field_name("type")

                if pattern:
                    name = self._get_node_text(pattern, content)
                    # Skip self parameters
                    if name in ("self", "&self", "&mut self", "mut self"):
                        continue

                    type_str = (
                        self._get_node_text(type_node, content) if type_node else None
                    )
                    params.append(
                        Parameter(name=name, type_annotation=type_str)
                    )
            elif child.type == "self_parameter":
                # Skip self
                continue
        return params

    def _extract_structs(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract struct definitions."""
        for node in self._find_nodes(root, ["struct_item"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)

            # Check visibility
            visibility = Visibility.PRIVATE
            for child in node.children:
                if child.type == "visibility_modifier":
                    if "pub" in self._get_node_text(child, content):
                        visibility = Visibility.PUBLIC
                    break

            # Parse generic parameters
            generic_params = []
            type_params = node.child_by_field_name("type_parameters")
            if type_params:
                for child in type_params.children:
                    if child.type == "type_identifier":
                        generic_params.append(self._get_node_text(child, content))

            # Extract fields as properties
            properties = []
            body = node.child_by_field_name("body")
            if body:
                for field in body.children:
                    if field.type == "field_declaration":
                        field_name_node = field.child_by_field_name("name")
                        field_type_node = field.child_by_field_name("type")

                        if field_name_node:
                            field_name = self._get_node_text(field_name_node, content)
                            field_type = (
                                self._get_node_text(field_type_node, content)
                                if field_type_node
                                else None
                            )

                            # Check field visibility
                            field_vis = Visibility.PRIVATE
                            for child in field.children:
                                if child.type == "visibility_modifier":
                                    if "pub" in self._get_node_text(child, content):
                                        field_vis = Visibility.PUBLIC
                                    break

                            properties.append(
                                VariableSymbol(
                                    name=field_name,
                                    kind=SymbolKind.PROPERTY,
                                    location=Location(
                                        file_path=file_path,
                                        line_start=field.start_point[0] + 1,
                                        line_end=field.end_point[0] + 1,
                                    ),
                                    type_annotation=field_type,
                                    visibility=field_vis,
                                )
                            )

            # Get docstring
            docstring = self._get_doc_comment(node, content)

            ast.classes.append(
                ClassSymbol(
                    name=name,
                    kind=SymbolKind.STRUCT,
                    location=Location(
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    ),
                    properties=properties,
                    docstring=docstring,
                    visibility=visibility,
                    generic_params=generic_params,
                )
            )

    def _extract_traits(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract trait definitions."""
        for node in self._find_nodes(root, ["trait_item"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)

            # Check visibility
            visibility = Visibility.PRIVATE
            for child in node.children:
                if child.type == "visibility_modifier":
                    if "pub" in self._get_node_text(child, content):
                        visibility = Visibility.PUBLIC
                    break

            # Parse generic parameters
            generic_params = []
            type_params = node.child_by_field_name("type_parameters")
            if type_params:
                for child in type_params.children:
                    if child.type == "type_identifier":
                        generic_params.append(self._get_node_text(child, content))

            # Extract method signatures
            methods = []
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "function_item":
                        method = self._parse_function(child, content, file_path)
                        if method:
                            method.kind = SymbolKind.METHOD
                            method.is_abstract = not self._has_body(child)
                            methods.append(method)

            # Parse super traits
            interfaces = []
            bounds = node.child_by_field_name("bounds")
            if bounds:
                for child in bounds.children:
                    if child.type == "type_identifier":
                        interfaces.append(self._get_node_text(child, content))

            # Get docstring
            docstring = self._get_doc_comment(node, content)

            ast.classes.append(
                ClassSymbol(
                    name=name,
                    kind=SymbolKind.TRAIT,
                    location=Location(
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    ),
                    methods=methods,
                    interfaces=interfaces,
                    docstring=docstring,
                    visibility=visibility,
                    generic_params=generic_params,
                )
            )

    def _has_body(self, func_node: tree_sitter.Node) -> bool:
        """Check if a function has a body (not just a signature)."""
        body = func_node.child_by_field_name("body")
        return body is not None

    def _extract_enums(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract enum definitions."""
        for node in self._find_nodes(root, ["enum_item"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)

            # Check visibility
            visibility = Visibility.PRIVATE
            for child in node.children:
                if child.type == "visibility_modifier":
                    if "pub" in self._get_node_text(child, content):
                        visibility = Visibility.PUBLIC
                    break

            # Parse generic parameters
            generic_params = []
            type_params = node.child_by_field_name("type_parameters")
            if type_params:
                for child in type_params.children:
                    if child.type == "type_identifier":
                        generic_params.append(self._get_node_text(child, content))

            # Extract variants as properties
            properties = []
            body = node.child_by_field_name("body")
            if body:
                for variant in body.children:
                    if variant.type == "enum_variant":
                        variant_name = variant.child_by_field_name("name")
                        if variant_name:
                            properties.append(
                                VariableSymbol(
                                    name=self._get_node_text(variant_name, content),
                                    kind=SymbolKind.CONSTANT,
                                    location=Location(
                                        file_path=file_path,
                                        line_start=variant.start_point[0] + 1,
                                        line_end=variant.end_point[0] + 1,
                                    ),
                                    is_mutable=False,
                                )
                            )

            # Get docstring
            docstring = self._get_doc_comment(node, content)

            ast.classes.append(
                ClassSymbol(
                    name=name,
                    kind=SymbolKind.ENUM,
                    location=Location(
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    ),
                    properties=properties,
                    docstring=docstring,
                    visibility=visibility,
                    generic_params=generic_params,
                )
            )

    def _extract_impls(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract impl blocks and attach methods to structs."""
        for node in self._find_nodes(root, ["impl_item"]):
            # Get the type being implemented
            type_node = node.child_by_field_name("type")
            if not type_node:
                continue

            type_name = self._get_node_text(type_node, content)

            # Get the trait being implemented (if any)
            trait_node = node.child_by_field_name("trait")
            trait_name = self._get_node_text(trait_node, content) if trait_node else None

            # Extract methods from impl block
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "function_item":
                        method = self._parse_function(child, content, file_path)
                        if method:
                            method.kind = SymbolKind.METHOD
                            # Store impl info in metadata
                            ast.metadata.setdefault("impls", []).append(
                                {
                                    "type": type_name,
                                    "trait": trait_name,
                                    "method": method.name,
                                }
                            )
                            # Also add to functions list for now
                            ast.functions.append(method)

    def _extract_variables(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract module-level constants and statics."""
        # Constants
        for node in self._find_nodes(root, ["const_item"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)
            type_node = node.child_by_field_name("type")
            value_node = node.child_by_field_name("value")

            # Check visibility
            visibility = Visibility.PRIVATE
            for child in node.children:
                if child.type == "visibility_modifier":
                    if "pub" in self._get_node_text(child, content):
                        visibility = Visibility.PUBLIC
                    break

            ast.variables.append(
                VariableSymbol(
                    name=name,
                    kind=SymbolKind.CONSTANT,
                    location=Location(
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    ),
                    type_annotation=(
                        self._get_node_text(type_node, content) if type_node else None
                    ),
                    initial_value=(
                        self._get_node_text(value_node, content) if value_node else None
                    ),
                    visibility=visibility,
                    is_mutable=False,
                )
            )

        # Statics
        for node in self._find_nodes(root, ["static_item"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)
            type_node = node.child_by_field_name("type")
            value_node = node.child_by_field_name("value")

            # Check if mutable
            is_mut = any(
                child.type == "mutable_specifier" for child in node.children
            )

            # Check visibility
            visibility = Visibility.PRIVATE
            for child in node.children:
                if child.type == "visibility_modifier":
                    if "pub" in self._get_node_text(child, content):
                        visibility = Visibility.PUBLIC
                    break

            ast.variables.append(
                VariableSymbol(
                    name=name,
                    kind=SymbolKind.VARIABLE,
                    location=Location(
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    ),
                    type_annotation=(
                        self._get_node_text(type_node, content) if type_node else None
                    ),
                    initial_value=(
                        self._get_node_text(value_node, content) if value_node else None
                    ),
                    visibility=visibility,
                    is_mutable=is_mut,
                )
            )

    def _extract_modules(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str
    ) -> None:
        """Extract module declarations."""
        for node in self._find_nodes(root, ["mod_item"]):
            name_node = node.child_by_field_name("name")
            if name_node:
                module_name = self._get_node_text(name_node, content)
                ast.metadata.setdefault("modules", []).append(module_name)

    def _get_doc_comment(
        self, node: tree_sitter.Node, content: str
    ) -> str | None:
        """Extract doc comments (/// or //!) preceding a node."""
        # Look at previous siblings for line comments
        lines = content.split("\n")
        start_line = node.start_point[0]

        doc_lines = []
        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("///") or line.startswith("//!"):
                doc_lines.insert(0, line[3:].strip())
            elif line.startswith("//"):
                continue  # Skip regular comments
            elif line == "":
                continue  # Skip empty lines
            else:
                break  # Stop at non-comment content

        return "\n".join(doc_lines) if doc_lines else None

    def _extract_calls(self, node: tree_sitter.Node, content: str) -> list[str]:
        """Extract function calls."""
        calls = []
        for call_node in self._find_nodes(node, ["call_expression"]):
            func = call_node.child_by_field_name("function")
            if func:
                if func.type == "identifier":
                    calls.append(self._get_node_text(func, content))
                elif func.type == "scoped_identifier":
                    # Get the last part of the path
                    name = func.child_by_field_name("name")
                    if name:
                        calls.append(self._get_node_text(name, content))
                elif func.type == "field_expression":
                    field = func.child_by_field_name("field")
                    if field:
                        calls.append(self._get_node_text(field, content))
        return list(set(calls))

    def _calculate_complexity(self, node: tree_sitter.Node) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        branch_nodes = [
            "if_expression",
            "match_arm",
            "for_expression",
            "while_expression",
            "loop_expression",
            "binary_expression",  # && and ||
        ]
        for child in self._find_nodes(node, branch_nodes):
            if child.type == "binary_expression":
                op = child.child_by_field_name("operator")
                if op:
                    op_text = self._get_node_text(op, node)
                    if op_text in ("&&", "||"):
                        complexity += 1
            else:
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
