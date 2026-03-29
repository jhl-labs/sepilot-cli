"""TypeScript language handler for tree-sitter parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..unified_ast import (
    ClassSymbol,
    FunctionSymbol,
    Language,
    Location,
    Parameter,
    SymbolKind,
    UnifiedAST,
    VariableSymbol,
)
from .javascript import JavaScriptHandler

if TYPE_CHECKING:
    import tree_sitter


class TypeScriptHandler(JavaScriptHandler):
    """Handler for TypeScript AST extraction.

    Extends JavaScript handler with TypeScript-specific features.
    """

    language = Language.TYPESCRIPT

    def extract_ast(
        self, file_path: str, content: str, tree: tree_sitter.Tree
    ) -> UnifiedAST:
        """Extract unified AST from TypeScript parse tree."""
        # Get base JavaScript AST
        ast = super().extract_ast(file_path, content, tree)
        ast.language = Language.TYPESCRIPT

        # Add TypeScript-specific constructs
        self._extract_interfaces(ast, tree.root_node, content, file_path)
        self._extract_type_aliases(ast, tree.root_node, content, file_path)
        self._extract_enums(ast, tree.root_node, content, file_path)

        # Enhance function/class info with types
        self._enhance_with_types(ast, tree.root_node, content)

        return ast

    def _extract_interfaces(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract interface definitions."""
        for node in self._find_nodes(root, ["interface_declaration"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)

            # Parse extends
            extends = []
            extends_clause = None
            for child in node.children:
                if child.type == "extends_type_clause":
                    extends_clause = child
                    break

            if extends_clause:
                for child in extends_clause.children:
                    if child.type in ("type_identifier", "generic_type"):
                        extends.append(self._get_node_text(child, content))

            # Parse generic parameters
            generic_params = []
            type_params = node.child_by_field_name("type_parameters")
            if type_params:
                for child in type_params.children:
                    if child.type == "type_parameter":
                        param_name = child.child_by_field_name("name")
                        if param_name:
                            generic_params.append(self._get_node_text(param_name, content))

            # Extract method signatures
            methods = []
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type in ("method_signature", "property_signature"):
                        method = self._parse_interface_member(child, content, file_path)
                        if method:
                            methods.append(method)

            ast.classes.append(
                ClassSymbol(
                    name=name,
                    kind=SymbolKind.INTERFACE,
                    location=Location(
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                    ),
                    interfaces=extends,
                    methods=methods,
                    generic_params=generic_params,
                )
            )

    def _parse_interface_member(
        self, node: tree_sitter.Node, content: str, file_path: str
    ) -> FunctionSymbol | None:
        """Parse an interface method or property signature."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, content)

        # Parse parameters
        params = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            params = self._parse_ts_parameters(params_node, content)

        # Parse return type
        return_type = None
        type_node = node.child_by_field_name("return_type")
        if type_node:
            return_type = self._get_node_text(type_node, content).lstrip(": ")

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
            is_abstract=True,  # Interface methods are inherently abstract
        )

    def _parse_ts_parameters(
        self, params_node: tree_sitter.Node, content: str
    ) -> list[Parameter]:
        """Parse TypeScript function parameters with types."""
        params = []
        for child in params_node.children:
            if child.type == "required_parameter":
                pattern = child.child_by_field_name("pattern")
                type_node = child.child_by_field_name("type")
                if pattern:
                    params.append(
                        Parameter(
                            name=self._get_node_text(pattern, content),
                            type_annotation=(
                                self._get_node_text(type_node, content).lstrip(": ")
                                if type_node
                                else None
                            ),
                        )
                    )
            elif child.type == "optional_parameter":
                pattern = child.child_by_field_name("pattern")
                type_node = child.child_by_field_name("type")
                value_node = child.child_by_field_name("value")
                if pattern:
                    params.append(
                        Parameter(
                            name=self._get_node_text(pattern, content),
                            type_annotation=(
                                self._get_node_text(type_node, content).lstrip(": ")
                                if type_node
                                else None
                            ),
                            default_value=(
                                self._get_node_text(value_node, content)
                                if value_node
                                else None
                            ),
                        )
                    )
            elif child.type == "rest_pattern":
                name_node = child.children[0] if child.children else None
                if name_node:
                    params.append(
                        Parameter(
                            name=self._get_node_text(name_node, content),
                            is_variadic=True,
                        )
                    )
        return params

    def _extract_type_aliases(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract type alias definitions."""
        for node in self._find_nodes(root, ["type_alias_declaration"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)

            # Get type value
            value_node = node.child_by_field_name("value")
            type_value = self._get_node_text(value_node, content) if value_node else None

            ast.variables.append(
                VariableSymbol(
                    name=name,
                    kind=SymbolKind.TYPE_ALIAS,
                    location=Location(
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    ),
                    type_annotation=type_value,
                    is_mutable=False,
                )
            )

    def _extract_enums(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str, file_path: str
    ) -> None:
        """Extract enum definitions."""
        for node in self._find_nodes(root, ["enum_declaration"]):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = self._get_node_text(name_node, content)

            # Extract enum members as properties
            properties = []
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "enum_assignment":
                        member_name = child.child_by_field_name("name")
                        member_value = child.child_by_field_name("value")
                        if member_name:
                            properties.append(
                                VariableSymbol(
                                    name=self._get_node_text(member_name, content),
                                    kind=SymbolKind.CONSTANT,
                                    location=Location(
                                        file_path=file_path,
                                        line_start=child.start_point[0] + 1,
                                        line_end=child.end_point[0] + 1,
                                    ),
                                    initial_value=(
                                        self._get_node_text(member_value, content)
                                        if member_value
                                        else None
                                    ),
                                    is_mutable=False,
                                )
                            )

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
                )
            )

    def _enhance_with_types(
        self, ast: UnifiedAST, root: tree_sitter.Node, content: str
    ) -> None:
        """Enhance AST with TypeScript type information."""
        # Enhance function return types
        for func in ast.functions:
            if func.return_type is None:
                # Try to find and extract return type from the source
                for node in self._find_nodes(root, ["function_declaration", "arrow_function"]):
                    if self._matches_function(node, func.name, content):
                        return_type_node = node.child_by_field_name("return_type")
                        if return_type_node:
                            func.return_type = self._get_node_text(
                                return_type_node, content
                            ).lstrip(": ")
                        break

        # Enhance class method types
        for cls in ast.classes:
            for method in cls.methods:
                if method.return_type is None:
                    for node in self._find_nodes(root, ["method_definition"]):
                        if self._matches_method(node, method.name, content):
                            return_type_node = node.child_by_field_name("return_type")
                            if return_type_node:
                                method.return_type = self._get_node_text(
                                    return_type_node, content
                                ).lstrip(": ")
                            break

    def _matches_function(
        self, node: tree_sitter.Node, name: str, content: str
    ) -> bool:
        """Check if a node matches a function by name."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return self._get_node_text(name_node, content) == name
        return False

    def _matches_method(
        self, node: tree_sitter.Node, name: str, content: str
    ) -> bool:
        """Check if a node matches a method by name."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return self._get_node_text(name_node, content) == name
        return False
