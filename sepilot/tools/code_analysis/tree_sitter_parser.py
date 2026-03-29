"""Tree-sitter Parser - Multi-language AST parsing

This module provides a unified parser interface using tree-sitter for
parsing Python, JavaScript, TypeScript, Go, and Rust source code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .language_detector import get_language_detector
from .unified_ast import (
    ClassSymbol,
    FunctionSymbol,
    ImportInfo,
    Language,
    Location,
    Parameter,
    SymbolKind,
    UnifiedAST,
    Visibility,
)

if TYPE_CHECKING:
    import tree_sitter

logger = logging.getLogger(__name__)


class TreeSitterParser:
    """Tree-sitter based multi-language parser.

    Provides unified AST parsing for Python, JavaScript, TypeScript, Go, and Rust.
    Falls back gracefully when tree-sitter is not available.
    """

    SUPPORTED_LANGUAGES = [
        Language.PYTHON,
        Language.JAVASCRIPT,
        Language.TYPESCRIPT,
        Language.GO,
        Language.RUST,
    ]

    def __init__(self):
        """Initialize the parser with language support."""
        self._parsers: dict[Language, tree_sitter.Parser] = {}
        self._languages: dict[Language, tree_sitter.Language] = {}
        self._detector = get_language_detector()
        self._initialized = False
        self._available = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of tree-sitter parsers.

        Returns:
            True if tree-sitter is available and initialized
        """
        if self._initialized:
            return self._available

        self._initialized = True

        try:
            import tree_sitter
        except ImportError:
            logger.warning(
                "tree-sitter not installed. Install with: pip install sepilot[code-intelligence]"
            )
            self._available = False
            return False

        # Try to load each language
        language_modules = {
            Language.PYTHON: "tree_sitter_python",
            Language.JAVASCRIPT: "tree_sitter_javascript",
            Language.TYPESCRIPT: "tree_sitter_typescript",
            Language.GO: "tree_sitter_go",
            Language.RUST: "tree_sitter_rust",
        }

        for lang, module_name in language_modules.items():
            try:
                if lang == Language.TYPESCRIPT:
                    # TypeScript module has typescript and tsx
                    import tree_sitter_typescript

                    ts_lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
                    self._languages[lang] = ts_lang
                    parser = tree_sitter.Parser(ts_lang)
                    self._parsers[lang] = parser
                else:
                    module = __import__(module_name)
                    language_fn = module.language
                    ts_lang = tree_sitter.Language(language_fn())
                    self._languages[lang] = ts_lang
                    parser = tree_sitter.Parser(ts_lang)
                    self._parsers[lang] = parser
                logger.debug(f"Loaded tree-sitter parser for {lang.value}")
            except ImportError:
                logger.debug(f"tree-sitter-{lang.value} not installed")
            except Exception as e:
                logger.debug(f"Failed to load {lang.value} parser: {e}")

        self._available = len(self._parsers) > 0
        return self._available

    def is_available(self) -> bool:
        """Check if tree-sitter parsing is available."""
        return self._ensure_initialized()

    def get_supported_languages(self) -> list[Language]:
        """Get list of languages with working parsers."""
        self._ensure_initialized()
        return list(self._parsers.keys())

    def parse_file(self, file_path: str | Path, content: str | None = None) -> UnifiedAST:
        """Parse a source file and return unified AST.

        Args:
            file_path: Path to the file
            content: Optional file content (read from file if not provided)

        Returns:
            UnifiedAST with parsed symbols
        """
        path = Path(file_path)
        file_path_str = str(path)

        # Detect language
        if content is None:
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                return UnifiedAST(
                    file_path=file_path_str,
                    language=Language.UNKNOWN,
                    errors=[f"Failed to read file: {e}"],
                )

        language = self._detector.detect(file_path, content)

        if language == Language.UNKNOWN:
            return UnifiedAST(
                file_path=file_path_str,
                language=Language.UNKNOWN,
                errors=["Unknown language"],
            )

        # Check if we have a parser for this language
        if not self._ensure_initialized() or language not in self._parsers:
            return self._fallback_parse(file_path_str, content, language)

        try:
            parser = self._parsers[language]
            tree = parser.parse(content.encode("utf-8"))
            return self._extract_ast(file_path_str, content, tree, language)
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
            return self._fallback_parse(file_path_str, content, language)

    def _fallback_parse(
        self, file_path: str, content: str, language: Language
    ) -> UnifiedAST:
        """Fallback parsing when tree-sitter is not available.

        Uses regex-based pattern matching as a fallback.
        """
        ast = UnifiedAST(
            file_path=file_path,
            language=language,
            errors=["Using fallback parser (tree-sitter not available)"],
        )

        lines = content.split("\n")

        if language == Language.PYTHON:
            ast = self._fallback_parse_python(file_path, content, lines)
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            ast = self._fallback_parse_javascript(file_path, content, lines, language)
        elif language == Language.GO:
            ast = self._fallback_parse_go(file_path, content, lines)
        elif language == Language.RUST:
            ast = self._fallback_parse_rust(file_path, content, lines)

        return ast

    def _fallback_parse_python(
        self, file_path: str, content: str, lines: list[str]
    ) -> UnifiedAST:
        """Regex-based Python parsing fallback."""
        import re

        ast = UnifiedAST(file_path=file_path, language=Language.PYTHON)

        # Parse imports
        import_pattern = re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+)$")
        for _i, line in enumerate(lines):
            stripped = line.strip()
            if match := import_pattern.match(stripped):
                module = match.group(1) or match.group(2).split(",")[0].strip()
                symbols = []
                if match.group(1):
                    symbols = [s.strip() for s in match.group(2).split(",")]
                ast.imports.append(
                    ImportInfo(
                        module=module,
                        symbols=symbols,
                        import_type="from" if match.group(1) else "import",
                    )
                )

        # Parse functions
        func_pattern = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)\s*\((.*?)\).*?:")
        for i, line in enumerate(lines):
            if match := func_pattern.match(line):
                indent = len(match.group(1))
                is_async = bool(match.group(2))
                name = match.group(3)
                params_str = match.group(4)

                # Parse parameters
                params = []
                if params_str.strip():
                    for p in params_str.split(","):
                        p = p.strip()
                        if p and p != "self" and p != "cls":
                            param_name = p.split(":")[0].split("=")[0].strip()
                            if param_name:
                                params.append(Parameter(name=param_name))

                # Determine if method (inside class)
                kind = SymbolKind.METHOD if indent > 0 else SymbolKind.FUNCTION

                # Find end of function
                end_line = self._find_python_block_end(lines, i)

                ast.functions.append(
                    FunctionSymbol(
                        name=name,
                        kind=kind,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=end_line + 1,
                        ),
                        parameters=params,
                        is_async=is_async,
                        visibility=Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC,
                    )
                )

        # Parse classes
        class_pattern = re.compile(r"^class\s+(\w+)(?:\((.*?)\))?:")
        for i, line in enumerate(lines):
            if match := class_pattern.match(line.strip()):
                name = match.group(1)
                bases_str = match.group(2) or ""
                bases = [b.strip() for b in bases_str.split(",") if b.strip()]

                end_line = self._find_python_block_end(lines, i)

                ast.classes.append(
                    ClassSymbol(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=end_line + 1,
                        ),
                        base_classes=bases,
                    )
                )

        return ast

    def _find_python_block_end(self, lines: list[str], start: int) -> int:
        """Find the end of a Python block by indentation."""
        if start >= len(lines):
            return start

        # Get base indentation
        base_line = lines[start]
        base_indent = len(base_line) - len(base_line.lstrip())

        for i in range(start + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue

            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent:
                return i - 1

        return len(lines) - 1

    def _fallback_parse_javascript(
        self, file_path: str, content: str, lines: list[str], language: Language
    ) -> UnifiedAST:
        """Regex-based JavaScript/TypeScript parsing fallback."""
        import re

        ast = UnifiedAST(file_path=file_path, language=language)

        # Parse imports
        import_patterns = [
            re.compile(r"import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"import\s+\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*require\(['\"]([^'\"]+)['\"]\)"),
        ]

        for _i, line in enumerate(lines):
            for pattern in import_patterns:
                if match := pattern.search(line):
                    if len(match.groups()) == 3:
                        symbols = match.group(1)
                        module = match.group(3)
                        if symbols:
                            symbols = [s.strip() for s in symbols.split(",")]
                        else:
                            symbols = [match.group(2)] if match.group(2) else []
                    else:
                        symbols = [match.group(1)]
                        module = match.group(2)

                    ast.imports.append(
                        ImportInfo(module=module, symbols=symbols, import_type="import")
                    )

        # Parse functions
        func_patterns = [
            re.compile(r"(async\s+)?function\s+(\w+)\s*\("),
            re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(async\s+)?\("),
            re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(async\s+)?function"),
            re.compile(r"(\w+)\s*:\s*(async\s+)?function\s*\("),
        ]

        for i, line in enumerate(lines):
            for pattern in func_patterns:
                if match := pattern.search(line):
                    groups = match.groups()
                    if "function" in pattern.pattern:
                        is_async = bool(groups[0]) if groups[0] else False
                        name = groups[1] if len(groups) > 1 else groups[0]
                    else:
                        name = groups[0]
                        is_async = bool(groups[1]) if len(groups) > 1 else False

                    if name and name not in ("if", "for", "while", "switch"):
                        ast.functions.append(
                            FunctionSymbol(
                                name=name,
                                kind=SymbolKind.FUNCTION,
                                location=Location(
                                    file_path=file_path,
                                    line_start=i + 1,
                                    line_end=i + 1,
                                ),
                                is_async=is_async,
                            )
                        )
                        break

        # Parse classes
        class_pattern = re.compile(r"class\s+(\w+)(?:\s+extends\s+(\w+))?")
        for i, line in enumerate(lines):
            if match := class_pattern.search(line):
                name = match.group(1)
                base = match.group(2)
                ast.classes.append(
                    ClassSymbol(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                        ),
                        base_classes=[base] if base else [],
                    )
                )

        return ast

    def _fallback_parse_go(
        self, file_path: str, content: str, lines: list[str]
    ) -> UnifiedAST:
        """Regex-based Go parsing fallback."""
        import re

        ast = UnifiedAST(file_path=file_path, language=Language.GO)

        # Parse imports
        import_pattern = re.compile(r'import\s+(?:\(\s*)?["\']([^"\']+)["\']')
        in_import_block = False
        for _i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ("):
                in_import_block = True
                continue
            if in_import_block:
                if stripped == ")":
                    in_import_block = False
                    continue
                if match := re.search(r'["\']([^"\']+)["\']', stripped):
                    ast.imports.append(
                        ImportInfo(module=match.group(1), import_type="import")
                    )
            elif match := import_pattern.search(stripped):
                ast.imports.append(
                    ImportInfo(module=match.group(1), import_type="import")
                )

        # Parse functions
        func_pattern = re.compile(r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\(")
        for i, line in enumerate(lines):
            if match := func_pattern.search(line):
                _receiver_var = match.group(1)
                receiver_type = match.group(2)
                name = match.group(3)

                kind = SymbolKind.METHOD if receiver_type else SymbolKind.FUNCTION
                visibility = (
                    Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE
                )

                ast.functions.append(
                    FunctionSymbol(
                        name=name,
                        kind=kind,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                        ),
                        visibility=visibility,
                    )
                )

        # Parse structs
        struct_pattern = re.compile(r"type\s+(\w+)\s+struct\s*\{")
        for i, line in enumerate(lines):
            if match := struct_pattern.search(line):
                name = match.group(1)
                ast.classes.append(
                    ClassSymbol(
                        name=name,
                        kind=SymbolKind.STRUCT,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                        ),
                    )
                )

        # Parse interfaces
        interface_pattern = re.compile(r"type\s+(\w+)\s+interface\s*\{")
        for i, line in enumerate(lines):
            if match := interface_pattern.search(line):
                name = match.group(1)
                ast.classes.append(
                    ClassSymbol(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                        ),
                    )
                )

        return ast

    def _fallback_parse_rust(
        self, file_path: str, content: str, lines: list[str]
    ) -> UnifiedAST:
        """Regex-based Rust parsing fallback."""
        import re

        ast = UnifiedAST(file_path=file_path, language=Language.RUST)

        # Parse use statements
        use_pattern = re.compile(r"use\s+([^;]+);")
        for _i, line in enumerate(lines):
            if match := use_pattern.search(line):
                module = match.group(1).strip()
                ast.imports.append(ImportInfo(module=module, import_type="use"))

        # Parse functions
        func_pattern = re.compile(r"(pub\s+)?(async\s+)?fn\s+(\w+)")
        for i, line in enumerate(lines):
            if match := func_pattern.search(line):
                is_pub = bool(match.group(1))
                is_async = bool(match.group(2))
                name = match.group(3)

                ast.functions.append(
                    FunctionSymbol(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                        ),
                        is_async=is_async,
                        visibility=Visibility.PUBLIC if is_pub else Visibility.PRIVATE,
                    )
                )

        # Parse structs
        struct_pattern = re.compile(r"(pub\s+)?struct\s+(\w+)")
        for i, line in enumerate(lines):
            if match := struct_pattern.search(line):
                is_pub = bool(match.group(1))
                name = match.group(2)

                ast.classes.append(
                    ClassSymbol(
                        name=name,
                        kind=SymbolKind.STRUCT,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                        ),
                        visibility=Visibility.PUBLIC if is_pub else Visibility.PRIVATE,
                    )
                )

        # Parse traits
        trait_pattern = re.compile(r"(pub\s+)?trait\s+(\w+)")
        for i, line in enumerate(lines):
            if match := trait_pattern.search(line):
                is_pub = bool(match.group(1))
                name = match.group(2)

                ast.classes.append(
                    ClassSymbol(
                        name=name,
                        kind=SymbolKind.TRAIT,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                        ),
                        visibility=Visibility.PUBLIC if is_pub else Visibility.PRIVATE,
                    )
                )

        # Parse enums
        enum_pattern = re.compile(r"(pub\s+)?enum\s+(\w+)")
        for i, line in enumerate(lines):
            if match := enum_pattern.search(line):
                is_pub = bool(match.group(1))
                name = match.group(2)

                ast.classes.append(
                    ClassSymbol(
                        name=name,
                        kind=SymbolKind.ENUM,
                        location=Location(
                            file_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                        ),
                        visibility=Visibility.PUBLIC if is_pub else Visibility.PRIVATE,
                    )
                )

        return ast

    def _extract_ast(
        self,
        file_path: str,
        content: str,
        tree: tree_sitter.Tree,
        language: Language,
    ) -> UnifiedAST:
        """Extract unified AST from tree-sitter parse tree.

        Args:
            file_path: Source file path
            content: Source content
            tree: Tree-sitter parse tree
            language: Detected language

        Returns:
            UnifiedAST with extracted symbols
        """
        from .languages import get_language_handler

        handler = get_language_handler(language)
        if handler:
            return handler.extract_ast(file_path, content, tree)

        # Fallback if no handler
        return self._fallback_parse(file_path, content, language)

    def extract_symbols(self, file_path: str | Path) -> list[FunctionSymbol | ClassSymbol]:
        """Extract all symbols from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of function and class symbols
        """
        ast = self.parse_file(file_path)
        symbols: list[FunctionSymbol | ClassSymbol] = []
        symbols.extend(ast.functions)
        symbols.extend(ast.classes)
        return symbols

    def extract_imports(self, file_path: str | Path) -> list[ImportInfo]:
        """Extract all imports from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of import information
        """
        ast = self.parse_file(file_path)
        return ast.imports

    def extract_function_calls(
        self, file_path: str | Path, function_name: str | None = None
    ) -> list[str]:
        """Extract function calls from a file.

        Args:
            file_path: Path to the file
            function_name: Optional function to extract calls from

        Returns:
            List of called function names
        """
        ast = self.parse_file(file_path)

        if function_name:
            func = ast.get_function(function_name)
            if func:
                return func.calls
            return []

        # Collect all calls from all functions
        all_calls: list[str] = []
        for func in ast.functions:
            all_calls.extend(func.calls)
        return list(set(all_calls))


# Singleton instance
_parser: TreeSitterParser | None = None


def get_tree_sitter_parser() -> TreeSitterParser:
    """Get the singleton TreeSitterParser instance.

    Returns:
        TreeSitterParser instance
    """
    global _parser
    if _parser is None:
        _parser = TreeSitterParser()
    return _parser


def parse_file(file_path: str | Path, content: str | None = None) -> UnifiedAST:
    """Convenience function to parse a file.

    Args:
        file_path: Path to the file
        content: Optional file content

    Returns:
        UnifiedAST with parsed symbols
    """
    return get_tree_sitter_parser().parse_file(file_path, content)
