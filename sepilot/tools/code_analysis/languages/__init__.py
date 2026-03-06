"""Language-specific handlers for tree-sitter parsing.

This module provides language-specific AST extraction handlers
for Python, JavaScript, TypeScript, Go, and Rust.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import tree_sitter

from ..unified_ast import Language, UnifiedAST


class LanguageHandler(Protocol):
    """Protocol for language-specific handlers."""

    language: Language

    def extract_ast(
        self, file_path: str, content: str, tree: "tree_sitter.Tree"
    ) -> UnifiedAST:
        """Extract unified AST from tree-sitter parse tree."""
        ...


# Registry of language handlers
_handlers: dict[Language, LanguageHandler] = {}


def register_handler(handler: LanguageHandler) -> None:
    """Register a language handler."""
    _handlers[handler.language] = handler


def get_language_handler(language: Language) -> LanguageHandler | None:
    """Get handler for a language."""
    return _handlers.get(language)


def get_all_handlers() -> dict[Language, LanguageHandler]:
    """Get all registered handlers."""
    return _handlers.copy()


# Import and register handlers
def _init_handlers() -> None:
    """Initialize and register all handlers."""
    from .golang import GoHandler
    from .javascript import JavaScriptHandler
    from .python import PythonHandler
    from .rust import RustHandler
    from .typescript import TypeScriptHandler

    register_handler(PythonHandler())
    register_handler(JavaScriptHandler())
    register_handler(TypeScriptHandler())
    register_handler(GoHandler())
    register_handler(RustHandler())


# Auto-initialize on import
try:
    _init_handlers()
except ImportError:
    # Handlers may not be available yet during initial setup
    pass
