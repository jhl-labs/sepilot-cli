"""Code analysis tools using AST parsing.

Provides both legacy Python-only AST parsing and new multi-language
tree-sitter based parsing.
"""

from sepilot.tools.code_analysis.analyzer import CodeAnalyzer
from sepilot.tools.code_analysis.ast_parser import (
    ASTParser,
    ComplexityCalculator,
    FunctionCallExtractor,
)
from sepilot.tools.code_analysis.models import (
    ClassInfo,
    ComplexityInfo,
    FileAnalysis,
    FromImport,
    FunctionInfo,
    Import,
    ImportInfo,
    Parameter,
    Reference,
)

# New multi-language support
from sepilot.tools.code_analysis.unified_ast import (
    ClassSymbol,
    FunctionSymbol,
    Language,
    Location,
    SymbolKind,
    UnifiedAST,
    VariableSymbol,
    Visibility,
)
from sepilot.tools.code_analysis.language_detector import (
    LanguageDetector,
    detect_language,
    get_language_detector,
    is_supported_language,
)
from sepilot.tools.code_analysis.tree_sitter_parser import (
    TreeSitterParser,
    get_tree_sitter_parser,
    parse_file,
)

__all__ = [
    # Legacy Models
    'Parameter',
    'FunctionInfo',
    'ClassInfo',
    'Import',
    'FromImport',
    'ImportInfo',
    'ComplexityInfo',
    'FileAnalysis',
    'Reference',
    # Legacy Parsers
    'ComplexityCalculator',
    'FunctionCallExtractor',
    'ASTParser',
    # Legacy Analyzer
    'CodeAnalyzer',
    # New Unified AST
    'Language',
    'SymbolKind',
    'Visibility',
    'Location',
    'FunctionSymbol',
    'ClassSymbol',
    'VariableSymbol',
    'UnifiedAST',
    # New Language Detection
    'LanguageDetector',
    'detect_language',
    'get_language_detector',
    'is_supported_language',
    # New Tree-sitter Parser
    'TreeSitterParser',
    'get_tree_sitter_parser',
    'parse_file',
]
