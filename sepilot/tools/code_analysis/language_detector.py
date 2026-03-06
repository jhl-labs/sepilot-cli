"""Language Detector - Automatic programming language detection

This module provides automatic detection of programming languages from file paths
and content. It supports Python, JavaScript, TypeScript, Go, and Rust.
"""

import re
from pathlib import Path

from .unified_ast import Language


# File extension to language mapping
EXTENSION_MAP: dict[str, Language] = {
    # Python
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".pyw": Language.PYTHON,
    # JavaScript
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    # TypeScript
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".mts": Language.TYPESCRIPT,
    ".cts": Language.TYPESCRIPT,
    # Go
    ".go": Language.GO,
    # Rust
    ".rs": Language.RUST,
}


# Content patterns for language detection (fallback)
CONTENT_PATTERNS: dict[Language, list[re.Pattern]] = {
    Language.PYTHON: [
        re.compile(r"^#!/.*python", re.MULTILINE),
        re.compile(r"^import\s+\w+", re.MULTILINE),
        re.compile(r"^from\s+\w+\s+import", re.MULTILINE),
        re.compile(r"^def\s+\w+\s*\(", re.MULTILINE),
        re.compile(r"^class\s+\w+.*:", re.MULTILINE),
    ],
    Language.JAVASCRIPT: [
        re.compile(r"^#!/.*node", re.MULTILINE),
        re.compile(r"const\s+\w+\s*=", re.MULTILINE),
        re.compile(r"let\s+\w+\s*=", re.MULTILINE),
        re.compile(r"function\s+\w+\s*\(", re.MULTILINE),
        re.compile(r"require\s*\(['\"]", re.MULTILINE),
        re.compile(r"module\.exports\s*=", re.MULTILINE),
    ],
    Language.TYPESCRIPT: [
        re.compile(r":\s*(string|number|boolean|any)\b"),
        re.compile(r"interface\s+\w+\s*\{"),
        re.compile(r"type\s+\w+\s*="),
        re.compile(r"<\w+>\s*\("),  # Generic function calls
        re.compile(r"as\s+(string|number|boolean|any)\b"),
    ],
    Language.GO: [
        re.compile(r"^package\s+\w+", re.MULTILINE),
        re.compile(r"^func\s+(\(\w+\s+\*?\w+\)\s+)?\w+\s*\(", re.MULTILINE),
        re.compile(r"^import\s+\(", re.MULTILINE),
        re.compile(r"^type\s+\w+\s+struct\s*\{", re.MULTILINE),
        re.compile(r":="),  # Short variable declaration
    ],
    Language.RUST: [
        re.compile(r"^fn\s+\w+\s*(<.*>)?\s*\(", re.MULTILINE),
        re.compile(r"^use\s+\w+::", re.MULTILINE),
        re.compile(r"^mod\s+\w+", re.MULTILINE),
        re.compile(r"^struct\s+\w+", re.MULTILINE),
        re.compile(r"^impl\s+(<.*>)?\s*\w+", re.MULTILINE),
        re.compile(r"let\s+mut\s+\w+"),
        re.compile(r"->\s*\w+"),  # Return type annotation
    ],
}


# Project marker files for language detection
PROJECT_MARKERS: dict[str, Language] = {
    # Python
    "pyproject.toml": Language.PYTHON,
    "setup.py": Language.PYTHON,
    "requirements.txt": Language.PYTHON,
    "Pipfile": Language.PYTHON,
    "poetry.lock": Language.PYTHON,
    # JavaScript/TypeScript
    "package.json": Language.JAVASCRIPT,  # Could be TS too
    "tsconfig.json": Language.TYPESCRIPT,
    "jsconfig.json": Language.JAVASCRIPT,
    # Go
    "go.mod": Language.GO,
    "go.sum": Language.GO,
    # Rust
    "Cargo.toml": Language.RUST,
    "Cargo.lock": Language.RUST,
}


class LanguageDetector:
    """Detects programming language from file path and content."""

    def __init__(self):
        """Initialize the language detector."""
        self._project_cache: dict[str, Language] = {}

    def detect_from_path(self, file_path: str | Path) -> Language:
        """Detect language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected Language enum value
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        return EXTENSION_MAP.get(extension, Language.UNKNOWN)

    def detect_from_content(self, content: str, hint_extension: str | None = None) -> Language:
        """Detect language from file content.

        Args:
            content: File content to analyze
            hint_extension: Optional file extension hint

        Returns:
            Detected Language enum value
        """
        # If we have an extension hint, prioritize that
        if hint_extension:
            lang = EXTENSION_MAP.get(hint_extension.lower(), None)
            if lang and lang != Language.UNKNOWN:
                return lang

        # Check content patterns
        scores: dict[Language, int] = {lang: 0 for lang in Language}

        for language, patterns in CONTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(content):
                    scores[language] += 1

        # Find language with highest score
        best_lang = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_lang]

        # Require at least 2 pattern matches for confidence
        if best_score >= 2:
            return best_lang

        return Language.UNKNOWN

    def detect(self, file_path: str | Path, content: str | None = None) -> Language:
        """Detect language from path and optionally content.

        This is the main detection method that combines path and content analysis.

        Args:
            file_path: Path to the file
            content: Optional file content for more accurate detection

        Returns:
            Detected Language enum value
        """
        # First try extension-based detection
        lang = self.detect_from_path(file_path)

        if lang != Language.UNKNOWN:
            return lang

        # Fallback to content-based detection if content provided
        if content:
            path = Path(file_path)
            return self.detect_from_content(content, path.suffix)

        return Language.UNKNOWN

    def detect_project_language(self, project_root: str | Path) -> Language | None:
        """Detect the primary language of a project.

        Args:
            project_root: Root directory of the project

        Returns:
            Primary Language or None if cannot determine
        """
        root = Path(project_root)
        root_str = str(root.resolve())

        # Check cache
        if root_str in self._project_cache:
            return self._project_cache[root_str]

        # Check for project marker files
        for marker, lang in PROJECT_MARKERS.items():
            if (root / marker).exists():
                self._project_cache[root_str] = lang
                return lang

        # Count files by language
        lang_counts: dict[Language, int] = {lang: 0 for lang in Language}

        try:
            for file in root.rglob("*"):
                if file.is_file():
                    lang = self.detect_from_path(file)
                    if lang != Language.UNKNOWN:
                        lang_counts[lang] += 1
        except (PermissionError, OSError):
            pass

        # Find most common language
        if any(lang_counts.values()):
            primary_lang = max(lang_counts, key=lang_counts.get)  # type: ignore
            if lang_counts[primary_lang] > 0:
                self._project_cache[root_str] = primary_lang
                return primary_lang

        return None

    def is_supported(self, file_path: str | Path) -> bool:
        """Check if a file is in a supported language.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is in a supported language
        """
        lang = self.detect_from_path(file_path)
        return lang != Language.UNKNOWN

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions (with dots)
        """
        return list(EXTENSION_MAP.keys())

    def clear_cache(self) -> None:
        """Clear the project language cache."""
        self._project_cache.clear()


# Singleton instance
_detector: LanguageDetector | None = None


def get_language_detector() -> LanguageDetector:
    """Get the singleton LanguageDetector instance.

    Returns:
        LanguageDetector instance
    """
    global _detector
    if _detector is None:
        _detector = LanguageDetector()
    return _detector


def detect_language(file_path: str | Path, content: str | None = None) -> Language:
    """Convenience function to detect language.

    Args:
        file_path: Path to the file
        content: Optional file content

    Returns:
        Detected Language enum value
    """
    return get_language_detector().detect(file_path, content)


def is_supported_language(file_path: str | Path) -> bool:
    """Check if file is in a supported language.

    Args:
        file_path: Path to the file

    Returns:
        True if supported
    """
    return get_language_detector().is_supported(file_path)
