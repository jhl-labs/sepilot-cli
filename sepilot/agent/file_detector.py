"""File Path Detection Module

Detects whether user requests contain explicit file paths or require
codebase exploration to identify relevant files.

Claude Code style: Before planning, detect if we need to explore the codebase.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileDetectionResult:
    """Result of file path detection in user prompt."""

    has_explicit_files: bool  # Explicit file paths found
    detected_files: list[str] = field(default_factory=list)  # Detected file paths
    needs_exploration: bool = False  # Whether codebase exploration is needed
    exploration_hints: list[str] = field(default_factory=list)  # Keywords/hints for search
    confidence: float = 0.5  # Detection confidence (0.0 - 1.0)
    project_type: str | None = None  # Detected project type (python, javascript, etc.)


class FilePathDetector:
    """Detects file paths and exploration needs in user prompts.

    Features:
    - Extracts explicit file paths (path/to/file.py, `file.py`, @file.py)
    - Detects exploration keywords ("find", "search", "class", "function")
    - Extracts exploration hints (function names, class names)
    - Auto-detects project type for file prioritization
    """

    # File extensions by language
    FILE_EXTENSIONS = (
        r"\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp|"
        r"md|json|yaml|yml|toml|xml|html|css|scss|sass|vue|svelte)"
    )

    # Explicit file path patterns
    EXPLICIT_PATH_PATTERNS = [
        rf"[\w/.-]+{FILE_EXTENSIONS}",  # path/to/file.py
        r"`[^`]+\.\w+`",  # `file.py`
        r'"[^"]+\.\w+"',  # "file.py"
        r"'[^']+\.\w+'",  # 'file.py'
        r"@[\w/.-]+\.\w+",  # @file.py (Claude Code style)
    ]

    # Keywords indicating exploration is needed
    EXPLORATION_KEYWORDS_EN = [
        "find",
        "search",
        "locate",
        "where is",
        "look for",
        "the function",
        "the class",
        "the method",
        "the file",
        "implement",
        "add",
        "create",
        "modify",
        "fix",
        "update",
        "refactor",
        "in the codebase",
        "in the project",
        "in this repo",
    ]

    EXPLORATION_KEYWORDS_KR = [
        "찾아",
        "검색",
        "어디",
        "위치",
        "파일",
        "함수",
        "클래스",
        "메서드",
        "구현",
        "추가",
        "생성",
        "수정",
        "고쳐",
        "버그",
        "리팩토링",
    ]

    # Project type detection files
    PROJECT_MARKERS = {
        "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
        "javascript": ["package.json", "package-lock.json", "yarn.lock"],
        "typescript": ["tsconfig.json"],
        "rust": ["Cargo.toml"],
        "go": ["go.mod", "go.sum"],
        "java": ["pom.xml", "build.gradle"],
    }

    # File patterns by project type
    FILE_PATTERNS_BY_TYPE = {
        "python": ["*.py"],
        "javascript": ["*.js", "*.jsx", "*.mjs", "*.cjs"],
        "typescript": ["*.ts", "*.tsx"],
        "rust": ["*.rs"],
        "go": ["*.go"],
        "java": ["*.java"],
    }

    def __init__(self, project_root: Path | None = None):
        """Initialize detector.

        Args:
            project_root: Project root directory. If None, uses cwd.
        """
        self.project_root = project_root or Path.cwd()
        self._project_type: str | None = None

    def detect(self, user_prompt: str) -> FileDetectionResult:
        """Detect file paths and exploration needs in user prompt.

        Args:
            user_prompt: The user's request text.

        Returns:
            FileDetectionResult with detection details.
        """
        prompt_lower = user_prompt.lower()

        # 1. Extract explicit file paths
        explicit_files = self._extract_explicit_files(user_prompt)

        # 2. Check for exploration keywords
        has_exploration_keywords = self._has_exploration_keywords(prompt_lower)

        # 3. Extract exploration hints (function/class names, concepts)
        hints = self._extract_exploration_hints(user_prompt)

        # 4. Detect project type
        project_type = self._detect_project_type()

        # 5. Determine if exploration is needed
        has_explicit = len(explicit_files) > 0
        needs_exploration = (not has_explicit and has_exploration_keywords) or (
            has_exploration_keywords and len(hints) > 0 and not has_explicit
        )

        # 6. Calculate confidence
        confidence = self._calculate_confidence(
            has_explicit, has_exploration_keywords, len(hints)
        )

        return FileDetectionResult(
            has_explicit_files=has_explicit,
            detected_files=explicit_files,
            needs_exploration=needs_exploration,
            exploration_hints=hints,
            confidence=confidence,
            project_type=project_type,
        )

    def _extract_explicit_files(self, text: str) -> list[str]:
        """Extract explicit file paths from text.

        Args:
            text: Text to search for file paths.

        Returns:
            List of detected file paths.
        """
        files: set[str] = set()

        for pattern in self.EXPLICIT_PATH_PATTERNS:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Clean up the match
                    cleaned = match.strip('`"\'@')
                    # Must have a file extension
                    if "." in cleaned and not cleaned.startswith("."):
                        # Filter out URLs
                        if not cleaned.startswith(("http://", "https://", "www.")):
                            files.add(cleaned)
            except re.error:
                continue

        return sorted(files)

    def _has_exploration_keywords(self, text_lower: str) -> bool:
        """Check if text contains exploration keywords.

        Args:
            text_lower: Lowercased text to check.

        Returns:
            True if exploration keywords found.
        """
        all_keywords = self.EXPLORATION_KEYWORDS_EN + self.EXPLORATION_KEYWORDS_KR
        return any(kw in text_lower for kw in all_keywords)

    def _extract_exploration_hints(self, text: str) -> list[str]:
        """Extract exploration hints (function/class names, concepts).

        Args:
            text: Text to extract hints from.

        Returns:
            List of extracted hints.
        """
        hints: set[str] = set()

        # Pattern 1: function/class/method references
        patterns = [
            r"(?:function|def|class|method|func)\s+[`\"']?(\w+)[`\"']?",
            r"[`\"'](\w{3,})[`\"']",  # Backtick/quoted names (min 3 chars)
            r"(?:the|a)\s+(\w+(?:Service|Controller|Manager|Handler|Tool|Agent|Node))",
            r"(\w+(?:Service|Controller|Manager|Handler|Tool|Agent|Node))",
        ]

        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Filter out common words
                    if len(match) >= 3 and match.lower() not in {
                        "the",
                        "and",
                        "for",
                        "that",
                        "this",
                        "with",
                        "from",
                        "have",
                        "been",
                    }:
                        hints.add(match)
            except re.error:
                continue

        # Pattern 2: CamelCase or snake_case identifiers
        camel_pattern = r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"
        snake_pattern = r"\b([a-z]+(?:_[a-z]+)+)\b"

        for pattern in [camel_pattern, snake_pattern]:
            try:
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) >= 5:  # Meaningful length
                        hints.add(match)
            except re.error:
                continue

        return list(hints)[:10]  # Limit to 10 hints

    def _detect_project_type(self) -> str | None:
        """Detect project type from marker files.

        Returns:
            Project type string or None if unknown.
        """
        if self._project_type is not None:
            return self._project_type

        for proj_type, markers in self.PROJECT_MARKERS.items():
            for marker in markers:
                if (self.project_root / marker).exists():
                    self._project_type = proj_type
                    return proj_type

        return None

    def get_priority_patterns(self) -> list[str]:
        """Get file patterns to prioritize based on project type.

        Returns:
            List of glob patterns (e.g., ["*.py"]).
        """
        project_type = self._detect_project_type()
        if project_type and project_type in self.FILE_PATTERNS_BY_TYPE:
            return self.FILE_PATTERNS_BY_TYPE[project_type]
        # Default to Python if unknown
        return ["*.py"]

    def _calculate_confidence(
        self, has_explicit: bool, has_keywords: bool, hint_count: int
    ) -> float:
        """Calculate detection confidence score.

        Args:
            has_explicit: Whether explicit files were found.
            has_keywords: Whether exploration keywords were found.
            hint_count: Number of exploration hints found.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if has_explicit:
            return 0.95
        if has_keywords and hint_count > 0:
            return 0.85
        if has_keywords:
            return 0.70
        if hint_count > 0:
            return 0.60
        return 0.50


def create_file_detector(project_root: Path | None = None) -> FilePathDetector:
    """Factory function to create a FilePathDetector.

    Args:
        project_root: Project root directory. If None, uses cwd.

    Returns:
        Configured FilePathDetector instance.
    """
    return FilePathDetector(project_root=project_root)
