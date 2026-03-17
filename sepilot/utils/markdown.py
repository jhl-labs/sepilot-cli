"""Shared markdown/frontmatter/glob utilities.

Consolidates commonly used helpers that were duplicated across
instructions_loader, rules_loader, and markdown_skill modules.
"""

import fnmatch
from pathlib import Path
from typing import Any


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Returns:
        Tuple of (metadata dict, body content)
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        import yaml
        metadata = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
        return metadata, body
    except Exception:
        return {}, content


def glob_match(path: str, pattern: str) -> bool:
    """Match path against glob pattern with ** support.

    Handles ** (zero or more directories) correctly on all Python versions
    (including 3.10 where Path.match doesn't fully support **).

    Args:
        path: File path to match
        pattern: Glob pattern (e.g. "src/**/*.py")

    Returns:
        True if the path matches the pattern
    """
    if not path or not pattern:
        return False

    # Try PurePath.match first (handles ** properly in 3.12+)
    if Path(path).match(pattern):
        return True
    # Simple fnmatch for non-** patterns
    if "**" not in pattern and fnmatch.fnmatch(path, pattern):
        return True
    # For **, also try without the ** segment (zero dirs)
    # e.g. "src/**/*.py" should match "src/main.py"
    if "**/" in pattern:
        simplified = pattern.replace("**/", "")
        if fnmatch.fnmatch(path, simplified):
            return True
    return False
