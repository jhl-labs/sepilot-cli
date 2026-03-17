"""Instructions Loader - Hierarchical instruction loading system.

Loads user/project instructions from multiple sources following
Claude Code-style configuration hierarchy:

Load order (later = higher priority):
1. ~/.sepilot/SEPILOT.md (global user instructions)
2. Parent directory chain up to git root
3. Project root .sepilot/context.md
4. .sepilot/rules/*.md (path-based conditional rules)
5. AGENT.md / CLAUDE.md (alternative names)

Supported instruction file names (checked in order):
- SEPILOT.md
- AGENT.md
- CLAUDE.md
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from sepilot.utils.markdown import glob_match as _glob_match
from sepilot.utils.markdown import parse_frontmatter as _parse_frontmatter

logger = logging.getLogger(__name__)

# Instruction file names in priority order
INSTRUCTION_FILENAMES = ["SEPILOT.md", "AGENT.md", "CLAUDE.md"]

# Size limits to prevent context window exhaustion
MAX_TOTAL_INSTRUCTIONS_CHARS = 16384  # 16K chars total
MAX_SINGLE_FILE_CHARS = 8192  # 8K chars per file

# Config directory names
CONFIG_DIRS = [".sepilot", ".claude", ".agent"]

# Trusted directories for dynamic command execution
TRUSTED_DIRS = [Path.home() / ".sepilot", Path.home() / ".claude"]


def _get_git_root(start_path: Path | None = None) -> Path | None:
    """Get git repository root directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
            cwd=str(start_path or Path.cwd())
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    return None


# _parse_frontmatter imported from sepilot.utils.markdown


def _expand_dynamic_commands(content: str, source_path: Path) -> str:
    """Expand !`command` patterns in instruction content.

    Only expands commands from trusted directories (user home config).
    Project-level files skip command expansion for security.

    Args:
        content: Instruction content with potential !`command` patterns
        source_path: Path of the source file (for trust check)

    Returns:
        Content with commands expanded
    """
    # Security: only expand commands from trusted directories
    # Use resolve() to prevent symlink bypass attacks
    resolved_source = source_path.resolve()
    is_trusted = any(
        str(resolved_source).startswith(str(td.resolve())) for td in TRUSTED_DIRS
    )
    if not is_trusted:
        return content

    def replace_command(match: re.Match) -> str:
        cmd = match.group(1)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=10, cwd=str(Path.cwd())
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(f"Dynamic command failed: {cmd}")
                return f"(command failed: {cmd})"
        except subprocess.TimeoutExpired:
            return f"(command timed out: {cmd})"
        except Exception as e:
            return f"(command error: {e})"

    # Match !`command` pattern
    return re.sub(r'!\`([^`]+)\`', replace_command, content)


class InstructionsLoader:
    """Loads and assembles instructions from multiple hierarchical sources.

    Follows Claude Code's configuration loading pattern:
    - Global user instructions (~/.sepilot/SEPILOT.md)
    - Directory chain from project root to cwd
    - Project config directory (.sepilot/)
    - Conditional rules (.sepilot/rules/*.md with path filters)
    - Alternative config names (AGENT.md, CLAUDE.md)
    """

    def __init__(self, working_dir: Path | None = None):
        """Initialize loader.

        Args:
            working_dir: Working directory (defaults to cwd)
        """
        self.working_dir = working_dir or Path.cwd()
        self._cache: dict[str, str] = {}
        self._cache_key: str = ""

    def load_all(
        self,
        working_dir: Path | None = None,
        active_files: list[str] | None = None
    ) -> str:
        """Load all instructions from all sources.

        Args:
            working_dir: Override working directory
            active_files: List of currently active file paths (for rules filtering)

        Returns:
            Combined instruction text
        """
        cwd = working_dir or self.working_dir
        cache_key = f"{cwd}:{','.join(active_files or [])}"

        if cache_key == self._cache_key and self._cache:
            return self._cache.get("result", "")

        sections: list[str] = []

        # 1. Global user instructions
        global_instructions = self._load_global_instructions()
        if global_instructions:
            sections.append(f"# Global Instructions\n\n{global_instructions}")

        # 2. Directory chain instructions (parent dirs -> project root)
        chain_instructions = self._load_directory_chain(cwd)
        if chain_instructions:
            sections.append(chain_instructions)

        # 3. Project config directory context
        project_context = self._load_project_context(cwd)
        if project_context:
            sections.append(f"# Project Context\n\n{project_context}")

        # 4. Conditional rules
        rules = self._load_conditional_rules(cwd, active_files)
        if rules:
            sections.append(f"# Active Rules\n\n{rules}")

        result = "\n\n---\n\n".join(sections) if sections else ""

        # Enforce total size limit
        if len(result) > MAX_TOTAL_INSTRUCTIONS_CHARS:
            result = result[:MAX_TOTAL_INSTRUCTIONS_CHARS] + "\n\n[... instructions truncated — limit: 16K chars ...]"
            logger.warning("Instructions truncated to %d chars", MAX_TOTAL_INSTRUCTIONS_CHARS)

        self._cache_key = cache_key
        self._cache["result"] = result

        return result

    @staticmethod
    def _truncate_file_content(content: str, filepath: Path) -> str:
        """Truncate content to per-file size limit."""
        if len(content) > MAX_SINGLE_FILE_CHARS:
            logger.warning("Truncating %s (%d chars > %d limit)", filepath, len(content), MAX_SINGLE_FILE_CHARS)
            return content[:MAX_SINGLE_FILE_CHARS] + f"\n\n[... truncated from {filepath.name} ...]"
        return content

    def _load_global_instructions(self) -> str:
        """Load global instructions from ~/.sepilot/ or ~/.claude/."""
        for config_dir in [Path.home() / ".sepilot", Path.home() / ".claude"]:
            for filename in INSTRUCTION_FILENAMES:
                filepath = config_dir / filename
                if filepath.exists():
                    try:
                        content = filepath.read_text(encoding="utf-8")
                        content = self._truncate_file_content(content, filepath)
                        return _expand_dynamic_commands(content, filepath)
                    except Exception as e:
                        logger.warning(f"Failed to read {filepath}: {e}")
        return ""

    def _load_directory_chain(self, cwd: Path) -> str:
        """Load instructions from directory chain (cwd up to git root).

        Returns instructions from all directories in the chain,
        ordered from root to cwd (root instructions first).
        """
        git_root = _get_git_root(cwd)
        stop_at = git_root or Path(cwd.anchor)

        # Collect directories from cwd up to stop_at
        directories: list[Path] = []
        current = cwd.resolve()
        stop_resolved = stop_at.resolve()

        while current >= stop_resolved:
            directories.append(current)
            parent = current.parent
            if parent == current:
                break
            current = parent

        # Reverse: load from root to cwd (root = lower priority)
        directories.reverse()

        sections: list[str] = []
        for directory in directories:
            content = self._load_instruction_file(directory)
            if content:
                rel_path = directory.relative_to(stop_resolved) if directory != stop_resolved else Path(".")
                sections.append(f"## Instructions from {rel_path}/\n\n{content}")

        return "\n\n".join(sections)

    def _load_instruction_file(self, directory: Path) -> str:
        """Load first matching instruction file from a directory.

        Checks for instruction files in both the directory root
        and config subdirectories (.sepilot/, .claude/, .agent/).
        """
        # Check directory root
        for filename in INSTRUCTION_FILENAMES:
            filepath = directory / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    content = self._truncate_file_content(content, filepath)
                    return _expand_dynamic_commands(content, filepath)
                except Exception:
                    continue

        # Check config subdirectories
        for config_dir_name in CONFIG_DIRS:
            config_dir = directory / config_dir_name
            if not config_dir.is_dir():
                continue
            for filename in INSTRUCTION_FILENAMES:
                filepath = config_dir / filename
                if filepath.exists():
                    try:
                        content = filepath.read_text(encoding="utf-8")
                        content = self._truncate_file_content(content, filepath)
                        return _expand_dynamic_commands(content, filepath)
                    except Exception:
                        continue

        return ""

    def _load_project_context(self, cwd: Path) -> str:
        """Load project-specific context from .sepilot/context.md."""
        git_root = _get_git_root(cwd) or cwd

        for config_dir_name in CONFIG_DIRS:
            context_file = git_root / config_dir_name / "context.md"
            if context_file.exists():
                try:
                    return context_file.read_text(encoding="utf-8")
                except Exception:
                    continue

        return ""

    def _load_conditional_rules(
        self,
        cwd: Path,
        active_files: list[str] | None = None
    ) -> str:
        """Load conditional rules from .sepilot/rules/ directory.

        Rules files can have YAML frontmatter with path patterns:
        ```
        ---
        paths:
          - "src/**/*.py"
          - "tests/**"
        ---
        Rule content here...
        ```

        Rules without paths are always active.
        Rules with paths are only active when active_files match.
        """
        git_root = _get_git_root(cwd) or cwd
        rules_sections: list[str] = []

        # Check project-level rules
        for config_dir_name in CONFIG_DIRS:
            rules_dir = git_root / config_dir_name / "rules"
            if rules_dir.is_dir():
                self._load_rules_from_dir(
                    rules_dir, git_root, active_files, rules_sections
                )

        # Check user-level rules
        for user_config in [Path.home() / ".sepilot", Path.home() / ".claude"]:
            user_rules_dir = user_config / "rules"
            if user_rules_dir.is_dir():
                self._load_rules_from_dir(
                    user_rules_dir, git_root, active_files, rules_sections
                )

        return "\n\n".join(rules_sections)

    def _load_rules_from_dir(
        self,
        rules_dir: Path,
        project_root: Path,
        active_files: list[str] | None,
        sections: list[str]
    ) -> None:
        """Load rules from a rules directory."""
        # Support both flat files and nested directories
        rule_files = sorted(rules_dir.rglob("*.md"))

        for rule_file in rule_files:
            try:
                content = rule_file.read_text(encoding="utf-8")
                metadata, body = _parse_frontmatter(content)

                if not body.strip():
                    continue

                # Check path filter
                path_patterns = metadata.get("paths", [])
                if path_patterns and active_files:
                    if not self._matches_any_path(
                        path_patterns, active_files, project_root
                    ):
                        continue
                elif path_patterns and not active_files:
                    # Has path filter but no active files - skip
                    continue

                # Include this rule
                rule_name = rule_file.stem
                description = metadata.get("description", "")
                header = f"### Rule: {rule_name}"
                if description:
                    header += f" - {description}"

                sections.append(f"{header}\n\n{body}")

            except Exception as e:
                logger.warning(f"Failed to load rule {rule_file}: {e}")

    def _matches_any_path(
        self,
        patterns: list[str],
        active_files: list[str],
        project_root: Path
    ) -> bool:
        """Check if any active file matches any of the path patterns."""
        for file_path in active_files:
            # Make relative to project root
            try:
                rel_path = str(Path(file_path).relative_to(project_root))
            except ValueError:
                rel_path = file_path

            for pattern in patterns:
                if _glob_match(rel_path, pattern):
                    return True

        return False

    def invalidate_cache(self) -> None:
        """Invalidate cached instructions."""
        self._cache.clear()
        self._cache_key = ""

    def get_instruction_sources(self, cwd: Path | None = None) -> list[dict[str, str]]:
        """Get list of instruction sources that would be loaded.

        Useful for debugging and UI display.

        Returns:
            List of dicts with 'path', 'type', and 'status' keys
        """
        cwd = cwd or self.working_dir
        sources: list[dict[str, str]] = []

        # Global
        for config_dir in [Path.home() / ".sepilot", Path.home() / ".claude"]:
            for filename in INSTRUCTION_FILENAMES:
                filepath = config_dir / filename
                sources.append({
                    "path": str(filepath),
                    "type": "global",
                    "status": "active" if filepath.exists() else "not found"
                })
                if filepath.exists():
                    break
            else:
                continue
            break

        # Directory chain
        git_root = _get_git_root(cwd) or Path(cwd.anchor)
        current = cwd.resolve()
        stop = git_root.resolve()

        while current >= stop:
            for filename in INSTRUCTION_FILENAMES:
                filepath = current / filename
                if filepath.exists():
                    sources.append({
                        "path": str(filepath),
                        "type": "directory_chain",
                        "status": "active"
                    })
                    break
            current = current.parent
            if current == current.parent:
                break

        # Rules
        for config_dir_name in CONFIG_DIRS:
            rules_dir = git_root / config_dir_name / "rules"
            if rules_dir.is_dir():
                for rule_file in sorted(rules_dir.rglob("*.md")):
                    sources.append({
                        "path": str(rule_file),
                        "type": "rule",
                        "status": "active"
                    })

        return sources


# Module-level convenience functions

_loader: InstructionsLoader | None = None


def get_instructions_loader() -> InstructionsLoader:
    """Get or create global InstructionsLoader instance."""
    global _loader
    if _loader is None:
        _loader = InstructionsLoader()
    return _loader


def load_all_instructions(
    working_dir: Path | None = None,
    active_files: list[str] | None = None
) -> str:
    """Load all instructions from all sources.

    Convenience function that uses the global loader.

    Args:
        working_dir: Override working directory
        active_files: Currently active file paths for rules filtering

    Returns:
        Combined instruction text
    """
    loader = get_instructions_loader()
    return loader.load_all(working_dir=working_dir, active_files=active_files)
