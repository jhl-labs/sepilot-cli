"""Rules Loader - Path-based conditional rules system.

Loads markdown rules from .sepilot/rules/ directories with
YAML frontmatter for path-based filtering.

Rule files can have path patterns in frontmatter:
```markdown
---
paths:
  - "src/**/*.py"
  - "tests/**"
description: "Python code rules"
priority: 10
---
Rule content here...
```

Rules without paths are always active.
Rules with paths are active only when working on matching files.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sepilot.utils.markdown import glob_match as _glob_match
from sepilot.utils.markdown import parse_frontmatter as _parse_frontmatter

logger = logging.getLogger(__name__)

# Config directory names to search
CONFIG_DIRS = [".sepilot", ".claude", ".agent"]


@dataclass
class Rule:
    """A single rule with optional path filtering."""
    name: str
    description: str
    content: str
    paths: list[str] = field(default_factory=list)
    priority: int = 0
    source_file: str = ""
    source_type: str = "project"  # "project" | "user"

    def matches_path(self, file_path: str, project_root: Path) -> bool:
        """Check if a file path matches this rule's patterns.

        If no paths specified, rule is always active.
        """
        if not self.paths:
            return True

        # Make file path relative to project root
        try:
            rel_path = str(Path(file_path).relative_to(project_root))
        except ValueError:
            rel_path = file_path

        return any(_glob_match(rel_path, pattern) for pattern in self.paths)

    def matches_any(self, file_paths: list[str], project_root: Path) -> bool:
        """Check if any of the file paths match this rule."""
        if not self.paths:
            return True
        return any(self.matches_path(fp, project_root) for fp in file_paths)


class RulesLoader:
    """Loads and manages path-based conditional rules.

    Search locations:
    1. .sepilot/rules/ (project-level)
    2. .claude/rules/ (Claude Code compat)
    3. ~/.sepilot/rules/ (user-level)
    4. ~/.claude/rules/ (user-level Claude Code compat)

    Supports:
    - Flat .md files in rules/
    - Nested directories (rules/frontend/react.md)
    - Symlinks to shared rules
    """

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self._rules: list[Rule] = []
        self._loaded = False

    def load(self) -> None:
        """Load all rules from all sources."""
        self._rules.clear()

        # Project-level rules (higher priority)
        for config_dir in CONFIG_DIRS:
            rules_dir = self.project_root / config_dir / "rules"
            if rules_dir.is_dir():
                self._load_from_dir(rules_dir, source_type="project", priority_base=100)

        # User-level rules (lower priority)
        for user_config in [Path.home() / ".sepilot", Path.home() / ".claude"]:
            user_rules = user_config / "rules"
            if user_rules.is_dir():
                self._load_from_dir(user_rules, source_type="user", priority_base=0)

        # Sort by priority (higher first)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        self._loaded = True

        logger.info(f"Loaded {len(self._rules)} rules")

    def _load_from_dir(
        self, rules_dir: Path, source_type: str, priority_base: int
    ) -> None:
        """Load rules from a directory (recursive)."""
        for md_file in sorted(rules_dir.rglob("*.md")):
            if md_file.name.lower() == "readme.md":
                continue

            # Follow symlinks
            resolved = md_file.resolve()
            if not resolved.exists():
                continue

            try:
                content = resolved.read_text(encoding="utf-8")
                metadata, body = _parse_frontmatter(content)

                if not body.strip():
                    continue

                # Compute rule name from relative path
                try:
                    rel = md_file.relative_to(rules_dir)
                    name = str(rel.with_suffix("")).replace("/", ".")
                except ValueError:
                    name = md_file.stem

                rule = Rule(
                    name=name,
                    description=metadata.get("description", ""),
                    content=body,
                    paths=metadata.get("paths", []),
                    priority=priority_base + metadata.get("priority", 0),
                    source_file=str(md_file),
                    source_type=source_type,
                )
                self._rules.append(rule)

            except Exception as e:
                logger.warning(f"Failed to load rule from {md_file}: {e}")

    def get_active_rules(
        self,
        active_files: list[str] | None = None,
        include_project_rules: bool = True,
        include_user_rules: bool = True,
    ) -> list[Rule]:
        """Get rules that are active for the given files.

        Args:
            active_files: List of file paths currently being worked on.
                If None, only returns rules without path filters.
            include_project_rules: Include project-local rules
            include_user_rules: Include user-level rules

        Returns:
            List of active rules sorted by priority
        """
        if not self._loaded:
            self.load()

        active = []
        for rule in self._rules:
            if rule.source_type == "project" and not include_project_rules:
                continue
            if rule.source_type == "user" and not include_user_rules:
                continue
            if not rule.paths:
                # No path filter = always active
                active.append(rule)
            elif active_files and rule.matches_any(active_files, self.project_root):
                active.append(rule)

        return active

    def get_rules_text(
        self,
        active_files: list[str] | None = None,
        include_project_rules: bool = True,
        include_user_rules: bool = True,
    ) -> str:
        """Get formatted text of all active rules.

        Returns concatenated rule content for prompt injection.
        """
        rules = self.get_active_rules(
            active_files,
            include_project_rules=include_project_rules,
            include_user_rules=include_user_rules,
        )
        if not rules:
            return ""

        sections = []
        for rule in rules:
            header = f"## Rule: {rule.name}"
            if rule.description:
                header += f"\n_{rule.description}_"
            sections.append(f"{header}\n\n{rule.content}")

        return "\n\n---\n\n".join(sections)

    def list_rules(self) -> list[dict[str, Any]]:
        """List all loaded rules with metadata."""
        if not self._loaded:
            self.load()

        return [
            {
                "name": r.name,
                "description": r.description,
                "paths": r.paths,
                "priority": r.priority,
                "source_type": r.source_type,
                "source_file": r.source_file,
                "always_active": len(r.paths) == 0,
            }
            for r in self._rules
        ]

    def reload(self) -> None:
        """Force reload all rules."""
        self._loaded = False
        self.load()


# Global singleton
_rules_loader: RulesLoader | None = None


def get_rules_loader(project_root: Path | None = None) -> RulesLoader:
    """Get or create the global RulesLoader instance."""
    global _rules_loader
    if _rules_loader is None:
        _rules_loader = RulesLoader(project_root=project_root)
    elif project_root is not None and _rules_loader.project_root != project_root:
        _rules_loader.project_root = project_root
        _rules_loader._rules.clear()
        _rules_loader._loaded = False
    return _rules_loader
