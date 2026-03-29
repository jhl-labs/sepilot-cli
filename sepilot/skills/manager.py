"""Skill Manager for SEPilot Skills System

Enhanced with Claude Code-style features:
- SKILL.md markdown-based skill loading
- Multi-directory discovery (user, project, nested)
- Claude Code compatible paths (.claude/skills/)
"""

import importlib
import importlib.util
import logging
import threading
from pathlib import Path

from .base import BaseSkill, SkillMetadata, SkillResult

logger = logging.getLogger(__name__)


class SkillManager:
    """Manages skill discovery, loading, and execution

    Skills are loaded from:
    1. Built-in skills: sepilot/skills/builtin/
    2. User skills: ~/.sepilot/skills/
    (Project skills are skipped for security)
    """

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def _get_skill_directories(self) -> list[Path]:
        """Get directories to search for user/project skills (not builtin)

        Builtin skills are loaded via package import in _load_builtin_skills()
        Python (.py) skills: only from trusted user directory (~/.sepilot/skills/)
        Markdown (.md/SKILL.md) skills: from both user and project directories
        """
        dirs = []

        # User skills (~/.sepilot/skills/ and ~/.claude/skills/)
        for user_config in [Path.home() / ".sepilot", Path.home() / ".claude"]:
            user_dir = user_config / "skills"
            if user_dir.exists():
                dirs.append(user_dir)

        return dirs

    def _get_markdown_skill_directories(self) -> list[tuple[Path, str]]:
        """Get directories for markdown-based SKILL.md discovery.

        Returns list of (directory, source_type) tuples.
        Markdown skills are safe to load from project directories
        since they don't execute arbitrary code.
        """
        dirs: list[tuple[Path, str]] = []

        # User-level (higher priority)
        for user_config in [Path.home() / ".sepilot", Path.home() / ".claude"]:
            skills_dir = user_config / "skills"
            if skills_dir.is_dir():
                dirs.append((skills_dir, "user"))

        # Project-level (safe: only .md files, no code execution)
        for config_dir in [".sepilot", ".claude", ".agent"]:
            proj_skills = Path.cwd() / config_dir / "skills"
            if proj_skills.is_dir():
                dirs.append((proj_skills, "project"))

        return dirs

    def _load_skill_from_file(self, filepath: Path) -> BaseSkill | None:
        """Load a skill from a Python file"""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"skill_{filepath.stem}",
                filepath
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find BaseSkill subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) and
                    issubclass(attr, BaseSkill) and
                    attr is not BaseSkill
                ):
                    skill = attr()
                    # Validate before accepting
                    errors = skill.validate()
                    if errors:
                        logger.warning(
                            f"Skill validation failed for {filepath}: {errors}"
                        )
                        return None
                    return skill

            return None

        except Exception as e:
            logger.error(f"Failed to load skill from {filepath}: {e}")
            return None

    def discover_skills(self) -> None:
        """Discover and load all available skills (thread-safe)"""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            self._skills.clear()

            for skill_dir in self._get_skill_directories():
                if not skill_dir.exists():
                    continue

                for filepath in skill_dir.glob("*.py"):
                    if filepath.name.startswith("_"):
                        continue

                    skill = self._load_skill_from_file(filepath)
                    if skill:
                        metadata = skill.get_metadata()
                        self._skills[metadata.name] = skill
                        logger.info(f"Loaded skill: {metadata.name}")

            # Load markdown-based skills (SKILL.md)
            self._load_markdown_skills()

            # Load built-in skills from the builtin subpackage
            self._load_builtin_skills()

            self._loaded = True
            logger.info(f"Discovered {len(self._skills)} skills")

    def _load_markdown_skills(self):
        """Load markdown-based skills (SKILL.md format).

        Discovers SKILL.md files from user and project directories.
        Markdown skills are safe for project dirs (no code execution).
        """
        try:
            from .markdown_skill import MarkdownSkill

            for skills_dir, source in self._get_markdown_skill_directories():
                # Pattern 1: skills/skill-name/SKILL.md
                if skills_dir.is_dir():
                    for item in sorted(skills_dir.iterdir()):
                        if item.is_dir():
                            skill_file = item / "SKILL.md"
                            if skill_file.exists():
                                skill = MarkdownSkill.from_file(skill_file, source=source)
                                if skill:
                                    name = skill.get_metadata().name
                                    if name not in self._skills:
                                        self._skills[name] = skill
                                        logger.info(f"Loaded markdown skill: {name} ({source})")

                    # Pattern 2: skills/*.md (flat files)
                    for md_file in sorted(skills_dir.glob("*.md")):
                        if md_file.name.lower() in ("readme.md", "skill.md"):
                            continue
                        skill = MarkdownSkill.from_file(md_file, source=source)
                        if skill:
                            name = skill.get_metadata().name
                            if name not in self._skills:
                                self._skills[name] = skill
                                logger.debug(f"Loaded markdown skill: {name} ({source})")
        except ImportError:
            logger.debug("markdown_skill module not available")

    def _load_builtin_skills(self):
        """Load built-in skills from the builtin subpackage"""
        try:
            from .builtin import get_builtin_skills
            for skill in get_builtin_skills():
                metadata = skill.get_metadata()
                if metadata.name not in self._skills:
                    # Validate builtin skills too
                    errors = skill.validate()
                    if errors:
                        logger.warning(f"Builtin skill '{metadata.name}' validation: {errors}")
                        continue
                    self._skills[metadata.name] = skill
                    logger.debug(f"Loaded builtin skill: {metadata.name}")
        except ImportError:
            logger.debug("No builtin skills package found")

    def get_skill(self, name: str) -> BaseSkill | None:
        """Get a skill by name"""
        self.discover_skills()
        return self._skills.get(name)

    def list_skills(self) -> list[SkillMetadata]:
        """List all available skills sorted by category and name"""
        self.discover_skills()
        metadata_list = [skill.get_metadata() for skill in self._skills.values()]
        return sorted(metadata_list, key=lambda m: (m.category, m.name))

    def list_skills_by_category(self) -> dict[str, list[SkillMetadata]]:
        """List skills grouped by category"""
        skills = self.list_skills()
        categories: dict[str, list[SkillMetadata]] = {}
        for s in skills:
            categories.setdefault(s.category, []).append(s)
        return categories

    def execute_skill(
        self,
        name: str,
        input_text: str,
        context: dict
    ) -> SkillResult:
        """Execute a skill by name

        Args:
            name: Skill name
            input_text: User input
            context: Execution context

        Returns:
            SkillResult with execution status
        """
        skill = self.get_skill(name)
        if not skill:
            # Try fuzzy match
            candidates = [
                s for s in self._skills
                if name.lower() in s.lower() or s.lower() in name.lower()
            ]
            if candidates:
                suggestion = ", ".join(candidates[:3])
                return SkillResult(
                    success=False,
                    message=f"Skill '{name}' not found. Did you mean: {suggestion}?"
                )
            return SkillResult(
                success=False,
                message=f"Skill '{name}' not found. Use 'list skills' to see available skills."
            )

        try:
            return skill.execute(input_text, context)
        except Exception as e:
            logger.error(f"Skill '{name}' execution failed: {e}")
            return SkillResult(
                success=False,
                message=f"Skill execution failed: {e}"
            )

    async def execute_skill_async(
        self,
        name: str,
        input_text: str,
        context: dict
    ) -> SkillResult:
        """Execute a skill asynchronously

        Args:
            name: Skill name
            input_text: User input
            context: Execution context

        Returns:
            SkillResult with execution status
        """
        skill = self.get_skill(name)
        if not skill:
            return SkillResult(
                success=False,
                message=f"Skill '{name}' not found"
            )

        try:
            return await skill.execute_async(input_text, context)
        except Exception as e:
            logger.error(f"Skill '{name}' async execution failed: {e}")
            return SkillResult(
                success=False,
                message=f"Skill execution failed: {e}"
            )

    def find_matching_skill(self, input_text: str) -> BaseSkill | None:
        """Find the best matching skill for the input.

        Uses trigger_score() to rank skills and returns the highest-scoring one.
        """
        self.discover_skills()

        best_skill = None
        best_score = 0.0

        for skill in self._skills.values():
            score = skill.trigger_score(input_text)
            if score > best_score:
                best_score = score
                best_skill = skill

        if best_skill and best_score > 0.0:
            logger.debug(
                f"Best matching skill: {best_skill.get_metadata().name} "
                f"(score={best_score:.2f})"
            )
            return best_skill

        return None

    def find_all_matching_skills(
        self, input_text: str, min_score: float = 0.1
    ) -> list[tuple[BaseSkill, float]]:
        """Find all matching skills with their scores, sorted by score.

        Args:
            input_text: User input text
            min_score: Minimum trigger score to include

        Returns:
            List of (skill, score) tuples sorted by score descending
        """
        self.discover_skills()

        matches = []
        for skill in self._skills.values():
            score = skill.trigger_score(input_text)
            if score >= min_score:
                matches.append((skill, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def reload_skills(self) -> None:
        """Force reload all skills"""
        with self._lock:
            self._loaded = False
            self._skills.clear()
        self.discover_skills()


# Global singleton instance
_skill_manager: SkillManager | None = None
_skill_manager_lock = threading.Lock()


def get_skill_manager() -> SkillManager:
    """Get or create the global skill manager instance (thread-safe)"""
    global _skill_manager
    if _skill_manager is None:
        with _skill_manager_lock:
            if _skill_manager is None:
                _skill_manager = SkillManager()
    return _skill_manager
