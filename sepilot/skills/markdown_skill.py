"""Markdown-based Skill - SKILL.md format support.

Supports Claude Code-style SKILL.md files with YAML frontmatter:

```markdown
---
name: my-skill
description: What this skill does
allowed-tools: [Read, Grep, Bash]
model: sonnet
context: fork
agent: Explore
user-invocable: true
disable-model-invocation: false
argument-hint: "[issue-number]"
---

Skill instructions and workflow here...
Use $ARGUMENTS for user-provided arguments.
Use $0, $1 for positional arguments.
```
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

from sepilot.utils.markdown import parse_frontmatter as _parse_frontmatter

from .base import BaseSkill, SkillMetadata, SkillResult

logger = logging.getLogger(__name__)


def _substitute_variables(
    content: str,
    arguments: str = "",
    skill_dir: str = "",
    session_id: str = ""
) -> str:
    """Substitute skill template variables.

    Supported variables:
    - $ARGUMENTS / ${ARGUMENTS} - All arguments as string
    - $ARGUMENTS[N] / $N - Nth argument (0-indexed)
    - ${SEPILOT_SESSION_ID} / ${CLAUDE_SESSION_ID} - Current session ID
    - ${SEPILOT_SKILL_DIR} / ${CLAUDE_SKILL_DIR} - Skill directory path
    """
    # Split arguments
    arg_parts = arguments.split() if arguments else []

    # Replace $ARGUMENTS[N] and $N patterns first (before $ARGUMENTS)
    def replace_indexed_arg(match: re.Match) -> str:
        idx = int(match.group(1))
        if idx < len(arg_parts):
            return arg_parts[idx]
        return match.group(0)  # Keep original if index out of range

    content = re.sub(r'\$ARGUMENTS\[(\d+)\]', replace_indexed_arg, content)
    content = re.sub(r'\$(\d+)(?!\w)', replace_indexed_arg, content)

    # Replace $ARGUMENTS / ${ARGUMENTS}
    content = content.replace("${ARGUMENTS}", arguments)
    content = content.replace("$ARGUMENTS", arguments)

    # Replace session/skill dir variables (support both SEPILOT_ and CLAUDE_ prefixes)
    for prefix in ["SEPILOT", "CLAUDE"]:
        content = content.replace(f"${{{prefix}_SESSION_ID}}", session_id)
        content = content.replace(f"${{{prefix}_SKILL_DIR}}", skill_dir)

    # Generic environment variable expansion: ${ENV_VAR}
    # Security: only allow SEPILOT_/CLAUDE_/AGENT_ prefixed vars to prevent
    # leaking sensitive env vars (AWS keys, DB passwords, etc.)
    ALLOWED_ENV_PREFIXES = ("SEPILOT_", "CLAUDE_", "AGENT_")

    def replace_env_var(match: re.Match) -> str:
        var_name = match.group(1)
        if any(var_name.startswith(p) for p in ALLOWED_ENV_PREFIXES):
            return os.environ.get(var_name, match.group(0))
        return match.group(0)  # Keep original for non-allowed vars

    content = re.sub(r'\$\{([A-Z_][A-Z0-9_]*)\}', replace_env_var, content)

    return content


class MarkdownSkill(BaseSkill):
    """A skill loaded from a SKILL.md markdown file.

    Supports Claude Code-style SKILL.md format with YAML frontmatter
    for metadata and markdown body for instructions/workflow.
    """

    def __init__(
        self,
        skill_path: Path,
        metadata: dict[str, Any],
        body: str,
        source: str = "project"
    ):
        """Initialize from parsed SKILL.md.

        Args:
            skill_path: Path to the SKILL.md file
            metadata: Parsed YAML frontmatter
            body: Markdown body content
            source: Source type ("builtin", "user", "project")
        """
        self.skill_path = skill_path
        self.skill_dir = skill_path.parent
        self._metadata = metadata
        self._body = body
        self._source = source

        # Parse metadata fields
        self._name = metadata.get("name", skill_path.parent.name)
        self._description = metadata.get("description", "")
        self._allowed_tools = metadata.get("allowed-tools", [])
        if isinstance(self._allowed_tools, str):
            self._allowed_tools = [t.strip() for t in self._allowed_tools.split(",")]
        self._model_override = metadata.get("model")
        self._context_mode = metadata.get("context", "inline")  # "inline" or "fork"
        self._agent_type = metadata.get("agent")
        self._user_invocable = metadata.get("user-invocable", True)
        self._disable_model_invocation = metadata.get("disable-model-invocation", False)
        self._argument_hint = metadata.get("argument-hint", "")
        self._triggers = metadata.get("triggers", [])
        self._category = metadata.get("category", "custom")
        self._priority = metadata.get("priority", 5)
        self._hooks = metadata.get("hooks", {})

    def get_metadata(self) -> SkillMetadata:
        """Return metadata for this skill."""
        return SkillMetadata(
            name=self._name,
            description=self._description,
            version=self._metadata.get("version", "1.0.0"),
            author=self._metadata.get("author", ""),
            triggers=self._triggers,
            category=self._category,
            priority=self._priority,
            # Extended fields
            allowed_tools=self._allowed_tools,
            model_override=self._model_override,
            context_mode=self._context_mode,
            agent_type=self._agent_type,
            user_invocable=self._user_invocable,
            disable_model_invocation=self._disable_model_invocation,
            argument_hint=self._argument_hint,
        )

    def execute(self, input_text: str, context: dict) -> SkillResult:
        """Execute the markdown skill.

        Processes the SKILL.md body with variable substitution
        and returns it as a prompt injection.
        """
        # Get session info from context
        session_id = context.get("session_id", "")

        # Perform variable substitution
        processed_body = _substitute_variables(
            content=self._body,
            arguments=input_text,
            skill_dir=str(self.skill_dir),
            session_id=session_id,
        )

        # Build result
        return SkillResult(
            success=True,
            message=f"Skill '{self._name}' activated",
            data={
                "name": self._name,
                "source": self._source,
                "allowed_tools": self._allowed_tools,
                "model_override": self._model_override,
                "context_mode": self._context_mode,
                "agent_type": self._agent_type,
            },
            prompt_injection=processed_body,
            context_addition=self._metadata.get("context-addition"),
        )

    def trigger_score(self, input_text: str) -> float:
        """Calculate trigger score for this skill."""
        if self._disable_model_invocation:
            return 0.0

        # Use description-based matching if no explicit triggers
        if not self._triggers and self._description:
            input_lower = input_text.lower()
            desc_words = set(self._description.lower().split())
            input_words = set(input_lower.split())
            overlap = len(desc_words & input_words)
            if overlap >= 2:
                return min(overlap / len(desc_words), 0.8)
            return 0.0

        return super().trigger_score(input_text)

    @property
    def is_user_invocable(self) -> bool:
        """Whether this skill can be invoked by the user via /command."""
        return self._user_invocable

    @property
    def allowed_tools(self) -> list[str]:
        """Tools this skill is allowed to use."""
        return self._allowed_tools

    @property
    def model_override(self) -> str | None:
        """Model override for this skill."""
        return self._model_override

    @property
    def context_mode(self) -> str:
        """Context mode: 'inline' or 'fork'."""
        return self._context_mode

    def get_help(self) -> str:
        """Get help text for this skill."""
        hint = f" {self._argument_hint}" if self._argument_hint else ""
        return f"/{self._name}{hint}: {self._description}"

    @classmethod
    def from_file(cls, skill_path: Path, source: str = "project") -> "MarkdownSkill | None":
        """Load a MarkdownSkill from a SKILL.md file.

        Args:
            skill_path: Path to SKILL.md file
            source: Source type

        Returns:
            MarkdownSkill instance or None if invalid
        """
        try:
            content = skill_path.read_text(encoding="utf-8")
            metadata, body = _parse_frontmatter(content)

            if not body.strip():
                logger.warning(f"Empty skill body: {skill_path}")
                return None

            skill = cls(
                skill_path=skill_path,
                metadata=metadata,
                body=body,
                source=source,
            )

            # Validate
            errors = skill.validate()
            if errors:
                logger.warning(f"Skill validation errors for {skill_path}: {errors}")
                return None

            return skill

        except Exception as e:
            logger.error(f"Failed to load markdown skill from {skill_path}: {e}")
            return None

    def validate(self) -> list[str]:
        """Validate the skill."""
        errors = []
        if not self._name:
            errors.append("Skill name is empty")
        if not self._name.replace("-", "").replace("_", "").isalnum():
            errors.append(f"Invalid skill name: {self._name}")
        if not self._body.strip():
            errors.append("Skill body is empty")
        if self._context_mode not in ("inline", "fork"):
            errors.append(f"Invalid context mode: {self._context_mode}")
        return errors


def discover_markdown_skills(
    search_dirs: list[Path] | None = None,
    project_root: Path | None = None
) -> list[MarkdownSkill]:
    """Discover SKILL.md files from multiple directories.

    Search locations:
    1. ~/.sepilot/skills/*/SKILL.md (user skills)
    2. ~/.claude/skills/*/SKILL.md (Claude Code compat)
    3. .sepilot/skills/*/SKILL.md (project skills - .md only, safe)
    4. .claude/skills/*/SKILL.md (Claude Code compat)

    Args:
        search_dirs: Additional directories to search
        project_root: Project root directory

    Returns:
        List of discovered MarkdownSkill instances
    """
    skills: list[MarkdownSkill] = []
    seen_names: set[str] = set()

    if project_root is None:
        project_root = Path.cwd()

    # Build search directories with source types
    dirs_with_source: list[tuple[Path, str]] = []

    # User-level directories (higher priority)
    for user_config in [Path.home() / ".sepilot", Path.home() / ".claude"]:
        skills_dir = user_config / "skills"
        if skills_dir.is_dir():
            dirs_with_source.append((skills_dir, "user"))

    # Project-level directories
    for config_dir_name in [".sepilot", ".claude", ".agent"]:
        proj_skills = project_root / config_dir_name / "skills"
        if proj_skills.is_dir():
            dirs_with_source.append((proj_skills, "project"))

    # Additional search directories
    if search_dirs:
        for d in search_dirs:
            if d.is_dir():
                dirs_with_source.append((d, "custom"))

    # Discover skills from all directories
    for skills_dir, source in dirs_with_source:
        # Pattern 1: skills/skill-name/SKILL.md
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                skill = MarkdownSkill.from_file(skill_file, source=source)
                if skill and skill._name not in seen_names:
                    skills.append(skill)
                    seen_names.add(skill._name)

        # Pattern 2: skills/*.md (flat files, skill name = filename stem)
        for md_file in sorted(skills_dir.glob("*.md")):
            if md_file.name == "README.md":
                continue
            skill = MarkdownSkill.from_file(md_file, source=source)
            if skill and skill._name not in seen_names:
                skills.append(skill)
                seen_names.add(skill._name)

    logger.info(f"Discovered {len(skills)} markdown skills")
    return skills
