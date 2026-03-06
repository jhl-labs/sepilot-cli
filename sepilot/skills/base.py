"""Base Skill class for SEPilot Skills System"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result from executing a skill"""
    success: bool
    message: str
    data: Any = None
    prompt_injection: str | None = None  # Text to inject into the prompt
    context_addition: str | None = None  # Additional context for the LLM


@dataclass
class SkillMetadata:
    """Metadata describing a skill"""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    triggers: list[str] = field(default_factory=list)  # Keywords that trigger this skill
    category: str = "general"
    priority: int = 0  # Higher priority = matched first when multiple skills match


class BaseSkill(ABC):
    """Base class for all skills

    Skills are specialized plugins that provide domain-specific capabilities.
    They can:
    - Be triggered by keywords or explicit /skill invocation
    - Inject prompts or context into the conversation
    - Perform specialized processing
    - Support both sync and async execution

    Example:
        class FrontendDesignSkill(BaseSkill):
            def get_metadata(self) -> SkillMetadata:
                return SkillMetadata(
                    name="frontend-design",
                    description="Create production-grade frontend interfaces",
                    triggers=["build web", "create component", "design ui"],
                    priority=10,
                )

            def execute(self, input_text: str, context: dict) -> SkillResult:
                return SkillResult(
                    success=True,
                    message="Frontend design skill activated",
                    prompt_injection="Design Guidelines: ..."
                )
    """

    @abstractmethod
    def get_metadata(self) -> SkillMetadata:
        """Return metadata describing this skill"""
        pass

    @abstractmethod
    def execute(self, input_text: str, context: dict) -> SkillResult:
        """Execute the skill (sync)

        Args:
            input_text: The user's input that triggered this skill
            context: Execution context including:
                - agent: The current agent instance
                - console: Rich console for output
                - conversation: Current conversation history

        Returns:
            SkillResult with execution status and optional prompt injection
        """
        pass

    async def execute_async(self, input_text: str, context: dict) -> SkillResult:
        """Execute the skill asynchronously.

        Default implementation wraps the sync execute().
        Override for truly async operations (API calls, file I/O, etc.)

        Args:
            input_text: The user's input that triggered this skill
            context: Execution context

        Returns:
            SkillResult with execution status and optional prompt injection
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute, input_text, context)

    def trigger_score(self, input_text: str) -> float:
        """Calculate how well this skill matches the input.

        Returns a score from 0.0 (no match) to 1.0 (perfect match).
        Higher scores = better match.

        Default implementation scores based on:
        - Number of matching triggers
        - Trigger length (longer = more specific = higher score)
        - Priority from metadata
        """
        metadata = self.get_metadata()
        if not metadata.triggers:
            return 0.0

        input_lower = input_text.lower()
        matched_triggers = [
            t for t in metadata.triggers if t.lower() in input_lower
        ]

        if not matched_triggers:
            return 0.0

        # Base score: ratio of matched triggers
        base_score = len(matched_triggers) / len(metadata.triggers)

        # Bonus for longer (more specific) triggers
        longest_match = max(len(t) for t in matched_triggers)
        specificity_bonus = min(longest_match / 20.0, 0.3)

        # Priority bonus
        priority_bonus = min(metadata.priority / 100.0, 0.2)

        return min(base_score + specificity_bonus + priority_bonus, 1.0)

    def should_trigger(self, input_text: str) -> bool:
        """Check if this skill should be triggered by the input

        Default implementation checks if any trigger keywords are in the input.
        Override for custom trigger logic.
        """
        return self.trigger_score(input_text) > 0.0

    def validate(self) -> list[str]:
        """Validate that the skill is properly configured.

        Returns:
            List of validation error messages (empty = valid)
        """
        errors = []
        try:
            metadata = self.get_metadata()
            if not metadata.name:
                errors.append("Skill name is empty")
            if not metadata.description:
                errors.append("Skill description is empty")
            if not metadata.name.replace("-", "").replace("_", "").isalnum():
                errors.append(f"Skill name contains invalid characters: {metadata.name}")
        except Exception as e:
            errors.append(f"Failed to get metadata: {e}")
        return errors

    def get_help(self) -> str:
        """Get help text for this skill"""
        metadata = self.get_metadata()
        triggers_str = ", ".join(metadata.triggers[:5])
        if len(metadata.triggers) > 5:
            triggers_str += f" (+{len(metadata.triggers) - 5} more)"
        return f"{metadata.name} (v{metadata.version}): {metadata.description}\n  Triggers: {triggers_str}"
