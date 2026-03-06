"""Prompt template loader and manager for SE Pilot"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptConfig:
    """Configuration for prompt behavior"""
    style: dict[str, Any]
    auto_finish: dict[str, Any]
    limits: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptConfig":
        """Create PromptConfig from dictionary"""
        return cls(
            style=data.get("style", {}),
            auto_finish=data.get("auto_finish", {}),
            limits=data.get("limits", {})
        )


class PromptTemplate:
    """Container for prompt templates"""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.system = data.get("system", {})
        self.think = data.get("think", {})
        self.act = data.get("act", {})
        self.config = PromptConfig.from_dict(data.get("config", {}))

    def get_system_prompt(self) -> str:
        """Get system prompt"""
        return self.system.get("main", "")

    def get_think_prompt(self, available_tools: str, context: str) -> str:
        """Get thinking prompt with placeholders filled"""
        prompt = self.think.get("main", "")

        # Ensure prompt is not empty
        if not prompt:
            prompt = """Based on the task, what should you do next?
Available tools: {available_tools}
Context: {context}
Think step by step."""

        try:
            return prompt.format(
                available_tools=available_tools,
                context=context
            )
        except KeyError:
            # Handle missing placeholders
            return prompt

    def get_act_system_prompt(self) -> str:
        """Get system prompt for action phase"""
        return self.act.get("system", "")

    def get_act_prompt(self, latest_thought: str, available_tools: str) -> str:
        """Get action prompt with placeholders filled"""
        prompt = self.act.get("main", "")

        # Ensure prompt is not empty
        if not prompt:
            prompt = """Based on your thought: "{latest_thought}"
Available tools: {available_tools}
Choose the appropriate tool and provide input as JSON."""

        try:
            return prompt.format(
                latest_thought=latest_thought,
                available_tools=available_tools
            )
        except KeyError:
            # Handle missing placeholders
            return prompt

    def get_auto_finish_config(self) -> dict[str, Any]:
        """Get auto-finish configuration"""
        return self.config.auto_finish

    def get_limits_config(self) -> dict[str, Any]:
        """Get limits configuration"""
        return self.config.limits


class PromptLoader:
    """Load and manage prompt templates"""

    def __init__(self, templates_dir: Path | None = None):
        """Initialize prompt loader

        Args:
            templates_dir: Directory containing prompt templates
        """
        if templates_dir is None:
            # Default to prompts/templates in package directory
            package_dir = Path(__file__).parent
            templates_dir = package_dir / "templates"

        self.templates_dir = Path(templates_dir)
        self._templates_cache: dict[str, PromptTemplate] = {}

    def list_available_profiles(self) -> list[str]:
        """List all available prompt profiles"""
        profiles = []
        if self.templates_dir.exists():
            for file in self.templates_dir.glob("*.yaml"):
                profile_name = file.stem
                profiles.append(profile_name)
        return sorted(profiles)

    def load_profile(self, profile_name: str = "default") -> PromptTemplate:
        """Load a specific prompt profile

        Args:
            profile_name: Name of the profile to load (without .yaml extension)

        Returns:
            PromptTemplate object

        Raises:
            FileNotFoundError: If profile doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        # Check cache first
        if profile_name in self._templates_cache:
            return self._templates_cache[profile_name]

        # Load from file
        template_path = self.templates_dir / f"{profile_name}.yaml"

        if not template_path.exists():
            available = self.list_available_profiles()
            raise FileNotFoundError(
                f"Profile '{profile_name}' not found. "
                f"Available profiles: {', '.join(available)}"
            )

        with open(template_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        template = PromptTemplate(data)
        self._templates_cache[profile_name] = template
        return template

    def reload_profile(self, profile_name: str) -> PromptTemplate:
        """Reload a profile from disk (bypassing cache)

        Args:
            profile_name: Name of the profile to reload

        Returns:
            PromptTemplate object
        """
        # Remove from cache if present
        if profile_name in self._templates_cache:
            del self._templates_cache[profile_name]

        # Load fresh from disk
        return self.load_profile(profile_name)

    def get_profile_info(self, profile_name: str) -> dict[str, Any]:
        """Get information about a profile without fully loading it

        Args:
            profile_name: Name of the profile

        Returns:
            Dictionary with profile metadata
        """
        template = self.load_profile(profile_name)
        return {
            "name": profile_name,
            "style": template.config.style,
            "max_iterations": template.config.limits.get("default_max_iterations", 5),
            "verbosity": template.config.style.get("verbosity", "normal"),
            "has_korean_support": bool(template.config.auto_finish.get("simple_read_triggers", {}).get("korean"))
        }

    @classmethod
    def get_default_loader(cls) -> "PromptLoader":
        """Get the default prompt loader instance"""
        return cls()


# Convenience function for quick access
def load_prompt_profile(profile_name: str = "default") -> PromptTemplate:
    """Quick function to load a prompt profile

    Args:
        profile_name: Name of the profile to load

    Returns:
        PromptTemplate object
    """
    loader = PromptLoader.get_default_loader()
    return loader.load_profile(profile_name)
