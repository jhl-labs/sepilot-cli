"""SEPilot Skills System

Skills are specialized plugins that provide domain-specific capabilities.
Similar to Claude Code's Skill tool.
"""

from .base import BaseSkill, SkillResult
from .manager import SkillManager, get_skill_manager

__all__ = [
    "SkillManager",
    "get_skill_manager",
    "BaseSkill",
    "SkillResult",
]
