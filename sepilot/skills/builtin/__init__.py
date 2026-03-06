"""Built-in skills for SEPilot"""

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import BaseSkill

logger = logging.getLogger(__name__)

# Registry of builtin skill modules and their class names
_BUILTIN_SKILLS: list[tuple[str, str]] = [
    (".code_review", "CodeReviewSkill"),
    (".explain_code", "ExplainCodeSkill"),
    (".git_helper", "GitHelperSkill"),
    (".debug_helper", "DebugHelperSkill"),
    (".test_writer", "TestWriterSkill"),
    (".frontend_design", "FrontendDesignSkill"),
    (".fastapi_design", "FastAPIDesignSkill"),
    (".project_intro", "ProjectIntroSkill"),
]


def get_builtin_skills() -> list["BaseSkill"]:
    """Get all built-in skill instances.

    Loads skills dynamically from the registry, gracefully handling
    import failures for individual skills.
    """
    skills = []

    for module_name, class_name in _BUILTIN_SKILLS:
        try:
            module = importlib.import_module(module_name, package=__package__)
            skill_class = getattr(module, class_name)
            skills.append(skill_class())
        except (ImportError, AttributeError) as e:
            logger.debug(f"Failed to load builtin skill {class_name}: {e}")

    return skills
