"""Built-in skills for SEPilot — auto-discovered from this package."""

import importlib
import inspect
import logging
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import BaseSkill

logger = logging.getLogger(__name__)


def get_builtin_skills() -> list["BaseSkill"]:
    """Discover and instantiate all BaseSkill subclasses in this package."""
    from ..base import BaseSkill

    skills: list[BaseSkill] = []

    for _finder, modname, ispkg in pkgutil.iter_modules(__path__):
        if ispkg:
            continue
        try:
            module = importlib.import_module(f".{modname}", __package__)
        except Exception as e:
            logger.debug("Failed to import builtin skill module %s: %s", modname, e)
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseSkill) and obj is not BaseSkill and obj.__module__ == module.__name__:
                try:
                    skills.append(obj())
                except Exception as e:
                    logger.debug("Failed to instantiate %s: %s", _name, e)

    return skills
