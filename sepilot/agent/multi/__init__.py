"""Multi-agent orchestration system."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sepilot.agent.multi.inbox import Inbox
    from sepilot.agent.multi.models import (
        AgentRole,
        Message,
        MessageType,
        PMAction,
        PMDecision,
        Strategy,
        TaskAssignment,
        TeamPreset,
    )
    from sepilot.agent.multi.pm import PMAgent
    from sepilot.agent.multi.preset import PresetManager
    from sepilot.agent.multi.ralph_loop import RalphLoop
    from sepilot.agent.multi.ralph_models import (
        RalphAction,
        RalphContext,
        RalphDecision,
        RalphResult,
        TeamChange,
    )
    from sepilot.agent.multi.team import AgentTeam, ManagedAgent

__all__ = [
    "AgentRole",
    "AgentTeam",
    "Inbox",
    "ManagedAgent",
    "Message",
    "MessageType",
    "PMAction",
    "PMAgent",
    "PMDecision",
    "PresetManager",
    "RalphAction",
    "RalphContext",
    "RalphDecision",
    "RalphLoop",
    "RalphResult",
    "Strategy",
    "TaskAssignment",
    "TeamChange",
    "TeamPreset",
]

_EXPORT_MODULES = {
    "AgentRole": "sepilot.agent.multi.models",
    "AgentTeam": "sepilot.agent.multi.team",
    "Inbox": "sepilot.agent.multi.inbox",
    "ManagedAgent": "sepilot.agent.multi.team",
    "Message": "sepilot.agent.multi.models",
    "MessageType": "sepilot.agent.multi.models",
    "PMAction": "sepilot.agent.multi.models",
    "PMAgent": "sepilot.agent.multi.pm",
    "PMDecision": "sepilot.agent.multi.models",
    "PresetManager": "sepilot.agent.multi.preset",
    "RalphAction": "sepilot.agent.multi.ralph_models",
    "RalphContext": "sepilot.agent.multi.ralph_models",
    "RalphDecision": "sepilot.agent.multi.ralph_models",
    "RalphLoop": "sepilot.agent.multi.ralph_loop",
    "RalphResult": "sepilot.agent.multi.ralph_models",
    "Strategy": "sepilot.agent.multi.models",
    "TaskAssignment": "sepilot.agent.multi.models",
    "TeamChange": "sepilot.agent.multi.ralph_models",
    "TeamPreset": "sepilot.agent.multi.models",
}

_SUBMODULES = {
    "inbox": "sepilot.agent.multi.inbox",
    "models": "sepilot.agent.multi.models",
    "pm": "sepilot.agent.multi.pm",
    "preset": "sepilot.agent.multi.preset",
    "ralph_loop": "sepilot.agent.multi.ralph_loop",
    "ralph_models": "sepilot.agent.multi.ralph_models",
    "team": "sepilot.agent.multi.team",
}


def __getattr__(name: str):
    """Lazily import exports so models stay usable if runtime-only modules fail."""
    submodule_name = _SUBMODULES.get(name)
    if submodule_name is not None:
        module = import_module(submodule_name)
        globals()[name] = module
        return module

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
