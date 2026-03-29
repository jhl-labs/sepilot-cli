"""SubAgent 시스템

복잡한 작업을 여러 하위 작업으로 분해하고 병렬 실행하는 시스템
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_subagent import BaseSubAgent, SimpleSubAgent
    from .models import (
        AggregatedResult,
        ExecutionPlan,
        ExecutionProgress,
        SubAgentResult,
        SubAgentTask,
        TaskPriority,
        TaskStatus,
    )
    from .orchestrator import SubAgentOrchestrator
    from .specialized_agents import (
        AnalyzerSubAgent,
        CodeGenSubAgent,
        DocumentationSubAgent,
        RefactoringSubAgent,
        TestingSubAgent,
    )
    from .team_agents import (
        ArchitectAgent,
        DebuggerAgent,
        DeveloperAgent,
        DevOpsAgent,
        PMAgent,
        ResearcherAgent,
        SecurityReviewerAgent,
        TesterAgent,
    )
    from .team_models import (
        InterAgentMessage,
        PhaseGateResult,
        TeamExecutionPlan,
        TeamPhase,
        TeamRole,
        TeamTaskAssignment,
    )
    from .team_orchestrator import TeamOrchestrator
    from .worktree_manager import Worktree, WorktreeManager

__all__ = [
    "SubAgentTask",
    "SubAgentResult",
    "ExecutionProgress",
    "ExecutionPlan",
    "AggregatedResult",
    "TaskStatus",
    "TaskPriority",
    "BaseSubAgent",
    "SimpleSubAgent",
    "SubAgentOrchestrator",
    "AnalyzerSubAgent",
    "CodeGenSubAgent",
    "TestingSubAgent",
    "DocumentationSubAgent",
    "RefactoringSubAgent",
    "TeamPhase",
    "TeamRole",
    "TeamTaskAssignment",
    "PhaseGateResult",
    "TeamExecutionPlan",
    "InterAgentMessage",
    "PMAgent",
    "DeveloperAgent",
    "TesterAgent",
    "DebuggerAgent",
    "ResearcherAgent",
    "ArchitectAgent",
    "SecurityReviewerAgent",
    "DevOpsAgent",
    "TeamOrchestrator",
    "WorktreeManager",
    "Worktree",
]

_EXPORT_MODULES = {
    "AggregatedResult": "sepilot.agent.subagent.models",
    "AnalyzerSubAgent": "sepilot.agent.subagent.specialized_agents",
    "ArchitectAgent": "sepilot.agent.subagent.team_agents",
    "BaseSubAgent": "sepilot.agent.subagent.base_subagent",
    "CodeGenSubAgent": "sepilot.agent.subagent.specialized_agents",
    "DebuggerAgent": "sepilot.agent.subagent.team_agents",
    "DeveloperAgent": "sepilot.agent.subagent.team_agents",
    "DevOpsAgent": "sepilot.agent.subagent.team_agents",
    "DocumentationSubAgent": "sepilot.agent.subagent.specialized_agents",
    "ExecutionPlan": "sepilot.agent.subagent.models",
    "ExecutionProgress": "sepilot.agent.subagent.models",
    "InterAgentMessage": "sepilot.agent.subagent.team_models",
    "PMAgent": "sepilot.agent.subagent.team_agents",
    "PhaseGateResult": "sepilot.agent.subagent.team_models",
    "RefactoringSubAgent": "sepilot.agent.subagent.specialized_agents",
    "ResearcherAgent": "sepilot.agent.subagent.team_agents",
    "SecurityReviewerAgent": "sepilot.agent.subagent.team_agents",
    "SimpleSubAgent": "sepilot.agent.subagent.base_subagent",
    "SubAgentOrchestrator": "sepilot.agent.subagent.orchestrator",
    "SubAgentResult": "sepilot.agent.subagent.models",
    "SubAgentTask": "sepilot.agent.subagent.models",
    "TaskPriority": "sepilot.agent.subagent.models",
    "TaskStatus": "sepilot.agent.subagent.models",
    "TeamExecutionPlan": "sepilot.agent.subagent.team_models",
    "TeamOrchestrator": "sepilot.agent.subagent.team_orchestrator",
    "TeamPhase": "sepilot.agent.subagent.team_models",
    "TeamRole": "sepilot.agent.subagent.team_models",
    "TeamTaskAssignment": "sepilot.agent.subagent.team_models",
    "TesterAgent": "sepilot.agent.subagent.team_agents",
    "TestingSubAgent": "sepilot.agent.subagent.specialized_agents",
    "Worktree": "sepilot.agent.subagent.worktree_manager",
    "WorktreeManager": "sepilot.agent.subagent.worktree_manager",
}

_SUBMODULES = {
    "base_subagent": "sepilot.agent.subagent.base_subagent",
    "models": "sepilot.agent.subagent.models",
    "orchestrator": "sepilot.agent.subagent.orchestrator",
    "specialized_agents": "sepilot.agent.subagent.specialized_agents",
    "team_agents": "sepilot.agent.subagent.team_agents",
    "team_models": "sepilot.agent.subagent.team_models",
    "team_orchestrator": "sepilot.agent.subagent.team_orchestrator",
    "worktree_manager": "sepilot.agent.subagent.worktree_manager",
}


def __getattr__(name: str):
    """Lazily import exports so model helpers remain available under partial failures."""
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
