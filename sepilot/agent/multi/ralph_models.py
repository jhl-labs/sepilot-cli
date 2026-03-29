"""Ralph-Loop orchestration data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sepilot.agent.multi.models import TaskAssignment


class RalphAction(str, Enum):
    """Ralph orchestrator decision actions."""

    ASSIGN = "assign"
    VERIFY = "verify"
    WAIT = "wait"
    DONE = "done"
    ABORT = "abort"


@dataclass
class TeamChange:
    """A change to the agent team composition."""

    type: str  # "add" | "remove"
    role: str
    agent_cmd: str = ""
    system_prompt: str = ""


@dataclass
class RalphDecision:
    """A decision made by Ralph after reviewing a round."""

    reasoning: str
    round_summary: str
    action: RalphAction
    team_changes: list[TeamChange] = field(default_factory=list)
    assignments: list[TaskAssignment] = field(default_factory=list)
    verify_task: str | None = None
    verify_mode: str = "agent"  # "command" (subprocess) | "agent" (CLI 에이전트)

    def __post_init__(self):
        """Detach list fields from caller-owned containers."""
        self.team_changes = list(self.team_changes)
        self.assignments = list(self.assignments)


@dataclass
class RalphContext:
    """Context provided to Ralph for decision-making."""

    task: str
    round_num: int
    max_rounds: int
    previous_summaries: list[str]
    current_results: dict[str, str]
    active_agents: dict[str, str]  # agent_id -> status
    available_clis: list[str]
    run_dir: str
    git_diff_stat: str = ""    # git diff --stat 출력
    git_status: str = ""       # git status --short 출력

    def __post_init__(self):
        """Detach prompt context fields from caller-owned containers."""
        self.previous_summaries = list(self.previous_summaries)
        self.current_results = dict(self.current_results)
        self.active_agents = dict(self.active_agents)
        self.available_clis = list(self.available_clis)

    def to_prompt(self) -> str:
        """Build a prompt string for the Ralph LLM call."""
        parts: list[str] = []

        parts.append(f"[원본 요청]\n{self.task}")
        parts.append(f"[라운드 {self.round_num}/{self.max_rounds}]")

        parts.append("[이전 라운드 요약]")
        if self.previous_summaries:
            for i, s in enumerate(self.previous_summaries, 1):
                parts.append(f"  R{i}: {s}")
        else:
            parts.append("  (없음)")

        parts.append("[현재 에이전트]")
        for aid, status in self.active_agents.items():
            parts.append(f"  {aid}: {status}")

        if self.git_diff_stat:
            parts.append("[파일 변경 사항 (git diff --stat)]")
            parts.append(f"  {self.git_diff_stat}")

        if self.git_status:
            parts.append("[워킹 디렉토리 상태 (git status)]")
            parts.append(f"  {self.git_status}")
            parts.append("  주의: 같은 파일을 여러 에이전트가 동시에 수정하지 않도록 작업을 분배하세요.")

        parts.append("[현재 라운드 결과]")
        if self.current_results:
            for aid, result in self.current_results.items():
                parts.append(f"  {aid}: {result}")
        else:
            parts.append("  (없음)")

        parts.append("[사용 가능한 CLI]")
        parts.append(f"  {', '.join(self.available_clis)}")

        parts.append(f"[결과 파일]\n  {self.run_dir}")

        return "\n".join(parts)


@dataclass
class RalphResult:
    """Final result of a Ralph-Loop execution."""

    task: str
    rounds_executed: int
    final_action: str
    final_results: dict[str, str]
    run_dir: str
    round_summaries: list[str]

    def __post_init__(self):
        """Detach result containers from caller-owned containers."""
        self.final_results = dict(self.final_results)
        self.round_summaries = list(self.round_summaries)
