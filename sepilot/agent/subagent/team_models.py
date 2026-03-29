"""Agent Teams 전용 데이터 모델 (Legacy)

PM 주도 역할 기반 Agent Teams 시스템에서 사용하는 데이터 모델을 정의합니다.

Note: 새로운 멀티 에이전트 모델은 ``sepilot.agent.multi.models`` 에 정의되어 있습니다.
이 모듈의 모델은 TeamOrchestrator(LLM SubAgent 기반)에서 계속 사용됩니다.
"""

from dataclasses import dataclass, field
from enum import Enum


class TeamPhase(str, Enum):
    """팀 작업 단계"""
    RESEARCH = "research"
    PLAN = "plan"
    DESIGN = "design"
    IMPLEMENT = "implement"
    TEST = "test"
    REVIEW = "review"
    DEPLOY = "deploy"


class TeamRole(str, Enum):
    """팀 역할"""
    PM = "pm"
    DEVELOPER = "developer"
    TESTER = "tester"
    DEBUGGER = "debugger"
    RESEARCHER = "researcher"
    ARCHITECT = "architect"
    SECURITY_REVIEWER = "security_reviewer"
    DEVOPS = "devops"


@dataclass
class TeamTaskAssignment:
    """팀 작업 할당 정보

    PM이 생성한 실행 계획에서 각 역할에 할당된 작업을 표현합니다.

    Attributes:
        task_id: 작업 고유 ID (예: "R1", "D1", "I1")
        role: 할당된 역할
        description: 작업 설명
        phase: 작업이 속하는 단계
        dependencies: 의존하는 다른 작업의 ID 목록
        context_from: 컨텍스트를 가져올 선행 작업 ID 목록
        acceptance_criteria: 완료 기준
    """
    task_id: str
    role: TeamRole
    description: str
    phase: TeamPhase
    dependencies: list[str] = field(default_factory=list)
    context_from: list[str] = field(default_factory=list)
    acceptance_criteria: str = ""

    def __post_init__(self):
        """Detach dependency lists from caller-owned containers."""
        self.dependencies = list(self.dependencies)
        self.context_from = list(self.context_from)


@dataclass
class PhaseGateResult:
    """단계 게이트 결과

    각 단계 완료 시 품질 게이트 체크 결과를 저장합니다.

    Attributes:
        phase: 체크한 단계
        passed: 통과 여부
        issues: 발견된 이슈 목록
        recommendations: 권장 사항
    """
    phase: TeamPhase
    passed: bool
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Detach issue lists from caller-owned containers."""
        self.issues = list(self.issues)
        self.recommendations = list(self.recommendations)


@dataclass
class TeamExecutionPlan:
    """팀 실행 계획

    PM이 생성한 전체 실행 계획을 표현합니다.

    Attributes:
        plan_id: 계획 고유 ID
        original_task: 원본 작업 설명
        assignments: 작업 할당 목록
        phases: 포함된 단계 목록 (실행 순서)
        gate_results: 단계별 게이트 결과
    """
    plan_id: str
    original_task: str
    assignments: list[TeamTaskAssignment] = field(default_factory=list)
    phases: list[TeamPhase] = field(default_factory=list)
    gate_results: list[PhaseGateResult] = field(default_factory=list)

    def __post_init__(self):
        """Detach plan lists from caller-owned containers."""
        self.assignments = list(self.assignments)
        self.phases = list(self.phases)
        self.gate_results = list(self.gate_results)

    @property
    def total_assignments(self) -> int:
        """전체 할당 수"""
        return len(self.assignments)

    def get_assignments_by_phase(self, phase: TeamPhase) -> list[TeamTaskAssignment]:
        """특정 단계의 할당 목록 반환"""
        return [a for a in self.assignments if a.phase == phase]

    def get_assignments_by_role(self, role: TeamRole) -> list[TeamTaskAssignment]:
        """특정 역할의 할당 목록 반환"""
        return [a for a in self.assignments if a.role == role]


@dataclass
class InterAgentMessage:
    """에이전트 간 메시지

    팀 에이전트 간 통신을 위한 메시지 모델입니다.

    Attributes:
        from_role: 송신 역할
        to_role: 수신 역할
        task_id: 관련 작업 ID
        content: 메시지 내용
        message_type: 메시지 유형 (result, request, feedback)
    """
    from_role: TeamRole
    to_role: TeamRole
    task_id: str
    content: str
    message_type: str = "result"
