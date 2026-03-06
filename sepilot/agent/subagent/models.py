"""SubAgent 시스템의 데이터 모델"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class TaskPriority(int, Enum):
    """작업 우선순위"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class SubAgentTask:
    """SubAgent에게 할당될 작업

    Attributes:
        task_id: 작업 고유 ID
        description: 작업 설명
        context: 작업 실행에 필요한 컨텍스트 데이터
        dependencies: 이 작업이 의존하는 다른 작업의 ID 목록
        priority: 작업 우선순위 (0: 낮음, 1: 보통, 2: 높음, 3: 긴급)
        agent_type: 이 작업을 처리할 SubAgent 타입 (예: "analyzer", "codegen")
        timeout: 최대 실행 시간 (초)
    """

    task_id: str
    description: str
    context: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    priority: int = TaskPriority.NORMAL
    agent_type: str | None = None
    timeout: float = 300.0  # 5분 기본

    def has_dependencies(self) -> bool:
        """의존성이 있는지 확인"""
        return len(self.dependencies) > 0

    def can_execute_after(self, completed_task_ids: set) -> bool:
        """모든 의존성이 완료되었는지 확인"""
        return set(self.dependencies).issubset(completed_task_ids)


@dataclass
class SubAgentResult:
    """SubAgent 실행 결과

    Attributes:
        task_id: 실행한 작업 ID
        status: 실행 상태 ("success", "failure", "partial")
        output: 실행 결과 데이터
        error: 에러 메시지 (실패 시)
        execution_time: 실행 시간 (초)
        tokens_used: 사용한 LLM 토큰 수
        metadata: 추가 메타데이터
    """

    task_id: str
    status: TaskStatus = TaskStatus.SUCCESS
    output: Any = None
    error: str | None = None
    execution_time: float = 0.0
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """성공 여부 확인"""
        return self.status == TaskStatus.SUCCESS

    def is_failure(self) -> bool:
        """실패 여부 확인"""
        return self.status == TaskStatus.FAILURE


@dataclass
class ExecutionProgress:
    """실행 진행 상황

    Attributes:
        total_tasks: 전체 작업 수
        completed_tasks: 완료된 작업 수
        failed_tasks: 실패한 작업 수
        running_tasks: 실행 중인 작업 수
        pending_tasks: 대기 중인 작업 수
    """

    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0

    @property
    def progress_percentage(self) -> float:
        """진행률 (0-100)"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    @property
    def success_rate(self) -> float:
        """성공률 (0-100)"""
        finished = self.completed_tasks + self.failed_tasks
        if finished == 0:
            return 0.0
        return (self.completed_tasks / finished) * 100

    def update_status(self, old_status: TaskStatus, new_status: TaskStatus):
        """작업 상태 변경 시 카운터 업데이트"""
        # 이전 상태 카운터 감소
        if old_status == TaskStatus.RUNNING:
            self.running_tasks = max(0, self.running_tasks - 1)
        elif old_status == TaskStatus.PENDING:
            self.pending_tasks = max(0, self.pending_tasks - 1)

        # 새 상태 카운터 증가
        if new_status == TaskStatus.SUCCESS or new_status == TaskStatus.PARTIAL:
            self.completed_tasks += 1
        elif new_status == TaskStatus.FAILURE:
            self.failed_tasks += 1
        elif new_status == TaskStatus.RUNNING:
            self.running_tasks += 1
        elif new_status == TaskStatus.PENDING:
            self.pending_tasks += 1


@dataclass
class AggregatedResult:
    """통합된 최종 결과

    Attributes:
        main_task: 메인 작업 설명
        subtask_results: 각 하위 작업의 결과
        final_output: 통합된 최종 결과
        total_execution_time: 전체 실행 시간 (초)
        total_tokens_used: 전체 사용 토큰 수
        success_rate: 성공률 (0-100)
        metadata: 추가 메타데이터
    """

    main_task: str
    subtask_results: dict[str, SubAgentResult]
    final_output: str = ""
    total_execution_time: float = 0.0
    total_tokens_used: int = 0
    success_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        main_task: str,
        results: dict[str, SubAgentResult]
    ) -> "AggregatedResult":
        """SubAgentResult 목록으로부터 생성"""
        total_time = sum(r.execution_time for r in results.values())
        total_tokens = sum(r.tokens_used for r in results.values())

        successful = sum(1 for r in results.values() if r.is_success())
        total = len(results)
        success_rate = (successful / total * 100) if total > 0 else 0.0

        return cls(
            main_task=main_task,
            subtask_results=results,
            total_execution_time=total_time,
            total_tokens_used=total_tokens,
            success_rate=success_rate
        )


@dataclass
class ExecutionPlan:
    """작업 실행 계획

    Attributes:
        phases: 실행 단계별 작업 그룹
                각 단계는 병렬로 실행 가능한 작업들의 리스트
        strategy: 실행 전략 ("parallel", "sequential", "mixed")
    """

    phases: list[list[SubAgentTask]] = field(default_factory=list)
    strategy: str = "mixed"

    def add_phase(self, tasks: list[SubAgentTask]):
        """실행 단계 추가"""
        self.phases.append(tasks)

    @property
    def total_tasks(self) -> int:
        """전체 작업 수"""
        return sum(len(phase) for phase in self.phases)

    @property
    def total_phases(self) -> int:
        """전체 단계 수"""
        return len(self.phases)

    def get_phase(self, phase_index: int) -> list[SubAgentTask]:
        """특정 단계의 작업 목록 반환"""
        if 0 <= phase_index < len(self.phases):
            return self.phases[phase_index]
        return []
