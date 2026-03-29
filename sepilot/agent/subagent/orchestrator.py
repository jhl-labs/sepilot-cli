"""SubAgent Orchestrator - SubAgent 조율 및 관리

Enhanced with Claude Code-style task management:
- TaskRegistry integration for centralized state tracking
- TaskExecutor for cancel/retry support
- Real-time progress callbacks
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .base_subagent import BaseSubAgent
from .models import (
    AggregatedResult,
    ExecutionPlan,
    ExecutionProgress,
    SubAgentResult,
    SubAgentTask,
    TaskPriority,
    TaskStatus,
)

if TYPE_CHECKING:
    from sepilot.agent.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class SubAgentOrchestrator:
    """SubAgent 조율자

    복잡한 작업을 분해하고, SubAgent에게 위임하며, 결과를 통합하는 역할

    Enhanced with Claude Code-style task management:
    - TaskRegistry for centralized state tracking
    - Individual task cancellation support
    - Real-time progress callbacks
    - Automatic retry on failure

    Attributes:
        max_parallel: 최대 동시 실행 SubAgent 수
        llm: 작업 분해에 사용할 LLM
        subagents: 등록된 SubAgent 목록
        active_tasks: 현재 실행 중인 작업
        results: 완료된 작업 결과
        registry: Optional TaskRegistry for enhanced tracking
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        max_parallel: int = 3,
        subagents: list[BaseSubAgent] | None = None,
        use_task_registry: bool = True,
        progress_callback: Callable[[str, str, float], None] | None = None
    ):
        """Initialize orchestrator.

        Args:
            llm: Language model for task decomposition
            max_parallel: Maximum concurrent SubAgents
            subagents: Pre-registered SubAgents
            use_task_registry: Use TaskRegistry for enhanced tracking
            progress_callback: Callback(task_id, status, progress) for updates
        """
        self.llm = llm
        self.max_parallel = max_parallel
        self.subagents: list[BaseSubAgent] = subagents or []
        self.active_tasks: dict[str, SubAgentTask] = {}
        self.results: dict[str, SubAgentResult] = {}
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.progress_callback = progress_callback

        # TaskRegistry integration
        self._registry: TaskRegistry | None = None
        self._use_task_registry = use_task_registry
        self._cancel_event = asyncio.Event()
        self._task_to_registry_id: dict[str, str] = {}

        if use_task_registry:
            self._init_registry()

    def reset_execution_state(self) -> None:
        """Reset per-run execution state before starting a new orchestration."""
        self.active_tasks.clear()
        self.results = {}
        self._task_to_registry_id = {}
        self._cancel_event = asyncio.Event()

    def _init_registry(self) -> None:
        """Initialize TaskRegistry integration."""
        try:
            from sepilot.agent.task_registry import get_task_registry
            self._registry = get_task_registry()
            logger.debug("TaskRegistry integration enabled")
        except ImportError:
            logger.warning("TaskRegistry not available, using legacy mode")
            self._use_task_registry = False

    @property
    def registry(self) -> "TaskRegistry | None":
        """Get the task registry."""
        return self._registry

    def register_subagent(self, subagent: BaseSubAgent):
        """SubAgent 등록

        Args:
            subagent: 등록할 SubAgent
        """
        self.subagents.append(subagent)
        logger.info(f"Registered SubAgent: {subagent.agent_id} ({subagent.agent_type})")

    async def decompose_task(
        self,
        main_task: str,
        context: dict | None = None
    ) -> list[SubAgentTask]:
        """작업 분해

        복잡한 작업을 여러 하위 작업으로 분해합니다.

        Args:
            main_task: 분해할 메인 작업
            context: 작업 컨텍스트

        Returns:
            하위 작업 목록
        """
        logger.info(f"Decomposing task: {main_task}")

        if not self.llm:
            # LLM이 없으면 단일 작업으로 반환
            task_id = str(uuid.uuid4())
            return [SubAgentTask(
                task_id=task_id,
                description=main_task,
                context=context or {}
            )]

        # LLM을 사용하여 작업 분해
        decomposition_prompt = f"""다음 작업을 독립적인 하위 작업들로 분해해주세요.

메인 작업: {main_task}

각 하위 작업은 다음 형식으로 작성해주세요:
1. [작업 설명]
2. [작업 설명]
...

각 작업은 가능한 독립적이어야 하며, 병렬로 실행 가능해야 합니다.
만약 순서가 중요한 작업이 있다면 "→" 기호로 의존성을 표시해주세요.
예: 1. 파일 목록 추출 → 2. 각 파일 분석

작업이 단순하여 분해가 불필요하면 "NOT_NEEDED"라고만 응답해주세요."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=decomposition_prompt)])
            response_text = response.content.strip()

            # "NOT_NEEDED" 응답 체크
            if "NOT_NEEDED" in response_text.upper():
                task_id = str(uuid.uuid4())
                return [SubAgentTask(
                    task_id=task_id,
                    description=main_task,
                    context=context or {}
                )]

            # 작업 파싱
            subtasks = self._parse_subtasks(response_text, context or {})

            logger.info(f"Decomposed into {len(subtasks)} subtasks")
            return subtasks

        except Exception as e:
            logger.error(f"Failed to decompose task: {e}")
            # 실패 시 단일 작업으로 반환
            task_id = str(uuid.uuid4())
            return [SubAgentTask(
                task_id=task_id,
                description=main_task,
                context=context or {}
            )]

    def _parse_subtasks(
        self,
        response_text: str,
        context: dict
    ) -> list[SubAgentTask]:
        """LLM 응답에서 하위 작업 파싱

        Args:
            response_text: LLM 응답 텍스트
            context: 작업 컨텍스트

        Returns:
            파싱된 하위 작업 목록
        """
        subtasks = []
        dependencies_map: dict[int, list[int]] = defaultdict(list)

        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 번호 매기기 패턴: "1. ", "2. " 등
            if line[0].isdigit() and '. ' in line:
                # 의존성 체크 (→ 기호)
                if '→' in line:
                    # 예: "1. 파일 목록 추출 → 2. 각 파일 분석"
                    parts = line.split('→')
                    for part in parts:
                        task_num, task_desc = self._extract_task_number_and_desc(part)
                        if task_num is not None:
                            if len(subtasks) >= task_num:
                                # 이미 추가된 작업
                                continue

                            task_id = f"subtask_{task_num}"
                            subtask = SubAgentTask(
                                task_id=task_id,
                                description=task_desc,
                                context=context.copy(),
                                dependencies=[],
                                priority=TaskPriority.NORMAL
                            )
                            subtasks.append(subtask)

                            # 의존성 기록
                            if task_num > 1:
                                dependencies_map[task_num].append(task_num - 1)
                else:
                    task_num, task_desc = self._extract_task_number_and_desc(line)
                    if task_num is not None:
                        task_id = f"subtask_{task_num}"
                        subtask = SubAgentTask(
                            task_id=task_id,
                            description=task_desc,
                            context=context.copy(),
                            dependencies=[],
                            priority=TaskPriority.NORMAL
                        )
                        subtasks.append(subtask)

        # 의존성 설정
        for task_num, dep_nums in dependencies_map.items():
            if task_num <= len(subtasks):
                subtask = subtasks[task_num - 1]
                subtask.dependencies = [f"subtask_{d}" for d in dep_nums]

        return subtasks

    def _extract_task_number_and_desc(self, text: str) -> tuple:
        """텍스트에서 작업 번호와 설명 추출

        Args:
            text: 파싱할 텍스트

        Returns:
            (작업 번호, 작업 설명) 튜플
        """
        text = text.strip()

        # "1. 작업 설명" 패턴
        if '. ' in text:
            parts = text.split('. ', 1)
            try:
                task_num = int(parts[0].strip())
                task_desc = parts[1].strip()
                return (task_num, task_desc)
            except ValueError:
                pass

        return (None, text)

    def create_execution_plan(
        self,
        tasks: list[SubAgentTask],
        strategy: str = "auto"
    ) -> ExecutionPlan:
        """실행 계획 생성

        작업 간 의존성을 분석하여 실행 단계를 구성합니다.

        Args:
            tasks: 실행할 작업 목록
            strategy: 실행 전략 ("auto", "parallel", "sequential")

        Returns:
            실행 계획
        """
        if strategy == "sequential":
            # 모든 작업을 순차 실행
            plan = ExecutionPlan(strategy="sequential")
            for task in tasks:
                plan.add_phase([task])
            return plan

        if strategy == "parallel":
            # 모든 작업을 병렬 실행 (의존성 무시)
            plan = ExecutionPlan(strategy="parallel")
            plan.add_phase(tasks)
            return plan

        # "auto": 의존성 기반 자동 계획
        plan = ExecutionPlan(strategy="mixed")

        # 위상 정렬 (Topological Sort)을 사용하여 실행 순서 결정
        _task_map = {task.task_id: task for task in tasks}  # noqa: F841
        completed_ids: set[str] = set()

        while len(completed_ids) < len(tasks):
            # 현재 실행 가능한 작업들 찾기
            ready_tasks = []

            for task in tasks:
                if task.task_id in completed_ids:
                    continue

                if task.can_execute_after(completed_ids):
                    ready_tasks.append(task)

            if not ready_tasks:
                # 순환 의존성 또는 잘못된 의존성 — 의존성을 무시하고 강제 실행
                remaining = [t for t in tasks if t.task_id not in completed_ids]
                if remaining:
                    logger.warning(
                        "Circular dependency or invalid dependencies detected "
                        "for %d tasks: %s. Clearing dependencies and forcing execution.",
                        len(remaining),
                        [t.task_id for t in remaining],
                    )
                    # 의존성을 제거하여 실행 가능하게 만듦
                    for t in remaining:
                        t.dependencies = []
                    plan.add_phase(remaining)
                break

            # 현재 단계에 준비된 작업들 추가
            plan.add_phase(ready_tasks)

            # 완료 목록에 추가
            for task in ready_tasks:
                completed_ids.add(task.task_id)

        logger.info(
            f"Created execution plan with {plan.total_phases} phases "
            f"for {plan.total_tasks} tasks"
        )

        return plan

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        progress_callback: Callable | None = None
    ) -> dict[str, SubAgentResult]:
        """실행 계획에 따라 작업 실행

        Args:
            plan: 실행 계획
            progress_callback: 진행 상황 콜백 함수

        Returns:
            모든 작업의 결과
        """
        logger.info(f"Executing plan with {plan.total_phases} phases")
        self.reset_execution_state()

        all_results: dict[str, SubAgentResult] = {}
        progress = ExecutionProgress(total_tasks=plan.total_tasks)

        for phase_index, phase_tasks in enumerate(plan.phases):
            logger.info(
                f"Executing phase {phase_index + 1}/{plan.total_phases} "
                f"with {len(phase_tasks)} tasks"
            )

            # 현재 단계의 작업들을 병렬 실행
            phase_results = await self.execute_parallel(phase_tasks)

            # 결과 수집
            all_results.update(phase_results)
            self.results.update(phase_results)

            # 진행 상황 업데이트
            for result in phase_results.values():
                if result.is_success():
                    progress.update_status(TaskStatus.RUNNING, TaskStatus.SUCCESS)
                else:
                    progress.update_status(TaskStatus.RUNNING, TaskStatus.FAILURE)

            # 콜백 호출
            if progress_callback:
                progress_callback(progress)

        logger.info(
            f"Plan execution completed: {progress.completed_tasks} succeeded, "
            f"{progress.failed_tasks} failed"
        )

        return all_results

    async def execute_parallel(
        self,
        tasks: list[SubAgentTask]
    ) -> dict[str, SubAgentResult]:
        """작업들을 병렬로 실행

        Enhanced with TaskRegistry for centralized state tracking.

        Args:
            tasks: 실행할 작업 목록

        Returns:
            각 작업의 결과
        """
        if not tasks:
            return {}

        # Register tasks in TaskRegistry if available
        if self._registry:
            for task in tasks:
                from sepilot.agent.task_registry import TaskPriority as RegPriority
                priority_value = getattr(task.priority, "value", task.priority)
                try:
                    registry_priority = RegPriority(int(priority_value))
                except (TypeError, ValueError):
                    registry_priority = RegPriority.NORMAL
                reg_task = self._registry.register(
                    name=task.description[:50],
                    description=task.description,
                    priority=registry_priority,
                    dependencies=[
                        self._task_to_registry_id.get(dep)
                        for dep in task.dependencies
                        if self._task_to_registry_id.get(dep)
                    ],
                    metadata={"subagent_task_id": task.task_id}
                )
                self._task_to_registry_id[task.task_id] = reg_task.task_id

        # 각 작업에 SubAgent 할당
        task_agent_pairs = []
        for task in tasks:
            agent = self._assign_subagent(task)
            task_agent_pairs.append((task, agent))

        # 병렬 실행
        results = await asyncio.gather(
            *[
                self._execute_with_limit(agent, task)
                for task, agent in task_agent_pairs
            ],
            return_exceptions=True
        )

        # 결과 수집
        result_dict = {}
        for (task, _agent), result in zip(task_agent_pairs, results, strict=False):
            if isinstance(result, Exception):
                # 예외 발생 시 실패 결과 생성
                result_dict[task.task_id] = SubAgentResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILURE,
                    error=str(result)
                )
                # Update registry
                if self._registry and task.task_id in self._task_to_registry_id:
                    reg_id = self._task_to_registry_id[task.task_id]
                    self._registry.fail(reg_id, str(result))
            else:
                result_dict[task.task_id] = result
                # Update registry
                if self._registry and task.task_id in self._task_to_registry_id:
                    reg_id = self._task_to_registry_id[task.task_id]
                    if result.is_success():
                        self._registry.complete(reg_id, result=result.output)
                    else:
                        self._registry.fail(reg_id, result.error or "Unknown error")

        return result_dict

    async def _execute_with_limit(
        self,
        agent: BaseSubAgent,
        task: SubAgentTask
    ) -> SubAgentResult:
        """세마포어를 사용하여 동시 실행 수 제한

        Enhanced with cancel event checking and progress callbacks.

        Args:
            agent: 실행할 SubAgent
            task: 실행할 작업

        Returns:
            실행 결과
        """
        # Check cancel event
        if self._cancel_event.is_set():
            return SubAgentResult(
                task_id=task.task_id,
                status=TaskStatus.FAILURE,
                error="Cancelled"
            )

        async with self.semaphore:
            # Track in active_tasks
            self.active_tasks[task.task_id] = task

            # Update registry state
            if self._registry and task.task_id in self._task_to_registry_id:
                reg_id = self._task_to_registry_id[task.task_id]
                self._registry.start(reg_id)

            # Notify progress callback
            if self.progress_callback:
                self.progress_callback(task.task_id, "running", 0.0)

            try:
                result = await agent.execute(task)

                # Notify completion
                if self.progress_callback:
                    progress = 1.0 if result.is_success() else 0.0
                    status = "completed" if result.is_success() else "failed"
                    self.progress_callback(task.task_id, status, progress)

                return result
            except asyncio.CancelledError:
                # Handle cancellation
                if self._registry and task.task_id in self._task_to_registry_id:
                    reg_id = self._task_to_registry_id[task.task_id]
                    self._registry.cancel(reg_id, "Task cancelled")

                if self.progress_callback:
                    self.progress_callback(task.task_id, "cancelled", 0.0)

                return SubAgentResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILURE,
                    error="Cancelled"
                )
            finally:
                # Remove from active tasks
                self.active_tasks.pop(task.task_id, None)

    def cancel_task(self, task_id: str, reason: str = "") -> bool:
        """Cancel a specific task.

        Args:
            task_id: Task ID to cancel
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        if self._registry and task_id in self._task_to_registry_id:
            reg_id = self._task_to_registry_id[task_id]
            self._registry.cancel(reg_id, reason)
            return True
        return False

    def cancel_all(self, reason: str = "Bulk cancellation") -> int:
        """Cancel all running tasks.

        Args:
            reason: Cancellation reason

        Returns:
            Number of tasks cancelled
        """
        self._cancel_event.set()

        count = 0
        if self._registry:
            for _task_id, reg_id in self._task_to_registry_id.items():
                task_info = self._registry.get(reg_id)
                if task_info and task_info.is_running():
                    self._registry.cancel(reg_id, reason)
                    count += 1

        return count

    def get_running_stats(self) -> dict:
        """Get statistics about running tasks.

        Returns:
            Dictionary with task statistics
        """
        if self._registry:
            return self._registry.get_stats()

        return {
            "total": len(self.active_tasks) + len(self.results),
            "running": len(self.active_tasks),
            "completed": sum(1 for r in self.results.values() if r.is_success()),
            "failed": sum(1 for r in self.results.values() if r.is_failure())
        }

    def _assign_subagent(self, task: SubAgentTask) -> BaseSubAgent:
        """작업에 적절한 SubAgent 할당

        Args:
            task: 할당할 작업

        Returns:
            할당된 SubAgent
        """
        # agent_type이 지정된 경우
        if task.agent_type:
            for agent in self.subagents:
                if agent.agent_type == task.agent_type:
                    return agent
            # 매칭 실패 시: developer 에이전트로 폴백 (도구 보유), 없으면 첫 번째 에이전트
            logger.warning(
                "No exact SubAgent match for task %s agent_type=%s, "
                "falling back to developer agent",
                task.task_id,
                task.agent_type,
            )
            fallback = self._find_fallback_agent()
            if fallback:
                return fallback
            raise ValueError(
                f"No SubAgent registered for agent_type='{task.agent_type}' "
                f"and no fallback available. "
                f"Registered: {[a.agent_type for a in self.subagents]}"
            )

        # can_handle()로 적절한 SubAgent 찾기
        for agent in self.subagents:
            if agent.can_handle(task):
                logger.info(f"Assigned {agent.agent_id} to task {task.task_id}")
                return agent

        # 적절한 SubAgent가 없으면 developer로 폴백
        logger.warning(f"No suitable SubAgent found for task {task.task_id}, using fallback")
        fallback = self._find_fallback_agent()
        if fallback:
            return fallback
        raise ValueError(
            f"No suitable SubAgent for task {task.task_id} and no fallback available"
        )

    def _find_fallback_agent(self) -> BaseSubAgent | None:
        """developer 에이전트를 폴백으로 찾고, 없으면 첫 번째 에이전트 반환."""
        for agent in self.subagents:
            if agent.agent_type == "developer":
                return agent
        return self.subagents[0] if self.subagents else None

    async def aggregate_results(
        self,
        main_task: str,
        results: dict[str, SubAgentResult],
        aggregation_strategy: str = "concatenate"
    ) -> AggregatedResult:
        """결과 통합

        Args:
            main_task: 메인 작업 설명
            results: 각 하위 작업의 결과
            aggregation_strategy: 통합 전략 ("concatenate", "summarize", "structured")

        Returns:
            통합된 결과
        """
        aggregated = AggregatedResult.from_results(main_task, results)

        if aggregation_strategy == "concatenate":
            aggregated.final_output = self._merge_by_concatenation(results)
        elif aggregation_strategy == "summarize":
            aggregated.final_output = await self._merge_by_summarization(results)
        elif aggregation_strategy == "structured":
            aggregated.final_output = self._merge_by_structure(results)
        else:
            aggregated.final_output = self._merge_by_concatenation(results)

        return aggregated

    def _merge_by_concatenation(self, results: dict[str, SubAgentResult]) -> str:
        """결과를 단순 연결하여 통합"""
        sections = []

        for task_id, result in results.items():
            if result.is_success():
                sections.append(f"### {task_id}\n{result.output}")
            else:
                sections.append(f"### {task_id}\n❌ Failed: {result.error}")

        return "\n\n".join(sections)

    async def _merge_by_summarization(self, results: dict[str, SubAgentResult]) -> str:
        """LLM을 사용하여 결과 요약"""
        if not self.llm:
            return self._merge_by_concatenation(results)

        combined = self._merge_by_concatenation(results)

        prompt = f"""다음은 여러 하위 작업의 결과입니다:

{combined}

위 결과들을 종합하여 간결하고 명확한 최종 보고서를 작성해주세요."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Failed to summarize results: {e}")
            return combined

    def _merge_by_structure(self, results: dict[str, SubAgentResult]) -> str:
        """구조화된 형식으로 결과 통합"""
        output = []
        output.append("# 작업 실행 결과\n")

        successful = [r for r in results.values() if r.is_success()]
        failed = [r for r in results.values() if r.is_failure()]

        output.append("## 요약")
        output.append(f"- 전체 작업: {len(results)}개")
        output.append(f"- 성공: {len(successful)}개")
        output.append(f"- 실패: {len(failed)}개\n")

        if successful:
            output.append("## 성공한 작업\n")
            for result in successful:
                output.append(f"### {result.task_id}")
                output.append(f"{result.output}\n")

        if failed:
            output.append("## 실패한 작업\n")
            for result in failed:
                output.append(f"### {result.task_id}")
                output.append(f"❌ 오류: {result.error}\n")

        return "\n".join(output)
