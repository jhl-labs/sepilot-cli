"""TeamOrchestrator - PM 주도 팀 워크플로우 오케스트레이터 (Legacy)

SubAgentOrchestrator를 내부적으로 재사용하면서, PM이 생성한 역할 기반 실행 계획에 따라
단계별로 팀 에이전트를 조율합니다.

Note: tmux 기반 멀티 에이전트 팀에는 ``sepilot.agent.multi.AgentTeam`` 을 사용하세요.
TeamOrchestrator는 LLM SubAgent 기반 팀 전용이며, 향후 AgentTeam과 통합될 예정입니다.
"""

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from .base_subagent import BaseSubAgent
from .models import SubAgentResult, SubAgentTask
from .orchestrator import SubAgentOrchestrator
from .team_agents import PMAgent
from .team_models import (
    InterAgentMessage,
    PhaseGateResult,
    TeamExecutionPlan,
    TeamPhase,
    TeamRole,
    TeamTaskAssignment,
)

logger = logging.getLogger(__name__)

# 단계 실행 순서 (고정)
PHASE_ORDER = [
    TeamPhase.RESEARCH,
    TeamPhase.PLAN,
    TeamPhase.DESIGN,
    TeamPhase.IMPLEMENT,
    TeamPhase.TEST,
    TeamPhase.REVIEW,
    TeamPhase.DEPLOY,
]


class TeamOrchestrator:
    """PM 주도 팀 워크플로우 오케스트레이터

    PMAgent가 생성한 실행 계획에 따라 역할별 에이전트를 단계적으로 실행하고,
    에이전트 간 컨텍스트 전달 및 품질 게이트 체크를 수행합니다.

    SubAgentOrchestrator와의 핵심 차이:
    - 작업 분해: PM 에이전트가 역할/단계 기반으로 분해
    - 에이전트 매칭: agent_type == role.value 직접 매핑
    - 실행 순서: 고정 단계 순서 + 단계 내 의존성
    - 컨텍스트 전달: context_from으로 선행 결과 자동 주입
    - 품질 관리: REVIEW 단계 품질 게이트
    - 통신: InterAgentMessage 메시지 버스
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        max_parallel: int = 3,
    ):
        self.llm = llm
        self.max_parallel = max_parallel

        # 역할별 에이전트 등록 맵
        self._agents: dict[TeamRole, BaseSubAgent] = {}

        # PM 에이전트
        self._pm: PMAgent | None = None

        # 메시지 버스: task_id -> InterAgentMessage 목록
        self._message_bus: dict[str, list[InterAgentMessage]] = {}

        # 작업 결과 저장: task_id -> 결과 문자열
        self._task_results: dict[str, str] = {}

        # 내부 SubAgentOrchestrator (병렬 실행 재사용)
        self._inner_orchestrator = SubAgentOrchestrator(
            llm=llm,
            max_parallel=max_parallel,
            use_task_registry=False,
        )

    def _reset_execution_state(self) -> None:
        """Reset per-run team context so previous executions cannot leak forward."""
        self._message_bus = {}
        self._task_results = {}
        self._inner_orchestrator.reset_execution_state()

    def register_agent(self, role: TeamRole, agent: BaseSubAgent) -> None:
        """역할별 에이전트 등록

        Args:
            role: 팀 역할
            agent: 등록할 에이전트
        """
        self._agents[role] = agent
        self._inner_orchestrator.register_subagent(agent)

        if role == TeamRole.PM and isinstance(agent, PMAgent):
            self._pm = agent

        logger.info(f"Registered team agent: {role.value} -> {agent.agent_id}")

    async def execute_team_task(
        self,
        main_task: str,
        context: dict | None = None,
    ) -> dict[str, Any]:
        """전체 팀 워크플로우 실행

        Args:
            main_task: 메인 작업 설명
            context: 추가 컨텍스트

        Returns:
            통합 결과 딕셔너리 (output, stats, gate_results)
        """
        start_time = time.time()
        logger.info(f"Starting team task: {main_task}")
        self._reset_execution_state()

        # 1. PM이 실행 계획 생성
        plan = await self._create_plan(main_task, context)
        logger.info(
            f"PM created plan: {plan.total_assignments} assignments, "
            f"{len(plan.phases)} phases"
        )

        # 2. 단계별 실행
        all_results: dict[str, SubAgentResult] = {}
        gate_results: list[PhaseGateResult] = []

        for phase in self._ordered_phases(plan):
            assignments = plan.get_assignments_by_phase(phase)
            if not assignments:
                continue

            logger.info(
                f"Executing phase: {phase.value} "
                f"({len(assignments)} assignments)"
            )

            # a. assignments -> SubAgentTask 변환 (컨텍스트 주입)
            tasks = self._to_subagent_tasks(assignments, self._task_results, base_context=context)

            # b. 병렬 실행
            phase_results = await self._execute_phase_tasks(tasks)
            all_results.update(phase_results)

            # c. 결과를 메시지 버스에 기록
            self._record_results_to_bus(assignments, phase_results)

            # d. 품질 게이트 체크 (REVIEW 단계)
            if phase == TeamPhase.REVIEW:
                gate = self._check_quality_gate(phase, phase_results)
                gate_results.append(gate)
                plan.gate_results.append(gate)

        # 3. 결과 통합
        total_time = time.time() - start_time
        aggregated = await self._inner_orchestrator.aggregate_results(
            main_task=main_task,
            results=all_results,
            aggregation_strategy="structured",
        )

        stats = self._compute_stats(all_results, total_time)

        return {
            "output": aggregated.final_output,
            "stats": stats,
            "gate_results": gate_results,
            "plan": plan,
            "results": all_results,
        }

    async def _create_plan(
        self,
        main_task: str,
        context: dict | None,
    ) -> TeamExecutionPlan:
        """PM으로 실행 계획 생성"""
        if self._pm:
            return await self._pm.create_team_plan(main_task, context)

        # PM이 없으면 임시 PM 생성
        temp_pm = PMAgent(agent_id="temp_pm", llm=self.llm)
        return await temp_pm.create_team_plan(main_task, context)

    def _ordered_phases(self, plan: TeamExecutionPlan) -> list[TeamPhase]:
        """계획에 포함된 단계를 PHASE_ORDER 순서로 반환"""
        plan_phases = set(plan.phases)
        assignment_phases = {assignment.phase for assignment in plan.assignments}
        plan_phases.update(assignment_phases)
        return [p for p in PHASE_ORDER if p in plan_phases]

    def _to_subagent_tasks(
        self,
        assignments: list[TeamTaskAssignment],
        prior_results: dict[str, str],
        base_context: dict[str, Any] | None = None,
    ) -> list[SubAgentTask]:
        """TeamTaskAssignment를 SubAgentTask로 변환 + 선행 결과 주입

        Args:
            assignments: 변환할 할당 목록
            prior_results: 선행 작업 결과 맵 (task_id -> 결과 문자열)
            base_context: 현재 팀 실행에 공통으로 전달할 컨텍스트

        Returns:
            SubAgentTask 목록
        """
        tasks = []
        for assignment in assignments:
            # 선행 결과 컨텍스트 구성
            team_results = {}
            for ctx_id in assignment.context_from:
                if ctx_id in prior_results:
                    team_results[ctx_id] = prior_results[ctx_id]

            context = {
                key: value
                for key, value in (base_context or {}).items()
                if key != "team_results"
            }
            context["team_results"] = team_results

            if assignment.acceptance_criteria:
                context["acceptance_criteria"] = assignment.acceptance_criteria

            task = SubAgentTask(
                task_id=assignment.task_id,
                description=assignment.description,
                context=context,
                dependencies=[
                    dep for dep in assignment.dependencies
                    if dep not in prior_results
                ],
                agent_type=assignment.role.value,
            )
            tasks.append(task)

        return tasks

    async def _execute_phase_tasks(
        self,
        tasks: list[SubAgentTask],
    ) -> dict[str, SubAgentResult]:
        """SubAgentOrchestrator를 재사용하여 병렬 실행"""
        if not tasks:
            return {}

        plan = self._inner_orchestrator.create_execution_plan(
            tasks, strategy="auto"
        )
        return await self._inner_orchestrator.execute_plan(plan)

    def _record_results_to_bus(
        self,
        assignments: list[TeamTaskAssignment],
        results: dict[str, SubAgentResult],
    ) -> None:
        """결과를 메시지 버스 및 task_results에 기록"""
        for assignment in assignments:
            result = results.get(assignment.task_id)
            if not result:
                continue

            output_str = self._stringify_result_output(result.output)
            if result.is_failure():
                output_str = f"[FAILED] {result.error or 'Unknown error'}"

            # task_results에 저장 (다음 단계 컨텍스트용)
            self._task_results[assignment.task_id] = output_str

            # 메시지 버스에 기록
            message = InterAgentMessage(
                from_role=assignment.role,
                to_role=TeamRole.PM,  # PM에게 보고
                task_id=assignment.task_id,
                content=output_str,
                message_type="result",
            )

            if assignment.task_id not in self._message_bus:
                self._message_bus[assignment.task_id] = []
            self._message_bus[assignment.task_id].append(message)

    @staticmethod
    def _stringify_result_output(output: Any) -> str:
        """Preserve falsy-but-meaningful outputs like 0 or False."""
        return "" if output is None else str(output)

    def _check_quality_gate(
        self,
        phase: TeamPhase,
        results: dict[str, SubAgentResult],
    ) -> PhaseGateResult:
        """품질 게이트 체크

        REVIEW 단계에서 결과를 검사하여 이슈를 식별합니다.
        """
        issues = []
        recommendations = []

        quality_keywords = ["failure", "vulnerability", "critical", "error", "bug"]

        for task_id, result in results.items():
            if result.is_failure():
                issues.append(f"[{task_id}] 작업 실패: {result.error}")
                continue

            output_lower = self._stringify_result_output(result.output).lower()
            found_keywords = [
                kw for kw in quality_keywords if kw in output_lower
            ]
            if found_keywords:
                issues.append(
                    f"[{task_id}] 품질 키워드 감지: {', '.join(found_keywords)}"
                )
                recommendations.append(
                    f"[{task_id}] 관련 이슈를 검토하고 수정을 고려하세요"
                )

        passed = len(issues) == 0

        return PhaseGateResult(
            phase=phase,
            passed=passed,
            issues=issues,
            recommendations=recommendations,
        )

    def _compute_stats(
        self,
        results: dict[str, SubAgentResult],
        total_time: float,
    ) -> dict[str, Any]:
        """실행 통계 계산"""
        total = len(results)
        success = sum(1 for r in results.values() if r.is_success())
        failed = total - success
        rate = (success / total * 100) if total > 0 else 0.0

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": round(rate, 1),
            "total_time": round(total_time, 2),
        }
