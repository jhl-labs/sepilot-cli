"""Agent Teams 팀 에이전트 구현

8개 역할별 팀 에이전트를 정의합니다. 모두 BaseSubAgent를 상속받습니다.
"""

import json
import logging
import uuid

from langchain_core.language_models import BaseChatModel

from .base_subagent import BaseSubAgent
from .models import SubAgentTask
from .team_models import (
    TeamExecutionPlan,
    TeamPhase,
    TeamRole,
    TeamTaskAssignment,
)
from .team_prompts import (
    ARCHITECT_SYSTEM_PROMPT,
    DEBUGGER_SYSTEM_PROMPT,
    DEVELOPER_SYSTEM_PROMPT,
    DEVOPS_SYSTEM_PROMPT,
    PM_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SECURITY_REVIEWER_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class _TeamAgentMixin:
    """팀 에이전트 공통 기능 믹스인"""

    role: TeamRole
    keywords: list[str]
    system_prompt: str

    def can_handle(self, task: SubAgentTask) -> bool:
        """키워드 매칭 또는 agent_type 직접 매칭으로 처리 가능 여부 판단"""
        if task.agent_type == self.agent_type:
            return True

        task_desc_lower = task.description.lower()
        return any(
            keyword.lower() in task_desc_lower
            for keyword in self.keywords
        )

    def _build_prompt_with_context(self, task: SubAgentTask) -> str:
        """선행 작업 결과 컨텍스트를 포함한 프롬프트 구성"""
        prompt_parts = [task.description]

        # team_results에서 선행 작업 결과 주입
        team_results = task.context.get("team_results", {})
        if team_results:
            prompt_parts.append("\n--- 선행 작업 결과 ---")
            for tid, result_content in team_results.items():
                prompt_parts.append(f"\n[{tid}] 결과:\n{result_content}")

        # 일반 컨텍스트 추가
        other_context = {
            k: v for k, v in task.context.items()
            if k != "team_results"
        }
        if other_context:
            prompt_parts.append("\n--- 추가 컨텍스트 ---")
            for key, value in other_context.items():
                prompt_parts.append(f"{key}: {value}")

        return "\n".join(prompt_parts)

    async def _execute_task(self, task: SubAgentTask) -> str:
        """선행 작업 결과 컨텍스트를 주입 후 LLM 호출"""
        if not self.llm:
            return f"작업 수신: {task.description} (LLM 미설정)"

        prompt = self._build_prompt_with_context(task)
        response = await self._call_llm(prompt, self.system_prompt)
        return response


class PMAgent(BaseSubAgent):
    """PM(프로젝트 매니저) 에이전트

    작업을 분해하고 역할별 할당 계획을 생성합니다. 도구 없이 LLM만 사용합니다.
    """

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="pm",
            tools=[],  # PM은 도구 없이 LLM만 사용
            llm=llm,
        )
        self.role = TeamRole.PM
        self.keywords = [
            "계획", "plan", "분해", "decompose",
            "할당", "assign", "관리", "manage",
        ]
        self.system_prompt = PM_SYSTEM_PROMPT

    def can_handle(self, task: SubAgentTask) -> bool:
        if task.agent_type == self.agent_type:
            return True
        task_desc_lower = task.description.lower()
        return any(k.lower() in task_desc_lower for k in self.keywords)

    async def _execute_task(self, task: SubAgentTask) -> str:
        if not self.llm:
            return f"작업 수신: {task.description} (LLM 미설정)"
        response = await self._call_llm(task.description, self.system_prompt)
        return response

    async def create_team_plan(
        self,
        main_task: str,
        context: dict | None = None,
    ) -> TeamExecutionPlan:
        """LLM으로 JSON 형식 팀 실행 계획 생성

        Args:
            main_task: 메인 작업 설명
            context: 추가 컨텍스트

        Returns:
            TeamExecutionPlan 객체
        """
        if not self.llm:
            return self._create_default_plan(main_task)

        prompt = f"다음 작업을 팀으로 수행하기 위한 실행 계획을 생성해주세요:\n\n{main_task}"
        if context:
            prompt += "\n\n추가 컨텍스트:\n"
            for k, v in context.items():
                prompt += f"- {k}: {v}\n"

        try:
            response = await self._call_llm(prompt, self.system_prompt)
            data = self._extract_json(response)
            if data:
                return self._parse_plan(main_task, data)
        except Exception as e:
            logger.warning(f"PM plan creation failed, using default: {e}")

        return self._create_default_plan(main_task)

    def _create_default_plan(self, main_task: str) -> TeamExecutionPlan:
        """LLM 없는 기본 폴백 계획 (research -> implement -> test)"""
        plan_id = str(uuid.uuid4())[:8]
        assignments = [
            TeamTaskAssignment(
                task_id="R1",
                role=TeamRole.RESEARCHER,
                description=f"조사: {main_task}",
                phase=TeamPhase.RESEARCH,
                dependencies=[],
                context_from=[],
                acceptance_criteria="관련 코드 및 패턴 파악 완료",
            ),
            TeamTaskAssignment(
                task_id="D1",
                role=TeamRole.DEVELOPER,
                description=f"구현: {main_task}",
                phase=TeamPhase.IMPLEMENT,
                dependencies=["R1"],
                context_from=["R1"],
                acceptance_criteria="코드 구현 완료",
            ),
            TeamTaskAssignment(
                task_id="T1",
                role=TeamRole.TESTER,
                description=f"테스트: {main_task}",
                phase=TeamPhase.TEST,
                dependencies=["D1"],
                context_from=["D1"],
                acceptance_criteria="테스트 통과",
            ),
        ]
        return TeamExecutionPlan(
            plan_id=plan_id,
            original_task=main_task,
            assignments=assignments,
            phases=[TeamPhase.RESEARCH, TeamPhase.IMPLEMENT, TeamPhase.TEST],
        )

    def _extract_json(self, text: str) -> dict | None:
        """LLM 응답에서 JSON 추출"""
        # JSON 블록 찾기 (```json ... ``` 또는 { ... })
        import re

        # ```json 블록 시도
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 직접 JSON 객체 시도
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _parse_plan(self, main_task: str, data: dict) -> TeamExecutionPlan:
        """JSON 데이터를 TeamExecutionPlan으로 변환"""
        plan_id = str(uuid.uuid4())[:8]
        assignments = []
        phases_set: list[TeamPhase] = []

        role_map = {r.value: r for r in TeamRole}
        phase_map = {p.value: p for p in TeamPhase}

        raw_assignments = data.get("assignments", [])
        if not isinstance(raw_assignments, list):
            raise ValueError("assignments must be a list")

        for item in raw_assignments:
            if not isinstance(item, dict):
                continue

            role_str = item.get("role", "developer")
            phase_str = item.get("phase", "implement")

            role = role_map.get(role_str, TeamRole.DEVELOPER)
            phase = phase_map.get(phase_str, TeamPhase.IMPLEMENT)
            dependencies = item.get("dependencies", [])
            if not isinstance(dependencies, list):
                dependencies = []

            context_from = item.get("context_from", [])
            if not isinstance(context_from, list):
                context_from = []

            assignment = TeamTaskAssignment(
                task_id=item.get("task_id") or f"T{len(assignments)+1}",
                role=role,
                description=item.get("description") or "",
                phase=phase,
                dependencies=dependencies,
                context_from=context_from,
                acceptance_criteria=item.get("acceptance_criteria") or "",
            )
            assignments.append(assignment)

            if phase not in phases_set:
                phases_set.append(phase)

        return TeamExecutionPlan(
            plan_id=plan_id,
            original_task=main_task,
            assignments=assignments,
            phases=phases_set,
        )


class DeveloperAgent(_TeamAgentMixin, BaseSubAgent):
    """Developer 에이전트 - 코드 구현 전문"""

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="developer",
            tools=tools or [],
            llm=llm,
        )
        self.role = TeamRole.DEVELOPER
        self.keywords = [
            "구현", "implement", "코드", "code", "작성", "write",
            "생성", "create", "수정", "modify", "개발", "develop",
        ]
        self.system_prompt = DEVELOPER_SYSTEM_PROMPT


class TesterAgent(_TeamAgentMixin, BaseSubAgent):
    """Tester 에이전트 - 테스트 작성/실행 전문"""

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="tester",
            tools=tools or [],
            llm=llm,
        )
        self.role = TeamRole.TESTER
        self.keywords = [
            "테스트", "test", "검증", "verify", "실행", "run",
            "pytest", "unittest", "coverage",
        ]
        self.system_prompt = TESTER_SYSTEM_PROMPT


class DebuggerAgent(_TeamAgentMixin, BaseSubAgent):
    """Debugger 에이전트 - 버그 분석/수정 전문"""

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="debugger",
            tools=tools or [],
            llm=llm,
        )
        self.role = TeamRole.DEBUGGER
        self.keywords = [
            "디버그", "debug", "버그", "bug", "오류", "error",
            "수정", "fix", "분석", "analyze", "근본 원인", "root cause",
        ]
        self.system_prompt = DEBUGGER_SYSTEM_PROMPT


class ResearcherAgent(_TeamAgentMixin, BaseSubAgent):
    """Researcher 에이전트 - 코드베이스 탐색/조사 전문"""

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="researcher",
            tools=tools or [],
            llm=llm,
        )
        self.role = TeamRole.RESEARCHER
        self.keywords = [
            "조사", "research", "탐색", "explore", "검색", "search",
            "찾기", "find", "패턴", "pattern", "문서", "document",
        ]
        self.system_prompt = RESEARCHER_SYSTEM_PROMPT


class ArchitectAgent(_TeamAgentMixin, BaseSubAgent):
    """Architect 에이전트 - 설계 리뷰/패턴 추천 전문"""

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="architect",
            tools=tools or [],
            llm=llm,
        )
        self.role = TeamRole.ARCHITECT
        self.keywords = [
            "설계", "design", "아키텍처", "architecture", "구조", "structure",
            "패턴", "pattern", "리뷰", "review",
        ]
        self.system_prompt = ARCHITECT_SYSTEM_PROMPT


class SecurityReviewerAgent(_TeamAgentMixin, BaseSubAgent):
    """SecurityReviewer 에이전트 - 보안 감사 전문"""

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="security_reviewer",
            tools=tools or [],
            llm=llm,
        )
        self.role = TeamRole.SECURITY_REVIEWER
        self.keywords = [
            "보안", "security", "취약점", "vulnerability", "감사", "audit",
            "owasp", "injection", "xss",
        ]
        self.system_prompt = SECURITY_REVIEWER_SYSTEM_PROMPT


class DevOpsAgent(_TeamAgentMixin, BaseSubAgent):
    """DevOps 에이전트 - CI/CD, 인프라 분석 전문"""

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="devops",
            tools=tools or [],
            llm=llm,
        )
        self.role = TeamRole.DEVOPS
        self.keywords = [
            "devops", "ci/cd", "docker", "배포", "deploy",
            "인프라", "infrastructure", "파이프라인", "pipeline",
        ]
        self.system_prompt = DEVOPS_SYSTEM_PROMPT
