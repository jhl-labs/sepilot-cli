"""PM (Project Manager) agent for multi-agent orchestration.

Responsible for:
- Decomposing a main task into per-role sub-tasks via LLM.
- Reviewing aggregated results and deciding next action.
- Routing messages between agents via the shared Inbox.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from sepilot.agent.multi.inbox import Inbox
from sepilot.agent.multi.models import (
    AgentRole,
    Message,
    PMAction,
    PMDecision,
    TaskAssignment,
)
from sepilot.agent.multi.ralph_models import (
    RalphAction,
    RalphContext,
    RalphDecision,
    TeamChange,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM_PROMPT = (
    "당신은 멀티 에이전트 팀의 PM입니다. "
    "작업을 역할별로 분해하여 JSON으로 반환하세요.\n"
    "반드시 아래 형식의 JSON만 반환하세요:\n"
    '{"assignments": [{"role_name": "...", "task_description": "...", '
    '"depends_on": [...], "priority": N}]}'
)

_RALPH_PM_SYSTEM_PROMPT = """당신은 멀티 에이전트 팀의 PM입니다. 매 사이클마다 전체 상황을 분석하고 다음 행동을 JSON으로 결정하세요.

사용 가능한 action:
- assign: 에이전트에게 작업 배정
- verify: 검증 실행 (verify_mode: "command"=셸 명령 직접 실행, "agent"=CLI 에이전트에게 위임)
- wait: 활성 에이전트 완료 대기
- done: 작업 완료 선언
- abort: 작업 중단

team_changes로 에이전트를 추가/제거할 수 있습니다.
round_summary에 이번 라운드 결과를 1-2줄로 요약하세요. 이 요약이 다음 라운드에서 당신의 기억이 됩니다.

[파일 변경 사항]과 [워킹 디렉토리 상태]를 참고하여 실제 코드 변경을 파악하세요.
같은 파일을 여러 에이전트가 동시에 수정하지 않도록 작업을 분배하세요.

반드시 아래 형식의 JSON만 반환하세요:
{"reasoning":"판단 근거","round_summary":"라운드 요약","action":"assign|verify|wait|done|abort","team_changes":[{"type":"add|remove","role":"역할명","agent_cmd":"CLI명","system_prompt":"프롬프트"}],"assignments":[{"role_name":"역할명","task_description":"작업"}],"verify_task":"검증 명령","verify_mode":"command|agent"}"""

_REVIEW_SYSTEM_PROMPT = (
    "각 에이전트의 작업 결과를 검토하고 JSON으로 판단을 반환하세요.\n"
    "action: done|retry|coordinate|abort\n\n"
    "판정 기준:\n"
    "- done: 결과가 사용자 요청에 부합하면 done을 반환하세요. "
    "완벽하지 않더라도 핵심 요구사항이 충족되면 done입니다.\n"
    "- retry: 결과에 명백한 오류가 있거나, 핵심 요구사항이 누락된 경우에만 사용하세요.\n"
    "- coordinate: 에이전트 간 결과가 상충하여 조율이 필요한 경우에만 사용하세요.\n"
    "- abort: 작업 자체가 불가능한 경우에만 사용하세요.\n\n"
    "반드시 아래 형식의 JSON만 반환하세요:\n"
    '{"action": "...", "reason": "...", "retry_targets": [...], '
    '"retry_instructions": {...}, "coordinate_pairs": [...]}'
)


class PMAgent:
    """LLM-backed PM that plans tasks and reviews results."""

    def __init__(self, llm: BaseChatModel, inbox: Inbox) -> None:
        self.llm = llm
        self.inbox = inbox

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_tasks(
        self, main_task: str, roles: list[AgentRole]
    ) -> list[TaskAssignment]:
        """Decompose *main_task* into per-role sub-tasks via LLM.

        On any LLM or parsing error, falls back to assigning *main_task*
        to every role with default priority.
        """
        role_names = [r.name for r in roles]
        valid_role_names = set(role_names)
        user_content = (
            f"메인 작업: {main_task}\n"
            f"사용 가능한 역할: {', '.join(role_names)}"
        )

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_PLAN_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
            )
            data = self._parse_json(response.content)
            assignments: list[TaskAssignment] = []
            raw_assignments = data.get("assignments", [])
            if not isinstance(raw_assignments, list):
                logger.warning(
                    "PMAgent.plan_tasks: malformed assignments payload %r, falling back to direct assignment",
                    raw_assignments,
                )
                raw_assignments = []
            for item in raw_assignments:
                if not isinstance(item, dict):
                    logger.warning(
                        "PMAgent.plan_tasks: skipping malformed assignment item: %r",
                        item,
                    )
                    continue
                role_name = item.get("role_name")
                task_description = item.get("task_description")
                if not isinstance(role_name, str) or not role_name:
                    logger.warning(
                        "PMAgent.plan_tasks: skipping assignment without valid role_name: %r",
                        item,
                    )
                    continue
                if role_name not in valid_role_names:
                    logger.warning(
                        "PMAgent.plan_tasks: skipping unknown/unavailable role from LLM plan: %s",
                        role_name,
                    )
                    continue
                if not isinstance(task_description, str) or not task_description:
                    logger.warning(
                        "PMAgent.plan_tasks: skipping assignment without valid task_description: %r",
                        item,
                    )
                    continue
                depends_on = item.get("depends_on", [])
                if not isinstance(depends_on, list):
                    logger.warning(
                        "PMAgent.plan_tasks: malformed depends_on %r for role %s, falling back to []",
                        depends_on,
                        role_name,
                    )
                    depends_on = []
                else:
                    depends_on = [
                        dependency
                        for dependency in depends_on
                        if isinstance(dependency, str) and dependency
                    ]
                priority = item.get("priority", 1)
                if not isinstance(priority, int):
                    logger.warning(
                        "PMAgent.plan_tasks: malformed priority %r for role %s, falling back to 1",
                        priority,
                        role_name,
                    )
                    priority = 1
                assignments.append(
                    TaskAssignment(
                        role_name=role_name,
                        task_description=task_description,
                        depends_on=depends_on,
                        priority=priority,
                    )
                )
            if assignments:
                return assignments
            logger.warning(
                "PMAgent.plan_tasks: no valid assignments in LLM response, falling back to direct assignment"
            )
        except Exception:
            logger.warning(
                "PMAgent.plan_tasks: LLM/parse error, falling back to direct assignment"
            )
        return [
            TaskAssignment(role_name=r.name, task_description=main_task)
            for r in roles
        ]

    def review_results(self, results: dict[str, str]) -> PMDecision:
        """Review aggregated agent results and decide next action.

        On any LLM or parsing error, falls back to ``PMDecision(action="done")``.
        """
        if not results:
            return PMDecision(
                action=PMAction.DONE.value,
                reason="에이전트 결과 없음",
            )
        results_text = "\n".join(
            f"[{role}] {text}" for role, text in results.items()
        )
        user_content = f"에이전트 결과:\n{results_text}"

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_REVIEW_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
            )
            data = self._parse_json(response.content)
            coordinate_pairs = [
                (pair[0], pair[1])
                for pair in data.get("coordinate_pairs", [])
                if isinstance(pair, (list, tuple)) and len(pair) >= 2
            ]
            raw_retry_targets = data.get("retry_targets", [])
            if not isinstance(raw_retry_targets, list):
                logger.warning(
                    "PMAgent.review_results: malformed retry_targets %r, falling back to []",
                    raw_retry_targets,
                )
                retry_targets: list[str] = []
            else:
                retry_targets = [
                    target for target in raw_retry_targets if isinstance(target, str) and target
                ]

            raw_retry_instructions = data.get("retry_instructions", {})
            if not isinstance(raw_retry_instructions, dict):
                logger.warning(
                    "PMAgent.review_results: malformed retry_instructions %r, falling back to {}",
                    raw_retry_instructions,
                )
                retry_instructions: dict[str, str] = {}
            else:
                retry_instructions = {
                    key: value
                    for key, value in raw_retry_instructions.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
            # action 값 검증 (LLM이 잘못된 값을 반환할 수 있음)
            raw_action = data.get("action", "done")
            try:
                action = PMAction(raw_action).value
            except ValueError:
                logger.warning(
                    "PMAgent.review_results: unknown action '%s', falling back to done",
                    raw_action,
                )
                action = PMAction.DONE.value

            return PMDecision(
                action=action,
                reason=data.get("reason", ""),
                retry_targets=retry_targets,
                retry_instructions=retry_instructions,
                coordinate_pairs=coordinate_pairs,
            )
        except Exception:
            logger.warning(
                "PMAgent.review_results: LLM/parse error, falling back to done"
            )
            return PMDecision(
                action=PMAction.DONE.value,
                reason="LLM 응답 파싱 실패로 자동 완료 처리",
            )

    def decide(self, context: RalphContext) -> RalphDecision:
        """Analyze the current round context and decide the next action.

        Uses the Ralph PM system prompt to produce a structured decision
        including action, team changes, and task assignments.

        On any LLM or parsing error, falls back to ``RalphAction.DONE``.
        """
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_RALPH_PM_SYSTEM_PROMPT),
                    HumanMessage(content=context.to_prompt()),
                ]
            )
            data = self._parse_json(response.content)
            raw_action = data.get("action", "done")
            try:
                action = RalphAction(raw_action)
            except ValueError:
                logger.warning(
                    "PMAgent.decide: unknown action '%s', falling back to done",
                    raw_action,
                )
                action = RalphAction.DONE

            team_changes: list[TeamChange] = []
            for item in data.get("team_changes", []):
                if not isinstance(item, dict):
                    logger.warning(
                        "PMAgent.decide: skipping malformed team_change item: %r",
                        item,
                    )
                    continue
                role = item.get("role")
                if not role:
                    logger.warning(
                        "PMAgent.decide: skipping team_change without role: %r",
                        item,
                    )
                    continue
                team_changes.append(
                    TeamChange(
                        type=item.get("type", "add"),
                        role=role,
                        agent_cmd=item.get("agent_cmd", ""),
                        system_prompt=item.get("system_prompt", ""),
                    )
                )

            assignments: list[TaskAssignment] = []
            for item in data.get("assignments", []):
                if not isinstance(item, dict):
                    logger.warning(
                        "PMAgent.decide: skipping malformed assignment item: %r",
                        item,
                    )
                    continue
                role_name = item.get("role_name")
                task_description = item.get("task_description")
                if not role_name or not task_description:
                    logger.warning(
                        "PMAgent.decide: skipping incomplete assignment item: %r",
                        item,
                    )
                    continue
                assignments.append(
                    TaskAssignment(
                        role_name=role_name,
                        task_description=task_description,
                    )
                )

            verify_mode = data.get("verify_mode", "agent")
            if verify_mode not in {"command", "agent"}:
                logger.warning(
                    "PMAgent.decide: unknown verify_mode '%s', falling back to agent",
                    verify_mode,
                )
                verify_mode = "agent"

            return RalphDecision(
                reasoning=data.get("reasoning", ""),
                round_summary=data.get("round_summary", ""),
                action=action,
                team_changes=team_changes,
                assignments=assignments,
                verify_task=data.get("verify_task"),
                verify_mode=verify_mode,
            )
        except Exception:
            logger.warning("PMAgent.decide failed, falling back to done")
            return RalphDecision(
                reasoning="LLM failure",
                round_summary="LLM 실패.",
                action=RalphAction.DONE,
            )

    def summarize_results(self, task: str, results: dict[str, str]) -> str:
        """에이전트 결과를 종합하여 사용자에게 전달할 최종 요약을 생성합니다."""
        results_text = "\n\n".join(
            f"[{role}]\n{text}" for role, text in results.items()
            if text.strip() and not text.strip().startswith("[error]") and not text.strip().startswith("[timeout]")
        )
        if not results_text:
            return "에이전트 팀이 유효한 결과를 생성하지 못했습니다."

        user_content = (
            f"사용자 요청: {task}\n\n"
            f"에이전트 결과:\n{results_text}\n\n"
            "위 결과를 종합하여 사용자에게 전달할 최종 답변을 작성하세요. "
            "중복을 제거하고 핵심 내용만 간결하게 정리하세요."
        )

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content="당신은 멀티 에이전트 팀의 PM입니다. 에이전트 결과를 종합하여 사용자에게 전달할 최종 답변을 작성하세요."),
                    HumanMessage(content=user_content),
                ]
            )
            return response.content
        except Exception:
            logger.warning("PMAgent.summarize_results: LLM 오류, 원본 결과 반환")
            return results_text

    def route_message(self, message: Message) -> None:
        """Forward a message to its receiver via the shared inbox."""
        self.inbox.send(
            sender=message.sender,
            receiver=message.receiver,
            content=message.content,
            msg_type=message.msg_type,
            reply_to=message.reply_to,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON, stripping markdown code fences if present.

        Handles cases where LLM wraps JSON in code fences with surrounding text.
        """
        stripped = text.strip()
        decoder = json.JSONDecoder()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Try extracting from code fence first (handles surrounding text)
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", stripped, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Fallback: extract the first JSON object from surrounding prose.
        start = stripped.find("{")
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError:
                start = stripped.find("{", start + 1)
                continue
            if isinstance(parsed, dict):
                return parsed
            start = stripped.find("{", start + 1)

        raise json.JSONDecodeError("No JSON object found", stripped, 0)
