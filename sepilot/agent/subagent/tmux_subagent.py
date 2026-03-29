"""tmux 세션을 통해 외부 CLI 에이전트를 실행하는 SubAgent.

TmuxSubAgent는 BaseSubAgent를 상속하여 tmux 세션을 통해
claude, opencode 등의 CLI 에이전트를 대화형으로 실행합니다.
TeamOrchestrator와 통합하여 팀 모드에서 사용할 수 있습니다.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from sepilot.agent.subagent.base_subagent import BaseSubAgent
from sepilot.agent.subagent.models import SubAgentTask

logger = logging.getLogger(__name__)


class TmuxSubAgent(BaseSubAgent):
    """tmux 세션을 통해 외부 CLI 에이전트를 실행하는 SubAgent.

    기존 LLM 기반 SubAgent와 달리, 실제 CLI 에이전트(claude, opencode 등)를
    tmux 세션에서 대화형으로 실행하여 에이전트 자체 도구를 활용합니다.

    Args:
        agent_id: SubAgent 고유 ID
        agent_name: CLI 에이전트 이름 (claude, opencode, codex, gemini)
        role: 팀 역할 (선택, TeamOrchestrator 연동용)
        timeout: 응답 대기 최대 시간 (초)
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "claude",
        role: str | None = None,
        timeout: int = 600,
        **kwargs: Any,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=f"tmux:{agent_name}",
            tools=[],
            llm=None,
        )
        self.agent_name = agent_name
        self.role = role
        self.timeout = timeout
        self._session_id: str | None = None
        self._manager = None  # lazy init

    def _get_manager(self):
        """TmuxSessionManager를 lazy 초기화합니다."""
        if self._manager is None:
            from sepilot.tools.tmux import TmuxSessionManager
            self._manager = TmuxSessionManager()
        return self._manager

    def can_handle(self, task: SubAgentTask) -> bool:
        """tmux 에이전트 타입 매칭 또는 명시적 지정 확인."""
        if task.agent_type == self.agent_type:
            return True
        if task.agent_type and task.agent_type.startswith("tmux:"):
            return task.agent_type == f"tmux:{self.agent_name}"
        return False

    async def _execute_task(self, task: SubAgentTask) -> str:
        """tmux 세션을 생성하고 작업을 실행합니다.

        blocking I/O(tmux subprocess 호출, polling sleep)를 asyncio.to_thread로
        감싸서 이벤트 루프를 블록하지 않습니다.

        Args:
            task: 실행할 작업

        Returns:
            에이전트 응답 텍스트
        """
        return await asyncio.to_thread(self._execute_task_sync, task)

    def _execute_task_sync(self, task: SubAgentTask) -> str:
        """동기 버전의 작업 실행 로직.

        이미 활성 세션이 있으면 재사용하고, 없으면 새로 생성합니다.
        """
        manager = self._get_manager()

        # 기존 세션 재사용 시도
        if self._session_id is not None:
            existing = manager.get_session(self._session_id)
            if existing is not None:
                failed_session_id = self._session_id
                try:
                    logger.info(
                        f"[{self.agent_id}] 기존 세션 재사용: {self._session_id}"
                    )
                    prompt = self._build_prompt(task)
                    response = manager.send_and_wait(
                        session_id=self._session_id,
                        text=prompt,
                        timeout=self.timeout,
                    )
                    logger.info(
                        f"[{self.agent_id}] 응답 수신: {len(response)}자"
                    )
                    return response
                except (RuntimeError, ValueError) as e:
                    logger.warning(
                        f"[{self.agent_id}] 세션 재사용 실패, 새 세션 생성: {e}"
                    )
                    try:
                        manager.destroy_session(failed_session_id)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"[{self.agent_id}] 실패한 기존 세션 정리 중 오류: {cleanup_error}"
                        )
                    self._session_id = None

        # 새 세션 생성
        cwd = task.context.get("worktree_path") or task.context.get("cwd") or os.getcwd()
        worktree_id = task.context.get("worktree_id")
        self._session_id = manager.create_session(
            agent_name=self.agent_name,
            cwd=cwd,
            session_name=f"sepilot_{self.agent_id}",
            worktree_id=worktree_id,
        )

        logger.info(
            f"[{self.agent_id}] tmux 세션 생성: {self._session_id} "
            f"(agent={self.agent_name}, cwd={cwd})"
        )

        # 역할 기반 프롬프트 구성
        prompt = self._build_prompt(task)

        # 프롬프트 전송 + 응답 대기
        try:
            response = manager.send_and_wait(
                session_id=self._session_id,
                text=prompt,
                timeout=self.timeout,
            )
        except Exception:
            session_id = self._session_id
            if session_id is not None:
                try:
                    manager.destroy_session(session_id)
                except Exception as cleanup_error:
                    logger.warning(
                        f"[{self.agent_id}] 실패한 세션 정리 중 오류: {cleanup_error}"
                    )
                finally:
                    if self._session_id == session_id:
                        self._session_id = None
            raise

        logger.info(
            f"[{self.agent_id}] 응답 수신: {len(response)}자"
        )

        return response

    def _build_prompt(self, task: SubAgentTask) -> str:
        """역할 기반 프롬프트를 구성합니다."""
        parts = []

        if self.role:
            parts.append(f"당신은 팀의 {self.role} 역할입니다.")

        parts.append(task.description)

        # 프로젝트 컨텍스트
        if task.context.get("project_files"):
            parts.append(f"프로젝트 주요 파일: {task.context['project_files']}")

        # 기타 컨텍스트
        context_keys = {"previous_results", "main_task", "team_context"}
        for key in context_keys:
            if key in task.context:
                parts.append(f"\n{key}: {task.context[key]}")

        return "\n\n".join(parts)

    def cleanup(self) -> None:
        """tmux 세션을 정리합니다."""
        if self._session_id:
            try:
                manager = self._get_manager()
                manager.destroy_session(self._session_id)
                logger.info(f"[{self.agent_id}] tmux 세션 정리: {self._session_id}")
            except Exception as e:
                logger.warning(f"[{self.agent_id}] tmux 세션 정리 실패: {e}")
            finally:
                self._session_id = None

    @property
    def session_id(self) -> str | None:
        """현재 tmux 세션 ID."""
        return self._session_id

    def __repr__(self) -> str:
        return (
            f"<TmuxSubAgent id={self.agent_id} agent={self.agent_name} "
            f"role={self.role} session={self._session_id} status={self.status}>"
        )
