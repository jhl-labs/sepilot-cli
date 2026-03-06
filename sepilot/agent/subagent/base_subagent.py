"""SubAgent 기본 클래스"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .models import SubAgentResult, SubAgentTask, TaskStatus

logger = logging.getLogger(__name__)


class BaseSubAgent(ABC):
    """SubAgent 기본 클래스

    모든 SubAgent는 이 클래스를 상속받아 구현해야 합니다.

    Attributes:
        agent_id: SubAgent 고유 ID
        agent_type: SubAgent 타입 (예: "analyzer", "codegen")
        tools: 이 SubAgent가 사용할 수 있는 도구 목록
        llm: 사용할 LLM 모델
        status: 현재 상태 ("idle", "busy", "error")
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.tools = tools or []
        self.llm = llm
        self.status = "idle"

    @abstractmethod
    def can_handle(self, task: SubAgentTask) -> bool:
        """이 SubAgent가 주어진 작업을 처리할 수 있는지 판단

        Args:
            task: 처리할 작업

        Returns:
            처리 가능 여부
        """
        pass

    async def execute(self, task: SubAgentTask) -> SubAgentResult:
        """작업 실행

        Args:
            task: 실행할 작업

        Returns:
            실행 결과
        """
        logger.info(f"[{self.agent_id}] Starting task: {task.task_id}")
        self.status = "busy"
        start_time = time.time()

        try:
            # 실제 작업 실행
            output = await self._execute_task(task)

            execution_time = time.time() - start_time

            result = SubAgentResult(
                task_id=task.task_id,
                status=TaskStatus.SUCCESS,
                output=output,
                execution_time=execution_time,
                tokens_used=0,  # TODO: 실제 토큰 사용량 추적
            )

            logger.info(
                f"[{self.agent_id}] Completed task: {task.task_id} "
                f"in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                f"[{self.agent_id}] Failed task: {task.task_id} - {str(e)}"
            )

            result = SubAgentResult(
                task_id=task.task_id,
                status=TaskStatus.FAILURE,
                error=str(e),
                execution_time=execution_time,
            )

            return result

        finally:
            self.status = "idle"

    @abstractmethod
    async def _execute_task(self, task: SubAgentTask) -> Any:
        """실제 작업 실행 로직 (서브클래스에서 구현)

        Args:
            task: 실행할 작업

        Returns:
            작업 결과
        """
        pass

    async def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """LLM 호출

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)

        Returns:
            LLM 응답
        """
        if not self.llm:
            raise ValueError("LLM not configured for this SubAgent")

        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        response = await self.llm.ainvoke(messages)
        return response.content

    def get_tool(self, tool_name: str) -> Any | None:
        """도구 이름으로 도구 가져오기

        Args:
            tool_name: 도구 이름

        Returns:
            도구 객체 또는 None
        """
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == tool_name:
                return tool
        return None

    def has_tool(self, tool_name: str) -> bool:
        """특정 도구를 가지고 있는지 확인

        Args:
            tool_name: 도구 이름

        Returns:
            보유 여부
        """
        return self.get_tool(tool_name) is not None

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"id={self.agent_id} type={self.agent_type} status={self.status}>"
        )


class SimpleSubAgent(BaseSubAgent):
    """간단한 SubAgent 구현

    주어진 프롬프트를 LLM에 전달하고 결과를 반환하는 기본 SubAgent
    """

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None,
        keywords: list[str] | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="simple",
            tools=tools,
            llm=llm
        )
        self.keywords = keywords or []

    def can_handle(self, task: SubAgentTask) -> bool:
        """키워드 기반 작업 처리 가능 여부 판단"""
        if not self.keywords:
            return True  # 키워드가 없으면 모든 작업 처리 가능

        task_desc_lower = task.description.lower()
        return any(keyword.lower() in task_desc_lower for keyword in self.keywords)

    async def _execute_task(self, task: SubAgentTask) -> str:
        """LLM을 사용하여 작업 실행"""
        if not self.llm:
            return f"Task received: {task.description} (No LLM configured)"

        # 시스템 프롬프트 구성
        system_prompt = f"""You are a SubAgent specialized in handling specific tasks.

Your capabilities:
- Agent ID: {self.agent_id}
- Agent Type: {self.agent_type}
- Available Tools: {', '.join([t.name for t in self.tools if hasattr(t, 'name')])}

Execute the given task efficiently and return the result clearly."""

        # 작업 컨텍스트를 포함한 프롬프트 구성
        user_prompt = task.description

        if task.context:
            context_str = "\n".join([
                f"{key}: {value}"
                for key, value in task.context.items()
            ])
            user_prompt = f"{user_prompt}\n\nContext:\n{context_str}"

        # LLM 호출
        response = await self._call_llm(user_prompt, system_prompt)

        return response
