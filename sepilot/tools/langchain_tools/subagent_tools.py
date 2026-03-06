"""SubAgent execution tools for LangChain agent."""

import logging
import os

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Module-level settings propagated from the main agent
_current_settings = None


def set_current_settings(settings) -> None:
    """Set the current agent settings for SubAgent LLM creation.

    Called by base_agent.update_llm() and _initialize_llm() to ensure
    SubAgent tools use the same LLM configuration as the main agent.
    """
    global _current_settings
    _current_settings = settings


def _create_subagent_llm():
    """Create LLM for SubAgent using LLM Provider Factory.

    Uses propagated settings from the main agent when available,
    falling back to default Settings() if not set.

    Returns:
        LLM instance or None if creation fails
    """
    try:
        from sepilot.config.llm_providers import LLMProviderFactory, LLMProviderError
        from sepilot.config.settings import Settings

        settings = _current_settings or Settings()
        # Override temperature to 0.0 for SubAgent (deterministic)
        settings = settings.model_copy(update={"temperature": 0.0})

        factory = LLMProviderFactory(settings)
        return factory.create_llm()

    except LLMProviderError as e:
        logger.warning(f"LLM provider error: {e}")
        if e.suggestion:
            logger.info(f"Tip: {e.suggestion}")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize SubAgent LLM: {e}")
        return None


@tool
def subagent_execute(
    main_task: str,
    decomposition_strategy: str = "auto",
    max_parallel: int = 3,
    aggregation_strategy: str = "concatenate",
    team_mode: bool = False,
) -> str:
    """Execute a complex task using SubAgent system.

    SubAgent 시스템을 사용하여 복잡한 작업을 여러 하위 작업으로 분해하고 병렬 실행합니다.

    Args:
        main_task: 실행할 메인 작업 설명
        decomposition_strategy: 작업 분해 전략
            - "auto": 자동으로 의존성 분석하여 결정 (기본)
            - "parallel": 모든 작업을 병렬 실행
            - "sequential": 모든 작업을 순차 실행
        max_parallel: 최대 병렬 실행 수 (기본: 3)
        aggregation_strategy: 결과 통합 전략
            - "concatenate": 결과를 단순 연결 (기본)
            - "summarize": LLM을 사용하여 요약
            - "structured": 구조화된 보고서 형식
        team_mode: PM 주도 팀 모드 활성화 (기본: False)
            - True: PM이 작업을 분해하고 8개 역할별 에이전트가 단계적으로 실행
            - False: 기존 SubAgent 시스템 사용 (하위 호환)

    Returns:
        통합된 최종 결과

    Examples:
        - "프로젝트의 모든 Python 파일을 분석하고 복잡도 보고서를 작성해줘"
        - "src 디렉토리의 각 모듈에 대해 테스트를 생성해줘"
        - "모든 Python 파일에서 print 문을 logging으로 변경해줘"
        - team_mode=True: "인증 기능 추가 + 테스트 + 보안 검토" (PM이 역할 배분)

    Use Cases:
        - 프로젝트 전체 분석 (여러 파일 병렬 분석)
        - Multi-file 리팩토링 (여러 파일 동시 수정)
        - 테스트 생성 및 실행 (모듈별 병렬 테스트 생성)
        - 문서 생성 (여러 모듈 문서 병렬 생성)
        - 팀 모드: 복합 작업 (구현 + 테스트 + 보안 검토 등)
    """
    import asyncio

    if team_mode:
        return _run_async(_async_team_execute(main_task, max_parallel))

    async def _async_execute():
        """Internal async implementation."""
        from sepilot.agent.subagent import SubAgentOrchestrator
        from sepilot.agent.subagent.specialized_agents import (
            AnalyzerSubAgent,
            CodeGenSubAgent,
            DocumentationSubAgent,
            RefactoringSubAgent,
            TestingSubAgent,
        )

        logger.info(f"SubAgent executing: {main_task}")

        # Get LLM using shared initialization logic
        llm = _create_subagent_llm()
        if llm is None:
            logger.warning("Failed to initialize LLM, SubAgent will run without LLM")

        # Create Orchestrator
        orchestrator = SubAgentOrchestrator(
            llm=llm,
            max_parallel=max_parallel
        )

        # Get all tools
        from sepilot.tools.langchain_tools import get_all_tools
        all_tools = get_all_tools()

        # Tool filtering function
        def filter_tools(tool_names: list[str]):
            return [t for t in all_tools if hasattr(t, 'name') and t.name in tool_names]

        # Register specialized SubAgents
        orchestrator.register_subagent(AnalyzerSubAgent(
            agent_id="analyzer_1",
            tools=filter_tools(["code_analyze", "file_read", "codebase"]),
            llm=llm
        ))

        orchestrator.register_subagent(CodeGenSubAgent(
            agent_id="codegen_1",
            tools=filter_tools(["file_write", "file_edit", "code_analyze", "file_read"]),
            llm=llm
        ))

        orchestrator.register_subagent(TestingSubAgent(
            agent_id="testing_1",
            tools=filter_tools(["bash_execute", "file_write", "file_read"]),
            llm=llm
        ))

        orchestrator.register_subagent(DocumentationSubAgent(
            agent_id="documentation_1",
            tools=filter_tools(["file_write", "code_analyze", "file_read"]),
            llm=llm
        ))

        orchestrator.register_subagent(RefactoringSubAgent(
            agent_id="refactoring_1",
            tools=filter_tools(["code_analyze", "file_edit", "file_read"]),
            llm=llm
        ))

        # Decompose task
        subtasks = await orchestrator.decompose_task(main_task)
        logger.info(f"Decomposed into {len(subtasks)} subtasks")

        # Create execution plan
        execution_plan = orchestrator.create_execution_plan(
            subtasks,
            strategy=decomposition_strategy
        )
        logger.info(f"Created execution plan with {execution_plan.total_phases} phases")

        # Execute
        results = await orchestrator.execute_plan(execution_plan)

        # Aggregate results
        aggregated = await orchestrator.aggregate_results(
            main_task=main_task,
            results=results,
            aggregation_strategy=aggregation_strategy
        )

        # Summary
        return f"""
🎯 SubAgent 실행 완료

📊 실행 통계:
- 전체 작업: {len(results)}개
- 성공: {sum(1 for r in results.values() if r.is_success())}개
- 실패: {sum(1 for r in results.values() if r.is_failure())}개
- 실행 시간: {aggregated.total_execution_time:.2f}초
- 성공률: {aggregated.success_rate:.1f}%

📝 결과:
{aggregated.final_output}
"""

    return _run_async(_async_execute())


def _run_async(coro) -> str:
    """비동기 코루틴을 동기적으로 실행하는 헬퍼"""
    import asyncio

    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            import nest_asyncio
            try:
                nest_asyncio.apply()
                return asyncio.run(coro)
            except ImportError:
                logger.warning("nest_asyncio not available, trying alternative method")
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(coro)
        raise
    except Exception as e:
        logger.error(f"Async execution failed: {e}")
        return f"❌ 실행 실패: {str(e)}"


async def _async_team_execute(main_task: str, max_parallel: int = 3) -> str:
    """팀 모드 비동기 실행 구현"""
    from sepilot.agent.subagent.team_agents import (
        ArchitectAgent,
        DebuggerAgent,
        DeveloperAgent,
        DevOpsAgent,
        PMAgent,
        ResearcherAgent,
        SecurityReviewerAgent,
        TesterAgent,
    )
    from sepilot.agent.subagent.team_models import TeamRole
    from sepilot.agent.subagent.team_orchestrator import TeamOrchestrator

    logger.info(f"Team mode executing: {main_task}")

    # LLM 생성
    llm = _create_subagent_llm()
    if llm is None:
        logger.warning("Failed to initialize LLM, team will run without LLM")

    # 도구 필터링
    from sepilot.tools.langchain_tools import get_all_tools
    all_tools = get_all_tools()

    def filter_tools(tool_names: list[str]):
        return [t for t in all_tools if hasattr(t, 'name') and t.name in tool_names]

    # TeamOrchestrator 생성
    orchestrator = TeamOrchestrator(llm=llm, max_parallel=max_parallel)

    # 8개 팀 에이전트 등록
    orchestrator.register_agent(
        TeamRole.PM,
        PMAgent(agent_id="pm_1", llm=llm),
    )
    orchestrator.register_agent(
        TeamRole.DEVELOPER,
        DeveloperAgent(
            agent_id="developer_1",
            tools=filter_tools(["file_write", "file_edit", "file_read", "code_analyze", "bash_execute"]),
            llm=llm,
        ),
    )
    orchestrator.register_agent(
        TeamRole.TESTER,
        TesterAgent(
            agent_id="tester_1",
            tools=filter_tools(["file_write", "file_read", "bash_execute"]),
            llm=llm,
        ),
    )
    orchestrator.register_agent(
        TeamRole.DEBUGGER,
        DebuggerAgent(
            agent_id="debugger_1",
            tools=filter_tools(["file_read", "bash_execute", "code_analyze", "search_content"]),
            llm=llm,
        ),
    )
    orchestrator.register_agent(
        TeamRole.RESEARCHER,
        ResearcherAgent(
            agent_id="researcher_1",
            tools=filter_tools(["file_read", "find_file", "search_content", "codebase", "web_search"]),
            llm=llm,
        ),
    )
    orchestrator.register_agent(
        TeamRole.ARCHITECT,
        ArchitectAgent(
            agent_id="architect_1",
            tools=filter_tools(["file_read", "code_analyze", "codebase", "search_content", "get_structure"]),
            llm=llm,
        ),
    )
    orchestrator.register_agent(
        TeamRole.SECURITY_REVIEWER,
        SecurityReviewerAgent(
            agent_id="security_reviewer_1",
            tools=filter_tools(["file_read", "search_content", "code_analyze"]),
            llm=llm,
        ),
    )
    orchestrator.register_agent(
        TeamRole.DEVOPS,
        DevOpsAgent(
            agent_id="devops_1",
            tools=filter_tools(["file_read", "bash_execute", "find_file"]),
            llm=llm,
        ),
    )

    # 팀 작업 실행
    result = await orchestrator.execute_team_task(main_task)

    # 결과 포맷팅
    stats = result["stats"]
    gate_results = result.get("gate_results", [])

    gate_summary = ""
    for gate in gate_results:
        status = "PASSED" if gate.passed else "FAILED"
        gate_summary += f"\n  - {gate.phase.value}: {status}"
        if gate.issues:
            for issue in gate.issues:
                gate_summary += f"\n    - {issue}"

    return f"""
🎯 Team 실행 완료

📊 실행 통계:
- 전체 작업: {stats['total']}개
- 성공: {stats['success']}개
- 실패: {stats['failed']}개
- 실행 시간: {stats['total_time']}초
- 성공률: {stats['success_rate']}%
{f"🔍 품질 게이트:{gate_summary}" if gate_summary else ""}

📝 결과:
{result['output']}
"""


__all__ = ['subagent_execute', 'set_current_settings']
