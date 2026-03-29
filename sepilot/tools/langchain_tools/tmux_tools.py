"""tmux 에이전트 세션 관리 LangChain 도구.

sepilot의 메인 LLM이 tmux 세션을 통해 외부 CLI 에이전트를
대화형으로 제어하고 오케스트레이션할 수 있게 합니다.

도구 목록:
- tmux_create_session: 에이전트 tmux 세션 생성
- tmux_send: 프롬프트/명령 전송 + 응답 대기
- tmux_read: 현재 출력 읽기
- tmux_status: 모든 세션 상태 조회
- tmux_destroy: 세션 종료
- tmux_orchestrate: 다중 에이전트 팀 오케스트레이션
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from types import SimpleNamespace
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_VALID_TMUX_ORCHESTRATE_STRATEGIES = {"parallel", "sequential"}


def _get_manager():
    """TmuxSessionManager 싱글톤을 가져옵니다."""
    from sepilot.tools.tmux import TmuxSessionManager
    return TmuxSessionManager()


def _format_tmux_orchestrate_task(
    main_task: str,
    task_desc: str,
    prior_context: str = "",
) -> str:
    """Build per-agent instructions without duplicating AgentTeam's wrapper."""
    parts = [task_desc]
    if prior_context:
        parts.extend([
            "",
            "이전 팀원들의 작업 결과:",
            prior_context.rstrip(),
        ])
    return "\n".join(parts)


def _build_tmux_orchestrate_assignments(
    main_task: str,
    agent_defs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build deterministic assignments that preserve per-agent task overrides."""
    assignments: list[dict[str, str]] = []
    for agent_def in agent_defs:
        role_name = str(agent_def.get("role", "worker"))
        task_desc = str(agent_def.get("task", main_task))
        assignments.append(
            {
                "role_name": role_name,
                "task_description": _format_tmux_orchestrate_task(main_task, task_desc),
            }
        )
    return assignments


class _TmuxOrchestrateFallbackLLM:
    """Fallback planner/reviewer that preserves tmux_orchestrate inputs."""

    def __init__(self, main_task: str, agent_defs: list[dict[str, Any]]):
        self._main_task = main_task
        self._agent_defs = agent_defs

    def invoke(self, messages):
        system_text = "\n".join(
            str(getattr(message, "content", ""))
            for message in messages
        )
        if "assignments" in system_text:
            payload = {
                "assignments": _build_tmux_orchestrate_assignments(
                    self._main_task,
                    self._agent_defs,
                )
            }
        else:
            payload = {
                "action": "done",
                "reason": "tmux_orchestrate fallback completed",
            }
        return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))


def _run_tmux_orchestrate_sequential(
    main_task: str,
    agent_defs: list[dict[str, Any]],
) -> dict[str, str]:
    """Preserve the legacy sequential behavior with prior-result handoff."""
    from sepilot.agent.multi.models import AgentRole, TeamPreset
    from sepilot.agent.multi.team import AgentTeam

    results: dict[str, str] = {}
    context_so_far = ""

    for index, agent_def in enumerate(agent_defs):
        role_name = str(agent_def.get("role", "worker"))
        agent_name = str(agent_def.get("agent", "claude"))
        workdir = agent_def.get("cwd")
        task_desc = _format_tmux_orchestrate_task(
            main_task,
            str(agent_def.get("task", main_task)),
            context_so_far,
        )

        preset = TeamPreset(
            name=f"tmux_orchestrate_step_{index}",
            description="tmux_orchestrate sequential step",
            strategy="sequential",
            roles=[
                AgentRole(
                    name=role_name,
                    agent_cmd=agent_name,
                    system_prompt="",
                    workdir=workdir,
                )
            ],
        )
        llm = _TmuxOrchestrateFallbackLLM(
            main_task,
            [{"role": role_name, "agent": agent_name, "task": task_desc}],
        )

        team: Any | None = None
        try:
            team = AgentTeam(llm=llm)
            step_results = team.run(main_task, preset)
        finally:
            if team is not None:
                try:
                    team.kill_all()
                except Exception:
                    pass

        for completed_role, response in step_results.items():
            results[completed_role] = response
            context_so_far += f"\n[{completed_role}]\n{response}\n"

    return results


@tool
def tmux_create_session(
    agent_name: str,
    cwd: str | None = None,
    session_name: str | None = None,
) -> str:
    """Create an interactive tmux session running a CLI agent.

    Starts the specified agent (claude, opencode, codex, gemini, etc.)
    in a detached tmux session. Use tmux_send to send prompts and
    tmux_read to read output.

    Args:
        agent_name: Agent to launch (e.g. "claude", "opencode", "codex", "gemini")
        cwd: Working directory for the agent (default: current directory)
        session_name: Custom tmux session name (default: auto-generated)

    Returns:
        Session ID and usage instructions
    """
    # tmux 설치 확인
    if not shutil.which("tmux"):
        return (
            "Error: tmux가 설치되어 있지 않습니다.\n"
            "설치: sudo apt install tmux (Ubuntu) / brew install tmux (macOS)"
        )

    # 에이전트 바이너리 확인
    if not shutil.which(agent_name):
        return f"Error: 에이전트 '{agent_name}'이(가) PATH에 없습니다."

    try:
        manager = _get_manager()
        session_id = manager.create_session(
            agent_name=agent_name,
            cwd=cwd,
            session_name=session_name,
        )

        return "\n".join([
            f"tmux 세션 생성 완료: {session_id}",
            f"  에이전트: {agent_name}",
            f"  작업 디렉토리: {cwd or os.getcwd()}",
            "",
            "사용 가능한 도구:",
            f"  • tmux_send(session_id='{session_id}', text='프롬프트') — 명령 전송",
            f"  • tmux_read(session_id='{session_id}') — 출력 읽기",
            f"  • tmux_destroy(session_id='{session_id}') — 세션 종료",
        ])

    except Exception as e:
        return f"Error: tmux 세션 생성 실패: {e}"


@tool
def tmux_send(
    session_id: str,
    text: str,
    wait_for_response: bool = True,
    timeout: int = 300,
) -> str:
    """Send a prompt or command to a tmux agent session.

    Sends text to the specified tmux session. If wait_for_response is True,
    waits until the agent finishes processing and returns the response.

    Args:
        session_id: ID of the tmux session (from tmux_create_session)
        text: Text to send (prompt, command, etc.)
        wait_for_response: Wait for agent response (default: True)
        timeout: Max wait time in seconds (default: 300)

    Returns:
        Agent response text (if waiting) or confirmation message
    """
    try:
        manager = _get_manager()

        if wait_for_response:
            response = manager.send_and_wait(
                session_id=session_id,
                text=text,
                timeout=timeout,
            )

            # 응답 길이 제한 (LLM 컨텍스트 보호)
            max_len = 8000
            if len(response) > max_len:
                truncated = response[:max_len]
                return (
                    f"{truncated}\n\n"
                    f"... (출력이 {len(response)}자로 잘림, "
                    f"tmux_read로 전체 출력 확인 가능)"
                )
            return response if response else "(빈 응답 — 에이전트가 아직 처리 중일 수 있음)"
        else:
            manager.send_keys(session_id, text)
            return f"전송 완료: '{text[:80]}{'...' if len(text) > 80 else ''}'"

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: tmux 전송 실패: {e}"


@tool
def tmux_read(
    session_id: str,
    lines: int = 100,
    new_only: bool = True,
) -> str:
    """Read current output from a tmux agent session.

    Captures the current pane content of the tmux session. Use new_only=True
    to get only output since the last send_keys.

    Args:
        session_id: ID of the tmux session
        lines: Number of lines to capture (default: 100)
        new_only: Only return new output since last read (default: True)

    Returns:
        Captured output text
    """
    try:
        manager = _get_manager()

        if new_only:
            output = manager.get_new_output(session_id)
        else:
            output = manager.capture_pane(session_id, lines=lines)

        if not output.strip():
            return "(새 출력 없음)"

        # 길이 제한
        max_len = 8000
        if len(output) > max_len:
            return (
                f"{output[:max_len]}\n\n"
                f"... (출력이 {len(output)}자로 잘림)"
            )
        return output

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: tmux 읽기 실패: {e}"


@tool
def tmux_status() -> str:
    """List all active tmux agent sessions and their status.

    Returns:
        Status summary of all managed tmux sessions
    """
    try:
        manager = _get_manager()
        sessions = manager.list_sessions()

        if not sessions:
            return "활성 tmux 에이전트 세션 없음"

        lines = ["활성 tmux 에이전트 세션:\n"]
        for sid, info in sessions.items():
            status_icon = {
                "idle": "🟢",
                "busy": "🔄",
                "starting": "⏳",
                "completed": "✅",
                "error": "❌",
            }.get(info["status"], "❓")

            lines.append(f"{status_icon} {sid}:")
            lines.append(f"   에이전트: {info['agent_name']}")
            lines.append(f"   tmux 세션: {info['tmux_session']}")
            lines.append(f"   디렉토리: {info['cwd']}")
            lines.append(f"   상태: {info['status']}")
            lines.append(f"   실행 시간: {info['runtime']}")
            if info.get("worktree_id"):
                lines.append(f"   워크트리: {info['worktree_id']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: tmux 상태 조회 실패: {e}"


@tool
def tmux_destroy(session_id: str) -> str:
    """Destroy a tmux agent session.

    Gracefully terminates the agent and kills the tmux session.

    Args:
        session_id: ID of the tmux session to destroy

    Returns:
        Confirmation message
    """
    try:
        manager = _get_manager()

        if manager.destroy_session(session_id):
            return f"tmux 세션 종료 완료: {session_id}"
        else:
            return f"Error: 세션을 찾을 수 없습니다: {session_id}"

    except Exception as e:
        return f"Error: tmux 세션 종료 실패: {e}"


@tool
def tmux_orchestrate(
    main_task: str,
    agents: str,
    strategy: str = "parallel",
) -> str:
    """Orchestrate multiple tmux agent sessions as a team.

    Creates multiple agent sessions with different roles, sends each
    a specialized prompt, waits for all responses, and aggregates results.

    Args:
        main_task: The main task description to be distributed
        agents: JSON array of agent definitions. Each object has:
            - role: Role name (e.g. "developer", "tester", "reviewer")
            - agent: Agent name (e.g. "claude", "opencode")
            - task: Specific task for this agent
            - cwd: Working directory (optional)
        strategy: Execution strategy - "parallel" or "sequential" (default: "parallel")

    Returns:
        Aggregated results from all agents

    Example agents JSON:
        [
            {"role": "developer", "agent": "claude", "task": "Implement the login API"},
            {"role": "tester", "agent": "claude", "task": "Write tests for login API"},
            {"role": "reviewer", "agent": "claude", "task": "Review login API for security"}
        ]
    """
    from sepilot.agent.multi.models import AgentRole, TeamPreset
    from sepilot.agent.multi.team import AgentTeam

    # JSON 파싱
    try:
        agent_defs: list[dict[str, Any]] = json.loads(agents)
    except json.JSONDecodeError as e:
        return f"Error: agents JSON 파싱 실패: {e}"

    if not isinstance(agent_defs, list):
        return "Error: agents는 JSON 배열이어야 합니다."
    if not agent_defs:
        return "Error: agents 목록이 비어 있습니다."
    if any(not isinstance(agent_def, dict) for agent_def in agent_defs):
        return "Error: agents 배열의 각 항목은 JSON 객체여야 합니다."

    strategy = str(strategy).strip().lower()
    if strategy not in _VALID_TMUX_ORCHESTRATE_STRATEGIES:
        return "Error: strategy는 'parallel' 또는 'sequential' 이어야 합니다."

    # JSON → AgentRole + TeamPreset 변환
    roles = []
    for agent_def in agent_defs:
        role_name = agent_def.get("role", "worker")
        agent_name = agent_def.get("agent", "claude")
        workdir = agent_def.get("cwd")
        role = AgentRole(
            name=role_name,
            agent_cmd=agent_name,
            system_prompt="",
            workdir=workdir,
        )
        roles.append(role)

    try:
        if strategy == "parallel":
            preset = TeamPreset(
                name="tmux_orchestrate",
                description="tmux_orchestrate tool에서 생성",
                strategy=strategy,
                roles=roles,
            )
            llm = _TmuxOrchestrateFallbackLLM(main_task, agent_defs)
            team = AgentTeam(llm=llm)
            results = team.run(main_task, preset)
        else:
            team = None
            results = _run_tmux_orchestrate_sequential(main_task, agent_defs)
    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: 오케스트레이션 실패: {e}"
    finally:
        if strategy == "parallel":
            try:
                team.kill_all()
            except Exception:
                pass

    # 결과 포맷팅 (기존 출력 형식 유지)
    output_lines = [
        "=== tmux 오케스트레이션 결과 ===",
        f"전체 작업: {main_task}",
        f"전략: {strategy}",
        f"에이전트 수: {len(roles)}",
        "",
    ]

    for role_name, response in results.items():
        truncated = response[:3000] if len(response) > 3000 else response
        output_lines.append(f"--- [{role_name}] ---")
        output_lines.append(truncated)
        if len(response) > 3000:
            output_lines.append(f"... (잘림, 원본 {len(response)}자)")
        output_lines.append("")

    return "\n".join(output_lines)
