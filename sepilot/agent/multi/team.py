"""AgentTeam: 멀티 에이전트 팀 라이프사이클 관리.

TmuxSubAgent 인스턴스를 생성하고 PMAgent를 통해 작업을 분배/리뷰하며,
ThreadPoolExecutor로 에이전트를 병렬 실행합니다.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel

from sepilot.agent.multi.inbox import Inbox
from sepilot.agent.multi.models import (
    AgentRole,
    MessageType,
    PMAction,
    PMDecision,
    Strategy,
    TaskAssignment,
    TeamPreset,
)
from sepilot.agent.multi.pm import PMAgent
from sepilot.agent.subagent.models import SubAgentTask
from sepilot.agent.subagent.tmux_subagent import TmuxSubAgent

logger = logging.getLogger(__name__)


@dataclass
class ManagedAgent:
    """팀 내에서 관리되는 에이전트 래퍼."""

    agent_id: str
    role: AgentRole
    tmux_agent: Any  # TmuxSubAgent instance
    status: str = "starting"
    started_at: float = field(default_factory=time.time)
    last_error: str = ""


class AgentTeam:
    """멀티 에이전트 팀 라이프사이클 관리.

    PMAgent가 작업을 분배하고, TmuxSubAgent 인스턴스를 병렬로 실행하며,
    결과를 리뷰하여 필요 시 재시도합니다.
    """

    MAX_RETRIES = 2

    def __init__(self, llm: BaseChatModel, timeout: int = 600) -> None:
        self.inbox = Inbox()
        self.pm = PMAgent(llm=llm, inbox=self.inbox)
        self.agents: dict[str, ManagedAgent] = {}
        self.preset: TeamPreset | None = None
        self._main_task = ""
        self._timeout = timeout
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._results: dict[str, str] = {}
        self._run_thread: threading.Thread | None = None
        self._run_error: Exception | None = None
        # UX: 진행 상황 추적
        self.assignments: list[TaskAssignment] = []
        self.pm_decision: PMDecision | None = None
        self.phase: str = "idle"  # idle, planning, running, reviewing, done

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _reset_runtime_state(self) -> None:
        """Reset per-run mutable state so a completed team can be reused safely."""
        for agent_id, managed in list(self.agents.items()):
            try:
                managed.tmux_agent.cleanup()
            except Exception as exc:
                logger.warning("이전 에이전트 %s 정리 중 오류: %s", agent_id, exc)
        self.agents.clear()
        self.inbox = Inbox()
        self.pm = PMAgent(llm=self.pm.llm, inbox=self.inbox)
        self.assignments = []
        self.pm_decision = None
        self.phase = "idle"

    def run(self, task: str, preset: TeamPreset) -> dict[str, str]:
        """팀을 구성하여 작업을 실행하고 결과를 반환합니다.

        동기 메서드. 내부적으로 ThreadPoolExecutor로 병렬 실행.

        1. tmux 설치 확인
        2. PM이 작업 분배
        3. 역할별 TmuxSubAgent 생성
        4. Inbox를 통해 작업 전달
        5. ThreadPoolExecutor로 병렬 실행
        6. PM이 결과 리뷰 → 필요 시 재시도
        7. {role_name: output} 반환
        """
        # 1. tmux 확인
        if shutil.which("tmux") is None:
            raise RuntimeError(
                "tmux가 설치되어 있지 않습니다. "
                "멀티 에이전트 팀 실행에는 tmux가 필요합니다."
            )

        self._reset_runtime_state()
        self.preset = preset
        self._main_task = task
        self._stop_event.clear()
        self._results.clear()
        self.phase = "planning"
        self.pm_decision = None
        self.inbox.register("pm")

        # 2. PM이 작업 분배 (available=False 역할 제외)
        available_roles = [r for r in preset.roles if r.available]
        if not available_roles:
            raise RuntimeError("사용 가능한 에이전트가 없습니다. CLI 에이전트를 설치하세요.")
        assignments = self.pm.plan_tasks(task, available_roles)
        self.assignments = assignments

        # 3. 역할별 에이전트 생성
        role_map: dict[str, AgentRole] = {r.name: r for r in available_roles}
        for assignment in assignments:
            role = role_map.get(assignment.role_name)
            if role is None:
                continue
            agent_id = f"agent_{role.name}"
            if agent_id in self.agents:
                continue
            tmux_agent = TmuxSubAgent(
                agent_id=agent_id,
                agent_name=role.agent_cmd,
                role=role.name,
                timeout=self._timeout,
            )
            ma = ManagedAgent(
                agent_id=agent_id,
                role=role,
                tmux_agent=tmux_agent,
            )
            self.agents[agent_id] = ma
            self.inbox.register(agent_id)

        # 4-5. 전략에 따라 초기 작업 전달 및 실행
        self.phase = "running"
        self._run_initial_assignments(assignments, preset)

        # 6. PM 리뷰 + 재시도 루프
        self.phase = "reviewing"
        retries = 0
        while retries < self.MAX_RETRIES and not self._stop_event.is_set():
            decision = self.pm.review_results(self._results)
            self.pm_decision = decision

            if decision.action in (PMAction.DONE, PMAction.ABORT):
                break

            if decision.action == PMAction.RETRY:
                if not decision.retry_targets:
                    break
                retries += 1
                self.phase = "retrying"
                self._handle_retry(decision.retry_targets, decision.retry_instructions)
                self.phase = "reviewing"
                continue

            if decision.action == PMAction.COORDINATE:
                if not decision.coordinate_pairs:
                    break
                retries += 1
                if not self._handle_coordinate(decision.coordinate_pairs, decision.reason):
                    break
                continue

            break

        # 7. 결과 반환
        self.phase = "done"
        self._done_event.set()
        return dict(self._results)

    def start(self, task: str, preset: TeamPreset) -> None:
        """팀 실행을 백그라운드 스레드에서 시작합니다. 비동기 시작."""
        if self._run_thread and self._run_thread.is_alive():
            raise RuntimeError("이미 팀이 실행 중입니다.")

        self._done_event.clear()
        self._run_error = None

        def _target():
            try:
                self.run(task, preset)
            except Exception as e:
                self._run_error = e
                self._done_event.set()

        self._run_thread = threading.Thread(target=_target, daemon=True)
        self._run_thread.start()

    @property
    def is_done(self) -> bool:
        """팀 실행이 완료되었는지."""
        return self._done_event.is_set()

    def wait(self, timeout: float | None = None) -> dict[str, str]:
        """팀 실행이 완료될 때까지 대기. start() 후 호출."""
        if self._run_thread:
            finished = self._done_event.wait(timeout=timeout)
            if not finished:
                return dict(self._results)
            if self._run_thread.is_alive() and not self._stop_event.is_set():
                self._run_thread.join()
        if self._run_error:
            raise self._run_error
        return dict(self._results)

    def status(self) -> dict[str, dict]:
        """각 에이전트의 상태 정보를 반환합니다."""
        result: dict[str, dict] = {}
        for agent_id, ma in self.agents.items():
            inbox_count = len(self.inbox.peek(agent_id))
            info: dict[str, Any] = {
                "role": ma.role.name,
                "agent_cmd": ma.role.agent_cmd,
                "status": ma.status,
                "elapsed": time.time() - ma.started_at,
                "inbox_count": inbox_count,
            }
            if ma.last_error:
                info["last_error"] = ma.last_error
            result[agent_id] = info
        return result

    def send_to(self, agent_id: str, message: str) -> None:
        """에이전트에게 FEEDBACK 메시지를 전송합니다."""
        self.inbox.send(
            sender="user",
            receiver=agent_id,
            content=message,
            msg_type=MessageType.FEEDBACK,
        )

    def kill(self, agent_id: str) -> None:
        """에이전트를 종료하고 tmux 세션을 정리합니다."""
        ma = self.agents.get(agent_id)
        if ma is None:
            return
        try:
            ma.tmux_agent.cleanup()
        except Exception as e:
            logger.warning("에이전트 %s 정리 중 오류: %s", agent_id, e)
        ma.status = "done"

    def kill_all(self) -> None:
        """모든 에이전트를 종료합니다."""
        self._stop_event.set()
        self._done_event.set()  # wait() 호출자 즉시 반환
        for agent_id in list(self.agents.keys()):
            self.kill(agent_id)

    def add_agent(self, role: AgentRole) -> str:
        """팀에 에이전트를 동적으로 추가합니다 (Ralph-Loop 지원)."""
        agent_id = f"agent_{role.name}"
        tmux_agent = TmuxSubAgent(
            agent_id=agent_id,
            agent_name=role.agent_cmd,
            role=role.name,
            timeout=self._timeout,
        )
        ma = ManagedAgent(agent_id=agent_id, role=role, tmux_agent=tmux_agent)
        self.agents[agent_id] = ma
        self.inbox.register(agent_id)
        return agent_id

    def remove_agent(self, role_name: str) -> None:
        """팀에서 에이전트를 제거하고 리소스를 정리합니다."""
        agent_id = f"agent_{role_name}"
        ma = self.agents.get(agent_id)
        if ma is None:
            return
        try:
            ma.tmux_agent.cleanup()
        except Exception:
            pass
        self.inbox.receive(agent_id)  # move pending to history
        self.inbox.unregister(agent_id)
        del self.agents[agent_id]

    def execute_round(
        self, assignments: list[TaskAssignment], task: str,
    ) -> dict[str, str]:
        """할당된 에이전트만 실행하는 단일 라운드 (Ralph-Loop 지원)."""
        # 현재 라운드 대상 역할의 이전 결과만 초기화 (다른 역할 결과는 보존)
        for assignment in assignments:
            self._results.pop(assignment.role_name, None)
        target_agents: dict[str, ManagedAgent] = {}
        assignment_map = self._group_assignments_by_agent(assignments)
        for agent_id, assignment_group in assignment_map.items():
            ma = assignment_group["agent"]
            task_descriptions = assignment_group["descriptions"]
            self.inbox.send(
                "pm",
                agent_id,
                self._build_round_prompt(
                    ma.role,
                    task,
                    self._merge_task_descriptions(task_descriptions),
                ),
                MessageType.TASK,
            )
            target_agents[agent_id] = ma
        self._execute_agents(target_agents, task_prefix="round")
        return dict(self._results)

    def get_log(self, agent_id: str) -> str:
        """에이전트의 tmux 로그를 캡처합니다. 세션 종료 시 _results에서 fallback."""
        ma = self.agents.get(agent_id)
        if ma is None:
            return ""
        # tmux 세션에서 직접 캡처 시도
        try:
            session_id = ma.tmux_agent.session_id
            if session_id is not None:
                manager = ma.tmux_agent._get_manager()  # no public API for manager
                pane = manager.capture_pane(session_id)
                if pane:
                    return pane
        except Exception as e:
            logger.debug("에이전트 %s tmux 캡처 실패, results fallback: %s", agent_id, e)
        # tmux 세션이 없거나 캡처 실패 시 저장된 결과 반환
        return self._results.get(ma.role.name, "")

    # ------------------------------------------------------------------
    # Internal: parallel execution
    # ------------------------------------------------------------------

    def _execute_agents(self, targets: dict[str, ManagedAgent], task_prefix: str = "task") -> None:
        """대상 에이전트를 ThreadPoolExecutor로 병렬 실행합니다."""
        if not targets:
            return

        def _run_single(agent_id: str, ma: ManagedAgent) -> tuple[str, str, str]:
            msgs = self.inbox.receive(agent_id)
            if not msgs:
                ma.status = "idle"
                return agent_id, ma.role.name, "(작업 없음)"
            cwd = getattr(ma.role, "workdir", None) or os.getcwd()

            # 프로젝트 컨텍스트 수집
            context: dict[str, Any] = {"cwd": cwd}
            try:
                # 소스 파일 우선 목록 (최대 20개)
                top_files = subprocess.run(
                    ["git", "ls-files", "--cached",
                     "*.py", "*.ts", "*.tsx", "*.js", "*.jsx",
                     "*.go", "*.rs", "*.java", "*.c", "*.cpp", "*.h"],
                    capture_output=True, text=True, cwd=cwd, timeout=5,
                )
                if top_files.returncode == 0 and top_files.stdout.strip():
                    files = [f for f in top_files.stdout.strip().split("\n") if f][:20]
                    if files:
                        context["project_files"] = ", ".join(files)
            except Exception:
                pass

            task = SubAgentTask(
                task_id=f"{task_prefix}_{agent_id}",
                description=msgs[0].content,
                context=context,
            )
            ma.status = "busy"
            result = ma.tmux_agent._execute_task_sync(task)
            ma.status = "done"
            return agent_id, ma.role.name, result

        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            futures = {
                pool.submit(_run_single, aid, ma): aid
                for aid, ma in targets.items()
            }
            try:
                for future in as_completed(futures, timeout=self._timeout):
                    if self._stop_event.is_set():
                        return
                    try:
                        _, role_name, result = future.result()
                        self._results[role_name] = result
                    except Exception as e:
                        aid = futures[future]
                        ma = self.agents[aid]
                        ma.status = "error"
                        ma.last_error = str(e)
                        self._results[ma.role.name] = f"[error] {e}"
                        logger.error("에이전트 %s 실행 오류: %s", aid, e)
            except (TimeoutError, FuturesTimeoutError):
                for future, aid in futures.items():
                    if not future.done():
                        ma = targets[aid]
                        ma.status = "error"
                        ma.last_error = "시간 초과"
                        self._results[ma.role.name] = "[error] 시간 초과"
                        logger.error("에이전트 %s 시간 초과", aid)

    def _poll_agents(self) -> None:
        """모든 에이전트를 병렬 실행합니다."""
        self._execute_agents(self.agents, task_prefix="task")

    def _run_initial_assignments(
        self,
        assignments: list[TaskAssignment],
        preset: TeamPreset,
    ) -> None:
        """Run the first PM assignments using the preset execution strategy."""
        strategy = self._resolve_initial_strategy(preset.strategy, assignments)

        if strategy == Strategy.SEQUENTIAL:
            ordered = self._sort_assignments_for_sequential(assignments, preset)
            for assignment in ordered:
                self._dispatch_assignments([assignment], include_dependencies=False)
            return

        if strategy == Strategy.PIPELINE:
            self._run_pipeline_assignments(assignments)
            return

        self._dispatch_assignments(assignments, include_dependencies=False)

    def _resolve_initial_strategy(
        self,
        preset_strategy: str,
        assignments: list[TaskAssignment],
    ) -> str:
        """Resolve AUTO into a concrete initial execution strategy."""
        try:
            strategy = Strategy(preset_strategy or Strategy.AUTO).value
        except ValueError:
            logger.warning(
                "AgentTeam: unknown preset strategy '%s', falling back to auto",
                preset_strategy,
            )
            strategy = Strategy.AUTO.value
        if strategy == Strategy.AUTO:
            if any(assignment.depends_on for assignment in assignments):
                return Strategy.PIPELINE
            return Strategy.PARALLEL
        return strategy

    def _sort_assignments_for_sequential(
        self,
        assignments: list[TaskAssignment],
        preset: TeamPreset,
    ) -> list[TaskAssignment]:
        """Use preset role order for sequential execution, then priority as tie-break."""
        role_order = {role.name: index for index, role in enumerate(preset.roles)}
        return sorted(
            assignments,
            key=lambda assignment: (
                role_order.get(assignment.role_name, len(role_order)),
                assignment.priority,
            ),
        )

    def _run_pipeline_assignments(self, assignments: list[TaskAssignment]) -> None:
        """Execute assignments in dependency-aware phases and pass prior results forward."""
        remaining = list(assignments)
        known_roles = {assignment.role_name for assignment in assignments}
        completed_roles: set[str] = set()

        while remaining:
            ready = [
                assignment
                for assignment in remaining
                if {
                    dep for dep in assignment.depends_on
                    if dep in known_roles
                }.issubset(completed_roles)
            ]
            if not ready:
                logger.warning(
                    "AgentTeam pipeline strategy: unresolved dependencies, falling back to priority order"
                )
                ready = sorted(remaining, key=lambda assignment: assignment.priority)

            self._dispatch_assignments(ready, include_dependencies=True)
            completed_roles.update(assignment.role_name for assignment in ready)
            completed_ids = {id(assignment) for assignment in ready}
            remaining = [
                assignment for assignment in remaining
                if id(assignment) not in completed_ids
            ]

    def _dispatch_assignments(
        self,
        assignments: list[TaskAssignment],
        *,
        include_dependencies: bool,
    ) -> None:
        """Send assignments to inbox and execute the targeted agents."""
        targets: dict[str, ManagedAgent] = {}
        assignment_map = self._group_assignments_by_agent(
            assignments,
            include_dependencies=include_dependencies,
        )
        for agent_id, assignment_group in assignment_map.items():
            ma = assignment_group["agent"]
            task_descriptions = assignment_group["descriptions"]
            self.inbox.send(
                sender="pm",
                receiver=agent_id,
                content=self._build_agent_prompt(
                    ma.role,
                    self._merge_task_descriptions(task_descriptions),
                ),
                msg_type=MessageType.TASK,
            )
            ma.status = "starting"
            targets[agent_id] = ma

        self._execute_agents(targets, task_prefix="task")

    def _group_assignments_by_agent(
        self,
        assignments: list[TaskAssignment],
        *,
        include_dependencies: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Collect assignment descriptions per agent while preserving order."""
        grouped: dict[str, dict[str, Any]] = {}
        for assignment in assignments:
            agent_id = f"agent_{assignment.role_name}"
            ma = self.agents.get(agent_id)
            if ma is None:
                continue
            task_description = assignment.task_description
            if include_dependencies:
                task_description = self._augment_with_dependency_results(assignment)
            entry = grouped.setdefault(
                agent_id,
                {"agent": ma, "descriptions": []},
            )
            entry["descriptions"].append(task_description)
        return grouped

    @staticmethod
    def _merge_task_descriptions(task_descriptions: list[str]) -> str:
        """Collapse multiple tasks for one role into a single ordered prompt."""
        if not task_descriptions:
            return ""
        if len(task_descriptions) == 1:
            return task_descriptions[0]

        merged = ["여러 작업이 같은 역할에 배정되었습니다. 순서대로 모두 처리하세요."]
        for index, description in enumerate(task_descriptions, start=1):
            merged.append(f"[작업 {index}]\n{description}")
        return "\n\n".join(merged)

    def _augment_with_dependency_results(self, assignment: TaskAssignment) -> str:
        """Append completed dependency outputs to the assignment prompt."""
        dependency_outputs = []
        for role_name in assignment.depends_on:
            if role_name not in self._results:
                continue
            dependency_outputs.append(f"[{role_name}]\n{self._results[role_name]}")

        if not dependency_outputs:
            return assignment.task_description

        return (
            f"{assignment.task_description}\n\n"
            "선행 역할 결과:\n"
            + "\n\n".join(dependency_outputs)
        )

    def _build_agent_prompt(self, role: AgentRole, task_description: str) -> str:
        """역할별 system prompt와 현재 팀 작업 컨텍스트를 결합합니다."""
        parts = []
        if role.system_prompt:
            parts.append(role.system_prompt)
        if self._main_task:
            parts.append(f"전체 목표: {self._main_task}")
        parts.append(f"당신의 작업:\n{task_description}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_round_prompt(role: AgentRole, main_task: str, task_description: str) -> str:
        """Build a prompt for execute_round without mutating team-level main task state."""
        parts = []
        if role.system_prompt:
            parts.append(role.system_prompt)
        parts.append(f"전체 목표: {main_task}")
        parts.append(f"당신의 작업:\n{task_description}")
        return "\n\n".join(parts)

    def _handle_retry(
        self,
        retry_targets: list[str],
        retry_instructions: dict[str, str],
    ) -> None:
        """재시도 대상 에이전트에게 새 작업을 전달하고 재실행합니다."""
        retry_agent_ids = []
        for role_name in retry_targets:
            agent_id = f"agent_{role_name}"
            ma = self.agents.get(agent_id)
            if ma is None:
                continue
            instruction = retry_instructions.get(role_name, "재시도하세요.")
            self.inbox.send(
                sender="pm",
                receiver=agent_id,
                content=self._build_agent_prompt(ma.role, instruction),
                msg_type=MessageType.TASK,
            )
            ma.status = "starting"
            retry_agent_ids.append(agent_id)

        # 재시도 대상만 병렬 실행
        retry_map = {aid: self.agents[aid] for aid in retry_agent_ids if aid in self.agents}
        self._execute_agents(retry_map, task_prefix="retry")

    def _handle_coordinate(
        self,
        coordinate_pairs: list[tuple[str, str]],
        reason: str,
    ) -> bool:
        """협업이 필요한 에이전트 쌍에 상대 결과를 공유하고 재실행합니다."""
        instructions: dict[str, list[str]] = {}
        seen_pairs: set[tuple[str, str]] = set()

        for role_a, role_b in coordinate_pairs:
            if role_a == role_b:
                continue

            pair_key = tuple(sorted((role_a, role_b)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            agent_a = f"agent_{role_a}"
            agent_b = f"agent_{role_b}"
            if agent_a not in self.agents or agent_b not in self.agents:
                continue

            result_a = self._results.get(role_a, "(결과 없음)")
            result_b = self._results.get(role_b, "(결과 없음)")
            instructions.setdefault(role_a, []).append(
                f"{role_b}의 현재 결과를 참고하세요:\n{result_b}"
            )
            instructions.setdefault(role_b, []).append(
                f"{role_a}의 현재 결과를 참고하세요:\n{result_a}"
            )

        if not instructions:
            return False

        coordination_targets: dict[str, ManagedAgent] = {}
        for role_name, notes in instructions.items():
            agent_id = f"agent_{role_name}"
            ma = self.agents.get(agent_id)
            if ma is None:
                continue
            prompt_parts = ["다른 역할과 협업하여 결과를 보완하세요."]
            if reason:
                prompt_parts.append(f"PM 판단: {reason}")
            prompt_parts.extend(notes)
            self.inbox.send(
                sender="pm",
                receiver=agent_id,
                content=self._build_agent_prompt(ma.role, "\n\n".join(prompt_parts)),
                msg_type=MessageType.COORDINATE,
            )
            ma.status = "starting"
            coordination_targets[agent_id] = ma

        self._execute_agents(coordination_targets, task_prefix="coordinate")
        return True
