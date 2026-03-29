"""RalphLoop: PM 주도 OBSERVE-THINK-ACT-WAIT 멀티 에이전트 루프.

PMAgent.decide()가 매 사이클마다 팀 상황을 분석하고,
ASSIGN/VERIFY/WAIT/DONE/ABORT 중 하나를 결정합니다.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from sepilot.agent.multi.models import AgentRole, TaskAssignment
from sepilot.agent.multi.ralph_models import (
    RalphAction,
    RalphContext,
    RalphDecision,
    RalphResult,
    TeamChange,
)
from sepilot.agent.multi.team import AgentTeam

logger = logging.getLogger(__name__)

MAX_IDLE_CYCLES = 5
MAX_ROUNDS_CAP = 100


def _detect_clis() -> list[str]:
    """시스템에 설치된 CLI 에이전트를 감지합니다."""
    return [
        name
        for name in ("claude", "opencode", "codex", "gemini")
        if shutil.which(name)
    ]


def _resolve_cli_binary(command: str) -> str:
    """Extract the executable name from a CLI command string."""
    try:
        parts = shlex.split(command, posix=os.name != "nt")
        return parts[0] if parts else ""
    except ValueError:
        stripped = command.strip()
        return stripped.split()[0] if stripped else ""


class RalphLoop:
    """PM 주도 멀티 에이전트 오케스트레이션 루프.

    라운드 0: plan_tasks → 에이전트 생성 → execute_round
    라운드 1+: OBSERVE → THINK(decide) → PERSIST → ACT
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_rounds: int = 10,
        run_base_dir: str | Path | None = None,
    ) -> None:
        self._llm = llm
        self.team = AgentTeam(llm=llm)
        self.max_rounds = max(1, min(max_rounds, MAX_ROUNDS_CAP))
        self._run_base_dir = (
            Path(run_base_dir) if run_base_dir else Path(".sepilot/agent_runs")
        )
        self._run_dir: Path | None = None
        self._round_summaries: list[str] = []
        self._current_round = 0
        self._available_clis = _detect_clis()
        self._bg_thread = None
        self._run_error: Exception | None = None
        self._ralph_result: RalphResult | None = None
        self._reset_prepared = False

    @property
    def current_round(self) -> int:
        return self._current_round

    def run(self, task: str, preset: object) -> RalphResult:
        """메인 루프를 실행합니다.

        Parameters
        ----------
        task:
            사용자의 원본 요청.
        preset:
            TeamPreset — plan_tasks에 필요한 roles 정보를 포함.

        Returns
        -------
        RalphResult
            최종 실행 결과.
        """
        self._ensure_run_prepared()

        if self.team._stop_event.is_set():
            result = RalphResult(
                task=task,
                rounds_executed=0,
                final_action=RalphAction.ABORT.value,
                final_results={},
                run_dir=str(self._run_dir) if self._run_dir else "",
                round_summaries=list(self._round_summaries),
            )
            self.team._done_event.set()
            return result

        # 0. preset을 팀에 저장 (모니터에서 프리셋 이름 표시용)
        self.team.preset = preset

        # 1. 실행 가능한 역할 확인
        available_roles = [r for r in preset.roles if r.available]
        if not available_roles:
            raise RuntimeError("사용 가능한 에이전트가 없습니다. CLI 에이전트를 설치하세요.")

        # 2. run_dir 생성
        self._run_dir = self._create_run_dir()

        # 3. Round 0: 초기 계획 + 에이전트 생성 + 실행
        self._current_round = 0
        assignments = self.team.pm.plan_tasks(task, available_roles)

        # 에이전트 생성
        for assignment in assignments:
            role = self._find_role(available_roles, assignment.role_name)
            if role is None:
                continue
            agent_id = f"agent_{role.name}"
            if agent_id not in self.team.agents:
                self.team.add_agent(role)

        # 실행
        results = self.team.execute_round(assignments, task)
        self._save_round_results(0, results)

        # 카운터
        executed_rounds = 1
        idle_cycles = 0
        last_results = results
        final_action = RalphAction.DONE

        if self.team._stop_event.is_set():
            final_action = RalphAction.ABORT

        # 3. 루프: OBSERVE → THINK → PERSIST → ACT
        # _current_round 상한: WAIT 반복으로 인한 무한루프 방지
        max_total_cycles = self.max_rounds * 3
        while executed_rounds < self.max_rounds and not self.team._stop_event.is_set() and self._current_round < max_total_cycles:
            self._current_round += 1

            # OBSERVE: 현재 상태 수집
            active_agents = {
                aid: ma.status for aid, ma in self.team.agents.items()
            }
            git_diff_stat, git_status = self._get_git_info()
            truncated_results = self._truncate_results(last_results)
            context = RalphContext(
                task=task,
                round_num=self._current_round,
                max_rounds=self.max_rounds,
                previous_summaries=list(self._round_summaries),
                current_results=truncated_results,
                active_agents=active_agents,
                available_clis=self._available_clis,
                run_dir=str(self._run_dir),
                git_diff_stat=git_diff_stat,
                git_status=git_status,
            )

            # THINK: PM 결정
            decision = self.team.pm.decide(context)

            # PERSIST: 요약 저장
            self._round_summaries.append(decision.round_summary)
            self._save_summary(self._current_round, decision.round_summary)
            self._save_status(self._current_round, decision)

            if self.team._stop_event.is_set():
                final_action = RalphAction.ABORT
                break

            # ACT: 결정에 따라 행동
            if decision.action == RalphAction.DONE:
                final_action = RalphAction.DONE
                break

            if decision.action == RalphAction.ABORT:
                final_action = RalphAction.ABORT
                self.team.kill_all()
                break

            if decision.action == RalphAction.WAIT:
                idle_cycles += 1
                if idle_cycles >= MAX_IDLE_CYCLES:
                    logger.warning(
                        "RalphLoop: %d idle cycles 도달, 루프 종료", MAX_IDLE_CYCLES
                    )
                    final_action = RalphAction.DONE
                    break
                continue

            # ASSIGN or VERIFY → 실행
            idle_cycles = 0

            # 팀 변경 적용
            if decision.team_changes:
                self._apply_team_changes(decision.team_changes)

            if decision.action == RalphAction.VERIFY:
                verify_task = decision.verify_task or "결과를 검증하세요."
                if decision.verify_mode == "command":
                    last_results = self._run_verify_command(verify_task)
                else:
                    last_results = self._run_verifier(verify_task, task)
            elif decision.action == RalphAction.ASSIGN:
                if decision.assignments:
                    last_results = self.team.execute_round(
                        decision.assignments, task
                    )
                else:
                    last_results = {}

            self._save_round_results(self._current_round, last_results)
            executed_rounds += 1

        if self.team._stop_event.is_set():
            final_action = RalphAction.ABORT

        # 4. 최종 보고서
        result = RalphResult(
            task=task,
            rounds_executed=executed_rounds,
            final_action=final_action.value
            if isinstance(final_action, RalphAction)
            else str(final_action),
            final_results=dict(last_results),
            run_dir=str(self._run_dir),
            round_summaries=list(self._round_summaries),
        )
        self._save_final_report(result)
        self.team._done_event.set()
        return result

    def start(self, task: str, preset: object) -> None:
        """백그라운드 스레드에서 run() 시작."""
        import threading

        if self._bg_thread and self._bg_thread.is_alive():
            raise RuntimeError("이미 백그라운드에서 실행 중입니다.")

        self._reset_for_run()
        self._run_error = None
        self._ralph_result = None
        self.team._done_event.clear()

        def _target() -> None:
            try:
                self._ralph_result = self.run(task, preset)
            except Exception as e:
                self._run_error = e
                self.team._done_event.set()

        self._bg_thread = threading.Thread(target=_target, daemon=True)
        self._bg_thread.start()

    @property
    def is_done(self) -> bool:
        """루프 완료 여부."""
        return self.team._done_event.is_set()

    @property
    def last_summary(self) -> str:
        """마지막 라운드 요약."""
        return self._round_summaries[-1] if self._round_summaries else ""

    def wait(self, timeout: float | None = None) -> RalphResult | None:
        """백그라운드 실행 완료 대기. start() 후 호출."""
        if hasattr(self, "_bg_thread") and self._bg_thread:
            finished = self.team._done_event.wait(timeout=timeout)
            if not finished:
                return getattr(self, "_ralph_result", None)
            if self._bg_thread.is_alive() and not self.team._stop_event.is_set():
                self._bg_thread.join(timeout=0)
        if self._run_error:
            raise self._run_error
        return getattr(self, "_ralph_result", None)

    def abort(self) -> None:
        """루프를 즉시 중단하고 모든 에이전트를 종료합니다."""
        self.team.kill_all()
        self.team._done_event.set()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_for_run(self) -> None:
        """Reset per-run state so a RalphLoop instance can be reused safely."""
        if self.team.agents:
            try:
                self.team.kill_all()
            except Exception as exc:
                logger.warning("RalphLoop: 이전 팀 정리 중 오류: %s", exc)
        self.team = AgentTeam(llm=self._llm)
        self._run_dir = None
        self._round_summaries = []
        self._current_round = 0
        self._available_clis = _detect_clis()
        self._reset_prepared = True

    def _ensure_run_prepared(self) -> None:
        """Prepare state exactly once for each run, preserving pre-run aborts."""
        if not self._reset_prepared:
            self._reset_for_run()
        self._reset_prepared = False

    def _create_run_dir(self) -> Path:
        """타임스탬프 기반 run 디렉토리를 생성합니다."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self._run_base_dir / f"run_{ts}"
        run_dir = base_dir
        suffix = 1
        while run_dir.exists():
            run_dir = self._run_base_dir / f"run_{ts}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _resolve_agent_cli(self) -> str:
        """Prefer a currently running team CLI before falling back to detected CLIs."""
        for managed in self.team.agents.values():
            if managed.role.agent_cmd:
                return managed.role.agent_cmd
        if self._available_clis:
            return self._available_clis[0]
        return ""

    def _apply_team_changes(self, changes: list[TeamChange]) -> None:
        """PM 결정에 따라 팀 구성을 변경합니다."""
        for change in changes:
            if change.type == "add":
                if not change.role:
                    logger.warning("RalphLoop: 역할명이 없는 team_change.add 건너뜀")
                    continue
                cli_name = change.agent_cmd or self._resolve_agent_cli()
                if not cli_name:
                    logger.warning(
                        "RalphLoop: 사용 가능한 CLI가 없어 에이전트 '%s' 추가 건너뜀",
                        change.role,
                    )
                    continue
                # 명시적 CLI가 있으면 설치 여부 확인
                if change.agent_cmd and not shutil.which(_resolve_cli_binary(change.agent_cmd)):
                    logger.warning(
                        "RalphLoop: CLI '%s' 미설치, 에이전트 추가 건너뜀",
                        change.agent_cmd,
                    )
                    continue
                role = AgentRole(
                    name=change.role,
                    agent_cmd=cli_name,
                    system_prompt=change.system_prompt,
                )
                self.team.add_agent(role)
            elif change.type == "remove":
                self.team.remove_agent(change.role)

    _VERIFY_BLOCKED = re.compile(
        r"(rm\s+-rf\s+/|mkfs|dd\s+if=|shutdown|reboot|:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;|chmod\s+-R\s+777\s+/)",
        re.IGNORECASE,
    )

    def _run_verify_command(self, command: str) -> dict[str, str]:
        """셸 명령을 직접 실행하여 검증. subprocess.run 사용."""
        # 위험 명령 블랙리스트 체크
        if self._VERIFY_BLOCKED.search(command):
            logger.warning("RalphLoop: 위험 명령 차단: %s", command)
            return {"verifier": f"[BLOCKED] 위험 명령 차단됨: {command}"}
        try:
            result = subprocess.run(
                command,
                shell=True,  # nosec B602 - blocklist-checked command
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.getcwd(),
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\n\n[STDERR]\n{result.stderr}" if result.stderr else ""
                output += f"\n\n[EXIT CODE] {result.returncode}"
            return {"verifier": output}
        except subprocess.TimeoutExpired:
            return {"verifier": f"[TIMEOUT] 명령 실행 시간 초과 (300초): {command}"}
        except Exception as e:
            return {"verifier": f"[ERROR] 명령 실행 실패: {e}"}

    def _run_verifier(self, verify_task: str, main_task: str) -> dict[str, str]:
        """검증 에이전트를 생성/실행합니다."""
        verifier_id = "agent_verifier"
        if verifier_id not in self.team.agents:
            # 사용 가능한 CLI로 verifier 생성
            cli = self._resolve_agent_cli() or "claude"
            role = AgentRole(
                name="verifier",
                agent_cmd=cli,
                system_prompt="당신은 코드 검증 전문가입니다. 결과를 검토하고 문제를 보고하세요.",
            )
            self.team.add_agent(role)

        assignment = TaskAssignment(
            role_name="verifier",
            task_description=verify_task,
        )
        return self.team.execute_round([assignment], main_task)

    def _round_dir(self, round_num: int) -> Path:
        """라운드별 디렉토리를 생성하고 반환합니다."""
        assert self._run_dir is not None
        d = self._run_dir / f"round_{round_num:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_round_results(self, round_num: int, results: dict[str, str]) -> None:
        """에이전트별 결과를 agent_<role>.md로 저장합니다."""
        rd = self._round_dir(round_num)
        for role_name, output in results.items():
            path = rd / f"agent_{role_name}.md"
            path.write_text(f"# {role_name} — Round {round_num}\n\n{output}\n")

    def _save_summary(self, round_num: int, summary: str) -> None:
        """라운드 요약을 summary.md로 저장합니다."""
        rd = self._round_dir(round_num)
        path = rd / "summary.md"
        path.write_text(f"# Round {round_num} Summary\n\n{summary}\n")

    def _save_status(self, round_num: int, decision: RalphDecision) -> None:
        """PM 결정 상태를 status.md로 저장합니다."""
        rd = self._round_dir(round_num)
        path = rd / "status.md"
        lines = [
            f"# Round {round_num} Status",
            "",
            f"**Action**: {decision.action.value}",
            f"**Reasoning**: {decision.reasoning}",
            "",
        ]
        if decision.team_changes:
            lines.append("## Team Changes")
            for tc in decision.team_changes:
                lines.append(f"- {tc.type}: {tc.role} ({tc.agent_cmd})")
            lines.append("")
        if decision.assignments:
            lines.append("## Assignments")
            for a in decision.assignments:
                lines.append(f"- {a.role_name}: {a.task_description}")
            lines.append("")
        path.write_text("\n".join(lines) + "\n")

    def _save_final_report(self, result: RalphResult) -> None:
        """최종 보고서를 final_report.md로 저장합니다."""
        assert self._run_dir is not None
        path = self._run_dir / "final_report.md"
        lines = [
            "# Final Report",
            "",
            f"**Task**: {result.task}",
            f"**Rounds Executed**: {result.rounds_executed}",
            f"**Final Action**: {result.final_action}",
            f"**Run Dir**: {result.run_dir}",
            "",
            "## Round Summaries",
        ]
        for i, s in enumerate(result.round_summaries):
            lines.append(f"- R{i + 1}: {s}")
        lines.append("")
        lines.append("## Final Results")
        for role, output in result.final_results.items():
            lines.append(f"### {role}")
            lines.append(output)
            lines.append("")
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _find_role(
        roles: list[AgentRole], role_name: str
    ) -> AgentRole | None:
        """역할 목록에서 이름으로 검색합니다."""
        for r in roles:
            if r.name == role_name:
                return r
        return None

    @staticmethod
    def _truncate_results(
        results: dict[str, str], max_chars: int = 2000,
    ) -> dict[str, str]:
        """PM 컨텍스트용으로 결과를 끝부분 max_chars로 truncate."""
        truncated = {}
        for role, output in results.items():
            if len(output) > max_chars:
                truncated[role] = f"(...앞부분 생략)\n{output[-max_chars:]}"
            else:
                truncated[role] = output
        return truncated

    @staticmethod
    def _get_git_info() -> tuple[str, str]:
        """git diff --stat과 git status --short를 반환. git 없으면 빈 문자열."""
        diff_stat = ""
        status = ""
        try:
            r = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True, text=True, timeout=10,
            )
            diff_stat = r.stdout.strip()[:1000]  # 1000자 제한
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True, text=True, timeout=10,
            )
            status = r.stdout.strip()[:1000]
        except Exception:
            pass
        return diff_stat, status
