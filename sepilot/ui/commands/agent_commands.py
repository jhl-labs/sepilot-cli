"""Agent team command handlers for Interactive Mode.

Handles the /agent command and its subcommands for multi-agent orchestration.
"""

from __future__ import annotations

import logging
import re
import shlex
import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ANSI helpers (shared with input_utils._ansi_select)
# ---------------------------------------------------------------------------

_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_LINE = "\033[2K\r"

_STATUS_ICON = {
    "starting": (f"{_YELLOW}...{_RESET}", "starting"),
    "busy": (f"{_CYAN}>>>{_RESET}", "busy"),
    "idle": (f"{_GREEN}---{_RESET}", "idle"),
    "done": (f"{_GREEN} OK{_RESET}", "done"),
    "error": (f"{_RED}ERR{_RESET}", "error"),
}


def _team_is_done(agent_team: Any | None) -> bool:
    """Return True only for a completed AgentTeam, not arbitrary mocks."""
    return getattr(agent_team, "is_done", False) is True


def _thread_is_alive(thread: Any | None) -> bool:
    """Best-effort thread liveness check for real objects and test doubles."""
    if thread is None:
        return False
    is_alive = getattr(thread, "is_alive", None)
    if not callable(is_alive):
        return False
    try:
        return bool(is_alive())
    except Exception:
        return False


def _team_is_active(agent_team: Any | None) -> bool:
    """Treat planning/running teams as active even before agents exist."""
    if agent_team is None or _team_is_done(agent_team):
        return False

    if getattr(agent_team, "phase", "") in {"planning", "running", "reviewing", "retrying"}:
        return True

    if _thread_is_alive(getattr(agent_team, "_run_thread", None)):
        return True

    return bool(getattr(agent_team, "agents", {}))


def _ralph_loop_is_active(ralph_loop: Any | None) -> bool:
    """Treat RalphLoop planning state as active before the first agent spawns."""
    if ralph_loop is None or getattr(ralph_loop, "is_done", False) is True:
        return False

    if _thread_is_alive(getattr(ralph_loop, "_bg_thread", None)):
        return True

    if _team_is_active(getattr(ralph_loop, "team", None)):
        return True

    if getattr(ralph_loop, "current_round", 0) > 0:
        return True

    return bool(getattr(ralph_loop, "last_summary", ""))


def _show_planning_status(*, console: Console, ralph_loop: Any | None = None) -> None:
    """Render status for planning/initializing runs before agents exist."""
    if _ralph_loop_is_active(ralph_loop):
        current = getattr(ralph_loop, "current_round", 0)
        total = getattr(ralph_loop, "max_rounds", 0)
        done = getattr(ralph_loop, "is_done", False)
        status_label = "[green]완료[/green]" if done else "[cyan]진행 중[/cyan]"
        console.print(f"  Ralph-Loop: 라운드 {current}/{total} {status_label}")
        last_summary = getattr(ralph_loop, "last_summary", "")
        if last_summary:
            console.print(f"  [dim]마지막 요약: {last_summary[:80]}[/dim]")

    console.print("[cyan]PM이 작업을 분배 중입니다...[/cyan]")


# ---------------------------------------------------------------------------
# Lazy singleton for PresetManager
# ---------------------------------------------------------------------------

_preset_manager_instance: Any | None = None


def _get_preset_manager() -> Any:
    """Return (or create) a module-level PresetManager singleton."""
    global _preset_manager_instance
    if _preset_manager_instance is None:
        from sepilot.agent.multi.preset import PresetManager

        _preset_manager_instance = PresetManager()
    return _preset_manager_instance


# ---------------------------------------------------------------------------
# Status icons
# ---------------------------------------------------------------------------

_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "starting": ("...", "yellow"),
    "busy": (">>>", "cyan"),
    "idle": ("---", "green"),
    "done": ("OK", "green"),
    "error": ("ERR", "red"),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handle_agent_command(
    input_text: str,
    *,
    console: Console,
    agent_team: Any | None = None,
    ralph_loop: Any | None = None,
    session: Any | None = None,
    llm: Any | None = None,
) -> Any | None:
    """Handle /agent command for multi-agent orchestration.

    Returns new AgentTeam if created, else None.
    """
    text = input_text.strip()
    if text.lower().startswith("/agent"):
        text = text[6:].strip()

    # Parse with shlex for proper quoting, fallback to split
    try:
        parts = shlex.split(text) if text else []
    except ValueError:
        parts = text.split() if text else []

    command = parts[0].lower() if parts else ""
    args = parts[1:]

    dispatch: dict[str, Any] = {
        "": _handle_default,
        "help": _show_help,
        "status": _handle_status,
        "run": _handle_run,
        "send": _handle_send,
        "kill": _handle_kill,
        "log": _handle_log,
        "inbox": _handle_inbox,
        "setup": _handle_setup,
        "preset": _handle_preset,
    }

    handler = dispatch.get(command)
    if handler is None:
        console.print(f"[red]알 수 없는 서브커맨드: {command}[/red]")
        console.print("[dim]/agent help 로 사용 가능한 커맨드를 확인하세요.[/dim]")
        return None

    return handler(
        args=args,
        console=console,
        agent_team=agent_team,
        ralph_loop=ralph_loop,
        session=session,
        llm=llm,
    )


# ---------------------------------------------------------------------------
# Default (no subcommand)
# ---------------------------------------------------------------------------


def _handle_default(*, console: Console, agent_team: Any | None = None, **kw: Any) -> None:
    """팀이 있으면 status, 없으면 help."""
    has_agents = agent_team is not None and bool(getattr(agent_team, "agents", {}))
    if has_agents or _team_is_active(agent_team) or _ralph_loop_is_active(kw.get("ralph_loop")):
        return _handle_status(console=console, agent_team=agent_team, **kw)
    return _show_help(console=console, **kw)


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def _show_help(*, console: Console, **_: Any) -> None:
    help_text = """\
[bold cyan]Multi-Agent Team Commands[/bold cyan]

[yellow]상태 확인:[/yellow]
  [cyan]/agent status[/cyan]                 팀 에이전트 상태 조회
  [cyan]/agent log <id>[/cyan]               에이전트 로그 보기
  [cyan]/agent inbox [id][/cyan]             에이전트 수신함 확인

[yellow]실행:[/yellow]
  [cyan]/agent run <task> [--preset X][/cyan]  팀을 구성하여 작업 실행
  [cyan]/agent send <id> "msg"[/cyan]        에이전트에게 메시지 전송

[yellow]관리:[/yellow]
  [cyan]/agent kill <id>[/cyan]              에이전트 종료
  [cyan]/agent kill --all[/cyan]             모든 에이전트 종료

[yellow]설정:[/yellow]
  [cyan]/agent setup[/cyan]                  인터랙티브 프리셋 생성
  [cyan]/agent preset list[/cyan]            프리셋 목록
  [cyan]/agent preset show <name>[/cyan]     프리셋 상세
  [cyan]/agent preset save <name>[/cyan]     프리셋 저장
  [cyan]/agent preset load <name>[/cyan]     프리셋 로드
  [cyan]/agent preset delete <name>[/cyan]   프리셋 삭제"""
    console.print(Panel(help_text, title="/agent", border_style="cyan"))


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _handle_status(
    *, console: Console, agent_team: Any | None = None, ralph_loop: Any | None = None, **_: Any
) -> None:
    has_agents = agent_team is not None and bool(getattr(agent_team, "agents", {}))
    if not has_agents and not _team_is_active(agent_team) and not _ralph_loop_is_active(ralph_loop):
        console.print("[yellow]활성 팀 없음[/yellow]")
        return

    if not has_agents:
        _show_planning_status(console=console, ralph_loop=ralph_loop)
        return

    # Ralph-Loop 라운드 진행 정보
    if _ralph_loop_is_active(ralph_loop):
        current = getattr(ralph_loop, "current_round", 0)
        total = getattr(ralph_loop, "max_rounds", 0)
        done = getattr(ralph_loop, "is_done", False)
        status_label = "[green]완료[/green]" if done else "[cyan]진행 중[/cyan]"
        console.print(f"  Ralph-Loop: 라운드 {current}/{total} {status_label}")
        last_summary = getattr(ralph_loop, "last_summary", "")
        if last_summary:
            console.print(f"  [dim]마지막 요약: {last_summary[:80]}[/dim]")

    statuses = agent_team.status()
    table = Table(title="Agent Team Status")
    table.add_column("ID", style="cyan")
    table.add_column("Role")
    table.add_column("Status")
    table.add_column("Elapsed")
    table.add_column("Inbox")

    for agent_id, info in statuses.items():
        status = info.get("status", "unknown")
        icon, style = _STATUS_STYLE.get(status, ("?", "white"))
        elapsed = info.get("elapsed", 0)
        status_text = f"[{style}]{icon} {status}[/{style}]"
        last_error = info.get("last_error", "")
        if last_error:
            short_err = last_error[:60] + ("..." if len(last_error) > 60 else "")
            status_text += f"\n[dim red]{short_err}[/dim red]"
        table.add_row(
            agent_id,
            info.get("role", ""),
            status_text,
            f"{elapsed:.1f}s",
            str(info.get("inbox_count", 0)),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Live monitor TUI
# ---------------------------------------------------------------------------


def _read_key_nonblocking(timeout: float = 0.5) -> str | None:
    """Read a keypress with timeout. Returns None if no key pressed."""
    import select
    import termios
    import tty

    if not sys.stdin.isatty():
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # ESC 이후 추가 바이트가 있는지 짧은 타임아웃으로 확인
            ready2, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready2:
                return "escape"  # ESC 단독 입력
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ready3, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready3:
                    return None
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return "up"
                if ch3 == "B":
                    return "down"
                return None
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "ctrl-c"
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


_LIVE_OUTPUT_LINES = 8  # 라이브 출력 패널 줄 수


def _get_agent_live_output(team: Any, agent_id: str, max_lines: int = _LIVE_OUTPUT_LINES) -> list[str]:
    """에이전트의 tmux 캡처에서 최근 유효 줄을 반환합니다."""
    ma = team.agents.get(agent_id)
    if ma is None:
        return []
    try:
        session_id = ma.tmux_agent.session_id
        if session_id is None:
            return []
        manager = ma.tmux_agent._get_manager()
        raw = manager.capture_pane(session_id)
        if not raw:
            return []
        # 유효 줄만 필터 (USAGE, 구분선, 빈 프롬프트, 배너 제외)
        lines = raw.strip().split("\n")
        filtered = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("USAGE:") or s.startswith("❯"):
                continue
            if all(c in "─═━╌╍" for c in s):
                continue
            if re.match(r"^[◐◓◑◒]", s):
                continue
            if re.match(r"^\S+@\S+:.*\$", s):
                continue
            if re.match(
                r"^cd\s+\S+.*&&\s*(NO_COLOR=1\s+)?(claude|opencode|codex|gemini)(?:\s+.+)?$",
                s,
            ):
                continue
            if re.match(r"^[▐▝▘]+", s) or re.match(r"^[▐▝▘\s]+[▛█▜▟]+", s):
                continue
            if re.match(r"^\[Pasted text #\d+", s):
                continue
            filtered.append(s)
        return filtered[-max_lines:]
    except Exception:
        return []


def _live_monitor(team: Any, console: Console) -> bool:
    """실시간 에이전트 상태 모니터. Q 누르면 True (백그라운드 전환)."""
    out = sys.stdout
    selected = 0

    _max_lines = 0

    def _render(first: bool = False):
        nonlocal _max_lines, selected
        agents = list(team.agents.items())

        if agents:
            selected = min(selected, len(agents) - 1)

        # 이전 렌더 영역 전체 클리어
        if not first and _max_lines > 0:
            out.write(f"\033[{_max_lines}A")
            for _ in range(_max_lines + 1):
                out.write(f"{_CLEAR_LINE}\n")
            out.write(f"\033[{_max_lines + 1}A")

        preset_name = team.preset.name if team.preset else "team"
        phase = getattr(team, "phase", "")
        cur_lines = 0

        if not agents:
            _spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            _spin_idx = int(time.time() * 8) % len(_spinner)
            out.write(f"{_CLEAR_LINE}{_BOLD} Agent Team — {preset_name}{_RESET}\n")
            out.write(f"{_CLEAR_LINE}\n")
            out.write(f"{_CLEAR_LINE} {_CYAN}{_spinner[_spin_idx]}{_RESET} PM이 작업을 분배하고 있습니다...\n")
            out.write(f"{_CLEAR_LINE}{_DIM} Q 백그라운드  Ctrl+C 취소{_RESET}")
            cur_lines = 3
        else:
            # 헤더
            phase_label = {"planning": "PM 분배 중", "running": "실행 중", "reviewing": "PM 리뷰 중", "retrying": "재시도 중", "done": "완료"}.get(phase, "")
            phase_str = f" · {phase_label}" if phase_label else ""
            out.write(f"{_CLEAR_LINE}{_BOLD} Agent Team — {preset_name} ({len(agents)} agents){phase_str}{_RESET}\n")
            out.write(f"{_CLEAR_LINE}\n")
            cur_lines += 2

            # 에이전트 목록
            for i, (agent_id, ma) in enumerate(agents):
                icon_str, _ = _STATUS_ICON.get(ma.status, ("?", ""))
                elapsed = int(time.time() - ma.started_at)
                mins, secs = divmod(elapsed, 60)
                time_str = f"{mins}m {secs:02d}s"
                err_str = ""
                if ma.status == "error" and ma.last_error:
                    err_str = f" {_RED}{ma.last_error[:40]}{_RESET}"

                # 할당된 작업 요약 표시
                task_hint = ""
                assignments = getattr(team, "assignments", [])
                for a in assignments:
                    if a.role_name == ma.role.name:
                        task_hint = f" {_DIM}{a.task_description[:40]}{_RESET}"
                        break

                out.write(_CLEAR_LINE)
                if i == selected:
                    out.write(f" {_CYAN}{_BOLD}> {agent_id:20s}{_RESET} [{icon_str}] {time_str}{err_str}{task_hint}")
                else:
                    out.write(f" {_DIM}  {agent_id:20s}{_RESET} [{icon_str}] {time_str}{err_str}{task_hint}")
                out.write("\n")
                cur_lines += 1

            # PM 리뷰 결과
            decision = getattr(team, "pm_decision", None)
            if decision and phase == "reviewing":
                reason = getattr(decision, "reason", "")[:60]
                out.write(f"{_CLEAR_LINE} {_CYAN}PM: {decision.action}{_RESET} — {_DIM}{reason}{_RESET}\n")
                cur_lines += 1

            # 선택된 에이전트의 라이브 출력 패널
            if agents:
                sel_agent_id = agents[selected][0]
                live_lines = _get_agent_live_output(team, sel_agent_id)
                out.write(f"{_CLEAR_LINE}{_DIM}{'─' * 60}{_RESET}\n")
                cur_lines += 1
                if live_lines:
                    for ll in live_lines:
                        truncated = ll[:78] if len(ll) > 78 else ll
                        out.write(f"{_CLEAR_LINE}{_DIM} {truncated}{_RESET}\n")
                        cur_lines += 1
                else:
                    out.write(f"{_CLEAR_LINE}{_DIM} (출력 대기 중...){_RESET}\n")
                    cur_lines += 1
                out.write(f"{_CLEAR_LINE}{_DIM}{'─' * 60}{_RESET}\n")
                cur_lines += 1

            out.write(f"{_CLEAR_LINE}{_DIM} ↑↓ 선택  Enter 전체로그  S 메시지  K 종료  Q 백그라운드{_RESET}")
            cur_lines += 1

        _max_lines = max(_max_lines, cur_lines)

        out.flush()

    out.write(_HIDE_CURSOR)
    out.flush()

    try:
        _render(first=True)

        while not team.is_done:
            key = _read_key_nonblocking(timeout=0.5)

            if key is None:
                _render()
                continue

            agents = list(team.agents.keys())
            if not agents:
                _render()
                continue

            if key == "up":
                selected = (selected - 1) % len(agents)
            elif key == "down":
                selected = (selected + 1) % len(agents)
            elif key == "enter":
                # 로그 보기
                aid = agents[selected]
                out.write(f"\n{_SHOW_CURSOR}")
                out.flush()
                log = team.get_log(aid)
                console.print(Panel(log or "(로그 없음)", title=f"Log: {aid}", border_style="cyan"))
                out.write(_HIDE_CURSOR)
                _max_lines = 0  # Panel 출력 후 커서 위치가 변경됨
                _render(first=True)
                continue
            elif key == "s":
                # 메시지 전송
                out.write(f"\n{_SHOW_CURSOR}")
                out.flush()
                from sepilot.ui.input_utils import INPUT_CANCELLED, prompt_text
                aid = agents[selected]
                msg = prompt_text(f"{aid}에게 메시지: ")
                if msg and msg != INPUT_CANCELLED:
                    team.send_to(aid, msg)
                    console.print("[green]전송됨[/green]")
                out.write(_HIDE_CURSOR)
                _max_lines = 0  # 인터랙션 후 커서 위치가 변경됨
                _render(first=True)
                continue
            elif key == "k":
                # 선택한 에이전트 종료
                aid = agents[selected]
                team.kill(aid)
                if selected >= len(agents) - 1:
                    selected = max(0, len(agents) - 2)
            elif key == "q" or key == "escape":
                out.write(f"\n{_SHOW_CURSOR}")
                out.flush()
                return True  # 백그라운드 전환
            elif key == "ctrl-c":
                # Ctrl+C = 팀 종료
                out.write(f"\n{_SHOW_CURSOR}")
                out.flush()
                team.kill_all()
                console.print("[yellow]팀 실행이 중단되었습니다.[/yellow]")
                return False

            _render()

        # 완료됨 — 최종 상태 한번 더 렌더
        _render()
        out.write(f"\n{_SHOW_CURSOR}")
        out.flush()
        return False

    except (EOFError, KeyboardInterrupt):
        out.write(f"\n{_SHOW_CURSOR}")
        out.flush()
        team.kill_all()
        return False
    finally:
        out.write(_SHOW_CURSOR)
        out.flush()


# ---------------------------------------------------------------------------
# Ralph-Loop live monitor TUI
# ---------------------------------------------------------------------------


def _ralph_live_monitor(loop: Any, console: Console) -> bool:
    """Ralph-Loop 실시간 모니터. Q 누르면 True (백그라운드 전환)."""
    out = sys.stdout
    selected = 0

    _max_lines = 0  # 렌더링 영역 최대 줄 수 (줄어들지 않음)

    def _render(first: bool = False):
        nonlocal _max_lines, selected
        agents = list(loop.team.agents.items())

        # selected 범위 보정
        if agents:
            selected = min(selected, len(agents) - 1)

        # 이전 렌더 영역을 완전히 지우고 처음 위치로 복귀
        if not first and _max_lines > 0:
            out.write(f"\033[{_max_lines}A")
            for _ in range(_max_lines + 1):
                out.write(f"{_CLEAR_LINE}\n")
            out.write(f"\033[{_max_lines + 1}A")

        preset_name = loop.team.preset.name if loop.team.preset else "team"
        out.write(f"{_CLEAR_LINE}{_BOLD} Ralph-Loop — {preset_name}{_RESET}\n")
        out.write(f"{_CLEAR_LINE}{_DIM} Round {loop.current_round}/{loop.max_rounds}{_RESET}\n")
        cur_lines = 2  # 헤더 2줄 반영

        if not agents:
            _spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            _spin_idx = int(time.time() * 8) % len(_spinner)
            out.write(f"{_CLEAR_LINE}\n")
            out.write(f"{_CLEAR_LINE} {_CYAN}{_spinner[_spin_idx]}{_RESET} PM이 작업을 분배하고 있습니다...\n")
            out.write(f"{_CLEAR_LINE}{_DIM} Q 백그라운드  Ctrl+C 취소{_RESET}")
            cur_lines += 2  # blank + spinner (hint는 \n 없음)
        else:
            # Last summary
            summary = loop.last_summary
            if summary:
                out.write(f"{_CLEAR_LINE}{_DIM} {summary[:80]}{_RESET}\n")
            else:
                out.write(f"{_CLEAR_LINE}\n")

            for i, (agent_id, ma) in enumerate(agents):
                icon_str, _ = _STATUS_ICON.get(ma.status, ("?", ""))
                elapsed = int(time.time() - ma.started_at)
                mins, secs = divmod(elapsed, 60)
                time_str = f"{mins}m {secs:02d}s"
                err_str = ""
                if ma.status == "error" and ma.last_error:
                    err_str = f" {_RED}{ma.last_error[:40]}{_RESET}"

                out.write(_CLEAR_LINE)
                if i == selected:
                    out.write(f" {_CYAN}{_BOLD}> {agent_id:20s}{_RESET} [{icon_str}] {time_str}{err_str}")
                else:
                    out.write(f" {_DIM}  {agent_id:20s}{_RESET} [{icon_str}] {time_str}{err_str}")
                out.write("\n")
            cur_lines += len(agents) + 1  # summary + agents (hint는 아래에서 +1)

            # 선택된 에이전트의 라이브 출력 패널
            sel_agent_id = agents[selected][0]
            live_lines = _get_agent_live_output(loop.team, sel_agent_id)
            out.write(f"{_CLEAR_LINE}{_DIM}{'─' * 60}{_RESET}\n")
            cur_lines += 1
            if live_lines:
                for ll in live_lines:
                    truncated = ll[:78] if len(ll) > 78 else ll
                    out.write(f"{_CLEAR_LINE}{_DIM} {truncated}{_RESET}\n")
                    cur_lines += 1
            else:
                out.write(f"{_CLEAR_LINE}{_DIM} (출력 대기 중...){_RESET}\n")
                cur_lines += 1
            out.write(f"{_CLEAR_LINE}{_DIM}{'─' * 60}{_RESET}\n")
            cur_lines += 1

            out.write(f"{_CLEAR_LINE}{_DIM} ↑↓ 선택  Enter 로그  S 메시지  K 종료  Q 백그라운드{_RESET}")
            cur_lines += 1

        _max_lines = max(_max_lines, cur_lines)

        out.flush()

    out.write(_HIDE_CURSOR)
    out.flush()

    try:
        _render(first=True)

        while not loop.is_done:
            key = _read_key_nonblocking(timeout=0.5)

            if key is None:
                _render()
                continue

            agents = list(loop.team.agents.keys())
            if not agents:
                _render()
                continue

            if key == "up":
                selected = (selected - 1) % len(agents)
            elif key == "down":
                selected = (selected + 1) % len(agents)
            elif key == "enter":
                aid = agents[selected]
                out.write(f"\n{_SHOW_CURSOR}")
                out.flush()
                log = loop.team.get_log(aid)
                console.print(Panel(log or "(로그 없음)", title=f"Log: {aid}", border_style="cyan"))
                out.write(_HIDE_CURSOR)
                _max_lines = 0
                _render(first=True)
                continue
            elif key == "s":
                out.write(f"\n{_SHOW_CURSOR}")
                out.flush()
                from sepilot.ui.input_utils import INPUT_CANCELLED, prompt_text
                aid = agents[selected]
                msg = prompt_text(f"{aid}에게 메시지: ")
                if msg and msg != INPUT_CANCELLED:
                    loop.team.send_to(aid, msg)
                    console.print("[green]전송됨[/green]")
                out.write(_HIDE_CURSOR)
                _max_lines = 0
                _render(first=True)
                continue
            elif key == "k":
                aid = agents[selected]
                loop.team.kill(aid)
                if selected >= len(agents) - 1:
                    selected = max(0, len(agents) - 2)
            elif key == "q" or key == "escape":
                out.write(f"\n{_SHOW_CURSOR}")
                out.flush()
                return True
            elif key == "ctrl-c":
                out.write(f"\n{_SHOW_CURSOR}")
                out.flush()
                loop.team.kill_all()
                console.print("[yellow]팀 실행이 중단되었습니다.[/yellow]")
                return False

            _render()

        _render()
        out.write(f"\n{_SHOW_CURSOR}")
        out.flush()
        return False

    except (EOFError, KeyboardInterrupt):
        out.write(f"\n{_SHOW_CURSOR}")
        out.flush()
        loop.team.kill_all()
        return False
    finally:
        out.write(_SHOW_CURSOR)
        out.flush()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _handle_run(
    *,
    args: list[str],
    console: Console,
    agent_team: Any | None = None,
    ralph_loop: Any | None = None,
    session: Any | None = None,
    llm: Any | None = None,
    **_: Any,
) -> Any | None:
    if llm is None:
        console.print("[red]LLM이 설정되지 않았습니다. /model 로 먼저 설정하세요.[/red]")
        return None

    if _team_is_active(agent_team) or _ralph_loop_is_active(ralph_loop):
        console.print("[red]이미 팀이 실행 중입니다. /agent kill --all 로 먼저 종료하세요.[/red]")
        return None

    # Parse --preset and --rounds flags
    preset_name: str | None = None
    max_rounds: int | None = None
    task_parts: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--preset" and i + 1 < len(args):
            preset_name = args[i + 1]
            i += 2
        elif args[i] == "--rounds" and i + 1 < len(args):
            try:
                max_rounds = int(args[i + 1])
            except ValueError:
                console.print("[red]--rounds는 숫자여야 합니다.[/red]")
                return None
            if max_rounds < 1:
                console.print("[red]--rounds는 1 이상의 숫자여야 합니다.[/red]")
                return None
            i += 2
        else:
            task_parts.append(args[i])
            i += 1

    task = " ".join(task_parts)
    if not task:
        console.print("[red]작업을 지정해야 합니다.[/red]")
        console.print("[dim]사용법: /agent run <task> [--preset <name>][/dim]")
        return None

    pm = _get_preset_manager()

    # Select preset
    if preset_name is None:
        from sepilot.ui.input_utils import interactive_select

        presets = pm.list_presets()
        if not presets:
            console.print("[red]사용 가능한 프리셋이 없습니다.[/red]")
            return None
        items = [
            {"label": p.name, "description": p.description} for p in presets
        ]
        idx = interactive_select(items, title="프리셋을 선택하세요:")
        if idx is None:
            console.print("[yellow]취소됨[/yellow]")
            return None
        preset_name = presets[idx].name

    preset = pm.load_preset(preset_name)
    if preset is None:
        console.print(f"[red]프리셋 '{preset_name}'을(를) 찾을 수 없습니다.[/red]")
        return None

    import shutil as _shutil

    # tmux 사전 체크 (team.run 전에 빠르게 실패)
    if _shutil.which("tmux") is None:
        console.print("[red]tmux가 설치되어 있지 않습니다.[/red]")
        console.print("[dim]설치: sudo apt install tmux (Ubuntu) / brew install tmux (macOS)[/dim]")
        return None

    if max_rounds is not None:
        from sepilot.agent.multi.ralph_loop import RalphLoop

        loop = RalphLoop(llm=llm, max_rounds=max_rounds)
        console.print(f"[cyan]Ralph-Loop: {preset_name}, max {max_rounds} rounds[/cyan]")

        try:
            loop.start(task, preset)

            # TTY이면 Ralph 전용 실시간 TUI
            if sys.stdin.isatty():
                bg = _ralph_live_monitor(loop, console)
                if bg:
                    console.print("[dim]백그라운드 실행 중. /agent status 로 확인 가능.[/dim]")
                    return loop

            # 완료 대기
            result = loop.wait()
            if loop._run_error:
                raise loop._run_error
            if result:
                console.print(f"\n[green]Ralph-Loop 완료 ({result.rounds_executed} rounds, {result.final_action})[/green]")
                for summary in result.round_summaries:
                    console.print(f"  {summary}")
                console.print(f"\n[dim]결과 디렉토리: {result.run_dir}[/dim]")
            return loop
        except Exception as e:
            console.print(f"[red]Ralph-Loop 오류: {e}[/red]")
            if loop.team.agents:
                console.print("[dim]/agent status 로 상태 확인, /agent kill --all 로 정리[/dim]")
            return loop

    from sepilot.agent.multi.team import AgentTeam

    team = AgentTeam(llm=llm)
    console.print(f"[cyan]팀 '{preset_name}' 으로 작업을 시작합니다...[/cyan]")

    try:
        # 백그라운드 스레드에서 실행 시작
        team.start(task, preset)

        # PM 작업 분배가 완료될 때까지 대기 (최대 30초)
        for _ in range(60):
            if team.assignments or team.phase != "planning":
                break
            time.sleep(0.5)

        # PM 작업 분배 내용 즉시 출력
        if team.assignments:
            console.print("\n[bold cyan]PM 작업 분배:[/bold cyan]")
            for a in team.assignments:
                console.print(f"  [cyan]{a.role_name}[/cyan] → {a.task_description[:80]}")
            console.print()

        # TTY이면 실시간 TUI, 아니면 blocking 대기
        if sys.stdin.isatty():
            bg = _live_monitor(team, console)
            if bg:
                console.print("[dim]백그라운드 실행 중. /agent status 로 확인 가능.[/dim]")
                return team

        # 완료 대기
        results = team.wait()
        if team._run_error:
            raise team._run_error

        # PM 리뷰 판정 출력
        decision = getattr(team, "pm_decision", None)
        if decision:
            action_label = {"done": "완료", "retry": "재시도", "coordinate": "협업", "abort": "중단"}.get(decision.action, decision.action)
            console.print(f"\n[bold cyan]PM 판정:[/bold cyan] {action_label}")
            if decision.reason:
                console.print(f"  [dim]{decision.reason[:120]}[/dim]")

        # 실패/빈 응답 확인
        failures = []
        successes = {}
        for role_name, output in results.items():
            text = output.strip() if isinstance(output, str) else ""
            if not text or text.startswith("[error]") or text.startswith("[timeout]"):
                failures.append((role_name, text or "(응답 없음)"))
            else:
                successes[role_name] = text

        if failures:
            console.print(f"\n[bold red]실패 ({len(failures)}건):[/bold red]")
            for role_name, err in failures:
                console.print(f"  [red]✗ {role_name}[/red]: {err[:80]}")

        # PM 종합 요약 생성
        if successes:
            console.print(f"\n[bold cyan]{'─' * 50}[/bold cyan]")
            console.print("[bold cyan]결과 요약[/bold cyan]")
            console.print(f"[bold cyan]{'─' * 50}[/bold cyan]")
            try:
                summary = team.pm.summarize_results(task, successes)
                console.print(summary)
            except Exception:
                for role_name, output in successes.items():
                    console.print(f"\n[bold]{role_name}:[/bold]")
                    text = output[:3000]
                    if len(output) > 3000:
                        text += f"\n... (잘림, 전체 {len(output)}자)"
                    console.print(text)

        console.print("\n[dim]상세 로그: /agent log <id> · 세션 정리: /agent kill --all[/dim]")
        return team
    except Exception as e:
        console.print(f"[red]팀 실행 중 오류: {e}[/red]")
        if team.agents:
            console.print("[dim]/agent status 로 상태 확인, /agent kill --all 로 정리 가능[/dim]")
        return team  # 에러 시에도 팀 유지 (수동 개입 가능)


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------


def _handle_send(
    *,
    args: list[str],
    console: Console,
    agent_team: Any | None = None,
    **_: Any,
) -> None:
    if agent_team is None or not agent_team.agents:
        console.print("[yellow]활성 팀 없음[/yellow]")
        return

    if len(args) < 2:
        console.print("[yellow]사용법: /agent send <id> \"메시지\"[/yellow]")
        return

    agent_id = args[0]
    message = " ".join(args[1:])
    if agent_id not in agent_team.agents:
        console.print(f"[red]에이전트 '{agent_id}'을(를) 찾을 수 없습니다.[/red]")
        return

    agent_team.send_to(agent_id, message)
    console.print(f"[green]'{agent_id}'에게 메시지를 전송했습니다.[/green]")


# ---------------------------------------------------------------------------
# Kill
# ---------------------------------------------------------------------------


def _handle_kill(
    *,
    args: list[str],
    console: Console,
    agent_team: Any | None = None,
    ralph_loop: Any | None = None,
    **_: Any,
) -> str | None:
    if args and args[0] == "--all":
        killed = False
        if _ralph_loop_is_active(ralph_loop) and hasattr(ralph_loop, "abort"):
            ralph_loop.abort()
            killed = True
        if agent_team is not None and (
            getattr(agent_team, "agents", {}) or _team_is_active(agent_team)
        ):
            agent_team.kill_all()
            killed = True
        if killed:
            console.print("[green]모든 에이전트를 종료했습니다.[/green]")
            return "TEAM_KILLED"
        console.print("[yellow]활성 팀 없음[/yellow]")
        return

    if agent_team is None or not getattr(agent_team, "agents", {}):
        console.print("[yellow]활성 팀 없음[/yellow]")
        return

    if not args:
        console.print("[yellow]사용법: /agent kill <id> 또는 /agent kill --all[/yellow]")
        return

    agent_id = args[0]
    if agent_id not in agent_team.agents:
        console.print(f"[red]에이전트 '{agent_id}'을(를) 찾을 수 없습니다.[/red]")
        return

    agent_team.kill(agent_id)
    console.print(f"[green]에이전트 '{agent_id}'을(를) 종료했습니다.[/green]")


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


def _handle_log(
    *,
    args: list[str],
    console: Console,
    agent_team: Any | None = None,
    **_: Any,
) -> None:
    if agent_team is None or not agent_team.agents:
        console.print("[yellow]활성 팀 없음[/yellow]")
        return

    if not args:
        console.print("[yellow]사용법: /agent log <id>[/yellow]")
        return

    agent_id = args[0]
    if agent_id not in agent_team.agents:
        console.print(f"[red]에이전트 '{agent_id}'을(를) 찾을 수 없습니다.[/red]")
        return

    log_content = agent_team.get_log(agent_id)
    if not log_content:
        console.print(f"[dim]{agent_id}: (로그 없음)[/dim]")
    else:
        console.print(Panel(log_content, title=f"Log: {agent_id}", border_style="cyan"))


# ---------------------------------------------------------------------------
# Inbox
# ---------------------------------------------------------------------------


def _handle_inbox(
    *,
    args: list[str],
    console: Console,
    agent_team: Any | None = None,
    **_: Any,
) -> None:
    if agent_team is None or not agent_team.agents:
        console.print("[yellow]활성 팀 없음[/yellow]")
        return

    if args:
        agent_id = args[0]
        if agent_id not in agent_team.agents:
            console.print(f"[red]에이전트 '{agent_id}'을(를) 찾을 수 없습니다.[/red]")
            return
        msgs = agent_team.inbox.peek(agent_id)
        console.print(f"[cyan]{agent_id}[/cyan] 수신함: {len(msgs)}건")
        for msg in msgs:
            console.print(f"  [{msg.msg_type.value}] {msg.sender}: {msg.content[:80]}")
    else:
        # Show all agents' inbox counts
        for agent_id in agent_team.agents:
            msgs = agent_team.inbox.peek(agent_id)
            console.print(f"  [cyan]{agent_id}[/cyan]: {len(msgs)}건")


# ---------------------------------------------------------------------------
# Setup (interactive preset creation)
# ---------------------------------------------------------------------------


def _handle_setup(
    *,
    console: Console,
    session: Any | None = None,
    **_: Any,
) -> None:
    from sepilot.agent.multi.models import AgentRole, Strategy, TeamPreset
    from sepilot.ui.input_utils import (
        INPUT_CANCELLED,
        interactive_select,
        prompt_confirm,
        prompt_text,
    )

    pm = _get_preset_manager()

    # Choose existing preset or create new
    presets = pm.list_presets()
    items = [{"label": p.name, "description": p.description} for p in presets]
    items.append({"label": "(새로 만들기)", "description": "새 프리셋 생성"})

    idx = interactive_select(items, title="프리셋을 선택하세요:")
    if idx is None:
        console.print("[yellow]취소됨[/yellow]")
        return

    existing_description = ""
    existing_strategy = Strategy.AUTO.value
    if idx < len(presets):
        preset = presets[idx]
        roles = list(preset.roles)
        preset_name = preset.name
        existing_description = preset.description
        try:
            existing_strategy = Strategy(preset.strategy).value
        except ValueError:
            existing_strategy = Strategy.AUTO.value
    else:
        preset_name = prompt_text("프리셋 이름: ", session=session)
        if preset_name == INPUT_CANCELLED or not preset_name:
            console.print("[yellow]취소됨[/yellow]")
            return
        roles = []

    # Role edit loop
    while True:
        action_items = [
            {"label": "역할 추가", "description": "새 역할을 추가합니다"},
            {"label": "역할 제거", "description": "기존 역할을 제거합니다"},
            {"label": "완료", "description": "편집을 완료합니다"},
        ]
        action_idx = interactive_select(action_items, title="작업을 선택하세요:")
        if action_idx is None or action_idx == 2:
            break
        if action_idx == 0:
            # Add role
            name = prompt_text("역할 이름: ", session=session)
            if name == INPUT_CANCELLED or not name:
                continue
            agent_cmd = prompt_text("에이전트 명령 (default: claude): ", session=session, default="claude")
            if agent_cmd == INPUT_CANCELLED:
                continue
            prompt = prompt_text("시스템 프롬프트: ", session=session)
            if prompt == INPUT_CANCELLED:
                continue
            roles.append(AgentRole(name=name, agent_cmd=agent_cmd or "claude", system_prompt=prompt or ""))
            console.print(f"[green]역할 '{name}' 추가됨[/green]")
        elif action_idx == 1:
            # Remove role
            if not roles:
                console.print("[yellow]제거할 역할이 없습니다.[/yellow]")
                continue
            role_items = [{"label": r.name, "description": r.system_prompt[:40]} for r in roles]
            rm_idx = interactive_select(role_items, title="제거할 역할:")
            if rm_idx is not None:
                removed = roles.pop(rm_idx)
                console.print(f"[yellow]역할 '{removed.name}' 제거됨[/yellow]")

    if not roles:
        console.print("[yellow]역할이 없어 프리셋을 저장하지 않습니다.[/yellow]")
        return

    description = prompt_text("프리셋 설명: ", session=session, default=existing_description)
    if description == INPUT_CANCELLED:
        description = existing_description

    # Strategy 선택
    strategy_items = [
        {"label": "auto", "description": "PM이 작업 성격에 따라 판단"},
        {"label": "parallel", "description": "모든 역할 동시 실행"},
        {"label": "sequential", "description": "역할 순서대로 하나씩"},
        {"label": "pipeline", "description": "선행 결과를 후행에 자동 전달"},
    ]
    strategy_idx = interactive_select(strategy_items, title="실행 전략을 선택하세요:")
    strategy = list(Strategy)[strategy_idx].value if strategy_idx is not None else existing_strategy

    new_preset = TeamPreset(name=preset_name, description=description, roles=roles, strategy=strategy)
    console.print(f"\n[bold]프리셋 '{preset_name}'[/bold] - 역할 {len(roles)}개")
    for r in roles:
        console.print(f"  - {r.name} ({r.agent_cmd})")

    save = prompt_confirm("저장하시겠습니까?", session=session)
    if save:
        pm.save_preset(preset_name, new_preset)
        console.print(f"[green]프리셋 '{preset_name}' 저장 완료[/green]")
    else:
        console.print("[yellow]저장하지 않았습니다.[/yellow]")


# ---------------------------------------------------------------------------
# Preset management
# ---------------------------------------------------------------------------


def _handle_preset(
    *,
    args: list[str],
    console: Console,
    agent_team: Any | None = None,
    **_: Any,
) -> None:
    sub = args[0].lower() if args else "list"
    name = args[1] if len(args) > 1 else None

    pm = _get_preset_manager()

    if sub == "list":
        presets = pm.list_presets()
        if not presets:
            console.print("[yellow]프리셋이 없습니다.[/yellow]")
            return
        table = Table(title="Agent Presets")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Roles")
        for p in presets:
            role_names = ", ".join(r.name for r in p.roles)
            table.add_row(p.name, p.description, role_names)
        console.print(table)

    elif sub == "show":
        if not name:
            console.print("[yellow]사용법: /agent preset show <name>[/yellow]")
            return
        preset = pm.load_preset(name)
        if preset is None:
            console.print(f"[red]프리셋 '{name}'을(를) 찾을 수 없습니다.[/red]")
            return
        console.print(f"[bold cyan]{preset.name}[/bold cyan] - {preset.description}")
        console.print(f"  전략: {preset.strategy}")
        for r in preset.roles:
            avail = "[green]OK[/green]" if r.available else "[red]N/A[/red]"
            console.print(f"  - {r.name} ({r.agent_cmd}) {avail}: {r.system_prompt[:60]}")

    elif sub == "save":
        if not name:
            console.print("[yellow]사용법: /agent preset save <name>[/yellow]")
            return
        if agent_team is None or agent_team.preset is None:
            console.print("[yellow]저장할 활성 팀이 없습니다. /agent setup으로 프리셋을 만드세요.[/yellow]")
            return
        from sepilot.agent.multi.models import TeamPreset
        preset_to_save = TeamPreset(
            name=name,
            description=agent_team.preset.description,
            roles=list(agent_team.preset.roles),
            strategy=agent_team.preset.strategy,
        )
        pm.save_preset(name, preset_to_save)
        console.print(f"[green]프리셋 '{name}' 저장 완료[/green]")

    elif sub == "load":
        if not name:
            console.print("[yellow]사용법: /agent preset load <name>[/yellow]")
            return
        preset = pm.load_preset(name)
        if preset is None:
            console.print(f"[red]프리셋 '{name}'을(를) 찾을 수 없습니다.[/red]")
            return
        console.print(f"[green]프리셋 '{name}' 로드 완료[/green]")
        for r in preset.roles:
            avail = "[green]OK[/green]" if r.available else "[red]N/A[/red]"
            console.print(f"  - {r.name} ({r.agent_cmd}) {avail}")
        console.print(f"[dim]/agent run <task> --preset {name} 으로 실행 가능[/dim]")

    elif sub == "delete":
        if not name:
            console.print("[yellow]사용법: /agent preset delete <name>[/yellow]")
            return
        pm.delete_preset(name)
        console.print(f"[green]프리셋 '{name}' 삭제 완료[/green]")

    else:
        console.print(f"[red]알 수 없는 preset 서브커맨드: {sub}[/red]")
