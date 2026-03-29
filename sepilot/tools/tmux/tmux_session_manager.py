"""tmux 세션 매니저 — 싱글톤 패턴으로 tmux 에이전트 세션을 관리합니다.

BackgroundShellManager 패턴을 참조하여 구현합니다.
tmux의 capture-pane을 활용하므로 별도의 출력 수집 스레드가 불필요합니다.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sepilot.tools.tmux.tmux_agent_configs import get_agent_config

logger = logging.getLogger(__name__)

# ANSI 이스케이프 시퀀스 제거 패턴 (포괄적)
# CSI sequences, OSC sequences, 마우스 이벤트, 커서 이동 등
_ANSI_RE = re.compile(
    r"\x1b"           # ESC
    r"(?:"
    r"\[[0-9;?]*[A-Za-z]"    # CSI: ESC [ ... letter (색상, 커서, 스크롤 등)
    r"|\].*?(?:\x07|\x1b\\)" # OSC: ESC ] ... (BEL 또는 ST로 종료)
    r"|\([A-B]"              # 문자셋 지정: ESC ( A/B
    r"|[=>]"                 # 키패드 모드: ESC = / ESC >
    r"|#[0-9]"               # 라인 속성: ESC # N
    r"|[ -/]*[0-~]"          # 기타 2-byte ESC sequences
    r")"
)

# tmux 세션 이름에 사용 불가한 문자 (. : 등)
_TMUX_NAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]")

# 기본 pane 크기
DEFAULT_PANE_WIDTH = 200
DEFAULT_PANE_HEIGHT = 50


def _sanitize_tmux_name(name: str) -> str:
    """tmux 세션 이름에서 허용되지 않는 문자를 제거합니다."""
    return _TMUX_NAME_UNSAFE_RE.sub("_", name)


@dataclass
class TmuxSession:
    """tmux 세션 상태를 추적하는 데이터 클래스."""

    session_id: str
    agent_name: str
    tmux_session_name: str
    cwd: str
    created_at: float
    last_capture: str = ""
    last_capture_at: float = 0.0
    status: str = "starting"  # starting, idle, busy, completed, error
    worktree_id: str | None = None
    _capture_cursor: int = 0  # 마지막으로 반환한 출력 위치


class TmuxSessionManager:
    """tmux 에이전트 세션을 관리하는 싱글톤 매니저.

    BackgroundShellManager 패턴을 따릅니다:
    - 싱글톤 인스턴스
    - threading.Lock으로 동기화
    - 세션 생명주기 관리 (생성, 전송, 캡처, 종료)
    """

    _instance: TmuxSessionManager | None = None
    _lock = threading.Lock()

    # set-buffer/paste-buffer는 전역 버퍼를 사용하므로
    # 병렬 호출 시 레이스 컨디션 방지를 위한 Lock
    _send_lock = threading.Lock()

    # _sessions dict 조작용 Lock (create/destroy 동시성)
    _sessions_lock = threading.Lock()

    def __new__(cls) -> TmuxSessionManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions: dict[str, TmuxSession] = {}
                    atexit.register(cls._instance._atexit_cleanup)
        return cls._instance

    def _atexit_cleanup(self) -> None:
        """프로세스 종료 시 모든 관리 세션을 정리합니다."""
        with self._sessions_lock:
            ids = list(self._sessions.keys())
        for sid in ids:
            try:
                self.destroy_session(sid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # tmux 설치 확인
    # ------------------------------------------------------------------

    @staticmethod
    def is_tmux_available() -> bool:
        """tmux가 설치되어 있는지 확인합니다."""
        return shutil.which("tmux") is not None

    @staticmethod
    def _require_tmux() -> None:
        """tmux 미설치 시 명확한 에러를 발생시킵니다."""
        if not TmuxSessionManager.is_tmux_available():
            raise RuntimeError(
                "tmux가 설치되어 있지 않습니다. "
                "설치: sudo apt install tmux (Ubuntu) / brew install tmux (macOS)"
            )

    # ------------------------------------------------------------------
    # 세션 생명주기
    # ------------------------------------------------------------------

    def create_session(
        self,
        agent_name: str,
        cwd: str | None = None,
        session_name: str | None = None,
        worktree_id: str | None = None,
        env: dict[str, str] | None = None,
        pane_width: int = DEFAULT_PANE_WIDTH,
        pane_height: int = DEFAULT_PANE_HEIGHT,
    ) -> str:
        """tmux 세션을 생성하고 에이전트를 실행합니다.

        Args:
            agent_name: 에이전트 이름 (claude, opencode, codex, gemini)
            cwd: 작업 디렉토리 (기본: 현재 디렉토리)
            session_name: tmux 세션 이름 (기본: 자동 생성)
            worktree_id: WorktreeManager 연동을 위한 worktree ID
            env: 추가 환경 변수
            pane_width: pane 너비
            pane_height: pane 높이

        Returns:
            session_id

        Raises:
            RuntimeError: tmux 미설치, 에이전트 미설치, 또는 cwd 없음
        """
        self._require_tmux()

        config = get_agent_config(agent_name)
        agent_cmd = config["command"]

        # 에이전트 바이너리 확인 (플래그 제외하고 실행 파일만 체크)
        agent_bin = agent_cmd
        try:
            parts = shlex.split(agent_cmd, posix=os.name != "nt")
            agent_bin = parts[0] if parts else ""
        except ValueError:
            logger.warning("에이전트 명령 파싱 실패, 원본 문자열로 판정합니다: %s", agent_cmd)
            agent_bin = agent_cmd.strip().split()[0] if agent_cmd.strip() else ""
        if not shutil.which(agent_bin):
            raise RuntimeError(
                f"에이전트 '{agent_bin}'이(가) PATH에 없습니다. "
                f"설치 후 다시 시도하세요."
            )

        # 작업 디렉토리 확인
        work_dir = cwd or os.getcwd()
        if not Path(work_dir).is_dir():
            raise RuntimeError(
                f"작업 디렉토리가 존재하지 않습니다: {work_dir}"
            )

        # 세션 ID 및 이름 생성 (특수문자 제거)
        session_id = f"tmux_{uuid.uuid4().hex[:8]}"
        raw_name = session_name or f"sepilot_{agent_name}_{session_id[-8:]}"
        tmux_name = _sanitize_tmux_name(raw_name)

        # 환경 변수 설정 (값 이스케이프로 공백/특수문자 안전)
        merged_env = {**config.get("env", {}), **(env or {})}
        env_prefix = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in merged_env.items()
        )

        # 동일 이름의 기존 세션이 있으면 정리 (이전 실행 잔여물)
        if self._is_session_alive(tmux_name):
            logger.warning(
                "기존 tmux 세션 '%s' 발견 — 정리 후 재생성합니다.", tmux_name
            )
            try:
                subprocess.run(
                    ["tmux", "kill-session", "-t", tmux_name],
                    capture_output=True, text=True, timeout=5,
                )
            except Exception:
                pass

        # tmux 세션 생성 (detached)
        create_cmd = [
            "tmux", "new-session",
            "-d",               # detached
            "-s", tmux_name,    # session name
            "-x", str(pane_width),
            "-y", str(pane_height),
        ]

        try:
            subprocess.run(
                create_cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=work_dir,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"tmux 세션 생성 실패: {e.stderr.strip()}"
            ) from e

        # 에이전트 명령 전송 (경로 이스케이프로 쉘 인젝션 방지)
        safe_dir = shlex.quote(work_dir)
        agent_launch = (
            f"{env_prefix} {agent_cmd}".strip() if env_prefix else agent_cmd
        )
        self._tmux_send_keys(tmux_name, f"cd {safe_dir} && {agent_launch}")

        # 세션 등록 (Lock 보호)
        session = TmuxSession(
            session_id=session_id,
            agent_name=agent_name,
            tmux_session_name=tmux_name,
            cwd=work_dir,
            created_at=time.time(),
            worktree_id=worktree_id,
        )
        with self._sessions_lock:
            self._sessions[session_id] = session

        try:
            # startup 대기
            startup_wait = config.get("startup_wait", 3)
            time.sleep(startup_wait)

            # 초기 상태 캡처
            initial = self._raw_capture(tmux_name)
            session.last_capture = initial
            session.last_capture_at = time.time()
            session.status = "idle"
        except Exception:
            # startup 실패 시 세션 정리
            self.destroy_session(session_id)
            raise

        logger.info(
            "tmux 세션 생성: %s (agent=%s, tmux=%s, cwd=%s)",
            session_id, agent_name, tmux_name, work_dir,
        )

        return session_id

    def destroy_session(self, session_id: str) -> bool:
        """tmux 세션을 종료합니다.

        Args:
            session_id: 세션 ID

        Returns:
            성공 여부
        """
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

        # 에이전트 종료 명령 전송 시도
        config = get_agent_config(session.agent_name)
        exit_cmd = config.get("exit_command", "exit")
        if self._is_session_alive(session.tmux_session_name):
            try:
                self._tmux_send_keys(session.tmux_session_name, exit_cmd)
                time.sleep(1)
            except Exception as e:
                logger.debug("에이전트 종료 명령 전송 실패 (무시): %s — %s", session_id, e)

        # tmux 세션 강제 종료
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", session.tmux_session_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"tmux kill-session 실패: {e}")

        session.status = "completed"
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

        logger.info(f"tmux 세션 종료: {session_id}")
        return True

    def destroy_all(self) -> int:
        """모든 관리 중인 tmux 세션을 종료합니다."""
        with self._sessions_lock:
            ids = list(self._sessions.keys())
        count = 0
        for sid in ids:
            if self.destroy_session(sid):
                count += 1
        return count

    # ------------------------------------------------------------------
    # 입출력
    # ------------------------------------------------------------------

    def send_keys(
        self,
        session_id: str,
        text: str,
        enter: bool = True,
    ) -> None:
        """tmux 세션에 텍스트를 전송합니다.

        Args:
            session_id: 세션 ID
            text: 전송할 텍스트
            enter: Enter 키를 누를지 여부

        Raises:
            ValueError: 세션이 존재하지 않음
            RuntimeError: tmux 세션이 죽어있음
        """
        session = self._get_session(session_id)

        # 세션 생존 확인
        if not self._is_session_alive(session.tmux_session_name):
            session.status = "error"
            raise RuntimeError(
                f"tmux 세션이 이미 종료되었습니다: {session_id} "
                f"(tmux={session.tmux_session_name})"
            )

        # 전송 전 현재 출력을 기록 (새 출력 추출용)
        # cleaned 기준으로 cursor를 설정하여 _extract_new_output과 일관성 유지
        current_raw = self._raw_capture(session.tmux_session_name)
        current_cleaned = self._clean_output(current_raw)
        session.last_capture = current_raw
        session.last_capture_at = time.time()
        session._capture_cursor = len(current_cleaned)

        self._tmux_send_keys(session.tmux_session_name, text, enter=enter)
        session.status = "busy"

    def capture_pane(
        self,
        session_id: str,
        lines: int = 100,
    ) -> str:
        """tmux pane의 현재 출력을 캡처합니다.

        Args:
            session_id: 세션 ID
            lines: 캡처할 줄 수

        Returns:
            클리닝된 pane 출력
        """
        session = self._get_session(session_id)
        raw = self._raw_capture(session.tmux_session_name, lines=lines)
        cleaned = self._clean_output(raw)
        # last_capture는 항상 raw로 저장 (일관성)
        session.last_capture = raw
        session.last_capture_at = time.time()
        return cleaned

    def get_new_output(self, session_id: str) -> str:
        """마지막 send_keys 이후의 새 출력만 반환합니다.

        _capture_cursor 기반으로 증분 추출합니다 (wait_for_idle과 동일 알고리즘).

        Args:
            session_id: 세션 ID

        Returns:
            새로운 출력 텍스트
        """
        session = self._get_session(session_id)
        current = self._raw_capture(session.tmux_session_name)

        session.last_capture = current
        session.last_capture_at = time.time()

        new_output = self._extract_new_output(session, current)
        # cursor 갱신하여 다음 호출 시 중복 방지
        session._capture_cursor = len(self._clean_output(current))
        return new_output

    def wait_for_idle(
        self,
        session_id: str,
        timeout: int = 300,
        stable_seconds: float | None = None,
        poll_interval: float = 0.5,
    ) -> str:
        """에이전트가 idle 상태가 될 때까지 대기합니다.

        4단계 idle 감지:
        1. min_wait: 에이전트가 처리를 시작하기까지 최소 대기
        2. busy 패턴: 매칭되면 아직 작업 중으로 판단
        3. 출력 안정성: stable_seconds 동안 새 출력 없음
        4. idle 패턴: 에이전트별 프롬프트 패턴 매칭
        (타임아웃 시 현재까지의 출력 반환)

        Args:
            session_id: 세션 ID
            timeout: 최대 대기 시간 (초)
            stable_seconds: 출력 안정 판정 시간 (None이면 에이전트 설정 사용)
            poll_interval: 폴링 간격 (초)

        Returns:
            에이전트 응답 텍스트 (send_keys 이후 새 출력)
        """
        session = self._get_session(session_id)
        config = get_agent_config(session.agent_name)
        idle_patterns = [re.compile(p) for p in config.get("idle_patterns", [])]
        busy_patterns = [re.compile(p) for p in config.get("busy_patterns", [])]
        min_wait = config.get("min_wait", 1.0)
        if stable_seconds is None:
            stable_seconds = config.get("stable_seconds", 3.0)

        # idle 패턴 없이 안정성만으로 idle 판정하는 fallback 임계값
        stability_fallback = stable_seconds * 4

        # send_keys 시점의 출력 길이 (새 출력 감지용)
        baseline_len = session._capture_cursor
        output_changed = False  # 에이전트가 실제로 출력을 생성했는지

        start = time.time()
        last_output = ""
        stable_since: float | None = None
        last_alive_check = 0.0
        session_died = False

        while time.time() - start < timeout:
            elapsed = time.time() - start

            # 세션 생존 확인 (즉시 1회 + 이후 10초마다)
            if last_alive_check == 0.0 or elapsed - last_alive_check >= 10.0:
                last_alive_check = elapsed
                if not self._is_session_alive(session.tmux_session_name):
                    session.status = "error"
                    logger.warning(f"tmux 세션이 죽었습니다: {session_id}")
                    session_died = True
                    break

            current = self._raw_capture(session.tmux_session_name)
            cleaned = self._clean_output(current)
            visible_new_output = self._extract_new_output(session, current)

            # 에이전트가 실제로 새 출력을 생성했는지 확인
            if (
                not output_changed
                and visible_new_output.strip()
                and len(cleaned) > baseline_len
            ):
                output_changed = True

            if cleaned != last_output:
                last_output = cleaned
                stable_since = time.time()
            elif stable_since is not None:
                elapsed_stable = time.time() - stable_since

                # min_wait 이전에는 idle 판정하지 않음
                if elapsed < min_wait:
                    time.sleep(poll_interval)
                    continue

                # 에이전트가 아직 출력을 생성하지 않았으면 idle 판정 보류
                if not output_changed:
                    time.sleep(poll_interval)
                    continue

                # busy 패턴이 매칭되면 아직 작업 중
                if self._matches_busy(cleaned, busy_patterns):
                    stable_since = time.time()
                    time.sleep(poll_interval)
                    continue

                if elapsed_stable >= stable_seconds:
                    # 안정 상태 — idle 패턴 확인
                    if self._matches_idle(cleaned, idle_patterns):
                        session.status = "idle"
                        result = self._extract_new_output(session, current)
                        session.last_capture = current
                        session.last_capture_at = time.time()
                        session._capture_cursor = len(cleaned)
                        return result

                    # fallback: idle 패턴 미매칭이어도 충분히 안정화되면 idle로 판정
                    if elapsed_stable >= stability_fallback:
                        logger.info(
                            "idle 패턴 미매칭이지만 출력 안정 %.1fs — idle 판정: %s",
                            elapsed_stable, session_id,
                        )
                        session.status = "idle"
                        result = self._extract_new_output(session, current)
                        session.last_capture = current
                        session.last_capture_at = time.time()
                        session._capture_cursor = len(cleaned)
                        return result

            if stable_since is None:
                stable_since = time.time()

            time.sleep(poll_interval)

        if session_died:
            session.last_capture = self._raw_capture(session.tmux_session_name)
            session.last_capture_at = time.time()
            result = self._extract_new_output(session, session.last_capture)
            session._capture_cursor = len(self._clean_output(session.last_capture))
            return result or "[error] tmux 세션이 종료되었습니다."

        # 타임아웃: 현재까지 출력 반환
        if not output_changed:
            logger.warning(
                "tmux wait_for_idle 타임아웃 (%ds), 에이전트 출력 없음: %s",
                timeout, session_id,
            )
        else:
            logger.warning(
                "tmux wait_for_idle 타임아웃 (%ds): %s", timeout, session_id,
            )
        session.status = "idle"
        session.last_capture = self._raw_capture(session.tmux_session_name)
        session.last_capture_at = time.time()
        result = self._extract_new_output(session, session.last_capture)
        session._capture_cursor = len(self._clean_output(session.last_capture))
        if not output_changed and not result.strip():
            return "[timeout] 에이전트가 응답하지 않았습니다."
        return result

    def send_and_wait(
        self,
        session_id: str,
        text: str,
        timeout: int = 300,
        stable_seconds: float | None = None,
    ) -> str:
        """텍스트를 전송하고 idle 상태까지 대기합니다.

        send_keys + wait_for_idle의 편의 래퍼입니다.
        """
        self.send_keys(session_id, text)
        return self.wait_for_idle(
            session_id,
            timeout=timeout,
            stable_seconds=stable_seconds,
        )

    # ------------------------------------------------------------------
    # 세션 조회
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> TmuxSession | None:
        """세션 정보를 반환합니다."""
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> dict[str, dict]:
        """모든 활성 세션의 상태를 반환합니다."""
        with self._sessions_lock:
            snapshot = list(self._sessions.items())

        result = {}
        dead_sids = []
        for sid, session in snapshot:
            runtime = time.time() - session.created_at
            alive = self._is_session_alive(session.tmux_session_name)
            if not alive:
                session.status = "completed"
                dead_sids.append(sid)

            result[sid] = {
                "agent_name": session.agent_name,
                "tmux_session": session.tmux_session_name,
                "cwd": session.cwd,
                "status": session.status,
                "runtime": f"{runtime:.0f}s",
                "worktree_id": session.worktree_id,
            }

        # 죽은 세션을 dict에서 제거하여 누적 방지
        if dead_sids:
            with self._sessions_lock:
                for sid in dead_sids:
                    self._sessions.pop(sid, None)

        return result

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str) -> TmuxSession:
        """세션을 가져오거나 에러를 발생시킵니다."""
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        return session

    @staticmethod
    def _tmux_send_keys(
        tmux_session: str,
        text: str,
        enter: bool = True,
    ) -> None:
        """tmux send-keys 명령을 실행합니다.

        멀티라인 텍스트는 줄바꿈이 Enter로 해석되는 것을 방지하기 위해
        tmux set-buffer + paste-buffer 방식을 사용합니다.
        모든 전송은 _send_lock으로 직렬화하여 병렬 호출 시 문자 섞임을 방지합니다.
        """
        try:
            if "\n" in text:
                # 멀티라인: set-buffer → paste-buffer (전역 버퍼 원자성)
                # Lock으로 보호하여 병렬 호출 시 버퍼 경합 방지
                with TmuxSessionManager._send_lock:
                    subprocess.run(
                        ["tmux", "set-buffer", text],
                        check=True, capture_output=True, text=True, timeout=10,
                    )
                    subprocess.run(
                        ["tmux", "paste-buffer", "-t", tmux_session],
                        check=True, capture_output=True, text=True, timeout=10,
                    )
                if enter:
                    # paste 처리 완료 대기 후 Enter (CLI 에이전트가 paste를 인식할 시간)
                    time.sleep(0.3)
                    subprocess.run(
                        ["tmux", "send-keys", "-t", tmux_session, "Enter"],
                        check=True, capture_output=True, text=True, timeout=10,
                    )
            else:
                # 단일 라인: -l로 리터럴 전송 (Enter/Space 등 특수 키 해석 방지)
                subprocess.run(
                    ["tmux", "send-keys", "-t", tmux_session, "-l", text],
                    check=True, capture_output=True, text=True, timeout=10,
                )
                if enter:
                    subprocess.run(
                        ["tmux", "send-keys", "-t", tmux_session, "Enter"],
                        check=True, capture_output=True, text=True, timeout=10,
                    )
        except subprocess.TimeoutExpired as exc:
            logger.warning("tmux send-keys 타임아웃 (10s): session=%s", tmux_session)
            raise RuntimeError(
                f"tmux send-keys 타임아웃: {tmux_session}"
            ) from exc
        except subprocess.CalledProcessError as e:
            logger.warning("tmux send-keys 실패: session=%s, err=%s", tmux_session, e)
            raise

    @staticmethod
    def _raw_capture(tmux_session: str, lines: int = 200) -> str:
        """tmux capture-pane으로 raw 출력을 캡처합니다."""
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "capture-pane",
                    "-p",  # stdout으로 출력
                    "-t",
                    tmux_session,
                    "-S",
                    f"-{lines}",  # 시작 줄 (음수 = 위로)
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            returncode = result.returncode if isinstance(result.returncode, int) else 0
            if returncode != 0:
                logger.warning(
                    "tmux capture-pane 비정상 종료 (rc=%d): %s",
                    returncode, result.stderr.strip(),
                )
                return ""
            return result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("tmux capture-pane 실패: %s", e)
            return ""

    @staticmethod
    def _clean_output(text: str) -> str:
        """ANSI 이스케이프 시퀀스와 불필요한 공백을 제거합니다."""
        # ANSI 이스케이프 제거
        text = _ANSI_RE.sub("", text)
        # 나머지 bare ESC 제거 (패턴에 안 잡힌 것들)
        text = text.replace("\x1b", "")
        # trailing 빈 줄 제거
        lines = text.rstrip().split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def _matches_idle(text: str, patterns: list[re.Pattern]) -> bool:
        """출력의 마지막 몇 줄에서 idle 패턴이 매칭되는지 확인합니다.

        에이전트 프롬프트가 반드시 마지막 줄에 있지 않을 수 있으므로
        (예: claude code의 상태바가 프롬프트 아래에 출력됨)
        마지막 5줄까지 확인합니다.
        """
        if not patterns:
            return True  # 패턴이 없으면 항상 idle로 판정

        lines = text.strip().split("\n")
        if not lines:
            return False

        check_lines = lines[-5:] if len(lines) >= 5 else lines
        return any(p.search(line) for line in check_lines for p in patterns)

    @staticmethod
    def _matches_busy(text: str, patterns: list[re.Pattern]) -> bool:
        """출력의 마지막 몇 줄에서 busy 패턴이 감지되는지 확인합니다."""
        if not patterns:
            return False

        lines = text.strip().split("\n")
        # 마지막 5줄까지 확인 (상태바/스피너가 프롬프트 아래에 있을 수 있음)
        check_lines = lines[-5:] if len(lines) >= 5 else lines
        return any(any(p.search(line) for p in patterns) for line in check_lines)

    def _extract_new_output(self, session: TmuxSession, raw_capture: str) -> str:
        """send_keys 이후 새 출력을 추출합니다.

        cursor 기반으로 새 출력을 추출한 후, CLI 에이전트 UI 장식을 제거합니다.
        """
        cleaned = self._clean_output(raw_capture)
        cursor = session._capture_cursor
        if cursor > 0 and cursor <= len(cleaned):
            new_text = cleaned[cursor:].strip()
        else:
            new_text = cleaned.strip()

        return self._strip_agent_decorations(new_text)

    @staticmethod
    def _strip_agent_decorations(text: str) -> str:
        """CLI 에이전트 UI 장식을 제거합니다.

        Claude Code, opencode 등의 터미널 UI 요소를 제거하여
        순수한 응답 텍스트만 반환합니다.
        """
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            # 구분선 (─, ═, ━ 등으로만 구성된 줄, 최소 5자 이상)
            if len(stripped) >= 5 and all(c in "─═━╌╍" for c in stripped):
                continue
            # 프롬프트 라인 (❯, >, $ 로 시작하는 빈 프롬프트)
            if re.match(r"^[❯>$]\s*$", stripped):
                continue
            # USAGE 상태바 (claude code 형식: "USAGE: 1.5K tok ...")
            if re.match(r"^USAGE:\s+[\d.]+K?\s+tok", stripped):
                continue
            # Claude Code 스피너/상태 표시
            if re.match(r"^[◐◓◑◒]", stripped):
                continue
            # tmux 시작 명령 에코 (cd /path && NO_COLOR=1 claude --flags)
            if re.match(
                r"^cd\s+\S+.*&&\s*(NO_COLOR=1\s+)?(claude|opencode|codex|gemini)(?:\s+.+)?$",
                stripped,
            ):
                continue
            # 쉘 프롬프트 에코
            if re.match(r"^\S+@\S+:.*\$", stripped):
                continue
            # Claude Code 배너 (블록 문자가 실제로 포함된 줄만)
            if re.match(r"^[▐▝▘]+", stripped) or re.match(r"^[▐▝▘\s]+[▛█▜▟]+", stripped):
                if "Claude Code" not in stripped:
                    continue
            # "[Pasted text #N +M lines]" 표시
            if re.match(r"^\[Pasted text #\d+", stripped):
                continue
            result.append(line)

        # 앞뒤 빈 줄 정리
        text = "\n".join(result).strip()
        return text

    @staticmethod
    def _is_session_alive(tmux_session: str) -> bool:
        """tmux 세션이 살아있는지 확인합니다."""
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", tmux_session],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.returncode == 0
        except Exception:
            return False
