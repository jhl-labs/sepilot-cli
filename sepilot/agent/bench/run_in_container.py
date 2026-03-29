"""Docker 컨테이너 내부에서 실행되는 독립 스크립트.

SWE-bench 인스턴스 이미지 안에서 에이전트를 실행하고,
생성된 패치를 수집하여 결과 파일로 출력합니다.

에이전트 모드 (DEFAULT_MODEL 환경변수로 결정):
  - 단일 에이전트 (기본): SEPilot ReactAgent가 도구를 사용하여 직접 코드 수정
  - CLI 에이전트 (cli:X):  외부 CLI(claude/codex/opencode)를 subprocess로 실행
  - tmux 에이전트 (tmux:X): 외부 CLI를 tmux 세션(pseudo-terminal)에서 실행
  - 팀 모드 (--team-mode):  PM이 계획 수립 → ReactAgent가 계획 기반 실행

Usage:
    python -m sepilot.agent.bench.run_in_container \
        --instance-id <id> \
        --problem-file /testbed/sepilot_problem.txt \
        --output-path /testbed/sepilot_result.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback

# Global state for signal handler
_RESULT_OUTPUT_PATH: str = ""
_RESULT_INSTANCE_ID: str = ""
_START_TIME: float = 0.0


def _sigterm_handler(signum, frame):
    """SIGTERM 핸들러: 컨테이너 kill 전에 현재 패치를 result.json에 저장."""
    if not _RESULT_OUTPUT_PATH:
        sys.exit(1)
    patch = collect_patch()
    result = {
        "instance_id": _RESULT_INSTANCE_ID,
        "model_patch": patch,
        "exit_code": 0 if patch else 1,
        "error": "SIGTERM received (timeout)",
        "duration_seconds": time.time() - _START_TIME,
    }
    _write_result(result, _RESULT_OUTPUT_PATH)
    sys.exit(0 if patch else 1)


def setup_environment():
    """컨테이너 내부 환경 설정.

    bootstrap_cmd에서 이미 의존성이 설치되고 PYTHONPATH가 설정되어 있지만,
    안전하게 한번 더 확인합니다.
    """
    sepilot_path = "/sepilot"
    if os.path.exists(sepilot_path) and sepilot_path not in sys.path:
        sys.path.insert(0, sepilot_path)


def collect_patch() -> str:
    """git diff로 현재 변경 사항을 패치로 수집.

    수집 순서:
    1. git diff HEAD — 커밋 안 된 변경 (대부분의 에이전트)
    2. git diff --cached — staged 변경
    3. git diff HEAD~1 HEAD — 마지막 커밋의 변경 (opencode 등 자동 커밋 에이전트)
    4. git log --oneline로 SWE-bench 기준 커밋 이후 모든 변경 수집
    """
    cwd = "/testbed"

    # 1. uncommitted changes
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, timeout=30, cwd=cwd,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # 2. staged changes
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True, text=True, timeout=30, cwd=cwd,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # 3. 에이전트가 커밋한 경우 (opencode 등):
    # SWE-bench 이미지의 초기 상태 커밋("SWE-bench" 메시지) 이후의 변경 수집
    try:
        # "SWE-bench" 커밋 해시 찾기 (swebench harness가 생성하는 초기 커밋)
        result = subprocess.run(
            ["git", "log", "--all", "--grep=SWE-bench", "--format=%H", "-1"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )
        if result.returncode == 0 and result.stdout.strip():
            base_commit = result.stdout.strip()
            result = subprocess.run(
                ["git", "diff", base_commit, "HEAD", "--", ".", ":(exclude)sepilot_agent_venv"],
                capture_output=True, text=True, timeout=30, cwd=cwd,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    except Exception:
        pass

    return ""


def run_agent(problem_statement: str) -> str:
    """SEPilot 에이전트를 실행하여 패치를 생성."""
    # 모든 import를 먼저 수행 (격리된 초기 CWD에서, testbed 코드와 충돌 없음)
    from sepilot.agent.base_agent import ReactAgent
    from sepilot.config.settings import Settings
    from sepilot.loggers.file_logger import FileLogger

    # import 완료 후 /testbed로 이동 (에이전트 도구가 작업할 디렉토리)
    # sys.path에는 /testbed가 없으므로 import 충돌 없음
    if os.path.isdir("/testbed"):
        os.chdir("/testbed")

    # 환경변수에서 LLM 설정 로드
    model = os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview")
    api_base_url = os.getenv("OPENAI_API_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")

    # 모델별 context window (API 제한에 맞춰 설정)
    _MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "step-3.5-flash": 32768,
        "step-3.5": 65536,
        "qwen3-vl-235b": 131072,
        "qwen3-coder": 131072,
        "glm-4.7-cloud": 131072,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
    }
    context_window = int(os.getenv("CONTEXT_WINDOW", "0"))
    if not context_window:
        # 모델명 부분 매칭으로 context window 결정
        model_lower = model.lower()
        for key, val in _MODEL_CONTEXT_WINDOWS.items():
            if key in model_lower:
                context_window = val
                break
        else:
            context_window = 128000  # 기본값

    # 소형 컨텍스트 모델: iterations 제한 + compact 프롬프트 + 긴 API 타임아웃
    _max_iters = 30 if context_window <= 32768 else 50
    _prompt_profile = "swe_bench_compact" if context_window <= 32768 else "swe_bench"
    # 소형 모델 (느린 추론): request_timeout 180초 (기본 60초에서 LLM 호출 타임아웃 발생)
    _request_timeout = 180 if context_window <= 32768 else 60
    # 소형 컨텍스트 모델: 낮은 temperature로 안정적 도구 호출 유도
    # 0.3: malformed call 빈발, 0.2: 4/5, 0.1: 결정론적이되 루프 리스크 낮음
    _temperature = 0.1 if context_window <= 32768 else 0.3

    settings = Settings(
        model=model,
        prompt_profile=_prompt_profile,
        max_iterations=_max_iters,
        verbose=False,
        enable_streaming=False,
        context_window=context_window,
        temperature=_temperature,
        request_timeout=_request_timeout,
    )

    # API 설정 오버라이드
    if api_base_url:
        settings.api_base_url = api_base_url
    if api_key:
        settings.openai_api_key = api_key

    # 멀티모델 tier 설정 (환경변수에서 자동 로드됨, 명시적 확인)
    for attr, env_key in [
        ("triage_model", "SEPILOT_TRIAGE_MODEL"),
        ("verifier_model", "SEPILOT_VERIFIER_MODEL"),
        ("reasoning_model", "SEPILOT_REASONING_MODEL"),
        ("quick_model", "SEPILOT_QUICK_MODEL"),
    ]:
        val = os.environ.get(env_key)
        if val:
            setattr(settings, attr, val)

    logger = FileLogger(log_dir=os.path.join(tempfile.gettempdir(), "sepilot-logs"))

    agent = ReactAgent(
        settings=settings,
        logger=logger,
        prompt_profile=_prompt_profile,
        auto_approve=True,
        show_progress=False,
        enable_memory=False,
    )

    agent.execute(problem_statement)

    # 세션 로그 경로를 환경변수에 저장 (main에서 result에 포함할 수 있도록)
    session_path = logger.get_session_path()
    if session_path and os.path.exists(session_path):
        os.environ["_AGENT_SESSION_LOG_PATH"] = str(session_path)

    return collect_patch()


def run_cli_agent(problem_statement: str) -> str:
    """CLI 에이전트(claude, codex, opencode, gemini)를 독립 실행하여 패치 생성.

    sepilot의 도구 체계를 사용하지 않고, CLI 에이전트 자체의 도구로
    /testbed의 코드를 직접 수정하게 합니다.
    """
    import shutil

    if os.path.isdir("/testbed"):
        os.chdir("/testbed")

    model = os.getenv("DEFAULT_MODEL", "cli:claude")
    agent_name = (model.split(":", 1)[1] if ":" in model else model).strip().lower()

    # CLI 에이전트별 명령어 구성
    cli_configs = {
        "claude": {
            "cmd": ["claude", "-p", "--allowedTools", "Edit,Read,Bash,Write,Glob,Grep"],
            "stdin": True,
        },
        "codex": {
            "cmd": ["codex", "exec", "-", "--sandbox", "danger-full-access"],
            "stdin": True,
        },
        "opencode": {
            "cmd": ["opencode", "run", "--model", "ollama-cloud/minimax-m2.5"],
            "stdin": False,  # opencode run takes message as positional args
        },
        "gemini": {
            "cmd": ["gemini", "--approval-mode", "yolo", "-p"],
            "stdin": False,
        },
    }

    config = cli_configs.get(agent_name)
    if not config:
        raise ValueError(f"Unknown CLI agent: {agent_name}")

    if not shutil.which(agent_name):
        raise FileNotFoundError(f"CLI agent '{agent_name}' not found in PATH")

    prompt = (
        "You are an expert software engineer. Fix the bug described below.\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the issue carefully. Reproduce it if possible to confirm your understanding.\n"
        "2. Search the codebase to find the root cause. Use grep/find to locate relevant code.\n"
        "3. Fix the root cause by editing the source files. Correctness over brevity.\n"
        "4. Preserve all necessary edge cases — don't simplify away required logic.\n"
        "5. Only modify source files. Do NOT create new test files.\n"
        "6. After editing, verify the fix makes sense by re-reading the changed code.\n\n"
        "The project is in the current directory (/testbed).\n\n"
        f"## Issue\n\n{problem_statement}"
    )

    timeout = int(os.getenv("CLI_AGENT_TIMEOUT", "1200"))

    try:
        cmd = config["cmd"]
        if config["stdin"]:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/testbed",
            )
        elif agent_name == "opencode":
            # opencode run: 프롬프트를 positional args로 직접 전달
            result = subprocess.run(
                cmd + [prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/testbed",
            )
        else:
            # gemini: -p flag takes prompt as argument
            result = subprocess.run(
                cmd + [prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/testbed",
            )

        if result.stderr:
            # Log stderr but don't fail — many CLI tools emit warnings
            sys.stderr.write(f"[CLI agent stderr]: {result.stderr[:2000]}\n")

    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[CLI agent] {agent_name} timed out after {timeout}s\n")
    except Exception as e:
        sys.stderr.write(f"[CLI agent] {agent_name} error: {e}\n")

    return collect_patch()


def run_tmux_agent(problem_statement: str) -> str:
    """tmux 세션에서 CLI 에이전트를 대화형으로 실행하여 패치 생성.

    subprocess.run(capture_output=True)와 달리, tmux는 pseudo-terminal을 제공하여
    CLI 에이전트가 완전한 인터랙티브 환경에서 동작합니다.

    모델명 형식: tmux:claude, tmux:codex, tmux:opencode
    """
    import shutil

    if os.path.isdir("/testbed"):
        os.chdir("/testbed")

    model = os.getenv("DEFAULT_MODEL", "tmux:claude")
    agent_name = (model.split(":", 1)[1] if ":" in model else model).strip().lower()

    # tmux 설치 확인
    if not shutil.which("tmux"):
        # tmux가 없으면 설치 시도
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "tmux"],
            capture_output=True, timeout=60,
        )
    if not shutil.which("tmux"):
        sys.stderr.write("[tmux agent] tmux not available, falling back to cli mode\n")
        os.environ["DEFAULT_MODEL"] = f"cli:{agent_name}"
        return run_cli_agent(problem_statement)

    # CLI 에이전트별 tmux 명령어 구성
    prompt = (
        "You are an expert software engineer. Fix the bug described below.\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the issue carefully. Reproduce it if possible to confirm your understanding.\n"
        "2. Search the codebase to find the root cause. Use grep/find to locate relevant code.\n"
        "3. Fix the root cause by editing the source files. Correctness over brevity.\n"
        "4. Preserve all necessary edge cases — don't simplify away required logic.\n"
        "5. Only modify source files. Do NOT create new test files.\n"
        "6. After editing, verify the fix makes sense by re-reading the changed code.\n\n"
        "The project is in the current directory (/testbed).\n\n"
        f"## Issue\n\n{problem_statement}"
    )

    # 프롬프트를 파일에 저장 (tmux에서 stdin 전달 대신 파일 사용)
    prompt_file = os.path.join(tempfile.gettempdir(), "tmux_prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    timeout = int(os.getenv("CLI_AGENT_TIMEOUT", "1200"))
    session_name = f"bench_{agent_name}"

    # 에이전트별 tmux 실행 명령
    tmux_cmds = {
        "claude": f"cat {prompt_file} | claude -p --allowedTools Edit,Read,Bash,Write,Glob,Grep",
        "codex": f"cat {prompt_file} | codex exec - --sandbox danger-full-access",
        "opencode": f"opencode run --model ollama-cloud/minimax-m2.5 \"$(cat {prompt_file})\"",
    }

    tmux_cmd = tmux_cmds.get(agent_name)
    if not tmux_cmd:
        sys.stderr.write(f"[tmux agent] Unknown agent: {agent_name}\n")
        return ""

    # 완료 마커 파일
    done_marker = os.path.join(tempfile.gettempdir(), "tmux_done")
    if os.path.exists(done_marker):
        os.remove(done_marker)

    # tmux 세션에서 에이전트 실행 + 완료 시 마커 생성
    full_cmd = f"cd /testbed && {tmux_cmd}; touch {done_marker}"

    try:
        # tmux 세션 생성 및 명령 실행
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "bash", "-c", full_cmd],
            capture_output=True, text=True, timeout=10, cwd="/testbed",
        )

        # 완료 대기 (polling)
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(done_marker):
                break
            time.sleep(5)
        else:
            sys.stderr.write(f"[tmux agent] {agent_name} timed out after {timeout}s\n")
            # 타임아웃 시 tmux 세션 종료
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True, timeout=10,
            )

    except Exception as e:
        sys.stderr.write(f"[tmux agent] {agent_name} error: {e}\n")

    return collect_patch()


async def _async_run_team_agent(problem_statement: str) -> str:
    """팀 모드: PM이 계획을 수립하고, ReactAgent가 계획 기반으로 실행.

    기존 TeamOrchestrator(LLM 텍스트 전용)는 컨테이너 환경에서 도구를 사용할 수 없으므로,
    PM → 계획 생성 → ReactAgent(도구 보유)가 계획을 실행하는 2단계 방식을 사용합니다.
    """
    from sepilot.agent.subagent.team_agents import PMAgent
    from sepilot.config.llm_providers import create_llm_from_settings
    from sepilot.config.settings import Settings

    # 1. 환경변수에서 LLM 설정 로드
    model = os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview")
    api_base_url = os.getenv("OPENAI_API_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")

    settings = Settings(
        model=model,
        prompt_profile="swe_bench",
        max_iterations=50,
        verbose=False,
        enable_streaming=False,
    )
    if api_base_url:
        settings.api_base_url = api_base_url
    if api_key:
        settings.openai_api_key = api_key

    # 2. PM이 작업 계획 생성 (LLM 텍스트)
    llm = create_llm_from_settings(settings)
    pm = PMAgent(agent_id="pm", llm=llm)

    plan = await pm.create_team_plan(problem_statement)
    plan_text = "\n".join(
        f"- [{a.role.value}] {a.description}"
        for a in plan.assignments
    ) if plan and plan.assignments else "No plan generated"

    # 3. ReactAgent가 계획을 기반으로 실제 코드 수정
    # import 후 /testbed로 이동
    if os.path.isdir("/testbed"):
        os.chdir("/testbed")

    from sepilot.agent.base_agent import ReactAgent
    from sepilot.loggers.file_logger import FileLogger

    enhanced_prompt = (
        f"다음은 PM이 수립한 실행 계획입니다. 이 계획을 참고하여 문제를 해결하세요.\n\n"
        f"## PM 실행 계획\n{plan_text}\n\n"
        f"## 원본 문제\n{problem_statement}"
    )

    logger = FileLogger(log_dir=os.path.join(tempfile.gettempdir(), "sepilot-logs"))
    agent = ReactAgent(
        settings=settings,
        logger=logger,
        prompt_profile="swe_bench",
        auto_approve=True,
        show_progress=False,
        enable_memory=False,
    )

    agent.execute(enhanced_prompt)

    session_path = logger.get_session_path()
    if session_path and os.path.exists(session_path):
        os.environ["_AGENT_SESSION_LOG_PATH"] = str(session_path)

    return collect_patch()


def run_team_agent(problem_statement: str) -> str:
    """팀 모드 에이전트 실행 (동기 래퍼)."""
    import asyncio
    return asyncio.run(_async_run_team_agent(problem_statement))


def main():
    parser = argparse.ArgumentParser(description="Run SEPilot agent in SWE-bench container")
    parser.add_argument("--instance-id", required=True, help="SWE-bench instance ID")
    parser.add_argument("--problem-file", required=True, help="Path to problem statement file")
    parser.add_argument("--output-path", required=True, help="Path for JSON result output")
    parser.add_argument("--team-mode", action="store_true", default=False,
                        help="팀 모드 활성화 (PMAgent 주도 다중 에이전트)")
    args = parser.parse_args()

    # SIGTERM 핸들러 등록 (docker kill → SIGTERM → 패치 저장 후 종료)
    global _RESULT_OUTPUT_PATH, _RESULT_INSTANCE_ID, _START_TIME
    _RESULT_OUTPUT_PATH = args.output_path
    _RESULT_INSTANCE_ID = args.instance_id
    _START_TIME = time.time()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    start_time = _START_TIME
    result = {
        "instance_id": args.instance_id,
        "model_patch": "",
        "exit_code": 1,
        "error": None,
        "duration_seconds": 0.0,
    }

    try:
        # 환경 설정 (초기 CWD 유지 - /testbed를 sys.path에 넣지 않음)
        # /testbed는 에이전트 도구를 통해 접근
        setup_environment()

        # 문제 설명 읽기
        with open(args.problem_file, encoding="utf-8") as f:
            problem_statement = f.read().strip()

        if not problem_statement:
            result["error"] = "Empty problem statement"
            _write_result(result, args.output_path)
            sys.exit(1)

        # 팀 모드 결정 (CLI 플래그 OR 환경변수)
        team_mode = args.team_mode or os.getenv("TEAM_MODE", "0") == "1"

        # 에이전트 실행
        model = os.getenv("DEFAULT_MODEL", "")
        is_cli_agent = model.lower().startswith("cli:")
        is_tmux_agent = model.lower().startswith("tmux:")

        if is_tmux_agent:
            patch = run_tmux_agent(problem_statement)
        elif is_cli_agent:
            patch = run_cli_agent(problem_statement)
        elif team_mode:
            patch = run_team_agent(problem_statement)
        else:
            patch = run_agent(problem_statement)

        result["model_patch"] = patch
        result["exit_code"] = 0 if patch else 1
        result["duration_seconds"] = time.time() - start_time

        # 에이전트 세션 로그를 결과에 포함
        session_log_path = os.environ.get("_AGENT_SESSION_LOG_PATH", "")
        if session_log_path and os.path.exists(session_log_path):
            try:
                with open(session_log_path, encoding="utf-8") as sf:
                    agent_logs = sf.read()
                result["agent_session_log"] = agent_logs[-50000:]  # 마지막 50K
            except Exception:
                pass

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["duration_seconds"] = time.time() - start_time
        traceback.print_exc()

    _write_result(result, args.output_path)
    sys.exit(result["exit_code"])


def _write_result(result: dict, output_path: str):
    """결과를 JSON 파일로 저장."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
