"""Docker 컨테이너 내부에서 실행되는 독립 스크립트.

SWE-bench 인스턴스 이미지 안에서 SEPilot 에이전트를 실행하고,
생성된 패치를 수집하여 결과 파일로 출력합니다.

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
    """git diff HEAD 로 현재 변경 사항을 패치로 수집."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, timeout=30,
            cwd="/testbed",
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # staged changes 도 확인
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True, text=True, timeout=30,
            cwd="/testbed",
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    return ""


def run_agent(problem_statement: str) -> str:
    """SEPilot 에이전트를 실행하여 패치를 생성."""
    # 모든 import를 먼저 수행 (격리된 초기 CWD에서, testbed 코드와 충돌 없음)
    from sepilot.config.settings import Settings
    from sepilot.loggers.file_logger import FileLogger
    from sepilot.agent.base_agent import ReactAgent

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


async def _async_run_team_agent(problem_statement: str) -> str:
    """팀 모드: TeamOrchestrator를 사용하여 에이전트 팀을 실행.

    패턴: import 먼저 완료 → /testbed로 chdir → LLM 생성 → 팀 실행 → collect_patch()
    """
    # 1. 모든 import를 먼저 (격리된 초기 CWD에서, testbed 코드와 충돌 없음)
    from sepilot.config.settings import Settings
    from sepilot.agent.subagent.team_orchestrator import TeamOrchestrator
    from sepilot.agent.subagent.team_agents import (
        PMAgent,
        ResearcherAgent,
        DeveloperAgent,
        TesterAgent,
        DebuggerAgent,
    )
    from sepilot.agent.subagent.team_models import TeamRole

    # 2. import 완료 후 /testbed로 이동
    if os.path.isdir("/testbed"):
        os.chdir("/testbed")

    # 3. 환경변수에서 LLM 설정 로드
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

    # 4. LLM 객체 생성
    llm = settings.create_llm()

    # 5. TeamOrchestrator 구성 (컨테이너 내 직렬 실행: max_parallel=1)
    orchestrator = TeamOrchestrator(llm=llm, max_parallel=1)

    pm = PMAgent(agent_id="pm", llm=llm)
    researcher = ResearcherAgent(agent_id="researcher", llm=llm)
    developer = DeveloperAgent(agent_id="developer", llm=llm)
    tester = TesterAgent(agent_id="tester", llm=llm)
    debugger = DebuggerAgent(agent_id="debugger", llm=llm)

    orchestrator.register_agent(TeamRole.PM, pm)
    orchestrator.register_agent(TeamRole.RESEARCHER, researcher)
    orchestrator.register_agent(TeamRole.DEVELOPER, developer)
    orchestrator.register_agent(TeamRole.TESTER, tester)
    orchestrator.register_agent(TeamRole.DEBUGGER, debugger)

    # 6. 팀 태스크 실행
    await orchestrator.execute_team_task(problem_statement)

    # 7. 팀 출력이 아닌 git diff로 패치 수집
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
        with open(args.problem_file, "r", encoding="utf-8") as f:
            problem_statement = f.read().strip()

        if not problem_statement:
            result["error"] = "Empty problem statement"
            _write_result(result, args.output_path)
            sys.exit(1)

        # 팀 모드 결정 (CLI 플래그 OR 환경변수)
        team_mode = args.team_mode or os.getenv("TEAM_MODE", "0") == "1"

        # 에이전트 실행
        if team_mode:
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
                with open(session_log_path, "r", encoding="utf-8") as sf:
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
