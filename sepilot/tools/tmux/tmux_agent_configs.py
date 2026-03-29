"""tmux 에이전트별 설정.

각 CLI 에이전트가 tmux 세션에서 대화형으로 실행될 때 필요한
설정(명령어, idle 감지 패턴, 환경 변수 등)을 정의합니다.
"""

from __future__ import annotations

from typing import Any

# 에이전트별 tmux 세션 설정
TMUX_AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "claude": {
        "command": "claude --dangerously-skip-permissions",
        "startup_wait": 5,
        "idle_patterns": [
            r"❯\s*$",              # claude code 실제 프롬프트
            r"^\s*[\$>]\s*$",      # 쉘 프롬프트 (줄 전체가 프롬프트인 경우만)
        ],
        "busy_patterns": [
            r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏◐◓◑◒]",  # 스피너 (braille + circle)
            r"Thinking",
            r"Working",
        ],
        # 에이전트가 처리를 시작하기까지 최소 대기 시간 (초)
        # — 이 시간 전에는 idle 판정하지 않음
        "min_wait": 2.0,
        # idle 판정을 위한 출력 안정 시간 (초)
        # — claude는 thinking 중 긴 pause가 있으므로 길게 설정
        "stable_seconds": 5.0,
        "exit_command": "/exit",
        "env": {
            "NO_COLOR": "1",
        },
    },
    "opencode": {
        "command": "opencode",
        "startup_wait": 3,
        "idle_patterns": [
            r"^\s*>\s*$",
        ],
        "busy_patterns": [
            r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]",
        ],
        "min_wait": 1.0,
        "stable_seconds": 4.0,
        "exit_command": "/exit",
        "env": {},
    },
    "codex": {
        "command": "codex",
        "startup_wait": 3,
        "idle_patterns": [
            r"^\s*[\$>]\s*$",
        ],
        "busy_patterns": [],
        "min_wait": 1.0,
        "stable_seconds": 3.0,
        "exit_command": "exit",
        "env": {},
    },
    "gemini": {
        "command": "gemini",
        "startup_wait": 3,
        "idle_patterns": [
            r"^\s*[\$>]\s*$",
        ],
        "busy_patterns": [],
        "min_wait": 1.0,
        "stable_seconds": 3.0,
        "exit_command": "/exit",
        "env": {},
    },
}


def get_agent_config(agent_name: str) -> dict[str, Any]:
    """에이전트 설정을 가져옵니다. 미등록 에이전트는 기본값을 반환합니다."""
    if agent_name in TMUX_AGENT_CONFIGS:
        return TMUX_AGENT_CONFIGS[agent_name]

    # 미등록 에이전트에 대한 기본 설정
    return {
        "command": agent_name,
        "startup_wait": 3,
        "idle_patterns": [r"^\s*[\$>]\s*$"],
        "busy_patterns": [],
        "min_wait": 1.0,
        "stable_seconds": 3.0,
        "exit_command": "exit",
        "env": {},
    }
