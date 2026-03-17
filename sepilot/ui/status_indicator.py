"""Lightweight progress indicator for LLM execution.

Shows a spinner + status text during agent execution in interactive mode.
Uses Rich's Console.status() which runs a spinner animation in a background thread,
so it works even when the main thread is blocked by LLM calls.

This is designed for the default (non-verbose) mode. When verbose=True,
the existing StatusPanel/ProgressDisplay provides detailed output instead.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from rich.status import Status

# LangGraph node name → user-friendly status text
NODE_STATUS_MAP: dict[str, str] = {
    "triage": "요청을 분석해 실행 방향과 우선순위를 정하고 있어요...",
    "agent": "요청 해결을 위해 핵심 작업을 순서대로 진행하고 있어요...",
    "orchestrator": "현재 단계에 맞게 작업 순서를 조정하고 있어요...",
    "hierarchical_planner": "작업 단계를 나누고 우선순위를 정리하고 있어요...",
    "planner": "작업 단계를 나누고 우선순위를 정리하고 있어요...",
    "tools": "필요한 근거를 수집하기 위해 도구를 실행하고 있어요...",
    "verifier": "결과가 요청과 맞는지 검증하고 있어요...",
    "reporter": "확인된 결과를 최종 답변으로 정리하고 있어요...",
    "iteration_guard": "같은 작업 반복 여부를 점검하고 있어요...",
    "context_manager": "문맥을 정리해 다음 단계 정확도를 높이고 있어요...",
}

# Tool name → display name for status text
_TOOL_DISPLAY_NAMES: dict[str, str] = {
    "bash_execute": "명령 실행",
    "get_structure": "프로젝트 구조 확인",
    "search_content": "코드 내용 검색",
    "find_file": "파일 찾기",
    "list_directory": "디렉터리 확인",
    "file_read": "파일 읽기",
    "file_write": "파일 작성",
    "file_edit": "파일 수정",
    "ripgrep_search": "코드 검색",
    "glob_search": "파일 탐색",
    "git": "Git 작업",
    "web_search": "웹 검색",
    "web_fetch": "링크 확인",
    "think": "추론 정리",
}


class AgentStatusIndicator:
    """Lightweight spinner indicator for agent execution progress.

    Thread-safe wrapper around Rich's Console.status().
    Designed to be set on an agent instance and updated as nodes execute.

    Usage::

        indicator = AgentStatusIndicator(console)
        agent.status_indicator = indicator
        with indicator:
            result = agent.run(prompt)
    """

    def __init__(self, console: Console | None = None) -> None:
        # Use a Console pinned to the real stderr (sys.__stderr__) so the spinner
        # is immune to sys.stdout/stderr redirects (e.g., TeeOutput).
        # force_terminal=True ensures Rich uses ANSI animations regardless.
        real_file = sys.__stderr__ or sys.stderr
        self._console = Console(file=real_file, force_terminal=True)
        self._status: Status | None = None
        self._lock = threading.Lock()
        self._active = False
        self._started_at = time.monotonic()
        self._latest_context_tokens: int | None = None
        self._latest_token_rate: float | None = None
        self._last_tokens_for_rate: int | None = None
        self._last_rate_update_at: float | None = None

    # -- lifecycle --

    def start(self, message: str = "요청을 처리하고 있어요...", *, reset_metrics: bool = True) -> None:
        """Start the spinner with an initial message.

        Args:
            message: Status text to display.
            reset_metrics: If True (default), clear accumulated token metrics.
                Set to False when resuming after a temporary pause (e.g. bash streaming).
        """
        with self._lock:
            if self._active:
                return
            if reset_metrics:
                self._started_at = time.monotonic()
                self._latest_context_tokens = None
                self._latest_token_rate = None
                self._last_tokens_for_rate = None
                self._last_rate_update_at = None
            self._status = self._console.status(
                f"[bold cyan]{message}{self._format_metrics_suffix()}[/bold cyan]",
                spinner="dots",
            )
            self._status.start()
            self._active = True

    def stop(self) -> None:
        """Stop and clear the spinner."""
        with self._lock:
            if self._status and self._active:
                try:
                    self._status.stop()
                except Exception:
                    pass
            self._active = False
            self._status = None

    # -- update helpers --

    def _update_metrics_from_payload(self, payload: object) -> None:
        """Extract lightweight metrics from a node payload."""
        if not isinstance(payload, dict):
            return

        token_like = payload.get("total_tokens_used")
        if token_like is None:
            token_like = payload.get("total_tokens")

        if isinstance(token_like, int) and token_like >= 0:
            self._latest_context_tokens = token_like
            now = time.monotonic()

            # Initialize baseline first, then compute token/s from deltas.
            # Using cumulative tokens / elapsed time causes large spikes at startup.
            if self._last_tokens_for_rate is None or self._last_rate_update_at is None:
                self._last_tokens_for_rate = token_like
                self._last_rate_update_at = now
                return

            delta_tokens = token_like - self._last_tokens_for_rate
            delta_t = now - self._last_rate_update_at
            self._last_tokens_for_rate = token_like
            self._last_rate_update_at = now

            if delta_t <= 0.25 or delta_tokens < 0:
                return

            inst_rate = delta_tokens / delta_t
            # Guard unrealistic spikes from bursty state updates.
            inst_rate = max(0.0, min(inst_rate, 500.0))

            if self._latest_token_rate is None:
                self._latest_token_rate = inst_rate
            else:
                # Exponential moving average for stable UX.
                alpha = 0.25
                self._latest_token_rate = (
                    (1.0 - alpha) * self._latest_token_rate + alpha * inst_rate
                )

    def _format_metrics_suffix(self) -> str:
        """Render compact status metrics for spinner text."""
        if self._latest_context_tokens is None and self._latest_token_rate is None:
            return ""

        parts: list[str] = []
        if self._latest_context_tokens is not None:
            parts.append(f"context: {self._latest_context_tokens:,} tokens")
        if self._latest_token_rate is not None:
            parts.append(f"{self._latest_token_rate:.1f} token/s")
        return f" ({', '.join(parts)})"

    def update(self, message: str) -> None:
        """Update the spinner status text."""
        with self._lock:
            if self._status and self._active:
                text = f"{message}{self._format_metrics_suffix()}"
                self._status.update(f"[bold cyan]{text}[/bold cyan]")

    def update_for_node(self, node_name: str, payload: object | None = None) -> None:
        """Update status based on LangGraph node name."""
        self._update_metrics_from_payload(payload)
        text = NODE_STATUS_MAP.get(node_name)
        if text:
            self.update(text)

    def update_for_tool(self, tool_name: str) -> None:
        """Update status for a specific tool execution."""
        display = _TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
        self.update(f"{display} 작업을 실행해 필요한 근거를 확인하고 있어요...")

    # -- context manager --

    def __enter__(self) -> "AgentStatusIndicator":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
