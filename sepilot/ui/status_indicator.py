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

# LangGraph node name → (short label, user-friendly status text)
NODE_STATUS_MAP: dict[str, tuple[str, str]] = {
    "triage": ("분석", "요청을 분석해 실행 방향과 우선순위를 정하고 있어요..."),
    "agent": ("실행", "요청 해결을 위해 핵심 작업을 순서대로 진행하고 있어요..."),
    "orchestrator": ("조율", "현재 단계에 맞게 작업 순서를 조정하고 있어요..."),
    "hierarchical_planner": ("계획", "작업 단계를 나누고 우선순위를 정리하고 있어요..."),
    "planner": ("계획", "작업 단계를 나누고 우선순위를 정리하고 있어요..."),
    "tools": ("도구", "필요한 근거를 수집하기 위해 도구를 실행하고 있어요..."),
    "verifier": ("검증", "결과가 요청과 맞는지 검증하고 있어요..."),
    "reporter": ("정리", "확인된 결과를 최종 답변으로 정리하고 있어요..."),
    "iteration_guard": ("반복점검", "같은 작업 반복 여부를 점검하고 있어요..."),
    "context_manager": ("문맥정리", "문맥을 정리해 다음 단계 정확도를 높이고 있어요..."),
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
        # Cumulative performance metrics (survive across start/stop cycles)
        self._perf_total_output_tokens: int = 0
        self._perf_total_llm_secs: float = 0.0
        self._perf_call_count: int = 0
        self._perf_rates: list[float] = []  # per-call rates for min/max/p50

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
        """Extract context token count from a node payload."""
        if not isinstance(payload, dict):
            return

        token_like = payload.get("total_tokens_used")
        if token_like is None:
            token_like = payload.get("total_tokens")

        if isinstance(token_like, int) and token_like >= 0:
            self._latest_context_tokens = token_like

    def update_token_rate(self, output_tokens: int | float, llm_elapsed_secs: float) -> None:
        """Update output generation speed from measured LLM call metrics.

        Called directly by agent nodes after each LLM invocation,
        using only output tokens and LLM wall-clock time (excluding tool execution).

        Args:
            output_tokens: Number of output tokens generated.
            llm_elapsed_secs: Wall-clock seconds spent in LLM invocation only.
        """
        if llm_elapsed_secs <= 0.1 or output_tokens <= 0:
            return

        inst_rate = output_tokens / llm_elapsed_secs
        inst_rate = max(0.0, min(inst_rate, 1000.0))

        # Accumulate for /performance summary
        self._perf_total_output_tokens += int(output_tokens)
        self._perf_total_llm_secs += llm_elapsed_secs
        self._perf_call_count += 1
        self._perf_rates.append(inst_rate)

        if self._latest_token_rate is None:
            self._latest_token_rate = inst_rate
        else:
            alpha = 0.3
            self._latest_token_rate = (
                (1.0 - alpha) * self._latest_token_rate + alpha * inst_rate
            )

    def _format_metrics_suffix(self) -> str:
        """Render compact status metrics for spinner text."""
        if self._latest_context_tokens is None and self._latest_token_rate is None:
            return ""

        parts: list[str] = []
        if self._latest_context_tokens is not None:
            parts.append(f"ctx {self._latest_context_tokens:,}")
        if self._latest_token_rate is not None:
            parts.append(f"out {self._latest_token_rate:.1f} tok/s")
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
        entry = NODE_STATUS_MAP.get(node_name)
        if entry:
            label, text = entry
            self.update(f"[{label}] {text}")

    def update_for_tool(self, tool_name: str) -> None:
        """Update status for a specific tool execution."""
        display = _TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
        self.update(f"{display} 작업을 실행해 필요한 근거를 확인하고 있어요...")

    def get_performance_summary(self) -> dict:
        """Return accumulated performance metrics for /performance command.

        Returns:
            Dict with keys: call_count, total_output_tokens, total_llm_secs,
            avg_rate, min_rate, max_rate, p50_rate, latest_rate.
            Returns empty dict if no data.
        """
        if not self._perf_rates:
            return {}

        sorted_rates = sorted(self._perf_rates)
        mid = len(sorted_rates) // 2
        p50 = (
            sorted_rates[mid]
            if len(sorted_rates) % 2 == 1
            else (sorted_rates[mid - 1] + sorted_rates[mid]) / 2
        )

        return {
            "call_count": self._perf_call_count,
            "total_output_tokens": self._perf_total_output_tokens,
            "total_llm_secs": self._perf_total_llm_secs,
            "avg_rate": self._perf_total_output_tokens / self._perf_total_llm_secs,
            "min_rate": sorted_rates[0],
            "max_rate": sorted_rates[-1],
            "p50_rate": p50,
            "latest_rate": self._latest_token_rate,
        }

    def reset_performance(self) -> None:
        """Reset accumulated performance metrics."""
        self._perf_total_output_tokens = 0
        self._perf_total_llm_secs = 0.0
        self._perf_call_count = 0
        self._perf_rates.clear()

    # -- context manager --

    def __enter__(self) -> AgentStatusIndicator:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
