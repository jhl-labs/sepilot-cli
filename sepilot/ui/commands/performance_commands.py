"""Performance command: /performance — LLM 출력 생성 속도 종합 통계."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table


def handle_performance(
    console: Console,
    agent: Any | None,
) -> None:
    """Show LLM output generation speed summary for the current session.

    Displays per-call and aggregate token throughput metrics collected by
    the AgentStatusIndicator.

    Args:
        console: Rich console for output.
        agent: Agent instance (needs .status_indicator and .cost_tracker).
    """
    indicator = getattr(agent, "status_indicator", None) if agent else None
    perf = indicator.get_performance_summary() if indicator else {}

    if not perf:
        console.print("[dim]아직 LLM 호출 기록이 없습니다.[/dim]")
        return

    model_name = "unknown"
    if agent and hasattr(agent, "settings"):
        model_name = getattr(agent.settings, "model", "unknown")

    # -- Session token breakdown from cost_tracker (accurate input/output) --
    cost_tracker = getattr(agent, "cost_tracker", None) if agent else None
    session = cost_tracker.get_session_summary() if cost_tracker else None
    session_input = session.input_tokens if session else 0
    session_output = session.output_tokens if session else 0
    session_total = session.total_tokens if session else 0
    session_cost = session.total_cost if session else 0.0
    session_requests = session.total_requests if session else 0

    # -- Build output --
    console.print()
    console.print("[bold cyan]⚡ LLM Performance Summary[/bold cyan]")
    console.print()

    # Speed table
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("지표", style="cyan")
    table.add_column("값", justify="right")

    table.add_row("모델", str(model_name))
    table.add_row("LLM 호출 횟수", f"{perf['call_count']}")
    table.add_row("총 출력 토큰", f"{perf['total_output_tokens']:,}")
    table.add_row("총 LLM 대기 시간", f"{perf['total_llm_secs']:.1f}s")
    table.add_row("", "")
    table.add_row("[bold]평균 속도[/bold]", f"[bold green]{perf['avg_rate']:.1f}[/bold green] tok/s")
    table.add_row("중앙값 (p50)", f"{perf['p50_rate']:.1f} tok/s")
    table.add_row("최소", f"{perf['min_rate']:.1f} tok/s")
    table.add_row("최대", f"{perf['max_rate']:.1f} tok/s")
    if perf.get("latest_rate") is not None:
        table.add_row("최근 (EMA)", f"{perf['latest_rate']:.1f} tok/s")

    console.print(table)

    # Session totals (if available)
    if session_total > 0:
        console.print()
        console.print("[bold cyan]📊 세션 토큰 사용량[/bold cyan]")
        console.print()

        tok_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        tok_table.add_column("항목", style="cyan")
        tok_table.add_column("값", justify="right")

        tok_table.add_row("총 요청 수", f"{session_requests}")
        tok_table.add_row("입력 토큰", f"{session_input:,}")
        tok_table.add_row("출력 토큰", f"{session_output:,}")
        tok_table.add_row("합계", f"[bold]{session_total:,}[/bold]")
        if session_cost > 0:
            tok_table.add_row("추정 비용", f"[bold green]${session_cost:.4f}[/bold green]")

        console.print(tok_table)

    console.print()
    console.print("[dim]* 속도는 출력 토큰 / LLM 호출 시간 기준 (도구 실행 시간 제외)[/dim]")
