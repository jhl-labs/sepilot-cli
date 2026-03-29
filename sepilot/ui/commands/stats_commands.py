"""Statistics commands: stats for cost and usage tracking.

This module follows the Single Responsibility Principle (SRP) by handling
only statistics-related commands.
"""

from typing import Any

from rich.console import Console
from rich.table import Table


def handle_stats(
    console: Console,
    input_text: str,
    agent: Any | None = None,
) -> bool:
    """Handle statistics commands.

    Usage:
        /stats              - Show current session and monthly stats
        /stats session      - Show current session stats only
        /stats monthly      - Show monthly statistics
        /stats all          - Show all-time statistics
        /stats model        - Show stats by model

    Args:
        console: Rich console for output
        input_text: Original command input for parsing args
        agent: Optional agent for additional context

    Returns:
        True if command was handled successfully
    """
    from sepilot.monitoring.cost_tracker import get_cost_tracker

    args = input_text.strip().split()
    subcommand = args[1] if len(args) > 1 else "overview"

    cost_tracker = get_cost_tracker()

    if subcommand == "session":
        return _handle_session_stats(console, cost_tracker)
    elif subcommand == "monthly":
        return _handle_monthly_stats(console, cost_tracker, args[2] if len(args) > 2 else None)
    elif subcommand == "all":
        return _handle_all_time_stats(console, cost_tracker)
    elif subcommand == "model":
        return _handle_model_stats(console, cost_tracker)
    else:
        # Default: show overview (session + current month)
        _handle_session_stats(console, cost_tracker)
        console.print()
        return _handle_monthly_stats(console, cost_tracker, None)


def _handle_session_stats(console: Console, cost_tracker) -> bool:
    """Show current session statistics."""
    session = cost_tracker.get_session_summary()

    console.print("[bold cyan]Current Session Statistics[/bold cyan]\n")

    if not session:
        console.print("[yellow]No active cost tracking session[/yellow]")
        console.print("[dim]Cost tracking starts when you run your first command[/dim]")
        return True

    # Session info
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("Session ID", session.session_id[:12] + "...")
    table.add_row("Model", session.model)
    table.add_row("Started", session.start_time[:19])
    table.add_row("Requests", f"{session.total_requests:,}")
    table.add_row("Input Tokens", f"{session.input_tokens:,}")
    table.add_row("Output Tokens", f"{session.output_tokens:,}")
    table.add_row("Total Tokens", f"{session.total_tokens:,}")
    table.add_row("[bold]Total Cost[/bold]", f"[bold green]${session.total_cost:.4f}[/bold green]")

    console.print(table)

    return True


def _handle_monthly_stats(console: Console, cost_tracker, month: str | None) -> bool:
    """Show monthly statistics."""
    summary = cost_tracker.get_monthly_summary(month)

    if month:
        console.print(f"[bold cyan]Statistics for {month}[/bold cyan]\n")
    else:
        from datetime import datetime
        current_month = datetime.now().strftime("%Y-%m")
        console.print(f"[bold cyan]Statistics for {current_month} (Current Month)[/bold cyan]\n")

    if summary["total_requests"] == 0:
        console.print("[yellow]No usage recorded for this period[/yellow]")
        return True

    # Summary table
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("Total Requests", f"{summary['total_requests']:,}")
    table.add_row("Input Tokens", f"{summary['input_tokens']:,}")
    table.add_row("Output Tokens", f"{summary['output_tokens']:,}")
    table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
    table.add_row("[bold]Total Cost[/bold]", f"[bold green]${summary['total_cost']:.4f}[/bold green]")

    console.print(table)

    # Model breakdown
    if summary.get("by_model"):
        console.print("\n[bold]By Model:[/bold]")
        model_table = Table(show_header=True, header_style="bold")
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Requests", justify="right")
        model_table.add_column("Tokens", justify="right")
        model_table.add_column("Cost", justify="right", style="green")

        for model, stats in sorted(
            summary["by_model"].items(),
            key=lambda x: x[1]["cost"],
            reverse=True
        ):
            model_table.add_row(
                model,
                f"{stats['requests']:,}",
                f"{stats['tokens']:,}",
                f"${stats['cost']:.4f}"
            )

        console.print(model_table)

    return True


def _handle_all_time_stats(console: Console, cost_tracker) -> bool:
    """Show all-time statistics."""
    summary = cost_tracker.get_all_time_summary()

    console.print("[bold cyan]All-Time Statistics[/bold cyan]\n")

    if summary["total_requests"] == 0:
        console.print("[yellow]No usage recorded yet[/yellow]")
        return True

    # Summary
    formatted = cost_tracker.format_summary(summary)
    console.print(formatted)

    # Monthly breakdown
    if summary.get("by_month"):
        console.print("\n[bold]By Month:[/bold]")
        month_table = Table(show_header=True, header_style="bold")
        month_table.add_column("Month", style="cyan")
        month_table.add_column("Requests", justify="right")
        month_table.add_column("Tokens", justify="right")
        month_table.add_column("Cost", justify="right", style="green")

        for month, stats in sorted(summary["by_month"].items(), reverse=True):
            month_table.add_row(
                month,
                f"{stats['requests']:,}",
                f"{stats['tokens']:,}",
                f"${stats['cost']:.4f}"
            )

        console.print(month_table)

    return True


def _handle_model_stats(console: Console, cost_tracker) -> bool:
    """Show statistics grouped by model."""
    summary = cost_tracker.get_all_time_summary()

    console.print("[bold cyan]Statistics by Model[/bold cyan]\n")

    if not summary.get("by_model"):
        console.print("[yellow]No model usage recorded yet[/yellow]")
        return True

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Model", style="cyan", width=30)
    table.add_column("Requests", justify="right", width=12)
    table.add_column("Tokens", justify="right", width=15)
    table.add_column("Cost", justify="right", width=12, style="green")
    table.add_column("Avg/Request", justify="right", width=12)

    for model, stats in sorted(
        summary["by_model"].items(),
        key=lambda x: x[1]["cost"],
        reverse=True
    ):
        avg_tokens = stats["tokens"] // stats["requests"] if stats["requests"] > 0 else 0
        table.add_row(
            model,
            f"{stats['requests']:,}",
            f"{stats['tokens']:,}",
            f"${stats['cost']:.4f}",
            f"{avg_tokens:,} tok"
        )

    console.print(table)

    # Total row
    console.print()
    console.print(f"[bold]Total: {summary['total_requests']:,} requests, "
                  f"{summary['total_tokens']:,} tokens, "
                  f"${summary['total_cost']:.4f}[/bold]")

    return True
