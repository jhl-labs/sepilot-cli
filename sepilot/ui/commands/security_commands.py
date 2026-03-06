"""Security / DevSecOps commands.

This module follows the Single Responsibility Principle (SRP) by handling
only security scanning, remediation, and baseline management commands.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel


def handle_security_command(
    console: Console,
    agent: Any | None,
    input_text: str,
) -> None:
    """Handle security / DevSecOps commands.

    Args:
        console: Rich console for output
        agent: Agent with settings and logger
        input_text: Original command for parsing args
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /security command disabled[/yellow]")
        return

    try:
        from sepilot.agent.security_agent import SecurityAgent
    except ImportError as exc:
        console.print(f"[red]❌ Failed to import SecurityAgent: {exc}[/red]")
        return

    if not hasattr(agent, "settings") or not hasattr(agent, "logger"):
        console.print("[red]❌ Agent configuration error[/red]")
        return

    security_agent = SecurityAgent(
        settings=agent.settings,
        logger=agent.logger,
        console=console,
    )

    input_text = input_text.strip()
    if input_text.lower().startswith("/security"):
        input_text = input_text[9:].strip()

    parts = input_text.split(maxsplit=1) if input_text else []
    command = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""

    if not command or command == "help":
        _show_security_help(console)
        return

    if command in ("scan", "check", "audit"):
        findings = security_agent.run_scan()
        if not findings:
            console.print("\n[yellow]⚠️  No security findings (or tools missing)[/yellow]")
        return

    if command == "baseline":
        _handle_baseline_command(console, security_agent, args)
        return

    if command in ("ai-fix", "fix", "remediate", "patch"):
        success = security_agent.run_ai_fix(args)
        if success:
            console.print("\n[bold green]✅ Security remediation flow completed[/bold green]")
        else:
            console.print("\n[yellow]⚠️  Security remediation aborted or failed[/yellow]")
        return

    console.print(f"[yellow]⚠️  Unknown security command: '{command}'[/yellow]")
    _show_security_help(console)


def _show_security_help(console: Console) -> None:
    """Display security command help."""
    help_text = """
[bold cyan]🛡️  Security / DevSecOps Commands[/bold cyan]

[bold yellow]🔍 Scans:[/bold yellow]
  [cyan]/security scan[/cyan]                Run Bandit / pip-audit / detect-secrets if available
  [cyan]/security scan --detail[/cyan]       Same as scan (placeholder for future options)

[bold yellow]🛠️  AI Remediation:[/bold yellow]
  [cyan]/security ai-fix[/cyan]              Generate DevSecOps remediation plan + optional patches
  [cyan]/security ai-fix <issue>[/cyan]      Focus on a specific vulnerability description

[bold yellow]📦 Baseline Management:[/bold yellow]
  [cyan]/security baseline save[/cyan]       Run scan & persist results for future comparisons
  [cyan]/security baseline show[/cyan]       Display stored baseline summary
  [cyan]/security baseline diff[/cyan]       Compare latest scan against baseline

[bold yellow]👥 Human-in-the-loop:[/bold yellow]
  • AI patches are only applied after explicit confirmation
  • Plans highlight secret management, supply-chain, dependency issues, etc.

[dim]Type /security <command> (scan, ai-fix) to get started[/dim]
    """
    console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))


def _handle_baseline_command(
    console: Console,
    security_agent: Any,
    args: str,
) -> None:
    """Handle baseline sub-commands.

    Args:
        console: Rich console for output
        security_agent: SecurityAgent instance
        args: Remaining arguments after 'baseline'
    """
    sub = args.split(maxsplit=1) if args else []
    action = sub[0].lower() if sub else "show"

    if action == "save":
        security_agent.save_baseline()
        return
    if action == "show":
        security_agent.show_baseline()
        return
    if action in ("diff", "compare"):
        security_agent.diff_baseline()
        return

    console.print(f"[yellow]⚠️  Unknown baseline action: '{action}'[/yellow]")
    console.print("[dim]Use one of: save, show, diff[/dim]")
