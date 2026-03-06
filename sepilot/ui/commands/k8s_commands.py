"""Kubernetes command handlers for Interactive Mode.

This module contains handlers for /k8s-health and related Kubernetes commands.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel


def handle_k8s_health_command(input_text: str, agent: Any, console: Console) -> None:
    """Handle /k8s-health command for Kubernetes cluster health monitoring.

    Args:
        input_text: Full input text including /k8s-health prefix
        agent: ReactAgent instance
        console: Rich console for output
    """
    if not agent:
        console.print("[yellow]Agent not available - /k8s-health command disabled[/yellow]")
        return

    try:
        from sepilot.agent.k8s_agent import K8sAgent

        if not hasattr(agent, "settings") or not hasattr(agent, "logger"):
            console.print("[red]Agent configuration error[/red]")
            return

        k8s_agent = K8sAgent(
            settings=agent.settings,
            logger=agent.logger,
            console=console,
        )

        # Remove /k8s-health prefix
        _input = input_text.strip()
        if _input.lower().startswith("/k8s-health"):
            _input = _input[11:].strip()

        parts = _input.split(maxsplit=1) if _input else []
        command = parts[0].lower() if len(parts) > 0 else ""
        args = parts[1] if len(parts) > 1 else ""

        def show_help():
            help_text = """
[bold cyan]Kubernetes Cluster Health Monitor[/bold cyan]

[bold yellow]Health Checks:[/bold yellow]
  [cyan]/k8s-health[/cyan]                   Full cluster health analysis with AI recommendations
  [cyan]/k8s-health -n <namespace>[/cyan]    Check specific namespace only
  [cyan]/k8s-health --verbose[/cyan]         Show detailed output including all events

[bold yellow]Quick Checks:[/bold yellow]
  [cyan]/k8s-health nodes[/cyan]             Quick node status check
  [cyan]/k8s-health pods[/cyan]              Show unhealthy pods
  [cyan]/k8s-health pods -n <ns>[/cyan]      Show unhealthy pods in namespace
  [cyan]/k8s-health events[/cyan]            Show recent warning/error events
  [cyan]/k8s-health resources[/cyan]         Show resource usage (requires metrics-server)

[bold yellow]Options:[/bold yellow]
  [dim]-n, --namespace <ns>[/dim]    Target specific namespace
  [dim]--verbose, -v[/dim]           Show detailed output
  [dim]--no-ai[/dim]                 Skip AI analysis

[bold yellow]Features:[/bold yellow]
  - Node health monitoring (Ready/NotReady conditions)
  - Pod crash detection (CrashLoopBackOff, high restart counts)
  - Service endpoint verification
  - Event log analysis (warnings and errors)
  - AI-powered diagnosis and recommendations

[dim]Example: /k8s-health -n production --verbose[/dim]
            """
            console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))

        # Parse options
        namespace = ""
        verbose = False
        include_ai = True

        # Parse args for options
        remaining_args = []
        i = 0
        arg_parts = args.split() if args else []

        while i < len(arg_parts):
            arg = arg_parts[i]
            if arg in ("-n", "--namespace") and i + 1 < len(arg_parts):
                namespace = arg_parts[i + 1]
                i += 2
            elif arg in ("-v", "--verbose"):
                verbose = True
                i += 1
            elif arg == "--no-ai":
                include_ai = False
                i += 1
            else:
                remaining_args.append(arg)
                i += 1

        # Handle command or default to full health check
        if command == "help":
            show_help()
            return

        elif command == "nodes":
            k8s_agent.run_node_check()

        elif command == "pods":
            k8s_agent.run_pod_check(namespace)

        elif command == "events":
            k8s_agent.run_events(namespace)

        elif command == "resources":
            k8s_agent.run_resources(namespace)

        elif not command or command.startswith("-"):
            # Default: Full health check
            # Re-parse if command starts with -
            if command.startswith("-"):
                all_args = _input
                arg_parts = all_args.split()
                i = 0
                while i < len(arg_parts):
                    arg = arg_parts[i]
                    if arg in ("-n", "--namespace") and i + 1 < len(arg_parts):
                        namespace = arg_parts[i + 1]
                        i += 2
                    elif arg in ("-v", "--verbose"):
                        verbose = True
                        i += 1
                    elif arg == "--no-ai":
                        include_ai = False
                        i += 1
                    else:
                        i += 1

            k8s_agent.run_health_check(
                namespace=namespace,
                include_ai=include_ai,
                verbose=verbose,
            )

        else:
            console.print(f"[yellow]Unknown subcommand: '{command}'[/yellow]\n")
            show_help()

    except ImportError as e:
        console.print(f"[red]Failed to import K8sAgent: {e}[/red]")
        console.print("[dim]Make sure all dependencies are installed[/dim]")
    except Exception as e:
        console.print(f"[red]Command failed: {e}[/red]")
        import traceback

        if agent and hasattr(agent, "settings") and agent.settings.verbose:
            traceback.print_exc()
