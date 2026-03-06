"""DevOps command handlers for Interactive Mode.

This module contains handlers for /container, /helm, /se, and /gitops commands.
Extracted from interactive.py for better maintainability.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel


def handle_container_command(input_text: str, agent: Any, console: Console) -> None:
    """Handle /container command for Docker operations.

    Args:
        input_text: Full input text including /container prefix
        agent: ReactAgent instance
        console: Rich console for output
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /container command disabled[/yellow]")
        return

    try:
        from sepilot.agent.container_agent import ContainerAgent

        if not hasattr(agent, 'settings') or not hasattr(agent, 'logger'):
            console.print("[red]❌ Agent configuration error[/red]")
            return

        container_agent = ContainerAgent(
            settings=agent.settings,
            logger=agent.logger,
            console=console
        )

        # Remove /container prefix
        _input = input_text.strip()
        if _input.lower().startswith('/container'):
            _input = _input[10:].strip()

        parts = _input.split(maxsplit=1) if _input else []
        command = parts[0].lower() if len(parts) > 0 else ""
        args = parts[1] if len(parts) > 1 else ""

        def show_help():
            help_text = """
[bold cyan]🐳 Container & Docker Commands with AI Assistance[/bold cyan]

[bold yellow]📋 Image Management:[/bold yellow]
  [cyan]/container images[/cyan]              List all Docker images
  [cyan]/container build <tag> [context] [dockerfile][/cyan]
                                 Build Docker image from Dockerfile

[bold yellow]📦 Container Management:[/bold yellow]
  [cyan]/container ps[/cyan]                  List all containers (running + stopped)
  [cyan]/container logs <container> [args][/cyan]  View container logs
  [cyan]/container inspect <target>[/cyan]    Inspect container or image details

[bold yellow]🤖 AI Features:[/bold yellow]
  [cyan]/container generate[/cyan]            Generate optimized Dockerfile for current project
  [cyan]/container diagnose[/cyan]            AI-powered Docker environment diagnostics

[bold yellow]🌐 Registry Management:[/bold yellow]
  [cyan]/container registry list[/cyan]       List configured container registries
  [cyan]/container registry add[/cyan]        Add container registry

[dim]Type /container <command> for help[/dim]
            """
            console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))

        if not command or command == "help":
            show_help()
            return

        elif command == "images":
            success = container_agent.run_images(args)
            if not success:
                console.print("\n[yellow]⚠️  Failed to list images[/yellow]")

        elif command == "ps":
            success = container_agent.run_ps(args)
            if not success:
                console.print("\n[yellow]⚠️  Failed to list containers[/yellow]")

        elif command == "logs":
            if not args:
                console.print("[yellow]⚠️  Please specify a container name[/yellow]")
                console.print("[dim]Example: /container logs nginx-1[/dim]")
                return
            parts = args.split(maxsplit=1)
            container = parts[0]
            remaining_args = parts[1] if len(parts) > 1 else ""
            success = container_agent.run_logs(container, remaining_args)
            if not success:
                console.print("\n[yellow]⚠️  Failed to get logs[/yellow]")

        elif command == "inspect":
            if not args:
                console.print("[yellow]⚠️  Please specify a container or image[/yellow]")
                return
            success = container_agent.run_inspect(args)
            if not success:
                console.print("\n[yellow]⚠️  Failed to inspect target[/yellow]")

        elif command == "build":
            if not args:
                console.print("[yellow]⚠️  Please specify an image tag[/yellow]")
                return
            parts = args.split()
            tag = parts[0]
            context = parts[1] if len(parts) > 1 else "."
            dockerfile = parts[2] if len(parts) > 2 else "Dockerfile"
            success = container_agent.run_build(tag, context, dockerfile)
            if not success:
                console.print("\n[yellow]⚠️  Build failed[/yellow]")

        elif command == "generate":
            project_path = args if args else "."
            success = container_agent.run_generate_dockerfile(project_path)
            if not success:
                console.print("\n[yellow]⚠️  Dockerfile generation failed[/yellow]")

        elif command == "diagnose":
            target = args if args else ""
            success = container_agent.run_ai_diagnose(target)
            if not success:
                console.print("\n[yellow]⚠️  Diagnostics failed[/yellow]")

        elif command == "registry":
            _handle_registry_subcommand(args, container_agent, console, show_help)

        else:
            console.print(f"[yellow]⚠️  Unknown container command: '{command}'[/yellow]\n")
            show_help()

    except ImportError as e:
        console.print(f"[red]❌ Failed to import ContainerAgent: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Container command failed: {e}[/red]")
        import traceback
        if console.is_terminal:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def _handle_registry_subcommand(args: str, container_agent: Any, console: Console, show_help) -> None:
    """Handle registry subcommands for /container registry."""
    if not args:
        show_help()
        return

    parts = args.split(maxsplit=1)
    subcommand = parts[0].lower()
    parts[1] if len(parts) > 1 else ""

    if subcommand == "list":
        registries = container_agent.config.list_registries()
        if not registries:
            console.print("[yellow]No container registries configured[/yellow]")
            console.print("[dim]Use '/container registry add' to add one[/dim]")
            return

        from rich.table import Table
        table = Table(title="🌐 Container Registries", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="yellow", width=20)
        table.add_column("URL", style="cyan", width=40)
        table.add_column("Username", style="green", width=20)
        table.add_column("Added", style="dim", width=20)

        for reg in registries:
            table.add_row(
                reg.get("name", ""),
                reg.get("url", ""),
                reg.get("username", ""),
                reg.get("added_at", "")[:10] if reg.get("added_at") else ""
            )
        console.print(table)

    elif subcommand == "add":
        console.print(Panel(
            "🌐 Add Container Registry\n\n"
            "Configure access to Harbor, Docker Hub, or private registry",
            title="Container Agent",
            border_style="bold cyan"
        ))

        try:
            from prompt_toolkit import prompt as pt_prompt
            prompt_fn = pt_prompt
        except Exception:
            prompt_fn = input

        name = prompt_fn("Registry name (e.g., 'harbor', 'dockerhub'): ").strip()
        if not name:
            console.print("[yellow]Registry name is required[/yellow]")
            return

        url = prompt_fn("Registry URL (e.g., 'https://harbor.example.com', 'docker.io'): ").strip()
        if not url:
            console.print("[yellow]Registry URL is required[/yellow]")
            return

        username = prompt_fn("Username (optional): ").strip()
        password = prompt_fn("Password (optional): ").strip()

        success = container_agent.config.add_registry(name, url, username, password)
        if success:
            console.print(f"\n[bold green]✅ Registry '{name}' added successfully![/bold green]")
            if password:
                console.print("[yellow]⚠️  Warning: Password is stored in plain text[/yellow]")
        else:
            console.print("\n[yellow]⚠️  Failed to add registry[/yellow]")
    else:
        console.print(f"[yellow]⚠️  Unknown registry command: '{subcommand}'[/yellow]")
        show_help()


def handle_helm_command(input_text: str, agent: Any, console: Console) -> None:
    """Handle /helm command for Helm chart operations.

    Args:
        input_text: Full input text including /helm prefix
        agent: ReactAgent instance
        console: Rich console for output
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /helm command disabled[/yellow]")
        return

    try:
        from sepilot.agent.helm_agent import HelmAgent

        if not hasattr(agent, 'settings') or not hasattr(agent, 'logger'):
            console.print("[red]❌ Agent configuration error[/red]")
            return

        helm_agent = HelmAgent(
            settings=agent.settings,
            logger=agent.logger,
            console=console
        )

        _input = input_text.strip()
        if _input.lower().startswith('/helm'):
            _input = _input[5:].strip()

        parts = _input.split(maxsplit=1) if _input else []
        command = parts[0].lower() if len(parts) > 0 else ""
        args = parts[1] if len(parts) > 1 else ""

        def show_help():
            help_text = """
[bold cyan]⎈ Helm Chart Generation with AI ReAct Workflow[/bold cyan]

[bold yellow]🤖 AI Chart Generation:[/bold yellow]
  [cyan]/helm generate[/cyan]               Generate Helm Chart for current project
  [cyan]/helm generate /path/to/project[/cyan]  Generate Chart for specific path

[bold yellow]🔍 Validation:[/bold yellow]
  [cyan]/helm lint <chart>[/cyan]            Validate Helm Chart with helm lint

[bold yellow]🚀 Deployment:[/bold yellow]
  [cyan]/helm install <name> <chart>[/cyan]  Install Chart to Kubernetes

[dim]Type /helm <command> for help[/dim]
            """
            console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))

        if not command or command == "help":
            show_help()
            return

        elif command == "generate":
            project_path = args if args else "."
            chart_name = None
            try:
                from prompt_toolkit import prompt as pt_prompt
                prompt_fn = pt_prompt
            except Exception:
                prompt_fn = input

            console.print("\n[bold]Chart Configuration:[/bold]")
            name_input = prompt_fn("Chart name (press Enter for auto-detect): ").strip()
            if name_input:
                chart_name = name_input

            success = helm_agent.run_generate(project_path, chart_name)
            if not success:
                console.print("\n[yellow]⚠️  Helm Chart generation failed[/yellow]")

        elif command == "lint":
            if not args:
                console.print("[yellow]⚠️  Please specify a chart path[/yellow]")
                return
            success = helm_agent.run_lint(args)
            if not success:
                console.print("\n[yellow]⚠️  Lint failed[/yellow]")

        elif command == "install":
            if not args:
                console.print("[yellow]⚠️  Please specify release name and chart path[/yellow]")
                return
            parts = args.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]⚠️  Both release name and chart path are required[/yellow]")
                return

            release_name = parts[0]
            chart_path = parts[1]

            try:
                from prompt_toolkit import prompt as pt_prompt
                prompt_fn = pt_prompt
            except Exception:
                prompt_fn = input

            namespace = prompt_fn("Kubernetes namespace (default: default): ").strip() or "default"
            success = helm_agent.run_install(release_name, chart_path, namespace)
            if not success:
                console.print("\n[yellow]⚠️  Installation failed[/yellow]")

        else:
            console.print(f"[yellow]⚠️  Unknown helm command: '{command}'[/yellow]\n")
            show_help()

    except ImportError as e:
        console.print(f"[red]❌ Failed to import HelmAgent: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Helm command failed: {e}[/red]")
        import traceback
        if console.is_terminal:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def handle_se_command(input_text: str, agent: Any, console: Console) -> None:
    """Handle /se command for software engineering operations.

    Args:
        input_text: Full input text including /se prefix
        agent: ReactAgent instance
        console: Rich console for output
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /se command disabled[/yellow]")
        return

    try:
        from sepilot.agent.se_agent import SEAgent

        if not hasattr(agent, 'settings') or not hasattr(agent, 'logger'):
            console.print("[red]❌ Agent configuration error[/red]")
            return

        se_agent = SEAgent(
            settings=agent.settings,
            logger=agent.logger,
            console=console
        )

        _input = input_text.strip()
        if _input.lower().startswith('/se'):
            _input = _input[3:].strip()

        parts = _input.split(maxsplit=1) if _input else []
        command = parts[0].lower() if len(parts) > 0 else ""
        args = parts[1] if len(parts) > 1 else ""

        def show_help():
            help_text = """
[bold cyan]🏗️  Software Engineering Commands with AI Assistance[/bold cyan]

[bold yellow]🏛️  Architecture & Quality:[/bold yellow]
  [cyan]/se architecture[/cyan]             Review project architecture with AI
  [cyan]/se quick[/cyan]                    Quick code quality check

[bold yellow]🧠 LangGraph ReAct:[/bold yellow]
  [cyan]/se diagnose[/cyan]                 Diagnose architecture/test/static-analysis risks
  [cyan]/se improve[/cyan]                  Generate improvement roadmap

[bold yellow]🔍 Static Analysis:[/bold yellow]
  [cyan]/se static[/cyan]                   Run all static analysis tools
  [cyan]/se static ruff|mypy|pylint|bandit[/cyan]  Run specific tool

[bold yellow]🛠️  Auto-Fix:[/bold yellow]
  [cyan]/se fix[/cyan]                      Auto-fix issues (default: ruff)
  [cyan]/se fix ruff|autopep8|black[/cyan]  Auto-fix with specific tool

[dim]Type /se <command> for help[/dim]
            """
            console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))

        if not command or command == "help":
            show_help()
            return

        elif command == "diagnose":
            success = se_agent.run_react_session("diagnose", args)
            if not success:
                console.print("\n[yellow]⚠️  Diagnose session failed[/yellow]")

        elif command == "improve":
            success = se_agent.run_react_session("improve", args)
            if not success:
                console.print("\n[yellow]⚠️  Improvement session failed[/yellow]")

        elif command == "architecture":
            success = se_agent.run_architecture_review()
            if not success:
                console.print("\n[yellow]⚠️  Architecture review failed[/yellow]")

        elif command == "quick":
            success = se_agent.run_quick_check()
            if not success:
                console.print("\n[yellow]⚠️  Quick check failed[/yellow]")

        elif command == "static":
            if args:
                report = se_agent._run_static_analysis_tools(tool_name=args.strip())
                console.print(report)
            else:
                result = se_agent.run_static_analysis()
                if isinstance(result, dict) and result.get("error"):
                    console.print(f"[red]❌ Static analysis failed: {result['error']}[/red]")

        elif command == "fix":
            tool_name = args if args else "ruff"
            success = se_agent.run_auto_fix(tool_name)
            if not success:
                console.print("\n[yellow]⚠️  Auto-fix failed[/yellow]")

        else:
            console.print(f"[yellow]⚠️  Unknown se command: '{command}'[/yellow]\n")
            show_help()

    except ImportError as e:
        console.print(f"[red]❌ Failed to import SEAgent: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ SE command failed: {e}[/red]")
        import traceback
        if console.is_terminal:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def handle_gitops_command(input_text: str, agent: Any, console: Console) -> None:
    """Handle /gitops command for GitOps/ArgoCD operations.

    Args:
        input_text: Full input text including /gitops prefix
        agent: ReactAgent instance
        console: Rich console for output
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /gitops command disabled[/yellow]")
        return

    try:
        from sepilot.agent.gitops_agent import GitOpsAgent

        if not hasattr(agent, 'settings') or not hasattr(agent, 'logger'):
            console.print("[red]❌ Agent configuration error[/red]")
            return

        gitops_agent = GitOpsAgent(
            settings=agent.settings,
            logger=agent.logger,
            console=console
        )

        _input = input_text.strip()
        if _input.lower().startswith('/gitops'):
            _input = _input[7:].strip()

        parts = _input.split(maxsplit=1) if _input else []
        command = parts[0].lower() if len(parts) > 0 else ""
        args = parts[1] if len(parts) > 1 else ""

        def show_help():
            help_text = """
[bold cyan]🚀 GitOps Monitoring & ArgoCD Traceability Commands[/bold cyan]

[bold yellow]📋 Application Management:[/bold yellow]
  [cyan]/gitops list[/cyan]                   List all ArgoCD Applications
  [cyan]/gitops get <app>[/cyan]              Get detailed application info

[bold yellow]🔍 Monitoring & Analysis:[/bold yellow]
  [cyan]/gitops status[/cyan]                 Show sync status summary
  [cyan]/gitops trace <app>[/cyan]            Trace Git commit to K8s resources

[bold yellow]🧠 AI-Powered Diagnostics:[/bold yellow]
  [cyan]/gitops diagnose[/cyan]               Diagnose sync failures across all apps
  [cyan]/gitops diagnose <app>[/cyan]         Diagnose specific application

[dim]Type /gitops <command> for help[/dim]
            """
            console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))

        if not command or command == "help":
            show_help()
            return

        elif command == "list":
            success = gitops_agent.run_list()
            if not success:
                console.print("\n[yellow]⚠️  Failed to list applications[/yellow]")

        elif command == "get":
            if not args:
                console.print("[yellow]⚠️  Please specify application name[/yellow]")
                return
            success = gitops_agent.run_get(args.strip())
            if not success:
                console.print("\n[yellow]⚠️  Failed to get application info[/yellow]")

        elif command == "status":
            success = gitops_agent.run_sync_status()
            if not success:
                console.print("\n[yellow]⚠️  Failed to get sync status[/yellow]")

        elif command == "trace":
            if not args:
                console.print("[yellow]⚠️  Please specify application name[/yellow]")
                return
            success = gitops_agent.run_trace(args.strip())
            if not success:
                console.print("\n[yellow]⚠️  Failed to trace application[/yellow]")

        elif command == "diagnose":
            app_name = args.strip() if args else ""
            success = gitops_agent.run_diagnose(app_name)
            if not success:
                console.print("\n[yellow]⚠️  Diagnosis failed[/yellow]")

        else:
            console.print(f"[yellow]⚠️  Unknown gitops command: '{command}'[/yellow]\n")
            show_help()

    except ImportError as e:
        console.print(f"[red]❌ Failed to import GitOpsAgent: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ GitOps command failed: {e}[/red]")
        import traceback
        if console.is_terminal:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


__all__ = [
    'handle_container_command',
    'handle_helm_command',
    'handle_se_command',
    'handle_gitops_command',
]
