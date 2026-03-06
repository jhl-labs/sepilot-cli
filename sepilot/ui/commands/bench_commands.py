"""Bench / SWE-Bench automation commands.

This module follows the Single Responsibility Principle (SRP) by handling
only SWE-bench dataset management, execution, and evaluation commands.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def handle_bench_command(
    console: Console,
    agent: Any | None,
    bench_agent_holder: dict,
    input_text: str,
) -> None:
    if not agent:
        console.print("[yellow]Agent not available - /bench command disabled[/yellow]")
        return

    try:
        from sepilot.agent.bench_agent import BenchAgent
    except ImportError as exc:
        console.print(f"[red]Failed to import BenchAgent: {exc}[/red]")
        return

    # Lazy initialization of bench agent
    if bench_agent_holder.get('agent') is None:
        if not hasattr(agent, "settings") or not hasattr(agent, "logger"):
            console.print("[red]Agent configuration error[/red]")
            return
        bench_agent_holder['agent'] = BenchAgent(agent.settings, agent.logger, console)

    bench_agent = bench_agent_holder['agent']

    input_text = input_text.strip()
    if input_text.lower().startswith("/bench"):
        input_text = input_text[6:].strip()

    parts = input_text.split(maxsplit=1) if input_text else []
    command = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""

    if not command or command == "help":
        _show_bench_help(console)
        return

    if command == "status":
        _show_bench_status(console, bench_agent)
        return

    if command == "instances":
        _handle_instances_command(console, bench_agent, args)
        return

    if command == "images":
        _handle_images_command(console, bench_agent, args)
        return

    if command == "enrich":
        _handle_enrich_command(console, bench_agent, args)
        return

    if command == "run":
        _handle_run_command(console, bench_agent, args)
        return

    if command == "export":
        _handle_export_command(console, bench_agent, args)
        return

    if command == "evaluate":
        _handle_evaluate_command(console, bench_agent, args)
        return

    console.print(f"[yellow]Unknown bench command: '{command}'[/yellow]")
    _show_bench_help(console)


def _show_bench_help(console: Console) -> None:
    help_text = """
[bold cyan]Bench / SWE-Bench Automation[/bold cyan]

[bold yellow]Dataset Management:[/bold yellow]
  [cyan]/bench instances load <file>[/cyan]   Load SWE-bench instance list (json/jsonl/predefined)
  [cyan]/bench instances show[/cyan]          Show loaded dataset summary

[bold yellow]Docker Images:[/bold yellow]
  [cyan]/bench images build[/cyan]            Pre-build SWE-bench Docker images

[bold yellow]Execution (Phase 1 - Inference):[/bold yellow]
  [cyan]/bench run[/cyan]                          Run agent in SWE-bench containers
  [cyan]/bench run --type verified --size 30[/cyan] Auto-load verified_30 preset
  [cyan]/bench run --type lite --size 7 --team[/cyan] 팀 모드로 lite_7 실행
  [cyan]/bench run --limit 5[/cyan]                Run first N instances
  [cyan]/bench run --workers 2[/cyan]              Run with parallel workers
  [cyan]/bench run --timeout 900[/cyan]            Set per-instance timeout (seconds)
  [cyan]/bench run --team[/cyan]                   팀 모드 활성화 (PM → Researcher → Developer → Tester)

[bold yellow]Preset Types:[/bold yellow]
  [dim]--type verified|lite|full   데이터셋 계열 (기본: verified)[/dim]
  [dim]--size 7|30|50              인스턴스 수 (기본: 7)[/dim]

[bold yellow]Evaluation (Phase 2 - Harness):[/bold yellow]
  [cyan]/bench evaluate[/cyan]                Evaluate last predictions with swebench harness
  [cyan]/bench evaluate <predictions.jsonl>[/cyan]  Evaluate specific predictions file

[bold yellow]Export:[/bold yellow]
  [cyan]/bench export <output.jsonl>[/cyan]   Export results to JSONL

[bold yellow]Status:[/bold yellow]
  [cyan]/bench status[/cyan]                  Show dataset / last run info

[bold yellow]Workflow:[/bold yellow]
  [dim]1. /bench run --type verified --size 7         (자동 로드 + Phase 1)[/dim]
  [dim]2. /bench run --type lite --size 7 --team      (팀 모드 실행)[/dim]
  [dim]3. /bench evaluate                             (Phase 2: swebench 평가)[/dim]
    """
    console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))


def _show_bench_status(console: Console, bench_agent: Any) -> None:
    status = bench_agent.get_status()
    table = Table(title="Bench Status", show_lines=True, header_style="bold cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key, value in status.items():
        table.add_row(str(key), str(value))
    console.print(table)


def _handle_instances_command(
    console: Console,
    bench_agent: Any,
    args: str,
) -> None:
    sub = args.split(maxsplit=1) if args else []
    action = sub[0].lower() if sub else "show"
    payload = sub[1] if len(sub) > 1 else ""

    if action == "load":
        if not payload:
            console.print("[yellow]Usage: /bench instances load <file|predefined>[/yellow]")
            return
        try:
            bench_agent.load_instances(payload)
        except Exception as exc:
            console.print(f"[red]Failed to load instances: {exc}[/red]")
        return

    if action == "show":
        status = bench_agent.get_status()
        console.print(f"Loaded dataset: {status['dataset']} ({status['count']} instances)")
        if status.get("source"):
            console.print(f"Source: {status['source']}")
        return

    console.print(f"[yellow]Unknown instances action: '{action}'[/yellow]")


def _handle_images_command(
    console: Console,
    bench_agent: Any,
    args: str,
) -> None:
    sub = args.split(maxsplit=1) if args else []
    action = sub[0].lower() if sub else "build"

    if action == "build":
        try:
            bench_agent.ensure_images()
        except Exception as exc:
            import traceback
            console.print(f"[red]Image build failed: {exc}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return

    console.print(f"[yellow]Unknown images action: '{action}'[/yellow]")


def _handle_enrich_command(
    console: Console,
    bench_agent: Any,
    args: str,
) -> None:
    if not args:
        console.print("[yellow]Usage: /bench enrich <full_dataset.jsonl>[/yellow]")
        return
    try:
        bench_agent.enrich_instances(args)
    except Exception as exc:
        console.print(f"[red]Failed to enrich instances: {exc}[/red]")


def _handle_run_command(
    console: Console,
    bench_agent: Any,
    args: str,
) -> None:
    limit = None
    run_tests = False
    max_workers = 1
    timeout = 600
    team_mode = False
    dataset_type = None
    dataset_size = None

    if args:
        tokens = args.split()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "--limit" and i + 1 < len(tokens):
                try:
                    limit = int(tokens[i + 1])
                    i += 1
                except ValueError:
                    console.print("[yellow]Limit must be integer[/yellow]")
                    return
            elif token == "--workers" and i + 1 < len(tokens):
                try:
                    max_workers = int(tokens[i + 1])
                    i += 1
                except ValueError:
                    console.print("[yellow]Workers must be integer[/yellow]")
                    return
            elif token == "--timeout" and i + 1 < len(tokens):
                try:
                    timeout = int(tokens[i + 1])
                    i += 1
                except ValueError:
                    console.print("[yellow]Timeout must be integer[/yellow]")
                    return
            elif token == "--type" and i + 1 < len(tokens):
                dataset_type = tokens[i + 1].lower()
                if dataset_type not in ("verified", "lite", "full"):
                    console.print("[yellow]--type must be verified, lite, or full[/yellow]")
                    return
                i += 1
            elif token == "--size" and i + 1 < len(tokens):
                try:
                    dataset_size = int(tokens[i + 1])
                    if dataset_size not in (7, 30, 50):
                        console.print("[yellow]--size must be 7, 30, or 50[/yellow]")
                        return
                    i += 1
                except ValueError:
                    console.print("[yellow]Size must be integer (7, 30, or 50)[/yellow]")
                    return
            elif token in ("--team", "--team-mode"):
                team_mode = True
            elif token == "--tests":
                run_tests = True
            i += 1

    # --type/--size 조합으로 프리셋 자동 로드
    if dataset_type or dataset_size:
        dtype = dataset_type or "verified"
        dsize = dataset_size or 7
        try:
            from sepilot.agent.bench.datasets import get_preset_config
            preset_config = get_preset_config(dtype, dsize)
            if not preset_config:
                console.print(
                    f"[red]Preset not found: {dtype}_{dsize}. "
                    f"Available: verified/lite/full × 7/30/50[/red]"
                )
                return
            bench_agent.load_instances_from_config(preset_config)
        except Exception as exc:
            console.print(f"[red]Failed to load preset: {exc}[/red]")
            return

    try:
        bench_agent.run_instances(
            limit=limit,
            run_tests=run_tests,
            max_workers=max_workers,
            timeout=timeout,
            team_mode=team_mode,
        )
    except Exception as exc:
        import traceback
        console.print(f"[red]Bench run failed: {exc}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


def _handle_export_command(
    console: Console,
    bench_agent: Any,
    args: str,
) -> None:
    if not args:
        console.print("[yellow]Usage: /bench export <output.jsonl> [--source dir][/yellow]")
        return

    tokens = args.split()
    output = tokens[0]
    source = None
    if len(tokens) == 3 and tokens[1] == "--source":
        source = tokens[2]

    try:
        bench_agent.export_jsonl(source, output)
    except Exception as exc:
        console.print(f"[red]Export failed: {exc}[/red]")


def _handle_evaluate_command(
    console: Console,
    bench_agent: Any,
    args: str,
) -> None:
    tokens = args.split() if args else []

    predictions_file = None
    max_workers = 4
    timeout = 900

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--workers" and i + 1 < len(tokens):
            try:
                max_workers = int(tokens[i + 1])
                i += 1
            except ValueError:
                pass
        elif token == "--timeout" and i + 1 < len(tokens):
            try:
                timeout = int(tokens[i + 1])
                i += 1
            except ValueError:
                pass
        elif not token.startswith("--") and predictions_file is None:
            predictions_file = token
        i += 1

    try:
        results = bench_agent.evaluate_predictions(
            predictions_file=predictions_file,
            max_workers=max_workers,
            timeout=timeout,
        )

        if results:
            resolved = sum(1 for r in results.values() if r.get("resolved"))
            total = len(results)
            console.print(
                f"\n[green]Evaluation completed![/green]\n"
                f"[cyan]Pass rate: {resolved / total * 100:.1f}% ({resolved}/{total})[/cyan]"
            )

    except Exception as exc:
        import traceback
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
