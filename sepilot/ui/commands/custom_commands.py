"""Custom commands system (Claude Code style).

This module follows the Single Responsibility Principle (SRP) by handling
only custom command listing, creation, deletion, and management.
"""

from typing import Any

from rich.console import Console


def handle_custom_commands_command(
    console: Console,
    cmd_manager: Any,
    input_text: str,
) -> None:
    """Handle custom commands system commands.

    Args:
        console: Rich console for output
        cmd_manager: Custom command manager instance
        input_text: Original command for parsing args
    """
    # Parse input - extract arguments after "/commands"
    parts = input_text.strip().split()
    if parts and parts[0].lower().lstrip('/') == 'commands':
        parts = parts[1:]
    args = parts
    command = args[0] if args else "list"

    if command == "list" or not args:
        _show_command_list(console, cmd_manager)
        return

    if command == "reload":
        cmd_manager.reload_commands()
        count = len(cmd_manager.list_commands())
        console.print(f"[green]✅ Reloaded {count} custom commands[/green]")
        return

    if command == "create":
        _create_command(console, cmd_manager, args)
        return

    if command == "delete":
        _delete_command(console, cmd_manager, args)
        return

    if command == "help":
        _show_commands_help(console)
        return

    # Check if it's a command name to show details
    cmd = cmd_manager.get_command(command)
    if cmd:
        _show_command_details(console, cmd)
        return

    _show_commands_help(console)


def _show_commands_help(console: Console) -> None:
    """Display custom commands help."""
    console.print("[bold cyan]📜 Custom Commands System (Claude Code Style)[/bold cyan]")
    console.print()
    console.print("[bold]Usage:[/bold]")
    console.print("  /commands              - List available commands")
    console.print("  /commands list         - List with details")
    console.print("  /commands reload       - Reload commands")
    console.print("  /commands create <name> - Create new command")
    console.print("  /commands delete <name> - Delete a command")
    console.print("  /commands help         - Show this guide")
    console.print()

    # Command registration guide
    console.print("[bold yellow]━━━ Custom Command 등록 가이드 ━━━[/bold yellow]")
    console.print()
    console.print("[bold]📁 저장 위치:[/bold]")
    console.print("  [cyan]~/.sepilot/commands/[/cyan]   - 개인용 (모든 프로젝트)")
    console.print("  [cyan].sepilot/commands/[/cyan]    - 프로젝트 전용")
    console.print()
    console.print("[bold]📝 기본 예시 (review-pr.md):[/bold]")
    console.print("[dim]# PR 리뷰")
    console.print()
    console.print("PR $ARGUMENTS를 상세히 리뷰해주세요.")
    console.print("코드 품질, 보안, 성능을 검토하세요.[/dim]")
    console.print()
    console.print("[bold]📝 고급 예시 (Frontmatter 사용):[/bold]")
    console.print("[dim]---")
    console.print("description: Git commit 생성")
    console.print("allowed-tools: Bash(git add:*), Bash(git commit:*)")
    console.print("model: claude-3-5-haiku-20241022")
    console.print("argument-hint: [commit message]")
    console.print("---")
    console.print()
    console.print("메시지 '$ARGUMENTS'로 git commit을 생성하세요.")
    console.print()
    console.print("## Context")
    console.print("- Git status: !`git status`")
    console.print("- Staged changes: !`git diff --staged`[/dim]")
    console.print()
    console.print("[bold]🔤 사용 가능한 변수:[/bold]")
    console.print("  [cyan]$ARGUMENTS[/cyan] - 명령어에 전달된 인자 전체")
    console.print("  [cyan]$1, $2...[/cyan]  - 개별 인자 (첫번째, 두번째...)")
    console.print("  [cyan]$FILE[/cyan]      - 현재 파일 내용")
    console.print("  [cyan]$SELECTION[/cyan] - 현재 선택 영역")
    console.print()
    console.print("[bold]🔧 특수 문법:[/bold]")
    console.print("  [cyan]!`command`[/cyan] - Bash 명령어 실행 결과 삽입")
    console.print("  [cyan]@file.py[/cyan]   - 파일 내용 참조")
    console.print()
    console.print("[bold]⚙️ Frontmatter 옵션:[/bold]")
    console.print("  [cyan]description[/cyan]   - 명령어 설명")
    console.print("  [cyan]allowed-tools[/cyan] - 허용할 도구 목록")
    console.print("  [cyan]model[/cyan]         - 사용할 모델 지정")
    console.print("  [cyan]argument-hint[/cyan] - 인자 힌트 표시")
    console.print()
    console.print("[bold]⚡ 빠른 시작:[/bold]")
    console.print("  [cyan]mkdir -p ~/.sepilot/commands[/cyan]")
    console.print('  [cyan]echo "코드 리뷰: $ARGUMENTS" > ~/.sepilot/commands/review.md[/cyan]')
    console.print("  [cyan]/commands reload[/cyan]")
    console.print()
    console.print("[bold]💡 Tip:[/bold]")
    console.print("  • Command는 [green]/명령어[/green] 형태로 명시적 호출")
    console.print("  • Skill과 달리 자동 감지되지 않음")
    console.print("  • 팀과 공유하려면 .sepilot/commands/에 생성 후 git commit")


def _show_command_list(console: Console, cmd_manager: Any) -> None:
    """Display list of available custom commands."""
    commands = cmd_manager.list_commands()
    if not commands:
        console.print("[yellow]No custom commands available[/yellow]")
        console.print("[dim]Create commands in ~/.sepilot/commands/ or .sepilot/commands/[/dim]")
        console.print()
        console.print("[bold]Example:[/bold]")
        console.print("  Create ~/.sepilot/commands/review-pr.md with content:")
        console.print("  [dim]# Review a Pull Request[/dim]")
        console.print("  [dim]Review PR $ARGUMENTS thoroughly.[/dim]")
        return

    console.print("[bold cyan]📜 Available Custom Commands[/bold cyan]")
    console.print()

    # Separate project and user commands
    project_cmds = [c for c in commands if c.is_project]
    user_cmds = [c for c in commands if not c.is_project]

    if project_cmds:
        console.print("[bold yellow]Project Commands[/bold yellow]")
        for cmd in project_cmds:
            console.print(f"  [cyan]/{cmd.name}[/cyan] - {cmd.description}")
        console.print()

    if user_cmds:
        console.print("[bold yellow]User Commands[/bold yellow]")
        for cmd in user_cmds:
            console.print(f"  [cyan]/{cmd.name}[/cyan] - {cmd.description}")
        console.print()

    console.print("[dim]💡 Use '/<command-name> [args]' to execute[/dim]")


def _create_command(
    console: Console,
    cmd_manager: Any,
    args: list,
) -> None:
    """Create a new custom command."""
    if len(args) < 2:
        console.print("[yellow]⚠️  Usage: /commands create <name>[/yellow]")
        return

    name = args[1]
    template = f"""# {name.replace('-', ' ').title()}

Your custom command description here.

$ARGUMENTS

Add your prompt template here.
Variables available: $ARGUMENTS, $FILE, $SELECTION
"""
    filepath = cmd_manager.create_command(name, template, project=False)
    console.print(f"[green]✅ Created command: /{name}[/green]")
    console.print(f"[dim]Edit: {filepath}[/dim]")


def _delete_command(
    console: Console,
    cmd_manager: Any,
    args: list,
) -> None:
    """Delete a custom command."""
    if len(args) < 2:
        console.print("[yellow]⚠️  Usage: /commands delete <name>[/yellow]")
        return

    name = args[1]
    if cmd_manager.delete_command(name):
        console.print(f"[green]✅ Deleted command: /{name}[/green]")
    else:
        console.print(f"[red]❌ Command not found: /{name}[/red]")


def _show_command_details(console: Console, cmd: Any) -> None:
    """Show details of a specific command."""
    console.print(f"[bold cyan]Command: /{cmd.name}[/bold cyan]")
    console.print(f"[dim]Source: {cmd.source_path}[/dim]")
    console.print()
    console.print("[bold]Content:[/bold]")
    console.print(cmd.content[:500] + "..." if len(cmd.content) > 500 else cmd.content)
