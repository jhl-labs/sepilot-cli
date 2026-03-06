"""Skills system commands (Claude Code style).

This module follows the Single Responsibility Principle (SRP) by handling
only skill listing, execution, and management commands.
"""

from typing import Any

from rich.console import Console

from sepilot.skills import get_skill_manager


def handle_skill_command(
    console: Console,
    agent: Any | None,
    conversation_context: list,
    skill_context_holder: dict,
    input_text: str,
) -> None:
    """Handle skill system commands.

    Args:
        console: Rich console for output
        agent: Agent reference for skill context
        conversation_context: Shared conversation history
        skill_context_holder: Dict holding skill context list for prompt injection
        input_text: Original command for parsing args
    """
    # Parse input - extract arguments after "/skill"
    parts = input_text.strip().split()
    if parts and parts[0].lower().lstrip('/') == 'skill':
        parts = parts[1:]
    args = parts
    command = args[0] if args else "list"
    skill_name = args[0] if args and args[0] not in ("list", "reload", "help") else None

    skill_manager = get_skill_manager()

    if command == "list" or not args:
        _show_skill_list(console, skill_manager)
        return

    if command == "reload":
        skill_manager.reload_skills()
        count = len(skill_manager.list_skills())
        console.print(f"[green]✅ Reloaded {count} skills[/green]")
        return

    if command == "help":
        _show_skill_help(console, skill_manager)
        return

    # Execute a specific skill
    if skill_name:
        _execute_skill(
            console,
            skill_manager,
            skill_name,
            args,
            agent,
            conversation_context,
            skill_context_holder,
        )
        return

    _show_skill_help(console, skill_manager)


def _show_skill_help(console: Console, skill_manager: Any) -> None:
    """Display skill command help."""
    console.print("[bold cyan]📚 Skills System (Claude Code Style)[/bold cyan]")
    console.print()
    console.print("[bold]Usage:[/bold]")
    console.print("  /skill              - List available skills")
    console.print("  /skill list         - List skills with details")
    console.print("  /skill <name>       - Execute a skill")
    console.print("  /skill reload       - Reload all skills")
    console.print("  /skill help         - Show this guide")
    console.print()

    # Skill registration guide
    console.print("[bold yellow]━━━ Custom Skill 등록 가이드 ━━━[/bold yellow]")
    console.print()
    console.print("[bold]📁 저장 위치:[/bold]")
    console.print("  [cyan]~/.sepilot/skills/[/cyan]    - 개인용 (모든 프로젝트)")
    console.print("  [cyan].sepilot/skills/[/cyan]     - 프로젝트 전용")
    console.print()
    console.print("[bold]📂 폴더 구조:[/bold]")
    console.print("  my-skill/")
    console.print("  ├── [green]SKILL.md[/green]      (필수)")
    console.print("  ├── examples.md   (선택)")
    console.print("  └── templates/    (선택)")
    console.print()
    console.print("[bold]📝 SKILL.md 예시:[/bold]")
    console.print("[dim]---")
    console.print("name: my-skill")
    console.print("description: 이 스킬의 설명 (자동 감지 트리거)")
    console.print("category: general")
    console.print("triggers:")
    console.print("  - keyword1")
    console.print("  - keyword2")
    console.print("---")
    console.print()
    console.print("# My Skill")
    console.print()
    console.print("## 지침")
    console.print("Claude가 따를 단계별 지침을 작성하세요.")
    console.print()
    console.print("## 예시")
    console.print("구체적인 사용 예시를 보여주세요.[/dim]")
    console.print()
    console.print("[bold]⚡ 빠른 시작:[/bold]")
    console.print("  [cyan]mkdir -p ~/.sepilot/skills/my-skill[/cyan]")
    console.print("  [cyan]touch ~/.sepilot/skills/my-skill/SKILL.md[/cyan]")
    console.print()
    console.print("[bold]💡 Tip:[/bold]")
    console.print("  • Skill은 키워드 기반으로 [green]자동 감지[/green]됩니다")
    console.print("  • Slash command와 달리 명시적 호출 없이도 작동")
    console.print("  • triggers에 키워드를 등록하면 자동 활성화")
    console.print()

    console.print("[bold]Available Skills:[/bold]")
    skills = skill_manager.list_skills()
    if skills:
        for meta in skills:
            console.print(f"  [cyan]{meta.name}[/cyan] - {meta.description}")
    else:
        console.print("  [dim]등록된 skill이 없습니다[/dim]")


def _show_skill_list(console: Console, skill_manager: Any) -> None:
    """Display list of available skills."""
    skills = skill_manager.list_skills()
    if not skills:
        console.print("[yellow]No skills available[/yellow]")
        console.print("[dim]Create skills in ~/.sepilot/skills/ or .sepilot/skills/[/dim]")
        return

    console.print("[bold cyan]📚 Available Skills[/bold cyan]")
    console.print()

    # Group by category
    by_category: dict[str, list] = {}
    for meta in skills:
        cat = meta.category or "general"
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(meta)

    for category, category_skills in sorted(by_category.items()):
        console.print(f"[bold yellow]{category.title()}[/bold yellow]")
        for meta in category_skills:
            triggers = ", ".join(meta.triggers[:3]) if meta.triggers else ""
            trigger_hint = f" [dim](triggers: {triggers})[/dim]" if triggers else ""
            console.print(f"  [cyan]{meta.name}[/cyan] - {meta.description}{trigger_hint}")
        console.print()

    console.print("[dim]💡 Use '/skill <name>' to execute a skill[/dim]")


def _execute_skill(
    console: Console,
    skill_manager: Any,
    skill_name: str,
    args: list,
    agent: Any | None,
    conversation_context: list,
    skill_context_holder: dict,
) -> None:
    """Execute a specific skill.

    Args:
        console: Rich console for output
        skill_manager: Skill manager instance
        skill_name: Name of skill to execute
        args: Command arguments
        agent: Agent reference
        conversation_context: Shared conversation history
        skill_context_holder: Dict for storing prompt injections
    """
    skill = skill_manager.get_skill(skill_name)
    if not skill:
        console.print(f"[red]❌ Skill '{skill_name}' not found[/red]")
        console.print()
        console.print("[bold]Available skills:[/bold]")
        for meta in skill_manager.list_skills():
            console.print(f"  [cyan]{meta.name}[/cyan]")
        return

    # Execute the skill
    context = {
        "agent": agent,
        "console": console,
        "conversation": conversation_context
    }

    remaining_args = " ".join(args[1:]) if len(args) > 1 else ""
    result = skill_manager.execute_skill(skill_name, remaining_args, context)

    if result.success:
        meta = skill.get_metadata()
        console.print(f"[green]✅ Skill '{meta.name}' activated[/green]")

        if result.prompt_injection:
            # Store the prompt injection for the next interaction
            if 'contexts' not in skill_context_holder:
                skill_context_holder['contexts'] = []
            skill_context_holder['contexts'].append(result.prompt_injection)
            console.print("[dim]📋 Skill context will be applied to your next message[/dim]")

        if result.message:
            console.print(f"[dim]{result.message}[/dim]")
    else:
        console.print(f"[red]❌ Skill execution failed: {result.message}[/red]")
