"""Permission management commands: permissions list, add, remove.

This module follows the Single Responsibility Principle (SRP) by handling
only permission-related commands.
"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from sepilot.ui.input_utils import INPUT_CANCELLED, prompt_text


def handle_permissions(
    console: Console,
    input_text: str,
    session: Any | None = None,
) -> bool:
    """Handle permission commands.

    Usage:
        /permissions              - Show current permission rules
        /permissions list         - List all permission rules
        /permissions add          - Add a new permission rule (interactive)
        /permissions remove <id>  - Remove a permission rule by index
        /permissions reset        - Reset to default rules
        /permissions test <tool> <cmd> - Test if a command would be allowed

    Args:
        console: Rich console for output
        input_text: Original command input for parsing args

    Returns:
        True if command was handled successfully
    """
    from sepilot.agent.permission_rules import get_permission_manager

    args = input_text.strip().split()
    subcommand = args[1] if len(args) > 1 else "list"

    permission_manager = get_permission_manager()

    if subcommand == "list":
        return _handle_permissions_list(console, permission_manager)
    elif subcommand == "add":
        return _handle_permissions_add(console, permission_manager, args[2:], session=session)
    elif subcommand == "remove" and len(args) > 2:
        return _handle_permissions_remove(console, permission_manager, args[2])
    elif subcommand == "reset":
        return _handle_permissions_reset(console, permission_manager)
    elif subcommand == "test" and len(args) > 3:
        return _handle_permissions_test(console, permission_manager, args[2], " ".join(args[3:]))
    elif subcommand == "file":
        return _handle_permissions_file(console, permission_manager)
    else:
        # Default: show rules
        return _handle_permissions_list(console, permission_manager)


def _handle_permissions_list(console: Console, permission_manager) -> bool:
    """List all permission rules."""
    console.print("[bold cyan]Permission Rules[/bold cyan]\n")

    rules = permission_manager.rule_set.rules

    # Separate user rules from built-in dangerous patterns
    user_rules = [r for r in rules if r.priority < 1000]
    builtin_rules = [r for r in rules if r.priority >= 1000]

    # User rules
    if user_rules:
        console.print("[bold]Custom Rules:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Tool", style="cyan", width=15)
        table.add_column("Pattern", style="white", width=30)
        table.add_column("Permission", width=10)
        table.add_column("Description", style="dim")

        for i, rule in enumerate(user_rules):
            perm_style = {
                "deny": "red",
                "ask": "yellow",
                "allow": "green",
            }.get(rule.permission.value, "white")

            table.add_row(
                str(i),
                rule.tool,
                rule.pattern or "*",
                f"[{perm_style}]{rule.permission.value.upper()}[/{perm_style}]",
                rule.description,
            )

        console.print(table)
    else:
        console.print("[dim]No custom rules defined[/dim]")

    # Built-in dangerous patterns summary
    console.print()
    console.print(f"[bold]Built-in Safety Rules:[/bold] {len(builtin_rules)} patterns")
    console.print("[dim]These patterns are always blocked for safety:[/dim]")

    # Show a few examples
    examples = builtin_rules[:5]
    for rule in examples:
        console.print(f"  [red]DENY[/red] {rule.pattern}")
    if len(builtin_rules) > 5:
        console.print(f"  [dim]... and {len(builtin_rules) - 5} more[/dim]")

    # Default sensitive tools
    console.print()
    console.print("[bold]Default Sensitive Tools (require approval):[/bold]")
    sensitive = permission_manager.rule_set.DEFAULT_SENSITIVE_TOOLS
    console.print(f"  [yellow]{', '.join(sensitive)}[/yellow]")

    # Show config file location
    console.print()
    console.print(f"[dim]Config file: {permission_manager.config_path}[/dim]")
    console.print("[dim]Use /permissions add to add rules interactively[/dim]")

    return True


def _handle_permissions_add(
    console: Console,
    permission_manager,
    args: list[str],
    session: Any | None = None,
) -> bool:
    """Add a new permission rule."""
    from sepilot.agent.permission_rules import PermissionLevel, PermissionRule

    # Interactive mode if no args
    if not args or len(args) < 3:
        console.print("[bold cyan]Add Permission Rule[/bold cyan]\n")
        tool = prompt_text(
            "Tool (*, bash_execute, file_write, file_edit, file_read, git): ",
            session=session,
        )
        if tool == INPUT_CANCELLED:
            console.print("[dim]Cancelled[/dim]")
            return False

        permission_str = prompt_text(
            "Permission (allow/ask/deny) [ask]: ",
            session=session,
            default="ask",
        )
        if permission_str == INPUT_CANCELLED:
            console.print("[dim]Cancelled[/dim]")
            return False

        pattern = prompt_text(
            "Pattern [*]: ",
            session=session,
            default="*",
        )
        if pattern == INPUT_CANCELLED:
            console.print("[dim]Cancelled[/dim]")
            return False

        description = prompt_text(
            "Description (optional): ",
            session=session,
            default="",
        )
        if description == INPUT_CANCELLED:
            console.print("[dim]Cancelled[/dim]")
            return False

        args = [tool or "*", permission_str or "ask", pattern or "*", description]

    tool = args[0]
    permission_str = args[1].lower()
    pattern = args[2] if len(args) > 2 else "*"
    description = " ".join(args[3:]) if len(args) > 3 else ""

    # Validate permission
    try:
        permission = PermissionLevel(permission_str)
    except ValueError:
        console.print(f"[red]Invalid permission: {permission_str}[/red]")
        console.print("[dim]Valid values: allow, ask, deny[/dim]")
        return False

    # Create and add rule
    rule = PermissionRule(
        tool=tool,
        pattern=pattern,
        permission=permission,
        description=description,
        priority=10,  # User rules have moderate priority
    )

    permission_manager.rule_set.add_rule(rule)

    # Save to config file
    _save_user_rules(permission_manager)

    console.print(f"[green]Added rule:[/green] {tool} {permission_str.upper()} '{pattern}'")

    return True


def _handle_permissions_remove(console: Console, permission_manager, index_str: str) -> bool:
    """Remove a permission rule by index."""
    try:
        index = int(index_str)
    except ValueError:
        console.print(f"[red]Invalid index: {index_str}[/red]")
        return False

    user_rules = [r for r in permission_manager.rule_set.rules if r.priority < 1000]

    if index < 0 or index >= len(user_rules):
        console.print(f"[red]Index out of range: {index}[/red]")
        console.print(f"[dim]Valid range: 0-{len(user_rules) - 1}[/dim]")
        return False

    rule = user_rules[index]
    permission_manager.rule_set.rules.remove(rule)

    # Save to config file
    _save_user_rules(permission_manager)

    console.print(f"[green]Removed rule:[/green] {rule.tool} {rule.permission.value.upper()} '{rule.pattern}'")

    return True


def _handle_permissions_reset(console: Console, permission_manager) -> bool:
    """Reset to default rules."""
    # Remove all user rules (keep only built-in)
    permission_manager.rule_set.rules = [
        r for r in permission_manager.rule_set.rules if r.priority >= 1000
    ]

    # Remove config file
    if permission_manager.config_path.exists():
        permission_manager.config_path.unlink()
        console.print(f"[green]Removed config file:[/green] {permission_manager.config_path}")

    console.print("[green]Permission rules reset to defaults[/green]")

    return True


def _handle_permissions_test(console: Console, permission_manager, tool: str, command: str) -> bool:
    """Test if a command would be allowed."""
    permission, reason = permission_manager.check_permission(tool, {"command": command})

    console.print(f"[bold cyan]Permission Test[/bold cyan]\n")
    console.print(f"Tool: {tool}")
    console.print(f"Command: {command}")
    console.print()

    perm_style = {
        "deny": "red",
        "ask": "yellow",
        "allow": "green",
    }.get(permission.value, "white")

    console.print(f"Result: [{perm_style}]{permission.value.upper()}[/{perm_style}]")
    if reason:
        console.print(f"Reason: {reason}")

    return True


def _handle_permissions_file(console: Console, permission_manager) -> bool:
    """Show config file info."""
    console.print("[bold cyan]Permission Configuration Files[/bold cyan]\n")

    console.print(f"[bold]User config:[/bold] {permission_manager.config_path}")
    if permission_manager.config_path.exists():
        console.print("  [green]exists[/green]")
    else:
        console.print("  [dim]not created yet[/dim]")

    project_config = permission_manager.project_path / ".sepilot" / "permissions.json"
    console.print(f"\n[bold]Project config:[/bold] {project_config}")
    if project_config.exists():
        console.print("  [green]exists[/green]")
    else:
        console.print("  [dim]not created[/dim]")

    console.print()
    console.print("[bold]Config file format:[/bold]")
    console.print("""```json
{
  "rules": [
    {"tool": "bash_execute", "pattern": "npm install *", "permission": "allow"},
    {"tool": "bash_execute", "pattern": "git push*", "permission": "ask"},
    {"tool": "file_write", "pattern": "/etc/*", "permission": "deny"}
  ]
}
```""")

    return True


def _save_user_rules(permission_manager) -> None:
    """Save user rules to config file."""
    import json

    user_rules = [r for r in permission_manager.rule_set.rules if r.priority < 1000]

    data = {
        "rules": [
            {
                "tool": r.tool,
                "pattern": r.pattern,
                "permission": r.permission.value,
                "description": r.description,
            }
            for r in user_rules
        ]
    }

    permission_manager.config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(permission_manager.config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
