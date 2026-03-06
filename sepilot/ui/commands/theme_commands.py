"""Theme management commands: theme list, theme set.

This module follows the Single Responsibility Principle (SRP) by handling
only theme-related commands.
"""

from rich.console import Console
from rich.table import Table


def handle_theme(
    console: Console,
    input_text: str,
) -> bool:
    """Handle theme commands.

    Usage:
        /theme              - Show current theme and available themes
        /theme list         - List all available themes
        /theme set <name>   - Set theme by name
        /theme preview      - Preview all themes

    Args:
        console: Rich console for output
        input_text: Original command input for parsing args

    Returns:
        True if command was handled successfully
    """
    from sepilot.ui.themes import get_theme_manager

    args = input_text.strip().split()
    subcommand = args[1] if len(args) > 1 else "list"

    theme_manager = get_theme_manager()

    if subcommand == "list":
        return _handle_theme_list(console, theme_manager)
    elif subcommand == "set" and len(args) > 2:
        return _handle_theme_set(console, theme_manager, args[2])
    elif subcommand == "preview":
        return _handle_theme_preview(console, theme_manager)
    elif subcommand == "current":
        return _handle_theme_current(console, theme_manager)
    else:
        # Default: show current and list
        _handle_theme_current(console, theme_manager)
        console.print()
        return _handle_theme_list(console, theme_manager)


def _handle_theme_list(console: Console, theme_manager) -> bool:
    """List all available themes."""
    console.print("[bold cyan]Available Themes[/bold cyan]\n")

    themes = theme_manager.list_themes()
    current_name = theme_manager.current_theme.name

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Theme", style="cyan", width=15)
    table.add_column("Type", style="dim", width=10)
    table.add_column("Description", style="white")
    table.add_column("", width=10)

    for theme_info in themes:
        name = theme_info["name"]
        is_current = name == current_name
        current_marker = "[green]* current[/green]" if is_current else ""

        table.add_row(
            name,
            theme_info["type"],
            theme_info["description"],
            current_marker,
        )

    console.print(table)
    console.print()
    console.print("[dim]Use /theme set <name> to change theme[/dim]")

    return True


def _handle_theme_set(console: Console, theme_manager, name: str) -> bool:
    """Set the current theme."""
    if theme_manager.set_theme(name):
        theme = theme_manager.current_theme
        console.print(f"[green]Theme set to:[/green] [bold]{name}[/bold]")
        console.print(f"[dim]{theme.description}[/dim]")

        # Show a preview of colors
        colors = theme.colors
        console.print()
        console.print(f"[{colors.primary}]Primary[/{colors.primary}] "
                      f"[{colors.secondary}]Secondary[/{colors.secondary}] "
                      f"[{colors.accent}]Accent[/{colors.accent}]")
        console.print(f"[{colors.success}]Success[/{colors.success}] "
                      f"[{colors.warning}]Warning[/{colors.warning}] "
                      f"[{colors.error}]Error[/{colors.error}]")

        return True
    else:
        console.print(f"[red]Theme not found:[/red] {name}")
        console.print("[dim]Use /theme list to see available themes[/dim]")
        return False


def _handle_theme_current(console: Console, theme_manager) -> bool:
    """Show current theme info."""
    theme = theme_manager.current_theme
    console.print(f"[bold cyan]Current Theme:[/bold cyan] {theme.name}")
    console.print(f"[dim]{theme.description}[/dim]")
    return True


def _handle_theme_preview(console: Console, theme_manager) -> bool:
    """Preview all themes with sample colors."""
    console.print("[bold cyan]Theme Preview[/bold cyan]\n")

    themes = theme_manager.list_themes()

    for theme_info in themes:
        name = theme_info["name"]
        theme = theme_manager.get_theme(name)
        if not theme:
            continue

        colors = theme.colors
        console.print(f"[bold]{name}[/bold] - {theme.description}")
        console.print(f"  [{colors.primary}]Primary[/{colors.primary}] "
                      f"[{colors.secondary}]Secondary[/{colors.secondary}] "
                      f"[{colors.accent}]Accent[/{colors.accent}] "
                      f"[{colors.success}]Success[/{colors.success}] "
                      f"[{colors.warning}]Warning[/{colors.warning}] "
                      f"[{colors.error}]Error[/{colors.error}]")
        console.print()

    return True
