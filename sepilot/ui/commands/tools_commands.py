"""Tools command handlers for Interactive Mode.

This module contains tool listing command handlers extracted from interactive.py.
"""

from typing import Any

from rich.console import Console

# Tool categories mapping based on langchain_tools.py
TOOL_CATEGORIES = {
    # File operations
    'file_read': 'File',
    'file_write': 'File',
    'file_edit': 'File',
    'notebook_edit': 'File',
    # Codebase exploration
    'codebase': 'Codebase',
    'search_content': 'Codebase',
    'find_file': 'Codebase',
    'find_definition': 'Codebase',
    'get_structure': 'Codebase',
    # Code analysis
    'code_analyze': 'Analysis',
    # Shell operations
    'bash_execute': 'Shell',
    'bash_background': 'Shell',
    'bash_output': 'Shell',
    'kill_shell': 'Shell',
    'list_shells': 'Shell',
    # Git operations
    'git': 'Git',
    # Web operations
    'web_search': 'Web',
    'web_fetch': 'Web',
    # Task management
    'plan': 'Task',
    'todo_manage': 'Task',
    # Interactive tools
    'ask_user': 'Interactive',
    # Command tools
    'slash_command': 'Command',
    'skill': 'Command',
    # Advanced
    'subagent_execute': 'SubAgent',
}

# Category display order
CATEGORY_ORDER = [
    'File', 'Codebase', 'Analysis', 'Shell', 'Git',
    'Web', 'Task', 'Interactive', 'Command', 'SubAgent', 'Other'
]


def handle_tools_command(input_text: str, agent: Any, console: Console) -> None:
    """Handle /tools command to list available LLM tools.

    Args:
        input_text: The raw input text from the user
        agent: Agent instance with langchain_tools attribute
        console: Rich console for output
    """
    # Get tools from agent
    tools: list[Any] = []
    if agent and hasattr(agent, 'langchain_tools'):
        tools = agent.langchain_tools

    if not tools:
        console.print("[yellow]No tools available[/yellow]")
        return

    # Parse subcommand
    input_text = input_text.strip()
    if input_text.lower().startswith("/tools"):
        input_text = input_text[6:].strip()
    show_list = input_text.lower() == 'list'

    # Group tools by category
    categorized: dict = {}
    for t in tools:
        name = getattr(t, 'name', str(t))
        category = TOOL_CATEGORIES.get(name, 'Other')
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(name)

    # Display tools
    total = len(tools)
    console.print(f"[bold cyan]🔧 Available LLM Tools ({total} total)[/bold cyan]")

    if show_list:
        # Compact list format
        all_tools = sorted([getattr(t, 'name', str(t)) for t in tools])
        console.print(", ".join(all_tools))
    else:
        # Grouped format
        for cat in CATEGORY_ORDER:
            if cat in categorized:
                tool_names = sorted(categorized[cat])
                console.print(f"[green]{cat}:[/green] {', '.join(tool_names)}")


__all__ = ['handle_tools_command', 'TOOL_CATEGORIES', 'CATEGORY_ORDER']
