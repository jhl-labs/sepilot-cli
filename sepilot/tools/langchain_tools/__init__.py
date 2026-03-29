"""LangChain-compatible tool implementations.

This package provides all tools for the LangChain-based agent.
Tools are organized by category for better maintainability.

Usage:
    from sepilot.tools.langchain_tools import get_all_tools
    tools = get_all_tools()

Or import specific tools:
    from sepilot.tools.langchain_tools import file_read, bash_execute
"""

# Code analysis
from sepilot.tools.langchain_tools.analysis_tools import code_analyze

# Bash execution
from sepilot.tools.langchain_tools.bash_tools import bash_execute

# Codebase exploration
from sepilot.tools.langchain_tools.codebase_tools import (
    codebase,
    find_definition,
    find_file,
    get_structure,
    search_content,
)

# File operations
from sepilot.tools.langchain_tools.file_tools import (
    apply_patch,
    file_edit,
    file_read,
    file_write,
)

# Git operations
from sepilot.tools.langchain_tools.git_tools import git

# Interactive tools
from sepilot.tools.langchain_tools.interactive_tools import (
    ask_user,
    skill,
    slash_command,
)

# Multimedia tools (image/PDF reading)
from sepilot.tools.langchain_tools.multimedia_tools import (
    image_read,
    multimedia_info,
    pdf_read,
)

# Notebook editing
from sepilot.tools.langchain_tools.notebook_tools import notebook_edit

# Plan mode tools
from sepilot.tools.langchain_tools.plan_mode_tools import (
    add_plan_step,
    enter_plan_mode,
    exit_plan_mode,
    get_plan_status,
    update_plan,
)

# Background shell operations
from sepilot.tools.langchain_tools.shell_tools import (
    bash_background,
    bash_output,
    kill_shell,
    list_shells,
)

# SubAgent execution
from sepilot.tools.langchain_tools.subagent_tools import subagent_execute

# Task management
from sepilot.tools.langchain_tools.task_tools import (
    plan,
    todo_manage,
)

# Think / Scratchpad
from sepilot.tools.langchain_tools.think_tools import think

# tmux agent session management
from sepilot.tools.langchain_tools.tmux_tools import (
    tmux_create_session,
    tmux_destroy,
    tmux_orchestrate,
    tmux_read,
    tmux_send,
    tmux_status,
)

# Web operations
from sepilot.tools.langchain_tools.web_tools import (
    web_fetch,
    web_search,
)


def get_all_tools() -> list:
    """Get all available tools for LangChain agent.

    Returns:
        List of all tool functions
    """
    tools = [
        # File operations
        file_read,
        file_write,
        file_edit,
        notebook_edit,

        # Multimedia (image/PDF)
        image_read,
        pdf_read,
        multimedia_info,

        # Codebase exploration
        codebase,
        search_content,
        find_file,
        find_definition,
        get_structure,

        # Code analysis
        code_analyze,

        # Shell operations
        bash_execute,
        bash_background,
        bash_output,
        kill_shell,
        list_shells,

        # Git operations
        git,

        # Web operations
        web_search,
        web_fetch,

        # Plan mode
        enter_plan_mode,
        exit_plan_mode,
        update_plan,
        add_plan_step,
        get_plan_status,

        # Task management
        plan,
        todo_manage,

        # Interactive tools
        ask_user,

        # Command tools
        slash_command,
        skill,

        # Advanced
        subagent_execute,

        # tmux agent sessions
        tmux_create_session,
        tmux_send,
        tmux_read,
        tmux_status,
        tmux_destroy,
        tmux_orchestrate,

        # Think / Scratchpad
        think,
    ]

    # Add apply_patch if available
    if apply_patch is not None:
        tools.append(apply_patch)

    return tools


__all__ = [
    # File operations
    'file_read',
    'file_write',
    'file_edit',
    'apply_patch',
    'notebook_edit',

    # Multimedia (image/PDF)
    'image_read',
    'pdf_read',
    'multimedia_info',

    # Codebase exploration
    'codebase',
    'search_content',
    'find_file',
    'find_definition',
    'get_structure',

    # Code analysis
    'code_analyze',

    # Shell operations
    'bash_execute',
    'bash_background',
    'bash_output',
    'kill_shell',
    'list_shells',

    # Git operations
    'git',

    # Web operations
    'web_search',
    'web_fetch',

    # Plan mode
    'enter_plan_mode',
    'exit_plan_mode',
    'update_plan',
    'add_plan_step',
    'get_plan_status',

    # Task management
    'plan',
    'todo_manage',

    # Interactive tools
    'ask_user',
    'slash_command',
    'skill',

    # Advanced
    'subagent_execute',

    # tmux agent sessions
    'tmux_create_session',
    'tmux_send',
    'tmux_read',
    'tmux_status',
    'tmux_destroy',
    'tmux_orchestrate',

    # Think / Scratchpad
    'think',

    # Registry
    'get_all_tools',
]
