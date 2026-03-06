"""SEPilot Custom Commands System

Custom commands are user-defined slash commands stored as markdown files.
Similar to Claude Code's SlashCommand tool.
"""

from .manager import CommandManager, CustomCommand, get_command_manager

__all__ = [
    "CommandManager",
    "get_command_manager",
    "CustomCommand",
]
