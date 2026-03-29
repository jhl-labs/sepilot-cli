"""Command handlers for Interactive Mode.

This package contains command handler modules extracted from interactive.py
following SOLID principles for better maintainability.
"""

from sepilot.ui.commands.bench_commands import (
    handle_bench_command,
)
from sepilot.ui.commands.context_commands import (
    handle_clear_context,
    handle_compact,
    handle_context,
    handle_cost,
)
from sepilot.ui.commands.core_commands import (
    handle_clearscreen,
    handle_help,
    handle_history,
    handle_license,
    handle_reset,
    handle_status,
)
from sepilot.ui.commands.custom_commands import (
    handle_custom_commands_command,
)
from sepilot.ui.commands.graph_commands import (
    handle_graph_command,
    show_basic_graph_info,
)
from sepilot.ui.commands.mcp_commands import (
    handle_mcp_command,
)
from sepilot.ui.commands.mode_commands import (
    handle_auto_mode,
    handle_code_mode,
    handle_exec_mode,
    handle_mode_command,
    handle_plan_mode,
)
from sepilot.ui.commands.model_commands import (
    apply_model_config_to_agent,
    create_llm_from_config,
    handle_model_command,
)
from sepilot.ui.commands.performance_commands import (
    handle_performance,
)
from sepilot.ui.commands.permission_commands import (
    handle_permissions,
)
from sepilot.ui.commands.rag_commands import (
    get_rag_context,
    handle_rag_command,
)
from sepilot.ui.commands.security_commands import (
    handle_security_command,
)
from sepilot.ui.commands.session_commands import (
    handle_multiline,
    handle_new,
    handle_resume,
    handle_rewind,
    handle_session,
    handle_session_export,
    handle_session_import,
    handle_yolo,
)
from sepilot.ui.commands.skill_commands import (
    handle_skill_command,
)
from sepilot.ui.commands.stats_commands import (
    handle_stats,
)
from sepilot.ui.commands.theme_commands import (
    handle_theme,
)
from sepilot.ui.commands.tools_commands import (
    handle_tools_command,
)
from sepilot.ui.commands.undo_redo_commands import (
    get_undo_redo_manager,
    handle_redo,
    handle_undo,
)

__all__ = [
    # Core commands
    'handle_help',
    'handle_clearscreen',
    'handle_status',
    'handle_history',
    'handle_license',
    'handle_reset',
    # Session commands
    'handle_resume',
    'handle_new',
    'handle_rewind',
    'handle_multiline',
    'handle_yolo',
    'handle_session',
    'handle_session_export',
    'handle_session_import',
    # Theme commands
    'handle_theme',
    # Stats commands
    'handle_stats',
    # Permission commands
    'handle_permissions',
    # Context commands
    'handle_compact',
    'handle_clear_context',
    'handle_context',
    'handle_cost',
    # Model commands
    'handle_model_command',
    'apply_model_config_to_agent',
    'create_llm_from_config',
    # MCP commands
    'handle_mcp_command',
    # RAG commands
    'handle_rag_command',
    'get_rag_context',
    # Tools commands
    'handle_tools_command',
    # Graph commands
    'handle_graph_command',
    'show_basic_graph_info',
    # Security commands
    'handle_security_command',
    # Bench commands
    'handle_bench_command',
    # Skill commands
    'handle_skill_command',
    # Custom commands
    'handle_custom_commands_command',
    # Undo/Redo commands
    'handle_undo',
    'handle_redo',
    'get_undo_redo_manager',
    # Performance commands
    'handle_performance',
    # Mode commands
    'handle_plan_mode',
    'handle_code_mode',
    'handle_exec_mode',
    'handle_auto_mode',
    'handle_mode_command',
]
