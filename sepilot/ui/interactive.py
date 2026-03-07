"""Interactive REPL mode for SE Pilot

Provides a rich interactive command-line interface with:
- Command history and auto-completion
- Multi-line input support
- Special commands (/help, /exit, /clearscreen, /compact, /clear, etc.)
- Session persistence
- Context management with auto-compaction

This module has been refactored following SOLID principles:
- Key bindings: sepilot.ui.key_bindings
- @ reference expansion: sepilot.ui.reference_expander
- Context display/auto-compact: sepilot.ui.context_display
- Memory management: sepilot.ui.memory_manager
- Output overlays: sepilot.ui.output_overlays
- Commands: sepilot.ui.commands.*
"""

import locale
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application.current import get_app_or_none
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    Completer = object  # Dummy base class when prompt_toolkit not available
    get_app_or_none = None

from rich.console import Console
from rich.markdown import Markdown

from sepilot.agent.enhanced_state import AgentMode, state_to_summary
from sepilot.commands import get_command_manager
from sepilot.config.model_profile import ModelProfileManager
from sepilot.skills import get_skill_manager
from sepilot.ui.commands.bench_commands import handle_bench_command

# Command handlers (extracted for modularity)
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
)
from sepilot.ui.commands.custom_commands import handle_custom_commands_command
from sepilot.ui.commands.devops_commands import (
    handle_container_command,
    handle_gitops_command,
    handle_helm_command,
    handle_se_command,
)
from sepilot.ui.commands.graph_commands import handle_graph_command
from sepilot.ui.commands.k8s_commands import handle_k8s_health_command
from sepilot.ui.commands.mcp_commands import handle_mcp_command
from sepilot.ui.commands.model_commands import (
    apply_model_config_to_agent,
    create_llm_from_config,
    handle_model_command,
)
from sepilot.ui.commands.rag_commands import get_rag_context, handle_rag_command
from sepilot.ui.commands.security_commands import handle_security_command
from sepilot.ui.commands.skill_commands import handle_skill_command
from sepilot.ui.commands.tools_commands import handle_tools_command
from sepilot.ui.commands.theme_commands import handle_theme
from sepilot.ui.commands.stats_commands import handle_stats
from sepilot.ui.commands.permission_commands import handle_permissions
from sepilot.ui.commands.session_commands import (
    handle_new,
    handle_resume,
    handle_rewind,
    handle_session,
)
from sepilot.ui.commands.mode_commands import (
    handle_auto_mode,
    handle_code_mode,
    handle_exec_mode,
    handle_mode_command,
    handle_plan_mode,
)
from sepilot.ui.commands.undo_redo_commands import (
    get_undo_redo_manager,
    handle_redo,
    handle_undo,
)

# SOLID-refactored UI components
from sepilot.ui.completer import FileReferenceCompleter
from sepilot.ui.context_display import ContextDisplayManager
from sepilot.ui.key_bindings import create_key_bindings
from sepilot.ui.memory_manager import MemoryManager
from sepilot.ui.output_overlays import OutputOverlayManager, TeeOutput
from sepilot.ui.reference_expander import ReferenceExpander
from sepilot.ui.setup_wizard import needs_setup, run_setup_wizard
from sepilot.ui.themes import get_theme_manager, set_theme
from sepilot.utils.text import sanitize_text


def _get_version_info() -> str:
    """Get version info based on dev/production mode.

    Dev mode (running from git repo): returns version + commit hash
    Production mode (installed package): returns version string
    """
    from sepilot import __version__
    version = f"v{__version__}"

    # Check if running from git repository (dev mode)
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        git_dir = os.path.join(project_root, ".git")

        if os.path.isdir(git_dir):
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
                return f"{version}-dev ({commit_hash})"
    except Exception:
        pass

    return version


def _build_shell_command(command: str) -> list[str]:
    """Build a platform-safe shell invocation command."""
    if os.name == "nt":
        return ["cmd", "/c", command]

    shell_path = os.environ.get("SHELL")
    if shell_path and os.path.exists(shell_path):
        return [shell_path, "-lc", command]
    if os.path.exists("/bin/bash"):
        return ["/bin/bash", "-lc", command]
    return ["/bin/sh", "-c", command]


class InteractiveMode:
    """Interactive REPL mode for SE Pilot"""

    def __init__(
        self,
        execute_callback: Callable[[str], str | None],
        console: Console | None = None,
        history_file: str | None = None,
        memory_file: str | None = None,
        manual_observer: Callable[[str, str], None] | None = None,
        agent: object | None = None,
        conversation_context: list | None = None,
        agent_factory: Callable | None = None
    ):
        """
        Initialize interactive mode

        Args:
            execute_callback: Function to call when user submits a command
            console: Rich console for output
            history_file: Path to command history file
            agent: Reference to the agent for advanced control (e.g., auto-approve toggle)
            conversation_context: Shared conversation history for all agents
        """
        self.execute_callback = execute_callback
        self.console = console or Console()
        self.running = False
        self.manual_observer = manual_observer
        self.agent = agent
        self.agent_factory = agent_factory
        self.conversation_context = conversation_context if conversation_context is not None else []

        # Session state
        self.session_start = datetime.now()
        self.command_count = 0
        self.total_tokens = 0  # Total token usage for this session

        # Setup command history
        if history_file is None:
            history_dir = os.path.join(os.path.expanduser("~"), ".sepilot")
            os.makedirs(history_dir, exist_ok=True)
            history_file = os.path.join(history_dir, "history.txt")

        self.history_file = history_file
        self._ensure_history_file_utf8()

        # SOLID: Memory management (Single Responsibility)
        if memory_file is None:
            history_dir = os.path.dirname(self.history_file)
            memory_file = os.path.join(history_dir, "memory.json")
        self.memory_file = memory_file
        self._memory_manager = MemoryManager(memory_file=memory_file, console=self.console)

        # SOLID: Output overlay management (Single Responsibility)
        self._overlay_manager = OutputOverlayManager(console=self.console)

        # SOLID: Reference expansion (Single Responsibility)
        self._reference_expander = ReferenceExpander(console=self.console)

        # SOLID: Context display management (Single Responsibility)
        self._context_display_manager = ContextDisplayManager(
            console=self.console,
            agent=self.agent,
            conversation_context=self.conversation_context,
        )

        # Initialize prompt session if prompt_toolkit is available
        if HAS_PROMPT_TOOLKIT:
            # Create file reference completer
            self.file_completer = FileReferenceCompleter(working_dir=os.getcwd())

            # Create custom style for completion menu
            completion_style = Style.from_dict({
                'completion-menu': 'bg:#333333 #ffffff',
                'completion-menu.completion': 'bg:#333333 #ffffff',
                'completion-menu.completion.current': 'bg:#00aaaa #000000',
                'completion-menu.meta.completion': 'bg:#444444 #ffffff',
                'completion-menu.meta.completion.current': 'bg:#00aaaa #000000',
            })

            # Check if we should enable mouse support (disabled by default for SSH)
            enable_mouse = os.getenv('SEPILOT_MOUSE', '0') == '1'
            if not enable_mouse and os.getenv('SSH_CONNECTION'):
                enable_mouse = False

            # SOLID: Key bindings (Single Responsibility)
            key_bindings = create_key_bindings(
                on_show_result=self._show_last_result,
                on_show_logs=self._show_execution_logs,
                on_rewind=self._cmd_rewind,
                on_command_palette=self._show_command_palette,
            )

            self.session: PromptSession | None = PromptSession(
                history=FileHistory(self.history_file),
                auto_suggest=AutoSuggestFromHistory(),
                enable_history_search=True,
                multiline=False,
                key_bindings=key_bindings,
                completer=self.file_completer,
                complete_while_typing=True,
                complete_in_thread=False,
                mouse_support=enable_mouse,
                style=completion_style,
                refresh_interval=0.1,
            )
        else:
            self.session = None
            self.file_completer = None

        # Model profile manager & agents
        self.model_profile_manager = ModelProfileManager()
        self.bench_agent = None

        # Core commands
        self.core_commands = {
            '/help': self._cmd_help,
            '/exit': self._cmd_exit,
            '/quit': self._cmd_exit,
            '/clearscreen': self._cmd_clearscreen,  # Clear screen (renamed from /clear)
            '/cls': self._cmd_clearscreen,  # Alias for clear screen
            '/clear': self._cmd_clearcontext,  # Clear conversation context (Claude Code style)
            '/history': self._cmd_history,
            '/status': self._cmd_status,
            '/license': self._cmd_license,
            '/resume': self._cmd_resume,
            '/new': self._cmd_new,
            '/rewind': self._cmd_rewind,
            '/undo': self._cmd_undo,  # Undo last exchange (OpenCode style)
            '/redo': self._cmd_redo,  # Redo undone exchange (OpenCode style)
            '/multiline': self._cmd_multiline,
            '/yolo': self._cmd_yolo,
            '/graph': self._cmd_graph,
            '/reset': self._cmd_reset,
            '/model': self._cmd_model,
            '/mcp': self._cmd_mcp,
            '/rag': self._cmd_rag,
            '/compact': self._cmd_compact,  # Compact conversation context
            '/tools': self._cmd_tools,  # List all available LLM tools
            '/context': self._cmd_context,  # Show context usage visualization
            '/cost': self._cmd_cost,  # Show cost estimate
            '/skill': self._cmd_skill,  # Skills system (Claude Code style)
            '/skills': self._cmd_skill,  # Alias for /skill
            '/commands': self._cmd_commands,  # Custom commands (Claude Code style)
            '/k8s-health': self._cmd_k8s_health,  # Kubernetes health checks
            '/theme': self._cmd_theme,  # Theme management
            '/stats': self._cmd_stats,  # Usage statistics and cost tracking
            '/permissions': self._cmd_permissions,  # Permission rules management
            '/session': self._cmd_session,  # Session export/import
            # Agent mode commands (PLAN/CODE/EXEC)
            '/plan': self._cmd_plan_mode,
            '/code': self._cmd_code_mode,
            '/exec': self._cmd_exec_mode,
            '/auto': self._cmd_auto_mode,
            '/mode': self._cmd_mode,
        }

        # Special commands
        self.special_commands = self.core_commands.copy()

        # Custom command manager (for dynamic slash commands)
        self._custom_command_manager = get_command_manager()

        # MCP configuration manager (lazy initialization)
        self._mcp_config_manager = None

    def _show_last_result(self) -> None:
        """Display the last command or agent output (delegates to overlay manager)."""
        self._overlay_manager.show_last_result()

    def _show_execution_logs(self) -> None:
        """Show captured execution logs (delegates to overlay manager)."""
        self._overlay_manager.show_execution_logs()

    def _show_command_palette(self) -> None:
        """Show command palette for quick command search (OpenCode style Ctrl+K)."""
        from sepilot.ui.command_palette import create_default_palette

        # Create palette with command handlers
        command_handlers = {
            "/help": lambda: self._handle_input("/help"),
            "/exit": lambda: self._handle_input("/exit"),
            "/status": lambda: self._handle_input("/status"),
            "/history": lambda: self._handle_input("/history"),
            "/new": lambda: self._handle_input("/new"),
            "/resume": lambda: self._handle_input("/resume"),
            "/rewind": lambda: self._handle_input("/rewind"),
            "/undo": lambda: self._handle_input("/undo"),
            "/redo": lambda: self._handle_input("/redo"),
            "/context": lambda: self._handle_input("/context"),
            "/compact": lambda: self._handle_input("/compact"),
            "/clear": lambda: self._handle_input("/clear"),
            "/cost": lambda: self._handle_input("/cost"),
            "/model": lambda: self._handle_input("/model"),
            "/theme": lambda: self._handle_input("/theme"),
            "/yolo": lambda: self._handle_input("/yolo"),
            "/tools": lambda: self._handle_input("/tools"),
            "/mcp": lambda: self._handle_input("/mcp"),
            "/rag": lambda: self._handle_input("/rag"),
            "/git": lambda: self._handle_input("/git"),
            "/graph": lambda: self._handle_input("/graph"),
            "/clearscreen": lambda: self._handle_input("/clearscreen"),
            "/multiline": lambda: self._handle_input("/multiline"),
            "/session": lambda: self._handle_input("/session"),
            "/permissions": lambda: self._handle_input("/permissions"),
            "/license": lambda: self._handle_input("/license"),
        }

        palette = create_default_palette(command_handlers)

        # Display palette in a simple way (Rich-based)
        self.console.print()
        self.console.print("[bold cyan]Command Palette[/bold cyan] [dim](type to search, Enter to select)[/dim]")
        self.console.print("[dim]" + "-" * 50 + "[/dim]")

        # Show all commands grouped by category
        commands_by_category: dict[str, list] = {}
        for cmd in palette.commands:
            if cmd.category not in commands_by_category:
                commands_by_category[cmd.category] = []
            commands_by_category[cmd.category].append(cmd)

        for category, cmds in commands_by_category.items():
            self.console.print(f"\n[bold yellow]{category}[/bold yellow]")
            for cmd in cmds:
                self.console.print(f"  [cyan]{cmd.name:<20}[/cyan] {cmd.description}")

        self.console.print()
        self.console.print("[dim]Tip: Type the command directly at the prompt[/dim]")
        self.console.print()

    def _get_agent_session_metrics(self) -> dict[str, Any] | None:
        """Fetch aggregated metrics from the active agent state (tokens, cost, etc.)."""
        if not self.agent or not hasattr(self.agent, "graph"):
            return None

        thread_config = getattr(self.agent, "thread_config", None)
        if not thread_config:
            return None

        try:
            state = self.agent.graph.get_state(thread_config)
            if not state or not getattr(state, "values", None):
                return None

            # Convert LangGraph state dict into summary metrics
            summary = state_to_summary(state.values)
            return summary
        except Exception:
            return None

    def _format_context_usage(self, compact: bool = False) -> str:
        """Format context usage for display - delegates to ContextDisplayManager."""
        return self._context_display_manager.format_context_usage(compact)

    def _format_status_snapshot(self) -> str:
        """Return a lightweight status summary string."""
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]

        agent_metrics = self._get_agent_session_metrics()

        token_total = agent_metrics["total_tokens"] if agent_metrics else self.total_tokens
        token_str = f"{token_total:,}"

        lines = [
            "[bold cyan]SE Pilot Status[/bold cyan]",
            f"[cyan]Session start:[/cyan] {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"[cyan]Duration:[/cyan] {elapsed_str}",
            f"[cyan]Commands:[/cyan] {self.command_count}",
            f"[cyan]Total tokens:[/cyan] {token_str}",
            "",  # Blank line before context usage
            self._format_context_usage(compact=False),  # Add context usage display
            "",  # Blank line after context usage
            f"[cyan]Memory notes:[/cyan] {len(self._memory_manager)} ({self.memory_file})",
            f"[cyan]History file:[/cyan] {self.history_file}",
            f"[cyan]prompt_toolkit:[/cyan] {'✅' if HAS_PROMPT_TOOLKIT else '❌'}",
        ]

        if agent_metrics:
            cost = agent_metrics.get("estimated_cost", 0.0)
            lines.append(f"[cyan]Estimated cost:[/cyan] ${cost:.4f}")
            lines.append(
                f"[cyan]Iterations:[/cyan] {agent_metrics.get('iteration', 0)}"
                f" · [cyan]Tool calls:[/cyan] {agent_metrics.get('tool_calls_count', 0)}"
            )
            lines.append(
                f"[cyan]Errors:[/cyan] {agent_metrics.get('error_count', 0)}"
                f" · [cyan]File changes:[/cyan] {agent_metrics.get('file_changes_count', 0)}"
            )
            strategy = agent_metrics.get("current_strategy", "")
            if strategy:
                lines.append(f"[cyan]Strategy:[/cyan] {strategy}")

        return "\n".join(lines)

    def _ensure_history_file_utf8(self) -> None:
        """Normalize the history file so prompt_toolkit can read it."""
        history_path = getattr(self, "history_file", None)
        if not history_path or not os.path.exists(history_path):
            return

        try:
            with open(history_path, encoding="utf-8") as handle:
                handle.read()
            self._sanitize_history_file_entries(history_path)
            return
        except UnicodeDecodeError:
            pass

        backup_path = f"{history_path}.bak"
        try:
            shutil.copy2(history_path, backup_path)
        except Exception:
            backup_path = None

        preferred_encoding = locale.getpreferredencoding(False) or "utf-8"
        try:
            with open(history_path, encoding=preferred_encoding, errors="ignore") as handle:
                data = handle.read()
        except Exception:
            data = ""

        sanitized = sanitize_text(data)
        with open(history_path, "w", encoding="utf-8") as handle:
            handle.write(sanitized)

        if self.console:
            notice = "[yellow]기존 history.txt 를 UTF-8 로 정규화했습니다"
            if backup_path:
                notice += f" · 백업: {backup_path}"
            notice += "[/yellow]"
            self.console.print(notice)

        self._sanitize_history_file_entries(history_path)

    def _sanitize_history_file_entries(self, history_path: str) -> None:
        """Remove assistant/status output accidentally saved in input history."""
        try:
            with open(history_path, encoding="utf-8") as handle:
                raw = handle.read()
        except Exception:
            return

        entries = self._parse_prompt_history_entries(raw)
        if not entries:
            return

        filtered = [entry for entry in entries if self._is_user_history_entry(entry)]
        if len(filtered) == len(entries):
            return

        backup_path = f"{history_path}.mixed-output.bak"
        try:
            shutil.copy2(history_path, backup_path)
        except Exception:
            backup_path = None

        serialized = self._serialize_prompt_history_entries(filtered)
        try:
            with open(history_path, "w", encoding="utf-8") as handle:
                handle.write(serialized)
        except Exception:
            return

        if self.console:
            removed = len(entries) - len(filtered)
            notice = f"[yellow]입력 history에서 비입력 항목 {removed}개를 제거했습니다"
            if backup_path:
                notice += f" · 백업: {backup_path}"
            notice += "[/yellow]"
            self.console.print(notice)

    @staticmethod
    def _parse_prompt_history_entries(raw: str) -> list[str]:
        """Parse prompt_toolkit history file content into entries."""
        entries: list[str] = []
        current_lines: list[str] = []

        for line in raw.splitlines():
            if line.startswith("+"):
                current_lines.append(line[1:])
                continue

            if line.startswith("#"):
                continue

            if not line.strip():
                if current_lines:
                    entries.append("\n".join(current_lines).strip())
                    current_lines = []
                continue

            if current_lines:
                entries.append("\n".join(current_lines).strip())
                current_lines = []
            entries.append(line.strip())

        if current_lines:
            entries.append("\n".join(current_lines).strip())

        return [entry for entry in entries if entry]

    @staticmethod
    def _serialize_prompt_history_entries(entries: list[str]) -> str:
        """Serialize entries back to prompt_toolkit file history format."""
        chunks: list[str] = []
        for entry in entries:
            chunks.append(f"# {datetime.now().isoformat()}")
            for line in entry.split("\n"):
                chunks.append(f"+{line}")
            chunks.append("")
        return "\n".join(chunks)

    @staticmethod
    def _is_user_history_entry(entry: str) -> bool:
        """Heuristic filter to keep only real user inputs in history."""
        text = (entry or "").strip()
        if not text:
            return False

        if any(ch in text for ch in "╭╮╰╯│─⠁⠂⠄⠆⠇⠋⠙⠚⠦"):
            return False

        lower = text.lower()
        blocked_prefixes = (
            "• ",
            "✓",
            "✅",
            "❌",
            "next:",
            "mode:",
            "se pilot ·",
            "ctrl+o last output",
            "🤖 sepilot>",
            "⌘ cmd>",
            "⚡ shell>",
            "📝 note>",
            "@ ref>",
        )
        if lower.startswith(blocked_prefixes):
            return False

        if re.match(r"^\([a-z_]+\)\s+sepilot>", lower):
            return False

        blocked_contains = (
            "작업 체크리스트",
            "요청을 분석해",
            "작업 계획을 구체화",
            "문맥을 정리하고 있어",
        )
        if any(marker in text for marker in blocked_contains):
            return False

        if text.count("\n") > 20 and len(text) > 2000:
            return False

        return True

    def start(self):
        """Start the interactive REPL"""
        self.running = True
        # Show welcome/help on initial launch
        self._show_welcome()

        # First-run setup wizard (if no model configured)
        if needs_setup():
            wizard_result = run_setup_wizard(self.console)
            if wizard_result:
                from sepilot.config.model_profile import ModelConfig

                config = ModelConfig(
                    base_url=wizard_result.get("base_url"),
                    model=wizard_result.get("model"),
                    api_key=wizard_result.get("api_key"),
                )
                self.model_profile_manager.current_config = config
                if self.agent:
                    apply_ok = apply_model_config_to_agent(
                        self.agent, config, self.console,
                        lambda c: create_llm_from_config(c, self.agent, self.console)
                    )
                    if not apply_ok and self.agent_factory:
                        new_agent = self.agent_factory(config)
                        if new_agent:
                            self.agent = new_agent
                            self._context_display_manager.agent = new_agent

        # Auto-load default profile if set
        try:
            default_profile = self.model_profile_manager.get_default_profile()
            if default_profile:
                success = self.model_profile_manager.load_profile(default_profile)
                if success:
                    self.console.print(f"[cyan]📋 Default profile '{default_profile}' loaded[/cyan]")
                    if self.agent:
                        config = self.model_profile_manager.get_current_config()
                        resolved_model = (
                            config.model
                            or getattr(getattr(self.agent, "settings", None), "model", None)
                            or "not set"
                        )
                        self.console.print(f"[dim]  Model: {resolved_model}[/dim]")
                        apply_ok = apply_model_config_to_agent(
                            self.agent, config, self.console,
                            lambda c: create_llm_from_config(c, self.agent, self.console)
                        )
                        if apply_ok:
                            self.console.print("[green]✅ Configuration applied to agent[/green]")
        except Exception:
            pass  # Default profile loading is best-effort

        while self.running:
            try:
                self._render_prompt_header()
                # Get user input
                if self.session:
                    try:
                        user_input = self.session.prompt(
                            self._get_dynamic_prompt_message,
                            rprompt=self._get_dynamic_prompt_rprompt,
                        )
                    except UnicodeDecodeError:
                        self.console.print(
                            "[yellow]입력 디코딩 오류가 발생하여 기본 입력 모드로 전환합니다.[/yellow]"
                        )
                        self.session = None
                        try:
                            user_input = input(prompt_text)
                        except (UnicodeDecodeError, UnicodeError):
                            self.console.print("[yellow]⚠️  입력 인코딩 오류 (무시됨)[/yellow]")
                            continue
                else:
                    # Fallback to basic input
                    try:
                        prompt_text = f"({self._get_current_mode_label()}) sepilot> "
                        user_input = input(prompt_text)
                    except (UnicodeDecodeError, UnicodeError):
                        self.console.print("[yellow]⚠️  입력 인코딩 오류 (무시됨)[/yellow]")
                        continue

                # Handle input
                self._handle_input(user_input)

            except (EOFError, KeyboardInterrupt):
                # Ctrl+D or Ctrl+C
                self.console.print("\n[yellow]Use /exit or /quit to exit[/yellow]")
                continue
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                continue

        self._show_goodbye()

    def _get_input_mode(self) -> str:
        """Infer input mode from current buffer prefix for dynamic prompt UX."""
        if not HAS_PROMPT_TOOLKIT or get_app_or_none is None:
            return "chat"

        try:
            app = get_app_or_none()
            if app is None or app.current_buffer is None:
                return "chat"
            text = (app.current_buffer.text or "").lstrip()
        except Exception:
            return "chat"

        if text.startswith("!"):
            return "shell"
        if text.startswith("#"):
            return "note"
        if text.startswith("/"):
            return "command"
        if text.startswith("@"):
            return "reference"
        return "chat"

    def _get_current_mode_label(self) -> str:
        """Resolve current mode label (AUTO/PLAN/CODE/EXEC) for prompt prefix."""
        if not self.agent:
            return "AUTO"

        pending = getattr(self.agent, "_pending_mode_update", None)
        if pending and "current_mode" in pending:
            mode = pending.get("current_mode")
            if isinstance(mode, AgentMode):
                return mode.value.upper()
            return str(mode).upper()

        try:
            thread_mgr = getattr(self.agent, "_thread_manager", None)
            graph = getattr(self.agent, "graph", None)
            thread_id = getattr(thread_mgr, "thread_id", None) if thread_mgr else None
            if graph is not None and thread_id:
                snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
                if snapshot and snapshot.values:
                    mode = snapshot.values.get("current_mode", AgentMode.AUTO)
                    if isinstance(mode, AgentMode):
                        return mode.value.upper()
                    return str(mode).upper()
        except Exception:
            pass

        return "AUTO"

    def _get_dynamic_prompt_message(self):
        """Return prompt label that updates as users type !/#//@ prefixes."""
        mode = self._get_input_mode()
        mode_label = self._get_current_mode_label()
        if mode == "shell":
            return [("bold ansiyellow", "⚡ shell> ")]
        if mode == "note":
            return [("bold ansimagenta", "📝 note> ")]
        if mode == "command":
            return [("bold ansigreen", "⌘ cmd> ")]
        if mode == "reference":
            return [("bold ansicyan", "@ ref> ")]
        return [("bold ansicyan", f"({mode_label}) sepilot> ")]

    def _get_dynamic_prompt_rprompt(self):
        """Return a minimal right-side mode hint."""
        mode = self._get_input_mode()
        if mode == "shell":
            return [("ansiyellow", "mode: shell")]
        if mode == "note":
            return [("ansimagenta", "mode: memory note")]
        if mode == "command":
            return [("ansigreen", "mode: slash command")]
        if mode == "reference":
            return [("ansicyan", "mode: file reference")]
        return ""

    @staticmethod
    def _is_direct_execution_request(user_input: str) -> bool:
        """Detect imperative requests that should skip auto-skill injection."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        imperative_keywords = (
            "commit 해줘",
            "커밋 해줘",
            "직접 commit",
            "직접 커밋",
            "실행해줘",
            "바로 실행",
            "git commit",
            "git add",
            "git push",
            "커밋",
        )
        return any(k in text for k in imperative_keywords)

    def _handle_input(self, user_input: str):
        """Handle user input"""
        user_input = sanitize_text(user_input.strip())

        # Empty input
        if not user_input:
            return

        # Shell command prefix
        if user_input.startswith('!'):
            self._run_shell_command(user_input[1:].strip())
            return

        # Memory command prefix
        if user_input.startswith('#'):
            self._handle_memory_command(user_input[1:].strip())
            return

        # Check for special commands
        if user_input.startswith('/'):
            lower_input = user_input.lower()

            # Sort commands by length (longest first) to match specific commands before base commands
            # e.g., "/git status" should match before "/git"
            sorted_commands = sorted(self.special_commands.keys(), key=len, reverse=True)

            # Find the first matching command
            for cmd in sorted_commands:
                if lower_input == cmd or lower_input.startswith(cmd + " "):
                    # Execute the matched command
                    self.special_commands[cmd](user_input)
                    return

            # Check for custom commands (Claude Code style)
            parts = user_input[1:].split(maxsplit=1)  # Remove leading /
            cmd_name = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""

            custom_cmd = self._custom_command_manager.get_command(cmd_name)
            if custom_cmd:
                # Extract @file references from arguments (Claude Code style)
                file_content = ""
                clean_args = cmd_args
                if "@" in cmd_args:
                    # Parse @file references
                    file_refs = re.findall(r'@(\S+)', cmd_args)
                    file_contents = []
                    for ref in file_refs:
                        filepath = os.path.expanduser(ref)
                        if os.path.isfile(filepath):
                            try:
                                with open(filepath, encoding='utf-8', errors='replace') as f:
                                    content = f.read()
                                file_contents.append(f"--- {ref} ---\n{content}")
                                self.console.print(f"[dim]📎 Attached: {ref}[/dim]")
                            except Exception as e:
                                self.console.print(f"[yellow]⚠️  Could not read {ref}: {e}[/yellow]")
                    if file_contents:
                        file_content = "\n\n".join(file_contents)
                    # Remove @file references from arguments
                    clean_args = re.sub(r'@\S+', '', cmd_args).strip()

                # Show execution status (Claude Code style)
                self.console.print(f"[cyan]📜 /{cmd_name} is running...[/cyan]")

                # Expand the custom command and execute it
                expanded_prompt = custom_cmd.expand(
                    arguments=clean_args,
                    file_content=file_content
                )

                # Show preview
                preview = expanded_prompt[:300] + "..." if len(expanded_prompt) > 300 else expanded_prompt
                self.console.print(f"[dim]{preview}[/dim]")
                self.console.print()

                # Execute the expanded prompt through the agent
                if self.execute_callback:
                    try:
                        result = self.execute_callback(expanded_prompt)
                        self._overlay_manager.update_result(
                            result or "",
                            f"Custom command: /{cmd_name}"
                        )
                        self.console.print(f"[green]✅ /{cmd_name} completed[/green]")
                    except Exception as e:
                        self.console.print(f"[red]❌ /{cmd_name} failed: {e}[/red]")
                return

            # No command matched
            command = user_input.split()[0].lower()
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[dim]Type /help for available commands[/dim]")
            self.console.print("[dim]Type /commands to see custom commands[/dim]")
            return

        # Greeting 빠른 경로: RAG/skill/reference expansion 모두 건너뛰기
        from sepilot.agent.request_classifier import GREETING_KEYWORDS
        text_lower = user_input.strip().lower()
        is_greeting = len(text_lower) < 50 and any(
            text_lower == kw or text_lower.startswith(kw + " ") or text_lower.startswith(kw + "!")
            for kw in GREETING_KEYWORDS
        )

        if is_greeting:
            expanded_input = user_input
        else:
            # Expand @file references (using reference expander module)
            expanded_input = self._reference_expander.expand_references(user_input)

            # Check for @agent: invocation markers (OpenCode style)
            if "__AGENT_INVOKE__:" in expanded_input:
                self._handle_agent_invocation(expanded_input, user_input)
                return

            # Prepend memory context if available (using memory manager module)
            memory_context = self._memory_manager.get_context_injection()
            if memory_context:
                expanded_input = memory_context + expanded_input

            # Add RAG context using semantic search on user query
            rag_context = self._get_rag_context(expanded_input)
            if rag_context:
                expanded_input = rag_context + expanded_input

            # Add skill context if activated (from /skill command)
            if hasattr(self, '_skill_context') and self._skill_context:
                skill_context = "\n# Skill Context (Guidelines)\n"
                for ctx in self._skill_context:
                    skill_context += ctx + "\n"
                skill_context += "\n# User Request\n"
                expanded_input = skill_context + expanded_input
                # Clear skill context after use
                self._skill_context = []

            # Auto-trigger skills based on input keywords (Claude Code style)
            skill_manager = get_skill_manager()
            auto_skill = None
            if not self._is_direct_execution_request(user_input):
                auto_skill = skill_manager.find_matching_skill(user_input)
            if auto_skill:
                metadata = auto_skill.get_metadata()
                self.console.print(f"[dim]🎯 Auto-activated skill: [cyan]{metadata.name}[/cyan][/dim]")
                context = {
                    "agent": self.agent,
                    "console": self.console,
                    "conversation": self.conversation_context
                }
                result = auto_skill.execute(user_input, context)
                if result.success and result.prompt_injection:
                    skill_context = "\n# Skill Context (Auto-activated: " + metadata.name + ")\n"
                    skill_context += result.prompt_injection + "\n"
                    skill_context += "\n# User Request\n"
                    expanded_input = skill_context + expanded_input

        # Execute user command
        self.command_count += 1
        self.console.print()  # Blank line before output

        # Clear previous logs and start capturing (using overlay manager)
        self._overlay_manager.clear_logs()

        # Start progress indicator (only in non-verbose mode)
        indicator = None
        agent_verbose = getattr(self.agent, 'verbose', False) if self.agent else False

        if not agent_verbose:
            from sepilot.ui.status_indicator import AgentStatusIndicator
            indicator = AgentStatusIndicator(self.console)
            if self.agent and hasattr(self.agent, 'status_indicator'):
                self.agent.status_indicator = indicator
            indicator.start("요청을 처리하고 있어요...")

            # Attach step logger for interactive progress display
            from sepilot.ui.step_logger import StepLogger
            if self.agent and hasattr(self.agent, 'step_logger'):
                self.agent.step_logger = StepLogger(
                    Console(stderr=True), enabled=True,
                )

        try:
            # Setup tee output (using TeeOutput from output_overlays module)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            tee_stdout = TeeOutput(original_stdout, self._overlay_manager.execution_logs)
            tee_stderr = TeeOutput(original_stderr, self._overlay_manager.execution_logs)

            # Redirect to tee (both display and capture)
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr

            try:
                # Execute with real-time output AND capturing
                result = self.execute_callback(expanded_input)
            finally:
                # Restore original stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            # Update token usage from agent if available
            # NOTE: We don't need to manually update self.total_tokens here anymore
            # because _get_context_usage_info() reads directly from the agent state
            # This avoids double-counting and ensures we always show the current state

            # Display completion indicator
            if HAS_PROMPT_TOOLKIT and self.session:
                self.console.print("[dim]✓ Execution complete (Ctrl+L to view full logs)[/dim]")

            if result:
                result_text = result if isinstance(result, str) else str(result)
                if result_text.strip():
                    # Display the result immediately
                    self.console.print()  # Blank line before output
                    self.console.print(result_text)
                    self.console.print()  # Blank line after output
                    # Also save to last_result for Ctrl+O viewing (using overlay manager)
                    self._overlay_manager.update_result(
                        result_text.strip(),
                        f"Agent response · {datetime.now().strftime('%H:%M:%S')}"
                    )

                    # Record exchange for undo/redo (OpenCode style)
                    try:
                        undo_manager = get_undo_redo_manager()
                        user_msg_idx = len(self.conversation_context)
                        assistant_msg_idx = user_msg_idx + 1
                        undo_manager.record_exchange(
                            user_message=user_input,
                            assistant_response=result_text,
                            user_msg_index=user_msg_idx,
                            assistant_msg_index=assistant_msg_idx,
                        )
                    except Exception:
                        pass  # Don't fail if undo/redo recording fails
        except Exception as e:
            self.console.print("[bold red]Error executing command:[/bold red]")
            self.console.print(f"[red]{str(e)}[/red]")
            import traceback
            if self.console.is_terminal:
                self.console.print("[dim]" + traceback.format_exc() + "[/dim]")
        finally:
            # Always stop the progress indicator
            if indicator:
                indicator.stop()
                if self.agent and hasattr(self.agent, 'status_indicator'):
                    self.agent.status_indicator = None
            # Clean up step logger
            if self.agent and hasattr(self.agent, 'step_logger'):
                self.agent.step_logger = None

        # Auto-compact check after each command (Claude Code style)
        self._check_auto_compact()

        self.console.print()  # Blank line before next prompt

    def _render_prompt_header(self):
        """Render a small status banner before each prompt (Claude Code style)."""
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        memory_count = len(self._memory_manager)

        # Add context usage to banner (Claude Code style)
        context_usage = self._format_context_usage(compact=True)

        banner_prefix = "[cyan]SE Pilot[/cyan]"

        banner = (
            f"{banner_prefix} · "
            f"[magenta]cmd {self.command_count}[/magenta] · "
            f"[green]mem {memory_count}[/green] · "
            f"{context_usage} · "
            f"[yellow]{elapsed_str}[/yellow]"
        )
        self.console.print(banner)
        self.console.print("[dim]Ctrl+O last output · Ctrl+L execution logs · Ctrl+K clear screen · F2 status · /help · @file attach · !<cmd> shell[/dim]")

    def _run_shell_command(self, command: str):
        """Execute a local shell command."""
        if not command:
            self.console.print("[yellow]Usage: !<shell command>[/yellow]")
            return

        self.console.print(f"[magenta]$ {command}[/magenta]")
        start = datetime.now()
        try:
            result = subprocess.run(
                _build_shell_command(command),
                capture_output=True,
                text=True,
                env=os.environ
            )
            duration = (datetime.now() - start).total_seconds()
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                self.console.print(stdout)
            if stderr:
                self.console.print(f"[red]{stderr}[/red]")
            if not stdout and not stderr:
                self.console.print("[dim]No output[/dim]")

            meta_line = f"[dim]exit {result.returncode} · {duration:.2f}s[/dim]"
            self.console.print(meta_line)

            snapshot_lines = []
            if stdout:
                snapshot_lines.append(stdout)
            if stderr:
                snapshot_lines.append(f"(stderr)\n{stderr}")
            if not snapshot_lines:
                snapshot_lines.append("No output")
            snapshot_lines.append(meta_line)
            combined_output = "\n".join(snapshot_lines)
            self._overlay_manager.update_result(combined_output, f"Shell: {command}")
            if self.manual_observer:
                self.manual_observer(command, combined_output)
        except Exception as exc:
            err_msg = f"Failed to run command: {exc}"
            self.console.print(f"[red]{err_msg}[/red]")
            self._overlay_manager.update_result(err_msg, f"Shell: {command}")
            if self.manual_observer:
                self.manual_observer(command, err_msg)

    def _handle_memory_command(self, body: str):
        """Process # memory commands (delegates to memory manager)."""
        self._memory_manager.handle_command(body)

    def _handle_agent_invocation(self, expanded_input: str, original_input: str):
        """Handle @agent:name invocation (OpenCode style).

        Detects __AGENT_INVOKE__:{agent_name} markers and routes to appropriate subagent.

        Args:
            expanded_input: Expanded input containing agent markers
            original_input: Original user input for context
        """
        import asyncio
        import re as re_module

        # Extract agent name from marker
        marker_pattern = r'__AGENT_INVOKE__:(\w+)'
        match = re_module.search(marker_pattern, expanded_input)

        if not match:
            self.console.print("[red]Error: Could not parse agent invocation[/red]")
            return

        agent_name = match.group(1).lower()

        # Remove the marker from input to get the actual task
        task_description = re_module.sub(marker_pattern, '', expanded_input).strip()
        if not task_description:
            task_description = original_input

        # Remove @agent:name from task description
        task_description = re_module.sub(r'@agent:\w+', '', task_description).strip()

        if not task_description:
            self.console.print("[yellow]Please provide a task for the agent[/yellow]")
            self.console.print("[dim]Example: @agent:coder 사용자 인증 함수를 구현해줘[/dim]")
            return

        # Agent name to type mapping
        agent_type_map = {
            "explore": "analyzer",
            "coder": "codegen",
            "reviewer": "analyzer",
            "refactor": "refactoring",
            "docs": "documentation",
            "test": "testing",
            "debug": "analyzer",
        }

        agent_type = agent_type_map.get(agent_name)
        if not agent_type:
            self.console.print(f"[red]Unknown agent: {agent_name}[/red]")
            self.console.print(f"[dim]Available: {', '.join(agent_type_map.keys())}[/dim]")
            return

        self.console.print()
        self.console.print(f"[bold cyan]🤖 Invoking @agent:{agent_name}[/bold cyan]")
        self.console.print(f"[dim]Task: {task_description[:100]}{'...' if len(task_description) > 100 else ''}[/dim]")
        self.console.print()

        try:
            # Import subagent modules
            from sepilot.agent.subagent.models import SubAgentTask
            from sepilot.agent.subagent.orchestrator import SubAgentOrchestrator
            from sepilot.agent.subagent.specialized_agents import (
                AnalyzerSubAgent,
                CodeGenSubAgent,
                DocumentationSubAgent,
                RefactoringSubAgent,
                TestingSubAgent,
            )

            # Get LLM from agent
            llm = None
            tools = []
            if self.agent:
                if hasattr(self.agent, 'llm'):
                    llm = self.agent.llm
                if hasattr(self.agent, 'tools'):
                    tools = self.agent.tools

            # Create appropriate subagent
            agent_classes = {
                "analyzer": AnalyzerSubAgent,
                "codegen": CodeGenSubAgent,
                "refactoring": RefactoringSubAgent,
                "documentation": DocumentationSubAgent,
                "testing": TestingSubAgent,
            }

            SubAgentClass = agent_classes.get(agent_type)
            if not SubAgentClass:
                self.console.print(f"[red]SubAgent class not found for type: {agent_type}[/red]")
                return

            # Create subagent instance
            subagent = SubAgentClass(
                agent_id=f"{agent_name}_subagent",
                tools=tools,
                llm=llm
            )

            # Create task
            task = SubAgentTask(
                task_id=f"agent_invoke_{agent_name}",
                description=task_description,
                agent_type=agent_type,
                context={
                    "original_input": original_input,
                    "agent_name": agent_name,
                }
            )

            # Create orchestrator
            orchestrator = SubAgentOrchestrator(
                llm=llm,
                max_parallel=1,
                subagents=[subagent]
            )

            # Execute task
            self.console.print("[dim]Executing agent task...[/dim]")

            # Run async task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    orchestrator.execute_parallel([task])
                )
            finally:
                loop.close()

            # Process results
            if results:
                for task_id, result in results.items():
                    if result.is_success():
                        self.console.print()
                        self.console.print(f"[bold green]✅ @agent:{agent_name} completed[/bold green]")
                        self.console.print()
                        if result.output:
                            # Display result
                            self.console.print(result.output)
                        self._overlay_manager.update_result(
                            result.output or "",
                            f"@agent:{agent_name}"
                        )
                    else:
                        self.console.print()
                        self.console.print(f"[bold red]❌ @agent:{agent_name} failed[/bold red]")
                        if result.error:
                            self.console.print(f"[red]{result.error}[/red]")
            else:
                self.console.print("[yellow]No results from agent[/yellow]")

        except ImportError as e:
            self.console.print(f"[red]SubAgent modules not available: {e}[/red]")
            self.console.print("[dim]Falling back to main agent...[/dim]")
            # Fallback: use main execute_callback
            if self.execute_callback:
                result = self.execute_callback(task_description)
                if result:
                    self.console.print(result)

        except Exception as e:
            self.console.print(f"[red]Agent invocation error: {e}[/red]")
            import traceback
            if self.console.is_terminal:
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def _show_welcome(self):
        """Show clean startup banner (Claude Code style)."""
        version_info = _get_version_info()
        logo_lines = [
            "  ____  _____ ____  _ _       _   ",
            " / ___|| ____|  _ \\(_) | ___ | |_ ",
            " \\___ \\|  _| | |_) | | |/ _ \\| __|",
            "  ___) | |___|  __/| | | (_) | |_ ",
            " |____/|_____|_|   |_|_|\\___/ \\__|",
        ]

        self.console.print()
        for line in logo_lines:
            self.console.print(f"[bold cyan]{line}[/bold cyan]")
        self.console.print(f"[dim]Interactive mode · {version_info}[/dim]")
        self.console.print("[dim]Prompt: ask naturally, @file, !cmd, #note[/dim]")
        self.console.print("[dim]Quick keys: Ctrl+O output · Ctrl+L logs · Ctrl+K clear · F2 status[/dim]")
        self.console.print("[dim]/help for full guide · /exit to quit[/dim]")
        self.console.print()

    def _show_welcome_skills(self):
        """Show available skills in welcome message."""
        try:
            from sepilot.skills import get_skill_manager

            skill_manager = get_skill_manager()
            skills = skill_manager.list_skills()

            if skills:
                self.console.print("[bold cyan]Available Skills:[/bold cyan]")
                skill_names = [f"[green]{m.name}[/green]" for m in skills]
                self.console.print(f"  {', '.join(skill_names)}")
                self.console.print("[dim]  Use `/skill <name>` to activate[/dim]")
        except Exception:
            pass

    def _show_welcome_commands(self):
        """Show available custom commands in welcome message."""
        try:
            from sepilot.commands import get_command_manager

            command_manager = get_command_manager()
            commands = command_manager.list_commands()

            if commands:
                self.console.print("[bold cyan]Custom Commands:[/bold cyan]")
                cmd_names = [f"[yellow]/{c.name}[/yellow]" for c in commands]
                self.console.print(f"  {', '.join(cmd_names)}")
                self.console.print("[dim]  Use `/<command>` to execute[/dim]")
        except Exception:
            pass

    def _show_goodbye(self):
        """Show goodbye message"""
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]

        self.console.print()
        self.console.print(f"[bold green]Goodbye![/bold green] Session duration {elapsed_str}, commands executed {self.command_count}")

    # Special command handlers - delegating to SOLID modules

    def _cmd_help(self, _input: str):
        """Show help message - delegates to core_commands module."""
        handle_help(self.console)

    def _cmd_exit(self, _input: str):
        """Exit interactive mode"""
        self.running = False

    def _cmd_clearscreen(self, _input: str):
        """Clear screen - delegates to core_commands module."""
        handle_clearscreen(self.console)

    def _cmd_history(self, _input: str):
        """Show command history - delegates to core_commands module."""
        handle_history(
            console=self.console,
            history_file=self.history_file,
            conversation_context=self.conversation_context,
            input_text=_input,
        )

    def _cmd_status(self, _input: str):
        """Show session status"""
        self.console.print(self._format_status_snapshot())

    def _cmd_license(self, _input: str):
        """Show SEPilot3 license information"""
        # Find LICENSE.md file
        # Try multiple possible locations
        possible_paths = [
            os.path.join(os.getcwd(), "LICENSE.md"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "LICENSE.md"),
            os.path.expanduser("~/.sepilot/LICENSE.md"),
        ]

        license_file = None
        for path in possible_paths:
            if os.path.exists(path):
                license_file = path
                break

        if not license_file:
            self.console.print("[yellow]⚠️  LICENSE.md file not found[/yellow]")
            self.console.print("[dim]Searched in:[/dim]")
            for path in possible_paths:
                self.console.print(f"[dim]  - {path}[/dim]")
            self.console.print("\n[bold cyan]SEPilot3 License Summary:[/bold cyan]")
            self.console.print("[yellow]Source-Available Proprietary License[/yellow]")
            self.console.print("• ✅ Free for personal, educational, and non-profit use")
            self.console.print("• ⚠️  Commercial use requires a commercial license")
            self.console.print("• ❌ Modification and redistribution prohibited")
            self.console.print("• ✅ Pull requests and contributions welcome")
            self.console.print("\n[dim]For full license text, see LICENSE.md in the project repository[/dim]")
            return

        try:
            # Read LICENSE.md file
            with open(license_file, encoding='utf-8') as f:
                license_content = f.read()

            # Parse command arguments for options
            args = _input.strip().split()

            # Check for --summary or -s flag
            if len(args) > 1 and args[1] in ['--summary', '-s', 'summary']:
                # Show summary only
                self.console.print("[bold cyan]📜 SEPilot3 License Summary[/bold cyan]\n")
                summary_text = """
**License Type:** Source-Available Proprietary License v1.0

**Quick Reference:**

| Activity | Individual/Non-Profit | Commercial Entity |
|----------|----------------------|-------------------|
| View source code | ✅ Allowed | ✅ Allowed |
| Use the software | ✅ Free | ⚠️ License required |
| Modify source code | ❌ Prohibited | ❌ Prohibited |
| Redistribute | ❌ Prohibited | ❌ Prohibited |
| Submit Pull Requests | ✅ Allowed | ✅ Allowed |
| Commercial use | N/A | ⚠️ License required |

**Key Points:**
• Free for personal, educational, research, and non-profit use
• Commercial entities require a commercial license
• Source code visible but not modifiable or redistributable
• Contributions via Pull Requests are welcome
• Based on permissive open-source dependencies (MIT, BSD, Apache 2.0)

**For Full License:**
Type `/license full` or see LICENSE.md in the project directory

**Contact:** See LICENSE.md for licensing inquiries
                """
                self.console.print(Markdown(summary_text))
            else:
                # Show full license
                self.console.print("[bold cyan]📜 SEPilot3 License[/bold cyan]")
                self.console.print(f"[dim]Reading from: {license_file}[/dim]\n")

                # Display using Rich Markdown for better formatting
                self.console.print(Markdown(license_content))

                self.console.print(f"\n[dim]Full license text: {license_file}[/dim]")
                self.console.print("[dim]💡 Tip: Use '/license summary' for a quick overview[/dim]")

        except Exception as e:
            self.console.print(f"[red]❌ Error reading LICENSE.md: {e}[/red]")
            import traceback
            if self.console.is_terminal:
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def _cmd_resume(self, input_text: str):
        """Resume a previous conversation thread - delegates to session_commands module."""
        handle_resume(self.console, self.agent, input_text, self.conversation_context)

    def _cmd_new(self, _input: str):
        """Start a new conversation thread - delegates to session_commands module."""
        result = handle_new(self.console, self.agent, self.conversation_context)
        if result[0] is not None:
            # Reset session stats on successful new thread
            self.command_count = result[1]
            self.total_tokens = result[2]

    def _cmd_rewind(self, _input: str):
        """Rewind conversation and/or code changes - delegates to session_commands module."""
        handle_rewind(self.console, self.agent, self.conversation_context, _input)

    def _cmd_undo(self, _input: str):
        """Undo last exchange (OpenCode style)

        Removes the most recent user message, all subsequent responses,
        and associated file changes.

        Usage:
            /undo         - Undo last exchange
            /undo --list  - Show undo stack
        """
        result = handle_undo(
            self.console,
            self.agent,
            self.conversation_context,
            _input
        )

    def _cmd_redo(self, _input: str):
        """Redo undone exchange (OpenCode style)

        Restores a previously undone message and its corresponding
        file modifications.

        Usage:
            /redo         - Redo last undone exchange
            /redo --list  - Show redo stack
        """
        result = handle_redo(
            self.console,
            self.agent,
            self.conversation_context,
            _input
        )

    def _cmd_multiline(self, _input: str):
        """Toggle multi-line input mode"""
        if not HAS_PROMPT_TOOLKIT or not self.session:
            self.console.print("[yellow]Multi-line mode requires prompt_toolkit[/yellow]")
            return

        # Toggle multiline
        current = self.session.multiline
        self.session.multiline = not current

        if self.session.multiline:
            self.console.print("[green]Multi-line mode enabled[/green]")
            self.console.print("[dim]Press Alt+Enter or Meta+Enter to submit[/dim]")
        else:
            self.console.print("[yellow]Multi-line mode disabled[/yellow]")
            self.console.print("[dim]Press Enter to submit[/dim]")

    def _cmd_yolo(self, _input: str):
        """Toggle YOLO mode (auto-approve all tool calls)"""
        if not self.agent:
            self.console.print("[yellow]⚠️  Agent not available - /yolo command disabled[/yellow]")
            return

        # Toggle auto-approve
        current = self.agent.auto_approve
        self.agent.auto_approve = not current

        if self.agent.auto_approve:
            self.console.print("[bold red]🚀 YOLO MODE ENABLED![/bold red]")
            self.console.print("[yellow]⚠️  All tool executions will be auto-approved[/yellow]")
            self.console.print("[dim]Agent will run without human-in-the-loop approval[/dim]")
            self.console.print("[dim]Type /yolo again to disable[/dim]")
        else:
            self.console.print("[green]✓ YOLO MODE DISABLED[/green]")
            self.console.print("[dim]Human-in-the-loop approval restored[/dim]")

    def _cmd_graph(self, _input: str):
        """Visualize LangGraph structure as ASCII - delegates to graph_commands module."""
        handle_graph_command(self.console, self.agent, _input)

    def _cmd_reset(self, _input: str):
        """Reset session statistics (command count and token usage)"""
        # Save old values for display
        old_commands = self.command_count
        old_tokens = self.total_tokens
        old_duration = datetime.now() - self.session_start

        # Reset session stats
        self.session_start = datetime.now()
        self.command_count = 0
        self.total_tokens = 0

        # Display reset confirmation
        self.console.print("[bold cyan]🔄 Session Reset[/bold cyan]")
        self.console.print("[dim]Previous session:[/dim]")
        self.console.print(f"  [cyan]Duration:[/cyan] {str(old_duration).split('.')[0]}")
        self.console.print(f"  [cyan]Commands:[/cyan] {old_commands}")
        self.console.print(f"  [cyan]Total tokens:[/cyan] {old_tokens:,}")
        self.console.print("\n[green]✓ Session statistics have been reset[/green]")

    def _cmd_container(self, _input: str):
        """Container and Docker commands - delegates to devops_commands module."""
        handle_container_command(_input, self.agent, self.console)

    def _cmd_helm(self, _input: str):
        """Helm Chart generation - delegates to devops_commands module."""
        handle_helm_command(_input, self.agent, self.console)

    def _cmd_se(self, _input: str):
        """Software Engineering commands - delegates to devops_commands module."""
        handle_se_command(_input, self.agent, self.console)

    def _cmd_gitops(self, _input: str):
        """GitOps/ArgoCD commands - delegates to devops_commands module."""
        handle_gitops_command(_input, self.agent, self.console)

    def _cmd_model(self, _input: str):
        """Model configuration commands - delegates to model_commands module."""
        result = handle_model_command(
            _input,
            self.model_profile_manager,
            self.agent,
            self.console,
            lambda config: create_llm_from_config(config, self.agent, self.console),
            agent_factory=self.agent_factory
        )
        # If a new agent was created (returned from handle_model_command)
        if result is not None and result is not self.agent:
            self.agent = result
            # Update stale agent reference in context display manager
            self._context_display_manager.agent = result

    def _cmd_security(self, _input: str):
        """Security / DevSecOps commands - delegates to security_commands module."""
        handle_security_command(self.console, self.agent, _input)

    def _cmd_bench(self, _input: str):
        """Bench automation commands - delegates to bench_commands module."""
        bench_holder = {'agent': self.bench_agent}
        handle_bench_command(self.console, self.agent, bench_holder, _input)
        # Update bench_agent reference if lazily initialized
        if bench_holder['agent'] is not self.bench_agent:
            self.bench_agent = bench_holder['agent']

    def _cmd_mcp(self, _input: str):
        """MCP server management - delegates to mcp_commands module."""
        # Use a dict holder for lazy initialization
        mcp_holder = {'manager': self._mcp_config_manager}
        handle_mcp_command(_input, mcp_holder, self.console, self.session)
        # Update the manager reference if it was lazily initialized
        if mcp_holder['manager'] is not self._mcp_config_manager:
            self._mcp_config_manager = mcp_holder['manager']

    def _get_rag_context(self, query: str) -> str:
        """Get RAG context using semantic search - delegates to rag_commands module."""
        return get_rag_context(query, self.console)

    def _cmd_rag(self, _input: str):
        """RAG management - delegates to rag_commands module."""
        handle_rag_command(_input, self.console)

    def _cmd_tools(self, _input: str):
        """List available tools - delegates to tools_commands module."""
        handle_tools_command(_input, self.agent, self.console)

    def _cmd_skill(self, _input: str):
        """Skills system - delegates to skill_commands module."""
        # Use holder dict for skill context (prompt injections)
        if not hasattr(self, '_skill_context_holder'):
            self._skill_context_holder = {'contexts': getattr(self, '_skill_context', [])}
        handle_skill_command(
            self.console,
            self.agent,
            self.conversation_context,
            self._skill_context_holder,
            _input
        )
        # Update _skill_context from holder
        self._skill_context = self._skill_context_holder.get('contexts', [])

    def _cmd_commands(self, _input: str):
        """Custom commands system - delegates to custom_commands module."""
        handle_custom_commands_command(self.console, self._custom_command_manager, _input)

    def _cmd_k8s_health(self, _input: str):
        """Kubernetes health checks - delegates to k8s_commands module."""
        handle_k8s_health_command(_input, self.agent, self.console)

    def _cmd_theme(self, _input: str):
        """Theme management - delegates to theme_commands module."""
        handle_theme(self.console, _input)

    def _cmd_stats(self, _input: str):
        """Usage statistics and cost tracking - delegates to stats_commands module."""
        handle_stats(self.console, _input, self.agent)

    def _cmd_permissions(self, _input: str):
        """Permission rules management - delegates to permission_commands module."""
        handle_permissions(self.console, _input)

    def _cmd_session(self, _input: str):
        """Session export/import - delegates to session_commands module."""
        handle_session(self.console, self.agent, _input)

    def _cmd_plan_mode(self, _input: str):
        """Switch to PLAN mode - delegates to mode_commands module."""
        handle_plan_mode(self.console, self.agent)

    def _cmd_code_mode(self, _input: str):
        """Switch to CODE mode - delegates to mode_commands module."""
        handle_code_mode(self.console, self.agent)

    def _cmd_exec_mode(self, _input: str):
        """Switch to EXEC mode - delegates to mode_commands module."""
        handle_exec_mode(self.console, self.agent)

    def _cmd_auto_mode(self, _input: str):
        """Return to AUTO mode - delegates to mode_commands module."""
        handle_auto_mode(self.console, self.agent)

    def _cmd_mode(self, _input: str):
        """Show current mode status - delegates to mode_commands module."""
        handle_mode_command(self.console, self.agent, _input)

    def _check_auto_compact(self) -> None:
        """Check context usage and auto-compact if needed - delegates to ContextDisplayManager."""
        llm = None
        if self.agent and hasattr(self.agent, 'llm'):
            llm = self.agent.llm
        self._context_display_manager.check_auto_compact(llm)

    def _cmd_compact(self, _input: str):
        """Compact conversation context - delegates to context_commands module."""
        handle_compact(self.console, self.conversation_context, self.agent, _input)

    def _cmd_clearcontext(self, _input: str):
        """Clear all conversation context - delegates to context_commands module."""
        handle_clear_context(self.console, self.conversation_context, self.agent)

    def _cmd_context(self, _input: str):
        """Show context usage visualization - delegates to context_commands module.

        Usage:
            /context           - Show context usage grid
            /context --detail  - Show detailed message breakdown
        """
        handle_context(self.console, self.conversation_context, self.agent, _input)

    def _cmd_cost(self, _input: str):
        """Show estimated cost - delegates to context_commands module."""
        handle_cost(self.console, self.total_tokens, self.command_count, self.agent)
