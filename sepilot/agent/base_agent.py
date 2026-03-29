"""Agent implementation using LangGraph standard pattern with ToolNode"""

import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

try:
    from langchain_openai import ChatOpenAI  # noqa: F401 – used dynamically via globals/locals
except ImportError:
    pass
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command, interrupt

# Try to import SqliteSaver, fall back to MemorySaver if not available
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    HAS_SQLITE_SAVER = True
except ImportError:
    HAS_SQLITE_SAVER = False
try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    HAS_JSONPLUS_SERIALIZER = True
except ImportError:
    HAS_JSONPLUS_SERIALIZER = False
import contextlib

from rich.console import Console

# prompt_toolkit for better Unicode input handling
try:
    from prompt_toolkit import prompt as pt_prompt  # noqa: F401
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

from sepilot.a2a.connector import A2AConnector
from sepilot.a2a.handoff import SessionHandoffManager

# A2A Protocol imports
from sepilot.a2a.router import A2ARouter
from sepilot.agent import state_helpers

# Static utility functions (extracted for modularity)
from sepilot.agent.agent_utils import (
    cache_entry_is_stale,
    canonicalize_args,
    check_task_completion,
    extract_paths_from_args,
    extract_required_files,
    extract_token_usage,
    format_tool_args,
    is_detailed_plan_response,
    is_plan_request,
    looks_like_plan,
    make_cache_entry,
    merge_updates,
    path_matches_requirement,
    safe_load_cache_value,
    should_skip_planning,
    truncate_text,
)
from sepilot.agent.approval_handler import ApprovalHandler, normalize_approval_response
from sepilot.agent.backtracking import BacktrackingManager, create_backtracking_manager
from sepilot.agent.debate_node import DebateOrchestrator, create_debate_orchestrator
from sepilot.agent.enhanced_state import (
    AgentMode,
    AgentStrategy,
    EnhancedAgentState,
    ErrorLevel,
    create_initial_state,
    state_to_summary,
)
from sepilot.agent.error_recovery import ErrorRecoveryStrategy
from sepilot.agent.execution_context import (
    find_execution_boundary,
    get_current_user_query,
    make_internal_human_message,
    make_user_prompt_message,
)
from sepilot.agent.file_detector import FilePathDetector, create_file_detector
from sepilot.agent.hierarchical_planner import HierarchicalPlanner, create_hierarchical_planner

# Advanced agent patterns
from sepilot.agent.hooks import HookManager, get_hook_manager
from sepilot.agent.instructions_loader import load_all_instructions
from sepilot.agent.memory_bank import MemoryBank, create_memory_bank
from sepilot.agent.mode_manager import (
    EXEC_MAX_ITERATIONS,
    ModeTransitionEngine,
    get_mode_filtered_tools,
    get_mode_prompt,
    mode_calibration_cap,
)

# Pattern nodes for LangGraph visualization
from sepilot.agent.pattern_nodes import (
    create_backtrack_check_node,
    create_codebase_exploration_node,
    create_debate_check_node,
    create_debate_node,
    create_hierarchical_planner_node,
    create_memory_retriever_node,
    create_memory_writer_node,
    create_orchestrator_node,
    create_tool_recommender_node,
    create_tool_recorder_node,
)
from sepilot.agent.pattern_orchestrator import (
    AdaptiveOrchestrator,
    create_adaptive_orchestrator,
)

# Self-reflection (Reflexion pattern)
from sepilot.agent.reflection_node import ReflectionDecision, create_reflection_node
from sepilot.agent.request_classifier import triage_prompt_for_tools
from sepilot.agent.rules_loader import get_rules_loader
from sepilot.agent.state_builder import StateUpdateBuilder
from sepilot.agent.subagent.worktree_manager import WorktreeManager

# File change tracking
from sepilot.agent.thread_manager import ThreadManager
from sepilot.agent.tool_call_fallback import try_parse_text_tool_calls

# Tool execution (extracted for modularity)
from sepilot.agent.tool_executor import create_enhanced_tool_node
from sepilot.agent.tool_learning import ToolLearningSystem, create_tool_learning_system
from sepilot.config.constants import (
    CONTEXT_COMPACT_THRESHOLD,
    CONTEXT_WARNING_THRESHOLD,
    RECURSION_DETECTOR_THRESHOLD,
    RECURSION_DETECTOR_WINDOW_SIZE,
)
from sepilot.config.settings import Settings
from sepilot.loggers.file_logger import FileLogger
from sepilot.prompts import load_prompt_profile
from sepilot.tools.codebase_tools import CodebaseExplorer
from sepilot.ui import ProgressDisplay, StatusPanel
from sepilot.utils.text import sanitize_text


class ReactAgent:
    """Agent using LangGraph standard pattern with ToolNode"""

    # Default cost per 1K tokens (can be overridden via settings.cost_per_1k_tokens)
    DEFAULT_COST_PER_1K_TOKENS = 0.01
    _CHECKPOINT_ALLOWED_MODULES = [
        ("sepilot.agent.enhanced_state", "TaskStatus"),
        ("sepilot.agent.enhanced_state", "TaskContext"),
        ("sepilot.agent.enhanced_state", "ToolCallRecord"),
        ("sepilot.agent.enhanced_state", "ErrorLevel"),
        ("sepilot.agent.enhanced_state", "ErrorRecord"),
        ("sepilot.agent.enhanced_state", "AgentMode"),
        ("sepilot.agent.enhanced_state", "AgentStrategy"),
    ]

    # Delegate to agent_utils functions for backward compatibility
    _truncate_text = staticmethod(truncate_text)
    _format_tool_args = staticmethod(format_tool_args)

    def _create_checkpoint_serde(self):
        """Create serializer allowlist for enhanced-state checkpoint types."""
        if not HAS_JSONPLUS_SERIALIZER:
            return None
        # Suppress langgraph serde deserialization warnings for our registered types
        import logging as _logging
        _logging.getLogger("langgraph.checkpoint.serde.jsonplus").setLevel(_logging.ERROR)
        try:
            return JsonPlusSerializer(
                allowed_json_modules=self._CHECKPOINT_ALLOWED_MODULES
            )
        except TypeError:
            # Fallback for older langgraph versions with different param names
            try:
                return JsonPlusSerializer(
                    allowed_msgpack_modules=self._CHECKPOINT_ALLOWED_MODULES
                )
            except Exception:
                return None

    def _verbose_llm_trace(
        self,
        node_name: str,
        messages: list,
        response,
        tokens: int = 0,
    ) -> None:
        """verbose 모드에서 LLM 호출의 입출력을 Rich Panel로 표시."""
        if not (self.console and self.verbose):
            return

        from rich.panel import Panel
        from rich.text import Text

        lines = Text()

        # 입력 요약: 시스템 프롬프트 + 메시지 구성
        for msg in messages:
            role = type(msg).__name__.replace("Message", "")
            content = getattr(msg, "content", "")
            preview = content[:120].replace("\n", " ")
            if len(content) > 120:
                preview += "..."
            lines.append(f"📥 {role}", style="bold cyan")
            lines.append(f" ({len(content)}c): {preview}\n")

        # 출력
        if hasattr(response, "content") and response.content:
            preview = response.content[:200].replace("\n", " ")
            if len(response.content) > 200:
                preview += "..."
            lines.append("📤 Response", style="bold green")
            lines.append(f": {preview}\n")

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                name = tc.get("name", "?")
                args = tc.get("args", {})
                args_str = self._format_tool_args(args, max_length=150)
                lines.append(f"🔧 Tool: {name}", style="bold yellow")
                lines.append(f"({args_str})\n")

        if tokens > 0:
            lines.append(f"📊 Tokens: {tokens:,}", style="dim")

        self.console.print(Panel(
            lines,
            title=f"[bold]🧠 {node_name}[/bold]",
            border_style="blue",
            padding=(0, 1),
        ))

    def _get_cost_per_1k_tokens(self) -> float:
        """Get cost per 1K tokens from settings or use default."""
        return getattr(self.settings, 'cost_per_1k_tokens', self.DEFAULT_COST_PER_1K_TOKENS)

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost for given token count."""
        return (tokens / 1000.0) * self._get_cost_per_1k_tokens()

    def __init__(self, settings: Settings, logger: FileLogger,
                 prompt_profile: str = "default",
                 thread_id: str | None = None,
                 enable_memory: bool = True,
                 auto_approve: bool = False,
                 show_progress: bool = True,
                 lazy_llm: bool = False):
        self.settings = settings
        self.logger = logger
        self.auto_approve = auto_approve  # Auto-approve sensitive tools (for interactive mode)
        self.auto_approve_session = False  # Session-scoped auto-approve (set by user in approval prompt)
        self.use_enhanced_state = True  # Enhanced state always enabled for CLI agent
        self.show_progress = show_progress
        self.verbose = settings.verbose
        self.status_indicator = None  # AgentStatusIndicator (set by UI layer)
        self.step_logger = None       # StepLogger (set by UI layer for interactive mode)
        # 콘솔을 먼저 초기화해 이후 초기화 경로에서 안전하게 사용
        self.console = Console() if settings.verbose else None

        # Sensitive tools requiring user approval (MUST be defined before wrapping tools)
        default_sensitive_tools = {"bash_execute", "file_write", "file_edit", "git", "web_search"}
        self.sensitive_tools = set(getattr(settings, 'sensitive_tools', default_sensitive_tools))

        # Get LangChain tools (needed for both eager and lazy init)
        from sepilot.tools.langchain_tools import get_all_tools
        self.langchain_tools = get_all_tools()

        # Load MCP tools for this agent
        try:
            from sepilot.mcp.integration import get_mcp_tools_for_agent
            agent_type = getattr(settings, 'agent_type', 'se')
            mcp_tools = get_mcp_tools_for_agent(agent_type)
            if mcp_tools:
                self.langchain_tools.extend(mcp_tools)
                if self.console:
                    self.console.print(
                        f"[dim cyan]🔌 Loaded {len(mcp_tools)} MCP tools for agent '{agent_type}'[/dim cyan]"
                    )
        except Exception as e:
            # MCP tool loading failure should not break agent initialization
            import logging
            logging.getLogger(__name__).warning(f"Failed to load MCP tools: {e}")

        # Initialize LLM (defer if lazy_llm=True for interactive mode)
        if lazy_llm:
            self.llm = None
            self.llm_with_tools = None
            self.is_thinking_model = False
            self.triage_llm = None
            self.verifier_llm = None
            self.reasoning_llm = None
            self.quick_llm = None
        else:
            self.llm = self._initialize_llm()
            # Initialize tier-specific LLMs (defaults to main LLM)
            self.triage_llm = self.llm
            self.verifier_llm = self.llm
            self.reasoning_llm = self.llm
            self.quick_llm = self.llm
            self._init_tier_llms()
            self._bind_tools_to_llm()

            # Propagate settings to SubAgent tools
            from sepilot.tools.langchain_tools.subagent_tools import set_current_settings
            set_current_settings(self.settings)

        self.iteration_count = 0

        # Progress tracking for checklist (persisted across conversation turns)
        self._progress_plan_steps: list[str] = []
        self._progress_planning_notes: list[str] = []
        self._progress_current_step: int = 0
        self._progress_current_task: dict[str, Any] | None = None

        # Agent Mode system (PLAN/CODE/EXEC)
        self.mode_engine = ModeTransitionEngine()
        self._mode_llm_cache: dict[str, Any] = {}
        self._pending_mode_update: dict | None = None
        # Persisted manual mode selection from slash commands (/plan, /code, /exec, /auto).
        # This is applied at the start of each execute() call so mode cannot drift due to
        # per-turn initial-state reset or stale checkpoint timing.
        self._session_mode_override: dict | None = None

        # Initialize UI components (only if verbose)
        self.progress_display: ProgressDisplay | None = None
        self.status_panel: StatusPanel | None = None
        if settings.verbose and self.console and show_progress:
            self.progress_display = ProgressDisplay(console=self.console)
            self.status_panel = StatusPanel(console=self.console)

        # Initialize recursion detector
        from sepilot.agent.recursion_detector import RecursionDetector
        self.recursion_detector = RecursionDetector(
            window_size=RECURSION_DETECTOR_WINDOW_SIZE,
            threshold=RECURSION_DETECTOR_THRESHOLD,
        )

        # Initialize tool call cache
        from sepilot.tools.tool_cache import ToolCallCache
        self.tool_cache = ToolCallCache(
            max_size=settings.tool_cache_size,
            ttl_seconds=settings.tool_cache_ttl
        )

        # Define cacheable tools (idempotent operations only)
        self.cacheable_tools = {
            'file_read', 'search_content', 'find_file', 'find_definition',
            'get_structure', 'codebase', 'list_directory', 'get_file_info'
        }

        # Reusable ContextManager (avoid re-creating on every iteration)
        from sepilot.agent.context_manager import ContextManager
        max_tokens = getattr(settings, 'context_window', None) or int(os.getenv('MAX_TOKENS', '96000'))
        self._context_manager = ContextManager(
            max_context_tokens=max_tokens,
            warning_threshold=CONTEXT_WARNING_THRESHOLD,
            compact_threshold=CONTEXT_COMPACT_THRESHOLD,
        )

        # Initialize memory monitor
        from sepilot.monitoring.memory_monitor import MemoryMonitor
        self.memory_monitor = MemoryMonitor(
            threshold_mb=getattr(settings, 'memory_threshold_mb', 500),
            use_psutil=True
        )
        self.memory_check_interval = getattr(settings, 'memory_check_interval', 10)

        # Initialize cost tracker
        from sepilot.monitoring.cost_tracker import get_cost_tracker
        self.cost_tracker = get_cost_tracker()

        # Load prompt template
        self.prompt_template = load_prompt_profile(prompt_profile)
        self.auto_finish_config = self.prompt_template.get_auto_finish_config() or {}
        self.logger.log_trace("prompt_profile", {"profile": prompt_profile})

        # Initialize LangGraph checkpoint system
        self.enable_memory = enable_memory
        self.thread_id = thread_id or f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        # Suppress legacy "Deserializing unregistered type" warnings from older langgraph
        import warnings
        warnings.filterwarnings("ignore", message=".*Deserializing unregistered type.*")
        warnings.filterwarnings("ignore", message=".*not in the deserialization allowlist.*")
        checkpoint_serde = self._create_checkpoint_serde()

        if self.enable_memory and HAS_SQLITE_SAVER:
            # Use LangGraph's SqliteSaver for persistent checkpoints
            checkpoint_dir = os.path.join(os.getcwd(), ".checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_db = os.path.join(checkpoint_dir, "checkpoints.db")

            try:
                if checkpoint_serde is not None:
                    cm = SqliteSaver.from_conn_string(
                        checkpoint_db,
                        serde=checkpoint_serde,
                    )
                else:
                    cm = SqliteSaver.from_conn_string(checkpoint_db)
            except TypeError:
                # Backward-compatible fallback for older LangGraph signatures
                cm = SqliteSaver.from_conn_string(checkpoint_db)

            # from_conn_string returns a context manager in newer langgraph versions.
            # Enter it to get the actual SqliteSaver instance (cleanup in self.cleanup()).
            if hasattr(cm, '__enter__'):
                self._checkpointer_cm = cm
                self.checkpointer = cm.__enter__()
            else:
                self._checkpointer_cm = None
                self.checkpointer = cm

            self.logger.log_trace("memory_initialized", {
                "thread_id": self.thread_id,
                "checkpoint_db": checkpoint_db,
                "memory_enabled": True,
                "checkpoint_type": "SqliteSaver"
            })
        else:
            # Use in-memory checkpointer (no persistence or SqliteSaver not available)
            self._checkpointer_cm = None
            if checkpoint_serde is not None:
                self.checkpointer = MemorySaver(serde=checkpoint_serde)
            else:
                self.checkpointer = MemorySaver()
            self.logger.log_trace("memory_initialized", {
                "thread_id": self.thread_id,
                "memory_enabled": self.enable_memory,
                "checkpoint_type": "MemorySaver",
                "note": "No persistence" if not HAS_SQLITE_SAVER else "Memory disabled"
            })

        # Initialize web monitoring (optional, set by web server)

        # File watcher for external change detection
        self.file_watcher = None
        try:
            from sepilot.agent.file_watcher import FileWatcher
            self.file_watcher = FileWatcher(os.getcwd())
            self.file_watcher.start()
        except Exception:
            pass  # watchdog not available or permission error

        # A2A Protocol Support
        self.a2a_router: A2ARouter | None = None
        self.a2a_connector: A2AConnector | None = None
        self.handoff_manager: SessionHandoffManager | None = None

        # Initialize Advanced Agent Patterns (skip for simplified graph mode)
        if getattr(self.settings, 'graph_mode', 'enhanced') == 'simplify':
            self._init_stub_patterns()
        else:
            self._init_agent_patterns()

        # Build the graph with checkpointer
        self.graph = self._build_graph()

        # SOLID: Approval handling (Single Responsibility)
        self._approval_handler = ApprovalHandler(
            console=self.console,
            sensitive_tools=self.sensitive_tools,
            auto_approve=self.auto_approve,
        )

        # SOLID: Thread/session management (Single Responsibility)
        self._thread_manager = ThreadManager(
            graph=self.graph,
            checkpointer=self.checkpointer,
            logger=self.logger,
            console=self.console,
            enable_memory=self.enable_memory,
        )
        self._thread_manager.thread_id = self.thread_id

    def _init_agent_patterns(self) -> None:
        """Initialize advanced agent patterns.

        Patterns initialized:
        - Memory Bank: Experience learning
        - Backtracking: State rollback
        - Hierarchical Planner: Multi-level planning
        - Tool Learning: Tool optimization
        - Debate Orchestrator: Multi-perspective analysis
        - Adaptive Orchestrator: Pattern selection
        """
        # Memory Bank - persistent experience learning
        self.memory_bank: MemoryBank = create_memory_bank(
            storage_path=Path.home() / ".sepilot" / "memories",
            project_id=None  # Auto-computed from cwd
        )

        # Backtracking - state rollback on failure
        self.backtracking: BacktrackingManager = create_backtracking_manager(
            working_directory=os.getcwd(),
            enable_git_tracking=True,
            console=self.console,
            verbose=self.verbose
        )

        # Hierarchical Planner - multi-level task decomposition
        self.hierarchical_planner: HierarchicalPlanner = create_hierarchical_planner(
            llm=self.reasoning_llm,
            console=self.console,
            verbose=self.verbose
        )

        # Tool Learning - tool usage optimization
        self.tool_learning: ToolLearningSystem = create_tool_learning_system(
            storage_path=Path.home() / ".sepilot" / "tool_learning",
            project_id=None  # Auto-computed
        )

        # Debate Orchestrator - multi-perspective analysis
        self.debate_orchestrator: DebateOrchestrator = create_debate_orchestrator(
            llm=self.reasoning_llm,
            console=self.console,
            verbose=self.verbose
        )

        # Adaptive Orchestrator - pattern selection
        self.pattern_orchestrator: AdaptiveOrchestrator = create_adaptive_orchestrator(
            llm=self.llm,
            console=self.console,
            verbose=self.verbose
        )

        # File Path Detector - detect if exploration is needed
        self.file_detector: FilePathDetector = create_file_detector(
            project_root=Path(os.getcwd())
        )

        # Codebase Explorer - automatic file discovery
        self.codebase_explorer: CodebaseExplorer = CodebaseExplorer(
            logger=self.logger
        )

        # Hook Manager - pre/post tool execution hooks
        self.hook_manager: HookManager = get_hook_manager()

        # Rules Loader - path-based conditional rules
        self.rules_loader = get_rules_loader(project_root=Path(os.getcwd()))

        # Worktree Manager - git worktree isolation for subagents
        self.worktree_manager: WorktreeManager = WorktreeManager(
            repo_root=Path(os.getcwd())
        )

        if self.console and self.verbose:
            self.console.print(
                "[dim cyan]🧠 Advanced agent patterns initialized: "
                "Memory, Backtracking, Hierarchical Planning, Tool Learning, Debate, Exploration, Hooks, Rules[/dim cyan]"
            )

    def _init_stub_patterns(self) -> None:
        """Stub pattern attributes for simplified graph mode.

        Sets None placeholders so attribute access in update_llm() and other
        methods doesn't crash, but avoids heavy initialization overhead.
        """
        self.memory_bank = None
        self.backtracking = None
        self.hierarchical_planner = None
        self.tool_learning = None
        self.debate_orchestrator = None
        self.pattern_orchestrator = None
        self.file_detector = None
        self.codebase_explorer = None
        self._reflection_node = None
        # Hooks and rules are lightweight - always initialize
        self.hook_manager: HookManager = get_hook_manager()
        self.rules_loader = get_rules_loader(project_root=Path(os.getcwd()))
        # Worktree Manager is lightweight - always initialize
        self.worktree_manager: WorktreeManager = WorktreeManager(
            repo_root=Path(os.getcwd())
        )

    def update_llm(self, new_llm, updated_settings: Settings | None = None) -> None:
        """Propagate LLM update to ALL dependent systems.

        Called by /model apply to ensure complex question paths
        (triage, verifier, planner, debate, reflection, etc.)
        all use the newly configured LLM.

        Args:
            new_llm: New LangChain LLM instance
            updated_settings: Updated settings (for subagent propagation)
        """
        # Core LLM
        self.llm = new_llm
        self._bind_tools_to_llm()
        self._mode_llm_cache.clear()  # Invalidate mode-filtered LLM cache

        # Reset all tier LLMs to main, then re-initialize from settings
        self.triage_llm = new_llm
        self.verifier_llm = new_llm
        self.reasoning_llm = new_llm
        self.quick_llm = new_llm

        # Update settings if provided (before _init_tier_llms so it reads new config)
        if updated_settings:
            self.settings = updated_settings

        # Re-initialize tier LLMs from settings (restores separate models if configured)
        self._init_tier_llms()

        # Propagate reasoning_llm to planning/debate/reflection components
        if hasattr(self, 'hierarchical_planner') and self.hierarchical_planner:
            self.hierarchical_planner.llm = self.reasoning_llm

        if hasattr(self, 'debate_orchestrator') and self.debate_orchestrator:
            self.debate_orchestrator.proposer.llm = self.reasoning_llm
            self.debate_orchestrator.critic.llm = self.reasoning_llm
            self.debate_orchestrator.resolver.llm = self.reasoning_llm

        # Pattern Orchestrator
        if hasattr(self, 'pattern_orchestrator') and self.pattern_orchestrator:
            self.pattern_orchestrator.llm = new_llm

        # Reflection Node (created during graph build)
        if hasattr(self, '_reflection_node') and self._reflection_node:
            self._reflection_node.llm = self.reasoning_llm

        # Propagate settings to SubAgent tools
        from sepilot.tools.langchain_tools.subagent_tools import set_current_settings
        set_current_settings(updated_settings or self.settings)

        import logging
        logging.getLogger(__name__).debug("LLM propagated to all dependent systems")

    # Delegate to agent_utils functions for cache and state management
    _merge_updates = staticmethod(merge_updates)
    _canonicalize_args = staticmethod(canonicalize_args)
    _extract_paths_from_args = staticmethod(extract_paths_from_args)
    _cache_entry_is_stale = staticmethod(cache_entry_is_stale)
    _make_cache_entry = staticmethod(make_cache_entry)
    _safe_load_cache_value = staticmethod(safe_load_cache_value)

    def _collapse_deltas(self, deltas: list[dict[str, Any]]) -> dict[str, Any]:
        """Collapse a list of deltas into a single update dict."""
        merged: dict[str, Any] = {}
        for delta in deltas:
            merged = merge_updates(merged, delta)
        return merged

    def _snapshot_workspace(self) -> dict[str, Any]:
        """Lightweight workspace snapshot (git-aware, ignores heavy dirs)."""
        root = Path(os.getcwd())
        files: dict[str, Any] = {}
        ignore_dirs = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", "logs", "temp"}

        def should_ignore(path: Path) -> bool:
            parts = set(path.parts)
            return bool(parts & ignore_dirs)

        try:
            if (root / ".git").exists():
                import subprocess
                proc = subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=all"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                for line in proc.stdout.splitlines():
                    if not line.strip():
                        continue
                    # Format: XY<space>path or XY<space>old -> new
                    path_str = line[3:].split(" -> ")[-1].strip()
                    path_obj = (root / path_str).resolve()
                    try:
                        rel_obj = path_obj.relative_to(root)
                    except ValueError:
                        continue  # 밖의 경로는 무시
                    if should_ignore(rel_obj):
                        continue
                    if path_obj.is_file():
                        stat = path_obj.stat()
                        files[str(rel_obj)] = {"size": stat.st_size, "mtime": stat.st_mtime}
                return files
        except Exception:
            # Fall back to filesystem walk
            pass

        try:
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                rel_path = file_path.relative_to(root)
                if should_ignore(rel_path):
                    continue
                st = file_path.stat()
                files[str(rel_path)] = {
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                }
        except Exception:
            return files

        return files

    # Delegate to agent_utils for approval handling
    _normalize_approval_response = staticmethod(normalize_approval_response)

    def _handle_interrupt_events(self, interrupts) -> Any:
        """Handle LangGraph interrupt tuples - delegates to ApprovalHandler."""
        # Sync auto_approve_session state
        self._approval_handler.auto_approve_session = self.auto_approve_session
        result = self._approval_handler.handle_interrupt_events(interrupts)
        # Sync back in case it was changed
        self.auto_approve_session = self._approval_handler.auto_approve_session
        return result

    def _assess_tool_risk(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Assess tool risk - delegates to ApprovalHandler."""
        return self._approval_handler.assess_tool_risk(tool_name, args)

    def _invoke_with_interrupts(self, initial_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Execute the graph, resuming automatically after human interrupts.

        Uses stream() to detect interrupts (default stream_mode shows interrupts).
        After completion, uses get_state() to retrieve final state.
        """
        # Max interrupts from settings (default: 50, reasonable limit for human-in-the-loop)
        max_interrupts = getattr(self.settings, 'max_interrupts', 50)  # type: int
        interrupt_count = 0
        next_input: Any = initial_input
        # Progress tracking uses instance variables (persisted across conversation turns)

        while True:
            # Use stream() with default mode (updates) to detect interrupts
            interrupt_found = False

            for chunk in self.graph.stream(next_input, config=config):
                # Update status indicator for node transitions
                if self.status_indicator:
                    for key in chunk:
                        if key != "__interrupt__":
                            self.status_indicator.update_for_node(key, chunk.get(key))

                # Step logger: node transitions
                if self.step_logger:
                    for key in chunk:
                        if key not in ("__interrupt__", "__start__", "__end__"):
                            self.step_logger.log_node(key)
                            payload = chunk.get(key)
                            if isinstance(payload, dict):
                                if isinstance(payload.get("plan_steps"), list):
                                    self._progress_plan_steps = payload.get("plan_steps", [])
                                if isinstance(payload.get("planning_notes"), list):
                                    self._progress_planning_notes = payload.get("planning_notes", [])
                                if payload.get("current_plan_step") is not None:
                                    try:
                                        self._progress_current_step = int(payload.get("current_plan_step"))
                                    except (TypeError, ValueError):
                                        pass
                                if isinstance(payload.get("current_task"), dict):
                                    self._progress_current_task = payload.get("current_task")

                            self.step_logger.log_plan(
                                self._progress_plan_steps,
                                self._progress_planning_notes,
                            )
                            self.step_logger.log_plan_progress(
                                self._progress_plan_steps,
                                self._progress_current_step,
                                self._progress_current_task,
                                self._progress_planning_notes,
                            )

                # --- Real-time node tracking ---
                # Record completed node + signal next nodes as "executing"
                # Check if this chunk contains an interrupt
                if "__interrupt__" in chunk:
                    # Determine whether this is an auto-approved interrupt
                    # If auto_approve or auto_approve_session is enabled, don't count toward limit
                    is_auto_approved = self.auto_approve or self.auto_approve_session

                    # Also check if all interrupt events are simple tool-approval requests
                    interrupt_events = chunk["__interrupt__"]
                    is_approval_only = False
                    try:
                        events = interrupt_events if isinstance(interrupt_events, list) else [interrupt_events]
                        is_approval_only = all(
                            isinstance(ev, dict) and ev.get("type") == "tool_approval"
                            or hasattr(ev, "value") and isinstance(ev.value, dict) and ev.value.get("type") == "tool_approval"
                            for ev in events
                        )
                    except Exception:
                        is_approval_only = False

                    # Don't count if auto-approved or approval-only interrupts
                    should_count = not is_auto_approved and not is_approval_only

                    if should_count:
                        interrupt_count += 1
                        if interrupt_count > max_interrupts:
                            error_msg = f"Too many interrupts ({interrupt_count}/{max_interrupts}). Stopping execution."
                            self.logger.log_error(error_msg)
                            if self.console:
                                self.console.print(f"[bold red]❌ {error_msg}[/bold red]")
                            # Graceful termination: return current state instead of raising error
                            # This allows the caller to see what was accomplished
                            try:
                                final_state = self.graph.get_state(config)
                                state_values = final_state.values if final_state else {}
                                state_values["error"] = error_msg
                                state_values["terminated_early"] = True
                                state_values["interrupt_count"] = interrupt_count
                                if self.console:
                                    self.console.print("[yellow]📋 Returning partial results from interrupted execution[/yellow]")
                                return state_values
                            except Exception as exc:
                                raise RuntimeError(error_msg) from exc

                    if self.console and should_count:
                        self.console.print(f"[bold magenta]🛑 Human-in-the-loop #{interrupt_count} · 승인 대기[/bold magenta]")

                    # Pause indicator during user interaction
                    if self.status_indicator:
                        self.status_indicator.stop()

                    # Handle the interrupt and get user response
                    resume_value = self._handle_interrupt_events(chunk["__interrupt__"])

                    if self.console:
                        self.console.print("[green]✓ Interrupt resolved, resuming execution[/green]")

                    # Resume indicator after interrupt
                    if self.status_indicator:
                        self.status_indicator.start("Resuming...", reset_metrics=False)

                    # Resume with Command - enhanced with state update
                    # Track approval in state for better context
                    state_update = {}
                    if isinstance(resume_value, dict):
                        if resume_value.get("status") == "approved":
                            state_update["last_approval_status"] = "approved"
                            state_update["approval_count"] = 1  # Will be added by reducer
                        elif resume_value.get("status") == "rejected":
                            state_update["last_approval_status"] = "rejected"

                        # Extract feedback if provided
                        if resume_value.get("message"):
                            state_update["user_feedback"] = [resume_value.get("message")]

                    # Use Command with state update
                    if state_update:
                        next_input = Command(resume=resume_value, update=state_update)
                    else:
                        next_input = Command(resume=resume_value)

                    interrupt_found = True
                    break  # Exit stream loop to resume

            if not interrupt_found:
                # No interrupt found - execution completed
                # Get final state from graph (contains full state with all messages)
                final_state = self.graph.get_state(config)
                return final_state.values if final_state else {}

    def _display_session_metadata(self, state: EnhancedAgentState) -> None:
        """Show planning and verification notes in verbose mode."""
        planning_notes = state.get("planning_notes", [])
        verification_notes = state.get("verification_notes", [])

        if planning_notes:
            self.logger.log_trace("planning_notes", planning_notes)
        if verification_notes:
            self.logger.log_trace("verification_notes", verification_notes)

        if not self.console:
            return

        if planning_notes:
            self.console.print("\n[bold magenta]🧭 Execution Plan[/bold magenta]")
            for idx, note in enumerate(planning_notes[-3:], 1):
                self.console.print(f"[dim]{idx}. {self._truncate_text(note, 400)}[/dim]")

        if verification_notes:
            self.console.print("\n[bold cyan]🧪 Verification[/bold cyan]")
            for idx, note in enumerate(verification_notes[-5:], 1):
                self.console.print(f"[cyan]{idx}. {note}[/cyan]")

    # Regex patterns for thinking/reasoning models that can't use bind_tools.
    # Uses word-boundary-aware matching to avoid false positives (e.g., 'modelo3' won't match 'o3').
    _THINKING_MODEL_RE = re.compile(
        r'deepseek-r1|'
        r'\bqwq\b|'
        r'qwen-think|'
        r'\bo1-|'
        r'\bo3\b|'          # matches 'o3', 'o3-mini' (word boundary before, boundary/hyphen after)
        r'\bo3-|'
        r'\breasoning\b|'
        r'deepseek.*think',
        re.IGNORECASE,
    )

    @staticmethod
    def _check_thinking_model(llm) -> bool:
        """Check if a given LLM is a thinking/reasoning model that can't bind_tools."""
        # CLI agents cannot bind_tools — treat them like thinking models
        if getattr(llm, '_llm_type', '') == 'cli_agent':
            return True
        model_name = str(getattr(llm, 'model_name', None) or getattr(llm, 'model', '') or '').lower()
        return bool(ReactAgent._THINKING_MODEL_RE.search(model_name))

    def _bind_tools_to_llm(self):
        """Bind LangChain tools to LLM and detect thinking models."""
        self.is_thinking_model = self._check_thinking_model(self.llm)

        if self.is_thinking_model:
            # CLI agents: pass tool schemas for prompt injection
            if getattr(self.llm, '_llm_type', '') == 'cli_agent':
                self.llm.tool_schemas = self.langchain_tools
                if self.console:
                    self.console.print(
                        f"[cyan]🔌 CLI Agent mode: {self.llm.cli_command} (tools via text fallback)[/cyan]"
                    )
            elif self.console:
                self.console.print(
                    "[yellow]⚠️  Thinking/reasoning model detected. Tool calling may be limited.[/yellow]"
                )
            self.llm_with_tools = self.llm
        else:
            self.llm_with_tools = self.llm.bind_tools(self.langchain_tools)

    def _get_mode_filtered_llm(self, mode: AgentMode, task_type: str | None = None):
        """Get LLM bound with mode-filtered tools. Results are cached per mode.

        Args:
            mode: Current agent mode
            task_type: Optional task type for tier routing ("reasoning", "quick")

        Returns:
            LLM instance with mode-appropriate tools bound
        """
        # Select base LLM based on task_type or mode
        if task_type == "reasoning":
            base_llm = self.reasoning_llm
        elif task_type == "quick":
            base_llm = self.quick_llm
        elif mode == AgentMode.PLAN:
            base_llm = self.reasoning_llm
        else:
            base_llm = self.llm

        # Fast path: AUTO mode with default LLM
        if mode == AgentMode.AUTO and base_llm is self.llm:
            return self.llm_with_tools

        # Thinking models can't bind_tools — rely on runtime filtering in tool_executor
        if self._check_thinking_model(base_llm):
            return base_llm

        prompt_profile = getattr(self, "prompt_profile", "default")
        cache_key = f"{mode.value}_{task_type or 'default'}_{prompt_profile}_{id(base_llm)}"
        if cache_key in self._mode_llm_cache:
            return self._mode_llm_cache[cache_key]

        filtered_tools = get_mode_filtered_tools(self.langchain_tools, mode, prompt_profile)
        if not filtered_tools:
            # No tool filtering needed — bind all tools if not the main LLM
            if base_llm is self.llm:
                return self.llm_with_tools
            bound = base_llm.bind_tools(self.langchain_tools)
            self._mode_llm_cache[cache_key] = bound
            return bound

        bound = base_llm.bind_tools(filtered_tools)
        self._mode_llm_cache[cache_key] = bound
        return bound

    def initialize_llm_from_config(self, base_url=None, model=None, api_key=None, **kwargs):
        """Initialize LLM from external config (for interactive mode /model apply).

        Args:
            base_url: OpenAI-compatible API base URL
            model: Model name
            api_key: API key
            **kwargs: Additional ChatOpenAI parameters (temperature, max_tokens, etc.)
        """
        from langchain_openai import ChatOpenAI

        llm_params = {}
        if model:
            llm_params['model'] = model
            self.settings.model = model
        else:
            llm_params['model'] = self.settings.model

        if api_key:
            llm_params['openai_api_key'] = api_key
        if base_url:
            from sepilot.config.llm_providers import _ensure_versioned_base_url
            llm_params['openai_api_base'] = _ensure_versioned_base_url(base_url)
        llm_params.update(kwargs)

        self.llm = ChatOpenAI(**llm_params)
        self._bind_tools_to_llm()

    def _initialize_llm(self):
        """Initialize the language model using LLM Provider Factory.

        Supports 9 providers:
        - OpenAI, Anthropic, Google, Ollama (original)
        - AWS Bedrock, Azure OpenAI, OpenRouter, Groq, GitHub Models (new)
        """
        from sepilot.config.llm_providers import LLMProviderError, LLMProviderFactory

        factory = LLMProviderFactory(self.settings, self.console)

        try:
            return factory.create_llm()
        except LLMProviderError as e:
            if e.suggestion:
                raise ValueError(f"{e}\nTip: {e.suggestion}") from e
            raise ValueError(str(e)) from e

    def _init_tier_llms(self) -> None:
        """Initialize tier-specific LLMs for classification tasks.

        Uses cheaper models (e.g. gpt-4o-mini) for triage and verification
        when configured, falling back to the main LLM if not set.
        """
        from sepilot.config.llm_providers import LLMProviderFactory

        factory = LLMProviderFactory(self.settings, self.console)

        if getattr(self.settings, 'triage_model', None):
            try:
                self.triage_llm = factory.create_llm(self.settings.triage_model)
                if self.console and self.verbose:
                    self.console.print(
                        f"[dim cyan]🏷️ Triage model: {self.settings.triage_model}[/dim cyan]"
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Tier LLM init failed for triage model '{self.settings.triage_model}': {e}. Using main LLM."
                )
                self.triage_llm = self.llm

        if getattr(self.settings, 'verifier_model', None):
            try:
                self.verifier_llm = factory.create_llm(self.settings.verifier_model)
                if self.console and self.verbose:
                    self.console.print(
                        f"[dim cyan]✅ Verifier model: {self.settings.verifier_model}[/dim cyan]"
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Tier LLM init failed for verifier model '{self.settings.verifier_model}': {e}. Using main LLM."
                )
                self.verifier_llm = self.llm

        if getattr(self.settings, 'reasoning_model', None):
            try:
                self.reasoning_llm = factory.create_llm(self.settings.reasoning_model)
                if self.console and self.verbose:
                    self.console.print(
                        f"[dim cyan]🧠 Reasoning model: {self.settings.reasoning_model}[/dim cyan]"
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Tier LLM init failed for reasoning model '{self.settings.reasoning_model}': {e}. Using main LLM."
                )
                self.reasoning_llm = self.llm

        if getattr(self.settings, 'quick_model', None):
            try:
                self.quick_llm = factory.create_llm(self.settings.quick_model)
                if self.console and self.verbose:
                    self.console.print(
                        f"[dim cyan]⚡ Quick model: {self.settings.quick_model}[/dim cyan]"
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Tier LLM init failed for quick model '{self.settings.quick_model}': {e}. Using main LLM."
                )
                self.quick_llm = self.llm

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow based on graph_mode setting."""
        if getattr(self.settings, 'graph_mode', 'enhanced') == 'simplify':
            return self._build_simplified_graph()
        return self._build_enhanced_graph()

    def _build_simplified_graph(self) -> StateGraph:
        """Minimal agent loop (Claude Code style) — 9 nodes.

        Designed for slow-inference / rate-limited environments.
        Skips: orchestrator, memory, planner, tool_recommender, tool_recorder,
               reflection, backtrack, debate, codebase_exploration.

        Graph structure:
            triage
              ├─→ direct_response → END
              └─→ iteration_guard → context_manager → agent
                                                        │
                              ┌─────────────────────────┴──────────┐
                              ↓                                    ↓
                          approval → tools → verifier         verifier
                              │                │                   │
                              └────────────────┴───────────────────↓
                                                iteration_guard ←→ reporter → END
        """
        workflow = StateGraph(EnhancedAgentState)

        # Tool node (reuse existing enhanced tool executor)
        enhanced_tool_node = create_enhanced_tool_node(self)

        # ===== 9 nodes only =====
        workflow.add_node("triage", self._triage_node)
        workflow.add_node("direct_response", self._direct_response_node)
        workflow.add_node("iteration_guard", self._iteration_guard_node)
        workflow.add_node("context_manager", self._context_manager_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("approval", self._approval_node)
        workflow.add_node("tools", enhanced_tool_node)
        workflow.add_node("verifier", self._verification_node)
        workflow.add_node("reporter", self._reporting_node)

        # ===== Edges =====
        workflow.set_entry_point("triage")

        # Triage: 2-way split (no orchestrator distinction)
        workflow.add_conditional_edges(
            "triage",
            self._triage_next_step_simplified,
            {"direct": "direct_response", "execute": "iteration_guard"}
        )
        workflow.add_edge("direct_response", END)

        # Iteration guard: stop → reporter directly (no memory_writer)
        workflow.add_conditional_edges(
            "iteration_guard",
            self._guard_decision,
            {"continue": "context_manager", "stop": "reporter"}
        )

        # Core loop: no tool_recommender
        workflow.add_edge("context_manager", "agent")

        # Agent routing (reuse existing)
        workflow.add_conditional_edges(
            "agent",
            self._agent_next_step,
            {"tools": "approval", "finalize": "verifier"}
        )

        # Approval routing (reuse existing)
        workflow.add_conditional_edges(
            "approval",
            self._approval_next_step,
            {"run_tools": "tools", "retry": "iteration_guard"}
        )

        # Tools → verifier (no tool_recorder)
        workflow.add_edge("tools", "verifier")

        # Verifier: simplified — no reflection/backtrack/debate pipeline
        workflow.add_conditional_edges(
            "verifier",
            self._verification_next_step_simplified,
            {"continue": "iteration_guard", "report": "reporter"}
        )

        # Exit
        workflow.add_edge("reporter", END)

        if self.console and self.verbose:
            self.console.print(
                "[dim cyan]⚡ Simplified graph mode: 9 nodes "
                "(triage → agent → tools loop)[/dim cyan]"
            )

        return workflow.compile(checkpointer=self.checkpointer)

    def _build_enhanced_graph(self) -> StateGraph:
        """Full agent pipeline with all pattern nodes for visualization.

        Graph Structure (17 nodes):
        ```
        triage
          ├─→ direct_response → END
          └─→ orchestrator → memory_retriever → hierarchical_planner → iteration_guard
                                                                           │
                          ┌────────────────────────────────────────────────┘
                          ↓
                   context_manager → tool_recommender → agent
                                                          │
                          ┌───────────────────────────────┴──────────────────┐
                          ↓                                                  ↓
                      approval → tools → verifier                   verifier (finalize)
                          │                │                              │
                          │                └──────────────────────────────┤
                          │                                               ↓
                          │                                          reflection
                          │                                               │
                          │                     ┌─────────────────────────┼─────────────────────┐
                          │                     ↓                         ↓                     ↓
                          │              revise_plan              backtrack_check           escalate
                          │           (→hier_planner)                   │               (→memory_writer)
                          │                                             ↓
                          │                                       debate_check
                          │                                       ┌─────┴─────┐
                          │                                       ↓           ↓
                          │                                    debate    (skip debate)
                          │                                       │           │
                          │                                       └─────┬─────┘
                          │                                             ↓
                          │                                      memory_writer
                          │                                             │
                          └─────────────────────────────────────────────┤
                                                                        ↓
                                                      ┌─────────────────┴─────────────────┐
                                                      ↓                                   ↓
                                               iteration_guard                       reporter → END
        ```

        Patterns integrated:
        - Orchestrator: Task analysis and pattern selection
        - Memory Retriever: Experience recall before planning
        - Hierarchical Planner: Multi-level task decomposition
        - Tool Recommender: Learned tool suggestions
        - Reflection: Self-critique and improvement
        - Backtrack Check: Rollback decision on failure
        - Debate: Multi-perspective analysis for reviews
        - Memory Writer: Experience storage for learning
        """
        workflow = StateGraph(EnhancedAgentState)

        # Create enhanced tool node
        enhanced_tool_node = create_enhanced_tool_node(self)

        # Create reflection node (store for LLM propagation)
        self._reflection_node = create_reflection_node(
            llm=self.reasoning_llm,
            console=self.console,
            verbose=self.verbose,
            logger=self.logger
        )
        reflection_node = self._reflection_node

        # Create pattern nodes
        orchestrator_node = create_orchestrator_node(
            self.pattern_orchestrator, self.console, self.verbose
        )
        memory_retriever_node = create_memory_retriever_node(
            self.memory_bank, self.console, self.verbose
        )
        hierarchical_planner_node = create_hierarchical_planner_node(
            self.hierarchical_planner, self.console, self.verbose
        )
        tool_recommender_node = create_tool_recommender_node(
            self.tool_learning, self.console, self.verbose
        )
        backtrack_check_node = create_backtrack_check_node(
            self.backtracking, self.console, self.verbose
        )
        debate_check_node = create_debate_check_node(
            self.console, self.verbose
        )
        debate_node = create_debate_node(
            self.debate_orchestrator, self.console, self.verbose
        )
        memory_writer_node = create_memory_writer_node(
            self.memory_bank, self.console, self.verbose
        )
        tool_recorder_node = create_tool_recorder_node(
            self.tool_learning, self.console, self.verbose
        )
        codebase_exploration_node = create_codebase_exploration_node(
            self.file_detector, self.codebase_explorer, self.console, self.verbose
        )

        # ===== Add all nodes =====
        # Entry nodes
        workflow.add_node("triage", self._triage_node)
        workflow.add_node("direct_response", self._direct_response_node)

        # Pattern nodes - Pre-execution
        workflow.add_node("orchestrator", orchestrator_node)
        workflow.add_node("codebase_exploration", codebase_exploration_node)
        workflow.add_node("memory_retriever", memory_retriever_node)
        workflow.add_node("hierarchical_planner", hierarchical_planner_node)

        # Core execution nodes
        workflow.add_node("iteration_guard", self._iteration_guard_node)
        workflow.add_node("context_manager", self._context_manager_node)
        workflow.add_node("tool_recommender", tool_recommender_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("approval", self._approval_node)
        workflow.add_node("tools", enhanced_tool_node)
        workflow.add_node("tool_recorder", tool_recorder_node)
        workflow.add_node("verifier", self._verification_node)

        # Pattern nodes - Post-execution
        workflow.add_node("reflection", reflection_node)
        workflow.add_node("backtrack_check", backtrack_check_node)
        workflow.add_node("debate_check", debate_check_node)
        workflow.add_node("debate", debate_node)
        workflow.add_node("memory_writer", memory_writer_node)

        # Exit node
        workflow.add_node("reporter", self._reporting_node)

        # ===== Define edges =====
        workflow.set_entry_point("triage")

        # Triage routing (Claude Code style: 3-way split)
        # - direct: Simple greetings/questions → direct response
        # - simple: File reads, simple modifications → skip orchestrator/memory/planner
        # - complex: Multi-step tasks → full pipeline
        workflow.add_conditional_edges(
            "triage",
            self._triage_next_step,
            {
                "direct": "direct_response",
                "simple": "iteration_guard",  # Skip orchestrator, memory, planner
                "complex": "orchestrator"      # Full pipeline
            }
        )
        workflow.add_edge("direct_response", END)

        # Pattern pre-execution flow (only for complex tasks)
        # orchestrator → [codebase_exploration, memory_retriever] → hierarchical_planner
        # Fan-out: both nodes run in parallel after orchestrator
        # Fan-in: planner waits for both to complete (LangGraph auto-join)
        workflow.add_edge("orchestrator", "codebase_exploration")
        workflow.add_edge("orchestrator", "memory_retriever")
        workflow.add_edge("codebase_exploration", "hierarchical_planner")
        workflow.add_edge("memory_retriever", "hierarchical_planner")
        workflow.add_edge("hierarchical_planner", "iteration_guard")

        # Iteration guard routing
        workflow.add_conditional_edges(
            "iteration_guard",
            self._guard_decision,
            {"continue": "context_manager", "stop": "memory_writer"}
        )

        # Core execution flow
        workflow.add_edge("context_manager", "tool_recommender")
        workflow.add_edge("tool_recommender", "agent")

        # Agent routing
        workflow.add_conditional_edges(
            "agent",
            self._agent_next_step,
            {"tools": "approval", "finalize": "verifier"}
        )

        # Approval routing
        workflow.add_conditional_edges(
            "approval",
            self._approval_next_step,
            {"run_tools": "tools", "retry": "iteration_guard"}
        )

        # Tool execution flow
        workflow.add_edge("tools", "tool_recorder")
        workflow.add_edge("tool_recorder", "verifier")

        # Verifier routing
        workflow.add_conditional_edges(
            "verifier",
            self._verification_next_step,
            {
                "continue": "reflection",
                "fast_continue": "memory_writer",  # Skip reflection/backtrack/debate in early iterations
                "report": "memory_writer",
            }
        )

        # Reflection routing
        workflow.add_conditional_edges(
            "reflection",
            self._reflection_next_step,
            {
                "revise_plan": "hierarchical_planner",
                "refine_strategy": "agent",
                "proceed": "backtrack_check",
                "escalate": "memory_writer"
            }
        )

        # Backtrack check routing
        workflow.add_conditional_edges(
            "backtrack_check",
            self._backtrack_next_step,
            {"rollback": "hierarchical_planner", "continue": "debate_check"}
        )

        # Debate check routing
        workflow.add_conditional_edges(
            "debate_check",
            self._debate_check_next_step,
            {"debate": "debate", "skip": "memory_writer"}
        )

        # Debate flow
        workflow.add_edge("debate", "memory_writer")

        # Memory writer routing (final decision point)
        workflow.add_conditional_edges(
            "memory_writer",
            self._memory_writer_next_step,
            {"continue": "iteration_guard", "report": "reporter"}
        )

        # Exit
        workflow.add_edge("reporter", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _triage_node(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Decide routing: direct response, simple task, or complex task.

        Claude Code style: 3-way routing for efficiency
        - direct: Greetings, simple questions → no tools
        - simple: File reads, simple edits → skip orchestrator/memory/planner
        - complex: Multi-step tasks → full pipeline
        """
        # Get user prompt from CURRENT execution only.
        user_prompt = get_current_user_query(state)[:2000]

        # Unified LLM-based triage: intent + complexity in a single call
        try:
            triage_result = triage_prompt_for_tools(
                user_prompt,
                self.auto_finish_config,
                llm=self.triage_llm
            )
        except Exception as e:
            self.logger.log_trace("triage_error", {"error": str(e)})
            # Fallback: treat as simple tool-use task
            from sepilot.agent.request_classifier import PromptTriageResult
            triage_result = PromptTriageResult(
                decision="use_tools",
                complexity="simple",
                reason=f"Triage fallback due to error: {type(e).__name__}"
            )

        # Complexity is already determined by the unified triage call
        complexity = triage_result.complexity

        self.logger.log_trace("triage_decision", {
            "decision": triage_result.decision,
            "complexity": complexity,
            "reason": triage_result.reason
        })

        if self.console and self.verbose:
            if triage_result.decision == "direct_response":
                self.console.print("[dim cyan]⚡️ Direct response (no tools)[/dim cyan]")
            elif complexity == "simple":
                self.console.print("[dim cyan]⚡️ Simple task → fast path (skip planning)[/dim cyan]")
            else:
                self.console.print("[dim cyan]📋 Complex task → full pipeline[/dim cyan]")
            self.console.print(
                f"[dim]   Reason: {triage_result.reason} | "
                f"Complexity: {complexity}[/dim]"
            )

        # Claude Code style: Set default active_patterns for simple tasks
        # This ensures memory_bank is active even for simple tasks
        # Complex tasks get patterns set by orchestrator
        default_patterns = []
        if complexity == "simple" and triage_result.decision != "direct_response":
            default_patterns = ["memory_bank"]

        pending_mode_update = None
        current_pending_mode_update = getattr(self, "_pending_mode_update", None)
        if current_pending_mode_update:
            pending_mode_update = dict(current_pending_mode_update)
            self._pending_mode_update = None

        builder = (StateUpdateBuilder()
            .set_decision(triage_result.decision, triage_result.reason)
            .merge({"task_complexity": complexity}))

        # Store classified strategy in state
        if hasattr(triage_result, 'strategy') and triage_result.strategy:
            builder.merge({"current_strategy": AgentStrategy(triage_result.strategy)})

        if default_patterns:
            builder.merge({"active_patterns": default_patterns})

        if pending_mode_update:
            builder.merge(pending_mode_update)

        # Simple path skips orchestrator/planner — explicitly initialize plan state
        if complexity == "simple" and triage_result.decision != "direct_response":
            builder.merge({
                "plan_created": False,
                "plan_steps": [],
                "current_plan_step": 0,
                "plan_execution_pending": False,
            })

        # Initialize Agent Mode:
        # - Default to AUTO for all profiles so modification tasks do not get
        #   stuck in read-only PLAN mode on first turn.
        # - Users can still explicitly lock PLAN/CODE/EXEC via slash commands.
        effective_mode_locked = bool(
            (pending_mode_update or {}).get("mode_locked", state.get("mode_locked", False))
        )
        if triage_result.decision != "direct_response" and not effective_mode_locked:
            builder.merge({
                "current_mode": AgentMode.AUTO,
                "mode_iteration_count": 0,
            })

        return builder.build()

    def _triage_next_step(self, state: EnhancedAgentState) -> Literal["direct", "simple", "complex"]:
        """Route based on triage decision and task complexity."""
        decision = state.get("triage_decision") or "use_tools"
        complexity = state.get("task_complexity") or "simple"

        if decision == "direct_response":
            return "direct"
        elif complexity == "simple":
            return "simple"
        else:
            return "complex"

    def _triage_next_step_simplified(self, state: EnhancedAgentState) -> Literal["direct", "execute"]:
        """Simplified triage: direct response or execute (no orchestrator distinction)."""
        decision = state.get("triage_decision") or "use_tools"
        if decision == "direct_response":
            return "direct"
        return "execute"

    def _extract_at_file_references(self, text: str) -> list[tuple[str, str]]:
        """Extract @file references and read their contents.

        Claude Code style: Parse @ file attachments for direct analysis.

        Returns:
            List of (file_path, content) tuples
        """
        import re
        results = []

        # Pattern: @path/to/file or @"path with spaces"
        patterns = [
            r'@"([^"]+)"',  # @"path with spaces"
            r"@'([^']+)'",  # @'path with spaces'
            r'@(\S+\.\w+)',  # @path/to/file.ext (must have extension)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                file_path = match.strip()
                if file_path and not file_path.startswith('http'):
                    try:
                        # Resolve relative paths
                        import os
                        if not os.path.isabs(file_path):
                            file_path = os.path.join(os.getcwd(), file_path)

                        if os.path.isfile(file_path):
                            with open(file_path, encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                # Limit content size
                                if len(content) > 10000:
                                    content = content[:10000] + "\n... (truncated)"
                                results.append((file_path, content))
                    except Exception:
                        pass  # Skip files that can't be read

        return results

    @staticmethod
    def _is_dangerous_command_request(text: str) -> bool:
        """Heuristic check for prompts requesting destructive shell commands."""
        if not text:
            return False

        lowered = text.lower()
        dangerous_patterns = [
            r"rm\s+-rf\s+/",
            r"dd\s+if=/dev/(zero|random|urandom)",
            r"mkfs\.",
            r"chmod\s+(777|666)",
            r"(curl|wget).*\|\s*(sh|bash)",
            r"/etc/(shadow|sudoers|passwd)",
        ]
        return any(re.search(pattern, lowered) for pattern in dangerous_patterns)

    def _normalize_safety_refusal(self, content: str, messages: list[BaseMessage]) -> str:
        """Ensure dangerous-command refusals contain explicit safety wording."""
        if not content:
            return content

        latest_user = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_user = str(msg.content or "")
                break

        if not self._is_dangerous_command_request(latest_user):
            return content

        lowered = content.lower()
        refusal_signals = ("can't", "can’t", "cannot", "won't", "will not", "sorry", "unable")
        has_refusal = any(signal in lowered for signal in refusal_signals)
        has_safety_words = any(word in lowered for word in ("dangerous", "blocked", "security", "destructive", "denied"))

        if has_refusal and not has_safety_words:
            return (
                f"{content.rstrip()}\n\n"
                "This command is dangerous and blocked for security reasons; I cannot execute it."
            )

        return content

    @staticmethod
    def _is_substantive_change_path(path: str) -> bool:
        """Return True when a changed path represents real project code/content."""
        if not path:
            return False
        normalized = path.replace("\\", "/").lstrip("./")
        parts = normalized.split("/")
        internal_dirs = {"logs", ".sepilot", "__pycache__", ".pytest_cache", "temp"}
        return not any(part in internal_dirs for part in parts)

    @staticmethod
    def _collect_substantive_changes(
        file_changes: list[Any],
        modified_files: list[str],
    ) -> tuple[list[str], list[str]]:
        """Filter file change lists to substantive project files."""
        changed_from_records: list[str] = []
        for fc in file_changes:
            path = fc if isinstance(fc, str) else getattr(fc, "file_path", None)
            if path and ReactAgent._is_substantive_change_path(path):
                changed_from_records.append(path)
        changed_from_snapshot = [
            path for path in modified_files
            if path and ReactAgent._is_substantive_change_path(path)
        ]
        return changed_from_records, changed_from_snapshot

    @staticmethod
    def _is_explicit_coding_request(user_query: str) -> bool:
        """Return True when user text clearly requests concrete code edits."""
        if not user_query:
            return False
        lowered = user_query.lower()
        coding_indicators = (
            "fix", "bug", "implement", "refactor", "patch",
            "edit file", "modify file", "update the code", "change the code",
            "write code", "add function", "remove function",
            "수정", "버그", "구현", "리팩토링", "패치", "코드 변경", "파일 수정",
        )
        return any(indicator in lowered for indicator in coding_indicators)

    @staticmethod
    def _is_analysis_only_request(user_query: str) -> bool:
        """Return True when request is primarily analysis/explanation/diagnosis without implementation intent.

        This keeps AUTO mode instead of transitioning to PLAN, which would
        block bash_execute needed for diagnostic commands (kubectl, docker, etc.).
        """
        if not user_query:
            return False
        lowered = user_query.lower()
        analysis_indicators = (
            "analyze", "analyse", "describe", "explain", "summarize", "summary",
            "review", "understand", "read",
            "diagnose", "investigate", "troubleshoot", "status", "health",
            "분석", "설명", "요약", "검토",
            "진단", "파악", "조사", "상태", "점검", "장애",
        )
        # Reuse strict coding detector to avoid over-blocking legitimate coding tasks.
        return any(indicator in lowered for indicator in analysis_indicators) and not ReactAgent._is_explicit_coding_request(user_query)

    @staticmethod
    def _is_execution_request(state: EnhancedAgentState) -> bool:
        """Return True when LLM triage classified the task as execution/verification/implementation.

        Includes TEST (verification), IMPLEMENT (code execution), and DEBUG (bug fix execution).
        """
        triage_decision = state.get("triage_decision")
        if triage_decision == "direct_response":
            return False

        strategy = state.get("current_strategy", AgentStrategy.EXPLORE)
        if isinstance(strategy, str):
            with contextlib.suppress(ValueError):
                strategy = AgentStrategy(strategy.lower())

        # Execution-oriented strategies: testing, implementing, and debugging all require tool execution
        return strategy in (AgentStrategy.TEST, AgentStrategy.IMPLEMENT, AgentStrategy.DEBUG)

    def _direct_response_node(self, state: EnhancedAgentState) -> dict[str, Any]:
        """LangGraph Multi-Agent Pattern: Direct Response Specialist.

        Claude Code style: Parse @ file attachments for direct analysis.
        """
        messages = state.get("messages", [])
        if not messages:
            return {}

        # Extract @ file references from the current user message only.
        file_contents = []
        current_user_query = get_current_user_query(state)
        if current_user_query:
            file_contents.extend(self._extract_at_file_references(current_user_query))

        # Build file context if any files were referenced
        file_context = ""
        if file_contents:
            file_context = "\n\n═══ ATTACHED FILES ═══\n"
            for path, content in file_contents[:5]:  # Max 5 files
                file_context += f"\n--- {path} ---\n{content}\n"
            file_context += "═══════════════════════\n\n"

            if self.console and self.verbose:
                self.console.print(f"[dim cyan]📎 Parsed {len(file_contents)} @ file attachment(s)[/dim cyan]")

        # LangGraph Multi-Agent Pattern: Specialized Direct Response Agent
        direct_response_specialist_msg = SystemMessage(content=(
            "💬 You are a DIRECT RESPONSE SPECIALIST agent.\n\n"
            "Your role: Provide immediate answers without using tools\n"
            "Your expertise: General knowledge, explanations, and code analysis\n"
            "Your constraint: Do NOT use tools - answer directly\n\n"
            "As a direct response specialist, you:\n"
            "- Answer questions using your knowledge base\n"
            "- Analyze file contents if provided in the user message\n"
            "- Provide explanations and clarifications\n"
            "- Offer guidance and best practices\n"
            "- Give examples when helpful\n"
            "- Keep responses concise and relevant\n\n"
            "IMPORTANT:\n"
            "- If the user attached files with @ or the message includes file contents,\n"
            "  analyze those contents directly without calling tools\n"
            "- Extract and explain the requested information from the provided content\n"
            "- For code analysis requests, identify key components, patterns, and functionality\n\n"
            "This query was routed to you because it doesn't require tool execution.\n"
            "The information needed is either in your knowledge base or already provided in the message.\n"
            "Provide a helpful, direct answer based on the available information."
            f"{file_context}"
        ))

        messages_with_context = [direct_response_specialist_msg] + messages

        try:
            _llm_t0 = time.monotonic()
            response = self.quick_llm.invoke(messages_with_context)
            _llm_elapsed = time.monotonic() - _llm_t0

            # Extract and track token usage (works for both regular and thinking models)
            tokens_used, output_tokens = self._track_token_usage(response, source="direct_response")

            self._verbose_llm_trace(
                "Direct Response",
                messages_with_context,
                response,
                tokens=tokens_used,
            )

            # Normalize refusal wording for dangerous command requests
            normalized_content = ""
            if hasattr(response, "content") and response.content:
                normalized_content = self._normalize_safety_refusal(str(response.content), messages)

            # Send response to chat
            if normalized_content:
                self._send_chat_response(normalized_content)

            # Log with additional info for debugging thinking models
            log_data = {
                "reason": state.get("triage_reason", "general_question"),
                "tokens_used": tokens_used,
                "has_usage_metadata": hasattr(response, 'usage_metadata'),
                "response_type": type(response).__name__
            }
            self.logger.log_trace("direct_response", log_data)

            # Use StateUpdateBuilder for cleaner, more maintainable code
            response_to_store: BaseMessage
            if normalized_content and normalized_content != str(getattr(response, "content", "")):
                response_to_store = AIMessage(content=normalized_content)
            else:
                response_to_store = response

            builder = (StateUpdateBuilder()
                .set_decision("direct_response")
                .add_message(response_to_store))

            if tokens_used > 0:
                cost = self._calculate_cost(tokens_used)
                # Compute absolute values (no reducer on these fields)
                current_tokens = state.get("total_tokens_used", 0)
                current_cost = state.get("estimated_cost", 0.0)
                builder.track_tokens(current_tokens + tokens_used, current_cost + cost)

            # Update status indicator with accurate output generation speed
            if self.status_indicator:
                self.status_indicator.update_token_rate(output_tokens, _llm_elapsed)

            return builder.build()
        except Exception as exc:
            # Log detailed error info for debugging (especially for thinking models)
            import traceback
            error_details = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "triage_reason": state.get("triage_reason")
            }
            self.logger.log_error(f"Direct response failed: {str(exc)}")
            self.logger.log_trace("direct_response_error", error_details)

            fallback = AIMessage(
                content="I tried to answer directly but hit an unexpected issue. "
                        "Please re-run the request."
            )
            err_delta = state_helpers.record_error(
                state,
                message=f"Direct response failed: {str(exc)}",
                level=ErrorLevel.WARNING,
                source="llm",
                context=error_details,
                return_delta=True
            )
            updates = {}  # Initialize updates before using it
            updates = self._merge_updates(updates, err_delta)
            updates = self._merge_updates(updates, {"messages": [fallback]})
            return updates

    def _iteration_guard_node(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Reset transient flags, increment iteration counter, and enforce iteration budget.

        Claude Code style:
        - Simple iteration limit enforcement
        - Stagnation detection (no progress patterns)
        - Strategy adjustment injection
        """
        updates: dict[str, Any] = {}

        # Reset transient flags
        if state.get("needs_additional_iteration"):
            updates["needs_additional_iteration"] = False
        if state.get("last_approval_status"):
            updates["last_approval_status"] = None

        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", self.settings.max_iterations)

        # Wall-clock guard: avoid hanging a single CLI execution for too long.
        deadline = state.get("_execution_deadline_monotonic")
        if isinstance(deadline, (int, float)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                updates["force_termination"] = True
                prompt_excerpt = get_current_user_query(state).strip()[:160] or "the current task"
                timeout_msg = AIMessage(
                    content=(
                        "Execution time limit reached before full completion. "
                        f"Task context: {prompt_excerpt}\n"
                        "Please rerun to continue from this point."
                    )
                )
                updates.setdefault("messages", []).append(timeout_msg)
                err_delta = state_helpers.record_error(
                    state,
                    message="Execution time limit reached.",
                    level=ErrorLevel.WARNING,
                    source="system",
                    context={"iteration": iteration},
                    return_delta=True
                )
                return self._merge_updates(updates, err_delta)

        # Structural safety checks (error loops, recursion detection)
        stagnation_detected = False
        stagnation_hint = ""

        if iteration >= 3:
            # Pattern 4: Same error repeated (stuck in error loop) — structural safety
            error_history = state.get("error_history", [])
            if len(error_history) >= 2:
                recent_errors = [
                    e.message[:50] if hasattr(e, 'message') else str(e)[:50]
                    for e in error_history[-3:]
                ]
                if len(recent_errors) >= 2 and len(set(recent_errors)) == 1:
                    stagnation_detected = True
                    stagnation_hint = f"Same error repeated: {recent_errors[0][:30]}... Try a different approach."

            # Pattern 5: Recursion detector flagged repetitive tool calls — structural safety
            if not stagnation_detected and state.get("repetition_detected"):
                rep_info = state.get("repetition_info", {})
                stagnation_detected = True
                pattern_type = rep_info.get("pattern", "unknown")
                stagnation_hint = (
                    f"Repetition pattern ({pattern_type}) detected: {rep_info.get('tool_name', 'unknown')}. "
                    "Change your approach."
                )
                updates["repetition_detected"] = False  # Reset flag

            # Pattern 6: Read-without-write detection — many reads, zero writes
            # Window of 8 calls gives enough context; trigger at 4+ reads with 0 writes.
            if not stagnation_detected and iteration >= 4:
                tool_history = state.get("tool_call_history", [])
                if tool_history:
                    recent = tool_history[-8:]
                    read_tools = {"file_read", "search_content", "find_file", "codebase", "get_structure"}
                    write_tools = {"file_edit", "file_write"}
                    reads = sum(1 for tc in recent if getattr(tc, "tool_name", "") in read_tools)
                    writes = sum(1 for tc in recent if getattr(tc, "tool_name", "") in write_tools)
                    if reads >= 4 and writes == 0:
                        stagnation_detected = True
                        stagnation_hint = (
                            f"Read-without-write: {reads} reads, 0 writes in last {len(recent)} calls. "
                            "You have gathered substantial context — consider applying code changes now."
                        )

        # Detect external file changes (from editors running outside the agent)
        if self.file_watcher and self.file_watcher.is_running:
            ext_changes = self.file_watcher.get_external_changes()
            if ext_changes:
                notice = self.file_watcher.format_changes_for_agent(ext_changes)
                if notice:
                    ext_msg = SystemMessage(
                        content=f"📂 {notice}\nThese files were modified externally. "
                        "Re-read them before making further changes."
                    )
                    updates.setdefault("messages", [])
                    if isinstance(updates["messages"], list):
                        updates["messages"].append(ext_msg)
                    else:
                        updates["messages"] = [ext_msg]

                    if self.console:
                        self.console.print(
                            f"[cyan]📂 External changes detected: {len(ext_changes)} file(s)[/cyan]"
                        )

        if iteration >= max_iter:
            # Simple termination - let the agent know clearly
            updates["force_termination"] = True

            if self.console:
                self.console.print(
                    f"[yellow]⚠️ Iteration limit reached ({iteration}/{max_iter}). Completing task.[/yellow]"
                )

            err_delta = state_helpers.record_error(
                state,
                message=f"Iteration limit reached ({iteration}/{max_iter}).",
                level=ErrorLevel.WARNING,
                source="system",
                context={"iteration": iteration},
                return_delta=True
            )
            updates = self._merge_updates(updates, err_delta)
        else:
            # Increment iteration count
            delta = state_helpers.increment_iteration(state, return_delta=True)
            updates = self._merge_updates(updates, delta)
            # Sync instance counter from delta (single source of truth: state)
            self.iteration_count = delta.get("iteration_count", self.iteration_count + 1)

            # Structural stagnation detection (error loops / recursion only)
            # Inject forced intervention message so weaker models break out of loops
            if stagnation_detected:
                if self.console:
                    self.console.print(f"[yellow]⚠️ Stagnation detected: {stagnation_hint}[/yellow]")
                if self.step_logger:
                    self.step_logger.log_context(f"stagnation: {stagnation_hint[:60]}")
                updates["stagnation_detected"] = True

                # Intervention: provide progress context + structural mode transition
                # Instead of commanding the LLM, give it awareness of its own state
                file_changes = state.get("file_changes", [])
                modified_files = state.get("modified_files", [])
                has_changes = bool(file_changes) or bool(modified_files)
                progress_ctx = (
                    f"Files modified so far: {len(file_changes) + len(modified_files)}"
                    if has_changes else "No files have been modified yet in this session."
                )
                intervention_msg = SystemMessage(
                    content=(
                        f"Progress: {stagnation_hint}\n"
                        f"{progress_ctx}\n"
                        "Consider changing your approach:\n"
                        "  1. Apply code changes based on your analysis so far\n"
                        "  2. Read different files to find alternative solutions\n"
                        "  3. Use bash_execute to gather more diagnostic info\n"
                        "  4. Try a completely different strategy"
                    )
                )
                updates.setdefault("messages", []).append(intervention_msg)
                updates["_skip_compaction"] = True

                # Structural: if in PLAN mode with no file changes, transition to CODE
                cur_mode = state.get("current_mode", AgentMode.AUTO)
                if cur_mode == AgentMode.PLAN and not has_changes:
                    updates["current_mode"] = AgentMode.CODE
                    updates["mode_iteration_count"] = 0

            # ===== Agent Mode transition logic =====
            # Process pending mode update from interactive commands (/plan, /code, /exec)
            pending_mode_applied = False
            pending_mode_locked = False
            if self._pending_mode_update:
                updates.update(self._pending_mode_update)
                pending_mode_locked = bool(self._pending_mode_update.get("mode_locked", False))
                self._pending_mode_update = None
                pending_mode_applied = True

            # Automatic mode transition (using deep copy to avoid shared nested lists)
            import copy
            effective_state = copy.deepcopy(state)
            effective_state.update(updates)
            current_mode = effective_state.get("current_mode", AgentMode.AUTO)
            _guard_profile = getattr(self, "prompt_profile", "default")
            suggested_mode = self.mode_engine.suggest_transition(effective_state, prompt_profile=_guard_profile)

            # Manual mode command should win for at least this iteration.
            if pending_mode_applied and pending_mode_locked:
                suggested_mode = None

            # Keep AUTO mode for analysis/plan/read-only requests to avoid
            # unnecessary PLAN/CODE gating and "mode change required" detours.
            if current_mode == AgentMode.AUTO:
                planning_notes = effective_state.get("planning_notes", [])
                is_read_only_request = any("[READ-ONLY]" in n for n in planning_notes)
                user_query = get_current_user_query(effective_state)
                if (
                    is_read_only_request
                    or is_plan_request(user_query)
                    or self._is_analysis_only_request(user_query)
                ):
                    suggested_mode = None

            if suggested_mode and suggested_mode != current_mode:
                updates["current_mode"] = suggested_mode
                updates["mode_iteration_count"] = 0
                updates["mode_history"] = [
                    f"iter={iteration}: {current_mode.value}→{suggested_mode.value}"
                ]
                guide_msg = SystemMessage(
                    content=f"🔄 Mode switched: {current_mode.value.upper()} → {suggested_mode.value.upper()}. "
                    f"{get_mode_prompt(suggested_mode, prompt_profile=_guard_profile) or ''}"
                )
                updates.setdefault("messages", []).append(guide_msg)
                updates["_skip_compaction"] = True

                if self.console:
                    self.console.print(
                        f"[bold cyan]🔄 Mode: {current_mode.value.upper()} → {suggested_mode.value.upper()}[/bold cyan]"
                    )
                if self.step_logger:
                    self.step_logger.log_mode(current_mode.value, suggested_mode.value)
            else:
                if pending_mode_applied and pending_mode_locked:
                    new_mode_iter = int(effective_state.get("mode_iteration_count", 0))
                else:
                    new_mode_iter = int(effective_state.get("mode_iteration_count", 0)) + 1
                updates["mode_iteration_count"] = new_mode_iter

                # EXEC mode completion enforcement: prevent timeout from repeated verification
                if current_mode == AgentMode.EXEC and self.mode_engine.should_force_exec_completion(new_mode_iter):
                    exec_nudge = SystemMessage(
                        content=(
                            "⏱️ EXEC MODE LIMIT REACHED. You have already verified the changes. "
                            "Do NOT run any more tests or commands. "
                            "Summarize your findings and COMPLETE the task immediately."
                        )
                    )
                    updates.setdefault("messages", []).append(exec_nudge)
                    updates["_skip_compaction"] = True
                    # Boost confidence to force completion
                    updates["confidence_score"] = max(
                        state.get("confidence_score", 0.5), 0.85
                    )
                    if self.console:
                        self.console.print(
                            f"[yellow]⏱️ EXEC mode limit ({EXEC_MAX_ITERATIONS} iterations). Forcing completion.[/yellow]"
                        )

        return updates

    def _guard_decision(self, state: EnhancedAgentState) -> Literal["continue", "stop"]:
        """Decide whether to continue or finish after the guard check.

        The "last chance" mechanism for coding tasks is handled in _iteration_guard_node.
        """
        if state.get("force_termination"):
            return "stop"

        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", self.settings.max_iterations)

        if iteration >= max_iter:
            return "stop"

        return "continue"

    def _context_manager_node(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Auto-compact context if it exceeds threshold.

        Claude Code style: Lazy token calculation for efficiency.
        Only compute exact tokens when approaching threshold.
        """
        # Skip compaction if flagged (e.g., after injecting nudge message)
        if state.get("_skip_compaction"):
            return {"_skip_compaction": False}  # Reset flag for next iteration

        messages = state.get("messages", [])
        if not messages:
            return {}

        # Compaction cooldown - don't compact again within 5 iterations after last compaction
        last_compaction_iter = state.get("_last_compaction_iter", 0)
        current_iter = state.get("iteration_count", 0)
        if last_compaction_iter > 0 and (current_iter - last_compaction_iter) < 5:
            return {}  # Skip compaction during cooldown

        max_tokens = self._context_manager.max_context_tokens

        # LAZY TOKEN CALCULATION: Quick character-based estimate first
        # Only compute exact tokens if estimate is above 70% threshold
        total_chars = sum(
            len(str(msg.content)) for msg in messages
            if hasattr(msg, 'content') and msg.content
        )
        estimated_tokens = total_chars // 4  # Rough estimate: ~4 chars per token

        # Early exit if clearly below threshold (70% of max)
        if estimated_tokens < max_tokens * 0.7:
            return {}  # No need for exact calculation

        # Only calculate exact tokens when approaching threshold
        current_tokens = estimated_tokens
        encoding = None
        try:
            from sepilot.agent.context_manager import _get_tiktoken_encoding
            encoding = _get_tiktoken_encoding()
            if encoding is not None:
                current_tokens = 0
                for msg in messages:
                    if hasattr(msg, 'content') and msg.content:
                        current_tokens += len(encoding.encode(str(msg.content)))
        except Exception:
            current_tokens = estimated_tokens  # Fall back to estimate

        context_manager = self._context_manager
        updates: dict[str, Any] = {}

        # Step 1: Prune large tool outputs from older messages (cheap, no LLM call)
        pruned_messages = context_manager.prune_tool_outputs(messages, keep_recent=10)
        # Recalculate tokens after pruning using same method as above (exact or estimate)
        pruned_chars = sum(
            len(str(msg.content)) for msg in pruned_messages
            if hasattr(msg, 'content') and msg.content
        )
        pruned_tokens = pruned_chars // 4
        pruning_changed = len(pruned_messages) != len(messages)
        if pruning_changed:
            # Pruning changed content — recalculate with tiktoken if available
            if encoding is not None:
                try:
                    current_tokens = 0
                    for msg in pruned_messages:
                        if hasattr(msg, 'content') and msg.content:
                            current_tokens += len(encoding.encode(str(msg.content)))
                except Exception:
                    current_tokens = pruned_tokens
            else:
                current_tokens = pruned_tokens
            # Use RemoveMessage + new messages instead of direct replacement
            # to work correctly with add_messages reducer
            updates["messages"] = self._compute_message_diff(messages, pruned_messages)
            messages = pruned_messages

        # Step 2: Predictive compaction - pre-identify candidates at 75%
        if context_manager.should_prepare_compaction(current_tokens):
            context_manager.identify_compaction_candidates(messages)

        # Step 3: Full compaction if still over threshold (use incremental first)
        if context_manager.should_compact(current_tokens):
            if self.console:
                self.console.print("[yellow]⚠️  Context usage high, auto-compacting...[/yellow]")

            # Use incremental compaction (summarizes oldest 50% by relevance)
            try:
                compacted_messages = context_manager.compact_incremental(
                    messages,
                    llm=self.quick_llm,
                )
                updates["messages"] = self._compute_message_diff(messages, compacted_messages)
                updates["_last_compaction_iter"] = current_iter  # Set cooldown

                if self.console:
                    self.console.print(f"[green]✅ Context compacted: {len(messages)} → {len(compacted_messages)} messages[/green]")

            except Exception:
                # If incremental fails, use simple compaction
                if self.console:
                    self.console.print("[yellow]Summarization failed, using simple compaction[/yellow]")

                compacted_messages = context_manager.compact_messages(messages, keep_recent=10)
                updates["messages"] = self._compute_message_diff(messages, compacted_messages)
                updates["_last_compaction_iter"] = current_iter  # Set cooldown

        return updates

    def _compute_message_diff(
        self, old_messages: list[BaseMessage], new_messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Replace the persisted message list exactly for add_messages compatibility."""
        if old_messages == new_messages:
            return []
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES, content=""), *new_messages]

    def _agent_next_step(self, state: EnhancedAgentState) -> Literal["tools", "finalize"]:
        """Route agent output toward tools or verification."""
        messages = state.get("messages", [])
        if not messages:
            return "finalize"
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "finalize"

    def _approval_node(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Request user approval for sensitive tool invocations."""
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return {"last_approval_status": "approved"}

        current_mode = state.get("current_mode", AgentMode.AUTO)
        force_prompt = current_mode == AgentMode.EXEC

        calls_requiring_approval = (
            tool_calls if force_prompt else [
                tc for tc in tool_calls
                if tc.get("name") in self.sensitive_tools
            ]
        )
        if not calls_requiring_approval:
            # Reset denial counter on implicit approval
            return {"last_approval_status": "approved", "consecutive_denials": 0}

        # Annotate with risk score for better UX
        annotated_calls = []
        for tc in calls_requiring_approval:
            tool_name = tc.get("name", "unknown")
            risk = self._assess_tool_risk(tool_name, tc.get("args", {}) or {})
            annotated_calls.append({"call": tc, "risk": risk})

        # Auto-approve in interactive mode or when explicitly enabled
        if self.auto_approve:
            if self.console:
                tool_list = ", ".join({tc.get("name", "unknown") for tc in calls_requiring_approval})
                self.console.print(f"[dim cyan]🔓 Auto-approving tools: {tool_list}[/dim cyan]")
            return {"last_approval_status": "approved"}

        # Prepare payload for interrupt
        def _is_noise_rationale(text: str) -> bool:
            normalized = (text or "").strip().lower()
            if not normalized:
                return True
            noise_tokens = (
                "experience stored",
                "quality:",
                "orchestrator",
                "retrieved",
                "memory writer",
                "rolled back",
                "debate skipped",
                "[read-only]",
            )
            return any(token in normalized for token in noise_tokens)

        llm_rationale = ""
        raw_content = getattr(last_message, "content", "")
        if isinstance(raw_content, str):
            llm_rationale = raw_content.strip()
        elif isinstance(raw_content, list):
            content_parts: list[str] = []
            for part in raw_content:
                if isinstance(part, str):
                    part_text = part.strip()
                    if part_text:
                        content_parts.append(part_text)
                    continue
                if isinstance(part, dict):
                    for key in ("text", "content"):
                        value = part.get(key)
                        if isinstance(value, str) and value.strip():
                            content_parts.append(value.strip())
                            break
            llm_rationale = "\n".join(content_parts).strip()

        if _is_noise_rationale(llm_rationale):
            llm_rationale = ""

        # Do not pull rationale from older assistant/planning messages:
        # it frequently leaks stale, unrelated text into approval UI.
        # When missing, ApprovalHandler will generate a concise tool-based rationale.

        risk_levels = [str(item["risk"].get("level", "low")).lower() for item in annotated_calls]
        if any(level == "high" for level in risk_levels):
            approval_reason = "고위험 도구 실행 전 사용자 승인 필요"
        elif any(level == "medium" for level in risk_levels):
            approval_reason = "중위험 도구 실행 전 사용자 승인 필요"
        else:
            approval_reason = "도구 실행 전 사용자 승인 필요"

        payload = {
            "type": "tool_approval",
            "reason": (
                "EXEC 모드: 모든 도구 실행 전 사용자 승인 필요"
                if force_prompt
                else approval_reason
            ),
            "llm_rationale": self._truncate_text(llm_rationale, max_length=600) if llm_rationale else "",
            "force_prompt": force_prompt,
            "tool_calls": [
                {
                    "id": item["call"].get("id"),
                    "name": item["call"].get("name"),
                    "args": item["call"].get("args", {}),
                    "risk_level": item["risk"]["level"],
                    "risk_reasons": item["risk"]["reasons"],
                }
                for item in annotated_calls
            ]
        }

        # Send input request to web UI (if monitor is available)
        # Pause execution and wait for human decision
        # This will cause graph.stream() to return a chunk with __interrupt__
        decision = interrupt(payload)

        # Clear input request after decision is made
        # Normalize the response
        decision = self._normalize_approval_response(decision)

        updates: dict[str, Any] = {}
        updates["pending_approvals"] = []
        updates["last_approval_status"] = decision["status"]

        tool_list = ", ".join({tc.get("name", "unknown") for tc in calls_requiring_approval})

        # Track consecutive denials to prevent infinite loops
        consecutive_denials = state.get("consecutive_denials", 0)

        if decision["status"] == "deny":
            consecutive_denials += 1
            updates["consecutive_denials"] = consecutive_denials

            # Force termination after 3 consecutive denials
            if consecutive_denials >= 3:
                if self.console:
                    self.console.print(
                        "[bold red]⚠️ 3회 연속 거부되어 작업을 중단합니다.[/bold red]"
                    )
                updates["force_termination"] = True
                termination_message = self._make_internal_human_message(
                    content="🛑 작업이 3회 연속 거부되어 중단합니다. "
                            "다른 접근 방식이 필요하거나 요청을 수정해 주세요."
                )
                return self._merge_updates(updates, {"messages": [termination_message]})
            reason = decision.get("reason") or "User denied the action."
            # Use HumanMessage so LLM treats this as user instruction, not its own statement
            denial_message = self._make_internal_human_message(
                content=f"🛑 STOP: 사용자가 도구 실행을 거부했습니다.\n\n"
                        f"거부된 도구: {tool_list}\n"
                        f"사용자 지시: {reason}\n\n"
                        f"위 사용자 지시를 반드시 따르세요. 거부된 작업을 다시 시도하지 마세요."
            )
            err_delta = state_helpers.record_error(
                state,
                message=f"User denied tool execution ({tool_list})",
                level=ErrorLevel.WARNING,
                source="user",
                context={"tools": [tc.get("name") for tc in calls_requiring_approval]},
                return_delta=True
            )
            updates = self._merge_updates(updates, err_delta)
            return self._merge_updates(updates, {"messages": [denial_message]})

        if decision["status"] == "feedback":
            feedback = decision.get("message") or "사용자가 추가 지시를 제공했습니다."
            return self._merge_updates(
                updates,
                {"messages": [self._make_internal_human_message(feedback)]}
            )

        # Approved - reset denial counter
        updates["consecutive_denials"] = 0
        return updates

    def _approval_next_step(self, state: EnhancedAgentState) -> Literal["run_tools", "retry"]:
        """Determine whether to run tools or retry planning after approval."""
        status = (state.get("last_approval_status") or "approved").lower()
        if status in {"deny", "feedback"}:
            return "retry"
        return "run_tools"

    def _calibrate_confidence(
        self,
        state: dict[str, Any],
        llm_confidence: float,
        is_complete: bool,
    ) -> float:
        """Calibrate LLM-reported confidence with heuristic cross-checks.

        Weak models tend to over-report confidence. This applies rule-based
        discounts when evidence contradicts the LLM's self-assessment.
        """
        calibrated = llm_confidence

        if not is_complete:
            return calibrated

        task_complexity = state.get("task_complexity_label", state.get("complexity", ""))
        iteration_count = state.get("iteration_count", 0)
        # Check BOTH sources of file changes:
        # - file_changes: from tool_tracker delta queue (FileChange objects)
        # - modified_files: from tool_executor detect_file_changes (file paths)
        file_changes = state.get("file_changes", [])
        modified_files = state.get("modified_files", [])
        substantive_file_changes, substantive_modified_files = ReactAgent._collect_substantive_changes(
            file_changes,
            modified_files,
        )
        has_any_edits = bool(substantive_file_changes) or bool(substantive_modified_files)
        tool_call_history = state.get("tool_call_history", [])

        # SWE-bench: only file_edit counts as a real fix (not file_write).
        # file_write creates new files (e.g. test scripts) — not a source fix.
        # BOTH conditions required: file_edit was called AND files were actually modified.
        # This prevents premature completion when file_edit fails (e.g. "String not found").
        prompt_profile = getattr(self, "prompt_profile", "")
        if prompt_profile.startswith("swe_bench"):
            file_edit_called = any(
                tc.tool_name == "file_edit" for tc in tool_call_history
            )
            actual_changes = has_any_edits
            has_any_edits = file_edit_called and actual_changes

        # Detect if this is a modification task
        user_query = get_current_user_query(state).lower()

        # SWE-bench profile: ALWAYS a modification task (bug fix)
        if prompt_profile.startswith("swe_bench"):
            is_modification = True
        else:
            modify_keywords = {"fix", "수정", "implement", "구현", "refactor", "리팩토링",
                               "create", "생성", "edit", "change", "modify", "update",
                               "add", "delete", "remove", "write"}
            is_modification = any(kw in user_query for kw in modify_keywords)

        read_only_tools = {"file_read", "find_file", "search_content",
                           "find_definition", "get_structure", "list_directory",
                           "get_file_info", "codebase"}

        # 1) Modification task with no file edits → cap at 0.35 (×0.85 = 0.2975 < 0.4 threshold)
        if is_modification and not has_any_edits:
            calibrated = min(calibrated, 0.35)

        # 2) Complex task completing on first iteration → cap at 0.6
        if task_complexity == "complex" and iteration_count <= 1:
            calibrated = min(calibrated, 0.6)

        # 3) Read-only tools used for a modification task → cap at 0.3 (×0.85 = 0.255 < 0.4 threshold)
        if is_modification and not has_any_edits and tool_call_history:
            used_tools = {tc.tool_name for tc in tool_call_history}
            if used_tools and used_tools.issubset(read_only_tools):
                calibrated = min(calibrated, 0.3)

        # 4) Mode-aware calibration cap
        current_mode = state.get("current_mode", AgentMode.AUTO)
        if current_mode != AgentMode.AUTO:
            plan_created = state.get("plan_created", False)
            has_file_changes = has_any_edits
            exec_tools = {"bash_execute", "bash_background", "git"}
            exec_tools_used = bool(tool_call_history and any(
                tc.tool_name in exec_tools for tc in tool_call_history
            ))
            mode_cap = mode_calibration_cap(
                current_mode, is_modification, plan_created,
                has_file_changes, exec_tools_used,
            )
            if mode_cap is not None:
                calibrated = min(calibrated, mode_cap)

        # 5) Modification task without file edits → hard cap at 0.1
        #    SWE-bench: file_edit 없이는 절대 완료 불가
        if is_modification and not has_any_edits:
            calibrated = min(calibrated, 0.1)

        # 6) General weak-model discount: × 0.85
        calibrated *= 0.85

        return max(min(calibrated, 1.0), 0.0)

    def _verification_node(self, state: EnhancedAgentState) -> dict[str, Any]:
        """LangGraph Multi-Agent Pattern: Verification Specialist Agent.

        Claude Code style: Fast-path for simple completions, LLM check only when needed.
        """
        messages = state.get("messages", [])
        if not messages:
            return {}

        updates: dict[str, Any] = {}

        # Get iteration and tool history early for use throughout verification
        iteration_count = state.get("iteration_count", 0)
        tool_call_history = state.get("tool_call_history", [])

        # Current execution window (ignore old thread history from previous turns)
        boundary_idx = self._find_execution_boundary(state)
        current_messages = messages[boundary_idx:] if boundary_idx < len(messages) else []

        # Resolve current-turn user query via the shared execution boundary helper.
        user_query = get_current_user_query(state)
        make_internal_human_message = getattr(
            self,
            "_make_internal_human_message",
            ReactAgent._make_internal_human_message,
        )

        current_turn_tool_messages = [
            msg for msg in current_messages if isinstance(msg, ToolMessage)
        ]
        current_turn_tool_plans = any(
            bool(getattr(msg, "tool_calls", None)) for msg in current_messages
        )

        # "Execution progress" means at least one tool actually ran and wasn't
        # merely a mode-blocking/error placeholder.
        execution_progress = False
        for tool_msg in current_turn_tool_messages:
            content = str(getattr(tool_msg, "content", "") or "")
            lowered = content.lower()
            if (
                "not available in" in lowered
                and "mode" in lowered
            ):
                continue
            if lowered.startswith("error: tool '") and "not available in" in lowered:
                continue
            execution_progress = True
            break

        no_tool_activity = (not current_turn_tool_messages) and (not current_turn_tool_plans)
        no_execution_progress = not execution_progress

        # If LLM triage marked this as execution/verification but no tools were called
        # in this turn, force mode alignment + another iteration.
        if self._is_execution_request(state) and (no_tool_activity or no_execution_progress):
            desired_mode = AgentMode.EXEC
            current_mode = state.get("current_mode", AgentMode.AUTO)
            mode_locked = bool(state.get("mode_locked", False))

            if current_mode != desired_mode:
                if mode_locked:
                    decision = interrupt(
                        {
                            "type": "mode_switch_request",
                            "current_mode": current_mode.value if isinstance(current_mode, AgentMode) else str(current_mode),
                            "suggested_mode": desired_mode.value,
                            "blocked_tools": ["bash_execute" if desired_mode == AgentMode.EXEC else "file_edit/file_write"],
                            "reason": (
                                f"실행 요청을 처리하려면 {desired_mode.value.upper()} 모드가 필요합니다. "
                                f"{desired_mode.value.upper()} 모드로 전환할까요?"
                            ),
                        }
                    )
                    normalized = self._normalize_approval_response(decision)
                    if normalized.get("status") == "approved":
                        updates["current_mode"] = desired_mode
                        updates["mode_locked"] = False
                        updates["mode_iteration_count"] = 0
                    else:
                        return self._merge_updates(
                            updates,
                            {
                                "verification_notes": ["실행 요청이 있었지만 모드 전환이 거부됨"],
                                "needs_additional_iteration": False,
                            },
                        )
                else:
                    updates["current_mode"] = desired_mode
                    updates["mode_iteration_count"] = 0

            reminder = make_internal_human_message(
                content=(
                    "요청은 실행 작업입니다. 설명만 하지 말고 즉시 도구를 호출하세요. "
                    "스크립트 실행의 경우 bash_execute(command=\"...\")를 사용하세요."
                )
            )
            return self._merge_updates(
                updates,
                {
                    "messages": [reminder],
                    "verification_notes": ["실행 요청 감지: 도구 실행 단계로 재진입"],
                    "needs_additional_iteration": True,
                },
            )

        # FAST-PATH: Skip LLM verification for simple read-only tasks
        task_complexity = state.get("task_complexity", "simple")
        planning_notes = state.get("planning_notes", [])
        is_read_only = any("[READ-ONLY]" in note for note in planning_notes)

        last_message = messages[-1]

        # Fast-path conditions:
        # 1. Simple read-only task
        # 2. Agent responded with text (no tool calls)
        # 3. Response is substantial (not an error)
        if task_complexity == "simple" and is_read_only:
            if isinstance(last_message, AIMessage):
                content = getattr(last_message, "content", "")
                has_tool_calls = bool(getattr(last_message, "tool_calls", None))

                if not has_tool_calls and content and len(content) > 50:
                    # Simple read-only task completed - skip LLM verification
                    return {
                        "verification_notes": ["✅ Fast-path: 읽기 전용 작업 완료"],
                        "needs_additional_iteration": False
                    }

        # Quick check for plan without execution
        plan_pending = state.get("plan_execution_pending")
        if plan_pending or (state.get("plan_created") and no_tool_activity):
            plan_steps = state.get("plan_steps", [])
            current_step = state.get("current_plan_step", 0)
            step_instruction = ""
            if plan_steps and 0 <= current_step < len(plan_steps):
                step_instruction = f"Execute step: '{plan_steps[current_step]}'."
            else:
                step_instruction = "Execute the next step using the appropriate tool."

            reminder = SystemMessage(
                content=(
                    "Execution plan is ready. Call tools (find_file, file_read, file_edit, "
                    "bash_execute) to execute the plan steps. "
                    "Do not repeat the plan — take action."
                )
            )
            execute_msg = make_internal_human_message(
                content=f"{step_instruction} Call the appropriate tool now."
            )
            return {
                "messages": [reminder, execute_msg],
                "verification_notes": ["Plan presented but not executed yet"],
                "needs_additional_iteration": True
            }

        # Skip required_files check for READ-ONLY tasks or when no file changes expected
        # (planning_notes and is_read_only already extracted above)
        # Check both file_changes (from state_helpers) and modified_files (from git snapshot)
        file_changes = state.get("file_changes", [])
        modified_files = state.get("modified_files", [])
        substantive_file_changes, substantive_modified_files = ReactAgent._collect_substantive_changes(
            file_changes,
            modified_files,
        )
        has_file_changes = bool(substantive_file_changes) or bool(substantive_modified_files)

        required_files = state.get("required_files", [])
        # Only check required_files for MODIFICATION tasks that should have file changes
        if required_files and not is_read_only and has_file_changes:
            # Get changed file paths from both sources
            changed_files = list(substantive_file_changes)
            changed_files.extend(substantive_modified_files)
            missing = []
            for req in required_files:
                matched = any(self._path_matches_requirement(path, req) for path in changed_files)
                if not matched:
                    missing.append(req)
            if missing:
                # Prevent infinite loop: after 3 iterations or 3 tool calls, just warn and continue
                # This allows the agent to proceed even if required_files heuristic was incorrect
                if iteration_count >= 3 or len(tool_call_history) >= 3:
                    # Log warning but don't block progress
                    self.logger.log_trace("required_files_warning", {
                        "missing_files": missing,
                        "iteration": iteration_count,
                        "tools_executed": len(tool_call_history)
                    })
                    if self.console:
                        self.console.print(
                            f"[yellow]⚠️ Required files not yet modified: {', '.join(missing)}[/yellow]"
                        )
                    # Clear required_files to avoid repeating this warning
                    updates["required_files"] = []
                else:
                    # Early iterations: request file modification
                    reminder = make_internal_human_message(
                        content=(
                            "아직 다음 파일이 요구사항을 충족하도록 수정되지 않았습니다: "
                            f"{', '.join(missing)}. 지금 해당 파일들을 업데이트하세요."
                        )
                    )
                    return {
                        "messages": [reminder],
                        "required_files": missing,
                        "verification_notes": ["⚠️ 요구된 파일 수정 미완료"],
                        "needs_additional_iteration": True
                    }
            else:
                updates["required_files"] = []

        # Optional: 테스트/린트 실행을 강제 요청 (환경 설정 기반)
        tests_needed = getattr(self.settings, "auto_verify_tests", False)
        lint_needed = getattr(self.settings, "auto_verify_lint", False)
        if file_changes and (tests_needed or lint_needed):
            requests: list[BaseMessage] = []
            notes: list[str] = []

            if tests_needed and not state.get("tests_requested"):
                test_cmd = getattr(self.settings, "test_command", "pytest")
                requests.append(make_internal_human_message(
                    content=f"코드가 수정되었습니다. bash_execute(command=\"{test_cmd}\")로 테스트를 실행하세요."
                ))
                notes.append(f"테스트 실행 요청: {test_cmd}")
                updates["tests_requested"] = True

            if lint_needed and not state.get("lint_requested"):
                lint_cmd = getattr(self.settings, "lint_command", "ruff check .")
                requests.append(make_internal_human_message(
                    content=f"린트를 수행하세요. bash_execute(command=\"{lint_cmd}\")로 실행하십시오."
                ))
                notes.append(f"린트 실행 요청: {lint_cmd}")
                updates["lint_requested"] = True

            if requests:
                updates["needs_additional_iteration"] = True
                updates["verification_notes"] = (updates.get("verification_notes", []) or []) + notes
                return self._merge_updates(updates, {"messages": requests})

        # LangGraph Multi-Agent Pattern: Use LLM-based Verification Specialist
        plan_steps = state.get("plan_steps", [])
        # file_changes already defined above
        last_message = messages[-1]
        content = getattr(last_message, "content", "")

        # Note: Using lightweight check_task_completion() for verification
        # Full LLM-based verification available if needed (more expensive but thorough)

        # If we just executed a tool, use LLM to check if task is complete
        if isinstance(last_message, ToolMessage):
            # Skip expensive LLM completion check in early iterations of complex tasks
            # — the task is clearly not done yet (e.g., still in find/read phase)
            current_step = state.get("current_plan_step", 0)
            skip_completion_check = (
                (task_complexity == "complex"
                 and iteration_count <= 2
                 and len(plan_steps) > 2
                 and current_step < len(plan_steps) - 1)
                or iteration_count == 0  # First iteration — always skip completion check
            )

            if skip_completion_check:
                # Clearly not done — advance without LLM call
                return {
                    "verification_notes": ["🔄 Early iteration — skipping completion check"],
                    "needs_additional_iteration": True,
                }

            # Get the latest user query (most relevant for completion check)
            user_query = get_current_user_query(state)

            # Get tool result and name
            tool_result = content if content else ""
            tool_name = getattr(last_message, "name", "unknown")

            # Progressive Verification: heuristic checks before LLM call
            # If search found no results, definitely incomplete (skip LLM)
            if tool_name in ("search_content", "find_file", "find_definition"):
                tool_text = str(tool_result).lower()
                if any(neg in tool_text for neg in ["no results", "no matches", "not found", "[]", "결과 없"]):
                    return {
                        "verification_notes": ["🔍 Search returned no results — continuing"],
                        "needs_additional_iteration": True,
                    }

            # bash_execute with test-like output containing "passed" → likely complete
            if tool_name == "bash_execute" and tool_result:
                tool_text = str(tool_result).lower()
                if any(kw in tool_text for kw in ["passed", "ok", "success", "0 failures"]):
                    # High signal of completion — still verify with LLM but boost confidence
                    pass  # Fall through to LLM check with natural confidence

            # SWE-bench: Never allow completion without file changes
            _vn_profile = getattr(self, "prompt_profile", "")
            max_iterations = state.get("max_iterations", self.settings.max_iterations)
            if (_vn_profile.startswith("swe_bench")
                    and not has_file_changes
                    and iteration_count < max_iterations):
                completion_result = {"complete": False, "reason": "SWE-bench requires file changes"}
            else:
                # Use LLM-based completion check (verifier_llm may be cheaper)
                from sepilot.agent.agent_utils import build_file_change_summary
                file_change_summary = build_file_change_summary(
                    file_changes=list(substantive_file_changes),
                    modified_files=list(substantive_modified_files),
                )
                completion_result = check_task_completion(
                    llm=self.verifier_llm,
                    user_query=user_query,
                    tool_result=tool_result,
                    tool_name=tool_name,
                    iteration_count=iteration_count,
                    max_iterations=max_iterations,
                    has_file_changes=has_file_changes,
                    file_change_summary=file_change_summary,
                )

            is_complete = completion_result.get("complete", False)
            completion_reason = completion_result.get("reason", "")
            raw_confidence = completion_result.get("confidence", 0.5)

            # Calibrate confidence to prevent weak model over-reporting
            confidence = self._calibrate_confidence(
                state, raw_confidence, is_complete
            )

            # Use calibrated confidence for is_complete override
            if is_complete and confidence < 0.4:
                is_complete = False
                completion_reason = f"calibrated_low_confidence ({confidence:.2f})"

            if self.console and self.verbose:
                from rich.panel import Panel
                from rich.text import Text
                lines = Text()
                lines.append(f"📥 Query: {self._truncate_text(user_query, 100)}\n")
                lines.append(f"📥 Tool: {tool_name}\n")
                lines.append(f"📥 Result: {self._truncate_text(str(tool_result), 150)}\n")
                lines.append(
                    f"📤 Complete: {is_complete} | Reason: {completion_reason} | "
                    f"Confidence: {confidence:.0%}",
                    style="bold green" if is_complete else "bold yellow",
                )
                self.console.print(Panel(
                    lines,
                    title="[bold]🧠 Verifier (Completion Check)[/bold]",
                    border_style="blue",
                    padding=(0, 1),
                ))

            # Log completion check
            self.logger.log_trace("task_completion_check", {
                "complete": is_complete,
                "reason": completion_reason,
                "confidence": confidence,
                "iteration": iteration_count,
                "tool_name": tool_name
            })

            if is_complete:
                note = f"✅ Task completed: {completion_reason}"

                # Check if we already requested a summary (prevent infinite loop)
                if state.get("_task_complete_pending_response"):
                    # Already asked for summary, now truly complete
                    return {
                        "verification_notes": [note],
                        "needs_additional_iteration": False
                    }

                # Patch self-review: before declaring completion, verify patch quality
                # Only for coding tasks with file changes, and only once per completion attempt
                if (has_file_changes
                        and not state.get("_patch_review_done")
                        and confidence >= 0.5):
                    try:
                        import subprocess

                        from sepilot.agent.agent_utils import verify_patch_quality
                        # Get git diff with extended context for better review
                        diff_result = subprocess.run(
                            ["git", "diff", "-U10", "HEAD"],
                            capture_output=True, text=True, timeout=10,
                            cwd=os.getcwd(),
                        )
                        file_diffs = diff_result.stdout if diff_result.returncode == 0 else ""
                        if not file_diffs:
                            # Try diff of unstaged changes
                            diff_result = subprocess.run(
                                ["git", "diff", "-U10"],
                                capture_output=True, text=True, timeout=10,
                                cwd=os.getcwd(),
                            )
                            file_diffs = diff_result.stdout or ""

                        if file_diffs:
                            review = verify_patch_quality(
                                llm=self.verifier_llm,
                                user_query=user_query,
                                file_diffs=file_diffs,
                                modified_files=list(substantive_modified_files),
                            )
                            if not review.get("approved", True):
                                issues = review.get("issues", [])
                                issue_text = "; ".join(issues) if issues else "Patch quality concerns"
                                if self.console:
                                    self.console.print(
                                        f"[bold yellow]🔍 Patch review: NEEDS_FIX — {issue_text}[/bold yellow]"
                                    )
                                return {
                                    "messages": [make_internal_human_message(
                                        f"Patch review identified issues: {issue_text}\n"
                                        "Please review and fix your changes before completing."
                                    )],
                                    "verification_notes": [f"🔍 Patch review failed: {issue_text}"],
                                    "needs_additional_iteration": True,
                                    "_patch_review_done": True,
                                }
                            else:
                                if self.console and self.verbose:
                                    self.console.print(
                                        "[dim green]✓ Patch review: APPROVED[/dim green]"
                                    )
                    except Exception as e:
                        if self.console and self.verbose:
                            self.console.print(
                                f"[dim yellow]⚠️ Patch review skipped: {e}[/dim yellow]"
                            )

                # High-confidence completion: skip summary only if agent already
                # provided a text response in the current execution.
                # Without a text response, the reporter has nothing to show
                # and falls back to a generic "작업이 완료되었습니다." message.
                # Note: use raw_confidence (pre-calibration) for summary skip threshold
                if raw_confidence >= 0.9:
                    boundary_idx = self._find_execution_boundary(state)
                    has_text_response = any(
                        isinstance(msg, AIMessage)
                        and not getattr(msg, 'tool_calls', None)
                        and getattr(msg, 'content', '').strip()
                        for msg in state.get("messages", [])[boundary_idx:]
                    )
                    if has_text_response:
                        return {
                            "verification_notes": [note],
                            "needs_additional_iteration": False
                        }

                # Lower confidence: give agent one more iteration to summarize
                summary_request = make_internal_human_message(
                    content=(
                        "Tool execution completed successfully. "
                        "Now provide a clear, concise response to the user summarizing the result. "
                        "Do NOT call any more tools - just respond with the answer."
                    )
                )
                return {
                    "messages": [summary_request],
                    "verification_notes": [note],
                    "needs_additional_iteration": True,  # One more iteration for agent to respond
                    "_task_complete_pending_response": True  # Flag to prevent infinite loop
                }
            else:
                note = f"🔄 Continuing: {completion_reason}"
                updates = {
                    "verification_notes": [note],
                    "needs_additional_iteration": True
                }

                # Advance plan step only when there was actual tool execution progress.
                # This prevents checklist from showing artificial completion before work runs.
                plan_steps = state.get("plan_steps", [])
                current_step = state.get("current_plan_step", 0)
                if execution_progress and plan_steps and current_step < len(plan_steps) - 1:
                    updates["current_plan_step"] = current_step + 1

                return updates

        # Check for errors in AIMessage
        needs_more = False
        consecutive_llm_errors = state.get("consecutive_llm_errors", 0)

        # Transient LLM/API failures should trigger another iteration while retries remain.
        if consecutive_llm_errors > 0 and not state.get("force_termination", False):
            return {
                "verification_notes": ["🔄 Retrying after transient LLM/API failure"],
                "needs_additional_iteration": True,
            }

        # SWE-bench: text-only AIMessage with no file edits → ALWAYS continue
        # The task is definitively not done; dashboard + self-assessment guide the LLM
        # Escalate urgency when many iterations consumed without any code changes
        _vn_profile = getattr(self, "prompt_profile", "")
        if (isinstance(last_message, AIMessage)
                and _vn_profile.startswith("swe_bench")
                and not has_file_changes):
            max_iter = state.get("max_iterations", self.settings.max_iterations)
            if iteration_count >= max_iter // 2:
                # Half of iterations consumed with zero code changes — provide progress context
                advisory_msg = make_internal_human_message(
                    content=(
                    f"Progress note: {iteration_count}/{max_iter} iterations used. "
                    "No code changes have been made yet. "
                    "Consider applying file modifications based on your analysis so far."
                ))
                return {
                    "messages": [advisory_msg],
                    "verification_notes": [f"SWE-bench: no edits at iter {iteration_count}, advising code action"],
                    "needs_additional_iteration": True,
                }
            return {
                "verification_notes": ["SWE-bench: no file edits yet, continuing"],
                "needs_additional_iteration": True,
            }

        if isinstance(last_message, AIMessage) and isinstance(content, str):
            # Empty assistant responses are never a valid completion signal.
            # In coding strategies this often means the model stopped mid-task.
            if not content.strip():
                current_strategy = state.get("current_strategy", AgentStrategy.EXPLORE)
                if isinstance(current_strategy, str):
                    with contextlib.suppress(ValueError):
                        current_strategy = AgentStrategy(current_strategy.lower())
                coding_strategies = {AgentStrategy.IMPLEMENT, AgentStrategy.DEBUG, AgentStrategy.REFACTOR, AgentStrategy.TEST}
                if current_strategy in coding_strategies and iteration_count < 15:
                    reminder = make_internal_human_message(
                        content=(
                            "Your previous response was empty. Continue the coding task: "
                            "complete the required file updates and provide a clear final summary. "
                            "Do not finish yet."
                        )
                    )
                    return self._merge_updates(updates, {
                        "messages": [reminder],
                        "verification_notes": ["⚠️ Empty assistant response during coding task"],
                        "needs_additional_iteration": True,
                    })
                needs_more = True

            lowered = content.lower()

            # Skip error-pattern retry if we're already in an LLM error state
            # This prevents the agent→verifier→guard→agent infinite loop
            is_llm_error_msg = (
                consecutive_llm_errors > 0
                or "error calling the llm" in lowered
            )

            if not is_llm_error_msg:
                # Check for error indicators (avoid false positives from success messages)
                import re
                error_patterns = [
                    r'\berror\b', r'\bfail(ed|ure)?\b', r'\bunable\b',
                    r'\bmissing\b', r'\bnot found\b', r'\binvalid\b'
                ]
                # Negative context: phrases that mention errors in a resolved/success context
                false_positive_patterns = [
                    r'\b(fixed|resolved|corrected|handled|cleared)\s+(the\s+)?error',
                    r'error\s+(has been|was|is)\s+(fixed|resolved|corrected|handled)',
                    r'no\s+(more\s+)?errors?\b',
                    r'error\s+(handling|recovery|check)',
                    r'without\s+(any\s+)?errors?\b',
                ]
                has_error = any(re.search(pattern, lowered) for pattern in error_patterns)
                is_false_positive = any(re.search(fp, lowered) for fp in false_positive_patterns)

                if has_error and not is_false_positive:
                    needs_more = True

            # LLM-based autonomous verification for modification tasks
            # Only apply for coding strategies (not explore/document) and not during LLM errors
            current_strategy = state.get("current_strategy", AgentStrategy.EXPLORE)
            if isinstance(current_strategy, str):
                with contextlib.suppress(ValueError):
                    current_strategy = AgentStrategy(current_strategy.lower())
            coding_strategies = {AgentStrategy.IMPLEMENT, AgentStrategy.DEBUG, AgentStrategy.REFACTOR, AgentStrategy.TEST}

            if (not is_llm_error_msg
                    and current_strategy in coding_strategies
                    and not has_file_changes and not is_read_only and iteration_count < 15):
                # Require concrete edits only when the request clearly asks for coding work.
                original_prompt = get_current_user_query(state)

                _vn_profile = getattr(self, "prompt_profile", "")
                analysis_only = self._is_analysis_only_request(original_prompt)
                coding_required = (
                    _vn_profile.startswith("swe_bench")
                    or (not original_prompt.strip())
                    or (self._is_explicit_coding_request(original_prompt) and not analysis_only)
                )

                if coding_required:
                    reminder = make_internal_human_message(
                        content=(
                            "This is a coding/fix task and no substantive project files were modified yet. "
                            "Apply concrete code edits now (for example with file_edit/file_write), "
                            "then verify the result. Do not finish yet."
                        )
                    )
                    return self._merge_updates(updates, {
                        "messages": [reminder],
                        "verification_notes": ["⚠️ Coding task incomplete: no substantive file changes detected"],
                        "needs_additional_iteration": True,
                    })

        note = "⚠️ 추가 검토 필요" if needs_more else "✅ 검증 완료"
        verification_notes = [note]
        if substantive_file_changes:
            changed_paths = substantive_file_changes
            preview = ", ".join(changed_paths[:3])
            verification_notes.append(f"변경 파일 {len(changed_paths)}개: {preview}")
        base_update = {
            "verification_notes": verification_notes,
            "needs_additional_iteration": needs_more
        }
        return self._merge_updates(updates, base_update)

    def _verification_next_step(self, state: EnhancedAgentState) -> Literal["continue", "fast_continue", "report"]:
        """Route verification outcome.

        Routes to:
        - fast_continue: Early iterations (<=2) skip reflection/backtrack/debate
        - continue: Later iterations go through full reflection pipeline
        - report: Task is complete
        """
        if state.get("needs_additional_iteration"):
            iteration = state.get("iteration_count", 0)
            if iteration <= 2:
                return "fast_continue"  # Skip reflection/backtrack/debate in early iterations
            return "continue"
        return "report"

    def _verification_next_step_simplified(self, state: EnhancedAgentState) -> Literal["continue", "report"]:
        """Simplified verifier routing: continue loop or report (no reflection pipeline)."""
        if state.get("needs_additional_iteration"):
            return "continue"
        return "report"

    def _reflection_next_step(
        self, state: EnhancedAgentState
    ) -> Literal["revise_plan", "refine_strategy", "proceed", "escalate"]:
        """Route reflection outcome based on self-critique analysis.

        Routes to:
        - revise_plan: Go back to planner with new insights
        - refine_strategy: Adjust strategy and retry agent
        - proceed: Continue normal iteration flow
        - escalate: Stop and report (needs human intervention)
        """
        decision = state.get("reflection_decision")

        if decision == ReflectionDecision.REVISE_PLAN.value:
            if self.console and self.verbose:
                self.console.print("[yellow]📋 Reflection: Revising execution plan...[/yellow]")
            return "revise_plan"

        elif decision == ReflectionDecision.REFINE_STRATEGY.value:
            if self.console and self.verbose:
                strategy = state.get("current_strategy")
                self.console.print(
                    f"[yellow]🔄 Reflection: Refining strategy to {strategy}...[/yellow]"
                )
            return "refine_strategy"

        elif decision == ReflectionDecision.ESCALATE.value:
            if self.console and self.verbose:
                self.console.print(
                    "[red]🆘 Reflection: Escalating - may need human guidance[/red]"
                )
            return "escalate"

        # Default: proceed with normal flow
        return "proceed"

    def _backtrack_next_step(
        self, state: EnhancedAgentState
    ) -> Literal["rollback", "continue"]:
        """Route backtrack check outcome.

        Routes to:
        - rollback: Go back to planner and restore previous state
        - continue: Proceed to debate check
        """
        # Check backtrack_decision first (set by BacktrackCheckNode)
        backtrack_decision = state.get("backtrack_decision")
        backtrack_performed = state.get("backtrack_performed", False)

        if backtrack_decision == "rollback" and backtrack_performed:
            if self.console and self.verbose:
                reason = state.get("backtrack_reason", "unknown")
                files = state.get("files_restored", [])
                self.console.print(
                    f"[yellow]⏪ Backtracking: Rolling back due to {reason} "
                    f"({len(files)} files restored)[/yellow]"
                )
            return "rollback"

        return "continue"

    def _debate_check_next_step(
        self, state: EnhancedAgentState
    ) -> Literal["debate", "skip"]:
        """Route debate check outcome.

        Debate is selectively enabled for code review tasks only.
        Multi-perspective analysis adds latency but is valuable for reviews.
        """
        # Check if debate pattern is active AND task is code review
        active_patterns = state.get("active_patterns", [])
        task_type = state.get("task_type_detected", "")
        debate_decision = state.get("debate_decision")

        # Only enable debate for code review tasks with high complexity
        if "debate" in active_patterns and task_type == "code_review":
            if debate_decision == "debate":
                if self.console and self.verbose:
                    self.console.print("[cyan]🎭 Debate: Multi-perspective analysis for code review...[/cyan]")
                return "debate"

        return "skip"

    def _memory_writer_next_step(
        self, state: EnhancedAgentState
    ) -> Literal["continue", "report"]:
        """Route memory writer outcome.

        Routes to:
        - continue: More iterations needed
        - report: All done, generate final report
        """
        # Force termination takes priority (iteration limit reached)
        if state.get("force_termination"):
            return "report"

        # Check if more iterations are explicitly needed
        if state.get("needs_additional_iteration"):
            return "continue"

        return "report"

    def _reporting_node(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Emit a final summary message."""
        # Get actual data from state
        tool_call_history = state.get("tool_call_history", [])
        # Check both file_changes (from state_helpers) and modified_files (from git snapshot)
        file_changes = state.get("file_changes", [])
        modified_files = state.get("modified_files", [])
        all_file_changes = list(file_changes) if file_changes else []
        if modified_files:
            # Add modified_files as simple file paths
            all_file_changes.extend(modified_files)
        task_history = state.get("task_history", [])
        error_history = state.get("error_history", [])
        verification_notes = state.get("verification_notes", [])

        # Find the last agent response (AIMessage without tool_calls)
        # This is the actual analysis/answer the agent provided to the user
        # IMPORTANT: Only search in messages from the CURRENT execution,
        # not in context_messages from previous conversations.
        # Without this boundary, the reporter may pick up an AIMessage from
        # a prior exchange (e.g., an old ingress table) when the current
        # execution only produced tool-call AIMessages.
        # NOTE: Uses ID-based boundary lookup to handle checkpoint message
        # accumulation across multiple execute() calls.
        all_messages = state.get("messages", [])
        boundary_idx = self._find_execution_boundary(state)
        current_messages = all_messages[boundary_idx:] if boundary_idx > 0 else all_messages

        last_agent_response = None
        for msg in reversed(current_messages):
            if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                # Make sure it's not a reporter message (simple heuristic)
                content = msg.content.strip()
                if content and not content.startswith("📋 작업 요약") and content != "작업이 완료되었습니다.":
                    last_agent_response = content
                    break

        latest_tool_output = ""
        for msg in reversed(current_messages):
            if isinstance(msg, ToolMessage):
                content = str(getattr(msg, "content", "")).strip()
                if content:
                    latest_tool_output = content
                    break

        # Detect stale mode-restricted responses: the LLM said "can't execute in X mode"
        # but tools actually ran successfully after an automatic mode switch.
        _stale_mode_response = False
        if last_agent_response and latest_tool_output:
            resp_lower = last_agent_response.lower()
            if ("모드" in resp_lower or "mode" in resp_lower) and (
                "실행할 수 없" in resp_lower
                or "전환" in resp_lower
                or "not available" in resp_lower
                or "cannot execute" in resp_lower
            ):
                _stale_mode_response = True

        # In non-verbose mode, prioritize agent response over metadata
        if not self.verbose:
            if last_agent_response and not _stale_mode_response:
                # Agent provided meaningful response (analysis, review, explanation, etc.)
                # This is what the user wants to see - return it directly
                # Note: Don't send to chat here - it was already sent in _agent_node
                return {"messages": [AIMessage(content=last_agent_response)]}
            elif file_changes:
                # No agent response but file changes - show changes
                lines = ["작업 완료:"]
                for fc in file_changes[:5]:
                    action = fc.action.value if hasattr(fc.action, 'value') else str(fc.action)
                    lines.append(f"  {fc.file_path} ({action})")
                if len(file_changes) > 5:
                    lines.append(f"  ... 외 {len(file_changes) - 5}개")
                if verification_notes:
                    lines.append(f"검증 메모: {verification_notes[-1]}")
                report = "\n".join(lines)
                self._send_chat_response(report)
                return {"messages": [AIMessage(content=report)]}
            elif latest_tool_output:
                # No plain AI summary, but tool output exists: return a useful fallback
                report = "도구 실행 결과 요약:\n" + self._truncate_text(latest_tool_output, 1800)
                self._send_chat_response(report)
                return {"messages": [AIMessage(content=report)]}
            else:
                # Nothing to show
                report = "작업이 완료되었습니다."
                self._send_chat_response(report)
                return {"messages": [AIMessage(content=report)]}

        # Verbose mode: show agent response + detailed metadata
        highlights = [
            f"- Tasks completed: {len(task_history)}",
            f"- Files touched: {len(file_changes)}",
            f"- Tool calls: {len(tool_call_history)}"
        ]

        # Generate detailed summary
        detail_lines = []

        # Summarize file changes
        if file_changes:
            detail_lines.append("파일 변경사항:")
            for fc in file_changes[:5]:  # Show first 5
                action = fc.action.value if hasattr(fc.action, 'value') else str(fc.action)
                detail_lines.append(f"  • {action}: {fc.file_path}")
            if len(file_changes) > 5:
                detail_lines.append(f"  ... 외 {len(file_changes) - 5}개")

        # Summarize tool calls
        if tool_call_history:
            detail_lines.append("\n도구 호출:")
            tool_counts = {}
            for tc in tool_call_history:
                tool_name = tc.tool_name if hasattr(tc, 'tool_name') else str(tc)
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:5]:
                detail_lines.append(f"  • {tool}: {count}회")

        # Summarize errors if any
        if error_history:
            warning_count = sum(1 for e in error_history if hasattr(e, 'level') and e.level == ErrorLevel.WARNING)
            error_count = len(error_history) - warning_count
            if error_count > 0:
                detail_lines.append(f"\n⚠️ 오류 {error_count}개 발생 (경고 {warning_count}개)")

        # Get conversation summary if available
        conversation_summary = state.get("conversation_summary")
        if conversation_summary:
            detail_lines.append(f"\n대화 요약: {conversation_summary}")

        # If no detailed info available
        if not detail_lines:
            if len(tool_call_history) == 0:
                detail_lines.append("도구가 호출되지 않았습니다.")
            else:
                detail_lines.append("작업이 완료되었습니다.")

        detailed_summary = "\n".join(detail_lines) if detail_lines else "상세 정보 없음"

        metadata_summary = (
            "📋 작업 요약\n"
            + "\n".join(highlights)
            + "\n\n세부 요약:\n"
            + detailed_summary
        )

        # Combine agent response (if any) with metadata
        if last_agent_response and not _stale_mode_response:
            # Agent provided meaningful response - show it first, then metadata
            report = last_agent_response + "\n\n" + "─" * 50 + "\n\n" + metadata_summary
            # Note: Don't send to chat here - it was already sent in _agent_node
        else:
            # No agent response - send metadata as chat message
            report = metadata_summary
            self._send_chat_response(metadata_summary)

        return {"messages": [AIMessage(content=report)]}

    def _find_execution_boundary(self, state: dict) -> int:
        """Find the index that separates context/previous messages from current execution.

        Uses _execution_boundary_msg_id (HumanMessage prompt ID) to locate the
        boundary robustly, even when LangGraph checkpoint accumulates messages
        across multiple execute() calls via the add_messages reducer.

        Falls back to _initial_message_count if boundary ID is not set (legacy).

        Returns the index of the first message AFTER the boundary (i.e., the
        start of current execution messages).
        """
        return find_execution_boundary(state)

    @staticmethod
    def _make_user_prompt_message(content: str, *, message_id: str | None = None) -> HumanMessage:
        """Create the persisted user-turn boundary message for an execution."""
        return make_user_prompt_message(content, message_id=message_id)

    @staticmethod
    def _make_internal_human_message(content: str) -> HumanMessage:
        """Create an internal control prompt that should not count as a user turn."""
        return make_internal_human_message(content)

    # Delegate to agent_utils for analysis utilities
    _looks_like_plan = staticmethod(looks_like_plan)
    _extract_token_usage = staticmethod(extract_token_usage)
    _extract_required_files = staticmethod(extract_required_files)
    _path_matches_requirement = staticmethod(path_matches_requirement)
    _should_skip_planning = staticmethod(should_skip_planning)

    def _estimate_message_tokens(self, messages: list) -> int:
        """Estimate token count for a list of messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Estimated total token count
        """
        try:
            from sepilot.agent.context_manager import ContextManager
            cm = ContextManager(max_context_tokens=128000)
            return cm._estimate_tokens(messages)
        except Exception:
            # Fallback: ~4 chars per token + overhead per message
            total_chars = sum(len(getattr(m, 'content', '') or '') for m in messages)
            return (total_chars // 4) + (len(messages) * 4)

    def _track_token_usage(self, response: Any, source: str = "agent") -> tuple[int, int]:
        """Extract and track token usage from LLM response.

        Args:
            response: LLM response message
            source: Source identifier for logging (agent, planner, verifier, etc.)

        Returns:
            Tuple of (total_tokens, output_tokens)
        """
        input_tokens = 0
        output_tokens = 0
        model_name = None

        # Extract detailed token usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        elif hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            if 'token_usage' in metadata:
                token_usage = metadata['token_usage']
                input_tokens = token_usage.get('prompt_tokens', token_usage.get('input_tokens', 0))
                output_tokens = token_usage.get('completion_tokens', token_usage.get('output_tokens', 0))
            # Try to get model name
            model_name = metadata.get('model_name', metadata.get('model'))

        # Use settings model name if not found in response
        if not model_name:
            model_name = str(self.settings.model) if hasattr(self, 'settings') else None

        total_tokens = input_tokens + output_tokens

        # Send to web monitor if available
        # Record usage in cost tracker
        if hasattr(self, 'cost_tracker') and self.cost_tracker and total_tokens > 0:
            try:
                self.cost_tracker.record_usage(
                    model=model_name or "unknown",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tool_name=source
                )
            except Exception:
                pass  # Don't let cost tracker errors break execution

        return total_tokens, output_tokens

    def _send_chat_response(self, content: str):
        """Send AI response to chat (web monitor)

        Args:
            content: AI response content
        """
    def _agent_node(self, state):
        """Agent node - calls LLM with tools

        Handles both normal operation and error recovery:
        - If previous message is ToolMessage with error, LLM sees it and can retry
        - LLM errors are caught and returned as AIMessage
        """
        messages = state["messages"]
        updates: dict[str, Any] = {}

        # Note: iteration_count is incremented in _iteration_guard_node at the start of each cycle
        # Here we just use the current count for logging and checks
        # Check if we need to force tool execution after planning
        tool_call_history = state.get("tool_call_history", [])
        planning_notes = state.get("planning_notes", [])

        iteration_count = state.get("iteration_count", 0)
        # Sync instance counter with state (single source of truth)
        self.iteration_count = iteration_count

        # If we have a plan but no tool calls yet, guide to execute
        if planning_notes and len(tool_call_history) == 0 and iteration_count <= 2:
            guide_msg = SystemMessage(
                content="Execute the plan by calling the appropriate tools or providing analysis."
            )
            messages = messages + [guide_msg]

        if state.get("iteration_count", 0) >= self.settings.max_iterations:
            warning_msg = (
                f"Iteration budget {self.settings.max_iterations} reached. "
                "I stopped to avoid infinite loops. Please adjust the request."
            )
            err_delta = state_helpers.record_error(
                state,
                message=warning_msg,
                level=ErrorLevel.WARNING,
                source="system",
                context={"iteration": state.get("iteration_count")},
                return_delta=True
            )
            updates = self._merge_updates(updates, err_delta)
            return self._merge_updates(
                updates,
                {"messages": [AIMessage(content=warning_msg)]}
            )

        # Check memory usage periodically
        current_memory_mb = 0.0
        if self.iteration_count % self.memory_check_interval == 0:
            memory_warning = self.memory_monitor.check_memory()
            if memory_warning:
                current_memory_mb = memory_warning.get('rss_mb', 0.0)
                if self.console:
                    self.console.print(
                        f"[yellow]💾 {memory_warning['message']} "
                        f"({memory_warning['percent']:.1f}% of system memory)[/yellow]"
                    )
                self.logger.log_trace("memory_warning", memory_warning)

        # Update status panel
        if self.status_panel:
            cache_stats = self.tool_cache.get_stats() if hasattr(self, 'tool_cache') else {}

            # Get plan progress from enhanced state
            plan_steps = state.get("plan_steps", [])
            current_plan_step = state.get("current_plan_step", 0)

            # Use plan progress if available, otherwise fall back to iteration count
            if plan_steps:
                # Plan-based progress
                iteration = current_plan_step + 1  # Convert to 1-based for display
                max_iterations = len(plan_steps)
            else:
                # Fallback to iteration-based progress
                iteration = self.iteration_count
                max_iterations = self.settings.max_iterations

            # Get current task from enhanced state
            current_task = "Thinking..."
            if messages:
                # Show current plan step if available
                if plan_steps and 0 <= current_plan_step < len(plan_steps):
                    current_task = self._truncate_text(plan_steps[current_plan_step], 50)
                else:
                    last_msg = messages[-1]
                    if isinstance(last_msg, HumanMessage):
                        current_task = self._truncate_text(last_msg.content, 50)
                    elif isinstance(last_msg, AIMessage) and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        tool_names = [tc.get('name', 'unknown') for tc in last_msg.tool_calls[:2]]
                        current_task = f"Calling {', '.join(tool_names)}"

            self.status_panel.update(
                session_id=getattr(self, 'thread_id', 'N/A'),
                iteration=iteration,
                max_iterations=max_iterations,
                current_task=current_task,
                memory_mb=current_memory_mb if current_memory_mb > 0 else None,
                cache_stats=cache_stats if cache_stats else None
            )

        if self.console:
            # StatusPanel already shows progress, so just show a simple iteration marker
            # No need to duplicate iteration count here since StatusPanel displays it
            if not self.status_panel:
                # Show iteration count with token context info
                max_iterations = self.settings.max_iterations
                progress_pct = (self.iteration_count / max_iterations) * 100
                current_tokens = state.get("total_tokens_used", 0)
                context_window = getattr(self.settings, 'context_window', 128000)
                context_pct = (current_tokens / context_window) * 100 if context_window > 0 else 0
                estimated_cost = state.get("estimated_cost", 0.0)

                self.console.print(f"\n[bold blue]🤔 Iteration {self.iteration_count}/{max_iterations} ({progress_pct:.0f}%)[/bold blue]")
                if current_tokens > 0:
                    self.console.print(
                        f"[dim]   Context: {current_tokens:,} tokens ({context_pct:.1f}%) | "
                        f"Cost: ${estimated_cost:.4f}[/dim]"
                    )
            else:
                # Just a simple separator when StatusPanel is active
                self.console.print()

            # Show previous tool results if available
            if messages and isinstance(messages[-1], ToolMessage):
                tool_msg = messages[-1]
                content = str(tool_msg.content)

                # Check if it's an error (more sophisticated check)
                # LangGraph ToolNode returns errors in specific format
                is_error = (
                    content.startswith("Error:") or
                    content.startswith("Error executing tool") or
                    "Error invoking tool" in content or
                    (content.startswith("Error") and len(content) < 500)  # Short error messages
                )

                if is_error:
                    self.console.print("[yellow]⚠️  Recovering from tool error...[/yellow]")
                    self.console.print(f"[red]   Error: {self._truncate_text(content, 300)}[/red]")
                else:
                    # Success - show result
                    tool_name = getattr(tool_msg, 'name', 'unknown')
                    self.console.print(f"[green]✓ Tool '{tool_name}' completed[/green]")
                    self.console.print(f"[dim]   Result: {self._truncate_text(content, 300)}[/dim]")

        # LangGraph Multi-Agent Pattern: Add Specialized Execution Agent Prompt
        plan_steps = state.get("plan_steps", [])
        current_step = state.get("current_plan_step", 0)
        tool_call_history = state.get("tool_call_history", [])

        # Get tool recommendations from ToolRecommenderNode
        tool_recommendations = state.get("tool_recommendations", [])
        suggested_sequence = state.get("suggested_tool_sequence", [])

        # Get reflection insights for strategy adjustment
        reflection_notes = state.get("reflection_notes", [])
        current_strategy = state.get("current_strategy")
        reflection_decision = state.get("reflection_decision")

        # Build structured tool recommendation section
        tool_section = ""
        if tool_recommendations:
            tool_lines = []
            for i, r in enumerate(tool_recommendations[:3], 1):
                tool_lines.append(f"  {i}. {r['tool']} (confidence: {r['confidence']:.0%})")
            tool_section = "\n".join(tool_lines)
        if suggested_sequence:
            if tool_section:
                tool_section += "\n"
            tool_section += f"  Sequence: {' → '.join(suggested_sequence[:5])}"

        # Build reflection section for strategy guidance
        reflection_section = ""
        if reflection_decision and reflection_decision != "proceed":
            parts = []
            if current_strategy:
                strategy_name = current_strategy.value if hasattr(current_strategy, 'value') else str(current_strategy)
                parts.append(f"Strategy: {strategy_name.upper()}")
            if reflection_notes:
                for note in reflection_notes[:2]:
                    parts.append(f"- {note}")
            parts.append("IMPORTANT: Apply these insights NOW.")
            reflection_section = "\n".join(parts)

        # Build codebase context section
        exploration_ctx = state.get("exploration_context", "")
        codebase_section = exploration_ctx[:500] if exploration_ctx else "(none)"

        # Build progress dashboard and self-assessment (replaces hardcoded nudges)
        from sepilot.agent.progress_dashboard import (
            build_progress_dashboard,
            build_self_assessment,
        )
        _agent_ctx_win = getattr(getattr(self, 'settings', None), 'context_window', 128000)
        _agent_is_swe = getattr(self, "prompt_profile", "").startswith("swe_bench")
        progress = build_progress_dashboard(state, _agent_ctx_win, _agent_is_swe)
        assessment = build_self_assessment(state, _agent_is_swe, _agent_ctx_win)

        # Assemble structured system message with ═══ sections
        sections = [
            "═══ ROLE ═══",
            "You are an EXECUTION SPECIALIST. Call tools to accomplish the task.",
            "You MUST call at least one tool. Text-only responses will be retried.",
            "",
            progress,
            "",
            assessment,
            "",
            "═══ CURRENT TASK ═══",
            f"Step {current_step + 1}/{len(plan_steps) if plan_steps else 1}: "
            f"{plan_steps[current_step] if plan_steps and 0 <= current_step < len(plan_steps) else 'Execute task'}",
            "",
        ]

        if tool_section:
            sections.extend([
                "═══ RECOMMENDED TOOLS ═══",
                tool_section,
                "",
            ])

        if codebase_section != "(none)":
            sections.extend([
                "═══ CODEBASE CONTEXT ═══",
                codebase_section,
                "",
            ])

        if reflection_section:
            sections.extend([
                "═══ REFLECTION ═══",
                reflection_section,
                "",
            ])

        # Scratchpad summary injection
        scratchpad_entries = state.get("scratchpad_entries", [])
        if scratchpad_entries:
            from sepilot.config.constants import SCRATCHPAD_MAX_SYSTEM_PROMPT_CHARS
            from sepilot.tools.langchain_tools.think_tools import get_scratchpad_summary
            scratchpad_text = get_scratchpad_summary(scratchpad_entries, SCRATCHPAD_MAX_SYSTEM_PROMPT_CHARS)
            if scratchpad_text:
                sections.extend([
                    "═══ SCRATCHPAD ═══",
                    scratchpad_text,
                    "",
                ])

        planning_notes = state.get("planning_notes", [])
        is_read_only_request = any("[READ-ONLY]" in n for n in planning_notes)
        user_query = get_current_user_query(state)
        is_plan_request_query = is_plan_request(user_query)
        require_tool_each_turn = not (is_read_only_request or is_plan_request_query)

        # Inject planning notes (last 5) so LLM can see prior analysis
        if planning_notes:
            meaningful_notes = [n for n in planning_notes if not n.startswith("[READ-ONLY]")]
            if meaningful_notes:
                sections.extend([
                    "═══ PLANNING NOTES ═══",
                    "\n".join(f"- {n[:200]}" for n in meaningful_notes[-5:]),
                    "",
                ])

        # Inject latest verification note so LLM knows prior verification status
        verification_notes = state.get("verification_notes", [])
        if verification_notes:
            sections.extend([
                "═══ VERIFICATION STATUS ═══",
                verification_notes[-1][:300],
                "",
            ])

        # Inject Karpathy coding guidelines for code-related tasks
        # Two sources: keyword-based TaskAnalyzer (task_type_detected)
        #              + LLM-based triage (current_strategy)
        task_type_detected = state.get("task_type_detected", "")
        current_strategy = state.get("current_strategy")
        strategy_value = (
            current_strategy.value
            if hasattr(current_strategy, "value")
            else str(current_strategy or "")
        )
        from sepilot.skills.builtin.karpathy_guidelines import (
            KARPATHY_CODE_STRATEGIES,
            KARPATHY_CODE_TASK_TYPES,
            KARPATHY_GUIDELINES_PROMPT,
        )
        if (
            task_type_detected in KARPATHY_CODE_TASK_TYPES
            or strategy_value in KARPATHY_CODE_STRATEGIES
        ):
            sections.extend([
                "═══ CODING GUIDELINES ═══",
                KARPATHY_GUIDELINES_PROMPT,
                "",
            ])

        sections.extend([
            "═══ RULES ═══",
            (
                "1. Call at least one tool per turn"
                if require_tool_each_turn
                else "1. For plan/read-only requests, tools are optional; "
                     "if enough context is available, provide a direct structured answer."
            ),
            "2. Use file_edit for modifications, file_read to inspect",
            "3. Advance the task with each step — avoid repeating actions",
            "4. Use bash_execute(command=\"...\", cwd=\"path\") for directory-specific commands",
        ])

        # Inject mode prompt if active
        current_mode = state.get("current_mode", AgentMode.AUTO)
        _prompt_profile = getattr(self, "prompt_profile", "default")
        mode_prompt = get_mode_prompt(current_mode, prompt_profile=_prompt_profile)
        if mode_prompt:
            sections.insert(0, "═══ MODE ═══")
            sections.insert(1, mode_prompt)
            sections.insert(2, "")

        execution_specialist_msg = SystemMessage(content="\n".join(sections))

        # Prepend execution specialist message to guide behavior
        messages_with_context = [execution_specialist_msg] + messages

        # Pre-invocation context size check to prevent 500/EOF errors
        context_window = getattr(self.settings, 'context_window', 128000)
        estimated_tokens = self._estimate_message_tokens(messages_with_context)
        # Small-context models (≤32K): compact earlier (75%) with fewer kept messages
        # Token estimation can be inaccurate (tiktoken vs llama.cpp tokenizer)
        # so we need extra headroom to prevent context overflow errors.
        _compact_ratio = 0.75 if context_window <= 32768 else 0.95
        _keep_recent = 4 if context_window <= 32768 else 6
        if estimated_tokens > context_window * _compact_ratio:
            if self.console:
                self.console.print(
                    f"[yellow]⚠️ Context too large ({estimated_tokens:,} tokens, "
                    f"{context_window:,} limit). Emergency compaction...[/yellow]"
                )
            if self.step_logger:
                pct = int(estimated_tokens / context_window * 100)
                self.step_logger.log_context(f"emergency compaction ({pct}%)")
            # Emergency compaction
            from sepilot.agent.context_manager import ContextManager
            emergency_cm = ContextManager(max_context_tokens=context_window)
            compacted = emergency_cm.compact_messages(messages, keep_recent=_keep_recent)
            messages_with_context = [execution_specialist_msg] + compacted

            # Re-check after compaction
            estimated_after = self._estimate_message_tokens(messages_with_context)
            if estimated_after > context_window * _compact_ratio:
                # Still too large — graceful termination
                if self.console:
                    self.console.print(
                        "[red]❌ Context still too large after compaction. Terminating gracefully.[/red]"
                    )
                updates["force_termination"] = True
                abort_msg = AIMessage(
                    content="I've reached the context window limit and cannot process more information. "
                    "Here is a summary of what I found so far. Please start a new conversation "
                    "to continue."
                )
                return self._merge_updates(updates, {"messages": [abort_msg]})

        # Call LLM with tools
        try:
            # Optimized retry loop: only keep the final response to avoid context explosion
            final_response: BaseMessage | None = None
            accumulated_tokens = 0
            accumulated_output_tokens = 0
            accumulated_llm_elapsed = 0.0
            retries = 0
            max_plan_retries = 2

            _is_read_only = any("[READ-ONLY]" in n for n in state.get("planning_notes", []))

            # Use mode-filtered LLM (restricts available tools per mode)
            llm_to_use = self._get_mode_filtered_llm(current_mode)

            while True:
                _llm_t0 = time.monotonic()
                response = llm_to_use.invoke(messages_with_context)
                _llm_elapsed = time.monotonic() - _llm_t0
                _total, _out = self._track_token_usage(response, source="agent")
                accumulated_tokens += _total
                accumulated_output_tokens += _out
                accumulated_llm_elapsed += _llm_elapsed

                no_tool_response = (
                    isinstance(response, AIMessage)
                    and not getattr(response, "tool_calls", None)
                )

                # ===== Text-based tool call fallback =====
                if no_tool_response and response.content:
                    # Use mode-filtered tools for fallback to respect mode restrictions
                    mode_tools = get_mode_filtered_tools(self.langchain_tools, current_mode, _prompt_profile)
                    valid_names = {t.name for t in mode_tools}
                    fallback_calls = try_parse_text_tool_calls(
                        response.content, valid_names
                    )
                    if fallback_calls:
                        # CLI agents may output many tool calls at once — cap to prevent runaway
                        MAX_FALLBACK_TOOL_CALLS = 5
                        if len(fallback_calls) > MAX_FALLBACK_TOOL_CALLS:
                            self.logger.log_trace(
                                "fallback_tool_calls_capped",
                                {"from": len(fallback_calls), "to": MAX_FALLBACK_TOOL_CALLS},
                            )
                            fallback_calls = fallback_calls[:MAX_FALLBACK_TOOL_CALLS]
                        response = response.copy()
                        response.tool_calls = fallback_calls
                        no_tool_response = False
                        if self.console:
                            self.console.print(
                                f"[dim cyan]🔄 Fallback parser: {len(fallback_calls)} tool call(s) "
                                f"from text: {[tc['name'] for tc in fallback_calls]}[/dim cyan]"
                            )
                # ===== END FALLBACK =====

                # plan_only: retry when the response has no tool calls.
                # SWE-bench: retry ALL text-only responses (tools are mandatory).
                # Other profiles: retry only plan-like text (LLM may need to think).
                _is_swe = getattr(self, "prompt_profile", "").startswith("swe_bench")
                user_query = get_current_user_query(state)

                requires_plan_detail = is_plan_request(user_query)
                requires_testing_detail = requires_plan_detail and any(
                    kw in user_query.lower() for kw in ("test", "testing", "pytest", "verification")
                )
                insufficient_plan = (
                    no_tool_response
                    and requires_plan_detail
                    and not is_detailed_plan_response(
                        str(getattr(response, "content", "")),
                        require_testing=requires_testing_detail,
                    )
                )
                plan_text_is_expected = (
                    (requires_plan_detail or _is_read_only or current_mode == AgentMode.PLAN)
                    and not _is_swe
                )
                plan_only = no_tool_response and (
                    _is_swe
                    or insufficient_plan
                    or (not plan_text_is_expected and self._looks_like_plan(response.content))
                )

                if not plan_only or retries >= max_plan_retries:
                    # This is the final response - keep it
                    final_response = response
                    break

                # Plan-only response: need to retry
                if insufficient_plan:
                    reminder_text = (
                        "Your previous response was too brief. Provide a detailed implementation plan "
                        "with numbered sections, concrete steps, and explicit testing strategy."
                    )
                else:
                    reminder_text = (
                        "Your response described a plan but did not call any tools. "
                        "Please call the appropriate tool to execute your next step."
                    )
                reminder = HumanMessage(content=reminder_text)
                # Extend context for next iteration (temporary, not persisted)
                messages_with_context = messages_with_context + [response, reminder]
                retries += 1

                if self.console and self.verbose:
                    self.console.print(
                        f"[dim yellow]⚠️ Plan-only response detected, retrying ({retries}/{max_plan_retries})...[/dim yellow]"
                    )

            # Normalize refusal wording for dangerous command requests
            if (
                final_response
                and isinstance(final_response, AIMessage)
                and final_response.content
                and not getattr(final_response, "tool_calls", None)
            ):
                normalized_content = self._normalize_safety_refusal(
                    str(final_response.content),
                    messages,
                )
                if normalized_content != str(final_response.content):
                    final_response = AIMessage(content=normalized_content)

            # Only add the final response to state (not intermediate plan-only responses)
            messages_to_add = [final_response] if final_response else []

            if accumulated_tokens > 0:
                current_tokens = state.get("total_tokens_used", 0)
                new_total_tokens = current_tokens + accumulated_tokens
                new_cost = self._calculate_cost(new_total_tokens)
                token_delta = {
                    "total_tokens_used": new_total_tokens,
                    "estimated_cost": new_cost,
                }
                updates = self._merge_updates(updates, token_delta)

                # Update status indicator with accurate output generation speed
                # (bypasses state to avoid polluting LangGraph TypedDict schema)
                if self.status_indicator:
                    self.status_indicator.update_token_rate(
                        accumulated_output_tokens, accumulated_llm_elapsed
                    )

                # Display token usage in real-time
                if self.console:
                    context_window = getattr(self.settings, 'context_window', 128000)
                    context_pct = (new_total_tokens / context_window) * 100 if context_window > 0 else 0
                    self.console.print(
                        f"[dim]📊 Tokens: {accumulated_tokens:,} this turn | "
                        f"{new_total_tokens:,} total ({context_pct:.1f}% of context) | "
                        f"Cost: ${new_cost:.4f}[/dim]"
                    )

            # Send final response to chat (only if it has content and no tool calls)
            if final_response and final_response.content and not getattr(final_response, 'tool_calls', None):
                self._send_chat_response(final_response.content)

            # Log the response
            if final_response:
                self._verbose_llm_trace(
                    "Agent (Execution Specialist)",
                    messages_with_context,
                    final_response,
                    tokens=accumulated_tokens,
                )

            # LLM call succeeded — reset consecutive error counter
            updates["consecutive_llm_errors"] = 0

            # Return updated state with new message
            return self._merge_updates(updates, {"messages": messages_to_add})

        except Exception as e:
            # LLM invocation failed — use ErrorRecoveryStrategy for classification & backoff
            consecutive_errors = state.get("consecutive_llm_errors", 0) + 1
            error_context = ErrorRecoveryStrategy.create_error_context(e, consecutive_errors)

            self.logger.log_error(
                f"LLM invocation failed (attempt {consecutive_errors}): {str(e)}"
            )

            if self.console:
                self.console.print(
                    f"[red]❌ LLM Error (attempt {consecutive_errors}/{error_context.total_attempts}): "
                    f"{str(e)[:100]}[/red]"
                )
                if error_context.suggested_action:
                    self.console.print(f"[yellow]   {error_context.suggested_action}[/yellow]")

            # Record error in enhanced state
            import traceback
            err_delta = state_helpers.record_error(
                state,
                message=f"LLM invocation failed: {str(e)}",
                level=ErrorLevel.ERROR,
                source="llm",
                stack_trace=traceback.format_exc(),
                return_delta=True
            )
            updates = self._merge_updates(updates, err_delta)
            updates["consecutive_llm_errors"] = consecutive_errors

            # Check if we've exceeded max retries → force termination
            if not error_context.can_retry:
                if self.console:
                    self.console.print(
                        f"[red]❌ Max retries exceeded ({consecutive_errors} errors). Stopping.[/red]"
                    )
                updates["force_termination"] = True

            # Apply backoff before returning (next iteration will retry or stop)
            if error_context.backoff_seconds > 0 and error_context.can_retry:
                if self.console:
                    self.console.print(
                        f"[dim]⏳ Backing off {error_context.backoff_seconds:.1f}s...[/dim]"
                    )
                time.sleep(error_context.backoff_seconds)

            error_msg = AIMessage(
                content=f"I encountered an error calling the LLM: {str(e)}. "
                "Please try simplifying your request or try again."
            )
            return self._merge_updates(updates, {"messages": [error_msg]})

    def _fast_greeting_response(self, prompt: str) -> str | None:
        """LangGraph를 거치지 않는 greeting 전용 빠른 응답 (스트리밍).

        콘솔이 있으면 토큰을 실시간 스트리밍하고 None을 반환하여
        interactive 모드에서 이중 출력을 방지한다.
        콘솔이 없으면 invoke 후 결과 문자열을 반환한다.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a helpful coding assistant. Respond briefly to greetings."),
            HumanMessage(content=prompt),
        ]

        try:
            if self.console:
                # 스트리밍: 토큰이 콘솔에 바로 출력됨
                from sepilot.ui.streaming import StreamingHandler
                handler = StreamingHandler(
                    console=self.console, show_panel=False
                )
                handler.start()
                for chunk in self.quick_llm.stream(messages):
                    token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if token:
                        handler.update(token)
                handler.finish()
                result = handler.get_text()
                self.logger.log_result(result)
                return None  # 이미 콘솔에 출력됨 → 이중 출력 방지
            else:
                # 비-인터렉티브: 동기 호출 후 결과 반환
                response = self.quick_llm.invoke(messages)
                result = response.content if hasattr(response, 'content') else str(response)
                self.logger.log_result(result)
                return result
        except Exception as e:
            self.logger.log_error(f"Fast greeting failed: {e}")
            fallback = "Hello! How can I help you today?"
            if self.console:
                self.console.print(fallback)
                return None
            return fallback

    def execute(self, prompt: str, context_messages: list[BaseMessage] | None = None) -> str | None:
        """Execute the agent with a given prompt"""
        prompt = sanitize_text(prompt)

        # Keep session-wide auto-approve preference across requests in interactive sessions.

        # Start trace session if monitor is available
        # Reset iteration counter for new execution
        self.iteration_count = 0
        if hasattr(self, "recursion_detector") and self.recursion_detector:
            with contextlib.suppress(Exception):
                self.recursion_detector.reset()

        # Start cost tracking session
        if hasattr(self, 'cost_tracker') and self.cost_tracker:
            model_name = str(self.settings.model) if hasattr(self.settings, 'model') else "unknown"
            self.cost_tracker.start_session(
                session_id=self.thread_id,
                model=model_name
            )

        # Start status panel if available
        if self.status_panel:
            self.status_panel.start_time = datetime.now()
            self.status_panel.start()

        # Log prompt
        self.logger.log_prompt(prompt)

        # Greeting 빠른 경로: LangGraph 파이프라인 전체 우회
        from sepilot.agent.request_classifier import GREETING_KEYWORDS
        text_lower = prompt.strip().lower()
        is_greeting = len(text_lower) < 50 and any(
            text_lower == kw or text_lower.startswith(kw + " ") or text_lower.startswith(kw + "!")
            for kw in GREETING_KEYWORDS
        )

        if is_greeting:
            result = self._fast_greeting_response(prompt)
            # End trace session if monitor is available
            return result

        # Create initial message stack
        import uuid as _uuid
        if getattr(self, "status_indicator", None):
            self.status_indicator.update("[준비] 시스템 프롬프트를 구성하고 있어요...")
        base_system_prompt = self.prompt_template.get_system_prompt()
        load_project_instructions = bool(getattr(self.settings, "load_project_instructions", False))
        load_project_rules = bool(getattr(self.settings, "load_project_rules", False))

        # Inject user/project instructions (SEPILOT.md, CLAUDE.md, AGENT.md)
        try:
            if getattr(self, "status_indicator", None):
                self.status_indicator.update("[준비] 사용자 지침과 규칙을 불러오고 있어요...")
            user_instructions = load_all_instructions(
                working_dir=Path(os.getcwd()),
                include_project_sources=load_project_instructions,
                include_project_rules=False,
                include_user_rules=False,
            )
            if user_instructions:
                base_system_prompt += f"\n\n# User/Project Instructions\n\n{user_instructions}"
        except Exception:
            pass  # Non-critical: instructions loading is best-effort

        # Inject active rules based on files mentioned in prompt
        try:
            if getattr(self, "rules_loader", None):
                rules_text = self.rules_loader.get_rules_text(
                    include_project_rules=load_project_rules,
                    include_user_rules=True,
                )
                if rules_text:
                    base_system_prompt += f"\n\n# Active Rules\n\n{rules_text}"
        except Exception:
            pass  # Non-critical: rules loading is best-effort

        system_msg = SystemMessage(content=base_system_prompt)
        human_msg = self._make_user_prompt_message(
            prompt,
            message_id=f"exec_{_uuid.uuid4().hex[:12]}",
        )
        message_stack: list[BaseMessage] = [system_msg]
        if context_messages:
            message_stack.extend(context_messages)
        message_stack.append(human_msg)

        if getattr(self, "status_indicator", None):
            self.status_indicator.update("[준비] 실행 상태를 초기화하고 있어요...")

        import uuid
        initial_state = create_initial_state(
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            working_directory=os.getcwd(),
            max_iterations=self.settings.max_iterations
        )
        if getattr(self, "_session_mode_override", None):
            initial_state.update(self._session_mode_override)
        exec_timeout_raw = os.getenv("SEPILOT_EXECUTION_TIMEOUT_SECONDS", "900")
        try:
            exec_timeout_seconds = max(float(exec_timeout_raw), 60.0)
        except ValueError:
            exec_timeout_seconds = 900.0
        initial_state["_execution_deadline_monotonic"] = time.monotonic() + exec_timeout_seconds
        initial_state["messages"] = list(message_stack)
        initial_state["_initial_message_count"] = len(message_stack)
        initial_state["_execution_boundary_msg_id"] = human_msg.id

        # Start initial task
        initial_state = state_helpers.start_task(
            initial_state,
            description=prompt[:100]  # First 100 chars as task description
        )

        # Set initial strategy
        initial_state = state_helpers.set_strategy(
            initial_state,
            strategy="explore",
            confidence=0.8
        )

        try:
            # Run the graph with LangGraph checkpoint configuration
            # Each iteration cycle goes through multiple nodes.
            # Enhanced: ~15 nodes/iter, Simplified: ~7 nodes/iter
            is_simplified = getattr(self.settings, 'graph_mode', 'enhanced') == 'simplify'
            recursion_multiplier = 8 if is_simplified else 15
            config = {
                "configurable": {
                    "thread_id": self.thread_id
                },
                "recursion_limit": self.settings.max_iterations * recursion_multiplier
            }

            if getattr(self, "status_indicator", None):
                self.status_indicator.update("[시작] 그래프를 실행하고 LLM 응답을 기다리고 있어요...")

            final_state = self._invoke_with_interrupts(initial_state, config)

            # Extract final answer
            messages = final_state.get("messages", [])
            if not messages:
                return "No response generated."
            last_message = messages[-1]

            if isinstance(last_message, AIMessage):
                result = last_message.content
            else:
                result = str(last_message)

            # Complete task and log state
            from sepilot.agent.enhanced_state import TaskStatus
            final_state = state_helpers.complete_task(
                final_state,
                status=TaskStatus.COMPLETED
            )

            # Log state summary
            summary = state_to_summary(final_state)
            self.logger.log_trace("enhanced_state_summary", summary)
            self._display_session_metadata(final_state)

            # Print enhanced session summary (only in verbose mode)
            if self.console and self.verbose:
                from sepilot.agent.enhanced_state import print_session_summary
                print_session_summary(final_state, self.console)

            self.logger.log_result(result)

            # End trace session if monitor is available
            return result

        except GraphRecursionError as e:
            # Use error recovery strategy for better error handling
            error_context = ErrorRecoveryStrategy.create_error_context(e, attempt_number=1)

            # Generate detailed error message with actual state information
            recent_actions = self.recursion_detector.get_recent_actions(n=5)

            # Get actual iteration and tool call counts
            actual_iterations = self.iteration_count
            actual_tool_calls = len(recent_actions)

            error_msg = (
                f"❌ Task Incomplete: Graph Recursion Limit Reached\n\n"
                f"Error Category: {error_context.category.value}\n"
                f"This usually means the iteration limit is too low for this task.\n\n"
                f"Cycles completed: {actual_iterations}/{self.settings.max_iterations}\n"
                f"Tool calls: {actual_tool_calls}\n"
            )

            # Show recent actions if available
            if recent_actions:
                error_msg += f"\nLast {len(recent_actions)} tool calls:\n"
                for i, action in enumerate(recent_actions, 1):
                    # Parse action signature
                    parts = action.split(':', 1)
                    tool_name = parts[0] if parts else 'unknown'
                    is_repeated = recent_actions.count(action) > 1

                    marker = " ← Repeated" if is_repeated else ""
                    error_msg += f"  {i}. {tool_name}{marker}\n"

                # Add suggestions from error recovery strategy
                if error_context.suggested_action:
                    error_msg += f"\n💡 Recovery Suggestions:\n{error_context.suggested_action}\n"

                # Add additional context-specific suggestions
                unique_actions = len(set(recent_actions))
                if unique_actions <= 2:
                    error_msg += (
                        f"\nAdditional Context:\n"
                        f"  - The same tool is being called repeatedly ({unique_actions} unique actions)\n"
                        f"  - This suggests a loop or insufficient progress\n"
                    )
                else:
                    error_msg += (
                        f"\nAdditional Context:\n"
                        f"  - High variety of tool calls ({unique_actions} unique actions)\n"
                        f"  - Task may genuinely require more iterations\n"
                    )
            else:
                # Use error recovery suggestion if available
                if error_context.suggested_action:
                    error_msg += f"\n💡 {error_context.suggested_action}\n"

            # Log error
            self.logger.log_error(error_msg)

            # Print user-friendly error
            if self.console:
                self.console.print(f"\n[bold red]{error_msg}[/bold red]")

            # Record error in enhanced state
            from sepilot.agent.enhanced_state import ErrorLevel, TaskStatus
            # Try to get current state from graph
            try:
                config = {"configurable": {"thread_id": self.thread_id}}
                current_state = self.graph.get_state(config)
                if current_state and current_state.values:
                    state = current_state.values
                    state = state_helpers.record_error(
                        state,
                        message=error_msg,
                        level=ErrorLevel.ERROR,
                        source="system"
                    )
                    state = state_helpers.complete_task(state, status=TaskStatus.FAILED)
            except Exception:
                pass

            # End trace with error
            return error_msg

        except Exception as e:
            # Use error recovery strategy for comprehensive error handling
            error_context = ErrorRecoveryStrategy.create_error_context(e, attempt_number=1)

            error_msg = (
                f"Error during execution: {str(e)}\n"
                f"Error Category: {error_context.category.value}\n"
            )

            if error_context.suggested_action:
                error_msg += f"\n💡 {error_context.suggested_action}\n"

            if error_context.can_retry:
                error_msg += (
                    f"\nThis error may be retryable.\n"
                    f"Retry attempts available: {error_context.total_attempts}\n"
                    f"Suggested backoff: {error_context.backoff_seconds:.1f}s\n"
                )

            self.logger.log_error(error_msg)

            # Record error in enhanced state
            import traceback

            from sepilot.agent.enhanced_state import ErrorLevel, TaskStatus
            try:
                config = {"configurable": {"thread_id": self.thread_id}}
                current_state = self.graph.get_state(config)
                if current_state and current_state.values:
                    state = current_state.values
                    state = state_helpers.record_error(
                        state,
                        message=error_msg,
                        level=ErrorLevel.CRITICAL,
                        source="system",
                        stack_trace=traceback.format_exc()
                    )
                    state = state_helpers.complete_task(state, status=TaskStatus.FAILED)
            except Exception:
                pass

            # End trace with error
            return error_msg

        finally:
            # Reset session mode override after execution completes
            # (Ensures next execute() call is not affected by previous mode selection)
            self._session_mode_override = None

            # Stop status panel if running
            if self.status_panel:
                self.status_panel.stop()

            # Log final memory statistics
            memory_stats = self.memory_monitor.get_statistics()
            if memory_stats['count'] > 0:
                self.logger.log_trace("memory_statistics", memory_stats)

                if self.console and memory_stats['max_mb'] > 100:  # Only show if significant
                    self.console.print(
                        f"\n[dim]💾 Memory Stats: Current={memory_stats['current_mb']:.1f}MB, "
                        f"Peak={memory_stats['max_mb']:.1f}MB, "
                        f"Avg={memory_stats['avg_mb']:.1f}MB, "
                        f"Trend={memory_stats['trend']}[/dim]"
                    )

    def get_thread_id(self) -> str | None:
        """Get the current thread ID."""
        return self.thread_id

    @property
    def thread_config(self) -> dict[str, Any] | None:
        """Return the active LangGraph thread config for UI/status helpers."""
        config = getattr(self, "config", None)
        if config:
            return config
        if getattr(self, "thread_id", None):
            return {"configurable": {"thread_id": self.thread_id}}
        return None

    def get_session_summary(self) -> str:
        """Get session summary - delegates to ThreadManager."""
        self._thread_manager.thread_id = self.thread_id
        return self._thread_manager.get_session_summary()

    def switch_thread(self, new_thread_id: str) -> bool:
        """Switch to a different thread - delegates to ThreadManager."""
        self._thread_manager.thread_id = self.thread_id
        result = self._thread_manager.switch_thread(new_thread_id)
        if result:
            self.thread_id = self._thread_manager.thread_id
            self.config = {"configurable": {"thread_id": new_thread_id}}
        return result

    def list_available_threads(self) -> list[dict[str, Any]]:
        """List all available threads - delegates to ThreadManager."""
        return self._thread_manager.list_available_threads()

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics - delegates to ThreadManager."""
        self._thread_manager.thread_id = self.thread_id
        return self._thread_manager.get_memory_stats()

    def create_new_thread(self) -> str:
        """Create a new conversation thread - delegates to ThreadManager."""
        self._thread_manager.thread_id = self.thread_id
        new_thread_id = self._thread_manager.create_new_thread()
        self.thread_id = new_thread_id
        self.config = {"configurable": {"thread_id": new_thread_id}}
        return new_thread_id

    def _reset_local_session_state(self) -> None:
        """Reset in-memory helpers that should not leak across threads or rewinds."""
        self.auto_approve_session = False
        if hasattr(self, "_approval_handler") and self._approval_handler:
            self._approval_handler.auto_approve_session = False

        self._pending_mode_update = None
        self._session_mode_override = None

        self._progress_plan_steps = []
        self._progress_current_step = 0
        self._progress_planning_notes = []
        self._progress_current_task = None

        if hasattr(self, "recursion_detector") and self.recursion_detector:
            with contextlib.suppress(Exception):
                self.recursion_detector.reset()

        if hasattr(self, "tool_cache") and self.tool_cache:
            with contextlib.suppress(Exception):
                self.tool_cache.invalidate()
                self.tool_cache.reset_stats()

        if self.step_logger:
            self.step_logger._last_plan_signature = None
            self.step_logger._last_plan_progress_signature = None
            self.step_logger._checklist_lines_rendered = 0

    def _build_context_reset_updates(self) -> dict[str, Any]:
        """Build graph-state updates that clear derived execution context."""
        import uuid

        defaults = create_initial_state(
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            working_directory=os.getcwd(),
            max_iterations=self.settings.max_iterations,
        )
        reset_keys = {
            "current_task",
            "task_history",
            "open_files",
            "file_changes",
            "staged_changes",
            "active_processes",
            "environment_vars",
            "error_history",
            "recent_errors",
            "warning_count",
            "tool_call_history",
            "tool_results_cache",
            "tools_used_count",
            "pending_user_input",
            "user_feedback",
            "pending_approvals",
            "iteration_count",
            "total_tokens_used",
            "estimated_cost",
            "plan_created",
            "plan_steps",
            "current_plan_step",
            "planning_notes",
            "verification_notes",
            "needs_additional_iteration",
            "required_files",
            "plan_execution_pending",
            "missing_tasks",
            "last_approval_status",
            "force_termination",
            "consecutive_llm_errors",
            "scratchpad_entries",
            "conversation_summary",
            "important_context",
            "triage_decision",
            "triage_reason",
            "current_strategy",
            "confidence_level",
            "needs_clarification",
            "query_complexity",
            "expected_tool_count",
            "_task_complete_pending_response",
            "_patch_review_done",
            "exploration_performed",
            "exploration_skipped",
            "exploration_context",
            "exploration_results",
            "exploration_hints",
            "explicit_files",
            "project_type",
            "reflection_count",
            "reflection_decision",
            "reflection_notes",
            "failure_patterns",
            "strategy_adjustment_history",
            "last_reflection_insight",
            "task_complexity",
            "repetition_detected",
            "repetition_info",
            "stagnation_detected",
            "consecutive_denials",
            "task_type_detected",
            "tests_requested",
            "lint_requested",
            "debate_decision",
            "_last_compaction_iter",
            "backtrack_decision",
            "backtrack_performed",
            "backtrack_reason",
            "files_restored",
            "active_patterns",
            "modified_files",
            "file_changes_count",
        }
        updates = {key: defaults[key] for key in reset_keys if key in defaults}
        updates.update(
            {
                "hierarchical_plan": {},
                "orchestration_plan": {},
                "retrieved_memories": [],
                "memory_context": "",
            }
        )
        return updates

    def reset_session_state(self, *, clear_checkpoint_state: bool = False) -> None:
        """Reset ephemeral state that should not survive thread boundaries."""
        if clear_checkpoint_state:
            try:
                config = {"configurable": {"thread_id": self.thread_id}}
                self.graph.update_state(config, self._build_context_reset_updates())
            except Exception:
                pass

        self._reset_local_session_state()

    def reset_plan_state(self) -> None:
        """Reset plan-related state fields in LangGraph checkpoint.

        Called by /clear to ensure stale plan steps don't persist
        after conversation history is cleared.
        """
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            self.graph.update_state(config, {
                "plan_created": False,
                "plan_steps": [],
                "current_plan_step": 0,
                "planning_notes": [],
                "plan_execution_pending": False,
                "hierarchical_plan": {},
                "triage_decision": None,
                "triage_reason": None,
                "exploration_context": "",
                "exploration_results": {},
            })
        except Exception:
            pass  # Best-effort: state may not exist yet

        # Also reset UI-side progress tracking
        self._progress_plan_steps = []
        self._progress_current_step = 0
        self._progress_planning_notes = []
        self._progress_current_task = None

        # Reset step_logger cached signatures so new plans render correctly
        if self.step_logger:
            self.step_logger._last_plan_signature = None
            self.step_logger._last_plan_progress_signature = None
            self.step_logger._checklist_lines_rendered = 0

    def rewind_messages(self, count: int = 1) -> dict[str, Any]:
        """Rewind conversation messages - delegates to ThreadManager."""
        self._thread_manager.thread_id = self.thread_id
        return self._thread_manager.rewind_messages(count)

    def get_conversation_messages(self) -> list[Any]:
        """Get all messages from current thread - delegates to ThreadManager."""
        self._thread_manager.thread_id = self.thread_id
        return self._thread_manager.get_conversation_messages()

    def append_conversation_message(self, message: BaseMessage) -> bool:
        """Append an out-of-band message to the current thread."""
        return self.append_conversation_messages([message])

    def append_conversation_messages(self, messages: list[BaseMessage]) -> bool:
        """Append out-of-band messages to the current thread in order."""
        if not self.enable_memory or not self.thread_config:
            return False

        try:
            self.graph.update_state(self.thread_config, {"messages": list(messages)})
            return True
        except Exception:
            return False

    def replace_conversation_messages(self, messages: list[BaseMessage]) -> bool:
        """Replace persisted thread messages exactly."""
        if not self.enable_memory or not self.thread_config:
            return False

        try:
            remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES, content="")
            self.graph.update_state(self.thread_config, {"messages": [remove_all, *messages]})
            return True
        except Exception:
            return False

    def clear_conversation_messages(self) -> bool:
        """Clear all persisted thread messages for the active conversation."""
        return self.replace_conversation_messages([])

    def compact_conversation_context(
        self,
        keep_recent: int = 10,
        focus_instruction: str | None = None,
        target_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Compact the persisted conversation thread instead of only local UI state."""
        if not self.enable_memory or not self.thread_config:
            return {"success": False, "error": "Memory is disabled"}

        state = self.graph.get_state(self.thread_config)
        messages = list(state.values.get("messages", [])) if state and state.values else []
        if not messages:
            return {"success": False, "error": "No conversation context to compact"}

        from sepilot.agent.context_manager import ContextManager

        max_tokens = getattr(self.settings, "context_window", None) or int(os.getenv("MAX_TOKENS", "96000"))
        context_manager = ContextManager(max_context_tokens=max_tokens)

        try:
            if self.llm:
                if target_tokens is not None:
                    compacted = context_manager.summarize_to_token_limit(
                        messages,
                        self.llm,
                        target_tokens=target_tokens,
                        focus_instruction=focus_instruction,
                    )
                else:
                    compacted = context_manager.summarize_messages(
                        messages,
                        self.llm,
                        keep_recent=keep_recent,
                        focus_instruction=focus_instruction,
                    )
                method = "summarized"
            else:
                raise RuntimeError("LLM unavailable")
        except Exception:
            if target_tokens is not None:
                compacted = context_manager.compact_to_token_limit(
                    messages,
                    target_tokens=target_tokens,
                    min_keep=max(keep_recent, 4),
                )
            else:
                compacted = context_manager.compact_messages(messages, keep_recent=keep_recent)
            method = "compacted"

        if not self.replace_conversation_messages(compacted):
            return {"success": False, "error": "Failed to update thread messages"}

        return {
            "success": True,
            "method": method,
            "messages_before": len(messages),
            "messages_after": len(compacted),
        }

    def cleanup(self) -> None:
        """Clean up resources"""
        # Stop file watcher thread if active
        if self.file_watcher:
            with contextlib.suppress(Exception):
                self.file_watcher.stop()

        # Clean up SqliteSaver context manager if used
        if hasattr(self, '_checkpointer_cm') and self._checkpointer_cm is not None:
            with contextlib.suppress(Exception):
                self._checkpointer_cm.__exit__(None, None, None)
        elif hasattr(self.checkpointer, "conn") and self.checkpointer.conn:
            with contextlib.suppress(Exception):
                self.checkpointer.conn.close()

        # Stop A2A connector if active
        if self.a2a_connector:
            self.a2a_connector.stop()

    # =========================================================================
    # A2A (Agent-to-Agent) Protocol Methods
    # =========================================================================

    def initialize_a2a(
        self,
        router: A2ARouter
    ) -> None:
        """Initialize A2A protocol support.

        Args:
            router: A2A router instance
        """
        self.a2a_router = router

        # Create connector for base agent
        self.a2a_connector = A2AConnector(
            agent_id="base_agent",
            display_name="Base Agent",
            router=router,
            agent_type="base"
        )

        # Register base agent capabilities
        self.a2a_connector.register_capability("general")
        self.a2a_connector.register_capability("code_analysis")
        self.a2a_connector.register_capability("file_operations")
        self.a2a_connector.register_capability("planning")

        # Register with router
        router.register_agent(self.a2a_connector)

        # Initialize handoff manager
        self.handoff_manager = SessionHandoffManager(console=self.console)

        if self.console:
            self.console.print("[dim]A2A Protocol initialized for base agent[/dim]")

        self.logger.log_trace("a2a_initialized", {
            "agent_id": "base_agent",
            "capabilities": ["general", "code_analysis", "file_operations", "planning"]
        })

    def is_a2a_enabled(self) -> bool:
        """Check if A2A protocol is enabled and initialized.

        Returns:
            True if A2A is ready for use
        """
        return self.a2a_router is not None and self.a2a_connector is not None
