"""UI components for SE Pilot

This module provides enhanced UI/UX components following SOLID principles:
- Progress bars and status displays
- Interactive command input
- Live dashboards
- Keyboard shortcuts
- Streaming output
- TUI Dashboard (Textual-based)
- Modular components: key bindings, reference expansion, context display,
  memory management, output overlays
"""

from .completer import FileReferenceCompleter
from .context_display import ContextDisplayManager, ContextUsageInfo, estimate_cost
from .dashboard import DashboardManager, create_dashboard, is_dashboard_available
from .interactive import InteractiveMode
from .key_bindings import KeyBindingsManager, create_key_bindings
from .memory_manager import MemoryEntry, MemoryManager, get_memory_manager
from .output_overlays import (
    ExecutionResult,
    OutputOverlayManager,
    TeeOutput,
    get_overlay_manager,
)
from .progress_display import ProgressDisplay, StatusPanel
from .status_indicator import AgentStatusIndicator
from .reference_expander import (
    ReferenceExpander,
    expand_file_references,
    get_expander,
)
from .streaming import (
    CallbackStreamingHandler,
    StreamingHandler,
    stream_llm_response,
    stream_with_callbacks,
)

__all__ = [
    # Core interactive mode
    "InteractiveMode",
    # Progress and streaming
    "ProgressDisplay",
    "StatusPanel",
    "AgentStatusIndicator",
    "StreamingHandler",
    "CallbackStreamingHandler",
    "stream_llm_response",
    "stream_with_callbacks",
    # Dashboard
    "DashboardManager",
    "create_dashboard",
    "is_dashboard_available",
    # Key bindings (SOLID: Single Responsibility)
    "KeyBindingsManager",
    "create_key_bindings",
    # Reference expansion (SOLID: Single Responsibility)
    "ReferenceExpander",
    "expand_file_references",
    "get_expander",
    # Context display (SOLID: Single Responsibility)
    "ContextDisplayManager",
    "ContextUsageInfo",
    "estimate_cost",
    # Memory management (SOLID: Single Responsibility)
    "MemoryManager",
    "MemoryEntry",
    "get_memory_manager",
    # Output overlays (SOLID: Single Responsibility)
    "OutputOverlayManager",
    "ExecutionResult",
    "TeeOutput",
    "get_overlay_manager",
    # File completion
    "FileReferenceCompleter",
]
