"""Centralized constants for SEPilot agent behavior.

All magic numbers used across agent modules are defined here to serve as a
single source of truth. Import from this module instead of hardcoding values.
"""

# ============================================================================
# State Management (enhanced_state.py reducers)
# ============================================================================

MAX_FILE_CHANGES_HISTORY = 100
MAX_ERROR_HISTORY = 100
MAX_TOOL_CALL_HISTORY = 200
FILE_CHANGE_MERGE_WINDOW_SECONDS = 60

# ============================================================================
# Repetition Control (base_agent.py → RecursionDetector)
# ============================================================================

RECURSION_DETECTOR_WINDOW_SIZE = 5
RECURSION_DETECTOR_THRESHOLD = 3

# ============================================================================
# Context Management (base_agent.py → ContextManager)
# ============================================================================

CONTEXT_WARNING_THRESHOLD = 0.70
CONTEXT_COMPACT_THRESHOLD = 0.78
CONTEXT_PRUNE_SUMMARY_MAX_CHARS = 200

# ============================================================================
# Codebase Exploration (pattern_nodes.py)
# ============================================================================

EXPLORATION_MAX_FILES = 50
EXPLORATION_MAX_SEARCH_TIME_MS = 3000
EXPLORATION_MAX_HINTS = 5

# ============================================================================
# Backtracking (backtracking.py)
# ============================================================================

BACKTRACK_MAX_SCAN_LENGTH = 8000
MAX_BACKTRACK_CHECKPOINTS = 10
BACKTRACK_AUTO_INTERVAL = 5

# ============================================================================
# Reflection (reflection_node.py)
# ============================================================================

MAX_REFLECTION_ITERATIONS = 2

# ============================================================================
# State List Limits (enhanced_state.py list reducers)
# ============================================================================

MAX_TASK_HISTORY = 50
MAX_PLANNING_NOTES = 50
MAX_VERIFICATION_NOTES = 50
MAX_REFLECTION_NOTES = 50
MAX_STRATEGY_HISTORY = 50

# ============================================================================
# Think / Scratchpad (think_tools.py)
# ============================================================================

MAX_SCRATCHPAD_ENTRIES = 20
SCRATCHPAD_MAX_SYSTEM_PROMPT_CHARS = 2000
