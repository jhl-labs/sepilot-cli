"""Memory Bank - Experience Learning System for Agents (Claude Code Style)

This module implements persistent memory for agent learning:
- Stores successful/failed approaches per task type
- Learns project-specific patterns
- Retrieves relevant past experiences for similar tasks

Claude Code Style Enhancements:
- ~/.sepilot/ folder structure (like ~/.claude/)
- Project-level .sepilot/ configuration (like .claude/)
- Session history persistence
- Markdown-based user instructions (SEPILOT.md)

Inspired by:
- Claude Code's file-based configuration system
- MemGPT: Memory Management for LLMs
- Generative Agents: Interactive Simulacra of Human Behavior
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sepilot.agent.enhanced_state import AgentStrategy, EnhancedAgentState


# ═══════════════════════════════════════════════════════════════════
# Claude Code Style: File-based Configuration
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEPILOT_DIR = Path.home() / ".sepilot"
GLOBAL_INSTRUCTIONS_FILE = "SEPILOT.md"
PROJECT_SEPILOT_DIR = ".sepilot"
PROJECT_CONTEXT_FILE = "context.md"

# Supported instruction file names (checked in order)
INSTRUCTION_FILENAMES = ["SEPILOT.md", "AGENT.md", "CLAUDE.md"]


def ensure_global_sepilot_dir() -> Path:
    """Ensure ~/.sepilot directory exists with proper structure."""
    global_dir = GLOBAL_SEPILOT_DIR
    global_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (global_dir / "memories").mkdir(exist_ok=True)
    (global_dir / "sessions").mkdir(exist_ok=True)
    (global_dir / "skills").mkdir(exist_ok=True)
    (global_dir / "rules").mkdir(exist_ok=True)
    (global_dir / "commands").mkdir(exist_ok=True)

    # Create default SEPILOT.md if not exists
    instructions_file = global_dir / GLOBAL_INSTRUCTIONS_FILE
    if not instructions_file.exists():
        instructions_file.write_text(
            "# SE Pilot Global Instructions\n\n"
            "Add your personal preferences and instructions here.\n"
            "These will be applied to all projects.\n\n"
            "## Supported file names:\n"
            "- SEPILOT.md (primary)\n"
            "- AGENT.md (alternative)\n"
            "- CLAUDE.md (Claude Code compatible)\n\n"
            "## Example:\n"
            "- 대답은 항상 한국어로 해\n"
            "- Always commit changes with descriptive messages\n",
            encoding="utf-8"
        )

    return global_dir


def get_global_instructions() -> str:
    """Get global user instructions from ~/.sepilot/.

    Checks for SEPILOT.md, AGENT.md, CLAUDE.md in order.
    Also checks ~/.claude/ for Claude Code compatibility.
    """
    ensure_global_sepilot_dir()

    # Check ~/.sepilot/ first
    for filename in INSTRUCTION_FILENAMES:
        instructions_file = GLOBAL_SEPILOT_DIR / filename
        if instructions_file.exists():
            return instructions_file.read_text(encoding="utf-8")

    # Check ~/.claude/ for Claude Code compatibility
    claude_dir = Path.home() / ".claude"
    if claude_dir.exists():
        for filename in INSTRUCTION_FILENAMES:
            instructions_file = claude_dir / filename
            if instructions_file.exists():
                return instructions_file.read_text(encoding="utf-8")

    return ""


def get_project_context(project_path: Path | None = None) -> str:
    """Get project-specific context from .sepilot/context.md.

    Also checks .claude/ and .agent/ directories,
    and looks for AGENT.md/CLAUDE.md at project root.
    """
    if project_path is None:
        project_path = Path.cwd()

    # Check config directories
    for config_dir in [PROJECT_SEPILOT_DIR, ".claude", ".agent"]:
        context_file = project_path / config_dir / PROJECT_CONTEXT_FILE
        if context_file.exists():
            return context_file.read_text(encoding="utf-8")

    # Check project root for instruction files
    for filename in INSTRUCTION_FILENAMES:
        root_file = project_path / filename
        if root_file.exists():
            return root_file.read_text(encoding="utf-8")

    return ""


def get_all_instructions(
    project_path: Path | None = None,
    active_files: list[str] | None = None
) -> str:
    """Get all instructions using the full InstructionsLoader.

    This is the recommended function for loading instructions.
    Falls back to simple loading if InstructionsLoader is not available.
    """
    try:
        from sepilot.agent.instructions_loader import load_all_instructions
        return load_all_instructions(
            working_dir=project_path,
            active_files=active_files,
        )
    except ImportError:
        # Fallback to simple loading
        parts = []
        global_inst = get_global_instructions()
        if global_inst:
            parts.append(global_inst)
        project_ctx = get_project_context(project_path)
        if project_ctx:
            parts.append(project_ctx)
        return "\n\n---\n\n".join(parts)


def save_project_context(context: str, project_path: Path | None = None) -> None:
    """Save project-specific context to .sepilot/context.md."""
    if project_path is None:
        project_path = Path.cwd()

    project_sepilot = project_path / PROJECT_SEPILOT_DIR
    project_sepilot.mkdir(exist_ok=True)

    context_file = project_sepilot / PROJECT_CONTEXT_FILE
    context_file.write_text(context, encoding="utf-8")


class MemoryType(str, Enum):
    """Types of memories stored."""
    TASK_SUCCESS = "task_success"      # Successful task completion
    TASK_FAILURE = "task_failure"      # Failed task with lessons
    TOOL_PATTERN = "tool_pattern"      # Effective tool sequences
    CODE_PATTERN = "code_pattern"      # Code patterns in this project
    ERROR_SOLUTION = "error_solution"  # Error → Solution mappings
    PROJECT_CONTEXT = "project_context"  # Project-specific knowledge


class MemoryImportance(str, Enum):
    """Importance levels for memory prioritization."""
    CRITICAL = "critical"    # Must remember (e.g., critical bug fix)
    HIGH = "high"           # Important lesson
    MEDIUM = "medium"       # Useful pattern
    LOW = "low"             # Minor observation


@dataclass
class Memory:
    """A single memory entry."""
    memory_id: str
    memory_type: MemoryType
    importance: MemoryImportance
    content: str
    context: dict[str, Any]
    task_description: str
    outcome: str  # "success" or "failure"
    strategy_used: AgentStrategy | None
    tools_used: list[str]
    files_involved: list[str]
    lessons_learned: list[str]
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime | None = None
    relevance_score: float = 0.0  # Computed during retrieval
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata for pattern learning

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "importance": self.importance.value,
            "content": self.content,
            "context": self.context,
            "task_description": self.task_description,
            "outcome": self.outcome,
            "strategy_used": self.strategy_used.value if self.strategy_used else None,
            "tools_used": self.tools_used,
            "files_involved": self.files_involved,
            "lessons_learned": self.lessons_learned,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            memory_type=MemoryType(data["memory_type"]),
            importance=MemoryImportance(data["importance"]),
            content=data["content"],
            context=data.get("context", {}),
            task_description=data["task_description"],
            outcome=data["outcome"],
            strategy_used=AgentStrategy(data["strategy_used"]) if data.get("strategy_used") else None,
            tools_used=data.get("tools_used", []),
            files_involved=data.get("files_involved", []),
            lessons_learned=data.get("lessons_learned", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            metadata=data.get("metadata", {})
        )


@dataclass
class MemorySearchResult:
    """Result of memory search."""
    memories: list[Memory]
    total_found: int
    search_query: str
    search_time_ms: float


class MemoryBank:
    """Persistent memory system for agent learning (Claude Code Style).

    Features:
    - File-based persistence in ~/.sepilot/ (like ~/.claude/)
    - Session history persistence
    - Semantic similarity search
    - Importance-based retention
    - Automatic memory consolidation
    """

    MAX_MEMORIES = 1000  # Maximum memories to retain
    CONSOLIDATION_THRESHOLD = 800  # Trigger consolidation at this count
    MAX_SESSIONS = 50  # Maximum session files to retain

    def __init__(
        self,
        storage_path: str | Path | None = None,
        project_id: str | None = None
    ):
        """Initialize memory bank with Claude Code style folder structure.

        Creates:
        - ~/.sepilot/memories/ - Memory persistence
        - ~/.sepilot/sessions/ - Session history
        - ~/.sepilot/SEPILOT.md - Global user instructions
        """
        # Ensure global sepilot directory structure exists
        ensure_global_sepilot_dir()

        if storage_path is None:
            storage_path = GLOBAL_SEPILOT_DIR / "memories"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.sessions_path = GLOBAL_SEPILOT_DIR / "sessions"
        self.sessions_path.mkdir(parents=True, exist_ok=True)

        self.project_id = project_id or self._compute_project_id()
        self.memories: dict[str, Memory] = {}
        self._loaded = False  # Lazy loading flag
        self._dirty = False  # Track unsaved changes

    def _ensure_loaded(self) -> None:
        """Ensure memories are loaded from disk (lazy loading)."""
        if not self._loaded:
            self._load_memories()
            self._loaded = True

    def _compute_project_id(self) -> str:
        """Compute project ID from project root directory.

        Uses git root if available, otherwise falls back to realpath of cwd
        to ensure consistent IDs regardless of symlinks or subdirectory access.
        """
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                project_root = result.stdout.strip()
            else:
                project_root = os.path.realpath(os.getcwd())
        except Exception:
            project_root = os.path.realpath(os.getcwd())
        return hashlib.md5(project_root.encode(), usedforsecurity=False).hexdigest()[:12]

    def _get_memory_file(self) -> Path:
        """Get path to memory file for current project."""
        return self.storage_path / f"{self.project_id}_memories.json"

    def _load_memories(self) -> None:
        """Load memories from disk."""
        memory_file = self._get_memory_file()
        if memory_file.exists():
            try:
                with open(memory_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for mem_data in data.get("memories", []):
                        memory = Memory.from_dict(mem_data)
                        self.memories[memory.memory_id] = memory
            except (json.JSONDecodeError, KeyError, ValueError):
                # Corrupted file, start fresh
                self.memories = {}

    def _save_memories(self, force: bool = False) -> None:
        """Persist memories to disk.

        Args:
            force: Write regardless of dirty flag.
        """
        if not force and not self._dirty:
            return
        memory_file = self._get_memory_file()
        data = {
            "project_id": self.project_id,
            "updated_at": datetime.now().isoformat(),
            "memory_count": len(self.memories),
            "memories": [mem.to_dict() for mem in self.memories.values()]
        }
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._dirty = False

    def _generate_memory_id(self, content: str) -> str:
        """Generate unique memory ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{content}{timestamp}".encode(), usedforsecurity=False).hexdigest()[:16]

    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        task_description: str,
        outcome: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        context: dict[str, Any] | None = None,
        strategy_used: AgentStrategy | None = None,
        tools_used: list[str] | None = None,
        files_involved: list[str] | None = None,
        lessons_learned: list[str] | None = None
    ) -> Memory:
        """Store a new memory.

        Args:
            content: Main content/description of the memory
            memory_type: Type of memory
            task_description: Original task description
            outcome: "success" or "failure"
            importance: Importance level
            context: Additional context dict
            strategy_used: Strategy that was used
            tools_used: List of tools used
            files_involved: Files that were modified/read
            lessons_learned: Key takeaways

        Returns:
            Created Memory object
        """
        memory_id = self._generate_memory_id(content)

        memory = Memory(
            memory_id=memory_id,
            memory_type=memory_type,
            importance=importance,
            content=content,
            context=context or {},
            task_description=task_description,
            outcome=outcome,
            strategy_used=strategy_used,
            tools_used=tools_used or [],
            files_involved=files_involved or [],
            lessons_learned=lessons_learned or []
        )

        self._ensure_loaded()
        self.memories[memory_id] = memory
        self._dirty = True

        # Check if consolidation needed
        if len(self.memories) >= self.CONSOLIDATION_THRESHOLD:
            self._consolidate_memories()

        self._save_memories(force=True)
        return memory

    def store_from_state(
        self,
        state: EnhancedAgentState,
        outcome: str,
        lessons: list[str] | None = None
    ) -> Memory | None:
        """Store memory from current agent state.

        Args:
            state: Current enhanced agent state
            outcome: "success" or "failure"
            lessons: Lessons learned (optional)

        Returns:
            Created memory or None if nothing to store
        """
        # Extract relevant information from state
        messages = state.get("messages", [])
        if not messages:
            return None

        # Get original task from first human message
        task_description = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task_description = getattr(msg, "content", "")[:500]
                break

        if not task_description:
            return None

        # Get tools used
        tool_history = state.get("tool_call_history", [])
        tools_used = list({tc.tool_name for tc in tool_history})

        # Get files involved
        file_changes = state.get("file_changes", [])
        files_involved = list({fc.file_path for fc in file_changes})

        # Get strategy
        strategy = state.get("current_strategy")

        # Determine importance based on outcome and complexity
        importance = MemoryImportance.MEDIUM
        if outcome == "success" and len(tools_used) > 5:
            importance = MemoryImportance.HIGH
        elif outcome == "failure":
            importance = MemoryImportance.HIGH  # Failures are valuable lessons

        # Build content summary
        content_parts = [
            f"Task: {task_description[:200]}",
            f"Outcome: {outcome}",
            f"Strategy: {strategy.value if strategy else 'unknown'}",
            f"Tools: {', '.join(tools_used[:5])}",
            f"Files: {', '.join(files_involved[:5])}"
        ]
        content = "\n".join(content_parts)

        # Determine memory type
        memory_type = MemoryType.TASK_SUCCESS if outcome == "success" else MemoryType.TASK_FAILURE

        return self.store_memory(
            content=content,
            memory_type=memory_type,
            task_description=task_description,
            outcome=outcome,
            importance=importance,
            context={
                "iteration_count": state.get("iteration_count", 0),
                "error_count": len(state.get("error_history", [])),
                "reflection_count": state.get("reflection_count", 0)
            },
            strategy_used=strategy,
            tools_used=tools_used,
            files_involved=files_involved,
            lessons_learned=lessons or []
        )

    def search(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        outcome_filter: str | None = None,
        limit: int = 5
    ) -> MemorySearchResult:
        """Search memories by relevance.

        Args:
            query: Search query (task description or keywords)
            memory_types: Filter by memory types
            outcome_filter: Filter by "success" or "failure"
            limit: Maximum results to return

        Returns:
            MemorySearchResult with matching memories
        """
        import time
        start_time = time.time()

        # Filter memories
        self._ensure_loaded()
        candidates = list(self.memories.values())

        if memory_types:
            candidates = [m for m in candidates if m.memory_type in memory_types]

        if outcome_filter:
            candidates = [m for m in candidates if m.outcome == outcome_filter]

        # Score by relevance (simple keyword matching)
        query_lower = query.lower()
        query_words = set(query_lower.split())
        _scores: dict[int, float] = {}  # id(memory) → relevance score

        for memory in candidates:
            score = 0.0

            # Task description similarity
            task_words = set(memory.task_description.lower().split())
            overlap = len(query_words & task_words)
            score += overlap * 2.0

            # Content similarity
            content_words = set(memory.content.lower().split())
            content_overlap = len(query_words & content_words)
            score += content_overlap * 1.0

            # File path matching
            for file_path in memory.files_involved:
                if any(word in file_path.lower() for word in query_words):
                    score += 1.5

            # Boost by importance
            importance_boost = {
                MemoryImportance.CRITICAL: 2.0,
                MemoryImportance.HIGH: 1.5,
                MemoryImportance.MEDIUM: 1.0,
                MemoryImportance.LOW: 0.5
            }
            score *= importance_boost.get(memory.importance, 1.0)

            # Recency boost (more recent = higher score)
            days_old = (datetime.now() - memory.timestamp).days
            recency_factor = max(0.5, 1.0 - (days_old / 30) * 0.1)
            score *= recency_factor

            # Store score in local map (avoid mutating original Memory object)
            _scores[id(memory)] = score

        # Sort by relevance and return top results
        candidates.sort(key=lambda m: _scores.get(id(m), 0.0), reverse=True)
        results = candidates[:limit]

        # Set relevance_score on returned results only, and update access metadata
        now = datetime.now()
        for memory in results:
            memory.relevance_score = _scores.get(id(memory), 0.0)
            memory.access_count += 1
            memory.last_accessed = now

        # Mark dirty; defer disk write until next store_memory() or explicit save
        if results:
            self._dirty = True

        search_time = (time.time() - start_time) * 1000

        return MemorySearchResult(
            memories=results,
            total_found=len(candidates),
            search_query=query,
            search_time_ms=search_time
        )

    def get_relevant_memories(
        self,
        state: EnhancedAgentState,
        limit: int = 3
    ) -> list[Memory]:
        """Get memories relevant to current task.

        Args:
            state: Current agent state
            limit: Maximum memories to return

        Returns:
            List of relevant memories
        """
        # Extract task from state
        messages = state.get("messages", [])
        task = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task = getattr(msg, "content", "")
                break

        if not task:
            return []

        result = self.search(query=task, limit=limit)
        return result.memories

    def get_error_solutions(self, error_message: str, limit: int = 3) -> list[Memory]:
        """Get memories of similar errors and their solutions.

        Args:
            error_message: Current error message
            limit: Maximum results

        Returns:
            Memories with similar errors
        """
        result = self.search(
            query=error_message,
            memory_types=[MemoryType.ERROR_SOLUTION, MemoryType.TASK_FAILURE],
            limit=limit
        )
        return result.memories

    def get_tool_patterns(self, task_type: str, limit: int = 3) -> list[Memory]:
        """Get effective tool patterns for task type.

        Args:
            task_type: Type of task (e.g., "bug fix", "refactor")
            limit: Maximum results

        Returns:
            Memories with tool patterns
        """
        result = self.search(
            query=task_type,
            memory_types=[MemoryType.TOOL_PATTERN, MemoryType.TASK_SUCCESS],
            outcome_filter="success",
            limit=limit
        )
        return result.memories

    def _consolidate_memories(self) -> None:
        """Consolidate memories when approaching limit.

        Removes low-importance, old, rarely-accessed memories.
        """
        if len(self.memories) < self.CONSOLIDATION_THRESHOLD:
            return

        # Score memories for retention
        scored_memories = []
        for memory in self.memories.values():
            score = 0.0

            # Importance weight
            importance_weights = {
                MemoryImportance.CRITICAL: 100,
                MemoryImportance.HIGH: 50,
                MemoryImportance.MEDIUM: 20,
                MemoryImportance.LOW: 5
            }
            score += importance_weights.get(memory.importance, 10)

            # Access frequency
            score += memory.access_count * 2

            # Recency
            days_old = (datetime.now() - memory.timestamp).days
            score += max(0, 30 - days_old)

            # Success memories slightly preferred
            if memory.outcome == "success":
                score += 10

            scored_memories.append((memory, score))

        # Sort by score and keep top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        keep_count = int(self.MAX_MEMORIES * 0.7)  # Keep 70%

        self.memories = {
            mem.memory_id: mem
            for mem, _ in scored_memories[:keep_count]
        }

    def format_memories_for_prompt(self, memories: list[Memory]) -> str:
        """Format memories for inclusion in LLM prompt.

        Args:
            memories: List of memories to format

        Returns:
            Formatted string for prompt
        """
        if not memories:
            return "No relevant past experiences found."

        lines = ["## Relevant Past Experiences:\n"]

        for i, mem in enumerate(memories, 1):
            lines.append(f"### Experience {i} ({mem.outcome.upper()}):")
            lines.append(f"- Task: {mem.task_description[:150]}")
            if mem.strategy_used:
                lines.append(f"- Strategy: {mem.strategy_used.value}")
            if mem.tools_used:
                lines.append(f"- Tools: {', '.join(mem.tools_used[:5])}")
            if mem.lessons_learned:
                lines.append(f"- Lessons: {'; '.join(mem.lessons_learned[:3])}")
            lines.append("")

        return "\n".join(lines)

    def clear_all(self) -> None:
        """Clear all memories (use with caution)."""
        self.memories = {}
        self._dirty = True
        self._save_memories(force=True)

    def get_stats(self) -> dict[str, Any]:
        """Get memory bank statistics."""
        self._ensure_loaded()
        type_counts = {}
        outcome_counts = {"success": 0, "failure": 0}

        for mem in self.memories.values():
            type_counts[mem.memory_type.value] = type_counts.get(mem.memory_type.value, 0) + 1
            outcome_counts[mem.outcome] = outcome_counts.get(mem.outcome, 0) + 1

        return {
            "total_memories": len(self.memories),
            "by_type": type_counts,
            "by_outcome": outcome_counts,
            "storage_path": str(self._get_memory_file())
        }


class MemoryAugmentedPrompt:
    """Augments prompts with relevant memories."""

    def __init__(self, memory_bank: MemoryBank):
        """Initialize with memory bank.

        Args:
            memory_bank: MemoryBank instance
        """
        self.memory_bank = memory_bank

    def augment_prompt(
        self,
        original_prompt: str,
        state: EnhancedAgentState,
        max_memories: int = 3
    ) -> str:
        """Augment prompt with relevant memories.

        Args:
            original_prompt: Original user prompt
            state: Current agent state
            max_memories: Maximum memories to include

        Returns:
            Augmented prompt with memory context
        """
        memories = self.memory_bank.get_relevant_memories(state, limit=max_memories)

        if not memories:
            return original_prompt

        memory_context = self.memory_bank.format_memories_for_prompt(memories)

        augmented = f"""{original_prompt}

---
{memory_context}
Use these past experiences to inform your approach, but adapt to the current context.
"""
        return augmented


def create_memory_bank(
    storage_path: str | Path | None = None,
    project_id: str | None = None
) -> MemoryBank:
    """Factory function to create MemoryBank.

    Args:
        storage_path: Path for memory storage
        project_id: Project identifier

    Returns:
        Configured MemoryBank instance
    """
    return MemoryBank(storage_path=storage_path, project_id=project_id)


# ═══════════════════════════════════════════════════════════════════
# Claude Code Style: Session Persistence
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SessionData:
    """Session data for persistence."""
    session_id: str
    project_id: str
    start_time: datetime
    end_time: datetime | None
    task_summary: str
    messages_count: int
    tools_used: list[str]
    files_modified: list[str]
    outcome: str  # "success", "partial", "failure"
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON persistence."""
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "task_summary": self.task_summary,
            "messages_count": self.messages_count,
            "tools_used": self.tools_used,
            "files_modified": self.files_modified,
            "outcome": self.outcome,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            project_id=data["project_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            task_summary=data.get("task_summary", ""),
            messages_count=data.get("messages_count", 0),
            tools_used=data.get("tools_used", []),
            files_modified=data.get("files_modified", []),
            outcome=data.get("outcome", "unknown"),
            notes=data.get("notes", [])
        )


class SessionManager:
    """Manages session persistence (Claude Code style)."""

    MAX_SESSIONS = 50

    def __init__(self, sessions_path: Path | None = None):
        """Initialize session manager."""
        if sessions_path is None:
            ensure_global_sepilot_dir()
            sessions_path = GLOBAL_SEPILOT_DIR / "sessions"
        self.sessions_path = sessions_path
        self.sessions_path.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: SessionData) -> Path:
        """Save session to file."""
        filename = f"{session.session_id}.json"
        filepath = self.sessions_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

        # Clean up old sessions if exceeding limit
        self._cleanup_old_sessions()

        return filepath

    def load_session(self, session_id: str) -> SessionData | None:
        """Load session from file."""
        filepath = self.sessions_path / f"{session_id}.json"
        if not filepath.exists():
            return None

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            return SessionData.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def list_sessions(self, project_id: str | None = None, limit: int = 10) -> list[SessionData]:
        """List recent sessions."""
        sessions = []

        for filepath in sorted(self.sessions_path.glob("*.json"), reverse=True):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                session = SessionData.from_dict(data)

                if project_id is None or session.project_id == project_id:
                    sessions.append(session)

                if len(sessions) >= limit:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        return sessions

    def _cleanup_old_sessions(self) -> None:
        """Remove old session files exceeding MAX_SESSIONS."""
        session_files = sorted(
            self.sessions_path.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for old_file in session_files[self.MAX_SESSIONS:]:
            old_file.unlink()

    def create_session_from_state(
        self,
        state: EnhancedAgentState,
        task_summary: str = "",
        outcome: str = "success"
    ) -> SessionData:
        """Create session data from agent state."""
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.now()).encode(), usedforsecurity=False).hexdigest()[:8]}"

        # Extract data from state
        messages = state.get("messages", [])
        tool_history = state.get("tool_call_history", [])
        file_changes = state.get("file_changes", [])

        # Get task from first human message if not provided
        if not task_summary:
            for msg in messages:
                if hasattr(msg, "type") and msg.type == "human":
                    task_summary = getattr(msg, "content", "")[:200]
                    break

        return SessionData(
            session_id=session_id,
            project_id=hashlib.md5(os.getcwd().encode(), usedforsecurity=False).hexdigest()[:12],
            start_time=datetime.now(),
            end_time=None,
            task_summary=task_summary,
            messages_count=len(messages),
            tools_used=list({tc.tool_name for tc in tool_history}),
            files_modified=list({fc.file_path for fc in file_changes}),
            outcome=outcome,
            notes=[]
        )


def create_session_manager(sessions_path: Path | None = None) -> SessionManager:
    """Factory function to create SessionManager."""
    return SessionManager(sessions_path=sessions_path)
