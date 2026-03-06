"""Pattern Node Wrappers for LangGraph Integration

This module provides LangGraph-compatible node wrappers for all agent patterns,
enabling full visualization in the web dashboard.

Graph Structure:
```
triage
  ├─→ direct_response → END
  └─→ orchestrator → memory_retriever → hierarchical_planner → iteration_guard
                                                                    │
                     ┌──────────────────────────────────────────────┘
                     ↓
              context_manager → tool_recommender → agent
                                                     │
                     ┌───────────────────────────────┴────────────────┐
                     ↓                                                ↓
                 approval → tools → verifier               verifier (finalize)
                     │                 │                         │
                     │                 └─────────────────────────┤
                     │                                           ↓
                     │                                      reflection
                     │                                           │
                     │                         ┌─────────────────┼─────────────────┐
                     │                         ↓                 ↓                 ↓
                     │                   revise_plan      backtrack_check     escalate
                     │                   (→planner)             │            (→reporter)
                     │                                          ↓
                     │                                    debate_check
                     │                                    ┌─────┴─────┐
                     │                                    ↓           ↓
                     │                                 debate    (skip debate)
                     │                                    │           │
                     │                                    └─────┬─────┘
                     │                                          ↓
                     │                                   memory_writer
                     │                                          │
                     └──────────────────────────────────────────┤
                                                                ↓
                                              ┌─────────────────┴─────────────────┐
                                              ↓                                   ↓
                                       iteration_guard                       reporter → END
```
"""

from typing import Any

from sepilot.agent.backtracking import BacktrackingManager, CheckpointType
from sepilot.config.constants import (
    EXPLORATION_MAX_FILES,
    EXPLORATION_MAX_HINTS,
    EXPLORATION_MAX_SEARCH_TIME_MS,
)
from sepilot.agent.debate_node import DebateOrchestrator
from sepilot.agent.enhanced_state import EnhancedAgentState
from sepilot.agent.file_detector import FileDetectionResult, FilePathDetector
from sepilot.agent.hierarchical_planner import HierarchicalPlanner, PlanDepth
from sepilot.agent.memory_bank import MemoryBank
from sepilot.agent.pattern_orchestrator import AdaptiveOrchestrator
from sepilot.agent.tool_learning import ToolLearningSystem
from sepilot.tools.codebase_tools import CodebaseExplorer

import logging

_logger = logging.getLogger(__name__)


def _is_error_resolved(error: Any) -> bool:
    """Support both ErrorRecord objects and plain dict entries."""
    if isinstance(error, dict):
        return bool(error.get("resolved", False))
    return bool(getattr(error, "resolved", False))

# =============================================================================
# Orchestrator Node - Pattern Selection
# =============================================================================

class OrchestratorNode:
    """Analyzes task and selects appropriate patterns.

    Claude Code style: Optimizes pattern combinations based on historical success.
    This is the entry point after triage, determining which patterns
    should be active for the current task.
    """

    # Pattern combination effectiveness scores (learned from experience)
    # Higher score = more effective combination
    PATTERN_SYNERGIES = {
        ("memory_bank", "tool_learning"): 1.2,  # Memory + tool learning synergy
        ("hierarchical_planning", "memory_bank"): 1.3,  # Planning benefits from memory
        ("backtracking", "hierarchical_planning"): 1.1,  # Backtrack works with plans
    }

    # Patterns that conflict or add overhead without benefit
    PATTERN_CONFLICTS = {
        ("debate", "memory_bank"): 0.8,  # Debate + memory can be redundant
    }

    # Class-level cache: (task_type, complexity_bucket) → orchestration result
    _pattern_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def __init__(
        self,
        orchestrator: AdaptiveOrchestrator,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.orchestrator = orchestrator
        self.console = console
        self.verbose = verbose

    def _optimize_pattern_combination(
        self,
        patterns: list,
        complexity: int,
        task_type: str
    ) -> list:
        """Optimize pattern combination based on synergies and conflicts.

        Claude Code style: Smart pattern selection for efficiency.

        Args:
            patterns: Initial pattern list
            complexity: Task complexity (1-5)
            task_type: Detected task type

        Returns:
            Optimized pattern list
        """
        pattern_values = [p.value if hasattr(p, 'value') else str(p) for p in patterns]

        # For simple tasks (complexity <= 2), minimize patterns
        # But keep backtracking for safety (file modifications need rollback capability)
        if complexity <= 2:
            # Keep only essential patterns + backtracking for safety
            essential = ["memory_bank", "tool_learning", "backtracking"]
            pattern_values = [p for p in pattern_values if p in essential]

        # For complex tasks, ensure good combinations
        elif complexity >= 4:
            # Add synergistic patterns if missing
            if "hierarchical_planning" in pattern_values and "memory_bank" not in pattern_values:
                pattern_values.append("memory_bank")

        # Remove conflicting patterns for efficiency
        patterns_to_remove = set()
        for (p1, p2), score in self.PATTERN_CONFLICTS.items():
            if p1 in pattern_values and p2 in pattern_values and score < 1.0:
                # Remove the less important one (second in tuple)
                patterns_to_remove.add(p2)

        pattern_values = [p for p in pattern_values if p not in patterns_to_remove]

        return pattern_values

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Execute orchestration and pattern selection with optimization and caching."""
        try:
            plan = self.orchestrator.create_orchestration_plan(state)
        except Exception as e:
            if self.console and self.verbose:
                self.console.print(f"[dim red]🎯 Orchestrator error: {e}[/dim red]")
            # Fallback: minimal orchestration with memory_bank only
            return {
                "orchestration_plan": {},
                "active_patterns": ["memory_bank"],
                "orchestration_complete": True,
            }

        # Cache key: (task_type, complexity bucket)
        # Bucket complexity into ranges: 1-2=low, 3=mid, 4-5=high
        complexity_bucket = min(plan.complexity, 5)
        if complexity_bucket <= 2:
            complexity_bucket = 1
        elif complexity_bucket <= 3:
            complexity_bucket = 3
        else:
            complexity_bucket = 5
        cache_key = (plan.task_type.value, complexity_bucket)

        # Check cache for pattern optimization
        if cache_key in self._pattern_cache:
            cached = self._pattern_cache[cache_key]
            if self.console and self.verbose:
                self.console.print(
                    f"[bold magenta]🎯 Orchestrator (cached):[/bold magenta] "
                    f"Task={plan.task_type.value}, Complexity={plan.complexity}/5"
                )
            return {
                "orchestration_plan": plan.to_dict(),
                "active_patterns": cached["active_patterns"],
                "task_type_detected": plan.task_type.value,
                "task_complexity": plan.complexity,
                "planning_notes": [f"🎯 Orchestrator (cached): {plan.reasoning}"]
            }

        # Optimize pattern combination
        optimized_patterns = self._optimize_pattern_combination(
            plan.active_patterns,
            plan.complexity,
            plan.task_type.value
        )

        # Store in cache
        self._pattern_cache[cache_key] = {"active_patterns": optimized_patterns}

        if self.console and self.verbose:
            original_count = len(plan.active_patterns)
            optimized_count = len(optimized_patterns)
            optimization_note = ""
            if original_count != optimized_count:
                optimization_note = f" (optimized: {original_count}→{optimized_count})"

            self.console.print(
                f"[bold magenta]🎯 Orchestrator:[/bold magenta] "
                f"Task={plan.task_type.value}, Complexity={plan.complexity}/5{optimization_note}"
            )
            self.console.print(
                f"   Patterns: {', '.join(optimized_patterns)}"
            )

        return {
            "orchestration_plan": plan.to_dict(),
            "active_patterns": optimized_patterns,
            "task_type_detected": plan.task_type.value,
            "task_complexity": plan.complexity,
            "planning_notes": [f"🎯 Orchestrator: {plan.reasoning}"]
        }


# =============================================================================
# Codebase Exploration Node - Automatic File Discovery
# =============================================================================

class CodebaseExplorationNode:
    """Automatically explores codebase when file paths are not specified.

    Claude Code style: Before planning, detect if we need to explore the codebase
    and automatically search for relevant files based on the user's request.

    This solves the problem of agents creating plans that say "find file X"
    but never actually executing the search.
    """

    def __init__(
        self,
        file_detector: FilePathDetector,
        codebase_explorer: CodebaseExplorer,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.file_detector = file_detector
        self.explorer = codebase_explorer
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Perform automatic codebase exploration if needed."""
        # 1. Extract user prompt
        messages = state.get("messages", [])
        user_prompt = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                user_prompt = getattr(msg, "content", "")
                break

        if not user_prompt:
            return {}

        # 2. Detect file paths in the request
        detection = self.file_detector.detect(user_prompt)

        # 3. If explicit files are provided, skip exploration
        if detection.has_explicit_files and not detection.needs_exploration:
            if self.console and self.verbose:
                self.console.print(
                    f"[dim cyan]🎯 Explicit files detected: "
                    f"{', '.join(detection.detected_files[:3])}[/dim cyan]"
                )
            return {
                "exploration_skipped": True,
                "explicit_files": detection.detected_files,
                "exploration_context": ""
            }

        # 4. If no exploration needed based on detection
        if not detection.needs_exploration:
            if self.console and self.verbose:
                self.console.print("[dim]🔍 No codebase exploration needed[/dim]")
            return {"exploration_skipped": True, "exploration_context": ""}

        # 5. Perform automatic exploration
        if self.console and self.verbose:
            self.console.print("[cyan]🔍 Auto-exploring codebase...[/cyan]")

        exploration_results = self._perform_exploration(user_prompt, detection)

        # 6. Format results for planning context
        exploration_context = self._format_exploration_results(exploration_results)

        if self.console and self.verbose:
            total_files = exploration_results.get("total_files_found", 0)
            self.console.print(
                f"[cyan]🔍 Exploration complete: {total_files} relevant files found[/cyan]"
            )
            for f in exploration_results.get("related_files", [])[:3]:
                self.console.print(f"   → {f}")

        return {
            "exploration_performed": True,
            "exploration_context": exploration_context,
            "exploration_results": exploration_results,
            "exploration_hints": detection.exploration_hints,
            "project_type": detection.project_type,
            "planning_notes": [
                f"🔍 Auto-exploration: {exploration_results.get('total_files_found', 0)} files found"
            ]
        }

    def _perform_exploration(
        self, user_prompt: str, detection: "FileDetectionResult"
    ) -> dict[str, Any]:
        """Perform the actual codebase exploration.

        Claude Code style: Optimized for large monorepos with:
        - Time limits for early termination
        - File count limits to prevent memory issues
        - Prioritized search order

        Args:
            user_prompt: User's request text.
            detection: FileDetectionResult with hints and exploration keywords.

        Returns:
            Dictionary with exploration results.
        """
        import time

        # Performance limits for large monorepos
        MAX_TOTAL_FILES = EXPLORATION_MAX_FILES
        MAX_SEARCH_TIME_MS = EXPLORATION_MAX_SEARCH_TIME_MS
        MAX_HINTS_TO_PROCESS = EXPLORATION_MAX_HINTS

        start_time = time.time()

        results: dict[str, Any] = {
            "project_structure": None,
            "keyword_search_results": [],
            "related_files": [],
            "definition_files": [],  # Files with class/function definitions
            "total_files_found": 0,
            "exploration_limited": False
        }

        # Get file patterns based on project type
        file_patterns = self.file_detector.get_priority_patterns()
        primary_pattern = file_patterns[0] if file_patterns else "*.py"

        # Track all files and definition files separately
        all_files: set[str] = set()
        definition_files: set[str] = set()

        def _should_stop() -> bool:
            """Check if we should stop exploration early."""
            if len(all_files) >= MAX_TOTAL_FILES:
                results["exploration_limited"] = True
                return True
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > MAX_SEARCH_TIME_MS:
                results["exploration_limited"] = True
                return True
            return False

        # 1. PRIORITY: Search for class/function definitions first
        # This ensures we find the actual definition files, not just references
        for hint in detection.exploration_hints[:MAX_HINTS_TO_PROCESS]:
            if _should_stop():
                break
            for def_pattern in [f"class {hint}", f"def {hint}"]:
                if _should_stop():
                    break
                try:
                    search_results = list(
                        self.explorer.search_content_incremental(
                            def_pattern, file_pattern=primary_pattern, max_results=5
                        )
                    )
                    for file_path, matches in search_results:
                        definition_files.add(file_path)
                        all_files.add(file_path)
                        results["keyword_search_results"].append({
                            "file": file_path,
                            "keyword": def_pattern,
                            "matches": matches[:3],
                            "is_definition": True
                        })
                except Exception as e:
                    _logger.debug(f"Definition search failed for '{def_pattern}': {e}")
                    continue

        # 2. Search for general references (if not enough definitions found)
        if not _should_stop():
            for hint in detection.exploration_hints[:MAX_HINTS_TO_PROCESS]:
                if _should_stop():
                    break
                try:
                    search_results = list(
                        self.explorer.search_content_incremental(
                            hint, file_pattern=primary_pattern, max_results=10
                        )
                    )
                    for file_path, matches in search_results:
                        if file_path not in all_files:
                            all_files.add(file_path)
                            results["keyword_search_results"].append({
                                "file": file_path,
                                "keyword": hint,
                                "matches": matches[:3],
                                "is_definition": False
                            })
                except Exception as e:
                    _logger.debug(f"Reference search failed for '{hint}': {e}")
                    continue

        # 3. Extract core keywords from prompt and search (only if we have capacity)
        if not _should_stop():
            core_keywords = self._extract_core_keywords(user_prompt)
            for keyword in core_keywords[:3]:
                if _should_stop():
                    break
                try:
                    search_results = list(
                        self.explorer.search_content_incremental(
                            keyword, file_pattern="*", max_results=5
                        )
                    )
                    for file_path, _ in search_results:
                        all_files.add(file_path)
                except Exception as e:
                    _logger.debug(f"Keyword search failed for '{keyword}': {e}")
                    continue

        # 4. Compile results - prioritize definition files
        results["definition_files"] = sorted(definition_files)
        # Put definition files first, then other related files
        other_files = sorted(all_files - definition_files)
        results["related_files"] = list(definition_files) + other_files[:15]
        results["total_files_found"] = len(all_files)

        # Log performance metrics
        elapsed_ms = (time.time() - start_time) * 1000
        if self.console and self.verbose and results["exploration_limited"]:
            self.console.print(
                f"[dim yellow]⚡ Exploration limited: {len(all_files)} files, "
                f"{elapsed_ms:.0f}ms[/dim yellow]"
            )

        return results

    def _extract_core_keywords(self, text: str) -> list[str]:
        """Extract core keywords from user prompt.

        Args:
            text: User's request text.

        Returns:
            List of extracted keywords.
        """
        import re

        keywords: set[str] = set()

        # Pattern 1: Action + target (implement X, fix Y, add Z)
        action_pattern = r"(?:implement|create|add|modify|fix|update|refactor)\s+(\w+)"
        matches = re.findall(action_pattern, text, re.IGNORECASE)
        keywords.update(matches)

        # Pattern 2: Korean action + target
        kr_pattern = r"(\w{4,})(?:를|을|에|의|에서|으로)"
        matches = re.findall(kr_pattern, text)
        # Only add ASCII keywords (filter out Korean)
        keywords.update(m for m in matches if m.isascii())

        # Pattern 3: Quoted identifiers
        quoted_pattern = r"[`\"'](\w{3,})[`\"']"
        matches = re.findall(quoted_pattern, text)
        keywords.update(matches)

        return list(keywords)

    def _format_exploration_results(self, results: dict[str, Any]) -> str:
        """Format exploration results as context for planning.

        Args:
            results: Exploration results dictionary.

        Returns:
            Formatted string for inclusion in planning context.
        """
        lines = ["## Automatic Codebase Exploration Results\n"]

        # Definition files (PRIORITY - these contain class/function definitions)
        if results.get("definition_files"):
            lines.append("### ⭐ Definition Files (Primary Targets)")
            lines.append("*These files contain the class/function definitions you need to modify:*")
            for f in results["definition_files"][:5]:
                lines.append(f"- `{f}` ← **START HERE**")
            lines.append("")

        # Other related files
        other_files = [
            f for f in results.get("related_files", [])
            if f not in results.get("definition_files", [])
        ]
        if other_files:
            lines.append("### Related Files (References)")
            for f in other_files[:8]:
                lines.append(f"- `{f}`")
            if len(other_files) > 8:
                lines.append(f"- ... and {len(other_files) - 8} more")
            lines.append("")

        # Keyword matches - prioritize definitions
        if results.get("keyword_search_results"):
            lines.append("### Code Locations")
            # Show definitions first
            definitions = [
                item for item in results["keyword_search_results"]
                if item.get("is_definition")
            ]
            references = [
                item for item in results["keyword_search_results"]
                if not item.get("is_definition")
            ]

            for item in (definitions + references)[:5]:
                is_def = "⭐ " if item.get("is_definition") else ""
                lines.append(f"- {is_def}**{item['file']}** (keyword: `{item['keyword']}`)")
                for line_no, content in item.get("matches", [])[:2]:
                    preview = content[:60] + "..." if len(content) > 60 else content
                    lines.append(f"  - Line {line_no}: `{preview}`")
            lines.append("")

        if len(lines) == 1:
            return ""  # No useful results

        lines.append(
            "\n**Note:** Use `file_read` to examine the definition files first before making changes."
        )

        return "\n".join(lines)


# =============================================================================
# Memory Retriever Node - Experience Recall
# =============================================================================

class MemoryRetrieverNode:
    """Retrieves relevant past experiences before planning.

    Claude Code style: Enhanced similarity-based search with keyword extraction
    and weighted scoring for better memory retrieval.
    """

    def __init__(
        self,
        memory_bank: MemoryBank,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.memory_bank = memory_bank
        self.console = console
        self.verbose = verbose

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text for similarity matching.

        Claude Code style: Smart keyword extraction for better matching.
        """
        import re

        # Normalize text
        text_lower = text.lower()

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall",
            "this", "that", "these", "those", "it", "its",
            "and", "or", "but", "if", "then", "else", "when", "where",
            "how", "what", "which", "who", "whom", "whose", "why",
            "to", "for", "with", "about", "into", "through", "during",
            "before", "after", "above", "below", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further",
            "해줘", "하세요", "좀", "것", "수", "등", "및", "또는",
        }

        # Extract words (alphanumeric + Korean)
        words = re.findall(r'[a-zA-Z가-힣]+', text_lower)

        # Filter and return unique keywords
        keywords = {w for w in words if len(w) > 2 and w not in stop_words}

        return keywords

    def _calculate_similarity(self, query_keywords: set[str], memory_keywords: set[str]) -> float:
        """Calculate Jaccard similarity between keyword sets."""
        if not query_keywords or not memory_keywords:
            return 0.0

        intersection = len(query_keywords & memory_keywords)
        union = len(query_keywords | memory_keywords)

        return intersection / union if union > 0 else 0.0

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Retrieve relevant memories with enhanced similarity search."""
        # Check if memory pattern is active
        active_patterns = state.get("active_patterns", [])
        if "memory_bank" not in active_patterns:
            if self.console and self.verbose:
                self.console.print("[dim]📚 Memory retrieval skipped (pattern not active)[/dim]")
            return {"memory_retrieval_skipped": True}

        # Extract user query for similarity matching
        messages = state.get("messages", [])
        user_query = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                user_query = getattr(msg, "content", "")
                break

        # Extract keywords from query
        query_keywords = self._extract_keywords(user_query)

        # Get memories from bank
        memories = self.memory_bank.get_relevant_memories(state, limit=10)

        if not memories:
            if self.console and self.verbose:
                self.console.print("[dim cyan]📚 No relevant past experiences found[/dim cyan]")
            return {"retrieved_memories": [], "memory_context": ""}

        # Enhanced similarity scoring
        scored_memories = []
        for mem in memories:
            # Extract keywords from memory task description
            mem_keywords = self._extract_keywords(mem.task_description)

            # Calculate similarity score
            similarity = self._calculate_similarity(query_keywords, mem_keywords)

            # Boost score for successful experiences
            outcome_boost = 1.2 if mem.outcome == "success" else 0.8

            # Boost score for high-quality experiences
            quality_boost = 1.0
            if hasattr(mem, 'metadata') and mem.metadata:
                quality = mem.metadata.get('quality_score', 0.5)
                quality_boost = 0.8 + (quality * 0.4)  # Range: 0.8-1.2

            final_score = similarity * outcome_boost * quality_boost
            scored_memories.append((mem, final_score))

        # Sort by score and take top 3
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [m for m, score in scored_memories[:3] if score > 0.1]

        if not top_memories:
            if self.console and self.verbose:
                self.console.print("[dim cyan]📚 No sufficiently similar experiences found[/dim cyan]")
            return {"retrieved_memories": [], "memory_context": ""}

        # Format memories for context
        memory_context = self.memory_bank.format_memories_for_prompt(top_memories)

        if self.console and self.verbose:
            self.console.print(
                f"[cyan]📚 Memory Retriever:[/cyan] Found {len(top_memories)} relevant experiences"
            )
            for mem, score in scored_memories[:2]:
                outcome_icon = "✅" if mem.outcome == "success" else "❌"
                self.console.print(f"   {outcome_icon} {mem.task_description[:50]}... (sim: {score:.0%})")

        return {
            "retrieved_memories": [m.to_dict() for m in top_memories],
            "memory_context": memory_context,
            "planning_notes": [f"📚 Retrieved {len(top_memories)} relevant past experiences"]
        }


# =============================================================================
# Hierarchical Planner Node - Multi-Level Planning
# =============================================================================

class HierarchicalPlannerNode:
    """Creates hierarchical execution plans.

    Decomposes complex tasks into Strategic → Tactical → Operational levels.
    """

    def __init__(
        self,
        planner: HierarchicalPlanner,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.planner = planner
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Create hierarchical plan with dynamic depth (Claude Code style)."""
        # Check if hierarchical planning is active
        active_patterns = state.get("active_patterns", [])
        if "hierarchical_planning" not in active_patterns:
            if self.console and self.verbose:
                self.console.print("[dim]📋 Hierarchical planning skipped (pattern not active)[/dim]")
            return {"hierarchical_planning_skipped": True}

        # Get task description
        messages = state.get("messages", [])
        task_description = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task_description = getattr(msg, "content", "")
                break

        if not task_description:
            return {}

        # Claude Code style: Map task_complexity to PlanDepth
        task_complexity = state.get("task_complexity", "simple")
        if task_complexity == "simple":
            depth = PlanDepth.MINIMAL  # No LLM call
        elif task_complexity == "complex":
            depth = PlanDepth.FULL     # Full 3-level planning
        else:
            depth = PlanDepth.TACTICAL  # 2-level planning

        # Include exploration and memory context if available
        context = {
            "exploration_context": state.get("exploration_context", ""),
            "memory_context": state.get("memory_context", ""),
            "files": [fc.file_path for fc in state.get("file_changes", [])]
        }

        # Add explored files to context if available
        exploration_results = state.get("exploration_results", {})
        if exploration_results.get("related_files"):
            context["files"] = exploration_results["related_files"][:10]

        # Create hierarchical plan with appropriate depth
        try:
            plan = self.planner.create_plan(task_description, context, depth=depth)
        except Exception as e:
            if self.console and self.verbose:
                self.console.print(f"[dim red]📋 Planner error: {e}[/dim red]")
            # Fallback: single-step plan
            return {
                "hierarchical_plan": {},
                "plan_steps": [f"[T] {task_description[:200]}"],
                "plan_created": True,
                "current_plan_step": 0,
                "plan_execution_pending": True,
            }

        if self.console and self.verbose:
            progress = plan.get_progress()
            depth_label = depth.value.upper()
            self.console.print(
                f"[cyan]📋 Plan ({depth_label}):[/cyan] "
                f"{progress['total_tasks']} tasks"
            )

        # Convert to flat plan_steps for compatibility
        flat_steps = []
        for goal in plan.strategic_goals:
            flat_steps.append(f"[S] {goal.description}")
        for task in plan.tactical_tasks:
            flat_steps.append(f"[T] {task.description}")
        for action in plan.operational_actions:
            tools = ", ".join(action.tools_required[:3]) if action.tools_required else "auto"
            flat_steps.append(f"[O] {action.description} ({tools})")

        return {
            "hierarchical_plan": plan.to_dict(),
            "plan_steps": flat_steps,
            "plan_created": True,
            "current_plan_step": 0,
            "planning_notes": [
                f"📋 {depth.value.capitalize()} plan: {len(plan.strategic_goals)}S/{len(plan.tactical_tasks)}T/{len(plan.operational_actions)}O"
            ]
        }


# =============================================================================
# Tool Recommender Node - Tool Selection Optimization
# =============================================================================

class ToolRecommenderNode:
    """Recommends optimal tools based on learned patterns.

    Provides tool suggestions before agent execution based on
    past success rates and task context.
    """

    def __init__(
        self,
        tool_learning: ToolLearningSystem,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.tool_learning = tool_learning
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Generate tool recommendations."""
        # Check if tool learning is active
        active_patterns = state.get("active_patterns", [])
        if "tool_learning" not in active_patterns:
            if self.console and self.verbose:
                self.console.print("[dim]🔧 Tool recommendations skipped (pattern not active)[/dim]")
            return {"tool_recommendations_skipped": True}

        # Get context
        messages = state.get("messages", [])
        task_description = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task_description = getattr(msg, "content", "")
                break

        # Early exit if no task description
        if not task_description:
            return {"tool_recommendations": [], "tool_recommendations_skipped": True}

        strategy = state.get("current_strategy")
        tool_history = state.get("tool_call_history", [])
        current_tool = tool_history[-1].tool_name if tool_history else None

        # Get recommendations (may return None or empty list)
        try:
            recommendations = self.tool_learning.recommend_tools(
                task_description=task_description,
                current_tool=current_tool,
                strategy=strategy,
                limit=3
            )
        except Exception as e:
            _logger.debug(f"Tool recommendation failed: {e}")
            return {"tool_recommendations": [], "tool_recommendations_error": str(e)[:100]}

        if not recommendations:
            return {"tool_recommendations": []}

        if self.console and self.verbose:
            self.console.print("[cyan]🔧 Tool Recommender:[/cyan]")
            for rec in recommendations[:3]:
                self.console.print(
                    f"   → {rec.tool_name} ({rec.confidence:.0%}) - {rec.reason}"
                )

        # Get optimal sequence if available (may return None)
        task_type = strategy.value if strategy else "general"
        try:
            optimal_sequence = self.tool_learning.get_optimal_sequence(task_type)
        except Exception:
            optimal_sequence = None

        return {
            "tool_recommendations": [
                {
                    "tool": rec.tool_name,
                    "confidence": rec.confidence,
                    "reason": rec.reason,
                    "expected_success": rec.expected_success_rate
                }
                for rec in recommendations
            ],
            "suggested_tool_sequence": optimal_sequence or [],
            "planning_notes": [
                f"🔧 Recommended: {', '.join(r.tool_name for r in recommendations[:3])}"
            ]
        }


# =============================================================================
# Backtrack Check Node - Rollback Decision
# =============================================================================

class BacktrackCheckNode:
    """Checks if rollback is needed and performs it.

    Analyzes current state for failure patterns and triggers
    rollback to a previous checkpoint if necessary.
    """

    def __init__(
        self,
        backtracking: BacktrackingManager,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.backtracking = backtracking
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Check for rollback conditions and execute if needed."""
        # Check if backtracking is active
        active_patterns = state.get("active_patterns", [])
        if "backtracking" not in active_patterns:
            if self.console and self.verbose:
                self.console.print("[dim]⏪ Backtracking skipped (pattern not active)[/dim]")
            return {"backtracking_skipped": True, "backtrack_decision": "skip"}

        # Create checkpoint if we don't have a recent one
        if not self.backtracking.get_latest_checkpoint():
            self.backtracking.create_checkpoint(
                state=state,
                checkpoint_type=CheckpointType.AUTO,
                description="Auto checkpoint before backtrack check"
            )

        # Check if rollback is needed
        should_rollback, reason = self.backtracking.should_rollback(state)

        if should_rollback and reason:
            checkpoint = self.backtracking.get_latest_checkpoint()

            if checkpoint:
                if self.console and self.verbose:
                    self.console.print(
                        f"[yellow]⏪ Backtrack Check:[/yellow] "
                        f"Rollback triggered - {reason.value}"
                    )

                result = self.backtracking.rollback_to_checkpoint(
                    checkpoint=checkpoint,
                    reason=reason,
                    restore_files=True
                )

                if result.success:
                    return {
                        "backtrack_decision": "rollback",
                        "backtrack_performed": True,
                        "backtrack_reason": reason.value,
                        "files_restored": result.files_restored,
                        "planning_notes": [f"⏪ Rolled back: {reason.value}"],
                        # Reset plan to force re-planning
                        "plan_created": False,
                        "needs_additional_iteration": True
                    }

        # No rollback needed - create checkpoint for future
        self.backtracking.create_auto_checkpoint(state)

        if self.console and self.verbose:
            self.console.print("[dim cyan]⏪ Backtrack Check: No rollback needed[/dim cyan]")

        return {
            "backtrack_decision": "continue",
            "backtrack_performed": False
        }


# =============================================================================
# Debate Check Node - Multi-Perspective Analysis Decision
# =============================================================================

class DebateCheckNode:
    """Determines if debate/review is needed for the current task.

    Triggers multi-perspective analysis for code review, security checks,
    or complex architectural decisions.
    """

    # Keywords that trigger debate
    DEBATE_TRIGGERS = [
        "review", "check", "verify", "audit", "security",
        "architecture", "design", "refactor", "리뷰", "검토"
    ]

    def __init__(
        self,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Check if debate should be triggered."""
        # Check if debate is active
        active_patterns = state.get("active_patterns", [])
        if "debate" not in active_patterns:
            if self.console and self.verbose:
                self.console.print("[dim]🎭 Debate skipped (pattern not active)[/dim]")
            return {"debate_decision": "skip", "debate_skipped": True}

        # Get task description
        messages = state.get("messages", [])
        task = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task = getattr(msg, "content", "").lower()
                break

        # Check for trigger keywords
        should_debate = any(trigger in task for trigger in self.DEBATE_TRIGGERS)

        # Also trigger for code review task type
        task_type = state.get("task_type_detected", "")
        if task_type == "code_review":
            should_debate = True

        if should_debate:
            if self.console and self.verbose:
                self.console.print("[cyan]🎭 Debate Check:[/cyan] Triggering multi-perspective analysis")
            return {"debate_decision": "debate"}
        else:
            if self.console and self.verbose:
                self.console.print("[dim cyan]🎭 Debate Check: No debate needed[/dim cyan]")
            return {"debate_decision": "skip"}


# =============================================================================
# Debate Node - Multi-Perspective Analysis
# =============================================================================

class DebateNode:
    """Conducts multi-perspective debate for quality decisions.

    Uses Proposer → Critic → Resolver pattern for thorough analysis.
    """

    def __init__(
        self,
        orchestrator: DebateOrchestrator,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.orchestrator = orchestrator
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Conduct debate on current work."""
        # Get context
        messages = state.get("messages", [])
        task = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task = getattr(msg, "content", "")
                break

        # Get code context
        file_changes = state.get("file_changes", [])
        code_context = ""
        for fc in file_changes[:3]:
            if fc.new_content:
                code_context += f"\n--- {fc.file_path} ---\n{fc.new_content[:1000]}\n"

        context = f"Task: {task}\n\nCode:\n{code_context}"

        if self.console and self.verbose:
            self.console.print("[cyan]🎭 Debate:[/cyan] Starting multi-perspective analysis...")

        # Conduct debate
        try:
            result = self.orchestrator.conduct_debate(
                topic=task[:200],
                context=context
            )
        except Exception as e:
            if self.console and self.verbose:
                self.console.print(f"[dim red]🎭 Debate error: {e}[/dim red]")
            # Fallback: approve with low confidence
            return {
                "debate_result": {},
                "debate_decision_final": "Approved (debate fallback)",
                "debate_confidence": 0.5,
                "debate_outcome": "approved",
                "planning_notes": ["🎭 Debate skipped due to error"],
                "verification_notes": [],
            }

        if self.console and self.verbose:
            self.console.print(
                f"[bold cyan]🎭 Debate Result:[/bold cyan] {result.final_outcome.value} "
                f"(confidence: {result.confidence:.0%})"
            )
            if result.key_points:
                self.console.print(f"   Key points: {'; '.join(result.key_points[:2])}")

        return {
            "debate_result": result.to_dict(),
            "debate_decision_final": result.final_decision,
            "debate_confidence": result.confidence,
            "debate_outcome": result.final_outcome.value,
            "planning_notes": [
                f"🎭 Debate: {result.final_outcome.value} ({result.confidence:.0%})"
            ],
            "verification_notes": result.recommendations
        }


# =============================================================================
# Memory Writer Node - Experience Storage
# =============================================================================

class MemoryWriterNode:
    """Stores execution experience to memory bank.

    Claude Code style: Always store experiences for learning.
    All task outcomes (success/failure) provide valuable learning data.
    """

    def __init__(
        self,
        memory_bank: MemoryBank,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.memory_bank = memory_bank
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Store current execution experience.

        Claude Code style: Always record experiences for continuous learning.
        Even simple tasks provide valuable patterns for future optimization.
        """
        # Claude Code style: Always store experiences (opt-out via explicit flag)
        # Skip only if explicitly disabled (e.g., for testing or privacy)
        if state.get("disable_memory_storage"):
            if self.console and self.verbose:
                self.console.print("[dim]💾 Memory write disabled by flag[/dim]")
            return {"memory_write_skipped": True}

        # Determine outcome
        error_history = state.get("error_history", [])
        verification_notes = state.get("verification_notes", [])

        # Simple heuristic for success/failure
        has_errors = len(error_history) > 0 and not all(_is_error_resolved(e) for e in error_history)
        has_success_notes = any("✅" in note or "완료" in note or "success" in note.lower()
                                for note in verification_notes)

        outcome = "failure" if has_errors and not has_success_notes else "success"

        # Claude Code style: Evaluate experience quality
        quality_score = self._evaluate_experience_quality(state, outcome)

        # Skip low-quality experiences to avoid polluting memory
        if quality_score < 0.2 and outcome == "failure":
            if self.console and self.verbose:
                self.console.print(
                    f"[dim yellow]💾 Low-quality experience skipped (score: {quality_score:.1%})[/dim yellow]"
                )
            return {"memory_stored": False, "skip_reason": "low_quality"}

        # Extract lessons learned from reflection
        reflection_notes = state.get("reflection_notes", [])
        lessons = [note for note in reflection_notes if "Decision:" in note or "Pattern:" in note]

        # Track pattern effectiveness (which patterns were active and their correlation to outcome)
        active_patterns = state.get("active_patterns", [])
        pattern_metrics = self._build_pattern_metrics(state, outcome, active_patterns)

        # Store memory with quality metadata
        try:
            memory = self.memory_bank.store_from_state(
                state=state,
                outcome=outcome,
                lessons=lessons[:5]
            )
        except Exception as e:
            if self.console and self.verbose:
                self.console.print(f"[dim red]💾 Memory store error: {e}[/dim red]")
            return {"memory_stored": False, "memory_error": str(e)}

        if memory:
            # Add quality score and pattern metrics to memory metadata
            if hasattr(memory, 'metadata'):
                memory.metadata = memory.metadata or {}
                memory.metadata['quality_score'] = quality_score
                memory.metadata['pattern_metrics'] = pattern_metrics
                memory.metadata['active_patterns'] = active_patterns

            if self.console and self.verbose:
                outcome_icon = "✅" if outcome == "success" else "❌"
                patterns_str = ", ".join(active_patterns[:3]) if active_patterns else "none"
                self.console.print(
                    f"[cyan]💾 Memory Writer:[/cyan] {outcome_icon} Experience stored "
                    f"(ID: {memory.memory_id[:8]}, quality: {quality_score:.0%}, patterns: {patterns_str})"
                )

            return {
                "memory_stored": True,
                "memory_id": memory.memory_id,
                "memory_outcome": outcome,
                "memory_quality": quality_score,
                "pattern_metrics": pattern_metrics,
                "planning_notes": [f"💾 Experience stored: {outcome} (quality: {quality_score:.0%})"]
            }

        return {"memory_stored": False}

    def _build_pattern_metrics(
        self,
        state: EnhancedAgentState,
        outcome: str,
        active_patterns: list
    ) -> dict[str, Any]:
        """Build pattern effectiveness metrics.

        Claude Code style: Track which patterns contributed to success/failure.

        Returns:
            Dictionary with pattern metrics for analysis.
        """
        metrics: dict[str, Any] = {
            "outcome": outcome,
            "patterns_used": active_patterns,
            "pattern_contributions": {}
        }

        # Track individual pattern contributions
        for pattern in active_patterns:
            contribution = {"active": True, "outcome": outcome}

            if pattern == "memory_bank":
                # Did memory retrieval help?
                retrieved = state.get("retrieved_memories", [])
                contribution["memories_retrieved"] = len(retrieved)
                contribution["helpful"] = len(retrieved) > 0 and outcome == "success"

            elif pattern == "hierarchical_planning":
                # Was the plan effective?
                plan = state.get("hierarchical_plan", {})
                plan_steps = state.get("plan_steps", [])
                contribution["steps_planned"] = len(plan_steps)
                contribution["helpful"] = len(plan_steps) > 0 and outcome == "success"

            elif pattern == "tool_learning":
                # Did tool recommendations help?
                recommendations = state.get("tool_recommendations", [])
                tool_history = state.get("tool_call_history", [])
                # Check if recommended tools were actually used
                rec_tools = {r.get("tool") for r in recommendations}
                used_tools = {tc.tool_name for tc in tool_history}
                overlap = len(rec_tools & used_tools)
                contribution["recommendations"] = len(recommendations)
                contribution["recommendations_followed"] = overlap
                contribution["helpful"] = overlap > 0 and outcome == "success"

            elif pattern == "backtracking":
                # Was backtracking triggered and helpful?
                backtrack_performed = state.get("backtrack_performed", False)
                contribution["triggered"] = backtrack_performed
                contribution["helpful"] = backtrack_performed and outcome == "success"

            elif pattern == "debate":
                # Did debate provide valuable insights?
                debate_result = state.get("debate_result", {})
                contribution["confidence"] = debate_result.get("confidence", 0.0)
                contribution["helpful"] = bool(debate_result) and outcome == "success"

            metrics["pattern_contributions"][pattern] = contribution

        # Calculate overall pattern effectiveness score
        helpful_count = sum(
            1 for c in metrics["pattern_contributions"].values()
            if c.get("helpful", False)
        )
        total_patterns = len(active_patterns) if active_patterns else 1
        metrics["effectiveness_score"] = helpful_count / total_patterns

        return metrics

    def _evaluate_experience_quality(self, state: EnhancedAgentState, outcome: str) -> float:
        """Evaluate experience quality for memory storage.

        Claude Code style: Only store high-quality experiences to improve future performance.

        Quality factors:
        - Iteration efficiency (fewer iterations = better)
        - Error ratio (fewer errors = better)
        - Tool success rate (higher = better)
        - Task completion (successful = better)

        Returns:
            Quality score between 0.0 and 1.0
        """
        scores = []

        # 1. Iteration efficiency (weight: 30%)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 10)
        if iteration_count > 0:
            iteration_score = max(0, 1 - (iteration_count / max_iterations))
            scores.append(iteration_score * 0.3)
        else:
            scores.append(0.3)  # Perfect score if no iterations

        # 2. Error ratio (weight: 25%)
        error_history = state.get("error_history", [])
        tool_history = state.get("tool_call_history", [])
        if tool_history:
            error_ratio = len(error_history) / len(tool_history)
            error_score = max(0, 1 - error_ratio)
            scores.append(error_score * 0.25)
        else:
            scores.append(0.25)

        # 3. Tool success rate (weight: 25%)
        if tool_history:
            success_count = sum(1 for tc in tool_history if tc.success)
            success_rate = success_count / len(tool_history)
            scores.append(success_rate * 0.25)
        else:
            scores.append(0.15)  # Partial score if no tools used

        # 4. Task outcome (weight: 20%)
        outcome_score = 0.2 if outcome == "success" else 0.0
        scores.append(outcome_score)

        return sum(scores)


# =============================================================================
# Tool Usage Recorder Node - Tool Learning Update
# =============================================================================

class ToolRecorderNode:
    """Records tool usage for learning optimization.

    Claude Code style: Always record tool usage for learning.
    All tool executions provide valuable data for optimization.
    """

    def __init__(
        self,
        tool_learning: ToolLearningSystem,
        console: Any | None = None,
        verbose: bool = False
    ):
        self.tool_learning = tool_learning
        self.console = console
        self.verbose = verbose

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Record tool usage from current execution.

        Claude Code style: Always record tool usage for continuous learning.
        """
        # Claude Code style: Always record (opt-out via explicit flag)
        if state.get("disable_tool_recording"):
            return {"tool_recording_skipped": True}

        # Determine if task completed
        verification_notes = state.get("verification_notes", [])
        task_completed = any("✅" in note or "완료" in note for note in verification_notes)

        # Record from state
        try:
            self.tool_learning.record_from_state(state, task_completed)
        except Exception as e:
            if self.console and self.verbose:
                self.console.print(f"[dim red]📊 Tool recording error: {e}[/dim red]")
            return {"tool_usage_recorded": False, "tool_recording_error": str(e)}

        if self.console and self.verbose:
            stats = self.tool_learning.get_statistics_summary()
            self.console.print(
                f"[dim cyan]📊 Tool Recorder: {stats['total_usages_recorded']} usages recorded[/dim cyan]"
            )

        return {"tool_usage_recorded": True}


# =============================================================================
# Factory Functions
# =============================================================================

def create_orchestrator_node(
    orchestrator: AdaptiveOrchestrator,
    console: Any | None = None,
    verbose: bool = False
) -> OrchestratorNode:
    """Create orchestrator node."""
    return OrchestratorNode(orchestrator, console, verbose)


def create_memory_retriever_node(
    memory_bank: MemoryBank,
    console: Any | None = None,
    verbose: bool = False
) -> MemoryRetrieverNode:
    """Create memory retriever node."""
    return MemoryRetrieverNode(memory_bank, console, verbose)


def create_hierarchical_planner_node(
    planner: HierarchicalPlanner,
    console: Any | None = None,
    verbose: bool = False
) -> HierarchicalPlannerNode:
    """Create hierarchical planner node."""
    return HierarchicalPlannerNode(planner, console, verbose)


def create_tool_recommender_node(
    tool_learning: ToolLearningSystem,
    console: Any | None = None,
    verbose: bool = False
) -> ToolRecommenderNode:
    """Create tool recommender node."""
    return ToolRecommenderNode(tool_learning, console, verbose)


def create_backtrack_check_node(
    backtracking: BacktrackingManager,
    console: Any | None = None,
    verbose: bool = False
) -> BacktrackCheckNode:
    """Create backtrack check node."""
    return BacktrackCheckNode(backtracking, console, verbose)


def create_debate_check_node(
    console: Any | None = None,
    verbose: bool = False
) -> DebateCheckNode:
    """Create debate check node."""
    return DebateCheckNode(console, verbose)


def create_debate_node(
    orchestrator: DebateOrchestrator,
    console: Any | None = None,
    verbose: bool = False
) -> DebateNode:
    """Create debate node."""
    return DebateNode(orchestrator, console, verbose)


def create_memory_writer_node(
    memory_bank: MemoryBank,
    console: Any | None = None,
    verbose: bool = False
) -> MemoryWriterNode:
    """Create memory writer node."""
    return MemoryWriterNode(memory_bank, console, verbose)


def create_tool_recorder_node(
    tool_learning: ToolLearningSystem,
    console: Any | None = None,
    verbose: bool = False
) -> ToolRecorderNode:
    """Create tool recorder node."""
    return ToolRecorderNode(tool_learning, console, verbose)


def create_codebase_exploration_node(
    file_detector: FilePathDetector,
    codebase_explorer: CodebaseExplorer,
    console: Any | None = None,
    verbose: bool = False
) -> CodebaseExplorationNode:
    """Create codebase exploration node.

    Args:
        file_detector: FilePathDetector instance for detecting file paths.
        codebase_explorer: CodebaseExplorer instance for searching files.
        console: Rich console for output (optional).
        verbose: Whether to print verbose output.

    Returns:
        Configured CodebaseExplorationNode instance.
    """
    return CodebaseExplorationNode(
        file_detector=file_detector,
        codebase_explorer=codebase_explorer,
        console=console,
        verbose=verbose
    )
