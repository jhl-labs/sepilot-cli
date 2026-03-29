"""Self-Reflection Node for Agent Self-Improvement

This module implements the Reflexion/Self-Refine pattern for AI agents.
The reflection node analyzes execution history, identifies failure patterns,
and proposes strategy adjustments for improved performance.

Inspired by:
- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)
- Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)
- ReAct with Reflection patterns
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sepilot.agent.enhanced_state import AgentMode, AgentStrategy, EnhancedAgentState
from sepilot.agent.execution_context import (
    get_current_execution_messages,
    get_current_user_query,
)
from sepilot.config.constants import MAX_REFLECTION_ITERATIONS as _MAX_REFLECTION_ITERATIONS


class ReflectionDecision(str, Enum):
    """Possible decisions from the reflection node."""
    REVISE_PLAN = "revise_plan"       # Go back to planner with new insights
    REFINE_STRATEGY = "refine_strategy"  # Change strategy and retry agent
    PROCEED = "proceed"               # Continue with current approach
    ESCALATE = "escalate"             # Request human intervention


@dataclass
class ReflectionResult:
    """Result of a reflection analysis."""
    decision: ReflectionDecision
    critique: str
    root_causes: list[str]
    suggested_strategy: AgentStrategy | None
    confidence_adjustment: float  # -1.0 to 1.0
    insights: list[str]
    should_reset_plan: bool = False
    recommended_tools: list[str] = field(default_factory=list)
    suggested_mode: AgentMode | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "decision": self.decision.value,
            "critique": self.critique,
            "root_causes": self.root_causes,
            "suggested_strategy": self.suggested_strategy.value if self.suggested_strategy else None,
            "suggested_mode": self.suggested_mode.value if self.suggested_mode else None,
            "confidence_adjustment": self.confidence_adjustment,
            "insights": self.insights,
            "should_reset_plan": self.should_reset_plan,
            "recommended_tools": self.recommended_tools,
            "timestamp": self.timestamp.isoformat()
        }


class FailurePatternDetector:
    """Detects common failure patterns in agent execution."""

    # Pattern definitions with detection logic
    PATTERNS = {
        "stuck_on_single_tool": {
            "description": "Agent repeatedly uses the same tool without progress",
            "min_repetitions": 3,
            "suggested_strategy": AgentStrategy.EXPLORE,
            "action": "Diversify tool usage - explore codebase first"
        },
        "repeating_error": {
            "description": "Same error occurring multiple times",
            "min_repetitions": 2,
            "suggested_strategy": AgentStrategy.DEBUG,
            "action": "Analyze error root cause before retrying"
        },
        "no_file_changes": {
            "description": "Expected file modifications but none made",
            "min_iterations": 3,
            "suggested_strategy": AgentStrategy.IMPLEMENT,
            "action": "Force tool execution for file modifications"
        },
        "plan_execution_gap": {
            "description": "Plan exists but not being executed",
            "min_iterations": 2,
            "suggested_strategy": AgentStrategy.IMPLEMENT,
            "action": "Execute plan steps immediately"
        },
        "tool_failure_cascade": {
            "description": "Multiple consecutive tool failures",
            "min_failures": 3,
            "suggested_strategy": AgentStrategy.EXPLORE,
            "action": "Re-examine assumptions and explore alternatives"
        },
        "circular_reasoning": {
            "description": "Agent revisiting same conclusions without action",
            "suggested_strategy": AgentStrategy.IMPLEMENT,
            "action": "Break loop with direct action"
        },
        "incomplete_analysis": {
            "description": "Analysis started but task incomplete",
            "suggested_strategy": AgentStrategy.IMPLEMENT,
            "action": "Move from analysis to implementation"
        },
        "wrong_file_target": {
            "description": "Editing files unrelated to the user request",
            "suggested_strategy": AgentStrategy.EXPLORE,
            "action": "Re-read the request and find the correct target files"
        },
        "read_without_write": {
            "description": "Multiple reads without any write/edit action",
            "min_reads": 3,
            "suggested_strategy": AgentStrategy.IMPLEMENT,
            "action": "Stop reading and start implementing changes"
        },
        "shallow_search": {
            "description": "Searched then immediately edited without reading",
            "suggested_strategy": AgentStrategy.EXPLORE,
            "action": "Read file content before making edits"
        },
        "overconfident_completion": {
            "description": "Claimed completion without evidence of actual changes",
            "suggested_strategy": AgentStrategy.IMPLEMENT,
            "action": "Verify that changes were actually made before completing"
        },
    }

    def detect_patterns(self, state: EnhancedAgentState) -> list[dict[str, Any]]:
        """Detect all active failure patterns in current state.

        Args:
            state: Current agent state

        Returns:
            List of detected patterns with metadata
        """
        detected = []

        # Get state data
        tool_history = state.get("tool_call_history", [])
        error_history = state.get("error_history", [])
        file_changes = state.get("file_changes", [])
        iteration_count = state.get("iteration_count", 0)
        plan_steps = state.get("plan_steps", [])
        plan_created = state.get("plan_created", False)

        # Pattern 1: Stuck on single tool
        if len(tool_history) >= 3:
            recent_tools = [tc.tool_name for tc in tool_history[-5:]]
            if len(set(recent_tools)) == 1:
                detected.append({
                    "pattern": "stuck_on_single_tool",
                    "details": f"Tool '{recent_tools[0]}' called {len(recent_tools)} times consecutively",
                    **self.PATTERNS["stuck_on_single_tool"]
                })

        # Pattern 2: Repeating errors
        if len(error_history) >= 2:
            recent_errors = [e.message for e in error_history[-3:]]
            if len(recent_errors) >= 2 and recent_errors[0] == recent_errors[-1]:
                detected.append({
                    "pattern": "repeating_error",
                    "details": f"Error repeated: {recent_errors[0][:100]}",
                    **self.PATTERNS["repeating_error"]
                })

        # Pattern 3: No file changes when expected
        planning_notes = state.get("planning_notes", [])
        is_modification_task = not any("[READ-ONLY]" in note for note in planning_notes)
        if is_modification_task and not file_changes and iteration_count >= 3:
            detected.append({
                "pattern": "no_file_changes",
                "details": f"No file modifications after {iteration_count} iterations",
                **self.PATTERNS["no_file_changes"]
            })

        # Pattern 4: Plan exists but not executed
        if plan_created and len(plan_steps) > 0:
            executed_tools = len(tool_history)
            if executed_tools == 0 and iteration_count >= 2:
                detected.append({
                    "pattern": "plan_execution_gap",
                    "details": f"Plan has {len(plan_steps)} steps but no tools executed",
                    **self.PATTERNS["plan_execution_gap"]
                })

        # Pattern 5: Tool failure cascade
        if len(tool_history) >= 3:
            recent_results = [tc.success for tc in tool_history[-5:]]
            consecutive_failures = 0
            for success in reversed(recent_results):
                if not success:
                    consecutive_failures += 1
                else:
                    break
            if consecutive_failures >= 3:
                detected.append({
                    "pattern": "tool_failure_cascade",
                    "details": f"{consecutive_failures} consecutive tool failures",
                    **self.PATTERNS["tool_failure_cascade"]
                })

        # Pattern 6: Circular reasoning (similar AI responses)
        current_messages = get_current_execution_messages(state)
        ai_responses = [
            msg.content[:200] for msg in current_messages
            if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None)
        ]
        if len(ai_responses) >= 3:
            # Simple similarity check - same prefix in multiple responses
            recent_responses = ai_responses[-3:]
            if len(set(recent_responses)) == 1:
                detected.append({
                    "pattern": "circular_reasoning",
                    "details": "Similar responses detected without progress",
                    **self.PATTERNS["circular_reasoning"]
                })

        # Pattern 7: Wrong file target - editing files not mentioned in request
        if file_changes and len(tool_history) >= 2:
            # Get user request to check file relevance
            user_request = get_current_user_query(state).lower()
            if user_request:
                edit_tools = [tc for tc in tool_history if tc.tool_name in ("file_edit", "file_write")]
                if edit_tools:
                    edited_files = [
                        str(getattr(tc, "args", {}).get("file_path", ""))
                        for tc in edit_tools
                        if hasattr(tc, "args")
                    ]
                    # Check if none of the edited files are mentioned in the request
                    if edited_files and all(
                        not any(part in user_request for part in f.split("/")[-1:] if part)
                        for f in edited_files if f
                    ):
                        # Only flag if the request mentions specific files
                        import re
                        if re.search(r'\b\w+\.\w{1,5}\b', user_request):
                            detected.append({
                                "pattern": "wrong_file_target",
                                "details": f"Edited {edited_files} but request mentions different files",
                                **self.PATTERNS["wrong_file_target"]
                            })

        # Pattern 8: Read without write (3+ reads, 0 writes for modification task)
        if len(tool_history) >= 3 and is_modification_task:
            recent = tool_history[-5:]
            read_count = sum(1 for tc in recent if tc.tool_name in ("file_read", "search_content", "find_file"))
            write_count = sum(1 for tc in recent if tc.tool_name in ("file_edit", "file_write"))
            if read_count >= 3 and write_count == 0:
                detected.append({
                    "pattern": "read_without_write",
                    "details": f"{read_count} reads, 0 writes in last {len(recent)} tool calls",
                    **self.PATTERNS["read_without_write"]
                })

        # Pattern 9: Shallow search - searched then edited without reading
        if len(tool_history) >= 2:
            for i in range(len(tool_history) - 1):
                if (tool_history[i].tool_name in ("find_file", "search_content")
                        and tool_history[i + 1].tool_name in ("file_edit", "file_write")):
                    detected.append({
                        "pattern": "shallow_search",
                        "details": f"Jumped from {tool_history[i].tool_name} to {tool_history[i+1].tool_name} without reading",
                        **self.PATTERNS["shallow_search"]
                    })
                    break  # Only report once

        # Pattern 10: Overconfident completion - verification said complete but no file changes on modification task
        verification_notes = state.get("verification_notes", [])
        if is_modification_task and not file_changes and verification_notes:
            if any("completed" in str(note).lower() or "✅" in str(note) for note in verification_notes):
                detected.append({
                    "pattern": "overconfident_completion",
                    "details": "Verification claimed completion but no file changes on modification task",
                    **self.PATTERNS["overconfident_completion"]
                })

        return detected


class CritiqueGenerator:
    """Generates self-critique using LLM analysis."""

    CRITIQUE_SYSTEM_PROMPT = """You are a SELF-REFLECTION SPECIALIST for an AI coding agent.

Your task: Analyze the agent's execution history and provide constructive critique.

You must be:
- Honest: Identify actual problems, not just validate success
- Specific: Point to exact issues in tool usage or reasoning
- Actionable: Suggest concrete improvements

Analyze the execution and respond in this JSON format:
{
    "goal_achieved": true/false,
    "execution_quality": "excellent|good|fair|poor",
    "issues_found": ["issue1", "issue2"],
    "root_causes": ["cause1", "cause2"],
    "suggested_improvements": ["improvement1", "improvement2"],
    "recommended_tools": ["tool1", "tool2"],
    "confidence_score": 0.0-1.0,
    "should_retry": true/false,
    "retry_strategy": "description of what to do differently"
}"""

    def __init__(self, llm: BaseChatModel):
        """Initialize critique generator.

        Args:
            llm: Language model for generating critique
        """
        self.llm = llm

    def generate_critique(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Generate LLM-based self-critique of execution.

        Args:
            state: Current agent state

        Returns:
            Critique analysis dict
        """
        # Build context from state
        tool_history = state.get("tool_call_history", [])
        error_history = state.get("error_history", [])
        file_changes = state.get("file_changes", [])
        plan_steps = state.get("plan_steps", [])
        iteration_count = state.get("iteration_count", 0)

        # Get original task
        current_messages = get_current_execution_messages(state)
        original_task = get_current_user_query(state)

        # Get recent agent responses
        recent_responses = []
        for msg in current_messages[-10:]:
            if isinstance(msg, AIMessage):
                content = getattr(msg, "content", "")
                if content:
                    recent_responses.append(content[:500])

        # Build context message
        context = f"""Execution Context:
- Original Task: {original_task[:500]}
- Iterations completed: {iteration_count}
- Plan steps: {len(plan_steps)}
- Tools executed: {len(tool_history)}
- Files modified: {len(file_changes)}
- Errors encountered: {len(error_history)}

Tool History (last 5):
{self._format_tool_history(tool_history[-5:])}

Error History:
{self._format_error_history(error_history[-3:])}

Recent Agent Responses:
{chr(10).join(recent_responses[-2:])}

Analyze this execution and provide your critique."""

        try:
            from sepilot.agent.output_validator import OutputValidator

            original_messages = [
                SystemMessage(content=self.CRITIQUE_SYSTEM_PROMPT),
                HumanMessage(content=context)
            ]
            response = self.llm.invoke(original_messages)

            # Validate JSON response with OutputValidator
            raw_content = response.content.strip()
            is_valid, parsed = OutputValidator.validate_json(raw_content)

            if is_valid and parsed is not None:
                return parsed

            # Retry once with correction prompt
            corrected = OutputValidator.retry_with_correction(
                llm=self.llm,
                original_messages=original_messages,
                original_response=raw_content,
                error_desc="Expected valid JSON with keys: goal_achieved, execution_quality, issues_found, etc.",
                max_retries=1,
            )
            if corrected:
                is_valid2, parsed2 = OutputValidator.validate_json(corrected)
                if is_valid2 and parsed2 is not None:
                    return parsed2

            # Last resort: try raw parse
            import json
            content = raw_content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content)

        except Exception as e:
            # Return default critique on error
            return {
                "goal_achieved": False,
                "execution_quality": "unknown",
                "issues_found": [f"Critique generation failed: {str(e)}"],
                "root_causes": [],
                "suggested_improvements": [],
                "recommended_tools": [],
                "confidence_score": 0.5,
                "should_retry": True,
                "retry_strategy": "Continue with current approach"
            }

    def _format_tool_history(self, tools: list) -> str:
        """Format tool history for prompt."""
        if not tools:
            return "No tools executed"

        lines = []
        for tc in tools:
            status = "SUCCESS" if tc.success else f"FAILED: {tc.error or 'unknown'}"
            lines.append(f"  - {tc.tool_name}: {status} ({tc.duration:.2f}s)")
        return "\n".join(lines)

    def _format_error_history(self, errors: list) -> str:
        """Format error history for prompt."""
        if not errors:
            return "No errors"

        def _resolved(err: Any) -> bool:
            if isinstance(err, dict):
                return bool(err.get("resolved", False))
            return bool(getattr(err, "resolved", False))

        def _message(err: Any) -> str:
            if isinstance(err, dict):
                return str(err.get("message", ""))
            return str(getattr(err, "message", ""))

        lines = []
        for err in errors:
            status = "RESOLVED" if _resolved(err) else "UNRESOLVED"
            lines.append(f"  - [{status}] {_message(err)[:100]}")
        return "\n".join(lines)


class StrategyAdjuster:
    """Determines optimal strategy adjustments based on reflection."""

    # Strategy transition recommendations based on patterns
    STRATEGY_TRANSITIONS = {
        AgentStrategy.EXPLORE: {
            "stuck_on_single_tool": AgentStrategy.IMPLEMENT,
            "no_file_changes": AgentStrategy.IMPLEMENT,
            "circular_reasoning": AgentStrategy.IMPLEMENT,
            "read_without_write": AgentStrategy.IMPLEMENT,
        },
        AgentStrategy.IMPLEMENT: {
            "repeating_error": AgentStrategy.DEBUG,
            "tool_failure_cascade": AgentStrategy.EXPLORE,
            "incomplete_analysis": AgentStrategy.EXPLORE,
            "wrong_file_target": AgentStrategy.EXPLORE,
            "shallow_search": AgentStrategy.EXPLORE,
            "overconfident_completion": AgentStrategy.EXPLORE,
        },
        AgentStrategy.DEBUG: {
            "stuck_on_single_tool": AgentStrategy.EXPLORE,
            "circular_reasoning": AgentStrategy.REFACTOR,
            "wrong_file_target": AgentStrategy.EXPLORE,
        },
        AgentStrategy.REFACTOR: {
            "no_file_changes": AgentStrategy.IMPLEMENT,
            "tool_failure_cascade": AgentStrategy.DEBUG,
            "read_without_write": AgentStrategy.IMPLEMENT,
        },
        AgentStrategy.TEST: {
            "repeating_error": AgentStrategy.DEBUG,
            "tool_failure_cascade": AgentStrategy.IMPLEMENT,
        },
        AgentStrategy.DOCUMENT: {
            "no_file_changes": AgentStrategy.IMPLEMENT,
        }
    }

    # Pattern → Mode transition mapping
    PATTERN_MODE_TRANSITIONS: dict[str, AgentMode] = {
        "read_without_write": AgentMode.CODE,
        "no_file_changes": AgentMode.CODE,
        "plan_execution_gap": AgentMode.CODE,
        "overconfident_completion": AgentMode.CODE,
    }

    def suggest_adjustment(
        self,
        current_strategy: AgentStrategy,
        detected_patterns: list[dict[str, Any]],
        critique: dict[str, Any]
    ) -> tuple[AgentStrategy | None, float]:
        """Suggest strategy adjustment based on patterns and critique.

        Args:
            current_strategy: Current agent strategy
            detected_patterns: List of detected failure patterns
            critique: LLM-generated critique

        Returns:
            Tuple of (suggested strategy, confidence adjustment)
        """
        if not detected_patterns:
            # No patterns detected, maintain current strategy
            return None, 0.0

        # Get transitions for current strategy
        transitions = self.STRATEGY_TRANSITIONS.get(current_strategy, {})

        # Find most relevant pattern
        primary_pattern = detected_patterns[0]["pattern"]

        # Check if we have a defined transition
        if primary_pattern in transitions:
            new_strategy = transitions[primary_pattern]
            # Reduce confidence when changing strategy
            confidence_adj = -0.2
            return new_strategy, confidence_adj

        # Use pattern's suggested strategy
        suggested = detected_patterns[0].get("suggested_strategy")
        if suggested and suggested != current_strategy:
            return suggested, -0.15

        # No adjustment needed
        return None, 0.0

    def suggest_mode(
        self,
        current_mode: AgentMode,
        detected_patterns: list[dict[str, Any]],
    ) -> AgentMode | None:
        """Suggest mode transition based on detected failure patterns.

        Args:
            current_mode: Current agent mode
            detected_patterns: List of detected failure patterns

        Returns:
            Suggested mode, or None if no transition needed
        """
        if not detected_patterns or current_mode == AgentMode.AUTO:
            return None

        for pattern in detected_patterns:
            suggested = self.PATTERN_MODE_TRANSITIONS.get(pattern["pattern"])
            if suggested and suggested != current_mode:
                return suggested
        return None


class ReflectionNode:
    """Main reflection node that orchestrates self-critique and strategy adjustment.

    Optimized Claude Code style - minimal LLM calls, heuristic-first approach.
    1. Detect failure patterns (heuristic only - no LLM)
    2. Only call LLM for severe patterns
    3. Quick strategy adjustment based on pattern type
    """

    MAX_REFLECTION_ITERATIONS = _MAX_REFLECTION_ITERATIONS

    # Severe patterns that warrant LLM critique
    SEVERE_PATTERNS = {"tool_failure_cascade", "circular_reasoning", "repeating_error", "overconfident_completion"}

    def __init__(
        self,
        llm: BaseChatModel,
        console: Any | None = None,
        verbose: bool = False,
        logger: Any | None = None
    ):
        """Initialize reflection node."""
        self.llm = llm
        self.console = console
        self.verbose = verbose
        self.logger = logger

        self.pattern_detector = FailurePatternDetector()
        self.critique_generator = CritiqueGenerator(llm)
        self.strategy_adjuster = StrategyAdjuster()

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Execute reflection analysis - optimized for minimal LLM calls.

        Claude Code style: Fast heuristic checks first, LLM only for severe issues.
        """
        reflection_count = state.get("reflection_count", 0)

        # Quick exit if max reflections reached
        if reflection_count >= self.MAX_REFLECTION_ITERATIONS:
            if self.console and self.verbose:
                self.console.print("[dim yellow]⚠️ Max reflections reached, proceeding[/dim yellow]")
            return {
                "reflection_decision": ReflectionDecision.PROCEED.value,
                "reflection_notes": ["Max reflections reached"],
                "reflection_count": reflection_count
            }

        # Step 1: Fast pattern detection (heuristic only - no LLM)
        detected_patterns = self.pattern_detector.detect_patterns(state)

        # No patterns? Quick proceed
        if not detected_patterns:
            return {
                "reflection_decision": ReflectionDecision.PROCEED.value,
                "reflection_notes": ["No issues detected"],
                "reflection_count": reflection_count
            }

        if self.console and self.verbose:
            self.console.print(f"[dim cyan]🔍 {len(detected_patterns)} pattern(s) detected[/dim cyan]")

        # Step 2: Only call LLM for SEVERE patterns (cost optimization)
        critique = {}
        has_severe = any(p["pattern"] in self.SEVERE_PATTERNS for p in detected_patterns)

        if has_severe and reflection_count == 0 and state.get("iteration_count", 0) >= 3:
            # First reflection with severe pattern - use LLM (only after 3+ iterations)
            try:
                critique = self.critique_generator.generate_critique(state)
            except Exception as e:
                if self.console and self.verbose:
                    self.console.print(f"[dim red]📝 Critique error: {e}[/dim red]")
                critique = {"error": str(e)}
            if self.console and self.verbose:
                self.console.print(f"[dim cyan]📝 LLM critique: {critique.get('execution_quality', 'unknown')}[/dim cyan]")
        # Otherwise, use heuristic-only approach (no LLM call)

        # Step 3: Determine strategy adjustment
        current_strategy = state.get("current_strategy", AgentStrategy.EXPLORE)
        new_strategy, confidence_adj = self.strategy_adjuster.suggest_adjustment(
            current_strategy,
            detected_patterns,
            critique
        )

        # Step 3b: Suggest mode transition based on patterns
        current_mode = state.get("current_mode", AgentMode.AUTO)
        suggested_mode = self.strategy_adjuster.suggest_mode(
            current_mode, detected_patterns
        )

        # Step 4: Decide next action
        result = self._decide_action(
            state,
            detected_patterns,
            critique,
            new_strategy,
            confidence_adj
        )
        # Attach mode suggestion to result
        if suggested_mode:
            result.suggested_mode = suggested_mode

        # Build state updates
        updates: dict[str, Any] = {
            "reflection_count": reflection_count + 1,
            "reflection_decision": result.decision.value,
            "reflection_notes": result.insights,
            "confidence_level": max(0.1, min(1.0,
                state.get("confidence_level", 0.8) + result.confidence_adjustment
            ))
        }

        if result.suggested_strategy:
            updates["current_strategy"] = result.suggested_strategy

        if result.should_reset_plan:
            updates["plan_created"] = False
            updates["plan_steps"] = []
            updates["current_plan_step"] = 0

        # Add guidance message based on decision
        guidance_msg = self._create_guidance_message(result, detected_patterns)
        if guidance_msg:
            updates["messages"] = [guidance_msg]

        # Log reflection result
        if self.logger:
            self.logger.log_trace("reflection_result", {
                "decision": result.decision.value,
                "patterns_detected": [p["pattern"] for p in detected_patterns],
                "strategy_change": result.suggested_strategy.value if result.suggested_strategy else None,
                "confidence_adjustment": result.confidence_adjustment
            })

        if self.console and self.verbose:
            decision_icon = {
                ReflectionDecision.REVISE_PLAN: "📋",
                ReflectionDecision.REFINE_STRATEGY: "🔄",
                ReflectionDecision.PROCEED: "➡️",
                ReflectionDecision.ESCALATE: "🆘"
            }.get(result.decision, "❓")
            self.console.print(
                f"[bold cyan]{decision_icon} Reflection Decision: {result.decision.value}[/bold cyan]"
            )

        return updates

    def _decide_action(
        self,
        state: EnhancedAgentState,
        patterns: list[dict[str, Any]],
        critique: dict[str, Any],
        suggested_strategy: AgentStrategy | None,
        confidence_adj: float
    ) -> ReflectionResult:
        """Decide the next action based on analysis.

        Args:
            state: Current state
            patterns: Detected failure patterns
            critique: LLM critique
            suggested_strategy: Suggested new strategy
            confidence_adj: Confidence adjustment

        Returns:
            ReflectionResult with decision and details
        """
        iteration_count = state.get("iteration_count", 0)
        reflection_count = state.get("reflection_count", 0)

        # Build insights list
        insights = []
        root_causes = []

        for p in patterns:
            insights.append(f"Pattern: {p['description']}")
            root_causes.append(p.get("action", "Unknown cause"))

        if critique:
            issues = critique.get("issues_found", [])
            insights.extend([f"Issue: {i}" for i in issues[:3]])
            root_causes.extend(critique.get("root_causes", []))

        # Decision logic
        decision = ReflectionDecision.PROCEED
        should_reset_plan = False

        # High severity patterns require plan revision
        severe_patterns = {"plan_execution_gap", "no_file_changes", "tool_failure_cascade"}
        has_severe = any(p["pattern"] in severe_patterns for p in patterns)

        if has_severe:
            if "plan_execution_gap" in [p["pattern"] for p in patterns]:
                decision = ReflectionDecision.REVISE_PLAN
                should_reset_plan = True
                insights.append("Decision: Reset plan and force execution")
            elif reflection_count >= 2:
                # Severe patterns persist after multiple reflections → escalate
                decision = ReflectionDecision.ESCALATE
                insights.append("Decision: Severe patterns persist despite reflection, escalating")
            else:
                decision = ReflectionDecision.REFINE_STRATEGY
                insights.append("Decision: Adjust strategy and retry")

        # Medium severity patterns suggest strategy refinement
        elif suggested_strategy and suggested_strategy != state.get("current_strategy"):
            decision = ReflectionDecision.REFINE_STRATEGY
            insights.append(f"Decision: Switch to {suggested_strategy.value} strategy")

        # Low confidence from critique suggests more work
        elif critique.get("confidence_score", 1.0) < 0.5:
            if critique.get("should_retry", False):
                decision = ReflectionDecision.REFINE_STRATEGY
                insights.append("Decision: Low confidence, refining approach")
            else:
                decision = ReflectionDecision.PROCEED
                insights.append("Decision: Continue despite low confidence")

        # Too many iterations without progress - escalate
        elif iteration_count > 10 and reflection_count >= 2:
            decision = ReflectionDecision.ESCALATE
            insights.append("Decision: Multiple iterations without success, may need human guidance")

        else:
            decision = ReflectionDecision.PROCEED
            insights.append("Decision: No critical issues, proceeding")

        return ReflectionResult(
            decision=decision,
            critique=critique.get("retry_strategy", "Continue execution"),
            root_causes=root_causes[:5],
            suggested_strategy=suggested_strategy,
            confidence_adjustment=confidence_adj,
            insights=insights,
            should_reset_plan=should_reset_plan,
            recommended_tools=critique.get("recommended_tools", [])
        )

    def _create_guidance_message(
        self,
        result: ReflectionResult,
        patterns: list[dict[str, Any]]
    ) -> SystemMessage | None:
        """Create a guidance message based on reflection result.

        Uses SystemMessage (not HumanMessage) to prevent the verification node
        from confusing reflection guidance with the user's original request.

        Args:
            result: Reflection result
            patterns: Detected patterns

        Returns:
            SystemMessage with guidance or None
        """
        if result.decision == ReflectionDecision.PROCEED:
            # Even for PROCEED, inject lightweight guidance when patterns were detected
            if patterns:
                actions = [p.get("action", "") for p in patterns[:2] if p.get("action")]
                if actions:
                    hint = " | ".join(actions)
                    return SystemMessage(
                        content=f"💡 Reflection hint: {hint}. Adjust your approach accordingly."
                    )
            return None

        guidance_parts = []

        if result.decision == ReflectionDecision.REVISE_PLAN:
            guidance_parts.append(
                "🔄 Self-Reflection: Previous approach was ineffective. "
                "Please create a NEW, more specific plan."
            )
            if result.root_causes:
                guidance_parts.append(f"Root causes identified: {', '.join(result.root_causes[:2])}")

        elif result.decision == ReflectionDecision.REFINE_STRATEGY:
            guidance_parts.append(
                "🔄 Self-Reflection: Adjusting approach."
            )
            if result.suggested_strategy:
                guidance_parts.append(
                    f"New strategy: {result.suggested_strategy.value.upper()}"
                )
            if patterns:
                action = patterns[0].get("action", "")
                if action:
                    guidance_parts.append(f"Recommendation: {action}")

        elif result.decision == ReflectionDecision.ESCALATE:
            guidance_parts.append(
                "🆘 Self-Reflection: Multiple attempts unsuccessful. "
                "Consider asking for clarification or simplifying the task."
            )

        if result.recommended_tools:
            guidance_parts.append(
                f"Recommended tools: {', '.join(result.recommended_tools[:3])}"
            )

        return SystemMessage(content="\n".join(guidance_parts))


def create_reflection_node(
    llm: BaseChatModel,
    console: Any | None = None,
    verbose: bool = False,
    logger: Any | None = None
) -> ReflectionNode:
    """Factory function to create a ReflectionNode instance.

    Args:
        llm: Language model for critique
        console: Rich console
        verbose: Verbose output flag
        logger: Logger instance

    Returns:
        Configured ReflectionNode
    """
    return ReflectionNode(
        llm=llm,
        console=console,
        verbose=verbose,
        logger=logger
    )
