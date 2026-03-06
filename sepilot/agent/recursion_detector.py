"""Recursion and repetition detection for agent loops"""

import json
from collections import Counter, deque
from typing import Any


class RecursionDetector:
    """Detects repetitive tool calls or thought patterns

    Features:
    - Sliding window pattern matching
    - Configurable threshold and window size
    - Action signature generation for comparison
    - Cyclic pattern detection (A → B → C → A → B → C)
    - Tool-only matching for similar operations
    """

    def __init__(self, window_size: int = 15, threshold: int = 3):
        """Initialize recursion detector

        Args:
            window_size: Number of recent actions to keep in memory (default: 15)
            threshold: Number of repetitions to trigger warning (default: 3)
        """
        self.window_size = window_size
        self.threshold = threshold
        self.recent_actions = deque(maxlen=window_size)
        self.recent_tools = deque(maxlen=window_size)  # Track tool names only
        # O(1) counters for repetition detection
        self._action_counts: Counter = Counter()
        self._tool_counts: Counter = Counter()

    def _get_action_signature(self, tool_name: str, args: dict[str, Any]) -> str:
        """Generate a signature for an action

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Signature string for comparison
        """
        # Sort keys for consistent comparison
        args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return f"{tool_name}:{args_str}"

    def _detect_cyclic_pattern(self, sequence: list[str], min_cycle: int = 2, max_cycle: int = 5) -> tuple[bool, int]:
        """Detect cyclic patterns in a sequence.

        Args:
            sequence: List of items to check
            min_cycle: Minimum cycle length to detect
            max_cycle: Maximum cycle length to detect

        Returns:
            Tuple of (cycle_detected, cycle_length)
        """
        if len(sequence) < min_cycle * 2:
            return False, 0

        for cycle_len in range(min_cycle, min(max_cycle + 1, len(sequence) // 2 + 1)):
            # Check if the last 'cycle_len * 2' items form a repeating pattern
            if len(sequence) >= cycle_len * 2:
                pattern = sequence[-cycle_len:]
                prev_pattern = sequence[-cycle_len * 2:-cycle_len]
                if pattern == prev_pattern:
                    # Verify with one more cycle if possible
                    if len(sequence) >= cycle_len * 3:
                        prev_prev = sequence[-cycle_len * 3:-cycle_len * 2]
                        if prev_prev == pattern:
                            return True, cycle_len
                    else:
                        return True, cycle_len

        return False, 0

    def add_action(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Add an action and check for repetition

        Args:
            tool_name: Name of the tool called
            args: Arguments passed to the tool

        Returns:
            True if repetition detected (warning triggered)
        """
        signature = self._get_action_signature(tool_name, args)

        # Evict oldest entry from counters when deque is full
        if len(self.recent_actions) == self.window_size:
            evicted_sig = self.recent_actions[0]
            self._action_counts[evicted_sig] -= 1
            if self._action_counts[evicted_sig] <= 0:
                del self._action_counts[evicted_sig]
        if len(self.recent_tools) == self.window_size:
            evicted_tool = self.recent_tools[0]
            self._tool_counts[evicted_tool] -= 1
            if self._tool_counts[evicted_tool] <= 0:
                del self._tool_counts[evicted_tool]

        self.recent_actions.append(signature)
        self.recent_tools.append(tool_name)
        self._action_counts[signature] += 1
        self._tool_counts[tool_name] += 1

        # Need at least 'threshold' actions to detect repetition
        if len(self.recent_actions) < self.threshold:
            return False

        # Check 1: Exact same action repeated (O(1) lookup)
        if self._action_counts[signature] >= self.threshold:
            return True

        # Check 2: Same tool called repeatedly (O(1) lookup)
        if self._tool_counts[tool_name] >= self.threshold + 2:
            return True

        # Check 3: Cyclic pattern detection (A → B → A → B or A → B → C → A → B → C)
        tools_list = list(self.recent_tools)
        cycle_detected, cycle_len = self._detect_cyclic_pattern(tools_list)
        if cycle_detected:
            return True

        return False

    def get_repetition_info(self) -> dict[str, Any] | None:
        """Get information about detected repetition

        Returns:
            Dictionary with repetition details or None
        """
        if len(self.recent_actions) < self.threshold:
            return None

        # Check exact repetition (O(1))
        last_signature = self.recent_actions[-1]
        exact_count = self._action_counts.get(last_signature, 0)
        if exact_count >= self.threshold:
            tool_name = last_signature.split(':', 1)[0]
            return {
                "tool_name": tool_name,
                "count": exact_count,
                "window_size": len(self.recent_actions),
                "pattern": "exact_repetition"
            }

        # Check tool-only repetition (O(1))
        if self.recent_tools:
            last_tool = self.recent_tools[-1]
            tool_count = self._tool_counts.get(last_tool, 0)
            if tool_count >= self.threshold + 2:
                return {
                    "tool_name": last_tool,
                    "count": tool_count,
                    "window_size": len(self.recent_tools),
                    "pattern": "tool_repetition"
                }

        # Check cyclic pattern
        tools_list = list(self.recent_tools)
        cycle_detected, cycle_len = self._detect_cyclic_pattern(tools_list)
        if cycle_detected:
            cycle_pattern = tools_list[-cycle_len:]
            return {
                "tool_name": " → ".join(cycle_pattern),
                "count": 3,  # At least 3 cycles detected
                "window_size": len(self.recent_tools),
                "pattern": "cyclic_repetition",
                "cycle_length": cycle_len
            }

        return None

    def reset(self):
        """Reset the detector (clear history)"""
        self.recent_actions.clear()
        self.recent_tools.clear()
        self._action_counts.clear()
        self._tool_counts.clear()

    def get_recent_actions(self, n: int = 5) -> list[str]:
        """Get the N most recent actions

        Args:
            n: Number of recent actions to return

        Returns:
            List of action signatures
        """
        return list(self.recent_actions)[-n:]

    def get_recent_tools(self, n: int = 10) -> list[str]:
        """Get the N most recent tool names

        Args:
            n: Number of recent tools to return

        Returns:
            List of tool names
        """
        return list(self.recent_tools)[-n:]


class ThoughtRepetitionDetector:
    """Detects repetitive thought patterns (for LLM reasoning loops)"""

    def __init__(self, similarity_threshold: float = 0.8):
        """Initialize thought repetition detector

        Args:
            similarity_threshold: Threshold for considering thoughts similar (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.recent_thoughts = deque(maxlen=10)

    def _calculate_similarity(self, thought1: str, thought2: str) -> float:
        """Calculate similarity between two thoughts

        Uses simple word-based similarity for now.
        Could be enhanced with embeddings or fuzzy matching.

        Args:
            thought1: First thought text
            thought2: Second thought text

        Returns:
            Similarity score (0.0-1.0)
        """
        # Normalize and tokenize
        words1 = set(thought1.lower().split())
        words2 = set(thought2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def add_thought(self, thought: str) -> bool:
        """Add a thought and check for repetition

        Args:
            thought: The thought text

        Returns:
            True if repetition detected
        """
        # Check similarity with recent thoughts
        for recent_thought in self.recent_thoughts:
            similarity = self._calculate_similarity(thought, recent_thought)
            if similarity >= self.similarity_threshold:
                return True

        self.recent_thoughts.append(thought)
        return False

    def reset(self):
        """Reset the detector"""
        self.recent_thoughts.clear()
