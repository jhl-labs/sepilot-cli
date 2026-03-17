"""Context management utilities for conversation history compression and summarization.

This module provides:
- Context summarization using LLM
- Message compaction (removing old messages)
- Automatic context window management
- Intelligent semantic-based context management
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from sepilot.config.constants import CONTEXT_PRUNE_SUMMARY_MAX_CHARS

logger = logging.getLogger(__name__)

# Module-level tiktoken encoding cache (avoid re-creating per call)
_tiktoken_encoding = None


def _get_tiktoken_encoding():
    """Get cached tiktoken encoding instance."""
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        try:
            import tiktoken
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass
    return _tiktoken_encoding


class ContextRelevance(Enum):
    """Relevance levels for context items (higher value = more relevant)"""
    MINIMAL = 1   # Can be dropped
    LOW = 2       # Can be summarized
    MEDIUM = 3    # Somewhat relevant
    HIGH = 4      # Very relevant
    CRITICAL = 5  # Must keep


@dataclass
class ContextItem:
    """A single item in the context with metadata"""
    message: BaseMessage
    timestamp: datetime
    relevance: ContextRelevance
    token_count: int
    references: set[str] = field(default_factory=set)
    dependencies: set[str] = field(default_factory=set)
    summary: str | None = None
    can_summarize: bool = True
    can_drop: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self.message).__name__,
            "relevance": self.relevance.name.lower(),  # Use name for string output
            "tokens": self.token_count,
            "references": list(self.references),
            "has_summary": self.summary is not None
        }


@dataclass
class ContextWindow:
    """Sliding window of context with intelligent management"""
    max_tokens: int
    current_tokens: int = 0
    items: list[ContextItem] = field(default_factory=list)
    file_context: dict[str, list[ContextItem]] = field(default_factory=lambda: defaultdict(list))
    symbol_context: dict[str, list[ContextItem]] = field(default_factory=lambda: defaultdict(list))
    task_context: dict[str, list[ContextItem]] = field(default_factory=lambda: defaultdict(list))

    def add_item(self, item: ContextItem):
        """Add item to context window"""
        self.items.append(item)
        self.current_tokens += item.token_count

        for ref in item.references:
            if ref.endswith('.py') or '.' in ref:
                self.file_context[ref].append(item)
            else:
                self.symbol_context[ref].append(item)

    def get_relevant_context(self, query: str, max_tokens: int) -> list[ContextItem]:
        """Get most relevant context for a query"""
        relevant = []
        tokens_used = 0

        entities = self._extract_entities(query)

        for entity in entities:
            if entity in self.file_context:
                relevant.extend(self.file_context[entity])
            if entity in self.symbol_context:
                relevant.extend(self.symbol_context[entity])

        relevant.sort(key=lambda x: (x.relevance.value, x.timestamp), reverse=True)

        selected = []
        for item in relevant:
            if tokens_used + item.token_count <= max_tokens:
                selected.append(item)
                tokens_used += item.token_count

        return selected

    def _extract_entities(self, text: str) -> set[str]:
        """Extract file names, symbols, etc. from text"""
        entities: set[str] = set()

        # File pattern - use non-capturing group to get full match
        file_pattern = r'\b([\w/.-]+\.(?:py|js|ts|java|go|rs|cpp|c|h))\b'
        for match in re.finditer(file_pattern, text):
            entities.add(match.group(1))

        # Function/class pattern
        func_pattern = r'\b(?:def|class|function|const)\s+(\w+)'
        for match in re.finditer(func_pattern, text):
            entities.add(match.group(1))

        return entities


class ContextManager:
    """Manages conversation context to stay within token limits."""

    # Tool outputs larger than this (in characters) are candidates for pruning
    PRUNE_OUTPUT_CHAR_THRESHOLD = 800
    # Pruned summary max length
    PRUNE_SUMMARY_MAX_CHARS = CONTEXT_PRUNE_SUMMARY_MAX_CHARS

    def __init__(
        self,
        max_context_tokens: int = 96000,
        warning_threshold: float = 0.8,
        compact_threshold: float = 0.92,
        predictive_threshold: float = 0.75,
        min_messages_to_keep: int = 15,
    ):
        """Initialize context manager.

        Args:
            max_context_tokens: Maximum tokens allowed in context
            warning_threshold: Percentage at which to warn user
            compact_threshold: Percentage at which to auto-compact
            predictive_threshold: Percentage at which to pre-identify compaction candidates
            min_messages_to_keep: Minimum messages to keep after compaction
        """
        self.max_context_tokens = max_context_tokens
        self.warning_threshold = warning_threshold
        self.compact_threshold = compact_threshold
        self.predictive_threshold = predictive_threshold
        self.min_messages_to_keep = min_messages_to_keep
        # Pre-identified compaction candidates (populated at predictive_threshold)
        self._compaction_candidates: list[int] = []
        # Incremental compaction round tracker
        self._incremental_round: int = 0

    def should_warn(self, current_tokens: int) -> bool:
        """Check if we should warn about context size."""
        usage = current_tokens / self.max_context_tokens
        return usage >= self.warning_threshold

    def should_compact(self, current_tokens: int) -> bool:
        """Check if we should auto-compact context."""
        usage = current_tokens / self.max_context_tokens
        return usage >= self.compact_threshold

    def should_prepare_compaction(self, current_tokens: int) -> bool:
        """Check if we should pre-identify compaction candidates (predictive).

        At predictive_threshold (default 75%), identify messages that would be
        summarized when actual compaction triggers, so compaction is faster.
        """
        usage = current_tokens / self.max_context_tokens
        return usage >= self.predictive_threshold and usage < self.compact_threshold

    def identify_compaction_candidates(
        self,
        messages: list[BaseMessage],
    ) -> list[int]:
        """Pre-identify messages that are candidates for summarization.

        Categorizes messages by relevance and marks LOW/MINIMAL ones
        for immediate summarization and MEDIUM ones for later rounds.

        Args:
            messages: Current message list

        Returns:
            List of message indices that are compaction candidates
        """
        candidates: list[int] = []
        total = len(messages)
        if total == 0:
            return candidates

        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                continue
            relevance = self._assess_message_relevance(msg, i, total)
            if relevance in (ContextRelevance.LOW, ContextRelevance.MINIMAL):
                candidates.append(i)
            elif relevance == ContextRelevance.MEDIUM and self._incremental_round >= 2:
                # MEDIUM messages become candidates from round 2 onwards
                candidates.append(i)

        self._compaction_candidates = candidates
        return candidates

    def _assess_message_relevance(
        self,
        message: BaseMessage,
        position: int,
        total_messages: int,
    ) -> ContextRelevance:
        """Assess the relevance of a single message for compaction decisions.

        Args:
            message: The message to assess
            position: Position index in the message list
            total_messages: Total number of messages

        Returns:
            ContextRelevance level
        """
        if isinstance(message, SystemMessage):
            return ContextRelevance.CRITICAL

        recency = position / total_messages if total_messages > 0 else 0

        # Recent messages (last 20%) are HIGH
        if recency > 0.8:
            return ContextRelevance.HIGH

        content = str(getattr(message, 'content', '') or '')

        # Error-related content is important
        if 'error' in content.lower() or 'exception' in content.lower():
            return ContextRelevance.HIGH

        # Human messages in the recent half are at least MEDIUM
        if isinstance(message, HumanMessage):
            return ContextRelevance.HIGH if recency > 0.5 else ContextRelevance.MEDIUM

        # AI messages with tool calls must stay paired
        if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
            return ContextRelevance.HIGH

        # Tool messages
        if isinstance(message, ToolMessage):
            if 'Error' in content or 'error' in content.lower():
                return ContextRelevance.HIGH
            elif len(content) > 1000:
                return ContextRelevance.MEDIUM
            else:
                return ContextRelevance.LOW

        # Position-based fallback
        if recency > 0.5:
            return ContextRelevance.MEDIUM
        elif recency > 0.2:
            return ContextRelevance.LOW
        else:
            return ContextRelevance.MINIMAL

    def compact_incremental(
        self,
        messages: list[BaseMessage],
        llm: Any = None,
        focus_instruction: str | None = None,
    ) -> list[BaseMessage]:
        """Incrementally compact messages by summarizing oldest low-relevance ones.

        Unlike full compact(), this preserves recent messages entirely and only
        summarizes the oldest 50% of non-system messages, filtering by relevance:
        - CRITICAL/HIGH: Never summarized
        - MEDIUM: Summarized from round 2+
        - LOW/MINIMAL: Always summarized

        Args:
            messages: Current message list
            llm: Optional LLM for summarization (falls back to simple compaction)
            focus_instruction: Optional focus topic for summarization

        Returns:
            Compacted message list with summary replacing old messages
        """
        self._incremental_round += 1

        if len(messages) <= self.min_messages_to_keep:
            return messages

        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(non_system) <= self.min_messages_to_keep:
            return messages

        # Determine split point: oldest 50% are candidates
        split_idx = len(non_system) // 2
        older_half = non_system[:split_idx]
        recent_half = non_system[split_idx:]

        # Filter older messages by relevance
        to_summarize: list[BaseMessage] = []
        to_keep: list[BaseMessage] = []

        for i, msg in enumerate(older_half):
            relevance = self._assess_message_relevance(msg, i, len(non_system))
            if relevance in (ContextRelevance.CRITICAL, ContextRelevance.HIGH):
                to_keep.append(msg)
            elif relevance == ContextRelevance.MEDIUM and self._incremental_round < 2:
                to_keep.append(msg)
            else:
                to_summarize.append(msg)

        if not to_summarize:
            return messages  # Nothing to summarize

        # Create summary
        if llm:
            try:
                conversation_text = self._format_messages_for_summary(to_summarize)

                focus_part = ""
                if focus_instruction:
                    focus_part = f"\n\n특히 다음 주제에 집중하여 요약해주세요: {focus_instruction}"

                summary_prompt = f"""다음은 이전 대화의 일부입니다. 핵심 내용을 간결하게 요약해주세요:

{conversation_text}

요약 시 다음을 포함해주세요:
- 수행된 주요 작업과 결과
- 중요한 결정사항
- 아직 관련 있는 컨텍스트{focus_part}

간결하고 명확하게 요약해주세요."""

                summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
                summary_content = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)

                summary_message = SystemMessage(
                    content=f"[점진적 대화 요약 - {len(to_summarize)}개 메시지, 라운드 {self._incremental_round}]\n{summary_content}"
                )

                return system_messages + [summary_message] + to_keep + recent_half

            except Exception as e:
                logger.warning("Incremental summarization failed: %s", e)
                # Fall through to safe compaction

        # Safe fallback: keep recent messages (same as legacy compact_messages)
        # Never silently drop message content
        keep_recent = max(self.min_messages_to_keep, len(recent_half))
        recent_kept = non_system[-keep_recent:]
        return system_messages + recent_kept

    def get_message_token_breakdown(
        self,
        messages: list[BaseMessage],
    ) -> list[dict[str, Any]]:
        """Get per-message token breakdown sorted by token count (descending).

        Args:
            messages: Message list to analyze

        Returns:
            List of dicts with 'index', 'role', 'tokens', 'preview' keys
        """
        breakdown = []
        for i, msg in enumerate(messages):
            tokens = self._count_message_tokens(msg)
            content = str(getattr(msg, 'content', '') or '')
            role = type(msg).__name__.replace('Message', '')

            breakdown.append({
                'index': i,
                'role': role,
                'tokens': tokens,
                'preview': content[:80].replace('\n', ' '),
            })

        breakdown.sort(key=lambda x: x['tokens'], reverse=True)
        return breakdown

    def get_instructions_token_ratio(
        self,
        messages: list[BaseMessage],
    ) -> dict[str, Any]:
        """Calculate the token ratio of system/instruction messages vs total.

        Args:
            messages: Message list

        Returns:
            Dict with 'instruction_tokens', 'total_tokens', 'ratio'
        """
        instruction_tokens = 0
        total_tokens = 0

        for msg in messages:
            tokens = self._count_message_tokens(msg)
            total_tokens += tokens
            if isinstance(msg, SystemMessage):
                instruction_tokens += tokens

        ratio = instruction_tokens / total_tokens if total_tokens > 0 else 0.0
        return {
            'instruction_tokens': instruction_tokens,
            'total_tokens': total_tokens,
            'ratio': ratio,
        }

    def estimate_remaining_turns(
        self,
        messages: list[BaseMessage],
        current_tokens: int,
    ) -> int:
        """Estimate how many more conversation turns fit in the context window.

        Based on average tokens per turn (human + AI message pair).

        Args:
            messages: Current messages
            current_tokens: Current total tokens

        Returns:
            Estimated number of remaining turns (human+AI pairs)
        """
        remaining = self.max_context_tokens - current_tokens
        if remaining <= 0:
            return 0

        # Count turns (human messages)
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        non_system_tokens = sum(
            self._count_message_tokens(m)
            for m in messages
            if not isinstance(m, SystemMessage)
        )

        if human_count == 0:
            # No history yet, assume ~2000 tokens per turn
            avg_tokens_per_turn = 2000
        else:
            avg_tokens_per_turn = max(non_system_tokens // human_count, 500)

        return max(remaining // avg_tokens_per_turn, 0)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        encoding = _get_tiktoken_encoding()
        if encoding is not None:
            return len(encoding.encode(text))
        # Fallback: ~4 chars per token
        return len(text) // 4

    def _count_message_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a single message.

        Args:
            message: Message to count tokens for

        Returns:
            Token count for the message
        """
        content = getattr(message, 'content', '')
        if not content:
            return 0
        return self._count_tokens(str(content))

    def compact_to_token_limit(
        self,
        messages: list[BaseMessage],
        target_tokens: int,
        min_keep: int = 4,
    ) -> list[BaseMessage]:
        """Compact messages to fit within a token limit (Claude Code style).

        Keeps the most recent messages that fit within the target token limit.
        Always keeps system messages and at least min_keep recent messages.

        Args:
            messages: List of messages to compact
            target_tokens: Target maximum token count
            min_keep: Minimum number of recent messages to keep

        Returns:
            Compacted list of messages fitting within token limit
        """
        if not messages:
            return messages

        # Separate system messages (always keep) and others
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        if not non_system_messages:
            return system_messages

        # Calculate system message tokens (always included)
        system_tokens = sum(self._count_message_tokens(m) for m in system_messages)
        available_tokens = target_tokens - system_tokens

        if available_tokens <= 0:
            # Even system messages exceed limit, just return them
            return system_messages

        # Build list from most recent, tracking tokens
        kept_reversed: list[BaseMessage] = []
        tokens_used = 0

        # Iterate from newest to oldest, append (O(1)), reverse at the end
        for msg in reversed(non_system_messages):
            msg_tokens = self._count_message_tokens(msg)

            # Always keep minimum messages, or if within budget
            if len(kept_reversed) < min_keep or tokens_used + msg_tokens <= available_tokens:
                kept_reversed.append(msg)
                tokens_used += msg_tokens
            else:
                # Budget exceeded, stop adding
                break

        kept_reversed.reverse()
        return system_messages + kept_reversed

    def summarize_to_token_limit(
        self,
        messages: list[BaseMessage],
        llm: Any,
        target_tokens: int,
        focus_instruction: str = None,
    ) -> list[BaseMessage]:
        """Summarize old messages to fit within token limit (Claude Code style).

        Keeps recent messages that fit within half the target, summarizes the rest.

        Args:
            messages: List of messages to process
            llm: LLM instance for summarization
            target_tokens: Target maximum token count
            focus_instruction: Optional focus topic for summarization

        Returns:
            List with summarized history + recent messages within token limit
        """
        if not messages:
            return messages

        # Separate system messages and others
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        if not non_system_messages:
            return system_messages

        # Calculate system message tokens
        system_tokens = sum(self._count_message_tokens(m) for m in system_messages)
        available_tokens = target_tokens - system_tokens

        if available_tokens <= 0:
            return system_messages

        # Reserve half for recent messages, half for summary
        recent_budget = available_tokens // 2
        summary_budget = available_tokens - recent_budget

        # Select recent messages that fit within budget
        recent_messages = []
        recent_tokens = 0

        for msg in reversed(non_system_messages):
            msg_tokens = self._count_message_tokens(msg)
            if recent_tokens + msg_tokens <= recent_budget:
                recent_messages.insert(0, msg)
                recent_tokens += msg_tokens
            else:
                break

        # Messages to summarize (everything not in recent)
        recent_set = {id(m) for m in recent_messages}
        to_summarize = [m for m in non_system_messages if id(m) not in recent_set]

        if not to_summarize:
            # Nothing to summarize
            return system_messages + recent_messages

        # Create summary
        conversation_text = self._format_messages_for_summary(to_summarize)

        focus_part = ""
        if focus_instruction:
            focus_part = f"\n\n특히 다음 주제에 집중하여 요약해주세요: {focus_instruction}"

        summary_prompt = f"""다음은 이전 대화 내용입니다. 핵심 내용을 {summary_budget}토큰 이내로 간결하게 요약해주세요:

{conversation_text}

요약 시 다음을 포함해주세요:
- 사용자가 요청한 주요 작업들
- 완료된 작업과 결과
- 중요한 결정사항이나 컨텍스트
- 아직 진행 중이거나 미완료된 작업{focus_part}

간결하고 명확하게 요약해주세요."""

        try:
            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary_content = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)

            # Truncate summary if it exceeds budget
            summary_tokens = self._count_tokens(summary_content)
            if summary_tokens > summary_budget:
                # Rough truncation by character ratio
                char_limit = int(len(summary_content) * (summary_budget / summary_tokens))
                summary_content = summary_content[:char_limit] + "\n[요약 일부 생략됨]"

            summary_message = SystemMessage(
                content=f"[대화 요약 - {len(to_summarize)} 메시지, ~{summary_tokens} 토큰]\n{summary_content}"
            )

            return system_messages + [summary_message] + recent_messages

        except Exception as e:
            print(f"Warning: Summarization failed ({e}), using simple compaction")
            return self.compact_to_token_limit(messages, target_tokens)

    def prune_tool_outputs(
        self,
        messages: list[BaseMessage],
        keep_recent: int = 6,
    ) -> list[BaseMessage]:
        """Prune large tool outputs from older messages to save tokens.

        Replaces the content of old, large ToolMessage outputs with a short
        summary line, preserving tool_call_id and structure so the conversation
        remains valid for the LLM.

        Recent messages (within *keep_recent* of the end) are never pruned.

        Args:
            messages: Full message list
            keep_recent: Number of recent messages to protect from pruning

        Returns:
            New message list with pruned tool outputs
        """
        if not messages:
            return messages

        threshold = self.PRUNE_OUTPUT_CHAR_THRESHOLD
        summary_max = self.PRUNE_SUMMARY_MAX_CHARS
        protected_start = max(0, len(messages) - keep_recent)
        result: list[BaseMessage] = []
        changed = False

        for idx, msg in enumerate(messages):
            if idx >= protected_start or not isinstance(msg, ToolMessage):
                result.append(msg)
                continue

            content = getattr(msg, "content", "") or ""
            if len(content) <= threshold:
                result.append(msg)
                continue

            # Build a short summary: first meaningful line + size info
            first_line = content.split("\n", 1)[0][:summary_max]
            # Use cheap char-based estimate (avoid tiktoken encode on large content)
            estimated_tokens = len(content) // 4
            pruned_content = (
                f"[pruned ~{estimated_tokens} tokens] {first_line}"
            )

            # Create replacement ToolMessage preserving structure
            pruned_msg = ToolMessage(
                content=pruned_content,
                tool_call_id=getattr(msg, "tool_call_id", ""),
                name=getattr(msg, "name", None),
            )
            result.append(pruned_msg)
            changed = True

        return result if changed else messages

    def compact_messages(
        self,
        messages: list[BaseMessage],
        keep_recent: int = None,
    ) -> list[BaseMessage]:
        """Compact messages by keeping only recent ones (legacy method).

        Args:
            messages: List of messages to compact
            keep_recent: Number of recent messages to keep

        Returns:
            Compacted list of messages
        """
        if keep_recent is None:
            keep_recent = self.min_messages_to_keep

        if len(messages) <= keep_recent:
            return messages

        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        recent_messages = non_system_messages[-keep_recent:]

        return system_messages + recent_messages

    def summarize_messages(
        self,
        messages: list[BaseMessage],
        llm: Any,
        keep_recent: int = None,
        focus_instruction: str = None,
    ) -> list[BaseMessage]:
        """Summarize old messages using LLM (Claude Code style with focus option).

        Args:
            messages: List of messages to summarize
            llm: LLM instance for summarization
            keep_recent: Number of recent messages to keep unsummarized
            focus_instruction: Optional focus topic for summarization (e.g., "authentication logic")

        Returns:
            List with summarized history + recent messages
        """
        if keep_recent is None:
            keep_recent = self.min_messages_to_keep

        if len(messages) <= keep_recent:
            return messages

        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(non_system_messages) <= keep_recent:
            return messages

        to_summarize = non_system_messages[:-keep_recent]
        recent_messages = non_system_messages[-keep_recent:]

        conversation_text = self._format_messages_for_summary(to_summarize)

        # Build focus instruction part
        focus_part = ""
        if focus_instruction:
            focus_part = f"\n\n특히 다음 주제에 집중하여 요약해주세요: {focus_instruction}"

        summary_prompt = f"""다음은 이전 대화 내용입니다. 핵심 내용을 간결하게 요약해주세요:

{conversation_text}

요약 시 다음을 포함해주세요:
- 사용자가 요청한 주요 작업들
- 완료된 작업과 결과
- 중요한 결정사항이나 컨텍스트
- 아직 진행 중이거나 미완료된 작업{focus_part}

간결하고 명확하게 요약해주세요."""

        try:
            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary_content = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)

            summary_message = SystemMessage(
                content=f"[대화 요약 - {len(to_summarize)} 메시지]\n{summary_content}"
            )

            return system_messages + [summary_message] + recent_messages

        except Exception as e:
            print(f"Warning: Summarization failed ({e}), using simple compaction")
            return self.compact_messages(messages, keep_recent)

    def _format_messages_for_summary(self, messages: list[BaseMessage]) -> str:
        """Format messages for summarization prompt."""
        lines = []
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                role = "사용자"
            elif isinstance(msg, AIMessage):
                role = "AI"
            else:
                role = "시스템"

            raw_content = getattr(msg, 'content', '') or ''
            content = str(raw_content)[:200]
            lines.append(f"{i}. [{role}] {content}")

        return "\n".join(lines)

    def clear_context(self) -> list[BaseMessage]:
        """Clear all context, returning empty list."""
        return []

    def get_context_stats(self, messages: list[BaseMessage], current_tokens: int) -> dict[str, Any]:
        """Get statistics about current context.

        Args:
            messages: Current message list
            current_tokens: Current token count

        Returns:
            Dictionary with context statistics
        """
        usage_percent = (current_tokens / self.max_context_tokens) * 100

        return {
            'message_count': len(messages),
            'total_tokens': current_tokens,
            'max_tokens': self.max_context_tokens,
            'usage_percent': usage_percent,
            'tokens_remaining': self.max_context_tokens - current_tokens,
            'should_warn': self.should_warn(current_tokens),
            'should_compact': self.should_compact(current_tokens),
        }


class IntelligentContextManager(ContextManager):
    """Advanced context management with semantic understanding.

    Extends ContextManager with semantic relevance scoring,
    multi-window context management, and intelligent compaction.
    """

    def __init__(self, max_tokens: int = 96000):
        super().__init__(
            max_context_tokens=max_tokens,
            warning_threshold=0.8,
            compact_threshold=0.92,
            min_messages_to_keep=15
        )
        self.max_tokens = max_tokens
        self.critical_threshold = 0.97

        # Context windows for different scopes
        self.global_context = ContextWindow(max_tokens=max_tokens // 2)
        self.local_context = ContextWindow(max_tokens=max_tokens // 4)
        self.working_context = ContextWindow(max_tokens=max_tokens // 4)

        # (summary_cache and relevance_cache removed — unused)

        # Statistics
        self.compaction_count = 0
        self.tokens_saved = 0

    def manage_context(self, state: Any) -> dict[str, Any]:
        """Main context management function"""
        messages = state.get("messages", [])
        current_tokens = self._estimate_tokens(messages)

        warning_level = self.max_tokens * self.warning_threshold
        compact_level = self.max_tokens * self.compact_threshold
        critical_level = self.max_tokens * self.critical_threshold

        updates: dict[str, Any] = {}

        if current_tokens >= critical_level:
            updates = self._emergency_compact(state)
        elif current_tokens >= compact_level:
            updates = self._intelligent_compact(state)
        elif current_tokens >= warning_level:
            updates["context_warning"] = f"Context usage: {current_tokens}/{self.max_tokens} tokens"

        updates["context_stats"] = {
            "current_tokens": current_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": (current_tokens / self.max_tokens) * 100,
            "compaction_count": self.compaction_count,
            "tokens_saved": self.tokens_saved
        }

        return updates

    def _intelligent_compact(self, state: Any) -> dict[str, Any]:
        """Intelligent context compaction preserving important information"""
        messages = state.get("messages", [])
        if len(messages) < 10:
            return {}

        context_items = self._analyze_messages(messages, state)

        by_relevance = defaultdict(list)
        for item in context_items:
            by_relevance[item.relevance].append(item)

        new_messages = []

        new_messages.extend([item.message for item in by_relevance[ContextRelevance.CRITICAL]])
        new_messages.extend([item.message for item in by_relevance[ContextRelevance.HIGH]])

        medium_summaries = self._summarize_group(by_relevance[ContextRelevance.MEDIUM])
        if medium_summaries:
            new_messages.extend(medium_summaries)

        low_summary = self._aggregate_low_relevance(by_relevance[ContextRelevance.LOW])
        if low_summary:
            new_messages.append(low_summary)

        dropped_count = len(by_relevance[ContextRelevance.MINIMAL])

        meta_message = SystemMessage(content=f"""
📊 Context Compaction Applied:
- Original messages: {len(messages)}
- Compacted to: {len(new_messages)}
- Dropped: {dropped_count} low-relevance messages
- Tokens saved: ~{self._estimate_tokens(messages) - self._estimate_tokens(new_messages)}
""")
        new_messages.append(meta_message)

        self.compaction_count += 1
        self.tokens_saved += self._estimate_tokens(messages) - self._estimate_tokens(new_messages)

        # Use RemoveMessage for add_messages reducer compatibility
        msg_ops: list[BaseMessage] = []
        new_ids = {id(m) for m in new_messages}
        for msg in messages:
            msg_id = getattr(msg, 'id', None)
            if id(msg) not in new_ids and msg_id:
                msg_ops.append(RemoveMessage(id=msg_id))
        old_ids = {id(m) for m in messages}
        for msg in new_messages:
            if id(msg) not in old_ids:
                msg_ops.append(msg)

        return {"messages": msg_ops, "context_compacted": True}

    def _emergency_compact(self, state: Any) -> dict[str, Any]:
        """Emergency compaction - aggressive reduction"""
        messages = state.get("messages", [])

        # Collect leading system messages (before first non-system message)
        leading_system = []
        first_non_system_idx = 0
        for i, m in enumerate(messages):
            if isinstance(m, SystemMessage):
                leading_system.append(m)
                first_non_system_idx = i + 1
            else:
                break

        recent_messages = messages[-10:]
        # Deduplicate: remove leading system messages that also appear in recent
        recent_ids = {id(m) for m in recent_messages}
        leading_system = [m for m in leading_system if id(m) not in recent_ids]

        to_summarize = messages[first_non_system_idx:-10]
        if to_summarize:
            summary = self._create_emergency_summary(to_summarize)
            new_messages = leading_system + [summary] + recent_messages
        else:
            new_messages = leading_system + recent_messages

        self.compaction_count += 1
        self.tokens_saved += self._estimate_tokens(messages) - self._estimate_tokens(new_messages)

        # Use RemoveMessage for add_messages reducer compatibility
        msg_ops: list[BaseMessage] = []
        new_ids = {id(m) for m in new_messages}
        for msg in messages:
            msg_id = getattr(msg, 'id', None)
            if id(msg) not in new_ids and msg_id:
                msg_ops.append(RemoveMessage(id=msg_id))
        old_ids = {id(m) for m in messages}
        for msg in new_messages:
            if id(msg) not in old_ids:
                msg_ops.append(msg)

        return {
            "messages": msg_ops,
            "emergency_compaction": True,
            "original_count": len(messages),
            "new_count": len(new_messages)
        }

    def _analyze_messages(self, messages: list[BaseMessage], state: Any) -> list[ContextItem]:
        """Analyze messages and create context items"""
        context_items = []
        current_task = state.get("current_task")
        recent_tools = state.get("tool_call_history", [])[-10:]
        recent_files = set()

        for tool_call in recent_tools:
            if hasattr(tool_call, 'args'):
                args = tool_call.args if isinstance(tool_call.args, dict) else {}
                if 'file_path' in args:
                    recent_files.add(args['file_path'])
                if 'path' in args:
                    recent_files.add(args['path'])

        for i, message in enumerate(messages):
            relevance = self._calculate_relevance(message, i, len(messages), recent_files, current_task)
            references = self._extract_references(message)

            item = ContextItem(
                message=message,
                timestamp=datetime.now() - timedelta(minutes=len(messages) - i),
                relevance=relevance,
                token_count=self._estimate_message_tokens(message),
                references=references,
                can_summarize=not isinstance(message, SystemMessage),
                can_drop=relevance in [ContextRelevance.LOW, ContextRelevance.MINIMAL]
            )

            context_items.append(item)

        return context_items

    def _calculate_relevance(
        self,
        message: BaseMessage,
        position: int,
        total_messages: int,
        recent_files: set[str],
        current_task: Any
    ) -> ContextRelevance:
        """Calculate relevance of a message"""
        if isinstance(message, SystemMessage):
            return ContextRelevance.CRITICAL

        recency_factor = position / total_messages
        if recency_factor > 0.9:
            return ContextRelevance.HIGH

        content = getattr(message, 'content', '')

        if 'error' in content.lower() or 'exception' in content.lower():
            return ContextRelevance.HIGH

        if any(file in content for file in recent_files):
            return ContextRelevance.HIGH

        if isinstance(message, ToolMessage):
            if 'Error' in content:
                return ContextRelevance.HIGH
            elif len(content) > 1000:
                return ContextRelevance.MEDIUM
            else:
                return ContextRelevance.LOW

        if isinstance(message, HumanMessage):
            return ContextRelevance.HIGH if recency_factor > 0.5 else ContextRelevance.MEDIUM

        if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
            # AIMessage with tool_calls must stay paired with corresponding ToolMessages
            # Mark as HIGH to prevent orphaned ToolMessage errors in LLM API calls
            return ContextRelevance.HIGH

        if recency_factor > 0.7:
            return ContextRelevance.MEDIUM
        elif recency_factor > 0.3:
            return ContextRelevance.LOW
        else:
            return ContextRelevance.MINIMAL

    def _extract_references(self, message: BaseMessage) -> set[str]:
        """Extract file and symbol references from a message"""
        references = set()
        content = getattr(message, 'content', '')

        if not content:
            return references

        file_pattern = r'[\w/]+\.\w+'
        references.update(re.findall(file_pattern, content))

        symbol_pattern = r'\b(class|def|function|const)\s+(\w+)'
        for match in re.finditer(symbol_pattern, content):
            references.add(match.group(2))

        if hasattr(message, 'tool_calls'):
            for tool_call in message.tool_calls:
                if isinstance(tool_call, dict):
                    args = tool_call.get('args', {})
                    if 'file_path' in args:
                        references.add(args['file_path'])
                    if 'symbol' in args:
                        references.add(args['symbol'])

        return references

    def _summarize_group(self, items: list[ContextItem]) -> list[BaseMessage]:
        """Summarize a group of context items"""
        if not items:
            return []

        by_type = defaultdict(list)
        for item in items:
            msg_type = type(item.message).__name__
            by_type[msg_type].append(item)

        summaries = []

        if 'ToolMessage' in by_type:
            tool_summary = self._summarize_tool_messages(by_type['ToolMessage'])
            if tool_summary:
                summaries.append(tool_summary)

        if 'AIMessage' in by_type:
            ai_summary = self._summarize_ai_messages(by_type['AIMessage'])
            if ai_summary:
                summaries.append(ai_summary)

        return summaries

    def _summarize_tool_messages(self, items: list[ContextItem]) -> SystemMessage | None:
        """Summarize multiple tool messages"""
        if not items:
            return None

        tool_results = []
        for item in items:
            content = item.message.content[:200]
            tool_results.append(f"- {content}")

        summary = f"""
📦 Tool Execution Summary ({len(items)} tools):
{chr(10).join(tool_results[:5])}
{"... and " + str(len(items) - 5) + " more" if len(items) > 5 else ""}
"""
        return SystemMessage(content=summary)

    def _summarize_ai_messages(self, items: list[ContextItem]) -> SystemMessage | None:
        """Summarize multiple AI messages"""
        if not items:
            return None

        key_points = []
        for item in items:
            content = item.message.content
            if any(phrase in content.lower() for phrase in ['decided', 'found', 'created', 'modified']):
                key_points.append(content[:150])

        if key_points:
            summary = f"""
🤖 AI Actions Summary:
{chr(10).join(f"- {p}" for p in key_points[:5])}
"""
            return SystemMessage(content=summary)

        return None

    def _aggregate_low_relevance(self, items: list[ContextItem]) -> SystemMessage | None:
        """Aggregate low relevance items into a single summary"""
        if not items:
            return None

        stats = {
            "total": len(items),
            "tools": sum(1 for i in items if isinstance(i.message, ToolMessage)),
            "ai": sum(1 for i in items if isinstance(i.message, AIMessage)),
            "human": sum(1 for i in items if isinstance(i.message, HumanMessage))
        }

        return SystemMessage(content=f"""
📝 Context Summary: {stats['total']} low-relevance messages aggregated
- Tool results: {stats['tools']}
- AI responses: {stats['ai']}
- User messages: {stats['human']}
""")

    def _create_emergency_summary(self, messages: list[BaseMessage]) -> SystemMessage:
        """Create emergency summary of messages"""
        msg_types = defaultdict(int)
        key_files = set()
        key_operations = []

        for msg in messages:
            msg_types[type(msg).__name__] += 1

            content = getattr(msg, 'content', '')
            if 'file' in content.lower():
                files = re.findall(r'[\w/]+\.\w+', content)
                key_files.update(files[:3])

            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls[:3]:
                    if isinstance(tc, dict):
                        key_operations.append(tc.get('name', 'unknown'))

        return SystemMessage(content=f"""
⚠️ EMERGENCY CONTEXT COMPACTION
Summarized {len(messages)} messages:
- Message types: {dict(msg_types)}
- Key files: {', '.join(list(key_files)[:5])}
- Key operations: {', '.join(key_operations[:5])}
[Details omitted to save tokens]
""")

    def _estimate_tokens(self, messages: list[BaseMessage]) -> int:
        """Estimate token count for messages using tiktoken or fallback.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Total token count for all messages
        """
        encoding = _get_tiktoken_encoding()
        if encoding is not None:
            total = 0
            for msg in messages:
                content = getattr(msg, 'content', '')
                if content:
                    # Add message overhead (role, formatting)
                    total += len(encoding.encode(str(content))) + 4
            return total
        # Fallback: ~4 chars per token
        total_chars = sum(len(getattr(m, 'content', '')) for m in messages)
        return total_chars // 4

    def _estimate_message_tokens(self, message: BaseMessage) -> int:
        """Estimate tokens for a single message using tiktoken or fallback.

        Args:
            message: Message to count tokens for

        Returns:
            Token count for the message
        """
        content = getattr(message, 'content', '')
        if not content:
            return 4  # Minimum for message overhead

        encoding = _get_tiktoken_encoding()
        if encoding is not None:
            # Add message overhead (role, formatting tokens)
            return len(encoding.encode(str(content))) + 4
        # Fallback: ~4 chars per token + overhead
        return (len(content) // 4) + 4

    def get_context_analytics(self) -> dict[str, Any]:
        """Get analytics about context usage"""
        return {
            "compaction_count": self.compaction_count,
            "total_tokens_saved": self.tokens_saved,
            "efficiency_ratio": self.tokens_saved / (self.max_tokens * max(1, self.compaction_count)),
            "global_context_size": self.global_context.current_tokens,
            "local_context_size": self.local_context.current_tokens,
            "working_context_size": self.working_context.current_tokens
        }

    def suggest_context_optimization(self, state: Any) -> list[str]:
        """Suggest optimizations for context usage"""
        suggestions = []
        messages = state.get("messages", [])

        tool_calls = [m for m in messages if isinstance(m, ToolMessage)]
        if len(tool_calls) > 20:
            suggestions.append("Consider caching frequently accessed file contents")

        large_messages = [m for m in messages if self._estimate_message_tokens(m) > 1000]
        if len(large_messages) > 5:
            suggestions.append("Consider summarizing large tool outputs immediately")

        topics = self._detect_topics(messages)
        if len(topics) > 3:
            suggestions.append("Consider splitting into separate conversations for different topics")

        return suggestions

    def _detect_topics(self, messages: list[BaseMessage]) -> set[str]:
        """Detect distinct topics in conversation"""
        topics = set()

        for msg in messages:
            content = getattr(msg, 'content', '').lower()

            if 'test' in content or 'pytest' in content:
                topics.add('testing')
            if 'bug' in content or 'error' in content or 'fix' in content:
                topics.add('debugging')
            if 'create' in content or 'implement' in content:
                topics.add('implementation')
            if 'refactor' in content or 'improve' in content:
                topics.add('refactoring')
            if 'document' in content or 'readme' in content:
                topics.add('documentation')

        return topics


class AdaptiveContextNode:
    """LangGraph node for adaptive context management"""

    def __init__(self, max_tokens: int = 96000):
        self.manager = IntelligentContextManager(max_tokens)

    def __call__(self, state: Any) -> dict[str, Any]:
        """Execute context management"""
        updates = self.manager.manage_context(state)

        suggestions = self.manager.suggest_context_optimization(state)

        if suggestions:
            suggestion_msg = SystemMessage(content=f"""
💡 Context Optimization Suggestions:
{chr(10).join(f"- {s}" for s in suggestions)}
""")
            if "messages" in updates:
                updates["messages"].append(suggestion_msg)
            else:
                updates["messages"] = [suggestion_msg]

        updates["context_analytics"] = self.manager.get_context_analytics()

        return updates


class SmartContextSelector:
    """Smart context selection using semantic scoring, dependencies, and symbol analysis.

    This is the main integration point for Claude Code-style context selection.
    Combines:
    - Semantic relevance scoring (embedding-based)
    - Related file discovery (imports/references)
    - Symbol context gathering (definitions/usages)
    """

    def __init__(
        self,
        project_root: str,
        max_context_tokens: int = 50000,
    ):
        """Initialize smart context selector.

        Args:
            project_root: Project root directory
            max_context_tokens: Maximum tokens for context
        """
        from pathlib import Path

        self.project_root = Path(project_root).resolve()
        self.max_context_tokens = max_context_tokens

        # Lazy initialization
        self._indexer = None
        self._lsp = None
        self._scorer = None
        self._related_finder = None
        self._symbol_gatherer = None

    def _ensure_initialized(self) -> None:
        """Lazy initialize components."""
        if self._scorer is not None:
            return

        try:
            from .semantic_scorer import SemanticScorer
            self._scorer = SemanticScorer()
        except ImportError:
            self._scorer = None

        try:
            from .related_files import RelatedFileFinder
            self._related_finder = RelatedFileFinder(self.project_root)
        except ImportError:
            self._related_finder = None

        try:
            from .symbol_context import SymbolContextGatherer
            self._symbol_gatherer = SymbolContextGatherer(self.project_root)
        except ImportError:
            self._symbol_gatherer = None

    def select_context(
        self,
        query: str,
        current_files: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Select relevant context for a query.

        Args:
            query: The user query
            current_files: Files currently being worked on
            max_tokens: Maximum tokens for context

        Returns:
            List of context items with content, file_path, and score
        """
        self._ensure_initialized()
        max_tokens = max_tokens or self.max_context_tokens

        context_items: list[dict[str, Any]] = []
        tokens_used = 0

        # 1. Extract entities from query
        entities = self._extract_entities(query)
        symbols = entities.get("symbols", [])
        files = entities.get("files", [])

        # Add current files to context
        if current_files:
            files.extend(current_files)
        files = list(set(files))

        # 2. Find related files through dependencies
        if self._related_finder and files:
            try:
                related = self._related_finder.find_related(files, symbols, max_results=5)
                for rel in related:
                    if tokens_used >= max_tokens:
                        break
                    content = self._read_file_content(rel.file_path)
                    if content:
                        tokens = self._estimate_tokens(content)
                        if tokens_used + tokens <= max_tokens:
                            context_items.append({
                                "content": content,
                                "file_path": rel.file_path,
                                "score": rel.relevance_score,
                                "reason": rel.reason,
                                "source": "related",
                            })
                            tokens_used += tokens
            except Exception:
                pass

        # 3. Gather symbol context
        if self._symbol_gatherer and symbols:
            try:
                for symbol in symbols[:5]:  # Limit symbols
                    if tokens_used >= max_tokens:
                        break
                    ctx = self._symbol_gatherer.gather(
                        symbol, include_source=True, max_source_lines=30
                    )
                    if ctx:
                        content = ctx.format(include_source=True)
                        tokens = self._estimate_tokens(content)
                        if tokens_used + tokens <= max_tokens:
                            context_items.append({
                                "content": content,
                                "file_path": ctx.file_path,
                                "score": 0.8,
                                "reason": f"symbol definition: {symbol}",
                                "source": "symbol",
                            })
                            tokens_used += tokens
            except Exception:
                pass

        # 4. Add current files if not already included
        included_paths = {item["file_path"] for item in context_items}
        for file_path in files:
            if file_path in included_paths:
                continue
            if tokens_used >= max_tokens:
                break
            content = self._read_file_content(file_path)
            if content:
                tokens = self._estimate_tokens(content)
                if tokens_used + tokens <= max_tokens:
                    context_items.append({
                        "content": content,
                        "file_path": file_path,
                        "score": 0.9,
                        "reason": "current file",
                        "source": "current",
                    })
                    tokens_used += tokens

        # 5. Score and rank by semantic relevance
        if self._scorer and context_items:
            for item in context_items:
                semantic_score = self._scorer.score(query, item["content"])
                # Combine with original score
                item["score"] = (item["score"] + semantic_score) / 2

        # Sort by combined score
        context_items.sort(key=lambda x: x["score"], reverse=True)

        return context_items

    def format_context_for_llm(
        self, context_items: list[dict[str, Any]], max_tokens: int | None = None
    ) -> str:
        """Format selected context for LLM consumption.

        Args:
            context_items: Context items from select_context
            max_tokens: Maximum tokens to include

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self.max_context_tokens
        tokens_used = 0
        parts = []

        parts.append("# Relevant Context\n")
        tokens_used += 10

        for item in context_items:
            if tokens_used >= max_tokens:
                break

            content = item["content"]
            tokens = self._estimate_tokens(content)

            if tokens_used + tokens > max_tokens:
                # Truncate content to fit
                remaining = max_tokens - tokens_used - 100  # Reserve for formatting
                if remaining > 100:
                    content = self._truncate_to_tokens(content, remaining)
                else:
                    continue

            file_path = item.get("file_path", "unknown")
            reason = item.get("reason", "")
            source = item.get("source", "")

            part = f"\n## {file_path}\n"
            if reason:
                part += f"_({reason})_\n"
            part += f"```\n{content}\n```\n"

            parts.append(part)
            tokens_used += self._estimate_tokens(part)

        return "".join(parts)

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract code-related entities from text."""
        entities = {
            "files": [],
            "symbols": [],
        }

        # File paths
        file_pattern = r"[\w./\\-]+\.\w{1,5}"
        entities["files"] = list(set(re.findall(file_pattern, text)))

        # Symbols (CamelCase or snake_case)
        camel_case = r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"
        snake_case = r"\b[a-z]+(?:_[a-z]+)+\b"
        all_symbols = re.findall(camel_case, text) + re.findall(snake_case, text)
        entities["symbols"] = list(set(all_symbols))

        return entities

    def _read_file_content(self, file_path: str, max_lines: int = 500) -> str | None:
        """Read file content with limit."""
        from pathlib import Path

        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.project_root / path
            if not path.exists():
                return None

            content = path.read_text(encoding="utf-8")
            lines = content.split("\n")
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines.append(f"\n... ({len(lines)} lines truncated)")
            return "\n".join(lines)
        except Exception:
            return None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        encoding = _get_tiktoken_encoding()
        if encoding is not None:
            return len(encoding.encode(text))
        return len(text) // 4

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        encoding = _get_tiktoken_encoding()
        if encoding is not None:
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated = encoding.decode(tokens[:max_tokens])
            return truncated + "\n... (truncated)"
        else:
            char_limit = max_tokens * 4
            if len(text) <= char_limit:
                return text
            return text[:char_limit] + "\n... (truncated)"


def get_smart_context_selector(project_root: str) -> SmartContextSelector:
    """Get a SmartContextSelector for a project.

    Args:
        project_root: Project root directory

    Returns:
        SmartContextSelector instance
    """
    return SmartContextSelector(project_root)
