"""Prompt triage helpers for deciding whether to use LangGraph tools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Only used for quick fallback checks
GREETING_KEYWORDS = [
    "hi", "hello", "hey", "good morning", "good evening",
    "thanks", "thank you", "bye", "goodbye",
    "안녕", "안녕하세요", "반가워", "고마워", "감사",
]

# Fast-path patterns to skip LLM call for obvious classifications
SIMPLE_FAST_PATTERNS = [
    re.compile(r"^(read|show|cat|open)\s+\S+\.\w+", re.IGNORECASE),
    re.compile(r"^git\s+(status|log|diff|branch)\b", re.IGNORECASE),
    re.compile(r"^(ls|pwd)\b", re.IGNORECASE),
    re.compile(r"^(읽어|보여|열어)\s+", re.IGNORECASE),
]
COMPLEX_FAST_PATTERNS = [
    re.compile(r"\b(fix|implement|refactor|create|build)\b.*\b(bug|feature|module|class|test)\b", re.IGNORECASE),
    re.compile(r"(수정|구현|리팩토링|생성)"),
    # Project/code analysis requires tools (file exploration, reading, structure analysis)
    re.compile(r"(프로젝트|코드|코드베이스|소스|디렉토리|파일).*(분석|탐색|조사|살펴|확인|파악)", re.IGNORECASE),
    re.compile(r"(분석|탐색|조사|살펴|확인|파악).*(프로젝트|코드|코드베이스|소스|디렉토리|파일)", re.IGNORECASE),
    re.compile(r"\b(analyze|explore|investigate|examine)\b.*\b(project|code|codebase|source|directory|repo)\b", re.IGNORECASE),
    re.compile(r"\b(project|code|codebase|source|repo)\b.*\b(analy|explor|investigat|examin|overview|structure)\b", re.IGNORECASE),
]


@dataclass
class PromptTriageResult:
    """Result of classifying whether a prompt needs LangGraph tools."""

    decision: str  # "direct_response" or "graph"
    reason: str
    complexity: str = "simple"  # "simple" or "complex" (populated by unified triage)
    strategy: str = "explore"  # AgentStrategy value: explore, implement, debug, etc.


# ============================================================================
# Strategy Classification
# ============================================================================

# Keyword → strategy mapping (priority order: first match wins)
_STRATEGY_KEYWORDS = [
    # debug
    ("debug", "debug"),
    ("bug", "debug"),
    ("error", "debug"),
    ("fix", "debug"),
    ("crash", "debug"),
    ("traceback", "debug"),
    ("exception", "debug"),
    ("broken", "debug"),
    ("버그", "debug"),
    ("오류", "debug"),
    ("에러", "debug"),
    ("고쳐", "debug"),
    ("고치", "debug"),
    ("수정", "debug"),
    # implement
    ("implement", "implement"),
    ("create", "implement"),
    ("build", "implement"),
    ("develop", "implement"),
    ("add feature", "implement"),
    ("new feature", "implement"),
    ("구현", "implement"),
    ("만들", "implement"),
    ("추가", "implement"),
    ("개발", "implement"),
    # refactor
    ("refactor", "refactor"),
    ("restructure", "refactor"),
    ("reorganize", "refactor"),
    ("clean up", "refactor"),
    ("리팩토", "refactor"),
    ("리팩터", "refactor"),
    ("정리", "refactor"),
    # test
    ("test", "test"),
    ("testing", "test"),
    ("unittest", "test"),
    ("pytest", "test"),
    ("테스트", "test"),
    # document
    ("document", "document"),
    ("documentation", "document"),
    ("docstring", "document"),
    ("readme", "document"),
    ("comment", "document"),
    ("문서", "document"),
    # explore (lowest priority - general analysis/info requests)
    ("explain", "explore"),
    ("analyze", "explore"),
    ("describe", "explore"),
    ("show", "explore"),
    ("list", "explore"),
    ("what is", "explore"),
    ("how does", "explore"),
    ("introduce", "explore"),
    ("architecture", "explore"),
    ("overview", "explore"),
    ("structure", "explore"),
    ("설명", "explore"),
    ("분석", "explore"),
    ("소개", "explore"),
    ("알려", "explore"),
    ("보여", "explore"),
    ("아키텍처", "explore"),
    ("구조", "explore"),
]


def classify_strategy(prompt: str) -> str:
    """Classify user prompt into an AgentStrategy value using keyword matching.

    Priority order: debug > implement > refactor > test > document > explore

    Args:
        prompt: Raw user prompt text.

    Returns:
        Strategy string matching AgentStrategy enum values.
    """
    if not prompt:
        return "explore"

    normalized = prompt.strip().lower()

    # Priority order: check high-priority strategies first
    priority_order = ["debug", "implement", "refactor", "test", "document", "explore"]
    matched = {}
    for keyword, strategy in _STRATEGY_KEYWORDS:
        if keyword in normalized and strategy not in matched:
            matched[strategy] = True

    for strategy in priority_order:
        if strategy in matched:
            return strategy

    return "explore"


def triage_prompt_with_llm(
    prompt: str,
    llm: Any,
    config: dict[str, Any] | None = None
) -> PromptTriageResult:
    """Use a single LLM call to classify intent AND complexity.

    Combines triage (direct vs tools) and complexity (simple vs complex)
    into one round-trip to save latency.

    Args:
        prompt: Raw user prompt.
        llm: LangChain LLM instance (without tools bound).
        config: Optional configuration.

    Returns:
        PromptTriageResult with decision, reason, and complexity.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # LLM-based path: both decision and strategy come from the LLM output.
    default_strategy = "explore"

    if not prompt or not prompt.strip():
        return PromptTriageResult(decision="direct_response", reason="empty_prompt", complexity="simple", strategy=default_strategy)

    # Respect auto_finish config: if disabled, never return direct_response
    effective_config = config or {}
    if effective_config and not effective_config.get("enabled", True):
        return PromptTriageResult(decision="graph", reason="auto_finish_disabled", complexity="complex", strategy=default_strategy)

    # Fast-path: skip LLM for obvious classifications (performance optimization)
    text = prompt.strip().lower()

    # Greeting fast-path (skip LLM for simple greetings)
    if len(text) < 50:
        for keyword in GREETING_KEYWORDS:
            if text == keyword or text.startswith(keyword + " ") or text.startswith(keyword + "!"):
                return PromptTriageResult(
                    decision="direct_response",
                    reason="greeting",
                    complexity="simple",
                    strategy="explore"
                )

    # Simple task fast-path (read file, git status, etc.)
    stripped = prompt.strip()
    for pattern in SIMPLE_FAST_PATTERNS:
        if pattern.search(stripped):
            return PromptTriageResult(
                decision="graph",
                reason="fast_path:simple",
                complexity="simple",
                strategy="explore"
            )

    # Complex task fast-path (fix bug, implement feature, etc.)
    for pattern in COMPLEX_FAST_PATTERNS:
        if pattern.search(stripped):
            return PromptTriageResult(
                decision="graph",
                reason="fast_path:complex",
                complexity="complex",
                strategy=classify_strategy(prompt)  # Use keyword-based strategy for fast-path
            )

    # Unified triage + complexity in a single LLM call
    unified_prompt = """You are a request classifier for a coding assistant with file/code/shell tools.
Classify the user request and return EXACTLY three lines in this format:

DECISION: <direct|simple|complex>
STRATEGY: <explore|implement|debug|refactor|test|document>
REASON: <short reason>

Decision rules:
- direct: only greeting/chitchat or general knowledge answerable without workspace access
- simple: needs tools but straightforward (1-2 tool calls), including single command execution/status checks
- complex: multi-step implementation/bugfix/refactor/analysis (3+ tool calls)

Strategy rules:
- explore: read/analyze/inspect
- implement: create/add new behavior
- debug: diagnose/fix issues
- refactor: restructure without feature change
- test: run commands/tests/verification/system checks
- document: docs/comments/readme
"""

    try:
        response = llm.invoke([
            SystemMessage(content=unified_prompt),
            HumanMessage(content=prompt[:1000])
        ])

        if not response or not getattr(response, 'content', None):
            return PromptTriageResult(decision="graph", reason="llm_empty_response", complexity="complex")

        answer = response.content.strip().lower()

        if not answer:
            return PromptTriageResult(decision="graph", reason="llm_empty_content", complexity="complex", strategy=default_strategy)

        decision_match = re.search(r"^\s*decision\s*:\s*(direct|simple|complex)\s*$", answer, flags=re.IGNORECASE | re.MULTILINE)
        strategy_match = re.search(
            r"^\s*strategy\s*:\s*(explore|implement|debug|refactor|test|document)\s*$",
            answer,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        decision_value = decision_match.group(1).lower() if decision_match else ""

        # Strategy: prefer LLM output, fallback to keyword-based classification
        if strategy_match:
            strategy_value = strategy_match.group(1).lower()
        else:
            # Fallback: use keyword-based strategy classification if LLM didn't provide it
            strategy_value = classify_strategy(prompt) or default_strategy

        if decision_value == "direct":
            return PromptTriageResult(
                decision="direct_response",
                reason="llm_unified:direct",
                complexity="simple",
                strategy=strategy_value,
            )
        if decision_value == "simple":
            return PromptTriageResult(
                decision="graph",
                reason="llm_unified:simple",
                complexity="simple",
                strategy=strategy_value,
            )
        if decision_value == "complex":
            return PromptTriageResult(
                decision="graph",
                reason="llm_unified:complex",
                complexity="complex",
                strategy=strategy_value,
            )

        # Fallback for malformed responses: keep tool path and default strategy.
        return PromptTriageResult(
            decision="graph",
            reason="llm_unified:parse_fallback",
            complexity="complex",
            strategy=strategy_value or default_strategy,
        )

    except Exception as e:
        return PromptTriageResult(decision="graph", reason=f"llm_error:{str(e)[:50]}", complexity="complex", strategy=default_strategy)


def triage_prompt_for_tools(
    prompt: str,
    config: dict[str, Any] | None = None,
    llm: Any = None
) -> PromptTriageResult:
    """Classify whether a prompt requires tool usage or can be answered directly.

    If LLM is provided, uses LLM-based classification (more accurate).
    Otherwise, falls back to simple heuristics.

    Args:
        prompt: Raw user prompt.
        config: auto_finish configuration block from prompt profile.
        llm: Optional LLM instance for intelligent classification.

    Returns:
        PromptTriageResult indicating routing decision and reason.
    """
    # If LLM is provided, use LLM-based triage
    if llm is not None:
        return triage_prompt_with_llm(prompt, llm, config)

    # Fallback: Simple heuristics (only for cases without LLM)
    effective_config = config or {}
    strategy = classify_strategy(prompt)

    if not prompt or not prompt.strip():
        return PromptTriageResult(decision="direct_response", reason="empty_prompt", complexity="simple", strategy=strategy)

    if effective_config and not effective_config.get("enabled", True):
        return PromptTriageResult(decision="graph", reason="auto_finish_disabled", complexity="complex", strategy=strategy)

    text = prompt.strip()
    normalized = text.lower()

    # Quick greeting check (exact word match, consistent with LLM path)
    if len(text) < 50:
        for keyword in GREETING_KEYWORDS:
            if normalized == keyword or normalized.startswith(keyword + " ") or normalized.startswith(keyword + "!"):
                return PromptTriageResult(decision="direct_response", reason=f"greeting:{keyword}", complexity="simple", strategy=strategy)

    # Code blocks or diffs always need tools (likely complex)
    if "```" in text or "diff --git" in normalized:
        return PromptTriageResult(decision="graph", reason="code_block_detected", complexity="complex", strategy=strategy)

    # File references need tools
    if any(ext in normalized for ext in [".py", ".js", ".ts", ".json", ".yaml", ".md"]):
        return PromptTriageResult(decision="graph", reason="file_reference", complexity="simple", strategy=strategy)

    # Default: use tools (safer for coding assistant)
    return PromptTriageResult(decision="graph", reason="default", complexity="complex", strategy=strategy)
