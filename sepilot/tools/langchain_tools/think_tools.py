"""Think/Scratchpad tool for LangChain agent.

Allows the agent to persist intermediate reasoning to disk and state,
keeping the message context lean while maintaining continuity via
the system-prompt scratchpad summary.
"""

import json
import os
from datetime import datetime, timezone

from langchain_core.tools import tool

from sepilot.agent.tool_tracker import queue_delta

VALID_CATEGORIES = frozenset({"analysis", "plan", "progress", "finding", "general"})


def _scratchpad_dir() -> str:
    """Return (and create if needed) the scratchpad directory under .sepilot/."""
    d = os.path.join(os.getcwd(), ".sepilot", "scratchpad")
    os.makedirs(d, exist_ok=True)
    return d


@tool
def think(thought: str, category: str = "general") -> str:
    """Record an intermediate thought to the scratchpad.

    Use this tool to persist your reasoning, plans, findings, or progress
    notes. Thoughts are stored on disk and injected into the system prompt
    so you can maintain context across turns without bloating messages.

    Args:
        thought: The thought or note to record (be concise).
        category: One of: analysis, plan, progress, finding, general.

    Returns:
        Short confirmation string.
    """
    if category not in VALID_CATEGORIES:
        category = "general"

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "category": category,
        "content": thought.strip()[:500],
        "timestamp": timestamp,
    }

    # Persist to disk as JSONL
    scratchpad_path = os.path.join(_scratchpad_dir(), "scratchpad.jsonl")
    with open(scratchpad_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Push to state via tool_tracker
    queue_delta({"scratchpad_entries": [entry]})

    return "Noted."


def get_scratchpad_summary(entries: list[dict], max_chars: int = 2000) -> str:
    """Build a compact summary string from scratchpad entries for system prompt injection.

    Groups the most recent entries by category and truncates to *max_chars*.
    """
    if not entries:
        return ""

    by_cat: dict[str, list[dict]] = {}
    for e in entries:
        cat = e.get("category", "general")
        by_cat.setdefault(cat, []).append(e)

    lines: list[str] = []
    for cat in ("plan", "progress", "finding", "analysis", "general"):
        items = by_cat.get(cat)
        if not items:
            continue
        lines.append(f"[{cat}]")
        for item in items[-5:]:  # last 5 per category
            lines.append(f"  - {item['content']}")

    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars - 3] + "..."
    return summary
