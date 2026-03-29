"""Helpers for current-execution message boundaries.

These utilities keep multi-turn threads focused on the active request instead
of accidentally reusing the first human message in the thread.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

HUMAN_MESSAGE_ORIGIN_KEY = "sepilot_human_origin"
USER_PROMPT_ORIGIN = "user_prompt"
INTERNAL_PROMPT_ORIGIN = "internal_prompt"


def _message_attr(message: Any, key: str, default: Any = None) -> Any:
    """Read a message attribute from objects or dict-like snapshots."""
    if isinstance(message, Mapping):
        return message.get(key, default)
    return getattr(message, key, default)


def get_message_type(message: Any) -> str:
    """Return the normalized message type for objects or dict snapshots."""
    msg_type = _message_attr(message, "type", "")
    if isinstance(msg_type, str) and msg_type:
        return msg_type
    role = _message_attr(message, "role", "")
    return role if isinstance(role, str) else ""


def get_message_content(message: Any) -> str:
    """Return message content as text for objects or dict snapshots."""
    content = _message_attr(message, "content", "")
    return content if isinstance(content, str) else str(content or "")


def get_message_id(message: Any) -> str | None:
    """Return the persisted message id if present."""
    message_id = _message_attr(message, "id", None)
    return message_id if isinstance(message_id, str) else None


def get_message_tool_calls(message: Any) -> list[Any]:
    """Return tool calls for an AI message from objects or dict snapshots."""
    tool_calls = _message_attr(message, "tool_calls", None)
    return tool_calls if isinstance(tool_calls, list) else []


def _is_human_message(message: Any) -> bool:
    """Support real LangChain messages and lightweight test doubles."""
    if isinstance(message, HumanMessage):
        return True
    return get_message_type(message) in ("human", "user")


def get_human_message_origin(message: Any) -> str | None:
    """Return the recorded origin for a human message, if present."""
    additional_kwargs = _message_attr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        origin = additional_kwargs.get(HUMAN_MESSAGE_ORIGIN_KEY)
        if isinstance(origin, str):
            return origin
    return None


def make_user_prompt_message(content: str, message_id: str | None = None) -> HumanMessage:
    """Create a persisted user-turn boundary message."""
    return HumanMessage(
        content=content,
        id=message_id,
        additional_kwargs={HUMAN_MESSAGE_ORIGIN_KEY: USER_PROMPT_ORIGIN},
    )


def make_internal_human_message(content: str) -> HumanMessage:
    """Create an internal control prompt that should not count as a user turn."""
    return HumanMessage(
        content=content,
        additional_kwargs={HUMAN_MESSAGE_ORIGIN_KEY: INTERNAL_PROMPT_ORIGIN},
    )


def _looks_like_internal_control_prompt(content: str) -> bool:
    """Best-effort legacy fallback for older persisted internal prompts."""
    normalized = (content or "").strip().lower()
    if not normalized:
        return False

    internal_prefixes = (
        "tool execution completed successfully.",
        "your previous response was too brief.",
        "your response described a plan but did not call any tools.",
        "your previous response was empty.",
        "this is a coding/fix task and no substantive project files were modified yet.",
        "progress note:",
        "patch review identified issues:",
        "요청은 실행 작업입니다.",
        "코드가 수정되었습니다.",
        "린트를 수행하세요.",
        "아직 다음 파일이 요구사항을 충족하도록 수정되지 않았습니다:",
        "🛑 stop: 사용자가 도구 실행을 거부했습니다.",
        "🛑 작업이 3회 연속 거부되어 중단합니다.",
    )
    return normalized.startswith(internal_prefixes)


def is_user_turn_boundary_message(message: Any) -> bool:
    """Return True only for real user turn boundaries stored in the thread."""
    if not _is_human_message(message):
        return False

    origin = get_human_message_origin(message)
    if origin == USER_PROMPT_ORIGIN:
        return True
    if origin == INTERNAL_PROMPT_ORIGIN:
        return False

    message_id = get_message_id(message)
    if isinstance(message_id, str) and message_id.startswith("exec_"):
        return True

    return not _looks_like_internal_control_prompt(get_message_content(message))


def is_user_visible_conversation_message(message: Any) -> bool:
    """Return True for end-user-visible conversation messages only."""
    msg_type = get_message_type(message)
    if msg_type in ("ai", "assistant"):
        return bool(get_message_content(message).strip()) or not get_message_tool_calls(message)
    return is_user_turn_boundary_message(message)


def find_execution_boundary(state: Mapping[str, Any]) -> int:
    """Return the index after the current execution prompt.

    The preferred boundary marker is ``_execution_boundary_msg_id``. Older
    states fall back to ``_initial_message_count``.
    """
    messages = state.get("messages", [])
    boundary_id = state.get("_execution_boundary_msg_id")

    if boundary_id:
        for index, message in enumerate(messages):
            if get_message_id(message) == boundary_id:
                return index + 1

    initial_count = state.get("_initial_message_count", 0)
    return initial_count if initial_count > 0 else 0


def get_execution_prompt_message(state: Mapping[str, Any]) -> Any | None:
    """Return the human prompt that started the current execution, if known."""
    messages = state.get("messages", [])
    boundary_id = state.get("_execution_boundary_msg_id")

    if boundary_id:
        for message in messages:
            if get_message_id(message) == boundary_id and _is_human_message(message):
                return message

    initial_count = state.get("_initial_message_count", 0)
    if initial_count > 0:
        prompt_index = initial_count - 1
        if 0 <= prompt_index < len(messages):
            message = messages[prompt_index]
            if _is_human_message(message):
                return message

    for message in reversed(messages):
        if _is_human_message(message):
            return message

    return None


def get_current_execution_messages(state: Mapping[str, Any]) -> list[BaseMessage]:
    """Return only messages created after the current execution prompt."""
    messages = state.get("messages", [])
    boundary_index = find_execution_boundary(state)
    if boundary_index >= len(messages):
        return []
    return list(messages[boundary_index:])


def get_current_user_query(state: Mapping[str, Any]) -> str:
    """Return the active request text for the current execution."""
    prompt_message = get_execution_prompt_message(state)
    if prompt_message:
        return get_message_content(prompt_message)
    return ""
