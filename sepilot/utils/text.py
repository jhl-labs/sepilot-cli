"""Utility helpers for safe text handling."""

from __future__ import annotations

from typing import Any


def sanitize_text(value: Any, *, replacement: str = "?") -> str:
    """Return a UTF-8 safe string by replacing surrogate characters.

    Args:
        value: Input value to sanitize.
        replacement: Replacement character used for undecodable bytes.

    Returns:
        Sanitized string that can be safely logged or displayed.
    """
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)

    # Encode/decode to drop surrogate code points gracefully.
    return text.encode("utf-8", errors="replace").decode(
        "utf-8", errors="replace"
    ).replace("\ufffd", replacement)
