"""Shared interactive input helpers for REPL command flows."""

from __future__ import annotations

import builtins
from typing import Any

try:
    from prompt_toolkit import prompt as pt_prompt

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    pt_prompt = None


INPUT_CANCELLED = "__INPUT_CANCELLED__"


def prompt_text(
    prompt: str,
    *,
    session: Any | None = None,
    default: str | None = None,
    is_password: bool = False,
) -> str:
    """Prompt for text using the active prompt session when available."""
    try:
        if session is not None:
            kwargs: dict[str, Any] = {"is_password": is_password}
            if default is not None:
                kwargs["default"] = default
            return session.prompt(prompt, **kwargs).strip()

        if HAS_PROMPT_TOOLKIT and pt_prompt is not None:
            kwargs = {"is_password": is_password}
            if default is not None:
                kwargs["default"] = default
            return pt_prompt(prompt, **kwargs).strip()

        raw = input(prompt).strip()
        if raw:
            return raw
        return default or ""
    except (EOFError, KeyboardInterrupt):
        return INPUT_CANCELLED


def prompt_confirm(
    prompt: str,
    *,
    session: Any | None = None,
    default: bool = False,
) -> bool | None:
    """Prompt for yes/no confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"

    while True:
        response = prompt_text(
            f"{prompt} {suffix}: ",
            session=session,
            default="yes" if default else "no",
        )
        if response == INPUT_CANCELLED:
            return None

        normalized = response.strip().lower()
        if normalized in ("y", "yes"):
            return True
        if normalized in ("n", "no"):
            return False

        builtins.print("Please answer yes or no.")
