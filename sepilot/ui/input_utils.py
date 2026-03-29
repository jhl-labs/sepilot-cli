"""Shared interactive input helpers for REPL command flows."""

from __future__ import annotations

import builtins
import sys
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


def interactive_select(
    items: list[dict[str, str]],
    *,
    title: str = "Select an item:",
    allow_cancel: bool = True,
) -> int | None:
    """Arrow-key interactive selector (Claude Code style).

    Uses ANSI escape sequences and raw terminal input for reliable
    inline rendering without full-screen takeover.

    Args:
        items: List of dicts with 'label' (required) and optional 'description'.
        title: Header text shown above the list.
        allow_cancel: If True, Escape/Ctrl-C returns None.

    Returns:
        Selected index (0-based), or None if cancelled.
    """
    if not items:
        return None

    if not sys.stdin.isatty():
        return _fallback_select(items, title=title)

    return _ansi_select(items, title=title, allow_cancel=allow_cancel)


def _read_key() -> str:
    """Read a single keypress from raw terminal, returning a key name."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        if ch == "\x1b":  # Escape sequence
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return "up"
                if ch3 == "B":
                    return "down"
                return "unknown"
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":  # Ctrl-C
            return "ctrl-c"
        if ch in ("k", "K"):
            return "up"
        if ch in ("j", "J"):
            return "down"
        return "unknown"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ANSI helpers
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_LINE = "\033[2K"


def _ansi_select(
    items: list[dict[str, str]],
    *,
    title: str,
    allow_cancel: bool,
) -> int | None:
    """ANSI escape code based interactive selector."""
    selected = 0
    total_lines = len(items) + 2  # title + blank + items (hint has no trailing \n)

    def render(first: bool = False):
        out = sys.stdout
        if not first:
            # Move cursor up to overwrite previous render
            out.write(f"\033[{total_lines}A")
        # Title
        out.write(f"{_CLEAR_LINE}{_BOLD} {title}{_RESET}\n")
        out.write(f"{_CLEAR_LINE}\n")
        # Items
        for i, item in enumerate(items):
            label = item.get("label", "")
            desc = item.get("description", "")
            out.write(_CLEAR_LINE)
            if i == selected:
                out.write(f" {_CYAN}{_BOLD}❯ {label}{_RESET}")
                if desc:
                    out.write(f"  {_CYAN}{desc}{_RESET}")
            else:
                out.write(f" {_DIM}  {label}")
                if desc:
                    out.write(f"  {desc}")
                out.write(_RESET)
            out.write("\n")
        # Hint
        out.write(f"{_CLEAR_LINE}{_DIM} ↑↓ 이동  Enter 선택  Esc 취소{_RESET}")
        out.flush()

    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()

    try:
        render(first=True)

        while True:
            key = _read_key()
            if key == "up":
                selected = (selected - 1) % len(items)
                render()
            elif key == "down":
                selected = (selected + 1) % len(items)
                render()
            elif key == "enter":
                # Move past the rendered content and clean up
                sys.stdout.write("\n")
                sys.stdout.write(_SHOW_CURSOR)
                sys.stdout.flush()
                return selected
            elif key in ("escape", "ctrl-c") and allow_cancel:
                sys.stdout.write("\n")
                sys.stdout.write(_SHOW_CURSOR)
                sys.stdout.flush()
                return None
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n")
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()
        return None
    finally:
        # Ensure cursor is always restored
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


def _fallback_select(
    items: list[dict[str, str]],
    *,
    title: str = "Select an item:",
) -> int | None:
    """Fallback numbered-list selector for non-TTY environments."""
    builtins.print(f"\n{title}\n")
    for i, item in enumerate(items, 1):
        label = item.get("label", "")
        desc = item.get("description", "")
        line = f"  {i}. {label}"
        if desc:
            line += f"  {desc}"
        builtins.print(line)
    builtins.print()
    try:
        choice = input(f"Select (1-{len(items)}): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(items):
                return idx - 1
        return None
    except (EOFError, KeyboardInterrupt):
        return None
