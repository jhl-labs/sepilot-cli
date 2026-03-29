"""Prompt loader for builtin skills — reads .md files from this directory."""

from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    """Load a prompt file by name (without extension)."""
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")
