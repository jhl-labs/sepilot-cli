"""Shared command metadata for discoverability surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommandEntry:
    """Command metadata used by completion and palette UIs."""

    name: str
    description: str
    category: str = "General"
    keywords: tuple[str, ...] = field(default_factory=tuple)
    include_in_palette: bool = True


BUILTIN_COMMANDS: tuple[CommandEntry, ...] = (
    CommandEntry("/help", "Show help information", "Help", ("info", "?")),
    CommandEntry("/status", "Show session status", "Help", ("state", "info")),
    CommandEntry("/license", "Show license information", "Help"),
    CommandEntry("/new", "Start new conversation", "Session", ("fresh", "reset")),
    CommandEntry("/resume", "Resume previous conversation", "Session", ("continue", "load")),
    CommandEntry("/session", "Session export/import", "Session", ("save", "load")),
    CommandEntry("/history", "Show conversation history", "History", ("log", "messages")),
    CommandEntry("/rewind", "Go back in conversation", "History", ("back", "undo")),
    CommandEntry("/undo", "Undo last exchange", "History", ("rollback",)),
    CommandEntry("/redo", "Redo undone exchange", "History"),
    CommandEntry("/reset", "Reset session statistics", "History"),
    CommandEntry("/context", "Show context usage", "Context", ("tokens", "memory")),
    CommandEntry("/compact", "Compact conversation context", "Context", ("summarize", "compress")),
    CommandEntry("/clear", "Clear conversation", "Context", ("reset",)),
    CommandEntry("/cost", "Show session cost", "Context", ("price", "tokens")),
    CommandEntry("/model", "Change AI model", "Settings", ("llm", "gpt", "claude")),
    CommandEntry("/theme", "Change UI theme", "Settings", ("colors", "style")),
    CommandEntry("/yolo", "Toggle auto-approve mode", "Settings", ("auto", "approve")),
    CommandEntry("/permissions", "Manage permissions", "Settings", ("security", "allow")),
    CommandEntry("/tools", "List available tools", "Tools", ("commands", "functions")),
    CommandEntry("/mcp", "MCP server management", "Tools", ("servers",)),
    CommandEntry("/rag", "RAG document management", "Tools", ("documents", "knowledge")),
    CommandEntry("/skill", "Manage skills", "Tools", ("prompt", "workflow")),
    CommandEntry("/commands", "Manage custom commands", "Tools", ("templates", "macros")),
    CommandEntry("/graph", "Show LangGraph visualization", "Dev"),
    CommandEntry("/plan", "Switch to PLAN mode", "Modes"),
    CommandEntry("/code", "Switch to CODE mode", "Modes"),
    CommandEntry("/exec", "Switch to EXEC mode", "Modes"),
    CommandEntry("/auto", "Return to AUTO mode", "Modes"),
    CommandEntry("/mode", "Show or change current mode", "Modes"),
    CommandEntry("/container", "Container and Docker commands", "DevOps", ("docker", "compose")),
    CommandEntry("/helm", "Helm chart commands", "DevOps", ("kubernetes",)),
    CommandEntry("/se", "Software engineering helpers", "DevOps", ("devops",)),
    CommandEntry("/gitops", "GitOps and ArgoCD commands", "DevOps", ("argocd", "deploy")),
    CommandEntry("/k8s-health", "Kubernetes health checks", "DevOps", ("cluster", "k8s")),
    CommandEntry("/stats", "Show usage statistics", "UI", ("usage", "metrics")),
    CommandEntry("/clearscreen", "Clear the screen", "UI", ("cls",)),
    CommandEntry("/multiline", "Toggle multiline mode", "UI"),
    CommandEntry("/exit", "Exit SE Pilot", "Exit", ("quit", "bye")),
    CommandEntry("/quit", "Exit SE Pilot", "Exit", ("bye",), include_in_palette=False),
    CommandEntry("/cls", "Clear the screen", "UI", ("clearscreen",), include_in_palette=False),
    CommandEntry("/skills", "Manage skills", "Tools", ("skill",), include_in_palette=False),
)


def iter_palette_commands() -> tuple[CommandEntry, ...]:
    """Return commands that should appear in the command palette."""
    return tuple(entry for entry in BUILTIN_COMMANDS if entry.include_in_palette)


def iter_completion_commands() -> tuple[CommandEntry, ...]:
    """Return commands that should appear in slash-command completion."""
    return BUILTIN_COMMANDS
