"""Hook System - Pre/post tool execution hooks.

Provides a hook system similar to Claude Code for executing shell commands
before or after tool executions.

Hook configuration is stored in ~/.sepilot/hooks.json or project-local
.sepilot/hooks.json file.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_shell_command(command: str) -> list[str]:
    """Build a platform-safe shell invocation command."""
    if os.name == "nt":
        return ["cmd", "/c", command]

    shell_path = os.environ.get("SHELL")
    if shell_path and Path(shell_path).exists():
        return [shell_path, "-lc", command]
    if Path("/bin/bash").exists():
        return ["/bin/bash", "-lc", command]
    return ["/bin/sh", "-c", command]


@dataclass
class HookResult:
    """Result of a hook execution."""
    success: bool
    output: str = ""
    error: str = ""
    exit_code: int = 0
    blocked: bool = False
    block_reason: str = ""


@dataclass
class Hook:
    """A single hook definition."""
    event: str  # pre_tool, post_tool, pre_commit, post_commit, etc.
    command: str  # Shell command to execute
    tool_filter: list[str] = field(default_factory=list)  # Tool names to filter
    enabled: bool = True
    timeout: int = 30  # seconds
    block_on_failure: bool = False  # Block tool execution if hook fails


class HookManager:
    """Manages hook registration and execution."""

    # Supported hook events
    EVENTS = {
        "pre_tool": "Before any tool execution",
        "post_tool": "After any tool execution",
        "pre_tool_approval": "Before tool approval prompt",
        "post_tool_approval": "After tool approval",
        "pre_bash": "Before bash command execution",
        "post_bash": "After bash command execution",
        "pre_file_write": "Before file write operation",
        "post_file_write": "After file write operation",
        "pre_file_edit": "Before file edit operation",
        "post_file_edit": "After file edit operation",
        "pre_git": "Before git operations",
        "post_git": "After git operations",
        "session_start": "When a new session starts",
        "session_end": "When a session ends",
        "user_prompt_submit": "When user submits a prompt",
    }

    def __init__(self, config_paths: list[str] | None = None):
        """Initialize Hook Manager.

        Args:
            config_paths: List of paths to search for hooks.json
        """
        self.hooks: list[Hook] = []
        self._config_paths = config_paths or [
            str(Path.cwd() / ".sepilot" / "hooks.json"),
            str(Path.home() / ".sepilot" / "hooks.json"),
        ]
        self._load_hooks()

    def _load_hooks(self):
        """Load hooks from configuration files."""
        self.hooks = []

        for config_path in self._config_paths:
            path = Path(config_path)
            if path.exists():
                try:
                    with open(path, encoding='utf-8') as f:
                        config = json.load(f)

                    hooks_data = config.get("hooks", [])
                    for hook_data in hooks_data:
                        hook = Hook(
                            event=hook_data.get("event", ""),
                            command=hook_data.get("command", ""),
                            tool_filter=hook_data.get("tool_filter", []),
                            enabled=hook_data.get("enabled", True),
                            timeout=hook_data.get("timeout", 30),
                            block_on_failure=hook_data.get("block_on_failure", False)
                        )
                        if hook.event and hook.command:
                            self.hooks.append(hook)

                    logger.info(f"Loaded {len(hooks_data)} hooks from {config_path}")

                except Exception as e:
                    logger.error(f"Failed to load hooks from {config_path}: {e}")

    def reload(self):
        """Reload hooks from configuration files."""
        self._load_hooks()

    def execute_hook(
        self,
        event: str,
        tool_name: str | None = None,
        context: dict[str, Any] | None = None
    ) -> list[HookResult]:
        """Execute all hooks for a given event.

        Args:
            event: Hook event name
            tool_name: Name of the tool being executed (for filtering)
            context: Additional context to pass to hooks

        Returns:
            List of HookResult objects
        """
        results = []
        context = context or {}

        # Find matching hooks
        matching_hooks = [
            h for h in self.hooks
            if h.enabled and h.event == event
        ]

        # Filter by tool name if specified
        if tool_name:
            matching_hooks = [
                h for h in matching_hooks
                if not h.tool_filter or tool_name in h.tool_filter
            ]

        for hook in matching_hooks:
            result = self._execute_single_hook(hook, tool_name, context)
            results.append(result)

            # Check if we should block
            if result.blocked:
                logger.warning(f"Hook blocked execution: {result.block_reason}")
                break

        return results

    def _execute_single_hook(
        self,
        hook: Hook,
        tool_name: str | None,
        context: dict[str, Any]
    ) -> HookResult:
        """Execute a single hook.

        Args:
            hook: Hook to execute
            tool_name: Tool name for context
            context: Additional context

        Returns:
            HookResult
        """
        try:
            # Prepare environment variables
            env = os.environ.copy()
            env["SEPILOT_EVENT"] = hook.event
            env["SEPILOT_TOOL"] = tool_name or ""

            # Add context as environment variables
            for key, value in context.items():
                env_key = f"SEPILOT_{key.upper()}"
                env[env_key] = str(value) if not isinstance(value, str) else value

            # Execute command
            result = subprocess.run(
                _build_shell_command(hook.command),
                capture_output=True,
                text=True,
                timeout=hook.timeout,
                env=env,
                cwd=str(Path.cwd())
            )

            success = result.returncode == 0
            blocked = not success and hook.block_on_failure

            return HookResult(
                success=success,
                output=result.stdout,
                error=result.stderr,
                exit_code=result.returncode,
                blocked=blocked,
                block_reason=f"Hook '{hook.command}' failed with exit code {result.returncode}"
                if blocked else ""
            )

        except subprocess.TimeoutExpired:
            return HookResult(
                success=False,
                error=f"Hook timed out after {hook.timeout} seconds",
                exit_code=-1,
                blocked=hook.block_on_failure,
                block_reason=f"Hook '{hook.command}' timed out"
                if hook.block_on_failure else ""
            )

        except Exception as e:
            return HookResult(
                success=False,
                error=str(e),
                exit_code=-1,
                blocked=hook.block_on_failure,
                block_reason=f"Hook execution error: {e}"
                if hook.block_on_failure else ""
            )

    def register_hook(
        self,
        event: str,
        command: str,
        tool_filter: list[str] | None = None,
        timeout: int = 30,
        block_on_failure: bool = False
    ) -> bool:
        """Register a new hook programmatically.

        Args:
            event: Hook event name
            command: Shell command to execute
            tool_filter: Tool names to filter
            timeout: Execution timeout
            block_on_failure: Block if hook fails

        Returns:
            True if registered successfully
        """
        if event not in self.EVENTS:
            logger.error(f"Unknown hook event: {event}")
            return False

        hook = Hook(
            event=event,
            command=command,
            tool_filter=tool_filter or [],
            timeout=timeout,
            block_on_failure=block_on_failure
        )
        self.hooks.append(hook)
        return True

    def unregister_hook(self, event: str, command: str) -> bool:
        """Unregister a hook.

        Args:
            event: Hook event
            command: Hook command

        Returns:
            True if unregistered successfully
        """
        for i, hook in enumerate(self.hooks):
            if hook.event == event and hook.command == command:
                del self.hooks[i]
                return True
        return False

    def list_hooks(self, event: str | None = None) -> list[dict[str, Any]]:
        """List registered hooks.

        Args:
            event: Filter by event (optional)

        Returns:
            List of hook information dictionaries
        """
        hooks = self.hooks if not event else [h for h in self.hooks if h.event == event]

        return [
            {
                "event": h.event,
                "command": h.command,
                "tool_filter": h.tool_filter,
                "enabled": h.enabled,
                "timeout": h.timeout,
                "block_on_failure": h.block_on_failure
            }
            for h in hooks
        ]

    def save_hooks(self, config_path: str | None = None):
        """Save hooks to configuration file.

        Args:
            config_path: Path to save to (default: first config path)
        """
        path = Path(config_path or self._config_paths[0])
        path.parent.mkdir(parents=True, exist_ok=True)

        config = {"hooks": self.list_hooks()}

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved {len(self.hooks)} hooks to {path}")

    def get_available_events(self) -> dict[str, str]:
        """Get available hook events and their descriptions."""
        return self.EVENTS.copy()


# Global singleton instance
_hook_manager: HookManager | None = None


def get_hook_manager() -> HookManager:
    """Get or create the global Hook manager instance."""
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = HookManager()
    return _hook_manager


def run_hooks(
    event: str,
    tool_name: str | None = None,
    context: dict[str, Any] | None = None
) -> list[HookResult]:
    """Convenience function to run hooks for an event.

    Args:
        event: Hook event name
        tool_name: Tool name (optional)
        context: Additional context (optional)

    Returns:
        List of HookResult objects
    """
    manager = get_hook_manager()
    return manager.execute_hook(event, tool_name, context)
