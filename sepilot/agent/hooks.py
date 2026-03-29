"""Hook System - Pre/post tool execution hooks.

Provides a hook system similar to Claude Code for executing shell commands,
HTTP requests, or prompt-based validation before or after tool executions.

Hook configuration is stored in ~/.sepilot/hooks.json or project-local
.sepilot/hooks.json file, or in settings.json under the "hooks" key.

Enhanced features (Claude Code-style):
- Regex matcher for tool name filtering
- HTTP hook type (POST to webhook URL)
- Exit code semantics: 0=allow, 2=block, other=warn
- JSON stdin/stdout protocol for hooks
- Async hook support
- settings.json integration
"""

import ipaddress
import json
import logging
import os
import re
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    """A single hook definition.

    Enhanced with Claude Code-style features:
    - matcher: regex pattern for tool name matching (e.g., "Edit|Write")
    - hook_type: "command" (shell), "http" (webhook), "prompt" (LLM)
    - http_url: URL for HTTP hook type
    - http_headers: headers for HTTP hook type
    - is_async: run hook in background without blocking
    """
    event: str  # pre_tool, post_tool, pre_commit, post_commit, etc.
    command: str  # Shell command to execute
    tool_filter: list[str] = field(default_factory=list)  # Tool names (exact match)
    matcher: str = ""  # Regex pattern for tool name matching (Claude Code style)
    hook_type: str = "command"  # "command" | "http" | "prompt"
    http_url: str = ""  # URL for HTTP hook type
    http_headers: dict[str, str] = field(default_factory=dict)  # HTTP headers
    enabled: bool = True
    timeout: int = 30  # seconds
    block_on_failure: bool = False  # Block tool execution if hook fails
    is_async: bool = False  # Run in background without blocking
    priority: int = 0  # Higher priority hooks execute first


_SSRF_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


class HookManager:
    """Manages hook registration and execution."""

    _executor: ThreadPoolExecutor | None = None

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
        "notification": "When a notification is sent to the user",
        "pre_compact": "Before conversation compaction",
        "subagent_tool_use": "When a subagent uses a tool",
        "stop": "When the agent completes a response",
    }

    def __init__(self, config_paths: list[str] | None = None):
        """Initialize Hook Manager.

        Args:
            config_paths: List of paths to search for hooks.json and settings.json
        """
        self.hooks: list[Hook] = []
        self._config_paths = config_paths or [
            str(Path.cwd() / ".sepilot" / "hooks.json"),
            str(Path.home() / ".sepilot" / "hooks.json"),
        ]
        # Also check settings.json files (Claude Code style)
        self._settings_paths = [
            str(Path.cwd() / ".sepilot" / "settings.json"),
            str(Path.cwd() / ".claude" / "settings.json"),
            str(Path.home() / ".sepilot" / "settings.json"),
            str(Path.home() / ".claude" / "settings.json"),
        ]
        self._load_hooks()

    def _load_hooks(self):
        """Load hooks from configuration files.

        Supports two formats:
        1. Legacy hooks.json: {"hooks": [{"event": "...", "command": "..."}]}
        2. Claude Code settings.json: {"hooks": {"PreToolUse": [{"matcher": "...", "hooks": [...]}]}}
        """
        self.hooks = []

        # Load from hooks.json (legacy format)
        for config_path in self._config_paths:
            path = Path(config_path)
            if path.exists():
                try:
                    with open(path, encoding='utf-8') as f:
                        config = json.load(f)

                    hooks_data = config.get("hooks", [])
                    if isinstance(hooks_data, list):
                        for hook_data in hooks_data:
                            hook = self._parse_hook_entry(hook_data)
                            if hook:
                                self.hooks.append(hook)
                        logger.info(f"Loaded {len(hooks_data)} hooks from {config_path}")

                except Exception as e:
                    logger.error(f"Failed to load hooks from {config_path}: {e}")

        # Load from settings.json (Claude Code format)
        for settings_path in self._settings_paths:
            path = Path(settings_path)
            if path.exists():
                try:
                    with open(path, encoding='utf-8') as f:
                        config = json.load(f)

                    hooks_config = config.get("hooks", {})
                    if isinstance(hooks_config, dict):
                        self._load_settings_hooks(hooks_config, settings_path)

                except Exception as e:
                    logger.error(f"Failed to load hooks from {settings_path}: {e}")

    def _load_settings_hooks(self, hooks_config: dict, source: str):
        """Load hooks from Claude Code-style settings.json format.

        Format:
        {
            "PreToolUse": [
                {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": "prettier"}]}
            ],
            "PostToolUse": [...]
        }
        """
        # Map Claude Code event names to our event names
        event_map = {
            # Claude Code PascalCase → internal snake_case
            "PreToolUse": "pre_tool",
            "PostToolUse": "post_tool",
            "SessionStart": "session_start",
            "SessionEnd": "session_end",
            "UserPromptSubmit": "user_prompt_submit",
            "Notification": "notification",
            "PreCompact": "pre_compact",
            "SubagentToolUse": "subagent_tool_use",
            "Stop": "stop",
            # Direct snake_case mappings also supported
            "pre_tool": "pre_tool",
            "post_tool": "post_tool",
            "pre_bash": "pre_bash",
            "post_bash": "post_bash",
            "pre_file_write": "pre_file_write",
            "post_file_write": "post_file_write",
            "pre_file_edit": "pre_file_edit",
            "post_file_edit": "post_file_edit",
            "pre_git": "pre_git",
            "post_git": "post_git",
        }

        count = 0
        for event_name, hook_groups in hooks_config.items():
            mapped_event = event_map.get(event_name, event_name)

            if not isinstance(hook_groups, list):
                continue

            for group in hook_groups:
                matcher = group.get("matcher", "")
                inner_hooks = group.get("hooks", [])

                for hook_def in inner_hooks:
                    hook_type = hook_def.get("type", "command")
                    command = hook_def.get("command", "")
                    http_url = hook_def.get("url", "")
                    http_headers = hook_def.get("headers", {})
                    is_async = hook_def.get("async", False)
                    timeout = hook_def.get("timeout", 30)
                    priority = hook_def.get("priority", 0)

                    hook = Hook(
                        event=mapped_event,
                        command=command,
                        matcher=matcher,
                        hook_type=hook_type,
                        http_url=http_url,
                        http_headers=http_headers,
                        timeout=timeout,
                        block_on_failure=(hook_type == "command"),
                        is_async=is_async,
                        priority=priority,
                    )

                    if hook.event and (hook.command or hook.http_url):
                        self.hooks.append(hook)
                        count += 1

        if count:
            logger.info(f"Loaded {count} hooks from {source}")

    def _parse_hook_entry(self, hook_data: dict) -> Hook | None:
        """Parse a single hook entry from legacy or new format."""
        event = hook_data.get("event", "")
        command = hook_data.get("command", "")

        if not event or not command:
            return None

        return Hook(
            event=event,
            command=command,
            tool_filter=hook_data.get("tool_filter", []),
            matcher=hook_data.get("matcher", ""),
            hook_type=hook_data.get("hook_type", hook_data.get("type", "command")),
            http_url=hook_data.get("http_url", hook_data.get("url", "")),
            http_headers=hook_data.get("http_headers", hook_data.get("headers", {})),
            enabled=hook_data.get("enabled", True),
            timeout=hook_data.get("timeout", 30),
            block_on_failure=hook_data.get("block_on_failure", False),
            is_async=hook_data.get("is_async", hook_data.get("async", False)),
            priority=hook_data.get("priority", 0),
        )

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

        # Filter by tool name using both exact match and regex matcher
        if tool_name:
            filtered = []
            for h in matching_hooks:
                # Check regex matcher first (Claude Code style)
                if h.matcher:
                    try:
                        if re.match(h.matcher, tool_name):
                            filtered.append(h)
                            continue
                    except re.error:
                        logger.warning(f"Invalid matcher regex: {h.matcher}")
                # Fall back to exact tool_filter list
                if h.tool_filter:
                    if tool_name in h.tool_filter:
                        filtered.append(h)
                elif not h.matcher:
                    # No filter at all = matches everything
                    filtered.append(h)
            matching_hooks = filtered

        # Sort by priority (higher priority first)
        matching_hooks.sort(key=lambda h: h.priority, reverse=True)

        for hook in matching_hooks:
            # Async hooks: fire-and-forget in background thread
            if hook.is_async:
                if HookManager._executor is None:
                    HookManager._executor = ThreadPoolExecutor(max_workers=4)
                HookManager._executor.submit(
                    self._execute_single_hook, hook, tool_name, context
                )
                results.append(HookResult(
                    success=True, output="async hook dispatched"
                ))
                continue

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
        """Execute a single hook based on its type.

        Hook types:
        - "command": Shell command execution (default)
        - "http": HTTP POST to webhook URL

        Exit code semantics (Claude Code style):
        - 0: Allow (success, stderr ignored)
        - 2: Block (stderr fed back to Claude as feedback)
        - other: Allow but log warning (stderr logged)
        """
        if hook.hook_type == "http":
            return self._execute_http_hook(hook, tool_name, context)

        return self._execute_command_hook(hook, tool_name, context)

    def _execute_command_hook(
        self,
        hook: Hook,
        tool_name: str | None,
        context: dict[str, Any]
    ) -> HookResult:
        """Execute a shell command hook with JSON stdin protocol."""
        try:
            # Prepare environment variables
            env = os.environ.copy()
            env["SEPILOT_EVENT"] = hook.event
            env["SEPILOT_TOOL"] = tool_name or ""

            # Add context as environment variables
            for key, value in context.items():
                # Sanitize key: only alphanumeric and underscore
                sanitized_key = re.sub(r'[^A-Za-z0-9_]', '_', key.upper())
                env_key = f"SEPILOT_{sanitized_key}"
                # Serialize complex types as JSON
                if isinstance(value, (dict, list)):
                    env[env_key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, str):
                    env[env_key] = value.replace('\x00', '')
                else:
                    env[env_key] = str(value)

            # Prepare JSON stdin (Claude Code style)
            stdin_data = json.dumps({
                "session_id": context.get("session_id", ""),
                "cwd": str(Path.cwd()),
                "hook_event_name": hook.event,
                "tool_name": tool_name or "",
                "tool_input": context.get("tool_input", {}),
            })

            # Execute command
            result = subprocess.run(
                _build_shell_command(hook.command),
                capture_output=True,
                text=True,
                input=stdin_data,
                timeout=hook.timeout,
                env=env,
                cwd=str(Path.cwd())
            )

            # Claude Code exit code semantics
            exit_code = result.returncode
            if exit_code == 0:
                # Success: allow, ignore stderr
                return HookResult(
                    success=True,
                    output=result.stdout,
                    exit_code=0,
                )
            elif exit_code == 2:
                # Block: stderr is feedback to agent
                block_reason = result.stderr.strip() or f"Blocked by hook: {hook.command}"
                return HookResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    exit_code=2,
                    blocked=True,
                    block_reason=block_reason,
                )
            else:
                # Other: allow but log warning
                if result.stderr:
                    logger.warning(f"Hook '{hook.command}' stderr: {result.stderr[:200]}")
                return HookResult(
                    success=True,
                    output=result.stdout,
                    error=result.stderr,
                    exit_code=exit_code,
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

    @staticmethod
    def _validate_url_ssrf(url: str) -> str | None:
        """Validate URL against SSRF attacks.

        Returns None if safe, or an error message string if blocked.
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return f"invalid URL: {url}"

        # Only allow http/https schemes
        if parsed.scheme not in ("http", "https"):
            return f"blocked scheme: {parsed.scheme}"

        hostname = parsed.hostname
        if not hostname:
            return "missing hostname"

        # Resolve hostname to IP addresses and check each
        try:
            addrinfos = socket.getaddrinfo(hostname, parsed.port or 80)
        except socket.gaierror:
            return f"DNS resolution failed: {hostname}"

        for _family, _type, _proto, _canonname, sockaddr in addrinfos:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in _SSRF_BLOCKED_NETWORKS:
                if ip in network:
                    return f"blocked internal IP: {ip} ({hostname})"

        return None

    def _execute_http_hook(
        self,
        hook: Hook,
        tool_name: str | None,
        context: dict[str, Any]
    ) -> HookResult:
        """Execute an HTTP webhook hook.

        Sends POST request with hook event data as JSON body.
        Response codes: 200=allow, 4xx=block, 5xx=error.
        """
        if not hook.http_url:
            return HookResult(
                success=False,
                error="HTTP hook has no URL configured",
                exit_code=-1,
            )

        # SSRF validation
        ssrf_error = self._validate_url_ssrf(hook.http_url)
        if ssrf_error:
            return HookResult(
                success=False,
                error=f"SSRF blocked: {ssrf_error}",
                exit_code=-1,
            )

        try:
            import urllib.error
            import urllib.request

            payload = json.dumps({
                "hook_event_name": hook.event,
                "tool_name": tool_name or "",
                "tool_input": context.get("tool_input", {}),
                "cwd": str(Path.cwd()),
            }).encode("utf-8")

            # Expand env vars in headers
            headers = {"Content-Type": "application/json"}
            for key, value in hook.http_headers.items():
                # Support ${ENV_VAR} in header values
                expanded = os.path.expandvars(value)
                headers[key] = expanded

            req = urllib.request.Request(
                hook.http_url,
                data=payload,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=hook.timeout) as resp:
                response_body = resp.read().decode("utf-8")
                status = resp.status

            if 200 <= status < 300:
                return HookResult(success=True, output=response_body, exit_code=0)
            elif 400 <= status < 500:
                return HookResult(
                    success=False, output=response_body, exit_code=2,
                    blocked=True, block_reason=response_body[:200],
                )
            else:
                return HookResult(
                    success=False, error=f"HTTP {status}: {response_body[:200]}",
                    exit_code=status,
                )

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200] if e.fp else ""
            blocked = 400 <= e.code < 500
            return HookResult(
                success=False,
                error=f"HTTP {e.code}: {body}",
                exit_code=e.code,
                blocked=blocked,
                block_reason=body if blocked else "",
            )

        except Exception as e:
            return HookResult(
                success=False,
                error=f"HTTP hook error: {e}",
                exit_code=-1,
            )

    def register_hook(
        self,
        event: str,
        command: str = "",
        tool_filter: list[str] | None = None,
        matcher: str = "",
        hook_type: str = "command",
        http_url: str = "",
        http_headers: dict[str, str] | None = None,
        timeout: int = 30,
        block_on_failure: bool = False,
        is_async: bool = False,
        priority: int = 0,
    ) -> bool:
        """Register a new hook programmatically.

        Args:
            event: Hook event name
            command: Shell command to execute (for command type)
            tool_filter: Tool names to filter (exact match)
            matcher: Regex pattern for tool name matching (Claude Code style)
            hook_type: "command" | "http" | "prompt"
            http_url: URL for HTTP hook type
            http_headers: Headers for HTTP hook type
            timeout: Execution timeout
            block_on_failure: Block if hook fails
            is_async: Run hook in background without blocking
            priority: Execution priority (higher = runs first)

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
            matcher=matcher,
            hook_type=hook_type,
            http_url=http_url,
            http_headers=http_headers or {},
            timeout=timeout,
            block_on_failure=block_on_failure,
            is_async=is_async,
            priority=priority,
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
                "matcher": h.matcher,
                "hook_type": h.hook_type,
                "http_url": h.http_url,
                "http_headers": h.http_headers,
                "enabled": h.enabled,
                "timeout": h.timeout,
                "block_on_failure": h.block_on_failure,
                "is_async": h.is_async,
                "priority": h.priority,
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
_hook_manager_lock = threading.Lock()


def get_hook_manager() -> HookManager:
    """Get or create the global Hook manager instance (thread-safe)."""
    global _hook_manager
    if _hook_manager is None:
        with _hook_manager_lock:
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
