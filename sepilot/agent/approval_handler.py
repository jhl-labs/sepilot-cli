"""Human-in-the-loop approval handling for tool execution.

This module follows the Single Responsibility Principle (SRP) by handling
only human approval interactions and tool risk assessment.

Integrates with PermissionManager for pattern-based permission rules.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from sepilot.agent.permission_rules import (
    PermissionLevel,
    PermissionManager,
    get_permission_manager,
)

# prompt_toolkit for better Unicode input handling
try:
    from prompt_toolkit import prompt as pt_prompt
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

_audit_logger = logging.getLogger(__name__)

INPUT_CANCELLED = "__INPUT_CANCELLED__"
INPUT_NO_INPUT = "__NO_INPUT__"


class AuditLogger:
    """Append-only JSONL audit logger for tool approval decisions.

    Logs every approve/deny/auto decision to ~/.sepilot/audit.jsonl.
    Logging failures are silently suppressed to never block approval flow.
    """

    def __init__(self, path: Path | None = None, session_id: str | None = None):
        self._path = path or (Path.home() / ".sepilot" / "audit.jsonl")
        self.session_id = session_id or uuid.uuid4().hex[:12]

    def log(
        self,
        tool: str,
        action: str,
        risk_level: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a single audit entry. Never raises."""
        try:
            entry: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "action": action,
                "risk_level": risk_level,
                "session_id": self.session_id,
            }
            if extra:
                entry.update(extra)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            # Audit logging must never block the approval flow
            _audit_logger.debug("Audit log write failed: %s", exc)


class ApprovalHandler:
    """Handles human-in-the-loop approval for sensitive tool executions.

    This class manages:
    - Interactive approval prompts
    - Tool risk assessment
    - Session-wide auto-approval
    - Web-based input (when available)
    """

    def __init__(
        self,
        console: Console | None = None,
        sensitive_tools: set[str] | None = None,
        auto_approve: bool = False,
        permission_manager: PermissionManager | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        """Initialize approval handler.

        Args:
            console: Rich console for output
            sensitive_tools: Set of tool names requiring approval
            auto_approve: If True, auto-approve all tool executions
            permission_manager: Permission manager for pattern-based rules
            audit_logger: Optional audit logger for recording decisions
        """
        # Always keep a Rich console so approval UI can render syntax highlight
        # even when the main agent runs in non-verbose mode.
        self.console = console or Console(
            file=(sys.__stderr__ or sys.stderr),
            force_terminal=True,
        )
        self.sensitive_tools = sensitive_tools or {
            "bash_execute", "file_write", "file_edit", "git", "web_search"
        }
        self.auto_approve = auto_approve
        self.auto_approve_session = False  # Session-wide auto-approve
        self.permission_manager = permission_manager or get_permission_manager()
        self.audit_logger = audit_logger or AuditLogger()

    def handle_interrupt_events(self, interrupts: list) -> Any:
        """Handle LangGraph interrupt tuples by prompting the user.

        Args:
            interrupts: List of interrupt events

        Returns:
            User response(s)
        """
        responses = []
        for interrupt_event in interrupts:
            payload = getattr(interrupt_event, "value", interrupt_event)
            responses.append(self.prompt_user_for_interrupt(payload))
        if len(responses) == 1:
            return responses[0]
        return responses

    def prompt_user_for_interrupt(self, payload: Any) -> Any:
        """Prompt CLI user for approval/feedback based on payload.

        Args:
            payload: Interrupt payload containing tool information

        Returns:
            User decision dictionary
        """
        if not isinstance(payload, dict):
            return payload

        # Handle tool approval from wrapped tools
        if payload.get("type") == "tool_approval":
            return self._handle_tool_approval(payload)
        if payload.get("type") == "mode_switch_request":
            return self._handle_mode_switch_request(payload)

        # Fallback for other interrupt types
        return payload

    def _audit_tool_calls(self, tool_calls: list, action: str) -> None:
        """Log audit entries for a batch of tool calls."""
        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            risk = tc.get("risk_level", "")
            self.audit_logger.log(tool=tool_name, action=action, risk_level=risk)

    def _handle_tool_approval(self, payload: dict) -> dict:
        """Handle tool approval request with pattern-based permission rules.

        Args:
            payload: Tool approval payload

        Returns:
            Approval decision dictionary
        """
        tool_calls = payload.get("tool_calls", [])
        force_prompt = bool(payload.get("force_prompt", False))

        # Check permission rules for each tool call.
        # Deny has highest priority across the whole batch.
        # Auto-approve only when every call is explicitly ALLOW.
        all_auto_allowed = bool(tool_calls)
        auto_allowed_tools: list[str] = []
        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            args = tc.get("args", {})

            permission, reason = self.permission_manager.check_permission(tool_name, args)

            # DENY: Block execution
            if permission == PermissionLevel.DENY:
                if self.console:
                    self.console.print(
                        f"[bold red]차단됨: {tool_name}[/bold red]\n"
                        f"[red]사유: {reason}[/red]"
                    )
                self._audit_tool_calls(tool_calls, "deny")
                return {
                    "type": "deny",
                    "decision": "deny",
                    "reason": f"Permission denied: {reason}"
                }

            # ALLOW: mark as auto-approved candidate
            if permission == PermissionLevel.ALLOW:
                auto_allowed_tools.append(tool_name)
                continue

            # ASK (or unknown): requires user prompt for this batch
            all_auto_allowed = False

        if all_auto_allowed and not force_prompt:
            if self.console:
                for tool_name in auto_allowed_tools:
                    self.console.print(
                        f"[dim green]✓ 자동 허용: {tool_name}[/dim green]"
                    )
            self._audit_tool_calls(tool_calls, "auto")
            return {"type": "accept", "decision": "approve"}

        # Auto-approve ASK-level calls when explicitly enabled.
        # DENY rules are still enforced above.
        if self.auto_approve:
            if self.console and tool_calls:
                tool_list = ", ".join({tc.get("name", "unknown") for tc in tool_calls})
                self.console.print(f"[dim cyan]🔓 자동 승인 모드: {tool_list}[/dim cyan]")
            self._audit_tool_calls(tool_calls, "auto")
            return {"type": "accept", "decision": "approve"}

        # ASK: Show tool calls and prompt user
        if tool_calls:
            self._display_tool_calls(
                tool_calls,
                reason=payload.get("reason", ""),
                llm_rationale=payload.get("llm_rationale", ""),
            )
        else:
            # Fallback for legacy format
            self._display_legacy_tool_call(payload)

        # Check for non-TTY mode.
        # If web input queue exists, we can still collect approval via web channel.
        if not sys.stdin.isatty():
            if self.console:
                self.console.print("[dim]TTY가 아니므로 자동으로 승인합니다.[/dim]")
            self._audit_tool_calls(tool_calls, "auto")
            return {"type": "accept", "decision": "approve"}

        # Check session-wide auto-approve
        if self.auto_approve_session:
            if self.console:
                self.console.print(
                    "[dim cyan]✓ 세션 자동 승인 모드 활성화됨 - 자동으로 승인합니다.[/dim cyan]"
                )
            self._audit_tool_calls(tool_calls, "auto")
            return {"type": "accept", "decision": "approve"}

        result = self._get_user_approval()
        # Log the user's interactive decision
        decision = result.get("decision", "deny")
        action = "approve" if decision == "approve" else "deny"
        self._audit_tool_calls(tool_calls, action)
        return result

    def _display_tool_calls(
        self,
        tool_calls: list,
        *,
        reason: str = "",
        llm_rationale: str = "",
    ) -> None:
        """Display tool calls for approval.

        Args:
            tool_calls: List of tool call dictionaries
            reason: Approval reason from the approval node
            llm_rationale: Model-provided rationale for this tool usage
        """
        from rich.console import Group
        from rich.rule import Rule
        from rich.table import Table
        from rich.text import Text

        preface_blocks: list[Any] = []
        reason_text = str(reason or "").strip()
        if reason_text:
            preface_blocks.append(Rule("approval reason", style="yellow"))
            preface_blocks.append(Text(reason_text, style="yellow"))

        rationale_text = self._sanitize_approval_rationale(str(llm_rationale or "").strip())
        if not rationale_text:
            rationale_text = self._infer_tool_rationale(tool_calls)
        if rationale_text:
            max_chars = 800
            if len(rationale_text) > max_chars:
                rationale_text = rationale_text[:max_chars] + "..."
            preface_blocks.append(Rule("llm 판단 근거", style="cyan"))
            preface_blocks.append(Text(rationale_text, style="cyan"))

        tool_blocks = []
        for idx, tc in enumerate(tool_calls, 1):
            tool_name = tc.get("name", "unknown")
            args = tc.get("args", {})
            content_value = args.get("content")
            file_path = (
                args.get("file_path") or
                args.get("path") or
                args.get("notebook_path")
            )

            # Assess risk for color coding
            risk_info = self.assess_tool_risk(tool_name, args)
            risk_level = tc.get("risk_level") or risk_info.get("level", "low")
            risk_reasons = tc.get("risk_reasons") or risk_info.get("reasons", [])

            # Risk-based color scheme
            risk_colors = {
                "high": ("bold red", "red", "HIGH"),
                "medium": ("bold yellow", "yellow", "MED"),
                "low": ("bold green", "green", "LOW"),
            }
            risk_style, risk_border, risk_label = risk_colors.get(
                str(risk_level).lower(), ("bold green", "green", "LOW")
            )

            meta = Table.grid(padding=(0, 1))
            meta.add_column(style="bold cyan", no_wrap=True)
            meta.add_column(style="white")
            meta.add_row("tool", tool_name)

            # Show risk level with color badge
            risk_display = Text()
            risk_display.append(f" {risk_label} ", style=f"bold white on {risk_border}")
            if isinstance(risk_reasons, list) and risk_reasons:
                reason_preview = "; ".join(str(item) for item in risk_reasons[:3])
                if len(reason_preview) > 120:
                    reason_preview = reason_preview[:120] + "..."
                risk_display.append(f" {reason_preview}", style="dim")
            meta.add_row("risk", risk_display)

            # Show key arguments in a compact summary
            arg_summary_parts = []
            for key in ("file_path", "path", "cwd", "timeout", "replace_all"):
                if key in args and args.get(key) is not None:
                    value = str(args[key])
                    if len(value) > 120:
                        value = value[:120] + "..."
                    meta.add_row(key, value)
                    if key in ("file_path", "path"):
                        arg_summary_parts.append(value)

            # Show git tool intent explicitly for safer approval decisions.
            if tool_name == "git":
                for key in ("operation", "args", "message", "path"):
                    if key in args and args.get(key) is not None:
                        value = str(args[key])
                        if len(value) > 120:
                            value = value[:120] + "..."
                        meta.add_row(key, value)

            if "command" in args:
                cmd = str(args["command"])
                if len(cmd) > 120:
                    cmd = cmd[:120] + "..."
                meta.add_row("command", cmd)

            # Build section with risk-colored border
            section: list[Any] = [
                Rule(f" [{risk_label}] request {idx}/{len(tool_calls)} ", style=risk_border),
                Text(f"  {tool_name}", style=risk_style),
                meta,
            ]

            edit_preview = self._build_edit_preview(tool_name, args, file_path)
            if edit_preview:
                section.append(edit_preview)
            elif tool_name == "git":
                git_preview = self._build_git_preview(args)
                if git_preview:
                    section.append(git_preview)
            elif content_value and tool_name in ("file_write", "file_edit", "notebook_edit"):
                syntax = self._create_syntax_panel(content_value, file_path)
                if syntax:
                    section.append(Rule("after", style="green"))
                    section.append(syntax)

            tool_blocks.extend(section)

        self.console.print()
        # Header with risk-aware styling
        max_risk = "low"
        for tc in tool_calls:
            tc_risk = tc.get("risk_level") or self.assess_tool_risk(
                tc.get("name", ""), tc.get("args", {})
            ).get("level", "low")
            levels = ["low", "medium", "high"]
            if levels.index(str(tc_risk).lower()) > levels.index(max_risk):
                max_risk = str(tc_risk).lower()

        header_colors = {"high": "bold red", "medium": "bold yellow", "low": "bold green"}
        border_colors = {"high": "red", "medium": "yellow", "low": "green"}
        header_style = header_colors.get(max_risk, "bold yellow")
        border_style = border_colors.get(max_risk, "yellow")

        self.console.print(Text("  도구 실행 승인 요청", style=header_style))
        self.console.print(Rule(style=border_style))
        self.console.print(Group(*(preface_blocks + tool_blocks)))
        self.console.print(Rule(style=border_style))

    def _sanitize_approval_rationale(self, text: str) -> str:
        """Drop internal system/meta notes from approval rationale text."""
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        lower = cleaned.lower()
        noise_tokens = (
            "experience stored",
            "quality:",
            "orchestrator",
            "memory writer",
            "retrieved",
            "rolled back",
            "debate skipped",
            "[read-only]",
        )
        if any(token in lower for token in noise_tokens):
            return ""
        return cleaned

    def _infer_tool_rationale(self, tool_calls: list[dict[str, Any]]) -> str:
        """Generate a concise rationale when model rationale is missing or noisy."""
        if not tool_calls:
            return ""

        first = tool_calls[0] or {}
        tool_name = str(first.get("name", "unknown"))
        args = first.get("args", {}) or {}
        file_path = args.get("file_path") or args.get("path") or args.get("notebook_path") or "대상 파일"

        if tool_name == "file_edit":
            old_value = str(args.get("old_string", "") or "").strip()
            new_value = str(args.get("new_string", "") or "").strip()
            if old_value and new_value:
                old_line = old_value.splitlines()[0][:80]
                new_line = new_value.splitlines()[0][:80]
                return f"{file_path}에서 '{old_line}' 항목을 '{new_line}'로 교체하려고 합니다."
            if new_value:
                new_line = new_value.splitlines()[0][:120]
                return f"{file_path}에 '{new_line}' 내용을 반영하려고 합니다."
            return f"{file_path}의 기존 내용을 요청사항에 맞게 수정하려고 합니다."

        if tool_name == "file_write":
            return f"{file_path} 파일 내용을 요청사항에 맞게 작성하려고 합니다."

        if tool_name == "bash_execute":
            command = str(args.get("command", "") or "").strip()
            if command:
                return f"명령 실행 결과로 변경 사항을 검증하기 위해 '{command}'를 실행하려고 합니다."
            return "명령 실행으로 현재 상태를 확인하려고 합니다."

        if tool_name == "git":
            operation = str(args.get("operation", "") or "").strip() or str(args.get("command", "") or "").strip()
            if operation:
                return f"변경 이력 점검을 위해 git {operation} 작업을 수행하려고 합니다."
            return "변경 이력을 확인하기 위해 git 작업을 수행하려고 합니다."

        return f"{tool_name} 도구를 사용해 요청사항을 반영하려고 합니다."

    def _display_legacy_tool_call(self, payload: dict) -> None:
        """Display legacy format tool call.

        Args:
            payload: Legacy tool call payload
        """
        tool_name = payload.get("action", "unknown")
        args = payload.get("args", {})

        self._display_tool_calls([{"name": tool_name, "args": args}])

    def _create_syntax_panel(
        self, content: str, file_path: str | None
    ) -> Any | None:
        """Create syntax-highlighted panel for file content.

        Args:
            content: File content
            file_path: Optional file path for language detection

        Returns:
            Syntax object or None
        """
        try:
            from rich.syntax import Syntax

            # Detect language from file extension
            lexer = "text"
            if file_path:
                ext = str(file_path).split(".")[-1].lower() if "." in str(file_path) else ""
                lexer_map = {
                    "py": "python", "js": "javascript", "ts": "typescript",
                    "tsx": "tsx", "jsx": "jsx", "json": "json", "yaml": "yaml",
                    "yml": "yaml", "md": "markdown", "html": "html", "css": "css",
                    "sh": "bash", "bash": "bash", "sql": "sql", "rs": "rust",
                    "go": "go", "java": "java", "kt": "kotlin", "rb": "ruby",
                }
                lexer = lexer_map.get(ext, "text")

            # Truncate long content
            content_str = str(content)
            max_lines = 50
            lines = content_str.split("\n")
            if len(lines) > max_lines:
                truncated = "\n".join(lines[:max_lines])
                truncated += f"\n\n... ({len(lines) - max_lines} more lines)"
                content_str = truncated

            return Syntax(
                content_str,
                lexer,
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
        except ImportError:
            return None

    def _build_edit_preview(self, tool_name: str, args: dict[str, Any], file_path: str | None) -> Any | None:
        """Build pretty preview for edit-like tool calls (old/new with separators)."""
        if not self.console:
            return None

        if tool_name not in ("file_edit", "file_write", "notebook_edit"):
            return None

        from rich.console import Group
        from rich.rule import Rule
        from rich.text import Text

        old_value = args.get("old_string")
        new_value = args.get("new_string")
        content_value = args.get("content")

        blocks: list[Any] = []
        if file_path:
            blocks.append(Text(f"  target: {file_path}", style="dim"))

        if old_value:
            old_syntax = self._create_syntax_panel(str(old_value), file_path)
            if old_syntax:
                blocks.append(Rule("before", style="red"))
                blocks.append(old_syntax)

        # file_write/content-only path
        if new_value is None and content_value is not None:
            new_value = content_value

        if new_value:
            new_syntax = self._create_syntax_panel(str(new_value), file_path)
            if new_syntax:
                blocks.append(Rule("after", style="green"))
                blocks.append(new_syntax)

        if not blocks:
            return None
        return Group(*blocks)

    def _build_git_preview(self, args: dict[str, Any]) -> Any | None:
        """Build a concise preview of the git command intent."""
        if not self.console:
            return None

        from rich.console import Group
        from rich.rule import Rule
        from rich.text import Text

        operation = str(args.get("operation") or "").strip()
        extra_args = str(args.get("args") or "").strip()
        message = str(args.get("message") or "").strip()

        if not operation and "command" in args:
            operation = str(args.get("command") or "").strip()

        if not operation:
            return None

        command_preview = f"git {operation}"
        if message and operation == "commit":
            command_preview += f" -m {message!r}"
        if extra_args:
            command_preview += f" {extra_args}"

        max_len = 180
        if len(command_preview) > max_len:
            command_preview = command_preview[: max_len - 3] + "..."

        blocks: list[Any] = [
            Rule("git intent", style="yellow"),
            Text(f"$ {command_preview}", style="bold yellow"),
        ]
        return Group(*blocks)

    def _get_user_approval(self) -> dict:
        """Get user approval decision interactively.

        Returns:
            Decision dictionary
        """
        prompt_text = "승인? (y)승인 (n)거부 (m)피드백 (a)세션 자동승인 > "

        while True:
            choice = self._safe_input(prompt_text).lower()

            if choice == INPUT_NO_INPUT.lower():
                return {
                    "type": "deny",
                    "decision": "deny",
                    "reason": "승인 입력을 받지 못해 자동 거부됨",
                }

            if choice == INPUT_CANCELLED.lower():
                return {
                    "type": "deny",
                    "decision": "deny",
                    "reason": "사용자가 승인 프롬프트를 취소함",
                }

            if choice in ("", "y", "yes"):
                return {"type": "accept", "decision": "approve"}

            if choice in ("n", "no"):
                reason = self._safe_input("거부 사유 (엔터로 생략): ")
                return {
                    "type": "deny",
                    "decision": "deny",
                    "reason": reason or "사용자가 거부함"
                }

            if choice in ("m", "msg", "message", "r"):
                message = self._safe_input("사용자 피드백을 입력하세요: ")
                return {
                    "type": "response",
                    "decision": "respond",
                    "message": message
                }

            if choice in ("a", "always", "all"):
                self.auto_approve_session = True
                if self.console:
                    self.console.print(
                        "[bold green]✓ 세션 자동 승인 모드 활성화! "
                        "이후 모든 도구 실행이 자동으로 승인됩니다.[/bold green]"
                    )
                else:
                    print("✓ 세션 자동 승인 모드 활성화! 이후 모든 도구 실행이 자동으로 승인됩니다.")
                return {"type": "accept", "decision": "approve"}

            if self.console:
                self.console.print(
                    "[red]유효하지 않은 입력입니다. y/n/m/a 중에서 선택하세요.[/red]"
                )
            else:
                print("유효하지 않은 입력입니다. y/n/m/a 중에서 선택하세요.")

    def _handle_mode_switch_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ask user whether blocked execution may switch mode."""
        current_mode = str(payload.get("current_mode", "plan")).upper()
        suggested_mode = str(payload.get("suggested_mode", "code")).upper()
        blocked_tools = payload.get("blocked_tools", [])
        reason = payload.get("reason", "")

        if self.console:
            self.console.print(
                f"[bold yellow]🔁 Mode switch required:[/bold yellow] {current_mode} → {suggested_mode}"
            )
            if blocked_tools:
                self.console.print(f"[dim]Blocked tools: {', '.join(map(str, blocked_tools))}[/dim]")
            if reason:
                self.console.print(f"[dim]{reason}[/dim]")

        prompt = f"{suggested_mode} 모드로 전환할까요? (y/n) > "
        choice = self._safe_input(prompt).lower()
        if choice in ("", "y", "yes"):
            return {
                "type": "accept",
                "decision": "approve",
                "status": "approved",
                "switch_mode": suggested_mode.lower(),
            }

        return {
            "type": "deny",
            "decision": "deny",
            "status": "deny",
            "reason": "사용자가 모드 전환을 거부함",
        }

    def _safe_input(self, prompt: str) -> str:
        """Get user input with proper Unicode handling.

        Args:
            prompt: Prompt text

        Returns:
            User input string
        """
        try:
            if HAS_PROMPT_TOOLKIT:
                return pt_prompt(prompt).strip()
            else:
                return input(prompt).strip()
        except (UnicodeDecodeError, UnicodeError) as e:
            if self.console:
                self.console.print(f"[yellow]⚠️  입력 인코딩 오류 (무시됨): {e}[/yellow]")
            else:
                print(f"⚠️  입력 인코딩 오류 (무시됨): {e}")
            return ""
        except EOFError:
            return INPUT_NO_INPUT
        except KeyboardInterrupt:
            return INPUT_CANCELLED

    def assess_tool_risk(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess risk level for a tool execution.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Risk assessment dictionary with level and reasons
        """
        level = "low"
        reasons = []
        command = str(args.get("command", "")).lower()
        path_candidates = self._extract_paths_from_args(args)
        cwd = args.get("cwd") or os.getcwd()
        cwd_path = Path(cwd).resolve()

        def bump(new_level: str, reason: str):
            nonlocal level
            levels = ["low", "medium", "high"]
            if levels.index(new_level) > levels.index(level):
                level = new_level
            reasons.append(reason)

        if tool_name == "bash_execute":
            dangerous_patterns = [
                "rm -rf", "mkfs", "dd if=", "shutdown", "reboot",
                ":(){", "chmod -R 777", "chown -R /"
            ]
            for pattern in dangerous_patterns:
                if pattern in command:
                    bump("high", f"명령에 위험 패턴 포함: {pattern}")
            if "sudo" in command:
                bump("high", "sudo 사용 요청")
            if "curl" in command or "wget" in command:
                bump("medium", "네트워크 다운로드 명령")

        elif tool_name in {"file_write", "file_edit", "git"}:
            for p in path_candidates:
                try:
                    resolved = p.resolve()
                    if cwd_path not in resolved.parents and resolved != cwd_path:
                        bump("medium", f"작업 디렉터리 외 파일 접근: {resolved}")
                except Exception:
                    bump("medium", "경로 확인 실패")

        if not reasons:
            reasons.append("기본 안전 수준")

        return {"level": level, "reasons": reasons}

    def _extract_paths_from_args(self, args: dict[str, Any]) -> list[Path]:
        """Extract file paths from tool arguments.

        Args:
            args: Tool arguments dictionary

        Returns:
            List of Path objects
        """
        paths = []
        path_keys = ["file_path", "path", "directory", "target", "source", "dest"]

        for key in path_keys:
            if key in args and args[key]:
                try:
                    paths.append(Path(args[key]))
                except Exception:
                    pass

        return paths


def normalize_approval_response(decision: Any) -> dict[str, Any]:
    """Normalize various approval response formats.

    Args:
        decision: Raw decision from interrupt handler

    Returns:
        Normalized decision dictionary
    """
    if isinstance(decision, dict):
        decision_type = decision.get("type") or decision.get("decision")

        if decision_type in ("accept", "approve", "yes", "y"):
            return {"status": "approved"}
        elif decision_type in ("deny", "reject", "no", "n"):
            return {
                "status": "deny",
                "reason": decision.get("reason") or decision.get("message")
            }
        elif decision_type in ("response", "respond", "feedback", "message"):
            return {
                "status": "feedback",
                "message": decision.get("message") or decision.get("feedback")
            }

    # Fail safe: reject unknown/invalid approval response formats
    return {"status": "deny", "reason": "Unrecognized approval response format"}
