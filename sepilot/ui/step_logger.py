"""Interactive step logger for real-time agent progress display.

Shows dim-styled status lines on stderr so the user can follow
the agent's internal workflow without cluttering stdout.
"""

from __future__ import annotations

import time
import shutil
import re
from typing import TextIO

from rich.console import Console
from rich.cells import cell_len, chop_cells, set_cell_size


class StepLogger:
    """Logs agent workflow events (node transitions, tool calls, mode changes)
    to stderr using Rich dim styling.

    Only active when ``enabled=True`` (interactive mode).
    """

    def __init__(self, console: Console | None = None, *, enabled: bool = False):
        self._console = console or Console(stderr=True)
        self.enabled = enabled
        self._stream: TextIO | None = getattr(self._console, "file", None)
        self._is_tty = bool(self._stream and hasattr(self._stream, "isatty") and self._stream.isatty())
        self._last_stage: str | None = None
        self._stage_last_seen: dict[str, float] = {}
        self._last_tool_key: tuple[str, str] | None = None
        self._last_tool_at = 0.0
        self._last_tool_result_key: tuple[str, str] | None = None
        self._last_tool_result_at = 0.0
        self._last_message_at = 0.0
        self._last_message_text: str | None = None
        self._last_node_stage_at = 0.0
        self._last_tool_name_at: dict[str, float] = {}
        self._last_context_kind_at: dict[str, float] = {}
        self._last_plan_signature: tuple[str, ...] | None = None
        self._last_plan_progress_signature: tuple[tuple[str, ...], int, str] | None = None
        self._last_current_task: str | None = None
        self._checklist_lines_rendered = 0

        # Only expose high-value stages to users; housekeeping nodes are noisy.
        self._node_stage_map: dict[str, tuple[str, str]] = {
            "triage": ("analysis", "요청을 분석해 어떤 순서로 처리할지 실행 방향을 정하고 있어요."),
            "agent": ("analysis", "요청 해결에 필요한 작업을 단계별로 진행하고 있어요."),
            "hierarchical_planner": ("plan", "작업 단계를 나누고 우선순위를 정리하고 있어요."),
            "planner": ("plan", "작업 단계를 나누고 우선순위를 정리하고 있어요."),
            "approval": ("approval", "실행 전 권한이 필요한 작업인지 확인하고 안전 규칙을 점검하고 있어요."),
            "reporter": ("report", "진행 결과를 정리해 최종 답변으로 구성하고 있어요."),
        }
        self._hidden_nodes = {
            "iteration_guard",
            "context_manager",
            "tool_recorder",
            "memory_writer",
            "orchestrator",
            "tool_recommender",
            "tools",
            "verifier",
        }

    def _print_message(self, message: str, *, min_interval: float = 0.35) -> None:
        """Print a concise progress message with light throttling."""
        now = time.monotonic()
        if message == self._last_message_text and (now - self._last_message_at) < 20.0:
            return
        if now - self._last_message_at < min_interval:
            return
        self._last_message_at = now
        self._last_message_text = message
        self._console.print(f"[dim]• {message}[/dim]", highlight=False)

    @staticmethod
    def _ansi(text: str, code: str) -> str:
        """Apply ANSI style code to text."""
        return f"\x1b[{code}m{text}\x1b[0m"

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape sequences for non-TTY output."""
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    @staticmethod
    def _truncate_line(text: str, max_width: int) -> str:
        """Truncate line to fit terminal width and avoid wrapped box rendering."""
        if max_width <= 0:
            return ""
        if cell_len(text) <= max_width:
            return text
        ellipsis = "…"
        ellipsis_width = cell_len(ellipsis)
        if max_width <= ellipsis_width:
            return chop_cells(text, max_width)[0]
        prefix = chop_cells(text, max_width - ellipsis_width)[0]
        return f"{prefix}{ellipsis}"

    def _render_or_update_checklist(self, lines: list[str]) -> None:
        """Render checklist in-place (TTY) or append-only (non-TTY)."""
        if not lines:
            return

        if not self._is_tty or not self._stream:
            for line in lines:
                self._console.print(self._strip_ansi(line), highlight=False, markup=False)
            return

        old = self._checklist_lines_rendered
        new = len(lines)
        max_lines = max(old, new)

        # Move cursor to checklist start when updating existing block.
        if old > 0:
            self._stream.write(f"\x1b[{old}F")

        # Rewrite block line-by-line with clear-line.
        for i in range(max_lines):
            self._stream.write("\x1b[2K")
            if i < new:
                self._stream.write(lines[i])
            self._stream.write("\n")

        self._stream.flush()
        self._checklist_lines_rendered = new

    def log_node(self, node_name: str) -> None:
        """Log user-facing progress instead of raw graph-node names."""
        if not self.enabled:
            return
        if self._checklist_lines_rendered > 0:
            return
        if node_name in self._hidden_nodes:
            return
        stage_info = self._node_stage_map.get(node_name)
        if not stage_info:
            return
        stage_key, message = stage_info
        now = time.monotonic()
        if stage_key == self._last_stage:
            return
        # Node-level updates are noisy; emit only when genuinely changed and not too frequently.
        if now - self._stage_last_seen.get(stage_key, 0.0) < 20.0:
            return
        if now - self._last_node_stage_at < 6.0:
            return
        self._last_stage = stage_key
        self._stage_last_seen[stage_key] = now
        self._last_node_stage_at = now
        self._print_message(message)

    def log_tool(self, tool_name: str, summary: str = "") -> None:
        """Log only meaningful tool activity with deduplication."""
        if not self.enabled:
            return
        if self._checklist_lines_rendered > 0:
            return
        tool_templates = {
            "bash_execute": "터미널 명령을 실행해 실제 동작을 확인하고 있어요",
            "get_structure": "프로젝트 구조를 확인해 어떤 파일을 볼지 범위를 좁히고 있어요",
            "search_content": "코드에서 관련 키워드를 검색해 근거 위치를 찾고 있어요",
            "find_file": "요청과 관련된 파일 후보를 찾고 있어요",
            "list_directory": "디렉터리 구성을 확인해 탐색 경로를 정리하고 있어요",
            "file_read": "파일 내용을 확인해 현재 동작과 원인을 파악하고 있어요",
            "file_write": "요청 사항을 반영하는 파일을 작성하고 있어요",
            "file_edit": "요청 사항을 반영하도록 기존 코드를 수정하고 있어요",
            "patch_file": "변경사항을 안전하게 패치로 적용하고 있어요",
            "git": "변경 이력을 확인하거나 커밋 상태를 정리하고 있어요",
            "web_search": "최신/외부 정보를 확인하기 위해 웹을 검색하고 있어요",
            "web_fetch": "참고 링크의 원문을 확인해 정확한 근거를 수집하고 있어요",
        }
        template = tool_templates.get(tool_name)
        if not template:
            return

        short = summary[:80].replace("\n", " ").strip()
        tool_key = (tool_name, short)
        now = time.monotonic()
        if now - self._last_tool_name_at.get(tool_name, 0.0) < 20.0:
            return
        if tool_key == self._last_tool_key and (now - self._last_tool_at) < 1.2:
            return
        self._last_tool_key = tool_key
        self._last_tool_at = now
        self._last_tool_name_at[tool_name] = now

        text = f"→ {template}"
        if short:
            text += f" ({short} 기준으로 진행 중)"
        self._print_message(text, min_interval=0.0)

    def log_tool_result(self, tool_name: str, *, success: bool, decision_hint: str = "") -> None:
        """Log a compact post-tool outcome with inferred decision."""
        if not self.enabled:
            return
        if self._checklist_lines_rendered > 0:
            return

        prefix = "✓" if success else "✗"
        outcome = "결과 확인" if success else "실행 이슈"
        msg = f"→ {prefix} {tool_name}: {outcome}"
        if decision_hint:
            msg += f" - {decision_hint}"

        key = (tool_name, decision_hint[:100])
        now = time.monotonic()
        if key == self._last_tool_result_key and (now - self._last_tool_result_at) < 10.0:
            return
        self._last_tool_result_key = key
        self._last_tool_result_at = now
        self._print_message(msg, min_interval=0.0)

    def log_mode(self, old_mode: str, new_mode: str) -> None:
        """Log a mode transition (e.g. ``PLAN → CODE``)."""
        if not self.enabled:
            return
        if self._checklist_lines_rendered > 0:
            return
        self._print_message(f"모드를 {old_mode.upper()}에서 {new_mode.upper()}로 전환했어요.")

    def log_context(self, event: str) -> None:
        """Log a context-management event (compaction, pruning, etc.)."""
        if not self.enabled:
            return
        if self._checklist_lines_rendered > 0:
            return
        ev = (event or "").strip()
        lower = ev.lower()
        if "stagnation" in lower and "read-without-write" in lower:
            kind = "stagnation_read_without_write"
            msg = "반복 탐색이 감지되어, 곧 코드 수정 단계로 전환하려고 해요."
        elif "stagnation" in lower and "repetition" in lower:
            kind = "stagnation_repetition"
            msg = "반복 패턴을 감지해 접근 방식을 조정하고 있어요."
        elif "stagnation" in lower:
            kind = "stagnation_generic"
            msg = "진행 정체를 감지해 전략을 조정하고 있어요."
        else:
            kind = "context_generic"
            msg = "문맥을 정리해 다음 답변 품질을 높이고 있어요."

        now = time.monotonic()
        if now - self._last_context_kind_at.get(kind, 0.0) < 30.0:
            return
        self._last_context_kind_at[kind] = now
        self._print_message(msg)

    def log_think(self, category: str, preview: str) -> None:
        """Log a safe high-level thinking signal without scratchpad details."""
        if not self.enabled:
            return
        if self._checklist_lines_rendered > 0:
            return
        cat = (category or "general").strip().lower()
        msg = "해결 전략을 점검하고 있어요."
        if cat in {"plan", "planning"}:
            msg = "다음 작업 계획을 다듬고 있어요."
        elif cat in {"verify", "verification"}:
            msg = "결과의 정확성을 다시 확인하고 있어요."
        elif cat in {"risk", "safety"}:
            msg = "리스크와 안전성을 점검하고 있어요."
        self._print_message(msg)

    def _normalize_plan_steps(
        self,
        plan_steps: list[str] | None = None,
        planning_notes: list[str] | None = None,
    ) -> list[str]:
        """Build a cleaned, deduplicated plan step list."""
        steps: list[str] = []
        if isinstance(plan_steps, list):
            for step in plan_steps:
                s = str(step).strip()
                if s:
                    steps.append(s)

        if not steps and isinstance(planning_notes, list) and planning_notes:
            # Use the latest meaningful note when explicit plan_steps are absent.
            last_note = str(planning_notes[-1]).strip()
            if last_note and not last_note.startswith("[READ-ONLY]"):
                for line in last_note.splitlines():
                    candidate = line.strip().lstrip("-*•").strip()
                    if candidate:
                        steps.append(candidate)

        cleaned: list[str] = []
        noisy_keywords = (
            "Orchestrator",
            "Auto-exploration",
            "Retrieved",
            "Experience stored",
            "(cached)",
            "memory",
            "debate",
        )
        for raw in steps:
            s = raw
            s = s.replace("🎯", "").replace("📋", "").strip()
            if s.startswith("[READ-ONLY]"):
                s = s.replace("[READ-ONLY]", "", 1).strip()
            # Strip internal plan tags
            if s.startswith("[T]") or s.startswith("[O]"):
                s = s[3:].strip()
            # Skip internal orchestration/meta notes
            if any(keyword.lower() in s.lower() for keyword in noisy_keywords):
                continue
            if s and s not in cleaned:
                cleaned.append(s)
            if len(cleaned) >= 6:
                break
        return cleaned

    def log_plan(self, plan_steps: list[str] | None = None, planning_notes: list[str] | None = None) -> None:
        """Track plan changes internally; checklist renderer is the single visible source."""
        if not self.enabled:
            return

        cleaned = self._normalize_plan_steps(plan_steps, planning_notes)
        if not cleaned:
            return

        signature = tuple(cleaned)
        if signature == self._last_plan_signature:
            return
        self._last_plan_signature = signature

    def log_plan_progress(
        self,
        plan_steps: list[str] | None = None,
        current_plan_step: int | None = None,
        current_task: dict | None = None,
        planning_notes: list[str] | None = None,
    ) -> None:
        """Show todo-like progress view (done/in-progress/pending)."""
        if not self.enabled:
            return

        cleaned = self._normalize_plan_steps(plan_steps, planning_notes)
        if not cleaned:
            return

        cur = 0 if current_plan_step is None else int(current_plan_step)
        if cur < 0:
            cur = 0
        if cur >= len(cleaned):
            cur = len(cleaned) - 1

        task_desc = ""
        if isinstance(current_task, dict):
            task_desc = str(current_task.get("description") or "").strip()
        signature = (tuple(cleaned), cur, task_desc)
        if signature == self._last_plan_progress_signature:
            return
        self._last_plan_progress_signature = signature

        completed = max(0, min(cur, len(cleaned)))
        current_step = cleaned[cur]
        next_step = cleaned[cur + 1] if (cur + 1) < len(cleaned) else "마무리/응답 정리"
        header = f"작업 체크리스트 {cur + 1}/{len(cleaned)} · 완료 {completed}"

        body_rows: list[tuple[str, str]] = []
        for idx, step in enumerate(cleaned):
            if idx < cur:
                body_rows.append((f"✓ {step}", "1;32"))  # bold green
            elif idx == cur:
                body_rows.append((f"▶ {step}", "1;36"))  # bold cyan
            else:
                body_rows.append((f"· {step}", "2;37"))  # dim white

        footer_rows: list[tuple[str, str]] = [(f"next: {next_step}", "33")]  # yellow
        if task_desc:
            footer_rows.append((f"now:  {task_desc}", "35"))  # magenta
            self._last_current_task = task_desc

        plain_lines = [header] + [row for row, _ in body_rows] + [row for row, _ in footer_rows]

        # Clamp box width to terminal columns to prevent wrapped borders.
        term_cols = shutil.get_terminal_size((120, 20)).columns
        # Box overhead: left border + space + content + space + right border
        max_content_width = max(24, term_cols - 6)
        clipped_lines = [self._truncate_line(line, max_content_width) for line in plain_lines]
        width = max(cell_len(line) for line in clipped_lines)

        clipped_header = clipped_lines[0]
        clipped_body = clipped_lines[1:1 + len(body_rows)]
        clipped_footer = clipped_lines[1 + len(body_rows):]

        checklist_lines: list[str] = [
            self._ansi(f"╭{'─' * (width + 2)}╮", "1;36"),
            self._ansi(f"│ {set_cell_size(clipped_header, width)} │", "1;37"),
            self._ansi(f"├{'─' * (width + 2)}┤", "36"),
        ]
        for (row, color), clipped in zip(body_rows, clipped_body, strict=False):
            checklist_lines.append(self._ansi(f"│ {set_cell_size(clipped, width)} │", color))
        checklist_lines.append(self._ansi(f"├{'─' * (width + 2)}┤", "36"))
        for (row, color), clipped in zip(footer_rows, clipped_footer, strict=False):
            checklist_lines.append(self._ansi(f"│ {set_cell_size(clipped, width)} │", color))
        checklist_lines.append(self._ansi(f"╰{'─' * (width + 2)}╯", "1;36"))

        self._render_or_update_checklist(checklist_lines)
