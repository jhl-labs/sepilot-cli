"""Tool execution node for LangGraph workflow.

This module contains the enhanced tool node implementation extracted from base_agent.py.
The tool node handles tool call normalization, caching, execution, and result tracking.

Features:
- Pre-operation checkpoints for risky operations (file changes, bash, git)
- Tool call normalization and deduplication
- Parallel execution with caching
- File change tracking
"""

import json
import os
import re
import time
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode

from sepilot.agent import state_helpers, tool_tracker
from sepilot.agent.agent_utils import (
    cache_entry_is_stale,
    canonicalize_args,
    make_cache_entry,
    merge_updates,
    safe_load_cache_value,
)
from sepilot.agent.enhanced_state import AgentMode, ErrorLevel
from sepilot.agent.error_recovery import ErrorRecoveryStrategy
from sepilot.agent.file_tracking import detect_file_changes
from sepilot.agent.mode_manager import is_tool_allowed

if TYPE_CHECKING:
    from sepilot.agent.base_agent import ReactAgent


# Risky operations that require pre-operation checkpoints
RISKY_TOOLS = {
    'file_write': 'file_edit',
    'file_edit': 'file_edit',
    'bash_execute': 'bash',
    'git_commit': 'git',
    'git_push': 'git',
    'git_checkout': 'git',
    'git_reset': 'git',
}


def _infer_required_mode(blocked_calls: list[dict[str, Any]]) -> AgentMode | None:
    """Infer the minimum mode needed for blocked tool calls."""
    if not blocked_calls:
        return None

    names = {str(tc.get("name", "")) for tc in blocked_calls}
    code_tools = {"file_edit", "file_write", "apply_patch", "notebook_edit", "patch_file"}
    exec_tools = {"bash_execute", "bash_background", "bash_output", "git", "kill_shell", "list_shells"}

    if names & code_tools:
        return AgentMode.CODE
    if names & exec_tools:
        return AgentMode.EXEC
    return AgentMode.CODE


def _is_approved_decision(decision: Any) -> bool:
    """Check whether an interrupt decision means approval."""
    if isinstance(decision, bool):
        return decision
    if not isinstance(decision, dict):
        return False

    status = str(decision.get("status", "")).lower()
    d_type = str(decision.get("type", "")).lower()
    d_decision = str(decision.get("decision", "")).lower()

    return (
        status in {"approved", "approve", "accept", "accepted", "yes", "y"}
        or d_type in {"accept", "approve", "approved", "yes", "y"}
        or d_decision in {"approve", "approved", "accept", "yes", "y"}
    )


def _shorten(text: str, limit: int = 90) -> str:
    """Return a single-line, bounded preview string."""
    compact = " ".join(str(text).split())
    return compact[:limit]


def _build_tool_context_summary(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Build a concise pre-execution context summary for step logger."""
    if tool_name in {"file_read", "file_edit", "file_write", "patch_file"}:
        path = tool_args.get("file_path") or tool_args.get("path")
        if path:
            return f"path={path}"
    if tool_name in {"search_content", "ripgrep_search"}:
        query = tool_args.get("query") or tool_args.get("pattern") or tool_args.get("search_term")
        if query:
            return f"query={_shorten(str(query), 60)}"
    if tool_name == "find_file":
        pattern = tool_args.get("pattern") or tool_args.get("query")
        if pattern:
            return f"pattern={_shorten(str(pattern), 60)}"
    if tool_name in {"get_structure", "list_directory"}:
        path = tool_args.get("path") or tool_args.get("directory") or "."
        return f"path={path}"
    if tool_name == "bash_execute":
        cmd = tool_args.get("command")
        if cmd:
            return f"cmd={_shorten(str(cmd), 80)}"
    return ""


def _extract_tool_decision_hint(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: Any,
    error: str | None,
) -> tuple[bool, str]:
    """Build a user-facing post-execution hint: what happened and what was inferred."""
    if error:
        return False, "실행 오류가 있어 입력값/경로/명령을 다시 확인해야 한다고 판단했어요"

    if tool_name == "file_read":
        path = tool_args.get("file_path") or tool_args.get("path")
        if path:
            return True, f"{path} 내용을 확인했고 수정 필요 구간을 판단했어요"
        return True, "파일 내용을 확인했고 수정 필요 구간을 판단했어요"

    if tool_name in {"search_content", "ripgrep_search", "find_file"}:
        query = tool_args.get("query") or tool_args.get("pattern") or tool_args.get("search_term")
        if query:
            return True, f"'{_shorten(str(query), 40)}' 기준으로 관련 위치 후보를 좁혔어요"
        return True, "관련 위치 후보를 좁혔어요"

    if tool_name in {"get_structure", "list_directory"}:
        return True, "탐색 범위를 정리했고 다음에 볼 파일 후보를 확정했어요"

    if tool_name in {"file_edit", "file_write", "patch_file"}:
        path = tool_args.get("file_path") or tool_args.get("path")
        if path:
            return True, f"{path}에 변경을 적용했고 후속 검증이 필요하다고 판단했어요"
        return True, "코드 변경을 적용했고 후속 검증이 필요하다고 판단했어요"

    if tool_name == "bash_execute":
        command = str(tool_args.get("command", ""))
        result_text = str(tool_result or "")
        exit_match = re.search(r"(?:Exit code|Return code):\s*(-?\d+)", result_text, re.IGNORECASE)
        exit_code = int(exit_match.group(1)) if exit_match else 0

        if any(k in command for k in ("pytest", "test", "ruff", "mypy", "uv run")):
            if exit_code == 0:
                return True, "검증 명령이 통과해서 현재 변경 방향이 유효하다고 판단했어요"
            return False, f"검증 명령이 실패(exit={exit_code})해서 추가 수정이 필요하다고 판단했어요"

        if exit_code == 0:
            return True, "명령 결과를 확인했고 다음 실행 단계를 이어갈 수 있다고 판단했어요"
        return False, f"명령이 비정상 종료(exit={exit_code})되어 원인 점검이 필요하다고 판단했어요"

    return True, "실행 결과를 확인했고 다음 단계를 진행할 수 있다고 판단했어요"


def extract_target_files(tool_calls: list[dict], workspace_path: str) -> list[str]:
    """Extract target files from tool calls for checkpoint tracking.

    Args:
        tool_calls: List of tool call dictionaries
        workspace_path: Current workspace path

    Returns:
        List of file paths that will be affected
    """
    target_files = []

    for tc in tool_calls:
        tool_name = tc.get('name', 'unknown')
        tool_args = tc.get('args', {})

        if tool_name in ('file_write', 'file_edit'):
            file_path = tool_args.get('file_path') or tool_args.get('path')
            if file_path:
                # Convert to absolute path if relative
                if not os.path.isabs(file_path):
                    file_path = os.path.join(workspace_path, file_path)
                target_files.append(file_path)

        elif tool_name == 'bash_execute':
            # Try to detect files from common bash patterns
            command = tool_args.get('command', '')
            # Simple heuristics for file-modifying commands
            for pattern in ['>', '>>', 'rm ', 'mv ', 'cp ', 'touch ']:
                if pattern in command:
                    # Track the workspace as potentially affected
                    target_files.append(workspace_path)
                    break

    return list(set(target_files))  # Remove duplicates


def should_create_checkpoint(tool_calls: list[dict]) -> tuple[bool, str]:
    """Determine if tool calls require a pre-operation checkpoint.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        Tuple of (should_checkpoint, operation_type)
    """
    for tc in tool_calls:
        tool_name = tc.get('name', 'unknown')
        if tool_name in RISKY_TOOLS:
            return True, RISKY_TOOLS[tool_name]
    return False, ''


# Parameter aliases for common LLM mistakes
PARAM_ALIASES = {
    'file_write': {'path': 'file_path'},
    'file_read': {'path': 'file_path'},
    'file_edit': {'path': 'file_path'},
    'notebook_edit': {'path': 'notebook_path'},
}


def deduplicate_tool_calls(tool_calls: list[dict], workspace_path: str, console=None) -> list[dict]:
    """Remove duplicate tool calls from the same LLM response.

    Args:
        tool_calls: List of tool call dictionaries
        workspace_path: Workspace path for canonicalizing arguments
        console: Rich console for output (optional)

    Returns:
        List of deduplicated tool call dictionaries
    """
    seen_signatures: set[str] = set()
    deduplicated = []
    duplicates_removed = 0

    for tc in tool_calls:
        tool_name = tc.get('name', 'unknown')
        tool_args = tc.get('args', {})
        canonical_args = canonicalize_args(tool_args or {}, workspace_path)

        # Create a signature for deduplication
        signature = f"{tool_name}:{json.dumps(canonical_args, sort_keys=True)}"

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            deduplicated.append(tc)
        else:
            duplicates_removed += 1

    if duplicates_removed > 0 and console:
        console.print(f"[dim yellow]⚠️ Removed {duplicates_removed} duplicate tool call(s)[/dim yellow]")

    return deduplicated


def normalize_tool_calls(tool_calls: list[dict], valid_tool_names: set, console=None) -> list[dict]:
    """Normalize tool names and parameters to fix common LLM mistakes.

    Args:
        tool_calls: List of tool call dictionaries
        valid_tool_names: Set of valid tool names
        console: Rich console for output (optional)

    Returns:
        List of normalized tool call dictionaries
    """
    normalized = []
    for tc in tool_calls:
        tool_name = tc.get('name', 'unknown')
        original_name = tool_name

        # Remove common invalid suffixes (.exec, .run, .call, .invoke)
        for suffix in ['.exec', '.run', '.call', '.invoke', '.execute']:
            if tool_name.endswith(suffix):
                tool_name = tool_name[:-len(suffix)]
                break

        # Check if normalized name is valid
        if tool_name not in valid_tool_names and original_name != tool_name:
            if console:
                console.print(f"[dim yellow]⚠️ Normalized tool name: {original_name} → {tool_name}[/dim yellow]")

        # Update tool call with normalized name
        tc_modified = False
        if tool_name != original_name:
            tc = dict(tc)  # Make a copy
            tc['name'] = tool_name
            tc_modified = True

        # Normalize parameter names
        if tool_name in PARAM_ALIASES:
            aliases = PARAM_ALIASES[tool_name]
            args = tc.get('args', {})
            new_args = dict(args)
            for wrong_param, correct_param in aliases.items():
                if wrong_param in new_args and correct_param not in new_args:
                    new_args[correct_param] = new_args.pop(wrong_param)
                    if console:
                        console.print(f"[dim yellow]⚠️ Normalized parameter: {wrong_param} → {correct_param}[/dim yellow]")
            if new_args != args:
                if not tc_modified:
                    tc = dict(tc)
                tc['args'] = new_args

        normalized.append(tc)

    return normalized


def create_enhanced_tool_node(agent: 'ReactAgent') -> callable:
    """Create an enhanced tool node function bound to an agent instance.

    Args:
        agent: ReactAgent instance

    Returns:
        A function that can be used as a LangGraph node
    """
    tool_node = ToolNode(agent.langchain_tools)
    valid_tool_names = {t.name for t in agent.langchain_tools}

    def enhanced_tool_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute tools with normalization, caching, deduplication, and tracking."""
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, 'tool_calls', [])
        state_updates: dict[str, Any] = {}

        # File change tracking - snapshot before (also need workspace_path for deduplication)
        workspace_path = os.getcwd()
        files_before = agent._snapshot_workspace()

        # Normalize tool calls
        tool_calls = normalize_tool_calls(tool_calls, valid_tool_names, agent.console)

        # Mode-based tool filtering: block tools not allowed in current mode
        current_mode = state.get("current_mode", AgentMode.AUTO)
        if current_mode != AgentMode.AUTO and tool_calls:
            allowed_calls = []
            blocked_calls = []
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                if is_tool_allowed(tool_name, current_mode):
                    allowed_calls.append(tc)
                else:
                    blocked_calls.append(tc)
            if blocked_calls:
                required_mode = _infer_required_mode(blocked_calls)
                mode_locked = bool(state.get("mode_locked", False))
                blocked_names = [tc.get("name") for tc in blocked_calls]
                if agent.console:
                    agent.console.print(
                        f"[yellow]🚫 Mode {current_mode.value.upper()}: blocked tools {blocked_names}[/yellow]"
                    )

                # AUTO-unlocked flow: switch mode automatically and continue this execution turn.
                if (not mode_locked) and required_mode and required_mode != current_mode:
                    state_updates["current_mode"] = required_mode
                    state_updates["mode_iteration_count"] = 0
                    if agent.console:
                        agent.console.print(
                            f"[cyan]🔄 Auto mode transition: {current_mode.value.upper()} → {required_mode.value.upper()}[/cyan]"
                        )
                    current_mode = required_mode

                    allowed_calls = []
                    still_blocked = []
                    for tc in tool_calls:
                        name = tc.get("name", "")
                        if is_tool_allowed(name, current_mode):
                            allowed_calls.append(tc)
                        else:
                            still_blocked.append(tc)
                    blocked_calls = still_blocked

                # Locked mode flow: ask human whether to unlock and switch.
                if blocked_calls and mode_locked and required_mode and required_mode != current_mode:
                    decision = interrupt(
                        {
                            "type": "mode_switch_request",
                            "current_mode": current_mode.value,
                            "suggested_mode": required_mode.value,
                            "blocked_tools": blocked_names,
                            "reason": (
                                f"{current_mode.value.upper()} 모드에서 요청 도구가 차단되었습니다. "
                                f"{required_mode.value.upper()} 모드로 전환할까요?"
                            ),
                        }
                    )
                    if _is_approved_decision(decision):
                        state_updates["current_mode"] = required_mode
                        state_updates["mode_locked"] = False
                        state_updates["mode_iteration_count"] = 0
                        if agent.console:
                            agent.console.print(
                                f"[cyan]🔄 User-approved mode switch: {current_mode.value.upper()} → {required_mode.value.upper()}[/cyan]"
                            )
                        current_mode = required_mode

                        allowed_calls = []
                        still_blocked = []
                        for tc in tool_calls:
                            name = tc.get("name", "")
                            if is_tool_allowed(name, current_mode):
                                allowed_calls.append(tc)
                            else:
                                still_blocked.append(tc)
                        blocked_calls = still_blocked

                # Return error ToolMessages for remaining blocked calls
                error_msgs = []
                for tc in blocked_calls:
                    error_msgs.append(ToolMessage(
                        content=f"Error: Tool '{tc.get('name')}' is not available in {current_mode.value.upper()} mode. "
                        f"Switch to an appropriate mode first.",
                        tool_call_id=tc.get("id", ""),
                    ))
                if error_msgs and not allowed_calls:
                    return merge_updates(state_updates, {"messages": error_msgs})
                tool_calls = allowed_calls

        # Deduplicate tool calls (remove identical tool calls from the same response)
        tool_calls = deduplicate_tool_calls(tool_calls, workspace_path, agent.console)

        # Update the message with normalized and deduplicated tool calls
        if tool_calls and hasattr(last_message, 'tool_calls'):
            last_message = last_message.copy()
            last_message.tool_calls = tool_calls
            state = dict(state)
            state["messages"] = list(state["messages"][:-1]) + [last_message]

        num_tools = len(tool_calls)

        # PRE-OPERATION CHECKPOINT: Create checkpoint before risky operations
        if tool_calls and hasattr(agent, 'backtracking') and agent.backtracking:
            needs_checkpoint, operation_type = should_create_checkpoint(tool_calls)
            if needs_checkpoint:
                target_files = extract_target_files(tool_calls, workspace_path)
                try:
                    agent.backtracking.create_pre_operation_checkpoint(
                        state=state,
                        operation_type=operation_type,
                        target_files=target_files if target_files else [workspace_path]
                    )
                    if agent.console and agent.verbose:
                        agent.console.print(
                            f"[dim cyan]📌 Pre-{operation_type} checkpoint created[/dim cyan]"
                        )
                except Exception as e:
                    # Don't block execution if checkpoint fails
                    if agent.console and agent.verbose:
                        agent.console.print(
                            f"[dim yellow]⚠️ Checkpoint creation failed: {e}[/dim yellow]"
                        )

        # Suppress file watcher for agent-owned writes
        _watcher_marked_files: list[str] = []
        if hasattr(agent, 'file_watcher') and agent.file_watcher and agent.file_watcher.is_running:
            write_tools = {'file_write', 'file_edit', 'patch_file'}
            for tc in tool_calls:
                if tc.get('name') in write_tools:
                    fpath = tc.get('args', {}).get('file_path') or tc.get('args', {}).get('path', '')
                    if fpath:
                        agent.file_watcher.mark_agent_write(fpath)
                        _watcher_marked_files.append(fpath)

        # Check for repetitive tool calls
        if tool_calls:
            first_tool = tool_calls[0]
            tool_name = first_tool.get('name', 'unknown')
            tool_args = first_tool.get('args', {})
            is_repetitive = agent.recursion_detector.add_action(tool_name, tool_args)
            if is_repetitive:
                repetition_info = agent.recursion_detector.get_repetition_info()
                if repetition_info:
                    if agent.console:
                        warning = (
                            f"\n⚠️  [bold yellow]Repetition Detected![/bold yellow]\n"
                            f"   Tool '{repetition_info['tool_name']}' called {repetition_info['count']} "
                            f"times in last {repetition_info['window_size']} actions.\n"
                            f"   💡 Consider simplifying the request or breaking it into steps.\n"
                        )
                        agent.console.print(warning)
                    agent.logger.log_trace("recursion_warning", repetition_info)
                    # Record in state so iteration_guard can act on it
                    state_updates = merge_updates(state_updates, {
                        "repetition_detected": True,
                        "repetition_info": repetition_info,
                    })

        # Update status indicator for tool execution.
        # For long-running shell tools (e.g., pyinstaller), spinner frames and
        # streaming logs can interleave and produce garbled lines.
        if hasattr(agent, 'status_indicator') and agent.status_indicator and tool_calls:
            tool_names_for_status = [tc.get('name', 'unknown') for tc in tool_calls]
            has_streaming_shell = any(
                name in {"bash_execute", "bash_background", "bash_output"}
                for name in tool_names_for_status
            )
            if has_streaming_shell:
                agent.status_indicator.stop()
            elif num_tools == 1:
                agent.status_indicator.update_for_tool(tool_names_for_status[0])
            else:
                agent.status_indicator.update(f"Executing {num_tools} tools...")

        # Print execution info - highlight parallel execution (Claude Code pattern)
        if agent.console and tool_calls:
            tool_names = [tc.get('name', 'unknown') for tc in tool_calls]
            if num_tools == 1:
                color = "cyan" if tool_names[0] == "bash_execute" else "yellow"
                agent.console.print(f"[{color}]⚙️  Executing tool: {tool_names[0]}...[/{color}]")
            else:
                # Highlight parallel execution - this is the Claude Code pattern
                agent.console.print(f"[bold green]🚀 PARALLEL: Executing {num_tools} tools simultaneously...[/bold green]")
                for i, name in enumerate(tool_names, 1):
                    agent.console.print(f"[dim]   {i}. {name}[/dim]")

        # Step logger: tool invocations
        if hasattr(agent, 'step_logger') and agent.step_logger and tool_calls:
            for tc in tool_calls:
                tc_name = tc.get('name', 'unknown')
                tc_args = tc.get('args', {})
                if tc_name == 'think':
                    agent.step_logger.log_think(
                        tc_args.get('category', 'general'),
                        tc_args.get('thought', ''),
                    )
                else:
                    summary = _build_tool_context_summary(tc_name, tc_args)
                    agent.step_logger.log_tool(tc_name, summary)

        # Cache management
        cache_hits: list[tuple] = []
        cache_misses = []
        tool_results_ordered: list[Any] = [None] * num_tools
        cached_positions: set = set()

        for idx, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('args', {})
            canonical_args = canonicalize_args(tool_args or {}, workspace_path)

            if tool_name in agent.cacheable_tools:
                cached_result = agent.tool_cache.get(tool_name, canonical_args)
                is_stale = cache_entry_is_stale(cached_result, tool_name, canonical_args) if cached_result else False
                if cached_result is not None and not is_stale:
                    cache_hits.append((idx, tool_call, cached_result))
                    if agent.console:
                        agent.console.print(f"[dim cyan]💾 Cache hit for {tool_name}[/dim cyan]")
                else:
                    if is_stale:
                        agent.tool_cache.invalidate(tool_name)
                        if agent.console:
                            agent.console.print(f"[dim yellow]Cache invalidated for {tool_name} (stale input detected)[/dim yellow]")
                    cache_misses.append(tool_call)
            else:
                cache_misses.append(tool_call)

        # Execute tools with retry logic
        start_time = time.time()
        tool_tracker.set_current_state(state)
        execution_error = None
        result: dict[str, Any] = {"messages": []}  # Safe default
        try:
            if cache_misses:
                temp_state = state.copy()
                temp_last_message = last_message.copy()
                temp_last_message.tool_calls = cache_misses
                temp_state["messages"] = state["messages"][:-1] + [temp_last_message]

                # Execute with exponential backoff retry
                max_tool_retries = 2  # Maximum retry attempts for tool execution
                last_exception = None

                for attempt in range(1, max_tool_retries + 2):
                    try:
                        result = tool_node.invoke(temp_state)
                        break  # Success, exit retry loop
                    except Exception as e:
                        last_exception = e
                        error_context = ErrorRecoveryStrategy.create_error_context(e, attempt)

                        # Check if we should retry
                        if error_context.can_retry and attempt <= max_tool_retries:
                            if agent.console:
                                agent.console.print(
                                    f"[yellow]⚠️ Tool execution error (attempt {attempt}/{max_tool_retries + 1}): "
                                    f"{str(e)[:80]}[/yellow]"
                                )
                                if error_context.backoff_seconds > 0:
                                    agent.console.print(
                                        f"[dim]   Retrying in {error_context.backoff_seconds:.1f}s...[/dim]"
                                    )

                            # Wait with exponential backoff
                            if error_context.backoff_seconds > 0:
                                time.sleep(error_context.backoff_seconds)
                            continue

                        # No more retries - handle final error
                        execution_error = str(e)
                        if agent.console:
                            agent.console.print(f"[red]❌ Tool execution failed after {attempt} attempts: {execution_error[:100]}[/red]")

                        # Create error messages for each failed tool
                        result = {
                            "messages": [
                                ToolMessage(
                                    content=f"Error: Tool execution failed - {execution_error[:200]}",
                                    tool_call_id=tc.get('id', f'error_{i}')
                                )
                                for i, tc in enumerate(cache_misses)
                            ]
                        }
                        # Record error in state
                        err_delta = state_helpers.record_error(
                            state,
                            message=f"Tool execution failed: {execution_error[:200]}",
                            level=ErrorLevel.ERROR,
                            source="tool_executor",
                            context={
                                "attempts": attempt,
                                "error_category": error_context.category.value,
                                "suggested_action": error_context.suggested_action
                            },
                            return_delta=True
                        )
                        state_updates = merge_updates(state_updates, err_delta)
                        break

                # Cache results (only if no execution error)
                if not execution_error:
                    # Build id→message map for safe matching
                    result_msgs = result.get('messages', [])
                    result_msg_by_id = {
                        getattr(m, 'tool_call_id', None): m for m in result_msgs
                    }
                    for tool_call in cache_misses:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        tc_id = tool_call.get('id')
                        canonical_args = canonicalize_args(tool_args or {}, workspace_path)
                        if tool_name in agent.cacheable_tools:
                            tool_msg = result_msg_by_id.get(tc_id)
                            if tool_msg is not None:
                                tool_result = getattr(tool_msg, 'content', '')
                                if not (isinstance(tool_result, str) and tool_result.startswith('Error:')):
                                    cache_entry = make_cache_entry(tool_result, tool_name, canonical_args)
                                    agent.tool_cache.set(tool_name, canonical_args, cache_entry)
            else:
                result = {"messages": []}
        finally:
            # Unmark file watcher suppression (must be in finally to avoid leaked marks)
            if _watcher_marked_files and hasattr(agent, 'file_watcher') and agent.file_watcher:
                for fpath in _watcher_marked_files:
                    agent.file_watcher.unmark_agent_write(fpath)

            pending_deltas = tool_tracker.flush_pending_deltas()
            state_updates = merge_updates(state_updates, agent._collapse_deltas(pending_deltas))
            tool_tracker.clear_current_state()

        # Handle cache hits
        if cache_hits:
            for idx, tool_call, cached_result in cache_hits:
                tool_msg = ToolMessage(
                    content=safe_load_cache_value(cached_result),
                    tool_call_id=tool_call.get('id', 'cached')
                )
                tool_results_ordered[idx] = tool_msg
                cached_positions.add(idx)

        total_duration = time.time() - start_time
        success = True

        # Merge cache misses into ordered slots (match by tool_call_id for safety)
        result_messages = result.get("messages", [])
        result_by_id = {
            getattr(msg, 'tool_call_id', None): msg
            for msg in result_messages
            if hasattr(msg, 'tool_call_id')
        }
        matched_ids: set = set()
        for idx, tc in enumerate(tool_calls):
            if tool_results_ordered[idx] is None:
                tc_id = tc.get('id')
                matched = result_by_id.get(tc_id) if tc_id else None
                if matched is not None:
                    tool_results_ordered[idx] = matched
                    matched_ids.add(tc_id)
        # Fallback: fill remaining slots with unmatched messages in order
        unmatched_msgs = (
            msg for msg in result_messages
            if getattr(msg, 'tool_call_id', None) not in matched_ids
        )
        for idx in range(len(tool_calls)):
            if tool_results_ordered[idx] is None:
                tool_results_ordered[idx] = next(unmatched_msgs, None)

        # Rebuild result messages from ordered slots (includes both cache hits and misses)
        result["messages"] = [msg for msg in tool_results_ordered if msg is not None]

        # Record tool call results
        # For non-cached tools, split total duration evenly (individual timing
        # is unavailable because ToolNode executes them as a batch).
        non_cached_count = sum(1 for i in range(num_tools) if i not in cached_positions)
        per_tool_duration = (total_duration / non_cached_count) if non_cached_count > 0 else 0.0

        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('args', {})
            tool_msg = tool_results_ordered[i]
            is_cached = i in cached_positions
            tool_result = getattr(tool_msg, 'content', '') if tool_msg else None
            error = None

            if isinstance(tool_result, str) and tool_result.startswith('Error:'):
                success = False
                error = tool_result

            duration = 0.0 if is_cached else per_tool_duration
            delta = state_helpers.record_tool_call(
                state,
                tool_name=tool_name,
                args=tool_args,
                result=str(tool_result)[:500] if tool_result else None,
                duration=duration,
                success=error is None,
                error=error,
                tokens_used=0,
                cached=is_cached,
                return_delta=True
            )
            state_updates = merge_updates(state_updates, delta)

            if error:
                err_delta = state_helpers.record_error(
                    state,
                    message=f"Tool '{tool_name}' failed: {error[:200]}",
                    level=ErrorLevel.WARNING,
                    source="tool",
                    context={"tool": tool_name, "args": tool_args},
                    return_delta=True
                )
                state_updates = merge_updates(state_updates, err_delta)

            if hasattr(agent, 'step_logger') and agent.step_logger:
                ok, decision_hint = _extract_tool_decision_hint(
                    tool_name,
                    tool_args,
                    tool_result,
                    error,
                )
                agent.step_logger.log_tool_result(
                    tool_name,
                    success=ok,
                    decision_hint=decision_hint,
                )

        # Print completion status
        if agent.console and tool_calls:
            if num_tools == 1:
                status = "✓" if success else "✗"
                color = "green" if success else "red"
                agent.console.print(f"[{color}]{status} Tool completed in {total_duration:.2f}s[/{color}]")
            else:
                success_count = 0
                for idx, _ in enumerate(tool_calls):
                    msg_list = result.get('messages', [{}])
                    if idx < len(msg_list):
                        content = getattr(msg_list[idx], "content", "")
                        if not (isinstance(content, str) and content.startswith('Error:')):
                            success_count += 1
                agent.console.print(f"[green]✓ {success_count}/{num_tools} tools completed in {total_duration:.2f}s[/green]")

        if num_tools > 0:
            state_updates = merge_updates(state_updates, {"plan_execution_pending": False})

        # File change tracking - detect changes after execution
        files_after = agent._snapshot_workspace()
        file_changes = detect_file_changes(files_before, files_after)

        if file_changes["added"] or file_changes["modified"]:
            total_changes = len(file_changes["added"]) + len(file_changes["modified"])
            current_file_changes = state.get("file_changes_count", 0)

            state_updates = merge_updates(state_updates, {
                "file_changes_count": current_file_changes + total_changes,
                "modified_files": state.get("modified_files", []) + file_changes["added"] + file_changes["modified"]
            })

            # Auto-resolve unresolved errors when files are changed (break EXEC→CODE loop)
            error_history = state.get("error_history", [])
            if error_history:
                resolved_count = 0
                for err in error_history:
                    if hasattr(err, 'resolved') and not err.resolved:
                        err.resolved = True
                        err.resolution = "Files modified — assuming fix attempt"
                        resolved_count += 1
                if resolved_count > 0 and agent.console and agent.verbose:
                    agent.console.print(
                        f"[dim cyan]✓ Auto-resolved {resolved_count} error(s) after file changes[/dim cyan]"
                    )

            if agent.console and agent.verbose:
                if file_changes["added"]:
                    agent.console.print(f"[dim green]📝 Files added: {', '.join(file_changes['added'][:5])}[/dim green]")
                if file_changes["modified"]:
                    agent.console.print(f"[dim blue]✏️  Files modified: {', '.join(file_changes['modified'][:5])}[/dim blue]")

        # Python syntax verification for modified .py files
        # Runs OUTSIDE the file_changes block to also catch changes missed by snapshot
        # Catches indentation corruption, missing colons, etc. immediately
        # Use both file change tracking AND direct tool call args for robustness
        changed_py: set[str] = set()
        if file_changes["added"] or file_changes["modified"]:
            changed_py.update(
                f for f in (file_changes["added"] + file_changes["modified"])
                if f.endswith(".py")
            )
        # Also extract .py files directly from file_edit/file_write tool args
        for tc in tool_calls:
            if tc.get("name") in ("file_edit", "file_write", "apply_patch"):
                fpath = tc.get("args", {}).get("file_path") or tc.get("args", {}).get("path", "")
                if fpath and fpath.endswith(".py"):
                    changed_py.add(fpath)

        if changed_py:
            syntax_errors: list[str] = []
            for py_file in changed_py:
                try:
                    abs_path = py_file if os.path.isabs(py_file) else os.path.join(workspace_path, py_file)
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                        source = fh.read()
                    compile(source, py_file, "exec")
                except SyntaxError as e:
                    err_detail = f"{py_file}:{e.lineno}: {e.msg}"
                    if e.text:
                        err_detail += f" → {e.text.strip()}"
                    syntax_errors.append(err_detail)
                except (OSError, UnicodeDecodeError):
                    pass  # File unreadable — skip silently

            if syntax_errors:
                warning_content = (
                    "⚠️ SYNTAX ERROR detected in modified file(s):\n"
                    + "\n".join(f"  - {err}" for err in syntax_errors)
                    + "\n\nThe file(s) have Python syntax errors (likely indentation or "
                    "structural issues). Please fix them before proceeding."
                )
                # Append as a ToolMessage so the LLM sees the warning
                syntax_warning_msg = ToolMessage(
                    content=warning_content,
                    tool_call_id=tool_calls[-1].get("id", "") if tool_calls else "",
                )
                result_msgs = result.get("messages", [])
                result_msgs.append(syntax_warning_msg)
                result["messages"] = result_msgs

                if agent.console:
                    agent.console.print(
                        f"[bold red]⚠️ Syntax errors in {len(syntax_errors)} file(s) — "
                        f"LLM will be notified[/bold red]"
                    )

        return merge_updates(state_updates, result)

    return enhanced_tool_node


__all__ = [
    'create_enhanced_tool_node',
    'normalize_tool_calls',
    'deduplicate_tool_calls',
    'extract_target_files',
    'should_create_checkpoint',
    'PARAM_ALIASES',
    'RISKY_TOOLS',
]
