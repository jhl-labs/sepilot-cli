"""Utility functions for ReactAgent.

This module contains stateless helper functions extracted from base_agent.py.
These functions don't depend on agent state and can be used independently.
"""

import logging
import re
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length, adding ... if truncated."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_tool_args(args: dict[str, Any], max_length: int = 150) -> str:
    """Format tool arguments for display, truncating long values."""
    if not args:
        return "{}"

    formatted_parts = []
    for key, value in args.items():
        value_str = str(value)
        # Show escaped version for strings with newlines/tabs for clarity
        if '\n' in value_str or '\t' in value_str:
            display_str = value_str.replace('\n', '\\n').replace('\t', '\\t')
            if len(display_str) > max_length:
                display_str = display_str[:max_length] + "..."
            formatted_parts.append(f"{key}={repr(display_str)}")
        else:
            if len(value_str) > max_length:
                value_str = value_str[:max_length] + "..."
            formatted_parts.append(f"{key}={value_str}")

    return ", ".join(formatted_parts)


def merge_updates(base: dict[str, Any], delta: dict[str, Any] | None) -> dict[str, Any]:
    """Merge LangGraph state deltas that might contain overlapping keys.

    Returns a new dict; neither *base* nor *delta* are mutated.
    """
    if not delta:
        return base
    result = dict(base)
    for key, value in delta.items():
        if key not in result:
            result[key] = value
            continue
        current = result[key]
        if isinstance(current, list) and isinstance(value, list):
            result[key] = current + value
        elif isinstance(current, dict) and isinstance(value, dict):
            merged = current.copy()
            merged.update(value)
            result[key] = merged
        else:
            result[key] = value
    return result


def canonicalize_args(args: dict[str, Any], cwd: str) -> dict[str, Any]:
    """Normalize tool args for stable cache keys."""
    normalized: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if isinstance(value, Path):
            normalized[key] = value.as_posix()
        else:
            normalized[key] = value
    normalized.setdefault("cwd", cwd)
    return normalized


def extract_paths_from_args(args: dict[str, Any]) -> list[Path]:
    """Collect candidate file paths from common arg names."""
    import os
    paths: list[Path] = []
    for key in ("path", "file_path", "root", "directory"):
        val = args.get(key)
        if isinstance(val, str):
            p = Path(val)
            if not p.is_absolute():
                p = Path(os.getcwd()) / p
            paths.append(p)
    return paths


def cache_entry_is_stale(entry: Any, tool_name: str, args: dict[str, Any]) -> bool:
    """Detect stale cache entries by comparing stored mtimes."""
    if not isinstance(entry, dict):
        return False
    meta = entry.get("meta") or {}
    tracked = meta.get("mtimes", {})
    if not tracked:
        return False
    for path_str, prev_mtime in tracked.items():
        try:
            current_mtime = Path(path_str).stat().st_mtime
        except FileNotFoundError:
            return True
        if current_mtime != prev_mtime:
            return True
    return False


def make_cache_entry(result: Any, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Wrap cacheable result with minimal freshness metadata."""
    mtimes: dict[str, float] = {}
    for path in extract_paths_from_args(args):
        if path.exists() and path.is_file():
            try:
                mtimes[path.as_posix()] = path.stat().st_mtime
            except OSError:
                continue
    return {"value": result, "meta": {"tool": tool_name, "mtimes": mtimes}}


def safe_load_cache_value(entry: Any) -> Any:
    """Unwrap cache value while tolerating legacy formats."""
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def normalize_approval_response(response: Any) -> dict[str, Any]:
    """Normalize human approval responses into a consistent structure."""
    if isinstance(response, dict):
        decision = response.get("decision") or response.get("status") or "approve"
        decision = str(decision).lower()
        if decision in ("approve", "approved", "ok", "yes", "y"):
            return {"status": "approved"}
        if decision in ("deny", "denied", "no", "n", "reject"):
            reason = response.get("reason") or response.get("message")
            return {"status": "deny", "reason": reason}
        if decision in ("respond", "feedback", "message"):
            return {
                "status": "feedback",
                "message": response.get("message") or response.get("reason", "")
            }
    return {"status": "approved"}


def parse_plan_steps(plan_text: str) -> list[str]:
    """Parse numbered plan steps from plan text.

    Extracts lines that start with numbers (e.g., "1. ", "2.", "3)", etc.)

    Args:
        plan_text: The full plan text from LLM

    Returns:
        List of individual step strings
    """
    steps = []
    lines = plan_text.split('\n')
    for line in lines:
        line = line.strip()
        # Match lines starting with number followed by . or ) or :
        if re.match(r'^\d+[\.\)\:]', line):
            steps.append(line)
    return steps


def looks_like_plan(text: str) -> bool:
    """Check if text looks like an execution plan."""
    if not text:
        return False
    lowered = text.lower()
    tool_keywords = [
        "find_file",
        "file_read",
        "file_edit",
        "file_write",
        "bash_execute",
        "codebase",
        "search_content",
        "nvidia-smi",
    ]
    keyword_hits = sum(1 for kw in tool_keywords if kw in lowered)
    has_numbered_steps = bool(re.search(r"\b\d+[\.)]", text))
    return keyword_hits >= 2 and has_numbered_steps


def is_plan_request(text: str) -> bool:
    """Return True when the user explicitly asks for a plan-only response."""
    if not text:
        return False
    lowered = text.lower()

    explicit_plan = "plan" in lowered or "roadmap" in lowered
    plan_only_constraints = [
        "before writing any code",
        "before writing code",
        "do not write any code",
        "do not write code",
        "only output the plan",
        "only provide the plan",
        "plan should include",
        "implementation plan",
        "step-by-step implementation order",
    ]
    return explicit_plan and any(kw in lowered for kw in plan_only_constraints)


def is_detailed_plan_response(text: str, require_testing: bool = False) -> bool:
    """Heuristic quality check for plan-only responses."""
    if not text or len(text.strip()) < 120:
        return False

    lowered = text.lower()
    numbered_steps = re.findall(r"^\s*\d+[\.)]\s+", text, re.MULTILINE)
    bullet_steps = re.findall(r"^\s*[-*•]\s+", text, re.MULTILINE)

    required_topics = [
        "structure",
        "directory",
        "file",
        "model",
        "endpoint",
        "route",
        "api",
        "step",
        "phase",
    ]
    topic_hits = sum(1 for topic in required_topics if topic in lowered)

    if len(numbered_steps) + len(bullet_steps) < 3 and topic_hits < 4:
        return False

    if require_testing:
        testing_keywords = ("test", "testing", "pytest", "verification", "qa")
        if not any(kw in lowered for kw in testing_keywords):
            return False

    return True


def extract_token_usage(response: Any) -> int:
    """Extract token usage from LLM response message."""
    tokens_used = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        tokens_used = usage.get('total_tokens', 0)
        if tokens_used == 0:
            tokens_used = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
    elif hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if 'token_usage' in metadata:
            token_usage = metadata['token_usage']
            tokens_used = token_usage.get('total_tokens', 0)
    return tokens_used or 0


def extract_required_files(prompt: str) -> list[str]:
    """Extract file references from user prompt.

    Only extracts files that are:
    1. Surrounded by backticks, quotes, or @ symbols
    2. Part of a clear file path (contains /)
    3. Explicitly mentioned file types (README, Dockerfile, etc.)

    This prevents extracting random words from code/documentation.
    """
    if not prompt:
        return []

    matches = set()

    # Pattern 1: Files with path separators (e.g., src/main.py, @path/to/file.py)
    path_pattern = re.compile(
        r'[@`"\']?([A-Za-z0-9_-]+/[A-Za-z0-9_/.-]+\.(?:py|md|txt|json|yaml|yml|ini|cfg|sh|js|ts|tsx|jsx|java|go|rs|c|cpp|h|hpp))[@`"\']?',
        re.IGNORECASE
    )
    for match in path_pattern.findall(prompt):
        cleaned = match.strip("`'\"@ ")
        if cleaned and '/' in cleaned:  # Must have path separator
            matches.add(cleaned)

    # Pattern 2: Files explicitly marked with backticks or quotes (e.g., `file.py`, "config.json")
    quoted_pattern = re.compile(
        r'[`"\']([A-Za-z0-9_/.-]+\.(?:py|md|txt|json|yaml|yml|ini|cfg|sh|js|ts|tsx|jsx|java|go|rs|c|cpp|h|hpp))[`"\']',
        re.IGNORECASE
    )
    for match in quoted_pattern.findall(prompt):
        cleaned = match.strip()
        if cleaned:
            matches.add(cleaned)

    # Pattern 3: Special files (README, Dockerfile, Makefile)
    if "README" in prompt and not any("readme" in m.lower() for m in matches):
        matches.add("README.md")

    return sorted(matches)


def path_matches_requirement(changed_path: str, requirement: str) -> bool:
    """Check if a changed file path matches a requirement."""
    if not changed_path or not requirement:
        return False
    changed = Path(changed_path)
    req_path = Path(requirement)
    changed_name = changed.name.lower()
    req_name = req_path.name.lower()
    return (
        changed_path.endswith(requirement)
        or changed_name == req_name
        or req_name in changed_name
    )


def should_skip_planning(user_prompt: str) -> bool:
    """Heuristic to skip planning when user already supplied detailed steps/code."""
    if not user_prompt:
        return False
    if "```" in user_prompt:
        return True
    if len(user_prompt) > 2000:
        return True
    keywords = [
        "코드를 다음과 같이",
        "apply the following code",
        "다음 파일",
        "수정해줘",
        "patch",
        "diff",
    ]
    return any(keyword in user_prompt for keyword in keywords)


def check_task_completion(
    llm: Any,
    user_query: str,
    tool_result: str,
    tool_name: str,
    iteration_count: int = 0,
    max_iterations: int = 15,
    has_file_changes: bool = False,
    file_change_summary: str = "",
) -> dict[str, Any]:
    """Use LLM to determine if the user's task is complete.

    This is the agent-level approach: let the LLM judge completion
    rather than using string matching or hardcoded rules.

    Args:
        llm: LangChain LLM instance (without tools bound).
        user_query: Original user request.
        tool_result: Result from tool execution.
        tool_name: Name of the executed tool.
        iteration_count: Current iteration number.
        max_iterations: Maximum allowed iterations.
        has_file_changes: Whether any files were modified in this session.
        file_change_summary: Summary of file changes for context.

    Returns:
        dict with:
            - complete: bool indicating if task is done
            - reason: explanation of the decision
            - confidence: float 0-1 indicating confidence
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Safety: force completion after max iterations
    if iteration_count >= max_iterations:
        return {
            "complete": True,
            "reason": "max_iterations_reached",
            "confidence": 1.0
        }

    # Truncate long content for efficiency
    query_preview = user_query[:500] if user_query else ""
    result_preview = tool_result[:1500] if tool_result else ""

    file_change_section = file_change_summary if file_change_summary else "No file changes recorded."

    completion_prompt = f"""You are a task completion evaluator for a coding assistant.

Analyze whether the user's request has been fulfilled based on the tool execution result.

USER REQUEST:
{query_preview}

TOOL EXECUTED: {tool_name}

TOOL RESULT:
{result_preview}

FILE CHANGES:
{file_change_section}

EVALUATION CRITERIA:
1. Does the tool result directly answer the user's question?
2. Does the result contain the information the user asked for?
3. Is this a simple query (lookup, search, status check) or a complex task (modification, multi-step)?
4. For coding/modification tasks: Were actual file changes made?
   - If the task requires code modification but NO files were changed → INCOMPLETE
   - If files were changed, do the changes appear relevant to the request?

For SIMPLE QUERIES (find, search, list, check, show, status):
- If the result contains the requested information → COMPLETE
- One tool execution is usually sufficient

For COMPLEX TASKS (fix, implement, modify, refactor, create):
- Multiple tool executions may be needed
- Check if the task goal has been achieved
- File changes are expected for code modification tasks

RESPOND WITH EXACTLY ONE LINE in this format:
COMPLETE|reason|confidence
or
INCOMPLETE|reason|confidence

Examples:
- COMPLETE|user asked for zombie processes and result shows process list|0.95
- INCOMPLETE|user asked to fix a bug but no file was modified yet|0.85
- COMPLETE|user asked to run a command and command executed successfully|0.90
- INCOMPLETE|task requires code changes but no files were modified|0.90

Your evaluation:"""

    try:
        from sepilot.agent.output_validator import OutputValidator

        original_messages = [
            SystemMessage(content="You are a precise task completion evaluator. Respond only with the requested format."),
            HumanMessage(content=completion_prompt)
        ]
        response = llm.invoke(original_messages)

        answer = response.content.strip()

        # Validate pipe format with OutputValidator
        is_valid, parts = OutputValidator.validate_pipe_format(answer, expected_parts=3)
        if is_valid:
            status = parts[0].strip().upper()
            reason = parts[1].strip()
            try:
                confidence = float(parts[2].strip())
            except ValueError:
                confidence = 0.7

            return {
                "complete": status == "COMPLETE",
                "reason": reason,
                "confidence": min(max(confidence, 0.0), 1.0)
            }

        # Format invalid — retry once with correction prompt
        corrected = OutputValidator.retry_with_correction(
            llm=llm,
            original_messages=original_messages,
            original_response=answer,
            error_desc="Expected format: COMPLETE|reason|confidence or INCOMPLETE|reason|confidence",
            max_retries=1,
        )
        if corrected:
            is_valid2, parts2 = OutputValidator.validate_pipe_format(corrected, expected_parts=3)
            if is_valid2:
                status = parts2[0].strip().upper()
                reason = parts2[1].strip()
                try:
                    confidence = float(parts2[2].strip())
                except ValueError:
                    confidence = 0.7
                return {
                    "complete": status == "COMPLETE",
                    "reason": reason,
                    "confidence": min(max(confidence, 0.0), 1.0)
                }

        # Final fallback parsing
        is_complete = "COMPLETE" in answer.upper() and "INCOMPLETE" not in answer.upper()
        return {
            "complete": is_complete,
            "reason": answer[:100],
            "confidence": 0.6
        }

    except Exception as e:
        # On error, be conservative - continue if early iteration, complete if late
        return {
            "complete": iteration_count >= 3,
            "reason": f"evaluation_error: {str(e)[:50]}",
            "confidence": 0.5
        }


def build_file_change_summary(
    file_changes: list[str],
    modified_files: list[str],
    max_entries: int = 10,
) -> str:
    """Build a concise summary of file changes for completion evaluation."""
    all_files = list(set(file_changes) | set(modified_files))
    if not all_files:
        return "NO FILE CHANGES have been made in this session."
    summary_lines = [f"FILES MODIFIED ({len(all_files)}):"]
    for f in all_files[:max_entries]:
        summary_lines.append(f"  - {f}")
    if len(all_files) > max_entries:
        summary_lines.append(f"  ... and {len(all_files) - max_entries} more")
    return "\n".join(summary_lines)


def verify_patch_quality(
    llm: Any,
    user_query: str,
    file_diffs: str,
    modified_files: list[str],
) -> dict[str, Any]:
    """LLM-based patch self-review before declaring task complete.

    Asks the LLM to review the actual code changes (diffs) against the
    original problem statement and flag potential issues like:
    - Indentation corruption
    - Incomplete fixes (partial changes)
    - Changes unrelated to the problem
    - Missing edge cases mentioned in the issue

    Args:
        llm: LangChain LLM instance (without tools bound).
        user_query: Original user request / problem statement.
        file_diffs: Git diff or file change summary showing actual changes.
        modified_files: List of modified file paths.

    Returns:
        dict with:
            - approved: bool — True if patch looks correct
            - issues: list of issue descriptions (empty if approved)
            - confidence: float 0-1
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    query_preview = user_query[:1500] if user_query else ""
    diff_preview = file_diffs[:4000] if file_diffs else "No diffs available."
    files_list = ", ".join(modified_files[:10]) if modified_files else "None"

    review_prompt = f"""You are a code review expert. Review the following patch/changes and determine if they correctly address the stated problem.

PROBLEM STATEMENT:
{query_preview}

FILES MODIFIED: {files_list}

CODE CHANGES (diff with surrounding context):
{diff_preview}

REVIEW CRITERIA:
1. Does the patch address the root cause described in the problem?
2. Is the code syntactically correct (proper indentation, no missing brackets/colons)?
3. Are there any obvious side effects or regressions?
4. Is the fix minimal and focused (not over-engineered)?
5. COMPLETENESS CHECK — Is the fix complete? Common incomplete patterns:
   - Adding a parameter to __init__ but not forwarding it to super().__init__()
   - Modifying __eq__ but not updating __hash__ or __lt__ when needed
   - Fixing one code path but missing another that has the same bug
   - Adding a check but not handling the new case properly
6. Are there unnecessary cosmetic changes (docstring reformatting, whitespace) that might break tests?

Respond in this exact format:
VERDICT: APPROVED or NEEDS_FIX
ISSUES: <comma-separated list of issues, or "none">
CONFIDENCE: <0.0-1.0>"""

    try:
        messages = [
            SystemMessage(content="You are a precise code reviewer. Be concise and accurate."),
            HumanMessage(content=review_prompt),
        ]
        response = llm.invoke(messages)
        answer = getattr(response, "content", str(response)).strip()

        # Parse response
        approved = "APPROVED" in answer.upper() and "NEEDS_FIX" not in answer.upper()
        issues: list[str] = []
        confidence = 0.7

        for line in answer.split("\n"):
            line_upper = line.strip().upper()
            if line_upper.startswith("ISSUES:"):
                issues_text = line.strip()[7:].strip()
                if issues_text.lower() not in ("none", "없음", "n/a", ""):
                    issues = [i.strip() for i in issues_text.split(",") if i.strip()]
            elif line_upper.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.strip().split(":")[1].strip())
                    confidence = min(max(confidence, 0.0), 1.0)
                except (ValueError, IndexError):
                    pass

        return {
            "approved": approved,
            "issues": issues,
            "confidence": confidence,
        }
    except Exception as e:
        # On error, don't block — approve by default
        return {
            "approved": True,
            "issues": [f"review_error: {str(e)[:50]}"],
            "confidence": 0.5,
        }


def assess_query_complexity(
    llm: Any,
    user_query: str
) -> dict[str, Any]:
    """Use LLM to assess the complexity of a user query.

    Determines if a query is simple (single tool call sufficient)
    or complex (requires multiple iterations).

    Args:
        llm: LangChain LLM instance.
        user_query: User's request.

    Returns:
        dict with:
            - complexity: 'simple' or 'complex'
            - expected_tools: estimated number of tool calls needed
            - reason: explanation
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    query_preview = user_query[:800] if user_query else ""

    assessment_prompt = f"""Classify this coding assistant request by complexity.

REQUEST:
{query_preview}

SIMPLE requests (1-2 tool calls):
- Find/search/list something
- Check status or information
- Run a single command
- Read a specific file

COMPLEX requests (3+ tool calls):
- Fix a bug (requires: find → read → edit → verify)
- Implement a feature (requires: understand → plan → implement → test)
- Refactor code (requires: analyze → modify multiple files)
- Debug an issue (requires: investigate → identify → fix)

RESPOND WITH ONE LINE:
SIMPLE|expected_tool_count|reason
or
COMPLEX|expected_tool_count|reason

Examples:
- SIMPLE|1|user wants to find zombie processes - single ps command
- COMPLEX|5|user wants to fix a bug - need to find, read, edit, test
- SIMPLE|2|user wants to see git status and diff

Your assessment:"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a request complexity assessor. Be concise."),
            HumanMessage(content=assessment_prompt)
        ])

        answer = response.content.strip()
        parts = answer.split("|")

        if len(parts) >= 3:
            complexity = "simple" if "SIMPLE" in parts[0].upper() else "complex"
            try:
                expected_tools = int(parts[1].strip())
            except ValueError:
                expected_tools = 1 if complexity == "simple" else 5
            reason = parts[2].strip()

            return {
                "complexity": complexity,
                "expected_tools": expected_tools,
                "reason": reason
            }
        else:
            # Default to complex if unclear
            return {
                "complexity": "complex",
                "expected_tools": 5,
                "reason": "unable to parse complexity"
            }

    except Exception as e:
        _logger.debug(f"Query complexity assessment failed: {e}")
        return {
            "complexity": "complex",
            "expected_tools": 5,
            "reason": "assessment_error"
        }


__all__ = [
    'truncate_text',
    'format_tool_args',
    'merge_updates',
    'canonicalize_args',
    'extract_paths_from_args',
    'cache_entry_is_stale',
    'make_cache_entry',
    'safe_load_cache_value',
    'normalize_approval_response',
    'parse_plan_steps',
    'looks_like_plan',
    'is_plan_request',
    'is_detailed_plan_response',
    'extract_token_usage',
    'extract_required_files',
    'path_matches_requirement',
    'should_skip_planning',
    'check_task_completion',
    'assess_query_complexity',
]
