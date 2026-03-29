"""Fallback parser for text-based tool calls from LLM responses.

Some models (deepseek-v3.2, qwen3, devstral-2, etc.) output tool calls as
plain text instead of using OpenAI function calling format. This module
extracts structured tool calls from text so the agent can execute them.

Supports 4 patterns in priority order:
1. XML function_calls: <function_calls><invoke name="..."><parameter name="...">
2. Hermes tool_call:   <tool_call>{"name": ...}</tool_call>
3. JSON in markdown:   ```json {"name": ...} ```
4. Bare JSON:          {"name": "...", "arguments": {...}}
"""

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def try_parse_text_tool_calls(
    content: str, valid_tool_names: set[str]
) -> list[dict]:
    """Parse tool calls from LLM text output.

    Tries each parser in priority order. Returns LangChain-compatible
    tool_call dicts on success, empty list on failure.
    """
    if not content or not content.strip():
        return []

    parsers = [
        _parse_xml_function_calls,
        _parse_hermes_tool_call,
        _parse_json_markdown,
        _parse_bare_json,
    ]

    for parser in parsers:
        try:
            raw_calls = parser(content)
            if raw_calls:
                validated = _validate_and_normalize(raw_calls, valid_tool_names)
                if validated:
                    logger.info(
                        "Fallback parser (%s) found %d tool call(s): %s",
                        parser.__name__,
                        len(validated),
                        [tc["name"] for tc in validated],
                    )
                    return validated
        except Exception:
            logger.debug("Parser %s failed", parser.__name__, exc_info=True)
            continue

    return []


# ---------------------------------------------------------------------------
# Parser 1: XML function_calls
# ---------------------------------------------------------------------------

# Match <invoke name="tool_name"> ... </invoke> blocks
_INVOKE_RE = re.compile(
    r"<invoke\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</invoke>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter\s+name\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</parameter>",
    re.DOTALL,
)


def _parse_xml_function_calls(content: str) -> list[dict]:
    """Parse <function_calls><invoke name="..."><parameter> patterns."""
    invokes = _INVOKE_RE.findall(content)
    if not invokes:
        return []

    results = []
    for name, body in invokes:
        params = _PARAM_RE.findall(body)
        args = {}
        for param_name, param_value in params:
            args[param_name.strip()] = _coerce_value(param_value.strip())
        results.append({"name": name.strip(), "args": args})
    return results


# ---------------------------------------------------------------------------
# Parser 2: Hermes <tool_call> JSON
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def _parse_hermes_tool_call(content: str) -> list[dict]:
    """Parse <tool_call>{"name": ...}</tool_call> patterns (Qwen3/Hermes)."""
    matches = _TOOL_CALL_RE.findall(content)
    if not matches:
        return []

    results = []
    for raw_json in matches:
        parsed = _relaxed_json_parse(raw_json)
        if not parsed or not isinstance(parsed, dict):
            continue
        name = parsed.get("name", "")
        args = parsed.get("arguments") or parsed.get("parameters") or {}
        if name:
            results.append({"name": name, "args": args if isinstance(args, dict) else {}})
    return results


# ---------------------------------------------------------------------------
# Parser 3: JSON in markdown code blocks
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?\s*(\{.*?\})\s*\n?\s*```", re.DOTALL
)


def _parse_json_markdown(content: str) -> list[dict]:
    """Parse ```json {"name": ...} ``` patterns."""
    matches = _JSON_BLOCK_RE.findall(content)
    if not matches:
        return []

    results = []
    for raw_json in matches:
        parsed = _relaxed_json_parse(raw_json)
        if not parsed or not isinstance(parsed, dict):
            # JSON might have literal newlines in string values (invalid JSON
            # but common with CLI agents). Fix by escaping unescaped newlines
            # inside string values.
            fixed = _fix_literal_newlines_in_json(raw_json)
            if fixed != raw_json:
                parsed = _relaxed_json_parse(fixed)
            if not parsed or not isinstance(parsed, dict):
                continue
        name = parsed.get("name", "")
        args = parsed.get("arguments") or parsed.get("parameters") or {}
        if name:
            results.append({"name": name, "args": args if isinstance(args, dict) else {}})
    return results


# ---------------------------------------------------------------------------
# Parser 4: Bare JSON objects
# ---------------------------------------------------------------------------

_BARE_JSON_RE = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"(?:arguments|parameters)"\s*:\s*\{',
)


def _parse_bare_json(content: str) -> list[dict]:
    """Parse bare {"name": "...", "arguments": {...}} patterns."""
    results = []
    # Find potential JSON start positions
    for match in _BARE_JSON_RE.finditer(content):
        start = match.start()
        obj = _extract_json_object(content, start)
        if obj is None:
            continue
        parsed = _relaxed_json_parse(obj)
        if not parsed or not isinstance(parsed, dict):
            continue
        name = parsed.get("name", "")
        args = parsed.get("arguments") or parsed.get("parameters") or {}
        if name:
            results.append({"name": name, "args": args if isinstance(args, dict) else {}})
    return results


def _extract_json_object(text: str, start: int) -> str | None:
    """Extract a balanced JSON object starting at position `start`."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, min(start + 10000, len(text))):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Validation & Normalization
# ---------------------------------------------------------------------------

# Common aliases: model may output a different name for the same tool
_TOOL_ALIASES: dict[str, str] = {
    "edit_file": "file_edit",
    "read_file": "file_read",
    "write_file": "file_write",
    "search_file": "file_search",
    "find_file": "file_find",
    "execute_bash": "bash_execute",
    "run_bash": "bash_execute",
    "bash": "bash_execute",
    "list_files": "file_list",
    "search_code": "code_search",
    "grep": "code_search",
    "find": "file_find",
}


def _validate_and_normalize(
    tool_calls: list[dict], valid_tool_names: set[str]
) -> list[dict]:
    """Validate tool names and normalize to LangChain format.

    Matching strategy (in order):
    1. Exact match
    2. Known alias mapping
    3. Suffix removal (e.g. "file_read_tool" → "file_read")
    4. Case-insensitive match
    """
    valid_lower = {n.lower(): n for n in valid_tool_names}
    result = []

    for tc in tool_calls:
        name = tc.get("name", "").strip()
        args = tc.get("args", {})
        resolved = None

        # 1. Exact match
        if name in valid_tool_names:
            resolved = name
        # 2. Alias
        elif name in _TOOL_ALIASES and _TOOL_ALIASES[name] in valid_tool_names:
            resolved = _TOOL_ALIASES[name]
        # 3. Suffix removal (_tool, _func)
        elif not resolved:
            for suffix in ("_tool", "_func", "_function"):
                stripped = name.removesuffix(suffix)
                if stripped != name and stripped in valid_tool_names:
                    resolved = stripped
                    break
        # 4. Case-insensitive
        if not resolved:
            lower = name.lower()
            if lower in valid_lower:
                resolved = valid_lower[lower]

        if resolved:
            result.append({
                "name": resolved,
                "args": args,
                "id": f"fallback_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            })
        else:
            logger.debug("Unrecognized tool name '%s', skipping", name)

    return result


# ---------------------------------------------------------------------------
# Fix literal newlines in JSON strings
# ---------------------------------------------------------------------------


def _fix_literal_newlines_in_json(raw: str) -> str:
    """Escape literal newlines/tabs inside JSON string values.

    CLI agents sometimes output JSON with real newlines inside string
    values (e.g. file content), which is invalid JSON. This function
    walks the string and escapes unescaped control characters within
    quoted regions.
    """
    result = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if not in_string:
            if ch == '"':
                in_string = True
            result.append(ch)
        else:
            if ch == '\\' and i + 1 < len(raw):
                # Already escaped — pass through
                result.append(ch)
                result.append(raw[i + 1])
                i += 2
                continue
            elif ch == '"':
                in_string = False
                result.append(ch)
            elif ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Relaxed JSON parsing
# ---------------------------------------------------------------------------


def _relaxed_json_parse(raw: str) -> dict | None:
    """Parse JSON with common LLM quirks handled.

    Fixes: trailing commas, single quotes, unquoted keys, Python booleans/None.
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()

    # Try standard parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fix common issues
    fixed = raw

    # Python bool/None → JSON
    fixed = re.sub(r'\bTrue\b', 'true', fixed)
    fixed = re.sub(r'\bFalse\b', 'false', fixed)
    fixed = re.sub(r'\bNone\b', 'null', fixed)

    # Single quotes → double quotes (simple cases)
    # Only do this if there are no double quotes already in key positions
    if "'" in fixed and fixed.count('"') < fixed.count("'"):
        fixed = _single_to_double_quotes(fixed)

    # Trailing commas before } or ]
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

    # Unquoted keys: { key: "value" } → { "key": "value" }
    fixed = re.sub(
        r'(?<=[\{,])\s*([a-zA-Z_]\w*)\s*:',
        r' "\1":',
        fixed,
    )

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        logger.debug("Relaxed JSON parse failed for: %s", raw[:200])
        return None


def _single_to_double_quotes(s: str) -> str:
    """Convert single-quoted JSON strings to double-quoted."""
    result = []
    in_string = False
    quote_char = None
    i = 0
    while i < len(s):
        ch = s[i]
        if not in_string:
            if ch == "'":
                result.append('"')
                in_string = True
                quote_char = "'"
            elif ch == '"':
                result.append('"')
                in_string = True
                quote_char = '"'
            else:
                result.append(ch)
        else:
            if ch == "\\" and i + 1 < len(s):
                result.append(ch)
                result.append(s[i + 1])
                i += 2
                continue
            elif ch == quote_char:
                result.append('"')
                in_string = False
                quote_char = None
            else:
                # Escape unescaped double quotes inside single-quoted strings
                if ch == '"' and quote_char == "'":
                    result.append('\\"')
                else:
                    result.append(ch)
        i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Value coercion for XML parameters
# ---------------------------------------------------------------------------


def _coerce_value(raw: str) -> Any:
    """Convert XML parameter string to appropriate Python type."""
    if not raw:
        return ""

    lower = raw.lower()

    # Boolean
    if lower == "true":
        return True
    if lower == "false":
        return False

    # Null
    if lower in ("null", "none"):
        return None

    # Integer
    try:
        return int(raw)
    except ValueError:
        pass

    # Float
    try:
        return float(raw)
    except ValueError:
        pass

    # JSON object/array embedded in XML param
    if (raw.startswith("{") and raw.endswith("}")) or (
        raw.startswith("[") and raw.endswith("]")
    ):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    return raw
