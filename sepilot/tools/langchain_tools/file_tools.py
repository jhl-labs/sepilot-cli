"""File operation tools for LangChain agent.

Provides file_read, file_write, file_edit tools.
"""

import difflib
import re
import tempfile
from pathlib import Path

from langchain_core.tools import tool


def _normalize_relative_path(path: Path, project_root: Path) -> str:
    """Return path relative to project root when possible."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _preferred_required_file_path(filename: str) -> str | None:
    """If the agent already knows an exact target file, return its canonical path."""
    try:
        from sepilot.agent import tool_tracker
    except Exception:
        return None

    state = tool_tracker.get_current_state()
    if not state:
        return None

    required_files = state.get("required_files") or []
    if not required_files:
        return None

    normalized_name = filename.strip().lower()
    project_root = Path.cwd().resolve()
    candidates: list[Path] = []
    for required in required_files:
        if not required:
            continue
        required_path = Path(str(required).strip())
        required_name = required_path.name.lower()
        if (
            normalized_name == required_name
            or required_name.endswith(normalized_name)
            or (normalized_name and normalized_name in required_name)
            or (filename and filename in str(required_path))
        ):
            candidates.append(required_path)

    ranked: list[tuple] = []
    for candidate in candidates:
        full_path = candidate if candidate.is_absolute() else project_root / candidate
        if not full_path.exists():
            continue
        workspace_bias = -1 if any(part.lower() == "workspaces" for part in full_path.parts) else 0
        ranked.append((workspace_bias, len(str(full_path)), full_path))

    if not ranked:
        return None

    ranked.sort()
    best_path = ranked[0][2]
    return _normalize_relative_path(best_path, project_root)


def _looks_like_diff_snippet(text: str) -> bool:
    """Detect if the provided text appears to be a diff chunk rather than raw code."""
    if not text:
        return False

    stripped = text.lstrip()
    diff_headers = ("diff --", "--- ", "+++ ", "@@")
    if any(stripped.startswith(header) for header in diff_headers):
        return True

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    diff_marker_count = sum(
        1 for line in lines
        if (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    )
    return diff_marker_count >= max(3, len(lines) // 2)


def _check_path_security(file_path: str, allow_read_system: bool = False) -> tuple[bool, str, Path]:
    """Check path security and return (is_valid, error_message, resolved_path)."""
    path = Path(file_path).resolve()
    project_root = Path.cwd().resolve()

    # Check if path is within project root
    try:
        path.relative_to(project_root)
    except ValueError:
        if allow_read_system:
            # Allow reading certain safe system files for debugging
            safe_system_paths = [tempfile.gettempdir(), '/var/log']
            path_str = str(path)
            if not any(path_str.startswith(safe_path) for safe_path in safe_system_paths):
                return False, f"Error: Path escape attempt detected. File path must be within project directory: {file_path}", path
        else:
            return False, f"Error: Path escape attempt detected. File path must be within project directory: {file_path}", path

    # Blacklist sensitive system paths
    forbidden_paths = ['/etc/shadow', '/etc/sudoers', '/root', '/sys', '/proc/kcore'] if allow_read_system else ['/etc', '/usr', '/bin', '/sbin', '/sys', '/proc', '/dev', '/boot', '/root']
    path_str = str(path)
    for forbidden in forbidden_paths:
        if path_str.startswith(forbidden):
            return False, f"Error: Access to {'sensitive ' if allow_read_system else ''}system path denied: {file_path}", path

    return True, "", path


def _check_forbidden_dirs(path: Path, project_root: Path) -> tuple[bool, str]:
    """Check if path is in forbidden directories. Returns (is_allowed, error_message)."""
    path_str = str(path)
    try:
        relative_path_str = str(path.relative_to(project_root))
    except ValueError:
        relative_path_str = path_str

    forbidden_dirs = ['node_modules', 'dist', 'build', 'out', '.git', '__pycache__', '.venv', 'venv']
    for forbidden_dir in forbidden_dirs:
        if f'/{forbidden_dir}/' in f'/{relative_path_str}' or relative_path_str.startswith(f'{forbidden_dir}/'):
            suggestion = ""
            if forbidden_dir == 'node_modules':
                suggestion = "\n💡 SUGGESTION: Don't modify node_modules. Changes are lost on npm install. Find a proper architectural solution (e.g., create an API server, use a different library)."
            elif forbidden_dir in ['dist', 'build', 'out']:
                suggestion = "\n💡 SUGGESTION: Don't modify build output. Edit source files in src/ and rebuild."
            return False, f"Error: Modification of '{forbidden_dir}/' is forbidden - these files are auto-generated or managed by package managers.{suggestion}"

    return True, ""


@tool
def file_read(file_path: str, encoding: str = "utf-8") -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read (e.g., README.md, test.py, config.json)
        encoding: File encoding (default: utf-8)

    Returns:
        File contents with line numbers
    """
    try:
        is_valid, error_msg, path = _check_path_security(file_path, allow_read_system=True)
        if not is_valid:
            return error_msg

        if not path.exists():
            return f"Error: File not found: {file_path}"

        if not path.is_file():
            return f"Error: Not a file: {file_path}"

        # Check file size and read accordingly
        file_size = path.stat().st_size
        max_lines = 1000  # Maximum lines to read for large files

        if file_size > 1_000_000:  # 1MB
            with open(path, encoding=encoding) as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip('\n'))

                content = '\n'.join(lines)
                warning = (
                    f"⚠️  File too large ({file_size / 1_000_000:.2f}MB), "
                    f"showing first {len(lines)} lines only.\n"
                    f"{'─' * 60}\n"
                )
        else:
            with open(path, encoding=encoding) as f:
                content = f.read()
                warning = ""

            lines = content.split('\n')
            if len(lines) > max_lines:
                content = '\n'.join(lines[:max_lines])
                warning = (
                    f"⚠️  File has {len(lines)} lines, "
                    f"showing first {max_lines} only.\n"
                    f"{'─' * 60}\n"
                )

        # Record file read in Enhanced State
        from sepilot.agent.enhanced_state import FileAction
        from sepilot.agent.tool_tracker import record_file_change_if_enabled

        record_file_change_if_enabled(
            file_path=str(file_path),
            action=FileAction.READ,
            old_content=None,
            new_content=content,
            tool_used="file_read"
        )

        # Add line numbers
        lines = content.split('\n')
        numbered_lines = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        result = '\n'.join(numbered_lines)

        # Include file size info
        size_info = f"File size: {file_size:,} bytes"
        if file_size > 1_000_000:
            size_info += f" ({file_size / 1_000_000:.2f}MB)"
        elif file_size > 1000:
            size_info += f" ({file_size / 1000:.2f}KB)"

        return f"Contents of {file_path} ({size_info}):\n{warning}{result}"

    except UnicodeDecodeError:
        return f"Error: Unable to decode file with {encoding} encoding: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def file_write(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """Write content to a file (creates new file or overwrites existing).

    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        encoding: File encoding (default: utf-8)

    Returns:
        Success or error message
    """
    try:
        is_valid, error_msg, path = _check_path_security(file_path, allow_read_system=False)
        if not is_valid:
            return error_msg

        project_root = Path.cwd().resolve()
        is_allowed, error_msg = _check_forbidden_dirs(path, project_root)
        if not is_allowed:
            return error_msg

        # Check if file exists
        file_exists = path.exists() and path.is_file()
        old_content = None

        if file_exists:
            try:
                with open(path, encoding=encoding) as f:
                    old_content = f.read()
            except Exception:
                pass

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)

        # Record file change
        from sepilot.agent.enhanced_state import FileAction
        from sepilot.agent.tool_tracker import (
            is_enhanced_state_enabled,
            record_file_change_if_enabled,
        )

        action = FileAction.MODIFY if (file_exists and old_content is not None) else FileAction.CREATE

        enabled = is_enhanced_state_enabled()
        recorded = record_file_change_if_enabled(
            file_path=str(file_path),
            action=action,
            old_content=old_content,
            new_content=content,
            tool_used="file_write"
        )

        if not recorded:
            import sys
            print(f"[DEBUG] file_write: Enhanced state enabled={enabled}, recorded={recorded}", file=sys.stderr)

        result = f"Successfully wrote {len(content)} characters to {file_path}"

        syntax_warning = _check_python_syntax(path, content)
        if syntax_warning:
            result += f"\n\n{syntax_warning}"

        return result

    except Exception as e:
        return f"Error writing file: {str(e)}"


def _check_python_syntax(path: Path, content: str) -> str:
    """Python 파일의 구문을 검증하고, 오류가 있으면 경고 메시지를 반환."""
    if path.suffix != ".py":
        return ""
    try:
        import ast
        ast.parse(content, filename=str(path))
        return ""
    except SyntaxError as e:
        line_info = f" (line {e.lineno})" if e.lineno else ""
        # 오류 주변 코드 표시
        context_lines = []
        if e.lineno and content:
            lines = content.split("\n")
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            for i in range(start, end):
                marker = " >>>" if i == e.lineno - 1 else "    "
                context_lines.append(f"{marker} {i+1}: {lines[i]}")
        context = "\n".join(context_lines)
        return (
            f"WARNING: Python syntax error after edit{line_info}: {e.msg}\n"
            f"{context}\n"
            f"The file was saved but contains a syntax error. "
            f"Please fix the indentation or syntax with another file_edit call."
        )


@tool
def file_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Edit a file by replacing old_string with new_string (Claude Code style).

    Args:
        file_path: Path to the file to edit
        old_string: String to find and replace
        new_string: String to replace with
        replace_all: If True, replace ALL occurrences. If False (default), require unique match.

    Returns:
        Success or error message
    """
    try:
        is_valid, error_msg, path = _check_path_security(file_path, allow_read_system=False)
        if not is_valid:
            return error_msg

        project_root = Path.cwd().resolve()
        is_allowed, error_msg = _check_forbidden_dirs(path, project_root)
        if not is_allowed:
            return error_msg

        if _looks_like_diff_snippet(old_string) or _looks_like_diff_snippet(new_string):
            return (
                "Error: file_edit received diff-style content. "
                "Provide the exact code to replace without '+'/'-' prefixes or use file_write."
            )

        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, encoding='utf-8') as f:
            old_content = f.read()

        # Multi-Layer Edit Strategy
        def remove_line_numbers(s):
            lines = s.split('\n')
            cleaned_lines = []
            for line in lines:
                match = re.match(r'^\s*\d+\s*\|\s?(.*)', line)
                if match:
                    cleaned_lines.append(match.group(1))
                else:
                    cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)

        def unescape_string(s):
            s = s.replace('\\n', '\n')
            s = s.replace('\\t', '\t')
            s = s.replace('\\r', '\r')
            return s

        def normalize_whitespace(s):
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in s.split('\n')]
            return '\n'.join(lines)

        def find_with_flexible_indent(content, target):
            target_lines = target.strip().split('\n')
            content_lines = content.split('\n')

            if len(target_lines) == 0:
                return None

            target_stripped = [line.strip() for line in target_lines]

            for i in range(len(content_lines) - len(target_lines) + 1):
                content_section = content_lines[i:i + len(target_lines)]
                content_stripped = [line.strip() for line in content_section]

                if content_stripped == target_stripped:
                    return '\n'.join(content_section)

            return None

        variations = [
            ("exact", old_string),
            ("line_numbers_removed", remove_line_numbers(old_string)),
            ("unescaped", unescape_string(old_string)),
            ("both_fixes", unescape_string(remove_line_numbers(old_string))),
            ("whitespace_normalized", normalize_whitespace(old_string)),
            ("all_normalized", normalize_whitespace(unescape_string(remove_line_numbers(old_string)))),
        ]

        matched_variation = None
        for variation_name, cleaned_old in variations:
            if cleaned_old in old_content:
                matched_variation = (variation_name, cleaned_old)
                break

        if not matched_variation and old_string not in old_content:
            for base_name, base_string in variations:
                flexible_match = find_with_flexible_indent(old_content, base_string)
                if flexible_match:
                    matched_variation = (f"indent_flexible_{base_name}", flexible_match)
                    break

        def _record_and_write(new_content, fix_msg=""):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            from sepilot.agent.enhanced_state import FileAction
            from sepilot.agent.tool_tracker import record_file_change_if_enabled

            record_file_change_if_enabled(
                file_path=str(file_path),
                action=FileAction.MODIFY,
                old_content=old_content,
                new_content=new_content,
                tool_used="file_edit"
            )

            result = f"Successfully edited {file_path}: replaced 1 occurrence{fix_msg}"

            # Python 파일 구문 검증
            syntax_warning = _check_python_syntax(path, new_content)
            if syntax_warning:
                result += f"\n\n{syntax_warning}"

            return result

        if old_string not in old_content and matched_variation:
            variation_name, cleaned_old = matched_variation
            new_content = old_content.replace(cleaned_old, new_string)

            fix_msg = {
                "line_numbers_removed": " (removed line number prefix)",
                "unescaped": " (converted escape sequences)",
                "both_fixes": " (removed line numbers + converted escape sequences)"
            }.get(variation_name, " (applied auto-fix)")

            return _record_and_write(new_content, fix_msg)

        elif old_string not in old_content:
            # Try fuzzy matching
            lines = old_content.split('\n')
            old_lines = old_string.split('\n')

            best_match_lines = None
            best_similarity = 0.0
            best_line_num = 0

            for i in range(len(lines) - len(old_lines) + 1):
                section = '\n'.join(lines[i:i+len(old_lines)])
                similarity = difflib.SequenceMatcher(None, old_string, section).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_lines = section
                    best_line_num = i + 1

            if best_similarity > 0.9 and best_match_lines:
                new_content = old_content.replace(best_match_lines, new_string)
                return _record_and_write(new_content, f" (fuzzy match {best_similarity*100:.1f}% at line {best_line_num})")

            # Provide hint
            snippet_preview = old_string[:100].replace('\n', '\\n').replace('\t', '\\t')
            similar_hint = ""
            if best_similarity > 0.6 and best_match_lines:
                preview_lines = best_match_lines.split('\n')[:5]
                preview = '\n'.join(f"  {line}" for line in preview_lines)
                if len(best_match_lines.split('\n')) > 5:
                    preview += "\n  ..."

                similar_hint = (
                    f"\n\n⚠️  Most similar section found at line {best_line_num} ({best_similarity*100:.1f}% match):\n"
                    f"{preview}\n\n"
                    f"💡 Suggestions:\n"
                    f"  1. Use file_read first to see the exact content\n"
                    f"  2. Copy the exact text from file_read output (including whitespace)\n"
                    f"  3. If the file has many similar sections, include more surrounding context in old_string\n"
                    f"  4. For large changes, consider using file_write to replace the entire file"
                )

            return f"Error: String not found in file.\nSearching for: {snippet_preview}{similar_hint}"

        # Check for duplicates
        count = old_content.count(old_string)
        if count > 1:
            # Claude Code style: replace_all option
            if replace_all:
                new_content = old_content.replace(old_string, new_string)
                return _record_and_write(new_content, f" (replaced all {count} occurrences)")

            lines = old_content.split('\n')
            occurrences = []
            search_preview = old_string[:50].replace('\n', '\\n')

            for i, line in enumerate(lines, 1):
                if old_string in line or (len(old_string) > 50 and old_string[:50] in line):
                    occurrences.append(f"  Line {i}: {line.strip()[:80]}")
                    if len(occurrences) >= 5:
                        break

            locations = "\n".join(occurrences) if occurrences else "  (occurrences span multiple lines)"

            return (
                f"Error: String appears {count} times in file.\n"
                f"Searching for: {search_preview}\n\n"
                f"Found at:\n{locations}\n\n"
                f"Tip: Use replace_all=True to replace all occurrences, "
                f"or include more context in old_string to make it unique."
            )

        # Single match - replace
        new_content = old_content.replace(old_string, new_string)
        return _record_and_write(new_content)

    except Exception as e:
        return f"Error editing file: {str(e)}"


# Import and create patch tool
def _get_apply_patch_tool():
    """Lazy load the apply_patch tool.

    Returns the patch tool or None if the module is not available.
    Logs warnings/errors for debugging purposes.
    """
    try:
        from sepilot.tools.file_tools.patch_tool import create_patch_tool
        return create_patch_tool()
    except ImportError as e:
        import logging
        logging.getLogger(__name__).debug(f"apply_patch tool not available (ImportError): {e}")
        return None
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load apply_patch tool: {e}")
        return None


apply_patch = _get_apply_patch_tool()


__all__ = [
    'file_read',
    'file_write',
    'file_edit',
    'apply_patch',
    '_normalize_relative_path',
    '_preferred_required_file_path',
]
