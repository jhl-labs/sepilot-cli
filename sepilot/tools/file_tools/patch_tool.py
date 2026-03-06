"""Patch tool for applying unified diffs.

Supports applying unified diff patches to files, similar to `patch -p1`.
Useful for complex multi-file changes.
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PatchHunk:
    """A single hunk from a unified diff"""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class FilePatch:
    """Patch for a single file"""
    old_file: str
    new_file: str
    hunks: list[PatchHunk]


class PatchError(Exception):
    """Patch application error"""
    pass


class UnifiedDiffParser:
    """Parser for unified diff format"""

    # Regex patterns
    DIFF_HEADER = re.compile(r'^diff --git a/(.*) b/(.*)$')
    OLD_FILE = re.compile(r'^--- (?:a/)?(.*)$')
    NEW_FILE = re.compile(r'^\+\+\+ (?:b/)?(.*)$')
    HUNK_HEADER = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

    def parse(self, diff_text: str) -> list[FilePatch]:
        """Parse a unified diff into FilePatch objects

        Args:
            diff_text: Unified diff text

        Returns:
            List of FilePatch objects
        """
        patches = []
        lines = diff_text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for diff header or --- line
            if self.DIFF_HEADER.match(line):
                i += 1
                continue

            old_match = self.OLD_FILE.match(line)
            if old_match:
                old_file = old_match.group(1)
                i += 1

                if i >= len(lines):
                    break

                new_match = self.NEW_FILE.match(lines[i])
                if new_match:
                    new_file = new_match.group(1)
                    i += 1

                    # Parse hunks
                    hunks = []
                    while i < len(lines):
                        hunk_match = self.HUNK_HEADER.match(lines[i])
                        if hunk_match:
                            hunk, i = self._parse_hunk(lines, i, hunk_match)
                            hunks.append(hunk)
                        elif lines[i].startswith('diff ') or self.OLD_FILE.match(lines[i]):
                            break
                        else:
                            i += 1

                    if hunks:
                        patches.append(FilePatch(
                            old_file=old_file,
                            new_file=new_file,
                            hunks=hunks,
                        ))
                continue

            i += 1

        return patches

    def _parse_hunk(
        self, lines: list[str], start_idx: int, header_match: re.Match
    ) -> tuple[PatchHunk, int]:
        """Parse a single hunk

        Args:
            lines: All diff lines
            start_idx: Index of hunk header
            header_match: Regex match for hunk header

        Returns:
            Tuple of (PatchHunk, next_index)
        """
        old_start = int(header_match.group(1))
        old_count = int(header_match.group(2) or 1)
        new_start = int(header_match.group(3))
        new_count = int(header_match.group(4) or 1)

        hunk_lines = []
        i = start_idx + 1

        while i < len(lines):
            line = lines[i]

            # End of hunk
            if (line.startswith('diff ') or
                line.startswith('--- ') or
                self.HUNK_HEADER.match(line)):
                break

            # Hunk content lines
            if line.startswith(' ') or line.startswith('+') or line.startswith('-'):
                hunk_lines.append(line)
            elif line == '':
                # Empty line might be part of context
                hunk_lines.append(' ')
            elif line.startswith('\\'):
                # "\ No newline at end of file"
                pass

            i += 1

        return PatchHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            lines=hunk_lines,
        ), i


class PatchApplier:
    """Applies patches to files"""

    def __init__(self, working_dir: str | Path | None = None):
        """Initialize patch applier

        Args:
            working_dir: Working directory for file paths
        """
        self.working_dir = Path(working_dir or os.getcwd())

    def apply_patch(
        self,
        patch: FilePatch,
        dry_run: bool = False,
        reverse: bool = False,
        strip: int = 0,
    ) -> dict[str, Any]:
        """Apply a patch to a file

        Args:
            patch: FilePatch to apply
            dry_run: If True, don't actually modify files
            reverse: If True, reverse the patch (unapply)
            strip: Number of leading path components to strip

        Returns:
            Result dictionary with status and details
        """
        # Determine target file
        file_path = patch.new_file if not reverse else patch.old_file

        # Strip leading path components
        if strip > 0:
            parts = file_path.split('/')
            file_path = '/'.join(parts[strip:])

        target_path = self.working_dir / file_path

        # Security: prevent path traversal outside working directory
        try:
            target_path.resolve().relative_to(self.working_dir.resolve())
        except ValueError:
            return {
                "file": str(file_path),
                "target": str(target_path),
                "hunks_total": len(patch.hunks),
                "hunks_applied": 0,
                "hunks_failed": len(patch.hunks),
                "dry_run": dry_run,
                "errors": [f"Path traversal blocked: {file_path} escapes working directory"],
            }

        result = {
            "file": str(file_path),
            "target": str(target_path),
            "hunks_total": len(patch.hunks),
            "hunks_applied": 0,
            "hunks_failed": 0,
            "dry_run": dry_run,
            "errors": [],
        }

        # Read original content
        if target_path.exists():
            try:
                with open(target_path, encoding="utf-8") as f:
                    original_lines = f.read().split('\n')
            except Exception as e:
                result["errors"].append(f"Failed to read file: {e}")
                return result
        else:
            # New file
            if reverse:
                result["errors"].append(f"File not found for reverse patch: {file_path}")
                return result
            original_lines = []

        # Apply hunks
        patched_lines = original_lines.copy()
        offset = 0

        for hunk in patch.hunks:
            try:
                if reverse:
                    patched_lines, hunk_offset = self._apply_hunk_reverse(
                        patched_lines, hunk, offset
                    )
                else:
                    patched_lines, hunk_offset = self._apply_hunk(
                        patched_lines, hunk, offset
                    )
                offset += hunk_offset
                result["hunks_applied"] += 1

            except PatchError as e:
                result["hunks_failed"] += 1
                result["errors"].append(str(e))

        # Write result
        if not dry_run and result["hunks_applied"] > 0:
            try:
                # Create backup
                if target_path.exists():
                    backup_path = target_path.with_suffix(target_path.suffix + '.orig')
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(original_lines))

                # Write patched file
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(patched_lines))

                result["success"] = True

            except Exception as e:
                result["errors"].append(f"Failed to write file: {e}")
                result["success"] = False
        else:
            result["success"] = result["hunks_failed"] == 0

        return result

    def _apply_hunk(
        self, lines: list[str], hunk: PatchHunk, offset: int
    ) -> tuple[list[str], int]:
        """Apply a single hunk

        Args:
            lines: Current file lines
            hunk: Hunk to apply
            offset: Line offset from previous hunks

        Returns:
            Tuple of (modified_lines, additional_offset)
        """
        # Calculate actual start position
        start = hunk.old_start - 1 + offset

        # Extract context and changes
        context_lines = []
        add_lines = []
        remove_lines = []

        for line in hunk.lines:
            if line.startswith(' '):
                context_lines.append(line[1:])
            elif line.startswith('+'):
                add_lines.append(line[1:])
            elif line.startswith('-'):
                remove_lines.append(line[1:])

        # Try to find exact match position
        best_pos = self._find_context_match(
            lines, start, context_lines, remove_lines
        )

        if best_pos is None:
            raise PatchError(
                f"Hunk @@ -{hunk.old_start},{hunk.old_count} could not be applied"
            )

        # Apply the hunk
        result = lines[:best_pos]

        # Process hunk lines in order
        old_idx = best_pos
        for line in hunk.lines:
            if line.startswith(' '):
                # Context line - keep it
                if old_idx < len(lines):
                    result.append(lines[old_idx])
                    old_idx += 1
            elif line.startswith('-'):
                # Remove line - skip it
                old_idx += 1
            elif line.startswith('+'):
                # Add line
                result.append(line[1:])

        # Add remaining lines
        result.extend(lines[old_idx:])

        # Calculate offset change
        offset_change = len(add_lines) - len(remove_lines)

        return result, offset_change

    def _apply_hunk_reverse(
        self, lines: list[str], hunk: PatchHunk, offset: int
    ) -> tuple[list[str], int]:
        """Apply a hunk in reverse (unapply)

        Args:
            lines: Current file lines
            hunk: Hunk to reverse
            offset: Line offset from previous hunks

        Returns:
            Tuple of (modified_lines, additional_offset)
        """
        # Swap + and - for reverse
        reversed_lines = []
        for line in hunk.lines:
            if line.startswith('+'):
                reversed_lines.append('-' + line[1:])
            elif line.startswith('-'):
                reversed_lines.append('+' + line[1:])
            else:
                reversed_lines.append(line)

        reversed_hunk = PatchHunk(
            old_start=hunk.new_start,
            old_count=hunk.new_count,
            new_start=hunk.old_start,
            new_count=hunk.old_count,
            lines=reversed_lines,
        )

        return self._apply_hunk(lines, reversed_hunk, offset)

    def _find_context_match(
        self,
        lines: list[str],
        expected_pos: int,
        context: list[str],
        removals: list[str],
    ) -> int | None:
        """Find the best position to apply a hunk

        Args:
            lines: File lines
            expected_pos: Expected position
            context: Context lines
            removals: Lines to remove

        Returns:
            Best position or None
        """
        # First try exact position
        if self._context_matches(lines, expected_pos, context, removals):
            return expected_pos

        # Search nearby (fuzz factor)
        for fuzz in range(1, 50):
            # Try before
            if expected_pos - fuzz >= 0:
                if self._context_matches(lines, expected_pos - fuzz, context, removals):
                    return expected_pos - fuzz

            # Try after
            if expected_pos + fuzz < len(lines):
                if self._context_matches(lines, expected_pos + fuzz, context, removals):
                    return expected_pos + fuzz

        return None

    def _context_matches(
        self,
        lines: list[str],
        pos: int,
        context: list[str],
        removals: list[str],
    ) -> bool:
        """Check if context matches at position

        Args:
            lines: File lines
            pos: Position to check
            context: Context lines
            removals: Lines to remove

        Returns:
            True if context matches
        """
        # Combine context and removals for matching
        expected = []
        for line in context + removals:
            expected.append(line)

        if not expected:
            return True

        if pos + len(expected) > len(lines):
            return False

        for i, exp_line in enumerate(expected):
            if lines[pos + i].rstrip() != exp_line.rstrip():
                return False

        return True


def apply_unified_diff(
    diff_text: str,
    working_dir: str | Path | None = None,
    dry_run: bool = False,
    strip: int = 1,
) -> list[dict[str, Any]]:
    """Apply a unified diff to files

    Args:
        diff_text: Unified diff text
        working_dir: Working directory for file paths
        dry_run: If True, don't actually modify files
        strip: Number of leading path components to strip (default: 1)

    Returns:
        List of result dictionaries for each file
    """
    parser = UnifiedDiffParser()
    applier = PatchApplier(working_dir)

    patches = parser.parse(diff_text)
    results = []

    for patch in patches:
        result = applier.apply_patch(patch, dry_run=dry_run, strip=strip)
        results.append(result)

    return results


def create_patch_tool():
    """Create a LangChain tool for applying patches"""
    from langchain_core.tools import tool

    @tool
    def apply_patch(
        diff: str,
        dry_run: bool = False,
        strip: int = 1,
    ) -> str:
        """Apply a unified diff patch to files.

        Applies changes specified in unified diff format to the target files.
        Similar to running `patch -p1` on the command line.

        Args:
            diff: Unified diff text (output of `git diff` or similar)
            dry_run: If True, show what would happen without making changes
            strip: Number of leading path components to strip (default: 1)

        Returns:
            Summary of applied changes

        Example diff format:
            --- a/src/file.py
            +++ b/src/file.py
            @@ -1,3 +1,4 @@
             existing line
            +new line
             another line
        """
        try:
            results = apply_unified_diff(diff, dry_run=dry_run, strip=strip)

            summary_lines = []
            total_applied = 0
            total_failed = 0

            for result in results:
                status = "OK" if result.get("success") else "FAILED"
                summary_lines.append(
                    f"{result['file']}: {status} "
                    f"({result['hunks_applied']}/{result['hunks_total']} hunks)"
                )
                total_applied += result["hunks_applied"]
                total_failed += result["hunks_failed"]

                for error in result.get("errors", []):
                    summary_lines.append(f"  Error: {error}")

            if dry_run:
                summary_lines.insert(0, "[DRY RUN - no changes made]")

            summary_lines.append(
                f"\nTotal: {total_applied} hunks applied, {total_failed} failed"
            )

            return "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"Patch failed: {e}")
            return f"Patch failed: {e}"

    return apply_patch
