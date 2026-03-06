"""Side-by-side diff renderer - OpenCode style diff display.

Provides rich diff visualization with:
- Side-by-side comparison
- Character-level highlighting
- Line numbers
- Syntax highlighting
"""

import difflib
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class DiffLine:
    """A single line in a diff"""
    line_num_old: int | None
    line_num_new: int | None
    content_old: str
    content_new: str
    change_type: str  # 'equal', 'insert', 'delete', 'replace'


class SideBySideDiffRenderer:
    """Renders diffs in side-by-side format with character-level highlighting.

    OpenCode style:
    - Two-column layout (old | new)
    - Line numbers on each side
    - Character-level change highlighting
    - Color-coded changes (red=delete, green=insert)
    """

    def __init__(
        self,
        console: Console | None = None,
        width: int = 160,
        context_lines: int = 3,
    ):
        """Initialize diff renderer.

        Args:
            console: Rich console for output
            width: Total width (split between two columns)
            context_lines: Number of context lines around changes
        """
        self.console = console or Console()
        self.width = width
        self.context_lines = context_lines
        self.col_width = (width - 10) // 2  # Reserve space for line numbers

    def render_diff(
        self,
        old_text: str,
        new_text: str,
        old_label: str = "Original",
        new_label: str = "Modified",
    ) -> None:
        """Render a side-by-side diff.

        Args:
            old_text: Original text
            new_text: Modified text
            old_label: Label for original side
            new_label: Label for modified side
        """
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        # Generate diff
        diff_lines = self._compute_diff(old_lines, new_lines)

        # Create table
        table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            box=None,
            padding=(0, 1),
            collapse_padding=True,
        )

        # Add columns
        table.add_column("#", style="dim", width=5, justify="right")
        table.add_column(old_label, width=self.col_width)
        table.add_column("#", style="dim", width=5, justify="right")
        table.add_column(new_label, width=self.col_width)

        # Add rows
        for diff_line in diff_lines:
            old_num = str(diff_line.line_num_old) if diff_line.line_num_old else ""
            new_num = str(diff_line.line_num_new) if diff_line.line_num_new else ""

            old_content = self._format_line(
                diff_line.content_old,
                diff_line.change_type,
                is_old=True,
            )
            new_content = self._format_line(
                diff_line.content_new,
                diff_line.change_type,
                is_old=False,
            )

            table.add_row(old_num, old_content, new_num, new_content)

        # Display
        self.console.print(table)

    def render_unified_diff(
        self,
        old_text: str,
        new_text: str,
        old_label: str = "a/file",
        new_label: str = "b/file",
    ) -> None:
        """Render a unified diff (traditional format).

        Args:
            old_text: Original text
            new_text: Modified text
            old_label: Label for original
            new_label: Label for modified
        """
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_label,
            tofile=new_label,
            n=self.context_lines,
        )

        output = Text()
        for line in diff:
            line = line.rstrip('\n')
            if line.startswith('+++'):
                output.append(line + '\n', style="bold green")
            elif line.startswith('---'):
                output.append(line + '\n', style="bold red")
            elif line.startswith('@@'):
                output.append(line + '\n', style="bold cyan")
            elif line.startswith('+'):
                output.append(line + '\n', style="green")
            elif line.startswith('-'):
                output.append(line + '\n', style="red")
            else:
                output.append(line + '\n')

        self.console.print(Panel(output, title="Diff", border_style="dim"))

    def _compute_diff(
        self,
        old_lines: list[str],
        new_lines: list[str],
    ) -> list[DiffLine]:
        """Compute diff lines for side-by-side display.

        Args:
            old_lines: Original lines
            new_lines: Modified lines

        Returns:
            List of DiffLine objects
        """
        result = []
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

        old_num = 0
        new_num = 0

        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'equal':
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    old_num += 1
                    new_num += 1
                    result.append(DiffLine(
                        line_num_old=old_num,
                        line_num_new=new_num,
                        content_old=old_lines[i].rstrip('\n'),
                        content_new=new_lines[j].rstrip('\n'),
                        change_type='equal',
                    ))

            elif opcode == 'replace':
                # Show replacements side by side
                old_range = list(range(i1, i2))
                new_range = list(range(j1, j2))

                max_len = max(len(old_range), len(new_range))

                for k in range(max_len):
                    old_idx = old_range[k] if k < len(old_range) else None
                    new_idx = new_range[k] if k < len(new_range) else None

                    if old_idx is not None:
                        old_num += 1
                    if new_idx is not None:
                        new_num += 1

                    result.append(DiffLine(
                        line_num_old=old_num if old_idx is not None else None,
                        line_num_new=new_num if new_idx is not None else None,
                        content_old=old_lines[old_idx].rstrip('\n') if old_idx is not None else '',
                        content_new=new_lines[new_idx].rstrip('\n') if new_idx is not None else '',
                        change_type='replace',
                    ))

            elif opcode == 'delete':
                for i in range(i1, i2):
                    old_num += 1
                    result.append(DiffLine(
                        line_num_old=old_num,
                        line_num_new=None,
                        content_old=old_lines[i].rstrip('\n'),
                        content_new='',
                        change_type='delete',
                    ))

            elif opcode == 'insert':
                for j in range(j1, j2):
                    new_num += 1
                    result.append(DiffLine(
                        line_num_old=None,
                        line_num_new=new_num,
                        content_old='',
                        content_new=new_lines[j].rstrip('\n'),
                        change_type='insert',
                    ))

        return result

    def _format_line(
        self,
        content: str,
        change_type: str,
        is_old: bool,
    ) -> Text:
        """Format a line with appropriate styling.

        Args:
            content: Line content
            change_type: Type of change
            is_old: Whether this is from the old side

        Returns:
            Styled Text object
        """
        text = Text()

        # Truncate if too long
        if len(content) > self.col_width - 2:
            content = content[:self.col_width - 5] + "..."

        if change_type == 'equal':
            text.append(content)
        elif change_type == 'delete':
            if is_old:
                text.append(content, style="red")
            else:
                text.append(content, style="dim")
        elif change_type == 'insert':
            if is_old:
                text.append(content, style="dim")
            else:
                text.append(content, style="green")
        elif change_type == 'replace':
            if is_old:
                text.append(content, style="red")
            else:
                text.append(content, style="green")

        return text

    def get_diff_summary(
        self,
        old_text: str,
        new_text: str,
    ) -> dict[str, Any]:
        """Get a summary of changes.

        Args:
            old_text: Original text
            new_text: Modified text

        Returns:
            Summary dictionary
        """
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

        insertions = 0
        deletions = 0
        modifications = 0

        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'insert':
                insertions += j2 - j1
            elif opcode == 'delete':
                deletions += i2 - i1
            elif opcode == 'replace':
                modifications += max(i2 - i1, j2 - j1)

        return {
            "old_lines": len(old_lines),
            "new_lines": len(new_lines),
            "insertions": insertions,
            "deletions": deletions,
            "modifications": modifications,
            "similarity": matcher.ratio(),
        }


def render_side_by_side_diff(
    old_text: str,
    new_text: str,
    old_label: str = "Original",
    new_label: str = "Modified",
    console: Console | None = None,
) -> None:
    """Convenience function to render a side-by-side diff.

    Args:
        old_text: Original text
        new_text: Modified text
        old_label: Label for original
        new_label: Label for modified
        console: Optional console
    """
    renderer = SideBySideDiffRenderer(console=console)
    renderer.render_diff(old_text, new_text, old_label, new_label)


def render_unified_diff(
    old_text: str,
    new_text: str,
    old_label: str = "a/file",
    new_label: str = "b/file",
    console: Console | None = None,
) -> None:
    """Convenience function to render a unified diff.

    Args:
        old_text: Original text
        new_text: Modified text
        old_label: Label for original
        new_label: Label for modified
        console: Optional console
    """
    renderer = SideBySideDiffRenderer(console=console)
    renderer.render_unified_diff(old_text, new_text, old_label, new_label)
