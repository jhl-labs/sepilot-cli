"""File pattern matching tool using glob"""

from pathlib import Path

from sepilot.tools.base_tool import BaseTool


class GlobTool(BaseTool):
    """Tool for finding files using glob patterns"""

    name = "glob"
    description = "Find files matching a pattern"
    parameters = {
        "pattern": "Glob pattern to match files (e.g., '*.py', '**/*.js') (required)",
        "root_dir": "Root directory to search from (default: current directory)",
        "recursive": "Search recursively in subdirectories (default: True)",
        "include_hidden": "Include hidden files (starting with .) (default: False)",
        "max_results": "Maximum number of results to return (default: 100)"
    }

    def execute(
        self,
        pattern: str,
        root_dir: str = ".",
        recursive: bool = True,
        include_hidden: bool = False,
        max_results: int = 100
    ) -> str:
        """Find files matching a glob pattern"""
        self.validate_params(pattern=pattern)

        try:
            # Convert to Path object
            root = Path(root_dir)
            if not root.exists():
                return f"Error: Directory not found: {root_dir}"

            # Use glob or rglob based on recursive flag
            if recursive and "**" not in pattern:
                # Add ** if recursive but not in pattern
                if "/" in pattern:
                    parts = pattern.split("/", 1)
                    pattern = f"**/{parts[-1]}"
                else:
                    pattern = f"**/{pattern}"

            # Get matches
            if recursive or "**" in pattern:
                matches = list(root.rglob(pattern))
            else:
                matches = list(root.glob(pattern))

            # Filter hidden files if needed
            if not include_hidden:
                matches = [
                    m for m in matches
                    if not any(part.startswith('.') for part in m.parts)
                ]

            # Sort by modification time (newest first)
            matches.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)

            # Limit results
            matches = matches[:max_results]

            if not matches:
                return f"No files found matching pattern: '{pattern}'"

            # Format results
            result_lines = [f"Found {len(matches)} files matching '{pattern}':\n"]

            for match in matches:
                # Get relative path for cleaner output
                try:
                    rel_path = match.relative_to(Path.cwd())
                except ValueError:
                    rel_path = match

                # Get file info
                if match.is_file():
                    size = match.stat().st_size
                    size_str = self._format_size(size)
                    result_lines.append(f"  📄 {rel_path} ({size_str})")
                elif match.is_dir():
                    result_lines.append(f"  📁 {rel_path}/")
                else:
                    result_lines.append(f"  ❓ {rel_path}")

            # Add summary
            total_size = sum(
                m.stat().st_size for m in matches
                if m.is_file() and m.exists()
            )
            result_lines.append(f"\nTotal size: {self._format_size(total_size)}")

            return "\n".join(result_lines)

        except Exception as e:
            return f"Error finding files: {str(e)}"

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
