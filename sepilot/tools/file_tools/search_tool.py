"""Code search tool using ripgrep"""

import json
import subprocess

from sepilot.tools.base_tool import BaseTool


class SearchTool(BaseTool):
    """Tool for searching code using ripgrep"""

    name = "search"
    description = "Search for patterns in files using ripgrep"
    parameters = {
        "pattern": "Search pattern or regex (required)",
        "path": "Directory or file to search in (default: current directory)",
        "file_type": "File type to search (e.g., 'py', 'js', 'md')",
        "case_sensitive": "Case sensitive search (default: False)",
        "max_results": "Maximum number of results to return (default: 50)"
    }

    def execute(
        self,
        pattern: str,
        path: str = ".",
        file_type: str | None = None,
        case_sensitive: bool = False,
        max_results: int = 50
    ) -> str:
        """Search for patterns using ripgrep"""
        self.validate_params(pattern=pattern)

        # Check if ripgrep is installed
        try:
            subprocess.run(["rg", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to grep if ripgrep not available
            return self._fallback_grep(pattern, path, file_type, case_sensitive, max_results)

        # Build ripgrep command
        cmd = ["rg", "--json", "--max-count", str(max_results)]

        if not case_sensitive:
            cmd.append("-i")

        if file_type:
            cmd.extend(["-t", file_type])

        cmd.extend([pattern, path])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0 and result.returncode != 1:  # 1 means no matches
                return f"Search failed: {result.stderr}"

            # Parse JSON output
            matches = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        match_data = data.get("data", {})
                        matches.append({
                            "file": match_data.get("path", {}).get("text", ""),
                            "line_number": match_data.get("line_number"),
                            "line": match_data.get("lines", {}).get("text", "").strip(),
                            "match": match_data.get("submatches", [{}])[0].get("match", {}).get("text", "")
                        })
                except json.JSONDecodeError:
                    continue

            if not matches:
                return f"No matches found for pattern: '{pattern}'"

            # Format results
            result_lines = [f"Found {len(matches)} matches for '{pattern}':\n"]
            for match in matches[:max_results]:
                result_lines.append(
                    f"{match['file']}:{match['line_number']}: {match['line']}"
                )

            return "\n".join(result_lines)

        except subprocess.TimeoutExpired:
            return "Search timed out after 30 seconds"
        except Exception as e:
            return f"Search error: {str(e)}"

    def _fallback_grep(
        self,
        pattern: str,
        path: str,
        file_type: str | None,
        case_sensitive: bool,
        max_results: int
    ) -> str:
        """Fallback to grep if ripgrep not available"""
        cmd = ["grep", "-r", "-n"]

        if not case_sensitive:
            cmd.append("-i")

        if file_type:
            cmd.extend(["--include", f"*.{file_type}"])

        cmd.extend([pattern, path])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[:max_results]
                return "Found matches (using grep):\n" + "\n".join(lines)
            else:
                return f"No matches found for pattern: '{pattern}'"

        except Exception as e:
            return f"Search error: {str(e)}"
