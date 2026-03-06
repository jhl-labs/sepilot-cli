"""File reading tool"""

from pathlib import Path

from sepilot.tools.base_tool import BaseTool


class FileReadTool(BaseTool):
    """Tool for reading file contents"""

    name = "file_read"
    description = "Read the contents of a file. Use this when you need to read any file like README.md, config files, source code, etc. (파일 내용 읽기)"
    parameters = {
        "file_path": "Path to the file to read (required) - e.g., README.md, test.py, config.json",
        "encoding": "File encoding (default: utf-8)"
    }

    def execute(self, file_path: str, encoding: str = "utf-8") -> str:
        """Read a file and return its contents"""
        self.validate_params(file_path=file_path)

        try:
            # Security: validate path before reading
            from sepilot.utils.security import SecurityValidator
            is_safe, error_msg, _ = SecurityValidator.validate_file_path(file_path, "read")
            if not is_safe:
                return f"Error: {error_msg}"

            path = Path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"

            if not path.is_file():
                return f"Error: Not a file: {file_path}"

            # Check file size (limit to 1MB)
            if path.stat().st_size > 1_000_000:
                return f"Error: File too large (>1MB): {file_path}"

            with open(path, encoding=encoding) as f:
                content = f.read()

            # Add line numbers for better readability
            lines = content.split('\n')
            numbered_lines = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
            result = '\n'.join(numbered_lines)

            return f"Contents of {file_path}:\n{result}"

        except UnicodeDecodeError:
            return f"Error: Unable to decode file with {encoding} encoding: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
