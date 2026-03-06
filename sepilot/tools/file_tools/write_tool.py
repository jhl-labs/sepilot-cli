"""File writing tool"""

from pathlib import Path

from sepilot.tools.base_tool import BaseTool


class FileWriteTool(BaseTool):
    """Tool for writing content to files"""

    name = "file_write"
    description = "Write content to a file (creates or overwrites)"
    parameters = {
        "file_path": "Path to the file to write (required)",
        "content": "Content to write to the file (required)",
        "encoding": "File encoding (default: utf-8)",
        "create_dirs": "Create parent directories if they don't exist (default: True)"
    }

    def execute(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> str:
        """Write content to a file"""
        self.validate_params(file_path=file_path, content=content)

        try:
            # Security: block symbolic links BEFORE resolve (TOCTOU prevention)
            raw_path = Path(file_path)
            if raw_path.is_symlink():
                return f"Error: Cannot write to symbolic links: {file_path}"

            # Security: Resolve path and check for path traversal
            path = raw_path.resolve()
            project_root = Path.cwd().resolve()

            # Check if path is within project root
            try:
                path.relative_to(project_root)
            except ValueError:
                return f"Error: Path escape attempt detected. File path must be within project directory: {file_path}"

            # Blacklist system paths
            forbidden_paths = ['/etc', '/usr', '/bin', '/sbin', '/sys', '/proc', '/dev', '/boot', '/root']
            path_str = str(path)
            for forbidden in forbidden_paths:
                if path_str.startswith(forbidden):
                    return f"Error: Access to system path denied: {file_path}"

            # Create parent directories if needed
            if create_dirs and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists (for logging)
            exists = path.exists()

            # Write the content
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)

            # Count lines and size for feedback
            lines = content.count('\n') + 1
            size = len(content.encode(encoding))

            action = "Updated" if exists else "Created"
            return f"{action} file: {file_path}\n  Lines: {lines}\n  Size: {size} bytes"

        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
