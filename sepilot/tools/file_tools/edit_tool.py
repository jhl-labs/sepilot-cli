"""File editing tool"""

from pathlib import Path

from sepilot.tools.base_tool import BaseTool


class FileEditTool(BaseTool):
    """Tool for editing specific parts of files"""

    name = "file_edit"
    description = "Edit a file by replacing specific text"
    parameters = {
        "file_path": "Path to the file to edit (required)",
        "old_text": "Text to search for and replace (required)",
        "new_text": "Text to replace with (required)",
        "encoding": "File encoding (default: utf-8)"
    }

    def execute(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
        encoding: str = "utf-8"
    ) -> str:
        """Edit a file by replacing text"""
        self.validate_params(file_path=file_path, old_text=old_text, new_text=new_text)

        try:
            # Security: block symbolic links BEFORE resolve (TOCTOU prevention)
            raw_path = Path(file_path)
            if raw_path.is_symlink():
                return f"Error: Cannot edit symbolic links: {file_path}"

            path = raw_path.resolve()

            # Security: prevent path traversal
            project_root = Path.cwd().resolve()
            try:
                path.relative_to(project_root)
            except ValueError:
                return f"Error: Path escapes project directory: {file_path}"

            # Security: block system-critical paths
            forbidden = ["/etc", "/usr", "/bin", "/sbin", "/var", "/boot", "/proc", "/sys"]
            if any(str(path).startswith(p) for p in forbidden):
                return f"Error: Cannot edit files in system directories: {file_path}"

            if not path.exists():
                return f"Error: File not found: {file_path}"

            if not path.is_file():
                return f"Error: Not a file: {file_path}"

            # Read current content
            with open(path, encoding=encoding) as f:
                content = f.read()

            # Check if old_text exists
            if old_text not in content:
                return f"Error: Text not found in file: '{old_text[:50]}...'"

            # Count occurrences
            count = content.count(old_text)

            # Replace the text
            new_content = content.replace(old_text, new_text)

            # Write back
            with open(path, 'w', encoding=encoding) as f:
                f.write(new_content)

            return f"Edited file: {file_path}\n  Replaced {count} occurrence(s)"

        except Exception as e:
            return f"Error editing file: {str(e)}"
