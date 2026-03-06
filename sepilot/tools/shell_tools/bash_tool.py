"""Bash command execution tool"""

import os
import re
import subprocess
from pathlib import Path

from sepilot.tools.base_tool import BaseTool


def _build_shell_command(command: str) -> list[str]:
    """Build a platform-safe shell invocation command."""
    if os.name == "nt":
        return ["cmd", "/c", command]

    shell_path = os.environ.get("SHELL")
    if shell_path and Path(shell_path).exists():
        return [shell_path, "-lc", command]
    if Path("/bin/bash").exists():
        return ["/bin/bash", "-lc", command]
    return ["/bin/sh", "-c", command]


class BashTool(BaseTool):
    """Tool for executing bash commands"""

    name = "bash"
    description = "Execute bash commands and return output"
    parameters = {
        "command": "The bash command to execute (required)",
        "cwd": "Working directory for command execution (default: current)",
        "timeout": "Command timeout in seconds (default: 30)",
        "max_output_lines": "Maximum output lines to return (default: 1000)"
    }

    # Commands that may produce excessive output
    EXCESSIVE_OUTPUT_COMMANDS = {
        'ls -R': 'ls <specific_directory>',
        'ls -lR': 'ls -l <specific_directory>',
        'find .': 'find <specific_directory> -name <pattern>',
        'find . -name': 'find <specific_directory> -name <pattern>',
        'du -a': 'du -sh <specific_directory>',
        'du -ah': 'du -sh <specific_directory>',
        'tree': 'tree -L 2 <specific_directory>',
        'cat /var/log/*': 'tail -n 100 <specific_file>',
    }

    def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = 30,
        max_output_lines: int = 1000
    ) -> str:
        """Execute a bash command with output size limit"""
        self.validate_params(command=command)

        # Enhanced security check - block dangerous commands and patterns
        dangerous_patterns = [
            # Destructive file operations
            r'rm\s+-rf\s+/',
            r'rm\s+-fr\s+/',
            r'rm\s+--recursive\s+--force\s+/',
            # Device access
            r'dd\s+if=/dev/(zero|random|urandom)',
            r'>\s*/dev/(sd[a-z]|hd[a-z]|nvme)',
            # Fork bombs and resource exhaustion
            r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;',
            # Filesystem operations
            r'mkfs\.',
            r'fdisk',
            r'parted',
            # Privilege escalation
            r'sudo\s+',
            r'su\s+',
            # Remote code execution
            r'curl.*\|.*sh',
            r'wget.*\|.*sh',
            r'curl.*\|.*bash',
            r'wget.*\|.*bash',
            # System path access
            r'/etc/(passwd|shadow|sudoers)',
            # Dangerous chmod operations
            r'chmod\s+(777|666)',
            # Background processes with dangerous commands
            r'&\s*rm\s+-rf',
            # Command chaining with dangerous patterns
            r'&&\s*rm\s+-rf\s+/',
            r';\s*rm\s+-rf\s+/',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return "Error: Dangerous command pattern detected. Command blocked for security."

        # Check for commands that may produce excessive output
        for excessive_cmd, suggestion in self.EXCESSIVE_OUTPUT_COMMANDS.items():
            if excessive_cmd in command:
                return (
                    f"⚠️ Warning: Command '{command}' may produce excessive output.\n\n"
                    f"💡 Suggested alternative: {suggestion}\n\n"
                    f"Large outputs can cause timeouts and memory issues.\n"
                    f"If you need to proceed, use a more specific path or add filters like '| head -n 100'."
                )

        # Use centralized security validator for shell command validation
        from sepilot.utils.security import SecurityValidator
        is_safe, sec_error = SecurityValidator.validate_shell_command(command)
        if not is_safe:
            return f"Error: {sec_error}"

        try:
            # Set working directory
            work_dir = Path(cwd) if cwd else Path.cwd()
            if cwd and not work_dir.exists():
                return f"Error: Working directory not found: {cwd}"

            # Execute the command
            result = subprocess.run(
                _build_shell_command(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            )

            # Prepare output with size limit
            output_lines = []

            if result.stdout:
                stdout_lines = result.stdout.split('\n')

                # Check if output exceeds limit
                if len(stdout_lines) > max_output_lines:
                    truncated_stdout = '\n'.join(stdout_lines[:max_output_lines])
                    output_lines.append("Output (truncated):")
                    output_lines.append(truncated_stdout)
                    output_lines.append("")
                    output_lines.append(
                        f"... (output truncated: {len(stdout_lines) - max_output_lines} more lines)\n"
                        f"💡 Tip: Use more specific commands or filters to reduce output"
                    )
                else:
                    output_lines.append("Output:")
                    output_lines.append(result.stdout)

            if result.stderr:
                stderr_lines = result.stderr.split('\n')

                # Limit stderr as well
                if len(stderr_lines) > 200:  # Lower limit for errors
                    truncated_stderr = '\n'.join(stderr_lines[:200])
                    output_lines.append("Errors/Warnings (truncated):")
                    output_lines.append(truncated_stderr)
                    output_lines.append(f"... ({len(stderr_lines) - 200} more error lines)")
                else:
                    output_lines.append("Errors/Warnings:")
                    output_lines.append(result.stderr)

            if result.returncode != 0:
                output_lines.append(f"Return code: {result.returncode}")

            if not output_lines:
                output_lines.append("Command executed successfully (no output)")

            return "\n".join(output_lines)

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except FileNotFoundError:
            return f"Error: Command not found: {command.split()[0]}"
        except Exception as e:
            return f"Error executing command: {str(e)}"
