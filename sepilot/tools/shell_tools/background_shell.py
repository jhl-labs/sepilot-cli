"""Background shell management tools"""

import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from sepilot.tools.base_tool import BaseTool


def _build_shell_command(command: str) -> list[str]:
    """Build a platform-safe shell invocation command."""
    import os

    if os.name == "nt":
        return ["cmd", "/c", command]

    shell_path = os.environ.get("SHELL")
    if shell_path and Path(shell_path).exists():
        return [shell_path, "-lc", command]
    if Path("/bin/bash").exists():
        return ["/bin/bash", "-lc", command]
    return ["/bin/sh", "-c", command]


class BackgroundShellManager:
    """Singleton manager for background shells"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.shells = {}
                    cls._instance.output_buffers = {}
        return cls._instance

    def create_shell(self, command: str, cwd: str | None = None) -> str:
        """Create a new background shell"""
        import os
        shell_id = f"shell_{uuid.uuid4().hex[:8]}"

        # Create output queue
        output_queue = queue.Queue()
        self.output_buffers[shell_id] = {
            'queue': output_queue,
            'all_output': [],
            'last_read_index': 0
        }

        # Set up non-interactive environment
        env = os.environ.copy()
        env['CI'] = 'true'
        env['DEBIAN_FRONTEND'] = 'noninteractive'
        env['GIT_TERMINAL_PROMPT'] = '0'
        env['NPM_CONFIG_YES'] = 'true'
        env['YARN_ENABLE_IMMUTABLE_INSTALLS'] = 'false'

        # Start process with stdin=DEVNULL to prevent interactive blocking
        process = subprocess.Popen(
            _build_shell_command(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=cwd,
            env=env
        )

        # Start output reader thread
        reader_thread = threading.Thread(
            target=self._read_output,
            args=(process, output_queue, shell_id),
            daemon=True
        )
        reader_thread.start()

        self.shells[shell_id] = {
            'process': process,
            'command': command,
            'cwd': cwd,
            'start_time': time.time(),
            'reader_thread': reader_thread
        }

        return shell_id

    def _read_output(self, process, output_queue, shell_id):
        """Read output from process in background"""
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        # Process terminated
                        break
                    time.sleep(0.01)
                    continue

                # Add to queue and all_output
                output_queue.put(line)
                self.output_buffers[shell_id]['all_output'].append(line)

        except Exception as e:
            output_queue.put(f"Error reading output: {str(e)}\n")

    def get_output(self, shell_id: str, filter_regex: str | None = None) -> dict[str, Any]:
        """Get new output from shell"""
        if shell_id not in self.shells:
            return {
                'status': 'not_found',
                'error': f"Shell {shell_id} not found"
            }

        shell_info = self.shells[shell_id]
        buffer_info = self.output_buffers[shell_id]

        # Get all new lines since last read
        all_output = buffer_info['all_output']
        last_index = buffer_info['last_read_index']
        new_lines = all_output[last_index:]

        # Update last read index
        buffer_info['last_read_index'] = len(all_output)

        # Apply filter if provided
        if filter_regex and new_lines:
            try:
                pattern = re.compile(filter_regex)
                new_lines = [line for line in new_lines if pattern.search(line)]
            except re.error:
                return {
                    'status': 'error',
                    'error': f"Invalid regex pattern: {filter_regex}"
                }

        # Check process status
        process = shell_info['process']
        is_running = process.poll() is None

        return {
            'status': 'running' if is_running else 'completed',
            'exit_code': process.returncode if not is_running else None,
            'output': ''.join(new_lines),
            'lines_read': len(new_lines),
            'total_lines': len(all_output)
        }

    def kill_shell(self, shell_id: str) -> bool:
        """Kill a background shell"""
        if shell_id not in self.shells:
            return False

        shell_info = self.shells[shell_id]
        process = shell_info['process']

        # Terminate process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        # Clean up
        del self.shells[shell_id]
        del self.output_buffers[shell_id]

        return True

    def list_shells(self) -> dict[str, dict[str, Any]]:
        """List all active shells"""
        result = {}
        for shell_id, info in self.shells.items():
            process = info['process']
            is_running = process.poll() is None
            runtime = time.time() - info['start_time']

            result[shell_id] = {
                'command': info['command'],
                'cwd': info['cwd'],
                'status': 'running' if is_running else 'completed',
                'runtime': f"{runtime:.1f}s",
                'exit_code': process.returncode if not is_running else None
            }
        return result


class BashBackgroundTool(BaseTool):
    """Tool for running bash commands in background"""

    name = "bash_background"
    description = "Run bash command in background"
    parameters = {
        "command": "Command to execute (required)",
        "cwd": "Working directory (optional)",
        "description": "Short description of the command"
    }

    def __init__(self, logger=None):
        super().__init__(logger)
        self.manager = BackgroundShellManager()

    def execute(
        self,
        command: str,
        cwd: str | None = None,
        description: str | None = None
    ) -> str:
        """Start command in background"""
        self.validate_params(command=command)

        # Security checks using centralized SecurityValidator
        from sepilot.utils.security import SecurityValidator
        is_safe, error_msg = SecurityValidator.validate_shell_command(command)
        if not is_safe:
            return f"Error: {error_msg}"

        # Resolve working directory
        if cwd:
            cwd_path = Path(cwd).resolve()
            if not cwd_path.exists():
                return f"Error: Working directory not found: {cwd}"
            if not cwd_path.is_dir():
                return f"Error: Not a directory: {cwd}"
            cwd = str(cwd_path)

        try:
            # Create background shell
            shell_id = self.manager.create_shell(command, cwd)

            result = [
                f"🚀 Started background shell: {shell_id}",
                f"Command: {command}"
            ]

            if cwd:
                result.append(f"Working directory: {cwd}")

            if description:
                result.append(f"Description: {description}")

            result.extend([
                "",
                "Use these tools to manage this shell:",
                f"  • bash_output(bash_id='{shell_id}') - Get output",
                f"  • kill_shell(shell_id='{shell_id}') - Terminate",
                "  • list_shells() - List all background shells"
            ])

            return '\n'.join(result)

        except Exception as e:
            return f"Error starting background shell: {str(e)}"


class BashOutputTool(BaseTool):
    """Tool for reading output from background shells"""

    name = "bash_output"
    description = "Get output from a background shell"
    parameters = {
        "bash_id": "ID of the background shell (required)",
        "filter": "Optional regex to filter output lines"
    }

    def __init__(self, logger=None):
        super().__init__(logger)
        self.manager = BackgroundShellManager()

    def execute(
        self,
        bash_id: str,
        filter: str | None = None
    ) -> str:
        """Get output from background shell"""
        self.validate_params(bash_id=bash_id)

        result = self.manager.get_output(bash_id, filter)

        if result['status'] == 'not_found' or result['status'] == 'error':
            return result['error']

        # Format output
        output_lines = [f"📊 Shell {bash_id} output:"]

        if result['status'] == 'completed':
            output_lines.append(f"Status: ✅ Completed (exit code: {result['exit_code']})")
        else:
            output_lines.append("Status: 🔄 Running")

        output_lines.append(f"Lines: {result['lines_read']} new / {result['total_lines']} total")

        if filter:
            output_lines.append(f"Filter: {filter}")

        if result['output']:
            output_lines.extend([
                "",
                "Output:",
                "─" * 40,
                result['output'].rstrip(),
                "─" * 40
            ])
        else:
            output_lines.append("\n(No new output)")

        return '\n'.join(output_lines)


class KillShellTool(BaseTool):
    """Tool for killing background shells"""

    name = "kill_shell"
    description = "Terminate a background shell"
    parameters = {
        "shell_id": "ID of the shell to kill (required)"
    }

    def __init__(self, logger=None):
        super().__init__(logger)
        self.manager = BackgroundShellManager()

    def execute(self, shell_id: str) -> str:
        """Kill a background shell"""
        self.validate_params(shell_id=shell_id)

        if self.manager.kill_shell(shell_id):
            return f"✅ Terminated shell: {shell_id}"
        else:
            return f"Error: Shell {shell_id} not found"


class ListShellsTool(BaseTool):
    """Tool for listing all background shells"""

    name = "list_shells"
    description = "List all active background shells"
    parameters = {}

    def __init__(self, logger=None):
        super().__init__(logger)
        self.manager = BackgroundShellManager()

    def execute(self) -> str:
        """List all background shells"""
        shells = self.manager.list_shells()

        if not shells:
            return "No active background shells"

        result = ["📋 Active background shells:\n"]
        for shell_id, info in shells.items():
            status_icon = "🔄" if info['status'] == 'running' else "✅"
            result.append(f"{status_icon} {shell_id}:")
            result.append(f"   Command: {info['command']}")
            if info['cwd']:
                result.append(f"   Directory: {info['cwd']}")
            result.append(f"   Runtime: {info['runtime']}")
            if info['exit_code'] is not None:
                result.append(f"   Exit code: {info['exit_code']}")
            result.append("")

        return '\n'.join(result)
