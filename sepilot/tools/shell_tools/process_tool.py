"""Process management tool for background tasks"""

import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from typing import Any

from sepilot.tools.base_tool import BaseTool


def _build_shell_command(command: str) -> list[str]:
    """Build a platform-safe shell invocation command."""
    if os.name == "nt":
        return ["cmd", "/c", command]

    shell_path = os.environ.get("SHELL")
    if shell_path and os.path.exists(shell_path):
        return [shell_path, "-lc", command]
    if os.path.exists("/bin/bash"):
        return ["/bin/bash", "-lc", command]
    return ["/bin/sh", "-c", command]


class ProcessTool(BaseTool):
    """Tool for managing background processes"""

    name = "process"
    description = "Manage background processes (start, stop, list)"
    parameters = {
        "action": "Action to perform (start/stop/list/output) (required)",
        "command": "Command to run (for start action)",
        "pid": "Process ID (for stop/output actions)",
        "timeout": "Timeout in seconds (default: no timeout)"
    }

    # Class-level storage for background processes (shared across calls intentionally)
    _processes: dict[int, dict[str, Any]] = {}
    _initialized: bool = False

    def execute(
        self,
        action: str,
        command: str | None = None,
        pid: int | None = None,
        timeout: int | None = None
    ) -> str:
        """Execute process management actions"""
        self.validate_params(action=action)

        try:
            if action == "start":
                return self._start_process(command, timeout)
            elif action == "stop":
                return self._stop_process(pid)
            elif action == "list":
                return self._list_processes()
            elif action == "output":
                return self._get_output(pid)
            else:
                return f"Error: Unknown action: {action}"

        except Exception as e:
            return f"Process error: {str(e)}"

    def _start_process(self, command: str | None, timeout: int | None) -> str:
        """Start a background process"""
        if not command:
            return "Error: Command is required for start action"

        try:
            # Start process in background
            process = subprocess.Popen(
                _build_shell_command(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )

            # Store process info
            proc_info = {
                "process": process,
                "command": command,
                "start_time": datetime.now(),
                "timeout": timeout,
                "output": [],
                "errors": [],
                "timer": None
            }

            # If timeout is set, schedule termination
            if timeout:
                def timeout_handler():
                    """Terminate process when timeout expires"""
                    try:
                        if process.poll() is None:  # Still running
                            process.terminate()
                            time.sleep(1)
                            if process.poll() is None:  # Still not dead
                                process.kill()
                            proc_info["errors"].append(f"Process terminated after {timeout}s timeout\n")
                    except Exception as e:
                        proc_info["errors"].append(f"Timeout handler error: {str(e)}\n")

                timer = threading.Timer(timeout, timeout_handler)
                timer.daemon = True  # Don't prevent program exit
                timer.start()
                proc_info["timer"] = timer

            self._processes[process.pid] = proc_info

            return f"Started process with PID: {process.pid}\nCommand: {command}" + \
                   (f"\nTimeout: {timeout}s" if timeout else "")

        except Exception as e:
            return f"Failed to start process: {str(e)}"

    def _stop_process(self, pid: int | None) -> str:
        """Stop a background process"""
        if pid is None:
            return "Error: PID is required for stop action"

        if pid not in self._processes:
            # Try to kill system process
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)  # Check if still running
                    os.kill(pid, signal.SIGKILL)  # Force kill
                except ProcessLookupError:
                    pass
                return f"Stopped process {pid}"
            except ProcessLookupError:
                return f"Process {pid} not found"
            except PermissionError:
                return f"Permission denied to stop process {pid}"

        # Stop managed process
        proc_info = self._processes[pid]
        process = proc_info["process"]

        # Cancel timeout timer if exists
        if proc_info.get("timer"):
            proc_info["timer"].cancel()

        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        # Collect final output
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                proc_info["output"].append(stdout)
            if stderr:
                proc_info["errors"].append(stderr)
        except Exception:
            pass  # Process already terminated

        # Remove from tracking
        del self._processes[pid]

        return f"Stopped process {pid}\nExit code: {process.returncode}"

    def _list_processes(self) -> str:
        """List all managed background processes"""
        if not self._processes:
            return "No managed background processes running"

        result = ["Managed background processes:\n"]
        for pid, info in self._processes.items():
            process = info["process"]
            status = "Running" if process.poll() is None else f"Exited ({process.returncode})"
            runtime = datetime.now() - info["start_time"]

            result.append(f"PID {pid}:")
            result.append(f"  Command: {info['command']}")
            result.append(f"  Status: {status}")
            result.append(f"  Runtime: {runtime}")
            result.append("")

        # Also show system processes for current user
        try:
            ps_result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if ps_result.returncode == 0:
                lines = ps_result.stdout.strip().split('\n')
                try:
                    username = os.getlogin()
                except OSError:
                    import getpass
                    username = getpass.getuser()
                user_processes = [line for line in lines[1:] if username in line][:5]
                if user_processes:
                    result.append("\nRecent system processes (top 5):")
                    for line in user_processes:
                        parts = line.split(None, 10)
                        if len(parts) > 10:
                            result.append(f"  PID {parts[1]}: {parts[10][:50]}")
        except Exception as e:
            import logging
            logging.debug(f"Failed to list system processes: {e}")

        return "\n".join(result)

    def _get_output(self, pid: int | None) -> str:
        """Get output from a background process"""
        if pid is None:
            return "Error: PID is required for output action"

        if pid not in self._processes:
            return f"Process {pid} not managed by this tool"

        proc_info = self._processes[pid]
        process = proc_info["process"]

        # Check if process is still running
        if process.poll() is None:
            # Try to get non-blocking output
            import select
            if hasattr(select, 'select'):
                ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                for stream in ready:
                    line = stream.readline()
                    if line:
                        if stream == process.stdout:
                            proc_info["output"].append(line)
                        else:
                            proc_info["errors"].append(line)

        # Format output
        result = [f"Output for process {pid}:"]

        if proc_info["output"]:
            result.append("\nStandard output:")
            result.extend(proc_info["output"][-20:])  # Last 20 lines

        if proc_info["errors"]:
            result.append("\nError output:")
            result.extend(proc_info["errors"][-10:])  # Last 10 error lines

        if not proc_info["output"] and not proc_info["errors"]:
            result.append("No output yet")

        return "\n".join(result)
