"""Bash execution tools for LangChain agent.

Provides the main bash_execute tool with security checks and real-time output.
"""

import contextlib
import getpass
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from langchain_core.tools import tool

# Dangerous command patterns to block
DANGEROUS_PATTERNS = [
    r'rm\s+-rf\s+/',
    r'rm\s+-fr\s+/',
    r'rm\s+--recursive\s+--force\s+/',
    r'dd\s+if=/dev/(zero|random|urandom)',
    r'>\s*/dev/(sd[a-z]|hd[a-z]|nvme)',
    r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;',
    r'mkfs\.',
    r'fdisk',
    r'parted',
    r'su\s+',
    r'curl.*\|.*sh',
    r'wget.*\|.*sh',
    r'curl.*\|.*bash',
    r'wget.*\|.*bash',
    r'/etc/(passwd|shadow|sudoers)',
    r'chmod\s+(777|666)',
    r'&\s*rm\s+-rf',
    r'&&\s*rm\s+-rf\s+/',
    r';\s*rm\s+-rf\s+/',
]

# Dev server patterns for auto-background detection
DEV_SERVER_PATTERNS = [
    'npm run dev', 'npm start', 'yarn dev', 'yarn start',
    'pnpm dev', 'pnpm start', 'bun dev', 'bun run dev',
    'vite', 'next dev', 'nuxt dev', 'gatsby develop',
    'python -m http.server', 'python3 -m http.server',
    'flask run', 'uvicorn', 'gunicorn', 'django runserver',
    'node server', 'nodemon', 'ts-node',
    'cargo run', 'go run',
]

# Ready signal patterns for dev servers
READY_PATTERNS = [
    'ready in', 'listening on', 'listening at', 'server running',
    'started server', 'local:', 'localhost:', '127.0.0.1:',
    'development server', 'compiled successfully', 'webpack compiled',
    'vite v', 'ready', 'started on port', 'serving on',
    'application startup complete', 'uvicorn running',
]

_SUDO_CMD_PATTERN = re.compile(r"^(?:/usr/bin/|/usr/local/bin/)?sudo(?:\s|$)", re.IGNORECASE)


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


def _check_dangerous_command(command: str) -> str | None:
    """Check if command matches dangerous patterns. Returns error message or None."""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return "Error: Dangerous command pattern detected. Command blocked for security."

    # Check for command injection attempts
    injection_patterns = [r'\$\(.*\)', r'`.*`']
    for pattern in injection_patterns:
        matches = re.findall(pattern, command)
        for match in matches:
            safe_commands = ['pwd', 'date', 'whoami', 'hostname']
            if not any(safe_cmd in match.lower() for safe_cmd in safe_commands):
                if any(danger in match.lower() for danger in ['rm', 'curl', 'wget', 'cat', '/etc/']):
                    return "Error: Potential command injection detected. Command blocked for security."

    return None


def _is_sudo_command(command: str) -> bool:
    return bool(_SUDO_CMD_PATTERN.match(command.strip()))


def _prepare_sudo_command(command: str) -> str:
    """Ensure sudo command uses stdin password mode with silent prompt."""
    stripped = command.lstrip()
    leading_ws = command[: len(command) - len(stripped)]
    match = _SUDO_CMD_PATTERN.match(stripped)
    if not match:
        return command

    sudo_token = match.group(0).strip()
    rest = stripped[match.end():].lstrip()

    has_stdin_password = bool(re.search(r"(^|\s)-S(\s|$)", rest))
    has_prompt_option = bool(re.search(r"(^|\s)-p(\s|$)", rest))

    option_parts: list[str] = []
    if not has_stdin_password:
        option_parts.append("-S")
    if not has_prompt_option:
        option_parts.append('-p ""')

    if option_parts:
        options = " ".join(option_parts)
        return f"{leading_ws}{sudo_token} {options} {rest}".rstrip()
    return command


@tool
def bash_execute(command: str, timeout: int = 30, cwd: str | None = None) -> str:
    """Execute a bash command in a specified directory.

    Args:
        command: Bash command to execute
        timeout: Timeout in seconds (default: 30)
        cwd: Working directory for command execution (default: current directory)
            Example: cwd="sepilot" to run command in sepilot directory

    Returns:
        Command output or error

    Examples:
        - bash_execute(command="cloc .", cwd="sepilot")
        - bash_execute(command="pytest", cwd="tests")
        - bash_execute(command="ls -la")
    """
    from rich.console import Console

    from sepilot.agent.tool_tracker import complete_process_if_enabled, start_process_if_enabled

    console = Console()
    process_id = f"bash_{uuid.uuid4().hex[:8]}"
    command_to_run = command
    sudo_password: str | None = None

    # Security check
    security_error = _check_dangerous_command(command)
    if security_error:
        complete_process_if_enabled(process_id, exit_code=-1, output="", error_output=security_error)
        return security_error

    if _is_sudo_command(command):
        if not sys.stdin.isatty():
            return "Error: sudo command requires an interactive terminal for password input."
        try:
            sudo_password = getpass.getpass("sudo password: ")
        except Exception:
            return "Error: Failed to read sudo password."
        if not sudo_password:
            return "Error: Empty sudo password. Command cancelled."
        command_to_run = _prepare_sudo_command(command)

    # Record process start
    start_process_if_enabled(command, process_id)

    try:
        # Validate working directory
        work_dir = None
        if cwd:
            cwd_path = Path(cwd)
            if not cwd_path.is_absolute():
                cwd_path = Path.cwd() / cwd_path
            cwd_path = cwd_path.resolve()

            if not cwd_path.exists():
                error_msg = f"Error: Working directory not found: {cwd}"
                console.print(f"[red]✗ {error_msg}[/red]\n")
                complete_process_if_enabled(process_id, exit_code=-1, output="", error_output=error_msg)
                return error_msg

            if not cwd_path.is_dir():
                error_msg = f"Error: Not a directory: {cwd}"
                console.print(f"[red]✗ {error_msg}[/red]\n")
                complete_process_if_enabled(process_id, exit_code=-1, output="", error_output=error_msg)
                return error_msg

            work_dir = str(cwd_path)

        # Print execution header
        console.print(f"\n[bold cyan]🔧 Executing:[/bold cyan] [white]{command}[/white]")
        if work_dir:
            console.print(f"[dim]📁 Working directory:[/dim] [yellow]{work_dir}[/yellow]")
        console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")

        # Set up non-interactive environment
        env = os.environ.copy()
        env['CI'] = 'true'
        env['DEBIAN_FRONTEND'] = 'noninteractive'
        env['GIT_TERMINAL_PROMPT'] = '0'
        env['NPM_CONFIG_YES'] = 'true'
        env['YARN_ENABLE_IMMUTABLE_INSTALLS'] = 'false'

        # Check if this is a dev server command
        is_dev_server = any(pattern in command.lower() for pattern in DEV_SERVER_PATTERNS)

        # Execute command
        process = subprocess.Popen(
            _build_shell_command(command_to_run),
            stdin=subprocess.PIPE if sudo_password is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=work_dir,
            env=env
        )

        output_lines = []
        start_time = time.time()
        interrupted = False

        # Set up interrupt handler
        old_handler = None
        def handle_interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True
            console.print("\n[yellow]⚠️  Ctrl+C detected - terminating process...[/yellow]")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        try:
            if threading.current_thread() is threading.main_thread():
                old_handler = signal.signal(signal.SIGINT, handle_interrupt)
        except (ValueError, RuntimeError):
            pass

        try:
            server_ready = False
            if sudo_password is not None and process.stdin is not None:
                process.stdin.write(f"{sudo_password}\n")
                process.stdin.flush()
                process.stdin.close()
                sudo_password = None

            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    console.print(f"\n[red]⏱️  Timeout after {timeout}s - terminating...[/red]")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break

                # Read output
                line = process.stdout.readline()
                if line:
                    console.print(f"[cyan]{line.rstrip()}[/cyan]")
                    output_lines.append(line)

                    # Check for dev server ready
                    if is_dev_server and not server_ready:
                        if any(pattern in line.lower() for pattern in READY_PATTERNS):
                            server_ready = True
                            time.sleep(0.5)

                            # Read remaining buffered output
                            while True:
                                extra_line = process.stdout.readline()
                                if extra_line:
                                    console.print(f"[cyan]{extra_line.rstrip()}[/cyan]")
                                    output_lines.append(extra_line)
                                else:
                                    break

                            # Transfer to background
                            return _transfer_to_background(process, command, work_dir, start_time, output_lines, console)

                elif process.poll() is not None:
                    break
                else:
                    time.sleep(0.01)

            # Get remaining output
            remaining = process.stdout.read()
            if remaining:
                for line in remaining.splitlines():
                    console.print(f"[cyan]{line}[/cyan]")
                    output_lines.append(line + '\n')

            exit_code = process.returncode

        finally:
            if old_handler is not None:
                with contextlib.suppress(ValueError, RuntimeError):
                    signal.signal(signal.SIGINT, old_handler)

        console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")

        output = ''.join(output_lines)

        if interrupted:
            error_msg = "Command interrupted by user (Ctrl+C)"
            console.print(f"[yellow]✗ {error_msg}[/yellow]\n")
            complete_process_if_enabled(process_id, exit_code=-2, output=output[:5000], error_output=error_msg)
            return f"{error_msg}\n\nPartial output:\n{output[:5000]}"

        # Smart error detection
        output_lower = output.lower()
        suggestions = _analyze_output_for_errors(output_lower, command)

        if exit_code == 0 and not suggestions:
            console.print(f"[green]✓ Command completed successfully (exit code: {exit_code})[/green]\n")
        elif exit_code == 0:
            console.print(f"[yellow]⚠️ Command completed with warnings (exit code: {exit_code})[/yellow]\n")
        else:
            console.print(f"[red]✗ Command failed (exit code: {exit_code})[/red]\n")

        complete_process_if_enabled(process_id, exit_code=exit_code, output=output[:5000], error_output="")

        result = output[:5000]
        if suggestions:
            result += "\n\n" + "─" * 50 + "\n🤖 SMART ANALYSIS:\n"
            for s in suggestions[:3]:
                result += f"  • {s}\n"
            result += "─" * 50

        return result

    except subprocess.TimeoutExpired:
        error_msg = f"Error: Command timed out after {timeout} seconds"
        console.print(f"[red]✗ {error_msg}[/red]\n")
        complete_process_if_enabled(process_id, exit_code=-1, output="", error_output=error_msg)
        return error_msg

    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        console.print(f"[red]✗ {error_msg}[/red]\n")
        complete_process_if_enabled(process_id, exit_code=-1, output="", error_output=error_msg)
        return error_msg


def _transfer_to_background(process, command, work_dir, start_time, output_lines, console):
    """Transfer a running process to background shell manager."""
    import queue

    from sepilot.tools.shell_tools.background_shell import BackgroundShellManager

    manager = BackgroundShellManager()
    shell_id = f"server_{uuid.uuid4().hex[:8]}"

    manager.shells[shell_id] = {
        'process': process,
        'command': command,
        'cwd': work_dir,
        'start_time': start_time,
        'reader_thread': None
    }
    manager.output_buffers[shell_id] = {
        'queue': None,
        'all_output': output_lines.copy(),
        'last_read_index': len(output_lines)
    }

    # Start background reader
    output_queue = queue.Queue()
    manager.output_buffers[shell_id]['queue'] = output_queue

    def read_remaining_output():
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue
                output_queue.put(line)
                manager.output_buffers[shell_id]['all_output'].append(line)
        except Exception:
            pass

    reader = threading.Thread(target=read_remaining_output, daemon=True)
    reader.start()
    manager.shells[shell_id]['reader_thread'] = reader

    console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
    console.print("[bold green]✓ Dev server started and running in background[/bold green]")
    console.print(f"[dim]   Shell ID: {shell_id}[/dim]")
    console.print(f"[dim]   Use bash_output(bash_id=\"{shell_id}\") to check output[/dim]")
    console.print(f"[dim]   Use kill_shell(shell_id=\"{shell_id}\") to stop[/dim]\n")

    output = ''.join(output_lines)
    return (
        f"{output}\n\n🚀 Server is running in background (ID: {shell_id})\n"
        f"   • Check output: bash_output(bash_id=\"{shell_id}\")\n"
        f"   • Stop server: kill_shell(shell_id=\"{shell_id}\")"
    )


def _analyze_output_for_errors(output_lower: str, command: str) -> list:
    """Analyze output for common error patterns and return suggestions."""
    suggestions = []

    # Environment mismatch errors
    env_errors = [
        ("process is not defined",
         "🚨 ENVIRONMENT ERROR: Node.js code in BROWSER. Create backend API instead."),
        ("os.type is not a function",
         "🚨 ENVIRONMENT ERROR: Node.js 'os' module in BROWSER. Use backend API."),
        ("fs is not defined",
         "🚨 ENVIRONMENT ERROR: Node.js 'fs' module in BROWSER. Create backend file API."),
        ("require is not defined",
         "🚨 ENVIRONMENT ERROR: CommonJS 'require' in browser/ESM context."),
        ("module externalized for browser",
         "🚨 VITE WARNING: Node.js-only module detected. Use backend API instead."),
    ]

    for pattern, suggestion in env_errors:
        if pattern in output_lower and suggestion not in suggestions:
            suggestions.insert(0, suggestion)

    # Common failure patterns
    failure_patterns = [
        ("operation cancelled", "Operation was cancelled. Check if directory/file exists."),
        ("already exists", "Target exists. Use 'ls' to check, or remove first."),
        ("permission denied", "Permission denied. Check file permissions."),
        ("command not found", "Command not found. Check if tool is installed."),
        ("npm err!", "npm error. Check the error details."),
    ]

    for pattern, suggestion in failure_patterns:
        if pattern in output_lower and suggestion not in suggestions:
            suggestions.append(suggestion)

    # Project creation specific
    cmd_lower = command.lower()
    if any(x in cmd_lower for x in ['create-vite', 'create-react-app', 'create-next-app']):
        if 'cancelled' in output_lower or 'already exists' in output_lower:
            suggestions.append("💡 Check if target directory exists with 'ls -la <path>'.")

    return suggestions


__all__ = ['bash_execute']
