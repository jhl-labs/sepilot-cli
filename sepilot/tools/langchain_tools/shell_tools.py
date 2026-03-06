"""Shell operation tools for LangChain agent.

Provides bash_background, bash_output, kill_shell, list_shells tools.
These are thin wrappers around the background shell implementation.
"""


from langchain_core.tools import tool


@tool
def bash_background(command: str, cwd: str | None = None, description: str | None = None) -> str:
    """Run a bash command in the background.

    Args:
        command: Command to execute (required)
        cwd: Working directory (optional)
        description: Short description of the command (optional)

    Returns:
        Shell ID and management instructions

    Examples:
        # Start a long-running server
        bash_background(command="python -m http.server 8000", description="Start web server")

        # Run tests in background
        bash_background(command="pytest -v", cwd="tests", description="Run test suite")
    """
    from sepilot.tools.shell_tools.background_shell import BashBackgroundTool
    tool_instance = BashBackgroundTool()
    return tool_instance.execute(command, cwd, description)


@tool
def bash_output(bash_id: str, filter: str | None = None) -> str:
    """Get output from a background shell.

    Args:
        bash_id: ID of the background shell (required)
        filter: Optional regex to filter output lines

    Returns:
        Shell output and status

    Examples:
        # Get all output
        bash_output(bash_id="shell_abc123")

        # Filter for errors
        bash_output(bash_id="shell_abc123", filter="ERROR|WARNING")
    """
    from sepilot.tools.shell_tools.background_shell import BashOutputTool
    tool_instance = BashOutputTool()
    return tool_instance.execute(bash_id, filter)


@tool
def kill_shell(shell_id: str) -> str:
    """Terminate a background shell.

    Args:
        shell_id: ID of the shell to kill (required)

    Returns:
        Success or error message

    Examples:
        kill_shell(shell_id="shell_abc123")
    """
    from sepilot.tools.shell_tools.background_shell import KillShellTool
    tool_instance = KillShellTool()
    return tool_instance.execute(shell_id)


@tool
def list_shells() -> str:
    """List all active background shells.

    Returns:
        List of active shells with their status

    Examples:
        list_shells()
    """
    from sepilot.tools.shell_tools.background_shell import ListShellsTool
    tool_instance = ListShellsTool()
    return tool_instance.execute()


__all__ = ['bash_background', 'bash_output', 'kill_shell', 'list_shells']
