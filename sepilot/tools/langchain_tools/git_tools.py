"""Git operation tools for LangChain agent."""

import subprocess
from pathlib import Path

from langchain_core.tools import tool


@tool
def git(operation: str, args: str = "", message: str = "", path: str = ".") -> str:
    """Perform Git operations (status, diff, add, commit, log, branch).

    Args:
        operation: Git operation to perform (status/diff/add/commit/log/branch)
        args: Additional arguments for the operation (optional)
        message: Commit message (required for commit operation)
        path: Path to Git repository (default: current directory)

    Returns:
        Git command output or error message

    Examples:
        - git(operation="status")
        - git(operation="diff")
        - git(operation="add", args=".")
        - git(operation="commit", message="Fix bug")
        - git(operation="log", args="--oneline -5")
    """
    repo_path = Path(path)
    if not (repo_path / ".git").exists():
        return f"Error: Not a Git repository: {path}"

    def _run_git_command(cmd: list, cwd: Path) -> tuple:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr

    try:
        if operation == "status":
            cmd = ["git", "status", "--short"]
            if args:
                cmd.extend(args.split())

            returncode, stdout, stderr = _run_git_command(cmd, repo_path)

            if returncode != 0:
                return f"Error: {stderr}"

            if not stdout.strip():
                return "Working directory clean - no changes"

            lines = stdout.strip().split('\n')
            result = ["Git status:"]

            for line in lines:
                if line.startswith("??"):
                    result.append(f"  🆕 Untracked: {line[3:]}")
                elif line.startswith(" M"):
                    result.append(f"  📝 Modified: {line[3:]}")
                elif line.startswith("M "):
                    result.append(f"  ✅ Staged: {line[3:]}")
                elif line.startswith("A "):
                    result.append(f"  ➕ Added: {line[3:]}")
                elif line.startswith("D "):
                    result.append(f"  ➖ Deleted: {line[3:]}")
                else:
                    result.append(f"  {line}")

            return "\n".join(result)

        elif operation == "diff":
            cmd = ["git", "diff", "--stat"]
            if args:
                if "--" not in args:
                    cmd = ["git", "diff"]
                cmd.extend(args.split())

            returncode, stdout, stderr = _run_git_command(cmd, repo_path)

            if returncode != 0:
                return f"Error: {stderr}"

            if not stdout.strip():
                return "No differences found"

            lines = stdout.strip().split('\n')
            if len(lines) > 100:
                return "\n".join(lines[:100]) + f"\n... ({len(lines)-100} more lines)"
            return stdout

        elif operation == "add":
            if not args:
                return "Error: Please specify files to add (use '.' for all)"

            cmd = ["git", "add"]
            cmd.extend(args.split())

            returncode, stdout, stderr = _run_git_command(cmd, repo_path)

            if returncode != 0:
                return f"Error: {stderr}"

            return f"Staged files: {args}"

        elif operation == "commit":
            if not message:
                return "Error: Commit message is required"

            cmd = ["git", "commit", "-m", message]
            if args:
                cmd.extend(args.split())

            returncode, stdout, stderr = _run_git_command(cmd, repo_path)

            if returncode != 0:
                return f"Error: {stderr}"

            lines = stdout.strip().split('\n')
            commit_info = lines[0] if lines else "Commit created"

            return f"✅ {commit_info}\n\nCommit message: {message}"

        elif operation == "log":
            cmd = ["git", "log", "--oneline", "-10"]
            if args:
                cmd = ["git", "log"]
                cmd.extend(args.split())

            returncode, stdout, stderr = _run_git_command(cmd, repo_path)

            if returncode != 0:
                return f"Error: {stderr}"

            if not stdout.strip():
                return "No commits found"

            return f"Commit history:\n{stdout}"

        elif operation == "branch":
            cmd = ["git", "branch"]
            if args:
                cmd.extend(args.split())

            returncode, stdout, stderr = _run_git_command(cmd, repo_path)

            if returncode != 0:
                return f"Error: {stderr}"

            if not args:
                return f"Branches:\n{stdout}"
            else:
                return stdout if stdout else f"Branch operation completed: {args}"

        else:
            return f"Error: Unknown operation: {operation}. Valid: status/diff/add/commit/log/branch"

    except Exception as e:
        return f"Git error: {str(e)}"


__all__ = ['git']
