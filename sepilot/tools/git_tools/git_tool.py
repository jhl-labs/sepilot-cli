"""Git operations tool"""

import subprocess
from pathlib import Path

from sepilot.tools.base_tool import BaseTool


class GitTool(BaseTool):
    """Tool for Git operations"""

    name = "git"
    description = "Perform Git operations (status, diff, commit, etc.)"
    parameters = {
        "operation": "Git operation to perform (status/diff/add/commit/log/branch) (required)",
        "args": "Additional arguments for the operation",
        "message": "Commit message (for commit operation)",
        "path": "Path to Git repository (default: current directory)"
    }

    def execute(
        self,
        operation: str,
        args: str | None = None,
        message: str | None = None,
        path: str = "."
    ) -> str:
        """Execute Git operations"""
        self.validate_params(operation=operation)

        # Check if directory is a git repository
        repo_path = Path(path)
        if not (repo_path / ".git").exists():
            return f"Error: Not a Git repository: {path}"

        try:
            if operation == "status":
                return self._git_status(repo_path, args)
            elif operation == "diff":
                return self._git_diff(repo_path, args)
            elif operation == "add":
                return self._git_add(repo_path, args)
            elif operation == "commit":
                return self._git_commit(repo_path, message, args)
            elif operation == "log":
                return self._git_log(repo_path, args)
            elif operation == "branch":
                return self._git_branch(repo_path, args)
            else:
                return f"Error: Unknown operation: {operation}"

        except Exception as e:
            return f"Git error: {str(e)}"

    def _run_git_command(self, cmd: list, cwd: Path) -> tuple[int, str, str]:
        """Run a git command and return result"""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr

    def _git_status(self, repo_path: Path, args: str | None) -> str:
        """Get git status"""
        cmd = ["git", "status", "--short"]
        if args:
            cmd.extend(args.split())

        returncode, stdout, stderr = self._run_git_command(cmd, repo_path)

        if returncode != 0:
            return f"Error: {stderr}"

        if not stdout.strip():
            return "Working directory clean - no changes"

        lines = stdout.strip().split('\n')
        result = ["Git status:"]

        # Parse and format status
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

    def _git_diff(self, repo_path: Path, args: str | None) -> str:
        """Get git diff"""
        cmd = ["git", "diff", "--stat"]
        if args:
            if "--" not in args:  # Add full diff if not just stats
                cmd = ["git", "diff"]
            cmd.extend(args.split())

        returncode, stdout, stderr = self._run_git_command(cmd, repo_path)

        if returncode != 0:
            return f"Error: {stderr}"

        if not stdout.strip():
            return "No differences found"

        # Limit output for large diffs
        lines = stdout.strip().split('\n')
        if len(lines) > 100:
            return "\n".join(lines[:100]) + f"\n... ({len(lines)-100} more lines)"
        return stdout

    def _git_add(self, repo_path: Path, args: str | None) -> str:
        """Stage files"""
        if not args:
            return "Error: Please specify files to add (use '.' for all)"

        cmd = ["git", "add"]
        cmd.extend(args.split())

        returncode, stdout, stderr = self._run_git_command(cmd, repo_path)

        if returncode != 0:
            return f"Error: {stderr}"

        # Show what was staged
        return f"Staged files: {args}\n" + self._git_status(repo_path, None)

    def _git_commit(self, repo_path: Path, message: str | None, args: str | None) -> str:
        """Create a commit"""
        if not message:
            return "Error: Commit message is required"

        # Check if using --amend
        is_amend = args and '--amend' in args

        # For regular commits (not --amend), check if there are staged changes
        if not is_amend:
            # Check if there are staged changes
            status_code, status_out, _ = self._run_git_command(["git", "diff", "--cached", "--quiet"], repo_path)
            if status_code == 0:  # No staged changes
                return "Error: No staged changes to commit. Use 'git add' to stage files first."

        cmd = ["git", "commit", "-m", message]
        if args:
            cmd.extend(args.split())

        returncode, stdout, stderr = self._run_git_command(cmd, repo_path)

        if returncode != 0:
            # Provide more helpful error messages
            error_msg = stderr.strip()
            if "nothing to commit" in error_msg or "no changes added to commit" in error_msg:
                return "Error: No staged changes to commit. Use 'git add <file>' to stage files before committing."
            elif is_amend and "nothing to commit" in error_msg:
                return "Error: Cannot amend - no staged changes and no previous commit to amend."
            return f"Error: {error_msg}"

        # Extract commit info
        lines = stdout.strip().split('\n')
        commit_info = lines[0] if lines else "Commit created"

        return f"✅ {commit_info}\n\nCommit message: {message}"

    def _git_log(self, repo_path: Path, args: str | None) -> str:
        """Show commit history"""
        cmd = ["git", "log", "--oneline", "-10"]  # Default to last 10 commits
        if args:
            cmd = ["git", "log"]
            cmd.extend(args.split())

        returncode, stdout, stderr = self._run_git_command(cmd, repo_path)

        if returncode != 0:
            return f"Error: {stderr}"

        if not stdout.strip():
            return "No commits found"

        return f"Commit history:\n{stdout}"

    def _git_branch(self, repo_path: Path, args: str | None) -> str:
        """Manage branches"""
        cmd = ["git", "branch"]
        if args:
            cmd.extend(args.split())

        returncode, stdout, stderr = self._run_git_command(cmd, repo_path)

        if returncode != 0:
            return f"Error: {stderr}"

        if not args:  # List branches
            return f"Branches:\n{stdout}"
        else:
            return stdout if stdout else f"Branch operation completed: {args}"
