"""Git Agent - AI-powered Git operations

This module provides intelligent git operations with:
- AI-powered commit grouping and message generation
- Analysis of timestamps, folder structure, and diff context
- Human-in-the-loop for all commit operations
- Reference to .SEPILOT.GIT.md for project-specific guidelines
"""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from sepilot.config.settings import Settings
from sepilot.loggers.file_logger import FileLogger


@dataclass
class FileChange:
    """Represents a single file change."""

    path: str
    status: str  # M (modified), A (added), D (deleted), R (renamed), etc.
    diff: str = ""
    lines_added: int = 0
    lines_deleted: int = 0
    folder: str = ""
    timestamp: datetime | None = None


@dataclass
class CommitGroup:
    """A proposed commit group."""

    title: str
    description: str
    files: list[str]
    commit_message: str
    reasoning: str
    priority: int = 0  # Lower is higher priority


@dataclass
class AICommitPlan:
    """Complete AI-generated commit plan."""

    groups: list[CommitGroup] = field(default_factory=list)
    summary: str = ""
    total_files: int = 0
    guidelines_used: str = ""


class GitAgent:
    """AI-powered Git Agent for intelligent git operations.

    Features:
    - Analyzes changes and proposes logical commit groups
    - References .SEPILOT.GIT.md for project-specific commit guidelines
    - Human-in-the-loop approval for all commits
    - Senior developer-level commit messages
    """

    def __init__(
        self,
        settings: Settings,
        logger: FileLogger,
        console: Console | None = None,
        working_dir: str | Path | None = None,
    ):
        """Initialize GitAgent.

        Args:
            settings: Application settings
            logger: File logger instance
            console: Rich console for output
            working_dir: Working directory for git operations
        """
        self.settings = settings
        self.logger = logger
        self.console = console or Console()
        self.working_dir = Path(working_dir or ".").resolve()

        # Initialize LLM for AI operations
        self._llm = None

    def _get_llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            from sepilot.llm.factory import create_llm

            self._llm = create_llm(
                model=self.settings.model,
                temperature=0.3,  # Lower temperature for more consistent commit messages
            )
        return self._llm

    def _run_git(self, *args: str, capture: bool = True) -> tuple[bool, str]:
        """Run a git command.

        Args:
            *args: Git command arguments
            capture: Whether to capture output

        Returns:
            Tuple of (success, output)
        """
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.working_dir,
                capture_output=capture,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def _get_git_status(self) -> list[FileChange]:
        """Get current git status with file changes.

        Returns:
            List of FileChange objects
        """
        changes = []

        # Get status with porcelain format
        success, output = self._run_git("status", "--porcelain", "-uall")
        if not success:
            return changes

        for line in output.strip().split("\n"):
            if not line:
                continue

            status = line[:2].strip()
            path = line[3:].strip()

            # Handle renamed files
            if " -> " in path:
                path = path.split(" -> ")[1]

            change = FileChange(
                path=path,
                status=status,
                folder=str(Path(path).parent) if "/" in path else ".",
            )
            changes.append(change)

        return changes

    def _get_file_diff(self, file_path: str) -> str:
        """Get diff for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Diff output
        """
        # Try staged diff first
        success, diff = self._run_git("diff", "--cached", "--", file_path)
        if success and diff.strip():
            return diff

        # Fall back to unstaged diff
        success, diff = self._run_git("diff", "--", file_path)
        if success:
            return diff

        # For new files, show the content
        success, content = self._run_git("show", f":{file_path}")
        if success:
            return f"New file:\n{content[:2000]}"

        return ""

    def _get_file_timestamp(self, file_path: str) -> datetime | None:
        """Get last modification timestamp for a file.

        Args:
            file_path: Path to the file

        Returns:
            Datetime or None
        """
        try:
            full_path = self.working_dir / file_path
            if full_path.exists():
                return datetime.fromtimestamp(full_path.stat().st_mtime)
        except Exception:
            pass
        return None

    def _load_git_guidelines(self) -> str:
        """Load .SEPILOT.GIT.md guidelines if available.

        Returns:
            Guidelines content or default guidelines
        """
        guidelines_path = self.working_dir / ".SEPILOT.GIT.md"

        if guidelines_path.exists():
            try:
                content = guidelines_path.read_text(encoding="utf-8")
                return f"Project-specific guidelines from .SEPILOT.GIT.md:\n\n{content}"
            except Exception:
                pass

        # Default guidelines
        return """Default Git Commit Guidelines:

## Commit Message Format
- Use conventional commits: type(scope): description
- Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build
- Keep subject line under 50 characters
- Use imperative mood ("Add feature" not "Added feature")
- Separate subject from body with blank line
- Explain what and why, not how

## Grouping Strategy
- Group related changes together
- Separate concerns (e.g., feature vs refactor)
- Keep commits atomic and focused
- Order: infrastructure → core changes → features → tests → docs
"""

    def _analyze_changes(self, changes: list[FileChange]) -> dict:
        """Analyze changes to understand the scope and context.

        Args:
            changes: List of file changes

        Returns:
            Analysis dict with folders, types, patterns
        """
        analysis = {
            "total_files": len(changes),
            "folders": {},
            "file_types": {},
            "status_counts": {"modified": 0, "added": 0, "deleted": 0, "renamed": 0},
            "patterns": [],
        }

        for change in changes:
            # Count by folder
            folder = change.folder
            if folder not in analysis["folders"]:
                analysis["folders"][folder] = []
            analysis["folders"][folder].append(change.path)

            # Count by file type
            ext = Path(change.path).suffix or "no_ext"
            if ext not in analysis["file_types"]:
                analysis["file_types"][ext] = 0
            analysis["file_types"][ext] += 1

            # Count by status
            if "M" in change.status:
                analysis["status_counts"]["modified"] += 1
            elif "A" in change.status or "?" in change.status:
                analysis["status_counts"]["added"] += 1
            elif "D" in change.status:
                analysis["status_counts"]["deleted"] += 1
            elif "R" in change.status:
                analysis["status_counts"]["renamed"] += 1

        # Detect patterns
        if len(analysis["folders"]) == 1:
            analysis["patterns"].append("single_folder")
        if analysis["status_counts"]["added"] == len(changes):
            analysis["patterns"].append("all_new_files")
        if "test" in str(analysis["folders"].keys()).lower():
            analysis["patterns"].append("includes_tests")

        return analysis

    def _generate_commit_plan(
        self, changes: list[FileChange], guidelines: str, analysis: dict
    ) -> AICommitPlan:
        """Use LLM to generate a commit plan.

        Args:
            changes: List of file changes
            guidelines: Git guidelines content
            analysis: Change analysis

        Returns:
            AICommitPlan with proposed commits
        """
        llm = self._get_llm()

        # Build context about changes
        changes_context = []
        for change in changes[:50]:  # Limit for token efficiency
            diff_preview = change.diff[:500] if change.diff else ""
            changes_context.append(
                f"- {change.path} ({change.status})\n"
                f"  Folder: {change.folder}\n"
                f"  Diff preview: {diff_preview[:200]}..."
            )

        changes_text = "\n".join(changes_context)

        system_prompt = f"""You are a senior software engineer helping to create well-organized git commits.

{guidelines}

## Analysis of Current Changes
- Total files: {analysis['total_files']}
- Folders affected: {', '.join(analysis['folders'].keys())}
- File types: {analysis['file_types']}
- Status: {analysis['status_counts']}
- Patterns detected: {analysis['patterns']}

## Your Task
1. Analyze the changes and group them into logical commits
2. Each commit should be atomic and focused on a single concern
3. Write professional commit messages following the guidelines
4. Explain your reasoning for each group

## Output Format
For each commit group, provide:
GROUP [number]:
FILES: file1.py, file2.py
TITLE: type(scope): short description
MESSAGE: |
  Full commit message here

  - Detail 1
  - Detail 2
REASONING: Why these files are grouped together
---
"""

        user_prompt = f"""Please analyze these changes and propose logical commit groups:

{changes_text}

Remember:
- Group related changes together
- Keep commits focused and atomic
- Use conventional commit format
- Explain your grouping decisions"""

        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )

            # Parse the response into CommitGroups
            plan = self._parse_commit_plan(response.content, changes)
            plan.guidelines_used = "Project guidelines" if ".SEPILOT.GIT.md" in guidelines else "Default guidelines"
            plan.total_files = len(changes)

            return plan

        except Exception as e:
            self.console.print(f"[red]Error generating commit plan: {e}[/red]")
            # Return a simple fallback plan
            return AICommitPlan(
                groups=[
                    CommitGroup(
                        title="chore: update files",
                        description="Batch update of changed files",
                        files=[c.path for c in changes],
                        commit_message="chore: update files\n\nBatch commit of all changes",
                        reasoning="Fallback: Could not generate AI plan",
                    )
                ],
                summary="Fallback plan due to error",
                total_files=len(changes),
            )

    def _parse_commit_plan(self, response: str, changes: list[FileChange]) -> AICommitPlan:
        """Parse LLM response into AICommitPlan.

        Args:
            response: LLM response text
            changes: Original list of changes

        Returns:
            Parsed AICommitPlan
        """
        groups = []
        current_group = None
        current_section = None
        message_lines = []

        all_files = {c.path for c in changes}

        for line in response.split("\n"):
            line_stripped = line.strip()

            if line_stripped.startswith("GROUP"):
                if current_group:
                    if message_lines:
                        current_group.commit_message = "\n".join(message_lines).strip()
                    groups.append(current_group)
                current_group = CommitGroup(
                    title="",
                    description="",
                    files=[],
                    commit_message="",
                    reasoning="",
                    priority=len(groups),
                )
                message_lines = []
                current_section = None

            elif line_stripped.startswith("FILES:"):
                current_section = "files"
                files_str = line_stripped[6:].strip()
                if current_group and files_str:
                    # Parse comma-separated files
                    for f in files_str.split(","):
                        f = f.strip()
                        if f and f in all_files:
                            current_group.files.append(f)

            elif line_stripped.startswith("TITLE:"):
                current_section = "title"
                if current_group:
                    current_group.title = line_stripped[6:].strip()

            elif line_stripped.startswith("MESSAGE:"):
                current_section = "message"
                message_lines = []

            elif line_stripped.startswith("REASONING:"):
                current_section = "reasoning"
                if current_group:
                    current_group.reasoning = line_stripped[10:].strip()

            elif line_stripped == "---":
                current_section = None

            elif current_section == "message" and current_group:
                message_lines.append(line.rstrip())

            elif current_section == "reasoning" and current_group and line_stripped:
                current_group.reasoning += " " + line_stripped

        # Don't forget the last group
        if current_group:
            if message_lines:
                current_group.commit_message = "\n".join(message_lines).strip()
            groups.append(current_group)

        # Validate and fix groups
        assigned_files = set()
        valid_groups = []

        for group in groups:
            if group.files and group.title:
                # Remove already assigned files
                group.files = [f for f in group.files if f not in assigned_files]
                if group.files:
                    assigned_files.update(group.files)
                    if not group.commit_message:
                        group.commit_message = group.title
                    valid_groups.append(group)

        # Handle unassigned files
        unassigned = all_files - assigned_files
        if unassigned:
            valid_groups.append(
                CommitGroup(
                    title="chore: update remaining files",
                    description="Files not grouped by AI",
                    files=list(unassigned),
                    commit_message="chore: update remaining files\n\nMiscellaneous changes",
                    reasoning="Remaining files not matched to AI groups",
                    priority=len(valid_groups),
                )
            )

        return AICommitPlan(
            groups=valid_groups,
            summary=f"Generated {len(valid_groups)} commit groups",
        )

    def _display_commit_plan(self, plan: AICommitPlan):
        """Display the commit plan to the user.

        Args:
            plan: AICommitPlan to display
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]AI Commit Plan[/bold]\n\n"
                f"Total files: {plan.total_files}\n"
                f"Proposed commits: {len(plan.groups)}\n"
                f"Guidelines: {plan.guidelines_used}",
                title="Commit Analysis",
                border_style="cyan",
            )
        )

        for i, group in enumerate(plan.groups, 1):
            table = Table(title=f"Commit {i}: {group.title}", show_header=True)
            table.add_column("File", style="cyan")
            table.add_column("Status", style="yellow")

            for file_path in group.files[:10]:
                table.add_row(file_path, "staged")

            if len(group.files) > 10:
                table.add_row(f"... and {len(group.files) - 10} more", "")

            self.console.print(table)
            self.console.print(f"[dim]Reasoning: {group.reasoning}[/dim]")
            self.console.print()

    def _execute_commit_group(self, group: CommitGroup) -> bool:
        """Execute a single commit group with human approval.

        Args:
            group: CommitGroup to commit

        Returns:
            True if committed successfully
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{group.title}[/bold]\n\n"
                f"Files ({len(group.files)}):\n"
                + "\n".join(f"  - {f}" for f in group.files[:10])
                + (f"\n  ... and {len(group.files) - 10} more" if len(group.files) > 10 else "")
                + f"\n\n[bold]Commit Message:[/bold]\n{group.commit_message}",
                title="Proposed Commit",
                border_style="green",
            )
        )

        # Human-in-the-loop: Get approval
        action = Prompt.ask(
            "\n[bold]Action[/bold]",
            choices=["approve", "edit", "skip", "abort"],
            default="approve",
        )

        if action == "abort":
            self.console.print("[red]Aborting commit process[/red]")
            return False

        if action == "skip":
            self.console.print("[yellow]Skipping this commit[/yellow]")
            return True

        if action == "edit":
            # Allow editing the commit message
            new_message = Prompt.ask(
                "Enter new commit message (or press Enter to keep)",
                default=group.commit_message.split("\n")[0],
            )
            if new_message:
                group.commit_message = new_message

        # Stage the files
        for file_path in group.files:
            success, _ = self._run_git("add", file_path)
            if not success:
                self.console.print(f"[red]Failed to stage {file_path}[/red]")

        # Create the commit
        success, output = self._run_git("commit", "-m", group.commit_message)

        if success:
            self.console.print(f"[green]Committed: {group.title}[/green]")
        else:
            self.console.print(f"[red]Commit failed: {output}[/red]")

        return success

    def run_ai_commit(self, args: str = "") -> bool:
        """Execute AI-powered commit workflow.

        Args:
            args: Optional arguments (e.g., specific files)

        Returns:
            True if successful
        """
        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]AI-Powered Commit Assistant[/bold cyan]\n\n"
                "I'll analyze timestamps, folder structure, and diff context "
                "to propose logical commit groups.\n\n"
                "You stay in the loop: confirm the plan, provide feedback, "
                "and approve each commit.",
                title="git ai-commit",
                border_style="cyan",
            )
        )

        # Step 1: Get current changes
        self.console.print("\n[bold]Step 1:[/bold] Analyzing changes...")
        changes = self._get_git_status()

        if not changes:
            self.console.print("[yellow]No changes to commit[/yellow]")
            return True

        self.console.print(f"  Found {len(changes)} changed files")

        # Step 2: Enrich changes with diff and timestamps
        self.console.print("\n[bold]Step 2:[/bold] Gathering context...")
        for change in changes:
            change.diff = self._get_file_diff(change.path)
            change.timestamp = self._get_file_timestamp(change.path)

        # Step 3: Load guidelines
        self.console.print("\n[bold]Step 3:[/bold] Loading commit guidelines...")
        guidelines = self._load_git_guidelines()

        # Step 4: Analyze changes
        self.console.print("\n[bold]Step 4:[/bold] Analyzing change patterns...")
        analysis = self._analyze_changes(changes)

        # Step 5: Generate commit plan
        self.console.print("\n[bold]Step 5:[/bold] Generating commit plan with AI...")
        plan = self._generate_commit_plan(changes, guidelines, analysis)

        # Step 6: Display plan and get overall approval
        self._display_commit_plan(plan)

        if not Confirm.ask("\n[bold]Proceed with this commit plan?[/bold]", default=True):
            self.console.print("[yellow]Commit plan cancelled[/yellow]")
            return False

        # Step 7: Execute commits with human-in-the-loop
        self.console.print("\n[bold]Step 6:[/bold] Executing commits...")
        committed = 0
        for group in plan.groups:
            if self._execute_commit_group(group):
                committed += 1
            else:
                break  # Abort on failure

        self.console.print(
            f"\n[bold green]Completed: {committed}/{len(plan.groups)} commits[/bold green]"
        )
        return committed > 0

    # Basic git operations (for compatibility with main.py)
    def run_add(self, args: str) -> bool:
        """Run git add."""
        success, output = self._run_git("add", *args.split())
        if success:
            self.console.print(f"[green]Added: {args}[/green]")
        else:
            self.console.print(f"[red]Failed: {output}[/red]")
        return success

    def run_commit(self, args: str) -> bool:
        """Run git commit."""
        if "-m" in args:
            parts = args.split("-m", 1)
            message = parts[1].strip().strip('"').strip("'")
            success, output = self._run_git("commit", "-m", message)
        else:
            success, output = self._run_git("commit", *args.split())

        if success:
            self.console.print("[green]Committed successfully[/green]")
        else:
            self.console.print(f"[red]Commit failed: {output}[/red]")
        return success

    def run_push(self, args: str = "") -> bool:
        """Run git push."""
        cmd_args = args.split() if args else []
        success, output = self._run_git("push", *cmd_args)
        if success:
            self.console.print("[green]Pushed successfully[/green]")
        else:
            self.console.print(f"[red]Push failed: {output}[/red]")
        return success

    def run_pull(self, args: str = "") -> bool:
        """Run git pull."""
        cmd_args = args.split() if args else []
        success, output = self._run_git("pull", *cmd_args)
        if success:
            self.console.print(f"[green]Pulled successfully[/green]\n{output}")
        else:
            self.console.print(f"[red]Pull failed: {output}[/red]")
        return success

    def run_status(self) -> bool:
        """Run git status."""
        success, output = self._run_git("status")
        self.console.print(output)
        return success

    def run_log(self, args: str = "") -> bool:
        """Run git log."""
        cmd_args = args.split() if args else ["--oneline", "-10"]
        success, output = self._run_git("log", *cmd_args)
        self.console.print(output)
        return success

    def run_diff(self, args: str = "") -> bool:
        """Run git diff."""
        cmd_args = args.split() if args else []
        success, output = self._run_git("diff", *cmd_args)
        self.console.print(output)
        return success

    def run_branch(self, args: str = "") -> bool:
        """Run git branch."""
        cmd_args = args.split() if args else []
        success, output = self._run_git("branch", *cmd_args)
        self.console.print(output)
        return success

    def run_switch(self, args: str) -> bool:
        """Run git switch."""
        success, output = self._run_git("switch", *args.split())
        if success:
            self.console.print(f"[green]Switched to: {args}[/green]")
        else:
            self.console.print(f"[red]Switch failed: {output}[/red]")
        return success
