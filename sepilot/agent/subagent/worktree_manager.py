"""Worktree Manager - Git worktree isolation for parallel agents.

Provides isolated working directories for subagents using
git worktree, preventing file conflicts during parallel execution.

Usage:
    manager = WorktreeManager()
    worktree = await manager.create_worktree("task-123")
    # ... run subagent in worktree.path ...
    await manager.cleanup_worktree(worktree)
"""

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Worktree:
    """Represents an active git worktree."""
    worktree_id: str
    path: Path
    branch: str
    base_commit: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    task_id: str = ""
    has_changes: bool = False

    def to_dict(self) -> dict:
        return {
            "worktree_id": self.worktree_id,
            "path": str(self.path),
            "branch": self.branch,
            "base_commit": self.base_commit,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "has_changes": self.has_changes,
        }


class WorktreeManager:
    """Manages git worktrees for isolated subagent execution.

    Creates temporary git worktrees so that parallel subagents
    can work on the repository without file conflicts.

    Features:
    - Automatic branch creation per worktree
    - Change detection on cleanup
    - Automatic cleanup of unchanged worktrees
    - Worktree directory management
    """

    WORKTREE_BASE_DIR = ".claude/worktrees"

    def __init__(self, repo_root: Path | None = None):
        """Initialize WorktreeManager.

        Args:
            repo_root: Git repository root. Auto-detected if None.
        """
        self.repo_root = repo_root or self._find_repo_root()
        self.worktrees: dict[str, Worktree] = {}
        self._lock = asyncio.Lock()

    def _find_repo_root(self) -> Path:
        """Find git repository root."""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
        return Path.cwd()

    async def _run_git(self, *args: str, cwd: Path | None = None) -> tuple[int, str, str]:
        """Run a git command asynchronously."""
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd or self.repo_root)
        )
        stdout, stderr = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout.decode().strip(),
            stderr.decode().strip()
        )

    async def create_worktree(
        self,
        task_id: str = "",
        base_branch: str = "HEAD"
    ) -> Worktree:
        """Create a new git worktree for isolated work.

        Args:
            task_id: Identifier for the task (used in branch name)
            base_branch: Base branch/commit to branch from

        Returns:
            Worktree object with path and branch info
        """
        async with self._lock:
            worktree_id = uuid.uuid4().hex[:8]
            branch_name = f"worktree-agent-{worktree_id}"

            returncode, base_commit, stderr = await self._run_git(
                "rev-parse", "--verify", f"{base_branch}^{{commit}}"
            )
            if returncode != 0:
                raise RuntimeError(f"Failed to resolve base branch: {stderr}")
            resolved_base_commit = base_commit.splitlines()[-1].strip()

            # Create worktree directory
            worktree_base = self.repo_root / self.WORKTREE_BASE_DIR
            worktree_base.mkdir(parents=True, exist_ok=True)
            worktree_path = worktree_base / worktree_id

            # Create git worktree with new branch
            returncode, stdout, stderr = await self._run_git(
                "worktree", "add", "-b", branch_name,
                str(worktree_path), resolved_base_commit
            )

            if returncode != 0:
                raise RuntimeError(
                    f"Failed to create worktree: {stderr}"
                )

            worktree = Worktree(
                worktree_id=worktree_id,
                path=worktree_path,
                branch=branch_name,
                base_commit=resolved_base_commit,
                task_id=task_id,
            )

            self.worktrees[worktree_id] = worktree
            logger.info(
                f"Created worktree {worktree_id} at {worktree_path} "
                f"(branch: {branch_name})"
            )

            return worktree

    async def has_changes(self, worktree: Worktree) -> bool:
        """Check if a worktree has uncommitted or new changes."""
        returncode, stdout, _ = await self._run_git(
            "status", "--porcelain", cwd=worktree.path
        )

        if stdout.strip():
            worktree.has_changes = True
            return True

        if worktree.base_commit:
            returncode, stdout, _ = await self._run_git(
                "rev-list", "--count", f"{worktree.base_commit}..HEAD",
                cwd=worktree.path
            )
            if returncode == 0:
                has = int(stdout.strip() or "0") > 0
                worktree.has_changes = has
                return has

        has = False
        worktree.has_changes = has
        return has

    async def cleanup_worktree(
        self,
        worktree: Worktree,
        force: bool = False
    ) -> dict:
        """Clean up a worktree.

        If worktree has changes, keeps it and returns info.
        If no changes (or force=True), removes worktree and branch.

        Args:
            worktree: Worktree to clean up
            force: Force cleanup even with changes

        Returns:
            Dict with cleanup result info
        """
        async with self._lock:
            result = {
                "worktree_id": worktree.worktree_id,
                "had_changes": False,
                "cleaned_up": False,
                "branch": worktree.branch,
                "path": str(worktree.path),
            }

            # Check for changes
            has_changes = await self.has_changes(worktree)
            result["had_changes"] = has_changes

            if has_changes and not force:
                logger.info(
                    f"Worktree {worktree.worktree_id} has changes, keeping "
                    f"(branch: {worktree.branch}, path: {worktree.path})"
                )
                return result

            # Remove worktree
            remove_args = ["worktree", "remove", str(worktree.path)]
            if force:
                remove_args.append("--force")
            returncode, _, stderr = await self._run_git(*remove_args)

            if returncode != 0:
                # Fallback: manual cleanup
                if worktree.path.exists():
                    shutil.rmtree(worktree.path, ignore_errors=True)
                await self._run_git("worktree", "prune")

            # Delete branch if no changes
            if not has_changes or force:
                await self._run_git(
                    "branch", "-D", worktree.branch
                )

            # Remove from tracking
            self.worktrees.pop(worktree.worktree_id, None)
            result["cleaned_up"] = True

            logger.info(f"Cleaned up worktree {worktree.worktree_id}")
            return result

    async def cleanup_all(self, force: bool = False) -> list[dict]:
        """Clean up all tracked worktrees."""
        results = []
        for worktree in list(self.worktrees.values()):
            result = await self.cleanup_worktree(worktree, force=force)
            results.append(result)
        return results

    async def list_worktrees(self) -> list[dict]:
        """List all git worktrees (including untracked ones)."""
        returncode, stdout, _ = await self._run_git("worktree", "list", "--porcelain")

        worktrees = []
        current: dict = {}

        for line in stdout.split("\n"):
            if line.startswith("worktree "):
                if current:
                    worktrees.append(current)
                current = {"path": line[9:]}
            elif line.startswith("HEAD "):
                current["head"] = line[5:]
            elif line.startswith("branch "):
                current["branch"] = line[7:]
            elif line == "bare":
                current["bare"] = True
            elif line == "detached":
                current["detached"] = True

        if current:
            worktrees.append(current)

        return worktrees

    async def merge_worktree(
        self,
        worktree: Worktree,
        target_branch: str = "main",
        auto_cleanup: bool = True
    ) -> dict:
        """Merge worktree branch back into target branch.

        Args:
            worktree: Worktree to merge
            target_branch: Branch to merge into
            auto_cleanup: Clean up worktree after merge

        Returns:
            Dict with merge result
        """
        result = {
            "worktree_id": worktree.worktree_id,
            "branch": worktree.branch,
            "target": target_branch,
            "success": False,
            "message": "",
        }

        # 현재 위치 저장 (merge 후 복원). detached HEAD면 commit으로 복원해야 한다.
        returncode, original_branch, _ = await self._run_git(
            "symbolic-ref", "-q", "--short", "HEAD"
        )
        original_ref = original_branch.strip() if returncode == 0 else None
        restore_detached = False
        if original_ref is None:
            returncode, original_commit, _ = await self._run_git(
                "rev-parse", "--verify", "HEAD"
            )
            if returncode == 0:
                original_ref = original_commit.splitlines()[-1].strip()
                restore_detached = True

        # checkout 전 target branch를 commit-ish로 검증해 옵션 주입을 막는다.
        returncode, _, stderr = await self._run_git(
            "rev-parse", "--verify", f"{target_branch}^{{commit}}"
        )
        if returncode != 0:
            result["message"] = f"Failed to resolve {target_branch}: {stderr}"
            return result

        # Checkout target branch before merging
        returncode, _, stderr = await self._run_git(
            "checkout", target_branch
        )
        if returncode != 0:
            result["message"] = f"Failed to checkout {target_branch}: {stderr}"
            return result

        # Merge branch
        returncode, stdout, stderr = await self._run_git(
            "merge", worktree.branch, "--no-edit",
        )

        merge_success = returncode == 0

        # 원래 위치로 복원 (merge 성공/실패 무관)
        if original_ref and (restore_detached or original_ref != target_branch):
            if restore_detached:
                await self._run_git("checkout", "--detach", original_ref)
            else:
                await self._run_git("checkout", original_ref)

        if not merge_success:
            result["message"] = f"Merge failed: {stderr}"
            return result

        result["success"] = True
        result["message"] = stdout or "Merge successful"

        # Auto cleanup
        if auto_cleanup:
            await self.cleanup_worktree(worktree, force=True)

        return result
