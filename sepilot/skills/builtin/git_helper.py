"""Git Helper Skill"""

from ..base import PromptSkill


class GitHelperSkill(PromptSkill):
    name = "git-helper"
    description = "Assist with git operations, commit messages, and branch management"
    triggers = ["git commit", "commit message", "git help", "merge conflict"]
    category = "version-control"
    prompt = """\
## Git Helper Guidelines

**IMPORTANT: For destructive operations (force push, reset --hard, rebase), always explain the impact and ask for confirmation first.**

When assisting with git operations:

### Commit Messages
- Use conventional commits format: type(scope): description
- Types: feat, fix, docs, style, refactor, test, chore
- Keep the subject line under 50 characters
- Use imperative mood ("add" not "added")

### Branch Naming
- Use descriptive names: feature/user-auth, fix/login-bug
- Prefix with type: feature/, fix/, hotfix/, release/

### Merge Conflicts
- Identify the conflicting changes clearly
- Suggest resolution strategy
- Preserve both changes when appropriate

### Best Practices
- Commit early and often
- Keep commits atomic and focused
- Write meaningful commit messages
- Review changes before committing

Always check `git status` before operations and provide safe commands.

**REMINDER: Explain what each git command will do before executing. Never run destructive commands without user confirmation.**"""
