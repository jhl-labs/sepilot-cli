"""Code Review Skill"""

from ..base import PromptSkill


class CodeReviewSkill(PromptSkill):
    name = "code-review"
    description = "Perform detailed code review with best practices analysis"
    triggers = ["review code", "code review", "review this", "pr review"]
    category = "development"
    prompt = """\
## Code Review Guidelines

**IMPORTANT: This is a READ-ONLY review task. Do NOT modify or edit any code unless explicitly requested by the user. Only analyze and provide feedback.**

When reviewing the code, analyze the following aspects:

### 1. Code Quality
- Is the code readable and well-organized?
- Are variable/function names meaningful?
- Is there unnecessary complexity?

### 2. Best Practices
- Does it follow language-specific conventions?
- Are there any anti-patterns?
- Is error handling appropriate?

### 3. Security
- Are there potential security vulnerabilities?
- Is input validation proper?
- Are sensitive data handled correctly?

### 4. Performance
- Are there obvious performance issues?
- Is resource management appropriate?
- Are there unnecessary operations?

### 5. Testing
- Is the code testable?
- Are edge cases considered?

Provide specific, actionable feedback with code examples where helpful.

**REMINDER: Only provide review feedback. Do NOT use file_edit or file_write unless the user explicitly asks you to fix the issues.**"""
