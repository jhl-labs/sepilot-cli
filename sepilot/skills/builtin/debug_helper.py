"""Debug Helper Skill - Debugging assistance"""

from ..base import BaseSkill, SkillMetadata, SkillResult


class DebugHelperSkill(BaseSkill):
    """Skill for debugging assistance"""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="debug-helper",
            description="Assist with debugging, error analysis, and troubleshooting",
            version="1.0.0",
            author="SEPilot",
            triggers=["debug", "error", "bug", "fix bug", "troubleshoot", "not working"],
            category="debugging"
        )

    def execute(self, input_text: str, context: dict) -> SkillResult:
        """Execute debug helper skill"""
        debug_prompt = """## Debugging Guidelines

**IMPORTANT: First analyze and explain the issue. Only apply fixes if the user explicitly asks you to fix it.**

When debugging, follow this systematic approach:

### 1. Understand the Problem
- What is the expected behavior?
- What is the actual behavior?
- When did the problem start?
- Can it be reproduced consistently?

### 2. Gather Information
- Check error messages and stack traces
- Review recent changes
- Check logs and outputs
- Verify inputs and configuration

### 3. Isolate the Issue
- Narrow down the problematic code
- Use binary search on recent changes if needed
- Test with minimal reproduction

### 4. Form and Test Hypotheses
- What could cause this behavior?
- Test each hypothesis systematically
- Use print/log statements or debugger

### 5. Report Your Findings
- Explain the root cause clearly
- Propose a fix with explanation
- **Ask the user if they want you to apply the fix before making any changes**

Be systematic and methodical. Document findings as you debug.

**REMINDER: Analyze first, then ask before modifying code.**
"""
        return SkillResult(
            success=True,
            message="Debug helper skill activated",
            prompt_injection=debug_prompt
        )
