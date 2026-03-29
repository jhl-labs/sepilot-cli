"""Karpathy Guidelines Skill - Behavioral guidelines to reduce common LLM coding mistakes"""

from ..base import PromptSkill

# Shared constant so both the skill and the orchestrator can reference it.
KARPATHY_GUIDELINES_PROMPT = """\
## Coding Guidelines

### 1. Understand the Root Cause First
- Read the relevant code thoroughly before writing any fix.
- Identify the exact root cause — don't guess from symptoms.
- Trace the data flow to confirm where the bug originates.
- Reproduce the issue first to confirm your understanding.

### 2. Fix the Root Cause, Not the Symptom
- Change the code that is actually wrong, not a workaround elsewhere.
- Preserve all necessary logic — don't simplify away required edge cases.
- If a fix needs multiple lines or branches, write them. Correctness > brevity.
- Don't "clean up" or refactor adjacent code — focus on the actual bug.

### 3. Surgical Changes
- Only modify lines directly related to the fix.
- Match existing code style exactly.
- Don't rename, reformat, or restructure unrelated code.
- You MUST produce a concrete code fix. Never stop at just reading code.

### 4. Verify Before Finishing
- Confirm the fix handles the reported scenario and edge cases.
- Run existing tests if available to check for regressions.
- If your first attempt is wrong, iterate — don't give up."""

# Task types (keyword-based TaskAnalyzer) that should receive the guidelines
KARPATHY_CODE_TASK_TYPES = frozenset({
    "bug_fix", "new_feature", "refactor", "testing",
})

# Strategies (LLM-based triage) that should receive the guidelines
# Maps to AgentStrategy enum values from enhanced_state.py
KARPATHY_CODE_STRATEGIES = frozenset({
    "implement", "debug", "refactor", "test",
})


class KarpathyGuidelinesSkill(PromptSkill):
    name = "karpathy-guidelines"
    description = (
        "Behavioral guidelines to reduce common LLM coding mistakes. "
        "Use when writing, reviewing, or refactoring code to avoid overcomplication, "
        "make surgical changes, surface assumptions, and define verifiable success criteria."
    )
    triggers = [
        "karpathy", "karpathy guidelines", "simplicity first",
        "surgical changes", "think before coding", "goal-driven",
    ]
    category = "development"
    author = "Andrej Karpathy / forrestchang"
    prompt = KARPATHY_GUIDELINES_PROMPT
