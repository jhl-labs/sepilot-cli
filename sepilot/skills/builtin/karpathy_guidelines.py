"""Karpathy Guidelines Skill - Behavioral guidelines to reduce common LLM coding mistakes"""

from ..base import BaseSkill, SkillMetadata, SkillResult

# Shared constant so both the skill and the orchestrator can reference it.
KARPATHY_GUIDELINES_PROMPT = """\
## Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, \
derived from Andrej Karpathy's observations on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
- The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"
- For multi-step tasks, state a brief plan with verification for each step.

Strong success criteria let you loop independently. \
Weak criteria ("make it work") require constant clarification."""

# Task types (keyword-based TaskAnalyzer) that should receive the guidelines
KARPATHY_CODE_TASK_TYPES = frozenset({
    "bug_fix", "new_feature", "refactor", "testing",
})

# Strategies (LLM-based triage) that should receive the guidelines
# Maps to AgentStrategy enum values from enhanced_state.py
KARPATHY_CODE_STRATEGIES = frozenset({
    "implement", "debug", "refactor", "test",
})


class KarpathyGuidelinesSkill(BaseSkill):
    """Skill for applying Andrej Karpathy's coding guidelines to reduce common LLM mistakes"""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="karpathy-guidelines",
            description="Behavioral guidelines to reduce common LLM coding mistakes. "
            "Use when writing, reviewing, or refactoring code to avoid overcomplication, "
            "make surgical changes, surface assumptions, and define verifiable success criteria.",
            version="1.0.0",
            author="Andrej Karpathy / forrestchang",
            triggers=[
                "karpathy",
                "karpathy guidelines",
                "simplicity first",
                "surgical changes",
                "think before coding",
                "goal-driven",
            ],
            category="development",
        )

    def execute(self, input_text: str, context: dict) -> SkillResult:
        """Execute Karpathy guidelines skill"""
        return SkillResult(
            success=True,
            message="Karpathy guidelines activated",
            prompt_injection=KARPATHY_GUIDELINES_PROMPT,
        )
