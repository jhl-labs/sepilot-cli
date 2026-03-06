"""Explain Code Skill - Code explanation and documentation"""

from ..base import BaseSkill, SkillMetadata, SkillResult


class ExplainCodeSkill(BaseSkill):
    """Skill for explaining code"""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="explain-code",
            description="Explain code with detailed breakdown and documentation",
            version="1.0.0",
            author="SEPilot",
            triggers=["explain code", "what does this do", "explain this", "how does this work"],
            category="documentation"
        )

    def execute(self, input_text: str, context: dict) -> SkillResult:
        """Execute explain code skill"""
        explain_prompt = """## Code Explanation Guidelines

**IMPORTANT: This is a READ-ONLY task. Do NOT modify, edit, or fix any code. Only explain it.**

When explaining the code, provide:

### 1. Overview
- What is the purpose of this code?
- What problem does it solve?

### 2. Step-by-Step Breakdown
- Walk through the code line by line or block by block
- Explain the logic and flow
- Highlight important decisions

### 3. Key Concepts
- What programming concepts are used?
- Are there patterns or idioms being used?

### 4. Dependencies
- What external libraries or modules are used?
- What are the inputs and outputs?

### 5. Usage Example
- How would this code be used in practice?
- Provide a simple example if helpful

Use clear, simple language suitable for someone learning this codebase.

**REMINDER: Do NOT use file_edit, file_write, or any modification tools. Only read and explain.**
"""
        return SkillResult(
            success=True,
            message="Explain code skill activated",
            prompt_injection=explain_prompt
        )
