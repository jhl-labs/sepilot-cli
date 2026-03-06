"""Interactive tools for LangChain agent.

Provides ask_user, slash_command, skill tools.
"""

from typing import Any

from langchain_core.tools import tool


@tool
def ask_user(questions: list[dict[str, Any]], context: str | None = None) -> str:
    """Ask the user questions to clarify requirements or get decisions.

    Args:
        questions: List of questions to ask (required)
            Each question is a dict with:
            - question: The question text (required)
            - header: Short label for the question (optional)
            - options: List of choice options (optional)
            - multiSelect: Allow multiple selections (optional)
        context: Context about why these questions are being asked (optional)

    Returns:
        User's responses

    Examples:
        # Simple text question
        ask_user(questions=[{"question": "What's the project name?", "header": "Project"}])

        # Multiple choice
        ask_user(questions=[{
            "question": "Which framework to use?",
            "header": "Framework",
            "options": [
                {"label": "FastAPI", "description": "Modern, fast web framework"},
                {"label": "Flask", "description": "Lightweight and simple"},
                {"label": "Django", "description": "Full-featured framework"}
            ]
        }])

        # Multiple questions with context
        ask_user(
            context="Setting up new project configuration",
            questions=[
                {"question": "Project name?", "header": "Name"},
                {"question": "Python version?", "header": "Version",
                 "options": [
                     {"label": "3.11", "description": "Latest stable"},
                     {"label": "3.10", "description": "LTS version"}
                 ]}
            ]
        )
    """
    from sepilot.tools.interactive.ask_user_tool import AskUserQuestionTool
    tool_instance = AskUserQuestionTool()
    return tool_instance.execute(questions, context)


@tool
def slash_command(command: str) -> str:
    """Execute a custom slash command.

    Args:
        command: The slash command to execute (e.g., '/review-pr 123') (required)

    Returns:
        Command output or error message

    Examples:
        # Execute a simple command
        slash_command(command="/hello")

        # Command with arguments
        slash_command(command="/review-pr 123")

        # Command with named arguments
        slash_command(command="/deploy env=staging version=1.2.3")
    """
    from sepilot.tools.command.slash_command_tool import SlashCommandTool
    tool_instance = SlashCommandTool()
    return tool_instance.execute(command)


@tool
def skill(skill: str) -> str:
    """Execute a specialized skill.

    Available built-in skills:
    - pdf: Process PDF files (extract text, images, metadata)
    - xlsx: Process Excel files (read data, convert formats)
    - database: Database operations (queries, migrations)
    - docker: Docker container management

    Args:
        skill: The skill name to execute (required)

    Returns:
        Skill prompt and instructions

    Examples:
        # Load PDF processing skill
        skill(skill="pdf")

        # Load Excel processing skill
        skill(skill="xlsx")

        # Load database skill
        skill(skill="database")
    """
    from sepilot.tools.command.slash_command_tool import SkillTool
    tool_instance = SkillTool()
    return tool_instance.execute(skill)


__all__ = ['ask_user', 'slash_command', 'skill']
