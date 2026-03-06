"""Interactive user question tool"""

import json
from pathlib import Path
from typing import Any

from sepilot.tools.base_tool import BaseTool


class AskUserQuestionTool(BaseTool):
    """Tool for asking users questions during execution"""

    name = "ask_user"
    description = "Ask the user questions to clarify requirements or get decisions"
    parameters = {
        "questions": "List of questions to ask (required)",
        "context": "Context about why these questions are being asked"
    }

    def execute(
        self,
        questions: list[dict[str, Any]],
        context: str | None = None
    ) -> str:
        """Ask user questions interactively"""
        self.validate_params(questions=questions)

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.prompt import Prompt

            console = Console()

            # Display context if provided
            if context:
                console.print(Panel(
                    context,
                    title="[bold cyan]Context[/bold cyan]",
                    border_style="cyan"
                ))
                console.print()

            # Collect answers
            answers = {}

            for i, question_data in enumerate(questions, 1):
                # Validate question structure
                if not isinstance(question_data, dict):
                    return f"Error: Question {i} must be a dictionary"

                question = question_data.get('question', '')
                header = question_data.get('header', f'Q{i}')
                options = question_data.get('options', [])
                multi_select = question_data.get('multiSelect', False)

                if not question:
                    return f"Error: Question {i} missing 'question' field"

                # Display question header
                console.print(f"\n[bold yellow]📝 {header}[/bold yellow]")
                console.print(f"[white]{question}[/white]\n")

                # Handle different question types
                if options:
                    # Multiple choice question
                    answer = self._ask_choice_question(
                        console, options, multi_select, header
                    )
                    answers[header] = answer
                else:
                    # Open text question
                    answer = Prompt.ask(
                        "[cyan]Your answer[/cyan]",
                        default="",
                        show_default=False
                    )
                    answers[header] = answer

            # Save answers for reference
            self._save_answers(answers, questions)

            # Format response
            return self._format_response(answers)

        except ImportError:
            # Fallback to simple input if rich not available
            return self._fallback_ask(questions, context)

        except KeyboardInterrupt:
            return "User cancelled question prompt"

        except Exception as e:
            return f"Error asking questions: {str(e)}"

    def _ask_choice_question(
        self,
        console,
        options: list[dict[str, str]],
        multi_select: bool,
        header: str
    ) -> Any:
        """Ask a multiple choice question"""
        from rich.prompt import Prompt
        from rich.table import Table

        # Create options table
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description")

        choice_map = {}
        for i, option in enumerate(options, 1):
            label = option.get('label', f'Option {i}')
            description = option.get('description', '')
            choice_map[str(i)] = label
            table.add_row(f"[{i}]", f"{label} - {description}")

        # Add "Other" option
        choice_map[str(len(options) + 1)] = "Other"
        table.add_row(f"[{len(options) + 1}]", "Other (custom input)")

        console.print(table)

        if multi_select:
            # Multiple selection
            console.print("\n[dim]Select multiple options (comma-separated) or 'Other':[/dim]")
            choices_str = Prompt.ask("[cyan]Your choices[/cyan]")

            selected = []
            for choice in choices_str.split(','):
                choice = choice.strip()
                if choice in choice_map:
                    if choice_map[choice] == "Other":
                        custom = Prompt.ask("[cyan]Enter custom value[/cyan]")
                        selected.append(custom)
                    else:
                        selected.append(choice_map[choice])

            return selected if selected else ["None selected"]

        else:
            # Single selection
            while True:
                choice = Prompt.ask(
                    "[cyan]Your choice[/cyan]",
                    choices=[str(i) for i in range(1, len(options) + 2)]
                )

                if choice_map.get(choice) == "Other":
                    return Prompt.ask("[cyan]Enter custom value[/cyan]")
                else:
                    return choice_map.get(choice, "Unknown")

    def _fallback_ask(
        self,
        questions: list[dict[str, Any]],
        context: str | None = None
    ) -> str:
        """Fallback question asking without rich library"""
        if context:
            print(f"\nContext: {context}\n")

        answers = {}

        for i, question_data in enumerate(questions, 1):
            question = question_data.get('question', '')
            header = question_data.get('header', f'Q{i}')
            options = question_data.get('options', [])

            print(f"\n{header}: {question}")

            if options:
                for j, option in enumerate(options, 1):
                    label = option.get('label', '')
                    desc = option.get('description', '')
                    print(f"  {j}. {label} - {desc}")
                print(f"  {len(options) + 1}. Other (custom input)")

                while True:
                    try:
                        choice = int(input("Your choice (number): "))
                        if 1 <= choice <= len(options):
                            answers[header] = options[choice - 1]['label']
                            break
                        elif choice == len(options) + 1:
                            answers[header] = input("Enter custom value: ")
                            break
                    except ValueError:
                        print("Please enter a valid number")
            else:
                answers[header] = input("Your answer: ")

        return self._format_response(answers)

    def _save_answers(self, answers: dict[str, Any], questions: list[dict[str, Any]]):
        """Save answers to file for reference"""
        try:
            answers_dir = Path.home() / ".sepilot" / "user_answers"
            answers_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = answers_dir / f"answers_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump({
                    'questions': questions,
                    'answers': answers,
                    'timestamp': timestamp
                }, f, indent=2)

        except Exception:
            pass  # Ignore save errors

    def _format_response(self, answers: dict[str, Any]) -> str:
        """Format answers for return"""
        result = ["✅ User responses collected:\n"]

        for header, answer in answers.items():
            if isinstance(answer, list):
                result.append(f"**{header}**: {', '.join(answer)}")
            else:
                result.append(f"**{header}**: {answer}")

        return '\n'.join(result)
