"""Command Manager for SEPilot Custom Commands System

Custom commands are markdown files that define prompt templates.
When invoked, the markdown content is expanded and used as the prompt.

Directory structure:
- ~/.sepilot/commands/*.md     (user global commands)
- .sepilot/commands/*.md       (project-specific commands)

File format:
- Filename becomes the command name (e.g., review-pr.md -> /review-pr)
- First line starting with # is used as description
- Content can include variables: $ARGUMENTS, $FILE, $SELECTION
- Variables are replaced when the command is executed
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CustomCommand:
    """A custom user-defined command"""
    name: str
    description: str
    content: str
    source_path: Path
    is_project: bool  # True if from project .sepilot/commands/

    def expand(self, arguments: str = "", file_content: str = "", selection: str = "") -> str:
        """Expand the command template with variables

        Variables:
            $ARGUMENTS - Command line arguments passed to the command
            $FILE      - Content of the current file (if any)
            $SELECTION - Current selection (if any)
            $PROMPT    - User's follow-up prompt
        """
        expanded = self.content

        # Replace variables
        expanded = expanded.replace("$ARGUMENTS", arguments)
        expanded = expanded.replace("$FILE", file_content)
        expanded = expanded.replace("$SELECTION", selection)

        # Remove unreplaced optional variables
        expanded = re.sub(r'\$\w+', '', expanded)

        return expanded.strip()


class CommandManager:
    """Manages custom command discovery and execution

    Commands are loaded from:
    1. User commands: ~/.sepilot/commands/*.md
    2. Project commands: .sepilot/commands/*.md

    Project commands override user commands with the same name.
    """

    def __init__(self):
        self._commands: dict[str, CustomCommand] = {}
        self._loaded = False

    def _get_command_directories(self) -> list[tuple[Path, bool]]:
        """Get directories to search for commands

        Returns:
            List of (path, is_project) tuples
        """
        dirs = []

        # User commands (~/.sepilot/commands/)
        user_dir = Path.home() / ".sepilot" / "commands"
        if user_dir.exists():
            dirs.append((user_dir, False))

        # Project commands (.sepilot/commands/)
        project_dir = Path.cwd() / ".sepilot" / "commands"
        if project_dir.exists():
            dirs.append((project_dir, True))

        return dirs

    def _parse_command_file(self, filepath: Path, is_project: bool) -> CustomCommand | None:
        """Parse a markdown command file"""
        try:
            content = filepath.read_text(encoding='utf-8')

            # Extract description from first heading
            description = ""
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    description = line[2:].strip()
                    break

            # If no heading, use first line as description
            if not description and lines:
                description = lines[0].strip()[:80]

            # Command name from filename (without .md)
            name = filepath.stem

            return CustomCommand(
                name=name,
                description=description,
                content=content,
                source_path=filepath,
                is_project=is_project
            )

        except Exception as e:
            logger.error(f"Failed to parse command file {filepath}: {e}")
            return None

    def discover_commands(self) -> None:
        """Discover and load all custom commands"""
        if self._loaded:
            return

        self._commands.clear()

        for cmd_dir, is_project in self._get_command_directories():
            if not cmd_dir.exists():
                continue

            for filepath in cmd_dir.glob("*.md"):
                if filepath.name.startswith("_"):
                    continue

                command = self._parse_command_file(filepath, is_project)
                if command:
                    # Project commands override user commands
                    if command.name in self._commands and not is_project:
                        continue
                    self._commands[command.name] = command
                    logger.debug(f"Loaded command: /{command.name}")

        self._loaded = True
        logger.info(f"Discovered {len(self._commands)} custom commands")

    def get_command(self, name: str) -> CustomCommand | None:
        """Get a command by name"""
        self.discover_commands()
        return self._commands.get(name)

    def list_commands(self) -> list[CustomCommand]:
        """List all available commands"""
        self.discover_commands()
        return list(self._commands.values())

    def execute_command(
        self,
        name: str,
        arguments: str = "",
        file_content: str = "",
        selection: str = ""
    ) -> str | None:
        """Execute a command and return the expanded prompt

        Args:
            name: Command name (without /)
            arguments: Arguments passed to the command
            file_content: Current file content (if applicable)
            selection: Current selection (if applicable)

        Returns:
            Expanded prompt string or None if command not found
        """
        command = self.get_command(name)
        if not command:
            return None

        return command.expand(arguments, file_content, selection)

    def reload_commands(self) -> None:
        """Force reload all commands"""
        self._loaded = False
        self._commands.clear()
        self.discover_commands()

    def create_command(self, name: str, content: str, project: bool = False) -> Path:
        """Create a new command file

        Args:
            name: Command name (will be used as filename.md)
            content: Command content (markdown)
            project: If True, create in project directory, else user directory

        Returns:
            Path to created file
        """
        if project:
            cmd_dir = Path.cwd() / ".sepilot" / "commands"
        else:
            cmd_dir = Path.home() / ".sepilot" / "commands"

        cmd_dir.mkdir(parents=True, exist_ok=True)

        filepath = cmd_dir / f"{name}.md"
        filepath.write_text(content, encoding='utf-8')

        # Reload to pick up new command
        self.reload_commands()

        return filepath

    def delete_command(self, name: str) -> bool:
        """Delete a command file

        Returns:
            True if deleted, False if not found
        """
        command = self.get_command(name)
        if not command:
            return False

        try:
            command.source_path.unlink()
            self.reload_commands()
            return True
        except Exception as e:
            logger.error(f"Failed to delete command {name}: {e}")
            return False


# Global singleton instance
_command_manager: CommandManager | None = None


def get_command_manager() -> CommandManager:
    """Get or create the global command manager instance"""
    global _command_manager
    if _command_manager is None:
        _command_manager = CommandManager()
    return _command_manager
