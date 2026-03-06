"""Slash command and skill execution tools"""

import json
import subprocess
from pathlib import Path
from typing import Any

from sepilot.tools.base_tool import BaseTool


class SlashCommandTool(BaseTool):
    """Tool for executing custom slash commands"""

    name = "slash_command"
    description = "Execute custom slash commands"
    parameters = {
        "command": "The slash command to execute (e.g., '/review-pr 123') (required)"
    }

    def __init__(self, logger=None):
        super().__init__(logger)
        self.commands_dir = Path.home() / ".sepilot" / "commands"
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        self._load_commands()

    def _load_commands(self):
        """Load available slash commands"""
        self.commands = {}

        # Load from commands directory
        for cmd_file in self.commands_dir.glob("*.md"):
            cmd_name = cmd_file.stem
            try:
                content = cmd_file.read_text(encoding='utf-8')
                # Parse command metadata from frontmatter if present
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        import yaml
                        metadata = yaml.safe_load(parts[1])
                        self.commands[cmd_name] = {
                            'description': metadata.get('description', ''),
                            'content': parts[2].strip(),
                            'file': str(cmd_file)
                        }
                    else:
                        self.commands[cmd_name] = {
                            'description': '',
                            'content': content,
                            'file': str(cmd_file)
                        }
                else:
                    self.commands[cmd_name] = {
                        'description': '',
                        'content': content,
                        'file': str(cmd_file)
                    }
            except Exception as e:
                if self.logger:
                    self.logger.log(f"Error loading command {cmd_name}: {e}")

    def execute(self, command: str) -> str:
        """Execute a slash command"""
        self.validate_params(command=command)

        # Reload commands to get latest
        self._load_commands()

        # Parse command
        if not command.startswith('/'):
            return "Error: Command must start with '/'"

        parts = command[1:].split(None, 1)
        cmd_name = parts[0] if parts else ''
        args = parts[1] if len(parts) > 1 else ''

        # Check if command exists
        if cmd_name not in self.commands:
            available = self._list_available_commands()
            return f"Error: Unknown command '/{cmd_name}'\n\n{available}"

        cmd_info = self.commands[cmd_name]

        # Execute command
        try:
            result = [f"🚀 Executing: /{cmd_name}"]

            if cmd_info['description']:
                result.append(f"Description: {cmd_info['description']}")

            if args:
                result.append(f"Arguments: {args}")

            result.append("")
            result.append("Command output:")
            result.append("─" * 40)

            # Process command content
            content = cmd_info['content']

            # Variable substitution
            if args:
                # Simple variable substitution
                content = content.replace('{{args}}', args)

                # Named arguments (e.g., /cmd arg1=value1 arg2=value2)
                if '=' in args:
                    for arg_pair in args.split():
                        if '=' in arg_pair:
                            key, value = arg_pair.split('=', 1)
                            content = content.replace(f'{{{{{key}}}}}', value)

            # Check if it's a script command
            if content.startswith('#!/'):
                # Execute as script
                script_result = self._execute_script(content, args)
                result.append(script_result)
            else:
                # Return processed content
                result.append(content)

            result.append("─" * 40)
            result.append(f"✅ Command completed: /{cmd_name}")

            return '\n'.join(result)

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _execute_script(self, content: str, args: str) -> str:
        """Execute content as a script"""
        try:
            # Determine interpreter from shebang
            first_line = content.split('\n')[0]
            if first_line.startswith('#!/usr/bin/env '):
                interpreter = first_line[15:].strip()
            elif first_line.startswith('#!'):
                interpreter = first_line[2:].strip()
            else:
                interpreter = '/bin/bash'

            # Create temporary script file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(content)
                script_path = f.name

            # Make executable
            Path(script_path).chmod(0o755)

            # Execute script
            result = subprocess.run(
                [interpreter, script_path] + args.split(),
                capture_output=True,
                text=True,
                timeout=30
            )

            # Clean up
            Path(script_path).unlink()

            if result.returncode == 0:
                return result.stdout
            else:
                return f"Script error (exit code {result.returncode}):\n{result.stderr}"

        except subprocess.TimeoutExpired:
            return "Script execution timed out after 30 seconds"
        except Exception as e:
            return f"Script execution error: {str(e)}"

    def _list_available_commands(self) -> str:
        """List available slash commands"""
        if not self.commands:
            return (
                "No custom slash commands found.\n\n"
                "To create a command, add a .md file to ~/.sepilot/commands/\n"
                "Example: ~/.sepilot/commands/hello.md"
            )

        result = ["Available slash commands:\n"]
        for cmd_name, info in sorted(self.commands.items()):
            desc = info['description'] or "(no description)"
            result.append(f"  /{cmd_name} - {desc}")

        result.append("\nTo create new commands, add .md files to ~/.sepilot/commands/")

        return '\n'.join(result)


class SkillTool(BaseTool):
    """Tool for executing specialized skills"""

    name = "skill"
    description = "Execute a specialized skill"
    parameters = {
        "skill": "The skill name to execute (e.g., 'pdf', 'xlsx') (required)"
    }

    def __init__(self, logger=None):
        super().__init__(logger)
        self.skills_dir = Path.home() / ".sepilot" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._load_skills()

    def _load_skills(self):
        """Load available skills"""
        self.skills = {}

        # Built-in skills
        self._register_builtin_skills()

        # Load custom skills from directory
        for skill_file in self.skills_dir.glob("*.json"):
            skill_name = skill_file.stem
            try:
                with open(skill_file) as f:
                    skill_data = json.load(f)
                    self.skills[skill_name] = skill_data
            except Exception as e:
                if self.logger:
                    self.logger.log(f"Error loading skill {skill_name}: {e}")

    def _register_builtin_skills(self):
        """Register built-in skills"""
        # PDF processing skill
        self.skills['pdf'] = {
            'name': 'pdf',
            'description': 'Process and extract content from PDF files',
            'capabilities': [
                'Extract text from PDF',
                'Extract images from PDF',
                'Convert PDF to text/markdown',
                'Search within PDF content',
                'Extract metadata'
            ],
            'type': 'builtin'
        }

        # Excel/XLSX skill
        self.skills['xlsx'] = {
            'name': 'xlsx',
            'description': 'Process Excel spreadsheets',
            'capabilities': [
                'Read Excel files',
                'Extract data from sheets',
                'Convert to CSV/JSON',
                'Analyze formulas',
                'Generate summary statistics'
            ],
            'type': 'builtin'
        }

        # Database skill
        self.skills['database'] = {
            'name': 'database',
            'description': 'Database operations and queries',
            'capabilities': [
                'Connect to databases',
                'Execute SQL queries',
                'Schema inspection',
                'Data migration',
                'Query optimization'
            ],
            'type': 'builtin'
        }

        # Docker skill
        self.skills['docker'] = {
            'name': 'docker',
            'description': 'Docker container management',
            'capabilities': [
                'List containers/images',
                'Build images',
                'Run containers',
                'Inspect logs',
                'Manage volumes'
            ],
            'type': 'builtin'
        }

    def execute(self, skill: str) -> str:
        """Execute a skill"""
        self.validate_params(skill=skill)

        # Reload skills
        self._load_skills()

        # Parse skill name (support namespace:skill format)
        if ':' in skill:
            _namespace, skill_name = skill.split(':', 1)  # noqa: F841
        else:
            _namespace = None  # noqa: F841
            skill_name = skill

        # Check if skill exists
        if skill_name not in self.skills:
            available = self._list_available_skills()
            return f"Error: Unknown skill '{skill_name}'\n\n{available}"

        skill_info = self.skills[skill_name]

        # Execute skill based on type
        if skill_info.get('type') == 'builtin':
            return self._execute_builtin_skill(skill_name, skill_info)
        else:
            return self._execute_custom_skill(skill_name, skill_info)

    def _execute_builtin_skill(self, skill_name: str, skill_info: dict[str, Any]) -> str:
        """Execute a built-in skill"""
        result = [f"🎯 Loading skill: {skill_name}"]
        result.append(f"Description: {skill_info['description']}\n")

        result.append("Capabilities:")
        for cap in skill_info.get('capabilities', []):
            result.append(f"  • {cap}")

        result.append("\n" + "─" * 40)

        # Skill-specific implementation
        if skill_name == 'pdf':
            result.append(self._pdf_skill_prompt())
        elif skill_name == 'xlsx':
            result.append(self._xlsx_skill_prompt())
        elif skill_name == 'database':
            result.append(self._database_skill_prompt())
        elif skill_name == 'docker':
            result.append(self._docker_skill_prompt())
        else:
            result.append(f"Skill '{skill_name}' is registered but not implemented yet.")

        return '\n'.join(result)

    def _execute_custom_skill(self, skill_name: str, skill_info: dict[str, Any]) -> str:
        """Execute a custom skill"""
        result = [f"🎯 Loading custom skill: {skill_name}"]

        if 'prompt' in skill_info:
            result.append("\nSkill prompt:")
            result.append(skill_info['prompt'])

        if 'tools' in skill_info:
            result.append("\nRequired tools:")
            for tool in skill_info['tools']:
                result.append(f"  • {tool}")

        return '\n'.join(result)

    def _pdf_skill_prompt(self) -> str:
        """Get PDF processing skill prompt"""
        return """
PDF Processing Skill Activated!

Available operations:
1. Extract text: `PyPDF2` or `pdfplumber` for text extraction
2. Extract images: Use `pdf2image` for image extraction
3. Search content: Text search within PDF
4. Extract metadata: Get document properties
5. Convert format: PDF to text/markdown/HTML

Example usage:
```python
import PyPDF2

# Read PDF
with open('document.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
```

What would you like to do with PDF files?
"""

    def _xlsx_skill_prompt(self) -> str:
        """Get Excel processing skill prompt"""
        return """
Excel/XLSX Processing Skill Activated!

Available operations:
1. Read Excel: Use `pandas` or `openpyxl`
2. Data extraction: Get specific sheets/ranges
3. Format conversion: Excel to CSV/JSON
4. Formula analysis: Inspect Excel formulas
5. Data analysis: Statistics and aggregations

Example usage:
```python
import pandas as pd

# Read Excel file
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
print(df.head())
print(df.describe())
```

What would you like to do with Excel files?
"""

    def _database_skill_prompt(self) -> str:
        """Get database skill prompt"""
        return """
Database Operations Skill Activated!

Supported databases:
- SQLite
- PostgreSQL
- MySQL/MariaDB
- MongoDB

Available operations:
1. Connect to database
2. Execute queries
3. Schema inspection
4. Data migration
5. Performance optimization

Example usage:
```python
import sqlite3

# Connect to database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Execute query
cursor.execute("SELECT * FROM users")
results = cursor.fetchall()
```

What database operation do you need?
"""

    def _docker_skill_prompt(self) -> str:
        """Get Docker skill prompt"""
        return """
Docker Management Skill Activated!

Available operations:
1. Container management: list, start, stop, remove
2. Image management: build, pull, push, list
3. Volume management: create, list, inspect
4. Network management: create, connect, disconnect
5. Logs and debugging: view logs, exec into containers

Example commands:
```bash
# List containers
docker ps -a

# Build image
docker build -t myapp .

# Run container
docker run -d -p 8080:80 myapp

# View logs
docker logs container_id
```

What Docker operation do you need?
"""

    def _list_available_skills(self) -> str:
        """List available skills"""
        if not self.skills:
            return "No skills available"

        result = ["Available skills:\n"]

        # Group by type
        builtin = [s for s in self.skills.values() if s.get('type') == 'builtin']
        custom = [s for s in self.skills.values() if s.get('type') != 'builtin']

        if builtin:
            result.append("Built-in skills:")
            for skill in builtin:
                result.append(f"  • {skill['name']}: {skill['description']}")

        if custom:
            result.append("\nCustom skills:")
            for skill in custom:
                desc = skill.get('description', '(no description)')
                result.append(f"  • {skill['name']}: {desc}")

        result.append("\nTo use a skill: skill(skill='skill_name')")

        return '\n'.join(result)
