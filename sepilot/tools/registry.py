"""Tool registry for managing available tools

DEPRECATED: This module is no longer used.
Use sepilot.tools.langchain_tools instead for tool definitions.

This file is kept for backward compatibility but will be removed in a future version.
All tools are now defined and used through langchain_tools.py.
"""

import warnings

from sepilot.loggers.file_logger import FileLogger
from sepilot.tools.base_tool import BaseTool

# Deprecation warning
warnings.warn(
    "ToolRegistry is deprecated and will be removed in a future version. "
    "Use langchain_tools instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import tool implementations
from sepilot.tools.codebase_tools import CodebaseTool
from sepilot.tools.file_tools.edit_tool import FileEditTool
from sepilot.tools.file_tools.glob_tool import GlobTool
from sepilot.tools.file_tools.read_tool import FileReadTool
from sepilot.tools.file_tools.search_tool import SearchTool
from sepilot.tools.file_tools.write_tool import FileWriteTool
from sepilot.tools.git_tools.git_tool import GitTool
from sepilot.tools.shell_tools.bash_tool import BashTool
from sepilot.tools.shell_tools.process_tool import ProcessTool
from sepilot.tools.task_tools.todo_tool import TodoTool
from sepilot.tools.web_tools.search_web_tool import WebSearchTool


class ToolRegistry:
    """Registry for managing and accessing tools"""

    def __init__(self, logger: FileLogger):
        self.logger = logger
        self.tools: dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tools"""
        # Codebase exploration tool (register first for priority)
        self.register(CodebaseTool(self.logger))

        # File tools
        self.register(FileReadTool(self.logger))
        self.register(FileWriteTool(self.logger))
        self.register(FileEditTool(self.logger))
        self.register(SearchTool(self.logger))
        self.register(GlobTool(self.logger))

        # Shell tools
        self.register(BashTool(self.logger))
        self.register(ProcessTool(self.logger))

        # Git tools
        self.register(GitTool(self.logger))

        # Web tools
        self.register(WebSearchTool(self.logger))

        # Task tools
        self.register(TodoTool(self.logger))

    def register(self, tool: BaseTool) -> None:
        """Register a new tool"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name"""
        return self.tools.get(name)

    def get_tool_descriptions(self) -> str:
        """Get descriptions of all available tools"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.get_description())
        return "\n\n".join(descriptions)

    def list_tools(self) -> list[str]:
        """List all available tool names"""
        return list(self.tools.keys())
