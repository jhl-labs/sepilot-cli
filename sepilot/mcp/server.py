"""MCP (Model Context Protocol) Server for SE Pilot

This module implements an MCP server that exposes SE Pilot's capabilities
to Claude Desktop and other MCP clients.

MCP Protocol: https://modelcontextprotocol.io/
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    # Fallback types for when MCP SDK is not installed
    Server = None
    stdio_server = None
    Tool = dict
    TextContent = dict
    ImageContent = dict
    EmbeddedResource = dict


def _build_shell_command(command: str) -> list[str]:
    """Build a platform-safe shell invocation command."""
    if os.name == "nt":
        return ["cmd", "/c", command]

    shell_path = os.environ.get("SHELL")
    if shell_path and Path(shell_path).exists():
        return [shell_path, "-lc", command]
    if Path("/bin/bash").exists():
        return ["/bin/bash", "-lc", command]
    return ["/bin/sh", "-c", command]


class SEPilotMCPServer:
    """MCP Server exposing SE Pilot functionality"""

    def __init__(self, working_directory: str | None = None):
        """
        Initialize MCP server

        Args:
            working_directory: Working directory for file operations
        """
        if not HAS_MCP:
            raise ImportError(
                "MCP SDK is required. Install with: pip install mcp"
            )

        self.server = Server("sepilot-mcp")
        self.working_directory = Path(working_directory or os.getcwd())

        # Register handlers
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def _register_tools(self):
        """Register SE Pilot tools as MCP tools"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="file_read",
                    description="Read contents of a file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="file_write",
                    description="Write content to a file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                ),
                Tool(
                    name="file_edit",
                    description="Edit a file by replacing text",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "old_text": {
                                "type": "string",
                                "description": "Text to find and replace"
                            },
                            "new_text": {
                                "type": "string",
                                "description": "New text to insert"
                            }
                        },
                        "required": ["file_path", "old_text", "new_text"]
                    }
                ),
                Tool(
                    name="search_content",
                    description="Search for text patterns in files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Pattern to search for (regex supported)"
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "File pattern to search in (e.g., '*.py')"
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory to search in (optional)"
                            }
                        },
                        "required": ["pattern"]
                    }
                ),
                Tool(
                    name="bash_execute",
                    description="Execute a bash command",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Bash command to execute"
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Timeout in seconds (optional)"
                            }
                        },
                        "required": ["command"]
                    }
                ),
                Tool(
                    name="codebase_analyze",
                    description="Analyze codebase structure and get overview",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to analyze (optional, defaults to working directory)"
                            },
                            "include_metrics": {
                                "type": "boolean",
                                "description": "Include code metrics (optional)"
                            }
                        }
                    }
                ),
                Tool(
                    name="git_operation",
                    description="Execute git operations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["status", "diff", "log", "add", "commit", "push"],
                                "description": "Git operation to perform"
                            },
                            "args": {
                                "type": "object",
                                "description": "Additional arguments for the operation"
                            }
                        },
                        "required": ["operation"]
                    }
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Execute a tool"""
            try:
                result = await self._execute_tool(name, arguments)
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _resolve_safe_path(self, relative_path: str) -> Path | None:
        """Resolve a relative path safely, preventing path traversal.

        Args:
            relative_path: Relative path string

        Returns:
            Resolved absolute Path, or None if traversal detected
        """
        try:
            resolved = (self.working_directory / relative_path).resolve()
            # Ensure the resolved path is within working directory
            resolved.relative_to(self.working_directory.resolve())
            return resolved
        except (ValueError, OSError):
            return None

    def _register_resources(self):
        """Register SE Pilot resources"""

        # Directories and patterns to exclude from resource listing
        _EXCLUDE_DIRS = {
            ".git", ".svn", ".hg", "__pycache__", "node_modules",
            ".mypy_cache", ".pytest_cache", ".tox", ".venv", "venv",
            ".eggs", "dist", "build", ".idea", ".vscode",
        }
        _EXCLUDE_EXTS = {
            ".pyc", ".pyo", ".so", ".dylib", ".dll",
            ".exe", ".bin", ".dat", ".db", ".sqlite",
        }

        @self.server.list_resources()
        async def list_resources() -> list[dict]:
            """List available resources"""
            resources = []

            try:
                for file_path in self.working_directory.rglob("*"):
                    if file_path.is_file():
                        # Skip hidden directories and excluded dirs
                        if any(
                            part.startswith(".") or part in _EXCLUDE_DIRS
                            for part in file_path.relative_to(self.working_directory).parts[:-1]
                        ):
                            continue
                        # Skip excluded extensions
                        if file_path.suffix.lower() in _EXCLUDE_EXTS:
                            continue

                        rel_path = file_path.relative_to(self.working_directory)
                        resources.append({
                            "uri": f"file:///{rel_path}",
                            "name": str(rel_path),
                            "description": f"File: {rel_path}",
                            "mimeType": self._get_mime_type(file_path)
                        })

                    if len(resources) >= 200:
                        break
            except Exception as e:
                print(f"Error listing resources: {e}", file=sys.stderr)

            return resources[:200]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource"""
            if uri.startswith("file:///"):
                relative_path = uri[8:]
                file_path = self._resolve_safe_path(relative_path)
                if file_path is None:
                    return "Error: path traversal detected"
                if not file_path.is_file():
                    return f"Error: file not found: {relative_path}"
                try:
                    content = file_path.read_text(encoding='utf-8')
                    # Limit response size to prevent OOM
                    max_size = 500_000
                    if len(content) > max_size:
                        return content[:max_size] + f"\n\n[Truncated at {max_size} chars]"
                    return content
                except UnicodeDecodeError:
                    return "Error: file is not valid UTF-8 text"
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            return "Unknown resource type"

    def _register_prompts(self):
        """Register SE Pilot prompts"""

        @self.server.list_prompts()
        async def list_prompts() -> list[dict]:
            """List available prompts"""
            return [
                {
                    "name": "analyze_code",
                    "description": "Analyze code structure and suggest improvements",
                    "arguments": [
                        {
                            "name": "file_path",
                            "description": "Path to the file to analyze",
                            "required": True
                        }
                    ]
                },
                {
                    "name": "write_tests",
                    "description": "Generate unit tests for code",
                    "arguments": [
                        {
                            "name": "file_path",
                            "description": "Path to the file to test",
                            "required": True
                        }
                    ]
                },
                {
                    "name": "refactor_code",
                    "description": "Suggest refactoring improvements",
                    "arguments": [
                        {
                            "name": "file_path",
                            "description": "Path to the file to refactor",
                            "required": True
                        }
                    ]
                },
            ]

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str]) -> dict:
            """Get a specific prompt"""
            prompts = {
                "analyze_code": {
                    "description": "Analyze code structure",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": f"Analyze the code in {arguments.get('file_path')} and provide:\n"
                                       f"1. Code structure overview\n"
                                       f"2. Potential issues\n"
                                       f"3. Suggestions for improvement\n"
                                       f"4. Best practices compliance"
                            }
                        }
                    ]
                },
                "write_tests": {
                    "description": "Generate unit tests",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": f"Generate comprehensive unit tests for {arguments.get('file_path')}. "
                                       f"Include:\n"
                                       f"1. Test cases for main functions\n"
                                       f"2. Edge cases\n"
                                       f"3. Error handling tests\n"
                                       f"4. Mock external dependencies if needed"
                            }
                        }
                    ]
                },
                "refactor_code": {
                    "description": "Suggest refactoring",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": f"Suggest refactoring improvements for {arguments.get('file_path')}:\n"
                                       f"1. Code smell identification\n"
                                       f"2. Design pattern suggestions\n"
                                       f"3. Performance optimizations\n"
                                       f"4. Readability improvements"
                            }
                        }
                    ]
                }
            }

            return prompts.get(name, {})

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return result.

        Handles both direct SE Pilot tool mapping and built-in MCP tool
        implementations for file operations, search, and git.
        """
        # Built-in tool implementations that work without full SE Pilot tools
        builtin_handlers = {
            "file_read": self._tool_file_read,
            "file_write": self._tool_file_write,
            "file_edit": self._tool_file_edit,
            "search_content": self._tool_search_content,
            "bash_execute": self._tool_bash_execute,
            "codebase_analyze": self._tool_codebase_analyze,
            "git_operation": self._tool_git_operation,
        }

        handler = builtin_handlers.get(name)
        if handler:
            return await handler(arguments)

        # Fallback: try SE Pilot LangChain tools
        try:
            from sepilot.tools.langchain_tools import get_all_tools

            tools = get_all_tools()
            tool_map = {tool.name: tool for tool in tools}

            if name not in tool_map:
                raise ValueError(f"Unknown tool: {name}")

            result = tool_map[name].invoke(arguments)
            return str(result)
        except ImportError:
            raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {str(e)}") from e

    async def _tool_file_read(self, args: dict[str, Any]) -> str:
        """Read a file safely."""
        file_path = args.get("file_path", "")
        resolved = self._resolve_safe_path(file_path)
        if resolved is None:
            return "Error: path traversal detected"
        if not resolved.is_file():
            return f"Error: file not found: {file_path}"
        try:
            content = resolved.read_text(encoding="utf-8")
            if len(content) > 500_000:
                return content[:500_000] + "\n\n[Truncated]"
            return content
        except UnicodeDecodeError:
            return "Error: file is not valid UTF-8 text"
        except Exception as e:
            return f"Error reading file: {e}"

    async def _tool_file_write(self, args: dict[str, Any]) -> str:
        """Write content to a file safely."""
        file_path = args.get("file_path", "")
        content = args.get("content", "")
        resolved = self._resolve_safe_path(file_path)
        if resolved is None:
            return "Error: path traversal detected"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} chars to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"

    async def _tool_file_edit(self, args: dict[str, Any]) -> str:
        """Edit a file by replacing text."""
        file_path = args.get("file_path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        resolved = self._resolve_safe_path(file_path)
        if resolved is None:
            return "Error: path traversal detected"
        if not resolved.is_file():
            return f"Error: file not found: {file_path}"
        try:
            content = resolved.read_text(encoding="utf-8")
            if old_text not in content:
                return "Error: old_text not found in file"
            count = content.count(old_text)
            if count > 1:
                return f"Error: old_text found {count} times, must be unique"
            new_content = content.replace(old_text, new_text, 1)
            resolved.write_text(new_content, encoding="utf-8")
            return f"Successfully edited {file_path}"
        except Exception as e:
            return f"Error editing file: {e}"

    async def _tool_search_content(self, args: dict[str, Any]) -> str:
        """Search for text patterns in files."""
        import subprocess

        pattern = args.get("pattern", "")
        file_pattern = args.get("file_pattern", "*")
        search_path = args.get("path", ".")

        resolved = self._resolve_safe_path(search_path)
        if resolved is None:
            return "Error: path traversal detected"

        try:
            # Use grep with line numbers for useful output
            cmd = ["grep", "-rn", "--include", file_pattern, pattern, str(resolved)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout[:50000]
            elif result.returncode == 1:
                return "No matches found"
            else:
                return f"Search error: {result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return "Error: search timed out"
        except Exception as e:
            return f"Error searching: {e}"

    async def _tool_bash_execute(self, args: dict[str, Any]) -> str:
        """Execute a bash command."""
        import subprocess

        command = args.get("command", "")
        timeout = args.get("timeout", 30)

        if not command:
            return "Error: empty command"

        try:
            result = subprocess.run(
                _build_shell_command(command),
                capture_output=True,
                text=True,
                timeout=min(timeout, 120),
                cwd=str(self.working_directory),
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output[:100000] or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"

    async def _tool_codebase_analyze(self, args: dict[str, Any]) -> str:
        """Analyze codebase structure."""
        import subprocess

        path = args.get("path", ".")
        resolved = self._resolve_safe_path(path)
        if resolved is None:
            return "Error: path traversal detected"

        lines = [f"# Codebase Analysis: {resolved}\n"]

        # Directory tree
        try:
            result = subprocess.run(
                ["find", str(resolved), "-type", "f", "-not", "-path", "*/.git/*",
                 "-not", "-path", "*/__pycache__/*", "-not", "-path", "*/node_modules/*"],
                capture_output=True, text=True, timeout=10
            )
            files = [f for f in result.stdout.strip().split("\n") if f]
            lines.append(f"Total files: {len(files)}")

            # Count by extension
            ext_counts: dict[str, int] = {}
            for f in files:
                ext = Path(f).suffix or "(no ext)"
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
            lines.append("\nFile types:")
            for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1])[:20]:
                lines.append(f"  {ext}: {count}")
        except Exception as e:
            lines.append(f"Error analyzing: {e}")

        return "\n".join(lines)

    async def _tool_git_operation(self, args: dict[str, Any]) -> str:
        """Execute git operations."""
        import subprocess

        operation = args.get("operation", "")
        extra_args = args.get("args", {})

        allowed_ops = {
            "status": ["git", "status", "--short"],
            "diff": ["git", "diff"],
            "log": ["git", "log", "--oneline", "-20"],
            "add": ["git", "add"],
            "commit": ["git", "commit"],
            "push": ["git", "push"],
        }

        if operation not in allowed_ops:
            return f"Error: unknown git operation '{operation}'. Allowed: {list(allowed_ops.keys())}"

        cmd = allowed_ops[operation].copy()

        # Handle extra arguments for specific operations
        if operation == "add":
            files = extra_args.get("files", ["."])
            if isinstance(files, str):
                files = [files]
            cmd.extend(files)
        elif operation == "commit":
            message = extra_args.get("message", "")
            if not message:
                return "Error: commit requires a message"
            cmd.extend(["-m", message])
        elif operation == "diff":
            if extra_args.get("staged"):
                cmd.append("--staged")
        elif operation == "log":
            n = extra_args.get("n", 20)
            cmd = ["git", "log", "--oneline", f"-{n}"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=30,
                cwd=str(self.working_directory),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"
            return output[:50000] or "(no output)"
        except Exception as e:
            return f"Error: {e}"

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for a file"""
        suffix = file_path.suffix.lower()
        mime_types = {
            ".py": "text/x-python",
            ".js": "text/javascript",
            ".ts": "text/typescript",
            ".md": "text/markdown",
            ".json": "application/json",
            ".yaml": "text/yaml",
            ".yml": "text/yaml",
            ".txt": "text/plain",
            ".html": "text/html",
            ".css": "text/css",
        }
        return mime_types.get(suffix, "text/plain")

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point for MCP server"""
    server = SEPilotMCPServer()
    await server.run()


def run_mcp_server():
    """Run MCP server (synchronous wrapper)"""
    if not HAS_MCP:
        print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main())


if __name__ == "__main__":
    run_mcp_server()
