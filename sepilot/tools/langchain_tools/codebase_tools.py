"""Codebase exploration tools for LangChain agent.

Provides codebase, search_content, find_file, find_definition, get_structure tools.
"""

from langchain_core.tools import tool

from sepilot.tools.langchain_tools.file_tools import _preferred_required_file_path

# Singleton explorer instance
_explorer = None


def _get_explorer():
    """Get or create CodebaseExplorer singleton."""
    global _explorer
    if _explorer is None:
        from sepilot.tools.codebase_tools import CodebaseExplorer
        _explorer = CodebaseExplorer()
    return _explorer


@tool
def codebase(action: str, filename: str = "", search_term: str = "", file_pattern: str = "*.py", name: str = "", file_path: str = "") -> str:
    """Explore and navigate the codebase to find files, classes, and functions.

    Args:
        action: The action to perform - REQUIRED. Valid actions:
            - "find_file": Find a file by name
            - "get_structure": Get project structure overview
            - "search_content": Search for text in files
            - "find_definition": Find class or function definitions
        filename: Name of file to find (for find_file action)
        search_term: Term to search for (for search_content action)
        file_pattern: File pattern to search in (for search_content, default: *.py)
        name: Name of class/function to find (for find_definition action)
        file_path: Path to file (for get_context action)

    Returns:
        Search results or file information

    Examples:
        - codebase(action="find_file", filename="main.py")
        - codebase(action="get_structure")
        - codebase(action="search_content", search_term="TODO")
        - codebase(action="find_definition", name="MyClass")
    """
    explorer = _get_explorer()

    if action == "find_file":
        if not filename:
            return "Error: filename parameter is required for find_file action"

        result = explorer.find_file_by_name(filename)
        if result:
            return f"Found file: {result}\n\nNEXT STEP: Use file_read tool with file_path=\"{result}\" to read this file."
        else:
            patterns = explorer.find_files_by_pattern(f"*{filename}*")
            if patterns:
                best_match = patterns[0]
                output = f"Found {len(patterns)} matching files. Best match: {best_match}\n"
                if len(patterns) > 1:
                    output += f"Other matches: {', '.join(patterns[1:6])}\n"
                output += f"\nNEXT STEP: Use file_read tool with file_path=\"{best_match}\" to read the file."
                return output
            return f"No files found matching '{filename}'. Try using get_structure to see all available files."

    elif action == "get_structure":
        all_files = explorer.get_all_files(max_files=100)

        output_lines = [f"Project root: {explorer.project_root}"]
        output_lines.append(f"Total: {len(all_files)} files found\n")
        output_lines.append("Key source files:")

        python_files = [f for f in all_files if f.endswith('.py')]
        config_files = [f for f in all_files if f.endswith(('.yaml', '.yml', '.json', '.toml', '.ini', '.cfg'))]
        doc_files = [f for f in all_files if f.endswith(('.md', '.txt', '.rst'))]

        if python_files:
            output_lines.append("\nPython files (.py):")
            for f in python_files[:30]:
                output_lines.append(f"  {f}")
            if len(python_files) > 30:
                output_lines.append(f"  ... and {len(python_files) - 30} more .py files")

        if config_files:
            output_lines.append("\nConfig files:")
            for f in config_files[:10]:
                output_lines.append(f"  {f}")

        if doc_files:
            output_lines.append("\nDocumentation:")
            for f in doc_files[:5]:
                output_lines.append(f"  {f}")

        output_lines.append("\nNEXT STEP: Use file_read to read a file from the list above with the exact path.")

        return "\n".join(output_lines)

    elif action == "search_content":
        if not search_term:
            return "Error: search_term parameter is required for search_content action"

        results = explorer.find_files_with_content(search_term, file_pattern)

        if results:
            output = [f"Found '{search_term}' in {len(results)} files:"]
            for file_path_result, matches in results[:5]:
                output.append(f"\n{file_path_result}:")
                for line_no, line in matches[:3]:
                    output.append(f"  Line {line_no}: {line[:100]}")
            return "\n".join(output)
        return f"No files containing '{search_term}' found"

    elif action == "find_definition":
        if not name:
            return "Error: name parameter is required for find_definition action"

        results = explorer.find_class_or_function(name)

        if results:
            output = [f"Found {len(results)} definitions of '{name}':"]
            for file_path_result, line_no in results:
                output.append(f"  {file_path_result}:{line_no}")
            output.append("\nNEXT STEP: Use file_read with file_path to read the file.")
            return "\n".join(output)
        return f"No definitions of '{name}' found"

    else:
        return f"Error: Unknown action '{action}'. Valid actions: find_file, get_structure, search_content, find_definition"


@tool
def search_content(search_term: str, file_pattern: str = "*.py") -> str:
    """Search for text content across files in the codebase.

    Args:
        search_term: The text/keyword to search for (e.g., "TODO", "class MyClass", "def my_function")
        file_pattern: File pattern to search in (default: *.py). Examples: *.py, *.js, *.md, *.txt

    Returns:
        List of files containing the search term with line numbers and context

    Examples:
        - search_content(search_term="TODO", file_pattern="*.py")
        - search_content(search_term="EnhancedAgentState", file_pattern="*.py")
        - search_content(search_term="import pandas", file_pattern="*.py")
    """
    explorer = _get_explorer()
    results = explorer.find_files_with_content(search_term, file_pattern)

    if results:
        output = [f"Found '{search_term}' in {len(results)} files:"]
        for file_path_result, matches in results[:5]:
            output.append(f"\n{file_path_result}:")
            for line_no, line in matches[:3]:
                output.append(f"  Line {line_no}: {line[:100]}")
        return "\n".join(output)
    return f"No files containing '{search_term}' found"


@tool
def find_file(filename: str) -> str:
    """Find a file by its name in the codebase.

    Args:
        filename: Name of the file to find (e.g., "main.py", "config.json", "README.md")

    Returns:
        Path to the file if found, or suggestions if multiple matches exist

    Examples:
        - find_file(filename="main.py")
        - find_file(filename="base_agent.py")
        - find_file(filename="README.md")
    """
    preferred_path = _preferred_required_file_path(filename)
    if preferred_path:
        return (
            f"Found file: {preferred_path}\n\n"
            f"NEXT STEP: Use file_read tool with file_path=\"{preferred_path}\" to read this file."
        )

    explorer = _get_explorer()

    result = explorer.find_file_by_name(filename)
    if result:
        return f"Found file: {result}\n\nNEXT STEP: Use file_read tool with file_path=\"{result}\" to read this file."
    else:
        patterns = explorer.find_files_by_pattern(f"*{filename}*")
        if patterns:
            best_match = patterns[0]
            output = f"Found {len(patterns)} matching files. Best match: {best_match}\n"
            if len(patterns) > 1:
                output += f"Other matches: {', '.join(patterns[1:6])}\n"
            output += f"\nNEXT STEP: Use file_read tool with file_path=\"{best_match}\" to read the file."
            return output
        return f"No files found matching '{filename}'. Try using get_structure to see all available files."


@tool
def find_definition(name: str) -> str:
    """Find where a class or function is defined in the codebase.

    Args:
        name: Name of the class or function to find (e.g., "MyClass", "my_function")

    Returns:
        File paths and line numbers where the definition is found

    Examples:
        - find_definition(name="BaseAgent")
        - find_definition(name="ReactAgent")
        - find_definition(name="execute")
    """
    explorer = _get_explorer()
    results = explorer.find_class_or_function(name)

    if results:
        output = [f"Found {len(results)} definitions of '{name}':"]
        for file_path_result, line_no in results:
            output.append(f"  {file_path_result}:{line_no}")
        output.append("\nNEXT STEP: Use file_read with file_path to read the file.")
        return "\n".join(output)
    return f"No definitions of '{name}' found"


@tool
def get_structure() -> str:
    """Get an overview of the project structure.

    Shows Python files, config files, and documentation in the project.
    Use this ONLY ONCE at the beginning to understand the codebase layout.

    Returns:
        Hierarchical list of files organized by type (Python, config, docs)

    Note:
        After calling this once, use find_file or file_read with specific paths.
        DO NOT call this repeatedly - it's meant for initial exploration only.
    """
    explorer = _get_explorer()
    all_files = explorer.get_all_files(max_files=100)

    output_lines = [f"Project root: {explorer.project_root}"]
    output_lines.append(f"Total: {len(all_files)} files found\n")
    output_lines.append("Key source files:")

    python_files = [f for f in all_files if f.endswith('.py')]
    config_files = [f for f in all_files if f.endswith(('.yaml', '.yml', '.json', '.toml', '.ini', '.cfg'))]
    doc_files = [f for f in all_files if f.endswith(('.md', '.txt', '.rst'))]

    if python_files:
        output_lines.append("\nPython files (.py):")
        for f in python_files[:30]:
            output_lines.append(f"  {f}")
        if len(python_files) > 30:
            output_lines.append(f"  ... and {len(python_files) - 30} more .py files")

    if config_files:
        output_lines.append("\nConfig files:")
        for f in config_files[:10]:
            output_lines.append(f"  {f}")

    if doc_files:
        output_lines.append("\nDocumentation:")
        for f in doc_files[:5]:
            output_lines.append(f"  {f}")

    output_lines.append("\nNEXT STEP: Use file_read to read a file from the list above with the exact path.")

    return "\n".join(output_lines)


__all__ = [
    'codebase',
    'search_content',
    'find_file',
    'find_definition',
    'get_structure',
    '_get_explorer',
]
