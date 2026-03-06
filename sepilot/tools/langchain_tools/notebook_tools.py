"""Notebook editing tools for LangChain agent."""


from langchain_core.tools import tool


@tool
def notebook_edit(
    notebook_path: str,
    new_source: str = "",
    cell_id: str | None = None,
    cell_type: str | None = None,
    edit_mode: str = "replace"
) -> str:
    """Edit cells in a Jupyter Notebook (.ipynb file).

    Args:
        notebook_path: Path to the notebook file (required)
        new_source: New content for the cell (required except for delete)
        cell_id: ID of the cell to edit (optional for insert mode)
        cell_type: Type of cell: 'code' or 'markdown' (required for insert)
        edit_mode: Edit mode: 'replace', 'insert', 'delete' (default: replace)

    Returns:
        Success or error message

    Examples:
        # Replace cell content
        notebook_edit(notebook_path="analysis.ipynb", cell_id="0", new_source="import pandas as pd")

        # Insert new markdown cell
        notebook_edit(notebook_path="report.ipynb", cell_type="markdown", edit_mode="insert",
                     new_source="# Results\\nHere are the findings...")

        # Delete a cell
        notebook_edit(notebook_path="notebook.ipynb", cell_id="cell_abc123", edit_mode="delete")
    """
    from sepilot.tools.file_tools.notebook_tool import NotebookTool
    tool_instance = NotebookTool()
    return tool_instance.execute(notebook_path, new_source, cell_id, cell_type, edit_mode)


__all__ = ['notebook_edit']
