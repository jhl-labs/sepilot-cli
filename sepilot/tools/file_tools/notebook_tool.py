"""Jupyter Notebook editing tool"""

import json
import uuid
from pathlib import Path
from typing import Any

from sepilot.tools.base_tool import BaseTool


class NotebookTool(BaseTool):
    """Tool for editing Jupyter Notebook files"""

    name = "notebook_edit"
    description = "Edit Jupyter Notebook (.ipynb) files"
    parameters = {
        "notebook_path": "Path to the notebook file (required)",
        "cell_id": "ID of the cell to edit (optional for insert mode)",
        "cell_type": "Type of cell: 'code' or 'markdown' (required for insert)",
        "edit_mode": "Edit mode: 'replace', 'insert', 'delete' (default: replace)",
        "new_source": "New content for the cell (required except for delete)"
    }

    def execute(
        self,
        notebook_path: str,
        new_source: str = "",
        cell_id: str | None = None,
        cell_type: str | None = None,
        edit_mode: str = "replace"
    ) -> str:
        """Edit a Jupyter notebook cell"""
        self.validate_params(notebook_path=notebook_path)

        try:
            path = Path(notebook_path).resolve()

            # Security check
            project_root = Path.cwd().resolve()
            try:
                path.relative_to(project_root)
            except ValueError:
                return f"Error: Path must be within project directory: {notebook_path}"

            # Check if file exists
            if not path.exists():
                if edit_mode == "insert" and cell_type:
                    # Create new notebook
                    notebook = self._create_empty_notebook()
                else:
                    return f"Error: Notebook not found: {notebook_path}"
            else:
                # Read existing notebook
                with open(path, encoding='utf-8') as f:
                    notebook = json.load(f)

            # Validate notebook structure
            if 'cells' not in notebook:
                notebook['cells'] = []

            cells = notebook['cells']

            # Handle different edit modes
            if edit_mode == "replace":
                if not cell_id:
                    return "Error: cell_id required for replace mode"

                # Find cell by ID
                cell_found = False
                for cell in cells:
                    if self._get_cell_id(cell) == cell_id:
                        # Update source
                        if isinstance(new_source, str):
                            cell['source'] = new_source.split('\n')
                        else:
                            cell['source'] = new_source
                        cell_found = True
                        break

                if not cell_found:
                    # Try by index if cell_id is numeric
                    try:
                        idx = int(cell_id)
                        if 0 <= idx < len(cells):
                            cells[idx]['source'] = new_source.split('\n')
                            cell_found = True
                    except ValueError:
                        pass

                if not cell_found:
                    return f"Error: Cell with ID '{cell_id}' not found"

            elif edit_mode == "insert":
                if not cell_type or cell_type not in ['code', 'markdown']:
                    return "Error: cell_type must be 'code' or 'markdown' for insert mode"

                # Create new cell
                new_cell = self._create_cell(cell_type, new_source)

                if cell_id:
                    # Insert after specified cell
                    inserted = False
                    for i, cell in enumerate(cells):
                        if self._get_cell_id(cell) == cell_id:
                            cells.insert(i + 1, new_cell)
                            inserted = True
                            break

                    if not inserted:
                        # Try by index
                        try:
                            idx = int(cell_id)
                            if 0 <= idx < len(cells):
                                cells.insert(idx + 1, new_cell)
                                inserted = True
                        except ValueError:
                            pass

                    if not inserted:
                        cells.append(new_cell)
                else:
                    # Insert at beginning
                    cells.insert(0, new_cell)

            elif edit_mode == "delete":
                if not cell_id:
                    return "Error: cell_id required for delete mode"

                # Find and delete cell
                deleted = False
                for i, cell in enumerate(cells):
                    if self._get_cell_id(cell) == cell_id:
                        del cells[i]
                        deleted = True
                        break

                if not deleted:
                    # Try by index
                    try:
                        idx = int(cell_id)
                        if 0 <= idx < len(cells):
                            del cells[idx]
                            deleted = True
                    except ValueError:
                        pass

                if not deleted:
                    return f"Error: Cell with ID '{cell_id}' not found"

            else:
                return f"Error: Invalid edit_mode: {edit_mode}"

            # Write back notebook
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)

            # Return success message
            if edit_mode == "replace":
                return f"✅ Updated cell '{cell_id}' in {notebook_path}"
            elif edit_mode == "insert":
                return f"✅ Inserted new {cell_type} cell in {notebook_path}"
            elif edit_mode == "delete":
                return f"✅ Deleted cell '{cell_id}' from {notebook_path}"

        except json.JSONDecodeError as e:
            return f"Error: Invalid notebook format: {str(e)}"
        except Exception as e:
            return f"Error editing notebook: {str(e)}"

    def _get_cell_id(self, cell: dict[str, Any]) -> str:
        """Get cell ID from metadata or generate one"""
        metadata = cell.get('metadata', {})
        if 'id' in metadata:
            return metadata['id']
        elif 'cell_id' in cell:
            return cell['cell_id']
        else:
            # Generate ID based on content hash
            import hashlib
            source = ''.join(cell.get('source', []))
            return hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()[:8]

    def _create_cell(self, cell_type: str, source: str) -> dict[str, Any]:
        """Create a new cell"""
        cell = {
            'cell_type': cell_type,
            'metadata': {
                'id': str(uuid.uuid4())[:8]
            },
            'source': source.split('\n') if isinstance(source, str) else source
        }

        if cell_type == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

        return cell

    def _create_empty_notebook(self) -> dict[str, Any]:
        """Create an empty notebook structure"""
        return {
            'cells': [],
            'metadata': {
                'kernelspec': {
                    'display_name': 'Python 3',
                    'language': 'python',
                    'name': 'python3'
                },
                'language_info': {
                    'name': 'python',
                    'version': '3.9.0'
                }
            },
            'nbformat': 4,
            'nbformat_minor': 5
        }
