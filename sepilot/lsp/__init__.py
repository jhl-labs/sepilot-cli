"""LSP (Language Server Protocol) Client Module.

Provides Language Server Protocol integration for enhanced code intelligence:
- Go to definition
- Find references
- Hover information
- Document symbols
- Workspace symbols
- Call hierarchy

Supports multiple language servers:
- Python: pyright
- TypeScript/JavaScript: typescript-language-server
- Go: gopls
- Rust: rust-analyzer
"""

from sepilot.lsp.models import (
    Location,
    Range,
    Position,
    HoverInfo,
    SymbolInformation,
    CallHierarchyItem,
    DiagnosticSeverity,
    Diagnostic,
)
from sepilot.lsp.client import LSPClient
from sepilot.lsp.servers import (
    LSPServerConfig,
    get_server_config,
    get_available_servers,
)
from sepilot.lsp.operations import (
    LSPOperations,
    get_lsp_operations,
    check_file_sync,
    get_diagnostics_sync,
)

__all__ = [
    # Models
    "Location",
    "Range",
    "Position",
    "HoverInfo",
    "SymbolInformation",
    "CallHierarchyItem",
    "DiagnosticSeverity",
    "Diagnostic",
    # Client
    "LSPClient",
    # Servers
    "LSPServerConfig",
    "get_server_config",
    "get_available_servers",
    # Operations
    "LSPOperations",
    "get_lsp_operations",
    "check_file_sync",
    "get_diagnostics_sync",
]
