"""LSP server configurations."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LSPServerConfig:
    """Configuration for a language server."""

    name: str
    language: str
    command: list[str]
    install_command: str | None = None
    install_check: str | None = None
    args: list[str] = field(default_factory=list)
    init_options: dict[str, Any] = field(default_factory=dict)
    workspace_config: dict[str, Any] = field(default_factory=dict)
    file_patterns: list[str] = field(default_factory=list)

    def is_installed(self) -> bool:
        """Check if the server is installed."""
        if self.install_check:
            return shutil.which(self.install_check) is not None
        # Try to find the main command
        if self.command:
            return shutil.which(self.command[0]) is not None
        return False

    def get_full_command(self) -> list[str]:
        """Get full command with arguments."""
        return self.command + self.args


# Server configurations
SERVER_CONFIGS: dict[str, LSPServerConfig] = {
    "python": LSPServerConfig(
        name="pyright",
        language="python",
        command=["pyright-langserver", "--stdio"],
        install_command="npm install -g pyright",
        install_check="pyright-langserver",
        file_patterns=["*.py", "*.pyi"],
        init_options={
            "python": {
                "analysis": {
                    "autoSearchPaths": True,
                    "useLibraryCodeForTypes": True,
                    "diagnosticMode": "openFilesOnly",
                }
            }
        },
    ),
    "typescript": LSPServerConfig(
        name="typescript-language-server",
        language="typescript",
        command=["typescript-language-server", "--stdio"],
        install_command="npm install -g typescript typescript-language-server",
        install_check="typescript-language-server",
        file_patterns=["*.ts", "*.tsx", "*.js", "*.jsx", "*.mjs", "*.cjs"],
        init_options={
            "preferences": {
                "includeInlayParameterNameHints": "all",
                "includeInlayPropertyDeclarationTypeHints": True,
                "includeInlayFunctionParameterTypeHints": True,
            }
        },
    ),
    "javascript": LSPServerConfig(
        name="typescript-language-server",
        language="javascript",
        command=["typescript-language-server", "--stdio"],
        install_command="npm install -g typescript typescript-language-server",
        install_check="typescript-language-server",
        file_patterns=["*.js", "*.jsx", "*.mjs", "*.cjs"],
    ),
    "go": LSPServerConfig(
        name="gopls",
        language="go",
        command=["gopls", "serve"],
        install_command="go install golang.org/x/tools/gopls@latest",
        install_check="gopls",
        file_patterns=["*.go"],
        init_options={
            "usePlaceholders": True,
            "completeUnimported": True,
        },
        workspace_config={
            "gopls": {
                "analyses": {
                    "unusedparams": True,
                    "shadow": True,
                },
                "staticcheck": True,
            }
        },
    ),
    "rust": LSPServerConfig(
        name="rust-analyzer",
        language="rust",
        command=["rust-analyzer"],
        install_command="rustup component add rust-analyzer",
        install_check="rust-analyzer",
        file_patterns=["*.rs"],
        init_options={
            "cargo": {"autoreload": True},
            "procMacro": {"enable": True},
            "checkOnSave": {"command": "clippy"},
        },
    ),
}


def get_server_config(language: str) -> LSPServerConfig | None:
    """Get server configuration for a language.

    Args:
        language: Language identifier

    Returns:
        Server configuration or None
    """
    return SERVER_CONFIGS.get(language.lower())


def get_available_servers() -> list[str]:
    """Get list of installed language servers.

    Returns:
        List of language identifiers with installed servers
    """
    available = []
    for lang, config in SERVER_CONFIGS.items():
        if config.is_installed():
            available.append(lang)
    return available


def get_server_for_file(file_path: str | Path) -> LSPServerConfig | None:
    """Get appropriate server for a file.

    Args:
        file_path: Path to the file

    Returns:
        Server configuration or None
    """
    import fnmatch

    path = Path(file_path)
    filename = path.name

    for config in SERVER_CONFIGS.values():
        for pattern in config.file_patterns:
            if fnmatch.fnmatch(filename, pattern):
                if config.is_installed():
                    return config
                break

    return None


def install_server(language: str) -> tuple[bool, str]:
    """Install a language server.

    Args:
        language: Language identifier

    Returns:
        Tuple of (success, message)
    """
    config = get_server_config(language)
    if not config:
        return False, f"Unknown language: {language}"

    if config.is_installed():
        return True, f"{config.name} is already installed"

    if not config.install_command:
        return False, f"No install command for {config.name}"

    try:
        import shlex
        logger.info(f"Installing {config.name}: {config.install_command}")
        result = subprocess.run(
            shlex.split(config.install_command),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return True, f"Successfully installed {config.name}"
        else:
            return False, f"Failed to install {config.name}: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, f"Installation of {config.name} timed out"
    except Exception as e:
        return False, f"Error installing {config.name}: {e}"


def install_all_servers() -> dict[str, tuple[bool, str]]:
    """Install all language servers.

    Returns:
        Dictionary of language -> (success, message)
    """
    results = {}
    for language in SERVER_CONFIGS:
        results[language] = install_server(language)
    return results
