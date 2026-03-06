"""MCP Configuration Manager

Manages MCP server configurations and persists them to disk.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MCPAccessControl:
    """Access control configuration for an MCP server"""

    # Allow list - agents explicitly allowed (highest priority)
    allow: list[str] = field(default_factory=list)

    # Deny list - agents explicitly denied (second priority)
    deny: list[str] = field(default_factory=list)

    # Default behavior when agent is not in allow or deny list
    default_allow: bool = True

    def can_access(self, agent_name: str) -> bool:
        """Check if an agent can access this MCP server

        Priority:
        1. If in allow list -> ALLOW
        2. If in deny list -> DENY
        3. Use default_allow

        Args:
            agent_name: Name of the agent (e.g., "github", "git", "se")

        Returns:
            True if agent can access, False otherwise
        """
        # Normalize agent name (case-insensitive)
        agent_name = agent_name.lower()

        # Check special keywords
        if "all" in self.allow:
            return True
        if "all" in self.deny and agent_name not in self.allow:
            return False

        # Priority 1: Allow list
        if agent_name in self.allow:
            return True

        # Priority 2: Deny list
        if agent_name in self.deny:
            return False

        # Priority 3: Default behavior
        return self.default_allow

    def add_allow(self, agent_name: str):
        """Add agent to allow list"""
        agent_name = agent_name.lower()
        if agent_name not in self.allow:
            self.allow.append(agent_name)
        # Remove from deny list if present
        if agent_name in self.deny:
            self.deny.remove(agent_name)

    def add_deny(self, agent_name: str):
        """Add agent to deny list"""
        agent_name = agent_name.lower()
        if agent_name not in self.deny:
            self.deny.append(agent_name)
        # Remove from allow list if present (but allow takes precedence)
        # Actually, we should NOT remove from allow, because allow has higher priority

    def remove_allow(self, agent_name: str):
        """Remove agent from allow list"""
        agent_name = agent_name.lower()
        if agent_name in self.allow:
            self.allow.remove(agent_name)

    def remove_deny(self, agent_name: str):
        """Remove agent from deny list"""
        agent_name = agent_name.lower()
        if agent_name in self.deny:
            self.deny.remove(agent_name)

    def clear_allow(self):
        """Clear allow list"""
        self.allow.clear()

    def clear_deny(self):
        """Clear deny list"""
        self.deny.clear()


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server

    Supports both local (stdio) and remote (HTTP/SSE) servers.
    """

    name: str
    command: str | None = None  # For local stdio servers
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    access_control: MCPAccessControl = field(default_factory=MCPAccessControl)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Remote server support (new fields)
    url: str | None = None  # HTTP/SSE endpoint URL
    transport: str = "stdio"  # "stdio" | "http" | "sse"

    # OAuth support for remote servers
    oauth: dict[str, Any] | None = None  # {client_id, auth_url, token_url, scopes}

    def is_remote(self) -> bool:
        """Check if this is a remote server"""
        return self.transport in ("http", "sse") and self.url is not None

    def is_local(self) -> bool:
        """Check if this is a local (stdio) server"""
        return self.transport == "stdio" and self.command is not None

    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert access_control to dict
        data['access_control'] = asdict(self.access_control)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MCPServerConfig':
        """Create from dictionary"""
        # Extract access_control and create MCPAccessControl object
        ac_data = data.pop('access_control', {})
        access_control = MCPAccessControl(**ac_data)

        # Create MCPServerConfig with remaining data
        return cls(access_control=access_control, **data)


class MCPConfigManager:
    """Manager for MCP server configurations

    Handles loading, saving, and managing MCP server configurations.
    Configurations are stored in ~/.sepilot/mcp_config.json
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize MCP config manager

        Args:
            config_path: Path to config file (defaults to ~/.sepilot/mcp_config.json)
        """
        if config_path is None:
            config_dir = Path.home() / ".sepilot"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "mcp_config.json"

        self.config_path = Path(config_path)
        self.servers: dict[str, MCPServerConfig] = {}

        # Load existing configuration
        self.load()

    def load(self):
        """Load configuration from disk"""
        if not self.config_path.exists():
            # Create default configuration
            self._create_default_config()
            return

        try:
            with open(self.config_path, encoding='utf-8') as f:
                data = json.load(f)

            # Load servers
            self.servers = {}
            for name, server_data in data.get('servers', {}).items():
                self.servers[name] = MCPServerConfig.from_dict(server_data)

        except Exception as e:
            # Backup corrupted config before overwriting
            backup_path = self.config_path.with_suffix('.bak')
            try:
                import shutil
                shutil.copy2(self.config_path, backup_path)
                print(f"Warning: MCP config corrupted, backed up to {backup_path}")
            except Exception:
                pass
            print(f"Warning: Failed to load MCP config: {e}")
            self._create_default_config()

    def save(self):
        """Save configuration to disk using atomic write to prevent corruption"""
        try:
            data = {
                'servers': {
                    name: server.to_dict()
                    for name, server in self.servers.items()
                },
                'version': '1.0',
                'updated_at': datetime.now().isoformat()
            }

            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: write to temp file, then rename
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self.config_path.parent,
                suffix='.tmp',
                prefix='.mcp_config_'
            )
            try:
                with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.config_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            raise RuntimeError(f"Failed to save MCP config: {e}") from e

    def _create_default_config(self):
        """Create default configuration with example servers"""
        # Example: filesystem MCP server
        self.servers = {}
        self.save()

    # Server management methods

    def add_server(
        self,
        name: str,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        description: str = "",
        enabled: bool = True,
        url: str | None = None,
        transport: str = "stdio",
        oauth: dict[str, Any] | None = None,
    ) -> MCPServerConfig:
        """Add a new MCP server

        Args:
            name: Unique name for the server
            command: Command to execute (e.g., "npx", "python")
            args: Command arguments
            env: Environment variables
            description: Human-readable description
            enabled: Whether server is enabled

        Returns:
            Created MCPServerConfig

        Raises:
            ValueError: If server with name already exists
        """
        if name in self.servers:
            raise ValueError(f"Server '{name}' already exists")

        # Validate configuration
        if transport == "stdio" and not command:
            raise ValueError("Local (stdio) servers require a command")
        if transport in ("http", "sse") and not url:
            raise ValueError("Remote (http/sse) servers require a URL")

        server = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            description=description,
            enabled=enabled,
            url=url,
            transport=transport,
            oauth=oauth,
        )

        self.servers[name] = server
        self.save()

        return server

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server

        Args:
            name: Name of server to remove

        Returns:
            True if removed, False if not found
        """
        if name in self.servers:
            del self.servers[name]
            self.save()
            return True
        return False

    def get_server(self, name: str) -> MCPServerConfig | None:
        """Get server configuration by name

        Args:
            name: Server name

        Returns:
            MCPServerConfig if found, None otherwise
        """
        return self.servers.get(name)

    def list_servers(self, enabled_only: bool = False) -> list[MCPServerConfig]:
        """List all servers

        Args:
            enabled_only: If True, only return enabled servers

        Returns:
            List of MCPServerConfig objects
        """
        servers = list(self.servers.values())
        if enabled_only:
            servers = [s for s in servers if s.enabled]
        return servers

    def update_server(
        self,
        name: str,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        description: str | None = None,
        enabled: bool | None = None
    ) -> bool:
        """Update server configuration

        Args:
            name: Server name
            command: New command (optional)
            args: New args (optional)
            env: New environment variables (optional)
            description: New description (optional)
            enabled: New enabled status (optional)

        Returns:
            True if updated, False if server not found
        """
        server = self.get_server(name)
        if not server:
            return False

        if command is not None:
            server.command = command
        if args is not None:
            server.args = args
        if env is not None:
            server.env = env
        if description is not None:
            server.description = description
        if enabled is not None:
            server.enabled = enabled

        server.update_timestamp()
        self.save()

        return True

    # Access control methods

    def allow_agent(self, server_name: str, agent_name: str) -> bool:
        """Allow an agent to access a server

        Args:
            server_name: MCP server name
            agent_name: Agent name (e.g., "github", "git", "all")

        Returns:
            True if updated, False if server not found
        """
        server = self.get_server(server_name)
        if not server:
            return False

        server.access_control.add_allow(agent_name)
        server.update_timestamp()
        self.save()

        return True

    def deny_agent(self, server_name: str, agent_name: str) -> bool:
        """Deny an agent from accessing a server

        Args:
            server_name: MCP server name
            agent_name: Agent name (e.g., "github", "git", "all")

        Returns:
            True if updated, False if server not found
        """
        server = self.get_server(server_name)
        if not server:
            return False

        server.access_control.add_deny(agent_name)
        server.update_timestamp()
        self.save()

        return True

    def remove_allow(self, server_name: str, agent_name: str) -> bool:
        """Remove agent from allow list

        Args:
            server_name: MCP server name
            agent_name: Agent name

        Returns:
            True if updated, False if server not found
        """
        server = self.get_server(server_name)
        if not server:
            return False

        server.access_control.remove_allow(agent_name)
        server.update_timestamp()
        self.save()

        return True

    def remove_deny(self, server_name: str, agent_name: str) -> bool:
        """Remove agent from deny list

        Args:
            server_name: MCP server name
            agent_name: Agent name

        Returns:
            True if updated, False if server not found
        """
        server = self.get_server(server_name)
        if not server:
            return False

        server.access_control.remove_deny(agent_name)
        server.update_timestamp()
        self.save()

        return True

    def clear_access_control(self, server_name: str, clear_allow: bool = True, clear_deny: bool = True) -> bool:
        """Clear access control lists for a server

        Args:
            server_name: MCP server name
            clear_allow: Clear allow list
            clear_deny: Clear deny list

        Returns:
            True if updated, False if server not found
        """
        server = self.get_server(server_name)
        if not server:
            return False

        if clear_allow:
            server.access_control.clear_allow()
        if clear_deny:
            server.access_control.clear_deny()

        server.update_timestamp()
        self.save()

        return True

    def can_agent_access(self, server_name: str, agent_name: str) -> bool:
        """Check if an agent can access a server

        Args:
            server_name: MCP server name
            agent_name: Agent name

        Returns:
            True if agent can access, False otherwise
        """
        server = self.get_server(server_name)
        if not server:
            return False

        if not server.enabled:
            return False

        return server.access_control.can_access(agent_name)

    def get_allowed_servers(self, agent_name: str) -> list[MCPServerConfig]:
        """Get all servers that an agent can access

        Args:
            agent_name: Agent name

        Returns:
            List of accessible MCPServerConfig objects
        """
        return [
            server for server in self.list_servers(enabled_only=True)
            if server.access_control.can_access(agent_name)
        ]
