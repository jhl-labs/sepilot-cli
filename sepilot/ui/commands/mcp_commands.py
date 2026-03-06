"""MCP (Model Context Protocol) command handlers for Interactive Mode.

This module contains MCP server management command handlers extracted from interactive.py.
"""

import asyncio
import shlex
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def handle_mcp_command(
    input_text: str,
    mcp_config_manager_holder: dict,
    console: Console,
    session: Any = None
) -> None:
    """Handle /mcp command for MCP server management.

    Args:
        input_text: The raw input text from the user
        mcp_config_manager_holder: Dict with 'manager' key (lazily initialized)
        console: Rich console for output
        session: PromptSession for interactive prompts (optional)
    """
    try:
        # Lazy initialization of MCP config manager
        if mcp_config_manager_holder.get('manager') is None:
            from sepilot.mcp.config_manager import MCPConfigManager
            mcp_config_manager_holder['manager'] = MCPConfigManager()

        manager = mcp_config_manager_holder['manager']

        def _mcp_not_found(name: str):
            """Unified not-found handler with suggestions."""
            console.print(f"[red]❌ Server '{name}' not found[/red]")
            available = [s.name for s in manager.list_servers()]
            if available:
                try:
                    import difflib
                    suggestion = difflib.get_close_matches(name, available, n=1)
                    if suggestion:
                        console.print(f"[dim]Did you mean: {suggestion[0]}?[/dim]")
                except Exception:
                    pass
                console.print(f"[dim]Available: {', '.join(available)}[/dim]")
            else:
                console.print("[dim]Use '/mcp add <name>' to create one[/dim]")

        # Parse command
        input_text = input_text.strip()
        if input_text.lower().startswith('/mcp'):
            input_text = input_text[4:].strip()  # Remove "/mcp"

        parts = shlex.split(input_text) if input_text else []
        command = parts[0].lower() if len(parts) > 0 else ""
        arg1 = parts[1].lower() if len(parts) > 1 else ""
        arg2 = parts[2].lower() if len(parts) > 2 else ""
        arg3 = parts[3].lower() if len(parts) > 3 else ""

        def show_help():
            """Display MCP command help"""
            help_text = """
[bold cyan]🔌 MCP Server Management (Claude Code Style)[/bold cyan]

[bold yellow]📋 List & View:[/bold yellow]
  [cyan]/mcp[/cyan]                          List all MCP servers
  [cyan]/mcp list[/cyan]                     List all MCP servers with details
  [cyan]/mcp <name> show[/cyan]              Show server details and access control
  [cyan]/mcp <name> tools[/cyan]             List tools from MCP server (connects to server)

[bold yellow]➕ Add & Configure:[/bold yellow]
  [cyan]/mcp add <name>[/cyan]               Add new MCP server (interactive)
  [cyan]/mcp remove <name>[/cyan]            Remove MCP server
  [cyan]/mcp edit <name>[/cyan]              Edit server configuration (interactive)
  [cyan]/mcp enable <name>[/cyan]            Enable MCP server
  [cyan]/mcp disable <name>[/cyan]           Disable MCP server

[bold yellow]🔐 Access Control:[/bold yellow]
  [cyan]/mcp <name> allow <agent>[/cyan]     Allow specific agent (github, git, se, etc.)
  [cyan]/mcp <name> deny <agent>[/cyan]      Deny specific agent
  [cyan]/mcp <name> allow all[/cyan]         Allow all agents to use this server
  [cyan]/mcp <name> deny all[/cyan]          Deny all agents (except explicitly allowed)
  [cyan]/mcp <name> clear allow[/cyan]       Clear allow list
  [cyan]/mcp <name> clear deny[/cyan]        Clear deny list

[bold yellow]📚 Examples:[/bold yellow]
  [cyan]# Allow GitHub agent to use filesystem MCP[/cyan]
  /mcp filesystem allow github

  [cyan]# Deny all agents except github and git[/cyan]
  /mcp filesystem deny all
  /mcp filesystem allow github
  /mcp filesystem allow git

  [cyan]# Add new MCP server[/cyan]
  /mcp add weather
  [dim]→ Prompts for command, args, description[/dim]

[bold yellow]🎯 Access Control Priority:[/bold yellow]
  1. [green]Allow list[/green] (highest priority) - If agent is in allow list → ALLOW
  2. [red]Deny list[/red] (second priority) - If agent is in deny list → DENY
  3. [yellow]Default[/yellow] (lowest priority) - If not in either list → ALLOW

  [dim]Example: If you add "deny all" and "allow github", only github can access.[/dim]

[bold yellow]💡 Tips:[/bold yellow]
  • MCP servers are stored in ~/.sepilot/mcp_config.json
  • Access control is enforced when agents request MCP tools
  • Disabled servers are not accessible by any agent
  • Use "all" keyword to apply to all agents at once

[dim]Type /mcp <command> for help, or use any command above[/dim]
            """
            console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))

        # Route to appropriate handler
        if not command or command == "help":
            show_help()
            return

        elif command == "list":
            _handle_mcp_list(manager, console)
            return

        elif command == "add":
            _handle_mcp_add(arg1, parts, manager, console, session)
            return

        elif command == "remove":
            _handle_mcp_remove(arg1, manager, console, session, _mcp_not_found)
            return

        elif command == "enable":
            _handle_mcp_enable(arg1, manager, console, _mcp_not_found)
            return

        elif command == "disable":
            _handle_mcp_disable(arg1, manager, console, _mcp_not_found)
            return

        elif command == "edit":
            _handle_mcp_edit(arg1, manager, console, session, _mcp_not_found)
            return

        # Server-specific commands: /mcp <server_name> <action> <args>
        else:
            _handle_mcp_server_action(
                command, arg1, arg2, arg3,
                manager, console, _mcp_not_found
            )
            return

    except ImportError as e:
        console.print(f"[red]❌ Failed to import MCP modules: {e}[/red]")
        console.print("[dim]MCP functionality may not be available[/dim]")
    except Exception as e:
        console.print(f"[red]❌ MCP command failed: {e}[/red]")
        import traceback
        if console.is_terminal:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def _handle_mcp_list(manager: Any, console: Console) -> None:
    """Handle /mcp list command."""
    servers = manager.list_servers()

    if not servers:
        console.print("[yellow]No MCP servers configured yet[/yellow]")
        console.print("[dim]Use '/mcp add <name>' to add a new server[/dim]")
        return

    from rich.table import Table

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Command", style="dim", max_width=30)
    table.add_column("Allow", style="green", width=10)
    table.add_column("Deny", style="red", width=10)
    table.add_column("Default", style="yellow", width=12)

    for server in servers:
        # Status
        status = "[green]✓ Enabled[/green]" if server.enabled else "[red]✗ Disabled[/red]"

        # Command
        cmd = f"{server.command}"
        if server.args:
            cmd += f" {' '.join(server.args[:2])}"
            if len(server.args) > 2:
                cmd += "..."

        # Access control summary
        ac = server.access_control
        allow_display = "all" if "all" in ac.allow else (", ".join(ac.allow) if ac.allow else "—")
        deny_display = "all" if "all" in ac.deny else (", ".join(ac.deny) if ac.deny else "—")
        default_policy = "allow" if ac.default_allow else "deny"
        table.add_row(server.name, status, cmd, allow_display, deny_display, default_policy)

    console.print(table)
    console.print(f"\n[dim]Total: {len(servers)} servers | Use '/mcp <name> show' for details[/dim]")


def _handle_mcp_add(
    server_name: str,
    parts: list,
    manager: Any,
    console: Console,
    session: Any
) -> None:
    """Handle /mcp add command."""
    if not server_name:
        console.print("[yellow]⚠️  Please provide a server name[/yellow]")
        console.print("[dim]Usage: /mcp add <name> [command] [args...] [--desc description][/dim]")
        return

    server_name = server_name.lower()

    # Check if already exists
    if manager.get_server(server_name):
        console.print(f"[red]❌ Server '{server_name}' already exists[/red]")
        console.print(f"[dim]Use '/mcp edit {server_name}' to modify it[/dim]")
        return

    # Non-interactive: if command is supplied, use remaining tokens
    provided_command = parts[2] if len(parts) > 2 else None
    description = ""
    args_list = []

    if provided_command and not provided_command.startswith("--"):
        cmd = provided_command
        # Remaining tokens as args/desc
        extra_tokens = parts[3:]
        tokens_iter = iter(extra_tokens)
        for tok in tokens_iter:
            if tok == "--desc":
                try:
                    description = next(tokens_iter)
                except StopIteration:
                    console.print("[yellow]⚠️  --desc provided without value[/yellow]")
            else:
                args_list.append(tok)
        command_input = cmd
    else:
        # Interactive prompts
        console.print(f"[bold cyan]Adding new MCP server: {server_name}[/bold cyan]\n")
        try:
            if session:
                command_input = session.prompt("Command (e.g., npx, python): ").strip()
            else:
                command_input = input("Command (e.g., npx, python): ").strip()
            if not command_input:
                console.print("[red]❌ Command cannot be empty[/red]")
                return
            if session:
                args_input = session.prompt("Arguments (space-separated, optional): ").strip()
            else:
                args_input = input("Arguments (space-separated, optional): ").strip()
            args_list = args_input.split() if args_input else []
            if session:
                desc_input = session.prompt("Description (optional): ").strip()
            else:
                desc_input = input("Description (optional): ").strip()
            description = desc_input
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled[/yellow]")
            return

    try:
        manager.add_server(
            name=server_name,
            command=command_input,
            args=args_list,
            description=description,
            enabled=True
        )
        console.print(f"\n[bold green]✅ MCP server '{server_name}' added successfully![/bold green]")
        console.print(f"[dim]Command: {command_input} {' '.join(args_list)}[/dim]")
        if description:
            console.print(f"[dim]Description: {description}[/dim]")
        console.print(f"[dim]Use '/mcp {server_name} allow <agent>' to set access control[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Failed to add server: {e}[/red]")


def _handle_mcp_remove(
    server_name: str,
    manager: Any,
    console: Console,
    session: Any,
    not_found_handler: callable
) -> None:
    """Handle /mcp remove command."""
    if not server_name:
        console.print("[yellow]⚠️  Please provide a server name[/yellow]")
        console.print("[dim]Usage: /mcp remove <name>[/dim]")
        return

    server_name = server_name.lower()

    if not manager.get_server(server_name):
        not_found_handler(server_name)
        return

    # Confirm removal
    try:
        if session:
            confirm = session.prompt(f"Remove '{server_name}'? (yes/no): ").strip().lower()
        else:
            confirm = input(f"Remove '{server_name}'? (yes/no): ").strip().lower()

        if confirm in ('yes', 'y'):
            manager.remove_server(server_name)
            console.print(f"[bold green]✅ Server '{server_name}' removed[/bold green]")
        else:
            console.print("[yellow]Cancelled[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")


def _handle_mcp_enable(
    server_name: str,
    manager: Any,
    console: Console,
    not_found_handler: callable
) -> None:
    """Handle /mcp enable command."""
    if not server_name:
        console.print("[yellow]⚠️  Please provide a server name[/yellow]")
        return

    server_name = server_name.lower()
    if manager.update_server(server_name, enabled=True):
        console.print(f"[bold green]✅ Server '{server_name}' enabled[/bold green]")
    else:
        not_found_handler(server_name)


def _handle_mcp_disable(
    server_name: str,
    manager: Any,
    console: Console,
    not_found_handler: callable
) -> None:
    """Handle /mcp disable command."""
    if not server_name:
        console.print("[yellow]⚠️  Please provide a server name[/yellow]")
        return

    server_name = server_name.lower()
    if manager.update_server(server_name, enabled=False):
        console.print(f"[bold yellow]⚠️  Server '{server_name}' disabled[/bold yellow]")
    else:
        not_found_handler(server_name)


def _handle_mcp_edit(
    server_name: str,
    manager: Any,
    console: Console,
    session: Any,
    not_found_handler: callable
) -> None:
    """Handle /mcp edit command."""
    if not server_name:
        console.print("[yellow]⚠️  Please provide a server name[/yellow]")
        return

    server_name = server_name.lower()
    server = manager.get_server(server_name)

    if not server:
        not_found_handler(server_name)
        return

    # Interactive edit
    console.print(f"[bold cyan]Editing MCP server: {server_name}[/bold cyan]")
    console.print("[dim]Press Enter to keep current value[/dim]\n")

    try:
        # Edit command
        if session:
            new_command = session.prompt(f"Command [{server.command}]: ").strip()
        else:
            new_command = input(f"Command [{server.command}]: ").strip()

        # Edit args
        current_args = ' '.join(server.args) if server.args else ''
        if session:
            new_args = session.prompt(f"Arguments [{current_args}]: ").strip()
        else:
            new_args = input(f"Arguments [{current_args}]: ").strip()

        # Edit description
        if session:
            new_desc = session.prompt(f"Description [{server.description}]: ").strip()
        else:
            new_desc = input(f"Description [{server.description}]: ").strip()

        # Apply updates
        updates = {}
        if new_command:
            updates['command'] = new_command
        if new_args is not None:  # Allow empty string
            updates['args'] = new_args.split() if new_args else []
        if new_desc is not None:
            updates['description'] = new_desc

        if updates:
            manager.update_server(server_name, **updates)
            console.print(f"\n[bold green]✅ Server '{server_name}' updated[/bold green]")
        else:
            console.print("\n[yellow]No changes made[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")


def _handle_mcp_server_action(
    server_name: str,
    action: str,
    agent_name: str,
    extra: str,
    manager: Any,
    console: Console,
    not_found_handler: callable
) -> None:
    """Handle server-specific commands: /mcp <server_name> <action> <args>"""
    server_name = server_name.lower()

    # Check if server exists
    server = manager.get_server(server_name)
    if not server:
        not_found_handler(server_name)
        return

    if action == "show":
        # Show server details
        console.print(f"[bold cyan]MCP Server: {server.name}[/bold cyan]\n")
        console.print(f"[cyan]Status:[/cyan] {'✓ Enabled' if server.enabled else '✗ Disabled'}")
        console.print(f"[cyan]Command:[/cyan] {server.command}")
        if server.args:
            console.print(f"[cyan]Arguments:[/cyan] {' '.join(server.args)}")
        if server.description:
            console.print(f"[cyan]Description:[/cyan] {server.description}")
        if server.env:
            console.print(f"[cyan]Environment:[/cyan] {len(server.env)} variables")

        # Access control
        ac = server.access_control
        console.print("\n[bold cyan]Access Control:[/bold cyan]")

        if ac.allow:
            console.print(f"  [green]Allow:[/green] {', '.join(ac.allow)}")
        else:
            console.print("  [dim]Allow list: empty[/dim]")

        if ac.deny:
            console.print(f"  [red]Deny:[/red] {', '.join(ac.deny)}")
        else:
            console.print("  [dim]Deny list: empty[/dim]")

        console.print(f"  [yellow]Default:[/yellow] {'Allow all' if ac.default_allow else 'Deny all'}")

        # Show which agents can access
        console.print("\n[bold cyan]Agent Access:[/bold cyan]")
        common_agents = ['github', 'git', 'se', 'wiki', 'k8s', 'test']
        for agent in common_agents:
            can_access = ac.can_access(agent)
            icon = "[green]✓[/green]" if can_access else "[red]✗[/red]"
            console.print(f"  {icon} {agent}")

        console.print(f"\n[dim]Created: {server.created_at}[/dim]")
        console.print(f"[dim]Updated: {server.updated_at}[/dim]")

    elif action == "tools":
        # List tools from server
        _handle_mcp_tools(server_name, server, console)

    elif action == "allow":
        # Allow agent
        if not agent_name:
            console.print("[yellow]⚠️  Please specify an agent name or 'all'[/yellow]")
            console.print(f"[dim]Usage: /mcp {server_name} allow <agent>[/dim]")
            console.print("[dim]Examples: github, git, se, wiki, k8s, all[/dim]")
            return

        manager.allow_agent(server_name, agent_name)
        console.print(f"[bold green]✅ Agent '{agent_name}' can now access '{server_name}'[/bold green]")

    elif action == "deny":
        # Deny agent
        if not agent_name:
            console.print("[yellow]⚠️  Please specify an agent name or 'all'[/yellow]")
            console.print(f"[dim]Usage: /mcp {server_name} deny <agent>[/dim]")
            return

        manager.deny_agent(server_name, agent_name)
        console.print(f"[bold red]🚫 Agent '{agent_name}' cannot access '{server_name}'[/bold red]")

    elif action == "clear":
        # Clear allow or deny list
        if agent_name == "allow":
            manager.clear_access_control(server_name, clear_allow=True, clear_deny=False)
            console.print(f"[bold green]✅ Allow list cleared for '{server_name}'[/bold green]")
        elif agent_name == "deny":
            manager.clear_access_control(server_name, clear_allow=False, clear_deny=True)
            console.print(f"[bold yellow]⚠️  Deny list cleared for '{server_name}'[/bold yellow]")
        else:
            console.print("[yellow]⚠️  Please specify 'allow' or 'deny'[/yellow]")
            console.print(f"[dim]Usage: /mcp {server_name} clear <allow|deny>[/dim]")

    else:
        console.print(f"[yellow]⚠️  Unknown action: '{action}'[/yellow]")
        console.print(f"[dim]Use '/mcp {server_name} show' to view server details[/dim]")


def _handle_mcp_tools(server_name: str, server: Any, console: Console) -> None:
    """Handle /mcp <name> tools command - list tools from MCP server."""
    from sepilot.mcp.client import MCPClient, MCPProtocolError

    if not server.enabled:
        console.print(f"[red]❌ Server '{server_name}' is disabled[/red]")
        console.print(f"[dim]Use '/mcp enable {server_name}' to enable it[/dim]")
        return

    console.print(f"[cyan]🔌 Connecting to MCP server '{server_name}'...[/cyan]")

    async def fetch_tools():
        client = MCPClient(agent_name="interactive", config_manager=None)
        # Manually set the server config since we have it
        from sepilot.mcp.config_manager import MCPConfigManager
        client.config_manager = MCPConfigManager()

        try:
            tools = await client.list_tools_from_server(server_name)
            return tools
        finally:
            await client.stop_all_servers()

    try:
        # Run async function
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, fetch_tools())
                    tools = future.result(timeout=60)
            else:
                tools = loop.run_until_complete(fetch_tools())
        except RuntimeError:
            tools = asyncio.run(fetch_tools())

        if not tools:
            console.print(f"[yellow]No tools available from '{server_name}'[/yellow]")
            console.print("[dim]The server may not provide any tools, or connection failed[/dim]")
            return

        # Display tools
        console.print(f"\n[bold cyan]🔧 Tools from '{server_name}' ({len(tools)} total)[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Tool Name", style="cyan", width=25)
        table.add_column("Description", width=50)
        table.add_column("Parameters", style="dim", width=20)

        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            if len(desc) > 50:
                desc = desc[:47] + "..."

            # Count parameters
            input_schema = tool.get("inputSchema", {})
            props = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            param_info = f"{len(props)} params"
            if required:
                param_info += f" ({len(required)} req)"

            table.add_row(name, desc, param_info)

        console.print(table)

        # Show detailed info for first few tools
        console.print("\n[dim]Use these tools via agent commands or MCP integration[/dim]")

    except MCPProtocolError as e:
        console.print(f"[red]❌ MCP Protocol Error: {e}[/red]")
        console.print("[dim]Check that the server command is correct and the server is running[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Failed to list tools: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


__all__ = ['handle_mcp_command']
