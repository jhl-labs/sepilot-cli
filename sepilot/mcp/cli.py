#!/usr/bin/env python3
"""CLI entry point for SE Pilot MCP Server"""

import sys

import click


@click.command()
@click.option(
    "--working-dir",
    default=None,
    help="Working directory for file operations (defaults to current directory)"
)
def main(working_dir):
    """Start SE Pilot MCP Server

    This server exposes SE Pilot's capabilities through the Model Context Protocol,
    allowing integration with Claude Desktop and other MCP clients.

    Example:
        python -m sepilot.mcp.cli --working-dir /path/to/project
    """
    try:
        import os

        from sepilot.mcp import run_mcp_server

        # Set working directory if specified
        if working_dir:
            os.chdir(working_dir)

        # Run MCP server
        run_mcp_server()

    except ImportError as e:
        if "mcp" in str(e):
            click.echo("Error: MCP SDK not installed", err=True)
            click.echo("Install with: pip install mcp", err=True)
            sys.exit(1)
        else:
            raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
