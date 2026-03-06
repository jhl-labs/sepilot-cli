"""Model configuration command handlers for Interactive Mode.

This module contains model configuration related command handlers extracted from interactive.py.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel


def handle_model_command(
    input_text: str,
    model_profile_manager: Any,
    agent: Any,
    console: Console,
    create_llm_func: Callable,
    agent_factory: Callable | None = None
) -> Any | None:
    """Handle /model command for model configuration.

    Args:
        input_text: The raw input text from the user
        model_profile_manager: ModelProfileManager instance
        agent: Agent instance (can be None)
        console: Rich console for output
        create_llm_func: Function to create LLM from config
        agent_factory: Optional factory to create a new agent (for deferred init)

    Returns:
        A new agent instance if one was created via agent_factory, else None.
    """
    # Remove /model prefix
    input_text = input_text.strip()
    if input_text.lower().startswith('/model'):
        input_text = input_text[6:].strip()

    # Parse command and arguments
    parts = input_text.split(maxsplit=2) if input_text else []
    command = parts[0].lower() if len(parts) > 0 else ""

    # Show help for empty command or "help"
    def show_help():
        """Display model command help"""
        help_text = """
[bold cyan]🤖 Model Configuration Commands[/bold cyan]

[bold yellow]📊 View Configuration:[/bold yellow]
  [cyan]/model[/cyan]                       Show current configuration
  [cyan]/model show[/cyan]                  Show current configuration

[bold yellow]⚙️ Set Parameters:[/bold yellow]
  [cyan]/model set <param> <value>[/cyan]   Set a configuration parameter
                                Examples:
                                  /model set base_url https://api.openai.com/v1
                                  /model set model gpt-4
                                  /model set temperature 0.7
                                  /model set top_k 50
                                  /model set max_tokens 4096

[bold yellow]📋 Custom Headers:[/bold yellow]
  [cyan]/model header <key> <value>[/cyan]  Set a custom HTTP header
  [cyan]/model header remove <key>[/cyan]   Remove a custom HTTP header
                                Example:
                                  /model header X-Custom-Auth mytoken123

[bold yellow]💾 Profile Management:[/bold yellow]
  [cyan]/model profile list[/cyan]          List all saved profiles
  [cyan]/model profile save <name>[/cyan]   Save current config as a profile
  [cyan]/model profile load <name>[/cyan]   Load a profile (auto-applies)
  [cyan]/model profile delete <name>[/cyan] Delete a profile
  [cyan]/model profile show <name>[/cyan]   Show profile details
  [cyan]/model profile default[/cyan]       Show current default profile
  [cyan]/model profile default set <n>[/cyan] Set default (auto-loads on start)
  [cyan]/model profile default clear[/cyan] Clear default profile

[bold yellow]🔄 Apply & Reset:[/bold yellow]
  [cyan]/model apply[/cyan]                 Apply current config to agents
  [cyan]/model reset[/cyan]                 Reset to environment defaults

[bold yellow]📝 Available Parameters:[/bold yellow]
  • base_url      - LLM API base URL
  • model         - Model name
  • api_key       - API key/token
  • temperature   - Sampling temperature (0.0-2.0)
  • top_k         - Top-K sampling
  • top_p         - Top-P sampling
  • max_tokens    - Maximum tokens

[bold yellow]💡 Tips:[/bold yellow]
  • Profiles are stored in ~/.sepilot/profiles/
  • Environment variables (.env) are used as defaults
  • Changes take effect after /model apply

[dim]Type /model <command> for help, or use any command above[/dim]
        """
        console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))

    # Route to appropriate handler
    if not command or command == "help":
        show_help()
        return None

    elif command == "show":
        _handle_model_show(model_profile_manager, agent, console)
        return None

    elif command == "set":
        _handle_model_set(parts, model_profile_manager, console)
        return None

    elif command == "header":
        _handle_model_header(parts, model_profile_manager, console)
        return None

    elif command == "reset":
        model_profile_manager.reset_to_defaults()
        console.print("[green]✅ Configuration reset to environment defaults[/green]")
        console.print("[dim]Run '/model apply' to apply changes to agents[/dim]")
        return None

    elif command == "profile":
        return _handle_model_profile(
            input_text, parts, model_profile_manager, agent, console,
            create_llm_func, agent_factory=agent_factory
        )

    elif command == "apply":
        return _handle_model_apply(
            model_profile_manager, agent, console,
            create_llm_func, agent_factory=agent_factory
        )

    else:
        # Unknown command - show help
        console.print(f"[yellow]⚠️  Unknown model command: '{command}'[/yellow]\n")
        show_help()
        return None


def _handle_model_show(model_profile_manager: Any, agent: Any, console: Console) -> None:
    """Handle /model show command."""
    config = model_profile_manager.get_current_config()
    config_dict = config.to_dict()

    display_data: dict[str, Any] = {}

    def set_default(key: str, value: Any):
        if value in (None, "", []):
            return
        if key not in display_data:
            display_data[key] = value

    # Include active agent settings (defaults from runtime config)
    settings = getattr(agent, "settings", None) if agent else None
    if settings:
        set_default("model", getattr(settings, "model", None))
        # Settings store Ollama base URL; OpenAI base URL usually set via env/profile
        set_default("base_url", getattr(settings, "ollama_base_url", None))
        set_default("temperature", getattr(settings, "temperature", None))
        set_default("max_tokens", getattr(settings, "max_tokens", None))
        set_default("api_key", getattr(settings, "openai_api_key", None))

    # Include data from the currently active LLM (if already constructed)
    agent_llm = getattr(agent, "llm", None) if agent else None
    if agent_llm is not None:
        llm_model = getattr(agent_llm, "model_name", None) or getattr(agent_llm, "model", None)
        set_default("model", llm_model)

        llm_base = (
            getattr(agent_llm, "openai_api_base", None) or
            getattr(agent_llm, "base_url", None) or
            getattr(agent_llm, "api_base", None)
        )
        set_default("base_url", llm_base)

        set_default("temperature", getattr(agent_llm, "temperature", None))
        set_default("top_p", getattr(agent_llm, "top_p", None))
        set_default("top_k", getattr(agent_llm, "top_k", None))
        set_default("max_tokens", getattr(agent_llm, "max_tokens", None))

    # Fall back to environment base URL if available
    env_base = (
        os.getenv("OPENAI_API_BASE") or
        os.getenv("LLM_BASE_URL") or
        os.getenv("API_BASE_URL") or
        os.getenv("OLLAMA_BASE_URL")
    )
    set_default("base_url", env_base)

    # Merge with explicit overrides from the profile manager (highest priority)
    for key, value in config_dict.items():
        if value not in (None, "", []):
            display_data[key] = value

    if not display_data:
        console.print("[yellow]⚠️  No configuration detected[/yellow]")
        console.print("[dim]Use '/model set' or environment variables to configure defaults[/dim]")
        return

    from rich.table import Table
    table = Table(title="Current Model Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="yellow", width=20)
    table.add_column("Value", style="white", width=60)

    preferred_order = [
        "model", "base_url", "temperature", "top_p",
        "top_k", "max_tokens", "custom_headers", "api_key"
    ]

    def add_row(key: str, value):
        if key == "custom_headers" and isinstance(value, dict):
            if value:
                headers_str = "\n".join([f"{k}: {v}" for k, v in value.items()])
                table.add_row(key, headers_str)
            else:
                table.add_row(key, "—")
        elif key == "api_key":
            if value:
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                table.add_row(key, masked)
            else:
                table.add_row(key, "—")
        else:
            display_value = "—" if value in (None, "", []) else str(value)
            table.add_row(key, display_value)

    seen = set()
    for key in preferred_order:
        if key in display_data:
            add_row(key, display_data[key])
            seen.add(key)

    for key, value in display_data.items():
        if key in seen:
            continue
        if key == "custom_headers" and isinstance(value, dict):
            add_row(key, value)
        else:
            add_row(key, value)
        seen.add(key)

    console.print(table)


def _handle_model_set(parts: list, model_profile_manager: Any, console: Console) -> None:
    """Handle /model set command."""
    if len(parts) < 3:
        console.print("[yellow]⚠️  Usage: /model set <param> <value>[/yellow]")
        console.print("[dim]Example: /model set temperature 0.7[/dim]")
        return

    param = parts[1].lower()
    value = parts[2]

    success = model_profile_manager.set_parameter(param, value)
    if success:
        console.print(f"[green]✅ Set {param} = {value}[/green]")
        console.print("[dim]Run '/model apply' to apply changes to agents[/dim]")
    else:
        console.print(f"[red]❌ Invalid parameter: {param}[/red]")
        console.print("[dim]Valid parameters: base_url, model, api_key, temperature, top_k, top_p, max_tokens[/dim]")


def _handle_model_header(parts: list, model_profile_manager: Any, console: Console) -> None:
    """Handle /model header command."""
    if len(parts) < 2:
        console.print("[yellow]⚠️  Usage: /model header <key> <value> OR /model header remove <key>[/yellow]")
        return

    subcommand = parts[1].lower()
    if subcommand == "remove":
        if len(parts) < 3:
            console.print("[yellow]⚠️  Usage: /model header remove <key>[/yellow]")
            return
        key = parts[2]
        success = model_profile_manager.remove_custom_header(key)
        if success:
            console.print(f"[green]✅ Removed header: {key}[/green]")
        else:
            console.print(f"[yellow]⚠️  Header not found: {key}[/yellow]")
    else:
        if len(parts) < 3:
            console.print("[yellow]⚠️  Usage: /model header <key> <value>[/yellow]")
            return
        key = parts[1]
        value = parts[2]
        model_profile_manager.set_custom_header(key, value)
        console.print(f"[green]✅ Set custom header: {key} = {value}[/green]")
        console.print("[dim]Run '/model apply' to apply changes to agents[/dim]")


def _handle_model_profile(
    input_text: str,
    parts: list,
    model_profile_manager: Any,
    agent: Any,
    console: Console,
    create_llm_func: Callable,
    agent_factory: Callable | None = None
) -> Any | None:
    """Handle /model profile commands.

    Returns:
        A new agent instance if one was created via agent_factory, else None.
    """
    if len(parts) < 2:
        console.print("[yellow]⚠️  Usage: /model profile <list|save|load|delete|show|default> [name][/yellow]")
        return None

    subcommand = parts[1].lower()

    if subcommand == "list":
        profiles = model_profile_manager.list_profiles()
        if not profiles:
            console.print("[yellow]⚠️  No profiles found[/yellow]")
            console.print(f"[dim]Profiles are stored in: {model_profile_manager.profile_dir}[/dim]")
            return None

        from rich.table import Table
        table = Table(title="Saved Model Profiles", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Profile Name", style="cyan", width=30)

        for i, profile in enumerate(profiles, 1):
            table.add_row(str(i), profile)

        console.print(table)
        console.print(f"\n[dim]Total: {len(profiles)} profiles[/dim]")

    elif subcommand == "save":
        if len(parts) < 3:
            console.print("[yellow]⚠️  Usage: /model profile save <name>[/yellow]")
            return None

        name = parts[2]
        success = model_profile_manager.save_profile(name)
        if success:
            console.print(f"[green]✅ Profile saved: {name}[/green]")
            console.print(f"[dim]Saved to: {model_profile_manager.profile_dir / f'{name}.json'}[/dim]")
        else:
            console.print(f"[red]❌ Failed to save profile: {name}[/red]")

    elif subcommand == "load":
        if len(parts) < 3:
            console.print("[yellow]⚠️  Usage: /model profile load <name>[/yellow]")
            return None

        name = parts[2]
        success = model_profile_manager.load_profile(name)
        if success:
            console.print(f"[green]✅ Profile loaded: {name}[/green]")
            # Auto-apply to agent (no separate /model apply needed)
            if agent:
                config = model_profile_manager.get_current_config()
                apply_ok = apply_model_config_to_agent(agent, config, console, create_llm_func)
                if apply_ok:
                    console.print("[green]✅ Configuration applied to agent[/green]")
                else:
                    console.print("[yellow]⚠️  Configuration applied with warnings[/yellow]")
            else:
                console.print("[dim]No agent available yet. Config will be used when agent is created.[/dim]")
        else:
            console.print(f"[red]❌ Failed to load profile: {name}[/red]")
            console.print("[dim]Use '/model profile list' to see available profiles[/dim]")

    elif subcommand == "delete":
        if len(parts) < 3:
            console.print("[yellow]⚠️  Usage: /model profile delete <name>[/yellow]")
            return None

        name = parts[2]
        # Confirm deletion
        from rich.prompt import Confirm
        if Confirm.ask(f"Delete profile '{name}'?", default=False):
            success = model_profile_manager.delete_profile(name)
            if success:
                console.print(f"[green]✅ Profile deleted: {name}[/green]")
            else:
                console.print(f"[red]❌ Failed to delete profile: {name}[/red]")
        else:
            console.print("[yellow]Deletion cancelled[/yellow]")

    elif subcommand == "show":
        if len(parts) < 3:
            console.print("[yellow]⚠️  Usage: /model profile show <name>[/yellow]")
            return None

        name = parts[2]
        profile_info = model_profile_manager.get_profile_info(name)
        if not profile_info:
            console.print(f"[red]❌ Profile not found: {name}[/red]")
            return None

        from rich.table import Table
        table = Table(title=f"Profile: {name}", show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="yellow", width=20)
        table.add_column("Value", style="white", width=60)

        for key, value in profile_info.items():
            if key == "custom_headers" and isinstance(value, dict):
                if value:
                    headers_str = "\n".join([f"{k}: {v}" for k, v in value.items()])
                    table.add_row(key, headers_str)
            elif key == "api_key":
                # Mask API key
                if value:
                    masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                    table.add_row(key, masked)
            else:
                table.add_row(key, str(value))

        console.print(table)

    elif subcommand == "default":
        # Re-parse without maxsplit for default subcommand
        default_parts = input_text.split()
        # default_parts: ['profile', 'default', ...]

        if len(default_parts) < 3:
            # Show current default
            current = model_profile_manager.get_default_profile()
            if current:
                console.print(f"[cyan]Default profile: [bold]{current}[/bold][/cyan]")
            else:
                console.print("[dim]No default profile set[/dim]")
            return None

        action = default_parts[2].lower()

        if action == "set":
            if len(default_parts) < 4:
                console.print("[yellow]⚠️  Usage: /model profile default set <name>[/yellow]")
                return None
            name = default_parts[3]
            success = model_profile_manager.set_default_profile(name)
            if success:
                console.print(f"[green]✅ Default profile set: {name}[/green]")
                console.print("[dim]This profile will auto-load on startup[/dim]")
            else:
                console.print(f"[red]❌ Profile not found: {name}[/red]")
                console.print("[dim]Use '/model profile list' to see available profiles[/dim]")

        elif action == "clear":
            model_profile_manager.clear_default_profile()
            console.print("[green]✅ Default profile cleared[/green]")

        else:
            console.print(f"[yellow]⚠️  Unknown default action: {action}[/yellow]")
            console.print("[dim]Usage: /model profile default [set <name>|clear][/dim]")

    else:
        console.print(f"[yellow]⚠️  Unknown profile command: {subcommand}[/yellow]")
        console.print("[dim]Valid commands: list, save, load, delete, show, default[/dim]")

    return None


def _handle_model_apply(
    model_profile_manager: Any,
    agent: Any,
    console: Console,
    create_llm_func: Callable,
    agent_factory: Callable | None = None
) -> Any | None:
    """Handle /model apply command.

    Returns:
        A new agent instance if one was created via agent_factory, else None.
    """
    console.print("[cyan]🔄 Applying model configuration to agents...[/cyan]")

    if not agent:
        if agent_factory:
            console.print("[cyan]🔄 에이전트를 생성합니다...[/cyan]")
            try:
                config = model_profile_manager.get_current_config()
                new_agent = agent_factory(config)
                if hasattr(model_profile_manager, "clear_dirty_parameters"):
                    model_profile_manager.clear_dirty_parameters()
                console.print("[green]✅ 에이전트가 성공적으로 초기화되었습니다![/green]")
                return new_agent
            except Exception as e:
                console.print(f"[red]❌ 에이전트 생성 실패: {e}[/red]")
                return None
        else:
            console.print("[yellow]⚠️  No agent available to apply configuration[/yellow]")
            return None

    try:
        config = model_profile_manager.get_current_config()
        changed_fields = None
        if hasattr(model_profile_manager, "get_dirty_parameters"):
            changed_fields = model_profile_manager.get_dirty_parameters()

        # Apply to current agent's LLM
        success = apply_model_config_to_agent(
            agent,
            config,
            console,
            create_llm_func,
            changed_fields=changed_fields,
        )

        if success:
            if hasattr(model_profile_manager, "clear_dirty_parameters"):
                model_profile_manager.clear_dirty_parameters()
            console.print("[green]✅ Configuration applied successfully![/green]")
            console.print("[dim]New LLM settings are now active[/dim]")
        else:
            console.print("[yellow]⚠️  Configuration applied with warnings (check logs)[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Failed to apply configuration: {e}[/red]")

    return None


def apply_model_config_to_agent(
    agent: Any,
    config: Any,
    console: Console,
    create_llm_func: Callable,
    changed_fields: set[str] | None = None,
) -> bool:
    """Apply model configuration to an agent's LLM.

    Args:
        agent: Agent instance to update
        config: ModelConfig instance
        console: Rich console for output
        create_llm_func: Function to create LLM from config

    Returns:
        True if applied successfully
    """
    try:
        # Recreate only when connection/provider level fields changed.
        if changed_fields is not None:
            needs_recreation = bool({"base_url", "model", "api_key", "custom_headers"} & changed_fields)
        else:
            needs_recreation = any([
                config.base_url is not None,
                config.model is not None,
                config.api_key is not None,
                config.custom_headers
            ])

        # Build updated settings for propagation
        updated_settings = _build_updated_settings(agent, config, console)


        # If agent has no LLM yet (lazy_llm=True), create one from config
        if not hasattr(agent, 'llm') or agent.llm is None:
            if needs_recreation:
                console.print("[cyan]🔧 Creating LLM from profile configuration...[/cyan]")
                new_llm = create_llm_func(config)
                if new_llm:
                    if hasattr(agent, 'update_llm'):
                        agent.update_llm(new_llm, updated_settings)
                    else:
                        agent.llm = new_llm
                        if hasattr(agent, 'llm_with_tools') and hasattr(agent, 'langchain_tools'):
                            agent.llm_with_tools = new_llm.bind_tools(agent.langchain_tools)
                    console.print("[green]  ✅ LLM initialized successfully[/green]")
                    return True
                else:
                    console.print("[yellow]⚠️  Failed to create LLM (check configuration)[/yellow]")
                    return False
            else:
                console.print("[yellow]⚠️  Agent has no LLM and profile has no model/api_key configured[/yellow]")
                return False

        if needs_recreation:
            # Recreate LLM with new configuration
            console.print("[cyan]🔧 Recreating LLM with new configuration...[/cyan]")
            new_llm = create_llm_func(config)
            if new_llm:
                # Use update_llm to propagate to ALL dependent systems
                if hasattr(agent, 'update_llm'):
                    agent.update_llm(new_llm, updated_settings)
                    console.print("[green]  ✅ LLM and all dependent systems updated successfully[/green]")
                else:
                    agent.llm = new_llm
                    if hasattr(agent, 'llm_with_tools') and hasattr(agent, 'langchain_tools'):
                        agent.llm_with_tools = new_llm.bind_tools(agent.langchain_tools)
                    console.print("[green]  ✅ LLM recreated successfully[/green]")
                if config.base_url:
                    console.print(f"[dim]  • Base URL: {config.base_url}[/dim]")
                if config.model:
                    console.print(f"[dim]  • Model: {config.model}[/dim]")
                if config.api_key:
                    masked = config.api_key[:8] + "..." + config.api_key[-4:] if len(config.api_key) > 12 else "***"
                    console.print(f"[dim]  • API Key: {masked}[/dim]")
            else:
                console.print("[yellow]⚠️  Failed to recreate LLM (check configuration)[/yellow]")
                return False
        else:
            # Just update simple parameters
            llm = agent.llm

            if config.temperature is not None and hasattr(llm, 'temperature'):
                llm.temperature = config.temperature
                console.print(f"[dim]  • Set temperature: {config.temperature}[/dim]")

            if config.max_tokens is not None and hasattr(llm, 'max_tokens'):
                llm.max_tokens = config.max_tokens
                console.print(f"[dim]  • Set max_tokens: {config.max_tokens}[/dim]")

            if config.top_p is not None and hasattr(llm, 'top_p'):
                llm.top_p = config.top_p
                console.print(f"[dim]  • Set top_p: {config.top_p}[/dim]")

        return True

    except Exception as e:
        console.print(f"[red]❌ Error applying configuration: {e}[/red]")
        import traceback
        if console.is_terminal:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


def _build_updated_settings(agent: Any, config: Any, console: Console) -> Any:
    """Build updated Settings from agent's current settings + config overrides.

    Returns:
        Updated Settings instance for propagation to dependent systems.
    """
    from sepilot.config.llm_providers import LLMProviderFactory
    from sepilot.config.settings import Settings

    settings = getattr(agent, 'settings', None) or Settings()

    if config.model:
        settings = settings.model_copy(update={"model": config.model})
    if config.temperature is not None:
        settings = settings.model_copy(update={"temperature": config.temperature})
    if config.max_tokens is not None:
        settings = settings.model_copy(update={"max_tokens": config.max_tokens})
    if config.base_url:
        settings = settings.model_copy(update={"api_base_url": config.base_url})
    if config.api_key:
        factory = LLMProviderFactory(settings, console)
        provider = factory.detect_provider(settings.model)
        key_field_map = {
            "anthropic": "anthropic_api_key",
            "google": "google_api_key",
            "openai": "openai_api_key",
            "groq": "groq_api_key",
            "openrouter": "openrouter_api_key",
            "github": "github_token",
            "azure": "azure_openai_api_key",
        }
        key_field = key_field_map.get(provider, "openai_api_key")
        settings = settings.model_copy(update={key_field: config.api_key})
    if config.custom_headers:
        settings = settings.model_copy(update={"custom_headers": config.custom_headers})

    return settings


def create_llm_from_config(config: Any, agent: Any, console: Console) -> Any | None:
    """Create a new LLM instance from configuration.

    Uses LLMProviderFactory for consistent provider detection and LLM creation
    across all 9 supported providers.

    Args:
        config: ModelConfig instance
        agent: Agent instance for fallback settings
        console: Rich console for output

    Returns:
        New LLM instance or None if creation fails
    """
    try:
        from sepilot.config.llm_providers import LLMProviderError, LLMProviderFactory
        from sepilot.config.settings import Settings

        # Get base settings from agent or create defaults
        settings = getattr(agent, 'settings', None)
        if not settings:
            settings = Settings()

        # Apply ModelConfig overrides to a copy of settings
        if config.model:
            settings = settings.model_copy(update={"model": config.model})
        if config.temperature is not None:
            settings = settings.model_copy(update={"temperature": config.temperature})
        if config.max_tokens is not None:
            settings = settings.model_copy(update={"max_tokens": config.max_tokens})
        if config.base_url:
            settings = settings.model_copy(update={"api_base_url": config.base_url})
        if config.api_key:
            # Detect provider to set the correct key field
            factory = LLMProviderFactory(settings, console)
            provider = factory.detect_provider(settings.model)
            key_field_map = {
                "anthropic": "anthropic_api_key",
                "google": "google_api_key",
                "openai": "openai_api_key",
                "groq": "groq_api_key",
                "openrouter": "openrouter_api_key",
                "github": "github_token",
                "azure": "azure_openai_api_key",
            }
            key_field = key_field_map.get(provider, "openai_api_key")
            settings = settings.model_copy(update={key_field: config.api_key})

        if config.custom_headers:
            settings = settings.model_copy(update={"custom_headers": config.custom_headers})

        # Create LLM using unified factory
        factory = LLMProviderFactory(settings, console)
        return factory.create_llm()

    except LLMProviderError as e:
        console.print(f"[red]❌ {e}[/red]")
        if e.suggestion:
            console.print(f"[dim]💡 {e.suggestion}[/dim]")
        return None
    except Exception as e:
        console.print(f"[red]❌ Failed to create LLM: {e}[/red]")
        import traceback
        if console.is_terminal:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None


__all__ = [
    'handle_model_command',
    'apply_model_config_to_agent',
    'create_llm_from_config',
]
