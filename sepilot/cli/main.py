#!/usr/bin/env python3
"""SE Pilot CLI - Main entry point"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

# langchain-core emits a known compatibility warning on Python 3.14+
# due to its legacy pydantic.v1 compatibility layer. Keep runtime noise low
# while preserving real errors.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from rich.console import Console

from sepilot.agent.base_agent import ReactAgent
from sepilot.agent.execution_context import is_user_visible_conversation_message
from sepilot.config.settings import Settings
from sepilot.loggers.file_logger import FileLogger
from sepilot.ui import InteractiveMode
from sepilot.ui.commands.rag_commands import get_rag_context

try:
    from sepilot.e2e.scripted_executor import try_run_e2e_script
    HAS_E2E = True
except ImportError:
    HAS_E2E = False
    try_run_e2e_script = None

# Load environment variables
load_dotenv()

console = Console()

# Keywords for detecting API-key-related errors in deferred agent initialization
_API_KEY_ERROR_KEYWORDS = ("API_KEY", "NOT FOUND", "CREDENTIALS")


def _is_transient_llm_failure_result(result: str | None) -> bool:
    """Detect transient upstream LLM/server failures in agent result text."""
    if not result:
        return False
    lowered = result.lower()
    if "i encountered an error calling the llm" not in lowered:
        return False
    transient_keywords = (
        "500",
        "internal server error",
        "api_error",
        "connection refused",
        "connection reset",
        "timed out",
        "timeout",
    )
    return any(kw in lowered for kw in transient_keywords)


def _execute_with_transient_retry(
    agent: ReactAgent,
    prompt: str,
    logger: FileLogger | None = None,
    attempts: int = 3,
) -> str | None:
    """Execute agent task with lightweight retry for transient upstream failures.

    Retries are capped by elapsed wall time to avoid re-running a long task
    multiple times within a single CLI invocation.
    """
    retry_budget_raw = os.getenv("SEPILOT_TRANSIENT_RETRY_BUDGET_SECONDS", "120")
    try:
        retry_budget_seconds = max(float(retry_budget_raw), 0.0)
    except ValueError:
        retry_budget_seconds = 120.0

    result: str | None = None
    started_at = time.monotonic()
    for attempt in range(1, attempts + 1):
        result = agent.execute(prompt)
        if not _is_transient_llm_failure_result(result):
            return result
        if logger:
            logger.log_error(f"Transient LLM failure detected (attempt {attempt}/{attempts})")
        elapsed = time.monotonic() - started_at
        can_retry = attempt < attempts and elapsed < retry_budget_seconds
        if can_retry:
            time.sleep(2 * attempt)
        else:
            if logger and attempt < attempts:
                logger.log_error(
                    "Skipping transient retry due to elapsed retry budget "
                    f"({elapsed:.1f}s >= {retry_budget_seconds:.1f}s)."
                )
            break
    return result


def _run_interactive_mode(
    model: str | None,
    log_dir: str,
    verbose: bool,
    thread_id: str | None,
    max_iterations: int,
    prompt_profile: str,
    no_memory: bool
):
    """Run SE Pilot in interactive REPL mode"""
    try:
        console_capture = None

        # Initialize settings
        interactive_verbose = verbose

        settings = Settings(
            model=model or os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
            verbose=interactive_verbose,
            max_iterations=max_iterations,
            prompt_profile=prompt_profile
        )

        # Initialize logger
        logger = FileLogger(
            log_dir=Path(log_dir),
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            continue_session=False
        )

        # Initialize agent (lazy_llm=True defers LLM init so interactive mode
        # can start without OPENAI_API_KEY; user configures via /model apply)
        console.print("[cyan]Initializing agent in interactive verbose mode...[/cyan]")
        agent = ReactAgent(
            settings=settings,
            logger=logger,
            prompt_profile=prompt_profile,
            thread_id=thread_id,
            enable_memory=not no_memory,
            auto_approve=False,  # Enable human-in-the-loop for interactive mode
            show_progress=False,  # Keep status panel hidden; rely on shortcut overlays instead
            lazy_llm=True
        )

        def agent_factory(model_config=None):
            """Recreate agent after model/API key configuration change."""
            nonlocal agent
            updated_settings = settings
            if model_config:
                updates = {}
                if getattr(model_config, 'model', None):
                    updates['model'] = model_config.model
                if getattr(model_config, 'api_key', None):
                    from sepilot.config.llm_providers import LLMProviderFactory
                    factory = LLMProviderFactory(updated_settings)
                    provider = factory.detect_provider(model_config.model or updated_settings.model)
                    key_map = {
                        "openai": "openai_api_key", "anthropic": "anthropic_api_key",
                        "google": "google_api_key", "groq": "groq_api_key",
                        "openrouter": "openrouter_api_key", "github": "github_token",
                        "azure": "azure_openai_api_key",
                    }
                    key_field = key_map.get(provider, "openai_api_key")
                    updates[key_field] = model_config.api_key
                if getattr(model_config, 'base_url', None):
                    updates['api_base_url'] = model_config.base_url
                if updates:
                    updated_settings = settings.model_copy(update=updates)

            # Cleanup existing agent before replacing
            if agent is not None:
                try:
                    agent.cleanup()
                except Exception:
                    pass

            new_agent = ReactAgent(
                settings=updated_settings,
                logger=logger,
                prompt_profile=prompt_profile,
                thread_id=thread_id,
                enable_memory=not no_memory,
                auto_approve=False,
                show_progress=False
            )
            agent = new_agent

            return new_agent

        conversation_context: list[BaseMessage] = []
        max_context_messages = 50

        def _uses_graph_context() -> bool:
            return bool(agent and getattr(agent, "enable_memory", False))

        def _sync_context_from_agent() -> bool:
            nonlocal conversation_context
            if not _uses_graph_context() or not agent or not hasattr(agent, "get_conversation_messages"):
                return False

            try:
                messages = agent.get_conversation_messages()
            except Exception:
                return False

            conversation_context.clear()
            for msg in messages:
                if is_user_visible_conversation_message(msg):
                    conversation_context.append(msg)
            return True

        def _trim_context():
            """Smart context management with auto-compaction based on token usage."""
            nonlocal conversation_context

            if _uses_graph_context():
                return

            if not conversation_context:
                return

            # Import context manager
            from sepilot.agent.context_manager import ContextManager

            # Get max tokens from environment
            max_tokens = int(os.getenv('MAX_TOKENS', '96000'))
            context_manager = ContextManager(
                max_context_tokens=max_tokens,
                warning_threshold=0.7,
                compact_threshold=0.8,
            )

            # Calculate current token usage
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                current_tokens = sum(
                    len(encoding.encode(str(m.content)))
                    for m in conversation_context
                    if hasattr(m, 'content') and m.content
                )
            except Exception:
                # Fallback estimation
                current_tokens = len(conversation_context) * 400

            # Check if we should compact
            if context_manager.should_compact(current_tokens):
                console.print("[yellow]⚠️  Context usage high (>80%), auto-compacting...[/yellow]")

                # Try to summarize using LLM
                if agent and hasattr(agent, 'llm'):
                    try:
                        conversation_context = context_manager.summarize_messages(
                            conversation_context,
                            agent.llm,
                            keep_recent=10
                        )
                        console.print("[green]✅ Context summarized successfully[/green]")
                        return
                    except Exception:
                        console.print("[yellow]Summarization failed, using simple compaction[/yellow]")

                # Fallback to simple compaction
                conversation_context = context_manager.compact_messages(
                    conversation_context,
                    keep_recent=10
                )
                console.print("[green]✅ Context compacted[/green]")

            elif context_manager.should_warn(current_tokens):
                # Warn user about high context usage
                usage_percent = (current_tokens / max_tokens) * 100
                console.print(
                    f"[dim yellow]💡 Context: {usage_percent:.1f}% "
                    f"({current_tokens:,}/{max_tokens:,} tokens)[/dim yellow]"
                )

            # Also apply message count limit as backup
            if len(conversation_context) > max_context_messages:
                conversation_context = conversation_context[-max_context_messages:]

        def register_manual_command(command: str, output: str):
            nonlocal conversation_context
            manual_summary = (
                f"[Manual Shell]\n"
                f"$ {command}\n"
                f"{output}"
            )
            if _uses_graph_context() and agent and hasattr(agent, "append_conversation_message"):
                appended = agent.append_conversation_message(SystemMessage(content=manual_summary))
                if appended:
                    _sync_context_from_agent()
                else:
                    conversation_context.append(HumanMessage(content=manual_summary))
            else:
                conversation_context.append(HumanMessage(content=manual_summary))
            logger.log_tool_call(
                tool_name="manual_shell",
                input_data={"command": command},
                output=output
            )
            _trim_context()

        def execute_with_context(user_input: str) -> str | None:
            nonlocal conversation_context
            if agent is None or agent.llm is None:
                console.print(
                    "[yellow]⚠️  LLM is not configured yet. Please set up your LLM first:\n"
                    "  /model set api_key <your-api-key>\n"
                    "  /model set model <model-name>\n"
                    "  /model apply[/yellow]"
                )
                return None
            try:
                if _uses_graph_context():
                    result = agent.execute(user_input)
                else:
                    context_snapshot = list(conversation_context)
                    result = agent.execute(user_input, context_messages=context_snapshot)
            except Exception as e:
                from rich.markup import escape
                console.print(f"[red]Error: {escape(str(e))}[/red]")
                if verbose:
                    import traceback
                    console.print(traceback.format_exc())
                return None

            # Extract pure user input (strip memory/context prefixes for storage)
            pure_user_input = user_input
            if user_input.strip().startswith('# User Memory') or user_input.strip().startswith('# Context'):
                # Find "# User Request" marker and extract content after it
                if '# User Request' in user_input:
                    pure_user_input = user_input.split('# User Request')[-1].strip()
                else:
                    # Fallback: get last paragraph
                    paragraphs = user_input.split('\n\n')
                    for para in reversed(paragraphs):
                        para = para.strip()
                        if para and not para.startswith('#'):
                            pure_user_input = para
                            break

            if _uses_graph_context():
                _sync_context_from_agent()
            else:
                conversation_context.append(HumanMessage(content=pure_user_input))
                if result:
                    conversation_context.append(AIMessage(content=result))
                _trim_context()
            return result

        # Show thread ID
        if agent and not thread_id and not no_memory and agent.get_thread_id():
            console.print(f"[dim]Thread ID: {agent.get_thread_id()}[/dim]")

        if _uses_graph_context():
            _sync_context_from_agent()

        # Start interactive mode
        interactive_mode = InteractiveMode(
            execute_callback=execute_with_context,
            console=console,
            manual_observer=register_manual_command,
            agent=agent,
            conversation_context=conversation_context,
            agent_factory=agent_factory
        )
        interactive_mode.start()

        # Clean up
        if agent:
            agent.cleanup()
        logger.save_session()

        console.print(f"\n[dim]Session log saved to: {logger.get_session_path()}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        from rich.markup import escape
        console.print(f"\n[bold red]❌ Error: {escape(str(e))}[/bold red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@click.command()
@click.version_option(version=__import__("sepilot").__version__, prog_name="SEPilot")
@click.argument("prompt_arg", required=False, default=None)
@click.option(
    "--prompt", "-p",
    required=False,
    help="Task to execute (can also be passed as positional argument)"
)
@click.option(
    "--model", "-m",
    default=None,
    help="LLM model to use (e.g., claude-3-opus-20240229, gpt-4)"
)
@click.option(
    "--log-dir",
    default="./logs",
    help="Directory to save logs"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--continue-session", "-c",
    is_flag=True,
    help="Continue from most recent conversation (Claude Code style)"
)
@click.option(
    "--resume", "-r",
    is_flag=True,
    help="Show thread selector to resume a previous conversation (Claude Code style)"
)
@click.option(
    "--thread-id", "-t",
    default=None,
    help="Thread ID to continue conversation (for session continuity)"
)
@click.option(
    "--max-iterations",
    default=30,
    help="Maximum iterations for ReAct loop"
)
@click.option(
    "--prompt-profile", "-pp",
    default="default",
    help="Prompt profile to use (default, claude_code, codex, gemini)"
)
@click.option(
    "--no-memory",
    is_flag=True,
    help="Disable memory system (no caching/checkpoints)"
)
@click.option(
    "--list-threads",
    is_flag=True,
    help="List available thread IDs and exit"
)
@click.option(
    "--fast",
    is_flag=True,
    help="Use fast 5-node ReAct graph (for small models like qwen3:8b)"
)
@click.option(
    "--interactive", "-i",
    is_flag=True,
    help="Start interactive REPL mode"
)
@click.option(
    "--git",
    default=None,
    help="Execute git command directly (e.g., --git 'add @file.py')"
)
@click.option(
    "--github",
    default=None,
    help="Execute GitHub command directly (e.g., --github 'issue list')"
)
@click.option(
    "--output-format", "-o",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format for non-interactive mode (text, json, markdown)"
)
@click.option(
    "--stdin",
    is_flag=True,
    help="Read prompt from stdin (for piping)"
)
@click.option(
    "--print-cost",
    is_flag=True,
    help="Print cost estimation at the end"
)
def main(
    prompt_arg: str | None,
    prompt: str | None,
    model: str | None,
    log_dir: str,
    verbose: bool,
    continue_session: bool,
    resume: bool,
    thread_id: str | None,
    max_iterations: int,
    prompt_profile: str,
    no_memory: bool,
    fast: bool,
    list_threads: bool,
    interactive: bool,    git: str | None,
    github: str | None,
    output_format: str,
    stdin: bool,
    print_cost: bool,
):
    # --fast flag sets graph mode via env var (picked up by Settings.graph_mode default_factory)
    if fast:
        os.environ["SEPILOT_GRAPH_MODE"] = "fast"
        console.print("[dim cyan]⚡ Fast graph mode enabled (5-node ReAct loop)[/dim cyan]")

    # Combine positional prompt with -p option (positional takes precedence)
    if prompt_arg and not prompt:
        prompt = prompt_arg
    elif prompt_arg and prompt:
        # Both provided - combine them
        prompt = f"{prompt_arg}\n\n{prompt}"
    """SE Pilot - Intelligent CLI Agent for Software Engineering

    Execute complex tasks using ReAct pattern with LangGraph.

    Examples:

        sepilot -p "Read main.py and write unit tests"

        sepilot -p "Find all TODO comments in the project"

        sepilot -p "Generate documentation for all Python files"

        sepilot -t thread123 -p "Continue with the implementation"

        sepilot --list-threads   # List available thread IDs
    """
    # Handle list threads option
    if list_threads:
        from sepilot.memory import SqliteCheckpointer
        checkpointer = SqliteCheckpointer()
        threads = checkpointer.list_threads()

        console.print("[bold green]Available Thread IDs:[/bold green]")
        if threads:
            for thread in threads:
                console.print(f"  - {thread}")
        else:
            console.print("  No threads found")
        return

    # Handle --continue: auto-select most recent thread (Claude Code style)
    if continue_session and not thread_id:
        from sepilot.memory import SqliteCheckpointer
        checkpointer = SqliteCheckpointer()
        threads = checkpointer.list_threads()

        if threads:
            thread_id = threads[0]  # Most recent thread
            console.print(f"[cyan]Continuing from most recent thread: {thread_id}[/cyan]")
        else:
            console.print("[yellow]No previous threads found. Starting new conversation.[/yellow]")

    # Handle --resume: show thread selector (Claude Code style)
    if resume and not thread_id:
        from sepilot.memory import SqliteCheckpointer
        from sepilot.ui.input_utils import interactive_select

        checkpointer = SqliteCheckpointer()
        threads_meta = checkpointer.list_threads_with_metadata()

        if not threads_meta:
            console.print("[yellow]No previous threads found. Starting new conversation.[/yellow]")
        else:
            # Build selector items from thread metadata
            selector_items: list[dict[str, str]] = []
            for t in threads_meta[:15]:
                tid = t["thread_id"]
                updated = t.get("updated_at", "")[:16]  # YYYY-MM-DD HH:MM
                msg_count = t.get("message_count", 0)
                preview = t.get("last_message_preview", "") or ""
                if len(preview) > 60:
                    preview = preview[:60] + "..."

                label = f"{tid[:20]}"
                desc_parts = []
                if updated:
                    desc_parts.append(updated)
                if msg_count:
                    desc_parts.append(f"{msg_count}msgs")
                if preview:
                    desc_parts.append(f'"{preview}"')
                description = "  ".join(desc_parts)

                selector_items.append({"label": label, "description": description})

            # Add "new conversation" option at the end
            selector_items.append({"label": "New conversation", "description": "Start fresh"})

            selected = interactive_select(
                selector_items,
                title="Resume a conversation:",
            )

            if selected is None:
                console.print("[dim]Cancelled. Starting new conversation.[/dim]")
            elif selected == len(selector_items) - 1:
                console.print("[dim]Starting new conversation...[/dim]")
            else:
                thread_id = threads_meta[selected]["thread_id"]
                console.print(f"[green]Resuming thread: {thread_id}[/green]")

    # Handle git command
    if git:
        from sepilot.agent.git_agent import GitAgent

        settings = Settings(
            model=model or os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
            verbose=verbose,
            max_iterations=max_iterations,
            prompt_profile=prompt_profile
        )

        logger = FileLogger(
            log_dir=Path(log_dir),
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            continue_session=False
        )

        git_agent = GitAgent(settings=settings, logger=logger, console=console)

        # Parse git command
        parts = git.split(maxsplit=1)
        command = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        # Execute git command
        try:
            if command == "add":
                success = git_agent.run_add(args)
            elif command == "commit":
                success = git_agent.run_commit(args)
            elif command == "push":
                success = git_agent.run_push(args)
            elif command == "pull":
                success = git_agent.run_pull(args)
            elif command == "status":
                success = git_agent.run_status()
            elif command == "log":
                success = git_agent.run_log(args)
            elif command == "diff":
                success = git_agent.run_diff(args)
            elif command == "branch":
                success = git_agent.run_branch(args)
            elif command == "switch":
                success = git_agent.run_switch(args)
            elif command == "ai-commit":
                success = git_agent.run_ai_commit(args)
            else:
                console.print(f"[red]Unknown git command: {command}[/red]")
                success = False

            sys.exit(0 if success else 1)
        except Exception as e:
            console.print(f"[red]Error executing git command: {e}[/red]")
            sys.exit(1)

    # Handle github command
    if github:
        from sepilot.agent.github_agent import GitHubAgent

        settings = Settings(
            model=model or os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
            verbose=verbose,
            max_iterations=max_iterations,
            prompt_profile=prompt_profile
        )

        logger = FileLogger(
            log_dir=Path(log_dir),
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            continue_session=False
        )

        github_agent = GitHubAgent(settings=settings, logger=logger, console=console)

        # Parse github command
        parts = github.split(maxsplit=2)
        command = parts[0] if parts else ""
        subcommand = parts[1] if len(parts) > 1 else ""
        args = parts[2] if len(parts) > 2 else ""

        # Execute github command
        try:
            if command == "issue":
                if subcommand == "list":
                    success = github_agent.run_issue_list(args)
                elif subcommand == "view":
                    success = github_agent.run_issue_view(args)
                else:
                    console.print(f"[red]Unknown issue subcommand: {subcommand}[/red]")
                    success = False
            elif command == "pr":
                if subcommand == "list":
                    success = github_agent.run_pr_list(args)
                elif subcommand == "view":
                    success = github_agent.run_pr_view(args)
                elif subcommand == "review":
                    success = github_agent.run_pr_review(args)
                else:
                    console.print(f"[red]Unknown pr subcommand: {subcommand}[/red]")
                    success = False
            elif command == "actions":
                if subcommand == "list":
                    success = github_agent.run_actions_list()
                elif subcommand == "runs":
                    success = github_agent.run_actions_runs(args)
                else:
                    console.print(f"[red]Unknown actions subcommand: {subcommand}[/red]")
                    success = False
            elif command == "config":
                if subcommand == "show":
                    success = github_agent.run_config_show()
                else:
                    console.print(f"[red]Unknown config subcommand: {subcommand}[/red]")
                    success = False
            elif command == "ai":
                # AI Assistant
                # --github "ai" or --github "ai what PRs need attention?"
                query = (subcommand + " " + args).strip()
                success = github_agent.run_ai_assistant(query)
            else:
                console.print(f"[red]Unknown github command: {command}[/red]")
                success = False

            sys.exit(0 if success else 1)
        except Exception as e:
            console.print(f"[red]Error executing github command: {e}[/red]")
            sys.exit(1)
    # Handle interactive mode
    if interactive:
        _run_interactive_mode(
            model=model,
            log_dir=log_dir,
            verbose=verbose,
            thread_id=thread_id,
            max_iterations=max_iterations,
            prompt_profile=prompt_profile,
            no_memory=no_memory
        )
        return

    # Handle stdin input (for piping)
    if stdin:
        if not sys.stdin.isatty():
            stdin_input = sys.stdin.read().strip()
            if stdin_input:
                prompt = f"{stdin_input}\n\n{prompt}" if prompt else stdin_input
        else:
            console.print("[yellow]Warning: --stdin specified but no piped input detected[/yellow]")

    # No prompt given → default to interactive mode only for real terminals.
    if not prompt:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise click.UsageError(
                "No prompt provided. Use --interactive, pass --prompt/-p, or pipe input with --stdin."
            )

        _run_interactive_mode(
            model=model,
            log_dir=log_dir,
            verbose=verbose,
            thread_id=thread_id,
            max_iterations=max_iterations,
            prompt_profile=prompt_profile,
            no_memory=no_memory
        )
        return

    # Try E2E script if available (development feature)
    if HAS_E2E:
        scripted_result = try_run_e2e_script(prompt)
        if scripted_result is not None:
            console.print(scripted_result)
            return

    try:
        # Initialize settings
        settings = Settings(
            model=model or os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
            verbose=verbose,
            max_iterations=max_iterations,
            prompt_profile=prompt_profile
        )

        # Initialize logger
        logger = FileLogger(
            log_dir=Path(log_dir),
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            continue_session=continue_session
        )

        # Log the prompt
        logger.log_prompt(prompt)

        # Add RAG context if available (auto-retrieval from indexed documents)
        rag_context = get_rag_context(prompt, console)
        if rag_context:
            prompt = rag_context + "\n\n# User Request\n" + prompt
            if verbose:
                console.print("[dim cyan]📚 RAG context added to prompt[/dim cyan]")

        # Show starting message (only in verbose mode)
        if verbose:
            from rich.panel import Panel
            from rich.text import Text

            # Create box logo
            logo_text = Text.from_markup(
                "[bold cyan]╔════════════════════╗[/bold cyan]\n"
                "[bold cyan]║[/bold cyan]                    [bold cyan]║[/bold cyan]\n"
                "[bold cyan]║[/bold cyan]  [bold white]SEPilot CLI[/bold white]   [bold cyan]║[/bold cyan]\n"
                "[bold cyan]║[/bold cyan]                    [bold cyan]║[/bold cyan]\n"
                "[bold cyan]╚════════════════════╝[/bold cyan]",
                justify="center"
            )
            logo_panel = Panel(
                logo_text,
                border_style="cyan",
                padding=(0, 2)
            )
            console.print(logo_panel)
            console.print()

            console.print(f"[dim]Model: {settings.model}[/dim]")
            console.print(f"[dim]Profile: {prompt_profile}[/dim]")
            if thread_id:
                console.print(f"[dim]Thread ID: {thread_id}[/dim]")
            console.print(f"[dim]Memory: {'Disabled' if no_memory else 'Enabled'}[/dim]")
            console.print(f"[dim]Task: {prompt}[/dim]\n")

        # Initialize agent
        if not verbose:
            # Silent mode for non-verbose - no progress spinner, no messages
            agent = ReactAgent(
                settings=settings,
                logger=logger,
                prompt_profile=prompt_profile,
                thread_id=thread_id,
                enable_memory=not no_memory,
                auto_approve=True  # Auto-approve in prompt mode (non-interactive)
            )

            # Execute the task
            result = _execute_with_transient_retry(agent, prompt, logger=logger)
        else:
            # Verbose mode - show all messages
            console.print("[cyan]Initializing agent...[/cyan]\n")

            agent = ReactAgent(
                settings=settings,
                logger=logger,
                prompt_profile=prompt_profile,
                thread_id=thread_id,
                enable_memory=not no_memory,
                auto_approve=True  # Auto-approve in prompt mode (non-interactive)
            )

            # Show thread ID if new one was created
            if not thread_id and not no_memory and agent.get_thread_id():
                console.print(f"[dim]New Thread ID: {agent.get_thread_id()}[/dim]\n")

            # Show session context if resuming
            if thread_id and not no_memory:
                summary = agent.get_session_summary()
                if "No messages" not in summary and "No active session" not in summary:
                    console.print("[dim]Session Context:[/dim]")
                    console.print(f"[dim]{summary}[/dim]\n")

            # Execute the task (verbose output will be shown by agent)
            result = _execute_with_transient_retry(agent, prompt, logger=logger)

        # Surface transient upstream LLM failures as non-zero exit so wrappers can retry.
        if _is_transient_llm_failure_result(result):
            console.print(result or "")
            if 'logger' in locals():
                logger.log_error("Transient LLM failure detected in final result.")
                logger.save_session()
            sys.exit(1)

        # Display result based on output format
        if output_format == "json":
            import json
            output_data = {
                "success": True,
                "result": result or "",
                "model": settings.model,
                "thread_id": agent.get_thread_id() if not no_memory else None,
            }
            if print_cost:
                # Estimate tokens and cost
                token_estimate = len(result or "") // 4 if result else 0
                output_data["estimated_tokens"] = token_estimate
                output_data["estimated_cost"] = f"${token_estimate * 0.00003:.4f}"  # Rough estimate
            print(json.dumps(output_data, ensure_ascii=False, indent=2))
        elif output_format == "markdown":
            if result:
                print(f"# Result\n\n{result}")
                if print_cost:
                    token_estimate = len(result) // 4
                    print(f"\n---\n*Estimated tokens: {token_estimate}*")
        else:
            # Default text format
            if not verbose:
                # Non-verbose: just print the result without decorations
                if result:
                    console.print(result)
            else:
                # Verbose: show completion message and detailed result
                console.print("\n[bold green]✅ Task completed successfully![/bold green]")
                if result:
                    console.print("\n[bold]Result:[/bold]")
                    console.print(result)

        # Show memory stats if enabled (only in verbose mode)
        if verbose and not no_memory:
            stats = agent.get_memory_stats()
            if stats:
                console.print("\n[dim]Memory Statistics:[/dim]")
                if "cache" in stats:
                    cache_stats = stats["cache"]
                    prompt_cache = cache_stats.get("prompt_cache", {})
                    tool_cache = cache_stats.get("tool_cache", {})
                    console.print(f"[dim]  Prompt Cache: {prompt_cache.get('size', 0)} entries, {prompt_cache.get('total_hits', 0)} hits[/dim]")
                    console.print(f"[dim]  Tool Cache: {tool_cache.get('size', 0)} entries, {tool_cache.get('total_hits', 0)} hits[/dim]")
                if "session" in stats:
                    session = stats["session"]
                    console.print(f"[dim]  Session Messages: {session.get('message_count', 0)}[/dim]")

        # Clean up agent memory
        agent.cleanup()

        # Save session
        logger.save_session()

        # Show cost estimation if requested (for text output)
        if print_cost and output_format == "text":
            token_estimate = len(result or "") // 4 if result else 0
            console.print(f"\n[dim]Estimated tokens: ~{token_estimate}[/dim]")
            console.print(f"[dim]Estimated cost: ~${token_estimate * 0.00003:.4f}[/dim]")

        # Show log path and thread ID (only in verbose mode)
        if verbose:
            console.print(f"\n[dim]Log saved to: {logger.get_session_path()}[/dim]")
            if not no_memory and agent.get_thread_id():
                console.print(f"[dim]To continue this conversation, use: sepilot -t {agent.get_thread_id()} -p \"your prompt\"[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Task interrupted by user[/yellow]")
        if 'logger' in locals():
            logger.save_session()
        sys.exit(1)

    except Exception as e:
        from rich.markup import escape
        console.print(f"\n[bold red]❌ Error: {escape(str(e))}[/bold red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        if 'logger' in locals():
            logger.log_error(str(e))
            logger.save_session()
        sys.exit(1)


@click.command()
@click.option(
    "--language", "-l",
    default="all",
    help="Language to install server for (python, typescript, go, rust, all)"
)
@click.option(
    "--check", "-c",
    is_flag=True,
    help="Check which servers are installed"
)
def install_lsp(language: str, check: bool):
    """Install LSP (Language Server Protocol) servers for code intelligence.

    Supported language servers:
        - python: pyright (npm install -g pyright)
        - typescript/javascript: typescript-language-server
        - go: gopls
        - rust: rust-analyzer

    Examples:

        sepilot-lsp --check           # Check installed servers

        sepilot-lsp -l python         # Install pyright

        sepilot-lsp -l all            # Install all servers
    """
    from sepilot.lsp.servers import (
        SERVER_CONFIGS,
        get_available_servers,
        install_all_servers,
        install_server,
    )

    if check:
        console.print("[bold cyan]LSP Server Status:[/bold cyan]\n")
        available = get_available_servers()

        for lang, config in SERVER_CONFIGS.items():
            status = "[green]✓ Installed[/green]" if lang in available else "[red]✗ Not installed[/red]"
            console.print(f"  {lang}: {status}")
            console.print(f"    [dim]Server: {config.name}[/dim]")
            if config.install_command:
                console.print(f"    [dim]Install: {config.install_command}[/dim]")
            console.print()
        return

    if language == "all":
        console.print("[bold cyan]Installing all LSP servers...[/bold cyan]\n")
        results = install_all_servers()
        for lang, (success, message) in results.items():
            if success:
                console.print(f"  [green]✓[/green] {lang}: {message}")
            else:
                console.print(f"  [red]✗[/red] {lang}: {message}")
    else:
        if language not in SERVER_CONFIGS:
            console.print(f"[red]Unknown language: {language}[/red]")
            console.print(f"[dim]Available: {', '.join(SERVER_CONFIGS.keys())}[/dim]")
            sys.exit(1)

        console.print(f"[cyan]Installing {language} LSP server...[/cyan]")
        success, message = install_server(language)
        if success:
            console.print(f"[green]✓ {message}[/green]")
        else:
            console.print(f"[red]✗ {message}[/red]")
            sys.exit(1)


if __name__ == "__main__":
    main()
