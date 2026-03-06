"""RAG (Retrieval-Augmented Generation) command handlers for Interactive Mode.

Real RAG implementation with:
- Vector DB (ChromaDB)
- Embeddings (OpenAI or HuggingFace)
- Document chunking
- Semantic search
"""

import contextlib
import logging
import shlex
from urllib.parse import urlparse

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

logger = logging.getLogger(__name__)


def get_rag_context(query: str, console: Console, n_results: int = 5) -> str:
    """Get RAG context using semantic search.

    Args:
        query: User query to search for relevant context
        console: Rich console for output
        n_results: Maximum number of chunks to return

    Returns:
        Formatted RAG context string
    """
    try:
        # 빠른 사전 검증: documents.json이 없거나 비어있으면 즉시 반환
        # chromadb import + vectorstore init을 완전히 건너뜀
        from pathlib import Path
        import json as _json

        docs_file = Path.home() / ".sepilot" / "rag" / "documents.json"
        if not docs_file.exists():
            return ""
        try:
            with open(docs_file, encoding="utf-8") as _f:
                docs = _json.load(_f)
            if not docs:
                return ""
        except Exception:
            return ""

        # 문서가 있을 때만 무거운 초기화 수행
        from sepilot.rag.manager import get_rag_manager

        rag = get_rag_manager()

        # Check if there are any indexed documents
        stats = rag.get_stats()
        if stats["total_chunks"] == 0:
            return ""

        console.print(f"[dim cyan]🔍 Searching RAG index ({stats['total_chunks']} chunks)...[/dim cyan]")

        # Semantic search
        context = rag.get_context(query, n_results=n_results)

        if context:
            console.print(f"[dim green]✅ Found relevant context from {len(rag.search(query, n_results))} chunks[/dim green]")

        return context

    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"RAG not available: {e}")
        console.print(f"[yellow]⚠️  RAG not available: {e}[/yellow]")
        console.print("[dim]Install with: pip install chromadb sentence-transformers[/dim]")
        return ""
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"RAG search failed: {e}")
        console.print(f"[yellow]⚠️  RAG search failed: {e}[/yellow]")
        return ""


def handle_rag_command(input_text: str, console: Console) -> None:
    """Handle /rag command for RAG management.

    Args:
        input_text: The raw input text from the user
        console: Rich console for output
    """
    # Remove /rag prefix
    input_text = input_text.strip()
    if input_text.lower().startswith('/rag'):
        input_text = input_text[4:].strip()

    # Parse command
    parts = shlex.split(input_text) if input_text else []
    command = parts[0].lower() if len(parts) > 0 else ""
    arg1 = parts[1] if len(parts) > 1 else ""

    def show_help():
        """Display RAG command help"""
        help_text = """
[bold cyan]🔍 RAG (Retrieval-Augmented Generation) - Vector DB Based[/bold cyan]

[bold yellow]📥 Document Indexing:[/bold yellow]
  [cyan]/rag add <url>[/cyan]                     Index URL content into vector DB
  [cyan]/rag add <url1> <url2> ...[/cyan]         Batch index multiple URLs
  [cyan]/rag add <url> --title "title"[/cyan]    Index with custom title
  [cyan]/rag add <url> --force[/cyan]             Force re-index even if exists
  [cyan]/rag remove <url>[/cyan]                  Remove URL from index
  [cyan]/rag refresh <url>[/cyan]                 Re-index URL content
  [cyan]/rag list[/cyan]                          List all indexed documents

[bold yellow]🔎 Search & Query:[/bold yellow]
  [cyan]/rag search <query>[/cyan]                Search for relevant content
  [cyan]/rag search <query> --top 10[/cyan]       Search with custom result count

[bold yellow]📊 Management:[/bold yellow]
  [cyan]/rag stats[/cyan]                         Show RAG statistics
  [cyan]/rag clear[/cyan]                         Clear all indexed data

[bold yellow]💡 How it works:[/bold yellow]
  1. [cyan]/rag add <url>[/cyan] fetches content, chunks it, and stores embeddings
  2. When you ask questions, relevant chunks are retrieved via semantic search
  3. Retrieved context is included in the LLM prompt automatically

[bold yellow]🔧 Embedding Providers:[/bold yellow]
  • [cyan]OpenAI[/cyan] - Set OPENAI_API_KEY (text-embedding-3-small)
  • [cyan]HuggingFace[/cyan] - Free, local (sentence-transformers)
  • Auto-detection: Uses OpenAI if API key available, else HuggingFace

[bold yellow]📝 Examples:[/bold yellow]
  [dim]/rag add https://docs.python.org/3/library/asyncio.html[/dim]
  [dim]/rag add https://url1.com https://url2.com https://url3.com[/dim]
  [dim]> explain how asyncio works[/dim]  (auto-retrieves relevant docs)

[dim]Data stored in: ~/.sepilot/rag/[/dim]
        """
        console.print(Panel(help_text.strip(), border_style="cyan", padding=(1, 2)))

    # Route to appropriate handler
    if not command or command == "help":
        show_help()
        return

    try:
        from sepilot.rag.manager import get_rag_manager
        rag = get_rag_manager()

        if command == "add":
            _handle_add(arg1, parts, rag, console)

        elif command == "remove":
            _handle_remove(arg1, rag, console)

        elif command == "refresh":
            _handle_refresh(arg1, rag, console)

        elif command == "list":
            _handle_list(rag, console)

        elif command == "search":
            query = " ".join(parts[1:]) if len(parts) > 1 else ""
            _handle_search(query, rag, console)

        elif command == "stats":
            _handle_stats(rag, console)

        elif command == "clear":
            _handle_clear(rag, console)

        else:
            console.print(f"[yellow]⚠️  Unknown RAG command: '{command}'[/yellow]\n")
            show_help()

    except ImportError as e:
        console.print(f"[red]❌ RAG dependencies not installed: {e}[/red]")
        console.print("[dim]Install with: pip install chromadb sentence-transformers[/dim]")
    except Exception as e:
        console.print(f"[red]❌ RAG command failed: {e}[/red]")
        import traceback
        if console.is_terminal:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL format.

    Args:
        url: URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is empty"

    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False, f"Invalid URL scheme '{parsed.scheme}'. Only http/https allowed"
        if not parsed.netloc:
            return False, "URL has no host"
        return True, ""
    except Exception as e:
        return False, f"URL parsing error: {e}"


def _handle_add(url: str, parts: list, rag, console: Console) -> None:
    """Handle /rag add command - index one or more URLs."""
    if not url:
        console.print("[yellow]⚠️  Please specify a URL[/yellow]")
        console.print("[dim]Usage: /rag add <url> [url2 url3 ...] [--title \"title\"] [--force][/dim]")
        return

    # Collect URLs and options
    urls: list[str] = []
    title = ""
    description = ""
    force = False

    # First URL
    is_valid, error_msg = _validate_url(url)
    if is_valid:
        urls.append(url)
    else:
        console.print(f"[red]❌ Invalid URL: {url}[/red]")
        console.print(f"[dim]{error_msg}[/dim]")
        return

    # Parse remaining args (could be URLs or options)
    token_iter = iter(parts[2:])
    for token in token_iter:
        if token == "--title":
            try:
                title = next(token_iter)
            except StopIteration:
                console.print("[yellow]⚠️  --title requires a value[/yellow]")
        elif token == "--desc":
            try:
                description = next(token_iter)
            except StopIteration:
                console.print("[yellow]⚠️  --desc requires a value[/yellow]")
        elif token in ("--force", "-f"):
            force = True
        elif token.startswith("-"):
            console.print(f"[yellow]⚠️  Unknown option: {token}[/yellow]")
        else:
            # Check if it's a URL
            is_valid, error_msg = _validate_url(token)
            if is_valid:
                urls.append(token)
            else:
                console.print(f"[yellow]⚠️  Skipping invalid URL: {token} ({error_msg})[/yellow]")

    # Single URL: use simple indexing
    if len(urls) == 1:
        _handle_single_url_add(urls[0], title, description, force, rag, console)
        return

    # Multiple URLs: use batch indexing
    _handle_batch_url_add(urls, force, rag, console)


def _handle_single_url_add(
    url: str,
    title: str,
    description: str,
    force: bool,
    rag,
    console: Console
) -> None:
    """Handle single URL indexing."""
    console.print(f"[cyan]📥 Indexing: {url}[/cyan]")
    console.print("[dim]Fetching content, chunking, and embedding...[/dim]")

    try:
        doc_info = rag.add_url_sync(
            url=url,
            title=title,
            description=description,
            force_refresh=force
        )

        if doc_info.status == "indexed":
            console.print("[bold green]✅ Successfully indexed![/bold green]")
            console.print(f"  [cyan]URL:[/cyan] {doc_info.url}")
            console.print(f"  [cyan]Chunks:[/cyan] {doc_info.chunk_count}")
            console.print(f"  [cyan]Indexed at:[/cyan] {doc_info.indexed_at[:19]}")
        elif doc_info.status == "error":
            console.print(f"[red]❌ Indexing failed: {doc_info.error_message}[/red]")
            _suggest_error_fix(doc_info.error_message, console)
        else:
            console.print(f"[yellow]⚠️  Status: {doc_info.status}[/yellow]")

    except Exception as e:
        logger.exception(f"Failed to index URL: {url}")
        console.print(f"[red]❌ Indexing failed: {e}[/red]")
        _suggest_error_fix(str(e), console)


def _handle_batch_url_add(
    urls: list[str],
    force: bool,
    rag,
    console: Console
) -> None:
    """Handle batch URL indexing with progress display."""
    console.print(f"[cyan]📥 Batch indexing {len(urls)} URLs...[/cyan]")

    results = {"success": 0, "error": 0, "skipped": 0}
    errors: list[tuple[str, str]] = []

    def progress_callback(current: int, total: int, message: str):
        # Progress is handled by rich progress bar
        pass

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Indexing...", total=len(urls))

            doc_infos = rag.add_urls_sync(
                urls=urls,
                force_refresh=force,
                max_concurrent=3,
                progress_callback=lambda c, t, m: progress.update(task, completed=c, description=m[:50])
            )

            progress.update(task, completed=len(urls))

        # Process results
        for doc_info in doc_infos:
            if doc_info.status == "indexed":
                results["success"] += 1
            elif doc_info.status == "error":
                results["error"] += 1
                errors.append((doc_info.url, doc_info.error_message or "Unknown error"))
            else:
                results["skipped"] += 1

        # Display summary
        console.print()
        console.print("[bold cyan]📊 Batch Indexing Complete[/bold cyan]")
        console.print(f"  [green]✅ Success:[/green] {results['success']}")
        if results["error"] > 0:
            console.print(f"  [red]❌ Failed:[/red] {results['error']}")
        if results["skipped"] > 0:
            console.print(f"  [yellow]⏭️  Skipped:[/yellow] {results['skipped']}")

        # Show error details
        if errors:
            console.print()
            console.print("[bold red]Failed URLs:[/bold red]")
            for url, error in errors[:5]:  # Show first 5 errors
                console.print(f"  [dim]• {url[:60]}...[/dim]" if len(url) > 60 else f"  [dim]• {url}[/dim]")
                console.print(f"    [red]{error}[/red]")
            if len(errors) > 5:
                console.print(f"  [dim]... and {len(errors) - 5} more[/dim]")

    except Exception as e:
        logger.exception("Batch indexing failed")
        console.print(f"[red]❌ Batch indexing failed: {e}[/red]")


def _suggest_error_fix(error_message: str, console: Console) -> None:
    """Suggest fixes for common errors."""
    error_lower = error_message.lower()

    if "timeout" in error_lower or "timed out" in error_lower:
        console.print("[dim]💡 Tip: The URL might be slow to respond. Try again later.[/dim]")
    elif "404" in error_lower or "not found" in error_lower:
        console.print("[dim]💡 Tip: The URL might not exist. Check if it's correct.[/dim]")
    elif "403" in error_lower or "forbidden" in error_lower:
        console.print("[dim]💡 Tip: The site might be blocking automated access.[/dim]")
    elif "ssl" in error_lower or "certificate" in error_lower:
        console.print("[dim]💡 Tip: SSL/certificate issue. The site might have security problems.[/dim]")
    elif "connection" in error_lower or "connect" in error_lower:
        console.print("[dim]💡 Tip: Network connection issue. Check your internet connection.[/dim]")
    elif "no content" in error_lower or "empty" in error_lower:
        console.print("[dim]💡 Tip: The page might be empty or dynamically loaded (requires JavaScript).[/dim]")


def _handle_remove(url: str, rag, console: Console) -> None:
    """Handle /rag remove command."""
    if not url:
        console.print("[yellow]⚠️  Please specify a URL[/yellow]")
        console.print("[dim]Usage: /rag remove <url>[/dim]")
        return

    if rag.remove_url(url):
        console.print(f"[bold green]✅ Removed from index:[/bold green] {url}")
    else:
        console.print(f"[yellow]⚠️  URL not found in index: {url}[/yellow]")


def _handle_refresh(url: str, rag, console: Console) -> None:
    """Handle /rag refresh command."""
    if not url:
        console.print("[yellow]⚠️  Please specify a URL[/yellow]")
        console.print("[dim]Usage: /rag refresh <url>[/dim]")
        return

    console.print(f"[cyan]🔄 Re-indexing: {url}[/cyan]")

    doc_info = rag.refresh_url(url)

    if doc_info is None:
        console.print(f"[yellow]⚠️  URL not found in index: {url}[/yellow]")
    elif doc_info.status == "indexed":
        console.print("[bold green]✅ Successfully re-indexed![/bold green]")
        console.print(f"  [cyan]Chunks:[/cyan] {doc_info.chunk_count}")
    else:
        console.print(f"[red]❌ Re-indexing failed: {doc_info.error_message}[/red]")


def _handle_list(rag, console: Console) -> None:
    """Handle /rag list command."""
    docs = rag.list_documents()

    if not docs:
        console.print("[yellow]No documents indexed yet[/yellow]")
        console.print("[dim]Use /rag add <url> to index a URL[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("URL", style="cyan", width=50, no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column("Indexed", style="dim")

    for doc in docs:
        status_style = {
            "indexed": "[green]indexed[/green]",
            "pending": "[yellow]pending[/yellow]",
            "error": "[red]error[/red]"
        }.get(doc.status, doc.status)

        # Truncate URL for display
        display_url = doc.url if len(doc.url) <= 50 else doc.url[:47] + "..."

        table.add_row(
            display_url,
            status_style,
            str(doc.chunk_count),
            doc.indexed_at[:10] if doc.indexed_at else "-"
        )

    console.print(f"\n[bold cyan]📚 Indexed Documents ({len(docs)})[/bold cyan]")
    console.print(table)
    console.print()


def _handle_search(query: str, rag, console: Console) -> None:
    """Handle /rag search command."""
    if not query:
        console.print("[yellow]⚠️  Please specify a search query[/yellow]")
        console.print("[dim]Usage: /rag search <query>[/dim]")
        return

    # Parse --top option
    n_results = 5
    if "--top" in query:
        parts = query.split("--top")
        query = parts[0].strip()
        with contextlib.suppress(ValueError, IndexError):
            n_results = int(parts[1].strip().split()[0])

    console.print(f"[cyan]🔎 Searching for: {query}[/cyan]")

    results = rag.search(query, n_results=n_results)

    if not results:
        console.print("[yellow]No relevant results found[/yellow]")
        return

    console.print(f"\n[bold green]Found {len(results)} relevant chunks:[/bold green]\n")

    for i, result in enumerate(results, 1):
        # Show score and source
        console.print(f"[bold cyan]#{i}[/bold cyan] [dim](score: {result.score:.3f})[/dim]")
        console.print(f"[dim]Source: {result.source}[/dim]")

        # Show content preview (first 300 chars)
        content = result.content[:300]
        if len(result.content) > 300:
            content += "..."
        console.print(f"[white]{content}[/white]")
        console.print()


def _handle_stats(rag, console: Console) -> None:
    """Handle /rag stats command."""
    stats = rag.get_stats()

    console.print("\n[bold cyan]📊 RAG Statistics[/bold cyan]")
    console.print(f"  [cyan]Total documents:[/cyan] {stats['total_documents']}")
    console.print(f"  [cyan]Indexed:[/cyan] {stats['indexed_documents']}")
    console.print(f"  [cyan]Errors:[/cyan] {stats['error_documents']}")
    console.print(f"  [cyan]Total chunks:[/cyan] {stats['total_chunks']}")

    console.print("\n[bold cyan]🔧 Configuration[/bold cyan]")
    console.print(f"  [cyan]Embedding provider:[/cyan] {stats['embedding_provider']}")
    console.print(f"  [cyan]Embedding dimension:[/cyan] {stats['embedding_dimension']}")
    console.print(f"  [cyan]Chunk size:[/cyan] {stats['chunk_size']}")
    console.print(f"  [cyan]Chunk overlap:[/cyan] {stats['chunk_overlap']}")
    console.print(f"  [cyan]Data directory:[/cyan] {stats['persist_dir']}")
    console.print()


def _handle_clear(rag, console: Console) -> None:
    """Handle /rag clear command."""
    stats = rag.get_stats()

    if stats['total_chunks'] == 0:
        console.print("[yellow]RAG index is already empty[/yellow]")
        return

    console.print(f"[yellow]⚠️  This will delete {stats['total_documents']} documents and {stats['total_chunks']} chunks.[/yellow]")

    try:
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            console.print("[dim]Cancelled[/dim]")
            return
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled[/dim]")
        return

    result = rag.clear()
    console.print(f"[bold green]✅ Cleared {result['documents']} documents and {result['chunks']} chunks[/bold green]")


__all__ = ['handle_rag_command', 'get_rag_context']
