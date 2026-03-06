"""Web operation tools for LangChain agent.

Provides web_fetch, web_search tools.
"""

from langchain_core.tools import tool


@tool
def web_fetch(url: str, prompt: str, timeout: int = 30, use_cache: bool = True) -> str:
    """Fetch content from a URL and process it with an AI prompt.

    Args:
        url: URL to fetch content from (required)
        prompt: AI prompt to process the content (required)
        timeout: Request timeout in seconds (default: 30)
        use_cache: Use cached content if available (default: True)

    Returns:
        Processed web content or error message

    Examples:
        # Extract specific information
        web_fetch(url="https://docs.python.org", prompt="What are the main features of Python 3.12?")

        # Summarize content
        web_fetch(url="https://example.com/article", prompt="Summarize the key points")

        # Extract code examples
        web_fetch(url="https://tutorial.com", prompt="Extract all code examples from this page")
    """
    from sepilot.tools.web_tools.web_fetch_tool import WebFetchTool
    tool_instance = WebFetchTool()
    return tool_instance.execute(url, prompt, timeout, use_cache)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: Search query (required)
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Search results with titles, snippets, and URLs

    Examples:
        web_search(query="Python asyncio tutorial")
        web_search(query="FastAPI best practices", max_results=10)
    """
    from sepilot.tools.web_tools.search_web_tool import WebSearchTool
    tool_instance = WebSearchTool()
    return tool_instance.execute(query=query, max_results=max_results)


__all__ = ['web_fetch', 'web_search']
