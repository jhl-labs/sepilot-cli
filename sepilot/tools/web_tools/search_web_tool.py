"""Web search tool using Tavily only."""

import html
import os
import re

import httpx

from sepilot.tools.base_tool import BaseTool


def _get_http_client_kwargs(timeout: int = 10) -> dict:
    """Get HTTP client kwargs with proxy and SSL configuration from environment.

    Args:
        timeout: Request timeout in seconds

    Returns:
        Dictionary of kwargs for httpx.Client
    """
    kwargs = {
        "timeout": timeout,
        "follow_redirects": True,
    }

    # Proxy configuration from environment
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    if https_proxy or http_proxy:
        kwargs["proxy"] = https_proxy or http_proxy

    # SSL verification from environment
    ssl_verify = os.getenv("SSL_VERIFY", "true").lower() not in ("false", "0", "no")
    ssl_cert_file = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")

    if not ssl_verify:
        kwargs["verify"] = False
    elif ssl_cert_file:
        kwargs["verify"] = ssl_cert_file

    return kwargs


class WebSearchTool(BaseTool):
    """Tool for searching the web"""

    name = "web_search"
    description = "Search the web for information"
    parameters = {
        "query": "Search query (required)",
        "max_results": "Maximum number of results (default: 5, max: 10)",
        "time_range": "Time range filter (day/week/month/year)",
        "provider": "Search provider: tavily/auto (default: auto)",
    }

    _TIME_RANGE_MAP = {
        "day": "d",
        "week": "w",
        "month": "m",
        "year": "y",
    }

    def execute(
        self,
        query: str,
        max_results: int = 5,
        time_range: str | None = None,
        provider: str = "auto",
    ) -> str:
        """Search the web using Tavily only."""
        self.validate_params(query=query)

        query = query.strip()
        if not query:
            return "Web search error: Query cannot be empty."

        max_results = max(1, min(max_results, 10))
        _, normalized_range = self._normalize_time_range(time_range)
        provider = (provider or "auto").strip().lower()
        api_key = os.getenv("TAVILY_API_KEY")
        notes: list[str] = []

        try:
            if provider not in ("auto", "tavily"):
                return (
                    f"Web search error: Provider '{provider}' is disabled. "
                    "Only 'tavily' is supported."
                )

            if not api_key:
                return (
                    "Web search error: TAVILY_API_KEY is not set. "
                    "Set TAVILY_API_KEY to enable web_search."
                )

            results = self._search_tavily(query, max_results, api_key)
            if normalized_range:
                notes.append(
                    "time_range is not supported by Tavily; returning unfiltered Tavily results."
                )

            invalid_range = time_range if time_range and not normalized_range else None
            return self._format_results(
                query,
                results,
                normalized_range,
                "tavily",
                invalid_range,
                notes,
            )

        except httpx.TimeoutException:
            return "Web search error: Request timed out."
        except httpx.HTTPError as e:
            return f"Web search error: HTTP failure ({e})."
        except Exception as e:
            return f"Web search error: {str(e)}"

    def _normalize_time_range(self, time_range: str | None) -> tuple[str | None, str | None]:
        """Convert user-friendly time_range into DuckDuckGo df param"""
        if not time_range:
            return None, None

        normalized = time_range.strip().lower()
        mapped = self._TIME_RANGE_MAP.get(normalized)
        return mapped, normalized if mapped else None

    def _search_tavily(self, query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
        """Search using Tavily API (requires TAVILY_API_KEY)"""
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        }

        headers = {"Content-Type": "application/json"}
        client_kwargs = _get_http_client_kwargs(timeout=10)
        with httpx.Client(**client_kwargs) as client:
            response = client.post("https://api.tavily.com/search", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results: list[dict[str, str]] = []
        for item in data.get("results", [])[:max_results]:
            results.append({
                "title": item.get("title") or item.get("url", ""),
                "snippet": item.get("content") or item.get("snippet") or "",
                "url": item.get("url", ""),
            })

        if not results and data.get("answer"):
            results.append({
                "title": "Summary",
                "snippet": data["answer"],
                "url": "",
            })

        return results[:max_results]

    def _clean_html(self, raw: str) -> str:
        """Strip HTML tags/whitespace and unescape entities"""
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return html.unescape(text)

    def _format_results(
        self,
        query: str,
        results: list[dict[str, str]],
        time_range: str | None,
        provider: str | None,
        invalid_time_range: str | None = None,
        notes: list[str] | None = None,
    ) -> str:
        """Format results into human-readable output"""
        header = f"Web search results for: '{query}'"
        if time_range:
            header += f" (time range: {time_range})"
        if provider:
            header += f" [provider: {provider}]"

        output = [header + "\n"]
        if invalid_time_range:
            output.append(
                f"Note: time_range '{invalid_time_range}' is not supported; returning unfiltered results.\n"
            )
        if notes:
            for note in notes:
                output.append(f"Note: {note}")
            if notes:
                output.append("")

        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            if result.get("snippet"):
                snippet = result["snippet"].replace("\n", " ").strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                output.append(f"   {snippet}")
            if result.get("url"):
                output.append(f"   {result['url']}")
            output.append("")

        return "\n".join(output)
