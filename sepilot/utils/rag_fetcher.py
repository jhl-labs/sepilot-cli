"""RAG Content Fetcher

This module fetches content from URLs and processes it for RAG context.

Features:
- Fetch HTML/text content from URLs
- Convert HTML to markdown
- Extract main content (remove boilerplate)
- Cache fetched content
- Handle errors gracefully

Example:
    >>> fetcher = RAGContentFetcher()
    >>> content = await fetcher.fetch_url("https://docs.python.org/3/")
    >>> print(content[:100])
"""

import asyncio
import hashlib
import ipaddress
import logging
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import html2text
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False


logger = logging.getLogger(__name__)


class RAGContentFetcher:
    """Fetch and process content from URLs for RAG context."""

    def __init__(
        self,
        cache_dir: str | None = None,
        cache_ttl: int = 3600,  # 1 hour default
        max_content_length: int = 50000,  # 50KB default
        timeout: int = 30
    ):
        """Initialize RAG content fetcher.

        Args:
            cache_dir: Directory for caching fetched content
            cache_ttl: Cache time-to-live in seconds
            max_content_length: Maximum content length to fetch
            timeout: Request timeout in seconds
        """
        if cache_dir is None:
            cache_dir = str(Path.home() / ".sepilot" / "rag_cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ttl = cache_ttl
        self.max_content_length = max_content_length
        self.timeout = timeout

        # Check dependencies
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed. Install with: pip install aiohttp")
        if not HAS_BS4:
            logger.warning("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
        if not HAS_HTML2TEXT:
            logger.warning("html2text not installed. Install with: pip install html2text")

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL.

        Args:
            url: The URL

        Returns:
            Path to cache file
        """
        # Use hash of URL as filename
        url_hash = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()
        return self.cache_dir / f"{url_hash}.txt"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached content is still valid.

        Args:
            cache_path: Path to cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False

        # Check age
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime

        return age < timedelta(seconds=self.cache_ttl)

    def _read_cache(self, cache_path: Path) -> str | None:
        """Read content from cache.

        Args:
            cache_path: Path to cache file

        Returns:
            Cached content or None
        """
        try:
            with open(cache_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read cache: {e}")
            return None

    def _write_cache(self, cache_path: Path, content: str) -> None:
        """Write content to cache.

        Args:
            cache_path: Path to cache file
            content: Content to cache
        """
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write cache: {e}")

    _BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", "metadata.google.internal"}  # nosec B104
    _BLOCKED_NETWORKS = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
    ]

    def _is_safe_url(self, url: str) -> tuple[bool, str]:
        """Check if URL is safe to fetch (SSRF protection)."""
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False, f"Blocked scheme: {parsed.scheme}"

        hostname = parsed.hostname or ""
        if hostname.lower() in self._BLOCKED_HOSTS:
            return False, f"Blocked host: {hostname}"

        try:
            resolved_ips = socket.getaddrinfo(hostname, None)
            for _, _, _, _, sockaddr in resolved_ips:
                ip = ipaddress.ip_address(sockaddr[0])
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False, f"Blocked private/internal IP: {ip}"
                for network in self._BLOCKED_NETWORKS:
                    if ip in network:
                        return False, f"Blocked IP in network {network}: {ip}"
        except (socket.gaierror, ValueError):
            pass

        return True, ""

    async def fetch_url(self, url: str, use_cache: bool = True) -> str | None:
        """Fetch content from a URL.

        Args:
            url: The URL to fetch
            use_cache: Whether to use cached content

        Returns:
            Extracted text content or None on error
        """
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(url)
            if self._is_cache_valid(cache_path):
                logger.info(f"Using cached content for: {url}")
                return self._read_cache(cache_path)

        # SSRF protection: block internal/private URLs
        safe, reason = self._is_safe_url(url)
        if not safe:
            logger.error(f"URL blocked (SSRF protection): {reason}")
            return None

        # Fetch from URL
        if not HAS_AIOHTTP:
            logger.error("aiohttp not installed. Cannot fetch URL.")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch {url}: HTTP {response.status}")
                        return None

                    # Check content type
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type or "text/plain" in content_type:
                        html_content = await response.text()
                    else:
                        logger.warning(f"Unsupported content type: {content_type}")
                        return None

                    # Process content
                    if "text/html" in content_type:
                        text_content = self._html_to_text(html_content)
                    else:
                        text_content = html_content

                    # Truncate if too long
                    if len(text_content) > self.max_content_length:
                        logger.warning(f"Content truncated from {len(text_content)} to {self.max_content_length} chars")
                        text_content = text_content[:self.max_content_length] + "\n\n[Content truncated...]"

                    # Cache the result
                    if use_cache:
                        cache_path = self._get_cache_path(url)
                        self._write_cache(cache_path, text_content)

                    logger.info(f"Successfully fetched content from: {url} ({len(text_content)} chars)")
                    return text_content

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text.

        Args:
            html: HTML content

        Returns:
            Plain text content
        """
        if not HAS_BS4:
            # Fallback: strip tags naively
            import re
            text = re.sub(r'<[^>]+>', '', html)
            return text.strip()

        try:
            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Use html2text if available (better markdown conversion)
            if HAS_HTML2TEXT:
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.ignore_emphasis = False
                h.body_width = 0  # Don't wrap
                text = h.handle(str(soup))
            else:
                # Fallback to get_text()
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n")]
            lines = [line for line in lines if line]  # Remove empty lines
            text = "\n".join(lines)

            return text

        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            return ""

    async def fetch_multiple_urls(
        self,
        urls: list[str],
        use_cache: bool = True
    ) -> dict[str, str | None]:
        """Fetch content from multiple URLs concurrently.

        Args:
            urls: List of URLs to fetch
            use_cache: Whether to use cached content

        Returns:
            Dictionary mapping URLs to their content
        """
        tasks = [self.fetch_url(url, use_cache) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for url, result in zip(urls, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {url}: {result}")
                output[url] = None
            else:
                output[url] = result

        return output

    def clear_cache(self, url: str | None = None) -> None:
        """Clear cached content.

        Args:
            url: Specific URL to clear, or None to clear all
        """
        if url is None:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.txt"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete cache file {cache_file}: {e}")
            logger.info("Cleared all RAG cache")
        else:
            # Clear specific URL
            cache_path = self._get_cache_path(url)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.info(f"Cleared cache for: {url}")
                except Exception as e:
                    logger.error(f"Failed to delete cache for {url}: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        cache_files = list(self.cache_dir.glob("*.txt"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "total_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }
