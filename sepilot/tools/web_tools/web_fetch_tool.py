"""Enhanced web fetching tool with AI content processing"""

import hashlib
import ipaddress
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from sepilot.tools.base_tool import BaseTool

# SSRF protection: blocked hosts and networks
_BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]",  # nosec B104
    "metadata.google.internal",
}
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),         # Private
    ipaddress.ip_network("172.16.0.0/12"),      # Private
    ipaddress.ip_network("192.168.0.0/16"),     # Private
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / AWS metadata
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),           # IPv6 private
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
]


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Check if URL is safe from SSRF attacks.

    Returns:
        (is_safe, reason) tuple
    """
    parsed = urlparse(url)

    # Only allow http/https schemes
    if parsed.scheme not in ("http", "https"):
        return False, f"Blocked scheme: {parsed.scheme}"

    hostname = parsed.hostname or ""

    # Check blocked hostnames
    if hostname.lower() in _BLOCKED_HOSTS:
        return False, f"Blocked host: {hostname}"

    # Resolve hostname and check IP ranges
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in _BLOCKED_NETWORKS:
                if ip in network:
                    return False, f"Blocked IP range: {ip} ({network})"
    except (socket.gaierror, ValueError):
        pass  # DNS resolution failure will be caught by httpx

    return True, ""


def _get_http_client_kwargs(timeout: int) -> dict:
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


class WebFetchTool(BaseTool):
    """Tool for fetching and processing web content with AI"""

    name = "web_fetch"
    description = "Fetch web content and process it with AI prompts"
    parameters = {
        "url": "URL to fetch content from (required)",
        "prompt": "AI prompt to process the content (required)",
        "timeout": "Request timeout in seconds (default: 30)",
        "use_cache": "Use cached content if available (default: True)"
    }

    def __init__(self, logger=None):
        super().__init__(logger)
        self.cache_dir = Path.home() / ".sepilot" / "web_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = 900  # 15 minutes

    def execute(
        self,
        url: str,
        prompt: str,
        timeout: int = 30,
        use_cache: bool = True
    ) -> str:
        """Fetch and process web content"""
        self.validate_params(url=url, prompt=prompt)

        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme:
                url = f"https://{url}"
            elif parsed.scheme == "http":
                # Upgrade to HTTPS
                url = url.replace("http://", "https://", 1)

            # SSRF protection: block internal/private network access
            is_safe, reason = _is_safe_url(url)
            if not is_safe:
                return f"Error: URL blocked for security ({reason})"

            # Check cache
            if use_cache:
                cached_content = self._get_cached_content(url)
                if cached_content:
                    return self._process_content(cached_content, prompt)

            # Fetch content
            content = self._fetch_url(url, timeout, prompt)

            # Cache the content
            if use_cache:
                self._cache_content(url, content)

            # Process with prompt
            return self._process_content(content, prompt)

        except httpx.HTTPError as e:
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            return f"Error processing web content: {str(e)}"

    def _fetch_url(self, url: str, timeout: int, prompt: str = "") -> str:
        """Fetch content from URL"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SepilotBot/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        client_kwargs = _get_http_client_kwargs(timeout)
        with httpx.Client(**client_kwargs) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            # Check for redirects to different hosts
            final_url = str(response.url)
            original_host = urlparse(url).netloc
            final_host = urlparse(final_url).netloc

            if original_host != final_host:
                return (
                    f"⚠️ Redirect detected to different host\n"
                    f"Original: {original_host}\n"
                    f"Redirected to: {final_host}\n\n"
                    f"To fetch from the redirected URL, use:\n"
                    f"web_fetch(url=\"{final_url}\", prompt=\"{prompt}\")"
                )

            # Convert HTML to markdown-like format
            content = response.text
            return self._html_to_markdown(content)

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to simplified markdown"""
        try:
            import re
            from html.parser import HTMLParser

            class HTMLToMarkdown(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.output = []
                    self.in_pre = False
                    self.in_code = False
                    self.list_stack = []

                def handle_starttag(self, tag, attrs):
                    if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        level = int(tag[1])
                        self.output.append('\n' + '#' * level + ' ')
                    elif tag == 'p':
                        self.output.append('\n\n')
                    elif tag == 'br':
                        self.output.append('\n')
                    elif tag == 'pre':
                        self.in_pre = True
                        self.output.append('\n```\n')
                    elif tag == 'code' and not self.in_pre:
                        self.in_code = True
                        self.output.append('`')
                    elif tag in ['ul', 'ol']:
                        self.list_stack.append(tag)
                        self.output.append('\n')
                    elif tag == 'li':
                        if self.list_stack and self.list_stack[-1] == 'ul':
                            self.output.append('\n- ')
                        else:
                            self.output.append('\n1. ')
                    elif tag == 'a':
                        for attr in attrs:
                            if attr[0] == 'href':
                                self.output.append('[')
                                self._current_href = attr[1]
                                break
                    elif tag == 'strong' or tag == 'b':
                        self.output.append('**')
                    elif tag == 'em' or tag == 'i':
                        self.output.append('*')

                def handle_endtag(self, tag):
                    if tag == 'pre':
                        self.in_pre = False
                        self.output.append('\n```\n')
                    elif tag == 'code' and not self.in_pre:
                        self.in_code = False
                        self.output.append('`')
                    elif tag in ['ul', 'ol']:
                        if self.list_stack:
                            self.list_stack.pop()
                    elif tag == 'a':
                        if hasattr(self, '_current_href'):
                            self.output.append(f']({self._current_href})')
                            delattr(self, '_current_href')
                    elif tag == 'strong' or tag == 'b':
                        self.output.append('**')
                    elif tag == 'em' or tag == 'i':
                        self.output.append('*')

                def handle_data(self, data):
                    if self.in_pre:
                        self.output.append(data)
                    else:
                        # Clean up whitespace
                        data = re.sub(r'\s+', ' ', data)
                        self.output.append(data)

            # Remove script and style tags
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)

            parser = HTMLToMarkdown()
            parser.feed(html)

            markdown = ''.join(parser.output)
            # Clean up excessive newlines
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)
            return markdown.strip()

        except Exception:
            # Fallback: just strip HTML tags
            import re
            text = re.sub(r'<[^>]+>', '', html)
            return text.strip()

    def _process_content(self, content: str, prompt: str) -> str:
        """Process content with AI prompt"""
        # Try to use LLM if available
        try:
            from langchain.schema import HumanMessage, SystemMessage

            llm = self._create_content_processing_llm()
            if llm:
                messages = [
                    SystemMessage(content="You are a helpful assistant that extracts and summarizes information from web content."),
                    HumanMessage(content=f"Content:\n{content[:8000]}\n\nTask: {prompt}")
                ]

                response = llm.invoke(messages)
                return response.content

        except Exception as e:
            if self.logger:
                self.logger.log(f"LLM processing failed: {e}", level="debug")

        # Fallback: Basic text extraction based on prompt
        return self._basic_extraction(content, prompt)

    def _create_content_processing_llm(self):
        """Create a lightweight LLM for web content processing.

        Uses a smaller/faster model when available for efficiency.
        Falls back to main model settings if no lightweight option available.
        """
        try:
            from sepilot.config.settings import Settings
            settings = Settings()

            model_name = settings.model
            model_lower = model_name.lower()

            # Anthropic - use haiku for speed
            if any(x in model_lower for x in ["claude", "anthropic"]):
                api_key = settings.anthropic_api_key
                if not api_key:
                    return None
                from langchain_anthropic import ChatAnthropic
                # Prefer haiku for web content (fast & cheap)
                fast_model = "claude-3-haiku-20240307"
                return ChatAnthropic(
                    model=fast_model,
                    anthropic_api_key=api_key,
                    max_tokens=2000,
                    temperature=0.3,
                )

            # Google - use flash for speed
            elif any(x in model_lower for x in ["gemini", "google", "palm"]):
                api_key = settings.google_api_key
                if not api_key:
                    return None
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Prefer flash for web content
                fast_model = "gemini-1.5-flash"
                return ChatGoogleGenerativeAI(
                    model=fast_model,
                    google_api_key=api_key,
                    max_output_tokens=2000,
                    temperature=0.3,
                )

            # OpenAI - use gpt-4o-mini for speed
            elif any(model_lower.startswith(x) for x in ["gpt-3.5", "gpt-4", "o1-", "o3-"]):
                api_key = settings.openai_api_key
                if not api_key:
                    return None
                from langchain_openai import ChatOpenAI
                # Prefer gpt-4o-mini for web content
                fast_model = "gpt-4o-mini"
                llm_kwargs = {
                    "model": fast_model,
                    "openai_api_key": api_key,
                    "max_tokens": 2000,
                    "temperature": 0.3,
                }
                if settings.api_base_url:
                    from sepilot.config.llm_providers import _ensure_versioned_base_url
                    llm_kwargs["openai_api_base"] = _ensure_versioned_base_url(settings.api_base_url)
                return ChatOpenAI(**llm_kwargs)

            # Ollama - use same model (no standard lightweight option)
            elif settings.ollama_base_url:
                from langchain_openai import ChatOpenAI

                from sepilot.config.llm_providers import _ensure_versioned_base_url
                base_url = _ensure_versioned_base_url(settings.ollama_base_url)
                return ChatOpenAI(
                    model=model_name,
                    openai_api_key=settings.ollama_api_key or "ollama",
                    openai_api_base=base_url,
                    max_tokens=2000,
                    temperature=0.3,
                )

            # Default: OpenAI-compatible
            else:
                from langchain_openai import ChatOpenAI

                from sepilot.config.llm_providers import _ensure_versioned_base_url
                base_url = _ensure_versioned_base_url(
                    settings.api_base_url or "http://localhost:11434"
                )
                api_key = settings.openai_api_key or settings.ollama_api_key or "ollama"
                return ChatOpenAI(
                    model=model_name,
                    openai_api_key=api_key,
                    openai_api_base=base_url,
                    max_tokens=2000,
                    temperature=0.3,
                )

        except Exception as e:
            if self.logger:
                self.logger.log(f"Failed to create content processing LLM: {e}", level="debug")
            return None

    def _basic_extraction(self, content: str, prompt: str) -> str:
        """Basic content extraction without LLM"""
        prompt_lower = prompt.lower()

        # Limit content size
        if len(content) > 10000:
            content = content[:10000] + "\n\n[Content truncated...]"

        result = ["📄 Web Content (processed locally)\n"]
        result.append(f"Prompt: {prompt}\n")
        result.append("=" * 50 + "\n")

        # Extract based on common prompt patterns
        if any(word in prompt_lower for word in ['summary', 'summarize', 'overview']):
            # Get first few paragraphs
            lines = content.split('\n\n')[:5]
            result.append("Summary (first sections):\n")
            result.append('\n'.join(lines))

        elif any(word in prompt_lower for word in ['link', 'url', 'href']):
            # Extract links
            import re
            links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
            if links:
                result.append("Links found:\n")
                for text, url in links[:20]:
                    result.append(f"- [{text}]({url})")
            else:
                result.append("No markdown links found in content.")

        elif any(word in prompt_lower for word in ['code', 'snippet', 'example']):
            # Extract code blocks
            import re
            code_blocks = re.findall(r'```[^\n]*\n(.*?)\n```', content, re.DOTALL)
            if code_blocks:
                result.append("Code blocks found:\n")
                for i, code in enumerate(code_blocks[:5], 1):
                    result.append(f"\nCode block {i}:\n```\n{code}\n```")
            else:
                result.append("No code blocks found in content.")

        else:
            # Default: return truncated content
            result.append(content)

        return '\n'.join(result)

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()

    def _get_cached_content(self, url: str) -> str | None:
        """Get content from cache if valid"""
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.txt"

        if cache_file.exists():
            # Check if cache is still valid
            age = time.time() - cache_file.stat().st_mtime
            if age < self.cache_ttl:
                try:
                    return cache_file.read_text(encoding='utf-8')
                except Exception:
                    pass

        return None

    def _cache_content(self, url: str, content: str):
        """Cache content for URL"""
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.txt"

        try:
            cache_file.write_text(content, encoding='utf-8')
        except Exception:
            pass  # Ignore cache write errors
