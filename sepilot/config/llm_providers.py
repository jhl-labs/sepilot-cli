"""LLM Provider Factory Pattern

Supports 9 LLM providers:
- OpenAI (GPT-3.5, GPT-4, O1, O3)
- Anthropic (Claude)
- Google (Gemini)
- Ollama (Local models)
- AWS Bedrock (Claude, Titan, etc.)
- Azure OpenAI
- OpenRouter
- Groq
- GitHub Models

Network Configuration:
- Proxy support via HTTP_PROXY, HTTPS_PROXY, NO_PROXY environment variables
- SSL verification control via SSL_VERIFY environment variable
- Custom CA certificates via SSL_CERT_FILE environment variable
- Configurable request timeout via REQUEST_TIMEOUT environment variable
"""

import logging
import os
import ssl
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

import httpx

from sepilot.config.settings import Settings

logger = logging.getLogger(__name__)


def _ensure_versioned_base_url(base_url: str) -> str:
    """Ensure base URL has a versioned path suffix for OpenAI compatibility.

    - Already versioned (e.g. /v1, /v4, /api/paas/v4) → keep as-is
    - No version suffix → append /v1
    """
    import re

    stripped = base_url.rstrip("/")
    if re.search(r"/v\d+$", stripped):
        return stripped
    return stripped + "/v1"


def create_http_client(settings: Settings, async_client: bool = False) -> httpx.Client | httpx.AsyncClient:
    """Create an HTTP client with proxy and SSL configuration.

    Args:
        settings: Application settings containing network configuration
        async_client: If True, create an AsyncClient instead of Client

    Returns:
        Configured httpx Client or AsyncClient
    """
    # Build proxy configuration
    proxies = None
    if settings.https_proxy or settings.http_proxy:
        proxies = {}
        if settings.http_proxy:
            proxies["http://"] = settings.http_proxy
        if settings.https_proxy:
            proxies["https://"] = settings.https_proxy

    # Build SSL configuration
    verify: bool | str | ssl.SSLContext = settings.ssl_verify
    if settings.ssl_cert_file and settings.ssl_verify:
        # Use custom CA bundle
        verify = settings.ssl_cert_file
    elif not settings.ssl_verify:
        verify = False
        logger.warning("SSL verification is disabled. This is insecure for production use.")

    # Build timeout
    timeout = httpx.Timeout(settings.request_timeout, connect=30.0)

    # Create client
    client_class = httpx.AsyncClient if async_client else httpx.Client
    client_kwargs = {
        "timeout": timeout,
        "follow_redirects": True,
    }

    if proxies:
        client_kwargs["proxy"] = proxies.get("https://") or proxies.get("http://")
        logger.debug(f"Using proxy: {client_kwargs['proxy']}")

    if verify is not True:
        client_kwargs["verify"] = verify

    return client_class(**client_kwargs)


def get_httpx_client_kwargs(settings: Settings) -> dict[str, Any]:
    """Get kwargs for httpx client configuration (for LangChain providers).

    Args:
        settings: Application settings

    Returns:
        Dictionary of kwargs to pass to httpx.Client
    """
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(settings.request_timeout, connect=30.0),
        "follow_redirects": True,
    }

    # Proxy configuration
    if settings.https_proxy or settings.http_proxy:
        proxy = settings.https_proxy or settings.http_proxy
        kwargs["proxy"] = proxy

    # SSL configuration
    if not settings.ssl_verify:
        kwargs["verify"] = False
    elif settings.ssl_cert_file:
        kwargs["verify"] = settings.ssl_cert_file

    return kwargs


class LLMProviderError(Exception):
    """LLM Provider initialization error"""

    def __init__(self, provider: str, message: str, suggestion: str | None = None):
        self.provider = provider
        self.suggestion = suggestion
        super().__init__(f"[{provider}] {message}")


class LLMProviderFactory:
    """Factory for creating LLM instances based on provider type"""

    # Model name patterns for provider detection
    # NOTE: Only use unambiguous patterns here.
    # Generic model names (llama-3, mixtral, gemma) are NOT provider-specific
    # and should NOT be listed - they could run on Ollama, Groq, etc.
    PROVIDER_PATTERNS = {
        "anthropic": ["claude", "anthropic"],
        "google": ["gemini", "google/", "palm"],
        "openai": ["gpt-3.5", "gpt-4", "o1-", "o3-", "chatgpt"],
        "bedrock": ["bedrock/", "amazon.", "anthropic.claude", "meta.llama"],
        "azure": ["azure/", "azure-"],
        "openrouter": ["openrouter/"],
        "groq": ["groq/"],
        "github": ["github/"],
    }

    def __init__(self, settings: Settings, console: Any | None = None):
        """Initialize LLM provider factory

        Args:
            settings: Application settings
            console: Rich console for output (optional)
        """
        self.settings = settings
        self.console = console

    def detect_provider(self, model_name: str) -> str:
        """Detect provider from model name.

        Detection priority:
        1. Explicit provider prefix (e.g., "groq/llama-3", "bedrock/claude")
        2. Unambiguous model name patterns (e.g., "claude-3", "gpt-4")
        3. Ollama base URL configured → ollama
        4. Provider-specific API key fallbacks (groq, openrouter)
        5. Custom base URL → openai_compatible
        6. Default → openai_compatible

        Args:
            model_name: Model identifier

        Returns:
            Provider name string
        """
        model_lower = model_name.lower()

        # 1. Check explicit provider prefix patterns
        for provider, patterns in self.PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if model_lower.startswith(pattern) or ("/" not in pattern and pattern in model_lower):
                    return provider

        # 2. Check Ollama base URL (user explicitly configured local server)
        if self.settings.ollama_base_url:
            return "ollama"

        # 3. Check provider-specific API keys for ambiguous model names
        # (e.g., "llama-3" could be Groq or Ollama - use API key as hint)
        if self.settings.groq_api_key:
            return "groq"

        if self.settings.openrouter_api_key:
            return "openrouter"

        # 4. Custom base URL → generic OpenAI-compatible
        if self.settings.api_base_url:
            return "openai_compatible"

        # 5. Default to OpenAI-compatible
        return "openai_compatible"

    def create_llm(self, model_name: str | None = None) -> "BaseChatModel":
        """Create LLM instance based on provider

        Args:
            model_name: Model name (uses settings.model if not provided)

        Returns:
            LangChain chat model instance
        """
        model_name = model_name or self.settings.model
        provider = self.detect_provider(model_name)

        # Clean model name (remove provider prefix if present)
        clean_model_name = self._clean_model_name(model_name, provider)

        # Get common LLM parameters
        llm_params = {
            "model": clean_model_name,
            "temperature": self.settings.temperature,
        }

        # Max tokens handling (different providers use different param names)
        max_tokens = self.settings.max_tokens or 4096

        # Provider-specific initialization
        provider_methods = {
            "anthropic": self._create_anthropic,
            "google": self._create_google,
            "openai": self._create_openai,
            "ollama": self._create_ollama,
            "bedrock": self._create_bedrock,
            "azure": self._create_azure,
            "openrouter": self._create_openrouter,
            "groq": self._create_groq,
            "github": self._create_github,
            "openai_compatible": self._create_openai_compatible,
        }

        # Pass custom headers from settings
        custom_headers = getattr(self.settings, 'custom_headers', None) or {}

        create_method = provider_methods.get(provider, self._create_openai_compatible)
        return create_method(clean_model_name, llm_params, max_tokens, custom_headers=custom_headers)

    def _clean_model_name(self, model_name: str, provider: str) -> str:
        """Remove provider prefix from model name"""
        prefixes = {
            "bedrock": "bedrock/",
            "azure": "azure/",
            "openrouter": "openrouter/",
            "groq": "groq/",
            "github": "github/",
        }
        prefix = prefixes.get(provider, "")
        if prefix and model_name.lower().startswith(prefix):
            return model_name[len(prefix):]
        return model_name

    def _needs_custom_http_client(self) -> bool:
        """Check if custom HTTP client is needed for proxy/SSL settings"""
        return bool(
            self.settings.http_proxy
            or self.settings.https_proxy
            or not self.settings.ssl_verify
            or self.settings.ssl_cert_file
        )

    def _create_anthropic(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create Anthropic/Claude LLM"""
        api_key = self.settings.anthropic_api_key
        if not api_key:
            raise LLMProviderError(
                "anthropic",
                "ANTHROPIC_API_KEY not found",
                "Set ANTHROPIC_API_KEY environment variable or in settings"
            )
        try:
            from langchain_anthropic import ChatAnthropic

            # Build kwargs with timeout
            llm_kwargs = {
                "model": model_name,
                "anthropic_api_key": api_key,
                "max_tokens": max_tokens,
                "temperature": params["temperature"],
                "timeout": float(self.settings.request_timeout),
            }

            if custom_headers:
                llm_kwargs["default_headers"] = custom_headers

            # Add custom HTTP client if proxy or SSL settings are configured
            if self._needs_custom_http_client():
                llm_kwargs["http_client"] = create_http_client(self.settings)

            return ChatAnthropic(**llm_kwargs)
        except ImportError as e:
            raise LLMProviderError(
                "anthropic",
                "langchain-anthropic not installed",
                "pip install langchain-anthropic"
            ) from e

    def _create_google(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create Google/Gemini LLM"""
        api_key = self.settings.google_api_key
        if not api_key:
            raise LLMProviderError(
                "google",
                "GOOGLE_API_KEY not found",
                "Set GOOGLE_API_KEY environment variable or in settings"
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm_kwargs = {
                "model": model_name,
                "google_api_key": api_key,
                "max_output_tokens": max_tokens,
                "temperature": params["temperature"],
                "timeout": self.settings.request_timeout,
            }

            # Google GenAI uses transport parameter for custom config
            # Note: Proxy/SSL may need additional configuration via google-generativeai library
            if self._needs_custom_http_client():
                logger.warning(
                    "Google GenAI may not fully support custom proxy/SSL settings. "
                    "Consider using GOOGLE_API_USE_CLIENT_CERTIFICATE or grpc proxy settings."
                )

            return ChatGoogleGenerativeAI(**llm_kwargs)
        except ImportError as e:
            raise LLMProviderError(
                "google",
                "langchain-google-genai not installed",
                "pip install langchain-google-genai"
            ) from e

    def _create_openai(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create OpenAI LLM"""
        api_key = self.settings.openai_api_key
        if not api_key:
            raise LLMProviderError(
                "openai",
                "OPENAI_API_KEY not found",
                "Set OPENAI_API_KEY environment variable or in settings"
            )
        try:
            from langchain_openai import ChatOpenAI

            llm_kwargs = {
                "model": model_name,
                "openai_api_key": api_key,
                "max_tokens": max_tokens,
                "temperature": params["temperature"],
                "request_timeout": self.settings.request_timeout,
            }

            if self.settings.api_base_url:
                llm_kwargs["openai_api_base"] = _ensure_versioned_base_url(self.settings.api_base_url)

            if custom_headers:
                llm_kwargs["default_headers"] = custom_headers

            # Add custom HTTP client for proxy/SSL support
            if self._needs_custom_http_client():
                llm_kwargs["http_client"] = create_http_client(self.settings)
                llm_kwargs["http_async_client"] = create_http_client(self.settings, async_client=True)

            return ChatOpenAI(**llm_kwargs)
        except ImportError as e:
            raise LLMProviderError(
                "openai",
                "langchain-openai not installed",
                "pip install langchain-openai"
            ) from e

    def _create_ollama(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create Ollama LLM"""
        from langchain_openai import ChatOpenAI

        base_url = self.settings.ollama_base_url
        if not base_url:
            base_url = "http://localhost:11434"

        # Ensure versioned path suffix for OpenAI compatibility
        base_url = _ensure_versioned_base_url(base_url)

        api_key = self.settings.ollama_api_key or os.getenv("OLLAMA_API_KEY", "ollama")

        llm_kwargs = {
            "model": model_name,
            "openai_api_key": api_key,
            "openai_api_base": base_url,
            "max_tokens": max_tokens,
            "temperature": params["temperature"],
            "request_timeout": self.settings.request_timeout,
        }

        if custom_headers:
            llm_kwargs["default_headers"] = custom_headers

        # Add custom HTTP client for proxy/SSL support
        if self._needs_custom_http_client():
            llm_kwargs["http_client"] = create_http_client(self.settings)
            llm_kwargs["http_async_client"] = create_http_client(self.settings, async_client=True)

        return ChatOpenAI(**llm_kwargs)

    def _create_bedrock(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create AWS Bedrock LLM"""
        if not self.settings.aws_access_key_id or not self.settings.aws_secret_access_key:
            raise LLMProviderError(
                "bedrock",
                "AWS credentials not found",
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
            )
        try:
            from langchain_aws import ChatBedrock
            import botocore.config

            # Configure boto client with timeout and proxy
            boto_config = botocore.config.Config(
                connect_timeout=30,
                read_timeout=self.settings.request_timeout,
                retries={"max_attempts": 3},
            )

            # Note: For proxy support with Bedrock, users should set:
            # AWS_CA_BUNDLE for custom CA certs
            # HTTP_PROXY/HTTPS_PROXY are handled by boto3/botocore automatically
            if self._needs_custom_http_client():
                if self.settings.ssl_cert_file:
                    os.environ.setdefault("AWS_CA_BUNDLE", self.settings.ssl_cert_file)
                if not self.settings.ssl_verify:
                    logger.warning(
                        "SSL verification disabled. For Bedrock, consider using AWS_CA_BUNDLE instead."
                    )

            return ChatBedrock(
                model_id=model_name,
                region_name=self.settings.aws_region or "us-east-1",
                model_kwargs={
                    "max_tokens": max_tokens,
                    "temperature": params["temperature"],
                },
                credentials_profile_name=None,  # Use explicit credentials
                config=boto_config,
            )
        except ImportError as e:
            raise LLMProviderError(
                "bedrock",
                "langchain-aws not installed",
                "pip install langchain-aws"
            ) from e

    def _create_azure(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create Azure OpenAI LLM"""
        api_key = self.settings.azure_openai_api_key
        endpoint = self.settings.azure_openai_endpoint

        if not api_key or not endpoint:
            raise LLMProviderError(
                "azure",
                "Azure OpenAI credentials not found",
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables"
            )
        try:
            from langchain_openai import AzureChatOpenAI

            # Extract deployment name from model name (e.g., "azure/my-gpt4-deployment")
            deployment_name = model_name

            llm_kwargs = {
                "deployment_name": deployment_name,
                "openai_api_key": api_key,
                "azure_endpoint": endpoint,
                "openai_api_version": self.settings.azure_openai_api_version or "2024-02-15-preview",
                "max_tokens": max_tokens,
                "temperature": params["temperature"],
                "request_timeout": self.settings.request_timeout,
            }

            if custom_headers:
                llm_kwargs["default_headers"] = custom_headers

            # Add custom HTTP client for proxy/SSL support
            if self._needs_custom_http_client():
                llm_kwargs["http_client"] = create_http_client(self.settings)
                llm_kwargs["http_async_client"] = create_http_client(self.settings, async_client=True)

            return AzureChatOpenAI(**llm_kwargs)
        except ImportError as e:
            raise LLMProviderError(
                "azure",
                "langchain-openai not installed",
                "pip install langchain-openai"
            ) from e

    def _create_openrouter(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create OpenRouter LLM"""
        api_key = self.settings.openrouter_api_key
        if not api_key:
            raise LLMProviderError(
                "openrouter",
                "OPENROUTER_API_KEY not found",
                "Set OPENROUTER_API_KEY environment variable or in settings"
            )
        try:
            from langchain_openai import ChatOpenAI

            # Allow custom OpenRouter base URL via environment variable
            base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

            # Merge OpenRouter-specific headers with custom headers
            headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://sepilot.ai"),
                "X-Title": "SE Pilot",
            }
            if custom_headers:
                headers.update(custom_headers)

            llm_kwargs = {
                "model": model_name,
                "openai_api_key": api_key,
                "openai_api_base": base_url,
                "max_tokens": max_tokens,
                "temperature": params["temperature"],
                "request_timeout": self.settings.request_timeout,
                "default_headers": headers,
            }

            # Add custom HTTP client for proxy/SSL support
            if self._needs_custom_http_client():
                llm_kwargs["http_client"] = create_http_client(self.settings)
                llm_kwargs["http_async_client"] = create_http_client(self.settings, async_client=True)

            return ChatOpenAI(**llm_kwargs)
        except ImportError as e:
            raise LLMProviderError(
                "openrouter",
                "langchain-openai not installed",
                "pip install langchain-openai"
            ) from e

    def _create_groq(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create Groq LLM"""
        api_key = self.settings.groq_api_key
        if not api_key:
            raise LLMProviderError(
                "groq",
                "GROQ_API_KEY not found",
                "Set GROQ_API_KEY environment variable or in settings"
            )
        try:
            from langchain_groq import ChatGroq

            llm_kwargs = {
                "model": model_name,
                "groq_api_key": api_key,
                "max_tokens": max_tokens,
                "temperature": params["temperature"],
                "timeout": self.settings.request_timeout,
            }

            if custom_headers:
                llm_kwargs["default_headers"] = custom_headers

            # Groq uses httpx internally, proxy/SSL configured via environment
            if self._needs_custom_http_client():
                logger.debug("Groq SDK uses httpx with system proxy/SSL settings from environment")

            return ChatGroq(**llm_kwargs)
        except ImportError as e:
            raise LLMProviderError(
                "groq",
                "langchain-groq not installed",
                "pip install langchain-groq"
            ) from e

    def _create_github(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create GitHub Models LLM"""
        api_key = self.settings.github_token
        if not api_key:
            raise LLMProviderError(
                "github",
                "GITHUB_TOKEN not found",
                "Set GITHUB_TOKEN environment variable or in settings"
            )
        try:
            from langchain_openai import ChatOpenAI

            # Allow custom GitHub Models base URL via environment variable
            base_url = _ensure_versioned_base_url(
                os.getenv("GITHUB_MODELS_API_BASE", "https://models.inference.ai.azure.com")
            )

            llm_kwargs = {
                "model": model_name,
                "openai_api_key": api_key,
                "openai_api_base": base_url,
                "max_tokens": max_tokens,
                "temperature": params["temperature"],
                "request_timeout": self.settings.request_timeout,
            }

            if custom_headers:
                llm_kwargs["default_headers"] = custom_headers

            # Add custom HTTP client for proxy/SSL support
            if self._needs_custom_http_client():
                llm_kwargs["http_client"] = create_http_client(self.settings)
                llm_kwargs["http_async_client"] = create_http_client(self.settings, async_client=True)

            return ChatOpenAI(**llm_kwargs)
        except ImportError as e:
            raise LLMProviderError(
                "github",
                "langchain-openai not installed",
                "pip install langchain-openai"
            ) from e

    def _create_openai_compatible(
        self, model_name: str, params: dict, max_tokens: int, custom_headers: dict | None = None
    ) -> "BaseChatModel":
        """Create OpenAI-compatible LLM (for vLLM, LocalAI, etc.)"""
        from langchain_openai import ChatOpenAI

        base_url = self.settings.api_base_url or self.settings.ollama_base_url
        if not base_url:
            base_url = "http://localhost:11434/v1"
            if self.console:
                self.console.print(
                    f"[yellow]Unknown model '{model_name}'. Trying local Ollama at {base_url}[/yellow]"
                )

        # Ensure versioned path suffix
        base_url = _ensure_versioned_base_url(base_url)

        api_key = (
            self.settings.openai_api_key
            or self.settings.ollama_api_key
            or os.getenv("OLLAMA_API_KEY", "ollama")
        )

        llm_kwargs = {
            "model": model_name,
            "openai_api_key": api_key,
            "openai_api_base": base_url,
            "max_tokens": max_tokens,
            "temperature": params["temperature"],
            "request_timeout": self.settings.request_timeout,
        }

        if custom_headers:
            llm_kwargs["default_headers"] = custom_headers

        # Add custom HTTP client for proxy/SSL support
        if self._needs_custom_http_client():
            llm_kwargs["http_client"] = create_http_client(self.settings)
            llm_kwargs["http_async_client"] = create_http_client(self.settings, async_client=True)

        return ChatOpenAI(**llm_kwargs)


def create_llm_from_settings(
    settings: Settings, console: Any | None = None
) -> "BaseChatModel":
    """Convenience function to create LLM from settings

    Args:
        settings: Application settings
        console: Rich console for output (optional)

    Returns:
        LangChain chat model instance
    """
    factory = LLMProviderFactory(settings, console)
    return factory.create_llm()


def get_provider_info() -> dict[str, dict[str, Any]]:
    """Get information about all supported providers

    Returns:
        Dictionary with provider info
    """
    return {
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"],
            "env_vars": ["OPENAI_API_KEY"],
            "install": "pip install langchain-openai",
        },
        "anthropic": {
            "name": "Anthropic",
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "env_vars": ["ANTHROPIC_API_KEY"],
            "install": "pip install langchain-anthropic",
        },
        "google": {
            "name": "Google",
            "models": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
            "env_vars": ["GOOGLE_API_KEY"],
            "install": "pip install langchain-google-genai",
        },
        "ollama": {
            "name": "Ollama",
            "models": ["llama3", "codellama", "mistral", "mixtral"],
            "env_vars": ["OLLAMA_BASE_URL"],
            "install": "Local - https://ollama.ai",
        },
        "bedrock": {
            "name": "AWS Bedrock",
            "models": ["anthropic.claude-3-sonnet", "amazon.titan-text"],
            "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
            "install": "pip install langchain-aws",
        },
        "azure": {
            "name": "Azure OpenAI",
            "models": ["Custom deployments"],
            "env_vars": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
            "install": "pip install langchain-openai",
        },
        "openrouter": {
            "name": "OpenRouter",
            "models": ["Various (aggregator)"],
            "env_vars": ["OPENROUTER_API_KEY"],
            "install": "pip install langchain-openai",
        },
        "groq": {
            "name": "Groq",
            "models": ["llama-3.1-70b", "mixtral-8x7b", "gemma-7b"],
            "env_vars": ["GROQ_API_KEY"],
            "install": "pip install langchain-groq",
        },
        "github": {
            "name": "GitHub Models",
            "models": ["gpt-4o", "Phi-3", "Llama-3"],
            "env_vars": ["GITHUB_TOKEN"],
            "install": "pip install langchain-openai",
        },
    }
