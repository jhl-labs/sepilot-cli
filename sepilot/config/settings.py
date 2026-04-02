"""Settings and configuration for SE Pilot"""

import os

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Global settings for SE Pilot"""

    model_config = ConfigDict()

    # Model configuration
    model: str = Field(default="gpt-4-turbo-preview")
    max_tokens: int = Field(default=4000, ge=1, le=200000, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature (0.0-2.0)")
    context_window: int = Field(default=128000, ge=1024, le=2000000, description="Maximum context window size in tokens")
    prompt_profile: str = Field(default="default")

    # Execution settings
    # Higher default to match agent workflow depth (triage/plan/execute/verify)
    max_iterations: int = Field(default=15, ge=1, le=100, description="Maximum agent iterations")
    verbose: bool = Field(default=False)
    timeout: int = Field(default=300)  # seconds
    use_enhanced_state: bool = Field(default=True)  # Use enhanced state management
    enable_streaming: bool = Field(default=True, description="Enable token-by-token streaming output")
    load_project_instructions: bool = Field(
        default_factory=lambda: os.getenv("SEPILOT_LOAD_PROJECT_INSTRUCTIONS", "0").lower() in ("1", "true", "yes"),
        description="Enable automatic loading of project-local instruction/context files"
    )
    load_project_rules: bool = Field(
        default_factory=lambda: os.getenv("SEPILOT_LOAD_PROJECT_RULES", "0").lower() in ("1", "true", "yes"),
        description="Enable automatic loading of project-local rules"
    )

    # Graph mode: 'enhanced' (full 17-node pipeline) or 'fast' (minimal ReAct loop for small models)
    graph_mode: str = Field(
        default_factory=lambda: os.getenv("SEPILOT_GRAPH_MODE", "enhanced"),
        description="Graph mode: 'enhanced' (full pipeline) or 'fast' (minimal ReAct loop)"
    )

    # Human-in-the-loop settings
    max_interrupts: int = Field(default=20, description="Maximum number of user approval prompts before stopping")
    sensitive_tools: set = Field(
        default_factory=lambda: {"bash_execute", "file_write", "file_edit", "git", "web_search"},
        description="Tools requiring user approval in enhanced mode"
    )

    # Model tier routing: use cheaper/specialized models for specific tasks
    # Note: Settings extends BaseModel (not BaseSettings), so env_prefix doesn't auto-map.
    # Each field uses explicit default_factory=os.getenv() for environment variable support.
    triage_model: str | None = Field(
        default_factory=lambda: os.getenv("SEPILOT_TRIAGE_MODEL"),
        description="Cheaper model for triage/classification (e.g., gpt-4o-mini)"
    )
    verifier_model: str | None = Field(
        default_factory=lambda: os.getenv("SEPILOT_VERIFIER_MODEL"),
        description="Cheaper model for verification/completion checks (e.g., gpt-4o-mini)"
    )
    reasoning_model: str | None = Field(
        default_factory=lambda: os.getenv("SEPILOT_REASONING_MODEL"),
        description="Large model for planning/reflection/debate (e.g., o3, claude-opus). Falls back to main model."
    )
    quick_model: str | None = Field(
        default_factory=lambda: os.getenv("SEPILOT_QUICK_MODEL"),
        description="Fast cheap model for direct responses (e.g., gpt-4o-mini, haiku). Falls back to main model."
    )

    # Performance optimization settings (centralized cache configuration)
    tool_cache_size: int = Field(default=100, description="Maximum number of cached tool results")
    tool_cache_ttl: int = Field(default=300, description="Time-to-live for cached tool results (seconds)")
    prompt_cache_ttl: int = Field(default=3600, description="Time-to-live for cached prompts (seconds)")
    session_buffer_size: int = Field(default=1000, description="Maximum messages in session buffer")
    memory_snapshot_size: int = Field(default=100, description="Maximum memory snapshots to retain")
    auto_verify_tests: bool = Field(default=False, description="자동 검증 단계에서 테스트 실행 지시")
    test_command: str = Field(default="pytest", description="자동 테스트 실행 명령")
    auto_verify_lint: bool = Field(default=False, description="자동 검증 단계에서 린트 실행 지시")
    lint_command: str = Field(default="ruff check .", description="자동 린트 실행 명령")

    # Memory monitoring settings
    memory_threshold_mb: int = Field(default=500, description="Memory usage warning threshold (MB)")
    memory_check_interval: int = Field(default=10, description="Check memory every N iterations")

    # tmux agent orchestration settings
    tmux_default_agent: str = Field(default="claude", description="tmux 오케스트레이션 기본 에이전트")
    tmux_session_timeout: int = Field(default=600, description="tmux 에이전트 응답 대기 타임아웃 (초)")
    tmux_pane_width: int = Field(default=200, description="tmux pane 너비")
    tmux_pane_height: int = Field(default=50, description="tmux pane 높이")
    tmux_max_parallel: int = Field(default=3, ge=1, le=10, description="tmux 동시 실행 최대 세션 수")

    # API Keys (loaded from environment)
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or os.getenv("SEPILOT_OPENAI_API_KEY")
    )
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or os.getenv("SEPILOT_ANTHROPIC_API_KEY")
    )
    google_api_key: str | None = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("SEPILOT_GOOGLE_API_KEY")
    )

    # Custom API configuration (OpenRouter, local servers, etc.)
    api_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE") or os.getenv("LLM_BASE_URL") or os.getenv("API_BASE_URL")
    )

    # Ollama configuration
    ollama_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL")
    )
    ollama_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OLLAMA_API_KEY"),
        description="Ollama API key (if required, usually not needed)"
    )

    # AWS Bedrock configuration
    aws_access_key_id: str | None = Field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID")
    )
    aws_secret_access_key: str | None = Field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    aws_region: str = Field(
        default_factory=lambda: os.getenv("AWS_REGION", "us-east-1")
    )

    # Azure OpenAI configuration
    azure_openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY")
    )
    azure_openai_endpoint: str | None = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    azure_openai_api_version: str | None = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    )

    # OpenRouter configuration
    openrouter_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )

    # Groq configuration
    groq_api_key: str | None = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )

    # GitHub Models configuration
    github_token: str | None = Field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN")
    )

    # Theme configuration
    theme: str = Field(default="default", description="UI theme (default, dark, light, monokai)")

    # Vi mode (Vim key bindings in prompt)
    vi_mode: bool = Field(
        default_factory=lambda: os.getenv("SEPILOT_VI_MODE", "0") in ("1", "true", "yes"),
        description="Enable vi/vim key bindings in interactive prompt"
    )

    # Permission rules file
    permission_rules_file: str | None = Field(
        default_factory=lambda: os.getenv("SEPILOT_PERMISSION_RULES"),
        description="Path to custom permission rules file"
    )

    # Network configuration (proxy, SSL, timeout)
    http_proxy: str | None = Field(
        default_factory=lambda: os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
        description="HTTP proxy URL (e.g., http://proxy.example.com:8080)"
    )
    https_proxy: str | None = Field(
        default_factory=lambda: os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
        description="HTTPS proxy URL (e.g., http://proxy.example.com:8080)"
    )
    no_proxy: str | None = Field(
        default_factory=lambda: os.getenv("NO_PROXY") or os.getenv("no_proxy"),
        description="Comma-separated list of hosts to bypass proxy"
    )
    ssl_verify: bool = Field(
        default_factory=lambda: os.getenv("SSL_VERIFY", "true").lower() not in ("false", "0", "no"),
        description="Verify SSL certificates (set SSL_VERIFY=false to disable)"
    )
    ssl_cert_file: str | None = Field(
        default_factory=lambda: os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE"),
        description="Path to custom CA certificate bundle"
    )
    request_timeout: int = Field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60")),
        ge=10, le=600,
        description="HTTP request timeout in seconds (default: 60)"
    )

    # Custom HTTP headers for LLM API calls (set via /model header)
    custom_headers: dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers for LLM API calls")

    def get_llm_config(self) -> dict:
        """Get LLM configuration based on model type.

        Uses LLMProviderFactory.detect_provider() for consistent detection.
        """
        from sepilot.config.llm_providers import LLMProviderFactory

        config = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Use unified provider detection
        factory = LLMProviderFactory(self)
        provider = factory.detect_provider(self.model)
        config["provider"] = provider

        # Set provider-specific API key and base_url
        provider_keys = {
            "anthropic": ("anthropic_api_key", None),
            "google": ("google_api_key", None),
            "openai": ("openai_api_key", "api_base_url"),
            "ollama": ("ollama_api_key", "ollama_base_url"),
            "bedrock": ("aws_access_key_id", None),
            "azure": ("azure_openai_api_key", "azure_openai_endpoint"),
            "openrouter": ("openrouter_api_key", None),
            "groq": ("groq_api_key", None),
            "github": ("github_token", None),
            "openai_compatible": ("openai_api_key", "api_base_url"),
        }

        key_attr, url_attr = provider_keys.get(provider, ("openai_api_key", "api_base_url"))
        api_key = getattr(self, key_attr, None) if key_attr else None
        base_url = getattr(self, url_attr, None) if url_attr else None

        if api_key:
            config["api_key"] = api_key
        elif provider == "ollama":
            config["api_key"] = "ollama"
        if base_url:
            config["base_url"] = base_url

        return config
