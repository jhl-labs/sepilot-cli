"""Model Profile Manager for dynamic LLM configuration

Manages model profiles for runtime LLM configuration changes.
Profiles are stored in ~/.sepilot/profiles/ as JSON files.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    # Core settings
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None

    # Advanced settings
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    # Custom headers
    custom_headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self, exclude_secrets: bool = False) -> dict[str, Any]:
        """Convert to dictionary, excluding None values and empty dicts.

        Args:
            exclude_secrets: If True, exclude api_key from output
        """
        result = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            if isinstance(value, dict) and not value:
                continue
            if exclude_secrets and key == "api_key":
                continue
            result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary, ignoring unknown keys for backward compatibility"""
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def merge(self, other: 'ModelConfig'):
        """Merge another config into this one (non-None values only)"""
        for key, value in asdict(other).items():
            if value is not None:
                setattr(self, key, value)


class ModelProfileManager:
    """Manages model configuration profiles"""

    def __init__(self, profile_dir: Path | None = None):
        """Initialize profile manager

        Args:
            profile_dir: Directory to store profiles (default: ~/.sepilot/profiles)
        """
        if profile_dir is None:
            profile_dir = Path.home() / ".sepilot" / "profiles"

        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        # Current active configuration
        self.current_config = ModelConfig()
        self._dirty_params: set[str] = set()

        # Load environment variables as defaults
        self._load_env_defaults()

    def _load_env_defaults(self):
        """Load default configuration from environment variables"""
        # Common environment variable names for different LLM providers
        env_mappings = {
            'base_url': ['OPENAI_API_BASE', 'LLM_BASE_URL', 'API_BASE_URL', 'OLLAMA_BASE_URL'],
            'model': ['OPENAI_MODEL', 'LLM_MODEL', 'MODEL_NAME', 'SEPILOT_MODEL'],
            'api_key': ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'LLM_API_KEY', 'API_KEY'],
            'temperature': ['LLM_TEMPERATURE', 'TEMPERATURE'],
            'max_tokens': ['LLM_MAX_TOKENS', 'MAX_TOKENS'],
        }

        for config_key, env_keys in env_mappings.items():
            for env_key in env_keys:
                value = os.getenv(env_key)
                if value:
                    # Type conversion
                    if config_key == 'temperature':
                        value = float(value)
                    elif config_key in ['max_tokens', 'top_k']:
                        value = int(value)

                    setattr(self.current_config, config_key, value)
                    break  # Use first found value

    def get_current_config(self) -> ModelConfig:
        """Get current active configuration"""
        return self.current_config

    def set_parameter(self, param: str, value: Any) -> bool:
        """Set a single parameter

        Args:
            param: Parameter name (base_url, model, api_key, temperature, etc.)
            value: Parameter value

        Returns:
            True if parameter was set successfully
        """
        # Normalize parameter name
        param = param.lower().replace('-', '_')

        if not hasattr(self.current_config, param):
            return False

        # Type conversion based on parameter
        try:
            if param == 'temperature' or param == 'top_p':
                value = float(value)
            elif param in ['top_k', 'max_tokens']:
                value = int(value)

            old_value = getattr(self.current_config, param)
            setattr(self.current_config, param, value)
            if old_value != value:
                self._dirty_params.add(param)
            return True
        except (ValueError, TypeError):
            return False

    def set_custom_header(self, key: str, value: str):
        """Set a custom HTTP header"""
        old_value = self.current_config.custom_headers.get(key)
        self.current_config.custom_headers[key] = value
        if old_value != value:
            self._dirty_params.add("custom_headers")

    def remove_custom_header(self, key: str) -> bool:
        """Remove a custom HTTP header"""
        if key in self.current_config.custom_headers:
            del self.current_config.custom_headers[key]
            self._dirty_params.add("custom_headers")
            return True
        return False

    def list_profiles(self) -> list[str]:
        """List all saved profiles

        Returns:
            List of profile names (without .json extension)
        """
        profiles = []
        for file in self.profile_dir.glob("*.json"):
            profiles.append(file.stem)
        return sorted(profiles)

    def save_profile(self, name: str) -> bool:
        """Save current configuration as a profile.

        Note: API keys are NOT saved to profile files for security.
        Use environment variables for API key management.

        Args:
            name: Profile name

        Returns:
            True if saved successfully
        """
        try:
            profile_path = self.profile_dir / f"{name}.json"
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_config.to_dict(exclude_secrets=True), f, indent=2)
            return True
        except Exception:
            return False

    def load_profile(self, name: str) -> bool:
        """Load a profile and replace current configuration.

        Replaces non-secret fields with profile values.
        API key is preserved from environment or current config since
        profiles don't store secrets.

        Args:
            name: Profile name

        Returns:
            True if loaded successfully
        """
        try:
            profile_path = self.profile_dir / f"{name}.json"
            if not profile_path.exists():
                return False

            with open(profile_path, encoding='utf-8') as f:
                data = json.load(f)

            previous_config = asdict(self.current_config)

            # Preserve current api_key (not saved in profiles)
            preserved_api_key = self.current_config.api_key

            # Replace config with profile (not merge)
            self.current_config = ModelConfig.from_dict(data)

            # Restore api_key from previous config or environment
            if not self.current_config.api_key:
                self.current_config.api_key = preserved_api_key

            current_config = asdict(self.current_config)
            for key in previous_config:
                if previous_config[key] != current_config[key]:
                    self._dirty_params.add(key)

            return True
        except Exception:
            return False

    def delete_profile(self, name: str) -> bool:
        """Delete a profile

        Args:
            name: Profile name

        Returns:
            True if deleted successfully
        """
        try:
            profile_path = self.profile_dir / f"{name}.json"
            if not profile_path.exists():
                return False

            profile_path.unlink()
            return True
        except Exception:
            return False

    def get_profile_info(self, name: str) -> dict[str, Any] | None:
        """Get profile information

        Args:
            name: Profile name

        Returns:
            Profile configuration dict or None if not found
        """
        try:
            profile_path = self.profile_dir / f"{name}.json"
            if not profile_path.exists():
                return None

            with open(profile_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def set_default_profile(self, name: str) -> bool:
        """Set a profile as the default to auto-load on startup

        Args:
            name: Profile name (must exist)

        Returns:
            True if set successfully
        """
        # Verify profile exists
        profile_path = self.profile_dir / f"{name}.json"
        if not profile_path.exists():
            return False

        try:
            default_path = self.profile_dir / ".default"
            default_path.write_text(name, encoding='utf-8')
            return True
        except Exception:
            return False

    def get_default_profile(self) -> str | None:
        """Get the default profile name

        Returns:
            Profile name or None if no default is set
        """
        try:
            default_path = self.profile_dir / ".default"
            if not default_path.exists():
                return None

            name = default_path.read_text(encoding='utf-8').strip()
            if not name:
                return None

            # Verify the profile still exists
            profile_path = self.profile_dir / f"{name}.json"
            if not profile_path.exists():
                return None

            return name
        except Exception:
            return None

    def clear_default_profile(self) -> bool:
        """Clear the default profile setting

        Returns:
            True if cleared successfully
        """
        try:
            default_path = self.profile_dir / ".default"
            if default_path.exists():
                default_path.unlink()
            return True
        except Exception:
            return False

    def reset_to_defaults(self):
        """Reset configuration to environment variable defaults"""
        previous_config = asdict(self.current_config)
        self.current_config = ModelConfig()
        self._load_env_defaults()
        current_config = asdict(self.current_config)
        for key in previous_config:
            if previous_config[key] != current_config[key]:
                self._dirty_params.add(key)

    def get_dirty_parameters(self) -> set[str]:
        """Return parameters changed since last apply."""
        return set(self._dirty_params)

    def clear_dirty_parameters(self):
        """Clear pending parameter-change tracking."""
        self._dirty_params.clear()
