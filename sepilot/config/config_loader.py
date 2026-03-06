"""Advanced configuration loading with variable substitution and schema generation.

Supports:
- Environment variable substitution: {env:VAR}, {env:VAR:default}
- File content substitution: {file:path}
- JSON Schema generation from Pydantic models
- Configuration validation
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration loading error"""
    pass


class ConfigLoader:
    """Advanced configuration loader with variable substitution

    Supports variable substitution syntax:
    - {env:VAR_NAME} - Environment variable (required)
    - {env:VAR_NAME:default} - Environment variable with default
    - {file:path/to/file} - File content
    - {file:path/to/file:encoding} - File content with encoding
    """

    # Pattern for variable substitution
    VAR_PATTERN = re.compile(r'\{(env|file):([^}:]+)(?::([^}]*))?\}')

    def __init__(self, base_path: Path | None = None):
        """Initialize config loader

        Args:
            base_path: Base path for relative file references
        """
        self.base_path = base_path or Path.cwd()

    def load_config(self, config_path: Path | str) -> dict[str, Any]:
        """Load configuration from JSON file with variable substitution

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            ConfigError: If configuration is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                raw_content = f.read()

            # Substitute variables in raw content
            substituted_content = self._substitute_variables(raw_content)

            # Parse JSON
            config = json.loads(substituted_content)

            # Remove $schema if present (for JSON Schema validation support)
            config.pop("$schema", None)

            return config

        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in {config_path}: {e}") from e
        except Exception as e:
            raise ConfigError(f"Failed to load config from {config_path}: {e}") from e

    def _substitute_variables(self, content: str) -> str:
        """Substitute all variables in content

        Args:
            content: Raw content with variable placeholders

        Returns:
            Content with variables substituted
        """
        def replacer(match: re.Match) -> str:
            var_type = match.group(1)
            var_name = match.group(2)
            default = match.group(3)

            if var_type == "env":
                return self._get_env_value(var_name, default)
            elif var_type == "file":
                encoding = default or "utf-8"
                return self._get_file_content(var_name, encoding)
            else:
                return match.group(0)  # Return original if unknown

        return self.VAR_PATTERN.sub(replacer, content)

    def _get_env_value(self, var_name: str, default: str | None) -> str:
        """Get environment variable value

        Args:
            var_name: Environment variable name
            default: Default value if not set

        Returns:
            Environment variable value

        Raises:
            ConfigError: If required variable is not set
        """
        value = os.getenv(var_name)

        if value is not None:
            return value

        if default is not None:
            return default

        raise ConfigError(
            f"Required environment variable not set: {var_name}\n"
            f"Set it with: export {var_name}=<value>"
        )

    def _get_file_content(self, file_path: str, encoding: str = "utf-8") -> str:
        """Get file content

        Args:
            file_path: Path to file (relative to base_path)
            encoding: File encoding

        Returns:
            File content

        Raises:
            ConfigError: If file not found or unreadable
        """
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_path / path

        if not path.exists():
            raise ConfigError(f"Referenced file not found: {path}")

        try:
            with open(path, encoding=encoding) as f:
                content = f.read()

            # Escape for JSON
            return json.dumps(content)[1:-1]  # Remove surrounding quotes

        except Exception as e:
            raise ConfigError(f"Failed to read file {path}: {e}") from e

    def validate_config(self, config: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        """Validate configuration against JSON Schema

        Args:
            config: Configuration dictionary
            schema: JSON Schema dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Check field types
        properties = schema.get("properties", {})
        for field, value in config.items():
            if field in properties:
                prop_schema = properties[field]
                expected_type = prop_schema.get("type")

                if expected_type and not self._check_type(value, expected_type):
                    errors.append(
                        f"Invalid type for {field}: expected {expected_type}, "
                        f"got {type(value).__name__}"
                    )

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type

        Args:
            value: Value to check
            expected_type: Expected type string

        Returns:
            True if type matches
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, allow

        return isinstance(value, expected_python_type)


class SchemaGenerator:
    """Generate JSON Schema from Pydantic models"""

    @staticmethod
    def generate_schema(model: type[BaseModel]) -> dict[str, Any]:
        """Generate JSON Schema from Pydantic model

        Args:
            model: Pydantic model class

        Returns:
            JSON Schema dictionary
        """
        return model.model_json_schema()

    @staticmethod
    def save_schema(
        model: type[BaseModel],
        output_path: Path | str,
        schema_id: str | None = None,
    ) -> None:
        """Generate and save JSON Schema to file

        Args:
            model: Pydantic model class
            output_path: Path to save schema
            schema_id: Optional $id for the schema
        """
        schema = SchemaGenerator.generate_schema(model)

        # Add $schema and optional $id
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        if schema_id:
            schema["$id"] = schema_id

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated JSON Schema at {output_path}")


def generate_settings_schema() -> dict[str, Any]:
    """Generate JSON Schema for Settings model

    Returns:
        JSON Schema dictionary
    """
    from sepilot.config.settings import Settings
    return SchemaGenerator.generate_schema(Settings)


def save_settings_schema(output_path: Path | str | None = None) -> Path:
    """Generate and save Settings JSON Schema

    Args:
        output_path: Optional output path (defaults to ~/.sepilot/config.schema.json)

    Returns:
        Path where schema was saved
    """
    from sepilot.config.settings import Settings

    if output_path is None:
        output_path = Path.home() / ".sepilot" / "config.schema.json"

    SchemaGenerator.save_schema(
        Settings,
        output_path,
        schema_id="https://sepilot.ai/config.schema.json"
    )

    return Path(output_path)


def load_config_with_substitution(config_path: Path | str) -> dict[str, Any]:
    """Convenience function to load config with variable substitution

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_config(config_path)


def create_example_config(output_path: Path | str | None = None) -> Path:
    """Create an example configuration file

    Args:
        output_path: Optional output path

    Returns:
        Path where example was created
    """
    if output_path is None:
        output_path = Path.home() / ".sepilot" / "config.example.json"

    output_path = Path(output_path)

    example_config = {
        "$schema": "https://sepilot.ai/config.schema.json",
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.7,
        "max_tokens": 4096,
        "anthropic_api_key": "{env:ANTHROPIC_API_KEY}",
        "openai_api_key": "{env:OPENAI_API_KEY:}",
        "google_api_key": "{env:GOOGLE_API_KEY:}",
        "theme": "default",
        "verbose": False,
        "enable_streaming": True,
        "_comment": "Use {env:VAR} for environment variables, {file:path} for file contents"
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)

    logger.info(f"Created example config at {output_path}")
    return output_path
