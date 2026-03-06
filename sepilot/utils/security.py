"""Security utilities for SEPilot3

This module provides centralized security validation functions to prevent:
- Shell injection attacks
- Path traversal attacks
- Command injection
- Access to sensitive system files
"""

import re
import shlex
import tempfile
from pathlib import Path


class SecurityValidator:
    """Centralized security validation for SEPilot3"""

    # Dangerous command patterns for shell injection prevention
    # NOTE: These patterns are applied AFTER command normalization
    DANGEROUS_SHELL_PATTERNS = [
        # Destructive file operations - comprehensive rm patterns
        r'rm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+-[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*\s+-[a-zA-Z]*r[a-zA-Z]*|-[a-zA-Z]*rf[a-zA-Z]*|-[a-zA-Z]*fr[a-zA-Z]*|--recursive\s+--force|--force\s+--recursive)\s+/',
        r'rm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+-[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*\s+-[a-zA-Z]*r[a-zA-Z]*|-[a-zA-Z]*rf[a-zA-Z]*|-[a-zA-Z]*fr[a-zA-Z]*|--recursive\s+--force|--force\s+--recursive)\s',
        # Device access
        r'dd\s+if=/dev/(zero|random|urandom)',
        r'>\s*/dev/(sd[a-z]|hd[a-z]|nvme)',
        # Fork bombs and resource exhaustion
        r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;',
        # Filesystem operations
        r'mkfs\.',
        r'fdisk',
        r'parted',
        # Privilege escalation (including absolute paths)
        r'(?:^|[\s;|&])sudo\s+',
        r'(?:^|[\s;|&])su\s+',
        # Remote code execution
        r'curl.*\|.*sh',
        r'wget.*\|.*sh',
        r'curl.*\|.*bash',
        r'wget.*\|.*bash',
        # System path access
        r'/etc/(passwd|shadow|sudoers)',
        # Dangerous chmod operations
        r'chmod\s+(777|666)',
        # Background processes with dangerous commands
        r'&\s*rm\s+-rf',
        # Command chaining with dangerous patterns
        r'&&\s*rm\s+-rf\s+/',
        r';\s*rm\s+-rf\s+/',
    ]

    # Command injection patterns
    INJECTION_PATTERNS = [
        r'\$\(.*\)',  # Command substitution
        r'`.*`',       # Backtick command substitution
    ]

    # Safe commands allowed in command substitution
    SAFE_SUBSTITUTION_COMMANDS = ['pwd', 'date', 'whoami', 'hostname']

    # Forbidden paths for file operations
    FORBIDDEN_WRITE_PATHS = ['/etc', '/usr', '/bin', '/sbin', '/sys', '/proc', '/dev', '/boot', '/root']
    FORBIDDEN_READ_PATHS = ['/etc/shadow', '/etc/sudoers', '/root', '/sys', '/proc/kcore']

    # Safe system paths for reading (debugging purposes)
    SAFE_READ_PATHS = [tempfile.gettempdir(), '/var/log']

    @classmethod
    def _normalize_command(cls, command: str) -> str:
        """Normalize a command for pattern matching.

        Converts absolute paths to binary names (e.g. /usr/bin/sudo -> sudo),
        collapses redundant whitespace.

        Args:
            command: Raw command string

        Returns:
            Normalized command string
        """
        # Replace absolute paths to common binaries with just the binary name
        normalized = re.sub(r'(?:^|(?<=[\s;|&]))(/usr/local/s?bin/|/usr/s?bin/|/s?bin/)(\w+)', r'\2', command)
        # Collapse whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    @classmethod
    def _is_safe_substitution(cls, substitution: str) -> bool:
        """Check if a command substitution is safe using shlex tokenization.

        Args:
            substitution: The content inside $(...) or backticks

        Returns:
            True if the substitution is safe
        """
        # Strip $( ... ) or ` ... ` wrappers
        inner = substitution
        if inner.startswith("$(") and inner.endswith(")"):
            inner = inner[2:-1]
        elif inner.startswith("`") and inner.endswith("`"):
            inner = inner[1:-1]

        inner = inner.strip()
        if not inner:
            return True

        try:
            tokens = shlex.split(inner)
        except ValueError:
            # Malformed command — treat as unsafe
            return False

        if not tokens:
            return True

        # Extract the base command name (strip path)
        first_cmd = tokens[0].rsplit("/", 1)[-1]
        return first_cmd in cls.SAFE_SUBSTITUTION_COMMANDS

    @classmethod
    def validate_shell_command(cls, command: str) -> tuple[bool, str | None]:
        """Validate a shell command for security risks

        Args:
            command: Shell command to validate

        Returns:
            Tuple of (is_safe, error_message)
            - is_safe: True if command is safe, False otherwise
            - error_message: Error description if unsafe, None if safe
        """
        # Normalize the command (resolve absolute paths, collapse whitespace)
        normalized = cls._normalize_command(command)

        # Check dangerous patterns against normalized command
        for pattern in cls.DANGEROUS_SHELL_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return False, "Dangerous command pattern detected. Command blocked for security."

        # Check for command injection attempts
        for pattern in cls.INJECTION_PATTERNS:
            matches = re.findall(pattern, normalized)
            for match in matches:
                if not cls._is_safe_substitution(match):
                    # Block if contains dangerous operations
                    try:
                        tokens = shlex.split(match.strip("$()` "))
                        first_cmd = tokens[0].rsplit("/", 1)[-1] if tokens else ""
                    except ValueError:
                        first_cmd = ""
                    danger_cmds = ['rm', 'curl', 'wget', 'cat', 'sudo', 'su',
                                   'eval', 'base64', 'python', 'python3', 'perl',
                                   'ruby', 'node', 'sh', 'bash', 'zsh']
                    if first_cmd in danger_cmds or any(d in match.lower() for d in ['/etc/']):
                        return False, "Potential command injection detected. Command blocked for security."

        return True, None

    @classmethod
    def validate_file_path(cls, file_path: str, operation: str = "write") -> tuple[bool, str | None, Path | None]:
        """Validate a file path for security risks

        Args:
            file_path: File path to validate
            operation: Type of operation ("read" or "write")

        Returns:
            Tuple of (is_safe, error_message, resolved_path)
            - is_safe: True if path is safe, False otherwise
            - error_message: Error description if unsafe, None if safe
            - resolved_path: Resolved absolute path if safe, None if unsafe
        """
        try:
            # Resolve to absolute path
            path = Path(file_path).resolve()
            project_root = Path.cwd().resolve()

            # Check if path is within project root
            try:
                path.relative_to(project_root)
            except ValueError:
                # Path is outside project root
                if operation == "read":
                    # Allow reading from safe system paths
                    path_str = str(path)
                    if not any(path_str.startswith(safe_path) for safe_path in cls.SAFE_READ_PATHS):
                        return False, f"Path escape attempt detected. File path must be within project directory: {file_path}", None
                else:
                    # Never allow writing outside project root
                    return False, f"Path escape attempt detected. File path must be within project directory: {file_path}", None

            # Check forbidden paths based on operation
            path_str = str(path)
            if operation == "write":
                for forbidden in cls.FORBIDDEN_WRITE_PATHS:
                    if path_str.startswith(forbidden):
                        return False, f"Access to system path denied: {file_path}", None
            else:  # read
                for forbidden in cls.FORBIDDEN_READ_PATHS:
                    if path_str.startswith(forbidden):
                        return False, f"Access to sensitive system path denied: {file_path}", None

            return True, None, path

        except Exception as e:
            return False, f"Invalid file path: {str(e)}", None

    @classmethod
    def sanitize_environment_variable(cls, value: str, default: str = "") -> str:
        """Sanitize environment variable value

        Args:
            value: Environment variable value to sanitize
            default: Default value if sanitization fails

        Returns:
            Sanitized value or default
        """
        if not value or not isinstance(value, str):
            return default

        # Remove potentially dangerous characters
        dangerous_chars = [';', '&', '|', '$', '`', '\n', '\r']
        sanitized = value
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')

        return sanitized.strip() or default


# Convenience functions
def validate_shell_command(command: str) -> tuple[bool, str | None]:
    """Validate shell command - convenience wrapper"""
    return SecurityValidator.validate_shell_command(command)


def validate_file_path(file_path: str, operation: str = "write") -> tuple[bool, str | None, Path | None]:
    """Validate file path - convenience wrapper"""
    return SecurityValidator.validate_file_path(file_path, operation)


def sanitize_env_var(value: str, default: str = "") -> str:
    """Sanitize environment variable - convenience wrapper"""
    return SecurityValidator.sanitize_environment_variable(value, default)
