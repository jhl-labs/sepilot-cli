"""Pattern-based permission rules system.

Provides fine-grained control over tool execution permissions using
command patterns similar to Claude Code's permission system.

Permission levels:
- DENY: Block execution completely
- ASK: Require user approval
- ALLOW: Execute without approval
"""

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for tool execution"""
    DENY = "deny"
    ASK = "ask"
    ALLOW = "allow"


@dataclass
class PermissionRule:
    """A single permission rule"""
    tool: str  # Tool name (e.g., "bash_execute", "file_write", "*")
    pattern: str  # Pattern to match (e.g., "rm -rf *", "git push*")
    permission: PermissionLevel
    description: str = ""
    priority: int = 0  # Higher priority rules are checked first

    def matches(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if this rule matches the given tool and arguments

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if rule matches
        """
        # Check tool name
        if self.tool != "*" and self.tool != tool_name:
            return False

        # If no pattern, match all calls to this tool
        if not self.pattern or self.pattern == "*":
            return True

        # Get the relevant value to match against
        match_value = self._get_match_value(tool_name, args)
        if match_value is None:
            return False

        # Match using fnmatch (glob-style patterns)
        return fnmatch.fnmatch(match_value, self.pattern)

    def _get_match_value(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Get the value to match against for a given tool

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            String value to match, or None
        """
        # Different tools have different primary arguments
        match_keys = {
            "bash_execute": ["command"],
            "file_write": ["file_path", "path"],
            "file_edit": ["file_path", "path"],
            "file_read": ["file_path", "path"],
            "git": ["command", "args"],
        }

        keys = match_keys.get(tool_name, ["command", "path", "file_path"])

        for key in keys:
            if key in args and args[key]:
                return str(args[key])

        # Fallback: join all string args
        string_args = [str(v) for v in args.values() if isinstance(v, str)]
        return " ".join(string_args) if string_args else None


@dataclass
class PermissionRuleSet:
    """Collection of permission rules with evaluation logic"""
    rules: list[PermissionRule] = field(default_factory=list)

    # Default sensitive tools that always require ASK
    DEFAULT_SENSITIVE_TOOLS = {"bash_execute", "file_write", "file_edit", "git", "web_search"}

    # Dangerous patterns that are always DENY
    DANGEROUS_PATTERNS = [
        # Destructive file operations
        ("bash_execute", "rm -rf /*"),
        ("bash_execute", "rm -rf /"),
        ("bash_execute", "mkfs*"),
        ("bash_execute", "dd if=*of=/dev/*"),
        # System operations
        ("bash_execute", "shutdown*"),
        ("bash_execute", "reboot*"),
        ("bash_execute", ":(){*"),  # Fork bomb
        # Dangerous chmod/chown
        ("bash_execute", "chmod -R 777 /*"),
        ("bash_execute", "chmod -R 777 /"),
        ("bash_execute", "chown -R *:* /*"),
        # Credential exposure
        ("bash_execute", "cat /etc/shadow*"),
        ("bash_execute", "cat /etc/passwd*"),
    ]

    # Safe patterns that can be ALLOW
    SAFE_PATTERNS = [
        # Read-only operations
        ("bash_execute", "ls *"),
        ("bash_execute", "cat *"),
        ("bash_execute", "grep *"),
        ("bash_execute", "find *"),
        ("bash_execute", "head *"),
        ("bash_execute", "tail *"),
        ("bash_execute", "wc *"),
        # Git read operations
        ("bash_execute", "git status*"),
        ("bash_execute", "git log*"),
        ("bash_execute", "git diff*"),
        ("bash_execute", "git branch*"),
        ("bash_execute", "git show*"),
    ]

    def __post_init__(self):
        """Add default dangerous rules"""
        # Add dangerous patterns as high-priority DENY rules
        for tool, pattern in self.DANGEROUS_PATTERNS:
            self.rules.append(PermissionRule(
                tool=tool,
                pattern=pattern,
                permission=PermissionLevel.DENY,
                description="Dangerous operation - blocked",
                priority=1000  # High priority
            ))

    def add_rule(self, rule: PermissionRule) -> None:
        """Add a new rule to the set"""
        self.rules.append(rule)
        # Sort by priority (descending)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> PermissionLevel:
        """Evaluate permission for a tool execution

        Priority order:
        1. Explicit rules (sorted by priority)
        2. Default sensitive tools -> ASK
        3. Everything else -> ALLOW

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Permission level
        """
        # Check explicit rules
        for rule in self.rules:
            if rule.matches(tool_name, args):
                logger.debug(
                    f"Permission rule matched: {rule.tool}:{rule.pattern} -> {rule.permission.value}"
                )
                return rule.permission

        # Default: sensitive tools require approval
        if tool_name in self.DEFAULT_SENSITIVE_TOOLS:
            return PermissionLevel.ASK

        # Everything else is allowed
        return PermissionLevel.ALLOW

    def is_dangerous(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
        """Check if an operation is dangerous (DENY)

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Tuple of (is_dangerous, reason)
        """
        for rule in self.rules:
            if rule.matches(tool_name, args) and rule.permission == PermissionLevel.DENY:
                return True, rule.description or f"Matched dangerous pattern: {rule.pattern}"

        return False, ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "rules": [
                {
                    "tool": r.tool,
                    "pattern": r.pattern,
                    "permission": r.permission.value,
                    "description": r.description,
                    "priority": r.priority,
                }
                for r in self.rules
                # Skip built-in dangerous patterns when serializing
                if r.priority < 1000
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PermissionRuleSet":
        """Create from dictionary"""
        rule_set = cls()
        for rule_data in data.get("rules", []):
            rule_set.add_rule(PermissionRule(
                tool=rule_data["tool"],
                pattern=rule_data["pattern"],
                permission=PermissionLevel(rule_data["permission"]),
                description=rule_data.get("description", ""),
                priority=rule_data.get("priority", 0),
            ))
        return rule_set


class PermissionManager:
    """Manages permission rules and evaluation

    Can load rules from:
    1. Default built-in rules
    2. User config file (~/.sepilot/permissions.json)
    3. Project-specific file (.sepilot/permissions.json)
    """

    def __init__(
        self,
        config_path: Path | None = None,
        project_path: Path | None = None,
        auto_persist: bool = True,
    ):
        """Initialize permission manager

        Args:
            config_path: Path to user config file
            project_path: Path to project root for project-specific rules
            auto_persist: If True, automatically save rules on add/remove
        """
        self.rule_set = PermissionRuleSet()
        self._auto_persist = auto_persist

        # Load user rules
        if config_path is None:
            config_path = Path.home() / ".sepilot" / "permissions.json"
        self.config_path = config_path

        # Load project rules
        if project_path is None:
            project_path = Path.cwd()
        self.project_path = project_path

        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from config files"""
        # User-level rules
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    data = json.load(f)
                    user_rules = PermissionRuleSet.from_dict(data)
                    for rule in user_rules.rules:
                        self.rule_set.add_rule(rule)
                    logger.info(f"Loaded {len(user_rules.rules)} user permission rules")
            except Exception as e:
                logger.warning(f"Failed to load user permission rules: {e}")

        # Project-level rules (higher priority)
        project_config = self.project_path / ".sepilot" / "permissions.json"
        if project_config.exists():
            try:
                with open(project_config, encoding="utf-8") as f:
                    data = json.load(f)
                    project_rules = PermissionRuleSet.from_dict(data)
                    for rule in project_rules.rules:
                        # Project rules get higher priority
                        rule.priority += 100
                        self.rule_set.add_rule(rule)
                    logger.info(f"Loaded {len(project_rules.rules)} project permission rules")
            except Exception as e:
                logger.warning(f"Failed to load project permission rules: {e}")

    def check_permission(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[PermissionLevel, str]:
        """Check permission for a tool execution

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Tuple of (permission_level, reason)
        """
        # First check for dangerous operations
        is_dangerous, reason = self.rule_set.is_dangerous(tool_name, args)
        if is_dangerous:
            return PermissionLevel.DENY, reason

        # Evaluate normal rules
        permission = self.rule_set.evaluate(tool_name, args)

        reasons = {
            PermissionLevel.DENY: "Blocked by permission rule",
            PermissionLevel.ASK: "Requires user approval",
            PermissionLevel.ALLOW: "Allowed",
        }

        return permission, reasons.get(permission, "")

    def _auto_save(self) -> None:
        """Persist rules to disk if auto_persist is enabled.

        Failures are logged but never raised to avoid breaking callers.
        """
        if not self._auto_persist:
            return
        try:
            self.save_rules()
        except Exception as exc:
            logger.warning("Auto-persist failed (non-fatal): %s", exc)

    def add_allow_rule(
        self, tool: str, pattern: str, description: str = ""
    ) -> None:
        """Add an ALLOW rule

        Args:
            tool: Tool name
            pattern: Pattern to match
            description: Rule description
        """
        self.rule_set.add_rule(PermissionRule(
            tool=tool,
            pattern=pattern,
            permission=PermissionLevel.ALLOW,
            description=description,
        ))
        self._auto_save()

    def add_ask_rule(
        self, tool: str, pattern: str, description: str = ""
    ) -> None:
        """Add an ASK rule

        Args:
            tool: Tool name
            pattern: Pattern to match
            description: Rule description
        """
        self.rule_set.add_rule(PermissionRule(
            tool=tool,
            pattern=pattern,
            permission=PermissionLevel.ASK,
            description=description,
        ))
        self._auto_save()

    def add_deny_rule(
        self, tool: str, pattern: str, description: str = ""
    ) -> None:
        """Add a DENY rule

        Args:
            tool: Tool name
            pattern: Pattern to match
            description: Rule description
        """
        self.rule_set.add_rule(PermissionRule(
            tool=tool,
            pattern=pattern,
            permission=PermissionLevel.DENY,
            description=description,
            priority=500,  # High priority for custom deny rules
        ))
        self._auto_save()

    def remove_rule(self, tool: str, pattern: str) -> bool:
        """Remove a user rule by tool name and pattern.

        Built-in dangerous rules (priority >= 1000) cannot be removed.

        Args:
            tool: Tool name
            pattern: Pattern string

        Returns:
            True if a rule was removed
        """
        for i, rule in enumerate(self.rule_set.rules):
            if rule.tool == tool and rule.pattern == pattern and rule.priority < 1000:
                del self.rule_set.rules[i]
                self._auto_save()
                return True
        return False

    def save_rules(self) -> None:
        """Save current rules to config file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.rule_set.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved permission rules to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save permission rules: {e}")
            raise

    def list_rules(self) -> list[dict[str, Any]]:
        """List all non-builtin rules

        Returns:
            List of rule dictionaries
        """
        return [
            {
                "tool": r.tool,
                "pattern": r.pattern,
                "permission": r.permission.value,
                "description": r.description,
            }
            for r in self.rule_set.rules
            if r.priority < 1000  # Skip built-in dangerous patterns
        ]


# Singleton instance
_permission_manager: PermissionManager | None = None


def get_permission_manager() -> PermissionManager:
    """Get or create the global permission manager

    Returns:
        PermissionManager instance
    """
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager()
    return _permission_manager


def check_tool_permission(
    tool_name: str, args: dict[str, Any]
) -> tuple[PermissionLevel, str]:
    """Convenience function to check tool permission

    Args:
        tool_name: Name of the tool
        args: Tool arguments

    Returns:
        Tuple of (permission_level, reason)
    """
    return get_permission_manager().check_permission(tool_name, args)
