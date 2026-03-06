"""Theme system for SE Pilot UI.

Provides customizable color schemes for the CLI interface.
Built-in themes: default, dark, light, monokai
Supports custom theme definitions.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.style import Style
from rich.theme import Theme as RichTheme

logger = logging.getLogger(__name__)


@dataclass
class ThemeColors:
    """Color definitions for a theme"""

    # Primary colors
    primary: str = "cyan"
    secondary: str = "blue"
    accent: str = "magenta"

    # Status colors
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    info: str = "blue"

    # UI element colors
    prompt: str = "bold cyan"
    user_input: str = "white"
    assistant_output: str = "white"
    tool_name: str = "bold yellow"
    tool_output: str = "dim white"
    code: str = "bright_white on grey23"
    comment: str = "dim"
    highlight: str = "bold white"

    # Panel colors
    panel_border: str = "blue"
    panel_title: str = "bold cyan"
    header: str = "bold white"
    footer: str = "dim"

    # Syntax highlighting (code blocks)
    keyword: str = "bold magenta"
    string: str = "green"
    number: str = "cyan"
    function: str = "yellow"
    class_name: str = "bold yellow"
    variable: str = "white"
    operator: str = "red"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary"""
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "info": self.info,
            "prompt": self.prompt,
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,
            "tool_name": self.tool_name,
            "tool_output": self.tool_output,
            "code": self.code,
            "comment": self.comment,
            "highlight": self.highlight,
            "panel_border": self.panel_border,
            "panel_title": self.panel_title,
            "header": self.header,
            "footer": self.footer,
            "keyword": self.keyword,
            "string": self.string,
            "number": self.number,
            "function": self.function,
            "class_name": self.class_name,
            "variable": self.variable,
            "operator": self.operator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ThemeColors":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class UITheme:
    """Complete theme definition"""

    name: str
    description: str = ""
    colors: ThemeColors = field(default_factory=ThemeColors)
    syntax_theme: str = "monokai"  # Rich syntax highlighting theme

    def to_rich_theme(self) -> RichTheme:
        """Convert to Rich Theme object"""
        styles = {}
        for name, color in self.colors.to_dict().items():
            try:
                styles[name] = Style.parse(color)
            except Exception:
                logger.warning(f"Invalid color '{color}' for '{name}'")

        return RichTheme(styles)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "colors": self.colors.to_dict(),
            "syntax_theme": self.syntax_theme,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UITheme":
        """Create from dictionary"""
        colors = ThemeColors.from_dict(data.get("colors", {}))
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            colors=colors,
            syntax_theme=data.get("syntax_theme", "monokai"),
        )


# Built-in themes

DEFAULT_THEME = UITheme(
    name="default",
    description="Default SE Pilot theme",
    colors=ThemeColors(),
    syntax_theme="monokai",
)

DARK_THEME = UITheme(
    name="dark",
    description="Optimized for dark terminal backgrounds",
    colors=ThemeColors(
        primary="bright_cyan",
        secondary="bright_blue",
        accent="bright_magenta",
        success="bright_green",
        warning="bright_yellow",
        error="bright_red",
        info="bright_blue",
        prompt="bold bright_cyan",
        user_input="bright_white",
        assistant_output="white",
        tool_name="bold bright_yellow",
        tool_output="grey70",
        code="bright_white on grey15",
        comment="grey50",
        highlight="bold bright_white",
        panel_border="bright_blue",
        panel_title="bold bright_cyan",
        header="bold bright_white",
        footer="grey50",
        keyword="bold bright_magenta",
        string="bright_green",
        number="bright_cyan",
        function="bright_yellow",
        class_name="bold bright_yellow",
        variable="bright_white",
        operator="bright_red",
    ),
    syntax_theme="monokai",
)

LIGHT_THEME = UITheme(
    name="light",
    description="Optimized for light terminal backgrounds",
    colors=ThemeColors(
        primary="dark_cyan",
        secondary="blue",
        accent="dark_magenta",
        success="dark_green",
        warning="dark_orange",
        error="dark_red",
        info="blue",
        prompt="bold dark_cyan",
        user_input="black",
        assistant_output="grey23",
        tool_name="bold dark_orange",
        tool_output="grey37",
        code="black on grey89",
        comment="grey50",
        highlight="bold black",
        panel_border="blue",
        panel_title="bold dark_cyan",
        header="bold black",
        footer="grey50",
        keyword="bold dark_magenta",
        string="dark_green",
        number="dark_cyan",
        function="dark_orange",
        class_name="bold dark_orange",
        variable="black",
        operator="dark_red",
    ),
    syntax_theme="github-dark",
)

MONOKAI_THEME = UITheme(
    name="monokai",
    description="Monokai color scheme",
    colors=ThemeColors(
        primary="#66d9ef",  # Cyan
        secondary="#ae81ff",  # Purple
        accent="#f92672",  # Pink
        success="#a6e22e",  # Green
        warning="#e6db74",  # Yellow
        error="#f92672",  # Pink/Red
        info="#66d9ef",  # Cyan
        prompt="bold #66d9ef",
        user_input="#f8f8f2",  # White
        assistant_output="#f8f8f2",
        tool_name="bold #e6db74",
        tool_output="#75715e",  # Comment grey
        code="#f8f8f2 on #272822",
        comment="#75715e",
        highlight="bold #f8f8f2",
        panel_border="#ae81ff",
        panel_title="bold #66d9ef",
        header="bold #f8f8f2",
        footer="#75715e",
        keyword="bold #f92672",
        string="#e6db74",
        number="#ae81ff",
        function="#a6e22e",
        class_name="bold #a6e22e",
        variable="#f8f8f2",
        operator="#f92672",
    ),
    syntax_theme="monokai",
)

# OpenCode-style additional themes

CATPPUCCIN_THEME = UITheme(
    name="catppuccin",
    description="Catppuccin Mocha color scheme",
    colors=ThemeColors(
        primary="#89b4fa",  # Blue
        secondary="#cba6f7",  # Mauve
        accent="#f5c2e7",  # Pink
        success="#a6e3a1",  # Green
        warning="#f9e2af",  # Yellow
        error="#f38ba8",  # Red
        info="#89dceb",  # Sky
        prompt="bold #89b4fa",
        user_input="#cdd6f4",  # Text
        assistant_output="#cdd6f4",
        tool_name="bold #f9e2af",
        tool_output="#6c7086",  # Overlay0
        code="#cdd6f4 on #1e1e2e",  # Text on Base
        comment="#6c7086",
        highlight="bold #cdd6f4",
        panel_border="#cba6f7",
        panel_title="bold #89b4fa",
        header="bold #cdd6f4",
        footer="#6c7086",
        keyword="bold #cba6f7",
        string="#a6e3a1",
        number="#fab387",  # Peach
        function="#89b4fa",
        class_name="bold #f9e2af",
        variable="#cdd6f4",
        operator="#f38ba8",
    ),
    syntax_theme="monokai",
)

DRACULA_THEME = UITheme(
    name="dracula",
    description="Dracula color scheme",
    colors=ThemeColors(
        primary="#bd93f9",  # Purple
        secondary="#8be9fd",  # Cyan
        accent="#ff79c6",  # Pink
        success="#50fa7b",  # Green
        warning="#f1fa8c",  # Yellow
        error="#ff5555",  # Red
        info="#8be9fd",  # Cyan
        prompt="bold #bd93f9",
        user_input="#f8f8f2",  # Foreground
        assistant_output="#f8f8f2",
        tool_name="bold #f1fa8c",
        tool_output="#6272a4",  # Comment
        code="#f8f8f2 on #282a36",  # Foreground on Background
        comment="#6272a4",
        highlight="bold #f8f8f2",
        panel_border="#bd93f9",
        panel_title="bold #8be9fd",
        header="bold #f8f8f2",
        footer="#6272a4",
        keyword="bold #ff79c6",
        string="#f1fa8c",
        number="#bd93f9",
        function="#50fa7b",
        class_name="bold #8be9fd",
        variable="#f8f8f2",
        operator="#ff79c6",
    ),
    syntax_theme="dracula",
)

GRUVBOX_THEME = UITheme(
    name="gruvbox",
    description="Gruvbox dark color scheme",
    colors=ThemeColors(
        primary="#83a598",  # Blue
        secondary="#d3869b",  # Purple
        accent="#fe8019",  # Orange
        success="#b8bb26",  # Green
        warning="#fabd2f",  # Yellow
        error="#fb4934",  # Red
        info="#83a598",  # Blue
        prompt="bold #83a598",
        user_input="#ebdbb2",  # Foreground
        assistant_output="#ebdbb2",
        tool_name="bold #fabd2f",
        tool_output="#928374",  # Grey
        code="#ebdbb2 on #282828",  # Foreground on Background
        comment="#928374",
        highlight="bold #ebdbb2",
        panel_border="#d3869b",
        panel_title="bold #83a598",
        header="bold #ebdbb2",
        footer="#928374",
        keyword="bold #fb4934",
        string="#b8bb26",
        number="#d3869b",
        function="#fabd2f",
        class_name="bold #8ec07c",  # Aqua
        variable="#ebdbb2",
        operator="#fe8019",
    ),
    syntax_theme="gruvbox-dark",
)

TOKYO_NIGHT_THEME = UITheme(
    name="tokyo-night",
    description="Tokyo Night color scheme",
    colors=ThemeColors(
        primary="#7aa2f7",  # Blue
        secondary="#bb9af7",  # Purple
        accent="#7dcfff",  # Cyan
        success="#9ece6a",  # Green
        warning="#e0af68",  # Yellow
        error="#f7768e",  # Red
        info="#7dcfff",  # Cyan
        prompt="bold #7aa2f7",
        user_input="#c0caf5",  # Foreground
        assistant_output="#c0caf5",
        tool_name="bold #e0af68",
        tool_output="#565f89",  # Comment
        code="#c0caf5 on #1a1b26",  # Foreground on Background
        comment="#565f89",
        highlight="bold #c0caf5",
        panel_border="#bb9af7",
        panel_title="bold #7aa2f7",
        header="bold #c0caf5",
        footer="#565f89",
        keyword="bold #bb9af7",
        string="#9ece6a",
        number="#ff9e64",  # Orange
        function="#7aa2f7",
        class_name="bold #e0af68",
        variable="#c0caf5",
        operator="#f7768e",
    ),
    syntax_theme="monokai",
)

TRON_THEME = UITheme(
    name="tron",
    description="Tron Legacy inspired color scheme",
    colors=ThemeColors(
        primary="#6fc3df",  # Cyan
        secondary="#df740c",  # Orange
        accent="#ffffff",  # White
        success="#6fc3df",  # Cyan
        warning="#df740c",  # Orange
        error="#ff0000",  # Red
        info="#6fc3df",  # Cyan
        prompt="bold #6fc3df",
        user_input="#ffffff",  # White
        assistant_output="#a0a0a0",  # Grey
        tool_name="bold #df740c",
        tool_output="#4a4a4a",  # Dark grey
        code="#ffffff on #0c141f",  # White on dark blue
        comment="#4a4a4a",
        highlight="bold #ffffff",
        panel_border="#6fc3df",
        panel_title="bold #6fc3df",
        header="bold #ffffff",
        footer="#4a4a4a",
        keyword="bold #6fc3df",
        string="#df740c",
        number="#6fc3df",
        function="#ffffff",
        class_name="bold #df740c",
        variable="#a0a0a0",
        operator="#6fc3df",
    ),
    syntax_theme="monokai",
)

# Built-in themes registry
BUILTIN_THEMES: dict[str, UITheme] = {
    "default": DEFAULT_THEME,
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "monokai": MONOKAI_THEME,
    "catppuccin": CATPPUCCIN_THEME,
    "dracula": DRACULA_THEME,
    "gruvbox": GRUVBOX_THEME,
    "tokyo-night": TOKYO_NIGHT_THEME,
    "tron": TRON_THEME,
}


class ThemeManager:
    """Manages UI themes

    Handles loading, saving, and switching themes.
    """

    CUSTOM_THEMES_FILE = Path.home() / ".sepilot" / "themes.json"

    def __init__(self):
        """Initialize theme manager"""
        self._current_theme: UITheme = DEFAULT_THEME
        self._custom_themes: dict[str, UITheme] = {}
        self._load_custom_themes()

    def _load_custom_themes(self) -> None:
        """Load custom themes from file"""
        if not self.CUSTOM_THEMES_FILE.exists():
            return

        try:
            with open(self.CUSTOM_THEMES_FILE, encoding="utf-8") as f:
                data = json.load(f)

            for theme_data in data.get("themes", []):
                theme = UITheme.from_dict(theme_data)
                self._custom_themes[theme.name] = theme

            logger.debug(f"Loaded {len(self._custom_themes)} custom themes")

        except Exception as e:
            logger.warning(f"Failed to load custom themes: {e}")

    def _save_custom_themes(self) -> None:
        """Save custom themes to file"""
        try:
            self.CUSTOM_THEMES_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "themes": [
                    theme.to_dict() for theme in self._custom_themes.values()
                ]
            }

            with open(self.CUSTOM_THEMES_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save custom themes: {e}")

    def get_theme(self, name: str) -> UITheme | None:
        """Get a theme by name

        Args:
            name: Theme name

        Returns:
            UITheme or None
        """
        # Check custom themes first
        if name in self._custom_themes:
            return self._custom_themes[name]

        # Check built-in themes
        return BUILTIN_THEMES.get(name)

    def set_theme(self, name: str) -> bool:
        """Set the current theme

        Args:
            name: Theme name

        Returns:
            True if theme was set successfully
        """
        theme = self.get_theme(name)
        if theme:
            self._current_theme = theme
            logger.info(f"Theme set to: {name}")
            return True

        logger.warning(f"Theme not found: {name}")
        return False

    @property
    def current_theme(self) -> UITheme:
        """Get current theme"""
        return self._current_theme

    def list_themes(self) -> list[dict[str, str]]:
        """List all available themes

        Returns:
            List of theme info dictionaries
        """
        themes = []

        # Built-in themes
        for name, theme in BUILTIN_THEMES.items():
            themes.append({
                "name": name,
                "description": theme.description,
                "type": "builtin",
            })

        # Custom themes
        for name, theme in self._custom_themes.items():
            themes.append({
                "name": name,
                "description": theme.description,
                "type": "custom",
            })

        return themes

    def add_custom_theme(self, theme: UITheme) -> None:
        """Add a custom theme

        Args:
            theme: Theme to add
        """
        self._custom_themes[theme.name] = theme
        self._save_custom_themes()
        logger.info(f"Added custom theme: {theme.name}")

    def remove_custom_theme(self, name: str) -> bool:
        """Remove a custom theme

        Args:
            name: Theme name

        Returns:
            True if theme was removed
        """
        if name in self._custom_themes:
            del self._custom_themes[name]
            self._save_custom_themes()
            logger.info(f"Removed custom theme: {name}")
            return True
        return False

    def get_rich_theme(self) -> RichTheme:
        """Get current theme as Rich Theme object

        Returns:
            Rich Theme object
        """
        return self._current_theme.to_rich_theme()

    def get_style(self, name: str) -> str:
        """Get a style from the current theme

        Args:
            name: Style name (e.g., 'primary', 'error')

        Returns:
            Style string
        """
        colors = self._current_theme.colors.to_dict()
        return colors.get(name, "white")

    def get_syntax_theme(self) -> str:
        """Get the syntax highlighting theme name

        Returns:
            Syntax theme name for Rich Syntax
        """
        return self._current_theme.syntax_theme


# Singleton instance
_theme_manager: ThemeManager | None = None


def get_theme_manager() -> ThemeManager:
    """Get or create the global theme manager

    Returns:
        ThemeManager instance
    """
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager


def get_current_theme() -> UITheme:
    """Get the current theme

    Returns:
        Current UITheme
    """
    return get_theme_manager().current_theme


def set_theme(name: str) -> bool:
    """Set the current theme

    Args:
        name: Theme name

    Returns:
        True if successful
    """
    return get_theme_manager().set_theme(name)


def get_style(name: str) -> str:
    """Get a style from the current theme

    Args:
        name: Style name

    Returns:
        Style string
    """
    return get_theme_manager().get_style(name)
