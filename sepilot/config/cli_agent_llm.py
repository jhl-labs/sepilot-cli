"""CLI Agent LLM wrapper — use CLI tools (claude, codex, etc.) as LLM backend.

Wraps CLI agents that support pipe mode (-p/-q) as a LangChain BaseChatModel.
Tool call instructions are injected into the prompt so that existing text-based
parsers (XML / JSON) in tool_call_fallback.py can extract them.
"""

import logging
import re
import shutil
import subprocess
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

# System prompt appended to CLI agents to make them act as a pure LLM backend.
_CLI_BACKEND_SYSTEM_PROMPT = (
    "IMPORTANT: You are being used as a backend LLM for another system. "
    "Do NOT use your own built-in tools. Do NOT ask for permissions. "
    "When you need to call a tool, output the tool call format EXACTLY as "
    "specified in the prompt. Output raw tool calls directly, no markdown wrapping."
)

# Per-CLI default configurations
CLI_AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "claude": {
        "flags": ["-p", "--tools", "", "--append-system-prompt", _CLI_BACKEND_SYSTEM_PROMPT],
        "format": "xml",
    },
    "opencode": {"flags": ["run", "--agent", "sepilot-backend"], "format": "json"},
    "codex": {"flags": ["exec", "-", "--sandbox", "read-only"], "format": "json"},
    "gemini": {"flags": ["--approval-mode", "yolo"], "format": "json", "stdin_as_flag": "-p"},
}

_TOOL_CALL_RULES = """
CRITICAL RULES:
- Call at most 3 tools per response. Wait for results before calling more.
- Use the EXACT tool names and parameter names listed above. Do NOT rename or abbreviate them.
- The [TOOL_RESULT] sections in the conversation are REAL results from previously executed tools. Do NOT re-read files that already have results.
- If a [TOOL_RESULT] shows file contents, the tool DID work. Do NOT retry it."""

_XML_TOOL_CALL_INSTRUCTION = """\
When you need to call a tool, you MUST use this exact format:
<function_calls>
<invoke name="tool_name">
<parameter name="param_name">value</parameter>
</invoke>
</function_calls>

Do NOT wrap tool calls in markdown code blocks. Output the XML directly.""" + _TOOL_CALL_RULES

_JSON_TOOL_CALL_INSTRUCTION = """\
When you need to call a tool, you MUST output a JSON block inside a markdown code fence:
```json
{"name": "tool_name", "arguments": {"param_name": "value"}}
```

For multiple tool calls, output each one in its own ```json ... ``` block.
Do NOT use any other format for tool calls.""" + _TOOL_CALL_RULES


class ChatCLIAgent(BaseChatModel):
    """LangChain BaseChatModel backed by a CLI agent subprocess."""

    cli_command: str = "claude"
    cli_flags: list[str] = ["-p"]
    model_name: str = ""
    timeout: int = 600
    tool_call_format: str = "xml"
    tool_schemas: list[Any] | None = None
    stdin_as_flag: str = ""  # If set, prompt is passed as this flag's value instead of stdin

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "cli_agent"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "cli_command": self.cli_command,
            "cli_flags": self.cli_flags,
            "model_name": self.model_name,
            "tool_call_format": self.tool_call_format,
        }

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Validate CLI is installed
        if not shutil.which(self.cli_command):
            from sepilot.config.llm_providers import LLMProviderError
            raise LLMProviderError(
                "cli_agent",
                f"'{self.cli_command}' not found in PATH",
                f"Install {self.cli_command} or check your PATH",
            )

        prompt_text = self._serialize_messages(messages)

        try:
            # Build command: some CLIs (gemini) need prompt as flag value, not stdin
            cmd = [self.cli_command] + self.cli_flags
            stdin_input = prompt_text
            if self.stdin_as_flag:
                cmd += [self.stdin_as_flag, prompt_text]
                stdin_input = None

            result = subprocess.run(
                cmd,
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"CLI agent '{self.cli_command}' timed out after {self.timeout}s"
            ) from exc

        stdout = (result.stdout or "").strip()
        if not stdout:
            stderr = (result.stderr or "").strip()
            error_detail = stderr or f"exit code {result.returncode}"
            raise RuntimeError(
                f"CLI agent '{self.cli_command}' returned empty response: {error_detail}"
            )

        # Strip CLI-specific noise from stdout
        stdout = self._clean_cli_output(stdout)

        message = AIMessage(content=stdout)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    # ------------------------------------------------------------------
    # Output cleaning
    # ------------------------------------------------------------------

    def _clean_cli_output(self, text: str) -> str:
        """Strip CLI-specific headers, footers, and ANSI codes from stdout."""
        # 1. Remove ANSI escape sequences
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)

        # 2. codex: strip header block (OpenAI Codex ... --------) and footer (tokens used ...)
        if self.cli_command == "codex":
            # Header: everything up to and including the "--------" separator after metadata
            text = re.sub(
                r"^OpenAI Codex.*?-{4,}\n(?:.*?\n)*?-{4,}\n",
                "",
                text,
                flags=re.DOTALL,
            )
            # Strip "user\n<prompt>\n" echo
            text = re.sub(r"^user\n.*?\n(?:mcp startup:.*?\n)?", "", text, flags=re.DOTALL)
            # Strip codex tool execution blocks (exec\n... succeeded in ...)
            text = re.sub(r"^exec\n.*?succeeded in \d+ms:\n.*?\n", "", text, flags=re.DOTALL | re.MULTILINE)
            # Footer: "tokens used\nN,NNN\n" at end
            text = re.sub(r"\ntokens used\n[\d,]+\n?$", "", text)
            # Strip leading "codex\n" label
            text = re.sub(r"^codex\n", "", text)

        # 3. opencode: strip header ("> agent-name · model")
        if self.cli_command == "opencode":
            text = re.sub(r"^>\s+\S+.*?\n", "", text)

        return text.strip()

    # ------------------------------------------------------------------
    # Message serialisation
    # ------------------------------------------------------------------

    def _serialize_messages(self, messages: list[BaseMessage]) -> str:
        """Convert LangChain messages to a single text prompt for CLI pipe mode."""
        parts: list[str] = []

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if isinstance(msg, SystemMessage):
                parts.append(f"[SYSTEM]\n{content}")
                # Inject tool schema + call format instructions after system message
                tool_block = self._format_tools_for_prompt(self.tool_schemas)
                if tool_block:
                    parts.append(tool_block)
                parts.append(self._get_tool_call_instruction())
            elif isinstance(msg, HumanMessage):
                parts.append(f"[USER]\n{content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"[ASSISTANT]\n{content}")
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None) or "unknown"
                parts.append(f"[TOOL_RESULT: {tool_name}]\n{content}")
            else:
                parts.append(f"[MESSAGE]\n{content}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Tool-call format instructions
    # ------------------------------------------------------------------

    def _get_tool_call_instruction(self) -> str:
        """Return tool-call format instructions matched to the CLI's base LLM."""
        if self.tool_call_format == "xml":
            return _XML_TOOL_CALL_INSTRUCTION
        return _JSON_TOOL_CALL_INSTRUCTION

    # ------------------------------------------------------------------
    # Tool schema formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tools_for_prompt(tools: list[Any] | None) -> str:
        """Convert tool schemas to a text description for prompt injection.

        Accepts LangChain tool objects (with .name, .description,
        .get_input_schema()) or raw dicts.
        """
        if not tools:
            return ""

        lines = ["═══ AVAILABLE TOOLS ═══"]
        for tool in tools:
            # Support both LangChain tool objects and dicts
            if hasattr(tool, "name"):
                name = tool.name
                desc = getattr(tool, "description", "").split("\n")[0]
                try:
                    schema = tool.get_input_schema().model_json_schema()
                    props = schema.get("properties", {})
                    required = set(schema.get("required", []))
                except Exception:
                    props = {}
                    required = set()
            else:
                name = tool.get("name", "unknown")
                desc = tool.get("description", "").split("\n")[0]
                params = tool.get("parameters", {})
                props = params.get("properties", {}) if isinstance(params, dict) else {}
                required = set(params.get("required", [])) if isinstance(params, dict) else set()

            # Filter out Pydantic v2 internal fields (e.g. v__args, v__kwargs)
            props = {k: v for k, v in props.items() if not k.startswith("v__")}
            required = {r for r in required if not r.startswith("v__")}

            lines.append(f"\n- {name}: {desc}")
            for pname, pinfo in props.items():
                req_mark = " (required)" if pname in required else ""
                pdesc = pinfo.get("description", pinfo.get("title", ""))
                ptype = pinfo.get("type", "")
                default = pinfo.get("default")
                default_str = f", default={default}" if default is not None else ""
                lines.append(f"    {pname} [{ptype}{default_str}]{req_mark}: {pdesc}")

        return "\n".join(lines)
