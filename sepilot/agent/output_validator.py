"""Output format validation and retry for weak LLM outputs.

Weak models often break structured output formats (pipe-delimited, JSON).
This module validates output and retries with a correction prompt when needed.
"""

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

_logger = logging.getLogger(__name__)


class OutputValidator:
    """Validates structured LLM output and retries on format errors."""

    @staticmethod
    def validate_pipe_format(text: str, expected_parts: int) -> tuple[bool, list[str]]:
        """Validate pipe-delimited format (e.g. 'COMPLETE|reason|0.85').

        Args:
            text: Raw LLM output.
            expected_parts: Expected number of pipe-separated parts.

        Returns:
            Tuple of (is_valid, parts_list).
        """
        if not text or not text.strip():
            return False, []

        # Try to extract the first valid pipe-delimited line
        for line in text.strip().splitlines():
            line = line.strip()
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= expected_parts:
                return True, parts

        return False, []

    @staticmethod
    def validate_json(text: str) -> tuple[bool, dict[str, Any] | None]:
        """Validate and extract JSON from LLM output.

        Handles JSON wrapped in code blocks (```json ... ```).

        Args:
            text: Raw LLM output.

        Returns:
            Tuple of (is_valid, parsed_json_or_None).
        """
        if not text or not text.strip():
            return False, None

        content = text.strip()

        # Extract from code block if present
        match = re.search(r'```json\s*(.*?)```', content, re.DOTALL)
        if not match:
            match = re.search(r'```\s*(.*?)```', content, re.DOTALL)
        if match:
            content = match.group(1).strip()

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return True, parsed
            return False, None
        except (json.JSONDecodeError, ValueError):
            return False, None

    @staticmethod
    def retry_with_correction(
        llm: BaseChatModel,
        original_messages: list,
        original_response: str,
        error_desc: str,
        max_retries: int = 1,
    ) -> str | None:
        """Retry LLM call with a correction prompt.

        Args:
            llm: LangChain LLM instance.
            original_messages: The original messages sent to the LLM.
            original_response: The original (malformed) response.
            error_desc: Description of the format error.
            max_retries: Maximum number of retry attempts.

        Returns:
            Corrected response text, or None if retry fails.
        """
        for attempt in range(max_retries):
            correction_prompt = (
                f"Your previous response was malformed:\n"
                f"---\n{original_response[:500]}\n---\n\n"
                f"Error: {error_desc}\n\n"
                f"Please respond again using EXACTLY the required format. "
                f"Do not include any extra text."
            )
            try:
                retry_messages = original_messages + [
                    HumanMessage(content=correction_prompt)
                ]
                response = llm.invoke(retry_messages)
                if response and getattr(response, "content", None):
                    return response.content.strip()
            except Exception as e:
                _logger.debug(f"Retry attempt {attempt + 1} failed: {e}")
                continue

        return None
