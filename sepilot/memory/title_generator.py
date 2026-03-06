"""Session title generator - OpenCode style automatic title generation.

Generates short, descriptive titles for conversation sessions based on
the first few messages.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# Fallback title generation without LLM
def generate_title_fallback(messages: list[dict[str, Any]]) -> str:
    """Generate a title from messages without using LLM.

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        Generated title (max 80 chars)
    """
    # Find the first user message
    user_message = None
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "human" or role == "user":
            user_message = content
            break

    if not user_message:
        return "New conversation"

    # Clean up the message
    title = user_message.strip()

    # Remove file references (@file.py)
    title = re.sub(r'@[\w\-_/\.]+', '', title)

    # Remove markdown formatting
    title = re.sub(r'[#*`_\[\]]', '', title)

    # Remove URLs
    title = re.sub(r'https?://\S+', '', title)

    # Take first sentence or first 80 chars
    title = title.strip()

    # Split by sentence boundaries
    sentences = re.split(r'[.!?\n]', title)
    if sentences:
        title = sentences[0].strip()

    # Truncate to 80 chars
    if len(title) > 77:
        title = title[:77] + "..."

    return title or "New conversation"


async def generate_title_with_llm(
    messages: list[dict[str, Any]],
    llm: Any,
) -> str:
    """Generate a title using LLM.

    Args:
        messages: List of message dictionaries
        llm: LangChain LLM instance

    Returns:
        Generated title (max 80 chars)
    """
    try:
        # Build context from first few messages
        context_parts = []
        for msg in messages[:5]:  # Only use first 5 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:500]  # Truncate content
            context_parts.append(f"{role}: {content}")

        context = "\n".join(context_parts)

        # Create title generation prompt
        prompt = f"""Generate a very short title (max 10 words) for this conversation.
The title should capture the main topic or task.
Do NOT include quotes or punctuation.
Just output the title, nothing else.

Conversation:
{context}

Title:"""

        # Invoke LLM
        if hasattr(llm, 'ainvoke'):
            response = await llm.ainvoke(prompt)
        elif hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
        else:
            return generate_title_fallback(messages)

        # Extract title from response
        if hasattr(response, 'content'):
            title = response.content.strip()
        else:
            title = str(response).strip()

        # Clean up
        title = title.strip('"\'')
        title = re.sub(r'^Title:\s*', '', title, flags=re.IGNORECASE)

        # Truncate if needed
        if len(title) > 80:
            title = title[:77] + "..."

        return title or generate_title_fallback(messages)

    except Exception as e:
        logger.warning(f"LLM title generation failed: {e}")
        return generate_title_fallback(messages)


def generate_title_sync(
    messages: list[dict[str, Any]],
    llm: Any | None = None,
) -> str:
    """Synchronous title generation.

    Uses LLM if available, otherwise falls back to simple extraction.

    Args:
        messages: List of message dictionaries
        llm: Optional LangChain LLM instance

    Returns:
        Generated title
    """
    if not messages:
        return "New conversation"

    if llm is None:
        return generate_title_fallback(messages)

    try:
        import asyncio
        return asyncio.run(generate_title_with_llm(messages, llm))
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        return generate_title_fallback(messages)


class SessionTitleGenerator:
    """Manages automatic title generation for sessions.

    OpenCode style:
    - Auto-generates title after first exchange
    - Uses LLM for smart titles
    - Falls back to simple extraction
    """

    def __init__(self, llm: Any | None = None):
        """Initialize title generator.

        Args:
            llm: Optional LangChain LLM for smart titles
        """
        self.llm = llm
        self._cache: dict[str, str] = {}  # session_id -> title

    def set_llm(self, llm: Any) -> None:
        """Set the LLM for title generation."""
        self.llm = llm

    def generate(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        force: bool = False,
    ) -> str:
        """Generate title for a session.

        Args:
            session_id: Session identifier
            messages: Session messages
            force: Force regeneration even if cached

        Returns:
            Generated title
        """
        # Check cache
        if not force and session_id in self._cache:
            return self._cache[session_id]

        # Generate title
        title = generate_title_sync(messages, self.llm)

        # Cache it
        self._cache[session_id] = title

        return title

    def get_cached_title(self, session_id: str) -> str | None:
        """Get cached title for a session.

        Args:
            session_id: Session identifier

        Returns:
            Cached title or None
        """
        return self._cache.get(session_id)

    def set_title(self, session_id: str, title: str) -> None:
        """Manually set a title for a session.

        Args:
            session_id: Session identifier
            title: Title to set
        """
        self._cache[session_id] = title

    def clear_cache(self, session_id: str | None = None) -> None:
        """Clear title cache.

        Args:
            session_id: Specific session to clear, or None for all
        """
        if session_id:
            self._cache.pop(session_id, None)
        else:
            self._cache.clear()


# Singleton instance
_title_generator: SessionTitleGenerator | None = None


def get_title_generator() -> SessionTitleGenerator:
    """Get or create the global title generator.

    Returns:
        SessionTitleGenerator instance
    """
    global _title_generator
    if _title_generator is None:
        _title_generator = SessionTitleGenerator()
    return _title_generator
