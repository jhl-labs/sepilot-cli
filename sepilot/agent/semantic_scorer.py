"""Semantic Scorer - Embedding-based relevance scoring for context selection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ScoredItem:
    """An item with its relevance score."""

    content: str
    file_path: str | None
    score: float
    source: str  # e.g., "semantic", "symbol", "dependency"
    metadata: dict[str, Any] | None = None

    def __lt__(self, other: "ScoredItem") -> bool:
        return self.score < other.score


class SemanticScorer:
    """Score context items by semantic relevance to a query.

    Uses embeddings to compute cosine similarity between query and content.
    Falls back to keyword matching when embeddings are unavailable.
    """

    def __init__(self, embedding_fn: Callable[[str], list[float]] | None = None):
        """Initialize the scorer.

        Args:
            embedding_fn: Optional function to compute embeddings
        """
        self._embedding_fn = embedding_fn
        self._cache: dict[str, list[float]] = {}
        self._max_cache_size = 1000

    def set_embedding_fn(self, fn: Callable[[str], list[float]]) -> None:
        """Set the embedding function.

        Args:
            fn: Function that takes text and returns embedding vector
        """
        self._embedding_fn = fn
        self._cache.clear()

    def score(self, query: str, content: str) -> float:
        """Score content relevance to query.

        Args:
            query: The search query
            content: Content to score

        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not query or not content:
            return 0.0

        if self._embedding_fn:
            try:
                return self._embedding_similarity(query, content)
            except Exception as e:
                logger.debug(f"Embedding failed, using keyword fallback: {e}")

        return self._keyword_similarity(query, content)

    def score_items(
        self, query: str, items: list[tuple[str, str | None, dict | None]]
    ) -> list[ScoredItem]:
        """Score multiple items and return sorted by relevance.

        Args:
            query: The search query
            items: List of (content, file_path, metadata) tuples

        Returns:
            List of ScoredItem sorted by score descending
        """
        scored = []
        for content, file_path, metadata in items:
            score = self.score(query, content)
            scored.append(
                ScoredItem(
                    content=content,
                    file_path=file_path,
                    score=score,
                    source="semantic",
                    metadata=metadata,
                )
            )

        return sorted(scored, reverse=True)

    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between embeddings."""
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        if not emb1 or not emb2:
            return 0.0

        return self._cosine_similarity(emb1, emb2)

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text (with caching)."""
        # Normalize text for caching
        cache_key = text[:500]  # Limit key size

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._embedding_fn:
            return []

        embedding = self._embedding_fn(text)

        # Cache management
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entries
            keys = list(self._cache.keys())
            for key in keys[: len(keys) // 2]:
                del self._cache[key]

        self._cache[cache_key] = embedding
        return embedding

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _keyword_similarity(self, query: str, content: str) -> float:
        """Fallback keyword-based similarity scoring."""
        # Tokenize
        query_tokens = self._tokenize(query)
        content_tokens = self._tokenize(content)

        if not query_tokens or not content_tokens:
            return 0.0

        # Create frequency maps
        query_freq = {}
        for token in query_tokens:
            query_freq[token] = query_freq.get(token, 0) + 1

        content_freq = {}
        for token in content_tokens:
            content_freq[token] = content_freq.get(token, 0) + 1

        # Calculate TF-IDF-like score
        score = 0.0
        for token, count in query_freq.items():
            if token in content_freq:
                # Boost for exact matches, especially for longer tokens
                token_weight = 1.0 + (len(token) / 10)
                score += content_freq[token] * token_weight

        # Normalize by query length
        max_score = sum(query_freq.values()) * 2  # Max possible
        if max_score > 0:
            score = min(score / max_score, 1.0)

        return score

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for keyword matching."""
        # Convert to lowercase
        text = text.lower()

        # Split on non-alphanumeric
        tokens = re.findall(r"\b\w+\b", text)

        # Filter short tokens and common words
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "is", "are",
            "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need",
            "if", "else", "then", "this", "that", "these", "those",
            "it", "its", "i", "you", "he", "she", "we", "they",
        }

        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract code-related entities from text.

        Args:
            text: Input text

        Returns:
            Dictionary with entity types and values
        """
        entities = {
            "files": [],
            "symbols": [],
            "classes": [],
            "functions": [],
        }

        # File paths
        file_pattern = r"[\w./\\]+\.\w{1,5}"
        entities["files"] = re.findall(file_pattern, text)

        # Symbols (CamelCase or snake_case)
        camel_case = r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b"
        snake_case = r"\b[a-z]+(?:_[a-z]+)+\b"
        all_symbols = re.findall(camel_case, text) + re.findall(snake_case, text)
        entities["symbols"] = list(set(all_symbols))

        # Class names (starts with uppercase)
        entities["classes"] = [s for s in entities["symbols"] if s[0].isupper()]

        # Function names (starts with lowercase or contains underscores)
        entities["functions"] = [s for s in entities["symbols"] if s[0].islower()]

        return entities

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


# Singleton instance
_scorer: SemanticScorer | None = None


def get_semantic_scorer() -> SemanticScorer:
    """Get the singleton SemanticScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = SemanticScorer()
    return _scorer
