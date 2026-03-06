"""Embedding Functions for RAG

Supports multiple embedding providers:
- OpenAI (text-embedding-3-small, ada-002)
- HuggingFace (sentence-transformers)
- Local models via sentence-transformers
"""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingFunction(ABC):
    """Abstract base class for embedding functions."""

    @abstractmethod
    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class OpenAIEmbedding(EmbeddingFunction):
    """OpenAI embedding function."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError as e:
            raise ImportError("openai package required. Install with: pip install openai") from e

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API."""
        if not texts:
            return []

        # OpenAI API call
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)


class HuggingFaceEmbedding(EmbeddingFunction):
    """HuggingFace/sentence-transformers embedding function."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            ) from e

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using sentence-transformers."""
        if not texts:
            return []

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class ChromaEmbeddingAdapter:
    """Adapter to make our EmbeddingFunction compatible with ChromaDB 1.3.x."""

    def __init__(self, embedding_fn: EmbeddingFunction):
        self._embedding_fn = embedding_fn

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function."""
        return "sepilot_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB calls with 'input' parameter."""
        return self._embedding_fn(input)

    def embed_query(self, input: list[str]):
        """Embed documents for ChromaDB 1.3.x."""
        import numpy as np
        embeddings = self._embedding_fn(input)
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def embed_with_retries(self, input: list[str], **retry_kwargs):
        """Embed with retries (delegates to embed_query)."""
        return self.embed_query(input)

    def is_legacy(self) -> bool:
        """Return False to use new ChromaDB 1.3.x interface."""
        return False

    def get_config(self):
        """Return configuration dict."""
        return {"name": self.name()}

    def default_space(self):
        """Return default distance space."""
        return "cosine"

    def supported_spaces(self):
        """Return supported distance spaces."""
        return ["cosine", "l2", "ip"]

    @classmethod
    def build_from_config(cls, config):
        """Build from config (not used but required)."""
        raise NotImplementedError("build_from_config not supported")

    @staticmethod
    def validate_config(config):
        """Validate config."""
        pass

    def validate_config_update(self, old_config, new_config):
        """Validate config update."""
        pass


def get_embedding_function(
    provider: str = "auto",
    model: str | None = None
) -> EmbeddingFunction:
    """Get an embedding function based on provider.

    Args:
        provider: "openai", "huggingface", or "auto" (tries openai first)
        model: Specific model name (optional)

    Returns:
        EmbeddingFunction instance
    """
    if provider == "auto":
        # Try OpenAI first if API key available
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIEmbedding(model=model or "text-embedding-3-small")
            except Exception as e:
                logger.warning(f"OpenAI embeddings failed: {e}, trying HuggingFace")

        # Fall back to HuggingFace (free, local)
        try:
            return HuggingFaceEmbedding(
                model_name=model or "sentence-transformers/all-MiniLM-L6-v2"
            )
        except ImportError as e:
            raise ImportError(
                "No embedding provider available. Install openai or sentence-transformers."
            ) from e

    elif provider == "openai":
        return OpenAIEmbedding(model=model or "text-embedding-3-small")

    elif provider == "huggingface":
        return HuggingFaceEmbedding(
            model_name=model or "sentence-transformers/all-MiniLM-L6-v2"
        )

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
