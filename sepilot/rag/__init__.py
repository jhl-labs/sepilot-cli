"""RAG (Retrieval-Augmented Generation) Module

Real RAG implementation with:
- Vector DB (ChromaDB)
- Embeddings (OpenAI or HuggingFace)
- Document chunking (text and code-aware)
- Semantic search
- Local file and directory indexing
"""

from sepilot.rag.chunker import CodeAwareChunker, DocumentChunker
from sepilot.rag.embeddings import get_embedding_function
from sepilot.rag.manager import RAGManager
from sepilot.rag.vectorstore import VectorStore

__all__ = [
    "RAGManager",
    "VectorStore",
    "DocumentChunker",
    "CodeAwareChunker",
    "get_embedding_function",
]
