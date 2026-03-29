"""Vector Store for RAG

ChromaDB-based vector storage for document embeddings.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sepilot.rag.chunker import Chunk
from sepilot.rag.embeddings import ChromaEmbeddingAdapter, EmbeddingFunction

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result from vector store."""
    content: str
    source: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """ChromaDB-based vector store for RAG."""

    def __init__(
        self,
        embedding_fn: EmbeddingFunction,
        persist_dir: str | None = None,
        collection_name: str = "sepilot_rag"
    ):
        """Initialize vector store.

        Args:
            embedding_fn: Embedding function to use
            persist_dir: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
        """
        self.embedding_fn = embedding_fn
        self.collection_name = collection_name

        if persist_dir is None:
            persist_dir = str(Path.home() / ".sepilot" / "rag_vectordb")

        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError(
                "chromadb package required. Install with: pip install chromadb"
            ) from e

        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create embedding function adapter for ChromaDB
        self._chroma_embedding = ChromaEmbeddingAdapter(embedding_fn)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._chroma_embedding,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            f"VectorStore initialized: {persist_dir}, "
            f"collection={collection_name}, documents={self.collection.count()}"
        )

    # ChromaDB batch size limit (embedding + upsert)
    _BATCH_SIZE = 100

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the vector store.

        Handles large batches by splitting into ChromaDB-friendly sizes
        and deduplicating against existing chunks.

        Args:
            chunks: List of Chunk objects to add

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Generate all IDs first, then batch-check existence
        all_chunk_ids = [self._generate_id(c.source, c.content) for c in chunks]

        # Batch check existence to avoid single large query
        existing_ids: set[str] = set()
        for i in range(0, len(all_chunk_ids), self._BATCH_SIZE):
            batch_ids = all_chunk_ids[i:i + self._BATCH_SIZE]
            try:
                existing_check = self.collection.get(ids=batch_ids)
                if existing_check and existing_check['ids']:
                    existing_ids.update(existing_check['ids'])
            except Exception as e:
                logger.warning(f"Error checking existing chunks: {e}")

        ids = []
        documents = []
        metadatas = []

        for chunk, chunk_id in zip(chunks, all_chunk_ids, strict=False):
            if chunk_id in existing_ids:
                logger.debug(f"Chunk already exists: {chunk_id}")
                continue

            # Sanitize metadata: ChromaDB only supports str/int/float/bool values
            safe_metadata = {
                "source": str(chunk.source),
                "chunk_index": int(chunk.chunk_index),
                "start_char": int(chunk.start_char),
                "end_char": int(chunk.end_char),
            }
            for k, v in (chunk.metadata or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    safe_metadata[k] = v
                else:
                    safe_metadata[k] = str(v)

            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(safe_metadata)

        if not ids:
            logger.info("All chunks already exist in vector store")
            return 0

        # Add in batches to respect ChromaDB limits
        total_added = 0
        for i in range(0, len(ids), self._BATCH_SIZE):
            batch_ids = ids[i:i + self._BATCH_SIZE]
            batch_docs = documents[i:i + self._BATCH_SIZE]
            batch_meta = metadatas[i:i + self._BATCH_SIZE]

            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                )
                total_added += len(batch_ids)
            except Exception as e:
                logger.error(f"Failed to add batch {i//self._BATCH_SIZE}: {e}")

        logger.info(f"Added {total_added} chunks to vector store")
        return total_added

    def search(
        self,
        query: str,
        n_results: int = 5,
        source_filter: str | None = None,
        score_threshold: float = 0.0
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query
            n_results: Maximum number of results
            source_filter: Filter by source URL/identifier
            score_threshold: Minimum similarity score (0-1, higher is better)

        Returns:
            List of SearchResult objects
        """
        if self.collection.count() == 0:
            logger.warning("Vector store is empty")
            return []

        # Build where filter
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}

        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to SearchResult objects
        search_results = []

        if results and results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
            distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)

            for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
                # Convert distance to similarity score (cosine distance to similarity)
                # ChromaDB returns L2 distance by default, we use cosine
                score = 1 - dist  # For cosine distance

                if score >= score_threshold:
                    search_results.append(SearchResult(
                        content=doc,
                        source=meta.get("source", "unknown"),
                        score=score,
                        metadata=meta
                    ))

        logger.debug(f"Search returned {len(search_results)} results for: {query[:50]}...")
        return search_results

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source.

        Args:
            source: Source URL/identifier

        Returns:
            Number of chunks deleted
        """
        # Get all chunks from this source
        results = self.collection.get(
            where={"source": source},
            include=["metadatas"]
        )

        if not results or not results['ids']:
            return 0

        ids_to_delete = results['ids']

        # Delete chunks
        self.collection.delete(ids=ids_to_delete)

        logger.info(f"Deleted {len(ids_to_delete)} chunks from source: {source}")
        return len(ids_to_delete)

    def get_sources(self) -> list[dict[str, Any]]:
        """Get all unique sources in the vector store.

        Returns:
            List of source info dictionaries
        """
        # Get all documents with metadata
        results = self.collection.get(include=["metadatas"])

        if not results or not results['metadatas']:
            return []

        # Group by source
        sources = {}
        for meta in results['metadatas']:
            source = meta.get("source", "unknown")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "chunk_count": 0,
                    "total_chunks": meta.get("total_chunks", 0)
                }
            sources[source]["chunk_count"] += 1

        return list(sources.values())

    def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics.

        Returns:
            Dictionary with stats
        """
        sources = self.get_sources()

        return {
            "total_chunks": self.collection.count(),
            "total_sources": len(sources),
            "sources": sources,
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_fn.dimension
        }

    def clear(self) -> int:
        """Clear all documents from the vector store.

        Returns:
            Number of documents deleted
        """
        count = self.collection.count()

        if count > 0:
            # Delete the collection and recreate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self._chroma_embedding,
                metadata={"hnsw:space": "cosine"}
            )

        logger.info(f"Cleared vector store: {count} documents deleted")
        return count

    def _generate_id(self, source: str, content: str) -> str:
        """Generate unique ID for a chunk."""
        hash_input = f"{source}:{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]
