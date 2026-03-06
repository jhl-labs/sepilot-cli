"""RAG Manager - Main interface for RAG operations

Orchestrates the full RAG pipeline:
1. Document ingestion: URL fetch → Chunk → Embed → Store
2. Query: Embed query → Search → Return relevant chunks
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sepilot.rag.chunker import CodeAwareChunker, DocumentChunker
from sepilot.rag.embeddings import EmbeddingFunction, get_embedding_function
from sepilot.rag.vectorstore import SearchResult, VectorStore

logger = logging.getLogger(__name__)


# Type alias for progress callback
ProgressCallback = Callable[[int, int, str], None]  # (current, total, message)


@dataclass
class DocumentInfo:
    """Information about an indexed document."""
    url: str
    title: str
    description: str
    chunk_count: int
    indexed_at: str
    last_updated: str
    status: str  # "indexed", "pending", "error"
    error_message: str | None = None


class RAGManager:
    """Main interface for RAG operations."""

    def __init__(
        self,
        embedding_provider: str = "auto",
        embedding_model: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_dir: str | None = None
    ):
        """Initialize RAG Manager.

        Args:
            embedding_provider: "openai", "huggingface", or "auto"
            embedding_model: Specific model name
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            persist_dir: Directory for persistence
        """
        if persist_dir is None:
            persist_dir = str(Path.home() / ".sepilot" / "rag")

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Document metadata storage
        self.docs_file = self.persist_dir / "documents.json"
        self.documents: dict[str, DocumentInfo] = self._load_documents()

        # Thread safety lock for document operations
        self._lock = threading.RLock()

        # Initialize components lazily
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._embedding_fn: EmbeddingFunction | None = None
        self._vectorstore: VectorStore | None = None
        self._chunker: DocumentChunker | None = None

        logger.info(f"RAGManager initialized: {persist_dir}")

    @property
    def embedding_fn(self) -> EmbeddingFunction:
        """Lazy initialization of embedding function."""
        if self._embedding_fn is None:
            self._embedding_fn = get_embedding_function(
                provider=self._embedding_provider,
                model=self._embedding_model
            )
        return self._embedding_fn

    @property
    def vectorstore(self) -> VectorStore:
        """Lazy initialization of vector store."""
        if self._vectorstore is None:
            self._vectorstore = VectorStore(
                embedding_fn=self.embedding_fn,
                persist_dir=str(self.persist_dir / "vectordb")
            )
        return self._vectorstore

    @property
    def chunker(self) -> DocumentChunker:
        """Lazy initialization of chunker."""
        if self._chunker is None:
            self._chunker = DocumentChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        return self._chunker

    def _load_documents(self) -> dict[str, DocumentInfo]:
        """Load document metadata from disk."""
        if not self.docs_file.exists():
            return {}

        try:
            with open(self.docs_file, encoding="utf-8") as f:
                data = json.load(f)
                return {
                    url: DocumentInfo(**info)
                    for url, info in data.items()
                }
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return {}

    def _save_documents(self):
        """Save document metadata to disk atomically (thread-safe)."""
        with self._lock:
            try:
                data = {url: asdict(info) for url, info in self.documents.items()}
                # Atomic write: write to temp file, then rename
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(self.persist_dir),
                    suffix=".tmp",
                    prefix=".documents_",
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, self.docs_file)
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            except Exception as e:
                logger.error(f"Failed to save documents: {e}")

    async def add_url(
        self,
        url: str,
        title: str = "",
        description: str = "",
        force_refresh: bool = False
    ) -> DocumentInfo:
        """Add a URL to the RAG index.

        Args:
            url: URL to index
            title: Document title
            description: Document description
            force_refresh: Re-index even if already indexed

        Returns:
            DocumentInfo with indexing status
        """
        # Check if already indexed
        if url in self.documents and not force_refresh:
            doc = self.documents[url]
            if doc.status == "indexed":
                logger.info(f"URL already indexed: {url}")
                return doc

        # Create pending document info
        now = datetime.now().isoformat()
        doc_info = DocumentInfo(
            url=url,
            title=title or url,
            description=description,
            chunk_count=0,
            indexed_at=now,
            last_updated=now,
            status="pending"
        )
        self.documents[url] = doc_info
        self._save_documents()

        try:
            # Fetch content
            content = await self._fetch_url(url)
            if not content:
                raise ValueError("Failed to fetch content")

            # Chunk content
            chunks = self.chunker.chunk_text(content, source=url)
            if not chunks:
                raise ValueError("No chunks generated")

            # Add to vector store
            chunk_count = self.vectorstore.add_chunks(chunks)

            # Update document info
            doc_info.chunk_count = chunk_count
            doc_info.status = "indexed"
            doc_info.last_updated = datetime.now().isoformat()
            self._save_documents()

            logger.info(f"Indexed URL: {url} ({chunk_count} chunks)")
            return doc_info

        except Exception as e:
            doc_info.status = "error"
            doc_info.error_message = str(e)
            doc_info.last_updated = datetime.now().isoformat()
            self._save_documents()
            logger.error(f"Failed to index URL {url}: {e}")
            return doc_info

    async def _fetch_url(self, url: str) -> str | None:
        """Fetch content from URL."""
        try:
            # Use the existing rag_fetcher
            from sepilot.utils.rag_fetcher import RAGContentFetcher
            fetcher = RAGContentFetcher(cache_ttl=86400)  # 24h cache
            content = await fetcher.fetch_url(url, use_cache=True)
            return content
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def add_url_sync(
        self,
        url: str,
        title: str = "",
        description: str = "",
        force_refresh: bool = False
    ) -> DocumentInfo:
        """Synchronous wrapper for add_url."""
        return asyncio.run(self.add_url(url, title, description, force_refresh))

    async def add_urls(
        self,
        urls: list[str],
        force_refresh: bool = False,
        max_concurrent: int = 3,
        progress_callback: ProgressCallback | None = None
    ) -> list[DocumentInfo]:
        """Add multiple URLs to the RAG index (batch indexing).

        Args:
            urls: List of URLs to index
            force_refresh: Re-index even if already indexed
            max_concurrent: Maximum concurrent indexing tasks
            progress_callback: Optional callback for progress updates
                              Called with (current_index, total, status_message)

        Returns:
            List of DocumentInfo for each URL
        """
        if not urls:
            return []

        results: list[DocumentInfo] = []
        total = len(urls)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def index_with_semaphore(url: str, index: int) -> DocumentInfo:
            async with semaphore:
                if progress_callback:
                    progress_callback(index + 1, total, f"Indexing: {url[:50]}...")

                result = await self.add_url(url, force_refresh=force_refresh)

                if progress_callback:
                    status = "✓" if result.status == "indexed" else "✗"
                    progress_callback(index + 1, total, f"{status} {url[:50]}")

                return result

        # Create tasks for all URLs
        tasks = [
            index_with_semaphore(url, i)
            for i, url in enumerate(urls)
        ]

        # Execute all tasks concurrently (with semaphore limiting)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to index {urls[i]}: {result}")
                # Create error DocumentInfo
                error_info = DocumentInfo(
                    url=urls[i],
                    title="",
                    description="",
                    chunk_count=0,
                    indexed_at=datetime.now().isoformat(),
                    last_updated=datetime.now().isoformat(),
                    status="error",
                    error_message=str(result)
                )
                final_results.append(error_info)
            else:
                final_results.append(result)

        if progress_callback:
            success_count = sum(1 for r in final_results if r.status == "indexed")
            progress_callback(total, total, f"완료: {success_count}/{total} 성공")

        return final_results

    def add_urls_sync(
        self,
        urls: list[str],
        force_refresh: bool = False,
        max_concurrent: int = 3,
        progress_callback: ProgressCallback | None = None
    ) -> list[DocumentInfo]:
        """Synchronous wrapper for add_urls (batch indexing)."""
        return asyncio.run(self.add_urls(urls, force_refresh, max_concurrent, progress_callback))

    def remove_url(self, url: str) -> bool:
        """Remove a URL from the RAG index.

        Args:
            url: URL to remove

        Returns:
            True if removed, False if not found
        """
        if url not in self.documents:
            return False

        # Remove from vector store
        self.vectorstore.delete_by_source(url)

        # Remove from documents
        del self.documents[url]
        self._save_documents()

        logger.info(f"Removed URL: {url}")
        return True

    def search(
        self,
        query: str,
        n_results: int = 5,
        source_filter: str | None = None,
        score_threshold: float = 0.3
    ) -> list[SearchResult]:
        """Search for relevant content.

        Args:
            query: Search query
            n_results: Maximum results to return
            source_filter: Filter by source URL
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of SearchResult objects
        """
        return self.vectorstore.search(
            query=query,
            n_results=n_results,
            source_filter=source_filter,
            score_threshold=score_threshold
        )

    def get_context(
        self,
        query: str,
        n_results: int = 5,
        max_chars: int = 10000
    ) -> str:
        """Get RAG context for a query.

        Args:
            query: User query
            n_results: Maximum chunks to include
            max_chars: Maximum total characters

        Returns:
            Formatted context string
        """
        results = self.search(query, n_results=n_results)

        if not results:
            return ""

        # Build context
        context_parts = []
        total_chars = 0

        for result in results:
            if total_chars + len(result.content) > max_chars:
                break

            context_parts.append(
                f"[Source: {result.source}]\n{result.content}"
            )
            total_chars += len(result.content)

        if not context_parts:
            return ""

        context = "\n\n---\n\n".join(context_parts)
        return f"\n# RAG Context (Semantic Search Results)\n\n{context}\n\n---\n"

    def list_documents(self) -> list[DocumentInfo]:
        """List all indexed documents.

        Returns:
            List of DocumentInfo objects
        """
        return list(self.documents.values())

    def get_stats(self) -> dict[str, Any]:
        """Get RAG statistics.

        Returns:
            Statistics dictionary
        """
        vs_stats = self.vectorstore.get_stats()

        indexed_count = sum(
            1 for doc in self.documents.values()
            if doc.status == "indexed"
        )
        error_count = sum(
            1 for doc in self.documents.values()
            if doc.status == "error"
        )

        return {
            "total_documents": len(self.documents),
            "indexed_documents": indexed_count,
            "error_documents": error_count,
            "total_chunks": vs_stats["total_chunks"],
            "embedding_provider": self._embedding_provider,
            "embedding_dimension": vs_stats["embedding_dimension"],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "persist_dir": str(self.persist_dir)
        }

    def clear(self) -> dict[str, int]:
        """Clear all RAG data.

        Returns:
            Counts of deleted items
        """
        doc_count = len(self.documents)
        chunk_count = self.vectorstore.clear()

        self.documents = {}
        self._save_documents()

        logger.info(f"Cleared RAG: {doc_count} documents, {chunk_count} chunks")
        return {
            "documents": doc_count,
            "chunks": chunk_count
        }

    def add_file(
        self,
        file_path: str | Path,
        title: str = "",
        description: str = "",
        code_aware: bool = True,
        force_refresh: bool = False,
    ) -> DocumentInfo:
        """Add a local file to the RAG index.

        Args:
            file_path: Path to the file to index
            title: Document title
            description: Document description
            code_aware: Use code-aware chunking for source files
            force_refresh: Re-index even if already indexed

        Returns:
            DocumentInfo with indexing status
        """
        file_path = Path(file_path).resolve()
        source_key = f"file://{file_path}"

        if not file_path.is_file():
            return DocumentInfo(
                url=source_key,
                title=title or file_path.name,
                description=description,
                chunk_count=0,
                indexed_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                status="error",
                error_message=f"File not found: {file_path}",
            )

        # Check if already indexed
        if source_key in self.documents and not force_refresh:
            doc = self.documents[source_key]
            if doc.status == "indexed":
                logger.info(f"File already indexed: {file_path}")
                return doc

        now = datetime.now().isoformat()
        doc_info = DocumentInfo(
            url=source_key,
            title=title or file_path.name,
            description=description or f"Local file: {file_path}",
            chunk_count=0,
            indexed_at=now,
            last_updated=now,
            status="pending",
        )
        self.documents[source_key] = doc_info
        self._save_documents()

        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                raise ValueError("File is empty")

            # Choose chunker
            if code_aware:
                chunker = CodeAwareChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            else:
                chunker = self.chunker

            chunks = chunker.chunk_text(content, source=source_key)
            if not chunks:
                raise ValueError("No chunks generated")

            chunk_count = self.vectorstore.add_chunks(chunks)

            doc_info.chunk_count = chunk_count
            doc_info.status = "indexed"
            doc_info.last_updated = datetime.now().isoformat()
            self._save_documents()

            logger.info(f"Indexed file: {file_path} ({chunk_count} chunks)")
            return doc_info

        except UnicodeDecodeError:
            doc_info.status = "error"
            doc_info.error_message = "File is not valid UTF-8 text"
            doc_info.last_updated = datetime.now().isoformat()
            self._save_documents()
            return doc_info
        except Exception as e:
            doc_info.status = "error"
            doc_info.error_message = str(e)
            doc_info.last_updated = datetime.now().isoformat()
            self._save_documents()
            logger.error(f"Failed to index file {file_path}: {e}")
            return doc_info

    def add_directory(
        self,
        directory: str | Path,
        patterns: list[str] | None = None,
        code_aware: bool = True,
        force_refresh: bool = False,
        max_files: int = 500,
    ) -> list[DocumentInfo]:
        """Add all matching files from a directory to the RAG index.

        Args:
            directory: Directory path
            patterns: Glob patterns to match (defaults to common code/text files)
            code_aware: Use code-aware chunking
            force_refresh: Re-index even if already indexed
            max_files: Maximum number of files to index

        Returns:
            List of DocumentInfo for each file
        """
        directory = Path(directory).resolve()
        if not directory.is_dir():
            logger.error(f"Directory not found: {directory}")
            return []

        if patterns is None:
            patterns = [
                "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
                "*.go", "*.rs", "*.java", "*.c", "*.cpp", "*.h",
                "*.md", "*.txt", "*.rst", "*.yaml", "*.yml",
                "*.json", "*.toml", "*.cfg", "*.ini",
            ]

        # Collect files
        skip_dirs = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "dist", "build", ".mypy_cache", ".pytest_cache",
        }

        files = []
        for pattern in patterns:
            for fp in directory.rglob(pattern):
                if fp.is_file() and not any(d in fp.parts for d in skip_dirs):
                    files.append(fp)
                    if len(files) >= max_files:
                        break
            if len(files) >= max_files:
                break

        results = []
        for fp in files:
            result = self.add_file(
                fp,
                code_aware=code_aware,
                force_refresh=force_refresh,
            )
            results.append(result)

        indexed = sum(1 for r in results if r.status == "indexed")
        logger.info(f"Indexed {indexed}/{len(files)} files from {directory}")
        return results

    def refresh_url(self, url: str) -> DocumentInfo | None:
        """Re-index a URL.

        Args:
            url: URL to refresh

        Returns:
            Updated DocumentInfo or None if not found
        """
        if url not in self.documents:
            return None

        # Get existing info
        old_info = self.documents[url]

        # Remove old chunks
        self.vectorstore.delete_by_source(url)

        # Re-index
        return self.add_url_sync(
            url=url,
            title=old_info.title,
            description=old_info.description,
            force_refresh=True
        )


# Global singleton instance
_rag_manager: RAGManager | None = None
_rag_init_failed: bool = False


def get_rag_manager() -> RAGManager:
    """Get or create the global RAG manager instance.

    초기화 실패 시 결과를 캐싱하여 매번 재시도하지 않음.
    (chromadb/embedding 초기화 에러가 반복되면 매 입력마다 수 초 지연 발생)
    """
    global _rag_manager, _rag_init_failed
    if _rag_init_failed:
        raise ImportError("RAG initialization previously failed; skipping retry")
    if _rag_manager is None:
        try:
            _rag_manager = RAGManager()
        except Exception:
            _rag_init_failed = True
            raise
    return _rag_manager
