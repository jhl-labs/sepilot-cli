"""Document Chunking for RAG

Splits documents into smaller chunks for embedding and retrieval.
Supports both plain text and code-aware chunking.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    content: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Normalize optional metadata to a dictionary."""
        if self.metadata is None:
            self.metadata = {}


class DocumentChunker:
    """Split documents into chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators to split on (in order of priority)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",
            "? ",
            "; ",
            ", ",
            " ",     # Words
            ""       # Characters (last resort)
        ]

    def chunk_text(self, text: str, source: str = "unknown") -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to split
            source: Source identifier (URL, filename, etc.)

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        # Clean text
        text = self._clean_text(text)

        # Split recursively
        chunks = self._recursive_split(text, self.separators)

        # Create Chunk objects with metadata
        result = []
        current_pos = 0

        for i, chunk_text in enumerate(chunks):
            # Find position in original text
            start = text.find(chunk_text, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(chunk_text)

            chunk = Chunk(
                content=chunk_text,
                source=source,
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata={
                    "chunk_size": len(chunk_text),
                    "total_chunks": len(chunks)
                }
            )
            result.append(chunk)
            current_pos = start + 1

        logger.info(f"Split '{source}' into {len(result)} chunks")
        return result

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _recursive_split(
        self,
        text: str,
        separators: list[str]
    ) -> list[str]:
        """Recursively split text using separators."""
        if not text:
            return []

        # If text is small enough, return as-is
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try each separator
        for sep in separators:
            if sep == "":
                # Last resort: split by characters
                return self._split_by_size(text)

            if sep in text:
                splits = text.split(sep)
                # Merge small splits
                chunks = self._merge_splits(splits, sep)
                return chunks

        # No separator found, split by size
        return self._split_by_size(text)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits into larger chunks."""
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split = split.strip()
            if not split:
                continue

            split_length = len(split) + len(separator)

            # If adding this split would exceed chunk_size
            if current_length + split_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_splits = self._get_overlap_splits(
                    current_chunk, separator
                )
                current_chunk = overlap_splits + [split]
                current_length = sum(len(s) + len(separator) for s in current_chunk)
            else:
                current_chunk.append(split)
                current_length += split_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _get_overlap_splits(
        self,
        splits: list[str],
        separator: str
    ) -> list[str]:
        """Get splits for overlap from end of previous chunk."""
        if not splits or self.chunk_overlap <= 0:
            return []

        overlap_splits = []
        overlap_length = 0

        for split in reversed(splits):
            if overlap_length + len(split) + len(separator) > self.chunk_overlap:
                break
            overlap_splits.insert(0, split)
            overlap_length += len(split) + len(separator)

        return overlap_splits

    def _split_by_size(self, text: str) -> list[str]:
        """Split text by character count."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at word boundary
            if end < len(text):
                # Look for last space before end
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap, ensure forward progress
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks


class CodeAwareChunker(DocumentChunker):
    """Chunker that respects code structure boundaries.

    Splits code by functions/classes first, then falls back to
    line-based splitting. This preserves semantic units.
    """

    # Patterns that indicate code block boundaries
    _CODE_BOUNDARY_PATTERNS = [
        # Python class/function
        re.compile(r'^(?:class |def |async def )', re.MULTILINE),
        # JS/TS function/class
        re.compile(r'^(?:export |)(function |class |const \w+ = )', re.MULTILINE),
        # Go function
        re.compile(r'^func ', re.MULTILINE),
        # Rust function/impl
        re.compile(r'^(?:pub |)(fn |impl |struct |enum |trait )', re.MULTILINE),
    ]

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
    ):
        # Code chunks can be larger; overlap is smaller since we split on boundaries
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\nclass ",
                "\ndef ",
                "\nasync def ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        )

    def chunk_text(self, text: str, source: str = "unknown") -> list[Chunk]:
        """Split code into chunks respecting function/class boundaries."""
        if not text or not text.strip():
            return []

        # Detect if this is code
        ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
        code_extensions = {
            "py", "js", "ts", "jsx", "tsx", "go", "rs", "java",
            "c", "cpp", "h", "hpp", "cs", "rb", "php", "swift", "kt",
        }

        if ext in code_extensions:
            return self._chunk_code(text, source)

        # Fall back to regular chunking for non-code
        return super().chunk_text(text, source)

    def _chunk_code(self, text: str, source: str) -> list[Chunk]:
        """Split code by top-level definitions."""
        # Find all top-level definition boundaries
        boundary_positions = []
        for pattern in self._CODE_BOUNDARY_PATTERNS:
            for match in pattern.finditer(text):
                boundary_positions.append(match.start())

        if not boundary_positions:
            # No boundaries found, fall back to regular chunking
            return super().chunk_text(text, source)

        boundary_positions = sorted(set(boundary_positions))

        # Split at boundaries
        segments = []
        for i, pos in enumerate(boundary_positions):
            end = boundary_positions[i + 1] if i + 1 < len(boundary_positions) else len(text)
            segment = text[pos:end].rstrip()
            if segment:
                segments.append((pos, segment))

        # Include any preamble (imports, etc.) before first boundary
        if boundary_positions[0] > 0:
            preamble = text[:boundary_positions[0]].rstrip()
            if preamble:
                segments.insert(0, (0, preamble))

        # Merge small segments, split large ones
        chunks = []
        current_text = ""
        current_start = 0

        for pos, segment in segments:
            if not current_text:
                current_text = segment
                current_start = pos
            elif len(current_text) + len(segment) + 1 <= self.chunk_size:
                current_text += "\n" + segment
            else:
                # Save current chunk
                chunks.append(Chunk(
                    content=current_text,
                    source=source,
                    chunk_index=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(current_text),
                    metadata={"chunk_type": "code"},
                ))
                current_text = segment
                current_start = pos

        # Don't forget the last chunk
        if current_text:
            chunks.append(Chunk(
                content=current_text,
                source=source,
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=current_start + len(current_text),
                metadata={"chunk_type": "code"},
            ))

        # Handle oversized chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > self.chunk_size * 2:
                # Split oversized chunks using parent logic
                sub_chunks = super().chunk_text(chunk.content, source)
                for sc in sub_chunks:
                    sc.start_char += chunk.start_char
                    sc.end_char += chunk.start_char
                    sc.chunk_index = len(final_chunks)
                    sc.metadata["chunk_type"] = "code"
                    final_chunks.append(sc)
            else:
                chunk.chunk_index = len(final_chunks)
                final_chunks.append(chunk)

        # Update total_chunks metadata
        for c in final_chunks:
            c.metadata["total_chunks"] = len(final_chunks)

        logger.info(f"Split code '{source}' into {len(final_chunks)} chunks")
        return final_chunks


def chunk_document(
    text: str,
    source: str = "unknown",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    code_aware: bool = False,
) -> list[Chunk]:
    """Convenience function to chunk a document.

    Args:
        text: Document text
        source: Source identifier
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        code_aware: Use code-aware chunking for source files

    Returns:
        List of Chunk objects
    """
    if code_aware:
        chunker = CodeAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return chunker.chunk_text(text, source)
