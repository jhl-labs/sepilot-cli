"""Multimedia file reading tool - Image and PDF support.

Provides functionality to read and extract content from:
- Image files (PNG, JPG, GIF, etc.)
- PDF files
"""

import base64
import logging
from pathlib import Path
from typing import Any

from sepilot.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class MultimediaTool(BaseTool):
    """Tool for reading multimedia files (images, PDFs)."""

    name = "multimedia_read"
    description = "Read and extract content from images and PDF files"
    parameters = {
        "file_path": "Path to the file to read (required)",
        "extract_text": "Whether to extract text from PDF (default: True)",
        "page_limit": "Maximum pages to process for PDF (default: 50)"
    }

    # Supported file extensions
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.ico'}
    PDF_EXTENSION = '.pdf'

    def execute(
        self,
        file_path: str,
        extract_text: bool = True,
        page_limit: int = 50
    ) -> str:
        """Read a multimedia file and return its content.

        Args:
            file_path: Path to the file
            extract_text: Whether to extract text from PDF
            page_limit: Maximum pages to process

        Returns:
            File content or error message
        """
        self.validate_params(file_path=file_path)

        try:
            path = Path(file_path).resolve()

            # Security check
            project_root = Path.cwd().resolve()
            home_dir = Path.home().resolve()

            # Allow paths within project or home directory
            try:
                path.relative_to(project_root)
            except ValueError:
                try:
                    path.relative_to(home_dir)
                except ValueError:
                    return f"Error: Path must be within project or home directory: {file_path}"

            if not path.exists():
                return f"Error: File not found: {file_path}"

            extension = path.suffix.lower()

            if extension in self.IMAGE_EXTENSIONS:
                return self._read_image(path)
            elif extension == self.PDF_EXTENSION:
                return self._read_pdf(path, extract_text, page_limit)
            else:
                return f"Error: Unsupported file type: {extension}"

        except Exception as e:
            return f"Error reading multimedia file: {str(e)}"

    def _read_image(self, path: Path) -> str:
        """Read an image file and return its information.

        Args:
            path: Path to the image file

        Returns:
            Image information and base64 encoded content
        """
        try:
            # Get file info
            file_size = path.stat().st_size
            extension = path.suffix.lower()

            result = [
                f"Image File: {path.name}",
                f"Type: {extension}",
                f"Size: {self._format_size(file_size)}",
                ""
            ]

            # Try to get image dimensions using PIL
            try:
                from PIL import Image

                with Image.open(path) as img:
                    width, height = img.size
                    mode = img.mode
                    result.append(f"Dimensions: {width}x{height} pixels")
                    result.append(f"Color Mode: {mode}")

                    # Check for EXIF data
                    if hasattr(img, '_getexif') and img._getexif():
                        result.append("EXIF data: Present")

            except ImportError:
                result.append("(PIL not installed - install with: pip install Pillow)")
            except Exception as e:
                result.append(f"Could not read image metadata: {e}")

            result.append("")

            # Encode to base64 for potential LLM vision use
            with open(path, 'rb') as f:
                image_data = f.read()

            # Only include base64 for smaller images (< 5MB)
            if file_size < 5 * 1024 * 1024:
                b64_data = base64.b64encode(image_data).decode('utf-8')
                mime_type = self._get_mime_type(extension)

                result.append("Base64 encoded content (for vision analysis):")
                result.append(f"data:{mime_type};base64,{b64_data[:100]}...")
                result.append("")
                result.append(
                    "Note: Full base64 data available. "
                    "Use vision capabilities to analyze the image content."
                )
            else:
                result.append("Image too large for base64 encoding (>5MB)")

            return "\n".join(result)

        except Exception as e:
            return f"Error reading image: {str(e)}"

    def _read_pdf(self, path: Path, extract_text: bool, page_limit: int) -> str:
        """Read a PDF file and extract its content.

        Args:
            path: Path to the PDF file
            extract_text: Whether to extract text
            page_limit: Maximum pages to process

        Returns:
            PDF content
        """
        try:
            file_size = path.stat().st_size

            result = [
                f"PDF File: {path.name}",
                f"Size: {self._format_size(file_size)}",
                ""
            ]

            # Try PyPDF2 first
            try:
                import PyPDF2

                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    num_pages = len(reader.pages)

                    result.append(f"Total Pages: {num_pages}")

                    # Extract metadata
                    metadata = reader.metadata
                    if metadata:
                        if metadata.title:
                            result.append(f"Title: {metadata.title}")
                        if metadata.author:
                            result.append(f"Author: {metadata.author}")
                        if metadata.subject:
                            result.append(f"Subject: {metadata.subject}")

                    result.append("")

                    if extract_text:
                        result.append("=" * 50)
                        result.append("EXTRACTED TEXT")
                        result.append("=" * 50)

                        pages_to_process = min(num_pages, page_limit)
                        for i in range(pages_to_process):
                            page = reader.pages[i]
                            text = page.extract_text()

                            if text:
                                result.append(f"\n--- Page {i + 1} ---")
                                result.append(text.strip())

                        if num_pages > page_limit:
                            result.append(
                                f"\n... ({num_pages - page_limit} more pages not shown)"
                            )

                return "\n".join(result)

            except ImportError:
                result.append("PyPDF2 not installed. Install with: pip install PyPDF2")
                result.append("")

                # Try pdfplumber as fallback
                try:
                    import pdfplumber

                    with pdfplumber.open(path) as pdf:
                        num_pages = len(pdf.pages)
                        result.append(f"Total Pages: {num_pages}")
                        result.append("")

                        if extract_text:
                            result.append("=" * 50)
                            result.append("EXTRACTED TEXT")
                            result.append("=" * 50)

                            pages_to_process = min(num_pages, page_limit)
                            for i in range(pages_to_process):
                                page = pdf.pages[i]
                                text = page.extract_text()

                                if text:
                                    result.append(f"\n--- Page {i + 1} ---")
                                    result.append(text.strip())

                    return "\n".join(result)

                except ImportError:
                    result.append(
                        "pdfplumber also not installed. "
                        "Install with: pip install pdfplumber"
                    )
                    return "\n".join(result)

        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def _format_size(self, size: int) -> str:
        """Format file size for display."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _get_mime_type(self, extension: str) -> str:
        """Get MIME type for image extension."""
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon'
        }
        return mime_types.get(extension, 'application/octet-stream')


def get_image_info(file_path: str) -> dict[str, Any]:
    """Get image file information.

    Args:
        file_path: Path to the image

    Returns:
        Dictionary with image info
    """
    try:
        path = Path(file_path)
        info = {
            "path": str(path),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size": path.stat().st_size
        }

        try:
            from PIL import Image
            with Image.open(path) as img:
                info["width"] = img.size[0]
                info["height"] = img.size[1]
                info["mode"] = img.mode
                info["format"] = img.format
        except ImportError:
            pass

        return info

    except Exception as e:
        return {"error": str(e)}


def get_pdf_info(file_path: str) -> dict[str, Any]:
    """Get PDF file information.

    Args:
        file_path: Path to the PDF

    Returns:
        Dictionary with PDF info
    """
    try:
        path = Path(file_path)
        info = {
            "path": str(path),
            "name": path.name,
            "size": path.stat().st_size
        }

        try:
            import PyPDF2
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                info["pages"] = len(reader.pages)
                metadata = reader.metadata
                if metadata:
                    info["title"] = metadata.title
                    info["author"] = metadata.author
        except ImportError:
            pass

        return info

    except Exception as e:
        return {"error": str(e)}
