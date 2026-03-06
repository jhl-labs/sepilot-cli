"""Multimedia tools for LangChain agent.

Provides image_read, pdf_read tools for processing multimedia files.
"""

from langchain_core.tools import tool


@tool
def image_read(file_path: str) -> str:
    """Read and analyze an image file.

    Supports common image formats: PNG, JPG/JPEG, GIF, BMP, WebP, SVG, ICO.

    This tool extracts image metadata (dimensions, color mode, EXIF data)
    and can provide base64 encoded content for vision analysis.

    Args:
        file_path: Absolute path to the image file (required)

    Returns:
        Image information and content

    Examples:
        # Read a screenshot
        image_read(file_path="/tmp/screenshot.png")

        # Read a diagram
        image_read(file_path="/home/user/docs/architecture.jpg")
    """
    from sepilot.tools.file_tools.multimedia_tool import MultimediaTool

    tool_instance = MultimediaTool()
    return tool_instance.execute(file_path, extract_text=False)


@tool
def pdf_read(
    file_path: str,
    extract_text: bool = True,
    page_limit: int = 50
) -> str:
    """Read and extract content from a PDF file.

    Extracts text content and metadata from PDF documents.
    Supports multi-page PDFs with configurable page limits.

    Args:
        file_path: Absolute path to the PDF file (required)
        extract_text: Whether to extract text content (default: True)
        page_limit: Maximum pages to process (default: 50)

    Returns:
        PDF metadata and extracted text content

    Examples:
        # Read a PDF document
        pdf_read(file_path="/home/user/docs/manual.pdf")

        # Read only first 10 pages
        pdf_read(file_path="/home/user/docs/large_report.pdf", page_limit=10)

        # Get metadata only without text extraction
        pdf_read(file_path="/home/user/docs/book.pdf", extract_text=False)
    """
    from sepilot.tools.file_tools.multimedia_tool import MultimediaTool

    tool_instance = MultimediaTool()
    return tool_instance.execute(file_path, extract_text=extract_text, page_limit=page_limit)


@tool
def multimedia_info(file_path: str) -> str:
    """Get information about an image or PDF file.

    Quick way to get metadata without full content extraction.
    Useful for checking file properties before processing.

    Args:
        file_path: Absolute path to the file (required)

    Returns:
        File information (type, size, dimensions/pages, etc.)

    Examples:
        # Check image info
        multimedia_info(file_path="/tmp/image.png")

        # Check PDF info
        multimedia_info(file_path="/home/user/report.pdf")
    """
    from pathlib import Path

    from sepilot.tools.file_tools.multimedia_tool import get_image_info, get_pdf_info

    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"

    extension = path.suffix.lower()

    if extension in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.ico'}:
        info = get_image_info(file_path)
    elif extension == '.pdf':
        info = get_pdf_info(file_path)
    else:
        return f"Error: Unsupported file type: {extension}"

    if "error" in info:
        return f"Error: {info['error']}"

    # Format output
    result = [f"File: {info.get('name', 'Unknown')}"]
    result.append(f"Type: {info.get('extension', 'Unknown')}")

    size = info.get('size', 0)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            result.append(f"Size: {size:.1f} {unit}")
            break
        size /= 1024

    if 'width' in info and 'height' in info:
        result.append(f"Dimensions: {info['width']}x{info['height']} pixels")
    if 'mode' in info:
        result.append(f"Color Mode: {info['mode']}")
    if 'pages' in info:
        result.append(f"Pages: {info['pages']}")
    if 'title' in info and info['title']:
        result.append(f"Title: {info['title']}")
    if 'author' in info and info['author']:
        result.append(f"Author: {info['author']}")

    return "\n".join(result)


__all__ = ['image_read', 'pdf_read', 'multimedia_info']
