"""PyInstaller runtime hook: set TIKTOKEN_CACHE_DIR before any imports.

This hook runs before the application entry point, ensuring tiktoken
finds the bundled encoding data and never attempts network downloads.
"""

import os
import sys

# Known tiktoken cache file (cl100k_base) used to locate the data directory
_KNOWN_CACHE_FILE = '9b5ad71b2ce5302211f9c61530b329a4922fc6a4'

def _find_tiktoken_data():
    """Search for bundled tiktoken data directory."""
    _meipass = getattr(sys, '_MEIPASS', None)
    if not _meipass:
        return None

    # Try known candidate paths
    candidates = [
        os.path.join(_meipass, 'sepilot', 'data', 'tiktoken'),
        os.path.join(_meipass, 'data', 'tiktoken'),
        os.path.join(_meipass, 'tiktoken'),
    ]

    for path in candidates:
        if os.path.isfile(os.path.join(path, _KNOWN_CACHE_FILE)):
            return path

    # Walk the bundle to find it
    for root, dirs, files in os.walk(_meipass):
        if _KNOWN_CACHE_FILE in files:
            return root

    return None


# 1) Find and set TIKTOKEN_CACHE_DIR
_tiktoken_dir = _find_tiktoken_data()
if _tiktoken_dir:
    os.environ['TIKTOKEN_CACHE_DIR'] = _tiktoken_dir
    os.environ['DATA_GYM_CACHE_DIR'] = _tiktoken_dir


# 2) Monkey-patch tiktoken.load.read_file to block network downloads
def _patch_tiktoken_read_file():
    try:
        import tiktoken.load as _tl
        _original_read_file = _tl.read_file

        def _read_file_no_network(blobpath: str, *args, **kwargs):
            """Block network downloads - only allow local/cached files."""
            if blobpath.startswith(("http://", "https://")):
                _cache_dir = os.environ.get('TIKTOKEN_CACHE_DIR', 'NOT SET')
                _dir_exists = os.path.isdir(_cache_dir) if _cache_dir != 'NOT SET' else False
                _dir_files = os.listdir(_cache_dir) if _dir_exists else []
                raise FileNotFoundError(
                    f"tiktoken: network download blocked (offline mode). "
                    f"URL: {blobpath} | "
                    f"TIKTOKEN_CACHE_DIR={_cache_dir} | "
                    f"dir_exists={_dir_exists} | "
                    f"files={_dir_files}"
                )
            return _original_read_file(blobpath, *args, **kwargs)

        _tl.read_file = _read_file_no_network
    except (ImportError, AttributeError):
        pass

_patch_tiktoken_read_file()
