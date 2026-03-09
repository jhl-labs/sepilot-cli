"""PyInstaller runtime hook: set TIKTOKEN_CACHE_DIR before any imports.

This hook runs before the application entry point, ensuring tiktoken
finds the bundled encoding data and never attempts network downloads.
"""

import os
import sys

# 1) Set TIKTOKEN_CACHE_DIR to bundled data (unconditionally override)
_meipass = getattr(sys, '_MEIPASS', None)
if _meipass:
    _tiktoken_dir = os.path.join(_meipass, 'sepilot', 'data', 'tiktoken')
    os.environ['TIKTOKEN_CACHE_DIR'] = _tiktoken_dir
    # Also set the legacy env var used by older tiktoken versions
    os.environ['DATA_GYM_CACHE_DIR'] = _tiktoken_dir

# 2) Monkey-patch tiktoken.load.read_file to block network downloads
#    If the cache file is missing, raise an error instead of downloading.
def _patch_tiktoken_read_file():
    try:
        import tiktoken.load as _tl
        _original_read_file = _tl.read_file

        def _read_file_no_network(blobpath: str, *args, **kwargs):
            """Block network downloads - only allow local/cached files."""
            if blobpath.startswith(("http://", "https://")):
                raise FileNotFoundError(
                    f"tiktoken: network download blocked (offline mode). "
                    f"Missing cache for: {blobpath}"
                )
            return _original_read_file(blobpath, *args, **kwargs)

        _tl.read_file = _read_file_no_network
    except (ImportError, AttributeError):
        pass

_patch_tiktoken_read_file()
