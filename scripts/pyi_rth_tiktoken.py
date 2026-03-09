"""PyInstaller runtime hook: set TIKTOKEN_CACHE_DIR before any imports.

This hook runs before the application entry point, ensuring tiktoken
finds the bundled encoding data and never attempts network downloads.
"""

import os
import sys

_meipass = getattr(sys, '_MEIPASS', None)
if _meipass:
    _tiktoken_dir = os.path.join(_meipass, 'sepilot', 'data', 'tiktoken')
    if os.path.isdir(_tiktoken_dir) and 'TIKTOKEN_CACHE_DIR' not in os.environ:
        os.environ['TIKTOKEN_CACHE_DIR'] = _tiktoken_dir
