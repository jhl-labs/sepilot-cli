"""SE Pilot - Intelligent CLI Agent for Software Engineering"""

__version__ = "0.9.0"

# Configure bundled tiktoken data for offline use.
# tiktoken downloads encoding files on first use; by pointing
# TIKTOKEN_CACHE_DIR at our bundled copy the download is skipped entirely.
import os as _os

# PyInstaller bundles data files under sys._MEIPASS, so we check both paths.
import sys as _sys

_base_dir = getattr(_sys, '_MEIPASS', _os.path.dirname(__file__))
_tiktoken_data = _os.path.join(_base_dir, 'sepilot', 'data', 'tiktoken')
if not _os.path.isdir(_tiktoken_data):
    # Fallback for non-PyInstaller (normal Python package) layout
    _tiktoken_data = _os.path.join(_os.path.dirname(__file__), 'data', 'tiktoken')
if _os.path.isdir(_tiktoken_data) and 'TIKTOKEN_CACHE_DIR' not in _os.environ:
    _os.environ['TIKTOKEN_CACHE_DIR'] = _os.path.abspath(_tiktoken_data)
del _os, _sys, _base_dir, _tiktoken_data
