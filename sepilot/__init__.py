"""SE Pilot - Intelligent CLI Agent for Software Engineering"""

__version__ = "0.6.1"

# Configure bundled tiktoken data for offline use.
# tiktoken downloads encoding files on first use; by pointing
# TIKTOKEN_CACHE_DIR at our bundled copy the download is skipped entirely.
import os as _os
_tiktoken_data = _os.path.join(_os.path.dirname(__file__), 'data', 'tiktoken')
if _os.path.isdir(_tiktoken_data) and 'TIKTOKEN_CACHE_DIR' not in _os.environ:
    _os.environ['TIKTOKEN_CACHE_DIR'] = _os.path.abspath(_tiktoken_data)
del _os, _tiktoken_data
