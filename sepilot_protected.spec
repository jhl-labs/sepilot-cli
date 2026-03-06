# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for sepilot

Features:
- UPX compression
- Debug symbols stripped
- Python optimization level 1 (remove asserts, keep docstrings)
- Unnecessary modules excluded

Build command:
    ./build_protected.sh
    # or: pyinstaller sepilot_protected.spec --clean

Output:
    dist/sepilot (single executable)
"""

import os
import sys
from pathlib import Path

# Project root
ROOT = Path(SPECPATH)
SOURCE_DIR = Path(os.environ.get("SEPILOT_SOURCE_DIR", str(ROOT / "sepilot")))
ENTRYPOINT = SOURCE_DIR / "cli" / "main.py"

# Collect data files
datas = [
    # Prompt templates
    (str(SOURCE_DIR / 'prompts' / 'templates'), 'sepilot/prompts/templates'),
    # Tokenizer data files
    (str(SOURCE_DIR / 'data'), 'sepilot/data'),
]

# Hidden imports - LangChain/LangGraph use dynamic imports
hiddenimports = [
    # LangChain core
    'langchain',
    'langchain.agents',
    'langchain_core',
    'langchain_core.callbacks',
    'langchain_core.language_models',
    'langchain_core.messages',
    'langchain_core.output_parsers',
    'langchain_core.prompts',
    'langchain_core.runnables',
    'langchain_core.tools',
    'langchain_community',
    'langchain_openai',
    'langchain_openai.chat_models',

    # LangGraph
    'langgraph',
    'langgraph.graph',
    'langgraph.prebuilt',
    'langgraph.checkpoint',
    'langgraph.checkpoint.memory',
    'langgraph.errors',

    # Pydantic
    'pydantic',
    'pydantic.fields',
    'pydantic_core',

    # Rich
    'rich',
    'rich.console',
    'rich.panel',
    'rich.table',
    'rich.syntax',
    'rich.markdown',
    'rich.progress',

    # Other dependencies
    'httpx',
    'httpx._transports',
    'httpx._transports.default',
    'yaml',
    'dotenv',
    'click',
    'prompt_toolkit',
    'html2text',

    # Standard library
    'asyncio',
    'json',
    'pathlib',
    'subprocess',
    'typing',
    'dataclasses',

    # OpenAI
    'openai',
    'openai.resources',
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
]

# Exclude unnecessary packages to reduce size
excludes = [
    # GUI
    'tkinter',
    '_tkinter',
    'tcl',
    'tk',

    # Scientific computing (not needed)
    'matplotlib',
    'numpy',
    'pandas',
    'scipy',
    'PIL',
    'cv2',

    # ML frameworks (RAG only)
    'torch',
    'tensorflow',
    'transformers',
    'sentence_transformers',
    'chromadb',

    # Testing
    'pytest',
    'unittest',
    'test',

    # Development
    'IPython',
    'ipython',
    'jupyter',
    'notebook',

    # Documentation
    'sphinx',
    'docutils',
]

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    # Optimization level 1: remove asserts only (keep docstrings for @tool)
    # Note: optimize=2 removes docstrings which breaks LangChain @tool decorator
    optimize=1,
)

# Remove unnecessary binaries to reduce size
a.binaries = [x for x in a.binaries if not any(
    exclude in x[0].lower() for exclude in [
        'tcl', 'tk', '_tkinter',
        'qt', 'gtk',
        'libx11', 'libxcb', 'libxrender',
        'libgl', 'libegl',
    ]
)]

pyz = PYZ(
    a.pure,
    a.zipped_data,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='sepilot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Strip debug symbols
    upx=True,    # Enable UPX compression
    upx_exclude=[
        # Some libraries don't work well with UPX
        'libpython*.so*',
        'libcrypto*.so*',
        'libssl*.so*',
    ],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=True,  # Hide traceback details
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── sepilot-lsp ──────────────────────────────────────────────────────────────

LSP_ENTRYPOINT = SOURCE_DIR / "cli" / "lsp_entry.py"

a_lsp = Analysis(
    [str(LSP_ENTRYPOINT)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    optimize=1,
)

a_lsp.binaries = [x for x in a_lsp.binaries if not any(
    exclude in x[0].lower() for exclude in [
        'tcl', 'tk', '_tkinter',
        'qt', 'gtk',
        'libx11', 'libxcb', 'libxrender',
        'libgl', 'libegl',
    ]
)]

pyz_lsp = PYZ(
    a_lsp.pure,
    a_lsp.zipped_data,
)

exe_lsp = EXE(
    pyz_lsp,
    a_lsp.scripts,
    a_lsp.binaries,
    a_lsp.zipfiles,
    a_lsp.datas,
    [],
    name='sepilot-lsp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[
        'libpython*.so*',
        'libcrypto*.so*',
        'libssl*.so*',
    ],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
