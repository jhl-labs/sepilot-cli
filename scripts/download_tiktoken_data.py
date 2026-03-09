#!/usr/bin/env python
"""Download tiktoken encoding data for offline bundling.

This script pre-downloads all tiktoken BPE encoding files into
sepilot/data/tiktoken/ so PyInstaller can bundle them.
At runtime, sepilot/__init__.py sets TIKTOKEN_CACHE_DIR to this
directory, allowing tiktoken to work fully offline.

Usage:
    python scripts/download_tiktoken_data.py
"""

import hashlib
import os
import re
import sys
import inspect
import urllib.request

def main():
    try:
        import tiktoken_ext.openai_public as pub
    except ImportError:
        print("ERROR: tiktoken not installed. Run: uv pip install tiktoken", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "sepilot", "data", "tiktoken"
    )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover all URLs from tiktoken_ext.openai_public encoding functions
    urls: dict[str, str] = {}  # cache_key -> url
    encoding_names = ["cl100k_base", "o200k_base", "p50k_base", "r50k_base", "p50k_edit", "gpt2"]

    for name in encoding_names:
        func = getattr(pub, name, None)
        if not func:
            continue
        src = inspect.getsource(func)
        found_urls = re.findall(r'"(https?://[^"]+)"', src)
        for url in found_urls:
            cache_key = hashlib.sha1(url.encode()).hexdigest()
            urls[cache_key] = url

    print(f"Downloading {len(urls)} tiktoken data files to {output_dir}")

    for cache_key, url in sorted(urls.items()):
        dest = os.path.join(output_dir, cache_key)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  SKIP  {cache_key} (already exists)")
            continue
        print(f"  GET   {cache_key} <- {url}")
        urllib.request.urlretrieve(url, dest)

    print("Done.")


if __name__ == "__main__":
    main()
