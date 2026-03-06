#!/bin/bash
# Build sepilot with optimization for distribution
#
# Features:
# - UPX compression (maximum level)
# - Debug symbols stripped
# - Python optimization level 1 (remove asserts, keep docstrings)
# - Unnecessary modules excluded

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
UPX_PATH="$HOME/.local/bin/upx"

cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           SEPILOT BUILD                                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  - Debug symbols stripped                                    ║"
echo "║  - Python optimization level 1 (remove asserts only)         ║"
echo "║  - UPX compression                                          ║"
echo "║  - Unnecessary modules excluded                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check dependencies
echo "[1/4] Checking dependencies..."

if ! command -v pyinstaller &> /dev/null; then
    echo "Error: pyinstaller not found. Install with: pip install pyinstaller"
    exit 1
fi

if [ ! -f "$UPX_PATH" ]; then
    echo "Warning: UPX not found at $UPX_PATH"
    echo "Installing UPX..."
    mkdir -p ~/.local/bin
    UPX_VERSION="4.2.4"
    curl -sL "https://github.com/upx/upx/releases/download/v${UPX_VERSION}/upx-${UPX_VERSION}-amd64_linux.tar.xz" -o /tmp/upx.tar.xz
    tar -xf /tmp/upx.tar.xz -C /tmp
    cp "/tmp/upx-${UPX_VERSION}-amd64_linux/upx" ~/.local/bin/
    rm -rf /tmp/upx*
    echo "UPX installed: $($UPX_PATH --version | head -1)"
fi

# Clean previous builds
echo ""
echo "[2/4] Cleaning previous builds..."
rm -rf build/ dist/ __pycache__/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Build with PyInstaller
echo ""
echo "[3/4] Building with PyInstaller..."
echo "      - Optimization: Level 1 (remove asserts, keep docstrings)"
echo "      - Strip: Enabled (remove debug symbols)"
echo ""

# Set UPX path for PyInstaller
export PATH="$HOME/.local/bin:$PATH"
export SEPILOT_SOURCE_DIR="$PROJECT_ROOT/sepilot"

# Run PyInstaller with protected spec
pyinstaller sepilot_protected.spec \
    --clean \
    --noconfirm \
    --log-level WARN \
    2>&1 | grep -v "^[0-9]* INFO:"

echo ""
echo "[4/4] Verifying build..."

FAILED=0

for BIN in sepilot sepilot-lsp; do
    if [ -f "dist/$BIN" ]; then
        SIZE=$(du -h "dist/$BIN" | cut -f1)
        if "./dist/$BIN" --help > /dev/null 2>&1; then
            echo "      $BIN ($SIZE): PASSED"
        else
            echo "      $BIN ($SIZE): FAILED"
            FAILED=1
        fi
    else
        echo "      $BIN: NOT FOUND"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "ERROR: Build verification failed!"
    exit 1
fi

# Result
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    BUILD COMPLETE                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Output:                                                     ║"

for BIN in sepilot sepilot-lsp; do
    if [ -f "dist/$BIN" ]; then
        SIZE=$(du -h "dist/$BIN" | cut -f1)
        printf "║    %-20s %s%*s║\n" "dist/$BIN" "$SIZE" $((33 - ${#SIZE})) ""
    fi
done

echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Applied:                                                    ║"
echo "║    ✓ Debug symbols stripped                                  ║"
echo "║    ✓ Assert statements removed (optimize=1)                  ║"
echo "║    ✓ UPX compression                                        ║"
echo "║    ✓ Unnecessary modules excluded                            ║"
echo "║    ✓ Docstrings preserved (required for @tool)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Test: ./dist/sepilot --help"
echo "      ./dist/sepilot-lsp --check"
