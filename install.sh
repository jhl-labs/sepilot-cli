#!/usr/bin/env bash
# SEPilot installer
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/install.sh | bash -s -- --version v0.5
set -euo pipefail

REPO="jhl-labs/sepilot-cli"
INSTALL_DIR="${HOME}/.local/bin"
VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version|-v)
      VERSION="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: install.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -v, --version VERSION  Install a specific version (e.g. v0.6)"
      echo "  -h, --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Detect OS and architecture
detect_platform() {
  local os arch

  case "$(uname -s)" in
    Linux*)  os="linux" ;;
    Darwin*) os="darwin" ;;
    *)
      echo "Error: Unsupported OS: $(uname -s)"
      exit 1
      ;;
  esac

  case "$(uname -m)" in
    x86_64|amd64) arch="amd64" ;;
    *)
      echo "Error: Unsupported architecture: $(uname -m)"
      exit 1
      ;;
  esac

  echo "${os}-${arch}"
}

# Resolve version (latest or specified)
resolve_version() {
  if [[ -n "$VERSION" ]]; then
    echo "$VERSION"
    return
  fi

  echo "Fetching latest version..." >&2
  local latest
  latest=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')

  if [[ -z "$latest" ]]; then
    echo "Error: Failed to fetch latest version" >&2
    exit 1
  fi

  echo "$latest"
}

main() {
  local platform version

  platform=$(detect_platform)
  version=$(resolve_version)

  echo "Installing SEPilot ${version} (${platform})..."
  echo ""

  local base_url="https://github.com/${REPO}/releases/download/${version}"
  local sepilot_name="sepilot-${platform}"
  local lsp_name="sepilot-lsp-${platform}"

  # Create install directory
  mkdir -p "$INSTALL_DIR"

  # Download binaries
  echo "Downloading sepilot..."
  if ! curl -fSL --progress-bar "${base_url}/${sepilot_name}" -o "${INSTALL_DIR}/sepilot"; then
    echo "Error: Failed to download sepilot. Check if version '${version}' exists."
    exit 1
  fi

  echo "Downloading sepilot-lsp..."
  if ! curl -fSL --progress-bar "${base_url}/${lsp_name}" -o "${INSTALL_DIR}/sepilot-lsp"; then
    echo "Error: Failed to download sepilot-lsp."
    exit 1
  fi

  # Make executable
  chmod +x "${INSTALL_DIR}/sepilot" "${INSTALL_DIR}/sepilot-lsp"

  echo ""
  echo "Successfully installed to ${INSTALL_DIR}/"
  echo "  - sepilot"
  echo "  - sepilot-lsp"
  echo ""

  # Check if INSTALL_DIR is in PATH
  if [[ ":${PATH}:" != *":${INSTALL_DIR}:"* ]]; then
    echo "WARNING: ${INSTALL_DIR} is not in your PATH."
    echo ""
    echo "Add the following line to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo ""
    echo "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
    echo ""
    echo "Then reload your shell:"
    echo ""
    echo "  source ~/.bashrc  # or source ~/.zshrc"
    echo ""
  fi

  echo "Run 'sepilot --help' to get started."
}

main
