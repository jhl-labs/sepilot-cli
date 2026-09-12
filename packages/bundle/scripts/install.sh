#!/usr/bin/env sh
# install.sh — download, verify, and install the standalone `sepilot` binary.
#
# One-liner:
#   curl -fsSL https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/packages/bundle/scripts/install.sh | sh
#
# Self-contained: assumes no repo checkout. Re-running upgrades in place.
#
# Environment overrides:
#   SEPILOT_INSTALL_DIR  install destination directory (default: $HOME/.local/bin)
#   SEPILOT_VERSION      "latest" (default) or a version like "0.2.10" / "v0.2.10"
#   SEPILOT_REPO         GitHub "owner/repo" (default: jhl-labs/sepilot-cli)
set -eu

REPO="${SEPILOT_REPO:-jhl-labs/sepilot-cli}"
VERSION="${SEPILOT_VERSION:-latest}"
INSTALL_DIR="${SEPILOT_INSTALL_DIR:-$HOME/.local/bin}"

err() {
  printf 'install.sh: error: %s\n' "$1" >&2
  exit 1
}

# ── Detect platform ─────────────────────────────────────────────────────────
os_raw="$(uname -s 2>/dev/null || echo unknown)"
case "$os_raw" in
  Linux) os=linux ;;
  Darwin) os=darwin ;;
  *) err "unsupported OS '$os_raw' — only Linux and macOS are supported. For Windows use install.ps1." ;;
esac

arch_raw="$(uname -m 2>/dev/null || echo unknown)"
case "$arch_raw" in
  x86_64 | amd64) arch=x64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) err "unsupported architecture '$arch_raw' — supported: x86_64/amd64, aarch64/arm64." ;;
esac

asset="sepilot-${os}-${arch}"

# ── Resolve download URLs ───────────────────────────────────────────────────
case "$VERSION" in
  latest)
    base_url="https://github.com/${REPO}/releases/latest/download"
    version_label="latest"
    ;;
  *)
    # Accept "0.2.10" or "v0.2.10"; the release tag is "v<version>".
    v="${VERSION#v}"
    base_url="https://github.com/${REPO}/releases/download/v${v}"
    version_label="v${v}"
    ;;
esac

bin_url="${base_url}/${asset}"
sha_url="${base_url}/${asset}.sha256"

# ── Pick a downloader ───────────────────────────────────────────────────────
if command -v curl >/dev/null 2>&1; then
  download() { curl -fsSL -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  download() { wget -qO "$2" "$1"; }
else
  err "need either 'curl' or 'wget' to download the binary."
fi

# ── Pick a sha256 tool ──────────────────────────────────────────────────────
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  err "need 'sha256sum' or 'shasum' to verify the download."
fi

# ── Download to a temp dir ──────────────────────────────────────────────────
tmp="$(mktemp -d 2>/dev/null || mktemp -d -t sepilot-install)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT INT TERM HUP

printf 'install.sh: downloading %s (%s) for %s-%s...\n' "$asset" "$version_label" "$os" "$arch"
download "$bin_url" "$tmp/$asset" || err "failed to download $bin_url — check the version/repo, or that the release has a $asset asset."
download "$sha_url" "$tmp/$asset.sha256" || err "failed to download $sha_url — the release is missing the checksum file."

# ── Verify sha256 ───────────────────────────────────────────────────────────
# The .sha256 file is either "<hex>  <filename>" (sha256sum format) or a bare
# "<hex>"; take the first whitespace-delimited token either way.
expected="$(awk '{print $1; exit}' "$tmp/$asset.sha256" | tr 'A-F' 'a-f' | tr -d '\r\n')"
[ -n "$expected" ] || err "could not read an expected SHA-256 from $asset.sha256."
actual="$(sha256_of "$tmp/$asset" | tr 'A-F' 'a-f')"
if [ "$expected" != "$actual" ]; then
  err "checksum mismatch for $asset
  expected: $expected
  actual:   $actual"
fi
printf 'install.sh: checksum OK (%s)\n' "$actual"

# ── Install ─────────────────────────────────────────────────────────────────
dest="${INSTALL_DIR%/}/sepilot"
if ! mkdir -p "$INSTALL_DIR" 2>/dev/null; then
  err "cannot create '$INSTALL_DIR'.
  Set SEPILOT_INSTALL_DIR to a writable location, e.g.:
    SEPILOT_INSTALL_DIR=\"\$HOME/.local/bin\" sh install.sh
  or, for a system-wide install:
    SEPILOT_INSTALL_DIR=/usr/local/bin sudo sh install.sh"
fi
# Copy into the install dir as `sepilot.new`, then rename over `sepilot` —
# `cp` (not `mv`) from the temp dir works across filesystems, and the final
# same-directory rename is atomic, so an existing install is replaced cleanly.
if ! cp "$tmp/$asset" "$dest.new" 2>/dev/null; then
  err "cannot write to '$INSTALL_DIR'.
  Set SEPILOT_INSTALL_DIR to a writable location, e.g.:
    SEPILOT_INSTALL_DIR=/usr/local/bin sudo sh install.sh"
fi
chmod 0755 "$dest.new"
mv "$dest.new" "$dest" || err "failed to move the new binary into place at '$dest'."

# ── PATH hint ───────────────────────────────────────────────────────────────
case ":${PATH}:" in
  *":${INSTALL_DIR%/}:"*) on_path=1 ;;
  *) on_path=0 ;;
esac
if [ "$on_path" -ne 1 ]; then
  printf '\ninstall.sh: NOTE — %s is not on your PATH.\n' "$INSTALL_DIR"
  printf '  Add it to your shell rc (e.g. ~/.bashrc or ~/.zshrc):\n'
  printf '    export PATH="%s:$PATH"\n' "${INSTALL_DIR%/}"
fi

printf '\nsepilot installed to %s. Run `sepilot --help` to get started.\n' "$dest"
