#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$script_dir
bundle_dir=$repo_root/packages/bundle

err() {
  printf 'install-cli.sh: error: %s\n' "$1" >&2
  exit 1
}

log() {
  printf 'install-cli.sh: %s\n' "$1"
}

for tool in git node pnpm bun; do
  command -v "$tool" >/dev/null 2>&1 || err "need '$tool' on PATH to build from source."
done

[ -f "$bundle_dir/package.json" ] || err "not a repo checkout: $bundle_dir/package.json missing."
grep -q '"name": "@sepilotd/bundle"' "$bundle_dir/package.json" 2>/dev/null \
  || err "unexpected package.json at $bundle_dir — is this the sepilotd repo?"

os_raw=$(uname -s 2>/dev/null || echo unknown)
arch_raw=$(uname -m 2>/dev/null || echo unknown)
case "$os_raw" in
  Darwin) os=darwin ;;
  Linux) os=linux ;;
  *) err "unsupported OS '$os_raw' — only Linux and macOS can build locally. For Windows use install.ps1." ;;
esac
case "$arch_raw" in
  x86_64 | amd64) arch=x64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) err "unsupported architecture '$arch_raw' — supported: x86_64/amd64, aarch64/arm64." ;;
esac
host_target="${os}-${arch}"

INSTALL_DIR="${SEPILOT_INSTALL_DIR:-$HOME/.local/bin}"
dest="${INSTALL_DIR%/}/sepilot"

TARGET="${SEPILOT_TARGET:-$host_target}"
log "building target: $TARGET (host: $host_target)"
(cd "$repo_root" && pnpm --filter @sepilotd/bundle exec tsx scripts/build.ts "$TARGET") \
  || err "build failed for target '$TARGET' — see output above."

out_file="sepilot-$TARGET"
case "$TARGET" in
  windows-x64) out_file="$out_file.exe" ;;
esac
artifact="$bundle_dir/out/$out_file"
[ -f "$artifact" ] || err "build finished but artifact missing: $artifact."

if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  err "need 'sha256sum' or 'shasum' to verify the built artifact."
fi
expected=$(awk '{print $1; exit}' "$artifact.sha256" | tr 'A-F' 'a-f' | tr -d '\r\n')
actual=$(sha256_of "$artifact" | tr 'A-F' 'a-f')
if [ "$expected" != "$actual" ]; then
  err "checksum mismatch for $artifact
  expected: $expected
  actual:   $actual"
fi
log "checksum OK ($actual)"

if ! mkdir -p "$INSTALL_DIR" 2>/dev/null; then
  err "cannot create '$INSTALL_DIR'.
  Set SEPILOT_INSTALL_DIR to a writable location, e.g.:
    SEPILOT_INSTALL_DIR=\"\$HOME/.local/bin\" $0"
fi
if ! cp "$artifact" "$dest.new" 2>/dev/null; then
  err "cannot write to '$INSTALL_DIR'.
  Set SEPILOT_INSTALL_DIR to a writable location, e.g.:
    SEPILOT_INSTALL_DIR=/usr/local/bin sudo $0"
fi
chmod 0755 "$dest.new"
mv "$dest.new" "$dest" || err "failed to move the new binary into place at '$dest'."

case ":${PATH}:" in
  *":${INSTALL_DIR%/}:"*) on_path=1 ;;
  *) on_path=0 ;;
esac
if [ "$on_path" -ne 1 ]; then
  printf '\ninstall-cli.sh: NOTE — %s is not on your PATH.\n' "$INSTALL_DIR"
  printf '  Add it to your shell rc (e.g. ~/.zshrc):\n'
  printf '    export PATH="%s:$PATH"\n' "${INSTALL_DIR%/}"
fi

version_output=$("$dest" --version 2>&1 || true)
printf '\nsepilot installed to %s (%s).\n' "$dest" "${version_output:-version unknown}"
printf 'Run `sepilot --help` to get started.\n'
