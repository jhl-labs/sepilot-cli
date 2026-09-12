#!/usr/bin/env sh
# install-cli.sh — 저장소 소스에서 스탠드얼론 `sepilot` 바이너리를 빌드해
# ~/.local/bin 에 원자적으로 설치한다.
#
# 사용법 (저장소 내부 어디서든):
#   ./install-cli.sh                                          # 호스트 트리플 자동 감지
#   SEPILOT_TARGET=darwin-arm64 ./install-cli.sh
#
# 동작:
#   1. 전제조건 확인: git, node, pnpm, bun (버전은 build.ts와 동일한 요구)
#   2. 스크립트 자기 위치에서 repo 루트/번들 디렉터리 해석
#   3. 빌드 대상 결정: SEPILOT_TARGET 오버라이드, 없으면 호스트 트리플
#      (macOS arm64 -> darwin-arm64 등; build.ts의 TARGETS 목록으로 검증)
#   4. 공식 파이프라인 실행: pnpm --filter @sepilotd/bundle build:binary <target>
#      (feature 매니페스트 → turbo 빌드 → vec 에셋 → bun --compile →
#      secret-scan 게이트 → 제외 스캔 → .sha256 → 스텁 복원)
#      ※ secret-scan 게이트는 비활성화하지 않는다 (CLAUDE.md 정책).
#   5. 산출물 packages/bundle/out/sepilot-<target> 을
#      $SEPILOT_INSTALL_DIR (기본 ~/.local/bin) 에 원자적 설치
#      (sepilot.new로 복사 후 같은 디렉터리 rename — 실행 중 교체도 안전)
#   6. 설치된 바이너리의 --version 출력 + PATH 힌트
#
# 기존 install.sh(릴리스 다운로드 전용)와 달리 로컬 소스 빌드 후 설치한다.
# 파이프(`| sh`) 실행은 지원하지 않는다 — 실제 체크아웃이 필요하다.
# CI 릴리스 흐름에는 포함되지 않는다.
#
# 환경변수 오버라이드:
#   SEPILOT_INSTALL_DIR   설치 대상 디렉터리 (기본: $HOME/.local/bin)
#   SEPILOT_TARGET        빌드 대상 (기본: 호스트 트리플, 예: darwin-arm64)
set -eu

# ── 위치 해석 ────────────────────────────────────────────────────────────────
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

# ── 전제조건 ────────────────────────────────────────────────────────────────
for tool in git node pnpm bun; do
  command -v "$tool" >/dev/null 2>&1 || err "need '$tool' on PATH to build from source."
done

# ── 저장소 확인 ────────────────────────────────────────────────────────────
[ -f "$bundle_dir/package.json" ] || err "not a repo checkout: $bundle_dir/package.json missing."
grep -q '"name": "@sepilotd/bundle"' "$bundle_dir/package.json" 2>/dev/null \
  || err "unexpected package.json at $bundle_dir — is this the sepilotd repo?"

# ── 빌드 대상 결정 ─────────────────────────────────────────────────────────
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

# ── 설치 위치 ──────────────────────────────────────────────────────────────
INSTALL_DIR="${SEPILOT_INSTALL_DIR:-$HOME/.local/bin}"
dest="${INSTALL_DIR%/}/sepilot"

# ── 빌드 ───────────────────────────────────────────────────────────────────
TARGET="${SEPILOT_TARGET:-$host_target}"
log "building target: $TARGET (host: $host_target)"
# 공식 진입점(build.ts)이 대상 검증·게이트·.sha256을 모두 수행한다.
(cd "$repo_root" && pnpm --filter @sepilotd/bundle exec tsx scripts/build.ts "$TARGET") \
  || err "build failed for target '$TARGET' — see output above."

out_file="sepilot-$TARGET"
case "$TARGET" in
  windows-x64) out_file="$out_file.exe" ;;
esac
artifact="$bundle_dir/out/$out_file"
[ -f "$artifact" ] || err "build finished but artifact missing: $artifact."

# ── 산출물 검증 ────────────────────────────────────────────────────────────
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

# ── 설치 ───────────────────────────────────────────────────────────────────
if ! mkdir -p "$INSTALL_DIR" 2>/dev/null; then
  err "cannot create '$INSTALL_DIR'.
  Set SEPILOT_INSTALL_DIR to a writable location, e.g.:
    SEPILOT_INSTALL_DIR=\"\$HOME/.local/bin\" $0"
fi
# 같은 디렉터리 내 rename이 원자적이므로 실행 중인 기존 설치도 안전하게 교체된다.
if ! cp "$artifact" "$dest.new" 2>/dev/null; then
  err "cannot write to '$INSTALL_DIR'.
  Set SEPILOT_INSTALL_DIR to a writable location, e.g.:
    SEPILOT_INSTALL_DIR=/usr/local/bin sudo $0"
fi
chmod 0755 "$dest.new"
mv "$dest.new" "$dest" || err "failed to move the new binary into place at '$dest'."

# ── 검증 + PATH 힌트 ───────────────────────────────────────────────────────
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