# sepilot-cli

[English](README.md) | 한국어

`sepilot-cli`는 로컬 퍼스트 AI 에이전트 CLI인 `sepilot`과, 세션·도구·승인·메모리·스킬·MCP
통합을 소유하는 로컬 데몬 `sepilotd`를 제공합니다. CLI는 HTTP와 WebSocket으로
데몬과 통신합니다.

## 주요 기능 (Features)

- **로컬 퍼스트 에이전트 데몬** — `sepilotd`가 사용자 머신에서 세션, 도구, 승인,
  메모리, 스킬, MCP 통합을 소유하고, `sepilot`은 HTTP와 WebSocket으로 데몬과
  통신하는 얇은 CLI 클라이언트입니다.
- **인터랙티브 터미널 UI** — Ink 기반 TUI에서 마크다운 렌더링, 아티팩트, 캔버스
  서피스를 갖춘 스트리밍 채팅을 제공합니다.
- **멀티 프로바이더 모델** — Anthropic, OpenAI, Ollama 등 교체 가능한 LLM
  프로바이더를 하나의 에이전트 루프 뒤에 연결합니다.
- **확장성** — 스킬, 플러그인, 익스텐션, 훅, MCP 서버를 지원합니다.
- **메모리 & 지식** — 벡터 검색이 가능한 SQLite 기반 메모리와 RAG, 개인 위키,
  스니펫, 개인/팀 문서를 제공합니다.
- **풍부한 도구 세트** — 영속 터미널(PTY), LSP, 브라우저 자동화, 문서·미디어
  처리, 샌드박스 실행을 지원합니다.
- **스케줄러 & 알림** — cron 스타일 작업과 리마인더를 Telegram 등 전달 채널과
  함께 제공합니다.
- **휴먼 인 더 루프 보안** — 권한 기반 도구 승인, 샌드박싱, 그리고 시크릿의
  저장소 유출 방지를 지원합니다.
- **운영 설계** — A2A/ACP 서피스를 갖춘 게이트웨이, 모니터링, 옵저버빌리티,
  진단, 백업, 마이그레이션, 보존 정책, 자가 업데이트를 제공합니다.
- **단독 바이너리** — 데몬을 내장한 하나의 자급형 `sepilot` 바이너리를 체크섬
  검증 스크립트로 macOS·Linux·Windows에 설치합니다.

## 라이선스 (License)

이 저장소는 [Sepilot Source-Available License](LICENSE)에 따라 **오픈 소스가 아닌
source-available** 소프트웨어입니다. 개인과 기업은 수정하지 않은 소프트웨어를
상업적 목적을 포함해 사용할 수 있습니다. 복사, 수정, 재배포 또는 다른 제품에서의
소스 사용은 저작권자의 사전 서면 허가가 필요합니다. 전체 조항은
[LICENSE](LICENSE)를 참고하세요.

## 릴리스 바이너리 설치

macOS 및 Linux:

```sh
curl -fsSL https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/packages/bundle/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/packages/bundle/scripts/install.ps1 | iex
```

각 설치 스크립트는 감지된 플랫폼의 최신 GitHub Release를 내려받아 설치 전에
SHA-256 체크섬을 검증합니다. 지원 릴리스 자산은 Linux(x64, arm64),
macOS(x64, arm64), Windows(x64)입니다.

## 소스에서 빌드

필요 요건: Node.js 22, pnpm 10, Bun(단독 바이너리 빌드 시에만 필요).

```sh
corepack enable
pnpm install --frozen-lockfile
pnpm build

# 빌드된 소스로 실행
node packages/daemon/dist/index.js
node packages/cli/dist/index.js --help

# 또는 이 머신용 단독 바이너리를 빌드해 설치
./install-cli.sh
```

단독 바이너리는 데몬을 내장하고 있습니다. `sepilot`은 필요할 때 데몬을
시작하며, `sepilot start`, `stop`, `restart`, `status`로 명시적으로 관리할 수
있습니다.

## 설정 (Configuration)

런타임 상태는 `~/.sepilotd/` 아래에 저장됩니다.
[`config.example.yaml`](config.example.yaml)을 `~/.sepilotd/config.yaml`로
복사하고, 자격 증명은 환경 변수 또는 로컬 설정 파일로 제공하세요. 자격 증명,
토큰, 런타임 상태를 저장소에 커밋하지 마세요.

## 패키지 구조 (Packages)

| 패키지 | 역할 |
| --- | --- |
| `@sepilotd/core` | 공유 계약과 도메인 타입 |
| `@sepilotd/api-client` | 타입이 적용된 데몬 HTTP/WebSocket 클라이언트 |
| `@sepilotd/presentation` | CLI 아티팩트·캔버스 렌더링 프리미티브 |
| `@sepilotd/daemon` | 로컬 데몬 런타임 |
| `@sepilotd/cli` | `sepilot` 명령어와 터미널 UI |
| `@sepilotd/bundle` | 자급형 `sepilot` 바이너리 빌더 |

## 개발 (Development)

```sh
pnpm typecheck
pnpm build
```

이슈나 풀 리퀘스트를 열기 전에 [CONTRIBUTING.md](CONTRIBUTING.md)를 먼저
읽어주세요. 취약점은 [SECURITY.md](SECURITY.md)에 설명된 대로 비공개로
보고해 주세요.

## 릴리스 (Releases)

`v1.0.5` 같은 태그를 푸시하면 릴리스 워크플로가 시작됩니다. 지원하는 모든
플랫폼의 바이너리와 SHA-256 체크섬 파일을 빌드한 뒤, 해당 자산을 담은 GitHub
Release를 생성합니다.

## 에이전트 모드와 그래프

데몬의 에이전트 실행 모드(`instant`, `react`, `auto`)와 내장 에이전트 그래프의
구조는 [GRAPH.md](GRAPH.md)를 참고하세요.