# SEPilot CLI

소프트웨어 엔지니어링 작업을 위한 지능형 CLI 에이전트. LangGraph 기반 ReAct 패턴으로 코드 분석, 편집, 디버깅, DevOps 작업을 자연어로 수행합니다.

## 주요 특징

- **Interactive REPL 모드** — 슬래시 명령어, 자동완성, 세션 지속성을 갖춘 대화형 인터페이스
- **Prompt 실행 모드** — 단일 프롬프트 실행 후 종료, 스크립트 자동화에 적합
- **내장 도구** — 파일 I/O, Bash 실행, Git/GitHub 통합, 웹 검색, 코드 분석, 멀티미디어(이미지/PDF) 등
- **RAG** — ChromaDB 기반 코드베이스 검색 증강 생성
- **MCP (Model Context Protocol)** — 외부 도구 서버 연동
- **LSP 통합** — Python, TypeScript/JavaScript, Go, Rust 언어 서버 지원
- **DevOps** — Kubernetes, Docker, Helm 관리 명령 내장
- **세션 관리** — Thread ID를 통한 대화 재개 및 히스토리 관리
- **자동 컨텍스트 관리** — 컨텍스트 윈도우 초과 시 자동 압축
- **에러 복구** — 트랜지언트 LLM 실패 자동 재시도, 백트래킹

## 설치

### 요구사항

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (패키지 관리자)

### 소스에서 설치

```bash
git clone https://github.com/yourrepo/sepilot-cli.git
cd sepilot-cli

# 의존성 설치
uv sync

# 개발 모드 설치
uv pip install -e .

# 전체 기능 설치 (벤치마크, Docker 등 포함)
uv pip install -e ".[all]"
```

## 사용법

### Prompt 실행 모드

단일 작업을 실행하고 결과를 출력한 후 종료합니다. 스크립트나 파이프라인에서의 자동화에 적합합니다.

```bash
# 기본 사용
sepilot -p "프로젝트의 모든 Python 파일에서 TODO를 찾아줘"

# 위치 인자로도 사용 가능
sepilot "src/main.py를 읽고 문서를 생성해줘"

# 모델 지정
sepilot -m gpt-oss:120b-cloud -p "이 코드의 버그를 찾아줘"

# 상세 출력
sepilot -p "코드 리팩토링" -v

# 출력 포맷 지정 (text, json, markdown)
sepilot -p "코드 줄 수 세기" -o json --print-cost

# 파이프에서 입력 받기
cat error.log | sepilot --stdin -p "이 에러 로그를 분석해줘"

# 이전 세션 이어서 작업
sepilot -c -p "아까 작업 계속해줘"

# Git 명령 직접 실행
sepilot --git "status"

# GitHub 명령 실행
sepilot --github "issue list"
```

### Interactive 모드

`-i` 플래그로 대화형 REPL 모드를 시작합니다. 지속적인 대화를 통해 복잡한 작업을 단계별로 수행할 수 있습니다.

```bash
sepilot -i
sepilot -i -m gpt-oss:120b-cloud       # 모델 지정하여 시작
```

#### Interactive 모드의 주요 기능

**슬래시 명령어**

| 명령어 | 설명 |
|--------|------|
| `/help` | 사용 가능한 명령어 목록 |
| `/model set <key> <value>` | 모델 설정 변경 (예: api_key, model) |
| `/model apply` | 변경된 모델 설정 적용 |
| `/model list` | 사용 가능한 모델 목록 |
| `/mode plan` | 계획 모드로 전환 (코드 변경 없이 분석만) |
| `/mode code` | 코드 모드로 전환 (실행 포함) |
| `/context` | 현재 컨텍스트 사용량 표시 |
| `/compact` | 컨텍스트 수동 압축 |
| `/session new` | 새 세션 시작 |
| `/session list` | 저장된 세션 목록 |
| `/resume` | 이전 세션 재개 |
| `/rag enable` | RAG 활성화 |
| `/mcp list` | MCP 서버 목록 |
| `/mcp add <name> <command>` | MCP 서버 추가 |
| `/tools` | 사용 가능한 도구 목록 |
| `/k8s` | Kubernetes 관련 명령 |
| `/cost` | 현재 세션 비용 추정 |

**자동완성 및 파일 참조**

- `@` 입력 시 파일 경로 자동완성
- 탭 키로 명령어 및 경로 자동완성
- 다중 라인 입력 지원

**세션 지속성**

- 대화 내용이 자동으로 저장되어 나중에 재개 가능
- Thread ID를 통해 특정 세션으로 복귀
- `/session` 명령어로 세션 관리

**컨텍스트 관리**

- 컨텍스트 사용률 80% 도달 시 경고
- 92% 도달 시 자동 압축 (LLM 기반 요약, 환경변수로 임계값 조정 가능)
- `/compact` 명령어로 수동 압축 가능

**Undo/Redo**

- 파일 변경 작업의 되돌리기/다시하기 지원

### 주요 CLI 옵션

| 옵션 | 설명 |
|------|------|
| `-p, --prompt TEXT` | 실행할 작업 프롬프트 |
| `-m, --model TEXT` | 사용할 LLM 모델 |
| `-i, --interactive` | Interactive REPL 모드 |
| `-v, --verbose` | 상세 출력 |
| `-t, --thread-id TEXT` | 특정 세션 재개 |
| `-c, --continue-session` | 가장 최근 세션 재개 |
| `-r, --resume` | 세션 선택자 표시 |
| `--list-threads` | 저장된 Thread ID 목록 |
| `--max-iterations N` | ReAct 루프 최대 반복 (기본: 30) |
| `-pp, --prompt-profile` | 프롬프트 프로필 (default, claude_code, codex, gemini) |
| `--no-memory` | 메모리 시스템 비활성화 |
| `--git TEXT` | Git 명령 직접 실행 |
| `--github TEXT` | GitHub 명령 실행 |
| `-o, --output-format` | 출력 포맷 (text, json, markdown) |
| `--stdin` | 표준 입력에서 프롬프트 읽기 |
| `--print-cost` | 비용 추정치 출력 |
| `--log-dir TEXT` | 로그 디렉토리 (기본: ./logs) |

## 내장 도구

SEPilot은 다양한 도구를 에이전트에게 제공하여 실제 작업을 수행합니다.

| 카테고리 | 도구 | 설명 |
|----------|------|------|
| **파일** | Read, Write, Edit, Glob, Search | 파일 읽기/쓰기/편집/검색 |
| **셸** | Bash, BackgroundShell | 명령어 실행, 백그라운드 프로세스 |
| **Git** | GitTool | commit, push, pull, branch, diff 등 |
| **웹** | WebSearch, WebFetch | 웹 검색 및 페이지 가져오기 |
| **코드 분석** | CodeAnalysis | 구조 분석, 심볼 검색, 정의 찾기 |
| **멀티미디어** | ImageRead, PDFRead | 이미지/PDF 파일 읽기 |
| **계획** | PlanMode | 코드 변경 없이 분석/계획 수립 |
| **작업** | TaskManager | TODO 관리, 플랜 작성 |

## 설정

### 설정 파일 위치

```
~/.sepilot/
├── profiles/          # 모델 프로필 저장
└── mcp_config.json    # MCP 서버 설정
```

### 주요 설정 항목

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `model` | — | 사용할 LLM 모델명 |
| `max_tokens` | 4000 | 최대 응답 토큰 수 |
| `temperature` | 0.7 | 생성 온도 |
| `context_window` | 128000 | 컨텍스트 윈도우 크기 |
| `max_iterations` | 15 | ReAct 루프 최대 반복 (CLI에서는 `--max-iterations` 기본값 30) |
| `timeout` | 300 | 작업 타임아웃 (초) |
| `tool_cache_size` | 100 | 도구 결과 캐시 크기 |
| `tool_cache_ttl` | 300 | 캐시 TTL (초) |

### 네트워크/프록시 설정

```bash
export HTTP_PROXY="http://proxy:8080"
export HTTPS_PROXY="http://proxy:8080"
export NO_PROXY="localhost,127.0.0.1"
export SSL_VERIFY="true"
export SSL_CERT_FILE="/path/to/ca-bundle.crt"
```

## sepilot-lsp

`sepilot-lsp`는 코드 분석 기능 강화를 위한 LSP(Language Server Protocol) 서버 관리 도구입니다. `sepilot`과 함께 설치되며, 언어 서버를 설치하면 에이전트가 코드 구조 분석, 심볼 검색, 정의 이동 등 고급 코드 인텔리전스 기능을 활용할 수 있습니다.

### 지원 언어 서버

| 언어 | 서버 | 설치 명령 |
|------|------|----------|
| Python | pyright | `npm install -g pyright` |
| TypeScript/JavaScript | typescript-language-server | `npm install -g typescript typescript-language-server` |
| Go | gopls | `go install golang.org/x/tools/gopls@latest` |
| Rust | rust-analyzer | `rustup component add rust-analyzer` |

### 사용법

```bash
# 설치 상태 확인
sepilot-lsp --check

# 특정 언어 LSP 설치
sepilot-lsp -l python
sepilot-lsp -l typescript
sepilot-lsp -l go
sepilot-lsp -l rust

# 모든 LSP 한번에 설치
sepilot-lsp -l all
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `-l, --language TEXT` | 설치할 언어 서버 (python, typescript, go, rust, all) |
| `-c, --check` | 설치된 서버 상태 확인 |
| `--help` | 도움말 표시 |

### 사용 예시

```bash
$ sepilot-lsp --check
LSP Server Status:

  python: ✓ Installed
    Server: pyright

  typescript: ✗ Not installed
    Server: typescript-language-server
    Install: npm install -g typescript typescript-language-server

  go: ✗ Not installed
    Server: gopls
    Install: go install golang.org/x/tools/gopls@latest

  rust: ✗ Not installed
    Server: rust-analyzer
    Install: rustup component add rust-analyzer
```

## 빌드

### uv (개발/배포 패키지)

```bash
# 패키지 빌드 (wheel + sdist)
uv build

# 개발 모드 설치 (sepilot, sepilot-lsp 모두 사용 가능)
uv pip install -e .
```

### PyInstaller (단일 실행 파일)

배포용 독립 바이너리를 생성합니다. `sepilot`과 `sepilot-lsp` 두 바이너리가 모두 빌드됩니다.

```bash
# 빌드 스크립트 실행 (의존성 확인, 빌드, 검증 자동 수행)
./build_protected.sh

# 또는 직접 빌드
pyinstaller sepilot_protected.spec --clean
```

빌드 결과물:
- `dist/sepilot` — 메인 CLI 에이전트
- `dist/sepilot-lsp` — LSP 서버 관리 도구

빌드 시 적용되는 최적화:
- UPX 압축
- 디버그 심볼 제거
- Python assert 문 제거 (optimize=1)
- 불필요한 모듈 제외
- docstring 보존 (LangChain `@tool` 데코레이터에 필요)

## 라이선스

이 프로젝트는 [SEPilot License v1.0](LICENSE)으로 배포됩니다.

- **허용**: 개인 사용, 학습, 학술 연구, 비영리 목적
- **의무**: 변경/배포 시 소스 코드 공개, 동일 라이선스 적용, 저작자 표시
- **금지**: 상업적 사용 (별도 상업 라이선스 필요)

