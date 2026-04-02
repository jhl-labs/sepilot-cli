<p align="center">
  <h1 align="center">SEPilot CLI</h1>
  <p align="center">
    소프트웨어 엔지니어링 작업을 위한 지능형 AI 에이전트
  </p>
</p>

<p align="center">
  <a href="https://github.com/jhl-labs/sepilot-cli/releases"><img src="https://img.shields.io/github/v/release/jhl-labs/sepilot-cli?style=flat-square" alt="Release"></a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey?style=flat-square" alt="Platform">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-SEPilot%20v1.0-green?style=flat-square" alt="License"></a>
</p>

---

LangGraph 기반 ReAct 패턴으로 코드 분석, 편집, 디버깅, DevOps 작업을 자연어로 수행합니다.
**멀티 LLM 프로바이더**(OpenAI, Anthropic, Google, Groq, AWS Bedrock, Azure, Ollama 등)를 지원하며,
대화형 REPL과 단일 프롬프트 실행 모드를 모두 제공합니다.

## 주요 특징

| | 기능 | 설명 |
|---|---|---|
| 💬 | **Interactive REPL** | 슬래시 명령어, 자동완성, 세션 지속성을 갖춘 대화형 인터페이스 |
| ⚡ | **Prompt 실행** | 단일 프롬프트 실행 후 종료, 스크립트·CI/CD 자동화에 적합 |
| 🔧 | **30+ 내장 도구** | 파일 I/O, Bash, Git, 웹 검색, 코드 분석, 이미지/PDF 읽기 등 |
| 🔌 | **MCP 통합** | Model Context Protocol로 외부 도구 서버 연동 |
| 🔍 | **RAG** | ChromaDB 기반 코드베이스 검색 증강 생성 |
| 🗂️ | **LSP 통합** | Python, TypeScript, Go, Rust 언어 서버 지원 |
| ☸️ | **DevOps** | Kubernetes, Docker, Helm 관리 내장 |
| 🔄 | **세션 관리** | Thread ID 기반 대화 재개 및 히스토리 |
| 📦 | **컨텍스트 관리** | 컨텍스트 윈도우 초과 시 자동 압축 |
| 🛡️ | **안전 실행** | Human-in-the-loop 승인, 권한 규칙, 트랜지언트 실패 자동 재시도 |

## 빠른 시작

### 설치

**바이너리 (권장)** — Python 환경 없이 바로 사용:

```bash
curl -fsSL https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/install.sh | bash
```

`~/.local/bin/`에 `sepilot`과 `sepilot-lsp`가 설치됩니다. PATH에 추가하세요:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

특정 버전을 설치하려면:

```bash
curl -fsSL https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/install.sh | bash -s -- --version v0.9.1
```

**소스에서 설치** — 개발 또는 기여 목적:

```bash
git clone https://github.com/jhl-labs/sepilot-cli.git
cd sepilot-cli
uv sync
uv pip install -e .

# 전체 기능 (RAG, 멀티미디어, 코드 분석 등)
uv pip install -e ".[all]"
```

### 환경 변수

사용할 LLM 프로바이더에 맞는 API 키를 설정합니다:

```bash
# OpenAI (기본)
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="..."

# 또는 프로젝트 루트에 .env 파일 생성
```

## 사용법

### Prompt 실행 모드

```bash
# 기본 사용
sepilot -p "프로젝트의 모든 Python 파일에서 TODO를 찾아줘"

# 위치 인자
sepilot "src/main.py를 읽고 문서를 생성해줘"

# 모델 지정
sepilot -m claude-3-opus-20240229 -p "이 코드의 버그를 찾아줘"

# 파이프 입력
cat error.log | sepilot --stdin -p "이 에러 로그를 분석해줘"

# 이전 세션 이어서 작업
sepilot -c -p "아까 작업 계속해줘"

# Git / GitHub 명령
sepilot --git "status"
sepilot --github "issue list"

# 출력 포맷 지정
sepilot -p "코드 줄 수 세기" -o json --print-cost
```

### Interactive 모드

```bash
sepilot -i                           # 대화형 REPL 시작
sepilot -i -m claude-3-opus-20240229 # 모델 지정하여 시작
```

#### 주요 슬래시 명령어

| 명령어 | 설명 |
|--------|------|
| `/help` | 사용 가능한 명령어 목록 |
| `/model set <key> <value>` | 모델 설정 변경 |
| `/model list` | 사용 가능한 모델 목록 |
| `/mode plan\|code` | 계획/코드 모드 전환 |
| `/context` | 현재 컨텍스트 사용량 표시 |
| `/compact` | 컨텍스트 수동 압축 |
| `/session new\|list` | 세션 관리 |
| `/resume` | 이전 세션 재개 |
| `/rag enable` | RAG 활성화 |
| `/mcp list\|add` | MCP 서버 관리 |
| `/tools` | 사용 가능한 도구 목록 |
| `/cost` | 현재 세션 비용 추정 |
| `/undo` / `/redo` | 파일 변경 되돌리기/다시하기 |
| `/graph` | 실행 그래프 시각화 |
| `/theme` | UI 테마 변경 |

`@` 입력으로 파일 경로 자동완성, 탭 키로 명령어 완성을 지원합니다.

## CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-p, --prompt TEXT` | 실행할 작업 프롬프트 | — |
| `-m, --model TEXT` | 사용할 LLM 모델 | `gpt-4-turbo-preview` |
| `-i, --interactive` | Interactive REPL 모드 | `false` |
| `-v, --verbose` | 상세 출력 | `false` |
| `-c, --continue-session` | 가장 최근 세션 재개 | `false` |
| `-r, --resume` | 세션 선택자 표시 | `false` |
| `-t, --thread-id TEXT` | 특정 세션 재개 | — |
| `--max-iterations N` | ReAct 루프 최대 반복 | `30` |
| `-pp, --prompt-profile` | 프롬프트 프로필 | `default` |
| `--fast` | 경량 5-노드 ReAct 그래프 | `false` |
| `--no-memory` | 메모리 시스템 비활성화 | `false` |
| `--git TEXT` | Git 명령 직접 실행 | — |
| `--github TEXT` | GitHub 명령 실행 | — |
| `-o, --output-format` | 출력 포맷 (`text`, `json`, `markdown`) | `text` |
| `--stdin` | 표준 입력에서 프롬프트 읽기 | `false` |
| `--print-cost` | 비용 추정치 출력 | `false` |

## 내장 도구

| 카테고리 | 도구 | 설명 |
|----------|------|------|
| **파일** | `file_read`, `file_write`, `file_edit`, `notebook_edit` | 파일/노트북 읽기·쓰기·편집 |
| **탐색** | `codebase`, `search_content`, `find_file`, `find_definition` | 코드베이스 탐색 및 심볼 검색 |
| **코드 분석** | `code_analyze` | 함수·클래스·심볼 구조 분석 |
| **셸** | `bash_execute`, `bash_background`, `bash_output` | 명령어 실행, 백그라운드 프로세스 |
| **Git** | `git` | status, diff, add, commit, log, branch 등 |
| **웹** | `web_search`, `web_fetch` | 웹 검색 및 콘텐츠 가져오기 |
| **멀티미디어** | `image_read`, `pdf_read` | 이미지/PDF 파일 읽기 |
| **계획** | `plan`, `todo_manage` | 작업 계획 수립 및 TODO 관리 |
| **기타** | `ask_user`, `apply_patch`, `subagent_execute` | 사용자 질의, 패치 적용, 서브에이전트 |

## 설정

### 설정 파일

```
~/.sepilot/
├── profiles/          # 모델 프로필
├── mcp_config.json    # MCP 서버 설정
└── config.yaml        # 전역 설정
```

### 주요 설정

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `model` | `gpt-4-turbo-preview` | 사용할 LLM 모델 |
| `max_tokens` | `4000` | 최대 응답 토큰 수 |
| `temperature` | `0.7` | 생성 온도 |
| `context_window` | `128000` | 컨텍스트 윈도우 크기 |
| `max_iterations` | `15` | ReAct 루프 최대 반복 |
| `timeout` | `300` | 작업 타임아웃 (초) |
| `graph_mode` | `enhanced` | 그래프 모드 (`enhanced` / `fast`) |
| `theme` | `default` | UI 테마 (`default`, `dark`, `light`, `monokai`) |
| `vi_mode` | `false` | Vi 키바인딩 |

### 모델 티어 라우팅

복잡도에 따라 다른 모델을 사용할 수 있습니다:

```bash
export SEPILOT_TRIAGE_MODEL="gpt-4o-mini"      # 작업 분류
export SEPILOT_QUICK_MODEL="gpt-3.5-turbo"      # 빠른 응답
export SEPILOT_REASONING_MODEL="claude-3-opus"   # 복잡한 추론
export SEPILOT_VERIFIER_MODEL="gpt-4o"           # 결과 검증
```

### 네트워크/프록시

```bash
export HTTP_PROXY="http://proxy:8080"
export HTTPS_PROXY="http://proxy:8080"
export NO_PROXY="localhost,127.0.0.1"
export SSL_VERIFY="true"
export SSL_CERT_FILE="/path/to/ca-bundle.crt"
```

### 지원하는 LLM 프로바이더

| 프로바이더 | 환경 변수 |
|-----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Ollama | `OLLAMA_BASE_URL` |
| GitHub Models | `GITHUB_TOKEN` |

## sepilot-lsp

코드 분석 기능 강화를 위한 LSP 서버 관리 도구입니다.

```bash
sepilot-lsp --check           # 설치 상태 확인
sepilot-lsp -l python          # Python LSP 설치
sepilot-lsp -l all             # 모든 LSP 설치
```

| 언어 | 서버 | 설치 명령 |
|------|------|----------|
| Python | pyright | `npm install -g pyright` |
| TypeScript | typescript-language-server | `npm install -g typescript typescript-language-server` |
| Go | gopls | `go install golang.org/x/tools/gopls@latest` |
| Rust | rust-analyzer | `rustup component add rust-analyzer` |

## 빌드

### 패키지 빌드

```bash
uv build                      # wheel + sdist
uv pip install -e .            # 개발 모드
```

### 바이너리 빌드 (PyInstaller)

```bash
./build_protected.sh           # sepilot + sepilot-lsp 바이너리 생성
```

빌드 결과물은 `dist/` 디렉토리에 생성됩니다. UPX 압축, 디버그 심볼 제거 등이 자동 적용됩니다.

## 기여하기

기여를 환영합니다! [CONTRIBUTING.md](CONTRIBUTING.md)를 참고해 주세요.

```bash
# 개발 환경 설정
git clone https://github.com/jhl-labs/sepilot-cli.git
cd sepilot-cli
uv sync && uv pip install -e ".[all]"
```

**브랜치 규칙:** `feat/`, `fix/`, `docs/`, `refactor/`, `test/`
**커밋 규칙:** [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:` 등)

## 라이선스

이 프로젝트는 [SEPilot License v1.0](LICENSE)으로 배포됩니다.

- **허용:** 개인 사용, 학습, 학술 연구, 비영리 목적
- **의무:** 변경/배포 시 소스 코드 공개, 동일 라이선스 적용, 저작자 표시
- **금지:** 상업적 사용 (별도 상업 라이선스 필요)
