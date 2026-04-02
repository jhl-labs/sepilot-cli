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
단일 에이전트부터 **멀티 에이전트 팀 모드**까지 다양한 실행 방식을 제공합니다.

## 주요 특징

| | 기능 | 설명 |
|---|---|---|
| 💬 | **Interactive REPL** | 40+ 슬래시 명령어, 자동완성, 세션 지속성을 갖춘 대화형 인터페이스 |
| ⚡ | **Prompt 실행** | 단일 프롬프트 실행 후 종료, 스크립트·CI/CD 자동화에 적합 |
| 👥 | **멀티 에이전트 팀** | PM이 작업을 분해하여 Researcher, Developer, Tester 등에 분배 |
| 🖥️ | **Tmux 오케스트레이션** | Claude, Codex, Gemini 등 외부 에이전트를 tmux 세션으로 협업 |
| 🔧 | **30+ 내장 도구** | 파일 I/O, Bash, Git, 웹 검색, 코드 분석, 이미지/PDF 등 |
| 🔌 | **MCP 통합** | Model Context Protocol로 외부 도구 서버 연동 |
| 🔍 | **RAG** | ChromaDB 기반 코드베이스 검색 증강 생성 |
| 🗂️ | **LSP 통합** | Python, TypeScript, Go, Rust 언어 서버 지원 |
| ☸️ | **DevOps** | Kubernetes, Docker, Helm 관리 내장 |
| 🛡️ | **보안** | Human-in-the-loop 승인, 권한 규칙, 보안 스캔, 감사 로깅 |
| 📊 | **Effort 레벨** | Fast(5노드) / Enhanced(17노드) 그래프 런타임 전환 |
| 🏆 | **SWE-Bench** | SWE-Bench 벤치마크 자동 실행 및 평가 |

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
sepilot --git "ai-commit"              # AI 기반 커밋 메시지 생성
sepilot --github "issue list"

# 출력 포맷 지정
sepilot -p "코드 줄 수 세기" -o json --print-cost

# 경량 모드 (소형 모델에 최적)
sepilot --fast -p "간단한 질문"
```

### Interactive 모드

```bash
sepilot -i                           # 대화형 REPL 시작
sepilot -i -m claude-3-opus-20240229 # 모델 지정하여 시작
```

#### 슬래시 명령어

**모드 전환**

| 명령어 | 설명 |
|--------|------|
| `/plan` | PLAN 모드 — 읽기 전용 탐색, 코드 변경 없음 |
| `/code` | CODE 모드 — 읽기 + 쓰기 + Bash 실행 |
| `/exec` | EXEC 모드 — 읽기 + Bash 실행 (쓰기 없음) |
| `/auto` | AUTO 모드 — 시스템이 자동 판단 |
| `/effort` | Effort 레벨 전환 (Fast ⚡ / Enhanced 🧠) |

**세션 & 컨텍스트**

| 명령어 | 설명 |
|--------|------|
| `/session new\|list` | 세션 생성/목록 |
| `/resume` | 이전 세션 재개 |
| `/context` | 컨텍스트 사용량 표시 |
| `/compact [focus]` | 컨텍스트 압축 (선택적 포커스 지시) |
| `/cost` | 세션 비용 추정 |
| `/undo` / `/redo` | 파일 변경 되돌리기/다시하기 |
| `/rewind` | N개 교환 되돌리기 또는 전체 히스토리 |

**도구 & 통합**

| 명령어 | 설명 |
|--------|------|
| `/tools` | 사용 가능한 도구 목록 |
| `/model set\|list\|apply` | 모델 설정 변경/적용 |
| `/mcp list\|add\|enable\|disable` | MCP 서버 관리 |
| `/rag enable` | RAG 활성화 |
| `/graph [--xray]` | LangGraph 워크플로우 시각화 |
| `/skill list\|<name>` | 스킬 목록/실행 |

**보안 & 권한**

| 명령어 | 설명 |
|--------|------|
| `/permissions list\|add\|remove\|test` | 도구 실행 권한 규칙 관리 |
| `/security scan` | Bandit, pip-audit, detect-secrets 실행 |
| `/security ai-fix [issue]` | AI 기반 보안 취약점 수정 |
| `/security baseline save\|show\|diff` | 보안 기준선 관리 |
| `/yolo` | 자동 승인 모드 토글 |

**모니터링 & 통계**

| 명령어 | 설명 |
|--------|------|
| `/stats [session\|monthly\|all\|model]` | 토큰/비용 통계 |
| `/performance` | LLM 출력 속도 (tokens/sec) |
| `/effort` | Effort 레벨 표시/전환 |

**커스터마이징**

| 명령어 | 설명 |
|--------|------|
| `/theme list\|set <name>` | UI 테마 변경 (`default`, `dark`, `light`, `monokai`) |
| `/commands create\|list\|delete` | 커스텀 명령어 관리 |
| `/k8s` | Kubernetes 관련 명령 |
| `/bench` | SWE-Bench 벤치마크 실행 |

`@` 입력으로 파일 경로 자동완성, 탭 키로 명령어 완성을 지원합니다.

## 멀티 에이전트 팀 모드

PM(Project Manager) 에이전트가 작업을 분해하여 전문 에이전트에게 분배합니다.

```bash
# 벤치마크에서 팀 모드 사용
/bench run --team

# Interactive 모드에서 에이전트 오케스트레이션
/agent
```

### 역할

| 역할 | 설명 |
|------|------|
| **PM** | 작업 분해, 위임, 결과 취합 |
| **Researcher** | 코드베이스 분석, 정보 수집 |
| **Developer** | 코드 구현 및 수정 |
| **Tester** | 테스트 작성 및 실행 |
| **SecurityReviewer** | 보안 취약점 검토 |
| **Architect** | 설계 및 아키텍처 결정 |

각 역할에 맞는 모델 티어가 자동 적용됩니다 (PM/Researcher → reasoning 모델, Tester → verifier 모델).
에이전트별 Git worktree 격리를 통해 충돌 없이 병렬 작업합니다.

## Tmux 오케스트레이션

외부 CLI 에이전트(Claude Code, OpenCode, Codex, Gemini CLI 등)를 tmux 세션으로 관리하여 협업합니다.

```
tmux_create_session  — 에이전트 세션 생성
tmux_send            — 프롬프트 전송 및 응답 대기
tmux_read            — 현재 출력 읽기
tmux_status          — 전체 세션 상태 조회
tmux_orchestrate     — 다중 에이전트 병렬/순차 오케스트레이션
tmux_destroy         — 세션 종료
```

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
| `-pp, --prompt-profile` | 프롬프트 프로필 (`default`, `claude_code`, `codex`, `gemini`) | `default` |
| `--fast` | 경량 5-노드 Fast ReAct 그래프 | `false` |
| `--no-memory` | 메모리/체크포인트 비활성화 | `false` |
| `--git TEXT` | Git 명령 직접 실행 (`ai-commit` 포함) | — |
| `--github TEXT` | GitHub 명령 실행 | — |
| `-o, --output-format` | 출력 포맷 (`text`, `json`, `markdown`) | `text` |
| `--stdin` | 표준 입력에서 프롬프트 읽기 | `false` |
| `--print-cost` | 비용 추정치 출력 | `false` |
| `--list-threads` | 저장된 Thread ID 목록 출력 | `false` |

## 내장 도구

| 카테고리 | 도구 | 설명 |
|----------|------|------|
| **파일** | `file_read`, `file_write`, `file_edit`, `notebook_edit` | 파일/노트북 읽기·쓰기·편집 |
| **탐색** | `codebase`, `search_content`, `find_file`, `find_definition` | 코드베이스 탐색 및 심볼 검색 |
| **코드 분석** | `code_analyze` | 함수·클래스·심볼 구조 분석 |
| **셸** | `bash_execute`, `bash_background`, `bash_output`, `kill_shell`, `list_shells` | 명령어 실행, 백그라운드 프로세스 관리 |
| **Git** | `git` | status, diff, add, commit, log, branch, ai-commit |
| **웹** | `web_search`, `web_fetch` | 웹 검색 및 콘텐츠 가져오기 |
| **멀티미디어** | `image_read`, `pdf_read`, `multimedia_info` | 이미지/PDF 파일 읽기 및 정보 조회 |
| **계획** | `plan`, `todo_manage` | 작업 계획 수립 및 TODO 관리 |
| **Tmux** | `tmux_create_session`, `tmux_send`, `tmux_read`, `tmux_status`, `tmux_orchestrate`, `tmux_destroy` | 외부 에이전트 세션 관리 |
| **서브에이전트** | `subagent_execute` | 복잡한 작업을 하위 에이전트에 위임 |
| **기타** | `ask_user`, `apply_patch`, `think`, `slash_command`, `skill` | 사용자 질의, 패치, 스크래치패드, 슬래시 명령어, 스킬 실행 |

## 아키텍처

### 그래프 모드

SEPilot은 두 가지 LangGraph 실행 모드를 제공합니다:

- **Enhanced (기본)** — 17노드 파이프라인: 계획(Planning) → 실행(Execution) → 검증(Validation) → 반성(Reflection) → 토론(Debate). 복잡한 작업에 적합.
- **Fast** — 5노드 경량 ReAct 루프. 소형 모델이나 단순 작업에 최적화. `--fast` 플래그 또는 `/effort` 명령어로 전환.

### 고급 에이전트 패턴

- **Debate Node** — 다중 관점 의사결정
- **Reflection Node** — 자기 평가 및 개선
- **Pattern Orchestrator** — 작업 유형에 따른 적응형 패턴 선택
- **Backtracking** — 체크포인트 기반 롤백 (도구 실패, 테스트 실패, 빌드 실패 시 자동 복구)
- **Request Classifier** — 사용자 요청 자동 분류 (EXPLORE, IMPLEMENT, DEBUG, REFACTOR, TEST, DOCUMENT)

### 커스텀 명령어 & 스킬

**커스텀 명령어** — 반복 작업을 템플릿화:

```
~/.sepilot/commands/    # 전역
.sepilot/commands/      # 프로젝트별
```

변수 치환 (`$ARGUMENTS`, `$1`~`$N`, `$FILE`, `$SELECTION`), Bash 명령 삽입 (`` !`command` ``), 파일 참조 (`@file.py`) 지원.

**스킬** — 키워드 기반 자동 트리거:

```
~/.sepilot/skills/      # 전역
.sepilot/skills/        # 프로젝트별
```

내장 스킬: `code_review`, `debug_helper`, `test_writer`, `frontend_design`, `k8s_health`, `gitops`, `helm`, `container` 등.

## SWE-Bench 벤치마크

SWE-Bench 데이터셋을 자동으로 로드, 실행, 평가합니다.

```bash
# Interactive 모드에서
/bench instances load                    # 데이터셋 로드
/bench run --type verified --size 30     # Verified 30개 실행
/bench run --team --workers 4            # 팀 모드 + 병렬 4워커
/bench images build                      # Docker 이미지 사전 빌드
/bench evaluate                          # swebench 하네스 평가
/bench export                            # 결과 JSONL 내보내기
```

## 설정

### 설정 파일

```
~/.sepilot/
├── profiles/          # 모델 프로필
├── commands/          # 전역 커스텀 명령어
├── skills/            # 전역 스킬
├── mcp_config.json    # MCP 서버 설정
├── permissions.json   # 권한 규칙
├── audit.jsonl        # 감사 로그
└── config.yaml        # 전역 설정
```

프로젝트별 설정:

```
.sepilot/
├── commands/          # 프로젝트 커스텀 명령어
├── skills/            # 프로젝트 스킬
├── instructions.md    # 프로젝트별 시스템 프롬프트
└── rules.md           # 프로젝트별 규칙
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
| `enable_streaming` | `true` | 스트리밍 출력 |
| `theme` | `default` | UI 테마 (`default`, `dark`, `light`, `monokai`) |
| `vi_mode` | `false` | Vi 키바인딩 |
| `load_project_instructions` | `false` | `.sepilot/instructions.md` 로드 |
| `auto_verify_tests` | `false` | 코드 변경 후 자동 테스트 실행 |
| `test_command` | `pytest` | 자동 테스트 명령어 |
| `auto_verify_lint` | `false` | 코드 변경 후 자동 린트 실행 |
| `lint_command` | `ruff check .` | 자동 린트 명령어 |

### 모델 티어 라우팅

복잡도에 따라 다른 모델을 자동 라우팅합니다:

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

**빌드 타겟:**
- `sepilot-linux-amd64` / `sepilot-lsp-linux-amd64`
- `sepilot-darwin-amd64` / `sepilot-lsp-darwin-amd64`
- `sepilot-windows-amd64.exe` / `sepilot-lsp-windows-amd64.exe`

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
