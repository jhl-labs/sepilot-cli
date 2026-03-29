# Changelog

이 프로젝트의 모든 주요 변경사항을 기록합니다. [Keep a Changelog](https://keepachangelog.com/) 형식을 따릅니다.

## [0.9.0] - 2026-03-30

### Added
- **멀티 에이전트 시스템** (`agent/multi/`): PM, Ralph Loop, Team, Inbox 기반 협업 에이전트 아키텍처
- **Tmux 서브에이전트**: tmux 세션 기반 격리 실행 지원 (`subagent/tmux_subagent.py`)
- **실행 컨텍스트 모듈** (`execution_context.py`): 현재 실행 경계 추적
- **CLI 에이전트 LLM 설정** (`config/cli_agent_llm.py`): CLI 전용 LLM 프로바이더 구성
- **DevOps 스킬 5종**: container, helm, gitops, k8s-health, se-helper를 Skills 시스템으로 이전
- **PromptSkill 기반 스킬 리팩토링**: fastapi_design, frontend_design을 마크다운 프롬프트로 분리
- **에이전트/성능 관리 명령어**: `/agent`, `/performance` 명령어 추가

### Changed
- `base_agent.py` 대규모 개선: 멀티 에이전트, tmux, 실행 컨텍스트 통합
- `instructions_loader.py` 리팩토링: 프로젝트/규칙/사용자 규칙 소스 분리 파라미터 추가
- `context_manager.py` 개선: predictive threshold, incremental compact 지원
- `subagent/` 모듈 전면 확장: team orchestrator, specialized agents 강화
- Builtin 스킬을 자동 디스커버리 방식으로 전환 (수동 등록 제거)
- `session_commands.py`, `undo_redo_commands.py` 등 UI 명령어 모듈 개선

### Removed
- `k8s_agent.py`: 전용 에이전트 제거 (Skills 시스템으로 대체)
- `devops_commands.py`, `k8s_commands.py`: 슬래시 명령어 제거 (Skills로 이전)
- `sepilot.web` 잔여 참조 (`estimate_cost`) 정리

## [0.8.1] - 2026-03-17

### Added
- **Karpathy 코딩 가이드라인 스킬**: LLM 코딩 실수를 줄이기 위한 행동 가이드라인 자동 주입
- **Investigation 플랜 템플릿**: 진단/조사/장애 분석 등 비코드 태스크를 위한 계층적 플래너 전략 추가 (`diagnose`, `investigate`)

### Changed
- 계층적 플래너 프롬프트를 "coding agent"에서 "software engineering agent"로 확장하여 인프라/운영 태스크 지원 개선
- `/clear` 명령 시 plan 관련 상태(`plan_steps`, `triage_decision` 등)도 함께 리셋하도록 개선

### Fixed
- Rich markup injection 방지: 에러 메시지에 `rich.markup.escape` 적용 (CLI 3곳)
- Legacy 빌드(manylinux2014) CI 수정: Rust 설치, 미지원 패키지 제외, 컨테이너 내 rename

### CI
- **Linux 호환성 테스트 자동화**: 릴리즈 시 6개 OS(CentOS 7 ~ Ubuntu 24.04)에서 바이너리 검증
- 릴리즈 노트에 호환성 테스트 결과 자동 첨부

## [0.8.0] - 2026-03-17

### Added
- **5-tier 멀티모델 라우팅**: reasoning_model, quick_model 등 용도별 모델 자동 라우팅
- **계층적 인스트럭션 로딩 시스템** (InstructionsLoader): 프로젝트/사용자/시스템 레벨 인스트럭션 지원
- **RulesLoader**: 규칙 기반 에이전트 행동 제어 모듈
- **WorktreeManager**: Git worktree 기반 서브에이전트 격리 실행
- **SKILL.md 마크다운 스킬 시스템**: Claude Code 호환 스킬 정의 및 실행
- **명령어 카탈로그 및 입력 유틸리티**: 체계적인 슬래시 명령어 관리
- **Vi 모드**: 대화형 입력에서 Vi 키바인딩 지원
- **Ubuntu 20.04 호환 Linux 빌드**: manylinux2014 (GLIBC 2.17+) 기반 바이너리 별도 배포

### Changed
- Hook 시스템: async 실행, SSRF 방지, API 보강, 싱글턴 스레드 안전성 개선
- MCP config_manager 심층 리뷰 후 12개 이슈 수정
- 컨텍스트 관리 및 압축을 Claude Code 수준으로 개선
- 자동완성 설명 표시, 도구 인라인 출력, 승인 UI 개선
- spinner/step_logger 라이프사이클을 헬퍼로 통합
- graph_mode 기본값을 enhanced로 복원

### Fixed
- spinner 재개 시 token 메트릭 유실 방지
- bash 스트리밍 도구 실행 후 spinner 미복구 버그
- hook blocked early return에서 spinner 복구 누락
- progress tracking 로컬 변수 재바인딩 버그
- compact_incremental fallback에서 메시지 내용 유실
- interactive.py 버그 3건 (NameError, 미등록 명령어, 입력 지연)
- 토큰 속도 계산 상한값 조정 (2000 → 500)
- 에이전트 기능 보안 강화 및 코드 중복 제거

## [0.6.0] - 2026-03-07

### Added
- Windows 빌드 지원 (GitHub Actions matrix에 windows-latest 추가)

### Changed
- Release workflow에 Linux, macOS, Windows 3개 플랫폼 빌드 통합

## [0.1.0] - 2026-03-07

### Added
- LangGraph 기반 ReAct 에이전트 구현
- Interactive REPL 모드 (슬래시 명령어, 자동완성, 세션 관리)
- Prompt 실행 모드 (단일 작업 실행, 파이프 입력, 출력 포맷)
- 내장 도구: 파일 I/O, Bash, Git, GitHub, 웹 검색, 코드 분석, 멀티미디어
- RAG (ChromaDB 기반 코드베이스 검색 증강 생성)
- MCP (Model Context Protocol) 서버 연동
- LSP 통합 (Python, TypeScript/JavaScript, Go, Rust)
- DevOps 명령 (Kubernetes, Docker, Helm)
- 세션 지속성 (Thread ID 기반 대화 재개)
- 자동 컨텍스트 관리 (경고 및 자동 압축)
- 에러 복구 (트랜지언트 LLM 실패 자동 재시도, 백트래킹)
- Undo/Redo 지원
- 멀티 LLM 프로바이더 지원 (OpenAI, Anthropic, Google, Ollama, AWS Bedrock 등)
- PyInstaller 기반 빌드 (sepilot, sepilot-lsp)
- SEPilot License v1.0
