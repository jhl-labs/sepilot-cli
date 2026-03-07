# Changelog

이 프로젝트의 모든 주요 변경사항을 기록합니다. [Keep a Changelog](https://keepachangelog.com/) 형식을 따릅니다.

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
