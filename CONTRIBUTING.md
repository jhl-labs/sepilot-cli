# Contributing to SEPilot CLI

SEPilot CLI에 기여해주셔서 감사합니다! 이 문서는 프로젝트에 기여하기 위한 가이드라인을 제공합니다.

## 시작하기

### 개발 환경 설정

```bash
# 1. 저장소 포크 및 클론
git clone https://github.com/<your-username>/sepilot-cli.git
cd sepilot-cli

# 2. 의존성 설치
uv sync

# 3. 개발 모드 설치
uv pip install -e ".[all]"

# 4. 정상 동작 확인
sepilot --help
sepilot-lsp --check
```

### 요구사항

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) 패키지 관리자

## 기여 방법

### 이슈 등록

- **버그 리포트**: 이슈 템플릿을 사용하여 재현 가능한 버그를 보고해주세요.
- **기능 요청**: 새로운 기능이나 개선사항을 제안해주세요.
- 이슈를 등록하기 전에 기존 이슈를 검색하여 중복을 확인해주세요.

### Pull Request

1. 관련 이슈가 없다면 먼저 이슈를 등록하여 논의합니다.
2. `main` 브랜치에서 기능 브랜치를 생성합니다.
   ```bash
   git checkout -b feat/my-feature main
   ```
3. 변경 사항을 커밋합니다 (아래 커밋 컨벤션 참고).
4. PR을 생성합니다.

### 브랜치 네이밍

| 접두사 | 용도 | 예시 |
|--------|------|------|
| `feat/` | 새로운 기능 | `feat/mcp-integration` |
| `fix/` | 버그 수정 | `fix/context-overflow` |
| `docs/` | 문서 변경 | `docs/update-readme` |
| `refactor/` | 리팩토링 | `refactor/agent-state` |
| `test/` | 테스트 추가/수정 | `test/tool-executor` |

### 커밋 컨벤션

[Conventional Commits](https://www.conventionalcommits.org/) 규칙을 따릅니다.

```
<type>: <description>

[optional body]
```

**타입:**

| 타입 | 설명 |
|------|------|
| `feat` | 새로운 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서 변경 |
| `refactor` | 리팩토링 (기능 변경 없음) |
| `test` | 테스트 추가/수정 |
| `chore` | 빌드, 설정 등 기타 변경 |
| `perf` | 성능 개선 |

**예시:**

```
feat: MCP 서버 자동 검색 기능 추가
fix: 컨텍스트 압축 시 메시지 유실 문제 수정
docs: LSP 설정 가이드 추가
```

## 코드 스타일

- Python 코드는 [PEP 8](https://peps.python.org/pep-0008/)을 따릅니다.
- 타입 힌트를 사용합니다.
- docstring은 기능이 복잡한 공개 함수에 작성합니다.

## 라이선스

기여한 코드는 프로젝트의 [SEPilot License v1.0](LICENSE)에 따라 배포됩니다. PR을 제출함으로써 이 라이선스 조건에 동의하는 것으로 간주합니다.
