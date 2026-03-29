"""SE Helper Skill"""

from ..base import PromptSkill


class SEHelperSkill(PromptSkill):
    name = "se-helper"
    description = "Software engineering quality analysis, static analysis, and architecture review"
    triggers = [
        "code quality", "static analysis", "architecture review",
        "ruff", "mypy", "pylint", "bandit", "lint",
    ]
    category = "development"
    prompt = """\
## 소프트웨어 엔지니어링 품질 분석 가이드

프로젝트의 코드 품질, 아키텍처, 보안을 분석합니다.

### 정적 분석 도구 실행
```bash
# Ruff (Python linter + formatter)
ruff check . --output-format=concise
ruff check . --statistics              # 규칙별 통계

# MyPy (타입 체크)
mypy . --ignore-missing-imports

# Pylint
pylint sepilot/ --output-format=colorized --score=yes

# Bandit (보안 분석)
bandit -r . -f screen -ll              # medium 이상만

# Black (포맷팅 체크)
black --check --diff .
```

### 자동 수정
```bash
ruff check . --fix                     # Ruff 자동 수정
ruff format .                          # Ruff 포맷팅
black .                                # Black 포맷팅
autopep8 --in-place --recursive .      # autopep8 수정
```

### 아키텍처 검토
프로젝트 아키텍처를 검토할 때:
1. **디렉토리 구조** 분석 (모듈 분리, 계층 구조)
2. **의존성 방향** 확인 (순환 의존성 탐지)
3. **관심사 분리** 평가 (UI/비즈니스/데이터 레이어)
4. **코드 중복** 탐지
5. **테스트 커버리지** 확인

### 개선 로드맵
분석 결과를 바탕으로:
- 🔴 즉시 수정 필요 (보안 취약점, 버그)
- 🟡 개선 권장 (코드 품질, 성능)
- 🟢 장기 과제 (아키텍처 리팩토링)

### 출력 지침
- 구체적인 파일:줄번호와 함께 문제 보고
- 수정 전/후 코드 예시 포함
- 우선순위별로 정렬
- 한국어로 결과 보고"""
