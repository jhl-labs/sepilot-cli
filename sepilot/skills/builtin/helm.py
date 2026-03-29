"""Helm Skill"""

from ..base import PromptSkill


class HelmSkill(PromptSkill):
    name = "helm"
    description = "Helm chart generation, linting, and Kubernetes deployment"
    triggers = ["helm", "helm chart", "helm install", "helm generate", "helm lint"]
    category = "devops"
    prompt = """\
## Helm Chart 관리 가이드

Helm CLI를 사용하여 차트를 생성, 검증, 배포합니다.

### Chart 생성
프로젝트를 분석하여 Helm Chart를 생성할 때:

1. **프로젝트 분석**
   - 언어/프레임워크 감지
   - 포트, 환경변수, 볼륨 요구사항 파악
   - 기존 Dockerfile/docker-compose.yml 참고

2. **Chart 구조 생성**
   ```
   chart-name/
   ├── Chart.yaml          # 차트 메타데이터
   ├── values.yaml         # 기본 설정값
   ├── templates/
   │   ├── deployment.yaml
   │   ├── service.yaml
   │   ├── ingress.yaml
   │   ├── configmap.yaml
   │   ├── hpa.yaml
   │   └── _helpers.tpl
   └── .helmignore
   ```

3. **모범 사례**
   - 리소스 제한 (requests/limits) 설정
   - 헬스체크 (liveness/readiness probe) 포함
   - ConfigMap/Secret으로 설정 분리
   - values.yaml에 충분한 주석 추가

### 검증
```bash
helm lint <chart-path>                 # Chart 문법 검증
helm template <chart-path>             # 렌더링 테스트
helm template <chart-path> | kubectl apply --dry-run=client -f -  # K8s 호환성 확인
```

### 배포
```bash
helm list -A                           # 설치된 릴리스 목록
helm install <name> <chart> -n <ns>    # 설치
helm upgrade <name> <chart> -n <ns>    # 업그레이드
helm rollback <name> <revision>        # 롤백
helm uninstall <name> -n <ns>          # 삭제
```

### 출력 지침
- 생성된 파일은 프로젝트 루트의 `helm/` 디렉토리에 저장
- 배포 전 반드시 `helm lint`로 검증
- 설치/업그레이드 전 사용자 확인 필요
- 한국어로 결과 보고"""
