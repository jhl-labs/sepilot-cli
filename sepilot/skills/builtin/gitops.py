"""GitOps Skill"""

from ..base import PromptSkill


class GitOpsSkill(PromptSkill):
    name = "gitops"
    description = "ArgoCD application monitoring, sync diagnosis, and Git-to-K8s traceability"
    triggers = ["argocd", "gitops", "argo sync", "sync failed", "argocd app"]
    category = "devops"
    prompt = """\
## GitOps / ArgoCD 모니터링 가이드

ArgoCD CLI를 사용하여 애플리케이션 상태를 모니터링하고 문제를 진단합니다.

### 애플리케이션 관리
```bash
argocd app list                                    # 전체 앱 목록
argocd app get <app-name>                          # 앱 상세 정보
argocd app get <app-name> --show-params            # 파라미터 포함
argocd app resources <app-name>                    # 리소스 목록
```

### 동기화 상태 확인
```bash
argocd app list -o json | jq '.[] | {name: .metadata.name, sync: .status.sync.status, health: .status.health.status}'
argocd app get <app-name> -o json | jq '.status.conditions'  # 에러 조건
argocd app diff <app-name>                         # Git vs Live 차이
```

### Git → K8s 트레이싱
특정 앱의 Git 커밋이 K8s에 어떻게 반영되었는지 추적:
1. `argocd app get <app>` → 현재 배포된 revision 확인
2. `git log` → 해당 revision의 커밋 내용 확인
3. `kubectl get` → 실제 K8s 리소스 상태 확인
4. 차이가 있으면 원인 분석

### 동기화 실패 진단
```bash
# 1. 동기화 상태 확인
argocd app get <app-name> -o json | jq '.status.sync'

# 2. 조건/에러 확인
argocd app get <app-name> -o json | jq '.status.conditions'

# 3. 리소스별 상태
argocd app resources <app-name>

# 4. K8s 이벤트 확인
kubectl get events -n <namespace> --sort-by='.lastTimestamp' --field-selector type=Warning
```

### 일반적인 동기화 실패 원인
- **ComparisonError**: Git 매니페스트 문법 오류
- **SyncError**: K8s API 거부 (RBAC, 리소스 제한)
- **HealthDegraded**: 배포는 되었으나 Pod 비정상
- **OutOfSync**: Git과 Live 상태 불일치

### 출력 지침
- 각 앱의 상태를 테이블로 표시 (앱명 | Sync | Health)
- 문제가 있는 앱은 🔴/🟡 표시
- 진단 결과에 **원인**, **영향**, **권장 조치** 포함
- 한국어로 결과 보고"""
