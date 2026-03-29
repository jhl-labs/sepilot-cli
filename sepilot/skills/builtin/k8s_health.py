"""Kubernetes Health Skill"""

from ..base import PromptSkill


class K8sHealthSkill(PromptSkill):
    name = "k8s-health"
    description = "Kubernetes cluster health monitoring, diagnosis, and troubleshooting"
    triggers = [
        "k8s health", "kubernetes health", "cluster health", "k8s status",
        "kubectl", "pod crash", "crashloopbackoff", "node not ready",
    ]
    category = "devops"
    prompt = """\
## Kubernetes 클러스터 건강 진단 가이드

kubectl을 사용하여 클러스터 상태를 점검하고 문제를 진단합니다.

### 점검 순서

#### 1. 클러스터 연결 확인
```bash
kubectl cluster-info
```

#### 2. 노드 상태 확인
```bash
kubectl get nodes -o wide
kubectl describe nodes | grep -A5 "Conditions:"
```
- Ready/NotReady 상태 확인
- 리소스 압박 (MemoryPressure, DiskPressure, PIDPressure) 확인

#### 3. Pod 건강 상태 확인
```bash
kubectl get pods --all-namespaces --field-selector=status.phase!=Running,status.phase!=Succeeded
kubectl get pods --all-namespaces | grep -E "CrashLoopBackOff|Error|ImagePullBackOff|Pending"
```
- 비정상 Pod 식별
- 재시작 횟수가 높은 Pod 확인 (5회 이상 주의)
- CrashLoopBackOff, ImagePullBackOff 등 문제 상태 확인

#### 4. 이벤트 확인
```bash
kubectl get events --all-namespaces --sort-by='.lastTimestamp' --field-selector type=Warning
```

#### 5. 서비스 엔드포인트 확인
```bash
kubectl get endpoints --all-namespaces | grep "<none>"
```
- 엔드포인트가 없는 서비스는 트래픽 라우팅 실패 의미

#### 6. 리소스 사용량 (metrics-server 필요)
```bash
kubectl top nodes
kubectl top pods --all-namespaces --sort-by=memory
```

### 진단 지침
- 특정 네임스페이스가 지정되면 해당 네임스페이스만 점검
- 각 문제에 대해 **원인**, **영향**, **권장 조치**를 제시
- 위험도를 분류: 🔴 Critical, 🟡 Warning, 🟢 Healthy
- 한국어로 결과를 보고

### 출력 형식

```
## 클러스터 건강 보고서

### 전체 상태: 🟢 Healthy / 🟡 Warning / 🔴 Critical

### 노드 상태
| 노드 | 상태 | 역할 | 버전 |
|------|------|------|------|

### 문제 발견 (있을 경우)
- 🔴/🟡 [문제 설명]
  - 원인: ...
  - 영향: ...
  - 권장 조치: ...

### 권장사항
1. ...
```"""
