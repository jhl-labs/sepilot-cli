"""Container Skill"""

from ..base import PromptSkill


class ContainerSkill(PromptSkill):
    name = "container"
    description = "Docker container management, Dockerfile generation, and diagnostics"
    triggers = [
        "docker", "container", "dockerfile", "docker compose",
        "docker image", "container logs",
    ]
    category = "devops"
    prompt = """\
## Docker 컨테이너 관리 가이드

Docker CLI를 사용하여 컨테이너/이미지를 관리하고 문제를 진단합니다.

### 이미지 관리
```bash
docker images                          # 이미지 목록
docker build -t <tag> .                # 이미지 빌드
docker rmi <image>                     # 이미지 삭제
```

### 컨테이너 관리
```bash
docker ps -a                           # 모든 컨테이너 목록
docker logs <container> --tail 100     # 최근 로그 확인
docker inspect <container>             # 상세 정보
docker exec -it <container> sh         # 컨테이너 접속
docker stop <container>                # 컨테이너 중지
```

### Docker Compose
```bash
docker compose ps                      # 서비스 상태
docker compose logs <service>          # 서비스 로그
docker compose up -d                   # 백그라운드 시작
docker compose down                    # 전체 중지
```

### Dockerfile 생성
프로젝트를 분석하여 최적화된 Dockerfile을 생성할 때:
1. 프로젝트 구조 확인 (언어, 프레임워크, 의존성)
2. 멀티스테이지 빌드 적용 (빌드 크기 최소화)
3. `.dockerignore` 포함
4. 보안 모범 사례 (non-root user, 최소 base image)
5. 레이어 캐싱 최적화 (의존성 먼저 복사)

### 진단 지침
문제 진단 시:
- `docker ps -a`로 컨테이너 상태 확인
- `docker logs`로 에러 로그 확인
- `docker inspect`로 네트워크/볼륨 설정 확인
- `docker system df`로 디스크 사용량 확인
- 각 문제에 대해 **원인**과 **해결 방법** 제시
- 한국어로 결과 보고"""
