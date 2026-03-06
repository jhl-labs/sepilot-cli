# MCP (Model Context Protocol) Integration for SEPilot

Claude Code 스타일의 MCP 서버 관리 및 Agent별 접근 제어 시스템입니다.

## 📋 목차

1. [개요](#개요)
2. [주요 기능](#주요-기능)
3. [설치](#설치)
4. [사용법](#사용법)
5. [Interactive 명령어](#interactive-명령어)
6. [Agent 통합](#agent-통합)
7. [접근 제어](#접근-제어)
8. [예시](#예시)

## 개요

SEPilot의 MCP 통합 시스템은 다음과 같은 기능을 제공합니다:

- **MCP 서버 관리**: MCP 서버 등록, 수정, 삭제, 활성화/비활성화
- **Agent별 접근 제어**: 각 Agent가 사용할 수 있는 MCP 서버 제어
- **우선순위 기반 권한 시스템**: Allow/Deny 리스트와 우선순위 관리
- **Interactive CLI**: `/mcp` 명령어를 통한 편리한 관리
- **영구 저장**: `~/.sepilot/mcp_config.json`에 설정 저장

## 주요 기능

### 1. MCP 서버 관리

```python
from sepilot.mcp import MCPConfigManager

# 설정 관리자 생성
manager = MCPConfigManager()

# MCP 서버 추가
manager.add_server(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    description="File system operations",
    enabled=True
)

# 서버 목록 조회
servers = manager.list_servers()

# 서버 정보 조회
server = manager.get_server("filesystem")

# 서버 수정
manager.update_server("filesystem", enabled=False)

# 서버 삭제
manager.remove_server("filesystem")
```

### 2. 접근 제어

```python
# GitHub agent에게만 허용
manager.allow_agent("filesystem", "github")

# Git agent 거부
manager.deny_agent("filesystem", "git")

# 모든 agent 거부
manager.deny_agent("filesystem", "all")

# 특정 agent만 허용 (나머지 모두 거부)
manager.deny_agent("filesystem", "all")
manager.allow_agent("filesystem", "github")
manager.allow_agent("filesystem", "git")

# 접근 권한 확인
can_access = manager.can_agent_access("filesystem", "github")
```

### 3. Agent 통합

Agent가 초기화될 때 MCP 도구를 자동으로 로드할 수 있습니다:

```python
from sepilot.mcp import get_mcp_tools_for_agent

# Agent 이름으로 접근 가능한 MCP 도구 가져오기
mcp_tools = get_mcp_tools_for_agent("github")

# 기존 도구와 병합
all_tools = standard_tools + mcp_tools
```

## 사용법

### Interactive 명령어

SEPilot Interactive 모드에서 `/mcp` 명령어를 사용합니다:

```bash
# SEPilot Interactive 모드 시작
sepilot -i

# MCP 서버 목록 보기
/mcp
/mcp list

# 도움말 보기
/mcp help

# 서버 상세 정보 보기
/mcp filesystem show
```

#### 서버 관리

```bash
# 새 MCP 서버 추가 (interactive)
/mcp add filesystem

# 서버 수정
/mcp edit filesystem

# 서버 활성화/비활성화
/mcp enable filesystem
/mcp disable filesystem

# 서버 삭제
/mcp remove filesystem
```

#### 접근 제어

```bash
# GitHub agent 허용
/mcp filesystem allow github

# Git agent 거부
/mcp filesystem deny git

# 모든 agent 허용
/mcp filesystem allow all

# 모든 agent 거부 (특정 agent만 허용할 때 사용)
/mcp filesystem deny all

# Allow 리스트 초기화
/mcp filesystem clear allow

# Deny 리스트 초기화
/mcp filesystem clear deny
```

## 접근 제어

### 우선순위 시스템

접근 제어는 다음 우선순위로 동작합니다:

1. **Allow 리스트** (최고 우선순위)
   - Agent가 Allow 리스트에 있으면 → **허용**

2. **Deny 리스트** (중간 우선순위)
   - Agent가 Deny 리스트에 있으면 → **거부**

3. **기본값** (최하 우선순위)
   - 둘 다 없으면 → **허용** (기본값)

### 예시 시나리오

#### 시나리오 1: 특정 Agent만 허용

```bash
# 모든 agent 거부
/mcp filesystem deny all

# GitHub와 Git만 허용
/mcp filesystem allow github
/mcp filesystem allow git
```

결과:
- ✅ GitHub agent: 접근 가능 (Allow 리스트)
- ✅ Git agent: 접근 가능 (Allow 리스트)
- ❌ SE agent: 접근 불가 (Deny 리스트의 'all')
- ❌ Wiki agent: 접근 불가 (Deny 리스트의 'all')

## 추가 정보

- MCP 프로토콜: https://modelcontextprotocol.io/
- SEPilot 문서: ../docs/

