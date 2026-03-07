# SEPilot Graph Architecture

SEPilot은 LangGraph 기반의 멀티-에이전트 코딩 시스템입니다. 이 문서는 그래프를 구성하는 20개의 노드와 라우팅 로직을 설명합니다.

## 전체 그래프 흐름

```
START
  |
triage ─────────────────────────────────────────────┐
  ├─ direct ──→ direct_response ──→ END             │
  ├─ simple ──→ iteration_guard ◄───────────────────┤
  └─ complex ─→ orchestrator                        │
                  ├─→ codebase_exploration ─┐       │
                  └─→ memory_retriever ─────┤       │
                                            ↓       │
                              hierarchical_planner  │
                                            ↓       │
                              iteration_guard ◄─────┘
                                    |
                              context_manager
                                    |
                              tool_recommender
                                    |
                                  agent
                                ╱       ╲
                          approval    verifier
                          ╱    ╲        ╱  |  ╲
                       tools  retry  reflection │ memory_writer
                         |      ↑    ╱ | ╲  ╲   │   ╱    ╲
                   tool_recorder│   │  │  │  └──┤  │   reporter→END
                         |      │   │  │  │     │  │
                       verifier─┘   │  │  backtrack_check
                                    │  │       ╱     ╲
                                    │  agent  │    debate_check
                                    │         │      ╱    ╲
                          hierarchical_planner│   debate  memory_writer
                                              │     |
                                              └─ memory_writer
                                                   ╱    ╲
                                          iteration_guard  reporter→END
```

## 노드 목록

### 진입 노드

| 노드 | 파일 | 설명 |
|------|------|------|
| **triage** | `base_agent.py` | 사용자 요청을 `direct`(직답), `simple`(단순), `complex`(복잡) 3가지 경로로 분류 |

### 사전 실행 노드 (Complex 경로 전용)

| 노드 | 파일 | 설명 |
|------|------|------|
| **orchestrator** | `pattern_nodes.py` | AdaptiveOrchestrator를 사용해 작업을 분석하고 실행 패턴을 선택 |
| **codebase_exploration** | `pattern_nodes.py` | FilePathDetector, CodebaseExplorer를 활용한 코드베이스 탐색 및 관련 파일 감지 |
| **memory_retriever** | `pattern_nodes.py` | MemoryBank에서 유사한 이전 작업 경험을 회상 |
| **hierarchical_planner** | `pattern_nodes.py` | 복잡한 작업을 다단계 서브태스크로 분해 |

> `codebase_exploration`과 `memory_retriever`는 `orchestrator` 이후 **병렬 실행**됩니다.

### 핵심 실행 루프

| 노드 | 파일 | 설명 |
|------|------|------|
| **iteration_guard** | `base_agent.py` | 반복 횟수 제한을 확인하여 `continue` 또는 `stop` 결정 |
| **context_manager** | `base_agent.py` | 토큰 사용량 최적화를 위한 컨텍스트 윈도우 관리 |
| **tool_recommender** | `pattern_nodes.py` | ToolLearningSystem 기반으로 현재 상황에 적합한 도구를 추천 |
| **agent** | `base_agent.py` | LLM을 사용한 메인 추론 노드. `tools`(도구 호출) 또는 `finalize`(완료) 결정 |
| **approval** | `base_agent.py` | bash, file_edit, git 등 민감한 도구 실행 전 사용자 승인 처리. `run_tools` 또는 `retry` 결정 |
| **tools** | `tool_executor.py` | LangGraph ToolNode 기반 실제 도구 실행 (52개 이상의 도구) |
| **tool_recorder** | `pattern_nodes.py` | ToolLearningSystem에 도구 호출 패턴을 기록 |

### 검증 및 반영 노드

| 노드 | 파일 | 설명 |
|------|------|------|
| **verifier** | `base_agent.py` | 도구 실행 결과를 검증하고 작업 완료도를 확인. `continue`, `fast_continue`, `report` 결정 |
| **reflection** | `reflection_node.py` | Reflexion 패턴 기반 자기비판. 실패 패턴 감지 및 전략 조정. `revise_plan`, `refine_strategy`, `proceed`, `escalate` 결정 |
| **backtrack_check** | `pattern_nodes.py` | BacktrackingManager를 사용해 실패 시 상태 롤백 여부 판단. `rollback` 또는 `continue` 결정 |
| **debate_check** | `pattern_nodes.py` | 다중 관점 분석이 필요한지 판단. `debate` 또는 `skip` 결정 |
| **debate** | `debate_node.py` | Proposer-Critic-Resolver 3역할 구조의 토론 기반 의사결정 (최대 2라운드) |

### 출구 노드

| 노드 | 파일 | 설명 |
|------|------|------|
| **memory_writer** | `pattern_nodes.py` | MemoryBank에 작업 경험을 저장. `continue`(다음 반복) 또는 `report`(보고) 결정 |
| **reporter** | `base_agent.py` | 최종 결과를 정리하여 보고. `END`로 이동 |
| **direct_response** | `base_agent.py` | 간단한 질문에 도구 없이 직접 응답. `END`로 이동 |

## 라우팅 상세

### Triage 라우팅
```
triage →
  ├─ "direct"  → direct_response → END
  ├─ "simple"  → iteration_guard (사전 실행 노드 스킵)
  └─ "complex" → orchestrator (전체 파이프라인)
```

### 핵심 실행 루프
```
iteration_guard →
  ├─ "continue" → context_manager → tool_recommender → agent
  └─ "stop"     → memory_writer

agent →
  ├─ "tools"    → approval
  └─ "finalize" → verifier

approval →
  ├─ "run_tools" → tools → tool_recorder → verifier
  └─ "retry"     → iteration_guard
```

### 검증-반영 체인
```
verifier →
  ├─ "continue"      → reflection
  ├─ "fast_continue"  → memory_writer (초기 반복에서 반영/백트랙 스킵)
  └─ "report"         → memory_writer

reflection →
  ├─ "revise_plan"     → hierarchical_planner → iteration_guard
  ├─ "refine_strategy" → agent
  ├─ "proceed"         → backtrack_check
  └─ "escalate"        → memory_writer

backtrack_check →
  ├─ "rollback"  → hierarchical_planner
  └─ "continue"  → debate_check

debate_check →
  ├─ "debate" → debate → memory_writer
  └─ "skip"   → memory_writer

memory_writer →
  ├─ "continue" → iteration_guard (다음 반복)
  └─ "report"   → reporter → END
```

## Reflection 노드 감지 패턴

| 패턴 | 설명 |
|------|------|
| `stuck_on_single_tool` | 같은 도구를 반복 사용 |
| `repeating_error` | 동일 에러 반복 발생 |
| `no_file_changes` | 예상된 파일 수정이 미실행 |
| `plan_execution_gap` | 계획은 있지만 실행하지 않음 |
| `tool_failure_cascade` | 연속적인 도구 실패 |
| `circular_reasoning` | 순환적 추론 반복 |
| `wrong_file_target` | 잘못된 파일을 편집 |
| `read_without_write` | 읽기만 반복 |
| `shallow_search` | 검색 후 읽지 않고 편집 |
| `overconfident_completion` | 불확실한 완료 주장 |

## Debate 노드 역할

| 역할 | 설명 |
|------|------|
| **ProposerAgent** | 해결책을 제안하고 이점과 위험을 분석 |
| **CriticAgent** | 제안의 약점, 보안, 성능 이슈를 비판적으로 분석 |
| **ResolverAgent** | 제안과 비판을 종합하여 최종 결정 (approve/reject/revise/escalate) |

## 주요 파일 경로

| 파일 | 역할 |
|------|------|
| `sepilot/agent/base_agent.py` | 메인 ReactAgent 및 그래프 구축 |
| `sepilot/agent/enhanced_state.py` | 그래프 상태(EnhancedAgentState) 정의 |
| `sepilot/agent/pattern_nodes.py` | 패턴 노드 팩토리 함수 |
| `sepilot/agent/reflection_node.py` | Reflection 노드 구현 |
| `sepilot/agent/debate_node.py` | Debate 노드 구현 |
| `sepilot/agent/memory_bank.py` | 경험 저장소 |
| `sepilot/agent/backtracking.py` | 상태 롤백 관리 |
| `sepilot/agent/tool_learning.py` | 도구 사용 패턴 학습 |
| `sepilot/agent/hierarchical_planner.py` | 다단계 작업 분해 |
| `sepilot/agent/pattern_orchestrator.py` | 패턴 자동 선택 |
| `sepilot/agent/tool_executor.py` | 도구 실행 엔진 |
