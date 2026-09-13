# Agent Modes & Graph Architecture

This document describes how the daemon (`packages/daemon`) executes agent
runs: the execution modes offered to callers, how `auto` routing picks one,
and the structure of the built-in agent graphs.

The mode catalog lives in `packages/daemon/src/agent/mode-catalog.ts`; the
graph runtime lives in `packages/daemon/src/agent/graph/`.

## Execution modes

`packages/daemon/src/agent/mode-catalog.ts` is the single source of truth for
the modes that semantic routing can choose between. The advisory intent router
(explicit `auto`) and the in-run `agent.transfer` catalog read the same list,
so the cheap classifier and the executing model are always told the same
modes.

Two engine modes are always offered:

| Mode | Description | Capabilities |
| --- | --- | --- |
| `instant` | Direct answers from current context, focused memory recall and memory persistence; no workspace, web or process tools. Transfers when the goal needs more. | `application-state` |
| `react` | General-purpose tool loop without a planning graph: one-shot questions, focused investigation, code search and inspection, a single precise edit, native attachments. | — |

Beyond the two engine modes, `listSemanticModes()` adds every registered graph
that is auto-routable (see below). Registered graphs are never hard-coded into
the candidate list.

### Auto-routability rules

`isAutoRoutableGraph()` decides which registered graphs join the candidate
list:

- `react` and `instant` graph ids are excluded (the engine modes are already in the list).
- `persona-panel` is always excluded — it needs an explicit roster from the caller; auto-selecting it with an empty roster degrades to a bare LLM turn without the normal system prompt, memory, or tool surface.
- `computer-use` is excluded unless the daemon runs on Windows (`win32`); the graph depends on the Windows automation backend.
- Graphs with `capabilities.direct` are excluded — they are deterministic, narrow capability paths.
- Graphs with an explicit `capabilities.autoRoutable` value use that value.
- Otherwise: builtin and capability graphs are not routable; user, YAML, and plugin graphs are routable by default ("their author registered them to be chosen").

### Fallback when classification fails

`resolveAutoFallbackMode()` decides the safe entry point when the router
cannot classify an intent:

- Active remote browser → `react`
- Surface in `WORKSPACE_SHELL_SURFACES` (`cli`, `desktop`, `web`) → `react` — workspace shells sit inside a repository, so an unclassified turn is more likely to need tools than not
- Everywhere else (channels, mobile, bare HTTP) → `instant`

## Graph registry

`graph/registry.ts` keeps `GraphAgentInfo` records in a `GraphAgentRegistry`:

- `id`, `name`, `description`, and a `builder` (a `GraphBuilder` that assembles an `AgentGraph` from deps)
- `source`: `builtin` | `yaml` | `plugin` | `capability` | `user`
- `limits`: e.g. `maxIterations`
- `capabilities`:

| Capability | Meaning |
| --- | --- |
| `artifactWrite` | Can create or update durable user-requested artifacts (reports, markdown, pages, repository files). |
| `nativeMultimodalInput` | Consumes structured `currentUserContent`, preserving its tool boundary for attachments. |
| `readOnly` | Intended to stay read-only unless a caller explicitly wraps it in a writer-capable flow. |
| `direct` | Deterministic narrow capability path; auto mode may run it without hydrating long-term memory or document context first. |
| `durableRunContract` | Attach the durable acceptance-criteria contract and completion gate (defaults to `true`). |
| `autoRoutable` | Offer this graph to semantic routing (builtin defaults to false; user/YAML/plugin default to true). |
| `executionCapabilities` | Semantic execution capabilities the graph can fulfill end-to-end (e.g. `process`, `terminal`, `filesystem-read`, `browser`, `network`). |

At registration time the registry builds each graph once against a stub deps
proxy (node builders only capture deps in closures, so an all-undefined proxy
is enough) and runs `validateGraph()`. Missing start nodes or edge targets
that do not exist fail registration immediately instead of surfacing as a
confusing routing error on the first run.

Non-builtin graphs come from:

- `graph/yaml-loader.ts` — YAML-defined graphs (`source: 'yaml'`)
- `plugins/loader.ts` — plugin graphs (`source: 'plugin'`)
- `agent/user-agents/loader.ts` — user-defined agents (`source: 'user'`)

## Runtime engine

`graph/engine.ts` executes an `AgentGraph`:

- **Nodes** — added with `addNode(name, fn, meta)`. Meta carries a `lifecycleState` (`thinking`, `acting`, `observing`, `done`), an optional `resumeStage` (maps to the run-resume stages `thinking` / `acting` / `observing` used by session run checkpoints), and `pendingToolExecutionNode` for nodes that park a run while awaiting tool approval.
- **Edges** — `addEdge(from, to)` plus `addConditionalEdge(from, router, allowedTargets)`. A router returning an undeclared target raises `GraphRoutingError`.
- **Streaming nodes** — a node function may return an async iterator (`StreamingNodeFn`); the engine detects this and streams intermediate events.
- **Checkpoints** — `graph/checkpoints.ts` provides `createGraphRunCheckpoint` / `cloneGraphState` so runs can resume mid-graph at the right stage.
- **State board** — `graph/state-board.ts` builds a live steering board (`AgentSteeringNote` items); `state_board` stream events are bounded (20 items, 120 chars per item) while the full board stays in the journal and the `/state` endpoint. `state-board-journal.ts` / `state-board-recover.ts` handle persistence and recovery.
- **Subgraphs** — `graph/subgraph.ts` wraps a child `AgentGraph` as a node (`agentSubgraphNode`) with explicit state mapping in/out (`childStateFrom`, `mergeChildInto` in `subgraph-state.ts`) and child iteration budgets (`iteration-budget.ts`). If a child graph already surfaced a terminal error, the parent stops via `GraphChildHaltError` without re-emitting the error event.
- **Stop reasons** — runs end with a typed `RunStopReason`: completed, budget, observation budget, no progress, stuck repeat, completion gate, user action required, approval denied, cost gate, provider error, user abort, wall clock, incomplete output.
- **Iteration budget** — `graph/iteration-budget.ts` splits the parent's iteration budget with specialist children (`childIterationBudget`).

## Node library

`graph/nodes.ts` implements the reusable node functions the presets compose,
including `memoryRetriever`, `planner`, `iterationGuard`, `contextManager`,
`agent` (the LLM turn with ReAct prompt/tool-call envelopes and answer
protocol repair), `toolExecutor`, `reflection`, `reporter`, `triage`,
`capabilityScout`, `specialistRouter`, `autoDecompose`, `codebaseExplorer`,
and the capture nodes for exploration/scout results.

Supporting machinery around the loop lives in dedicated modules:
`completion-gate.ts` (acceptance-criteria completion gate), `evidence-ledger.ts`,
`open-questions.ts`, `planner-working-memory.ts`, `rollup-findings.ts`,
`debate.ts` (review debate), `validator.ts`, and `analysis.ts`.

## Built-in graphs

`graph/presets/index.ts` registers 16 builtin graphs:

| Id | Name | Description | maxIterations | Capabilities |
| --- | --- | --- | --- | --- |
| `chat-only` | Chat Only (ChatGPT) | Conversation with read-only RAG and stored knowledge. No actions, writes, delegation, or mode transfers. | 8 | readOnly, no durable run contract, native multimodal input |
| `enhanced` | Enhanced Agent | General-purpose orchestration for mixed or unclear requests; triages internally and routes to simple, coder, researcher, creative, or reviewer specialists | — | artifactWrite, full execution capabilities |
| `reviewer` | Code Reviewer | Review code for bugs, security issues, and improvements with debate quality gate | — | readOnly, filesystem-read |
| `coder` | Coder Agent | Code implementation, validation, debugging and review | 36 | artifactWrite, autoRoutable, native multimodal; process/terminal/fs-read/fs-write/network |
| `codebase-scout` | Codebase Scout | Read-only workspace exploration, dependency mapping, and architecture analysis | 24 | readOnly, filesystem-read |
| `researcher` | Research Agent | Evidence gathering and synthesis across many sources, including broad repository or document investigation when no write is required | — | readOnly, autoRoutable; browser/fs-read/network |
| `cowork` | Cowork Team | Team orchestration decomposing one run across planner/coder/reviewer/researcher roles — reserve for large ambiguous multi-step work | — | artifactWrite, full execution capabilities |
| `sequential-thinking` | Sequential Thinking | Linear multi-step reasoning with explicit intermediate conclusions | 8 | — |
| `tree-of-thought` | Tree Of Thought | Compare multiple candidate reasoning branches before converging | 9 | — |
| `deep-thinking` | Deep Thinking | Deliberate reasoning with explicit verification of critical assumptions | 12 | — |
| `deep-web-research` | Deep Web Research | Verification-heavy research workflow with broader search and stronger source checking | 10 | readOnly; browser/fs-read/network |
| `persona-panel` | Persona Panel | Multiple personas answer the same prompt; a moderator pass weaves the replies together | 1 | explicit-roster only (auto-router excluded) |
| `browser-agent` | Browser Agent | Browser navigation, extraction, and page inspection | 8 | browser/network |
| `computer-use` | Computer Use Agent | Windows desktop GUI automation (observe, hover, click, drag, type, hotkey, scroll, wait) | 32 | Windows only for auto routing |
| `editor-agent` | Editor Agent | Precise source editing with file and patch tools | 10 | artifactWrite; fs-read/fs-write |
| `terminal-agent` | Terminal Agent | Command-driven workflows with tighter shell discipline | 9 | no durable run contract; process/terminal/fs-read/network |

`GENERAL_EXECUTION_CAPABILITIES` = `process`, `service`, `terminal`, `browser`,
`filesystem-read`, `filesystem-write`, `network`, `application-state`.
`RESEARCH_EXECUTION_CAPABILITIES` = `browser`, `filesystem-read`, `network`.

## Enhanced orchestration graph

`enhanced` (built by `buildEnhancedGraph` in `graph/builder.ts`) is the
general-purpose orchestrator:

```
(auto_decompose) ──► triage ──► capability_scout ──► specialist_router ──► specialist ──► reporter ──► __end__
                                       │
                                       └─ (tool calls) capability_scout_tools ──► specialist_router
```

- An optional `auto_decompose` stage (enabled with `SEPILOTD_AUTO_DECOMPOSE=1` and long input) fans out to `subagent.dispatch` calls before triage: `auto_decompose → decompose_tools → triage`.
- `capability_scout` may run tools to scout capabilities before routing.
- `specialist_router` picks one of six specialist subgraphs based on `state.specialistRoute`:

```
simple ─► simple_subgraph         (taskType: 'simple')
generalist / default ─► generalist_subgraph   (taskType: 'complex')
creative ─► creative_subgraph     (taskType: 'creative')
reviewer ─► reviewer_subgraph     (taskType: 'complex')
coder ─► coder_subgraph           (taskType: 'code')
researcher ─► researcher_subgraph (taskType: 'complex')
```

Each specialist is a nested `AgentGraph` attached via `agentSubgraphNode` with
explicit child-state mapping (`createChildState` clones the parent state,
resets iteration counters, and deep-copies the seed contract;
`mapSpecialistStateOut` merges messages, memories, usage, plans, and the
contract back into the parent).

### Specialist inner loop

Simple, generalist, and creative specialists share
`buildIterativeSpecialistGraph` (with a specialist-specific system prompt):

```
memory_retriever
      │
      ▼
   planner
      │
      ▼
iteration_guard ──shouldStop──► reporter ──► __end__
      │
      ▼
context_manager
      │
      ▼
    agent ──no tool calls──► reporter
      │
   tool calls
      ▼
    tools
      │
      ▼
 reflection ──output complete──► reporter
      │
   no output / plan steps remain
      │
      ▼
iteration_guard   (loop)
```

All three exits land on `reporter`:

- `iteration_guard` when `state.shouldStop`
- `agent` when the turn produced no tool calls
- `reflection` when output exists and every plan step is consumed

## Coder graph

`coder` (built by `buildCoderGraph` in `graph/presets/coder.ts`,
`maxIterations: 36`) is the largest preset. It starts at `memory_retriever`
and its `execution_intent_router` reads `state.seedContract.executionIntent`,
logging one of three entry routes:

```
memory_retriever ──► execution_intent_router ──► { operational_subgraph | current_document_inventory | codebase_exploration }
```

1. **`operational_subgraph`** — when `workspaceMutation === 'forbidden'` and the intent kind is `operational-action` or `inspection`. Built by `buildFocusedLoopGraph` with:
   - tool allowlist: `process.*`, `service.*`, `todowrite`, `terminal.run`, `browser.*`, `webfetch`, `fs.read`, `fs.list`, `fs.glob`, `fs.search`, `git.status`, `git.diff`, `git.log`, `system.info`
   - an initial tool call derived from the contract: `requestedProcessStart` becomes `process.start`, otherwise `requestedTerminalCommand` becomes `terminal.run`
   - `completeAfterInitialToolResult: true` — one successful structured observation plus a summary is already the durable evidence boundary, so the loop does not force prose criterion verdicts
2. **`current_document_inventory`** — document-phase turns (`inputLimitsCurrentTurnToDocumentArtifact`) skip the code pipeline: the graph runs an `fs.list` inventory of the current document scope and the turn finalizes once the requested artifact is written.
3. **`codebase_exploration`** — the default path. Results are captured (`capture_codebase_exploration`); large repositories add a `large_codebase_scout` fan-out (`large_codebase_scout_tools` → `capture_large_codebase_scout`).

### Implementation → validation → review pipeline

After exploration the graph moves through phases. The routing functions
(`routeImplementationAgentResult`, `routeImplementationGuardResult`,
`routeValidationAgentResult`, `routePostEditAnalysisResult`,
`routeCodingFinalizerResult`) show the shape:

- implementation with guards: `implementation_guard`, `implementation_file_edit_guard`, `capture_implementation`, `implement`, `finalizer`
- validation: `validation_tools`, `capture_validation`, `validation_completion_guard`
- post-edit analysis routes to `mark_validation_phase` by default, or `mark_finalize_phase` when the run stopped without semantic completion, when the turn is a document phase, or when `SEPILOTD_CODER_SKIP_VALIDATION_REVIEW=1` (ablation toggle) is set
- the finalizer routes back to `implementation_guard` when `completionDiagnostics.gate.decision === 'block'`, otherwise `__end__`

Review is an embedded evidence-only reviewer subgraph (`buildReviewerGraph`
with `preserveProtocolOutput`). It may inspect source, diffs, diagnostics, and
retained validation results, but must not rerun tests/builds/formatters,
mutate files, or manage processes. Its summary must end with exactly one line:

```
VERIFIED: <evidence>
UNVERIFIED: <one-line blocker>
```

The orchestrator reads only this stem; `UNVERIFIED` routes back to validation.
Edits are wrapped in approval checkpoints via `openEditCheckpointNode`
(`presets/edit-checkpoint-node.ts`).

## Reviewer graph

`reviewer` registers with `enableDebate: true` — the debate quality gate lives
in `graph/debate.ts`. The graph is read-only (`filesystem-read`) and is also
embedded inside the coder pipeline as the evidence-only reviewer above.

## Reusable preset building blocks

Besides the 16 registered graphs, `graph/presets/` ships composable pieces:

- `focused-loop.ts` — bounded tool loop (used by the coder operational path)
- `edit-checkpoint-node.ts` — approval checkpoint around edits
- `tool-boundary.ts`, `prerequisite.ts` — shared node helpers
- `workspace-agents.ts` — builds `coder` (`buildWorkspaceCoderGraph`) and `codebase-scout` (`buildCodebaseScoutGraph`)

## Where to look

```
packages/daemon/src/agent/
├── mode-catalog.ts        # engine modes, auto-routability, fallback rules
├── intent-router.ts
├── mode-router.ts
├── coordinator.ts
├── engine.ts
└── graph/
    ├── engine.ts          # AgentGraph runtime: nodes, edges, checkpoints, state board
    ├── nodes.ts           # node library
    ├── builder.ts         # enhanced orchestrator + specialist graphs
    ├── registry.ts        # GraphAgentRegistry + structural validation
    ├── types.ts           # AgentState, GraphRuntimeState, node/edge definitions
    └── presets/           # 16 builtin graphs + reusable building blocks
```

Custom graphs use the same `GraphAgentInfo` shape — see `graph/yaml-loader.ts`
(`source: 'yaml'`), `plugins/loader.ts` (`source: 'plugin'`), and
`agent/user-agents/loader.ts` (`source: 'user'`).