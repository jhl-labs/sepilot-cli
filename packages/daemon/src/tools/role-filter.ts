import { ToolRegistry } from './registry.js'
import type { ToolDefinitionRuntime } from './registry.js'
import {
  explicitCanonicalToolNames,
  inputPositiveCapabilityScope,
} from '../agent/task-contract.js'
import { resolveClosedExactToolNames } from '../agent/tool-call-budget.js'
import { isVisibleUrlOpenRequest } from '../agent/desktop-control-intent.js'

type DeviceRole = 'desktop' | 'server' | 'edge'

/**
 * Memory toolset exposed to *direct-API* surfaces (CLI `sepilot ask`/TUI,
 * web, desktop). The full ~40-tool set (pin/diff/audit/documents/reminders/
 * tag-rename/maintenance/...) is the channel / auto-memory ("OpenClaw")
 * concept and only bloats the LLM tool list for one-shot CLI work, so direct
 * API defaults to `lean`. The channel pipeline keeps the full registry — this
 * filter is never applied there.
 *
 * - `lean`: core memory recall/write tools plus read-only memory graph lookup
 * - `full`: every registered `memory.*` tool (same as channels)
 * - `none`: no `memory.*` tools at all
 */
export type DirectApiMemoryToolset = 'lean' | 'full' | 'none'

export const TOOL_EXPOSURE_GROUPS = [
  'files',
  'code',
  'web',
  'browser',
  'process',
  'apps',
  'office',
  'computer',
  'documents',
  'media',
  'image',
  'services',
  'memory-advanced',
  'scheduler-advanced',
  'swarm',
  'delegation',
  'integrations',
] as const

export type ToolExposureGroup = (typeof TOOL_EXPOSURE_GROUPS)[number]

export const TOOL_EXPOSURE_GROUP_DESCRIPTIONS: Record<ToolExposureGroup, string> = {
  files: 'Read, search, and modify workspace files or run terminal commands.',
  code: 'Inspect source code, Git history, diagnostics, notebooks, and language servers.',
  web: 'Search or fetch current information from the web.',
  browser: 'Navigate and interact with web pages in a browser.',
  process: 'Start, inspect, wait for, or stop operating-system processes.',
  apps: 'Read or update personal calendars, todos, notes, boards, briefings, and custom Desktop app data.',
  office: 'Inspect or edit live Office documents and presentations.',
  computer: 'Control desktop windows, mouse, keyboard, and screenshots.',
  documents: 'Edit an active writing document with doc.* tools.',
  media: 'Inspect, transcribe, extract, or synthesize audio and media.',
  image: 'Generate or manage generated images.',
  services: 'Install, start, stop, or inspect long-running services.',
  'memory-advanced': 'Audit, graph, document, tag, merge, import, or maintain memory.',
  'scheduler-advanced': 'Inspect runs, pause, resume, or manually trigger scheduled jobs.',
  swarm: 'Coordinate agents inside an active swarm run.',
  delegation: 'Delegate work to subagents, devices, ACP, or A2A agents.',
  integrations: 'Use MCP or plugin-provided external integrations.',
}

function registerPreservingSource(
  target: ToolRegistry,
  source: ToolRegistry,
  tool: ToolDefinitionRuntime,
): void {
  const registrationSource = source.registrationSource(tool.name)
  target.register(tool, registrationSource ? { source: registrationSource } : undefined)
}

const LEAN_MEMORY_TOOLS = new Set<string>([
  'memory.remember',
  'memory.update',
  'memory.list',
  'memory.search',
  'memory.graph.search',
  'memory.graph.neighbors',
  'memory.graph.page',
  'memory.forget',
])

export function isLeanMemoryTool(name: string): boolean {
  return LEAN_MEMORY_TOOLS.has(name)
}

const DEFAULT_PERSONAL_TOOL_NAMES = new Set([
  'question',
  'skill',
  'todowrite',
  'assistant.status',
  'self.info',
  'system.info',
  'usage.report',
  'schedule_create',
  'schedule_list',
  'schedule_update',
  'schedule_cancel',
  'schedule_get',
  'schedule_runs',
  'schedule_pause',
  'schedule_resume',
  'schedule_run_now',
  'monitor.evaluate',
  'monitor.report',
  // Personal data is a shared capability, independent of reasoning mode or
  // intent-router availability. Exact caller/skill/persona ceilings still win.
  'knowledge.search',
  'knowledge.save',
  'knowledge.read',
  'knowledge.edit',
  'knowledge.review',
  'apps.list',
  'apps.read',
  'apps.search',
  'apps.mutate',
  'apps.write',
  ...LEAN_MEMORY_TOOLS,
])

// The terminal surface starts with its general development/command toolbox.
// Specialist schemas remain discoverable through agent.tools / agent.transfer;
// this is a visibility projection, never an authority restriction.
const CLI_REACT_GROUPS: readonly ToolExposureGroup[] = ['files', 'code', 'process', 'web']
const CLI_REACT_TOOL_NAMES = new Set([
  'question', 'skill', 'todowrite', 'system.info',
  'memory.search', 'memory.list', 'knowledge.search', 'knowledge.read',
])

export function toolExposureGroupForTool(name: string): ToolExposureGroup | null {
  if (
    name === 'terminal.run'
    || name === 'apply_patch'
    || name === 'workspace.prepare'
    || name.startsWith('fs.')
  ) return 'files'
  if (
    name.startsWith('git.')
    || name.startsWith('code.')
    || name === 'lsp'
    || name.startsWith('notebook.')
  ) return 'code'
  if (name === 'web.search' || name === 'webfetch' || name === 'market.quote') return 'web'
  if (name.startsWith('browser.')) return 'browser'
  if (name.startsWith('process.')) return 'process'
  if (name.startsWith('apps.')) return 'apps'
  if (name.startsWith('office.')) return 'office'
  if (name.startsWith('computer.')) return 'computer'
  if (name.startsWith('doc.')) return 'documents'
  if (name.startsWith('media.')) return 'media'
  if (name.startsWith('image_gen.')) return 'image'
  if (name.startsWith('service.')) return 'services'
  if (name.startsWith('memory.')) return 'memory-advanced'
  if (name.startsWith('schedule_')) return 'scheduler-advanced'
  if (name.startsWith('monitor.')) return 'scheduler-advanced'
  if (name.startsWith('swarm.')) return 'swarm'
  if (
    name === 'subagent.dispatch'
    || name === 'device.delegate'
    || name === 'external_acp.run'
    || name.startsWith('a2a.')
  ) return 'delegation'
  if (
    name.startsWith('mcp.')
    || name.startsWith('plugin.')
    || name.startsWith('gitea.')
    || name.startsWith('jpad.')
    || name.startsWith('notification.')
  ) return 'integrations'
  return null
}

/**
 * Preserve observe-only integration tools when the user names their canonical
 * namespace in ordinary prose (for example "Gitea Actions" or "JPAD pages").
 * A concrete caller mode controls the execution loop, not which requested
 * read capability exists; without this projection an explicit `react` mode
 * can hide the only canonical reader and push the model toward filesystem or
 * terminal guesses.
 *
 * Keep the fallback deliberately narrow:
 * - match registry-derived namespaces rather than prompt-specific phrases;
 * - inspect only the affirmative request scope, so explicit exclusions win;
 * - expose only daemon-attested `observe` tools. Mutation tools still require
 *   intent routing, a selected skill, or an explicit canonical tool name.
 */
function explicitlyNamedObserveIntegrationTools(
  source: ToolRegistry,
  requestInput: string,
): ReadonlySet<string> {
  const requestTokens = new Set(
    [...inputPositiveCapabilityScope(requestInput).matchAll(/[a-z][a-z0-9_-]*/giu)]
      .map((match) => match[0].toLowerCase()),
  )
  if (requestTokens.size === 0) return new Set()

  const names = new Set<string>()
  for (const tool of source.list()) {
    if (toolExposureGroupForTool(tool.name) !== 'integrations') continue
    const namespace = tool.name.split('.', 1)[0]?.toLowerCase()
    if (!namespace || !requestTokens.has(namespace)) continue
    if (source.securityDescriptor(tool.name).effect !== 'observe') continue
    names.add(tool.name)
  }
  return names
}

const MODE_EXPOSURE_GROUPS: Record<string, readonly ToolExposureGroup[]> = {
  // This is the capability ceiling passed into the coder graph, not the tool
  // list shown on every coder-model call. The graph applies its own
  // node/phase profiles afterwards. Preserve every family that a coder
  // subgraph may activate; otherwise a later grounded contract cannot restore
  // a browser or service tool removed by this earlier routing projection.
  coder: ['files', 'code', 'web', 'browser', 'process', 'services'],
  researcher: ['web', 'browser'],
  'deep-web-research': ['web', 'browser'],
  'computer-use': ['computer', 'office', 'media'],
  writing: ['documents'],
}

const VISIBLE_OPEN_URL_TOOL_NAME = 'computer.open_url'

export interface ContextualToolExposureOptions {
  surface?: string | null
  /** The model owns natural-language selection; projection uses only structured metadata. */
  semanticRouting?: boolean
  selectedGroups?: readonly ToolExposureGroup[]
  routedMode?: string
  routingFallback?: boolean
  activeRemoteBrowser?: boolean
  activeWritingDocument?: boolean
  swarmSession?: boolean
  declaredSkillToolNames?: ReadonlySet<string>
  /**
   * Tools needed by an automatically inferred skill. Unlike an explicitly
   * selected skill allowlist, these tools augment the contextual surface and
   * must never remove capabilities selected by the intent router.
   */
  supplementalToolNames?: ReadonlySet<string>
  /**
   * Complete current-turn input used to preserve explicitly named registered
   * tools when contextual routing defers their specialist exposure group.
   * This affects visibility only; policy, autonomy, approval, persona, and
   * exact caller/skill allowlists remain the authority boundary.
   */
  requestInput?: string
  explicitToolNames?: readonly string[]
  personaAllowedTools?: readonly string[]
  personaDeniedTools?: readonly string[]
}

/**
 * Build the per-turn personal-agent surface. Registration/global enablement is
 * only the capability ceiling; this function decides what the model actually
 * sees for one request.
 */
export function withContextualToolExposure(
  source: ToolRegistry,
  options: ContextualToolExposureOptions,
): ToolRegistry {
  const denied = new Set(options.personaDeniedTools ?? [])
  const declaredExactAllowlist = options.explicitToolNames
    ?? (options.declaredSkillToolNames && options.declaredSkillToolNames.size > 0
      ? [...options.declaredSkillToolNames]
      : options.personaAllowedTools)
  const structuralClosedAllowlist = !options.semanticRouting && options.requestInput
    ? resolveClosedExactToolNames(options.requestInput, source.list())
    : null
  const exactAllowlist = structuralClosedAllowlist
    ? declaredExactAllowlist === undefined
      ? structuralClosedAllowlist
      : structuralClosedAllowlist.filter((name) => declaredExactAllowlist.includes(name))
    : declaredExactAllowlist

  if (exactAllowlist !== undefined) {
    return withOptionalToolNameAllowlist(
      source,
      exactAllowlist.filter((name) => !denied.has(name)),
    )
  }

  const cliReact = options.surface === 'cli' && options.semanticRouting === true && options.routedMode === 'react'
  const groups = new Set<ToolExposureGroup>(options.selectedGroups ?? [])
  if (cliReact) for (const group of CLI_REACT_GROUPS) groups.add(group)
  if (options.routedMode) {
    for (const group of MODE_EXPOSURE_GROUPS[options.routedMode] ?? []) groups.add(group)
  }
  if (options.routingFallback && !cliReact) {
    // The intent router is an advisory first pass, not the final graph/capability
    // decision.  In particular, a local or otherwise slower provider can time
    // out here and the graph router can still correctly select coder/cowork
    // from the structured request.  Do not irreversibly remove capabilities
    // that those graphs may activate later: graph/node profiles will narrow
    // this ceiling before each model request.  Keep this to the general coding
    // families rather than restoring the 100+ tool catalog (Office,
    // media, integrations, and other specialist surfaces remain deferred).
    for (const group of MODE_EXPOSURE_GROUPS.coder ?? []) groups.add(group)
  }
  if (options.activeWritingDocument) groups.add('documents')
  if (options.swarmSession) groups.add('swarm')

  const allowed = new Set(cliReact ? CLI_REACT_TOOL_NAMES : DEFAULT_PERSONAL_TOOL_NAMES)
  for (const tool of source.list()) {
    const group = toolExposureGroupForTool(tool.name)
    // Semantic mode control discovers ungrouped integrations through the
    // authorized catalog and activates exact names on demand. Legacy callers
    // without that discovery channel still expose ungrouped tools directly.
    if ((!group && !options.semanticRouting) || (group && groups.has(group))) allowed.add(tool.name)
  }
  // A session-bound tab is an explicit user-selected capability, even when
  // the advisory intent router omits the browser group. Exact caller/skill
  // allowlists above and persona denials below remain authoritative.
  if (options.activeRemoteBrowser) {
    allowed.add('browser.remote_snapshot')
    allowed.add('browser.remote_action')
  }
  for (const name of options.supplementalToolNames ?? []) allowed.add(name)
  if (!options.semanticRouting) for (const name of explicitCanonicalToolNames(
    options.requestInput ?? '',
    source.list().map((tool) => tool.name),
  )) {
    allowed.add(name)
  }
  if (!options.semanticRouting) for (const name of explicitlyNamedObserveIntegrationTools(
    source,
    options.requestInput ?? '',
  )) {
    allowed.add(name)
  }
  // "Open/show this page for me" states an outcome outright, so the advisory
  // intent router must not be able to defer the one tool that satisfies it.
  // Without this, guessing the `browser` group drops computer.open_url, which
  // makes the visible-open route unreachable and so confirms the guess —
  // leaving headless inspection as the only way to answer a request that asked
  // for a visible browser. Visibility only; the ask policy still gates it.
  if (!options.semanticRouting && options.requestInput && isVisibleUrlOpenRequest(options.requestInput)) {
    if (source.get(VISIBLE_OPEN_URL_TOOL_NAME)) allowed.add(VISIBLE_OPEN_URL_TOOL_NAME)
  }
  for (const name of denied) allowed.delete(name)
  return withToolNameAllowlist(source, allowed)
}

/** Preserve the authority ceiling independently of temporary mode visibility. */
export function withAuthorizedToolExposure(
  source: ToolRegistry,
  options: Pick<ContextualToolExposureOptions, 'explicitToolNames' | 'declaredSkillToolNames' | 'personaAllowedTools' | 'personaDeniedTools'>,
): ToolRegistry {
  const ceilings = [options.explicitToolNames, options.personaAllowedTools,
    options.declaredSkillToolNames?.size ? [...options.declaredSkillToolNames] : undefined,
  ].filter((names): names is readonly string[] => names !== undefined)
  const denied = new Set(options.personaDeniedTools ?? [])
  return withToolNameAllowlist(source, new Set(source.list()
    .filter((tool) => !denied.has(tool.name) && ceilings.every((names) => names.includes(tool.name)))
    .map((tool) => tool.name)))
}

export function isDefaultPersonalTool(name: string): boolean {
  return DEFAULT_PERSONAL_TOOL_NAMES.has(name)
}

export function withDirectApiMemoryToolset(
  source: ToolRegistry,
  toolset: DirectApiMemoryToolset,
): ToolRegistry {
  if (toolset === 'full') return source
  const filtered = new ToolRegistry()
  for (const tool of source.list()) {
    if (tool.name.startsWith('memory.')) {
      if (toolset === 'none') continue
      if (!LEAN_MEMORY_TOOLS.has(tool.name)) continue
    }
    registerPreservingSource(filtered, source, tool)
  }
  return filtered
}

/**
 * Every swarm.* tool starts by resolving the swarm run for its session and
 * fails with "not in a swarm run" when there is none — so outside a swarm
 * session the whole family is uncallable. Their schemas were still sent on
 * every ordinary chat call (11 tools, ~3,500 characters), inviting the model
 * to reach for something that can only fail. The supervisor prompt that
 * explains them is already gated on the same session-id prefix.
 */
export function withSwarmToolsForSession(
  source: ToolRegistry,
  sessionId: string | undefined,
): ToolRegistry {
  if (sessionId?.startsWith('swarm_')) return source
  const filtered = new ToolRegistry()
  for (const tool of source.list()) {
    if (tool.name.startsWith('swarm.')) continue
    registerPreservingSource(filtered, source, tool)
  }
  return filtered
}

/**
 * The doc.* family edits the active writing document and nothing else — none
 * of the eight tools can open or create one, so with no active document every
 * call returns "활성 글쓰기 문서가 없습니다". Their schemas rode on every
 * ordinary chat call anyway (8 tools, ~2,000 characters). Callers pass the
 * same value the tools resolve (`writingDocId ?? registry.getActiveId()`), so
 * the gate cannot disagree with the tools about what is active.
 */
export function withWritingDocTools(
  source: ToolRegistry,
  activeDocId: string | null | undefined,
): ToolRegistry {
  if (activeDocId) return source
  const filtered = new ToolRegistry()
  for (const tool of source.list()) {
    if (tool.name.startsWith('doc.')) continue
    registerPreservingSource(filtered, source, tool)
  }
  return filtered
}

export function withToolNameAllowlist(
  source: ToolRegistry,
  allowedToolNames: ReadonlySet<string>,
): ToolRegistry {
  const filtered = new ToolRegistry()
  for (const tool of source.list()) {
    if (allowedToolNames.has(tool.name)) {
      registerPreservingSource(filtered, source, tool)
    }
  }
  return filtered
}

export function withOptionalToolNameAllowlist(
  source: ToolRegistry,
  allowedToolNames?: readonly string[],
): ToolRegistry {
  if (allowedToolNames === undefined) return source
  const allowed = new Set(allowedToolNames)
  const filtered = new ToolRegistry()
  for (const tool of source.list()) {
    if (allowed.has(tool.name)) {
      registerPreservingSource(filtered, source, tool)
    }
  }
  return filtered
}

/** Tools available per device role */
const ROLE_TOOLS: Record<DeviceRole, string[] | '*'> = {
  desktop: '*',  // All tools
  server: ['terminal.run', 'fs.read', 'fs.list', 'fs.write', 'fs.append', 'todowrite', 'assistant.status', 'gitea.*', 'web.search', 'webfetch', 'market.quote', 'monitor.*', 'browser.navigate', 'browser.extract', 'browser.screenshot', 'browser.click', 'browser.evaluate', 'device.delegate', 'pages.*', 'mcp.*'],
  edge: ['terminal.run', 'fs.read', 'fs.list', 'fs.write', 'fs.append', 'todowrite'],  // Minimal set for resource-constrained devices
}

/** Tools explicitly blocked per role */
const ROLE_BLOCKED: Record<DeviceRole, string[]> = {
  desktop: [],
  server: [],
  edge: ['browser.navigate', 'browser.screenshot', 'browser.extract', 'browser.click', 'browser.evaluate', 'device.delegate'],
}

export function filterToolsByRole(registry: ToolRegistry, role: DeviceRole): ToolDefinitionRuntime[] {
  const allowed = ROLE_TOOLS[role]
  const blocked = new Set(ROLE_BLOCKED[role])

  return registry.list().filter(tool => {
    if (blocked.has(tool.name)) return false
    if (allowed === '*') return true
    return allowed.some(pattern => {
      if (pattern.endsWith('.*')) {
        return tool.name.startsWith(pattern.slice(0, -1))
      }
      return tool.name === pattern
    })
  })
}

export function applyRoleFilter(_registry: ToolRegistry, _role: DeviceRole): void {
  // Note: actual filtering happens at engine level, not registry level
}
