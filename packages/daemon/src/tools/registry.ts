import type {
  AgentEvent,
  AgentRunContract,
  AutonomyLevel,
  ContentPart,
  ToolDefinition,
  ToolExecutionPosture,
  ToolResultMetadata,
  ToolSecurityDescriptor,
} from '@sepilotd/core'
import { resolveToolSecurityDescriptor, UNKNOWN_TOOL_SECURITY } from './security.js'

/**
 * Effective parent-turn security posture inherited by tools that start a
 * nested agent. This is execution-scoped on purpose: consulting daemon-wide
 * defaults from the nested run can silently widen a Desktop turn's authority.
 */
export interface DelegatedAgentExecutionPolicy {
  autonomy: AutonomyLevel
  requireToolApproval: boolean
  /** Concrete registry visible to the parent engine for this turn. */
  allowedToolNames: readonly string[]
}

export type ToolResumeSafety = 'replay-safe' | 'replay-risky'
export type ToolSchedulingMode = 'sequential' | 'parallel-safe'

export interface ToolSchedulingHint {
  mode: ToolSchedulingMode
  resource?: string
  key?: (input: Record<string, unknown>) => string | null
}

export interface ResolvedToolSchedulingHint {
  mode: ToolSchedulingMode
  resource: string | null
  key: string | null
}

/**
 * Calls emitted in one model response cannot react to an earlier sibling's
 * result. A tool may use this contract when some calls are target fan-out of
 * one shared input template: after a deterministic template failure, the
 * executor skips only still-unexecuted siblings with the same key and lets the
 * model repair the template first. Target-specific and transient failures must
 * return false from `blocksSiblings`.
 */
export interface ToolBatchFailureCoupling {
  groupKey(input: Record<string, unknown>): string | null
  blocksSiblings(result: ToolResult): boolean
}

export interface EditCheckpointHandle {
  recordPreEdit(path: string): Promise<void>
}

export interface WorkspaceMutationHandle {
  recordRead(path: string): Promise<void>
  recordWrite(path: string): Promise<void>
  detectStale(path: string): Promise<{ path: string; description: string } | null>
  lookupReadObservation?(
    path: string,
    viewKey: string,
  ): Promise<import('../agent/workspace-mutation/tracker.js').WorkspaceReadObservationLookup>
  recordReadObservation?(
    path: string,
    viewKey: string,
    output: string,
  ): Promise<import('../agent/workspace-mutation/tracker.js').WorkspaceReadObservation | null>
}

export interface ToolExecutionContext {
  executionId: string
  sessionId: string
  startedAt: string
  cwd?: string
  /** Strict filesystem capability root inherited from the chat turn. */
  workspaceRoot?: string
  writingDocId?: string
  signal?: AbortSignal
  editCheckpoint?: EditCheckpointHandle
  workspaceMutation?: WorkspaceMutationHandle
  /**
   * Canonical scope tags for the active session/sender (e.g. "scope:user:42",
   * "scope:channel:telegram:99"). Memory tools use this to:
   *  - tag newly-saved memories so multi-user setups don't leak,
   *  - default-filter search/audit results to entries the caller can see,
   *  - reject mutations on memories owned by another scope.
   * Empty/undefined means "no scope" — the tools fall back to global behavior.
   */
  scopeTags?: string[]
  /**
   * When the agent is invoked from a channel pipeline (telegram/cli/web/...),
   * tools can use this to capture the originating channel for follow-up
   * actions (e.g. scheduling a future reply back to the same chat).
   * Undefined when invoked from REST or internal contexts.
   */
  channelContext?: {
    channel: string
    chatKey: string
    triggerMessageId?: string
  }
  /**
   * Optional sink for nested agent events a tool wants surfaced on the
   * parent run's stream — currently subagent.dispatch forwarding its
   * subagent's progress as `subagent_progress`. The agent loop buffers
   * these during tool execution and yields them before the tool result.
   * Undefined for tools/contexts that don't produce nested events.
   */
  emitEvent?: (event: AgentEvent) => void
  /** Active parent run contract, forwarded to tools that spawn nested agent work. */
  runContract?: AgentRunContract
  /** Parent-turn authority for tools that spawn nested agent work. */
  delegatedAgentPolicy?: DelegatedAgentExecutionPolicy
}

export interface ToolInputNormalizationContext {
  cwd?: string
  workspaceRoot?: string
}

export interface ToolInputValidationContext {
  cwd?: string
  workspaceRoot?: string
}

/**
 * Declares when one successful read in the active turn contains every fact a
 * later call would return. The comparison must be conservative and pure: a
 * false result merely permits another execution, while a false positive can
 * hide required evidence.
 */
export interface ToolObservationCoverage {
  covers(
    observedInput: Record<string, unknown>,
    requestedInput: Record<string, unknown>,
    context: ToolInputNormalizationContext,
  ): boolean
  /**
   * Optional union coverage for paged observations. This is useful when no
   * single successful read covers the request, but adjacent pages in the same
   * current turn do. Implementations must remain conservative and pure.
   */
  coversCollectively?(
    observedInputs: readonly Record<string, unknown>[],
    requestedInput: Record<string, unknown>,
    context: ToolInputNormalizationContext,
  ): boolean
  /**
   * Optional conservative subtraction for partially overlapping observations.
   * Return only the input slices whose evidence is still missing. Return
   * undefined when the request cannot be narrowed safely. An empty array means
   * the observations collectively cover the request and is equivalent to
   * coversCollectively=true.
   */
  uncoveredInputs?(
    observedInputs: readonly Record<string, unknown>[],
    requestedInput: Record<string, unknown>,
    context: ToolInputNormalizationContext,
  ): readonly Record<string, unknown>[] | undefined
}

export interface ToolRecoveryProbeContext {
  executionId: string
  startedAt: string
  sessionId?: string
  cwd?: string
  workspaceRoot?: string
}

export interface ToolDefinitionRuntime {
  name: string
  description: string
  inputSchema: Record<string, unknown>
  /**
   * Bounded, credential-free resource metadata for semantic discovery. Called
   * only on an authorized, available tool, once per turn. Must not read record
   * contents, execute actions, or expand tool authority. Not a tool receipt.
   */
  discoveryContext?: (input: string) => Promise<string>
  /** Runtime availability for the current execution boundary; not permission. */
  unavailableReason?: (context: Pick<ToolExecutionContext, 'cwd' | 'workspaceRoot'>) => string | undefined
  /**
   * Stable public identifiers that a user may name when bounding this
   * capability's call count without knowing the daemon's canonical tool name.
   *
   * Examples are a configured provider id or endpoint hostname. Keep these
   * structural and credential-free; free-form natural-language synonyms do
   * not belong here. A callback lets runtime configuration changes take
   * effect without rebuilding the registry.
   */
  cardinalityAliases?: readonly string[] | (() => readonly string[])
  /**
   * Absolute HTTP(S) GET URL templates owned by this integration tool.
   *
   * A template may use a whole path segment such as `{pageId}`. When a model
   * selects a generic URL transport for one of these routes, the agent can
   * preserve the integration's runtime authentication, API version, timeout,
   * and response contract by asking for this canonical tool instead. Keep
   * templates structural and credential-free. Explicit user transport choices
   * still take precedence over this advisory routing metadata.
   */
  canonicalReadUrlTemplates?: readonly string[] | (() => readonly string[])
  /**
   * Declares that this observe-only tool reads one already-identified source
   * directly, so a research verification phase may retain it alongside
   * generic source readers. Discovery/listing tools should not opt in: the
   * distinction keeps verification on its fixed evidence packet while
   * preserving authenticated integration readers for named sources.
   */
  researchVerification?: 'direct-source'
  /** Canonical effect metadata consumed by autonomy, plan and approval gates. */
  security?: ToolSecurityDescriptor
  resumeSafety?: ToolResumeSafety
  resumeSafetyForInput?(input: Record<string, unknown>): ToolResumeSafety
  scheduling?: ToolSchedulingHint
  batchFailureCoupling?: ToolBatchFailureCoupling
  observationCoverage?: ToolObservationCoverage
  normalizeInput?(
    input: Record<string, unknown>,
    context: ToolInputNormalizationContext,
  ): Promise<Record<string, unknown>> | Record<string, unknown>
  validateInput?(
    input: Record<string, unknown>,
    context: ToolInputValidationContext,
  ): Promise<ToolResult | null> | ToolResult | null
  recoverInterruptedExecution?(
    input: Record<string, unknown>,
    context: ToolRecoveryProbeContext,
  ): Promise<ToolResult | null>
  execute(
    input: Record<string, unknown>,
    context?: ToolExecutionContext,
  ): Promise<ToolResult>
}

/**
 * Image attachment a tool can return so the agent loop forwards it as a
 * `ContentPart` in the next chat round. Used by `computer.observe` so a
 * vision-capable LLM can actually look at the screenshot it just took
 * — without this the model only sees the file path string and is blind.
 */
export interface ToolResultImage {
  /** MIME type, e.g. `image/png`. */
  mediaType: string
  /** Raw base64 payload (no data URL prefix). */
  data: string
  /**
   * Also expose this image to the live interactive surface. This is opt-in:
   * model-only screenshots remain private to the agent turn by default.
   */
  displayToClient?: boolean
}

export interface ToolResult {
  output: string
  status: 'success' | 'error'
  /**
   * Optional machine-readable metadata about the execution. This is not
   * appended to the model-facing text output; consumers that need audit or
   * posture details can read it from emitted events, hooks, and execution
   * journal records.
   */
  metadata?: ToolResultMetadata
  /**
   * Optional multimodal content that should be made available to the
   * next model turn in addition to the text output. Large/binary data is
   * intentionally kept out of `output` so logs and tool result events
   * remain readable.
   */
  contentParts?: ContentPart[]
  executionPosture?: ToolExecutionPosture
  durationMs: number
  /**
   * Optional structured error code on `status === 'error'`. Conventions:
   * - `*_PERMANENT` (e.g. `EACCES_PERMANENT`, `INVALID_URL_PERMANENT`,
   *   `POLICY_PERMANENT`) — same call will fail the same way; the agent
   *   must switch tools/arguments instead of retrying.
   * - `*_TRANSIENT` (e.g. `TIMEOUT_TRANSIENT`, `NETWORK_TRANSIENT`,
   *   `5XX_TRANSIENT`) — retry once with a longer timeout / backoff is
   *   reasonable.
   * - `*_USER` — the failure depends on a user decision/state change.
   * Tools that do not categorise their errors leave this undefined; the
   * agent then falls back to its generic failure-recovery rules.
   */
  code?: string
  /**
   * Image attachments to forward into the next chat round. The agent loop
   * embeds these as `ContentPart[]` on the resulting `tool` message so a
   * vision-capable model can see them. Tools with no visual output (the
   * common case) leave this undefined.
   */
  images?: ToolResultImage[]
}

export function resolveToolResumeSafety(
  tool: ToolDefinitionRuntime | undefined,
  input: Record<string, unknown>,
): ToolResumeSafety {
  return (
    tool?.resumeSafetyForInput?.(input)
    ?? tool?.resumeSafety
    ?? 'replay-risky'
  )
}

export function resolveToolSchedulingHint(
  tool: ToolDefinitionRuntime | undefined,
  input: Record<string, unknown>,
): ResolvedToolSchedulingHint {
  if (!tool?.scheduling || tool.scheduling.mode !== 'parallel-safe') {
    return {
      mode: 'sequential',
      resource: null,
      key: null,
    }
  }

  return {
    mode: 'parallel-safe',
    resource: tool.scheduling.resource ?? tool.name,
    key: tool.scheduling.key?.(input) ?? null,
  }
}

/**
 * Resolve a tool-owned failure-coupling group without letting a buggy plugin
 * break the executor. The tool name namespaces otherwise identical keys.
 */
export function resolveToolBatchFailureCouplingKey(
  tool: ToolDefinitionRuntime | undefined,
  input: Record<string, unknown>,
): string | null {
  if (!tool?.batchFailureCoupling) return null
  try {
    const key = tool.batchFailureCoupling.groupKey(input)?.trim()
    return key ? `${tool.name}:${key}` : null
  } catch {
    return null
  }
}

export function toolResultBlocksFailureCoupledSiblings(
  tool: ToolDefinitionRuntime | undefined,
  result: ToolResult,
): boolean {
  if (!tool?.batchFailureCoupling) return false
  try {
    return tool.batchFailureCoupling.blocksSiblings(result) === true
  } catch {
    return false
  }
}

export type ToolRegistrationSource = 'builtin' | 'plugin'

export class ToolRegistry {
  private tools = new Map<string, ToolDefinitionRuntime>()
  private sources = new Map<string, ToolRegistrationSource>()
  private disabledTools = new Set<string>()
  private generation = 0
  private cachedDefinitionsGeneration = -1
  private cachedDefinitions: ToolDefinition[] | undefined
  // Active plugin registration scope: while set, register() defaults to the
  // 'plugin' source and records the names it added so a partially-registered
  // plugin can be unwound on failure.
  private pluginScope: { pluginId: string; names: Set<string> } | null = null

  register(tool: ToolDefinitionRuntime, options?: { source?: ToolRegistrationSource }): void {
    const source = options?.source ?? (this.pluginScope ? 'plugin' : 'builtin')
    const existing = this.sources.get(tool.name)
    // A plugin may never shadow a built-in tool, nor silently drop/override a
    // tool already registered by another plugin (order-dependent shadowing).
    if (source === 'plugin' && existing) {
      throw new Error(
        `plugin cannot register tool '${tool.name}': already registered by ${existing}`,
      )
    }
    this.tools.set(tool.name, {
      ...tool,
      security: source === 'plugin' && !tool.security
        ? UNKNOWN_TOOL_SECURITY
        : resolveToolSecurityDescriptor(tool.name, tool.security),
    })
    this.sources.set(tool.name, source)
    if (source === 'plugin') this.pluginScope?.names.add(tool.name)
    this.generation += 1
  }

  /** Begin recording plugin registrations for possible unwind. */
  beginPluginScope(pluginId: string): void {
    this.pluginScope = { pluginId, names: new Set() }
  }

  /** End the scope and return the tool names the plugin registered. */
  endPluginScope(): string[] {
    const names = this.pluginScope ? [...this.pluginScope.names] : []
    this.pluginScope = null
    return names
  }

  get(name: string): ToolDefinitionRuntime | undefined {
    if (this.disabledTools.has(name)) return undefined
    return this.tools.get(name)
  }

  /** Return a registered tool regardless of its operator-controlled state. */
  getRegistered(name: string): ToolDefinitionRuntime | undefined {
    return this.tools.get(name)
  }

  isEnabled(name: string): boolean {
    return this.tools.has(name) && !this.disabledTools.has(name)
  }

  setDisabledTools(names: Iterable<string>): void {
    const next = new Set(Array.from(names, (name) => name.trim()).filter(Boolean))
    if (
      next.size === this.disabledTools.size
      && [...next].every((name) => this.disabledTools.has(name))
    ) {
      return
    }
    this.disabledTools = next
    this.generation += 1
  }

  registrationSource(name: string): ToolRegistrationSource | undefined {
    return this.sources.get(name)
  }

  securityDescriptor(name: string): ToolSecurityDescriptor {
    return resolveToolSecurityDescriptor(name, this.tools.get(name)?.security)
  }

  unregister(name: string): boolean {
    const removed = this.tools.delete(name)
    if (removed) {
      this.sources.delete(name)
      this.generation += 1
    }
    return removed
  }

  unregisterByPrefix(prefix: string): number {
    let removed = 0
    for (const name of this.tools.keys()) {
      if (!name.startsWith(prefix)) continue
      this.tools.delete(name)
      this.sources.delete(name)
      removed += 1
    }
    if (removed > 0) {
      this.generation += 1
    }
    return removed
  }

  list(context?: Pick<ToolExecutionContext, 'cwd' | 'workspaceRoot'>): ToolDefinitionRuntime[] {
    return Array.from(this.tools.values()).filter((tool) => !this.disabledTools.has(tool.name)
      && (!context || !tool.unavailableReason?.(context)))
  }

  /** Catalog view used by settings and diagnostics, including disabled tools. */
  listRegistered(): ToolDefinitionRuntime[] {
    return Array.from(this.tools.values())
  }

  /** Convert to LLM-friendly tool definitions */
  toToolDefinitions(context?: Pick<ToolExecutionContext, 'cwd' | 'workspaceRoot'>): ToolDefinition[] {
    if (context) {
      // Availability may change independently of registry generation.
      return this.toToolDefinitions().filter((tool) => !this.tools.get(tool.name)?.unavailableReason?.(context))
    }
    if (
      this.cachedDefinitions
      && this.cachedDefinitionsGeneration === this.generation
    ) {
      return this.cachedDefinitions
    }

    this.cachedDefinitions = this.list().map(t => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    }))
    this.cachedDefinitionsGeneration = this.generation
    return this.cachedDefinitions
  }
}
