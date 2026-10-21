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

export interface DelegatedAgentExecutionPolicy {
  autonomy: AutonomyLevel
  requireToolApproval: boolean
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
  recordScratch?(path: string): Promise<void>
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
  autonomy?: AutonomyLevel
  provider?: string
  model?: string
  cwd?: string
  workspaceRoot?: string
  writingDocId?: string
  signal?: AbortSignal
  editCheckpoint?: EditCheckpointHandle
  workspaceMutation?: WorkspaceMutationHandle
  scopeTags?: string[]
  channelContext?: {
    channel: string
    chatKey: string
    triggerMessageId?: string
  }
  emitEvent?: (event: AgentEvent) => void
  runContract?: AgentRunContract
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

export interface ToolObservationCoverage {
  covers(
    observedInput: Record<string, unknown>,
    requestedInput: Record<string, unknown>,
    context: ToolInputNormalizationContext,
  ): boolean
  coversCollectively?(
    observedInputs: readonly Record<string, unknown>[],
    requestedInput: Record<string, unknown>,
    context: ToolInputNormalizationContext,
  ): boolean
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
  discoveryContext?: (input: string) => Promise<string>
  inputContractHint?: string
  unavailableReason?: (context: Pick<ToolExecutionContext, 'cwd' | 'workspaceRoot'>) => string | undefined
  cardinalityAliases?: readonly string[] | (() => readonly string[])
  canonicalReadUrlTemplates?: readonly string[] | (() => readonly string[])
  researchVerification?: 'direct-source'
  security?: ToolSecurityDescriptor
  resumeSafety?: ToolResumeSafety
  resumeSafetyForInput?(input: Record<string, unknown>): ToolResumeSafety
  scheduling?: ToolSchedulingHint
  timeoutPolicy?: 'wall-clock' | 'inactivity'
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

export interface ToolResultImage {
  mediaType: string
  data: string
  displayToClient?: boolean
}

export interface ToolResult {
  output: string
  status: 'success' | 'error'
  metadata?: ToolResultMetadata
  contentParts?: ContentPart[]
  executionPosture?: ToolExecutionPosture
  durationMs: number
  code?: string
  images?: ToolResultImage[]
}

export function rejectUnsupportedToolInputFields(
  tool: Pick<ToolDefinitionRuntime, 'name' | 'inputSchema' | 'inputContractHint'>,
  input: Record<string, unknown>,
  compatibilityFields: readonly string[] = [],
): ToolResult | null {
  if (tool.inputSchema.additionalProperties !== false) return null
  const properties = tool.inputSchema.properties
  const supported = new Set([
    ...Object.keys(properties && typeof properties === 'object' ? properties : {}),
    ...compatibilityFields,
  ])
  const unsupported = Object.keys(input).filter((key) => !supported.has(key))
  if (unsupported.length === 0) return null
  const fieldLabel = (key: string) => JSON.stringify(key.slice(0, 64).replace(/[^a-zA-Z0-9_.-]/g, '?'))
  const preview = unsupported.slice(0, 5).map(fieldLabel).join(', ')
  return {
    status: 'error',
    code: 'UNSUPPORTED_INPUT_FIELD_PERMANENT',
    durationMs: 0,
    output: `${tool.name}: unsupported input field(s): ${preview}${unsupported.length > 5 ? ', …' : ''}. Nothing was executed. Use only supported fields: ${[...supported].map(fieldLabel).join(', ')}. Correct the arguments before retrying; unsupported options are not applied.${tool.inputContractHint ? ` ${tool.inputContractHint}` : ''}`,
  }
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
  private pluginScope: { pluginId: string; names: Set<string> } | null = null

  register(tool: ToolDefinitionRuntime, options?: { source?: ToolRegistrationSource }): void {
    const source = options?.source ?? (this.pluginScope ? 'plugin' : 'builtin')
    const existing = this.sources.get(tool.name)
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

  beginPluginScope(pluginId: string): void {
    this.pluginScope = { pluginId, names: new Set() }
  }

  endPluginScope(): string[] {
    const names = this.pluginScope ? [...this.pluginScope.names] : []
    this.pluginScope = null
    return names
  }

  get(name: string): ToolDefinitionRuntime | undefined {
    if (this.disabledTools.has(name)) return undefined
    return this.tools.get(name)
  }

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

  listRegistered(): ToolDefinitionRuntime[] {
    return Array.from(this.tools.values())
  }

  toToolDefinitions(context?: Pick<ToolExecutionContext, 'cwd' | 'workspaceRoot'>): ToolDefinition[] {
    if (context) {
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
