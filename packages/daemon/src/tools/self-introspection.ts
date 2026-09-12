import type { AutonomyLevel, ILLMProvider, SkillMetadata } from '@sepilotd/core'
import type { SepilotdConfig } from '../config/schema.js'
import type { MarketplaceCatalog } from '../skills/marketplace-catalog.js'
import { MarketplaceSource } from '../skills/sources/marketplace.js'
import type { FileSkillRegistry } from '../skills/registry.js'
import type { InstallPipeline } from '../skills/install-pipeline.js'
import type { SkillSourceUrlPolicy } from '../skills/source-url-policy.js'
import {
  SkillAlreadyExistsError,
  SkillDigestMismatchError,
  SkillDigestRequiredError,
  SkillSourceUrlNotAllowedError,
  SkillValidationError,
} from '../skills/errors.js'
import type { PolicyEngine } from '../security/policy-engine.js'
import type { SkillStore } from '../skills/store.js'
import type { JobStore, ScheduledJob } from '../scheduler/job-store.js'
import { validateSkill } from '../skills/validator.js'
import { autonomyAllows } from '../utils/autonomy.js'
import type {
  ToolDefinitionRuntime,
  ToolExecutionContext,
  ToolRegistry,
  ToolResult,
} from './registry.js'

type ProviderRegistryView = {
  getDefault(): ILLMProvider | undefined
  list(): ILLMProvider[]
}

export interface SelfInfoToolDeps {
  config(): SepilotdConfig
  dataDir(): string
  providerRegistry(): ProviderRegistryView
  toolRegistry(): ToolRegistry
  skillRegistry(): FileSkillRegistry
  jobStore(): JobStore | undefined
  autonomy(): AutonomyLevel
}

export interface SkillHubSearchToolDeps {
  skillRegistry(): FileSkillRegistry
  toolRegistry(): ToolRegistry
  skillStore(): SkillStore | undefined
  marketplaceCatalog(): MarketplaceCatalog | undefined
  sourceUrlPolicy?(): SkillSourceUrlPolicy | undefined
  autonomy(): AutonomyLevel
}

export interface SkillHubInstallToolDeps extends SkillHubSearchToolDeps {
  policyEngine(): PolicyEngine
  installPipeline(): InstallPipeline
}

function truncate(value: string, max: number): string {
  const clean = value.replace(/\s+/g, ' ').trim()
  return clean.length <= max ? clean : `${clean.slice(0, max - 3).trimEnd()}...`
}

function defaultModelForProvider(config: SepilotdConfig, provider: ILLMProvider | undefined): string | null {
  if (!provider) return null
  return config.agent.defaultModel ?? provider.models[0]?.id ?? null
}

function currentModel(config: SepilotdConfig, registry: ProviderRegistryView) {
  const defaultProvider = config.agent.defaultProvider
  const defaultModel = config.agent.defaultModel
  if (defaultProvider && defaultModel) {
    return { providerId: defaultProvider, modelId: defaultModel }
  }
  const provider = registry.getDefault()
  const model = defaultModelForProvider(config, provider)
  return provider && model ? { providerId: provider.id, modelId: model } : null
}

function providerModels(config: SepilotdConfig, registry: ProviderRegistryView) {
  return registry.list().map((provider) => ({
    providerId: provider.id,
    providerName: provider.name,
    default: provider.id === config.agent.defaultProvider || registry.getDefault()?.id === provider.id,
    models: provider.models.map((model) => ({
      id: model.id,
      contextWindow: model.contextWindow,
      maxOutputTokens: model.maxOutputTokens,
      capabilities: model.capabilities,
    })),
  }))
}

function formatJob(job: ScheduledJob) {
  return {
    id: job.id,
    name: job.name,
    kind: job.kind,
    status: job.status,
    enabled: job.enabled,
    nextRunAt: new Date(job.nextRunAt).toISOString(),
    timezone: job.timezone,
    channelType: job.channelType,
    channelTarget: job.channelTarget,
    instruction: truncate(job.instruction, 220),
    lastError: job.lastError,
  }
}

function requestedSections(input: Record<string, unknown>): Set<string> | null {
  const raw = input.sections
  if (!Array.isArray(raw) || raw.length === 0) return null
  const values = raw
    .filter((item): item is string => typeof item === 'string')
    .map((item) => item.toLowerCase())
  return values.length > 0 ? new Set(values) : null
}

function wants(sections: Set<string> | null, name: string): boolean {
  return !sections || sections.has(name) || sections.has('all')
}

export function createSelfInfoTool(deps: SelfInfoToolDeps): ToolDefinitionRuntime {
  return {
    name: 'self.info',
    description:
      'Return an objective snapshot of this agent runtime: identity, current/default model, available models, tools, installed skills, active workspace, scheduled tasks, and known limits. Use this instead of guessing when the user asks what you are, what you can do, what model is active, where you are working, or what is scheduled.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'self-info' },
    inputSchema: {
      type: 'object',
      properties: {
        sections: {
          type: 'array',
          items: {
            type: 'string',
            enum: ['all', 'identity', 'model', 'tools', 'skills', 'workspace', 'schedules', 'limits'],
          },
          description: 'Optional subset to return. Defaults to all sections.',
        },
      },
    },
    async execute(input: Record<string, unknown>, context?: ToolExecutionContext): Promise<ToolResult> {
      const start = Date.now()
      const sections = requestedSections(input)
      const config = deps.config()
      const registry = deps.providerRegistry()
      const snapshot: Record<string, unknown> = {}

      if (wants(sections, 'identity')) {
        snapshot.identity = {
          daemon: 'sepilotd',
          device: {
            name: config.device.name,
            role: config.device.role,
          },
          autonomy: deps.autonomy(),
          ...(context?.workspaceRoot ? {} : { dataDir: deps.dataDir() }),
        }
      }

      if (wants(sections, 'model')) {
        snapshot.model = {
          current: currentModel(config, registry),
          providers: providerModels(config, registry),
        }
      }

      if (wants(sections, 'tools')) {
        const tools = deps.toolRegistry().list()
        snapshot.tools = {
          count: tools.length,
          names: tools.map((tool) => tool.name).sort(),
        }
      }

      if (wants(sections, 'skills')) {
        const skills = await deps.skillRegistry().listForCwd(
          context?.cwd,
          context?.workspaceRoot,
        )
        snapshot.skills = {
          count: skills.length,
          installed: skills
            .slice(0, 40)
            .map((skill) => ({
              id: skill.id,
              name: skill.name,
              description: skill.description,
              tags: skill.tags ?? [],
              autonomyRequired: skill.autonomy_required,
            })),
          truncated: skills.length > 40,
        }
      }

      if (wants(sections, 'workspace')) {
        snapshot.workspace = {
          cwd: context?.cwd ?? null,
          channel: context?.channelContext?.channel ?? null,
          channelTarget: context?.channelContext?.chatKey ?? null,
        }
      }

      if (wants(sections, 'schedules')) {
        const store = deps.jobStore()
        const channelTarget = context?.channelContext?.chatKey
        const jobs = store
          ? store.list({
              status: ['pending'],
              ...(channelTarget ? { channelTarget } : {}),
            })
          : []
        snapshot.schedules = {
          scope: channelTarget ? 'current-channel' : 'all-visible',
          count: jobs.length,
          pending: jobs
            .filter((job) => job.enabled)
            .sort((a, b) => a.nextRunAt - b.nextRunAt)
            .slice(0, 20)
            .map(formatJob),
          truncated: jobs.length > 20,
        }
      }

      if (wants(sections, 'limits')) {
        snapshot.limits = {
          autonomy: deps.autonomy(),
          statements: [
            'Only registered tools and installed skills are directly usable.',
            'External skills are not installed automatically; search skillhub.search, assess metadata, then ask the user before installation through skillhub.install.',
            'Credentials, secrets, destructive changes, and shared-workspace writes require explicit care and may require approval.',
            'Memory, schedules, prices, versions, and provider/model availability can be stale; verify with tools before relying on them.',
          ],
        }
      }

      return {
        status: 'success',
        output: JSON.stringify(snapshot, null, 2),
        durationMs: Date.now() - start,
      }
    },
  }
}

function assessSkill(
  metadata: SkillMetadata,
  deps: SkillHubSearchToolDeps,
  installed: boolean,
  source: string | null,
) {
  const currentAutonomy = deps.autonomy()
  const missingTools = (metadata.tools ?? []).filter((tool) => !deps.toolRegistry().get(tool))
  const warnings = [
    installed ? 'already installed' : '',
    !autonomyAllows(metadata.autonomy_required, currentAutonomy)
      ? `requires autonomy ${metadata.autonomy_required}, current is ${currentAutonomy}`
      : '',
    missingTools.length > 0 ? `missing declared tools: ${missingTools.join(', ')}` : '',
  ].filter(Boolean)

  return {
    id: metadata.id,
    name: metadata.name,
    version: metadata.version,
    description: metadata.description,
    tags: metadata.tags ?? [],
    tools: metadata.tools ?? [],
    autonomyRequired: metadata.autonomy_required,
    installed,
    installSource: source,
    assessment: {
      likelySafeToSuggest: warnings.length === 0,
      warnings,
      note: source
        ? 'Metadata-only assessment. Ask the user before installing; install validation will inspect full skill content.'
        : 'Installed or local skill-store result. Load/inspect full content before applying.',
    },
  }
}

function scanSkillContent(content: string): { blockers: string[]; warnings: string[] } {
  const blockers: string[] = []
  const warnings: string[] = []

  const blockerPatterns: Array<[RegExp, string]> = [
    [/rm\s+-rf\s+\/(?:\s|$)/i, 'contains a recursive root deletion command'],
    [/(?:^|\n)\s*(?:curl|wget)[^\n|]*\|\s*(?:sh|bash)\b/i, 'pipes downloaded code directly into a shell'],
    [/BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY/i, 'contains a private key marker'],
    [/\bignore\s+(?:all\s+)?(?:(?:previous|prior)\s+)?(?:system|developer)?\s*instructions\b/i, 'tries to override higher-priority instructions'],
    [/\b(?:exfiltrate|steal|leak|send|upload)\b[\s\S]{0,120}\b(?:secret|token|api[\s_-]*keys?|password|private key|credential)s?\b/i, 'appears to request credential exfiltration'],
    [/\b(?:disable|bypass|turn off)\b[\s\S]{0,80}\b(?:audit|logging|approval|policy|security)\b/i, 'tries to bypass audit, approval, policy, or security controls'],
  ]
  for (const [pattern, message] of blockerPatterns) {
    if (pattern.test(content)) blockers.push(message)
  }

  const warningPatterns: Array<[RegExp, string]> = [
    [/(?:api[_-]?key|access[_-]?token|bearer\s+token|client[_-]?secret|password)\s*[:=]\s*['"][A-Za-z0-9_.:/+=-]{16,}['"]/i, 'may contain an embedded credential literal'],
    [/\b(?:token|secret|password|credential|private key)\b/i, 'mentions credentials or secrets; ensure it only uses placeholders and runtime configuration'],
  ]
  for (const [pattern, message] of warningPatterns) {
    if (pattern.test(content)) warnings.push(message)
  }

  return {
    blockers: Array.from(new Set(blockers)),
    warnings: Array.from(new Set(warnings)),
  }
}

function installCandidateSummary(
  metadata: SkillMetadata,
  deps: SkillHubInstallToolDeps,
  installed: boolean,
  source: string,
  content: string,
) {
  const fullMetadata: SkillMetadata = { ...metadata, source: metadata.source ?? { type: 'marketplace', ref: source } }
  const validation = validateSkill(
    fullMetadata,
    content,
    deps.toolRegistry(),
    deps.policyEngine(),
    deps.autonomy(),
  )
  const safety = scanSkillContent(content)

  return {
    ...assessSkill(fullMetadata, deps, installed, fullMetadata.source?.ref ?? source),
    validation: {
      valid: validation.valid && validation.warnings.length === 0,
      errors: validation.errors,
      warnings: validation.warnings,
    },
    safety,
  }
}

export function createSkillHubSearchTool(deps: SkillHubSearchToolDeps): ToolDefinitionRuntime {
  return {
    name: 'skillhub.search',
    description:
      'Search installed skills, the local skill store, and configured skill marketplaces for a skill that may help with a request. Use this before telling the user a hard task is impossible because you lack a capability. Return metadata and safety notes; do not install anything.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'skillhub-search' },
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Capability or task to search for, e.g. "kubernetes ops" or "figma design review".',
        },
        marketplace: {
          type: 'string',
          description: 'Optional marketplace name to restrict search.',
        },
        limit: {
          type: 'number',
          description: 'Maximum candidates per source, default 5.',
        },
      },
      required: ['query'],
    },
    async execute(input: Record<string, unknown>, context?: ToolExecutionContext): Promise<ToolResult> {
      const start = Date.now()
      const query = typeof input.query === 'string' ? input.query.trim() : ''
      if (!query) {
        return {
          status: 'error',
          output: 'query must be a non-empty string',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }

      const limit = typeof input.limit === 'number' && Number.isFinite(input.limit)
        ? Math.max(1, Math.min(20, Math.floor(input.limit)))
        : 5
      const marketplace = typeof input.marketplace === 'string' && input.marketplace.trim()
        ? input.marketplace.trim()
        : undefined
      const installed = await deps.skillRegistry().listForCwd(
        context?.cwd,
        context?.workspaceRoot,
      )
      const installedIds = new Set(installed.map((skill) => skill.id))
      const localInstalled = (await deps.skillRegistry().searchForCwd(
        query,
        context?.cwd,
        context?.workspaceRoot,
      ))
        .slice(0, limit)
        .map((skill) => assessSkill(skill, deps, true, null))

      const localStore = deps.skillStore()
        ?.search(query, limit)
        .map((skill) => assessSkill(skill, deps, installedIds.has(skill.id), null)) ?? []

      const marketplaceCatalog = deps.marketplaceCatalog()
      let marketplaceResults: ReturnType<typeof assessSkill>[] = []
      let marketplaceError: string | undefined
      if (marketplaceCatalog) {
        try {
          const source = new MarketplaceSource({
            catalog: marketplaceCatalog,
            urlPolicy: deps.sourceUrlPolicy?.(),
          })
          const found = await source.search(query, { marketplace, limit })
          marketplaceResults = await Promise.all(found.map(async (result) => {
            const isInstalled = await deps.skillRegistry().get(result.metadata.id) !== null
            const assessed = assessSkill(
              result.metadata,
              deps,
              isInstalled,
              `${result.marketplace}/${result.metadata.id}`,
            )
            if (result.metadataOnly) {
              assessed.assessment.likelySafeToSuggest = false
              assessed.assessment.warnings.push(
                'marketplace search returned metadata only; preview the exact source with skillhub.install confirm=false before asking for install approval',
              )
            }
            return assessed
          }))
        } catch (error) {
          marketplaceError = error instanceof Error ? error.message : String(error)
        }
      }

      const output = {
        query,
        results: {
          installed: localInstalled,
          localStore,
          marketplace: marketplaceResults,
        },
        marketplaceError,
        guidance: [
          'If a marketplace result has likelySafeToSuggest=true, propose it as an optional install source and ask for explicit approval.',
          'Do not claim the skill is installed until installation succeeds.',
          'If warnings mention missing tools or insufficient autonomy, explain those limits before recommending it.',
        ],
      }

      return {
        status: 'success',
        output: JSON.stringify(output, null, 2),
        durationMs: Date.now() - start,
      }
    },
  }
}

export function createSkillHubInstallTool(deps: SkillHubInstallToolDeps): ToolDefinitionRuntime {
  return {
    name: 'skillhub.install',
    description:
      'Preview and install an external skill from a marketplace id/name, HTTPS SKILL.md URL, or GitHub repo/path. This is side-effectful and must only be called with confirm=true after explicit user approval. It validates required tools, autonomy, policy, and dangerous content before writing the skill.',
    resumeSafetyForInput(input: Record<string, unknown>) {
      return input.confirm === true ? 'replay-risky' : 'replay-safe'
    },
    scheduling: { mode: 'parallel-safe', resource: 'skillhub-install' },
    inputSchema: {
      type: 'object',
      properties: {
        source: {
          type: 'string',
          description: 'Install source such as "openai/foo", "foo", an HTTPS SKILL.md URL, or a GitHub repo/tree URL.',
        },
        confirm: {
          type: 'boolean',
          description: 'Must be true only after explicit user approval. False previews validation and asks for approval.',
        },
        force: {
          type: 'boolean',
          description: 'Unsafe force installs are refused by this agent-facing tool; use admin surfaces for manual overrides.',
        },
        expectedDigest: {
          type: 'string',
          description: 'Digest returned by confirm=false preview. Required with confirm=true so the exact reviewed content is installed.',
        },
      },
      required: ['source'],
    },
    async execute(input: Record<string, unknown>, _context?: ToolExecutionContext): Promise<ToolResult> {
      const start = Date.now()
      const source = typeof input.source === 'string' ? input.source.trim() : ''
      if (!source) {
        return {
          status: 'error',
          output: 'source must be a non-empty string',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }
      if (input.force === true) {
        return {
          status: 'error',
          output: JSON.stringify({
            source,
            error: 'force installs are disabled for skillhub.install',
            guidance: 'Use the admin skill install endpoint or CLI only after manual review.',
          }, null, 2),
          code: 'FORCE_INSTALL_UNAVAILABLE_PERMANENT',
          durationMs: Date.now() - start,
        }
      }
      const expectedDigest = typeof input.expectedDigest === 'string'
        ? input.expectedDigest.trim()
        : ''

      try {
        const preview = await deps.installPipeline().preview({ source })
        const candidates = await Promise.all(preview.fetched.map(async (item) => {
          const metadata: SkillMetadata = { ...item.metadata, source: item.source }
          const installed = await deps.skillRegistry().get(metadata.id) !== null
          return {
            metadata,
            content: item.content,
            summary: installCandidateSummary(metadata, deps, installed, item.source.ref, item.content),
          }
        }))

        const blocking = candidates.flatMap(({ summary }) => [
          summary.installed ? `skill already installed: ${summary.id}` : '',
          ...summary.validation.errors,
          ...summary.validation.warnings,
          ...summary.safety.blockers,
        ].filter(Boolean))
        const readyToInstall = blocking.length === 0
        const outputBase = {
          source,
          digest: preview.digest,
          readyToInstall,
          candidates: candidates.map(({ summary }) => summary),
        }

        if (input.confirm !== true) {
          return {
            status: 'success',
            output: JSON.stringify({
              ...outputBase,
              installed: false,
              requiredAction: readyToInstall
                ? 'Ask the user for explicit approval, then call skillhub.install with confirm=true.'
                : 'Do not install. Explain validation or safety blockers to the user.',
              nextInput: readyToInstall
                ? { source, confirm: true, expectedDigest: preview.digest }
                : undefined,
            }, null, 2),
            durationMs: Date.now() - start,
          }
        }

        if (!expectedDigest) {
          return {
            status: 'error',
            output: JSON.stringify({
              ...outputBase,
              installed: false,
              error: 'expectedDigest is required when confirm=true; run skillhub.install with confirm=false and ask the user to approve that exact digest first',
            }, null, 2),
            code: 'EXPECTED_DIGEST_REQUIRED_PERMANENT',
            durationMs: Date.now() - start,
          }
        }

        if (expectedDigest !== preview.digest) {
          return {
            status: 'error',
            output: JSON.stringify({
              ...outputBase,
              installed: false,
              expectedDigest,
              actualDigest: preview.digest,
              error: 'skill source changed between preview and install',
            }, null, 2),
            code: 'SKILL_DIGEST_MISMATCH_PERMANENT',
            durationMs: Date.now() - start,
          }
        }

        if (!readyToInstall) {
          return {
            status: 'error',
            output: JSON.stringify({
              ...outputBase,
              installed: false,
              blockers: Array.from(new Set(blocking)),
            }, null, 2),
            code: 'SKILL_VALIDATION_FAILED_PERMANENT',
            durationMs: Date.now() - start,
          }
        }

        const result = await deps.installPipeline().install({
          source,
          expectedDigest,
          requireExpectedDigest: true,
        })
        const installed: SkillMetadata[] = result.installed

        return {
          status: 'success',
          output: JSON.stringify({
            ...outputBase,
            installed: true,
            installedSkills: installed.map((skill) => ({
              id: skill.id,
              name: skill.name,
              version: skill.version,
              source: skill.source,
            })),
            guidance: 'The skill is installed. Use self.info or the skill tool to verify availability before relying on it.',
          }, null, 2),
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        let code = 'FETCH_FAILED_TRANSIENT'
        if (error instanceof SkillDigestMismatchError) {
          code = 'SKILL_DIGEST_MISMATCH_PERMANENT'
        } else if (error instanceof SkillDigestRequiredError) {
          code = 'EXPECTED_DIGEST_REQUIRED_PERMANENT'
        } else if (error instanceof SkillAlreadyExistsError) {
          code = 'SKILL_ALREADY_INSTALLED_PERMANENT'
        } else if (error instanceof SkillValidationError) {
          code = 'SKILL_VALIDATION_FAILED_PERMANENT'
        } else if (error instanceof SkillSourceUrlNotAllowedError) {
          code = 'SOURCE_NOT_ALLOWED_PERMANENT'
        } else if (error instanceof Error && error.name === 'SkillPathTraversalError') {
          code = 'PATH_TRAVERSAL_PERMANENT'
        } else if (/Only https|Unrecognised source|empty source/i.test(message)) {
          code = 'INVALID_SOURCE_PERMANENT'
        }
        return {
          status: 'error',
          output: JSON.stringify({
            source,
            error: message,
            ...(error instanceof SkillSourceUrlNotAllowedError
              ? { url: error.url, reason: error.reason }
              : {}),
            ...(error instanceof SkillDigestMismatchError
              ? { expectedDigest: error.expectedDigest, actualDigest: error.actualDigest }
              : {}),
            ...(error instanceof SkillDigestRequiredError
              ? { actualDigest: error.actualDigest }
              : {}),
          }, null, 2),
          code,
          durationMs: Date.now() - start,
        }
      }
    },
  }
}
