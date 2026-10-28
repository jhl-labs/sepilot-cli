import { modelRoutingPolicySchema } from '../providers/model-routing-policy.js'
import { randomUUID } from 'node:crypto'
import { z } from 'zod'
import { rejectPrivateLiteralUrl } from '../utils/ssrf-guard.js'
import { eventRoutingConfigSchema } from '../event-routing/config-schema.js'

export const DEFAULT_RESUME_ARTIFACT_RETENTION_DAYS = 30

export const skillSourceSecurityConfigSchema = z.object({
  enforceUrlAllowlist: z.boolean().default(false),
  allowedHosts: z.array(z.string().trim().min(1)).default([]),
  allowedUrlPrefixes: z
    .array(
      z
        .string()
        .trim()
        .url()
        .refine((value) => /^https:\/\//i.test(value), {
          message: 'skill source URL prefixes must use https://',
        }),
    )
    .default([]),
  allowLocalPaths: z.boolean().default(false),
})

export const skillSourceSecurityUpdateSchema = skillSourceSecurityConfigSchema.partial()

export const skillSourceSecuritySchema = skillSourceSecurityConfigSchema.default({
  enforceUrlAllowlist: false,
  allowedHosts: [],
  allowedUrlPrefixes: [],
  allowLocalPaths: false,
})

const providerModelCapabilitiesOverrideSchema = z.object({
  vision: z.boolean().optional(),
  toolUse: z.boolean().optional(),
  streaming: z.boolean().optional(),
  embedding: z.boolean().optional(),
  thinking: z.boolean().optional(),
  thinkingControl: z.enum(['reasoning-effort', 'chat-template-kwargs', 'think-levels']).optional(),
  adaptivePromptReact: z.boolean().optional(),
  promptReactPreferred: z.boolean().optional(),
  deepCoderAnalysis: z.boolean().optional(),
})

const providerModelCompatibilitySchema = z.object({
  toolTransport: z.enum(['auto', 'native', 'prompt-react', 'adaptive']).optional(),
  reasoningEffortOffValue: z.enum(['none', 'minimal']).optional(),
  reasoningEffortAliases: z.record(z.enum(['low', 'medium', 'high']), z.string().trim().min(1)).optional(),
  auxiliaryThinkingOffFallback: z.enum(['low', 'medium', 'high', 'max']).optional(),
  repairThinkingOffFallback: z.enum(['low', 'medium', 'high', 'max']).optional(),
  answerProtocol: z.enum(['auto', 'repair-left-truncated-answer-stem']).optional(),
  notes: z.array(z.string().trim().min(1).max(500)).max(16).optional(),
}).refine(value => !value.reasoningEffortOffValue || Boolean(value.notes?.length), {
  message: 'An Off wire mapping requires endpoint/model evidence in compatibility.notes.',
  path: ['notes'],
}).refine(value => !value.auxiliaryThinkingOffFallback || Boolean(value.notes?.length), {
  message: 'An auxiliary thinking fallback requires endpoint/model evidence in compatibility.notes.',
  path: ['notes'],
}).refine(value => !value.repairThinkingOffFallback || Boolean(value.notes?.length), {
  message: 'A repair thinking fallback requires endpoint/model evidence in compatibility.notes.',
  path: ['notes'],
}).refine(value => !value.reasoningEffortAliases || Boolean(value.notes?.length), {
  message: 'Reasoning-effort aliases require endpoint/model evidence in compatibility.notes.',
  path: ['notes'],
})

export const promptCacheSchema = z.object({
  keyMode: z.enum(['session', 'prefix']).optional(),
  keyNamespace: z.string().trim().min(1).max(64).optional(),
  retention: z.enum(['in_memory', '24h']).optional(),
  options: z.object({ mode: z.enum(['implicit', 'explicit']), ttl: z.literal('30m').optional() }).strict().optional(),
  cachePrompt: z.boolean().optional(),
  breakpoint: z.enum(['cache-control', 'openai']).optional(),
  ttl: z.enum(['5m', '1h']).optional(),
}).strict().refine(value => !value.ttl || value.breakpoint === 'cache-control', {
  message: 'ttl requires the cache-control breakpoint dialect.',
}).refine(value => value.options?.mode !== 'explicit' || value.breakpoint === 'openai', {
  message: 'Explicit-only caching requires openai breakpoints.',
})

const providerModelOverrideSchema = z.object({
  id: z.string().trim().min(1),
  contextWindow: z.number().int().positive().optional(),
  maxOutputTokens: z.number().int().positive().optional(),
  capabilities: providerModelCapabilitiesOverrideSchema.optional(),
  compatibility: providerModelCompatibilitySchema.optional(),
  promptCache: promptCacheSchema.optional(),
})

const agentCapabilitiesSchema = z
  .object({
    hostSystemInfo: z.boolean().default(true),
  })
  .default({ hostSystemInfo: true })

export const sandboxDockerConfigSchema = z
  .object({
    image: z.string().trim().min(1).default('node:22-slim'),
    networkMode: z.enum(['none', 'bridge']).default('none'),
    mountMode: z.enum(['rw', 'ro']).default('rw'),
    cpuLimit: z.string().trim().min(1).default('1.0'),
    memoryLimit: z.string().trim().min(1).default('512m'),
    pidsLimit: z.number().int().positive().max(4096).default(100),
    readOnlyRootfs: z.boolean().default(true),
    noNewPrivileges: z.boolean().default(true),
    capDrop: z.array(z.string().trim().min(1)).default(['ALL']),
  })
  .default({
    image: 'node:22-slim',
    networkMode: 'none',
    mountMode: 'rw',
    cpuLimit: '1.0',
    memoryLimit: '512m',
    pidsLimit: 100,
    readOnlyRootfs: true,
    noNewPrivileges: true,
    capDrop: ['ALL'],
  })

export const sandboxBubblewrapConfigSchema = z
  .object({
    networkMode: z.enum(['none', 'host']).default('none'),
    readOnlyWorkspace: z.boolean().default(false),
    projectHostToolchains: z.boolean().default(true),
    bwrapPath: z.string().trim().min(1).optional(),
    processIsolation: z
      .enum(['namespace', 'container-boundary'])
      .default('namespace'),
    resourceLimits: z
      .object({
        pidsMax: z.number().int().nonnegative().optional(),
        memoryMaxBytes: z.number().int().nonnegative().optional(),
        tmpfsSizeBytes: z.number().int().nonnegative().optional(),
      })
      .optional(),
  })
  .default({
    networkMode: 'none',
    readOnlyWorkspace: false,
    projectHostToolchains: true,
    processIsolation: 'namespace',
  })

export const agentGraphNodeModelOverrideSchema = z
  .object({
    model: z.string().trim().min(1).optional(),
  })
  .default({})

export const agentGraphNodeModelOverridesSchema = z
  .record(z.record(agentGraphNodeModelOverrideSchema))
  .default({})

export const DEFAULT_NETWORK_CONFIG = {
  proxyMode: 'environment',
  proxyUrl: null,
  noProxy: null,
  customCaPath: null,
  tlsRejectUnauthorized: true,
  timeoutMs: 900_000,
  maxConcurrency: 4,
} as const

const networkProxyUrlSchema = z
  .string()
  .trim()
  .url()
  .superRefine((value, context) => {
    let url: URL
    try {
      url = new URL(value)
    } catch {
      return
    }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'network.proxyUrl must use http:// or https://',
      })
    }
    if (url.username || url.password) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'network.proxyUrl must not contain a username or password',
      })
    }
  })

const LEGACY_UNSAFE_PROXY_INPUT_MARKER = '__sepilotdLegacyUnsafeProxyInput' as const
const legacyUnsafeProxyRuntimeMarker: unique symbol = Symbol(
  'sepilotd.legacyUnsafeProxyRuntimeMarker',
)

type NetworkConfigRuntimeMetadata = {
  [legacyUnsafeProxyRuntimeMarker]?: true
}

export function hasLegacyUnsafeProxyMigration(value: unknown): boolean {
  return Boolean(
    value
    && typeof value === 'object'
    && (value as NetworkConfigRuntimeMetadata)[legacyUnsafeProxyRuntimeMarker],
  )
}

export function copyNetworkConfigRuntimeMetadata(
  source: unknown,
  target: object,
): void {
  if (!hasLegacyUnsafeProxyMigration(source)) return
  Object.defineProperty(target, legacyUnsafeProxyRuntimeMarker, {
    configurable: false,
    enumerable: false,
    value: true,
    writable: false,
  })
}

export const networkConfigSchema = z
  .preprocess(
    (input) => {
      if (!input || typeof input !== 'object' || Array.isArray(input)) return input
      const {
        [LEGACY_UNSAFE_PROXY_INPUT_MARKER]: _ignoredRuntimeMarker,
        ...legacy
      } = input as Record<string, unknown>
      if (legacy.proxyMode === undefined && legacy.proxyUrl != null) {
        if (
          typeof legacy.proxyUrl === 'string'
          && networkProxyUrlSchema.safeParse(legacy.proxyUrl).success
        ) {
          return { ...legacy, proxyMode: 'manual' }
        }
        return {
          ...legacy,
          proxyMode: 'direct',
          proxyUrl: null,
          [LEGACY_UNSAFE_PROXY_INPUT_MARKER]: true,
        }
      }
      return legacy
    },
    z
      .object({
        proxyMode: z.enum(['environment', 'direct', 'manual']).default('environment'),
        proxyUrl: networkProxyUrlSchema.nullable().default(null),
        noProxy: z.string().trim().min(1).max(4_096).nullable().default(null),
        customCaPath: z.string().trim().min(1).nullable().default(null),
        tlsRejectUnauthorized: z.boolean().default(true),
        timeoutMs: z.number().int().min(1_000).max(3_600_000).default(900_000),
        maxConcurrency: z.number().int().min(1).max(64).default(4),
        [LEGACY_UNSAFE_PROXY_INPUT_MARKER]: z.literal(true).optional(),
      })
      .superRefine((value, context) => {
        if (value.proxyMode === 'manual' && !value.proxyUrl) {
          context.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['proxyUrl'],
            message: 'network.proxyUrl is required when proxyMode is manual',
          })
        }
      })
      .transform((parsed) => {
        const {
          [LEGACY_UNSAFE_PROXY_INPUT_MARKER]: legacyUnsafeProxy,
          ...network
        } = parsed
        if (legacyUnsafeProxy) {
          copyNetworkConfigRuntimeMetadata({
            [legacyUnsafeProxyRuntimeMarker]: true,
          }, network)
        }
        return network
      }),
  )
  .default(DEFAULT_NETWORK_CONFIG)

export const providerSchema = z.object({
  id: z.string(),
  type: z.string().trim().min(1),
  apiKey: z.string().optional(),
  baseUrl: z.string().url().optional(),
  headers: z.record(z.string()).default({}),
  promptCache: promptCacheSchema.optional(),
  models: z.array(z.string()).default([]),
  defaultContextWindow: z.number().int().positive().optional(),
  defaultMaxOutputTokens: z.number().int().positive().optional(),
  disableDefaultPlugins: z.boolean().optional(),
  structuredToolCalling: z.boolean().optional(),
  capabilities: providerModelCapabilitiesOverrideSchema.optional(),
  modelOverrides: z.array(providerModelOverrideSchema).default([]),
  default: z.boolean().optional().default(false),
})

export const DECISION_PROVIDER_TYPES = ['ollama', 'systemone', 'ai-gateway', 'openai'] as const
export const decisionProviderSchema = z.object({
  id: z.string().trim().min(1),
  type: z.enum(DECISION_PROVIDER_TYPES),
  apiKey: z.string().optional(),
  baseUrl: z.string().url().optional(),
  headers: z.record(z.string()).default({}),
  models: z.array(z.string().trim().min(1)).default([]),
  timeoutMs: z.number().int().min(1000).max(120000).default(15000),
  default: z.boolean().optional().default(false),
})

export const channelConfigSchema = z.object({
  type: z.string().trim().min(1),
  enabled: z.boolean().optional().default(false),
  config: z.record(z.unknown()).optional(),
})

const mcpProvenanceSignatureSchema = z.object({
  algorithm: z.literal('ed25519'),
  keyId: z.string().trim().min(1).optional(),
  value: z.string().trim().min(1),
  verified: z.boolean(),
})

const mcpServerProvenanceSchema = z.object({
  source: z.enum(['builtin', 'marketplace', 'manual']),
  marketplace: z.string().trim().min(1).optional(),
  publisher: z.string().trim().min(1).optional(),
  homepage: z.string().trim().min(1).optional(),
  repository: z.string().trim().min(1).optional(),
  sourceRef: z.string().trim().min(1).optional(),
  templateDigest: z.string().trim().min(1).optional(),
  declaredDigest: z.string().trim().min(1).optional(),
  verified: z.boolean(),
  verification: z.enum(['builtin', 'signature', 'digest', 'manual', 'unverified']),
  signature: mcpProvenanceSignatureSchema.optional(),
  installedAt: z.string().trim().min(1).optional(),
})

const mcpToolManifestSchema = z.object({
  version: z.literal(1),
  generatedAt: z.string().trim().min(1),
  digest: z.string().trim().min(1),
  tools: z.array(
    z.object({
      name: z.string().trim().min(1),
      digest: z.string().trim().min(1),
    }),
  ),
})

const mcpToolSecuritySchema = z
  .record(z.enum(['observe', 'external-write']))
  .optional()

const mcpServerSecurityFieldsSchema = {
  provenance: mcpServerProvenanceSchema.optional(),
  toolManifest: mcpToolManifestSchema.optional(),
  toolSecurity: mcpToolSecuritySchema,
  toolCapabilities: z.record(z.array(z.enum(['process', 'service', 'terminal', 'browser', 'filesystem-read', 'filesystem-write', 'network', 'application-state']))).optional(),
}

const mcpServerToolTimeoutMsSchema = z
  .number()
  .int()
  .min(100)
  .max(600_000)
  .optional()

const stdioMcpServerSchema = z.object({
  name: z.string().trim().min(1),
  enabled: z.boolean().default(true),
  transport: z.literal('stdio').default('stdio'),
  command: z.string().trim().min(1),
  args: z.array(z.string()).default([]),
  env: z.record(z.string()).default({}),
  disabledTools: z.array(z.string()).default([]),
  timeoutMs: mcpServerToolTimeoutMsSchema,
  ...mcpServerSecurityFieldsSchema,
})

const sseMcpServerSchema = z.object({
  name: z.string().trim().min(1),
  enabled: z.boolean().default(true),
  transport: z.literal('sse'),
  url: z.string().url(),
  headers: z.record(z.string()).default({}),
  disabledTools: z.array(z.string()).default([]),
  timeoutMs: mcpServerToolTimeoutMsSchema,
  ...mcpServerSecurityFieldsSchema,
})

const httpMcpServerSchema = z.object({
  name: z.string().trim().min(1),
  enabled: z.boolean().default(true),
  transport: z.literal('http'),
  url: z.string().url(),
  headers: z.record(z.string()).default({}),
  disabledTools: z.array(z.string()).default([]),
  timeoutMs: mcpServerToolTimeoutMsSchema,
  ...mcpServerSecurityFieldsSchema,
})

export const mcpServerSchema = z.union([
  stdioMcpServerSchema,
  sseMcpServerSchema,
  httpMcpServerSchema,
])

export interface McpRootConfig {
  uri: string
  name?: string
  _meta?: Record<string, unknown>
}

export interface McpClientConfig {
  roots: {
    enabled: boolean
    listChanged: boolean
    entries: McpRootConfig[]
  }
  sampling: {
    enabled: boolean
    provider?: string
    model?: string
    maxTokens: number
    temperature?: number
  }
  elicitation: {
    enabled: boolean
    mode: 'decline' | 'accept-defaults'
    applyDefaults: boolean
  }
}

export function createDefaultMcpClientConfig(): McpClientConfig {
  return {
    roots: {
      enabled: false,
      listChanged: true,
      entries: [],
    },
    sampling: {
      enabled: false,
      maxTokens: 1024,
    },
    elicitation: {
      enabled: false,
      mode: 'decline',
      applyDefaults: true,
    },
  }
}

export const DEFAULT_MCP_CLIENT_CONFIG: McpClientConfig = createDefaultMcpClientConfig()

export const mcpRootSchema = z.object({
  uri: z
    .string()
    .trim()
    .min(1)
    .refine((value) => value.startsWith('file://'), { message: 'MCP roots must use file:// URIs' }),
  name: z.string().trim().min(1).optional(),
  _meta: z.record(z.unknown()).optional(),
})

export const mcpClientConfigSchema = z
  .object({
    roots: z
      .object({
        enabled: z.boolean().default(DEFAULT_MCP_CLIENT_CONFIG.roots.enabled),
        listChanged: z.boolean().default(DEFAULT_MCP_CLIENT_CONFIG.roots.listChanged),
        entries: z.array(mcpRootSchema).default([]),
      })
      .default(() => createDefaultMcpClientConfig().roots),
    sampling: z
      .object({
        enabled: z.boolean().default(DEFAULT_MCP_CLIENT_CONFIG.sampling.enabled),
        provider: z.string().trim().min(1).optional(),
        model: z.string().trim().min(1).optional(),
        maxTokens: z
          .number()
          .int()
          .positive()
          .max(8192)
          .default(DEFAULT_MCP_CLIENT_CONFIG.sampling.maxTokens),
        temperature: z.number().min(0).max(2).optional(),
      })
      .default(() => createDefaultMcpClientConfig().sampling),
    elicitation: z
      .object({
        enabled: z.boolean().default(DEFAULT_MCP_CLIENT_CONFIG.elicitation.enabled),
        mode: z
          .enum(['decline', 'accept-defaults'])
          .default(DEFAULT_MCP_CLIENT_CONFIG.elicitation.mode),
        applyDefaults: z.boolean().default(DEFAULT_MCP_CLIENT_CONFIG.elicitation.applyDefaults),
      })
      .default(() => createDefaultMcpClientConfig().elicitation),
  })
  .default(createDefaultMcpClientConfig)

export const memoryQdrantSchema = z.object({
  url: z.string().url(),
  apiKey: z.string().optional(),
  collection: z.string().trim().min(1).optional(),
})

export const memorySearchEngineSchema = z.object({
  url: z.string().url(),
  index: z.string().trim().min(1).optional(),
  apiKey: z.string().optional(),
  username: z.string().optional(),
  password: z.string().optional(),
})

export const memoryMeilisearchSchema = z.object({
  url: z.string().url(),
  index: z.string().trim().min(1).optional(),
  apiKey: z.string().optional(),
  embedder: z.string().trim().min(1).optional(),
})

export const memoryHttpAuthSchema = z.object({
  type: z.enum(['none', 'bearer', 'api-key', 'basic']).default('bearer'),
  headerName: z.string().trim().min(1).optional(),
  username: z.string().optional(),
  password: z.string().optional(),
})

export const memoryCustomApiSchema = z.object({
  url: z.string().url(),
  apiKey: z.string().optional(),
  auth: memoryHttpAuthSchema.optional(),
  headers: z.record(z.string()).default({}),
  healthPath: z.string().trim().min(1).default('/health'),
  configurePath: z.string().trim().min(1).optional(),
  upsertPath: z.string().trim().min(1).default('/vectors/upsert'),
  deletePath: z.string().trim().min(1).default('/vectors/delete'),
  searchPath: z.string().trim().min(1).default('/vectors/search'),
  clearPath: z.string().trim().min(1).optional(),
  timeoutMs: z.number().int().positive().max(120_000).default(15_000),
})

export const memoryRagRerankSchema = z.object({
  enabled: z.boolean().default(false),
  provider: z.enum(['local', 'custom-api']).default('local'),
  model: z.string().trim().min(1).optional(),
  endpoint: z.string().url().optional(),
  apiKey: z.string().optional(),
  auth: memoryHttpAuthSchema.optional(),
  headers: z.record(z.string()).default({}),
  healthPath: z.string().trim().min(1).optional(),
  timeoutMs: z.number().int().positive().max(120_000).default(10_000),
  weight: z.number().min(0).max(1).default(0.35),
})

export const memoryRagSettingsSchema = z
  .object({
    defaultLimit: z.number().int().min(1).max(50).default(8),
    candidateMultiplier: z.number().int().min(1).max(20).default(5),
    scoreThreshold: z.number().min(0).max(1).default(0),
    rerank: memoryRagRerankSchema.default({
      enabled: false,
      provider: 'local',
      timeoutMs: 10_000,
      weight: 0.35,
    }),
  })
  .default({
    defaultLimit: 8,
    candidateMultiplier: 5,
    scoreThreshold: 0,
    rerank: {
      enabled: false,
      provider: 'local',
      timeoutMs: 10_000,
      weight: 0.35,
    },
  })

export const memoryDreamingSchema = z
  .object({
    enabled: z.boolean().default(false),
    schedule: z
      .string()
      .trim()
      .min(1)
      .refine((value) => /^(\d+)(s|m|h|d)$/.test(value) || /^\*\/(\d+) \* \* \* \*$/.test(value), {
        message:
          'memory.dreaming.schedule must be an interval like 6h or a */N minute cron expression',
      })
      .default('6h'),
  })
  .default({
    enabled: false,
    schedule: '6h',
  })

export const memoryMaintenanceSchema = z
  .object({
    enabled: z.boolean().default(false),
    schedule: z
      .string()
      .trim()
      .min(1)
      .refine((value) => /^(\d+)(s|m|h|d)$/.test(value) || /^\*\/(\d+) \* \* \* \*$/.test(value), {
        message:
          'memory.maintenance.schedule must be an interval like 1d or a */N minute cron expression',
      })
      .default('1d'),
    maxAgeDays: z.number().int().min(1).default(90),
    maxImportance: z.number().min(0).max(1).default(0.1),
    dryRun: z.boolean().default(false),
  })
  .default({
    enabled: false,
    schedule: '1d',
    maxAgeDays: 90,
    maxImportance: 0.1,
    dryRun: false,
  })

export const memoryUserProfileSchema = z
  .object({
    enabled: z.boolean().default(false),
    schedule: z
      .string()
      .trim()
      .min(1)
      .refine((value) => /^(\d+)(s|m|h|d)$/.test(value) || /^\*\/(\d+) \* \* \* \*$/.test(value), {
        message:
          'memory.userProfile.schedule must be an interval like 1d or a */N minute cron expression',
      })
      .default('1d'),
    section: z.string().trim().min(1).default('User Profile'),
  })
  .default({
    enabled: false,
    schedule: '1d',
    section: 'User Profile',
  })

export const proactivityDigestSchema = z
  .object({
    enabled: z.boolean().default(false),
    schedule: z.string().trim().min(1).default('0 9 * * *'),
    channelType: z.string().trim().min(1).optional(),
    channelTarget: z.string().trim().min(1).optional(),
  })
  .default({
    enabled: false,
    schedule: '0 9 * * *',
  })

export const proactivitySchema = z
  .object({
    digest: proactivityDigestSchema.optional(),
  })
  .default({})

export const schedulerSurfaceAccessSchema = z
  .object({
    cli: z.boolean().default(false),
    desktop: z.boolean().default(false),
    mobile: z.boolean().default(true),
  })
  .default({
    cli: false,
    desktop: false,
    mobile: true,
  })

export const schedulerConfigSchema = z
  .object({
    enabled: z.boolean().default(true),
    timezone: z.string().trim().min(1).optional(),
    dailyTokenBudget: z.number().int().positive().optional(),
    maxConsecutiveFailures: z.number().int().min(1).max(100).default(5),
    maxIterations: z.number().int().min(1).max(200).default(40),
    surfaces: schedulerSurfaceAccessSchema,
  })
  .default({
    enabled: true,
    maxConsecutiveFailures: 5,
    maxIterations: 40,
    surfaces: {
      cli: false,
      desktop: false,
      mobile: true,
    },
  })

export const channelPipelineConfigSchema = z
  .object({
    sharedGroupContext: z.boolean().default(false),
    maxGlobalRuns: z.number().int().min(1).max(100).default(8),
    maxGlobalQueuedRuns: z.number().int().min(0).max(100).default(0),
    defaultWorkspaceRoot: z.string().trim().min(1).optional(),
  })
  .default({
    sharedGroupContext: false,
    maxGlobalRuns: 8,
    maxGlobalQueuedRuns: 0,
  })

export const outboundHookEventSchema = z.enum([
  'post:agent:run',
  'post:tool:execute',
  'post:process:start',
  'post:process:exit',
  'post:llm:call',
  'post:channel:msg',
])

export const DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG = {
  maxAttempts: 3,
  backoffMs: 250,
} as const

export const outboundWebhookRetrySchema = z
  .object({
    maxAttempts: z
      .number()
      .int()
      .min(1)
      .max(10)
      .default(DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG.maxAttempts),
    backoffMs: z
      .number()
      .int()
      .min(0)
      .max(60_000)
      .default(DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG.backoffMs),
  })
  .default(DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG)

export const outboundWebhookSchema = z.object({
  enabled: z.boolean().default(true),
  url: z
    .string()
    .url()
    .superRefine((value, ctx) => {
      if (process.env.SEPILOTD_OUTBOUND_WEBHOOK_ALLOW_PRIVATE === '1') return
      try {
        rejectPrivateLiteralUrl(new URL(value))
      } catch {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'outbound webhook url must not target a private/loopback/metadata address',
        })
      }
    }),
  events: z.array(outboundHookEventSchema).min(1),
  secret: z.string().optional(),
  headers: z.record(z.string()).default({}),
  retry: outboundWebhookRetrySchema.default(DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG),
})

export const commandHookEventSchema = z.enum([
  'post:subagent:start',
  'post:subagent:stop',
  'pre:agent:run',
  'post:agent:run',
  'pre:tool:execute',
  'post:tool:execute',
  'post:process:start',
  'post:process:exit',
  'pre:llm:call',
  'post:llm:call',
  'pre:channel:msg',
  'post:channel:msg',
  'post:file:edit',
  'pre:user:prompt',
  'pre:context:compact',
  'post:context:compact',
])

export const commandHookSchema = z.object({
  enabled: z.boolean().default(true),
  event: commandHookEventSchema,
  toolMatcher: z.string().optional(),
  command: z.string().min(1),
  timeoutMs: z.number().int().min(100).max(60_000).optional(),
  async: z.boolean().optional(),
}).strict().refine((hook) => !hook.async || hook.event.startsWith('post:'), {
  message: 'Async command hooks are observational and only supported for post events; they cannot gate an operation',
  path: ['async'],
})

export const DEFAULT_CHANNEL_PIPELINE_HEALTH_THRESHOLDS = {
  minRecentEvents: 5,
  degradeFailureRate: 0.25,
  minAgentSamples: 3,
  degradeAgentAvgLatencyMs: 5_000,
} as const

export const DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG = {
  ...DEFAULT_CHANNEL_PIPELINE_HEALTH_THRESHOLDS,
  hotChannelTopN: 3,
  byChannelType: {},
} as const

export const DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG = {
  degradeWhenUnreadyEndpointsAtLeast: 1,
  detailTopMissingRequirements: 3,
  detailTopUnreadyRoutes: 3,
} as const

export const DEFAULT_WEBHOOK_SECURITY_POLICY = {
  signatureMaxSkewSeconds: 300,
  verificationUnavailableStatus: 'service_unavailable',
  byChannelType: {},
} as const

export const channelPipelineHealthThresholdSchema = z.object({
  minRecentEvents: z
    .number()
    .int()
    .positive()
    .default(DEFAULT_CHANNEL_PIPELINE_HEALTH_THRESHOLDS.minRecentEvents),
  degradeFailureRate: z
    .number()
    .min(0)
    .max(1)
    .default(DEFAULT_CHANNEL_PIPELINE_HEALTH_THRESHOLDS.degradeFailureRate),
  minAgentSamples: z
    .number()
    .int()
    .positive()
    .default(DEFAULT_CHANNEL_PIPELINE_HEALTH_THRESHOLDS.minAgentSamples),
  degradeAgentAvgLatencyMs: z
    .number()
    .int()
    .nonnegative()
    .default(DEFAULT_CHANNEL_PIPELINE_HEALTH_THRESHOLDS.degradeAgentAvgLatencyMs),
})

export const channelPipelineHealthOverrideSchema = channelPipelineHealthThresholdSchema.partial()

export const channelPipelineHealthConfigSchema = channelPipelineHealthThresholdSchema.extend({
  hotChannelTopN: z
    .number()
    .int()
    .positive()
    .max(20)
    .default(DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG.hotChannelTopN),
  byChannelType: z.record(channelPipelineHealthOverrideSchema).default({}),
})

export const webhookSecurityHealthConfigSchema = z.object({
  degradeWhenUnreadyEndpointsAtLeast: z
    .number()
    .int()
    .positive()
    .max(1000)
    .default(DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG.degradeWhenUnreadyEndpointsAtLeast),
  detailTopMissingRequirements: z
    .number()
    .int()
    .min(0)
    .max(20)
    .default(DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG.detailTopMissingRequirements),
  detailTopUnreadyRoutes: z
    .number()
    .int()
    .min(0)
    .max(20)
    .default(DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG.detailTopUnreadyRoutes),
})

export const webhookSecurityPolicyThresholdSchema = z.object({
  signatureMaxSkewSeconds: z
    .number()
    .int()
    .positive()
    .max(3600)
    .default(DEFAULT_WEBHOOK_SECURITY_POLICY.signatureMaxSkewSeconds),
  verificationUnavailableStatus: z
    .enum(['service_unavailable', 'not_found'])
    .default(DEFAULT_WEBHOOK_SECURITY_POLICY.verificationUnavailableStatus),
})

export const webhookSecurityPolicyOverrideSchema = webhookSecurityPolicyThresholdSchema.partial()

export const webhookSecurityPolicyConfigSchema = webhookSecurityPolicyThresholdSchema.extend({
  byChannelType: z.record(webhookSecurityPolicyOverrideSchema).default({}),
})

export const DEFAULT_APP_LOG_ROTATION = {
  enabled: true,
  maxBytes: 10 * 1024 * 1024,
  maxFiles: 3,
} as const

export const DEFAULT_TRACE_LOG_ROTATION = {
  enabled: true,
  maxBytes: 50 * 1024 * 1024,
  maxFiles: 3,
} as const

export const logRotationSchema = z.object({
  enabled: z.boolean().default(true),
  maxBytes: z
    .number()
    .int()
    .min(1024)
    .max(10_737_418_240)
    .default(DEFAULT_APP_LOG_ROTATION.maxBytes),
  maxFiles: z.number().int().min(1).max(50).default(DEFAULT_APP_LOG_ROTATION.maxFiles),
})

export const loggingSchema = z
  .object({
    level: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
    jsonFormat: z.boolean().default(false),
    app: logRotationSchema.default(DEFAULT_APP_LOG_ROTATION),
    trace: logRotationSchema.default(DEFAULT_TRACE_LOG_ROTATION),
  })
  .default({
    level: 'info',
    jsonFormat: false,
    app: DEFAULT_APP_LOG_ROTATION,
    trace: DEFAULT_TRACE_LOG_ROTATION,
  })

export type LogRotationConfig = z.infer<typeof logRotationSchema>
export type LoggingConfig = z.infer<typeof loggingSchema>

export const voiceConfigSchema = z
  .object({
    transcription: z
      .object({
        whisperBin: z.string().trim().min(1).optional(),
        model: z.string().trim().min(1).optional(),
        language: z.string().trim().min(1).optional(),
      })
      .default({}),
    synthesis: z
      .object({
        piperBin: z.string().trim().min(1).optional(),
        model: z.string().trim().min(1).optional(),
        language: z.string().trim().min(1).optional(),
      })
      .default({}),
    fileRetentionMs: z.number().int().positive().default(60 * 60 * 1000),
    maxFiles: z.number().int().positive().default(200),
    maxConcurrent: z.number().int().positive().default(2),
  })
  .default({
    transcription: {},
    synthesis: {},
    fileRetentionMs: 60 * 60 * 1000,
    maxFiles: 200,
    maxConcurrent: 2,
  })

export type VoiceConfig = z.infer<typeof voiceConfigSchema>

export const pluginSecurityConfigSchema = z
  .object({
    strict: z.boolean().default(false),
    loadTimeoutMs: z.number().int().min(100).max(120_000).default(10_000),
    trustedSignatureKeys: z
      .array(
        z.object({
          id: z.string().trim().min(1),
          publicKey: z.string().trim().min(1),
        }),
      )
      .default([]),
  })
  .default({
    strict: false,
    loadTimeoutMs: 10_000,
    trustedSignatureKeys: [],
  })

const TRUSTED_SEARCH_DOMAIN_PATTERN =
  /^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/i

export const webSearchTrustedDomainSchema = z
  .string()
  .trim()
  .refine(
    (value) =>
      TRUSTED_SEARCH_DOMAIN_PATTERN.test(value) &&
      !/^(?:\d{1,3}\.){3}\d{1,3}$/.test(value),
    {
      message: 'Expected a domain such as example.com (without scheme, path, port, or wildcard)',
    },
  )
  .transform((value) => value.toLowerCase())

export const webSearchTrustedDomainsSchema = z
  .array(webSearchTrustedDomainSchema)
  .max(100)
  .transform((domains) => [...new Set(domains)])

export const webSearchProviderIdSchema = z.enum([
  'auto',
  'duckduckgo',
  'brave',
  'tavily',
  'searxng',
  'ai-search',
])

export const webSearchProviderUpdateSchema = z.object({
  provider: webSearchProviderIdSchema,
  apiKey: z.string().trim().optional(),
  endpoint: z.union([z.string().url(), z.literal('')]).optional(),
})

export const webSearchConfigSchema = z
  .object({
    trustedDomains: webSearchTrustedDomainsSchema.default([]),
    provider: webSearchProviderIdSchema.default('auto'),
    apiKey: z.string().trim().min(1).optional(),
    endpoint: z.string().url().optional(),
  })
  .default({
    trustedDomains: [],
  })

export const DEFAULT_DEVICE_NAME = 'sepilot-device'

const browserSiteHostPattern = z
  .string()
  .trim()
  .min(1)
  .max(253)
  .regex(/^(\*\.)?[a-z0-9.-]+$/i, 'host pattern must be a host or *.host')
const browserSecurityConfigSchema = z
  .object({
    sites: z
      .object({
        allow: z.array(browserSiteHostPattern).default([]),
        deny: z.array(browserSiteHostPattern).default([]),
      })
      .default({ allow: [], deny: [] }),
  })
  .default({ sites: { allow: [], deny: [] } })
const browserConfigSchema = z
  .object({
    cdp: z
      .object({
        enabled: z.boolean().default(true),
        channel: z.enum(['stable', 'beta', 'dev', 'canary']).default('stable'),
        userDataDir: z.string().trim().min(1).optional(),
      })
      .default({ enabled: true, channel: 'stable' }),
  })
  .default({ cdp: { enabled: true, channel: 'stable' } })

const viewsConfigSchema = z
  .object({
    stills: z
      .object({
        enabled: z.boolean().default(true),
        maxWidth: z.number().int().min(160).max(1280).default(480),
        format: z.enum(['jpeg', 'png']).default('jpeg'),
      })
      .default({ enabled: true, maxWidth: 480, format: 'jpeg' }),
  })
  .default({ stills: { enabled: true, maxWidth: 480, format: 'jpeg' } })

export const configSchema = z.object({
  version: z.literal(1),
  configRevision: z.number().int().nonnegative().optional(),
  device: z
    .object({
      id: z.string().default(() => randomUUID()),
      name: z.string().default(DEFAULT_DEVICE_NAME),
      role: z.enum(['desktop', 'server', 'edge']).default('desktop'),
    })
    .default(() => ({
      id: randomUUID(),
      name: DEFAULT_DEVICE_NAME,
      role: 'desktop' as const,
    })),
  daemon: z
    .object({
      port: z.number().int().min(1024).max(65535).default(17600),
      host: z.string().default('127.0.0.1'),
      resumeArtifactRetentionDays: z
        .number()
        .int()
        .min(1)
        .max(3650)
        .default(DEFAULT_RESUME_ARTIFACT_RETENTION_DAYS),
    })
    .default({
      port: 17600,
      host: '127.0.0.1',
      resumeArtifactRetentionDays: DEFAULT_RESUME_ARTIFACT_RETENTION_DAYS,
    }),
  gateway: z
    .object({
      url: z.string().url().default('http://127.0.0.1:17610'),
    })
    .default({ url: 'http://127.0.0.1:17610' }),
  network: networkConfigSchema,
  webSearch: webSearchConfigSchema,
  retention: z
    .object({
      enabled: z.boolean().default(true),
      sessionDays: z.number().int().min(1).max(3650).optional(),
      memoryDays: z.number().int().min(1).max(3650).optional(),
      usageDays: z.number().int().min(1).max(3650).optional(),
      sweepIntervalHours: z.number().int().min(1).max(720).default(24),
    })
    .default({ enabled: true, sweepIntervalHours: 24 }),
  providers: z.array(providerSchema).min(0),
  decisionProviders: z.array(decisionProviderSchema).default([]),
  agent: z
    .object({
      autonomy: z
        .enum(['readonly', 'accept-edits', 'workspace-write', 'supervised', 'autonomous'])
        .default('supervised'),
      thinkingLevel: z.enum(['auto', 'off', 'low', 'medium', 'high', 'max']).default('medium'),
      defaultProvider: z.string().optional(),
      defaultModel: z.string().optional(),
      modelRouting: modelRoutingPolicySchema.optional(),
      auxModel: z.string().trim().min(1).optional(),
      mode: z.string().trim().min(1).default('instant'),
      persona: z.string().trim().min(1).optional(),
      outputLanguage: z.string().trim().min(1).default('auto'),
      intentRouter: z
        .object({
          enabled: z.boolean().default(true),
          provider: z.string().optional(),
          model: z.string().optional(),
          timeoutMs: z.number().int().min(500).max(60_000).default(12_000),
          maxPreviousMessages: z.number().int().min(0).max(20).default(4),
          perMessageCharLimit: z.number().int().min(50).max(2000).default(200),
          reasonMaxChars: z.number().int().min(50).max(2000).default(500),
        })
        .default({
          enabled: true,
          timeoutMs: 12_000,
          maxPreviousMessages: 4,
          perMessageCharLimit: 200,
          reasonMaxChars: 500,
        }),
      capabilities: agentCapabilitiesSchema,
      disabledTools: z.array(z.string().trim().min(1)).default([]),
      graphNodeModelOverrides: agentGraphNodeModelOverridesSchema,
    })
    .default({
      autonomy: 'supervised',
      thinkingLevel: 'medium',
      mode: 'instant',
      capabilities: { hostSystemInfo: true },
      disabledTools: [],
      graphNodeModelOverrides: {},
      intentRouter: {
        enabled: true,
        timeoutMs: 12_000,
        maxPreviousMessages: 4,
        perMessageCharLimit: 200,
        reasonMaxChars: 500,
      },
    }),
  limits: z
    .object({
      dailyUsdBudget: z.number().positive().optional(),
      sessionUsdBudget: z.number().positive().optional(),
    })
    .default({}),
  channels: z.array(channelConfigSchema).optional().default([]),
  plugins: pluginSecurityConfigSchema,
  channelPipeline: channelPipelineConfigSchema,
  hooks: z
    .object({
      outboundWebhooks: z.array(outboundWebhookSchema).default([]),
      commandHooks: z.array(commandHookSchema).default([]),
    })
    .default({ outboundWebhooks: [], commandHooks: [] }),
  eventRouting: eventRoutingConfigSchema,
  mcp: z
    .object({
      servers: z.array(mcpServerSchema).default([]),
      client: mcpClientConfigSchema,
    })
    .default(() => ({
      servers: [],
      client: createDefaultMcpClientConfig(),
    })),
  memory: z
    .object({
      encryption: z.boolean().default(true),
      allowPlaintextSecrets: z.boolean().default(false),
      vectorBackend: z
        .enum([
          'auto',
          'sqlite-vec',
          'sqlite-scan',
          'qdrant',
          'opensearch',
          'elasticsearch',
          'meilisearch',
          'custom-api',
        ])
        .default('auto'),
      directApiToolset: z.enum(['lean', 'full', 'none']).default('lean'),
      qdrant: memoryQdrantSchema.optional(),
      opensearch: memorySearchEngineSchema.optional(),
      elasticsearch: memorySearchEngineSchema.optional(),
      meilisearch: memoryMeilisearchSchema.optional(),
      customApi: memoryCustomApiSchema.optional(),
      rag: memoryRagSettingsSchema.optional(),
      dreaming: memoryDreamingSchema.optional(),
      maintenance: memoryMaintenanceSchema.optional(),
      userProfile: memoryUserProfileSchema.optional(),
      embeddingProvider: z.string().optional(),
      embeddingModel: z.string().optional(),
    })
    .refine(
      (memory) =>
        (memory.embeddingProvider && memory.embeddingModel) ||
        (!memory.embeddingProvider && !memory.embeddingModel),
      {
        message: 'memory.embeddingProvider and memory.embeddingModel must be set together',
        path: ['embeddingProvider'],
      },
    )
    .default({ encryption: true, vectorBackend: 'auto' }),
  proactivity: proactivitySchema.optional(),
  notifications: z
    .object({
      chatCompletion: z
        .object({
          enabled: z.boolean().default(true),
          includeFailures: z.boolean().default(true),
        })
        .default({ enabled: true, includeFailures: true }),
    })
    .optional(),
  voice: voiceConfigSchema,
  scheduler: schedulerConfigSchema.optional(),
  skills: z
    .object({
      trustProjectSkills: z.boolean().default(false),
    })
    .default({ trustProjectSkills: false }),
  files: z
    .object({
      searchRoots: z.array(z.string().trim().min(1)).default([]),
    })
    .default({ searchRoots: [] }),
  security: z
    .object({
      toolPolicy: z.string().default('policies.yaml'),
      auditLog: z.boolean().default(true),
      sandbox: z.enum(['local', 'docker', 'bubblewrap']).default('local'),
      sandboxHostFileTools: z.enum(['deny', 'allow']).optional(),
      sandboxDocker: sandboxDockerConfigSchema.optional(),
      sandboxBubblewrap: sandboxBubblewrapConfigSchema.optional(),
      egressAllowlist: z.array(z.string().trim().min(1)).optional(),
      skillSources: skillSourceSecuritySchema,
      webhooks: webhookSecurityPolicyConfigSchema.default(DEFAULT_WEBHOOK_SECURITY_POLICY),
      browser: browserSecurityConfigSchema,
    })
    .default({
      toolPolicy: 'policies.yaml',
      auditLog: true,
      sandbox: 'local',
      sandboxDocker: sandboxDockerConfigSchema.parse({}),
      skillSources: skillSourceSecuritySchema.parse({}),
      webhooks: DEFAULT_WEBHOOK_SECURITY_POLICY,
      browser: browserSecurityConfigSchema.parse({}),
    }),
  browser: browserConfigSchema,
  views: viewsConfigSchema,
  share: z
    .object({
      public: z
        .object({
          enabled: z.boolean().default(false),
          secret: z.string().default(''),
          ttlSeconds: z.number().int().positive().max(86_400).default(600),
          baseUrl: z.string().trim().url().optional(),
        })
        .default({
          enabled: false,
          secret: '',
          ttlSeconds: 600,
        }),
    })
    .default({
      public: {
        enabled: false,
        secret: '',
        ttlSeconds: 600,
      },
    }),
  observability: z
    .object({
      telemetry: z.boolean().default(false),
      otlpEndpoint: z.string().default(''),
      channelPipelineHealth: channelPipelineHealthConfigSchema.default(
        DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG,
      ),
      webhookSecurityHealth: webhookSecurityHealthConfigSchema.default(
        DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG,
      ),
    })
    .default({
      telemetry: false,
      otlpEndpoint: '',
      channelPipelineHealth: DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG,
      webhookSecurityHealth: DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG,
    }),
  logging: loggingSchema,
})

export type SepilotdConfig = z.infer<typeof configSchema>
export type ChannelPipelineHealthThresholdConfig = z.infer<
  typeof channelPipelineHealthThresholdSchema
>
export type ChannelPipelineHealthOverrideConfig = z.infer<
  typeof channelPipelineHealthOverrideSchema
>
export type ChannelPipelineHealthConfig = z.infer<typeof channelPipelineHealthConfigSchema>
export type WebhookSecurityHealthConfig = z.infer<typeof webhookSecurityHealthConfigSchema>
export type WebhookSecurityPolicyThresholdConfig = z.infer<
  typeof webhookSecurityPolicyThresholdSchema
>
export type WebhookSecurityPolicyOverrideConfig = z.infer<
  typeof webhookSecurityPolicyOverrideSchema
>
export type WebhookSecurityPolicyConfig = z.infer<typeof webhookSecurityPolicyConfigSchema>
