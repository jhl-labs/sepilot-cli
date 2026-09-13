import { randomUUID } from 'node:crypto'
import { z } from 'zod'
import { rejectPrivateLiteralUrl } from '../utils/ssrf-guard.js'

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
  thinkingControl: z.enum(['reasoning-effort', 'chat-template-kwargs']).optional(),
  adaptivePromptReact: z.boolean().optional(),
  promptReactPreferred: z.boolean().optional(),
  deepCoderAnalysis: z.boolean().optional(),
})

const providerModelCompatibilitySchema = z.object({
  toolTransport: z.enum(['auto', 'native', 'prompt-react', 'adaptive']).optional(),
  answerProtocol: z.enum(['auto', 'repair-left-truncated-answer-stem']).optional(),
  notes: z.array(z.string().trim().min(1).max(500)).max(16).optional(),
})

const providerModelOverrideSchema = z.object({
  id: z.string().trim().min(1),
  contextWindow: z.number().int().positive().optional(),
  maxOutputTokens: z.number().int().positive().optional(),
  capabilities: providerModelCapabilitiesOverrideSchema.optional(),
  compatibility: providerModelCompatibilitySchema.optional(),
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
    // Host-DoS resource caps applied via prlimit + a size-bounded /tmp tmpfs.
    // Omit to use safe defaults; set a field to 0 to disable that cap.
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
    // `z.string().url()` records malformed strings as validation errors, but
    // Zod still runs refinements afterwards.  Do not turn an invalid request
    // into an uncaught `new URL()` exception (and therefore a 500 response).
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

/**
 * A pre-proxyMode config with an unsupported or credential-bearing proxy URL
 * is kept parseable for upgrades, but must remain fail-closed until the user
 * explicitly saves a current network policy.
 */
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

/**
 * Process-wide outbound HTTP policy used by the daemon's global undici
 * dispatcher. `maxConcurrency` is the per-origin connection-pool limit, not a
 * daemon-wide agent-run semaphore.
 */
export const networkConfigSchema = z
  .preprocess(
    (input) => {
      if (!input || typeof input !== 'object' || Array.isArray(input)) return input
      const {
        [LEGACY_UNSAFE_PROXY_INPUT_MARKER]: _ignoredRuntimeMarker,
        ...legacy
      } = input as Record<string, unknown>
      // Before proxyMode existed, a non-null proxyUrl meant manual proxy.
      // Old releases accepted schemes and embedded credentials that the
      // canonical contract intentionally rejects. Do not let one unsafe old
      // value brick an upgrade: retain a canonical direct-shaped value with
      // non-serialized runtime metadata that forces fail-closed egress. A
      // current client that explicitly sends proxyMode=manual still goes
      // through strict validation below and receives a 400 from the API.
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
  // /v1/models discovery가 실패하거나 endpoint가 catalog를 안 주는 provider도
  // 일단 등록해두고 나중에 모델을 채울 수 있도록 0개 허용. 모델 없는 provider는
  // chat 흐름에서 단순히 후보로 잡히지 않을 뿐, config 자체는 유효.
  models: z.array(z.string()).default([]),
  defaultContextWindow: z.number().int().positive().optional(),
  defaultMaxOutputTokens: z.number().int().positive().optional(),
  capabilities: providerModelCapabilitiesOverrideSchema.optional(),
  modelOverrides: z.array(providerModelOverrideSchema).default([]),
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

const mcpServerSecurityFieldsSchema = {
  provenance: mcpServerProvenanceSchema.optional(),
  toolManifest: mcpToolManifestSchema.optional(),
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

// Opt-in agent-driven "deepening user model" loop. When enabled, the
// scheduler creates a recurring headless run that re-invokes the agent
// to reflect on recent daily notes / conversations and refresh a "User
// Profile" section of MEMORY.md (durable role / preferences / working
// style / recurring projects). Default off — it writes to the default
// memory scope, so multi-user deployments should leave it disabled or
// configure it deliberately.
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

// Opt-in "agent reaches out first" digest. When enabled with a target
// chat, the scheduler creates a recurring channel-bound run that
// re-invokes the agent to surface anything pending or noteworthy
// (open-loop daily notes, due reminders, pending scheduled tasks) and
// delivers a short briefing to that chat. Default off; needs an
// explicit channelType + channelTarget so it does not message a chat
// the operator did not pick. The schedule accepts a full cron string
// (so "0 9 * * *" = 9am daily) as well as the interval shorthand.
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
    // A scheduled run collects before it reports — quotes, searches, fetches,
    // an artifact write and its read-back — so the interactive-chat default of
    // 10 iterations cannot finish one. A nightly market briefing over 13
    // tickers exhausted it and reported INCOMPLETE with the report already
    // written.
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
    /** Optional collection root applied to new channel sessions without an explicit /cwd. */
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
  // Fail-fast egress guard at config load: reject a literal private/loopback/
  // metadata target so a misconfigured exfil URL is caught before any delivery.
  // Hostnames are validated at delivery time by the DNS-resolving guard in
  // hooks/outbound-webhook.ts. Bypass with SEPILOTD_OUTBOUND_WEBHOOK_ALLOW_PRIVATE=1.
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
  /** Regex matched against the tool name for tool-scoped events. */
  toolMatcher: z.string().optional(),
  /** Shell command. Receives the hook payload as JSON on stdin; exit 2 aborts. */
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

// Application log rotation. Applies to `~/.sepilotd/logs/daemon.log` and
// `~/.sepilotd/logs/agent-trace.jsonl`. Without rotation these grow without
// bound on long-running daemons, especially the trace file which records full
// LLM payloads. Defaults match service-supervisor: 10 MiB × 3 files.
//
// Disabling rotation (`enabled: false`) reverts to the legacy unbounded
// behaviour. Use it only when an external log shipper handles rolling.
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
    // Retention for server-synthesized voice WAVs: evicted + unlinked past the
    // TTL or when the count exceeds the max (was an unbounded disk/memory leak).
    fileRetentionMs: z.number().int().positive().default(60 * 60 * 1000),
    maxFiles: z.number().int().positive().default(200),
    // Max concurrent STT/TTS jobs; bounds per-call model reloads so N parallel
    // voice requests do not fan out into N model loads and OOM the host.
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

/**
 * Granular update payload for the search backend. `apiKey` accepts the
 * `***redacted***` placeholder a settings UI gets back from GET /config, which
 * means "keep the stored key" — without it, saving any other field on the form
 * would wipe the credential.
 */
export const webSearchProviderUpdateSchema = z.object({
  provider: webSearchProviderIdSchema,
  apiKey: z.string().trim().optional(),
  endpoint: z.union([z.string().url(), z.literal('')]).optional(),
})

export const webSearchConfigSchema = z
  .object({
    trustedDomains: webSearchTrustedDomainsSchema.default([]),
    /**
     * Search backend. `auto` picks a keyed provider when its credential is in
     * the environment and otherwise falls back to keyless DuckDuckGo, which
     * frequently answers automated queries with an anti-bot challenge.
     */
    provider: webSearchProviderIdSchema.default('auto'),
    /** Credential for the selected provider; may also come from the environment. */
    apiKey: z.string().trim().min(1).optional(),
    /** Base URL of a self-hosted SearXNG or compatible AI Search instance. */
    endpoint: z.string().url().optional(),
  })
  .default({
    trustedDomains: [],
  })

/** Host-neutral placeholder: never derived from OS user/host identity. */
export const DEFAULT_DEVICE_NAME = 'sepilot-device'

export const configSchema = z.object({
  version: z.literal(1),
  configRevision: z.number().int().nonnegative().optional(),
  // First boot into an empty data dir must not require the operator to
  // hand-write a device block. The identity is auto-provisioned as a random
  // UUID (never derived from the OS username, hostname or home directory —
  // those would leak host identity into config artifacts) and persisted back
  // to config.yaml by the bootstrap path so it stays stable across restarts.
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
  // Data retention. Each *Days field is optional; unset means unlimited (the
  // historical behavior — no automatic aging). When set, a periodic sweeper
  // deletes items older than the window. Session deletion reuses the complete
  // deletion path (usage/memory/embeddings/uploads) so sweeping leaves no
  // orphans. Disable the whole sweeper with enabled:false or the env kill-switch.
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
  agent: z
    .object({
      autonomy: z
        .enum(['readonly', 'accept-edits', 'workspace-write', 'supervised', 'autonomous'])
        .default('supervised'),
      thinkingLevel: z.enum(['auto', 'off', 'low', 'medium', 'high', 'max']).default('medium'),
      defaultProvider: z.string().optional(),
      defaultModel: z.string().optional(),
      auxModel: z.string().trim().min(1).optional(),
      mode: z.string().trim().min(1).default('instant'),
      persona: z.string().trim().min(1).optional(),
      // Output language for assistant replies. 'auto' (default) mirrors the
      // user's language per turn; a specific value (e.g. 'en', 'ko', 'English',
      // '日本語') instructs the model to always answer in that language. Personas
      // no longer hardcode Korean — language is governed here so non-Korean
      // users are not forced into Korean output.
      outputLanguage: z.string().trim().min(1).default('auto'),
      intentRouter: z
        .object({
          // Enabled by default so natural-language chat turns can select
          // mode/persona/skills through the LLM router. Per-turn callers can
          // still bypass it with `intentRouting: { enabled: false }`.
          enabled: z.boolean().default(true),
          provider: z.string().optional(),
          model: z.string().optional(),
          // 4s only fits a co-located classifier. When the router shares the
          // main model — the default when `model` is unset — a hosted or cloud
          // endpoint routinely needs longer, and every timeout silently
          // downgrades the turn to a low-confidence fallback decision that
          // picks the wrong mode. 12s still bounds the added latency while
          // letting a normal remote classification finish.
          timeoutMs: z.number().int().min(500).max(60_000).default(12_000),
          maxPreviousMessages: z.number().int().min(0).max(20).default(4),
          perMessageCharLimit: z.number().int().min(50).max(2000).default(200),
          reasonMaxChars: z.number().int().min(50).max(2000).default(500),
        })
        .default({
          enabled: true,
          // Keep the nested-object default identical to timeoutMs.default().
          // Zod applies this object wholesale when `agent` exists but
          // `intentRouter` is omitted, so a stale value here silently bypasses
          // the field-level default.
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
      // Daily / per-session USD spend caps checked before every LLM dispatch.
      // undefined (default) means unlimited. Only priced (costKnown) spend
      // counts, so local models never trip the cap. env
      // SEPILOTD_DAILY_USD_BUDGET / SEPILOTD_SESSION_USD_BUDGET override.
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
      // Which `memory.*` tools direct-API surfaces (CLI/web/desktop) expose to
      // the agent. The full set (~40 tools) is the channel/auto-memory concept;
      // direct API defaults to `lean` (remember/list/search/forget) so one-shot
      // CLI work isn't paying ~8k tokens of memory-tool schemas. Channels are
      // unaffected — they always get the full registry.
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
          enabled: z.boolean().default(false),
          includeFailures: z.boolean().default(true),
        })
        .default({ enabled: false, includeFailures: true }),
    })
    .optional(),
  voice: voiceConfigSchema,
  scheduler: schedulerConfigSchema.optional(),
  skills: z
    .object({
      // When true, auto-discovered project/home skills (e.g. `.claude/skills`)
      // are treated as trusted (no untrusted-provenance banner). Default false:
      // they still load but are flagged untrusted supply-chain content.
      trustProjectSkills: z.boolean().default(false),
    })
    .default({ trustProjectSkills: false }),
  files: z
    .object({
      // Absolute directory roots that `/files/search` may enumerate. When empty
      // (default) the search is contained to the daemon user's home directory so
      // an arbitrary `cwd` cannot enumerate the whole filesystem layout.
      searchRoots: z.array(z.string().trim().min(1)).default([]),
    })
    .default({ searchRoots: [] }),
  security: z
    .object({
      toolPolicy: z.string().default('policies.yaml'),
      auditLog: z.boolean().default(true),
      sandbox: z.enum(['local', 'docker', 'bubblewrap']).default('local'),
      // Docker/bubblewrap isolate terminal.run. Keep built-in host-side file
      // tools out of those profiles by default so they cannot silently pierce the
      // configured filesystem boundary. `allow` is an explicit compatibility
      // escape hatch for deployments that still need the legacy behavior.
      sandboxHostFileTools: z.enum(['deny', 'allow']).optional(),
      sandboxDocker: sandboxDockerConfigSchema.optional(),
      sandboxBubblewrap: sandboxBubblewrapConfigSchema.optional(),
      egressAllowlist: z.array(z.string().trim().min(1)).optional(),
      skillSources: skillSourceSecuritySchema,
      webhooks: webhookSecurityPolicyConfigSchema.default(DEFAULT_WEBHOOK_SECURITY_POLICY),
    })
    .default({
      toolPolicy: 'policies.yaml',
      auditLog: true,
      sandbox: 'local',
      sandboxDocker: sandboxDockerConfigSchema.parse({}),
      skillSources: skillSourceSecuritySchema.parse({}),
      webhooks: DEFAULT_WEBHOOK_SECURITY_POLICY,
    }),
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
