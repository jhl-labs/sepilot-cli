/**
 * Single source of truth for build-time opt-in features. Consumed by the
 * bundle build script (validation + manifest generation + binary exclusion
 * scan) and by tests. Runtime code never reads this to alter behavior —
 * exclusion is purely a build-time decision (traceless at runtime).
 */
export interface FeatureDescriptor {
  id: string
  description: string
  /** Always included; disabling fails the build. */
  locked?: boolean
  /** Feature ids that must stay enabled while this one is enabled. */
  requires?: string[]
  /**
   * Distinctive strings guaranteed to appear in this feature's compiled
   * code (tool names, route paths). Used to prove exclusion: when the
   * feature is disabled none of these may appear in the final binary.
   */
  binaryMarkers: string[]
}

const CHANNEL_IDS = [
  'telegram',
  'slack',
  'discord',
  'mattermost',
  'teams',
  'line',
  'whatsapp',
  'webchat',
  'webhook',
  'github-issue',
] as const

/**
 * Each channel factory's `registerChannelFactory` in
 * `features/entries/channel-<name>.ts` returns this literal `skipReason`
 * string when required config is missing. Unlike the channel class name
 * (imported unconditionally by shared webhook route plumbing for several
 * channels) or a `sepilot-channel-*` placeholder (never existed in source),
 * this string is confined to that channel's own entries file.
 */
const CHANNEL_SKIP_REASONS: Record<(typeof CHANNEL_IDS)[number], string> = {
  telegram: 'botToken is required',
  slack: 'botToken and signingSecret are required',
  discord: 'botToken, applicationId, and a valid publicKey are required',
  mattermost: 'serverUrl, botToken, and webhookToken are required',
  teams: "registry.register('teams', (channel) => {",
  line: 'channelAccessToken and channelSecret are required',
  whatsapp: 'phoneNumberId, accessToken, verifyToken, and appSecret are required',
  webchat: "registry.register('webchat', () => ({",
  webhook: 'at least one endpoint is required',
  'github-issue': 'github-issue-processing.db',
}

export const FEATURE_CATALOG: FeatureDescriptor[] = [
  {
    id: 'browser',
    description: 'Playwright browser automation tools',
    binaryMarkers: [
      'real headless Chromium browser and return rendered text',
      'DOM/canvas layout audit plus browser console/page audit',
    ],
  },
  {
    id: 'computer-use',
    description: 'Desktop automation tools',
    binaryMarkers: [
      'top-level Windows application windows',
      'Move the Windows mouse pointer to absolute screen pixel',
    ],
  },
  {
    id: 'micro-apps',
    description: 'apps.* micro app tools and routes',
    binaryMarkers: [
      'Apply schema-aware mutations to one AI-writable sepilot desktop micro app',
      '/apps/:id/mutate',
    ],
  },
  ...CHANNEL_IDS.map((channel) => ({
    id: `channels.${channel}`,
    description: `${channel} channel integration`,
    // Each channel factory's own `skipReason` string (returned when required
    // config is missing) is a literal confined to that channel's
    // `features/entries/channel-<name>.ts` and nowhere else in the daemon —
    // unlike the channel class name or a `sepilot-channel-*` placeholder,
    // which either leak into shared webhook route plumbing or never existed
    // in source at all.
    binaryMarkers: [CHANNEL_SKIP_REASONS[channel]],
  })),
  {
    id: 'voice',
    description: 'Whisper/Piper voice routes and speech tools',
    requires: ['media'],
    // NOTE: `server/routes/voice.ts` (and everything it imports, including
    // `media/speech.ts`) is a bad marker source — `server/app.ts` statically
    // imports `voiceOpenApiComponents`/`voiceOpenApiOverrides` from that file
    // for OpenAPI docs regardless of the `voice` feature flag, and bun's
    // bundler does not tree-shake unused exports out of an already-imported
    // module, so any literal there (route paths, `media/speech.ts`'s own
    // "locally-installed `piper` CLI" comment, etc.) leaks into the binary
    // even when voice is disabled. This marker instead comes from the
    // `media.transcribe`/`media.speak` tool descriptions in `tools/media.ts`,
    // which is exclusively imported by `features/entries/voice.ts` — its
    // wording ("locally-installed whisper CLI") intentionally does not match
    // `media/speech.ts`'s own comment text ("locally-installed
    // `openai-whisper`"), so it stays confined to the voice-gated import.
    binaryMarkers: ['locally-installed whisper CLI'],
  },
  {
    id: 'swarm',
    description: 'Multi-agent swarm orchestration',
    binaryMarkers: ['/api/v1/swarm/runs'],
  },
  {
    id: 'pages',
    description: 'GitHub Pages studio tools and routes',
    binaryMarkers: ['/pages/scaffold', '/pages/scan'],
  },
  {
    id: 'image-gen',
    description: 'Image generation tools',
    binaryMarkers: ['List sepilotd image generation providers available to the daemon'],
  },
  { id: 'market', description: 'Market quote tool', binaryMarkers: ['Ticker, 6-digit KRX code'] },
  {
    id: 'media',
    description: 'Media inspect/extract-text tools',
    binaryMarkers: [
      'Inspect a file and report whether it is text, PDF, OCR-capable image, or generic binary',
      'Extract readable text from text files, PDFs, and OCR-capable images',
    ],
  },
  {
    id: 'office',
    description: 'Office document tools',
    binaryMarkers: ['Preview a live edit against the active Office document'],
  },
  {
    id: 'mcp',
    description: 'Model Context Protocol client + routes',
    binaryMarkers: ['/mcp/marketplaces'],
  },
  {
    id: 'delegation',
    description: 'Cross-device delegation tool',
    binaryMarkers: ['Delegate a task to another device on the local network'],
  },
  {
    id: 'extensions',
    description: 'Daemon extension system',
    binaryMarkers: ['extensions-capability'],
  },
  {
    id: 'a2a',
    description: 'Agent-to-Agent protocol server',
    binaryMarkers: ["app.post('/api/v1/a2a'"],
  },
  {
    id: 'acp',
    description: 'External ACP agent runner',
    // NOTE: `/acp/agents/run` (the route path) is a bad marker — it also appears
    // in `packages/api-client`'s daemon HTTP client, which the standalone binary
    // always bundles, so it can never prove exclusion. This marker instead comes
    // from the `external_acp.run` tool description in `tools/external-acp-run.ts`,
    // which is imported exclusively by `features/entries/acp-runtime.ts` and is
    // absent from the client, so it stays confined to the acp-gated import.
    binaryMarkers: ['Run a configured external ACP coding agent such as opencode or Codex'],
  },
  {
    id: 'notebook',
    description: 'Jupyter notebook inspection tool',
    binaryMarkers: ['Inspect a Jupyter notebook and summarize its cells'],
  },
  {
    id: 'lsp',
    description: 'LSP-backed code intelligence tools',
    binaryMarkers: [
      'Search for symbol declarations or references across the workspace',
      'Return the latest LSP diagnostics for a file URI',
    ],
  },
  { id: 'wiki', description: 'Wiki subsystem', binaryMarkers: ['wiki_nodes'] },
  {
    id: 'plugins',
    description: 'Daemon plugin loader + routes',
    // The `List plugins` summary lives in the plugin OpenAPI override
    // (`server/routes/plugins-openapi.ts`). `server/app.ts` no longer imports it
    // statically — it is routed through the generated manifest
    // (`getPluginsOpenApiOverrides`) so both the route handler and its OpenAPI
    // docs drop from the binary when plugins are disabled.
    binaryMarkers: ['List plugins'],
  },
  {
    id: 'memory',
    description: 'Memory subsystem (core)',
    locked: true,
    binaryMarkers: ['memory.remember'],
  },
  {
    id: 'scheduler',
    description: 'Scheduler/cron (core)',
    locked: true,
    binaryMarkers: ['schedule_create'],
  },
  {
    id: 'skills',
    description: 'Skills subsystem (core)',
    locked: true,
    binaryMarkers: ['skillhub.search'],
  },
]

const CATALOG_BY_ID = new Map(FEATURE_CATALOG.map((feature) => [feature.id, feature]))

export function validateFeatureSelection(disabled: Set<string>): string[] {
  const errors: string[] = []
  for (const id of disabled) {
    const feature = CATALOG_BY_ID.get(id)
    if (!feature) {
      errors.push(`unknown feature '${id}' — known ids: ${[...CATALOG_BY_ID.keys()].join(', ')}`)
      continue
    }
    if (feature.locked) {
      errors.push(`feature '${id}' is locked (core-entangled) and cannot be disabled in this build`)
    }
  }
  for (const feature of FEATURE_CATALOG) {
    if (disabled.has(feature.id)) continue
    for (const dep of feature.requires ?? []) {
      if (disabled.has(dep)) {
        errors.push(
          `feature '${feature.id}' requires '${dep}' — disable '${feature.id}' too, or keep '${dep}' enabled`,
        )
      }
    }
  }
  return errors
}
