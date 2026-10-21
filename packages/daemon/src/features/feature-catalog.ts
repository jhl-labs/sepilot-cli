export interface FeatureDescriptor {
  id: string
  description: string
  locked?: boolean
  requires?: string[]
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
    binaryMarkers: [CHANNEL_SKIP_REASONS[channel]],
  })),
  {
    id: 'voice',
    description: 'Whisper/Piper voice routes and speech tools',
    requires: ['media'],
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
    binaryMarkers: ['List plugins'],
  },
  {
    id: 'event-routing',
    description: 'Event routing service, management routes and LLM rule runner',
    binaryMarkers: ['CREATE TABLE IF NOT EXISTS event_delivery_outbox'],
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
