import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import { publishQuickInput } from '../../quickinput/bus.js'
import { readConfigYaml, writeConfigYamlAtomic } from '../../config/yaml-store.js'
import { openJsonWatch } from './watch-sse.js'

// A "Quick Question" is a user-defined global hotkey that, when
// pressed, opens the main window and starts a fresh chat with a
// pre-filled prompt + the current clipboard contents combined into a
// single user message. Mirrors the sepilot_desktop-private pattern:
// `${prompt}\n\n${clipboard}` for fast translate / summarize / review
// flows from outside the app. Capped at 16 entries to keep the global
// shortcut budget predictable.
const QuickQuestionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1).max(80),
  prompt: z.string().min(1).max(4000),
  shortcut: z.string().min(1),
  enabled: z.boolean(),
})

const QuickInputSettingsSchema = z.object({
  hotkey: z.string().min(1),
  prefix: z.string(),
  quickQuestions: z.array(QuickQuestionSchema).max(16).default([]),
})

type QuickInputSettings = z.infer<typeof QuickInputSettingsSchema>
type QuickInputWatchPayload =
  | { type: 'snapshot'; settings: QuickInputSettings }
  | { type: 'heartbeat'; timestamp: string }

const QuickInputDefaults: QuickInputSettings = {
  hotkey: 'CommandOrControl+Alt+Space',
  prefix: '',
  quickQuestions: [],
}

function readQuickInputSettings(): QuickInputSettings {
  const cfg = readConfigYaml()
  const raw = (cfg.quickInput as Record<string, unknown> | undefined) ?? {}
  const parsed = QuickInputSettingsSchema.safeParse({
    ...QuickInputDefaults,
    ...raw,
  })
  return parsed.success ? parsed.data : QuickInputDefaults
}

async function writeQuickInputSettings(next: QuickInputSettings): Promise<void> {
  const cfg = readConfigYaml()
  cfg.quickInput = next
  await writeConfigYamlAtomic(cfg)
}

export async function registerQuickInputRoutes(app: FastifyInstance): Promise<void> {
  const watchSubscribers = new Set<(payload: QuickInputWatchPayload) => void>()

  function buildWatchSnapshot(): QuickInputWatchPayload {
    return {
      type: 'snapshot',
      settings: readQuickInputSettings(),
    }
  }

  function publishWatchSnapshot(): void {
    if (watchSubscribers.size === 0) return
    const payload = buildWatchSnapshot()
    for (const subscriber of watchSubscribers) subscriber(payload)
  }

  await bindCapability(
    app,
    {
      name: 'quick-input',
      version: '1',
      methods: [
        { method: 'GET', path: '/quick-input/settings' },
        { method: 'GET', path: '/quick-input/settings/watch' },
        { method: 'PUT', path: '/quick-input/settings' },
        { method: 'POST', path: '/quick-input/publish' },
      ],
    },
    async (a) => {
      a.get('/quick-input/settings', async () => readQuickInputSettings())
      a.get('/quick-input/settings/watch', async (req, reply) => {
        openJsonWatch(req, reply, {
          eventName: 'quick-input',
          subscribers: watchSubscribers,
          buildSnapshot: buildWatchSnapshot,
          buildHeartbeat: () => ({ type: 'heartbeat' as const, timestamp: new Date().toISOString() }),
        })
      })
      a.put('/quick-input/settings', async (req, reply) => {
        const parsed = QuickInputSettingsSchema.safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        await writeQuickInputSettings(parsed.data)
        const settings = readQuickInputSettings()
        publishWatchSnapshot()
        return settings
      })
      a.post('/quick-input/publish', async (req, reply) => {
        const parsed = z.object({ text: z.string().min(1) }).safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        publishQuickInput(parsed.data.text)
        return { ok: true }
      })
    },
  )
}
