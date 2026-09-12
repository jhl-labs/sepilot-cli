import type { FastifyInstance } from 'fastify'
import {
  existsSync,
  readFileSync,
  readdirSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { join } from 'node:path'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import { sepilotdHome } from '../../storage/home.js'

export interface SkillItem {
  name: string
  path: string
  enabled: boolean
  description: string | null
}

function readSkillFront(file: string): { description: string | null } {
  const text = readFileSync(file, 'utf-8').slice(0, 2_000)
  const fm = text.match(/^---\n([\s\S]*?)\n---/)
  if (!fm?.[1]) return { description: null }
  const body = fm[1]
  const desc = body.match(/^description:\s*(.+)$/m)
  return { description: desc?.[1]?.trim() ?? null }
}

function listSkills(): SkillItem[] {
  const root = join(sepilotdHome(), 'skills')
  if (!existsSync(root)) return []
  const out: SkillItem[] = []
  for (const name of readdirSync(root)) {
    const dir = join(root, name)
    if (!statSync(dir).isDirectory()) continue
    const md = join(dir, 'SKILL.md')
    if (!existsSync(md)) continue
    const { description } = readSkillFront(md)
    const disabledMarker = join(dir, '.disabled')
    out.push({
      name,
      path: md,
      enabled: !existsSync(disabledMarker),
      description,
    })
  }
  return out
}

function setEnabled(name: string, enabled: boolean): void {
  const dir = join(sepilotdHome(), 'skills', name)
  if (!existsSync(dir)) return
  const marker = join(dir, '.disabled')
  if (enabled) {
    if (existsSync(marker)) unlinkSync(marker)
  } else if (!existsSync(marker)) {
    writeFileSync(marker, '', 'utf-8')
  }
}

export async function registerSkillsCapabilityRoutes(
  app: FastifyInstance,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'skills',
      version: '1',
      methods: [
        { method: 'GET', path: '/skills' },
        { method: 'POST', path: '/skills/:name/enable' },
        { method: 'POST', path: '/skills/:name/disable' },
      ],
    },
    async (a) => {
      a.get('/skills', async () => listSkills())
      a.post('/skills/:name/enable', async (req) => {
        const { name } = z
          .object({ name: z.string().min(1) })
          .parse(req.params)
        setEnabled(name, true)
        return { ok: true }
      })
      a.post('/skills/:name/disable', async (req) => {
        const { name } = z
          .object({ name: z.string().min(1) })
          .parse(req.params)
        setEnabled(name, false)
        return { ok: true }
      })
    },
  )
}
