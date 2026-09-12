import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { readFile, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

const execFileAsync = promisify(execFile)

export interface ManagedContainer {
  id: string
  name: string
  image: string
  state: string
  status: string
  createdAt: string
  purpose?: string
  session?: string
  workdir?: string
  mode?: string
}

export interface ContainerLogEntry {
  id?: string
  name: string
  image?: string
  purpose?: string
  sessionId?: string
  createdAt?: string
  workdir?: string
  mode?: string
}

const MANAGED_FILTER = 'label=sepilot.managed=true'
const LOG_PATH = join(homedir(), '.sepilotd', 'containers.jsonl')

function parseLabels(raw: string): Record<string, string> {
  const out: Record<string, string> = {}
  for (const pair of raw.split(',')) {
    const i = pair.indexOf('=')
    if (i > 0) out[pair.slice(0, i)] = pair.slice(i + 1)
  }
  return out
}

export function parseDockerPsJsonLines(stdout: string): ManagedContainer[] {
  const out: ManagedContainer[] = []
  for (const line of stdout.split('\n')) {
    const t = line.trim()
    if (!t) continue
    let obj: Record<string, unknown>
    try {
      obj = JSON.parse(t) as Record<string, unknown>
    } catch {
      continue
    }
    const labels = parseLabels(String(obj.Labels ?? ''))
    out.push({
      id: String(obj.ID ?? obj.Id ?? ''),
      name: String(obj.Names ?? obj.Name ?? ''),
      image: String(obj.Image ?? ''),
      state: String(obj.State ?? '').toLowerCase() || 'unknown',
      status: String(obj.Status ?? ''),
      createdAt: String(obj.CreatedAt ?? ''),
      purpose: labels['sepilot.purpose'],
      session: labels['sepilot.session'],
    })
  }
  return out
}

export function mergeContainerMetadata(
  live: ManagedContainer[],
  log: ContainerLogEntry[],
): ManagedContainer[] {
  const byName = new Map(log.map((e) => [e.name, e]))
  return live.map((c) => {
    const e = byName.get(c.name)
    if (!e) return c
    return { ...c, purpose: e.purpose ?? c.purpose, workdir: e.workdir, mode: e.mode }
  })
}

export async function dockerAvailable(): Promise<boolean> {
  try {
    await execFileAsync('docker', ['info'], { timeout: 2000 })
    return true
  } catch {
    return false
  }
}

export async function listManagedContainers(): Promise<ManagedContainer[]> {
  const { stdout } = await execFileAsync(
    'docker',
    ['ps', '-a', '--filter', MANAGED_FILTER, '--format', '{{json .}}'],
    { timeout: 3000 },
  )
  return mergeContainerMetadata(parseDockerPsJsonLines(stdout), await readContainerLog())
}

export async function removeContainers(idsOrNames: string[]): Promise<void> {
  if (idsOrNames.length === 0) return
  await execFileAsync('docker', ['rm', '-f', ...idsOrNames], { timeout: 30000 })
}

export async function readContainerLog(): Promise<ContainerLogEntry[]> {
  let raw: string
  try {
    raw = await readFile(LOG_PATH, 'utf-8')
  } catch {
    return []
  }
  const out: ContainerLogEntry[] = []
  for (const line of raw.split('\n')) {
    const t = line.trim()
    if (!t) continue
    try {
      out.push(JSON.parse(t) as ContainerLogEntry)
    } catch {
      // skip malformed lines
    }
  }
  return out
}

export async function pruneContainerLog(removedNames: Set<string>): Promise<void> {
  const entries = (await readContainerLog()).filter((e) => !removedNames.has(e.name))
  await writeFile(
    LOG_PATH,
    entries.length ? `${entries.map((e) => JSON.stringify(e)).join('\n')}\n` : '',
    'utf-8',
  )
}
