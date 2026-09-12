import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { join } from 'node:path'
import yaml from 'yaml'
import { sepilotdHome } from '../storage/home.js'
import { secureFile } from '../utils/secure-file.js'

function configPath(): string {
  return join(sepilotdHome(), 'config.yaml')
}

export function readConfigYaml(): Record<string, unknown> {
  const p = configPath()
  if (!existsSync(p)) return {}
  const text = readFileSync(p, 'utf-8')
  const parsed = yaml.parse(text)
  return parsed && typeof parsed === 'object'
    ? (parsed as Record<string, unknown>)
    : {}
}

export async function writeConfigYamlAtomic(
  next: Record<string, unknown>,
): Promise<void> {
  const home = sepilotdHome()
  mkdirSync(home, { recursive: true })
  const p = configPath()
  const tmp = `${p}.tmp`
  let serialized: string
  try {
    serialized = yaml.stringify(next)
  } catch (err) {
    if (existsSync(tmp)) unlinkSync(tmp)
    throw err
  }
  if (existsSync(tmp)) unlinkSync(tmp)
  writeFileSync(tmp, serialized, { encoding: 'utf-8', mode: 0o600, flag: 'wx' })
  renameSync(tmp, p)
  secureFile(p)
}
