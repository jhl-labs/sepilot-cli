import { access, mkdir, symlink } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

const BORROWED_CODEX_HOME_ENTRIES = ['auth.json', 'config.toml'] as const

export function sourceCodexHome(env: NodeJS.ProcessEnv): string {
  return env.CODEX_HOME?.trim() || join(homedir(), '.codex')
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

export async function prepareCodexProviderHome(
  env: NodeJS.ProcessEnv,
  privateHome: string,
): Promise<NodeJS.ProcessEnv> {
  const source = sourceCodexHome(env)
  if (!(await exists(join(source, 'auth.json')))) return env
  try {
    await mkdir(privateHome, { recursive: true, mode: 0o700 })
    for (const entry of BORROWED_CODEX_HOME_ENTRIES) {
      const target = join(source, entry)
      if (await exists(target)) await symlink(target, join(privateHome, entry))
    }
  } catch {
    return env
  }
  return { ...env, CODEX_HOME: privateHome }
}
