import { homedir, platform } from 'node:os'
import { existsSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import type { SourceManifest } from './types.js'

export function candidateSourcePaths(): string[] {
  const home = homedir()
  switch (platform()) {
    case 'darwin':
      return [join(home, 'Library/Application Support/sepilot-desktop')]
    case 'win32':
      return [
        join(
          process.env.APPDATA ?? join(home, 'AppData/Roaming'),
          'sepilot-desktop',
        ),
      ]
    default:
      return [join(home, '.config/sepilot-desktop')]
  }
}

export function detectSource(explicit?: string): SourceManifest | null {
  const targets = explicit ? [explicit] : candidateSourcePaths()
  for (const t of targets) {
    if (!existsSync(t) || !statSync(t).isDirectory()) continue
    const dirs = new Set(readdirSync(t))
    return {
      root: t,
      version: 1,
      features: {
        conversations: dirs.has('conversations'),
        rag: dirs.has('knowledge'),
        wiki: dirs.has('wiki'),
        persona: dirs.has('personas'),
        snippets: dirs.has('snippets'),
        prompts: dirs.has('prompts'),
        personalDocs: dirs.has('docs'),
        skills: dirs.has('skills'),
        extensions: dirs.has('extensions'),
        settings:
          dirs.has('settings.json') ||
          existsSync(join(t, 'settings.json')),
      },
    }
  }
  return null
}
