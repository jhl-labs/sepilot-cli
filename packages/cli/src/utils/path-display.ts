import { homedir } from 'node:os'

export function tildify(path: string): string {
  const home = homedir()
  if (!home || !path.startsWith(home)) return path
  const next = path.charAt(home.length)
  if (next !== '' && next !== '/') return path
  return `~${path.slice(home.length)}`
}
