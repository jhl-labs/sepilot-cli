import { homedir } from 'node:os'

/**
 * Render an absolute path with the user's home directory replaced
 * by `~`. Keeps logs reading naturally without exposing whatever
 * username the daemon happens to run as. The original (absolute)
 * path is what hits the filesystem; only display strings should
 * pass through this helper.
 */
export function tildify(path: string): string {
  const home = homedir()
  if (!home || !path.startsWith(home)) return path
  // Avoid replacing /home/foo with ~ when the path is /home/foobar.
  const next = path.charAt(home.length)
  if (next !== '' && next !== '/') return path
  return `~${path.slice(home.length)}`
}
