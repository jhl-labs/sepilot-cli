import { mkdtemp, readFile, rm, readdir, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve, relative, sep } from 'node:path'
import type { SkillSource, SkillRef, FetchedSkill } from './types.js'
import { gitCompatibleSkillRoots } from '../compat.js'
import { parseSkillMd } from '../loader.js'
import { SkillFetchError, SkillPathTraversalError } from '../errors.js'
import type { SkillSourceUrlPolicy } from '../source-url-policy.js'
import { checkoutDetachedGitRef } from './git-ref.js'
import { prepareSafeGitTransport, type SafeGitRuntime } from './safe-git.js'

export interface GitSourceOptions extends Pick<SafeGitRuntime, 'resolveUrl'> {
  cloneDir?: string | null
  urlPolicy?: SkillSourceUrlPolicy
}

export class GitSource implements SkillSource {
  constructor(private opts: GitSourceOptions = {}) {}

  async fetch(ref: SkillRef): Promise<FetchedSkill[]> {
    if (ref.type !== 'git') throw new SkillFetchError(`GitSource cannot handle ${ref.type}`)
    const transport = await prepareSafeGitTransport(ref.repo, this.opts)
    const targetDir = this.opts.cloneDir ?? (await mkdtemp(join(tmpdir(), 'sepilotd-gs-')))
    const manageDir = this.opts.cloneDir == null
    try {
      const git = transport.createGit()
      await git.clone(transport.source, targetDir, ['--depth', '1', '--no-tags'])
      const repoGit = transport.createGit(targetDir)
      if (ref.branch) {
        await checkoutDetachedGitRef(repoGit, ref.branch)
      }
      const sourceRef = (await repoGit.revparse(['HEAD'])).trim()

      const scanRoots = await this.resolveScanRoots(targetDir, ref.path)
      const fetched: FetchedSkill[] = []
      const seen = new Set<string>()
      for (const scanRoot of scanRoots) {
        const skillDirs = await this.findSkillDirs(scanRoot)
        for (const dir of skillDirs) {
          const skillPath = join(dir, 'SKILL.md')
          const raw = await readFile(skillPath, 'utf-8')
          const fallbackId = relative(scanRoot, dir).split(sep).pop() ?? 'skill'
          const parsed = parseSkillMd(raw, fallbackId)
          if (seen.has(parsed.metadata.id)) continue
          seen.add(parsed.metadata.id)
          fetched.push({
            metadata: parsed.metadata,
            content: parsed.content,
            source: { type: 'git', ref: sourceRecordRef(ref) },
            sourceRef,
          })
        }
      }
      return fetched
    } catch (err) {
      if (err instanceof SkillFetchError) throw err
      if (err instanceof SkillPathTraversalError) throw err
      throw new SkillFetchError(`git fetch failed: ${(err as Error).message}`, err)
    } finally {
      if (manageDir) {
        await rm(targetDir, { recursive: true, force: true })
      }
    }
  }

  private async resolveScanRoots(root: string, sub: string | null): Promise<string[]> {
    if (!sub) {
      const candidates = await gitCompatibleSkillRoots(root)
      const roots: string[] = []
      for (const candidate of candidates) {
        try {
          const s = await stat(candidate)
          if (s.isDirectory()) roots.push(candidate)
        } catch { /* fall through */ }
      }
      return roots.length ? roots : [root]
    }
    const resolved = resolve(root, sub)
    const rel = relative(root, resolved)
    if (rel.startsWith('..') || rel.startsWith(`..${sep}`) || rel === '') {
      throw new SkillPathTraversalError(sub)
    }
    return [resolved]
  }

  private async findSkillDirs(root: string): Promise<string[]> {
    const out: string[] = []
    const rootResolved = resolve(root)
    const walk = async (dir: string) => {
      let entries
      try {
        entries = await readdir(dir, { withFileTypes: true })
      } catch {
        return
      }
      for (const e of entries) {
        if (e.isSymbolicLink()) continue // Refuse to follow symlinks
        const full = join(dir, e.name)
        const rel = relative(rootResolved, resolve(full))
        if (rel.startsWith('..') || rel.startsWith(`..${sep}`)) continue // Outside root
        if (e.isDirectory()) {
          await walk(full)
        } else if (e.isFile() && e.name === 'SKILL.md') {
          out.push(dir)
        }
      }
    }
    await walk(root)
    return out
  }
}

function sourceRecordRef(ref: Extract<SkillRef, { type: 'git' }>): string {
  if (ref.branch && ref.path && /^https:\/\/github\.com\/[^/]+\/[^/]+\/?$/i.test(ref.repo)) {
    return `${ref.repo.replace(/\/$/, '')}/tree/${ref.branch}/${ref.path}`
  }
  return ref.repo
}
