import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { randomUUID } from 'node:crypto'
import { mkdir, readFile, writeFile, readdir, lstat, realpath } from 'node:fs/promises'
import { join } from 'node:path'
import { z } from 'zod'
import { sepilotdHome } from '../storage/home.js'
import { openDomainDb } from '../storage/domain-db.js'
import { createKnowledgeActivityStore } from '../knowledge/activity.js'
import { decode, digest, encode, keyOf, plan, type Change, type Entry } from './model.js'

const exec = promisify(execFile)
export const Connection = z
  .object({
    source: z
      .string()
      .trim()
      .min(1)
      .max(2000)
      .refine((s) => !s.startsWith('-') && !/[\r\n\0]/u.test(s), 'Invalid Git repository'),
    branch: z
      .string()
      .regex(/^[a-zA-Z0-9][a-zA-Z0-9/_.-]*$/u)
      .max(200)
      .refine((s) => !s.includes('..') && !s.endsWith('.lock')),
    docs: z.boolean(),
    wiki: z.boolean(),
  })
  .strict()
  .refine((c) => c.docs || c.wiki, 'Select Docs or Wiki')
type Config = z.infer<typeof Connection> & { checkout: string }
interface Store {
  snapshot(): Record<string, string>
  apply(entries: Entry[], expected?: Record<string, string>): void
}
interface Preview {
  id: string
  changes: Change[]
  head: string
  local: Record<string, string>
  remote: Record<string, string>
}
class SyncError extends Error {
  readonly statusCode = 400
}
const rootName = '.sepilot-docs'

export function createDocsGitService(store: Store, changed: () => void) {
  const db = openDomainDb({ name: 'docs-git' })
  db.exec('CREATE TABLE IF NOT EXISTS sync_state (id INTEGER PRIMARY KEY, data TEXT NOT NULL)')
  const saved = db.prepare('SELECT data FROM sync_state WHERE id=1').get() as
    | { data: string }
    | undefined
  let state: {
    connection: Config | null
    base: Record<string, string>
    lastSync: number | null
    error: string | null
  } = saved ? JSON.parse(saved.data) : { connection: null, base: {}, lastSync: null, error: null }
  let busy = false
  let preview: Preview | null = null
  const activity = createKnowledgeActivityStore()
  const persist = () =>
    db.prepare('INSERT OR REPLACE INTO sync_state VALUES (1, ?)').run(JSON.stringify(state))
  const selected = (key: string, c: Config) => (key.startsWith('docs/') ? c.docs : c.wiki)
  async function git(cwd: string, args: string[]) {
    const result = await exec(
      'git',
      [
        '-c',
        'protocol.ext.allow=never',
        '-c',
        'core.hooksPath=/dev/null',
        '-c',
        'commit.gpgSign=false',
        ...args,
      ],
      {
        cwd,
        timeout: 30000,
        maxBuffer: 5 * 1024 * 1024,
        env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_SSH_COMMAND: 'ssh -oBatchMode=yes' },
      },
    )
    return result.stdout.trim()
  }
  async function exclusive<T>(summary: string, fn: (activityId: string) => Promise<T>): Promise<T> {
    if (busy)
      throw Object.assign(new SyncError('Git 동기화 작업이 진행 중입니다.'), { statusCode: 409 })
    busy = true
    const id = activity.begin({ kind: 'git-sync', summary })
    try {
      const result = await fn(id)
      state.error = null
      persist()
      activity.finish(id, { preview: { lastSync: state.lastSync } })
      return result
    } catch (error) {
      // Git stderr can contain credential-bearing remote URLs. Keep it out of UI/audit.
      state.error =
        error instanceof SyncError
          ? error.message
          : 'Git 동기화에 실패했습니다. 인증, 브랜치, 충돌 또는 파일 형식을 확인하세요. 기존 데이터는 삭제하지 않습니다.'
      persist()
      activity.fail(id, new SyncError(state.error))
      throw new SyncError(state.error)
    } finally {
      busy = false
    }
  }
  const config = () => {
    if (!state.connection) throw new SyncError('Git 저장소를 먼저 연결하세요.')
    return state.connection
  }
  async function refresh(c: Config) {
    if (await git(c.checkout, ['status', '--porcelain']))
      throw new SyncError('Managed checkout has uncommitted changes')
    await git(c.checkout, ['fetch', 'origin', '--prune'])
    const remote = await git(c.checkout, [
      'for-each-ref',
      '--format=%(refname)',
      `refs/remotes/origin/${c.branch}`,
    ])
    if (remote) await git(c.checkout, ['merge', '--ff-only', `refs/remotes/origin/${c.branch}`])
  }
  async function files(c: Config): Promise<Record<string, string>> {
    const result: Record<string, string> = {}
    const root = join(c.checkout, rootName)
    try {
      if (!(await lstat(root)).isDirectory() || (await lstat(root)).isSymbolicLink())
        throw new SyncError('Unsafe sync directory')
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return result
      throw e
    }
    let bytes = 0
    for (const collection of ['docs', 'wiki']) {
      if (!selected(`${collection}/`, c)) continue
      const dir = join(root, collection)
      try {
        const stat = await lstat(dir)
        if (!stat.isDirectory() || stat.isSymbolicLink())
          throw new SyncError('Unsafe collection directory')
        for (const name of await readdir(dir)) {
          if (!/^[a-f0-9]{64}\.md$/u.test(name)) throw new SyncError('Unsupported sync filename')
          const file = join(dir, name),
            info = await lstat(file)
          if (!info.isFile() || info.isSymbolicLink() || info.size > 2 * 1024 * 1024)
            throw new SyncError('Unsafe sync file')
          bytes += info.size
          if (bytes > 50 * 1024 * 1024 || Object.keys(result).length >= 10000)
            throw new SyncError('Sync size limit')
          const entry = decode(await readFile(file, 'utf8'))
          const key = `${collection}/${name}`
          if (keyOf(entry) !== key) throw new SyncError('Record identity mismatch')
          result[key] = encode(entry)
        }
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e
      }
    }
    return result
  }
  const snapshot = (c: Config) =>
    Object.fromEntries(Object.entries(store.snapshot()).filter(([k]) => selected(k, c)))
  async function head(c: Config) {
    return git(c.checkout, ['rev-parse', '--verify', 'HEAD']).catch(() => '')
  }
  return {
    status: () => ({
      connection: state.connection && { ...state.connection, checkout: undefined },
      busy,
      lastSync: state.lastSync,
      error: state.error,
      preview: preview && { id: preview.id, changes: preview.changes },
    }),
    connect: (raw: unknown) =>
      exclusive('Docs·Wiki Git 저장소 연결', async () => {
        const input = Connection.parse(raw)
        // Credentials belong to Git's credential helper / SSH agent, never this config.
        if (/^[a-z]+:\/\//iu.test(input.source)) {
          const url = new URL(input.source)
          if (
            !['https:', 'ssh:'].includes(url.protocol) ||
            url.password ||
            (url.protocol === 'https:' && url.username) ||
            url.search ||
            url.hash
          )
            throw new SyncError('Use Git credentials outside the URL')
        }
        const parent = join(sepilotdHome(), 'docs-git', 'checkouts')
        await mkdir(parent, { recursive: true, mode: 0o700 })
        const checkout = join(await realpath(parent), randomUUID())
        await git(parent, ['clone', '--no-checkout', '--', input.source, checkout])
        const remote = await git(checkout, [
          'for-each-ref',
          '--format=%(refname)',
          `refs/remotes/origin/${input.branch}`,
        ])
        if (remote)
          await git(checkout, [
            'checkout',
            '-B',
            input.branch,
            `refs/remotes/origin/${input.branch}`,
          ])
        else {
          if (await git(checkout, ['for-each-ref', '--format=%(refname)', 'refs/remotes/origin']))
            throw new SyncError('Branch does not exist')
          await git(checkout, ['symbolic-ref', 'HEAD', `refs/heads/${input.branch}`])
        }
        state = { connection: { ...input, checkout }, base: {}, lastSync: null, error: null }
        preview = null
      }),
    disconnect: () =>
      exclusive('Git 연결 해제', async () => {
        state = { connection: null, base: {}, lastSync: null, error: null }
        preview = null
      }),
    preview: () =>
      exclusive('Git 변경 사항 비교', async () => {
        const c = config()
        await refresh(c)
        const local = snapshot(c),
          remote = await files(c)
        preview = {
          id: randomUUID(),
          changes: plan(local, remote, state.base),
          head: await head(c),
          local,
          remote,
        }
        return { id: preview.id, changes: preview.changes }
      }),
    sync: (id: string, resolutions: Record<string, 'local' | 'remote'>) =>
      exclusive('Docs·Wiki Git 동기화', async (activityId) => {
        const c = config(),
          p = preview
        if (!p || p.id !== id) throw new SyncError('Compare changes first')
        await refresh(c)
        const remoteNow = await files(c)
        const headNow = await head(c)
        if (
          headNow !== p.head ||
          JSON.stringify(snapshot(c)) !== JSON.stringify(p.local) ||
          JSON.stringify(remoteNow) !== JSON.stringify(p.remote)
        ) {
          preview = null
          throw new SyncError('Changes became stale; compare again')
        }
        const uploads: Record<string, string> = {},
          imports: Entry[] = []
        for (const change of p.changes) {
          const choice =
            change.action === 'conflict'
              ? resolutions[change.key]
              : change.action === 'upload'
                ? 'local'
                : 'remote'
          if (!choice) throw new SyncError('Resolve conflicts first')
          const text = choice === 'local' ? change.local : change.remote
          if (text === null)
            throw new SyncError(
              'Deletion is not propagated; restore the retained version or archive in Wiki',
            )
          if (choice === 'local') uploads[change.key] = text
          else imports.push(decode(text))
        }
        // Import first, before any await, while the local snapshot still matches the preview.
        // An interrupted push retains local commits and retries without force-pushing.
        activity.event(
          activityId,
          `Git → 앱 ${imports.length}개, 앱 → Git ${Object.keys(uploads).length}개; 충돌 선택 ${Object.keys(resolutions).length}개`,
        )
        store.apply(imports, p.local)
        changed()
        for (const [key, value] of Object.entries(uploads)) {
          const path = join(c.checkout, rootName, key)
          await mkdir(join(c.checkout, rootName, key.split('/')[0]!), { recursive: true })
          await writeFile(path, value, { mode: 0o600 })
        }
        if (Object.keys(uploads).length) {
          await git(c.checkout, [
            'add',
            '--',
            ...Object.keys(uploads).map((k) => `${rootName}/${k}`),
          ])
          if (await git(c.checkout, ['diff', '--cached', '--name-only'])) {
            await git(c.checkout, [
              '-c',
              'user.name=Sepilot',
              '-c',
              'user.email=sepilot@localhost',
              'commit',
              '-m',
              'Sync Docs and Wiki',
            ])
          }
        }
        if (await head(c)) await git(c.checkout, ['push', 'origin', `HEAD:refs/heads/${c.branch}`])
        const synced = snapshot(c),
          remote = await files(c)
        state.base = Object.fromEntries(
          Object.entries(synced)
            .filter(([k, v]) => remote[k] === v)
            .map(([k, v]) => [k, digest(v)!]),
        )
        activity.event(activityId, `동기화 완료: ${(await head(c)) || '빈 저장소'}`)
        state.lastSync = Date.now()
        preview = null
      }),
  }
}
