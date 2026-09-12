import { randomUUID } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { lstat, mkdir, writeFile, unlink, readdir, realpath, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { isAbsolute, join, extname, relative, resolve, sep } from 'node:path'
import type { ISessionStore } from '@sepilotd/core'
import type { FastifyInstance } from 'fastify'
import multipart from '@fastify/multipart'
import '../fastify-types.js'
import {
  FileExtractionError,
  fileToContentPart,
  getSupportedExtensions,
  getSupportedTextExtensions,
  readExtractableFileContent,
} from '../../media/pipeline.js'
import { fastifySchemaFromZod, getRuntimeDataDir, zodRequestValidation } from './utils.js'
import {
  getUploadedFile,
  persistUploadedFileMetadata,
  removeUploadedFilesByIds,
  restoreUploadedFiles,
  storeUploadedFile,
  sweepOrphanUploadedFiles,
  type UploadedFile,
} from './file-registry.js'
import {
  fileIdParamsSchema,
  filesSearchQuerySchema,
  uploadedFileIndexRequestSchema,
  type FileIdParams,
  type FilesSearchQuery,
  type UploadedFileIndexBody,
} from './files-schema.js'

export { fileOpenApiComponents, fileOpenApiOverrides } from './files-schema.js'

const UPLOAD_DIR_NAME = 'uploads'
const MAX_FILE_SIZE = 20 * 1024 * 1024 // 20MB
export const ORPHAN_UPLOAD_SWEEP_INTERVAL_MS = 6 * 60 * 60 * 1000
const SUPPORTED_EXT = new Set(getSupportedExtensions())
const TEXT_EXT = new Set(getSupportedTextExtensions())
const MEDIA_UPLOAD_EXT = new Set([
  '.aac',
  '.aiff',
  '.flac',
  '.m4a',
  '.mkv',
  '.mov',
  '.mp3',
  '.mp4',
  '.oga',
  '.ogg',
  '.opus',
  '.wav',
  '.webm',
])

function getUploadDir(dataDir: string): string {
  return join(dataDir, UPLOAD_DIR_NAME)
}

async function collectReferencedUploadIds(
  sessions: ISessionStore,
): Promise<Set<string> | null> {
  const referencedIds = new Set<string>()
  const seenSessionIds = new Set<string>()
  const perPage = 200
  let expectedTotal: number | null = null

  try {
    for (let page = 1; ; page += 1) {
      const result = await sessions.list({ page, perPage })
      if (expectedTotal === null) expectedTotal = result.totalCount
      if (result.totalCount !== expectedTotal) return null
      if (result.hasNextPage && result.items.length === 0) return null

      for (const session of result.items) {
        if (seenSessionIds.has(session.id)) return null
        seenSessionIds.add(session.id)
        const events = await sessions.getEvents(session.id)
        for (const event of events) {
          if (event.type !== 'user_message') continue
          for (const attachment of event.attachments ?? []) {
            referencedIds.add(attachment.fileId)
          }
        }
      }

      if (!result.hasNextPage) break
    }
  } catch {
    return null
  }

  return seenSessionIds.size === expectedTotal ? referencedIds : null
}

export async function sweepOrphanUploadsForSessions(
  uploadDir: string,
  sessions: ISessionStore,
): Promise<number | null> {
  const referencedFileIds = await collectReferencedUploadIds(sessions)
  if (!referencedFileIds) return null
  return sweepOrphanUploadedFiles(uploadDir, { referencedFileIds })
}

async function runOrphanUploadSweep(
  app: FastifyInstance,
  uploadDir: string,
  sessions: ISessionStore,
): Promise<void> {
  const removed = await sweepOrphanUploadsForSessions(uploadDir, sessions)
  if (removed === null) {
    app.log.warn('skipped orphan upload cleanup because session history was unavailable')
  } else if (removed > 0) {
    app.log.debug({ removed }, 'removed orphaned uploaded files')
  }
}

export async function fileRoutes(app: FastifyInstance) {
  const startupRuntime = app.runtime
  const runtimeUploadDir = startupRuntime ? getUploadDir(getRuntimeDataDir(startupRuntime)) : null
  if (runtimeUploadDir && startupRuntime) {
    const restored = await restoreUploadedFiles(runtimeUploadDir)
    if (restored > 0) {
      app.log.debug({ restored }, 'restored uploaded file metadata')
    }
    const sessions = startupRuntime.sessions
    if (sessions && typeof sessions.list === 'function' && typeof sessions.getEvents === 'function') {
      let inFlight: Promise<void> | null = null
      const triggerSweep = () => {
        if (inFlight) return
        const pending = runOrphanUploadSweep(app, runtimeUploadDir, sessions)
          .catch((error: unknown) => {
            app.log.warn(
              { error: error instanceof Error ? error.message : String(error) },
              'orphan upload cleanup failed',
            )
          })
          .finally(() => {
            if (inFlight === pending) inFlight = null
          })
        inFlight = pending
      }
      // Do not hold daemon readiness behind a potentially large history scan.
      // Registry last-access checks make the background sweep race-safe with
      // requests that begin once route registration completes.
      if (restored > 0) triggerSweep()
      const timer = setInterval(triggerSweep, ORPHAN_UPLOAD_SWEEP_INTERVAL_MS)
      timer.unref()
      app.addHook('onClose', async () => {
        clearInterval(timer)
        await inFlight
      })
    }
  }

  await app.register(multipart, {
    limits: { fileSize: MAX_FILE_SIZE, files: 10 },
  })

  // POST /files/upload — multipart file upload
  app.post('/files/upload', async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const uploadDir = getUploadDir(getRuntimeDataDir(runtime))
    await mkdir(uploadDir, { recursive: true })

    const parts = request.parts()
    const results: UploadedFile[] = []

    for await (const part of parts) {
      if (part.type !== 'file' || !part.filename) continue

      const originalFilename =
        part.filename.replace(/\\/g, '/').split('/').pop()?.trim() || 'upload'
      const ext = extname(originalFilename).toLowerCase()
      const isSupported = SUPPORTED_EXT.has(ext) || TEXT_EXT.has(ext) || MEDIA_UPLOAD_EXT.has(ext)
      if (!isSupported) {
        return reply.status(400).send({
          error: {
            code: 'UNSUPPORTED_FILE_TYPE',
            message: `Unsupported file type: ${ext}. Supported: ${[...SUPPORTED_EXT, ...TEXT_EXT].join(', ')}`,
          },
        })
      }

      const id = randomUUID()
      const safeFilename = `${id}${ext}`
      const filePath = join(uploadDir, safeFilename)

      const buf = await part.toBuffer()
      if (buf.length > MAX_FILE_SIZE) {
        return reply.status(400).send({
          error: {
            code: 'FILE_TOO_LARGE',
            message: `File exceeds ${MAX_FILE_SIZE / 1024 / 1024}MB limit`,
          },
        })
      }

      await writeFile(filePath, buf)

      const uploaded: UploadedFile = {
        id,
        filename: originalFilename,
        mimeType: part.mimetype,
        size: buf.length,
        path: filePath,
        uploadedAt: new Date().toISOString(),
        kind: 'upload',
      }
      try {
        await persistUploadedFileMetadata(uploadDir, uploaded)
        storeUploadedFile(uploaded)
      } catch (error) {
        await unlink(filePath).catch(() => {})
        throw error
      }
      results.push(uploaded)
    }

    if (results.length === 0) {
      return reply.status(400).send({
        error: { code: 'NO_FILES', message: 'No files were uploaded' },
      })
    }

    return {
      data: {
        files: results.map((f) => ({
          id: f.id,
          filename: f.filename,
          mimeType: f.mimeType,
          size: f.size,
          uploadedAt: f.uploadedAt,
        })),
      },
    }
  })

  // GET /files/:id — get file info
  app.get<{ Params: FileIdParams }>(
    '/files/:id',
    {
      schema: fastifySchemaFromZod({ params: fileIdParamsSchema }),
    },
    async (request, reply) => {
      const params = request.params
      const file = getUploadedFile(params.id)
      if (!file) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'File not found' } })
      }
      return {
        data: {
          id: file.id,
          filename: file.filename,
          mimeType: file.mimeType,
          size: file.size,
          uploadedAt: file.uploadedAt,
        },
      }
    },
  )

  // GET /files/:id/download — stream uploaded/generated file bytes
  app.get<{ Params: FileIdParams }>(
    '/files/:id/download',
    {
      schema: fastifySchemaFromZod({ params: fileIdParamsSchema }),
    },
    async (request, reply) => {
      const params = request.params
      const file = getUploadedFile(params.id)
      if (!file) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'File not found' } })
      }

      return reply
        .type(file.mimeType || 'application/octet-stream')
        .header('content-length', file.size)
        .header('content-disposition', `inline; filename="${encodeURIComponent(file.filename)}"`)
        .send(createReadStream(file.path))
    },
  )

  // DELETE /files/:id — delete uploaded file
  app.delete<{ Params: FileIdParams }>(
    '/files/:id',
    {
      schema: fastifySchemaFromZod({ params: fileIdParamsSchema }),
    },
    async (request, reply) => {
      const params = request.params
      const file = getUploadedFile(params.id)
      if (!file) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'File not found' } })
      }
      const removed = await removeUploadedFilesByIds(new Set([params.id]))
      if (removed.length === 0) {
        return reply.status(409).send({
          error: {
            code: 'FILE_DELETE_FAILED',
            message: 'File is currently in use and could not be deleted. Retry shortly.',
          },
        })
      }
      return reply.status(204).send()
    },
  )

  // POST /files/:id/content-part — convert uploaded file to LLM ContentPart
  app.get<{ Params: FileIdParams }>(
    '/files/:id/content-part',
    {
      schema: fastifySchemaFromZod({ params: fileIdParamsSchema }),
    },
    async (request, reply) => {
      const params = request.params
      const file = getUploadedFile(params.id)
      if (!file) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'File not found' } })
      }

      try {
        const part = await fileToContentPart(file.path, file.filename)
        return { data: part }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        if (error instanceof FileExtractionError) {
          return reply.status(400).send({
            error: { code: error.code, message },
          })
        }
        if (message.includes('File too large')) {
          return reply.status(400).send({
            error: { code: 'FILE_TOO_LARGE', message },
          })
        }
        if (message.includes('Unsupported file type')) {
          return reply.status(400).send({
            error: { code: 'UNSUPPORTED_FILE_TYPE', message },
          })
        }
        throw error
      }
    },
  )

  app.post<{ Params: FileIdParams; Body: UploadedFileIndexBody }>(
    '/files/:id/index',
    {
      schema: fastifySchemaFromZod({
        params: fileIdParamsSchema,
        body: uploadedFileIndexRequestSchema.optional(),
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }

      const params = request.params
      const body = request.body
      const file = getUploadedFile(params.id)
      if (!file) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'File not found' } })
      }

      let content: string
      try {
        content = await readExtractableFileContent(file.path, {
          allowOcr: body?.ocr,
          ocrLanguages: body?.ocrLanguages,
          ocrMaxPages: body?.ocrMaxPages,
        })
      } catch (error) {
        return reply.status(400).send({
          error: {
            code: error instanceof FileExtractionError ? error.code : 'FILE_EXTRACTION_FAILED',
            message:
              error instanceof Error
                ? error.message
                : 'Unsupported file type for semantic indexing',
          },
        })
      }

      const document = await runtime.semanticIndex.ingestDocument({
        title: body?.title?.trim() || file.filename,
        content,
        path: body?.path ?? file.filename,
        mimeType: file.mimeType,
        sourceFileId: file.id,
        tags: body?.tags ?? [],
      })

      return { data: document }
    },
  )

  // GET /files/search?q=&cwd=&limit= — workspace-scoped fuzzy file search
  // for chat composer @mention completion. Walks `cwd` (capped at
  // MAX_SEARCH_FILES entries) skipping common heavy directories, returns
  // up to `limit` (default 8, max 32) entries scored by basename
  // prefix → basename substring → path substring.
  app.get<{ Querystring: FilesSearchQuery }>(
    '/files/search',
    {
      preValidation: zodRequestValidation({
        query: {
          schema: filesSearchQuerySchema,
          message: 'Invalid /files/search query',
        },
      }),
    },
    async (request, reply) => {
      // After zod validation: cwd is required string, q is optional,
      // limit is optional coerced number. Route-boundaries lint forbids
      // direct field access on the request query object, so destructure
      // into a typed local first and pull the validated values from
      // there.
      const runtime = app.runtime
      const query = request.query
      const q = (query.q ?? '').trim()
      const cwd = query.cwd.trim()
      if (!isAbsolute(cwd)) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_REQUEST',
            message: 'cwd must be an absolute path',
          },
        })
      }
      // Contain the search to configured workspace roots (default: the daemon
      // user's home directory) so an arbitrary cwd cannot enumerate the whole
      // filesystem layout.
      let canonicalCwd: string
      let searchRoots: string[]
      try {
        canonicalCwd = await realpath(cwd)
        searchRoots = (
          await Promise.all(
            allowedSearchRoots(runtime?.config?.files?.searchRoots).map(async (root) => {
              try {
                return await realpath(root)
              } catch {
                return null
              }
            }),
          )
        ).filter((root): root is string => root !== null)
      } catch (error) {
        request.log.warn({ err: error }, 'files.search cwd unreadable')
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: 'cwd not found or not readable',
          },
        })
      }
      if (!isContainedInRoots(canonicalCwd, searchRoots)) {
        return reply.status(403).send({
          error: {
            code: 'FORBIDDEN',
            message: 'cwd is outside the allowed workspace roots',
          },
        })
      }
      const limit = Math.min(Math.max(query.limit ?? 8, 1), 32)

      let entries: Array<{ path: string; basename: string; isDirectory: boolean }>
      try {
        entries = await scanWorkspace(canonicalCwd)
      } catch (error) {
        // Do not leak the raw fs error (path/errno). Log details server-side.
        request.log.warn({ err: error }, 'files.search cwd unreadable')
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: 'cwd not found or not readable',
          },
        })
      }

      const matches = scoreMatches(entries, q).slice(0, limit)
      return { data: { matches } }
    },
  )
}

const SEARCH_SKIP_DIRS = new Set([
  'node_modules',
  '.git',
  '.next',
  '.turbo',
  'dist',
  'out',
  'build',
  '.cache',
  '.venv',
  'venv',
  '__pycache__',
])
const MAX_SEARCH_FILES = 2_000

function allowedSearchRoots(configured: string[] | undefined): string[] {
  const roots = (configured ?? [])
    .map((root) => root.trim())
    .filter((root) => root && isAbsolute(root))
    .map((root) => resolve(root))
  if (roots.length > 0) {
    return roots
  }
  const home = homedir()
  return home ? [resolve(home)] : []
}

function isContainedInRoots(cwd: string, roots: string[]): boolean {
  if (roots.length === 0) {
    return false
  }
  const target = resolve(cwd)
  return roots.some((root) => {
    const relativePath = relative(resolve(root), target)
    return relativePath === '' || (
      relativePath !== '..'
      && !relativePath.startsWith(`..${sep}`)
      && !isAbsolute(relativePath)
    )
  })
}

async function scanWorkspace(
  cwd: string,
): Promise<Array<{ path: string; basename: string; isDirectory: boolean }>> {
  const out: Array<{ path: string; basename: string; isDirectory: boolean }> = []
  const root = cwd
  // Confirm cwd exists and is a dir before walking — surfaces 404 in the
  // route instead of an opaque ENOENT.
  await stat(cwd)
  const queue: string[] = [cwd]
  while (queue.length > 0 && out.length < MAX_SEARCH_FILES) {
    const dir = queue.shift()!
    let names: string[]
    try {
      names = await readdir(dir)
    } catch {
      continue
    }
    for (const name of names) {
      if (out.length >= MAX_SEARCH_FILES) break
      if (SEARCH_SKIP_DIRS.has(name)) continue
      if (name.startsWith('.') && name !== '.env.example') continue
      const abs = join(dir, name)
      let info
      try {
        info = await lstat(abs)
      } catch {
        continue
      }
      // Workspace search is a discovery surface, so following a directory
      // symlink/junction can disclose names from outside the selected root.
      // File access tools perform their own canonical boundary checks; the
      // mention picker simply omits links entirely and never traverses them.
      if (info.isSymbolicLink()) continue
      const rel = relative(root, abs).split(sep).join('/')
      out.push({
        path: rel,
        basename: name,
        isDirectory: info.isDirectory(),
      })
      if (info.isDirectory()) queue.push(abs)
    }
  }
  return out
}

function scoreMatches(
  entries: Array<{ path: string; basename: string; isDirectory: boolean }>,
  query: string,
): Array<{ path: string; basename: string; isDirectory: boolean }> {
  if (!query) {
    // No query: surface a stable, alphabetised slice (most common
    // first-letter shortcuts work).
    return entries.slice().sort((a, b) => a.path.localeCompare(b.path))
  }
  const q = query.toLowerCase()
  const scored = entries
    .map((entry) => {
      const baseLower = entry.basename.toLowerCase()
      const pathLower = entry.path.toLowerCase()
      let score = 0
      if (baseLower === q) score = 100
      else if (baseLower.startsWith(q)) score = 90
      else if (baseLower.includes(q)) score = 70
      else if (pathLower.includes(q)) score = 40
      return { entry, score }
    })
    .filter((item) => item.score > 0)
    .sort(
      (a, b) =>
        b.score - a.score ||
        a.entry.basename.length - b.entry.basename.length ||
        a.entry.path.localeCompare(b.entry.path),
    )
    .map((item) => item.entry)
  return scored
}
