import { randomUUID } from 'node:crypto'
import { mkdir, readdir, unlink } from 'node:fs/promises'
import { join } from 'node:path'
import type { FastifyInstance, FastifyReply } from 'fastify'
import '../fastify-types.js'
import {
  getRuntimeDataDir,
  zodRequestValidation,
} from './utils.js'
import {
  getProjectsDir,
  loadProject,
  saveProject,
  type Project,
} from './project-store.js'
import {
  InvalidCwdError,
  invalidCwdResponse,
  resolveRequestCwd,
} from './request-cwd.js'
import {
  addProjectSessionBodySchema,
  createProjectBodySchema,
  projectIdParamsSchema,
  updateProjectBodySchema,
  type AddProjectSessionBody,
  type CreateProjectBody,
  type ProjectIdParams,
  type UpdateProjectBody,
} from './projects-schema.js'

export {
  projectOpenApiComponents,
  projectOpenApiOverrides,
} from './projects-schema.js'

async function resolveProjectWorkingDirectory(
  raw: string | undefined,
  reply: FastifyReply,
): Promise<{ ok: true; workingDirectory: string } | { ok: false }> {
  try {
    return {
      ok: true,
      workingDirectory: await resolveRequestCwd(raw) ?? '',
    }
  } catch (error) {
    if (error instanceof InvalidCwdError) {
      reply.status(400).send(invalidCwdResponse(error))
      return { ok: false }
    }
    throw error
  }
}

export async function projectRoutes(app: FastifyInstance) {
  // GET /projects — list all projects
  app.get('/projects', async (_request, _reply) => {
    const runtime = app.runtime
    if (!runtime)
      return { data: [] }

    const dir = getProjectsDir(getRuntimeDataDir(runtime))
    await mkdir(dir, { recursive: true })

    const files = await readdir(dir).catch(() => [])
    const projects: Project[] = []
    for (const f of files) {
      if (!f.endsWith('.json')) continue
      const p = await loadProject(dir, f.replace('.json', ''))
      if (p) projects.push(p)
    }
    projects.sort(
      (a, b) =>
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
    )
    return { data: projects }
  })

  // POST /projects — create project
  app.post<{ Body: CreateProjectBody }>('/projects', {
    preValidation: zodRequestValidation({
      body: {
        schema: createProjectBodySchema,
        message: 'Invalid project create request body',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime)
      return reply
        .status(503)
        .send({
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: 'Runtime not initialized',
          },
        })

    const body = request.body
    const { name, description, instructions } = body
    const resolvedWorkingDirectory = await resolveProjectWorkingDirectory(
      body.workingDirectory,
      reply,
    )
    if (!resolvedWorkingDirectory.ok) return

    const dir = getProjectsDir(getRuntimeDataDir(runtime))
    const now = new Date().toISOString()

    const project: Project = {
      id: randomUUID(),
      name,
      description: description ?? '',
      instructions: instructions ?? '',
      workingDirectory: resolvedWorkingDirectory.workingDirectory,
      sessionIds: [],
      fileIds: [],
      createdAt: now,
      updatedAt: now,
    }
    await saveProject(dir, project)
    return { data: project }
  })

  // GET /projects/:id
  app.get<{ Params: ProjectIdParams }>(
    '/projects/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: projectIdParamsSchema,
          message: 'Invalid project id',
        },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime)
        return reply
          .status(503)
          .send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })

      const params = request.params
      const dir = getProjectsDir(getRuntimeDataDir(runtime))
      const project = await loadProject(dir, params.id)
      if (!project)
        return reply
          .status(404)
          .send({
            error: { code: 'NOT_FOUND', message: 'Project not found' },
          })
      return { data: project }
    },
  )

  // PUT /projects/:id — update project
  app.put<{ Params: ProjectIdParams; Body: UpdateProjectBody }>(
    '/projects/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: projectIdParamsSchema,
          message: 'Invalid project id',
        },
        body: {
          schema: updateProjectBodySchema,
          message: 'Invalid project update request body',
        },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime)
        return reply
          .status(503)
          .send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })

      const params = request.params
      const body = request.body
      const dir = getProjectsDir(getRuntimeDataDir(runtime))
      const project = await loadProject(dir, params.id)
      if (!project)
        return reply
          .status(404)
          .send({
            error: { code: 'NOT_FOUND', message: 'Project not found' },
          })

      if (body.name !== undefined) project.name = body.name
      if (body.description !== undefined)
        project.description = body.description
      if (body.instructions !== undefined)
        project.instructions = body.instructions
      if (body.workingDirectory !== undefined) {
        const resolvedWorkingDirectory = await resolveProjectWorkingDirectory(
          body.workingDirectory,
          reply,
        )
        if (!resolvedWorkingDirectory.ok) return
        project.workingDirectory = resolvedWorkingDirectory.workingDirectory
      }
      if (body.sessionIds !== undefined)
        project.sessionIds = body.sessionIds
      if (body.fileIds !== undefined) project.fileIds = body.fileIds
      project.updatedAt = new Date().toISOString()

      await saveProject(dir, project)
      return { data: project }
    },
  )

  // DELETE /projects/:id
  app.delete<{ Params: ProjectIdParams }>(
    '/projects/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: projectIdParamsSchema,
          message: 'Invalid project id',
        },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime)
        return reply
          .status(503)
          .send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })

      const params = request.params
      const dir = getProjectsDir(getRuntimeDataDir(runtime))
      try {
        await unlink(join(dir, `${params.id}.json`))
      } catch {
        /* not found is ok */
      }
      return reply.status(204).send()
    },
  )

  // POST /projects/:id/sessions — add session to project
  app.post<{ Params: ProjectIdParams; Body: AddProjectSessionBody }>(
    '/projects/:id/sessions',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: projectIdParamsSchema,
          message: 'Invalid project id',
        },
        body: {
          schema: addProjectSessionBodySchema,
          message: 'Invalid project session request body',
        },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime)
        return reply
          .status(503)
          .send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })

      const params = request.params
      const body = request.body
      const dir = getProjectsDir(getRuntimeDataDir(runtime))
      const project = await loadProject(dir, params.id)
      if (!project)
        return reply
          .status(404)
          .send({
            error: { code: 'NOT_FOUND', message: 'Project not found' },
          })

      const { sessionId } = body
      if (sessionId && !project.sessionIds.includes(sessionId)) {
        project.sessionIds.push(sessionId)
        project.updatedAt = new Date().toISOString()
        await saveProject(dir, project)
      }
      return { data: project }
    },
  )
}
