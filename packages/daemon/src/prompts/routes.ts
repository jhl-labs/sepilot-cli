import type { FastifyInstance } from 'fastify'
import { crudRouter } from '../server/crud/crud-router.js'
import { PromptTemplateInput, type PromptTemplate } from './schema.js'
import { createPromptTemplatesRepo } from './repo.js'

export async function registerPromptTemplatesRoutes(
  app: FastifyInstance,
): Promise<void> {
  const repo = createPromptTemplatesRepo()
  await crudRouter<PromptTemplate, PromptTemplateInput>(app, repo, {
    capability: {
      name: 'prompt-templates',
      version: '1',
      basePath: '/prompt-templates',
    },
    schema: PromptTemplateInput,
  })
}
