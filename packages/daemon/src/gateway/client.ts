import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { Comment, CommentInput, JobHandle, JobInput } from '@sepilotd/core'
import { GatewayClient as SharedGatewayClient } from '@sepilotd/api-client'
import { isNodeFsError } from '../utils/fs-error.js'

/**
 * Load the gateway bearer token the daemon presents on every gateway
 * call. Returns null when the file does not exist (the gateway is
 * optional). Other read errors throw so a permissions flap doesn't
 * silently demote the daemon to anonymous gateway requests.
 */
export async function loadGatewayToken(dataDir: string): Promise<string | null> {
  const tokenPath =
    process.env.GATEWAY_TOKEN_PATH ?? join(dataDir, 'security', 'gateway.token')
  try {
    return (await readFile(tokenPath, 'utf-8')).trim()
  } catch (err) {
    if (isNodeFsError(err, 'ENOENT')) {
      return null
    }
    throw new Error(
      `Failed to read gateway auth token at ${tokenPath}: ${
        err instanceof Error ? err.message : String(err)
      }`,
      { cause: err },
    )
  }
}

export class GatewayClient extends SharedGatewayClient {
  constructor(baseUrl: string, token?: string) {
    super({ baseUrl, token: token ?? null })
  }

  override addComment(ticketId: string, comment: CommentInput): Promise<Comment>
  addComment(ticketId: string, body: string, type?: string): Promise<Comment>
  override async addComment(
    ticketId: string,
    commentOrBody: CommentInput | string,
    type?: string,
  ): Promise<Comment> {
    if (ticketId === 'delegation') {
      const comment = typeof commentOrBody === 'string'
        ? {
            body: commentOrBody,
            type:
              (type as 'progress' | 'result' | 'error' | 'general' | undefined) ??
              'general',
          }
        : commentOrBody
      return super.addDelegationComment(comment)
    }

    if (typeof commentOrBody === 'string') {
      return super.addComment(ticketId, {
        body: commentOrBody,
        type:
          (type as 'progress' | 'result' | 'error' | 'general' | undefined) ??
          'general',
      })
    }
    return super.addComment(ticketId, commentOrBody)
  }

  override async getComments(ticketId: string): Promise<Comment[]> {
    if (ticketId === 'delegation') {
      return super.getDelegationComments()
    }
    return super.getComments(ticketId)
  }

  override dispatchJob(input: JobInput): Promise<JobHandle>
  dispatchJob(
    workflow: string,
    inputs: Record<string, string>,
  ): Promise<JobHandle>
  override async dispatchJob(
    inputOrWorkflow: JobInput | string,
    inputs?: Record<string, string>,
  ): Promise<JobHandle> {
    if (typeof inputOrWorkflow === 'string') {
      return super.dispatchJob({
        workflow: inputOrWorkflow,
        inputs: inputs ?? {},
      })
    }
    return super.dispatchJob(inputOrWorkflow)
  }
}
