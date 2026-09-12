import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import {
  completeOAuth,
  connectWithToken,
  disconnect,
  getStatus,
  startOAuth,
} from '../../github/oauth.js'

const StartBody = z.object({
  clientId: z.string().min(1).optional(),
  redirectUri: z.string().url().optional(),
  scopes: z.array(z.string()).optional(),
})
const CallbackQuery = z.object({
  code: z.string().min(1).optional(),
  state: z.string().min(1).optional(),
  error: z.string().optional(),
  error_description: z.string().optional(),
})
const TokenBody = z.object({
  token: z.string().min(1),
})

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function callbackHtml(input: {
  ok: boolean
  title: string
  message: string
}): string {
  const color = input.ok ? '#047857' : '#b91c1c'
  const title = escapeHtml(input.title)
  const message = escapeHtml(input.message)
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${title}</title>
  <style>
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0b1020; color: #e5e7eb; }
    main { min-height: 100vh; display: grid; place-items: center; padding: 24px; }
    section { max-width: 480px; border: 1px solid #25304a; border-radius: 12px; padding: 24px; background: #111827; }
    h1 { margin: 0 0 8px; color: ${color}; font-size: 20px; }
    p { margin: 0; line-height: 1.5; color: #cbd5e1; }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>${title}</h1>
      <p>${message}</p>
    </section>
  </main>
</body>
</html>`
}

export async function registerGitHubOAuthRoutes(
  app: FastifyInstance,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'github',
      version: '1',
      methods: [
        { method: 'POST', path: '/github/oauth/start' },
        { method: 'GET', path: '/github/oauth/callback' },
        { method: 'GET', path: '/github/oauth/status' },
        { method: 'POST', path: '/github/oauth/token' },
        { method: 'DELETE', path: '/github/oauth' },
      ],
    },
    async (a) => {
      a.post('/github/oauth/start', async (req, reply) => {
        const body = StartBody.parse(req.body ?? {})
        const clientId =
          body.clientId ?? process.env.GITHUB_OAUTH_CLIENT_ID ?? ''
        const clientSecret = process.env.GITHUB_OAUTH_CLIENT_SECRET ?? ''
        const redirectUri =
          body.redirectUri ??
          process.env.GITHUB_OAUTH_REDIRECT ??
          'http://127.0.0.1:17600/github/oauth/callback'
        const scopes = body.scopes ?? ['repo', 'read:org', 'gist']
        if (!clientId || !clientSecret) {
          void reply.status(503).send({
            code: 'GITHUB_OAUTH_NOT_CONFIGURED',
            message: 'GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET must be configured, or connect with a personal access token.',
            retriable: false,
          })
          return reply
        }
        return startOAuth({ clientId, redirectUri, scopes })
      })
      a.get('/github/oauth/callback', async (req, reply) => {
        const query = CallbackQuery.parse(req.query ?? {})
        const clientId = process.env.GITHUB_OAUTH_CLIENT_ID ?? ''
        const clientSecret = process.env.GITHUB_OAUTH_CLIENT_SECRET ?? ''
        const redirectUri =
          process.env.GITHUB_OAUTH_REDIRECT ??
          'http://127.0.0.1:17600/github/oauth/callback'
        if (query.error) {
          const message = query.error_description ?? query.error
          reply.type('text/html').send(callbackHtml({
            ok: false,
            title: 'GitHub 연결 실패',
            message,
          }))
          return reply
        }
        if (!query.code || !query.state || !clientId || !clientSecret) {
          reply.type('text/html').status(400).send(callbackHtml({
            ok: false,
            title: 'GitHub 연결 실패',
            message: 'OAuth callback is missing code/state or daemon OAuth configuration.',
          }))
          return reply
        }
        try {
          const status = await completeOAuth({
            code: query.code,
            state: query.state,
            clientId,
            clientSecret,
            redirectUri,
          })
          reply.type('text/html').send(callbackHtml({
            ok: true,
            title: 'GitHub 연결 완료',
            message: `@${status.login ?? 'GitHub'} 계정이 연결되었습니다. 이 창을 닫고 sepilotd로 돌아가세요.`,
          }))
          return reply
        } catch (error) {
          reply.type('text/html').status(400).send(callbackHtml({
            ok: false,
            title: 'GitHub 연결 실패',
            message: error instanceof Error ? error.message : 'GitHub OAuth failed.',
          }))
          return reply
        }
      })
      a.get('/github/oauth/status', async () => getStatus())
      a.post('/github/oauth/token', async (req) => {
        const body = TokenBody.parse(req.body ?? {})
        return connectWithToken(body.token)
      })
      a.delete('/github/oauth', async () => disconnect())
    },
  )
}
