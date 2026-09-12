import { request } from 'node:https'
import { rootCertificates } from 'node:tls'
import { createHash } from 'node:crypto'
import type { IDocumentMemoryStore } from '@sepilotd/core'
import type { RagFolder, RagStore } from './store.js'

export function webSourceUrl(raw: string): URL {
  const url = new URL(raw)
  if (url.protocol !== 'https:' || url.username || url.password || url.hash)
    throw new Error('웹 자료는 사용자 정보와 fragment가 없는 HTTPS URL을 입력하세요.')
  return url
}
/** Explicit first-party source registration may target intranet hosts; no redirects or ambient credentials. */
export async function readWebSource(
  folder: Pick<RagFolder, 'path' | 'tlsVerify' | 'caCert'>,
): Promise<string> {
  const url = webSourceUrl(folder.path ?? '')
  return new Promise((resolve, reject) => {
    const req = request(
      url,
      {
        method: 'GET',
        rejectUnauthorized: folder.tlsVerify !== false,
        ...(folder.caCert ? { ca: [...rootCertificates, folder.caCert] } : {}),
        headers: {
          accept: 'text/html, text/plain, text/markdown, application/json',
          'accept-encoding': 'identity',
          'user-agent': 'sepilotd-docs/1.0',
        },
      },
      (res) => {
        if ((res.statusCode ?? 0) < 200 || (res.statusCode ?? 0) >= 300) {
          res.destroy()
          reject(
            new Error(
              `웹 자료 응답 ${res.statusCode}: 리디렉션은 최종 HTTPS URL로 직접 등록하세요.`,
            ),
          )
          return
        }
        const type = (res.headers['content-type'] ?? '').split(';')[0]!.trim().toLowerCase()
        if (
          ![
            'text/html',
            'text/plain',
            'text/markdown',
            'application/json',
            'application/xml',
            'text/xml',
          ].includes(type)
        ) {
          res.destroy()
          reject(new Error(`지원하지 않는 웹 자료 형식: ${type || 'Content-Type 없음'}`))
          return
        }
        const chunks: Buffer[] = []
        let size = 0
        res.on('data', (chunk: Buffer) => {
          size += chunk.length
          if (size > 1_000_000) {
            req.destroy(new Error('웹 자료는 1MB 이하만 연결할 수 있습니다.'))
            return
          }
          chunks.push(chunk)
        })
        res.on('error', reject)
        res.on('aborted', () => reject(new Error('웹 자료 응답이 중단되었습니다.')))
        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf8')
          const body =
            type === 'text/html'
              ? raw
                  .replace(/<(script|style|noscript)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, '')
                  .replace(/<[^>]*>/g, ' ')
                  .replace(/&nbsp;/g, ' ')
                  .replace(/&amp;/g, '&')
                  .replace(/&lt;/g, '<')
                  .replace(/&gt;/g, '>')
                  .replace(/[ \t]+/g, ' ')
                  .trim()
              : raw.trim()
          if (!body || body.includes('\0'))
            reject(new Error('읽을 수 있는 웹 자료 본문이 없습니다.'))
          else resolve(body)
        })
      },
    )
    const timeout = setTimeout(
      () => req.destroy(new Error('웹 자료 연결 시간이 초과되었습니다.')),
      20_000,
    )
    req.on('close', () => clearTimeout(timeout))
    req.on('error', reject)
    req.end()
  })
}
export async function syncWebFolder(
  store: RagStore,
  folder: RagFolder,
  semanticIndex?: IDocumentMemoryStore,
) {
  const body = await readWebSource(folder)
  const id = createHash('sha256').update(`web:${folder.id}`).digest('hex')
  const sourceFileId = `web:${folder.id}`
  await store.upsertDocument({
    id,
    folderId: folder.id,
    title: folder.name,
    body,
    path: folder.path,
    sourceFileId,
    size: Buffer.byteLength(body),
  })
  await semanticIndex?.ingestDocument({
    id,
    title: folder.name,
    content: body,
    path: folder.path,
    sourceFileId,
    tags: ['rag', `rag-folder:${folder.id}`, 'source:web'],
  })
  return { indexed: 1, deleted: 0, skipped: 0 }
}
