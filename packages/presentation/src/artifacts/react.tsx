import { useEffect, useMemo, useState } from 'react'
import { isSafeA2UIImageSrc } from '@sepilotd/core'
import {
  artifactLabel,
  artifactPreview,
  type ArtifactLike,
} from './shared.js'

export type ArtifactGalleryItem = ArtifactLike

export interface ArtifactGalleryProps {
  artifacts: ArtifactGalleryItem[]
  className?: string
  emptyState?: string | null
}

const STACK_STYLE = {
  display: 'grid',
  gap: '12px',
} as const

const CARD_STYLE = {
  border: '1px solid color-mix(in srgb, currentColor 12%, transparent)',
  borderRadius: '12px',
  background: 'color-mix(in srgb, Canvas 96%, currentColor 4%)',
} as const

const MUTED_STYLE = {
  color: 'color-mix(in srgb, currentColor 62%, transparent)',
} as const

export function ArtifactGallery({
  artifacts,
  className,
  emptyState = null,
}: ArtifactGalleryProps) {
  const orderedArtifacts = useMemo(
    () => artifacts.slice().reverse(),
    [artifacts],
  )
  const [selectedArtifactId, setSelectedArtifactId] = useState<string>()

  useEffect(() => {
    const nextSelectedId = orderedArtifacts[0]?.id
    if (!nextSelectedId) {
      setSelectedArtifactId(undefined)
      return
    }

    setSelectedArtifactId((current) =>
      current && orderedArtifacts.some((artifact) => artifact.id === current)
        ? current
        : nextSelectedId,
    )
  }, [orderedArtifacts])

  const selectedArtifact =
    orderedArtifacts.find((artifact) => artifact.id === selectedArtifactId)
    ?? orderedArtifacts[0]

  if (orderedArtifacts.length === 0) {
    return emptyState ? <div className={className}>{emptyState}</div> : null
  }

  return (
    <div className={className} style={STACK_STYLE}>
      <div style={STACK_STYLE}>
        {orderedArtifacts.map((artifact) => {
          const selected = artifact.id === selectedArtifact?.id
          return (
            <button
              key={artifact.id}
              type="button"
              onClick={() => setSelectedArtifactId(artifact.id)}
              style={{
                ...CARD_STYLE,
                display: 'grid',
                gap: '6px',
                padding: '10px 12px',
                textAlign: 'left',
                cursor: 'pointer',
                background: selected
                  ? 'color-mix(in srgb, #2563eb 10%, Canvas 90%)'
                  : CARD_STYLE.background,
                borderColor: selected
                  ? 'color-mix(in srgb, #2563eb 42%, transparent)'
                  : CARD_STYLE.border,
              }}
            >
              <div style={{ display: 'flex', gap: '8px', justifyContent: 'space-between', alignItems: 'start' }}>
                <strong style={{ minWidth: 0 }}>{artifactLabel(artifact)}</strong>
                <ArtifactBadge artifact={artifact} />
              </div>
              <div style={{ ...MUTED_STYLE, fontSize: '0.82rem', lineHeight: 1.5 }}>
                {artifactPreview(artifact)}
              </div>
            </button>
          )
        })}
      </div>

      {selectedArtifact ? (
        <div style={{ ...CARD_STYLE, padding: '12px' }}>
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'space-between', alignItems: 'start', marginBottom: '10px' }}>
            <div style={{ minWidth: 0 }}>
              <strong>{artifactLabel(selectedArtifact)}</strong>
              <div style={{ ...MUTED_STYLE, fontSize: '0.82rem', marginTop: '4px' }}>
                {selectedArtifact.id}
              </div>
            </div>
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              <ArtifactBadge artifact={selectedArtifact} />
              {selectedArtifact.language ? (
                <span style={badgeStyle('#0f766e', '#ccfbf1')}>{selectedArtifact.language}</span>
              ) : null}
            </div>
          </div>

          <ArtifactBody artifact={selectedArtifact} />
        </div>
      ) : null}
    </div>
  )
}

function ArtifactBadge({ artifact }: { artifact: ArtifactGalleryItem }) {
  return (
    <span style={badgeStyle('#2563eb', '#dbeafe')}>
      {artifact.type}
    </span>
  )
}

function ArtifactBody({ artifact }: { artifact: ArtifactGalleryItem }) {
  switch (artifact.type) {
    case 'html':
      return (
        <iframe
          title={artifactLabel(artifact)}
          sandbox=""
          srcDoc={wrapHtmlPreview(artifact.content)}
          style={{
            width: '100%',
            minHeight: '280px',
            border: '1px solid color-mix(in srgb, currentColor 10%, transparent)',
            borderRadius: '10px',
            background: 'white',
          }}
        />
      )
    case 'svg':
      return (
        <iframe
          title={artifactLabel(artifact)}
          sandbox=""
          srcDoc={wrapSvgPreview(artifact.content)}
          style={{
            width: '100%',
            minHeight: '280px',
            border: '1px solid color-mix(in srgb, currentColor 10%, transparent)',
            borderRadius: '10px',
            background: 'white',
          }}
        />
      )
    case 'image':
      // Gate the agent-controlled image source with the same safe-src check the
      // canvas renderer uses, so a raw <img src> to an arbitrary URL cannot fire
      // a beacon/exfil GET the moment the artifact is viewed.
      if (!isSafeA2UIImageSrc(artifact.content)) {
        return (
          <div
            role="note"
            style={{
              padding: '16px',
              border: '1px solid color-mix(in srgb, currentColor 12%, transparent)',
              borderRadius: '10px',
              opacity: 0.7,
            }}
          >
            Image source blocked
          </div>
        )
      }
      return (
        <img
          src={artifact.content}
          alt={artifactLabel(artifact)}
          style={{
            display: 'block',
            maxWidth: '100%',
            maxHeight: '520px',
            borderRadius: '10px',
            objectFit: 'contain',
          }}
        />
      )
    case 'document':
      return (
        <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
          {artifact.content}
        </div>
      )
    case 'mermaid':
      return (
        <pre style={preStyle}>
          <code>{artifact.content}</code>
        </pre>
      )
    case 'code':
    default:
      return (
        <pre style={preStyle}>
          <code>{artifact.content}</code>
        </pre>
      )
  }
}

function badgeStyle(color: string, background: string) {
  return {
    padding: '3px 8px',
    borderRadius: '999px',
    background,
    color,
    fontSize: '0.72rem',
    fontWeight: 600,
    textTransform: 'uppercase' as const,
  }
}

const preStyle = {
  margin: 0,
  overflowX: 'auto' as const,
  whiteSpace: 'pre-wrap' as const,
  lineHeight: 1.55,
  fontSize: '0.86rem',
  padding: '12px',
  borderRadius: '10px',
  background: 'color-mix(in srgb, Canvas 94%, currentColor 6%)',
  border: '1px solid color-mix(in srgb, currentColor 10%, transparent)',
}

function wrapHtmlPreview(content: string): string {
  if (/<html[\s>]/i.test(content)) {
    return content
  }

  return `<!DOCTYPE html>
<html>
  <body style="margin:0;padding:16px;font-family:ui-sans-serif,system-ui,sans-serif;">
    ${content}
  </body>
</html>`
}

function wrapSvgPreview(content: string): string {
  return `<!DOCTYPE html>
<html>
  <body style="margin:0;padding:16px;display:grid;place-items:center;background:white;">
    ${content}
  </body>
</html>`
}
