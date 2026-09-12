import React from 'react'
import { extractA2UI, type A2UIComponent } from '@sepilotd/core'

export interface ContextItemLike {
  id: string
  citationLabel: string
  kind: string
  title: string
  snippet: string
}

export interface AssistantRichContentProps {
  content: string
  MarkdownRenderer: React.ComponentType<{ content: string }>
  CanvasRenderer: React.ComponentType<{ components: A2UIComponent[] }>
  canvasContainerStyle?: React.CSSProperties
}

export interface ContextMessageCardClassNames {
  root: string
  head: string
  copy: string
  kicker: string
  list: string
  item: string
  itemHead: string
}

export interface AssistantCitationsClassNames {
  root: string
  kicker: string
  list: string
  item: string
  itemHead: string
}

export interface ContextMessageCardProps<T extends ContextItemLike> {
  messageId: string
  items: readonly T[]
  classNames: ContextMessageCardClassNames
  kicker?: string
  documentKindLabel?: string
}

export interface AssistantCitationsProps<T extends ContextItemLike> {
  messageId: string
  items: readonly T[]
  classNames: AssistantCitationsClassNames
  kicker?: string
  documentKindLabel?: string
}

export function AssistantRichContent({
  content,
  MarkdownRenderer,
  CanvasRenderer,
  canvasContainerStyle,
}: AssistantRichContentProps) {
  const { text, canvas } = extractA2UI(content)

  return (
    <>
      {text ? <MarkdownRenderer content={text} /> : null}
      {canvas?.components.length ? (
        <div style={{ marginTop: text ? '12px' : 0, ...canvasContainerStyle }}>
          <CanvasRenderer components={canvas.components} />
        </div>
      ) : null}
    </>
  )
}

export function ContextMessageCard<T extends ContextItemLike>({
  messageId,
  items,
  classNames,
  kicker = 'Retrieved context',
  documentKindLabel = 'Document',
}: ContextMessageCardProps<T>) {
  return (
    <div className={classNames.root}>
      <div className={classNames.head}>
        <div className={classNames.copy}>
          <span className={classNames.kicker}>{kicker}</span>
          <strong>{items.length} source{items.length === 1 ? '' : 's'}</strong>
        </div>
      </div>
      <div className={classNames.list}>
        {items.map((item) => (
          <div key={`${messageId}:${item.id}`} className={classNames.item}>
            <div className={classNames.itemHead}>
              <strong>{item.citationLabel}</strong>
              <span>{item.kind === 'document' ? documentKindLabel : item.title}</span>
            </div>
            <p>{item.snippet}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export function AssistantCitations<T extends ContextItemLike>({
  messageId,
  items,
  classNames,
  kicker = 'Sources used',
  documentKindLabel = 'Document',
}: AssistantCitationsProps<T>) {
  if (items.length === 0) {
    return null
  }

  return (
    <div className={classNames.root}>
      <span className={classNames.kicker}>{kicker}</span>
      <div className={classNames.list}>
        {items.map((item) => (
          <div key={`${messageId}:${item.id}`} className={classNames.item}>
            <div className={classNames.itemHead}>
              <strong>{item.citationLabel}</strong>
              <span>{item.kind === 'document' ? documentKindLabel : item.title}</span>
            </div>
            <p>{item.snippet}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
