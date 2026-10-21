import { createElement, type ReactNode } from 'react'

export interface FrameViewerProps {
  open: boolean
  title: string
  subtitle?: string
  src: string | undefined
  live?: boolean
  onClose: () => void
  actions?: ReactNode
  classNames?: { root?: string; header?: string; body?: string; image?: string; closeButton?: string }
  labels?: { close?: string; waiting?: string }
}

export function FrameViewer(props: FrameViewerProps) {
  if (!props.open) return null
  const labels = { close: 'Close', waiting: 'Waiting for the first frame…', ...props.labels }
  return createElement(
    'div',
    { role: 'dialog', 'aria-modal': true, 'aria-label': props.title, 'data-live': props.live ? 'true' : 'false', className: props.classNames?.root },
    createElement(
      'header',
      { className: props.classNames?.header },
      createElement('strong', null, props.title),
      props.subtitle ? createElement('span', null, props.subtitle) : null,
      props.actions ?? null,
      createElement('button', { type: 'button', 'aria-label': labels.close, onClick: props.onClose, className: props.classNames?.closeButton }, '×'),
    ),
    createElement(
      'div',
      { className: props.classNames?.body },
      props.src
        ? createElement('img', { src: props.src, alt: props.title, className: props.classNames?.image })
        : createElement('p', null, labels.waiting),
    ),
  )
}
