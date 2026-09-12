export type A2UIComponent =
  | A2UIText
  | A2UITable
  | A2UIChart
  | A2UIForm
  | A2UICode
  | A2UIImage
  | A2UIList
  | A2UIProgress

export interface A2UIText {
  type: 'text'
  content: string
  format?: 'plain' | 'markdown' | 'html'
}

export interface A2UITable {
  type: 'table'
  title?: string
  headers: string[]
  rows: string[][]
}

export interface A2UIChart {
  type: 'chart'
  title?: string
  chartType: 'bar' | 'line' | 'pie' | 'scatter'
  data: { labels: string[]; datasets: Array<{ label: string; values: number[] }> }
}

export interface A2UIForm {
  type: 'form'
  title?: string
  fields: Array<{
    name: string
    label: string
    type: 'text' | 'number' | 'select' | 'checkbox' | 'textarea'
    options?: string[]
    required?: boolean
    defaultValue?: string
  }>
  submitLabel?: string
}

export interface A2UICode {
  type: 'code'
  language: string
  code: string
  title?: string
}

export interface A2UIImage {
  type: 'image'
  src: string
  alt?: string
  width?: number
  height?: number
  mime?: string
}

export interface A2UIList {
  type: 'list'
  title?: string
  items: Array<{ text: string; description?: string; icon?: string }>
  ordered?: boolean
}

export interface A2UIProgress {
  type: 'progress'
  title?: string
  steps: Array<{ label: string; status: 'done' | 'active' | 'pending' }>
}

export interface A2UIPayload {
  kind?: 'a2ui'
  version?: 1
  components: A2UIComponent[]
  interactive?: boolean
  metadata?: Record<string, string | number | boolean | null>
}

export function isSafeA2UIImageSrc(src: string): boolean {
  const value = src.trim()
  if (!value) return false
  if (/^https?:\/\//i.test(value)) return true
  if (/^blob:/i.test(value)) return true
  if (/^data:image\/(?:png|jpe?g|gif|webp|avif);base64,/i.test(value)) return true
  if (value.startsWith('/') && !value.startsWith('//')) return true
  return false
}
