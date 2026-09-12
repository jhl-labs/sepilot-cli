import type { Embedder } from './types.js'

const DIMENSION = 128

function hashStringToFloats(s: string, dim: number): number[] {
  const out = new Float32Array(dim)
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i)
    const slot = (code * 2654435761) % dim
    out[(slot + dim) % dim] += Math.sin(code)
  }
  let n = 0
  for (let i = 0; i < dim; i++) n += out[i]! * out[i]!
  n = Math.sqrt(n) || 1
  const arr: number[] = new Array(dim)
  for (let i = 0; i < dim; i++) arr[i] = out[i]! / n
  return arr
}

export const hashEmbedder: Embedder = {
  id: 'hash-128',
  dimension: DIMENSION,
  async embed(texts) {
    return texts.map((t) => hashStringToFloats(t, DIMENSION))
  },
}
