export interface Embedder {
  readonly id: string
  readonly dimension: number
  embed(texts: string[]): Promise<number[][]>
}
