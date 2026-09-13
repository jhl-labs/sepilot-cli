/** Lossless text transport for providers that alter literal whitespace. */
export const contentEncodingSchema = {
  type: 'string',
  enum: ['utf8', 'base64'],
  description: 'Default utf8: literal content. Use base64 of UTF-8 text when exact leading/trailing whitespace must survive transport (a newline is Cg==). Decoded text is shown in approval previews. This does not change permissions.',
}

export function decodeFileContent(input: Record<string, unknown>): string | undefined {
  if (typeof input.content !== 'string') return undefined
  if (input.contentEncoding === undefined || input.contentEncoding === 'utf8') return input.content
  if (input.contentEncoding !== 'base64') return undefined
  const bytes = Buffer.from(input.content, 'base64')
  // Buffer decoding is permissive; reject ignored characters and invalid UTF-8.
  if (bytes.toString('base64') !== input.content) return undefined
  const text = bytes.toString('utf8')
  return Buffer.from(text, 'utf8').equals(bytes) ? text : undefined
}

export function normalizeFileContent(input: Record<string, unknown>): Record<string, unknown> {
  const content = decodeFileContent(input)
  return input.contentEncoding === 'base64' && content !== undefined
    ? { ...input, content, contentEncoding: 'utf8' }
    : input
}
