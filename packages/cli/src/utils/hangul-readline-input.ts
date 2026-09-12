import { Transform } from 'node:stream'
import { StringDecoder } from 'node:string_decoder'
import {
  applyHangulInputChange,
  createHangulInputState,
  type HangulInputState,
} from '../tui/utils/hangul.js'
import {
  countGraphemes,
  previousGraphemeOffset,
} from '../tui/utils/graphemes.js'

interface TtyReadable {
  isTTY?: boolean
  pipe<T extends NodeJS.WritableStream>(destination: T): T
  unpipe(destination?: unknown): void
  setRawMode?: (enabled: boolean) => unknown
  ref?: () => unknown
  unref?: () => unknown
}

const BRACKETED_PASTE_START = '\x1b[200~'
const BRACKETED_PASTE_END = '\x1b[201~'

function charLength(value: string): number {
  return countGraphemes(value)
}

function isCompleteEscapeSequence(sequence: string): boolean {
  if (sequence === '\x1b') return false

  if (sequence.startsWith('\x1b[')) {
    if (sequence.length < 3) return false
    const finalByte = sequence.charCodeAt(sequence.length - 1)
    return finalByte >= 0x40 && finalByte <= 0x7E
  }

  if (sequence.startsWith('\x1b]')) {
    return sequence.endsWith('\x07') || sequence.endsWith('\x1b\\')
  }

  return sequence.length >= 2
}

export function applyReadlineControlSequence(line: string, sequence: string): string {
  let nextLine = line
  for (const ch of sequence) {
    if (ch === '\x7f' || ch === '\b') {
      nextLine = nextLine.slice(0, previousGraphemeOffset(nextLine, nextLine.length))
    } else if (ch === '\r' || ch === '\n' || ch === '\x03' || ch === '\x04') {
      nextLine = ''
    }
  }
  return nextLine
}

export class HangulReadlineInputNormalizer {
  private readonly decoder = new StringDecoder('utf8')
  private line = ''
  private state: HangulInputState = createHangulInputState()
  private passthroughUntilLineEnd = false
  private pendingEscape = ''

  write(chunk: Buffer | string): string {
    const text = typeof chunk === 'string'
      ? chunk
      : this.decoder.write(chunk)
    return this.normalizeText(text)
  }

  end(): string {
    const output = this.normalizeText(this.decoder.end())
    if (!this.pendingEscape) return output

    const pendingEscape = this.pendingEscape
    this.pendingEscape = ''
    return `${output}${pendingEscape}`
  }

  private resetComposition(): void {
    this.state = createHangulInputState()
  }

  private resetLine(): void {
    this.line = ''
    this.passthroughUntilLineEnd = false
    this.pendingEscape = ''
    this.resetComposition()
  }

  private consumeEscapeSequence(sequence: string): string {
    if (sequence === BRACKETED_PASTE_START || sequence === BRACKETED_PASTE_END) {
      return ''
    }

    this.passthroughUntilLineEnd = true
    this.resetComposition()
    return sequence
  }

  private normalizePrintableRun(text: string): string {
    if (!text) return ''

    const previousLine = this.line
    const applied = applyHangulInputChange(
      previousLine,
      `${previousLine}${text}`,
      this.state,
    )
    this.line = applied.value
    this.state = applied.state

    if (this.line === `${previousLine}${text}`) {
      return text
    }
    return `${'\x7f'.repeat(charLength(previousLine))}${this.line}`
  }

  private normalizeText(text: string): string {
    let output = ''
    let printableRun = ''
    const flushPrintableRun = (): void => {
      output += this.normalizePrintableRun(printableRun)
      printableRun = ''
    }

    for (const ch of text) {
      if (this.pendingEscape) {
        this.pendingEscape += ch
        if (isCompleteEscapeSequence(this.pendingEscape)) {
          flushPrintableRun()
          output += this.consumeEscapeSequence(this.pendingEscape)
          this.pendingEscape = ''
        }
        continue
      }

      if (ch === '\x1b') {
        flushPrintableRun()
        this.pendingEscape = ch
        continue
      }

      if (this.passthroughUntilLineEnd) {
        flushPrintableRun()
        output += ch
        this.line = applyReadlineControlSequence(this.line, ch)
        if (ch === '\r' || ch === '\n') {
          this.resetLine()
        }
        continue
      }

      if (ch === '\r' || ch === '\n' || ch === '\x03' || ch === '\x04') {
        flushPrintableRun()
        output += ch
        this.resetLine()
        continue
      }

      if (ch === '\x7f' || ch === '\b') {
        flushPrintableRun()
        output += ch
        this.line = applyReadlineControlSequence(this.line, ch)
        this.resetComposition()
        continue
      }

      if (ch < ' ' || ch === '\x7f') {
        flushPrintableRun()
        output += ch
        this.resetComposition()
        continue
      }

      printableRun += ch
    }

    flushPrintableRun()
    return output
  }
}

export function createHangulReadlineInput(input: TtyReadable): NodeJS.ReadableStream {
  const normalizer = new HangulReadlineInputNormalizer()
  const stream = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      const normalized = normalizer.write(chunk)
      if (normalized) {
        this.push(normalized)
      }
      callback()
    },
    flush(callback) {
      const rest = normalizer.end()
      if (rest) {
        this.push(rest)
      }
      callback()
    },
  }) as Transform & {
    isTTY?: boolean
    setRawMode?: (enabled: boolean) => unknown
    ref?: () => unknown
    unref?: () => unknown
  }

  stream.isTTY = input.isTTY
  stream.setRawMode = (enabled: boolean) => input.setRawMode?.(enabled)
  stream.ref = () => input.ref?.()
  stream.unref = () => input.unref?.()
  input.pipe(stream)
  stream.once('close', () => {
    input.unpipe(stream)
  })

  return stream
}
