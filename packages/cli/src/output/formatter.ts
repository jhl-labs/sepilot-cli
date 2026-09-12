export type OutputFormat = 'text' | 'json' | 'stream-json'

let format: OutputFormat = 'text'
/**
 * Send preformatted output through the stream at every size. The compiled
 * console path can truncate even modest payloads when stdout is a pipe.
 */
export function writeOutputText(value: string): void {
  process.stdout.write(value.endsWith('\n') ? value : `${value}\n`)
}

export function setOutputFormat(f: OutputFormat): void {
  format = f
}

export function getOutputFormat(): OutputFormat {
  return format
}

export function output<T>(data: T, textFormatter?: (data: T) => string): void {
  if (format === 'json') {
    writeJson(data)
  } else if (textFormatter) {
    console.log(textFormatter(data))
  } else {
    console.log(data)
  }
}

/**
 * Like `output` but writes the human (text) variant to stderr so a
 * caller can still pipe stdout for normal output. Under `--json`,
 * the envelope still goes to stdout — scripts pipe stdout for jq.
 */
export function outputError<T>(data: T, textFormatter: (data: T) => string): void {
  if (format === 'json') {
    writeJson(data)
  } else {
    console.error(textFormatter(data))
  }
}

function writeJson(data: unknown): void {
  writeOutputText(JSON.stringify(data, null, 2))
}
